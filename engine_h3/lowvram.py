"""VRAM 16GB mode: the DiT's denoise with its activations cut into pieces (台帳 §1-101).

The off path (``H3DiTBlock.forward`` / ``H3DiT.final_rows``) is untouched; a
block whose ``vram16`` attribute holds a :class:`Vram16Plan` comes here
instead (``engine_h3/dit.py`` branches at the top of the two methods). Nothing
here is new maths: every number is produced by the SAME functions the off path
calls (``_modulate``, ``_gate``, ``block.mlp``, ``_attention``, ``apply_rope``,
``_linear_f32``, the norms, and the negative prompt's ``vsf.flip_values``),
only over a range of rows or a group of heads at a time, written into
preallocated outputs. The one exception is how the qkv input reaches the
int8 kernels: it is quantized once per block and the head groups' slices
reuse it (``H3Int8Linear.quantize_input`` / ``forward_rows_q``,
台帳 §1-105 ①; the same comfy-kitchen kernels with the same arguments, see
``engine_h3.quant_linear``), or, when ``shares_input`` says no, once per slice
by ``forward_rows`` as before. Every one of those operations is
per row or per head, so the target is the off path's result bit for bit
(裁定 46; checked by the CPU tests and the GPU gates A-G2 / A-G3b). sdpa is
bit-identical; with sage, 2 of the 50 blocks differ (the SageAttention kernel
depends on the heads per call), accepted by 裁定 53 (VERIFICATION_LOG §169.3).

What is cut, and why it lowers the peak (VERIFICATION_LOG §168):
  * attention: q, k and v are projected per head group straight from the
    rows of the stacked ``[q; k; v]`` weight (``H3Int8Linear.forward_rows``, or
    ``forward_rows_q`` on the input quantized once per block),
    so the fused ``[S, 3 * 7168]`` qkv never exists; the group's output is
    copied into one ``[S, 7168]`` buffer before ``out_proj``;
  * FFN: ``block.mlp`` over equal row chunks (``row_chunks``), so the
    ``[S, 2 * ffn]`` fc1 output exists for one chunk at a time;
  * AdaLN modulation / gates and the final layer: per row chunk, so their
    float32 temporaries are chunk sized.

Split units come from the shape alone (裁定 51; ``engine_h3.constants``).
Called inside ``torch.no_grad()`` (the pipeline's), like the off path.
"""

from __future__ import annotations

import contextlib
import functools
import math
from dataclasses import dataclass

import torch

from engine_h3 import dit as _dit
from engine_h3 import lora as _lora
from engine_h3.constants import (
    VRAM16_BLOCKS_ON_GPU,
    VRAM16_HEAD_GROUP,
    VRAM16_ROW_CHUNK,
)
from engine_h3.vsf import flip_values

@dataclass(frozen=True)
class Vram16Plan:
    """The split units of one job. The attention backend is not here: the block
    reads ``block.attn.attention_backend`` (set per forward by ``H3DiT``)."""

    head_group: int = VRAM16_HEAD_GROUP
    row_chunk: int = VRAM16_ROW_CHUNK


def row_chunks(s: int, chunk: int) -> list[tuple[int, int]]:
    """``[0, s)`` cut into ``ceil(s / chunk)`` EQUAL ranges (sizes differ by at most 1).

    Equal, not ``chunk`` + remainder: the last range is never a sliver, so
    every chunk of a split sequence has at least ``chunk // 2`` rows and stays
    clear of the int8 kernel's small-m paths (m <= 2, and m <= 128 on Turing).
    ``s <= chunk`` is one range, i.e. not split.
    """
    s, chunk = int(s), int(chunk)
    if s <= chunk:
        return [(0, s)]
    n = math.ceil(s / chunk)
    base, extra = divmod(s, n)
    out = []
    r0 = 0
    for i in range(n):
        r1 = r0 + base + (1 if i < extra else 0)
        out.append((r0, r1))
        r0 = r1
    return out


def blocks_on_gpu_for(n_load: int, *, num_blocks: int) -> int:
    """The swap window of a VRAM 16GB mode job (the caller passes ``n_load`` as is when off).

    ``num_blocks``: the DiT's block count (``len(dit.blocks)``).
    ``n_load <= 0`` or ``>= num_blocks`` (the whole DiT on the GPU) is kept;
    otherwise at most ``VRAM16_BLOCKS_ON_GPU`` blocks, whatever the sequence
    length (gate A-G5 found no time difference between 4 and 8).
    """
    n = int(n_load)
    if n <= 0 or n >= int(num_blocks):
        return n
    return min(n, VRAM16_BLOCKS_ON_GPU)


def derive_memory_cap(free: int, total: int, reserved: int, margin: int) -> float:
    """The allocator cap as a fraction of the device's total memory (all in bytes).

    What this process already holds (``reserved``) plus what the device has
    free (``mem_get_info``), less ``margin``. The cap only binds this
    process: memory another process takes during the job is not covered
    (裁定 47).
    """
    fraction = (int(reserved) + int(free) - int(margin)) / int(total)
    return min(1.0, max(0.0, fraction))


def _rows_into(out: torch.Tensor, chunks, fn) -> torch.Tensor:
    """``out[r0:r1] = fn(r0, r1)`` for every chunk; returns ``out``."""
    for r0, r1 in chunks:
        out[r0:r1] = fn(r0, r1)
    return out


def block_forward(
    block,
    x: torch.Tensor,
    t_emb: torch.Tensor,
    adaln_index: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    """``H3DiTBlock.forward`` with the activations cut by ``block.vram16``.

    Same steps, same functions, same order as the off path; each one over a
    row range (AdaLN, FFN) or a head group (attention).
    """
    plan: Vram16Plan = block.vram16
    s = x.shape[0]
    chunks = row_chunks(s, plan.row_chunk)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)

    # attention input: norm1, then the modulation per row chunk
    n1 = block.norm1(x)
    h = _rows_into(torch.empty_like(n1), chunks,
                   lambda r0, r1: _dit._modulate(n1[r0:r1], shift_msa, scale_msa, adaln_index[r0:r1]))
    del n1

    # attention per head group; q, k, v straight from the [q; k; v] weight rows
    attn = block.attn
    heads, d = attn.heads, attn.head_dim
    inner = heads * d
    o = torch.empty((s, inner), dtype=torch.bfloat16, device=x.device)
    qkv = attn.qkv_proj
    xa = _lora.xa_for(qkv, h)   # LoRA's f32(h) @ A^T once per block (before h goes); None without LoRA
    if qkv.shares_input(h):
        # h quantized once for every head group's q, k, v slice (台帳 §1-105 ①)
        proj = functools.partial(qkv.forward_rows_q, *qkv.quantize_input(h), lora_xa=xa)
    else:
        proj = functools.partial(qkv.forward_rows, h, lora_xa=xa)
    del h   # shared: freed here; otherwise the partial keeps it, as before
    for h0 in range(0, heads, plan.head_group):
        h1 = min(h0 + plan.head_group, heads)
        g = h1 - h0
        q = proj(0 * inner + h0 * d, 0 * inner + h1 * d).view(s, g, d)
        k = proj(1 * inner + h0 * d, 1 * inner + h1 * d).view(s, g, d)
        v = proj(2 * inner + h0 * d, 2 * inner + h1 * d).view(s, g, d)
        if attn.vsf is not None:
            flip_values(v, attn.vsf)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        if rope_cos is not None:
            q = _dit.apply_rope(q, rope_cos, rope_sin)
            k = _dit.apply_rope(k, rope_cos, rope_sin)
        if attn.vsf_probe is not None:
            attn.vsf_probe.observe(q, k, head0=h0)
        out = _dit._attention(q, k, v, attn.attention_backend)
        del q, k, v
        o[:, h0 * d:h1 * d].view(s, g, d).copy_(out)
        del out
    del proj, xa
    a = attn.out_proj(o)
    del o

    # attention residual, per row chunk
    x2 = _rows_into(torch.empty_like(x), chunks,
                    lambda r0, r1: _dit._gate(x[r0:r1], gate_msa, a[r0:r1], adaln_index[r0:r1]))
    del a

    # FFN input, then the FFN per row chunk
    n2 = block.norm2(x2)
    h2 = _rows_into(torch.empty_like(n2), chunks,
                    lambda r0, r1: _dit._modulate(n2[r0:r1], shift_mlp, scale_mlp, adaln_index[r0:r1]))
    del n2
    # fc1 / fc2 run once per chunk: an fp8 layer on the bf16 path keeps its dequantized
    # weight for this loop only (H3QuantLinear.dequant_cache; a no-op for the others)
    with _dequant_cache(block.mlp.fc1), _dequant_cache(block.mlp.fc2):
        m = _rows_into(torch.empty_like(x2), chunks, lambda r0, r1: block.mlp(h2[r0:r1]))
    del h2

    # FFN residual, per row chunk
    return _rows_into(torch.empty_like(x2), chunks,
                      lambda r0, r1: _dit._gate(x2[r0:r1], gate_mlp, m[r0:r1], adaln_index[r0:r1]))


def _dequant_cache(layer):
    """``layer.dequant_cache()``; a plain bf16 ``nn.Linear`` has none (a partly
    quantized file whose fc1 / fc2 were left in bf16)."""
    return getattr(layer, "dequant_cache", contextlib.nullcontext)()


def embed_rows_packed(dit, text_refined, audio_rows, video_rows, cond_rows, *, ref_video_rows, ref_audio_rows):
    """``H3DiT.embed_rows`` written straight into one preallocated bf16 output.

    The same matmuls are called once each with the same shapes, and each
    f32 -> bf16 result is copied into its rows of the output: no ``torch.cat``
    and no intermediate bf16 parts.
    """
    parts = [(text_refined, None)]
    for rows, proj in (
        (cond_rows, dit.video_patch_proj),
        (ref_video_rows, dit.video_patch_proj),
        (ref_audio_rows, dit.audio_patch_proj),
    ):
        if rows is not None and rows.shape[0] > 0:
            parts.append((rows, proj))
    parts.append((audio_rows, dit.audio_patch_proj))
    parts.append((video_rows, dit.video_patch_proj))
    total = sum(rows.shape[0] for rows, _ in parts)
    out = torch.empty((total, dit.cfg.hidden_size), dtype=torch.bfloat16, device=video_rows.device)
    r0 = 0
    for rows, proj in parts:
        r1 = r0 + rows.shape[0]
        out[r0:r1].copy_(rows if proj is None else _dit._linear_f32(rows, proj))
        r0 = r1
    return out


def final_rows_chunked(dit, x: torch.Tensor, t_emb: torch.Tensor, time_index: torch.Tensor, layout, plan: Vram16Plan):
    """``H3DiT.final_rows`` with the norm and the modulation per row chunk.

    Each chunk's bf16 result is copied into one float32 buffer (bf16 -> f32 is
    exact) and the output head runs ONCE over the whole buffer: the matmul is
    not split.
    """
    shift, scale = dit.final_layer.adaln_proj(t_emb)  # [M, hidden] each, f32
    fl = dit.final_layer

    def head(rows: slice, out) -> torch.Tensor:
        xr = x[rows]
        idx = time_index[rows]
        n = xr.shape[0]
        buf = torch.empty((n, xr.shape[1]), dtype=torch.float32, device=xr.device)
        for r0, r1 in row_chunks(n, plan.row_chunk):
            buf[r0:r1].copy_(_dit._modulate(fl.norm(xr[r0:r1]), shift, scale, idx[r0:r1]))
        return _dit._linear_f32(buf, out)

    return head(layout.video, fl.video_out), head(layout.audio, fl.audio_out)
