# Ported from DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio),
# commit 4e34969:
#   diffsynth/models/minimax_h3_dit.py        (the DiT, its blocks, patchify / pack helpers)
#   diffsynth/models/minimax_h3_dit_comfy.py  (stacked qkv, pruned AdaLN table)
# Copyright the DiffSynth-Studio authors. Licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Modified for
# Nz-Videomni: only the Comfy-Org stacked layout and the pruned AdaLN table;
# one unmasked attention over the whole sequence (no cu_seqlens loop); the
# token refiner hoisted out of the step loop; explicit float32 islands;
# tensor-only block arguments for block swap; VRAM 16GB mode's branch into
# engine_h3/lowvram.py (台帳 §1-101). Cross-checked against diffusers
# models/transformers/transformer_minimax_h3.py (Apache-2.0).
"""MiniMax H3 DiT (FL2VA, Comfy-Org pruned int8 ConvRot checkpoint).

One packed sequence, one block stack
------------------------------------
Rows are ``[text (L) | cond (K x grid) | audio (2A) | video (T' x grid)]``
(``H3SeqLayout``); every block runs full self-attention over all of them with
no mask (the reference pads to a multiple of 64 and loops over cu_seqlens
segments; the real-data segment is the whole unmasked sequence, so one SDPA /
SageAttention call is the same maths without the padding rows, the
``.tolist()`` sync and the per-segment copies — R2 (2) C, see engine_h3/constants.py).

What each row gets from the step is chosen by an index, not by separate
weights: AdaLN row = ``time_index * 3 + modality`` (0 video / 1 text /
2 audio) into the ``[M * 3, 6 * hidden]`` table each block projects from the
M distinct timesteps of this step (video and audio run different sigma
schedules, so M is 1..3). The final layer is indexed by ``time_index`` only.

Layout conventions this file pins (R2 (3), each checked against the source)
--------------------------------------------------------------------------
* qkv rows are stacked ``[q; k; v]`` (``minimax_h3_dit_comfy.py:20``,
  ``view(total, 3, heads, head_dim)``) — NOT the official per-head interleave
  (``minimax_h3_dit.py:139``); the two cannot be told apart from the file.
* fc1 rows are ``[gate; value]`` and SiLU goes on the first half
  (``minimax_h3_dit.py:160-162``).
* Pruned AdaLN: ``e = lerp(table[i0], table[i0+1], w)`` with
  ``pos = clamp(t, 0, 1) * (G - 1)``, ``i0 = min(floor(pos), G - 2)``, and the
  projection applied with NO SiLU (``minimax_h3_dit_comfy.py:9-14, 39-45``).
  ``t = 1 - sigma`` (1 = clean).
* The output is the data-direction velocity (x0 = x + sigma * v), returned
  as is; DiffSynth negates it for its own scheduler, we do not.

Precision islands (diffusers ``_keep_in_fp32_modules``, ComfyUI's
``adaln_dtype=float32``): the two patch projections, the AdaLN embedding and
every AdaLN projection, and the two output heads compute in float32; the block
stack, the text projection and the token refiner in bf16. The casts are
explicit in the forward, so the result does not depend on the dtype a tensor
was loaded with (``load_state_dict(assign=True)`` keeps the FILE's dtype: the
AdaLN projections are F16 in the file).

Block swap
----------
The block list is ``self.blocks`` (a name ``BlockSwapService._get_blocks``
finds) and ``H3DiTBlock.forward`` takes tensors only, positionally: the swap
wrapper moves every argument to the GPU with ``_move_to_device``, and a
non-tensor argument would be passed through but is a trap for a future
caller. The attention backend is therefore a module attribute, set once per
forward, not an argument -- and so is VRAM 16GB mode's plan
(``H3DiTBlock.vram16``, a ``lowvram.Vram16Plan`` or None), set on every block
by every ``H3DiT.forward``.

VRAM 16GB mode
--------------
With a plan, ``H3DiTBlock.forward`` and ``H3DiT.final_rows`` hand over to
``engine_h3.lowvram`` at their first line; the off path below that line is
unchanged. ``H3DiT.forward`` additionally takes ``tokens_box`` so that the
caller holds no reference to the token embedding (Docs/MINIMAX_H3_DESIGN.md
§6.5).

The model is built under ``with torch.device("meta"):``; ``swap_in_int8_linears``
(``engine_h3.quant_linear``) then replaces the four quantized Linears of each
block, and ``engine_h3.sft_load`` assigns the tensors. State-dict keys are the
Comfy-Org tensor names with the prefix stripped, so the loader needs no name
map.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine_h3.constants import ATTENTION_BACKENDS, MODALITY_AUDIO, MODALITY_TEXT, MODALITY_VIDEO
from engine_h3.quant_linear import H3Int8Linear
from engine_h3.rope import apply_rope
from h3_format import H3DiTConfig

#: AdaLN rows per timestep: one per modality tag (0 video, 1 text, 2 audio).
_MODALITY_NUM = 3
#: shift/scale/gate for attention and for the MLP.
_BLOCK_EXPAND = 6
#: shift/scale of the final norm.
_FINAL_EXPAND = 2
#: DiT patch over the VAE latent (t, h, w).
_PATCH_T, _PATCH_H, _PATCH_W = 1, 2, 2
#: Every norm in the checkpoint (config.json: norm_eps, qk_norm_eps, final_norm_eps).
_EPS = 1e-5

assert (MODALITY_VIDEO, MODALITY_TEXT, MODALITY_AUDIO) == (0, 1, 2), (
    "AdaLN の行の並びは [映像, テキスト, 音声] で固定（R2 B: 96768 = 6 x 5376 x 3）"
)


# --------------------------------------------------------------------------- #
# sequence layout and row helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class H3SeqLayout:
    """Row ranges of the packed sequence ``[text | cond | audio | video]``.
    T2V has an empty cond (``slice(L, L)``).

    ``cond``: the I2V key frame's rows (empty for T2V). Order and coordinates match
    DiffSynth (VERIFICATION_LOG §165 G4 A, §166 2a-G3); the embedding is checked end to end only (§166 2a-G4).
    """

    text: slice
    cond: slice
    audio: slice
    video: slice
    total: int

    @classmethod
    def make(cls, *, text_rows: int, cond_rows: int, audio_rows: int, video_rows: int) -> "H3SeqLayout":
        a = text_rows
        b = a + cond_rows
        c = b + audio_rows
        d = c + video_rows
        return cls(text=slice(0, a), cond=slice(a, b), audio=slice(b, c), video=slice(c, d), total=d)


def patchify_video(latent: torch.Tensor) -> torch.Tensor:
    """[1, C, T', H/16, W/16] -> [T' * (H/32) * (W/32), C * 4], rows t -> h -> w,
    each row ``(c, pt, ph, pw)`` (DiffSynth ``patchify_video``,
    einsum ``nctrhpwq->nthwcrpq``)."""
    if latent.dim() != 5 or latent.shape[0] != 1:
        raise ValueError(f"patchify_video: [1, C, T, H, W] が必要です（{list(latent.shape)}）")
    _, c, ft, fh, fw = (int(s) for s in latent.shape)
    if ft % _PATCH_T or fh % _PATCH_H or fw % _PATCH_W:
        raise ValueError(f"patchify_video: 潜在の縦横がパッチ (1,2,2) で割り切れません（{list(latent.shape)}）")
    t, h, w = ft // _PATCH_T, fh // _PATCH_H, fw // _PATCH_W
    packed = latent.reshape(1, c, t, _PATCH_T, h, _PATCH_H, w, _PATCH_W)
    packed = packed.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return packed.reshape(t * h * w, c * _PATCH_T * _PATCH_H * _PATCH_W).contiguous()


def unpatchify_video(rows: torch.Tensor, latent_frames: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Inverse of :func:`patchify_video`. ``grid_h``/``grid_w`` are the TOKEN
    grid (H/32, W/32); the result is [1, C, T', 2*grid_h, 2*grid_w]."""
    patch = _PATCH_T * _PATCH_H * _PATCH_W
    if rows.dim() != 2 or rows.shape[1] % patch or rows.shape[0] != latent_frames * grid_h * grid_w:
        raise ValueError(
            f"unpatchify_video: 行 {list(rows.shape)} が T'={latent_frames}・格子 {grid_h}x{grid_w} と合いません"
        )
    c = rows.shape[1] // patch
    packed = rows.reshape(1, latent_frames, grid_h, grid_w, c, _PATCH_T, _PATCH_H, _PATCH_W)
    latent = packed.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return latent.reshape(1, c, latent_frames * _PATCH_T, grid_h * _PATCH_H, grid_w * _PATCH_W).contiguous()


def audio_to_rows(latent: torch.Tensor) -> torch.Tensor:
    """[2, 32, A] -> [2A, 32]: all of channel 0's steps, then channel 1's
    (DiffSynth ``pack_audio``; the RoPE w coordinate relies on this order)."""
    if latent.dim() != 3:
        raise ValueError(f"audio_to_rows: [チャンネル, 32, A] が必要です（{list(latent.shape)}）")
    ac, ld, steps = (int(s) for s in latent.shape)
    return latent.permute(0, 2, 1).reshape(ac * steps, ld).contiguous()


def rows_to_audio(rows: torch.Tensor, audio_latents: int) -> torch.Tensor:
    """Inverse of :func:`audio_to_rows`: [2A, 32] -> [2, 32, A]."""
    if rows.dim() != 2 or rows.shape[0] % audio_latents:
        raise ValueError(f"rows_to_audio: 行 {list(rows.shape)} が A={audio_latents} と合いません")
    channels = rows.shape[0] // audio_latents
    return rows.reshape(channels, audio_latents, rows.shape[1]).permute(0, 2, 1).contiguous()


# --------------------------------------------------------------------------- #
# attention backends
# --------------------------------------------------------------------------- #

_SAGEATTN = None


def _sageattn():
    """``sageattention.sageattn``, imported on first use (the
    ``engine/transformer/sage_attention_service._sage_callable`` idiom: an
    absent or broken wheel only matters to a job that asked for sage)."""
    global _SAGEATTN
    if _SAGEATTN is None:
        from sageattention import sageattn

        _SAGEATTN = sageattn
    return _SAGEATTN


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str) -> torch.Tensor:
    """q, k, v [S, H, D] -> [S, H, D]; one document, no mask, scale D^-0.5."""
    qh = q.transpose(0, 1).unsqueeze(0)  # [1, H, S, D] ("HND")
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    if backend == "sage":
        out = _sageattn()(qh.contiguous(), kh.contiguous(), vh.contiguous(), tensor_layout="HND", is_causal=False)
    else:
        out = F.scaled_dot_product_attention(qh, kh, vh)
    return out.squeeze(0).transpose(0, 1)


# --------------------------------------------------------------------------- #
# modules (names = Comfy-Org tensor names)
# --------------------------------------------------------------------------- #


class H3Attention(nn.Module):
    """qkv_proj (stacked [q; k; v]) -> per-head RMSNorm on q, k -> RoPE ->
    attention -> out_proj. ``inner = heads * head_dim`` (7168) is larger than
    the hidden size (5376) in H3; that is the checkpoint's shape, not a bug."""

    def __init__(self, hidden: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, 3 * inner, bias=False, dtype=torch.bfloat16)
        self.q_norm = nn.RMSNorm(head_dim, eps=_EPS, dtype=torch.bfloat16)
        self.k_norm = nn.RMSNorm(head_dim, eps=_EPS, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(inner, hidden, bias=False, dtype=torch.bfloat16)
        #: Set by ``H3DiT.forward`` (a non-tensor must not be a block argument).
        self.attention_backend = "sdpa"

    def forward(
        self, x: torch.Tensor, rope_cos: torch.Tensor | None, rope_sin: torch.Tensor | None
    ) -> torch.Tensor:
        s = x.shape[0]
        qkv = self.qkv_proj(x).view(s, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
        out = _attention(q, k, v, self.attention_backend)
        return self.out_proj(out.reshape(s, self.heads * self.head_dim))


class H3MLP(nn.Module):
    """SwiGLU: fc1 -> [gate; value] -> silu(gate) * value -> fc2."""

    def __init__(self, hidden: int, ffn: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, 2 * ffn, bias=False, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(ffn, hidden, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * value)


class H3AdaLNProj(nn.Module):
    """Pruned-checkpoint AdaLN projection: ``linear(e)`` in float32, NO SiLU.

    Returns ``expand`` tensors of [M * modality_num, hidden] float32; row
    ``m * modality_num + tag`` is timestep m, modality tag (the
    ``view(M * modality_num, expand * hidden)`` of ``minimax_h3_dit.py:182``).
    """

    def __init__(self, rank: int, hidden: int, *, expand: int, modality_num: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.expand = expand
        self.modality_num = modality_num
        self.linear = nn.Linear(rank, expand * hidden * modality_num, bias=True, dtype=torch.float32)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        w = self.linear.weight.to(torch.float32)
        b = self.linear.bias.to(torch.float32)
        x = F.linear(t_emb.to(torch.float32), w, b)
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand * self.hidden)
        return tuple(x.chunk(self.expand, dim=-1))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``x * (1 + scale[idx]) + shift[idx]`` in float32, cast back to x's dtype —
    the promotion and the cast of DiffSynth ``_modulate_scale_shift``.

    Written in place on the gathered float32 copy: the same operations in the
    same order (products and sums commute bit for bit), but two [S, hidden]
    float32 temporaries instead of four — at 1344x768 x 124f one is ~0.8 GB.
    """
    out = scale.index_select(0, idx)
    out.add_(1.0)
    out.mul_(x)
    out.add_(shift.index_select(0, idx))
    return out.to(x.dtype)


def _gate(x: torch.Tensor, gate: torch.Tensor, other: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``x + gate[idx] * other`` in float32, cast back (DiffSynth ``_modulate_gate``)."""
    out = gate.index_select(0, idx)
    out.mul_(other)
    out.add_(x)
    return out.to(x.dtype)


class H3DiTBlock(nn.Module):
    """One of the 50 blocks (``minimax_h3_dit.py:214-234``)."""

    #: VRAM 16GB mode: a ``lowvram.Vram16Plan`` set by ``H3DiT.forward`` (None = off).
    vram16 = None

    def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int, rank: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=_EPS, dtype=torch.bfloat16)
        self.norm2 = nn.RMSNorm(hidden, eps=_EPS, dtype=torch.bfloat16)
        self.attn = H3Attention(hidden, heads, head_dim)
        self.mlp = H3MLP(hidden, ffn)
        self.adaln_proj = H3AdaLNProj(rank, hidden, expand=_BLOCK_EXPAND, modality_num=_MODALITY_NUM)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        adaln_index: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        # Tensors only, positional: see the module docstring (block swap).
        if self.vram16 is not None:
            from engine_h3 import lowvram  # noqa: PLC0415

            return lowvram.block_forward(self, x, t_emb, adaln_index, rope_cos, rope_sin)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _modulate(self.norm1(x), shift_msa, scale_msa, adaln_index)
        h = self.attn(h, rope_cos, rope_sin)
        x = _gate(x, gate_msa, h, adaln_index)
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp, adaln_index)
        h = self.mlp(h)
        return _gate(x, gate_mlp, h, adaln_index)


class H3TokenRefinerBlock(nn.Module):
    """Plain pre-norm block over the text rows only: no AdaLN, no RoPE."""

    def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=_EPS, dtype=torch.bfloat16)
        self.norm2 = nn.RMSNorm(hidden, eps=_EPS, dtype=torch.bfloat16)
        self.attn = H3Attention(hidden, heads, head_dim)
        self.mlp = H3MLP(hidden, ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), None, None)
        return x + self.mlp(self.norm2(x))


class H3TokenRefiner(nn.Module):
    def __init__(self, layers: int, hidden: int, heads: int, head_dim: int, ffn: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(H3TokenRefinerBlock(hidden, heads, head_dim, ffn) for _ in range(layers))
        self.final_norm = nn.RMSNorm(hidden, eps=_EPS, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class H3FinalLayer(nn.Module):
    """RMSNorm + shift/scale (indexed by timestep only) -> video_out, audio_out."""

    def __init__(self, hidden: int, rank: int, video_patch_dim: int, audio_dim: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=_EPS, dtype=torch.bfloat16)
        self.adaln_proj = H3AdaLNProj(rank, hidden, expand=_FINAL_EXPAND, modality_num=1)
        self.video_out = nn.Linear(hidden, video_patch_dim, bias=True, dtype=torch.float32)
        self.audio_out = nn.Linear(hidden, audio_dim, bias=True, dtype=torch.float32)


class _RopeFreqs(nn.Module):
    """Holds the file's ``rope.inv_freq`` under its own key (``rope.inv_freq``)."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.empty(n, dtype=torch.float32), persistent=True)


def _linear_f32(x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    """A float32 island: input, weight and bias in float32 whatever was loaded."""
    b = layer.bias.to(torch.float32) if layer.bias is not None else None
    return F.linear(x.to(torch.float32), layer.weight.to(torch.float32), b)


def _linear_bf16(x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    b = layer.bias.to(torch.bfloat16) if layer.bias is not None else None
    return F.linear(x.to(torch.bfloat16), layer.weight.to(torch.bfloat16), b)


class H3DiT(nn.Module):
    """The FL2VA DiT of the Comfy-Org pruned checkpoint. Build under meta."""

    def __init__(self, cfg: H3DiTConfig) -> None:
        super().__init__()
        if not cfg.pruned or cfg.adaln_grid is None or cfg.adaln_rank is None:
            raise NotImplementedError(
                "MiniMax H3: 枝刈り版（adaln_t_table を持つ）の DiT だけを扱います。非枝刈り版には対応していません"
            )
        self.cfg = cfg
        hidden, heads, hd, ffn, rank = (
            cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.ffn_hidden, cfg.adaln_rank,
        )
        video_patch_dim = cfg.latents_dim * _PATCH_T * _PATCH_H * _PATCH_W
        if 6 * cfg.rope_inv_freq_len > hd:
            raise ValueError(
                f"MiniMax H3: RoPE の次元 {6 * cfg.rope_inv_freq_len} がヘッド次元 {hd} を超えます"
            )

        self.video_patch_proj = nn.Linear(video_patch_dim, hidden, bias=True, dtype=torch.float32)
        self.audio_patch_proj = nn.Linear(cfg.audio_latents_dim, hidden, bias=True, dtype=torch.float32)
        self.condition_proj = nn.Linear(cfg.text_dim, hidden, bias=True, dtype=torch.bfloat16)
        self.token_refiner = H3TokenRefiner(cfg.token_refiner_layers, hidden, heads, hd, ffn)
        self.blocks = nn.ModuleList(H3DiTBlock(hidden, heads, hd, ffn, rank) for _ in range(cfg.num_layers))
        self.final_layer = H3FinalLayer(hidden, rank, video_patch_dim, cfg.audio_latents_dim)
        self.rope = _RopeFreqs(cfg.rope_inv_freq_len)
        self.register_buffer(
            "adaln_t_table", torch.empty(cfg.adaln_grid, rank, dtype=torch.float32), persistent=True
        )
        # Inference only: no Parameter should ever ask for grad (assign=True
        # keeps this flag through the load).
        self.requires_grad_(False)

    def normalize_dtypes(self) -> int:
        """Bring every floating tensor to the precision it is computed in. Call
        once after loading; returns how many tensors changed dtype.

        ``load_state_dict(assign=True)`` keeps the FILE's dtype. The float32
        islands cast at use anyway, but the plain bf16 Linears and norms
        outside them (token refiner, condition_proj, every RMSNorm) would
        fail on, or leak float32 from, an F16 / F32 community file. So:

        * float32: video_patch_proj, audio_patch_proj, final_layer.video_out /
          audio_out, every ``adaln_proj.linear``, ``adaln_t_table``,
          ``rope.inv_freq`` (the islands);
        * untouched: ``H3Int8Linear`` (int8 weight, float32 scale, its own bias
          cast);
        * bfloat16: every other floating Parameter / buffer.

        On the official Comfy-Org file the only change is the AdaLN
        projections, F16 -> F32: exact (every F16 value is an F32 value) and
        the same cast the forward did per call, so the outputs are bit-identical.
        """
        f32_modules = {self.video_patch_proj, self.audio_patch_proj,
                       self.final_layer.video_out, self.final_layer.audio_out, self.rope}
        f32_modules.update(m.linear for m in self.modules() if isinstance(m, H3AdaLNProj))
        changed = 0
        for mod in self.modules():
            if isinstance(mod, H3Int8Linear):
                continue
            target = torch.float32 if (mod in f32_modules or mod is self) else torch.bfloat16
            for name, p in list(mod._parameters.items()):
                if p is not None and p.is_floating_point() and p.dtype != target:
                    mod._parameters[name] = nn.Parameter(p.detach().to(target), requires_grad=False)
                    changed += 1
            for name, b in list(mod._buffers.items()):
                if b is not None and b.is_floating_point() and b.dtype != target:
                    mod._buffers[name] = b.to(target)
                    changed += 1
        return changed

    @property
    def rope_inv_freq(self) -> torch.Tensor:
        """The file's ``rope.inv_freq`` [16] float32 (state-dict key ``rope.inv_freq``)."""
        return self.rope.inv_freq

    # ----------------------------------------------------------------------- #
    # once per job
    # ----------------------------------------------------------------------- #

    def refine_text(self, text_hidden: torch.Tensor) -> torch.Tensor:
        """[L, text_dim] -> condition_proj -> token refiner -> final_norm -> [L, hidden] bf16.

        Depends on neither the timestep nor the position, so it runs once
        before the step loop (ComfyUI ``preprocess_text_embeds`` does the same).
        """
        x = _linear_bf16(text_hidden, self.condition_proj)
        return self.token_refiner(x)

    def embed_rows(
        self,
        text_refined: torch.Tensor,
        audio_rows: torch.Tensor,
        video_rows: torch.Tensor,
        cond_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Packed tokens [S, hidden] bf16 in the order ``[text | cond | audio | video]``.

        The patch projections are float32 islands; only their results are
        rounded to bf16 (diffusers aligns inputs with the f32 projection
        dtype the same way).

        ``cond_rows``: the I2V condition rows (already mixed with their noise), projected
        like video rows; ``None`` / empty for T2V. Checked end to end in VERIFICATION_LOG §166 (2a-G4).
        """
        parts = [text_refined.to(torch.bfloat16)]
        if cond_rows is not None and cond_rows.shape[0] > 0:
            parts.append(_linear_f32(cond_rows, self.video_patch_proj).to(torch.bfloat16))
        parts.append(_linear_f32(audio_rows, self.audio_patch_proj).to(torch.bfloat16))
        parts.append(_linear_f32(video_rows, self.video_patch_proj).to(torch.bfloat16))
        return torch.cat(parts, dim=0)

    # ----------------------------------------------------------------------- #
    # every step
    # ----------------------------------------------------------------------- #

    def adaln_embed(self, t_unique: torch.Tensor) -> torch.Tensor:
        """[M] t = 1 - sigma -> [M, rank] float32 by linear interpolation of the table.

        ``i0`` is clamped to ``G - 2`` so t = 1.0 (a condition row at 0.999
        drifting to 1, or the last step) interpolates inside the last
        interval instead of reading ``table[G]``.
        """
        table = self.adaln_t_table.to(torch.float32)
        g = table.shape[0]
        pos = t_unique.to(device=table.device, dtype=torch.float32).clamp(0.0, 1.0) * (g - 1)
        i0 = pos.floor().long().clamp(max=g - 2)
        w = (pos - i0).unsqueeze(1)
        return torch.lerp(table[i0], table[i0 + 1], w)

    def final_rows(
        self, x: torch.Tensor, t_emb: torch.Tensor, time_index: torch.Tensor, layout: H3SeqLayout,
        vram16=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Final norm + modulation + output heads, on the video and audio rows only.

        Every op is per row, so selecting the rows first is the reference's
        "run both heads over everything, then index_select" minus the waste.
        ``vram16``: VRAM 16GB mode's plan (``engine_h3.lowvram``), None = off.
        """
        if vram16 is not None:
            from engine_h3 import lowvram  # noqa: PLC0415

            return lowvram.final_rows_chunked(self, x, t_emb, time_index, layout, vram16)
        shift, scale = self.final_layer.adaln_proj(t_emb)  # [M, hidden] each, f32
        fl = self.final_layer

        def head(rows: slice, out: nn.Linear) -> torch.Tensor:
            h = fl.norm(x[rows])
            h = _modulate(h, shift, scale, time_index[rows])
            return _linear_f32(h, out)

        return head(layout.video, fl.video_out), head(layout.audio, fl.audio_out)

    def forward(
        self,
        tokens: torch.Tensor | None,
        *,
        layout: H3SeqLayout,
        t_unique: torch.Tensor,
        time_index: torch.Tensor,
        modality: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_backend: str = "sdpa",
        vram16=None,
        tokens_box: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One model evaluation. Returns (v_video [Nv, 96], v_audio [Na, 32]) float32,
        the data-direction velocity (x0 = x + sigma * v).

        ``vram16``: VRAM 16GB mode's plan (``engine_h3.lowvram.Vram16Plan``), None
        = off. ``tokens_box``: with ``tokens=None``, a one-item list holding the
        tokens; it is emptied here, so that with ``vram16`` the embedding is freed
        as soon as block 0 is done (the caller keeps no reference to it).
        """
        if tokens is None:
            tokens = tokens_box.pop()
        s = layout.total
        if tokens.dim() != 2 or tokens.shape[0] != s or tokens.shape[1] != self.cfg.hidden_size:
            raise ValueError(f"H3DiT: tokens {list(tokens.shape)} が系列長 {s}・幅 {self.cfg.hidden_size} と合いません")
        for name, t in (("time_index", time_index), ("modality", modality)):
            if t.shape != (s,):
                raise ValueError(f"H3DiT: {name} は [{s}] が必要です（{list(t.shape)}）")
        for name, t in (("rope_cos", rope_cos), ("rope_sin", rope_sin)):
            if t.dim() != 2 or t.shape[0] != s:
                raise ValueError(f"H3DiT: {name} は [{s}, 96] が必要です（{list(t.shape)}）")
        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"H3DiT: attention_backend は {ATTENTION_BACKENDS} のどれかです（{attention_backend!r}）")
        for block in self.blocks:
            block.attn.attention_backend = attention_backend
            block.vram16 = vram16  # every forward, None too: a kept DiT must not carry it over

        t_emb = self.adaln_embed(t_unique)
        time_index = time_index.to(device=t_emb.device, dtype=torch.long)
        adaln_index = time_index * _MODALITY_NUM + modality.to(device=t_emb.device, dtype=torch.long)

        x = tokens.to(torch.bfloat16)
        if vram16 is not None:
            del tokens
        for block in self.blocks:
            x = block(x, t_emb, adaln_index, rope_cos, rope_sin)
        # Blocks may have run on the swap device; the rest of the model lives
        # with the AdaLN table.
        x = x.to(t_emb.device)
        return self.final_rows(x, t_emb, time_index, layout, vram16=vram16)
