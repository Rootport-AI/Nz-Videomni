r"""pi-H3's DWD (Decoding Without a Decoder): the DiT's last hidden state -> pixels.

Written from a reading of the author's reference implementation, LakonLab
(https://github.com/Lakonik/LakonLab, commit 2ee2b334; Apache-2.0):
``lakonlab/models/architectures/sol_h3/model.py`` / ``sol_h3.py`` (the head:
windows, global tokens, rope, layers, projections) and
``diffusers/minimax_h3.py`` (``MiniMaxH3DWDAdaLayerNormOut``, the features at
the last step). Copyright (c) the LakonLab authors. Licensed under the Apache
License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Changes: one
clip per call (batch 1); the head's int8 layers are this engine's
``H3Int8Linear``; the window-by-window attention and the chunks below.
Cross-checked against ComfyUI-piFlow (tensor names, colour codec) read as a
specification only; nothing is copied or imported from it, and no ComfyUI code
is used (Docs/MINIMAX_H3_DESIGN.md 裁定 17). The computation is the one the experiment
ran (Docs/VERIFICATION_LOG.md §223; research notes §18.5-§18.7), moved here
with two changes that keep every number: the attention of each layer is built
window by window instead of for every row at once (:func:`_layer`), and the
clip always goes through the head in chunks (:func:`plan_chunks`).

What it computes (one clip, batch 1):

1. :func:`normalize_features`: affine-free LayerNorm (eps 1e-5) of the hidden
   state of the generated video rows, ``* (1 + scale[row]) + shift[row]`` with
   ``(scale, shift) = linear(silu(t_emb))``; ``t_emb`` the pruned 8-dim time
   coordinates, indexed by each row's time number.
2. ``feature_expand`` (5376 -> 2048 x 2 x 2): each DiT token becomes 2 x 2 head
   tokens (16 px).
3. 8 layers: tokens attend inside 5-latent x 16 x 16-token windows (odd layers
   shift by 2 latents / 8 tokens) plus the 5 global tokens (4 registers + 1
   zero, per native 5-latent block) of the native blocks the window overlaps;
   the global tokens attend to their whole block. Rope: window-local
   coordinates, the first 48 of 64 channels; global rows unrotated.
4. ``norm_out`` -> per native block: latent 0 through ``first_frame_proj_out``
   (1 frame), the others through ``proj_out`` (4 frames each): 5n + 2 latents
   -> 17n + 5 frames.
5. colour codec ``p * STD + MEAN`` ([-1, 1]), ``(c + 1) / 2``, clamp [0, 1],
   x255, round, uint8 (the video decoder's order).

Chunks (research notes §18.7): a shifted layer carries information one native
block further, so an output depends on at most 4 blocks on either side. Each
chunk keeps whole native blocks and runs with :data:`MARGIN_BLOCKS` (4, fixed)
blocks of context on each side (none past the clip's true ends); its kept
blocks are then exactly those of one pass over the whole clip
(VERIFICATION_LOG §223.8: difference 0 on the GPU). The chunk size comes from
the shape alone (:data:`CHUNK_MAX_ROWS`; :func:`plan_chunks`); a clip no longer
than the largest split chunk goes through in one pass.

Memory: per chunk, only that chunk's hidden rows go to the device (freed once
normalized); the normalized features live until ``feature_expand`` has
consumed them (``refine`` has no ``no_grad`` decorator, which would hold its
arguments to the end of the call; ``decode`` runs it inside its own); after each
chunk the CUDA cache is emptied (chunks differ in size, and the cache would
otherwise pile up their freed blocks: reserved far above allocated); each
layer computes q, k and v for one batch of windows (or one block's global
attention) at a time and writes the residual of those rows at once (every
operation is per row -- the int8 layers quantize their input per row -- so the
numbers are those of the all-rows-at-once form; the tests compare the two);
each kept block's pixels become uint8 at once and go into the preallocated CPU
``[F, H, W, 3]``.

Precision (on the GPU): the residual stream is the hidden state's dtype
(bf16); RMSNorms in float32 cast back; float Linears in the stream's dtype with
the file's F16 weight cast to it; the int8 Linears are ``H3Int8Linear`` (bf16
out); residual adds ``torch.addcmul(x, out, scale)``; attention is plain
``scaled_dot_product_attention``. On the CPU (tests) the same code runs in
float64 / float32 with plain ``nn.Linear`` layers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine_h3 import h3_pih3_format as PF
from engine_h3.quant_linear import H3Int8Linear, H3QuantLinear
from engine_h3.video_vae import _apply_rope

BLOCK = 5            # native latents per block (17 frames)
WINDOW = 16          # spatial window, head tokens
T_SHIFT = 2          # temporal shift of the odd layers (spatial shift = WINDOW // 2)
EPS = 1e-5
ROPE_THETA = 100.0
COLOR_MEAN = (-0.03, -0.088, -0.188)
COLOR_STD = (0.458, 0.448, 0.45)
#: Context blocks on each side of a chunk: the reach of the 4 shifted layers
#: (research notes §18.7). Fixed: fewer changes the pixels (VERIFICATION_LOG §223.8).
MARGIN_BLOCKS = 4
#: The largest chunk, in DiT rows (latents x (H/32) x (W/32)) including its
#: context. Set from the GPU measurement of this module on the product's path
#: (pi-H3 stage 1; the record of the measurement is the verification log's):
#: the decode phase (DWD plus the audio VAE) peaked at 2,108 MiB allocated for
#: 21,120 rows (1280 x 768), 4,268 MiB for 75,480 rows (1920 x 1088) and
#: 6,319 MiB for 80,960 rows (2560 x 1472, the highest per row: one block's
#: pixel temporaries grow with the resolution) -- about 0.040 MiB per row,
#: against the experiment's 0.102 MiB before the window-by-window attention.
#: 131,072 rows, scaled from the 2560 x 1472 figure, comes to about 10 GB. The
#: former 65,536 split a 1920 x 1088, 158-frame clip into 10 chunks and made
#: the head about 7 times slower. A chunk never keeps less than one native
#: block (so up to 9 blocks with its context, which may exceed this cap).
CHUNK_MAX_ROWS = 131072
#: Rows per piece of the per-row steps (norms, projections, residuals) and
#: windows per attention call: the experiment's values.
ROW_CHUNK = 8192
WIN_BATCH = 16

HeadConfig = PF.DWDHeadDims


# --------------------------------------------------------------------------- #
# modules (state-dict keys = the pi-H3 file's dwd_ keys)
# --------------------------------------------------------------------------- #

def float_linear(dtype=torch.float16):
    def make(name: str, i: int, o: int) -> nn.Module:
        return nn.Linear(i, o, bias=True, dtype=dtype)
    return make


def int8_linear(specs: dict, bias_dtype=torch.float16):
    """``specs``: refiner-relative layer name ('blocks.0.attn.to_qkv') -> QuantSpec."""
    def make(name: str, i: int, o: int) -> nn.Module:
        return H3Int8Linear(i, o, bias=True, spec=specs[name], bias_dtype=bias_dtype)
    return make


class _NormOut(nn.Module):
    def __init__(self, cfg: HeadConfig, dtype) -> None:
        super().__init__()
        self.linear = nn.Linear(cfg.t_dim, 2 * cfg.in_channels, bias=True, dtype=dtype)


class _Pair(nn.Module):
    def __init__(self, **mods) -> None:
        super().__init__()
        for k, v in mods.items():
            setattr(self, k, v)


class _Block(nn.Module):
    def __init__(self, cfg: HeadConfig, make, i: int, dtype) -> None:
        super().__init__()
        d = cfg.dim
        self.norm1 = nn.RMSNorm(d, eps=EPS, dtype=dtype)
        self.norm2 = nn.RMSNorm(d, eps=EPS, dtype=dtype)
        self.scale1 = nn.Parameter(torch.empty(d, dtype=dtype), requires_grad=False)
        self.scale2 = nn.Parameter(torch.empty(d, dtype=dtype), requires_grad=False)
        self.attn = _Pair(to_qkv=make(f"blocks.{i}.attn.to_qkv", d, 3 * d),
                          to_out=make(f"blocks.{i}.attn.to_out", d, d))
        self.ff = _Pair(w1=make(f"blocks.{i}.ff.w1", d, 2 * cfg.ffn_mult * d),
                        w2=make(f"blocks.{i}.ff.w2", cfg.ffn_mult * d, d))


class _Refiner(nn.Module):
    def __init__(self, cfg: HeadConfig, make, dtype) -> None:
        super().__init__()
        d, (eh, ew), p = cfg.dim, cfg.expansion, cfg.patch
        self.feature_expand = nn.Linear(cfg.in_channels, d * eh * ew, bias=True, dtype=dtype)
        self.register_tokens = nn.Parameter(torch.empty(1, cfg.num_register, d, dtype=dtype), requires_grad=False)
        self.blocks = nn.ModuleList(_Block(cfg, make, i, dtype) for i in range(cfg.num_layers))
        self.norm_out = nn.LayerNorm(d, eps=EPS, dtype=dtype)
        self.proj_out = nn.Linear(d, cfg.out_ch * cfg.patch_t * p * p, bias=True, dtype=dtype)
        self.first_frame_proj_out = nn.Linear(d, cfg.out_ch * cfg.first_t * p * p, bias=True, dtype=dtype)


class DWDHead(nn.Module):
    def __init__(self, cfg: HeadConfig | None = None, make=None, dtype=torch.float16) -> None:
        super().__init__()
        cfg = cfg or PF.DWD_HEAD
        self.cfg = cfg
        make = make or float_linear(dtype)
        self.dwd_norm_out = _NormOut(cfg, dtype)
        self.dwd_pixel_refiner = _Refiner(cfg, make, dtype)
        self.requires_grad_(False)


def load_head(path, device, *, cfg: HeadConfig | None = None) -> tuple[DWDHead, dict]:
    """The head with the file's tensors as stored (int8 + F32 scales, F16 elsewhere), on ``device``."""
    from engine_h3.sft_load import load_streaming  # noqa: PLC0415

    started = time.perf_counter()
    cfg = cfg or PF.DWD_HEAD
    info = PF.inspect_file(path, head=cfg)
    with torch.device("meta"):
        head = DWDHead(cfg, make=int8_linear(info.dwd_specs))
    want = len(PF.dwd_tensor_table(cfg))
    rep = load_streaming(head, str(path), info.header, key_map=lambda k: k if k.startswith("dwd_") else None)
    if rep.loaded != want:
        raise RuntimeError(f"DWD: {rep.loaded} 個のテンソルを読みました（{want} 個のはず）")
    head.requires_grad_(False).eval()
    head.to(device)
    return head, {"loaded_tensors": int(rep.loaded), "loaded_mib": round(rep.bytes / 2**20, 1),
                  "int8_layers": len(info.dwd_specs), "seconds": round(time.perf_counter() - started, 2)}


# --------------------------------------------------------------------------- #
# geometry (LakonLab _partition / _window_rows / _global_rows / _rope, rewritten)
# --------------------------------------------------------------------------- #

def _spans(n: int, shift: int, size: int) -> list[tuple[int, int]]:
    """Windows [lo, hi) along one axis of length n, the grid shifted by ``shift`` (empty ones dropped)."""
    out = []
    for i in range(-(-(n + shift) // size)):
        lo, hi = max(i * size - shift, 0), min((i + 1) * size - shift, n)
        if hi > lo:
            out.append((lo, hi))
    return out


@dataclass
class Geometry:
    T: int
    H: int
    W: int
    nb: int                 # native blocks
    N: int                  # token rows (t, y, x order); the nb * NUM_GLOBAL global rows follow
    R: int
    block_rows: list        # [(r0, r1)] token rows of each native block
    windows: dict           # shifted -> [(q rows [n, Lq], k rows [n, Lk])] grouped by shape
    glob: list              # per block: (q rows [S], k rows [block tokens + S])
    rope: dict              # shifted -> (cos, sin) float32 [N, 1, rope_dim] on the CPU
    window_count: dict


def _rope_tables(T: int, H: int, W: int, s: int, st: int, rope_dim: int):
    lt = ((torch.arange(T) + st) % BLOCK).float()
    ly = ((torch.arange(H) + s) % WINDOW).float()
    lx = ((torch.arange(W) + s) % WINDOW).float()
    axes = ((lt + 0.5) / BLOCK * 2 - 1, (ly + 0.5) / WINDOW * 2 - 1, (lx + 0.5) / WINDOW * 2 - 1)
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inv_freq = 1.0 / ROPE_THETA ** torch.arange(0, 1, 6 / rope_dim, dtype=torch.float32)
    angles = (2.0 * math.pi * grid[:, :, None] * inv_freq[None, None, :]).flatten(1, 2)
    angles = angles.tile(2)[:, None, :]            # both rotary halves share the angles
    return angles.cos(), angles.sin()


def build_geometry(T: int, H: int, W: int, rope_dim: int, num_global: int) -> Geometry:
    hw = H * W
    N = T * hw
    nb = -(-T // BLOCK)
    S = num_global
    idx = torch.arange(N).view(T, H, W)
    block_rows = [(b * BLOCK * hw, min((b + 1) * BLOCK, T) * hw) for b in range(nb)]
    glob_rows = [torch.arange(N + b * S, N + (b + 1) * S) for b in range(nb)]
    windows, counts, rope = {}, {}, {}
    for shifted in (False, True):
        s, st = (WINDOW // 2, T_SHIFT) if shifted else (0, 0)
        groups: dict[tuple[int, int], list] = {}
        n = 0
        for t0, t1 in _spans(T, st, BLOCK):
            blocks = sorted({t0 // BLOCK, (t1 - 1) // BLOCK})
            gk = torch.cat([glob_rows[b] for b in blocks])
            for y0, y1 in _spans(H, s, WINDOW):
                for x0, x1 in _spans(W, s, WINDOW):
                    qi = idx[t0:t1, y0:y1, x0:x1].reshape(-1)
                    ki = torch.cat([qi, gk])
                    groups.setdefault((len(qi), len(ki)), []).append((qi, ki))
                    n += 1
        windows[shifted] = [(torch.stack([a for a, _ in g]), torch.stack([b for _, b in g])) for g in groups.values()]
        counts[shifted] = n
        rope[shifted] = _rope_tables(T, H, W, s, st, rope_dim)
    glob = [(glob_rows[b], torch.cat([torch.arange(r0, r1), glob_rows[b]])) for b, (r0, r1) in enumerate(block_rows)]
    return Geometry(T=T, H=H, W=W, nb=nb, N=N, R=N + nb * S, block_rows=block_rows, windows=windows,
                    glob=glob, rope=rope, window_count=counts)


def frames_for_latents(t: int) -> int:
    """Pixel frames of ``t`` latents: 5n + 2 -> 17n + 5."""
    return sum(block_frames(t))


def block_frames(t: int) -> list[int]:
    """Pixel frames of each native block of a ``t``-latent clip."""
    return [1 + (min((b + 1) * BLOCK, t) - b * BLOCK - 1) * 4 for b in range(-(-int(t) // BLOCK))]


# --------------------------------------------------------------------------- #
# the computation
# --------------------------------------------------------------------------- #

def _chunks(n: int, size: int):
    return [(i, min(i + size, n)) for i in range(0, n, size)]


def _hi(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


def _linear(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(layer, H3QuantLinear):
        return layer(x)
    b = None if layer.bias is None else layer.bias.to(x.dtype)
    return F.linear(x, layer.weight.to(x.dtype), b)


def _rms(x: torch.Tensor, weight) -> torch.Tensor:
    xh = _hi(x)
    w = None if weight is None else weight.to(xh.dtype)
    return F.rms_norm(xh, (x.shape[-1],), weight=w, eps=EPS).to(x.dtype)


@torch.no_grad()
def normalize_features(head: DWDHead, hidden: torch.Tensor, t_emb: torch.Tensor, time_index: torch.Tensor,
                       row_chunk: int = ROW_CHUNK) -> torch.Tensor:
    """``dwd_norm_out`` on ``hidden`` [rows, 5376] (the residual dtype); ``t_emb`` [M, 8], ``time_index`` [rows]."""
    lin = head.dwd_norm_out.linear
    th = _hi(t_emb.to(hidden.device))
    emb = F.linear(F.silu(th), lin.weight.to(th.dtype), lin.bias.to(th.dtype)).to(hidden.dtype)
    scale, shift = emb.chunk(2, dim=-1)
    idx = time_index.to(device=hidden.device, dtype=torch.long)
    out = torch.empty_like(hidden)
    for r0, r1 in _chunks(hidden.shape[0], row_chunk):
        y = F.layer_norm(hidden[r0:r1], (hidden.shape[-1],), eps=EPS)
        out[r0:r1] = y.mul_(1 + scale[idx[r0:r1]]).add_(shift[idx[r0:r1]])
    return out


def _qkv_rows(blk: _Block, X: torch.Tensor, rows: torch.Tensor, n_token: int, cfg: HeadConfig, row_chunk: int,
              rope: tuple, *, q_from: int = 0):
    """q, k, v ``[len(rows), heads, head_dim]`` of the CPU row indices ``rows`` of ``X`` (norm1, to_qkv,
    per-head RMSNorm, rope on the token rows ``< n_token``). Only ``rows[q_from:]`` get q (the others: None)."""
    nh, hd = cfg.heads, cfg.head_dim
    dev, dt = X.device, X.dtype
    n = int(rows.numel())
    k = torch.empty(n, nh, hd, dtype=dt, device=dev)
    v = torch.empty(n, nh, hd, dtype=dt, device=dev)
    q = torch.empty(n - q_from, nh, hd, dtype=dt, device=dev)
    for r0, r1 in _chunks(n, row_chunk):
        idx = rows[r0:r1].to(dev)
        y = _linear(blk.attn.to_qkv, _rms(X[idx], blk.norm1.weight)).view(r1 - r0, nh, 3 * hd)
        qq, kk, vv = y.chunk(3, dim=-1)                # per head [q; k; v]
        k[r0:r1], v[r0:r1] = _rms(kk, None), vv
        if r1 > q_from:
            a = max(r0, q_from)
            q[a - q_from:r1 - q_from] = _rms(qq[a - r0:], None)
        del y, qq, kk, vv
    pos = (rows < n_token).nonzero().squeeze(1)        # positions (in ``rows``) of the token rows
    if pos.numel():
        cos_all, sin_all = rope
        cos, sin = cos_all[rows[pos]].to(dev), sin_all[rows[pos]].to(dev)
        pd = pos.to(dev)
        k[pd] = _apply_rope(k[pd], cos, sin)
        has_q = pos >= q_from
        if bool(has_q.any()):
            qp = (pos[has_q] - q_from).to(dev)
            sel = has_q.to(dev)
            q[qp] = _apply_rope(q[qp], cos[sel], sin[sel])
    return (q if q_from < n else None), k, v


def _residual(blk: _Block, X: torch.Tensor, rows: torch.Tensor, o: torch.Tensor, s1: torch.Tensor,
              row_chunk: int) -> None:
    """``X[rows] = addcmul(X[rows], to_out(o), s1)`` (``o`` [len(rows), dim]) in pieces of ``row_chunk``."""
    for r0, r1 in _chunks(int(rows.numel()), row_chunk):
        idx = rows[r0:r1].to(X.device)
        X[idx] = torch.addcmul(X[idx], _linear(blk.attn.to_out, o[r0:r1]), s1)


def _layer(blk: _Block, X: torch.Tensor, geo: Geometry, shifted: bool, cfg: HeadConfig,
           row_chunk: int, win_batch: int) -> None:
    """One head layer on ``X`` [R, dim] in place, attention built window by window.

    Order (each step reads only rows no earlier step of this layer has changed):
    1. the global tokens' attention, block by block (its keys are the block's
       token rows and its own global rows, before any residual);
    2. the windows, ``win_batch`` at a time: q, k, v of their rows and of the
       global rows they see, the attention, then those rows' residual at once
       (a token row is a query of exactly one window per layer and a key of
       that window only);
    3. the global rows' residual;
    4. the FFN, per row piece.
    The attention calls get the same q, k, v tensors (values, shapes, strides)
    as the all-rows-at-once form (the windows grouped and batched the same way).
    """
    R, D = X.shape
    nh, hd = cfg.heads, cfg.head_dim
    dev, dt = X.device, X.dtype
    rope = geo.rope[shifted]
    s1, s2 = blk.scale1.to(dt), blk.scale2.to(dt)
    n_glob = R - geo.N
    out_g = torch.empty(n_glob, nh, hd, dtype=dt, device=dev)
    for gq, gk in geo.glob:
        q, k, v = _qkv_rows(blk, X, gk, geo.N, cfg, row_chunk, rope, q_from=int(gk.numel() - gq.numel()))
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])
        out_g[(gq - geo.N).to(dev)] = o[0].transpose(0, 1)
        del q, k, v, o
    local = torch.full((R,), -1, dtype=torch.long)
    for qi_all, ki_all in geo.windows[shifted]:
        for c0, c1 in _chunks(qi_all.shape[0], win_batch):
            qi, ki = qi_all[c0:c1], ki_all[c0:c1]
            q_flat = qi.reshape(-1)
            extra = torch.unique(ki[:, qi.shape[1]:])        # the global rows these windows see
            rows = torch.cat([q_flat, extra])
            local[rows] = torch.arange(rows.numel())
            q, k, v = _qkv_rows(blk, X, rows, geo.N, cfg, row_chunk, rope)
            lq, lk = local[qi].to(dev), local[ki].to(dev)
            o = F.scaled_dot_product_attention(q[lq].transpose(1, 2), k[lk].transpose(1, 2), v[lk].transpose(1, 2))
            del q, k, v
            o = o.transpose(1, 2).reshape(q_flat.numel(), D)
            _residual(blk, X, q_flat, o, s1, row_chunk)
            local[rows] = -1
            del o
    if n_glob:
        _residual(blk, X, torch.arange(geo.N, R), out_g.reshape(n_glob, D), s1, row_chunk)
    del out_g
    for r0, r1 in _chunks(R, row_chunk):
        gate, val = _linear(blk.ff.w1, _rms(X[r0:r1], blk.norm2.weight)).chunk(2, dim=-1)
        X[r0:r1] = torch.addcmul(X[r0:r1], _linear(blk.ff.w2, F.silu(gate) * val), s2)


def refine(head: DWDHead, feats: torch.Tensor, grid: tuple[int, int, int], sink, *, keep_blocks=None,
           row_chunk: int = ROW_CHUNK, win_batch: int = WIN_BATCH) -> None:
    """Normalized features [t * gh * gw, 5376] on a (t, gh, gw) DiT token grid -> the pixels of the
    native blocks ``keep_blocks`` ([b0, b1), default all): ``sink(b, pixels)`` per kept block, ``b``
    relative to ``b0``, ``pixels`` [f, 3, 16H, 16W] (the head's normalized RGB, before the colour
    codec) in float32 (float64 for a float64 stream) on the stream's device.

    Call it inside ``torch.no_grad()`` (``decode`` and ``run_refiner`` do). It carries no
    ``no_grad`` decorator of its own on purpose: such a wrapper holds its arguments until the
    call returns, and ``feats`` must go as soon as ``feature_expand`` has read it -- it is freed
    here when the caller kept no reference (``decode`` passes the expression's result, and keeps
    none)."""
    cfg, ref = head.cfg, head.dwd_pixel_refiner
    t, gh, gw = (int(v) for v in grid)
    eh, ew = cfg.expansion
    H, W, D, P = gh * eh, gw * ew, cfg.dim, cfg.patch
    if feats.shape[0] != t * gh * gw:
        raise ValueError(f"DWD: 特徴 {list(feats.shape)} が格子 {grid} と合いません")
    dev, dt = feats.device, feats.dtype
    geo = build_geometry(t, H, W, cfg.rope_dim, cfg.num_register + 1)
    X = torch.empty(geo.R, D, dtype=dt, device=dev)
    hw, HW = gh * gw, H * W
    step = max(1, row_chunk // hw)
    for t0, t1 in _chunks(t, step):
        y = _linear(ref.feature_expand, feats[t0 * hw:t1 * hw])
        X[t0 * HW:t1 * HW] = y.view(t1 - t0, gh, gw, eh, ew, D).permute(0, 1, 3, 2, 4, 5).reshape(-1, D)
        del y
    del feats   # the last reference when the caller kept none (decode)
    g = torch.cat([ref.register_tokens[0].to(dt), torch.zeros(1, D, dtype=dt, device=dev)])
    X[geo.N:] = g.repeat(geo.nb, 1)
    for li, blk in enumerate(ref.blocks):
        _layer(blk, X, geo, li % 2 == 1, cfg, row_chunk, win_batch)
    b0, b1 = (0, geo.nb) if keep_blocks is None else keep_blocks
    oc, pt, ft = cfg.out_ch, cfg.patch_t, cfg.first_t
    nw, bw = ref.norm_out.weight.to(dt), ref.norm_out.bias.to(dt)
    odt = dt if dt in (torch.float32, torch.float64) else torch.float32
    for b in range(b0, b1):
        r0, r1 = geo.block_rows[b]
        n = (r1 - r0) // HW
        y = F.layer_norm(X[r0:r1], (D,), nw, bw, EPS)
        first = _linear(ref.first_frame_proj_out, y[:HW]).view(H, W, oc, ft, P, P)
        frames = [first.permute(3, 2, 0, 4, 1, 5).reshape(ft, oc, H * P, W * P).to(odt)]
        del first
        if n > 1:
            rest = _linear(ref.proj_out, y[HW:]).view(n - 1, H, W, oc, pt, P, P)
            frames.append(rest.permute(0, 4, 3, 1, 5, 2, 6).reshape((n - 1) * pt, oc, H * P, W * P).to(odt))
            del rest
        del y
        sink(b - b0, torch.cat(frames))
        del frames


def run_refiner(head: DWDHead, feats: torch.Tensor, grid, *, keep_blocks=None,
                row_chunk: int = ROW_CHUNK, win_batch: int = WIN_BATCH) -> torch.Tensor:
    """:func:`refine` with every kept block's pixels concatenated (``[F, 3, 16H, 16W]``; for tests),
    inside ``torch.no_grad()``. The caller's ``feats`` stays the caller's."""
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        refine(head, feats, grid, lambda b, px: pieces.append(px), keep_blocks=keep_blocks,
               row_chunk=row_chunk, win_batch=win_batch)
    return torch.cat(pieces)


def plan_chunks(t: int, gh: int, gw: int) -> list[dict]:
    """The chunks of a ``t``-latent clip on a (gh, gw) DiT grid, from the shape alone.

    ``max_blocks = CHUNK_MAX_ROWS // (5 gh gw)`` (at least 1) native blocks fit
    one chunk; a split chunk keeps ``content = max_blocks - 2 x MARGIN_BLOCKS``
    (at least 1) and runs with the margins, so it holds up to ``max(max_blocks,
    content + 2 x MARGIN_BLOCKS)`` blocks. A clip of at most that many blocks
    goes through in one pass: split, its chunks would each be (nearly) the
    whole clip, the same work repeated. Longer clips are split.
    """
    nb = -(-int(t) // BLOCK)
    max_blocks = max(1, CHUNK_MAX_ROWS // (BLOCK * int(gh) * int(gw)))
    content = max(1, max_blocks - 2 * MARGIN_BLOCKS)
    if nb <= max(max_blocks, content + 2 * MARGIN_BLOCKS):
        content = nb
    return _plan(t, content)


def _plan(t: int, content_blocks: int) -> list[dict]:
    """Kept native blocks [k0, k1) of ``content_blocks`` each, with context [c0, c1) of
    :data:`MARGIN_BLOCKS` on each side (none past the clip's true ends)."""
    nb = -(-int(t) // BLOCK)
    out = []
    for k0 in range(0, nb, int(content_blocks)):
        k1 = min(k0 + int(content_blocks), nb)
        out.append({"keep": (k0, k1), "context": (max(0, k0 - MARGIN_BLOCKS), min(nb, k1 + MARGIN_BLOCKS))})
    return out


def _codec(px: torch.Tensor) -> torch.Tensor:
    """float32 [f, 3, H, W] (the head's normalized RGB) -> uint8 [f, H, W, 3] where ``px`` is."""
    mean = torch.tensor(COLOR_MEAN, dtype=torch.float32, device=px.device).view(1, 3, 1, 1)
    std = torch.tensor(COLOR_STD, dtype=torch.float32, device=px.device).view(1, 3, 1, 1)
    c = px.float() * std + mean                                  # colour codec: [-1, 1]
    x = ((c + 1.0) * 0.5).clamp_(0.0, 1.0).permute(0, 2, 3, 1)
    return (x * 255.0).round().to(torch.uint8)


@dataclass
class DWDFeatures:
    """What the denoise's last evaluation hands the decode (all on the CPU).

    ``hidden`` [rows, 5376] (the DiT's dtype) of the generated video rows in
    t -> h -> w order; ``t_emb`` [M, 8] float32; ``time_index`` [rows] long;
    ``grid`` (latents, H/32, W/32).
    """

    hidden: torch.Tensor
    t_emb: torch.Tensor
    time_index: torch.Tensor
    grid: tuple[int, int, int]


@torch.no_grad()
def decode(head: DWDHead, feats_in: DWDFeatures, device, *, row_chunk: int = ROW_CHUNK,
           win_batch: int = WIN_BATCH) -> tuple[torch.Tensor, list[dict]]:
    """``feats_in`` -> (CPU uint8 [F, 16H, 16W, 3], per-chunk records), chunk by chunk (:func:`plan_chunks`)."""
    t, gh, gw = (int(v) for v in feats_in.grid)
    hw = gh * gw
    hidden, tidx = feats_in.hidden, feats_in.time_index
    if hidden.shape[0] != t * hw or tidx.shape[0] != t * hw:
        raise ValueError(f"DWD: 隠れ状態 {list(hidden.shape)} が格子 {[t, gh, gw]} と合いません")
    cfg = head.cfg
    H, W = gh * cfg.expansion[0] * cfg.patch, gw * cfg.expansion[1] * cfg.patch
    frames = block_frames(t)
    offsets = [sum(frames[:b]) for b in range(len(frames))]
    out = torch.empty((sum(frames), H, W, 3), dtype=torch.uint8)
    dev = torch.device(device)
    records = []
    for ch in plan_chunks(t, gh, gw):
        (k0, k1), (c0, c1) = ch["keep"], ch["context"]
        l0, l1 = c0 * BLOCK, min(c1 * BLOCK, t)
        started = time.perf_counter()
        box = [hidden[l0 * hw:l1 * hw].to(dev)]

        def sink(b: int, px: torch.Tensor, k0=k0) -> None:
            f0 = offsets[k0 + b]
            out[f0:f0 + px.shape[0]] = _codec(px).cpu()

        # The device copy of the hidden rows goes once normalized; the features
        # once feature_expand has read them (refine drops its own reference).
        refine(head, normalize_features(head, box.pop(), feats_in.t_emb, tidx[l0 * hw:l1 * hw], row_chunk=row_chunk),
               (l1 - l0, gh, gw), sink, keep_blocks=(k0 - c0, k1 - c0), row_chunk=row_chunk, win_batch=win_batch)
        del box, sink   # refine has returned: nothing of this chunk is left on the device
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            # Chunks differ in size, so the caching allocator would keep the previous
            # chunk's freed blocks and take new ones beside them: on the GPU, 10 chunks
            # of 2560 x 1472 reserved 16.9 GB for 7.7 GB allocated. Each chunk starts empty.
            torch.cuda.empty_cache()
        records.append({"keep_blocks": [k0, k1], "context_blocks": [c0, c1], "latents": [l0, l1],
                        "frames": sum(frames[k0:k1]), "seconds": round(time.perf_counter() - started, 3)})
    return out, records


class DWDDecoder:
    """The video decoder's shape for pi-H3: ``from_file`` reads the head, ``decode`` returns CPU uint8
    ``[F, H, W, 3]``. Built for one job and dropped after it (never kept)."""

    def __init__(self, head: DWDHead, device, load_report: dict) -> None:
        self.head, self.device = head, torch.device(device)
        self.load_report = load_report
        self.chunks: list[dict] = []

    @classmethod
    def from_file(cls, path, *, device) -> "DWDDecoder":
        head, rep = load_head(path, device)
        return cls(head, device, rep)

    def decode(self, feats: DWDFeatures) -> torch.Tensor:
        frames, self.chunks = decode(self.head, feats, self.device)
        return frames
