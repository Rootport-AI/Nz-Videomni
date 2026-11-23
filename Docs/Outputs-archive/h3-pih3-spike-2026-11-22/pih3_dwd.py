r"""pi-H3 DWD (Decoding Without Decoder) head for the MiniMax H3 spike (2026-11-22).

Written from a reading of the author's reference implementation, LakonLab (Apache-2.0):
``lakonlab/models/architectures/sol_h3/model.py`` / ``sol_h3.py`` (the head: windows, global
tokens, rope, layers, projections) and ``diffusers/minimax_h3.py`` (``MiniMaxH3DWDAdaLayerNormOut``,
the features at the last step). Cross-checked against ComfyUI-piFlow (``minimax_h3_dwd.py``,
``sol_h3.py``: tensor names, colour codec) and ComfyUI's ``NonCausalViT3DDecoder`` (geometry,
the unfused ``linear_input_act`` path). Nothing is copied or imported from them.

What it computes (one clip, batch 1):

1. ``normalize_features``: affine-free LayerNorm (eps 1e-5) of the DiT's last hidden state of the
   generated video rows, ``* (1 + scale[row]) + shift[row]`` with ``(scale, shift) =
   linear(silu(t_emb))`` (chunk order scale, shift), ``t_emb`` the pruned 8-dim time
   coordinates indexed by each row's time number.
2. ``feature_expand`` (5376 -> 2048 x 2 x 2): each DiT token becomes 2 x 2 Sol tokens (16 px).
3. 8 Sol layers. Tokens attend inside 5-latent x 16 x 16-token windows (odd layers shift by 2
   latents / 8 tokens) plus the 5 global tokens (4 registers + 1 zero, per native 5-latent block)
   of the native blocks the window overlaps; the global tokens attend to their whole block.
   Rope: window-local coordinates, rotating the first 48 of 64 channels; global rows unrotated.
4. ``norm_out`` -> per native block: latent 0 through ``first_frame_proj_out`` (1 frame), the
   others through ``proj_out`` (4 frames each). 5n + 2 latents -> 17n + 5 frames.
5. ``to_uint8``: colour codec ``p * STD + MEAN`` ([-1, 1]), ``(c + 1) / 2``, then the engine's
   ``_frames_uint8`` order: clamp [0, 1] -> x255 -> round -> uint8.

Precision (on the GPU): the residual stream is bf16 (both references); RMSNorms are computed in
float32 and cast to bf16 (LakonLab), with the file's F16 norm weight lifted to float32 (not
rounded to bf16 -- undecided by the references, see the report); float Linears compute in bf16
with the F16 weight cast to bf16 (both references); the INT8 Linears are the product's
``H3Int8Linear`` (bf16 out); residual adds are ``torch.addcmul(x, out, scale_bf16)`` (ComfyUI's
unfused path); attention is plain ``scaled_dot_product_attention``. On the CPU the same code
runs in float64 / float32 with plain ``nn.Linear`` layers (``make=float_linear``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine_h3.quant_linear import H3Int8Linear, H3QuantLinear
from engine_h3.video_vae import _apply_rope

BLOCK = 5            # native latents per block (17 frames)
WINDOW = 16          # spatial window, Sol tokens
T_SHIFT = 2          # temporal shift of the odd layers (spatial shift = WINDOW // 2)
NUM_REGISTER = 4
NUM_GLOBAL = NUM_REGISTER + 1
EPS = 1e-5
ROPE_THETA = 100.0
FRAMES_TO_TRIM = 3   # the first latent of every block keeps 1 of its 4 frames
COLOR_MEAN = (-0.03, -0.088, -0.188)
COLOR_STD = (0.458, 0.448, 0.45)
PREFIX_NORM = "dwd_norm_out."
PREFIX_REF = "dwd_pixel_refiner."
INT8_LEAVES = ("attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2")


@dataclass(frozen=True)
class HeadConfig:
    in_channels: int = 5376
    t_dim: int = 8
    dim: int = 2048
    heads: int = 32
    head_dim: int = 64
    ffn_mult: int = 4
    num_layers: int = 8
    expansion: tuple[int, int] = (2, 2)
    patch: int = 16
    patch_t: int = 4
    out_ch: int = 3

    @property
    def rope_dim(self) -> int:
        return int(self.head_dim * 0.75)

    @property
    def first_t(self) -> int:
        return self.patch_t - FRAMES_TO_TRIM


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
        self.register_tokens = nn.Parameter(torch.empty(1, NUM_REGISTER, d, dtype=dtype), requires_grad=False)
        self.blocks = nn.ModuleList(_Block(cfg, make, i, dtype) for i in range(cfg.num_layers))
        self.norm_out = nn.LayerNorm(d, eps=EPS, dtype=dtype)
        self.proj_out = nn.Linear(d, cfg.out_ch * cfg.patch_t * p * p, bias=True, dtype=dtype)
        self.first_frame_proj_out = nn.Linear(d, cfg.out_ch * cfg.first_t * p * p, bias=True, dtype=dtype)


class DWDHead(nn.Module):
    def __init__(self, cfg: HeadConfig = HeadConfig(), make=None, dtype=torch.float16) -> None:
        super().__init__()
        self.cfg = cfg
        make = make or float_linear(dtype)
        self.dwd_norm_out = _NormOut(cfg, dtype)
        self.dwd_pixel_refiner = _Refiner(cfg, make, dtype)
        self.requires_grad_(False)


def int8_layer_names(cfg: HeadConfig = HeadConfig()) -> list[str]:
    return [f"blocks.{i}.{leaf}" for i in range(cfg.num_layers) for leaf in INT8_LEAVES]


# --------------------------------------------------------------------------- #
# the pi-H3 file
# --------------------------------------------------------------------------- #

def read_markers(path, header) -> dict[str, dict]:
    """Refiner-relative INT8 layer -> its comfy_quant JSON (only the tiny marker tensors are read)."""
    import json

    import sft_quant_format

    out = {}
    for key in sorted(header.tensors):
        if key.startswith(PREFIX_REF) and key.endswith(".comfy_quant"):
            layer = key[len(PREFIX_REF):-len(".comfy_quant")]
            out[layer] = json.loads(sft_quant_format.read_tensor_bytes(path, header, key).decode("utf-8"))
    return out


def specs_from_markers(markers: dict[str, dict]) -> dict:
    import h3_format

    return {layer: h3_format.parse_marker(doc, accepted=(h3_format.INT8,)) for layer, doc in markers.items()}


def meta_check(path, header) -> dict:
    """Build the head on meta and match the file's dwd_ keys by name, shape and (int8 layers) dtype."""
    from engine.sft_quant.sft_reader import TORCH_DTYPES

    markers = read_markers(path, header)
    specs = specs_from_markers(markers)
    with torch.device("meta"):
        head = DWDHead(make=int8_linear(specs))
    sd = head.state_dict(keep_vars=True)
    file_keys = sorted(k for k in header.tensors if k.startswith("dwd_"))
    markers_k = [k for k in file_keys if k.endswith(".comfy_quant")]
    tensors_k = [k for k in file_keys if not k.endswith(".comfy_quant")]
    shape_bad = [f"{k}: file {list(header.tensors[k].shape)} head {list(sd[k].shape)}"
                 for k in tensors_k if k in sd and tuple(header.tensors[k].shape) != tuple(sd[k].shape)]
    int8_dtype_bad = []
    for layer in int8_layer_names():
        for leaf in ("weight", "weight_scale"):
            k = f"{PREFIX_REF}{layer}.{leaf}"
            if k in sd and TORCH_DTYPES.get(header.tensors[k].dtype) != sd[k].dtype:
                int8_dtype_bad.append(f"{k}: file {header.tensors[k].dtype} head {sd[k].dtype}")
    want_markers = sorted(f"{PREFIX_REF}{n}.comfy_quant" for n in int8_layer_names())
    rec = {
        "file_dwd_keys": len(file_keys), "file_marker_keys": len(markers_k), "file_tensor_keys": len(tensors_k),
        "head_state_dict_keys": len(sd),
        "missing_in_head": sorted(set(tensors_k) - set(sd)), "missing_in_file": sorted(set(sd) - set(tensors_k)),
        "shape_mismatch": shape_bad, "int8_dtype_mismatch": int8_dtype_bad,
        "markers_match_int8_layers": sorted(markers_k) == want_markers,
        "int8_layers": len(specs),
    }
    rec["passed"] = (len(file_keys) == 171 and len(sd) == 139 and len(markers_k) == 32
                     and not rec["missing_in_head"] and not rec["missing_in_file"] and not shape_bad
                     and not int8_dtype_bad and rec["markers_match_int8_layers"])
    return rec


def load_head(path, device, header=None) -> tuple[DWDHead, dict]:
    """The head with the file's tensors as stored (int8 + F32 scales, F16 elsewhere), on ``device``."""
    import sft_quant_format
    from engine_h3.sft_load import load_streaming

    t0 = time.perf_counter()
    header = header or sft_quant_format.read_header(str(path))
    specs = specs_from_markers(read_markers(path, header))
    with torch.device("meta"):
        head = DWDHead(make=int8_linear(specs))
    rep = load_streaming(head, str(path), header, key_map=lambda k: k if k.startswith("dwd_") else None)
    if rep.loaded != 139:
        raise RuntimeError(f"DWD head: {rep.loaded} tensors loaded, 139 expected")
    head.requires_grad_(False).eval()
    head.to(device)
    return head, {"loaded_tensors": rep.loaded, "loaded_mib": round(rep.bytes / 2**20, 1),
                  "int8_layers": len(specs), "seconds": round(time.perf_counter() - t0, 2)}


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


def build_geometry(T: int, H: int, W: int, rope_dim: int) -> Geometry:
    hw = H * W
    N = T * hw
    nb = -(-T // BLOCK)
    S = NUM_GLOBAL
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
    nb, tail = divmod(int(t), BLOCK)
    n = nb * (1 + (BLOCK - 1) * 4)
    if tail:
        n += 1 + (tail - 1) * 4
    return n


def block_frames(t: int) -> list[int]:
    """Pixel frames of each native block of a ``t``-latent clip."""
    return [1 + (min((b + 1) * BLOCK, t) - b * BLOCK - 1) * 4 for b in range(-(-t // BLOCK))]


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
                       row_chunk: int = 8192) -> torch.Tensor:
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


def _layer(blk: _Block, X: torch.Tensor, geo: Geometry, shifted: bool, cfg: HeadConfig,
           row_chunk: int, win_batch: int, rope_dev: dict) -> None:
    R, D = X.shape
    nh, hd = cfg.heads, cfg.head_dim
    dev, dt = X.device, X.dtype
    q, k, v = (torch.empty(R, nh, hd, dtype=dt, device=dev) for _ in range(3))
    for r0, r1 in _chunks(R, row_chunk):
        y = _linear(blk.attn.to_qkv, _rms(X[r0:r1], blk.norm1.weight)).view(r1 - r0, nh, 3 * hd)
        qq, kk, vv = y.chunk(3, dim=-1)                # per head [q; k; v]
        q[r0:r1], k[r0:r1], v[r0:r1] = _rms(qq, None), _rms(kk, None), vv
    cos, sin = rope_dev[shifted]
    for r0, r1 in _chunks(geo.N, row_chunk):
        q[r0:r1] = _apply_rope(q[r0:r1], cos[r0:r1], sin[r0:r1])
        k[r0:r1] = _apply_rope(k[r0:r1], cos[r0:r1], sin[r0:r1])
    out = torch.empty(R, nh, hd, dtype=dt, device=dev)
    for qi_all, ki_all in geo.windows[shifted]:
        for c0, c1 in _chunks(qi_all.shape[0], win_batch):
            qi, ki = qi_all[c0:c1].to(dev), ki_all[c0:c1].to(dev)
            o = F.scaled_dot_product_attention(q[qi].transpose(1, 2), k[ki].transpose(1, 2), v[ki].transpose(1, 2))
            out[qi] = o.transpose(1, 2)
    for gq, gk in geo.glob:
        gq, gk = gq.to(dev), gk.to(dev)
        o = F.scaled_dot_product_attention(q[gq].transpose(0, 1)[None], k[gk].transpose(0, 1)[None],
                                           v[gk].transpose(0, 1)[None])
        out[gq] = o[0].transpose(0, 1)
    del q, k, v
    s1, s2 = blk.scale1.to(dt), blk.scale2.to(dt)
    for r0, r1 in _chunks(R, row_chunk):
        X[r0:r1] = torch.addcmul(X[r0:r1], _linear(blk.attn.to_out, out[r0:r1].reshape(r1 - r0, D)), s1)
    del out
    for r0, r1 in _chunks(R, row_chunk):
        gate, val = _linear(blk.ff.w1, _rms(X[r0:r1], blk.norm2.weight)).chunk(2, dim=-1)
        X[r0:r1] = torch.addcmul(X[r0:r1], _linear(blk.ff.w2, F.silu(gate) * val), s2)


@torch.no_grad()
def run_refiner(head: DWDHead, feats: torch.Tensor, grid: tuple[int, int, int], *, keep_blocks=None,
                row_chunk: int = 8192, win_batch: int = 16, out_device="cpu") -> torch.Tensor:
    """Normalized features [t * gh * gw, 5376] on a (t, gh, gw) DiT token grid -> float32 (float64 for a float64 stream) pixels
    [F, 3, 16H, 16W] (Sol's normalized RGB, before the colour codec) of native blocks ``keep_blocks``
    ([b0, b1), default all), on ``out_device``. The residual stream has ``feats``' dtype."""
    cfg, ref = head.cfg, head.dwd_pixel_refiner
    t, gh, gw = (int(v) for v in grid)
    eh, ew = cfg.expansion
    H, W, D, P = gh * eh, gw * ew, cfg.dim, cfg.patch
    if feats.shape[0] != t * gh * gw:
        raise ValueError(f"DWD: features {list(feats.shape)} do not match the grid {grid}")
    dev, dt = feats.device, feats.dtype
    geo = build_geometry(t, H, W, cfg.rope_dim)
    X = torch.empty(geo.R, D, dtype=dt, device=dev)
    hw, HW = gh * gw, H * W
    step = max(1, row_chunk // hw)
    for t0, t1 in _chunks(t, step):
        y = _linear(ref.feature_expand, feats[t0 * hw:t1 * hw])
        X[t0 * HW:t1 * HW] = y.view(t1 - t0, gh, gw, eh, ew, D).permute(0, 1, 3, 2, 4, 5).reshape(-1, D)
    g = torch.cat([ref.register_tokens[0].to(dt), torch.zeros(1, D, dtype=dt, device=dev)])
    X[geo.N:] = g.repeat(geo.nb, 1)
    rope_dev = {k: (c.to(dev), s.to(dev)) for k, (c, s) in geo.rope.items()}
    for li, blk in enumerate(ref.blocks):
        _layer(blk, X, geo, li % 2 == 1, cfg, row_chunk, win_batch, rope_dev)
    b0, b1 = (0, geo.nb) if keep_blocks is None else keep_blocks
    oc, pt, ft = cfg.out_ch, cfg.patch_t, cfg.first_t
    nw, bw = ref.norm_out.weight.to(dt), ref.norm_out.bias.to(dt)
    odt = dt if dt in (torch.float32, torch.float64) else torch.float32
    pieces = []
    for b in range(b0, b1):
        r0, r1 = geo.block_rows[b]
        n = (r1 - r0) // HW
        y = F.layer_norm(X[r0:r1], (D,), nw, bw, EPS)
        first = _linear(ref.first_frame_proj_out, y[:HW]).view(H, W, oc, ft, P, P)
        frames = [first.permute(3, 2, 0, 4, 1, 5).reshape(ft, oc, H * P, W * P).to(out_device, odt)]
        if n > 1:
            rest = _linear(ref.proj_out, y[HW:]).view(n - 1, H, W, oc, pt, P, P)
            frames.append(rest.permute(0, 4, 3, 1, 5, 2, 6).reshape((n - 1) * pt, oc, H * P, W * P).to(out_device, odt))
        pieces.append(torch.cat(frames))
        del y
    return torch.cat(pieces)


def plan_chunks(t: int, content_blocks: int | None, margin_blocks: int) -> list[dict]:
    """Chunks: kept native blocks [k0, k1) with context [c0, c1) of ``margin_blocks`` on each side
    (none past the clip's true ends). ``content_blocks`` None = one chunk, the whole clip."""
    nb = -(-int(t) // BLOCK)
    if content_blocks is None:
        return [{"keep": (0, nb), "context": (0, nb)}]
    out = []
    for k0 in range(0, nb, int(content_blocks)):
        k1 = min(k0 + int(content_blocks), nb)
        out.append({"keep": (k0, k1), "context": (max(0, k0 - margin_blocks), min(nb, k1 + margin_blocks))})
    return out


@torch.no_grad()
def decode(head: DWDHead, feats: torch.Tensor, grid, *, content_blocks=None, margin_blocks: int = 4,
           row_chunk: int = 8192, win_batch: int = 16, measure_peak: bool = False):
    """Normalized features -> (float32 CPU pixels [F, 3, 16H, 16W], per-chunk records).

    Each chunk starts at a native block head (latent index a multiple of 5) and is decoded with
    its own geometry; only its kept blocks are projected and appended. ``measure_peak`` resets the
    CUDA peak counter per chunk (standalone use only: inside the pipeline's decode phase it would
    disturb the product's own peak)."""
    t, gh, gw = (int(v) for v in grid)
    hw = gh * gw
    pieces, records = [], []
    for ch in plan_chunks(t, content_blocks, margin_blocks):
        (k0, k1), (c0, c1) = ch["keep"], ch["context"]
        l0, l1 = c0 * BLOCK, min(c1 * BLOCK, t)
        cuda = feats.is_cuda
        if cuda:
            torch.cuda.synchronize()
            if measure_peak:
                torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        px = run_refiner(head, feats[l0 * hw:l1 * hw], (l1 - l0, gh, gw), keep_blocks=(k0 - c0, k1 - c0),
                         row_chunk=row_chunk, win_batch=win_batch)
        if cuda:
            torch.cuda.synchronize()
        rec = {"keep_blocks": [k0, k1], "context_blocks": [c0, c1], "latents": [l0, l1],
               "frames": int(px.shape[0]), "seconds": round(time.perf_counter() - started, 3)}
        if cuda:
            rec["peak_alloc_mib"] = int(round(torch.cuda.max_memory_allocated() / 2**20))
            rec["peak_reserved_mib"] = int(round(torch.cuda.max_memory_reserved() / 2**20))
        records.append(rec)
        pieces.append(px)
    return torch.cat(pieces), records


def to_uint8(pixels: torch.Tensor, frame_chunk: int = 16) -> tuple[torch.Tensor, dict]:
    """float32 [F, 3, H, W] (Sol's normalized RGB) -> CPU uint8 [F, H, W, 3] and value statistics."""
    mean = torch.tensor(COLOR_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(COLOR_STD, dtype=torch.float32).view(1, 3, 1, 1)
    out = torch.empty((pixels.shape[0], pixels.shape[2], pixels.shape[3], 3), dtype=torch.uint8)
    lo, hi, outside, total = math.inf, -math.inf, 0, 0
    for f0, f1 in _chunks(pixels.shape[0], frame_chunk):
        p = pixels[f0:f1].float().cpu()
        lo, hi = min(lo, float(p.min())), max(hi, float(p.max()))
        c = p * std + mean                                  # colour codec: [-1, 1]
        outside += int(((c < -1) | (c > 1)).sum())
        total += c.numel()
        x = ((c + 1.0) * 0.5).clamp_(0.0, 1.0).permute(0, 2, 3, 1)
        out[f0:f1] = (x * 255.0).round().to(torch.uint8)
    return out, {"raw_min": lo, "raw_max": hi, "codec_outside_pm1_ratio": outside / max(total, 1)}


# --------------------------------------------------------------------------- #
# CPU check of the chunked decode (dry-run item 8)
# --------------------------------------------------------------------------- #

def chunking_self_test(dtype=torch.float64, seed: int = 20261122) -> dict:
    """Random-weight float head on the CPU: the whole clip vs chunked decodes.

    14 native blocks (67 latents), a 12 x 12 DiT grid (24 x 24 Sol tokens: 2 windows per axis),
    the real window / block / shift constants and 8 layers; tiny widths and a 2-px patch."""
    cfg = HeadConfig(in_channels=12, t_dim=8, dim=16, heads=2, head_dim=8, ffn_mult=4, num_layers=8,
                     expansion=(2, 2), patch=2, patch_t=4, out_ch=3)
    g = torch.Generator().manual_seed(seed)
    head = DWDHead(cfg, make=float_linear(dtype), dtype=dtype)
    with torch.no_grad():
        for p in head.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=dtype) / math.sqrt(max(p.shape[-1], 1)))
        for blk in head.dwd_pixel_refiner.blocks:      # norm weights / layer scales near their usual size
            for p in (blk.norm1.weight, blk.norm2.weight):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g, dtype=dtype))
            for p in (blk.scale1, blk.scale2):
                p.copy_(0.5 + 0.1 * torch.randn(p.shape, generator=g, dtype=dtype))
    t, gh, gw = 67, 12, 12
    hidden = torch.randn(t * gh * gw, cfg.in_channels, generator=g, dtype=dtype)
    t_emb = torch.randn(3, cfg.t_dim, generator=g, dtype=dtype)
    time_index = torch.randint(0, 3, (t * gh * gw,), generator=g)
    feats = normalize_features(head, hidden, t_emb, time_index)
    started = time.perf_counter()
    whole, _ = decode(head, feats, (t, gh, gw), row_chunk=4096, win_batch=64)
    rec = {"dtype": str(dtype).replace("torch.", ""), "latents": t, "native_blocks": -(-t // BLOCK),
           "grid": [t, gh, gw], "sol_tokens_hw": [gh * 2, gw * 2], "frames": int(whole.shape[0]),
           "frames_expected": frames_for_latents(t), "whole_seconds": round(time.perf_counter() - started, 2),
           "variants": {}}
    for content, margin in ((5, 4), (1, 4), (5, 3)):
        px, chunks = decode(head, feats, (t, gh, gw), content_blocks=content, margin_blocks=margin,
                            row_chunk=4096, win_batch=64)
        d = (px - whole).abs()
        rec["variants"][f"content{content}_margin{margin}"] = {
            "chunks": len(chunks), "frames": int(px.shape[0]), "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()), "pixel_scale_max_abs": float(whole.abs().max())}
    tol = 1e-9 if dtype == torch.float64 else 1e-4
    v = rec["variants"]
    rec["tolerance"] = tol
    rec["margin4_match"] = all(v[k]["max_abs_diff"] <= tol for k in ("content5_margin4", "content1_margin4"))
    # informative only (the reach of 4 shifted layers is exactly 4 blocks; how much leaks past 3 is weight-dependent)
    rec["margin3_differs"] = v["content5_margin3"]["max_abs_diff"] > tol
    rec["passed"] = bool(rec["margin4_match"] and rec["frames"] == rec["frames_expected"]
                         and all(x["frames"] == rec["frames"] for x in v.values()))
    return rec
