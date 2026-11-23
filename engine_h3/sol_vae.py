r"""Sol-H3-VAE: the video decoder distilled to 24 layers, for a ``sol_vae`` job's decode.

Written from a reading of NVIDIA's reference implementation, Sol-H3-VAE
(Hugging Face ``Efficient-Large-Model/Sol-H3-VAE``, ``code/sol_h3_vae/``; the
model card states "The code is released under Apache-2.0"): the structure of
the decoder (windows, shifts, global tokens, window-local rope coordinates, the
frames dropped from each block's first latent). Copyright (c) NVIDIA
CORPORATION. Licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Changes: the reference pads every
window and masks the padding, this module gathers each window's tokens (no
padding, no mask) and attends windows of the same shape together; the clip is
walked native block by native block inside each layer (below); the arithmetic
is fp16 throughout with comfy-kitchen's functions (``int8_linear`` with its
fused arguments for the int8 Linears, ``rms_rope_split_half_`` for the q / k
norm and rope), the order the experiment plan's appendix describes for the
Comfy-Org weights. No ComfyUI (GPL-3.0) or Unsloth (AGPL-3.0) code was read or
copied. The computation is the experiment's configuration B
(``outputs/h3-sol-spike-2026-11-23/sol_decoder.py``: the INT8 ConvRot file,
the float attention, fp16), moved here without its other configurations.

Product parts used as they are: ``engine_h3.dwd`` (``_Block``, ``_spans``,
``_rope_tables``, ``block_frames``, the int8 / float Linear makers; the same
rules: width 2048, 32 heads of 64, FFN x4, time normalized by 5),
``engine_h3.video_vae`` (the latent statistics of the file's metadata, the
ImageNet constants), ``engine_h3.quant_linear.H3Int8Linear`` (only as the
container of an int8 layer's tensors: its ``forward`` is never called; the
int8 matmul is ``comfy_kitchen.int8_linear`` in fp16, called here).

What it computes (one clip, batch 1):

1. ``z * std + mean`` in fp16 (the file's 24 + 24 statistics rounded to
   fp16), ``post_quant_conv`` (1x1x1, a Linear over the channels) and
   ``x_embedder`` (24 -> 2048): one token per latent voxel, rows in (t, y, x).
2. Per native block (5 latents; the last may be shorter) 5 global tokens: the
   4 register tokens and one zero token.
3. 24 layers; odd layers shift the windows by 2 latents in time and 8 tokens
   in space. A 5-latent x 16 x 16 window attends to its tokens and to the
   global tokens of the 1 or 2 blocks it overlaps; a block's global tokens
   attend to the block's tokens and themselves. q / k: per-head RMSNorm without
   affine (eps 1e-5), split-half rope on the first 48 of 64 channels (token
   rows only). ``x += to_out(o) * scale1``; ``x += w2(silu(gate) * up) * scale2``
   with ``[gate | up] = w1(rms(x) * norm2)``.
4. Per latent: LayerNorm, ``proj_out`` (2048 -> 3 x 4 x 16 x 16), float32,
   the first 3 of the 4 frames of each block's first latent dropped (5n + 2
   latents -> 17n + 5 frames), ImageNet revert, clamp [0, 1], x255, round,
   uint8 into a CPU tensor allocated up front.

Order of the work (memory): the layers are the outer loop; inside a layer the
temporal windows go in time order. A block's q, k, v are computed from the
residual stream the first time a window needs it, its attention output is
collected until every token of the block has been a query, then ``to_out`` +
residual and the FFN run over the block's rows and its q, k, v are dropped:
at most 2 blocks' worth live at once. The whole clip keeps only the residual
stream on the device, so the memory grows with the clip's latent tokens
(:data:`MAX_LATENT_TOKENS` bounds what is decoded here).

Built for one decode and dropped after it (never kept resident), like the
pi-H3 head. Everything runs inside ``torch.no_grad()``.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine_h3 import dwd as D
from engine_h3 import video_vae as VV
from engine_h3.quant_linear import H3Int8Linear

BLOCK = 5            # latents per native block (17 frames)
WINDOW = 16          # spatial window, tokens (256 px)
T_SHIFT = 2          # temporal shift of the odd layers (the spatial shift is WINDOW // 2)
NUM_REGISTER = 4
NUM_GLOBAL = NUM_REGISTER + 1   # 4 register tokens + 1 zero token per native block
EPS = 1e-5
PATCH, PATCH_T, OUT_CH = 16, 4, 3
TRIM = 3             # frames dropped from the first latent of every block
#: Rows per piece of the per-row steps (norm + Linear, residual, FFN).
ROW_CHUNK = 8192
#: Windows of the same token count per attention call.
WIN_BATCH = 64
LATENT_CHANNELS = 24
#: The largest clip decoded here, in latent tokens (latent frames x latent
#: height x latent width, the decoder's input after a chain's join). A larger
#: clip is decoded by the stock video decoder (``sol_vae_used`` "on->off"):
#: the residual stream grows with the clip, the stock decoder's does not.
#: Set to hold the three chains the experiment measured within 16 GB
#: (1280 x 768 x 940 frames = 1,063,680; 1920 x 1088 x 685 = 1,648,320;
#: 2560 x 1472 x 362 = 1,575,040 tokens; the measurement is the verification
#: log's). A clip of exactly this size still uses Sol.
MAX_LATENT_TOKENS = 1_650_000

_DECODER_PREFIXES = ("decoder.", "post_quant_conv.")
_INT8_LEAVES = ("attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2")

#: The reasons :func:`file_fallback` / :func:`size_fallback` return.
FALLBACK_FILE_MISSING = "file_missing"
FALLBACK_TOO_LARGE = "too_large"


@dataclass(frozen=True)
class SolDims:
    """The widths of the decoder (``engine_h3.dwd._Block`` reads ``dim`` and ``ffn_mult``)."""

    dim: int = 2048
    heads: int = 32
    head_dim: int = 64
    ffn_mult: int = 4
    num_layers: int = 24
    latent_channels: int = LATENT_CHANNELS
    num_register: int = NUM_REGISTER

    @property
    def rope_dim(self) -> int:
        return int(self.head_dim * 0.75)


#: The released decoder's widths (the INT8 ConvRot file).
SOL_DIMS = SolDims()


# --------------------------------------------------------------------------- #
# the fallback decisions
# --------------------------------------------------------------------------- #

def file_fallback(path: str) -> str | None:
    """Why a ``sol_vae`` job cannot use the file at ``path`` (decided once, at the job's
    start): :data:`FALLBACK_FILE_MISSING` for an empty path or no file there, else None
    (the file is then checked with :func:`file_problems`)."""
    if not str(path or "").strip() or not os.path.isfile(str(path)):
        return FALLBACK_FILE_MISSING
    return None


def latent_tokens(shape) -> int:
    """The decoder's input size: ``T' x h x w`` of a ``[1, 24, T', h, w]`` latent shape."""
    t, h, w = (int(v) for v in tuple(shape)[-3:])
    return t * h * w


def size_fallback(shape) -> str | None:
    """Why the latents of ``shape`` are not decoded here (decided at the decode's
    entry, on the latents themselves): :data:`FALLBACK_TOO_LARGE` above
    :data:`MAX_LATENT_TOKENS`, else None."""
    if latent_tokens(shape) > MAX_LATENT_TOKENS:
        return FALLBACK_TOO_LARGE
    return None


def kitchen_problem() -> str | None:
    """comfy-kitchen's CUDA backend must carry ``rms_rope_split_half_`` (the q / k norm
    and rope on the GPU; ``int8_linear`` is required at load). None when it does."""
    import comfy_kitchen  # noqa: PLC0415

    cuda = comfy_kitchen.list_backends().get("cuda") or {}
    if "rms_rope_split_half_" not in (cuda.get("capabilities") or []):
        return "comfy-kitchen の CUDA バックエンドに rms_rope_split_half_ がありません"
    return None


# --------------------------------------------------------------------------- #
# modules (state-dict keys = the Sol file's ``decoder.*`` / ``post_quant_conv.*``)
# --------------------------------------------------------------------------- #

class _Decoder(nn.Module):
    def __init__(self, dims: SolDims, make, dtype) -> None:
        super().__init__()
        d = dims.dim
        self.x_embedder = nn.Linear(dims.latent_channels, d, bias=True, dtype=dtype)
        self.register_tokens = nn.Parameter(torch.empty(1, dims.num_register, d, dtype=dtype), requires_grad=False)
        self.blocks = nn.ModuleList(D._Block(dims, make, i, dtype) for i in range(dims.num_layers))
        self.norm_out = nn.LayerNorm(d, eps=EPS, dtype=dtype)
        self.proj_out = nn.Linear(d, OUT_CH * PATCH_T * PATCH * PATCH, bias=True, dtype=dtype)


class SolModel(nn.Module):
    def __init__(self, dims: SolDims = SOL_DIMS, make=None, dtype=torch.float16) -> None:
        super().__init__()
        self.dims = dims
        c = dims.latent_channels
        self.post_quant_conv = nn.Conv3d(c, c, kernel_size=1, bias=True, dtype=dtype)
        self.decoder = _Decoder(dims, make or D.float_linear(dtype), dtype)
        self.requires_grad_(False)


def sol_key_map(name: str) -> str | None:
    """File tensor name -> :class:`SolModel` key (None = not the decoder's: encoder, statistics)."""
    return name if name.startswith(_DECODER_PREFIXES) else None


# --------------------------------------------------------------------------- #
# the file
# --------------------------------------------------------------------------- #

@dataclass
class SolFileInfo:
    header: object
    quant: dict            # "blocks.N.attn.to_qkv" -> h3_format.QuantSpec
    mean: list
    std: list
    problems: list = field(default_factory=list)


def _few(items, n: int = 3) -> str:
    items = sorted(items)
    head = "・".join(str(x) for x in items[:n])
    return head + (f" ほか {len(items) - n} 個" if len(items) > n else "")


def _meta_model(quant: dict, dims: SolDims) -> SolModel:
    with torch.device("meta"):
        return SolModel(dims, D.int8_linear(quant, bias_dtype=torch.float16), torch.float16)


def inspect_file(path, *, dims: SolDims = SOL_DIMS) -> SolFileInfo:
    """The header only: the layers counted from the keys, the int8 markers, the latent
    statistics and the decoder's tensor names and shapes against the model. Every
    mismatch is a sentence of ``problems`` (nothing raised for a file that does not fit)."""
    import h3_format  # noqa: PLC0415
    import sft_quant_format  # noqa: PLC0415

    try:
        header = sft_quant_format.read_header(str(path))
    except (sft_quant_format.QuantFormatError, OSError) as exc:
        return SolFileInfo(header=None, quant={}, mean=[], std=[],
                           problems=[f"ヘッダーを読めません（{exc}）"])
    t = header.tensors
    names = set(t)
    problems: list[str] = []
    layers = {int(m.group(1)) for k in names if (m := re.match(r"^decoder\.blocks\.(\d+)\.", k))}
    want_layers = set(range(dims.num_layers))
    if layers != want_layers:
        problems.append(
            f"復号器の層が {len(layers)} 層で、層 0〜{dims.num_layers - 1} と合いません"
            + (f"（無い層: {_few(want_layers - layers)}）" if want_layers - layers else "")
            + (f"（余分な層: {_few(layers - want_layers)}）" if layers - want_layers else "")
        )
    want_quant = {f"blocks.{i}.{leaf}" for i in sorted(layers & want_layers) for leaf in _INT8_LEAVES}
    markers = {k[len("decoder."):-len(".comfy_quant")] for k in names
               if k.startswith("decoder.") and k.endswith(".comfy_quant")}
    if markers != want_quant:
        problems.append(
            f"int8 の量子化の印が {len(markers)} 層にあり、各層の 4 つの線形層（{len(want_quant)} 層）と合いません"
            + (f"（無い層: {_few(want_quant - markers)}）" if want_quant - markers else "")
            + (f"（余分な層: {_few(markers - want_quant)}）" if markers - want_quant else "")
        )
    quant: dict = {}
    for layer in sorted(markers & want_quant):
        key = f"decoder.{layer}.comfy_quant"
        try:
            raw = sft_quant_format.read_tensor_bytes(str(path), header, key)
            quant[layer] = h3_format.parse_comfy_quant(raw, accepted=(h3_format.INT8,))
        except (h3_format.H3FormatError, sft_quant_format.QuantFormatError, OSError) as exc:
            problems.append(f"{key} を int8 の量子化の印として読めません（{exc}）")
        if f"decoder.{layer}.weight_scale" not in names:
            problems.append(f"decoder.{layer}.weight_scale がありません")
    mean: list = []
    std: list = []
    try:
        mean, std = VV.latent_stats_from_metadata(header.metadata, VV.METADATA_KEY, LATENT_CHANNELS)
    except ValueError as exc:
        problems.append(str(exc))
    if not problems:
        # The names, shapes and dtypes of the decoder's tensors against the model the
        # markers build (on the meta device).
        model = _meta_model(quant, dims).state_dict()
        file_keys = {k for k in names if sol_key_map(k) is not None and not k.endswith(".comfy_quant")}
        if file_keys - set(model):
            problems.append(f"復号器に無いテンソルが {len(file_keys - set(model))} 個あります"
                            f"（{_few(file_keys - set(model))}）")
        if set(model) - file_keys:
            problems.append(f"復号器のテンソルが {len(set(model) - file_keys)} 個足りません"
                            f"（{_few(set(model) - file_keys)}）")
        want_dtype = {torch.int8: "I8", torch.float32: "F32", torch.float16: "F16"}
        for k in sorted(file_keys & set(model)):
            info, m = t[k], model[k]
            if tuple(info.shape) != tuple(m.shape) or info.dtype != want_dtype.get(m.dtype):
                problems.append(f"{k} が {info.dtype} {list(info.shape)} です"
                                f"（{want_dtype.get(m.dtype)} {list(m.shape)} のはず）")
    return SolFileInfo(header=header, quant=quant, mean=mean, std=std, problems=problems)


def file_problems(path, *, dims: SolDims = SOL_DIMS) -> list[str]:
    """The file at ``path`` against the released decoder's layout; ``[]`` when it fits."""
    return inspect_file(path, dims=dims).problems


# --------------------------------------------------------------------------- #
# q / k normalization + rope
# --------------------------------------------------------------------------- #

def rope_cos_sin(H: int, W: int, shifted: bool, rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos / sin float32 ``[BLOCK * H * W, rope_dim // 2]`` of one full native block's token rows.

    ``engine_h3.dwd._rope_tables`` over a 5-latent clip: its time coordinate is ``(t + st) % 5``, so
    every block's rows (local time 0..4) get the same table and a shorter last block uses its prefix.
    The two rotary halves share the angles, so only the first half is kept."""
    s, st = (WINDOW // 2, T_SHIFT) if shifted else (0, 0)
    cos, sin = D._rope_tables(BLOCK, H, W, s, st, rope_dim)
    half = rope_dim // 2
    return cos[:, 0, :half].contiguous(), sin[:, 0, :half].contiguous()


def kitchen_freqs(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``[1, rows, 1, half, 2, 2]`` float32: per pair ``[[cos, -sin], [sin, cos]]`` (comfy-kitchen's
    split-half layout: ``out_first = f00 x_first + f01 x_second``, ``out_second = f10 x_first + f11 x_second``)."""
    f = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
    return f[None, :, None].contiguous()


def rms_rope_torch(x: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None) -> torch.Tensor:
    """The same formula in torch (the CPU path): RMSNorm (no affine, eps 1e-5) over the head dim in
    float32, rounded to ``x``'s dtype, then the first ``2 * half`` channels rotated split-half in
    float32 (``x1 cos - x2 sin``, ``x2 cos + x1 sin``) and cast back (the order the CUDA kernel was
    measured to follow in the experiment). ``x`` [n, heads, head_dim]; ``cos`` / ``sin`` [n, half]
    float32 or None (no rotation). A float64 ``x`` stays float64."""
    hi = torch.float64 if x.dtype == torch.float64 else torch.float32
    xf = F.rms_norm(x.to(hi), (x.shape[-1],), eps=EPS).to(x.dtype).to(hi)
    if cos is not None:
        half = cos.shape[-1]
        c, s = cos[:, None, :].to(xf), sin[:, None, :].to(xf)
        x1, x2, rest = xf[..., :half], xf[..., half:2 * half], xf[..., 2 * half:]
        xf = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s, rest), dim=-1)
    return xf.to(x.dtype)


# --------------------------------------------------------------------------- #
# the decoder
# --------------------------------------------------------------------------- #

def _ck():
    import comfy_kitchen  # noqa: PLC0415

    return comfy_kitchen


@dataclass
class _Section:
    """One native block of one layer: its q / k / v (k, v with the block's global rows appended) and
    the attention output collected for its token rows."""

    b: int
    t0: int
    r0: int
    n: int
    q: torch.Tensor          # [n, heads, head_dim]
    k: torch.Tensor          # [n + S, heads, head_dim]
    v: torch.Tensor          # [n + S, heads, head_dim]
    og: torch.Tensor         # [S, dim] the global tokens' attention output
    o: torch.Tensor          # [n, dim] filled by the windows
    written: int = 0


def _chunks(n: int, size: int):
    return [(i, min(i + size, n)) for i in range(0, n, size)]


class SolDecoder:
    """Normalized latents -> CPU uint8 frames (the video decoder's contract). Build with
    :meth:`from_file` (the INT8 file, fp16); :meth:`from_model` takes a built model (tests)."""

    def __init__(self, model: SolModel, mean, std, device, load_report: dict | None = None,
                 *, rope_impl: str | None = None) -> None:
        self.model = model
        self.dims = model.dims
        self.device = torch.device(device)
        #: The compute dtype is the model's: fp16 for the file; the CPU tests' models may be
        #: float32 / float64.
        self.dtype = model.decoder.x_embedder.weight.dtype
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.load_report = load_report or {}
        #: "kitchen" (comfy_kitchen.rms_rope_split_half_, fp16 on CUDA) or "torch" (rms_rope_torch).
        self.rope_impl = rope_impl or ("kitchen" if self.device.type == "cuda" else "torch")
        self.last_stats: dict = {}

    @classmethod
    def from_file(cls, path, *, device, dims: SolDims = SOL_DIMS) -> "SolDecoder":
        """Read the decoder's tensors of ``path`` as stored (int8 + F32 scales, F16 elsewhere) onto
        ``device``; ``RuntimeError`` for a file that does not fit (:func:`inspect_file`)."""
        from engine_h3.sft_load import load_streaming  # noqa: PLC0415

        started = time.perf_counter()
        info = inspect_file(path, dims=dims)
        if info.problems:
            raise RuntimeError(f"Sol-H3-VAE のファイル（{os.path.basename(str(path))}）: " + "；".join(info.problems))
        model = _meta_model(info.quant, dims)
        want = len(model.state_dict())
        rep = load_streaming(model, str(path), info.header, key_map=sol_key_map)
        if rep.loaded != want:
            raise RuntimeError(f"Sol-H3-VAE: {rep.loaded} 個のテンソルを読みました（{want} 個のはず）")
        model.requires_grad_(False).eval()
        model.to(torch.device(device))
        report = {"file": os.path.basename(str(path)), "int8_layers": len(info.quant), "loaded_tensors": int(rep.loaded),
                  "loaded_mib": round(rep.bytes / 2**20, 1), "seconds": round(time.perf_counter() - started, 2)}
        return cls(model, info.mean, info.std, device, report)

    @classmethod
    def from_model(cls, model: SolModel, *, mean, std, device, rope_impl=None) -> "SolDecoder":
        return cls(model, mean, std, device, {"from": "model"}, rope_impl=rope_impl)

    # -- per-row steps ------------------------------------------------------

    def _int8(self, layer: H3Int8Linear, x: torch.Tensor, **fused) -> torch.Tensor:
        return _ck().int8_linear(
            x.contiguous(), layer.weight, layer.weight_scale, layer.bias, out_dtype=torch.float16,
            convrot=bool(layer.spec.convrot), convrot_groupsize=int(layer.spec.convrot_groupsize), **fused)

    def _qkv(self, blk, x: torch.Tensor) -> torch.Tensor:
        """``to_qkv(rms_norm(x) * norm1)``: [m, dim] -> [m, 3 * dim] (per head [q | k | v])."""
        lin = blk.attn.to_qkv
        if isinstance(lin, H3Int8Linear):
            return self._int8(lin, x, input_act="rms_norm", input_act_weight=blk.norm1.weight, input_act_eps=EPS)
        h = F.rms_norm(x, (x.shape[-1],), weight=blk.norm1.weight, eps=EPS)
        return F.linear(h, lin.weight, lin.bias)

    def _out_residual(self, blk, o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``x + to_out(o) * scale1``."""
        lin = blk.attn.to_out
        if isinstance(lin, H3Int8Linear):
            return self._int8(lin, o, residual=x, residual_scale=blk.scale1)
        return torch.addcmul(x, F.linear(o, lin.weight, lin.bias), blk.scale1)

    def _ffn_residual(self, blk, x: torch.Tensor) -> torch.Tensor:
        """``x + w2(silu(gate) * up) * scale2`` with ``[gate | up] = w1(rms_norm(x) * norm2)``."""
        w1, w2 = blk.ff.w1, blk.ff.w2
        if isinstance(w1, H3Int8Linear):
            a = self._int8(w1, x, input_act="rms_norm", input_act_weight=blk.norm2.weight, input_act_eps=EPS)
        else:
            a = F.linear(F.rms_norm(x, (x.shape[-1],), weight=blk.norm2.weight, eps=EPS), w1.weight, w1.bias)
        if isinstance(w2, H3Int8Linear):
            return self._int8(w2, a, input_act="swiglu", residual=x, residual_scale=blk.scale2)
        gate, up = a.chunk(2, dim=-1)
        return torch.addcmul(x, F.linear(F.silu(gate) * up, w2.weight, w2.bias), blk.scale2)

    def _norm_rope(self, q: torch.Tensor, k: torch.Tensor, table) -> None:
        """q / k [n, heads, head_dim] in place: RMSNorm + rope (``table`` None = no rotation)."""
        if self.rope_impl == "kitchen":
            n = q.shape[0]
            freqs = self._identity_freqs(n) if table is None else table[0][:, :n]
            _ck().rms_rope_split_half_(q.view(1, *q.shape), k.view(1, *k.shape), freqs, self._q_scale, None,
                                       EPS, self.dims.rope_dim)
            return
        if table is None:
            cos = sin = None
        else:
            cos, sin = table[1][:q.shape[0]], table[2][:q.shape[0]]
        q.copy_(rms_rope_torch(q, cos, sin))
        k.copy_(rms_rope_torch(k, cos, sin))

    def _identity_freqs(self, n: int) -> torch.Tensor:
        half = self.dims.rope_dim // 2
        one = torch.ones(n, half, dtype=torch.float32, device=self.device)
        return kitchen_freqs(one, torch.zeros_like(one)).to(self.dtype)

    # -- attention ----------------------------------------------------------

    def _window_attention(self, q, k, v) -> torch.Tensor:
        """[nW, L, h, d], [nW, Lk, h, d] -> [nW, L, h, d]; NaN / Inf replaced by finite values."""
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return torch.nan_to_num(o.transpose(1, 2))

    # -- one block of one layer -----------------------------------------------

    def _section(self, blk, X, G, b: int, table) -> _Section:
        """q, k, v of block ``b``'s token rows and global rows, read from the stream before any of its rows
        changes in this layer; the global tokens' attention is computed here (it needs only this block)."""
        g = self._geom
        hw, nh, hd, S, dim = g["hw"], self.dims.heads, self.dims.head_dim, NUM_GLOBAL, self.dims.dim
        t0, t1 = b * BLOCK, min((b + 1) * BLOCK, g["T"])
        r0, n = t0 * hw, (t1 - t0) * hw
        dev, dt = X.device, X.dtype
        q = torch.empty(n, nh, hd, dtype=dt, device=dev)
        k = torch.empty(n + S, nh, hd, dtype=dt, device=dev)
        v = torch.empty(n + S, nh, hd, dtype=dt, device=dev)
        for c0, c1 in _chunks(n, ROW_CHUNK):
            y = self._qkv(blk, X[r0 + c0:r0 + c1]).view(c1 - c0, nh, 3 * hd)
            q[c0:c1], k[c0:c1], v[c0:c1] = y[..., :hd], y[..., hd:2 * hd], y[..., 2 * hd:]
            del y
        y = self._qkv(blk, G[b]).view(S, nh, 3 * hd)
        qg = y[..., :hd].contiguous()
        k[n:], v[n:] = y[..., hd:2 * hd], y[..., 2 * hd:]
        del y
        self._norm_rope(q, k[:n], table)
        kg = k[n:]                    # contiguous rows of k
        self._norm_rope(qg, kg, None)
        og = F.scaled_dot_product_attention(qg.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])
        og = og[0].transpose(0, 1).reshape(S, dim)
        o = torch.empty(n, dim, dtype=dt, device=dev)
        return _Section(b=b, t0=t0, r0=r0, n=n, q=q, k=k, v=v, og=og, o=o)

    def _windows(self, alive: dict, t0: int, t1: int, s: int) -> None:
        """Every spatial window of the temporal window [t0, t1), WIN_BATCH windows of one shape per call."""
        g = self._geom
        H, W, hw, dim = g["H"], g["W"], g["hw"], self.dims.dim
        parts = []                                   # (section, ta, tb): the window's rows in that block
        for b in range(t0 // BLOCK, (t1 - 1) // BLOCK + 1):
            sec = alive[b]
            parts.append((sec, max(t0, sec.t0), min(t1, sec.t0 + sec.n // hw)))
        groups: dict[tuple[int, int], list] = {}
        for y0, y1 in D._spans(H, s, WINDOW):
            for x0, x1 in D._spans(W, s, WINDOW):
                groups.setdefault((y1 - y0, x1 - x0), []).append((y0, y1, x0, x1))
        for shape, wins in groups.items():
            for c0, c1 in _chunks(len(wins), WIN_BATCH):
                batch = wins[c0:c1]
                idx = [self._window_rows(ta - sec.t0, tb - sec.t0, s, shape, c0, batch) for sec, ta, tb in parts]
                nw = len(batch)
                qw = torch.cat([sec.q[i] for (sec, _, _), i in zip(parts, idx)], dim=1)
                kw = torch.cat([sec.k[i] for (sec, _, _), i in zip(parts, idx)]
                               + [sec.k[sec.n:][None].expand(nw, -1, -1, -1) for sec, _, _ in parts], dim=1)
                vw = torch.cat([sec.v[i] for (sec, _, _), i in zip(parts, idx)]
                               + [sec.v[sec.n:][None].expand(nw, -1, -1, -1) for sec, _, _ in parts], dim=1)
                o = self._window_attention(qw, kw, vw)
                del qw, kw, vw
                off = 0
                for (sec, _, _), i in zip(parts, idx):
                    m = i.shape[1]
                    sec.o.index_copy_(0, i.reshape(-1), o[:, off:off + m].reshape(-1, dim))
                    sec.written += i.numel()
                    off += m
                del o, idx

    def _window_rows(self, la: int, lb: int, s: int, shape, c0: int, batch) -> torch.Tensor:
        """Device long ``[len(batch), (lb - la) * ly * lx]``: the block-local rows (local latents [la, lb))
        of each window of ``batch``. The same pattern recurs in every block, so it is kept per decode."""
        key = (la, lb, s, shape, c0)
        hit = self._idx_cache.get(key)
        if hit is None:
            g = self._geom
            hw, W = g["hw"], g["W"]
            tt = torch.arange(la, lb)
            rows = [(tt[:, None, None] * hw + torch.arange(y0, y1)[None, :, None] * W
                     + torch.arange(x0, x1)[None, None, :]).reshape(-1) for y0, y1, x0, x1 in batch]
            hit = self._idx_cache[key] = torch.stack(rows).to(self.device)
        return hit

    def _finish(self, blk, X, G, sec: _Section) -> None:
        """``to_out`` + residual and the FFN over the block's token rows and its global rows."""
        if sec.written != sec.n:
            raise RuntimeError(f"Sol-H3-VAE: block {sec.b} got attention output for {sec.written} of {sec.n} rows")
        for c0, c1 in _chunks(sec.n, ROW_CHUNK):
            r0, r1 = sec.r0 + c0, sec.r0 + c1
            X[r0:r1] = self._out_residual(blk, sec.o[c0:c1], X[r0:r1])
            X[r0:r1] = self._ffn_residual(blk, X[r0:r1])
        G[sec.b] = self._ffn_residual(blk, self._out_residual(blk, sec.og, G[sec.b]))

    def _layer(self, blk, X, G, li: int) -> None:
        g = self._geom
        shifted = li % 2 == 1
        s, st = (WINDOW // 2, T_SHIFT) if shifted else (0, 0)
        table = self._tables[shifted]
        alive: dict[int, _Section] = {}
        for t0, t1 in D._spans(g["T"], st, BLOCK):
            for b in range(t0 // BLOCK, (t1 - 1) // BLOCK + 1):
                if b not in alive:
                    alive[b] = self._section(blk, X, G, b, table)
            self.last_stats["max_sections_alive"] = max(self.last_stats["max_sections_alive"], len(alive))
            self._windows(alive, t0, t1, s)
            for b in sorted(alive):
                if min((b + 1) * BLOCK, g["T"]) <= t1:          # every row of block b has been a query
                    self._finish(blk, X, G, alive.pop(b))
        if alive:
            raise RuntimeError(f"Sol-H3-VAE: layer {li} left blocks {sorted(alive)} unfinished")

    # -- the whole clip -----------------------------------------------------

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, output: str = "uint8") -> torch.Tensor:
        """``latents`` [1, 24, T, h, w] (normalized, any device) -> CPU ``[F, 16h, 16w, 3]``:
        uint8 (``output="uint8"``, the decode's) or float32 in [0, 1] after the clamp
        (``output="float"``, for the tests' comparisons before the rounding).

        ``last_stats["nonfinite_pixels"]``: how many values of ``proj_out``'s output (before the
        clamp) were not finite -- counted only, nothing is changed or refused for them."""
        if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[1] != self.dims.latent_channels:
            raise ValueError(f"Sol-H3-VAE: [1, 24, T, h, w] を期待しましたが {tuple(latents.shape)} でした")
        if output not in ("uint8", "float"):
            raise ValueError(f"output={output!r}")
        mdl, dec = self.model, self.model.decoder
        dims, dt, dev = self.dims, self.dtype, self.device
        T, h, w = (int(v) for v in latents.shape[2:])
        hw, N, dim = h * w, T * h * w, dims.dim
        nb = -(-T // BLOCK)
        self._geom = {"T": T, "H": h, "W": w, "hw": hw, "nb": nb}
        self.last_stats = {"latent_tokens": N, "max_sections_alive": 0}
        # 1. latents -> tokens. ``z * std + mean`` in the compute dtype (fp16: the statistics
        # rounded to fp16, the order the experiment follows).
        mean = self.mean.to(device=dev, dtype=dt).view(-1, 1, 1, 1)
        std = self.std.to(device=dev, dtype=dt).view(-1, 1, 1, 1)
        raw = (latents[0].to(device=dev, dtype=dt) * std + mean).permute(1, 2, 3, 0).reshape(N, -1)
        pq = mdl.post_quant_conv
        X = torch.empty(N, dim, dtype=dt, device=dev)
        for r0, r1 in _chunks(N, 4 * ROW_CHUNK):
            z = F.linear(raw[r0:r1].to(dt), pq.weight.reshape(pq.weight.shape[0], -1), pq.bias)
            X[r0:r1] = F.linear(z, dec.x_embedder.weight, dec.x_embedder.bias)
        del raw
        glob = torch.cat([dec.register_tokens[0].to(dt), torch.zeros(1, dim, dtype=dt, device=dev)])
        G = glob[None].repeat(nb, 1, 1).contiguous()          # [nb, S, dim]
        # rope tables (one block's rows per shift): (kitchen freqs or None, cos, sin)
        self._tables = {}
        for shifted in (False, True):
            cos, sin = rope_cos_sin(h, w, shifted, dims.rope_dim)
            if dt == torch.float16:
                # fp16: cos / sin computed in float32, then rounded to fp16 (the table's dtype);
                # the torch formula gets the same rounded values
                cos, sin = cos.half().float(), sin.half().float()
            cos, sin = cos.to(dev), sin.to(dev)
            self._tables[shifted] = (kitchen_freqs(cos, sin).to(dt) if self.rope_impl == "kitchen" else None, cos, sin)
        self._q_scale = torch.ones(dims.head_dim, dtype=dt, device=dev)
        self._idx_cache = {}
        # 2. layers
        for li, blk in enumerate(dec.blocks):
            self._layer(blk, X, G, li)
        del G
        self._tables, self._idx_cache = {}, {}
        # 3. pixels, latent by latent
        frames = D.block_frames(T)
        F_total = sum(frames)
        odt = torch.float64 if dt == torch.float64 else torch.float32
        out = torch.empty((F_total, h * PATCH, w * PATCH, 3), dtype=torch.uint8 if output == "uint8" else odt)
        pm = torch.tensor(VV.IMAGENET_MEAN, dtype=odt, device=dev).view(1, 3, 1, 1)
        ps = torch.tensor(VV.IMAGENET_STD, dtype=odt, device=dev).view(1, 3, 1, 1)
        nonfinite = torch.zeros((), dtype=torch.long, device=dev)
        f0 = 0
        for t in range(T):
            y = F.layer_norm(X[t * hw:(t + 1) * hw], (dim,), dec.norm_out.weight, dec.norm_out.bias, EPS)
            p = F.linear(y, dec.proj_out.weight, dec.proj_out.bias).to(odt)
            del y
            nonfinite += (~torch.isfinite(p)).sum()
            px = p.view(h, w, OUT_CH, PATCH_T, PATCH, PATCH).permute(3, 2, 0, 4, 1, 5).reshape(
                PATCH_T, OUT_CH, h * PATCH, w * PATCH)
            del p
            if t % BLOCK == 0:
                px = px[TRIM:]
            x = (px * ps + pm).clamp_(0.0, 1.0).permute(0, 2, 3, 1)
            if output == "uint8":
                x = (x * 255.0).round().to(torch.uint8)
            out[f0:f0 + x.shape[0]] = x.cpu()
            f0 += x.shape[0]
            del px, x
        if f0 != F_total:
            raise RuntimeError(f"Sol-H3-VAE: {f0} frames written, {F_total} expected")
        del X
        self.last_stats["nonfinite_pixels"] = int(nonfinite)
        return out


__all__ = ["BLOCK", "FALLBACK_FILE_MISSING", "FALLBACK_TOO_LARGE", "MAX_LATENT_TOKENS", "SOL_DIMS",
           "SolDecoder", "SolDims", "SolModel", "file_fallback", "file_problems", "inspect_file",
           "kitchen_freqs", "kitchen_problem", "latent_tokens", "rms_rope_torch", "rope_cos_sin",
           "size_fallback", "sol_key_map"]
