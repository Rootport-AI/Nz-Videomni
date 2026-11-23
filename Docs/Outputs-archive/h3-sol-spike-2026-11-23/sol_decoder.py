r"""Experimental Sol-H3-VAE decoder for MiniMax H3 (the experiment of 2026-11-23; no product file changed).

Sol-H3-VAE is the released 36-layer video decoder distilled to 24 layers that decodes whole native
5-latent blocks with shifted window attention. This module turns normalized latents into CPU uint8
frames with the product's decoder contract (``engine_h3.video_vae.H3VideoDecoder.decode``):

    dec = SolDecoder.from_file(path, config=CONFIGS["A"], device="cuda")
    frames = dec.decode(latents)          # [1, 24, T, h, w] normalized -> uint8 [F, 16h, 16w, 3] on the CPU

Origin
------
* The structure (windows, shifts, global tokens, rope coordinates, frame trimming) follows NVIDIA's
  Apache-2.0 reference ``NonCausalViTDecoder`` (NVlabs/Sana,
  ``models/Open-H3-Regenerate-2K/open_h3/fast_vae/decoder.py``; a copy for the cross-check sits in
  ``ref_nvidia/``). Rewritten here, not copied: the reference pads every window and masks the padding,
  this module gathers the tokens of each window (no padding, no mask) and walks the clip block by block.
* The arithmetic follows ComfyUI's order as the experiment plan's appendix describes it in words
  (fp16 everywhere, comfy-kitchen's fused functions). No ComfyUI (GPL-3.0) or Unsloth (AGPL-3.0)
  code was read or copied for this module.
* Product parts used as they are (imported, not changed): ``engine_h3.dwd`` (``_Block``, ``_spans``,
  ``_rope_tables``, ``block_frames``, the int8 / float Linear makers), ``engine_h3.video_vae``
  (latent statistics from the metadata, ImageNet constants and the uint8 order),
  ``engine_h3.sft_load`` / ``h3_format`` / ``sft_quant_format`` (reading), ``engine_h3.quant_linear``
  (``H3Int8Linear`` as the container of an int8 layer; ``require_cuda_quant``).

What it computes (one clip, batch 1)
------------------------------------
1. ``z * std + mean`` in the compute dtype (the 24 values of the metadata ``minimax_h3_video_vae``,
   rounded to fp16 on the fp16 path), then
   ``post_quant_conv`` (1x1x1, as a Linear over the channels) and ``x_embedder`` (24 -> 2048) in the
   compute dtype: one token per latent voxel, rows in (t, y, x) order.
2. Per native block b (5 latents; the last one may be shorter) 5 global tokens: the 4 register tokens
   and one zero token.
3. 24 layers. Layer i: odd layers shift the windows by 2 latents in time and 8 tokens in space.
   * window attention: the tokens of a 5-latent x 16 x 16 window attend to the window's tokens and to
     the global tokens of the 1 or 2 native blocks the window overlaps;
   * global attention: the 5 global tokens of block b attend to every token of block b and to
     themselves (always the float attention);
   * q / k: per-head RMSNorm without affine (eps 1e-5); the token rows get split-half rope on the
     first 48 of 64 channels with window-local coordinates (time normalized by 5, space by 16;
     ``engine_h3.dwd._rope_tables``); the global rows are not rotated;
   * ``x += to_out(o) * scale1``; ``x += w2(silu(gate) * up) * scale2`` with ``[gate | up] = w1(rms(x))``.
4. Per latent: LayerNorm, ``proj_out`` (2048 -> 3 x 4 x 16 x 16), float32, the first 3 of the 4
   frames of each block's first latent dropped (5n + 2 latents -> 17n + 5 frames), ImageNet revert,
   clamp [0, 1], x255, round, uint8 (``video_vae.H3VideoDecoder._frames_uint8``'s order).

Order of the work (memory)
--------------------------
The layers are the outer loop; inside a layer the temporal windows are walked in time order. A
block's q, k, v (its tokens and its 5 global tokens) are computed from the residual stream the first
time a window needs that block, before any row of it is changed in this layer; its attention output
is collected until every token of the block has been a query, then ``to_out`` + residual and the FFN
run over the block's rows and the block's q, k, v and attention output are dropped. A window
overlaps at most 2 blocks, so at most 2 blocks' q, k, v and attention output live at once (counted in
``last_stats["max_sections_alive"]``). The whole clip keeps only the residual stream (tokens and
global tokens) on the device. The pixels are made latent by latent and written as uint8 into a CPU
tensor allocated up front.

The three switches (:class:`SolConfig`)
---------------------------------------
* ``weights``: ``fp16`` (the Comfy-Org FP16 file) / ``int8`` (the INT8 ConvRot file: the 4 Linears
  of each layer are int8, everything else F16).
* ``attention``: ``float`` (``F.scaled_dot_product_attention``) / ``int8``
  (``comfy_kitchen.int8_attention``, the window attention only; the global attention stays float).
* ``compute``: ``fp16`` (the production order: every tensor fp16, the pixels float32 from
  ``proj_out`` on; q / k norm + rope with ``comfy_kitchen.rms_rope_split_half_``; the int8 Linears
  with ``comfy_kitchen.int8_linear`` and its fused ``input_act`` / ``residual`` arguments, fp16 out) /
  ``fp32`` (the cross-check: the FP16 file's weights lifted to float32 (exact), everything float32,
  q / k norm + rope written in torch -- the kitchen function takes fp16 / bf16 only).
"""

from __future__ import annotations

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
#: Windows of the same token count per attention call (the plan's appendix: up to 64).
WIN_BATCH = 64
LATENT_CHANNELS = 24

_DECODER_PREFIXES = ("decoder.", "post_quant_conv.")
_INT8_LEAVES = ("attn.to_qkv", "attn.to_out", "ff.w1", "ff.w2")


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


@dataclass(frozen=True)
class SolConfig:
    weights: str      # "fp16" | "int8"
    attention: str    # "float" | "int8"
    compute: str      # "fp16" | "fp32"

    def problems(self) -> list[str]:
        bad = []
        if self.weights not in ("fp16", "int8"):
            bad.append(f"weights={self.weights!r}")
        if self.attention not in ("float", "int8"):
            bad.append(f"attention={self.attention!r}")
        if self.compute not in ("fp16", "fp32"):
            bad.append(f"compute={self.compute!r}")
        if self.compute == "fp32" and (self.weights != "fp16" or self.attention != "float"):
            bad.append("the fp32 cross-check runs the FP16 file with the float attention only")
        return bad

    def label(self) -> str:
        return f"weights={self.weights} attention={self.attention} compute={self.compute}"


#: The plan's 5.4 (A, B, C) and the cross-check path of 5.2 (F32).
CONFIGS = {
    "A": SolConfig("fp16", "float", "fp16"),
    "B": SolConfig("int8", "float", "fp16"),
    "C": SolConfig("int8", "int8", "fp16"),
    "F32": SolConfig("fp16", "float", "fp32"),
}


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
    def __init__(self, dims: SolDims, make=None, dtype=torch.float16) -> None:
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
    num_layers: int
    quant: dict            # "blocks.N.attn.to_qkv" -> h3_format.QuantSpec (empty for the FP16 file)
    mean: list
    std: list
    decoder_keys: list     # the file keys sol_key_map accepts (markers excluded)
    problems: list = field(default_factory=list)


def inspect_sol_file(path) -> SolFileInfo:
    """Header only: the layer count from the keys, the int8 markers, the latent statistics."""
    import re

    import h3_format
    import sft_quant_format

    header = sft_quant_format.read_header(str(path))
    names = set(header.tensors)
    layers = sorted({int(m.group(1)) for k in names if (m := re.match(r"^decoder\.blocks\.(\d+)\.", k))})
    problems = []
    if layers != list(range(len(layers))):
        problems.append(f"the decoder blocks are not 0..n-1: {layers[:5]}...{layers[-5:]}")
    quant: dict = {}
    for k in sorted(names):
        if not (k.startswith("decoder.blocks.") and k.endswith(".comfy_quant")):
            continue
        layer = k[len("decoder."):-len(".comfy_quant")]
        raw = sft_quant_format.read_tensor_bytes(str(path), header, k)
        quant[layer] = h3_format.parse_comfy_quant(raw, accepted=(h3_format.INT8,))
    i8 = {k[len("decoder."):-len(".weight")] for k, v in header.tensors.items()
          if k.startswith("decoder.") and k.endswith(".weight") and v.dtype == "I8"}
    if i8 != set(quant):
        problems.append(f"int8 weights without a marker or markers without an int8 weight: "
                        f"{sorted(i8 ^ set(quant))[:8]}")
    if quant:
        want = {f"blocks.{i}.{leaf}" for i in layers for leaf in _INT8_LEAVES}
        if set(quant) != want:
            problems.append(f"the int8 layers are not the 4 Linears of every block: {sorted(set(quant) ^ want)[:8]}")
        for layer in quant:
            if f"decoder.{layer}.weight_scale" not in names:
                problems.append(f"{layer} has no weight_scale")
    mean, std = VV.latent_stats_from_metadata(header.metadata, VV.METADATA_KEY, LATENT_CHANNELS)
    keys = [k for k in sorted(names) if sol_key_map(k) is not None and not k.endswith(".comfy_quant")]
    return SolFileInfo(header=header, num_layers=len(layers), quant=quant, mean=mean, std=std,
                       decoder_keys=keys, problems=problems)


def build_on_meta(info: SolFileInfo) -> SolModel:
    """The decoder for ``info``'s file on the meta device (int8 layers = ``H3Int8Linear`` holders)."""
    dims = SolDims(num_layers=info.num_layers)
    make = D.int8_linear(info.quant, bias_dtype=torch.float16) if info.quant else D.float_linear(torch.float16)
    with torch.device("meta"):
        return SolModel(dims, make, torch.float16)


def _lift_to_fp32(model: nn.Module) -> None:
    """Every floating parameter / buffer to float32 where it is (F16 -> F32 is exact)."""
    for module in model.modules():
        for name, p in list(module._parameters.items()):
            if p is not None and p.is_floating_point() and p.dtype != torch.float32:
                module._parameters[name] = nn.Parameter(p.data.float(), requires_grad=False)
        for name, b in list(module._buffers.items()):
            if b is not None and b.is_floating_point() and b.dtype != torch.float32:
                module._buffers[name] = b.float()


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
    """The same formula in torch: RMSNorm (no affine, eps 1e-5) over the head dim in float32, rounded to
    ``x``'s dtype, then the first ``2 * half`` channels rotated split-half in float32 (``x1 cos - x2 sin``,
    ``x2 cos + x1 sin``) and cast back. The rounding between the two steps is the order the CUDA kernel
    was measured to follow (``--selfcheck``, 2026-11-23); for float32 it changes nothing.

    ``x`` [n, heads, head_dim]; ``cos`` / ``sin`` [n, half] float32 or None (no rotation)."""
    xf = F.rms_norm(x.float(), (x.shape[-1],), eps=EPS).to(x.dtype).float()
    if cos is not None:
        half = cos.shape[-1]
        c, s = cos[:, None, :].to(xf.device), sin[:, None, :].to(xf.device)
        x1, x2, rest = xf[..., :half], xf[..., half:2 * half], xf[..., 2 * half:]
        xf = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s, rest), dim=-1)
    return xf.to(x.dtype)


# --------------------------------------------------------------------------- #
# the decoder
# --------------------------------------------------------------------------- #

def _ck():
    import comfy_kitchen

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
    """Normalized latents -> CPU uint8 frames. Build with :meth:`from_file` (or :meth:`from_model`)."""

    def __init__(self, model: SolModel, config: SolConfig, mean, std, device, load_report: dict | None = None,
                 *, rope_impl: str | None = None) -> None:
        bad = config.problems()
        if bad:
            raise ValueError(f"SolConfig: {bad}")
        self.model, self.config = model, config
        self.dims = model.dims
        self.device = torch.device(device)
        self.dtype = torch.float32 if config.compute == "fp32" else torch.float16
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.load_report = load_report or {}
        #: "kitchen" (comfy_kitchen.rms_rope_split_half_, fp16 on CUDA) or "torch" (rms_rope_torch).
        self.rope_impl = rope_impl or ("kitchen" if config.compute == "fp16" else "torch")
        self.int8_layers = sum(isinstance(m, H3Int8Linear) for m in model.modules())
        if config.weights == "int8" and self.int8_layers == 0:
            raise ValueError("SolConfig weights=int8 but the model has no int8 layer (the FP16 file?)")
        if config.weights == "fp16" and self.int8_layers:
            raise ValueError("SolConfig weights=fp16 but the model has int8 layers (the INT8 file?)")
        self.last_stats: dict = {}

    # -- building ---------------------------------------------------------

    @classmethod
    def from_file(cls, path, *, config: SolConfig, device) -> "SolDecoder":
        """Read the decoder's tensors of ``path`` as stored, onto ``device``; float32 for compute fp32."""
        from engine_h3.sft_load import load_streaming

        bad = config.problems()
        if bad:
            raise ValueError(f"SolConfig: {bad}")
        started = time.perf_counter()
        info = inspect_sol_file(path)
        if info.problems:
            raise RuntimeError(f"Sol file {path}: {info.problems}")
        model = build_on_meta(info)
        rep = load_streaming(model, str(path), info.header, key_map=sol_key_map)
        read_s = time.perf_counter() - started
        model.requires_grad_(False).eval()
        model.to(torch.device(device))
        if config.compute == "fp32":
            _lift_to_fp32(model)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        report = {"path_name": str(getattr(path, "name", path)).replace("\\", "/").rsplit("/", 1)[-1],
                  "num_layers": info.num_layers, "int8_layers": len(info.quant),
                  "loaded_tensors": int(rep.loaded), "loaded_mib": round(rep.bytes / 2**20, 1),
                  "read_seconds": round(read_s, 2), "seconds": round(time.perf_counter() - started, 2)}
        return cls(model, config, info.mean, info.std, device, report)

    @classmethod
    def from_model(cls, model: SolModel, *, config: SolConfig, mean, std, device, rope_impl=None) -> "SolDecoder":
        return cls(model, config, mean, std, device, {"from": "model"}, rope_impl=rope_impl)

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
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.config.attention == "int8":
            # contiguous [B, H, S, D]: the kernel's own check covers only the last stride
            o = _ck().int8_attention(qt.contiguous(), kt.contiguous(), vt.contiguous())
        else:
            o = F.scaled_dot_product_attention(qt, kt, vt)
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
                self.last_stats["window_calls"] += 1

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
            raise RuntimeError(f"Sol: block {sec.b} got attention output for {sec.written} of {sec.n} rows")
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
                    self.last_stats["sections"] += 1
            self.last_stats["max_sections_alive"] = max(self.last_stats["max_sections_alive"], len(alive))
            self._windows(alive, t0, t1, s)
            for b in sorted(alive):
                if min((b + 1) * BLOCK, g["T"]) <= t1:          # every row of block b has been a query
                    self._finish(blk, X, G, alive.pop(b))
        if alive:
            raise RuntimeError(f"Sol: layer {li} left blocks {sorted(alive)} unfinished")

    # -- the whole clip -----------------------------------------------------

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor, *, output: str = "uint8", profile: bool = False) -> torch.Tensor:
        """``latents`` [1, 24, T, h, w] (normalized, any device) -> CPU ``[F, 16h, 16w, 3]``:
        uint8 (``output="uint8"``) or float32 in [0, 1] after the clamp (``output="float"``; the
        cross-check compares these before the rounding)."""
        if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[1] != self.dims.latent_channels:
            raise ValueError(f"Sol decode: [1, 24, T, h, w] を期待しましたが {tuple(latents.shape)} でした")
        if output not in ("uint8", "float"):
            raise ValueError(f"output={output!r}")
        cuda = self.device.type == "cuda"
        mdl, dec = self.model, self.model.decoder
        dims, dt, dev = self.dims, self.dtype, self.device
        T, h, w = (int(v) for v in latents.shape[2:])
        hw, N, dim = h * w, T * h * w, dims.dim
        nb = -(-T // BLOCK)
        self._geom = {"T": T, "H": h, "W": w, "hw": hw, "nb": nb}
        self.last_stats = {"T": T, "h": h, "w": w, "blocks": nb, "sections": 0, "max_sections_alive": 0,
                           "window_calls": 0, "rope_impl": self.rope_impl, "config": self.config.label()}
        timeline = {}

        def mark(name):
            if cuda:
                torch.cuda.synchronize(dev)
            timeline[name] = time.perf_counter()

        mark("start")
        # 1. latents -> tokens
        # ``z * std + mean`` in the compute dtype: fp16 (the statistics rounded to fp16, as ComfyUI casts
        # its buffers to the latent's dtype) / float32 for the cross-check
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
                # fp16 path: cos / sin computed in float32, then rounded to fp16 (ComfyUI's table dtype);
                # the torch formula (CPU tests) gets the same rounded values
                cos, sin = cos.half().float(), sin.half().float()
            cos, sin = cos.to(dev), sin.to(dev)
            self._tables[shifted] = (kitchen_freqs(cos, sin).to(dt) if self.rope_impl == "kitchen" else None, cos, sin)
        self._q_scale = torch.ones(dims.head_dim, dtype=dt, device=dev)
        self._idx_cache = {}
        mark("embed")
        # 2. layers
        layer_s = []
        for li, blk in enumerate(dec.blocks):
            t = time.perf_counter()
            self._layer(blk, X, G, li)
            if profile:
                if cuda:
                    torch.cuda.synchronize(dev)
                layer_s.append(round(time.perf_counter() - t, 4))
        del G
        self.last_stats["window_index_patterns"] = len(self._idx_cache)
        self._tables, self._idx_cache = {}, {}
        mark("layers")
        # 3. pixels, latent by latent
        frames = D.block_frames(T)
        F_total = sum(frames)
        out = torch.empty((F_total, h * PATCH, w * PATCH, 3), dtype=torch.uint8 if output == "uint8" else torch.float32)
        pm = torch.tensor(VV.IMAGENET_MEAN, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
        ps = torch.tensor(VV.IMAGENET_STD, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
        f0 = 0
        for t in range(T):
            y = F.layer_norm(X[t * hw:(t + 1) * hw], (dim,), dec.norm_out.weight, dec.norm_out.bias, EPS)
            p = F.linear(y, dec.proj_out.weight, dec.proj_out.bias).float()
            del y
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
            raise RuntimeError(f"Sol: {f0} frames written, {F_total} expected")
        del X
        mark("pixels")
        self.last_stats["frames"] = F_total
        self.last_stats["seconds"] = {
            "embed": round(timeline["embed"] - timeline["start"], 4),
            "layers": round(timeline["layers"] - timeline["embed"], 4),
            "pixels": round(timeline["pixels"] - timeline["layers"], 4),
            "total": round(timeline["pixels"] - timeline["start"], 4)}
        if profile:
            self.last_stats["layer_seconds"] = layer_s
        return out


def frames_for_latents(t: int) -> int:
    return sum(D.block_frames(int(t)))


def comfy_memory_estimate_mib(T: int, h: int, w: int, element_bytes: int = 2) -> float:
    """The plan's appendix: ComfyUI's estimate ``(tokens + block x 28) x 2048 x size x 1.1``
    with tokens = T h w and block = 5 h w + 5 (h, w: one per 16 px). MiB, without the weights."""
    tokens = T * h * w
    block = BLOCK * h * w + NUM_GLOBAL
    return (tokens + block * 28) * 2048 * element_bytes * 1.1 / 2**20


def latent_count(frames: int) -> int:
    """17n + 5 frames -> 5n + 2 latents (the product's rule, ``engine_h3.constants.latent_frames``)."""
    from engine_h3.constants import latent_frames

    return latent_frames(int(frames))


__all__ = ["BLOCK", "CONFIGS", "SolConfig", "SolDecoder", "SolDims", "SolModel", "build_on_meta",
           "comfy_memory_estimate_mib", "frames_for_latents", "inspect_sol_file", "kitchen_freqs",
           "latent_count", "rms_rope_torch", "rope_cos_sin", "sol_key_map"]
