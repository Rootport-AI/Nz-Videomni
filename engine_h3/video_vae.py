"""MiniMax H3 video VAE: causal 3D CNN encoder + ViT decoder (24 latent channels).

Origin and licence
------------------
Ported from DiffSynth-Studio ``diffsynth/models/minimax_h3_video_vae.py`` and
``diffsynth/utils/state_dict_converters/minimax_h3_video_vae.py`` at commit
``4e34969`` (https://github.com/modelscope/DiffSynth-Studio), which is
licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Changes from the original: the
DiffSynth ``core`` imports (attention dispatcher, torchvision ``Normalize``,
einops) are replaced by plain torch; the module is built on the meta device and
filled by :func:`engine_h3.sft_load.load_streaming`; the temporal decode uses
the simpler concatenating form (same arithmetic as the streaming one); the
decoder keeps its norms and residual stream in float32 (see "Precision");
``MiniMaxH3VideoVAESimple`` is not ported. The diffusers port
(``autoencoder_kl_minimax_h3.py``, Apache-2.0) was used to cross-check every
step. No ComfyUI (GPL-3.0) code is copied.

What the file holds
-------------------
The Comfy-Org ``minimax_h3_video_vae_fp16.safetensors`` is the official
``FL2VA/video_vae/source`` checkpoint re-packed to F16 with the same 560 tensor
names and shapes, plus ``latents_mean``/``latents_std`` (F16 [24]) and an
``__metadata__["minimax_h3_video_vae"]`` JSON carrying the same 24+24 values in
full float precision. The JSON is the one read here (R2 (3)-16: the F16 tensors
are a rounded copy).

Name mapping (``video_vae_key_map``) mirrors the DiffSynth converter:
``decoder.mask_token`` is unused at inference and dropped;
``decoder.register_tokens`` and every ``*.scale1``/``*.scale2`` live inside a
one-parameter ``WarpedTensor`` and gain a ``.weight`` suffix. Everything else
maps 1:1.

Pixel and latent conventions (R2 F, (3)-13/14/15)
-------------------------------------------------
* Pixels are ImageNet-normalised RGB over a [0, 1] base, NOT [-1, 1].
* Latents are normalised per channel with the 24 mean/std values:
  ``encode`` returns ``(z - mean) / std``, ``decode`` takes that and applies
  ``z * std + mean`` first.
* Time: 17 pixel frames per encoder clip, 3 trailing latent frames dropped per
  encode, so ``17n + 5`` frames <-> ``5n + 2`` latents. The decoder walks
  7-latent windows (5 + 2 overlap) and cross-fades the 5 overlapping pixel
  frames, which is why ``17n + 5`` is exactly what it emits.
* Spatial tiling is ON by default (256 px tiles, >= 64 px overlap widened in
  16 px steps to cover the frame exactly, linear blend): the released model's
  frames are the blended-tile ones, so turning tiling off changes the output.
* A single latent frame decodes badly on its own (ComfyUI issue #15416), so
  ``decode`` repeats it to one full encoder chunk (5 latents), decodes that
  clip, and returns its first real pixel frame (the one after the 3 implicit
  pre-padding frames). Measured on the real weights (CPU, float32, a 128 px
  still): 5 copies give 31.1 dB at that frame, 2 copies only 21.3 dB there --
  a 2-latent clip is not a shape the decoder ever saw, while a chunk of copies
  is exactly what encoding 17 identical frames would produce.

Precision
---------
The heavy work (convolutions, decoder linears, attention) runs in the module
dtype (bfloat16 in production). The decoder's RMSNorm/LayerNorm weights,
``scale1``/``scale2``, ``x_embedder`` and ``post_quant_conv`` are kept in
float32 and the decoder's residual stream is float32, as the reference does
(diffusers ``_keep_in_fp32_modules`` and its "the reference normalizes in
float32" note); they are a few MB, so this costs nothing. Inputs and outputs of
``encode``/``decode`` are float32.
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine_h3 import constants as C

#: ImageNet statistics; the VAE's pixel space (R2 F).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: ``__metadata__`` key of the Comfy-Org file that carries the 24+24 values.
METADATA_KEY = "minimax_h3_video_vae"
LATENT_CHANNELS = 24

#: File tensors with no counterpart in the module (see the module docstring).
_DROPPED_KEYS = frozenset({"decoder.mask_token", "latents_mean", "latents_std"})


def video_vae_key_map(name: str) -> str | None:
    """File tensor name -> module state_dict key (None = not loaded)."""
    if name in _DROPPED_KEYS:
        return None
    if name == "decoder.register_tokens" or name.endswith(".scale1") or name.endswith(".scale2"):
        return name + ".weight"
    return name


def latent_stats_from_metadata(metadata: dict, key: str, channels: int) -> tuple[list[float], list[float]]:
    """Read ``latents_mean``/``latents_std`` from a Comfy-Org VAE ``__metadata__`` JSON.

    Shared with :mod:`engine_h3.audio_vae`. Raises ``ValueError`` (Japanese
    message, it reaches the worker's error event) when the JSON is missing or
    the lengths are wrong -- a file without it is not the one this engine knows.
    """
    raw = metadata.get(key)
    if raw is None:
        raise ValueError(
            f"VAE のメタデータに '{key}' がありません。Comfy-Org 版の MiniMax H3 の VAE ではない可能性があります"
        )
    try:
        cfg = json.loads(raw)
        mean = [float(v) for v in cfg["latents_mean"]]
        std = [float(v) for v in cfg["latents_std"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"VAE のメタデータ '{key}' を読めません: {exc}") from exc
    if len(mean) != channels or len(std) != channels:
        raise ValueError(
            f"VAE のメタデータ '{key}' の latents_mean/std は {channels} 値のはずですが "
            f"{len(mean)}/{len(std)} 値でした"
        )
    if any(s <= 0.0 for s in std):
        raise ValueError(f"VAE のメタデータ '{key}' の latents_std に 0 以下の値があります")
    return mean, std


# ---------------------------------------------------------------------------
# Building blocks (names match the checkpoint)
# ---------------------------------------------------------------------------


class WarpedTensor(nn.Module):
    """A bare tensor wrapped as ``.weight`` (DiffSynth's layout; hence the key_map suffix)."""

    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape), requires_grad=False)

    def forward(self) -> torch.Tensor:
        return self.weight


class BaseConv3d(nn.Conv3d):
    """Conv3d with explicit padding: spatial ``pad_mode`` on H/W, causal zeros in time.

    ``padding`` is passed to ``nn.Conv3d`` only to record it; the convolution
    itself runs with padding 0 after the manual pads (as in the original).
    Causal: ``2 * p_t`` zero frames in front, none behind, so a T=1 input
    (a still image) sees the same zeros as the first frame of a clip.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_mode="zeros"):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
        self.pad_mode = "constant" if padding_mode == "zeros" else padding_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_t, p_h, p_w = self.padding
        if p_h or p_w:
            x = F.pad(x, (p_w, p_w, p_h, p_h, 0, 0), mode=self.pad_mode)
        if p_t:
            x = F.pad(x, (0, 0, 0, 0, 2 * p_t, 0), mode="constant")
        return F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


class TemporalIsolatedSpatialGroupNorm(nn.GroupNorm):
    """GroupNorm per latent frame: time folded into batch so statistics never mix frames."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = super().forward(y)
        return y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _group_norm(ch: int) -> nn.GroupNorm:
    return TemporalIsolatedSpatialGroupNorm(32, ch, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2, padding_mode="reflect"):
        super().__init__()
        self.conv = BaseConv3d(
            in_channels, out_channels, kernel_size=3, padding=(1, 0, 0),
            stride=(time_stride, space_stride, space_stride), padding_mode=padding_mode,
        )
        self.space_stride = space_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.space_stride == 2:
            # Asymmetric bottom/right pad so the output is exactly ceil(size / 2).
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode=self.conv.pad_mode)
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode="reflect"):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.norm2 = _group_norm(out_channels)
        self.conv1 = BaseConv3d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv2 = BaseConv3d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        if in_channels != out_channels:
            self.nin_shortcut = BaseConv3d(in_channels, out_channels, kernel_size=1, padding_mode=padding_mode)
        else:
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class EncoderFCN3D(nn.Module):
    """Causal 3D CNN encoder (pixels -> mean/logvar). 第 1 弾（T2V）では未使用・未検証（CPU の往復を 1 回試しただけ）。第 2 弾の先頭フレームで使う。"""

    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels,
                 padding_mode="reflect"):
        super().__init__()
        levels = len(ch_mult)
        block_mid = [ch * m for m in ch_mult]
        block_in = [block_mid[0]] + block_mid[:-1]
        self.conv_in = BaseConv3d(in_channels, block_in[0], kernel_size=3, padding=1, padding_mode=padding_mode)
        self.down = nn.ModuleList()
        for i in range(levels):
            level = nn.Module()
            level.block = nn.ModuleList([
                ResnetBlock3D(block_in[i] if j == 0 else block_mid[i], block_mid[i], padding_mode=padding_mode)
                for j in range(num_res_blocks)
            ])
            if space_down[i] * time_down[i] > 1:
                level.downsample = Downsample3D(
                    block_mid[i], block_mid[i], time_stride=time_down[i], space_stride=space_down[i],
                    padding_mode=padding_mode,
                )
            self.down.append(level)
        self.norm_out = _group_norm(block_mid[-1])
        # double_z: mean and logvar.
        self.conv_out = BaseConv3d(block_mid[-1], 2 * z_channels, kernel_size=3, padding=1, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for level in self.down:
            for block in level.block:
                h = block(h)
            if hasattr(level, "downsample"):
                h = level.downsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first ``cos.shape[-1]`` channels of every head; the rest pass through."""
    cos, sin = cos.to(t.dtype), sin.to(t.dtype)
    rot = cos.shape[-1]
    t_rot, t_pass = t[..., :rot], t[..., rot:]
    return torch.cat((t_rot * cos + _rotate_half(t_rot) * sin, t_pass), dim=-1)


class Attention(nn.Module):
    """Self-attention with a fused qkv laid out per head as ``[q_h; k_h; v_h]``.

    The ``view(B, S, heads, 3 * dim_head)`` + ``chunk(3)`` split is the
    original layout (ComfyUI reads it the same way); diffusers re-packs it into
    separate to_q/to_k/to_v at conversion time, which is why its code differs.
    Q/K are RMS-normalised (no affine) in float32 like the reference.
    """

    def __init__(self, heads, dim_head, eps=1e-5):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        dim = heads * dim_head
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q, k, v = self.to_qkv(x).view(b, s, self.heads, 3 * self.dim_head).chunk(3, dim=-1)
        q = self.norm_q(q.float()).to(v.dtype)
        k = self.norm_k(k.float()).to(v.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.to_out(out.transpose(1, 2).reshape(b, s, -1))


class FeedForward(nn.Module):
    """Gated SiLU FFN; ``w1`` is ``[gate; value]`` and SiLU goes on the FIRST half."""

    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 8, bias=True)
        self.w2 = nn.Linear(dim * 4, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * value)


class TransformerBlock(nn.Module):
    def __init__(self, heads, dim_head, eps=1e-5):
        super().__init__()
        dim = heads * dim_head
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = Attention(heads, dim_head, eps=eps)
        self.scale1 = WarpedTensor(dim)
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = FeedForward(dim)
        self.scale2 = WarpedTensor(dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x is the float32 residual stream; only the projections run in the module dtype.
        cd = self.attn.to_qkv.weight.dtype
        h = self.norm1(x.float()).to(cd)
        x = x + self.attn(h, cos, sin).float() * self.scale1().float()
        h = self.norm2(x).to(cd)
        return x + self.ff(h).float() * self.scale2().float()


class ViT3DDecoder(nn.Module):
    """Every latent voxel is one token; 4 register tokens + 1 zero token are appended at position 0."""

    def __init__(self, patch_size=16, patch_size_t=4, in_channels=24, out_channels=3, num_layers=36,
                 heads=32, dim_head=64, rope_theta=100.0, rope_dim_ratio=0.75, eps=1e-5, num_register_tokens=4):
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.rope_dim = int(dim_head * rope_dim_ratio)
        self.rope_theta = rope_theta
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = WarpedTensor((1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(heads, dim_head, eps=eps) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def _rope(self, t: int, h: int, w: int, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin [B, S, 1, rope_dim] in float32.

        Coordinates are cell centres normalised to [-1, 1) per axis; the
        suffix tokens sit at 0. Angles are ``2 pi * coord * inv_freq`` with
        ``rope_dim / 6`` frequencies per axis, concatenated over (t, h, w) and
        duplicated (rotate_half layout). Built here rather than as a buffer so
        nothing is left on the meta device after loading.
        """
        n_dim = 3
        inv_freq = 1.0 / self.rope_theta ** torch.arange(0, 1, 2 * n_dim / self.rope_dim, dtype=torch.float32, device=device)
        axes = [2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=device) / size) - 1.0 for size in (t, h, w)]
        ids = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        ids = torch.cat([ids, ids.new_zeros(self.num_register_tokens + 1, 3)], dim=0)
        angles = (2.0 * math.pi * ids[:, :, None] * inv_freq[None, None, :]).flatten(1, 2).tile(2)
        angles = angles[None, :, None, :].expand(batch, -1, -1, -1)
        return angles.cos(), angles.sin()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = z.shape
        tokens = z.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        x = self.x_embedder(tokens.to(self.x_embedder.weight.dtype)).float()
        num_patches = x.shape[1]
        x = torch.cat([x, self.register_tokens().float().expand(b, -1, -1), torch.zeros_like(x[:, :1])], dim=1)
        cos, sin = self._rope(t, h, w, b, z.device)
        for block in self.transformer_blocks:
            x = block(x, cos, sin)
        x = self.norm_out(x.float()).to(self.proj_out.weight.dtype)
        out = self.proj_out(x)[:, :num_patches]
        p, pt = self.patch_size, self.patch_size_t
        out = out.view(b, t, h, w, self.out_channels, pt, p, p).permute(0, 4, 1, 5, 2, 6, 3, 7)
        return out.reshape(b, self.out_channels, t * pt, h * p, w * p)


# ---------------------------------------------------------------------------
# The VAE
# ---------------------------------------------------------------------------


class H3VideoVAE(nn.Module):
    """See the module docstring. Build on the meta device; use :meth:`from_file`."""

    def __init__(self, *, ch=128, ch_mult=(1, 2, 2, 4, 4, 8), space_down=(2, 2, 2, 2, 1, 1),
                 time_down=(1, 2, 2, 1, 1, 1), num_res_blocks=2, z_channels=LATENT_CHANNELS,
                 decoder_layers=36, decoder_heads=32, decoder_dim_head=64,
                 clip_length=C.VAE_CLIP_LENGTH, token_drop=C.VAE_TOKEN_DROP,
                 tile_size=256, tile_overlap_min=64):
        # The keyword arguments exist so CPU tests can build a small decoder;
        # production always uses the defaults (the released configuration,
        # ``__metadata__["minimax_h3_video_vae"]["source_config"]``).
        super().__init__()
        self.vae_ratio = math.prod(space_down)
        self.vae_ratio_t = math.prod(time_down)
        self.clip_length = clip_length
        self.token_drop = token_drop
        self.frame_pre_padding = (-clip_length) % self.vae_ratio_t           # 3
        self.tokens_chunk_size = math.ceil(clip_length / self.vae_ratio_t)   # 5
        self.token_overlap = (-token_drop) % self.tokens_chunk_size          # 2
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)  # 5
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min

        self.encoder = EncoderFCN3D(ch, ch_mult, space_down, time_down, num_res_blocks, 3, z_channels)
        # quant_conv: encoder side only. 第 1 弾（T2V）では未使用・未検証（CPU の往復を 1 回試しただけ）。第 2 弾の先頭フレームで使う。
        self.quant_conv = nn.Conv3d(2 * z_channels, 2 * z_channels, 1)
        self.post_quant_conv = nn.Conv3d(z_channels, z_channels, 1)
        self.decoder = ViT3DDecoder(
            patch_size=self.vae_ratio, patch_size_t=self.vae_ratio_t, in_channels=z_channels,
            num_layers=decoder_layers, heads=decoder_heads, dim_head=decoder_dim_head,
        )
        # Filled from __metadata__ (not the F16 tensors) by from_file; not in the state_dict.
        self.register_buffer("latents_mean", torch.zeros(z_channels, dtype=torch.float32), persistent=False)
        self.register_buffer("latents_std", torch.ones(z_channels, dtype=torch.float32), persistent=False)

    # -- loading ----------------------------------------------------------

    @classmethod
    def from_file(cls, path: str, *, device, dtype=torch.bfloat16) -> "H3VideoVAE":
        """Build on meta, stream the Comfy-Org file in, move to ``device`` and cast to ``dtype``.

        The cast happens after the move so the CPU never holds an F16 and a
        BF16 copy of the ~4.8 GB decoder at the same time.
        """
        import sft_quant_format
        from engine_h3.sft_load import load_streaming

        header = sft_quant_format.read_header(path)
        mean, std = latent_stats_from_metadata(header.metadata, METADATA_KEY, LATENT_CHANNELS)
        with torch.device("meta"):
            model = cls()
        model.latents_mean = torch.tensor(mean, dtype=torch.float32)
        model.latents_std = torch.tensor(std, dtype=torch.float32)
        load_streaming(model, path, header, key_map=video_vae_key_map)
        model.requires_grad_(False).eval()
        model.to(device=device)
        model.apply_precision(dtype)
        return model

    def float32_islands(self) -> list[nn.Module]:
        """The small cast-sensitive modules kept in float32 (see "Precision")."""
        mods: list[nn.Module] = [self.post_quant_conv, self.decoder.x_embedder, self.decoder.norm_out]
        for block in self.decoder.transformer_blocks:
            mods += [block.norm1, block.norm2, block.scale1, block.scale2]
        return mods

    def apply_precision(self, dtype: torch.dtype) -> None:
        """Cast each tensor ONCE to its final dtype: islands to float32, everything else to ``dtype``.

        Casting the whole module to bf16 first and the islands back afterwards
        would round the file's F16 island values through bf16 (3 mantissa bits
        lost); here an island tensor goes F16 -> F32 directly, which is exact.
        Runs after the move to the device so the cast happens there. The
        latent statistics buffers stay float32 (they are never cast).
        """
        island_params = {id(p) for m in self.float32_islands() for p in m.parameters()}
        for module in self.modules():
            for name, param in module.named_parameters(recurse=False):
                target = torch.float32 if id(param) in island_params else dtype
                if param.dtype != target:
                    param.data = param.data.to(target)
            for name, buf in module.named_buffers(recurse=False):
                if module is self and name in ("latents_mean", "latents_std"):
                    continue
                if buf.is_floating_point() and buf.dtype != dtype:
                    module._buffers[name] = buf.to(dtype)

    # -- spatial tiling ---------------------------------------------------

    def split_tiles(self, length: int) -> tuple[list[int], list[int], list[int]]:
        """Fewest ``tile_size`` tiles covering ``length`` with overlaps >= the minimum,
        the slack spread round-robin over the overlaps in ``vae_ratio`` steps."""
        ts, ov_min = self.tile_size, self.tile_overlap_min
        if ts >= length:
            return [0], [length], []
        n = math.ceil(length / ts)
        while ts * n - ov_min * (n - 1) - length < 0:
            n += 1
        overlaps = [ov_min] * (n - 1)
        remaining = ts * n - sum(overlaps) - length
        for i in range(remaining // self.vae_ratio):
            overlaps[i % (n - 1)] += self.vae_ratio
        starts = [0]
        for i in range(n - 1):
            starts.append(starts[-1] + ts - overlaps[i])
        return starts, [ts] * n, overlaps

    @staticmethod
    def blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
        """Linear cross-fade of the tail of ``a`` into the head of ``b`` along ``dim``; returns ``b``'s extent."""
        extent = min(a.shape[dim], b.shape[dim], extent)
        w = torch.arange(extent, device=b.device, dtype=b.dtype) / extent
        shape = [1] * a.ndim
        shape[dim] = extent
        w = w.view(shape)
        mixed = a.narrow(dim, a.shape[dim] - extent, extent) * (1 - w) + b.narrow(dim, 0, extent) * w
        if extent < b.shape[dim]:
            return torch.cat([mixed, b.narrow(dim, extent, b.shape[dim] - extent)], dim=dim)
        return mixed

    @staticmethod
    def _stitch(rows: list[list[torch.Tensor]], y_ov: list[int], x_ov: list[int]) -> torch.Tensor:
        out_rows = []
        for i, row in enumerate(rows):
            out_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = H3VideoVAE.blend(rows[i - 1][j], tile, y_ov[i - 1], dim=-2)
                if j > 0:
                    tile = H3VideoVAE.blend(row[j - 1], tile, x_ov[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., : -y_ov[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -x_ov[j]]
                out_row.append(tile)
            out_rows.append(torch.cat(out_row, dim=-1))
        return torch.cat(out_rows, dim=-2)

    def _encode_clip(self, x: torch.Tensor, tiled: bool) -> torch.Tensor:
        """Pixels (normalised, module dtype) -> float32 moments [B, 48, t, h, w] for one clip.

        第 1 弾（T2V）では未使用・未検証（CPU の往復を 1 回試しただけ）。第 2 弾の先頭フレームで使う。
        """
        if not tiled:
            return self.quant_conv(self.encoder(x)).float()
        y_idx, y_len, y_ov = self.split_tiles(x.shape[-2])
        x_idx, x_len, x_ov = self.split_tiles(x.shape[-1])
        rows = [
            [self.quant_conv(self.encoder(x[..., i:i + il, j:j + jl])).float() for j, jl in zip(x_idx, x_len)]
            for i, il in zip(y_idx, y_len)
        ]
        r = self.vae_ratio
        return self._stitch(rows, [o // r for o in y_ov], [o // r for o in x_ov])

    def _decode_clip(self, z: torch.Tensor, tiled: bool) -> torch.Tensor:
        """Raw latents (float32) -> float32 pixels in ImageNet space for one clip."""
        pq = self.post_quant_conv
        if not tiled:
            return self.decoder(pq(z.to(pq.weight.dtype))).float()
        r = self.vae_ratio
        y_idx, y_len, y_ov = self.split_tiles(z.shape[-2] * r)
        x_idx, x_len, x_ov = self.split_tiles(z.shape[-1] * r)
        rows = []
        for i, il in zip(y_idx, y_len):
            row = []
            for j, jl in zip(x_idx, x_len):
                tile = z[..., i // r:(i + il) // r, j // r:(j + jl) // r]
                row.append(self.decoder(pq(tile.to(pq.weight.dtype))).float())
            rows.append(row)
        return self._stitch(rows, y_ov, x_ov)

    # -- public API -------------------------------------------------------

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.encoder.conv_in.weight.dtype

    def _pixel_stats(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
        return mean, std

    def encode(self, video: torch.Tensor, *, tiled: bool = True) -> torch.Tensor:
        """[1, 3, F, H, W] float32 in [0, 1] -> normalised posterior mean [1, 24, T', H/16, W/16] float32.

        第 1 弾（T2V）では未使用・未検証（CPU の往復を 1 回試しただけ）。第 2 弾の先頭フレームで使う。

        F = 1 (a key frame) goes through the spatial encoder alone and gives
        one latent frame; repeating it to 17 frames would give 2 latents and
        is not the conditioning the model was trained with (R2 (3)-13).
        Otherwise the clip is padded to a multiple of 17 by repeating the last
        frame, encoded 17 frames at a time, and the trailing 3 latents dropped.
        """
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"encode: [B, 3, F, H, W] を期待しましたが {tuple(video.shape)} でした")
        if video.shape[-2] % self.vae_ratio or video.shape[-1] % self.vae_ratio:
            raise ValueError(f"encode: 高さと幅は {self.vae_ratio} の倍数が必要です（{tuple(video.shape[-2:])}）")
        pm, ps = self._pixel_stats(video.device)
        x = ((video.float() - pm) / ps).to(self.compute_dtype)
        num_frames = x.shape[2]
        if num_frames == 1:
            moments = self._encode_clip(x, tiled)[:, :, :1]
        else:
            if num_frames % self.clip_length:
                x = torch.cat([x, x[:, :, -1:].repeat(1, 1, (-num_frames) % self.clip_length, 1, 1)], dim=2)
            moments = torch.cat(
                [self._encode_clip(x[:, :, i:i + self.clip_length], tiled) for i in range(0, x.shape[2], self.clip_length)],
                dim=2,
            )
            if self.token_drop > 0:
                moments = moments[:, :, : -self.token_drop]
        z = moments.chunk(2, dim=1)[0]
        mean = self.latents_mean.to(z.device).view(1, -1, 1, 1, 1)
        std = self.latents_std.to(z.device).view(1, -1, 1, 1, 1)
        return (z - mean) / std

    def decode(self, latents: torch.Tensor, *, tiled: bool = True) -> torch.Tensor:
        """Normalised latents [1, 24, T', h, w] float32 -> pixels [1, 3, F, 16h, 16w] float32 in [0, 1].

        T' = 5n + 2 gives F = 17n + 5. T' = 1 is decoded as one chunk of copies and the
        first real frame is returned (F = 1, ComfyUI issue #15416).
        """
        if latents.ndim != 5 or latents.shape[1] != self.latents_mean.shape[0]:
            raise ValueError(f"decode: [B, 24, T', h, w] を期待しましたが {tuple(latents.shape)} でした")
        mean = self.latents_mean.to(latents.device).view(1, -1, 1, 1, 1)
        std = self.latents_std.to(latents.device).view(1, -1, 1, 1, 1)
        z = latents.float() * std + mean
        if z.shape[2] == 1:
            clip = self._decode_clip(z.repeat(1, 1, self.tokens_chunk_size, 1, 1), tiled)
            pixels = clip[:, :, self.frame_pre_padding:self.frame_pre_padding + 1]
        else:
            pixels = self._decode_temporal(z, tiled)
        pm, ps = self._pixel_stats(pixels.device)
        return (pixels * ps + pm).clamp_(0.0, 1.0)

    def _decode_temporal(self, z: torch.Tensor, tiled: bool) -> torch.Tensor:
        """Walk (5 + 2)-latent windows, drop the 3 pre-padding frames of each
        17+ frame half, cross-fade the 5 overlapping frames, and trim the frames
        that come from repeated pad latents (when T' is not 5n + 2)."""
        chunk = self.tokens_chunk_size
        chunk_frames = chunk * self.vae_ratio_t
        num_tokens = z.shape[2] + self.token_drop
        pad_tokens = (-num_tokens) % chunk
        num_chunks = (num_tokens + pad_tokens) // chunk - int(self.token_drop > 0)
        if num_chunks < 1:
            raise ValueError(
                f"decode: 潜在フレーム数 {z.shape[2]} は短すぎます（1 か、3 以上が必要。17n+5 なら 5n+2）"
            )
        if pad_tokens:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        parts = []
        overlap = None
        for i in range(num_chunks):
            start = i * chunk
            clip = self._decode_clip(z[:, :, start:start + chunk + self.token_overlap], tiled)
            for j in range(int(self.token_drop > 0) + 1):
                piece = clip[:, :, j * chunk_frames:(j + 1) * chunk_frames][:, :, self.frame_pre_padding:]
                if j == 0:
                    if overlap is not None:
                        piece = self.blend(overlap, piece, self.frame_overlap, dim=-3)
                    parts.append(piece)
                else:
                    overlap = piece
            del clip
        if overlap is not None:
            parts.append(overlap)
        out = torch.cat(parts, dim=2)
        if pad_tokens:
            tail = self.clip_length % self.vae_ratio_t
            before = z.shape[2] - pad_tokens
            pad_frames = sum(
                tail if tail and (before + k) % chunk == 0 else self.vae_ratio_t for k in range(pad_tokens)
            )
            out = out[:, :, :-pad_frames]
        return out
