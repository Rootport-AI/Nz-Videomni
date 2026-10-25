"""MiniMax H3 audio VAE: decoder (latents -> 32 kHz waveform, BigVGAN) and encoder (waveform -> latents).

Origin and licence
------------------
Ported from DiffSynth-Studio ``diffsynth/models/minimax_h3_audio_vae.py`` and
``diffsynth/utils/state_dict_converters/minimax_h3_audio_vae.py`` at commit
``4e34969`` (https://github.com/modelscope/DiffSynth-Studio), which is
licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Changes from the original: the
weight-normalised convolutions are plain ``nn.Conv1d``/``nn.ConvTranspose1d``
(see below); the anti-aliasing filters are buffers loaded from the file; the
decoder and the encoder are two separate modules (the posterior ``logs_proj``
is not ported: H3 always uses the mean); each is built on the meta device and
filled by :func:`engine_h3.sft_load.load_streaming`. The diffusers port
(``autoencoder_kl_minimax_h3_audio.py``, Apache-2.0) was used to cross-check.
No ComfyUI (GPL-3.0) code is copied.

What the file holds
-------------------
The Comfy-Org ``minimax_h3_audio_vae_fp32.safetensors`` (917 tensors, F32) is
the official ``FL2VA/audio_vae`` checkpoint with every weight-norm pair
``*.weight_g``/``*.weight_v`` already folded into one ``*.weight`` of the
``weight_v`` shape (R2 論点 2, (3)-16). So every convolution here is an
ordinary convolution with an ordinary ``weight`` -- recomputing
``g * v / ||v||`` would be wrong for this file. The official (unfolded) file is
NOT supported by this module.

The 32+32 latent statistics are read from
``__metadata__["minimax_h3_audio_vae"]`` (the same values as the F32
``latents_mean``/``latents_std`` tensors, read from the same place as the video
VAE for one rule).

The file is split into parts by one prefix table (``_FILE_PARTS``), from which
both key maps are made:

* :class:`H3AudioVAE` (the decoder, ``dec_in_proj`` + ``decoder.*``) is what
  T2V and the keyframe I2V use; ``audio_vae_key_map`` drops the encoder side,
  ``logs_proj.*`` and the statistic tensors.
* :class:`H3AudioEncoder` (A2V, 台帳 §1-103) reads ``encoder.*`` (110 tensors),
  ``pre_block.*`` (22) and ``mean_proj.*`` (2) through
  ``audio_encoder_key_map``, 134 in all; ``logs_proj.*``, the decoder and the
  statistic tensors are dropped. The attention projection's
  ``q_bias``/``v_bias``/``zero_k_bias`` are read under the file's own names
  (the DiffSynth converter suffixes them with ``.weight`` for its wrapper).

Shapes (R2 F)
-------------
Latents are ``[2, 32, A]``: the two stereo channels run as a batch of 2 through
a mono VAE. ``A = round(F / 24 * 40)``; one latent is 800 samples at 32 kHz, so
the waveform is ``[2, A * 800]`` and can be up to half a latent (12.5 ms)
longer than the video -- :func:`engine_h3.media_out.write_mp4` trims it.
Everything runs in float32.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine_h3 import constants as C
from engine_h3.video_vae import latent_stats_from_metadata

METADATA_KEY = "minimax_h3_audio_vae"
LATENT_CHANNELS = 32

# Which part of the file each tensor belongs to: the one table both key maps are made from.
PART_DECODER, PART_ENCODER, PART_UNUSED, PART_STATS = "decoder", "encoder", "unused", "stats"
_FILE_PARTS: dict[str, str] = {
    "dec_in_proj.": PART_DECODER,
    "decoder.": PART_DECODER,
    "encoder.": PART_ENCODER,
    "pre_block.": PART_ENCODER,
    "mean_proj.": PART_ENCODER,
    "logs_proj.": PART_UNUSED,     # the posterior's log-scale: never evaluated (the mean is used)
}
_STAT_KEYS = frozenset({"latents_mean", "latents_std"})   # read from __metadata__ instead

# 110 (encoder.*) + 22 (pre_block.*) + 2 (mean_proj.*) in the Comfy-Org fp32 file.
ENCODER_TENSOR_COUNT = 134


def file_part(name: str) -> str | None:
    """The part of the file a tensor name belongs to (None = not in the table)."""
    if name in _STAT_KEYS:
        return PART_STATS
    for prefix, part in _FILE_PARTS.items():
        if name.startswith(prefix):
            return part
    return None


def audio_vae_key_map(name: str) -> str | None:
    """File tensor name -> decoder state_dict key (None = not loaded; encoder side and stats).

    A name outside the table is passed through, so ``load_streaming`` logs it
    as unexpected (the decoder's behaviour before the encoder existed).
    """
    if file_part(name) in (PART_ENCODER, PART_UNUSED, PART_STATS):
        return None
    return name


def audio_encoder_key_map(name: str) -> str | None:
    """File tensor name -> encoder state_dict key (None = not loaded; decoder, logs_proj and stats)."""
    return name if file_part(name) == PART_ENCODER else None


class SnakeBeta(nn.Module):
    """``x + 1 / (exp(beta) + 1e-9) * sin(exp(alpha) * x)^2``; alpha/beta are stored in log space, ``[C]``."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(channels), requires_grad=False)
        self.beta = nn.Parameter(torch.empty(channels), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1).exp()
        beta = self.beta.view(1, -1, 1).exp()
        return x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class LowPassFilter1d(nn.Module):
    """Depthwise Kaiser-sinc low-pass with a stride (the anti-aliased downsampler).

    The filter taps come from the file (``*.lowpass.filter``), so nothing is
    recomputed; replicate padding of (k/2 - 1, k/2) for the even k = 12.
    """

    def __init__(self, stride: int = 2, kernel_size: int = 12):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.register_buffer("filter", torch.empty(1, 1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.kernel_size // 2 - 1, self.kernel_size // 2), mode="replicate")
        return F.conv1d(x, self.filter.expand(x.shape[1], -1, -1), stride=self.stride, groups=x.shape[1])


class UpSample1d(nn.Module):
    """Anti-aliased ``ratio`` x upsampler (transposed depthwise Kaiser-sinc)."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer("filter", torch.empty(1, 1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(c, -1, -1), stride=self.stride, groups=c)
        return x[..., self.pad_left:-self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.lowpass = LowPassFilter1d(stride=ratio, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    """Upsample -> SnakeBeta -> downsample (BigVGAN's alias-free activation)."""

    def __init__(self, channels: int):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = UpSample1d(2, 12)
        self.downsample = DownSample1d(2, 12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class AMPBlock1(nn.Module):
    """BigVGAN AMP block: per dilation, act -> dilated conv -> act -> conv, added to the input."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=(kernel_size * d - d) // 2)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=(kernel_size - 1) // 2)
            for _ in dilation
        ])
        self.activations = nn.ModuleList([Activation1d(channels) for _ in range(2 * len(dilation))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            x = c2(a2(c1(a1(x)))) + x
        return x


class BigVGAN(nn.Module):
    """[B, 2048, A] -> [B, 1, A * 800], clamped to [-1, 1] (rates 5,5,2,2,2,2,2)."""

    UP_RATES = (5, 5, 2, 2, 2, 2, 2)
    UP_KERNELS = (9, 9, 4, 4, 4, 4, 4)
    RES_KERNELS = (3, 7, 11)

    def __init__(self, in_channels: int = 2048, initial_channels: int = 1024):
        super().__init__()
        self.conv_pre = nn.Conv1d(in_channels, initial_channels, 7, 1, padding=3)
        # One-element ModuleLists: the checkpoint's names are ``ups.<i>.0``.
        self.ups = nn.ModuleList([
            nn.ModuleList([nn.ConvTranspose1d(
                initial_channels // (2 ** i), initial_channels // (2 ** (i + 1)), k, u, padding=(k - u) // 2,
            )])
            for i, (u, k) in enumerate(zip(self.UP_RATES, self.UP_KERNELS))
        ])
        self.resblocks = nn.ModuleList()
        ch = initial_channels
        for i in range(len(self.UP_RATES)):
            ch = initial_channels // (2 ** (i + 1))
            for k in self.RES_KERNELS:
                self.resblocks.append(AMPBlock1(ch, k, (1, 3, 5)))
        self.activation_post = Activation1d(ch)
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        n = len(self.RES_KERNELS)
        for i, up in enumerate(self.ups):
            x = up[0](x)
            xs = self.resblocks[i * n](x)
            for j in range(1, n):
                xs = xs + self.resblocks[i * n + j](x)
            x = xs / n
        x = self.conv_post(self.activation_post(x))
        return torch.clamp(x, min=-1.0, max=1.0)


class H3AudioVAE(nn.Module):
    """Decoder half of the audio VAE (see the module docstring). Use :meth:`from_file`."""

    def __init__(self, *, latent_channels: int = LATENT_CHANNELS, latent_dim: int = 2048):
        super().__init__()
        self.sample_rate = C.AUDIO_SAMPLE_RATE
        self.hop_length = C.AUDIO_HOP
        self.dec_in_proj = nn.Conv1d(latent_channels, latent_dim, 1)
        self.decoder = BigVGAN(in_channels=latent_dim)
        self.register_buffer("latents_mean", torch.zeros(latent_channels, dtype=torch.float32), persistent=False)
        self.register_buffer("latents_std", torch.ones(latent_channels, dtype=torch.float32), persistent=False)

    @classmethod
    def from_file(cls, path: str, *, device) -> "H3AudioVAE":
        import sft_quant_format
        from engine_h3.sft_load import load_streaming

        header = sft_quant_format.read_header(path)
        mean, std = latent_stats_from_metadata(header.metadata, METADATA_KEY, LATENT_CHANNELS)
        with torch.device("meta"):
            model = cls()
        model.latents_mean = torch.tensor(mean, dtype=torch.float32)
        model.latents_std = torch.tensor(std, dtype=torch.float32)
        load_streaming(model, path, header, key_map=audio_vae_key_map)
        model.requires_grad_(False).eval()
        # The file is F32 already; .float() only guards against a future F16 copy.
        return model.to(device=device).float()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalised latents [2, 32, A] float32 -> waveform [2, A * 800] float32 in [-1, 1]."""
        if latents.ndim != 3 or latents.shape[1] != self.latents_mean.shape[0]:
            raise ValueError(f"audio decode: [2, 32, A] を期待しましたが {tuple(latents.shape)} でした")
        mean = self.latents_mean.to(latents.device).view(1, -1, 1)
        std = self.latents_std.to(latents.device).view(1, -1, 1)
        z = latents.float() * std + mean
        wav = self.decoder(self.dec_in_proj(z))   # [2, 1, A * 800]
        return wav[:, 0, :]


# --------------------------------------------------------------------------
# Encoder (A2V): DAC-style convolutional encoder -> causal-attention projection -> mean_proj
# --------------------------------------------------------------------------


class Snake(nn.Module):
    """``x + 1 / (alpha + 1e-9) * sin(alpha * x)^2`` with the RAW alpha ``[1, C, 1]`` (no exp, unlike SnakeBeta)."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class ResidualUnit(nn.Module):
    """Snake -> dilated conv (k=7) -> Snake -> conv (k=1), added to the input."""

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.block = nn.Sequential(
            Snake(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        # The padding keeps the length; the original's centre crop is kept as it was.
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    """Three residual units (dilation 1, 3, 9) on ``dim / 2`` channels, Snake, then a strided conv to ``dim``."""

    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake(dim // 2),
            nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DACEncoder(nn.Module):
    """[B, 1, L] -> [B, d_model * 2^len(strides), L / prod(strides)]; ``block.0`` .. ``block.<n+2>`` as in the file."""

    def __init__(self, d_model: int = 64, strides: tuple[int, ...] = (2, 4, 4, 5, 5)):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            layers.append(EncoderBlock(d_model, stride=stride))
        layers += [Snake(d_model), nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GeGluMlp(nn.Module):
    """LayerNorm -> ``gelu_tanh(w0(x)) * w1(x)`` -> w2."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.w2(F.gelu(self.w0(x), approximate="tanh") * self.w1(x))


class CausalAttention(nn.Module):
    """Causal self-attention that narrows ``in_dim`` to ``out_dim``.

    One bias-less ``qkv`` Linear plus separate ``q_bias``/``zero_k_bias``/``v_bias``
    under the file's names. ``zero_k_bias`` is a PERSISTENT buffer: a
    non-persistent one is left out of ``load_state_dict`` and would stay on
    meta (``sft_load`` then refuses the model). The heads are averaged away
    (not concatenated) and the remaining head width is pooled down to
    ``out_dim`` with ``adaptive_avg_pool1d``.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.empty(in_dim), requires_grad=False)
        self.v_bias = nn.Parameter(torch.empty(in_dim), requires_grad=False)
        self.register_buffer("zero_k_bias", torch.empty(in_dim), persistent=True)
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = q + self.q_bias, k + self.zero_k_bias, v + self.v_bias
        # "b s (n d)" -> [b, n, s, d]
        q, k, v = (t.reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # [b, n, s, d]
        x = x.mean(dim=1)                                              # [b, s, d]
        x = F.adaptive_avg_pool1d(x, self.out_dim)                     # [b, s, out_dim]
        return self.proj(x)


class AttnProjection(nn.Module):
    """``pre_block``: ``x = proj(norm3(x)) + attn(norm1(x))``; ``x = x + mlp(norm2(x))``; [B, T, in] -> [B, T, out]."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(out_dim, out_dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class H3AudioEncoder(nn.Module):
    """Encoder half of the audio VAE: waveform [2, L] -> normalised latents [2, 32, ceil(L / 800)].

    Use :meth:`from_file`; afterwards ``load_report`` holds the
    :class:`engine_h3.sft_load.LoadReport` of the read (``loaded`` should be
    :data:`ENCODER_TENSOR_COUNT`). The constructor's sizes default to the
    file's; smaller ones exist for CPU shape tests only.
    """

    def __init__(self, *, encoder_dim: int = 64, strides: tuple[int, ...] = (2, 4, 4, 5, 5),
                 latent_channels: int = LATENT_CHANNELS, num_heads: int = 8):
        super().__init__()
        latent_dim = encoder_dim * 2 ** len(strides)
        self.sample_rate = C.AUDIO_SAMPLE_RATE
        self.hop_length = math.prod(strides)
        self.encoder = DACEncoder(encoder_dim, tuple(strides))
        self.pre_block = AttnProjection(latent_dim, latent_channels, num_heads=num_heads)
        self.mean_proj = nn.Conv1d(latent_channels, latent_channels, 1)
        self.register_buffer("latents_mean", torch.zeros(latent_channels, dtype=torch.float32), persistent=False)
        self.register_buffer("latents_std", torch.ones(latent_channels, dtype=torch.float32), persistent=False)
        self.load_report = None

    @classmethod
    def from_file(cls, path: str, *, device) -> "H3AudioEncoder":
        import sft_quant_format
        from engine_h3.sft_load import load_streaming

        header = sft_quant_format.read_header(path)
        mean, std = latent_stats_from_metadata(header.metadata, METADATA_KEY, LATENT_CHANNELS)
        with torch.device("meta"):
            model = cls()
        model.latents_mean = torch.tensor(mean, dtype=torch.float32)
        model.latents_std = torch.tensor(std, dtype=torch.float32)
        report = load_streaming(model, path, header, key_map=audio_encoder_key_map)
        model.requires_grad_(False).eval()
        # The file is F32 already; .float() only guards against a future F16 copy.
        model = model.to(device=device).float()
        model.load_report = report
        return model

    @torch.no_grad()
    def encode(self, wave: torch.Tensor) -> torch.Tensor:
        """Waveform [2, L] float32 at 32 kHz -> normalised latents [2, 32, ceil(L / 800)] float32.

        The channels run as a batch of mono signals. The waveform is padded on
        the right with zeros to a multiple of the hop, the posterior mean is
        used (no sampling), and ``(z - mean) / std`` is what ``decode`` undoes.
        """
        if wave.ndim != 2 or wave.shape[1] == 0:
            raise ValueError(f"audio encode: [2, L]（L > 0）を期待しましたが {tuple(wave.shape)} でした")
        x = wave.to(device=self.latents_mean.device, dtype=torch.float32).unsqueeze(1)   # [2, 1, L]
        right_pad = math.ceil(x.shape[-1] / self.hop_length) * self.hop_length - x.shape[-1]
        if right_pad:
            x = F.pad(x, (0, right_pad))
        z = self.encoder(x)                                        # [2, 2048, T]
        z = self.pre_block(z.transpose(1, 2)).transpose(1, 2)      # [2, 32, T]
        z = self.mean_proj(z)
        mean = self.latents_mean.view(1, -1, 1)
        std = self.latents_std.view(1, -1, 1)
        return (z.float() - mean) / std
