"""MiniMax H3 audio VAE, decoder half: 32-channel latents -> 32 kHz waveform (BigVGAN).

Origin and licence
------------------
Ported from DiffSynth-Studio ``diffsynth/models/minimax_h3_audio_vae.py`` and
``diffsynth/utils/state_dict_converters/minimax_h3_audio_vae.py`` at commit
``4e34969`` (https://github.com/modelscope/DiffSynth-Studio), which is
licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Changes from the original: the
weight-normalised convolutions are plain ``nn.Conv1d``/``nn.ConvTranspose1d``
(see below); the anti-aliasing filters are buffers loaded from the file; only
the decoder path is ported; the module is built on the meta device and filled
by :func:`engine_h3.sft_load.load_streaming`. The diffusers port
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

Only the decoder is built (``dec_in_proj`` + ``decoder.*``): T2V and the
first-frame I2V need no audio encode. ``encoder.*``, ``pre_block.*`` (the attention projection
whose ``q_bias``/``v_bias``/``zero_k_bias`` the DiffSynth converter suffixes
with ``.weight``), ``mean_proj.*``, ``logs_proj.*`` and the statistic tensors
are mapped to None by ``audio_vae_key_map`` and never read.

Shapes (R2 F)
-------------
Latents are ``[2, 32, A]``: the two stereo channels run as a batch of 2 through
a mono VAE. ``A = round(F / 24 * 40)``; one latent is 800 samples at 32 kHz, so
the waveform is ``[2, A * 800]`` and can be up to half a latent (12.5 ms)
longer than the video -- :func:`engine_h3.media_out.write_mp4` trims it.
Everything runs in float32.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine_h3 import constants as C
from engine_h3.video_vae import latent_stats_from_metadata

METADATA_KEY = "minimax_h3_audio_vae"
LATENT_CHANNELS = 32

_DROPPED_KEYS = frozenset({"latents_mean", "latents_std"})
_DROPPED_PREFIXES = ("encoder.", "pre_block.", "mean_proj.", "logs_proj.")


def audio_vae_key_map(name: str) -> str | None:
    """File tensor name -> module state_dict key (None = not loaded; encoder side and stats)."""
    if name in _DROPPED_KEYS or name.startswith(_DROPPED_PREFIXES):
        return None
    return name


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
