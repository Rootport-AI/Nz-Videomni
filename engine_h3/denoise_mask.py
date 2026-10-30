"""Per-row continuous denoise masks for MiniMax H3 (A2V and the H3 chain; V2A later).

Docs/MINIMAX_H3_DESIGN.md §3.1 is the home of the formulas (台帳 §1-103). The
semantics follow the ComfyUI H3 specification (read as a specification only:
ComfyUI is GPL-3.0, none of its code is used here) and agree with DiffSynth's
Retake at m = 0.

A mask value ``m`` in [0, 1] per row: 0 = keep the row's original latent
(``clean``), 1 = generate it as usual. With the stream's sigma ``s`` (always
> 0 when a step is evaluated) and the DiT's data-direction velocity ``v``
(x0 = x + s v, engine_h3/scheduler.py), every step:

1. DiT input      ``x_in = m x + (1 - m) inject``
   (audio: ``inject = clean``; video: ``c clean + (1 - c) noise`` with the
   row's INITIAL noise, so no new random number is drawn; ``c`` = 0.999 by
   default, ``inject_clean=1.0`` for the chain, 裁定 84-2)
2. row time       ``t = min(1 - m s, cap)`` for m < 1 rows (cap 0.999 video,
   1.0 audio); m = 1 rows keep the stream's t
3. estimate       ``x0_hat = x_in + s (m v)``
4. blend          ``x0_b = m x0_hat + (1 - m) clean``
5. state velocity ``v_s = (x0_b - x) / s``, computed as written (not the
   expanded form) and handed to the unchanged ``euler_step``.

After the loop the m = 0 rows are set to ``clean`` exactly (``finish``).

A mask whose values are all 1 is never built (the builders return None), and
every helper returns its input object itself for ``None``: the no-mask path
adds no tensor operation at all. Everything here is float32.

Granularity (V1): video per latent frame ``[T']`` (one value for the frame's
``grid`` rows), audio per latent ``[A]`` (shared by both channels).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

#: The highest t a masked row gets (ComfyUI's VISUAL_COND_TIMESTEP / AUDIO_COND_TIMESTEP).
VIDEO_TIME_CAP = 0.999
AUDIO_TIME_CAP = 1.0
#: Video rows are fed ``VIDEO_INJECT_CLEAN * clean + (1 - VIDEO_INJECT_CLEAN) * noise``.
VIDEO_INJECT_CLEAN = 0.999


@dataclass(frozen=True)
class StreamMask:
    """One stream's mask, built once per job (after the noise is drawn).

    * ``m``: float32, broadcastable to the state (audio ``[1, 1, A]``, video
      ``[1, 1, T', 1, 1]``).
    * ``rows``: float32 ``[rows]`` in the packed sequence's order (audio:
      channel 0's A values then channel 1's; video: t -> h -> w).
    * ``clean``: the original latent, float32, the state's shape.
    * ``inject``: what the DiT sees for the kept part, float32, the state's shape.
    * ``zero``: bool, broadcastable like ``m``: the rows with m == 0.
    """

    m: torch.Tensor
    rows: torch.Tensor
    clean: torch.Tensor
    inject: torch.Tensor
    zero: torch.Tensor


def _values(per, n: int, what: str, device) -> torch.Tensor:
    """``per`` as a float32 ``[n]`` tensor on ``device``, every value finite and in [0, 1]."""
    values = torch.as_tensor(per, dtype=torch.float32).reshape(-1).to(device)
    if int(values.numel()) != int(n):
        raise ValueError(f"{what} mask: {int(values.numel())} values for {int(n)} latents")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{what} mask: values must be finite")
    if bool((values < 0).any()) or bool((values > 1).any()):
        raise ValueError(f"{what} mask: values must be in [0, 1]")
    return values


def audio_mask(clean: torch.Tensor, per_latent) -> StreamMask | None:
    """Audio ``clean [2, 32, A]`` and one value per latent ``[A]`` -> mask, or None when all are 1.

    The audio rows are channel 0's A latents, then channel 1's
    (``engine_h3.dit.audio_to_rows``), so ``rows`` is the values repeated twice.
    """
    if clean.dim() != 3:
        raise ValueError(f"audio mask: clean must be [channels, 32, A] ({list(clean.shape)})")
    channels, _dim, steps = (int(s) for s in clean.shape)
    values = _values(per_latent, steps, "audio", clean.device)
    if bool((values == 1).all()):
        return None
    clean32 = clean.to(torch.float32)
    m = values.view(1, 1, steps)
    return StreamMask(m=m, rows=values.repeat(channels), clean=clean32, inject=clean32, zero=m == 0)


def video_mask(
    clean: torch.Tensor, per_frame, *, grid: int, noise: torch.Tensor, inject_clean: float | None = None,
) -> StreamMask | None:
    """Video ``clean [1, C, T', H/16, W/16]`` and one value per latent frame ``[T']`` -> mask, or None.

    ``grid``: token rows per latent frame (``(H/32) * (W/32)``); the video rows
    are t -> h -> w (``engine_h3.dit.patchify_video``), so each frame's value
    is repeated ``grid`` times in a row. ``noise``: the job's initial video
    noise (the same draw the state started from); ``inject`` is
    ``c clean + (1 - c) noise``, fixed for the whole job, with ``c`` =
    ``inject_clean`` (finite, in [0, 1]; None = ``VIDEO_INJECT_CLEAN``, 0.999).
    The chain passes 1.0: its fixed rows are the previous clip's own latents
    (Docs/MINIMAX_H3_DESIGN.md 裁定 84-2).
    """
    if clean.dim() != 5 or clean.shape[0] != 1:
        raise ValueError(f"video mask: clean must be [1, C, T', H, W] ({list(clean.shape)})")
    if tuple(noise.shape) != tuple(clean.shape):
        raise ValueError(f"video mask: noise {list(noise.shape)} does not match clean {list(clean.shape)}")
    c = VIDEO_INJECT_CLEAN if inject_clean is None else inject_clean
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not (math.isfinite(c) and 0.0 <= c <= 1.0):
        raise ValueError(f"video mask: inject_clean must be finite and in [0, 1] ({inject_clean!r})")
    c = float(c)
    frames = int(clean.shape[2])
    values = _values(per_frame, frames, "video", clean.device)
    if bool((values == 1).all()):
        return None
    clean32 = clean.to(torch.float32)
    noise32 = noise.to(device=clean.device, dtype=torch.float32)
    inject = c * clean32 + (1.0 - c) * noise32
    m = values.view(1, 1, frames, 1, 1)
    rows = values.repeat_interleave(int(grid))
    return StreamMask(m=m, rows=rows, clean=clean32, inject=inject, zero=m == 0)


def dit_input(x: torch.Tensor, mask: StreamMask | None) -> torch.Tensor:
    """What the DiT is given for the state ``x``: ``m x + (1 - m) inject`` (``x`` itself without a mask)."""
    if mask is None:
        return x
    return mask.m * x.to(torch.float32) + (1.0 - mask.m) * mask.inject


def state_velocity(v: torch.Tensor, x: torch.Tensor, mask: StreamMask | None, sigma: float) -> torch.Tensor:
    """The velocity the Euler step applies to the state: ``(x0_b - x) / sigma`` (``v`` itself without a mask).

    ``v`` is the DiT's output for ``dit_input(x, mask)`` at the stream's
    ``sigma`` (> 0).
    """
    if mask is None:
        return v
    s = float(sigma)
    x32 = x.to(torch.float32)
    x_in = dit_input(x32, mask)
    x0_hat = x_in + s * (mask.m * v.to(torch.float32))
    x0_b = mask.m * x0_hat + (1.0 - mask.m) * mask.clean
    return (x0_b - x32) / s


def finish(x: torch.Tensor, mask: StreamMask | None) -> torch.Tensor:
    """After the last step: the m == 0 rows become ``clean`` exactly (``x`` itself without a mask)."""
    if mask is None:
        return x
    return torch.where(mask.zero, mask.clean, x)


def masked_row_times(t_stream: float, sigma: float, rows: torch.Tensor, cap: float) -> torch.Tensor:
    """float32 ``[rows]``: ``min(1 - m sigma, cap)`` on the m < 1 rows, the stream's t on the m == 1 rows."""
    rows = rows.to(torch.float32)
    stream = torch.full(rows.shape, float(t_stream), dtype=torch.float32, device=rows.device)
    masked = (1.0 - rows * float(sigma)).clamp(max=float(cap))
    return torch.where(rows < 1, masked, stream)
