# Ported from DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio),
# commit 4e34969:
#   diffsynth/models/minimax_h3_dit.py            (_rotate_half, _apply_rope, MiniMaxH3Rope)
#   diffsynth/pipelines/minimax_h3_audio_video.py (MiniMaxH3Unit_PackedSequenceBuilder,
#                                                  the coordinate helpers and _build_packed_fl2va)
# Copyright the DiffSynth-Studio authors. Licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Modified for
# Nz-Videomni: one function per step, no padding rows, cos/sin computed once
# per job instead of per forward. Cross-checked against diffusers
# (modular_pipelines/minimax_h3/before_denoise.py:36-133, Apache-2.0).
"""3-axis rotary position embedding of the MiniMax H3 packed sequence.

Three steps, each its own function so the pipeline can build the tables once
per job (the coordinates depend only on the layout, never on the step):

    build_coords   (t, h, w) per row, float64, in sequence order
    rope_tables    cos / sin float32 [S, 96] from the coordinates and the
                   file's ``rope.inv_freq`` (16 values shared by the 3 axes)
    apply_rope     rotate the first 96 of each head's 128 channels

Why float64 coordinates: the latent-frame spacing is non-uniform
(5/3 x (1, 4, 4, 4, 4)) and the spatial axes are an aspect-normalized
``np.linspace(..., endpoint=False)``; both reference implementations build
them in float64 with numpy precisely so the grid is reproduced bit for bit
(``torch.linspace`` computes a different sequence). The angles are float32,
as in both references (``pos.to(torch.float32)`` before the product).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

#: One latent frame spans 5/3 x frames_per_latent rotary units; the (1,4,4,4,4)
#: period mirrors the VAE's 17-frame -> 5-latent grouping.
_FRAME_RESCALE = 5.0 / 3.0
_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
#: The aspect-normalized spatial axes are scaled up by this.
_SPATIAL_SCALE = 32
#: Patch size of the DiT over the VAE latent (h, w).
_PATCH_HW = 2
#: Audio rows per latent step: two mono VAE channels (left, right).
_AUDIO_CHANNELS = 2


def _spatial_axis(latent_dim: int, sqrt_area: float) -> torch.Tensor:
    """One spatial axis: ``latent_dim // 2`` coordinates, centred, x32.

    ``latent_dim`` is the VAE latent size (H/16 or W/16), exactly as the
    references pass it; the ratio uses the latent area, not the token grid.
    """
    ratio = latent_dim / sqrt_area
    left = (1.0 - ratio) * 0.5
    right = left + ratio
    grid = np.linspace(left, right, latent_dim // _PATCH_HW, endpoint=False) * _SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _frame_grid(latent_h: int, latent_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(h, w) of one latent frame's rows [grid_h * grid_w, 2] (row-major), and
    the w axis it was built from (audio rows sit on its two ends)."""
    sqrt_area = np.sqrt(latent_h * latent_w)
    h_grid = _spatial_axis(latent_h, sqrt_area)
    w_grid = _spatial_axis(latent_w, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_grid


def _video_t_grid(n: int, origin: float) -> torch.Tensor:
    """Rotary time of each of ``n`` latent frames, starting at ``origin``."""
    spans = torch.tensor(
        [_FRAME_RESCALE * _FRAMES_PER_LATENT[k % len(_FRAMES_PER_LATENT)] for k in range(n)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def keyframe_t(frame_idx: int, latent_frames: int, text_len: int) -> float:
    """Rotary time of a keyframe's condition rows: pixel frame ``frame_idx``
    of a ``latent_frames``-latent video, after ``text_len`` text rows.

    The time follows the PIXEL frame (5/3 per frame), not a latent-frame
    boundary, so any ``0 <= frame_idx <= F-1`` has one
    (``F = (T'-2)//5*17 + 5``, the inverse of ``constants.latent_frames``).

    * ``0``:   ``L`` (the first video latent's time).
    * ``F-1``: ``(L + span) - 5/3``, span = the float64 sum of the latent
      spacing 5/3 x (1,4,4,4,4) over all T' latents — DiffSynth
      ``_build_packed_fl2va``'s last-frame anchor, written in the same order
      so it matches to the last bit. ``L + (5/3)(F-1)`` is the same number in
      exact arithmetic but NOT in float64 for about 4 in 10 (F, L) pairs
      (tests/test_h3_engine.py brute-forces it), so this branch stays.
    * otherwise: ``L + (5/3) * frame_idx`` (only ComfyUI places these; no
      bit-level reference exists).
    """
    if latent_frames < 2 or (latent_frames - 2) % 5 != 0:
        raise ValueError(f"keyframe_t: latent_frames は 5n+2 が必要です（{latent_frames}）")
    num_frames = (latent_frames - 2) // 5 * 17 + 5
    if not 0 <= frame_idx <= num_frames - 1:
        raise ValueError(
            f"keyframe_t: frame_idx は 0〜{num_frames - 1} が必要です（{frame_idx}）"
        )
    L = float(text_len)
    if frame_idx == 0:
        return L
    if frame_idx == num_frames - 1:
        spans = np.ones(latent_frames, dtype=np.float64) * _FRAME_RESCALE
        for k, per in enumerate(_FRAMES_PER_LATENT):
            spans[k::len(_FRAMES_PER_LATENT)] *= per
        return (L + float(spans.sum())) - _FRAME_RESCALE
    return L + _FRAME_RESCALE * frame_idx


def build_coords(
    *,
    text_len: int,
    latent_frames: int,
    grid_h: int,
    grid_w: int,
    audio_latents: int,
    cond_frame_idx: Sequence[int] = (),
) -> torch.Tensor:
    """Rotary coordinates [S, 3] float64 (t, h, w), rows in sequence order
    ``[text | cond | audio | video]``.

    ``grid_h`` / ``grid_w`` are the TOKEN grid (H/32, W/32 = latent / 2).

    * text  (L rows):  t = 0 .. L-1, h = w = 0.
    * cond  (one block of grid rows per keyframe, I2V, in the order of
      ``cond_frame_idx`` — the pixel frame of each; the pipeline passes them
      ascending): t = :func:`keyframe_t` of that frame, (h, w) = the video
      grid. The first-frame block matches DiffSynth float64-exactly
      (VERIFICATION_LOG §165 G4 A, §166 2a-G3).
    * audio (2A rows, ch0 then ch1): t = L + k, h = 0, w = min of the w grid
      for ch0, max for ch1.
    * video (T' x grid rows, t -> h -> w): t starts at L with the
      5/3 x (1,4,4,4,4) spacing; (h, w) the aspect-normalized grid.

    DiffSynth ``_build_packed_fl2va`` (pipelines/minimax_h3_audio_video.py:
    649-709) and diffusers ``before_denoise.py:36-133`` build the same grid.
    """
    if text_len < 0 or latent_frames < 1 or grid_h < 1 or grid_w < 1 or audio_latents < 0:
        raise ValueError(
            "build_coords: text_len>=0, latent_frames>=1, grid_h>=1, grid_w>=1, audio_latents>=0 "
            f"が必要です（{text_len}, {latent_frames}, {grid_h}, {grid_w}, {audio_latents}）"
        )
    if len(set(cond_frame_idx)) != len(cond_frame_idx):
        raise ValueError(f"build_coords: キーフレームの位置が重複しています（{list(cond_frame_idx)}）")
    cond_t = [keyframe_t(n, latent_frames, text_len) for n in cond_frame_idx]
    latent_h, latent_w = grid_h * _PATCH_HW, grid_w * _PATCH_HW
    frame_rows = grid_h * grid_w
    cond_rows = len(cond_t) * frame_rows
    audio_rows = _AUDIO_CHANNELS * audio_latents
    video_rows = latent_frames * frame_rows
    total = text_len + cond_rows + audio_rows + video_rows

    g = torch.zeros(total, 3, dtype=torch.float64)
    L = float(text_len)
    g[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)

    frame, w_grid = _frame_grid(latent_h, latent_w)

    start = text_len
    for k, t in enumerate(cond_t):
        cond = g[start + k * frame_rows:start + (k + 1) * frame_rows]
        cond[:, 0] = t
        cond[:, 1:] = frame
    start += cond_rows

    if audio_rows:
        audio = g[start:start + audio_rows]
        audio[:, 0] = (L + torch.arange(audio_latents, dtype=torch.float64)).repeat(_AUDIO_CHANNELS)
        audio[:, 2] = torch.cat(
            [
                torch.full((audio_latents,), float(w_grid[0]), dtype=torch.float64),
                torch.full((audio_latents,), float(w_grid[-1]), dtype=torch.float64),
            ]
        )
    start += audio_rows

    video = g[start:start + video_rows].view(latent_frames, frame_rows, 3)
    video[:, :, 0] = _video_t_grid(latent_frames, L)[:, None]
    video[:, :, 1:] = frame[None]
    return g


def rope_tables(coords: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin float32 [S, 6 * len(inv_freq)] (= [S, 96]) on ``inv_freq``'s device.

    Angles per axis are ``pos * inv_freq`` in float32; the three axes are
    concatenated (t, h, w) and the block is repeated twice so ``rotate_half``
    pairs channel c with c + 48 (DiffSynth ``MiniMaxH3Rope.forward``).
    """
    if coords.dim() != 2 or coords.shape[1] != 3:
        raise ValueError(f"rope_tables: coords は [S, 3] が必要です（{list(coords.shape)}）")
    inv = inv_freq.to(torch.float32).reshape(1, 1, -1)
    pos = coords.to(device=inv.device, dtype=torch.float32)
    per_axis = pos.unsqueeze(-1) * inv  # [S, 3, F]
    t_f, h_f, w_f = per_axis.unbind(dim=1)
    half = torch.cat((t_f, h_f, w_f), dim=-1)
    freqs = torch.cat((half, half), dim=-1)
    return freqs.cos(), freqs.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [S, heads, head_dim]: rotate the first ``cos.shape[-1]`` channels, pass
    the rest through. cos/sin are cast to x's dtype first, as in DiffSynth
    ``_apply_rope`` (the product runs at the block stack's precision)."""
    rot_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    c = cos.to(x.dtype).unsqueeze(1)
    s = sin.to(x.dtype).unsqueeze(1)
    x_rot = (x_rot * c) + (_rotate_half(x_rot) * s)
    return torch.cat((x_rot, x_pass), dim=-1)
