"""Shifted flow-match sigma schedule and the Euler step for MiniMax H3.

Ported from DiffSynth-Studio (Apache-2.0), commit 4e34969:
  diffsynth/diffusion/flow_match.py  ``FlowMatchScheduler.set_timesteps_minimax_h3``
Copyright (c) the DiffSynth-Studio authors. Licensed under the Apache License,
Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Changes: reduced to
the two pure functions this engine needs; no scheduler object, no x1000
"timestep" scale (the DiT is fed t = 1 - sigma directly).

The two conventions that are easy to get backwards (R2 (3) items 6-8):

* N steps = N model evaluations. ``base = linspace(1, 0, N+1)[:-1]`` -- the
  end point 0 is NOT one of the evaluated sigmas; it is appended afterwards only
  as the target of the last step. (diffusers counts N points instead and so
  evaluates N-1 times; DiffSynth and SGLang count it this way.)
* The model output is the velocity TOWARDS THE DATA: x0 = x_t + sigma * v.
  So the Euler step is ``x + (sigma - sigma_next) * v`` -- the opposite sign
  of the common flow-match "x - sigma * v" convention. DiffSynth negates the
  model output and then uses the common step; this engine keeps the model's
  own direction and writes the step in that direction instead.

Video and audio run on two different schedules (shift 12 and 3) indexed by the
same step i; never reuse the video sigmas for audio.
"""

from __future__ import annotations

import torch


def sigma_schedule(n_steps: int, shift: float) -> torch.Tensor:
    """float32 ``[n_steps + 1]``: the n evaluated sigmas, then a terminal 0."""
    n = int(n_steps)
    if n < 1:
        raise ValueError(f"ステップ数は 1 以上にしてください（{n_steps}）")
    s = float(shift)
    if not s > 0.0:
        raise ValueError(f"shift は正の値にしてください（{shift}）")
    base = torch.linspace(1.0, 0.0, n + 1, dtype=torch.float32)[:-1]
    sigmas = s * base / (1.0 + (s - 1.0) * base)
    return torch.cat([sigmas, torch.zeros(1, dtype=torch.float32)])


def euler_step(x: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    """One Euler step in the model's direction: ``x + (sigma - sigma_next) * v``.

    Computed in float32 whatever the inputs are (diffusers' scheduler does the
    same) and returned in ``x``'s dtype.
    """
    dt = float(sigma) - float(sigma_next)
    out = x.to(torch.float32) + dt * v.to(torch.float32)
    return out.to(x.dtype)
