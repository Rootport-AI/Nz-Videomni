# Vendored from https://github.com/kangben258/UETrack @ fd13b0ea
# Source file: lib/test/utils/hann.py  (the two functions the tracker calls, only)
#
# NZ: upstream's hann.py also carries hann2d_bias / hann2d_clipped / the gaussian
# label helpers / max2d, none of which the UETrack tracker reaches. The window
# itself is NOT a user setting in this project (owner's ruling: "Hanning window
# is fixed internally"), so hann2d is called once at load time with the search
# feature grid size and never again.

import math

import torch


def hann1d(sz: int, centered = True) -> torch.Tensor:
    """1D cosine window."""
    if centered:
        return 0.5 * (1 - torch.cos((2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))
    w = 0.5 * (1 + torch.cos((2 * math.pi / (sz + 2)) * torch.arange(0, sz//2 + 1).float()))
    return torch.cat([w, w[1:sz-sz//2].flip((0,))])


def hann2d(sz: torch.Tensor, centered = True) -> torch.Tensor:
    """2D cosine window."""
    return hann1d(sz[0].item(), centered).reshape(1, 1, -1, 1) * hann1d(sz[1].item(), centered).reshape(1, 1, 1, -1)
