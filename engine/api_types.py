"""``ImageConditioningInput``: the keyframe conditioning triplet the engine passes around."""

from __future__ import annotations

from typing import NamedTuple


class ImageConditioningInput(NamedTuple):
    """Conditioning triplet: keyframe image path, frame index, strength."""

    path: str
    frame_idx: int
    strength: float
