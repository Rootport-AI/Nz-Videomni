"""``FrameProcessor`` protocol — the contract every control-signal converter obeys.

A processor maps ONE decoded video frame to ONE control-signal frame of the same
spatial dimensions. Keeping this a single-frame contract lets the video driver
(``driver.preprocess_video``) own all decode/encode + logging concerns, so a new
control type (e.g. the DWPose skeleton in Slice 3) only has to implement
``process`` — regardless of whether it is stateless (Canny) or holds cached
TorchScript models across calls.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameProcessor(Protocol):
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convert one BGR frame (H, W, 3) to a BGR control frame (H, W, 3).

        The output MUST have the same height and width as the input (the driver
        preserves resolution end-to-end; 64-multiple alignment is handled on the
        generation side, not here).
        """
        ...


@runtime_checkable
class VideoProcessor(Protocol):
    """The whole-clip counterpart of ``FrameProcessor``.

    Some control signals cannot be produced one frame at a time: the depth
    processor runs a temporal sliding window and normalises over the whole clip,
    so per-frame calls would be both wrong and slower. Such a processor
    implements ``process_video`` INSTEAD of ``process``; the driver dispatches on
    which of the two is present, and everything else (decode, encode, FPS,
    ``frame_cap``, ``release``) stays in the driver exactly as before.
    """

    def process_video(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Convert a whole BGR clip to a BGR control clip.

        The returned list MUST have the same length as the input and each frame
        MUST keep the input's height and width (same contract as
        ``FrameProcessor.process``, applied clip-wide).
        """
        ...
