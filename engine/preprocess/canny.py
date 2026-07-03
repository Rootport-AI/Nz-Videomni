"""Canny edge-map control-signal processor (IC-LoRA Phase C, Slice 2).

Faithful re-implementation of the deleted low-VRAM fork's ``apply_canny``
(``vendor/LTX-Desktop-LOW-VRAM/backend/services/video_processor/video_processor_impl.py``
at commit ``d0d3df5^``, ~L44): pad each frame up to a 64-multiple with edge
replication (matches the training-time flow), run ``cv2.Canny(gray, 100, 200)``,
crop back to the original size, and expand the single-channel edge map to 3
channels. Thresholds (100, 200) are the fork's proven values. This is the
official IC-LoRA default control type and needs no dependency beyond cv2.
"""

from __future__ import annotations

import cv2
import numpy as np


class CannyProcessor:
    """Stateless BGR-frame -> Canny edge-map (3-channel BGR) converter."""

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = frame_bgr.copy()

        # Pad up to a multiple of 64 (edge replication) for parity with the
        # training preprocessing flow, then crop the result back so the output
        # dimensions equal the input dimensions.
        H, W = img.shape[:2]
        H_pad = int(np.ceil(H / 64.0) * 64) - H
        W_pad = int(np.ceil(W / 64.0) * 64) - W
        if H_pad > 0 or W_pad > 0:
            img = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode="edge")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Remove padding.
        edges = edges[:H, :W]

        # HWC3: single-channel edge map -> 3-channel BGR.
        edges_3ch = np.concatenate([edges[:, :, None]] * 3, axis=2)
        return edges_3ch
