"""Engine-side raw-video -> control-signal preprocessing (IC-LoRA Phase C).

The Union-Control IC-LoRA does not consume a raw reference video; it expects a
*control signal* video (edge map / skeleton) that is positionally aligned with
the output. This package converts an uploaded raw video into that control video
INSIDE the engine venv (``.venv-engine``), which is the only environment with
cv2 + torch. The app venv (``.venv``) has neither, so preprocessing must live
here and run at generate time (see ``engine/worker.py`` ``_do_generate``).

Design (mirrors — but does not copy — the deleted low-VRAM fork's
``video_processor`` service):

  * ``FrameProcessor`` — a Protocol: one BGR frame in, one BGR control frame out,
    dimensions preserved. Stateless implementations (Canny) and stateful ones
    (a future DWPose that caches TorchScript models in-process, Slice 3) both
    satisfy it.
  * ``CannyProcessor`` — ``cv2.Canny(gray, 100, 200)`` edge map, 1ch -> 3ch.
  * ``preprocess_video`` — the driver: decode a video frame-by-frame, run each
    frame through a ``FrameProcessor``, and re-encode to mp4. FPS / frame count /
    resolution are all preserved (the wheel's ``frame_cap`` truncates to the
    generation length; no resampling or resize happens here).
  * ``get_processor`` — kind ("canny" / future "dwpose") -> cached processor
    instance. Unknown kinds fail loud.
"""

from __future__ import annotations

from engine.preprocess.base import FrameProcessor
from engine.preprocess.canny import CannyProcessor
from engine.preprocess.driver import get_processor, preprocess_video

__all__ = [
    "FrameProcessor",
    "CannyProcessor",
    "get_processor",
    "preprocess_video",
]
