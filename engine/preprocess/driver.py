"""Video-level driver + processor registry for control-signal preprocessing.

``preprocess_video`` streams a source video through a ``FrameProcessor`` and
re-encodes the control signal to mp4, preserving FPS / frame count / resolution
(decode + encode via cv2, ``mp4v`` fourcc — the same approach the fork used and
that the wheel's reference-video loader accepts). ``get_processor`` maps a
preprocess kind to a cached processor instance and fails loud on unknown kinds.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from engine.preprocess.base import FrameProcessor
from engine.preprocess.canny import CannyProcessor

# Kind -> zero-arg factory. Slice 3 registers ``"dwpose": DwposeProcessor`` here.
_FACTORIES: dict[str, "type"] = {
    "canny": CannyProcessor,
}

# Process-level instance cache: a stateless Canny is cheap, but a future DWPose
# holds TorchScript models and must be built once per worker process, not per
# job. Caching here keeps that concern out of the worker.
_CACHE: dict[str, FrameProcessor] = {}


def get_processor(kind: str) -> FrameProcessor:
    """Return the cached ``FrameProcessor`` for ``kind``; fail loud on unknown.

    ``"none"`` is a caller error (no preprocessing should be requested) and is
    rejected here alongside any unregistered kind so a bad payload fails the job
    loudly rather than silently passing the raw video through.
    """
    cached = _CACHE.get(kind)
    if cached is not None:
        return cached
    factory = _FACTORIES.get(kind)
    if factory is None:
        raise ValueError(
            f"unknown preprocess kind: {kind!r} (known: {sorted(_FACTORIES)})"
        )
    proc = factory()
    _CACHE[kind] = proc
    return proc


def preprocess_video(src: Path, dst: Path, processor: FrameProcessor) -> int:
    """Convert ``src`` -> ``dst`` frame-by-frame through ``processor``.

    Preserves FPS, frame count, and resolution (no resampling / resize — the
    wheel's ``frame_cap`` truncates to the generation length downstream).
    Returns the number of frames written. Fails loud if the source cannot be
    opened or the writer cannot be created.
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"preprocess: cannot open source video: {src}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"preprocess: source has invalid dimensions {width}x{height}: {src}"
            )

        dst.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"preprocess: cannot open mp4 writer: {dst}")
        try:
            n = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                control = processor.process(frame)
                writer.write(control)
                n += 1
        finally:
            writer.release()
    finally:
        cap.release()

    if n == 0:
        raise RuntimeError(f"preprocess: decoded 0 frames from source: {src}")
    return n
