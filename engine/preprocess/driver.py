"""Video-level driver + processor registry for control-signal preprocessing.

``preprocess_video`` streams a source video through a ``FrameProcessor`` and
re-encodes the control signal to mp4, preserving FPS / frame count / resolution
(decode + encode via cv2, ``mp4v`` fourcc — the same approach the fork used and
that the wheel's reference-video loader accepts). A ``VideoProcessor`` (depth)
takes the whole decoded clip in one call instead; everything else — decode,
encode, ``frame_cap``, ``release`` — is identical for both. ``get_processor``
maps a preprocess kind to a cached processor instance and fails loud on unknown
kinds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2

from engine.preprocess.base import FrameProcessor, VideoProcessor
from engine.preprocess.canny import CannyProcessor
from engine.preprocess.dwpose import DwposeProcessor

Processor = FrameProcessor | VideoProcessor


def _make_depth_processor() -> VideoProcessor:
    """Factory for the depth processor (``DepthProcessor``).

    The heavy part of depth, the vendored Video-Depth-Anything tree
    (torchvision + the DINOv2 stack), is imported inside
    ``DepthProcessor._ensure_loaded`` on the first depth job, so a canny/pose
    job does not pay for it and an import failure there does not take
    canny/pose down with it. The import of ``engine.preprocess.depth`` below
    is not where that deferral happens: the package ``__init__`` imports it
    eagerly, and its top level needs only cv2, numpy and torch.
    """
    from engine.preprocess.depth import DepthProcessor

    return DepthProcessor()


# Kind -> zero-arg factory. Canny is stateless; DWPose holds TorchScript models
# it loads lazily on first frame and frees via ``release()`` (see below); Depth
# holds the VDA network and works clip-at-a-time.
_FACTORIES: dict[str, Callable[[], Processor]] = {
    "canny": CannyProcessor,
    "dwpose": DwposeProcessor,
    "depth": _make_depth_processor,
}

# Process-level instance cache: a stateless Canny is cheap; DWPose and Depth
# are built once per worker process, not per job (``release()`` frees their
# weights after each video and the next call reloads them into the same
# instance). Caching here keeps that concern out of the worker.
_CACHE: dict[str, Processor] = {}


def get_processor(kind: str) -> Processor:
    """Return the cached processor for ``kind``; fail loud on unknown.

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


def preprocess_video(
    src: Path,
    dst: Path,
    processor: Processor,
    frame_cap: int | None = None,
) -> int:
    """Convert ``src`` -> ``dst`` through ``processor``.

    Preserves FPS, frame count, and resolution (no resampling / resize — the
    wheel's ``frame_cap`` truncates to the generation length downstream).
    ``frame_cap`` stops the decode after that many frames: a ``VideoProcessor``
    normalises over everything it is given, so handing it footage the generation
    will never use would both cost time and shift the normalisation range.
    ``None`` decodes the whole source (what the workers pass when the payload
    gives no frame count; see each worker's ``_preprocess_frame_cap``).
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
            if isinstance(processor, VideoProcessor):
                frames = []
                while frame_cap is None or len(frames) < frame_cap:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                if not frames:
                    raise RuntimeError(
                        f"preprocess: decoded 0 frames from source: {src}"
                    )
                controls = processor.process_video(frames)
                if len(controls) != len(frames):
                    raise RuntimeError(
                        f"preprocess: processor returned {len(controls)} frames "
                        f"for {len(frames)} input frames: {src}"
                    )
                for control in controls:
                    writer.write(control)
                n = len(controls)
            else:
                n = 0
                while frame_cap is None or n < frame_cap:
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
        # Evict any GPU-resident weights the processor loaded for this video
        # (DWPose's two TorchScript models, Depth's VDA network; resident
        # sizes in VERIFICATION_LOG §22.2 and §49.2) BEFORE the caller runs the
        # 16 GB-tight generation denoise. Optional per the processor protocols:
        # stateless processors (Canny) have no ``release``. The cached
        # processor instance itself survives; its next call reloads.
        release = getattr(processor, "release", None)
        if callable(release):
            release()

    if n == 0:
        raise RuntimeError(f"preprocess: decoded 0 frames from source: {src}")
    return n
