"""Standalone self-check for engine/preprocess (driver dispatch + depth helpers).

Run with the ENGINE venv (needs cv2; the app venv has neither cv2 nor torch, and
.venv-engine has no fastapi so tests/conftest.py cannot be collected there —
which is why this is a script and not a pytest module, the same split
block_swap_prefetch_selfcheck.py uses):

    .venv-engine\\Scripts\\python.exe -m engine.preprocess.preprocess_selfcheck

Same conventions as the transformer self-checks: every check either PASSes or
FAILs loudly (nothing is skipped), exit code 0 only when all of them pass. No
GPU and no model weights are needed — the driver is exercised with fake
processors, and only ``DepthProcessor``'s pure array helpers are called. Real
VDA inference is the G1 gate (see depth_g1_gate.py), not this script.

What the checks prove, in one line each:

  C1  get_processor resolves canny / dwpose / depth, and rejects "none"/unknown.
  C2  Only the depth processor is a VideoProcessor; canny/dwpose stay
      FrameProcessors, so the driver's dispatch cannot mis-route them.
  C3  The whole-clip branch hands the processor EVERY frame in ONE call, in
      order, and writes exactly as many frames back.
  C4  frame_cap truncates the whole-clip branch from the head of the source.
  C5  frame_cap=None (what canny/dwpose always get) decodes the whole source,
      and the frame branch still sees each frame exactly once, in order.
  C6  FPS and resolution survive both branches.
  C7  release() runs after both branches, and after a failure too.
  C8  A whole-clip processor that returns the wrong frame count fails loud.
  C9  DepthProcessor._inference_size: <=960 is untouched, longer is scaled with
      the aspect preserved and both edges even.
  C10 DepthProcessor._to_control_frames: near=white, clip-wide (not per-frame)
      normalisation, 3 identical channels, uint8, resized back to the source.
  C11 The checkpoint the wrapper expects is where the wrapper looks for it.
"""

from __future__ import annotations

import gc
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from engine.preprocess import get_processor, preprocess_video
from engine.preprocess.base import FrameProcessor, VideoProcessor
from engine.preprocess.canny import CannyProcessor
from engine.preprocess.depth import _CHECKPOINT_PATH, DepthProcessor
from engine.preprocess.driver import _FACTORIES
from engine.preprocess.dwpose import DwposeProcessor

_RESULTS: list[tuple[str, bool, str]] = []

_W, _H, _FPS, _N = 64, 48, 12.0, 20


def _record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    _RESULTS.append((name, passed, detail))


def _run_check(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        _record(name, True)
    except Exception as exc:  # noqa: BLE001 — a broken check must FAIL loudly, never skip
        _record(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        gc.collect()


# --------------------------------------------------------------------- fixtures


def _write_source(path: Path) -> None:
    """A _N-frame clip whose frame i is the solid grey (i*10, i*10, i*10)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), _FPS, (_W, _H))
    if not writer.isOpened():
        raise RuntimeError("selfcheck: cannot open the source mp4 writer")
    try:
        for i in range(_N):
            writer.write(np.full((_H, _W, 3), i * 10, dtype=np.uint8))
    finally:
        writer.release()


def _probe(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = 0
        while cap.read()[0]:
            n += 1
    finally:
        cap.release()
    return n, fps, width, height


class _RecordingVideoProcessor:
    """VideoProcessor that records what it was handed and echoes it back."""

    def __init__(self) -> None:
        self.calls: list[list[np.ndarray]] = []
        self.released = 0

    def process_video(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        self.calls.append(list(frames_bgr))
        return [f.copy() for f in frames_bgr]

    def release(self) -> None:
        self.released += 1


class _RecordingFrameProcessor:
    """FrameProcessor that records every frame it was handed, in order."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = 0

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        self.frames.append(frame_bgr.copy())
        return frame_bgr

    def release(self) -> None:
        self.released += 1


class _BadCountVideoProcessor:
    def process_video(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        return list(frames_bgr)[:-1]


class _ExplodingFrameProcessor:
    def __init__(self) -> None:
        self.released = 0

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        raise RuntimeError("boom")

    def release(self) -> None:
        self.released += 1


# ----------------------------------------------------------------------- checks


def check_c1_registry() -> None:
    assert sorted(_FACTORIES) == ["canny", "depth", "dwpose"], sorted(_FACTORIES)
    assert isinstance(get_processor("canny"), CannyProcessor)
    assert isinstance(get_processor("dwpose"), DwposeProcessor)
    assert isinstance(get_processor("depth"), DepthProcessor)
    # Cached: the same instance comes back (the worker must not rebuild per job).
    assert get_processor("depth") is get_processor("depth")
    for bad in ("none", "", "canny2", "Depth"):
        try:
            get_processor(bad)
        except ValueError:
            continue
        raise AssertionError(f"get_processor({bad!r}) did not fail loud")


def check_c2_protocol_dispatch() -> None:
    depth = get_processor("depth")
    assert isinstance(depth, VideoProcessor), "depth must take the whole-clip branch"
    assert not isinstance(depth, FrameProcessor), "depth must not expose process()"
    for kind in ("canny", "dwpose"):
        proc = get_processor(kind)
        assert isinstance(proc, FrameProcessor), kind
        assert not isinstance(proc, VideoProcessor), (
            f"{kind} gained process_video -- the driver would silently re-route it"
        )


def check_c3_video_branch_single_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "src.mp4", Path(tmp) / "dst.mp4"
        _write_source(src)
        proc = _RecordingVideoProcessor()
        n = preprocess_video(src, dst, proc)
        assert n == _N, n
        assert len(proc.calls) == 1, f"expected ONE whole-clip call, got {len(proc.calls)}"
        assert len(proc.calls[0]) == _N, len(proc.calls[0])
        # Order preserved: frame i is brighter than frame i-1 by construction.
        means = [float(f.mean()) for f in proc.calls[0]]
        assert all(a < b for a, b in zip(means, means[1:])), means
        assert _probe(dst)[0] == _N


def check_c4_video_branch_frame_cap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "src.mp4", Path(tmp) / "dst.mp4"
        _write_source(src)
        cap = 7
        proc = _RecordingVideoProcessor()
        n = preprocess_video(src, dst, proc, frame_cap=cap)
        assert n == cap, n
        assert len(proc.calls) == 1 and len(proc.calls[0]) == cap
        # From the HEAD of the source, not an arbitrary window.
        assert float(proc.calls[0][0].mean()) < float(proc.calls[0][-1].mean())
        assert _probe(dst)[0] == cap
        # A cap larger than the source is not an error; the source ends first.
        proc2 = _RecordingVideoProcessor()
        assert preprocess_video(src, dst, proc2, frame_cap=_N + 50) == _N


def check_c5_frame_branch_unchanged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "src.mp4", Path(tmp) / "dst.mp4"
        _write_source(src)
        proc = _RecordingFrameProcessor()
        # No frame_cap argument at all: exactly how the worker calls canny/dwpose.
        n = preprocess_video(src, dst, proc)
        assert n == _N, n
        assert len(proc.frames) == _N, len(proc.frames)
        means = [float(f.mean()) for f in proc.frames]
        assert all(a < b for a, b in zip(means, means[1:])), means
        # The parameter is generic, not depth-only plumbing.
        proc2 = _RecordingFrameProcessor()
        assert preprocess_video(src, dst, proc2, frame_cap=3) == 3
        assert len(proc2.frames) == 3


def check_c6_fps_and_resolution_preserved() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.mp4"
        _write_source(src)
        for label, proc in (
            ("video", _RecordingVideoProcessor()),
            ("frame", _RecordingFrameProcessor()),
        ):
            dst = Path(tmp) / f"dst_{label}.mp4"
            preprocess_video(src, dst, proc, frame_cap=None)
            n, fps, width, height = _probe(dst)
            assert (width, height) == (_W, _H), (label, width, height)
            assert abs(fps - _FPS) < 0.01, (label, fps)
            assert n == _N, (label, n)


def check_c7_release_always_runs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "src.mp4", Path(tmp) / "dst.mp4"
        _write_source(src)

        video_proc = _RecordingVideoProcessor()
        preprocess_video(src, dst, video_proc)
        assert video_proc.released == 1, video_proc.released

        frame_proc = _RecordingFrameProcessor()
        preprocess_video(src, dst, frame_proc)
        assert frame_proc.released == 1, frame_proc.released

        # A processor blowing up mid-clip must still hand the GPU back.
        boom = _ExplodingFrameProcessor()
        try:
            preprocess_video(src, dst, boom)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a failing processor did not propagate")
        assert boom.released == 1, boom.released

        # Canny has no release() at all -- the driver must not require one.
        preprocess_video(src, dst, CannyProcessor())


def check_c8_frame_count_mismatch_fails_loud() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "src.mp4", Path(tmp) / "dst.mp4"
        _write_source(src)
        try:
            preprocess_video(src, dst, _BadCountVideoProcessor())
        except RuntimeError as exc:
            assert "frames" in str(exc), str(exc)
            return
        raise AssertionError("a short whole-clip result was accepted silently")


def check_c9_inference_size() -> None:
    proc = DepthProcessor()
    # At or below the cap: untouched (None == "do not resize").
    assert proc._inference_size(960, 540) is None
    assert proc._inference_size(512, 256) is None
    assert proc._inference_size(540, 960) is None
    # Above the cap: long edge lands on 960, aspect preserved, both edges even.
    for width, height in ((1920, 1088), (1280, 768), (1088, 1920), (1000, 999)):
        size = proc._inference_size(width, height)
        assert size is not None, (width, height)
        new_w, new_h = size
        assert max(new_w, new_h) == 960, (width, height, size)
        assert new_w % 2 == 0 and new_h % 2 == 0, (width, height, size)
        assert abs((new_w / new_h) - (width / height)) < 0.01, (width, height, size)


def check_c10_control_frames() -> None:
    proc = DepthProcessor()
    # Frame 0 spans the clip's range; frame 1 is uniformly mid-range. With a
    # clip-wide normalisation frame 1 must come out uniformly mid-grey -- a
    # per-frame normalisation would blow it up to 0 or 255 and flicker.
    depths = np.stack(
        [
            np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32),
            np.full((2, 2), 2.0, dtype=np.float32),
        ]
    )
    frames = proc._to_control_frames(depths, width=2, height=2)
    assert len(frames) == 2
    for frame in frames:
        assert frame.dtype == np.uint8 and frame.shape == (2, 2, 3), frame.shape
        assert (frame[:, :, 0] == frame[:, :, 1]).all()
        assert (frame[:, :, 1] == frame[:, :, 2]).all()
    gray0 = frames[0][:, :, 0]
    # near = large depth = white; far = small depth = black. No inversion.
    assert gray0[0, 0] == 0 and gray0[1, 1] == 255, gray0
    assert gray0[0, 1] < gray0[1, 0] < gray0[1, 1], gray0
    gray1 = frames[1][:, :, 0]
    assert len(np.unique(gray1)) == 1, "frame 1 should stay flat"
    assert 120 <= int(gray1[0, 0]) <= 135, int(gray1[0, 0])
    assert int(gray1[0, 0]) == int(gray0[1, 0]), "same depth must map to the same gray"

    # Resized back to the SOURCE resolution, not the inference resolution.
    big = proc._to_control_frames(depths, width=8, height=6)
    assert all(f.shape == (6, 8, 3) for f in big), [f.shape for f in big]

    # A flat clip must not divide by zero.
    flat = proc._to_control_frames(np.zeros((3, 2, 2), dtype=np.float32), 2, 2)
    assert all(int(f.max()) == 0 for f in flat)


def check_c11_checkpoint_location() -> None:
    assert _CHECKPOINT_PATH.name == "video_depth_anything_vits.pth", _CHECKPOINT_PATH
    assert _CHECKPOINT_PATH.parent.name == "preprocessors-vda", _CHECKPOINT_PATH
    assert _CHECKPOINT_PATH.exists(), f"checkpoint not installed at {_CHECKPOINT_PATH}"
    # release() on a never-loaded processor is a no-op, not an AttributeError.
    DepthProcessor().release()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    checks: list[tuple[str, Callable[[], None]]] = [
        ("C1  get_processor resolves canny/dwpose/depth and rejects the rest", check_c1_registry),
        ("C2  only depth is a VideoProcessor (dispatch cannot mis-route)", check_c2_protocol_dispatch),
        ("C3  the whole-clip branch passes every frame in one ordered call", check_c3_video_branch_single_batch),
        ("C4  frame_cap truncates the whole-clip branch from the head", check_c4_video_branch_frame_cap),
        ("C5  the frame branch is unchanged, and frame_cap is generic", check_c5_frame_branch_unchanged),
        ("C6  FPS and resolution survive both branches", check_c6_fps_and_resolution_preserved),
        ("C7  release() runs after success and after failure", check_c7_release_always_runs),
        ("C8  a wrong whole-clip frame count fails loud", check_c8_frame_count_mismatch_fails_loud),
        ("C9  depth inference size: cap 960, aspect kept, even edges", check_c9_inference_size),
        ("C10 depth output: near=white, clip-wide normalisation, 3ch, uint8", check_c10_control_frames),
        ("C11 the checkpoint is where the wrapper looks for it", check_c11_checkpoint_location),
    ]

    for name, fn in checks:
        _run_check(name, fn)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_total = len(_RESULTS)
    print(f"\n{n_pass}/{n_total} checks passed")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
