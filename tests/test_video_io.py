"""Unit tests for the frame-extraction helpers added for Phase 3 boundary
verification (services/video_io.py: frame_count, extract_frame_at,
extract_last_frame).

Builds a tiny synthetic mp4 out of distinct solid-color frames (via the
production ``encode_frames_to_mp4`` helper) so each frame index has a known,
distinguishable expected color -- this lets us assert exact-frame accuracy,
not just "some frame came back".
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from services import video_io

# 6 visually distinct colors -> 6 frames, each unambiguous by average RGB.
_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


@pytest.fixture()
def color_mp4(tmp_path):
    frames = [Image.new("RGB", (64, 64), c) for c in _COLORS]
    out = tmp_path / "colors.mp4"
    video_io.encode_frames_to_mp4(frames, out, frame_rate=10.0)
    return out


def _avg_rgb(png_path) -> np.ndarray:
    arr = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float64)
    return arr.reshape(-1, 3).mean(axis=0)


def _closest_color_index(avg: np.ndarray) -> int:
    dists = [np.linalg.norm(avg - np.array(c, dtype=np.float64)) for c in _COLORS]
    return int(np.argmin(dists))


def test_frame_count_matches_encoded_frames(color_mp4):
    assert video_io.frame_count(color_mp4) == len(_COLORS)


def test_extract_frame_at_is_exact(color_mp4, tmp_path):
    # For every index, the decoded frame's dominant color must match the
    # frame that was encoded at that index (h264 is lossy, so match by
    # nearest color rather than exact bytes).
    for i in range(len(_COLORS)):
        out_png = tmp_path / f"frame_{i}.png"
        video_io.extract_frame_at(color_mp4, i, out_png)
        assert out_png.exists()
        avg = _avg_rgb(out_png)
        assert _closest_color_index(avg) == i, f"frame {i} decoded to unexpected color {avg}"


def test_extract_frame_at_out_of_range_raises(color_mp4, tmp_path):
    with pytest.raises(video_io.FFmpegError):
        video_io.extract_frame_at(color_mp4, 999, tmp_path / "oob.png")


def test_extract_frame_at_rejects_negative_index(color_mp4, tmp_path):
    with pytest.raises(video_io.FFmpegError):
        video_io.extract_frame_at(color_mp4, -1, tmp_path / "neg.png")


def test_extract_last_frame_matches_final_index(color_mp4, tmp_path):
    last_via_helper = tmp_path / "last.png"
    video_io.extract_last_frame(color_mp4, last_via_helper)

    n = video_io.frame_count(color_mp4)
    last_via_index = tmp_path / "explicit_last.png"
    video_io.extract_frame_at(color_mp4, n - 1, last_via_index)

    avg_helper = _avg_rgb(last_via_helper)
    avg_index = _avg_rgb(last_via_index)
    assert _closest_color_index(avg_helper) == len(_COLORS) - 1
    assert _closest_color_index(avg_helper) == _closest_color_index(avg_index)


# ------------------------------------------------------ V2V source tail helpers


def test_probe_fps_reads_encoded_rate(tmp_path):
    frames = [Image.new("RGB", (64, 64), c) for c in _COLORS]
    out = tmp_path / "r30.mp4"
    video_io.encode_frames_to_mp4(frames, out, frame_rate=30.0)
    fps = video_io.probe_fps(out)
    assert fps is not None and abs(fps - 30.0) < 0.5


def test_cut_tail_mp4_frame_exact_no_resample(color_mp4, tmp_path):
    # 6 distinct color frames @10fps; tail of 3 == the LAST 3 (indices 3,4,5).
    out = tmp_path / "tail3.mp4"
    info = video_io.cut_tail_mp4(color_mp4, out, context_frames=3, fps=10.0)
    assert info["resampled"] is False
    assert video_io.frame_count(out) == 3

    first_png = tmp_path / "t0.png"
    video_io.extract_frame_at(out, 0, first_png)
    assert _closest_color_index(_avg_rgb(first_png)) == 3  # source frame index 3

    last_png = tmp_path / "t2.png"
    video_io.extract_frame_at(out, 2, last_png)
    assert _closest_color_index(_avg_rgb(last_png)) == len(_COLORS) - 1


def test_cut_tail_mp4_resample_path_frame_exact(tmp_path):
    # 30 frames @30fps (=1.0s) resampled to 24fps -> ~24 frames; tail of 9 == 9.
    frames = [Image.new("RGB", (64, 64), (i * 8 % 256, 100, 150)) for i in range(30)]
    src = tmp_path / "src30.mp4"
    video_io.encode_frames_to_mp4(frames, src, frame_rate=30.0)
    out = tmp_path / "tail9.mp4"
    info = video_io.cut_tail_mp4(src, out, context_frames=9, fps=24.0)
    assert info["resampled"] is True
    assert abs(info["source_fps"] - 30.0) < 0.5
    assert video_io.frame_count(out) == 9


def test_cut_tail_mp4_too_short_raises(color_mp4, tmp_path):
    with pytest.raises(video_io.FFmpegError):
        video_io.cut_tail_mp4(color_mp4, tmp_path / "oops.mp4", context_frames=25, fps=10.0)
