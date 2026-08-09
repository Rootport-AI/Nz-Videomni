"""``video_io.cut_window_mp4`` — the retake window cutter.

The engine treats the window's frame count as load-bearing geometry (8n+1, one
stage-2 tile), so this cutter must be frame-EXACT or fail loudly. These tests
run real ffmpeg (same skip discipline as tests/test_video_io.py).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from services import video_io

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def _make(path, *, fps, seconds, frames, audio=True):
    cmd = [shutil.which("ffmpeg"), "-y",
           "-f", "lavfi", "-i", f"testsrc2=size=128x64:rate={fps}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i",
                f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr", "-r", str(fps)]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def test_same_fps_window_is_frame_exact(tmp_path):
    src = _make(tmp_path / "src.mp4", fps=24, seconds=10, frames=241)
    out = tmp_path / "win.mp4"
    info = video_io.cut_window_mp4(src, out, window_start_sec=1.0, num_frames=169, fps=24.0)
    assert info["written_frames"] == 169 == video_io.frame_count(out)
    assert info["resampled"] is False
    assert info["start_frame"] == 24
    assert info["has_audio"] is True


def test_resampled_source_is_still_frame_exact(tmp_path):
    src = _make(tmp_path / "src30.mp4", fps=30, seconds=10, frames=301)
    out = tmp_path / "win.mp4"
    info = video_io.cut_window_mp4(src, out, window_start_sec=1.0, num_frames=73, fps=24.0)
    assert info["resampled"] is True
    assert info["source_fps"] == pytest.approx(30.0, abs=0.01)
    # The count is MEASURED after the resample, not predicted from the request.
    assert info["written_frames"] == 73 == video_io.frame_count(out)


def test_window_running_past_the_end_raises(tmp_path):
    src = _make(tmp_path / "short.mp4", fps=24, seconds=4, frames=97)
    with pytest.raises(video_io.FFmpegError, match="runs past the source"):
        video_io.cut_window_mp4(src, tmp_path / "w.mp4",
                                window_start_sec=3.0, num_frames=169, fps=24.0)


def test_silent_source_reports_has_audio_false(tmp_path):
    src = _make(tmp_path / "mute.mp4", fps=24, seconds=8, frames=193, audio=False)
    out = tmp_path / "win.mp4"
    info = video_io.cut_window_mp4(src, out, window_start_sec=0.0, num_frames=169, fps=24.0)
    assert info["has_audio"] is False
    assert info["written_frames"] == 169


def test_zero_or_negative_frames_rejected(tmp_path):
    src = _make(tmp_path / "src.mp4", fps=24, seconds=4, frames=97)
    for bad in (0, -8):
        with pytest.raises(video_io.FFmpegError, match="num_frames must be"):
            video_io.cut_window_mp4(src, tmp_path / "w.mp4", 0.0, bad, 24.0)


def test_output_is_cfr_at_the_requested_rate(tmp_path):
    src = _make(tmp_path / "src30.mp4", fps=30, seconds=10, frames=301)
    out = tmp_path / "win.mp4"
    video_io.cut_window_mp4(src, out, window_start_sec=0.0, num_frames=169, fps=24.0)
    assert video_io.probe_fps(out) == pytest.approx(24.0, abs=0.01)


def test_resample_temp_file_is_cleaned_up(tmp_path):
    src = _make(tmp_path / "src30.mp4", fps=30, seconds=10, frames=301)
    out = tmp_path / "win.mp4"
    video_io.cut_window_mp4(src, out, window_start_sec=0.0, num_frames=73, fps=24.0)
    assert not (tmp_path / "win_resampled.mp4").exists()
