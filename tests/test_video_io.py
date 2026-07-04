"""Unit tests for the frame-extraction helpers added for Phase 3 boundary
verification (services/video_io.py: frame_count, extract_frame_at,
extract_last_frame).

Builds a tiny synthetic mp4 out of distinct solid-color frames (via the
production ``encode_frames_to_mp4`` helper) so each frame index has a known,
distinguishable expected color -- this lets us assert exact-frame accuracy,
not just "some frame came back".
"""

from __future__ import annotations

import subprocess
import wave

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


# ------------------------------------------------------------- join_v2v tests


def _make_v2v_clip(
    path,
    num_frames: int,
    fps: float,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    freq: float = 440.0,
    volume_db: float = 0.0,
    with_audio: bool = True,
    sample_rate: int = 48000,
    width: int = 64,
    height: int = 64,
) -> None:
    """Build a tiny synthetic mp4 with a solid-color video track and,
    optionally, a sine-tone audio track at a controlled volume -- via ffmpeg
    ``lavfi`` sources directly (unlike ``encode_frames_to_mp4``, this can
    produce audio, which the join_v2v tests need). ``-frames:v`` pins the
    exact frame count regardless of any rounding in ``-t``.
    """
    exe = video_io.ffmpeg_path()
    duration = num_frames / fps
    color_hex = f"0x{color[0]:02x}{color[1]:02x}{color[2]:02x}"
    cmd = [exe, "-y", "-f", "lavfi", "-i", f"color=c={color_hex}:s={width}x{height}:r={fps}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate={sample_rate}"]
    cmd += ["-t", f"{duration:.6f}"]
    if with_audio:
        cmd += ["-af", f"volume={volume_db}dB", "-c:a", "aac", "-shortest"]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-frames:v", str(num_frames), str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg synthetic v2v clip build failed: {proc.stderr[-2000:]}")


def _extract_wav(mp4, out_wav, sr: int = 48000) -> None:
    exe = video_io.ffmpeg_path()
    cmd = [exe, "-y", "-i", str(mp4), "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", str(out_wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg extract_wav failed: {proc.stderr[-1000:]}")


def _read_wav(wav_path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return samples, sr


def _rms_envelope(samples: np.ndarray, sr: int, center_t: float, half_window_sec: float, win_sec: float = 0.02) -> np.ndarray:
    """Sliding-window RMS envelope in ``[center_t - half_window_sec, center_t
    + half_window_sec]``. Coarse on purpose (20ms windows) -- robust rather
    than sample-exact, since encoder priming/resampling shifts sample
    boundaries by a few samples."""
    win = max(1, int(win_sec * sr))
    lo = max(0, int((center_t - half_window_sec) * sr))
    hi = min(samples.shape[0], int((center_t + half_window_sec) * sr))
    vals = []
    i = lo
    while i + win <= hi:
        seg = samples[i : i + win]
        vals.append(float(np.sqrt(np.mean(seg**2))))
        i += win
    return np.array(vals)


def _mean_volume_db(path, start: float | None = None, duration: float | None = None) -> float:
    exe = video_io.ffmpeg_path()
    cmd = [exe, "-hide_banner"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-af", "volumedetect", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stderr.splitlines():
        if "mean_volume:" in line:
            return float(line.strip().split("mean_volume:")[1].strip().split(" ")[0])
    raise RuntimeError(f"mean_volume not found in ffmpeg volumedetect output: {proc.stderr[-1000:]}")


def test_join_v2v_basic_join_video_and_audio(tmp_path):
    fps = 24.0
    src = tmp_path / "src.mp4"
    cont = tmp_path / "cont.mp4"
    _make_v2v_clip(src, num_frames=48, fps=fps, color=(200, 30, 30), freq=440.0, volume_db=-6.0)
    _make_v2v_clip(cont, num_frames=36, fps=fps, color=(30, 30, 200), freq=880.0, volume_db=-18.0)

    out = tmp_path / "joined.mp4"
    info = video_io.join_v2v(src, cont, out)

    assert out.exists()
    assert video_io.frame_count(out) == 48 + 36

    src_dur = video_io.probe_duration(src)
    cont_dur = video_io.probe_duration(cont)
    out_dur = video_io.probe_duration(out)
    assert abs(out_dur - (src_dur + cont_dur)) < (1.0 / fps) + 1e-3

    assert video_io.has_audio_stream(out)
    assert info["loudness_matched"] is True
    assert info["fade_ms_applied"] == 400
    assert info["source_lufs"] is not None
    assert info["continuation_lufs_before"] is not None


def test_join_v2v_resolution_mismatch_raises(tmp_path):
    fps = 24.0
    src = tmp_path / "src.mp4"
    cont = tmp_path / "cont.mp4"
    _make_v2v_clip(src, num_frames=24, fps=fps, width=64, height=64)
    _make_v2v_clip(cont, num_frames=24, fps=fps, width=96, height=64)
    with pytest.raises(video_io.FFmpegError):
        video_io.join_v2v(src, cont, tmp_path / "joined.mp4")


def test_join_v2v_fade_creates_junction_envelope_dip(tmp_path):
    fps = 24.0
    n = 48  # 2.0s each @24fps
    src = tmp_path / "src.mp4"
    cont = tmp_path / "cont.mp4"
    # Same volume/frequency on both sides: a hard concat is a flat envelope,
    # so any dip at the junction below must come from the fade itself, not a
    # pre-existing amplitude difference between the two clips.
    _make_v2v_clip(src, num_frames=n, fps=fps, freq=440.0, volume_db=-10.0)
    _make_v2v_clip(cont, num_frames=n, fps=fps, freq=440.0, volume_db=-10.0)

    joined = tmp_path / "joined.mp4"
    video_io.join_v2v(src, cont, joined, audio_fade_ms=400, loudness_match=False)

    hard = tmp_path / "hard.mp4"
    video_io.concat_mp4s([src, cont], hard, frame_rate=fps)

    src_dur = video_io.probe_duration(src)

    joined_wav = tmp_path / "joined.wav"
    hard_wav = tmp_path / "hard.wav"
    _extract_wav(joined, joined_wav)
    _extract_wav(hard, hard_wav)
    joined_samples, sr_j = _read_wav(joined_wav)
    hard_samples, sr_h = _read_wav(hard_wav)

    # Envelope in a window straddling the junction (the fade is 400ms/side).
    joined_junction_env = _rms_envelope(joined_samples, sr_j, center_t=src_dur, half_window_sec=0.45)
    hard_junction_env = _rms_envelope(hard_samples, sr_h, center_t=src_dur, half_window_sec=0.45)
    # A steady-state reference window, well away from any junction.
    joined_steady_env = _rms_envelope(joined_samples, sr_j, center_t=src_dur / 2, half_window_sec=0.3)

    assert joined_junction_env.size > 0 and hard_junction_env.size > 0 and joined_steady_env.size > 0

    steady_level = float(np.median(joined_steady_env))
    dip_level = float(np.min(joined_junction_env))
    hard_min = float(np.min(hard_junction_env))

    # join_v2v: fade-out meeting fade-in creates a near-silent notch at the seam.
    assert dip_level < steady_level * 0.5
    # The hard concat has no fade, so its junction-region minimum stays far
    # above the joined clip's notch (same source material, no envelope shaping).
    assert dip_level < hard_min * 0.5


def test_join_v2v_loudness_match_changes_continuation_level(tmp_path):
    fps = 24.0
    src = tmp_path / "src.mp4"
    cont = tmp_path / "cont.mp4"
    _make_v2v_clip(src, num_frames=48, fps=fps, freq=440.0, volume_db=-6.0)  # loud source
    _make_v2v_clip(cont, num_frames=48, fps=fps, freq=880.0, volume_db=-30.0)  # quiet continuation

    out_matched = tmp_path / "matched.mp4"
    out_unmatched = tmp_path / "unmatched.mp4"
    info_matched = video_io.join_v2v(src, cont, out_matched, audio_fade_ms=50, loudness_match=True)
    info_unmatched = video_io.join_v2v(src, cont, out_unmatched, audio_fade_ms=50, loudness_match=False)

    assert info_matched["loudness_matched"] is True
    assert info_unmatched["loudness_matched"] is False

    src_dur = video_io.probe_duration(src)
    # Measure the continuation's segment only, skipping its short fade-in head.
    seg_start = src_dur + 0.2
    matched_db = _mean_volume_db(out_matched, start=seg_start, duration=1.0)
    unmatched_db = _mean_volume_db(out_unmatched, start=seg_start, duration=1.0)

    assert matched_db - unmatched_db > 6.0  # loudness-matched continuation is audibly louder


def test_join_v2v_no_audio_input_falls_back_to_video_only(tmp_path):
    fps = 24.0
    src = tmp_path / "src.mp4"
    cont = tmp_path / "cont.mp4"
    _make_v2v_clip(src, num_frames=24, fps=fps, with_audio=True)
    _make_v2v_clip(cont, num_frames=24, fps=fps, with_audio=False)

    out = tmp_path / "joined.mp4"
    info = video_io.join_v2v(src, cont, out)

    assert out.exists()
    assert video_io.frame_count(out) == 48
    assert video_io.has_audio_stream(out) is False
    assert info["fade_ms_applied"] == 0
    assert info["loudness_matched"] is False
    assert info["source_lufs"] is None
    assert info["continuation_lufs_before"] is None


def test_join_v2v_clamps_fade_longer_than_clip(tmp_path):
    fps = 10.0
    n = 5  # 0.5s clips -- much shorter than the requested 5000ms fade
    src = tmp_path / "src.mp4"
    cont = tmp_path / "cont.mp4"
    _make_v2v_clip(src, num_frames=n, fps=fps, volume_db=-6.0)
    _make_v2v_clip(cont, num_frames=n, fps=fps, volume_db=-6.0)

    out = tmp_path / "joined.mp4"
    info = video_io.join_v2v(src, cont, out, audio_fade_ms=5000)

    assert out.exists()
    assert video_io.frame_count(out) == 2 * n
    assert info["fade_ms_applied"] == 500  # clamped to the 0.5s clip duration


# ------------------------------------------- join_v2v HANDLE true-crossfade mode


def _make_sine_wav(path, dur_sec: float, freq: float = 440.0, sr: int = 48000, volume_db: float = -10.0) -> None:
    """Write a mono sine wav via ffmpeg lavfi (deterministic phase-0 start) -- the
    stand-in engine audio-handle sidecar for the handle-mode join tests."""
    exe = video_io.ffmpeg_path()
    cmd = [
        exe, "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate={sr}",
        "-t", f"{dur_sec:.6f}", "-af", f"volume={volume_db}dB",
        "-ac", "1", "-c:a", "pcm_s16le", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg sine-wav build failed: {proc.stderr[-2000:]}")


def test_join_v2v_handle_mode_duration_and_frames(tmp_path):
    fps = 24.0
    n_src, n_cont = 48, 36  # 2.0s + 1.5s @24fps
    src = tmp_path / "src.mp4"
    cont = tmp_path / "cont.mp4"
    # context = entire source (mirrors E2E-A2), so handle_context_seconds == src_dur
    _make_v2v_clip(src, num_frames=n_src, fps=fps, freq=440.0, volume_db=-10.0)
    _make_v2v_clip(cont, num_frames=n_cont, fps=fps, freq=440.0, volume_db=-10.0)
    # handle = full untrimmed timeline audio: context (2.0s) + continuation (1.5s)
    handle = tmp_path / "output_audio_handle.wav"
    _make_sine_wav(handle, dur_sec=(n_src + n_cont) / fps, freq=440.0, volume_db=-10.0)

    out = tmp_path / "joined_handle.mp4"
    info = video_io.join_v2v(src, cont, out, handle_audio=handle, handle_crossfade_ms=300)

    assert out.exists()
    assert video_io.frame_count(out) == n_src + n_cont
    assert info["join_mode"] == "handle_crossfade"
    assert info["handle_crossfade_ms_applied"] == 300
    # derived handle_context_seconds = handle_dur - continuation_dur ~= source_dur
    assert abs(info["handle_context_seconds"] - n_src / fps) < 0.03
    assert info["loudness_matched"] is True

    out_dur = video_io.probe_duration(out)
    assert abs(out_dur - (n_src + n_cont) / fps) < 0.15


def test_join_v2v_handle_mode_no_valley_vs_fade_pair(tmp_path):
    """The whole point of handle mode: the junction shows NO energy valley,
    unlike the fade-pair path which cuts a near-silent notch at the seam."""
    fps = 24.0
    n = 48  # 2.0s each
    src = tmp_path / "src.mp4"
    cont = tmp_path / "cont.mp4"
    _make_v2v_clip(src, num_frames=n, fps=fps, freq=440.0, volume_db=-10.0)
    _make_v2v_clip(cont, num_frames=n, fps=fps, freq=440.0, volume_db=-10.0)
    # A phase-continuous full-timeline sine standing in for the vocoder handle.
    # Different tone from the source so the equal-power crossfade blends two
    # *uncorrelated* streams (as a real vocoder render vs a real recording are) —
    # equal-power then holds RMS ~flat instead of the pathological phase
    # cancellation two identical sines would show. Same level (-10dB) both sides.
    handle = tmp_path / "output_audio_handle.wav"
    _make_sine_wav(handle, dur_sec=(2 * n) / fps, freq=660.0, volume_db=-10.0)

    src_dur = video_io.probe_duration(src)

    handle_join = tmp_path / "handle.mp4"
    video_io.join_v2v(src, cont, handle_join, handle_audio=handle,
                      handle_crossfade_ms=300, loudness_match=False)
    fade_join = tmp_path / "fade.mp4"
    video_io.join_v2v(src, cont, fade_join, audio_fade_ms=400, loudness_match=False)

    handle_wav = tmp_path / "handle.wav"
    fade_wav = tmp_path / "fade.wav"
    _extract_wav(handle_join, handle_wav)
    _extract_wav(fade_join, fade_wav)
    handle_samples, sr_h = _read_wav(handle_wav)
    fade_samples, sr_f = _read_wav(fade_wav)

    # RMS envelope straddling the junction, and a steady reference away from it.
    handle_junction = _rms_envelope(handle_samples, sr_h, center_t=src_dur, half_window_sec=0.30)
    fade_junction = _rms_envelope(fade_samples, sr_f, center_t=src_dur, half_window_sec=0.30)
    handle_steady = _rms_envelope(handle_samples, sr_h, center_t=src_dur / 2, half_window_sec=0.25)

    assert handle_junction.size and fade_junction.size and handle_steady.size
    steady = float(np.median(handle_steady))
    handle_dip = float(np.min(handle_junction))
    fade_dip = float(np.min(fade_junction))

    # Fade-pair notches deeply below steady; handle mode does NOT valley.
    assert fade_dip < steady * 0.5, (fade_dip, steady)
    assert handle_dip > steady * 0.7, (handle_dip, steady)
    # And the handle junction is unambiguously fuller than the fade-pair notch.
    assert handle_dip > fade_dip * 1.8, (handle_dip, fade_dip)
