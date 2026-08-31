"""Contract + parity tests for mcp_server/batch_planning.py and
mcp_server/tools/batch.py (W6).

The frame-suggestion arithmetic (``suggest_frames_for_audio``) is a literal
copy of ``gradio_ui.handlers.suggest_frames_for_audio`` -- that module imports
``gradio`` at package-init time, which ``mcp_server`` must never do (plan D2 /
module docstring). This is pinned by an exhaustive comparison against the real
thing below; the row-scan conventions mirror ``gradio_ui.manifest.scan_wav_folder``.
"""

from __future__ import annotations

import os
import struct
import time
import wave
from pathlib import Path

import anyio
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from mcp_server import batch_planning
from mcp_server.client import set_client
from mcp_server.tools import batch

_DURATIONS = (0.1, 0.5, 1, 2, 3, 5, 6, 10, 19, 20, 30)
_FPS_VALUES = (24, 30, 23.976, 60)


def _make_wav(path: Path, *, seconds: float = 2.0, sr: int = 16000, channels: int = 1) -> Path:
    n = int(round(seconds * sr))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % (n * channels), *([0] * (n * channels))))
    return path


# --------------------------------------------------------------------- parity


def test_suggest_frames_for_audio_matches_gradio_ui_handlers_exhaustively():
    from gradio_ui.handlers import suggest_frames_for_audio as reference

    for dur in _DURATIONS:
        for fps in _FPS_VALUES:
            expected = reference(dur, fps)
            actual = batch_planning.suggest_frames_for_audio(dur, fps)
            assert actual == expected, (dur, fps, expected, actual)


def test_suggest_frames_for_audio_always_8n_plus_1_in_range():
    for dur in _DURATIONS:
        for fps in _FPS_VALUES:
            nf = batch_planning.suggest_frames_for_audio(dur, fps)
            assert 9 <= nf <= 481
            assert (nf - 1) % 8 == 0


# ------------------------------------------------------------------ plan_rows


def test_plan_rows_sorts_by_mtime_ascending(tmp_path):
    a = _make_wav(tmp_path / "a.wav")
    b = _make_wav(tmp_path / "b.wav")
    now = time.time()
    os.utime(a, (now - 10, now - 10))
    os.utime(b, (now - 100, now - 100))

    rows = batch_planning.plan_rows(tmp_path, 24.0)

    assert [r["filename"] for r in rows] == ["b.wav", "a.wav"]
    assert [r["index"] for r in rows] == [1, 2]


def test_plan_rows_non_wav_visible_skip_row(tmp_path):
    (tmp_path / "song.mp3").write_bytes(b"\x00\x01\x02\x03")

    rows = batch_planning.plan_rows(tmp_path, 24.0)

    assert len(rows) == 1
    assert rows[0]["filename"] == "song.mp3"
    assert rows[0]["skip_reason"] == "wav-only-alpha"
    assert rows[0]["suggested_num_frames"] is None
    assert rows[0]["duration_seconds"] == 0.0


def test_plan_rows_reserved_filenames_and_tmp_excluded(tmp_path):
    _make_wav(tmp_path / "voice.wav")
    (tmp_path / "batch_a2v_manifest.csv").write_text("queue,wav\n", encoding="utf-8")
    (tmp_path / "batch_a2v_manifest.autosave.csv").write_text("queue,wav\n", encoding="utf-8")
    (tmp_path / "scratch.wav.tmp").write_bytes(b"\x00")

    rows = batch_planning.plan_rows(tmp_path, 24.0)

    assert [r["filename"] for r in rows] == ["voice.wav"]


def test_plan_rows_long_wav_over_cap_skip(tmp_path):
    _make_wav(tmp_path / "long.wav", seconds=30.0)

    rows = batch_planning.plan_rows(tmp_path, 24.0)

    assert rows[0]["skip_reason"] == "over-cap"
    assert rows[0]["suggested_num_frames"] is None
    assert rows[0]["duration_seconds"] > 0


def test_plan_rows_max_frames_over_hard_cap_clamped_to_481(tmp_path):
    # A raw frame count just over the hard cap (481) must still be Skipped
    # even when the caller passes max_frames=9999 -- over_frame_limit() must
    # clamp to min(max_frames, 481) internally (mirrors
    # gradio_ui/manifest.py::over_frame_limit; DURATION values above the
    # server's hard cap are not a valid escape hatch).
    _make_wav(tmp_path / "long.wav", seconds=30.0)

    rows = batch_planning.plan_rows(tmp_path, 24.0, max_frames=9999)

    assert rows[0]["skip_reason"] == "over-cap"
    assert rows[0]["suggested_num_frames"] is None


def test_plan_rows_plannable_row_has_suggested_frames(tmp_path):
    _make_wav(tmp_path / "short.wav", seconds=2.0)

    rows = batch_planning.plan_rows(tmp_path, 24.0)

    assert rows[0]["skip_reason"] == ""
    assert rows[0]["suggested_num_frames"] is not None
    assert (rows[0]["suggested_num_frames"] - 1) % 8 == 0


def test_plan_rows_image_same_stem_match(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _make_wav(wav_dir / "clip01.wav")

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "clip01.png").write_bytes(b"\x89PNG\r\n")
    (img_dir / "other.png").write_bytes(b"\x89PNG\r\n")

    rows = batch_planning.plan_rows(wav_dir, 24.0, img_dir)

    assert rows[0]["image_path"] == str(img_dir / "clip01.png")


def test_plan_rows_image_none_when_no_match_or_no_dir(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _make_wav(wav_dir / "clip01.wav")

    rows_no_dir = batch_planning.plan_rows(wav_dir, 24.0)
    assert rows_no_dir[0]["image_path"] is None

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "unrelated.png").write_bytes(b"\x89PNG\r\n")
    rows_no_match = batch_planning.plan_rows(wav_dir, 24.0, img_dir)
    assert rows_no_match[0]["image_path"] is None


# ------------------------------------------------------------- plan_a2v_batch


@pytest.fixture(autouse=True)
def _reset_client():
    set_client(None)
    yield
    set_client(None)


def test_plan_a2v_batch_works_with_no_client_installed(tmp_path):
    set_client(None)  # no backend client at all -- plan_a2v_batch must not need HTTP
    _make_wav(tmp_path / "a.wav", seconds=2.0)

    result = anyio.run(batch.plan_a2v_batch, str(tmp_path), 24.0)

    assert result["fps"] == 24.0
    assert result["wav_dir"] == str(tmp_path)
    assert result["counts"] == {"total": 1, "plannable": 1, "skipped": 0}
    assert result["rows"][0]["filename"] == "a.wav"
    assert "next_steps" in result and result["next_steps"]


def test_plan_a2v_batch_counts_mixed_rows(tmp_path):
    _make_wav(tmp_path / "ok.wav", seconds=2.0)
    (tmp_path / "bad.mp3").write_bytes(b"\x00")
    _make_wav(tmp_path / "toolong.wav", seconds=30.0)

    result = anyio.run(batch.plan_a2v_batch, str(tmp_path), 24.0)

    assert result["counts"] == {"total": 3, "plannable": 1, "skipped": 2}


def test_plan_a2v_batch_missing_wav_dir_raises():
    with pytest.raises(ToolError) as exc_info:
        anyio.run(batch.plan_a2v_batch, "S:/does-not-exist-xyz-mcp-test", 24.0)

    assert "WAV_DIR_NOT_FOUND" in str(exc_info.value)


def test_plan_a2v_batch_missing_image_dir_raises(tmp_path):
    _make_wav(tmp_path / "a.wav", seconds=2.0)

    with pytest.raises(ToolError) as exc_info:
        anyio.run(batch.plan_a2v_batch, str(tmp_path), 24.0, "S:/no-such-image-dir-xyz")

    assert "IMAGE_DIR_NOT_FOUND" in str(exc_info.value)
