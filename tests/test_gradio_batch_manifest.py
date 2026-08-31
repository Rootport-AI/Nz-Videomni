"""Unit tests for the Batch A2V CSV manifest module (gradio_ui/manifest.py).

Pure-Python, filesystem-only (tmp_path) — no Gradio, no HTTP, no live server.
Mirrors the flat-function style of tests/test_gradio_handlers.py.
"""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import pytest

from gradio_ui.manifest import (
    ALLOWED_AUDIO_EXTENSIONS,
    AUTOSAVE_NAME,
    CSV_FIELDS,
    IMAGE_SHARED,
    MANIFEST_NAME,
    STAT_DONE,
    STAT_SKIP,
    STAT_WAITING,
    BatchRow,
    compute_spill_warnings,
    merge_rows,
    read_manifest,
    resolve_output_dir,
    scan_wav_folder,
    unique_output_name,
    write_manifest_atomic,
)


def _write_wav(path: Path, seconds: float, rate: int = 100) -> None:
    """A tiny real .wav (silence) with an exact duration, cheap to generate
    (low sample rate keeps the frame count — and file size — small)."""
    n = int(round(seconds * rate))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n)


# --------------------------------------------------------------------------- #
# CSV round trip (BOM, Japanese, embedded comma/newline, header shape).
# --------------------------------------------------------------------------- #
def test_write_then_read_roundtrip_bom_and_japanese(tmp_path):
    rows = [
        BatchRow(
            queue=1, wav="こんにちは.wav", duration_s=12.5,
            image="立ち絵1.png", prompt="喜び,元気\n改行入りプロンプト",
            stat=STAT_WAITING, output="", frames=121,
            skip_reason="", error="",
        ),
        BatchRow(
            queue=2, wav="second.wav", duration_s=3.0,
            image=IMAGE_SHARED, prompt="plain", stat=STAT_DONE,
            output="second.mp4", frames=41, skip_reason="", error="",
        ),
    ]
    result = write_manifest_atomic(tmp_path, rows)
    assert result.ok is True

    raw = (tmp_path / MANIFEST_NAME).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # utf-8-sig BOM

    with open(tmp_path / MANIFEST_NAME, "r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert header == CSV_FIELDS
    assert len(header) == 10

    read_back = read_manifest(tmp_path)
    assert read_back is not None
    assert len(read_back) == 2
    r0, r1 = read_back
    assert r0.wav == "こんにちは.wav"
    assert r0.image == "立ち絵1.png"
    assert r0.prompt == "喜び,元気\n改行入りプロンプト"
    assert r0.duration_s == 12.5
    assert r0.frames == 121
    assert r1.wav == "second.wav"
    assert r1.stat == STAT_DONE
    assert r1.output == "second.mp4"


def test_read_manifest_missing_file_returns_none(tmp_path):
    assert read_manifest(tmp_path) is None


# --------------------------------------------------------------------------- #
# Broken / hand-edited CSV tolerance.
# --------------------------------------------------------------------------- #
def test_read_manifest_tolerates_missing_columns(tmp_path):
    path = tmp_path / MANIFEST_NAME
    path.write_text("queue,wav,duration\n1,short_row.wav,9.5\n", encoding="utf-8-sig")

    rows = read_manifest(tmp_path)
    assert rows is not None
    assert len(rows) == 1
    row = rows[0]
    assert row.wav == "short_row.wav"
    assert row.duration_s == 9.5
    # missing columns fall back to defaults rather than raising
    assert row.image == IMAGE_SHARED
    assert row.stat == STAT_WAITING
    assert row.frames == 0


def test_read_manifest_tolerates_bad_types(tmp_path):
    path = tmp_path / MANIFEST_NAME
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        writer.writerow(["not-a-number", "bad.wav", "also-not-a-number", "Shared",
                          "p", "Waiting", "", "nope", "", ""])

    rows = read_manifest(tmp_path)
    assert rows is not None
    assert len(rows) == 1
    row = rows[0]
    assert row.wav == "bad.wav"
    assert row.duration_s == 0.0
    assert row.frames == 0
    # queue falls back to the 1-based row index when unparsable
    assert row.queue == 1


# --------------------------------------------------------------------------- #
# scan_wav_folder
# --------------------------------------------------------------------------- #
def test_scan_orders_by_mtime_not_name(tmp_path):
    # "b" is named to sort first alphabetically but should scan LAST (newest
    # mtime) — VOICEROID script order is mtime order, not filename order.
    b = tmp_path / "b_first_name.wav"
    a = tmp_path / "a_second_name.wav"
    _write_wav(b, 1.0)
    _write_wav(a, 1.0)
    os_utime = __import__("os").utime
    os_utime(b, (1000.0, 1000.0))
    os_utime(a, (2000.0, 2000.0))

    rows = scan_wav_folder(tmp_path, fps=24, frames_for=lambda dur, fps: 9)
    assert [r.wav for r in rows] == ["b_first_name.wav", "a_second_name.wav"]
    assert [r.queue for r in rows] == [1, 2]


def test_scan_skips_non_wav_as_wav_only_alpha(tmp_path):
    (tmp_path / "voice.mp3").write_bytes(b"not a real mp3, just bytes")
    (tmp_path / "ignored.txt").write_bytes(b"not audio at all")

    rows = scan_wav_folder(tmp_path, fps=24, frames_for=lambda dur, fps: 9)
    # the .txt file is not an allowed audio extension at all -> excluded entirely
    assert [r.wav for r in rows] == ["voice.mp3"]
    assert rows[0].stat == STAT_SKIP
    assert rows[0].skip_reason == "wav-only-alpha"


def test_scan_skips_over_481_frames(tmp_path):
    long_wav = tmp_path / "too_long.wav"
    _write_wav(long_wav, seconds=25.0, rate=100)  # ~593 raw frames @24fps > 481

    rows = scan_wav_folder(tmp_path, fps=24, frames_for=lambda dur, fps: 999)
    assert len(rows) == 1
    assert rows[0].stat == STAT_SKIP
    assert rows[0].skip_reason == "over-cap"
    # frames_for must NOT have been consulted for a Skip row
    assert rows[0].frames == 0


def test_scan_skips_over_caller_supplied_cap(tmp_path):
    """§4-29: the cap is the Generate tab's own frame count, so a wav well
    under 481 frames is skipped once max_frames is lower than it needs."""
    wav = tmp_path / "medium.wav"
    _write_wav(wav, seconds=15.0, rate=100)  # ~353 raw frames @24fps

    rows = scan_wav_folder(tmp_path, fps=24, frames_for=lambda dur, fps: 353,
                           max_frames=257)
    assert rows[0].stat == STAT_SKIP
    assert rows[0].skip_reason == "over-cap"

    # The same wav with the default (481) cap is NOT skipped.
    rows = scan_wav_folder(tmp_path, fps=24, frames_for=lambda dur, fps: 353)
    assert rows[0].stat == STAT_WAITING


def test_scan_cap_is_clamped_to_the_server_hard_limit(tmp_path):
    """A max_frames above the server's 481 hard cap is clamped down to it
    (mirrors the frontend's Math.min(cap, 481)) — an absurd Frames value can
    never let an over-481 wav through."""
    long_wav = tmp_path / "too_long.wav"
    _write_wav(long_wav, seconds=25.0, rate=100)  # ~593 raw frames @24fps

    rows = scan_wav_folder(tmp_path, fps=24, frames_for=lambda dur, fps: 999,
                           max_frames=9999)
    assert rows[0].stat == STAT_SKIP
    assert rows[0].skip_reason == "over-cap"


def test_scan_uses_injected_frames_for(tmp_path):
    short_wav = tmp_path / "ok.wav"
    _write_wav(short_wav, seconds=3.0, rate=100)

    seen = {}

    def frames_for(dur, fps):
        seen["dur"] = dur
        seen["fps"] = fps
        return 41

    rows = scan_wav_folder(tmp_path, fps=24, frames_for=frames_for)
    assert len(rows) == 1
    assert rows[0].stat == STAT_WAITING
    assert rows[0].frames == 41
    assert seen["dur"] == pytest.approx(3.0)
    assert seen["fps"] == 24


def test_allowed_audio_extensions_matches_config_set():
    assert set(ALLOWED_AUDIO_EXTENSIONS) == {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


# --------------------------------------------------------------------------- #
# compute_spill_warnings
# --------------------------------------------------------------------------- #
def test_compute_spill_warnings_flags_rows_over_threshold():
    rows = [
        BatchRow(queue=1, wav="a.wav", frames=300, stat=STAT_WAITING),
        BatchRow(queue=2, wav="b.wav", frames=100, stat=STAT_WAITING),
        BatchRow(queue=3, wav="c.wav", frames=999, stat=STAT_SKIP, skip_reason="over-cap"),
    ]
    warnings = compute_spill_warnings(rows, 1280, 768, {"1280x768": 257})
    assert len(warnings) == 1
    assert "a.wav" in warnings[0]


def test_compute_spill_warnings_unknown_resolution_is_empty():
    rows = [BatchRow(queue=1, wav="a.wav", frames=300)]
    assert compute_spill_warnings(rows, 999, 999, {"1280x768": 257}) == []


# --------------------------------------------------------------------------- #
# merge_rows: the 4 rules + queue renumbering.
# --------------------------------------------------------------------------- #
def test_merge_rows_all_four_rules():
    existing = [
        BatchRow(queue=1, wav="a.wav", image="img1.png", prompt="edited a",
                  stat=STAT_DONE, output="a.mp4", frames=41),
        BatchRow(queue=2, wav="b.wav", image=IMAGE_SHARED, prompt="",
                  stat=STAT_WAITING, output="", frames=41),
        BatchRow(queue=3, wav="gone.wav", image=IMAGE_SHARED, prompt="",
                  stat=STAT_WAITING, output="", frames=41),
        BatchRow(queue=4, wav="gone_done.wav", image=IMAGE_SHARED, prompt="",
                  stat=STAT_DONE, output="gone_done.mp4", frames=41),
    ]
    scanned = [
        # rescan order = mtime order: b, a, c
        BatchRow(queue=1, wav="b.wav", duration_s=5.0, frames=50, stat=STAT_WAITING),
        BatchRow(queue=2, wav="a.wav", duration_s=99.0, frames=0,
                  stat=STAT_SKIP, skip_reason="over-cap"),
        BatchRow(queue=3, wav="c.wav", duration_s=3.0, frames=30, stat=STAT_WAITING),
    ]

    merged, warnings = merge_rows(existing, scanned)

    by_wav = {r.wav: r for r in merged}
    assert list(by_wav.keys()) == ["b.wav", "a.wav", "c.wav", "gone_done.wav"]

    # rule 1 (normal): existing prompt/image/stat/output kept, duration/frames refreshed
    assert by_wav["b.wav"].stat == STAT_WAITING
    assert by_wav["b.wav"].frames == 50
    assert by_wav["b.wav"].duration_s == 5.0

    # rule 1 (Done protection): Done row is never demoted to Skip by a rescan
    a_row = by_wav["a.wav"]
    assert a_row.stat == STAT_DONE
    assert a_row.image == "img1.png"
    assert a_row.prompt == "edited a"
    assert a_row.output == "a.mp4"
    assert a_row.skip_reason == "over-cap"  # recomputed value still recorded

    # rule 2: brand new row, added as Waiting
    c_row = by_wav["c.wav"]
    assert c_row.stat == STAT_WAITING
    assert c_row.frames == 30

    # rule 3: "gone.wav" (non-Done, missing wav) is dropped entirely
    assert "gone.wav" not in by_wav

    # rule 3 (Done protection): "gone_done.wav" kept at the tail
    assert by_wav["gone_done.wav"].stat == STAT_DONE
    assert by_wav["gone_done.wav"].output == "gone_done.mp4"

    # rule 4: queue renumbered 1..4 in [scanned order..., done-leftovers]
    assert [r.queue for r in merged] == [1, 2, 3, 4]
    assert merged[-1].wav == "gone_done.wav"

    assert len(warnings) == 2
    assert any("a.wav" in w for w in warnings)
    assert any("gone_done.wav" in w for w in warnings)


def test_merge_rows_rescan_clears_a_stale_skip():
    """§4-29 Skip recovery: the cap is DURATION-linked now, so a row skipped
    under a low Frames value must come back once the rescan says it fits
    (raise Frames -> Set audios). A Done row still wins over the rescan."""
    existing = [
        BatchRow(queue=1, wav="a.wav", prompt="kept", stat=STAT_SKIP,
                 skip_reason="over-cap", frames=0),
        BatchRow(queue=2, wav="legacy.wav", stat=STAT_SKIP,
                 skip_reason="over-481f", frames=0),
        BatchRow(queue=3, wav="done.wav", stat=STAT_DONE, output="done.mp4"),
    ]
    scanned = [
        BatchRow(queue=1, wav="a.wav", duration_s=10.0, frames=241, stat=STAT_WAITING),
        BatchRow(queue=2, wav="legacy.wav", duration_s=8.0, frames=193, stat=STAT_WAITING),
        BatchRow(queue=3, wav="done.wav", duration_s=4.0, frames=97, stat=STAT_WAITING),
    ]

    merged, warnings = merge_rows(existing, scanned)
    by_wav = {r.wav: r for r in merged}

    # Both stale Skips (current + legacy reason code) return to Waiting, with
    # the refreshed frames and a cleared reason — and user edits survive.
    assert by_wav["a.wav"].stat == STAT_WAITING
    assert by_wav["a.wav"].frames == 241
    assert by_wav["a.wav"].skip_reason == ""
    assert by_wav["a.wav"].prompt == "kept"
    assert by_wav["legacy.wav"].stat == STAT_WAITING

    # Done protection unchanged: a completed row is not re-queued by a rescan.
    assert by_wav["done.wav"].stat == STAT_DONE
    assert by_wav["done.wav"].output == "done.mp4"
    assert warnings == []


def test_merge_rows_first_scan_no_existing_manifest():
    scanned = [
        BatchRow(queue=1, wav="only.wav", duration_s=4.0, frames=41, stat=STAT_WAITING),
    ]
    merged, warnings = merge_rows(None, scanned)
    assert warnings == []
    assert len(merged) == 1
    assert merged[0].wav == "only.wav"
    assert merged[0].queue == 1


# --------------------------------------------------------------------------- #
# unique_output_name
# --------------------------------------------------------------------------- #
def test_unique_output_name_no_collision(tmp_path):
    assert unique_output_name(tmp_path, "voice001.wav") == "voice001.mp4"


def test_unique_output_name_one_collision(tmp_path):
    (tmp_path / "voice001.mp4").write_bytes(b"x")
    assert unique_output_name(tmp_path, "voice001.wav") == "voice001_2.mp4"


def test_unique_output_name_multiple_collisions(tmp_path):
    (tmp_path / "voice001.mp4").write_bytes(b"x")
    (tmp_path / "voice001_2.mp4").write_bytes(b"x")
    (tmp_path / "voice001_3.mp4").write_bytes(b"x")
    assert unique_output_name(tmp_path, "voice001.wav") == "voice001_4.mp4"


# --------------------------------------------------------------------------- #
# resolve_output_dir
# --------------------------------------------------------------------------- #
def test_resolve_output_dir_auto(tmp_path):
    wav_dir = tmp_path / "voice"
    wav_dir.mkdir()
    out = resolve_output_dir(wav_dir, "auto")
    assert out == tmp_path / "voice_a2v_out"


def test_resolve_output_dir_custom(tmp_path):
    wav_dir = tmp_path / "voice"
    wav_dir.mkdir()
    custom = tmp_path / "somewhere_else"
    out = resolve_output_dir(wav_dir, "custom", custom_dir=custom)
    assert out == custom


# --------------------------------------------------------------------------- #
# write_manifest_atomic
# --------------------------------------------------------------------------- #
def test_write_manifest_atomic_normal(tmp_path):
    rows = [BatchRow(queue=1, wav="x.wav", duration_s=1.0)]
    result = write_manifest_atomic(tmp_path, rows)
    assert result.ok is True
    assert result.locked is False
    assert result.path == tmp_path / MANIFEST_NAME
    assert (tmp_path / MANIFEST_NAME).exists()
    # temp file must not linger after a successful replace
    assert not (tmp_path / (MANIFEST_NAME + ".tmp")).exists()


def test_write_manifest_atomic_locked_falls_back_to_autosave(tmp_path, monkeypatch):
    def _always_locked(*args, **kwargs):
        raise PermissionError("locked by Excel")

    monkeypatch.setattr("gradio_ui.manifest.os.replace", _always_locked)
    monkeypatch.setattr("gradio_ui.manifest.time.sleep", lambda *_a, **_k: None)

    rows = [BatchRow(queue=1, wav="x.wav", duration_s=1.0, prompt="p")]
    result = write_manifest_atomic(tmp_path, rows)

    assert result.ok is False
    assert result.locked is True
    assert result.autosave_path == tmp_path / AUTOSAVE_NAME
    assert (tmp_path / AUTOSAVE_NAME).exists()
    # the primary manifest was never created (replace always failed)
    assert not (tmp_path / MANIFEST_NAME).exists()

    autosaved = read_manifest(tmp_path.parent)  # sanity: wrong dir -> None
    assert autosaved is None
    with open(tmp_path / AUTOSAVE_NAME, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        saved_rows = list(reader)
    assert len(saved_rows) == 1
    assert saved_rows[0]["wav"] == "x.wav"
    assert saved_rows[0]["prompt"] == "p"
