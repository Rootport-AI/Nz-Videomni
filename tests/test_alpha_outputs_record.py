"""``services/alpha_outputs_record.py`` -- the record ``alpha_outputs.json`` of
the Alpha Gen exit (第 3 弾). Pure file I/O, no ffmpeg."""

from __future__ import annotations

import json

from services import alpha_outputs_record as rec
from services.alpha_outputs_record import AlphaOutputsRecord, AlphaThresholds


def _touch(path):
    path.write_bytes(b"x")
    return path


def test_missing_record_reads_empty(tmp_path):
    assert rec.read(tmp_path) == AlphaOutputsRecord()
    assert rec.existing(tmp_path) == AlphaOutputsRecord()


def test_broken_json_reads_empty(tmp_path):
    (tmp_path / rec.RECORD_FILENAME).write_text("{not json", encoding="utf-8")
    assert rec.read(tmp_path) == AlphaOutputsRecord()


def test_non_object_json_reads_empty(tmp_path):
    (tmp_path / rec.RECORD_FILENAME).write_text("[1, 2]", encoding="utf-8")
    assert rec.read(tmp_path) == AlphaOutputsRecord()


def test_write_then_read_round_trip(tmp_path):
    record = AlphaOutputsRecord(
        rgba=AlphaThresholds(7, 248),
        chroma={"green": AlphaThresholds(0, 255), "blue": AlphaThresholds(128, 128)},
    )
    rec.write(tmp_path, record)
    assert rec.read(tmp_path) == record
    data = json.loads((tmp_path / rec.RECORD_FILENAME).read_text(encoding="utf-8"))
    assert data == {
        "rgba": {"transparent_threshold": 7, "opaque_threshold": 248},
        "chroma": {
            "green": {"transparent_threshold": 0, "opaque_threshold": 255},
            "blue": {"transparent_threshold": 128, "opaque_threshold": 128},
        },
    }
    assert not (tmp_path / "_alpha_outputs.tmp.json").exists()


def test_invalid_entries_are_ignored(tmp_path):
    (tmp_path / rec.RECORD_FILENAME).write_text(
        json.dumps({
            "rgba": {"transparent_threshold": 200, "opaque_threshold": 100},
            "chroma": {
                "green": {"transparent_threshold": 7, "opaque_threshold": 248},
                "red": {"transparent_threshold": "7", "opaque_threshold": 248},
                "blue": {"transparent_threshold": 7, "opaque_threshold": 300},
                "aqua": "nonsense",
            },
        }),
        encoding="utf-8",
    )
    assert rec.read(tmp_path) == AlphaOutputsRecord(
        rgba=None, chroma={"green": AlphaThresholds(7, 248)}
    )


def test_existing_needs_the_files_too(tmp_path):
    rec.write(
        tmp_path,
        AlphaOutputsRecord(
            rgba=AlphaThresholds(7, 248),
            chroma={"green": AlphaThresholds(7, 248), "blue": AlphaThresholds(7, 248)},
        ),
    )
    # Nothing on disk yet: the record alone promises nothing.
    assert rec.existing(tmp_path) == AlphaOutputsRecord()

    # The transparent video needs BOTH the MOV and its preview.
    _touch(tmp_path / rec.RGBA_FILENAME)
    assert rec.existing(tmp_path).rgba is None
    _touch(tmp_path / rec.PREVIEW_FILENAME)
    assert rec.existing(tmp_path).rgba == AlphaThresholds(7, 248)

    _touch(tmp_path / rec.chroma_filename("green"))
    assert rec.existing(tmp_path).chroma == {"green": AlphaThresholds(7, 248)}


def test_a_file_without_a_record_entry_does_not_count(tmp_path):
    _touch(tmp_path / rec.RGBA_FILENAME)
    _touch(tmp_path / rec.PREVIEW_FILENAME)
    _touch(tmp_path / rec.chroma_filename("green"))
    assert rec.existing(tmp_path) == AlphaOutputsRecord()


def test_chroma_filename():
    assert rec.chroma_filename("magenta") == "chroma_magenta.mp4"
