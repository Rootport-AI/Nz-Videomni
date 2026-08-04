"""POST /upload/video — optional ribbon-trim window (§1-6, V2V ribbon trim).

The endpoint gained two optional query arguments (``trim_start_sec`` /
``trim_duration_sec``). The contract these tests pin down:

1. Without them the endpoint is byte-identical to before -- in particular
   ``cut_range_mp4`` is never even called (the monkeypatched stand-in raises if
   it is), and the stored file equals the posted bytes exactly.
2. With them, the window is forwarded verbatim and the response reports
   ``trimmed: true``.
3. Anything that cannot produce a valid cut -- only one argument given, NaN/inf,
   a negative start, a non-positive duration, or an ffmpeg failure -- degrades
   to the untrimmed upload with HTTP 200 and ``trimmed: false``. The trim
   arguments must never introduce a new error response.

Most tests stub ``cut_range_mp4`` (the cut itself is unit-tested in
test_video_io.py); the last one runs the real ffmpeg end to end.
"""

from __future__ import annotations

import pytest
from PIL import Image

import services.video_upload_store as video_upload_store
from services import video_io
from services.video_io import FFmpegError

# Same "not a real video, never decoded" blob used by test_ic_lora_api.py.
FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64

TRIMMED_BYTES = b"TRIMMED-OUTPUT" * 8


def _no_cut_allowed(*args, **kwargs):
    raise AssertionError(f"cut_range_mp4 must not be called; got args={args!r} kwargs={kwargs!r}")


def _install_recording_cut(monkeypatch) -> list[tuple]:
    """Replace cut_range_mp4 with a stand-in that records its call and writes a
    small recognizable file to ``out`` (so the swap-in path runs for real)."""
    calls: list[tuple] = []

    def fake(src, out, start_sec, duration_sec):
        calls.append((src, out, start_sec, duration_sec))
        out.write_bytes(TRIMMED_BYTES)
        return {
            "source_fps": 30.0,
            "total_frames": 300,
            "start_frame": 30,
            "end_frame": 89,
            "written_frames": 60,
        }

    monkeypatch.setattr(video_upload_store, "cut_range_mp4", fake)
    return calls


def _video_dir(client, video_id):
    return client.app_context.video_upload_store.video_dir / video_id


def test_no_trim_query_never_cuts_and_stores_exact_bytes(client, monkeypatch):
    monkeypatch.setattr(video_upload_store, "cut_range_mp4", _no_cut_allowed)

    r = client.post("/api/v1/upload/video", files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is False
    assert body["content_type"] == "video/mp4"
    assert body["size_bytes"] == len(FAKE_MP4)

    stored = client.app_context.video_upload_store.path_for(body["video_id"])
    assert stored.read_bytes() == FAKE_MP4


def test_trim_query_is_forwarded_and_reported(client, monkeypatch):
    calls = _install_recording_cut(monkeypatch)

    r = client.post(
        "/api/v1/upload/video",
        params={"trim_start_sec": 1.0, "trim_duration_sec": 2.0},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is True
    assert body["content_type"] == "video/mp4"
    assert body["size_bytes"] == len(TRIMMED_BYTES)
    assert body["stored_path"].endswith("/input.mp4")

    assert len(calls) == 1
    src, out, start_sec, duration_sec = calls[0]
    assert (start_sec, duration_sec) == (1.0, 2.0)
    assert src.name == "input.mp4"
    assert out.name.startswith("_")  # never matches path_for's glob("input.*")

    stored = client.app_context.video_upload_store.path_for(body["video_id"])
    assert stored.name == "input.mp4"
    assert stored.read_bytes() == TRIMMED_BYTES
    # the temp work file is gone
    assert sorted(p.name for p in _video_dir(client, body["video_id"]).iterdir()) == ["input.mp4"]


@pytest.mark.parametrize(
    "params",
    [
        {"trim_start_sec": 1.0},
        {"trim_duration_sec": 2.0},
    ],
)
def test_half_specified_window_passes_through(client, monkeypatch, params):
    monkeypatch.setattr(video_upload_store, "cut_range_mp4", _no_cut_allowed)

    r = client.post(
        "/api/v1/upload/video",
        params=params,
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is False
    assert body["size_bytes"] == len(FAKE_MP4)
    stored = client.app_context.video_upload_store.path_for(body["video_id"])
    assert stored.read_bytes() == FAKE_MP4


def test_ffmpeg_failure_falls_back_to_the_untrimmed_upload(client, monkeypatch):
    def boom(src, out, start_sec, duration_sec):
        out.write_bytes(b"partial garbage")  # a half-written cut must not survive
        raise FFmpegError("ffmpeg not found on PATH")

    monkeypatch.setattr(video_upload_store, "cut_range_mp4", boom)

    r = client.post(
        "/api/v1/upload/video",
        params={"trim_start_sec": 1.0, "trim_duration_sec": 2.0},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is False
    assert body["size_bytes"] == len(FAKE_MP4)
    assert body["content_type"] == "video/mp4"

    # the original upload is untouched and is still the only input.* file
    d = _video_dir(client, body["video_id"])
    assert sorted(p.name for p in d.glob("input.*")) == ["input.mp4"]
    assert (d / "input.mp4").read_bytes() == FAKE_MP4
    assert sorted(p.name for p in d.iterdir()) == ["input.mp4"]  # temp file cleaned up


@pytest.mark.parametrize(
    "start_sec,duration_sec,may_call",
    [
        (float("nan"), 2.0, False),
        (1.0, float("nan"), False),
        (float("inf"), 2.0, False),
        (1.0, float("inf"), False),
        (-1.0, 2.0, False),
        (1.0, 0.0, False),
        (1.0, -2.0, False),
        (1e30, 1e30, True),  # finite but absurd -> reaches ffmpeg, which fails
    ],
)
def test_unusable_trim_values_pass_through_with_200(client, monkeypatch, start_sec, duration_sec, may_call):
    def stub(src, out, start_sec_, duration_sec_):
        if not may_call:
            raise AssertionError("cut_range_mp4 must not be called for an unusable window")
        raise FFmpegError("window starts past the end of the source")

    monkeypatch.setattr(video_upload_store, "cut_range_mp4", stub)

    r = client.post(
        "/api/v1/upload/video",
        params={"trim_start_sec": start_sec, "trim_duration_sec": duration_sec},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is False
    assert body["size_bytes"] == len(FAKE_MP4)
    stored = client.app_context.video_upload_store.path_for(body["video_id"])
    assert stored.read_bytes() == FAKE_MP4


def test_non_mp4_upload_is_normalized_to_a_single_input_mp4(client, monkeypatch):
    _install_recording_cut(monkeypatch)

    r = client.post(
        "/api/v1/upload/video",
        params={"trim_start_sec": 0.5, "trim_duration_sec": 1.5},
        files={"file": ("clip.mkv", FAKE_MP4, "video/x-matroska")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is True
    assert body["content_type"] == "video/mp4"
    assert body["stored_path"].endswith("/input.mp4")

    d = _video_dir(client, body["video_id"])
    # exactly one input.* survives, so path_for is unambiguous
    assert sorted(p.name for p in d.glob("input.*")) == ["input.mp4"]
    stored = client.app_context.video_upload_store.path_for(body["video_id"])
    assert stored.suffix == ".mp4"
    assert stored.read_bytes() == TRIMMED_BYTES


def test_real_ffmpeg_trim_end_to_end(client, tmp_path):
    # 10 frames @10fps -> [0.2s, +0.3s) keeps frames 2,3,4.
    src = tmp_path / "src.mp4"
    frames = [Image.new("RGB", (64, 64), (i * 20 % 256, 90, 160)) for i in range(10)]
    video_io.encode_frames_to_mp4(frames, src, frame_rate=10.0)
    payload = src.read_bytes()

    r = client.post(
        "/api/v1/upload/video",
        params={"trim_start_sec": 0.2, "trim_duration_sec": 0.3},
        files={"file": ("src.mp4", payload, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is True

    stored = client.app_context.video_upload_store.path_for(body["video_id"])
    assert video_io.frame_count(stored) == 3
    assert body["size_bytes"] == stored.stat().st_size
    assert stored.stat().st_size != len(payload)

    # An absurd (but finite) window fails inside ffmpeg -> untrimmed 200.
    r2 = client.post(
        "/api/v1/upload/video",
        params={"trim_start_sec": 1e30, "trim_duration_sec": 1.0},
        files={"file": ("src.mp4", payload, "video/mp4")},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["trimmed"] is False
    stored2 = client.app_context.video_upload_store.path_for(r2.json()["video_id"])
    assert stored2.read_bytes() == payload
