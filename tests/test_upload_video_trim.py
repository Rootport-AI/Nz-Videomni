"""POST /upload/video — optional ribbon-trim window (§1-6, V2V ribbon trim) and
the §1-15 upload-time frame-count ceiling (clip-wise IC-LoRA reference).

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
test_video_io.py); some run the real ffmpeg end to end.

§1-15 adds a THIRD, independent query argument, ``max_frames`` (the FE sends
11544 -- ``api/models.py``'s ``MAX_CHAIN_TOTAL_PIXEL_FRAMES``). Its contract,
pinned in the ``max_frames`` section below:

4. At or under the limit -- a no-op: ``frame_count`` alone decides, ``cut_
   range_mp4`` is never called, the response is ``trimmed: false`` and the
   stored bytes are exactly the posted bytes (no re-encode).
5. Over the limit -- the first ``max_frames`` frames only, via the same tmp-
   then-``os.replace`` swap as the trim path, reported ``trimmed: true``.
6. A failed probe (ffprobe missing) or a failed cut degrades to the untrimmed
   upload with HTTP 200 and ``trimmed: false``, same as the trim window's own
   failure handling.
7. An explicit trim window always wins over ``max_frames`` when both are sent
   (the V2V ribbon-trim path is unaffected by this feature).

The end source's automatic band length adds ``frame_count`` / ``fps`` to the
response (the server measures the material so the app does not have to guess).
``max_frames`` doubles as the opt-in, so the rules are pinned per PATH in the
last section: an upload that did not ask to be measured must not pay for an
extra ffprobe and must report both as null, and every probe/cut failure must
report null rather than an error.
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


# ── §1-15 max_frames (upload-time frame-count ceiling) ──────────────────────
def _install_recording_cut_with_frames(monkeypatch) -> list[tuple]:
    """Same as ``_install_recording_cut`` but the stand-in also accepts the
    keyword-only ``start_frame``/``num_frames`` the max_frames path calls
    ``cut_range_mp4`` with (the trim-window path never passes them)."""
    calls: list[tuple] = []

    def fake(src, out, start_sec, duration_sec, *, start_frame=None, num_frames=None):
        calls.append((src, out, start_sec, duration_sec, start_frame, num_frames))
        out.write_bytes(TRIMMED_BYTES)
        return {
            "source_fps": 30.0,
            "total_frames": 300,
            "start_frame": start_frame or 0,
            "end_frame": (start_frame or 0) + (num_frames or 1) - 1,
            "written_frames": num_frames or 1,
        }

    monkeypatch.setattr(video_upload_store, "cut_range_mp4", fake)
    return calls


def test_no_max_frames_query_never_probes_or_cuts(client, monkeypatch):
    monkeypatch.setattr(video_upload_store, "cut_range_mp4", _no_cut_allowed)

    def boom(p):
        raise AssertionError("frame_count must not be called without max_frames")

    monkeypatch.setattr(video_upload_store, "frame_count", boom)

    r = client.post("/api/v1/upload/video", files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is False
    assert body["size_bytes"] == len(FAKE_MP4)


def test_max_frames_under_the_limit_is_a_no_op(client, monkeypatch):
    monkeypatch.setattr(video_upload_store, "cut_range_mp4", _no_cut_allowed)
    monkeypatch.setattr(video_upload_store, "frame_count", lambda p: 5)

    r = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 10},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is False
    assert body["content_type"] == "video/mp4"
    assert body["size_bytes"] == len(FAKE_MP4)
    stored = client.app_context.video_upload_store.path_for(body["video_id"])
    assert stored.read_bytes() == FAKE_MP4


def test_max_frames_exactly_at_the_limit_is_a_no_op(client, monkeypatch):
    """Boundary: total == max_frames must NOT cut (only total > max_frames does)."""
    monkeypatch.setattr(video_upload_store, "cut_range_mp4", _no_cut_allowed)
    monkeypatch.setattr(video_upload_store, "frame_count", lambda p: 10)

    r = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 10},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["trimmed"] is False


def test_max_frames_over_the_limit_keeps_only_the_head(client, monkeypatch):
    calls = _install_recording_cut_with_frames(monkeypatch)
    monkeypatch.setattr(video_upload_store, "frame_count", lambda p: 20)

    r = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 10},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is True
    assert body["content_type"] == "video/mp4"
    assert body["size_bytes"] == len(TRIMMED_BYTES)
    assert body["stored_path"].endswith("/input.mp4")

    assert len(calls) == 1
    src, out, start_sec, duration_sec, start_frame, num_frames = calls[0]
    assert (start_frame, num_frames) == (0, 10)
    assert src.name == "input.mp4"
    assert out.name.startswith("_")  # never matches path_for's glob("input.*")

    stored = client.app_context.video_upload_store.path_for(body["video_id"])
    assert stored.name == "input.mp4"
    assert stored.read_bytes() == TRIMMED_BYTES
    assert sorted(p.name for p in _video_dir(client, body["video_id"]).iterdir()) == ["input.mp4"]


def test_max_frames_probe_failure_falls_back_to_the_untrimmed_upload(client, monkeypatch):
    """ffprobe missing (or any frame_count failure) must degrade to the
    untrimmed upload, exactly like an ffmpeg cut failure does."""
    monkeypatch.setattr(video_upload_store, "cut_range_mp4", _no_cut_allowed)

    def boom(p):
        raise FFmpegError("ffprobe not found on PATH. Install ffmpeg and add it to PATH.")

    monkeypatch.setattr(video_upload_store, "frame_count", boom)

    r = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 10},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is False
    assert body["size_bytes"] == len(FAKE_MP4)
    stored = client.app_context.video_upload_store.path_for(body["video_id"])
    assert stored.read_bytes() == FAKE_MP4


def test_max_frames_cut_failure_falls_back_to_the_untrimmed_upload(client, monkeypatch):
    monkeypatch.setattr(video_upload_store, "frame_count", lambda p: 20)

    def boom(src, out, start_sec, duration_sec, *, start_frame=None, num_frames=None):
        out.write_bytes(b"partial garbage")  # a half-written cut must not survive
        raise FFmpegError("ffmpeg not found on PATH")

    monkeypatch.setattr(video_upload_store, "cut_range_mp4", boom)

    r = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 10},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is False
    assert body["size_bytes"] == len(FAKE_MP4)

    d = _video_dir(client, body["video_id"])
    assert sorted(p.name for p in d.glob("input.*")) == ["input.mp4"]
    assert (d / "input.mp4").read_bytes() == FAKE_MP4
    assert sorted(p.name for p in d.iterdir()) == ["input.mp4"]  # temp file cleaned up


def test_negative_or_zero_max_frames_is_a_no_op(client, monkeypatch):
    """Deliberately unconstrained (no ge=/le=), same convention as the trim
    window: a nonsensical value falls through to a plain untrimmed upload
    rather than a 422 or a crash."""
    monkeypatch.setattr(video_upload_store, "cut_range_mp4", _no_cut_allowed)

    def boom(p):
        raise AssertionError("frame_count must not be called for max_frames <= 0")

    monkeypatch.setattr(video_upload_store, "frame_count", boom)

    for bad in (0, -1):
        r = client.post(
            "/api/v1/upload/video",
            params={"max_frames": bad},
            files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
        )
        assert r.status_code == 200, r.text
        assert r.json()["trimmed"] is False


def test_explicit_trim_window_wins_over_max_frames(client, monkeypatch):
    """When both are sent, the trim window wins outright -- max_frames /
    frame_count is never even consulted (V2V ribbon-trim path unaffected)."""
    calls = _install_recording_cut(monkeypatch)

    def boom(p):
        raise AssertionError("frame_count must not be called when a trim window is present")

    monkeypatch.setattr(video_upload_store, "frame_count", boom)

    r = client.post(
        "/api/v1/upload/video",
        params={"trim_start_sec": 1.0, "trim_duration_sec": 2.0, "max_frames": 1},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is True
    assert len(calls) == 1
    _src, _out, start_sec, duration_sec = calls[0]
    assert (start_sec, duration_sec) == (1.0, 2.0)


# ── UploadVideoResponse.frame_count / fps (end source's automatic band) ─────
def _no_probe_allowed(path):
    raise AssertionError(f"probe_fps must not be called; got {path!r}")


def _install_recording_probe(monkeypatch, value=24.0) -> list:
    calls: list = []

    def fake(path):
        calls.append(path)
        return value

    monkeypatch.setattr(video_upload_store, "probe_fps", fake)
    return calls


def test_plain_upload_reports_no_measurement_and_never_probes(client, monkeypatch):
    """The path every ordinary upload takes: no query arguments at all. Both
    fields are null and NOTHING is probed -- the response is byte-identical to
    what it was before the two fields existed."""
    monkeypatch.setattr(video_upload_store, "cut_range_mp4", _no_cut_allowed)
    monkeypatch.setattr(video_upload_store, "probe_fps", _no_probe_allowed)

    def boom(p):
        raise AssertionError("frame_count must not be called without max_frames")

    monkeypatch.setattr(video_upload_store, "frame_count", boom)

    r = client.post("/api/v1/upload/video", files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["frame_count"] is None
    assert body["fps"] is None


def test_trim_window_without_max_frames_reports_no_measurement(client, monkeypatch):
    """The V2V ribbon-trim upload did not ask to be measured, so it is not --
    even though the cut it ran happens to know both numbers. Keeping this path
    unchanged is the point; the end source always sends max_frames as well (see
    the test below)."""
    _install_recording_cut(monkeypatch)
    monkeypatch.setattr(video_upload_store, "probe_fps", _no_probe_allowed)

    r = client.post(
        "/api/v1/upload/video",
        params={"trim_start_sec": 1.0, "trim_duration_sec": 2.0},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is True
    assert body["frame_count"] is None
    assert body["fps"] is None


def test_max_frames_under_the_limit_reports_the_stored_length_and_one_probe(client, monkeypatch):
    """Nothing was cut, so the file's length IS the count the ceiling comparison
    already measured. The rate costs exactly ONE added ffprobe -- the entire
    runtime price of the feature."""
    monkeypatch.setattr(video_upload_store, "cut_range_mp4", _no_cut_allowed)
    monkeypatch.setattr(video_upload_store, "frame_count", lambda p: 137)
    probes = _install_recording_probe(monkeypatch, 30.0)

    r = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 685},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is False
    assert body["frame_count"] == 137
    assert body["fps"] == 30.0
    assert len(probes) == 1


def test_max_frames_truncation_reports_the_cut_files_numbers(client, monkeypatch):
    """After a cut the answer must describe what was KEPT (685), never the
    original length (2000) -- the band length is derived from it, and an
    over-reported count would ask the engine for frames that are not there.
    cut_range_mp4 already returns both, so no extra probe runs."""
    calls = _install_recording_cut_with_frames(monkeypatch)
    monkeypatch.setattr(video_upload_store, "frame_count", lambda p: 2000)
    monkeypatch.setattr(video_upload_store, "probe_fps", _no_probe_allowed)

    r = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 685},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is True
    assert body["frame_count"] == 685  # written_frames, not the 2000 we had
    assert body["fps"] == 30.0  # source_fps from the cut's own info
    assert len(calls) == 1


def test_trim_window_with_max_frames_reports_the_cut_files_numbers(client, monkeypatch):
    """End source + ribbon trim: the app sends BOTH, the window wins over the
    ceiling, and the measurement still has to come back -- otherwise the band
    length would fall back to a duration estimate on the one path that trims."""
    _install_recording_cut(monkeypatch)
    monkeypatch.setattr(video_upload_store, "probe_fps", _no_probe_allowed)

    r = client.post(
        "/api/v1/upload/video",
        params={"trim_start_sec": 1.0, "trim_duration_sec": 2.0, "max_frames": 685},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trimmed"] is True
    assert body["frame_count"] == 60  # the recording stand-in's written_frames
    assert body["fps"] == 30.0


def test_measurement_failures_report_unknown_not_an_error(client, monkeypatch):
    """Every failure mode degrades to (null, null) with HTTP 200. "Unknown" is a
    real answer the client falls back from; a 4xx/5xx here would turn a working
    upload into a broken one."""
    # (a) the probe itself fails
    monkeypatch.setattr(video_upload_store, "cut_range_mp4", _no_cut_allowed)
    monkeypatch.setattr(
        video_upload_store, "frame_count",
        lambda p: (_ for _ in ()).throw(FFmpegError("ffprobe not found on PATH")),
    )
    r = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 685},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["frame_count"] is None
    assert r.json()["fps"] is None

    # (b) the cut fails after a successful probe: the untouched original comes
    # back, and the count we measured describes a file we did NOT keep... except
    # we did keep it -- but it is still reported unknown, deliberately, so the
    # fallback path is the one exercised whenever anything went wrong.
    monkeypatch.setattr(video_upload_store, "frame_count", lambda p: 2000)

    def boom(src, out, start_sec, duration_sec, *, start_frame=None, num_frames=None):
        raise FFmpegError("ffmpeg not found on PATH")

    monkeypatch.setattr(video_upload_store, "cut_range_mp4", boom)
    r2 = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 685},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["frame_count"] is None
    assert r2.json()["fps"] is None

    # (c) ffprobe present but the rate is unreadable -> a count without a rate
    monkeypatch.setattr(video_upload_store, "cut_range_mp4", _no_cut_allowed)
    monkeypatch.setattr(video_upload_store, "frame_count", lambda p: 137)
    monkeypatch.setattr(video_upload_store, "probe_fps", lambda p: None)
    r3 = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 685},
        files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")},
    )
    assert r3.status_code == 200, r3.text
    assert r3.json()["frame_count"] == 137
    assert r3.json()["fps"] is None


def test_real_ffmpeg_measurement_end_to_end(client, tmp_path):
    """No stand-ins: a real 10-frame 10 fps mp4 through both max_frames paths."""
    src = tmp_path / "src.mp4"
    frames = [Image.new("RGB", (64, 64), (i * 20 % 256, 90, 160)) for i in range(10)]
    video_io.encode_frames_to_mp4(frames, src, frame_rate=10.0)
    payload = src.read_bytes()

    r_under = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 20},
        files={"file": ("src.mp4", payload, "video/mp4")},
    )
    assert r_under.status_code == 200, r_under.text
    assert r_under.json()["frame_count"] == 10
    assert abs(r_under.json()["fps"] - 10.0) < 0.01

    r_over = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 4},
        files={"file": ("src.mp4", payload, "video/mp4")},
    )
    assert r_over.status_code == 200, r_over.text
    body = r_over.json()
    assert body["trimmed"] is True
    assert body["frame_count"] == 4
    stored = client.app_context.video_upload_store.path_for(body["video_id"])
    # the reported count really is the stored file's own length
    assert video_io.frame_count(stored) == body["frame_count"]
    assert abs(body["fps"] - 10.0) < 0.01


def test_real_ffmpeg_max_frames_end_to_end(client, tmp_path):
    src = tmp_path / "src.mp4"
    frames = [Image.new("RGB", (64, 64), (i * 20 % 256, 90, 160)) for i in range(10)]
    video_io.encode_frames_to_mp4(frames, src, frame_rate=10.0)
    payload = src.read_bytes()

    # Under the limit -> byte-identical, no re-encode.
    r_under = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 20},
        files={"file": ("src.mp4", payload, "video/mp4")},
    )
    assert r_under.status_code == 200, r_under.text
    assert r_under.json()["trimmed"] is False
    stored_under = client.app_context.video_upload_store.path_for(r_under.json()["video_id"])
    assert stored_under.read_bytes() == payload

    # Over the limit -> first max_frames frames only.
    r_over = client.post(
        "/api/v1/upload/video",
        params={"max_frames": 4},
        files={"file": ("src.mp4", payload, "video/mp4")},
    )
    assert r_over.status_code == 200, r_over.text
    body = r_over.json()
    assert body["trimmed"] is True
    stored_over = client.app_context.video_upload_store.path_for(body["video_id"])
    assert video_io.frame_count(stored_over) == 4
    assert body["size_bytes"] == stored_over.stat().st_size
