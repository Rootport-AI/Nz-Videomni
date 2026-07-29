"""Unit tests for the Batch A2V execution engine (gradio_ui/batch.py).

Driven fully offline: ApiClient is fed an ``httpx.MockTransport`` (the
``_make_client`` pattern from tests/test_gradio_handlers.py) and the runner is
driven with ``sync=True`` so the whole batch runs inline — no live server, no
real thread scheduling, deterministic assertions on the emitted
/generate/chain payloads, the copied output files, and the CSV manifest.
"""

from __future__ import annotations

import json
import os
import threading
import time
import wave
from pathlib import Path

import httpx
import pytest

from gradio_ui.api_client import ApiClient
from gradio_ui.batch import (
    STATE_IDLE,
    BatchRunner,
    BatchSnapshot,
    compose_prompt,
    get_runner,
)
from gradio_ui.handlers import suggest_frames_for_audio
from gradio_ui.manifest import (
    IMAGE_SHARED,
    STAT_DONE,
    STAT_FAILED,
    STAT_GENERATING,
    STAT_SKIP,
    STAT_WAITING,
    BatchRow,
    read_manifest,
    write_manifest_atomic,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers.
# --------------------------------------------------------------------------- #
def _make_client(handler, *, api_key: str | None = "secret") -> ApiClient:
    transport = httpx.MockTransport(handler)
    hc = httpx.Client(transport=transport)
    return ApiClient("http://test", api_key=api_key, client=hc)


def _write_wav(path: Path, seconds: float = 1.0, rate: int = 100) -> None:
    n = int(round(seconds * rate))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n)


def _snapshot(wav_dir: Path, out_dir: Path, **overrides) -> BatchSnapshot:
    kw = dict(
        wav_dir=str(wav_dir),
        out_dir=str(out_dir),
        prompt_common="base",
        negative="bad",
        prompt_mode="add",
        width=512,
        height=320,
        crop_output=None,
        frame_rate=24.0,
        seed=7,
        loras=[],
        shared_images=[],
        use_adapter=False,
        ref_video_path=None,
        control_adherence=1.0,
        reference_strength=1.0,
        poll_interval=0.0,
        poll_timeout_s=30.0,
    )
    kw.update(overrides)
    return BatchSnapshot(**kw)


class _Server:
    """A tiny stateful mock of the frozen REST API for the batch flow.

    upload/* -> ids, generate/chain -> a fresh job id (captured payloads),
    jobs/{id} -> a status that immediately reports ``completed`` (or ``failed``
    for wavs named in ``fail_wavs``), jobs/{id}/video -> mp4 bytes. Records
    every captured chain payload and every upload for assertions.
    """

    def __init__(self, fail_jobs=()):
        self.fail_jobs = set(fail_jobs)      # job ids that report "failed"
        self.payloads = []                   # captured /generate/chain bodies
        self.image_uploads = 0
        self.audio_uploads = 0
        self.video_uploads = 0
        self.chain_calls = 0
        self._job_seq = 0
        self.on_chain = None                 # optional hook(job_id) after submit

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/upload/audio"):
            self.audio_uploads += 1
            return httpx.Response(200, json={"audio_id": f"aud-{self.audio_uploads}"})
        if path.endswith("/upload/image"):
            self.image_uploads += 1
            return httpx.Response(200, json={"image_id": f"img-{self.image_uploads}"})
        if path.endswith("/upload/video"):
            self.video_uploads += 1
            return httpx.Response(200, json={"video_id": f"vid-{self.video_uploads}"})
        if path.endswith("/generate/chain"):
            self.chain_calls += 1
            self._job_seq += 1
            job_id = f"job-{self._job_seq}"
            self.payloads.append((job_id, json.loads(request.content)))
            if self.on_chain is not None:
                self.on_chain(job_id)
            return httpx.Response(202, json={"job_id": job_id})
        if path.endswith("/video"):
            return httpx.Response(200, content=b"MP4DATA")
        if "/jobs/" in path:
            job_id = path.rsplit("/jobs/", 1)[1]
            if job_id in self.fail_jobs:
                return httpx.Response(200, json={"status": "failed", "error": "boom"})
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(404, json={"error": "unexpected"})


def _rows(*specs) -> list[BatchRow]:
    """Build BatchRows from (wav, stat, image, prompt) tuples (image/prompt
    optional). ``image`` defaults to "" (pure audio-only A2V, no keyframe) —
    pass ``IMAGE_SHARED`` explicitly to exercise the shared-image path."""
    out = []
    for i, spec in enumerate(specs, start=1):
        wav = spec[0]
        stat = spec[1] if len(spec) > 1 else STAT_WAITING
        image = spec[2] if len(spec) > 2 else ""
        prompt = spec[3] if len(spec) > 3 else ""
        out.append(BatchRow(queue=i, wav=wav, image=image, prompt=prompt,
                            stat=stat, frames=49))
    return out


def _wait_until(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------- #
# compose_prompt (pure).
# --------------------------------------------------------------------------- #
def test_compose_prompt_add_and_replace_and_empty():
    assert compose_prompt("base", "extra", "add") == "base extra"
    assert compose_prompt("base", "extra", "replace") == "extra"
    assert compose_prompt("base", "", "add") == "base"          # empty row -> common
    assert compose_prompt("base", "   ", "replace") == "base"   # ws row -> common
    assert compose_prompt("", "only", "add") == "only"          # empty common
    assert compose_prompt("base", "extra", "unknown") == "base extra"  # default add


# --------------------------------------------------------------------------- #
# 1) three rows all Done: payloads, outputs, CSV.
# --------------------------------------------------------------------------- #
def test_three_rows_all_done(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    for name in ("a.wav", "b.wav", "c.wav"):
        _write_wav(wav_dir / name)
    img = tmp_path / "kf.png"
    img.write_bytes(b"\x89PNG\r\n")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir,
                     shared_images=[(str(img), 0, 0.8)],
                     prompt_mode="add")
    rows = _rows(("a.wav", STAT_WAITING, IMAGE_SHARED, "smiling"),
                 ("b.wav", STAT_WAITING, IMAGE_SHARED, ""),
                 ("c.wav", STAT_WAITING, IMAGE_SHARED, "waving"))

    runner = BatchRunner()
    started, _ = runner.start(snap, rows, api, sync=True)
    assert started is True
    assert runner.state == STATE_IDLE

    # every row Done
    assert [r.stat for r in rows] == [STAT_DONE, STAT_DONE, STAT_DONE]

    # payloads: compose_prompt reflected, row.frames on the clip, conditioning.
    prompts = [p["prompt"] for _jid, p in server.payloads]
    assert prompts == ["base smiling", "base", "base waving"]
    for _jid, p in server.payloads:
        assert p["clips"][0]["num_frames"] == 49
        assert p["clips"][0]["conditioning_images"] == [
            {"image_id": "img-1", "frame_idx": 0, "strength": 0.8}
        ]
        assert p["source_audio"]["audio_id"].startswith("aud-")
        assert p["width"] == 512 and p["height"] == 320
        assert p["seed"] == 7

    # output mp4s copied into out_dir under the wav basename.
    assert (out_dir / "a.mp4").read_bytes() == b"MP4DATA"
    assert (out_dir / "b.mp4").exists()
    assert (out_dir / "c.mp4").exists()
    assert [r.output for r in rows] == ["a.mp4", "b.mp4", "c.mp4"]

    # CSV reflects the final Done state.
    disk = read_manifest(wav_dir)
    assert [r.stat for r in disk] == [STAT_DONE, STAT_DONE, STAT_DONE]
    assert [r.output for r in disk] == ["a.mp4", "b.mp4", "c.mp4"]


# --------------------------------------------------------------------------- #
# 2) server "failed" on row 2 -> row 2 Failed, row 3 still runs.
# --------------------------------------------------------------------------- #
def test_middle_row_failed_batch_continues(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    for name in ("a.wav", "b.wav", "c.wav"):
        _write_wav(wav_dir / name)
    out_dir = tmp_path / "out"

    # job-2 (the 2nd chain submission = row b) reports failed.
    server = _Server(fail_jobs={"job-2"})
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(("a.wav",), ("b.wav",), ("c.wav",))

    runner = BatchRunner()
    runner.start(snap, rows, api, sync=True)

    assert [r.stat for r in rows] == [STAT_DONE, STAT_FAILED, STAT_DONE]
    assert "boom" in rows[1].error
    assert server.chain_calls == 3           # row 3 still submitted after the failure
    assert (out_dir / "a.mp4").exists()
    assert not (out_dir / "b.mp4").exists()  # failed row produced no output
    assert (out_dir / "c.mp4").exists()


# --------------------------------------------------------------------------- #
# 3) resume: Done/Skip skipped (zero API), Failed/Generating remnants re-run.
# --------------------------------------------------------------------------- #
def test_resume_skips_done_and_skip_reruns_failed(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    for name in ("a.wav", "b.wav", "c.wav", "d.wav"):
        _write_wav(wav_dir / name)
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(
        ("a.wav", STAT_DONE),         # already done -> skip
        ("b.wav", STAT_SKIP),         # skipped -> skip
        ("c.wav", STAT_FAILED),       # failed -> re-run
        ("d.wav", STAT_GENERATING),   # crash remnant -> re-run
    )
    rows[0].output = "a.mp4"

    runner = BatchRunner()
    runner.start(snap, rows, api, sync=True)

    # only the failed + generating remnants were submitted.
    assert server.chain_calls == 2
    assert server.audio_uploads == 2
    assert [r.stat for r in rows] == [STAT_DONE, STAT_SKIP, STAT_DONE, STAT_DONE]
    # the pre-existing Done row is untouched (no new output overwrite).
    assert rows[0].output == "a.mp4"
    submitted_audio = server.audio_uploads
    assert submitted_audio == 2


# --------------------------------------------------------------------------- #
# 4) stop request after row 1 -> rows 2+ never submitted.
# --------------------------------------------------------------------------- #
def test_stop_after_first_row_leaves_rest_unsubmitted(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    for name in ("a.wav", "b.wav", "c.wav"):
        _write_wav(wav_dir / name)
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    runner = BatchRunner()

    # After the first chain submission completes, set stop_event so the loop
    # halts before submitting row 2.
    def after_chain(job_id):
        if job_id == "job-1":
            runner.stop_event.set()
    server.on_chain = after_chain

    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(("a.wav",), ("b.wav",), ("c.wav",))
    runner.start(snap, rows, api, sync=True)

    assert server.chain_calls == 1
    assert rows[0].stat == STAT_DONE
    assert rows[1].stat == STAT_WAITING   # never submitted
    assert rows[2].stat == STAT_WAITING
    assert runner.state == STATE_IDLE     # always returns to idle


# --------------------------------------------------------------------------- #
# 5) double-start guard: a second start while running returns False.
# --------------------------------------------------------------------------- #
def test_double_start_guard(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    release = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/upload/audio"):
            return httpx.Response(200, json={"audio_id": "aud-1"})
        if path.endswith("/generate/chain"):
            return httpx.Response(202, json={"job_id": "job-1"})
        if path.endswith("/video"):
            return httpx.Response(200, content=b"MP4DATA")
        if "/jobs/" in path:
            # Stay "running" until the test releases us.
            status = "completed" if release.is_set() else "running"
            return httpx.Response(200, json={"status": status})
        return httpx.Response(404, json={})

    api = _make_client(handler)
    snap = _snapshot(wav_dir, out_dir, poll_interval=0.01)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, _ = runner.start(snap, rows, api, sync=False)  # background thread
    assert started is True
    assert _wait_until(lambda: runner.state == "running")

    # Second start while running -> rejected.
    again, reason = runner.start(snap, _rows(("a.wav",)), api)
    assert again is False
    assert "running" in reason

    release.set()
    assert _wait_until(lambda: runner.state == STATE_IDLE)
    assert rows[0].stat == STAT_DONE


# --------------------------------------------------------------------------- #
# 6) image_id cache: the same Shared image over two rows uploads once.
# --------------------------------------------------------------------------- #
def test_shared_image_uploaded_once(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    _write_wav(wav_dir / "b.wav")
    img = tmp_path / "kf.png"
    img.write_bytes(b"\x89PNG\r\n")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, shared_images=[(str(img), 0, 0.9)])
    rows = _rows(("a.wav", STAT_WAITING, IMAGE_SHARED),
                 ("b.wav", STAT_WAITING, IMAGE_SHARED))

    runner = BatchRunner()
    runner.start(snap, rows, api, sync=True)

    assert server.image_uploads == 1        # cached across both rows
    assert server.audio_uploads == 2        # audio still uploaded per row
    for _jid, p in server.payloads:
        assert p["clips"][0]["conditioning_images"][0]["image_id"] == "img-1"


# --------------------------------------------------------------------------- #
# 7) gacha numbering: a pre-existing same-name mp4 -> "_2" output.
# --------------------------------------------------------------------------- #
def test_output_gets_unique_suffix_when_name_exists(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "a.mp4").write_bytes(b"OLD")   # pre-existing output

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    runner.start(snap, rows, api, sync=True)

    assert (out_dir / "a.mp4").read_bytes() == b"OLD"   # original untouched
    assert (out_dir / "a_2.mp4").read_bytes() == b"MP4DATA"
    assert rows[0].output == "a_2.mp4"


# --------------------------------------------------------------------------- #
# 8) defensive validation: empty shared_images + a Shared row -> start fails.
# --------------------------------------------------------------------------- #
def test_start_rejects_shared_row_without_shared_images(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, shared_images=[])
    rows = _rows(("a.wav", STAT_WAITING, IMAGE_SHARED))

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is False
    assert "shared" in reason.lower()
    assert server.chain_calls == 0
    assert rows[0].stat == STAT_WAITING     # untouched


def test_start_rejects_shared_row_when_only_non_first_slot_set(tmp_path):
    """A shared_images entry from slot 2+ (frame_idx > 0) does NOT count as the
    "first keyframe" (slot 1, frame 0) — a Shared-image target row must still
    block the start, and nothing gets touched."""
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    img = tmp_path / "kf.png"
    img.write_bytes(b"\x89PNG\r\n")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    # only slot 2 (frame_idx=5) is set — slot 1 / frame 0 is not.
    snap = _snapshot(wav_dir, out_dir, shared_images=[(str(img), 5, 0.8)])
    rows = _rows(("a.wav", STAT_WAITING, IMAGE_SHARED))

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is False
    assert "shared" in reason.lower()
    assert server.chain_calls == 0
    assert server.image_uploads == 0
    assert rows[0].stat == STAT_WAITING     # untouched
    assert not (wav_dir / "batch_a2v_manifest.csv").exists()   # CSV untouched


def test_start_allows_shared_row_when_first_slot_set(tmp_path):
    """A shared_images entry at frame_idx == 0 (slot 1) is the "first keyframe"
    and is sufficient on its own to unblock Shared-image target rows, even
    when other slots are also present."""
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    img = tmp_path / "kf.png"
    img.write_bytes(b"\x89PNG\r\n")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, shared_images=[(str(img), 0, 0.8)])
    rows = _rows(("a.wav", STAT_WAITING, IMAGE_SHARED))

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True
    assert reason == "completed"
    assert server.chain_calls == 1
    assert rows[0].stat == STAT_DONE


def test_start_allows_no_first_slot_when_all_targets_have_own_image(tmp_path):
    """No first keyframe set at all — but every target row carries its own
    individual image, so the start is allowed (no Shared row to block on)."""
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    _write_wav(wav_dir / "b.wav")
    (wav_dir / "face_a.png").write_bytes(b"\x89PNG\r\n")
    (wav_dir / "face_b.png").write_bytes(b"\x89PNG\r\n")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, shared_images=[])
    rows = _rows(("a.wav", STAT_WAITING, "face_a.png"),
                 ("b.wav", STAT_WAITING, "face_b.png"))

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True
    assert reason == "completed"
    assert server.chain_calls == 2
    assert [r.stat for r in rows] == [STAT_DONE, STAT_DONE]


def test_start_rejects_missing_wav_dir_and_no_targets(tmp_path):
    out_dir = tmp_path / "out"
    server = _Server()
    api = _make_client(server.handler)

    # missing wav_dir
    snap = _snapshot(tmp_path / "does_not_exist", out_dir)
    ok, reason = BatchRunner().start(snap, _rows(("a.wav",)), api, sync=True)
    assert ok is False and "not found" in reason

    # wav_dir exists but zero unfinished rows
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    snap2 = _snapshot(wav_dir, out_dir)
    done = _rows(("a.wav", STAT_DONE), ("b.wav", STAT_SKIP))
    ok2, reason2 = BatchRunner().start(snap2, done, api, sync=True)
    assert ok2 is False and "no rows" in reason2


# --------------------------------------------------------------------------- #
# Individual (non-Shared) image + summary + singleton.
# --------------------------------------------------------------------------- #
def test_individual_image_attaches_frame0_strength1(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    (wav_dir / "face.png").write_bytes(b"\x89PNG\r\n")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(("a.wav", STAT_WAITING, "face.png", "hi"))

    runner = BatchRunner()
    runner.start(snap, rows, api, sync=True)

    _jid, p = server.payloads[0]
    assert p["clips"][0]["conditioning_images"] == [
        {"image_id": "img-1", "frame_idx": 0, "strength": 1.0}
    ]


# --------------------------------------------------------------------------- #
# M-1: a per-row image lives in a SEPARATE image folder (img_dir != wav_dir).
# The bare filename must resolve against img_dir, not wav_dir (where the UI's
# choices came from), so the upload succeeds instead of FileNotFound -> Failed.
# --------------------------------------------------------------------------- #
def test_individual_image_resolved_from_separate_img_dir(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    img_dir = tmp_path / "imgs"      # distinct from the audio folder
    img_dir.mkdir()
    (img_dir / "face.png").write_bytes(b"\x89PNG\r\n")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, img_dir=str(img_dir))
    rows = _rows(("a.wav", STAT_WAITING, "face.png", "hi"))

    runner = BatchRunner()
    started, _ = runner.start(snap, rows, api, sync=True)
    assert started is True

    # Resolved from img_dir and uploaded (no FileNotFoundError -> no Failed row).
    assert rows[0].stat == STAT_DONE
    assert server.image_uploads == 1
    _jid, p = server.payloads[0]
    assert p["clips"][0]["conditioning_images"] == [
        {"image_id": "img-1", "frame_idx": 0, "strength": 1.0}
    ]


def test_individual_image_falls_back_to_wav_dir_when_img_dir_unset(tmp_path):
    # img_dir empty -> the old behaviour: resolve the bare name against wav_dir.
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    (wav_dir / "face.png").write_bytes(b"\x89PNG\r\n")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)     # img_dir defaults to ""
    rows = _rows(("a.wav", STAT_WAITING, "face.png", "hi"))

    runner = BatchRunner()
    runner.start(snap, rows, api, sync=True)

    assert rows[0].stat == STAT_DONE
    assert server.image_uploads == 1


# --------------------------------------------------------------------------- #
# S-1: the fetch_video() temp mp4 is deleted after being copied to the output
# folder (so an overnight run does not accumulate temp files in %TEMP%).
# --------------------------------------------------------------------------- #
def test_completed_temp_video_is_removed(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(("a.wav",))

    # Spy on fetch_video to capture the temp path the runner receives.
    captured: list[str] = []
    orig_fetch = api.fetch_video

    def _spy(job_id):
        p = orig_fetch(job_id)
        captured.append(p)
        return p

    api.fetch_video = _spy

    runner = BatchRunner()
    runner.start(snap, rows, api, sync=True)

    assert rows[0].stat == STAT_DONE
    assert (out_dir / "a.mp4").read_bytes() == b"MP4DATA"   # copied to output
    assert captured, "fetch_video was not called"
    assert not os.path.exists(captured[0])                  # temp mp4 cleaned up


def test_summary_counts(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    for name in ("a.wav", "b.wav"):
        _write_wav(wav_dir / name)
    out_dir = tmp_path / "out"

    server = _Server(fail_jobs={"job-2"})
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(("a.wav",), ("b.wav",))

    runner = BatchRunner()
    runner.start(snap, rows, api, sync=True)
    s = runner.summary()
    assert s["done"] == 1 and s["failed"] == 1
    assert s["state"] == STATE_IDLE


# --------------------------------------------------------------------------- #
# 2nd-round FB — modification C: start-time frame / 481-Skip re-judgment.
# --------------------------------------------------------------------------- #
def test_start_remarks_over481_row_to_skip_and_excludes_it(tmp_path):
    # A row hand-rewound to Waiting whose duration now overruns the 481-frame
    # cap (at the snapshot's frame rate) is re-marked Skip at start-time and is
    # never submitted; the rest of the batch still runs. The re-mark persists.
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "ok.wav")
    _write_wav(wav_dir / "big.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)                 # frame_rate 24, common "base"
    # ok.wav: 1s -> stays a target; big.wav: 25s @ 24fps -> raw 8n+1 count > 481.
    # image="" (audio-only, no keyframe) so the image foolproof gate is n/a here.
    rows = [
        BatchRow(queue=1, wav="ok.wav", duration_s=1.0, stat=STAT_WAITING,
                 image="", frames=25),
        BatchRow(queue=2, wav="big.wav", duration_s=25.0, stat=STAT_WAITING,
                 image="", frames=481),
    ]

    runner = BatchRunner()
    started, _ = runner.start(snap, rows, api, sync=True)
    assert started is True
    assert rows[1].stat == STAT_SKIP and rows[1].skip_reason == "over-481f"
    assert rows[0].stat == STAT_DONE
    assert server.chain_calls == 1                     # only ok.wav submitted
    assert server.audio_uploads == 1

    # the Skip re-mark is flushed to the CSV.
    disk = {r.wav: r for r in read_manifest(wav_dir)}
    assert disk["big.wav"].stat == STAT_SKIP
    assert disk["big.wav"].skip_reason == "over-481f"


def test_start_recomputes_frames_at_snapshot_fps(tmp_path):
    # A stale ``frames`` value is recomputed for the snapshot's frame rate at
    # start-time, and the recomputed count is what reaches the payload.
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    new_fps = 30.0
    snap = _snapshot(wav_dir, out_dir, frame_rate=new_fps)
    rows = [BatchRow(queue=1, wav="a.wav", duration_s=3.0, stat=STAT_WAITING,
                     image="", frames=99)]              # deliberately stale

    runner = BatchRunner()
    runner.start(snap, rows, api, sync=True)

    expected = suggest_frames_for_audio(3.0, new_fps)
    assert expected != 99
    assert rows[0].frames == expected
    _jid, p = server.payloads[0]
    assert p["clips"][0]["num_frames"] == expected


# --------------------------------------------------------------------------- #
# 2nd-round FB — modification D: start-time foolproof (prompt / image).
# --------------------------------------------------------------------------- #
def test_start_blocks_add_mode_empty_common_prompt(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, prompt_common="", prompt_mode="add")
    rows = [BatchRow(queue=1, wav="a.wav", duration_s=1.0, stat=STAT_WAITING,
                     image="", prompt="", frames=25)]
    # Seed a manifest so we can prove a blocked start leaves the CSV untouched.
    write_manifest_atomic(wav_dir, rows)
    before = (wav_dir / "batch_a2v_manifest.csv").read_bytes()

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is False
    assert reason == "prompt-empty-add"
    assert server.chain_calls == 0
    assert rows[0].stat == STAT_WAITING                # rows untouched
    assert (wav_dir / "batch_a2v_manifest.csv").read_bytes() == before  # CSV untouched


def test_start_allows_replace_empty_common_when_all_rows_have_prompt(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    _write_wav(wav_dir / "b.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, prompt_common="", prompt_mode="replace")
    rows = [BatchRow(queue=1, wav="a.wav", duration_s=1.0, stat=STAT_WAITING,
                     image="", prompt="hello", frames=25),
            BatchRow(queue=2, wav="b.wav", duration_s=1.0, stat=STAT_WAITING,
                     image="", prompt="world", frames=25)]

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason
    assert server.chain_calls == 2
    assert [r.stat for r in rows] == [STAT_DONE, STAT_DONE]
    assert [p["prompt"] for _j, p in server.payloads] == ["hello", "world"]


def test_start_blocks_replace_empty_common_with_some_empty_rows(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    for n in ("a.wav", "b.wav", "c.wav"):
        _write_wav(wav_dir / n)
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, prompt_common="   ", prompt_mode="replace")
    rows = [BatchRow(queue=1, wav="a.wav", duration_s=1.0, stat=STAT_WAITING,
                     image="", prompt="hi", frames=25),
            BatchRow(queue=2, wav="b.wav", duration_s=1.0, stat=STAT_WAITING,
                     image="", prompt="", frames=25),       # empty
            BatchRow(queue=3, wav="c.wav", duration_s=1.0, stat=STAT_WAITING,
                     image="", prompt="   ", frames=25)]     # whitespace-only

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is False
    assert reason == "prompt-rows-empty:2"                   # two empty rows
    assert server.chain_calls == 0


def test_start_allows_no_shared_image_when_all_rows_have_own_image(tmp_path):
    # Regression: with no shared keyframe but every target carrying its own
    # per-row image, the image foolproof passes and the batch starts.
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    (wav_dir / "face.png").write_bytes(b"\x89PNG\r\n")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, shared_images=[])
    rows = [BatchRow(queue=1, wav="a.wav", duration_s=1.0, stat=STAT_WAITING,
                     image="face.png", prompt="", frames=25)]

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason
    assert rows[0].stat == STAT_DONE
    assert server.image_uploads == 1


def test_get_runner_is_singleton():
    assert get_runner() is get_runner()


# --------------------------------------------------------------------------- #
# NAG (Normalized Attention Guidance / non-CFG Negative): BatchSnapshot's four
# nag_* fields must reach every row's /generate/chain payload (via
# build_a2v_chain_payload) when enabled, and stay entirely absent by default.
# --------------------------------------------------------------------------- #
def test_batch_nag_enabled_snapshot_adds_four_keys_to_payload(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, negative="blurry",
                     nag_enabled=True, nag_scale=9.0, nag_tau=3.0, nag_alpha=0.4)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert p["nag_enabled"] is True
    assert p["nag_scale"] == 9.0
    assert p["nag_tau"] == 3.0
    assert p["nag_alpha"] == 0.4


def test_batch_vsf_enabled_snapshot_adds_two_keys_to_payload(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, negative="blurry",
                     nag_enabled=True, neg_method="vsf", vsf_scale=2.0)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert p["neg_method"] == "vsf"
    assert p["vsf_scale"] == 2.0


def test_batch_default_snapshot_omits_nag_keys_from_payload(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    for key in ("nag_enabled", "nag_scale", "nag_tau", "nag_alpha",
                "neg_method", "vsf_scale"):
        assert key not in p
