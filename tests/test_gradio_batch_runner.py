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
    ROW_IMAGE_STRENGTH_DEFAULT,
    STATE_IDLE,
    BatchRunner,
    BatchSnapshot,
    compose_prompt,
    get_runner,
    prepare_batch_rows,
    row_image_strength,
    rows_have_lora_tokens,
)
from gradio_ui.handlers import (
    make_chain_handler,
    make_generate_handler,
    suggest_frames_for_audio,
)
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
from gradio_ui.presets import KF_MAX_SLOTS


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


def _start(runner: BatchRunner, snap: BatchSnapshot, rows, api, **kwargs):
    """``BatchRunner.start`` preceded by the UI-thread freeze that ui.py's
    ``dispatch()`` always performs first.

    Since §3-1 the runner sends ``BatchRow.send_prompt`` / ``BatchRow.loras``
    verbatim — composing the common + row prompt and parsing its
    ``<lora:...>`` tokens happens on the UI thread, where the known-name list
    and ``gr.Warning`` live. A test that called ``start()`` on raw rows would
    therefore submit an empty prompt, which is not a state the app can reach;
    this wrapper keeps every scenario below faithful to the real call order.
    Tests that need real tokens call :func:`prepare_batch_rows` themselves with
    a known-name list."""
    err = prepare_batch_rows(rows, snap.prompt_common, snap.prompt_mode, ())
    assert err is None, err
    return runner.start(snap, rows, api, **kwargs)


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
    started, _ = _start(runner, snap, rows, api, sync=True)
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
    _start(runner, snap, rows, api, sync=True)

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
    _start(runner, snap, rows, api, sync=True)

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
    _start(runner, snap, rows, api, sync=True)

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
    started, _ = _start(runner, snap, rows, api, sync=False)  # background thread
    assert started is True
    assert _wait_until(lambda: runner.state == "running")

    # Second start while running -> rejected.
    again, reason = _start(runner, snap, _rows(("a.wav",)), api)
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
    _start(runner, snap, rows, api, sync=True)

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
    _start(runner, snap, rows, api, sync=True)

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
    started, reason = _start(runner, snap, rows, api, sync=True)
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
    started, reason = _start(runner, snap, rows, api, sync=True)
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
    started, reason = _start(runner, snap, rows, api, sync=True)
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
    started, reason = _start(runner, snap, rows, api, sync=True)
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
    ok, reason = _start(BatchRunner(), snap, _rows(("a.wav",)), api, sync=True)
    assert ok is False and "not found" in reason

    # wav_dir exists but zero unfinished rows
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    snap2 = _snapshot(wav_dir, out_dir)
    done = _rows(("a.wav", STAT_DONE), ("b.wav", STAT_SKIP))
    ok2, reason2 = _start(BatchRunner(), snap2, done, api, sync=True)
    assert ok2 is False and "no rows" in reason2


# --------------------------------------------------------------------------- #
# Individual (non-Shared) image + summary + singleton.
#
# A row's own image is a keyframe like any other, so it is conditioned at the
# LEADING shared keyframe's strength (lowest frame_idx — the Generate tab's
# slots are not in frame order) and falls back to the app-wide keyframe default
# when there is no shared keyframe at all. The old fixed 1.0 made a row image
# the one keyframe in the app that ignored the slider.
# --------------------------------------------------------------------------- #
def test_row_image_strength_picks_lowest_frame_idx_else_default():
    # The pure rule, in isolation: (path, frame_idx, strength) triples.
    assert row_image_strength([]) == ROW_IMAGE_STRENGTH_DEFAULT
    assert row_image_strength([("a.png", 24, 0.9), ("b.png", 0, 0.55),
                               ("c.png", 8, 0.7)]) == 0.55
    # Slot order is irrelevant — only frame_idx decides which one leads.
    assert row_image_strength([("a.png", 5, 0.4)]) == 0.4


def test_individual_image_attaches_frame0_at_the_shared_strength(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    (wav_dir / "face.png").write_bytes(b"\x89PNG\r\n")
    kf_late = tmp_path / "kf_late.png"
    kf_late.write_bytes(b"\x89PNG\r\n")
    kf_first = tmp_path / "kf_first.png"
    kf_first.write_bytes(b"\x89PNG\r\n")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    # The leading shared keyframe (frame 0, strength 0.55) is listed LAST, so
    # a slot-order reading would pick 0.9.
    snap = _snapshot(wav_dir, out_dir,
                     shared_images=[(str(kf_late), 24, 0.9),
                                    (str(kf_first), 0, 0.55)])
    rows = _rows(("a.wav", STAT_WAITING, "face.png", "hi"))

    runner = BatchRunner()
    _start(runner, snap, rows, api, sync=True)

    _jid, p = server.payloads[0]
    assert p["clips"][0]["conditioning_images"] == [
        {"image_id": "img-1", "frame_idx": 0, "strength": 0.55}
    ]


def test_individual_image_without_shared_keyframe_uses_the_default_strength(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    (wav_dir / "face.png").write_bytes(b"\x89PNG\r\n")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)          # no shared keyframes at all
    rows = _rows(("a.wav", STAT_WAITING, "face.png", "hi"))

    runner = BatchRunner()
    _start(runner, snap, rows, api, sync=True)

    _jid, p = server.payloads[0]
    assert p["clips"][0]["conditioning_images"] == [
        {"image_id": "img-1", "frame_idx": 0,
         "strength": ROW_IMAGE_STRENGTH_DEFAULT}
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
    started, _ = _start(runner, snap, rows, api, sync=True)
    assert started is True

    # Resolved from img_dir and uploaded (no FileNotFoundError -> no Failed row).
    assert rows[0].stat == STAT_DONE
    assert server.image_uploads == 1
    _jid, p = server.payloads[0]
    assert p["clips"][0]["conditioning_images"] == [
        {"image_id": "img-1", "frame_idx": 0,
         "strength": ROW_IMAGE_STRENGTH_DEFAULT}
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
    _start(runner, snap, rows, api, sync=True)

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
    _start(runner, snap, rows, api, sync=True)

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
    _start(runner, snap, rows, api, sync=True)
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
    started, _ = _start(runner, snap, rows, api, sync=True)
    assert started is True
    assert rows[1].stat == STAT_SKIP and rows[1].skip_reason == "over-cap"
    assert rows[0].stat == STAT_DONE
    assert server.chain_calls == 1                     # only ok.wav submitted
    assert server.audio_uploads == 1

    # the Skip re-mark is flushed to the CSV.
    disk = {r.wav: r for r in read_manifest(wav_dir)}
    assert disk["big.wav"].stat == STAT_SKIP
    assert disk["big.wav"].skip_reason == "over-cap"


def test_start_rejudges_against_the_snapshot_frame_cap(tmp_path):
    """Docs/PENDING_TASKS_CLOSED.md's old §4-29 (closed 2026-09-01): the
    snapshot carries the Generate tab's frame count, so a row well under 481
    frames is re-marked Skip at start-time when it exceeds that cap — the
    same judgment "Set audios" made."""
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "ok.wav")
    _write_wav(wav_dir / "medium.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, num_frames=257)
    # medium.wav: 15s @ 24fps -> ~353 raw frames: under 481, over the 257 cap.
    rows = [
        BatchRow(queue=1, wav="ok.wav", duration_s=1.0, stat=STAT_WAITING,
                 image="", frames=25),
        BatchRow(queue=2, wav="medium.wav", duration_s=15.0, stat=STAT_WAITING,
                 image="", frames=353),
    ]

    runner = BatchRunner()
    started, _ = _start(runner, snap, rows, api, sync=True)
    assert started is True
    assert rows[1].stat == STAT_SKIP and rows[1].skip_reason == "over-cap"
    assert rows[0].stat == STAT_DONE
    assert server.chain_calls == 1                     # only ok.wav submitted


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
    _start(runner, snap, rows, api, sync=True)

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
    started, reason = _start(runner, snap, rows, api, sync=True)
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
    started, reason = _start(runner, snap, rows, api, sync=True)
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
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is False
    assert reason == "prompt-rows-empty:2"                   # two empty rows
    assert server.chain_calls == 0


def test_start_blocks_a_tag_only_common_prompt_in_add_mode(tmp_path):
    # A card prompt of nothing but <lora:...> tokens is EMPTY once the parser
    # has had it: the freeze leaves send_prompt "", and every row would then
    # 422 on GenerateChainRequest.prompt (min_length=1). The preflight has to
    # judge the tag-stripped text, not the raw cell it is handed verbatim.
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, prompt_common="<lora:A:0.8>",
                     prompt_mode="add")
    rows = [BatchRow(queue=1, wav="a.wav", duration_s=1.0, stat=STAT_WAITING,
                     image="", prompt="", frames=25)]
    # The real dispatch freezes first, with the known-name list to hand.
    assert prepare_batch_rows(rows, snap.prompt_common, snap.prompt_mode,
                              ("A",)) is None
    assert rows[0].send_prompt == ""          # nothing left to send

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is False
    assert reason == "prompt-empty-add"
    assert server.chain_calls == 0


def test_start_blocks_a_tag_only_row_prompt_in_replace_mode(tmp_path):
    # Same rule one level down: in "replace" mode with a tag-only card prompt,
    # a row whose own prompt is tags alone counts toward the empty-row tally.
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    for n in ("a.wav", "b.wav"):
        _write_wav(wav_dir / n)
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, prompt_common="<lora:A:0.8>",
                     prompt_mode="replace")
    rows = [BatchRow(queue=1, wav="a.wav", duration_s=1.0, stat=STAT_WAITING,
                     image="", prompt="a cat", frames=25),
            BatchRow(queue=2, wav="b.wav", duration_s=1.0, stat=STAT_WAITING,
                     image="", prompt="<lora:B:0.5>", frames=25)]  # tags only
    assert prepare_batch_rows(rows, snap.prompt_common, snap.prompt_mode,
                              ("A", "B")) is None

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is False
    assert reason == "prompt-rows-empty:1"
    assert server.chain_calls == 0


def test_start_allows_a_common_prompt_with_a_tag_beside_real_text(tmp_path):
    # The other side of the same rule: a token NEXT TO real text still leaves a
    # body, so the batch starts and the stripped text is what goes on the wire.
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, prompt_common="a cat <lora:A:0.8>",
                     prompt_mode="add")
    rows = [BatchRow(queue=1, wav="a.wav", duration_s=1.0, stat=STAT_WAITING,
                     image="", prompt="", frames=25)]
    assert prepare_batch_rows(rows, snap.prompt_common, snap.prompt_mode,
                              ("A",)) is None

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason
    assert server.chain_calls == 1
    _jid, p = server.payloads[0]
    assert p["prompt"] == "a cat"
    assert p["loras"] == [{"name": "A", "strength": 0.8}]


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
    started, reason = _start(runner, snap, rows, api, sync=True)
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
    started, reason = _start(runner, snap, rows, api, sync=True)
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
    started, reason = _start(runner, snap, rows, api, sync=True)
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
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    for key in ("nag_enabled", "nag_scale", "nag_tau", "nag_alpha",
                "neg_method", "vsf_scale"):
        assert key not in p


# --------------------------------------------------------------------------- #
# Acceleration: the batch runner builds every row's payload from the SNAPSHOT,
# not from handler arguments, so attention_backend has to travel through
# BatchSnapshot -- otherwise an overnight batch would silently stay on sdpa
# while the UI claims sage is selected.
# --------------------------------------------------------------------------- #
def test_batch_sage_snapshot_adds_attention_backend_to_payload(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, attention_backend="sage")
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert p["attention_backend"] == "sage"
    assert list(p.keys())[-1] == "attention_backend"


def test_batch_default_snapshot_omits_attention_backend(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert "attention_backend" not in p


# --------------------------------------------------------------------------- #
# Block-swap prefetch: same reasoning as attention_backend above -- the
# snapshot is the only path from the Settings-tab checkbox into a batch row's
# payload, so an overnight batch would silently drift from the checkbox
# without this wiring. S4 (2026-08-01) flipped the default to True (real-
# device gate G1-G7 passed); the payload now reaches the wire only when the
# snapshot's value DIFFERS from BLOCK_SWAP_PREFETCH_DEFAULT (build_a2v_chain_payload's
# discipline), so it is the OFF case that now appends the key.
# --------------------------------------------------------------------------- #
def test_batch_prefetch_off_snapshot_adds_block_swap_prefetch_to_payload(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, block_swap_prefetch=False)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert p["block_swap_prefetch"] is False
    assert list(p.keys())[-1] == "block_swap_prefetch"


def test_batch_default_snapshot_omits_block_swap_prefetch(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert "block_swap_prefetch" not in p


# --------------------------------------------------------------------------- #
# keep-resident: same snapshot-is-the-only-path reasoning again. This is the
# setting a batch benefits from MOST (every row after the first is a cache
# HIT), so a dropped wire here is a large silent regression -- and, being
# default-off, it is the CHECKED case that appends the key.
# --------------------------------------------------------------------------- #
def test_batch_keep_resident_snapshot_adds_key_to_payload(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, keep_resident=True)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert p["keep_resident"] is True
    assert list(p.keys())[-1] == "keep_resident"


def test_batch_default_snapshot_omits_keep_resident(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert "keep_resident" not in p


# --------------------------------------------------------------------------- #
# Fused GGUF dequantization kernel (§1-11): same snapshot-is-the-only-path
# reasoning once more. Default ON since 2026-08-04 (§51), so it is the
# UNCHECKED case that appends the key -- and it is appended LAST, after
# keep_resident.
# --------------------------------------------------------------------------- #
def test_batch_fused_dequant_off_snapshot_adds_key_to_payload(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, fused_gguf_dequant_kernel=False)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert p["fused_gguf_dequant_kernel"] is False
    assert list(p.keys())[-1] == "fused_gguf_dequant_kernel"


def test_batch_default_snapshot_omits_fused_dequant(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert "fused_gguf_dequant_kernel" not in p


# --------------------------------------------------------------------------- #
# keep-resident embeddings (LTX 2.5): same snapshot-is-the-only-path reasoning
# once more. Default OFF, so it is the CHECKED case that appends the key -- and
# it is appended LAST, after vae_mode.
# --------------------------------------------------------------------------- #
def test_batch_keep_resident_embeddings_snapshot_adds_key_to_payload(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, keep_resident_embeddings=True)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert p["keep_resident_embeddings"] is True
    assert list(p.keys())[-1] == "keep_resident_embeddings"


def test_batch_default_snapshot_omits_keep_resident_embeddings(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert "keep_resident_embeddings" not in p


# --------------------------------------------------------------------------- #
# chunked_upsample: the batch accordion's own checkbox. Unlike every toggle
# above it is sent EXPLICITLY either way (the plugin's batch does the same) --
# omitting it would silently fall back to the slow one-pass upsample.
# --------------------------------------------------------------------------- #
def test_batch_default_snapshot_sends_chunked_upsample_true(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir)          # checkbox default
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert p["chunked_upsample"] is True
    keys = list(p.keys())
    assert keys[keys.index("source_audio") + 1] == "chunked_upsample"


def test_batch_unchecked_snapshot_sends_chunked_upsample_false(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, chunked_upsample=False)
    rows = _rows(("a.wav",))

    runner = BatchRunner()
    started, reason = _start(runner, snap, rows, api, sync=True)
    assert started is True, reason

    _jid, p = server.payloads[0]
    assert p["chunked_upsample"] is False      # present, not merely falsy


# --------------------------------------------------------------------------- #
# Per-row <lora:...> tags (§3-1). ONE rule: the row sends the COMPOSED prompt,
# and that composed string is what the parser sees -- so "add" is the union
# with the row's strength winning, "replace" is the row's tags alone, and a
# "replace" row with an empty prompt falls back to the card's. The reference-
# video CONTROL adapter is prepended for every row, tags or not.
# --------------------------------------------------------------------------- #
_KNOWN_LORAS = ("A", "B")


def _prepared_row(common, row_prompt, mode, **kwargs) -> BatchRow:
    row = BatchRow(queue=1, wav="a.wav", prompt=row_prompt, image="", frames=49)
    err = prepare_batch_rows([row], common, mode, _KNOWN_LORAS, **kwargs)
    assert err is None, err
    return row


def test_row_tags_add_mode_unions_card_and_row_tags():
    row = _prepared_row("card <lora:A:0.6>", "row <lora:B:0.8>", "add")
    assert row.loras == [{"name": "A", "strength": 0.6},
                         {"name": "B", "strength": 0.8}]
    assert row.send_prompt == "card row"          # both tags stripped
    assert row.prompt == "row <lora:B:0.8>"       # the user's cell is untouched


def test_row_tags_add_mode_row_strength_wins_for_the_same_name():
    row = _prepared_row("card <lora:A:0.6>", "row <lora:A:1.2>", "add")
    assert row.loras == [{"name": "A", "strength": 1.2}]


def test_row_tags_replace_mode_takes_only_the_row_tags():
    row = _prepared_row("card <lora:A:0.6>", "row <lora:B:0.8>", "replace")
    assert row.loras == [{"name": "B", "strength": 0.8}]
    assert row.send_prompt == "row"


def test_row_tags_replace_mode_empty_row_falls_back_to_the_card_tags():
    row = _prepared_row("card <lora:A:0.6>", "", "replace")
    assert row.loras == [{"name": "A", "strength": 0.6}]
    assert row.send_prompt == "card"


def test_row_without_any_tag_keeps_the_composed_prompt_and_no_loras():
    row = _prepared_row("card", "row", "add")
    assert row.send_prompt == "card row"
    assert row.loras == []


def test_row_with_a_reference_adapter_puts_the_control_lora_first():
    # A row carrying a reference_video_id and an EMPTY loras list is a 422, so
    # the control adapter has to be combined in for every row -- including the
    # ones with no tags of their own.
    row = _prepared_row("card <lora:A:0.6>", "row <lora:B:0.8>", "add",
                        use_adapter=True, adapter="canny-control",
                        adapter_strength=0.7)
    assert row.loras[0] == {"name": "canny-control", "strength": 0.7}
    assert [entry["name"] for entry in row.loras] == ["canny-control", "A", "B"]

    plain = _prepared_row("card", "row", "add", use_adapter=True,
                          adapter="canny-control", adapter_strength=1.0)
    assert plain.loras == [{"name": "canny-control", "strength": 1.0}]


def test_row_with_an_unknown_tag_aborts_the_whole_dispatch():
    rows = [BatchRow(queue=1, wav="a.wav", prompt="", image="", frames=49),
            BatchRow(queue=2, wav="b.wav", prompt="<lora:nope>", image="",
                     frames=49)]
    err = prepare_batch_rows(rows, "card", "add", _KNOWN_LORAS)
    assert err is not None and "nope" in err


def test_rows_have_lora_tokens_reads_the_composed_text():
    card_only = [BatchRow(queue=1, wav="a.wav", prompt="row", image="")]
    assert rows_have_lora_tokens(card_only, "card <lora:A>", "add") is True
    # "replace" with a non-empty row prompt never shows the card's tags.
    assert rows_have_lora_tokens(card_only, "card <lora:A>", "replace") is False
    row_only = [BatchRow(queue=1, wav="a.wav", prompt="<lora:B>", image="")]
    assert rows_have_lora_tokens(row_only, "card", "replace") is True
    assert rows_have_lora_tokens(card_only, "card", "add") is False


def test_a_finished_row_with_an_unknown_tag_does_not_block_the_dispatch():
    # Resume case: a Done row is never submitted, so a stale (or since-deleted)
    # LoRA name left in its cell must not abort the whole run -- and the row's
    # own freeze fields stay exactly as they were.
    rows = [BatchRow(queue=1, wav="a.wav", prompt="<lora:gone:1>", image="",
                     stat=STAT_DONE, frames=49),
            BatchRow(queue=2, wav="b.wav", prompt="row <lora:A:0.4>", image="",
                     stat=STAT_WAITING, frames=49)]
    assert prepare_batch_rows(rows, "card", "add", _KNOWN_LORAS) is None
    assert rows[0].send_prompt == ""
    assert rows[0].loras == []
    assert rows[1].send_prompt == "card row"
    assert rows[1].loras == [{"name": "A", "strength": 0.4}]


def test_rows_have_lora_tokens_ignores_finished_rows():
    # A token that only a Done / Skip row carries must not cost a GET /loras.
    done = [BatchRow(queue=1, wav="a.wav", prompt="<lora:A>", image="",
                     stat=STAT_DONE)]
    assert rows_have_lora_tokens(done, "card", "add") is False
    skipped = [BatchRow(queue=1, wav="a.wav", prompt="<lora:A>", image="",
                        stat=STAT_SKIP)]
    assert rows_have_lora_tokens(skipped, "card", "add") is False
    # ...while a Waiting row alongside it still answers True.
    waiting = BatchRow(queue=2, wav="b.wav", prompt="<lora:B>", image="",
                       stat=STAT_WAITING)
    assert rows_have_lora_tokens(done + [waiting], "card", "add") is True


def test_row_loras_reach_the_payload(tmp_path):
    # End to end through the runner: what prepare_batch_rows froze onto the row
    # is what the submitted body carries, per row.
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    _write_wav(wav_dir / "b.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, prompt_common="card <lora:A:0.6>")
    rows = _rows(("a.wav", STAT_WAITING, "", "row <lora:B:0.8>"),
                 ("b.wav", STAT_WAITING, "", "plain"))
    assert prepare_batch_rows(rows, snap.prompt_common, snap.prompt_mode,
                              _KNOWN_LORAS) is None

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason

    payloads = [p for _jid, p in server.payloads]
    assert payloads[0]["loras"] == [{"name": "A", "strength": 0.6},
                                    {"name": "B", "strength": 0.8}]
    assert payloads[0]["prompt"] == "card row"
    assert payloads[1]["loras"] == [{"name": "A", "strength": 0.6}]
    assert payloads[1]["prompt"] == "card plain"


def test_row_prompt_column_survives_a_run(tmp_path):
    # The composed/stripped text lives in send_prompt, NEVER in the CSV's own
    # prompt column -- a rescan after a run must not feed the composed text
    # back in and compose it a second time.
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    out_dir = tmp_path / "out"

    server = _Server()
    api = _make_client(server.handler)
    snap = _snapshot(wav_dir, out_dir, prompt_common="card")
    rows = _rows(("a.wav", STAT_WAITING, "", "row <lora:A>"))

    runner = BatchRunner()
    assert prepare_batch_rows(rows, snap.prompt_common, snap.prompt_mode,
                              _KNOWN_LORAS) is None
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason

    assert rows[0].prompt == "row <lora:A>"
    assert read_manifest(wav_dir)[0].prompt == "row <lora:A>"


def test_fused_dequant_reaches_the_wire_on_all_three_backend_paths(tmp_path):
    """§1-11 regression: single, chain AND batch all put the key on the wire.

    The three paths assemble their bodies in three DIFFERENT places
    (``make_generate_handler``'s own dict, ``make_chain_handler``'s own dict,
    and ``build_a2v_chain_payload`` reached through ``BatchSnapshot``), so a
    per-path unit test can pass while one path quietly drops the flag -- the
    batch one especially, since nothing else in the batch flow would notice.
    This test walks all three end to end against a mock transport and asserts
    the key is actually in the JSON body each time. Since the 2026-08-04 flip
    (§51) the server default is ON, so the case that PUTS the key on the wire
    is the unchecked one -- the assertion is on ``False``.
    """
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/upload/audio"):
            return httpx.Response(200, json={"audio_id": "aud-3paths"})
        if path.endswith("/generate") or path.endswith("/generate/chain"):
            bodies.append(json.loads(request.content))
            return httpx.Response(202, json={"job_id": f"job-{len(bodies)}"})
        if path.endswith("/video"):
            return httpx.Response(200, content=b"MP4DATA")
        if "/jobs/" in path:
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(404, json={"error": f"unexpected {path}"})

    api = _make_client(handler)

    # 1) single generate (T2V): the whole keyframe grid as ONE argument -- a
    # KF_MAX_SLOTS-long list of disabled (enabled, image, frame_idx, strength)
    # slots.
    gen = make_generate_handler(api)(
        "a calm river", "", [(False, None, 0, 0.8)] * KF_MAX_SLOTS,
        512, 320, False, 0, 0, 49, 24.0, -1,
        fused_gguf_dequant_kernel=False,
    )
    for _out in gen:
        if bodies:
            gen.close()
            break

    # 2) chain (2 clips): 11 leading positionals, then 24 clip slots (slot 1
    # carries image + strength), then the config state.
    chain_args = ["a calm river", "", 512, 320, False, 0, 0, 24.0, -1, 3, 0.5]
    chain_args.extend([True, "", 121, None, 0.8])          # clip 1
    chain_args.extend([True, "", 121])                     # clip 2
    for _i in range(22):                                   # clips 3..24 (off)
        chain_args.extend([False, "", 121])
    chain_args.append(None)                                # config state
    cgen = make_chain_handler(api)(*chain_args,
                                   fused_gguf_dequant_kernel=False)
    for out in cgen:
        if out[1]:
            cgen.close()
            break

    # 3) batch A2V (one row) through the runner + snapshot.
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "a.wav")
    snap = _snapshot(wav_dir, tmp_path / "out", fused_gguf_dequant_kernel=False)
    started, reason = _start(BatchRunner(), snap, _rows(("a.wav",)), api, sync=True)
    assert started is True, reason

    assert len(bodies) == 3, f"expected 3 submissions, got {len(bodies)}"
    for i, body in enumerate(bodies):
        assert body.get("fused_gguf_dequant_kernel") is False, (
            f"path #{i + 1} did not send fused_gguf_dequant_kernel: "
            f"{sorted(body)}"
        )
