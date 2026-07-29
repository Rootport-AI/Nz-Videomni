"""Batch A2V — full-stack integration test (WP5).

The unit tests in ``tests/test_gradio_batch_runner.py`` drive
:class:`gradio_ui.batch.BatchRunner` against an ``httpx.MockTransport`` (a hand
-rolled fake server). That is enough to pin the runner's *own* logic, but it
never exercises the real FastAPI routing, pydantic validation, job store, or
on-disk metadata — none of which the mock server reproduces.

This module wires the SAME :class:`gradio_ui.api_client.ApiClient` the runner
uses in production to a real ``main.build_app(...)`` instance (``model.backend
= "mock"`` so no GPU/weights are needed) — the connection-injection point
``ApiClient.__init__`` already exposes (``client: httpx.Client | None``), the
same seam ``tests/test_gradio_batch_runner.py`` uses for its
``httpx.MockTransport``. No adapter class is needed: ``fastapi.testclient.
TestClient`` (the same one ``tests/test_a2v_chain.py`` builds its mock e2e
client from) IS an ``httpx.Client`` subclass with a synchronous ASGI-bridging
transport (plain ``httpx.Client(transport=httpx.ASGITransport(...))`` does NOT
work here — ``ASGITransport`` only implements the async ``handle_async_
request``; ``TestClient`` supplies the sync bridge). Passed straight into
``ApiClient(client=...)`` it talks to the actual ``/api/v1/*`` routes,
pydantic models, ``JobStore``, and upload stores exactly as it would over a
socket — only the network hop is elided.

Every scenario below therefore exercises: manifest scan/merge/write (pure
``gradio_ui.manifest``), the real upload endpoints, real ``POST
/generate/chain`` routing + validation (incl. the A2V length preflight), the
real (mock) worker's completion path, and the runner's file-copy + CSV-flush
side effects — end to end, without a GPU.
"""

from __future__ import annotations

import argparse
import struct
import time
import wave
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

import main
from gradio_ui import manifest as batch_manifest
from gradio_ui.api_client import ApiClient
from gradio_ui.batch import STATE_IDLE, BatchRunner, BatchSnapshot, compose_prompt
from gradio_ui.handlers import suggest_frames_for_audio
from gradio_ui.manifest import (
    IMAGE_SHARED,
    STAT_DONE,
    STAT_FAILED,
    STAT_WAITING,
    read_manifest,
)

FPS = 24.0
WIDTH = 384    # /64 == 6
HEIGHT = 256   # /64 == 4


# --------------------------------------------------------------------------- #
# Fixture helpers.
# --------------------------------------------------------------------------- #
def _build_app_client(base_dir: Path) -> ApiClient:
    """A real ``main.build_app`` (mock backend) wired to an ``ApiClient`` via a
    ``TestClient`` (the sync ASGI-bridging ``httpx.Client`` subclass) — this IS
    the frozen REST API, in-process, exactly as ``tests/test_a2v_chain.py``'s
    ``_build_client`` builds its mock e2e client (built without entering the
    context-manager: ``build_app`` already sets ``app.state.context``
    synchronously, so no lifespan startup is needed here)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "server": {"log_dir": (base_dir / "logs").as_posix()},
        "model": {"backend": "mock"},
        "output": {"dir": (base_dir / "outputs").as_posix()},
        "upload": {"dir": (base_dir / "uploads").as_posix()},
    }
    cfg_path = base_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    args = argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=cfg_path.as_posix(), te_offload=None, dit_cpu_load=None,
    )
    app = main.build_app(args)
    tc = TestClient(app)
    return ApiClient("http://testserver", client=tc)


def _make_wav(path: Path, *, seconds: float = 3.0, sr: int = 16000, channels: int = 1) -> Path:
    """Silent PCM wav fixture with an exact duration (mirrors
    tests/test_a2v_chain.py's ``_make_wav`` — pure stdlib, no ffmpeg needed)."""
    n = int(round(seconds * sr))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % (n * channels), *([0] * (n * channels))))
    return path


def _make_png(path: Path) -> Path:
    Image.new("RGB", (64, 48), (40, 120, 200)).save(path)
    return path


def _scan_merge_write(wav_dir: Path):
    """The exact scan -> read -> merge -> write sequence gradio_ui/ui.py's
    "Set audios" handler runs (see ui.py on_batch_set_audios)."""
    scanned = batch_manifest.scan_wav_folder(wav_dir, FPS, frames_for=suggest_frames_for_audio)
    existing = batch_manifest.read_manifest(wav_dir)
    rows, _warnings = batch_manifest.merge_rows(existing, scanned)
    res = batch_manifest.write_manifest_atomic(wav_dir, rows)
    assert res.ok, res
    return rows


def _snapshot(wav_dir: Path, out_dir: Path, **overrides) -> BatchSnapshot:
    kw = dict(
        wav_dir=str(wav_dir),
        out_dir=str(out_dir),
        prompt_common="base prompt",
        negative="",
        prompt_mode="add",
        width=WIDTH,
        height=HEIGHT,
        crop_output=None,
        frame_rate=FPS,
        seed=123,
        loras=[],
        shared_images=[],
        use_adapter=False,
        ref_video_path=None,
        control_adherence=1.0,
        reference_strength=1.0,
        poll_interval=0.02,
        poll_timeout_s=20.0,
    )
    kw.update(overrides)
    return BatchSnapshot(**kw)


def _by_wav(rows) -> dict:
    return {r.wav: r for r in rows}


# --------------------------------------------------------------------------- #
# 1) Happy path: 3 wavs (one Japanese filename) + a shared keyframe -> all Done.
# --------------------------------------------------------------------------- #
def test_happy_path_all_rows_done_fullstack(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    names = ["a.wav", "b.wav", "こんにちは.wav"]
    for name in names:
        _make_wav(wav_dir / name, seconds=3.0)
        time.sleep(0.01)  # distinct mtimes -> deterministic scan order

    img = _make_png(tmp_path / "kf.png")
    out_dir = batch_manifest.resolve_output_dir(wav_dir, "custom", tmp_path / "out")

    rows = _scan_merge_write(wav_dir)
    assert len(rows) == 3
    assert all(r.stat == STAT_WAITING for r in rows)      # nothing Skip-flagged
    assert all(r.image == IMAGE_SHARED for r in rows)      # scan default

    api = _build_app_client(tmp_path / "srv")
    snap = _snapshot(wav_dir, out_dir, shared_images=[(str(img), 0, 0.8)])

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason
    assert runner.state == STATE_IDLE

    by_wav = _by_wav(rows)
    for name in names:
        row = by_wav[name]
        assert row.stat == STAT_DONE, (name, row.error)
        expected_out = Path(name).stem + ".mp4"
        assert row.output == expected_out
        out_path = out_dir / expected_out
        assert out_path.exists() and out_path.stat().st_size > 0

    # CSV round-trips the final state.
    disk = _by_wav(read_manifest(wav_dir))
    for name in names:
        assert disk[name].stat == STAT_DONE
        assert disk[name].output == Path(name).stem + ".mp4"


# --------------------------------------------------------------------------- #
# 1b) M-1: a per-row image in a SEPARATE image folder (img_dir != wav_dir)
#     resolves + uploads against img_dir and the row completes end-to-end
#     (the pre-fix code joined it to wav_dir -> FileNotFound -> Failed row).
# --------------------------------------------------------------------------- #
def test_individual_image_from_separate_folder_fullstack(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _make_wav(wav_dir / "a.wav", seconds=3.0)

    img_dir = tmp_path / "imgs"      # deliberately not the audio folder
    img_dir.mkdir()
    _make_png(img_dir / "face.png")
    out_dir = batch_manifest.resolve_output_dir(wav_dir, "custom", tmp_path / "out")

    rows = _scan_merge_write(wav_dir)
    assert len(rows) == 1
    rows[0].image = "face.png"       # per-row image: a bare name living in img_dir

    api = _build_app_client(tmp_path / "srv")
    snap = _snapshot(wav_dir, out_dir, img_dir=str(img_dir))

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason

    assert rows[0].stat == STAT_DONE, rows[0].error
    assert rows[0].output == "a.mp4"
    assert (out_dir / "a.mp4").exists()


# --------------------------------------------------------------------------- #
# 2) + 3) One row is sabotaged to fail on the server side -> that row Failed,
#    the batch continues; a later resume (CSV read back) skips the Done rows
#    untouched and retries ONLY the Failed row (which now succeeds once its
#    sabotage is undone), producing a fresh output for it alone.
#
#    NOTE (2nd-round FB, modification C): the failure vector here can NO LONGER
#    be an inflated ``frames`` value — start() now re-judges every unfinished
#    row's frame count from the snapshot's frame rate, so a hand-set 481 would
#    just be recomputed back to a fitting value before the run. We instead point
#    the row at a per-row image that does not exist: the image upload raises
#    (FileNotFound) -> the row Failed, a vector the frame re-judgment never
#    touches (so the resume, which restores the image, retries it cleanly).
# --------------------------------------------------------------------------- #
def test_middle_row_image_missing_then_resume_retries_only_failed(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    for name in ("a.wav", "b.wav", "c.wav"):
        _make_wav(wav_dir / name, seconds=3.0)
        time.sleep(0.01)

    out_dir = batch_manifest.resolve_output_dir(wav_dir, "custom", tmp_path / "out")
    img = _make_png(tmp_path / "kf.png")
    rows = _scan_merge_write(wav_dir)

    by_wav = _by_wav(rows)
    # Sabotage row "b": point it at a per-row image filename that exists nowhere
    # (img_dir unset -> resolves against wav_dir, which has no such file) so the
    # runner's image upload raises FileNotFound -> the row is marked Failed.
    by_wav["b.wav"].image = "ghost.png"

    api = _build_app_client(tmp_path / "srv")
    snap = _snapshot(wav_dir, out_dir, shared_images=[(str(img), 0, 0.8)])

    runner1 = BatchRunner()
    started, reason = runner1.start(snap, rows, api, sync=True)
    assert started is True, reason

    assert by_wav["a.wav"].stat == STAT_DONE
    assert by_wav["c.wav"].stat == STAT_DONE
    assert by_wav["b.wav"].stat == STAT_FAILED
    assert by_wav["b.wav"].error   # a non-empty error was recorded

    assert (out_dir / "a.mp4").exists()
    assert (out_dir / "c.mp4").exists()
    assert not (out_dir / "b.mp4").exists()

    mtime_a = (out_dir / "a.mp4").stat().st_mtime
    mtime_c = (out_dir / "c.mp4").stat().st_mtime

    # --- resume: read the manifest back, fix the sabotage, re-run. ---
    persisted = _by_wav(read_manifest(wav_dir))
    assert persisted["a.wav"].stat == STAT_DONE
    assert persisted["b.wav"].stat == STAT_FAILED
    assert persisted["c.wav"].stat == STAT_DONE
    persisted["b.wav"].image = IMAGE_SHARED  # undo the sabotage for the retry

    runner2 = BatchRunner()
    started2, reason2 = runner2.start(snap, list(persisted.values()), api, sync=True)
    assert started2 is True, reason2

    assert persisted["a.wav"].stat == STAT_DONE
    assert persisted["b.wav"].stat == STAT_DONE
    assert persisted["c.wav"].stat == STAT_DONE
    assert persisted["b.wav"].output == "b.mp4"
    assert (out_dir / "b.mp4").exists()

    # the two already-Done rows were never re-touched.
    assert (out_dir / "a.mp4").stat().st_mtime == mtime_a
    assert (out_dir / "c.mp4").stat().st_mtime == mtime_c
    assert persisted["a.wav"].output == "a.mp4"
    assert persisted["c.wav"].output == "c.mp4"


# --------------------------------------------------------------------------- #
# 4) Gacha numbering: re-running an already-Done row (stat manually rewound to
#    Waiting) must not clobber the existing output — it earns a "_2" suffix.
# --------------------------------------------------------------------------- #
def test_gacha_numbering_on_rerun_of_done_row(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _make_wav(wav_dir / "solo.wav", seconds=3.0)
    out_dir = batch_manifest.resolve_output_dir(wav_dir, "custom", tmp_path / "out")
    img = _make_png(tmp_path / "kf.png")

    rows = _scan_merge_write(wav_dir)
    assert len(rows) == 1

    api = _build_app_client(tmp_path / "srv")
    snap = _snapshot(wav_dir, out_dir, shared_images=[(str(img), 0, 0.8)])

    runner1 = BatchRunner()
    started, reason = runner1.start(snap, rows, api, sync=True)
    assert started is True, reason
    assert rows[0].stat == STAT_DONE
    assert rows[0].output == "solo.mp4"

    original_path = out_dir / "solo.mp4"
    original_bytes = original_path.read_bytes()
    original_mtime = original_path.stat().st_mtime

    # User re-queues the same (already Done) row for another gacha pull.
    rows[0].stat = STAT_WAITING

    runner2 = BatchRunner()
    started2, reason2 = runner2.start(snap, rows, api, sync=True)
    assert started2 is True, reason2

    assert rows[0].stat == STAT_DONE
    assert rows[0].output == "solo_2.mp4"
    assert (out_dir / "solo_2.mp4").exists()

    # the original output is byte-for-byte untouched.
    assert original_path.exists()
    assert original_path.read_bytes() == original_bytes
    assert original_path.stat().st_mtime == original_mtime

    disk = read_manifest(wav_dir)
    assert disk[0].output == "solo_2.mp4"
    assert disk[0].stat == STAT_DONE


# --------------------------------------------------------------------------- #
# 5) Prompt composition ("add" / "replace") is what the SERVER actually
#    received for the job (GET /jobs -> JobResponse.request.prompt), not just
#    what compose_prompt() returns in isolation.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["add", "replace"])
def test_prompt_composition_reaches_server_job_request(tmp_path, mode):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _make_wav(wav_dir / "voice.wav", seconds=3.0)
    out_dir = batch_manifest.resolve_output_dir(wav_dir, "custom", tmp_path / "out")
    img = _make_png(tmp_path / "kf.png")

    rows = _scan_merge_write(wav_dir)
    rows[0].prompt = "sunset colors"
    common = "misty forest"
    expected = compose_prompt(common, rows[0].prompt, mode)

    api = _build_app_client(tmp_path / "srv")
    snap = _snapshot(wav_dir, out_dir, prompt_common=common, prompt_mode=mode,
                      shared_images=[(str(img), 0, 0.8)])

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason
    assert rows[0].stat == STAT_DONE

    jobs = api.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["request"]["prompt"] == expected


# --------------------------------------------------------------------------- #
# 6) NAG (non-CFG Negative): BatchSnapshot's nag_* fields reach the REAL
#    server's job store (GET /jobs -> JobResponse.request.nag_enabled/...),
#    exercising api/models.py's GenerateChainRequest validation + job_store's
#    to_clip_request round-trip end-to-end -- not just the payload dict this
#    module's unit-level counterpart (test_gradio_batch_runner.py) checks.
# --------------------------------------------------------------------------- #
def test_batch_nag_enabled_reaches_server_job_request(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _make_wav(wav_dir / "voice.wav", seconds=3.0)
    out_dir = batch_manifest.resolve_output_dir(wav_dir, "custom", tmp_path / "out")
    img = _make_png(tmp_path / "kf.png")

    rows = _scan_merge_write(wav_dir)

    api = _build_app_client(tmp_path / "srv")
    snap = _snapshot(wav_dir, out_dir, negative="blurry, low quality",
                     shared_images=[(str(img), 0, 0.8)],
                     nag_enabled=True, nag_scale=9.0, nag_tau=3.0, nag_alpha=0.4)

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason
    assert rows[0].stat == STAT_DONE

    jobs = api.list_jobs()
    assert len(jobs) == 1
    req = jobs[0]["request"]
    assert req["nag_enabled"] is True
    assert req["nag_scale"] == 9.0
    assert req["nag_tau"] == 3.0
    assert req["nag_alpha"] == 0.4


# --------------------------------------------------------------------------- #
# 7) VSF (Value Sign Flip): BatchSnapshot's neg_method/vsf_scale
#    fields reach the REAL server's job store the same way the nag_* fields
#    do above -- exercising api/models.py's neg_method/vsf_scale
#    validation end-to-end.
# --------------------------------------------------------------------------- #
def test_batch_vsf_enabled_reaches_server_job_request(tmp_path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _make_wav(wav_dir / "voice.wav", seconds=3.0)
    out_dir = batch_manifest.resolve_output_dir(wav_dir, "custom", tmp_path / "out")
    img = _make_png(tmp_path / "kf.png")

    rows = _scan_merge_write(wav_dir)

    api = _build_app_client(tmp_path / "srv")
    snap = _snapshot(wav_dir, out_dir, negative="blurry, low quality",
                     shared_images=[(str(img), 0, 0.8)],
                     nag_enabled=True, neg_method="vsf", vsf_scale=2.0)

    runner = BatchRunner()
    started, reason = runner.start(snap, rows, api, sync=True)
    assert started is True, reason
    assert rows[0].stat == STAT_DONE

    jobs = api.list_jobs()
    assert len(jobs) == 1
    req = jobs[0]["request"]
    assert req["neg_method"] == "vsf"
    assert req["vsf_scale"] == 2.0
