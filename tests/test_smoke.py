"""End-to-end smoke tests through the REST API with the mock runner (spec 18)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_status_reports_low_vram(client):
    r = client.get("/api/v1/status")
    assert r.status_code == 200
    s = r.json()
    assert s["server"] == "running"
    assert isinstance(s["pipeline_loaded"], bool)
    v = s["vram_optimization"]
    assert v["low_vram_mode"] is True
    assert v["low_vram_profile"] == "16gb_safe"
    assert "available" in s["gpu"]


def test_upload_image(client, png_bytes):
    r = client.post("/api/v1/upload/image", files={"file": ("first.png", png_bytes, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["image_id"]
    assert body["stored_path"].endswith("/input.png")
    ctx = client.app_context
    assert (ctx.config.upload_dir / body["image_id"] / "input.png").exists()


def test_upload_invalid_type(client):
    r = client.post("/api/v1/upload/image", files={"file": ("bad.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "UPLOAD_INVALID_TYPE"


def test_t2v_smoke_generation(client):
    payload = {
        "prompt": "A red ball rolling on a white floor",
        "negative_prompt": "blurry, low quality",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "frame_rate": 24.0,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "seed": 42,
        "pipeline": "distilled",
        "conditioning_images": [],
    }
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    # TestClient runs the background task synchronously -> already terminal.
    jr = client.get(f"/api/v1/jobs/{job_id}")
    assert jr.status_code == 200
    job = jr.json()
    assert job["status"] == "completed", job
    assert job["result"]["seed_used"] == 42

    vid = client.get(f"/api/v1/jobs/{job_id}/video")
    assert vid.status_code == 200
    assert vid.headers["content-type"] == "video/mp4"
    assert len(vid.content) > 0

    ctx = client.app_context
    meta_path: Path = ctx.config.output_dir / job_id / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["generation_mode"] == "t2v"
    assert meta["seed_used"] == 42
    assert meta["vram_optimization"]["low_vram_mode"] is True
    assert "peak_vram_mb" in meta["vram_optimization"]


def test_i2v_smoke_generation(client, png_bytes):
    up = client.post("/api/v1/upload/image", files={"file": ("first.png", png_bytes, "image/png")})
    image_id = up.json()["image_id"]

    payload = {
        "prompt": "The scene slowly comes alive, subtle camera movement",
        "negative_prompt": "blurry",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "frame_rate": 24.0,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "seed": 7,
        "pipeline": "distilled",
        "conditioning_images": [{"image_id": image_id, "frame_idx": 0, "strength": 0.8}],
    }
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    assert meta["generation_mode"] == "i2v"
    assert meta["request"]["conditioning_images"][0]["image_id"] == image_id
    assert meta["request"]["conditioning_images"][0]["frame_idx"] == 0


def test_generate_with_unknown_image_404(client):
    payload = {
        "prompt": "x",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "pipeline": "distilled",
        "conditioning_images": [{"image_id": "does-not-exist", "frame_idx": 0, "strength": 0.8}],
    }
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "IMAGE_NOT_FOUND"


def test_job_busy_returns_409(client):
    # Seed an active (queued) job directly, then a new generate must 409.
    from api.models import GenerateRequest

    ctx = client.app_context
    active = GenerateRequest(prompt="busy", width=384, height=256, num_frames=17)
    ctx.job_store.create(active)  # status defaults to queued -> active

    payload = {
        "prompt": "second",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "pipeline": "distilled",
    }
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "JOB_BUSY"


def test_delete_queued_job_cancels_in_place(client):
    # B-2: a queued (not yet started) job is cancelled IN PLACE by DELETE so the
    # single-job guard frees up immediately -- no server restart required.
    from api.models import GenerateRequest

    ctx = client.app_context
    rec = ctx.job_store.create(
        GenerateRequest(prompt="stuck", width=384, height=256, num_frames=17)
    )
    assert ctx.job_store.has_active()

    r = client.delete(f"/api/v1/jobs/{rec.job_id}")
    assert r.status_code == 200
    assert r.json() == {"job_id": rec.job_id, "cancelled": True, "status": "cancelled"}
    assert rec.completed_at is not None
    assert not ctx.job_store.has_active()  # guard released

    # A fresh generate is now accepted (202), not 409 JOB_BUSY.
    payload = {
        "prompt": "after",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "pipeline": "distilled",
    }
    r2 = client.post("/api/v1/generate", json=payload)
    assert r2.status_code == 202


def test_delete_running_job_stays_best_effort(client):
    # B-2: the running-job DELETE behaviour is UNCHANGED -- inference cannot be
    # interrupted mid-flight, so it only sets cancel_requested (record kept,
    # status still "running", no completed_at).
    from api.models import GenerateRequest, JobStatus

    ctx = client.app_context
    rec = ctx.job_store.create(
        GenerateRequest(prompt="running", width=384, height=256, num_frames=17)
    )
    rec.status = JobStatus.running

    r = client.delete(f"/api/v1/jobs/{rec.job_id}")
    assert r.status_code == 200
    assert r.json() == {
        "job_id": rec.job_id, "cancel_requested": True, "status": "running",
    }
    assert rec.cancel_requested is True
    assert rec.completed_at is None
    assert ctx.job_store.get(rec.job_id) is not None  # not dropped


def test_run_job_honours_cancel_before_dispatch(client):
    # B-2 race guard: DELETE can flip cancel_requested after the job is queued but
    # before the worker picks it up. run_job must bail straight to cancelled
    # without ever entering the running path.
    from api.models import GenerateRequest, JobStatus

    ctx = client.app_context
    rec = ctx.job_store.create(
        GenerateRequest(prompt="cancel-before-start", width=384, height=256, num_frames=17)
    )
    rec.cancel_requested = True

    ctx.pipeline_manager.run_job(rec)

    assert rec.status == JobStatus.cancelled
    assert rec.started_at is None       # never entered the running path
    assert rec.completed_at is not None


def test_job_store_cancel_and_start_cas_are_mutually_exclusive():
    # S1: DELETE's queued->cancelled and the worker's queued->running promotion
    # both go through lock-guarded compare-and-set helpers -- exactly one wins.
    from api.models import GenerateRequest, JobStatus
    from services.job_store import JobStore

    store = JobStore()

    # cancel wins: the worker's start CAS must fail and must NOT resurrect the
    # cancelled record into a running one.
    rec = store.create(GenerateRequest(prompt="a", width=384, height=256, num_frames=17))
    assert store.cancel_if_queued(rec) is True
    assert rec.status == JobStatus.cancelled
    assert rec.completed_at is not None
    assert store.start_job(rec) is False
    assert rec.status == JobStatus.cancelled  # never flipped back to running
    assert rec.started_at is None

    # start wins: DELETE's in-place cancel must fail (it falls back to the
    # best-effort cancel_requested flag) and leave the running record untouched.
    rec2 = store.create(GenerateRequest(prompt="b", width=384, height=256, num_frames=17))
    assert store.start_job(rec2) is True
    assert rec2.status == JobStatus.running
    assert rec2.started_at is not None
    assert store.cancel_if_queued(rec2) is False
    assert rec2.status == JobStatus.running
    assert rec2.cancel_requested is False  # helper does not touch a running job

    # a cancel_requested flag raised while still queued blocks the start CAS
    # (the worker then finalizes via cancel_if_queued).
    rec3 = store.create(GenerateRequest(prompt="c", width=384, height=256, num_frames=17))
    rec3.cancel_requested = True
    assert store.start_job(rec3) is False
    assert rec3.status == JobStatus.queued  # start CAS leaves the record untouched


@pytest.fixture()
def thread_dispatch_client(tmp_path):
    """A TestClient whose backend is NOT "mock" ("auto" still resolves to the
    mock at run time in this weightless tmp dir), so POST /generate takes the
    dedicated-daemon-thread dispatch branch instead of BackgroundTasks.
    raise_server_exceptions=False lets a worker-spawn failure surface as a 500
    response instead of re-raising into the test."""
    import argparse

    import yaml
    from fastapi.testclient import TestClient

    import main

    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {"backend": "auto"},
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    args = argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=cfg_path.as_posix(), te_offload=None, dit_cpu_load=None,
    )
    app = main.build_app(args)
    with TestClient(app, raise_server_exceptions=False) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        yield c


def test_thread_start_failure_fails_job_and_frees_guard(thread_dispatch_client, monkeypatch):
    # S2: if the real-backend worker thread cannot be started, the job must not
    # be orphaned in queued (is_active would 409-block every later /generate
    # until a restart): it is failed in place and the guard is released.
    import types

    import api.generate as generate_mod
    from api.models import JobStatus

    class _BoomThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("cannot start new thread")

    # Swap the module's `threading` reference only (never the global module).
    monkeypatch.setattr(generate_mod, "threading",
                        types.SimpleNamespace(Thread=_BoomThread))

    client = thread_dispatch_client
    ctx = client.app_context
    payload = {
        "prompt": "spawn-fails",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "pipeline": "distilled",
    }
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 500

    jobs = ctx.job_store.list()
    assert len(jobs) == 1
    rec = jobs[0]
    assert rec.status == JobStatus.failed
    assert rec.completed_at is not None
    assert not ctx.job_store.has_active()  # guard released, no restart needed

    # With a working Thread again, a fresh generate is accepted (202), not 409.
    monkeypatch.undo()
    r2 = client.post("/api/v1/generate", json=payload)
    assert r2.status_code == 202
