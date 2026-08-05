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


def test_status_reports_acceleration(client):
    # Acceleration capability block (ADDITIVE, top level). The mock backend has
    # no engine at all, so sage_available is False regardless of what the real
    # engine venv on this machine happens to contain -- that mock rule is what
    # keeps this assertion machine-independent. block_swap_prefetch_available
    # (backend §44) follows the same mock rule.
    r = client.get("/api/v1/status")
    assert r.status_code == 200
    accel = r.json()["acceleration"]
    assert accel["attention_backends"] == ["sdpa", "sage"]
    assert accel["sage_available"] is False
    assert accel["block_swap_prefetch_available"] is False


def test_status_vram_optimization_key_set_is_unchanged(client):
    # FROZEN contract (spec 7.4): the acceleration feature must NOT smuggle any
    # key into vram_optimization -- it lives in its own top-level block.
    r = client.get("/api/v1/status")
    assert set(r.json()["vram_optimization"]) == {
        "low_vram_mode",
        "low_vram_profile",
        "fp8_transformer",
        "cpu_offload_text_encoder",
        "vae_tiling",
        "attention_tiling",
        "block_swap",
        "low_vram_disabled_required",
    }


def test_acceleration_status_truth_table(client):
    # PipelineManager.acceleration_status_block is the single place the
    # sage-availability truth table lives; exercise all four rows of it.
    import types

    pm = client.app_context.pipeline_manager
    runner = pm.runner
    runner._sage_probe_cache = True  # pretend the engine venv HAS sageattention

    # (1) mock backend -> False, and the file probe cannot override it.
    assert pm.acceleration_status_block()["sage_available"] is False

    # From here on the backend is no longer the mock.
    runner.config.model.backend = "real"

    # (2) not loaded -> the file probe answers.
    assert pm.acceleration_status_block()["sage_available"] is True
    runner._sage_probe_cache = False
    assert pm.acceleration_status_block()["sage_available"] is False

    # (3) loaded -> the WORKER's own probe wins over the file probe.
    runner._backend = types.SimpleNamespace(loaded=True, sage_available=True)
    assert pm.acceleration_status_block()["sage_available"] is True

    # (4) load failed / unloaded (no live worker) -> back to the file probe.
    runner._backend = types.SimpleNamespace(loaded=False, sage_available=True)
    assert pm.acceleration_status_block()["sage_available"] is False


def test_block_swap_prefetch_available_truth_table(client):
    # PipelineManager._block_swap_prefetch_available (backend §44) mirrors the
    # REAL gate (services/ltx_runner.py's ``block_swap_blocks_on_gpu or 8``
    # expression), NOT the display-only low_vram.block_swap bool -- that bool
    # is never read on the real path and would make this field lie by default.
    pm = client.app_context.pipeline_manager
    runner = pm.runner

    # (1) mock backend -> False regardless of low_vram config.
    pm.low_vram.block_swap_blocks_on_gpu = 8
    assert pm.acceleration_status_block()["block_swap_prefetch_available"] is False

    # From here on the backend is no longer the mock.
    runner.config.model.backend = "real"

    # (2) real + unset (None) -> falls back to 8 (the `or 8` expression) -> True.
    pm.low_vram.block_swap_blocks_on_gpu = None
    assert pm.acceleration_status_block()["block_swap_prefetch_available"] is True

    # (3) real + explicit positive value -> True.
    pm.low_vram.block_swap_blocks_on_gpu = 8
    assert pm.acceleration_status_block()["block_swap_prefetch_available"] is True

    # (4) real + 0 -> the ``or 8`` expression treats 0 as falsy too (same quirk
    # as the real worker-payload formula at services/ltx_runner.py's
    # ``block_swap_blocks_on_gpu or 8``), so it ALSO falls back to 8 -> True.
    # This mirrors the real gate exactly rather than a "nicer" 0-means-off
    # reading, by design (§8.5): the two must never disagree.
    pm.low_vram.block_swap_blocks_on_gpu = 0
    assert pm.acceleration_status_block()["block_swap_prefetch_available"] is True

    # (5) real + a negative value is the only way this expression yields False
    # on a real backend (0/None both fall back to 8 above).
    pm.low_vram.block_swap_blocks_on_gpu = -1
    assert pm.acceleration_status_block()["block_swap_prefetch_available"] is False


def test_sage_file_probe_checks_engine_venv_site_packages(tmp_path):
    # The pre-load probe is a pure file-existence check on the engine venv's
    # site-packages (Windows layout), and BOTH sageattention/ and triton/ are
    # required -- sage kernels are Triton-backed.
    from config import AppConfig
    from services.low_vram import build_low_vram_settings
    from services.ltx_runner import LTXRunner

    venv = tmp_path / ".venv-engine"
    site = venv / "Lib" / "site-packages"
    site.mkdir(parents=True)
    (venv / "Scripts").mkdir()

    def _runner() -> LTXRunner:
        cfg = AppConfig()
        cfg.model.engine_python = (venv / "Scripts" / "python.exe").as_posix()
        return LTXRunner(cfg, build_low_vram_settings(cfg))

    assert _runner().sage_available is False  # neither package present
    (site / "sageattention").mkdir()
    assert _runner().sage_available is False  # triton missing -> still False
    (site / "triton").mkdir()
    assert _runner().sage_available is True

    # Cached after the first read: removing the dirs does not flip it back.
    r = _runner()
    assert r.sage_available is True
    (site / "triton").rmdir()
    assert r.sage_available is True


def test_metadata_records_attention_used(client):
    # The engine's ACTUAL attention backend is recorded next to seed_used, so a
    # "requested sage but sdpa ran" case is visible in the job record. The mock
    # backend runs no attention at all -> null, but the KEY must be present (an
    # absent key would let the real path regress unnoticed).
    payload = {
        "prompt": "A red ball rolling on a white floor",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "seed": 42,
        "pipeline": "distilled",
        "attention_backend": "sage",
        "vae_mode": "prune_vaed",
    }
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert "attention_used" in meta
    assert meta["attention_used"] is None  # mock backend ran no engine attention
    # vae_mode_used rides the same route (PrunaVAED, §3-50): null on the mock
    # backend, which decodes nothing, but the KEY must be present so a real-path
    # regression cannot hide.
    assert "vae_mode_used" in meta
    assert meta["vae_mode_used"] is None
    # The request dump carries both request fields verbatim -- same precedent as
    # the two_stage_hq pipeline value; the *_used keys above are what say which
    # of them actually took effect.
    assert meta["request"]["attention_backend"] == "sage"
    assert meta["request"]["vae_mode"] == "prune_vaed"


def test_metadata_records_block_swap_prefetch_used(client):
    # Mirrors test_metadata_records_attention_used (backend §44): the effective
    # value (from the worker's done event, via outcome.block_swap_prefetch_used)
    # is recorded, not the raw request value -- same "actually applied" contract
    # as attention_used. The mock backend never installs block swap -> null, but
    # the KEY must be present unconditionally.
    payload = {
        "prompt": "A red ball rolling on a white floor",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "seed": 42,
        "pipeline": "distilled",
        "block_swap_prefetch": True,
    }
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert "block_swap_prefetch_used" in meta
    assert meta["block_swap_prefetch_used"] is None  # mock backend ran no engine
    assert "peak_vram_reserved_mb" in meta
    assert meta["peak_vram_reserved_mb"] is None
    # The request dump carries the raw requested value regardless.
    assert meta["request"]["block_swap_prefetch"] is True


def test_chain_metadata_records_attention_used(client):
    # Same key on the chain writer (a separate metadata builder -- it does not
    # share _write_metadata, so it needs its own guard).
    payload = {
        "prompt": "a serene mountain lake at dawn",
        "width": 384,
        "height": 256,
        "frame_rate": 24.0,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "pipeline": "distilled",
        "overlap_frames": 2,
        "overlap_strength": 0.5,
        "clips": [{"num_frames": 25}, {"num_frames": 25}],
        "attention_backend": "sage",
    }
    r = client.post("/api/v1/generate/chain", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert "attention_used" in meta
    assert meta["attention_used"] is None
    assert meta["request"]["attention_backend"] == "sage"


def test_chain_metadata_records_block_swap_prefetch_used(client):
    # Same key on the chain writer (a separate metadata builder -- it does not
    # share _write_metadata, so it needs its own guard).
    payload = {
        "prompt": "a serene mountain lake at dawn",
        "width": 384,
        "height": 256,
        "frame_rate": 24.0,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "pipeline": "distilled",
        "overlap_frames": 2,
        "overlap_strength": 0.5,
        "clips": [{"num_frames": 25}, {"num_frames": 25}],
        "block_swap_prefetch": True,
    }
    r = client.post("/api/v1/generate/chain", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert "block_swap_prefetch_used" in meta
    assert meta["block_swap_prefetch_used"] is None
    assert "peak_vram_reserved_mb" in meta
    assert meta["peak_vram_reserved_mb"] is None
    assert meta["request"]["block_swap_prefetch"] is True


def test_metadata_records_fused_gguf_dequant_kernel_used(client):
    # Mirrors test_metadata_records_block_swap_prefetch_used (§1-11): the
    # EFFECTIVE value (from the worker's done event) is recorded, not the raw
    # request value. The mock backend runs no engine -> null, but the KEY must be
    # present unconditionally on BOTH metadata writers (single + chain), since an
    # absent key would let the real path regress unnoticed.
    payload = {
        "prompt": "A red ball rolling on a white floor",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "seed": 42,
        "pipeline": "distilled",
        "fused_gguf_dequant_kernel": True,
    }
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert "fused_gguf_dequant_kernel_used" in meta
    assert meta["fused_gguf_dequant_kernel_used"] is None  # mock backend
    # The request dump carries the raw requested value regardless.
    assert meta["request"]["fused_gguf_dequant_kernel"] is True


def test_chain_metadata_records_fused_gguf_dequant_kernel_used(client):
    # Same key on the chain writer (a separate metadata builder -- it does not
    # share _write_metadata, so it needs its own guard).
    payload = {
        "prompt": "a serene mountain lake at dawn",
        "width": 384,
        "height": 256,
        "frame_rate": 24.0,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "pipeline": "distilled",
        "overlap_frames": 2,
        "overlap_strength": 0.5,
        "clips": [{"num_frames": 25}, {"num_frames": 25}],
        "fused_gguf_dequant_kernel": True,
    }
    r = client.post("/api/v1/generate/chain", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert "fused_gguf_dequant_kernel_used" in meta
    assert meta["fused_gguf_dequant_kernel_used"] is None
    assert meta["request"]["fused_gguf_dequant_kernel"] is True


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


def test_t2v_smoke_generation_with_nag(client):
    # NAG-enabled single /generate completes under the mock backend, and the
    # stored job meta request reflects nag_enabled (pipeline_manager model_dump
    # carries it through to metadata.json).
    payload = {
        "prompt": "A red ball rolling on a white floor",
        "negative_prompt": "blurry, low quality, distorted",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "frame_rate": 24.0,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "seed": 42,
        "pipeline": "distilled",
        "nag_enabled": True,
        "nag_scale": 11.0,
        "nag_tau": 2.5,
        "nag_alpha": 0.25,
    }
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    assert meta["request"]["nag_enabled"] is True
    assert meta["request"]["nag_scale"] == 11.0


def test_generate_with_nag_enabled_and_empty_negative_422(client):
    payload = {
        "prompt": "x",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "pipeline": "distilled",
        "nag_enabled": True,
        "negative_prompt": "",
    }
    r = client.post("/api/v1/generate", json=payload)
    assert r.status_code == 422


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
