"""End-to-end smoke tests through the REST API with the mock runner (spec 18)."""

from __future__ import annotations

import json
from pathlib import Path


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
