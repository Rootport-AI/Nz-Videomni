"""GenerateRequest validation (spec 18.2)."""

from __future__ import annotations


BASE = {
    "prompt": "test",
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "pipeline": "distilled",
}


def test_width_not_multiple_of_64(client):
    # 544 is a multiple of 32 but NOT 64 -> rejected (two-stage distilled needs ÷64).
    r = client.post("/api/v1/generate", json={**BASE, "width": 544, "height": 512, "num_frames": 121})
    assert r.status_code == 422
    assert "multiple of 64" in r.text


def test_height_not_multiple_of_64(client):
    # 288 (the old phase1_default height) is ÷32 but not ÷64 -> rejected.
    r = client.post("/api/v1/generate", json={**BASE, "width": 512, "height": 288, "num_frames": 49})
    assert r.status_code == 422
    assert "multiple of 64" in r.text


def test_num_frames_not_8n_plus_1(client):
    r = client.post("/api/v1/generate", json={**BASE, "width": 960, "height": 576, "num_frames": 120})
    assert r.status_code == 422
    assert "8n+1" in r.text


def test_num_frames_at_cap_accepted(client):
    # 481 = 8*60+1 = 20s@24fps: new cap, 8n+1 -> accepted (202, async job).
    r = client.post("/api/v1/generate", json={**BASE, "width": 512, "height": 320, "num_frames": 481})
    assert r.status_code == 202


def test_num_frames_above_cap_rejected(client):
    # 489 = 8*61+1: valid 8n+1 but > 481 cap -> rejected by the le= bound.
    r = client.post("/api/v1/generate", json={**BASE, "width": 512, "height": 320, "num_frames": 489})
    assert r.status_code == 422


def test_num_frames_within_cap_but_not_8n_plus_1_rejected(client):
    # 480 <= 481 cap but not 8n+1 -> still rejected.
    r = client.post("/api/v1/generate", json={**BASE, "width": 512, "height": 320, "num_frames": 480})
    assert r.status_code == 422
    assert "8n+1" in r.text


def test_too_many_conditioning_images(client):
    r = client.post(
        "/api/v1/generate",
        json={
            **BASE,
            "width": 512,
            "height": 320,
            "num_frames": 49,
            "conditioning_images": [
                {"image_id": "image-a", "frame_idx": 0, "strength": 0.8},
                {"image_id": "image-b", "frame_idx": 8, "strength": 0.8},
            ],
        },
    )
    assert r.status_code == 422
    assert "at most one conditioning image" in r.text


def test_frame_idx_must_be_zero(client):
    r = client.post(
        "/api/v1/generate",
        json={
            **BASE,
            "width": 512,
            "height": 320,
            "num_frames": 49,
            "conditioning_images": [{"image_id": "image-a", "frame_idx": 8, "strength": 0.8}],
        },
    )
    assert r.status_code == 422
    assert "frame_idx=0" in r.text


def test_distilled_requires_8_steps(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49, "num_inference_steps": 20},
    )
    assert r.status_code == 422
    assert "num_inference_steps=8" in r.text
