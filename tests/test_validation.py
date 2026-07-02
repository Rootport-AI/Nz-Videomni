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
    # SIX images (all otherwise valid) exceeds the cap of 5 -> rejected.
    r = client.post(
        "/api/v1/generate",
        json={
            **BASE,
            "width": 512,
            "height": 320,
            "num_frames": 49,
            "conditioning_images": [
                {"image_id": f"image-{i}", "frame_idx": i * 8, "strength": 0.8}
                for i in range(6)
            ],
        },
    )
    assert r.status_code == 422
    assert "at most 5" in r.text


def _upload(client, png_bytes) -> str:
    up = client.post("/api/v1/upload/image", files={"file": ("k.png", png_bytes, "image/png")})
    assert up.status_code == 200, up.text
    return up.json()["image_id"]


def test_five_conditioning_images_accepted(client, png_bytes):
    # Exactly 5 keyframes at 0,8,16,24,32 (real uploaded images) -> accepted (202).
    ids = [_upload(client, png_bytes) for _ in range(5)]
    r = client.post(
        "/api/v1/generate",
        json={
            **BASE,
            "width": 512,
            "height": 320,
            "num_frames": 49,
            "conditioning_images": [
                {"image_id": iid, "frame_idx": i * 8, "strength": 0.8}
                for i, iid in enumerate(ids)
            ],
        },
    )
    assert r.status_code == 202


def test_multi_keyframe_accepted(client, png_bytes):
    # 3 keyframes at 0,24,48 (all on-grid, in range for num_frames=49) -> accepted.
    ids = [_upload(client, png_bytes) for _ in range(3)]
    r = client.post(
        "/api/v1/generate",
        json={
            **BASE,
            "width": 512,
            "height": 320,
            "num_frames": 49,
            "conditioning_images": [
                {"image_id": ids[0], "frame_idx": 0, "strength": 0.8},
                {"image_id": ids[1], "frame_idx": 24, "strength": 0.8},
                {"image_id": ids[2], "frame_idx": 48, "strength": 0.8},
            ],
        },
    )
    assert r.status_code == 202


def test_frame_idx_nonzero_accepted_and_snapped(client, png_bytes):
    # A single image with a non-zero, off-grid frame_idx is now ACCEPTED
    # (server snaps it to the nearest multiple of 8) rather than rejected.
    iid = _upload(client, png_bytes)
    r = client.post(
        "/api/v1/generate",
        json={
            **BASE,
            "width": 512,
            "height": 320,
            "num_frames": 49,
            "conditioning_images": [{"image_id": iid, "frame_idx": 10, "strength": 0.8}],
        },
    )
    assert r.status_code == 202


def test_frame_idx_snap_and_clamp_math():
    # Pure-model unit test of the snap+clamp math, no HTTP layer.
    # Values chosen to be unambiguous under Python's banker's rounding of round():
    #   10/8=1.25 -> round=1 -> 8 ; 24/8=3.0 -> 24 ; 100 clamps to num_frames-1=48 ; 0 stays 0.
    from api.models import ConditioningImage, GenerateRequest

    def snapped(idx: int, num_frames: int = 49) -> int:
        req = GenerateRequest(
            prompt="x",
            width=512,
            height=320,
            num_frames=num_frames,
            num_inference_steps=8,
            guidance_scale=1.0,
            pipeline="distilled",
            conditioning_images=[ConditioningImage(image_id="a", frame_idx=idx, strength=0.8)],
        )
        return req.conditioning_images[0].frame_idx

    assert snapped(10) == 8
    assert snapped(24) == 24
    assert snapped(100, num_frames=49) == 48  # clamped to num_frames-1
    assert snapped(0) == 0


def test_distilled_requires_8_steps(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49, "num_inference_steps": 20},
    )
    assert r.status_code == 422
    assert "num_inference_steps=8" in r.text
