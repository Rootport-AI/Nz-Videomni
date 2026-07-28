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
    # 288 (the old minimal height) is ÷32 but not ÷64 -> rejected.
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
    # (server snaps it to the 8n+1 latent-frame-start grid) rather than rejected.
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
    # Pure-model unit test of the official snap+clamp math, no HTTP layer.
    # frame_idx 0 stays 0 (latent-replace start frame). Every other keyframe snaps
    # to the 8n+1 latent-frame-START grid via (f-1)//8*8+1, then clamps to
    # [1, num_frames-8] (num_frames=49 -> last start = 41). Matches ComfyUI
    # LTXVAddGuide.get_latent_index. Each value hand-verified.
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

    assert snapped(0) == 0  # start frame, unchanged (latent-replace path)
    assert snapped(1) == 1  # (0)//8*8+1
    assert snapped(8) == 1  # (7)//8*8+1 -> last-latent-start below 8
    assert snapped(10) == 9  # (9)//8*8+1
    assert snapped(16) == 9  # (15)//8*8+1
    assert snapped(24) == 17  # (23)//8*8+1
    assert snapped(48) == 41  # (47)//8*8+1 = last start pixel for num_frames=49
    assert snapped(100, num_frames=49) == 41  # snap 97 clamped to num_frames-8=41


def test_distilled_requires_8_steps(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49, "num_inference_steps": 20},
    )
    assert r.status_code == 422
    assert "num_inference_steps=8" in r.text


# ─────────────────────────────────────────────────────────────────────────────
# NAG (Normalized Attention Guidance) — GenerateRequest / GenerateChainRequest.
# ─────────────────────────────────────────────────────────────────────────────

CHAIN_BASE = {
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
}


def test_nag_enabled_requires_nonempty_negative_generate(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "nag_enabled": True, "negative_prompt": ""},
    )
    assert r.status_code == 422
    assert "nag_enabled requires a non-empty negative_prompt" in r.text


def test_nag_enabled_requires_nonempty_negative_generate_whitespace_only(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "nag_enabled": True, "negative_prompt": "   "},
    )
    assert r.status_code == 422
    assert "nag_enabled requires a non-empty negative_prompt" in r.text


def test_nag_enabled_requires_nonempty_negative_chain(client):
    r = client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BASE, "nag_enabled": True, "negative_prompt": ""},
    )
    assert r.status_code == 422
    assert "nag_enabled requires a non-empty negative_prompt" in r.text


def test_nag_enabled_requires_nonempty_negative_chain_whitespace_only(client):
    r = client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BASE, "nag_enabled": True, "negative_prompt": "   "},
    )
    assert r.status_code == 422
    assert "nag_enabled requires a non-empty negative_prompt" in r.text


def test_nag_enabled_with_negative_prompt_accepted_and_defaults():
    from api.models import GenerateRequest

    req = GenerateRequest(
        prompt="x", width=512, height=320, num_frames=49,
        num_inference_steps=8, guidance_scale=1.0, pipeline="distilled",
        nag_enabled=True, negative_prompt="blurry, low quality",
    )
    assert req.nag_enabled is True
    assert req.nag_scale == 11.0
    assert req.nag_tau == 2.5
    assert req.nag_alpha == 0.25


def test_nag_enabled_with_negative_prompt_accepted_and_defaults_chain():
    from api.models import GenerateChainRequest

    req = GenerateChainRequest(
        **{**CHAIN_BASE, "nag_enabled": True, "negative_prompt": "blurry, low quality"}
    )
    assert req.nag_enabled is True
    assert req.nag_scale == 11.0
    assert req.nag_tau == 2.5
    assert req.nag_alpha == 0.25


def test_nag_scale_out_of_range_rejected(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "negative_prompt": "x", "nag_scale": 0.5},
    )
    assert r.status_code == 422


def test_nag_tau_out_of_range_rejected(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "negative_prompt": "x", "nag_tau": 0.5},
    )
    assert r.status_code == 422


def test_nag_alpha_out_of_range_rejected(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "negative_prompt": "x", "nag_alpha": 1.5},
    )
    assert r.status_code == 422


def test_nag_fields_omitted_defaults_disabled(client):
    # Regression guard: a request that omits every nag field is unaffected.
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49},
    )
    assert r.status_code == 202
    from api.models import GenerateRequest

    req = GenerateRequest(
        prompt="x", width=512, height=320, num_frames=49,
        num_inference_steps=8, guidance_scale=1.0, pipeline="distilled",
    )
    assert req.nag_enabled is False


def test_negative_prompt_over_2000_chars_rejected(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "negative_prompt": "x" * 2001},
    )
    assert r.status_code == 422


def test_negative_prompt_over_2000_chars_rejected_chain(client):
    r = client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BASE, "negative_prompt": "x" * 2001},
    )
    assert r.status_code == 422
