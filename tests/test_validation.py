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


# ─────────────────────────────────────────────────────────────────────────────
# VSF (Value Sign Flip, arXiv:2508.10931) — GenerateRequest / GenerateChainRequest.
# ─────────────────────────────────────────────────────────────────────────────


def test_neg_method_invalid_value_rejected_generate(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "neg_method": "not-a-method"},
    )
    assert r.status_code == 422


def test_neg_method_invalid_value_rejected_chain(client):
    r = client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BASE, "neg_method": "not-a-method"},
    )
    assert r.status_code == 422


def test_vsf_scale_out_of_range_rejected_generate(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "vsf_scale": -0.1},
    )
    assert r.status_code == 422


def test_vsf_scale_above_cap_rejected_generate(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "vsf_scale": 10.1},
    )
    assert r.status_code == 422


def test_vsf_scale_out_of_range_rejected_chain(client):
    r = client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BASE, "vsf_scale": -0.1},
    )
    assert r.status_code == 422


def test_vsf_scale_above_cap_rejected_chain(client):
    r = client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BASE, "vsf_scale": 10.1},
    )
    assert r.status_code == 422


def test_neg_method_vsf_requires_nonempty_negative_generate(client):
    # neg_method="vsf" is also covered by the shared nag_enabled validator
    # (nag_enabled is the non-CFG-negative master toggle for both methods).
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "nag_enabled": True, "negative_prompt": "", "neg_method": "vsf"},
    )
    assert r.status_code == 422
    assert "nag_enabled requires a non-empty negative_prompt" in r.text


def test_neg_method_vsf_requires_nonempty_negative_chain(client):
    r = client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BASE, "nag_enabled": True, "negative_prompt": "",
              "neg_method": "vsf"},
    )
    assert r.status_code == 422
    assert "nag_enabled requires a non-empty negative_prompt" in r.text


def test_vsf_fields_default_values_generate():
    from api.models import GenerateRequest

    req = GenerateRequest(
        prompt="x", width=512, height=320, num_frames=49,
        num_inference_steps=8, guidance_scale=1.0, pipeline="distilled",
    )
    assert req.neg_method == "nag"
    assert req.vsf_scale == 1.5


def test_vsf_fields_default_values_chain():
    from api.models import GenerateChainRequest

    req = GenerateChainRequest(**CHAIN_BASE)
    assert req.neg_method == "nag"
    assert req.vsf_scale == 1.5


def test_vsf_fields_accepted_generate():
    from api.models import GenerateRequest

    req = GenerateRequest(
        prompt="x", width=512, height=320, num_frames=49,
        num_inference_steps=8, guidance_scale=1.0, pipeline="distilled",
        nag_enabled=True, negative_prompt="blurry, low quality",
        neg_method="vsf", vsf_scale=1.7,
    )
    assert req.neg_method == "vsf"
    assert req.vsf_scale == 1.7


# ─────────────────────────────────────────────────────────────────────────────
# Acceleration — attention_backend / block_swap_prefetch /
# fused_gguf_dequant_kernel / vae_mode. All four are implemented; vae_mode was
# the last mock and became real on 2026-08-05 (PrunaVAED, §3-50). Only its
# DEFAULT is special: permanently "default", because the pruned decoder changes
# the picture.
# ─────────────────────────────────────────────────────────────────────────────


def test_acceleration_fields_default_values_generate():
    from api.models import GenerateRequest

    req = GenerateRequest(
        prompt="x", width=512, height=320, num_frames=49,
        num_inference_steps=8, guidance_scale=1.0, pipeline="distilled",
    )
    assert req.attention_backend == "sdpa"
    # S4 (2026-08-01): default flipped True once the real-device gate
    # (bit-exact output + VRAM headroom, G1-G7) passed.
    assert req.block_swap_prefetch is True
    # §1-11 (2026-08-04): default flipped True once the real-device gates
    # G1-G8 passed (bit-identical output, ~17.5% faster) and the owner approved
    # -- the same "gate green -> default on" step block_swap_prefetch took.
    assert req.fused_gguf_dequant_kernel is True
    assert req.vae_mode == "default"


def test_acceleration_fields_default_values_chain():
    from api.models import GenerateChainRequest

    req = GenerateChainRequest(**CHAIN_BASE)
    assert req.attention_backend == "sdpa"
    assert req.block_swap_prefetch is True
    assert req.fused_gguf_dequant_kernel is True
    assert req.vae_mode == "default"


def test_acceleration_fields_accepted_generate(client):
    # All four at non-default values are ACCEPTED (no validator gates them —
    # sage availability is a runtime capability, not a request constraint: an
    # engine without sage degrades to sdpa rather than rejecting the job).
    # block_swap_prefetch=True is likewise accepted regardless of whether block
    # swap is actually configured server-side — it silently no-ops there.
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "attention_backend": "sage", "block_swap_prefetch": True,
              "fused_gguf_dequant_kernel": True,
              "vae_mode": "prune_vaed"},
    )
    assert r.status_code == 202, r.text


def test_acceleration_fields_accepted_chain(client):
    r = client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BASE, "attention_backend": "sage",
              "block_swap_prefetch": True,
              "fused_gguf_dequant_kernel": True, "vae_mode": "prune_vaed"},
    )
    assert r.status_code == 202, r.text


def test_attention_backend_invalid_value_rejected_generate(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "attention_backend": "flash"},
    )
    assert r.status_code == 422


def test_attention_backend_invalid_value_rejected_chain(client):
    r = client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BASE, "attention_backend": "flash"},
    )
    assert r.status_code == 422


def test_block_swap_prefetch_invalid_value_rejected_generate(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "block_swap_prefetch": "not-a-bool"},
    )
    assert r.status_code == 422


def test_block_swap_prefetch_invalid_value_rejected_chain(client):
    r = client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BASE, "block_swap_prefetch": "not-a-bool"},
    )
    assert r.status_code == 422


def test_vae_mode_invalid_value_rejected_generate(client):
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "vae_mode": "not-a-mode"},
    )
    assert r.status_code == 422


def test_chain_to_clip_request_transcribes_acceleration_fields():
    # DIRECT guard for the to_clip_request transcription (the LIVE path
    # job_store.create_chain_if_idle uses to build JobRecord.request). Unlike the
    # nag fields, an omission here does NOT fail validation -- it silently
    # degrades a chain job's stored request to the sdpa/default values, so GET
    # /jobs and metadata.json would mis-report what was asked for. Only this
    # direct check can catch it.
    from api.models import GenerateChainRequest

    model = GenerateChainRequest(**{
        **CHAIN_BASE,
        "attention_backend": "sage",
        "block_swap_prefetch": True,
        "fused_gguf_dequant_kernel": True,
        "vae_mode": "prune_vaed",
    })
    clip0 = model.to_clip_request(0)
    assert clip0.attention_backend == "sage"
    assert clip0.block_swap_prefetch is True
    assert clip0.fused_gguf_dequant_kernel is True
    assert clip0.vae_mode == "prune_vaed"

    # ...and the default chain transcribes the defaults (no accidental flip).
    plain = GenerateChainRequest(**CHAIN_BASE).to_clip_request(0)
    assert plain.attention_backend == "sdpa"
    assert plain.block_swap_prefetch is True
    assert plain.fused_gguf_dequant_kernel is True
    assert plain.vae_mode == "default"


# ─────────────────────────────────────────────────────────────────────────────
# keep_resident (cross-job CPU-skeleton cache, §48)
# ─────────────────────────────────────────────────────────────────────────────


def test_keep_resident_defaults_off_and_is_accepted(client):
    from api.models import (
        KEEP_RESIDENT_DEFAULT,
        GenerateChainRequest,
        GenerateRequest,
    )

    # The default is OFF on BOTH models -- the opposite direction from
    # block_swap_prefetch, which is the single easiest thing to get backwards
    # when copying that feature's wiring.
    assert KEEP_RESIDENT_DEFAULT is False
    req = GenerateRequest(
        prompt="x", width=512, height=320, num_frames=49,
        num_inference_steps=8, guidance_scale=1.0, pipeline="distilled",
    )
    assert req.keep_resident is False
    assert GenerateChainRequest(**CHAIN_BASE).keep_resident is False

    # Accepted at both endpoints with no availability gate: whether ~20GB of
    # main memory is a good idea is a property of the user's machine, not of
    # the request, so the server never rejects it (and /status does not
    # advertise it either).
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "keep_resident": True},
    )
    assert r.status_code == 202, r.text
    r = client.post(
        "/api/v1/generate/chain",
        json={**CHAIN_BASE, "keep_resident": True},
    )
    assert r.status_code == 202, r.text

    # A non-bool is still a 422 (pydantic), like block_swap_prefetch.
    r = client.post(
        "/api/v1/generate",
        json={**BASE, "width": 512, "height": 320, "num_frames": 49,
              "keep_resident": "not-a-bool"},
    )
    assert r.status_code == 422


def test_chain_to_clip_request_transcribes_keep_resident():
    # Same trap as the acceleration fields above: an omission here does NOT
    # fail validation, it just makes a chain job's stored request (GET /jobs,
    # metadata.json) claim keep_resident=False for a run that asked for True.
    from api.models import GenerateChainRequest

    model = GenerateChainRequest(**{**CHAIN_BASE, "keep_resident": True})
    assert model.to_clip_request(0).keep_resident is True
    assert GenerateChainRequest(**CHAIN_BASE).to_clip_request(0).keep_resident is False


# ─────────────────────────────────────────────────────────────────────────────
# EndSourceSpec (end source — the chain's LAST frames come from an upload).
# Schema-level only; the geometry cross-checks (does the band fit the final clip
# / the last stage-2 tile?) live in chain_math and are covered by
# tests/test_end_source_chain.py.
# ─────────────────────────────────────────────────────────────────────────────


def test_end_source_spec_defaults_to_72_context_frames():
    from api.models import EndSourceSpec

    spec = EndSourceSpec(video_id="vid")
    assert spec.context_frames == 72
    assert spec.image_id is None


def test_end_source_spec_accepts_an_image_id_instead():
    from api.models import EndSourceSpec

    spec = EndSourceSpec(image_id="img", context_frames=8)
    assert spec.video_id is None
    assert spec.context_frames == 8


def test_end_source_spec_rejects_both_ids():
    import pytest
    from api.models import EndSourceSpec

    with pytest.raises(ValueError, match="exactly one of video_id / image_id"):
        EndSourceSpec(video_id="vid", image_id="img")


def test_end_source_spec_rejects_neither_id():
    import pytest
    from api.models import EndSourceSpec

    with pytest.raises(ValueError, match="exactly one of video_id / image_id"):
        EndSourceSpec()


def test_end_source_spec_rejects_non_multiple_of_8():
    # The tail grid is a MULTIPLE of 8, not the head grid's 8n+1: 73 is a legal
    # head span and an illegal tail span, which is exactly the confusion this
    # check exists to catch.
    import pytest
    from api.models import EndSourceSpec

    with pytest.raises(ValueError, match="multiple of 8"):
        EndSourceSpec(video_id="vid", context_frames=73)


def test_end_source_spec_enforces_the_published_bounds():
    import pytest
    from api.models import EndSourceSpec
    from config import LimitsConfig

    limits = LimitsConfig()
    with pytest.raises(ValueError, match=f">= {limits.end_context_frames_min}"):
        EndSourceSpec(video_id="vid", context_frames=4)
    with pytest.raises(ValueError, match=f"<= {limits.end_context_frames_max}"):
        EndSourceSpec(video_id="vid", context_frames=144)
    # Both ends of the published range are themselves legal.
    assert EndSourceSpec(video_id="v", context_frames=8).context_frames == 8
    assert EndSourceSpec(video_id="v", context_frames=136).context_frames == 136


def test_chain_to_clip_request_does_not_transcribe_end_source():
    # Deliberate, following the retake / source_video precedent: GenerateRequest
    # has no counterpart to drop it into, and the authoritative copy lives on
    # JobRecord.chain_request (plus metadata["request"] via model_dump).
    from api.models import GenerateChainRequest

    model = GenerateChainRequest(**{
        **CHAIN_BASE,
        "clips": [{"num_frames": 25}, {"num_frames": 41}],
        "end_source": {"video_id": "vid", "context_frames": 24},
    })
    assert model.end_source is not None
    assert not hasattr(model.to_clip_request(0), "end_source")
