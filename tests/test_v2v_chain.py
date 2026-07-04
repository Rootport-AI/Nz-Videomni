"""Video-to-video continuation (source_video on POST /generate/chain) — API +
mock e2e (no GPU).

The mock runner mirrors the engine geometry via chain_math
(``compute_chain_layout(source_context_px=...)``): it renders ONE synthetic mp4
of ``new_frames_px`` frames (the NEW part only, matching the engine's context
trim) and emits the same ``chain.v2v`` key set the real worker does, so the full
additive contract (schema, endpoint 404/422, tail-cut, metadata v2v block) is
pinned without weights. Frozen-API discipline: a request omitting source_video
is byte-shape identical to before (regression test below).
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

import chain_math
from services import video_io


BASE = {
    "prompt": "a bustling town square at dusk, cinematic trailer",
    "width": 384,
    "height": 256,
    "frame_rate": 24.0,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": 123,
    "pipeline": "distilled",
    "overlap_frames": 2,
    "overlap_strength": 0.5,
}


def _run_chain(client, clips, **overrides):
    payload = {**BASE, "clips": clips, **overrides}
    return client.post("/api/v1/generate/chain", json=payload)


def _make_source_mp4(path, n_frames, fps, size=(96, 64)):
    frames = [Image.new("RGB", size, (i * 4 % 256, 90, 160)) for i in range(n_frames)]
    video_io.encode_frames_to_mp4(frames, path, frame_rate=fps)
    return path


def _upload_source(client, mp4_path) -> str:
    data = mp4_path.read_bytes()
    r = client.post("/api/v1/upload/video", files={"file": ("src.mp4", data, "video/mp4")})
    assert r.status_code == 200, r.text
    return r.json()["video_id"]


# ---------------------------------------------------------------- (a) regression


def test_chain_without_source_regression(client):
    """A chain omitting source_video: source_video defaults to None, no top-level
    or chain.v2v block — the pre-V2V metadata key set is unchanged."""
    r = _run_chain(client, [{"num_frames": 25}, {"num_frames": 25}])
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    assert meta["request"]["source_video"] is None
    assert "v2v" not in meta
    assert "v2v" not in meta["chain"]


# ------------------------------------------------------------------- (b) 422s


def test_v2v_context_not_8n1_422(client):
    r = _run_chain(client, [{"num_frames": 49}], source_video={"video_id": "x", "context_frames": 26})
    assert r.status_code == 422
    assert "8n+1" in r.text


def test_v2v_context_below_min_422(client):
    r = _run_chain(client, [{"num_frames": 49}], source_video={"video_id": "x", "context_frames": 17})
    assert r.status_code == 422
    assert ">= 25" in r.text


def test_v2v_context_above_max_422(client):
    # 153 = 8*19+1 (valid 8n+1) but exceeds the conservative v1 cap of 145.
    r = _run_chain(client, [{"num_frames": 481}], source_video={"video_id": "x", "context_frames": 153})
    assert r.status_code == 422
    assert "145" in r.text


def test_v2v_context_ge_clip0_422(client):
    r = _run_chain(client, [{"num_frames": 49}], source_video={"video_id": "x", "context_frames": 49})
    assert r.status_code == 422
    assert "must be <" in r.text


def test_chain_single_clip_without_source_still_422(client):
    # Preserve the pre-V2V rejection: 1 clip and no source is just /generate.
    r = _run_chain(client, [{"num_frames": 49}])
    assert r.status_code == 422


def test_v2v_conflicts_with_clip0_conditioning_422(client):
    clips = [{
        "num_frames": 49,
        "conditioning_images": [{"image_id": "x", "frame_idx": 0, "strength": 0.8}],
    }]
    r = _run_chain(client, clips, source_video={"video_id": "x", "context_frames": 25})
    assert r.status_code == 422
    assert "mutually exclusive" in r.text


# -------------------------------------------------------------------- (c) 404


def test_v2v_unknown_video_id_404(client):
    r = _run_chain(client, [{"num_frames": 49}], source_video={"video_id": "no-such-id", "context_frames": 25})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "SOURCE_VIDEO_NOT_FOUND"


# ---------------------------------------------------------------- (d) mock e2e


def test_v2v_mock_e2e_resample(client, tmp_path):
    """Upload a 30fps source, request a 24fps continuation (1 clip) -> completed;
    mp4 holds new_frames_px frames (NEW part only) and the metadata v2v block is
    correct incl. the resampled flag."""
    src = _make_source_mp4(tmp_path / "src30.mp4", n_frames=60, fps=30.0)
    vid = _upload_source(client, src)

    r = _run_chain(
        client, [{"num_frames": 49}],
        frame_rate=24.0,
        source_video={"video_id": vid, "context_frames": 25},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    body = r.json()
    assert body["num_clips"] == 1

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    assert out.exists() and out.stat().st_size > 0

    layout = chain_math.compute_chain_layout([49], 24.0, kv=2, source_context_px=25)
    assert video_io.frame_count(out) == layout.new_frames_px  # NEW part only (49-25=24)

    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    v2v = meta["v2v"]
    # geometry (from chain_math)
    assert v2v["context_frames"] == 25
    assert v2v["source_context_px"] == 25
    assert v2v["trimmed_px"] == 25
    assert v2v["new_frames_px"] == layout.new_frames_px
    assert v2v["n_ctx_v"] == layout.n_ctx_v
    assert v2v["n_ctx_a"] == layout.n_ctx_a
    assert v2v["decoded_frames_px"] == layout.total_px
    assert v2v["v2v_context_junction_px"] == layout.v2v_context_junction_px
    # provenance (app-side)
    assert v2v["source_video_id"] == vid
    assert v2v["resampled"] is True
    assert abs(v2v["source_fps"] - 30.0) < 0.5
    # no-audio source -> free audio head
    assert v2v["source_had_audio"] is False
    assert v2v["freeze_ka"] == 0
    # audio_head_frozen (Fix 2): distinct from source_had_audio, = freeze_ka > 0.
    # Here the source has no audio at all, so both are False/0 in lockstep, but
    # the key must exist and carry the right (derived) value regardless.
    assert v2v["audio_head_frozen"] is False
    # the additive request field round-trips
    assert meta["request"]["source_video"]["context_frames"] == 25
    assert meta["request"]["source_video"]["video_id"] == vid


def test_v2v_mock_e2e_same_fps_not_resampled(client, tmp_path):
    src = _make_source_mp4(tmp_path / "src24.mp4", n_frames=50, fps=24.0)
    vid = _upload_source(client, src)
    r = _run_chain(
        client, [{"num_frames": 49}],
        frame_rate=24.0,
        source_video={"video_id": vid, "context_frames": 25},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    ctx = client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    assert meta["v2v"]["resampled"] is False


def test_v2v_mock_e2e_audio_handle_sidecar(client, tmp_path):
    """The mock mirrors the engine's opt-in audio-handle sidecar: metadata v2v
    carries ``audio_handle_filename`` + ``handle_context_seconds`` and the placeholder
    wav (synthetic silence) exists next to output.mp4 with a plausible duration."""
    src = _make_source_mp4(tmp_path / "src24.mp4", n_frames=50, fps=24.0)
    vid = _upload_source(client, src)
    r = _run_chain(
        client, [{"num_frames": 49}],
        frame_rate=24.0,
        source_video={"video_id": vid, "context_frames": 25},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    job_dir = ctx.config.output_dir / job_id
    meta = json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))
    v2v = meta["v2v"]

    layout = chain_math.compute_chain_layout([49], 24.0, kv=2, source_context_px=25)
    # additive keys present
    assert v2v["audio_handle_filename"] == "output_audio_handle.wav"
    # junction offset inside the handle = context (trim) duration
    assert abs(v2v["handle_context_seconds"] - layout.trim_px / 24.0) < 1e-4

    # sidecar exists with a plausible duration (full untrimmed timeline @ 24fps)
    handle_wav = job_dir / v2v["audio_handle_filename"]
    assert handle_wav.exists() and handle_wav.stat().st_size > 0
    handle_dur = video_io.probe_duration(handle_wav)
    assert handle_dur is not None
    assert abs(handle_dur - layout.total_px / 24.0) < 0.05


# -------------------------------------------------------------- (e) too short


def test_v2v_source_too_short_422(client, tmp_path):
    src = _make_source_mp4(tmp_path / "short.mp4", n_frames=15, fps=24.0)
    vid = _upload_source(client, src)
    r = _run_chain(
        client, [{"num_frames": 49}],
        frame_rate=24.0,
        source_video={"video_id": vid, "context_frames": 25},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "SOURCE_VIDEO_TOO_SHORT"


# ---------------------------------------------------- (f) stage-2 tile-fit invariant
#
# The frozen source head (n_ctx_v latent frames) is hard-frozen ONLY inside
# stage-2 TILE 0 (variant-B hard-freeze). compute_chain_layout is the single
# source of truth that enforces this BEFORE any GPU work (worker + direct
# harness both funnel through it) — see chain_math.compute_chain_layout and
# api.models.SourceVideoSpec's docstring for the same invariant restated at the
# API-cap layer.


def test_chain_layout_source_context_exceeds_tile_raises():
    # 177 = 8*22+1 -> n_ctx_v = v_latent_frames(177) = 23 > STAGE2_V_TILE (22).
    with pytest.raises(ValueError, match="exceeds stage-2 tile"):
        chain_math.compute_chain_layout(
            [481], 24.0, kv=2,
            v_tile=chain_math.STAGE2_V_TILE, v_adv=chain_math.STAGE2_V_ADV,
            source_context_px=177,
        )


def test_chain_layout_source_context_exactly_fills_tile_ok():
    # 169 = 8*21+1 -> n_ctx_v = v_latent_frames(169) = 22 == STAGE2_V_TILE: fits
    # tile 0 exactly, must NOT raise.
    layout = chain_math.compute_chain_layout(
        [481], 24.0, kv=2,
        v_tile=chain_math.STAGE2_V_TILE, v_adv=chain_math.STAGE2_V_ADV,
        source_context_px=169,
    )
    assert layout.n_ctx_v == chain_math.STAGE2_V_TILE


def test_v2v_context_frames_max_within_chain_math_ceiling():
    """Guard: config.limits.v2v_context_frames_max must stay <= the chain_math
    stage-2 tile-fit ceiling (px_from_v_latent(STAGE2_V_TILE) = 169). If a future
    config bump raises the cap past this, chain_math.compute_chain_layout would
    reject the config's own advertised max — this test fails CI first."""
    from config import LimitsConfig

    ceiling = chain_math.px_from_v_latent(chain_math.STAGE2_V_TILE)
    assert LimitsConfig().v2v_context_frames_max <= ceiling
