"""Generate-chain (clip-concatenation) tests through the REST API (mock backend).

The mock runner renders ONE synthetic full-timeline mp4 + junction metadata
(via chain_math), so the chain endpoint, job layer, and metadata are exercised
end-to-end without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path


BASE = {
    "prompt": "a serene mountain lake at dawn",
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
    r = client.post("/api/v1/generate/chain", json=payload)
    return r


def test_chain_2clip_completes_single_mp4(client):
    clips = [{"num_frames": 25}, {"num_frames": 25}]
    r = _run_chain(client, clips)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["num_clips"] == 2
    job_id = body["job_id"]

    jr = client.get(f"/api/v1/jobs/{job_id}")
    assert jr.status_code == 200
    job = jr.json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    assert out.exists() and out.stat().st_size > 0

    vid = client.get(f"/api/v1/jobs/{job_id}/video")
    assert vid.status_code == 200
    assert vid.headers["content-type"] == "video/mp4"


def test_chain_3clip_completes(client):
    clips = [{"num_frames": 25}, {"num_frames": 17}, {"num_frames": 17}]
    r = _run_chain(client, clips)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert meta["kind"] == "chain"
    assert meta["chain"]["num_clips"] == 3
    assert meta["chain"]["clip_num_frames"] == [25, 17, 17]
    assert meta["chain"]["architecture"] == "masked_av_latent_concat"


def test_chain_junction_metadata_matches_chain_math(client):
    # Junction indices in metadata must equal the shared chain_math layout, and
    # every junction must be a valid 0-based frame index of the final mp4.
    import chain_math
    from services import video_io

    clips = [{"num_frames": 25}, {"num_frames": 25}, {"num_frames": 25}]
    r = _run_chain(client, clips)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    layout = chain_math.compute_chain_layout([25, 25, 25], 24.0, kv=BASE["overlap_frames"])
    ch = meta["chain"]
    assert ch["segment_seam_junctions"] == layout.segment_seam_junctions
    assert ch["tile_seam_junctions"] == layout.tile_seam_junctions
    assert ch["all_junctions"] == layout.all_junctions
    assert ch["total_frames"] == layout.total_px

    out = ctx.config.output_dir / job_id / "output.mp4"
    nf = video_io.frame_count(out)
    if nf is not None:
        assert nf == layout.total_px
        for j in layout.all_junctions:
            assert 0 <= j < nf - 1


def test_chain_prompt_propagation(client):
    # clip 0 uses the global base prompt; clip 1 overrides.
    clips = [
        {"num_frames": 25},
        {"num_frames": 25, "prompt": "the camera pushes in on a red boat"},
    ]
    r = _run_chain(client, clips)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    req = meta["request"]
    # base prompt stored; clip-1 override preserved in the clip spec.
    assert req["prompt"] == BASE["prompt"]
    assert req["clips"][0]["prompt"] is None  # falls back to base
    assert req["clips"][1]["prompt"] == "the camera pushes in on a red boat"

    # Confirm the resolved effective prompt logic directly on the model.
    from api.models import GenerateChainRequest

    model = GenerateChainRequest(**{**BASE, "clips": clips})
    assert model.clip_prompt(0) == BASE["prompt"]
    assert model.clip_prompt(1) == "the camera pushes in on a red boat"


def test_chain_duration_matches_timeline(client):
    # Masked AV-latent chain: total timeline = chain_math geometry (sum of the
    # per-clip video latent frames minus the shared K_v overlaps, back to pixels).
    import chain_math

    clips = [{"num_frames": 25}, {"num_frames": 25}]
    r = _run_chain(client, clips)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    layout = chain_math.compute_chain_layout([25, 25], 24.0, kv=BASE["overlap_frames"])
    expected_dur = layout.total_px / 24.0
    assert abs(job["result"]["duration_seconds"] - expected_dur) < 0.01

    # ffprobe the actual mp4 duration to within ~2 frames.
    from services import video_io

    ctx = client.app_context
    out: Path = ctx.config.output_dir / job_id / "output.mp4"
    probed = video_io.probe_duration(out)
    if probed is not None:  # ffprobe present
        assert abs(probed - expected_dur) < (2.0 / 24.0)


def test_chain_busy_returns_409(client):
    from api.models import GenerateRequest

    ctx = client.app_context
    active = GenerateRequest(prompt="busy", width=384, height=256, num_frames=17)
    ctx.job_store.create(active)  # queued -> active

    r = _run_chain(client, [{"num_frames": 25}, {"num_frames": 25}])
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "JOB_BUSY"


def test_chain_rejects_single_clip(client):
    r = _run_chain(client, [{"num_frames": 25}])
    assert r.status_code == 422


def test_chain_rejects_bad_frames(client):
    # 24 is not 8n+1.
    r = _run_chain(client, [{"num_frames": 25}, {"num_frames": 24}])
    assert r.status_code == 422
    assert "8n+1" in r.text


def test_chain_rejects_bad_resolution(client):
    r = _run_chain(client, [{"num_frames": 25}, {"num_frames": 25}], width=544)
    assert r.status_code == 422
    assert "multiple of 64" in r.text


def test_chain_rejects_empty_clip_list(client):
    r = _run_chain(client, [])
    assert r.status_code == 422


def test_chain_rejects_conditioning_on_nonfirst_clip(client):
    clips = [
        {"num_frames": 25},
        {"num_frames": 25, "conditioning_images": [
            {"image_id": "x", "frame_idx": 0, "strength": 0.8}
        ]},
    ]
    r = _run_chain(client, clips)
    assert r.status_code == 422
    assert "only clip 0" in r.text


def test_chain_rejects_overlap_larger_than_clip(client):
    # overlap_frames must be < each non-first clip's stage-1 latent frames.
    # num_frames=9 -> stage1 = (9-1)//8+1 = 2 latent frames; overlap 8 >= 2.
    clips = [{"num_frames": 25}, {"num_frames": 9}]
    r = _run_chain(client, clips, overlap_frames=8)
    assert r.status_code == 422
    assert "overlap_frames" in r.text


def test_chain_clip0_conditioning_image_missing_404(client):
    clips = [
        {"num_frames": 25, "conditioning_images": [
            {"image_id": "nope", "frame_idx": 0, "strength": 0.8}
        ]},
        {"num_frames": 25},
    ]
    r = _run_chain(client, clips)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "IMAGE_NOT_FOUND"


# ─────────────────────────────────────────────────────────────────────────────
# Unequal-length chains (regression for the stage-1 carry init-shape bug).
#
# The engine used to size segment i>0's init latents with ``zeros_like(prev)`` —
# the PREVIOUS segment's frame count — which crashed create_initial_state whenever
# consecutive clips had different num_frames (first surfaced by V2V e2e E2E-B,
# clips [145, 73]). The fix sizes the init tensors from the CURRENT segment's
# latent shape. Every prior chain test used uniform clip lengths, so it stayed
# latent. These tests pin unequal lengths as a first-class, accepted input.
# ─────────────────────────────────────────────────────────────────────────────


def test_chain_unequal_lengths_completes_and_matches_chain_math(client):
    # [145, 73] is the exact E2E-B repro geometry (minus source_video, which the
    # mock backend does not encode). Longer-then-shorter is the direction that
    # crashed: prev segment (145 -> 19 latent) larger than current (73 -> 10).
    import chain_math

    clips = [{"num_frames": 145}, {"num_frames": 73}]
    r = _run_chain(client, clips)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    layout = chain_math.compute_chain_layout([145, 73], 24.0, kv=BASE["overlap_frames"])
    ch = meta["chain"]
    assert ch["clip_num_frames"] == [145, 73]
    assert ch["total_frames"] == layout.total_px
    assert ch["segment_seam_junctions"] == layout.segment_seam_junctions
    assert ch["all_junctions"] == layout.all_junctions


def test_chain_unequal_lengths_shorter_then_longer(client):
    # [73, 145] — the reverse direction (Gate 2b geometry): current segment
    # larger than prev. Both directions must be accepted and complete.
    import chain_math

    clips = [{"num_frames": 73}, {"num_frames": 145}]
    r = _run_chain(client, clips)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    layout = chain_math.compute_chain_layout([73, 145], 24.0, kv=BASE["overlap_frames"])
    assert meta["chain"]["clip_num_frames"] == [73, 145]
    assert meta["chain"]["total_frames"] == layout.total_px


def test_chain_carry_always_fits_current_segment():
    """Invariant the engine stage-1 carry relies on: for every inter-clip join the
    frozen carry (K_v video / K_a audio latent frames, copied from the previous
    segment's tail into the current segment's head) fits inside BOTH adjacent
    segments' latent frame counts.

    The video half is guaranteed by the request/geometry guard
    (``overlap_frames < every clip's stage-1 latent frames``). The audio half has
    no separate validator; this test confirms the video guard + geometry implies
    ``ka_list[j] <= min(seg_audio[j], seg_audio[j+1])`` for a broad set of unequal
    configs (including the E2E-B repro and high-fps stress, where audio latent
    frames are scarcest). If this ever fails, an explicit ka guard belongs in
    chain_math (the geometry SoT), not the engine.
    """
    import chain_math

    configs = [
        ([145, 73], 24.0),
        ([73, 145], 24.0),
        ([145, 25, 73], 24.0),
        ([25, 145], 24.0),
        ([25, 33, 33, 33], 60.0),   # min-slack case from the exhaustive scan
        ([9, 9, 9], 60.0),          # shortest legal audio segments, high fps
        ([73, 73], 60.0),
    ]
    for clip_frames, fps in configs:
        for kv in range(1, 9):
            seg_lat = [chain_math.v_latent_frames(f) for f in clip_frames]
            if any(kv >= L for L in seg_lat):   # rejected by the video guard
                continue
            try:
                layout = chain_math.compute_chain_layout(clip_frames, fps, kv=kv)
            except ValueError:
                continue   # geometrically-degenerate config -> 422 before the engine

            for j, ka in enumerate(layout.ka_list):
                assert kv < layout.seg_latent[j] and kv < layout.seg_latent[j + 1], (
                    clip_frames, fps, kv, j, layout.seg_latent
                )
                limit = min(layout.seg_audio[j], layout.seg_audio[j + 1])
                assert ka <= limit, (clip_frames, fps, kv, j, ka, layout.seg_audio)
