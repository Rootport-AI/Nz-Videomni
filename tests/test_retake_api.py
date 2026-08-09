"""Retake (temporal inpainting) — API validation + mock e2e (no GPU).

The mock runner mirrors the engine geometry through the same
``compute_chain_layout(retake_glue_px=...)``, so the whole additive contract
(schema 422s, 404, the app-side window cut, the metadata block) is pinned here
without weights. Frozen-API discipline: a request omitting ``retake`` behaves
exactly as before (regression test below).
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

BASE = {
    "prompt": "a fox trotting through tall grass",
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

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _post(client, clips, **over):
    return client.post("/api/v1/generate/chain", json={**BASE, "clips": clips, **over})


def _retake(**over):
    return {"video_id": "vid", "window_start_sec": 0.0, **over}


def _upload_window_source(client, tmp_path, *, frames=337, fps=24, audio=True):
    """Upload a real mp4 long enough to cut a 169-frame window out of."""
    src = tmp_path / "retake_src.mp4"
    cmd = [shutil.which("ffmpeg"), "-y",
           "-f", "lavfi", "-i", f"testsrc2=size=384x256:rate={fps}:duration=20"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=20"]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr", "-r", str(fps)]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(src)]
    subprocess.run(cmd, check=True, capture_output=True)
    r = client.post("/api/v1/upload/video",
                    files={"file": ("src.mp4", src.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    return r.json()["video_id"]


# ── regression: nothing changes when retake is absent ───────────────────────
def test_chain_without_retake_is_unchanged(client):
    r = _post(client, [{"num_frames": 25}, {"num_frames": 25}])
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    ctx = client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text("utf-8"))
    assert meta["request"]["retake"] is None
    assert "retake" not in meta
    assert "retake" not in meta["chain"]


def test_two_clip_chain_without_any_source_still_requires_two_clips(client):
    # The clips>=2 floor must keep rejecting a bare single-clip chain; adding
    # retake to that condition must not have punched a hole in it.
    r = _post(client, [{"num_frames": 25}])
    assert r.status_code == 422
    assert "at least 2 clips" in r.text


# ── the 422 matrix ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "clips, over, needle",
    [
        # window geometry (delegated to chain_math)
        ([{"num_frames": 65}], {"retake": _retake()}, "[73, 169]"),
        ([{"num_frames": 177}], {"retake": _retake()}, "[73, 169]"),
        # glue-band grids (schema-level)
        ([{"num_frames": 169}], {"retake": _retake(head_px=24)}, "head_px must be 8n+1"),
        ([{"num_frames": 169}], {"retake": _retake(tail_px=25)}, "tail_px must be a multiple of 8"),
        # glue bands that leave no middle
        ([{"num_frames": 73}], {"retake": _retake(head_px=41, tail_px=40)}, "free middle"),
        # exclusivity
        ([{"num_frames": 169}],
         {"retake": _retake(), "source_video": {"video_id": "v", "context_frames": 25}},
         "retake and source_video are mutually exclusive"),
        ([{"num_frames": 169}],
         {"retake": _retake(), "source_audio": {"audio_id": "a"}},
         "retake and source_audio are mutually exclusive"),
        ([{"num_frames": 169}],
         {"retake": _retake(), "reference_video_id": "r"},
         "retake and reference_video_id are mutually exclusive"),
        ([{"num_frames": 169, "conditioning_images": [{"image_id": "i", "frame_idx": 0}]}],
         {"retake": _retake()},
         "mutually exclusive with clips[0].conditioning_images"),
        # clip count
        ([{"num_frames": 169}, {"num_frames": 169}], {"retake": _retake()},
         "retake requires exactly 1 clip"),
        # stage-2 window
        ([{"num_frames": 169}], {"retake": _retake(), "stage2_window": "high_resolution"},
         "retake requires stage2_window='standard'"),
    ],
)
def test_422_matrix(client, clips, over, needle):
    r = _post(client, clips, **over)
    assert r.status_code == 422, r.text
    assert needle in r.text, r.text


def test_negative_window_start_is_rejected(client):
    r = _post(client, [{"num_frames": 169}], retake=_retake(window_start_sec=-0.5))
    assert r.status_code == 422


# ── 404 ─────────────────────────────────────────────────────────────────────
def test_unknown_video_id_is_404_with_its_own_code(client):
    r = _post(client, [{"num_frames": 169}], retake=_retake(video_id="nope"))
    assert r.status_code == 404, r.text
    assert "RETAKE_VIDEO_NOT_FOUND" in r.text


# ── to_clip_request must not transcribe retake ──────────────────────────────
def test_retake_never_leaks_into_the_per_clip_request():
    from api.models import GenerateChainRequest

    req = GenerateChainRequest(
        **BASE, clips=[{"num_frames": 169}],
        retake={"video_id": "v", "window_start_sec": 1.0},
    )
    clip_req = req.to_clip_request(0)
    assert not hasattr(clip_req, "retake")
    assert "retake" not in clip_req.model_dump()
    # ...and the authoritative copy is still on the chain request itself.
    assert req.retake.video_id == "v"


# ── mock e2e ────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_mock_e2e_delivers_the_whole_window_and_the_metadata_contract(client, tmp_path):
    vid = _upload_window_source(client, tmp_path)
    r = _post(client, [{"num_frames": 169}],
              retake=_retake(video_id=vid, window_start_sec=2.0))
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    out_dir = ctx.config.output_dir / job_id
    meta = json.loads((out_dir / "metadata.json").read_text("utf-8"))
    rt = meta["retake"]

    # geometry (from chain_math, identical to what the engine resolves)
    assert (rt["window_px"], rt["head_px"], rt["tail_px"]) == (169, 25, 24)
    assert (rt["n_head_v"], rt["n_tail_v"]) == (4, 3)
    assert (rt["n_head_a"], rt["n_tail_a"]) == (26, 24)
    assert rt["free_middle_px"] == [25, 145]
    # runtime
    assert rt["regenerate_audio"] is True
    assert rt["decoded_frames_px"] == 169
    # provenance (app side)
    assert rt["retake_video_id"] == vid
    assert rt["window_start_sec"] == 2.0
    assert rt["window_start_frame"] == 48
    assert rt["written_frames"] == 169
    assert rt["resampled"] is False
    # the mock must NOT fabricate a freeze proof
    assert "freeze_proof" not in rt

    # the deliverable is the WHOLE window: no trim, and no audio-handle sidecar
    from services import video_io
    assert video_io.frame_count(out_dir / "output.mp4") == 169
    assert meta["output"]["duration_seconds"] == pytest.approx(169 / 24.0, abs=1e-3)
    assert not list(out_dir.glob("*_audio_handle.wav"))
    # the cut window is kept next to the output as provenance
    assert (out_dir / "_retake_window.mp4").exists()
    assert video_io.frame_count(out_dir / "_retake_window.mp4") == 169


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_window_past_the_end_of_the_upload_is_422(client, tmp_path):
    vid = _upload_window_source(client, tmp_path, frames=193)
    r = _post(client, [{"num_frames": 169}],
              retake=_retake(video_id=vid, window_start_sec=6.0))
    assert r.status_code == 422, r.text
    assert "RETAKE_WINDOW_OUT_OF_RANGE" in r.text


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_keeping_the_original_audio_of_a_silent_upload_is_422(client, tmp_path):
    vid = _upload_window_source(client, tmp_path, audio=False)
    r = _post(client, [{"num_frames": 169}],
              retake=_retake(video_id=vid, regenerate_audio=False))
    assert r.status_code == 422, r.text
    assert "RETAKE_WINDOW_OUT_OF_RANGE" in r.text
    assert "no audio stream" in r.text


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_silent_upload_is_fine_when_the_audio_is_regenerated(client, tmp_path):
    vid = _upload_window_source(client, tmp_path, audio=False)
    r = _post(client, [{"num_frames": 73}], retake=_retake(video_id=vid))
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    ctx = client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text("utf-8"))
    assert meta["retake"]["upload_has_audio"] is False
    assert meta["retake"]["source_had_audio"] is False
    assert meta["retake"]["audio_frozen"] is False


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_a_30fps_upload_is_resampled_and_still_frame_exact(client, tmp_path):
    vid = _upload_window_source(client, tmp_path, frames=421, fps=30)
    r = _post(client, [{"num_frames": 169}],
              retake=_retake(video_id=vid, window_start_sec=1.0))
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    ctx = client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text("utf-8"))
    assert meta["retake"]["resampled"] is True
    assert meta["retake"]["written_frames"] == 169


def test_config_publishes_the_window_bounds(client):
    limits = client.get("/api/v1/config").json()["limits"]
    assert limits["retake_window_min_frames"] == 73
    assert limits["retake_window_max_frames"] == 169
