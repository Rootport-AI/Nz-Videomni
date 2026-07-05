"""V2V server-side join (POST /jobs/{id}/join + GET /jobs/{id}/joined) — unit
(normalize_clip) + API contract via the mock backend (no GPU).

The mock V2V job renders a no-audio continuation at the request resolution while
the uploaded synthetic source is a different (tiny) resolution — so the happy
path exercises the R3 normalization pass (scale + center-crop + setsar=1 + fps)
for real, through real ffmpeg. Audio-mode specifics (fade_pair vs
handle_crossfade vs hard concat energy) are covered by tests/test_video_io.py;
here we pin the REST contract: status codes, error codes, response fields, and
that joined.mp4 lands next to output.mp4 with source+continuation frames.
"""

from __future__ import annotations

import json

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


def _make_source_mp4(path, n_frames, fps, size=(96, 64)):
    frames = [Image.new("RGB", size, (i * 4 % 256, 90, 160)) for i in range(n_frames)]
    video_io.encode_frames_to_mp4(frames, path, frame_rate=fps)
    return path


def _run_v2v_job(client, tmp_path, *, n_src=50, src_fps=24.0, context_frames=25,
                 clip_frames=49) -> tuple[str, str]:
    """Upload a synthetic source + run a 1-clip mock V2V chain to completion.
    Returns (job_id, video_id)."""
    src = _make_source_mp4(tmp_path / "join_src.mp4", n_frames=n_src, fps=src_fps)
    r = client.post("/api/v1/upload/video",
                    files={"file": ("src.mp4", src.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    vid = r.json()["video_id"]

    payload = {
        **BASE,
        "clips": [{"num_frames": clip_frames}],
        "source_video": {"video_id": vid, "context_frames": context_frames},
    }
    r = client.post("/api/v1/generate/chain", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    return job_id, vid


# ----------------------------------------------------------- unit: normalize_clip


def test_normalize_clip_matches_target_geometry(tmp_path):
    src = _make_source_mp4(tmp_path / "n.mp4", n_frames=30, fps=30.0, size=(96, 64))
    out = tmp_path / "norm.mp4"
    video_io.normalize_clip(src, out, 128, 64, 24.0)
    assert video_io.probe_resolution(out) == (128, 64)
    assert abs(video_io.probe_fps(out) - 24.0) < 1e-3


def test_normalize_clip_output_joins_cleanly(tmp_path):
    """R3 pin: a resolution+fps-mismatched source fails join_v2v, and the
    normalized re-encode of the SAME source succeeds (incl. the setsar=1 fix —
    without it ffmpeg's concat rejects the scaled pair on SAR mismatch)."""
    src = _make_source_mp4(tmp_path / "s.mp4", n_frames=30, fps=30.0, size=(96, 64))
    cont = _make_source_mp4(tmp_path / "c.mp4", n_frames=24, fps=24.0, size=(128, 64))

    import pytest

    with pytest.raises(video_io.FFmpegError, match="mismatch"):
        video_io.join_v2v(src, cont, tmp_path / "bad.mp4")

    norm = tmp_path / "s_norm.mp4"
    video_io.normalize_clip(src, norm, 128, 64, 24.0)
    out = tmp_path / "joined.mp4"
    info = video_io.join_v2v(norm, cont, out)
    assert out.exists()
    assert info["join_mode"] == "video_only"  # synthetic clips carry no audio


# ------------------------------------------------------------------ happy path


def test_join_happy_path_normalizes_and_writes_joined(client, tmp_path):
    job_id, _vid = _run_v2v_job(client, tmp_path)

    r = client.post(f"/api/v1/jobs/{job_id}/join", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == job_id
    assert body["joined_path"] == f"outputs/{job_id}/joined.mp4"
    # source (96x64) != mock output (384x256) -> the R3 normalization ran.
    assert body["source_normalized"] is True
    # mock continuation has no audio stream -> video-only join.
    assert body["join_mode"] == "video_only"

    ctx = client.app_context
    job_dir = ctx.config.output_dir / job_id
    joined = job_dir / "joined.mp4"
    assert joined.exists() and joined.stat().st_size > 0
    # full source (50 frames) + NEW continuation part (new_frames_px).
    layout = chain_math.compute_chain_layout([49], 24.0, kv=2, source_context_px=25)
    assert video_io.frame_count(joined) == 50 + layout.new_frames_px
    # output.mp4 untouched (additive), norm temp cleaned up.
    assert (job_dir / "output.mp4").exists()
    assert not (job_dir / "_join_source_norm.mp4").exists()


def test_join_empty_body_and_get_joined_roundtrip(client, tmp_path):
    job_id, _vid = _run_v2v_job(client, tmp_path)

    # GET before any join -> 404 JOINED_NOT_READY
    r = client.get(f"/api/v1/jobs/{job_id}/joined")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "JOINED_NOT_READY"

    # body is optional (no JSON at all)
    r = client.post(f"/api/v1/jobs/{job_id}/join")
    assert r.status_code == 200, r.text

    r = client.get(f"/api/v1/jobs/{job_id}/joined")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert len(r.content) > 0


def test_join_hard_concat_mode(client, tmp_path):
    job_id, _vid = _run_v2v_job(client, tmp_path)
    r = client.post(f"/api/v1/jobs/{job_id}/join", json={"audio_smoothing": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["join_mode"] == "hard_concat"
    assert body["loudness_matched"] is False

    ctx = client.app_context
    layout = chain_math.compute_chain_layout([49], 24.0, kv=2, source_context_px=25)
    joined = ctx.config.output_dir / job_id / "joined.mp4"
    assert video_io.frame_count(joined) == 50 + layout.new_frames_px


# -------------------------------------------------------------------- negatives


def test_join_unknown_job_404(client):
    r = client.post("/api/v1/jobs/no-such-job/join", json={})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "JOB_NOT_FOUND"


def test_join_non_v2v_job_422(client):
    """A plain (no-source) chain job has no metadata v2v block -> JOB_NOT_JOINABLE."""
    r = client.post("/api/v1/generate/chain",
                    json={**BASE, "clips": [{"num_frames": 25}, {"num_frames": 25}]})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    r = client.post(f"/api/v1/jobs/{job_id}/join", json={})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "JOB_NOT_JOINABLE"


def test_join_not_completed_job_409(client):
    """A queued (never-run) job is not joinable yet -> VIDEO_NOT_READY."""
    from api.models import GenerateRequest

    ctx = client.app_context
    record = ctx.job_store.create(GenerateRequest(prompt="pending"))
    r = client.post(f"/api/v1/jobs/{record.job_id}/join", json={})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "VIDEO_NOT_READY"


def test_join_purged_source_404(client, tmp_path):
    import shutil

    job_id, vid = _run_v2v_job(client, tmp_path)
    ctx = client.app_context
    shutil.rmtree(ctx.video_upload_store.video_dir / vid)

    r = client.post(f"/api/v1/jobs/{job_id}/join", json={})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "SOURCE_VIDEO_NOT_FOUND"


def test_join_bad_crossfade_ms_422(client, tmp_path):
    job_id, _vid = _run_v2v_job(client, tmp_path)
    r = client.post(f"/api/v1/jobs/{job_id}/join", json={"handle_crossfade_ms": -1})
    assert r.status_code == 422
    r = client.post(f"/api/v1/jobs/{job_id}/join", json={"handle_crossfade_ms": 9999})
    assert r.status_code == 422


# -------------------------------------------------- regression: /video untouched


def test_join_does_not_change_video_endpoint(client, tmp_path):
    job_id, _vid = _run_v2v_job(client, tmp_path)
    before = client.get(f"/api/v1/jobs/{job_id}/video").content
    r = client.post(f"/api/v1/jobs/{job_id}/join", json={})
    assert r.status_code == 200
    after = client.get(f"/api/v1/jobs/{job_id}/video").content
    assert before == after  # byte-identical output.mp4 delivery

    # metadata.json also untouched by the join
    ctx = client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    assert meta["v2v"]["source_video_id"] == _vid
