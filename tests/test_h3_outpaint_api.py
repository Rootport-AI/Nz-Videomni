"""MiniMax H3 Outpainting (``POST /generate`` with ``outpaint.source_video_id``,
台帳 §1-125) — API + mock (no GPU).

Pins the app half of H3's Outpainting in one place: what ``GET /models``
publishes (``constraints.outpaint`` on H3 only; ``outpaint`` no longer among
H3's ``unsupported_features``), the family's rules (the published defaults
written back, an explicit ``freeze_source_audio`` and the A2V / Ref2VA
combinations refused, the LTX shape refused as ``reference_video``), the
endpoint's checks on the source (its own 404, an even size, the resolution
against the keep rectangle, the frame count at the request's 24 fps, the
silent-upload preflight for a Strength above 0), and the mock end to end (the
app's ``_outpaint_source.mp4`` cut, ``metadata["h3"]["outpaint"]``, the
canvas-size output).
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from services import video_io
from test_engine_h3_adapter import PROMPT, _activate, _base

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")

#: Canvas 640x384 (32 multiples), keep 512x256 (the schema's 256 floor).
SRC_W, SRC_H = 512, 256
PADS = {"pad_left": 64, "pad_right": 64, "pad_top": 64, "pad_bottom": 64}
#: The one length used throughout (17 * 2 + 5).
FRAMES = 39


def _body(source_video_id: str = "vid", num_frames: int = FRAMES, **outpaint) -> dict:
    return {
        "prompt": PROMPT,
        "width": 640,
        "height": 384,
        "num_frames": num_frames,
        "frame_rate": 24.0,
        "seed": 5,
        "outpaint": {"source_video_id": source_video_id, **PADS, **outpaint},
    }


def _upload(client, tmp_path, *, width=SRC_W, height=SRC_H, frames=48, fps=24,
            audio=True, pix_fmt="yuv420p") -> str:
    src = tmp_path / f"op_{width}x{height}_{frames}_{fps}_{int(audio)}.mp4"
    cmd = [shutil.which("ffmpeg"), "-y", "-v", "error",
           "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}:duration=4"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4"]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", pix_fmt,
            "-fps_mode", "cfr", "-r", str(fps)]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(src)]
    subprocess.run(cmd, check=True, capture_output=True)
    r = client.post("/api/v1/upload/video", files={"file": ("src.mp4", src.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    return r.json()["video_id"]


def _metadata(client, job_id: str) -> dict:
    ctx = client.app_context
    return json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text("utf-8"))


def _job_count(client) -> int:
    return len(client.get("/api/v1/jobs").json())


# --------------------------------------------------------------------------- #
# GET /models
# --------------------------------------------------------------------------- #


def test_models_publishes_constraints_outpaint_on_h3_only(three_family_client):
    h3_entry = _base(three_family_client, "MiniMaxH3")
    assert h3_entry["constraints"]["outpaint"] == {
        "audio_strength": {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5},
        "blend_dilation_stage1": {"min": 0, "max": 15, "step": 1, "default": 1},
    }
    assert "outpaint" not in h3_entry["unsupported_features"]
    # The LTX shape stays refused on H3 under its own name.
    assert "reference_video" in h3_entry["unsupported_features"]
    for ltx in ("LTX23", "LTX25"):
        entry = _base(three_family_client, ltx)
        assert "outpaint" not in entry["constraints"]
        assert "outpaint" not in entry["unsupported_features"]


# --------------------------------------------------------------------------- #
# H3's 422s before any upload is looked up (the source id is a dummy)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"outpaint": {"freeze_source_audio": True}}, "outpaint.freeze_source_audio is not accepted by MiniMax H3"),
        ({"outpaint": {"freeze_source_audio": False}}, "outpaint.freeze_source_audio is not accepted by MiniMax H3"),
        ({"source_audio": {"audio_id": "aud"}}, "outpaint and source_audio are mutually exclusive on MiniMax H3"),
        (
            {"references": [{"kind": "picture", "id": "img", "slot": 1}]},
            "outpaint and references are mutually exclusive on MiniMax H3",
        ),
    ],
)
def test_h3_outpaint_family_422s(three_family_client, extra, message):
    _activate(three_family_client, "MiniMaxH3")
    before = _job_count(three_family_client)
    body = _body()
    body["outpaint"].update(extra.pop("outpaint", {}))
    body.update(extra)
    r = three_family_client.post("/api/v1/generate", json=body)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert message in r.text
    assert _job_count(three_family_client) == before


def test_the_ltx_shape_on_h3_is_refused_as_reference_video(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    body = _body()
    del body["outpaint"]["source_video_id"]
    body["reference_video_id"] = "vid"
    body["loras"] = [{"name": "in-outpainting", "strength": 1.0}]
    r = three_family_client.post("/api/v1/generate", json=body)
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert "'reference_video'" in error["message"]


def test_h3_outpaint_keeps_the_family_grid(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = three_family_client.post("/api/v1/generate", json=_body(num_frames=41))
    assert r.status_code == 422, r.text
    assert "num_frames must be 17n+5" in r.text


def test_an_unknown_source_video_is_404_outpaint_source_not_found(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = three_family_client.post("/api/v1/generate", json=_body("no-such-video"))
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "OUTPAINT_SOURCE_NOT_FOUND"
    assert "outpaint.source_video_id not found: no-such-video" in r.text


# --------------------------------------------------------------------------- #
# the endpoint's checks on the source file
# --------------------------------------------------------------------------- #


@needs_ffmpeg
def test_an_odd_sized_source_is_422_odd_size(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    # 513 wide: yuv444p is the one pixel format that can carry it into the upload.
    vid = _upload(three_family_client, tmp_path, width=513, pix_fmt="yuv444p")
    body = _body(vid, pad_left=64, pad_right=63)
    before = _job_count(three_family_client)
    r = three_family_client.post("/api/v1/generate", json=body)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "OUTPAINT_SOURCE_ODD_SIZE"
    assert "513x256" in r.json()["error"]["detail"]
    assert _job_count(three_family_client) == before


@needs_ffmpeg
def test_a_source_that_is_not_the_keep_region_is_422_mismatch(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path, width=448)
    r = three_family_client.post("/api/v1/generate", json=_body(vid))
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "OUTPAINT_SOURCE_MISMATCH"


@needs_ffmpeg
def test_the_frame_count_is_compared_at_the_request_fps(three_family_client, tmp_path):
    """45 frames at 30 fps are 36 at 24 fps: enough by the raw count (45 >= 39),
    not after the cut's resample — refused before a job is made."""
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path, frames=45, fps=30)
    r = three_family_client.post("/api/v1/generate", json=_body(vid))
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "OUTPAINT_SOURCE_TOO_SHORT"
    assert "has 36 frames, request needs 39" in error["detail"]


@needs_ffmpeg
def test_a_30fps_source_long_enough_at_24fps_is_accepted(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path, frames=60, fps=30)  # 48 at 24 fps
    r = three_family_client.post("/api/v1/generate", json=_body(vid))
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert three_family_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    out_dir = three_family_client.app_context.config.output_dir / job_id
    assert video_io.frame_count(out_dir / "_outpaint_source.mp4") == FRAMES


@needs_ffmpeg
def test_the_omitted_strength_on_a_silent_source_is_422_no_audio(three_family_client, tmp_path):
    """The written-back default 0.5 is above zero, so the source's audio is
    needed — a silent upload is refused before a job is made."""
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path, audio=False)
    before = _job_count(three_family_client)
    r = three_family_client.post("/api/v1/generate", json=_body(vid))
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "OUTPAINT_VIDEO_NO_AUDIO"
    assert "has no audio stream" in error["detail"]
    assert _job_count(three_family_client) == before


# --------------------------------------------------------------------------- #
# the mock end to end
# --------------------------------------------------------------------------- #


@needs_ffmpeg
def test_h3_mock_outpaint_writes_the_defaults_back_and_the_record(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path)
    r = three_family_client.post("/api/v1/generate", json=_body(vid))
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = three_family_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    meta = _metadata(three_family_client, job_id)

    # the omitted fields, as validate_request wrote them back
    request = meta["request"]
    assert request["outpaint"]["blend_dilation_stage1"] == 1
    assert request["outpaint"]["audio_strength"] == 0.5
    assert request["reference_video_id"] is None

    # the engine-side record, inside metadata["h3"] (not lifted to the top)
    record = meta["h3"]["outpaint"]
    assert record == {
        "canvas": [640, 384],
        "pads": {"left": 64, "right": 64, "top": 64, "bottom": 64},
        "source": [SRC_W, SRC_H],
        "aligned": None,
        "fixed_tokens": None,
        "generated_tokens": None,
        "feather_px": 10,
        "audio_strength": 0.5,
        "source_had_audio": True,
        "audio_latents_encoded": None,
        "audio_output": "decoded",
        "inject_clean": None,
        "mock": True,
    }
    assert "outpaint" not in meta

    # the app's cut (24 fps, exactly num_frames) and no green canvas
    out_dir = three_family_client.app_context.config.output_dir / job_id
    cut = out_dir / "_outpaint_source.mp4"
    assert video_io.frame_count(cut) == FRAMES
    assert video_io.probe_resolution(cut) == (SRC_W, SRC_H)
    assert not (out_dir / "outpaint_canvas.mp4").exists()
    # the deliverable is the canvas
    assert video_io.probe_resolution(out_dir / "output.mp4") == (640, 384)


@needs_ffmpeg
def test_h3_mock_outpaint_silent_source_at_strength_zero(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path, audio=False)
    r = three_family_client.post(
        "/api/v1/generate", json=_body(vid, audio_strength=0.0, blend_dilation_stage1=3)
    )
    assert r.status_code == 202, r.text
    meta = _metadata(three_family_client, r.json()["job_id"])
    record = meta["h3"]["outpaint"]
    assert record["source_had_audio"] is False
    assert record["audio_output"] == "generated"
    assert record["feather_px"] == 30  # floor(3 * 640 / 64 + 0.5)
    assert meta["request"]["outpaint"]["blend_dilation_stage1"] == 3


@needs_ffmpeg
def test_h3_mock_outpaint_full_strength_records_the_original_audio(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path)
    r = three_family_client.post("/api/v1/generate", json=_body(vid, audio_strength=1.0))
    assert r.status_code == 202, r.text
    meta = _metadata(three_family_client, r.json()["job_id"])
    assert meta["h3"]["outpaint"]["audio_output"] == "original"
    assert meta["h3"]["outpaint"]["audio_strength"] == 1.0
