"""MiniMax H3 Inpainting (``POST /generate`` with ``inpaint.source_video_id``,
台帳 §1-124) — API + mock (no GPU).

Pins the app half of H3's Inpainting in one place: what ``GET /models``
publishes (``constraints.inpaint`` on H3 only; ``inpaint`` no longer among
H3's ``unsupported_features``), the request's shape (exactly one source route,
the LTX shape refused on H3 as ``reference_video``, the H3 fields refused on
LTX), the family's grid (17n+5, 24 fps), the endpoint's checks in their order
(source 404, odd size and the 256 floor, the canvas = the source rounded up to
32, mask 404 / resolution / frame count at 24 fps, the window, the silent
upload under a Strength above 0), the defaults written back, and the mock end
to end (the app's ``_inpaint_window.mp4`` / ``_inpaint_mask.mp4`` cuts,
``metadata["inpaint"]`` merged from the engine record and the provenance, the
deliverable at the source's own size).
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

#: A source that is not a multiple of 32 on either side (canvas 320x288).
SRC_W, SRC_H = 304, 272
CANVAS_W, CANVAS_H = 320, 288
#: The one length used throughout (17 * 2 + 5).
FRAMES = 39

#: The engine record's keys (mock) and the app's provenance keys, merged into
#: ``metadata["inpaint"]``.
ENGINE_KEYS = {
    "canvas", "pads", "source", "mask_rule", "chunk_frames", "chunks",
    "latent_frames", "fixed_tokens_total", "generated_tokens_total",
    "generated_per_latent_frame", "audio_strength", "source_had_audio",
    "audio_latents_encoded", "audio_output", "feather_px", "inject_clean",
    "self_check", "mask_proof", "cropped_to", "mock",
}
PROVENANCE_KEYS = {
    "source_video_id", "mask_video_id", "source_fps", "mask_fps", "resampled",
    "mask_resampled", "window_start_sec", "window_start_frame",
    "window_written_frames", "mask_written_frames", "window_has_audio",
}
_LORA = [{"name": "in-outpainting", "strength": 1.0}]


def _body(source_video_id: str = "vid", mask_video_id: str = "mask", *,
          width: int = CANVAS_W, height: int = CANVAS_H, num_frames: int = FRAMES,
          **inpaint) -> dict:
    return {
        "prompt": PROMPT,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": 24.0,
        "seed": 5,
        "inpaint": {"source_video_id": source_video_id, "mask_video_id": mask_video_id, **inpaint},
    }


def _upload(client, tmp_path, *, width=SRC_W, height=SRC_H, frames=72, fps=24,
            audio=True, pix_fmt="yuv420p", tag="src") -> str:
    src = tmp_path / f"{tag}_{width}x{height}_{frames}_{fps}_{int(audio)}_{pix_fmt}.mp4"
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


def _mask(client, tmp_path, **kw) -> str:
    kw.setdefault("audio", False)
    return _upload(client, tmp_path, tag="mask", **kw)


def _metadata(client, job_id: str) -> dict:
    ctx = client.app_context
    return json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text("utf-8"))


def _job_count(client) -> int:
    return len(client.get("/api/v1/jobs").json())


def _refused(client, body: dict, status: int, code: str) -> dict:
    """POST ``body``; expect ``status`` / ``code`` and no job made."""
    before = _job_count(client)
    r = client.post("/api/v1/generate", json=body)
    assert r.status_code == status, r.text
    error = r.json()["error"]
    assert error["code"] == code, error
    assert _job_count(client) == before
    return error


def _completed(client, body: dict) -> tuple[str, dict]:
    r = client.post("/api/v1/generate", json=body)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    return job_id, job


# --------------------------------------------------------------------------- #
# GET /models
# --------------------------------------------------------------------------- #


def test_models_publishes_constraints_inpaint_on_h3_only(three_family_client):
    h3_entry = _base(three_family_client, "MiniMaxH3")
    assert h3_entry["constraints"]["inpaint"] == {
        "audio_strength": {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5},
        "blend_dilation_stage1": {"min": 0, "max": 15, "step": 1, "default": 1},
    }
    assert "inpaint" not in h3_entry["unsupported_features"]
    # The LTX shape stays refused on H3 under its own name.
    assert "reference_video" in h3_entry["unsupported_features"]
    for ltx in ("LTX23", "LTX25"):
        entry = _base(three_family_client, ltx)
        assert "inpaint" not in entry["constraints"]
        assert "inpaint" not in entry["unsupported_features"]


# --------------------------------------------------------------------------- #
# the request's shape (no upload is looked up; the ids are dummies)
# --------------------------------------------------------------------------- #


def test_the_ltx_shape_on_h3_is_refused_as_reference_video(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    body = _body(width=384, height=384)  # a canvas the LTX shape's 128 grid accepts
    del body["inpaint"]["source_video_id"]
    body["reference_video_id"] = "vid"
    body["loras"] = _LORA
    error = _refused(three_family_client, body, 422, "FEATURE_UNSUPPORTED")
    assert "'reference_video'" in error["message"]


def test_both_source_routes_are_422_exactly_one(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    body = _body()
    body["reference_video_id"] = "vid"
    # One LoRA, so the older "reference_video_id requires at least one lora"
    # does not answer first.
    body["loras"] = _LORA
    error = _refused(three_family_client, body, 422, "VALIDATION_ERROR")
    assert "inpaint requires exactly one of reference_video_id or inpaint.source_video_id" in json.dumps(error)


def test_no_source_route_is_422_exactly_one(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    body = _body()
    del body["inpaint"]["source_video_id"]
    error = _refused(three_family_client, body, 422, "VALIDATION_ERROR")
    assert "inpaint requires exactly one of reference_video_id or inpaint.source_video_id" in json.dumps(error)


@pytest.mark.parametrize("ltx", ["LTX23", "LTX25"])
@pytest.mark.parametrize(
    "inpaint, message",
    [
        ({"source_video_id": "vid"}, "inpaint.source_video_id is only accepted by MiniMax H3"),
        ({"audio_strength": 0.5}, "inpaint.audio_strength is only accepted by MiniMax H3"),
    ],
)
def test_the_h3_fields_are_refused_on_ltx(three_family_client, ltx, inpaint, message):
    _activate(three_family_client, ltx)
    body = {
        "prompt": PROMPT,
        "width": 384,
        "height": 384,
        "num_frames": 41,
        "frame_rate": 24.0,
        "inpaint": {"mask_video_id": "mask", **inpaint},
    }
    if "source_video_id" not in inpaint:
        body["reference_video_id"] = "vid"
        body["loras"] = _LORA
    before = _job_count(three_family_client)
    r = three_family_client.post("/api/v1/generate", json=body)
    assert r.status_code == 422, r.text
    assert message in r.text
    assert _job_count(three_family_client) == before


# --------------------------------------------------------------------------- #
# the family's grid
# --------------------------------------------------------------------------- #


def test_h3_inpaint_keeps_the_17n_plus_5_grid(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = three_family_client.post("/api/v1/generate", json=_body(num_frames=41))
    assert r.status_code == 422, r.text
    assert "num_frames must be 17n+5" in r.text


def test_h3_inpaint_keeps_24_fps(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    body = _body()
    body["frame_rate"] = 30.0
    r = three_family_client.post("/api/v1/generate", json=body)
    assert r.status_code == 422, r.text
    assert "frame_rate must be 24" in r.text


# --------------------------------------------------------------------------- #
# the endpoint's checks on the source
# --------------------------------------------------------------------------- #


def test_an_unknown_source_video_is_404_inpaint_source_not_found(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    error = _refused(three_family_client, _body("no-such-video"), 404, "INPAINT_SOURCE_NOT_FOUND")
    assert "inpaint.source_video_id not found: no-such-video" in error["message"]


@needs_ffmpeg
def test_a_canvas_that_is_not_the_source_rounded_up_to_32_is_422(three_family_client, tmp_path):
    """The source's own 304x272 is refused by the schema (not a 32 multiple);
    a 32-multiple canvas other than 320x288 reaches the endpoint and is refused
    there; 320x288 itself is accepted."""
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path)
    mask = _mask(three_family_client, tmp_path)
    error = _refused(
        three_family_client, _body(vid, mask, width=SRC_W, height=SRC_H), 422, "VALIDATION_ERROR"
    )
    assert "width must be a multiple of 32" in json.dumps(error)
    error = _refused(
        three_family_client, _body(vid, mask, width=352, height=288), 422, "INPAINT_SOURCE_MISMATCH"
    )
    assert "multiple of 32" in error["message"]
    assert "canvas from the source would be 320x288, source 304x272" in error["detail"]
    _completed(three_family_client, _body(vid, mask, width=CANVAS_W, height=CANVAS_H))


@needs_ffmpeg
def test_an_odd_sized_source_is_422_source_mismatch(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    # 305 wide: yuv444p is the one pixel format that can carry it into the upload.
    vid = _upload(three_family_client, tmp_path, width=305, pix_fmt="yuv444p")
    error = _refused(three_family_client, _body(vid), 422, "INPAINT_SOURCE_MISMATCH")
    assert "305x272" in error["detail"]
    assert "must both be even" in error["detail"]


@needs_ffmpeg
def test_a_source_side_below_256_is_422_source_mismatch(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path, width=240)  # canvas would be 256x288
    error = _refused(three_family_client, _body(vid, width=256), 422, "INPAINT_SOURCE_MISMATCH")
    assert "at least 256px" in error["detail"]


# --------------------------------------------------------------------------- #
# the endpoint's checks on the mask
# --------------------------------------------------------------------------- #


@needs_ffmpeg
def test_an_unknown_mask_is_404_inpaint_mask_not_found(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path)
    error = _refused(three_family_client, _body(vid, "no-such-mask"), 404, "INPAINT_MASK_NOT_FOUND")
    assert "no-such-mask" in error["message"]


@needs_ffmpeg
def test_a_mask_of_another_size_is_422_mask_resolution_mismatch(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path)
    # The canvas size is not the source's size either: the mask is never resized.
    mask = _mask(three_family_client, tmp_path, width=CANVAS_W, height=CANVAS_H)
    error = _refused(three_family_client, _body(vid, mask), 422, "INPAINT_MASK_RESOLUTION_MISMATCH")
    assert error["detail"] == "source 304x272, mask 320x288"


@needs_ffmpeg
def test_a_mask_short_at_24fps_is_422_mask_frame_mismatch(three_family_client, tmp_path):
    """45 frames at 30 fps are 36 at 24 fps: enough by the raw count (45 >= 39),
    not after the cut's resample."""
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path)
    mask = _mask(three_family_client, tmp_path, frames=45, fps=30)
    error = _refused(three_family_client, _body(vid, mask), 422, "INPAINT_MASK_FRAME_MISMATCH")
    assert "at least num_frames" in error["message"]
    assert error["detail"] == "mask has 36 frames, request needs 39"


@needs_ffmpeg
def test_a_mask_longer_than_num_frames_is_accepted_and_cut(three_family_client, tmp_path):
    """60 frames at 30 fps (48 at 24 fps) for a 39-frame request: the app cuts
    the first 39 at 24 fps."""
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path)
    mask = _mask(three_family_client, tmp_path, frames=60, fps=30)
    job_id, _ = _completed(three_family_client, _body(vid, mask))
    out_dir = three_family_client.app_context.config.output_dir / job_id
    assert video_io.frame_count(out_dir / "_inpaint_mask.mp4") == FRAMES
    assert _metadata(three_family_client, job_id)["inpaint"]["mask_written_frames"] == FRAMES


@needs_ffmpeg
def test_a_window_past_the_source_end_is_422_window_out_of_range(three_family_client, tmp_path):
    """72 frames (3 s) cannot hold 39 frames starting at 2.0 s."""
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path)
    mask = _mask(three_family_client, tmp_path)
    _refused(
        three_family_client, _body(vid, mask, window_start_sec=2.0), 422, "INPAINT_WINDOW_OUT_OF_RANGE"
    )


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #


@needs_ffmpeg
def test_the_omitted_strength_on_a_silent_source_is_422_no_audio(three_family_client, tmp_path):
    """The written-back default 0.5 is above zero, so the source's audio is
    needed — a silent upload is refused before a job is made."""
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path, audio=False)
    mask = _mask(three_family_client, tmp_path)
    error = _refused(three_family_client, _body(vid, mask), 422, "INPAINT_VIDEO_NO_AUDIO")
    assert "has no audio stream" in error["detail"]


@needs_ffmpeg
def test_a_silent_source_at_strength_zero_is_accepted(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path, audio=False)
    mask = _mask(three_family_client, tmp_path)
    job_id, _ = _completed(three_family_client, _body(vid, mask, audio_strength=0.0))
    record = _metadata(three_family_client, job_id)["inpaint"]
    assert record["audio_strength"] == 0.0
    assert record["source_had_audio"] is False


# --------------------------------------------------------------------------- #
# the defaults written back, stage 2 ignored
# --------------------------------------------------------------------------- #


@needs_ffmpeg
def test_the_omitted_fields_are_written_back_as_the_family_defaults(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path)
    mask = _mask(three_family_client, tmp_path)
    job_id, _ = _completed(three_family_client, _body(vid, mask))
    request = _metadata(three_family_client, job_id)["request"]
    assert request["inpaint"]["blend_dilation_stage1"] == 1
    assert request["inpaint"]["audio_strength"] == 0.5
    assert request["reference_video_id"] is None


@needs_ffmpeg
def test_blend_dilation_stage2_is_ignored_not_refused(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path)
    mask = _mask(three_family_client, tmp_path)
    job_id, _ = _completed(
        three_family_client, _body(vid, mask, blend_dilation_stage1=3, blend_dilation_stage2=5)
    )
    meta = _metadata(three_family_client, job_id)
    assert meta["request"]["inpaint"]["blend_dilation_stage1"] == 3


# --------------------------------------------------------------------------- #
# the mock end to end
# --------------------------------------------------------------------------- #


@needs_ffmpeg
def test_h3_mock_inpaint_end_to_end(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path)
    mask = _mask(three_family_client, tmp_path)
    job_id, job = _completed(three_family_client, _body(vid, mask, window_start_sec=0.5))
    assert job["result"]["resolution"] == f"{SRC_W}x{SRC_H}"
    assert _metadata(three_family_client, job_id)["output"]["resolution"] == f"{SRC_W}x{SRC_H}"

    meta = _metadata(three_family_client, job_id)
    record = meta["inpaint"]
    assert ENGINE_KEYS <= record.keys(), sorted(ENGINE_KEYS - record.keys())
    assert PROVENANCE_KEYS <= record.keys(), sorted(PROVENANCE_KEYS - record.keys())
    assert record["mock"] is True
    assert record["source_video_id"] == vid
    assert record["mask_video_id"] == mask
    assert record["window_start_sec"] == 0.5
    assert record["window_start_frame"] == 12  # round(0.5 * 24)
    assert record["window_written_frames"] == FRAMES
    assert record["mask_written_frames"] == FRAMES
    assert record["window_has_audio"] is True
    # The engine record is lifted out of metadata["h3"], not duplicated there.
    assert "inpaint" not in meta.get("h3", {})

    out_dir = three_family_client.app_context.config.output_dir / job_id
    window = out_dir / "_inpaint_window.mp4"
    mask_cut = out_dir / "_inpaint_mask.mp4"
    assert video_io.frame_count(window) == FRAMES
    assert video_io.frame_count(mask_cut) == FRAMES
    assert video_io.probe_resolution(window) == (SRC_W, SRC_H)
    assert video_io.probe_resolution(mask_cut) == (SRC_W, SRC_H)
    # The deliverable is the source's own size, not the 32-multiple canvas.
    assert video_io.probe_resolution(out_dir / "output.mp4") == (SRC_W, SRC_H)


@needs_ffmpeg
def test_h3_mock_inpaint_pads_the_bottom_and_delivers_the_source_size(three_family_client, tmp_path):
    """The 1080p case in small: a height that is not a multiple of 32 (264 ->
    canvas 288, 24 rows mirrored at the bottom and cropped off again)."""
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload(three_family_client, tmp_path, width=320, height=264)
    mask = _mask(three_family_client, tmp_path, width=320, height=264)
    job_id, job = _completed(three_family_client, _body(vid, mask, width=320, height=288))
    assert job["result"]["resolution"] == "320x264"
    record = _metadata(three_family_client, job_id)["inpaint"]
    assert record["pads"]["bottom"] == 24
    assert record["pads"]["right"] == 0
    out_dir = three_family_client.app_context.config.output_dir / job_id
    assert video_io.probe_resolution(out_dir / "output.mp4") == (320, 264)
