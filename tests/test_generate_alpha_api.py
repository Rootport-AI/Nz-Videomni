"""Alpha Gen (台帳 §1-83 第 1 弾) — app-layer tests for ``POST /generate/alpha``,
``GET /jobs/{id}/matte`` and ``JobResponse.matte``.

Mock backend, no GPU. The endpoint measures the uploaded source with ffprobe
and the job cuts / pads / finalizes with ffmpeg, so most tests need REAL video
files and skip without ffmpeg on PATH (put ``tools\\ffmpeg\\bin`` first).

The fixture is ``two_family_client``'s world (LTX 2.3 + LTX 2.5, mock) plus
two things that world lacks: the ``alpha-gen`` adapter registered in
``model.ic_loras``, and a ``comfort`` block on the LTX 2.5 descriptor so the
budget path (row selection -> shrink) is exercised. The stub transformer has
no tensors, so its weight class is unknown and the endpoint takes the
SMALLEST value across the rows (the documented exception).
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import main
from api.models import GenerateRequest
from conftest import (
    _make_args,
    base_model_descriptor,
    build_model_layout,
    write_gguf_with_kv,
    write_model_file,
)
from services import video_io

ALPHA_LORA = "alpha-gen"
FPS = 24
FRAMES = 9

# Smallest row wins (unknown weight class): alpha 400, single 100000.
# 640x512x9f -> canvas tokens 20*16*2 = 640 > 400 -> working 480x384,
# canvas 512x384 (16*12*2 = 384). Light mode (single 100000) -> no shrink.
COMFORT = {
    "spatial_factor": 32,
    "temporal_factor": 8,
    "rows": [
        {"requires": {"weight_class": "4bit"}, "single_budget": 100000,
         "chain_budget": 100000, "alpha_gen_budget": 900},
        {"requires": {"weight_class": "8bit"}, "single_budget": 200000,
         "chain_budget": 200000, "alpha_gen_budget": 400},
    ],
}

has_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="Alpha Gen measures the source and builds/finalizes with ffmpeg",
)


def _control_safetensors(path: Path, factor: str = "1") -> None:
    """Minimal safetensors whose header declares ``reference_downscale_factor``
    (control kind, preprocess none) — copied from tests/test_inpaint_api.py."""
    header = {
        "__metadata__": {"reference_downscale_factor": factor},
        "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": {
            "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
        },
    }
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 4)


def _build(tmp_path: Path, *, register_lora: bool = True):
    ltx23 = base_model_descriptor()
    ltx25 = base_model_descriptor("LTX25")
    ltx25["display_name"] = "LTX 2.5"
    ltx25["engine_family"] = "ltx25"
    ltx25["comfort"] = COMFORT

    fragment = build_model_layout(tmp_path, [ltx23, ltx25])
    models_dir = tmp_path / "models"
    for category, spec in ltx25["categories"].items():
        target = models_dir / spec["default_file"]
        if category == "transformer":
            write_gguf_with_kv(target, **{"general.architecture": "ltxv", "model_version": "2.5.0"})
        else:
            write_model_file(target)
    write_gguf_with_kv(
        models_dir / ltx23["categories"]["transformer"]["default_file"],
        **{"general.architecture": "ltxv", "model_version": "2.3.0"},
    )

    model = {"backend": "mock", **fragment}
    if register_lora:
        lora = tmp_path / "alpha-gen.safetensors"
        _control_safetensors(lora)
        model["ic_loras"] = {ALPHA_LORA: lora.as_posix()}

    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": model,
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
        "state_file": (tmp_path / "state.json").as_posix(),
        "tracking": {"backend": "mock"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return main.build_app(_make_args(cfg_path.as_posix()))


def _activate_25(client) -> None:
    r = client.post("/api/v1/pipeline/load", json={"base_model": "LTX25"})
    assert r.status_code == 200, r.text
    assert client.app_context.pipeline_manager.active_engine_family == "ltx25"


@pytest.fixture()
def alpha_client(tmp_path):
    app = _build(tmp_path)
    with TestClient(app) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        c.tmp_path = tmp_path  # type: ignore[attr-defined]
        _activate_25(c)
        yield c


@pytest.fixture()
def alpha_client_no_lora(tmp_path):
    app = _build(tmp_path, register_lora=False)
    with TestClient(app) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        c.tmp_path = tmp_path  # type: ignore[attr-defined]
        _activate_25(c)
        yield c


def _write_source(path: Path, width: int, height: int, frames: int = FRAMES, fps: int = FPS) -> None:
    # FFV1 in Matroska: also takes ODD sizes (libx264 yuv420p would not).
    subprocess.run(
        [
            shutil.which("ffmpeg"), "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={frames / fps + 1.0:.3f}",
            "-frames:v", str(frames),
            "-c:v", "ffv1", "-pix_fmt", "yuv444p",
            str(path),
        ],
        check=True, capture_output=True,
    )


def _upload_source(client, width: int, height: int, frames: int = FRAMES) -> str:
    src = client.tmp_path / f"src_{width}x{height}_{frames}.mkv"
    if not src.exists():
        _write_source(src, width, height, frames)
    with src.open("rb") as fh:
        resp = client.post(
            "/api/v1/upload/video", files={"file": (src.name, fh, "video/x-matroska")}
        )
    assert resp.status_code == 200, resp.text
    return resp.json()["video_id"]


def _body(video_id: str, **overrides) -> dict:
    body = {"reference_video_id": video_id, "num_frames": FRAMES, "frame_rate": FPS, "seed": 7}
    body.update(overrides)
    return body


def _job_count(client) -> int:
    return len(client.get("/api/v1/jobs").json())


def _occupy(client):
    """Reserve the single job slot without running anything (queued job)."""
    job = client.app_context.job_store.create_if_idle(GenerateRequest(prompt="busy"))
    assert job is not None
    return job


# ─────────────────────────── engine scope (LTX 2.3) ─────────────────────────


def test_ltx23_refuses_before_looking_at_the_upload(two_family_client):
    """(a) comes first: the id does not even exist, and the answer is still
    FEATURE_UNSUPPORTED with this endpoint's wording, not a 404."""
    before = _job_count(two_family_client)
    r = two_family_client.post(
        "/api/v1/generate/alpha", json=_body("no-such-upload")
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert "LTX 2.5" in error["detail"]
    assert _job_count(two_family_client) == before


def test_generate_with_alpha_gen_block_is_refused(alpha_client):
    body = {
        "prompt": " ",
        "width": 320,
        "height": 256,
        "num_frames": FRAMES,
        "reference_video_id": "whatever",
        "loras": [{"name": ALPHA_LORA, "strength": 1.0}],
        "alpha_gen": {"working_width": 320, "working_height": 256},
    }
    r = alpha_client.post("/api/v1/generate", json=body)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "ALPHA_GEN_INVALID"
    assert "POST /generate/alpha" in r.json()["error"]["detail"]


# ──────────────────────────────── body shape ────────────────────────────────


@pytest.mark.parametrize(
    "override",
    [{"num_frames": 10}, {"num_frames": 153}, {"frame_rate": 23.5}],
    ids=["not-8n+1", "over-145", "fractional-fps"],
)
def test_body_shape_is_422(alpha_client, override):
    r = alpha_client.post("/api/v1/generate/alpha", json=_body("vid", **override))
    assert r.status_code == 422, r.text


# ──────────────────────────────────── 404 ───────────────────────────────────


def test_missing_reference_is_404(alpha_client):
    r = alpha_client.post("/api/v1/generate/alpha", json=_body("no-such-upload"))
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "REFERENCE_VIDEO_NOT_FOUND"


@has_ffmpeg
def test_unregistered_adapter_is_404(alpha_client_no_lora):
    vid = _upload_source(alpha_client_no_lora, 320, 256)
    r = alpha_client_no_lora.post("/api/v1/generate/alpha", json=_body(vid))
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "LORA_NOT_FOUND"


# ─────────────────────────── 422 ALPHA_GEN_INVALID ──────────────────────────


@has_ffmpeg
@pytest.mark.parametrize(
    "size,needle",
    [((321, 256), "even"), ((320, 240), "at least 256")],
    ids=["odd-side", "below-256"],
)
def test_source_size_is_checked(alpha_client, size, needle):
    vid = _upload_source(alpha_client, *size)
    before = _job_count(alpha_client)
    r = alpha_client.post("/api/v1/generate/alpha", json=_body(vid))
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "ALPHA_GEN_INVALID"
    assert needle in r.json()["error"]["detail"]
    assert _job_count(alpha_client) == before


@has_ffmpeg
def test_window_out_of_range(alpha_client):
    vid = _upload_source(alpha_client, 320, 256)
    r = alpha_client.post("/api/v1/generate/alpha", json=_body(vid, window_start_sec=1.0))
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "ALPHA_GEN_INVALID"
    assert "window starts at frame 24" in r.json()["error"]["detail"]


@has_ffmpeg
def test_working_size_outside_request_bounds(alpha_client):
    """256x256 x 145f (19 latents): canvas tokens 8*8*19 = 1216 > 400, and the
    first fitting working size is 128x128 (16*19 = 304) -- below
    GenerateRequest's 256 width floor -> 422 rather than a 500 from the
    internal request."""
    vid = _upload_source(alpha_client, 256, 256, frames=145)
    r = alpha_client.post("/api/v1/generate/alpha", json=_body(vid, num_frames=145))
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "ALPHA_GEN_INVALID"
    assert "outside" in r.json()["error"]["detail"]


# ──────────────────────────────────── 409 ───────────────────────────────────


@has_ffmpeg
def test_job_busy_is_409(alpha_client):
    vid = _upload_source(alpha_client, 320, 256)
    _occupy(alpha_client)
    r = alpha_client.post("/api/v1/generate/alpha", json=_body(vid))
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "JOB_BUSY"


# ───────────────────────────── accepted + mock run ──────────────────────────


def _probe(path: Path):
    return video_io.probe_resolution(path), video_io.frame_count(path)


@has_ffmpeg
def test_accepted_builds_internal_request_and_finishes(alpha_client):
    """640x512 -> working 480x384 / canvas 512x384 under the smallest alpha
    budget (400). The job runs to completion on the mock backend; both
    deliverables come back at the SOURCE size."""
    vid = _upload_source(alpha_client, 640, 512)
    r = alpha_client.post(
        "/api/v1/generate/alpha",
        json=_body(
            vid,
            attention_backend="sage",
            block_swap_prefetch=False,
            fused_gguf_dequant_kernel=False,
            keep_resident=True,
            keep_resident_embeddings=True,
        ),
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert set(r.json()) >= {"job_id", "status", "created_at"}

    job = alpha_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    req = job["request"]
    assert req["prompt"] == " "
    assert req["loras"] == [{"name": ALPHA_LORA, "strength": 1.0, "audio_strength": None}]
    assert req["reference_video_id"] == vid
    assert (req["width"], req["height"]) == (512, 384)
    assert req["num_frames"] == FRAMES
    assert req["frame_rate"] == FPS
    assert req["seed"] == 7
    assert req["alpha_gen"] == {
        "window_start_sec": 0.0,
        "one_stage": True,
        "working_width": 480,
        "working_height": 384,
        "budget_tokens": 400,
    }
    # pass-through, verbatim
    assert req["attention_backend"] == "sage"
    assert req["block_swap_prefetch"] is False
    assert req["fused_gguf_dequant_kernel"] is False
    assert req["keep_resident"] is True
    assert req["keep_resident_embeddings"] is True

    out = alpha_client.tmp_path / "outputs" / job_id
    assert _probe(out / "output.mp4") == ((640, 512), FRAMES)
    assert _probe(out / "matte.mkv") == ((640, 512), FRAMES)
    assert not (out / "_alpha_engine.mp4").exists()
    assert (out / "_alpha_window.mp4").exists()
    assert _probe(out / "alpha_reference.mp4") == ((512, 384), FRAMES)

    assert job["matte"] is True
    assert job["result"]["resolution"] == "640x512"

    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    ag = meta["alpha_gen"]
    assert set(ag) == {
        "mode", "source_video_id", "source_width", "source_height", "source_fps",
        "resampled", "window_start_sec", "window_start_frame",
        "window_written_frames", "working_width", "working_height",
        "canvas_width", "canvas_height", "pad_right", "pad_bottom", "scale",
        "budget_tokens", "canvas_tokens", "matte", "matte_codec",
    }
    assert ag["mode"] == "one_stage"
    assert (ag["source_width"], ag["source_height"]) == (640, 512)
    assert (ag["pad_right"], ag["pad_bottom"]) == (32, 0)
    assert ag["scale"] == 0.75
    assert ag["canvas_tokens"] == 384
    assert ag["matte"] == "matte.mkv"
    assert ag["matte_codec"] == "ffv1/gray"

    m = alpha_client.get(f"/api/v1/jobs/{job_id}/matte")
    assert m.status_code == 200
    assert m.headers["content-type"] == "video/x-matroska"
    assert f"{job_id}_matte.mkv" in m.headers["content-disposition"]


@has_ffmpeg
def test_light_mode_uses_single_budget(alpha_client):
    vid = _upload_source(alpha_client, 640, 512)
    r = alpha_client.post("/api/v1/generate/alpha", json=_body(vid, light_mode=True))
    assert r.status_code == 202, r.text
    job = alpha_client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "completed", job
    ag = job["request"]["alpha_gen"]
    assert ag["one_stage"] is False
    assert (ag["working_width"], ag["working_height"]) == (640, 512)
    assert ag["budget_tokens"] == 100000
    meta_path = alpha_client.tmp_path / "outputs" / job["job_id"] / "metadata.json"
    assert json.loads(meta_path.read_text(encoding="utf-8"))["alpha_gen"]["mode"] == "light"


@has_ffmpeg
@pytest.mark.parametrize(
    "weight_class,light_mode,budget",
    [("4bit", False, 900), ("4bit", True, 100000), ("8bit", True, 200000)],
    ids=["4bit-alpha", "4bit-light", "8bit-light"],
)
def test_known_weight_class_picks_its_row(alpha_client, monkeypatch, weight_class, light_mode, budget):
    """With the weight class known, the matching row is used (not the
    smallest-value fallback): ``alpha_gen_budget`` by default,
    ``single_budget`` in light mode. 640x512x9f is 640 tokens, under every
    value here, so the working size stays the source size. (4bit-light equals
    the fallback's value; 8bit-light is the case that tells them apart.)"""
    import api.generate_alpha as generate_alpha

    monkeypatch.setattr(generate_alpha, "transformer_weight_class", lambda *_a: weight_class)
    vid = _upload_source(alpha_client, 640, 512)
    r = alpha_client.post("/api/v1/generate/alpha", json=_body(vid, light_mode=light_mode))
    assert r.status_code == 202, r.text
    job = alpha_client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    ag = job["request"]["alpha_gen"]
    assert ag["budget_tokens"] == budget
    assert (ag["working_width"], ag["working_height"]) == (640, 512)


# ───────────────────────────── GET /jobs/{id}/matte ─────────────────────────


def test_matte_unknown_job_is_404(client):
    r = client.get("/api/v1/jobs/no-such-job/matte")
    assert r.status_code == 404, r.text


def test_matte_not_ready_is_409(client):
    job = _occupy(client)
    r = client.get(f"/api/v1/jobs/{job.job_id}/matte")
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "VIDEO_NOT_READY"


def test_ordinary_job_has_no_matte(client):
    r = client.post(
        "/api/v1/generate",
        json={"prompt": "a red ball", "width": 320, "height": 256, "num_frames": FRAMES},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed"
    assert job["matte"] is False
    assert client.get(f"/api/v1/jobs/{job_id}/matte").status_code == 404
