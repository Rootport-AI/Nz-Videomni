"""Outpainting (§1-13) — app-layer tests: schema, endpoint guards, runner payload.

Mock backend, no GPU. The engine's two-stage driver
(``engine/pipeline/outpaint_pipeline.py``) and the pure blend parts are tested
separately; this file covers the plumbing from ``POST /generate`` down to the
worker payload, plus the green-canvas ffmpeg helper.

Several tests need a REAL video file rather than the ``FAKE_MP4`` blob the other
IC-LoRA tests use, because outpainting's endpoint guards deliberately ffprobe the
reference video: its resolution has to equal the keep rectangle and its frame
count has to cover ``num_frames``. Those tests skip without ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess

import pytest
import yaml
from fastapi.testclient import TestClient

import main
from services import video_io

OUTPAINT_LORA = "in-outpainting"
CANNY_LORA = "canny-control"

# Keep region 512x256, canvas 768x384 -> pads 128/128 left/right, 64/64 top/bottom.
# 512x256 clears OUTPAINT_MIN_KEEP_SIDE (256) on both sides, and both canvas
# dimensions are multiples of 128 as the reference-video rule requires.
SRC_W, SRC_H = 512, 256
CANVAS_W, CANVAS_H = 768, 384
PADS = {"pad_left": 128, "pad_right": 128, "pad_top": 64, "pad_bottom": 64}
SRC_FRAMES = 33

has_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="outpainting's endpoint guards and canvas builder both shell out to ffmpeg",
)


def _control_safetensors(path, factor: str = "1") -> None:
    """Write a minimal safetensors file whose header declares
    ``reference_downscale_factor`` — which is exactly what makes
    ``services.lora_registry`` classify the adapter as CONTROL kind with
    ``preprocess: none``, the combination outpainting requires."""
    header = {
        "__metadata__": {"reference_downscale_factor": factor},
        "dummy": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 4)


def _make_args(config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=config_path, te_offload=None, dit_cpu_load=None,
    )


def _write_test_mp4(path, width: int, height: int, frames: int, fps: int = 24) -> None:
    subprocess.run(
        [
            shutil.which("ffmpeg"), "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={frames / fps + 0.5:.3f}",
            "-frames:v", str(frames),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True, capture_output=True,
    )


@pytest.fixture()
def outpaint_client(tmp_path):
    control = tmp_path / "in-outpainting.safetensors"
    _control_safetensors(control)
    canny = tmp_path / "canny.safetensors"
    _control_safetensors(canny)

    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {
            "backend": "mock",
            "ic_loras": {
                OUTPAINT_LORA: control.as_posix(),
                CANNY_LORA: {"path": canny.as_posix(), "preprocess": "canny"},
            },
        },
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    app = main.build_app(_make_args(cfg_path.as_posix()))
    with TestClient(app) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        c.tmp_path = tmp_path  # type: ignore[attr-defined]
        yield c


def _upload_source(client, width=SRC_W, height=SRC_H, frames=SRC_FRAMES) -> str:
    src = client.tmp_path / f"src_{width}x{height}_{frames}.mp4"
    if not src.exists():
        _write_test_mp4(src, width, height, frames)
    with src.open("rb") as fh:
        resp = client.post("/api/v1/upload/video", files={"file": ("src.mp4", fh, "video/mp4")})
    assert resp.status_code == 200, resp.text
    return resp.json()["video_id"]


def _payload(video_id: str | None = None, **overrides) -> dict:
    body = {
        "prompt": "a wide suburban kitchen",
        "width": CANVAS_W,
        "height": CANVAS_H,
        "num_frames": 25,
        "frame_rate": 24,
        "seed": 7,
        "loras": [{"name": OUTPAINT_LORA, "strength": 1.0}],
        "outpaint": dict(PADS),
    }
    if video_id is not None:
        body["reference_video_id"] = video_id
    body.update(overrides)
    return body


# ─────────────────────────────── schema (no I/O) ────────────────────────────


def test_outpaint_requires_reference_video(outpaint_client):
    resp = outpaint_client.post("/api/v1/generate", json=_payload())
    assert resp.status_code == 422
    assert "reference_video_id" in resp.text


def test_outpaint_requires_a_nonzero_pad(outpaint_client):
    vid = _upload_source(outpaint_client)
    body = _payload(vid, outpaint={"pad_left": 0, "pad_right": 0, "pad_top": 0, "pad_bottom": 0})
    resp = outpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 422
    assert "non-zero pad" in resp.text


def test_outpaint_excludes_conditioning_images(outpaint_client):
    vid = _upload_source(outpaint_client)
    body = _payload(vid, conditioning_images=[{"image_id": "x", "frame_idx": 0}])
    resp = outpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 422
    assert "conditioning_images" in resp.text


def test_outpaint_excludes_crop_output(outpaint_client):
    vid = _upload_source(outpaint_client)
    body = _payload(vid, crop_output={"width": 640, "height": 320})
    resp = outpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 422
    assert "crop_output" in resp.text


def test_outpaint_rejects_a_keep_region_the_blend_would_consume(outpaint_client):
    """The blend's mask dilation reaches ~a tenth of the canvas long side inward,
    so a tiny keep rectangle is entirely replaced by generated pixels."""
    vid = _upload_source(outpaint_client)
    # canvas 768x384, pads leaving a 128x128 keep region -> below the 256 floor.
    body = _payload(
        vid,
        outpaint={"pad_left": 320, "pad_right": 320, "pad_top": 128, "pad_bottom": 128},
    )
    resp = outpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 422
    assert "keep region" in resp.text


def test_plain_generate_is_unaffected_by_the_new_field(outpaint_client):
    """A request that omits ``outpaint`` must behave exactly as before."""
    resp = outpaint_client.post(
        "/api/v1/generate",
        json={"prompt": "hello", "width": 512, "height": 320, "num_frames": 9, "seed": 1},
    )
    assert resp.status_code == 202


# ─────────────────────────── endpoint guards (ffprobe) ──────────────────────


@has_ffmpeg
def test_outpaint_rejects_a_source_whose_resolution_is_not_the_keep_region(outpaint_client):
    vid = _upload_source(outpaint_client, width=640, height=256)
    resp = outpaint_client.post("/api/v1/generate", json=_payload(vid))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "OUTPAINT_SOURCE_MISMATCH"


@has_ffmpeg
def test_outpaint_rejects_a_source_shorter_than_num_frames(outpaint_client):
    vid = _upload_source(outpaint_client, frames=17)
    resp = outpaint_client.post("/api/v1/generate", json=_payload(vid, num_frames=25))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "OUTPAINT_SOURCE_TOO_SHORT"


@has_ffmpeg
def test_outpaint_rejects_a_control_lora_that_preprocesses_the_reference(outpaint_client):
    """Running the green canvas through canny/depth would hand the model an edge
    map of a sentinel colour, which is meaningless."""
    vid = _upload_source(outpaint_client)
    body = _payload(vid, loras=[{"name": CANNY_LORA, "strength": 1.0}])
    resp = outpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "OUTPAINT_PREPROCESS_CONFLICT"


@has_ffmpeg
def test_outpaint_completes_on_the_mock_backend_at_canvas_size(outpaint_client):
    vid = _upload_source(outpaint_client)
    resp = outpaint_client.post("/api/v1/generate", json=_payload(vid))
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]

    status = outpaint_client.get(f"/api/v1/jobs/{job_id}").json()
    assert status["status"] == "completed", status

    out_dir = outpaint_client.tmp_path / "outputs" / job_id
    canvas = out_dir / "outpaint_canvas.mp4"
    assert canvas.exists(), "the green canvas must be written next to the output"
    assert video_io.probe_resolution(canvas) == (CANVAS_W, CANVAS_H)
    assert video_io.frame_count(canvas) == 25
    # The placeholder itself comes out at the CANVAS size, not the source size.
    assert video_io.probe_resolution(out_dir / "output.mp4") == (CANVAS_W, CANVAS_H)


# ──────────────────────────── green canvas helper ───────────────────────────


@has_ffmpeg
def test_pad_green_mp4_keeps_the_sentinel_exact_and_the_frame_count_exact(tmp_path):
    """The sentinel must survive the encode bit-exactly: the IC-LoRA replaces
    exactly RGB(102, 255, 0), and a yuv420p round trip would shift it."""
    from PIL import Image

    src = tmp_path / "src.mp4"
    _write_test_mp4(src, SRC_W, SRC_H, SRC_FRAMES)
    canvas = tmp_path / "canvas.mp4"
    video_io.pad_green_mp4(
        src, canvas,
        canvas_width=CANVAS_W, canvas_height=CANVAS_H,
        pad_left=PADS["pad_left"], pad_top=PADS["pad_top"],
        frame_rate=24, num_frames=41,  # deliberately MORE than the source has
    )

    assert video_io.probe_resolution(canvas) == (CANVAS_W, CANVAS_H)
    # 41 > 33: tpad clone-extends the tail so the count is exact regardless.
    assert video_io.frame_count(canvas) == 41
    assert not video_io.has_audio_stream(canvas), "the canvas is written video-only"

    frame_png = tmp_path / "f0.png"
    subprocess.run(
        [shutil.which("ffmpeg"), "-y", "-v", "error", "-i", str(canvas),
         "-frames:v", "1", str(frame_png)],
        check=True, capture_output=True,
    )
    px = Image.open(frame_png).convert("RGB")
    assert px.getpixel((5, 5)) == (102, 255, 0)  # top-left pad band
    assert px.getpixel((CANVAS_W - 5, CANVAS_H - 5)) == (102, 255, 0)  # bottom-right


# ───────────────────────────── worker payload ───────────────────────────────


def test_runner_payload_carries_the_outpaint_block_only_when_requested(tmp_path):
    """The worker payload is an additive contract: a non-outpaint job's payload
    must be byte-identical to before this feature existed."""
    from api.models import GenerateRequest
    from services.ltx_runner import _RealBackend

    captured: list[dict] = []

    class _Spy(_RealBackend):
        def __init__(self):  # noqa: D107 - test double, no real worker
            pass

        def _send(self, payload):
            captured.append(payload)
            raise _Stop

        def _stderr_tail(self):
            return ""

        @property
        def loaded(self):
            return True

    class _Stop(Exception):
        pass

    def _run(request):
        spy = _Spy()
        spy._lock = _NullLock()
        try:
            spy.generate(
                request,
                output_dir=tmp_path,
                reference_video_path=tmp_path / "canvas.mp4",
                outpaint_source_path=tmp_path / "src.mp4",
                seed=5,
            )
        except RuntimeError:
            pass  # _send raises through as "worker died"; the payload is captured

    class _NullLock:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    plain = GenerateRequest(prompt="p", width=512, height=320, num_frames=9, seed=1)
    _run(plain)
    assert "outpaint" not in captured[-1]

    outpainting = GenerateRequest(
        prompt="p", width=CANVAS_W, height=CANVAS_H, num_frames=9, seed=1,
        loras=[{"name": OUTPAINT_LORA, "strength": 1.0}],
        reference_video_id="vid",
        outpaint=dict(PADS),
    )
    _run(outpainting)
    block = captured[-1]["outpaint"]
    assert block["canvas_width"] == CANVAS_W
    assert block["canvas_height"] == CANVAS_H
    assert block["pad_left"] == PADS["pad_left"]
    assert block["pad_bottom"] == PADS["pad_bottom"]
    assert block["blend_dilation_stage1"] == 5
    assert block["blend_dilation_stage2"] == 2
    assert block["freeze_source_audio"] is True
    assert block["source_path"].endswith("src.mp4")
    # The reference the engine conditions on is the CANVAS, not the source.
    assert captured[-1]["reference_video"]["path"].endswith("canvas.mp4")
