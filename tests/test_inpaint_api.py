"""Inpainting (台帳 §3-55) — app-layer tests: schema, endpoint guards, payload,
and one end-to-end run on the mock backend.

Mock backend, no GPU. The engine's two-stage driver
(``engine/pipeline/inpaint_pipeline.py``) and the pure geometry/mask helpers are
tested in the engine venvs (``test_inpaint_geometry.py`` /
``test_inpaint_mask_decode.py``); the green canvas has its own file
(``test_inpaint_green_fill.py``). This one covers the plumbing from
``POST /generate`` down to the worker payload and back out to
``metadata.json``.

Most tests need REAL video files rather than a fake blob, because inpainting's
endpoint guards deliberately ffprobe both the source and the mask: the canvas
must be the source rounded up to the 128 grid, the mask must be exactly the
source's resolution, and it must carry exactly ``num_frames`` frames. Those
tests skip without ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import threading
import types
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from api.models import GenerateRequest
from services import video_io
from services.ltx_runner import _RealBackend

INPAINT_LORA = "in-outpainting"
SECOND_CONTROL = "deblur-control"
CANNY_LORA = "canny-control"

# The source is 320x256: both sides clear INPAINT_MIN_SOURCE_SIDE (256), the
# width needs a 64px right band to reach the 128 grid and the height needs none.
# One fixture therefore exercises "a band" and "no band" at once.
SRC_W, SRC_H = 320, 256
CANVAS_W, CANVAS_H = 384, 256
FRAMES = 9
FPS = 24

has_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="inpainting's endpoint guards and canvas builder both shell out to ffmpeg",
)


def _control_safetensors(path, factor: str = "1") -> None:
    """A minimal safetensors file whose header declares
    ``reference_downscale_factor`` — what makes ``services.lora_registry``
    classify the adapter as CONTROL kind with ``preprocess: none``, which is the
    combination inpainting requires (copied from tests/test_outpaint_api.py)."""
    header = {
        "__metadata__": {"reference_downscale_factor": factor},
        "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": {
            "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
        },
    }
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 4)


def _make_args(config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=config_path, te_offload=None, dit_cpu_load=None,
    )


def _write_source_mp4(path, width: int, height: int, frames: int, fps: int = FPS) -> None:
    subprocess.run(
        [
            shutil.which("ffmpeg"), "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={frames / fps + 1.0:.3f}",
            "-frames:v", str(frames),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True, capture_output=True,
    )


def _write_mask_mp4(path, width: int, height: int, frames: int, fps: int = FPS) -> None:
    """A white rectangle on black, drifting one pixel per frame — a mask that
    really moves, which is the case this feature exists for."""
    from PIL import Image, ImageDraw

    stage = path.parent / f"{path.stem}_png"
    stage.mkdir(exist_ok=True)
    for i in range(frames):
        img = Image.new("RGB", (width, height), (0, 0, 0))
        x0, y0 = width // 4 + i, height // 4
        ImageDraw.Draw(img).rectangle(
            [x0, y0, x0 + width // 4, y0 + height // 4], fill=(255, 255, 255)
        )
        img.save(stage / f"{i:06d}.png")
    subprocess.run(
        [
            shutil.which("ffmpeg"), "-y", "-v", "error",
            "-framerate", str(fps),
            "-i", str(stage / "%06d.png"),
            "-frames:v", str(frames),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True, capture_output=True,
    )


@pytest.fixture()
def inpaint_client(tmp_path):
    control = tmp_path / "in-outpainting.safetensors"
    _control_safetensors(control)
    second = tmp_path / "deblur.safetensors"
    _control_safetensors(second)
    canny = tmp_path / "canny.safetensors"
    _control_safetensors(canny)

    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {
            "backend": "mock",
            "ic_loras": {
                INPAINT_LORA: control.as_posix(),
                SECOND_CONTROL: second.as_posix(),
                CANNY_LORA: {"path": canny.as_posix(), "preprocess": "canny"},
            },
        },
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
        "state_file": (tmp_path / "state.json").as_posix(),
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    app = main.build_app(_make_args(cfg_path.as_posix()))
    with TestClient(app) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        c.tmp_path = tmp_path  # type: ignore[attr-defined]
        yield c


def _upload(client, path) -> str:
    with path.open("rb") as fh:
        resp = client.post(
            "/api/v1/upload/video", files={"file": (path.name, fh, "video/mp4")}
        )
    assert resp.status_code == 200, resp.text
    return resp.json()["video_id"]


def _upload_source(client, width=SRC_W, height=SRC_H, frames=FRAMES) -> str:
    src = client.tmp_path / f"src_{width}x{height}_{frames}.mp4"
    if not src.exists():
        _write_source_mp4(src, width, height, frames)
    return _upload(client, src)


def _upload_mask(client, width=SRC_W, height=SRC_H, frames=FRAMES) -> str:
    mask = client.tmp_path / f"mask_{width}x{height}_{frames}.mp4"
    if not mask.exists():
        _write_mask_mp4(mask, width, height, frames)
    return _upload(client, mask)


def _payload(video_id=None, mask_id="mask-id", **overrides) -> dict:
    body = {
        "prompt": "a clean wall where the sign was",
        "width": CANVAS_W,
        "height": CANVAS_H,
        "num_frames": FRAMES,
        "frame_rate": FPS,
        "seed": 7,
        "loras": [{"name": INPAINT_LORA, "strength": 1.0}],
        "inpaint": {"mask_video_id": mask_id, "window_start_sec": 0.0},
    }
    if video_id is not None:
        body["reference_video_id"] = video_id
    body.update(overrides)
    return body


# ─────────────────────────────── schema (no I/O) ────────────────────────────
# The six shape-only rules of Docs/INPAINTING_DESIGN.md §6.2.


def test_rule1_inpaint_requires_a_reference_video(inpaint_client):
    resp = inpaint_client.post("/api/v1/generate", json=_payload())
    assert resp.status_code == 422
    assert "reference_video_id" in resp.text


def test_rule2_inpaint_and_outpaint_are_mutually_exclusive(inpaint_client):
    # No upload: the pydantic validator answers before any file is touched, so
    # a bare id is enough and this test needs no ffmpeg.
    body = _payload("any-id", outpaint={"pad_left": 128, "pad_right": 0, "pad_top": 0,
                                   "pad_bottom": 0})
    resp = inpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 422
    assert "mutually exclusive" in resp.text


def test_rule3_inpaint_excludes_conditioning_images(inpaint_client):
    body = _payload("any-id", conditioning_images=[{"image_id": "x", "frame_idx": 0}])
    resp = inpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 422
    assert "conditioning_images" in resp.text


def test_rule4_inpaint_excludes_crop_output(inpaint_client):
    body = _payload("any-id", crop_output={"width": 256, "height": 256})
    resp = inpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 422
    assert "crop_output" in resp.text


@pytest.mark.parametrize("field,value", [("width", 320), ("height", 320)])
def test_rule5_the_canvas_must_sit_on_the_128_grid(inpaint_client, field, value):
    """320 is a multiple of 64 (so the generic reference rule would pass it) but
    not of 128, which is the grid the two-stage upscale needs."""
    resp = inpaint_client.post(
        "/api/v1/generate", json=_payload("any-id", **{field: value})
    )
    assert resp.status_code == 422
    assert "multiple of 128" in resp.text


def test_rule6_num_frames_is_still_8n_plus_1(inpaint_client):
    """Inherited from the existing schema — inpainting adds no length rule of
    its own, and the 481 ceiling is the existing Field bound (D12)."""
    resp = inpaint_client.post(
        "/api/v1/generate", json=_payload("any-id", num_frames=10)
    )
    assert resp.status_code == 422
    assert "8n+1" in resp.text


def test_the_schema_accepts_the_two_dilation_knobs_and_defaults_them():
    """The frontend sends neither; the GPU gate needs both. Defaults are
    outpainting's own 5 / 2."""
    req = GenerateRequest(
        prompt="x", width=CANVAS_W, height=CANVAS_H, num_frames=FRAMES,
        reference_video_id="vid", loras=[{"name": INPAINT_LORA, "strength": 1.0}],
        inpaint={"mask_video_id": "m"},
    )
    assert req.inpaint.blend_dilation_stage1 == 5
    assert req.inpaint.blend_dilation_stage2 == 2
    assert req.inpaint.window_start_sec == 0.0
    with pytest.raises(ValidationError, match="less than or equal to 15"):
        GenerateRequest(
            prompt="x", width=CANVAS_W, height=CANVAS_H, num_frames=FRAMES,
            reference_video_id="vid", loras=[{"name": INPAINT_LORA, "strength": 1.0}],
            inpaint={"mask_video_id": "m", "blend_dilation_stage1": 16},
        )


def test_plain_generate_is_unaffected_by_the_new_field(inpaint_client):
    """A request that omits ``inpaint`` must behave exactly as before."""
    resp = inpaint_client.post(
        "/api/v1/generate",
        json={"prompt": "hello", "width": 512, "height": 320, "num_frames": 9, "seed": 1},
    )
    assert resp.status_code == 202


# ─────────────────────────── endpoint guards (ffprobe) ──────────────────────
# The seven error codes of Docs/INPAINTING_DESIGN.md §6.2, one test each.


@has_ffmpeg
def test_guard_preprocess_conflict(inpaint_client):
    """A canny/pose/depth adapter would be handed the edge map of a sentinel
    colour — meaningless, and the In-Outpainting adapter expects raw pixels."""
    vid = _upload_source(inpaint_client)
    mask = _upload_mask(inpaint_client)
    body = _payload(vid, mask, loras=[{"name": CANNY_LORA, "strength": 1.0}])
    resp = inpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INPAINT_PREPROCESS_CONFLICT"


@has_ffmpeg
def test_guard_lora_invalid_when_two_control_adapters_are_requested(inpaint_client):
    vid = _upload_source(inpaint_client)
    mask = _upload_mask(inpaint_client)
    body = _payload(
        vid, mask,
        loras=[{"name": INPAINT_LORA, "strength": 1.0},
               {"name": SECOND_CONTROL, "strength": 1.0}],
    )
    resp = inpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INPAINT_LORA_INVALID"


@has_ffmpeg
def test_guard_source_mismatch_when_the_canvas_is_not_the_round_up(inpaint_client):
    vid = _upload_source(inpaint_client)
    mask = _upload_mask(inpaint_client)
    resp = inpaint_client.post(
        "/api/v1/generate", json=_payload(vid, mask, width=512)
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INPAINT_SOURCE_MISMATCH"
    assert "384x256" in err["detail"] and "320x256" in err["detail"]


@has_ffmpeg
def test_guard_source_mismatch_when_the_source_is_below_the_floor(inpaint_client):
    """A 128px side rounds up to a 128px canvas side, whose stage-1 half is 64px
    — no context at all around the mask. The WIDTH is 256 here only because the
    schema's own ``width >= 256`` bound would otherwise answer first, and this
    test is about the inpaint floor rather than that one."""
    vid = _upload_source(inpaint_client, width=256, height=128)
    mask = _upload_mask(inpaint_client, width=256, height=128)
    body = _payload(vid, mask, width=256, height=128)
    resp = inpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INPAINT_SOURCE_MISMATCH"
    assert "256px" in err["detail"]


@has_ffmpeg
def test_guard_mask_not_found(inpaint_client):
    vid = _upload_source(inpaint_client)
    resp = inpaint_client.post(
        "/api/v1/generate", json=_payload(vid, "no-such-mask")
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "INPAINT_MASK_NOT_FOUND"


@has_ffmpeg
def test_guard_mask_resolution_mismatch(inpaint_client):
    """The mask is NEVER resized (§14): a mismatch is refused, not scaled."""
    vid = _upload_source(inpaint_client)
    mask = _upload_mask(inpaint_client, width=256, height=256)
    resp = inpaint_client.post("/api/v1/generate", json=_payload(vid, mask))
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INPAINT_MASK_RESOLUTION_MISMATCH"
    assert "320x256" in err["detail"] and "256x256" in err["detail"]


@has_ffmpeg
def test_guard_mask_frame_mismatch(inpaint_client):
    """A short mask is the dangerous one: ffmpeg's default framesync repeats the
    last frame, so the canvas would come out full-length with an unmasked tail."""
    vid = _upload_source(inpaint_client, frames=25)
    mask = _upload_mask(inpaint_client, frames=17)
    resp = inpaint_client.post(
        "/api/v1/generate", json=_payload(vid, mask, num_frames=25)
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INPAINT_MASK_FRAME_MISMATCH"
    assert "17 frames" in err["detail"] and "25" in err["detail"]


@has_ffmpeg
def test_guard_window_out_of_range(inpaint_client):
    vid = _upload_source(inpaint_client)
    mask = _upload_mask(inpaint_client)
    body = _payload(vid, mask)
    body["inpaint"]["window_start_sec"] = 5.0  # the source is 9 frames @ 24fps
    resp = inpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INPAINT_WINDOW_OUT_OF_RANGE"


@has_ffmpeg
def test_guard_source_mismatch_when_the_source_cannot_be_probed(inpaint_client, monkeypatch):
    """ffprobe missing or failing is a THIRD situation, not a mismatch: there is
    no source size, so the message must say so and quote the REQUESTED canvas
    rather than pretend a canvas was derived from a size nobody could read."""
    vid = _upload_source(inpaint_client)
    mask = _upload_mask(inpaint_client)
    monkeypatch.setattr(video_io, "probe_resolution", lambda p: None)
    resp = inpaint_client.post("/api/v1/generate", json=_payload(vid, mask))
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INPAINT_SOURCE_MISMATCH"
    assert "could not be probed" in err["detail"]
    assert f"{CANVAS_W}x{CANVAS_H} canvas" in err["detail"]
    assert "canvas from the source would be" not in err["detail"]


@has_ffmpeg
def test_the_mock_refuses_a_canvas_that_is_not_the_sources_round_up(inpaint_client, tmp_path):
    """DEFENCE IN DEPTH, and the reason the end-to-end assertion is worth
    anything. The mock derives its render size from the MASK and then checks it
    against the REQUEST with the same 128-grid rule the engine applies — so
    "the output came out at the source size" is a claim about two independent
    inputs agreeing, not the mask being compared with itself.

    Unreachable through the endpoint (the guards above catch the mismatch
    first), so the backend is called directly."""
    from api.errors import APIError  # noqa: F401  (kept close to the raise below)

    mask = inpaint_client.tmp_path / f"mask_{SRC_W}x{SRC_H}_{FRAMES}.mp4"
    if not mask.exists():
        _write_mask_mp4(mask, SRC_W, SRC_H, FRAMES)
    runner = inpaint_client.app_context.pipeline_manager.runner
    # 512x256 is a legal canvas for the schema but NOT 320x256 rounded up (384x256).
    request = GenerateRequest(**{
        "prompt": "x", "width": 512, "height": 256, "num_frames": FRAMES,
        "frame_rate": float(FPS), "seed": 1,
        "loras": [{"name": INPAINT_LORA, "strength": 1.0}],
        "reference_video_id": "vid",
        "inpaint": {"mask_video_id": "mask", "window_start_sec": 0.0},
    })
    with pytest.raises(RuntimeError, match="inpaint geometry mismatch"):
        runner.generate(request, tmp_path / "out", inpaint_mask_path=mask)


def test_the_mock_refuses_an_inpaint_job_with_no_mask_path(inpaint_client, tmp_path):
    """``mask_path=None`` must never reach the engine: the whole job is about a
    mask. Fail here rather than let the worker decode nothing."""
    runner = inpaint_client.app_context.pipeline_manager.runner
    request = GenerateRequest(**{
        "prompt": "x", "width": CANVAS_W, "height": CANVAS_H, "num_frames": FRAMES,
        "frame_rate": float(FPS), "seed": 1,
        "loras": [{"name": INPAINT_LORA, "strength": 1.0}],
        "reference_video_id": "vid",
        "inpaint": {"mask_video_id": "mask", "window_start_sec": 0.0},
    })
    with pytest.raises(RuntimeError, match="never reached the backend"):
        runner.generate(request, tmp_path / "out")


@has_ffmpeg
def test_the_guards_run_in_the_order_a_user_can_act_on(inpaint_client):
    """Adapters, then the source, then the mask. A request that is wrong in two
    ways names the FIRST thing to fix, not an arbitrary one."""
    vid = _upload_source(inpaint_client)
    body = _payload(vid, "no-such-mask", loras=[{"name": CANNY_LORA, "strength": 1.0}])
    resp = inpaint_client.post("/api/v1/generate", json=body)
    assert resp.json()["error"]["code"] == "INPAINT_PREPROCESS_CONFLICT"


# ───────────────────────────── worker payload spy ───────────────────────────


def _capturing_backend(captured: list[dict]) -> _RealBackend:
    """The ``__new__`` harness from tests/test_ltx_runner_payload.py: no
    subprocess, ``_send`` captured, the terminal event faked."""
    be = _RealBackend.__new__(_RealBackend)
    be._proc = types.SimpleNamespace(poll=lambda: None)  # type: ignore[attr-defined]
    be._lock = threading.Lock()

    def _send(msg: dict) -> None:
        captured.append(msg)
        out = Path(msg["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 16)

    be._send = _send  # type: ignore[attr-defined]
    be._read_worker_events = (  # type: ignore[attr-defined]
        lambda cb, chain, prefix: {
            "event": "done", "seed_used": 7, "peak_vram_mb": 100,
            "inpaint": {"mask_proof": {"decoded_frames": 9}},
        }
    )
    return be


def _request(**over) -> GenerateRequest:
    base = dict(
        prompt="a clean wall",
        width=CANVAS_W,
        height=CANVAS_H,
        num_frames=FRAMES,
        frame_rate=float(FPS),
        seed=99,
        loras=[{"name": INPAINT_LORA, "strength": 1.0}],
        reference_video_id="vid",
        inpaint={"mask_video_id": "mask", "window_start_sec": 1.5},
    )
    base.update(over)
    return GenerateRequest(**base)


@has_ffmpeg
def test_the_payload_carries_the_geometry_and_both_paths(tmp_path):
    window = tmp_path / "_inpaint_window.mp4"
    _write_source_mp4(window, SRC_W, SRC_H, FRAMES)
    mask = tmp_path / "mask.mp4"
    _write_mask_mp4(mask, SRC_W, SRC_H, FRAMES)

    captured: list[dict] = []
    be = _capturing_backend(captured)
    outcome = be.generate(
        _request(),
        tmp_path / "out",
        lora_paths=[(tmp_path / "adapter.safetensors", 1.0, "none")],
        reference_video_path=tmp_path / "canvas.mp4",
        inpaint_source_path=window,
        inpaint_mask_path=mask,
    )
    block = captured[0]["inpaint"]
    assert block == {
        "source_path": str(window),
        "mask_path": str(mask),
        "canvas_width": CANVAS_W,
        "canvas_height": CANVAS_H,
        "source_width": SRC_W,
        "source_height": SRC_H,
        "blend_dilation_stage1": 5,
        "blend_dilation_stage2": 2,
    }
    # ...and no pads: the engine derives them from canvas - source, which is the
    # same single-source-of-truth rule the API enforces.
    assert "pad_right" not in block and "pad_bottom" not in block
    # The engine's own block comes back on the outcome, for metadata.json.
    assert outcome.inpaint == {"mask_proof": {"decoded_frames": 9}}


def test_a_plain_job_has_no_inpaint_key_at_all(tmp_path):
    """The additive contract, stated as a payload fact: a request without an
    ``inpaint`` block produces a payload without an ``inpaint`` key."""
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate(
        GenerateRequest(prompt="hello", width=512, height=320, num_frames=9),
        tmp_path / "plain",
    )
    assert "inpaint" not in captured[0]
    assert "outpaint" not in captured[0]


def test_an_outpaint_job_still_has_no_inpaint_key(tmp_path):
    """The two blocks never travel together — the schema forbids it, and the
    worker asserts it. This is the payload half of that promise."""
    captured: list[dict] = []
    be = _capturing_backend(captured)
    be.generate(
        GenerateRequest(
            prompt="wider", width=768, height=384, num_frames=9,
            reference_video_id="vid",
            loras=[{"name": INPAINT_LORA, "strength": 1.0}],
            outpaint={"pad_left": 128, "pad_right": 128, "pad_top": 64, "pad_bottom": 64},
        ),
        tmp_path / "outpaint",
        lora_paths=[(tmp_path / "a.safetensors", 1.0, "none")],
        reference_video_path=tmp_path / "canvas.mp4",
        outpaint_source_path=tmp_path / "src.mp4",
    )
    assert "outpaint" in captured[0]
    assert "inpaint" not in captured[0]


# ──────────────────────────── mock end to end ───────────────────────────────


@has_ffmpeg
def test_inpaint_completes_on_the_mock_backend_at_the_source_size(inpaint_client):
    vid = _upload_source(inpaint_client)
    mask = _upload_mask(inpaint_client)
    resp = inpaint_client.post("/api/v1/generate", json=_payload(vid, mask))
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]

    status = inpaint_client.get(f"/api/v1/jobs/{job_id}").json()
    assert status["status"] == "completed", status

    out_dir = inpaint_client.tmp_path / "outputs" / job_id

    # 1. The cut window, at the SOURCE resolution and exactly num_frames long.
    window = out_dir / "_inpaint_window.mp4"
    assert window.exists(), "the cut window is kept as the record of what went in"
    assert video_io.probe_resolution(window) == (SRC_W, SRC_H)
    assert video_io.frame_count(window) == FRAMES

    # 2. The green canvas, at the CANVAS resolution and the same length.
    canvas = out_dir / "inpaint_canvas.mp4"
    assert canvas.exists()
    assert video_io.probe_resolution(canvas) == (CANVAS_W, CANVAS_H)
    assert video_io.frame_count(canvas) == FRAMES

    # 3. THE OUTPUT IS THE SOURCE SIZE, not the canvas size. The engine crops
    #    the pad bands off; the mock renders at the source size for the same
    #    reason, so a geometry mistake shows up here rather than on a GPU.
    assert video_io.probe_resolution(out_dir / "output.mp4") == (SRC_W, SRC_H)
    assert status["result"]["resolution"] == f"{SRC_W}x{SRC_H}"


@has_ffmpeg
def test_the_metadata_carries_the_inpaint_provenance(inpaint_client):
    vid = _upload_source(inpaint_client)
    mask = _upload_mask(inpaint_client)
    resp = inpaint_client.post("/api/v1/generate", json=_payload(vid, mask))
    job_id = resp.json()["job_id"]
    assert inpaint_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    meta = json.loads(
        (inpaint_client.tmp_path / "outputs" / job_id / "metadata.json").read_text("utf-8")
    )
    block = meta["inpaint"]
    # THE WHOLE PROVENANCE, key for key. On the mock this block IS the app's
    # provenance and nothing else (the mock reports no engine block), so the key
    # set is a complete statement rather than a spot check — a key quietly lost
    # on the way to metadata.json fails here.
    assert set(block) == {
        "source_video_id", "mask_video_id", "source_fps", "mask_fps", "resampled",
        "window_start_sec", "window_start_frame", "window_written_frames",
        "window_has_audio", "canvas_frames", "canvas_codec",
    }
    assert block["source_video_id"] == vid
    assert block["mask_video_id"] == mask
    assert block["source_fps"] == float(FPS)
    assert block["mask_fps"] == float(FPS)
    assert block["window_start_sec"] == 0.0
    assert block["window_start_frame"] == 0
    assert block["window_written_frames"] == FRAMES
    assert block["canvas_frames"] == FRAMES
    assert block["canvas_codec"] == "libx264rgb"
    assert block["resampled"] is False
    assert meta["output"]["resolution"] == f"{SRC_W}x{SRC_H}"
    # The source's own SIZE is deliberately NOT here: it belongs to the engine's
    # geometry block, and repeating it app-side would let the app's copy
    # overwrite the engine's when the two dicts are merged. The delivered size
    # is stated once, in ``output.resolution`` above.
    assert "source_width" not in block and "source_height" not in block


@has_ffmpeg
def test_a_non_zero_window_start_cuts_from_the_right_place(inpaint_client):
    vid = _upload_source(inpaint_client, frames=25)
    mask = _upload_mask(inpaint_client, frames=FRAMES)
    body = _payload(vid, mask)
    body["inpaint"]["window_start_sec"] = 8 / FPS  # frame 8
    resp = inpaint_client.post("/api/v1/generate", json=body)
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    assert inpaint_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    meta = json.loads(
        (inpaint_client.tmp_path / "outputs" / job_id / "metadata.json").read_text("utf-8")
    )
    assert meta["inpaint"]["window_start_frame"] == 8
    assert meta["inpaint"]["window_written_frames"] == FRAMES


@has_ffmpeg
def test_a_plain_mock_job_writes_no_inpaint_block(inpaint_client):
    resp = inpaint_client.post(
        "/api/v1/generate",
        json={"prompt": "hello", "width": 512, "height": 320, "num_frames": 9, "seed": 1},
    )
    job_id = resp.json()["job_id"]
    assert inpaint_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    meta = json.loads(
        (inpaint_client.tmp_path / "outputs" / job_id / "metadata.json").read_text("utf-8")
    )
    assert "inpaint" not in meta
    assert not (inpaint_client.tmp_path / "outputs" / job_id / "inpaint_canvas.mp4").exists()
