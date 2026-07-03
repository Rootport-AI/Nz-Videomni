"""IC-LoRA Phase B — API exposure end-to-end tests (mock backend, no GPU).

Covers POST /upload/video, the additive GenerateRequest fields (loras +
reference_video_id), their cross-validation, the adapter-name registry
(unknown/path-like rejection), and a full mock-mode generate that completes with
loras + a reference video. The engine weight-patch mechanism itself is unit
tested in test_ic_lora_forward.py; here we exercise the app-layer plumbing.
"""

from __future__ import annotations

import argparse
import json

import pytest
import yaml
from fastapi.testclient import TestClient

import main

# A tiny but non-empty mp4-ish blob. The video store validates extension + size
# only (no decode), and the mock backend never opens it, so bytes are arbitrary.
FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64

REGISTERED_LORA = "pixel-spatial-upscaler-x2"


def _make_args(config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=config_path, te_offload=None, dit_cpu_load=None,
    )


@pytest.fixture()
def lora_client(tmp_path):
    """A mock-backend client whose config registers one IC-LoRA adapter name
    pointing at a dummy (existing) safetensors file — the registry checks the
    file exists, the mock backend never reads it."""
    lora_file = tmp_path / "adapter.safetensors"
    lora_file.write_bytes(b"\x00" * 8)  # must exist for the registry resolve()

    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {
            "backend": "mock",
            "ic_loras": {REGISTERED_LORA: lora_file.as_posix()},  # absolute -> used as-is
        },
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    app = main.build_app(_make_args(cfg_path.as_posix()))
    with TestClient(app) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        yield c


def _upload_video(client) -> str:
    r = client.post("/api/v1/upload/video", files={"file": ("ref.mp4", FAKE_MP4, "video/mp4")})
    assert r.status_code == 200, r.text
    return r.json()["video_id"]


# ---------------------------------------------------------------- upload/video


def test_upload_video_roundtrip(lora_client):
    r = lora_client.post("/api/v1/upload/video", files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["video_id"]
    assert body["stored_path"].endswith("/input.mp4")
    assert body["content_type"] == "video/mp4"
    assert body["size_bytes"] == len(FAKE_MP4)

    ctx = lora_client.app_context
    stored = ctx.video_upload_store.path_for(body["video_id"])
    assert stored.exists()
    assert stored.read_bytes() == FAKE_MP4


def test_upload_video_bad_extension_rejected(lora_client):
    r = lora_client.post("/api/v1/upload/video", files={"file": ("bad.txt", b"nope", "text/plain")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "UPLOAD_INVALID_TYPE"


def test_upload_video_empty_rejected(lora_client):
    r = lora_client.post("/api/v1/upload/video", files={"file": ("empty.mp4", b"", "video/mp4")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "UPLOAD_INVALID_TYPE"


# ------------------------------------------------------------ generate + loras


def _base_payload(**over) -> dict:
    payload = {
        "prompt": "A busy town street, a woman walks and says 'LTX 2.3!'",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "frame_rate": 24.0,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "seed": 99,
        "pipeline": "distilled",
    }
    payload.update(over)
    return payload


def test_generate_with_loras_and_reference_completes(lora_client):
    vid = _upload_video(lora_client)
    payload = _base_payload(
        loras=[{"name": REGISTERED_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    r = lora_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    job = lora_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = lora_client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    # Additive request fields recorded.
    assert meta["request"]["loras"][0]["name"] == REGISTERED_LORA
    assert meta["request"]["reference_video_id"] == vid
    # Additive ic_lora block present for a lora job.
    assert meta["ic_lora"]["loras"][0]["name"] == REGISTERED_LORA
    assert meta["ic_lora"]["reference_video_id"] == vid


def test_loras_without_reference_video_422(lora_client):
    payload = _base_payload(loras=[{"name": REGISTERED_LORA, "strength": 1.0}])
    r = lora_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 422
    assert "reference_video_id" in r.text


def test_reference_without_loras_422(lora_client):
    vid = _upload_video(lora_client)
    payload = _base_payload(reference_video_id=vid)
    r = lora_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 422
    assert "requires at least one lora" in r.text


def test_unknown_adapter_name_404(lora_client):
    vid = _upload_video(lora_client)
    payload = _base_payload(
        loras=[{"name": "no-such-adapter", "strength": 1.0}],
        reference_video_id=vid,
    )
    r = lora_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "LORA_NOT_FOUND"


def test_path_like_lora_name_rejected_422(lora_client):
    vid = _upload_video(lora_client)
    payload = _base_payload(
        loras=[{"name": "../../etc/passwd", "strength": 1.0}],
        reference_video_id=vid,
    )
    r = lora_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 422
    assert "not a path" in r.text


def test_unknown_reference_video_404(lora_client):
    payload = _base_payload(
        loras=[{"name": REGISTERED_LORA, "strength": 1.0}],
        reference_video_id="does-not-exist",
    )
    r = lora_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "REFERENCE_VIDEO_NOT_FOUND"


def test_lora_strength_out_of_range_422(lora_client):
    vid = _upload_video(lora_client)
    payload = _base_payload(
        loras=[{"name": REGISTERED_LORA, "strength": 5.0}],  # > 2.0 bound
        reference_video_id=vid,
    )
    r = lora_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 422


def test_generate_without_new_fields_regression(lora_client):
    """A request omitting loras/reference_video_id behaves exactly as before:
    the new fields default (empty list / None), no ic_lora metadata block."""
    r = lora_client.post("/api/v1/generate", json=_base_payload(seed=7))
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = lora_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = lora_client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    assert meta["request"]["loras"] == []
    assert meta["request"]["reference_video_id"] is None
    assert "ic_lora" not in meta  # additive block absent for non-lora jobs
