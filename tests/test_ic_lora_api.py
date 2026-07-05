"""IC-LoRA Phase B/C — API exposure end-to-end tests (mock backend, no GPU).

Covers POST /upload/video, the additive GenerateRequest fields (loras +
reference_video_id), their cross-validation, the adapter-name registry
(unknown/path-like rejection), and a full mock-mode generate that completes with
loras + a reference video. The engine weight-patch mechanism itself is unit
tested in test_ic_lora_forward.py; here we exercise the app-layer plumbing.

Phase C additions: the registry's dict-valued entries (``IcLoraEntry`` ->
``preprocess`` kind, config.py) alongside legacy string-valued entries
(backward compat), and the >1-distinct-preprocess-kind conflict rejection
(``LORA_PREPROCESS_CONFLICT``, 400). No engine preprocessing is exercised here
(that is Phase C slice 2/3) -- this file only checks that the registry/config
plumbing resolves and reaches the runner payload boundary correctly.
"""

from __future__ import annotations

import argparse
import json

import pytest
import yaml
from fastapi.testclient import TestClient

import main
from api.errors import APIError
from config import AppConfig
from services.lora_registry import LoraRegistry
from services.ltx_runner import _resolve_reference_preprocess

# A tiny but non-empty mp4-ish blob. The video store validates extension + size
# only (no decode), and the mock backend never opens it, so bytes are arbitrary.
FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64

REGISTERED_LORA = "pixel-spatial-upscaler-x2"
CANNY_LORA = "canny-control"
POSE_LORA = "pose-control"


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


@pytest.fixture()
def mixed_registry_client(tmp_path):
    """A mock-backend client whose registry mixes a legacy string-valued entry
    (Phase B, preprocess implied "none") with two Phase C dict-valued entries
    (same file, different preprocess kinds) -- exactly the config.yaml shape
    from IC_LORA_PHASE_C_WORKORDER.md §3-3."""
    lora_file = tmp_path / "adapter.safetensors"
    lora_file.write_bytes(b"\x00" * 8)
    control_file = tmp_path / "union-control.safetensors"
    control_file.write_bytes(b"\x00" * 8)

    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {
            "backend": "mock",
            "ic_loras": {
                REGISTERED_LORA: lora_file.as_posix(),  # legacy string value
                CANNY_LORA: {"path": control_file.as_posix(), "preprocess": "canny"},
                POSE_LORA: {"path": control_file.as_posix(), "preprocess": "dwpose"},
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


def test_reference_resolution_not_divisible_by_128_422(lora_client):
    """All registered adapters use reference_downscale_factor=2, so the
    reference is consumed at half output resolution on the 64-grid -- 512x320
    (320 % 128 != 0) is rejected up front instead of crashing the worker's VAE
    encode deep in the job (real failure reproduced during Phase C prep)."""
    vid = _upload_video(lora_client)
    payload = _base_payload(
        width=512,
        height=320,
        loras=[{"name": REGISTERED_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    r = lora_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "REFERENCE_RESOLUTION_INVALID"


def test_reference_resolution_divisible_by_128_completes(lora_client):
    """512x256 (both divisible by 128) is unaffected by the new check and
    still completes end to end, exactly as before."""
    vid = _upload_video(lora_client)
    payload = _base_payload(
        width=512,
        height=256,
        loras=[{"name": REGISTERED_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    r = lora_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = lora_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job


def test_non_reference_job_ignores_128_divisibility(lora_client):
    """No reference_video_id (and no loras) -- 512x320 is unaffected by the
    new check, exactly like before this change."""
    r = lora_client.post("/api/v1/generate", json=_base_payload(width=512, height=320))
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = lora_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job


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
    # S1 control-adjustability fields default to None in the frozen request dump.
    assert meta["request"]["conditioning_attention_strength"] is None
    assert meta["request"]["reference_video_strength"] is None
    assert "ic_lora" not in meta  # additive block absent for non-lora jobs


# ------------------------------------------ S1: IC-LoRA control adjustability


def test_attention_strength_requires_loras(lora_client):
    payload = _base_payload(conditioning_attention_strength=0.6)
    r = lora_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 422
    assert "conditioning_attention_strength requires at least one lora" in r.text


def test_reference_strength_requires_loras(lora_client):
    payload = _base_payload(reference_video_strength=0.8)
    r = lora_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 422
    assert "reference_video_strength requires at least one lora" in r.text


def test_strength_fields_accept_range(lora_client):
    vid = _upload_video(lora_client)
    # 0.0 and 1.0 both accepted (with loras + reference present).
    for value in (0.0, 1.0):
        payload = _base_payload(
            width=512, height=256,  # 128-divisible reference resolution
            loras=[{"name": REGISTERED_LORA, "strength": 1.0}],
            reference_video_id=vid,
            conditioning_attention_strength=value,
            reference_video_strength=value,
        )
        r = lora_client.post("/api/v1/generate", json=payload)
        assert r.status_code == 202, r.text
    # Out-of-range values rejected per field.
    for field in ("conditioning_attention_strength", "reference_video_strength"):
        for bad in (-0.1, 1.1):
            payload = _base_payload(
                width=512, height=256,
                loras=[{"name": REGISTERED_LORA, "strength": 1.0}],
                reference_video_id=vid,
                **{field: bad},
            )
            r = lora_client.post("/api/v1/generate", json=payload)
            assert r.status_code == 422, (field, bad, r.text)


def test_strength_fields_in_request_dump(lora_client):
    vid = _upload_video(lora_client)
    payload = _base_payload(
        width=512, height=256,
        loras=[{"name": REGISTERED_LORA, "strength": 1.0}],
        reference_video_id=vid,
        conditioning_attention_strength=0.6,
        reference_video_strength=0.8,
    )
    r = lora_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = lora_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = lora_client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    # Frozen-additive request dump carries both.
    assert meta["request"]["conditioning_attention_strength"] == 0.6
    assert meta["request"]["reference_video_strength"] == 0.8
    # ic_lora block records both (only-when-meaningful).
    assert meta["ic_lora"]["conditioning_attention_strength"] == 0.6
    assert meta["ic_lora"]["reference_video_strength"] == 0.8


# --------------------------------------------------------- Phase C: registry


def test_registry_string_entry_resolves_backward_compat(tmp_path):
    """A legacy string-valued ic_loras entry (Phase B) still resolves, now as a
    3-tuple whose preprocess is "none" -- LoraRegistry.resolve unit-level."""
    lora_file = tmp_path / "adapter.safetensors"
    lora_file.write_bytes(b"\x00" * 8)
    config = AppConfig.model_validate(
        {"model": {"ic_loras": {REGISTERED_LORA: lora_file.as_posix()}}}
    )
    registry = LoraRegistry(config)
    path, strength, preprocess = registry.resolve(REGISTERED_LORA, 1.0)
    assert path == lora_file.resolve()
    assert strength == 1.0
    assert preprocess == "none"


def test_registry_dict_entry_resolves_with_preprocess(tmp_path):
    """A Phase C dict-valued entry (IcLoraEntry) resolves with its declared
    preprocess kind -- LoraRegistry.resolve unit-level."""
    control_file = tmp_path / "union-control.safetensors"
    control_file.write_bytes(b"\x00" * 8)
    config = AppConfig.model_validate(
        {
            "model": {
                "ic_loras": {
                    POSE_LORA: {"path": control_file.as_posix(), "preprocess": "dwpose"},
                }
            }
        }
    )
    registry = LoraRegistry(config)
    path, strength, preprocess = registry.resolve(POSE_LORA, 1.0)
    assert path == control_file.resolve()
    assert strength == 1.0
    assert preprocess == "dwpose"


def test_generate_with_dict_valued_adapter_completes(mixed_registry_client):
    """A dict-valued (Phase C) adapter name resolves through the full app-layer
    plumbing (registry -> pipeline_manager -> runner) exactly like a legacy
    string-valued one; the mock backend ignores preprocess and still completes."""
    vid = _upload_video(mixed_registry_client)
    payload = _base_payload(
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    r = mixed_registry_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    job = mixed_registry_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = mixed_registry_client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    assert meta["ic_lora"]["loras"][0]["name"] == CANNY_LORA


def test_generate_with_string_and_dict_adapters_mixed_completes(mixed_registry_client):
    """A "none" preprocess adapter alongside a canny adapter is NOT a conflict
    (only >1 distinct non-"none" kind is rejected)."""
    vid = _upload_video(mixed_registry_client)
    payload = _base_payload(
        loras=[
            {"name": REGISTERED_LORA, "strength": 1.0},
            {"name": CANNY_LORA, "strength": 1.0},
        ],
        reference_video_id=vid,
    )
    r = mixed_registry_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = mixed_registry_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job


def test_generate_with_conflicting_preprocess_kinds_400(mixed_registry_client):
    """canny-control + pose-control together imply 2 distinct control kinds for
    ONE reference video -- rejected up front (400 LORA_PREPROCESS_CONFLICT)."""
    vid = _upload_video(mixed_registry_client)
    payload = _base_payload(
        loras=[
            {"name": CANNY_LORA, "strength": 1.0},
            {"name": POSE_LORA, "strength": 1.0},
        ],
        reference_video_id=vid,
    )
    r = mixed_registry_client.post("/api/v1/generate", json=payload)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "LORA_PREPROCESS_CONFLICT"


def test_resolve_reference_preprocess_all_none():
    assert _resolve_reference_preprocess([]) == "none"
    from pathlib import Path

    assert _resolve_reference_preprocess([(Path("a"), 1.0, "none")]) == "none"


def test_resolve_reference_preprocess_single_kind():
    from pathlib import Path

    lora_paths = [(Path("a"), 1.0, "none"), (Path("b"), 1.0, "canny")]
    assert _resolve_reference_preprocess(lora_paths) == "canny"


def test_resolve_reference_preprocess_conflict_raises_400():
    from pathlib import Path

    lora_paths = [(Path("a"), 1.0, "canny"), (Path("b"), 1.0, "dwpose")]
    with pytest.raises(APIError) as exc_info:
        _resolve_reference_preprocess(lora_paths)
    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "LORA_PREPROCESS_CONFLICT"
