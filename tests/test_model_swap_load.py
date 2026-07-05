"""Model-management S2: POST /pipeline/load model selection + golden payload.

Two contracts under test, both GPU-free:

1. BYTE-IDENTITY — the worker load payload built with NO selection (None or
   empty) equals the golden snapshot of what the pre-model-management code
   sent (key set, key ORDER, and values). This is the regression guard for
   "default combination stays byte-identical".
2. SWAP SEMANTICS — an explicit ``models`` block resolves names through the
   registry, prechecks magic bytes, forces a rebuild only when the selection
   actually changes, updates ``active_models`` on success, and fails loud
   (404/422/409) without native-crash risk.
"""

from __future__ import annotations

import argparse
import struct

import pytest
import yaml
from fastapi.testclient import TestClient

import main
from api.errors import APIError
from config import AppConfig
from services.low_vram import build_low_vram_settings
from services.ltx_runner import _RealBackend
from services.model_registry import DEFAULT_NAME, precheck_model_file


def _touch(path, data: bytes = b"\0\0\0\0"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- #
# 1) golden payload snapshot (design §4.2)
# --------------------------------------------------------------------------- #

# The exact key ORDER of the {"op":"load"} payload the worker receives. JSON
# serialization preserves insertion order, so order changes ARE byte changes.
GOLDEN_PAYLOAD_KEYS = [
    "op",
    "checkpoint_path",
    "gemma_root",
    "upsampler_path",
    "gguf_transformer_path",
    "gguf_gemma_path",
    "component_video_vae_path",
    "component_audio_vae_path",
    "component_text_projection_path",
    "gguf_per_layer_quant",
    "block_swap_blocks_on_gpu",
    "vae_spatial_tile_size",
    "vae_temporal_tile_size",
]


@pytest.fixture()
def real_paths(tmp_path):
    """A config whose every payload-relevant path exists (dummy files)."""
    m = tmp_path / "m"
    gemma_root = m / "tokenizer"
    gemma_root.mkdir(parents=True)
    paths = {
        "ups": _touch(m / "upsampler.safetensors"),
        "tr": _touch(m / "gguf" / "release" / "transformer-Q4.gguf"),
        "te": _touch(m / "gemma" / "gemma-Q4.gguf"),
        "vv": _touch(m / "components" / "vae" / "video_vae.safetensors"),
        "av": _touch(m / "components" / "vae" / "audio_vae.safetensors"),
        "tp": _touch(m / "components" / "te" / "text_projection.safetensors"),
    }
    cfg = AppConfig.model_validate(
        {
            "model": {
                "backend": "mock",
                "gemma_root": str(gemma_root),
                "spatial_upsampler_path": str(paths["ups"]),
                "gguf_transformer_path": str(paths["tr"]),
                "gguf_gemma_path": str(paths["te"]),
                "component_video_vae_path": str(paths["vv"]),
                "component_audio_vae_path": str(paths["av"]),
                "component_text_projection_path": str(paths["tp"]),
            }
        }
    )
    return cfg, gemma_root, paths


def _golden(cfg: AppConfig, gemma_root, paths) -> dict:
    """What the pre-model-management load() sent (transcribed literally from
    the removed inline dict — values from config, knob defaults)."""
    return {
        "op": "load",
        "checkpoint_path": "",  # checkpoint_path=None default -> ""
        "gemma_root": str(gemma_root),
        "upsampler_path": str(paths["ups"]),
        "gguf_transformer_path": str(paths["tr"]),
        "gguf_gemma_path": str(paths["te"]),
        "component_video_vae_path": str(paths["vv"]),
        "component_audio_vae_path": str(paths["av"]),
        "component_text_projection_path": str(paths["tp"]),
        "gguf_per_layer_quant": True,
        "block_swap_blocks_on_gpu": 8,  # low_vram default None -> `or 8`
        "vae_spatial_tile_size": 0,
        "vae_temporal_tile_size": 0,
    }


def test_load_payload_byte_identical_without_selection(real_paths):
    cfg, gemma_root, paths = real_paths
    backend = _RealBackend(cfg, build_low_vram_settings(cfg))
    p_none = backend._build_load_payload(None)
    p_empty = backend._build_load_payload({})
    golden = _golden(cfg, gemma_root, paths)
    assert p_none == p_empty == golden
    # Key ORDER is part of the byte contract (json.dumps preserves it).
    assert list(p_none) == GOLDEN_PAYLOAD_KEYS


def test_load_payload_selection_overrides_only_named_field(real_paths):
    cfg, gemma_root, paths = real_paths
    backend = _RealBackend(cfg, build_low_vram_settings(cfg))
    alt = _touch(paths["tr"].parent.parent / "alt" / "alt-Q6.gguf")
    p = backend._build_load_payload({"transformer": str(alt)})
    golden = _golden(cfg, gemma_root, paths)
    assert p["gguf_transformer_path"] == str(alt)
    for key in GOLDEN_PAYLOAD_KEYS:
        if key != "gguf_transformer_path":
            assert p[key] == golden[key]
    assert list(p) == GOLDEN_PAYLOAD_KEYS


def test_load_payload_all_four_categories_map_to_expected_fields(real_paths):
    cfg, _gemma_root, paths = real_paths
    backend = _RealBackend(cfg, build_low_vram_settings(cfg))
    sel = {
        "transformer": "X_TR",
        "text_encoder": "X_TE",
        "video_vae": "X_VV",
        "audio": "X_AU",
    }
    p = backend._build_load_payload(sel)
    assert p["gguf_transformer_path"] == "X_TR"
    assert p["gguf_gemma_path"] == "X_TE"
    assert p["component_video_vae_path"] == "X_VV"
    assert p["component_audio_vae_path"] == "X_AU"
    # Non-swappable fields untouched.
    assert p["component_text_projection_path"] == str(paths["tp"])


# --------------------------------------------------------------------------- #
# 2) magic-bytes precheck (design §5.1)
# --------------------------------------------------------------------------- #

def test_precheck_gguf_magic(tmp_path):
    good = _touch(tmp_path / "good.gguf", b"GGUF" + b"\0" * 16)
    precheck_model_file("transformer", "good", good)  # no raise
    bad = _touch(tmp_path / "bad.gguf", b"XXXX" + b"\0" * 16)
    with pytest.raises(APIError) as ei:
        precheck_model_file("transformer", "bad", bad)
    assert ei.value.code == "MODEL_INCOMPATIBLE" and ei.value.status_code == 422


def test_precheck_safetensors_header(tmp_path):
    good = _touch(tmp_path / "good.safetensors", struct.pack("<Q", 2) + b"{}")
    precheck_model_file("video_vae", "good", good)  # no raise
    bad_len = _touch(tmp_path / "badlen.safetensors", struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(APIError) as ei:
        precheck_model_file("audio", "badlen", bad_len)
    assert ei.value.code == "MODEL_INCOMPATIBLE"
    bad_json = _touch(tmp_path / "badjson.safetensors", struct.pack("<Q", 2) + b"!!")
    with pytest.raises(APIError):
        precheck_model_file("audio", "badjson", bad_json)


def test_precheck_extension_category_mismatch(tmp_path):
    st = _touch(tmp_path / "weights.safetensors", struct.pack("<Q", 2) + b"{}")
    with pytest.raises(APIError) as ei:
        precheck_model_file("transformer", "weights", st)
    assert ei.value.code == "MODEL_INCOMPATIBLE"


# --------------------------------------------------------------------------- #
# 3) POST /pipeline/load API semantics (mock backend, temp config)
# --------------------------------------------------------------------------- #

def _make_args(config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=config_path, te_offload=None, dit_cpu_load=None,
    )


@pytest.fixture()
def swap_client(tmp_path):
    """Like conftest's client, plus a transformers registry with a valid-magic
    alt GGUF, a registered-but-absent ghost, and a wrong-magic file."""
    weights = tmp_path / "weights"
    alt = _touch(weights / "alt-transformer.gguf", b"GGUF" + b"\0" * 16)
    _touch(weights / "bad-magic.gguf", b"XXXX" + b"\0" * 16)
    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {
            "backend": "mock",
            "transformers": {
                "alt": alt.as_posix(),
                "ghost": (weights / "ghost.gguf").as_posix(),  # never created
                "bad-magic": (weights / "bad-magic.gguf").as_posix(),
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


def _mock_backend(client):
    return client.app_context.pipeline_manager.runner._backend


def test_bodyless_load_response_unchanged(swap_client):
    r = swap_client.post("/api/v1/pipeline/load")
    assert r.status_code == 200
    body = r.json()
    # Legacy response key set exactly — no additive "models" key on this path.
    assert set(body) == {"pipeline_loaded", "pipeline_type", "state"}
    assert body["pipeline_loaded"] is True
    assert _mock_backend(swap_client).last_selection is None


def test_swap_to_alt_updates_active_and_propagates_selection(swap_client):
    r = swap_client.post("/api/v1/pipeline/load", json={"models": {"transformer": "alt"}})
    assert r.status_code == 200
    body = r.json()
    assert body["pipeline_loaded"] is True
    assert body["models"]["transformer"] == "alt"
    assert body["models"]["text_encoder"] == DEFAULT_NAME
    # Selection reached the backend as the resolved absolute path.
    sel = _mock_backend(swap_client).last_selection
    assert sel is not None and sel["transformer"].endswith("alt-transformer.gguf")
    # GET /models reflects the new active.
    models = swap_client.get("/api/v1/models").json()
    assert models["categories"]["transformer"]["active"] == "alt"
    assert models["categories"]["audio"]["active"] == DEFAULT_NAME


def test_same_selection_is_noop_no_worker_restart(swap_client):
    swap_client.post("/api/v1/pipeline/load", json={"models": {"transformer": "alt"}})
    backend = _mock_backend(swap_client)
    calls_before = backend.load_calls
    r = swap_client.post("/api/v1/pipeline/load", json={"models": {"transformer": "alt"}})
    assert r.status_code == 200
    assert backend.load_calls == calls_before  # no unload/load cycle
    assert r.json()["models"]["transformer"] == "alt"


def test_partial_block_keeps_other_categories_active(swap_client):
    swap_client.post("/api/v1/pipeline/load", json={"models": {"transformer": "alt"}})
    backend = _mock_backend(swap_client)
    calls_before = backend.load_calls
    # Naming only text_encoder=default leaves transformer=alt in effect ->
    # effective selection equals the live one -> no-op.
    r = swap_client.post(
        "/api/v1/pipeline/load", json={"models": {"text_encoder": "default"}}
    )
    assert r.status_code == 200
    assert r.json()["models"]["transformer"] == "alt"
    assert backend.load_calls == calls_before


def test_swap_back_to_default_forces_rebuild(swap_client):
    swap_client.post("/api/v1/pipeline/load", json={"models": {"transformer": "alt"}})
    backend = _mock_backend(swap_client)
    calls_before = backend.load_calls
    r = swap_client.post(
        "/api/v1/pipeline/load", json={"models": {"transformer": "default"}}
    )
    assert r.status_code == 200
    assert backend.load_calls == calls_before + 1  # unload -> load cycle
    assert r.json()["models"]["transformer"] == DEFAULT_NAME
    # All-default selection -> NO overrides (byte-identical payload semantics).
    assert backend.last_selection is None
    models = swap_client.get("/api/v1/models").json()
    assert models["categories"]["transformer"]["active"] == DEFAULT_NAME


def test_bodyless_load_after_unload_reuses_active_selection(swap_client):
    """§4.1: an omitted selection keeps the current active — auto-load after an
    unload must not silently revert a swap."""
    swap_client.post("/api/v1/pipeline/load", json={"models": {"transformer": "alt"}})
    swap_client.post("/api/v1/pipeline/unload")
    r = swap_client.post("/api/v1/pipeline/load")  # bodyless
    assert r.status_code == 200
    sel = _mock_backend(swap_client).last_selection
    assert sel is not None and sel["transformer"].endswith("alt-transformer.gguf")
    models = swap_client.get("/api/v1/models").json()
    assert models["categories"]["transformer"]["active"] == "alt"


def test_unknown_category_404(swap_client):
    r = swap_client.post("/api/v1/pipeline/load", json={"models": {"upscaler": "x"}})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_unknown_name_404(swap_client):
    r = swap_client.post(
        "/api/v1/pipeline/load", json={"models": {"transformer": "no-such"}}
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_registered_but_missing_file_422(swap_client):
    r = swap_client.post(
        "/api/v1/pipeline/load", json={"models": {"transformer": "ghost"}}
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "MODEL_FILE_MISSING"


def test_bad_magic_422_before_any_load(swap_client):
    backend_before = _mock_backend(swap_client)  # None until first load
    r = swap_client.post(
        "/api/v1/pipeline/load", json={"models": {"transformer": "bad-magic"}}
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "MODEL_INCOMPATIBLE"
    # The precheck rejected it BEFORE any backend load happened.
    assert backend_before is None or backend_before.load_calls == 0


def test_busy_guard_409_for_swap(swap_client):
    swap_client.app_context.job_store.has_active = lambda: True  # type: ignore[method-assign]
    r = swap_client.post(
        "/api/v1/pipeline/load", json={"models": {"transformer": "alt"}}
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "JOB_BUSY"
