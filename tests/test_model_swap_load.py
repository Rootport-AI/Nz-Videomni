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
from config import PROJECT_ROOT, AppConfig
from conftest import build_model_layout
from services.base_models import BaseModelDescriptor, CategoryDescriptor
from services.low_vram import build_low_vram_settings
from services.ltx_runner import _RealBackend
from services.model_registry import DEFAULT_NAME, precheck_model_file

#: A structurally valid, empty GGUF header: magic, version 3, tensor_count 0,
#: kv_count 0. The precheck now PARSES the header (it reads the engine-selection
#: KV out of it), so a stub has to be a real header rather than four magic bytes
#: followed by zeros — those zeros read as GGUF version 0 and are rejected, which
#: is the point.
GGUF_STUB = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)


def _touch(path, data: bytes = b"\0\0\0\0"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _gguf(path, **kv: str):
    """Write a minimal GGUF file carrying the given string KV entries."""
    body = b"".join(
        _gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value)
        for key, value in kv.items()
    )
    return _touch(path, b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + body)


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


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
    # PrunaVAED (§3-50, 2026-08-05): the pruned video VAE decoder's path. Rides
    # the load payload like the other component paths, but is NOT required to
    # exist (a job asking for it downgrades instead) and is NOT swappable
    # through the model registry.
    "component_video_vae_pruned_path",
    "gguf_per_layer_quant",
    "block_swap_blocks_on_gpu",
    "vae_spatial_tile_size",
    "vae_temporal_tile_size",
]


#: Where the PrunaVAED decoder sits inside the model store, as the SHIPPED
#: descriptor declares it (scripts/manifests/10-ltx23.json ``assets``). The
#: golden payload's pruned path is the only one that is NOT a fixture file (it
#: need not exist), so the fixture below hands the backend this exact location
#: — and tests/test_base_model_contract.py
#: (``test_shipped_ltx23_asset_paths_match_the_removed_config_fields``) is what
#: keeps the constant honest about the real descriptor.
PRUNED_DECODER_REL = "models/LTX23/VAE/prunavaed/PrunaVAED-decoder-bf16.safetensors"


@pytest.fixture()
def real_paths(tmp_path):
    """A BASE MODEL whose every payload-relevant path exists (dummy files).

    The fixed weight paths moved out of config into the base-model descriptor
    (§3-97 P3b), so this builds a descriptor instead of a config block. It is
    constructed in Python rather than written as JSON on purpose: a descriptor
    normally spells its paths relative to ``models_dir``, and this fixture needs
    its files scattered across tmp_path (they are the historical config values,
    verbatim). ``models_dir`` joined with an ABSOLUTE path yields that absolute
    path, so each entry below resolves exactly where the fixture put it, while
    the pruned decoder stays a project-relative path — proving the same builder
    resolves both.
    """
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
    descriptor = BaseModelDescriptor(
        id="LTXFIXTURE",
        display_name="LTX fixture",
        engine_family="ltx",
        categories={
            "transformer": _category("transformer", ".gguf", paths["tr"]),
            "text_encoder": _category("text_encoder", ".gguf", paths["te"]),
            "video_vae": _category("video_vae", ".safetensors", paths["vv"]),
            "audio": _category("audio", ".safetensors", paths["av"]),
        },
        assets={
            "gemma_root": str(gemma_root),
            "spatial_upsampler_path": str(paths["ups"]),
            "component_text_projection_path": str(paths["tp"]),
            # Relative to models_dir (which is the project default here), i.e.
            # the shipped location — and deliberately NOT created on disk: the
            # pruned decoder is the one payload path that need not exist.
            "component_video_vae_pruned_path": PRUNED_DECODER_REL.split("/", 1)[1],
        },
    )
    cfg = AppConfig.model_validate({"model": {"backend": "mock"}})
    return cfg, gemma_root, paths, descriptor


def _category(name: str, extension: str, default_file) -> CategoryDescriptor:
    return CategoryDescriptor(
        name=name,
        scan=(str(default_file.parent),),
        extensions=(extension,),
        default_file=str(default_file),
    )


def _backend(cfg, descriptor) -> _RealBackend:
    return _RealBackend(cfg, build_low_vram_settings(cfg), descriptor)


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
        # Not overridden by the fixture, so this is the config DEFAULT resolved
        # against the project root — spelled out here rather than read back off
        # the config, so a silent change to the default location trips the test.
        # The file need not exist (and does not, in this fixture).
        "component_video_vae_pruned_path": str(
            (
                PROJECT_ROOT
                / "models/LTX23/VAE/prunavaed/PrunaVAED-decoder-bf16.safetensors"
            ).resolve()
        ),
        "gguf_per_layer_quant": True,
        "block_swap_blocks_on_gpu": 8,  # low_vram default None -> `or 8`
        "vae_spatial_tile_size": 0,
        "vae_temporal_tile_size": 0,
    }


def test_load_payload_byte_identical_without_selection(real_paths):
    cfg, gemma_root, paths, descriptor = real_paths
    backend = _backend(cfg, descriptor)
    p_none = backend._build_load_payload(None)
    p_empty = backend._build_load_payload({})
    golden = _golden(cfg, gemma_root, paths)
    assert p_none == p_empty == golden
    # Key ORDER is part of the byte contract (json.dumps preserves it).
    assert list(p_none) == GOLDEN_PAYLOAD_KEYS


def test_load_payload_selection_overrides_only_named_field(real_paths):
    cfg, gemma_root, paths, descriptor = real_paths
    backend = _backend(cfg, descriptor)
    alt = _touch(paths["tr"].parent.parent / "alt" / "alt-Q6.gguf")
    p = backend._build_load_payload({"transformer": str(alt)})
    golden = _golden(cfg, gemma_root, paths)
    assert p["gguf_transformer_path"] == str(alt)
    for key in GOLDEN_PAYLOAD_KEYS:
        if key != "gguf_transformer_path":
            assert p[key] == golden[key]
    assert list(p) == GOLDEN_PAYLOAD_KEYS


def test_load_payload_all_four_categories_map_to_expected_fields(real_paths):
    cfg, _gemma_root, paths, descriptor = real_paths
    backend = _backend(cfg, descriptor)
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
    good = _touch(tmp_path / "good.gguf", GGUF_STUB)
    precheck_model_file("transformer", "good", good)  # no raise
    bad = _touch(tmp_path / "bad.gguf", b"XXXX" + b"\0" * 16)
    with pytest.raises(APIError) as ei:
        precheck_model_file("transformer", "bad", bad)
    assert ei.value.code == "MODEL_INCOMPATIBLE" and ei.value.status_code == 422


def test_precheck_returns_gguf_engine_kv(tmp_path):
    """The header read that already happens here hands the engine-selection KV
    back to the caller, so nothing has to re-open the file (§2.2)."""
    path = _gguf(
        tmp_path / "kv.gguf",
        **{"general.architecture": "ltxv", "model_version": "2.3.0"},
    )
    assert precheck_model_file("transformer", "kv", path) == {
        "general.architecture": "ltxv",
        "model_version": "2.3.0",
    }
    # A GGUF without those keys is not an error here (check_kv rules on that).
    assert precheck_model_file("transformer", "bare", _touch(tmp_path / "b.gguf", GGUF_STUB)) == {}
    # safetensors carry no such metadata at all.
    st = _touch(tmp_path / "vae.safetensors", struct.pack("<Q", 2) + b"{}")
    assert precheck_model_file("video_vae", "vae", st) == {}


def test_precheck_rejects_unknown_weight_file_type(tmp_path):
    """Without a descriptor there is no category gate, but the file still has
    to be a type this precheck can look inside."""
    with pytest.raises(APIError) as ei:
        precheck_model_file("transformer", "x", _touch(tmp_path / "weights.bin"))
    assert ei.value.code == "MODEL_INCOMPATIBLE"


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
    """The category gate is the DESCRIPTOR's statement about which extensions a
    category accepts, so the caller passes the category descriptor in (P3b —
    the module no longer keeps a table of its own)."""
    st = _touch(tmp_path / "weights.safetensors", struct.pack("<Q", 2) + b"{}")
    transformer = CategoryDescriptor(
        name="transformer", scan=("Weights",), extensions=(".gguf",)
    )
    with pytest.raises(APIError) as ei:
        precheck_model_file("transformer", "weights", st, descriptor=transformer)
    assert ei.value.code == "MODEL_INCOMPATIBLE"
    assert ei.value.status_code == 422


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
    alt = _touch(weights / "alt-transformer.gguf", GGUF_STUB)
    _touch(weights / "bad-magic.gguf", b"XXXX" + b"\0" * 16)
    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {
            "backend": "mock",
            **build_model_layout(tmp_path),  # hermetic descriptor + models dir
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
