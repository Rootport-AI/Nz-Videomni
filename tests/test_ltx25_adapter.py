"""LTX 2.5 engine adapter — payload contract and v1 feature scope (§3-98 P3b).

Three things are pinned here, none of which needs a GPU or a weight file:

1. the ``{"op":"load"}`` payload engine25's worker receives — key SET, key
   ORDER and values, as a golden snapshot, the same discipline the 2.3 payload
   has had since model management (tests/test_model_swap_load.py). The 2.3
   golden is deliberately untouched by this file; the two engines' payloads are
   independent contracts;
2. the v1 feature scope: which ``GenerateRequest`` fields are a 422, which are
   ignored-and-logged, and — the part a table alone cannot state — that those
   two sets do not overlap and that ``crop_output`` is in NEITHER;
3. the seams: LTX 2.5 reuses the 2.3 MOCK backend class rather than declaring
   its own, which is a design ruling ("やらない" list) and therefore a test.
"""

from __future__ import annotations

import logging

import pytest

from api.errors import APIError
from api.models import GenerateRequest
from config import AppConfig
from services.base_models import BaseModelDescriptor, CategoryDescriptor
from services.engines.ltx import adapter as ltx23
from services.engines.ltx25 import adapter as ltx25
from services.low_vram import build_low_vram_settings

LTX23_KV = {"general.architecture": "ltxv", "model_version": "2.3.0"}
LTX25_KV = {"general.architecture": "ltxv", "model_version": "2.5.0"}

# The exact key ORDER of the load payload. JSON serialization preserves
# insertion order, so an order change IS a byte change.
GOLDEN_PAYLOAD_KEYS_25 = [
    "op",
    "transformer_path",
    "text_encoder_path",
    "video_vae_path",
    "audio_vae_path",
    "spatial_upsampler_path",
    "blocks_on_gpu",
    "cache_weights",
    "deterministic",
]


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _category(name: str, extension: str, default_file) -> CategoryDescriptor:
    return CategoryDescriptor(
        name=name,
        scan=(str(default_file.parent),),
        extensions=(extension,),
        default_file=str(default_file),
    )


@pytest.fixture()
def ltx25_paths(tmp_path):
    """An LTX 2.5 base model whose every payload path exists (dummy files).

    Built in Python rather than as JSON for the same reason the 2.3 fixture is:
    descriptor paths are ``models_dir``-relative, and joining ``models_dir``
    with an ABSOLUTE path yields that absolute path, so each file resolves
    exactly where the fixture put it.
    """
    m = tmp_path / "m"
    paths = {
        "tr": _touch(m / "Weights" / "ltx-2.5-transformer-Q4_K_M.gguf"),
        "te": _touch(m / "TextEncoder" / "gemma-4-Q6_K.gguf"),
        "vv": _touch(m / "VAE" / "ltx25_video_vae.safetensors"),
        "av": _touch(m / "VAE" / "ltx25_audio_vae.safetensors"),
        "ups": _touch(m / "Upscaler" / "ltx-2.5-spatial-upscaler-x2.safetensors"),
    }
    descriptor = BaseModelDescriptor(
        id="LTX25FIXTURE",
        display_name="LTX 2.5 fixture",
        engine_family="ltx25",
        categories={
            "transformer": _category("transformer", ".gguf", paths["tr"]),
            "text_encoder": _category("text_encoder", ".gguf", paths["te"]),
            "video_vae": _category("video_vae", ".safetensors", paths["vv"]),
            "audio": _category("audio", ".safetensors", paths["av"]),
        },
        assets={"spatial_upsampler_path": str(paths["ups"])},
    )
    cfg = AppConfig.model_validate({"model": {"backend": "mock"}})
    return cfg, paths, descriptor


def _backend(cfg, descriptor) -> ltx25._RealBackend25:
    return ltx25._RealBackend25(cfg, build_low_vram_settings(cfg), descriptor)


def _golden(paths) -> dict:
    return {
        "op": "load",
        "transformer_path": str(paths["tr"]),
        "text_encoder_path": str(paths["te"]),
        "video_vae_path": str(paths["vv"]),
        "audio_vae_path": str(paths["av"]),
        "spatial_upsampler_path": str(paths["ups"]),
        "blocks_on_gpu": 8,  # low_vram default None -> `or DEFAULT_BLOCKS_ON_GPU`
        "cache_weights": True,
        "deterministic": True,
    }


# --------------------------------------------------------------------------- #
# 1) golden load payload
# --------------------------------------------------------------------------- #


def test_load_payload_golden_without_selection(ltx25_paths):
    cfg, paths, descriptor = ltx25_paths
    backend = _backend(cfg, descriptor)
    p_none = backend._build_load_payload(None)
    p_empty = backend._build_load_payload({})
    assert p_none == p_empty == _golden(paths)
    assert list(p_none) == GOLDEN_PAYLOAD_KEYS_25


def test_load_payload_selection_overrides_only_the_named_field(ltx25_paths):
    cfg, paths, descriptor = ltx25_paths
    backend = _backend(cfg, descriptor)
    alt = _touch(paths["tr"].parent / "alt-Q6.gguf")
    p = backend._build_load_payload({"transformer": str(alt)})
    golden = _golden(paths)
    assert p["transformer_path"] == str(alt)
    for key in GOLDEN_PAYLOAD_KEYS_25:
        if key != "transformer_path":
            assert p[key] == golden[key]
    assert list(p) == GOLDEN_PAYLOAD_KEYS_25


def test_load_payload_all_four_categories_map_to_expected_fields(ltx25_paths):
    cfg, _paths, descriptor = ltx25_paths
    backend = _backend(cfg, descriptor)
    p = backend._build_load_payload(
        {"transformer": "X_TR", "text_encoder": "X_TE", "video_vae": "X_VV", "audio": "X_AU"}
    )
    assert p["transformer_path"] == "X_TR"
    assert p["text_encoder_path"] == "X_TE"
    assert p["video_vae_path"] == "X_VV"
    assert p["audio_vae_path"] == "X_AU"


def test_load_payload_fails_loud_on_a_missing_asset(tmp_path, ltx25_paths):
    """The spatial upsampler is the ONE required asset; a descriptor that does
    not declare it must stop the load in the app layer, not in the worker."""
    cfg, _paths, descriptor = ltx25_paths
    stripped = BaseModelDescriptor(
        id=descriptor.id,
        display_name=descriptor.display_name,
        engine_family=descriptor.engine_family,
        categories=descriptor.categories,
        assets={},
    )
    with pytest.raises(RuntimeError, match="spatial_upsampler_path"):
        _backend(cfg, stripped)._build_load_payload(None)


def test_category_names_are_the_same_axes_as_ltx23():
    """The four categories are the axes a user picks a file on — a property of
    the product, not of an engine generation. state.json remembers a selection
    by these names, so they must not diverge per base model."""
    assert set(ltx25.SELECTION_FIELDS) == set(ltx23.SELECTION_FIELDS)
    # ...while every payload FIELD differs (the protocols are unrelated).
    assert not set(ltx25.SELECTION_FIELDS.values()) & set(ltx23.SELECTION_FIELDS.values())


def test_required_assets_is_the_upsampler_alone():
    assert ltx25.REQUIRED_ASSETS == ("spatial_upsampler_path",)
    # The audio VAE is required by the worker but arrives on the CATEGORY axis,
    # so it must NOT be declared a second time as an asset.
    assert "audio" in ltx25.SELECTION_FIELDS


def test_2_3_golden_payload_is_untouched_by_this_engine():
    """The 2.3 field table is a frozen byte contract; adding a second engine
    must not have edited it."""
    assert ltx23.SELECTION_FIELDS == {
        "transformer": "gguf_transformer_path",
        "text_encoder": "gguf_gemma_path",
        "video_vae": "component_video_vae_path",
        "audio": "component_audio_vae_path",
    }


# --------------------------------------------------------------------------- #
# 2) KV ruling
# --------------------------------------------------------------------------- #


def test_check_kv_accepts_2_5_and_refuses_2_3():
    ltx25.check_kv("transformer", "default", LTX25_KV)  # no raise
    with pytest.raises(APIError) as ei:
        ltx25.check_kv("transformer", "old", LTX23_KV)
    assert ei.value.code == "MODEL_INCOMPATIBLE" and ei.value.status_code == 422
    assert "ltxv 2.3.0" in ei.value.detail


def test_check_kv_patch_segment_is_ignored():
    ltx25.check_kv("transformer", "respin", {**LTX25_KV, "model_version": "2.5.9"})


def test_check_kv_refuses_a_foreign_architecture():
    with pytest.raises(APIError) as ei:
        ltx25.check_kv("transformer", "wan", {"general.architecture": "wan"})
    assert "'wan'系のモデルです" in ei.value.detail


def test_check_kv_missing_keys_warn_but_pass(caplog):
    with caplog.at_level(logging.WARNING, logger="ltx25.runner"):
        ltx25.check_kv("transformer", "homebrew", {})
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


@pytest.mark.parametrize("category", ["text_encoder", "video_vae", "audio"])
def test_check_kv_judges_only_the_transformer(category):
    ltx25.check_kv(category, "default", LTX23_KV)
    ltx25.check_kv(category, "default", {"general.architecture": "gemma3"})


# --------------------------------------------------------------------------- #
# 3) v1 feature scope — the GenerateRequest field table
# --------------------------------------------------------------------------- #


def _request(**overrides) -> GenerateRequest:
    return GenerateRequest(prompt="a quiet harbour at first light", **overrides)


_LORAS = [{"name": "style-a", "strength": 1.0}]

#: One VALID request per rejected field that sets that field non-default, so
#: the table below drives real ``GenerateRequest`` objects rather than
#: asserting against itself. Several entries carry companions because the
#: SCHEMA already couples them (outpaint needs the video it extends; a
#: reference video needs an IC-LoRA to condition; NAG needs a negative prompt)
#: — this engine's ruling has to survive those couplings, not dodge them.
REQUEST_OVERRIDES: dict[str, dict] = {
    "pipeline": {"pipeline": "two_stage_hq"},
    "outpaint": {
        "outpaint": {"pad_left": 64, "pad_right": 0, "pad_top": 0, "pad_bottom": 0},
        "reference_video_id": "vid-123",
        "loras": _LORAS,
    },
    "loras": {"loras": _LORAS},
    "reference_video_id": {"reference_video_id": "vid-123", "loras": _LORAS},
    "nag_enabled": {"nag_enabled": True, "negative_prompt": "blurry, low quality"},
    "vae_mode": {"vae_mode": "prune_vaed"},
    "attention_backend": {"attention_backend": "sage"},
    "keep_resident": {"keep_resident": True},
}


def test_reject_table_names_only_real_request_fields():
    fields = set(GenerateRequest.model_fields)
    assert {f for f, _feat, _p in ltx25.REJECT_TABLE} <= fields
    assert set(ltx25.IGNORED_FIELDS) <= fields


def test_reject_and_ignore_sets_are_disjoint():
    """A field is either refused or tolerated. Both would mean the answer
    depends on which check runs first."""
    assert not {f for f, _feat, _p in ltx25.REJECT_TABLE} & set(ltx25.IGNORED_FIELDS)


def test_crop_output_is_in_neither_set():
    """crop_output is an ffmpeg post-process the app applies to the finished
    mp4 — engine-independent, so it must be honoured, not refused or dropped."""
    assert "crop_output" not in {f for f, _feat, _p in ltx25.REJECT_TABLE}
    assert "crop_output" not in ltx25.IGNORED_FIELDS


def test_a_default_request_is_accepted():
    reqs = [_request(), _request(generation_mode="t2v"), _request(crop_output=None)]
    for r in reqs:
        ltx25.reject_unsupported(r)  # no raise


@pytest.mark.parametrize("field", list(REQUEST_OVERRIDES))
def test_every_rejected_field_raises_feature_unsupported(field):
    request = _request(**REQUEST_OVERRIDES[field])
    # The field under test really is non-default for this request...
    predicate = next(p for f, _feat, p in ltx25.REJECT_TABLE if f == field)
    assert predicate(request)
    # ...and the FIRST offender in table order is what the message names (a
    # schema-coupled companion field can legitimately be refused first).
    expected = next(feat for _f, feat, pred in ltx25.REJECT_TABLE if pred(request))
    with pytest.raises(APIError) as ei:
        ltx25.reject_unsupported(request)
    assert ei.value.code == "FEATURE_UNSUPPORTED" and ei.value.status_code == 422
    assert expected in ei.value.detail
    assert "LTX 2.3" in ei.value.detail


def test_reject_table_covers_every_field_the_fixture_names():
    """The parametrized test above is only as good as its request table."""
    assert set(REQUEST_OVERRIDES) == {f for f, _feat, _p in ltx25.REJECT_TABLE}


def test_unsupported_features_is_the_reject_table_plus_the_chain_family():
    features = set(ltx25.UNSUPPORTED_FEATURES)
    assert {feat for _f, feat, _p in ltx25.REJECT_TABLE} <= features
    assert {"chain", "retake", "end_source", "v2v", "a2v"} <= features
    assert len(ltx25.UNSUPPORTED_FEATURES) == len(features), "no duplicates"


def test_ignored_fields_are_logged_only_when_set(caplog):
    with caplog.at_level(logging.INFO, logger="ltx25.runner"):
        ltx25._log_ignored(_request())
    assert not [r for r in caplog.records if "ignores" in r.getMessage()]

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ltx25.runner"):
        ltx25._log_ignored(_request(negative_prompt="blurry", block_swap_prefetch=False))
    messages = [r.getMessage() for r in caplog.records if "ignores" in r.getMessage()]
    assert len(messages) == 1
    assert "negative_prompt" in messages[0] and "block_swap_prefetch" in messages[0]


def test_generate_chain_is_refused(ltx25_paths):
    cfg, _paths, descriptor = ltx25_paths
    backend = _backend(cfg, descriptor)
    with pytest.raises(APIError) as ei:
        backend.generate_chain(object(), output_dir=None)
    assert ei.value.code == "FEATURE_UNSUPPORTED" and ei.value.status_code == 422
    assert "Chained" in ei.value.detail


# --------------------------------------------------------------------------- #
# 4) seams
# --------------------------------------------------------------------------- #


def test_the_mock_backend_class_is_shared_with_ltx23():
    """Design ruling: no _MockBackend25. A synthetic gradient clip says nothing
    about which engine would have rendered it, so a second copy would be a copy
    with no content — only the reported label differs."""
    assert ltx25.LTX25Runner._MOCK_BACKEND_CLS is ltx23._MockBackend
    assert ltx23.LTXRunner._MOCK_BACKEND_CLS is ltx23._MockBackend
    assert ltx25.LTX25Runner._MOCK_BACKEND_LABEL != ltx23.LTXRunner._MOCK_BACKEND_LABEL


def test_mock_backend_label_reaches_the_outcome(tmp_path):
    """The label is not decoration: it is how metadata.json says WHICH engine's
    mock ran, which is what makes a GPU-free 2.3<->2.5 round trip checkable."""
    cfg = AppConfig.model_validate({"model": {"backend": "mock"}})
    runner = ltx25.LTX25Runner(cfg, build_low_vram_settings(cfg), _descriptor_stub())
    backend = runner._select_backend()
    assert isinstance(backend, ltx23._MockBackend)
    assert backend.backend_label == ltx25.MOCK_BACKEND_25


def _descriptor_stub() -> BaseModelDescriptor:
    return BaseModelDescriptor(
        id="LTX25", display_name="LTX 2.5", engine_family="ltx25",
        categories={"transformer": CategoryDescriptor("transformer", ("x",), (".gguf",))},
    )


def test_worker_seams_point_at_engine25():
    cls = ltx25._RealBackend25
    assert cls._WORKER_MODULE == "engine25.worker"
    assert cls._ENGINE_DIR_VALUE == "./engine25"
    # A SEPARATE log file: after a 2.3<->2.5 swap both logs must survive for the
    # round-trip gate to be checkable at all.
    assert cls._LOG_NAME != ltx23._RealBackend._LOG_NAME
    # The protocol frame prefix is deliberately SHARED (one parent-side reader).
    assert cls._PREFIX == ltx23._RealBackend._PREFIX


def test_engine_python_comes_from_its_own_config_key():
    cfg = AppConfig.model_validate({})
    assert ltx25._RealBackend25._engine_python_value(cfg).endswith(
        ".venv-engine-ltx25/Scripts/python.exe"
    )
    assert ltx23._RealBackend._engine_python_value(cfg).endswith(
        ".venv-engine/Scripts/python.exe"
    )
    cfg2 = AppConfig.model_validate({"model": {"engine_python_ltx25": "./other/python.exe"}})
    assert ltx25._RealBackend25._engine_python_value(cfg2) == "./other/python.exe"
    # ...and overriding 2.5's key leaves 2.3's alone.
    assert ltx23._RealBackend._engine_python_value(cfg2).endswith(
        ".venv-engine/Scripts/python.exe"
    )


def test_child_env_carries_no_2_3_engine_knobs(ltx25_paths, tmp_path):
    """Every LTX_* variable the 2.3 worker reads names a code path inside
    engine/. Inheriting them into engine25 would be the borrowed-assumption bug
    the separate engine exists to avoid.

    Measured as "which LTX_* keys this method ADDS to the inherited
    environment", not "are there any" — the parent process's own environment
    passes through by design (that is what makes a manual override possible),
    and the test process itself sets one (LTX_DISABLE_GRADIO)."""
    import os

    cfg, _paths, descriptor = ltx25_paths
    inherited = {k for k in os.environ if k.startswith("LTX_")}
    env = _backend(cfg, descriptor)._build_child_env(tmp_path)
    assert {k for k in env if k.startswith("LTX_")} == inherited
    assert env["PYTHONPATH"] == str(tmp_path)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"

    ltx23_env = ltx23._RealBackend(cfg, build_low_vram_settings(cfg), descriptor)._build_child_env(
        tmp_path
    )
    assert {"LTX_COMPONENT_FILES", "LTX_TE_OFFLOAD", "LTX_DIT_CPU_LOAD"} <= set(ltx23_env)


def test_sage_is_permanently_off_for_this_engine():
    cfg = AppConfig.model_validate({})
    runner = ltx25.LTX25Runner(cfg, build_low_vram_settings(cfg), _descriptor_stub())
    assert runner.sage_available is False
