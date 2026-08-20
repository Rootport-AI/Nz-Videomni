"""Contracts between the SHIPPED base-model descriptors and the code that
reads them (multi-engine foundation §3-97 P3b).

Three separate promises, all of them cheap and all of them the kind that break
silently:

1. every category a descriptor declares can actually reach the worker
   (``SELECTION_FIELDS``);
2. the fixed asset paths the engine adapter asks for by key exist in the
   descriptor, spelled exactly as the removed ``config.model`` fields spelled
   them;
3. a ``config.yaml`` that still carries one of those removed keys says so out
   loud instead of ignoring it in silence.
"""

from __future__ import annotations

import logging

import yaml

from config import DEPRECATED_MODEL_KEYS, AppConfig, load_config
from conftest import base_model_descriptor, build_model_layout, write_model_file
from services.base_models import BaseModelDescriptor, load_base_models
from services.engines.ltx.adapter import REQUIRED_ASSETS, SELECTION_FIELDS
from services.low_vram import build_low_vram_settings
from services.ltx_runner import LTXRunner

#: The asset keys the LTX adapter resolves by name, with the ``config.model``
#: field each one replaced and the path that field held. A LITERAL, so the test
#: below compares the descriptor against history rather than against itself.
HISTORICAL_ASSET_PATHS: dict[str, str] = {
    "gemma_root": "LTX23/TextEncoder/tokenizer",
    "spatial_upsampler_path": "LTX23/Upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "component_text_projection_path": (
        "LTX23/TextEncoder/ltx-2.3_text_projection_bf16.safetensors"
    ),
    "component_video_vae_pruned_path": (
        "LTX23/VAE/prunavaed/PrunaVAED-decoder-bf16.safetensors"
    ),
}


def _shipped() -> dict[str, BaseModelDescriptor]:
    return load_base_models(AppConfig().manifest_dir)


def test_selection_fields_cover_every_declared_category():
    """§4.4: the descriptor's category set IS the payload's swappable set.

    A category added to a manifest without a ``SELECTION_FIELDS`` entry would
    be listed by GET /models and selectable through POST /pipeline/load, and
    then quietly do nothing — the load payload has no field to put it in.
    """
    for base_id, descriptor in _shipped().items():
        if descriptor.engine_family != "ltx":
            continue
        assert set(descriptor.categories) == set(SELECTION_FIELDS), base_id


def test_shipped_ltx23_asset_paths_match_the_removed_config_fields():
    """The engine adapter asks for its fixed files by asset KEY, so a renamed
    or dropped key is a load-time RuntimeError. Pin both the keys and the paths
    to what ``config.model`` held before P3b removed those fields."""
    descriptor = _shipped()["LTX23"]
    assert descriptor.assets == HISTORICAL_ASSET_PATHS
    # The three the real backend refuses to start without (the pruned decoder is
    # deliberately not among them — a missing one downgrades a job, not the
    # server).
    assert set(REQUIRED_ASSETS) <= set(descriptor.assets)
    assert "component_video_vae_pruned_path" not in REQUIRED_ASSETS


def test_deprecated_model_keys_warn_and_are_ignored(tmp_path, caplog):
    """A config.yaml left over from before P3b still loads (Pydantic ignores
    unknown keys) — but every surviving key is named at WARNING, because a
    value that is read by nothing and reported by nothing is exactly how a
    stale path used to demote the backend to mock unnoticed."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "backend": "mock",
                    "gemma_root": "./models/OLD/tokenizer",
                    "spatial_upsampler_path": "./models/OLD/upsampler.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="ltx.config"):
        cfg = load_config(cfg_path)

    assert cfg.model.backend == "mock"  # the rest of the file still applies
    assert not hasattr(cfg.model, "gemma_root")  # the field is gone for good
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 2
    assert any("gemma_root" in m for m in warned)
    assert any("spatial_upsampler_path" in m for m in warned)
    assert all("記述子" in m for m in warned)


def test_deprecated_key_list_covers_every_removed_field():
    """All eight removed ``model:`` keys are in the warning list — a key left
    out of it is a key that would vanish in silence."""
    assert set(DEPRECATED_MODEL_KEYS) == {
        "gguf_transformer_path",
        "gguf_gemma_path",
        "component_video_vae_path",
        "component_audio_vae_path",
        "component_text_projection_path",
        "component_video_vae_pruned_path",
        "spatial_upsampler_path",
        "gemma_root",
    }
    for key in DEPRECATED_MODEL_KEYS:
        assert not hasattr(AppConfig().model, key)


def _runner(tmp_path, descriptor: dict):
    """An LTXRunner wired to a tmp descriptor + tmp model store."""
    cfg = AppConfig.model_validate(
        {"model": {"backend": "auto", **build_model_layout(tmp_path, [descriptor])}}
    )
    return LTXRunner(cfg, build_low_vram_settings(cfg))


def test_real_available_names_every_missing_file(tmp_path, caplog):
    """§3-97 U3: a False answer here silently demotes generation to the mock
    backend, so it must say WHICH files were missing. The descriptor decides
    what "every file" means (four default_files + three assets)."""
    runner = _runner(tmp_path, base_model_descriptor())  # descriptor declares no assets
    # Point the engine venv/dir at tmp too: this machine really has both, and
    # the test must not depend on that.
    runner.config.model.engine_python = (tmp_path / "venv" / "python.exe").as_posix()
    runner.config.model.engine_dir = (tmp_path / "engine").as_posix()
    with caplog.at_level(logging.WARNING, logger="ltx.runner"):
        assert runner._real_available() is False
    message = "\n".join(r.getMessage() for r in caplog.records)
    assert "real backend unavailable" in message
    for asset in REQUIRED_ASSETS:  # undeclared by this descriptor -> named anyway
        assert asset in message
    assert "engine_python" in message
    assert "worker.py" in message
    # The four category default_files DO exist here (build_model_layout writes
    # them), so they must not be listed as missing.
    assert "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf" not in message


def test_real_available_true_once_the_descriptor_files_all_exist(tmp_path):
    descriptor = base_model_descriptor()
    descriptor["assets"] = {
        "gemma_root": "LTX23/TextEncoder/tokenizer",
        "spatial_upsampler_path": "LTX23/Upscaler/upsampler.safetensors",
        "component_text_projection_path": "LTX23/TextEncoder/text_projection.safetensors",
    }
    runner = _runner(tmp_path, descriptor)
    models_dir = tmp_path / "models"
    (models_dir / descriptor["assets"]["gemma_root"]).mkdir(parents=True)
    for key in ("spatial_upsampler_path", "component_text_projection_path"):
        write_model_file(models_dir / descriptor["assets"][key])
    engine_python = tmp_path / ".venv-engine" / "Scripts" / "python.exe"
    write_model_file(engine_python)
    runner.config.model.engine_python = engine_python.as_posix()
    runner.config.model.engine_dir = (tmp_path / "engine").as_posix()
    write_model_file(tmp_path / "engine" / "worker.py")

    assert runner._real_available() is True


def test_clean_config_warns_about_nothing(tmp_path, caplog):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"model": {"backend": "mock"}}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="ltx.config"):
        load_config(cfg_path)
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
