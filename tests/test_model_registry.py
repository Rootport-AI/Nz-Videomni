"""Model-management S1: ModelRegistry (services) + GET /models (API).

All tests are mock/tmp-dir based — no GPU, no real model files. The registry is
descriptor-driven (services/base_models.py), so each test writes a tmp
base-model descriptor and a tmp models tree instead of pointing config fields at
files; the client fixture (conftest) does the same for the whole app.
"""

from __future__ import annotations

import json
import logging

import pytest

from api.errors import APIError
from config import PROJECT_ROOT, AppConfig
from services.base_models import load_base_models
from services.model_registry import (
    CATEGORIES,
    DEFAULT_NAME,
    ModelRegistry,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

#: Category -> default file, relative to the tmp models_dir. Mirrors the shipped
#: LTX 2.3 layout in shape: the transformer default sits directly in its own
#: (recursively scanned) directory, and the two VAEs share one.
LAYOUT_DEFAULTS: dict[str, str] = {
    "transformer": "ltx-gguf/LTX-Q4_K_M.gguf",
    "text_encoder": "gemma-gguf/gemma-Q4_K_M.gguf",
    "video_vae": "components/vae/LTX_video_vae_bf16.safetensors",
    "audio": "components/vae/LTX_audio_vae_bf16.safetensors",
}

LAYOUT_DESCRIPTOR = {
    "schema": 2,
    "id": "LTXTEST",
    "display_name": "LTX Test",
    "engine_family": "ltx",
    "categories": {
        "transformer": {
            "scan": ["ltx-gguf"],
            "extensions": [".gguf"],
            "recursive": True,
            "default_file": LAYOUT_DEFAULTS["transformer"],
        },
        "text_encoder": {
            "scan": ["gemma-gguf"],
            "extensions": [".gguf"],
            "default_file": LAYOUT_DEFAULTS["text_encoder"],
        },
        "video_vae": {
            "scan": ["components/vae"],
            "extensions": [".safetensors"],
            "name_hint": "video",
            "default_file": LAYOUT_DEFAULTS["video_vae"],
        },
        "audio": {
            "scan": ["components/vae"],
            "extensions": [".safetensors"],
            "name_hint": "audio",
            "default_file": LAYOUT_DEFAULTS["audio"],
        },
    },
    "default_selection": {c: DEFAULT_NAME for c in LAYOUT_DEFAULTS},
}


def _touch(path, size: int = 4):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def _write_descriptor(tmp_path, descriptor: dict | None = None):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "10-base.json").write_text(
        json.dumps(descriptor or LAYOUT_DESCRIPTOR), encoding="utf-8"
    )
    return manifest_dir


def _config(tmp_path, descriptor: dict | None = None) -> AppConfig:
    """A config wired to a tmp descriptor + tmp models dir (no real weights)."""
    manifest_dir = _write_descriptor(tmp_path, descriptor)
    return AppConfig.model_validate(
        {
            "model": {
                "backend": "mock",
                "manifest_dir": manifest_dir.as_posix(),
                "models_dir": (tmp_path / "models").as_posix(),
            }
        }
    )


def _config_with_layout(tmp_path) -> AppConfig:
    """:func:`_config` plus the four default weight files on disk."""
    cfg = _config(tmp_path)
    for rel in LAYOUT_DEFAULTS.values():
        _touch(tmp_path / "models" / rel)
    return cfg


# --------------------------------------------------------------------------- #
# default auto-registration
# --------------------------------------------------------------------------- #

def test_default_entry_always_present(tmp_path):
    reg = ModelRegistry(_config_with_layout(tmp_path))
    for cat in CATEGORIES:
        names = reg.names(cat)
        assert names[0] == DEFAULT_NAME
        entries = reg.entries(cat)
        assert entries[0].is_default and entries[0].name == DEFAULT_NAME
        assert entries[0].exists
        assert entries[0].source == "config"


def test_resolve_default_matches_descriptor_default_file(tmp_path):
    """resolve(cat, "default") must be exactly the descriptor's default_file —
    the byte-identical guarantee for the no-selection load."""
    cfg = _config_with_layout(tmp_path)
    reg = ModelRegistry(cfg)
    for category, rel in LAYOUT_DEFAULTS.items():
        assert reg.resolve(category, DEFAULT_NAME) == cfg._abs(
            (tmp_path / "models" / rel).as_posix()
        )


#: The four fixed default paths ``config.model`` held before P3b moved them into
#: the base-model descriptor. A LITERAL transcription on purpose: the point of
#: the test below is that the descriptor still names the very same files, so
#: reading them back off anything descriptor-driven would prove nothing.
HISTORICAL_CONFIG_DEFAULTS: dict[str, str] = {
    "transformer": "./models/LTX23/Weights/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
    "text_encoder": "./models/LTX23/TextEncoder/gemma-3-12b-it-Q4_K_M.gguf",
    "video_vae": "./models/LTX23/VAE/LTX23_video_vae_bf16.safetensors",
    "audio": "./models/LTX23/VAE/LTX23_audio_vae_bf16.safetensors",
}


def test_shipped_descriptor_default_files_match_config_defaults():
    """P3a/P3b invariant: every default_file in the SHIPPED LTX 2.3 descriptor
    is a verbatim transcription of the config default path the worker payload
    used to be built from, so moving first the registry and then the engine
    adapter onto descriptors cannot have changed which file "default" means."""
    cfg = AppConfig()
    descriptor = next(iter(load_base_models(cfg.manifest_dir).values()))
    assert set(descriptor.categories) == set(HISTORICAL_CONFIG_DEFAULTS)
    for category, config_path in HISTORICAL_CONFIG_DEFAULTS.items():
        descriptor_path = cfg.models_dir / descriptor.categories[category].default_file
        assert descriptor_path == cfg._abs(config_path)
        # ...and the stored form still displays as the project-relative path.
        assert descriptor_path.relative_to(PROJECT_ROOT).as_posix() == (
            cfg._abs(config_path).relative_to(PROJECT_ROOT).as_posix()
        )


def test_reserved_default_name_in_config_is_ignored(tmp_path):
    cfg = _config_with_layout(tmp_path)
    other = _touch(tmp_path / "models" / "ltx-gguf" / "other" / "shadow.gguf")
    cfg.model.transformers = {"default": str(other)}
    reg = ModelRegistry(cfg)
    # "default" still resolves to the descriptor's default_file, not the shadow.
    assert reg.resolve("transformer", DEFAULT_NAME) == cfg._abs(
        (tmp_path / "models" / LAYOUT_DEFAULTS["transformer"]).as_posix()
    )


# --------------------------------------------------------------------------- #
# scanning the existing layout
# --------------------------------------------------------------------------- #

def test_scan_discovers_sibling_gguf_recursively(tmp_path):
    cfg = _config_with_layout(tmp_path)
    # A sibling release in its own subdirectory (transformer scan is recursive
    # from the default file's own directory).
    _touch(tmp_path / "models" / "ltx-gguf" / "dev-1.2" / "LTX-dev-Q6_K.gguf")
    reg = ModelRegistry(cfg)
    assert "LTX-dev-Q6_K" in reg.names("transformer")
    entry = {e.name: e for e in reg.entries("transformer")}["LTX-dev-Q6_K"]
    assert entry.source == "scan" and entry.exists


def test_scan_discovers_sibling_gguf_alongside_default(tmp_path):
    """A sibling release placed directly next to the (also top-level) default
    file must also be discovered -- the common case now that the default
    itself is no longer nested one subdirectory deep."""
    cfg = _config_with_layout(tmp_path)
    _touch(tmp_path / "models" / "ltx-gguf" / "LTX-dev-Q8.gguf")
    reg = ModelRegistry(cfg)
    assert "LTX-dev-Q8" in reg.names("transformer")


def test_scan_text_encoder_non_recursive(tmp_path):
    cfg = _config_with_layout(tmp_path)
    _touch(tmp_path / "models" / "gemma-gguf" / "gemma-Q6_K.gguf")
    _touch(tmp_path / "models" / "gemma-gguf" / "sub" / "nested.gguf")  # not scanned
    reg = ModelRegistry(cfg)
    names = reg.names("text_encoder")
    assert "gemma-Q6_K" in names
    assert "nested" not in names


def test_scan_excludes_cache_and_wrong_extensions(tmp_path):
    cfg = _config_with_layout(tmp_path)
    _touch(tmp_path / "models" / "ltx-gguf" / ".cache" / "junk.gguf")
    _touch(tmp_path / "models" / "ltx-gguf" / "notes.txt")
    _touch(tmp_path / "models" / "gemma-gguf" / ".cache" / "junk2.gguf")
    reg = ModelRegistry(cfg)
    assert "junk" not in reg.names("transformer")
    assert "notes" not in reg.names("transformer")
    assert "junk2" not in reg.names("text_encoder")


def test_shared_vae_dir_video_audio_classification(tmp_path):
    cfg = _config_with_layout(tmp_path)
    vae_dir = tmp_path / "models" / "components" / "vae"
    _touch(vae_dir / "custom_video_vae.safetensors")
    _touch(vae_dir / "custom_audio_vae.safetensors")
    _touch(vae_dir / "mystery_weights.safetensors")          # neither hint -> skipped
    _touch(vae_dir / "video_audio_combo.safetensors")        # both hints -> skipped
    reg = ModelRegistry(cfg)
    video_names = reg.names("video_vae")
    audio_names = reg.names("audio")
    assert "custom_video_vae" in video_names
    assert "custom_video_vae" not in audio_names
    assert "custom_audio_vae" in audio_names
    assert "custom_audio_vae" not in video_names
    for skipped in ("mystery_weights", "video_audio_combo"):
        assert skipped not in video_names and skipped not in audio_names


def test_shared_vae_dir_scan_is_silent_about_the_sibling_category(tmp_path, caplog):
    """The healthy two-file layout must log NOTHING.

    video_vae and audio share one scan directory, so each necessarily walks
    past the other's file. That is the design working, not something the owner
    can act on — the startup log must not mention it (see _scan_category)."""
    cfg = _config_with_layout(tmp_path)
    vae_dir = tmp_path / "models" / "components" / "vae"
    _touch(vae_dir / "custom_video_vae.safetensors")
    _touch(vae_dir / "custom_audio_vae.safetensors")
    with caplog.at_level(logging.INFO, logger="ltx.models"):
        ModelRegistry(cfg)
    assert [r for r in caplog.records if "not classifiable" in r.getMessage()] == []


def test_scan_reports_a_file_no_category_can_classify_once(tmp_path, caplog):
    """A file NEITHER hint claims really is invisible until it is registered,
    so it still gets its notice — exactly one, not one per category that
    shares the directory."""
    cfg = _config_with_layout(tmp_path)
    vae_dir = tmp_path / "models" / "components" / "vae"
    _touch(vae_dir / "mystery_weights.safetensors")
    with caplog.at_level(logging.INFO, logger="ltx.models"):
        ModelRegistry(cfg)
    notices = [r for r in caplog.records if "not classifiable" in r.getMessage()]
    assert len(notices) == 1
    assert "mystery_weights.safetensors" in notices[0].getMessage()


def test_unclassifiable_file_exposed_via_explicit_config(tmp_path):
    """Config registration is the authoritative override for files the
    filename heuristic cannot classify (design ruling §9-2)."""
    cfg = _config_with_layout(tmp_path)
    mystery = _touch(
        tmp_path / "models" / "components" / "vae" / "mystery_weights.safetensors"
    )
    cfg.model.video_vaes = {"mystery": str(mystery)}
    reg = ModelRegistry(cfg)
    assert "mystery" in reg.names("video_vae")
    assert reg.resolve("video_vae", "mystery") == mystery


def test_scan_does_not_duplicate_default_or_config_entries(tmp_path):
    cfg = _config_with_layout(tmp_path)
    extra = _touch(tmp_path / "models" / "gemma-gguf" / "gemma-Q8.gguf")
    cfg.model.text_encoders = {"q8": str(extra)}
    reg = ModelRegistry(cfg)
    names = reg.names("text_encoder")
    # Both files are on disk in the scan dir, but each is already covered by
    # the default / the explicit registration -> exactly one name per file.
    assert names == [DEFAULT_NAME, "q8"]


def test_rescan_picks_up_new_files_without_rebuild(tmp_path):
    cfg = _config_with_layout(tmp_path)
    reg = ModelRegistry(cfg)
    assert "late-arrival" not in reg.names("text_encoder")
    _touch(tmp_path / "models" / "gemma-gguf" / "late-arrival.gguf")
    reg.rescan()
    assert "late-arrival" in reg.names("text_encoder")


def test_missing_scan_dirs_are_tolerated(tmp_path):
    """Defaults pointing at absent files/dirs (fresh checkout without models/)
    must not crash the registry — the default entry just reports exists=False."""
    cfg = _config(tmp_path)  # descriptor written, but no models tree at all
    reg = ModelRegistry(cfg)
    for cat in CATEGORIES:
        entries = reg.entries(cat)
        assert entries[0].name == DEFAULT_NAME
        assert entries[0].exists is False


# --------------------------------------------------------------------------- #
# resolve failure modes (fail loud, lora_registry precedent)
# --------------------------------------------------------------------------- #

def test_resolve_unknown_name_raises_model_not_found(tmp_path):
    reg = ModelRegistry(_config_with_layout(tmp_path))
    with pytest.raises(APIError) as ei:
        reg.resolve("transformer", "no-such-model")
    assert ei.value.code == "MODEL_NOT_FOUND" and ei.value.status_code == 404


def test_resolve_unknown_category_raises_model_not_found(tmp_path):
    reg = ModelRegistry(_config_with_layout(tmp_path))
    with pytest.raises(APIError) as ei:
        reg.resolve("upscaler", DEFAULT_NAME)
    assert ei.value.code == "MODEL_NOT_FOUND"


def test_resolve_pathlike_name_rejected(tmp_path):
    reg = ModelRegistry(_config_with_layout(tmp_path))
    for bad in ("../evil", "sub/model", "c:\\evil.gguf"):
        with pytest.raises(APIError) as ei:
            reg.resolve("transformer", bad)
        assert ei.value.code == "MODEL_NOT_FOUND"


def test_resolve_registered_but_missing_file(tmp_path):
    cfg = _config_with_layout(tmp_path)
    cfg.model.transformers = {"ghost": str(tmp_path / "models" / "ltx-gguf" / "gone.gguf")}
    reg = ModelRegistry(cfg)
    with pytest.raises(APIError) as ei:
        reg.resolve("transformer", "ghost")
    assert ei.value.code == "MODEL_FILE_MISSING" and ei.value.status_code == 422


def test_display_path_never_leaks_absolute_outside_project(tmp_path):
    reg = ModelRegistry(_config_with_layout(tmp_path))
    for cat in CATEGORIES:
        for entry in reg.entries(cat):
            # tmp_path is outside PROJECT_ROOT -> display must reduce to a
            # filename, never the raw absolute path.
            assert str(tmp_path) not in entry.path


# --------------------------------------------------------------------------- #
# GET /models endpoint (client fixture: mock backend, temp config)
# --------------------------------------------------------------------------- #

def test_get_models_shape_and_defaults(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    body = r.json()
    cats = body["categories"]
    assert set(cats) == {"transformer", "text_encoder", "video_vae", "audio"}
    for block in cats.values():
        assert block["default"] == DEFAULT_NAME
        assert block["active"] == DEFAULT_NAME  # nothing explicitly loaded yet
        entries = block["entries"]
        assert entries[0]["name"] == DEFAULT_NAME
        assert entries[0]["is_default"] is True
        assert set(entries[0]) == {"name", "path", "is_default", "exists", "source"}


def test_get_config_gains_only_additive_registry_keys(client):
    """GET /config keeps working and the 4 new model dict keys are additive
    (empty by default) — no existing key is removed or reshaped."""
    r = client.get("/api/v1/config")
    assert r.status_code == 200
    model = r.json()["model"]
    for key in ("transformers", "text_encoders", "video_vaes", "audio_models"):
        assert model[key] == {}
    # Spot-check pre-existing frozen keys are still present. The fixed default
    # WEIGHT paths are deliberately no longer among them (P3b moved them into
    # the base-model descriptors); what stays is the two DIRECTORIES that say
    # where to look, plus the registries and the backend switch.
    for key in ("manifest_dir", "models_dir", "ic_loras", "backend"):
        assert key in model
