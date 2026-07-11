"""Model-management S1: ModelRegistry (services) + GET /models (API).

All tests are mock/tmp-dir based — no GPU, no real model files. The client
fixture (conftest) boots the app with backend=mock against a temp config, so
GET /models exercises the real registry against the (absent) default layout.
"""

from __future__ import annotations

import pytest

from api.errors import APIError
from config import AppConfig
from services.model_registry import (
    CATEGORIES,
    DEFAULT_NAME,
    ModelRegistry,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _touch(path, size: int = 4):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def _config_with_layout(tmp_path) -> AppConfig:
    """A config whose 4 default paths point into a tmp models layout that
    mirrors the real one (transformer default directly in its models dir,
    sibling releases in subdirectories; video/audio VAEs sharing one
    directory)."""
    gguf_dir = tmp_path / "models" / "ltx-gguf"
    default_tr = _touch(gguf_dir / "LTX-Q4_K_M.gguf")
    te_dir = tmp_path / "models" / "gemma-gguf"
    default_te = _touch(te_dir / "gemma-Q4_K_M.gguf")
    vae_dir = tmp_path / "models" / "components" / "vae"
    default_vv = _touch(vae_dir / "LTX_video_vae_bf16.safetensors")
    default_au = _touch(vae_dir / "LTX_audio_vae_bf16.safetensors")
    return AppConfig.model_validate(
        {
            "model": {
                "backend": "mock",
                "gguf_transformer_path": str(default_tr),
                "gguf_gemma_path": str(default_te),
                "component_video_vae_path": str(default_vv),
                "component_audio_vae_path": str(default_au),
            }
        }
    )


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


def test_resolve_default_matches_config_field(tmp_path):
    """resolve(cat, "default") must be exactly the config default path — the
    byte-identical guarantee for the no-selection load."""
    cfg = _config_with_layout(tmp_path)
    reg = ModelRegistry(cfg)
    assert reg.resolve("transformer", DEFAULT_NAME) == cfg._abs(
        cfg.model.gguf_transformer_path
    )
    assert reg.resolve("text_encoder", DEFAULT_NAME) == cfg._abs(cfg.model.gguf_gemma_path)
    assert reg.resolve("video_vae", DEFAULT_NAME) == cfg._abs(
        cfg.model.component_video_vae_path
    )
    assert reg.resolve("audio", DEFAULT_NAME) == cfg._abs(
        cfg.model.component_audio_vae_path
    )


def test_reserved_default_name_in_config_is_ignored(tmp_path):
    cfg = _config_with_layout(tmp_path)
    other = _touch(tmp_path / "models" / "ltx-gguf" / "other" / "shadow.gguf")
    cfg.model.transformers = {"default": str(other)}
    reg = ModelRegistry(cfg)
    # "default" still resolves to the fixed default field, not the shadow.
    assert reg.resolve("transformer", DEFAULT_NAME) == cfg._abs(
        cfg.model.gguf_transformer_path
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
    cfg = AppConfig.model_validate({"model": {"backend": "mock"}})
    reg = ModelRegistry(cfg)
    for cat in CATEGORIES:
        entries = reg.entries(cat)
        assert entries[0].name == DEFAULT_NAME


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
    # Spot-check pre-existing frozen keys are still present.
    for key in ("gguf_transformer_path", "gguf_gemma_path", "ic_loras", "backend"):
        assert key in model
