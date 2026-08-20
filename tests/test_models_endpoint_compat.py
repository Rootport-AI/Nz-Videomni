"""GET /models: the legacy ``categories`` block did NOT change when the
registry became descriptor-driven and the response gained a base-model layer
(multi-engine P3a, Docs/MULTI_ENGINE_DESIGN.md §5.1).

The expected value below was CAPTURED FROM THE PRE-CHANGE SERVER (commit
13c3437, the CATEGORY_SPECS registry driven by config's fixed default paths)
against exactly the models tree built by ``_build_layout``. It is therefore a
before/after comparison, not a restatement of what the new code happens to do:
every key, every order, every path string and every flag has to come out the
same for a client that only knows the two-layer shape.

The layout is deliberately awkward on purpose — a recursive transformer scan
with a sibling release in a subdirectory, a NON-recursive text-encoder scan with
a nested file that must stay invisible, two categories sharing one VAE
directory and separated only by the filename hint, plus a .cache/ directory and
a wrong-extension file that must both be skipped — so an accidental change in
any of the scan rules would move at least one entry.
"""

from __future__ import annotations

import argparse
import json

import yaml
from fastapi.testclient import TestClient

import main
from conftest import base_model_descriptor, write_model_file

# --------------------------------------------------------------------------- #
# the fixed tmp layout
# --------------------------------------------------------------------------- #

DEFAULT_TRANSFORMER = "LTX23/Weights/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
DEFAULT_TEXT_ENCODER = "LTX23/TextEncoder/gemma-3-12b-it-Q4_K_M.gguf"
DEFAULT_VIDEO_VAE = "LTX23/VAE/LTX23_video_vae_bf16.safetensors"
DEFAULT_AUDIO = "LTX23/VAE/LTX23_audio_vae_bf16.safetensors"

LAYOUT_FILES = (
    DEFAULT_TRANSFORMER,
    "LTX23/Weights/dev-1.2/LTX-dev-Q6_K.gguf",   # recursive sibling -> listed
    "LTX23/Weights/put_GGUF_here.txt",           # wrong extension -> skipped
    "LTX23/Weights/.cache/junk.gguf",            # dot-dir -> skipped
    DEFAULT_TEXT_ENCODER,
    "LTX23/TextEncoder/gemma-3-12b-it-Q6_K.gguf",  # sibling -> listed
    "LTX23/TextEncoder/sub/nested.gguf",           # non-recursive -> skipped
    DEFAULT_VIDEO_VAE,
    DEFAULT_AUDIO,
    "LTX23/VAE/custom_video_vae.safetensors",    # hint "video" -> video_vae
    "LTX23/VAE/custom_audio_vae.safetensors",    # hint "audio" -> audio
    "LTX23/VAE/mystery_weights.safetensors",     # neither hint -> skipped
)


def _entry(name: str, path: str, is_default: bool, source: str) -> dict:
    return {
        "name": name,
        "path": path,
        "is_default": is_default,
        "exists": True,
        "source": source,
    }


#: Captured from the pre-P3a server. Paths are bare filenames because the tmp
#: models tree lives outside PROJECT_ROOT (``_display_path`` never leaks an
#: absolute path); the project-relative form is covered by
#: ``test_model_registry.test_shipped_descriptor_default_files_match_config_defaults``.
EXPECTED_CATEGORIES = {
    "transformer": {
        "default": "default",
        "active": "default",
        "entries": [
            _entry("default", "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf", True, "config"),
            _entry("LTX-dev-Q6_K", "LTX-dev-Q6_K.gguf", False, "scan"),
        ],
    },
    "text_encoder": {
        "default": "default",
        "active": "default",
        "entries": [
            _entry("default", "gemma-3-12b-it-Q4_K_M.gguf", True, "config"),
            _entry("gemma-3-12b-it-Q6_K", "gemma-3-12b-it-Q6_K.gguf", False, "scan"),
        ],
    },
    "video_vae": {
        "default": "default",
        "active": "default",
        "entries": [
            _entry("default", "LTX23_video_vae_bf16.safetensors", True, "config"),
            _entry("custom_video_vae", "custom_video_vae.safetensors", False, "scan"),
        ],
    },
    "audio": {
        "default": "default",
        "active": "default",
        "entries": [
            _entry("default", "LTX23_audio_vae_bf16.safetensors", True, "config"),
            _entry("custom_audio_vae", "custom_audio_vae.safetensors", False, "scan"),
        ],
    },
}


def _build_layout(tmp_path):
    models_dir = tmp_path / "models"
    for rel in LAYOUT_FILES:
        write_model_file(models_dir / rel)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "10-ltx23.json").write_text(
        json.dumps(base_model_descriptor()), encoding="utf-8"
    )
    return manifest_dir, models_dir


def _client(tmp_path, manifest_dir, models_dir) -> TestClient:
    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {
            "backend": "mock",
            "manifest_dir": manifest_dir.as_posix(),
            "models_dir": models_dir.as_posix(),
        },
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
        # §3-97 P5: the runtime-state file, in tmp like every other
        # writable location -- never the repository's own state.json.
        "state_file": (tmp_path / "state.json").as_posix(),
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    args = argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=cfg_path.as_posix(), te_offload=None, dit_cpu_load=None,
    )
    return TestClient(main.build_app(args))


# --------------------------------------------------------------------------- #
# the regression
# --------------------------------------------------------------------------- #

def test_categories_block_is_unchanged_by_the_base_model_layer(tmp_path):
    manifest_dir, models_dir = _build_layout(tmp_path)
    with _client(tmp_path, manifest_dir, models_dir) as c:
        body = c.get("/api/v1/models").json()

    assert body["categories"] == EXPECTED_CATEGORIES
    # Key ORDER matters too: the dropdown order clients render is the JSON
    # order, and it must stay transformer -> text_encoder -> video_vae -> audio.
    assert list(body["categories"]) == list(EXPECTED_CATEGORIES)
    for category, block in body["categories"].items():
        assert list(block) == ["default", "active", "entries"]
        for entry, expected in zip(block["entries"], EXPECTED_CATEGORIES[category]["entries"]):
            assert list(entry) == list(expected)


def test_base_model_layer_is_purely_additive(tmp_path):
    manifest_dir, models_dir = _build_layout(tmp_path)
    with _client(tmp_path, manifest_dir, models_dir) as c:
        body = c.get("/api/v1/models").json()

    assert set(body) == {"categories", "active_base_model", "base_models"}
    assert body["active_base_model"] == "LTX23"
    (base,) = body["base_models"]
    assert base["id"] == "LTX23"
    assert base["display_name"] == "LTX 2.3"
    assert base["engine_family"] == "ltx"
    assert base["active"] is True
    assert base["installed"] is True and base["present"] is True
    assert base["missing_categories"] == []
    # The active base model's own three-layer listing is the same content as
    # the legacy block (that is what "the legacy block describes the active
    # base model" means).
    assert base["categories"] == body["categories"]


def test_category_display_order_comes_from_the_descriptor_as_an_array(tmp_path):
    """The dropdown order the WebUI renders is the DESCRIPTOR's declaration
    order, published as ``category_order`` — an array, because object key order
    is not something a client can rely on after transport (the WebUI reaches
    this response through the AviUtl2 plugin's WebView2 message channel, and on
    2026-08-20 the Settings dropdowns came out alphabetised there).

    The second descriptor declares its categories in a deliberately odd order —
    neither alphabetical nor the ``CATEGORIES`` literal's — so an order taken
    from anywhere but the descriptor fails here.
    """
    manifest_dir, models_dir = _build_layout(tmp_path)
    scrambled = base_model_descriptor("LTX25")
    scrambled["display_name"] = "LTX 2.5"
    declared = ["audio", "video_vae", "transformer", "text_encoder"]
    scrambled["categories"] = {c: scrambled["categories"][c] for c in declared}
    (manifest_dir / "20-ltx25.json").write_text(json.dumps(scrambled), encoding="utf-8")

    with _client(tmp_path, manifest_dir, models_dir) as c:
        body = c.get("/api/v1/models").json()

    ltx23, ltx25 = body["base_models"]
    assert ltx25["category_order"] == declared
    assert list(ltx25["categories"]) == declared
    # ...and the first descriptor keeps ITS own order, which is the shipped
    # exchange-frequency one (see test_base_model_contract.py for the pin).
    assert ltx23["category_order"] == ["transformer", "text_encoder", "video_vae", "audio"]
    assert list(ltx23["categories"]) == ltx23["category_order"]


def test_partly_installed_base_model_is_listed_as_such(tmp_path):
    """A second base model whose weights are not downloaded stays listed, with
    installed=False and the categories that are missing named."""
    manifest_dir, models_dir = _build_layout(tmp_path)
    second = base_model_descriptor("LTX25")
    second["display_name"] = "LTX 2.5"
    # Only the transformer default is published for this one.
    for category, spec in second["categories"].items():
        if category != "transformer":
            spec.pop("default_file", None)
    (manifest_dir / "20-ltx25.json").write_text(json.dumps(second), encoding="utf-8")
    write_model_file(models_dir / second["categories"]["transformer"]["default_file"])

    with _client(tmp_path, manifest_dir, models_dir) as c:
        body = c.get("/api/v1/models").json()

    # The active base (and therefore the legacy block) is still the FIRST one.
    assert body["active_base_model"] == "LTX23"
    assert body["categories"] == EXPECTED_CATEGORIES
    assert [b["id"] for b in body["base_models"]] == ["LTX23", "LTX25"]
    ltx25 = body["base_models"][1]
    assert ltx25["active"] is False
    assert ltx25["installed"] is False and ltx25["present"] is True
    assert ltx25["missing_categories"] == ["text_encoder", "video_vae", "audio"]
    # Its own transformer listing resolves against ITS scan root, and every
    # category still offers the reserved "default" name.
    assert ltx25["categories"]["transformer"]["entries"][0]["exists"] is True
    assert ltx25["categories"]["audio"]["entries"][0] == {
        "name": "default",
        "path": "",
        "is_default": True,
        "exists": False,
        "source": "config",
    }
