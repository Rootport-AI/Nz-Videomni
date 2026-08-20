"""Shared pytest fixtures: a TestClient backed by a temp-dir config.

The TestClient runs FastAPI background tasks synchronously, so a POST /generate
has already produced output.mp4 + metadata.json by the time the call returns.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

os.environ["LTX_DISABLE_GRADIO"] = "1"  # keep tests fast / headless

import main  # noqa: E402  (import after env set)


def _make_args(config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False, config=config_path,
        te_offload=None,
        dit_cpu_load=None,
    )


# --------------------------------------------------------------------------- #
# base-model descriptors + model layout, in a temp dir
#
# The registry is descriptor-driven (services/base_models.py) and resolves every
# descriptor path against ``model.models_dir``. Left at their defaults, tests
# would read the REPOSITORY's scripts/manifests and scan the developer's real
# models/ tree -- making results depend on which weights that machine happens to
# have downloaded. Both are therefore pointed at tmp_path, with a descriptor
# that mirrors the shipped LTX 2.3 one in shape (same category names, same
# scan/extension/name_hint wiring) so the app under test behaves exactly like
# production while staying hermetic.
# --------------------------------------------------------------------------- #

#: Category -> default weight file, relative to the base model's own directory.
TEST_DEFAULT_FILES: dict[str, str] = {
    "transformer": "Weights/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
    "text_encoder": "TextEncoder/gemma-3-12b-it-Q4_K_M.gguf",
    "video_vae": "VAE/LTX23_video_vae_bf16.safetensors",
    "audio": "VAE/LTX23_audio_vae_bf16.safetensors",
}

TEST_BASE_MODEL_ID = "LTX23"


def default_files(base_id: str = TEST_BASE_MODEL_ID) -> dict[str, str]:
    """Category -> default weight file, relative to ``models_dir``."""
    return {c: f"{base_id}/{rel}" for c, rel in TEST_DEFAULT_FILES.items()}

_GGUF_STUB = b"GGUF" + b"\0" * 16
_SAFETENSORS_STUB = struct.pack("<Q", 2) + b"{}"


def write_model_file(path: Path) -> Path:
    """Create a stub weight file whose first bytes pass ``precheck_model_file``
    (GGUF magic / a parsable safetensors header) -- no weights, no size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_GGUF_STUB if path.suffix == ".gguf" else _SAFETENSORS_STUB)
    return path


def base_model_descriptor(base_id: str = TEST_BASE_MODEL_ID) -> dict:
    """The shipped LTX 2.3 descriptor's shape, with its four categories."""
    defaults = default_files(base_id)
    return {
        "schema": 2,
        "id": base_id,
        "display_name": "LTX 2.3",
        "engine_family": "ltx",
        "categories": {
            "transformer": {
                "scan": [f"{base_id}/Weights"],
                "extensions": [".gguf"],
                "recursive": True,
                "default_file": defaults["transformer"],
            },
            "text_encoder": {
                "scan": [f"{base_id}/TextEncoder"],
                "extensions": [".gguf"],
                "default_file": defaults["text_encoder"],
            },
            "video_vae": {
                "scan": [f"{base_id}/VAE"],
                "extensions": [".safetensors"],
                "name_hint": "video",
                "default_file": defaults["video_vae"],
            },
            "audio": {
                "scan": [f"{base_id}/VAE"],
                "extensions": [".safetensors"],
                "name_hint": "audio",
                "default_file": defaults["audio"],
            },
        },
        "default_selection": {c: "default" for c in TEST_DEFAULT_FILES},
    }


def write_manifests(manifest_dir: Path, descriptors: list[dict] | None = None) -> Path:
    """Write ``<index>-<id>.json`` descriptors (BOM-less on purpose: the loader
    reads utf-8-sig, which handles both) and return the directory."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for index, descriptor in enumerate(descriptors or [base_model_descriptor()]):
        name = f"{10 * (index + 1)}-{descriptor['id'].lower()}.json"
        (manifest_dir / name).write_text(
            json.dumps(descriptor, indent=2), encoding="utf-8"
        )
    return manifest_dir


def build_model_layout(tmp_path, descriptors: list[dict] | None = None) -> dict:
    """Create tmp manifests + the default weight files of the FIRST descriptor,
    and return the ``model`` config fragment pointing at both."""
    manifest_dir = write_manifests(tmp_path / "manifests", descriptors)
    models_dir = tmp_path / "models"
    first = (descriptors or [base_model_descriptor()])[0]
    for spec in first["categories"].values():
        if spec.get("default_file"):
            write_model_file(models_dir / spec["default_file"])
    return {
        "manifest_dir": manifest_dir.as_posix(),
        "models_dir": models_dir.as_posix(),
    }


def _build_app(tmp_path):
    """Build a ``main.build_app`` FastAPI app wired to a temp-dir config.

    Extracted from the ``client`` fixture (below) so ``mcp_app`` can share the
    exact same app construction without duplicating it -- 15+ test files
    depend on ``client``'s observable behavior staying byte-identical, so this
    extraction must not change anything about it (see MEMORY.md: aux2/conftest
    extraction lessons).
    """
    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {
            "backend": "mock",  # deterministic: tests never hit the real pipeline
            **build_model_layout(tmp_path),  # hermetic descriptors + models dir
        },
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return main.build_app(_make_args(cfg_path.as_posix()))


@pytest.fixture()
def client(tmp_path):
    app = _build_app(tmp_path)
    with TestClient(app) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        yield c


@pytest.fixture()
def mcp_app(tmp_path):
    """Same app as ``client``, for MCP tool tests that talk to it via
    ``httpx.ASGITransport`` instead of ``TestClient`` (the MCP tools use an
    async ``httpx.AsyncClient`` throughout, so an async-native transport keeps
    the test path closer to production than driving the sync TestClient would).

    Yields ``(app, output_dir_path)`` -- the output dir is a ``pathlib.Path``
    tests can inspect directly.
    """
    app = _build_app(tmp_path)
    with TestClient(app):  # runs startup/shutdown events, same as `client`
        yield app, (tmp_path / "outputs")


@pytest.fixture()
def png_bytes() -> bytes:
    img = Image.new("RGB", (64, 48), (40, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
