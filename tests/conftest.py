"""Shared pytest fixtures: a TestClient backed by a temp-dir config.

The TestClient runs FastAPI background tasks synchronously, so a POST /generate
has already produced output.mp4 + metadata.json by the time the call returns.
"""

from __future__ import annotations

import argparse
import io
import os

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
        "model": {"backend": "mock"},  # deterministic: tests never hit the real pipeline
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
