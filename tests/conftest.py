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
    )


@pytest.fixture()
def client(tmp_path):
    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {"backend": "mock"},  # deterministic: tests never hit the real pipeline
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
def png_bytes() -> bytes:
    img = Image.new("RGB", (64, 48), (40, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
