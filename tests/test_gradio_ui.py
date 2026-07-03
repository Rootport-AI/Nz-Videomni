"""Smoke test for the Gradio UI mount (spec ch.12).

conftest.py forces LTX_DISABLE_GRADIO=1 for every other test so the suite stays
fast/headless -- which means no test ever exercised mount_gradio()/gradio_ui.py.
This file is the one exception: it explicitly re-enables the UI for a single
app build and checks that /ui actually serves the Gradio app, closing that
blind spot.

Uses the same mock backend as the rest of the suite (no GPU, no models).
"""

from __future__ import annotations

import argparse

import pytest
import yaml

import main


def _make_args(config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False, config=config_path,
        te_offload=None,
        dit_cpu_load=None,
    )


@pytest.fixture()
def gradio_client(tmp_path, monkeypatch):
    """Same as conftest's `client` fixture, but with the Gradio UI ENABLED."""
    monkeypatch.delenv("LTX_DISABLE_GRADIO", raising=False)

    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {"backend": "mock"},
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    app = main.build_app(_make_args(cfg_path.as_posix()))

    # TestClient as a context manager runs FastAPI's startup/shutdown events,
    # which is what tears down Gradio's queue/background threads cleanly
    # (without it, a mounted Gradio app can leave threads dangling between
    # tests).
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def test_ui_mounts_and_serves_html(gradio_client):
    # /ui (no trailing slash) 307-redirects to /ui/ -- follow_redirects=True
    # (the TestClient default) exercises both hops in one call.
    r = gradio_client.get("/ui")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "LTX-AviUtl2-Bridge" in r.text


def test_ui_config_endpoint(gradio_client):
    r = gradio_client.get("/ui/config")
    assert r.status_code == 200
    body = r.json()
    # Sanity check: it's a real Gradio Blocks config with a component tree,
    # not an empty/broken app.
    assert "components" in body
    assert len(body["components"]) > 0


def test_root_redirects_to_ui_when_mounted(gradio_client):
    r = gradio_client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/ui"


def test_root_is_json_landing_when_ui_disabled(client):
    # `client` (conftest.py) has LTX_DISABLE_GRADIO=1 -- / must not 404 here.
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["ui"] is None
    assert body["docs"] == "/docs"
