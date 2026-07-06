"""S1: GET/POST /loras + GET /loras/{name}/thumbnail (mock backend, no GPU).

Exercises the additive enumeration endpoints end to end against a temp lora_dir
holding synthetic safetensors headers (pure Python) — a style LoRA (with a
thumbnail), a control LoRA (detected via reference_downscale_factor metadata),
and a broken file (dropped from the scan).
"""

from __future__ import annotations

import argparse
import json
import struct

import pytest
import yaml
from fastapi.testclient import TestClient

import main


def _write_safetensors(path, metadata=None, data_bytes=16):
    header: dict = {}
    if metadata is not None:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    blob = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * data_bytes)
    return path


def _make_args(config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=config_path, te_offload=None, dit_cpu_load=None,
    )


@pytest.fixture()
def loras_client(tmp_path):
    lora_dir = tmp_path / "loras"
    _write_safetensors(
        lora_dir / "Pixar_Toon.safetensors",
        metadata={"ss_network_alpha": "16", "ss_network_dim": "32"},
    )
    (lora_dir / "Pixar_Toon.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # thumbnail
    _write_safetensors(
        lora_dir / "union-ctrl.safetensors",
        metadata={"reference_downscale_factor": "2"},
    )
    (lora_dir / "broken.safetensors").write_bytes(b"\x00" * 8)  # dropped from scan

    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {"backend": "mock", "lora_dir": lora_dir.as_posix()},
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    app = main.build_app(_make_args(cfg_path.as_posix()))
    with TestClient(app) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        yield c


def test_get_loras_shape(loras_client):
    r = loras_client.get("/api/v1/loras")
    assert r.status_code == 200, r.text
    loras = {e["name"]: e for e in r.json()["loras"]}
    assert set(loras) == {"Pixar_Toon", "union-ctrl"}  # broken file excluded
    for e in loras.values():
        assert set(e) == {"name", "kind", "has_thumbnail", "exists", "source"}
    assert loras["Pixar_Toon"]["kind"] == "style"
    assert loras["Pixar_Toon"]["has_thumbnail"] is True
    assert loras["Pixar_Toon"]["source"] == "scan"
    assert loras["union-ctrl"]["kind"] == "control"
    assert loras["union-ctrl"]["has_thumbnail"] is False


def test_reload_counts(loras_client):
    r = loras_client.post("/api/v1/loras/reload")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"total": 2, "styles": 1, "controls": 1}


def test_thumbnail_served(loras_client):
    r = loras_client.get("/api/v1/loras/Pixar_Toon/thumbnail")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")


def test_thumbnail_missing_404(loras_client):
    r = loras_client.get("/api/v1/loras/union-ctrl/thumbnail")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "LORA_THUMBNAIL_NOT_FOUND"


def test_thumbnail_unknown_name_404(loras_client):
    r = loras_client.get("/api/v1/loras/no-such-lora/thumbnail")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "LORA_NOT_FOUND"


def test_reload_picks_up_new_drop_in(loras_client):
    ctx = loras_client.app_context
    lora_dir = ctx.config._abs(ctx.config.model.lora_dir)
    _write_safetensors(lora_dir / "Late_Style.safetensors", metadata={"x": "1"})
    r = loras_client.post("/api/v1/loras/reload")
    assert r.json()["total"] == 3
    names = {e["name"] for e in loras_client.get("/api/v1/loras").json()["loras"]}
    assert "Late_Style" in names
