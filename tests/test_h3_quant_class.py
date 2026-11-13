"""台帳 §1-286: the kind of the selected H3 transformer's weights
(``h3_format.quant_class``, ``PipelineManager.identify_quant_class``, GET /models
``base_models[].transformer_quant_class``).

The kind is read from the DiT header when the selection changes -- the success
path of a load/reload, and once at startup for the restored selection -- and
GET /models only reads the recorded value. An unload keeps it, an LTX load
clears it, and a header that cannot be read is a WARNING with ``null``, never a
failed load.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest
import yaml
from fastapi.testclient import TestClient

import h3_format
import main
from conftest import (
    _make_args,
    base_model_descriptor,
    build_model_layout,
    h3_base_model_descriptor,
    write_h3_model_files,
    write_manifests,
)
from services.runtime_state import SCHEMA_VERSION
from test_h3_format import WIDE, _dit_tensors, _write


def _all(fmt):
    """fmt_of: every block Linear in ``fmt``, the refiner bf16."""
    return lambda layer: fmt if layer.startswith("blocks.") else None


def _mixed(layer):
    """int8 attention + w4a8 MLP in the blocks (the shape of Eros), refiner bf16."""
    if not layer.startswith("blocks."):
        return None
    return "w4a8" if ".mlp." in layer else "int8"


#: name -> (tensors of the DiT, the kind it must report).
DITS = {
    "int8": (lambda: _dit_tensors(), "int8"),
    "fp8": (lambda: _dit_tensors(layers=2, refiner=1, fmt_of=_all("fp8"), **WIDE), "fp8"),
    "w4a8": (lambda: _dit_tensors(layers=2, refiner=1, fmt_of=_all("w4a8"), **WIDE), "w4a8"),
    "mixed": (lambda: _dit_tensors(layers=2, refiner=1, fmt_of=_mixed, **WIDE), "w4a8"),
    "unquantized": (lambda: _dit_tensors(quant=False), "int8"),
}


@contextmanager
def _client(tmp_path, monkeypatch, dit_tensors=None, *, start_on_h3=False):
    """LTX 2.3 and MiniMax H3 on the mock backend (as ``h3_known_client``
    writes them), the H3 default DiT replaced by ``dit_tensors`` when given.
    ``start_on_h3`` writes a state file that restores MiniMax H3 at startup."""
    ltx23 = base_model_descriptor()
    h3 = h3_base_model_descriptor()
    fragment = build_model_layout(tmp_path, [ltx23, h3])
    models_dir = tmp_path / "models"
    write_h3_model_files(models_dir, h3, monkeypatch)
    if dit_tensors is not None:
        _write(models_dir / h3["categories"]["transformer"]["default_file"], dit_tensors)
    write_manifests(tmp_path / "manifests", [ltx23, h3])

    state_path = tmp_path / "state.json"
    if start_on_h3:
        state_path.write_text(
            json.dumps({"schema": SCHEMA_VERSION, "active_base_model": "MiniMaxH3", "selections": {}}),
            encoding="utf-8",
        )
    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {"backend": "mock", **fragment},
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
        "state_file": state_path.as_posix(),
        "tracking": {"backend": "mock"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    app = main.build_app(_make_args(cfg_path.as_posix()))
    with TestClient(app) as client:
        client.app_context = app.state.context  # type: ignore[attr-defined]
        yield client


def _kind(client, base_id: str):
    body = client.get("/api/v1/models").json()
    return next(b for b in body["base_models"] if b["id"] == base_id)["transformer_quant_class"]


def _load(client, **body) -> None:
    r = client.post("/api/v1/pipeline/load", json=body)
    assert r.status_code == 200, r.text


def test_mixed_file_really_mixes_int8_and_w4a8(tmp_path):
    _, cfg = h3_format.inspect_h3_dit(_write(tmp_path / "dit.safetensors", DITS["mixed"][0]()))
    assert {q.format for q in cfg.quant.values()} == {h3_format.INT8, h3_format.W4A8}
    assert h3_format.quant_class(cfg) == "w4a8"


@pytest.mark.parametrize("name", list(DITS))
def test_load_reports_the_kind(tmp_path, monkeypatch, name):
    tensors, expected = DITS[name]
    with _client(tmp_path, monkeypatch, tensors()) as client:
        assert _kind(client, "MiniMaxH3") is None  # LTX is active at start
        assert _kind(client, "LTX23") is None
        _load(client, base_model="MiniMaxH3")
        assert _kind(client, "MiniMaxH3") == expected
        assert _kind(client, "LTX23") is None  # non-active base models stay null


def test_restored_selection_reports_the_kind_before_any_load(tmp_path, monkeypatch):
    tensors, _ = DITS["w4a8"]
    with _client(tmp_path, monkeypatch, tensors(), start_on_h3=True) as client:
        pm = client.app_context.pipeline_manager
        assert pm.active_base_model == "MiniMaxH3"
        assert not pm.loaded
        assert _kind(client, "MiniMaxH3") == "w4a8"


def test_restored_selection_without_the_file_is_null(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, start_on_h3=True) as client:
        pm = client.app_context.pipeline_manager
        assert pm.transformer_quant_class == "int8"
        dit = tmp_path / "models" / h3_base_model_descriptor()["categories"]["transformer"]["default_file"]
        dit.unlink()
        pm.identify_quant_class()
        assert _kind(client, "MiniMaxH3") is None


def test_unload_keeps_the_kind_and_an_ltx_load_clears_it(tmp_path, monkeypatch):
    tensors, _ = DITS["fp8"]
    with _client(tmp_path, monkeypatch, tensors()) as client:
        _load(client, base_model="MiniMaxH3")
        r = client.post("/api/v1/pipeline/unload")
        assert r.status_code == 200, r.text
        assert _kind(client, "MiniMaxH3") == "fp8"

        _load(client, base_model="LTX23")
        assert client.app_context.pipeline_manager.transformer_quant_class is None
        assert _kind(client, "LTX23") is None
        assert _kind(client, "MiniMaxH3") is None


def test_unreadable_header_is_a_warning_and_does_not_fail_the_load(tmp_path, monkeypatch, caplog):
    with _client(tmp_path, monkeypatch, start_on_h3=True) as client:
        pm = client.app_context.pipeline_manager
        assert pm.transformer_quant_class == "int8"

        def broken(path):
            raise h3_format.H3FormatError("broken header")

        monkeypatch.setattr(h3_format, "inspect_h3_dit", broken)
        pm.load()
        assert pm.loaded
        assert _kind(client, "MiniMaxH3") is None
        assert any("重みの形式の種類を判定できませんでした" in r.getMessage() for r in caplog.records)


def test_models_never_reads_the_file(tmp_path, monkeypatch):
    tensors, _ = DITS["w4a8"]
    with _client(tmp_path, monkeypatch, tensors()) as client:
        _load(client, base_model="MiniMaxH3")
        calls: list[str] = []

        def counting(path):
            calls.append(path)
            raise AssertionError("GET /models must not read the header")

        monkeypatch.setattr(h3_format, "inspect_h3_dit", counting)
        for _ in range(3):
            assert _kind(client, "MiniMaxH3") == "w4a8"
        assert calls == []


def test_manager_without_registry_records_null(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, start_on_h3=True) as client:
        pm = client.app_context.pipeline_manager
        monkeypatch.setattr(pm, "model_registry", None)
        pm.identify_quant_class()
        assert pm.transformer_quant_class is None
