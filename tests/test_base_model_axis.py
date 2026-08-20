"""POST /pipeline/load's BASE-MODEL axis, its guards, and GET /status (§3-97 P6).

``models`` picks a file within a base model; ``base_model`` picks the base model
itself. What is pinned here:

* a body carrying ONLY ``base_model`` switches — and, on the way, runs the full
  resolve -> precheck -> KV ruling over EVERY category, which is what makes an
  LTX 2.5 selection fail loud with the "next stage" message instead of reaching
  a worker that would mis-run it;
* an unchanged base model with an all-default selection still produces NO
  payload override, so the golden byte-identical load is untouched;
* a second load arriving while one is in flight is a 409, and an auto-load from
  inside a generate job surfaces that as a readable job error;
* ``unload`` is deliberately NOT behind that 409 — it is the recovery door.

Everything runs on the mock backend, so no weights and no GPU are involved: the
"LTX 2.5" file here is a hand-built GGUF header carrying ``model_version 2.5.0``,
which is all the ruling reads.
"""

from __future__ import annotations

import argparse
import json
import struct

import pytest
import yaml
from fastapi.testclient import TestClient

import main
from conftest import base_model_descriptor, build_model_layout, write_model_file

LTX25_ID = "LTX25"


# --------------------------------------------------------------------------- #
# fixtures: a two-base-model install
# --------------------------------------------------------------------------- #

def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_gguf(path, **kv: str):
    """A minimal but structurally valid GGUF carrying string KV entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = b"".join(
        _gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value)
        for key, value in kv.items()
    )
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + body)
    return path


def _make_args(config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=config_path, te_offload=None, dit_cpu_load=None,
    )


def _second_descriptor() -> dict:
    """A SECOND base model, same four-category shape, own directory."""
    descriptor = base_model_descriptor(LTX25_ID)
    descriptor["display_name"] = "LTX 2.5"
    return descriptor


def _build(tmp_path, *, second_version: str):
    """Two installed base models; the second's transformer declares
    ``second_version`` ("2.5.0" -> unrunnable, "2.3.0" -> runnable)."""
    descriptors = [base_model_descriptor(), _second_descriptor()]
    fragment = build_model_layout(tmp_path, descriptors)  # writes base #1's files
    models_dir = tmp_path / "models"
    for category, spec in descriptors[1]["categories"].items():
        target = models_dir / spec["default_file"]
        if category == "transformer":
            _write_gguf(
                target,
                **{"general.architecture": "ltxv", "model_version": second_version},
            )
        else:
            write_model_file(target)
    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {"backend": "mock", **fragment},
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
        "state_file": (tmp_path / "state.json").as_posix(),
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return main.build_app(_make_args(cfg_path.as_posix()))


@pytest.fixture()
def client25(tmp_path):
    """Base model #2 is an LTX 2.5 this engine cannot run yet."""
    app = _build(tmp_path, second_version="2.5.0")
    with TestClient(app) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        c.tmp_path = tmp_path  # type: ignore[attr-defined]
        yield c


@pytest.fixture()
def client23(tmp_path):
    """Base model #2 is a second, RUNNABLE 2.3 install (a switch that works)."""
    app = _build(tmp_path, second_version="2.3.0")
    with TestClient(app) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        c.tmp_path = tmp_path  # type: ignore[attr-defined]
        yield c


def _backend(client):
    return client.app_context.pipeline_manager.runner._backend


# --------------------------------------------------------------------------- #
# the axis itself
# --------------------------------------------------------------------------- #

def test_base_model_alone_switches_and_reports_it(client23):
    r = client23.post("/api/v1/pipeline/load", json={"base_model": LTX25_ID})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["base_model"] == LTX25_ID
    assert body["pipeline_loaded"] is True
    # The NAMES stay "default" — a base change is not a model selection.
    assert set(body["models"].values()) == {"default"}
    # GET /models and GET /status both follow, without a second request.
    models = client23.get("/api/v1/models").json()
    assert models["active_base_model"] == LTX25_ID
    assert [b["id"] for b in models["base_models"] if b["active"]] == [LTX25_ID]
    assert client23.get("/api/v1/status").json()["base_model"] == LTX25_ID


def test_switching_passes_every_category_as_an_explicit_path(client23):
    """N-3: a base change cannot leave a category on "no override" — that would
    send the OLD base model's default weight to the worker."""
    client23.post("/api/v1/pipeline/load", json={"base_model": LTX25_ID})
    selection = _backend(client23).last_selection
    assert set(selection) == {"transformer", "text_encoder", "video_vae", "audio"}
    assert all(f"/{LTX25_ID}/" in p.replace("\\", "/") for p in selection.values())


def test_switching_back_restores_the_first_base_model(client23):
    client23.post("/api/v1/pipeline/load", json={"base_model": LTX25_ID})
    r = client23.post("/api/v1/pipeline/load", json={"base_model": "LTX23"})
    assert r.status_code == 200, r.text
    assert r.json()["base_model"] == "LTX23"
    selection = _backend(client23).last_selection
    assert all("/LTX23/" in p.replace("\\", "/") for p in selection.values())


def test_the_switch_is_remembered_across_a_restart(client23):
    client23.post("/api/v1/pipeline/load", json={"base_model": LTX25_ID})
    written = json.loads((client23.tmp_path / "state.json").read_text(encoding="utf-8"))
    assert written["active_base_model"] == LTX25_ID


def test_unknown_base_model_404_names_the_known_ids(client23):
    r = client23.post("/api/v1/pipeline/load", json={"base_model": "LTX99"})
    assert r.status_code == 404
    error = r.json()["error"]
    assert error["code"] == "MODEL_NOT_FOUND"
    assert "LTX23" in error["detail"] and LTX25_ID in error["detail"]


def test_no_op_is_judged_against_the_base_model_too(client23):
    """The same category names on a DIFFERENT base model are a different
    combination, so the no-op shortcut must not swallow the switch."""
    client23.post("/api/v1/pipeline/load", json={"base_model": LTX25_ID})
    backend = _backend(client23)
    calls_before = backend.load_calls

    # Same base, same (all-default) names -> genuinely nothing to do.
    r = client23.post("/api/v1/pipeline/load", json={"base_model": LTX25_ID})
    assert r.status_code == 200
    assert backend.load_calls == calls_before
    assert r.json()["base_model"] == LTX25_ID

    # Different base, identical names -> a rebuild. The backend OBJECT is
    # replaced rather than reloaded (whether the real stack is available is a
    # question about the NEW base model's files), so the proof is a fresh
    # instance that has loaded once — not a bumped counter on the old one.
    r = client23.post("/api/v1/pipeline/load", json={"base_model": "LTX23"})
    assert r.status_code == 200
    rebuilt = _backend(client23)
    assert rebuilt is not backend and rebuilt.load_calls == 1


def test_an_unchanged_base_with_all_defaults_still_sends_no_override(client23):
    """The byte-identity guarantee: naming the CURRENT base model explicitly
    must not start filling the payload with paths."""
    r = client23.post(
        "/api/v1/pipeline/load",
        json={"base_model": "LTX23", "models": {"transformer": "default"}},
    )
    assert r.status_code == 200, r.text
    assert _backend(client23).last_selection is None
    assert r.json()["base_model"] == "LTX23"


def test_a_bodyless_load_is_unaffected_by_the_new_axis(client23):
    r = client23.post("/api/v1/pipeline/load")
    assert r.status_code == 200
    # Legacy response key set exactly: neither "models" nor "base_model".
    assert set(r.json()) == {"pipeline_loaded", "pipeline_type", "state"}
    assert _backend(client23).last_selection is None


# --------------------------------------------------------------------------- #
# the KV ruling, reached through the API
# --------------------------------------------------------------------------- #

def test_selecting_ltx25_is_refused_with_the_next_stage_message(client25):
    r = client25.post("/api/v1/pipeline/load", json={"base_model": LTX25_ID})
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "MODEL_INCOMPATIBLE"
    assert "ltxv 2.5.0" in error["detail"]
    assert "次段階" in error["detail"] and "§3-98" in error["detail"]


def test_a_refused_switch_leaves_the_base_model_untouched(client25):
    """The ruling fires BEFORE the pipeline is touched, so the server is still
    on LTX 2.3 and still usable (G10's "switch back and keep generating")."""
    client25.post("/api/v1/pipeline/load")
    backend = _backend(client25)
    calls_before = backend.load_calls

    assert client25.post(
        "/api/v1/pipeline/load", json={"base_model": LTX25_ID}
    ).status_code == 422

    assert backend.load_calls == calls_before
    assert client25.get("/api/v1/status").json()["base_model"] == "LTX23"
    assert client25.get("/api/v1/models").json()["active_base_model"] == "LTX23"
    assert client25.app_context.pipeline_manager.state == "ready"


def test_the_ruling_runs_for_a_named_model_on_the_same_base(client25, tmp_path):
    """A 2.5 file dropped into the CURRENT base model's scan directory is
    refused just the same — the KV, not the folder, decides."""
    _write_gguf(
        tmp_path / "models" / "LTX23" / "Weights" / "smuggled.gguf",
        **{"general.architecture": "ltxv", "model_version": "2.5.0"},
    )
    r = client25.post(
        "/api/v1/pipeline/load", json={"models": {"transformer": "smuggled"}}
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "MODEL_INCOMPATIBLE"


# --------------------------------------------------------------------------- #
# the loading guard (409) and its deliberate exception
# --------------------------------------------------------------------------- #

def _pin_loading(client) -> None:
    """Park the manager in the state a real in-flight load spends minutes in.

    Set directly rather than by stalling the mock backend: the guard is a pure
    state test, and a thread held open across a TestClient request is a far
    less readable way to reach the same line.
    """
    pm = client.app_context.pipeline_manager
    pm.state = pm.STATE_LOADING


def test_a_second_load_while_loading_is_409(client23):
    _pin_loading(client23)
    r = client23.post("/api/v1/pipeline/load")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "PIPELINE_LOADING"


def test_a_swap_while_loading_is_409(client23):
    client23.post("/api/v1/pipeline/load")  # loaded -> the reload path
    _pin_loading(client23)
    r = client23.post("/api/v1/pipeline/load", json={"base_model": LTX25_ID})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "PIPELINE_LOADING"


def test_unload_is_the_way_out_of_a_stuck_loading_state(client23):
    """DELIBERATELY not behind the 409: a load that dies where ``except
    Exception`` cannot see it would otherwise lock the server out of loading
    for good."""
    _pin_loading(client23)
    r = client23.post("/api/v1/pipeline/unload")
    assert r.status_code == 200
    assert r.json()["state"] == "unloaded"
    assert client23.post("/api/v1/pipeline/load").status_code == 200


def test_auto_load_during_a_load_fails_the_job_readably(client23):
    """Auto-load-on-generate goes through the same guard, where the 409 is
    re-wrapped as GENERATION_FAILED — the operator must still be able to read
    WHY out of the job record."""
    _pin_loading(client23)
    payload = {
        "prompt": "A red ball rolling on a white floor",
        "width": 384, "height": 256, "num_frames": 17,
        "num_inference_steps": 8, "guidance_scale": 1.0, "seed": 42,
        "pipeline": "distilled",
    }
    r = client23.post("/api/v1/generate", json=payload)
    assert r.status_code == 202, r.text
    job = client23.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert "GENERATION_FAILED" in job["error"]
    assert "読み込み中" in job["error"]


# --------------------------------------------------------------------------- #
# GET /status
# --------------------------------------------------------------------------- #

def test_status_publishes_state_and_base_model(client23):
    status = client23.get("/api/v1/status").json()
    assert status["state"] == "unloaded"
    assert status["base_model"] == "LTX23"

    client23.post("/api/v1/pipeline/load")
    assert client23.get("/api/v1/status").json()["state"] == "ready"

    # A selection survives an unload; only the LOAD state changes.
    client23.post("/api/v1/pipeline/load", json={"base_model": LTX25_ID})
    client23.post("/api/v1/pipeline/unload")
    status = client23.get("/api/v1/status").json()
    assert status["state"] == "unloaded"
    assert status["base_model"] == LTX25_ID
