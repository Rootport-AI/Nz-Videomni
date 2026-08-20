"""Server runtime state (§3-97 P5): remember the last combination, and never
let remembering it break anything.

``services/runtime_state.py`` is a CACHE with a file behind it, so the contract
under test is mostly about what it does when the file is NOT usable: every
failure mode has to degrade to "start on the defaults", never to an exception
reaching a request or a boot. The happy path (a load writes the file, a restart
reads it back) is pinned end-to-end through the real app at the bottom.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import pytest
import yaml
from fastapi.testclient import TestClient

import main
from conftest import build_model_layout
from services.runtime_state import SCHEMA_VERSION, RuntimeState

STATE = {
    "schema": SCHEMA_VERSION,
    "active_base_model": "LTX23",
    "selections": {"LTX23": {"transformer": "alt"}},
}


def _write(path, payload) -> None:
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# read: absent / corrupt / wrong schema / wrong types
# --------------------------------------------------------------------------- #

def test_missing_file_yields_defaults_silently(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="ltx.state"):
        state = RuntimeState.load(tmp_path / "state.json")
    assert state.active_base_model is None
    assert state.selections == {}
    assert state.selection_for("LTX23") == {}
    # A fresh install has no state yet -- that is not worth a warning.
    assert caplog.records == []


def test_roundtrip_reads_back_what_was_saved(tmp_path):
    path = tmp_path / "state.json"
    RuntimeState(path).save("LTX23", {"transformer": "alt", "audio": "default"})
    again = RuntimeState.load(path)
    assert again.active_base_model == "LTX23"
    assert again.selection_for("LTX23") == {"transformer": "alt", "audio": "default"}


def test_save_keeps_the_other_base_models_selection(tmp_path):
    """Switching base model must not erase where the previous one was left:
    switching back is supposed to restore it."""
    path = tmp_path / "state.json"
    state = RuntimeState(path)
    state.save("LTX23", {"transformer": "alt"})
    state.save("LTX25", {"transformer": "default"})
    again = RuntimeState.load(path)
    assert again.active_base_model == "LTX25"
    assert again.selection_for("LTX23") == {"transformer": "alt"}


@pytest.mark.parametrize(
    "payload",
    [
        "{ this is not json",
        [1, 2, 3],                                        # not an object
        {"schema": 99, **{k: v for k, v in STATE.items() if k != "schema"}},
        {**STATE, "active_base_model": 5},                # wrong type
        {**STATE, "selections": ["LTX23"]},               # wrong type
        {**STATE, "selections": {"LTX23": "alt"}},        # per-base not an object
        {**STATE, "selections": {"LTX23": {"transformer": 7}}},  # name not a string
    ],
    ids=["broken-json", "not-object", "schema", "active-type", "selections-type",
         "per-base-type", "name-type"],
)
def test_unusable_file_is_set_aside_and_defaults_apply(tmp_path, caplog, payload):
    path = tmp_path / "state.json"
    _write(path, payload)
    with caplog.at_level(logging.WARNING, logger="ltx.state"):
        state = RuntimeState.load(path)
    assert state.active_base_model is None and state.selections == {}
    # Renamed aside, not deleted: the broken file stays inspectable, and the
    # next save starts from a clean slate instead of re-warning every boot.
    assert not path.exists()
    bad = tmp_path / "state.json.bad"
    assert bad.exists()
    assert len(caplog.records) == 1 and caplog.records[0].levelno == logging.WARNING


def test_a_second_corruption_overwrites_the_previous_bad_file(tmp_path):
    path = tmp_path / "state.json"
    _write(path, "{ broken")
    RuntimeState.load(path)
    _write(path, "{ broken again")
    RuntimeState.load(path)  # os.replace overwrites, no PermissionError on Windows
    assert (tmp_path / "state.json.bad").read_text(encoding="utf-8") == "{ broken again"


def test_unknown_ids_are_not_corruption(tmp_path):
    """A base model or category name that no longer exists is a legitimate
    change of the model store, not a broken file: it is read back as-is and the
    CALLER decides (AppContext falls back to the first base model; an unknown
    model name is caught by the registry's own 404 at resolve time)."""
    path = tmp_path / "state.json"
    _write(path, {"schema": SCHEMA_VERSION, "active_base_model": "GONE",
                  "selections": {"GONE": {"transformer": "vanished"}}})
    state = RuntimeState.load(path)
    assert state.active_base_model == "GONE"
    assert state.selection_for("GONE") == {"transformer": "vanished"}
    assert path.exists()  # not set aside


# --------------------------------------------------------------------------- #
# write: an unwritable location is a warning, never an exception
# --------------------------------------------------------------------------- #

def test_save_into_an_unwritable_location_only_warns(tmp_path, caplog):
    # A PATH THAT CANNOT BE A DIRECTORY: state.json's parent is an existing
    # FILE here, so mkdir/mkstemp fail on every OS alike (a read-only directory
    # is not a portable way to make a write fail -- Windows ignores the
    # read-only attribute for this).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    state = RuntimeState(blocker / "state.json")
    with caplog.at_level(logging.WARNING, logger="ltx.state"):
        state.save("LTX23", {"transformer": "alt"})  # must not raise
    assert len(caplog.records) == 1
    # The in-memory state still updated: the running server keeps behaving
    # correctly, only the memory across a restart is lost.
    assert state.active_base_model == "LTX23"
    assert state.selection_for("LTX23") == {"transformer": "alt"}


def test_save_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "state.json"
    RuntimeState(path).save("LTX23", {"transformer": "alt"})
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_save_is_atomic_against_a_failed_write(tmp_path, monkeypatch):
    """os.replace is what makes the file whole-old or whole-new. If the write
    dies before it, the PREVIOUS state must still be readable."""
    path = tmp_path / "state.json"
    state = RuntimeState(path)
    state.save("LTX23", {"transformer": "alt"})

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    state.save("LTX25", {"transformer": "default"})  # warns, does not raise
    assert RuntimeState.load(path).active_base_model == "LTX23"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]  # temp cleaned up


# --------------------------------------------------------------------------- #
# wiring: the app writes it on a successful load and resumes from it on boot
# --------------------------------------------------------------------------- #

def _make_args(config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=config_path, te_offload=None, dit_cpu_load=None,
    )


def _app(tmp_path, *, state_file, descriptors=None, transformers=None):
    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {
            "backend": "mock",
            **build_model_layout(tmp_path, descriptors),
            **({"transformers": transformers} if transformers else {}),
        },
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
        "state_file": state_file.as_posix(),
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return main.build_app(_make_args(cfg_path.as_posix()))


def test_a_successful_load_writes_the_state_file(tmp_path):
    state_file = tmp_path / "state.json"
    with TestClient(_app(tmp_path, state_file=state_file)) as c:
        assert not state_file.exists()  # nothing written just by booting
        assert c.post("/api/v1/pipeline/load").status_code == 200
    written = json.loads(state_file.read_text(encoding="utf-8"))
    assert written["schema"] == SCHEMA_VERSION
    assert written["active_base_model"] == "LTX23"
    assert written["selections"]["LTX23"]["transformer"] == "default"


def test_unload_does_not_forget_the_selection(tmp_path):
    """Unloading frees VRAM; it is not "forget my choice" (the same reason
    ``active_models`` survives an unload)."""
    state_file = tmp_path / "state.json"
    with TestClient(_app(tmp_path, state_file=state_file)) as c:
        c.post("/api/v1/pipeline/load")
        before = state_file.read_text(encoding="utf-8")
        c.post("/api/v1/pipeline/unload")
    assert state_file.read_text(encoding="utf-8") == before


def test_startup_resumes_the_remembered_selection(tmp_path):
    state_file = tmp_path / "state.json"
    _write(state_file, {
        "schema": SCHEMA_VERSION,
        "active_base_model": "LTX23",
        "selections": {"LTX23": {"transformer": "alt"}},
    })
    weights = tmp_path / "models" / "LTX23" / "Weights"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "alt.gguf").write_bytes(b"GGUF" + b"\0" * 16)
    with TestClient(_app(tmp_path, state_file=state_file)) as c:
        models = c.get("/api/v1/models").json()
        assert models["categories"]["transformer"]["active"] == "alt"
        # Categories the state never mentioned still span the full set.
        assert models["categories"]["audio"]["active"] == "default"


def test_startup_resolves_the_remembered_names_to_real_paths(tmp_path):
    """P6b: restoring the NAMES alone would make the server REPORT "alt" while
    a body-less load (and auto-load-on-generate) still sent the DEFAULT file —
    silent, and only visible in the generated video. The paths are resolved at
    startup so the two halves can never disagree."""
    state_file = tmp_path / "state.json"
    _write(state_file, {
        "schema": SCHEMA_VERSION,
        "active_base_model": "LTX23",
        "selections": {"LTX23": {"transformer": "alt"}},
    })
    weights = tmp_path / "models" / "LTX23" / "Weights"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "alt.gguf").write_bytes(b"GGUF" + b"\0" * 16)

    app = _app(tmp_path, state_file=state_file)
    pm = app.state.context.pipeline_manager
    # Seeded before any request: the path is there from construction.
    assert pm._active_selection_paths["transformer"].endswith("alt.gguf")
    with TestClient(app) as c:
        assert c.post("/api/v1/pipeline/load").status_code == 200  # BODY-LESS
    # ...and that is what actually reached the worker.
    selection = pm.runner._backend.last_selection
    assert selection is not None
    assert selection["transformer"].endswith("alt.gguf")


def test_a_remembered_name_that_vanished_falls_back_to_default(tmp_path, caplog):
    """The model store legitimately changes between runs (a file moved, a
    config registration removed). That is a WARNING naming the lost name and a
    fallback to "default" — never a failed boot, and never a silent revert."""
    state_file = tmp_path / "state.json"
    _write(state_file, {
        "schema": SCHEMA_VERSION,
        "active_base_model": "LTX23",
        "selections": {"LTX23": {"transformer": "gone-away", "audio": "default"}},
    })
    with caplog.at_level(logging.WARNING, logger="ltx.state"):
        app = _app(tmp_path, state_file=state_file)
    pm = app.state.context.pipeline_manager
    assert pm.active_models["transformer"] == "default"
    assert pm._active_selection_paths == {}
    assert any("gone-away" in r.getMessage() for r in caplog.records)
    with TestClient(app) as c:
        assert c.post("/api/v1/pipeline/load").status_code == 200
        models = c.get("/api/v1/models").json()
        assert models["categories"]["transformer"]["active"] == "default"
    # The state file is not rewritten by the fallback itself -- the next
    # successful load is what updates it, and it just did.
    written = json.loads(state_file.read_text(encoding="utf-8"))
    assert written["selections"]["LTX23"]["transformer"] == "default"


def test_a_remembered_name_whose_file_is_gone_falls_back_too(tmp_path, caplog):
    """The other half of the same failure: the NAME is still registered (it is
    in config.yaml), but the weight file behind it is no longer on disk."""
    state_file = tmp_path / "state.json"
    _write(state_file, {
        "schema": SCHEMA_VERSION,
        "active_base_model": "LTX23",
        "selections": {"LTX23": {"transformer": "ghost"}},
    })
    with caplog.at_level(logging.WARNING, logger="ltx.state"):
        app = _app(
            tmp_path,
            state_file=state_file,
            transformers={"ghost": (tmp_path / "never-created.gguf").as_posix()},
        )
    pm = app.state.context.pipeline_manager
    assert pm.active_models["transformer"] == "default"
    assert any("ghost" in r.getMessage() for r in caplog.records)


def test_startup_falls_back_to_the_first_base_model(tmp_path, caplog):
    state_file = tmp_path / "state.json"
    _write(state_file, {
        "schema": SCHEMA_VERSION,
        "active_base_model": "REMOVED",
        "selections": {"REMOVED": {"transformer": "alt"}},
    })
    with caplog.at_level(logging.INFO, logger="ltx.state"):
        app = _app(tmp_path, state_file=state_file)
    with TestClient(app) as c:
        assert c.get("/api/v1/models").json()["active_base_model"] == "LTX23"
    assert any("REMOVED" in r.getMessage() for r in caplog.records)


def test_broken_state_file_does_not_stop_the_server(tmp_path, caplog):
    state_file = tmp_path / "state.json"
    _write(state_file, "{ not json at all")
    with caplog.at_level(logging.WARNING, logger="ltx.state"):
        with TestClient(_app(tmp_path, state_file=state_file)) as c:
            assert c.get("/api/v1/status").status_code == 200
            # ...and the next successful load writes a fresh, valid file.
            assert c.post("/api/v1/pipeline/load").status_code == 200
    assert json.loads(state_file.read_text(encoding="utf-8"))["schema"] == SCHEMA_VERSION
    assert (tmp_path / "state.json.bad").exists()


def test_a_failed_load_does_not_overwrite_the_last_good_combination(tmp_path):
    state_file = tmp_path / "state.json"
    app = _app(tmp_path, state_file=state_file)
    with TestClient(app) as c:
        c.post("/api/v1/pipeline/load")
        c.post("/api/v1/pipeline/unload")
        pm = app.state.context.pipeline_manager
        pm.runner.load = _raise  # type: ignore[method-assign]
        r = c.post("/api/v1/pipeline/load")
        assert r.status_code == 503
    written = json.loads(state_file.read_text(encoding="utf-8"))
    assert written["selections"]["LTX23"]["transformer"] == "default"


def _raise(*args, **kwargs):
    raise RuntimeError("the worker refused to start")
