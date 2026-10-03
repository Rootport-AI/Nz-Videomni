"""§1-36: while a load is in flight, unload is a 409 too — and no exit path of
``load``/``reload`` leaves the state stuck at "loading".

``PipelineManager.load``/``reload`` run the worker build OUTSIDE the lock, so
an unload that slipped through mid-load used to flip the state to "unloaded"
and let a second load past the 409 into an overlapping worker build. Unload
was kept outside the guard only as the way out of a stuck "loading"; the
``finally`` in ``load``/``reload`` removes that stuck state, so one rule now
covers every lifecycle call.
"""

from __future__ import annotations

import threading

import pytest

from api.errors import APIError


class _Interrupted(KeyboardInterrupt):
    """A BaseException that ``except Exception`` does not see."""


def _stall_runner_load(monkeypatch, pm):
    """Make ``runner.load`` wait on a gate; return (entered, gate, calls)."""
    entered = threading.Event()
    gate = threading.Event()
    calls: list[int] = []
    orig = pm.runner.load

    def slow_load(*a, **k):
        calls.append(1)
        entered.set()
        gate.wait(5)
        return orig(*a, **k)

    monkeypatch.setattr(pm.runner, "load", slow_load)
    return entered, gate, calls


def test_unload_during_load_is_rejected_and_the_load_finishes(client, monkeypatch):
    pm = client.app_context.pipeline_manager
    pm.unload()
    assert pm.state == pm.STATE_UNLOADED
    entered, gate, calls = _stall_runner_load(monkeypatch, pm)

    t = threading.Thread(target=pm.load)
    t.start()
    try:
        assert entered.wait(5)
        assert pm.state == pm.STATE_LOADING
        with pytest.raises(APIError) as unload_exc:
            pm.unload()
        assert unload_exc.value.status_code == 409
        assert unload_exc.value.code == "PIPELINE_LOADING"
        assert pm.state == pm.STATE_LOADING
        # The guard stayed closed, so a second load is still refused.
        with pytest.raises(APIError) as load_exc:
            pm.load()
        assert load_exc.value.code == "PIPELINE_LOADING"
    finally:
        gate.set()
        t.join(5)
    assert not t.is_alive()
    assert calls == [1]
    assert pm.state == pm.STATE_READY


def test_http_unload_during_load_is_409(client, monkeypatch):
    pm = client.app_context.pipeline_manager
    client.post("/api/v1/pipeline/unload")
    entered, gate, _calls = _stall_runner_load(monkeypatch, pm)

    t = threading.Thread(target=pm.load)
    t.start()
    try:
        assert entered.wait(5)
        r = client.post("/api/v1/pipeline/unload")
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "PIPELINE_LOADING"
        assert client.get("/api/v1/status").json()["state"] == "loading"
    finally:
        gate.set()
        t.join(5)
    assert client.get("/api/v1/status").json()["state"] == "ready"
    assert client.post("/api/v1/pipeline/unload").status_code == 200


def _interrupt_after_start(monkeypatch, runner):
    """Make ``runner.load`` start the worker and THEN raise a BaseException
    (a half-started worker: ``runner.loaded`` is already true); record every
    ``runner.unload`` call. Return the list of unload calls."""
    orig_load = runner.load
    orig_unload = runner.unload
    unloads: list[int] = []

    def half_load(*a, **k):
        orig_load(*a, **k)
        assert runner.loaded
        raise _Interrupted()

    def recording_unload():
        unloads.append(1)
        orig_unload()

    monkeypatch.setattr(runner, "load", half_load)
    monkeypatch.setattr(runner, "unload", recording_unload)
    return unloads


def _count_runner_loads(monkeypatch, runner):
    """Wrap ``runner.load`` to count calls (to see the next load really
    reaches the runner instead of early-returning on a stale worker)."""
    orig_load = runner.load
    calls: list[int] = []

    def counting_load(*a, **k):
        calls.append(1)
        return orig_load(*a, **k)

    monkeypatch.setattr(runner, "load", counting_load)
    return calls


def test_load_never_leaves_loading_behind_on_a_base_exception(client, monkeypatch):
    """A BaseException gets the same cleanup as ``except Exception``: the
    half-started worker is stopped, so the next load rebuilds it instead of
    adopting it as "ready"."""
    pm = client.app_context.pipeline_manager
    pm.unload()
    unloads = _interrupt_after_start(monkeypatch, pm.runner)
    with pytest.raises(_Interrupted):
        pm.load()
    assert unloads, "the half-started worker must be stopped"
    assert pm.state == "unloaded"
    assert not pm.runner.loaded
    # Not locked out, and not an early return: the next load reaches the runner.
    monkeypatch.undo()
    calls = _count_runner_loads(monkeypatch, pm.runner)
    pm.load()
    assert calls == [1]
    assert pm.state == pm.STATE_READY


def test_reload_never_leaves_loading_behind_on_a_base_exception(client, monkeypatch):
    pm = client.app_context.pipeline_manager
    pm.load()
    unloads = _interrupt_after_start(monkeypatch, pm.runner)
    with pytest.raises(_Interrupted):
        pm.reload(selection={}, active_names={})
    assert unloads
    assert pm.state == "unloaded"
    assert not pm.runner.loaded
    monkeypatch.undo()
    calls = _count_runner_loads(monkeypatch, pm.runner)
    pm.load()
    assert calls == [1]
    assert pm.state == pm.STATE_READY


@pytest.fixture()
def client_two_bases(tmp_path):
    """Two runnable base models (LTX23 and a second 2.3 install named LTX25)."""
    from fastapi.testclient import TestClient

    from test_base_model_axis import _build

    app = _build(tmp_path, second_version="2.3.0")
    with TestClient(app) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        yield c


def test_reload_rolls_the_base_model_back_on_a_base_exception(
    client_two_bases, monkeypatch
):
    from test_base_model_axis import LTX25_ID

    pm = client_two_bases.app_context.pipeline_manager
    pm.load()
    assert pm.active_base_model == "LTX23"
    unloads = _interrupt_after_start(monkeypatch, pm.runner)
    with pytest.raises(_Interrupted):
        pm.reload(selection={}, active_names={}, base_model=LTX25_ID)
    assert unloads
    assert pm.state == "unloaded"
    assert not pm.runner.loaded
    # ``_restore_base_model`` ran: the manager and the runner are back on LTX23.
    assert pm.active_base_model == "LTX23"
    assert pm.runner.descriptor.id == "LTX23"
    monkeypatch.undo()
    calls = _count_runner_loads(monkeypatch, pm.runner)
    pm.load()
    assert calls == [1]
    assert pm.state == pm.STATE_READY
    assert pm.active_base_model == "LTX23"


def _fail(*_a, **_k):
    raise RuntimeError("boom")


def test_load_failure_still_ends_unloaded_via_cleanup(client, monkeypatch):
    """The ordinary failure path is unchanged: ``_cleanup_after_error`` leaves
    UNLOADED exactly once, and the ``finally`` has nothing to override."""
    pm = client.app_context.pipeline_manager
    pm.unload()
    cleanups: list[int] = []
    orig_cleanup = pm._cleanup_after_error

    def counting_cleanup():
        cleanups.append(1)
        orig_cleanup()

    monkeypatch.setattr(pm, "_cleanup_after_error", counting_cleanup)
    monkeypatch.setattr(pm.runner, "load", _fail)
    with pytest.raises(APIError) as exc:
        pm.load()
    assert exc.value.code == "PIPELINE_LOAD_FAILED"
    assert pm.state == pm.STATE_UNLOADED
    assert cleanups == [1]


def test_reload_failure_still_ends_unloaded_via_cleanup(client, monkeypatch):
    pm = client.app_context.pipeline_manager
    pm.load()
    cleanups: list[int] = []
    orig_cleanup = pm._cleanup_after_error

    def counting_cleanup():
        cleanups.append(1)
        orig_cleanup()

    monkeypatch.setattr(pm, "_cleanup_after_error", counting_cleanup)
    monkeypatch.setattr(pm.runner, "load", _fail)
    with pytest.raises(APIError) as exc:
        pm.reload(selection={}, active_names={})
    assert exc.value.code == "PIPELINE_LOAD_FAILED"
    assert pm.state == pm.STATE_UNLOADED
    assert cleanups == [1]
