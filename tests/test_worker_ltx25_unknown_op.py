"""The LTX 2.5 worker answers an unknown op after load (台帳 §1-42).

``engine25.worker.main`` is driven through a fake stdin. ``_do_load`` is a
no-op, ``_emit`` and ``_log`` are recorders, and ``_shutdown`` raises
``SystemExit`` so the loop can end without touching the GPU.

Two facts are pinned:

* after load, an unknown op gets exactly one ``error`` event whose ``detail``
  names the op (so a caller waiting for the reply is not left hanging);
* before load, an unknown op gets NO event at all (an ``error`` there would be
  read by the adapter as the reply to the next ``load``).

NO PIPELINE AND NO GPU. Run with ``.venv-engine-ltx25`` and ``--noconftest``::

    .venv-engine-ltx25/Scripts/python.exe \\
        Docs/Outputs-archive/start-end-bridge-2026-09-07/implA_engine_runner/run_ltx25_pytest.py \\
        tests/test_worker_ltx25_unknown_op.py
"""

from __future__ import annotations

import io
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

from engine25 import worker  # noqa: E402


def _drive(monkeypatch, lines: list[dict]) -> tuple[list[dict], int | None]:
    """Feed ``lines`` to ``worker.main`` and return the emitted events."""
    events: list[dict] = []

    def _shutdown():
        raise SystemExit(0)

    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(worker, "_log", lambda msg: None)
    monkeypatch.setattr(worker, "_do_load", lambda msg: None)
    monkeypatch.setattr(worker, "_shutdown", _shutdown)
    monkeypatch.setattr(
        worker.sys, "stdin", io.StringIO("".join(json.dumps(x) + "\n" for x in lines))
    )
    with pytest.raises(SystemExit) as exc:
        worker.main()
    return events, exc.value.code


def test_an_unknown_op_after_load_is_answered_with_one_error(monkeypatch):
    events, code = _drive(monkeypatch, [{"op": "load"}, {"op": "frobnicate"}])
    assert code == 0  # EOF -> graceful shutdown
    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert "frobnicate" in events[0]["detail"]


def test_each_unknown_op_after_load_gets_its_own_error(monkeypatch):
    """The process stays alive: a second unknown op is answered too."""
    events, _code = _drive(
        monkeypatch, [{"op": "load"}, {"op": "frobnicate"}, {"op": "generate_chain2"}]
    )
    assert [e["event"] for e in events] == ["error", "error"]
    assert "frobnicate" in events[0]["detail"]
    assert "generate_chain2" in events[1]["detail"]


def test_an_unknown_op_before_load_emits_nothing(monkeypatch):
    events, code = _drive(monkeypatch, [{"op": "frobnicate"}])
    assert code == 0
    assert events == []


def test_an_unknown_op_before_load_does_not_answer_the_later_load(monkeypatch):
    """The pre-load unknown op stays silent even when a load follows it, so the
    adapter never reads a stray ``error`` as the load's reply."""
    events, _code = _drive(monkeypatch, [{"op": "frobnicate"}, {"op": "load"}])
    assert events == []
