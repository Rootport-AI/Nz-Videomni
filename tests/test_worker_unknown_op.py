"""LTX 2.3 worker: an unknown op after load gets an ``error`` reply (§1-42).

``engine.worker.main`` is driven with a fake stdin; ``_do_load`` is a no-op,
``_emit`` / ``_log`` record, and ``_shutdown`` raises SystemExit (as the real one
does). Before load an unknown op is still dropped without any event — an
``error`` there would be read by the adapter as the reply to its ``load``.
"""

from __future__ import annotations

import io
import json

import pytest

pytest.importorskip("torch")

from engine import worker  # noqa: E402


def _drive(monkeypatch, ops):
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append((event, f)))
    monkeypatch.setattr(worker, "_log", lambda _m: None)
    monkeypatch.setattr(worker, "_do_load", lambda _msg: None)

    def _fake_shutdown():
        raise SystemExit(0)

    monkeypatch.setattr(worker, "_shutdown", _fake_shutdown)
    stdin = io.StringIO("".join(json.dumps(o) + "\n" for o in ops))
    monkeypatch.setattr(worker.sys, "stdin", stdin)
    with pytest.raises(SystemExit):
        worker.main()
    return events


def test_unknown_op_after_load_emits_one_error(monkeypatch):
    events = _drive(monkeypatch, [{"op": "load"}, {"op": "frobnicate"}])
    assert len(events) == 1
    event, fields = events[0]
    assert event == "error"
    assert "frobnicate" in fields["detail"]


def test_second_load_after_load_emits_error(monkeypatch):
    events = _drive(monkeypatch, [{"op": "load"}, {"op": "load"}])
    assert [e for e, _ in events] == ["error"]
    assert "'load'" in events[0][1]["detail"]


def test_unknown_op_before_load_emits_nothing(monkeypatch):
    events = _drive(monkeypatch, [{"op": "frobnicate"}, {"op": "load"}])
    assert events == []
