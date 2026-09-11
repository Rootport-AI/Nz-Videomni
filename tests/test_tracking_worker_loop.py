"""The worker's receive loop (``tracking/worker.py``), driven on fake pipes.

Runs on the APP venv: nothing in ``tracking/worker.py`` imports torch at module
level -- the tracker is imported inside the ``load`` handler precisely so this
file can exist -- and the checkpoint is replaced here by a stub module in
``sys.modules``. So the loop's CONTRACT is checked on every ordinary pytest run,
while ``tests/test_tracking_runtime_smoke.py`` checks the model it loads.

The four rules under test are the ones inherited from ``engine/worker.py``:
ignore everything before ``load``, die loudly if ``load`` fails, keep serving
after a failure that is not ``load``, and exit on ``shutdown`` or EOF.
"""

from __future__ import annotations

import io
import sys
import types

import pytest

from tracking import protocol, worker

WIDTH, HEIGHT = 8, 6
FRAME = bytes(WIDTH * HEIGHT * 4)
BOX = [2.0, 1.0, 4.0, 3.0]


class _FakeTracker:
    """Records what it was asked, answers with something recognisable."""

    def __init__(self, *, fail_track: bool = False) -> None:
        self.missing_keys: list[str] = []
        self.unexpected_keys: list[str] = []
        self.initialized_with = None
        self.tracks = 0
        self._fail_track = fail_track

    def initialize(self, frame, box, **kwargs):
        self.initialized_with = (frame.shape, tuple(box), kwargs)

    def track(self, frame):
        if self._fail_track:
            raise RuntimeError("tracker exploded")
        self.tracks += 1
        return (BOX[0] + self.tracks, BOX[1], BOX[2], BOX[3]), 0.75


def _install_runtime_stub(monkeypatch, tracker: _FakeTracker) -> dict:
    """Stand in for ``tracking.uetrack_runtime`` (which needs torch)."""
    calls: dict = {}
    module = types.ModuleType("tracking.uetrack_runtime")

    def _load_tracker(checkpoint_path, device="cpu"):
        calls["checkpoint_path"] = checkpoint_path
        calls["device"] = device
        return tracker

    module.load_tracker = _load_tracker
    monkeypatch.setitem(sys.modules, "tracking.uetrack_runtime", module)
    return calls


def _drive(monkeypatch, messages: list[bytes]) -> tuple[list[dict], int]:
    """Run ``worker.main`` over a scripted stdin; return its events and exit code.

    The worker keeps its tracker and the current session's frame size in module
    globals -- one resident tracker per PROCESS is the design. pytest runs every
    test in the same process, so they are reset here; without that, one test's
    session would be visible to the next, which is a state no real worker can be
    in (a fresh process starts with none).
    """
    monkeypatch.setattr(worker, "_TRACKER", None)
    monkeypatch.setattr(worker, "_WIDTH", 0)
    monkeypatch.setattr(worker, "_HEIGHT", 0)
    out = io.BytesIO()
    monkeypatch.setattr(
        worker.sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(b"".join(messages)))
    )
    monkeypatch.setattr(worker.sys, "stdout", types.SimpleNamespace(buffer=out))
    with pytest.raises(SystemExit) as exit_info:
        worker.main()
    stream = io.BytesIO(out.getvalue())
    events: list[dict] = []
    while True:
        try:
            event, _payload = protocol.read_message(stream)
        except protocol.ProtocolEOF:
            break
        events.append(event)
    return events, exit_info.value.code


def _load_msg(path: str = "C:/weights/uetrack_base.safetensors") -> bytes:
    return protocol.encode_message({"op": protocol.OP_LOAD, "checkpoint_path": path})


def _init_msg(**over) -> bytes:
    body = {
        "op": protocol.OP_INIT,
        "width": WIDTH,
        "height": HEIGHT,
        "box": BOX,
        "search_factor": 4.0,
    }
    body.update(over)
    return protocol.encode_message(body, FRAME)


def _track_msg() -> bytes:
    return protocol.encode_message({"op": protocol.OP_TRACK}, FRAME)


# --------------------------------------------------------------------------- #


def test_load_then_init_then_track(monkeypatch):
    tracker = _FakeTracker()
    calls = _install_runtime_stub(monkeypatch, tracker)
    events, code = _drive(
        monkeypatch,
        [_load_msg(), _init_msg(), _track_msg(), protocol.encode_message({"op": "shutdown"})],
    )

    assert code == 0
    assert calls == {"checkpoint_path": "C:/weights/uetrack_base.safetensors", "device": "cpu"}
    assert [e["event"] for e in events] == ["ready", "box", "box"]
    assert events[0] == {"event": "ready", "model": "uetrack_base", "device": "cpu"}
    # init answers with the seed itself, at score 1.0 -- it is not a prediction.
    assert events[1] == {"event": "box", "x": 2.0, "y": 1.0, "w": 4.0, "h": 3.0, "score": 1.0}
    assert events[2]["x"] == BOX[0] + 1
    assert events[2]["score"] == 0.75
    # The frame reached the tracker as an (H, W, 3) RGB array, alpha dropped.
    assert tracker.initialized_with[0] == (HEIGHT, WIDTH, 3)
    assert tracker.initialized_with[2] == {"search_factor": 4.0}


def test_an_absent_search_factor_is_left_to_the_runtime(monkeypatch):
    """The default lives in uetrack_runtime; the worker must not restate it."""
    tracker = _FakeTracker()
    _install_runtime_stub(monkeypatch, tracker)
    _drive(monkeypatch, [_load_msg(), _init_msg(search_factor=None)])
    assert tracker.initialized_with[2] == {}


def test_ops_before_load_are_ignored(monkeypatch):
    """A frame that arrives before the model does must not crash the worker."""
    tracker = _FakeTracker()
    _install_runtime_stub(monkeypatch, tracker)
    events, code = _drive(
        monkeypatch, [_track_msg(), _init_msg(), _load_msg(), _init_msg(), _track_msg()]
    )
    assert code == 0
    # Nothing at all was said until the load succeeded -- and the ignored ops
    # left NO trace: the session that follows starts from its own init.
    assert [e["event"] for e in events] == ["ready", "box", "box"]


def test_a_load_failure_reports_and_exits_one(monkeypatch):
    """The parent must see the worker DIE, not wait for a ready that never comes."""
    module = types.ModuleType("tracking.uetrack_runtime")

    def _boom(checkpoint_path, device="cpu"):
        raise FileNotFoundError(checkpoint_path)

    module.load_tracker = _boom
    monkeypatch.setitem(sys.modules, "tracking.uetrack_runtime", module)

    events, code = _drive(monkeypatch, [_load_msg("nowhere.safetensors")])
    assert code == 1
    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert "FileNotFoundError" in events[0]["detail"]


def test_a_failure_after_load_keeps_the_worker_serving(monkeypatch):
    """One bad frame must not cost the next session a model load."""
    tracker = _FakeTracker(fail_track=True)
    _install_runtime_stub(monkeypatch, tracker)
    events, code = _drive(
        monkeypatch, [_load_msg(), _init_msg(), _track_msg(), _init_msg(), _track_msg()]
    )
    assert code == 0
    assert [e["event"] for e in events] == ["ready", "box", "error", "box", "error"]
    assert "tracker exploded" in events[2]["detail"]


def test_a_track_before_init_is_an_error_not_a_crash(monkeypatch):
    tracker = _FakeTracker()
    _install_runtime_stub(monkeypatch, tracker)
    events, code = _drive(monkeypatch, [_load_msg(), _track_msg()])
    assert code == 0
    assert events[1]["event"] == "error"
    assert "track before init" in events[1]["detail"]


def test_a_frame_of_the_wrong_length_is_an_error(monkeypatch):
    """The worker checks too, even though the API layer checked first: it is the
    side that would otherwise reshape into nonsense."""
    tracker = _FakeTracker()
    _install_runtime_stub(monkeypatch, tracker)
    short = protocol.encode_message({"op": protocol.OP_INIT, "width": WIDTH, "height": HEIGHT,
                                     "box": BOX}, FRAME[:-4])
    events, code = _drive(monkeypatch, [_load_msg(), short])
    assert code == 0
    assert events[1]["event"] == "error"
    assert "ValueError" in events[1]["detail"]


def test_eof_before_load_exits_zero(monkeypatch):
    """A parent that closes the pipe is not a fault -- it is the exit signal."""
    _install_runtime_stub(monkeypatch, _FakeTracker())
    events, code = _drive(monkeypatch, [])
    assert (events, code) == ([], 0)


def test_shutdown_before_load_exits_zero(monkeypatch):
    _install_runtime_stub(monkeypatch, _FakeTracker())
    events, code = _drive(monkeypatch, [protocol.encode_message({"op": "shutdown"})])
    assert (events, code) == ([], 0)


def test_an_unknown_op_is_ignored(monkeypatch):
    tracker = _FakeTracker()
    _install_runtime_stub(monkeypatch, tracker)
    events, code = _drive(
        monkeypatch,
        [_load_msg(), protocol.encode_message({"op": "reset"}), _init_msg()],
    )
    # "reset" is deliberately NOT an op (re-seeding is what init already does),
    # and an unknown op must be inert rather than fatal.
    assert code == 0
    assert [e["event"] for e in events] == ["ready", "box"]


def test_a_desynchronised_pipe_is_fatal(monkeypatch):
    """There is no safe place to resume reading from, so the worker stops."""
    _install_runtime_stub(monkeypatch, _FakeTracker())
    events, code = _drive(
        monkeypatch, [_load_msg(), protocol.PREFIX + b"{ not json }\n"]
    )
    assert code == 1
    assert events[-1]["event"] == "error"
