"""engine_h3.worker: the ``te_combined_read`` key of ``generate`` and
``generate_chain`` (台帳 1-303 方向 B).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_worker_te_combined.py

CPU only, no model: the request building and the log lines of
``_do_generate`` / ``_do_generate_chain`` (with stubs). The combined read itself
is in tests/test_h3_te_combined.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine_h3 import worker  # noqa: E402

BASE = {"prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124, "output_path": "o.mp4"}


def test_te_combined_read_key(monkeypatch):
    """Read by ``generate`` and ``generate_chain`` (absent / null = off),
    logged, never an ignored key."""
    assert "te_combined_read" in worker._GENERATE_FIELDS
    assert "te_combined_read" in worker._CHAIN_FIELDS
    assert worker._generate_request(dict(BASE)).te_combined_read is False
    assert worker._generate_request({**BASE, "te_combined_read": None}).te_combined_read is False
    assert worker._generate_request({**BASE, "te_combined_read": False}).te_combined_read is False
    assert worker._generate_request({**BASE, "te_combined_read": True}).te_combined_read is True
    seen, logs = {}, []

    class StubPipe:
        def generate(self, req):
            seen["req"] = req
            return {}

    monkeypatch.setattr(worker, "_PIPE", StubPipe())
    monkeypatch.setattr(worker, "_ATTENTION", ("sdpa", False))
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: None)
    worker._do_generate({**BASE, "op": "generate", "te_combined_read": True})
    assert seen["req"].te_combined_read is True
    assert "te_combined_read=True" in logs[-1] and "ignored keys" not in logs[-1]
    worker._do_generate({**BASE, "op": "generate"})
    assert seen["req"].te_combined_read is False and "te_combined_read=False" in logs[-1]


def test_te_combined_read_reaches_the_chain_base(monkeypatch):
    import engine_h3.chain_h3 as CH

    seen, logs = {}, []

    def run(pipe, spec, progress, log):
        seen["spec"] = spec
        return {}

    monkeypatch.setattr(worker, "_PIPE", object())
    monkeypatch.setattr(worker, "_ATTENTION", ("sdpa", False))
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: None)
    monkeypatch.setattr(CH, "run_chain_h3", run)
    msg = {
        "op": "generate_chain", "seed": 1, "width": 640, "height": 384, "output_path": "o.mp4",
        "clips": [{"prompt": "a", "num_frames": 124}, {"prompt": "b", "num_frames": 39}],
    }
    worker._do_generate_chain({**msg, "te_combined_read": True})
    spec = seen["spec"]
    assert spec.base.te_combined_read is True
    assert all(r.te_combined_read is True for r in CH.clip_requests(spec))   # dataclasses.replace keeps it
    assert "te_combined_read=True" in logs[-1] and "ignored keys" not in logs[-1]
    worker._do_generate_chain(msg)
    assert seen["spec"].base.te_combined_read is False and "te_combined_read=False" in logs[-1]


def test_selftest_message_does_not_send_it():
    args = worker._selftest_parser().parse_args([
        "--transformer", "t", "--text-encoder", "te", "--video-vae", "vv", "--audio-vae", "av",
        "--tokenizer-dir", "tok", "--output", "o.mp4",
    ])
    assert "te_combined_read" not in worker._selftest_generate_message(args, "o.mp4")
