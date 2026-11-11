"""engine_h3.worker: the Inpainting ``inpaint`` key of ``generate`` (台帳 §1-124, 計画 §4.4).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_worker_inpaint.py

CPU only, no model: the parsing, the request building, the refusal of
``outpaint`` with ``inpaint``, the log line of ``_do_generate`` (a stub
pipeline) and the chain's ignored key.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from engine_h3 import worker  # noqa: E402
from engine_h3.pipeline_h3 import H3InpaintInput  # noqa: E402

BASE = {"prompt": "p", "seed": 1, "width": 1280, "height": 736, "num_frames": 124, "output_path": "o.mp4"}
IP = {"source_path": "dir/src.mp4", "mask_path": "dir/mask.mp4", "audio_strength": 0.5, "feather_px": 30}
OP = {"source_path": "dir/src.mp4", "pad_left": 12, "pad_right": 20, "pad_top": 0, "pad_bottom": 64,
      "audio_strength": 0.5, "feather_px": 30}


def test_inpaint_fields():
    assert worker._INPAINT_FIELDS == frozenset({"source_path", "mask_path", "audio_strength", "feather_px"})
    assert "inpaint" in worker._GENERATE_FIELDS
    assert "inpaint" not in worker._CHAIN_FIELDS     # a chain does not take it silently


def test_inpaint_absent_or_null_is_none():
    assert worker._inpaint({}) == (None, [])
    assert worker._inpaint({"inpaint": None}) == (None, [])


def test_inpaint_valid():
    ip, ignored = worker._inpaint({"inpaint": IP})
    assert ip == H3InpaintInput(path="dir/src.mp4", mask_path="dir/mask.mp4", audio_strength=0.5, feather_px=30)
    assert ignored == []
    ip, _ = worker._inpaint({"inpaint": {**IP, "audio_strength": 1}})
    assert ip.audio_strength == 1.0 and isinstance(ip.audio_strength, float)


@pytest.mark.parametrize("bad", [
    "src.mp4",                                           # not an object
    {k: v for k, v in IP.items() if k != "source_path"},
    {**IP, "source_path": ""},
    {**IP, "source_path": "   "},
    {**IP, "source_path": 3},
    {k: v for k, v in IP.items() if k != "mask_path"},
    {**IP, "mask_path": ""},
    {**IP, "mask_path": None},
    {k: v for k, v in IP.items() if k != "feather_px"},
    {**IP, "feather_px": 2.5},
    {**IP, "feather_px": True},
    {k: v for k, v in IP.items() if k != "audio_strength"},
    {**IP, "audio_strength": "half"},
    {**IP, "audio_strength": True},
    {**IP, "audio_strength": float("nan")},
    {**IP, "audio_strength": float("inf")},
])
def test_inpaint_malformed_is_refused_in_japanese(bad):
    with pytest.raises(ValueError, match="inpaint"):
        worker._inpaint({"inpaint": bad})


def test_inpaint_unknown_inner_keys_are_reported():
    msg = {"inpaint": {**IP, "blend_dilation_stage1": 5, "window_start_sec": 1.0}}
    ip, ignored = worker._inpaint(msg)
    assert ip.mask_path == "dir/mask.mp4"
    assert ignored == ["blend_dilation_stage1", "window_start_sec"]


def test_generate_request_with_and_without_inpaint():
    plain = worker._generate_request(BASE)
    assert plain.inpaint is None
    assert plain == worker._generate_request({**BASE, "inpaint": None})
    req = worker._generate_request({**BASE, "inpaint": IP})
    assert req.inpaint == worker._inpaint({"inpaint": IP})[0] and req.outpaint is None
    assert replace(req, inpaint=None) == plain           # nothing else moved
    with pytest.raises(ValueError):
        worker._generate_request({**BASE, "inpaint": {**IP, "mask_path": ""}})


def test_outpaint_and_inpaint_together_are_refused():
    with pytest.raises(ValueError, match="outpaint と inpaint"):
        worker._generate_request({**BASE, "outpaint": OP, "inpaint": IP})


def _stub(monkeypatch):
    seen, logs, events = [], [], []

    class _Pipe:
        def generate(self, req):
            seen.append(req)
            return {"h3": {"phases": {}, "inpaint": {"fixed_tokens_total": 3}}}

    monkeypatch.setattr(worker, "_PIPE", _Pipe())
    monkeypatch.setattr(worker, "_ATTENTION", ("sdpa", False))
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    return seen, logs, events


def test_do_generate_passes_inpaint_and_logs_ignored(monkeypatch):
    seen, logs, events = _stub(monkeypatch)
    worker._do_generate({**BASE, "inpaint": {**IP, "blend_dilation_stage2": 2}})
    assert seen[0].inpaint == worker._inpaint({"inpaint": IP})[0]
    line = next(line for line in logs if line.startswith("generate:"))
    assert "inpaint=('src.mp4', 'mask.mp4', 0.5, 30)" in line
    assert "ignored keys: inpaint.blend_dilation_stage2" in line
    assert events[0]["h3"]["inpaint"] == {"fixed_tokens_total": 3}   # passed through unchanged
    worker._do_generate(BASE)
    assert seen[1].inpaint is None
    assert "inpaint" not in logs[-1]


def test_do_generate_refuses_outpaint_and_inpaint(monkeypatch):
    seen, _logs, _events = _stub(monkeypatch)
    with pytest.raises(ValueError, match="outpaint と inpaint"):
        worker._do_generate({**BASE, "outpaint": OP, "inpaint": IP})
    assert seen == []


def test_chain_payload_inpaint_is_an_ignored_key(monkeypatch):
    seen, logs = [], []

    def run(pipe, spec, progress, log):
        seen.append(spec)
        return {"h3": {}}

    monkeypatch.setattr(worker, "_PIPE", object())
    monkeypatch.setattr(worker, "_ATTENTION", ("sdpa", False))
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: None)
    monkeypatch.setattr("engine_h3.chain_h3.run_chain_h3", run)
    msg = {"op": "generate_chain", "prompt": "a", "seed": 5, "width": 640, "height": 384, "frame_rate": 24,
           "output_path": "o.mp4", "clips": [{"prompt": "a", "num_frames": 124}, {"prompt": "b", "num_frames": 39}],
           "inpaint": IP}
    worker._do_generate_chain(msg)
    assert seen[0].base.inpaint is None
    line = next(line for line in logs if line.startswith("generate_chain:"))
    assert "ignored keys: inpaint" in line
