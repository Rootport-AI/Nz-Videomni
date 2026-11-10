"""engine_h3.worker: the Outpainting ``outpaint`` key of ``generate`` (台帳 §1-125, 計画 §6.1).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_worker_outpaint.py

CPU only, no model: the parsing, the request building, the log line of
``_do_generate`` (a stub pipeline), the chain's ignored key, and the
pipeline's ``validate_request`` / the chain's ``_check_spec`` refusals.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from engine_h3 import worker  # noqa: E402
from engine_h3.pipeline_h3 import (  # noqa: E402
    H3AudioInput,
    H3GenerateRequest,
    H3ImageInput,
    H3OutpaintInput,
    H3ReferenceInput,
    validate_request,
)

BASE = {"prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124, "output_path": "o.mp4"}
OP = {"source_path": "dir/src.mp4", "pad_left": 12, "pad_right": 20, "pad_top": 0, "pad_bottom": 64,
      "audio_strength": 0.5, "feather_px": 30}


def test_outpaint_fields():
    assert worker._OUTPAINT_FIELDS == frozenset({
        "source_path", "pad_left", "pad_right", "pad_top", "pad_bottom", "audio_strength", "feather_px",
    })
    assert "outpaint" in worker._GENERATE_FIELDS
    assert "outpaint" not in worker._CHAIN_FIELDS     # a chain does not take it silently


def test_outpaint_absent_or_null_is_none():
    assert worker._outpaint({}) == (None, [])
    assert worker._outpaint({"outpaint": None}) == (None, [])


def test_outpaint_valid():
    op, ignored = worker._outpaint({"outpaint": OP})
    assert op == H3OutpaintInput(path="dir/src.mp4", pad_left=12, pad_right=20, pad_top=0, pad_bottom=64,
                                 audio_strength=0.5, feather_px=30)
    assert ignored == []
    op, _ = worker._outpaint({"outpaint": {**OP, "audio_strength": 1}})
    assert op.audio_strength == 1.0 and isinstance(op.audio_strength, float)


@pytest.mark.parametrize("bad", [
    "src.mp4",                                           # not an object
    {k: v for k, v in OP.items() if k != "source_path"},
    {**OP, "source_path": ""},
    {**OP, "source_path": "   "},
    {**OP, "source_path": 3},
    {k: v for k, v in OP.items() if k != "pad_left"},    # every pad is required
    {**OP, "pad_top": 1.0},
    {**OP, "pad_right": True},
    {**OP, "pad_bottom": "8"},
    {k: v for k, v in OP.items() if k != "feather_px"},
    {**OP, "feather_px": 2.5},
    {k: v for k, v in OP.items() if k != "audio_strength"},
    {**OP, "audio_strength": "half"},
    {**OP, "audio_strength": True},
    {**OP, "audio_strength": float("nan")},
    {**OP, "audio_strength": float("inf")},
])
def test_outpaint_malformed_is_refused_in_japanese(bad):
    with pytest.raises(ValueError, match="outpaint"):
        worker._outpaint({"outpaint": bad})


def test_outpaint_unknown_inner_keys_are_reported():
    # The LTX worker's keys are not read by this engine.
    msg = {"outpaint": {**OP, "blend_dilation_stage1": 5, "blend_dilation_stage2": 2,
                        "freeze_source_audio": False}}
    op, ignored = worker._outpaint(msg)
    assert op.path == "dir/src.mp4"
    assert ignored == ["blend_dilation_stage1", "blend_dilation_stage2", "freeze_source_audio"]


def test_generate_request_with_and_without_outpaint():
    plain = worker._generate_request(BASE)
    assert plain.outpaint is None
    assert plain == worker._generate_request({**BASE, "outpaint": None})
    req = worker._generate_request({**BASE, "outpaint": OP})
    assert req.outpaint == worker._outpaint({"outpaint": OP})[0]
    assert replace(req, outpaint=None) == plain          # nothing else moved
    with pytest.raises(ValueError):
        worker._generate_request({**BASE, "outpaint": {**OP, "source_path": ""}})


def _stub(monkeypatch):
    seen, logs, events = [], [], []

    class _Pipe:
        def generate(self, req):
            seen.append(req)
            return {"h3": {"phases": {}, "outpaint": {"fixed_tokens": 3}}}

    monkeypatch.setattr(worker, "_PIPE", _Pipe())
    monkeypatch.setattr(worker, "_ATTENTION", ("sdpa", False))
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    return seen, logs, events


def test_do_generate_passes_outpaint_and_logs_ignored(monkeypatch):
    seen, logs, events = _stub(monkeypatch)
    worker._do_generate({**BASE, "outpaint": {**OP, "blend_dilation_stage2": 2}})
    assert seen[0].outpaint == worker._outpaint({"outpaint": OP})[0]
    line = next(line for line in logs if line.startswith("generate:"))
    assert "outpaint=('src.mp4', (12, 20, 0, 64), 0.5, 30)" in line
    assert "ignored keys: outpaint.blend_dilation_stage2" in line
    assert events[0]["h3"]["outpaint"] == {"fixed_tokens": 3}   # passed through unchanged
    worker._do_generate(BASE)
    assert seen[1].outpaint is None
    assert "outpaint" not in logs[-1]


def test_chain_payload_outpaint_is_an_ignored_key(monkeypatch):
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
           "outpaint": OP}
    worker._do_generate_chain(msg)
    assert seen[0].base.outpaint is None
    line = next(line for line in logs if line.startswith("generate_chain:"))
    assert "ignored keys: outpaint" in line


# --------------------------------------------------------------------------- #
# the pipeline's validate_request and the chain's _check_spec
# --------------------------------------------------------------------------- #

def _req(tmp_path, **op_kw):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"x")
    kw = {"path": str(src), "pad_left": 12, "pad_right": 20, "pad_top": 0, "pad_bottom": 64,
          "audio_strength": 0.5, "feather_px": 30, **op_kw}
    return H3GenerateRequest(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4",
                             outpaint=H3OutpaintInput(**kw))


def test_validate_request_accepts_an_outpaint_job(tmp_path):
    validate_request(_req(tmp_path))
    validate_request(_req(tmp_path, audio_strength=0, feather_px=0))
    validate_request(_req(tmp_path, audio_strength=1.0, pad_left=0, pad_right=0, pad_top=1, pad_bottom=0))


@pytest.mark.parametrize("op_kw, match", [
    ({"path": ""}, "パスが空"),
    ({"pad_left": -1}, "pad_left は 0 以上"),
    ({"pad_top": 1.5}, "pad_top は整数"),
    ({"pad_bottom": True}, "pad_bottom は整数"),
    ({"pad_left": 0, "pad_right": 0, "pad_top": 0, "pad_bottom": 0}, "pad の合計は 1 以上"),
    ({"audio_strength": 1.5}, "音声の強度は 0〜1"),
    ({"audio_strength": -0.1}, "音声の強度は 0〜1"),
    ({"audio_strength": float("nan")}, "音声の強度は 0〜1"),
    ({"audio_strength": "x"}, "音声の強度は数値"),
    ({"feather_px": -1}, "なじませ幅は 0 以上の整数"),
    ({"feather_px": 2.0}, "なじませ幅は 0 以上の整数"),
])
def test_validate_request_refuses_bad_outpaint_values(tmp_path, op_kw, match):
    with pytest.raises(ValueError, match=match):
        validate_request(_req(tmp_path, **op_kw))


def test_validate_request_refuses_a_missing_source(tmp_path):
    with pytest.raises(ValueError, match="元動画が見つかりません"):
        validate_request(_req(tmp_path, path=str(tmp_path / "none.mp4")))


def test_validate_request_refuses_outpaint_with_other_conditioning(tmp_path):
    req = _req(tmp_path)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    with pytest.raises(ValueError, match="キーフレーム"):
        validate_request(replace(req, images=(H3ImageInput(str(audio), 0),)))
    with pytest.raises(ValueError, match="A2V"):
        validate_request(replace(req, source_audio=H3AudioInput(str(audio), 0.5)))
    with pytest.raises(ValueError, match="Ref2VA"):
        validate_request(replace(req, references=(H3ReferenceInput("audio", str(audio), 1),)))


def test_chain_check_spec_refuses_outpaint(tmp_path):
    from engine_h3.chain_h3 import H3ChainClipSpec, H3ChainSpec, _check_spec

    base = replace(_req(tmp_path), num_frames=124)
    spec = H3ChainSpec(base=base, clips=(H3ChainClipSpec("a", 124), H3ChainClipSpec("b", 39)), seed=1)
    with pytest.raises(ValueError, match="Outpainting は連結では使えません"):
        _check_spec(spec)
