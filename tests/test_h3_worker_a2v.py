"""engine_h3.worker: the A2V ``source_audio`` key of ``generate`` (台帳 §1-103, plan §2.4).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_worker_a2v.py

CPU only, no model: the parsing, the request building, the selftest CLI and the
log line of ``_do_generate`` (with a stub pipeline). The existing
``_generate_request`` cases are in tests/test_h3_engine.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine_h3 import worker  # noqa: E402
from engine_h3.pipeline_h3 import H3AudioInput  # noqa: E402

BASE = {"prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124, "output_path": "o.mp4"}


def _args(*extra):
    return worker._selftest_parser().parse_args([
        "--transformer", "t", "--text-encoder", "te", "--video-vae", "vv", "--audio-vae", "av",
        "--tokenizer-dir", "tok", "--output", "o.mp4", *extra,
    ])


def test_source_audio_absent_or_null_is_none():
    assert worker._source_audio({}) == (None, [])
    assert worker._source_audio({"source_audio": None}) == (None, [])


def test_source_audio_valid_and_default_strength():
    audio, ignored = worker._source_audio({"source_audio": {"path": "a.wav", "strength": 0.5}})
    assert audio == H3AudioInput(path="a.wav", strength=0.5) and ignored == []
    audio, _ = worker._source_audio({"source_audio": {"path": "a.wav"}})
    assert audio == H3AudioInput(path="a.wav", strength=0.5)
    audio, _ = worker._source_audio({"source_audio": {"path": "a.wav", "strength": None}})
    assert audio.strength == 0.5
    audio, _ = worker._source_audio({"source_audio": {"path": "a.wav", "strength": 1}})
    assert audio.strength == 1.0 and isinstance(audio.strength, float)


@pytest.mark.parametrize("bad", [
    "a.wav",                                   # not an object
    {},                                        # no path
    {"path": ""},
    {"path": "   "},
    {"path": 3},
    {"path": "a.wav", "strength": "loud"},
    {"path": "a.wav", "strength": True},
    {"path": "a.wav", "strength": float("nan")},
    {"path": "a.wav", "strength": float("inf")},
])
def test_source_audio_malformed_is_refused_in_japanese(bad):
    with pytest.raises(ValueError, match="source_audio"):
        worker._source_audio({"source_audio": bad})


def test_source_audio_unknown_inner_keys_are_reported():
    audio, ignored = worker._source_audio({"source_audio": {"path": "a.wav", "gain": 2, "channels": 1}})
    assert audio == H3AudioInput(path="a.wav") and ignored == ["channels", "gain"]
    assert worker._SOURCE_AUDIO_FIELDS == frozenset({"path", "strength"})
    assert "source_audio" in worker._GENERATE_FIELDS


def test_generate_request_with_and_without_source_audio():
    plain = worker._generate_request(BASE)
    assert plain.source_audio is None
    # The same request built without the key at all (today's shape).
    assert plain == worker._generate_request({**BASE, "source_audio": None})
    from engine_h3.pipeline_h3 import H3GenerateRequest

    assert plain == H3GenerateRequest(
        prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4",
        frame_rate=plain.frame_rate,
    )
    req = worker._generate_request({**BASE, "source_audio": {"path": "a.wav", "strength": 0.25}})
    assert req.source_audio == H3AudioInput(path="a.wav", strength=0.25)
    from dataclasses import replace

    assert replace(req, source_audio=None) == plain   # nothing else moved
    with pytest.raises(ValueError):
        worker._generate_request({**BASE, "source_audio": {"path": ""}})


def test_selftest_cli_source_audio():
    args = _args()
    assert (args.source_audio, args.audio_strength) == (None, 0.5)
    msg = worker._selftest_generate_message(args, "o.mp4")
    assert "source_audio" not in msg           # the golden key list is unchanged
    args = _args("--source-audio", "a.wav", "--audio-strength", "0.4")
    msg = worker._selftest_generate_message(args, "o.mp4")
    assert msg["source_audio"] == {"path": "a.wav", "strength": 0.4}
    assert list(msg)[-1] == "source_audio"
    assert worker._generate_request(msg).source_audio == H3AudioInput("a.wav", 0.4)


def test_do_generate_passes_source_audio_and_logs_ignored(monkeypatch):
    seen, logs, events = [], [], []

    class _Pipe:
        def generate(self, req):
            seen.append(req)
            return {"h3": {"phases": {}, "a2v": {"mask_value": 0.0}}}

    monkeypatch.setattr(worker, "_PIPE", _Pipe())
    monkeypatch.setattr(worker, "_ATTENTION", ("sdpa", False))
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    worker._do_generate({**BASE, "source_audio": {"path": "dir/a.wav", "strength": 0.5, "gain": 2}})
    assert seen[0].source_audio == H3AudioInput("dir/a.wav", 0.5)
    assert any("source_audio.gain" in line and "a.wav" in line for line in logs)
    assert events[0]["h3"]["a2v"] == {"mask_value": 0.0}   # passed through unchanged
    worker._do_generate(BASE)
    assert seen[1].source_audio is None
    assert not any("source_audio" in line for line in logs[-1:])
