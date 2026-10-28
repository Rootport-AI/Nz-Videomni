"""engine_h3.worker: the Ref2VA ``references`` key of ``generate`` (台帳 §1-107, plan §3.1).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_worker_ref.py

CPU only, no model: the parsing, the request building, the selftest CLI, the
log line of ``_do_generate`` (with a stub pipeline) and the pipeline's
``validate_request`` for references. The pipeline's own reference tests (the
denoise, the encode stage, ``done``) are in tests/test_h3_engine.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine_h3 import worker  # noqa: E402
from engine_h3.pipeline_h3 import H3GenerateRequest, H3ReferenceInput, validate_request  # noqa: E402

BASE = {"prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124, "output_path": "o.mp4"}


def _args(*extra):
    return worker._selftest_parser().parse_args([
        "--transformer", "t", "--text-encoder", "te", "--video-vae", "vv", "--audio-vae", "av",
        "--tokenizer-dir", "tok", "--output", "o.mp4", *extra,
    ])


def test_reference_fields():
    assert "references" in worker._GENERATE_FIELDS
    assert worker._REFERENCE_FIELDS == frozenset({"kind", "path", "slot", "max_sec"})


def test_references_absent_null_or_empty_is_none():
    assert worker._references({}) == ((), [])
    assert worker._references({"references": None}) == ((), [])
    assert worker._references({"references": []}) == ((), [])


def test_references_parsed_in_the_order_sent():
    refs, ignored = worker._references({"references": [
        {"kind": "video", "path": "v.mp4", "slot": 2, "max_sec": 5},
        {"kind": "picture", "path": "p.png", "slot": 1},
        {"kind": "audio", "path": "a.wav", "slot": 1, "max_sec": None},
    ]})
    assert refs == (
        H3ReferenceInput("video", "v.mp4", 2, 5.0),
        H3ReferenceInput("picture", "p.png", 1, None),
        H3ReferenceInput("audio", "a.wav", 1, None),
    )
    assert isinstance(refs[0].max_sec, float) and ignored == []


def test_unknown_inner_keys_are_returned_for_the_log():
    refs, ignored = worker._references({"references": [
        {"kind": "picture", "path": "p.png", "slot": 1, "strength": 0.5, "id": "abc"},
        {"kind": "audio", "path": "a.wav", "slot": 1, "id": "def"},
    ]})
    assert len(refs) == 2 and ignored == ["id", "strength"]


@pytest.mark.parametrize("bad", [
    {"kind": "picture"},                                        # not a list
    ["p.png"],                                                  # not an object
    [{"path": "p.png", "slot": 1}],                             # no kind
    [{"kind": 3, "path": "p.png", "slot": 1}],
    [{"kind": "picture", "slot": 1}],                           # no path
    [{"kind": "picture", "path": "  ", "slot": 1}],
    [{"kind": "picture", "path": 3, "slot": 1}],
    [{"kind": "picture", "path": "p.png"}],                     # no slot
    [{"kind": "picture", "path": "p.png", "slot": "1"}],
    [{"kind": "picture", "path": "p.png", "slot": 1.0}],
    [{"kind": "picture", "path": "p.png", "slot": True}],
    [{"kind": "video", "path": "v.mp4", "slot": 1, "max_sec": "5"}],
    [{"kind": "video", "path": "v.mp4", "slot": 1, "max_sec": True}],
    [{"kind": "video", "path": "v.mp4", "slot": 1, "max_sec": float("inf")}],
    [{"kind": "video", "path": "v.mp4", "slot": 1, "max_sec": float("nan")}],
])
def test_malformed_references_are_refused_in_japanese(bad):
    with pytest.raises(ValueError, match="references"):
        worker._references({"references": bad})


def test_generate_request_carries_the_references():
    msg = {**BASE, "references": [{"kind": "audio", "path": "a.wav", "slot": 3}]}
    req = worker._generate_request(msg)
    assert req.references == (H3ReferenceInput("audio", "a.wav", 3),)
    assert worker._generate_request(dict(BASE)).references == ()


def test_do_generate_logs_the_ignored_inner_keys(monkeypatch):
    seen, logs, events = {}, [], []

    class StubPipe:
        def generate(self, req):
            seen["req"] = req
            return {"attention_used": "sdpa"}

    monkeypatch.setattr(worker, "_PIPE", StubPipe())
    monkeypatch.setattr(worker, "_ATTENTION", ("sdpa", False))
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append(event))
    worker._do_generate({**BASE, "op": "generate", "references": [
        {"kind": "picture", "path": "C:/x/p.png", "slot": 1, "id": "abc"},
        {"kind": "video", "path": "C:/x/v.mp4", "slot": 1, "max_sec": 4.5},
    ]})
    assert events == ["done"]
    assert seen["req"].references == (
        H3ReferenceInput("picture", "C:/x/p.png", 1), H3ReferenceInput("video", "C:/x/v.mp4", 1, 4.5))
    line = logs[-1]
    assert "references=[('picture', 1, 'p.png', None), ('video', 1, 'v.mp4', 4.5)]" in line
    assert "(ignored keys: references[].id)" in line


def test_do_generate_without_references_logs_as_before(monkeypatch):
    logs = []

    class StubPipe:
        def generate(self, req):
            return {}

    monkeypatch.setattr(worker, "_PIPE", StubPipe())
    monkeypatch.setattr(worker, "_ATTENTION", ("sdpa", False))
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: None)
    worker._do_generate({**BASE, "op": "generate"})
    assert "references=" not in logs[-1] and "ignored keys" not in logs[-1]


def test_selftest_reference_options():
    assert "references" not in worker._selftest_generate_message(_args(), "o.mp4")   # as before
    args = _args("--ref-audio", "a1.wav", "--ref-image", "p1.png", "--ref-video", "v1.mp4",
                 "--ref-image", "p2.png", "--ref-audio", "a2.wav")
    msg = worker._selftest_generate_message(args, "o.mp4")
    assert list(msg)[-1] == "references"
    assert msg["references"] == [
        {"kind": "picture", "path": "p1.png", "slot": 1, "max_sec": None},
        {"kind": "picture", "path": "p2.png", "slot": 2, "max_sec": None},
        {"kind": "video", "path": "v1.mp4", "slot": 1, "max_sec": None},
        {"kind": "audio", "path": "a1.wav", "slot": 1, "max_sec": None},
        {"kind": "audio", "path": "a2.wav", "slot": 2, "max_sec": None},
    ]
    refs, ignored = worker._references(msg)
    assert len(refs) == 5 and ignored == []


# --------------------------------------------------------------------------- #
# validate_request: the reference checks, before anything is loaded
# --------------------------------------------------------------------------- #

def _req(*refs):
    return H3GenerateRequest(**BASE, references=tuple(refs))


def test_validate_accepts_one_of_each_kind_and_gaps_in_slots(tmp_path):
    f = tmp_path / "r.bin"
    f.write_bytes(b"x")
    validate_request(_req(
        H3ReferenceInput("picture", str(f), 2), H3ReferenceInput("picture", str(f), 5),
        H3ReferenceInput("video", str(f), 2, 7.5), H3ReferenceInput("audio", str(f), 2, 1),
    ))
    validate_request(_req())


@pytest.mark.parametrize("make, match", [
    (lambda f: [H3ReferenceInput("image", f, 1)], "種類"),
    (lambda f: [H3ReferenceInput("Picture", f, 1)], "種類"),
    (lambda f: [H3ReferenceInput("picture", f, 0)], "1 以上"),
    (lambda f: [H3ReferenceInput("audio", f, -1)], "1 以上"),
    (lambda f: [H3ReferenceInput("picture", f, "1")], "整数"),
    (lambda f: [H3ReferenceInput("picture", f, False)], "整数"),
    (lambda f: [H3ReferenceInput("picture", f, 1), H3ReferenceInput("picture", f, 1)], "重複"),
    (lambda f: [H3ReferenceInput("picture", "", 1)], "パスが空"),
    (lambda f: [H3ReferenceInput("video", f + ".missing", 1)], "見つかりません"),
    (lambda f: [H3ReferenceInput("video", f, 1, 0)], "正の有限"),
    (lambda f: [H3ReferenceInput("audio", f, 1, -2.0)], "正の有限"),
    (lambda f: [H3ReferenceInput("audio", f, 1, float("inf"))], "正の有限"),
    (lambda f: [H3ReferenceInput("audio", f, 1, True)], "正の有限"),
])
def test_validate_refuses_bad_references_in_japanese(tmp_path, make, match):
    f = tmp_path / "r.bin"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match=match):
        validate_request(_req(*make(str(f))))


def test_same_slot_in_different_kinds_is_allowed(tmp_path):
    f = tmp_path / "r.bin"
    f.write_bytes(b"x")
    validate_request(_req(H3ReferenceInput("picture", str(f), 1), H3ReferenceInput("video", str(f), 1),
                          H3ReferenceInput("audio", str(f), 1)))
