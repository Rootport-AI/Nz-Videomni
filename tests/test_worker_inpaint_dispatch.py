"""Worker dispatch for an ``inpaint`` job (台帳 §3-55).

``engine.worker._do_generate`` is the fork in the road: one message either goes
to ``generate_inpaint``, or to ``generate_outpaint``, or to the plain
``generate``. That choice is made by the mere PRESENCE of a payload block, and
nothing downstream re-checks it — so the routing, the geometry the branch
builds, the mutual exclusion and the shape of the terminal ``done`` event are
what this file pins.

NO PIPELINE AND NO GPU. ``_PIPE`` is a recording double, and the handful of
``torch.cuda`` calls ``_do_generate`` makes for its VRAM record are stubbed, so
this never initialises a CUDA context. What is exercised is the worker's own
code and nothing else.

Run with ``.venv-engine`` and ``--noconftest``: importing ``engine.worker``
pulls in torch + ltx_core (the whole module skips in the app venv), and the app
conftest builds a FastAPI app the engine venv does not have.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

import torch  # noqa: E402

from engine import worker  # noqa: E402
from engine.inpaint.canvas import InpaintGeometry  # noqa: E402

CANVAS_W, CANVAS_H = 384, 256
SRC_W, SRC_H = 320, 256
FRAMES = 9


class _RecordingPipe:
    """Records which entry point was called, with which keyword arguments, and
    writes the output file the worker checks for afterwards."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def _record(self, name: str, kwargs: dict) -> dict:
        self.calls.append((name, kwargs))
        out = kwargs["output_path"]
        with open(out, "wb") as fh:
            fh.write(b"\x00" * 32)
        return {name: {"marker": name}}

    def generate_inpaint(self, **kwargs):
        return self._record("inpaint", kwargs)

    def generate_outpaint(self, **kwargs):
        return self._record("outpaint", kwargs)

    def generate(self, **kwargs):
        self._record("plain", kwargs)
        return None


@pytest.fixture()
def harness(monkeypatch, tmp_path):
    """``_PIPE`` replaced, the VRAM bookkeeping stubbed, ``_emit`` captured."""
    pipe = _RecordingPipe()
    events: list[dict] = []

    monkeypatch.setattr(worker, "_PIPE", pipe)
    monkeypatch.setattr(worker, "_log", lambda msg: None)
    monkeypatch.setattr(worker, "_log_job_start_vram", lambda: None)
    monkeypatch.setattr(worker, "_peak_vram_reserved_mb", lambda: 0)
    monkeypatch.setattr(worker, "_attention_used", lambda a, b: "sdpa")
    monkeypatch.setattr(worker, "_block_swap_prefetch_used", lambda: "off")
    monkeypatch.setattr(worker, "_keep_resident_used", lambda m, k: "off")
    monkeypatch.setattr(worker, "_fused_gguf_dequant_kernel_used", lambda: "off")
    monkeypatch.setattr(worker, "_vae_mode_used", lambda: "off")
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    # Never touch the device: these are the only torch.cuda calls in the path.
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)

    return pipe, events, tmp_path


def _msg(tmp_path, **extra) -> dict:
    msg = {
        "prompt": "a clean wall",
        "seed": 7,
        "width": CANVAS_W,
        "height": CANVAS_H,
        "num_frames": FRAMES,
        "frame_rate": 24.0,
        "num_steps": 8,
        "output_path": str(tmp_path / "output.mp4"),
    }
    msg.update(extra)
    return msg


def _inpaint_block(tmp_path, **over) -> dict:
    block = {
        "source_path": str(tmp_path / "_inpaint_window.mp4"),
        "mask_path": str(tmp_path / "mask.mp4"),
        "canvas_width": CANVAS_W,
        "canvas_height": CANVAS_H,
        "source_width": SRC_W,
        "source_height": SRC_H,
        "blend_dilation_stage1": 5,
        "blend_dilation_stage2": 2,
    }
    block.update(over)
    return block


def _reference(tmp_path) -> dict:
    return {"path": str(tmp_path / "inpaint_canvas.mp4"), "strength": 1.0}


# ── 1. routing ──────────────────────────────────────────────────────────────
def test_an_inpaint_message_calls_generate_inpaint(harness):
    pipe, events, tmp_path = harness
    worker._do_generate(
        _msg(tmp_path, inpaint=_inpaint_block(tmp_path), reference_video=_reference(tmp_path))
    )

    assert [name for name, _ in pipe.calls] == ["inpaint"]
    kwargs = pipe.calls[0][1]
    # The canvas the IC-LoRA conditions on IS the substituted reference video.
    assert kwargs["canvas_path"] == str(tmp_path / "inpaint_canvas.mp4")
    assert kwargs["source_path"] == str(tmp_path / "_inpaint_window.mp4")
    assert kwargs["mask_path"] == str(tmp_path / "mask.mp4")
    assert kwargs["blend_dilation_stage1"] == 5
    assert kwargs["blend_dilation_stage2"] == 2
    # ...and every shared knob rode through the same ``common`` dict.
    assert kwargs["prompt"] == "a clean wall" and kwargs["seed"] == 7
    assert kwargs["num_frames"] == FRAMES and kwargs["num_steps"] == 8
    assert "height" not in kwargs and "width" not in kwargs, (
        "the two-stage driver takes its size from the geometry, never from the "
        "request's canvas fields"
    )


def test_the_branch_builds_a_real_inpaint_geometry(harness):
    """Not a dict, not four loose ints: the dataclass, so ``validate()`` and the
    pad derivation are the same ones the pipeline uses."""
    pipe, events, tmp_path = harness
    worker._do_generate(
        _msg(tmp_path, inpaint=_inpaint_block(tmp_path), reference_video=_reference(tmp_path))
    )

    geom = pipe.calls[0][1]["geometry"]
    assert isinstance(geom, InpaintGeometry)
    assert (geom.canvas_width, geom.canvas_height) == (CANVAS_W, CANVAS_H)
    assert (geom.source_width, geom.source_height) == (SRC_W, SRC_H)
    assert (geom.pad_right, geom.pad_bottom) == (CANVAS_W - SRC_W, CANVAS_H - SRC_H)
    geom.validate()  # the geometry the worker built is a legal one


def test_the_dilations_default_when_the_block_omits_them(harness):
    """The frontend sends neither; an older app might send neither."""
    pipe, _events, tmp_path = harness
    block = _inpaint_block(tmp_path)
    del block["blend_dilation_stage1"], block["blend_dilation_stage2"]
    worker._do_generate(_msg(tmp_path, inpaint=block, reference_video=_reference(tmp_path)))
    kwargs = pipe.calls[0][1]
    assert kwargs["blend_dilation_stage1"] == 5
    assert kwargs["blend_dilation_stage2"] == 2


def test_a_plain_message_never_reaches_generate_inpaint(harness):
    pipe, events, tmp_path = harness
    worker._do_generate(_msg(tmp_path))
    assert [name for name, _ in pipe.calls] == ["plain"]


def test_an_outpaint_message_still_routes_to_generate_outpaint(harness):
    """The branch was inserted ABOVE outpaint's; this is the regression guard
    that says inserting it changed nothing for the feature already shipping."""
    pipe, events, tmp_path = harness
    worker._do_generate(_msg(
        tmp_path,
        outpaint={
            "source_path": str(tmp_path / "src.mp4"),
            "canvas_width": 768, "canvas_height": 384,
            "pad_left": 128, "pad_right": 128, "pad_top": 64, "pad_bottom": 64,
        },
        reference_video=_reference(tmp_path),
    ))
    assert [name for name, _ in pipe.calls] == ["outpaint"]


# ── 2. mutual exclusion ─────────────────────────────────────────────────────
def test_inpaint_and_outpaint_together_assert(harness):
    """The schema forbids the combination; the worker restates it where it would
    actually bite, because by this point the two would fight over one reference
    video and one output file."""
    pipe, _events, tmp_path = harness
    with pytest.raises(AssertionError, match="mutually exclusive"):
        worker._do_generate(_msg(
            tmp_path,
            inpaint=_inpaint_block(tmp_path),
            outpaint={
                "source_path": None, "canvas_width": 768, "canvas_height": 384,
                "pad_left": 128, "pad_right": 128, "pad_top": 64, "pad_bottom": 64,
            },
            reference_video=_reference(tmp_path),
        ))
    assert pipe.calls == [], "nothing may be dispatched once the assert fires"


def test_inpaint_without_a_reference_video_asserts(harness):
    """The reference video IS the green canvas; without it there is nothing for
    the IC-LoRA to read the sentinel from."""
    pipe, _events, tmp_path = harness
    with pytest.raises(AssertionError, match="requires a reference video"):
        worker._do_generate(_msg(tmp_path, inpaint=_inpaint_block(tmp_path)))
    assert pipe.calls == []


# ── 3. the terminal done event ──────────────────────────────────────────────
def test_the_done_event_carries_the_inpaint_block(harness):
    pipe, events, tmp_path = harness
    worker._do_generate(
        _msg(tmp_path, inpaint=_inpaint_block(tmp_path), reference_video=_reference(tmp_path))
    )
    done = [e for e in events if e["event"] == "done"]
    assert len(done) == 1
    assert done[0]["inpaint"] == {"marker": "inpaint"}
    assert "outpaint" not in done[0]
    # It has to survive the JSON framing the protocol puts it through.
    assert json.loads(json.dumps(done[0]))["inpaint"] == {"marker": "inpaint"}


def test_an_outpaint_done_event_is_unchanged(harness):
    pipe, events, tmp_path = harness
    worker._do_generate(_msg(
        tmp_path,
        outpaint={
            "source_path": None, "canvas_width": 768, "canvas_height": 384,
            "pad_left": 128, "pad_right": 128, "pad_top": 64, "pad_bottom": 64,
        },
        reference_video=_reference(tmp_path),
    ))
    done = [e for e in events if e["event"] == "done"][0]
    assert done["outpaint"] == {"marker": "outpaint"}
    assert "inpaint" not in done


def test_a_plain_done_event_carries_neither_key(harness):
    """The additive contract at its last hop: a plain job's terminal event is
    byte-identical to what it was before either feature existed."""
    pipe, events, tmp_path = harness
    worker._do_generate(_msg(tmp_path))
    done = [e for e in events if e["event"] == "done"][0]
    assert "inpaint" not in done and "outpaint" not in done
