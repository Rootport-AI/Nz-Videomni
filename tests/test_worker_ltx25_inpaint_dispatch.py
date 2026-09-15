"""Worker dispatch for an ``inpaint`` job on LTX 2.5 (台帳 §3-150).

The 2.5 twin of ``tests/test_worker_inpaint_dispatch.py``.
``engine25.worker._do_generate`` is the fork in the road: one message either
goes to ``engine25.inpaint25.run_inpaint``, or to
``engine25.outpaint25.run_outpaint``, or to the plain ``Ltx25Pipeline.generate``.
That choice is made by the mere PRESENCE of a payload block and nothing
downstream re-checks it -- so the routing, the geometry the branch builds, the
mutual exclusion and the shape of the terminal ``done`` event are what this file
pins.

ONE OF THOSE IS NOT LIKE THE OTHERS. The ``done`` event's ``inpaint`` key
carries the SUB-DICT, not the whole metadata dict that ``outpaint`` sends, and
the difference is not cosmetic: ``services/pipeline_manager.py`` writes
``metadata["inpaint"] = {**outcome.inpaint, **provenance}``, so a worker that
sent the whole dict would nest ``inpaint``/``ltx25``/``seed``/``width`` INSIDE
the block and break the contract ``mask_proof`` rides on. It is asserted by
IDENTITY here (``is``), because "equal to something the same shape" is exactly
what a wrong-shaped send would also satisfy.

NO PIPELINE AND NO GPU. ``_PIPE`` is a recording double and both drivers are
replaced by fakes, so no model is built and no CUDA context is created. What is
exercised is the worker's own code and nothing else.

Run with ``.venv-engine-ltx25`` and ``--noconftest``::

    .venv-engine-ltx25/Scripts/python.exe \\
        outputs/start-end-bridge-2026-09-07/implA_engine_runner/run_ltx25_pytest.py \\
        tests/test_worker_ltx25_inpaint_dispatch.py
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

from engine.inpaint.canvas import InpaintGeometry  # noqa: E402
from engine25 import inpaint25, outpaint25, worker  # noqa: E402
from engine25.inpaint25 import InpaintResult  # noqa: E402
from engine25.outpaint25 import OutpaintResult  # noqa: E402

CANVAS_W, CANVAS_H = 384, 256
SRC_W, SRC_H = 320, 256
FRAMES = 9


class _RecordingPipe:
    """Everything ``_do_generate`` reads off ``_PIPE``, and nothing else.

    The five echo methods and ``video_vae_kind`` are what the ``done`` event
    reports; the four job-state methods are the arm/reset pair the worker keeps
    outside its ``try``. A real ``Ltx25Pipeline`` is a model holder, so none of
    it is reachable without a checkpoint -- which is the whole reason this
    double exists.
    """

    sampler = "ancestral"
    video_vae_kind = "conv"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def set_acceleration_job(self, **_kw) -> None:
        pass

    def set_nag_job(self, _nag) -> None:
        pass

    def reset_nag_job(self) -> None:
        pass

    def reset_acceleration_job(self) -> None:
        pass

    def block_swap_prefetch_used(self) -> str:
        return "on"

    def fused_gguf_dequant_kernel_used(self) -> str:
        return "on"

    def keep_resident_used(self) -> str:
        return "off"

    def keep_resident_embeddings_used(self) -> str:
        return "off"

    def generate(self, **kwargs):
        self.calls.append(("plain", kwargs))
        return _result(OutpaintResult, {"ltx25": {"marker": "plain-unused"}})


def _result(cls, metadata: dict):
    """A real result object, so the ``done`` builder's nine reads are real too."""
    return cls(
        output_path="out.mp4",
        seed=7,
        width=SRC_W,
        height=SRC_H,
        num_frames=FRAMES,
        frame_rate=24.0,
        encode_fps=24,
        num_images=0,
        size_bytes=1024,
        seconds=12.5,
        metadata=metadata,
        phases={"21_stage1_denoise": {"seconds": 1.0}},
        peak_allocated_gib=9.0,
        peak_reserved_gib=10.0,
        rss_peak_gib=12.0,
        video_chunks=2,
        tiling="TileSizeConfig(...)",
    )


#: What the fake ``run_inpaint`` returns: the SUB-DICT the app reads and the
#: ``ltx25`` dict the worker re-sends, as two distinguishable objects.
INPAINT_BLOCK = {
    "canvas_width": CANVAS_W,
    "canvas_height": CANVAS_H,
    "mask_proof": {"decoded_frames": FRAMES, "white_ratio": 0.04, "dilated_ratio": 0.09},
}
LTX25_BLOCK = {"stage1_sampler": "ancestral", "marker": "ltx25"}


@pytest.fixture()
def harness(monkeypatch, tmp_path):
    """``_PIPE`` replaced, both drivers faked, ``_emit`` captured."""
    pipe = _RecordingPipe()
    events: list[dict] = []

    def _fake_run_inpaint(_pipeline, **kwargs):
        pipe.calls.append(("inpaint", kwargs))
        return _result(
            InpaintResult,
            {"inpaint": INPAINT_BLOCK, "seed": 7, "width": SRC_W, "ltx25": LTX25_BLOCK},
        )

    def _fake_run_outpaint(_pipeline, **kwargs):
        pipe.calls.append(("outpaint", kwargs))
        return _result(
            OutpaintResult,
            {"outpaint": {"marker": "outpaint"}, "ltx25": LTX25_BLOCK},
        )

    monkeypatch.setattr(worker, "_PIPE", pipe)
    monkeypatch.setattr(worker, "_log", lambda msg: None)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(inpaint25, "run_inpaint", _fake_run_inpaint)
    monkeypatch.setattr(outpaint25, "run_outpaint", _fake_run_outpaint)

    # ``_resolve_ic_reference`` checks the reference really is a file, so the
    # canvas has to exist on disk. Nothing ever reads it.
    (tmp_path / "inpaint_canvas.mp4").write_bytes(b"\x00" * 32)

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


def _outpaint_block(tmp_path) -> dict:
    return {
        "source_path": str(tmp_path / "src.mp4"),
        "canvas_width": 768,
        "canvas_height": 384,
        "pad_left": 128,
        "pad_right": 128,
        "pad_top": 64,
        "pad_bottom": 64,
    }


def _reference(tmp_path) -> dict:
    return {"path": str(tmp_path / "inpaint_canvas.mp4"), "strength": 1.0}


# ── 1. routing ──────────────────────────────────────────────────────────────
def test_an_inpaint_message_calls_run_inpaint(harness):
    pipe, _events, tmp_path = harness
    worker._do_generate(
        _msg(tmp_path, inpaint=_inpaint_block(tmp_path), reference_video=_reference(tmp_path))
    )

    assert [name for name, _ in pipe.calls] == ["inpaint"]
    kwargs = pipe.calls[0][1]
    # The canvas the IC-LoRA conditions on IS the substituted reference video,
    # and it rides BOTH as the canvas and as the reference -- one file, two
    # names, which is what lets the IC-LoRA plumbing stay ignorant of the mode.
    assert kwargs["canvas_path"] == str(tmp_path / "inpaint_canvas.mp4")
    assert kwargs["ic_reference"] == (str(tmp_path / "inpaint_canvas.mp4"), 1.0)
    # ...and the two files the canvas cannot supply.
    assert kwargs["source_path"] == str(tmp_path / "_inpaint_window.mp4")
    assert kwargs["mask_path"] == str(tmp_path / "mask.mp4")
    assert kwargs["blend_dilation_stage1"] == 5
    assert kwargs["blend_dilation_stage2"] == 2
    # ...and every shared knob.
    assert kwargs["prompt"] == "a clean wall" and kwargs["seed"] == 7
    assert kwargs["num_frames"] == FRAMES and kwargs["num_steps"] == 8
    assert kwargs["frame_rate"] == 24.0
    assert kwargs["progress"] is worker._emit_progress
    assert "height" not in kwargs and "width" not in kwargs, (
        "the two-stage driver takes both sizes from the geometry, never from "
        "the request's canvas fields"
    )
    assert "freeze_source_audio" not in kwargs, (
        "the app's inpaint block has no such field; passing one would invent a "
        "switch the API has not got"
    )


def test_the_branch_builds_a_real_inpaint_geometry(harness):
    """Not a dict, not four loose ints: the dataclass, so ``validate()`` and the
    pad derivation are the same ones the driver and the API use."""
    pipe, _events, tmp_path = harness
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
    """The panel sends both; a payload written before it existed sends neither,
    and the worker must then use the workflow's own values rather than zero."""
    pipe, _events, tmp_path = harness
    block = _inpaint_block(tmp_path)
    del block["blend_dilation_stage1"], block["blend_dilation_stage2"]
    worker._do_generate(_msg(tmp_path, inpaint=block, reference_video=_reference(tmp_path)))
    kwargs = pipe.calls[0][1]
    assert kwargs["blend_dilation_stage1"] == 5
    assert kwargs["blend_dilation_stage2"] == 2


def test_a_plain_message_never_reaches_run_inpaint(harness):
    pipe, _events, tmp_path = harness
    worker._do_generate(_msg(tmp_path))
    assert [name for name, _ in pipe.calls] == ["plain"]


def test_an_outpaint_message_still_routes_to_run_outpaint(harness):
    """The branch was inserted ABOVE outpaint's; this is the regression guard
    that says inserting it changed nothing for the feature already shipping."""
    pipe, _events, tmp_path = harness
    worker._do_generate(
        _msg(tmp_path, outpaint=_outpaint_block(tmp_path), reference_video=_reference(tmp_path))
    )
    assert [name for name, _ in pipe.calls] == ["outpaint"]


# ── 2. mutual exclusion ─────────────────────────────────────────────────────
def test_inpaint_and_outpaint_together_assert(harness):
    """The schema forbids the combination; the worker restates it where it would
    actually bite, because by this point the two would be fighting over one
    reference video and one output file. It fires BEFORE either branch, which is
    what makes the ``if``/``elif`` order a detail rather than a precedence
    rule."""
    pipe, _events, tmp_path = harness
    with pytest.raises(AssertionError, match="mutually exclusive"):
        worker._do_generate(_msg(
            tmp_path,
            inpaint=_inpaint_block(tmp_path),
            outpaint=_outpaint_block(tmp_path),
            reference_video=_reference(tmp_path),
        ))
    assert pipe.calls == [], "nothing may be dispatched once the assert fires"


def test_inpaint_without_a_reference_video_asserts(harness):
    """The reference video IS the green-filled canvas; without it there is
    nothing for the IC-LoRA to read the sentinel from -- and ``canvas_path``
    would have nothing to be."""
    pipe, _events, tmp_path = harness
    with pytest.raises(AssertionError, match="requires a reference video"):
        worker._do_generate(_msg(tmp_path, inpaint=_inpaint_block(tmp_path)))
    assert pipe.calls == []


# ── 3. the terminal done event ──────────────────────────────────────────────
def test_the_done_event_carries_the_inpaint_SUB_DICT(harness):
    """THE SHAPE, not merely the presence. ``services/pipeline_manager.py``
    merges this into ``metadata.json`` as ``{**outcome.inpaint, **provenance}``,
    so the whole metadata dict would nest the block inside itself and take
    ``mask_proof`` out of the place §6/§7 says it lives."""
    _pipe, events, tmp_path = harness
    worker._do_generate(
        _msg(tmp_path, inpaint=_inpaint_block(tmp_path), reference_video=_reference(tmp_path))
    )
    done = [e for e in events if e["event"] == "done"]
    assert len(done) == 1
    assert done[0]["inpaint"] is INPAINT_BLOCK
    # The sub-dict and NOT the wrapper: these three would be nested inside the
    # block if the whole metadata dict had been sent.
    assert "inpaint" not in done[0]["inpaint"]
    assert "ltx25" not in done[0]["inpaint"]
    assert "seed" not in done[0]["inpaint"]
    assert "outpaint" not in done[0]
    # ...and ``mask_proof`` survives the JSON framing the protocol puts it
    # through, which is the last hop before the app writes it out.
    relayed = json.loads(json.dumps(done[0]))
    assert relayed["inpaint"]["mask_proof"]["decoded_frames"] == FRAMES


def test_the_done_event_carries_the_drivers_own_ltx25_dict(harness):
    """台帳 §3-131's key, on 台帳 §3-150's job. For an inpaint job this is the
    ONLY route the sub-dict takes to metadata.json (``extra`` carries the
    ``inpaint`` block, not the whole metadata dict), so a worker that rebuilt it
    from the plain-generate fields would publish a second, drifting answer to
    "what did 2.5 do"."""
    _pipe, events, tmp_path = harness
    worker._do_generate(
        _msg(tmp_path, inpaint=_inpaint_block(tmp_path), reference_video=_reference(tmp_path))
    )
    done = [e for e in events if e["event"] == "done"][0]
    assert done["ltx25"] is LTX25_BLOCK


def test_an_outpaint_done_event_is_unchanged(harness):
    """The regression guard for the feature already shipping: outpaint still
    sends the WHOLE metadata dict (nothing app-side reads it) and still reuses
    the driver's ``ltx25``."""
    _pipe, events, tmp_path = harness
    worker._do_generate(
        _msg(tmp_path, outpaint=_outpaint_block(tmp_path), reference_video=_reference(tmp_path))
    )
    done = [e for e in events if e["event"] == "done"][0]
    assert done["outpaint"] == {
        "outpaint": {"marker": "outpaint"},
        "ltx25": LTX25_BLOCK,
    }
    assert done["ltx25"] is LTX25_BLOCK
    assert "inpaint" not in done


def test_a_plain_done_event_carries_neither_key(harness):
    """The additive contract at its last hop: a plain job's terminal event is
    byte-identical to what it was before either feature existed -- ``ltx25``
    aside, which rides EVERY job and is built fresh from the result's own
    fields."""
    _pipe, events, tmp_path = harness
    worker._do_generate(_msg(tmp_path))
    done = [e for e in events if e["event"] == "done"][0]
    assert "inpaint" not in done and "outpaint" not in done
    assert done["ltx25"] == {
        "encode_fps": 24,
        "video_chunks": 2,
        "tiling": "TileSizeConfig(...)",
        "size_bytes": 1024,
        "phases": {"21_stage1_denoise": {"seconds": 1.0}},
    }


def test_the_done_event_still_reports_the_nine_generation_facts(harness):
    """``InpaintResult`` is a superset of ``GenerationResult``, which is what
    lets the ``done`` builder stay branch-free -- and ``peak_vram_mb`` is what
    the app stores into ``metadata.json``'s ``vram_optimization`` block for the
    job-VRAM gate to read. A result that answered to fewer names would leave
    that number empty and the gate blind."""
    _pipe, events, tmp_path = harness
    worker._do_generate(
        _msg(tmp_path, inpaint=_inpaint_block(tmp_path), reference_video=_reference(tmp_path))
    )
    done = [e for e in events if e["event"] == "done"][0]
    assert done["seed_used"] == 7
    assert done["num_frames"] == FRAMES
    assert done["encode_fps"] == 24
    assert done["size_bytes"] == 1024
    assert done["seconds"] == 12.5
    assert done["rss_peak_gib"] == 12.0
    assert done["phases"] == {"21_stage1_denoise": {"seconds": 1.0}}
    assert done["peak_vram_mb"] == worker._gib_to_mb(9.0)
    assert done["peak_vram_reserved_mb"] == worker._gib_to_mb(10.0)
    assert done["vae_mode_used"] == "conv"
