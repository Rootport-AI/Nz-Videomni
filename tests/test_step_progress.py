"""F2 (G3 gate) per-step denoise progress tests.

Two GPU-free layers:

* engine/progress_shim.py — the tqdm stand-in the worker installs into the
  wheel's samplers module. Pure Python (no torch / ltx_*), so the app venv can
  drive it directly: pass-through iteration, per-step emission, phase
  inference, and the emit-failure fail-safe.
* services/ltx_runner.py — the worker-event receipt loop: the single-generate
  path now ACCEPTS progress events (it used to RuntimeError on them), step and
  total pass through to the progress callback, and the job fraction is
  monotone non-decreasing across coarse + per-step event interleavings.
"""

from __future__ import annotations

import logging

import pytest

from engine import progress_shim
from services.ltx_runner import _RealBackend


@pytest.fixture(autouse=True)
def _reset_shim_state():
    """The shim coordinates phase via module globals; leave them clean."""
    yield
    progress_shim.set_emitter(None)
    progress_shim.end_op()


def _capture_emitter(sink: list):
    def emit(**fields):
        sink.append(fields)

    return emit


# ------------------------------------------------------------ tqdm shim (engine)


def test_shim_yields_items_unchanged_and_emits_per_step():
    emitted: list[dict] = []
    progress_shim.set_emitter(_capture_emitter(emitted))
    progress_shim.set_phase("stage1_denoise", outer_index=1, outer_total=3)

    items = ["a", "b", "c"]
    out = list(progress_shim.TqdmShim(items))

    assert out == items  # observation only — the wheel sees its own items
    assert [e["index"] for e in emitted] == [1, 2, 3]  # 1-based completed count
    assert all(e["total"] == 3 for e in emitted)
    assert all(e["stage"] == "stage1_denoise" for e in emitted)
    assert all(e["outer_index"] == 1 and e["outer_total"] == 3 for e in emitted)
    # it_s is the cumulative steps/s (may be None only if the clock stalls).
    assert all(("it_s" in e) for e in emitted)


def test_shim_single_op_infers_stage1_then_stage2():
    emitted: list[dict] = []
    progress_shim.set_emitter(_capture_emitter(emitted))
    progress_shim.begin_single_op()

    list(progress_shim.TqdmShim(range(8)))   # wheel loop #1 -> stage 1
    list(progress_shim.TqdmShim(range(3)))   # wheel loop #2 -> stage 2
    list(progress_shim.TqdmShim(range(2)))   # unexpected extra loop -> generic

    stages = [e["stage"] for e in emitted]
    assert stages[:8] == ["stage1_denoise"] * 8
    assert stages[8:11] == ["stage2_denoise"] * 3
    assert stages[11:] == ["denoise"] * 2
    assert [e["outer_index"] for e in emitted] == [None] * 13


def test_shim_survives_broken_emitter():
    def broken(**fields):
        raise RuntimeError("emit pipe broke")

    progress_shim.set_emitter(broken)
    progress_shim.set_phase("stage1_denoise")
    # Generation must not be interrupted by observation failures.
    assert list(progress_shim.TqdmShim([1, 2, 3])) == [1, 2, 3]


def test_shim_without_emitter_is_a_plain_wrapper():
    progress_shim.set_emitter(None)
    assert list(progress_shim.TqdmShim(iter([1, 2]))) == [1, 2]  # no len() needed
    assert len(progress_shim.TqdmShim([1, 2, 3])) == 3


# --------------------------------------- runner event-receipt loop (_RealBackend)


def _backend_with_events(events: list[dict]) -> _RealBackend:
    it = iter(events)
    be = _RealBackend.__new__(_RealBackend)
    be._read_event = lambda timeout=None: next(it)  # type: ignore[attr-defined]
    return be


def test_single_generate_loop_accepts_step_progress_and_returns_done():
    """The old one-shot read turned the first progress event into a job
    failure; the receipt loop forwards step/total and returns the terminal."""
    events = [
        {"event": "progress", "stage": "stage1_denoise", "index": 1, "total": 8, "it_s": 0.5},
        {"event": "progress", "stage": "stage1_denoise", "index": 8, "total": 8, "it_s": 0.5},
        {"event": "progress", "stage": "stage2_denoise", "index": 3, "total": 3, "it_s": 0.4},
        {"event": "done", "seed_used": 1, "peak_vram_mb": 1000},
    ]
    calls: list[tuple] = []
    be = _backend_with_events(events)

    result = be._read_worker_events(
        lambda step, total, frac, stage=None: calls.append((step, total, frac, stage)),
        chain=False,
        prefix="generate",
    )

    assert result["event"] == "done"
    assert [(c[0], c[1]) for c in calls] == [(1, 8), (8, 8), (3, 3)]  # step passthrough
    assert [c[3] for c in calls] == ["stage1_denoise", "stage1_denoise", "stage2_denoise"]
    fracs = [c[2] for c in calls]
    assert fracs == sorted(fracs)  # monotone
    assert fracs[-1] <= 0.85 < 0.90  # stays below the post-done 0.90 milestone


def test_single_generate_loop_returns_error_terminal():
    be = _backend_with_events(
        [
            {"event": "progress", "stage": "stage1_denoise", "index": 1, "total": 8},
            {"event": "error", "detail": "boom"},
        ]
    )
    result = be._read_worker_events(lambda *a: None, chain=False, prefix="generate")
    assert result == {"event": "error", "detail": "boom"}


def test_single_generate_loop_returns_unexpected_event_for_caller_check():
    """Non-progress, non-terminal events surface to the caller unchanged (the
    existing 'unexpected event' RuntimeError in generate() still fires)."""
    be = _backend_with_events([{"event": "ready"}])
    result = be._read_worker_events(lambda *a: None, chain=False, prefix="generate")
    assert result == {"event": "ready"}


def test_chain_interleaved_step_and_coarse_progress_is_monotone(caplog):
    """Chain: per-step shim events interleave with the coarse per-segment /
    per-tile events; fractions never regress and land on the historical coarse
    milestones (0.275/0.5 for stage1 seg 1/2, 0.9 for last tile, 0.95 decode)."""
    events = [
        {"event": "progress", "stage": "encode", "index": 0, "total": 1},
        # segment 0/2 per-step, then its coarse completion event
        {"event": "progress", "stage": "stage1_denoise", "index": 4, "total": 8,
         "outer_index": 0, "outer_total": 2, "it_s": 0.5},
        {"event": "progress", "stage": "stage1_denoise", "index": 8, "total": 8,
         "outer_index": 0, "outer_total": 2, "it_s": 0.5},
        {"event": "progress", "stage": "stage1", "index": 0, "total": 2},
        # segment 1/2
        {"event": "progress", "stage": "stage1_denoise", "index": 8, "total": 8,
         "outer_index": 1, "outer_total": 2, "it_s": 0.5},
        {"event": "progress", "stage": "stage1", "index": 1, "total": 2},
        # tile 0/1
        {"event": "progress", "stage": "stage2_denoise", "index": 1, "total": 3,
         "outer_index": 0, "outer_total": 1, "it_s": 0.4},
        {"event": "progress", "stage": "stage2_denoise", "index": 3, "total": 3,
         "outer_index": 0, "outer_total": 1, "it_s": 0.4},
        {"event": "progress", "stage": "tile", "index": 0, "total": 1},
        {"event": "progress", "stage": "decode", "index": 0, "total": 1},
        {"event": "done", "seed_used": 7},
    ]
    calls: list[tuple] = []
    be = _backend_with_events(events)

    with caplog.at_level(logging.INFO, logger="ltx.runner"):
        result = be._read_chain_events(
            lambda step, total, frac, stage=None: calls.append((step, total, frac, stage))
        )

    assert result["event"] == "done"
    fracs = [c[2] for c in calls]
    assert fracs == sorted(fracs), fracs
    by_stage = {}
    for step, total, frac, stage in calls:
        by_stage.setdefault(stage, []).append((step, total, frac))
    # per-step events pass step/total through; coarse events keep (None, None).
    assert by_stage["stage1_denoise"][0][:2] == (4, 8)
    assert by_stage["stage1"] == [(None, None, 0.275), (None, None, 0.5)]
    assert by_stage["tile"] == [(None, None, 0.9)]
    assert by_stage["decode"] == [(None, None, 0.95)]
    # segment interpolation: seg 0 step 4/8 -> 0.05 + 0.45 * (0.5 / 2) = 0.1625
    assert by_stage["stage1_denoise"][0][2] == pytest.approx(0.163, abs=1e-3)
    # encode maps just above the 0.03 dispatch fraction
    assert by_stage["encode"] == [(None, None, 0.04)]

    joined = "\n".join(
        rec.getMessage() for rec in caplog.records if rec.name == "ltx.runner"
    )
    assert "chain text encode started" in joined
    assert "chain stage-1 denoise [1/2] started (8 steps)" in joined
    assert "chain stage-1 denoise [1/2] 8/8 steps" in joined
    assert "chain stage-2 denoise [1/1] 3/3 steps" in joined


def test_unknown_stage_never_moves_the_fraction():
    events = [
        {"event": "progress", "stage": "tile", "index": 0, "total": 2},   # 0.7
        {"event": "progress", "stage": "mystery", "index": 0, "total": 1},
        {"event": "progress", "stage": "tile", "index": 1, "total": 2},   # 0.9
        {"event": "done"},
    ]
    fracs: list[float] = []
    be = _backend_with_events(events)
    be._read_worker_events(
        lambda step, total, frac, stage=None: fracs.append(frac),
        chain=True,
        prefix="chain",
    )
    assert fracs == [0.7, 0.7, 0.9]  # unknown stage repeats the last fraction
