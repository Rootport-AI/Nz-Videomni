"""Console/log-overhaul regression tests (子C: console-log-overhaul).

These pin the observable logging contract the overhaul introduces:
  * S0 — httpx/httpcore polling noise is muted at the logger level.
  * S2 — job lifecycle lines carry mode/resolution/frames/seed + duration +
    peak VRAM, and the real-backend chain progress consumer emits rate-limited
    stage-progress INFO lines (unit-tested against a synthetic event stream, so
    no GPU/worker is needed).
"""

from __future__ import annotations

import logging

import pytest


# --------------------------------------------------------------- S0: httpx mute


def test_configure_logging_mutes_httpx(tmp_path):
    import main

    # Start from a noisy baseline so we can prove configure_logging lowers it.
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)

    main.configure_logging(tmp_path / "logs")

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


# ------------------------------------------ S2: job lifecycle (mock end-to-end)


def test_job_lifecycle_logs(client, caplog):
    """A single T2V job emits a start line (mode/res/frames/steps/seed) and a
    done line (duration) through the mock backend — no GPU."""
    payload = {
        "prompt": "A red ball rolling on a white floor",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "frame_rate": 24.0,
        "num_inference_steps": 8,
        "seed": 42,
        "conditioning_images": [],
    }
    with caplog.at_level(logging.INFO, logger="ltx.pipeline"):
        r = client.post("/api/v1/generate", json=payload)
        assert r.status_code == 202

    msgs = [rec.getMessage() for rec in caplog.records if rec.name == "ltx.pipeline"]
    start = [m for m in msgs if "start mode=" in m]
    done = [m for m in msgs if "done in" in m]
    assert start, msgs
    assert "steps=8" in start[0] and "seed=42" in start[0] and "frames=17" in start[0]
    assert done, msgs
    assert "done in" in done[0]


# --------------------------------- S2: chain stage-progress (real-backend unit)


def test_chain_stage_progress_logging(caplog):
    """`_RealBackend._read_chain_events` turns the coarse @@LTX@@ progress stream
    into rate-limited, stage-labelled INFO lines. Driven with a synthetic event
    stream so no worker/GPU is needed (the method only calls self._read_event)."""
    from services.ltx_runner import _RealBackend

    events = [
        {"event": "progress", "stage": "stage1", "index": 0, "total": 2},
        {"event": "progress", "stage": "stage1", "index": 1, "total": 2},
        {"event": "progress", "stage": "tile", "index": 0, "total": 3},
        {"event": "progress", "stage": "tile", "index": 2, "total": 3},
        {"event": "progress", "stage": "decode", "index": 0, "total": 1},
        {"event": "done", "seed_used": 7, "peak_vram_mb": 12345},
    ]
    it = iter(events)

    be = _RealBackend.__new__(_RealBackend)
    be._read_event = lambda timeout=None: next(it)  # type: ignore[attr-defined]

    seen = []
    with caplog.at_level(logging.INFO, logger="ltx.runner"):
        result = be._read_chain_events(lambda step, total, frac: seen.append(frac))

    assert result["event"] == "done"
    assert seen and seen[-1] == 0.95  # decode -> 0.95 fraction (unchanged path)

    msgs = [rec.getMessage() for rec in caplog.records if rec.name == "ltx.runner"]
    joined = "\n".join(msgs)
    assert "stage-1 denoise started (2 segments)" in joined, joined
    assert "stage-2 tiled upsample started (3 tiles)" in joined, joined
    assert "VAE decode started" in joined, joined
    # the final unit of a multi-unit stage always logs a x/total progress line
    assert any("stage-1 denoise 2/2 segments" in m for m in msgs), joined
    assert any("stage-2 tiled upsample 3/3 tiles" in m for m in msgs), joined
