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
import re

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


# --------------------------- job-info console line (seed/base/loras/prompt)


def test_job_info_line_and_resolved_seed(client, caplog):
    """A single T2V job emits an additive info line carrying the base weight
    file, ``loras=none`` (no adapters), and the prompt; and the start line shows
    the RESOLVED seed (a concrete value) even though the request asked for -1."""
    payload = {
        "prompt": "A neon koi swimming through rain",
        "width": 384,
        "height": 256,
        "num_frames": 17,
        "frame_rate": 24.0,
        "num_inference_steps": 8,
        "seed": -1,  # random -> the console must show the value actually used
        "conditioning_images": [],
    }
    with caplog.at_level(logging.INFO, logger="ltx.pipeline"):
        r = client.post("/api/v1/generate", json=payload)
        assert r.status_code == 202

    msgs = [rec.getMessage() for rec in caplog.records if rec.name == "ltx.pipeline"]
    start = [m for m in msgs if "start mode=" in m]
    info = [m for m in msgs if "base=" in m and "prompt=" in m]
    assert start, msgs
    assert info, msgs
    # The random request resolved to a concrete non-negative seed (not -1).
    m = re.search(r"seed=(-?\d+)", start[0])
    assert m and int(m.group(1)) >= 0, start[0]
    assert "loras=none" in info[0], info[0]
    assert ".gguf" in info[0] or "base=" in info[0], info[0]
    assert "A neon koi swimming through rain" in info[0], info[0]


def test_prompt_for_log_escapes_newlines_and_truncates():
    """Raw user prompts are one-lined (newlines/tabs escaped -> no log injection)
    and capped in length so a huge prompt can't flood the console."""
    from services.pipeline_manager import _prompt_for_log

    out = _prompt_for_log("line1\nline2\tcol\r\nINFO forged")
    assert "\n" not in out and "\r" not in out and "\t" not in out
    assert "\\n" in out and "\\t" in out

    long = _prompt_for_log("x" * 500, limit=200)
    assert len(long) == 203 and long.endswith("...")  # 200 chars + ellipsis


def test_loras_for_log_names_and_effective_strength():
    """The info line lists adapter NAMES with requested strength, adding the
    effective (alpha-scaled) strength only when it differs; empty -> ``none``."""
    import types

    from services.pipeline_manager import _loras_for_log

    assert _loras_for_log([], []) == "none"

    spec = types.SimpleNamespace(name="Pixar_Toon", strength=0.45)
    # scale 1.0 (effective == requested) -> terse, no "effective="
    terse = _loras_for_log([spec], [("/p/a.safetensors", 0.45, "none")])
    assert terse == "Pixar_Toon(strength=0.45)"
    # alpha/rank convolution changed the strength -> both are shown
    scaled = _loras_for_log([spec], [("/p/a.safetensors", 0.30, "none")])
    assert "Pixar_Toon(strength=0.45, effective=0.3)" == scaled


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
        result = be._read_chain_events(
            # clip/clip_count arrive with chain stage-1 events (additive
            # ProgressCallback contract) — this test only tracks fractions.
            lambda step, total, frac, stage=None, **kw: seen.append(frac)
        )

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


# ----------------------- F1: uvicorn access-log job-polling filter (G3 gate)


def _access_record(args) -> logging.LogRecord:
    """Forge a uvicorn.access-shaped LogRecord (message %s slots + args tuple)."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=args,
        exc_info=None,
    )


@pytest.mark.parametrize(
    "args,expected",
    [
        # THE polling line: successful single-job status GET -> dropped.
        (("127.0.0.1:5000", "GET", "/api/v1/jobs/abc-123", "1.1", 200), False),
        # Trailing slash / query string are still the same resource -> dropped.
        (("127.0.0.1:5000", "GET", "/api/v1/jobs/abc-123/", "1.1", 200), False),
        (("127.0.0.1:5000", "GET", "/api/v1/jobs/abc-123?x=1", "1.1", 200), False),
        # Subresources (video/joined/...) are real downloads -> kept.
        (("127.0.0.1:5000", "GET", "/api/v1/jobs/abc-123/video", "1.1", 200), True),
        (("127.0.0.1:5000", "GET", "/api/v1/jobs/abc-123/joined", "1.1", 200), True),
        # The job LIST is not per-second polling -> kept.
        (("127.0.0.1:5000", "GET", "/api/v1/jobs", "1.1", 200), True),
        # Non-GET methods on the same path -> kept.
        (("127.0.0.1:5000", "POST", "/api/v1/jobs/abc-123", "1.1", 200), True),
        (("127.0.0.1:5000", "DELETE", "/api/v1/jobs/abc-123", "1.1", 200), True),
        # Non-200 statuses (a failing poll is signal) -> kept.
        (("127.0.0.1:5000", "GET", "/api/v1/jobs/abc-123", "1.1", 404), True),
        (("127.0.0.1:5000", "GET", "/api/v1/jobs/abc-123", "1.1", 500), True),
        # Unrelated endpoints -> kept.
        (("127.0.0.1:5000", "GET", "/api/v1/status", "1.1", 200), True),
    ],
)
def test_job_polling_access_filter(args, expected):
    from main import JobPollingAccessFilter

    assert JobPollingAccessFilter().filter(_access_record(args)) is expected


@pytest.mark.parametrize(
    "args",
    [
        None,  # no args at all
        ("only", "three", "slots"),  # wrong arity
        ("127.0.0.1:5000", "GET", 42, "1.1", 200),  # non-str path
        {"client": "127.0.0.1", "path": "/api/v1/jobs/x"},  # not a tuple
    ],
)
def test_job_polling_access_filter_fails_open(args):
    """Unexpected record.args shapes must pass through (fail-open)."""
    from main import JobPollingAccessFilter

    assert JobPollingAccessFilter().filter(_access_record(args)) is True


def test_build_uvicorn_log_config_attaches_filter_without_mutating_default():
    import uvicorn

    from main import (
        GradioApiInternalAccessFilter,
        JobPollingAccessFilter,
        build_uvicorn_log_config,
    )

    before = uvicorn.config.LOGGING_CONFIG.get("handlers", {}).get("access", {}).get("filters")
    cfg = build_uvicorn_log_config()
    # The returned config wires both access-log filters into the access handler...
    assert cfg["handlers"]["access"]["filters"] == [
        "job_polling_access", "gradio_api_internal_access"]
    assert cfg["filters"]["job_polling_access"]["()"] is JobPollingAccessFilter
    assert (cfg["filters"]["gradio_api_internal_access"]["()"]
            is GradioApiInternalAccessFilter)
    # ...and uvicorn's module-level default template is untouched.
    after = uvicorn.config.LOGGING_CONFIG.get("handlers", {}).get("access", {}).get("filters")
    assert before == after
    # The config must remain loadable by logging.config.dictConfig.
    import logging.config

    logging.config.dictConfig(cfg)
