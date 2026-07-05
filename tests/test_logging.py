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
