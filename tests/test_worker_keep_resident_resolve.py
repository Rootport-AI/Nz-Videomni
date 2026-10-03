"""Worker-side resolution of the per-job ``keep_resident`` flag (§48).

Covers ``engine.worker._resolve_keep_resident`` and its ``done``-event
companion ``_keep_resident_used``: the missing-key default, the two guards
(G-B + G-C auto-off), and the used-string each case reports.

``_PIPE`` is monkeypatched to a ``SimpleNamespace`` carrying just the one
attribute the guards read (``_dit_cpu_load``) — the
resolver never builds or touches a real pipeline, which is exactly why these
guards can be pinned without a GPU or any weights.

Run with ``.venv-engine`` and ``--noconftest``: importing ``engine.worker``
pulls in torch + ltx_core and initialises CUDA, neither of which exists in the
app venv (there the whole module skips), and the app conftest builds a FastAPI
app the engine venv does not have.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

from engine import worker  # noqa: E402


def _pipe(*, dit_cpu_load: bool = True):
    return types.SimpleNamespace(_dit_cpu_load=dit_cpu_load)


@pytest.fixture
def healthy_pipe(monkeypatch):
    """The normal production configuration: every guard satisfied."""
    p = _pipe()
    monkeypatch.setattr(worker, "_PIPE", p)
    return p


# ── missing key / explicit off ───────────────────────────────────────────────


def test_missing_key_resolves_off_without_touching_the_pipeline(monkeypatch):
    # Deliberately leaves _PIPE as None: an additive payload must resolve
    # BEFORE any pipeline state is consulted, so a pre-keep_resident caller
    # can never be affected by a guard.
    monkeypatch.setattr(worker, "_PIPE", None)
    assert worker._resolve_keep_resident({}, True) == (False, None)
    assert worker._keep_resident_used({}, False) == "off"


def test_explicit_false_resolves_off(monkeypatch):
    monkeypatch.setattr(worker, "_PIPE", None)
    msg = {"keep_resident": False}
    assert worker._resolve_keep_resident(msg, True) == (False, None)
    # "off", not "on->off": nothing was downgraded, it was never asked for.
    assert worker._keep_resident_used(msg, False) == "off"


def test_on_with_every_guard_satisfied(healthy_pipe):
    msg = {"keep_resident": True}
    assert worker._resolve_keep_resident(msg, True) == (True, None)
    assert worker._keep_resident_used(msg, True) == "on"


# ── G-B / G-C: main-memory doubling -> warn + auto-off ───────────────────────


def test_g_b_dit_cpu_load_off_auto_offs(monkeypatch, capsys):
    monkeypatch.setattr(worker, "_PIPE", _pipe(dit_cpu_load=False))
    msg = {"keep_resident": True}
    effective, reason = worker._resolve_keep_resident(msg, True)
    assert effective is False
    assert reason == "dit_cpu_load=0"
    # The reason must be in the log: "on->off" alone cannot tell G-B from G-C
    # from a mid-job degrade.
    err = capsys.readouterr().err
    assert "dit_cpu_load=0" in err
    assert "WARNING" in err
    assert worker._keep_resident_used(msg, effective) == "on->off"


def test_g_c_prefetch_off_auto_offs(monkeypatch, capsys):
    healthy = _pipe()
    monkeypatch.setattr(worker, "_PIPE", healthy)
    msg = {"keep_resident": True}
    effective, reason = worker._resolve_keep_resident(msg, False)
    assert effective is False
    assert reason == "block_swap_prefetch=0"
    err = capsys.readouterr().err
    assert "block_swap_prefetch=0" in err
    assert "WARNING" in err
    assert worker._keep_resident_used(msg, effective) == "on->off"


def test_g_b_is_checked_before_g_c(monkeypatch):
    # Both broken -> the DiT guard reports first. Pinned only so the reason
    # string in the logs is deterministic when someone flips both checkboxes.
    monkeypatch.setattr(worker, "_PIPE", _pipe(dit_cpu_load=False))
    _, reason = worker._resolve_keep_resident({"keep_resident": True}, False)
    assert reason == "dit_cpu_load=0"


# ── §3-167: load refuses two transformer sources ─────────────────────────────


def test_load_with_both_transformer_sources_raises(monkeypatch):
    def _never(**_kw):
        raise AssertionError("the pipeline must not be built")

    monkeypatch.setattr(worker, "_PIPE", None)
    monkeypatch.setattr(worker, "probe_sage", lambda: False)
    monkeypatch.setattr(worker.LTXFastVideoPipeline, "create", staticmethod(_never))
    msg = {
        "gguf_transformer_path": "t.gguf",
        "safetensors_transformer_path": "t.safetensors",
    }
    with pytest.raises(RuntimeError, match="exactly one transformer source"):
        worker._do_load(msg)
    assert worker._PIPE is None


# ── a failed arm reports "on->off" (§1-55) ───────────────────────────────────


class _ArmFailsPipe:
    """Guards all satisfied, but the registry swap raises -- the way a ledger
    missing a builder attribute does. ``generate`` arms keep_resident exactly as
    the real entry points do, through the REAL ``_set_keep_resident_job``,
    which swallows the failure and leaves ``_keep_resident_enabled`` False."""

    _dit_cpu_load = True

    def __init__(self) -> None:
        self._keep_resident_enabled = False

    def _swap_registry(self, enabled):
        raise AssertionError("keep_resident: ModelLedger is missing builder attribute(s)")

    def generate(self, **kwargs):
        worker.LTXFastVideoPipeline._set_keep_resident_job(self, kwargs["keep_resident"])
        with open(kwargs["output_path"], "wb") as fh:
            fh.write(b"\x00" * 32)


def test_a_failed_arm_reports_on_to_off(monkeypatch, tmp_path):
    import torch

    pipe = _ArmFailsPipe()
    events: list[dict] = []
    monkeypatch.setattr(worker, "_PIPE", pipe)
    monkeypatch.setattr(worker, "_log", lambda msg: None)
    monkeypatch.setattr(worker, "_log_job_start_vram", lambda: None)
    monkeypatch.setattr(worker, "_peak_vram_allocated_mb", lambda: 0)
    monkeypatch.setattr(worker, "_peak_vram_reserved_mb", lambda: 0)
    monkeypatch.setattr(worker, "_attention_used", lambda a, b: "sdpa")
    monkeypatch.setattr(worker, "_block_swap_prefetch_used", lambda: "on")
    monkeypatch.setattr(worker, "_fused_gguf_dequant_kernel_used", lambda: "off")
    monkeypatch.setattr(worker, "_vae_mode_used", lambda: "off")
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)

    worker._do_generate({
        "prompt": "p", "seed": 1, "width": 64, "height": 64, "num_frames": 9,
        "frame_rate": 24.0, "num_steps": 8,
        "output_path": str(tmp_path / "out.mp4"),
        "keep_resident": True, "block_swap_prefetch": True,
    })

    assert pipe._keep_resident_enabled is False
    done = [e for e in events if e["event"] == "done"][0]
    assert done["keep_resident_used"] == "on->off"
