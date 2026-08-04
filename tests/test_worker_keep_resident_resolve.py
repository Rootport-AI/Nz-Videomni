"""Worker-side resolution of the per-job ``keep_resident`` flag (§48).

Covers ``engine.worker._resolve_keep_resident`` and its ``done``-event
companion ``_keep_resident_used``: the missing-key default, the three guards
(G-A fail-loud / G-B + G-C auto-off), and the used-string each case reports.

``_PIPE`` is monkeypatched to a ``SimpleNamespace`` carrying just the two
attributes the guards read (``_gguf_per_layer_quant`` / ``_dit_cpu_load``) — the
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


def _pipe(*, per_layer_quant: bool = True, dit_cpu_load: bool = True):
    return types.SimpleNamespace(
        _gguf_per_layer_quant=per_layer_quant,
        _dit_cpu_load=dit_cpu_load,
    )


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


# ── G-A: bf16 fused-LoRA path -> fail loud ───────────────────────────────────


def test_g_a_per_layer_quant_off_raises(monkeypatch):
    # Correctness, not speed: the bf16 path's in-place weight.add_() would be
    # written into the persistent cache, so every later job silently inherits
    # this job's LoRA. Auto-off would hide a wrong-output risk behind a knob
    # nobody reads -> RuntimeError instead.
    monkeypatch.setattr(worker, "_PIPE", _pipe(per_layer_quant=False))
    with pytest.raises(RuntimeError, match="gguf_per_layer_quant"):
        worker._resolve_keep_resident({"keep_resident": True}, True)


def test_g_a_does_not_fire_when_keep_resident_is_off(monkeypatch):
    # The guard is on the ON path only -- a bf16-path job that never asks for
    # keep_resident must still run.
    monkeypatch.setattr(worker, "_PIPE", _pipe(per_layer_quant=False))
    assert worker._resolve_keep_resident({}, True) == (False, None)


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
