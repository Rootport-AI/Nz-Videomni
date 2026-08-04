"""Worker-side resolution of the per-job ``fused_gguf_dequant_kernel`` flag
(§1-11).

Covers ``engine.worker._resolve_fused_dequant``: the missing-key default, the
explicit values, and the ``bool()`` coercion of a wrongly-typed value. Unlike
``_resolve_attention`` this resolver is deliberately NOT fail-loud — it is a
speed knob (output is bit-identical either way), so the same relaxed regime as
``_resolve_block_swap_prefetch`` applies and a junk value must never kill a job.

The echo companion ``_fused_gguf_dequant_kernel_used`` is NOT covered here: it
reads pipeline state (``dequant_triton``'s module globals via ``_PIPE``), so its
"off"/"on"/"on->off" truth table belongs to the dequant_triton selfcheck (C6/C7)
and to the real-device gates, not to a resolver unit test.

Run with ``.venv-engine`` and ``--noconftest``: importing ``engine.worker``
pulls in torch + ltx_core and initialises CUDA, neither of which exists in the
app venv (there the whole module skips), and the app conftest builds a FastAPI
app the engine venv does not have.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

from engine import worker  # noqa: E402


def test_missing_key_resolves_off(monkeypatch):
    # An additive payload must resolve BEFORE any pipeline state is consulted,
    # so a caller that predates the feature can never be affected by it.
    monkeypatch.setattr(worker, "_PIPE", None)
    assert worker._resolve_fused_dequant({}) is False


def test_explicit_false_resolves_off(monkeypatch):
    monkeypatch.setattr(worker, "_PIPE", None)
    assert worker._resolve_fused_dequant({"fused_gguf_dequant_kernel": False}) is False


def test_explicit_true_resolves_on(monkeypatch):
    monkeypatch.setattr(worker, "_PIPE", None)
    assert worker._resolve_fused_dequant({"fused_gguf_dequant_kernel": True}) is True


@pytest.mark.parametrize(
    "sent,expected",
    [
        (1, True),
        (0, False),
        ("true", True),  # any non-empty string is truthy -- coerced, not rejected
        ("", False),
        (None, False),
        ([], False),
    ],
)
def test_invalid_types_are_coerced_not_fatal(monkeypatch, sent, expected):
    # Speed knob, not a correctness precondition: bool() coerces whatever was
    # sent instead of failing the job loudly (same discipline as
    # _resolve_block_swap_prefetch).
    monkeypatch.setattr(worker, "_PIPE", None)
    got = worker._resolve_fused_dequant({"fused_gguf_dequant_kernel": sent})
    assert got is expected
    assert isinstance(got, bool)
