"""Worker-side resolution of the per-job ``vae_mode`` field (PrunaVAED, §3-50).

Covers ``engine.worker._resolve_vae_mode`` (the missing-key default, the two
legal values, and the fail-loud on anything else) and ``_vae_mode_used`` (the
echo, which reads the pipeline rather than the request).

Two deliberate asymmetries against the neighbouring resolvers:

  * ``_resolve_vae_mode`` is FAIL-LOUD like ``_resolve_attention``, not coercing
    like ``_resolve_block_swap_prefetch`` / ``_resolve_fused_dequant``. It is an
    enum, the API's ``Literal`` already 422s anything else, and a job that
    quietly ignored the request would still record a truthful-looking
    ``vae_mode_used``. Note this is not in tension with "never kill a job over a
    speed knob": the case that actually happens in the field — the weight file
    being absent — is NOT fatal and degrades to "on->off" inside the pipeline.
  * ``_vae_mode_used`` reads ``_PIPE`` (the ``_block_swap_prefetch_used``
    pattern) rather than recomputing from the request (the
    ``_keep_resident_used`` pattern), because the only downgrade this feature
    has is decided inside the pipeline's per-job file check.

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


def test_missing_key_resolves_to_default(monkeypatch):
    # An additive payload must resolve BEFORE any pipeline state is consulted,
    # so a caller that predates the feature can never be affected by it.
    monkeypatch.setattr(worker, "_PIPE", None)
    assert worker._resolve_vae_mode({}) == "default"


def test_explicit_default_resolves_to_default(monkeypatch):
    monkeypatch.setattr(worker, "_PIPE", None)
    assert worker._resolve_vae_mode({"vae_mode": "default"}) == "default"


def test_explicit_prune_vaed_resolves_to_prune_vaed(monkeypatch):
    monkeypatch.setattr(worker, "_PIPE", None)
    assert worker._resolve_vae_mode({"vae_mode": "prune_vaed"}) == "prune_vaed"


@pytest.mark.parametrize("sent", ["prunavaed", "PRUNE_VAED", "pruned", "", "taehv", 1, None])
def test_unknown_value_fails_loud(monkeypatch, sent):
    # Reaching the worker with a value outside the enum can only mean the app
    # and the engine disagree about the protocol. Falling back silently would
    # write a truthful-looking vae_mode_used for a job that ignored the request.
    monkeypatch.setattr(worker, "_PIPE", None)
    with pytest.raises(RuntimeError, match="unknown vae_mode"):
        worker._resolve_vae_mode({"vae_mode": sent})


@pytest.mark.parametrize("verdict", ["off", "on", "on->off"])
def test_used_echo_comes_from_the_pipeline(monkeypatch, verdict):
    monkeypatch.setattr(
        worker,
        "_PIPE",
        types.SimpleNamespace(vae_mode_used=lambda: verdict),
    )
    assert worker._vae_mode_used() == verdict


def test_used_echo_ignores_the_request(monkeypatch):
    # The request asked for the pruned decoder; the pipeline says it degraded.
    # The pipeline wins — that is the whole reason this echo is not computed
    # from the message the way _keep_resident_used is.
    monkeypatch.setattr(
        worker,
        "_PIPE",
        types.SimpleNamespace(vae_mode_used=lambda: "on->off"),
    )
    assert worker._vae_mode_used() == "on->off"
