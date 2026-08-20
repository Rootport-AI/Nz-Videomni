"""LTX engine-generation ruling from a transformer GGUF's KV header (§2.2).

``services.engines.ltx.adapter.check_kv`` is the second half of the two-step
contract: ``services.model_registry.precheck_model_file`` reads
``general.architecture`` + ``model_version`` out of the header, and this
function decides whether THIS engine can run what it found. The KV values are
synthesized here (no weight files involved) — the parser itself has its own
tests in tests/test_gguf_kv.py, and the real 2.3/2.5 files were verified by
hand in P2.

Wiring this into POST /pipeline/load is the API-axis phase (§3-97 P6); what is
pinned here is the ruling and, above all, its wording — the LTX 2.5 message is
what an owner sees when they select a base model the engine cannot run yet.
"""

from __future__ import annotations

import logging

import pytest

from api.errors import APIError
from services.engines.ltx.adapter import SUPPORTED_MODEL_VERSIONS, check_kv

LTX23_KV = {"general.architecture": "ltxv", "model_version": "2.3.0"}
LTX25_KV = {"general.architecture": "ltxv", "model_version": "2.5.0"}


def test_supported_generation_passes():
    check_kv("transformer", "default", LTX23_KV)  # no raise
    assert SUPPORTED_MODEL_VERSIONS == frozenset({"2.3"})


def test_patch_segment_is_ignored():
    """The engine is chosen by the MINOR version: a 2.3.7 respin is still 2.3."""
    check_kv("transformer", "respin", {**LTX23_KV, "model_version": "2.3.7"})


def test_ltx25_is_refused_with_the_next_stage_message():
    with pytest.raises(APIError) as ei:
        check_kv("transformer", "ltx25", LTX25_KV)
    assert ei.value.code == "MODEL_INCOMPATIBLE" and ei.value.status_code == 422
    detail = ei.value.detail
    assert "ltxv 2.5.0" in detail
    assert "次段階" in detail and "§3-98" in detail
    assert "まだ読み込めません" in detail


def test_foreign_architecture_is_refused():
    with pytest.raises(APIError) as ei:
        check_kv(
            "transformer",
            "wan",
            {"general.architecture": "wan", "model_version": "2.2.0"},
        )
    assert ei.value.code == "MODEL_INCOMPATIBLE" and ei.value.status_code == 422
    assert "'wan'系のモデルです" in ei.value.detail
    assert "'ltxv'" in ei.value.detail


def test_missing_keys_warn_but_pass(caplog):
    """A hand-made or third-party GGUF may carry neither key. Refusing all of
    those would be a bigger regression than letting the engine's own loader
    have the last word (design §2.5), so this is a WARNING, not a 422."""
    with caplog.at_level(logging.WARNING, logger="ltx.runner"):
        check_kv("transformer", "homebrew", {})
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(messages) == 2
    assert any("general.architecture" in m for m in messages)
    assert any("model_version" in m for m in messages)
    assert all("homebrew" in m for m in messages)


def test_missing_version_alone_passes_a_known_architecture(caplog):
    with caplog.at_level(logging.WARNING, logger="ltx.runner"):
        check_kv("transformer", "no-version", {"general.architecture": "ltxv"})
    assert len(caplog.records) == 1


def test_missing_architecture_does_not_hide_an_unrunnable_version():
    """A 2.5 file that lost its architecture stamp is still a 2.5 file."""
    with pytest.raises(APIError):
        check_kv("transformer", "ltx25", {"model_version": "2.5.0"})


@pytest.mark.parametrize("category", ["text_encoder", "video_vae", "audio"])
def test_only_the_transformer_is_judged(category):
    """The transformer defines the generation. The GGUF Gemma carries a
    ``general.architecture`` of its own (``gemma3``), which is correct for what
    it is and must never be read as "the wrong engine"."""
    check_kv(category, "default", {"general.architecture": "gemma3"})
    check_kv(category, "default", LTX25_KV)
