"""The LTX 2.5 chain's prompts: each clip's stage-1 string, and stage 2's.

§1-110 changed the chain from "a clip's own prompt REPLACES the shared one" to
"a clip's own text is APPENDED to the shared one". The composing happens app
side (``GenerateChainRequest.clip_prompt``); the engine's part is to encode
what it is handed -- each clip's composed string for stage 1, and the
payload's top-level shared prompt for stage 2 -- and to keep the two apart.
What is checkable here, without a GPU:

1. ``_encode_chain_prompts`` calls the prompt encoder ONCE, with every
   distinct string in order (clips first, then stage 2), and maps the results
   back by value. When a clip has no text of its own its string IS the shared
   prompt, so stage 2 shares that encode instead of adding one.
2. ``run_chain`` refines with ``stage2_ctx``, not with clip 0's context, and
   ``ChainSpec.prompt`` is a required field -- a payload that drops it fails
   loudly rather than quietly refining with a clip's string.

CPU-only and model-free: the encoder is a stub that records its argument.

Run with ``.venv-engine-ltx25`` and ``--noconftest`` (the app conftest builds a
FastAPI app that venv does not have).
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

from engine25 import chain25  # noqa: E402
from engine25.chain25 import ChainSpec, _encode_chain_prompts  # noqa: E402


class _RecordingEncoder:
    """Stands in for ``PromptEncoder``: one result per string, in order."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, prompts: list[str]) -> list[SimpleNamespace]:
        self.calls.append(list(prompts))
        return [SimpleNamespace(video_encoding=f"v:{p}", audio_encoding=f"a:{p}") for p in prompts]


# ---------------------------------------------------------------------------
# 1. _encode_chain_prompts
# ---------------------------------------------------------------------------


def test_every_clip_with_its_own_text_adds_the_shared_prompt_as_one_more_encode() -> None:
    encoder = _RecordingEncoder()

    seg_ctx, stage2_ctx = _encode_chain_prompts(encoder, ["s a", "s b"], "s")

    assert encoder.calls == [["s a", "s b", "s"]]
    assert seg_ctx == [("v:s a", "a:s a"), ("v:s b", "a:s b")]
    assert stage2_ctx == ("v:s", "a:s")


def test_a_clip_without_its_own_text_shares_its_encode_with_stage_2() -> None:
    encoder = _RecordingEncoder()

    seg_ctx, stage2_ctx = _encode_chain_prompts(encoder, ["s", "s b"], "s")

    assert encoder.calls == [["s", "s b"]]
    assert seg_ctx == [("v:s", "a:s"), ("v:s b", "a:s b")]
    assert stage2_ctx == ("v:s", "a:s")
    assert stage2_ctx == seg_ctx[0]


def test_no_clip_with_its_own_text_is_one_encode_for_everything() -> None:
    encoder = _RecordingEncoder()

    seg_ctx, stage2_ctx = _encode_chain_prompts(encoder, ["s", "s", "s"], "s")

    assert encoder.calls == [["s"]]
    assert seg_ctx == [("v:s", "a:s")] * 3
    assert stage2_ctx == ("v:s", "a:s")


def test_seg_ctx_is_a_list_in_clip_order() -> None:
    seg_ctx, _ = _encode_chain_prompts(_RecordingEncoder(), ["s b", "s a", "s b"], "s")

    assert isinstance(seg_ctx, list)
    assert seg_ctx == [("v:s b", "a:s b"), ("v:s a", "a:s a"), ("v:s b", "a:s b")]


def test_an_encoder_returning_the_wrong_count_is_an_error_not_a_misalignment() -> None:
    def short_encoder(prompts: list[str]) -> list[SimpleNamespace]:
        return [SimpleNamespace(video_encoding="v", audio_encoding="a")] * (len(prompts) - 1)

    with pytest.raises(ValueError):
        _encode_chain_prompts(short_encoder, ["s a", "s b"], "s")


def test_the_log_line_says_how_many_encodes_and_where_stage_2_comes_from(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=chain25.logger.name):
        _encode_chain_prompts(_RecordingEncoder(), ["s a", "s b"], "s")
        _encode_chain_prompts(_RecordingEncoder(), ["s", "s b"], "s")

    lines = [r.getMessage() for r in caplog.records if r.name == chain25.logger.name]
    assert lines == [
        "chain prompts: 3 distinct encoded for 2 clip(s); stage 2 (shared prompt) "
        "is one extra encode",
        "chain prompts: 2 distinct encoded for 2 clip(s); stage 2 (shared prompt) "
        "shares clip 0's encode",
    ]


# ---------------------------------------------------------------------------
# 2. the wiring: run_chain refines with stage2_ctx; the field is required
# ---------------------------------------------------------------------------


def test_stage_2_refines_with_the_shared_prompt_context_not_clip_0s() -> None:
    source = Path(chain25.__file__).read_text(encoding="utf-8")

    assert "stage2_vctx, stage2_actx = stage2_ctx" in source
    assert "= seg_ctx[0]" not in source


def test_chain_spec_prompt_is_required() -> None:
    field = next(f for f in dataclasses.fields(ChainSpec) if f.name == "prompt")

    assert field.default is dataclasses.MISSING
    assert field.default_factory is dataclasses.MISSING


def test_the_worker_reads_the_top_level_prompt_into_the_spec() -> None:
    source = (Path(chain25.__file__).parent / "worker.py").read_text(encoding="utf-8")

    assert 'prompt=str(msg["prompt"])' in source
