"""Chained prompts on LTX 2.3 — which strings the engine encodes, and for which stage.

The app composes each clip's stage-1 string (the shared prompt, then the clip's
own text appended) and sends the shared prompt on its own as the worker
message's top-level ``prompt``. The engine only encodes what it is given:

  1. ``_encode_chain_prompts``: every DISTINCT string once, in
     ``dict.fromkeys([*clip_prompts, stage2_prompt])`` order; ``stage2_ctx`` is
     the shared prompt's encoding, shared with a clip when the strings match.
  2. ``run_chain`` wiring: the text encoder sees the clips' strings and then the
     stage-2 prompt, after ``_set_nag_job`` and before any heavy build.
  3. Source guard: stage 2 reads ``stage2_ctx``, not clip 0's context.

VENV: torch-only, no fastapi -> this file runs in the ENGINE venv:

  .venv-engine\\Scripts\\python.exe -m pytest tests\\test_chain_prompt_encode.py ^
      -p no:cacheprovider --noconftest

(The app ``.venv`` has no torch, so ``importorskip`` skips the whole module
there; ``--noconftest`` is required because tests/conftest.py imports fastapi,
which the engine venv does not have. Same recipe as
tests/test_chain_reference_engine.py.)
"""

from __future__ import annotations

import logging
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")


# ── 1. _encode_chain_prompts ─────────────────────────────────────────────────


def _recording_encoder():
    calls: list[str] = []

    def encode_one(p: str):
        calls.append(p)
        return (f"v:{p}", f"a:{p}")

    return encode_one, calls


def test_every_clip_has_its_own_text_adds_one_stage2_encode():
    """Both clips carry their own text, so the shared prompt matches no clip
    string: three encodes in first-appearance order, stage 2 gets the last."""
    from engine.pipeline.chain_pipeline import _encode_chain_prompts

    encode_one, calls = _recording_encoder()
    seg_ctx, stage2_ctx = _encode_chain_prompts(encode_one, ["s a", "s b"], "s")

    assert calls == ["s a", "s b", "s"]
    assert seg_ctx == [("v:s a", "a:s a"), ("v:s b", "a:s b")]
    assert stage2_ctx == ("v:s", "a:s")


def test_clip0_without_own_text_shares_its_encoding_with_stage2():
    """Clip 0 has no text of its own, so its stage-1 string IS the shared
    prompt: two encodes, and stage 2 reuses clip 0's encoding."""
    from engine.pipeline.chain_pipeline import _encode_chain_prompts

    encode_one, calls = _recording_encoder()
    seg_ctx, stage2_ctx = _encode_chain_prompts(encode_one, ["s", "s b"], "s")

    assert calls == ["s", "s b"]
    assert seg_ctx == [("v:s", "a:s"), ("v:s b", "a:s b")]
    assert stage2_ctx is seg_ctx[0]


def test_seg_ctx_is_a_list_with_one_entry_per_clip():
    """Repeated clip strings are encoded once but still yield one entry per
    clip, and the result is a list (run_chain appends to it in
    ``internal_segment`` mode)."""
    from engine.pipeline.chain_pipeline import _encode_chain_prompts

    encode_one, calls = _recording_encoder()
    seg_ctx, _ = _encode_chain_prompts(encode_one, ["x", "x", "x"], "s")

    assert calls == ["x", "s"]
    assert isinstance(seg_ctx, list)
    assert len(seg_ctx) == 3


# ── 2. run_chain wiring ──────────────────────────────────────────────────────


class _StopBeforeBuild(Exception):
    """Raised by the fake video_encoder() to halt run_chain right after the
    text encode, before any heavy build."""


class _LenientShape:
    def __init__(self, *a, **k):
        pass


def _ltx_stubs(encode_text):
    return {
        "ltx_core": {},
        "ltx_core.components": {},
        "ltx_core.model": {},
        "ltx_core.text_encoders": {},
        "ltx_pipelines": {},
        "ltx_pipelines.utils": {},
        "ltx_core.components.diffusion_steps": {"EulerDiffusionStep": object},
        "ltx_core.components.noisers": {"GaussianNoiser": _LenientShape},
        "ltx_core.model.audio_vae": {"decode_audio": lambda *a, **k: None},
        "ltx_core.model.upsampler": {"upsample_video": lambda *a, **k: None},
        "ltx_core.model.video_vae": {"decode_video": lambda *a, **k: None},
        "ltx_core.text_encoders.gemma": {"encode_text": encode_text},
        "ltx_core.types": {
            "AudioLatentShape": _LenientShape,
            "VideoLatentShape": _LenientShape,
            "VideoPixelShape": _LenientShape,
        },
        "ltx_pipelines.utils.constants": {
            "DISTILLED_SIGMA_VALUES": [1.0],
            "STAGE_2_DISTILLED_SIGMA_VALUES": [1.0],
        },
        "ltx_pipelines.utils.helpers": {"cleanup_memory": lambda *a, **k: None},
    }


def _drive_encode(monkeypatch, clip_prompts, stage2_prompt):
    """Run ``run_chain`` up to the video_encoder build with ltx_core stubbed
    out; return the ordered event log (job-state calls, ledger builds and one
    ``("encode", prompt)`` per text encode)."""
    events: list = []

    def encode_text(te, prompts):
        events.extend(("encode", p) for p in prompts)
        return [(object(), object()) for _ in prompts]

    for name, attrs in _ltx_stubs(encode_text).items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)

    import engine.pipeline.chain_pipeline as cp

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(cp, "default_tiling_config", lambda **k: object())

    class FakeLedger:
        def text_encoder(self):
            events.append("text_encoder")
            return object()

        def video_encoder(self):
            events.append("video_encoder")
            raise _StopBeforeBuild()

        def transformer(self):
            events.append("transformer")
            raise _StopBeforeBuild()

    class FakeDP:
        device = "cpu"
        model_ledger = FakeLedger()
        pipeline_components = object()

    class FakeNag:
        requested = False

    class FakePipe:
        pipeline = FakeDP()
        _vae_spatial_tile_size = 0
        _vae_temporal_tile_size = 0
        _nag = FakeNag()

        def _set_nag_job(self, nag):
            events.append("set_nag_job")

        def _set_ic_job(self, loras, ref, attn):
            events.append("set_ic_job")

    clips = [cp.ChainClipSpec(prompt=p, num_frames=25, images=[]) for p in clip_prompts]
    with pytest.raises(_StopBeforeBuild):
        cp.run_chain(
            FakePipe(), clips=clips, stage2_prompt=stage2_prompt, width=384,
            height=256, frame_rate=24.0, num_steps=8, seed=1, overlap_frames=2,
            overlap_strength=0.5, output_path="x.mp4", ic_loras=[],
        )
    return events


def test_run_chain_encodes_clip_strings_then_the_stage2_prompt(monkeypatch, caplog):
    """Two clips with their own strings plus a distinct stage-2 prompt: the
    encoder sees ``a``, ``b``, ``shared`` once each, after ``_set_nag_job`` and
    before the video_encoder build, and the log line reports one extra encode."""
    with caplog.at_level(logging.INFO, logger="engine.pipeline.chain_pipeline"):
        events = _drive_encode(monkeypatch, ["a", "b"], "shared")

    encodes = [e[1] for e in events if isinstance(e, tuple) and e[0] == "encode"]
    assert encodes == ["a", "b", "shared"]
    first_encode = events.index(("encode", "a"))
    assert events.index("set_nag_job") < events.index("text_encoder") < first_encode
    assert events.index(("encode", "shared")) < events.index("video_encoder")
    assert "3 distinct encoded; stage-2 prompt is one extra encode" in caplog.text


def test_run_chain_stage2_prompt_equal_to_clip0_is_not_encoded_twice(monkeypatch, caplog):
    """Clip 0 has no text of its own (its string is the shared prompt): only
    two encodes, and the log line reports the stage-2 prompt as shared."""
    with caplog.at_level(logging.INFO, logger="engine.pipeline.chain_pipeline"):
        events = _drive_encode(monkeypatch, ["shared", "shared b"], "shared")

    encodes = [e[1] for e in events if isinstance(e, tuple) and e[0] == "encode"]
    assert encodes == ["shared", "shared b"]
    assert "2 distinct encoded; stage-2 prompt shared with a clip" in caplog.text


# ── 3. source guard ──────────────────────────────────────────────────────────


def test_stage2_reads_the_stage2_context_not_clip0s():
    """Stage 2 takes its one uniform context from ``stage2_ctx``; the old
    clip-0 read must not come back."""
    src = (
        pathlib.Path(__file__).resolve().parent.parent
        / "engine" / "pipeline" / "chain_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "= stage2_ctx" in src
    assert "= seg_ctx[0]" not in src
