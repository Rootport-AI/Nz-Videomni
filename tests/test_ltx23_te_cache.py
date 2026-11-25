"""LTX 2.3: the text encoder's conditioning kept in main memory between jobs
(keep_text_conditioning, 台帳 §1-299).

What is pinned here, all GPU-free (the text encoder is a counting fake; every
``torch.cuda`` call the paths make is a no-op):

  1. ``negative_from_encoded`` gives the same values as ``encode_negative``
     (NAG and VSF).
  2. Single generate, through the real ``generate`` -> ``_run_inference`` and a
     fake DistilledPipeline that calls the wheel's steps in the wheel's order:
     the second job builds no text encoder, hits return equal fresh copies,
     changing only the positive / only the negative encodes only that one, an
     all-hit job swaps ``ledger.text_encoder`` for the stand-in and restores it
     (also on an exception), the wrapper never calls itself, an OFF job empties
     the cache and touches nothing new, a failed job drops only what it put,
     editing a handed-out value changes no later hit, the ON flag never
     outlives a job (``_run_inference`` called directly stays OFF), the record.
  3. Chain (one clip changed, one string on two clips, with a negative),
     outpaint and inpaint, through their real encode paragraphs.
  4. Values made under ``torch.inference_mode()`` are hashed outside it and
     used by the next job.
  5. The worker passes the knob and adds ``ltx23`` only when it is true.

Run with ``.venv-engine`` and ``--noconftest`` (the app conftest imports
fastapi, which the engine venv does not have):

  .venv-engine\\Scripts\\python.exe -m pytest tests\\test_ltx23_te_cache.py ^
      -p no:cacheprovider --noconftest
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ltx_core")

import ltx_core.loader  # noqa: E402,F401  (import order the wheel needs)
import ltx_core.text_encoders.gemma as _gemma  # noqa: E402
import ltx_core.text_encoders.gemma.encoders.base_encoder as _base_encoder  # noqa: E402
import ltx_pipelines.distilled as _distilled  # noqa: E402

from engine.pipeline import chain_pipeline as _cp  # noqa: E402
from engine.pipeline import fast_video_pipeline as fvp  # noqa: E402
from engine.pipeline.fast_video_pipeline import LTXFastVideoPipeline  # noqa: E402
from engine.ref_latent_cache import RefLatentCache  # noqa: E402
from engine.te_conditioning import tensor_sha256  # noqa: E402
from engine.transformer.nag_service import (  # noqa: E402
    NagParams,
    NagState,
    encode_negative,
    negative_from_encoded,
)
from engine.transformer.vsf_service import VsfParams  # noqa: E402

SEQ = 16
_REAL_RUN_CHAIN = _cp.run_chain


# ── fakes ────────────────────────────────────────────────────────────────────
def _ctx_for(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The deterministic (video, audio) a string encodes to (fresh tensors)."""
    base = float(sum(ord(c) for c in text) % 97)
    video = (torch.arange(SEQ * 8, dtype=torch.float32).reshape(1, SEQ, 8) + base).to(torch.bfloat16)
    audio = (torch.arange(SEQ * 4, dtype=torch.float32).reshape(1, SEQ, 4) - base).to(torch.bfloat16)
    return video, audio


class _Tokenizer:
    def tokenize_with_weights(self, text: str):
        n = len(text.split())
        return {"gemma": [(1, 1)] * n + [(0, 0)] * (SEQ - n)}


class _FakeTE:
    """A built text encoder: has a tokenizer and a marker the fake encode reads."""

    marker = "real"

    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()


class _Env:
    """Event log plus the counting encode installed over every encode_text name."""

    def __init__(self) -> None:
        self.events: list = []
        self.encoded: list[str] = []
        self.builds = 0

    def encode_text(self, te, prompts):
        assert te.marker == "real"  # the stand-in raises on any read
        out = []
        for p in prompts:
            self.encoded.append(p)
            self.events.append(("encode", p))
            out.append(_ctx_for(p))
        return out

    def build(self):
        self.builds += 1
        self.events.append("build")
        return _FakeTE()


class _SpyLedger:
    """Logs every read and write of ``text_encoder`` (an instance attribute,
    as ``engine/gemma/gguf_quant_service.py`` sets it)."""

    def __init__(self, env: _Env) -> None:
        object.__setattr__(self, "_log", [])
        object.__setattr__(self, "text_encoder", env.build)

    def __getattribute__(self, name):
        if name == "text_encoder":
            object.__getattribute__(self, "_log").append("read")
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == "text_encoder":
            self._log.append("write")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name == "text_encoder":
            self._log.append("del")
        object.__delattr__(self, name)


class _StopAfterEncode(Exception):
    """Ends a chain / outpaint / inpaint driver right after its encode paragraph."""


class _FakeDistilled:
    """The wheel's DistilledPipeline.__call__, reduced to its text-encoder steps
    in the wheel's order (ltx_pipelines/distilled.py 96-104), then a check that
    NagState is ready (what NagService.install requires) and a record."""

    def __init__(self, pipe, env: _Env, ledger: _SpyLedger) -> None:
        self.pipe = pipe
        self.env = env
        self.model_ledger = ledger
        self.device = "cpu"
        self.pipeline_components = object()
        self.raise_after_encode = False
        self.mutate = False
        self.seen: dict = {}

    def __call__(self, prompt, seed, height, width, num_frames, frame_rate, images, tiling_config):
        d = sys.modules["ltx_pipelines.distilled"]
        self.seen["encode_text_is_fake"] = d.encode_text == self.env.encode_text
        text_encoder = self.model_ledger.text_encoder()
        self.seen["text_encoder"] = text_encoder
        context_p = d.encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p
        torch.cuda.synchronize()
        del text_encoder
        d.cleanup_memory()
        nag = self.pipe._nag
        if nag.requested:
            assert nag.ready
        self.seen["positive"] = (video_context.clone(), audio_context.clone())
        self.seen["negative"] = (
            None if not nag.requested else (nag.video_context.clone(), nag.audio_context.clone())
        )
        if self.mutate:
            video_context.add_(1)
            audio_context.add_(1)
            if nag.requested:
                nag.video_context.add_(1)
        if self.raise_after_encode:
            raise RuntimeError("boom after encode")
        return iter([]), None


def _new_pipe(monkeypatch):
    """A bare pipeline (``__new__``: no weights) with the cache state of
    ``__init__``, the job knobs it does not need stubbed, and the fakes."""
    env = _Env()
    for mod in (_gemma, _base_encoder, _distilled):
        monkeypatch.setattr(mod, "encode_text", env.encode_text)
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, _n=name, **k: env.events.append(_n))
    monkeypatch.setattr(fvp, "default_tiling_config", lambda **k: None)
    monkeypatch.setattr(fvp, "video_chunks_number", lambda *a, **k: 1)
    monkeypatch.setattr(fvp, "encode_video_output", lambda **k: None)

    pipe = LTXFastVideoPipeline.__new__(LTXFastVideoPipeline)
    pipe._nag = NagState()
    pipe._te_conditioning = RefLatentCache()
    pipe._keep_text_conditioning_job = False
    pipe._te_notes_positive = {}
    pipe._te_note_negative = None
    pipe._te_job_clips = None
    pipe._te_job_prompt = ""
    pipe._ltx23_record = None
    pipe._ic_loras_default = []
    pipe._ic_reference_default = None
    pipe._ic_attention_strength_default = 1.0
    pipe._ic_reference = None
    pipe._vae_spatial_tile_size = 0
    pipe._vae_temporal_tile_size = 0
    pipe._pre_reset_peak_allocated = pipe._pre_reset_peak_reserved = 0
    for name in (
        "_set_sage_job", "_set_block_swap_prefetch_job", "_set_fused_dequant_job",
        "_set_vae_mode_job", "_reset_block_swap_prefetch_job",
        "_release_block_swap_transformer", "_reset_fused_dequant_job",
    ):
        setattr(pipe, name, lambda *a, **k: None)
    pipe._sage = types.SimpleNamespace(reset=lambda: None)
    ledger = _SpyLedger(env)
    pipe.pipeline = _FakeDistilled(pipe, env, ledger)
    return pipe, env, ledger


def _generate(pipe, prompt="a cat", nag=None, keep=True):
    kwargs = dict(
        prompt=prompt, seed=1, height=256, width=384, num_frames=9, frame_rate=24.0,
        images=[], output_path="unused.mp4", ic_loras=[], nag=nag,
    )
    if keep:
        kwargs["keep_text_conditioning"] = True
    pipe.generate(**kwargs)


def _nag(neg="blurry"):
    return NagParams(negative_prompt=neg, scale=11.0, tau=2.5, alpha=0.25)


def _vsf(neg="blurry"):
    return VsfParams(negative_prompt=neg, scale=1.0)


def _sha(text):
    v, a = _ctx_for(text)
    return {"video": tensor_sha256(v), "audio": tensor_sha256(a)}


# ── 1. negative_from_encoded == encode_negative ──────────────────────────────
@pytest.mark.parametrize("vsf", [False, True])
def test_negative_from_encoded_matches_encode_negative(monkeypatch, vsf):
    env = _Env()
    monkeypatch.setattr(_base_encoder, "encode_text", env.encode_text)
    te = _FakeTE()
    text = "low quality blurry"

    want_v, want_a = encode_negative(te, text, slice_to_real_tokens=vsf)
    video, audio = env.encode_text(te, [text])[0]
    n_real = int(sum(int(w) for _t, w in te.tokenizer.tokenize_with_weights(text)["gemma"]))
    got_v, got_a = negative_from_encoded(video, audio, n_real if vsf else None)

    assert torch.equal(got_v, want_v) and torch.equal(got_a, want_a)
    assert got_v.shape == want_v.shape and got_a.dtype == want_a.dtype
    assert got_v.shape[1] == (3 if vsf else SEQ)


def test_negative_from_encoded_refuses_a_bad_token_count():
    v, a = _ctx_for("x")
    with pytest.raises(RuntimeError, match="0 < N <= seq_len"):
        negative_from_encoded(v, a, 0)
    with pytest.raises(RuntimeError, match="0 < N <= seq_len"):
        negative_from_encoded(v, a, SEQ + 1)


# ── 2. single generate ───────────────────────────────────────────────────────
def test_second_job_builds_no_text_encoder_and_hits_equal_values(monkeypatch):
    pipe, env, ledger = _new_pipe(monkeypatch)
    _generate(pipe, "a cat")
    assert env.builds == 1 and env.encoded == ["a cat"]
    first = pipe.pipeline.seen["positive"]

    _generate(pipe, "a cat")
    assert env.builds == 1, "the all-hit job must not build the text encoder"
    assert env.encoded == ["a cat"]
    got = pipe.pipeline.seen["positive"]
    want_v, want_a = _ctx_for("a cat")
    for g, w, f in zip(got, (want_v, want_a), first):
        assert torch.equal(g, w) and torch.equal(g, f)
        assert g.dtype == w.dtype and g.device == w.device
    assert pipe.ltx23_record() == {
        "te_cache": {"positive": "hit"},
        "conditioning_sha256": {"positive": _sha("a cat")},
    }


def test_a_hit_is_a_fresh_copy_of_the_held_value(monkeypatch):
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _generate(pipe, "a cat")
    held = pipe._te_conditioning.get(("a cat", ()))
    handed = pipe._encode_conditioning(["a cat"], env.build, release=False)[0]
    assert handed[0] is not held[0] and handed[1] is not held[1]
    assert handed[0].data_ptr() != held[0].data_ptr()
    assert torch.equal(handed[0], held[0]) and handed[0].dtype == held[0].dtype


def test_changing_only_the_positive_encodes_only_the_positive(monkeypatch):
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _generate(pipe, "a cat", nag=_nag("blurry"))
    assert env.builds == 1 and env.encoded == ["a cat", "blurry"]

    _generate(pipe, "a dog", nag=_nag("blurry"))
    assert env.builds == 2 and env.encoded[2:] == ["a dog"]
    assert pipe.ltx23_record()["te_cache"] == {"positive": "miss", "negative": "hit"}
    # The negative from the hit is what encode_negative would give (NAG: no slice).
    want = encode_negative(_FakeTE(), "blurry", slice_to_real_tokens=False)
    assert torch.equal(pipe.pipeline.seen["negative"][0], want[0])
    assert torch.equal(pipe.pipeline.seen["negative"][1], want[1])


def test_changing_only_the_negative_encodes_only_the_negative(monkeypatch):
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _generate(pipe, "a cat", nag=_nag("blurry"))
    _generate(pipe, "a cat", nag=_nag("noisy"))
    assert env.builds == 2 and env.encoded[2:] == ["noisy"]
    assert pipe.ltx23_record()["te_cache"] == {"positive": "hit", "negative": "miss"}


def test_vsf_slices_a_negative_held_by_a_nag_job(monkeypatch):
    """The same string's value serves NAG and VSF: the slice comes after."""
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _generate(pipe, "a cat", nag=_nag("low quality blurry"))
    _generate(pipe, "a cat", nag=_vsf("low quality blurry"))
    assert env.builds == 1
    want = encode_negative(_FakeTE(), "low quality blurry", slice_to_real_tokens=True)
    got = pipe.pipeline.seen["negative"]
    assert got[0].shape[1] == 3
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    held = pipe._te_conditioning.get(("low quality blurry", ()))
    assert held[0].shape[1] == SEQ, "the held value is never sliced"


def test_a_string_used_as_positive_and_negative_is_encoded_once(monkeypatch):
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _generate(pipe, "same", nag=_nag("same"))
    assert env.builds == 1 and env.encoded == ["same"]
    assert pipe.ltx23_record()["te_cache"] == {"positive": "miss", "negative": "hit"}


def test_all_hit_job_never_touches_the_stand_in_and_restores_the_builder(monkeypatch):
    pipe, env, ledger = _new_pipe(monkeypatch)
    builder = object.__getattribute__(ledger, "text_encoder")
    _generate(pipe, "a cat", nag=_nag("blurry"))
    _generate(pipe, "a cat", nag=_nag("blurry"))

    assert isinstance(pipe.pipeline.seen["text_encoder"], fvp._TextEncoderNotBuilt)
    assert env.builds == 1
    assert object.__getattribute__(ledger, "text_encoder") is builder
    # The stand-in fails on any use, so a wheel that used it would have failed.
    stand_in = fvp._TextEncoderNotBuilt()
    with pytest.raises(RuntimeError):
        stand_in.tokenizer
    with pytest.raises(RuntimeError):
        stand_in("x")


def test_the_builder_is_restored_when_the_job_raises(monkeypatch):
    pipe, env, ledger = _new_pipe(monkeypatch)
    builder = object.__getattribute__(ledger, "text_encoder")
    _generate(pipe, "a cat")
    pipe.pipeline.raise_after_encode = True
    with pytest.raises(RuntimeError, match="boom after encode"):
        _generate(pipe, "a cat")
    assert object.__getattribute__(ledger, "text_encoder") is builder
    assert _distilled.encode_text == env.encode_text, "encode_text restored too"


class _MethodLedger:
    """A ledger whose ``text_encoder`` is a class method (the wheel's own
    ``ModelLedger``, without the GGUF Gemma install's instance attribute)."""

    def __init__(self, env: _Env) -> None:
        self._env = env

    def text_encoder(self):
        return self._env.build()


def test_a_class_method_builder_is_restored_by_deleting_the_stand_in(monkeypatch):
    pipe, env, _ledger = _new_pipe(monkeypatch)
    ledger = _MethodLedger(env)
    pipe.pipeline.model_ledger = ledger
    _generate(pipe, "a cat")
    _generate(pipe, "a cat")
    assert isinstance(pipe.pipeline.seen["text_encoder"], fvp._TextEncoderNotBuilt)
    assert "text_encoder" not in vars(ledger), "the stand-in is deleted, not left behind"

    pipe.pipeline.raise_after_encode = True
    with pytest.raises(RuntimeError, match="boom after encode"):
        _generate(pipe, "a cat")
    assert "text_encoder" not in vars(ledger)

    pipe.pipeline.raise_after_encode = False
    _generate(pipe, "a dog")
    assert env.builds == 2, "the class method builds again on the next miss"


def test_an_on_job_drops_at_its_start_what_it_does_not_use(monkeypatch):
    """The start rule, seen through a failed job (whose end keeps what an
    earlier job left): a string this job does not use is already gone."""
    pipe, _env, _ledger = _new_pipe(monkeypatch)
    _generate(pipe, "a cat", nag=_nag("blurry"))
    pipe.pipeline.raise_after_encode = True
    with pytest.raises(RuntimeError, match="boom after encode"):
        _generate(pipe, "a dog", nag=_nag("blurry"))
    assert pipe._te_conditioning.get(("a cat", ())) is None, "dropped at the start"
    assert pipe._te_conditioning.get(("blurry", ())) is not None, "used by this job: kept"
    assert pipe._te_conditioning.get(("a dog", ())) is None, "put by the failed job: dropped"


def test_the_wrapper_never_calls_itself(monkeypatch):
    """During an ON job the wheel's encode_text name is the cache wrapper; the
    wrapper encodes with ltx_core's own function (else this would recurse)."""
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _generate(pipe, "a cat", nag=_nag("blurry"))
    assert pipe.pipeline.seen["encode_text_is_fake"] is False
    assert env.encoded == ["a cat", "blurry"]
    src = fvp.__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    body = text.split("def _encode_conditioning(", 1)[1].split("\n    def ", 1)[0]
    assert "from ltx_core.text_encoders.gemma import encode_text" in body
    assert "_distilled_mod" not in body


def test_off_job_empties_the_cache_and_is_todays_path(monkeypatch):
    pipe, env, ledger = _new_pipe(monkeypatch)
    _generate(pipe, "a cat", nag=_nag("blurry"))
    assert len(pipe._te_conditioning) == 2

    installed = []
    real_make = LTXFastVideoPipeline._make_nag_encode_text

    def spy(self, orig):
        installed.append(orig)
        return real_make(self, orig)

    monkeypatch.setattr(LTXFastVideoPipeline, "_make_nag_encode_text", spy)
    builder = object.__getattribute__(ledger, "text_encoder")
    ledger._log.clear()

    _generate(pipe, "a cat", nag=_nag("blurry"), keep=False)

    assert len(pipe._te_conditioning) == 0
    assert ledger._log == ["read"], "only the wheel's own call reads it; nobody writes it"
    assert object.__getattribute__(ledger, "text_encoder") is builder
    assert installed == [env.encode_text], "today's NAG wrapper over today's encode_text"
    assert env.builds == 2 and env.encoded[2:] == ["a cat", "blurry"]
    assert pipe.ltx23_record() is None
    assert pipe._keep_text_conditioning_job is False


def test_on_job_does_not_install_the_nag_wrapper(monkeypatch):
    pipe, _env, _ledger = _new_pipe(monkeypatch)
    calls = []
    monkeypatch.setattr(
        LTXFastVideoPipeline, "_make_nag_encode_text", lambda self, orig: calls.append(orig)
    )
    _generate(pipe, "a cat", nag=_nag("blurry"))
    assert calls == []


def test_a_failed_job_drops_only_what_it_put(monkeypatch):
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _generate(pipe, "a cat")
    pipe.pipeline.raise_after_encode = True
    with pytest.raises(RuntimeError):
        _generate(pipe, "a cat", nag=_nag("blurry"))
    assert pipe._te_conditioning.get(("a cat", ())) is not None, "the earlier job's entry stays"
    assert pipe._te_conditioning.get(("blurry", ())) is None, "the failed job's put is dropped"

    pipe.pipeline.raise_after_encode = False
    _generate(pipe, "a cat", nag=_nag("blurry"))
    assert pipe.ltx23_record()["te_cache"] == {"positive": "hit", "negative": "miss"}


def test_editing_a_handed_out_value_changes_no_later_hit(monkeypatch):
    pipe, _env, _ledger = _new_pipe(monkeypatch)
    pipe.pipeline.mutate = True
    _generate(pipe, "a cat", nag=_nag("blurry"))   # miss: the GPU original is edited
    _generate(pipe, "a cat", nag=_nag("blurry"))   # hit: the copy is edited
    pipe.pipeline.mutate = False
    _generate(pipe, "a cat", nag=_nag("blurry"))
    want_v, want_a = _ctx_for("a cat")
    assert torch.equal(pipe.pipeline.seen["positive"][0], want_v)
    assert torch.equal(pipe.pipeline.seen["positive"][1], want_a)
    assert torch.equal(pipe.pipeline.seen["negative"][0], _ctx_for("blurry")[0])


def test_the_on_flag_never_outlives_a_job(monkeypatch):
    pipe, env, ledger = _new_pipe(monkeypatch)
    _generate(pipe, "test warmup")
    assert pipe._keep_text_conditioning_job is False
    pipe.pipeline.raise_after_encode = True
    with pytest.raises(RuntimeError):
        _generate(pipe, "test warmup")
    assert pipe._keep_text_conditioning_job is False
    pipe.pipeline.raise_after_encode = False

    # warmup() calls _run_inference directly: it must take the OFF path.
    held = pipe._te_conditioning.get(("test warmup", ()))
    with torch.inference_mode():
        pipe._run_inference(
            prompt="test warmup", seed=42, height=256, width=384, num_frames=9,
            frame_rate=8, images=[], tiling_config=None,
        )
    assert env.builds == 2, "the OFF path builds the text encoder as today"
    assert pipe.pipeline.seen["encode_text_is_fake"] is True
    assert pipe._te_conditioning.get(("test warmup", ())) is held


def test_single_record_shape(monkeypatch):
    pipe, _env, _ledger = _new_pipe(monkeypatch)
    _generate(pipe, "a cat", nag=_nag("blurry"))
    assert pipe.ltx23_record() == {
        "te_cache": {"positive": "miss", "negative": "miss"},
        "conditioning_sha256": {"positive": _sha("a cat"), "negative": _sha("blurry")},
    }
    _generate(pipe, "a cat")
    rec = pipe.ltx23_record()
    assert rec == {
        "te_cache": {"positive": "hit"},
        "conditioning_sha256": {"positive": _sha("a cat")},
    }


# ── 4. inference_mode values ─────────────────────────────────────────────────
def test_values_made_in_inference_mode_hash_outside_and_serve_the_next_job(monkeypatch):
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _generate(pipe, "a cat")  # generate() runs under torch.inference_mode()
    held_v, held_a, _rt, _dev = pipe._te_conditioning.get(("a cat", ()))
    assert held_v.is_inference()
    assert not torch.is_inference_mode_enabled()
    assert {"video": tensor_sha256(held_v), "audio": tensor_sha256(held_a)} == _sha("a cat")
    _generate(pipe, "a cat")
    assert env.builds == 1
    assert torch.equal(pipe.pipeline.seen["positive"][0], _ctx_for("a cat")[0])


# ── 3. chain / outpaint / inpaint ────────────────────────────────────────────
def _drive_chain(monkeypatch, pipe, env, clip_prompts, stage2_prompt, nag=None, keep=True):
    import engine.pipeline.chain_pipeline as cp

    monkeypatch.setattr(cp, "default_tiling_config", lambda **k: object())

    def video_encoder():
        env.events.append("video_encoder")
        # What run_chain hands on: one context per clip, and the stage-2 one.
        caller = sys._getframe(1).f_locals
        env.chain_seen = (list(caller["seg_ctx"]), caller["stage2_ctx"])
        raise _StopAfterEncode()

    object.__setattr__(pipe.pipeline.model_ledger, "video_encoder", video_encoder)

    def run_chain_until_encoded(p, **kw):
        try:
            _REAL_RUN_CHAIN(p, **kw)
        except _StopAfterEncode:
            return {"n_clips": len(kw["clips"])}
        raise AssertionError("run_chain went past the stop")

    monkeypatch.setattr(cp, "run_chain", run_chain_until_encoded)
    clips = [cp.ChainClipSpec(prompt=p, num_frames=25, images=[]) for p in clip_prompts]
    kwargs = dict(
        clips=clips, width=384, height=256, frame_rate=24.0, num_steps=8, seed=1,
        overlap_frames=2, overlap_strength=0.5, output_path="x.mp4", ic_loras=[],
        nag=nag, stage2_prompt=stage2_prompt,
    )
    if keep:
        kwargs["keep_text_conditioning"] = True
    return pipe.generate_chain(**kwargs)


def test_chain_one_clip_changed_encodes_only_that_clip(monkeypatch, caplog):
    pipe, env, _ledger = _new_pipe(monkeypatch)
    with caplog.at_level(logging.INFO, logger="engine.pipeline.chain_pipeline"):
        meta = _drive_chain(monkeypatch, pipe, env, ["s a", "s b", "s c"], "s")
    assert meta == {"n_clips": 3}
    assert env.builds == 1 and env.encoded == ["s a", "s b", "s c", "s"]
    assert "4 distinct encoded; stage-2 prompt is one extra encode" in caplog.text
    # Freed in today's order: after the last encode, before the next build.
    last = env.events.index(("encode", "s"))
    assert "synchronize" in env.events[last:env.events.index("video_encoder")]

    env.events.clear()
    _drive_chain(monkeypatch, pipe, env, ["s a", "s b", "s c"], "s")
    assert env.builds == 1, "an all-hit chain builds no text encoder"
    assert "build" not in env.events
    rec = pipe.ltx23_record()
    assert rec["te_cache"] == {"clips": ["hit", "hit", "hit"], "stage2": "hit"}

    _drive_chain(monkeypatch, pipe, env, ["s a", "s B", "s c"], "s")
    assert env.builds == 2 and env.encoded[4:] == ["s B"]
    # Each clip (hit or miss) and stage 2 get their own string's values.
    seg_ctx, stage2_ctx = env.chain_seen
    for (v, a), p in zip(seg_ctx, ["s a", "s B", "s c"], strict=True):
        assert torch.equal(v, _ctx_for(p)[0]) and torch.equal(a, _ctx_for(p)[1])
    assert torch.equal(stage2_ctx[0], _ctx_for("s")[0])
    rec = pipe.ltx23_record()
    assert rec == {
        "te_cache": {"clips": ["hit", "miss", "hit"], "stage2": "hit"},
        "conditioning_sha256": {
            "clips": [_sha("s a"), _sha("s B"), _sha("s c")],
            "stage2": _sha("s"),
        },
    }


def test_chain_same_string_on_two_clips(monkeypatch):
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _drive_chain(monkeypatch, pipe, env, ["x", "x"], "s")
    assert env.encoded == ["x", "s"]
    rec = pipe.ltx23_record()
    assert rec["te_cache"] == {"clips": ["miss", "miss"], "stage2": "miss"}
    assert rec["conditioning_sha256"]["clips"] == [_sha("x"), _sha("x")]


def test_chain_with_a_negative(monkeypatch):
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _drive_chain(monkeypatch, pipe, env, ["a", "b"], "a", nag=_vsf("low quality blurry"))
    assert env.encoded == ["a", "b", "low quality blurry"]
    assert pipe._nag.requested is False, "reset at the job's end, as today"
    rec = pipe.ltx23_record()
    assert rec["te_cache"] == {"clips": ["miss", "miss"], "stage2": "miss", "negative": "miss"}
    assert rec["conditioning_sha256"]["negative"] == _sha("low quality blurry")

    seen = {}
    real_set = pipe._nag.set_contexts

    def set_contexts(v, a):
        seen["neg"] = (v, a)
        real_set(v, a)

    pipe._nag.set_contexts = set_contexts
    _drive_chain(monkeypatch, pipe, env, ["a", "b"], "a", nag=_vsf("low quality blurry"))
    assert env.builds == 1
    assert pipe.ltx23_record()["te_cache"]["negative"] == "hit"
    want = encode_negative(_FakeTE(), "low quality blurry", slice_to_real_tokens=True)
    assert torch.equal(seen["neg"][0], want[0]) and torch.equal(seen["neg"][1], want[1])


def test_chain_off_job_is_todays_paragraph(monkeypatch):
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _drive_chain(monkeypatch, pipe, env, ["a", "b"], "s")
    meta = _drive_chain(monkeypatch, pipe, env, ["a", "b"], "s", nag=_nag("n"), keep=False)
    assert len(pipe._te_conditioning) == 0
    assert env.builds == 2 and env.encoded[3:] == ["a", "b", "s", "n"]
    assert pipe.ltx23_record() is None and "ltx23" not in meta


def test_chain_failure_drops_only_what_it_put(monkeypatch):
    import engine.pipeline.chain_pipeline as cp

    pipe, env, _ledger = _new_pipe(monkeypatch)
    _drive_chain(monkeypatch, pipe, env, ["a"], "s")

    def video_encoder():
        raise RuntimeError("chain failed")

    # cp.run_chain is still _drive_chain's wrapper, which lets this error through.
    object.__setattr__(pipe.pipeline.model_ledger, "video_encoder", video_encoder)
    clips = [cp.ChainClipSpec(prompt=p, num_frames=25, images=[]) for p in ["a", "b"]]
    with pytest.raises(RuntimeError, match="chain failed"):
        pipe.generate_chain(
            clips=clips, width=384, height=256, frame_rate=24.0, num_steps=8, seed=1,
            overlap_frames=2, overlap_strength=0.5, output_path="x.mp4", ic_loras=[],
            stage2_prompt="s", keep_text_conditioning=True,
        )
    assert pipe._te_conditioning.get(("a", ())) is not None
    assert pipe._te_conditioning.get(("s", ())) is not None
    assert pipe._te_conditioning.get(("b", ())) is None
    assert pipe._keep_text_conditioning_job is False


def _drive_canvas(monkeypatch, pipe, kind, prompt, nag=None, keep=True):
    """Run outpaint / inpaint through their real encode paragraph and stop at
    the audio freeze right after it, capturing the positive it was handed."""
    if kind == "outpaint":
        import engine.pipeline.outpaint_pipeline as mod
        from engine.outpaint.canvas import OutpaintGeometry

        geometry = OutpaintGeometry(
            canvas_width=768, canvas_height=384,
            pad_left=128, pad_right=128, pad_top=64, pad_bottom=64,
        )
        extra = {}
        entry = pipe.generate_outpaint
    else:
        import engine.pipeline.inpaint_pipeline as mod
        from engine.inpaint.canvas import InpaintGeometry

        geometry = InpaintGeometry(
            canvas_width=384, canvas_height=256, source_width=320, source_height=256,
        )
        extra = {"mask_path": "mask.mp4"}
        entry = pipe.generate_inpaint
    seen = {}

    def freeze(**kw):
        caller = sys._getframe(1).f_locals
        seen["positive"] = (caller["video_ctx"].clone(), caller["audio_ctx"].clone())
        raise _StopAfterEncode()

    monkeypatch.setattr(mod, "_freeze_source_audio", freeze)
    monkeypatch.setattr(mod, "default_tiling_config", lambda **k: None)
    monkeypatch.setattr(mod, "_blend_chunk_size", lambda: None)
    kwargs = dict(
        prompt=prompt, canvas_path="canvas.mp4", source_path=None, geometry=geometry,
        num_frames=9, frame_rate=24.0, num_steps=8, seed=1, output_path="x.mp4",
        ic_loras=[], nag=nag, **extra,
    )
    if keep:
        kwargs["keep_text_conditioning"] = True
    with pytest.raises(_StopAfterEncode):
        entry(**kwargs)
    return seen


@pytest.mark.parametrize("kind", ["outpaint", "inpaint"])
def test_canvas_drivers_hit_on_the_second_job(monkeypatch, kind):
    """The drivers stop right after their encode, so each job here FAILS: the
    first one's puts are dropped. What is pinned is the encode paragraph
    itself: it goes through the cache, and a value held by an earlier
    successful job is used without building the text encoder."""
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _generate(pipe, "fill the band", nag=_nag("blurry"))  # a successful job holds both
    assert env.builds == 1

    seen = _drive_canvas(monkeypatch, pipe, kind, "fill the band", nag=_nag("blurry"))
    assert env.builds == 1, "every string held: no text encoder is built"
    want_v, want_a = _ctx_for("fill the band")
    assert torch.equal(seen["positive"][0], want_v) and torch.equal(seen["positive"][1], want_a)
    assert pipe._te_note_negative[0] == "hit"
    assert pipe._te_notes_positive["fill the band"][0] == "hit"

    seen = _drive_canvas(monkeypatch, pipe, kind, "another", nag=_nag("blurry"))
    assert env.builds == 2 and env.encoded[-1:] == ["another"]
    assert "synchronize" in env.events[env.events.index(("encode", "another")):]
    assert torch.equal(seen["positive"][0], _ctx_for("another")[0])
    assert pipe._te_conditioning.get(("another", ())) is None, "the failed job's put is dropped"
    assert pipe._te_conditioning.get(("blurry", ())) is not None


@pytest.mark.parametrize("kind", ["outpaint", "inpaint"])
def test_canvas_drivers_record_on_success(monkeypatch, kind):
    """A job that gets past its driver records the single-shape ``ltx23``."""
    pipe, env, _ledger = _new_pipe(monkeypatch)
    mod_name = "engine.pipeline.outpaint_pipeline" if kind == "outpaint" else "engine.pipeline.inpaint_pipeline"
    mod = __import__(mod_name, fromlist=["x"])
    run_name = "run_outpaint" if kind == "outpaint" else "run_inpaint"
    real_run = getattr(mod, run_name)

    def run_until_encoded(p, **kw):
        try:
            real_run(p, **kw)
        except _StopAfterEncode:
            return {kind: {}}
        raise AssertionError("went past the stop")

    monkeypatch.setattr(mod, run_name, run_until_encoded)

    def freeze(**kw):
        raise _StopAfterEncode()

    monkeypatch.setattr(mod, "_freeze_source_audio", freeze)
    monkeypatch.setattr(mod, "default_tiling_config", lambda **k: None)
    monkeypatch.setattr(mod, "_blend_chunk_size", lambda: None)
    if kind == "outpaint":
        from engine.outpaint.canvas import OutpaintGeometry

        geometry = OutpaintGeometry(
            canvas_width=768, canvas_height=384,
            pad_left=128, pad_right=128, pad_top=64, pad_bottom=64,
        )
        entry, extra = pipe.generate_outpaint, {}
    else:
        from engine.inpaint.canvas import InpaintGeometry

        geometry = InpaintGeometry(
            canvas_width=384, canvas_height=256, source_width=320, source_height=256,
        )
        entry, extra = pipe.generate_inpaint, {"mask_path": "mask.mp4"}
    for _ in range(2):
        meta = entry(
            prompt="fill", canvas_path="canvas.mp4", source_path=None, geometry=geometry,
            num_frames=9, frame_rate=24.0, num_steps=8, seed=1, output_path="x.mp4",
            ic_loras=[], nag=_nag("blurry"), keep_text_conditioning=True, **extra,
        )
    assert meta == {kind: {}}
    assert env.builds == 1
    assert pipe.ltx23_record() == {
        "te_cache": {"positive": "hit", "negative": "hit"},
        "conditioning_sha256": {"positive": _sha("fill"), "negative": _sha("blurry")},
    }


@pytest.mark.parametrize("kind", ["outpaint", "inpaint"])
def test_canvas_drivers_off_job_is_todays_paragraph(monkeypatch, kind):
    pipe, env, _ledger = _new_pipe(monkeypatch)
    _generate(pipe, "fill", nag=_nag("blurry"))
    _drive_canvas(monkeypatch, pipe, kind, "fill", nag=_nag("blurry"), keep=False)
    assert len(pipe._te_conditioning) == 0
    assert env.builds == 2 and env.encoded[2:] == ["fill", "blurry"]


# ── 5. worker ────────────────────────────────────────────────────────────────
class _RecordingPipe:
    _pre_reset_peak_allocated = 0
    _pre_reset_peak_reserved = 0
    _keep_resident_enabled = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def _touch(self, name, kwargs):
        self.calls.append((name, kwargs))
        with open(kwargs["output_path"], "wb") as fh:
            fh.write(b"\x00" * 32)

    def generate(self, **kwargs):
        self._touch("plain", kwargs)

    def generate_outpaint(self, **kwargs):
        self._touch("outpaint", kwargs)
        return {"outpaint": {"marker": 1}}

    def generate_chain(self, **kwargs):
        self._touch("chain", kwargs)
        return {"n_clips": len(kwargs["clips"])}

    def ltx23_record(self):
        return {"te_cache": {"positive": "hit"}, "conditioning_sha256": {}}


@pytest.fixture()
def worker_harness(monkeypatch, tmp_path):
    from engine import worker

    pipe = _RecordingPipe()
    events: list[dict] = []
    monkeypatch.setattr(worker, "_PIPE", pipe)
    monkeypatch.setattr(worker, "_log", lambda msg: None)
    monkeypatch.setattr(worker, "_log_job_start_vram", lambda: None)
    monkeypatch.setattr(worker, "_peak_vram_allocated_mb", lambda: 0)
    monkeypatch.setattr(worker, "_peak_vram_reserved_mb", lambda: 0)
    monkeypatch.setattr(worker, "_attention_used", lambda a, b: "sdpa")
    monkeypatch.setattr(worker, "_block_swap_prefetch_used", lambda: "off")
    monkeypatch.setattr(worker, "_keep_resident_used", lambda m, k: "off")
    monkeypatch.setattr(worker, "_fused_gguf_dequant_kernel_used", lambda: "off")
    monkeypatch.setattr(worker, "_vae_mode_used", lambda: "off")
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    return worker, pipe, events, tmp_path


def _gen_msg(tmp_path, **extra):
    msg = {
        "prompt": "a cat", "seed": 7, "width": 384, "height": 256, "num_frames": 9,
        "frame_rate": 24.0, "num_steps": 8, "output_path": str(tmp_path / "o.mp4"),
    }
    msg.update(extra)
    return msg


def _chain_msg(tmp_path, **extra):
    msg = {
        "prompt": "s", "seed": 7, "width": 384, "height": 256, "frame_rate": 24.0,
        "num_steps": 8, "overlap_frames": 2, "overlap_strength": 0.5,
        "output_path": str(tmp_path / "c.mp4"),
        "clips": [{"prompt": "s a", "num_frames": 25}, {"prompt": "s b", "num_frames": 25}],
    }
    msg.update(extra)
    return msg


@pytest.mark.parametrize("value", [None, False, "absent"])
def test_worker_off_job_is_todays_call_and_done(worker_harness, value):
    worker, pipe, events, tmp_path = worker_harness
    extra = {} if value == "absent" else {"keep_text_conditioning": value}
    worker._do_generate(_gen_msg(tmp_path, **extra))
    worker._do_generate_chain(_chain_msg(tmp_path, **extra))
    for _name, kwargs in pipe.calls:
        assert "keep_text_conditioning" not in kwargs
    done = [e for e in events if e["event"] == "done"]
    assert "ltx23" not in done[0]
    assert "ltx23" not in done[1]["chain"]


def test_worker_on_job_passes_the_knob_and_adds_ltx23(worker_harness):
    worker, pipe, events, tmp_path = worker_harness
    worker._do_generate(_gen_msg(tmp_path, keep_text_conditioning=True))
    worker._do_generate(_gen_msg(
        tmp_path, keep_text_conditioning=True,
        outpaint={
            "source_path": None, "canvas_width": 768, "canvas_height": 384,
            "pad_left": 128, "pad_right": 128, "pad_top": 64, "pad_bottom": 64,
        },
        reference_video={"path": str(tmp_path / "canvas.mp4"), "strength": 1.0},
    ))
    worker._do_generate_chain(_chain_msg(tmp_path, keep_text_conditioning=True))
    assert [n for n, _ in pipe.calls] == ["plain", "outpaint", "chain"]
    for _name, kwargs in pipe.calls:
        assert kwargs["keep_text_conditioning"] is True
    done = [e for e in events if e["event"] == "done"]
    rec = _RecordingPipe().ltx23_record()
    assert done[0]["ltx23"] == rec
    assert done[1]["ltx23"] == rec and done[1]["outpaint"] == {"marker": 1}
    assert "ltx23" not in done[2], "a chain carries it inside chain"
    assert done[2]["chain"] == {"n_clips": 2, "ltx23": rec}
