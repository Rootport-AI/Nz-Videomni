"""LTX 2.5: the text encoder's conditioning kept between jobs (keep_text_conditioning, 台帳 §1-299).

The unit is the BUNDLE: the whole list one official encode is handed (the
negative prompt last), because the same string encoded inside bundles of
different sizes does not give the same values (measured on the GPU,
2026-11-25). Pinned here, CPU only and with no weights:

1. ``Ltx25PromptEncoder._encode_bundle`` with the official encode
   (``PromptEncoder.__call__``) replaced by a recording stand-in: OFF is the
   official call with today's list and arguments; ON misses once, then hits
   without the official call; the held values are never reachable by the
   caller; ``enhance_first_prompt`` never goes through the cache; the VSF slice
   on a hit uses the kept tokenizer.
2. ``Ltx25Pipeline``'s start / end / report on the real methods.
3. The worker: the key, the calls' order, the record added only on a job that
   keeps conditioning (on a copy), the log line.
4. Every entry (single, Outpainting, Inpainting, Alpha Gen, chain): the bundle
   the worker starts the job with is the one the encoder looks up -- driven
   through the worker and the REAL drivers down to the encoder call.

Run with ``.venv-engine-ltx25``::

    .venv-engine-ltx25/Scripts/python.exe scripts/run_ltx25_pytest.py tests/test_ltx25_te_cache.py
"""

from __future__ import annotations

import inspect
import zlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ltx_core")

from ltx_core.text_encoders.gemma import EmbeddingsProcessorOutput  # noqa: E402

from engine.ref_latent_cache import RefLatentCache  # noqa: E402
from engine.te_conditioning import tensor_sha256  # noqa: E402
from engine25 import alphagen25, chain25, inpaint25, outpaint25, pipeline25, worker  # noqa: E402
from engine25.neg_prompt25 import NagParams, NagState, VsfParams  # noqa: E402
from engine25.outpaint25 import OutpaintResult  # noqa: E402
from engine25.pipeline25 import (  # noqa: E402
    GenerationResult,
    Ltx25Pipeline,
    Ltx25PromptEncoder,
    PromptEncoder,
    text_conditioning_bundle,
)

CPU = torch.device("cpu")
SEQ, D_VIDEO, D_AUDIO = 16, 8, 4
#: What ``DistilledPipeline.__call__`` hands the encoder on a plain job.
OFFICIAL_KWARGS = {"enhance_first_prompt": False, "enhance_static_cache": False, "enhance_prompt_image": None}
NAG = NagParams(negative_prompt="blurry, low quality", scale=11.0, tau=2.5, alpha=0.25)
VSF = VsfParams(negative_prompt="blurry, low quality", scale=1.5)


# --------------------------------------------------------------------------- #
# the stand-ins
# --------------------------------------------------------------------------- #

def _n_real(prompt: str) -> int:
    return max(1, min(SEQ, len(prompt.split())))


class _Tokenizer:
    """``tokenize_with_weights`` in the Gemma tokenizer's shape: one weight-1 pair
    per word, then weight-0 pads."""

    def tokenize_with_weights(self, prompt: str) -> dict:
        n = _n_real(prompt)
        return {"gemma": [(i + 1, 1.0) for i in range(n)] + [(0, 0.0)] * (SEQ - n)}


def _output(prompt: str, bundle_len: int, *, audio: bool = True) -> EmbeddingsProcessorOutput:
    """One string's result: a function of the string AND the bundle's size (as on
    the GPU), bf16, the mask real-tokens-first."""
    g = torch.Generator().manual_seed(zlib.crc32(f"{bundle_len}|{prompt}".encode()))
    video = torch.randn(1, SEQ, D_VIDEO, generator=g).to(torch.bfloat16)
    audio_t = torch.randn(1, SEQ, D_AUDIO, generator=g).to(torch.bfloat16) if audio else None
    mask = torch.zeros(1, SEQ, dtype=torch.int64)
    mask[:, : _n_real(prompt)] = 1
    return EmbeddingsProcessorOutput(video, audio_t, mask)


@pytest.fixture()
def official(monkeypatch):
    """The official encode and its two builds, replaced. ``calls`` = one
    ``(list object, its items, kwargs)`` per official call; ``builds`` counts
    the two builds (which, like the official, the encode makes itself)."""
    rec = SimpleNamespace(calls=[], builds={"te": 0, "ep": 0}, audio=True, returned=[])

    def fake_call(enc, prompts, **kwargs):
        rec.calls.append((prompts, list(prompts), dict(kwargs)))
        enc._build_text_encoder()
        enc._build_embeddings_processor()
        out = [_output(p, len(prompts), audio=rec.audio) for p in prompts]
        rec.returned.append(out)
        return out

    def fake_te(_enc):
        rec.builds["te"] += 1
        return SimpleNamespace(tokenizer=_Tokenizer())

    def fake_ep(_enc):
        rec.builds["ep"] += 1
        return object()

    monkeypatch.setattr(PromptEncoder, "__call__", fake_call)
    monkeypatch.setattr(PromptEncoder, "_build_text_encoder", fake_te)
    monkeypatch.setattr(PromptEncoder, "_build_embeddings_processor", fake_ep)
    return rec


class _Vram:
    def __init__(self) -> None:
        self.records: list[str] = []
        self.phases: dict = {}

    def reset(self) -> None:
        pass

    def record(self, name: str, _seconds: float) -> None:
        self.records.append(name)


def _encoder(monkeypatch, cls=Ltx25PromptEncoder, **providers):
    """A real ``Ltx25PromptEncoder.__init__`` over a stub official ``__init__``."""

    def fake_init(self, _paths, _dtype, device, **_kw):
        self._embeddings_processor_builder = None
        self._device = device

    monkeypatch.setattr(PromptEncoder, "__init__", fake_init)
    return cls(None, torch.bfloat16, CPU, text_encoder_builder=None, embeddings_processor_builder=None, **providers)


def _pipe(monkeypatch, encoder_cls=Ltx25PromptEncoder, cache=None):
    """An ``Ltx25Pipeline`` holding only the held-conditioning and NAG state, and an
    encoder wired the way ``__init__`` wires it (pinned by a test below)."""
    pipe = object.__new__(Ltx25Pipeline)
    pipe._nag = NagState()
    pipe._te_conditioning = cache if cache is not None else RefLatentCache()
    pipe._te_on = False
    pipe.prompt_encoder = _encoder(
        monkeypatch,
        encoder_cls,
        neg_state_provider=lambda: pipe._nag,
        te_cache_provider=lambda: pipe._te_conditioning if pipe._te_on else None,
    )
    return pipe


def _job(pipe, prompts, *, on, nag=None, ok=True, **kwargs):
    """One job's encode, the worker's order: arm NAG, start, encode, end, reset NAG.
    Returns ``(encoded, report, (negative video, negative audio))``."""
    pipe.set_nag_job(nag)
    pipe.set_text_conditioning_job(on, prompts)
    try:
        encoded = pipe.prompt_encoder(list(prompts), **(kwargs or OFFICIAL_KWARGS))
    finally:
        pipe.end_text_conditioning_job(ok)
    contexts = (pipe._nag.video_context, pipe._nag.audio_context)
    pipe.reset_nag_job()
    return encoded, pipe.text_conditioning_report(), contexts


def _same(a: EmbeddingsProcessorOutput, b: EmbeddingsProcessorOutput) -> None:
    assert type(a) is EmbeddingsProcessorOutput and type(b) is EmbeddingsProcessorOutput
    for x, y in zip(a, b, strict=True):
        if x is None or y is None:
            assert x is None and y is None
            continue
        assert x.dtype == y.dtype and x.device == y.device and x.shape == y.shape
        assert torch.equal(x, y)


def _clone(outputs):
    return [EmbeddingsProcessorOutput(*(None if t is None else t.clone() for t in o)) for o in outputs]


# --------------------------------------------------------------------------- #
# 1. the encoder
# --------------------------------------------------------------------------- #

def test_the_pipeline_wires_the_encoder_with_this_closure():
    source = inspect.getsource(Ltx25Pipeline.__init__)
    assert "self._te_conditioning = RefLatentCache()" in source
    assert "te_cache_provider=lambda: self._te_conditioning if self._te_on else None" in source


def test_off_is_the_official_encode_with_todays_list_and_arguments(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    prompts = ["a cat on a sofa"]
    pipe.set_nag_job(None)
    pipe.set_text_conditioning_job(False, prompts)
    result = pipe.prompt_encoder(prompts, **OFFICIAL_KWARGS)
    pipe.end_text_conditioning_job(True)

    assert len(official.calls) == 1
    handed, items, kwargs = official.calls[0]
    assert handed is prompts  # the caller's own list, as today
    assert items == ["a cat on a sofa"] and kwargs == OFFICIAL_KWARGS
    assert result is official.returned[0]
    assert len(pipe._te_conditioning) == 0
    assert pipe.text_conditioning_report() is None


def test_off_with_a_negative_prompt_is_todays_appended_list(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    _job(pipe, ["a", "b"], on=False, nag=NAG)
    assert official.calls[0][1] == ["a", "b", NAG.negative_prompt]
    assert official.calls[0][2] == OFFICIAL_KWARGS
    assert len(pipe._te_conditioning) == 0


def test_an_off_job_drops_everything_held(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    _job(pipe, ["a"], on=True)
    assert len(pipe._te_conditioning) == 1
    pipe.set_nag_job(None)
    pipe.set_text_conditioning_job(False, ["a"])
    assert len(pipe._te_conditioning) == 0


def test_on_miss_is_todays_call_and_puts_then_a_hit_skips_the_official_encode(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    _job(pipe, ["a dog"], on=False)
    off_call = official.calls[0]

    first, report1, _ = _job(pipe, ["a dog"], on=True)
    assert len(official.calls) == 2
    assert official.calls[1][1:] == off_call[1:]  # same list, same arguments as OFF
    assert first is official.returned[1]  # the official result itself
    assert report1["te_cache"] == "miss"
    assert len(pipe._te_conditioning) == 1

    builds = dict(official.builds)
    second, report2, _ = _job(pipe, ["a dog"], on=True)
    assert len(official.calls) == 2  # no official call
    assert official.builds == builds  # neither the text encoder nor the processor built
    assert report2["te_cache"] == "hit"
    assert isinstance(second, list) and second is not first
    assert len(second) == 1
    _same(second[0], official.returned[1][0])
    for held, got in zip(official.returned[1][0], second[0]):
        assert got.data_ptr() != held.data_ptr()  # a new copy


def test_a_hit_still_reports_progress_and_the_prompt_encode_phase(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    progress: list = []
    pipe.prompt_encoder.progress = lambda *a: progress.append(a)
    pipe.prompt_encoder.vram = _Vram()
    _job(pipe, ["x"], on=True)
    assert pipe.prompt_encoder.vram.records == ["10a_te_build", "10b_ep_build", "10_prompt_encode"]

    progress.clear()
    pipe.prompt_encoder.vram = _Vram()
    _job(pipe, ["x"], on=True)
    assert progress == [("encode", 0, 1), ("encode", 1, 1)]
    assert pipe.prompt_encoder.vram.records == ["10_prompt_encode"]


def test_the_nag_bundle_ends_with_the_negative_prompt_and_a_hit_gives_the_same_contexts(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    pos, report, (v1, a1) = _job(pipe, ["a", "b"], on=True, nag=NAG)
    assert official.calls[0][1] == ["a", "b", NAG.negative_prompt]
    assert report["te_cache"] == "miss" and len(pos) == 2
    key = next(iter(pipe._te_conditioning._values))
    assert key == (("a", "b", NAG.negative_prompt), ())
    assert key[0] == text_conditioning_bundle(["a", "b"], NAG.negative_prompt)

    pos2, report2, (v2, a2) = _job(pipe, ["a", "b"], on=True, nag=NAG)
    assert report2["te_cache"] == "hit" and len(official.calls) == 1
    for x, y in zip(pos, pos2):
        _same(x, y)
    assert torch.equal(v1, v2) and torch.equal(a1, a2) and v1.dtype == v2.dtype


@pytest.mark.parametrize(
    "second",
    [
        (["a", "b"], VsfParams(negative_prompt="other", scale=1.5)),  # the negative only
        (["a", "c"], NAG),  # a positive only
        (["b", "a"], NAG),  # the order only
        (["a", "b"], None),  # without the negative
    ],
)
def test_any_change_in_the_bundle_misses(monkeypatch, official, second):
    pipe = _pipe(monkeypatch)
    _job(pipe, ["a", "b"], on=True, nag=NAG)
    prompts, nag = second
    _, report, _ = _job(pipe, prompts, on=True, nag=nag)
    assert report["te_cache"] == "miss"
    assert len(official.calls) == 2


def test_the_nag_and_vsf_share_one_bundle(monkeypatch, official):
    """The method is not part of the key: the same strings are the same bundle."""
    pipe = _pipe(monkeypatch)
    _job(pipe, ["a"], on=True, nag=NAG)
    _, report, _ = _job(pipe, ["a"], on=True, nag=VsfParams(negative_prompt=NAG.negative_prompt, scale=1.5))
    assert report["te_cache"] == "hit" and len(official.calls) == 1


def test_what_the_caller_does_with_the_result_never_reaches_the_held_values(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    first, _, _ = _job(pipe, ["a", "b"], on=True, nag=NAG)
    reference = _clone(official.returned[0])
    # The official result (returned as it is on a miss), written over and cut.
    first[0].video_encoding.add_(1)
    first[1].attention_mask.zero_()
    first.pop()
    official.returned[0][-1].video_encoding.mul_(0)  # the negative row the encoder popped

    second, _, _ = _job(pipe, ["a", "b"], on=True, nag=NAG)
    for x, y in zip(second, reference[:2]):
        _same(x, y)
    second[0].video_encoding.add_(5)
    second.clear()

    third, _, (v, _a) = _job(pipe, ["a", "b"], on=True, nag=NAG)
    for x, y in zip(third, reference[:2]):
        _same(x, y)
    expected = reference[2].video_encoding
    assert torch.equal(v, expected.view(expected.shape[0], -1, expected.shape[-1]))


def test_enhance_first_prompt_never_goes_through_the_cache(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    kwargs = {"enhance_first_prompt": True, "enhance_static_cache": False, "enhance_prompt_image": None}
    for _ in range(2):
        _, report, _ = _job(pipe, ["a"], on=True, **kwargs)
        assert report is None
    assert len(official.calls) == 2
    assert all(call[2] == kwargs for call in official.calls)
    assert len(pipe._te_conditioning) == 0


def test_vsf_on_a_hit_slices_with_the_kept_tokenizer_as_today(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    _, _, (v_off, a_off) = _job(pipe, ["a"], on=False, nag=VSF)
    n = _n_real(VSF.negative_prompt)
    assert v_off.shape[1] == n and a_off.shape[1] == n

    _job(pipe, ["a"], on=True, nag=VSF)
    tokenizer = pipe.prompt_encoder._tokenizer
    assert tokenizer is not None  # an entry is only put after a build stashed it
    builds = dict(official.builds)
    _, report, (v_hit, a_hit) = _job(pipe, ["a"], on=True, nag=VSF)
    assert report["te_cache"] == "hit" and official.builds == builds
    assert pipe.prompt_encoder._tokenizer is tokenizer
    assert torch.equal(v_hit, v_off) and torch.equal(a_hit, a_off)


def test_the_record_is_the_bundle_in_order_with_the_negative_last(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    _, miss, _ = _job(pipe, ["a", "b"], on=True, nag=NAG)
    held = official.returned[0]
    assert miss == {
        "te_cache": "miss",
        "conditioning_sha256": [
            {
                "video": tensor_sha256(o.video_encoding),
                "audio": tensor_sha256(o.audio_encoding),
                "mask": tensor_sha256(o.attention_mask),
            }
            for o in held
        ],
    }
    assert len(miss["conditioning_sha256"]) == 3
    _, hit, _ = _job(pipe, ["a", "b"], on=True, nag=NAG)
    assert hit == {**miss, "te_cache": "hit"}


def test_an_absent_audio_encoding_is_held_and_recorded_as_none(monkeypatch, official):
    official.audio = False
    pipe = _pipe(monkeypatch)
    _, miss, _ = _job(pipe, ["a"], on=True)
    assert miss["conditioning_sha256"][0]["audio"] is None
    got, hit, _ = _job(pipe, ["a"], on=True)
    assert hit["te_cache"] == "hit" and got[0].audio_encoding is None
    _same(got[0], official.returned[0][0])


# --------------------------------------------------------------------------- #
# 2. start and end
# --------------------------------------------------------------------------- #

def test_an_on_start_drops_every_bundle_but_this_jobs(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    _job(pipe, ["a"], on=True)
    pipe._te_conditioning.put((("b",), ()), ("held", CPU))  # a second entry, as if kept
    pipe.set_nag_job(None)
    pipe.set_text_conditioning_job(True, ["a"])
    assert list(pipe._te_conditioning._values) == [(("a",), ())]
    pipe.end_text_conditioning_job(True)


def test_the_start_bundle_takes_the_armed_negative_prompt(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    _job(pipe, ["a"], on=True, nag=NAG)
    pipe.set_nag_job(None)  # the same strings, no negative: another bundle
    pipe.set_text_conditioning_job(True, ["a"])
    assert len(pipe._te_conditioning) == 0
    pipe.end_text_conditioning_job(True)


def test_a_failed_job_drops_only_what_it_put(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    _job(pipe, ["a"], on=True, ok=False)
    assert len(pipe._te_conditioning) == 0  # put by the failed job

    _job(pipe, ["a"], on=True)
    _, report, _ = _job(pipe, ["a"], on=True, ok=False)  # a hit, then a failure
    assert report["te_cache"] == "hit"
    assert len(pipe._te_conditioning) == 1  # an earlier job's: kept


def test_the_end_always_clears_the_on_mark(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    pipe.set_nag_job(None)
    pipe.set_text_conditioning_job(True, ["a"])
    assert pipe._te_on is True
    pipe.end_text_conditioning_job(False)
    assert pipe._te_on is False
    # After the end, an encode is the official one and touches nothing held.
    pipe.prompt_encoder(["a"], **OFFICIAL_KWARGS)
    assert len(pipe._te_conditioning) == 0


def test_the_end_clears_the_mark_even_when_the_cache_raises(monkeypatch, official):
    class _Raising(RefLatentCache):
        def end(self, ok):
            raise RuntimeError("boom")

    pipe = _pipe(monkeypatch, cache=_Raising())
    pipe.set_nag_job(None)
    pipe.set_text_conditioning_job(True, ["a"])
    pipe.end_text_conditioning_job(True)  # never raises
    assert pipe._te_on is False


def test_an_off_start_clears_the_last_report(monkeypatch, official):
    pipe = _pipe(monkeypatch)
    _, report, _ = _job(pipe, ["a"], on=True)
    assert report is not None
    _, report, _ = _job(pipe, ["a"], on=False)
    assert report is None


# --------------------------------------------------------------------------- #
# 3. the worker
# --------------------------------------------------------------------------- #

REPORT = {"te_cache": "hit", "conditioning_sha256": [{"video": "v", "audio": None, "mask": "m"}]}


def _result(cls=GenerationResult, **extra):
    return cls(
        output_path="out.mp4", seed=7, width=384, height=256, num_frames=9, frame_rate=24.0,
        encode_fps=24, num_images=0, size_bytes=1024, seconds=1.0,
        phases={"21_stage1_denoise": {"seconds": 1.0}},
        peak_allocated_gib=1.0, peak_reserved_gib=1.0, rss_peak_gib=1.0,
        video_chunks=1, tiling="t", **extra,
    )


class _RecordingPipe:
    """What ``_do_generate`` / ``_do_generate_chain`` read off ``_PIPE``, recording
    the job-state calls in order."""

    sampler = "ancestral"
    video_vae_kind = "conv"

    def __init__(self) -> None:
        self.order: list = []
        self.report_reads = 0

    def set_acceleration_job(self, **_kw):
        self.order.append("set_acceleration_job")

    def set_nag_job(self, nag):
        self.order.append(("set_nag_job", nag))

    def set_text_conditioning_job(self, on, prompts):
        self.order.append(("set_text_conditioning_job", on, list(prompts)))

    def end_text_conditioning_job(self, ok):
        self.order.append(("end_text_conditioning_job", ok))

    def text_conditioning_report(self):
        self.report_reads += 1
        return REPORT

    def reset_nag_job(self):
        self.order.append("reset_nag_job")

    def reset_acceleration_job(self):
        self.order.append("reset_acceleration_job")

    def block_swap_prefetch_used(self):
        return "off"

    def fused_gguf_dequant_kernel_used(self):
        return "off"

    def keep_resident_used(self):
        return "off"

    def keep_resident_embeddings_used(self):
        return "off"

    def generate(self, **_kw):
        self.order.append("generate")
        return _result()


@pytest.fixture()
def harness(monkeypatch, tmp_path):
    pipe = _RecordingPipe()
    events: list[dict] = []
    logs: list[str] = []
    monkeypatch.setattr(worker, "_PIPE", pipe)
    monkeypatch.setattr(worker, "_log", logs.append)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    (tmp_path / "ref.mp4").write_bytes(b"\x00" * 32)
    return pipe, events, logs, tmp_path


def _msg(tmp_path, **extra) -> dict:
    msg = {
        "prompt": "a red car", "seed": 7, "width": 384, "height": 256, "num_frames": 9,
        "frame_rate": 24.0, "output_path": str(tmp_path / "output.mp4"), "loras": [],
    }
    msg.update(extra)
    return msg


def test_the_worker_key_is_off_unless_true():
    assert worker._resolve_keep_text_conditioning({}) is False
    assert worker._resolve_keep_text_conditioning({"keep_text_conditioning": None}) is False
    assert worker._resolve_keep_text_conditioning({"keep_text_conditioning": False}) is False
    assert worker._resolve_keep_text_conditioning({"keep_text_conditioning": True}) is True


def test_an_off_job_starts_off_and_its_ltx25_is_todays(harness):
    pipe, events, logs, tmp_path = harness
    worker._do_generate(_msg(tmp_path))
    assert ("set_text_conditioning_job", False, ["a red car"]) in pipe.order
    assert pipe.report_reads == 0
    ltx25 = events[-1]["ltx25"]
    assert set(ltx25) == {"encode_fps", "video_chunks", "tiling", "size_bytes", "phases"}
    assert "keep_text_conditioning=False" in logs[0]


def test_an_on_job_gains_the_record_and_the_calls_keep_their_order(harness):
    pipe, events, logs, tmp_path = harness
    worker._do_generate(_msg(tmp_path, keep_text_conditioning=True))
    assert pipe.order == [
        "set_acceleration_job",
        ("set_nag_job", None),
        ("set_text_conditioning_job", True, ["a red car"]),
        "generate",
        ("end_text_conditioning_job", True),
        "reset_nag_job",
        "reset_acceleration_job",
    ]
    ltx25 = events[-1]["ltx25"]
    assert ltx25["te_cache"] == "hit"
    assert ltx25["conditioning_sha256"] == REPORT["conditioning_sha256"]
    assert {"encode_fps", "phases"} <= set(ltx25)
    assert "keep_text_conditioning=True" in logs[0]


def test_a_failed_job_ends_with_ok_false(harness, monkeypatch):
    pipe, _events, _logs, tmp_path = harness

    def boom(**_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipe, "generate", boom)
    with pytest.raises(RuntimeError):
        worker._do_generate(_msg(tmp_path, keep_text_conditioning=True))
    assert pipe.order[-3:] == [("end_text_conditioning_job", False), "reset_nag_job", "reset_acceleration_job"]


def test_an_outpaint_job_adds_the_record_on_a_copy(harness, monkeypatch):
    pipe, events, _logs, tmp_path = harness
    shared = {"stage1_sampler": "ancestral", "phases": {}}
    metadata = {"outpaint": {"m": 1}, "ltx25": shared}

    def fake_run_outpaint(_pipeline, **_kw):
        return _result(OutpaintResult, metadata=metadata)

    monkeypatch.setattr(outpaint25, "run_outpaint", fake_run_outpaint)
    outpaint = {"source_path": "src.mp4", "canvas_width": 768, "canvas_height": 384,
                "pad_left": 128, "pad_right": 128, "pad_top": 64, "pad_bottom": 64}
    worker._do_generate(_msg(
        tmp_path, keep_text_conditioning=True, outpaint=outpaint,
        reference_video={"path": str(tmp_path / "ref.mp4"), "strength": 1.0},
    ))
    done = events[-1]
    assert done["ltx25"]["te_cache"] == "hit"
    assert "te_cache" not in shared and done["outpaint"]["ltx25"] is shared


def test_the_chain_starts_with_the_distinct_strings_and_adds_the_record_on_a_copy(harness, monkeypatch):
    pipe, events, logs, tmp_path = harness
    metadata = {"n_clips": 3, "ltx25": {"phases": {}, "vram": {}}}

    def fake_run_chain(_pipeline, spec, _progress, **_kw):
        pipe.order.append("run_chain")
        return SimpleNamespace(metadata=metadata, output_path=spec.output_path)

    monkeypatch.setattr(chain25, "run_chain", fake_run_chain)
    msg = _chain_msg(tmp_path, ["s", "s b", "s b"], keep_text_conditioning=True, nag=_nag_block())
    worker._do_generate_chain(msg)
    assert pipe.order[1][0] == "set_nag_job" and pipe.order[1][1] is not None
    assert pipe.order[2] == ("set_text_conditioning_job", True, ["s", "s b"])
    assert pipe.order[3:5] == ["run_chain", ("end_text_conditioning_job", True)]
    chain = events[-1]["chain"]
    assert chain["ltx25"]["te_cache"] == "hit"
    assert "te_cache" not in metadata["ltx25"] and chain is not metadata
    assert any("keep_text_conditioning=True" in s for s in logs)


def test_an_off_chain_is_todays(harness, monkeypatch):
    pipe, events, logs, tmp_path = harness
    metadata = {"n_clips": 1, "ltx25": {"phases": {}}}
    monkeypatch.setattr(
        chain25, "run_chain",
        lambda _p, spec, _pr, **_kw: SimpleNamespace(metadata=metadata, output_path=spec.output_path),
    )
    worker._do_generate_chain(_chain_msg(tmp_path, ["s"]))
    assert ("set_text_conditioning_job", False, ["s"]) in pipe.order
    assert pipe.report_reads == 0
    assert events[-1]["chain"] is metadata
    assert any("keep_text_conditioning=False" in s for s in logs)


def test_distinct_chain_prompts_is_the_list_the_chain_encodes():
    assert chain25.distinct_chain_prompts(["s a", "s b", "s a"], "s") == ["s a", "s b", "s"]
    assert chain25.distinct_chain_prompts(["s", "s b"], "s") == ["s", "s b"]
    source = inspect.getsource(chain25._encode_chain_prompts)
    assert "distinct = distinct_chain_prompts(clip_prompts, stage2_prompt)" in source


# --------------------------------------------------------------------------- #
# 4. every entry: the bundle started is the bundle looked up
# --------------------------------------------------------------------------- #

class _Stop(Exception):
    """Raised right after the encode: how far the entry tests drive a job."""


class _StoppingEncoder(Ltx25PromptEncoder):
    def __call__(self, prompts, **kwargs):
        super().__call__(prompts, **kwargs)
        raise _Stop


class _SpyCache(RefLatentCache):
    def __init__(self) -> None:
        super().__init__()
        self.begun: list[set] = []
        self.looked_up: list = []

    def begin(self, identities) -> None:
        self.begun.append(set(identities))
        super().begin(identities)

    def get(self, key):
        self.looked_up.append(key)
        return super().get(key)


class _Stage:
    progress = None

    def set_loras(self, _entries) -> None:
        pass

    def begin_job(self) -> None:
        pass


def _entry_pipe(monkeypatch):
    """A light ``Ltx25Pipeline`` the REAL drivers run on up to the encode: the
    real job-state methods, a ``_StoppingEncoder``, and the official
    ``DistilledPipeline.__call__`` for the single job."""
    pipe = _pipe(monkeypatch, _StoppingEncoder, cache=_SpyCache())
    pipe.set_acceleration_job = lambda **_kw: None
    pipe.reset_acceleration_job = lambda: None
    pipe.stage = _Stage()
    pipe.vram = _Vram()
    pipe.device = CPU
    pipe.progress = None
    decoder = SimpleNamespace(checkpoint_path="vae.safetensors", diffvae_optimization=None)
    dp = SimpleNamespace(
        duration_predictor=None,
        image_conditioner=SimpleNamespace(resolve_crf=lambda images: images),
        stage=SimpleNamespace(),
        device=CPU,
        prompt_encoder=pipe.prompt_encoder,
        video_decoder=decoder,
    )

    class _Official:
        video_decoder = decoder

        def __call__(self, **kwargs):
            return pipeline25.DistilledPipeline.__call__(dp, **kwargs)

    pipe.pipeline = _Official()
    for module in (pipeline25, inpaint25, outpaint25, alphagen25, chain25):
        monkeypatch.setattr(module, "resolve_reference_downscale_factor", lambda *_a, **_k: 1)
    monkeypatch.setattr(chain25, "ensure_tiling_config", lambda *_a, **_k: None)
    monkeypatch.setattr(chain25, "tiling_scale_factors_for_vae", lambda *_a, **_k: None)
    monkeypatch.setattr(worker, "_PIPE", pipe)
    monkeypatch.setattr(worker, "_log", lambda _m: None)
    monkeypatch.setattr(worker, "_emit", lambda *_a, **_k: None)
    return pipe


def _nag_block() -> dict:
    return {"method": "nag", "negative_prompt": NAG.negative_prompt, "scale": NAG.scale,
            "tau": NAG.tau, "alpha": NAG.alpha}


def _chain_msg(tmp_path, clip_prompts, **extra) -> dict:
    msg = {
        "prompt": "s", "seed": 7, "width": 384, "height": 256, "frame_rate": 24.0,
        "overlap_frames": 3, "overlap_strength": 1.0, "output_path": str(tmp_path / "chain.mp4"),
        "clips": [{"prompt": p, "num_frames": 49} for p in clip_prompts],
    }
    msg.update(extra)
    return msg


def _entry_messages(tmp_path) -> dict:
    ref = {"path": str(tmp_path / "ref.mp4"), "strength": 1.0}
    lora = [{"path": str(tmp_path / "ic.safetensors"), "strength": 1.0}]
    return {
        "single": _msg(tmp_path),
        "outpaint": _msg(
            tmp_path, loras=lora, reference_video=ref,
            outpaint={"source_path": str(tmp_path / "src.mp4"), "canvas_width": 768, "canvas_height": 384,
                      "pad_left": 128, "pad_right": 128, "pad_top": 64, "pad_bottom": 64},
        ),
        "inpaint": _msg(
            tmp_path, loras=lora, reference_video=ref,
            inpaint={"source_path": str(tmp_path / "window.mp4"), "mask_path": str(tmp_path / "mask.mp4"),
                     "canvas_width": 384, "canvas_height": 256, "source_width": 320, "source_height": 256},
        ),
        "alpha_gen": _msg(tmp_path, prompt=" ", loras=lora, reference_video=ref, alpha_gen={"mode": "one_stage"}),
    }


@pytest.mark.parametrize("nag", [None, _nag_block()], ids=["no-negative", "nag"])
@pytest.mark.parametrize("entry", ["single", "outpaint", "inpaint", "alpha_gen"])
def test_each_single_op_entry_looks_up_the_bundle_it_started(monkeypatch, official, tmp_path, entry, nag):
    (tmp_path / "ref.mp4").write_bytes(b"\x00" * 32)
    pipe = _entry_pipe(monkeypatch)
    msg = _entry_messages(tmp_path)[entry]
    msg["keep_text_conditioning"] = True
    if nag is not None:
        msg["nag"] = nag
    with pytest.raises(_Stop):
        worker._do_generate(msg)

    expected = [msg["prompt"]] + ([] if nag is None else [NAG.negative_prompt])
    assert [call[1] for call in official.calls] == [expected]
    cache = pipe._te_conditioning
    assert cache.begun == [{tuple(expected)}]
    assert cache.looked_up == [(tuple(expected), ())]
    assert pipe._te_on is False  # ended by the worker's finally


def test_the_single_entry_hands_the_encoder_the_official_arguments(monkeypatch, official, tmp_path):
    _entry_pipe(monkeypatch)
    with pytest.raises(_Stop):
        worker._do_generate(_msg(tmp_path))
    assert official.calls[0][2] == OFFICIAL_KWARGS


@pytest.mark.parametrize("nag", [None, _nag_block()], ids=["no-negative", "nag"])
def test_the_chain_entry_looks_up_the_bundle_it_started(monkeypatch, official, tmp_path, nag):
    pipe = _entry_pipe(monkeypatch)
    msg = _chain_msg(tmp_path, ["s a", "s", "s a"], keep_text_conditioning=True)
    if nag is not None:
        msg["nag"] = nag
    with pytest.raises(_Stop):
        worker._do_generate_chain(msg)

    expected = ["s a", "s"] + ([] if nag is None else [NAG.negative_prompt])
    assert [call[1] for call in official.calls] == [expected]
    cache = pipe._te_conditioning
    assert cache.begun == [{tuple(expected)}]
    assert cache.looked_up == [(tuple(expected), ())]
    assert pipe._te_on is False
