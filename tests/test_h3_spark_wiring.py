"""Spark-H3's wiring in the MiniMax H3 engine, stage 1 of the plan "Spark-H3 を製品に組み込む計画"
(Docs/MINIMAX_H3_DESIGN.md §6.11): the attention's one entry (``dit.attend``), VRAM 16GB mode,
the clip's ``SparkClip`` in ``_denoise``, the records and the worker.

Runs in ``.venv-engine-h3`` through the engine-venv runner, without a GPU and without the
``spark_h3_kitchen`` wheel (the sparse attention is faked by one that returns the sdpa)::

    CUDA_VISIBLE_DEVICES=-1 .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_spark_wiring.py

What is pinned:
  * off: ``spark=None`` and no argument give the same output bit for bit and the same
    ``_attention`` calls (count, shapes, backend), on the normal path and in VRAM 16GB mode,
    and the same output as the attention called directly (the code before ``attend``);
  * the wiring: a dense evaluation never reaches the sparse attention, a sparse one reaches it
    from blocks 0..N-1 in order (once per block, or once per head group in VRAM 16GB mode), the
    token refiner never; a sparse attention that returns the sdpa gives the off output;
  * ``_denoise``: one ``SparkClip`` per clip, closed also after a failure; none for a job
    without Spark-H3 (and ``engine_h3.spark_h3`` not imported);
  * the records ("on" / "on->off" with each reason) of a single job and of a chain;
  * the worker's ``spark_h3`` key and ``ready``'s ``spark_h3_available`` (both paths).
"""

from __future__ import annotations

import builtins

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("comfy_kitchen")

import h3_format  # noqa: E402
from test_h3_engine import (  # noqa: E402
    GOLDEN_DONE_KEYS,
    GOLDEN_H3_KEYS,
    TEXT,
    _fake_int8,
    _tiny_dit_tensors,
    _tiny_forward_inputs,
    _tiny_vram16_dit,
    _write_sft,
)

from engine_h3 import dit as D  # noqa: E402
from engine_h3 import spark_h3 as SP  # noqa: E402

_REAL_ATTENTION = D._attention


class FakeSpark:
    """Stands in for a ``SparkClip`` handed to the DiT: notes (layer, heads) and returns the sdpa."""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def attention(self, q, k, v, layer):
        self.calls.append((int(layer), int(q.shape[1])))
        return _REAL_ATTENTION(q, k, v, "sdpa")


def _spy_attention(monkeypatch, *, compute="same"):
    """``dit._attention`` replaced by a spy noting (q, k, v shapes, dtype, backend); it computes
    the backend asked, or the sdpa when ``compute`` is "sdpa" (no sageattention on the CPU)."""
    calls = []

    def spy(q, k, v, backend):
        calls.append((tuple(q.shape), tuple(k.shape), tuple(v.shape), q.dtype, backend))
        return _REAL_ATTENTION(q, k, v, backend if compute == "same" else "sdpa")

    monkeypatch.setattr(D, "_attention", spy)
    return calls


def _plan(vram16: bool):
    from engine_h3.lowvram import Vram16Plan

    return Vram16Plan(head_group=3, row_chunk=16) if vram16 else None   # 4 heads: groups of 3 and 1


# --------------------------------------------------------------------------- #
# the DiT: off is unchanged
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("backend", ["sdpa", "sage"])
@pytest.mark.parametrize("vram16", [False, True])
def test_off_is_the_attention_as_before(monkeypatch, tmp_path, vram16, backend):
    _fake_int8(monkeypatch)
    dit = _tiny_vram16_dit(tmp_path, layers=3)
    tokens, kw = _tiny_forward_inputs(dit)
    kw["attention_backend"] = backend
    plan = _plan(vram16)
    calls = _spy_attention(monkeypatch, compute="sdpa")
    with torch.no_grad():
        plain = dit(tokens.clone(), vram16=plan, **kw)
        first = list(calls)
        calls.clear()
        none = dit(tokens.clone(), vram16=plan, spark=None, **kw)
        second = list(calls)
        # the code before ``attend``: the attention called directly with the block's backend
        monkeypatch.setattr(D, "attend", lambda attn, q, k, v: D._attention(q, k, v, attn.attention_backend))
        calls.clear()
        direct = dit(tokens.clone(), vram16=plan, **kw)
        third = list(calls)
    for a, b, c in zip(plain, none, direct):
        assert torch.equal(a, b) and torch.equal(a, c)
    groups = [3, 1] if vram16 else [4]
    want = [((41, g, 128),) * 3 + (torch.bfloat16, backend) for _block in range(3) for g in groups]
    assert first == second == third == want
    assert all(b.attn.spark is None and b.attn.spark_layer == i for i, b in enumerate(dit.blocks))


def test_attend_is_the_one_entry_of_both_paths():
    import inspect

    from engine_h3 import lowvram

    assert "attend(self, q, k, v)" in inspect.getsource(D.H3Attention.forward)
    assert "_attention(" not in inspect.getsource(D.H3Attention.forward)
    src = inspect.getsource(lowvram.block_forward)
    assert "_dit.attend(attn, q, k, v)" in src and "_attention(" not in src
    assert D.H3Attention.spark is None and D.H3Attention.spark_layer == -1


# --------------------------------------------------------------------------- #
# the DiT: the wiring
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("vram16", [False, True])
def test_sparse_evaluation_reaches_every_block_in_order(monkeypatch, tmp_path, vram16):
    _fake_int8(monkeypatch)
    dit = _tiny_vram16_dit(tmp_path, layers=3)
    tokens, kw = _tiny_forward_inputs(dit)
    text = torch.randn(6, dit.cfg.text_dim, generator=torch.Generator().manual_seed(9)).to(torch.bfloat16)
    plan = _plan(vram16)
    calls = _spy_attention(monkeypatch)
    spark = FakeSpark()
    with torch.no_grad():
        refined = dit.refine_text(text)
        refiner_calls = len(calls)
        assert refiner_calls > 0                            # the token refiner attends through _attention
        calls.clear()
        plain = dit(tokens.clone(), vram16=plan, **kw)
        assert spark.calls == []
        calls.clear()
        got = dit(tokens.clone(), vram16=plan, spark=spark, **kw)
        assert calls == []                                  # no block went to _attention
        groups = [3, 1] if vram16 else [4]
        assert spark.calls == [(layer, g) for layer in range(3) for g in groups]
        assert all(b.attn.spark is spark for b in dit.blocks)
        assert [b.attn.spark_layer for b in dit.blocks] == [0, 1, 2]
        assert all(b.attn.spark is None and b.attn.spark_layer == -1 for b in dit.token_refiner.blocks)
        calls.clear()
        again_text = dit.refine_text(text)                  # the refiner after a sparse forward
        assert len(calls) == refiner_calls and len(spark.calls) == 3 * len(groups)
        assert torch.equal(again_text, refined)
        dense = dit(tokens.clone(), vram16=plan, spark=None, **kw)   # a dense evaluation after it
        assert len(spark.calls) == 3 * len(groups)
    assert all(b.attn.spark is None for b in dit.blocks)
    for a, b, c in zip(plain, got, dense):
        assert torch.equal(a, b) and torch.equal(a, c)


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

def _job_pipe(monkeypatch, tmp_path, *, heads=4, layers=3):
    """The REAL generate / _denoise on a tiny DiT, the encode, the decoders and the mp4 faked.
    ``rec``: the latents the video decoder got, the sparse attention's calls
    ``(evaluation_index, layer, heads)`` and the SparkClips made / closed."""
    from engine_h3 import audio_vae as AV
    from engine_h3 import media_out as MO
    from engine_h3 import pipeline_h3 as P
    from engine_h3 import video_vae as VV

    _fake_int8(monkeypatch)
    dit_path = tmp_path / "tiny.safetensors"
    if not dit_path.exists():
        _write_sft(dit_path, _tiny_dit_tensors(layers=layers, refiner=1, grid=1025, rank=8, heads=heads))
    _header, cfg = h3_format.inspect_h3_dit(str(dit_path))
    pipe = P.H3Pipeline(P.H3Paths(str(dit_path), "te", "vv", "av", "tok"), P.H3LoadOptions(blocks_on_gpu=1),
                        device="cpu", progress=lambda *a: None, log=lambda m: None)
    pipe._dit_cfg = cfg
    rec = {"latents": [], "spark": [], "made": [], "closed": [], "fail_at": None}
    monkeypatch.setattr(pipe, "_encode", lambda prompt, phases, **kw: (
        torch.randn(5, TEXT, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16),
        torch.ones(5, dtype=torch.long)))

    class FakeDecoder(torch.nn.Module):
        def decode(self, z, tiled=True):
            rec["latents"].append(z.clone())
            return torch.zeros((_frames(z), 64, 64, 3), dtype=torch.uint8)

    class FakeAudio(torch.nn.Module):
        def decode(self, a):
            return torch.zeros((2, 800 * a.shape[-1]))

    monkeypatch.setattr(VV.H3VideoDecoder, "from_file", classmethod(lambda cls, *a, **k: FakeDecoder()))
    monkeypatch.setattr(AV.H3AudioVAE, "from_file", classmethod(lambda cls, *a, **k: FakeAudio()))
    monkeypatch.setattr(MO, "write_mp4", lambda path, frames, **k: 1)

    monkeypatch.setattr(SP, "probe", lambda: None)
    real_init, real_close = SP.SparkClip.__init__, SP.SparkClip.close

    def init(self, *a, **k):
        real_init(self, *a, **k)
        rec["made"].append(self)

    def close(self):
        rec["closed"].append(self)
        real_close(self)

    def attention(self, q, k, v, layer):
        i = self.controller.evaluation_index
        if rec["fail_at"] == i:
            raise RuntimeError("kernel failed")
        rec["spark"].append((i, int(layer), int(q.shape[1])))
        return _REAL_ATTENTION(q, k, v, "sdpa")

    monkeypatch.setattr(SP.SparkClip, "__init__", init)
    monkeypatch.setattr(SP.SparkClip, "close", close)
    monkeypatch.setattr(SP.SparkClip, "attention", attention)

    def job(spark=None, steps=4, vram16=False, keep=False, frames=22):
        req = P.H3GenerateRequest(prompt="p", seed=7, width=64, height=64, num_frames=frames, output_path="o.mp4",
                                  num_inference_steps=steps, keep_resident=keep, vram16_mode=vram16,
                                  shift_video=12.0, shift_audio=3.0, spark_h3=spark)
        return pipe.generate(req)

    return pipe, rec, job


def _frames(z) -> int:
    from engine_h3 import dwd

    return dwd.frames_for_latents(int(z.shape[2]))


def _spark(ratio=0.2, full=1):
    from engine_h3.pipeline_h3 import H3SparkInput

    return H3SparkInput(topk_ratio=ratio, full_steps=full)


@pytest.mark.parametrize("vram16", [False, True])
def test_job_sparse_after_the_full_steps_and_off_equal(monkeypatch, tmp_path, vram16):
    monkeypatch.setattr(SP, "MIN_TOKENS", 0)       # the tiny sequence counts as long
    _pipe, rec, job = _job_pipe(monkeypatch, tmp_path)
    off = job(None, vram16=vram16)
    on = job(_spark(0.2, 1), vram16=vram16)
    assert torch.equal(rec["latents"][0], rec["latents"][1])   # a sparse attention that is the sdpa = off
    assert [(i, layer) for i, layer, _h in rec["spark"]] == [(i, layer) for i in (1, 2, 3) for layer in range(3)]
    assert {h for _i, _l, h in rec["spark"]} == {4}             # all 4 heads (one head group of 8)
    assert len(rec["made"]) == 1 and rec["closed"] == rec["made"] and rec["made"][0].controller is None
    assert "spark_h3_used" not in off and "spark_h3" not in off["h3"]
    assert on["spark_h3_used"] == "on" and list(on)[-2:] == ["spark_h3_used", "h3"]
    r = on["h3"]["spark_h3"]
    seq = on["h3"]["phases"]["denoise"]["sequence"]
    assert r["used"] == "on" and "reason" not in r
    assert (r["topk_ratio"], r["full_steps"], r["steps"], r["min_tokens"]) == (0.2, 1, 4, 0)
    assert (r["sequence_tokens"], r["video_tokens"], r["grid"]) == (seq["total"], seq["video"], [7, 2, 2])
    assert r["sparse_evaluations"] == 3 and r["kitchen_version"] == SP.kitchen_version()
    assert list(on["h3"])[-1] == "spark_h3" and "spark_h3" not in on["h3"]["phases"]["denoise"]


def test_ratio_reaches_the_config_unrounded(monkeypatch, tmp_path):
    monkeypatch.setattr(SP, "MIN_TOKENS", 0)
    _pipe, rec, job = _job_pipe(monkeypatch, tmp_path)
    done = job(_spark(0.123456789, 2))
    clip = rec["made"][0]
    assert clip.topk_ratio == 0.123456789 and clip.full_steps == 2 and clip.steps == 4
    assert done["h3"]["spark_h3"]["topk_ratio"] == 0.123456789
    assert [i for i, layer, _h in rec["spark"] if layer == 0] == [2, 3]


def test_ratio_out_of_the_authors_range_fails_the_job(monkeypatch, tmp_path):
    _pipe, rec, job = _job_pipe(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="topk_ratio must lie in"):
        job(_spark(0.0, 1))
    assert rec["made"] == [] and rec["latents"] == []


@pytest.mark.parametrize("vram16", [False, True])
def test_a_failure_in_the_loop_still_closes(monkeypatch, tmp_path, vram16):
    monkeypatch.setattr(SP, "MIN_TOKENS", 0)
    pipe, rec, job = _job_pipe(monkeypatch, tmp_path)
    rec["fail_at"] = 2
    with pytest.raises(RuntimeError, match="kernel failed"):
        job(_spark(0.2, 1), vram16=vram16, keep=True)
    assert len(rec["made"]) == 1 and rec["closed"] == rec["made"] and rec["made"][0].controller is None
    assert rec["latents"] == [] and pipe._resident is None


@pytest.mark.parametrize("fail", ["close", "record", "both"])
@pytest.mark.parametrize("vram16", [False, True])
def test_a_failing_spark_teardown_neither_skips_the_rest_nor_fails_the_job(monkeypatch, tmp_path, vram16, fail):
    """The Spark-H3 teardown raising: the job completes with the same latents, and the denoise's
    other teardown still runs (VRAM 16GB mode's record, the kept DiT parked, the release mark)."""
    monkeypatch.setattr(SP, "MIN_TOKENS", 0)
    pipe, rec, job = _job_pipe(monkeypatch, tmp_path)
    logs = []
    monkeypatch.setattr(pipe, "_log", logs.append)
    detached = []
    from engine_h3 import lora as L

    real_detach = L.detach
    monkeypatch.setattr(L, "detach", lambda dit: detached.append(dit) or real_detach(dit))
    off = job(None, vram16=vram16, keep=True)
    wrapped_close = SP.SparkClip.close           # _job_pipe's recording close

    def close(self):
        wrapped_close(self)                      # the real close ran (the state is let go) ...
        if fail in ("close", "both"):
            raise RuntimeError("close failed")   # ... and then it raised

    def record(self):
        raise RuntimeError("record failed")

    monkeypatch.setattr(SP.SparkClip, "close", close)
    if fail in ("record", "both"):
        monkeypatch.setattr(SP.SparkClip, "record", record)
    pipe._resident = None
    done = job(_spark(0.2, 1), vram16=vram16, keep=True)
    assert torch.equal(rec["latents"][0], rec["latents"][1])
    assert len(rec["made"]) == 1 and rec["closed"] == rec["made"] and rec["made"][0].controller is None
    denoise = done["h3"]["phases"]["denoise"]
    assert ("vram16" in denoise) is vram16 and ("vram16" in off["h3"]["phases"]["denoise"]) is vram16
    assert "dit_parked_tensors" in denoise and pipe._resident is not None and "dit" in pipe._resident
    assert len(detached) == 2                    # the LoRA detach ran after both jobs
    for what in {"close": ["close"], "record": ["record"], "both": ["record", "close"]}[fail]:
        assert any(f"Spark-H3 {what} failed (ignored)" in m for m in logs), what
    r = done["h3"]["spark_h3"]
    assert done["spark_h3_used"] == "on" and r["used"] == "on" and r["sparse_evaluations"] == 3
    if fail == "close":
        assert r["kernel_calls"] == 0 and r["sequence_tokens"] == denoise["sequence"]["total"]
    else:
        assert r["kernel_calls"] is None and r["record_error"] == "RuntimeError('record failed')"
        assert "sequence_tokens" not in r and "reason" not in r


def test_teardown_stand_in_names_the_reason_of_a_clip_without_sparse_evaluations():
    from engine_h3.pipeline_h3 import _spark_clip_teardown

    class Clip:
        sparse_evaluations = 0

        def record(self):
            raise RuntimeError("x")

        def reason(self):
            return SP.FALLBACK_FEW_STEPS

        def close(self):
            raise OSError("y")

    logs = []
    assert _spark_clip_teardown(Clip(), logs.append) == {
        "sparse_evaluations": 0, "reason": SP.FALLBACK_FEW_STEPS, "record_error": "RuntimeError('x')"}
    assert len(logs) == 2 and "record failed" in logs[0] and "close failed" in logs[1]


@pytest.mark.parametrize("spark, steps, short, reason", [
    (_spark(0.2, 4), 4, False, SP.FALLBACK_FEW_STEPS),
    (_spark(0.2, 10), 2, False, SP.FALLBACK_FEW_STEPS),
    (_spark(0.2, 1), 4, True, SP.FALLBACK_SHORT_SEQUENCE),
])
def test_clip_without_a_sparse_evaluation_is_on_to_off(monkeypatch, tmp_path, spark, steps, short, reason):
    if not short:
        monkeypatch.setattr(SP, "MIN_TOKENS", 0)
    _pipe, rec, job = _job_pipe(monkeypatch, tmp_path)
    off = job(None, steps=steps)
    done = job(spark, steps=steps)
    assert torch.equal(rec["latents"][0], rec["latents"][1])
    assert rec["spark"] == [] and len(rec["made"]) == 1 and rec["closed"] == rec["made"]
    assert done["spark_h3_used"] == "on->off"
    r = done["h3"]["spark_h3"]
    assert r["used"] == "on->off" and r["reason"] == reason and r["sparse_evaluations"] == 0
    assert r["sequence_tokens"] == off["h3"]["phases"]["denoise"]["sequence"]["total"]
    if short:
        assert r["min_tokens"] == 12288 and r["sequence_tokens"] < 12288


def test_unavailable_runs_without_spark_and_says_why(monkeypatch, tmp_path):
    _pipe, rec, job = _job_pipe(monkeypatch, tmp_path)
    monkeypatch.setattr(SP, "probe", lambda: "Spark-H3 の部品（spark-h3-kitchen）が入っていません")
    monkeypatch.setattr(SP, "kitchen_version", lambda: None)
    job(None)
    done = job(_spark(0.3, 1))
    assert torch.equal(rec["latents"][0], rec["latents"][1])
    assert rec["made"] == [] and rec["spark"] == []
    assert done["spark_h3_used"] == "on->off"
    assert done["h3"]["spark_h3"] == {
        "used": "on->off", "topk_ratio": 0.3, "full_steps": 1, "min_tokens": 12288, "kitchen_version": None,
        "reason": "unavailable", "detail": "Spark-H3 の部品（spark-h3-kitchen）が入っていません",
    }


def test_off_job_makes_nothing_and_imports_nothing(monkeypatch, tmp_path):
    from engine_h3 import pipeline_h3 as P
    from engine_h3.dit import H3DiT

    _pipe, rec, job = _job_pipe(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise AssertionError("an off job reached the Spark-H3 code")

    monkeypatch.setattr(SP, "probe", boom)
    monkeypatch.setattr(SP.SparkClip, "__init__", boom)
    imports = []
    real_import = builtins.__import__

    def spy_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "engine_h3.spark_h3" or (name == "engine_h3" and "spark_h3" in (fromlist or ())):
            imports.append(name)
        return real_import(name, globals, locals, fromlist, level)

    real_forward = H3DiT.forward

    def forward(self, *a, **k):
        assert "spark" not in k                  # the DiT's call is the one before Spark-H3
        return real_forward(self, *a, **k)

    monkeypatch.setattr(H3DiT, "forward", forward)
    monkeypatch.setattr(builtins, "__import__", spy_import)
    try:
        done = job(None)
        assert P.spark_start(P.H3GenerateRequest(prompt="p", seed=1, width=64, height=64, num_frames=22,
                                                 output_path="o.mp4")) is None
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
    assert imports == []
    assert list(done) == GOLDEN_DONE_KEYS and list(done["h3"]) == GOLDEN_H3_KEYS
    assert "spark_h3" not in str(done)


def test_validate_request_full_steps(tmp_path):
    from engine_h3.pipeline_h3 import H3GenerateRequest, validate_request

    base = dict(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4")
    for full in (0, 1, 10, 50):
        validate_request(H3GenerateRequest(**base, spark_h3=_spark(0.2, full)))
    validate_request(H3GenerateRequest(**base, spark_h3=_spark(7.0, 1)))   # the ratio is the author's to refuse
    for bad in (-1, True, 1.5, "2", None):
        with pytest.raises(ValueError, match=r"Spark-H3 のフル演算するステップ数は 0 以上の整数にしてください"):
            validate_request(H3GenerateRequest(**base, spark_h3=_spark(0.2, bad)))


def test_spark_record_function(monkeypatch):
    from engine_h3.pipeline_h3 import SPARK_CLIP_FACTS, spark_record

    monkeypatch.setattr(SP, "kitchen_version", lambda: "v")
    start = {"topk_ratio": 0.2, "full_steps": 1, "unavailable": None}
    settings = {"topk_ratio": 0.2, "full_steps": 1, "min_tokens": 12288, "kitchen_version": "v"}

    def clip(sparse, reason, seq):
        return {"topk_ratio": 0.2, "full_steps": 1, "steps": 4, "min_tokens": 12288, "sequence_tokens": seq,
                "video_tokens": seq - 100, "grid": [7, 2, 2], "sparse_evaluations": sparse,
                "kernel_calls": 50 * sparse, "backend": "b", "kitchen_version": "v", "reason": reason}

    # one clip (a single job, or chain.clips[i]): the settings and the clip's own facts
    used, r = spark_record(start, [clip(3, None, 20000)])
    assert used == "on" and r == {**{k: v for k, v in clip(3, None, 20000).items() if k != "reason"}, "used": "on"}
    assert list(r)[:5] == ["used", *settings]
    # a whole chain: the settings, used and the sums only; no clip's size
    short_then_long = [clip(0, "short_sequence", 9000), clip(3, None, 20000)]
    used, r = spark_record(start, short_then_long, whole_chain=True)
    assert used == "on" and r == {"used": "on", **settings, "sparse_evaluations": 3, "kernel_calls": 150}
    assert not set(SPARK_CLIP_FACTS) & set(r)
    used, r = spark_record(start, [clip(0, "short_sequence", 9000), clip(0, "few_steps", 20000)], whole_chain=True)
    assert used == "on->off" and r == {"used": "on->off", **settings, "sparse_evaluations": 0, "kernel_calls": 0,
                                       "reason": "short_sequence"}
    # a clip whose record could not be taken: its stand-in, kernel_calls unknown
    stand_in = {"sparse_evaluations": 2, "reason": None, "record_error": "RuntimeError('x')"}
    used, r = spark_record(start, [stand_in])
    assert used == "on" and r == {"used": "on", **settings, "sparse_evaluations": 2, "kernel_calls": None,
                                  "record_error": "RuntimeError('x')"}
    used, r = spark_record(start, [clip(3, None, 20000), stand_in], whole_chain=True)
    assert (used, r["sparse_evaluations"], r["kernel_calls"]) == ("on", 5, None)
    # unavailable: the settings and the reason
    used, r = spark_record({**start, "unavailable": "CUDA が使えません"}, [], whole_chain=True)
    assert used == "on->off" and r == {"used": "on->off", **settings, "reason": "unavailable",
                                       "detail": "CUDA が使えません"}


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #

def _chain(job_pipe, spark, frames=(22, 39), steps=3):
    from engine_h3 import chain_h3 as CH
    from engine_h3 import pipeline_h3 as P

    pipe = job_pipe
    base = P.H3GenerateRequest(prompt="p", seed=0, width=64, height=64, num_frames=frames[0], output_path="o.mp4",
                               num_inference_steps=steps, keep_resident=False, shift_video=12.0, shift_audio=3.0,
                               spark_h3=spark)
    spec = CH.H3ChainSpec(base=base, clips=tuple(CH.H3ChainClipSpec(f"c{i}", f) for i, f in enumerate(frames)),
                          seed=11)
    return CH.run_chain_h3(pipe, spec, lambda *a, **k: None, lambda m: None)


def test_chain_makes_one_state_per_clip_and_counts_steps_from_0(monkeypatch, tmp_path):
    monkeypatch.setattr(SP, "MIN_TOKENS", 0)
    pipe, rec, _job = _job_pipe(monkeypatch, tmp_path)
    off = _chain(pipe, None)
    done = _chain(pipe, _spark(0.2, 1))
    assert torch.equal(rec["latents"][0], rec["latents"][1])
    assert len(rec["made"]) == 2 and rec["made"][0] is not rec["made"][1] and rec["closed"] == rec["made"]
    by_clip = [[(i, layer) for i, layer, _h in rec["spark"][k * 6:(k + 1) * 6]] for k in range(2)]
    assert by_clip == [[(i, layer) for i in (1, 2) for layer in range(3)]] * 2 and len(rec["spark"]) == 12
    assert done["spark_h3_used"] == "on"
    assert done["h3"]["spark_h3"] == {"used": "on", "topk_ratio": 0.2, "full_steps": 1, "min_tokens": 0,
                                      "kitchen_version": SP.kitchen_version(), "sparse_evaluations": 4,
                                      "kernel_calls": 0}   # the fake attention never reaches the kernel
    clips = done["chain"]["clips"]
    assert [c["spark_h3"]["used"] for c in clips] == ["on", "on"]
    assert [c["spark_h3"]["sparse_evaluations"] for c in clips] == [2, 2]
    seqs = [p["denoise"]["sequence"]["total"] for p in done["h3"]["phases"]["clips"]]
    assert [c["spark_h3"]["sequence_tokens"] for c in clips] == seqs and seqs[0] < seqs[1]
    assert "spark_h3" not in done["h3"]["phases"]["clips"][0]["denoise"]
    assert "spark_h3_used" not in off and "spark_h3" not in off["h3"]
    assert all("spark_h3" not in c for c in off["chain"]["clips"])


def test_chain_records_each_clip_and_the_whole(monkeypatch, tmp_path):
    pipe, rec, _job = _job_pipe(monkeypatch, tmp_path)
    monkeypatch.setattr(SP, "MIN_TOKENS", 0)
    first = _chain(pipe, _spark(0.2, 1))
    seqs = [c["spark_h3"]["sequence_tokens"] for c in first["chain"]["clips"]]
    # clip 0 short, clip 1 long: "on" for the chain, clip 0 "on->off" with its reason
    monkeypatch.setattr(SP, "MIN_TOKENS", seqs[1])
    mixed = _chain(pipe, _spark(0.2, 1))
    c0, c1 = (c["spark_h3"] for c in mixed["chain"]["clips"])
    assert (c0["used"], c0["reason"], c0["sparse_evaluations"]) == ("on->off", SP.FALLBACK_SHORT_SEQUENCE, 0)
    assert (c1["used"], c1["sparse_evaluations"]) == ("on", 2) and "reason" not in c1
    assert mixed["spark_h3_used"] == "on" and "reason" not in mixed["h3"]["spark_h3"]
    whole = mixed["h3"]["spark_h3"]   # no short clip 0's size beside "on"
    assert "sequence_tokens" not in whole and "grid" not in whole and "steps" not in whole
    assert (whole["used"], whole["sparse_evaluations"]) == ("on", 2)
    # every clip short: the chain is "on->off" with the first clip's reason
    monkeypatch.setattr(SP, "MIN_TOKENS", seqs[1] + 1)
    short = _chain(pipe, _spark(0.2, 1))
    assert short["spark_h3_used"] == "on->off"
    assert short["h3"]["spark_h3"]["reason"] == SP.FALLBACK_SHORT_SEQUENCE
    assert [c["spark_h3"]["used"] for c in short["chain"]["clips"]] == ["on->off"] * 2
    # too few steps in every clip
    few = _chain(pipe, _spark(0.2, 3))
    assert few["h3"]["spark_h3"]["reason"] == SP.FALLBACK_FEW_STEPS
    # unavailable: no state at all, the same record in every clip and for the chain
    made = len(rec["made"])
    monkeypatch.setattr(SP, "probe", lambda: "CUDA が使えません")
    gone = _chain(pipe, _spark(0.2, 1))
    assert len(rec["made"]) == made and gone["spark_h3_used"] == "on->off"
    assert gone["h3"]["spark_h3"]["reason"] == "unavailable" and gone["h3"]["spark_h3"]["detail"] == "CUDA が使えません"
    assert [c["spark_h3"] for c in gone["chain"]["clips"]] == [gone["h3"]["spark_h3"]] * 2


def test_chain_failure_in_clip_1_closes_its_state(monkeypatch, tmp_path):
    monkeypatch.setattr(SP, "MIN_TOKENS", 0)
    pipe, rec, _job = _job_pipe(monkeypatch, tmp_path)
    real = SP.SparkClip.attention

    def attention(self, q, k, v, layer):
        if len(rec["made"]) == 2:
            raise RuntimeError("kernel failed in clip 1")
        return real(self, q, k, v, layer)

    monkeypatch.setattr(SP.SparkClip, "attention", attention)
    with pytest.raises(RuntimeError, match="kernel failed in clip 1"):
        _chain(pipe, _spark(0.2, 1))
    assert len(rec["made"]) == 2 and rec["closed"] == rec["made"]


# --------------------------------------------------------------------------- #
# the worker
# --------------------------------------------------------------------------- #

MSG = {"op": "generate", "prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124,
       "output_path": "o.mp4"}


def test_worker_spark_h3_key():
    from engine_h3 import worker
    from engine_h3.pipeline_h3 import H3SparkInput

    assert worker._generate_request(MSG).spark_h3 is None
    assert worker._generate_request({**MSG, "spark_h3": None}).spark_h3 is None
    req = worker._generate_request({**MSG, "spark_h3": {"topk_ratio": 0.15, "full_steps": 2}})
    assert req.spark_h3 == H3SparkInput(topk_ratio=0.15, full_steps=2)
    one = worker._spark_h3({"spark_h3": {"topk_ratio": 1, "full_steps": 0, "extra": 1}})
    assert one == (H3SparkInput(1.0, 0), ["extra"]) and type(one[0].topk_ratio) is float
    assert worker._spark_h3({"spark_h3": {"topk_ratio": 0.2, "full_steps": -1}})[0].full_steps == -1   # validate_request's
    cases = [
        ([0.2, 1], "spark_h3 は topk_ratio と full_steps を持つオブジェクトで渡してください（list）"),
        ("on", "spark_h3 は topk_ratio と full_steps を持つオブジェクトで渡してください（str）"),
        ({"topk_ratio": True, "full_steps": 1}, "spark_h3 の topk_ratio は数値で渡してください（True）"),
        ({"topk_ratio": "0.2", "full_steps": 1}, "spark_h3 の topk_ratio は数値で渡してください（'0.2'）"),
        ({"full_steps": 1}, "spark_h3 の topk_ratio は数値で渡してください（None）"),
        ({"topk_ratio": 0.2, "full_steps": False}, "spark_h3 の full_steps は整数で渡してください（False）"),
        ({"topk_ratio": 0.2, "full_steps": "2"}, "spark_h3 の full_steps は整数で渡してください（'2'）"),
        ({"topk_ratio": 0.2, "full_steps": 2.0}, "spark_h3 の full_steps は整数で渡してください（2.0）"),
    ]
    for bad, text in cases:
        with pytest.raises(ValueError) as exc:
            worker._spark_h3({"spark_h3": bad})
        assert str(exc.value) == text
    assert "spark_h3" in worker._GENERATE_FIELDS and "spark_h3" in worker._CHAIN_FIELDS


def test_worker_negative_full_steps_is_refused_by_the_pipeline():
    from engine_h3 import worker
    from engine_h3.pipeline_h3 import validate_request

    req = worker._generate_request({**MSG, "spark_h3": {"topk_ratio": 0.2, "full_steps": -1}})
    with pytest.raises(ValueError, match=r"フル演算するステップ数は 0 以上の整数にしてください（-1）"):
        validate_request(req)


def test_worker_generate_passes_and_logs_spark_h3(monkeypatch):
    from engine_h3 import worker
    from engine_h3.pipeline_h3 import H3SparkInput

    class FakePipe:
        seen = []

        def generate(self, req):
            FakePipe.seen.append(req)
            return {"seed_used": 1, **({"spark_h3_used": "on"} if req.spark_h3 is not None else {})}

    events, logs = [], []
    monkeypatch.setattr(worker, "_PIPE", FakePipe())
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(worker, "_log", logs.append)
    worker._do_generate({**MSG, "spark_h3": {"topk_ratio": 0.2, "full_steps": 2, "x": 1}})
    worker._do_generate(dict(MSG))
    assert FakePipe.seen[0].spark_h3 == H3SparkInput(0.2, 2) and FakePipe.seen[1].spark_h3 is None
    assert events[0]["spark_h3_used"] == "on" and "spark_h3_used" not in events[1]
    assert "spark_h3=(topk_ratio=0.2, full_steps=2)" in logs[0] and "ignored keys: spark_h3.x" in logs[0]
    assert "spark_h3" not in logs[1] and "ignored keys" not in logs[1]


def test_worker_chain_reads_spark_h3(monkeypatch):
    from engine_h3 import chain_h3 as CH
    from engine_h3 import worker
    from engine_h3.pipeline_h3 import H3SparkInput

    seen, events, logs = [], [], []

    def run(pipe, spec, progress, log):
        seen.append(spec)
        return {"seed_used": 1, **({"spark_h3_used": "on->off"} if spec.base.spark_h3 is not None else {})}

    monkeypatch.setattr(CH, "run_chain_h3", run)
    monkeypatch.setattr(worker, "_PIPE", object())
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(worker, "_log", logs.append)
    msg = {"op": "generate_chain", "seed": 1, "width": 640, "height": 384, "output_path": "o.mp4",
           "clips": [{"prompt": "a", "num_frames": 124}, {"prompt": "b", "num_frames": 124}]}
    worker._do_generate_chain({**msg, "spark_h3": {"topk_ratio": 0.1, "full_steps": 1, "x": 1}})
    worker._do_generate_chain(dict(msg))
    assert seen[0].base.spark_h3 == H3SparkInput(0.1, 1) and seen[1].base.spark_h3 is None
    assert events[0]["spark_h3_used"] == "on->off" and "spark_h3_used" not in events[1]
    assert "spark_h3=(topk_ratio=0.1, full_steps=1)" in logs[0] and "ignored keys: spark_h3.x" in logs[0]
    assert "spark_h3" not in logs[1]
    with pytest.raises(ValueError, match="spark_h3 の full_steps は整数"):
        worker._do_generate_chain({**msg, "spark_h3": {"topk_ratio": 0.1, "full_steps": True}})


@pytest.mark.parametrize("reason, available", [(None, True), ("CUDA が使えません", False)])
def test_worker_ready_carries_spark_h3_available_after_the_load(monkeypatch, reason, available):
    import engine.transformer.sage_attention_service as sage

    from engine_h3 import pipeline_h3 as P
    from engine_h3 import worker

    order = []

    class FakePipeline:
        sampler = "euler"

        def __init__(self, paths, options, device, progress, log):
            order.append("init")

        def load(self):
            order.append("load")
            return {}

    def probe():
        order.append("probe")
        return reason

    events = []
    monkeypatch.setattr(worker, "_PIPE", None)
    monkeypatch.setattr(worker, "_ATTENTION", worker._ATTENTION)
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(worker, "_log", lambda m: None)
    monkeypatch.setattr(h3_format, "check_model_paths", lambda files, dirs: [])
    monkeypatch.setattr(sage, "probe_sage", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(P, "H3Pipeline", FakePipeline)
    monkeypatch.setattr(SP, "probe", probe)
    msg = {"op": "load", "transformer_path": "t", "text_encoder_path": "te", "video_vae_path": "vv",
           "audio_vae_path": "av", "tokenizer_dir": "tok"}
    worker._do_load(msg)
    assert order == ["init", "load", "probe"]                  # the probe after the load
    worker._do_load(msg)                                        # "already loaded": answered again
    assert order == ["init", "load", "probe", "probe"]
    assert events == [{"event": "ready", "sampler": "euler", "sage_available": True,
                       "spark_h3_available": available}] * 2
