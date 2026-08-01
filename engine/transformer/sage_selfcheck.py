"""Standalone self-check for engine.transformer.sage_attention_service.

Run with the ENGINE venv (needs the real ltx_core wheel, a CUDA GPU, and the
sageattention wheel — none of which the app venv has):

    .venv-engine\\Scripts\\python.exe -m engine.transformer.sage_selfcheck

Same conventions as nag_selfcheck / vsf_selfcheck: NOT a pytest module, every
check either PASSes or FAILs loudly (nothing is skipped), exit code 0 only when
all of them pass. Deliberately SLIM — three checks, one per property that can
actually break silently:

  1. NUMERICAL PARITY. A real ``Attention`` module's patched
     ``attention_function`` produces the same tensor as the wheel's own
     PytorchAttention to within quantization noise (relative L2 + cosine
     similarity — see _REL_TOL/_COS_TOL for why both), for both head dims this
     model uses (128 video / 64 audio), in bf16 and fp16. The fallback is
     replaced by a counting spy first, so "the numbers matched" cannot be
     satisfied by quietly not using sage at all.

  2. FALLBACK MATRIX. Each degradation route hands the ORIGINAL, untouched
     q/k/v/mask to the original callable and returns its result BIT-identically
     — i.e. a fallback call is indistinguishable from an sdpa run. Six rows:
     attention mask present (the IC-LoRA attention_strength path — which must
     also be counted per call and logged exactly ONCE per job, since it leaves
     attention_used at "sage" and would otherwise be untraceable), fp32 dtype,
     non-CUDA tensors, a stale wrapper whose job already ended, a kernel that
     raises (which must latch for the rest of the job and be attempted exactly
     once), and the install-time static skip of an unsupported head_dim.
     ``reset()`` must clear the mask counter along with everything else.

  3. INSTALL COUNT. A 48-block transformer yields exactly 288 wrapped modules —
     6 per block (attn1 / attn2 / audio_attn1 / audio_attn2 /
     audio_to_video_attn / video_to_audio_attn), matching the real-device spike.
     Plus the OFF case: a job that did not request sage leaves every module's
     ``attention_function`` untouched, identity-checked.

Why no end-to-end generation here: that is the real-device gate's job (G4/G5),
and it needs the 16GB model. This file only has to prove the swap itself is
correct and safe.
"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import Callable

import torch

from engine.transformer import sage_attention_service as sage_mod
from engine.transformer.sage_attention_service import (
    SageAttentionService,
    SageState,
    probe_sage,
)

# Error budget for the sage kernel vs PyTorch SDPA. SageAttention quantizes Q/K
# to INT8 and (on Ada) accumulates PV in FP8, so it is never bit-exact; these
# bounds exist to catch a kernel that returns zeros, garbage, or a transposed
# layout — not to pin down quantization noise.
#
# Why two statistics, and why the L2 bound looks so loose: the inputs here are
# i.i.d. Gaussians, which is the WORST case for relative error. Random q/k make
# the softmax nearly uniform, so the output is an average of ~S random value
# vectors and its magnitude collapses to roughly ‖v‖/sqrt(S) — the numerator of
# the relative error stays put while the denominator shrinks. Measured on the
# 4070 Ti SUPER (RTX 40xx / SM89, sageattention 2.2.0), rel_l2 sits at 0.037-0.039
# and cosine similarity at 0.9993, both stable across bf16/fp16, d_head 64/128 and
# S from 256 to 4096. Real activations are far more peaked and score better.
#
# So the cosine bound is the sharp one (a layout error or a dropped head tanks it
# immediately); the L2 bound is the coarse "is this even attention" net, set at
# ~2x the observed value.
_REL_TOL = 0.08
_COS_TOL = 0.995

# The 6 attention modules every dual-stream block carries.
_EXPECTED_KINDS = (
    "attn1",
    "attn2",
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
)

_RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    _RESULTS.append((name, passed, detail))


def _run_check(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        _record(name, True)
    except Exception as exc:  # noqa: BLE001 — a broken check must FAIL loudly, never skip
        _record(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


class _FakeTransformer(torch.nn.Module):
    """Minimal stand-in for a built transformer.

    ``_attention_modules`` walks ``named_modules()``, so a plain ModuleList of
    real blocks is all it needs — no X0Model/velocity_model wrapper (that one is
    NAG's requirement, whose traversal goes by attribute name).
    """

    def __init__(self, blocks: list[torch.nn.Module]) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(blocks)


class _LogCapture(logging.Handler):
    """Collect the sage service's own log records for the duration of a block.

    Used to assert the "logged once per job, not once per call" contract — the
    thing that makes the masked-call fallback observable (G5's evidence) without
    turning an IC-LoRA job's log into tens of thousands of identical lines.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []
        self._saved_level = logging.NOTSET

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def __enter__(self) -> "_LogCapture":
        self._saved_level = sage_mod.logger.level
        sage_mod.logger.setLevel(logging.DEBUG)
        sage_mod.logger.addHandler(self)
        return self

    def __exit__(self, *exc_info: object) -> bool:
        sage_mod.logger.removeHandler(self)
        sage_mod.logger.setLevel(self._saved_level)
        return False


class _Spy:
    """Original-callable stand-in that records how it was invoked.

    Wraps the wheel's real PytorchAttention so its RESULT is the genuine sdpa
    answer, while ``calls`` proves whether the sage path was taken.
    """

    def __init__(self) -> None:
        from ltx_core.model.transformer.attention import PytorchAttention

        self._inner = PytorchAttention()
        self.calls = 0
        self.last_args: tuple = ()

    def __call__(self, q, k, v, heads, mask=None):
        self.calls += 1
        self.last_args = (q, k, v, heads, mask)
        return self._inner(q, k, v, heads, mask)


def _build_block(video_d_head: int, audio_d_head: int, heads: int = 2, dim: int = 32):
    """One real BasicAVTransformerBlock in the production configuration
    (gated attention + cross-attention AdaLN), with per-stream head dims chosen
    by the caller so the install-time static filter can be exercised.

    ``attention_function`` is stateless (AttentionFunction.PYTORCH is an enum
    member, not a tensor holder), so the block can stay on CPU even when the
    tensors handed to it live on the GPU.
    """
    from ltx_core.model.transformer.attention import AttentionFunction
    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig

    torch.manual_seed(42)
    block = BasicAVTransformerBlock(
        idx=0,
        video=TransformerConfig(
            dim=dim,
            heads=heads,
            d_head=video_d_head,
            context_dim=dim,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        ),
        audio=TransformerConfig(
            dim=dim,
            heads=heads,
            d_head=audio_d_head,
            context_dim=dim,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        ),
        attention_function=AttentionFunction.PYTORCH,
    )
    block.eval()
    return block


def _install_on(block: torch.nn.Module, backend: str = "sage") -> tuple[int, SageState]:
    state = SageState()
    state.set_backend(backend)
    count = SageAttentionService(lambda: state).install(_FakeTransformer([block]))
    return count, state


def _qkv(batch: int, tokens: int, heads: int, dim_head: int, dtype, device):
    torch.manual_seed(1234)
    shape = (batch, tokens, heads * dim_head)
    return (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn(shape, dtype=dtype, device=device),
    )


# --------------------------------------------------------------------------- #
# Check 1: numerical parity against the wheel's own PytorchAttention           #
# --------------------------------------------------------------------------- #


def check_numerical_parity() -> None:
    if not torch.cuda.is_available():
        raise AssertionError("no CUDA device: the sage kernel cannot be exercised")
    if not probe_sage():
        raise AssertionError(
            "probe_sage() is False — sageattention is missing or unusable in "
            "this venv (see the [sage] line on STDERR above)"
        )
    device = torch.device("cuda:0")

    # (module attribute, head dim, tokens): the video streams run d_head=128 and
    # the audio streams d_head=64 on the production model.
    cases = [("attn1", 128, 1024), ("audio_attn1", 64, 256)]
    for dtype in (torch.bfloat16, torch.float16):
        for attr, dim_head, tokens in cases:
            block = _build_block(video_d_head=128, audio_d_head=64)
            attn = getattr(block, attr)
            spy = _Spy()
            attn.attention_function = spy  # becomes the wrapper's fallback

            count, _state = _install_on(block)
            if count != 6:
                raise AssertionError(f"expected 6 wrapped modules, got {count}")

            q, k, v = _qkv(1, tokens, attn.heads, dim_head, dtype, device)
            got = attn.attention_function(q, k, v, attn.heads, None)
            if spy.calls != 0:
                raise AssertionError(
                    f"{attr}/{dtype}: the call fell back to sdpa ({spy.calls} "
                    "fallback calls) — parity would be vacuously true"
                )

            ref = spy(q, k, v, attn.heads, None)
            if got.shape != ref.shape:
                raise AssertionError(f"{attr}/{dtype}: shape {got.shape} != {ref.shape}")
            if got.dtype != ref.dtype:
                raise AssertionError(f"{attr}/{dtype}: dtype {got.dtype} != {ref.dtype}")
            if not torch.isfinite(got.float()).all():
                raise AssertionError(f"{attr}/{dtype}: sage output contains NaN/Inf")

            got_f, ref_f = got.float(), ref.float()
            rel = float((got_f - ref_f).norm() / ref_f.norm().clamp_min(1e-12))
            cos = float(
                torch.nn.functional.cosine_similarity(
                    got_f.flatten(), ref_f.flatten(), dim=0
                )
            )
            label = f"{attr}/{dtype} d_head={dim_head} tokens={tokens}"
            if rel > _REL_TOL:
                raise AssertionError(f"{label}: relative L2 error {rel:.4f} exceeds {_REL_TOL}")
            if cos < _COS_TOL:
                raise AssertionError(f"{label}: cosine similarity {cos:.6f} below {_COS_TOL}")
            print(
                f"       {attr:<20} {str(dtype):<16} d_head={dim_head:<4} "
                f"rel_l2={rel:.5f} cos={cos:.6f}"
            )


# --------------------------------------------------------------------------- #
# Check 2: the fallback matrix                                                 #
# --------------------------------------------------------------------------- #


def _assert_fell_back(label: str, attn, spy: _Spy, q, k, v, mask=None) -> None:
    """The wrapped call must equal the original callable's answer BIT for bit,
    and must have reached it by actually calling it (not by coincidence)."""
    before = spy.calls
    got = attn.attention_function(q, k, v, attn.heads, mask)
    if spy.calls != before + 1:
        raise AssertionError(f"{label}: fallback was not called ({spy.calls - before} calls)")
    fq, fk, fv, _fh, fm = spy.last_args
    if fq is not q or fk is not k or fv is not v or fm is not mask:
        raise AssertionError(
            f"{label}: the fallback received rebound tensors instead of the "
            "untouched originals (a reshape leaked into the sdpa path)"
        )
    want = spy(q, k, v, attn.heads, mask)
    if not torch.equal(got, want):
        raise AssertionError(f"{label}: fallback result is not bit-identical to sdpa")


def check_fallback_matrix() -> None:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # --- rows 1-3: dynamic, per-call conditions --------------------------- #
    block = _build_block(video_d_head=128, audio_d_head=64)
    attn = block.attn1
    spy = _Spy()
    attn.attention_function = spy
    count, state = _install_on(block)
    if count != 6:
        raise AssertionError(f"expected 6 wrapped modules, got {count}")

    # (1) attention mask present — the only producer on this pipeline is the
    #     IC-LoRA ConditioningItemAttentionStrengthWrapper (strength < 1.0).
    #     This is a PARTIAL fallback (attention_used stays "sage"), so it is the
    #     one degradation with no other trace: assert it is counted per call but
    #     logged exactly ONCE however many calls take it.
    q, k, v = _qkv(1, 64, attn.heads, 128, torch.bfloat16, device)
    mask = torch.zeros(1, attn.heads, 64, 64, dtype=torch.bfloat16, device=device)
    with _LogCapture() as logs:
        _assert_fell_back("mask is not None (call 1)", attn, spy, q, k, v, mask)
        _assert_fell_back("mask is not None (call 2)", attn, spy, q, k, v, mask)
    if state.masked_calls != 2:
        raise AssertionError(
            f"masked_calls={state.masked_calls} after 2 masked calls, expected 2"
        )
    masked_logs = [m for m in logs.messages if "attention mask present" in m]
    if len(masked_logs) != 1:
        raise AssertionError(
            f"expected exactly 1 masked-fallback log line for 2 masked calls, "
            f"got {len(masked_logs)}: {masked_logs}"
        )

    # (2) unsupported dtype (sage quantizes from fp16/bf16 only).
    q32, k32, v32 = _qkv(1, 64, attn.heads, 128, torch.float32, device)
    _assert_fell_back("dtype=float32", attn, spy, q32, k32, v32)

    # (3) non-CUDA tensors (bf16, i.e. a dtype sage WOULD accept — so this row
    #     can only be passing because of the device check).
    qc, kc, vc = _qkv(1, 64, attn.heads, 128, torch.bfloat16, torch.device("cpu"))
    _assert_fell_back("cpu tensors", attn, spy, qc, kc, vc)

    if state.latched:
        raise AssertionError("a per-call fallback must NOT latch the whole job")
    if state.attention_used != "sage":
        raise AssertionError(
            f"per-call fallbacks changed attention_used to {state.attention_used!r}"
        )

    # Leak guard: the transformer is rebuilt per job, so a wrapper should never
    # outlive the job that installed it — but if one ever did, it must be inert
    # rather than silently accelerating (and mis-reporting) an sdpa job.
    qb, kb, vb = _qkv(1, 64, attn.heads, 128, torch.bfloat16, device)
    state.reset()  # what the pipeline's finally does at the end of the job
    if state.masked_calls != 0:
        raise AssertionError(
            f"reset() left masked_calls={state.masked_calls}; the counter is "
            "per-job and must not carry into the next job on a resident worker"
        )
    _assert_fell_back("stale wrapper after state.reset()", attn, spy, qb, kb, vb)
    if state.last_attention_used != "sage":
        raise AssertionError(
            f"reset() snapshotted attention_used={state.last_attention_used!r}, expected 'sage'"
        )

    # --- row 4: a raising kernel latches for the rest of the job ----------- #
    # The module-level sage entry point is swapped for a raising stub so the
    # catastrophic path is exercised without needing a kernel that actually
    # misbehaves. Restored in the finally below.
    raising_calls = {"n": 0}

    def _raising(_q, _k, _v):
        raising_calls["n"] += 1
        raise RuntimeError("simulated kernel failure")

    saved = sage_mod._SAGE_CALL
    try:
        sage_mod._SAGE_CALL = _raising
        block2 = _build_block(video_d_head=128, audio_d_head=64)
        attn2 = block2.attn1
        spy2 = _Spy()
        attn2.attention_function = spy2
        _count2, state2 = _install_on(block2)

        q2, k2, v2 = _qkv(1, 64, attn2.heads, 128, torch.bfloat16, device)
        _assert_fell_back("kernel raises (1st call)", attn2, spy2, q2, k2, v2)
        if not state2.latched:
            raise AssertionError("a kernel exception did not latch the job onto sdpa")
        if state2.attention_used != "sage->sdpa":
            raise AssertionError(
                f"latched job reports attention_used={state2.attention_used!r}"
            )
        _assert_fell_back("kernel raises (2nd call, latched)", attn2, spy2, q2, k2, v2)
        if raising_calls["n"] != 1:
            raise AssertionError(
                f"the failing kernel was attempted {raising_calls['n']} times; the "
                "latch must stop it after the first"
            )
        # A later transformer build in the SAME job (chain path) must not re-wrap.
        block3 = _build_block(video_d_head=128, audio_d_head=64)
        orig3 = block3.attn1.attention_function
        if SageAttentionService(lambda: state2).install(_FakeTransformer([block3])) != 0:
            raise AssertionError("install() re-wrapped modules after the job latched")
        if block3.attn1.attention_function is not orig3:
            raise AssertionError("install() mutated attention_function after latching")
    finally:
        sage_mod._SAGE_CALL = saved

    # --- row 5: install-time static skip of an unsupported head_dim -------- #
    # Video streams at 128 (supported), audio streams at 8 (not) -> only the two
    # video modules are wrapped; the four audio-dimensioned ones (incl. both
    # audio<->video cross-attentions, which use the AUDIO head dim) are left
    # completely alone rather than paying a per-call check.
    block4 = _build_block(video_d_head=128, audio_d_head=8)
    originals = {
        name: getattr(block4, name).attention_function for name in _EXPECTED_KINDS
    }
    count4, _state4 = _install_on(block4)
    if count4 != 2:
        raise AssertionError(f"expected 2 wrapped modules (video only), got {count4}")
    for name in ("audio_attn1", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"):
        if getattr(block4, name).attention_function is not originals[name]:
            raise AssertionError(f"{name} (head_dim=8) was wrapped despite being unsupported")
    for name in ("attn1", "attn2"):
        if getattr(block4, name).attention_function is originals[name]:
            raise AssertionError(f"{name} (head_dim=128) was NOT wrapped")


# --------------------------------------------------------------------------- #
# Check 3: install count on a 48-block transformer, and the OFF no-op          #
# --------------------------------------------------------------------------- #


def check_install_count_288() -> None:
    blocks = [_build_block(video_d_head=128, audio_d_head=64) for _ in range(48)]
    transformer = _FakeTransformer(blocks)

    # --- OFF: a job that never asked for sage must touch nothing at all ---- #
    originals = {
        id(getattr(b, name)): getattr(b, name).attention_function
        for b in blocks
        for name in _EXPECTED_KINDS
    }
    off_state = SageState()  # defaults to "sdpa"
    n_off = SageAttentionService(lambda: off_state).install(transformer)
    if n_off != 0:
        raise AssertionError(f"install() with sage not requested returned {n_off}, expected 0")
    for b in blocks:
        for name in _EXPECTED_KINDS:
            module = getattr(b, name)
            if module.attention_function is not originals[id(module)]:
                raise AssertionError(f"install() with sage OFF mutated {name}.attention_function")

    # --- ON: all 48 x 6 modules, and exactly those ------------------------- #
    on_state = SageState()
    on_state.set_backend("sage")
    n_on = SageAttentionService(lambda: on_state).install(transformer)
    if n_on != 288:
        raise AssertionError(f"install() wrapped {n_on} modules, expected 288 (48 blocks x 6)")

    kinds = {
        full_name.rsplit(".", 1)[-1]
        for full_name, _module in sage_mod._attention_modules(transformer)
    }
    if kinds != set(_EXPECTED_KINDS):
        raise AssertionError(
            f"traversal found attention kinds {sorted(kinds)}, expected {sorted(_EXPECTED_KINDS)}"
        )


def main() -> int:
    # Windows consoles are frequently cp932/cp1252 — a stray non-ASCII character
    # in an exception message must never crash the reporter itself.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    torch.manual_seed(0)
    checks: list[tuple[str, Callable[[], None]]] = [
        ("numerical parity vs PytorchAttention (d_head 64+128, bf16+fp16)", check_numerical_parity),
        ("fallback matrix (mask / dtype / device / stale / kernel error+latch / static skip)", check_fallback_matrix),
        ("install wraps exactly 288 modules on 48 blocks, and 0 when OFF", check_install_count_288),
    ]

    for name, fn in checks:
        _run_check(name, fn)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_total = len(_RESULTS)
    print(f"\n{n_pass}/{n_total} checks passed")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
