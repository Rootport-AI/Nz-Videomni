"""Standalone numerical self-check for engine.transformer.nag_service.

Run with the ENGINE venv (needs the real ltx_core wheel, not just torch):

    .venv-engine\\Scripts\\python.exe -m engine.transformer.nag_selfcheck

This is deliberately NOT a pytest test module (no `test_` prefix, not under
tests/): it is meant to run against .venv-engine, which the app-venv pytest
suite (.venv) never touches, and it exercises a real `BasicAVTransformerBlock`
from the ltx_core wheel — something the app venv cannot even import.

Every check below either PASSes or FAILs loudly (raises, gets caught, and is
reported as FAIL with a traceback) — nothing is silently skipped. Exit code
is 0 only if every check reported PASS.

What is checked, and why:
  1. `nag_combine` numerically matches a KJNodes-faithful reference formula
     (nan_to_num/where instead of clamp+eps, native-dtype norms) on bf16 AND
     fp32 tensors, across shapes, and across parameter sets chosen to hit
     BOTH branches of the tau clamp (engaged and not engaged).
  2. The alpha=0.0 / scale=1.0 identity short-circuits return the exact same
     z_pos object (not just an equal-valued copy) — this is what makes NAG's
     "OFF" state bit-identical to the unpatched path (real-device gate G2).
  3. Degenerate zero-norm inputs (z_pos all zeros, z_neg all zeros, and a
     constructed case where the extrapolated z_g is exactly zero) never
     produce NaN/Inf, thanks to the epsilon terms in nag_combine.
  4. A REAL `BasicAVTransformerBlock` (cross_attention_adaln=True,
     apply_gated_attention=True — the production GGUF configuration) is built
     on CPU. Against its actual attn2 module we verify: (a) NagService.install
     with NAG not requested patches nothing and returns 0; (b) with NAG
     requested, the patched forward's output matches an independently
     hand-written reference built directly from attn2's own submodules
     (q/k/v projections, norms, attention_function, gate, to_out) — this is
     NOT a call into nag_service's own closure, so it actually exercises
     equivalence rather than tautology, and by construction proves the NAG
     combine happens BEFORE per-head gating; (c) the patched forward raises
     RuntimeError for call shapes (pe set, context=None) that should never
     occur on this pipeline's text cross-attention.
  5. `NagService.install()` against a minimal fake transformer object
     (`namespace.velocity_model.transformer_blocks = [...]`) with NAG not
     requested returns 0 and leaves every module's `forward` attribute
     untouched (identity-checked, not just behaviourally unchanged).

A bonus check (6) is included beyond the plan's five: `install()` raises
RuntimeError when NAG is requested but the negative context was never
encoded (`NagState.ready is False`) — the fail-loud path D2 relies on.
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable

import torch

from engine.transformer.nag_service import (
    NagParams,
    NagService,
    NagState,
    nag_combine,
)

_RESULTS: list[tuple[str, bool, str]] = []


class _FakeVelocityModel:
    def __init__(self, blocks: list) -> None:
        self.transformer_blocks = blocks


class _FakeTransformer:
    """Minimal stand-in for the X0Model wrapper _cross_attn_modules expects."""

    def __init__(self, blocks: list) -> None:
        self.velocity_model = _FakeVelocityModel(blocks)


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


# --------------------------------------------------------------------------- #
# Check 1: nag_combine vs a KJNodes-faithful reference                         #
# --------------------------------------------------------------------------- #


def _kjnodes_reference(
    z_pos: torch.Tensor, z_neg: torch.Tensor, scale: float, tau: float, alpha: float
) -> torch.Tensor:
    """Reference written directly from KJNodes' LTX2_NAG forward math (native-
    dtype L1 norms, nan_to_num + where instead of nag_combine's clamp+eps-on-
    both-sides), independent of nag_service's own implementation."""
    if alpha == 0.0 or scale == 1.0:
        return z_pos
    z_g = z_pos * scale - z_neg * (scale - 1.0)
    norm_pos = torch.norm(z_pos, p=1, dim=-1, keepdim=True)
    norm_g = torch.norm(z_g, p=1, dim=-1, keepdim=True)
    ratio = torch.nan_to_num(norm_g / norm_pos, nan=10.0)
    mask = ratio > tau
    adjustment = (norm_pos * tau) / (norm_g + 1e-7)
    factor = torch.where(mask, adjustment, torch.ones_like(adjustment))
    z_g_normalized = z_g * factor
    return z_g_normalized * alpha + z_pos * (1.0 - alpha)


def check_combine_matches_kjnodes_reference() -> None:
    shapes = [
        (1, 77, 4096),  # production-scale token/dim count
        (2, 8, 64),
        (3, 5, 16),
        (1, 1, 1),  # degenerate single-token/single-channel
    ]
    # (scale, tau, alpha): one production-default, one tuned to make the tau
    # clamp engage almost everywhere (tau=1.0), one tuned so it almost never
    # engages (tau=10.0) — exercises both branches of the where/clamp.
    param_sets = [
        (11.0, 2.5, 0.25),
        (20.0, 1.0, 1.0),
        (1.5, 10.0, 0.1),
    ]
    for dtype in (torch.float32, torch.bfloat16):
        for shape in shapes:
            for scale, tau, alpha in param_sets:
                torch.manual_seed(hash((shape, scale, tau, alpha)) % (2**31))
                z_pos = torch.randn(shape, dtype=dtype)
                z_neg = torch.randn(shape, dtype=dtype)
                got = nag_combine(z_pos.clone(), z_neg.clone(), scale, tau, alpha)
                want = _kjnodes_reference(z_pos, z_neg, scale, tau, alpha)
                rtol = 1e-3
                # bf16 has ~3 significant decimal digits; nag_combine
                # accumulates its L1 norms in fp32 (see _NORM_DTYPE) while the
                # reference above uses native bf16 norms (matching KJNodes),
                # so the two legitimately diverge by more than bf16 ULP alone
                # — this atol was set empirically from the observed max diff
                # at the most aggressive parameter set (scale=20, tau=1.0)
                # over a (1, 77, 4096) tensor, with headroom.
                atol = 8e-2 if dtype is torch.bfloat16 else 1e-5
                if not torch.allclose(got.float(), want.float(), rtol=rtol, atol=atol):
                    max_diff = (got.float() - want.float()).abs().max().item()
                    raise AssertionError(
                        f"dtype={dtype} shape={shape} scale={scale} tau={tau} "
                        f"alpha={alpha}: max_abs_diff={max_diff} exceeds "
                        f"rtol={rtol}/atol={atol}"
                    )


# --------------------------------------------------------------------------- #
# Check 2: identity short-circuit                                              #
# --------------------------------------------------------------------------- #


def check_identity_shortcircuit() -> None:
    torch.manual_seed(7)
    z_pos = torch.randn(2, 4, 16)
    z_neg = torch.randn(2, 4, 16)

    out_alpha0 = nag_combine(z_pos, z_neg, scale=11.0, tau=2.5, alpha=0.0)
    if out_alpha0 is not z_pos:
        raise AssertionError("alpha=0.0 did not return the identical z_pos object")

    out_scale1 = nag_combine(z_pos, z_neg, scale=1.0, tau=2.5, alpha=0.25)
    if out_scale1 is not z_pos:
        raise AssertionError("scale=1.0 did not return the identical z_pos object")


# --------------------------------------------------------------------------- #
# Check 3: zero-norm degenerate inputs                                         #
# --------------------------------------------------------------------------- #


def check_zero_norm_degenerate() -> None:
    torch.manual_seed(3)
    zeros = torch.zeros(2, 3, 8)
    nonzero = torch.randn(2, 3, 8)
    # scale=11 => z_neg = nonzero * scale/(scale-1) makes z_g == 0 EXACTLY:
    # z_g = scale*z_pos - (scale-1)*z_neg = scale*nonzero - (scale-1)*(scale/(scale-1))*nonzero = 0
    z_g_zero_neg = nonzero * (11.0 / 10.0)

    cases = [
        ("z_pos=0, z_neg=0", zeros, zeros.clone()),
        ("z_pos=0, z_neg=random", zeros, nonzero),
        ("z_g extrapolates to exactly 0", nonzero, z_g_zero_neg),
    ]
    for label, z_pos, z_neg in cases:
        out = nag_combine(z_pos.clone(), z_neg.clone(), scale=11.0, tau=2.5, alpha=0.25)
        if not torch.isfinite(out).all():
            raise AssertionError(f"NaN/Inf produced for degenerate case: {label}")


# --------------------------------------------------------------------------- #
# Check 4: real BasicAVTransformerBlock, AdaLN asymmetry, fail-loud shapes      #
# --------------------------------------------------------------------------- #


def _build_test_block():
    from ltx_core.model.transformer.attention import AttentionFunction
    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig

    cfg = TransformerConfig(
        dim=16,
        heads=2,
        d_head=8,
        context_dim=16,
        apply_gated_attention=True,
        cross_attention_adaln=True,
    )
    torch.manual_seed(42)
    block = BasicAVTransformerBlock(
        idx=0,
        video=cfg,
        audio=cfg,
        attention_function=AttentionFunction.PYTORCH,
    )
    block.eval()
    return block


def _manual_cross_attn(attn: torch.nn.Module, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Independent (non-nag_service) reproduction of Attention.forward's
    cross-attention math, built directly from attn's own submodules."""
    q = attn.q_norm(attn.to_q(x))
    k = attn.k_norm(attn.to_k(context))
    v = attn.to_v(context)
    return attn.attention_function(q, k, v, attn.heads, None)


def _manual_gate_and_out(attn: torch.nn.Module, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    gate_logits = attn.to_gate_logits(x)
    b, t, _ = out.shape
    out = out.view(b, t, attn.heads, attn.dim_head)
    gates = 2.0 * torch.sigmoid(gate_logits)
    out = out * gates.unsqueeze(-1)
    out = out.view(b, t, attn.heads * attn.dim_head)
    return attn.to_out(out)


def check_real_block_off_then_on() -> None:
    block = _build_test_block()
    attn2 = block.attn2

    batch, tokens, ctx_tokens = 2, 5, 7
    torch.manual_seed(99)
    x = torch.randn(batch, tokens, 16)
    pos_ctx_raw = torch.randn(batch, ctx_tokens, 16)
    neg_ctx_raw = torch.randn(1, ctx_tokens, 16)

    # Simulate the AdaLN modulation the production path applies to the
    # POSITIVE context before it reaches attn2/audio_attn2
    # (apply_cross_attention_adaln, transformer.py:391):
    #   encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    # The NAG negative context is intentionally never modulated this way
    # (D1/D4's deliberate asymmetry) — it stays raw below.
    scale_kv = torch.randn(batch, 1, 16) * 0.1
    shift_kv = torch.randn(batch, 1, 16) * 0.1
    pos_ctx_modulated = pos_ctx_raw * (1 + scale_kv) + shift_kv

    # --- (a) NAG OFF: install() must patch nothing ---
    off_state = NagState()  # requested is False: set_params was never called
    off_service = NagService(state_provider=lambda: off_state)

    orig_attn2_forward = attn2.forward
    orig_audio_attn2_forward = block.audio_attn2.forward

    n_off = off_service.install(_FakeTransformer([block]))
    if n_off != 0:
        raise AssertionError(f"install() with NAG not requested returned {n_off}, expected 0")
    if attn2.forward != orig_attn2_forward:
        raise AssertionError("install() with NAG not requested mutated attn2.forward")
    if block.audio_attn2.forward != orig_audio_attn2_forward:
        raise AssertionError("install() with NAG not requested mutated audio_attn2.forward")

    # --- (b) NAG ON: patched output must match an independent reference ---
    params = NagParams(negative_prompt="blurry, low quality", scale=11.0, tau=2.5, alpha=0.25)
    on_state = NagState()
    on_state.set_params(params)
    on_state.set_contexts(neg_ctx_raw, neg_ctx_raw.clone())
    on_service = NagService(state_provider=lambda: on_state)

    n_on = on_service.install(_FakeTransformer([block]))
    if n_on != 2:
        raise AssertionError(f"install() with NAG requested patched {n_on} modules, expected 2 (attn2+audio_attn2)")

    patched_out = attn2.forward(x, context=pos_ctx_modulated)

    # Independently hand-built reference: proves by construction that the
    # combine happens BEFORE gating (gate/to_out are applied to the ALREADY-
    # combined tensor here, never to z_pos or z_neg individually).
    z_pos_ref = _manual_cross_attn(attn2, x, pos_ctx_modulated)
    # Batch-expand neg context ourselves here too — the patched forward does
    # this internally (neg_ctx_raw is batch=1, x/pos_ctx are batch=2) so the
    # reference must replicate it to stay comparable.
    z_neg_ref = _manual_cross_attn(attn2, x, neg_ctx_raw.expand(batch, -1, -1))
    combined_ref = nag_combine(z_pos_ref, z_neg_ref, params.scale, params.tau, params.alpha)
    expected = _manual_gate_and_out(attn2, x, combined_ref)

    if not torch.allclose(patched_out, expected, rtol=1e-5, atol=1e-6):
        max_diff = (patched_out - expected).abs().max().item()
        raise AssertionError(f"patched attn2 output != hand-written reference, max_abs_diff={max_diff}")

    # --- (c) fail-loud on call shapes that should never occur ---
    try:
        attn2.forward(x, context=pos_ctx_modulated, pe=torch.zeros(1))
    except RuntimeError:
        pass
    else:
        raise AssertionError("patched forward did not raise RuntimeError for pe != None")

    try:
        attn2.forward(x, context=None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("patched forward did not raise RuntimeError for context=None")


# --------------------------------------------------------------------------- #
# Check 5: install() on a minimal fake transformer, NAG not requested          #
# --------------------------------------------------------------------------- #


def check_install_off_on_fake_transformer() -> None:
    block = _build_test_block()
    fake_transformer = _FakeTransformer([block])

    state = NagState()
    service = NagService(state_provider=lambda: state)

    orig_attn2_forward = block.attn2.forward
    orig_audio_attn2_forward = block.audio_attn2.forward

    n = service.install(fake_transformer)
    if n != 0:
        raise AssertionError(f"install() on fake transformer with NAG off returned {n}, expected 0")
    if block.attn2.forward != orig_attn2_forward:
        raise AssertionError("install() with NAG off mutated attn2.forward")
    if block.audio_attn2.forward != orig_audio_attn2_forward:
        raise AssertionError("install() with NAG off mutated audio_attn2.forward")


# --------------------------------------------------------------------------- #
# Bonus check 6: fail-loud when requested but negative context never encoded  #
# --------------------------------------------------------------------------- #


def check_install_raises_when_requested_not_ready() -> None:
    block = _build_test_block()
    state = NagState()
    state.set_params(NagParams(negative_prompt="x", scale=11.0, tau=2.5, alpha=0.25))
    # set_contexts() deliberately never called: state.requested is True but
    # state.ready is False.
    service = NagService(state_provider=lambda: state)

    try:
        service.install(_FakeTransformer([block]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("install() did not raise RuntimeError when requested but not ready")


def main() -> int:
    # Windows consoles are frequently cp932/cp1252, not UTF-8 — a stray
    # non-ASCII character in an exception message must never crash the
    # reporter itself (that would defeat "fail loudly, never skip").
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    torch.manual_seed(0)
    checks: list[tuple[str, Callable[[], None]]] = [
        ("nag_combine matches KJNodes reference (bf16+fp32, both clamp branches)", check_combine_matches_kjnodes_reference),
        ("identity short-circuit (alpha=0 / scale=1) is the same z_pos object", check_identity_shortcircuit),
        ("zero-norm degenerate inputs produce no NaN/Inf", check_zero_norm_degenerate),
        ("real BasicAVTransformerBlock: OFF no-op, ON matches reference, fail-loud shapes", check_real_block_off_then_on),
        ("NagService.install() on fake transformer with NAG off is a no-op", check_install_off_on_fake_transformer),
        ("NagService.install() raises when requested but negative context not encoded", check_install_raises_when_requested_not_ready),
    ]

    for name, fn in checks:
        _run_check(name, fn)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_total = len(_RESULTS)
    print(f"\n{n_pass}/{n_total} checks passed")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
