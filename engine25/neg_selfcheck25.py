"""Standalone self-check for the LTX 2.5 negative-prompt patch (NAG / VSF).

Run with the 2.5 ENGINE venv (needs the official ltx_core 1.2.0 wheel; no GPU
and no model weights are required -- every check below runs on CPU against a
tiny real ``BasicAVTransformerBlock``):

    .venv-engine-ltx25\\Scripts\\python.exe -m engine25.neg_selfcheck25

Same conventions as :mod:`engine25.sage_selfcheck25` and
:mod:`engine.transformer.nag_selfcheck`: NOT a pytest module (it needs a venv the
app-venv suite never touches), every check either PASSes or FAILs loudly and
nothing is skipped, exit code 0 only when all of them pass.

WHAT IS SHARED AND WHAT IS NEW
------------------------------
The ALGEBRA is 2.3's -- ``nag_combine`` is imported, not reimplemented -- so the
three checks that judge the FORMULA are 2.3's own, imported from
:mod:`engine.transformer.nag_selfcheck` and re-run here rather than copied. A
second copy could only drift.

The eight checks below have no 2.3 counterpart, because they are about the two
things this engine does differently: the replacement ``forward`` is written
against 2.5's ``Attention`` (one ``preattention_function`` call over
CONCATENATED keys, a four-argument ``attention_function``, the official
``gated_attention_function`` slot), and the model SHELL is reused across jobs so
the patch has an uninstall.

**THE REAL-MODULE CHECKS CALL ``attn2`` DIRECTLY, AND MUST KEEP DOING SO.**
There is a tempting shortcut -- drive a whole block through
``engine25.gguf_transformer``'s selftest ``_dummy_video_modality`` -- and it is
WRONG here: that dummy carries ``context_mask=torch.ones(...)``, so the block
sends attn2 down ``Attention.forward``'s MASKED route with a non-None ``mask``,
which this patch refuses by design (the fail-loud guard, check 11). Every check
that exercises a patched forward therefore builds its own ``x`` / ``context``
tensors and calls ``attn2`` itself, which is also what makes the reference
comparison non-tautological: the expected value is assembled from attn2's OWN
submodules, never from :mod:`engine25.neg_prompt25`.
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable

import torch

from engine.transformer import nag_selfcheck as base
from engine25.neg_prompt25 import (
    NagParams,
    NagState,
    NegPromptService,
    VsfParams,
    nag_combine,
)


def _record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    base._RESULTS.append((name, passed, detail))


def _run_check(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        _record(name, True)
    except Exception as exc:  # noqa: BLE001 -- a broken check must FAIL loudly, never skip
        _record(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


class _FakeVelocityModel(torch.nn.Module):
    def __init__(self, blocks: list[torch.nn.Module]) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(blocks)


class _FakeTransformer(torch.nn.Module):
    """The X0Model shape ``_cross_attn_modules`` expects in production.

    The traversal looks for ``velocity_model.transformer_blocks`` first, which
    is exactly what ``X0Model(LTXModel(...))`` presents -- so the fixture uses
    that shape rather than the flat one, and the production traversal is what is
    exercised.
    """

    def __init__(self, blocks: list[torch.nn.Module]) -> None:
        super().__init__()
        self.velocity_model = _FakeVelocityModel(blocks)


def _build_block(dim: int = 32, heads: int = 2, d_head: int = 16):
    """2.5's ``BasicAVTransformerBlock`` in the PRODUCTION configuration.

    ``apply_gated_attention`` and ``cross_attention_adaln`` are both True
    because all three production GGUF configs set them, and both matter here:
    gating is the step the NAG combine has to happen BEFORE, and the AdaLN flag
    is what makes the positive context arrive modulated while the negative one
    stays raw (the deliberate asymmetry this patch inherits from 2.3).

    Both attention slots are pinned to PYTORCH rather than AUTOMATIC so the
    reference comparison cannot pass for an uninteresting reason on a machine
    where AUTOMATIC happened to resolve elsewhere.
    """
    from ltx_core.model.transformer.attention import AttentionFunction, MaskedAttentionFunction
    from ltx_core.model.transformer.transformer import (
        BasicAVTransformerBlock,
        TransformerConfig,
        TransformerOpsConfig,
    )

    torch.manual_seed(42)
    cfg = TransformerConfig(
        dim=dim,
        heads=heads,
        d_head=d_head,
        context_dim=dim,
        apply_gated_attention=True,
        cross_attention_adaln=True,
    )
    block = BasicAVTransformerBlock(
        video=cfg,
        audio=cfg,
        ops=TransformerOpsConfig.from_functions(
            attention=AttentionFunction.PYTORCH,
            masked_attention=MaskedAttentionFunction.PYTORCH,
        ),
    )
    block.eval()
    return block


def _tensors(dim: int = 32, batch: int = 2, tokens: int = 5, ctx: int = 7, neg: int = 3):
    """``x``, an AdaLN-MODULATED positive context, and a raw batch-1 negative one.

    The modulation mirrors ``apply_cross_attention_adaln``'s
    ``context * (1 + scale_kv) + shift_kv``: the positive context attn2 really
    receives has been through it, the negative context in :class:`NagState`
    never is. The negative tensor is batch 1 and SHORTER, which is the
    production shape (encoded once, and sliced to its real tokens for VSF).
    """
    torch.manual_seed(99)
    x = torch.randn(batch, tokens, dim)
    pos_raw = torch.randn(batch, ctx, dim)
    scale_kv = torch.randn(batch, 1, dim) * 0.1
    shift_kv = torch.randn(batch, 1, dim) * 0.1
    pos = pos_raw * (1 + scale_kv) + shift_kv
    neg_ctx = torch.randn(1, neg, dim)
    return x, pos, neg_ctx


def _nag_state(**over) -> NagState:
    state = NagState()
    params = NagParams(
        negative_prompt=over.get("negative_prompt", "blurry, low quality"),
        scale=over.get("scale", 11.0),
        tau=over.get("tau", 2.5),
        alpha=over.get("alpha", 0.25),
    )
    state.set_params(params)
    return state


def _vsf_state(scale: float = 1.5) -> NagState:
    state = NagState()
    state.set_params(VsfParams(negative_prompt="blurry, low quality", scale=scale))
    return state


def _ref_cross_attn(attn, x, context):
    """One UNPATCHED cross-attention output, assembled from attn's own parts.

    Deliberately NOT a call into :mod:`engine25.neg_prompt25` and not a call
    into ``Attention.forward`` either: the four lines below are what the patched
    forward has to be equal to, written out here so the comparison tests the
    arithmetic rather than restating it.
    """
    q = attn.to_q(x)
    k = attn.to_k(context)
    q, k = attn.preattention_function(q, k, attn, None, None, None)
    v = attn.to_v(context)
    return attn.attention_function(q, k, v, attn.heads)


def _ref_gate_and_out(attn, x, out):
    if attn.to_gate_logits is not None:
        out = attn.gated_attention_function(x, out, attn)
    return attn.to_out(out)


# --------------------------------------------------------------------------- #
# Check 4: one preattention over concatenated keys == two separate calls       #
# --------------------------------------------------------------------------- #


def check_concat_preattention_equals_split() -> None:
    """The equivalence the whole single-preattention shortcut rests on.

    ``preattention_function`` normalises ``q`` and ``k`` with RMSNorm, which is
    per-row over the LAST dimension, so concatenating two key halves along the
    SEQUENCE axis, normalising once and splitting must equal normalising each
    half on its own. Checked bit-exactly, on contiguous AND non-contiguous
    splits and at two batch sizes, because a future norm with cross-position
    statistics would break this silently rather than loudly.
    """
    block = _build_block()
    attn = block.attn2
    for batch, n_pos, n_neg in ((1, 7, 3), (2, 7, 3), (2, 8, 8)):
        torch.manual_seed(batch * 100 + n_pos)
        k_pos_raw = torch.randn(batch, n_pos, attn.heads * attn.dim_head)
        k_neg_raw = torch.randn(batch, n_neg, attn.heads * attn.dim_head)
        q_raw = torch.randn(batch, 5, attn.heads * attn.dim_head)

        q_once, k_cat = attn.preattention_function(
            q_raw, torch.cat([k_pos_raw, k_neg_raw], dim=1), attn, None, None, None
        )
        got_pos, got_neg = k_cat[:, :n_pos], k_cat[:, n_pos:]

        q_a, want_pos = attn.preattention_function(q_raw, k_pos_raw, attn, None, None, None)
        _q_b, want_neg = attn.preattention_function(q_raw, k_neg_raw, attn, None, None, None)

        if not torch.equal(q_once, q_a):
            raise AssertionError(f"q differs between the one-call and two-call forms (batch={batch})")
        if not torch.equal(got_pos, want_pos):
            raise AssertionError(f"positive keys differ after the split (batch={batch})")
        if not torch.equal(got_neg, want_neg):
            raise AssertionError(f"negative keys differ after the split (batch={batch})")
        # ...and once more on a CONTIGUOUS copy of each half, so the result is
        # not an artefact of the slices being views into one storage.
        if not torch.equal(got_pos.contiguous(), want_pos.contiguous()):
            raise AssertionError(f"positive keys differ once made contiguous (batch={batch})")


# --------------------------------------------------------------------------- #
# Check 5: the patched NAG forward vs an independent reference                 #
# --------------------------------------------------------------------------- #


def check_nag_forward_matches_reference() -> None:
    """The patched attn2 output equals a reference built from attn2's own parts.

    By CONSTRUCTION this also proves the ordering the plan requires: the
    reference gates the ALREADY-COMBINED tensor, never z_pos or z_neg
    individually, so a patch that gated before combining could not match it.

    attn2 is called DIRECTLY (see the module docstring on why the block-level
    dummy modality is not usable here).
    """
    block = _build_block()
    attn = block.attn2
    x, pos, neg = _tensors()

    state = _nag_state()
    state.set_contexts(neg, neg.clone())
    service = NegPromptService(state_provider=lambda: state)
    n = service.install(_FakeTransformer([block]))
    if n != 2:
        raise AssertionError(f"install patched {n} modules on one block, expected 2")

    got = attn.forward(x, context=pos)

    neg_expanded = neg.expand(x.shape[0], -1, -1)
    z_pos = _ref_cross_attn(attn, x, pos)
    z_neg = _ref_cross_attn(attn, x, neg_expanded)
    params = state.params
    combined = nag_combine(z_pos, z_neg, params.scale, params.tau, params.alpha)
    want = _ref_gate_and_out(attn, x, combined)

    if not torch.allclose(got, want, rtol=1e-5, atol=1e-6):
        raise AssertionError(
            f"patched NAG output != reference, max_abs_diff={(got - want).abs().max().item()}"
        )


# --------------------------------------------------------------------------- #
# Check 6: the patched VSF forward vs an independent reference                 #
# --------------------------------------------------------------------------- #


def check_vsf_forward_matches_reference() -> None:
    """The concatenated-softmax formula, against a hand-written reference.

    ``Z = attn(Q, [K+ ; K-], [V+ ; -alpha V-])`` -- ONE attention call, the
    negative half's values negated and scaled. The reference below concatenates
    the RAW projections and runs ONE preattention call over them, which is the
    same thing the patch does; the split-vs-concat equality that makes that
    legitimate is check 4's job, not this one's.
    """
    block = _build_block()
    attn = block.audio_attn2  # the audio module, so both modalities are exercised
    x, pos, neg = _tensors()

    state = _vsf_state(scale=1.5)
    state.set_contexts(neg.clone(), neg)
    service = NegPromptService(state_provider=lambda: state)
    service.install(_FakeTransformer([block]))

    got = attn.forward(x, context=pos)

    neg_expanded = neg.expand(x.shape[0], -1, -1)
    q = attn.to_q(x)
    k = torch.cat([attn.to_k(pos), attn.to_k(neg_expanded)], dim=1)
    q, k = attn.preattention_function(q, k, attn, None, None, None)
    v = torch.cat([attn.to_v(pos), attn.to_v(neg_expanded) * -1.5], dim=1)
    want = _ref_gate_and_out(attn, x, attn.attention_function(q, k, v, attn.heads))

    if not torch.allclose(got, want, rtol=1e-5, atol=1e-6):
        raise AssertionError(
            f"patched VSF output != reference, max_abs_diff={(got - want).abs().max().item()}"
        )


# --------------------------------------------------------------------------- #
# Check 7: OFF is a no-op that touches nothing                                 #
# --------------------------------------------------------------------------- #


def check_off_is_inert() -> None:
    """A job that asked for no negative prompt must leave the model untouched.

    ``install`` returning 0 is not enough on its own: the assertion that matters
    is that no module gained an instance-level ``forward``, because THAT is what
    would change the arithmetic. Checked on the ``__dict__`` directly rather
    than by comparing bound methods, which compare equal even when one is
    freshly created.
    """
    block = _build_block()
    transformer = _FakeTransformer([block])
    state = NagState()  # set_params never called: requested is False
    service = NegPromptService(state_provider=lambda: state)

    x, pos, _neg = _tensors()
    before = block.attn2.forward(x, context=pos)

    n = service.install(transformer)
    if n != 0:
        raise AssertionError(f"install() with nothing requested returned {n}, expected 0")
    for attn, modality in ((block.attn2, "video"), (block.audio_attn2, "audio")):
        if "forward" in attn.__dict__:
            raise AssertionError(f"install() with nothing requested patched {modality}")

    after = block.attn2.forward(x, context=pos)
    if not torch.equal(before, after):
        raise AssertionError("an OFF install changed the module's output")


# --------------------------------------------------------------------------- #
# Check 8: install -> uninstall round trip on a 48-block shell                 #
# --------------------------------------------------------------------------- #


def check_install_uninstall_roundtrip() -> None:
    """96 patched, 96 peeled, and the output bit-identical to the unpatched one.

    Run on 48 blocks rather than one, because 96 is the production count and a
    traversal that quietly stopped early would otherwise pass. The bit-identity
    at the end is the property the reused shell needs: after the peel, the very
    next job's build must find the model exactly as it would have been.
    """
    blocks = [_build_block() for _ in range(48)]
    transformer = _FakeTransformer(blocks)
    attn = blocks[0].attn2
    x, pos, neg = _tensors()
    before = attn.forward(x, context=pos)

    state = _nag_state()
    state.set_contexts(neg, neg.clone())
    service = NegPromptService(state_provider=lambda: state)

    if service.install(transformer) != 96:
        raise AssertionError("install did not patch exactly 96 modules on 48 blocks")
    if torch.equal(attn.forward(x, context=pos), before):
        raise AssertionError("the patched forward returned the unpatched value")

    removed = service.uninstall(transformer)
    if removed != 96:
        raise AssertionError(f"uninstall peeled {removed} modules, expected 96")
    for module in (blocks[0].attn2, blocks[47].audio_attn2):
        if "forward" in module.__dict__:
            raise AssertionError("uninstall left an instance-level forward behind")
    if not torch.equal(attn.forward(x, context=pos), before):
        raise AssertionError("the peeled module does not reproduce the unpatched output")

    # ...and a second uninstall on a clean shell peels nothing and does not
    # raise, which is what makes the unconditional strip on every build safe.
    if service.uninstall(transformer) != 0:
        raise AssertionError("a second uninstall reported peeling something")


# --------------------------------------------------------------------------- #
# Check 9: a direct double install raises                                      #
# --------------------------------------------------------------------------- #


def check_double_install_raises() -> None:
    """The invariant guard fires when ``uninstall`` was skipped.

    THIS STATE IS UNREACHABLE ON THE SHIPPING PATH -- the diffusion stage
    uninstalls unconditionally right before installing -- so this check is about
    the guard itself: it exists to catch a direct caller (a future refactor, a
    library user) whose second install would otherwise NEST one patch inside
    another and go on returning plausible numbers forever.
    """
    block = _build_block()
    transformer = _FakeTransformer([block])
    state = _nag_state()
    state.set_contexts(*(_tensors()[2],) * 2)
    service = NegPromptService(state_provider=lambda: state)
    service.install(transformer)
    try:
        service.install(transformer)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a second install() without uninstall() did not raise")


# --------------------------------------------------------------------------- #
# Check 10: uninstall completes on a disposed (meta) shell                     #
# --------------------------------------------------------------------------- #


def check_uninstall_on_meta_shell() -> None:
    """The reason ``uninstall`` may not touch a tensor.

    In production it runs FIRST on the build path, against a shell that has been
    through ``Disposable.dispose()``: every parameter is a ``device="meta"``
    tensor, where any operator dispatch raises ``NotImplementedError``. Moving
    the block to meta here reproduces that exactly, so a future ``.cpu()`` or
    ``empty_cache()`` added "to be tidy" fails here rather than on a real job.
    """
    block = _build_block()
    transformer = _FakeTransformer([block])
    state = _nag_state()
    state.set_contexts(*(_tensors()[2],) * 2)
    service = NegPromptService(state_provider=lambda: state)
    service.install(transformer)

    block.to(torch.device("meta"))
    removed = service.uninstall(transformer)
    if removed != 2:
        raise AssertionError(f"uninstall on a meta shell peeled {removed}, expected 2")
    if "forward" in block.attn2.__dict__:
        raise AssertionError("uninstall left the patch on the meta shell")


# --------------------------------------------------------------------------- #
# Check 11: every fail-loud condition really fails                             #
# --------------------------------------------------------------------------- #


def check_fail_loud_matrix() -> None:
    """All six guarded argument shapes raise, for BOTH methods.

    The set is 2.3's exactly. ``pe``/``k_pe`` carry extra weight on this engine:
    they are the runtime half of compat pin (14d)'s first leg, so a call that
    smuggled a positional embedding into text cross-attention has to die rather
    than run the concatenated-key shortcut on a position-DEPENDENT preattention.

    ``requested but not encoded`` is checked here too, because it is the same
    kind of fact -- a wiring mistake that must not become a silent skip.
    """
    x, pos, neg = _tensors()
    one = torch.zeros(1)
    cases = {
        "context=None": dict(context=None),
        "pe set": dict(context=pos, pe=one),
        "k_pe set": dict(context=pos, k_pe=one),
        "mask set": dict(context=pos, mask=one),
        "perturbation_mask set": dict(context=pos, perturbation_mask=one),
        "all_perturbed": dict(context=pos, all_perturbed=True),
    }
    for label, state in (("NAG", _nag_state()), ("VSF", _vsf_state())):
        block = _build_block()
        state.set_contexts(neg, neg.clone())
        NegPromptService(state_provider=lambda: state).install(_FakeTransformer([block]))
        for name, kwargs in cases.items():
            try:
                block.attn2.forward(x, **kwargs)
            except RuntimeError:
                continue
            raise AssertionError(f"{label}: patched forward did not raise for {name}")

    # requested-but-not-ready: set_params was called, set_contexts never was.
    not_ready = _nag_state()
    try:
        NegPromptService(state_provider=lambda: not_ready).install(
            _FakeTransformer([_build_block()])
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("install() did not raise when requested but not encoded")

    # ...and the negative context vanishing AFTER install (a reset mid-job) is
    # the other half of the same fact.
    live = _nag_state()
    block = _build_block()
    live.set_contexts(neg, neg.clone())
    NegPromptService(state_provider=lambda: live).install(_FakeTransformer([block]))
    live.reset()
    try:
        block.attn2.forward(x, context=pos)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a patched forward survived its state being reset")


def main() -> int:
    # Windows consoles are frequently cp932/cp1252 -- a stray non-ASCII
    # character in an exception message must never crash the reporter itself.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    torch.manual_seed(0)
    base._RESULTS.clear()

    shared: list[tuple[str, Callable[[], None]]] = [
        (
            "[shared with 2.3] nag_combine matches the KJNodes reference (bf16+fp32, both clamp branches)",
            base.check_combine_matches_kjnodes_reference,
        ),
        (
            "[shared with 2.3] identity short-circuit (alpha=0 / scale=1) returns the same z_pos object",
            base.check_identity_shortcircuit,
        ),
        (
            "[shared with 2.3] zero-norm degenerate inputs produce no NaN/Inf",
            base.check_zero_norm_degenerate,
        ),
    ]
    ltx25_only: list[tuple[str, Callable[[], None]]] = [
        ("[2.5] one preattention over concatenated keys == two separate calls (bit-exact)", check_concat_preattention_equals_split),
        ("[2.5] patched NAG forward == an independent reference from attn2's own parts", check_nag_forward_matches_reference),
        ("[2.5] patched VSF forward == an independent concatenated-softmax reference", check_vsf_forward_matches_reference),
        ("[2.5] OFF install patches nothing and changes no output", check_off_is_inert),
        ("[2.5] install/uninstall round trip on 48 blocks (96 modules, output restored)", check_install_uninstall_roundtrip),
        ("[2.5] a direct second install without uninstall raises RuntimeError", check_double_install_raises),
        ("[2.5] uninstall completes on a disposed (meta) shell without touching a tensor", check_uninstall_on_meta_shell),
        ("[2.5] fail-loud matrix: six argument shapes, both methods, plus the two wiring bugs", check_fail_loud_matrix),
    ]

    for name, fn in shared + ltx25_only:
        _run_check(name, fn)

    n_pass = sum(1 for _, ok, _ in base._RESULTS if ok)
    n_total = len(base._RESULTS)
    print(f"\n{n_pass}/{n_total} checks passed")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
