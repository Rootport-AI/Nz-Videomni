"""Value Sign Flip (VSF) service for the LTX-2 distilled transformer.

VSF (arXiv:2508.10931) is the SECOND non-CFG negative-prompt method in this
engine, alongside NAG (see nag_service.py). Same problem statement — this
backend's distilled pipeline pins ``guidance_scale == 1.0``, so classic
two-pass CFG has nothing to push away from — but a different mechanism:

    Z = softmax(Q · [K+ ; K-]^T / sqrt(d)) · [V+ ; -alpha * V-]

The positive and negative contexts are CONCATENATED along the key/value axis
and run through ONE attention call, with the negative half's values negated
and scaled. The essential part is that both halves share a single softmax
denominator: the negative keys steal probability mass from the positive keys
(attenuation) AND contribute a sign-flipped value (repulsion). That double
action is why VSF has no renormalisation and no blend factor — unlike NAG
there is nothing to combine after the fact, so ``nag_combine``'s
extrapolate/renormalise/blend has no analogue here.

Relationship to nag_service (deliberately one-directional): this module
imports FROM nag_service (NagState, _cross_attn_modules) and nag_service
never imports from here at runtime — a circular import would be a build-order
hazard on a module both pipelines import at startup. Everything shared is
shared by reuse, not by inheritance: VsfService is a sibling of NagService,
not a subclass, so neither method's forward has to carry an "which method am
I?" branch.

*** The dominant failure mode is "installed correctly but has no effect" ***
The positive context that reaches attn2 is always (B, 1024, dim) on this
wheel: the real prompt tokens are packed at the FRONT and the remainder is
filled by the connector's learned register embeddings, which are real trained
data, not padding. The negative context is sliced down to its real tokens at
ENCODE time (nag_service.encode_negative(..., slice_to_real_tokens=True)) —
sign-flipping learned registers would inject a strong, meaningless repulsion
and is unambiguously harmful, so the slice is a correctness requirement, not
an optimisation. But the consequence is a mass imbalance: ~10 negative keys
against 1024 positive ones, so the share of softmax mass the negative half
captures (call it ``m``) can be ~1%. The output's negative term is
-alpha * m * mean(V-), so a small ``m`` is recoverable by raising alpha —
which is exactly why ``vsf_scale``'s API range goes to 100 instead of the
reference implementation's 10. To make this measurable rather than guessable,
the first few patched forwards of every job log ``m`` at INFO (see
_log_negative_mass): that number, not a pixel diff of the output video, is
the primary "is it working?" signal.

AdaLN asymmetry is a first-class experiment here (``adaln_mode``). The
production call path modulates the POSITIVE context per timestep via
``apply_cross_attention_adaln`` (transformer.py:373-392) while a
NagState-held negative context stays raw forever. NAG keeps that asymmetry
because its reference implementation (KJNodes) does; VSF's paper is silent
about it, and under a shared softmax the K-side scale mismatch directly
changes ``m``. So all three hypotheses are selectable and decided on real
hardware: "raw" (negative stays raw, NAG-like), "modulated" (negative gets
the same context modulation, matching K scales), "v_scale" (keys raw, only
the negative VALUE path scaled).

``adaln_stash_window`` is what makes modes other than "raw" possible, and its
placement matters: ``apply_cross_attention_adaln`` is resolved by LOAD_GLOBAL
inside ``BasicAVTransformerBlock._apply_text_cross_attention`` at FORWARD
time, so the patch window must stay open across the whole denoise, not merely
across the model-construction call. Callers therefore wrap the entire
inference (single: the ``self.pipeline(...)`` call; chain: from
``ledger.transformer()`` through the last stage-2 tile). The window is a
no-op for every other job — NAG, VSF with adaln_mode="raw", and no negative
prompt at all — which is what lets callers place the ``with`` unconditionally
without ever touching a wheel global on the OFF path.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass
from typing import Callable, Iterator

import torch

from engine.transformer.nag_service import NagState, _cross_attn_modules

logger = logging.getLogger(__name__)

# The three AdaLN hypotheses (see module docstring). Anything else is a wiring
# bug in the API/worker layer, never a runtime fallback.
ADALN_MODES: tuple[str, ...] = ("raw", "modulated", "v_scale")

# Per-module attribute adaln_stash_window's wrapper writes the current step's
# (shift_kv, scale_kv) to, and the patched forward POPS. Pop (not read) is
# deliberate: apply_cross_attention_adaln calls the attention module exactly
# once per stash, so a leftover stash can only mean the window closed early or
# a module was called outside it — and popping turns that into the fail-loud
# "no stash" RuntimeError instead of silently reusing a previous step's
# coefficients.
_STASH_ATTR = "_vsf_kv_mod"

# How many patched forwards per job log their negative softmax mass. Cross-
# attention runs in block order within a denoising step, so 4 covers block 0
# and block 1 for both modalities (video/audio interleaved) — enough to see
# whether m is uniform across blocks without paying for the extra matmul on
# all 96 modules x every step.
_MASS_LOG_BUDGET = 4

# Query rows sampled (evenly strided) when measuring m. The full attention
# probability matrix at production sizes is (B, H, ~30k, ~1040) — a gigabyte
# of logging. A strided sample of a few hundred query rows estimates the mean
# negative mass to far better precision than the decision it informs ("is m
# 1% or 20%?") requires.
_MASS_LOG_QUERIES = 256


@dataclass(frozen=True)
class VsfParams:
    """One job's VSF request.

    ``scale`` is the paper's alpha (the negative values' multiplier); the
    reference implementations' default is 1.5 and Wan's tuned value is 1.7,
    but see the module docstring on why this backend allows much larger
    values. ``adaln_mode`` must be one of ADALN_MODES; it is validated ONCE,
    here in __post_init__, rather than separately at each of install/forward/
    window — a bad value fails as soon as the job's params object is built,
    with the same fail-loud RuntimeError style as the rest of this module.
    """

    negative_prompt: str
    scale: float
    adaln_mode: str

    def __post_init__(self) -> None:
        if self.adaln_mode not in ADALN_MODES:
            raise RuntimeError(
                f"VsfParams: unknown adaln_mode {self.adaln_mode!r} — expected "
                f"one of {ADALN_MODES}. The API layer validates this as a "
                "Literal, so reaching here means a payload bypassed validation."
            )


class _MassLogBudget:
    """Per-install counter that limits the ``m`` logging to the first few
    forwards of a job. One instance is shared by all 96 patched closures, and
    a fresh transformer (hence a fresh install) per job resets it."""

    def __init__(self, remaining: int) -> None:
        self.remaining = int(remaining)

    def take(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


def _log_negative_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    heads: int,
    n_neg: int,
    modality: str,
    context_shape: tuple[int, ...],
) -> None:
    """Log the mean share of softmax mass the NEGATIVE keys capture.

    This is the primary "is VSF actually doing anything?" instrument (see the
    module docstring's mass-imbalance section). Computed on a strided sample
    of query rows in fp32, entirely outside the value path — it reads q/k and
    writes nothing, so it cannot perturb the generation, only slow the first
    few forwards of a job down by a negligible amount.
    """
    b, t, inner = q.shape
    dim_head = inner // heads
    stride = max(1, t // _MASS_LOG_QUERIES)
    q_sample = q[:, ::stride][:, :_MASS_LOG_QUERIES]

    # detach(): production runs under inference_mode so this is already a
    # no-op there, but the selfcheck (and any future grad-enabled caller)
    # must not build an autograd graph for a log line.
    q_h = q_sample.detach().reshape(b, -1, heads, dim_head).transpose(1, 2).float()
    k_h = k.detach().reshape(b, -1, heads, dim_head).transpose(1, 2).float()
    weights = torch.softmax(
        torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(dim_head), dim=-1
    )
    m = float(weights[..., -n_neg:].sum(dim=-1).mean())

    logger.info(
        "VSF m=%.4f (modality=%s, N_neg=%d, L_k=%d, context=%s, "
        "queries_sampled=%d/%d)",
        m,
        modality,
        n_neg,
        int(k.shape[1]),
        tuple(int(s) for s in context_shape),
        int(q_sample.shape[1]),
        t,
    )


def _make_vsf_forward(
    attn: torch.nn.Module,
    orig_forward: Callable[..., torch.Tensor],
    state: NagState,
    modality: str,
    log_budget: "_MassLogBudget | None" = None,
) -> Callable[..., torch.Tensor]:
    """Build a VSF-patched replacement for one Attention module's forward.

    Structurally this is nag_service._make_nag_forward with a different
    middle: the same fail-loud argument guards (this patch is only ever
    installed on attn2/audio_attn2 text cross-attention, which never receives
    pe/k_pe/mask/perturbation arguments), the same never-called ``orig_forward``
    parameter (kept for symmetry with the engine's other patch closures; there
    is no uninstall path, so an unexpected call shape must fail rather than
    fall back), and the same per-head-gate + to_out tail copied from
    Attention.forward (attention.py:237-249).

    The middle is the whole feature: ONE attention call over concatenated
    keys/values, with the negative half's values multiplied by ``-scale``.
    No mask is ever passed — the negative half participates fully, which is
    the point, and staying mask-free keeps every attention backend (SDPA
    here) on its fast path.
    """

    def vsf_forward(
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        # orig_forward (captured above) is intentionally never called here —
        # see this factory's docstring.
        if (
            context is None
            or pe is not None
            or k_pe is not None
            or mask is not None
            or perturbation_mask is not None
            or all_perturbed
        ):
            raise RuntimeError(
                f"VSF forward ({modality}): received an argument shape this "
                "patch was never designed to handle (context is "
                f"None={context is None}, pe set={pe is not None}, k_pe "
                f"set={k_pe is not None}, mask set={mask is not None}, "
                f"perturbation_mask set={perturbation_mask is not None}, "
                f"all_perturbed={all_perturbed}). This module is only ever "
                "installed on attn2/audio_attn2 text cross-attention, which "
                "never sets these — the call site changed underneath VSF."
            )

        neg_context = state.context_for(modality)
        if neg_context is None:
            # VsfService.install only patches modules when state.ready is
            # True, so this can only fire if the state was reset (or a fresh,
            # unpopulated NagState got substituted) after install ran — a
            # wiring bug in the caller, not a normal runtime path.
            raise RuntimeError(
                f"VSF forward ({modality}): negative context is missing at "
                "call time even though this module was patched — the shared "
                "NagState was reset or replaced after VsfService.install() ran."
            )

        if neg_context.shape[0] != context.shape[0]:
            # Batch-expand safety net, identical in spirit to NAG's: the
            # negative prompt is encoded once per job (batch 1) while the
            # positive context can be batched upstream.
            neg_context = neg_context.expand(context.shape[0], -1, -1)

        params = state.params
        assert params is not None  # implied by state.ready, checked at install()
        mode = params.adaln_mode  # validated once, at VsfParams.__post_init__

        # AdaLN hypotheses. "raw" needs nothing from the window at all, which
        # is why the window stays closed (and the wheel global untouched) for
        # that mode.
        neg_k_input = neg_context
        neg_v_input = neg_context
        if mode != "raw":
            stash = getattr(attn, _STASH_ATTR, None)
            if stash is None:
                raise RuntimeError(
                    f"VSF forward ({modality}): adaln_mode={mode!r} needs the "
                    "current step's (shift_kv, scale_kv), but nothing was "
                    "stashed on this module. adaln_stash_window() must be open "
                    "around the ENTIRE denoise (apply_cross_attention_adaln is "
                    "resolved by LOAD_GLOBAL at forward time, so wrapping only "
                    "the model build patches nothing)."
                )
            delattr(attn, _STASH_ATTR)  # pop: one stash per call, never reused
            shift_kv, scale_kv = stash
            if shift_kv.shape[1] != 1 or scale_kv.shape[1] != 1:
                raise RuntimeError(
                    f"VSF forward ({modality}): expected per-step AdaLN "
                    "coefficients broadcast over the sequence axis "
                    f"(shape[1] == 1), got shift_kv{tuple(shift_kv.shape)} "
                    f"scale_kv{tuple(scale_kv.shape)}. That means the wheel "
                    "now emits per-token prompt timesteps; slicing them onto "
                    "the negative context's (different) token count would be "
                    "a guess, so this fails instead."
                )
            if mode == "modulated":
                # Same modulation the positive context gets: matches K scales
                # on both halves of the shared softmax.
                modulated = neg_context * (1 + scale_kv) + shift_kv
                neg_k_input = modulated
                neg_v_input = modulated
            else:  # "v_scale"
                # Keys stay raw (so the mass split is the "raw" one) while the
                # VALUE magnitude is brought onto the positive half's scale.
                neg_v_input = neg_context * (1 + scale_kv)

        q = attn.q_norm(attn.to_q(x))

        # q_norm/k_norm are RMSNorm, i.e. position-independent, so projecting
        # and normalising each half separately and then concatenating is
        # exactly equal to normalising a pre-concatenated tensor.
        k = torch.cat(
            [attn.k_norm(attn.to_k(context)), attn.k_norm(attn.to_k(neg_k_input))],
            dim=1,
        )
        # Out-of-place negation: attn.to_v's output for the negative half is a
        # fresh tensor, but multiplying in place would still be a trap if the
        # projection ever starts returning a view.
        v = torch.cat(
            [attn.to_v(context), attn.to_v(neg_v_input) * (-params.scale)],
            dim=1,
        )

        if log_budget is not None and log_budget.take():
            _log_negative_mass(
                q, k, attn.heads, int(neg_context.shape[1]), modality, tuple(context.shape)
            )

        out = attn.attention_function(q, k, v, attn.heads, None)
        del k, v, q

        # Per-head gating, identical to Attention.forward's tail (attention.py:
        # 237-247) and to NAG's — gating is always the last step before to_out.
        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(x)  # (B, T, H)
            b, t, _ = out.shape
            out = out.view(b, t, attn.heads, attn.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, attn.heads * attn.dim_head)

        return attn.to_out(out)

    return vsf_forward


@contextlib.contextmanager
def adaln_stash_window(
    transformer_or_none: torch.nn.Module | None,
    params: object | None,
) -> Iterator[None]:
    """Make the current step's AdaLN (shift_kv, scale_kv) visible to the
    patched cross-attention forwards, for the duration of a denoise.

    ``apply_cross_attention_adaln`` computes those coefficients from the
    prompt timestep and immediately consumes them; the attention module it
    calls never sees them. This window replaces that module-level function
    with a wrapper that recomputes the same two tensors, stashes them on the
    ``attn`` argument (video and audio cross-attention are separate module
    instances, so the two streams cannot collide), and then delegates to the
    ORIGINAL function for the actual work.

    Delegating rather than re-implementing means the positive path's math is
    the wheel's own, for ever, at the cost of computing the coefficients
    twice. The duplicated work is one broadcast add over a (B, 1, 2, dim)
    tensor per module per step — utterly negligible next to the attention it
    precedes — whereas an inlined copy would silently drift if the wheel's
    formula ever changed. That trade is the whole reason for the choice.

    No-op window: for NAG jobs, VSF jobs with adaln_mode="raw", and jobs with
    no negative prompt at all, this yields WITHOUT reading or writing the
    wheel's module global, so callers can place the ``with`` unconditionally
    and an OFF job's process state stays identical (identity-checked by
    vsf_selfcheck).

    ``transformer_or_none`` is used only for the finally-time sweep of any
    stash that a raised exception left behind mid-step; the single-generate
    caller passes None because its transformer is built inside the wheel's
    pipeline call and is discarded with the job (and the forward pops its
    stash on every normal call anyway).
    """
    if not isinstance(params, VsfParams) or params.adaln_mode == "raw":
        yield
        return

    # adaln_mode is validated once, at VsfParams.__post_init__ — reaching here
    # with a VsfParams instance means it is already one of ADALN_MODES.
    from ltx_core.model.transformer import transformer as _transformer_mod

    original = _transformer_mod.apply_cross_attention_adaln

    # Logged once per window (not once per call — this closure fires on every
    # cross-attention module of every denoise step, and the shape is the same
    # every time within a job): confirms the stash actually carries the
    # per-step-broadcast shape adaln_mode's forward branch expects, without
    # spamming a job's log once per module per step.
    logged = False

    def stashing_adaln(
        x: torch.Tensor,
        context: torch.Tensor,
        attn: object,
        q_shift: torch.Tensor,
        q_scale: torch.Tensor,
        q_gate: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor,
        prompt_timestep: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ) -> torch.Tensor:
        # Mirrors transformer.py:394-397 exactly (the only lines of the
        # original this wrapper needs to know about).
        batch_size = x.shape[0]
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)
        setattr(attn, _STASH_ATTR, (shift_kv, scale_kv))
        nonlocal logged
        if not logged:
            logged = True
            logger.info(
                "VSF adaln stash: shift_kv=%s scale_kv=%s",
                tuple(shift_kv.shape),
                tuple(scale_kv.shape),
            )
        return original(
            x,
            context,
            attn,
            q_shift,
            q_scale,
            q_gate,
            prompt_scale_shift_table,
            prompt_timestep,
            context_mask,
            norm_eps,
        )

    _transformer_mod.apply_cross_attention_adaln = stashing_adaln
    try:
        yield
    finally:
        _transformer_mod.apply_cross_attention_adaln = original
        if transformer_or_none is not None:
            for attn, _modality in _cross_attn_modules(transformer_or_none):
                if hasattr(attn, _STASH_ATTR):
                    delattr(attn, _STASH_ATTR)


class VsfService:
    """Patches a freshly-built transformer's text cross-attention modules to
    apply VSF, or does nothing at all when the current job didn't request it.

    Sibling of NagService with the same contract — state read through
    ``state_provider()`` on every install so one long-lived instance always
    sees the CURRENT job's state, no uninstall (``ledger.transformer()``
    builds a new transformer per job), no double-patch guard (same reason),
    and "requested but not encoded yet" is a wiring bug rather than a silent
    skip. The pipeline picks between the two services by the type of the
    params the job set, so neither service carries a method branch.
    """

    def __init__(self, state_provider: Callable[[], NagState]) -> None:
        self._state_provider = state_provider

    def install(self, transformer: torch.nn.Module) -> int:
        state = self._state_provider()
        if not state.requested:
            return 0
        if not state.ready:
            raise RuntimeError(
                "VSF requested for this job but its negative prompt was "
                "never encoded (NagState.ready is False) — encode_negative()/"
                "set_contexts() must complete before ledger.transformer() is "
                "built (the same ordering guarantee NAG relies on)."
            )

        params = state.params
        assert params is not None  # implied by state.requested
        if not isinstance(params, VsfParams):
            raise RuntimeError(
                "VsfService.install() was called for a job whose params are "
                f"{type(params).__name__}, not VsfParams — the pipeline's "
                "service selection and the worker's method resolution "
                "disagree about this job's negative-prompt method."
            )
        # adaln_mode is validated once, at VsfParams.__post_init__.

        budget = _MassLogBudget(_MASS_LOG_BUDGET)
        count = 0
        for attn, modality in _cross_attn_modules(transformer):
            orig_forward = attn.forward
            attn.forward = _make_vsf_forward(  # type: ignore[method-assign]
                attn, orig_forward, state, modality, log_budget=budget
            )
            count += 1

        if count == 0:
            raise RuntimeError(
                "VSF requested but _cross_attn_modules found no attn2/"
                "audio_attn2 modules on this transformer — either the "
                "traversal no longer matches the model's structure, or an "
                "empty/wrong object was passed to install()."
            )

        logger.info(
            "VSF installed on %d cross-attention modules "
            "(scale=%.2f adaln=%s negative=%r)",
            count,
            params.scale,
            params.adaln_mode,
            params.negative_prompt[:60],
        )
        return count
