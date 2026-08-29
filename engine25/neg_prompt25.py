"""Non-CFG negative-prompt guidance (NAG / VSF) for the LTX 2.5 engine.

WHAT THIS FILE IS
-----------------
LTX 2.5 distilled is a CFG-free model: the pipeline runs ONE denoising pass and
``guidance_scale`` has nothing to attach to, exactly as on 2.3. The two methods
that recover a negative prompt without CFG were built for 2.3 first
(``engine/transformer/nag_service.py`` — NAG, arXiv:2505.21179 — and
``engine/transformer/vsf_service.py`` — VSF, arXiv:2508.10931), and the ALGEBRA
of both is engine-independent. What is NOT engine-independent is the shape of
the ``Attention`` module the replacement ``forward`` has to reproduce, and that
is the whole content of this file.

BORROWED FROM 2.3, NOT COPIED (seven names)
-------------------------------------------
:class:`~engine.transformer.nag_service.NagParams`,
:class:`~engine.transformer.vsf_service.VsfParams`,
:class:`~engine.transformer.nag_service.NagState`,
:func:`~engine.transformer.nag_service.nag_combine`,
:func:`~engine.transformer.nag_service._cross_attn_modules`,
:class:`~engine.transformer.vsf_service._MassLogBudget` and
:func:`~engine.transformer.vsf_service._log_negative_mass` are IMPORTED from the
2.3 modules and used unchanged. A second copy of the combine formula, of the
state machine or of the mass instrument could only drift, and a drifted copy of
a validated numeric path is worse than no copy.

**FOUR OF THE SEVEN ARE PRIVATE NAMES CROSSING A MODULE BOUNDARY**
(``_cross_attn_modules``, ``_MassLogBudget``, ``_log_negative_mass``, and the
``_`` -prefixed constants they read). engine25 depends on them, so **the 2.3
side may not rename those four without updating this module** — the leading
underscore means "not part of 2.3's public API", not "unused". This is stated
here and in ``Docs/MULTI_ENGINE_DESIGN.md``; the 2.3 files themselves are
untouched by this theme (zero lines).

WHAT IS *NOT* BORROWED: the replacement ``forward``
---------------------------------------------------
2.3's ``Attention.forward`` and 2.5's are different functions, so the patched
forward is written fresh here. Four differences, each load bearing:

1. **ONE ``preattention_function`` call, over CONCATENATED keys.** 2.5 moved
   the q/k RMSNorm (and RoPE, which cross-attention never uses) out of
   ``Attention.forward`` and into the pluggable ``preattention_function`` slot,
   which takes ``(q, k, attn_module, mask, pe, k_pe)`` and returns ``(q, k)``.
   NAG needs a normalised ``q`` plus TWO normalised key tensors. Calling
   preattention twice would be mathematically correct — see the note below — but
   would materialise ``q``'s normalisation twice (about +230 MB at production
   token counts), so this module calls it ONCE with ``torch.cat([k_pos, k_neg],
   dim=1)`` and splits the result.

   The equivalence rests on TWO legs, and both are pinned:

   * ``pe`` is ``None`` on every text cross-attention call (so the position-
     dependent ``apply_rotary_emb`` branch inside preattention is dead) — pinned
     as compat pin (14d)'s source leg in :mod:`engine25.ltxcore_compat`, and
     re-checked at call time by this module's own fail-loud guard, which refuses
     a non-None ``pe`` outright;
   * ``q_norm`` / ``k_norm`` are ``torch.nn.RMSNorm``, i.e. normalise over the
     LAST dimension independently per row, so splitting a concatenated sequence
     afterwards gives the same numbers as normalising the halves separately —
     pinned as (14d)'s runtime leg by :meth:`NegPromptService.install`'s
     ``isinstance`` check, in the same style as the ``caption_projection`` check
     beside it.

   If either leg ever breaks, the RETREAT IS A LEGITIMATE ONE and not a
   redesign: call ``preattention_function`` twice (once per key half) and pay
   the extra ``q`` allocation. That is what 2.3 effectively does; it is
   mathematically correct, just more expensive.

2. **``attention_function`` takes FOUR arguments on 2.5** — ``(q, k, v,
   heads)``. 2.3's five-argument call (with a trailing ``mask``) raises
   ``TypeError`` against the FA3/FA4 callables 2.5 can install, so the mask is
   not passed at all. It could not be used here anyway: a mask would route
   through ``masked_attention_function``, a DIFFERENT slot this module never
   touches. Pinned as (14b).

3. **Per-head gating goes through the official slot.** 2.5 hoisted the gate
   arithmetic into ``gated_attention_function(x, attn_out, attn_module)``, so
   this module calls that rather than re-implementing the four lines 2.3 had to
   inline. Pinned as (14c), which also fixes the ORDER: combine (or the VSF
   single softmax) → gate → ``to_out``, matching both the vanilla forward and
   KJNodes' patched one.

4. **Fail-loud guards are 2.3's set exactly** — ``context is None``, ``pe``,
   ``k_pe``, ``mask``, ``perturbation_mask``, ``all_perturbed``. Identical
   coverage, because the argument shape a text cross-attention call can have is
   the same on both engines.

The asymmetry is 2.3's and is deliberate: the POSITIVE context arrives already
AdaLN-modulated for the timestep (``apply_cross_attention_adaln``), while the
NEGATIVE context in :class:`NagState` is encoded once and stays raw. KJNodes'
``LTX2_NAG`` has the same asymmetry and its tuned defaults were chosen under it.

INSTALL / UNINSTALL: THE SHELL IS REUSED
----------------------------------------
2.3 builds a brand new transformer per job and therefore has no uninstall at
all. 2.5 hands back the SAME ``LTXModel`` instance on every build, so a patch
left behind is a patch the next job inherits. The answer is the one
:meth:`engine25.gguf_transformer.Ltx25DiffusionStage._ensure_sage_installed`
already uses for SageAttention, in the same order and for the same reason:
**every build unconditionally uninstalls, then installs.**

:meth:`NegPromptService.uninstall` pops ``"forward"`` out of the module's
``__dict__``, which restores the CLASS method. It deliberately does **not** keep
the original bound method anywhere:

* a saved ``orig_forward`` from build N would be captured by build N+1's closure
  and the patches would NEST one level deeper per build — silently, because a
  nested patch still returns plausible numbers;
* and a module holding a bound method of itself is a reference cycle on a
  22B-parameter shell.

``uninstall`` **touches no tensor and never raises**, verbatim the discipline
``Ltx25DiffusionStage._unpatch_block_swap`` states in capitals: it runs on a
shell that has been through ``Disposable.dispose()``, where every parameter is a
``device="meta"`` tensor and any operator dispatch raises
``NotImplementedError``. Restoring a Python attribute is the only thing that is
safe here, and the only thing that is needed. Do not add a ``.to()``, a
``.cpu()`` or an ``empty_cache()`` "just to be tidy".

A non-zero strip count is therefore **NORMAL, not a defect**: a resident worker
running two NAG jobs in a row finds job N's forwards still on the modules when
job N+1 builds. It is logged at INFO for that reason.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

import torch

# THE SEVEN BORROWED NAMES. See the module docstring: these are imported, never
# copied, and the four private ones are a cross-module dependency the 2.3 side
# must not rename without updating this file.
from engine.transformer.nag_service import (  # noqa: PLC2701 -- private-by-agreement, see docstring
    NagParams,
    NagState,
    _cross_attn_modules,
    nag_combine,
)
from engine.transformer.vsf_service import (  # noqa: PLC2701 -- ditto
    VsfParams,
    _log_negative_mass,
    _MASS_LOG_BUDGET,
    _MassLogBudget,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NagParams",
    "NagState",
    "NegPromptService",
    "VsfParams",
    "make_negative_forward",
    "nag_combine",
]


# --------------------------------------------------------------------------- #
# The two replacement forwards                                                 #
# --------------------------------------------------------------------------- #


def _reject_unexpected_call(
    method: str,
    modality: str,
    context: torch.Tensor | None,
    mask: torch.Tensor | None,
    pe: torch.Tensor | None,
    k_pe: torch.Tensor | None,
    perturbation_mask: torch.Tensor | None,
    all_perturbed: bool,
) -> None:
    """2.3's fail-loud guard, the SAME six conditions, shared by both forwards.

    This patch is only ever installed on ``attn2`` / ``audio_attn2`` text
    cross-attention, and on 2.5 that call site is
    ``BasicAVTransformerBlock._apply_text_cross_attention`` /
    ``apply_cross_attention_adaln``, neither of which passes ``pe``, ``k_pe``,
    ``perturbation_mask`` or ``all_perturbed`` at all, and whose ``mask`` is
    ``context_mask`` — hardcoded ``None`` by
    ``ltx_pipelines.utils.helpers.modality_from_latent_state`` (compat pin
    (14a)).

    Any of these arriving means the call site changed underneath this patch.
    There is no fallback to take: ``orig_forward`` is deliberately not kept (see
    the module docstring), and guessing would produce a plausible-looking video
    computed by the wrong algorithm. So it raises.

    ``pe``/``k_pe`` in particular are the RUNTIME half of (14d)'s first leg: the
    single-preattention-over-concatenated-keys shortcut is only equal to the
    per-half computation while no positional embedding is applied inside
    ``preattention_function``.
    """
    if (
        context is None
        or pe is not None
        or k_pe is not None
        or mask is not None
        or perturbation_mask is not None
        or all_perturbed
    ):
        raise RuntimeError(
            f"{method} forward ({modality}): received an argument shape this "
            "patch was never designed to handle (context is "
            f"None={context is None}, pe set={pe is not None}, k_pe "
            f"set={k_pe is not None}, mask set={mask is not None}, "
            f"perturbation_mask set={perturbation_mask is not None}, "
            f"all_perturbed={all_perturbed}). This module is only ever "
            "installed on attn2/audio_attn2 text cross-attention, which never "
            "sets these — the call site changed underneath the patch."
        )


def _negative_context(
    method: str, state: NagState, modality: str, context: torch.Tensor
) -> torch.Tensor:
    """The job's negative context, batch-expanded to match ``context``.

    Shared by both forwards because the two questions it answers are the same
    for both: "is the state still populated?" (it must be — ``install`` only
    patches when ``state.ready``, so a None here means the state was reset or
    replaced AFTER the install, i.e. a wiring bug) and "does the batch match?"
    (the negative prompt is encoded once, at batch 1, while the positive context
    can be batched upstream).
    """
    neg_context = state.context_for(modality)
    if neg_context is None:
        raise RuntimeError(
            f"{method} forward ({modality}): negative context is missing at "
            "call time even though this module was patched — the NagState was "
            "reset or replaced after NegPromptService.install() ran."
        )
    if neg_context.shape[0] != context.shape[0]:
        neg_context = neg_context.expand(context.shape[0], -1, -1)
    return neg_context


def _make_nag_forward(
    attn: torch.nn.Module, state: NagState, modality: str
) -> Callable[..., torch.Tensor]:
    """Build NAG's replacement for one ``Attention.forward`` (2.5 shape).

    NO ``orig_forward`` PARAMETER, unlike 2.3's factory. This engine's
    ``uninstall`` restores the class method by popping the instance attribute,
    so nothing here ever needs the previous bound method — and capturing one
    would nest the patches across builds (module docstring).

    Two attention calls, ONE preattention call: ``q`` is normalised once and the
    two key halves ride through the same call concatenated, then split. See
    difference 1 in the module docstring for why, and for the legitimate retreat
    if the equivalence ever stops holding.
    """

    def nag_forward(
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        _reject_unexpected_call(
            "NAG", modality, context, mask, pe, k_pe, perturbation_mask, all_perturbed
        )
        assert context is not None  # implied by the guard above
        neg_context = _negative_context("NAG", state, modality, context)

        params = state.params
        assert params is not None  # implied by state.ready, checked at install()

        n_pos = int(context.shape[1])
        q = attn.to_q(x)
        k = torch.cat([attn.to_k(context), attn.to_k(neg_context)], dim=1)
        # ONE preattention call. mask/pe/k_pe are all None here by the guard
        # above; the four-argument shape below is 2.5's
        # ``PreAttentionCallable`` protocol, not a convenience.
        q, k = attn.preattention_function(q, k, attn, None, None, None)
        k_pos, k_neg = k[:, :n_pos], k[:, n_pos:]
        del k

        # Both attention calls go through whatever backend is installed on the
        # module -- including SageAttention when the job asked for it (the sage
        # service swaps THIS attribute while this module swaps ``forward``), so
        # a NAG+sage job runs its positive AND its negative attention on the
        # sage kernel. Four arguments, never five: see difference 2.
        v_pos = attn.to_v(context)
        z_pos = attn.attention_function(q, k_pos, v_pos, attn.heads)
        del k_pos, v_pos

        v_neg = attn.to_v(neg_context)
        z_neg = attn.attention_function(q, k_neg, v_neg, attn.heads)
        del k_neg, v_neg, q

        out = nag_combine(z_pos, z_neg, params.scale, params.tau, params.alpha)
        del z_pos, z_neg  # free promptly: this call's peak is 3 tensors, not 1

        # Combine BEFORE gating, through the official slot: difference 3.
        if attn.to_gate_logits is not None:
            out = attn.gated_attention_function(x, out, attn)
        return attn.to_out(out)

    return nag_forward


def _make_vsf_forward(
    attn: torch.nn.Module,
    state: NagState,
    modality: str,
    log_budget: _MassLogBudget | None = None,
) -> Callable[..., torch.Tensor]:
    """Build VSF's replacement for one ``Attention.forward`` (2.5 shape).

    Structurally :func:`_make_nag_forward` with a different middle: same guards,
    same single preattention call over concatenated keys — except that VSF
    WANTS the concatenation and never splits it back. One attention call over
    ``[K+ ; K-]`` / ``[V+ ; -scale * V-]``, so both halves share one softmax
    denominator, which is the whole mechanism (attenuation + repulsion).

    ``_log_negative_mass`` is 2.3's instrument, called directly: it reads ``q``
    and ``k`` and writes nothing, so it cannot perturb the generation. It is the
    primary "is VSF doing anything?" signal, because the negative half is a
    handful of real tokens against 1024 positive ones.
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
        _reject_unexpected_call(
            "VSF", modality, context, mask, pe, k_pe, perturbation_mask, all_perturbed
        )
        assert context is not None  # implied by the guard above
        neg_context = _negative_context("VSF", state, modality, context)

        params = state.params
        assert params is not None  # implied by state.ready, checked at install()

        q = attn.to_q(x)
        k = torch.cat([attn.to_k(context), attn.to_k(neg_context)], dim=1)
        q, k = attn.preattention_function(q, k, attn, None, None, None)

        # Out-of-place negation: ``to_v``'s output is a fresh tensor today, but
        # an in-place multiply would be a trap the day the projection returns a
        # view.
        v = torch.cat(
            [attn.to_v(context), attn.to_v(neg_context) * (-params.scale)], dim=1
        )

        if log_budget is not None and log_budget.take():
            _log_negative_mass(
                q, k, attn.heads, int(neg_context.shape[1]), modality, tuple(context.shape)
            )

        out = attn.attention_function(q, k, v, attn.heads)
        del k, v, q

        if attn.to_gate_logits is not None:
            out = attn.gated_attention_function(x, out, attn)
        return attn.to_out(out)

    return vsf_forward


def make_negative_forward(
    attn: torch.nn.Module,
    state: NagState,
    modality: str,
    *,
    log_budget: _MassLogBudget | None = None,
) -> Callable[..., torch.Tensor]:
    """Pick the forward for this job's method. THE ONLY place that branches.

    The params TYPE carries the method, exactly as on 2.3 — ``VsfParams`` means
    VSF, anything else means NAG — so neither forward has to ask "which method
    am I?" on every call, and the service below is method-agnostic.
    """
    if isinstance(state.params, VsfParams):
        return _make_vsf_forward(attn, state, modality, log_budget=log_budget)
    return _make_nag_forward(attn, state, modality)


# --------------------------------------------------------------------------- #
# The service                                                                  #
# --------------------------------------------------------------------------- #


class NegPromptService:
    """Install / uninstall the negative-prompt forwards on one transformer.

    ONE service class for BOTH methods, unlike 2.3's ``NagService`` +
    ``VsfService`` pair. The strip/install lifecycle is what this class is, and
    that lifecycle does not depend on the method: only
    :func:`make_negative_forward` does, and it is one call.

    State is read through ``state_provider()`` on every call rather than
    captured at construction, so one long-lived service always sees the CURRENT
    job's state — the same closure pattern
    :class:`~engine.transformer.sage_attention_service.SageAttentionService` uses
    on this engine.
    """

    def __init__(self, state_provider: Callable[[], NagState]) -> None:
        self._state_provider = state_provider

    # -- uninstall ---------------------------------------------------------- #

    def uninstall(self, transformer: torch.nn.Module) -> int:
        """Pop every patched ``forward`` off the modules. Returns how many.

        **NEVER RAISES, AND TOUCHES NO TENSOR.** Both halves are load bearing
        and neither is decoration — the module docstring gives the full
        argument; the short version is that this is the FIRST thing that happens
        on the build path, and it runs against a shell whose parameters are all
        on ``device="meta"`` after ``Disposable.dispose()``.

        ``__dict__.pop("forward")`` and nothing else: the instance attribute
        that shadows the class method is removed, so the module goes back to
        ``Attention.forward``. No original is restored because none was kept
        (see the module docstring on nesting and reference cycles).

        A non-zero return is the NORMAL state of a reused shell after a
        negative-prompt job, not a defect.
        """
        removed = 0
        try:
            for attn, _modality in _cross_attn_modules(transformer):
                if attn.__dict__.pop("forward", None) is not None:
                    removed += 1
        except Exception:  # noqa: BLE001 — never take the build down
            logger.exception(
                "negative-prompt: uninstall could not complete (ignored)"
            )
        return removed

    # -- install ------------------------------------------------------------ #

    def install(self, transformer: torch.nn.Module) -> int:
        """Patch every text cross-attention module, or do nothing at all.

        Returns the number of modules patched — 0 when this job did not ask for
        a negative prompt, in which case **nothing on the model is touched**:
        that early return is what keeps an ordinary 2.5 job byte-identical to
        what it was before this feature existed.

        "Requested but not encoded" is a WIRING BUG, not a reason to skip:
        by the time a transformer is built the prompt encoder has already run
        (:class:`engine25.pipeline25.Ltx25PromptEncoder` encodes the negative
        prompt in the same pass as the positive one), so a ``requested and not
        ready`` state can only mean the arm and the encode came apart.
        """
        state = self._state_provider()
        if not state.requested:
            # The zero-overhead-when-off gate. Nothing below this line runs for
            # an ordinary job.
            return 0
        if not state.ready:
            raise RuntimeError(
                "a negative prompt was requested for this job but was never "
                "encoded (NagState.ready is False) — Ltx25PromptEncoder must "
                "encode it and call set_contexts() before the transformer is "
                "built."
            )

        params = state.params
        assert params is not None  # implied by state.requested

        self._check_no_caption_projection(transformer)

        budget = _MassLogBudget(_MASS_LOG_BUDGET) if isinstance(params, VsfParams) else None
        count = 0
        for attn, modality in _cross_attn_modules(transformer):
            self._check_module_shape(attn, modality)
            attn.forward = make_negative_forward(  # type: ignore[method-assign]
                attn, state, modality, log_budget=budget
            )
            count += 1

        if count == 0:
            raise RuntimeError(
                "a negative prompt was requested but _cross_attn_modules found "
                "no attn2/audio_attn2 modules on this transformer — either the "
                "traversal no longer matches the model's structure, or an "
                "empty/wrong object was passed to install()."
            )

        logger.info(
            "%s installed on %d cross-attention modules (%s negative=%r)",
            "VSF" if isinstance(params, VsfParams) else "NAG",
            count,
            f"scale={params.scale:.2f}"
            if isinstance(params, VsfParams)
            else f"scale={params.scale:.2f} tau={params.tau:.2f} alpha={params.alpha:.2f}",
            params.negative_prompt[:60],
        )
        return count

    # -- install-time invariant checks -------------------------------------- #

    def _check_module_shape(self, attn: torch.nn.Module, modality: str) -> None:
        """The two structural facts one patched module has to satisfy.

        **BOTH ARE INVARIANT GUARDS, not error handling for a reachable state.**

        (a) ``"forward" not in attn.__dict__``. On the shipping path this is
        structurally unreachable: ``Ltx25DiffusionStage._ensure_neg_installed``
        uninstalls unconditionally immediately before calling this, so nothing
        can be there. It is kept because it catches the OTHER caller — a direct
        ``install()`` without the matching ``uninstall()``, i.e. the misuse a
        future refactor could introduce — and because a double install would
        nest one patch inside another SILENTLY (a nested patch still returns
        plausible numbers). ``engine25.neg_selfcheck25``'s check 6 is what
        proves the detection actually fires.

        (b) ``q_norm`` / ``k_norm`` are ``torch.nn.RMSNorm``. This is the RUNTIME
        leg of compat pin (14d): the single-preattention-over-concatenated-keys
        shortcut is exact only because RMSNorm normalises each row over the last
        dimension independently, so splitting after the fact is the same
        arithmetic as normalising the halves apart. A different norm class (one
        with cross-position statistics) would make the shortcut silently wrong,
        which is precisely the failure a numeric check could not see. Same
        style, same reason, as the ``caption_projection`` check beside it.
        """
        if "forward" in attn.__dict__:
            raise RuntimeError(
                f"negative-prompt install ({modality}): this attention module "
                "already carries an instance-level `forward`, which means "
                "uninstall() did not run before install(). Patching on top of "
                "it would nest one patch inside another silently. (Unreachable "
                "from Ltx25DiffusionStage, which always uninstalls first; this "
                "guards a direct caller.)"
            )
        for name in ("q_norm", "k_norm"):
            norm = getattr(attn, name, None)
            if not isinstance(norm, torch.nn.RMSNorm):
                raise RuntimeError(
                    f"negative-prompt install ({modality}): attn.{name} is "
                    f"{type(norm).__name__}, not torch.nn.RMSNorm. This patch "
                    "normalises the positive and negative keys in ONE "
                    "preattention call over a concatenated tensor and then "
                    "splits the result, which is only equal to normalising the "
                    "halves separately while the norm is position-independent "
                    "(compat pin (14d)). Refusing rather than producing "
                    "silently different numbers."
                )

    def _check_no_caption_projection(self, transformer: torch.nn.Module) -> None:
        """Refuse if the transformer carries a caption projection.

        The negative context stored in :class:`NagState` is the embeddings
        processor's output reshaped exactly the way
        ``TransformerArgsPreprocessor._prepare_context`` reshapes the POSITIVE
        one — a bare ``.view(B, -1, D)``, which is all that method does while
        ``caption_projection is None``. On the 22B checkpoints this engine runs
        it always is None (``caption_proj_before_connector: true`` puts the
        projection in the TEXT ENCODER instead, so
        ``_build_caption_projections`` returns ``(None, None)``).

        If a future checkpoint brought the projection back into the transformer,
        the positive context would be projected and the negative one would not:
        two tensors at different representation stages fed to the same
        attention, which is a silently wrong result rather than a crash. Hence
        an explicit check, in the same style as the RMSNorm one above.
        """
        inner = getattr(transformer, "velocity_model", transformer)
        for name in ("caption_projection", "audio_caption_projection"):
            if getattr(inner, name, None) is not None:
                raise RuntimeError(
                    f"negative-prompt install: this transformer has a `{name}`. "
                    "The negative context is prepared as if `_prepare_context` "
                    "were a bare reshape (true only while the projection lives "
                    "in the text encoder), so with a projection present the "
                    "positive and negative contexts would reach attention at "
                    "different representation stages. Refusing."
                )


def cross_attention_modules(transformer: torch.nn.Module) -> Iterator[tuple[Any, str]]:
    """Public alias for the borrowed private traversal, for the selfcheck.

    Exists so :mod:`engine25.neg_selfcheck25` (and a future reader) has one
    obvious name to reach for instead of importing 2.3's underscore-prefixed
    function a second time.
    """
    return _cross_attn_modules(transformer)
