"""Normalized Attention Guidance (NAG) service for the LTX-2 distilled transformer.

NAG (arXiv:2505.21179) recovers negative-prompt guidance for a model whose
CFG is frozen at 1.0. This backend's distilled LTX 2.3 pipeline validates
`guidance_scale == 1.0` (see api/models.py), which makes classic two-pass
CFG unavailable: there is only ever one denoising pass, so the traditional
"push away from the unconditional prediction" trick has nothing to push
away from. NAG works entirely inside a single pass instead: it runs the
prompt's cross-attention twice per call (once against the positive context,
once against a negative context) and extrapolates/renormalizes/blends the
two attention *outputs* — no extra denoising step, no second sampler pass.

What is patched: only `attn2` (video prompt cross-attention) and
`audio_attn2` (audio prompt cross-attention) on every dual-stream transformer
block. Self-attention (attn1/audio_attn1) and the audio<->video cross-attention
(audio_to_video_attn/video_to_audio_attn) are untouched — NAG only ever
argues with the *text* prompt, never with the other modality.

Deliberate asymmetry (do not "fix" this without re-reading D1/D4 of the NAG
plan): this wheel's GGUF configs all set `cross_attention_adaln: true`, so
the production call path is `apply_cross_attention_adaln`
(ltx_core/model/transformer/transformer.py:373-392), which feeds attn2 a
POSITIVE context that has already been AdaLN-modulated for the current
timestep (`context * (1 + scale_kv) + shift_kv`). The NAG NEGATIVE context
held in `NagState`, by contrast, is encoded once up front and never touched
by AdaLN again — it stays raw for every step. This is not an oversight: the
reference implementation this feature mirrors, kijai/ComfyUI-KJNodes'
`LTX2_NAG` node, has the exact same asymmetry (its patched forward also
receives an AdaLN-modulated positive context and a raw negative context), and
its tuned defaults (scale=11.0, tau=2.5, alpha=0.25) were chosen under that
asymmetry. Symmetrizing (e.g. modulating the negative context too) would be
a different algorithm with untuned defaults, so it is intentionally not done
here.

Combine-before-gating order: per-head gating (`2 * sigmoid(to_gate_logits(x))`,
active whenever `apply_gated_attention: true`, which all three production GGUF
configs set) is applied AFTER the NAG combine, matching both the vanilla
`Attention.forward` (attention.py:237-247, gating is the last step before
`to_out`) and KJNodes' patched forward (NAG combine, then gate, then to_out).
The AdaLN `q_gate` multiply that wraps the whole `attn(...)` call in
`apply_cross_attention_adaln` happens outside this module's patched forward
entirely and needs no changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator

import torch

if TYPE_CHECKING:
    # Type-only import: VSF (the second non-CFG negative-prompt method, see
    # vsf_service.py) reuses NagState verbatim and stores its own params in
    # it. vsf_service imports FROM this module at runtime, so importing it
    # back here for real would be a circular import — under
    # `from __future__ import annotations` every annotation below is a string,
    # so this guard costs nothing at runtime.
    from engine.transformer.vsf_service import VsfParams

logger = logging.getLogger(__name__)

# torch.linalg.vector_norm accumulates in this dtype regardless of the input's
# dtype (bf16 in production) — L1 norms of ~thousands of terms are sensitive
# to accumulation error, and this is cheap insurance relative to the matmuls
# around it. KJNodes computes the norm in native dtype instead; the selfcheck
# script confirms both are within tolerance of each other.
_NORM_DTYPE = torch.float32

# Added to both norms before dividing, and to the positive norm's numerator
# term — guards the zero-norm degenerate case (e.g. amp-zeroed activations in
# a unit test) without special-casing it in nag_combine's control flow.
_EPS = 1e-7


@dataclass(frozen=True)
class NagParams:
    """One job's NAG request. Field names mirror KJNodes' `LTX2_NAG` node
    inputs so the defaults (scale=11.0, tau=2.5, alpha=0.25) transfer as-is."""

    negative_prompt: str
    scale: float
    tau: float
    alpha: float


class NagState:
    """Mutable NAG state for exactly one job.

    Lifetime: one instance lives on the pipeline (`FastVideoPipeline._nag`)
    for as long as the process is resident; `set_params`/`reset` scope it to a
    single generate()/generate_chain() call so a keep_resident worker never
    leaks one job's negative prompt into the next job's transformer.

    `requested` and `ready` are deliberately different: a job can request NAG
    (`set_params` called with non-None params) before its negative prompt has
    actually been encoded. `NagService.install` treats "requested but not
    ready" as a wiring bug (RuntimeError) rather than silently skipping,
    because by the time `transformer()` is built, encoding must already have
    happened (D2's ordering guarantee) — see nag_service.encode_negative and
    callers in fast_video_pipeline.py / chain_pipeline.py (Wave 1).
    """

    def __init__(self) -> None:
        self._params: NagParams | VsfParams | None = None
        self._video_context: torch.Tensor | None = None
        self._audio_context: torch.Tensor | None = None

    @property
    def params(self) -> NagParams | VsfParams | None:
        return self._params

    @property
    def video_context(self) -> torch.Tensor | None:
        return self._video_context

    @property
    def audio_context(self) -> torch.Tensor | None:
        return self._audio_context

    @property
    def requested(self) -> bool:
        """A job has asked for NAG (params were set), regardless of whether
        the negative prompt has been encoded into contexts yet."""
        return self._params is not None

    @property
    def ready(self) -> bool:
        """Params AND both negative contexts are present — the only state
        NagService.install() is willing to patch against."""
        return (
            self._params is not None
            and self._video_context is not None
            and self._audio_context is not None
        )

    def set_params(self, params: NagParams | VsfParams | None) -> None:
        """Set (or clear, with None) this job's NAG (or VSF) request.

        Always clears any previously-encoded contexts, even if params is
        unchanged from the prior job: this is the ONLY entry point that
        starts a job's NAG state, so a caller that calls this without a
        matching set_contexts() afterwards will correctly hit the
        "requested but not ready" RuntimeError in install() instead of
        silently reusing a stale encoding from a different prompt.
        """
        self._params = params
        self._video_context = None
        self._audio_context = None

    def set_contexts(self, video_context: torch.Tensor, audio_context: torch.Tensor) -> None:
        self._video_context = video_context
        self._audio_context = audio_context

    def context_for(self, modality: str) -> torch.Tensor | None:
        """modality is "video" or "audio" (the same strings _cross_attn_modules
        yields), one call per patched attention module per forward pass."""
        if modality == "video":
            return self._video_context
        if modality == "audio":
            return self._audio_context
        raise ValueError(f"NagState.context_for: unknown modality {modality!r}")

    def reset(self) -> None:
        """Called from the pipeline's try/finally so NAG state never survives
        past the job that requested it (resident-worker leak guard, same
        reasoning as BlockSwapService's resident-reuse fix)."""
        self._params = None
        self._video_context = None
        self._audio_context = None


def encode_negative(
    text_encoder: object,
    prompt: str,
    slice_to_real_tokens: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the NAG (or VSF) negative prompt into (video_context, audio_context).

    Import of `encode_text` is local to this function: it is the one place in
    this module that needs the live Gemma encoder wheel import, and keeping it
    out of the module header lets nag_service.py be imported (e.g. by
    nag_selfcheck.py's identity/degenerate-input checks) without requiring a
    loaded text encoder or its transformers/tokenizer dependencies.

    Reshape rationale: `encode_text` (ltx_core/text_encoders/gemma/encoders/
    base_encoder.py:230) returns raw (video_encoding, audio_encoding) tensors
    straight out of the embeddings processor. The production path instead
    runs every context through `TransformerArgsPreprocessor._prepare_context`
    (transformer_args.py:77-86):

        if self.caption_projection is not None:
            context = self.caption_projection(context)
        return context.view(batch_size, -1, x.shape[-1])

    All three production GGUF configs set `caption_proj_before_connector:
    true`, which means `caption_projection` is None on this wheel's
    transformer — so `_prepare_context` is *only* the trailing `.view(...)`,
    never a projection. We apply the identical reshape here (using the
    tensor's own last dimension, which already equals `x.shape[-1]` since
    there is no projection layer to change it) so the negative context that
    NagState hands to the patched attn2/audio_attn2 forward is at the exact
    same representation stage as the positive context those modules receive
    from the real preprocessing path.

    ``slice_to_real_tokens`` (default False — every NAG caller keeps the full,
    historically-encoded context, so this argument cannot change NAG's output
    by a single bit): trim both contexts to the prompt's REAL token count.
    The encoder always returns a fixed-length sequence whose tail is filled by
    the connector's learned register embeddings, and those registers are real
    trained data rather than padding. NAG can live with them because it
    combines two SEPARATE attention outputs; VSF cannot, because it
    concatenates the negative keys/values into one shared softmax and negates
    the negative values — sign-flipping learned registers would inject a large
    prompt-independent repulsion. Hence the slice is a correctness requirement
    for VSF and is applied HERE, at encode time, where the method is known,
    so the forward path stays method-agnostic and simply uses whatever tensor
    NagState holds. The real token count comes from the tokenizer's own
    attention weights (the same tokenize_with_weights call the encoder itself
    makes), so no second Gemma run is needed.
    """
    from ltx_core.text_encoders.gemma.encoders.base_encoder import encode_text

    [(video_context, audio_context)] = encode_text(text_encoder, [prompt])
    video_context = video_context.view(video_context.shape[0], -1, video_context.shape[-1])
    audio_context = audio_context.view(audio_context.shape[0], -1, audio_context.shape[-1])

    if slice_to_real_tokens:
        tokenizer = getattr(text_encoder, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "encode_negative(slice_to_real_tokens=True): the text encoder "
                "has no .tokenizer, so the real token count cannot be "
                "determined. Slicing is a correctness requirement for VSF "
                "(see this function's docstring), so this fails rather than "
                "silently sign-flipping the learned register embeddings."
            )
        pairs = tokenizer.tokenize_with_weights(prompt)["gemma"]
        n_real = int(sum(int(weight) for _token, weight in pairs))
        # min() over both modalities: the video and audio connectors emit the
        # same token count on this wheel, but the bound has to hold for the
        # tensor actually being sliced, not just the one we happened to check.
        seq_len = int(min(video_context.shape[1], audio_context.shape[1]))
        if n_real <= 0 or n_real > seq_len:
            raise RuntimeError(
                f"encode_negative(slice_to_real_tokens=True): tokenizer "
                f"reports {n_real} real tokens for a context of length "
                f"{seq_len} — expected 0 < N <= seq_len. Refusing to guess."
            )
        video_context = video_context[:, :n_real, :]
        audio_context = audio_context[:, :n_real, :]

    return video_context, audio_context


def nag_combine(
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    scale: float,
    tau: float,
    alpha: float,
) -> torch.Tensor:
    """Combine one cross-attention call's positive/negative outputs per NAG's
    extrapolate -> L1-renormalize -> blend formula (arXiv:2505.21179 eq. 7-10,
    KJNodes' `x_positive*scale - x_negative*(scale-1)` convention).

    z_pos is NEVER mutated (every op on it below is out-of-place or read-only)
    because the caller (KJNodes' Eq. 10 analogue, the final alpha-blend) needs
    its original value again after the extrapolated/renormalized branch is
    computed.

    Identity short-circuit: floating-point rounding means the general formula
    below does not reduce to bit-exact z_pos even at alpha=0 or scale=1 (the
    scale-1=0 term and the alpha=0 blend both still touch every element via a
    zero multiply/add, which is a fresh floating-point op, not a no-op at the
    bit level). Short-circuiting here is what makes "NAG OFF" (alpha=0) and
    "NAG scale=1" bit-identical to the unpatched path, which is the G2 real-
    device gate.
    """
    if alpha == 0.0 or scale == 1.0:
        return z_pos

    z_g = z_pos.mul(scale)
    z_g.sub_(z_neg, alpha=(scale - 1.0))

    norm_pos = torch.linalg.vector_norm(z_pos, ord=1, dim=-1, keepdim=True, dtype=_NORM_DTYPE)
    norm_g = torch.linalg.vector_norm(z_g, ord=1, dim=-1, keepdim=True, dtype=_NORM_DTYPE)

    # clamp(max=1.0): only shrink z_g towards z_pos's scale when the
    # extrapolation overshot (norm_g > tau * norm_pos); never grow it.
    factor = (tau * (norm_pos + _EPS) / (norm_g + _EPS)).clamp_(max=1.0).to(z_g.dtype)
    z_g.mul_(factor)

    return z_g.mul_(alpha).add_(z_pos, alpha=(1.0 - alpha))


def _make_nag_forward(
    attn: torch.nn.Module,
    orig_forward: Callable[..., torch.Tensor],
    state: NagState,
    modality: str,
) -> Callable[..., torch.Tensor]:
    """Build a NAG-patched replacement for one Attention module's forward.

    This is an equivalent expansion of `Attention.forward`
    (attention.py:180-249) for the cross-attention case ONLY, with the
    RoPE/perturbation-mask branches dropped: cross-attention calls on this
    pipeline never pass pe/k_pe/mask/perturbation_mask/all_perturbed (the
    AdaLN path, transformer.py:392, calls `attn(attn_input, context=...,
    mask=context_mask)` with context_mask always None for text cross-
    attention — confirmed by tracing helpers.py's modality construction).
    `orig_forward` is accepted for signature/documentation symmetry with
    other engine patch closures (e.g. BlockSwapService keeps the original to
    restore on uninstall) but is intentionally never called: an unexpected
    call shape below means NAG's install-time assumptions broke, and this
    module has no uninstall path to fall back to, so failing loud is the
    only safe response (D5).
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
                f"NAG forward ({modality}): received an argument shape this "
                "patch was never designed to handle (context is "
                f"None={context is None}, pe set={pe is not None}, k_pe "
                f"set={k_pe is not None}, mask set={mask is not None}, "
                f"perturbation_mask set={perturbation_mask is not None}, "
                f"all_perturbed={all_perturbed}). This module is only ever "
                "installed on attn2/audio_attn2 text cross-attention, which "
                "never sets these — the call site changed underneath NAG."
            )

        neg_context = state.context_for(modality)
        if neg_context is None:
            # NagService.install only patches modules when state.ready is
            # True, so this can only fire if the state was reset (or a fresh,
            # unpopulated NagState got substituted) after install ran — a
            # wiring bug in the caller, not a normal runtime path.
            raise RuntimeError(
                f"NAG forward ({modality}): negative context is missing at "
                "call time even though this module was patched — NagState "
                "was reset or replaced after NagService.install() ran."
            )

        if neg_context.shape[0] != context.shape[0]:
            # Batch-expand safety net: negative context is encoded once per
            # job (batch size 1) while positive context can be batched (e.g.
            # classifier-free-guidance-free batched T2V requests still batch
            # over multiple prompts/seeds upstream). Broadcasting keeps the
            # same negative prompt applied to every item in the batch.
            neg_context = neg_context.expand(context.shape[0], -1, -1)

        params = state.params
        assert params is not None  # implied by state.ready, checked at install()

        q = attn.q_norm(attn.to_q(x))

        k_pos = attn.k_norm(attn.to_k(context))
        v_pos = attn.to_v(context)
        z_pos = attn.attention_function(q, k_pos, v_pos, attn.heads, None)
        del k_pos, v_pos

        k_neg = attn.k_norm(attn.to_k(neg_context))
        v_neg = attn.to_v(neg_context)
        z_neg = attn.attention_function(q, k_neg, v_neg, attn.heads, None)
        del k_neg, v_neg, q

        out = nag_combine(z_pos, z_neg, params.scale, params.tau, params.alpha)
        del z_pos, z_neg  # free promptly: peak VRAM for this call is 3 tensors, not 1

        # Per-head gating, identical to Attention.forward's tail (attention.py:
        # 237-247) — NAG combine happens BEFORE gating (D4/KJNodes order).
        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(x)  # (B, T, H)
            b, t, _ = out.shape
            out = out.view(b, t, attn.heads, attn.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, attn.heads * attn.dim_head)

        return attn.to_out(out)

    return nag_forward


def _cross_attn_modules(transformer: torch.nn.Module) -> Iterator[tuple[torch.nn.Module, str]]:
    """Yield (attn2, "video") and (audio_attn2, "audio") for every dual-stream
    transformer block, 96 total on the production 48-block model.

    Traversal mirrors block_swap_service._get_blocks: `ledger.transformer()`
    returns an X0Model wrapping the real LTXModel as `velocity_model`, so we
    look there first, then fall back to the object itself (this also lets the
    selfcheck script call this directly against a bare LTXModel/stub without
    the X0Model wrapper).
    """
    inner = getattr(transformer, "velocity_model", None)
    blocks = getattr(inner, "transformer_blocks", None) if inner is not None else None
    if blocks is None:
        blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        return

    for block in blocks:
        attn2 = getattr(block, "attn2", None)
        if attn2 is not None:
            yield attn2, "video"
        audio_attn2 = getattr(block, "audio_attn2", None)
        if audio_attn2 is not None:
            yield audio_attn2, "audio"


class NagService:
    """Patches a freshly-built transformer's text cross-attention modules to
    apply NAG, or does nothing at all when the current job didn't request it.

    Reads NagState through `state_provider()` on every `install()` call
    (rather than capturing a NagState reference at construction time) so one
    long-lived NagService instance always sees the CURRENT job's state, the
    same pattern GGUFQuantLoaderService uses for its IC-LoRA provider.

    No uninstall: `ledger.transformer()` builds a brand new transformer
    instance per job (ModelLedger never caches it — see D1), so a patched
    instance is simply never reused; there is nothing to restore. No double-
    patch guard either, for the same reason (this method never runs twice
    against the same instance).
    """

    def __init__(self, state_provider: Callable[[], NagState]) -> None:
        self._state_provider = state_provider

    def install(self, transformer: torch.nn.Module) -> int:
        state = self._state_provider()
        if not state.requested:
            return 0
        if not state.ready:
            raise RuntimeError(
                "NAG requested for this job but its negative prompt was "
                "never encoded (NagState.ready is False) — encode_negative()/"
                "set_contexts() must complete before ledger.transformer() is "
                "built (D2's ordering guarantee)."
            )

        params = state.params
        assert params is not None  # implied by state.requested

        count = 0
        for attn, modality in _cross_attn_modules(transformer):
            orig_forward = attn.forward
            attn.forward = _make_nag_forward(attn, orig_forward, state, modality)  # type: ignore[method-assign]
            count += 1

        if count == 0:
            raise RuntimeError(
                "NAG requested but _cross_attn_modules found no attn2/"
                "audio_attn2 modules on this transformer — either the "
                "traversal no longer matches the model's structure, or an "
                "empty/wrong object was passed to install()."
            )

        logger.info(
            "NAG installed on %d cross-attention modules "
            "(scale=%.2f tau=%.2f alpha=%.2f negative=%r)",
            count,
            params.scale,
            params.tau,
            params.alpha,
            params.negative_prompt[:60],
        )
        return count
