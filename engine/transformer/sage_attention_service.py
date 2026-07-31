"""SageAttention backend for the LTX-2 distilled transformer's attention.

SageAttention (https://github.com/thu-ml/SageAttention) replaces the softmax
attention kernel with an INT8-quantized-QK / FP8-or-FP16-PV one. It is a pure
SPEED optimization: same inputs, same shapes, same output *semantics*, but the
numbers differ by quantization noise, so a sage job and an sdpa job with the
same seed produce videos that differ in fine detail. Measured on the real box
(RTX 4070 Ti SUPER, 720p, VERIFICATION_LOG §43): 1.17x end-to-end, 1.56x on
stage 2, no VRAM increase.

What is patched: ``Attention.attention_function`` — a plain Python attribute on
every ``Attention`` module (attention.py:157), NOT an nn.Module child. Assigning
a non-Module there lands in ``object.__setattr__``, so the module tree, the
state dict and block-swap's per-block ``.to()`` bookkeeping are all untouched.
The production 48-block model has 6 attention modules per block (attn1 / attn2 /
audio_attn1 / audio_attn2 / audio_to_video_attn / video_to_audio_attn) = 288
modules, and ALL of them are candidates — unlike NAG/VSF, which only argue with
the text prompt and therefore only patch the 96 cross-attention modules.

Interaction with NAG/VSF (load-bearing, do not "simplify" away): those services
patch ``Attention.forward`` and their replacement forwards call
``attn.attention_function(...)`` directly (nag_service.py:378/:383,
vsf_service.py:249). So a NAG or VSF job that also asks for sage runs its
positive/negative attention calls through the sage kernel too — including VSF's
concatenated ``[K+; K-]`` / ``[V+; -s*V-]`` tensors. The two patches compose by
construction (different attributes) and install order does not matter.

Discipline, and why it differs from NAG's (D3):
  * NAG changes the OUTPUT the user asked for, so any failure is fail-loud.
  * sage only changes the SPEED, so a job must never die because of it. The
    degradations are, in order of when they are decided:
      1. install-time, static — a module whose ``dim_head`` is outside the
         kernel's supported set is simply never wrapped, so it costs nothing
         at run time (the alternative, a per-call check, would run 288 times
         per block per step for a condition that cannot change).
      2. call-time, dynamic — an attention mask (only the IC-LoRA
         ``attention_strength < 1.0`` path produces one), a non-fp16/bf16
         dtype, or a non-CUDA tensor: fall back to the original callable for
         THAT call only.
      3. call-time, catastrophic — the kernel itself raises. That is latched
         for the whole job (see ``SageState.latch_fallback``): one warning,
         then every remaining call in the job goes straight to sdpa. Without
         the latch a systematically-unhappy kernel would log tens of thousands
         of lines and pay the try/except + reshape cost on every one of them.
  * Whether sage was *actually* used is reported back to the app as
    ``attention_used`` ("sage" / "sage->sdpa"), so a degraded job is visible in
    metadata.json rather than only in a log nobody reads (the "fp8 display-only"
    trap, D3).

Availability is probed once per worker process by ``probe_sage()`` and published
on the ``ready`` event; the app never has to guess.
"""

from __future__ import annotations

import logging
import sys
from typing import Callable, Iterator

import torch

logger = logging.getLogger(__name__)

# The sage CUDA kernels are compiled for these head dims only. This pipeline
# uses 128 for the video streams (TransformerConfig.d_head = attention_head_dim)
# and 64 for the audio streams (audio_attention_head_dim), so in practice every
# module qualifies — the check exists so a future config change degrades to sdpa
# on the odd module instead of raising mid-denoise.
_SAGE_HEAD_DIMS = (64, 128)

# sageattn quantizes FROM half precision; anything else has to take the
# original path. ``v`` carries the authoritative compute dtype here, mirroring
# XFormersAttention's ``memory_efficient_attention(q.to(v.dtype), ...)``
# (attention.py:89).
_SAGE_DTYPES = (torch.float16, torch.bfloat16)

# Resolved-once sage entry point (see _sage_callable) and probe result (see
# probe_sage). Both are module-level caches on purpose: a FAILED import is not
# cached by Python itself, so without these a broken/absent wheel would re-run
# the whole import machinery on every job.
_SAGE_CALL: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None
_PROBE_RESULT: bool | None = None


def _sage_callable() -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return ``f(q, k, v) -> out`` for NHD-layout tensors, importing on first use.

    Raises (rather than returning None) when the wheel is missing or unusable —
    every caller is either ``probe_sage`` (which swallows it) or ``install``
    (which latches and degrades), so the exception never escapes to a job.

    Layout: ``Attention.forward`` hands the attention function q/k/v still in
    the FLAT per-token layout ``(B, S, heads * dim_head)``. Viewing that as
    ``(B, S, H, D)`` is exactly SageAttention's "NHD" and exactly what
    XFormersAttention does (attention.py:65/:90) before calling its kernel —
    i.e. the cheap round trip, no transpose, and a shape this wheel already
    ships a first-class AttentionCallable for. "HND" would additionally cost
    two transposes per call for nothing.
    """
    global _SAGE_CALL
    if _SAGE_CALL is not None:
        return _SAGE_CALL

    from sageattention import sageattn

    # Kernel signatures have churned across sageattention releases. The two
    # kwargs used below have been stable throughout, but check anyway so an
    # incompatible wheel is reported by probe_sage() at load time (-> the UI
    # greys the option out) instead of by a TypeError on job 1.
    try:
        import inspect

        params = inspect.signature(sageattn).parameters
    except (TypeError, ValueError):  # C-implemented / wrapped: nothing to check
        params = None
    if params is not None:
        missing = [name for name in ("tensor_layout", "is_causal") if name not in params]
        if missing:
            raise RuntimeError(
                f"sageattention is installed but its sageattn() does not accept "
                f"{missing} — this build is not compatible with the engine's "
                "NHD call convention."
            )

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return sageattn(q, k, v, tensor_layout="NHD", is_causal=False)

    _SAGE_CALL = call
    return call


def probe_sage() -> bool:
    """Is SageAttention usable in THIS process? Cached; never raises.

    Called by the worker BEFORE the pipeline is constructed and reported on the
    ``ready`` event, so the app can publish `acceleration.sage_available`
    without loading a model. The guard is ``BaseException`` on purpose: a
    mismatched CUDA/torch ABI can surface as ImportError, OSError, SystemError
    or a bare RuntimeError out of the DLL loader, and NONE of those may be
    allowed to take the worker down — an unavailable accelerator is a feature
    being off, not a load failure.

    The failure reason goes to STDERR directly rather than through ``logging``
    so it is captured even if this is called before the worker's logging
    handler is installed (the parent redirects the worker's STDERR to
    logs/ltx_worker.log).
    """
    global _PROBE_RESULT
    if _PROBE_RESULT is not None:
        return _PROBE_RESULT
    try:
        _sage_callable()
        _PROBE_RESULT = True
    except BaseException as exc:  # noqa: BLE001 - see docstring
        print(
            f"[sage] SageAttention unavailable, 'sage' requests will run on SDPA: {exc!r}",
            file=sys.stderr,
            flush=True,
        )
        _PROBE_RESULT = False
    return _PROBE_RESULT


class SageState:
    """Which attention backend one job asked for, and what it actually got.

    Lifetime mirrors ``NagState``: one instance lives on the pipeline
    (``LTXFastVideoPipeline._sage``) for as long as the process is resident, and
    ``set_backend``/``reset`` scope it to a single generate()/generate_chain()
    call so a keep_resident worker never leaks one job's backend choice into the
    next job.

    Only a backend string is held — deliberately no ``SageParams`` dataclass
    (there is exactly one knob and no room for a second: kernel selection is
    sageattn's own per-GPU dispatch).
    """

    __slots__ = ("_backend", "_latched", "_last_attention_used")

    def __init__(self) -> None:
        self._backend = "sdpa"
        self._latched = False
        self._last_attention_used = "sdpa"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def requested(self) -> bool:
        """This job asked for sage. ``SageAttentionService.install`` returns 0
        immediately when this is False, which is what keeps an sdpa job
        completely untouched by this feature."""
        return self._backend == "sage"

    @property
    def latched(self) -> bool:
        """A sage kernel call raised earlier in this job — everything from that
        point on runs on the original callable."""
        return self._latched

    @property
    def use_sage(self) -> bool:
        """The single gate the per-call hot path consults.

        Covers BOTH ways sage can be off at call time: the job never asked for
        it, or a kernel failure latched the job onto sdpa. Checking "requested"
        here and not just "latched" is the leak guard: the transformer is
        rebuilt per job, so a wrapper should never survive into a job that did
        not ask for sage — but if that invariant ever breaks, this makes the
        stale wrapper inert instead of silently accelerating (and mis-reporting)
        an sdpa job. It costs one string compare against a matmul.
        """
        return self._backend == "sage" and not self._latched

    @property
    def attention_used(self) -> str:
        """What the CURRENT job is running on: "sdpa", "sage", or "sage->sdpa"
        (sage was requested but the job degraded)."""
        if self._backend != "sage":
            return "sdpa"
        return "sage->sdpa" if self._latched else "sage"

    @property
    def last_attention_used(self) -> str:
        """``attention_used`` as of the last ``reset()``.

        The pipeline resets this state in its own ``finally``, so by the time
        the worker gets control back the live value is already cleared. This
        snapshot is what the worker reports on the ``done`` event — the
        judging criterion for the real-device gates is that metadata field,
        not a log line.
        """
        return self._last_attention_used

    def set_backend(self, backend: str) -> None:
        """Start a job on ``backend``. Never raises (see
        ``LTXFastVideoPipeline._set_sage_job`` for why that matters): an
        unrecognised value degrades to "sdpa" here, because the worker has
        already rejected unknown values fail-loud at the protocol edge and a
        second, later, exception-throwing gate would only be able to fire from
        outside the pipeline's try/finally."""
        self._backend = "sage" if backend == "sage" else "sdpa"
        self._latched = False

    def latch_fallback(self, reason: str) -> None:
        """Give up on sage for the remainder of this job, warning exactly once."""
        if self._latched:
            return
        self._latched = True
        logger.warning(
            "SageAttention degraded to SDPA for the rest of this job (%s). "
            "The job continues normally, just without the speedup.",
            reason,
        )

    def reset(self) -> None:
        """Resident-worker leak guard, called from the pipeline's try/finally.
        Snapshots ``attention_used`` first — that is the only record of what the
        finished job actually ran on."""
        self._last_attention_used = self.attention_used
        self._backend = "sdpa"
        self._latched = False


class _SageAttentionFunction:
    """Drop-in ``AttentionCallable`` that routes to sage when it is safe to.

    Not an ``nn.Module`` (see this module's docstring). Every rejection calls
    ``self._fallback`` — the ORIGINAL callable found on the module, normally
    ``AttentionFunction.DEFAULT`` which resolves to ``PytorchAttention`` in this
    venv — with the untouched q/k/v/mask, so a fallback call is bit-identical to
    an sdpa run. That is why the NHD views below are bound to NEW names: rebinding
    q/k/v would hand the fallback reshaped tensors and silently change its result.
    """

    __slots__ = ("_fallback", "_sage", "_state", "_name", "_dim_head")

    def __init__(
        self,
        fallback: Callable[..., torch.Tensor],
        sage: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        state: SageState,
        name: str,
        dim_head: int,
    ) -> None:
        self._fallback = fallback
        self._sage = sage
        self._state = state
        self._name = name
        self._dim_head = dim_head

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # State gate first: after a kernel failure (or on a stale wrapper that
        # outlived its job) this is the only branch that runs. See
        # SageState.use_sage.
        if not self._state.use_sage:
            return self._fallback(q, k, v, heads, mask)

        # (a) Any attention mask. sageattn takes ``is_causal`` and nothing else
        #     — it cannot express an additive/boolean bias. On this pipeline a
        #     mask appears only on the IC-LoRA
        #     ConditioningItemAttentionStrengthWrapper path
        #     (conditioning_attention_strength < 1.0).
        if mask is not None:
            return self._fallback(q, k, v, heads, mask)

        # (b) dtype: the kernel quantizes from fp16/bf16 only.
        if v.dtype not in _SAGE_DTYPES:
            return self._fallback(q, k, v, heads, mask)

        # (c) CUDA only.
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            return self._fallback(q, k, v, heads, mask)

        # (d) The head split must be the one this wrapper was installed for.
        #     ``heads`` comes from the caller (``self.heads`` in
        #     Attention.forward, ``attn.heads`` in the NAG/VSF forwards), so this
        #     can only disagree if a call site starts passing a different split
        #     than the module was built with — cheap insurance, and it keeps the
        #     supported-head_dim decision entirely at install time.
        b, _, inner = q.shape
        if inner != heads * self._dim_head:
            return self._fallback(q, k, v, heads, mask)

        # reshape (not view): RoPE can hand back a non-contiguous q/k, and
        # reshape degrades to a contiguous copy instead of raising.
        dim_head = self._dim_head
        qh = q.reshape(b, -1, heads, dim_head)
        kh = k.reshape(b, -1, heads, dim_head)
        vh = v.reshape(b, -1, heads, dim_head)
        # Mirror XFormersAttention's ``q.to(v.dtype)`` (attention.py:89): v holds
        # the compute dtype, q/k may still be in the norm's dtype.
        if qh.dtype != vh.dtype:
            qh = qh.to(vh.dtype)
        if kh.dtype != vh.dtype:
            kh = kh.to(vh.dtype)
        if not qh.is_contiguous():
            qh = qh.contiguous()
        if not kh.is_contiguous():
            kh = kh.contiguous()
        if not vh.is_contiguous():
            vh = vh.contiguous()

        try:
            out = self._sage(qh, kh, vh)
        except Exception as exc:  # noqa: BLE001 - a kernel reject must not kill the job
            self._state.latch_fallback(f"{self._name} raised {exc!r}")
            return self._fallback(q, k, v, heads, mask)

        # NHD out -> flat (B, S, H*D): the identical unwind XFormersAttention
        # uses (attention.py:90).
        return out.reshape(b, -1, heads * dim_head)


def _attention_modules(transformer: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Module]]:
    """Yield every ``ltx_core`` ``Attention`` module on a built transformer.

    ``named_modules()`` rather than NAG's attribute walk (``block.attn2`` /
    ``block.audio_attn2``): sage applies to ALL six attention kinds per block,
    and enumerating them by name would silently miss a seventh if the model
    ever grows one — exactly the failure mode this feature cannot afford (a
    partially-accelerated run still reports "sage").

    The ltx_core import is local so this module can be imported without the
    wheel present (the app venv never has it).
    """
    from ltx_core.model.transformer.attention import Attention

    for name, module in transformer.named_modules():
        if isinstance(module, Attention):
            yield name, module


class SageAttentionService:
    """Swaps every eligible ``attention_function`` for the sage kernel, or does
    nothing at all when the current job didn't ask for sage.

    Reads ``SageState`` through ``state_provider()`` on every ``install()`` call
    (rather than capturing it at construction) so one long-lived service always
    sees the CURRENT job's state — the same pattern NagService and
    GGUFQuantLoaderService use.

    No uninstall and no double-patch guard, for NAG's reason: ``ledger.
    transformer()`` builds a brand new transformer per job, so a patched
    instance is never reused. (A chain job builds several, which is why the
    kernel-failure latch lives on the shared state and not on the wrapper.)
    """

    def __init__(self, state_provider: Callable[[], SageState]) -> None:
        self._state_provider = state_provider

    def install(self, transformer: torch.nn.Module) -> int:
        state = self._state_provider()
        if not state.requested:
            # The zero-overhead-when-off gate: an sdpa job leaves this method
            # having touched nothing at all.
            return 0
        if state.latched:
            # A previous transformer build in this same job already hit a kernel
            # failure (chain path: one build per segment/stage). Don't wrap what
            # we already know will fall back on every call.
            return 0

        try:
            sage_call = _sage_callable()
        except BaseException as exc:  # noqa: BLE001 - availability is never fatal
            # probe_sage() said yes at load time but the import is failing now
            # (partially-installed wheel, GPU driver swapped under a resident
            # worker...). Degrade rather than kill the job; the latch makes this
            # surface as attention_used="sage->sdpa".
            state.latch_fallback(f"sageattention became unusable: {exc!r}")
            return 0

        count = 0
        skipped = 0
        for name, module in _attention_modules(transformer):
            # Static condition, decided once here instead of 8 steps x 48 blocks
            # x N tokens times at run time.
            if int(module.dim_head) not in _SAGE_HEAD_DIMS:
                skipped += 1
                continue
            module.attention_function = _SageAttentionFunction(  # type: ignore[assignment]
                fallback=module.attention_function,
                sage=sage_call,
                state=state,
                name=name,
                dim_head=int(module.dim_head),
            )
            count += 1

        if count == 0:
            # Unlike every other degradation in this module, this one is
            # fail-loud: it cannot be caused by a runtime condition, only by the
            # traversal no longer matching the model (or a wrong object being
            # passed in). Continuing would report attention_used="sage" for a
            # run that never touched the kernel — the exact "accelerator that
            # only exists in the UI" trap this feature is designed to avoid.
            raise RuntimeError(
                "SageAttention requested but ZERO eligible Attention modules "
                f"were found on this transformer ({skipped} were skipped for an "
                "unsupported head_dim) — the swap would have been a silent no-op."
            )

        logger.info(
            "SageAttention installed on %d attention modules (%d skipped: head_dim not in %s)",
            count,
            skipped,
            _SAGE_HEAD_DIMS,
        )
        return count
