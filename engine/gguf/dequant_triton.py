"""Job-scoped switch and safety net for the fused GGUF dequantisation kernels.

``quant_service.dequantize_ggml_tensor`` asks this module first and falls back
to its own eager kernels whenever ``dequant()`` returns ``None``. That is the
whole contract: **this module can never make a job fail, only make it faster.**

Why the state lives here and not in the kernel module
-----------------------------------------------------
``quant_service`` imports stdlib and torch only, and a great deal downstream
(the app venv's test suite among other things) depends on that staying true. So
the import of ``triton`` is pushed one module further out, into
``dequant_triton_kernels``, and reached only through ``_kernels()`` - lazily, on
the first call of a job that actually asked for the feature. A machine with no
Triton, a broken wheel or an incompatible CUDA driver therefore pays nothing and
notices nothing except one warning.

The four ways this degrades, all of which end in "eager runs instead"
--------------------------------------------------------------------
  1. The job did not ask for it (``enabled()`` is False).
  2. ``import triton`` fails, or importing the kernels raises. Cached, so the
     import machinery does not re-run on every tensor of every job.
  3. A kernel launch raises. Caught as ``BaseException`` on purpose: a CUDA
     error surfacing as something exotic must still degrade, not propagate.
  4. **The kernel runs and returns wrong numbers.** 1-3 are all "something
     raised", and a hand-written index formula that is off by one raises
     nothing at all - it just quietly poisons the weights. So the first call of
     each quant type in the process is recomputed with the eager kernel and
     compared bit for bit; a mismatch latches the feature off exactly like an
     exception would. Cost: three extra eager calls per process, once.

Any of 2-4 latches for the rest of the job, so the remaining ~1600 tensors of
that forward pass go straight to eager without paying a try/except each. The
latch is cleared by ``set_job()``, so a transient failure does not poison the
worker for the rest of its life.

Echo semantics (``fused_gguf_dequant_kernel_used``), same as block-swap prefetch:
  * ``"off"``    - the job never asked.
  * ``"on"``     - asked, and at least one tensor really went through Triton.
  * ``"on->off"``- asked, and it did not happen: latched by any of 2-4, or the
                   job had no eligible tensor at all. Deliberately NOT split
                   into finer states; "you asked and did not get it" is the one
                   thing an operator needs to see in metadata.json.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# GGML type ids handled here. Re-declared rather than imported from
# quant_service: that module imports this one, so the dependency has to point
# one way only.
_GGML_Q4_K = 12
_GGML_Q5_K = 13
_GGML_Q6_K = 14

# ggml type -> (kernel wrapper name, eager reference name in quant_service)
_DISPATCH = {
    _GGML_Q4_K: ("dequant_q4_k", "_dequant_q4_k"),
    _GGML_Q5_K: ("dequant_q5_k", "_dequant_q5_k"),
    _GGML_Q6_K: ("dequant_q6_k", "_dequant_q6_k"),
}

# ── Per-job state ────────────────────────────────────────────────────────────
# Module globals rather than an object: the call site is a leaf function called
# ~1600 times per forward pass from inside the model's forward, with no job
# context to hand. Same shape as SageState / the prefetch flags.
_REQUESTED = False      # did THIS job ask for the fused kernels?
_LATCHED = False        # has something gone wrong in THIS job?
_CALLS = 0              # tensors this job actually dequantised on Triton
_WARNED = False         # one warning per job, not per tensor
_LAST_USED = "off"      # echo value for the job that just finished

# Quant types whose first-call bit-exactness check has already passed. Process
# lifetime, not job lifetime: the kernels are static code, so re-verifying them
# on every job would be pure cost. Nothing is ever removed - a type that failed
# verification latched the job, and the next job re-runs the check from scratch
# because the type never got in here.
_VERIFIED: set[int] = set()

# Lazily imported kernel module, and a sticky record of a failed import (Python
# does not cache import failures, so without this a missing Triton would re-run
# the whole import machinery on every job).
_KERNEL_MOD: Any = None
_KERNEL_IMPORT_FAILED = False


def set_job(requested: bool) -> None:
    """Arm (or disarm) the feature for the job that is about to run.

    Called OUTSIDE the pipeline's try block so that the matching
    ``reset_job()`` in its ``finally`` always has something to snapshot.
    Clears the latch: a job that failed for a transient reason must not
    disable the feature for every job that follows.
    """
    global _REQUESTED, _LATCHED, _CALLS, _WARNED
    _REQUESTED = bool(requested)
    _LATCHED = False
    _CALLS = 0
    _WARNED = False


def reset_job() -> None:
    """Disarm the feature and freeze this job's echo value for ``last_used()``."""
    global _REQUESTED, _LATCHED, _CALLS, _WARNED, _LAST_USED
    if not _REQUESTED:
        _LAST_USED = "off"
    elif _LATCHED or _CALLS == 0:
        # Asked for, did not happen. _CALLS == 0 covers the "no eligible tensor
        # in the whole job" case (a non-GGUF or non-K-quant model), which is
        # just as much a "you did not get what you asked for" as a latch.
        _LAST_USED = "on->off"
    else:
        _LAST_USED = "on"
    _REQUESTED = False
    _LATCHED = False
    _CALLS = 0
    _WARNED = False


def last_used() -> str:
    """Echo value of the most recently finished job: "off" / "on" / "on->off"."""
    return _LAST_USED


def enabled() -> bool:
    """Is it worth the call site even trying? Cheap enough for the hot path."""
    return _REQUESTED and not _LATCHED


def _warn(msg: str) -> None:
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    logger.warning(
        "Fused GGUF dequant kernel disabled for this job (%s) - falling back to "
        "the eager PyTorch dequant. The job continues, only slower.", msg,
    )


def _latch(msg: str) -> None:
    global _LATCHED
    _LATCHED = True
    _warn(msg)


def _kernels() -> Any:
    """Import ``dequant_triton_kernels`` on first use; ``None`` if it cannot be.

    Kept as a separate function so the selfcheck can replace it with one that
    raises and prove that the whole chain degrades (C5).
    """
    global _KERNEL_MOD, _KERNEL_IMPORT_FAILED
    if _KERNEL_MOD is not None or _KERNEL_IMPORT_FAILED:
        return _KERNEL_MOD
    try:
        from engine.gguf import dequant_triton_kernels as mod
    except BaseException as exc:  # noqa: BLE001 - any import failure degrades
        _KERNEL_IMPORT_FAILED = True
        logger.warning("Triton dequant kernels unavailable: %s: %s", type(exc).__name__, exc)
        return None
    _KERNEL_MOD = mod
    return mod


def _verify(raw: torch.Tensor, ggml_type: int, original_shape: Any, got: torch.Tensor) -> bool:
    """Recompute with the eager kernel and compare bf16 bit patterns.

    ``quant_service`` is imported here, not at module scope: it imports THIS
    module, so a top-level import would be a cycle. By the time this runs it is
    long since in ``sys.modules`` - the only caller is the call site inside
    ``dequantize_ggml_tensor`` itself.

    The comparison is on the int16 reinterpretation rather than ``torch.equal``
    on bf16 so that a NaN produced identically by both paths still counts as
    equal (random GGUF-shaped test data does produce them).
    """
    from engine.gguf import quant_service as qs

    eager = getattr(qs, _DISPATCH[ggml_type][1])
    ref = eager(raw, tuple(original_shape), torch.bfloat16)
    return bool(
        torch.equal(
            got.reshape(-1).view(torch.int16),
            ref.reshape(-1).view(torch.int16),
        )
    )


def dequant(
    raw: torch.Tensor,
    ggml_type: int,
    original_shape: Any,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Dequantise on Triton, or return ``None`` and let the caller run eager.

    ``raw`` is the 1-D uint8 payload, exactly what the eager kernels take. The
    caller has already checked ``enabled()``, the type and ``raw.is_cuda``; the
    checks are repeated here anyway because this is also the entry point the
    selfcheck drives directly.
    """
    global _CALLS

    if out_dtype is not torch.bfloat16:
        # Only bf16 is wired up. Every production caller asks for bf16 (it is
        # the transformer's compute dtype); anything else is a diagnostic path
        # and can afford the eager kernel.
        return None
    if not enabled() or ggml_type not in _DISPATCH:
        return None

    try:
        mod = _kernels()
        if mod is None:
            _latch("Triton kernels could not be imported")
            return None
        out = getattr(mod, _DISPATCH[ggml_type][0])(raw, original_shape)
        if ggml_type not in _VERIFIED:
            if not _verify(raw, ggml_type, original_shape, out):
                _latch(f"first-call bit-exactness check failed for ggml type {ggml_type}")
                return None
            _VERIFIED.add(ggml_type)
    except BaseException as exc:  # noqa: BLE001 - a speed feature never kills a job
        _latch(f"{type(exc).__name__}: {exc}")
        return None

    _CALLS += 1
    return out
