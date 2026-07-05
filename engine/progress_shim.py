"""Per-step denoise progress observation for the engine worker (F2, G3 gate).

The installed wheel's denoising loops (``ltx_pipelines/utils/samplers.py``:
``euler_denoising_loop`` and friends) expose NO callback — but they all wrap
their sigma iteration in the module-level ``tqdm`` binding (``from tqdm import
tqdm`` at samplers.py:7). That binding is the single per-step observation
point shared by the single-generate pipeline AND the chain pipeline, so the
worker swaps it for :class:`TqdmShim` at startup (:func:`install`) — the same
module-global monkeypatch mechanism engine/worker.py already uses for
``denoise_audio_video`` and fast_video_pipeline uses for sigma schedules.

STRICT OBSERVATION ONLY: the shim yields the wrapped iterable's items
unchanged, never touches tensors / seeds / schedules, and swallows every
emit-side exception (a progress hiccup must never kill a generation).

Phase coordination
------------------
The wheel loop does not know which pipeline phase invoked it, so callers
declare it here (module-level cooperation, single-threaded worker):

* chain (engine/pipeline/chain_pipeline.py) calls :func:`set_phase`
  explicitly before each stage-1 segment / stage-2 tile denoise, including the
  segment/tile position (``outer_index``/``outer_total``) so the app can
  interpolate an overall fraction.
* single generate (engine/worker.py::_do_generate) calls
  :func:`begin_single_op`; the wheel's ``DistilledPipeline.__call__`` then
  runs its two denoising loops back-to-back with no seam we can hook, so the
  shim infers the phase from the LOOP INVOCATION COUNT within the op: 1st loop
  -> ``stage1_denoise``, 2nd -> ``stage2_denoise``, any further -> the generic
  ``denoise``. This is the documented F2 "phase inference" trade-off: it holds
  for the distilled two-stage pipeline this backend ships and degrades to a
  still-truthful generic label if a future pipeline runs more loops.

This module is deliberately dependency-free (no torch, no ltx_*) so the app
venv's pytest can unit-test the shim; only :func:`install` imports the wheel.
"""

from __future__ import annotations

import time
from typing import Callable

# emit(stage=..., index=..., total=..., it_s=..., outer_index=..., outer_total=...)
_EMIT: Callable[..., None] | None = None

# Current phase label + (outer_index, outer_total) chain context. ``"single"``
# is the begin_single_op sentinel resolved per-loop in _resolve_phase().
_PHASE: str | None = None
_OUTER: tuple[int | None, int | None] = (None, None)
_SINGLE_LOOP_COUNT = 0


def set_emitter(fn: Callable[..., None] | None) -> None:
    """Register the worker's protocol-emit function (None disables emission)."""
    global _EMIT
    _EMIT = fn


def begin_single_op() -> None:
    """Enter single-generate mode: phases are inferred per loop invocation."""
    global _PHASE, _OUTER, _SINGLE_LOOP_COUNT
    _PHASE, _OUTER, _SINGLE_LOOP_COUNT = "single", (None, None), 0


def set_phase(
    phase: str | None,
    outer_index: int | None = None,
    outer_total: int | None = None,
) -> None:
    """Declare the phase of the NEXT denoising-loop invocation (chain path)."""
    global _PHASE, _OUTER
    _PHASE, _OUTER = phase, (outer_index, outer_total)


def end_op() -> None:
    """Clear the phase context after an op (single or chain) finishes."""
    set_phase(None)


def _resolve_phase() -> str:
    global _SINGLE_LOOP_COUNT
    if _PHASE == "single":
        _SINGLE_LOOP_COUNT += 1
        if _SINGLE_LOOP_COUNT == 1:
            return "stage1_denoise"
        if _SINGLE_LOOP_COUNT == 2:
            return "stage2_denoise"
        return "denoise"
    return _PHASE or "denoise"


class TqdmShim:
    """Drop-in for ``tqdm(iterable)`` as the wheel's samplers use it.

    Emits one progress event AFTER each completed step: ``index`` is the
    1-based count of completed steps, ``total`` the step count, ``it_s`` the
    cumulative steps/second (tqdm-equivalent rate). No progress bar is drawn —
    the framed protocol event replaces tqdm's stderr bar entirely.
    """

    def __init__(self, iterable=None, *args, **kwargs):
        # tqdm takes many kwargs (desc/leave/...); all are display-only and
        # deliberately ignored. Positional extras are tolerated the same way.
        self._iterable = iterable

    def __iter__(self):
        iterable = self._iterable if self._iterable is not None else ()
        try:
            total = len(iterable)
        except Exception:
            total = None
        phase = _resolve_phase()
        outer_index, outer_total = _OUTER
        t0 = time.monotonic()
        step = 0
        for item in iterable:
            yield item
            step += 1
            emit = _EMIT
            if emit is None:
                continue
            elapsed = time.monotonic() - t0
            try:
                emit(
                    stage=phase,
                    index=step,
                    total=total if total is not None else step,
                    it_s=round(step / elapsed, 2) if elapsed > 0 else None,
                    outer_index=outer_index,
                    outer_total=outer_total,
                )
            except Exception:
                # Observation must never break generation. The worker-side
                # emitter logs its own failures; a broken emitter object is
                # simply skipped here.
                pass

    def __len__(self):
        return len(self._iterable)

    # Manual-update tqdm API surface: harmless no-ops so any non-iterator use
    # of the patched name cannot crash the wheel.
    def update(self, n=1):  # pragma: no cover - defensive no-op
        return None

    def close(self):  # pragma: no cover - defensive no-op
        return None

    def __enter__(self):  # pragma: no cover - defensive no-op
        return self

    def __exit__(self, *exc):  # pragma: no cover - defensive no-op
        return False


def install(emit_fn: Callable[..., None]) -> None:
    """Patch ``ltx_pipelines.utils.samplers.tqdm`` -> :class:`TqdmShim`.

    Engine-venv only (the wheel import). The wheel FILE is untouched — this
    rebinds the samplers module's ``tqdm`` name at runtime, exactly like the
    worker's existing ``denoise_audio_video`` patch.
    """
    set_emitter(emit_fn)
    import ltx_pipelines.utils.samplers as _samplers

    _samplers.tqdm = TqdmShim
