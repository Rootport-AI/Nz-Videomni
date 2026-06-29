"""Importable, idempotent monkeypatch: CPU-RAM cache for the merged Gemma sd.

Phase 5(A) reuse fix. Root cause of the exit-139 reuse crash is that a persistent
worker rebuilds the Gemma text encoder per job, re-reading the mmap'd base
safetensors and re-dequantizing the GGUF every time. This patch wraps
``GemmaGGUFQuantStateDictLoader.load`` so the heavy disk-mmap + dequant merge runs
EXACTLY ONCE (job 1, forced ``device=cpu`` so the full merged dict lands in CPU
RAM via the proven path); jobs 2+ skip ``.load()`` entirely and just move fresh
copies of the cached CPU tensors onto the requested device.

The cached CPU dict and the held CPU embedding are shared READ-ONLY across jobs
(never mutated); each job gets its own GPU copies via ``.to(device)``, so
``GGMLQuantizedTensor.to(cuda)`` (which re-attaches ``_ggml_type`` /
``_float_shape``) keeps the quant subclass intact.

Mirrors the ``sft_loader_safe_patch`` pattern so BOTH the persistent worker and
the ``outputs/phase5b_diag/reuse_loop.py`` verification harness install the EXACT
same cache code via one importable apply-function:

    from services.gemma_sd_cache import apply_gemma_sd_cache_patch
    apply_gemma_sd_cache_patch()
"""

from __future__ import annotations

import sys

_APPLIED = False

# Single-entry CPU cache: {"key": tuple(path), "sd": {cpu tensors}, "embed": held_embed_cpu}
_gemma_sd_cache: dict | None = None


def _clog(msg: str) -> None:
    """Cache MISS/HIT logging -> STDERR (grep-able by the reuse_loop harness)."""
    print(f"[gemma_sd_cache] {msg}", file=sys.stderr, flush=True)


def apply_gemma_sd_cache_patch() -> bool:
    """Monkeypatch GemmaGGUFQuantStateDictLoader.load with a CPU-RAM cache.

    Idempotent: a module-level ``_APPLIED`` guard makes a second call a no-op
    returning the same status. Returns True if applied (or already applied),
    False on failure.
    """
    global _APPLIED
    if _APPLIED:
        return True
    try:
        import torch
        import services.gemma_gguf_quant_service as _gemma_svc
        from ltx_core.loader.primitives import StateDict

        _GemmaLoader = _gemma_svc.GemmaGGUFQuantStateDictLoader
        _orig_gemma_load = _GemmaLoader.load

        def _gemma_load_cached(self, path, sd_ops=None, device=None):
            """Cached wrapper around GemmaGGUFQuantStateDictLoader.load (CPU-RAM cache)."""
            global _gemma_sd_cache

            target_device = device or torch.device("cpu")
            key = tuple(path) if isinstance(path, list) else (path,)

            if _gemma_sd_cache is None or _gemma_sd_cache.get("key") != key:
                # MISS: build the full merged dict ONCE in CPU RAM via the proven path.
                cpu = torch.device("cpu")
                result = _orig_gemma_load(self, path, sd_ops=sd_ops, device=cpu)
                # Snapshot the CPU merged dict + the held CPU embedding (Lever 3 offload).
                _gemma_sd_cache = {
                    "key": key,
                    "sd": dict(result.sd),
                    "embed": self.held_embed_cpu,
                }
                _clog("gemma sd cache: MISS -> cached CPU merged dict")
            else:
                _clog("gemma sd cache: HIT -> no disk/dequant")

            # Restore the held CPU embedding onto this loader instance (used post-build
            # by the service to re-tie embed_tokens/lm_head on CPU). Shared read-only.
            self.held_embed_cpu = _gemma_sd_cache["embed"]

            cached_sd = _gemma_sd_cache["sd"]
            if target_device.type == "cpu":
                # Shallow-copy the dict so callers can't mutate our cache mapping; the
                # tensors themselves stay shared (CPU, read-only).
                merged = dict(cached_sd)
            else:
                # Per-job fresh GPU copies; never mutate the cached CPU tensors. The
                # GGMLQuantizedTensor.to override preserves the subclass + quant metadata.
                merged = {}
                for k, v in cached_sd.items():
                    if isinstance(v, torch.Tensor) and v.device != target_device:
                        merged[k] = v.to(target_device)
                    else:
                        merged[k] = v

            return StateDict(
                sd=merged,
                device=target_device,
                size=sum(_gemma_svc._safe_numel(v) for v in merged.values()),
                dtype={torch.uint8, torch.bfloat16},
            )

        _GemmaLoader.load = _gemma_load_cached  # type: ignore[assignment]
        _APPLIED = True
        _clog("installed Gemma merged-state_dict CPU cache monkeypatch (Phase 5A reuse fix)")
        return True
    except Exception as exc:  # pragma: no cover
        _clog(f"gemma sd cache patch failed to apply: {exc}")
        return False
