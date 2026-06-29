"""Runtime policy decisions for forced API mode."""

from __future__ import annotations


def decide_force_api_generations(
    system: str,
    cuda_available: bool,
    vram_gb: int | None,
    total_vram_gb: int | None = None,
) -> bool:
    """Return whether API-only generation must be forced for this runtime.

    total_vram_gb is the sum across all detected GPUs. If provided, it is used
    instead of single-GPU vram_gb so that multi-GPU setups (e.g. 2x 16GB) are
    not incorrectly forced into API mode.
    """
    if system == "Darwin":
        return True

    if system in ("Windows", "Linux"):
        if not cuda_available:
            return True
        # Use combined VRAM if available (multi-GPU), else single-GPU figure.
        effective_vram = total_vram_gb if total_vram_gb is not None else vram_gb
        if effective_vram is None:
            return True
        # Lowered threshold: 8GB minimum for local inference with our
        # VRAM optimisations (block swap, FP8, attention tiling).
        return effective_vram < 8

    # Fail closed for non-target platforms unless explicitly relaxed.
    return True