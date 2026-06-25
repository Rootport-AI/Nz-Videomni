"""GPU / VRAM information.

torch is NOT a Phase-1 dependency (it ships with the official LTX stack). This
module therefore degrades gracefully: if torch is unavailable or no CUDA device
is present, it reports ``available=False`` instead of raising. The real LTX
runner (Step 7) will make torch importable and these readings become live.
"""

from __future__ import annotations

from typing import Any


def _torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def gpu_available() -> bool:
    torch = _torch()
    return bool(torch and torch.cuda.is_available())


def get_gpu_info() -> dict[str, Any]:
    """Return the spec 7.4 ``gpu`` block. Never raises."""
    torch = _torch()
    if not torch or not torch.cuda.is_available():
        return {
            "available": False,
            "name": None,
            "vram_total_mb": 0,
            "vram_used_mb": 0,
            "vram_free_mb": 0,
        }

    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total = int(props.total_memory)
        free, _total = torch.cuda.mem_get_info(idx)
        used = total - int(free)
        return {
            "available": True,
            "name": props.name,
            "vram_total_mb": total // (1024 * 1024),
            "vram_used_mb": used // (1024 * 1024),
            "vram_free_mb": int(free) // (1024 * 1024),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "available": True,
            "name": "unknown",
            "vram_total_mb": 0,
            "vram_used_mb": 0,
            "vram_free_mb": 0,
            "error": str(exc),
        }


def peak_vram_mb() -> int | None:
    """Peak allocated VRAM since the last reset, in MB, or None if no GPU."""
    torch = _torch()
    if not torch or not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_allocated() // (1024 * 1024))
    except Exception:  # pragma: no cover - defensive
        return None


def reset_peak_vram() -> None:
    """Reset the CUDA peak-memory counter before a generation, if possible."""
    torch = _torch()
    if torch and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # pragma: no cover - defensive
            pass
