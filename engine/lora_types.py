"""LoRA data types shared by the fast video pipeline.

Extracted from ``lora_service.py`` so the pipeline can reference the
``LoraEntry`` dataclass without importing the heavy LoRA loading/apply
implementation (which is not exercised by the current T2V/GGUF path).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoraEntry:
    """A single LoRA to load and apply."""
    path: str
    strength: float = 1.0
    enabled: bool = True
