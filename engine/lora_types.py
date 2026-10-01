"""LoRA data type ``LoraEntry`` (path, strength, enabled flag).

Nothing in the repository imports this module. The IC-LoRA entries the
engine passes around are ``engine.gguf.ic_lora_common.IcLoraEntry`` tuples.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoraEntry:
    """A single LoRA to load and apply."""
    path: str
    strength: float = 1.0
    enabled: bool = True
