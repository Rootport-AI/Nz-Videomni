"""IC-LoRA adapter-name registry (Phase B).

Resolves a server-side adapter NAME (accepted in ``GenerateRequest.loras[].name``)
to the safetensors file on disk. The name space is declared in ``config.yaml``
under ``model.ic_loras`` (name -> project-relative path); this is the ONLY way a
client can select a LoRA — arbitrary filesystem paths are never accepted (the
Pydantic ``LoraSpec`` validator already rejects path-like names, and this layer
also refuses to treat a name as a path).

Fail loud, no silent skip:
  * absent/empty registry section -> any loras request rejected with a clear msg;
  * unknown name -> LORA_NOT_FOUND (404, mirroring IMAGE_NOT_FOUND);
  * registered name whose file is missing on disk -> LORA_NOT_FOUND with detail.
"""

from __future__ import annotations

from pathlib import Path

from api.errors import lora_not_found
from config import AppConfig


class LoraRegistry:
    def __init__(self, config: AppConfig):
        self.config = config
        # name -> project-relative (or absolute) safetensors path.
        self.registry: dict[str, str] = dict(config.model.ic_loras or {})

    def names(self) -> list[str]:
        return sorted(self.registry)

    def resolve(self, name: str, strength: float) -> tuple[Path, float]:
        """Resolve ``name`` -> ``(safetensors_path, strength)``.

        Raises ``lora_not_found`` (404) for a path-like name, an empty registry,
        an unknown name, or a registered-but-missing file.
        """
        if "/" in name or "\\" in name or ".." in name:
            raise lora_not_found(name, detail="adapter name must not be a path")
        if not self.registry:
            raise lora_not_found(
                name, detail="no ic_loras registry configured (model.ic_loras is empty)"
            )
        rel = self.registry.get(name)
        if rel is None:
            raise lora_not_found(name, detail=f"known adapters: {self.names()}")
        path = self.config._abs(rel)
        if not path.exists():
            raise lora_not_found(name, detail=f"registered adapter file missing: {path}")
        return path, float(strength)
