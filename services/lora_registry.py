"""IC-LoRA adapter-name registry (Phase B, extended Phase C).

Resolves a server-side adapter NAME (accepted in ``GenerateRequest.loras[].name``)
to the safetensors file on disk plus its ``preprocess`` kind. The name space is
declared in ``config.yaml`` under ``model.ic_loras`` (name -> project-relative
path, or name -> ``IcLoraEntry`` for Phase C control adapters); this is the ONLY
way a client can select a LoRA — arbitrary filesystem paths are never accepted
(the Pydantic ``LoraSpec`` validator already rejects path-like names, and this
layer also refuses to treat a name as a path).

Fail loud, no silent skip:
  * absent/empty registry section -> any loras request rejected with a clear msg;
  * unknown name -> LORA_NOT_FOUND (404, mirroring IMAGE_NOT_FOUND);
  * registered name whose file is missing on disk -> LORA_NOT_FOUND with detail.
"""

from __future__ import annotations

from pathlib import Path

from api.errors import lora_not_found
from config import AppConfig, IcLoraEntry


class LoraRegistry:
    def __init__(self, config: AppConfig):
        self.config = config
        # name -> project-relative (or absolute) safetensors path (Phase B, string)
        # or an IcLoraEntry (Phase C: path + preprocess kind).
        self.registry: dict[str, str | IcLoraEntry] = dict(config.model.ic_loras or {})

    def names(self) -> list[str]:
        return sorted(self.registry)

    def preprocess_for(self, name: str) -> str:
        """Return the ``preprocess`` kind for a registered ``name`` (no file check).

        ``"none"`` for legacy string entries or an unknown name; the
        ``IcLoraEntry.preprocess`` value for dict entries. Used for metadata
        annotation (``pipeline_manager._write_metadata``), where the name has
        already been resolved successfully by the time the job completes.
        """
        entry = self.registry.get(name)
        if entry is None or isinstance(entry, str):
            return "none"
        return entry.preprocess

    def resolve(self, name: str, strength: float) -> tuple[Path, float, str]:
        """Resolve ``name`` -> ``(safetensors_path, strength, preprocess)``.

        ``preprocess`` is ``"none"`` for legacy string-valued entries (Phase B)
        or the ``IcLoraEntry.preprocess`` value for dict-valued entries (Phase C
        control adapters: ``"canny"`` / ``"dwpose"``).

        Raises ``lora_not_found`` (404) for a path-like name, an empty registry,
        an unknown name, or a registered-but-missing file.
        """
        if "/" in name or "\\" in name or ".." in name:
            raise lora_not_found(name, detail="adapter name must not be a path")
        if not self.registry:
            raise lora_not_found(
                name, detail="no ic_loras registry configured (model.ic_loras is empty)"
            )
        entry = self.registry.get(name)
        if entry is None:
            raise lora_not_found(name, detail=f"known adapters: {self.names()}")
        if isinstance(entry, str):
            rel, preprocess = entry, "none"
        else:
            rel, preprocess = entry.path, entry.preprocess
        path = self.config._abs(rel)
        if not path.exists():
            raise lora_not_found(name, detail=f"registered adapter file missing: {path}")
        return path, float(strength), preprocess
