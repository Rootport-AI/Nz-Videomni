"""IC-LoRA / style-LoRA adapter-name registry (Phase B, extended Phase C, S1).

Resolves a server-side adapter NAME (accepted in ``GenerateRequest.loras[].name``)
to the safetensors file on disk plus its ``preprocess`` kind and an alpha-scaled
strength. Names come from two sources, config authoritative over the scan:

1. ``config.yaml`` ``model.ic_loras`` (name -> project-relative path, or name ->
   ``IcLoraEntry`` for Phase C control adapters) — the authoritative registration;
2. a directory scan of ``config.model.lora_dir`` (S1): every ``*.safetensors``
   there is exposed under its filename stem, so a style/character LoRA dropped in
   the folder shows up without any config edit (mirrors ``services.model_registry``).

Arbitrary filesystem paths are never accepted (the Pydantic ``LoraSpec`` validator
rejects path-like names, and this layer refuses to treat a name as a path).

Per-entry metadata (read from the safetensors header, pure Python — the app venv
has no torch):
  * ``kind`` — ``"control"`` when the header ``__metadata__`` carries
    ``reference_downscale_factor`` OR the config entry declares a non-``"none"``
    ``preprocess`` (union-control / pixel-spatial-upscaler: they derive their
    conditioning from a REFERENCE video); ``"style"`` otherwise (画風/character
    LoRAs that need no reference). Drives the endpoint's reference-required check.
  * ``scale`` — the LoRA alpha/rank convolution factor from ``ss_network_alpha``
    / ``ss_network_dim`` (kohya metadata); folded into the strength returned by
    ``resolve()`` per the user decision (auto-convolution, outside the frozen
    weight-patch mechanism). No metadata -> ``scale=1.0`` (behaviour unchanged;
    the existing IC-LoRA files carry no ``ss_network_alpha``).

Fail loud, no silent skip — but only for CONFIG registrations:
  * config name whose file is missing on disk -> ``LORA_NOT_FOUND`` at resolve;
  * path-like name / unknown name -> ``LORA_NOT_FOUND`` (404).
Directory-scanned files are best-effort: a file with an unreadable/broken
safetensors header is dropped from the scan (never surfaced as a broken adapter),
since the folder is a drop-in path the operator did not explicitly register.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from api.errors import lora_not_found
from config import AppConfig, IcLoraEntry

logger = logging.getLogger("ltx.loras")


class ResolvedLora(NamedTuple):
    """``resolve()``'s return shape. Indices 0-2 (``path``/``strength``/
    ``preprocess``) are positionally compatible with the pre-audio_strength
    3-tuple, so callers that still unpack/index only those stay correct.
    """

    path: Path
    strength: float
    preprocess: str
    audio_strength: float | None

# A safetensors JSON header beyond this is implausible for a LoRA and more likely
# a corrupt/foreign file than a real header (services.model_registry precedent).
_MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024


@dataclass
class LoraEntryInfo:
    """One resolved adapter: a config registration or a directory-scan hit."""

    name: str
    path: Path  # absolute (config._abs of the registered/scanned path)
    preprocess: str  # "none" | "canny" | "dwpose" | "depth"
    kind: str  # "style" | "control"
    scale: float  # alpha/rank convolution factor (1.0 when unknown)
    has_thumbnail: bool  # a sibling <stem>.png exists
    source: str  # "config" | "scan"
    exists: bool

    def as_dict(self) -> dict:
        """GET /loras row (mirrors ModelEntryInfo.as_dict — no path leak)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "has_thumbnail": self.has_thumbnail,
            "exists": self.exists,
            "source": self.source,
        }


def _read_safetensors_header(path: Path) -> dict:
    """Return the parsed safetensors JSON header (pure Python, weights untouched).

    Reads only the 8-byte length + the JSON header (services.model_registry
    ``precheck_model_file`` method). Raises ``ValueError``/``OSError`` on any
    malformed header so callers can decide skip-vs-fail per source.
    """
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError("file too small for a safetensors header")
        (header_len,) = struct.unpack("<Q", raw)
        size = path.stat().st_size
        if header_len == 0 or header_len > size - 8 or header_len > _MAX_SAFETENSORS_HEADER:
            raise ValueError(f"implausible safetensors header length {header_len}")
        header = json.loads(fh.read(header_len).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    return header


def _metadata(header: dict) -> dict:
    md = header.get("__metadata__", {})
    return md if isinstance(md, dict) else {}


def _rank_from_tensors(header: dict) -> float | None:
    """Fallback rank = shape[0] of any ``lora_A`` tensor (kohya down-projection
    ``[rank, in_features]``). Used only when ``ss_network_dim`` is absent."""
    for tensor_name, info in header.items():
        if tensor_name == "__metadata__":
            continue
        if "lora_a" in tensor_name.lower() and isinstance(info, dict):
            shape = info.get("shape")
            if isinstance(shape, list) and shape:
                try:
                    return float(shape[0])
                except (TypeError, ValueError):
                    return None
    return None


def _alpha_scale(header: dict) -> float:
    """LoRA alpha/rank convolution factor from kohya metadata; 1.0 if unknown."""
    md = _metadata(header)
    alpha_raw = md.get("ss_network_alpha")
    if alpha_raw is None:
        return 1.0
    try:
        alpha = float(alpha_raw)
    except (TypeError, ValueError):
        return 1.0
    rank: float | None = None
    dim_raw = md.get("ss_network_dim")
    if dim_raw is not None:
        try:
            rank = float(dim_raw)
        except (TypeError, ValueError):
            rank = None
    if rank is None:
        rank = _rank_from_tensors(header)
    if not rank or rank <= 0:
        return 1.0
    return alpha / rank


class LoraRegistry:
    def __init__(self, config: AppConfig):
        self.config = config
        # name -> project-relative (or absolute) safetensors path (Phase B, string)
        # or an IcLoraEntry (Phase C: path + preprocess kind). Config-authoritative
        # source; kept verbatim so ``preprocess_for`` stays byte-compatible.
        self.registry: dict[str, str | IcLoraEntry] = dict(config.model.ic_loras or {})
        # name -> LoraEntryInfo, rebuilt by rescan() (config entries + scan hits).
        self._entries: dict[str, LoraEntryInfo] = {}
        self.rescan()

    # ------------------------------------------------------------------ scan

    def rescan(self) -> None:
        """Rebuild the entry cache: config registrations (authoritative) merged
        with a fresh ``lora_dir`` scan. Cheap (a directory listing + header reads
        of small JSON blobs), so the API layer runs it per GET /loras — a newly
        dropped-in file appears without a restart (model_registry precedent)."""
        entries: dict[str, LoraEntryInfo] = {}
        known_paths: set[Path] = set()

        # 1. Config registrations win.
        for name, raw in self.registry.items():
            if isinstance(raw, str):
                rel, preprocess = raw, "none"
            else:
                rel, preprocess = raw.path, raw.preprocess
            abs_path = self.config._abs(rel)
            entries[name] = self._build_entry(name, abs_path, preprocess, "config")
            known_paths.add(abs_path.resolve())

        # 2. Directory scan (best-effort; broken files silently dropped).
        for found in self._scan_lora_dir():
            resolved = found.resolve()
            if resolved in known_paths:
                continue  # already covered by a config registration (same file)
            name = found.stem
            if name in entries:
                # Duplicate stem -> retreat to a parent-qualified name
                # (model_registry collision rule).
                name = f"{found.parent.name}__{found.stem}"
            if name in entries:
                logger.warning("lora scan: name collision for %s (ignored)", found)
                continue
            info = self._build_entry(name, found, "none", "scan")
            if info is None:
                continue  # unreadable header -> not surfaced from the scan
            entries[name] = info
            known_paths.add(resolved)

        self._entries = entries

    def _scan_lora_dir(self) -> list[Path]:
        """Flat scan of ``config.model.lora_dir`` for ``*.safetensors`` files."""
        lora_dir = getattr(self.config.model, "lora_dir", None)
        if not lora_dir:
            return []
        base = self.config._abs(lora_dir)
        if not base.is_dir():
            return []
        found: list[Path] = []
        for path in sorted(base.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() != ".safetensors":
                continue
            found.append(path)
        return found

    def _build_entry(
        self, name: str, abs_path: Path, preprocess: str, source: str
    ) -> LoraEntryInfo | None:
        """Build a LoraEntryInfo, reading kind/scale from the header when possible.

        Returns ``None`` (drop it) ONLY for a scan entry whose header is
        unreadable. A config entry is always kept — a missing/broken file falls
        back to preprocess-derived kind + scale 1.0, and resolve() fails loud
        later if the file is absent when a job actually needs it.
        """
        exists = abs_path.exists()
        has_thumbnail = abs_path.with_suffix(".png").exists()
        # preprocess is authoritative for control (a canny/pose registration is a
        # control adapter regardless of what the header says).
        kind = "control" if preprocess != "none" else "style"
        scale = 1.0
        if exists:
            try:
                header = _read_safetensors_header(abs_path)
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                if source == "scan":
                    logger.warning(
                        "lora scan: skipping %s (unreadable header: %s)", abs_path, exc
                    )
                    return None
                logger.warning(
                    "lora '%s': unreadable header for %s (%s); kind/scale from "
                    "registration only",
                    name,
                    abs_path,
                    exc,
                )
            else:
                if "reference_downscale_factor" in _metadata(header):
                    kind = "control"
                scale = _alpha_scale(header)
        return LoraEntryInfo(
            name=name,
            path=abs_path,
            preprocess=preprocess,
            kind=kind,
            scale=scale,
            has_thumbnail=has_thumbnail,
            source=source,
            exists=exists,
        )

    # ----------------------------------------------------------------- reads

    def names(self) -> list[str]:
        return sorted(self._entries)

    def entries(self) -> list[LoraEntryInfo]:
        """GET /loras rows (sorted by name)."""
        return [self._entries[name] for name in self.names()]

    def info(self, name: str) -> LoraEntryInfo:
        """Return the cached :class:`LoraEntryInfo` for ``name`` (404 if unknown).

        Used by the endpoint layer for the kind-aware reference-required check.
        """
        entry = self._entries.get(name)
        if entry is None:
            raise lora_not_found(name, detail=f"known adapters: {self.names()}")
        return entry

    def preprocess_for(self, name: str) -> str:
        """Return the ``preprocess`` kind for a registered ``name`` (no file check).

        ``"none"`` for legacy string entries, scanned files, or an unknown name;
        the ``IcLoraEntry.preprocess`` value for config dict entries. Used for
        metadata annotation (``pipeline_manager._write_metadata``).
        """
        entry = self.registry.get(name)
        if entry is None or isinstance(entry, str):
            return "none"
        return entry.preprocess

    # --------------------------------------------------------------- resolve

    def resolve(
        self, name: str, strength: float, audio_strength: float | None = None
    ) -> ResolvedLora:
        """Resolve ``name`` -> ``ResolvedLora(path, scaled_strength, preprocess,
        scaled_audio_strength)``.

        ``scaled_strength`` is ``strength * scale`` where ``scale`` is the LoRA
        alpha/rank convolution factor read from the header (1.0 when absent, i.e.
        byte-identical to the pre-S1 behaviour for the existing IC-LoRA files).
        ``preprocess`` is ``"none"`` for legacy string entries / scanned files or
        the ``IcLoraEntry.preprocess`` value for config dict entries.
        ``scaled_audio_strength`` mirrors ``scaled_strength`` (same ``scale``
        factor) but stays ``None`` when ``audio_strength`` is ``None`` (video-axis
        follow — the caller didn't ask for an independent audio strength).

        Raises ``lora_not_found`` (404) for a path-like name, an empty registry,
        an unknown name, or a registered-but-missing file.
        """
        if "/" in name or "\\" in name or ".." in name:
            raise lora_not_found(name, detail="adapter name must not be a path")
        entry = self._entries.get(name)
        if entry is None:
            if not self._entries:
                raise lora_not_found(
                    name,
                    detail="no LoRA registry configured (model.ic_loras is empty "
                    "and no files found in model.lora_dir)",
                )
            raise lora_not_found(name, detail=f"known adapters: {self.names()}")
        if not entry.path.exists():
            raise lora_not_found(
                name, detail=f"registered adapter file missing: {entry.path}"
            )
        scaled_audio = (
            None if audio_strength is None else float(audio_strength) * entry.scale
        )
        return ResolvedLora(
            entry.path, float(strength) * entry.scale, entry.preprocess, scaled_audio
        )
