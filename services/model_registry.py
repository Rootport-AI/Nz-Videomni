"""Category-scoped model registry (model management S1).

Mirrors :mod:`services.lora_registry`: resolves a server-side model NAME (what
``GET /models`` lists and what ``POST /pipeline/load`` accepts in its optional
``models`` block) to a weight file on disk, per fixed category. Names come from
three sources, in priority order:

1. the injected ``"default"`` entry — always present, pointing at the SAME
   fixed default-path field of :class:`config.ModelConfig` that the worker
   payload is built from today, so "no selection" stays byte-identical;
2. explicit ``config.yaml`` registrations (``model.transformers`` /
   ``text_encoders`` / ``video_vaes`` / ``audio_models``, name -> path) — the
   authoritative way to expose a file the scanner cannot classify;
3. directory scanning of the EXISTING layout (no re-organization): each
   category scans the directory its default file lives in, so a fine-tune
   dropped next to the stock weight shows up without any config edit.

Category design (see Docs/MODEL_MANAGEMENT_DESIGN.md §0): the ``audio``
category is ONE file/entry on purpose — the engine sources the audio VAE
decoder/encoder AND the vocoder from the single ``component_audio_vae_path``
file (engine/pipeline/fast_video_pipeline.py ``_install_component_sources``).

Fail loud, no silent skip (lora_registry precedent):
  * path-like name -> MODEL_NOT_FOUND (a name is never treated as a path);
  * unknown category / unknown name -> MODEL_NOT_FOUND (404);
  * registered name whose file is missing on disk -> MODEL_FILE_MISSING (422).
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path

from api.errors import APIError, model_file_missing, model_incompatible, model_not_found
from config import PROJECT_ROOT, AppConfig

logger = logging.getLogger("ltx.models")

#: Reserved name of the injected per-category default entry.
DEFAULT_NAME = "default"


@dataclass(frozen=True)
class CategorySpec:
    """Static wiring of one dropdown category to the existing config layout."""

    default_field: str  # ModelConfig field holding today's fixed default path
    config_field: str  # ModelConfig dict field with explicit registrations
    extensions: tuple[str, ...]  # accepted weight-file extensions
    # How many parents above the DEFAULT FILE the scan roots at. 1 = the file's
    # own directory. The transformer default lives one subdirectory deep
    # (models/ltx-2.3-gguf/LTX-2.3-distilled-1.1/*.gguf) and sibling releases
    # get sibling subdirectories, so it scans the grandparent recursively.
    parent_levels: int = 1
    recursive: bool = False
    # Filename classifier for categories sharing one directory: the video and
    # audio VAEs both live in models/ltx-2.3-components/vae/, so a scanned
    # filename must contain this hint ("video"/"audio") AND not the opposite
    # hint. Ambiguous/unclassifiable files are skipped with a log line; config
    # registration is the authoritative override (design ruling §9-2).
    name_hint: str | None = None


CATEGORY_SPECS: dict[str, CategorySpec] = {
    "transformer": CategorySpec(
        default_field="gguf_transformer_path",
        config_field="transformers",
        extensions=(".gguf",),
        parent_levels=2,
        recursive=True,
    ),
    "text_encoder": CategorySpec(
        default_field="gguf_gemma_path",
        config_field="text_encoders",
        extensions=(".gguf",),
    ),
    "video_vae": CategorySpec(
        default_field="component_video_vae_path",
        config_field="video_vaes",
        extensions=(".safetensors",),
        name_hint="video",
    ),
    "audio": CategorySpec(
        default_field="component_audio_vae_path",
        config_field="audio_models",
        extensions=(".safetensors",),
        name_hint="audio",
    ),
}

#: Fixed category order (dropdown/GET /models order).
CATEGORIES: tuple[str, ...] = tuple(CATEGORY_SPECS)

_OPPOSITE_HINT = {"video": "audio", "audio": "video"}


@dataclass
class ModelEntryInfo:
    """One row of a GET /models category listing."""

    name: str
    path: str  # display path (project-relative when possible; never a bare abs leak)
    is_default: bool
    exists: bool
    source: str  # "config" | "scan"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "is_default": self.is_default,
            "exists": self.exists,
            "source": self.source,
        }


_GGUF_MAGIC = b"GGUF"
# A safetensors JSON header beyond this is implausible for these models and
# more likely a corrupt/foreign file than a real header.
_MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024


def precheck_model_file(category: str, name: str, path: Path) -> None:
    """Cheap compatibility precheck BEFORE the worker is (re)started.

    Reads only a few bytes: GGUF magic for .gguf, and the 8-byte length +
    JSON header parse for .safetensors (weights are never loaded). This stops
    "right extension, wrong file" inputs from reaching the engine's native
    loaders (where they would crash without a friendly error). Deep key/shape
    validation stays with the engine's fail-fast load path (design §5.2).

    Raises ``model_incompatible`` (422) on any failure.
    """
    spec = CATEGORY_SPECS.get(category)
    if spec is None:
        raise model_not_found(
            category, name, detail=f"unknown category; known: {list(CATEGORIES)}"
        )
    suffix = path.suffix.lower()
    if suffix not in spec.extensions:
        raise model_incompatible(
            category,
            name,
            detail=f"expected one of {list(spec.extensions)}, got '{path.suffix}'",
        )
    try:
        with path.open("rb") as fh:
            if suffix == ".gguf":
                magic = fh.read(4)
                if magic != _GGUF_MAGIC:
                    raise model_incompatible(
                        category, name, detail=f"not a GGUF file (magic {magic!r})"
                    )
            else:  # .safetensors
                raw = fh.read(8)
                if len(raw) != 8:
                    raise model_incompatible(
                        category, name, detail="file too small for a safetensors header"
                    )
                (header_len,) = struct.unpack("<Q", raw)
                size = path.stat().st_size
                if header_len == 0 or header_len > size - 8 or header_len > _MAX_SAFETENSORS_HEADER:
                    raise model_incompatible(
                        category,
                        name,
                        detail=f"implausible safetensors header length {header_len}",
                    )
                try:
                    json.loads(fh.read(header_len).decode("utf-8"))
                except Exception as exc:
                    raise model_incompatible(
                        category, name, detail=f"unparsable safetensors header: {exc}"
                    ) from exc
    except APIError:
        raise
    except OSError as exc:
        raise model_incompatible(category, name, detail=f"unreadable file: {exc}") from exc


def _hint_matches(filename: str, hint: str | None) -> bool:
    """Classify a shared-directory filename. No hint -> always accepted."""
    if hint is None:
        return True
    lower = filename.lower()
    return hint in lower and _OPPOSITE_HINT[hint] not in lower


class ModelRegistry:
    """Per-category name -> path registry with existing-layout scanning."""

    def __init__(self, config: AppConfig):
        self.config = config
        # category -> name -> (path-as-configured-or-scanned, source)
        self._registry: dict[str, dict[str, tuple[str, str]]] = {}
        self.rescan()

    # ------------------------------------------------------------------ scan

    def rescan(self) -> None:
        """Rebuild the registry: default entry + config entries + a fresh
        directory scan. Cheap (a few directory listings), so GET /models runs
        it per request — a newly downloaded file appears without a restart."""
        registry: dict[str, dict[str, tuple[str, str]]] = {}
        for category, spec in CATEGORY_SPECS.items():
            entries: dict[str, tuple[str, str]] = {}
            default_path = getattr(self.config.model, spec.default_field) or ""
            entries[DEFAULT_NAME] = (default_path, "config")

            configured: dict[str, str] = getattr(self.config.model, spec.config_field) or {}
            for name, rel in configured.items():
                if name == DEFAULT_NAME:
                    # "default" is reserved for the injected entry (the byte-
                    # identical guarantee); shadowing it would silently change
                    # what "no selection" means. Refuse + log, keep serving.
                    logger.warning(
                        "model.%s: entry name 'default' is reserved (ignored); "
                        "the default always maps to model.%s",
                        spec.config_field,
                        spec.default_field,
                    )
                    continue
                entries[name] = (rel, "config")

            known_paths = {
                self.config._abs(path).resolve()
                for path, _source in entries.values()
                if path
            }
            for found in self._scan_category(spec):
                if found.resolve() in known_paths:
                    continue  # the default / an explicit registration already covers it
                name = found.stem
                if name in entries:
                    name = f"{found.parent.name}__{found.stem}"
                if name in entries:
                    logger.warning(
                        "model scan (%s): name collision for %s (ignored)",
                        category,
                        found,
                    )
                    continue
                entries[name] = (str(found), "scan")

            registry[category] = entries
        self._registry = registry

    def _scan_category(self, spec: CategorySpec) -> list[Path]:
        """Discover weight files for one category in the existing layout."""
        default_path = getattr(self.config.model, spec.default_field) or ""
        if not default_path:
            return []
        base = self.config._abs(default_path)
        for _ in range(spec.parent_levels):
            base = base.parent
        if not base.is_dir():
            return []
        found: list[Path] = []
        candidates = base.rglob("*") if spec.recursive else base.iterdir()
        for path in sorted(candidates):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(base).parts
            # Skip HF download caches (.cache/) and any dot-directories/files.
            if any(part.startswith(".") for part in rel_parts):
                continue
            if path.suffix.lower() not in spec.extensions:
                continue
            if not _hint_matches(path.name, spec.name_hint):
                if spec.name_hint is not None:
                    logger.info(
                        "model scan: %s not classifiable as '%s' by filename "
                        "(register it explicitly in config model.%s to expose it)",
                        path.name,
                        spec.name_hint,
                        spec.config_field,
                    )
                continue
            found.append(path)
        return found

    # ----------------------------------------------------------------- reads

    def categories(self) -> list[str]:
        return list(CATEGORIES)

    def default_name(self, category: str) -> str:
        return DEFAULT_NAME

    def names(self, category: str) -> list[str]:
        """Registered names, ``"default"`` first, the rest sorted."""
        entries = self._registry.get(category, {})
        rest = sorted(n for n in entries if n != DEFAULT_NAME)
        return [DEFAULT_NAME] + rest if DEFAULT_NAME in entries else rest

    def entries(self, category: str) -> list[ModelEntryInfo]:
        """GET /models rows for one category (default first, rest sorted)."""
        entries = self._registry.get(category, {})
        infos: list[ModelEntryInfo] = []
        for name in self.names(category):
            raw, source = entries[name]
            abs_path = self.config._abs(raw) if raw else None
            infos.append(
                ModelEntryInfo(
                    name=name,
                    path=self._display_path(raw),
                    is_default=(name == DEFAULT_NAME),
                    exists=bool(abs_path and abs_path.exists()),
                    source=source,
                )
            )
        return infos

    def _display_path(self, raw: str) -> str:
        """Project-relative display path; an absolute path outside the project
        is reduced to its filename (never leak absolute paths to clients)."""
        if not raw:
            return ""
        p = Path(raw)
        if not p.is_absolute():
            return p.as_posix()
        try:
            return p.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return p.name

    # --------------------------------------------------------------- resolve

    def resolve(self, category: str, name: str) -> Path:
        """Resolve ``(category, name)`` -> absolute weight-file path.

        Raises ``model_not_found`` (404) for an unknown category, a path-like
        name, or an unknown name; ``model_file_missing`` (422) for a registered
        name whose file is absent on disk.
        """
        if category not in self._registry:
            raise model_not_found(
                category, name, detail=f"unknown category; known: {list(CATEGORIES)}"
            )
        if "/" in name or "\\" in name or ".." in name:
            raise model_not_found(category, name, detail="model name must not be a path")
        entry = self._registry[category].get(name)
        if entry is None:
            raise model_not_found(
                category, name, detail=f"known models: {self.names(category)}"
            )
        raw, _source = entry
        if not raw:
            raise model_file_missing(
                category, name, detail="no path configured for this entry"
            )
        path = self.config._abs(raw)
        if not path.exists():
            raise model_file_missing(
                category, name, detail=f"registered model file missing: {path}"
            )
        return path
