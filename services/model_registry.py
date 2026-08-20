"""Category-scoped model registry (model management S1, multi-engine P3a).

Mirrors :mod:`services.lora_registry`: resolves a server-side model NAME (what
``GET /models`` lists and what ``POST /pipeline/load`` accepts in its optional
``models`` block) to a weight file on disk, per category, per BASE MODEL.

Base models and their categories come from the JSON descriptors in
``config.model.manifest_dir`` (:mod:`services.base_models`) — the registry
itself hard-codes nothing about LTX 2.3's layout any more. Every descriptor
path is relative to ``config.model.models_dir``. Names within one
(base model, category) come from three sources, in priority order:

1. the injected ``"default"`` entry — always present, pointing at the
   descriptor's ``default_file`` for that category, which is the VERBATIM
   transcription of the fixed default path the worker payload is built from
   today, so "no selection" stays byte-identical;
2. explicit ``config.yaml`` registrations (``model.transformers`` /
   ``text_encoders`` / ``video_vaes`` / ``audio_models``, name -> path) — the
   authoritative way to expose a file the scanner cannot classify. These maps
   have no base-model axis, so they apply to the ACTIVE base model only (P3a:
   the first descriptor; the pipeline-driven active base arrives with the API
   axis in a later phase);
3. directory scanning of every ``scan`` root the descriptor declares for that
   category, so a fine-tune dropped next to the stock weight shows up without
   any config edit. Multiple roots per category are supported, which is what
   lets two base models list their own transformers side by side.

Category design (see Docs/MODEL_MANAGEMENT_DESIGN.md §0): the ``audio``
category is ONE file/entry on purpose — the engine sources the audio VAE
decoder/encoder AND the vocoder from the single ``component_audio_vae_path``
file (engine/pipeline/fast_video_pipeline.py ``_install_component_sources``).

Fail loud, no silent skip (lora_registry precedent):
  * path-like name -> MODEL_NOT_FOUND (a name is never treated as a path);
  * unknown base model / category / name -> MODEL_NOT_FOUND (404);
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
from services.base_models import BaseModelDescriptor, CategoryDescriptor, load_base_models
from services.gguf_kv import GgufParseError, read_gguf_kv

logger = logging.getLogger("ltx.models")

#: Reserved name of the injected per-category default entry.
DEFAULT_NAME = "default"

#: Fixed category order (dropdown/GET /models order). A LITERAL, not derived
#: from a descriptor: three modules import it to validate request categories
#: and to build the legacy two-layer ``GET /models`` block, and that contract
#: is the 4 categories LTX 2.3 declares. Making the SET of categories itself
#: descriptor-driven is deliberately out of scope until a second base model
#: actually needs a different set (Docs/MULTI_ENGINE_DESIGN.md §4.1, S-6).
CATEGORIES: tuple[str, ...] = ("transformer", "text_encoder", "video_vae", "audio")

#: Category -> the ``config.model`` field holding explicit name->path
#: registrations for it. This map is config's, not a descriptor's: the config
#: sections have no base-model axis (see the module docstring, source 2).
CONFIG_REGISTRATION_FIELDS: dict[str, str] = {
    "transformer": "transformers",
    "text_encoder": "text_encoders",
    "video_vae": "video_vaes",
    "audio": "audio_models",
}

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


# A safetensors JSON header beyond this is implausible for these models and
# more likely a corrupt/foreign file than a real header.
_MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024

#: The weight-file types the precheck knows how to look inside. NOT a
#: category->extension table (that is the descriptor's job) — just "which
#: structural check does this suffix get".
_STRUCTURAL_CHECKS: frozenset[str] = frozenset({".gguf", ".safetensors"})


#: The KV keys the GGUF precheck lifts out of a transformer header, and hands
#: back to the caller for the engine-generation ruling (§2.2's two-step
#: contract: ``general.architecture`` = engine family, ``model_version`` =
#: variant). Reading them costs one header scan that already happens here, so
#: the caller never re-opens the file. WHAT they mean is the engine adapter's
#: business (``services.engines.ltx.adapter.check_kv``), not the registry's.
GGUF_ENGINE_KV_KEYS = frozenset({"general.architecture", "model_version"})


def precheck_model_file(
    category: str,
    name: str,
    path: Path,
    *,
    descriptor: CategoryDescriptor | None = None,
) -> dict[str, str]:
    """Cheap compatibility precheck BEFORE the worker is (re)started.

    Reads only a few bytes: the GGUF header's KV section for .gguf, and the
    8-byte length + JSON header parse for .safetensors (weights are never
    loaded). This stops "right extension, wrong file" inputs from reaching the
    engine's native loaders (where they would crash without a friendly error).
    Deep key/shape validation stays with the engine's fail-fast load path
    (design §5.2).

    ``descriptor`` (KEYWORD-ONLY, additive) is the CATEGORY descriptor of the
    base model the file is being selected for. Given one, the file's extension
    must be one this category accepts — the "a .safetensors cannot be the
    transformer" gate, which only a descriptor can state. Without one there is
    no category gate at all (the caller has not said which base model it means,
    and this module refuses to guess with a hard-coded table): the file's own
    suffix then picks the structural check, and any other suffix is rejected.

    Returns the GGUF KV metadata read along the way (``{}`` for safetensors and
    for a GGUF that declares none of :data:`GGUF_ENGINE_KV_KEYS`). Raises
    ``model_incompatible`` (422) on any failure, including a malformed GGUF
    header (:class:`services.gguf_kv.GgufParseError`).
    """
    suffix = path.suffix.lower()
    if descriptor is not None and suffix not in descriptor.extensions:
        raise model_incompatible(
            category,
            name,
            detail=f"expected one of {list(descriptor.extensions)}, got '{path.suffix}'",
        )
    if suffix not in _STRUCTURAL_CHECKS:
        raise model_incompatible(
            category,
            name,
            detail=f"unsupported weight-file type '{path.suffix}' "
            f"(expected one of {sorted(_STRUCTURAL_CHECKS)})",
        )
    try:
        if suffix == ".gguf":
            return _precheck_gguf(category, name, path)
        _precheck_safetensors(category, name, path)
        return {}
    except APIError:
        raise
    except OSError as exc:
        raise model_incompatible(category, name, detail=f"unreadable file: {exc}") from exc


def _precheck_gguf(category: str, name: str, path: Path) -> dict[str, str]:
    """Parse the GGUF header far enough to read the engine-selection KV.

    Supersedes the old 4-byte magic sniff: :func:`~services.gguf_kv.read_gguf_kv`
    checks the magic itself and then walks only the KV section (never the
    tensor data), so the stronger check costs the same single header read.
    """
    try:
        return read_gguf_kv(path, set(GGUF_ENGINE_KV_KEYS))
    except GgufParseError as exc:
        raise model_incompatible(
            category, name, detail=f"not a readable GGUF file: {exc}"
        ) from exc


def _precheck_safetensors(category: str, name: str, path: Path) -> None:
    with path.open("rb") as fh:
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


def _hint_matches(filename: str, hint: str | None) -> bool:
    """Classify a shared-directory filename. No hint -> always accepted."""
    if hint is None:
        return True
    lower = filename.lower()
    return hint in lower and _OPPOSITE_HINT[hint] not in lower


class ModelRegistry:
    """Base-model -> category -> name -> path registry, descriptor-driven."""

    def __init__(
        self,
        config: AppConfig,
        base_models: dict[str, BaseModelDescriptor] | None = None,
    ):
        self.config = config
        # Injected by AppContext (loaded once at startup); loaded here when the
        # registry is built standalone (tests, tools).
        self.base_models: dict[str, BaseModelDescriptor] = (
            base_models if base_models is not None else load_base_models(config.manifest_dir)
        )
        if not self.base_models:
            raise RuntimeError("no base-model descriptors available for the model registry")
        # base model id -> category -> name -> (raw path, source)
        self._registry: dict[str, dict[str, dict[str, tuple[str, str]]]] = {}
        self.rescan()

    # ------------------------------------------------------------ base models

    @property
    def base_model_ids(self) -> list[str]:
        """Declared base-model ids, in manifest-filename order."""
        return list(self.base_models)

    @property
    def default_base_model(self) -> str:
        """The base model used when a caller names none — the first descriptor.

        P3a: the pipeline has no base-model axis yet, so "first" IS "active".
        """
        return next(iter(self.base_models))

    def descriptor(self, base_model: str | None = None) -> BaseModelDescriptor:
        return self.base_models[self._base_id(base_model)]

    def _base_id(self, base_model: str | None) -> str:
        if base_model is None:
            return self.default_base_model
        if base_model not in self.base_models:
            raise model_not_found(
                "base_model",
                base_model,
                detail=f"unknown base model; known: {self.base_model_ids}",
            )
        return base_model

    # ------------------------------------------------------------------ paths

    def _models_root(self) -> Path:
        return self.config.models_dir

    def _models_abs(self, rel: str) -> Path:
        """Descriptor-relative path -> absolute path under the model store."""
        return self.config._abs((Path(self.config.model.models_dir) / rel).as_posix())

    def _store_path(self, rel: str) -> str:
        """The RAW string stored in the registry for a descriptor-relative path.

        Normalized to ``<models_dir>/<rel>`` with POSIX separators, i.e.
        ``models/LTX23/Weights/....gguf`` for the shipped ``models_dir``. That
        keeps :meth:`_display_path` emitting exactly the project-relative
        strings clients have always seen, while ``config._abs`` still resolves
        it (an absolute ``models_dir`` — as tests use — simply yields an
        absolute raw path, which displays as a bare filename, as before).
        """
        return (Path(self.config.model.models_dir) / rel).as_posix()

    def _store_scanned(self, path: Path) -> str:
        """Same normalization for a path found by scanning: expressed relative
        to the model store when it lives there, verbatim otherwise."""
        try:
            rel = path.relative_to(self._models_root())
        except ValueError:
            return str(path)
        return self._store_path(rel.as_posix())

    # ------------------------------------------------------------------- scan

    def rescan(self) -> None:
        """Rebuild the registry: for every base model, per declared category,
        the default entry + config entries + a fresh directory scan of all its
        scan roots. Cheap (a few directory listings), so GET /models runs it per
        request — a newly downloaded file appears without a restart."""
        active_base = self.default_base_model
        registry: dict[str, dict[str, dict[str, tuple[str, str]]]] = {}
        for base_id, descriptor in self.base_models.items():
            per_category: dict[str, dict[str, tuple[str, str]]] = {}
            for category, spec in descriptor.categories.items():
                entries: dict[str, tuple[str, str]] = {}
                # The default entry always exists, even for a category whose
                # descriptor declares no default_file yet (a base model that is
                # only partly published): it then resolves to MODEL_FILE_MISSING
                # instead of vanishing from the listing.
                entries[DEFAULT_NAME] = (
                    self._store_path(spec.default_file) if spec.default_file else "",
                    "config",
                )

                config_field = CONFIG_REGISTRATION_FIELDS.get(category)
                if config_field and base_id == active_base:
                    configured: dict[str, str] = (
                        getattr(self.config.model, config_field, None) or {}
                    )
                    for name, rel in configured.items():
                        if name == DEFAULT_NAME:
                            # "default" is reserved for the injected entry (the
                            # byte-identical guarantee); shadowing it would
                            # silently change what "no selection" means. Refuse
                            # + log, keep serving.
                            logger.warning(
                                "model.%s: entry name 'default' is reserved (ignored); "
                                "the default always maps to the descriptor's default_file",
                                config_field,
                            )
                            continue
                        entries[name] = (rel, "config")

                known_paths = {
                    self.config._abs(path).resolve()
                    for path, _source in entries.values()
                    if path
                }
                for found in self._scan_category(category, spec):
                    if found.resolve() in known_paths:
                        continue  # the default / an explicit registration covers it
                    name = found.stem
                    if name in entries:
                        name = f"{found.parent.name}__{found.stem}"
                    if name in entries:
                        logger.warning(
                            "model scan (%s/%s): name collision for %s (ignored)",
                            base_id,
                            category,
                            found,
                        )
                        continue
                    entries[name] = (self._store_scanned(found), "scan")

                per_category[category] = entries
            registry[base_id] = per_category
        self._registry = registry

    def _scan_category(self, category: str, spec: CategoryDescriptor) -> list[Path]:
        """Discover weight files for one category across ALL its scan roots."""
        found: list[Path] = []
        seen: set[Path] = set()
        for root in spec.scan:
            base = self._models_abs(root)
            if not base.is_dir():
                continue
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
                            CONFIG_REGISTRATION_FIELDS.get(category, category),
                        )
                    continue
                if path in seen:  # overlapping scan roots list a file once
                    continue
                seen.add(path)
                found.append(path)
        return found

    # ----------------------------------------------------------------- reads

    def categories(self, *, base_model: str | None = None) -> list[str]:
        """Declared category names of one base model, in declaration order."""
        return list(self.descriptor(base_model).categories)

    def default_name(self, category: str, *, base_model: str | None = None) -> str:
        return DEFAULT_NAME

    def names(self, category: str, *, base_model: str | None = None) -> list[str]:
        """Registered names, ``"default"`` first, the rest sorted."""
        entries = self._registry.get(self._base_id(base_model), {}).get(category, {})
        rest = sorted(n for n in entries if n != DEFAULT_NAME)
        return [DEFAULT_NAME] + rest if DEFAULT_NAME in entries else rest

    def entries(self, category: str, *, base_model: str | None = None) -> list[ModelEntryInfo]:
        """GET /models rows for one category (default first, rest sorted)."""
        entries = self._registry.get(self._base_id(base_model), {}).get(category, {})
        infos: list[ModelEntryInfo] = []
        for name in self.names(category, base_model=base_model):
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

    def default_file_presence(self, *, base_model: str | None = None) -> dict[str, bool]:
        """Category -> "this base model's default file is on disk".

        Drives the installed/partly-installed badge of ``GET /models``: a base
        model whose descriptor declares no ``default_file`` for a category (not
        published yet) counts as missing it, same as a declared-but-absent file.
        """
        descriptor = self.descriptor(base_model)
        return {
            category: bool(spec.default_file)
            and self._models_abs(spec.default_file).exists()
            for category, spec in descriptor.categories.items()
        }

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

    def resolve(self, category: str, name: str, *, base_model: str | None = None) -> Path:
        """Resolve ``(base model, category, name)`` -> absolute weight-file path.

        Raises ``model_not_found`` (404) for an unknown base model, an unknown
        category, a path-like name, or an unknown name; ``model_file_missing``
        (422) for a registered name whose file is absent on disk.
        """
        base_id = self._base_id(base_model)
        per_category = self._registry.get(base_id, {})
        if category not in per_category:
            raise model_not_found(
                category, name, detail=f"unknown category; known: {list(per_category)}"
            )
        if "/" in name or "\\" in name or ".." in name:
            raise model_not_found(category, name, detail="model name must not be a path")
        entry = per_category[category].get(name)
        if entry is None:
            raise model_not_found(
                category,
                name,
                detail=f"known models: {self.names(category, base_model=base_model)}",
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
