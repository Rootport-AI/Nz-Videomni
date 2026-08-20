"""Base-model descriptors — the single source of truth for what a base model
is made of (multi-engine foundation, Docs/MULTI_ENGINE_DESIGN.md §4).

A descriptor is one ``scripts/manifests/*.json`` file with ``schema: 2`` AND an
``engine_family`` field. Those two conditions together mark a BASE MODEL (LTX
2.3, LTX 2.5, ...). Every other manifest in that directory is a SHARED-ASSET
manifest (``00-preprocessors.json``: schema 1, no ``engine_family``) — it is
installer-only and is silently ignored here, never listed as a base model.

What this module reads from a descriptor:

``categories``
    The set of selectable model categories of THIS base model, in declaration
    order, each with its scan roots (``models_dir``-relative), accepted file
    extensions, recursion flag, shared-directory ``name_hint`` classifier and
    the ``default_file`` that backs the injected ``"default"`` registry entry.
``assets``
    Fixed, non-selectable files the engine needs (tokenizer dir, spatial
    upsampler, ...), ``models_dir``-relative. Read by the engine adapter, not
    by the registry.
``default_selection``
    Category -> registered NAME (or the ``"default"`` sentinel) for the initial
    selection when this base model is picked.

The installer's ``downloads`` / ``migrate`` blocks are deliberately NOT parsed:
they belong to ``scripts/install_ltx.ps1`` and nothing on the Python side has
any business acting on them.

FAIL LOUD AT STARTUP: a descriptor that is broken JSON, or that claims
``schema: 2`` + ``engine_family`` while missing/mistyping a required field,
raises :class:`RuntimeError` naming the file and the offending field. A model
manifest is a shipped repository artifact — a typo in one is a build defect,
not a runtime condition to degrade around.

INVARIANT — THIS MODULE MUST NOT IMPORT ``services.engines`` (nor any engine
adapter, directly or transitively). Descriptors are pure data read by the
registry, the API and (later) the engine layer; an engine adapter importing a
descriptor while a descriptor imports an adapter is a circular import waiting
to happen, and would additionally drag the multi-thousand-line LTX adapter into
every process that merely wants to enumerate base models. ``engine_family`` is
a plain string here on purpose; mapping it to an implementation is the engine
layer's job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: Descriptor schema version that carries base-model information. Manifests
#: with any other schema are shared-asset manifests as far as Python cares.
BASE_MODEL_SCHEMA = 2


@dataclass(frozen=True)
class CategoryDescriptor:
    """One selectable model category of one base model."""

    name: str
    #: Scan roots, relative to ``config.models_dir`` (never absolute).
    scan: tuple[str, ...]
    #: Accepted weight-file extensions, lowercase, dot-prefixed.
    extensions: tuple[str, ...]
    #: Recurse into subdirectories of each scan root.
    recursive: bool = False
    #: Filename classifier for categories that SHARE a scan root (the video and
    #: audio VAEs both live in models/LTX23/VAE/): a scanned filename must
    #: contain this hint and not the opposite one.
    name_hint: str | None = None
    #: The file behind the injected ``"default"`` registry entry, relative to
    #: ``config.models_dir``. ``None`` when the base model does not (yet)
    #: declare one — that category then has no usable default.
    default_file: str | None = None


@dataclass(frozen=True)
class BaseModelDescriptor:
    """One base model: what it is called, which engine family runs it, and
    which categories/assets it is made of."""

    id: str
    display_name: str
    engine_family: str
    #: Category name -> descriptor, in DECLARATION ORDER (the dropdown order).
    categories: dict[str, CategoryDescriptor] = field(default_factory=dict)
    #: Asset key -> ``models_dir``-relative path (engine-level fixed files).
    assets: dict[str, str] = field(default_factory=dict)
    #: Category -> registered NAME / ``"default"`` sentinel.
    default_selection: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# loading / validation
# --------------------------------------------------------------------------- #

def _fail(source: str, message: str) -> RuntimeError:
    return RuntimeError(f"base-model descriptor {source}: {message}")


def _require_str(source: str, obj: dict, key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _fail(source, f"'{key}' must be a non-empty string (got {value!r})")
    return value


def _require_rel_path(source: str, where: str, value: str) -> str:
    """Descriptor paths are ALWAYS relative to ``config.models_dir``: they are
    joined onto a directory the operator controls, so an absolute path or a
    ``..`` escape would silently reach outside the model store."""
    p = Path(value)
    if p.is_absolute() or ".." in p.parts:
        raise _fail(source, f"{where} must be a relative path inside models_dir (got {value!r})")
    return value


def _parse_category(source: str, name: str, raw: object) -> CategoryDescriptor:
    where = f"categories.{name}"
    if not isinstance(raw, dict):
        raise _fail(source, f"{where} must be an object (got {type(raw).__name__})")

    scan = raw.get("scan")
    if not isinstance(scan, list) or not scan or not all(isinstance(s, str) and s for s in scan):
        raise _fail(source, f"{where}.scan must be a non-empty array of strings")
    for root in scan:
        _require_rel_path(source, f"{where}.scan entry", root)

    extensions = raw.get("extensions")
    if (
        not isinstance(extensions, list)
        or not extensions
        or not all(isinstance(e, str) and e.startswith(".") for e in extensions)
    ):
        raise _fail(
            source, f"{where}.extensions must be a non-empty array of '.ext' strings"
        )

    recursive = raw.get("recursive", False)
    if not isinstance(recursive, bool):
        raise _fail(source, f"{where}.recursive must be a boolean")

    name_hint = raw.get("name_hint")
    if name_hint is not None and (not isinstance(name_hint, str) or not name_hint):
        raise _fail(source, f"{where}.name_hint must be a non-empty string when present")

    default_file = raw.get("default_file")
    if default_file is not None:
        if not isinstance(default_file, str) or not default_file:
            raise _fail(source, f"{where}.default_file must be a non-empty string when present")
        _require_rel_path(source, f"{where}.default_file", default_file)

    return CategoryDescriptor(
        name=name,
        scan=tuple(scan),
        extensions=tuple(e.lower() for e in extensions),
        recursive=recursive,
        name_hint=name_hint,
        default_file=default_file,
    )


def _parse_descriptor(source: str, raw: dict) -> BaseModelDescriptor:
    identifier = _require_str(source, raw, "id")
    display_name = _require_str(source, raw, "display_name")
    engine_family = _require_str(source, raw, "engine_family")

    categories_raw = raw.get("categories")
    if not isinstance(categories_raw, dict) or not categories_raw:
        raise _fail(source, "'categories' must be a non-empty object")
    categories: dict[str, CategoryDescriptor] = {}
    for name, spec in categories_raw.items():  # declaration order preserved
        if not isinstance(name, str) or not name:
            raise _fail(source, "category names must be non-empty strings")
        categories[name] = _parse_category(source, name, spec)

    assets_raw = raw.get("assets", {})
    if not isinstance(assets_raw, dict):
        raise _fail(source, "'assets' must be an object when present")
    assets: dict[str, str] = {}
    for key, value in assets_raw.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value:
            raise _fail(source, f"assets.{key} must map to a non-empty string")
        assets[key] = _require_rel_path(source, f"assets.{key}", value)

    selection_raw = raw.get("default_selection", {})
    if not isinstance(selection_raw, dict):
        raise _fail(source, "'default_selection' must be an object when present")
    default_selection: dict[str, str] = {}
    for key, value in selection_raw.items():
        if not isinstance(value, str) or not value:
            raise _fail(source, f"default_selection.{key} must map to a non-empty string")
        if key not in categories:
            raise _fail(
                source,
                f"default_selection names unknown category '{key}' "
                f"(declared: {list(categories)})",
            )
        default_selection[key] = value

    return BaseModelDescriptor(
        id=identifier,
        display_name=display_name,
        engine_family=engine_family,
        categories=categories,
        assets=assets,
        default_selection=default_selection,
    )


def load_base_models(manifest_dir: Path) -> dict[str, BaseModelDescriptor]:
    """Load every base-model descriptor in ``manifest_dir``.

    Files are read in FILENAME ORDER, and the returned mapping preserves it —
    the numeric prefix (``10-ltx23.json`` before ``20-ltx25.json``) is what
    decides which base model is listed, and defaulted to, first.

    Manifests are read as ``utf-8-sig``: every shipped manifest carries a UTF-8
    BOM (they are edited on Windows), and plain ``utf-8`` would drag the BOM
    into the first key name. ``utf-8-sig`` reads BOM-less files unchanged, so
    hand-written test fixtures need no BOM.

    Raises :class:`RuntimeError` for a missing directory, unreadable/broken
    JSON, a duplicate base-model id, or a schema-2 descriptor with a missing or
    mistyped required field — see the module docstring on failing loud.
    """
    manifest_dir = Path(manifest_dir)
    if not manifest_dir.is_dir():
        raise RuntimeError(
            f"base-model manifest directory not found: {manifest_dir} "
            "(expected the shipped scripts/manifests/*.json descriptors)"
        )

    descriptors: dict[str, BaseModelDescriptor] = {}
    origin: dict[str, str] = {}
    for path in sorted(manifest_dir.glob("*.json"), key=lambda p: p.name):
        source = path.name
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except OSError as exc:
            raise _fail(source, f"unreadable file: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise _fail(source, f"invalid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise _fail(source, "top level must be a JSON object")

        # Not a base model -> a shared-asset manifest (installer-only). Ignored
        # on purpose and without noise: 00-preprocessors.json is expected here.
        if raw.get("schema") != BASE_MODEL_SCHEMA or "engine_family" not in raw:
            continue

        descriptor = _parse_descriptor(source, raw)
        if descriptor.id in descriptors:
            raise _fail(
                source,
                f"duplicate base-model id '{descriptor.id}' "
                f"(already declared by {origin[descriptor.id]})",
            )
        descriptors[descriptor.id] = descriptor
        origin[descriptor.id] = source

    if not descriptors:
        raise RuntimeError(
            f"no base-model descriptors found in {manifest_dir} "
            f"(a base model is a *.json with schema {BASE_MODEL_SCHEMA} and an "
            "'engine_family' field)"
        )
    return descriptors
