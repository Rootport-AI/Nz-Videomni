"""IC-LoRA / style-LoRA adapter-name registry.

Resolves a server-side adapter NAME (accepted in ``GenerateRequest.loras[].name``)
to the safetensors file on disk plus its ``preprocess`` kind, passing the
requested strength through untouched. Names come from two sources, config
authoritative over the scan:

1. ``config.yaml`` ``model.ic_loras`` (name -> project-relative path, or name ->
   ``IcLoraEntry`` for control adapters) — the authoritative registration;
2. a directory scan of the scope's LoRA directory: every ``*.safetensors``
   there is exposed under its filename stem, so a style/character LoRA dropped in
   the folder shows up without any config edit (mirrors ``services.model_registry``).

FAMILY-SCOPED (台帳 §1-104). Which directory is scanned, whether the config
registrations take part, and who judges a file's format is a
:class:`LoraScope`, handed in by a ``scope_provider`` (the app passes
``services.engines.lora_scope_for`` on the runner's descriptor — the ONE place
that maps a family to a directory). No provider = today's single scope:
``config.model.lora_dir`` plus ``ic_loras``, judged by :func:`_detect_layout`.
Every read (``resolve`` / ``info`` / ``names`` / ``preprocess_for``) first checks
the provider's scope and rescans when it changed, so a job submitted right after
a base-model switch never resolves a name from the previous family's listing.

Arbitrary filesystem paths are never accepted (the Pydantic ``LoraSpec`` validator
rejects path-like names, and this layer refuses to treat a name as a path).

Per-entry metadata (read from the safetensors header, pure Python — the app venv
has no torch):
  * ``kind`` — ``"control"`` when the header ``__metadata__`` carries
    ``reference_downscale_factor`` OR the config entry declares a non-``"none"``
    ``preprocess`` (union-control / pixel-spatial-upscaler: they derive their
    conditioning from a REFERENCE video); ``"style"`` otherwise (画風/character
    LoRAs that need no reference). Drives the endpoint's reference-required check.
  * ``layout`` — which weight-key layout the file uses, so an adapter the engine
    loader cannot read is refused up front (``LORA_FORMAT_UNSUPPORTED``, 422)
    instead of silently producing a LoRA-free video. See :func:`_detect_layout`.

There is no alpha/rank multiplier on ``strength``: musubi-tuner's A/B
conversion already bakes alpha into the weights and copies the
``ss_network_alpha`` / ``ss_network_dim`` header metadata over verbatim, so
scaling by that metadata would apply alpha a second time. Alpha is handled
where it is real: the engine loader folds a kohya file's ``.alpha`` tensor into
B at load time, and this layer passes ``strength`` through exactly as requested
(ComfyUI semantics).

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
from typing import Callable, NamedTuple

from api.errors import lora_format_unsupported, lora_not_found
from config import PROJECT_ROOT, AppConfig, IcLoraEntry

logger = logging.getLogger("ltx.loras")


class ResolvedLora(NamedTuple):
    """``resolve()``'s return shape. Indices 0-2 (``path``/``strength``/
    ``preprocess``) keep fixed positions, so index access to those three
    (``lp[2]`` and the like) does not depend on ``audio_strength`` at index 3.
    """

    path: Path
    strength: float
    preprocess: str
    audio_strength: float | None

# A safetensors JSON header beyond this is implausible for a LoRA and more likely
# a corrupt/foreign file than a real header (services.model_registry precedent).
_MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024

# Weight-key suffixes, kept in sync with ``engine.gguf.ic_lora_common`` (which
# cannot be imported here — this layer runs in the torch-free app venv).
_SUFFIX_A = ".lora_A.weight"
_SUFFIX_DOWN = ".lora_down.weight"
_SUFFIX_DORA = ".dora_scale"

#: ``LoraEntryInfo.layout`` values. Anything starting with ``_LAYOUT_UNSUPPORTED``
#: is refused by ``resolve()``; the reason follows a ``:`` so the entry needs no
#: second field to carry it.
_LAYOUT_AB = "ab"
_LAYOUT_KOHYA = "kohya"
_LAYOUT_UNSUPPORTED = "unsupported"

#: A family's format judge: the safetensors header (``__metadata__`` included)
#: -> ``(supported, reason)``. Declared by an engine adapter as
#: ``LORA_VERDICT``.
LoraVerdict = Callable[[dict], "tuple[bool, str | None]"]


@dataclass(frozen=True)
class LoraScope:
    """What one registry listing covers.

    ``verdict`` is ``None`` -> today's rule: :func:`_detect_layout` judges and
    ``resolve()`` refuses an unsupported file with 422 (LTX). A ``verdict`` ->
    it judges, and the judgement is DISPLAY ONLY: ``resolve()`` lets the file
    through (MiniMax H3: an unsupported file may still run, 裁定 2).
    """

    family: str | None
    #: Absolute directory scanned flat for ``*.safetensors`` (``None``: no scan).
    lora_dir: Path | None
    #: The same directory as GET /loras shows it (project-relative POSIX).
    lora_dir_display: str
    #: Whether ``config.model.ic_loras`` registrations are part of the listing.
    include_config: bool
    verdict: LoraVerdict | None = None

    @property
    def key(self) -> tuple:
        return (self.family, str(self.lora_dir), self.include_config)


def make_scope(
    family: str | None,
    lora_dir: Path | None,
    *,
    include_config: bool,
    verdict: LoraVerdict | None = None,
) -> LoraScope:
    """A :class:`LoraScope` with its display path filled in: relative to the
    project root in POSIX form (``models/MiniMaxH3/LoRA``), else absolute."""
    if lora_dir is None:
        display = ""
    else:
        try:
            display = lora_dir.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            display = lora_dir.as_posix()
    return LoraScope(family, lora_dir, display, include_config, verdict)


def shared_scope(config: AppConfig, family: str | None = None) -> LoraScope:
    """Today's scope: ``config.model.lora_dir`` + the config registrations,
    judged by :func:`_detect_layout`."""
    raw = getattr(config.model, "lora_dir", None)
    return make_scope(family, config._abs(raw) if raw else None, include_config=True)


@dataclass
class LoraEntryInfo:
    """One resolved adapter: a config registration or a directory-scan hit."""

    name: str
    path: Path  # absolute (config._abs of the registered/scanned path)
    preprocess: str  # IcLoraEntry.preprocess ("none" for string/scan entries)
    kind: str  # "style" | "control"
    layout: str  # "ab" | "kohya" | "unsupported:<reason>" (see _detect_layout)
    has_thumbnail: bool  # a sibling <stem>.png exists
    source: str  # "config" | "scan"
    exists: bool
    # Clip-wise IC-LoRA reference: the safetensors header's own
    # ``reference_downscale_factor`` (union-control=2, deblur=1) — the SAME
    # value ``engine/pipeline``'s ``_ic_reference_downscale_factor`` reads at
    # generation time (test_ic_lora_engine_conditioning.py), just surfaced here
    # too so the WebUI can size its stage-1 comfort-budget estimate
    # (tokenBudget.ts's ``chainStage1Tokens``) BEFORE a job is ever submitted.
    # None when the header carries no such key (style LoRAs; a config control
    # entry whose header is missing/unreadable) or the value is unparsable.
    reference_downscale_factor: float | None = None
    #: The scope's judgement (GET /loras shows it). Without a family verdict it
    #: is ``layout`` read as a flag; with one, the verdict's answer.
    supported: bool = True
    unsupported_reason: str | None = None

    def as_dict(self) -> dict:
        """GET /loras row (mirrors ModelEntryInfo.as_dict — no path leak)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "has_thumbnail": self.has_thumbnail,
            "exists": self.exists,
            "source": self.source,
            # Both fields are always included: a frontend build that does not
            # know them simply ignores unknown keys — unlike
            # webui/src/api/types.ts's LoraEntry.preprocess, which marks
            # it OPTIONAL only because an OLDER SERVER may omit it, not because
            # this server ever leaves it out.
            "preprocess": self.preprocess,
            "reference_downscale_factor": self.reference_downscale_factor,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
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


def _detect_layout(header: dict) -> str:
    """Classify the file's weight-key layout from its tensor NAMES alone.

    Returns ``"ab"``, ``"kohya"``, or ``"unsupported:<reason>"`` — the reason
    travels inside the string so an entry needs no second field for it, and
    ``resolve()`` hands it to the client as the 422's ``detail``.

    Evaluation order is part of the contract:

    1. ``.dora_scale`` -> DoRA. Checked FIRST because a DoRA file also carries
       ordinary A/B or down/up keys; reading it as a plain LoRA would silently
       drop the magnitude vector and change the result.
    2. ``.lora_A.weight`` -> A/B, matching ``load_ic_lora_pairs``' own
       A/B-first file-unit guard (one A key makes the whole file an A/B file).
    3. ``.lora_down.weight`` -> kohya, but ONLY when at least one such key has a
       DOTTED module path once the suffix is stripped. LyCORIS/sd-scripts also
       emit underscore-joined names (``lora_unet_transformer_blocks_0_...``)
       that no ``named_modules()`` lookup can resolve — those would attach to
       0 Linears, i.e. exactly the silent no-op this check exists to prevent.
    4. ``hada_`` / ``lokr_`` -> LoHa / LoKr: different factorisations
       altogether (Hadamard product / Kronecker product), not a B@A delta.
    5. Nothing recognisable -> unknown.
    """
    names = [name for name in header if name != "__metadata__"]
    if any(name.endswith(_SUFFIX_DORA) for name in names):
        return f"{_LAYOUT_UNSUPPORTED}:DoRA (weight-decomposed, '{_SUFFIX_DORA}' keys)"
    if any(name.endswith(_SUFFIX_A) for name in names):
        return _LAYOUT_AB
    down = [name for name in names if name.endswith(_SUFFIX_DOWN)]
    if down:
        if any("." in name[: -len(_SUFFIX_DOWN)] for name in down):
            return _LAYOUT_KOHYA
        return (
            f"{_LAYOUT_UNSUPPORTED}:kohya keys joined by underscores instead of "
            f"dots (e.g. '{down[0]}'), which match no module path"
        )
    if any("hada_" in name for name in names):
        return f"{_LAYOUT_UNSUPPORTED}:LoHa (Hadamard-product 'hada_' keys)"
    if any("lokr_" in name for name in names):
        return f"{_LAYOUT_UNSUPPORTED}:LoKr (Kronecker-product 'lokr_' keys)"
    sample = ", ".join(f"'{name}'" for name in names[:3]) or "(no tensors)"
    return (
        f"{_LAYOUT_UNSUPPORTED}:no LoRA weight keys found; supported layouts are "
        f"'{_SUFFIX_A}'/'.lora_B.weight' and '{_SUFFIX_DOWN}'/'.lora_up.weight'. "
        f"Sample keys: {sample}"
    )


class LoraRegistry:
    def __init__(
        self,
        config: AppConfig,
        scope_provider: Callable[[], LoraScope] | None = None,
    ):
        self.config = config
        # name -> project-relative (or absolute) safetensors path (string form)
        # or an IcLoraEntry (path + preprocess kind). Config-authoritative
        # source (part of a listing only when the scope includes config).
        self.registry: dict[str, str | IcLoraEntry] = dict(config.model.ic_loras or {})
        self._scope_provider = scope_provider or (lambda: shared_scope(config))
        # (scope key, name -> LoraEntryInfo, scope), rebuilt by rescan() and
        # swapped in ONE assignment, so a reader never sees the entries of one
        # scope paired with another scope's key.
        self._state: tuple[tuple, dict[str, LoraEntryInfo], LoraScope]
        self.rescan()

    @property
    def _entries(self) -> dict[str, LoraEntryInfo]:
        return self._state[1]

    @property
    def scope(self) -> LoraScope:
        """The scope the cached listing was built for."""
        return self._state[2]

    @property
    def lora_dir_display(self) -> str:
        """The scanned directory as GET /loras shows it (``lora_dir``)."""
        return self.scope.lora_dir_display

    # ------------------------------------------------------------------ scan

    def _ensure_scope(self) -> None:
        """Rescan when the provider's scope is not the one the cache holds."""
        if self._scope_provider().key != self._state[0]:
            self.rescan()

    def rescan(self) -> None:
        """Rebuild the entry cache for the provider's CURRENT scope: config
        registrations (authoritative, when the scope includes them) merged with
        a fresh scan of the scope's directory. Cheap (a directory listing +
        header reads of small JSON blobs), so the API layer runs it per GET
        /loras — a newly dropped-in file appears without a restart
        (model_registry precedent)."""
        scope = self._scope_provider()
        entries: dict[str, LoraEntryInfo] = {}
        known_paths: set[Path] = set()

        # 1. Config registrations win.
        for name, raw in (self.registry.items() if scope.include_config else ()):
            if isinstance(raw, str):
                rel, preprocess = raw, "none"
            else:
                rel, preprocess = raw.path, raw.preprocess
            abs_path = self.config._abs(rel)
            entries[name] = self._build_entry(name, abs_path, preprocess, "config", scope)
            known_paths.add(abs_path.resolve())

        # 2. Directory scan (best-effort; broken files silently dropped).
        for found in self._scan_lora_dir(scope.lora_dir):
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
            info = self._build_entry(name, found, "none", "scan", scope)
            if info is None:
                continue  # unreadable header -> not surfaced from the scan
            entries[name] = info
            known_paths.add(resolved)

        self._state = (scope.key, entries, scope)

    @staticmethod
    def _scan_lora_dir(base: Path | None) -> list[Path]:
        """Flat scan of the scope's directory for ``*.safetensors`` files."""
        if base is None or not base.is_dir():
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
        self, name: str, abs_path: Path, preprocess: str, source: str, scope: LoraScope
    ) -> LoraEntryInfo | None:
        """Build a LoraEntryInfo, reading kind/layout from the header when possible.

        Returns ``None`` (drop it) ONLY for a scan entry whose header is
        unreadable. A config entry is always kept — a missing/broken file falls
        back to preprocess-derived kind + layout ``"ab"``, and resolve() fails
        loud later if the file is absent when a job actually needs it. That
        ``"ab"`` default is deliberate: with no header there is nothing to judge
        a layout on, so the file is accepted and the engine is left to
        speak, rather than inventing a rejection out of missing evidence.

        ``supported`` / ``unsupported_reason``: the scope's ``verdict`` on the
        header when it has one; otherwise ``layout`` read as a flag (the reason
        is the text after ``unsupported:``).
        """
        exists = abs_path.exists()
        has_thumbnail = abs_path.with_suffix(".png").exists()
        # preprocess is authoritative for control (a canny/pose registration is a
        # control adapter regardless of what the header says).
        kind = "control" if preprocess != "none" else "style"
        layout = _LAYOUT_AB
        ref_downscale: float | None = None
        verdict: tuple[bool, str | None] | None = None
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
                    "lora '%s': unreadable header for %s (%s); kind/layout from "
                    "registration only",
                    name,
                    abs_path,
                    exc,
                )
            else:
                md = _metadata(header)
                if "reference_downscale_factor" in md:
                    kind = "control"
                    try:
                        ref_downscale = float(md["reference_downscale_factor"])
                    except (TypeError, ValueError):
                        ref_downscale = None
                layout = _detect_layout(header)
                if scope.verdict is not None:
                    try:
                        verdict = scope.verdict(header)
                    except Exception as exc:  # a malformed entry must not fail the scan
                        verdict = (False, f"header unreadable: {type(exc).__name__}")
        if verdict is None:
            _, _, reason = layout.partition(":")
            verdict = (not layout.startswith(_LAYOUT_UNSUPPORTED), reason or None)
        supported, unsupported_reason = verdict
        return LoraEntryInfo(
            name=name,
            path=abs_path,
            preprocess=preprocess,
            kind=kind,
            layout=layout,
            has_thumbnail=has_thumbnail,
            source=source,
            exists=exists,
            reference_downscale_factor=ref_downscale,
            supported=bool(supported),
            unsupported_reason=None if supported else unsupported_reason,
        )

    # ----------------------------------------------------------------- reads

    def names(self) -> list[str]:
        self._ensure_scope()
        return sorted(self._entries)

    def entries(self) -> list[LoraEntryInfo]:
        """GET /loras rows (sorted by name)."""
        return [self._entries[name] for name in self.names()]

    def info(self, name: str) -> LoraEntryInfo:
        """Return the cached :class:`LoraEntryInfo` for ``name`` (404 if unknown).

        Used by the endpoint layer for the kind-aware reference-required check.
        """
        self._ensure_scope()
        entry = self._entries.get(name)
        if entry is None:
            raise lora_not_found(name, detail=f"known adapters: {self.names()}")
        return entry

    def preprocess_for(self, name: str) -> str:
        """Return the ``preprocess`` kind for ``name`` in the current scope (no
        file check).

        ``"none"`` for string-form entries, scanned files, or a name the scope
        does not list; the ``IcLoraEntry.preprocess`` value for config dict
        entries. Used for metadata annotation
        (``pipeline_manager._lora_metadata_entry``).
        """
        self._ensure_scope()
        entry = self._entries.get(name)
        return "none" if entry is None else entry.preprocess

    # --------------------------------------------------------------- resolve

    def resolve(
        self, name: str, strength: float, audio_strength: float | None = None
    ) -> ResolvedLora:
        """Resolve ``name`` -> ``ResolvedLora(path, strength, preprocess,
        audio_strength)``.

        ``strength`` and ``audio_strength`` are returned EXACTLY as requested
        (no alpha/rank metadata multiplier is applied — see the module
        docstring); ``audio_strength`` stays ``None`` when the caller passed
        ``None`` (video-axis follow — no independent audio strength was asked
        for). ``preprocess`` is ``"none"`` for string-form entries / scanned
        files or the ``IcLoraEntry.preprocess`` value for config dict entries.

        Raises ``lora_not_found`` (404) for a path-like name, an empty registry,
        an unknown name, or a registered-but-missing file; and
        ``lora_format_unsupported`` (422) for a file whose weight-key layout the
        engine loader cannot read — only in a scope WITHOUT a family verdict
        (LTX); a verdict's "unsupported" is display only (MiniMax H3). Both
        endpoint validation loops (single and chain) call this per requested
        adapter, so the layout refusal needs no check of its own there.
        """
        if "/" in name or "\\" in name or ".." in name:
            raise lora_not_found(name, detail="adapter name must not be a path")
        self._ensure_scope()
        # One read of the state: the entries and the scope below are the same pair.
        _key, entries, scope = self._state
        entry = entries.get(name)
        if entry is None:
            if not entries:
                detail = f"no LoRA files found in {scope.lora_dir_display}"
                if scope.include_config:
                    detail += " and model.ic_loras is empty"
                raise lora_not_found(name, detail=detail)
            raise lora_not_found(name, detail=f"known adapters: {sorted(entries)}")
        if not entry.path.exists():
            raise lora_not_found(
                name, detail=f"registered adapter file missing: {entry.path}"
            )
        if not entry.supported and scope.verdict is None:
            raise lora_format_unsupported(name, entry.unsupported_reason or entry.layout)
        return ResolvedLora(
            entry.path,
            float(strength),
            entry.preprocess,
            None if audio_strength is None else float(audio_strength),
        )
