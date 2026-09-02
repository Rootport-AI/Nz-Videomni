"""IC-LoRA / style-LoRA adapter-name registry (Phase B, extended Phase C, S1).

Resolves a server-side adapter NAME (accepted in ``GenerateRequest.loras[].name``)
to the safetensors file on disk plus its ``preprocess`` kind, passing the
requested strength through untouched. Names come from two sources, config
authoritative over the scan:

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
  * ``layout`` — which weight-key layout the file uses, so an adapter the engine
    loader cannot read is refused up front (``LORA_FORMAT_UNSUPPORTED``, 422)
    instead of silently producing a LoRA-free video. See :func:`_detect_layout`.

§3-108 (2026-09-02) removed the former ``scale`` field: it multiplied the
requested strength by ``ss_network_alpha / ss_network_dim`` read from the
header metadata, but musubi-tuner's A/B conversion already bakes alpha into the
weights and copies that metadata over verbatim — so the multiplier was a second,
fossil application (``Pixar_Toon`` and ``LTX-2.3-Henshin`` ran at half the
requested strength). Alpha is now handled where it is real: the engine loader
folds a kohya file's ``.alpha`` tensor into B at load time, and this layer
passes ``strength`` through exactly as requested (ComfyUI semantics).

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

from api.errors import lora_format_unsupported, lora_not_found
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


@dataclass
class LoraEntryInfo:
    """One resolved adapter: a config registration or a directory-scan hit."""

    name: str
    path: Path  # absolute (config._abs of the registered/scanned path)
    preprocess: str  # "none" | "canny" | "dwpose" | "depth"
    kind: str  # "style" | "control"
    layout: str  # "ab" | "kohya" | "unsupported:<reason>" (see _detect_layout)
    has_thumbnail: bool  # a sibling <stem>.png exists
    source: str  # "config" | "scan"
    exists: bool
    # §1-15 (clip-wise IC-LoRA reference): the safetensors header's own
    # ``reference_downscale_factor`` (union-control=2, deblur=1) — the SAME
    # value ``engine/pipeline``'s ``_ic_reference_downscale_factor`` reads at
    # generation time (test_ic_lora_engine_conditioning.py), just surfaced here
    # too so the WebUI can size its stage-1 comfort-budget estimate
    # (tokenBudget.ts's ``chainStage1Tokens``) BEFORE a job is ever submitted.
    # None when the header carries no such key (style LoRAs; a config control
    # entry whose header is missing/unreadable) or the value is unparsable.
    reference_downscale_factor: float | None = None

    def as_dict(self) -> dict:
        """GET /loras row (mirrors ModelEntryInfo.as_dict — no path leak)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "has_thumbnail": self.has_thumbnail,
            "exists": self.exists,
            "source": self.source,
            # §1-15 additive fields (both new on GET /loras; an old FE build
            # simply ignores unknown keys, so this is safe to always include —
            # unlike webui/src/api/types.ts's LoraEntry.preprocess, which marks
            # it OPTIONAL only because an OLDER SERVER may omit it, not because
            # this server ever leaves it out).
            "preprocess": self.preprocess,
            "reference_downscale_factor": self.reference_downscale_factor,
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
       0 Linears, i.e. exactly the silent no-op §3-108 exists to end.
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
        """Build a LoraEntryInfo, reading kind/layout from the header when possible.

        Returns ``None`` (drop it) ONLY for a scan entry whose header is
        unreadable. A config entry is always kept — a missing/broken file falls
        back to preprocess-derived kind + layout ``"ab"``, and resolve() fails
        loud later if the file is absent when a job actually needs it. That
        ``"ab"`` default is deliberate: with no header there is nothing to judge
        a layout on, so the pre-§3-108 behaviour (accept, let the engine speak)
        is kept rather than inventing a rejection out of missing evidence.
        """
        exists = abs_path.exists()
        has_thumbnail = abs_path.with_suffix(".png").exists()
        # preprocess is authoritative for control (a canny/pose registration is a
        # control adapter regardless of what the header says).
        kind = "control" if preprocess != "none" else "style"
        layout = _LAYOUT_AB
        ref_downscale: float | None = None
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
        """Resolve ``name`` -> ``ResolvedLora(path, strength, preprocess,
        audio_strength)``.

        ``strength`` and ``audio_strength`` are returned EXACTLY as requested
        (§3-108: the alpha/rank metadata multiplier is gone — see the module
        docstring); ``audio_strength`` stays ``None`` when the caller passed
        ``None`` (video-axis follow — no independent audio strength was asked
        for). ``preprocess`` is ``"none"`` for legacy string entries / scanned
        files or the ``IcLoraEntry.preprocess`` value for config dict entries.

        Raises ``lora_not_found`` (404) for a path-like name, an empty registry,
        an unknown name, or a registered-but-missing file; and
        ``lora_format_unsupported`` (422) for a file whose weight-key layout the
        engine loader cannot read. Both endpoint validation loops (single and
        chain) already call this per requested adapter, so neither needed a new
        check of its own.
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
        if entry.layout.startswith(_LAYOUT_UNSUPPORTED):
            _, _, reason = entry.layout.partition(":")
            raise lora_format_unsupported(name, reason or entry.layout)
        return ResolvedLora(
            entry.path,
            float(strength),
            entry.preprocess,
            None if audio_strength is None else float(audio_strength),
        )
