"""Assets-only text-encoder safetensors, extracted from the TE GGUF (§3-98 Phase 2c, F2).

Why this file exists
--------------------
``DistilledPipeline.__init__`` builds a ``PromptEncoder``, and ``PromptEncoder``
reaches for the text encoder's *assets* -- the HuggingFace config, the tokenizer,
the processor sidecars -- long before anybody gets a chance to substitute a
weight loader::

    encode_type = gemma_model_type(text_encoder_path)        # GemmaAssets.load(...)
    EmbeddingsProcessorConfigurator.with_gemma_model_path(text_encoder_path)
    module_ops_from_gemma_root(text_encoder_path)            # GemmaAssets.load(...)

``GemmaAssets.load`` accepts exactly two things: an HF directory, or a single
``.safetensors`` file. Point it at a ``.gguf`` and it raises FileNotFoundError
before the pipeline object even exists. Patching every one of those call sites
would mean forking ``PromptEncoder``, ``EmbeddingsProcessorConfigurator`` and
``base_encoder`` -- three private surfaces engine25 would then own forever.

The cheaper and far more durable trick: give the official code the file it is
asking for. Everything ``GemmaAssets`` wants is *already inside the TE GGUF* --
the converter packed the same five U8 sidecar payloads the official bf16 TE file
carries (``tokenizer_json`` plus four ``hf_asset__*``), and the same HF config
JSON, as a ``config`` KV. So this module writes a ~32 MB ``.safetensors`` that is
byte-identical to the official 26.3 GB file *minus its 681 weight tensors*, and
engine25 hands that path to ``ModelPaths.from_split(text_encoder_path=...)``.

The official asset code then runs completely unmodified. The weights come from
the GGUF through engine25's own builder (see :mod:`engine25.gguf_gemma4`), which
is the only part that ever needed replacing.

What is written
---------------
* metadata ``gemma_config``  -- the HF config JSON, verbatim from the GGUF ``config`` KV
* metadata ``format``        -- ``"pt"``, as the official packer writes
* metadata ``nz_assets_export`` -- provenance (source file + size + schema version);
  ignored by ``GemmaAssets``, and the reason a regenerated GGUF invalidates a stale export
* tensors ``tokenizer_json`` + ``hf_asset__*`` -- uint8, byte-for-byte from the GGUF

Idempotence
-----------
:func:`ensure_assets_only` regenerates only when the file is missing, unreadable,
or does not match the GGUF byte-for-byte (metadata *and* payloads). A match is a
no-op with a log line, so it is safe to call on every worker start. The write is
atomic (temp file + ``os.replace``), so an interrupted export never leaves a
half-file that would then load as a corrupt tokenizer.

Usage::

    python -m engine25.assets_export <te.gguf> [--out PATH] [--force] [--json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# On-disk contract constants.
#
# These three strings are the *file format* of an official TE safetensors pack,
# not an API: ltx_core/text_encoders/gemma/gemma_assets.py declares the same
# values and both the official bf16 TE file and our GGUF already carry them on
# disk. Re-declared rather than imported so this exporter has no ltx_core import
# at all -- it must be able to run and produce the file before anything else in
# engine25 is importable. The round-trip is *proved* rather than assumed:
# :func:`verify_with_official_loader` loads the result through the real
# ``GemmaAssets`` (via engine25.ltxcore_compat, never ltx_core directly), and the
# gguf_gemma4 selftest calls it.
# ---------------------------------------------------------------------------
GEMMA_CONFIG_METADATA_KEY = "gemma_config"
TOKENIZER_JSON_TENSOR_KEY = "tokenizer_json"
HF_ASSET_TENSOR_PREFIX = "hf_asset__"

#: KV holding the HF config JSON in a converter-produced TE GGUF.
GGUF_CONFIG_KEY = "config"

#: Suffix appended to the GGUF stem. The file lands beside the GGUF so the two
#: travel together: a weights file and its assets are one unit, and the manifest
#: step (Phase 4) copies/links them side by side.
ASSETS_SUFFIX = ".assets.safetensors"

#: Bumped when the written layout changes, so old exports regenerate instead of
#: being silently accepted by a newer engine.
SCHEMA_VERSION = 1

#: Sidecars ``GemmaAssets._require_sidecars`` insists on. Checked here so a
#: truncated GGUF is reported by name at export time rather than as a confusing
#: "missing required sidecar" from deep inside the official loader.
REQUIRED_SIDECARS = ("tokenizer_config.json", "processor_config.json")


class AssetsExportError(RuntimeError):
    """The TE GGUF does not carry the payloads an assets-only file needs."""


@dataclass(frozen=True)
class ExportResult:
    """Outcome of :func:`ensure_assets_only`."""

    path: Path
    action: str  # "created" | "regenerated" | "reused"
    reason: str
    size_bytes: int
    tensor_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "action": self.action,
            "reason": self.reason,
            "size_bytes": self.size_bytes,
            "size_mib": round(self.size_bytes / 2**20, 2),
            "tensors": list(self.tensor_names),
        }


def assets_path_for(gguf_path: str | Path) -> Path:
    """Where the assets-only file for *gguf_path* lives (beside the GGUF)."""
    gguf = Path(gguf_path)
    return gguf.with_name(gguf.stem + ASSETS_SUFFIX)


# ---------------------------------------------------------------------------
# Reading the GGUF side
# ---------------------------------------------------------------------------


def read_payload(gguf_path: str | Path) -> tuple[dict[str, bytes], str]:
    """Return ``({tensor_name: bytes}, gemma_config_json)`` from a TE GGUF.

    The sidecars are stored as GGML type I8 -- gguf-py has no U8 writer type, and
    I8 is bit-identical storage, so the bytes are simply reinterpreted. Only the
    five payload tensors are touched; the 681 weight tensors are never faulted in
    (the reader memory-maps the file), which is what keeps this a sub-second
    operation on a 9.2 GB input.
    """
    import gguf as gguf_lib
    import numpy as np

    gguf_path = str(gguf_path)
    reader = gguf_lib.GGUFReader(gguf_path, mode="r")
    try:
        field = reader.fields.get(GGUF_CONFIG_KEY)
        if field is None:
            raise AssetsExportError(
                f"{Path(gguf_path).name}: no {GGUF_CONFIG_KEY!r} KV. This is not a text-encoder "
                f"GGUF from the Nz converter's gemma4-ltx25 profile."
            )
        config_json = field.contents()
        if not isinstance(config_json, str):
            raise AssetsExportError(
                f"{Path(gguf_path).name}: KV {GGUF_CONFIG_KEY!r} is {type(config_json).__name__}, "
                f"expected a JSON string."
            )
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as exc:
            raise AssetsExportError(f"{Path(gguf_path).name}: KV {GGUF_CONFIG_KEY!r} is not valid JSON.") from exc
        if not isinstance(config, dict) or not config.get("model_type"):
            raise AssetsExportError(
                f"{Path(gguf_path).name}: KV {GGUF_CONFIG_KEY!r} carries no 'model_type'; "
                f"``build_gemma_hf_config`` would reject it."
            )

        payloads: dict[str, bytes] = {}
        for tensor in reader.tensors:
            name = tensor.name
            if name != TOKENIZER_JSON_TENSOR_KEY and not name.startswith(HF_ASSET_TENSOR_PREFIX):
                continue
            payloads[name] = np.asarray(tensor.data).view(np.uint8).tobytes()
    finally:
        del reader

    if TOKENIZER_JSON_TENSOR_KEY not in payloads:
        raise AssetsExportError(
            f"{Path(gguf_path).name}: no {TOKENIZER_JSON_TENSOR_KEY!r} tensor; the tokenizer cannot be rebuilt."
        )
    missing = [n for n in REQUIRED_SIDECARS if f"{HF_ASSET_TENSOR_PREFIX}{n}" not in payloads]
    if missing:
        raise AssetsExportError(
            f"{Path(gguf_path).name}: missing required sidecar tensor(s) "
            f"{[HF_ASSET_TENSOR_PREFIX + n for n in missing]}."
        )
    return payloads, config_json


def build_expected(gguf_path: str | Path) -> tuple[dict[str, bytes], dict[str, str]]:
    """The exact ``(payloads, metadata)`` an assets-only file for *gguf_path* must hold."""
    gguf = Path(gguf_path)
    payloads, config_json = read_payload(gguf)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "generator": "engine25.assets_export",
        "source_gguf": gguf.name,
        "source_bytes": gguf.stat().st_size,
    }
    metadata = {
        # "format" first is cosmetic; safetensors sorts its own header anyway.
        "format": "pt",
        GEMMA_CONFIG_METADATA_KEY: config_json,
        # Provenance is what makes a re-converted GGUF invalidate a stale export.
        # GemmaAssets ignores unknown metadata keys, so this costs nothing.
        "nz_assets_export": json.dumps(provenance, sort_keys=True, separators=(",", ":")),
    }
    return payloads, metadata


# ---------------------------------------------------------------------------
# Reading / writing the safetensors side
# ---------------------------------------------------------------------------


def _read_existing(path: Path) -> tuple[dict[str, bytes], dict[str, str]] | None:
    """``(payloads, metadata)`` of an existing export, or None if unreadable."""
    import safetensors

    try:
        with safetensors.safe_open(str(path), framework="pt") as handle:
            metadata = dict(handle.metadata() or {})
            payloads = {key: _tensor_bytes(handle.get_tensor(key)) for key in handle.keys()}  # noqa: SIM118
    except Exception as exc:  # noqa: BLE001 -- any read failure means "regenerate"
        logger.warning("assets-only file %s could not be read (%s); it will be regenerated", path.name, exc)
        return None
    return payloads, metadata


def _tensor_bytes(tensor: Any) -> bytes:
    import numpy as np

    return np.asarray(tensor.detach().cpu().numpy()).view(np.uint8).tobytes()


def _write(path: Path, payloads: dict[str, bytes], metadata: dict[str, str]) -> None:
    """Write atomically: full temp file first, then a single rename."""
    import numpy as np
    import torch
    from safetensors.torch import save_file

    tensors = {
        name: torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy())
        for name, data in payloads.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        save_file(tensors, str(tmp), metadata=metadata)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _difference(
    have: tuple[dict[str, bytes], dict[str, str]],
    want: tuple[dict[str, bytes], dict[str, str]],
) -> str | None:
    """First reason *have* is not *want*, or None when they match exactly."""
    have_payloads, have_meta = have
    want_payloads, want_meta = want
    if have_meta != want_meta:
        changed = sorted(
            key for key in set(have_meta) | set(want_meta) if have_meta.get(key) != want_meta.get(key)
        )
        return f"metadata differs: {changed}"
    if set(have_payloads) != set(want_payloads):
        missing = sorted(set(want_payloads) - set(have_payloads))
        extra = sorted(set(have_payloads) - set(want_payloads))
        return f"tensor set differs (missing={missing}, extra={extra})"
    for name in sorted(want_payloads):
        if have_payloads[name] != want_payloads[name]:
            return f"tensor {name!r} content differs"
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def ensure_assets_only(
    gguf_path: str | Path,
    out_path: str | Path | None = None,
    *,
    force: bool = False,
) -> ExportResult:
    """Make sure an assets-only ``.safetensors`` for *gguf_path* exists and is current.

    Safe to call on every worker start: when the file already matches the GGUF
    this reads ~32 MB, compares, and returns ``action="reused"``.
    """
    gguf = Path(gguf_path)
    if not gguf.is_file():
        raise FileNotFoundError(f"text-encoder GGUF not found: {gguf}")
    target = Path(out_path) if out_path is not None else assets_path_for(gguf)

    want = build_expected(gguf)
    names = tuple(sorted(want[0]))

    if force:
        reason = "--force requested"
        action = "regenerated" if target.exists() else "created"
    elif not target.exists():
        reason = "file does not exist"
        action = "created"
    else:
        have = _read_existing(target)
        if have is None:
            reason, action = "existing file is unreadable", "regenerated"
        else:
            difference = _difference(have, want)
            if difference is None:
                size = target.stat().st_size
                logger.info(
                    "assets-only text encoder up to date: %s (%.1f MiB, %d tensors)",
                    target.name, size / 2**20, len(names),
                )
                return ExportResult(target, "reused", "matches the GGUF", size, names)
            reason, action = difference, "regenerated"

    logger.info("writing assets-only text encoder %s (%s)", target.name, reason)
    _write(target, *want)
    size = target.stat().st_size
    logger.info(
        "assets-only text encoder %s: %s (%.1f MiB, %d tensors: %s)",
        target.name, action, size / 2**20, len(names), ", ".join(names),
    )
    return ExportResult(target, action, reason, size, names)


def verify_with_official_loader(path: str | Path) -> dict[str, Any]:
    """Load *path* through the real ``GemmaAssets`` and report what it found.

    This is the proof that the exported file satisfies the official contract --
    the constants at the top of this module are re-declared rather than imported,
    so something has to actually exercise the round trip. Imported lazily and via
    ``engine25.ltxcore_compat`` (never ``ltx_core`` directly), so the exporter
    itself still runs in an interpreter without the official packages.
    """
    from engine25.ltxcore_compat import GemmaAssets

    assets = GemmaAssets.load(str(path))
    return {
        "model_type": assets.config_dict.get("model_type"),
        "gemma_version": assets.config_dict.get("gemma_version"),
        "tokenizer_json_bytes": len(assets.tokenizer_json),
        "sidecars": sorted(assets.sidecars),
        "weight_paths": list(assets.weight_paths),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="engine25.assets_export",
        description="Extract the assets-only text-encoder safetensors from an LTX 2.5 TE GGUF.",
    )
    parser.add_argument("gguf", help="text-encoder GGUF produced by the gemma4-ltx25 converter profile")
    parser.add_argument("--out", default=None, help=f"output path (default: <gguf stem>{ASSETS_SUFFIX} beside it)")
    parser.add_argument("--force", action="store_true", help="rewrite even when the existing file matches")
    parser.add_argument("--verify", action="store_true", help="also load the result through the official GemmaAssets")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="[ltx25_assets] %(asctime)s %(name)s: %(message)s",
    )

    result = ensure_assets_only(args.gguf, args.out, force=args.force)
    report = result.as_dict()
    if args.verify:
        report["official_loader"] = verify_with_official_loader(result.path)
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else result.path)
    return 0


if __name__ == "__main__":
    # Works as `python engine25/assets_export.py` as well as `python -m engine25.assets_export`.
    if __package__ in (None, ""):  # pragma: no cover
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
