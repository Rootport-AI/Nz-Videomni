"""fp8 safetensors transformer: header reader + acceptance check (§3-167 B-1, B-2).

SINGLE SOURCE OF TRUTH for "which fp8 safetensors transformer does this
product accept", shared by:
  * services/model_registry.py ``precheck_model_file`` (API side; a refusal
    becomes MODEL_INCOMPATIBLE 422 before the worker is touched),
  * engine/sft_quant/ (engine side; the same refusal fails the load loudly).

ZERO heavy deps (no torch / numpy / safetensors) so it imports in BOTH the app
venv (.venv) and the engine venv (.venv-engine) — the ``chain_math.py``
precedent. It reads only the JSON header and the few-dozen-byte
``comfy_quant`` tensors; the multi-GB weight body is never read and the file is
never memory-mapped (Windows commit-charge constraint, plan §2).

The acceptance rules were FINALIZED by the owner on 2026-09-25 (criterion: an
fp8 file a typical ComfyUI workflow runs must run here when dropped in). The
canonical statement of the rules is Docs/VERIFICATION_LOG.md §121.3; the code
below implements it and does not restate it.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DTYPE_ITEMSIZE",
    "QuantFormatError",
    "Header",
    "Layout",
    "TensorInfo",
    "detect_prefix",
    "inspect",
    "parse_metadata",
    "read_header",
    "read_tensor_bytes",
]


class QuantFormatError(ValueError):
    """The file is not an fp8 safetensors transformer this product accepts."""


#: safetensors dtype string -> bytes per element.
DTYPE_ITEMSIZE: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}

#: A JSON header beyond this is implausible for these models (same bound as
#: services/model_registry.py's generic safetensors precheck).
MAX_HEADER_LEN = 100 * 1024 * 1024

#: fp8 dtypes accepted as weights / bias (both upcast to bf16 at forward time).
_FP8_DTYPES = frozenset({"F8_E4M3", "F8_E5M2"})
#: comfy_quant / _quantization_metadata ``format`` values that are fp8.
_FP8_FORMATS = frozenset({"float8_e4m3fn", "float8_e5m2"})
_FLOAT_DTYPES = frozenset({"BF16", "F32"})
_SCALE_SUFFIX = ".weight_scale"
_QUANT_SUFFIX = ".comfy_quant"
#: Last key segments of the old ComfyUI "scaled_fp8" layout (not supported).
_LEGACY_LEAVES = frozenset({"scaled_fp8", "scale_weight"})
_CONNECTOR_MARK = "_embeddings_connector."
#: Text-encoder side projection; a bare-named file keeps it inside the (empty)
#: prefix, so it is excluded from ``scaled_layers`` like the connectors.
_TEXT_PROJ_HEAD = "text_embedding_projection."
#: Key prefix of a ComfyUI-style file; a bare-named file has none ("").
_COMFY_PREFIX = "model.diffusion_model."
_FIRST_BLOCK = "transformer_blocks.0."


@dataclass(frozen=True)
class TensorInfo:
    dtype: str  # safetensors spelling: "BF16" "F32" "F8_E4M3" "U8" ...
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]  # relative to Header.data_base


@dataclass(frozen=True)
class Header:
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str]  # __metadata__ ({} when absent)
    header_len: int
    data_base: int  # = 8 + header_len (absolute offset of the data section)
    file_size: int


@dataclass(frozen=True)
class Layout:
    flavor: str  # "scaled" if any layer is scaled, else "plain" (kept for compatibility)
    config: dict  # json.loads(metadata["config"]) (required)
    model_version: str | None  # metadata.get("model_version")
    prefix: str  # "model.diffusion_model." or "" (bare names), see detect_prefix
    scaled_layers: frozenset[str]  # prefix-less transformer Linear names, ".weight" removed (plain: empty)
    connector_keys: tuple[str, ...]  # full (with prefix) "*_embeddings_connector.*" keys
    n_blocks: int


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #


def read_header(path) -> Header:
    """Parse and structurally validate a safetensors header (no weight I/O)."""
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise QuantFormatError("safetensors として読めません: 先頭 8 バイトのヘッダ長がありません")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len == 0 or header_len > file_size - 8 or header_len > MAX_HEADER_LEN:
            raise QuantFormatError(
                f"safetensors として読めません: ヘッダ長 {header_len} が不正です"
                f"（ファイル長 {file_size}・上限 {MAX_HEADER_LEN}）"
            )
        blob = fh.read(header_len)
    try:
        doc = json.loads(blob.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — any decode failure is a format error
        raise QuantFormatError(f"safetensors として読めません: ヘッダの JSON が壊れています（{exc}）") from exc
    if not isinstance(doc, dict):
        raise QuantFormatError("safetensors として読めません: ヘッダが JSON オブジェクトではありません")

    metadata = doc.pop("__metadata__", None) or {}
    if not isinstance(metadata, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
    ):
        raise QuantFormatError("safetensors として読めません: __metadata__ が文字列→文字列の表ではありません")

    data_base = 8 + header_len
    data_len = file_size - data_base
    tensors: dict[str, TensorInfo] = {}
    for key, entry in doc.items():
        tensors[key] = _tensor_info(key, entry, data_len)
    return Header(
        tensors=tensors,
        metadata=dict(metadata),
        header_len=header_len,
        data_base=data_base,
        file_size=file_size,
    )


def _tensor_info(key: str, entry, data_len: int) -> TensorInfo:
    try:
        dtype = entry["dtype"]
        shape = tuple(int(d) for d in entry["shape"])
        begin, end = (int(o) for o in entry["data_offsets"])
    except Exception as exc:  # noqa: BLE001
        raise QuantFormatError(f"safetensors として読めません: '{key}' の記述が不完全です（{exc}）") from exc
    if dtype not in DTYPE_ITEMSIZE:
        raise QuantFormatError(f"safetensors として読めません: '{key}' の dtype '{dtype}' は未知です")
    if any(d < 0 for d in shape):
        raise QuantFormatError(f"safetensors として読めません: '{key}' の shape {list(shape)} が不正です")
    if not (0 <= begin <= end <= data_len):
        raise QuantFormatError(
            f"safetensors として読めません: '{key}' の data_offsets [{begin}, {end}] が"
            f"データ領域 [0, {data_len}] の外です"
        )
    count = 1
    for d in shape:
        count *= d
    if end - begin != count * DTYPE_ITEMSIZE[dtype]:
        raise QuantFormatError(
            f"safetensors として読めません: '{key}' の長さ {end - begin} バイトが"
            f" {dtype}{list(shape)} の {count * DTYPE_ITEMSIZE[dtype]} バイトと一致しません"
        )
    return TensorInfo(dtype=dtype, shape=shape, data_offsets=(begin, end))


def read_tensor_bytes(path, header: Header, key: str) -> bytes:
    """Raw bytes of ONE small tensor (comfy_quant and the like). seek + read, never a memory map."""
    with Path(path).open("rb") as fh:
        return _read_range(fh, header, key)


def _read_range(fh, header: Header, key: str) -> bytes:
    begin, end = header.tensors[key].data_offsets
    fh.seek(header.data_base + begin)
    data = fh.read(end - begin)
    if len(data) != end - begin:
        raise QuantFormatError(f"safetensors の読み取りが途中で終わりました: '{key}'")
    return data


def parse_metadata(header: Header) -> dict:
    """``__metadata__`` with each value JSON-parsed when it is valid JSON, else the raw string.

    Same shape as ltx_core 1.2's ``SafetensorsModelStateDictLoader.metadata``
    (``ltx_core/loader/sft_loader.py``), without opening the file again.
    """
    parsed: dict = {}
    for key, value in header.metadata.items():
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


# --------------------------------------------------------------------------- #
# acceptance (owner ruling 2026-09-25: "an fp8 file a typical ComfyUI workflow
# runs must run here when dropped in")
# --------------------------------------------------------------------------- #

_NG = "fp8 safetensors の検査に不合格: "


def detect_prefix(header: Header) -> str:
    """The transformer key prefix: ``"model.diffusion_model."`` or ``""`` (bare names).

    Judged by where ``transformer_blocks.0.`` lives; neither -> refused.
    """
    keys = header.tensors
    if any(k.startswith(_COMFY_PREFIX + _FIRST_BLOCK) for k in keys):
        return _COMFY_PREFIX
    if any(k.startswith(_FIRST_BLOCK) for k in keys):
        return ""
    raise QuantFormatError(
        _NG + f"'{_COMFY_PREFIX}{_FIRST_BLOCK}' も '{_FIRST_BLOCK}' もありません"
        "（接頭辞が違うか、transformer ではありません）"
    )


def inspect(path, *, expected_blocks: int = 48) -> Layout:
    """Accept or refuse an fp8 safetensors transformer.

    Rules finalized by the owner on 2026-09-25 (canonical: Docs/VERIFICATION_LOG.md
    §121.3). Failures raise :class:`QuantFormatError` naming the failing spot in
    one line.

    Reads the header plus the ``comfy_quant`` payloads only. The key prefix is
    detected (:func:`detect_prefix`). Tensors outside it (``vae.*`` /
    ``audio_vae.*`` / ``vocoder.*`` / ``text_embedding_projection.*`` of a
    monolithic checkpoint) are ignored — except that no fp8 tensor may live
    there. A bare-named file (prefix ``""``) has nothing outside.
    """
    header = read_header(path)
    tensors = header.tensors

    # 1) __metadata__.config with a transformer section (required: without it
    #    ComfyUI cannot build the LTX model either)
    raw_config = header.metadata.get("config")
    if raw_config is None:
        raise QuantFormatError(_NG + "__metadata__ に config がありません（ComfyUI でも LTX として組めない形です）")
    try:
        config = json.loads(raw_config)
    except Exception as exc:  # noqa: BLE001
        raise QuantFormatError(_NG + f"__metadata__.config が JSON として読めません（{exc}）") from exc
    if not isinstance(config, dict) or "transformer" not in config:
        raise QuantFormatError(_NG + "__metadata__.config に transformer がありません")

    # 2) prefix + block count
    prefix = detect_prefix(header)
    block_re = re.compile(re.escape(prefix) + r"transformer_blocks\.(\d+)\.")
    blocks = {int(m.group(1)) for k in tensors if (m := block_re.match(k))}
    if blocks != set(range(expected_blocks)):
        raise QuantFormatError(
            _NG + f"transformer_blocks が {len(blocks)} 個（番号 {min(blocks)}..{max(blocks)}）で、"
            f"{expected_blocks} 個（0..{expected_blocks - 1}）ではありません"
        )

    # 3) dtypes / fp8 placement / old layout
    fp8_weights: list[str] = []
    for key, info in tensors.items():
        is_fp8 = info.dtype in _FP8_DTYPES
        if info.dtype.startswith("F8_") and not is_fp8:
            raise QuantFormatError(_NG + f"'{key}' の {info.dtype} は未対応です（受理: F8_E4M3・F8_E5M2）")
        if not key.startswith(prefix):
            if is_fp8:
                raise QuantFormatError(_NG + f"fp8 テンソル '{key}' が '{prefix}' の外にあります")
            continue
        if key.rsplit(".", 1)[-1] in _LEGACY_LEAVES:
            raise QuantFormatError(_NG + f"旧形式（scaled_fp8／scale_weight）は未対応です（'{key}'）")
        if is_fp8:
            is_weight = key.endswith(".weight") and len(info.shape) == 2
            is_bias = key.endswith(".bias") and len(info.shape) == 1
            if not (is_weight or is_bias):
                raise QuantFormatError(
                    _NG + f"fp8 は 2 次元 .weight か 1 次元 .bias に限ります（'{key}' shape {list(info.shape)}）"
                )
            if is_weight:
                fp8_weights.append(key)
            continue
        if info.dtype == "U8" and key.endswith(_QUANT_SUFFIX):
            continue
        if info.dtype not in _FLOAT_DTYPES:
            raise QuantFormatError(
                _NG + f"'{key}' の dtype {info.dtype} は未対応です"
                "（fp8 以外の量子化の可能性。受理: BF16・F32・fp8・comfy_quant の U8）"
            )
    if not fp8_weights:
        raise QuantFormatError(_NG + "fp8 の重みが 1 本もありません（bf16 等の非 fp8 ファイルは未対応です）")

    # 4) per-layer scales and quantization markers (connectors included: same
    #    rules). Only the transformer's own Linears go to scaled_layers — the
    #    connectors and text_embedding_projection belong to the text encoder side.
    scaled = _scaled_layers(path, header, prefix, {k[: -len(".weight")] for k in fp8_weights})
    scaled = {
        layer for layer in scaled
        if _CONNECTOR_MARK not in layer and not layer[len(prefix):].startswith(_TEXT_PROJ_HEAD)
    }

    # 5) Gemma-side embeddings connector must be present (dtypes were checked in step 3)
    connector_keys = tuple(sorted(k for k in tensors if k.startswith(prefix) and _CONNECTOR_MARK in k))
    if not connector_keys:
        raise QuantFormatError(_NG + f"'{prefix}*{_CONNECTOR_MARK}*'（テキスト埋め込みの connector）がありません")

    return Layout(
        flavor="scaled" if scaled else "plain",
        config=config,
        model_version=header.metadata.get("model_version"),
        prefix=prefix,
        scaled_layers=frozenset(layer[len(prefix):] for layer in scaled),
        connector_keys=connector_keys,
        n_blocks=len(blocks),
    )


def _scaled_layers(path, header: Header, prefix: str, fp8_layers: set[str]) -> set[str]:
    """Judge scales and markers layer by layer; return the prefixed scaled layers.

    * ``<layer>.weight_scale`` present -> scaled (must be a scalar F32 — shape
      ``()`` or ``(1,)`` — and belong to an fp8 ``.weight``); absent -> plain
      cast. Both may coexist.
    * ``<layer>.comfy_quant`` (optional) and ``__metadata__._quantization_metadata``
      (optional) must name fp8 formats only.
    * ``<layer>.input_scale`` is allowed and ignored (activation scale; this
      engine computes in bf16).
    """
    tensors = header.tensors
    scaled: set[str] = set()
    quant_keys: list[str] = []
    for key, info in tensors.items():
        if not key.startswith(prefix):
            continue
        if key.endswith(_QUANT_SUFFIX):
            quant_keys.append(key)
        if not key.endswith(_SCALE_SUFFIX):
            continue
        layer = key[: -len(_SCALE_SUFFIX)]
        if layer not in fp8_layers:
            raise QuantFormatError(_NG + f"倍率 '{key}' に対応する fp8 の .weight がありません（孤立した倍率）")
        if info.dtype != "F32" or info.shape not in ((), (1,)):
            raise QuantFormatError(
                _NG + f"倍率 '{key}' が {info.dtype}{list(info.shape)} です"
                "（受理するのは F32 のスカラー倍率のみ。per-row／per-block は未対応）"
            )
        scaled.add(layer)

    if quant_keys:
        with Path(path).open("rb") as fh:
            for key in sorted(quant_keys):
                if len(tensors[key].shape) != 1:
                    raise QuantFormatError(_NG + f"'{key}' は 1 次元の U8 ではありません")
                fmt = _comfy_format(key, _read_range(fh, header, key))
                if fmt not in _FP8_FORMATS:
                    raise QuantFormatError(_NG + f"量子化の印 format='{fmt}' は fp8 ではなく未対応です（'{key}'）")

    for fmt in _metadata_formats(header.metadata.get("_quantization_metadata")):
        if fmt not in _FP8_FORMATS:
            raise QuantFormatError(
                _NG + f"__metadata__._quantization_metadata に fp8 以外の format '{fmt}' があり未対応です"
            )
    return scaled


def _metadata_formats(raw) -> list[str]:
    """Every string under a ``"format"`` key anywhere in the JSON (unreadable -> [])."""
    if not isinstance(raw, str):
        return []
    try:
        doc = json.loads(raw)
    except Exception:  # noqa: BLE001 — a structure we cannot read is ignored (ruling)
        return []
    found: list[str] = []
    stack = [doc]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "format" and isinstance(v, str):
                    found.append(v)
                else:
                    stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    return found


def _comfy_format(key: str, data: bytes):
    try:
        doc = json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise QuantFormatError(_NG + f"'{key}' が JSON として読めません（{exc}）") from exc
    if not isinstance(doc, dict):
        raise QuantFormatError(_NG + f"'{key}' が JSON オブジェクトではありません")
    return doc.get("format")
