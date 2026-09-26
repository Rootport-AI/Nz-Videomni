"""Quantized safetensors transformer: header reader + acceptance check (§3-167, §3-168).

SINGLE SOURCE OF TRUTH for "which quantized (fp8 / int8) safetensors
transformer does this product accept, and how is each layer stored", shared by:
  * services/model_registry.py ``precheck_model_file`` (API side; a refusal
    becomes MODEL_INCOMPATIBLE 422 before the worker is touched),
  * engine/sft_quant/ and engine25/ (engine side; the same refusal fails the
    load loudly, and :data:`SCHEME_TABLE` drives the auxiliary tensors).

ZERO heavy deps (no torch / numpy / safetensors) so it imports in BOTH the app
venv (.venv) and the engine venv (.venv-engine) — the ``chain_math.py``
precedent. It reads only the JSON header and the few-dozen-byte
``comfy_quant`` tensors; the multi-GB weight body is never read and the file is
never memory-mapped (Windows commit-charge constraint).

Criterion (owner): a file a typical ComfyUI workflow runs must run here when
dropped in. The canonical statement of the rules is
Docs/VERIFICATION_LOG.md §121.3; the code below implements it and does not
restate it.
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
    "AuxRule",
    "Header",
    "Layout",
    "Scheme",
    "SCHEMES",
    "SCHEME_TABLE",
    "SKIPPED_SUFFIXES",
    "TensorInfo",
    "aux_shape",
    "AUX_NAMES",
    "aux_specs",
    "detect_prefix",
    "inspect",
    "layer_schemes",
    "parse_metadata",
    "read_header",
    "read_tensor_bytes",
    "weight_shape",
]


class QuantFormatError(ValueError):
    """The file is not a quantized safetensors transformer this product accepts."""


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


# --------------------------------------------------------------------------- #
# the scheme table (ONE table; placement, aux_specs and weight_shape derive from it)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AuxRule:
    """One auxiliary tensor ``<layer>.<name>`` of a scheme.

    ``raw_dtypes`` / ``raw_shapes`` are what a file may hold; ``dtype`` /
    ``shape`` are what the loader normalizes it to. Shape rules are the
    strings :func:`aux_shape` understands.
    """

    raw_dtypes: frozenset[str]
    raw_shapes: tuple[str, ...]
    dtype: str
    shape: str


@dataclass(frozen=True)
class Scheme:
    """How one quantized Linear is stored.

    ``weight`` is the raw dtype class of ``.weight`` (a key of
    ``_WEIGHT_DTYPES``); ``packing`` input columns share one stored column
    (see :func:`weight_shape`); the input dimension must be a multiple of
    ``in_multiple``; ``quant_bias`` lets ``.bias`` share the weight's dtype.
    """

    weight: str
    aux: dict[str, AuxRule]
    packing: int = 1
    in_multiple: int = 1
    quant_bias: bool = False


_SCALAR_SCALE = AuxRule(frozenset({"F32"}), ("()", "(1,)"), "F32", "()")
_ROW_SCALE = AuxRule(frozenset({"F32"}), ("()", "(1,)", "(o,1)"), "F32", "(o,1)")

#: Scheme name -> storage. C-3 adds the "w4a8" row.
SCHEME_TABLE: dict[str, Scheme] = {
    "fp8": Scheme(weight="F8", aux={}, quant_bias=True),
    "fp8_scaled": Scheme(weight="F8", aux={"weight_scale": _SCALAR_SCALE}, quant_bias=True),
    "int8": Scheme(weight="I8", aux={"weight_scale": _ROW_SCALE}),
    "int8_convrot": Scheme(weight="I8", aux={"weight_scale": _ROW_SCALE}, in_multiple=256),
}

SCHEMES: tuple[str, ...] = tuple(SCHEME_TABLE)

#: Leaves read past everywhere (activation scale / the marker itself).
SKIPPED_SUFFIXES: tuple[str, ...] = (".input_scale", ".comfy_quant")

#: Raw ``.weight`` dtypes of each weight class used in SCHEME_TABLE.
_WEIGHT_DTYPES: dict[str, frozenset[str]] = {
    "F8": frozenset({"F8_E4M3", "F8_E5M2"}),
    "I8": frozenset({"I8"}),
}
#: Marker ``format`` -> the weight class it describes. Anything else is refused.
_FORMATS: dict[str, str] = {
    "float8_e4m3fn": "F8",
    "float8_e5m2": "F8",
    "int8_tensorwise": "I8",
}
_CONVROT_GROUPSIZE = 256

_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})
_MARKER_DTYPES = frozenset({"U8", "I8"})
_QUANT_WEIGHT_DTYPES = frozenset().union(*_WEIGHT_DTYPES.values())
#: Every dtype accepted inside the transformer prefix.
_ACCEPTED_DTYPES = _FLOAT_DTYPES | _QUANT_WEIGHT_DTYPES | _MARKER_DTYPES
#: Union of every scheme's auxiliary names (an orphan of one is refused; the
#: loaders use it to recognise auxiliary leaves).
AUX_NAMES: frozenset[str] = frozenset(name for s in SCHEME_TABLE.values() for name in s.aux)


def _placement() -> dict[str, frozenset[str]]:
    """Non-float dtype -> the leaves it may be stored under (derived from SCHEME_TABLE)."""
    place: dict[str, set[str]] = {d: set() for d in _QUANT_WEIGHT_DTYPES | _MARKER_DTYPES}
    for d in _MARKER_DTYPES:
        place[d].add("comfy_quant")
    for scheme in SCHEME_TABLE.values():
        for d in _WEIGHT_DTYPES[scheme.weight]:
            place[d].add("weight")
            if scheme.quant_bias:
                place[d].add("bias")
        for name, rule in scheme.aux.items():
            for d in rule.raw_dtypes & place.keys():
                place[d].add(name)
    return {d: frozenset(leaves) for d, leaves in place.items()}


#: I8 -> {weight, comfy_quant} / U8 -> {comfy_quant} / F8_* -> {weight, bias}
_PLACEMENT: dict[str, frozenset[str]] = _placement()


def aux_specs(scheme: str) -> dict[str, tuple[str, str]]:
    """Auxiliary name -> (normalized safetensors dtype, normalized shape rule)."""
    return {name: (rule.dtype, rule.shape) for name, rule in SCHEME_TABLE[scheme].aux.items()}


def aux_shape(rule: str, o: int, i: int) -> tuple[int, ...]:
    """A shape rule made concrete for a Linear with ``o`` outputs and ``i`` inputs."""
    if rule == "()":
        return ()
    if rule == "(1,)":
        return (1,)
    if rule == "(o,1)":
        return (o, 1)
    if rule == "(o,)":
        return (o,)
    if rule == "(16,)":
        return (16,)
    if rule == "(o,i/16)":
        return (o, i // 16)
    raise ValueError(f"unknown shape rule {rule!r}")


def weight_shape(scheme: str, o: int, i: int) -> tuple[int, int]:
    """Stored ``.weight`` shape of a Linear with ``o`` outputs and ``i`` inputs."""
    return (o, i // SCHEME_TABLE[scheme].packing)


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #

#: Last key segments of the old ComfyUI "scaled_fp8" layout (not supported).
_LEGACY_LEAVES = frozenset({"scaled_fp8", "scale_weight"})
_CONNECTOR_MARK = "_embeddings_connector."
#: Text-encoder side projection; a bare-named file keeps it inside the (empty)
#: prefix, so it is excluded from ``Layout.layers`` like the connectors.
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
    config: dict  # json.loads(metadata["config"]) (required)
    model_version: str | None  # metadata.get("model_version")
    prefix: str  # "model.diffusion_model." or "" (bare names), see detect_prefix
    #: prefix-less transformer Linear name (".weight" removed) -> scheme. Only
    #: quantized layers (bf16 ones are absent); the connectors and
    #: text_embedding_projection are excluded.
    layers: dict[str, str]
    connector_keys: tuple[str, ...]  # full (with prefix) "*_embeddings_connector.*" keys
    n_blocks: int

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
# acceptance (owner: "a file a typical ComfyUI workflow runs must run here when
# dropped in"; canonical rules: VERIFICATION_LOG §121.3)
# --------------------------------------------------------------------------- #

_NG = "量子化 safetensors の検査に不合格: "


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
    """Accept or refuse a quantized safetensors transformer.

    Failures raise :class:`QuantFormatError` naming the failing spot in one
    line. Reads the header plus the ``comfy_quant`` payloads only. Tensors
    outside the detected prefix (``vae.*`` / ``audio_vae.*`` / ``vocoder.*`` /
    ``text_embedding_projection.*`` of a monolithic checkpoint) are ignored —
    except that no quantized weight dtype may live there.
    """
    # ① header
    header = read_header(path)
    tensors = header.tensors

    # ② __metadata__.config with a transformer section (required: without it
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

    # ③ prefix + block count
    prefix = detect_prefix(header)
    block_re = re.compile(re.escape(prefix) + r"transformer_blocks\.(\d+)\.")
    blocks = {int(m.group(1)) for k in tensors if (m := block_re.match(k))}
    if blocks != set(range(expected_blocks)):
        raise QuantFormatError(
            _NG + f"transformer_blocks が {len(blocks)} 個（番号 {min(blocks)}..{max(blocks)}）で、"
            f"{expected_blocks} 個（0..{expected_blocks - 1}）ではありません"
        )

    # ④ scheme of every quantized layer (connectors included: same rules)
    schemes = layer_schemes(path, header, prefix)

    # ⑤ at least one quantized weight
    if not schemes:
        raise QuantFormatError(_NG + "量子化された重みが 1 本もありません（bf16 等の非量子化ファイルは未対応です）")

    # ⑥ Gemma-side embeddings connector must be present
    connector_keys = tuple(sorted(k for k in tensors if k.startswith(prefix) and _CONNECTOR_MARK in k))
    if not connector_keys:
        raise QuantFormatError(_NG + f"'{prefix}*{_CONNECTOR_MARK}*'（テキスト埋め込みの connector）がありません")

    # Only the transformer's own Linears go to Layout.layers — the connectors
    # and text_embedding_projection belong to the text encoder side.
    layers = {}
    for layer, scheme in schemes.items():
        name = layer[len(prefix):]
        if _CONNECTOR_MARK in layer or name.startswith(_TEXT_PROJ_HEAD):
            continue
        layers[name] = scheme
    return Layout(
        config=config,
        model_version=header.metadata.get("model_version"),
        prefix=prefix,
        layers=layers,
        connector_keys=connector_keys,
        n_blocks=len(blocks),
    )


def layer_schemes(path, header: Header, prefix: str) -> dict[str, str]:
    """Prefixed Linear name (".weight" removed) -> scheme, for every quantized layer.

    Top-down (§121.3 ④):
      A  global dtype / placement checks, key by key;
      B  keys grouped into layer (all but the last segment) and leaf (the last);
      C  markers resolved per layer (``__metadata__._quantization_metadata.layers``
         first, else ``<layer>.comfy_quant``), ``params`` flattened, format checked;
      D  per layer: scheme from (weight dtype class, format, convrot), then the
         auxiliary set must equal the scheme's, each aux dtype/shape checked.
    Connectors and text_embedding_projection are included (the caller filters).
    """
    tensors = header.tensors

    # A — dtypes and placement
    groups: dict[str, dict[str, str]] = {}  # B is folded into the same pass
    for key, info in tensors.items():
        dtype = info.dtype
        if dtype.startswith("F8_") and dtype not in _QUANT_WEIGHT_DTYPES:
            raise QuantFormatError(_NG + f"'{key}' の {dtype} は未対応です（受理: F8_E4M3・F8_E5M2）")
        if not key.startswith(prefix):
            if dtype in _QUANT_WEIGHT_DTYPES:
                raise QuantFormatError(_NG + f"量子化テンソル '{key}' が '{prefix}' の外にあります")
            continue
        layer, _, leaf = key.rpartition(".")
        if leaf in _LEGACY_LEAVES:
            raise QuantFormatError(_NG + f"旧形式（scaled_fp8／scale_weight）は未対応です（'{key}'）")
        if dtype not in _ACCEPTED_DTYPES:
            raise QuantFormatError(
                _NG + f"'{key}' の dtype {dtype} は未対応です"
                "（受理: BF16・F16・F32・F8_E4M3・F8_E5M2・I8・U8）"
            )
        if dtype in _PLACEMENT:
            places = _PLACEMENT[dtype]
            if leaf not in places:
                allowed = "・".join("." + p for p in sorted(places))
                raise QuantFormatError(_NG + f"{dtype} を置けるのは {allowed} だけです（'{key}'）")
            if leaf == "weight" and len(info.shape) != 2:
                raise QuantFormatError(
                    _NG + f"量子化された .weight は 2 次元に限ります（'{key}' shape {list(info.shape)}）"
                )
            if leaf in ("bias", "comfy_quant") and len(info.shape) != 1:
                raise QuantFormatError(
                    _NG + f"{dtype} の .{leaf} は 1 次元に限ります（'{key}' shape {list(info.shape)}）"
                )
        # B — layer / leaf
        groups.setdefault(layer, {})[leaf] = key

    # C — markers, per layer
    markers = _resolve_markers(path, header, groups)

    # D — scheme per layer
    schemes: dict[str, str] = {}
    for layer, leaves in groups.items():
        weight_key = leaves.get("weight")
        weight_dtype = tensors[weight_key].dtype if weight_key else None
        if weight_dtype not in _QUANT_WEIGHT_DTYPES:
            orphans = sorted(AUX_NAMES & leaves.keys())
            if orphans:
                raise QuantFormatError(
                    _NG + f"倍率 '{leaves[orphans[0]]}' に対応する量子化された .weight がありません（孤立した倍率）"
                )
            continue  # a float layer: bf16 whatever its marker says
        scheme = _scheme_of(layer, weight_key, weight_dtype, leaves, markers.get(layer))
        _check_aux(layer, scheme, tensors[weight_key].shape, leaves, tensors)
        schemes[layer] = scheme
    return schemes


def _scheme_of(layer: str, weight_key: str, weight_dtype: str, leaves: dict, marker) -> str:
    """D: (weight dtype class, marker format, convrot) -> scheme."""
    fmt, conf, where = marker if marker is not None else (None, {}, None)
    if weight_dtype in _WEIGHT_DTYPES["F8"]:
        if fmt is not None and _FORMATS[fmt] != "F8":
            raise QuantFormatError(
                _NG + f"量子化の印 format='{fmt}' と重みの dtype {weight_dtype} が合いません（{where}）"
            )
        return "fp8_scaled" if "weight_scale" in leaves else "fp8"
    # I8: the marker is mandatory
    if fmt is None:
        raise QuantFormatError(
            _NG + f"I8 の重み '{weight_key}' に量子化の印がありません"
            "（comfy_quant も __metadata__._quantization_metadata もありません）"
        )
    if _FORMATS[fmt] != "I8":
        raise QuantFormatError(
            _NG + f"量子化の印 format='{fmt}' と重みの dtype {weight_dtype} が合いません（{where}）"
        )
    convrot = conf.get("convrot", False)
    if not isinstance(convrot, bool):
        raise QuantFormatError(_NG + f"量子化の印の convrot が真偽値ではありません（{convrot!r}・{where}）")
    groupsize = conf.get("convrot_groupsize", _CONVROT_GROUPSIZE)
    if groupsize != _CONVROT_GROUPSIZE:
        raise QuantFormatError(
            _NG + f"convrot_groupsize={groupsize!r} は未対応です（受理: {_CONVROT_GROUPSIZE} のみ・{where}）"
        )
    return "int8_convrot" if convrot else "int8"


def _check_aux(layer: str, scheme: str, stored: tuple[int, ...], leaves: dict, tensors: dict) -> None:
    """D: the aux set equals the scheme's; each aux dtype/shape; the input-dim constraint."""
    row = SCHEME_TABLE[scheme]
    present = {leaf for leaf in leaves if leaf not in ("weight", "bias") and "." + leaf not in SKIPPED_SUFFIXES}
    missing = sorted(row.aux.keys() - present)
    extra = sorted(present - row.aux.keys())
    if missing:
        raise QuantFormatError(_NG + f"'{layer}'（方式 {scheme}）に補助テンソル {missing} がありません")
    if extra:
        raise QuantFormatError(
            _NG + f"'{layer}'（方式 {scheme}）に未対応の補助テンソル {extra} があります（未対応の量子化の可能性）"
        )
    o, i = stored[0], stored[1] * row.packing
    for name, rule in row.aux.items():
        key = leaves[name]
        info = tensors[key]
        shapes = [aux_shape(r, o, i) for r in rule.raw_shapes]
        if info.dtype not in rule.raw_dtypes or info.shape not in shapes:
            accepted = "・".join(sorted(rule.raw_dtypes)) + " の " + "／".join(str(list(s)) for s in shapes)
            label = "倍率" if name == "weight_scale" else "補助テンソル"
            raise QuantFormatError(
                _NG + f"{label} '{key}' が {info.dtype}{list(info.shape)} です"
                f"（方式 {scheme} で受理するのは {accepted} のみ）"
            )
    if i % row.in_multiple:
        raise QuantFormatError(
            _NG + f"'{layer}'（方式 {scheme}）の入力次元 {i} が {row.in_multiple} の倍数ではありません"
        )


def _resolve_markers(path, header: Header, groups: dict[str, dict[str, str]]) -> dict[str, tuple]:
    """C: layer -> (format, flattened marker, where). Metadata first, then comfy_quant.

    Mirrors ComfyUI (``utils.convert_old_quants`` writes each metadata entry
    over ``<layer>.comfy_quant``; ``ops`` flattens ``params``). Unknown keys and
    ``full_precision_matrix_mult`` are ignored. An unreadable
    ``_quantization_metadata`` is ignored as a whole (as before).
    """
    resolved: dict[str, tuple] = {}
    for layer, conf in _metadata_layers(header.metadata.get("_quantization_metadata")).items():
        where = f"__metadata__._quantization_metadata.layers['{layer}']"
        resolved[layer] = _read_marker(conf, where)

    tensor_markers = sorted(
        (layer, leaves["comfy_quant"])
        for layer, leaves in groups.items()
        if "comfy_quant" in leaves and layer not in resolved
    )
    if tensor_markers:
        with Path(path).open("rb") as fh:
            for layer, key in tensor_markers:
                resolved[layer] = _read_marker(_comfy_json(key, _read_range(fh, header, key)), f"'{key}'")
    return resolved


def _read_marker(conf, where: str) -> tuple:
    if not isinstance(conf, dict):
        raise QuantFormatError(_NG + f"量子化の印が JSON オブジェクトではありません（{where}）")
    params = conf.get("params", {})
    if not isinstance(params, dict):
        raise QuantFormatError(_NG + f"量子化の印の params が JSON オブジェクトではありません（{where}）")
    conf = {**params, **conf}
    fmt = conf.get("format")
    if fmt is None:
        raise QuantFormatError(_NG + f"量子化の印に format がありません（旧 INT8-Fast 形式などは未対応・{where}）")
    if fmt not in _FORMATS:
        accepted = "・".join(_FORMATS)
        raise QuantFormatError(_NG + f"量子化の印 format='{fmt}' は未対応です（受理: {accepted}・{where}）")
    return fmt, conf, where


def _metadata_layers(raw) -> dict:
    """``_quantization_metadata`` -> its ``layers`` table (unreadable or absent -> {})."""
    if not isinstance(raw, str):
        return {}
    try:
        doc = json.loads(raw)
    except Exception:  # noqa: BLE001 — a structure we cannot read is ignored (ruling)
        return {}
    layers = doc.get("layers") if isinstance(doc, dict) else None
    return layers if isinstance(layers, dict) else {}


def _comfy_json(key: str, data: bytes):
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise QuantFormatError(_NG + f"'{key}' が JSON として読めません（{exc}）") from exc
