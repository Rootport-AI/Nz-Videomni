#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""§3-168 G1(ii) 実ファイルの全層の方式判定の集計。

safetensors の**ヘッダだけ**（先頭8バイトの長さ＋JSON）と、`comfy_quant` の
数十〜数百バイトのテンソルだけを seek + read で読む（mmap も全読みもしない、
Windows コミット制約への配慮は §3-167 以来の方針を踏襲）。ヘッダの構造解釈は
このファイル自身が行う（製品側 `sft_fp8_format.py` は import しない -- あちら
は fp8 専用の受理／拒否ロジックを持つが、ここでは受理判定はせず**集計だけ**を
行うため独立に実装する）。

層ごとの印（marker）の解釈は変換ツール
`Nz-GGUF-Converter-LTX23/src/converter/comfy_dequant.py` の
`parse_quant_marker` にそのまま委譲する（変換ツールは編集しない・
`sys.path` に `src` を足して import するだけ）。

印の解決規則（ComfyUI `utils.py` の `convert_old_quants` と同じ、計画 §2.1）:
層ごとに、``__metadata__._quantization_metadata.layers`` にその層があれば
それ、無ければ ``<層>.comfy_quant``（U8 または I8 の1次元・JSON）。

使い方:
    python g1_census.py PATH [--json OUT.json] [--product]

``--product`` は製品側の判定（``sft_quant_format.layer_schemes``、§3-168 で
これから実装される）との突き合わせをオンにするスタブ。今はモジュールが存在
しないため、フラグを立てても「未実装」の1行を出すだけで正常終了する。
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- 変換ツール（編集しない・importするだけ） --------------------------------
CONVERTER_SRC = Path(r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-GGUF-Converter-LTX23\src")
if str(CONVERTER_SRC) not in sys.path:
    sys.path.insert(0, str(CONVERTER_SRC))

from converter.comfy_dequant import (  # noqa: E402  (sys.path fixup must run first)
    ComfyDequantError,
    SUPPORTED_FORMATS,
    parse_quant_marker,
)


# === safetensors ヘッダの自前読み（mmap・全読み禁止） ==========================
#: safetensors dtype 文字列 -> バイト/要素。sft_fp8_format.py の同名テーブルと
#: 同じ既知の事実を独立に書いたもの（import はしない）。
DTYPE_ITEMSIZE: dict[str, int] = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}

#: このプロダクトの transformer が使う知られたヘッダ長の上限（sft_fp8_format.py
#: と同じ桁。壊れたファイルを大きく読み込まないための素朴な安全弁）。
MAX_HEADER_LEN = 100 * 1024 * 1024

_COMFY_PREFIX = "model.diffusion_model."
_FIRST_BLOCK = "transformer_blocks.0."
_QUANT_SUFFIX = ".comfy_quant"
_METADATA_QUANT_KEY = "_quantization_metadata"


class CensusError(ValueError):
    """ヘッダが safetensors として読めない（構造エラー。量子化の判定失敗とは別）。"""


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]  # data_base からの相対オフセット


@dataclass(frozen=True)
class Header:
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str]
    header_len: int
    data_base: int
    file_size: int


def read_header(path: Path) -> Header:
    """ヘッダ（先頭8バイトの長さ＋JSON）だけを読む。データ本体には触れない。"""
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise CensusError("safetensors として読めません: 先頭8バイトのヘッダ長がありません")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len == 0 or header_len > file_size - 8 or header_len > MAX_HEADER_LEN:
            raise CensusError(
                f"safetensors として読めません: ヘッダ長 {header_len} が不正です"
                f"（ファイル長 {file_size}・上限 {MAX_HEADER_LEN}）"
            )
        blob = fh.read(header_len)
    try:
        doc = json.loads(blob.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CensusError(f"safetensors として読めません: ヘッダの JSON が壊れています（{exc}）") from exc
    if not isinstance(doc, dict):
        raise CensusError("safetensors として読めません: ヘッダが JSON オブジェクトではありません")

    metadata = doc.pop("__metadata__", None) or {}
    if not isinstance(metadata, dict):
        raise CensusError("safetensors として読めません: __metadata__ がオブジェクトではありません")

    data_base = 8 + header_len
    data_len = file_size - data_base
    tensors: dict[str, TensorInfo] = {}
    for key, entry in doc.items():
        try:
            dtype = entry["dtype"]
            shape = tuple(int(d) for d in entry["shape"])
            begin, end = (int(o) for o in entry["data_offsets"])
        except Exception as exc:  # noqa: BLE001
            raise CensusError(f"safetensors として読めません: '{key}' の記述が不完全です（{exc}）") from exc
        if not (0 <= begin <= end <= data_len):
            raise CensusError(
                f"safetensors として読めません: '{key}' の data_offsets [{begin}, {end}] が"
                f" データ領域 [0, {data_len}] の外です"
            )
        tensors[key] = TensorInfo(dtype=dtype, shape=shape, data_offsets=(begin, end))
    return Header(tensors=dict(tensors), metadata=dict(metadata), header_len=header_len,
                  data_base=data_base, file_size=file_size)


def read_tensor_bytes(path: Path, header: Header, key: str) -> bytes:
    """1本の小さなテンソル（comfy_quant 等）の生バイトだけを seek + read する。"""
    info = header.tensors[key]
    begin, end = info.data_offsets
    with path.open("rb") as fh:
        fh.seek(header.data_base + begin)
        data = fh.read(end - begin)
    if len(data) != end - begin:
        raise CensusError(f"safetensors の読み取りが途中で終わりました: '{key}'")
    return data


def detect_prefix(header: Header) -> str:
    """参考情報として接頭辞を判定する（``model.diffusion_model.`` か裸名か）。"""
    keys = header.tensors
    if any(k.startswith(_COMFY_PREFIX + _FIRST_BLOCK) for k in keys):
        return _COMFY_PREFIX
    if any(k.startswith(_FIRST_BLOCK) for k in keys):
        return ""
    return "?"  # transformer_blocks.0. が見当たらない(connectorのみのファイル等)


# === 層ごとの印の解決（層ごとにメタデータ優先→comfy_quant） ====================
@dataclass
class LayerMarker:
    layer: str
    source: str  # "metadata" | "comfy_quant"
    raw_format: str | None  # 印の生の "format" 値（parse_quant_marker が拒否しても記録する）
    parsed: dict | None  # parse_quant_marker の正規化結果（成功時のみ）
    error: str | None  # parse_quant_marker が拒否した理由（失敗時のみ）


def resolve_layer_markers(path: Path, header: Header) -> dict[str, LayerMarker]:
    """層ごとに印を解決し、変換ツールの ``parse_quant_marker`` で解釈する。

    計画 §2.1 の規則そのまま: 層ごとに、``__metadata__._quantization_metadata
    .layers`` にあればそれ、無ければ ``<層>.comfy_quant``。
    """
    metadata_layers: dict[str, Any] = {}
    raw_qm = header.metadata.get(_METADATA_QUANT_KEY)
    if raw_qm:
        try:
            qm = json.loads(raw_qm)
        except Exception as exc:  # noqa: BLE001
            raise CensusError(
                f"__metadata__.{_METADATA_QUANT_KEY} が JSON として読めません（{exc}）"
            ) from exc
        if isinstance(qm, dict):
            metadata_layers = qm.get("layers") or {}
            if not isinstance(metadata_layers, dict):
                raise CensusError(f"__metadata__.{_METADATA_QUANT_KEY}.layers がオブジェクトではありません")

    comfy_quant_layers = {
        key[: -len(_QUANT_SUFFIX)] for key in header.tensors if key.endswith(_QUANT_SUFFIX)
    }
    all_layers = set(metadata_layers) | comfy_quant_layers

    out: dict[str, LayerMarker] = {}
    for layer in sorted(all_layers):
        if layer in metadata_layers:
            source = "metadata"
            marker_obj = metadata_layers[layer]
            if not isinstance(marker_obj, dict):
                out[layer] = LayerMarker(layer, source, None, None,
                                          f"metadata の層エントリがオブジェクトではありません（{type(marker_obj).__name__}）")
                continue
            raw_bytes = json.dumps(marker_obj).encode("utf-8")
            raw_format = marker_obj.get("format")
        else:
            source = "comfy_quant"
            raw_bytes = read_tensor_bytes(path, header, layer + _QUANT_SUFFIX)
            raw_format = None
            try:
                raw_format = json.loads(raw_bytes.decode("utf-8", "replace")).get("format")
            except Exception:  # noqa: BLE001
                pass

        try:
            parsed = parse_quant_marker(raw_bytes)
            out[layer] = LayerMarker(layer, source, parsed["format"], parsed, None)
        except ComfyDequantError as exc:
            out[layer] = LayerMarker(layer, source, raw_format, None, str(exc))
    return out


def scheme_name(fmt: str, convrot: bool) -> str:
    """変換ツールの format 名 -> 製品側 SCHEME_TABLE の方式名（計画 §2.2）。"""
    if fmt == "int8_tensorwise":
        return "int8_convrot" if convrot else "int8"
    if fmt == "asym_w4a8_int8":
        return "w4a8"
    return fmt  # 未知（parse_quant_marker が成功している以上、原理上は来ない）


# === 集計 ====================================================================
def _shape_of(header: Header, key: str) -> tuple[int, ...] | None:
    info = header.tensors.get(key)
    return info.shape if info else None


def _dtype_of(header: Header, key: str) -> str | None:
    info = header.tensors.get(key)
    return info.dtype if info else None


def census(path: Path, header: Header, markers: dict[str, LayerMarker]) -> dict:
    prefix = detect_prefix(header)

    scheme_counts: collections.Counter = collections.Counter()
    fail_counts: collections.Counter = collections.Counter()
    weight_dtype_counts: collections.Counter = collections.Counter()
    aux_shape_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    source_counts: collections.Counter = collections.Counter()
    failures: list[dict] = []

    for layer, lm in markers.items():
        source_counts[lm.source] += 1
        w_dtype = _dtype_of(header, layer + ".weight")
        weight_dtype_counts[w_dtype] += 1
        for suffix in ("weight_scale", "weight_s_channel", "weight_s_rel", "weight_codebook"):
            shp = _shape_of(header, layer + "." + suffix)
            if shp is not None:
                aux_shape_counts[suffix][shp] += 1

        if lm.parsed is not None:
            scheme = scheme_name(lm.parsed["format"], lm.parsed["convrot"])
            scheme_counts[(scheme, lm.parsed["format"], lm.parsed["convrot"])] += 1
        else:
            bucket = lm.raw_format or "(印なし/解釈不能)"
            fail_counts[bucket] += 1
            failures.append({"layer": layer, "source": lm.source, "raw_format": lm.raw_format,
                             "error": lm.error, "weight_dtype": w_dtype})

    return {
        "path": str(path),
        "file_size": header.file_size,
        "prefix": prefix,
        "metadata_keys": sorted(header.metadata.keys()),
        "n_tensors": len(header.tensors),
        "n_layers_with_marker": len(markers),
        "source_counts": dict(source_counts),
        "scheme_counts": [
            {"scheme": scheme, "format": fmt, "convrot": convrot, "count": n}
            for (scheme, fmt, convrot), n in sorted(scheme_counts.items())
        ],
        "fail_counts": dict(sorted(fail_counts.items(), key=lambda kv: -kv[1])),
        "weight_dtype_counts": {str(k): v for k, v in weight_dtype_counts.items()},
        "aux_shape_counts": {
            suffix: {str(shape): n for shape, n in counter.items()}
            for suffix, counter in aux_shape_counts.items()
        },
        "failures_sample": failures[:20],
        "n_failures": len(failures),
        "converter_supported_formats": list(SUPPORTED_FORMATS),
    }


# === --product スタブ（製品側 sft_quant_format.layer_schemes、まだ存在しない）==
def product_layer_schemes(path: Path) -> dict[str, str] | None:
    """製品側の判定（§3-168 で実装される ``sft_quant_format.layer_schemes``）。

    まだ実装されていないので import は失敗する想定。実装後は
    ``layer_schemes(path, header, prefix) -> dict[str, str]``（接頭辞付き・
    connector 含む、計画 §3 C-1b の公開API）をそのまま呼べるようにするための
    差し込み口。
    """
    try:
        import sft_quant_format  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        print(f"[product] sft_quant_format の import に失敗（未実装・想定内）: {exc}")
        return None
    header = read_header(path)
    prefix = detect_prefix(header)
    return sft_quant_format.layer_schemes(path, header, prefix)  # type: ignore[attr-defined]


def compare_with_product(path: Path, census_result: dict, markers_by_layer: dict[str, LayerMarker]) -> None:
    product = product_layer_schemes(path)
    if product is None:
        print("[product] 比較スキップ（製品側 layer_schemes が使えない）")
        return
    mismatches = []
    prefix = census_result["prefix"] if census_result["prefix"] != "?" else ""
    for layer, lm in markers_by_layer.items():
        if lm.parsed is None:
            continue
        mine = scheme_name(lm.parsed["format"], lm.parsed["convrot"])
        theirs = product.get(prefix + layer)
        if theirs != mine:
            mismatches.append({"layer": layer, "mine": mine, "product": theirs})
    print(f"[product] 突き合わせ対象 {len(markers_by_layer)} 層のうち不一致 {len(mismatches)} 件")
    for m in mismatches[:20]:
        print(f"    {m['layer']}: mine={m['mine']} product={m['product']}")


# === 表示 ====================================================================
def _print_table(title: str, rows: list[tuple], headers: tuple) -> None:
    print(f"\n== {title} ==")
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    print("  " + "  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


def print_report(result: dict) -> None:
    print(f"path            = {result['path']}")
    print(f"file_size       = {result['file_size']:,} bytes")
    print(f"prefix          = {result['prefix']!r}")
    print(f"metadata keys   = {result['metadata_keys']}")
    print(f"n_tensors       = {result['n_tensors']}")
    print(f"layers w/marker = {result['n_layers_with_marker']}  (source内訳: {result['source_counts']})")

    _print_table(
        "方式別の本数（converter parse_quant_marker が受理した層）",
        [(r["scheme"], r["format"], r["convrot"], r["count"]) for r in result["scheme_counts"]],
        ("scheme", "format", "convrot", "count"),
    )
    if result["fail_counts"]:
        _print_table(
            "converter が受理しなかった層（raw format 別。fp8 の float8_e4m3fn 等は想定内）",
            [(fmt, n) for fmt, n in result["fail_counts"].items()],
            ("raw_format", "count"),
        )
    _print_table(
        "重み(.weight)の dtype 分布（マーカー付き層のみ）",
        [(k, v) for k, v in result["weight_dtype_counts"].items()],
        ("dtype", "count"),
    )
    for suffix, dist in result["aux_shape_counts"].items():
        _print_table(
            f"{suffix} の shape 分布",
            [(shape, n) for shape, n in dist.items()],
            ("shape", "count"),
        )
    if result["n_failures"]:
        print(f"\n(先頭最大20件を failures_sample に記録・全{result['n_failures']}件)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="safetensors ファイルのパス")
    ap.add_argument("--json", dest="json_out", type=Path, default=None, help="結果をJSONでも保存する")
    ap.add_argument("--product", action="store_true",
                    help="製品側 sft_quant_format.layer_schemes との突き合わせを試みる（まだ未実装ならスタブとして1行出すだけ）")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"ERROR: ファイルがありません: {args.path}", file=sys.stderr)
        return 1

    header = read_header(args.path)
    markers = resolve_layer_markers(args.path, header)
    result = census(args.path, header, markers)
    print_report(result)

    if args.product:
        compare_with_product(args.path, result, markers)

    if args.json_out:
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
