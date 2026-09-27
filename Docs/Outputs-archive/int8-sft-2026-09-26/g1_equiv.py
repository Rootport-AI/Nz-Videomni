#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""§3-168 G1(i)(iii) NumPy（変換ツール）と torch（製品側）の突き合わせ。

**このスクリプトは engine 側の venv（torch 入り）で実行すること**:
    S:\...\Nz-Videomni\.venv-engine\Scripts\python.exe g1_equiv.py ...
    S:\...\Nz-Videomni\.venv-engine-ltx25\Scripts\python.exe g1_equiv.py ...
（どちらも torch==2.9.1+cu128・CUDA 使用可能なことを確認済み、2026-09-27）。
NumPy 側の変換ツール（`comfy_dequant.py`）はどちらの venv にも入っている
numpy で動く。torch が全く無い環境（アプリ用 `.venv` 等）で実行した場合は、
torch 側の比較を丸ごとスキップして NumPy 側だけ実行する（クラッシュしない）。

製品側 `engine.sft_quant.dequant`（`dequantize` / `normalize_aux`）と
`sft_quant_format` は §3-168 の実装用 worktree にある。既定は
``--worktree S:\OriginalApps\12_Nz-LTX23-AviUtl2\_wt_3168`` を sys.path の
先頭に足してから import する（環境変数 PYTHONPATH を使わなくても動く。
`--worktree ""` で無効化すればアプリ本体側の実装が入ったときにも流用できる）。
補助テンソルの正規化は必ず製品の ``normalize_aux(scheme, leaf, raw, o, i)``
を通す（このスクリプト側で (o,1) 等へ手で整形しない -- 正規化ロジックその
ものを検証対象に含めるため）。

比較対象:
  (i)   乱数テンソル（形 [16384,4096]・[4096,16384]・[32,4096]、方式
        int8（スカラー倍率・行ごと）・int8+ConvRot（同）・w4a8）を
        NumPy（変換ツール `dequantize_layer`）と
        torch（製品側 `engine.sft_quant.dequant.dequantize`）の両方で復元し
        突き合わせる。CPU と CUDA の両方（``--device``）。w4a8 は
        `SCHEME_TABLE` にまだ無い（C-3 でこれから実装される）ため、比較は
        失敗して当然 -- その旨を記録するだけで異常終了はしない。
  (iii) ``--file PATH --layer NAME`` で実ファイルの層1本の生バイトをヘッダの
        オフセットから seek して読み、同じ比較を行う。

## 判定（2026-09-27 改訂: bf16 ulp の絶対の床つき OR 判定）

当初は「fp32 で max|Δ| <= 1e-5*max|W|」と「bf16 化後1 ulp 以内」を両方要求
していたが、int8_convrot で偽陽性が出た: ConvRot（アダマール回転）は成分の
線形結合なので、+1/-1 の係数がほぼ打ち消し合う成分は真値が 0 近傍
（|W| ~ 1e-7）になる。この領域では bf16 の指数部が非常に小さく隣接値の間隔
が極小なので、NumPy と torch の丸め誤差（加算順の違いによる ~1e-7 の差）
だけで符号が反転し、ulp距離が数万に達する（実測: shape 512x4096 の
int8_convrot で ulp>1 の670要素すべてが |W|<=7.7e-7 かつ |Δ|<=1.7e-6 -- fp32
の絶対の床 6.8e-5 の内側）。詳細は下記「診断結果」参照。

**修正後の判定は要素ごとの OR**: 各要素について
``(bf16 ulp <= 1) or (fp32 |Δ| <= 1e-5 * max|W|)`` が真なら合格。
両方とも偽の要素だけを「真の不一致」として数える（``n_bad_or``）。
ulp>1 の要素数・その中の |W|・|Δ| の分布（診断用）は常に記録・表示する。

NumPy 側の変換ツール（`converter/comfy_dequant.py`）は編集しない・
sys.path に src を足して import するだけ。ヘッダ読み（``--file`` モード用）は
同じフォルダの `g1_census.py` の実装をそのまま使う（mmap・全読み禁止の方針は
共通）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# --- 変換ツール（編集しない・importするだけ） --------------------------------
CONVERTER_SRC = Path(r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-GGUF-Converter-LTX23\src")
if str(CONVERTER_SRC) not in sys.path:
    sys.path.insert(0, str(CONVERTER_SRC))

from converter.comfy_dequant import (  # noqa: E402
    ComfyDequantError,
    dequantize_layer,
    f32_to_bf16_u16,
)

# g1_census.py（同じフォルダ）のヘッダ読みを --file モードで再利用する。
from g1_census import (  # noqa: E402
    Header,
    read_header,
    read_tensor_bytes,
    resolve_layer_markers,
    scheme_name,
)

#: §3-168 実装用 worktree（製品側 engine.sft_quant.dequant / sft_quant_format
#: の在り処。bare import ``import sft_quant_format`` が要求するので、この
#: worktree のルート自体を sys.path に足す）。
WORKTREE_DEFAULT = Path(r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\_wt_3168")

SHAPES: list[tuple[int, int]] = [(16384, 4096), (4096, 16384), (32, 4096)]
#: 内部テストケース名 -> 製品側 scheme 文字列（計画 §2.2 の SCHEMES）。
TEST_SCHEMES = {
    "int8_scalar": "int8",
    "int8_rowwise": "int8",
    "int8_convrot_scalar": "int8_convrot",
    "int8_convrot_rowwise": "int8_convrot",
    "w4a8": "w4a8",  # SCHEME_TABLE にまだ無い(C-3)。失敗して当然、記録するだけ。
}
FP32_REL_TOL = 1e-5
BF16_ULP_TOL = 1


# === torch 側（製品）の import =============================================
def try_import_torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception as exc:  # noqa: BLE001
        print(f"[torch] このインタプリタに torch がありません（engine 側の venv で実行してください）: {exc}")
        return None


def try_import_torch_product(worktree: Path | None):
    """製品側 ``engine.sft_quant.dequant`` の ``dequantize``/``normalize_aux``。

    ``worktree`` を sys.path の先頭へ足してから import する（``dequant.py``
    は bare ``import sft_quant_format`` をするので、worktree のルート自体が
    sys.path に乗っている必要がある）。見つからない場合は (None, None) を返す
    （呼び出し側はこれを見て NumPy 単独モードへフォールバックする）。
    """
    if worktree is not None:
        wt = str(worktree)
        if wt and wt not in sys.path:
            sys.path.insert(0, wt)
    try:
        from engine.sft_quant.dequant import dequantize, normalize_aux
        return dequantize, normalize_aux
    except Exception as exc:  # noqa: BLE001
        print(f"[torch product] engine.sft_quant.dequant の import に失敗: {exc}")
        return None, None


# === ケース生成（乱数テンソル） ==============================================
def _numpy_ready_scale(raw_scale: np.ndarray, o: int) -> np.ndarray:
    """変換ツール(``_per_row_vector``)が受け付ける形（``[o]`` か ``[o,1]``）へ。

    真のスカラー（shape ``()`` か ``(1,)``）は (o,1) へブロードキャストする
    （変換ツールはこの広げ方を自分ではしないため、G1台本側の責務）。
    """
    if raw_scale.shape == ():
        return np.full((o, 1), float(raw_scale), dtype=np.float32)
    if raw_scale.shape == (1,):
        return np.full((o, 1), float(raw_scale.reshape(())), dtype=np.float32)
    return raw_scale  # 既に (o,) か (o,1)


def make_case(
    rng: np.random.Generator, o: int, i: int, case: str
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """内部テストケース名から (marker, numpy_tensors, raw_aux) を作る。

    ``numpy_tensors`` は変換ツール ``dequantize_layer`` がそのまま受け付ける
    形。``raw_aux`` は「実ファイルに格納されているとおりの生の形」
    （スカラーは真に shape ``()``）で、torch 側は必ずこれを製品の
    ``normalize_aux`` に通してから使う（このスクリプトでは (o,1) 等へ手で
    整形しない）。
    """
    if case.startswith("int8"):
        weight = rng.integers(-127, 127, size=(o, i), endpoint=True, dtype=np.int8)
        if "rowwise" in case:
            raw_scale = (rng.random(o).astype(np.float32) * 0.02 + 0.001).reshape(o, 1)
        else:
            s = float(rng.random() * 0.02 + 0.001)
            raw_scale = np.array(s, dtype=np.float32)  # 真のスカラー shape ()
        marker = {"format": "int8_tensorwise", "convrot": "convrot" in case}
        numpy_tensors = {"weight": weight, "weight_scale": _numpy_ready_scale(raw_scale, o)}
        raw_aux = {"weight_scale": raw_scale}
        return marker, numpy_tensors, raw_aux

    if case == "w4a8":
        if i % 16:
            raise ValueError(f"w4a8 の in_features {i} は group_size 16 の倍数ではありません")
        packed = rng.integers(0, 256, size=(o, i // 2), dtype=np.uint16).astype(np.uint8).view(np.int8)
        codebook = (rng.random(16).astype(np.float32) * 2 - 1)
        s_channel = (rng.random(o).astype(np.float32) * 0.02 + 0.001)
        s_rel = rng.integers(0, 256, size=(o, i // 16), dtype=np.uint16).astype(np.uint8)
        # FP8 E4M3FN の NaN パターン(0x7F/0xFF)は非有限値になり
        # dequantize_layer が拒否するので、有限値へ差し替える(~1.0相当)。
        s_rel[(s_rel == 0x7F) | (s_rel == 0xFF)] = 0x38
        marker = {"format": "asym_w4a8_int8"}
        numpy_tensors = {"weight": packed, "weight_codebook": codebook,
                         "weight_s_channel": s_channel, "weight_s_rel": s_rel}
        raw_aux = {"weight_codebook": codebook, "weight_s_channel": s_channel, "weight_s_rel": s_rel}
        return marker, numpy_tensors, raw_aux

    raise ValueError(f"未知のケース: {case}")


# === bf16 の 1 ulp 判定（IEEE準拠の全順序変換） ================================
def _bf16_ordered(bits: np.ndarray) -> np.ndarray:
    """bf16のビットパターン(uint16) -> 単調順序のint(ulp距離を単純な差で測れる形)。

    符号ビットが立っている(負)側は「大きさが大きいほど小さい順序値」になる
    ように反転する（IEEE系フォーマットの標準的な全順序変換。隣接bf16値どうし
    は必ず差1になることを既知ペア(1.0とその次に大きい表現可能値)で確認済み）。
    """
    b = bits.astype(np.int64)
    neg = (b & 0x8000) != 0
    return np.where(neg, -(b & 0x7FFF) - 1, b)


def bf16_ulp_diff(a_fp32: np.ndarray, b_fp32: np.ndarray) -> np.ndarray:
    """要素ごとの bf16 ulp 距離（配列。丸めは変換ツール自身の f32_to_bf16_u16）。"""
    a_bits = f32_to_bf16_u16(np.ascontiguousarray(a_fp32, dtype=np.float32))
    b_bits = f32_to_bf16_u16(np.ascontiguousarray(b_fp32, dtype=np.float32))
    oa, ob = _bf16_ordered(a_bits), _bf16_ordered(b_bits)
    return np.abs(oa - ob)


def compare_arrays(numpy_fp32: np.ndarray, torch_fp32_np: np.ndarray, torch_bf16_fp32_np: np.ndarray) -> dict:
    """OR判定: 要素ごとに (bf16 ulp<=1) or (fp32 |Δ|<=絶対の床) なら合格。

    ``floor`` はテンソル全体の ``max|W|`` から出す従来どおりの絶対の床
    （§3-168計画のとおり ``1e-5 * max|W|``）。0 近傍での符号反転による ulp
    爆発を吸収しつつ、それ以外の本当の数値ずれ（大きい要素の不一致）は
    従来どおり拾う設計。
    """
    diff = np.abs(numpy_fp32.astype(np.float64) - torch_fp32_np.astype(np.float64))
    max_w = float(np.max(np.abs(numpy_fp32))) if numpy_fp32.size else 0.0
    floor = FP32_REL_TOL * max_w
    fp32_max_diff = float(diff.max()) if diff.size else 0.0

    ulp = bf16_ulp_diff(numpy_fp32, torch_bf16_fp32_np)
    ulp_max = int(ulp.max()) if ulp.size else 0

    ulp_gt1 = ulp > BF16_ULP_TOL
    n_ulp_gt1 = int(ulp_gt1.sum())
    bad_mask = ulp_gt1 & (diff > floor)
    n_bad = int(bad_mask.sum())

    diag: dict[str, Any] = {}
    if n_ulp_gt1:
        w_at = np.abs(numpy_fp32)[ulp_gt1]
        d_at = diff[ulp_gt1]
        diag = {
            "count": n_ulp_gt1,
            "max_abs_W": float(w_at.max()), "min_abs_W": float(w_at.min()),
            "max_abs_diff": float(d_at.max()), "min_abs_diff": float(d_at.min()),
        }

    return {
        "fp32_max_diff": fp32_max_diff, "fp32_tol": floor,
        "bf16_ulp_max": ulp_max, "n_elements": int(ulp.size),
        "n_ulp_gt1": n_ulp_gt1, "n_bad_or": n_bad, "ok": n_bad == 0,
        "diag_ulp_gt1": diag,
    }


# === torch 側の実行（重み・aux を製品の normalize_aux 経由で正規化） ===========
def _run_torch(torch_mod, torch_dequantize, torch_normalize_aux, scheme: str,
               weight_np: np.ndarray, raw_aux: dict[str, np.ndarray], o: int, i: int,
               device: str) -> tuple[np.ndarray, np.ndarray]:
    weight_t = torch_mod.from_numpy(np.ascontiguousarray(weight_np)).to(device)
    aux_t: dict[str, Any] = {}
    for name, raw_np in raw_aux.items():
        raw_t = torch_mod.from_numpy(np.ascontiguousarray(raw_np)).to(device)
        aux_t[name] = torch_normalize_aux(scheme, name, raw_t, o, i)
    torch_fp32 = torch_dequantize(scheme, weight_t, aux_t, torch_mod.float32).detach().to("cpu").numpy()
    torch_bf16_fp32 = (torch_dequantize(scheme, weight_t, aux_t, torch_mod.bfloat16)
                       .detach().to("cpu").to(torch_mod.float32).numpy())
    return torch_fp32, torch_bf16_fp32


def run_case(rng: np.random.Generator, shape: tuple[int, int], case: str,
             torch_mod, torch_dequantize, torch_normalize_aux, device: str) -> dict:
    o, i = shape
    marker, numpy_tensors, raw_aux = make_case(rng, o, i, case)
    layer_name = f"synthetic/{o}x{i}/{case}"

    try:
        numpy_fp32 = dequantize_layer(marker, numpy_tensors, in_features=i, layer_name=layer_name)
    except ComfyDequantError as exc:
        return {"shape": shape, "case": case, "device": device, "error": f"numpy側で失敗: {exc}"}

    result: dict[str, Any] = {
        "shape": shape, "case": case, "device": device,
        "numpy_ok": True, "numpy_max_abs": float(np.max(np.abs(numpy_fp32))),
        "numpy_finite": bool(np.isfinite(numpy_fp32).all()),
    }

    if torch_mod is None or torch_dequantize is None:
        result["torch_compared"] = False
        return result

    scheme = TEST_SCHEMES[case]
    try:
        torch_fp32_np, torch_bf16_fp32_np = _run_torch(
            torch_mod, torch_dequantize, torch_normalize_aux, scheme,
            numpy_tensors["weight"], raw_aux, o, i, device)
        cmp = compare_arrays(numpy_fp32, torch_fp32_np, torch_bf16_fp32_np)
        result["torch_compared"] = True
        result.update(cmp)
    except Exception as exc:  # noqa: BLE001 -- 製品側APIが固まるまでは何が起きても記録して続行
        result["torch_compared"] = False
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    return result


def print_result(r: dict) -> None:
    o, i = r["shape"]
    if r.get("error"):
        print(f"  [{o}x{i} {r['case']} @ {r['device']}] ERROR: {r['error']}")
        return
    if not r.get("torch_compared"):
        extra = f" ({r['torch_error']})" if r.get("torch_error") else ""
        print(f"  [{o}x{i} {r['case']} @ {r['device']}] numpy_ok finite={r['numpy_finite']} "
              f"max|W|={r['numpy_max_abs']:.4g}  -- torch比較スキップ{extra}")
        return
    status = "OK" if r["ok"] else "MISMATCH"
    diag = r.get("diag_ulp_gt1") or {}
    diag_txt = ""
    if diag:
        diag_txt = (f"  [ulp>1: {diag['count']}/{r['n_elements']}要素 "
                    f"|W|∈[{diag['min_abs_W']:.2e},{diag['max_abs_W']:.2e}] "
                    f"|Δ|∈[{diag['min_abs_diff']:.2e},{diag['max_abs_diff']:.2e}]]")
    print(f"  [{o}x{i} {r['case']} @ {r['device']}] {status}  "
          f"fp32: |Δ|max={r['fp32_max_diff']:.3e} tol={r['fp32_tol']:.3e}  "
          f"bf16: ulp_max={r['bf16_ulp_max']}  真の不一致(ulp>1かつ床超)={r['n_bad_or']}件{diag_txt}")


# === --file / --layer（実ファイルの層1本） ====================================
def _np_array_from_tensor(path: Path, header: Header, key: str) -> np.ndarray:
    info = header.tensors[key]
    raw = read_tensor_bytes(path, header, key)
    # F8_E4M3/F8_E5M2(weight_s_rel)は変換ツール側も生のU8バイト列として受け取り
    # 自前のFP8デコードテーブルで解釈する（numpyにfp8のネイティブ型は無い）。
    np_dtype = {
        "I8": np.int8, "U8": np.uint8, "F32": np.float32,
        "F8_E4M3": np.uint8, "F8_E5M2": np.uint8,
    }.get(info.dtype)
    if np_dtype is None:
        raise ValueError(f"'{key}' の dtype {info.dtype} はこのスクリプトでは未対応です")
    # .copy(): frombuffer は読み取り専用の view を返す。torch.from_numpy は
    # 書き込み不可の配列に警告を出す（dequantize の内部が in-place 演算を
    # 使うため）ので、コピーして書き込み可能にしておく。
    return np.frombuffer(raw, dtype=np_dtype).reshape(info.shape).copy()


def run_file_layer(path: Path, layer: str, torch_mod, torch_dequantize, torch_normalize_aux,
                   device: str) -> dict:
    header = read_header(path)
    markers = resolve_layer_markers(path, header)
    lm = markers.get(layer)
    if lm is None:
        return {"error": f"'{layer}' に量子化の印(metadata/comfy_quant)が見つかりません"}
    if lm.parsed is None:
        return {"error": f"'{layer}' の印を変換ツールが解釈できません: {lm.error}"}

    fmt, convrot = lm.parsed["format"], lm.parsed["convrot"]
    scheme = scheme_name(fmt, convrot)
    weight_info = header.tensors.get(layer + ".weight")
    if weight_info is None:
        return {"error": f"'{layer}.weight' がヘッダにありません"}
    o = weight_info.shape[0]
    in_features = weight_info.shape[1] * (2 if fmt == "asym_w4a8_int8" else 1)

    weight_np = _np_array_from_tensor(path, header, layer + ".weight")
    numpy_tensors: dict[str, np.ndarray] = {"weight": weight_np}
    raw_aux: dict[str, np.ndarray] = {}
    if fmt == "int8_tensorwise":
        raw_scale = _np_array_from_tensor(path, header, layer + ".weight_scale")
        numpy_tensors["weight_scale"] = _numpy_ready_scale(raw_scale, o)
        raw_aux["weight_scale"] = raw_scale  # 実ファイルに入っている生の形のまま
    else:
        codebook = _np_array_from_tensor(path, header, layer + ".weight_codebook")
        s_channel = _np_array_from_tensor(path, header, layer + ".weight_s_channel")
        s_rel = _np_array_from_tensor(path, header, layer + ".weight_s_rel")
        numpy_tensors.update({"weight_codebook": codebook, "weight_s_channel": s_channel,
                             "weight_s_rel": s_rel})
        raw_aux.update({"weight_codebook": codebook, "weight_s_channel": s_channel, "weight_s_rel": s_rel})

    try:
        numpy_fp32 = dequantize_layer(lm.parsed, numpy_tensors, in_features=in_features, layer_name=layer)
    except ComfyDequantError as exc:
        return {"error": f"numpy側で失敗: {exc}"}

    result: dict[str, Any] = {
        "layer": layer, "scheme": scheme, "shape": tuple(weight_info.shape), "in_features": in_features,
        "numpy_ok": True, "numpy_max_abs": float(np.max(np.abs(numpy_fp32))),
        "numpy_finite": bool(np.isfinite(numpy_fp32).all()),
    }
    if torch_mod is None or torch_dequantize is None:
        result["torch_compared"] = False
        return result
    try:
        torch_fp32_np, torch_bf16_fp32_np = _run_torch(
            torch_mod, torch_dequantize, torch_normalize_aux, scheme,
            weight_np, raw_aux, o, in_features, device)
        cmp = compare_arrays(numpy_fp32, torch_fp32_np, torch_bf16_fp32_np)
        result["torch_compared"] = True
        result.update(cmp)
    except Exception as exc:  # noqa: BLE001
        result["torch_compared"] = False
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    return result


# === main ====================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None,
                    help="省略時は cpu と（使えれば）cuda の両方を実行する")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--file", type=Path, default=None, help="実ファイルの層1本を比較する場合のsafetensorsパス")
    ap.add_argument("--layer", default=None, help="--file と併用。接頭辞込みの層名（例: model.diffusion_model.transformer_blocks.0.attn1.to_k）")
    ap.add_argument("--worktree", default=str(WORKTREE_DEFAULT),
                    help="製品側 engine.sft_quant.dequant / sft_quant_format が置かれた worktree のルート"
                         "（既定: %(default)s）。空文字列で sys.path 操作を無効化")
    args = ap.parse_args()

    torch_mod = try_import_torch()
    worktree = Path(args.worktree) if args.worktree else None
    torch_dequantize, torch_normalize_aux = (
        try_import_torch_product(worktree) if torch_mod is not None else (None, None)
    )
    devices = [args.device] if args.device else (
        ["cpu", "cuda"] if (torch_mod is not None and torch_mod.cuda.is_available()) else ["cpu"]
    )
    if torch_mod is None:
        devices = ["cpu"]  # numpyのみ。deviceラベルは記録用。

    if torch_mod is not None:
        print(f"torch.backends.cuda.matmul.allow_tf32 = {torch_mod.backends.cuda.matmul.allow_tf32}")

    if args.file:
        if not args.layer:
            print("ERROR: --file を指定する場合は --layer も必要です", file=sys.stderr)
            return 2
        print(f"== --file モード: {args.file}  layer={args.layer} ==")
        any_bad = False
        for device in devices:
            r = run_file_layer(args.file, args.layer, torch_mod, torch_dequantize, torch_normalize_aux, device)
            if r.get("error"):
                print(f"  ERROR: {r['error']}")
                any_bad = True
                continue
            print(f"  scheme={r['scheme']} shape={r['shape']} in_features={r['in_features']}")
            print_result({**r, "shape": (r["shape"][0], r["in_features"]), "case": r["scheme"], "device": device})
            if r.get("torch_compared") and not r["ok"]:
                any_bad = True
        return 1 if any_bad else 0

    print(f"torch: {'利用可' if torch_mod is not None else '利用不可'}"
          f"（製品dequantize: {'利用可' if torch_dequantize is not None else '未実装/利用不可'}）")
    print(f"devices = {devices}")

    any_bad = False
    for shape in SHAPES:
        print(f"\n== shape {shape} ==")
        for case in TEST_SCHEMES:
            for device in devices:
                rng = np.random.default_rng(args.seed)  # 形・ケースごとに再現可能な同一シード
                r = run_case(rng, shape, case, torch_mod, torch_dequantize, torch_normalize_aux, device)
                print_result(r)
                if r.get("error"):
                    any_bad = True
                elif r.get("torch_compared") and not r["ok"]:
                    any_bad = True

    if torch_dequantize is None:
        print("\n[まとめ] torch 側(engine.sft_quant.dequant)が使えないため、NumPy 側の健全性"
              "（有限値・形状）だけを確認した。")
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
