#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""§3-168 実機検証 G3〜G8 の連鎖台本。計画 `floofy-enchanting-sundae.md` §4。

`runner.py` の副命令を正しい順序（load → single → ... → restore）で
サブプロセスとして呼び出すだけの薄いオーケストレーター。MCP には直接触れない
（1コマンド1セッションの規律は runner.py 側に委ねる）。

対象の変換器5本（登録名は GET /models の慣習＝拡張子を除いた語。実行時に
`models` 副命令で存在を確認してから進む）:
  (a) LTX23 ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot（Kijai）
  (b) LTX23 ltx-2.3-22b-distilled-1.1_int8mixedtensorwise（silveroxides）
  (c) LTX23 ltx-2.3-22b-distilled-1.1_w4a8（JoaoZaokk）
  (d) LTX25 ltx-2.5-22b-distilled-transformer-comfy-int8-convrot
  (e) LTX25 redgraftLTX25Fast2K_ltx25RedgraftNSFW

各本の手順（G3+G4+G5をまとめて1本ずつ通す。必ず load→...→restoreの順）:
  models確認 → load → single → single+Pixar_Toon → [(a)のみ: 先読みオフsingle]
  → iclora(deblur) → chain(2クリップ) → keep2 → 基準GGUFをload
  → single(基準GGUF) → psnr-ssim(int8 single対GGUF) → psnr-ssim(Pixar対single)
  → restore(元へ)

G6（fp8回帰）: 2.3fp8→single(LoRAなし/あり)、2.5fp8→singleを、C-0の基準
（`baseline\*.mp4`）と同条件で再生成し、映像/音声ストリームMD5を比較する。

G7（偽ヘッダ422）: `runner.py` の `BAD_HEADER_CASES`（6ケース）を順に
`bad-header --cleanup` まで実行する（LTX23のWeightsに置く。計画§2.1の
「paramsネストのconvrot（受理側）」はここに含まれない -- README参照）。

G8（資源）: `report` に生成秒・peak_vram・コミット最大が載る。追加で
ConvRot((a)(d))とfp8同条件(g6_*)の生成秒の比を出す。

## 判定基準（2026-09-27改訂: mp4コンテナのメタデータはコンテンツ同一性の
証拠にならない）

(a)のkeep2で、r1・r2はPSNR∞・SSIM1.0・映像/音声ストリームMD5も完全一致
だったのに、ファイル全体のSHA-256は不一致だった。原因は mp4 コンテナの
メタデータ（§3-164で埋め込む job_id等）が2本で異なるため。**keep2の合否は
「映像/音声ストリームMD5一致（同梱ffmpegの `-map 0:v:0 -c copy -f md5 -` と
`-map 0:a:0 ...`）かつPSNR∞」に変更した**（ファイル全体SHA-256は参考値として
記録するだけ）。**G6の比較も同じ基準**（新しい生成 対 `baseline\*.mp4` の
ストリームMD5）に変更した。

使い方:
    python gates.py --dry-run                  # 全コマンドを表示するだけ（実行しない）
    python gates.py                             # G3〜G8を通しで実行
    python gates.py --only a,b                  # 変換器(a)(b)だけ
    python gates.py --stage g6                  # G6だけ
    python gates.py --skip g7                   # G7を飛ばす
    python gates.py --resume                    # 完了済みの手順を飛ばして続きから
    python gates.py --baseline-md5              # baseline\stream_md5.json を(再)計算するだけ

**失敗したらそこで止まる**（exit code != 0 の副命令があれば、可能な範囲で
restoreだけ試みたうえで台本全体を打ち切る。次の変換器へは進まない）。

**`--resume`**: `results/<tag>.json` が既に存在し `status=="completed"` の
single/iclora/chain 手順は再実行しない。`keep2` は `<tag>_r1.json`/
`<tag>_r2.json` の両方が揃っていれば `keep2 --reuse` で（生成をやり直さず）
新基準の再判定だけを行う。`models確認`・`load`・`restore`・`psnr-ssim` は
安価なので `--resume` でも毎回実行する。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner.py"
RESULTS = HERE / "results"
BASELINE_DIR = HERE / "baseline"
BASELINE_MD5_JSON = BASELINE_DIR / "stream_md5.json"
ROOT = Path(r"S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni")
PYEXE = r"S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\.venv\Scripts\python.exe"

# IC-LoRA deblur の参照動画。前回の実機確認（Docs/VERIFICATION_LOG.md §117.8・
# stage3/results/fp8_iclora.json）と同じ動画を使い回す（ROOT相対、現存確認済み
# 2026-09-27）。IC-LoRA名 "deblur" も同じ節で確認済み（強さ1.0・512x384）。
DEBLUR_REFERENCE = "outputs/99951e90-6b90-4dc7-af40-a78f9d536177/output.mp4"
DEBLUR_IC_LORA = "deblur"
DEBLUR_STRENGTH = 1.0
DEBLUR_WIDTH, DEBLUR_HEIGHT = 512, 384

#: 変換器5本の定義。順序はコーディネーター指定のとおり (a)-(e)。
CONVERTERS: list[dict[str, Any]] = [
    {
        "key": "a", "base": "LTX23",
        "transformer": "ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot",
        "label": "Kijai int8_convrot",
        "ref_transformer": "default",
        "restore_transformer": "sulphur_distil_fp8mixed",  # LTX23の「元」＝fp8
        "fp8_compare_tag": "g6_23_fp8",  # G8のConvRotコスト比の対象
        "prefetch_off_single": True,  # G3「先読みオフの単体」を(a)だけ足す
    },
    {
        "key": "b", "base": "LTX23",
        "transformer": "ltx-2.3-22b-distilled-1.1_int8mixedtensorwise",
        "label": "silveroxides int8mixedtensorwise",
        "ref_transformer": "default",
        "restore_transformer": "sulphur_distil_fp8mixed",
    },
    {
        "key": "c", "base": "LTX23",
        "transformer": "ltx-2.3-22b-distilled-1.1_w4a8",
        "label": "JoaoZaokk w4a8",
        "ref_transformer": "default",
        "restore_transformer": "sulphur_distil_fp8mixed",
    },
    {
        "key": "d", "base": "LTX25",
        "transformer": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot",
        "label": "LTX25 comfy int8_convrot",
        "ref_transformer": "default",  # 2.5公式GGUF
        "restore_transformer": "default",  # LTX25の「元」＝default
        "fp8_compare_tag": "g6_25_fp8",
    },
    {
        "key": "e", "base": "LTX25",
        "transformer": "redgraftLTX25Fast2K_ltx25RedgraftNSFW",
        "label": "REDGraft (CivitAI 3250230)",
        "ref_transformer": "redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K",
        "restore_transformer": "default",
    },
]

# G6（fp8回帰）の定義。
G6_STEPS = [
    {"base": "LTX23", "transformer": "sulphur_distil_fp8mixed", "tag": "g6_23_fp8",
     "load_tag": "g6_load_23", "lora": None},
    {"base": "LTX23", "transformer": "sulphur_distil_fp8mixed", "tag": "g6_23_fp8_pixar",
     "load_tag": "g6_load_23_pixar", "lora": "Pixar_Toon:0.8"},
    {"base": "LTX25", "transformer": "ltx25_uncensored_v1.1-fp8_scaled", "tag": "g6_25_fp8",
     "load_tag": "g6_load_25", "lora": None},
]

#: G6タグ -> C-0基準mp4のファイル名（scratchpad\int8\baseline\ 配下。
#: `--baseline-md5` で先にストリームMD5を計算しておく）。
G6_BASELINE_MP4 = {
    "g6_23_fp8": "c0_23_fp8.mp4",
    "g6_23_fp8_pixar": "c0_23_fp8_pixar.mp4",
    "g6_25_fp8": "c0_25_fp8.mp4",
}

#: G7 の対象ベースモデル（コーディネーター指定: どのケースも LTX23 の
#: Weights に置く）と、その配下のディレクトリ。
G7_BASE = "LTX23"
G7_WEIGHTS_DIR = "models/LTX23/Weights"


class GateFailure(Exception):
    """途中の副命令が exit code != 0 を返した。"""


def _fmt_cmd(cmd: list[str]) -> str:
    return " ".join(('"%s"' % c) if (" " in c or "\\" in c) else c for c in cmd)


def run_step(label: str, cmd: list[str], dry_run: bool) -> None:
    print("\n# %s" % label)
    print("+ %s" % _fmt_cmd(cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise GateFailure("%s が失敗しました（exit=%d）: %s" % (label, proc.returncode, _fmt_cmd(cmd)))


def _import_runner():
    """runner.py をモジュールとして import する（BAD_HEADER_CASES や
    _stream_md5 のような1箇所の定義／実装を使い回すため）。"""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import runner as _runner  # noqa: E402
    return _runner


# === --resume: 既存の results/*.json を見て完了済みかどうかを判定 ============
def _load_json(tag: str) -> dict | None:
    p = RESULTS / ("%s.json" % tag)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _is_job_done(tag: str) -> bool:
    """single/iclora/chain 用: results/<tag>.json が status=="completed"。"""
    rec = _load_json(tag)
    return bool(rec and rec.get("status") == "completed")


def _keep2_children_done(tag: str) -> bool:
    """keep2 用: <tag>_r1.json / <tag>_r2.json の両方が status=="completed"。"""
    r1, r2 = _load_json("%s_r1" % tag), _load_json("%s_r2" % tag)
    return bool(r1 and r1.get("status") == "completed" and r2 and r2.get("status") == "completed")


def _maybe_skip_job(label: str, tag: str, resume: bool) -> bool:
    """resume時、results/<tag>.jsonが既にcompletedならその旨を表示してTrueを
    返す（呼び出し側はTrueならそのステップの実行そのものを省く）。"""
    if resume and _is_job_done(tag):
        print("\n# %s" % label)
        print("[gates] --resume: results/%s.json は既に completed なのでスキップします" % tag)
        return True
    return False


# === runner.py 副命令のコマンド組み立て ======================================
def cmd_models(base: str, transformer: str) -> list[str]:
    return [PYEXE, str(RUNNER), "models", "--base-model", base, "--transformer", transformer]


def cmd_load(base: str, transformer: str, tag: str) -> list[str]:
    return [PYEXE, str(RUNNER), "load", "--base-model", base, "--transformer", transformer, "--tag", tag]


def cmd_restore(base: str, transformer: str) -> list[str]:
    return [PYEXE, str(RUNNER), "restore", "--base-model", base, "--transformer", transformer]


def cmd_single(base: str, transformer: str, tag: str, *, lora: str | None = None,
               prefetch: str | None = None, width: int = 512, height: int = 320,
               frames: int = 49, seed: int = 12345) -> list[str]:
    c = [PYEXE, str(RUNNER), "single", "--base-model", base, "--transformer", transformer, "--tag", tag,
         "--width", str(width), "--height", str(height), "--frames", str(frames), "--seed", str(seed)]
    if lora:
        c += ["--lora", lora]
    if prefetch:
        c += ["--prefetch", prefetch]
    return c


def cmd_iclora(base: str, transformer: str, tag: str) -> list[str]:
    return [PYEXE, str(RUNNER), "iclora", "--base-model", base, "--transformer", transformer, "--tag", tag,
            "--reference", DEBLUR_REFERENCE, "--ic-lora", DEBLUR_IC_LORA,
            "--strength", str(DEBLUR_STRENGTH), "--width", str(DEBLUR_WIDTH), "--height", str(DEBLUR_HEIGHT),
            "--seed", "12345"]


def cmd_chain(base: str, transformer: str, tag: str) -> list[str]:
    return [PYEXE, str(RUNNER), "chain", "--base-model", base, "--transformer", transformer, "--tag", tag,
            "--width", "512", "--height", "320", "--clip-frames", "49", "49", "--seed", "12345"]


def cmd_keep2(base: str, transformer: str, tag: str, *, reuse: bool = False) -> list[str]:
    c = [PYEXE, str(RUNNER), "keep2", "--base-model", base, "--transformer", transformer, "--tag", tag,
         "--width", "512", "--height", "320", "--frames", "49", "--seed", "12345"]
    if reuse:
        c += ["--reuse"]
    return c


def cmd_psnr_ssim(a: str, b: str, tag: str) -> list[str]:
    return [PYEXE, str(RUNNER), "psnr-ssim", "--a", a, "--b", b, "--tag", tag]


def cmd_bad_header(base: str, weights_dir: str, case: str, *, cleanup: bool = False,
                   tag: str | None = None) -> list[str]:
    c = [PYEXE, str(RUNNER), "bad-header", "--base-model", base, "--weights-dir", weights_dir, "--case", case]
    if tag:
        c += ["--tag", tag]
    if cleanup:
        c += ["--cleanup"]
    return c


def cmd_report() -> list[str]:
    return [PYEXE, str(RUNNER), "report"]


# === G3+G4+G5（変換器1本ぶん） ================================================
def run_converter_gate(conv: dict, dry_run: bool, resume: bool) -> None:
    base, key, transformer = conv["base"], conv["key"], conv["transformer"]
    ref, restore_to, label = conv["ref_transformer"], conv["restore_transformer"], conv["label"]

    print("\n" + "=" * 70)
    print("### (%s) %s — base=%s transformer=%s" % (key, label, base, transformer))
    print("=" * 70)

    try:
        # models確認・load は安価なので --resume でも毎回実行する。
        run_step("(%s) models確認（候補）" % key, cmd_models(base, transformer), dry_run)
        run_step("(%s) load 候補" % key, cmd_load(base, transformer, "g3_%s_load" % key), dry_run)

        tag = "g3_%s_single" % key
        if not _maybe_skip_job("(%s) single 512x320x49f seed12345" % key, tag, resume):
            run_step("(%s) single 512x320x49f seed12345" % key, cmd_single(base, transformer, tag), dry_run)

        tag = "g3_%s_pixar" % key
        if not _maybe_skip_job("(%s) single + Pixar_Toon:0.8" % key, tag, resume):
            run_step("(%s) single + Pixar_Toon:0.8" % key,
                     cmd_single(base, transformer, tag, lora="Pixar_Toon:0.8"), dry_run)

        if conv.get("prefetch_off_single"):
            tag = "g3_%s_single_prefetch_off" % key
            if not _maybe_skip_job("(%s) single 先読みオフ（同期経路）" % key, tag, resume):
                run_step("(%s) single 先読みオフ（同期経路）" % key,
                         cmd_single(base, transformer, tag, prefetch="off"), dry_run)

        tag = "g3_%s_iclora" % key
        if not _maybe_skip_job("(%s) iclora deblur（参照動画 %s）" % (key, DEBLUR_REFERENCE), tag, resume):
            run_step("(%s) iclora deblur（参照動画 %s）" % (key, DEBLUR_REFERENCE),
                     cmd_iclora(base, transformer, tag), dry_run)

        tag = "g3_%s_chain" % key
        if not _maybe_skip_job("(%s) chain 2クリップ" % key, tag, resume):
            run_step("(%s) chain 2クリップ" % key, cmd_chain(base, transformer, tag), dry_run)

        # keep2: --resume かつ r1/r2 が両方completedなら --reuse（生成せず
        # 新基準の再判定だけ）。それ以外は通常どおり2本生成する。
        keep2_tag = "g4_%s" % key
        reuse_keep2 = resume and _keep2_children_done(keep2_tag)
        note = "（--resume: 既存のr1/r2から新基準で再判定のみ・生成なし）" if reuse_keep2 else ""
        run_step("(%s) keep2（G4）%s" % (key, note),
                 cmd_keep2(base, transformer, keep2_tag, reuse=reuse_keep2), dry_run)

        run_step("(%s) models確認（基準GGUF）" % key, cmd_models(base, ref), dry_run)
        run_step("(%s) load 基準GGUF" % key, cmd_load(base, ref, "g5_%s_load_ref" % key), dry_run)

        tag = "g5_%s_ref" % key
        if not _maybe_skip_job("(%s) single 基準GGUF（同条件・G5）" % key, tag, resume):
            run_step("(%s) single 基準GGUF（同条件・G5）" % key, cmd_single(base, ref, tag), dry_run)

        # psnr-ssimはローカルffmpegだけの軽い処理なので毎回実行する。
        run_step("(%s) psnr-ssim: int8 single 対 基準GGUF" % key,
                 cmd_psnr_ssim("g3_%s_single" % key, "g5_%s_ref" % key, "g5_%s_vs_ref" % key), dry_run)
        run_step("(%s) psnr-ssim: Pixar 対 single（LoRA有無の差）" % key,
                 cmd_psnr_ssim("g3_%s_pixar" % key, "g3_%s_single" % key,
                              "g3_%s_pixar_vs_single" % key), dry_run)
    finally:
        # 途中で失敗しても、可能な範囲で選択だけは元へ戻す（次の変換器の実験が
        # 汚染されないようにする安全策。失敗自体はそのまま上へ伝える）。
        run_step("(%s) restore 元へ（%s）" % (key, restore_to), cmd_restore(base, restore_to), dry_run)


# === G6（fp8回帰。baseline\*.mp4 とのストリームMD5比較） =======================
def _job_video_path(tag: str) -> Path | None:
    rec = _load_json(tag)
    if not rec:
        return None
    vp = rec.get("video_path")
    return Path(vp) if vp else None


def precompute_baseline_md5() -> dict:
    """C-0基準mp4（baseline\\*.mp4）の映像/音声ストリームMD5を計算して
    baseline\\stream_md5.json に保存する（GPU不使用・ffmpegのみ）。"""
    _runner = _import_runner()
    out: dict[str, dict] = {}
    for tag, fname in G6_BASELINE_MP4.items():
        path = BASELINE_DIR / fname
        if not path.exists():
            print("[gates] baseline mp4 が見つかりません: %s" % path)
            continue
        v = _runner._stream_md5(path, "0:v:0")
        a = _runner._stream_md5(path, "0:a:0")
        out[tag] = {"file": fname, "video_md5": v, "audio_md5": a}
        print("[gates] baseline %s: video=%s audio=%s" % (fname, v, a))
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_MD5_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[gates] wrote %s" % BASELINE_MD5_JSON)
    return out


def _compare_g6(tag_new: str, dry_run: bool) -> None:
    print("\n# G6比較: %s 対 基準mp4（映像/音声ストリームMD5）" % tag_new)
    if dry_run:
        print("+ (python内) stream_md5(results/%s.json の video_path) 対 "
              "baseline/stream_md5.json[%s]" % (tag_new, tag_new))
        return
    new_p = _job_video_path(tag_new)
    if not new_p or not new_p.exists():
        print("[gates] G6比較スキップ（新しい生成の記録が見つからない）: %s" % tag_new)
        return
    if not BASELINE_MD5_JSON.exists():
        print("[gates] baseline/stream_md5.json がありません。先に "
              "`python gates.py --baseline-md5` を実行してください。")
        return
    baselines = json.loads(BASELINE_MD5_JSON.read_text(encoding="utf-8"))
    base = baselines.get(tag_new)
    if not base:
        print("[gates] baseline/stream_md5.json にタグ %s の記録がありません" % tag_new)
        return
    _runner = _import_runner()
    v_new = _runner._stream_md5(new_p, "0:v:0")
    a_new = _runner._stream_md5(new_p, "0:a:0")
    video_match = v_new is not None and v_new == base.get("video_md5")
    audio_match = a_new == base.get("audio_md5")
    match = video_match and audio_match
    print("[gates] video_md5: new=%s base=%s match=%s" % (v_new, base.get("video_md5"), video_match))
    print("[gates] audio_md5: new=%s base=%s match=%s" % (a_new, base.get("audio_md5"), audio_match))
    print("[gates] G6 判定(%s): %s" % (tag_new, "OK" if match else "MISMATCH"))
    out = RESULTS / ("g6_compare_%s.json" % tag_new)
    out.write_text(json.dumps({
        "tag_new": tag_new, "baseline_file": base.get("file"), "video_new": str(new_p),
        "video_md5_new": v_new, "audio_md5_new": a_new,
        "video_md5_baseline": base.get("video_md5"), "audio_md5_baseline": base.get("audio_md5"),
        "video_match": video_match, "audio_match": audio_match, "match": match,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[gates] wrote %s" % out)


def run_g6(dry_run: bool, resume: bool) -> None:
    print("\n" + "=" * 70)
    print("### G6 — fp8 回帰（C-0 の基準mp4と完全に同じ設定で再生成・"
          "ストリームMD5一致を期待）")
    print("=" * 70)
    for step in G6_STEPS:
        # load は安価なので --resume でも毎回実行する。
        run_step("G6 load %s/%s" % (step["base"], step["transformer"]),
                 cmd_load(step["base"], step["transformer"], step["load_tag"]), dry_run)
        tag = step["tag"]
        if not _maybe_skip_job("G6 single %s" % tag, tag, resume):
            run_step("G6 single %s" % tag,
                     cmd_single(step["base"], step["transformer"], tag, lora=step["lora"]), dry_run)
        # 比較はローカルffmpegだけの軽い処理なので毎回実行する
        # （判定基準の変更を確実に反映させるため）。
        _compare_g6(tag, dry_run)
    # LTX25 は g6_25_fp8 の後 uncensored のままなので default へ戻す
    # （LTX23 は sulphur_distil_fp8mixed のままで「元」と一致するので不要）。
    run_step("G6 restore LTX25 元へ（default）", cmd_restore("LTX25", "default"), dry_run)


# === G7（偽ヘッダ422） =======================================================
def run_g7(dry_run: bool) -> None:
    print("\n" + "=" * 70)
    print("### G7 — 偽ヘッダ422（BAD_HEADER_CASES 全ケース、LTX23 Weights 配下）")
    print("=" * 70)
    # runner.py から直接ケース名一覧を読む（1箇所の定義を使い回す。二重管理しない）。
    _runner = _import_runner()
    for case in _runner.BAD_HEADER_CASES:
        run_step("G7 bad-header %s" % case,
                 cmd_bad_header(G7_BASE, G7_WEIGHTS_DIR, case, tag="g7_%s" % case), dry_run)
        run_step("G7 bad-header %s --cleanup" % case,
                 cmd_bad_header(G7_BASE, G7_WEIGHTS_DIR, case, cleanup=True), dry_run)
    print("\n[gates] 計画§2.1「paramsネストのconvrot（受理側）」はここに含まれない"
          "（拒否ではなく受理を確かめるケースのため。README『G7』節の代替手順を参照）。")


# === G8（資源。report＋ConvRotコスト比） ======================================
def _gen_seconds(tag: str) -> float | None:
    rec = _load_json(tag)
    return (rec.get("meta") or {}).get("generation_time_seconds") if rec else None


def run_g8(dry_run: bool) -> None:
    print("\n" + "=" * 70)
    print("### G8 — 資源（生成秒・peak_vram・コミット最大は report に集計。"
          "ここでは ConvRot と fp8 同条件の生成秒の比を追加で出す）")
    print("=" * 70)
    run_step("G8 report", cmd_report(), dry_run)

    print("\n# G8 ConvRotコスト比（(a)(d) の g3_<key>_single 対 g6_* fp8 同条件）")
    if dry_run:
        print("+ (python内) generation_time_seconds(g3_a_single) / generation_time_seconds(g6_23_fp8) 等")
        return
    rows = []
    for conv in CONVERTERS:
        fp8_tag = conv.get("fp8_compare_tag")
        if not fp8_tag:
            continue
        convrot_tag = "g3_%s_single" % conv["key"]
        t_conv, t_fp8 = _gen_seconds(convrot_tag), _gen_seconds(fp8_tag)
        ratio = (t_conv / t_fp8) if (t_conv and t_fp8) else None
        rows.append({"label": conv["label"], "convrot_tag": convrot_tag, "convrot_seconds": t_conv,
                     "fp8_tag": fp8_tag, "fp8_seconds": t_fp8, "ratio": ratio})
        print("[gates] (%s) %s: convrot=%ss fp8=%ss 比=%s"
              % (conv["key"], conv["label"], t_conv, t_fp8, ratio))
    out = RESULTS / "g8_convrot_cost_ratio.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[gates] wrote %s" % out)
    print("\n[gates] 計画の元々のG8にある「connectorの復元時間とジョブごとの再発の"
          "有無（2.5はEP常駐オン・オフ）」はこの台本では自動化していない"
          "（現状のmetadata.jsonに該当フィールドが無く追加の計測実装が要るため。"
          "README『未解決点』参照）。")


# === main ====================================================================
def main() -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="全コマンドを表示するだけで実行しない")
    ap.add_argument("--only", default=None, help="変換器を絞る（例: a,b。既定は全5本）")
    ap.add_argument("--stage", choices=["g3to5", "g6", "g7", "g8"], default=None,
                    help="指定した段だけ実行する（既定は g3to5→g6→g7→g8 を通しで）")
    ap.add_argument("--skip", action="append", choices=["g3to5", "g6", "g7", "g8"], default=[],
                    help="指定した段を飛ばす（複数可）")
    ap.add_argument("--resume", action="store_true",
                    help="results/<tag>.json が既にcompletedのsingle/iclora/chainは再実行しない。"
                         "keep2はr1/r2が揃っていれば--reuseで再判定のみ。"
                         "models確認/load/restore/psnr-ssimは安価なので毎回実行する")
    ap.add_argument("--baseline-md5", action="store_true",
                    help="G6比較用に baseline/stream_md5.json を(再)計算して終了する"
                         "（GPU不使用、ローカルの baseline\\*.mp4 だけを読む）")
    args = ap.parse_args()

    if args.baseline_md5:
        precompute_baseline_md5()
        return 0

    stages = [args.stage] if args.stage else ["g3to5", "g6", "g7", "g8"]
    stages = [s for s in stages if s not in args.skip]

    only_keys = set(args.only.split(",")) if args.only else None
    converters = [c for c in CONVERTERS if not only_keys or c["key"] in only_keys]

    try:
        if "g3to5" in stages:
            for conv in converters:
                run_converter_gate(conv, args.dry_run, args.resume)
        if "g6" in stages:
            run_g6(args.dry_run, args.resume)
        if "g7" in stages:
            run_g7(args.dry_run)
        if "g8" in stages:
            run_g8(args.dry_run)
    except GateFailure as exc:
        print("\n[gates] STOP: %s" % exc, file=sys.stderr)
        print("[gates] 次の対象へは進みません。原因を確認してから再実行してください"
              "（`--resume` で完了済みの手順を飛ばして続きから再開できます）。",
              file=sys.stderr)
        return 1

    print("\n[gates] 完了（dry_run=%s、resume=%s、stages=%s）" % (args.dry_run, args.resume, stages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
