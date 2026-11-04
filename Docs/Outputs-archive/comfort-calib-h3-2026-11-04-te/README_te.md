> **このファイルは実機の `outputs/comfort-calib-h3-2026-11-04-te/README_te.md`（git追跡外）のスナップショットである（2026-11-05複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# 再較正 (B)（台帳 §1-275）の作業場所

計画の正本は `C:\Users\phenotype nezu\.claude\plans\smooth-watching-lobster.md` の §6（合否は §6.5、止める条件は §9）。台本の使い方の詳細は複製元から写した `README.md`（11-04 の手順。ここでは本フォルダへ読み替える）。

## 位置づけ

TE の注意の塊を等分に変えた後で、快適上限の境界が動いていないことを確かめる（オーナー裁定 (B)）。境界の両側の点と、同日の基準点だけを測る。RESULTS.md にもこの 1 行を書く。

## 準備（GPU なし。2026-11-04 夜に済ませた）

- 複製元 `outputs/comfort-calib-h3-2026-11-04/` から台本一式（K 対応の calib.py）・`selftest/`（前回の出力 `*.txt` は除いた）・`materials/`（5 点。sha256 は元と一致）・`keyframe_f0.png`・README.md・`plans/` の 12 本（中身は変えていない）。
- 固定パスを本フォルダへ: drive2.sh:9・12、mk.sh:8、rp.sh:2、wait.sh:3、cool.sh:3、commit_table.py:10、digest.py:5。
- `calib.py preflight` はサーバーなしで動く。12 本とも rc 0（`preflight_<label>.rc`・`preflight_<label>_output.txt`・`preflight.json`）。`calib.py run --dry-run` の 12 本（`dryrun/`）を `selftest/dryrun_validate.py` に通して失敗 0（`selftest/dryrun_validate_te_output.txt`）。

## 段の開始（S5 の切替の後）

オーナーが AviUtl2 と操作パネルを今夜と同じ形で開き、サーバー（run.bat）を起動する。監督が `calib.py server start --pid <PID>`・`calib.py luid`・`commit_monitor.ps1 -Out commit_log.csv` を起動する。

## 点の順（重みの切替 1 回）

各点は `plans/p_<label>.json` を目で確認 → `bash cool.sh` → `bash rp.sh <label> > rp_<label>.txt 2>&1`（裏）→ `bash wait.sh <label>` → `progress.md`。

1. FL2VA（この弾の rp.sh は環境変数 `TRANSFORMER=default WARM_PLAN=plans/p_warm_fl.json` を付けて呼ぶ。OOM の後の自動復帰で FL2VA を読み直すため）:
   `calib.py load --base-model MiniMaxH3 --transformer default` → `calib.py idle-calib --minutes 3` → warm_fl → b1080 → s1080_345 → s1080_362 → b1440 → s1440_192 → s1440_209
2. Ref2VA（環境変数なし。既定が Ref2VA と p_warm）:
   `calib.py load --base-model MiniMaxH3 --transformer minimax_h3_ref2va_pruned_int8_convrot` → `calib.py idle-calib --minutes 3` → warm → b720r → b1080r → v720_379_15s → i1080_243_p9

## 合否（計画 §6.5）

- 判定が 11-04 と同じ: s1080_345・s1440_192・v720_379_15s・i1080_243_p9 は快適、s1080_362・s1440_209 はノイズ除去の段で OOM。
- 全点の cap が 13,999〜14,033MiB の帯の内。
- i1080_243_p9 の encode 窓の共有メモリ差が +300MB 未満。TE 段の予約が 16,074MiB より低い。
- `used_check_h3` が全点で真。text 行の差 0。

## 止める条件（計画 §9）

判定が 11-04 と違う（c が OOM、または U が完走）、cap が帯の外、i1080 の encode 窓の差が +300MB 以上、主メモリのコミット 95%（drive2.sh が止める）、停止線の時刻。止めるのは点の境目だけ。止めたらオーナーへ戻す（全腕へ広げない）。

## 復元

`calib.py restore-state`（`state.json` と `original_state.json` が全一致）→ `POST /api/v1/pipeline/unload` → `/status` が unloaded・キュー空 → commit_monitor を止める → 自分の台本・待ちループの残りがない。
