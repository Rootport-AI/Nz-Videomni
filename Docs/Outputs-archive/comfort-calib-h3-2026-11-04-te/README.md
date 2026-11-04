# 台帳 1-275 の再較正 (B) の記録の複写（2026-11-05）

TE〔テキストエンコーダー〕の注意の塊の計算順を変えた修正（第 2 版。最も高い塊を最初に計算する）の後に、快適上限の境界の両側だけを測り直した記録のうち、文字の記録だけをここへ写しました。**結論と数値の正本は記録簿 [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §197** です（全点の表は §197.7）。快適上限表 [`../../COMFORT_LIMIT_TABLE.md`](../../COMFORT_LIMIT_TABLE.md) §17・§18 は値を変えず、確認の注記だけを足しました。台帳の記録は [`../../PENDING_TASKS_CLOSED.md`](../../PENDING_TASKS_CLOSED.md) 1-275、門の記録の複写は [`../h3-te-chunk-order-2026-11-04/`](../h3-te-chunk-order-2026-11-04/) です。

一次記録は、git 管理外の **`outputs/comfort-calib-h3-2026-11-04-te/`** にあります。下の表のパスは、その中のパスと同じです。数値の集計と考察（`RESULTS.md`）は門のフォルダの側にあり、複写は [`../h3-te-chunk-order-2026-11-04/RESULTS.md`](../h3-te-chunk-order-2026-11-04/RESULTS.md) です。

## 写したもの

`.md` と `.txt` の先頭には、複写の断り書きを 1 行足しました。その後ろは元のファイルのままです。

| ファイル | 中身 |
|---|---|
| `README.md` | 本ファイル |
| `progress.md` | 計測担当の時系列の記録（点ごとの表・復元） |
| `README_te.md` | 作業場所の説明（位置づけ・準備・点の順・合否・止める条件・復元）。「位置づけ」の「塊を等分に変えた後で」は第 1 版のときの書きぶりの名残で、実際に測ったのは第 2 版（計算順の変更）の後のコードです（`progress.md` の冒頭と `git_state_*.txt` の hash-object `75274125…`） |
| `git_state_recalib_start.txt`・`git_state_before_load_ref2va.txt` | 再較正の開始時と Ref2VA の読み込みの前の git の状態 |

## 写さなかったもの

どれも原本の `outputs/comfort-calib-h3-2026-11-04-te/` にあります。

- 点ごとの記録 `runs/<label>/`（`run.json`〔点ごとの正本〕・`request.json`・`job.json`・`wddm.csv`・`server_slice.log` ほか）と `manifest.jsonl`。**作業場所にだけあります。**
- 計画ファイル `plans/`（12 本）。
- 計測台の台本一式（`calib.py`・`mkplan.py`・`drive2.sh`・`rp.sh` ほか）・`selftest/`・`materials/`。**台本は 11-04 の作業場所 `outputs/comfort-calib-h3-2026-11-04/` からの複製で、中身は同じです**（固定パスだけを本フォルダへ書き換えた。`README_te.md` の「準備」）。11-04 の計測台の差分は [`../comfort-calib-h3-2026-11-04/calib_k.diff`](../comfort-calib-h3-2026-11-04/calib_k.diff) に写してあります。
- ログと状態のファイル（`drive2_*.log`・`commit_log.csv`・`preflight_*`・`rp_*`・`run_p_*`・`idle_*`・`load_*`・`original_state.json`・`restore_state_output.txt`・`luid.json` ほか）、待機の較正 `calib/`。
