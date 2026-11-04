# 台帳 1-274（MiniMax H3 の快適上限の再較正）の記録の複写（2026-11-04）

2026-11-04 の断片化の修正（[`../../PENDING_TASKS_CLOSED.md`](../../PENDING_TASKS_CLOSED.md) 1-270・1-271）の後に、MiniMax H3 の快適上限を測り直した記録のうち、文字の記録だけをここへ写しました。**結論と数値の正本は記録簿 [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §196** です。腕ごとの値と規則の形は快適上限表 [`../../COMFORT_LIMIT_TABLE.md`](../../COMFORT_LIMIT_TABLE.md) §17・§18、台帳の記録は [`../../PENDING_TASKS_CLOSED.md`](../../PENDING_TASKS_CLOSED.md) 1-274 にあります。

一次記録は、git 管理外の `outputs/comfort-calib-h3-2026-11-04/`（計測の作業場所）にあります。下の表のパスは、その中のパスと同じです（`calib_k.diff` だけは `outputs/comfort-calib-h3-2026-11-04-patch/` から写しました）。

## 写したもの

`.md` と `.txt` の先頭には、複写の断り書きを 1 行足しました。その後ろは元のファイルのままです。`.py`・`.json`・`.diff` は中身のままです。

| ファイル | 中身 |
|---|---|
| `README.md` | 本ファイル（写したもの・写さなかったもの・正本の所在） |
| `RESULTS.md` | 数値の集計と仕組みの考察（結論・全点の表・突き合わせ・結論の形・仕組み・修正前との比較・弾 E・留保・終了処理） |
| `progress.md` | 計測担当の時系列の記録（点ごとの表と、途中で止めて報告したこと） |
| `TE_STAGE_INVESTIGATION.md` | TE（テキストエンコーダー）の段の VRAM 溢れの調査報告（読むだけの調査。回避の候補を含む） |
| `analysis/collect.py`・`collect_output.txt`・`points.json`・`points_table.md` | `runs/` から全点の値を抜き出す台本と、その出力（全点の表） |
| `analysis/check_progress.py`・`check_progress_output.txt` | `progress.md` と `run.json` の突き合わせ |
| `analysis/line_forms.py`・`line_forms_output.txt` | 軸ごとの分かれ方と、参照の行・text の行の重さの範囲 |
| `analysis/fit_and_time.py`・`fit_and_time_output.txt` | 割り当ての山の式と時間モデルの照合 |
| `analysis/reserve_gap.py`・`reserve_gap_output.txt` | 予約と割り当ての差（当てはめ） |
| `analysis/boundary_estimate.py`・`boundary_estimate_output.txt` | 当てはめの形で解いた境界 |
| `calib_k.diff` | 計測台 `calib.py` と自己試験 `selftest/parser_selftest.py` の差分（11-03 の版に、キーフレームあり〔K>0〕の点の要求〔`conditioning_images`〕と text の行の予測を戻した分。K>0 と参照の併用は断る） |

## 写さなかったもの

どれも原本の `outputs/comfort-calib-h3-2026-11-04/` にあります。

- 点ごとの記録 `runs/<label>/`（`run.json`〔点ごとの正本〕・`job.json`・`request.json`・`server_slice.log`・`wddm.csv`・`worker_tail.csv` ほか）。**作業場所にだけあります。**
- 計画ファイル `plans/`、計測台のコード（`calib.py`・`mkplan.py` ほか）・素材 `materials/`・自己試験 `selftest/`・運転の台本の写し `drivers_scratch/`。
- ログ（`drive2_*.log`・`commit_log.csv`・`preflight_*`・`rp_*`・`run_p_*`・`run_fl_*`・`run_r_*`・`tree*.log`・`load_*`・`idle_*` ほか）と、開始時・読み込み前の git の状態（`git_state_*.txt`）・終了処理の出力（`restore_state_output.txt`）。
