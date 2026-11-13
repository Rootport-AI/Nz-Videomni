# 台帳 1-285（MiniMax H3 の fp8 と w4a8 の重みでの快適上限の較正）の記録の複写（2026-11-13）

MiniMax H3 の快適上限を、重みの形式ごと（Comfy-Org の fp8・Kijai の w4a8・TenStrip Eros）に測った記録のうち、文字の記録だけをここへ写しました。**結論と数値の正本は記録簿 [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §212** です。線の判断に要る値と規則の形は快適上限表 [`../../COMFORT_LIMIT_TABLE.md`](../../COMFORT_LIMIT_TABLE.md) §21、台帳の記録は [`../../PENDING_TASKS_CLOSED.md`](../../PENDING_TASKS_CLOSED.md) 1-285 にあります。

一次記録は、git 管理外の `outputs/comfort-calib-h3-quant-2026-11-13/`（計測の作業場所）にあります。下の表のパスは、その中のパスと同じです。ただし `analysis/check_results_quant.py` だけは、原本では作業場所の直下（`outputs/comfort-calib-h3-quant-2026-11-13/check_results_quant.py`）にあり、ここでは集計の台本をまとめるため `analysis/` に入れました。計画の写し `計画.md` の原本は、監督の計画ファイル（`graceful-fluttering-reddy.md`。Claude Code の計画フォルダにあり、git 管理外）です。

## 写したもの

`.md` と `.txt` の先頭には、複写の断り書きを 1 行（と空行）足しました。その後ろは元のファイルのままです。`.py`・`.json`・`.diff` は中身のままです。

| ファイル | 中身 |
|---|---|
| `README.md` | 本ファイル（写したもの・写さなかったもの・正本の所在） |
| `RESULTS.md` | 結果の集計（結論・条件・対照・全点の表と注記・木の結論と帯・予測との照合・仕組みの要約・留保・判断点・記録の所在）。`analysis/make_results_md.py` が `analysis/results.json` から組み立てたもの |
| `progress.md` | 計測担当の時系列の記録（弾ごとの点の表・基準点の二重起動の事故と取り直し・復元） |
| `計画.md` | 監督の計測計画（オーナー裁定・条件の突き合わせ・仮説と予測・判定規則・点の表と木・結論の形・敵対的レビューの採否）。承認時の本文のまま |
| `analysis/check_results_quant.py` | `runs/` の `run.json` から全点の値を組み直し、計画と記録を突き合わせる台本（原本は作業場所の直下） |
| `analysis/results.json`・`analysis/results_table.md` | `check_results_quant.py` の出力（全点の値・結論・予測との照合・注記） |
| `analysis/make_results_md.py` | `RESULTS.md` を `results.json` から組み立てる台本 |
| `analysis/mechanism_w4a8.md` | w4a8 で線が下がる仕組みの考察（OOM の場所の特定・comfy-kitchen の二重確保・割り当てと予約の読み・Eros・fp8 への影響） |
| `calib_quant.diff`・`drive2_quant.diff` | 計測台 `calib.py`・運転の台本 `drive2.sh` の差分（11-04 の写しに、形式を run.json に残す・基準点の重みの照合・復帰の重みを点の計画から読む、を足した分） |
| `selftest/parser_selftest_quant.diff` | 自己試験 `selftest/parser_selftest.py` の差分（上の差分に合わせた試験の追加） |
| `load_int8.txt`・`load_w4a8.txt`・`load_fp8.txt`・`load_eros.txt`・`load_int8_x.txt`・`load_eros_x.txt` | 重みを読み込んだ直後のワーカーの `LOAD_OK` 行（形式ごとの層の数と `fp8_mode`。`_x` は計画外の 2 点〔弾 X〕のときの読み込み） |
| `git_state_*.txt` | 開始時と各重みの読み込み直前の git の状態（HEAD・作業ツリーの差・時刻ほか） |

## 写さなかったもの

どれも原本の `outputs/comfort-calib-h3-quant-2026-11-13/` にあります。

- 点ごとの記録 `runs/<label>/`（`run.json`〔点ごとの正本〕・`metadata.json`・`point.json`・`request.json`・`job.json`・`server_slice.log`・`wddm.csv` ほか）。**作業場所にだけあります。**
- 計画ファイル `plans/`、素材 `materials/`、ログ（`drive2_*.log`・`rp_*.txt` ほか）。
- 計測台と台本の本体（`calib.py`・`calib_quant.py`・`mkplan.py`・`drive2.sh`・`drive2_quant.sh`・`sum.py` ほか。差分だけを写しました）と、自己試験の本体と出力（`selftest/` の差分以外）。
- `commit_log.csv`（主メモリのコミット量の監視）・`manifest.jsonl`（投入の記録）・`original_state.json`（復元先の状態）・`luid.json`、各コマンドの出力 `*_output.txt`、事前検査 `preflight_*`。

## 個人情報の除去

写したファイルをすべて利用者のフォルダの名前（Windows の利用者フォルダの英語名）で検索し、利用者名を含むパスがあれば `<user>` に置き換える方針で確かめました。該当は 0 箇所で、置き換えはしていません（本文は原本のまま）。
