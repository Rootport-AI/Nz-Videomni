# 台帳 1-276（MiniMax H3 の Chained〔連結生成〕の快適上限の較正）の記録の複写（2026-11-08）

MiniMax H3 の Chained の各クリップの快適上限を測った記録のうち、文字の記録だけをここへ写しました。**結論と数値の正本は記録簿 [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §199** です。線と行数の式は快適上限表 [`../../COMFORT_LIMIT_TABLE.md`](../../COMFORT_LIMIT_TABLE.md) §19 にあります。

一次記録は、git 管理外の `outputs/comfort-calib-h3-chain-2026-11-08/`（計測の作業場所）にあります。下の表のパスは、その中のパスと同じです。

## 写したもの

`.md` と `.txt` の先頭には、複写の断り書きを 1 行足しました。その後ろは元のファイルのままです。`.py`・`.json`・`.diff` は中身のままです。

| ファイル | 中身 |
|---|---|
| `README.md` | 本ファイル（写したもの・写さなかったもの・正本の所在） |
| `README_chain.md` | 較正台の使い方と、11-04 の較正台（`outputs/comfort-calib-h3-2026-11-04-te/`）からの差分の説明 |
| `RESULTS.md` | 数値の集計と考察（結論・条件・判定規則・点の表・A-U の OOM〔GPU のメモリ不足〕・線の 2 択の判定・cap・decode の窓・腕 D・留保・所要時間・終了処理） |
| `progress.md` | 時系列の記録（作業場所のログから翌朝に組み立てたもの。出典は行ごとに書いてある） |
| `analysis/collect.py`・`collect_output.txt`・`points.json`・`points_table.md` | `runs/` から全点の値をクリップごとに抜き出す台本と、その出力（全点の表） |
| `analysis/compare_1104.py`・`compare_1104_output.txt` | 連結のクリップと 11-04 の Single の同じ形の点の比較（読む 11-04 の値は [`../comfort-calib-h3-2026-11-04/analysis/points.json`](../comfort-calib-h3-2026-11-04/analysis/points.json)） |
| `analysis/make_results_h3_output.md`・`make_results_wall_output.md` | 計測台の `make_results.py` の出力（H3 の全計測点の表と、所要時間の表） |
| `analysis/digest_output.md` | 計測台の `digest.py` の出力（点ごとの条件・判定・主メモリの要約） |
| `analysis/commit_table_output.md` | 計測台の `commit_table.py` の出力（点ごと・窓ごとの主メモリのコミット量の山） |
| `calib_chain.diff` | 計測台の差分（11-04 の `-te` フォルダとの `diff -ruN`。`calib.py`・`mkplan.py`・集計の台本・運転の台本・自己試験に、連結の点を打つ分を足したもの。Single の経路は変えていない） |

## 写さなかったもの

どれも原本の `outputs/comfort-calib-h3-chain-2026-11-08/` にあります。

- 点ごとの記録 `runs/<label>/`（`run.json`〔点ごとの正本〕・`job.json`・`request.json` ほか）。**作業場所にだけあります。**
- 計画ファイル `plans/`、計測台のコード（`calib.py`・`mkplan.py` ほか。11-04 からの変更は `calib_chain.diff` にある）・素材 `materials/`・`keyframe_f0.png`・自己試験 `selftest/`・段ごとの受け入れの出力 `stage_logs/`。
- ログ（`drive2_*.log`・`commit_log.csv`・`manifest.jsonl`・`preflight_*`・`rp_*`・`run_p_*` ほか）・計測台の `report.md`・開始時の状態（`git_state_start.txt`・`original_state.json`・`luid.json`）・自動復帰の出力（`recover_output.txt`）。
- 作業場所の `README.md` と `README_te.md`。複製元から持ち越した前の較正の説明で、今回の較正の説明ではありません（11-03 の参照つきの較正と 11-04 の再較正のもの。それぞれ [`../comfort-calib-h3-ref-2026-11-03/`](../comfort-calib-h3-ref-2026-11-03/) と [`../comfort-calib-h3-2026-11-04-te/`](../comfort-calib-h3-2026-11-04-te/) に写しがあります）。今回の説明は `README_chain.md` です。
