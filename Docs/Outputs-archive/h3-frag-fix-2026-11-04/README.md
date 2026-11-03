# 台帳 §1-270・§1-271 の記録の複写（2026-11-04）

MiniMax H3 の VRAM 16GB mode で、ステップの境目の float32 の穴と、TE（テキストエンコーダー）の段から GPU に返らないセグメントをなくした修正（コミット `c5b1095`・`bcf0caa`）の記録のうち、文字の記録だけをここへ写しました。**結論と数値の正本は記録簿 [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §195** です。設計は設計正本 [`../../MINIMAX_H3_DESIGN.md`](../../MINIMAX_H3_DESIGN.md) §6.5 と裁定 92 にあります。

門（実機での確かめ）の一次記録は、git 管理外の `outputs/h3-frag-fix-2026-11-04/` にあります。下の表のパスは、その中のパスと同じです。

## 写したもの

各ファイルの先頭に複写の断り書きを 1 行足しました（`.py` はコメント行）。その後ろは元のファイルのままです（改行コードだけ CRLF にそろえました）。

| ファイル | 中身 |
|---|---|
| `README.md` | 本ファイル（写したもの・写さなかったもの・正本の所在） |
| `research_1-270_1-271.md` | 設計の前の調査報告（読み取り専用。監督の作業領域で作られたもの。候補の評価・割り当て器と cuBLAS の作業領域の裏取り・門の材料） |
| `GATE.md` | 門の記録（工程ごとの合否・指紋の比較・r1080_277 の時間とメモリ・境目の全セグメントと判定・気づいたこと・expandable segments の確認） |
| `preconditions.txt`・`postconditions.txt` | 門の前と後の状態（git・GPU の使用量・python のプロセス・ワーカーの孤児） |
| `arms.log` | 各工程の開始と終了の時刻・終了コード・待ち明けの GPU の使用量・指紋 |
| `g3/compare.txt` | 4c の指紋と時間とピークの、前回の門との比較の全表 |
| `analysis/r1080_277_fix_step1.md` | 4d のステップ 1 の境目のスナップショットの集計（`analyze_snapshot.py` の出力） |
| `analysis/judge_r1080_277_fix_step1.txt` | 4d の判定 (i)〜(iv) の出力（全セグメントの表を含む） |
| `analysis/judge_segments.py` | 判定の台本（集計の json の `segments[]` から判定する。読み取りだけ） |
| `analysis/memory_stats_summary.txt` | 各ステップの境目の `memory_stats` の要点 |
| `4e_new_name.txt`・`4e_old_name.txt` | 4e の出力（環境変数の新名 `PYTORCH_ALLOC_CONF` と旧名 `PYTORCH_CUDA_ALLOC_CONF`） |

## 写さなかったもの

どれも原本の `outputs/h3-frag-fix-2026-11-04/` にあります。

- 動画（`g3/g3_off.mp4`・`g3/g3_on.mp4`・`runs/r1080_277_fix/out.mp4`）。
- スナップショット（`runs/r1080_277_fix/snapshot_step1.pickle`）。
- json（`runs/r1080_277_fix/run.json`・`memory_stats_step1〜4.json`・`luid.json`、`analysis/r1080_277_fix_step1.json`、`g3/g3_off.json`・`g3/g3_on.json`）と csv（`runs/r1080_277_fix/gpu_process_memory_*.csv`・`wddm.csv`）。
- ログの全文（`g3/g3_*.log`・`g3/g3_*.ts.log`・`runs/r1080_277_fix/stdout.log`・`dmon.txt`・`console_r1080_277_fix.txt`・`4b_gpu_tests.txt`・`analysis/analyze_console.txt`）。
- 4c の台本と比較の台本（`g3/g3_arm.sh`・`g3/compare_g3.py`・`g3/step_times.py`）。
- 実施計画（監督の作業領域にあります。要点は記録簿 §195.2〜195.6）。
