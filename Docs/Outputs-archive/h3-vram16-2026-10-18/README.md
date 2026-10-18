# MiniMax H3「VRAM 16GB mode」（台帳 §1-101 の第 2 段〜）の計画・報告書・門の成果物

作成: 2026-10-18。門の成果物 `outputs/h3-vram16-gates/` と計測台 `outputs/h3-vram-2026-10-18/` は git の管理外なので、計画・報告書・台本・結果をここに複写した。結論と読み方は [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §169、設計は [`../../MINIMAX_H3_DESIGN.md`](../../MINIMAX_H3_DESIGN.md) §6.5。

## 読む順番

1. `PLAN_vram16.md`: 承認済みの計画（裁定 44〜52・on の経路で行うこと・エンジン側とアプリ側の設計・弾と門）。
2. `reports/`: 棚卸しと敵対的レビュー。
   - `E1_engine.md`（エンジン）・`E2_app.md`（アプリ側）・`E3_webui.md`（操作パネル）: 計画の前の棚卸し。
   - `REVIEW_plan_design.md`: 計画のレビュー（Critical 1・Major 8・minor 13。全件採用）。
   - `REVIEW_codeA.md`: 弾 A（エンジン）のコードレビュー（Critical 0・Major 1・minor 12）。
   - `REVIEW_codeB.md`: 弾 B（アプリ側と操作パネル）のコードレビュー（Critical 0・Major 1・minor 9）。
3. `gates/`: 弾 A の GPU の門（A-G5・A-G2・A-G3・A-G3b・A-G4）の結果と台本。
4. `measure_h3_vram.py`: `--vram16` 系の引数を足した最新の計測台。第 1 段の写し（`../h3-vram-2026-10-18/measure_h3_vram.py`）とは別物で、こちらが新しい。

## `gates/` の中身

- 集計: `G5_summary.jsonl`（分割の単位と常駐数の比較。8 腕）・`G4_summary.jsonl`（製品の条件と台帳の目標の形。3 本）・`arms.log`。
- A-G2: `g2_block_equal.json`（製品の形の 50 ブロック比較）・`g2_repeat_b{0,22,40}.json`（再現性）・`g2_sage_diag_b22.json`（sage のヘッド数依存の切り分け）・`g2_smoke.json`・`g2_block_equal.log`。
- A-G3／A-G3b: `g3_off.json`・`g3_on.json`（と `.log`）。mp4 は省いた。
- B-G5: `b_g5_metadata.json`（オーナーが操作パネルから行った実行〔job `871a9b96`、2026-10-18 23:01〜23:10〕の `metadata.json`。動画は省いた）。
- 台本: `g2_block_equal.py`・`g2_sage_diag.py`・`g3.sh`・`g4.sh`・`arm.sh`・`g5a.sh`・`g5b.sh`・`g5c.sh`・`summarize.py`、プロンプト `f0_prompt.txt`・`g5_prompt.txt`。
- `runs/<腕>/`: 各腕の `run.json`・`memory_stats_step*.json`・`wddm.csv`・`dmon.txt`・`stdout.log`。腕の名前は `G5a_h8_r4096_b4`（ヘッド群 8・行の塊 4096・常駐 4）のように条件を並べたもの。
  - `G5a_h8_r4096_b4_try1_devbug`: 最初の試行。`set_per_process_memory_fraction` に番号の無い `cuda` を渡す実装の誤りで上限が課せなかった（`_cuda_index` で直した）。
  - `G5a_h8_r4096_b4_overlapped`: 取り直す前の h8 の回。止めたはずの h28 の腕が孤児プロセスとして残り、重なって走った（`arms.log` の注記）。取り直した `G5a_h8_r4096_b4` を結果として使った。

## 複写していないもの

- 出力動画 `out.mp4`・`g3_off.mp4`・`g3_on.mp4`。
- メモリの履歴の pickle（今回の門では採取していない）。
- `gpu_process_memory_*.csv`・`luid.json`（採取器の補助。正本の `outputs/h3-vram16-gates/runs/` には残っている）。
