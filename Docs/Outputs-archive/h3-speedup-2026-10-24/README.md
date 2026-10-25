# MiniMax H3 の 1 ステップの高速化・第 1 弾（台帳 §1-105。QKV の量子化の共有）の調査・計画・レビュー・門の成果物

作成: 2026-10-25。作業場所 `outputs/h3-speedup-2026-10-24/` は git の管理外なので、報告書・台本・結果をここに複写した。結論と読み方は [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §177、設計は [`../../MINIMAX_H3_DESIGN.md`](../../MINIMAX_H3_DESIGN.md) §5（例外）と §6.5。数値の正本は §177 で、本書は数値を書き写さない。

ファイルの並びは作業場所と同じにしてある（報告書の中に出てくるファイル名が、そのままここで引ける）。

## 読む順番

1. `R1_investigation.md`: 事前調査（GPU を使わないコード読解）。① QKV の量子化の共有・② ステップ間の先読み・③ CPU から固定メモリへの先読みの仕組みと見積もり。
2. `R2_measurement.md`: 実測（M0 のマイクロベンチ・腕 N／E／P）。① の短縮の上限と、②③ を見送った根拠の数値。
3. `P1_plan_qkv_share.md`: ① の実装計画（Opus 起草）。
4. `P1_review.md`: 計画の敵対的レビュー（Critical 0・Major 2・minor 13）。採否は §177。
5. `A3_code_review.md`: 実装差分のコードレビュー（門 A-3。Critical 0・Major 0・minor 5）。採否は §177。
6. `R3_gates.md`: GPU の門 A-4〜A-10 の結果（条件・ステップごとの表・計画と食い違った点）。

## 台本と結果

- **M0（量子化 1 回の時間）**: `m0_quant_bench.py`・`m0_result.json`。
- **計測台**: `measure_h3_speedup.py`（第 1 段の計測台 `outputs/h3-vram-2026-10-18/measure_h3_vram.py` の写しに、腕のフラグ〔`--arm-events`・`--arm-profile`〕と、「前」を作るための `--no-qkv-share`〔計測台の中で共有の検査を不合格に差し替える。製品にはつまみを足していない〕を足したもの）。
- **外部の GPU の負荷の判定**: `step_rx.py`（各ステップの時間の窓で、dmon の PCIe の受信の中央値を出す）・`step_rx.jsonl`（R2 の各回）・`r3/a8_step_rx.jsonl`（A-8 の各回）。
- **A-0〜A-2・A-4（テスト）**: `a0_pytest_before.txt`（変更前の件数の基準）・`a1_pytest_after.txt`・`a1b_pytest_after_review.txt`（A-3 の指摘を直した後）・`a2_app_pytest.txt`（アプリの仮想環境の全件）・`a4_pre_gpu_tests.txt`（実装担当が回した GPU テスト）・`r3/a4_gpu_pytest.txt`（門 A-4）。
- **A-5（製品の形の 50 ブロック比較と呼び出しの計数）**: `a5_g2_wrapped.py`（包みの台本）・`g2_block_equal_r3.py`（`outputs/h3-vram16-gates/g2_block_equal.py` の写し。`rope.build_coords` の引数名の変更〔§1-97〕に合わせた 1 か所と、`--report` の必須化だけを変えた）・`a5_g2_block_equal.json`・`a5_counts.json`。
- **A-6（指紋 g3 の off）**: `a6_g3_off.sh`・`r3/a6_g3_off.json`。
- **A-7（LoRA の門の on 側 6 本）**: `a7_gates.sh`（保管庫 `../h3-lora-2026-10-24/gates/gates.sh` の写し。出力先だけ変更）・`a67_chain.sh`（A-6 と A-7 を 1 本ずつ続けて回す台本）・`r3/a7_gates/*.json`。
- **A-8・A-9（前後の速度と VRAM）**: `arm_r3.sh`（腕の間の冷まし方）・`a8_runs.sh`・`r3/a8_summary.json`。
- **A-10（長尺 T2V。記録だけ）**: `a10_runs.sh`・`r3/a10_summary.json`。

## 複写していないもの

- 出力動画（`*.mp4`）。指紋（sha256・md5）は各 json と §177 にある。
- 腕 P の Chrome trace `runs/arm_P_h8_r4096_b4/trace_step2.json`（約 88 MB）。
- 各回の採取物 `runs/`（run.json・dmon.txt・wddm.csv・stdout.log・memory_stats）・各ログ（`r3/*.log`・`arms.log`・`m0.log`）・`*.console.log`・回す直前の `nvidia-smi` の記録（`*.precheck_compute_apps.csv`）・補助の台本（`r3/precheck.sh`・`r3/wait_quiet.sh`・`r3/a8_summary.py`）・R2 の腕の台本（`arm.sh`・`arm_n*.sh`・`arm_e.sh`・`arm_p.sh`）・計測台の変更前の写し（`r3/measure_h3_speedup.before_r3.py`）。正本の `outputs/h3-speedup-2026-10-24/` には残っている。
