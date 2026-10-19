# MiniMax H3 の末尾キーフレームと中間キーフレーム（台帳 §1-97）第 1 弾の計画・報告書・門の成果物

作成: 2026-10-19。門の成果物 `outputs/h3-keyframes-2026-10-19/` は git の管理外なので、計画・報告書・台本・結果をここに複写した。結論と読み方は [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §170、設計は [`../../MINIMAX_H3_DESIGN.md`](../../MINIMAX_H3_DESIGN.md) §2（裁定 56・57・59〜61）と §3（キーフレームの条件行）。

## 読む順番

1. `PLAN.md`: 承認済みの計画（裁定 54〜61・位置の規則と強度の省略・エンジンの条件行の一般化・変えるファイル・テスト・弾と門・目視の前に伝える画面の挙動）。
2. `REVIEW_plan.md`: 計画の敵対的レビュー（Critical 2・Major 5・minor 13。全件採用）。
3. `reports/`: 調査・実装・レビュー・門の報告書。
   - `R1_official_lastframe_and_addguide.md`: 一次資料の調査（公式の末尾フレームと ComfyUI の `MiniMaxH3AddGuide`）。
   - `1a_report.md`（バックエンド）・`1b_report.md`（操作パネル）・`1c_report.md`（Gradio と MCP）: 実装の報告。float64 の総当たりの結果は `1a_report.md`。
   - `REVIEW_code.md`: コードレビュー 1-G5（Critical 0・Major 1・minor 6）。
   - `fixes_report.md`: 1-G5 の minor 6 件の小修正。
   - `1d_part1_report.md`: 文書の第 1 部（設計正本・仕様書ほか 8 本）の報告。
   - `g2_prep_report.md`: GPU の門の準備（台本・試験画像・指紋の条件の出典）。
   - `gates_report.md`: GPU の門 1-G2〜1-G4 と API 経由の実行の報告。
4. `gates/`: GPU の門の結果と台本。

## `gates/` の中身

- 集計: `GATES_RESULT.md`（人が読む形）・`gates_result.json`（機械が読む形）。手順書は `RUNBOOK.md`。
- 1-G1 の GPU 付き pytest: `pytest_gpu.log`。
- 1-G2: `g2_keyframe_parts.py`（台本）・`g2_keyframe_parts.cmd`・`g2_keyframe_parts.json`（結果）。正解側のファイルは `outputs/h3-gate2/oracle/diffsynth_h3/` にだけあり、ここには写していない。
- 1-G3・1-G4 のセルフテスト 6 本: 各回の `.cmd`（コマンド）と `.json`（ワーカーの結果）。名前は `g3_t2v`・`g3_i2v_first`・`g3_i2v_first_vram16`・`g4_first_last`・`g4_last_only`・`g4_kf10`。
- PSNR: `psnr_keyframes.py`（台本）と `*.psnr.json`。`baseline_J1.psnr.json` は比べる相手（§166 の J1）、`g3_t2v_vs_kf10.psnr.json` は試験画像が基準の動画から取り出したものであることの確かめ。
- 試験画像の取り出し: `extract_keyframes.py`。
- 第 2 弾 2-G1（中間のキーフレーム。§170.10）で足したもの: `g5_mid_only`・`g5_first_mid_last` の `.cmd`・`.json`・`.psnr.json`、比べる行の測り直し `baseline_mid62_*.psnr.json`、試験画像 `kf_mid62.png` の取り出しの台本 `extract_mid62.py` と記録 `extract_kf_mid62.json`、報告 `reports/gate2_report.md`。`GATES_RESULT.md`・`gates_result.json` は第 2 弾を足した版に差し替えた。目視用の動画と画像は Hugging Face の非公開データセット `Rootport/Nz-LTX23-eval-clips` の `round9-h3-keyframes/` にある。

## 複写していないもの

- 出力動画 `*.mp4` と試験画像 `kf_first.png`・`kf_last.png`・`kf10_*.png`。どれも `outputs/h3-keyframes-2026-10-19/` にある。
- API 経由の実行のフォルダ `g3_api/`（`g2b_api.json`・3 本の mp4 と `*_metadata.json`・ログの抜粋）。`outputs/h3-keyframes-2026-10-19/g3_api/` にある。製品の置き場に残ったジョブのフォルダとアップロードは §170.7 (d)。
- 各回の標準出力・終了コード・実行前の GPU の状態（`*.stdout`・`*.exit.txt`・`*.gpu_before.txt`）、1-G2 の予行（`*.dryrun.*`）とログ。`outputs/h3-keyframes-2026-10-19/` にある。
