# MiniMax H3 の A2V（音声から映像。台帳 §1-103）の計画・調査・レビュー・門の成果物

作成: 2026-10-25。元の報告書は監督のセッションの作業場所（git の管理外）にあったので、ここに複写した。結論と読み方は [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §178、式と仕組みは [`../../MINIMAX_H3_DESIGN.md`](../../MINIMAX_H3_DESIGN.md) §3.1、API の契約は [`../../../Videomni_Backend_Specification.md`](../../../Videomni_Backend_Specification.md)（v0.5.101）。数値の正本は §178 で、本書は数値を書き写さない。

## 読む順番

1. `R1_audio_encoder_retake.md`: 事前調査。音声 VAE の符号化器の構造、DiffSynth の Retake と ComfyUI の連続値マスクの仕組み、長さの規則（一次資料つき）。
2. `R2_chain_api_entry.md`: 事前調査。受け口の棚卸し（`/generate/chain` が H3 に使えない理由と、単発の `/generate` に載せる案）。
3. `P0_plan.md`: 承認された実装計画（設計 §2.1〜§2.8・門 §4・レビューの採否 §9）。**計画の時点の Strength の既定は 1.0 で、オーナーの目視の後に 0.5 へ変えた**（§178.7）。
4. `R3_plan_review.md`: 計画の敵対的レビュー（Critical 0・Major 5・Minor 11）。採否は計画の §9 と §178.3。
5. `R4_engine_diff_review.md`: 実装差分のレビュー（弾 1〜2。エンジン側）。採否は §178.8。
6. `R5_app_fe_diff_review.md`: 実装差分のレビュー（弾 3〜5。アプリ・操作パネル・MCP）。採否は §178.8。
7. `G3_report.md`: GPU の門 G3-0（変更の前後の一致）と G3（オーナーの素材で Strength 1.0 と 0.5）の報告。
8. `G3b_report.md`: GPU の門 G3b（既定を決めるための比べ合わせ。キーフレームの有無 × Strength 0.5／0.75）の報告。

## 記録

- `g0.json`: 門 G0（CPU の往復試験。`scripts/h3_a2v_g0_roundtrip.py` の出力）。作業場所は `outputs/h3_a2v_g0/`。
- `g2/`: 門 G2 の試験の記録。`g2_*` は実装の直後、`g2b_*` は実装差分のレビューの手直しの後（アプリの pytest・H3 の pytest・vitest・typecheck）。

## 複写していないもの

- 出力の動画と音声（G0 の wav 2 本、G3・G3b の mp4）。作業場所 `outputs/h3_a2v_g0/`・`outputs/h3_a2v_g3/`・`outputs/h3_a2v_g3b/` にある（git の管理外）。G3・G3b の台本（`drive.py`・`analyze.py`・`drive_g3b.py`）も同じ作業場所にある。
