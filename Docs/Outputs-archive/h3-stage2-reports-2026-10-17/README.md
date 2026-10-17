# MiniMax H3 第 2 弾の調査報告・計画・レビューの保管（2026-10-17）

第 1 弾の保管（`../h3-stage1-reports-2026-10-17/`）と同じ趣旨で、セッション限りの一時領域にあった文書を保管する。**正本は設計正本 `Docs/MINIMAX_H3_DESIGN.md` と記録 `Docs/VERIFICATION_LOG.md` §166** で、本フォルダの文書と食い違うときは正本を採る。行番号は 2026-10-17 時点。

| ファイル | 内容 |
|---|---|
| `R4a_i2v_vae_encode_external.md` | I2V の先頭画像の符号化を ComfyUI・diffusers・DiffSynth で確認（平均かサンプルか・型・タイル・合わせ方・条件行・正規化・1 フレーム） |
| `R4b_i2v_convention_internal.md` | 本製品の既存 LTX の I2V の規約と `engine_h3/video_vae.py` の現状、モック |
| `R5a_backend_inventory.md` | バックエンド側の棚卸し（常駐・API の制約・事前検査と記述子・アダプタ・Gradio と MCP・復号 VRAM・落ちるテスト） |
| `R5b_frontend_inventory.md` | 操作パネル側の棚卸し（3 本のスライダー・機能スコープ・快適上限・フレーム規約と fps・Settings・モックとテストとデプロイ） |
| `PLAN_stage2.md` | 第 2 弾の計画（敵対的レビュー反映済み。了承 2026-10-17） |
| `REVIEW_plan2.md` | 計画の敵対的レビュー（Opus。Critical 1・Major 13・Minor 22。採否は計画 §10） |
| `INTERFACE2.md` | （実装中に追加）I2V の視覚塔の書式など参照実装から写した事項 |
