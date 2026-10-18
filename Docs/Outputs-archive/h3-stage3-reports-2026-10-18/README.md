# MiniMax H3 第 3 弾の調査報告・計画・レビューの保管（2026-10-18）

第 1 弾・第 2 弾の保管（`../h3-stage1-reports-2026-10-17/`・`../h3-stage2-reports-2026-10-17/`）と同じ趣旨。**正本は設計正本 `Docs/MINIMAX_H3_DESIGN.md`・README・記録 `Docs/VERIFICATION_LOG.md` §167** で、本フォルダの文書と食い違うときは正本を採る。行番号は 2026-10-18 時点。

| ファイル | 内容 |
|---|---|
| `I1_installer.md` | インストーラの機構の棚卸し（`install_model.ps1`・`install_ltx.ps1`・`Ensure-EngineVenv`・空き容量・検証の表・分岐点） |
| `I2_hf_sources.md` | 配布元（Comfy-Org/MiniMax-H3・MiniMaxAI/MiniMax-H3）の実物・ゲートの有無・ライセンスの要点 |
| `I3_gpu_support.md` | comfy-kitchen と SageAttention の GPU 世代対応・cu130 の要件 |
| `I4_readme.md` | README の改修箇所の特定（凍結ゾーン・不変条件・R-1〜R-5） |
| `PLAN_stage3.md` | 第 3 弾の計画（敵対的レビュー反映済み。了承 2026-10-18） |
| `REVIEW_plan3.md` | 計画の敵対的レビュー（Opus。Critical 0・Major 8・Minor 11。採否は計画 §9） |
| `REVIEW_readme3.md` | README の読み合わせレビュー（初めて H3 を足す利用者の視点。Major 2・Minor 17。全採用） |
| `REVIEW_code3.md` | インストーラのコードレビュー（Critical 0・Major 0・Minor 7。全採用） |
| `REVIEW_close3.md` | セッションのクローズ時の文書とコメントのレビュー（次のセッションの担当者の視点。Major 5・Minor 11。全採用） |
