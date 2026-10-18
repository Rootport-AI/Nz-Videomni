# MiniMax H3 対応 α版 第 1 弾の調査報告・計画・レビューの保管（2026-10-17）

第 1 弾（記録は [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §165）で使った調査報告と計画・レビューの**そのままの写し**である。もとはセッション限りの一時領域にあり、次のセッションから読めなかったため保管した。**設計の正本は [`../../MINIMAX_H3_DESIGN.md`](../../MINIMAX_H3_DESIGN.md)** で、本フォルダの文書と食い違うときは正本を採る。行番号・版・数値はすべて 2026-10-17 時点のもの。

| ファイル | 中身 |
|---|---|
| `R1_stack.md` | 仮想環境のスタック（torch cu130・comfy-kitchen の公開 API と制約・issue #136・SageAttention・uv の解決）。2-2 追加＝量子化テンソル型が block swap に乗らない理由と関数 API `int8_linear` |
| `R2_model.md` | モデル構造と参照実装の読解（DiffSynth の棚卸し・DiT の前向き・RoPE・系列の並び・TE・条件付け・VAE と 17n+5・サンプラ・ComfyUI の判定の振る舞い・間違えやすい点 19 件） |
| `R3_inventory.md` | 本製品の受け皿の棚卸し（ワーカーの作法・block swap の前提・ヘッダ判定の経路・記述子・ディスパッチ・API の 8n+1・インストーラ・落ちるテスト・置き場） |
| `INTERFACE.md` | 第 1 弾の担当 3 体が共有した境界の仕様（署名・形・dtype・メッセージのキー名・所有者）。第 2 弾以降の正本は設計正本 §8 |
| `PLAN_stage1.md` | オーナーが了承した第 1 弾の計画（敵対的レビュー反映後） |
| `REVIEW_plan1.md` | 計画の敵対的レビュー（Critical 1・Major 6・Minor 14。全採用） |
| `REVIEW_code1.md` | コードの敵対的レビュー（Critical 0・Major 5・Minor 16。採否は §165.6） |
