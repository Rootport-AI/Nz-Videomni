# 1d（文書・第 1 部）の報告 — 台帳 §1-97

## 結論

計画 §3.6 の文書 8 本を直しました。コード・テストは触っていません。コミットも GPU の使用もしていません。VERIFICATION_LOG・PENDING_TASKS・PENDING_TASKS_CLOSED は第 2 部に残してあります。

確かめたこと:

- MINIMAX_H3_DESIGN.md の「視覚塔」は 0 件です。
- README の 1〜134 行は HEAD と同じです（改行コードをそろえて比べました。`git diff` の最初の差分は 670 行目）。
- 新しい裁定 54〜61 は研究ノート §10 に 1 回ずつあります。設計正本 §2 には、設計の規則の 56・57・59・60・61 が 1 回ずつあります（進め方の 54・55・58 は研究ノートだけで、既存の規則どおりです）。

## git diff --stat（文書のみ）

```
 .../Docs/API_REFERENCE.md                          | 16 +++--
 .../Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md    | 41 +++++++++++-
 Docs/MCP_SERVER_DESIGN.md                          |  5 +-
 Docs/MINIMAX_H3_DESIGN.md                          | 77 +++++++++++++---------
 Docs/MINIMAX_H3_RESEARCH_NOTES.md                  | 13 ++++
 Docs/MULTI_ENGINE_DESIGN.md                        | 13 +++-
 README.md                                          | 18 +++--
 Videomni_Backend_Specification.md                  | 38 ++++++-----
```

改行コードは、作業ツリーの元の形（CRLF。index は LF・autocrlf=true）に戻してあります。

## 文書ごとの中身

1. **Docs/MINIMAX_H3_DESIGN.md**
   - 冒頭の「現在の実装範囲」にキーフレームの対応を足しました。関連文書の表に §170 も足しました。
   - §2: 裁定 3 と裁定 19 に「更新（2026-10-19）」の行を足しました。裁定 56・57・59・60・61 を【オーナー裁定】として足しました（61 の了承の取り方は研究ノートを参照する形）。
   - §3: 「I2V（先頭フレーム）の条件行」を「I2V のキーフレームの条件行」にしました。並べる順・K 枚の値・ノイズの共有（`repeat(K,1)`）・時刻・タグ・RoPE の 3 つの式と F−1 だけ別の式にする理由を書きました（件数は §170 と `tests/test_h3_engine.py` を参照）。
   - §3: 画像読み取り部（vision tower）の規則（0 と F−1 だけ・ラベルは昇順・末尾だけなら `<Picture 1>`・中間だけなら読まない）と、キャンバスへの合わせ方も書き直しました。
   - §4: 段の表を K 枚の符号化にし、画像読み取り部を読む条件を直しました。
   - §6: meta の検査を直しました。
   - §8: ファイル表、`images`（最大 10 枚・0〜F−1・重複なし）、進捗の分母、`images_used`（枚数・改名しない）、`phases.image_encode.source_size`（リスト）、`H3VideoEncoder`、セルフテストの `--keyframe` を直しました。
   - §9: 規則表の位置の項目を足しました。
   - §11: キーフレームの行を足しました。
   - §12: 末尾・中間の行を消しました。
   - §13: 中間の品質の go／no-go と、§1-95 への含意（K×grid 行）を足しました。
2. **Docs/MULTI_ENGINE_DESIGN.md**
   - §5.2: 規則表が位置と強度の既定も持つことを 1 行で書きました。
   - §5.6 (4): JSON の例に `keyframes` を、説明に箇条を 1 つ足しました。
   - §7: 3 項目・呼ぶ順（fill_defaults → violation → normalize_keyframes）・スキーマで寄せないこと・強度の省略を書きました（式は request_rules.py の docstring を参照）。連結の validator が LTX の規則を呼ぶことも書きました。
   - §9.2: 行を 1 つ足しました。
3. **Videomni_Backend_Specification.md**
   - 版を v0.5.94・日付を 2026-10-19 にし、履歴の行を足しました。
   - §6.2 の凍結制約 7 を系統別に書き直しました（422 の順序・`normalize_keyframes`・H3 の範囲）。
   - §6.2b は、(b) の順序、(c) の表 2 行と強度の既定を公開しないこと、(e) の `keyframes` と説明、(f) の H3 の一覧（multi_keyframes を削除・keyframe_strength は「null 以外」・H3 の文言）、(g) の H3 の `strength: null` を直しました。
   - ConditioningImage の表（frame_idx は系統別・strength は `float | null` 既定 `null`）を直しました。
   - §6.6 の h3 欄（source_size のリスト・images_used）を直しました。
   - §6.7 に 2 鍵の削除と、同時に配置することの注意を書きました。
   - §12.2 の Gradio（系統別の寄せ・強度のスライダーを隠す・0.8 なら送らない）を直しました。
   - §16 に新しいテストの名前、付録 A・B に H3 の位置を足しました。
4. **Docs/MCP_SERVER_DESIGN.md**
   - D29 を足し、見出しを D1〜D29 にしました。D29 の中身は次のとおりです。
     - `strength: float | None = None`
     - H3 は任意のフレーム・最大 10 枚・strength は指定しない
     - `list_models` の説明の `constraints.keyframes`（system.py には既に入っていました）
   - §10 のテストの行も足しました。
5. **README.md**（134 行目より後だけ）
   - §7.6 の「できること」を書き直しました。キーフレームは最大 10 枚で、どのフレームにも置けます。先頭と末尾は公式の FL2VA と同じ扱い、中間は ComfyUI と同じ考え方の拡張で目視の途中です。強度はありません。「追加」を押すと 2 枚目は中点に置かれます。
   - 「できないこと」の表は、末尾と中間の行を消し、強度の行を書き直しました。
   - 新しい段落「プロンプトの書き方」を足しました。公式の指南 VIDEO_PROMPT_WRITING_GUIDE へのリンクを置き、本製品は自動で入れないことを書きました。
   - あわせて §4 の LTX の規則の行と §7.5 の「最終フレームちょうど」の行に、§7.6 への 1 文を足しました。
6. **フロントの API_REFERENCE.md と DEVLOG.md**
   - API_REFERENCE.md は次の 4 か所を直しました。
     - 最終更新の行
     - §3.2 の limits（2 鍵を削除し、同時に配置することの注意を書いた）
     - §3.5 の constraints の例と説明（`keyframes`）
     - §5.1 の conditioning_images の行と ConditioningImage（系統の格子・strength は省略可・omitStrength）
   - DEVLOG.md は、最終更新の行を直し、新しい §143 を足しました。§143 の中身は次のとおりです。
     - 結論
     - 変更（強度の行を非表示・multi_keyframes の削除・activeConstraints の格子・validPositions の offset 0 の修正・snapFrameIdx・omitStrength）
     - テスト
     - 仕様どおりの画面の挙動 4 点
     - 状態
7. **Docs/MINIMAX_H3_RESEARCH_NOTES.md**
   - §10 に「2026-10-19 の裁定（キーフレームの着手時・追記）」として 54〜61 を足しました。設計の規則には「→ 設計正本」の印を付けています。§13.3 は変えていません。

## 確信が持てない記述（監督に確かめてほしい点）

1. **冒頭の「現在の実装範囲」の書き方**: 指示は「末尾は正解データと照合済み」でした。ただ、1-G2 の本番（GPU）はまだ走っていません（`outputs/h3-keyframes-2026-10-19/` に `g2_keyframe_parts.json` はなく、dryrun だけがあります）。そこで「CPU で計算できる部分（トークン列・画像の入力・条件行の座標）は一致した。重みを使う照合と生成の門の結果は §170 に記録する」と、事実の範囲で書きました。門が通ったら、ここを「照合済み」に直してください。
2. **README の S.SS の式**: 「S.SS は末尾のフレームの時刻で、24fps なので (フレーム数 − 1) ÷ 24 秒」と書きました。これは私の推論で、公式の指南の本文は読んでいません。
3. **公式の指南の URL**: `https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` にしました。R1 の報告にある `docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` から組み立てたもので、開けるかどうかは確かめていません。「末尾だけを指定するときの書き方も同じ指南にあります」も、R1 の「末尾のみなら同 `:31`」を根拠にしています。
4. **MULTI_ENGINE_DESIGN の §4 には書いていません**: キーフレームの規則は記述子（マニフェストの JSON）には無く、アダプタの規則表にあるためです。§5.2・§5.6 (4)・§7・§9.2 に書きました。§5.6／§5.7 には H3 の非対応の一覧が元から無かったので、変えていません。
5. **仕様書 §6.2 の凍結制約 7 の「H3 は最短の 22 フレームでも 10 枚を置ける」**: 置ける位置が 0〜21 の 22 個あることからの算術です。
6. **MINIMAX_H3_DESIGN §3 の枚数の上限**: 「API の検査で、値は仕様書を見よ」と参照にしました。一方で、冒頭・裁定 3 の更新行・§8 には、指示どおり「最大 10 枚」と書いています。
7. **研究ノート §10 の裁定 61**: 「後半を設計正本へ移管（了承の取り方は本節）」という分け方にしました。
