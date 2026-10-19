# 1b フロントエンド（操作パネル）実装報告 — 台帳 §1-97 第 1 弾

対象: `Nz-Videomni/AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/webui` のみ。コミット・ビルド・配置はしていない。バックエンドと Docs は触っていない。

## 結論

計画 §3.3 と §4.4 を全部実装した。型検査はエラー 0、vitest は全件合格（166 ファイル・3317 件）、変更した 26 ファイルの lint（oxlint）は警告 0。設計を変えた箇所は無い。エスカレーションが要る設計上の疑問は無い。気づいた点を末尾の「留意」に 4 つ書いた。

## 検査の結果

- `npm run typecheck`（tsc -b --noEmit）: エラー 0
- `npx vitest run --exclude "**/backend.integration.test.ts"`: Test Files 166 passed / Tests 3317 passed / 失敗 0。H3 の画面テスト（AppShell.minimaxH3.test.tsx）は 2 回続けて回して 2 回とも 8/8 合格
- lint: このパッケージの lint は eslint ではなく oxlint（`npm run lint` = `oxlint`。eslint は入っていない）。変更した 26 ファイルに `npx oxlint <files>` を実行し、終了コード 0・出力なし（警告 0）
- 改行コードは元のまま LF（git の CRLF 警告は既存の設定によるもので、全ファイル共通）

## git diff --stat（webui/）

```
 .../webui/src/api/types.ts                         | 23 ++++---
 .../webui/src/bridge/mockBridge.h3.test.ts         | 10 ++-
 .../webui/src/bridge/mockBridge.ts                 |  5 +-
 .../webui/src/i18n/strings.ts                      |  6 --
 .../src/modes/edit/EditScreen.inpaint.test.tsx     |  2 -
 .../src/modes/edit/EditScreen.retake.test.tsx      |  2 -
 .../modes/single/GenerationForm.accordion.test.tsx |  5 +-
 .../single/GenerationForm.referenceVideo.test.tsx  |  5 +-
 .../single/GenerationForm.sourceAudio.test.tsx     |  5 +-
 .../webui/src/modes/single/GenerationForm.tsx      |  5 +-
 .../webui/src/modes/single/KeyframeCard.tsx        | 42 +++++++------
 .../webui/src/modes/single/KeyframeTimeline.tsx    |  4 +-
 .../webui/src/modes/single/KeyframesPanel.test.tsx | 28 +++++++--
 .../webui/src/modes/single/KeyframesPanel.tsx      | 23 +++----
 .../webui/src/modes/single/SingleScreen.tsx        | 35 +++++------
 .../webui/src/modes/single/defaultConfig.ts        |  2 -
 .../webui/src/modes/single/keyframeGrid.test.ts    | 57 ++++++++++++++++-
 .../webui/src/modes/single/keyframeGrid.ts         | 28 +++++----
 .../webui/src/modes/single/keyframeUtils.test.ts   | 39 +++++++++++-
 .../webui/src/modes/single/keyframeUtils.ts        | 25 +++++---
 .../src/modes/single/useDurationShrinkGuard.ts     |  4 +-
 .../webui/src/shell/AppShell.minimaxH3.test.tsx    | 73 +++++++++++++++++-----
 .../webui/src/shell/featureScope.test.ts           |  1 -
 .../webui/src/shell/featureScope.ts                | 16 ++---
 .../webui/src/shell/generationConstraints.test.ts  | 13 +++-
 .../webui/src/shell/generationConstraints.ts       |  7 +++
 26 files changed, 314 insertions(+), 151 deletions(-)
```

## 削除した記号

- `AppLimits.conditioning_frame_idx_multiple`／`AppLimits.conditioning_keyframe_grid_offset`（types.ts）と、その値の写し 4 か所（mockBridge.ts の config、defaultConfig.ts、EditScreen.inpaint.test.tsx、EditScreen.retake.test.tsx）
- `UiTarget` の `"createSection.keyframesMulti"`、`FEATURE_UI.multi_keyframes`、`SingleSectionsDisabled.keyframesMulti`、featureScope.ts の `multi_keyframes` を前提にした説明文（`keyframe_strength` は「非表示にし、strength も送らない」という説明に書き換えた）
- `KeyframesPanelProps.multiUnavailable` と、パネル下の注記 2 行（枚数の注記と強度の注記）
- 文言 `single.unavailableOnBaseModel.keyframesMulti` と `.keyframeStrength`（英語・日本語）
- `KeyframeCardProps.strengthDisabled`（`strengthHidden` に改名。真なら強度の行そのものを描かない）
- SingleScreen.tsx の `maxItemsOverride: 1` の分岐（`useKeyframes` の `maxItemsOverride` は Chained が使うので残した）
- mockBridge.ts の H3 の `unsupported_features` にあった `multi_keyframes`

## 置き換えと追加

- types.ts: `ConditioningImage.strength?: number`、`BaseModelConstraints.keyframes?: { step; offset }`。説明文を系統別の格子と強度の省略に合わせた
- generationConstraints.ts: `GenerationConstraints.keyframeGrid: FrameGrid`。LTX の予備値 `{ step: 8, offset: 1 }`。`resolveGenerationConstraints` は `raw.keyframes` が無ければ LTX の値を使う
- SingleScreen.tsx: 2 か所（追加ボタン用の ref と、パネルへ渡す値）を `(constraints ?? LTX_GENERATION_CONSTRAINTS).keyframeGrid.step/offset` に。送信本文は `sectionsUnavailable?.keyframeStrength` が真なら `omitStrength(conditioningImages)`（`/generate` の側だけ。A2V＝chain の側は H3 では `chain` で閉じているので変えていない）
- keyframeGrid.ts `validPositions`: 数え始めを `offset > 0 ? offset : multiple` に（offset 0 で 0 が二重に入る不具合の修正）。空の判定も同じ「最初の 0 でない格子点」で行う。(8,1) では今と同じ。冒頭の説明を「系統の規則表から届く格子」に
- keyframeUtils.ts: `snapFrameIdx` を `floor((f − offset)/multiple)*multiple + offset` に。純関数 `omitStrength(images)` を追加（分割代入で鍵そのものを外す。undefined は入れない）
- KeyframesPanel.tsx: `strengthHidden={strengthUnavailable}` を渡す
- mockBridge.ts: `MOCK_CONSTRAINTS_BY_FAMILY` に LTX 8/1、H3 1/0 の `keyframes`
- 必須の部品の値を渡す 4 つのテスト（GenerationForm.accordion／referenceVideo／sourceAudio、KeyframesPanel）は `LTX_GENERATION_CONSTRAINTS.keyframeGrid.step/offset` に置き換えた
- `config.limits.*` を引いていた古いコメントの現行化: GenerationForm.tsx、KeyframeCard.tsx、KeyframesPanel.tsx、KeyframeTimeline.tsx、useDurationShrinkGuard.ts、keyframeGrid.test.ts

## 足したテスト・変えたテスト（§4.4）

- keyframeGrid.test.ts: (1,0) の describe を追加。validPositions が [0..F−1] で重複なし、max = F−1、nextAddPosition（0 → 中点 10 → 15 → 末尾が埋まると null）、stepPin、resolveLanding、findOverflowFrameIdxs。(8,1) の既存の期待値は変えていない
- keyframeUtils.test.ts: snapFrameIdx が (1,0) で恒等、(8,1) で f = 1..481 の全部が旧式と同じ値。omitStrength が鍵を外し、入力を変えない
- generationConstraints.test.ts: H3／LTX の固定値に `keyframes` を足し、解決と「鍵の無い古い応答は LTX の格子」を確かめる 1 件を追加。予備値の確認に keyframeGrid を追加。keyframesMulti を 3 か所から削除
- AppShell.minimaxH3.test.tsx: 旧「1 枚だけ・強度は灰色」を置き換え、(a) LTX の対照で強度の行がある、(b) H3 で 2 枚目が足せる・強度の行が DOM に無い・注記も無い、を追加。(c) 送信の body の `conditioning_images` に strength の鍵が無いことは、既存の H3 送信テストにキーフレーム 2 枚を足して確かめる形にした（理由は留意 3）
- featureScope.test.ts・mockBridge.h3.test.ts: 一覧から multi_keyframes を外し、mock の H3 には multi_keyframes が無いこと、両系統の `keyframes` の値を確かめる
- KeyframesPanel.test.tsx: `strengthUnavailable` で強度の行が DOM から消えること（灰色ではない）を 1 件追加

## 留意（オーナー・監督の判断材料。設計は変えていない）

1. **既存テストの期待値を 1 つ変えた**: keyframeUtils.test.ts の「別の multiple/offset」の例（4/2 の格子）で、13 の行き先が 14 → 10 に変わった。旧式は offset ではなく 1 を引いていたため、間の値が上の格子点へ繰り上がっていた。新しい式はサーバーの `(raw−offset)//step*step+offset` と同じで、計画が指定した式。実際の系統（8/1・1/0）では値は変わらない。
2. **lint は eslint ではなく oxlint**: 指示は `npx eslint` だったが、このパッケージには eslint が無く、`npm run lint` が oxlint を呼ぶ。oxlint で確かめた。
3. **送信の確認を既存テストに合流させた理由**: H3 の送信は全体で 1 つの模擬サーバー（`bridge` の singleton）を通る。そのサーバーの「選択中のベースモデル」と「実行中のジョブ」はテストの間で残るので、送信するテストを 2 つにすると後のテストが H3 のまま始まって落ちる（実際に落ちた）。生成中はベースモデルを戻せない。そこで送信の確認は既存の H3 送信テスト 1 本に足し、2 枚目と強度の行の確認は別の模擬サーバーで行う形にした。キーフレームの撮り込みは模擬サーバーの往復が 200 ミリ秒ずつかかるので、「空の注記も読み込み中の表示も無い」まで最大 5 秒待つ補助関数を置いた。
4. **H3 で 2 枚目を足したときの位置**: 「キーフレームを追加」は H3 でも今までどおり「最後の点と末尾の中点」に置く（例: 124 フレームで 0 の次は 61）。末尾（第 1 弾の主目的）に置くには、利用者が位置を打ち込むかピンを動かす。計画は「(1,0) で問題なし」としているので変えていない。末尾を既定にしたい場合は別の判断になる。
5. （参考・今の系統では起きない）offset 0 で step が 2 以上の格子が将来できた場合、画面の予告は 1..step−1 を 0 に寄せ、サーバーは first = step に寄せるので食い違う。今は H3 の step が 1 なので起きない。

## 残したこと

なし（Docs の更新は 1d の担当。build.ps1／deploy.ps1 による配置は門 1-G6 で監督が行う）。
