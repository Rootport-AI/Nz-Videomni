# R5b 棚卸し: MiniMax H3 第 2 弾のうち操作パネル側（webui と aux2）

調査日 2026-10-17。読むだけ（編集・テスト・ビルド・GPU 使用なし）。ブランチ dev（`ca6f047`）。

- 表記の約束: `FE/` は `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/`、`src/` は `FE/webui/src/` を指す。行番号は調査時点のもの。
- R3（`Docs/Outputs-archive/h3-stage1-reports-2026-10-17/R3_inventory.md`）に書いてある事実は繰り返さない。R3 と食い違った点は ③ にまとめた。
- 「事実」はコードを読んで確かめたこと、「推測」はそこから考えたことで、文中で分けて書く。

---

## ① 結論の要約

**A. 3 本のスライダー**
- CFG スライダーのモックは `src/modes/single/CommonGenerationFields.tsx:544` の `ReservedFields`（状態なし・送信なし・`disabled` 固定・値 1.0 の表示だけ）。Single（`GenerationForm.tsx:240`）と Chained（`ChainedScreen.tsx:1014`）の 2 か所に置かれている。i18n のキーは `single.cfgScale.label` と `single.reservedNote`。
- 1 行に並ぶか 3 行になるかは CSS クラスだけで決まる。`.field` は縦並び、`.field field-inline` は横並び（`src/modes/single/SingleScreen.css:65-79`）。小数の範囲入力と数値入力の組は `FloatSliderField.tsx` として既にあるが、`.field`（縦並び）なので 3 行になる。整数用の汎用部品は無い。
- 要求への写し方は「フォームのフック → `toGenerateRequest`（`useGenerationForm.ts:1093`）→ `SingleScreen.buildSubmission`（`:971`）→ `client.generate` → `bridge.request("backend.request")`」。系統を見て項目を出し分けた前例は無い。系統で変わるのは快適上限の値だけで、それも `engineFamily` を引数として下へ渡して表を引くだけである。

**B. 機能スコープ**
- `featureScope.ts` の `FEATURE_UI` は「機能名 → 閉じる UI 部品（`UiTarget`、13 種）」の表。閉じ方は部品ごとに決まっていて、タブとサブタブと Chained のパネルは**グレーアウト**（`disabled`）、Settings の 2 行は**非表示**。1 つの部品に「隠す／灰色」を選ばせる仕組みは無い。
- オーナー裁定の項目のうち、今の `UiTarget` で表せるのは Chained タブ全体・Inpainting・Outpainting・Alpha Gen・Retake・Chained の V2V／A2V／参照動画パネル・バッチ A2V だけで、どれもグレーアウトになる。**Alpha Gen は今は「灰色」しかできない**（裁定は非表示）。Single の IC-LoRA・A2V・中間キーフレーム、Style LoRA、NAG／VSF、Chained の Stage-2 窓には `UiTarget` が無い。
- Gradio の `gradio_ui/feature_scope.py` は同じ機能名を使うが、Settings 相当の 2 行（`prune_vaed`・`keep_resident_embeddings`）しか持たない。

**C. 快適上限マーカー**
- `resolveComfortRow`（`src/shell/comfortTable.ts:169`）は、`engineFamily` が `"h3"` で表にその系統が無いと `null` を返す。すると Single は `spill_free_frames`（LTX の実測表）へ、Chained と Retake は `chain_comfort_token_budget` へ落ちて線を描く。**マーカー以外にも、右クリックの事前入力でフレーム数を快適上限に合わせる処理が 3 か所あり、同じ落ち方で LTX の値と 8n+1 の格子を使う**。
- 「出さない」を表すのに触る場所は、Single の `spillThresholdFrames`、Chained の `chainWindowMarkers`・`chainWindowOverBudget`・`stage2WindowLabel`、右クリックの 3 か所（`AppShell.tsx`・`SingleScreen.tsx`・`timeline/prefillSeed.ts`）。Settings の快適上限表は、系統の対応表に `h3` を載せなければ今のままで出ない（`comfortDisplayTable.ts`）。

**D. フレーム規約・解像度・fps**
- R3 の「37 ファイル」は**テストを除いたファイル数**で、`8n+1` という文字列はそのうち 35 ファイルではコメントの中にしか無い。実際の計算は `% 8`・`* 8 + 1`・`step={8}`・`VIDEO_TIME_FACTOR` などの形で約 20 ファイルに散っている（② D-1 の表）。H3 の Single で系統を知る必要がある計算の中心は `paramUtils.ts` の 2 関数・`DurationField` の `step`・`SizeFields` の `step`・`GENERAL_MULTIPLE`・`MIN_NUM_FRAMES` の 5 系列。
- 要求の `frame_rate` はフォームの値で、その初期値はサーバー既定（24）・プロジェクトの fps・素材の fps のどれか（Settings の事前入力方針で決まる）。仮オブジェクト（リボン）の長さは aux2 が `round(numFrames × projectFps ÷ genFps)` で計算し、`genFps` はフォームの fps がそのまま渡る。aux2 の C++ には 8n+1 の関数があるが、**本番の経路からは呼ばれていない**（テストからだけ）。
- **aux2 の C++ に手を入れる必要は見当たらない**。ただし webui は aux2 に埋め込まれるので、webui を変えれば `build.ps1` での再ビルドと `deploy.ps1` での配置は必ず要る。

**E. ベースモデルの表示と Settings**
- ヘッダーのドロップダウンは `GET /models` の `base_models[]` の `display_name` を並べるだけ。`transformer_weight_class` は表示には使わず、快適上限の行を選ぶのにだけ使う。Settings のモデル選択欄は `category_order` に従うが、カテゴリ名は 4 語（`transformer`・`text_encoder`・`video_vae`・`audio`）の型で閉じていて、知らない名前は**黙って捨てる**。`assets`（tokenizer のフォルダ）は応答の型に無く、画面には出ない。
- 「モデル骨格の常駐」トグルは `keepResident`（要求では `keep_resident`。送るのは ON のときだけ）。先読み block swap が実際に効いているときだけ押せる。説明文は LTX 2.3／2.5 のメモリ量を名指しで書いている。

**F. モックとテストとデプロイ**
- モックで系統を足すときに触るのは `MOCK_BASE_MODELS`（`mockBridge.ts:315`）・`MOCK_UNSUPPORTED_FEATURES`（`:355`）・`isDefaultFilePresent`（`:1147`。LTX23 以外は全部 `ltx25Install` の設定に従う）・`MOCK_CONFIG_BODY.limits.comfort_budgets`（`:142`）と `defaultConfig.ts` の写し・`handleGenerate` の検証（`:1580-1590`）。
- 関係するテストは、8n+1 が 22 ファイル、featureScope／機能名が 12 ファイル、快適上限が 23 ファイル、Settings／高速化が 18 ファイル（重なりあり。一覧は ② F-2）。型の検査は `npm run typecheck`（中身は `tsc -b --noEmit`）、テストは `npx vitest run --exclude "**/backend.integration.test.ts"`。
- デプロイ先は 2 つ。実機の `D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin\NzVideomni\NzVideomni.aux2` と、配布用のコピー `AviUtl2-Plugin\NzVideomni.aux2`（`FE/scripts/deploy.ps1:66-83`）。

---

## ② 詳細

### A. 3 本のスライダー

#### A-1. CFG スライダーのモック

事実:
- 部品: `src/modes/single/CommonGenerationFields.tsx:544-563` `ReservedFields()`。引数なし。描くのは `<label className="field field-inline field-reserved">` の中に、項目名・`<input type="range" min={1} max={1} value={1} disabled>`・数値の表示 `<span className="field-hint">1.0</span>` の 3 つ。
- 「モック」と分かる理由: 状態を持たず、どのフックにも値が無い（同ファイル `:530-542` の説明「never enabled, read, or included in any request body」「task brief M7b: 値はどこにも送らない」）。`disabled` が固定で、ツールチップは `single.reservedNote`。見た目は `.field-reserved`（`SingleScreen.css:475-482`。`opacity: 0.55`・`cursor: not-allowed`）。
- 置き場所: Single（作る）タブは `src/modes/single/GenerationForm.tsx:240`（`FrameRateSeedFields` の直後、`KeyframesPanel` の直前）。Chained タブは `src/modes/chained/ChainedScreen.tsx:1014`（`FrameRateSeedFields` の直後、`SourceInputPanel` の直前）。Edit タブには無い。
- i18n: `src/i18n/strings.ts` の英語 `:1502-1505`、日本語 `:3180-3183`。キーは `single.cfgScale.label`（"CFG scale"／「CFGスケール」）と `single.reservedNote`。
- CSS の `.field-reserved` は NAG の「Other」ラジオ（`shell/NagAccordion.tsx`）でも使っている（`SingleScreen.css:465-474` の説明）。

#### A-2. スライダー部品の作りと 1 行／3 行の違い

事実:
- `.field` は `display:flex; flex-direction:column; gap:4px`、`.field-inline` は `flex-direction:row; align-items:center; gap:8px`、`.field-inline .field-label` は `min-width:80px`（`src/modes/single/SingleScreen.css:65-79`）。420px 未満では `.field-inline` に `flex-wrap: wrap` が付く（`:46-53`）。
- 「幅」が 3 行になるのは `SizeFields`（`CommonGenerationFields.tsx:155-238`）が `<label className="field">`（縦並び）の中に項目名・`SizeSlider`・`<input type="number">` を順に置くから。さらに幅と高さは `.size-field-stack`（`SingleScreen.css:90-94`。縦並び）に入っている。
- 1 行の例は `ReservedFields`（CFG）と `FrameRateSeedFields` の fps 欄（`CommonGenerationFields.tsx:478-488`。`field field-inline`。ただしこちらは数値入力だけでスライダーは無い）。
- 小数の部品: `src/modes/single/FloatSliderField.tsx:24-54` `FloatSliderField({label, value, min, max, step, disabled, onChange})`。範囲入力と数値入力の両方を同じ処理（数値化 → 有限でなければ無視 → `[min,max]` に収める）に通す。`step` は引数なので 0.1 も 1 も渡せる。**ただし根要素が `<label className="field">`（縦並び）なので、このままでは 3 行になる**。使っているのは NAG の 4 つだけ（`shell/NagAccordion.tsx:132-182`）。NAG 側は `.nag-scale-row .field { flex: 1 }` と `.nag-accordion input[type="range"] { max-width: 320px }`（`src/shell/AppShell.css:137-153`）で見た目を整えている。
- 整数の汎用スライダー部品は無い。整数の範囲入力は `DurationField`（`step={8}` 固定）・`SizeSlider`（`step={64}` 固定）・Chained の重なりフレーム（`ChainedScreen.tsx:1105-1131`。部品化されていない生の `<input type="range">`）だけ。

推測:
- `FloatSliderField` に「横並び」を選べる引数を足すか、`field-inline` 版を別に作るかのどちらかで 3 本とも賄えそう（整数のステップ数も `step=1` で通る。ただし `Math.round` などの整数化は `FloatSliderField` には無い）。

#### A-3. 設定値が要求へ写る経路と、系統で出し分ける前例

事実:
- Single: `useGenerationForm.ts` の状態（`values` など）→ `toGenerateRequest`（`:1093-1142`）が `GenerateRequest` を組む（`nagRequestFields(nag)`・`accelerationRequestFields(acceleration)` を展開する形。どちらも「既定値のときは鍵を出さない」約束）→ `SingleScreen.tsx:971-980` `buildSubmission` がキーフレームを `conditioning_images` として足す → `useGeneration.ts:143` `submit` → `api/client.ts:193` `generate`（`POST /generate`）→ `call` が `nativeBridge.request("backend.request", {method, path, body})`（`:161-188`）。A2V が付いていると `buildA2vRequest` 経由で `POST /generate/chain` に変わる。
- Chained: `useChainForm.ts` の `buildRequest`（`ChainedScreen.tsx:860` で呼ぶ）→ `generateChain`。
- 連打（Repeat count）の同一性判定 `submissionFingerprint(buildSubmission())`（`SingleScreen.tsx:981`）は要求本体をそのまま使うので、項目を足せば自動で含まれる。
- 型の側の禁止規則: `src/api/types.ts:7-11` が「`guidance_scale`・`num_inference_steps`・`pipeline` は `GenerateRequest` から意図的に外す。WebUI はこれらの操作を決して出してはならない」と書いている。根拠は `FE/Docs/API_REFERENCE.md:728-737` §7「UIに出してはいけない機能」（`num_inference_steps` は「8固定。引き続き禁止」）と、その元の `FE/Mock/AVIUTL2_DESIGN_BRIEF.md` §5。**裁定 22 はこの規則と食い違う**ので、文書の更新が要る（④ に記載）。
- バッチ A2V の要求組み立て（`src/modes/batch/buildA2vChainPayload.ts:175-176・253-254`）は `num_inference_steps: 8`・`guidance_scale: 1.0` を型と値で固定している。H3 ではバッチ A2V は閉じる見込みなので直接の影響は無い（推測）。
- 系統で出し分ける前例: **無い**。系統（`baseModels.activeEngineFamily`）は `AppShell.tsx:1611-1643・1680` で Single／Chained／Edit／Settings に `engineFamily` として渡っているが、使い道は快適上限の表を引くことだけ（`resolveComfortRow`・`resolveOutpaintComfortBudget`・`comfortDisplayTableFor`）。LTX 2.3 と 2.5 で違う項目は、機能名を通した表示の切り替え（Settings の 2 行の非表示・Alpha Gen の灰色）だけで、系統 id を直接見る場所は無い（`featureScope.ts:318-321` が「base-model id ではなく機能名で読む」と明記）。

### B. 機能スコープ

#### B-1. 仕組みの全体

事実:
- 流れ: `GET /models` → `src/shell/useBaseModels.ts:216-230` `refresh` が `base_models[]` を `toOption`（`:166-185`）で `unsupportedFeatures: block.unsupported_features ?? []` などに写す → `:276-279` で読み込み中のベースモデルの `unsupportedFeatures` を取り出す → `AppShell.tsx` が `featureScope.ts` の関数に渡して真偽値にする → 各画面へ真偽値として渡す。
- 表: `src/shell/featureScope.ts:109-124` `FEATURE_UI`。11 個の機能名。
- `UiTarget`（`:31-44`）は 13 種。`mode.edit` だけは機能名から直接は閉じず、Edit の 4 つのサブタブが全部閉じたときに閉じる（`CONTAINER_TARGETS` `:153-163`）。

| `UiTarget` | どの機能名が閉じるか | 受け取る画面と閉じ方 |
|---|---|---|
| `mode.chained` | `chain` | `ModeTabs.tsx:45-62` が `<button disabled>`＋ツールチップ `modes.unsupportedByBaseModel`（**灰色**）。開いていれば Single へ戻す（`AppShell.tsx:1442-1446`）。右クリックで来ても「このベースモデルでは使えない」の注記を出して止める（`AppShell.tsx:647-650`） |
| `mode.edit` | （サブタブ 4 つが全部閉じたとき） | 同上 |
| `chainPanel.v2v` | `v2v` | `ChainedScreen.tsx:1025-1034`。注意書きを出し、`SourceInputPanel` の**動画の半分だけ**を閉じる（先頭画像は残す） |
| `chainPanel.a2v` | `a2v` | `ChainedScreen.tsx:1084-1092`。注意書き＋パネルを `disabled`（**灰色**） |
| `chainPanel.endSource` | `end_source` | `ChainedScreen.tsx:1047-1055`。同上 |
| `chainPanel.reference` | `reference_video` | `ChainedScreen.tsx:1069-1077`。同上 |
| `editSubTab.retake` | `retake` | `EditSubTabs.tsx:51-110`。`<button disabled>`＋理由のツールチップ（**灰色**） |
| `editSubTab.outpainting` | `outpaint` | 同上 |
| `editSubTab.inpainting` | `inpaint` | 同上 |
| `editSubTab.alphaGen` | `alpha_gen` | 同上。加えて右クリックの Alpha Gen を注記で止める（`AppShell.tsx:665-675`） |
| `createSection.batchA2v` | `chain`・`a2v` | `SingleScreen.tsx` → `BatchSection.tsx:117・266・272`。注意書き＋開始ボタンを `disabled`（**灰色**） |
| `settingsRow.vae` | `prune_vaed` | `SettingsPanel.tsx:545-572`。**行ごと非表示**。保存値も既定へ戻す（`resets: "vaeMode"`） |
| `settingsRow.keepResidentEmbeddings` | `keep_resident_embeddings` | `SettingsPanel.tsx:498-530`。**行ごと非表示**＋既定へ戻す |

- 「灰色か非表示か」は表の外、各画面が決める（`featureScope.ts:23-26` の説明「What this module deliberately does NOT decide: how the closing LOOKS」）。つまり**同じ `UiTarget` で灰色と非表示を切り替える仕組みは無い**。
- `FEATURE_UI` に載っていない機能名は無視される（`:126-136`。`two_stage_hq`・`nag`・`loras`・`sage_attention` などは「閉じる部品が無い」語として明記）。
- 保存値の書き戻し: `accelerationResetsFor`（`:190-199`）→ `AppShell.tsx:1475-1483` の副作用で `resetToServerDefaults`。

#### B-2. オーナー裁定の各項目への当てはめ

事実（今の仕組みで表せるか）:

| 項目 | 裁定 | 今の `UiTarget` | 今できる閉じ方 | 画面上の場所 |
|---|---|---|---|---|
| IC-LoRA（Single のカード） | 非表示 | **無し** | どちらもできない | `GenerationForm.tsx:250-262`（参照動画のアコーディオン。常に描画） |
| IC-LoRA（Chained の参照動画・長尺） | 非表示 | `chainPanel.reference` | 灰色だけ | `ChainedScreen.tsx:1066-1079` |
| Edit の Alpha Gen | 非表示 | `editSubTab.alphaGen` | 灰色だけ | `EditSubTabs.tsx:76-80` |
| Chained の Stage-2 窓（clip length） | 非表示 | **無し** | どちらもできない | `ChainedScreen.tsx:1157-1176`（`stage2Window` の選択欄）。同じ段の「重なり」「chunked upsample」（`:1100-1146`）も同様に無し |
| 中間キーフレーム（2 枚目以降・`frame_idx > 0`） | 灰色 | **無し** | どちらもできない | `KeyframesPanel`（`GenerationForm.tsx:242-249`）。枚数の上限は `useKeyframes(config, deps, { maxItemsOverride })`（`useKeyframes.ts:136-148`）で絞れる。Chained は既に `maxItemsOverride: 1` を使う（`useChainForm.ts:1458`） |
| Chained タブ全体 | 灰色 | `mode.chained` | 灰色 | `ModeTabs.tsx` |
| Inpainting | 灰色 | `editSubTab.inpainting` | 灰色 | `EditSubTabs.tsx` |
| Outpainting | 灰色 | `editSubTab.outpainting` | 灰色 | 同上 |
| Style LoRA | （裁定の文面に明示なし） | **無し**（`loras` は表に無い語） | どちらもできない | プロンプト欄の `<lora:…>` 札（`shell/PromptBar.tsx:55` の `LoraChips`）・在庫タブの LoRA 一覧 |
| A2V（Single） | （同上） | **無し** | どちらもできない | `GenerationForm.tsx:264-270`（常に描画） |
| A2V（Chained・バッチ） | （同上） | `chainPanel.a2v`・`createSection.batchA2v` | 灰色 | 上表 |
| V2V（Chained の継続） | （同上） | `chainPanel.v2v` | 灰色（動画の半分だけ） | 上表 |
| V2V（Retake） | （同上） | `editSubTab.retake` | 灰色 | 上表 |
| V2V（Join） | （同上） | 無し | — | ジョブカードの Join は V2V のジョブにだけ出る（`src/jobs/joinUtils.ts:14` `isV2VJob`）。H3 で V2V のジョブが作れなければ出ない（推測） |
| NAG／VSF | （同上） | **無し**（`nag` は表に無い語） | どちらもできない | 全タブ共通のアコーディオン（`AppShell.tsx:1548` `NagAccordion`） |
| 参照動画 | （同上） | Single は無し・Chained は `chainPanel.reference` | Chained のみ灰色 | 上表 |
| 連結（Chained） | 灰色 | `mode.chained` | 灰色 | 上表。バッチ i2v-long は Chained 画面の中（`ChainedScreen.tsx:1220`）なので一緒に閉じる |

- 注意（事実）: Edit の 4 つのサブタブ（Retake・Outpainting・Inpainting・Alpha Gen）が全部閉じると、`CONTAINER_TARGETS` により **Edit タブそのものが灰色になる**（`featureScope.ts:153-163・180-182`）。H3 で Retake も閉じれば、Edit タブは丸ごと灰色になる。
- 注意（事実）: 右クリックの経路のうち Single 行きのもの（`reference-video`・`video-audio-to-video`・`audio-to-video`・`add-image-keyframe`・`add-current-frame-keyframe`。`src/timeline/menuRouting.ts:206・221・245・277・358`）は、`disabledModes` に Single が載らないので今は止まらない（`AppShell.tsx:640-650` の説明「Single is never in disabledModes」）。

「系統を見て隠す」を足す場合の最小の場所（場所の特定だけ。設計はしない）:
- 判断を集める場所の候補は `src/shell/featureScope.ts` 1 か所（「サーバーの機能名を知ってよいのはこのファイルだけ」という規則が `:10-13` にある）。`UiTarget` の追加と、新しい写像関数（既存の `chainPanelsDisabledFor` などと同じ形）を足す形になる。
- 真偽値を受け取る口: `AppShell.tsx:1457-1497`（既存の `settingsRowsHidden`・`chainPanels`・`editSubTabs` を計算している箇所）と、そこから `SingleScreen`（`:1599-1619`）・`ChainedScreen`（`:1621-1650`）・`EditScreen`（`:1661-1690`）へ渡す属性。Single には今、機能スコープの真偽値の受け口が `batchUnavailable` 1 つしか無い（`SingleScreen.tsx:104・177`）。
- Alpha Gen を「非表示」にするなら `EditSubTabs.tsx:61-81` の `tabs` 配列の組み方（今は 4 つ固定で `disabled` を付けるだけ）が場所。
- 系統 id そのものを見る場合の材料は `baseModels.activeEngineFamily`（`useBaseModels.ts:280-283`）。ただし既存の方針は「機能名で読む」（`featureScope.ts:318-321`）。

#### B-3. Gradio との対応

事実:
- `gradio_ui/feature_scope.py:31-34` `FEATURE_UI` は `prune_vaed → accel_vae`、`keep_resident_embeddings → accel_keep_resident_embeddings` の 2 行だけ。`RESET_VALUES`（`:41-44`）・`GATED_CONTROLS`（`:49-50`）・`hidden_controls`（`:53-62`）。閉じ方は「非表示＋既定へ書き戻し」だけ。呼び出しは `gradio_ui/ui.py:61・2434`。
- 機能名は WebUI と同じ語（同ファイル `:4` が「Mirror of … featureScope.ts」と明記）。ただし WebUI の 11 語のうち 9 語（`chain`・`retake`・`outpaint`・`inpaint`・`alpha_gen`・`v2v`・`a2v`・`end_source`・`reference_video`）は Gradio 側に行が無い。
- Gradio の快適上限は R3 の記述どおり `gradio_ui/comfort.py:109-113` でスカラ値へ落ちる（今回は再確認していない）。

### C. 快適上限マーカー

事実:
- 判定の芯: `src/shell/comfortTable.ts:169-220` `resolveComfortRow(limits, engineFamily, acceleration, sageAvailable, weightClass)`。
  - 表が無い、または `engineFamily` が `undefined`／`""` → 互換用の行（`:183-193`。LTX の 2 つのスカラ値と `temporalFactor = 8`）。
  - 表に系統が無い（`h3` で `comfort` を持たない場合）→ **`null`**（`:195-196`）。
  - 系統はあるが一致する行が無い → `null`（`:219`）。
- 換算: `comfortFramesForBudget`（`:258-272`）は `temporalFactor × (n − 1) + 1` で画素フレームに戻すので、8n+1 型（一般には `tf·n+1` 型）の格子しか作れない。17n+5 は表せない（事実。式の形から）。
- `null` のときの落ち先と、「出さない」を表すのに触る場所:

| 画面・処理 | `null` のときの今の振る舞い | 場所 |
|---|---|---|
| Single のマーカーと警告 | `spillThresholdFrames = smartComfortFrames ?? resolveSpillFreeFrames(config.limits.spill_free_frames, …)`。LTX の実測表から線を引く | `src/modes/single/useGenerationForm.ts:702-723`。描画は `DurationField` の `spillThresholdFrames != null` のときだけ `<datalist>`（`CommonGenerationFields.tsx:405-437`）、警告は `GenerationForm.tsx:212-224` |
| Single の A2V 自動調整 | 同じ `spillThresholdFrames` で尺を頭打ちにする | `useGenerationForm.ts:958-966` |
| Chained のマーカー・超過警告・Stage-2 窓の推奨サイズ表示 | `comfortBudget = comfortRow?.chainBudget ?? resolveChainComfortBudget(config.limits.chain_comfort_token_budget)` | `src/modes/chained/useChainForm.ts:2471-2489`。`chainWindowMarkers` は `SizeFields` の `budgetMarkers`（`ChainedScreen.tsx:994`）、`chainWindowOverBudget` は `:962-964`、`stage2WindowLabel` は `:1166-1171`。型 `ChainWindowBudgetMarkers` の `comfortWidth` は常に数（`CommonGenerationFields.tsx:22-27` の `SizeBudgetMarkers` も同じ）なので、「線なし」は `budgetMarkers` 自体を渡さない形でしか表せない |
| Chained の参照動画の超過警告 | 別の予算（§1-15） | `ChainedScreen.tsx:970-972`（今回は中身を追っていない） |
| 右クリック #9／#10（カーソル位置の T2V・I2V） | `computeTargetNumFrames` が `spill_free_frames` を引いて**フォームのフレーム数を書き換える** | `src/shell/AppShell.tsx:929-961` |
| 右クリックの事前入力（Single の再マウント） | 同上 | `src/modes/single/SingleScreen.tsx:461-487`、`src/timeline/prefillSeed.ts:392-418` |
| 上の 3 か所が使う換算 | `snapNumFrames`（8n+1）で丸める。素材の長さは `floorToFrameGrid`（8n+1） | `src/timeline/deriveDuration.ts:168-201・211-212` |
| Retake（Edit） | `chainBudget ?? resolveChainComfortBudget(...)` | `src/modes/edit/useRetakeForm.ts:383-387` |
| Alpha Gen（Edit） | 行が無ければ線なし（推測。`alphaGenBudget` を読む） | `src/modes/edit/useAlphaGenForm.ts:264-283` |
| Outpainting（Edit） | `null`（線なし） | `src/shell/outpaintBudget.ts:32-36`。ただし `OutpaintingPanel.tsx:103` は `FALLBACK_APP_CONFIG.limits.spill_free_frames` を直接読む別の線を持つ |
| Inpainting（Edit） | `comfortFramesForBudget(..., 32, 8)` を直書き | `src/modes/edit/useInpaintForm.ts:416` |
| Settings の快適上限表 | 系統→表の対応 `COMFORT_TABLE_BY_ENGINE_FAMILY` に `h3` が無ければ `null` → 節ごと出ない | `src/shell/comfortDisplayTable.ts`（`comfortDisplayTableFor`。対応表は `ltx`・`ltx25` の 2 行）、`SettingsPanel.tsx:207・624-651` |

- 系統と種別（`weightClass`）は `AppShell.tsx` から各画面へ渡る（`:1611-1614`・`:1640-1643`・`:1680-1688`）。
- R3 が書いた「記述子から `comfort` を省くだけでは非表示にならない」は、Single と Chained と右クリックの 3 経路で今も成り立つ（事実）。Settings の表と Outpainting は今のままで出ない。

### D. フレーム規約・解像度・fps

#### D-1. スナップと検証の今の行と呼び出し元

事実:
- `src/modes/single/paramUtils.ts`: `snapNumFrames`（`:45-50`。`n*8+1`）・`isValidNumFrames`（`:53-55`。`(v-1)%8===0`）・`isNumFramesOnGrid`（`:180-182`）・`isValidDimension`（`:125-127`。64 の倍数）・`isDimensionOnGrid`（`:167-169`。倍数は引数）・`roundToMultiple`／`ceilToMultiple`／`floorToMultiple`（`:21-43`。倍数は引数）・`FRAME_RATE_MIN/MAX`（`:67-69`）・`FRAME_RATE_FALLBACK = 24`（`:84`）・`snapFrameRate`（`:119-122`）。
- これらを呼んでいるファイル（テスト除く）: `useGenerationForm.ts`（28 か所）・`useChainForm.ts`（21）・`timeline/deriveDuration.ts`（8）・`useRetakeForm.ts`（5）・`useAlphaGenForm.ts`（5）・`batch/useBatchForm.ts`（4）・`timeline/deriveGenerationParams.ts`（4）・`chainUtils.ts`（3）・`useInpaintForm.ts`（3）・`useOutpaintForm.ts`（2）・`SingleScreen.tsx`（2）・`comfortTable.ts`（2）・`useDurationShrinkGuard.ts`（1）。

「8n+1」が書かれたテスト以外の 37 ファイルの役割（`8n+1` の文字列が実際にどこにあるか）:

| 役割 | ファイル |
|---|---|
| 文字列がコードにある（検証の文言） | `bridge/mockBridge.ts`（`:1589`・`:1651` の 422 の文面） |
| JSX 内のコメントだけ | `modes/edit/InpaintingPanel.tsx:141`・`modes/edit/OutpaintingPanel.tsx:180` |
| 値の注記だけ | `modes/single/estimateUtils.ts:19` |
| 説明コメントだけ（残り 33 ファイル） | `api/types.ts`・`i18n/strings.ts`（3 件ともコメント。画面の文言には「8n+1」は出ない）・`batch/manifestMerge.ts`・`batch/useBatchForm.ts`・`chained/ClipCard.tsx`・`chained/audioFit.ts`・`chained/chainUtils.ts`・`chained/useChainForm.ts`・`edit/EditScreen.tsx`・`edit/RangeBand.tsx`・`edit/RetakePanel.tsx`・`edit/inpaintWindow.ts`・`edit/outpaintGeometry.ts`・`edit/useAlphaGenForm.ts`・`edit/useInpaintForm.ts`・`edit/useOutpaintForm.ts`・`edit/useRetakeForm.ts`・`single/CommonGenerationFields.tsx`・`single/KeyframeCard.tsx`・`single/KeyframeTimeline.tsx`・`single/SingleScreen.tsx`・`single/defaultConfig.ts`・`single/keyframeGrid.ts`・`single/keyframeUtils.ts`・`single/paramUtils.ts`・`single/useGenerationForm.ts`・`shell/AppShell.tsx`・`shell/comfortTable.ts`・`shell/tokenBudget.ts`・`timeline/deriveDuration.ts`・`timeline/deriveGenerationParams.ts`・`timeline/prefillSeed.ts`・`timeline/retakeWindow.ts` |

実際の時間方向の計算（`8` の直書き）がある場所（テスト除く）:

| 役割 | 場所 | H3 の Single で系統を知る必要 |
|---|---|---|
| スナップ | `paramUtils.ts:45-50` `snapNumFrames` | **要る**（フレーム数欄・プリセット・初期値・右クリックがここを通る） |
| 検証（送信前の門） | `paramUtils.ts:53-55` `isValidNumFrames` → `isNumFramesOnGrid`（`useGenerationForm.ts:1024`） | **要る** |
| 刻み | `DurationField` の `step={8}`（`CommonGenerationFields.tsx:411・424`） | **要る**（つまみの刻み） |
| 下限 | `MIN_NUM_FRAMES = 9`（`single/defaultConfig.ts:158`）。`useGenerationForm.ts:508`・`AppShell.tsx`・`SingleScreen.tsx`・`comfortDisplayTable.ts` が使う | **要る**（H3 は 22） |
| 右クリックの尺の換算 | `deriveDuration.ts:211-212` `floorToFrameGrid`、`snapNumFrames` 経由 | 要る（Single 行きの右クリック #9／#10 などを H3 で通すなら） |
| 快適上限の換算 | `comfortTable.ts:270`（`tf` は引数）・`DEFAULT_TEMPORAL_FACTOR = 8`（`:62`） | マーカーを出さないなら不要（推測） |
| Chained の計算 | `chainUtils.ts:156・162・174・375・461・662・703`・`audioFit.ts:136・249-270`・`useChainForm.ts:2417`・`tokenBudget.ts:304-418`・`ChainAudioPanel.tsx:208`・`SourceInputPanel.tsx:203` | 不要（Chained は閉じる。推測） |
| Edit の計算 | `outpaintGeometry.ts:178・185・324`・`OutpaintingPanel.tsx:188・216`・`inpaintWindow.ts:33`・`retakeWindow.ts:60`・`useInpaintForm.ts:416` | 不要（Edit の各機能は閉じる。推測） |
| キーフレームの格子 | `keyframeUtils.ts:51-54` `snapFrameIdx`。倍数とずれはサーバーの `/config`（`conditioning_frame_idx_multiple`・`conditioning_keyframe_grid_offset`。`SingleScreen.tsx:924-925`）から来る | 先頭 1 枚だけなら不要（推測） |
| モック | `mockBridge.ts:1588・1650` | 要る（モックで H3 を試すなら） |

- 補足（事実）: サーバーから来る `config.generation_presets`（プリセットの選択欄）は `snapNumFrames` を通して入る（`useGenerationForm.ts:804`）。`/config` は系統によらず 1 つなので、プリセットの値は LTX 向けのまま。
- 補足（事実）: 17n+5 の上端について、`max_num_frames` の 481 は 17×28＋5 で 17n+5 の格子に乗っている。下端は H3 が 22、今の `MIN_NUM_FRAMES` は 9。

#### D-2. fps の扱い

事実:
- 要求の `frame_rate` はフォームの値（`useGenerationForm.ts:1121` `frame_rate: values.frameRate`）。初期値は `snapFrameRate(initial.frameRate ?? config.generation_defaults.frame_rate) ?? FRAME_RATE_FALLBACK`（`:681`）。`initial.frameRate` は右クリックの事前入力が決め、その元は Settings の fps 方針（`defaults`＝サーバー既定・`project`＝プロジェクトの fps・`material`＝素材の fps。`timeline/prefillSeed.ts:344-361`）。利用者は fps 欄（`CommonGenerationFields.tsx:478-488`。`min=1`・`max=60`）で自由に変えられる。
- 仮オブジェクト（リボン）の長さ: webui が `timeline.insertProvisional` などに `numFrames` と `genFps` を渡し（`src/bridge/types.ts:415-434`）、aux2 が `ProjectFramesForPixels(num_frames, gen_fps, project_fps)` ＝ `floor(num_frames × project_fps ÷ gen_fps + 0.5)` で長さにする（`FE/native/src/timeline_math.cpp:96-102`、呼び出しは `FE/native/src/bridge_core.cpp:327-329・488-490`）。`genFps` に渡るのはフォームの fps（`SingleScreen.tsx:865`・`ChainedScreen.tsx:521・845・852`・`AppShell.tsx:1198-1221` など）。式は fps の値によらない。
- 生成した mp4 をタイムラインへ置く処理: `bridge.cpp:71-108` が素材の実時間（`total_time`）× プロジェクト fps で長さを決める（`ProjectFramesForSeconds`）。生成時の fps や 8n+1 には依存しない。
- aux2 の C++ にある 8n+1 の前提: `timeline_math.h:25-30`（`kVideoTimeFactor = 8`・`kMinNumFrames = 9`・`kMaxNumFrames = 481`）と `timeline_math.cpp` の `IsValidNumFrames`・`NearestValidNumFrames`・`ComputeSingleClip` など。**これらの関数を本番のコードから呼んでいる箇所は無い**（`grep` の結果、`src/` で使われているのは `ProjectFramesForPixels`・`ProjectFramesForSeconds`・`SpansOverlap` だけ。8n+1 系は `native/tests/test_timeline_math.cpp` からだけ呼ばれる）。
- 固定の生成 fps: `src/timeline/menuSelection.ts:154` `GEN_FPS = 24`。右クリックで素材の最短長を判断するのに使う（V2V・IC-LoRA・末尾素材）。H3 の 24fps と一致。

推測:
- 24fps 固定にしても、リボンの長さの計算と mp4 の配置には影響しない。影響が出るのは「フォームの fps が 24 以外のまま H3 に送られる」場合で、これは webui 側の fps 欄と事前入力方針の扱いで決まる。
- aux2 の C++ を変える必要は見当たらない。

#### D-3. 解像度の 64 の倍数

事実（テスト除く・H3 の Single に関係するもの）:
- `paramUtils.ts:125-127` `isValidDimension`（64 固定。呼び出し元は今回の検索では見つからなかった）。
- `useGenerationForm.ts:49` `GENERAL_MULTIPLE = 64`（IC-LoRA のときは `IC_LORA_MULTIPLE = 128`、`:48`）→ 送信前の門 `:1016-1017`。
- `SizeSlider` と数値入力の `step={64}`（`CommonGenerationFields.tsx:99・194・218`）。
- 右クリックの事前入力: `timeline/deriveGenerationParams.ts:90` `const multiple = isICLora ? 128 : 64`。
- モックの検証: `mockBridge.ts:1580-1586`。
- Chained・Edit・バッチの 64: `useChainForm.ts:182・1224-1225・2734-2735`・`useRetakeForm.ts:467・475・564-565`・`useAlphaGenForm.ts:32`・`batch/useBatchForm.ts:991-992`（H3 では閉じる見込み。推測）。
- 下限 `MIN_WIDTH`／`MIN_HEIGHT`（256／128。`defaultConfig.ts`）はどちらも 32 の倍数。

### E. ベースモデルの表示と Settings

#### E-1. ドロップダウンとモデル選択欄

事実:
- ヘッダー: `AppShell.tsx:1511-1517` → `src/shell/BaseModelSelect.tsx`。各行は `display_name`（未導入なら「(未導入)」などの文言を足す。`strings.toolVersion.optionPartial／optionNotInstalled`）。並びはサーバーの順のまま。Settings の「ベースモデル」欄も同じ部品（`SettingsPanel.tsx:341-349`）。
- `transformer_weight_class` は `useBaseModels.ts:183` で `weightClass` に写るが、**表示には使っていない**。使い道は快適上限の行選び（`resolveComfortRow` の第 5 引数）だけ。
- 未導入のベースモデルを選ぶと、リクエストを出さずに `install-<id>.bat` の案内になる（`useBaseModels.ts:98-100・244-246`）。H3 の案内名は `install-<記述子の id>.bat` になる（事実。id 次第）。
- Settings のモデル選択欄: `src/shell/ModelsPanel.tsx:125-136` が `models.categoryOrder` の順に 1 カテゴリ 1 つの選択欄を描く。順序は `src/shell/useModels.ts:41-49` `resolveCategoryOrder`（読み込み中のベースモデルの `category_order` → その `categories` の鍵 → 4 語の既定）。**知らないカテゴリ名は捨てる**（`known = MODEL_CATEGORIES`、`:19` の 4 語）。型 `ModelCategory`（`src/api/types.ts:1041`）も 4 語で閉じている。各欄の見出しは `strings.models.categories[category]`。選択状態の初期化も 4 語を直書き（`useModels.ts:137-140`）。
- `assets`（tokenizer のフォルダ）: `BaseModelBlock`（`src/api/types.ts:1096-1177`）に `assets` の欄が無く、webui のどこにも出てこない（`grep` で該当なし）。設計書（`Docs/MINIMAX_H3_DESIGN.md` §8「tokenizer の渡し方」）も「記述子のカテゴリではなく `assets` で渡す」としている。

#### E-2. Settings「生成の高速化」の区画

事実:
- 見出し: `SettingsPanel.tsx:364` `settings.accelerationHeading`（英 "Acceleration:"・日「生成の高速化:」、`strings.ts:2154・3500`）。並びは fused GGUF 逆量子化カーネル → Attention → 先読み block swap → **モデル骨格の常駐** → Embeddings processor の常駐（`prune_vaed` と同じく機能名で非表示になる行）→ VAE（PrunaVAED）。
- 「モデル骨格の常駐」の行: `SettingsPanel.tsx:457-481`。2 つのボタン（ON／OFF）。`keepResidentUsable = blockSwapPrefetchEffective(acceleration, prefetchAvailable)` が偽なら両方 `disabled`、表示は OFF（`:187-196`）。ツールチップ `accelKeepResidentPrefetchOffTooltip`（「先読みblock swapが有効なときだけ使えます」）。ON のときだけ説明 `accelKeepResidentNote`。
- 文言（i18n。`strings.ts` 英 `:2199-2213`・日 `:3520-3525`）:
  - 見出し「モデル骨格の常駐（ジョブ間キャッシュ）」。
  - 説明「メモリ64GB以上を推奨（**LTX 2.3では約20GB、LTX 2.5では約8GB**をメインメモリに常駐で使用します。2.5で常駐するのは文章を読み取る部分だけです）。モデルのCPU側骨格をジョブ間で保持し、2回目以降の生成の前処理を大幅に短縮します。生成結果は変わりません。」
- 導出（`src/shell/accelerationSettings.ts`）:
  - 型 `AccelerationSettings.keepResident`（`:120`）。サーバー既定 `KEEP_RESIDENT_SERVER_DEFAULT = false`（`:177`）。保存は `localStorage` の `nzvideomni.acceleration`（`:223`。読み書き `:343-400`）。
  - `keepResidentEffective`（`:458-463`）＝ `keepResident && blockSwapPrefetchEffective(...)`。`effectiveAcceleration`（`:475-482`）が畳み込んだ値を各画面へ配る（`useAccelerationSettings.ts`）。
  - 要求への写し: `accelerationRequestFields`（`:553-617`）が ON のときだけ `keep_resident: true` を出す。呼び出しは Single の `toGenerateRequest`・`buildA2vRequest`、Chained の `buildRequest`、Edit 各画面。
  - 快適上限の照合用 `effectiveAccelerationFields`（`:519-532`）にも `keep_resident` が入る。
- 先読みの可否は `GET /status` の `acceleration.block_swap_prefetch_available`（`blockSwapPrefetchAvailability` `:428-431`）。系統ごとの値ではなくサーバーの 1 つの値として読んでいる（事実。webui 側）。
- 設計書の現状: H3 第 1 弾は `keep_resident` と常駐を書いていない（`Docs/MINIMAX_H3_DESIGN.md` §12）。ワーカーの `load` は `block_swap_prefetch` と `attention_backend` を受ける（同 §8「ワーカーのメッセージ」）。

判断材料（推測を含む）:
- トグルの意味づけ（「CPU 側骨格をジョブ間で保持」「前処理の短縮」「出力は変わらない」）は系統を問わない言い方だが、説明文のメモリ量は LTX 2.3／2.5 を名指ししているので、H3 で同じトグルを使うなら文言の手当てが要る。
- 「先読みが効いているときだけ押せる」連動は LTX 2.3 のワーカーの都合（`accelerationSettings.ts:441-447` の説明「backend's own guard (`engine/worker.py`, G-C)」）。H3 に同じ連動を持たせるかはバックエンド側の決め事。
- H3 が `keep_resident` を受け付けない段階では、`UNSUPPORTED_FEATURES` に `keep_resident` を載せても、`FEATURE_UI` に `keep_resident` の行が無いので画面は変わらない（事実。表に無い語は無視）。
- 同じ区画の fused GGUF カーネル・VAE（PrunaVAED）・Embeddings processor の行は H3 では意味を持たない見込み。VAE と Embeddings は機能名を載せれば非表示になる。fused GGUF カーネルの行には機能名の受け口が無い（事実）。

### F. モックとテストとデプロイ

#### F-1. モック（`src/bridge/mockBridge.ts`・2939 行）

事実（新系統を足すときに触る場所）:
- `MOCK_BASE_MODELS`（`:315-318`）。`{ id, display_name, engine_family }` の 2 件。
- `MOCK_UNSUPPORTED_FEATURES`（`:355-` 以降）。基底モデル id ごとの機能名。
- `MOCK_LTX25_PRESENT_CATEGORIES`（`:503`）と `isDefaultFilePresent`（`:1147-1148`）。**LTX23 以外は全部 `ltx25Install` の設定に従う**ので、3 件目を足すと LTX 2.5 の導入状態と連動してしまう（事実。式から）。
- `MOCK_MODEL_CATEGORIES`（`:225`。4 語）。`handleModels`（`:1295-` 付近）が全ベースモデルに同じ 4 カテゴリを付け、`transformer_weight_class` は読み込み中のものだけ `"4bit"`（`:1310-1313`）。
- `MOCK_CONFIG_BODY.limits.comfort_budgets`（`:142-` 以降。`ltx`・`ltx25` の 2 系統）。同じ内容の写しが `src/modes/single/defaultConfig.ts:99-` にあり、一致は `mockBridge.test.ts` が見る（`:135-141` の説明）。
- 要求の検証: `handleGenerate`（`:1563`）の `width`／`height` の 64 の倍数（`:1580-1586`）と `num_frames` の 8n+1（`:1588-1590`）。Alpha Gen の検証（`:1650-1651`）。単発 `/generate` には機能名ごとの拒否の繰り返しが無い（`:396-` の説明）。Chained 側は `MOCK_CHAIN_FEATURE_FIELDS`（`:457`）と `handleGenerateChain`（`:1799`・`:1816-1822`）が読み込み中のベースモデルの機能名で 422 を返す。
- 進捗の段数 `MOCK_TOTAL_STEPS = 8`（`:513`）、要求のこだま `num_inference_steps: num(request.num_inference_steps, MOCK_TOTAL_STEPS)`（`:716`）。
- ベースモデルの読み込み: `handlePipelineLoad`（`:1369`）。知らない id は拒否（`:1386-1393`）。

#### F-2. テスト

事実（`src/` 配下。重なりあり）:
- 8n+1 に触れるテスト（22）: `App.prefill.test.tsx`・`batch/buildI2vGeneratePayload.test.ts`・`batch/manifestMerge.test.ts`・`batch/useBatchForm.test.ts`・`chained/ChainAudioPanel.test.tsx`・`chained/ChainedScreen.endSource.test.tsx`・`chained/audioFit.test.ts`・`chained/chainUtils.test.ts`・`chained/useChainForm.test.ts`・`edit/EditScreen.retake.test.tsx`・`edit/OutpaintingPanel.test.tsx`・`edit/outpaintGeometry.test.ts`・`edit/useAlphaGenForm.test.tsx`・`edit/useInpaintForm.test.tsx`・`edit/useRetakeForm.test.tsx`・`single/keyframeGrid.test.ts`・`single/paramUtils.test.ts`・`single/useGenerationForm.test.ts`・`shell/comfortTable.test.ts`・`timeline/deriveDuration.test.ts`・`timeline/prefillSeed.test.ts`・`timeline/retakeWindow.test.ts`。
- featureScope／機能名（12）: `shell/featureScope.test.ts`・`shell/AppShell.featureScope.test.tsx`・`shell/ModeTabs.test.tsx`・`shell/SettingsPanel.test.tsx`・`shell/accelerationSettings.test.ts`・`shell/keepResidentEmbeddings.paths.test.ts`・`shell/useBaseModels.test.ts`・`bridge/mockBridge.test.ts`・`App.editRoute.test.tsx`・`chained/useChainForm.test.ts`・`edit/EditScreen.alphaGen.test.tsx`・`edit/EditScreen.test.tsx`。補助に `src/test/unsupportedFeatures.ts`。
- 快適上限（23）: `shell/comfortTable.test.ts`・`shell/comfortDisplayTable.test.ts`・`shell/outpaintBudget.test.ts`・`shell/AppShell.weightClass.test.tsx`・`shell/accelerationSettings.test.ts`・`shell/useBaseModels.test.ts`・`single/useGenerationForm.test.ts`・`single/spillUtils.test.ts`・`single/GenerationForm.referenceVideo.test.tsx`・`chained/useChainForm.test.ts`・`chained/chainUtils.test.ts`・`chained/stage2Window.test.tsx`・`edit/EditScreen.alphaGen.test.tsx`・`edit/EditScreen.inpaint.test.tsx`・`edit/EditScreen.retake.test.tsx`・`edit/OutpaintingPanel.test.tsx`・`edit/useAlphaGenForm.test.tsx`・`edit/useRetakeForm.test.tsx`・`timeline/deriveDuration.test.ts`・`timeline/prefillSeed.test.ts`・`bridge/mockBridge.test.ts`・`App.prefill.test.tsx`・`App.prefillPolicy.test.tsx`。
- Settings／高速化（18）: `shell/SettingsPanel.test.tsx`・`shell/accelerationSettings.test.ts`・`shell/useAccelerationSettings.test.ts`・`shell/keepResidentEmbeddings.paths.test.ts`・`shell/featureScope.test.ts`・`shell/AppShell.featureScope.test.tsx`・`shell/comfortTable.test.ts`・`single/useGenerationForm.test.ts`・`chained/useChainForm.test.ts`・`chained/chainUtils.test.ts`・`edit/useAlphaGenForm.test.tsx`・`edit/useRetakeForm.test.tsx`・`batch/batchRunner.test.ts`・`batch/buildA2vChainPayload.test.ts`・`batch/useBatchForm.test.ts`・`timeline/prefillSeed.test.ts`・`App.prefill.test.tsx`・`App.prefillPolicy.test.tsx`。
- モックのベースモデルの中身を名指しで固定している例: `bridge/mockBridge.test.ts:969`（LTX23 の `unsupported_features`）。
- 実行方法（`FE/webui/package.json`）: 型の検査は `npm run typecheck`（＝`tsc -b --noEmit`。`npx tsc --noEmit` は偽の合格になる、という既知の注意）。テストは `npx vitest run --exclude "**/backend.integration.test.ts"`（`npm test` は `vitest run` で実バックエンドに触れるテストを含む）。lint は `oxlint`。

#### F-3. デプロイと aux2

事実:
- `FE/scripts/build.ps1`（232 行）: Ninja＋MSVC で aux2 を作る。既定で `npm run build:single` を走らせて webui を 1 ファイルに固め、aux2 に埋め込む（`-NoEmbedWebui` で外せるが、その成果物は `deploy.ps1` が拒否する）。`-RunTests` で C++ のテスト（統合テストは除外）。
- `FE/scripts/deploy.ps1`（230 行）: 埋め込み済みの aux2 だけを受け付ける。既定の配置先は実機の `D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin`（`:69`）の `NzVideomni\NzVideomni.aux2`、加えて Language ファイル 2 本を `Plugin` の兄弟の `Language\` へ。配布用のコピーは既定で `AviUtl2-Plugin\`（モノレポの中。`:80-83`）。AviUtl2 が起動中なら止まる。`-PluginDir ""` や `-DistDir ""` でそれぞれを飛ばせる。
- 第 2 弾で aux2 の C++ に手が入る見込み: D-2 のとおり、生成の fps・フレーム規約に依存する本番の C++ は見当たらないので、**見込みは低い**（推測）。ただし webui を変えたら埋め込みのために `build.ps1` → `deploy.ps1` は毎回必要（事実。埋め込み方式のため）。

---

## ③ R3 からずれていた点

| R3 の記述 | 今の事実 |
|---|---|
| `useGenerationForm.ts:699-717` が `null` のとき `spill_free_frames` へ落ちる | 該当は `:702-723`（`comfortRow` の `useMemo` が `:702`、落ち先の式が `:720-721`） |
| `comfortTable.ts` の `resolveComfortRow` は系統のプロファイルが無いと `null`（181-189 行付近） | 系統が無いときの `null` は `:195-196`。`:183-184` は「表が無い・系統が不明」のときの互換行の分岐 |
| `mockBridge.ts:316-317` に `MOCK_BASE_MODELS` | 宣言は `:315-318`（中身の 2 行が `:316-317` なので実質同じ） |
| 8n+1 の記述は 37 ファイル | **テストを除いた数**。テストを含めると 59 ファイル。37 のうち 33 は説明コメントだけで、実際の計算は `% 8`・`* 8 + 1`・`step={8}` などの形で別に数える必要がある（② D-1） |
| `outpaintBudget.ts:33-35` だけは `null`（線なし） | 関数は `:32-36`。ただし Outpainting の画面には `OutpaintingPanel.tsx:103` が `FALLBACK_APP_CONFIG.limits.spill_free_frames` を直接読む別の線がある |
| （R3 に無い）快適上限の落ち先はマーカーだけ | 右クリックの事前入力 3 か所（`AppShell.tsx:929-961`・`SingleScreen.tsx:461-487`・`prefillSeed.ts:392-418`）が同じ落ち方で**フォームのフレーム数を書き換える** |
| （R4b）`featureScope.ts:104-116` に `FEATURE_UI` | `:109-124` |

その他の R3 の行番号（`paramUtils.ts:45`・`:53`、`mockBridge.ts:1588`・`:1650`、`useChainForm.ts:2475`、`featureScope.ts` が 329 行）は一致。

---

## ④ 計画で決めなければならない分岐点（選択肢を並べるだけ）

1. **3 本のスライダーの部品**: (a) `FloatSliderField` に横並びの選択肢を足す (b) 横並びの新しい部品を作る (c) `ReservedFields` を中身ごと置き換える。整数（ステップ数）の扱いを同じ部品に入れるか別にするか。
2. **スライダーを出す範囲**: (a) H3 のときだけ出し、LTX では今の CFG モックのまま (b) H3 のときだけ出し、LTX では何も出さない (c) どの系統でも出し、LTX では灰色。Chained タブ（H3 では灰色）の `ReservedFields` をどうするか。
3. **スライダーの値の持ち主と送り方**: (a) Single のフォームのフック（`useGenerationForm`）の状態 (b) NAG や高速化と同じく `AppShell` が持つ全タブ共通の状態（`localStorage` に保存するかも含む）。要求に載せるのは「既定と違うときだけ」か「H3 のときは常に」か。LTX に送ったときの扱い（無視＋ログの表に入れるか）はバックエンド側の決め事と対になる。
4. **「UI に出してはいけない」規則との整理**: `src/api/types.ts:7-11`・`FE/Docs/API_REFERENCE.md` §7・`FE/Mock/AVIUTL2_DESIGN_BRIEF.md` §5 の「`num_inference_steps` は出さない」をどう書き換えるか（H3 に限った例外として書くか、規則自体を改めるか）。
5. **系統で隠す仕組みの置き場**: (a) `featureScope.ts` に `UiTarget` を足し、H3 のアダプタが機能名で宣言する（今の「機能名で読む」方針のまま） (b) 系統 id（`activeEngineFamily`）を見る分岐を足す (c) 両方。
6. **「非表示」と「灰色」の表し方**: (a) 部品ごとに固定（今の作法） (b) `FEATURE_UI` の行に閉じ方を持たせる (c) 「非表示」用と「灰色」用で別の機能名を立てる。
7. **Alpha Gen の非表示**: `EditSubTabs` のタブ配列から外す形にするか。外したときに `CONTAINER_TARGETS`（全部閉じたら Edit を閉じる）の数え方をどうするか。
8. **Edit タブ全体**: H3 で Retake も閉じると Edit タブ全体が灰色になる。(a) それでよい (b) Inpainting・Outpainting のサブタブを見せるために Edit を開いたまま中を灰色にする。
9. **Chained の Stage-2 窓**: Chained タブ全体を灰色にするなら中は見えない。(a) タブごと灰色で済ませる (b) タブは開けて中の項目ごとに灰色／非表示を分ける。
10. **Single の IC-LoRA と A2V のアコーディオン**: 今は `UiTarget` が無い。(a) 新しい `UiTarget` で非表示 (b) 灰色 (c) 今回は扱わず送信時の 422 に任せる。
11. **中間キーフレーム**: (a) `useKeyframes` の `maxItemsOverride: 1` を H3 で使う (b) 2 枚目以降のカードを灰色にする (c) `frame_idx` を 0 に固定する。先頭 1 枚の `frame_idx` の格子（`/config` の値は系統共通）をどうするか。
12. **Style LoRA と NAG／VSF**: H3 で (a) 非表示 (b) 灰色 (c) 何もしない（送ると 422）。プロンプトの `<lora:…>` 札は画面の部品ではなく文字列なので、隠しようが他と違う。
13. **Single 行きの右クリック**: IC-LoRA・A2V・キーフレーム追加の経路（`menuRouting.ts:206・221・245・277・358`）を H3 で (a) 注記で止める (b) 通して画面側で閉じる (c) そのまま。
14. **フレーム規約を系統で切り替える方法**: (a) `paramUtils` の関数に格子（倍数とずれ）を引数で渡す (b) 系統→格子の対応表を webui に持つ (c) サーバーが `GET /models` の記述子ごとに格子を配る (d) H3 と LTX で重なる値（73・209・345・481）だけに絞る。下限 22（`MIN_NUM_FRAMES` は 9）と `DurationField` の `step` も同じ判断に入る。
15. **解像度の倍数**: 32 にするのは H3 のときだけか。`SizeSlider` の `step={64}` を引数にするか。右クリックの `deriveGenerationParams.ts:90` も同じ判断。
16. **fps 24 固定の表し方**: (a) fps 欄を H3 のとき灰色で 24 に固定 (b) 欄を隠す (c) 送信時に 24 に書き換える。右クリックの fps 方針（`project`／`material`）が 24 以外を入れたときの扱い。
17. **快適上限マーカーの「出さない」の表し方**: (a) `resolveComfortRow` に「系統が無い＝出さない」の第 3 の答えを足す (b) 呼び出し側（Single・Chained・右クリック 3 か所）で系統を見て分岐 (c) サーバーが「この系統は快適上限を持たない」を明示する。右クリックで尺を快適上限に合わせる処理を H3 でどうするか（既定値のまま／何もしない）。
18. **Settings の常駐トグル**: (a) 今のトグルをそのまま H3 にも効かせる（説明文の手当てのみ） (b) H3 用の別の行 (c) H3 では非表示。先読みとの連動を H3 でも持つか。fused GGUF カーネル行など H3 に無関係な行の扱い。
19. **モックの H3**: (a) `MOCK_BASE_MODELS` に 3 件目を足す（`isDefaultFilePresent` を系統ごとに分ける必要あり） (b) 足さない。モックの検証（8n+1・64）を系統別にするか。
20. **Settings のモデル選択欄**: H3 の記述子のカテゴリ名が 4 語（`transformer`・`text_encoder`・`video_vae`・`audio`）に揃うか。揃わない名前を使うなら `ModelCategory` 型・`MODEL_CATEGORIES`・`useModels.ts:137-140`・i18n の見出しまで広げる必要がある。
21. **Gradio の同じ 3 本**: `gradio_ui/feature_scope.py` に同じ機能名の行を足すか、Gradio は Settings の 2 行の作法のままにするか。

---

## ⑤ 未確認事項

- H3 の記述子のカテゴリ名（4 語に揃うか）。設計書には tokenizer を `assets` で渡すことまでしか書かれていない。
- `GET /status` の `block_swap_prefetch_available`・`sage_available` が、H3 を読み込んだときにどう返るか（バックエンドの実装前のため）。
- `useAlphaGenForm.ts:264-283` で行が無いときに線が本当に消えるか（`alphaGenBudget` の扱いまで追っていない）。
- `ChainedScreen.tsx:970-972` の参照動画の超過警告が、どの予算から来ているか。
- `paramUtils.ts:125` `isValidDimension` の呼び出し元（今回の検索ではテスト以外に見つからなかったが、網羅の確認はしていない）。
- Gradio の快適上限（`gradio_ui/comfort.py`）と 3 本のスライダーを置く位置（今回は操作パネル側に絞った）。
- MCP（`mcp_server/`）の 8n+1 の記述（R3 に一覧があり、今回は再確認していない）。
- `FE/Mock/AVIUTL2_DESIGN_BRIEF.md` §5 の本文（`API_REFERENCE.md` §7 からの引用でしか読んでいない）。
- 実機の WebView2 で、1 行に並べたスライダーが狭い幅でどう折り返すか（CSS の読みだけ。`.field-inline` は 420px 未満で `wrap` になる）。
