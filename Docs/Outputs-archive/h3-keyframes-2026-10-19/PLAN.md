# 台帳 §1-97 MiniMax H3 の末尾キーフレームと中間キーフレームへの対応 — 実装計画

作成 2026-10-19。監督＝Fable（本セッション）。実装・調査・レビューはサブエージェント（Opus 以下）。ComfyUI は論理の参考に留め、コードは写さない。敵対的レビュー（Opus・C2/M5/m13）は全採用済み（§8）。

## 0. 背景と目的

- α版（CLOSED 3-239）の H3 の I2V は、先頭フレーム 1 枚を「条件行」として系列に足す方式で、先頭以外を受け付けない。本計画で、末尾（第 1 弾）と中間（第 2 弾）のキーフレームを受け付ける。
- 一次資料で確かめた事実（報告書 `scratchpad\kf197\R1_official_lastframe_and_addguide.md`）: 公式の末尾フレームは先頭と同じ手順（VAE 符号化 1 枚・平均、0.999 の混ぜ方、AdaLN の時刻 max(t, 0.999)、種別の印＝映像）で、違いは位置符号化（RoPE）の時間座標だけ。時間座標は画素フレーム番号で決まり、潜在フレームの境界に縛られない（ComfyUI の AddGuide は 0〜F−1 のどの整数も受ける）。末尾の画像もテキストエンコーダの画像読み取り部（vision tower）に `<Picture 2>` として入る。中間フレームは ComfyUI だけの拡張で、条件行のみ（画像読み取り部に入らない）。強度の概念はどの実装にも無い。
- 公式の末尾の式（正解データ＝DiffSynth の写し `Docs/Outputs-archive/h3-gate1/oracle/diffsynth_h3/pipeline_coords.py:86-99`）: `(float(L) + temporal_span) − 5/3`。temporal_span は `np.ones(T')×5/3` に刻み 5 で (1,4,4,4,4) を掛けた float64 配列の `np.sum()`。

## 1. オーナー裁定（本計画で確定。設計正本 §2 へ 54 以降として転記する）

1. 到達点は了承後の実装まで（本セッション）。
2. 末尾を第 1 弾、中間を第 2 弾。ComfyUI は論理の参考のみ、コードは写さない。
3. キーフレームの置ける位置は、既存の系統別の規則表 `RequestRules` に 1 項目を足して系統別にする。全系統共通の config の 2 値（`conditioning_frame_idx_multiple`／`conditioning_keyframe_grid_offset`）と、系統が分かる前に走る pydantic の 8n+1 の丸めは廃止し、系統別の検査へ移す。UI の表 FEATURE_UI は使い回さない（on／off の表と数値の規則は別物）。
4. 強度は操作パネルで非表示（灰色ではない）。記録には「強度が効いていない」と分かる形にする（1.0 にしない）。仕組みは「強度を省略可にし、系統の規則表が既定を書き戻す」。LTX は 0.8 を書き戻し、H3 は書き戻さず null のまま。
5. バッチは本項で触らない。
6. 中間フレームの画像は画像読み取り部に渡さない（条件行のみ）。足りなければ「渡す」を比較の腕として後で検討。
7. 公式のプロンプト指南の指示文（「Picture 1 は 0.00 秒、Picture 2 は S.SS 秒」）は README に書くだけで、製品は入れない。
8. GPU を使う実測は弾ごとにまとめて了承を得る。T2V と先頭だけの I2V の結果は変えない（指紋で回帰を確かめる）。

## 2. 設計の要点

### 2.1 位置の規則（RequestRules の新しい項目）

`RequestRules`（`services/engines/ltx/request_rules.py:72-99`）に既定値つきの 3 項目を足す。

| 項目 | 既定 | LTX（`LTX_REQUEST_RULES` :189-200。2.3 と 2.5 が共有） | H3（`services/engines/h3/adapter.py:124-143`） |
|---|---|---|---|
| `keyframe_step` | 1 | 8 | 1 |
| `keyframe_offset` | 0 | 1 | 0 |
| `keyframe_strength_default` | None | 0.8 | None |

- 公開: `constraints()`（:103-119）に `"keyframes": {"step": s, "offset": o}` を足す。強度の既定は公開しない（`flag_defaults` と同じく画面が写し DEFAULT_STRENGTH=0.8 を持つ。使えるかどうかは機能名 `keyframe_strength` で伝わる）。
- 正規化: `RequestRules.normalize_keyframes(images, num_frames, *, where="") -> str | None` を 1 つ足す。今の `_normalize_conditioning_images`（`api/models.py:192-242`）の一般化で、文言は今のまま。
  ```text
  first = o if o > 0 else s            # 0 より大きい最小の格子点
  last  = (F − o)//s*s + o − s         # フロントの maxPlaceablePosition と同じ式
  snapped = 0 if raw == 0 else clamp((raw − o)//s*s + o, first, last)
  重複の検査（0 の重複は start frame の文言）→ frame_idx を書き戻す
  strength が None で keyframe_strength_default があれば書き戻す
  ```
  LTX (8,1) では今の `max(1, min((raw−1)//8*8+1, F−8))` と同じ値（F=9 でも first=last=1）。H3 (1,0) では 1..F−1 がそのまま通り、F 以上は F−1 に寄る。
- 呼ぶ場所: `validate_for_family`（`services/engines/__init__.py:217-238`）を `fill_defaults` → `violation` → `normalize_keyframes` の順にする（寄せる計算は F が正しいと分かってから）。違反は `request_validation_failed`。**422 の順序が 2 点変わる**: 位置の重複のエラーは FEATURE_UNSUPPORTED の後、かつ系統の数値の違反（8n+1／17n+5 など）の後になる（422 の形は同じ。仕様書に両方書く）。
- 連結（`api/models.py:1527-1529`）は LTX 専用（:1509 に 8n+1 の検査がある）なので、そこで `LTX_REQUEST_RULES.normalize_keyframes` を直接呼び、文言があれば `ValueError`。`api/models.py` は既に `request_rules` を import しており（:36）、`request_rules.py` は読み込み時に api を import しない（:29-39）ので循環は無い。これで chain の clips[0] の強度も 0.8 で埋まり、`ltx/adapter.py:2533`・`ltx25/adapter.py:1228` に None は届かない。
- 枚数の上限 `MAX_CONDITIONING_IMAGES`（10）は系統に依存しないので pydantic に残す（GenerateRequest :694-709 と chain の validator に枚数の検査だけ残す）。`_normalize_conditioning_images` は削除。
- config: `config.py:345-350` の 2 項目とコメント、`config.yaml:178-179`、`config.yaml.example:189-190` を削除。`LimitsConfig` は pydantic 既定の extra='ignore'（:342・:578 のコメント。`model_config` は無い）なので、利用者の古い config.yaml に残っていても壊れない。バックエンドで 2 キーを読むのは config.py だけ。

### 2.2 強度の省略（None ＝ 系統の既定）

- `ConditioningImage.strength: float | None = Field(None, ge=0.0, le=1.0)`（`api/models.py:188`）。
- LTX は `normalize_keyframes` が 0.8 を書き戻す → アダプタ（`ltx/adapter.py:2214`・`ltx25/adapter.py:822`）・ワーカー・metadata.json の request の写し（`pipeline_manager.py:1866`）は今と同じ数値。LTX のアダプタに None が届く経路は無い（単発＝`api/generate.py:90`、Alpha＝`generate_alpha.py:278`〔画像なし〕、連結＝pydantic の中で寄せたオブジェクトを `to_clip_request` と job_store が使い回す。模擬のバックエンドは strength を読まない）。
- H3 は None のまま → metadata.json の request は `"strength": null`。実際に使った位置は、同じ写しの `request.conditioning_images[].frame_idx`（サーバーが寄せた後の値。エンジンは位置を動かさない）に残る。**done の `images_used`（枚数）は改名しない**（§1-95 の計測台 `measure_h3_vram.py` と保管庫の台本が読むため）。
- H3 のアダプタ: `_keyframe_strength`（:157-159）を `any(ci.strength is not None ...)` に。`_STRENGTH_DEFAULT`（:104）、`_multi_keyframes`（:151-154）と REJECT_TABLE の `multi_keyframes` 行（:181）は削除（引き算）。`_refusal_detail`（:242-251）は `keyframe_strength` の分岐に置き換え、文言は「MiniMax H3 にはキーフレームの強度がありません（strength は指定しないでください）。」。機能名 `keyframe_strength` は残るので、操作パネルの FEATURE_UI（`shell/featureScope.ts:147`）は引き続き `createSection.keyframeStrength` を閉じる。描き方だけ灰色→非表示に変える。重なりの監査 `_ALLOWED_OVERLAP = {"conditioning_images"}` はそのまま成り立つ。
- `reject_unsupported`（`api/generate.py:79`）は `validate_for_family`（:90）より先に走るので、H3 の判定は書き戻し前の値（None か明示の数値）を見る。アダプタの generate 冒頭の再検査も同じ。
- 送る側: 操作パネルは強度の部品が閉じているときは strength を送らない（鍵そのものを外す）。Gradio は既定 0.8 と同じなら送らない（vram16 の前例「既定と違うときだけ送る」。LTX はサーバーが 0.8 で埋めるので結果は同じ）。MCP は `strength: float | None = None`（`model_dump` は null を送るが、`float | None` は None を受ける）。

### 2.3 エンジンの条件行の一般化（engine_h3）

- 並べる順: 画像を frame_idx の昇順に並べる（安定ソート）。条件行も画像読み取り部への入力もこの順。先頭＋末尾では公式の入力順（先頭→末尾）と同じになる。
- 画像読み取り部に渡す規則: **frame_idx が 0 か F−1 の画像だけ**を昇順で渡し、ラベルは渡す順に `<Picture 1>`、`<Picture 2>`。それ以外（中間）の画像は条件行だけ（L を増やさない）。末尾だけのジョブは `<Picture 1>`、中間だけのジョブは画像読み取り部を読まず T2V と同じ形。
- RoPE の時間座標（フレーム n の条件行。正解データと同じ書き方で）:
  - n = 0: `float(L)`（今と同じ）
  - n = F−1: `(float(L) + span) − 5/3`。span は `np.ones(T', float64) × (5/3)` に `[k::5] *= (1,4,4,4,4)[k]` を掛けてから `np.sum()`（正解データと同じ組み方・同じ順序でないと全桁一致しない）
  - それ以外: `float(L) + (5.0/3.0) × n`（ComfyUI と同じく掛け算が先）
  - テスト（4.1）で全ての F と L について `L + (5/3)×(F−1)` が正解データの式と全桁一致すると分かれば、F−1 の特別扱いを消して一本にする（引き算）。結果は §170 に書く。
- ノイズ: 今の 3 回目の取り出しで `[grid, 96]` を 1 度だけ引き、`repeat(K, 1)` で K 枚に繰り返す（DiffSynth・ComfyUI と同じ「全条件で同じノイズ」。K=1 は今と同じ値）。
- 値・時刻・印: `0.999·clean + 0.001·noise`（clean は K 枚をつなげたもの）、`max(t_video, 0.999)`、印＝映像（0）。区間全体に入れる今の仕組みのまま。
- 変えないもの: `H3SeqLayout.make`（`dit.py:113-133`）と `embed_rows`（:492-513）は cond を一塊で通す。`time_rows`／`modality_rows`（`pipeline_h3.py:198-219`）は区間全体に 1 つの値。DiT の出力の切り出しは `layout.audio/video`。`lowvram.py` は cond を参照しない。dit.py・lowvram.py はコメントの複数形化以外変えない。
- 進捗: encode の段の分母は「条件行の画像が 1 枚以上あれば 4、無ければ 3」。画像読み取り部に入る画像の有無では決めない（中間だけのジョブで分母 3 のあと 4/4 が出るのを防ぐ）。

### 2.4 全桁一致を保つもの（回帰の約束）

- T2V: 画像 0 枚。トークン列・乱数の取り出し・条件行とも今と同じ。指紋 `0da6111c…`（§165 の訂正行・§167.6 の条件＝黄色いレインコートのプロンプト・seed 42・640×384・124 フレーム）。
- 先頭だけの I2V: トークン列（`<Picture 1>`＋画像トークン N）・画像処理器に 1 枚・ノイズ `[grid,96]` 1 回・t = `float(L)` が今と同じ → 指紋 `77e29687…`（§166 2a-G4 の J1 の条件＝同じプロンプト・画像 `outputs/h3-gate2/i2v_input.png`〔git 管理外〕・先読み off・sdpa・VRAM 16GB mode off・常駐なし）。
- LTX: 寄せる式が同じ値、強度 0.8 の書き戻し、操作パネルは LTX では今までどおり strength を送る。

### 2.5 バッチ（裁定 5）

- H3 では `unsupported_features` に `chain` があるため `batchA2vDisabledFor` が真になり、バッチの枠は i2v 行も含めて丸ごと灰色（`modes/batch/BatchSection.tsx:53-66` の `unavailable`。`featureScope.test.ts:197-207` で固定済み）。Batch I2V-Long は Chained 画面の中（閉じる）、Edit のタブも全部閉じる。Gradio のバッチは `POST /generate/chain` を送り、H3 は `reject_chain` で断る。
- よってバッチのコードは変えず、テストも足さない。

## 3. 変更するファイルと中身

### 3.1 engine_h3

- `rope.py`: `build_coords(..., cond_frame_idx: Sequence[int] = ())`（`cond_frames` の置き換え）。`keyframe_t(frame_idx, latent_frames, text_len)` を 2.3 の式で追加（F は潜在フレーム数から `F = (T'−2)//5×17 + 5` で逆算。`constants.latent_frames` の逆）。`NotImplementedError`（:110-114）を削除し「0 ≤ n ≤ F−1・重複なし」の ValueError に。docstring（:94-95）の「§1-97 で扱う」を書き換え。
- `pipeline_h3.py`: `H3ImageInput` と `images` のコメント（:131-136, :150）。`validate_request`（:186-193）の「1 枚だけ・frame_idx=0」を削除し「0 ≤ frame_idx ≤ F−1・重複なし・パス非空」に。`generate`（:494-510）は昇順に並べて全枚を `load_canvas_image` で読み、画像読み取り部へは 2.3 の規則で選んだ画像だけ（`source_size` はリスト）。`_encode(prompt, phases, images=(), *, has_cond)`（分母は 2.3 の進捗の規則）。`_encode_image(images, entry)` は `H3VideoEncoder` を 1 度だけ組み 1 枚ずつ符号化して `torch.cat`（`(1,24,1)` の形の検査は 1 枚ずつ）。`_denoise`（:806-857）は K 枚・形 `(K·grid, 96)`・ノイズの繰り返し・`build_coords(cond_frame_idx=…)`・`entry["sequence"]["cond"]`。done の `images_used` は不変。
- `text_encoder.py`: `presentation(tokenizer, prompt, image_tokens: Sequence[int] = ())`（画像ごとに `PICTURE_LABEL.format(i+1)`＋画像トークン。int を受ける互換の形は消す）。`vision_inputs(images)` は処理器を 1 度呼び画像ごとのトークン数を返す。`encode(prompt, images=())`。`read_report["vision_tokens"]` は合計＋`vision_tokens_per_image`。`mm_token_type_ids`（:337-340）は不変。
- `image_cond.py`: 文言の「先頭フレーム」（:38, 45, 48）→「キーフレーム」。
- `worker.py`: `_images` の docstring。CLI に `--keyframe FRAME=PATH`（複数可）を足し、`--image` は frame 0 の別名として残す（既存の指紋取りのコマンドを変えない）。`--image` と `--keyframe 0=…` を両方渡したら重複として拒否。docstring（:64-65）の「T2V は 3 段、I2V は 4 段」を 2.3 の進捗の規則に合わせる。
- `dit.py`・`lowvram.py`: コードは不変。docstring の単数形だけ直す（`dit.py:108-119, 505-506`）。

### 3.2 API と services

- `api/models.py`: strength を省略可（:188）。`_normalize_conditioning_images` 削除（:192-242）。validator 2 か所（:704-705, :1527-1529）に枚数の検査だけ残し、chain 側は `LTX_REQUEST_RULES.normalize_keyframes`。コメント（:696-700）を現行化。
- `services/engines/ltx/request_rules.py`: 3 項目・`constraints()["keyframes"]`・`normalize_keyframes`・LTX の表に 8／1／0.8。`fill_defaults` の docstring に分担を書く。
- `services/engines/__init__.py`: `validate_for_family` の 3 段目。docstring。
- `services/engines/h3/adapter.py`: `REQUEST_RULES` に `keyframe_step=1, keyframe_offset=0`。2.2 の削除と変更。REJECT_TABLE のコメント（:162-169）を「conditioning_images は 1 行だけ重なる」に。generate の docstring（「先頭フレームの I2V」）。payload（:420-425）は不変（今も全枚と frame_idx を送る）。
- `config.py`・`config.yaml`・`config.yaml.example`: 2 キー削除。
- `api/generate.py:90` のコメントに「位置もここで寄せる」を 1 行。
- 古くなるコメントの現行化: `services/engines/ltx/adapter.py:2208`、`engine/pipeline/fast_video_pipeline.py:1587`、`engine/pipeline/chain_pipeline.py:315`、`tests/test_chain.py:343`（`_normalize_conditioning_images` への言及）。

### 3.3 フロントエンド（webui/src）

削除（引き算）:
- `api/types.ts:866-867`、`bridge/mockBridge.ts:118-119`、`modes/single/defaultConfig.ts:52-53` の limits の 2 キー。config の写しとして消すのは `modes/edit/EditScreen.inpaint.test.tsx:160-161`・`EditScreen.retake.test.tsx:130-131` の 2 ファイル。
- `shell/featureScope.ts:146` の `multi_keyframes` 行、`:45` の `createSection.keyframesMulti`、`SingleSectionsDisabled.keyframesMulti`（:367-368, 380）、`:108-111` の説明文。
- `GenerationForm.tsx:280` の `multiUnavailable={sectionsUnavailable?.keyframesMulti}`。
- `SingleScreen.tsx:530-538` の `maxItemsOverride: 1` の分岐（`useKeyframes` の `maxItemsOverride` 自体は Chained が使うので残す）。
- `KeyframesPanel.tsx:22-31, 137-138` の `multiUnavailable` と強度の注記、`i18n/strings.ts:1530-1532`／`3234-3236` の対応する文字列。`mockBridge.ts:499` の `multi_keyframes`。

置き換え:
- `api/types.ts`: `BaseModelConstraints`（:1208-1239）に `keyframes?: {step; offset}`。`ConditioningImage.strength?: number`（:46-50）。
- `shell/generationConstraints.ts`: `GenerationConstraints.keyframeGrid: FrameGrid`（既存の型）。LTX の予備値 `{step: 8, offset: 1}`。`resolveGenerationConstraints` が `raw.keyframes ?? LTX` を読む。
- `SingleScreen.tsx:761-762, 967-968`: `config.limits.*` → `(constraints ?? LTX_GENERATION_CONSTRAINTS).keyframeGrid.step/offset`（:474 と同じ書き方。`constraints` は :114・:196 で部品の値として届いている）。下流への props の流れは不変。
- テスト 4 ファイル（`GenerationForm.accordion.test.tsx:42-43`・`GenerationForm.referenceVideo.test.tsx:34-35`・`GenerationForm.sourceAudio.test.tsx:28-29`・`KeyframesPanel.test.tsx:20-21`）は必須の部品の値を渡す行なので、消さず `LTX_GENERATION_CONSTRAINTS.keyframeGrid.step/offset`（または 8／1）に置き換える。
- `modes/single/keyframeGrid.ts` `validPositions`（:37-46）: 数え始めを `offset > 0 ? offset : multiple` に（offset 0 で 0 が二重に入る不具合の修正）。冒頭の説明を「系統の規則」に。`maxPlaceablePosition`・`nextAddPosition`・`stepPin`・`resolveLanding`・`findOverflowFrameIdxs` は (1,0) で問題なし。
- `modes/single/keyframeUtils.ts`: `snapFrameIdx`（:51-54）を `floor((f − offset)/multiple)*multiple + offset` に（(1,0) で恒等。(8,1) は今と同じ値）。純関数 `omitStrength(images)` を追加（`exactOptionalPropertyTypes: true` なので分割代入で鍵そのものを外す。`undefined` を入れない）。
- `SingleScreen.tsx:1023` の送信本文: `sectionsUnavailable?.keyframeStrength` が真なら `omitStrength(conditioningImages)`。
- `KeyframeCard.tsx`: `strengthDisabled` → `strengthHidden`。真なら強度の行（:250-261）を描かない。`KeyframesPanel.tsx:106` は `strengthHidden={strengthUnavailable}`。
- `mockBridge.ts:335` の `MOCK_CONSTRAINTS_BY_FAMILY` に系統ごとの `keyframes`。

### 3.4 Gradio

- `gradio_ui/presets.py`: `KF_STRENGTH_DEFAULT = 0.8` を 1 つ置き、`ui.py:654`（`value=0.8`）・`handlers.py:1477`・`RESET_VALUES` から参照する。
- `gradio_ui/feature_scope.py:30-49`: `"keyframe_strength": ("kf_strength_1", …, "kf_strength_<KF_MAX_SLOTS>")` と `RESET_VALUES` に各 `KF_STRENGTH_DEFAULT`。
- `gradio_ui/ui.py:654-656`: スライダーを `kf_strength_i` として `_gated_components`（:2535-2604）に登録（隠す処理と値の戻しはそのまま効く。`refresh_model_dropdowns` は更新ボタン・ページの読み込み・読み込み後の再取得で呼ばれる）。
- `gradio_ui/handlers.py:897-904`: `strength == KF_STRENGTH_DEFAULT` なら strength の key を送らない。
- 文言: `gradio_ui/i18n.py:174` の `cap_kf_grid`「8n+1 に寄せる」と `ui.py:633-637` を「位置はベースモデルの規則に寄せる（LTX は 8n+1、MiniMax H3 は任意のフレーム）」の趣旨に（EN／JA）。
- Gradio の制約の写し `constraints.py` には `keyframes` を足さない（Gradio は位置を寄せず、`normalize_constraints` も読まない。足すと H3 にも LTX の格子が写り、配信値そのものと比べるテストが落ちる）。
- Gradio のバッチは `POST /generate/chain` を送るので H3 には届かない（2.5）。変えない。

### 3.5 MCP

- `mcp_server/params.py:18-31`: `strength: float | None = None`。docstring（:22-27）「省略で系統の既定（LTX 0.8）。H3 は指定すると 422」。
- `mcp_server/tools/generate.py:259, 330-333, 745-750`・`server.py:36-37, 48`: H3 の「任意のフレーム・最大 10 枚・強度なし」に文言を直す。`model_dump` は不変。

### 3.6 文書（事実は 1 か所、ほかは参照）

| 文書 | 直す中身 |
|---|---|
| `Docs/MINIMAX_H3_DESIGN.md` | §2 に裁定 54〜（本計画 §1）と、裁定 3 の末尾「α版の I2V は先頭フレームだけ」への更新行。§3 の「I2V（先頭フレーム）の条件行」をキーフレームの条件行に一般化（並べる順・時刻の式・ノイズの共有・画像読み取り部の規則・進捗の規則）。「視覚塔」→「画像読み取り部（vision tower）」。§4 の段の表（画像の符号化は K 枚）。§8 のワーカーのメッセージ（done のキーは不変）。§12 から末尾・中間を外し §13 を現行化 |
| `Docs/MULTI_ENGINE_DESIGN.md` | §4・§5.2 に RequestRules の新項目と `constraints.keyframes`。§9 の対応表 |
| `Videomni_Backend_Specification.md` | strength の省略。`constraints.keyframes`。limits の 2 キーの削除（古い操作パネルとの同時配置の注意）。寄せる場所の移動と 422 の順序 2 点（:748, :887, :2079 の 8n+1 の記述）。H3 の非対応一覧（:869）から multi_keyframes を消す。版を上げる |
| `Docs/MCP_SERVER_DESIGN.md` | strength と H3 の位置 |
| `README.md` | H3 の節にキーフレームの使い方と、プロンプト指南の指示文（裁定 7）。「できないこと」の表の 2 行（:976-977）を消すか書き換える。1〜134 行は不変（境界は 134 行目の見出し） |
| フロント `Docs/API_REFERENCE.md:156`・`DEVLOG.md` §143 | 型と画面の変更。§5.1 の「8n+1」の説明を系統別に |
| `Docs/VERIFICATION_LOG.md` §170 | 門の結果・float64 の結論・レビューの採否 |
| `Docs/MINIMAX_H3_RESEARCH_NOTES.md` §10 | 裁定 54〜（進め方の分）。§13.3 の AddGuide の記述は不変 |
| `Docs/PENDING_TASKS.md`／`PENDING_TASKS_CLOSED.md` | 最後に 3-241 としてクローズ。§1-95 に「条件行が K×grid 行増える」含意を 1 行。「次の番号」の更新行（CLOSED 3-242・§171・DEVLOG §144） |

## 4. テスト

### 4.1 エンジン（`tests/test_h3_engine.py`）
- 新規・float64 の総当たり: F = 17n+5（22..481 の 28 通り）× L = 0..8192 で、正解データの式 `(float(L) + span) − 5/3`（span の組み方を 2.3 のとおりテスト内に 3 行で写す。保管庫の import はしない）と `float(L) + (5/3)×(F−1)` の全桁一致を数える。一致・不一致どちらでも記録し、実装の `keyframe_t` が正解データの式と全桁一致することは常に確かめる。
- 新規・座標: `build_coords` で keyframes [0]・[F−1]・[0, F−1] × 画角 3 通りの t と (h,w) と区間の長さ。中間 n = 1・F//2・F−2 で t = L + (5/3)·n、複数区間がそれぞれの t を持つ。範囲外と重複は ValueError。正解データ（`torch.equal`）との照合は 1-G2 の台本で行う。
- 変更・presentation（:447-460）: 1 枚が今と同じ列／2 枚で `<Picture 1>`＋塊 1＋`<Picture 2>`＋塊 2＋プロンプト／タグと `mm_token_type_ids` の数／末尾だけで `<Picture 1>`。
- 変更・validate_request（:781-790, :974-983）: 複数枚・F−1・中間が通り、範囲外・重複・空パスが拒否される。
- 新規・ノイズ: 同じ種で K=1 の条件行が今の式と全桁一致、K=3 で各区間のノイズ部分が同じ。
- 新規・進捗: 中間だけのジョブで encode の分母が 4、T2V で 3。
- 変更・worker: `_generate_request`（:992-1002）の複数枚と `--keyframe`、`--image` との重複の拒否。:960-986・:1131・:1344 付近の先頭限定の期待値。

### 4.2 API・アダプタ・検証
- `tests/test_validation.py`: :158-348 の寄せ・重複を `LTX_REQUEST_RULES.normalize_keyframes` を直接呼ぶ形に移す（期待値は不変）。H3 の表で「任意の位置がそのまま通る・F 以上は F−1・重複・0 の重複の文言」。「GenerateRequest を組んだだけでは frame_idx 5 が 5 のまま／validate_for_family で寄る」。strength 省略は LTX 0.8・H3 None。chain の clips[0] の省略が 0.8 になる。:942-953 の上限は残す。
- `tests/test_engine_h3_adapter.py`: :145-146（multi_keyframes の例を消し keyframe_strength の例を `strength: 0.8` に＝既定値でも拒否される）、:170-173（文言）、:193-194・:674（一覧）、:524-532（複数枚の payload）、`H3_CONSTRAINTS`／`LTX_CONSTRAINTS` の辞書（:221-242）と `GET /models` の完全一致（:585-586）に `keyframes` を足す。
- `test_ltx25_adapter.py:1626-1629`（0.8 で埋まった写し）。`test_chain.py`・`test_a2v_chain.py`・`test_smoke.py`（strength 省略の chain と metadata の写し）。LTX の監査テスト（`test_ltx_adapter_audit.py`・`test_ltx25_api_guard.py`）は変わらない。

### 4.3 Gradio と MCP
- `test_gradio_ui.py`: H3 で 10 本のスライダーが隠れて 0.8 に戻り、LTX では見える。`test_gradio_handlers.py`: 0.8 なら key 無し（:359-363 の LTX の期待値「0.8 の鍵がある」は「鍵が無い」に変わる）、0.5 ならあり。`test_mcp_tools_generate.py:198-206, 1349-1362`: 省略で null、明示ならその値。

### 4.4 フロントエンド（vitest）
- `keyframeGrid.test.ts`: (1,0) で validPositions が [0..F−1] 重複なし・max F−1・`nextAddPosition`／`stepPin`／`resolveLanding`／`findOverflowFrameIdxs`。(8,1) の既存の期待値は不変。
- `keyframeUtils.test.ts`: `snapFrameIdx` が (1,0) で恒等・(8,1) で今と同じ。`omitStrength` が鍵を外す。
- `generationConstraints.test.ts`: `keyframes` の解決と LTX の予備値。:60-61・:172-188 から keyframesMulti を消す。
- `AppShell.minimaxH3.test.tsx:162-179`: 2 枚目が足せる・強度の行が DOM に無い・送信の body に strength が無い。LTX の対照では強度の行がある。
- `featureScope.test.ts:31-32`・`mockBridge.h3.test.ts:43`: 一覧。`KeyframesPanel.test.tsx`: `strengthHidden`。
- 実行: `npm run typecheck`・`npx vitest run --exclude "**/backend.integration.test.ts"`・変更行の lint。

## 5. 弾と門

### 第 1 弾（末尾。実装一式と照合）

| 門 | 中身 | GPU |
|---|---|---|
| 1-G1 | 全テスト（pytest／型検査／vitest／lint）合格。float64 の総当たりの結論を §170 に書き、2.3 の式を確定 | 不要 |
| 1-G2 | 正解データとの照合。`Docs/Outputs-archive/h3-gate2/g3_i2v_parts.py` を土台に**新しい台本**（1 枚の呼び出し `vision_inputs(canvas)`・`encode(image=)`・`build_coords(cond_frames=1)`・`img_pos[:grid_rows]` は全部引数が変わるので書き直し）。正解側は `outputs/h3-gate2/oracle/diffsynth_h3/`（テキストエンコーダと VAE の写しは保管庫に無く、ここにだけある。`image_token_counts`・`presentation_fl2va`・`_build_packed_fl2va(keyframe_indices=[0,-1]／[-1])` は 2 枚に対応済み）。試験画像は `outputs/h3-gate1/g5_t2v_after_review.mp4` の先頭と末尾のフレーム。[先頭＋末尾] と [末尾だけ] で、A トークン列一致、B 画像入力と画像読み取り部の出力一致、C テキストエンコーダの配線の差 0、E 条件行の並び・時刻の番号・RoPE 座標の float64 一致（`torch.equal`）。D（VAE）は 2a-G3 と同じく差を記録。台本と結果は `outputs/h3-keyframes-2026-10-19/`（後で Outputs-archive へ） | 要（小） |
| 1-G3 | 指紋の回帰（合否）: セルフテストで T2V `0da6111c…`（黄色いレインコート・seed 42）と先頭 I2V `77e29687…`（J1 の条件: 同じプロンプト・`--image outputs/h3-gate2/i2v_input.png`・`--no-prefetch`・sdpa・vram16 off・常駐なし）。記録のみ（基準なし）: vram16 on の先頭 I2V、API 経由の T2V と先頭 I2V（新しい基準として §170 に残す） | 要 |
| 1-G4 | 先頭＋末尾と末尾だけを 1 本ずつ（640×384・124 フレーム・製品の既定＝vram16 on）。末尾フレームと入力画像の PSNR を記録（合否の線は置かない）。条件行 10 枚の VRAM の山を 1 度記録 | 要 |
| 1-G5 | コードの敵対的レビュー（Opus。採否は 2 フィルタ＝過剰設計・裏付け） | 不要 |
| 1-G6 | フロントを 2 か所に配置（実機の Plugin フォルダ＋backend の配布用の写し。`build.ps1`→`deploy.ps1`）。dev にコミット | 不要 |
| 1-G7 | オーナーの目視: H3 で末尾に置ける・強度の行が無い・2 枚以上置ける・LTX は 8n+1 の格子と強度の行が今のまま・1-G4 の動画 | 要（目視の生成） |

1-G2〜1-G4 の GPU は 1 度にまとめて了承を得る。

**目視の前にオーナーへ伝える画面の挙動（仕様どおり・本計画では変えない）**: ① H3 で「追加」を押すと 2 枚目は最後のピンと F−1 の中点に置かれる（末尾にするにはドラッグか数値入力）。② 尺を延ばすと、F−1 にあったピンは中間のフレームになり、画像読み取り部から黙って外れる（条件行には残る）。③ 1 フレーム刻みでは左端の近く（幅 600px・481 フレームで 1〜11 番あたり）はドラッグで 0 に吸われる。数値入力と矢印キーでは置ける。④ H3 で格子から外れた位置（例: 61）に置いたピンは、ベースモデルを LTX に切り替えても画面では寄せ直されない。送信するとサーバーが 8n+1 へ寄せ（61→57）、同じ点に寄る 2 枚があれば 422（重複）になる（1-G5 の M1。直すなら格子の変化を見る仕掛けが要るので本計画では変えない）。

### 第 2 弾（中間。オーナーの目視で go／no-go）

- 2-G1（GPU、了承が要る）: 中間 1 枚（F//2）と、先頭＋中間＋末尾を生成。指紋を §170 に記録。
- 2-G2: オーナーの目視で go／no-go。第 1 弾の実装はもともと任意の位置を受けるので、第 2 弾はコードの追加なし（確かめるだけ）。第 1 弾と第 2 弾の間、中間の位置は開発中の build でしか触れない。
- No-go のとき: `RequestRules.keyframe_ends_only: bool = False` を 1 項目だけ足し H3 を True に。サーバーは `constraints.keyframes.ends_only` を公開し、`normalize_keyframes` は 0 < n < F−1 を寄せずに拒否（「MiniMax H3 は先頭と末尾だけ」）。フロントの `validPositions` は `ends_only` なら [0, F−1]。エンジンは一般のまま残す。再配置と目視をもう一度。Go なら作らない。
- 足りないときの比較の腕「中間の画像を画像読み取り部に渡す」は、オーナーの合図があれば別途計画する（裁定 6）。

### クローズ
- CLOSED 3-241・VERIFICATION_LOG §170・DEVLOG §143。main へ `--no-ff` で merge し push。記憶ファイルを書く。

## 6. 進め方（担当・順序・コミット）

- 1a バックエンド（engine_h3＋API＋services＋テスト 4.1／4.2）: Opus 1 体。1b フロント（3.3＋4.4）: Opus 1 体（1a の API の形が決まってから着手。types.ts は 1b が所有）。1c Gradio＋MCP（3.4／3.5＋4.3）: Opus 1 体（1a の後）。1d 文書（3.6）: Opus 1 体（門の結果が出てから）。
- 1 ファイル 1 所有者。GPU と実機を使う検証は同時 1 体。担当には「設計に疑問があれば作業を止めて監督へエスカレーション・設計の変更は禁止・ComfyUI のコードは写さない・`engine_h3` に inference_mode を書かない」を毎回明記。
- コミットは監督が行う（dev）。本計画はコードを伴うので 1-G5 を必ず通す。

## 7. リスクと未決

1. float64 の一致は未知。総当たりで決める。不一致なら末尾だけ正解データの式。中間には比べる相手が無く目視だけ。
2. 寄せる場所の移動: `validate_for_family` を通らない経路で LTX の strength が None のままアダプタに届くと `float(None)`（`engine/worker.py:912, 1159`・`engine25/pipeline25.py:1889`）。経路は 2.2 のとおり全部覆う。テストで GenerateRequest を直接 runner に渡す場面は `validate_for_family` を通す形に直す。
3. 古い操作パネルとの組み合わせ: GET /config から 2 キーが消えるので古い build では格子の計算が NaN。フロントとサーバーは同時に配置する。仕様書の変更履歴に書く。
4. テキストエンコーダの長さ: 画像読み取り部に渡すのは最大 2 枚（1344×768 で 1 枚あたり約 1,008 トークン）。中間はトークンを増やさない。
5. 系列の長さ: 条件行が K×grid 行増える（1344×768・345 フレーム・10 枚で約 +10,080 行＝映像行の約 10%）。VRAM 16GB mode の分割は全長を見るだけ。1-G4 で山を記録し、快適上限の較正（§1-95）への含意を台帳に 1 行書く。
6. 画像処理器に 2 枚を一度に渡したときの `pixel_values` と `grid` は 1-G2 の B で正解データと照合。
7. 1-G3 の vram16 on と API 経由の I2V は基準が無い（§166・§169 で指紋を取ったのは J1/J3 と T2V だけ）。本計画では記録に留め、新しい基準にする。

## 8. 敵対的レビューの採否（2026-10-19・Opus・C2/M5/m13）

- 全採用。C1（`GenerationForm.tsx:280` と `featureScope.ts:108-111` の削除漏れ）・C2（fixture 4 ファイルは置き換え）・M1（進捗の分母）・M2（constraints の完全一致テスト）・M3（`images_used` は改名しない＝引き算）・M4（1-G3 の J1 の条件と基準の無い 2 つ）・M5（1-G2 は書き直し・正解側の置き場所・試験画像）・m1〜m13（監査テスト不変・Gradio バッチは届かない・Gradio の制約の写しに足さない・現行化の一覧・422 の順序 2 点・画面の挙動 3 点の事前共有・行番号・鍵を外す書き方・0.8 の写しの一本化・式の書き方・CLI の重複・任意のテストは足さない）。
- レビューの正本は `C:\Users\phenotype nezu\.claude\plans\scalable-floating-thimble-agent-aae12c0a332568922.md`（実装後に保管庫 `Docs/Outputs-archive/h3-keyframes-2026-10-19/` へ写す）。

### 実装の中心になるファイル
- `Nz-Videomni/services/engines/ltx/request_rules.py`
- `Nz-Videomni/engine_h3/pipeline_h3.py`・`rope.py`・`text_encoder.py`
- `Nz-Videomni/api/models.py`
- `Nz-Videomni/services/engines/h3/adapter.py`
- `webui/src/modes/single/keyframeGrid.ts`・`keyframeUtils.ts`・`shell/generationConstraints.ts`
