# R4b 既存 LTX の I2V（画像から動画）の規約の棚卸し — H3 第 2 弾の下調べ

調査日 2026-10-17。読むだけの調査で、リポジトリの編集・テスト・GPU の使用はしていない。唯一の実行は、LTX 2.5 の映像 VAE（変分自己符号化器。画素と潜在の相互変換器）の safetensors のヘッダ（`__metadata__`）を Python で読んだことだけ（重みは読んでいない）。

パスの略記（すべて `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\` からの相対）:

- `V23` = `.venv-engine/Lib/site-packages`（LTX 2.3 のワーカーが使う公式パッケージ `ltx_pipelines`／`ltx_core` 1.0.0）
- `V25` = `.venv-engine-ltx25/Lib/site-packages`（LTX 2.5 のワーカーが使う同 1.2.0）
- `FE` = `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2`（操作パネルとネイティブ部）

---

## ① 結論の表

| 項目 | LTX 2.3（`engine/`） | LTX 2.5（`engine25/`） |
|---|---|---|
| A1 画像の入口 | **両系統とも同じ**。画像はまず `POST /upload/image` でアップロードし、`image_id`（UUID）を受け取る。保存時に EXIF の向きを反映し RGB に変換して `uploads/{image_id}/input.png` に PNG で置く。生成要求は `GenerateRequest.conditioning_images: list[ConditioningImage]`（各要素 `image_id`・`frame_idx`〔既定 0〕・`strength`〔既定 0.8〕・`crf`〔隠し項目〕）。空配列なら T2V（文章から動画）、1 件以上なら I2V。先頭フレームは `frame_idx == 0`。0 以外は API の検査で 8n+1 の格子へ寄せられる（上限 10 枚）。オーケストレータ（`pipeline_manager`）が `image_id` をファイルパスへ解決し、アダプタがワーカーへのメッセージ `"images": [{"path", "frame_idx", "strength"}]` を作る（`crf` は落とす）。**ワーカーへはファイルパスで渡る。** | 2.3 と同じ（アダプタは別ファイルだが、`images` の組み立ては同じ形）。ワーカーは `image_conditionings()` で公式の `ImageConditioningInput` に詰め直す（`crf=None`）。 |
| A2 キャンバスへの合わせ方 | 公式の `load_image_conditioning`。① PIL で開き RGB の 3 チャンネルを取る → ② **H.264 で 1 枚だけ再圧縮して戻す（CRF＝画質係数 33。公式の既定。アプリは CRF を渡していない）** → ③ torch の `interpolate(mode="bilinear", align_corners=False)` で**縦横比を保って覆うように拡大縮小し、中央で切り抜く**（引き伸ばしでもレターボックスでもない）→ ④ `x/127.5 − 1` で −1〜1 に正規化し bf16 へ。**段 1 は半分の解像度（width/2 × height/2）、段 2 は全解像度で、それぞれ画像から作り直す。** 操作パネル側は寸法を合わせない（元ファイルをそのまま送る）。 | 処理の形は同じ（中央切り抜き・bilinear・−1〜1・段 1 と段 2 で 2 回）。違いは 2 点。(a) **H.264 再圧縮の CRF は 18**。公式が VAE ファイルの `model_version` から決める（2.4 以上は 18。手元の 2.5 の VAE 2 本はどちらも `model_version=2.5.0`）。(b) 画像の読み込み（`decode_image`）が EXIF の向き・ICC プロファイル・RGBA を扱う（アップロード時に RGB の PNG へ正規化済みなので、実害の差はほぼ無いと推測）。 |
| A3 VAE の符号化 | **分布の平均を取る**（符号化器の出力を平均と対数分散に分け、平均だけを使う。サンプリングは無いので乱数の種は関係しない）。その後、チャネルごとの統計で正規化（`per_channel_statistics.normalize`）。dtype は bf16。**タイル分割なし**（`video_encoder(image)` を直接呼ぶ）。§3-76 のタイル化は IC-LoRA の参照動画だけで、先頭画像には効かない。 | 2.3 と同じ（平均・正規化・bf16・タイル分割なし）。符号化器は `ImageConditioner` が呼ばれるたびに組み立てて使い終わったら解放する。 |
| A4 条件付けの仕方 | `frame_idx == 0` は**潜在の置き換え**（`VideoConditionByLatentIndex`）。潜在フレーム 0 のトークンを符号化結果で上書きし、`clean_latent` にも同じ値を入れ、`denoise_mask = 1 − strength` とする。初期ノイズは `denoise_mask` の割合だけ混ぜる。各ステップで `denoised × mask + clean × (1 − mask)` と**毎ステップ入れ直す**。時刻（timestep）もトークンごとに `mask × sigma`。strength 1.0 なら完全に固定、0.8 なら弱く固定。0 以外の `frame_idx` は**ガイドとして末尾にトークンを追加**（`VideoConditionByKeyframeIndex`。位置符号を画素フレーム数だけずらす）。2.3 は公式 1.0.0 にこの振り分けが無いので、アプリ側で関数を差し替えて振り分けている。 | 2.3 と同じ仕組み（公式 1.2.0 の `combined_image_conditionings` が同じ振り分けを持つので差し替え不要）。段 1 は祖先サンプラー（毎ステップ新しいノイズを注ぐ方式）を強制オンにしているが、その場合も各ステップで `denoise_mask` による入れ直しが入る。 |
| A5 操作パネルの表現 | **両系統とも同じ**（系統を見ない）。Single 画面のキーフレームパネル（`KeyframesPanel`／`useKeyframes`）で画像カードを置き、`frame_idx` と `strength`（既定 0.8）を指定する。枚数の上限はサーバーの `/config` の `max_conditioning_images`（10・系統共通の定数）。Chained 画面は `maxItemsOverride: 1` で先頭 1 枚に絞る（これは画面ごとの絞りで、系統ごとではない）。**`featureScope.ts` にはキーフレームや I2V の項目は無い**。系統によって末尾・中間のキーフレームを隠す／拒否する仕組みは、操作パネルにもアダプタにも**今は存在しない**。拒否の仕組みとして使えるのは、アダプタの `REJECT_TABLE`（要求を見て 422 を返す述語と、`GET /models` に載る機能名の表）と、`featureScope.ts` の `FEATURE_UI`（機能名から閉じる画面部品への対応表）の 2 つ。 | 2.3 と同じ。 |
| A6 Gradio と MCP | **両系統とも同じ**。Gradio: キーフレームのスロット（有効・画像パス・frame_idx・strength〔既定 0.8〕）ごとに `upload_image()` で `POST /api/v1/upload/image` へ送り、返った `image_id` で `conditioning_images` を作る。MCP: `upload_image(file_path)` ツールで `image_id` を得て、`submit_generate` の `conditioning_images: [{image_id, frame_idx=0, strength=0.8}]` に渡す（`crf` は出さない）。 | 2.3 と同じ。 |

---

## ② 根拠

### A1 画像の入口

- API の型: `api/models.py:159-163` `ConditioningImage`（`image_id: str`・`frame_idx: int = 0`・`strength: float = 0.8`・`crf: int | None = None`）。`api/models.py:600` `GenerateRequest.conditioning_images`。`api/models.py:790` で `"i2v" if self.conditioning_images else "t2v"`。
- 格子への寄せ: `api/models.py:166-216` `_normalize_conditioning_images`。「frame_idx == 0 -> start frame, routed to the latent-replace path」「frame_idx > 0 ... Snap via (f-1)//8*8+1 ... clamp to [1, num_frames-8]」。0 の重複は拒否。呼び出しは `api/models.py:668`。連結（Chained）は `clips[0]` だけが持てる（`api/models.py:1480-1486`）。
- 上限: `config.py:322-325` `MAX_CONDITIONING_IMAGES = 10`（「LTX 2.3 / 2.5 共通で揃える」）。`config.py:334-337` で `/config` の `max_conditioning_images` として公開。
- アップロード: `api/uploads.py:22-38` `upload_image`。`services/upload_store.py:45-74` `UploadStore.save`（`ImageOps.exif_transpose` → `convert("RGB")` → `dest_dir / "input.png"` に PNG 保存）。
- id → パス: `services/pipeline_manager.py:681-684`（`self.upload_store.path_for(ci.image_id) for ci in job.request.conditioning_images`）。連結は `:1341-1344`。
- アダプタ（2.3）: `services/engines/ltx/adapter.py:2170-2181` `_RealBackend.generate`。「The engine's ImageConditioningInput has NO crf -> drop it.」で `{"path": str(path), "frame_idx": ci.frame_idx, "strength": ci.strength}`。`:2237` で payload の `"images"`。
- アダプタ（2.5）: `services/engines/ltx25/adapter.py:795-800` `_RealBackend25.generate`（同じ形）。`:853` で `"images"`。`HONOURED_FIELDS` に `conditioning_images`（`:241`）。
- ワーカー（2.3）: `engine/worker.py:907-915`（`ImageConditioningInput(path=..., frame_idx=..., strength=...)`、型は `engine/api_types.py:8-13` で `crf` 無し）。`:1072-1076` で `_PIPE.generate(..., images=images)`。エンジン内で公式の型へ詰め直すとき `crf` を渡さない: `engine/pipeline/fast_video_pipeline.py:1716` `_LtxImageInput(img.path, img.frame_idx, img.strength)` → 公式の既定 `crf: int = DEFAULT_IMAGE_CRF`（`V23/ltx_pipelines/utils/args.py:16-20`）。
- ワーカー（2.5）: `engine25/worker.py:745-749` → `engine25/pipeline25.py:1871-1893` `image_conditionings`（ファイルの存在を確かめ、`crf=None` のまま公式の型へ）。`engine25/pipeline25.py:1697` で `self.pipeline(..., images=images, ...)`。

### A2 キャンバスへの合わせ方

- 2.3: `V23/ltx_pipelines/utils/media_io.py:82-99` `load_image_conditioning`（`decode_image` → `preprocess(crf)` → `torch.tensor(float32)` → `resize_and_center_crop` → `normalize_latent`）。
  - `:43-79` `resize_and_center_crop`: 「Resize tensor preserving aspect ratio (filling target), then center crop」。`scale = max(height/src_h, width/src_w)`・`math.ceil`・`interpolate(..., mode="bilinear", align_corners=False)`・中央切り抜き。
  - `:78-79` `normalize_latent`: `(latent / 127.5 - 1.0).to(device, dtype)`。
  - `:113-116` `decode_image`: `np.array(Image.open(path))[..., :3]`。
  - `:382-391` `preprocess`: `crf == 0` 以外は `encode_single_frame`（libx264・yuv420p・偶数へ切り詰め）→ `decode_single_frame`。`V23/ltx_pipelines/utils/constants.py:76` `DEFAULT_IMAGE_CRF = 33`。
  - 段 1／段 2: `V23/ltx_pipelines/distilled.py:133-140`（`height=stage_1_output_shape.height` = 半分）と `:164-171`（全解像度）。dtype は `:60` `self.dtype = torch.bfloat16`。
- 2.5: `V25/ltx_pipelines/utils/media_io/decode.py:46-79` `load_image_and_preprocess`（同じ流れ。`.exr` のときだけ HDR の別経路）。`V25/ltx_pipelines/utils/media_io/resize.py:41-74` `resize_and_center_crop`（2.3 と同一の本文）。`V25/ltx_pipelines/utils/media_io/range_map.py:8-9` `normalize_images`（`/127.5 − 1`）。`decode.py:139-` `decode_image`（EXIF の向き・ICC・RGBA の扱いあり）。
  - CRF: `V25/ltx_pipelines/distilled.py:212` `images = self.image_conditioner.resolve_crf(images)`。`V25/ltx_pipelines/utils/blocks.py:936-983` `ImageConditioner.default_image_crf`／`resolve_crf`（VAE の checkpoint の `model_version` から決める）。`V25/ltx_pipelines/utils/constants.py:36-37` `DEFAULT_IMAGE_CRF = 33`・`LTX_2_4_IMAGE_CRF = 18`、`:124-133` `(2, 4)` 以上は 2.4 の値。`V25/ltx_pipelines/distilled.py:120-121` で `ImageConditioner` に渡すのは `model_paths.video_vae()`。
  - 観測事実: `models/LTX25/VAE/ltx-2.5-video-vae-conv-bf16.safetensors` と `models/LTX25/VAE/diffvae/ltx-2.5-video-vae-bf16.safetensors` の `__metadata__` に `model_version: 2.5.0` がある（ヘッダを直接読んで確認）。ここから CRF は 18 になると**コードから読み取れる**（実行時のログでは未確認）。
  - 段 1／段 2: `V25/ltx_pipelines/distilled.py:251-261`（`stage_1_w, stage_1_h = width // 2, height // 2`）と `:280-290`（全解像度）。dtype は `:218` `dtype = torch.bfloat16`。`engine25/pipeline25.py` は `color_space` を渡していない（`.exr` 以外では無関係）。
- 操作パネル側: 画像の用意は `timeline.captureFrame`（AviUtl2 のシーンを描画して WIC で PNG 化。`FE/native/src/bridge.cpp:380-`）か `ui.pickFile`。送信は `backend.uploadFile` が元ファイルを multipart でそのまま送る（`FE/native/src/bridge_core.h:636-680` `UploadFileRequest`・`BuildMultipartHeader`）。寸法合わせのコードは見当たらない。

### A3 VAE の符号化

- 2.3: `V23/ltx_core/model/video_vae/video_vae.py:251-320` `VideoEncoder.forward`。docstring「Normalized latent means」、`:319-320` `means, _ = torch.chunk(sample, 2, dim=1)` / `return self.per_channel_statistics.normalize(means)`。呼び出しは `V23/ltx_pipelines/utils/helpers.py:65`・`:95` `encoded_image = video_encoder(image)`（`tiled_encode` ではない）。
- 2.5: `V25/ltx_core/model/video_vae/video_vae.py:335-336`（同じ `chunk` → `normalize(means)`）。呼び出しは `V25/ltx_pipelines/utils/helpers.py:272-308` `combined_image_conditionings` 内の `video_encoder(image)`。
- タイル化: `tiled_encode` を使うのは参照・V2V・末尾素材・Retake の経路（例 `engine/pipeline/chain_pipeline.py:600`・`:610`・`:736`）。§3-76 は「参照動画 VAE エンコードのタイル化」（`Docs/PENDING_TASKS_CLOSED.md` §3-76、`Docs/ACCELERATION_RESEARCH_NOTES.md:43`）。

### A4 条件付けの仕方

- 置き換え: `V23/ltx_core/conditioning/types/latent_cond.py:16-44` `VideoConditionByLatentIndex.apply_to`（`latent[:, start:stop] = tokens`・`clean_latent[...] = tokens`・`denoise_mask[...] = 1.0 - self.strength`）。2.5 は `V25/ltx_core/conditioning/types/latent_cond.py:40-41` で同じ。
- ガイド追加: `V23/ltx_core/conditioning/types/keyframe_cond.py`（`positions[:, 0, ...] += self.frame_idx`、`denoise_mask = 1.0 - strength`、`torch.cat` で末尾に追加、`causal_fix` は `frame_idx == 0` のときだけ）。
- 初期ノイズ: 2.3 `V23/ltx_core/components/noisers.py:22-34`（`scaled_mask = denoise_mask * noise_scale`、`noise * scaled_mask + latent * (1 - scaled_mask)`）。2.5 `V25/ltx_core/components/noisers.py:30-33`（`lerp(latent, noise, noise_scale)` の後 `lerp(clean_latent, latent, denoise_mask)`）。
- 毎ステップの入れ直し: `V23/ltx_pipelines/utils/helpers.py:201-203` `post_process_latent`（`denoised * mask + clean * (1 - mask)`）、`:228-233` `timesteps_from_mask`（`mask * sigma`）。サンプラーからの呼び出しは `V23/ltx_pipelines/utils/samplers.py:60` ほか。2.5 は `V25/ltx_pipelines/utils/helpers.py:461-463`、祖先サンプラーの経路でも `V25/ltx_pipelines/utils/samplers.py:484` 付近・`:557` 付近で呼ばれる。祖先サンプラーの強制は `engine25/pipeline25.py:1120`。
- 2.3 の振り分けの差し替え: `engine/pipeline/fast_video_pipeline.py:1570-1622`（`_hybrid_image_conditionings`。`frame_idx == 0` は `_orig_replace`、0 より大きいものは `image_conditionings_by_adding_guiding_latent`）、`:1697` で差し替え、`:1729` で戻す。2.5 は公式の `combined_image_conditionings`（`V25/ltx_pipelines/utils/helpers.py:272-308`）がそのまま同じ振り分けをする。

### A5 操作パネル側の表現

- 要求の組み立て: `FE/webui/src/modes/single/keyframeUtils.ts:73-82` `buildConditioningImages`（`"ready"` のカードだけ `{image_id, frame_idx, strength}`。`crf` は送らない）。`:34` `DEFAULT_STRENGTH = 0.8`。`:50-53` `snapFrameIdx`（サーバーの寄せの予告表示）。`FE/webui/src/modes/single/SingleScreen.tsx:978`（1 件以上なら `conditioning_images` を付け、0 件なら鍵ごと省く）。
- 上限: `FE/webui/src/modes/single/useKeyframes.ts:18-25` `maxItemsOverride`（「Chain mode's single "start frame" slot passes `1`」）。
- 系統ごとの制御: `FE/webui/src/shell/featureScope.ts:104-116` `FEATURE_UI` の項目は `chain`・`retake`・`outpaint`・`inpaint`・`alpha_gen`・`v2v`・`a2v`・`end_source`・`reference_video`・`prune_vaed`・`keep_resident_embeddings` の 11 個だけで、キーフレームは無い。`UiTarget`（`:31-44`）にもキーフレームのパネルは無い。`grep` でも `modes/single` 配下は `unsupported_features` を読んでいない。
- サーバー側の拒否の仕組み: `services/engines/ltx25/adapter.py:140-143` `REJECT_TABLE`（`(フィールド名, 機能名, 要求を見る述語)` の組。今は `pipeline` と `vae_mode` の 2 行だけ）。`:462-` でこの表の機能名が `GET /models` の `unsupported_features` になる。`:472-` `reject_unsupported`。

### A6 Gradio と MCP

- Gradio: `gradio_ui/ui.py:579-582`（スロットの strength スライダー既定 0.8）。`gradio_ui/handlers.py:809-889`（スロットの検査 → `api.upload_image(image_path)` → `{"image_id", "frame_idx", "strength"}`）。Chained の先頭画像は `:1438-1452`（`frame_idx: 0`）。`gradio_ui/api_client.py:121-127` `upload_image`（ファイルを読んで `POST /api/v1/upload/image`）。`gradio_ui/feature_scope.py` にキーフレームの語は無い。
- MCP: `mcp_server/params.py:18-30` `ConditioningImageArg`（`image_id`・`frame_idx = 0`・`strength = 0.8`。「crf はサーバー側の隠しフィールド ... ここには出さない」）。`mcp_server/tools/uploads.py:78-103` `upload_image(file_path)`。`mcp_server/tools/generate.py:488-489`（`payload["conditioning_images"] = [ci.model_dump() ...]`）。

---

## ③ B（`engine_h3/video_vae.py` の現状）と C（モック）

### B1 符号化の入口と返すもの

- 入口は `H3VideoVAE.encode(video, *, tiled=True)`（`engine_h3/video_vae.py:567-597`）。入力は `[1, 3, F, H, W]` の float32、値域 **0〜1**。H と W は 16 の倍数でないと `ValueError`。
- 画素の正規化は **ImageNet の平均・標準偏差**（`:79-81` `IMAGENET_MEAN`／`IMAGENET_STD`、`:562-565` `_pixel_stats`、`:584` `((video - pm) / ps).to(self.compute_dtype)`）。LTX の −1〜1 とは違う（docstring `:35-36`「Pixels are ImageNet-normalised RGB over a [0, 1] base, NOT [-1, 1].」）。
- 返すもの: **事後分布の平均**（`:593` `z = moments.chunk(2, dim=1)[0]`。docstring `:568`「normalised posterior mean」）。サンプリングは無い。
- dtype: 計算は符号化器の重みの dtype（`:559-560` `compute_dtype`。本番は bf16、`:423` `from_file(..., dtype=torch.bfloat16)`）。`_encode_clip` の出力は float32（`:523-537`）。返り値も float32。
- `latents_mean`／`latents_std`: 符号化の最後に `(z − mean) / std`（`:594-596`）。値は `from_file` が VAE ファイルの `__metadata__` の JSON から読む（`:433-437`、`latent_stats_from_metadata` `:98-123`）。バッファとしては float32 のまま（`:417-418`、`:467`）。
- タイル分割: 既定でオン（`tiled=True`。256 px のタイル、64 px 以上の重なり。`:44-46`、`:528-537`）。

### B2 T'=1 の扱い（符号化側）

- **ある**。`:586-587` `if num_frames == 1: moments = self._encode_clip(x, tiled)[:, :, :1]`。docstring `:571-574`「F = 1 (a key frame) goes through the spatial encoder alone and gives one latent frame; repeating it to 17 frames would give 2 latents and is not the conditioning the model was trained with (R2 (3)-13).」
- 2 枚以上のときは 17 の倍数まで末尾を複製して 17 枚ずつ符号化し、末尾 3 潜在を落とす（`:588-592`）。

### B3 テストの範囲

- `tests/test_h3_media.py:91-97` `test_encode_shapes` の 1 本だけ。小さい乱数重みの VAE（`:39-41` `ch=32, decoder_layers=1, ...`）で、`F=22` → `(1, 24, latent_frames(22), 2, 2)` と `F=1` → `(1, 24, 1, 2, 2)` の**形だけ**を確かめる。入力 32×32 なのでタイルは 1 枚で、タイルの継ぎ目は通らないと推測。
- 値の正しさ（公式実装との数値照合・実重みでの往復）はテストに無い。ソース自身も未検証と書いている（`:215` `EncoderFCN3D`、`:525` `_encode_clip`、`:570` `encode` の各 docstring「第 1 弾（T2V）では未使用・未検証（CPU の往復を 1 回試しただけ）。第 2 弾の先頭フレームで使う。」）。
- 関連: `engine_h3/constants.py:34` に `KEYFRAME_NOISE_AUG = 0.999  # used from the second stage on (I2V condition rows)` が既に定義されている。`engine_h3/pipeline_h3.py`・`engine_h3/worker.py` に画像入力の経路はまだ無い（`grep` で `image` が出ない）。

### C モック

- **操作パネルのモック（`FE/webui/src/bridge/mockBridge.ts`）は絵を作らない**。アップロードは架空の応答を返すだけ（`:2304-2320` `uploadResponseBody`。`image_id: mock-image-N`、`width/height: 512` 固定、ファイルは読まない）。生成ジョブは要求の `conditioning_images` をそのまま記録に写して返す（`:699-701`・`:720` `buildEchoedRequest`）。出力は架空のパス（`:787` `outputs/${job.id}/output.mp4`）。
- **バックエンドのモック**（アプリ側の `_MockBackend`。`services/engines/ltx/adapter.py`。2.5 のアダプタも `services/engines/ltx25/adapter.py:51`・`:1451` でこれを使う）は画像から絵を作る。最初の 1 枚だけを使い（`:1140-1142` docstring「uses only the first resolved image path as the start frame and ignores the rest」）、`Image.open(...).convert("RGB").resize((width, height))` で**縦横比を無視して引き伸ばす**（`:1223-1226`。補間フィルタは指定なし＝Pillow の既定。アプリの venv は Pillow 12.2.0）。その後フレームごとに少し拡大して横に揺らしながら切り抜き、動く球を描く（`_render_frames` `:1612-1620`）。連結も同じ（`:1348-1352`、`_render_chain_frames` `:1546-1553`）。

---

## ④ H3 の I2V を揃えるときの仕分け（設計ではなく仕分けだけ）

### そのまま使い回せる部品（事実に基づく）

- API の形: `ConditioningImage`／`conditioning_images`／`/upload/image`／`UploadStore`（EXIF 反映・RGB の PNG 化）／`pipeline_manager` の id → パス解決。
- アダプタからワーカーへのメッセージの形 `"images": [{"path", "frame_idx", "strength"}]`（2.3 と 2.5 で同じ形）。
- アダプタの拒否の仕組み（`REJECT_TABLE` 形式の表 → `GET /models` の `unsupported_features`）。
- 操作パネルのキーフレームパネル・`buildConditioningImages`・`maxItemsOverride`、Gradio のスロット、MCP の `upload_image`／`ConditioningImageArg`（どれも系統を見ないので、受け口としてはそのまま動く）。
- バックエンドのモック `_MockBackend`（先頭 1 枚だけを使う動きは H3 の先頭のみと一致）。
- `engine_h3/video_vae.py` の `encode`（平均・`latents_mean/std`・F=1 の扱い・タイル）。ただし値は未検証。

### H3 で新しく書く必要がある部分

- ワーカー側の画像の読み込み・寸法合わせ・値域（H3 の VAE は 0〜1 の ImageNet 正規化、寸法は 32 の倍数〔`engine_h3/constants.py:26`〕。LTX の公式関数は −1〜1 を返し、`ltx_pipelines` は H3 の venv に無いと推測〔未確認〕なので、関数の流用は効かない）。中央切り抜き・bilinear・H.264 再圧縮（CRF）に揃えるかどうかは決めごと。
- 符号化結果を H3 のデノイズに渡す条件付け（`KEYFRAME_NOISE_AUG` の使い方を含む。LTX の `denoise_mask` 方式とは別物になると推測）。
- `encode` の数値照合（実重み・公式実装との比較）。
- H3 のアダプタと「先頭のみ」の拒否（`frame_idx > 0` や 2 枚以上を 422 にする述語、その機能名）。今の `featureScope.ts` にはキーフレームを閉じる部品（`UiTarget`）が無いので、操作パネルで隠すなら新しい対応が要る。Gradio の `gradio_ui/feature_scope.py` も同様。
- 2.3 の `crf` 落としに相当する扱い（API の `crf` はどの系統も使っていない）。

---

## ⑤ 未確認事項

- LTX 2.5 で実際に CRF 18 が使われていること（コードとヘッダからの読み取りで、実行時のログでは確かめていない）。`detect_params` の該当行（`V25/ltx_pipelines/utils/constants.py:175` 付近）は末尾まで読んでいない。
- 2.5 の祖先サンプラーの各ステップで、ノイズを注いだ後に `denoise_mask` で入れ直す順序の細部（`samplers.py` の該当箇所の行は見たが、関数全体は読んでいない）。
- 2.3 の `ltx_pipelines` 1.0.0 の `decode_image` は EXIF を扱わないが、アップロード時に正規化済みのため実害は無いと推測。ICC プロファイルが PNG に残るかどうかは未確認。
- 操作パネル（ネイティブ部の `timeline.captureFrame`）が書き出す PNG の寸法がシーンの解像度そのままかどうか（コードの冒頭だけ見た）。
- `.venv-engine-h3` に `ltx_pipelines` が入っていないこと（ディレクトリを見ていない）。
- `tests/test_h3_media.py` の `test_encode_shapes` が実際にタイル 1 枚で通ること（`split_tiles` の 32 px での振る舞いは読んでいない）。
