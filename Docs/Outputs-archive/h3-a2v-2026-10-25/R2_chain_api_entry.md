# R2 チェーン API 入口調査（H3 A2V を /generate/chain に載せるか）

- 調査日: 2026-10-25。担当: Sonnet（読み取りのみ）。監督が本文を保存した（担当に書き込み権限が無かったため。内容は担当の報告のまま）。
- パスはすべて `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\` 起点の相対表記（以下 BE）。フロントは `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/webui/src/`（以下 FE）。

## 結論（先に）

`/generate/chain` は「どのエンジンでも動く純粋な入口」ではない。入口の関門は 2 層ある。

- エンドポイント層は、エンジンを問わない作り。1 枚目の関門 `engines.reject_chain` は、アダプタに丸投げされている。LTX 2.5 が同じ呼び出し形で受理している前例もある。
- リクエストスキーマ `GenerateChainRequest` の Pydantic validator は LTX の規則を直書きしている。8n+1 フレーム、64 の倍数、distilled なら steps=8 かつ CFG=1.0、`chain_math` のレイアウト計算がそう。H3 の規則（17n+5、32 の倍数、20 steps）とは衝突する。
- この validator は、エンドポイントの手前（リクエスト解析の段）で走る。アダプタの `reject_chain` を直しても通らない。
- 音声ジオメトリ（25 latents/s）も `chain_math` と `preflight_source_audio` に固定されている。H3 は 40 latents/s。
- さらに根本的な点: H3 の音声 VAE は復号器しか持っていない（BE/engine_h3/audio_vae.py:32-35 は「符号化器は読まない」と明記）。H3 の A2V は、音声の符号化とワーカーのプロトコル拡張が先に必要。起票は Docs/PENDING_TASKS.md:26-31（台帳 §1-103）。

したがって「1 クリップ＋source_audio だけ受理」の分岐を足すだけでは済まない。chain スキーマに「H3 向けの別規則」を持たせる必要がある。比べると、単発 `/generate` 側は `validate_for_family` と `REQUEST_RULES` で既に系統別規則の仕組みが整っている（BE/services/engines/__init__.py:269-308、BE/api/generate.py:91）。拡張のしやすさは `/generate` の方が上。ただし `/generate` に `source_audio` を足すと、ジョブ種別や metadata の `a2v` ブロックを別に作ることになる。

## 1. GenerateChainRequest の全フィールド（BE/api/models.py:1170-1700）

### (i) エンジン中立（H3 の単発 GenerateRequest にも対応物がある）
- `prompt: str`（min_length=1、:1190）、`negative_prompt: str=""`（:1191）
- `width: int=512`（256〜4096、:1239）、`height: int=320`（128〜4096、:1240）
- `crop_output: CropOutput|None`（:1241）
- `frame_rate: float=24.0`（1〜60、:1243）
- `seed: int=-1`（:1246）
- `num_inference_steps: int=8`（1〜100、:1244）
- `source_audio: SourceAudioSpec|None`（:1270。`audio_id: str`（min 1）のみ、:905）
- `loras: list[LoraSpec]=[]`（:1288）
- `attention_backend`（sdpa|sage）、`block_swap_prefetch`、`keep_resident`、`embed_mp4_metadata`（:1211-1234）
- `clips[].conditioning_images`（ConditioningImage: `image_id`、`frame_idx>=0`、`strength: float|None`（0〜1）、`crf`、:187-193）

### (ii) LTX 固有
- `guidance_scale: float=1.0`（:1245）、`pipeline: distilled|two_stage_hq`（:1247）
- `stage2_window`（standard/high_resolution/full_length/w25〜w61、:1252 付近）、`chunked_upsample: bool`（:1319）
- `overlap_frames: int=3`（1〜8）、`overlap_strength: float=0.5`（:1254-1255）
- NAG/VSF 系（`nag_enabled`、`nag_scale`、`nag_tau`、`nag_alpha`、`neg_method`、`vsf_scale`、:1196-1204）
- `keep_resident_embeddings`、`fused_gguf_dequant_kernel`、`vae_mode`
- `source_video`、`retake`、`end_source`、`reference_video_id`、`conditioning_attention_strength`、`reference_video_strength`（:1265-1310）

### (iii) クリップ単位（ChainClip、:823-838）
- `prompt: str|None`、`num_frames: int=49`（9〜481）、`conditioning_images`（クリップ 0 のみ）
- `clips` は min 1、max 24。

### 1 クリップ＋source_audio に効く validator（:1352-1700）
- `width/height % 64 == 0` が必須。H3 は 32 の倍数なので衝突する。
- `pipeline=="distilled"` なら `num_inference_steps==8` かつ `guidance_scale==1.0` が必須。H3 既定の 20 steps は 422 になる。
- `source_audio` と `source_video` の併用は不可。`retake`、`end_source` との併用も不可。
- `clips` の下限は、source_audio があれば 1 で通る（:1447-1461）。
- 各クリップは `(num_frames-1)%8==0`（8n+1）が必須。H3 の 17n+5 は不可。
- `overlap_frames < stage-1 潜在フレーム数` の検査がある（1 クリップでも走る）。
- `LTX_REQUEST_RULES.normalize_keyframes` が直書きで走る（:1493 付近）。
  - LTX の格子（step 8 / offset 1）へキーフレームを寄せる。
  - strength=None に 0.8 を書き戻す。
  - H3 の規則（step 1 / offset 0、strength なし）とは逆。H3 は単発 `/generate` で strength 指定を 422 にする（BE/services/engines/h3/adapter.py:183-197、REJECT_TABLE の `keyframe_strength`）。
  - chain 経由だと、LTX 規則で strength=0.8 が補完され、H3 の流儀と衝突する。
- `stage2_window=="full_length"` は「1 クリップ＋source_audio」専用。
- `chain_math.compute_chain_layout(...)` を呼ぶ。ここで LTX の 8n+1 や 25/s のジオメトリが固定される。
- 合計が `MAX_CHAIN_TOTAL_PIXEL_FRAMES` 以下であること。
- `reference_video_id` や強度は loras 必須。
- `nag_enabled` なら negative_prompt が必須。

### スキーマに無い H3 の項目
- `shift_video`、`shift_audio`、`vram16_mode` は chain スキーマに無い（単発は GenerateRequest :594 に `shift_audio` がある）。
- `LoraSpec.audio_strength` を H3 向けに None に書き戻す処理は単発 `validate_for_family` にだけある。chain エンドポイントは `validate_for_family` を呼ばない（BE/api/generate_chain.py に呼び出し無し）。

### LTX アダプタがワーカーへ写す項目（BE/services/engines/ltx/adapter.py:2477-2700。2.5 も同形）
- payload の基本キー: `op=generate_chain`、`prompt`（shared）、`width`、`height`、`frame_rate`、`num_steps`、`seed`、`overlap_frames`、`overlap_strength`、`chunked_upsample`、`output_path`、`clips[{prompt, num_frames, images[{path, frame_idx, strength}]}]`（:2570-2583）
- 条件付きキー: `source`、`audio_source{path}`（:2595）、`retake`、`end_source`、`loras`、`reference_video`、`nag`、`attention_backend`、`block_swap_prefetch`、`keep_resident`、`fused_gguf_dequant_kernel`、`vae_mode`、`stage2_window`（非既定のみ）

## 2. ディスパッチ経路

### chain の流れ
1. `POST /generate/chain`（BE/api/generate_chain.py:34-）
2. `engines.reject_chain(active_engine_family, request)`（:58）。「UPLOAD 参照の 404 より前」に置く、とコメントで明記（:50-53）。
3. 事前検査: クリップ 0 の画像の存在／`source_video` 検査／`source_audio`: 404 と `preflight_source_audio`（:80-93）／`retake`、`end_source`、reference、loras 検査
4. `reject_if_loading()` の後、`job_store.create_chain_if_idle(request)`（:176、BE/services/job_store.py:158-175）。busy なら 409。JobRecord の `request` はクリップ 0 の `to_clip_request(0)` で作った GenerateRequest。`chain_request` に全体を持つ。
5. mock バックエンドなら BackgroundTasks、実機なら `spawn_job_thread(run_chain_job)`。202 の応答は `num_clips` 付き。
6. `PipelineManager.run_chain_job`（BE/services/pipeline_manager.py:1286-）が `self.runner.generate_chain(...)` を呼ぶ。全部で 13 個のキーワードを渡し、エンジンで分岐しない。
7. アダプタが worker payload を組む。

### 単発との差
- 単発は `run_job`（pipeline_manager.py:636）。結果は `_finalize`、`_write_metadata`（:1855-）。
- chain は `_write_chain_metadata`（:1626-）。`"kind":"chain"`、`generation_mode:"chain"`。
- 出力は、どちらも `outputs/<job_id>/output.mp4` と `metadata.json`（:1860 付近と :1650-1700）。
- chain は `output_dir` に中間ファイル（`_source_tail.mp4`、`_retake_window.mp4`、`_end_source.mp4`、crop 時は `_full.mp4`）を置く。A2V は音声をそのまま渡すだけで中間物は無い。
- 進捗は `_progress_frac(chain=True)`（ltx/adapter.py:483-）で、stage1 0.05〜0.50、stage2 0.50〜0.90、decode 0.95。単発の H3 は `chain=False, one_stage=True`（H3 adapter :545）で、stage1_denoise が 0.06〜0.90。H3 のワーカーは 1 段なので chain の帯域とは合わない。
- 単発 run_job は `outcome.generation_mode`（t2v/i2v 等）を使う。chain は固定で "chain"。

### reject_chain の位置
- 入口は BE/services/engines/__init__.py:311-327。`getattr(_adapter(family), "reject_chain", None)` で取り次ぐだけで、判断はアダプタ側。
- H3 の実装は BE/services/engines/h3/adapter.py:323-328 で、中身は無条件の `feature_unsupported("chain")`。H3 の `generate_chain` は到達不能のバックストップ（:584-587）。
- 「1 クリップ＋source_audio のみ受理」にするなら必要な分岐:
  1. h3 `reject_chain` を、LTX 2.5 の `CHAIN_REJECT_TABLE` 方式（述語表、ltx25/adapter.py:326-335、`reject_chain` :543-566）に倒す。
  2. 要求スキーマの validator を系統別にする。これが一番大きな作業。
  3. `preflight_source_audio` に系統別の音声ジオメトリを注入する（§5）。
  4. H3 の `generate_chain` を実装する（ワーカーのプロトコル拡張込み）。
  5. UNSUPPORTED_FEATURES から `chain` を外し、代わりに新名称を載せる（FE 側の `featureScope` も同期）。

### LTX 2.5 の前例
- 2.5 は chain を受理している。`reject_chain` は表式で、実際に拒否するのは `pipeline!="distilled"` と `vae_mode!="default"` だけ（ltx25/adapter.py:326-335）。
- `generate_chain` のシグネチャは 2.3 の 13 キーワードそのまま（ltx25/adapter.py:1164-1180）。payload のキー集合と順序も 2.3 と同じ（:1278-1300）。A2V は `audio_source{path}`（:1308-1311）。
- 重要な違い: 2.5 は 2.3 と同じ GenerateChainRequest スキーマをそのまま使える（8n+1、64 の倍数、25/s を共有）。H3 にはこの前提が無い。
- metadata には 2.5 専用の `ltx25` ブロックがある（pipeline_manager.py:1721 付近）。H3 向けの `h3` ブロックは chain 側にはまだ無い（単発 `_write_metadata` にだけある、:1956-1960）。

## 3. フロントエンド

### A2V カードがリクエストを組む箇所
- カード本体は FE/modes/single/GenerationForm.tsx:306-320（`<details>`、`SourceAudioSection`）。
- 送信の分岐は FE/modes/single/SingleScreen.tsx:1030-1056。`isA2v` なら `{kind:"chain", body: buildA2vRequest(conditioningImages)}`、そうでなければ `/generate`。
- `buildA2vRequest` は FE/modes/single/useGenerationForm.ts:1283-1345。実際の組み立ては FE/modes/batch/buildA2vChainPayload.ts:232-292。Batch A2V と同じ関数を共有。
- 送るキー: `prompt`、`negative_prompt`、(nag 系)、`width`、`height`、`crop_output`（null 可）、`frame_rate`、`seed`／固定値: `num_inference_steps: 8`、`guidance_scale: 1.0`、`pipeline: "distilled"`、`overlap_frames: 3`、`overlap_strength: 0.5`、`stage2_window: "full_length"`／`chunked_upsample: true`（Single 側は常に true、useGenerationForm.ts:1309）／`clips: [ {num_frames, conditioning_images?} ]`（1 クリップ）、`source_audio: {audio_id}`／条件付き: `loras`、`reference_video_id`（＋強度）、accelerationRequestFields
- 固定値は `A2vChainPayload` 型のリテラル（:168-218）。H3 の 20 steps や 32 の倍数とは合わない。
- 注意点: キーフレーム strength の除去（`omitStrength`）は `/generate` 側だけに適用される（SingleScreen.tsx:1032 は chain 側で `conditioningImages` をそのまま渡している）。H3 の A2V では chain 側にも `omitStrength` が要る。
- 音声フィットも LTX 固定: FE/modes/chained/chainUtils.ts:285 に `AUDIO_LATENTS_PER_SEC = 25.0`。`suggestFramesForAudio`（:389-400）は `audioLatentsRequired`（8n+1 の `vLatentFrames` 前提）と 8 刻みの縮小を使う。取り込み時の自動 `numFrames` 調整は useGenerationForm.ts:1043-1095。取り込み後に `snapNumFrames(..., frameGrid)` が掛かるが、縮小ループの `-= 8` と 25/s は LTX 固定のまま。

### Strength スライダ
- FE/modes/single/KeyframeCard.tsx:249-263 の素の `<input type="range">`。min/max/step は `STRENGTH_MIN=0.0`、`STRENGTH_MAX=1.0`、`STRENGTH_STEP=0.05`、既定 `DEFAULT_STRENGTH=0.8`（FE/modes/single/keyframeUtils.ts:36-42）。
- カード単位の状態は `useKeyframes.ts` の `setStrength`（:398-402、`clampStrength`）、初期値は `DEFAULT_STRENGTH`（:210 他）。
- API への写像は `buildConditioningImages`（keyframeUtils.ts:89-97）で `strength: item.strength`。`keyframe_strength` が閉じているときは `omitStrength`（:103-105）で除去され、`strengthHidden`（KeyframeCard :32-34）で行が非表示になる。
- 別の汎用スライダ `FloatSliderField.tsx`（range＋number）もあるが、キーフレームでは使っていない（KeyframeCard は専用実装）。

### featureScope.ts
- `FEATURE_UI.chain = { disables: ["mode.chained", "createSection.batchA2v", "createSection.a2v"] }`（FE/shell/featureScope.ts:151）。
- `a2v` は `chainPanel.a2v` と `createSection.batchA2v` を閉じる（:157）。`createSection.a2v` は閉じない。
- `singleSectionsDisabledFor`（:398-405）が `createSection.a2v` を読む。AppShell.tsx:1540 で算出し、:1706 で `sectionsUnavailable` として渡す。
- H3 は `chain` を公開しているので、A2V カードは灰色（GenerationForm.tsx:309-315）。
- 設計上の根拠: Docs/MINIMAX_H3_DESIGN.md:63-64（裁定 35 と 2026-10-21 の訂正）。

### H3 に A2V カードを使わせる最小のフロント変更案
1. サーバーが `chain` を「複数クリップの連結」だけに意味を絞る。または新しい機能名（例 `chain_multi`）を作る。
2. FEATURE_UI を変える。`chain`（または新名称）: `mode.chained` と `createSection.batchA2v` を閉じる。`a2v`: `chainPanel.a2v`、`createSection.batchA2v`、さらに `createSection.a2v` を閉じる。
3. H3 は `a2v` を公開しない（A2V カードを生かす）。複数クリップを使う Chained タブは `chain` で閉じたまま。
4. `buildA2vChainPayload` を H3 対応にする（steps、shift、32 の倍数、17n+5 のフレーム格子、`stage2_window`／`overlap_*`／`chunked_upsample` の除去、キーフレームの strength 除去）。
5. 音声フィットの定数（25/s、8n+1）を系統の `constraints` から読むようにする。
- 注意: `chain` の意味を変えると既存テストが連動する（featureScope.test.ts の表健全性チェックなど）。

### Batch A2V と MCP
- Batch A2V（FE/modes/batch/）は `buildA2vChainPayload` を使うので `/generate/chain` と `source_audio` を送る。`batchA2vDisabledFor`（featureScope.ts:278）で `chain` または `a2v` が閉じるとパネルごと閉じる（AppShell.tsx:1702）。H3 に `a2v` を公開させない場合はバッチも使えるようになる。バッチ側のペイロードも H3 向けにする必要がある。
- MCP（BE/mcp_server/）: `submit_chain` は `/generate/chain` に `source_audio` を送る（mcp_server/tools/generate.py:1067-1112）。ツールの説明に「連結生成は H3 では使えないので submit_chain には出さない」とある（generate.py:12、:40）。サーバー側（server.py:48-51）にも H3 の chain は 422 との説明。H3 判定のコード上のゲートは無く、サーバーの 422 に任せている。`plan_a2v_batch`（tools/batch.py、batch_planning.py:108-126）は `chain_math.audio_latents_required` と `AUDIO_LATENTS_PER_SEC`（25/s）、`kv=3`、8n+1 を使う純ローカルの計画ツール。H3 向けには音声幾何を差し替えないと誤った提案になる。
- Gradio 画面（BE/gradio_ui/handlers.py の `build_a2v_chain_payload`）も同系の chain 送信（未確認: H3 でのゲート有無）。

## 4. モック

- H3 のモックは独自実装ではなく、LTX 2.3 の `_MockBackend` を流用（BE/services/engines/h3/adapter.py:14-24 のコメントと `H3Runner._MOCK_BACKEND_CLS`、:589-596 付近）。ラベルだけ `mock-h3`。
- `_MockBackend.generate_chain`（BE/services/engines/ltx/adapter.py:1322-）は `chain_math` のレイアウトで合成 mp4 を作る。音声は `source_audio_path` の ffprobe 結果から `chain.a2v` を組み立てる（:1554-1580）。キーは `source_audio_path`、`a_total`、`encoded_audio_frames_available`、`muxed_original_waveform: True`、`vocoder_skipped: True`、`muxed_audio_samples`、`audio_sampling_rate`、`audio_channels`。モックは音声を mux しない。`AUDIO_LATENTS_PER_SEC`（25）直使い（:1565-1566）。
- H3 の chain 拒否テスト（更新対象）: BE/tests/test_engine_h3_adapter.py:189-208 の `UNSUPPORTED_FEATURES` 完全一致（`"chain"` を含む）／同 :259-263 の `test_every_chain_is_refused`／同 :896-902 の `test_h3_refuses_a_chain_over_http`。
- 他にも要確認の周辺テスト（未確認）: BE/tests/test_engine_dispatch.py、test_ltx_adapter_audit.py、test_gradio_models.py、test_gradio_constraints.py、FE の featureScope.test.ts。
- 参考にできる既存テスト: test_a2v_chain.py、test_chain_math_a2v.py、test_ltx25_adapter.py（`CHAIN_ACCEPTED_SOURCES` 等）。

## 5. プリフライト

BE/services/pipeline_manager.py:1243-1284 の `preflight_source_audio` は LTX 固定。
- `chain_math.audio_latents_required(clip_frames, frame_rate, kv=overlap_frames)` で必要量を出す。
- `available = round(duration * chain_math.AUDIO_LATENTS_PER_SEC)`（定数は BE/chain_math.py:45 の 25.0 = 16000/160/4）。
- `available < required` なら `source_audio_too_short`（422）。ffprobe で長さが取れなければ return してエンジンのバックストップに任せる。
- `audio_latents_required`（BE/chain_math.py:1709-）はチェーンの全体幅から `a_frames_for_px`（:448）で求める。8n+1 の潜在幾何を前提。

H3 の規則は BE/engine_h3/constants.py:34-35、:90-92。`AUDIO_LATENTS_PER_SECOND=40`、`AUDIO_SAMPLE_RATE=32000`、`audio_latents(F)=round(F/24*40)`、フレームは 17n+5。

エンジン固有の音声幾何を注入する候補地点: (a) `preflight_source_audio` の引数かアダプタ関数経由（`engines.audio_geometry(family)` のような取り次ぎ。既存の `request_rules(family)` と同型）／(b) `GenerateChainRequest` validator 内の `compute_chain_layout` 呼び出し（models.py:1517-1530）／(c) FE の `chainUtils.ts` の定数と `suggestFramesForAudio`／(d) MCP の `batch_planning.py`／(e) mock の `_MockBackend.generate_chain`（:1565-1566）。

## 6. メタデータと出力

- `a2v` ブロック（BE/services/pipeline_manager.py:1704-1708）は、エンジン（または mock）が返す `chain_meta["a2v"]` に、アプリ側のプロベナンス `{"source_audio_id": ...}`（:1424）を合成して書く。`chain` ブロックには `architecture: "masked_av_latent_concat"`、`overlap_*`、`stage2_window`、`v_tile`、`kt_v` などの LTX 概念が固定で入る（:1656-1682）。H3 では意味を持たないので H3 用のブロック形を決める必要がある。`request: chain.model_dump()` も丸ごと記録される。
- mp4 への埋め込みは、`chain.embed_mp4_metadata` が真なら `_embed_recipe`（:1993-1999）で metadata.json の本文を mp4 の `comment` タグに入れる。単発と同じ仕組みでエンジンに依存しない。
- H3 ワーカーの `media_out.write_mp4`（BE/engine_h3/media_out.py:81-）: シグネチャは `write_mp4(path, frames: uint8[F,H,W,3] CPU, *, fps:int, audio: float32[2,N] CPU (-1..1) | None, sample_rate=32000)`。外部波形は mux できる（`_write_audio` :147-、`audio_keep_samples` が映像長 `round(F*sample_rate/fps)` で切り詰める。パディングはしない）。受け側は `[2, N]` に限る。モノラルや別サンプルレートは呼び出し側で 32000Hz ステレオに変換しておく必要がある。現行の呼び出しは `_decode`（BE/engine_h3/pipeline_h3.py:1086-1130）で音声 VAE の復号結果 `wav` を渡している。
- H3 のワーカーの generate プロトコルには `source_audio` の入力が無い（worker.py :28-83）。

## 判断材料の要約

| 観点 | /generate/chain に乗せる | /generate を拡張 |
|---|---|---|
| 系統別規則の仕組み | 無い（スキーマ validator が LTX 直書き） | あり（REQUEST_RULES / validate_for_family） |
| 前例 | LTX 2.5 が受理（ただし規則を共有している） | 単発で H3 は既に稼働 |
| UI の接続 | 既存の A2V カード／バッチがそのまま chain を送る | カード送信先の変更が要る |
| metadata | chain ブロックに LTX 概念が固定 | `h3` ブロックの流用が可能 |
| 進捗帯域 | chain 帯（stage1/2）が H3 の 1 段と不一致 | `one_stage=True` が既存 |

どちらを選んでも、H3 側で必要な本体（音声の符号化、音声行の時刻固定、ワーカーのプロトコル拡張、40/s の音声幾何）は同じ。違いは入口の周辺作業の量。
