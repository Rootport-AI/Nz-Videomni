# R4a MiniMax H3 の I2V で先頭画像を映像 VAE で潜在にする工程 — 外部の参照実装 3 つの調査

作成: 2026-10-17／調査担当（読むだけの調査。GPU 不使用・インストールなし）

## 読んだ版

| 実装 | リポジトリ | コミット（日付） |
|---|---|---|
| ComfyUI | https://github.com/comfyanonymous/ComfyUI | `5cc4b870094c89090db54bd8d12649dcaf89f2ba`（2026-10-16） |
| ComfyUI 公式テンプレート | https://github.com/Comfy-Org/workflow_templates | `9afccb38bb9173575411189a75f4fc6cc8acd06c`（2026-10-17） |
| diffusers | https://github.com/huggingface/diffusers | `b800f828f3951c4d206e154411bcbfa693cd322e`（2026-10-16） |
| DiffSynth-Studio | https://github.com/modelscope/DiffSynth-Studio | `4e3496991455b300fd60ab023113422cd16899f5`（2026-10-16。第 1 弾の門で使った `4e34969` と同じ） |

写しは一時領域の `scratchpad\h3\src\` に浅く取得して読んだ（実行はしていない）。ローカルの `outputs\h3-gate1\oracle\diffsynth_h3\` には DiT 2 本と座標の写ししか無く、VAE とパイプラインは含まれていなかった。

補足: diffusers の H3 は「モジュラーパイプライン」（処理を小さな部品の連鎖で組む方式）だけで、`src/diffusers/pipelines/minimax_h3/` は存在しない。実体は `src/diffusers/modular_pipelines/minimax_h3/`。

---

## ① 結論の表（7 事項 × 3 実装）

| 事項 | ComfyUI | diffusers | DiffSynth-Studio |
|---|---|---|---|
| 1. 平均かサンプルか | **平均**（分布の平均＝moments の前半 24 チャンネル。乱数なし） | **サンプル**（平均＋標準偏差×乱数）。乱数は**固定の種 42** の新しい CPU 生成器（ジョブの種とは無関係） | **平均**（乱数なし） |
| 2. 計算の型 | 符号化器は VAE の型。既定の選び方では対応 GPU なら **fp16**（候補は fp16→f32 の順）。出力の分布は f32 に上げてから平均を取り正規化。潜在は f32 で返る | 符号化器は **f32 に固定**（`_keep_in_fp32_modules` に encoder と quant_conv）。入力も f32。サンプル後に **fp16 に丸めてから f32 に戻し**、その後に正規化。返り値は f32（CPU） | 符号化器は `pipe.torch_dtype`（既定・例ともに **bf16**）。入力は f32 で正規化したあと bf16 に変換。平均の取り出しと正規化は f32、返り値は f32。直後の条件行は **bf16 に丸められる** |
| 3. タイル分割 | 空間のみ・常に有効。タイル 256px・重なり最小 64px（端がぴったり収まるよう 16px 単位で重なりを広げる）。潜在空間で線形ブレンド。1 フレームなので時間の分割は無し | 同じ規則（256／64・16px 単位で拡張・潜在で線形ブレンド）。**既定で有効** | 同じ規則（256／64）。画像の経路（`process_image=True`）では `tiled` 引数に関係なく常にタイル符号化 |
| 4. キャンバスへの合わせ方 | 先頭フレームは**引き伸ばし**（アスペクト無視）。**PIL の LANCZOS**（いったん uint8 に量子化）。値は [0,1]→[−1,1]→[0,1]→**ImageNet の平均・標準偏差で正規化** | 先頭フレームは**引き伸ばし**（PIL LANCZOS と同一と明記）。高さ・幅を省くと先頭画像の縦横比から 768 短辺・768×1344 上限・32 の倍数でキャンバスを決める。値は uint8/255→ImageNet 正規化 | 先頭も末尾も**引き伸ばし**（PIL LANCZOS）。キャンバスは引数（既定 768×1344）で、画像からは決めない。値は uint8/255→ImageNet 正規化 |
| 5. 条件行 | 0.999 混ぜ・時刻 max(t, 0.999)・RoPE の t=L・毎ステップ入れ直し、**すべて一致**。混ぜる乱数は**ジョブの種**の CPU 生成器を条件ごとに作り直したもの（行の形 [行数, 96] で生成・f32）。前向き計算のたびに同じ乱数を作り直して混ぜる | **すべて一致**。混ぜる乱数は**リクエストの生成器**（ジョブの種）から条件ごとに 1 回、潜在の形 [1,24,1,h,w]・f32 で引く。引く順は条件の乱数→映像の初期乱数→音声の初期乱数。1 回だけ作って全ステップで使い回す | **すべて一致**。混ぜる乱数は**ジョブの種**で形 [1,24,T+キーフレーム数,h,w] を **bf16** で引いてから先頭の 1 フレームを切り出す（全キーフレームで同じ乱数）。混ぜる計算も bf16。1 回作って全ステップで使い回す |
| 6. 潜在の正規化 | 有り。VAE 内の `encode` の最後で `(平均 − latents_mean) / latents_std`（f32） | 有り。`encode_vae_condition` の最後で、fp16 に丸めた後に同じ式（f32・CPU） | 有り。`encode_video` の最後で同じ式（f32） |
| 7. T'=1 の扱い | 複製しない。17 フレームへの水増しも時間の分割もせず、1 フレームのまま空間の符号化器へ。出力の時間軸の最後の 1 枚を取る（実質 1 枚）。因果パディングはゼロ | 複製しない（docstring に「繰り返しで 17 に水増しすると学習時の条件と違う」と明記）。1 フレームのまま空間の符号化器へ。因果パディングはゼロ | 複製しない。1 フレームのままタイル符号化し、時間軸の先頭 1 枚を取る。因果パディングはゼロ |

---

## ② 実装ごとの根拠

### ComfyUI（GPL-3.0。振る舞いだけを記す）

**画像の流れ（ノード）**
- `comfy_extras/nodes_minimax_h3.py` の `MiniMaxH3ImageToVideo.execute`（139〜163 行）。`first_frame` は `_resize(first_frame[:1], width, height, "disabled")`（147 行、コメント「geometry anchor: plain stretch to canvas」）。`last_frame` は `"center"`（152 行、アスペクトを保つ切り抜き）。その後 `vae.encode(画像)`（161 行）で潜在にし、`minimax_keyframes` として条件に付ける。
- `_resize`（68〜72 行）は `comfy.utils.common_upscale(..., "lanczos", crop)`。`comfy/utils.py` の `lanczos`（1097〜1105 行）は、0〜1 の値を 255 倍して uint8 に切り詰め、PIL の `Image.Resampling.LANCZOS` で拡縮し、/255 で戻す。
- 先頭フレームは画像の縦横比に関係なくキャンバスに引き伸ばされる。

**公式テンプレート**（`templates/video_minimax_h3_i2v.json`）
- `LoadImage`（id 114）の出力が、そのままサブグラフ「Image to Video (MiniMax H3)」の `first_frame` に入る。サブグラフの中では `MiniMaxH3ImageToVideo`（id 104）に直結しており、`ImageScale` 系のノードも `VAEEncode` ノードも通らない（VAE は `VAELoader` から `MiniMaxH3ImageToVideo` の `vae` 入力へ渡され、ノードの中で `vae.encode` が呼ばれる）。`blueprints/image_to_video_minimax_h3.json` と `templates/video_minimax_h3_i2v_continuation.json` も同じ構成。
- 幅と高さは `ResolutionSelector`（id 115、既定値 `"1:1 (Square)", 0.4, 32`）から来る。入力画像の縦横比からは決まらないので、比が違えば引き伸ばされる。
- 同じテンプレートに `ImageScaleToTotalPixels`（id 119、`nearest-exact`）と `GetImageSize`（id 120）が置かれているが、119 の入力に何もつながっておらず、120 の出力もどこにもつながっていない。実行の経路には入らないと思われる（ComfyUI は出力ノードにつながるノードだけを実行するため。推測）。
- VAE のファイルは `minimax_h3_video_vae_int8_convrot.safetensors`。`comfy/sd.py` 1033〜1034 行のコメントでは int8 量子化の対象は decoder（復号器）とされ、量子化用の演算は `ViT3DDecoder` にしか渡されない。符号化器は通常の `ops` で組まれる（`comfy/ldm/minimax/vae.py` 469〜480 行付近）。

**VAE の包み方と型**（`comfy/sd.py`）
- 1030〜1060 行: H3 の映像 VAE を判別して `MiniMaxH3VideoVAE` を組む。`self.working_dtypes = [torch.float16, torch.float32]`、`self.handles_tiling = True`。`process_input` は上書きしないため、既定の `image * 2.0 - 1.0`（514 行）が使われる。
- 1112〜1115 行付近: `dtype = model_management.vae_dtype(self.device, self.working_dtypes)`。`model_management.vae_dtype`（1290〜1305 行）は、起動引数で fp16／bf16／fp32 が指定されていなければ、候補を順に見て fp16 が使える機器なら fp16 を返す。
- `VAE.encode`（1418〜1474 行）: `vae_encode_crop_pixels`（32 の倍数のキャンバスなら何もしない）→ `[1,3,1,H,W]` に並べ替え（1423〜1425 行）→ `process_input(...).to(self.vae_dtype)`（1438 行。[−1,1] にしてから VAE の型に変換）→ `first_stage_model.encode(pixels_in, device=...)`（1440 行）→ 出力を `vae_output_dtype()`（`intermediate_dtype()`。既定は f32、1273〜1277 行）に変換。
- 確率分布からのサンプルや `DiagonalGaussianDistribution` の類は H3 の経路に登場しない。

**VAE 本体**（`comfy/ldm/minimax/vae.py`）
- `_normalize_pixels`（502〜503 行）: `x.add(1.0).mul_(0.5).sub_(pixel_mean).div_(pixel_std)`。[−1,1]→[0,1]→ImageNet の平均 `(0.485, 0.456, 0.406)`・標準偏差 `(0.229, 0.224, 0.225)` で正規化（17〜18 行）。この計算は VAE の型（fp16）で行われる。
- `encode`（792〜809 行）: `x.shape[2] == 1` なら `_adaptive_encode(_normalize_pixels(x))` を通し、`moments[:, :, -1:]` を取る（800〜801 行）。複数フレームなら `encode_temporal`（17 フレームずつ・不足分は最後のフレームの繰り返しで埋め・最後に 3 潜在を落とす。664〜677 行）。その後 `mean = torch.chunk(moments.float(), 2, dim=1)[0]`（805 行）で**平均だけ**を取り、`(mean - latents_mean) / latents_std`（809 行）を返す。
- `_adaptive_encode`（519〜522 行）は `self.tiling`（既定 True）なら `tiled_encode`。`split_tiles`（531〜553 行）: タイル 256・最小重なり 64、余りを 16px 単位で各重なりに配る。`tiled_encode`（580〜610 行）は潜在空間で `blend`（線形の重み）してつなぐ。`encode_tiled` も内部の規則を使い、外からのタイル指定は無視する（811〜813 行）。
- `CausalConv3d.forward`（78〜119 行）: 時間方向は前だけにゼロを詰める因果パディング。1 フレームのときは `autopad="causal_zero"` でゼロフレーム分のタップを省く（ゼロ詰めと同じ計算と思われる）。空間は reflect。
- fp16 の畳み込みで fp16 累積の専用カーネルを使う分岐（`_fp16_accum_conv`、64〜69 行）は、`torch.backends.cuda.matmul.allow_fp16_accumulation` を利用者が有効にしたときだけ動く（`comfy/ops.py` 1021〜1025 行）。

**条件行**（`comfy/ldm/minimax/model.py`・`comfy/model_base.py`）
- `VISUAL_COND_TIMESTEP = 0.999`（model.py 32 行）。
- `_cond_video_rows`（533〜546 行）: 潜在を f32 で行の形に並べ（patchify）、`torch.Generator("cpu").manual_seed(seed)` で行と同じ形の乱数を f32 で作り、`aug * r + (1 - aug) * noise`。コメント「every condition intentionally restarts the same RNG stream」のとおり、条件ごとに同じ種から作り直す。
- `seed` は `model_base.py` 2248 行 `payload["seed"] = kwargs.get("seed", 0)`。これはサンプラーに渡されたジョブの種（`comfy/samplers.py` 1048 行で `seed=seed` を渡している）。
- `_embed_and_pack`（561〜600 行付近）は前向き計算のたびに `_cond_video_rows` を呼び、条件行の位置に書き込む。つまり毎ステップ入れ直し、乱数も毎回同じ種から同じ値を作り直す。
- 時刻: `seg_t["cond"] = max(t_v, vis_aug)`（677〜681 行）。
- RoPE: `PackedLayout`（354〜395 行）で `cond_t = cursor + FRAME_RESCALE * resolved_frame_index`。`cursor` はテキスト長（参照ブロックが無い FL2VA では L のまま）。先頭フレームは `resolved_frame_index = 0` なので t = L。

### diffusers

**画像の前処理**（`modular_pipelines/minimax_h3/before_encoder.py`）
- `MiniMaxH3ResizeStep.__call__`（115〜164 行）: 高さと幅が無ければ `resolve_canvas_size`（`modular_pipeline.py` 42〜94 行。768 短辺・768×1344 上限・32 の倍数に丸め）で先頭画像の縦横比からキャンバスを決める。
- 先頭画像（index 0）は `image_processor.resize(keyframe, height, width)`（141〜145 行）。コメントに「`resize_mode="default"` is exactly PIL's `resize((width, height), LANCZOS)`, verified pixel-identical」とあり、引き伸ばし。
- 末尾画像はアスペクトを保って拡大し中央を切り抜く（147〜160 行。丸めと中心の取り方を `VaeImageProcessor` の crop とは変えている）。
- 画像が既にキャンバスと同じ大きさなら拡縮しない（138〜139 行）。

**符号化**（`modular_pipelines/minimax_h3/encoders.py`）
- `MiniMaxH3KeyframeVaeEncoderStep.__call__`（361〜381 行）: PIL 画像を uint8 の `[1,3,1,H,W]` にして `encode_vae_condition` を呼ぶ。種は `components.keyframe_encode_seed`。
- `encode_vae_condition`（102〜136 行）:
  - 132 行 `pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std`（ImageNet 正規化）
  - 133〜134 行 `posterior = vae.encode(...)`、`latents = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))`
  - 135 行 `latents = latents.to(torch.float16).float().cpu()`
  - 136 行 `return (latents - latents_mean) / latents_std`
  - docstring（107〜111 行）は「the posterior is *sampled* under a fresh generator seeded independently of the request, and the sampled latent is rounded to float16」と書き、これを「released model's recipe」（公開モデルの作り方）と説明している。
- `keyframe_encode_seed`（`modular_pipeline.py` 278〜286 行）は `return 42`。docstring は「Fixed at 42 in the reference implementation」。
- **申し送りの「seed 42 でサンプルして fp16 に丸める」は、diffusers については事実**。種は `MiniMaxH3ModularPipeline.keyframe_encode_seed` プロパティの固定値 42 で、リクエストの `generator` とは別。
- `DiagonalGaussianDistribution.sample`（`models/autoencoders/vae.py` 703〜712 行）: `randn_tensor(mean.shape, generator, device=parameters.device, dtype=parameters.dtype)`、`mean + std * sample`。生成器は CPU のものなので、乱数は CPU で f32 で作ってから GPU へ送られると思われる（`randn_tensor` の一般的な振る舞い。この関数の中身は今回は読んでいない）。logvar は −30〜20 に切り詰められる（694 行）。

**VAE 本体**（`models/autoencoders/autoencoder_kl_minimax_h3.py`）
- 533〜544 行: `_keep_in_fp32_modules = ["encoder", "quant_conv", "norm1", "norm2", "norm_out", "scale1", "scale2"]`。パイプラインを bf16 で読み込んでも符号化器は f32。
- `encode`（856〜881 行）: `x = x.to(get_parameter_dtype(self.encoder))`（873 行、f32 に合わせる）。
- `_encode`（776〜802 行）: `if num_frames == 1: return self._encode_clip(x)`（787〜788 行）。docstring（778〜783 行）は「Padding it up to `clip_length` by repetition instead ... is not the conditioning MiniMax-H3 was trained with」。
- タイル: `use_tiling = True` が既定（619 行）。256／64（623〜626 行）。`_split_tiles`（656〜677 行）と `_encode_clip`（721〜741 行）は ComfyUI と同じ規則。クラスの docstring（524〜526 行）は「released with tiling enabled ... disabling tiling changes the output」。
- `MiniMaxH3VideoCausalConv3d`（34〜65 行）: 空間は reflect、時間は前にゼロを詰める。

**条件行**（`modular_pipelines/minimax_h3/before_denoise.py`）
- `MiniMaxH3PrepareConditionLatentsStep.__call__`（955〜984 行）: 条件ごとに `randn_tensor(condition.shape, generator=block_state.generator, device, dtype=torch.float32)`（966 行）→ `scheduler.scale_noise(condition, 0.999, noise)`（969 行。`schedulers/scheduling_minimax_h3.py` 221 行で `t * sample + (1 - t) * noise`）→ 行の形に並べる。
- 乱数を引く順は 862〜865 行のコメントのとおり「条件の乱数 → 映像の初期乱数 → 音声の初期乱数」で、すべてリクエストの生成器から引く。
- `MiniMaxH3FL2VAPrepareLatentsStep` の説明（989〜993 行）は「The denoising loop only ever writes the generated rows, so the conditioning rides through every step unchanged」。乱数を混ぜた条件行は 1 回だけ作られ、毎ステップ同じ値が系列に入る。
- 時刻: `max(float(timestep), components.keyframe_noise_aug)`（1261 行）。`keyframe_noise_aug` は 0.999（`modular_pipeline.py` 288〜295 行）。
- RoPE: `anchor == "first"` なら `anchor_time = float(num_text_tokens)`（327〜342 行）。t = L。

### DiffSynth-Studio（Apache-2.0）

**画像の前処理と符号化**（`diffsynth/pipelines/minimax_h3_audio_video.py`）
- `MiniMaxH3Unit_KeyframeEncoder.process`（440〜467 行）:
  - 449 行 `pipe.preprocess_image(img.convert("RGB").resize((width, height), Image.LANCZOS), torch_dtype=torch.float32, min_value=0)`。先頭も末尾も引き伸ばし（切り抜きなし）。値は uint8/255 の [0,1]（`diffsynth/diffusion/base_pipeline.py` 118〜124 行）。
  - 450 行 `pipe.video_vae.encode_video(img_tensor, dtype=pipe.torch_dtype, process_image=True)`
- キャンバスは `__call__` の引数 `height=768, width=1344`（102〜103 行）から決まり、画像の縦横比からは決めない。
- `pipe.torch_dtype` は既定で bf16（27 行・60 行）。int8 ConvRot の FL2VA の例（`examples/minimax_h3/model_inference/MiniMax-H3-Int8-ConvRot-FL2VA.py`）も `torch_dtype=torch.bfloat16`、計算型 bf16。

**VAE 本体**（`diffsynth/models/minimax_h3_video_vae.py`）
- `encode_video`（522〜537 行）: `x = self.processor.transform_tensor(video.to(torch.float32)).to(device, dtype=dtype or out_dtype)`（524 行。f32 で ImageNet 正規化してから bf16 に変換）→ `process_image` なら `self.tiled_encode(x)`（528〜529 行）→ `z = torch.chunk(moments, 2, dim=1)[0]`（**平均**）→ `z[:, :, :1]` → `((z.float() - mean) / std).to(out_dtype)`（535〜537 行。out_dtype は入力の型＝f32）。
- 正規化は `Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))`（335・350 行）。
- `split_tiles`（368〜383 行）・`blend`（385〜394 行）は ComfyUI・diffusers と同じ規則。`tile_size`／`tile_overlap` は `encode_video` の既定値 256／64 が使われる（キーフレーム符号化ではタイルの引数を渡していない）。
- `BaseConv3d._temporal_pad`（58〜64 行）: 1 フレームで `constant` のときは、前に kernel−1 枚のゼロフレームを付ける。

**条件行**
- `self.imgvid_cond_noise_aug = 0.999`（38 行）。
- 455〜466 行: `noise = pipe.generate_noise((1, 24, video_latent_t + len(keyframes), latent_h, latent_w), seed, rand_device, pipe.torch_dtype)[:,:,:1]`。`generate_noise`（`base_pipeline.py` 183〜188 行）は `torch.Generator(rand_device).manual_seed(seed)` で `torch.randn(..., dtype=rand_torch_dtype)`。種はジョブの `seed`（既定 42、106 行）、`rand_device` は既定 `"cpu"`、型は **bf16**。混ぜる計算は `ts * clean + (1 - ts) * noise` を bf16 で行い、**全キーフレームに同じ乱数**を使う。
- `model_fn_minimax_h3`（955 行〜）は呼び出しのたびに `keyframe_cond_anchor` を条件行の位置に `index_copy_` する（1003〜1005 行付近）。値は 1 回作ったものの使い回し。
- 時刻: `timesteps[img_pos[:cond_rows_count]] = max(t_video, imgvid_cond_noise_aug)`（1019 行）。
- RoPE: `_build_packed_fl2va` で `idx == 0` なら `cond_t = float(text_len)`（676〜679 行）。t = L。

---

## ③ 3 実装の違いの要約と、本製品への考察

**3 つとも同じ点（事実）**
- 先頭フレームはキャンバスへ PIL LANCZOS で引き伸ばす（ComfyUI は 0〜1 の値を uint8 に切り詰めてから PIL へ渡す）。
- 画素は [0,1] にしてから ImageNet の平均・標準偏差で正規化する（[−1,1] のままでは入れない）。
- 1 フレームは複製せずに空間の符号化器へ通す。時間方向は前にゼロを詰める因果パディング。
- 空間タイル 256px・最小重なり 64px・16px 単位で重なりを広げる・潜在空間で線形ブレンド。どの実装でも既定で有効。
- 潜在は `(z − latents_mean) / latents_std` で正規化する（値の表も同じ）。
- 条件行は 0.999 で乱数と混ぜ、時刻は max(t, 0.999)、RoPE の時間は t = L、毎ステップ系列に入れ直す。**申し送りの記述は 3 実装すべてで正しい**。

**違う点（事実）**
1. **平均かサンプルか**: diffusers だけが固定の種 42 でサンプルし、fp16 に丸める。ComfyUI と DiffSynth は平均。
2. **符号化器の型**: diffusers は f32、ComfyUI は fp16（既定の選び方の場合）、DiffSynth は bf16。
3. **0.999 で混ぜる乱数**: 3 つとも「ジョブの種」から作るが、作り方がばらばら。ComfyUI は行の形・f32・条件ごとに同じ種から作り直し。diffusers は潜在の形・f32・リクエストの生成器から最初に引く。DiffSynth は大きめの形を bf16 で引いて先頭を切り出し、全キーフレームで共有。同じ種でも 3 者の間でビット一致はしない。
4. **キャンバスの決め方**: diffusers は先頭画像の縦横比から自動で決める。ComfyUI（テンプレートでは ResolutionSelector）と DiffSynth は利用者の指定。縦横比が違えば 3 つとも引き伸ばす。
5. **末尾フレーム**（I2V では関係ないが参考）: ComfyUI と diffusers はアスペクトを保って切り抜く（丸め方は異なる）。DiffSynth は引き伸ばす。

**本製品が「ComfyUI と同等以上」を出すうえで効く点（考察・推測を含む）**
- 平均かサンプルかの差は、標準偏差の分だけ潜在が揺れる量に相当する。条件行はさらに 0.1% の乱数と混ぜられるので、相対的にはサンプル側の揺れのほうが大きい可能性がある（大きさは未計測）。ComfyUI と同等を目指すなら**平均を取る**のが最短で、決定的でもある。diffusers が「公開モデルの作り方」としてサンプル＋fp16 丸めを再現しているので、それに合わせる余地は残る。ただし MiniMax 公式のコードは今回読んでいないので、それが本当に公式の作り方かは未確認。
- 型は、ComfyUI の fp16 より f32 で回すほうが「同等以上」になると考えられる（diffusers は f32 に固定している）。1 枚の画像なら符号化器を f32 で回しても VRAM の負担は小さいと思われる（未計測）。
- タイル分割（256／64・線形ブレンド）は 3 実装とも同じで、diffusers は「切ると出力が変わる」と明記している。**同じ規則で入れることが必須**と考えられる。
- 拡縮は PIL LANCZOS の引き伸ばしと、uint8 を経由する値の流れを揃えるのが安全。ComfyUI の公式テンプレート（キャンバスを画像の縦横比に合わせない）では引き伸ばしで形が歪むことがあるが、それも ComfyUI と同じ振る舞いになる。
- ImageNet 正規化を忘れて [−1,1] のまま入れると、3 実装のどれとも合わなくなる。
- 0.999 で混ぜる乱数の作り方は、3 実装でビット単位では揃わない。影響は 0.1% の混ぜ込みなので小さいと思われる。決定性（同じ種で同じ結果）さえ守れば、どの方式でも精度の差は小さいと推測する。

---

## ④ 未確認事項

- MiniMax 公式の推論コード（もしあれば）で、平均とサンプルのどちらを使うか、fp16 に丸めるか。diffusers は「reference implementation で 42 固定」と書いているが、その参照元は今回読んでいない。
- `minimax_h3_video_vae_int8_convrot.safetensors` の中で、符号化器の重みがどの型で入っているか（ComfyUI は符号化器を量子化しない作りだが、ファイルの中身は見ていない）。
- ComfyUI で fp16 になるかどうかは、機器ごとの `should_use_fp16` の判定と起動引数（`--fp32-vae` など）に依存する。本製品の機（RTX 4070 Ti SUPER）での実際の型は確かめていない（GPU を使わない調査のため）。
- diffusers の `randn_tensor` が CPU の生成器のとき、CPU で作って GPU へ移すかどうか（一般的な振る舞いとしては知られているが、今回のコミットでは関数本体を読んでいない）。
- 平均とサンプル、fp16・bf16・f32 の違いが、最終的な映像にどれくらい効くか（計測していない）。
- テンプレートの `ImageScaleToTotalPixels`（id 119）が実行されないこと（グラフのつながりから判断しただけで、ComfyUI で実行して確かめてはいない）。
