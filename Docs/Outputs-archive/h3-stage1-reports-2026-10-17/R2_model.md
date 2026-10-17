# R2: MiniMax H3 の自前ワーカー実装に向けたモデル構造と推論手順の調査

調査日 2026-10-17。調査担当 R2。コードは書いていない（ヘッダ解析用の小さなスクリプトだけ `scratchpad\h3\tools\` に置いた）。何もインストールしておらず、GPU も使っていない。

**読んだもの（すべて浅い clone を読んだだけで、実行はしていない）**

| 名前 | 置き場所（`scratchpad\h3\src\` の下） | 版 |
|---|---|---|
| DiffSynth-Studio（Apache-2.0） | `DiffSynth-Studio` | `4e34969`（2026-10-16） |
| diffusers（Apache-2.0）。Windows ではチェックアウトが一部失敗したので、H3 関係の 41 ファイルだけを `git show` で取り出した | `dx`（取り出した分）・`diffusers`（git 本体） | `d42ebf9`（2026-10-15） |
| ComfyUI（GPL-3.0）。仕様書として読んだだけで、コードは写していない | `ComfyUI` | `e087d8f`（2026-10-16） |
| MiniMax-AI/MiniMax-H3（GitHub。HF の重み以外を写したもの） | `MiniMax-H3` | `d21241f`（2026-08-15） |

このほかに、Comfy-Org の workflow_templates の JSON 3 本、docs.comfy.org の H3 の手引き、SGLang の cookbook、ComfyUI の issue #15416 を取得して読んだ（置き場所は `scratchpad\h3\dl\`）。safetensors はヘッダだけを HTTP Range で取得した（`dl\hdr\`）。

表記の決まり: 【事実】はファイル名と行番号、または URL で確認したこと。【推測】は根拠からの推論。行番号は上の版のもの。

---

## (1) 結論（論点 3 つ）

### 論点 1: DiffSynth-Studio は移植するか、版を固定した依存として入れるか → **移植を勧める**（必要な部品だけを Apache-2.0 の表示つきで写す）

- 【事実】H3 の本体部品は小さい。DiT が `diffsynth/models/minimax_h3_dit.py` の 401 行、枝刈り版の差分が `minimax_h3_dit_comfy.py` の 78 行、映像 VAE が 621 行、音声 VAE が 481 行、テキストエンコーダの包みが 231 行（transformers の `Qwen3VLModel` を呼ぶだけ）。
- 【事実】ただし `diffsynth/__init__.py` が `from .core import *` を実行し、`core/__init__.py` が core の全体（attention・data・gradient・quant・loader・vram・device・offload_training。計 4,227 行）を読み込む。そのため、モデル部品を 1 つ import するだけで core 全体が動き、modelscope・pandas・imageio・huggingface_hub・einops・torchvision・transformers・safetensors の import が必要になる。
- 【事実】依存として入れても、自前で差し替える部分が多い。attention は区間ごとの Python ループ（`minimax_h3_dit.py:70-81`）、オフロードは DiffSynth 独自の VRAM 管理（`core/vram`）、int8 は comfy-kitchen 経由（CUDA 13 未満では CUDA 拡張を無効にする。`comfy_kitchen.py:205-206`）。
- 【推測】移植する量は、DiT と枝刈り（約 480 行）・2 つの VAE（約 1,100 行）・状態辞書の変換（約 70 行）・系列の組み立て（パイプラインの `_build_packed_fl2va` 周辺の約 130 行）で、合計 2,000 行弱。attention・オフロード・int8 Linear は自前（または comfy-kitchen を直接）に置き換えるので、DiffSynth の core は不要になる。

### 論点 2: Comfy-Org の VAE と MiniMax 公式の VAE は同じ重みの詰め替えか → **詰め替えである（構造の範囲で確定。データ本体は比べていない）**

- 【事実・ヘッダ実測】映像 VAE は、Comfy-Org の `vae/minimax_h3_video_vae_fp16.safetensors`（562 本）と公式 `FL2VA/video_vae/source/model.safetensors`（560 本・F32）で、560 本の名前と形状がすべて一致する。dtype は F16 と F32 の違いだけで、データのバイト数の比はちょうど 2.0。Comfy 側に余分にあるのは `latents_mean`／`latents_std`（F16 [24]）の 2 本だけ。
- 【事実・ヘッダ実測】音声 VAE は、Comfy-Org の `vae/minimax_h3_audio_vae_fp32.safetensors`（917 本・F32）と公式 `FL2VA/audio_vae/model.safetensors`（1,087 本・F32）を比べた。共通は 743 本で形状はすべて一致する。差は weight_norm の畳み込みだけで、公式の `*.weight_g`／`*.weight_v` の 172 組が、Comfy では `*.weight` の 1 本ずつに合成されている（形状は `weight_v` と同じ）。Comfy 側の残りの余分は `latents_mean`／`latents_std` の 2 本。
- 【事実】Comfy の `__metadata__` にある `latents_mean`／`latents_std` の値は、公式 `vae/config.json`・`audio_vae/config.json` の値と完全に一致する。
- 【事実】公式の FL2VA と Ref2VA の VAE は、映像も音声も sha256 が同じ（同じファイル）。公式の diffusers 形式 `vae/`（3 分割・703 本・F32）は名前の付け替え版で、データのバイト数は元形式と 8,192 バイトしか違わない（DiffSynth が捨てる `decoder.mask_token` の分とみられる。この内訳は推測）。音声の diffusers 形式 `audio_vae/diffusion_pytorch_model.safetensors` は、元形式と全テンソルのオフセットまで一致する。
- 未確認: 重みの値そのもの（F32 を F16 に丸めたものと一致するか）は、データ本体を落とさない制約のため比べていない。

### 論点 3: ガイダンス蒸留（negative prompt も CFG も使わない）は本当か → **本当である。1 ステップにつき前向き計算は 1 回、CFG は 1.0**

- 【事実】MiniMax 公式の README の 191 行目に「The released checkpoints are CFG-distilled Omni Transformer model weights.」とある。
- 【事実】diffusers は `modular_pipelines/minimax_h3/modular_pipeline.py:166-167` で「guidance-distilled … no guider, no `negative_prompt` and no `guidance_scale`, and every step runs exactly one forward pass」と書いている（`denoise.py:55-56`・`encoders.py:146` にも同じ記述）。ガイダンス用の部品も、2 回目の前向き計算もない（`denoise.py:111-133` は前向き 1 回）。
- 【事実】DiffSynth は `cfg_scale` の既定が 1.0（`pipelines/minimax_h3_audio_video.py:108`）。`cfg_scale != 1.0` のときだけ 2 回目を回す（`diffusion/base_pipeline.py:339`）。文書（`docs/en/Model_Details/MiniMax-H3.md:102,109`）にも「This model is CFG-distilled … keeping the default value is recommended」とある。さらに学習用に「蒸留を保つ逆 CFG」の引数がある（同 :224）。
- 【事実】Comfy-Org 公式の workflow_templates（`video_minimax_h3_t2v.json`・`_i2v.json`・`_r2v.json`）と、ComfyUI 本体の blueprint は、どれも `BasicGuider`（CFG なし）を使っている。CFGGuider や KSampler の cfg は現れない。
- 【事実】SGLang の cookbook にも「CFG distillation removes the negative branch; it is not the few-step timestep distillation used by Turbo releases」とある。
- 「cfg 4 の例が複数ある」という件: 上の公式テンプレートと手引きの中には見つからなかった。コミュニティ配布の workflow 由来とみられる（推測）。蒸留済みのモデルに cfg 4 をかけると、計算が 2 倍になるうえ、意図しない過剰ガイダンスになると考えられる（推測）。

---

## (2) 調査結果 A〜I

### A. DiffSynth-Studio の H3 コードの棚卸し

**H3 固有のファイル（パス・行数・役割・diffsynth 内部への依存）**

| ファイル | 行数 | 役割 | diffsynth 内部への依存 |
|---|---|---|---|
| `diffsynth/models/minimax_h3_dit.py` | 401 | DiT の本体（非枝刈り版。qkv はヘッドごとの交互配置） | `core.attention.attention_forward`（`core/attention/attention.py` 255 行）・`core.gradient`（72 行） |
| `diffsynth/models/minimax_h3_dit_comfy.py` | 78 | Comfy 形式（qkv は `[q;k;v]` の積み重ね）と枝刈り版（`adaln_t_table`） | 上の dit |
| `diffsynth/models/minimax_h3_text_encoder.py` | 231 | Qwen3-VL を 50 層で構築する包みと、提示文（presentation）の組み立て | なし（transformers だけ） |
| `diffsynth/models/minimax_h3_video_vae.py` | 621 | 映像 VAE（3D CNN エンコーダと ViT デコーダ。タイルと時間方向の分割つき） | `core.attention`・einops・torchvision |
| `diffsynth/models/minimax_h3_audio_vae.py` | 481 | 音声 VAE（DAC エンコーダと BigVGAN デコーダ。weight_g/v の形式） | `core.attention` |
| `diffsynth/pipelines/minimax_h3_audio_video.py` | 1079 | パイプライン（系列の組み立て・条件付け・ループ） | core・`diffusion.FlowMatchScheduler`（`flow_match.py` 500 行）・`base_pipeline.py`（542 行）・controlnet・vdn（822 行）・utils |
| `diffsynth/models/minimax_h3_dit_vdn.py`／`_vdn_fp8.py` | 822／196 | VDN（ハイブリッド attention）。α版の範囲外 | — |
| `diffsynth/models/minimax_h3_controlnet.py` | 105 | 範囲外 | — |
| `diffsynth/utils/state_dict_converters/minimax_h3_dit.py` | 7 | `model.diffusion_model.` の接頭辞を剥がす（Singularity 系の配布物向け） | — |
| `diffsynth/utils/state_dict_converters/minimax_h3_text_encoder.py` | 45 | テキストエンコーダの名前の変換（公式形式と Comfy 形式） | — |
| `diffsynth/utils/state_dict_converters/minimax_h3_video_vae.py`／`_audio_vae.py` | 11／9 | `mask_token` を捨てる・`register_tokens`／`scale1`／`scale2`／`q_bias` などに `.weight` を付ける | — |
| `diffsynth/utils/lora/minimax_h3.py` | 189 | LoRA（範囲外） | — |
| `diffsynth/core/quant/backends/comfy_kitchen.py` | 318 | `comfy_quant` を読む int8／fp8 のバックエンド | `core/quant/base.py`（243 行）・`config.py`（570 行） |

- 登録表は `diffsynth/configs/model_configs.py:1517-1610`。Comfy-Org の枝刈り int8 は `MiniMaxH3DiTComfyPruned` で、`quant_config` は `comfy_kitchen_int8_w8a8`・`load_prequantized`。除外は patch_proj・condition_proj・`adaln_proj.linear`・final の出力 2 本・token_refiner の 8 本（:1561-1567）。テキストエンコーダは `MiniMaxH3TextEncoderComfyOrgStateDictConverter` で、除外は視覚塔の `qkv`／`proj`／`linear_fc1`／`linear_fc2`（:1540-1547。名前の照合は「完全一致または `.名前` で終わる」。`core/quant/config.py:306-311`）。
- **ライセンス表示**: リポジトリは Apache-2.0（`LICENSE`・`pyproject.toml:11`）。ファイル単位の表示は `minimax_h3_video_vae.py` と `minimax_h3_audio_vae.py` の 1 行目の `# SPDX-License-Identifier: Apache-2.0` だけで、ほかの H3 ファイルには表示がない。
- **「本体のモデル部品だけ使う」場合に import せざるを得ない範囲**: 論点 1 のとおり core の全体（4,227 行）と、その先の第三者パッケージ（modelscope・pandas・imageio など）。`diffsynth.models.minimax_h3_dit` を単独で import する経路はない。

**`comfy_kitchen.py` が `comfy_quant` を読んで Linear を組む手順（要点）**

1. `unflatten_state_dict`（:236-255）: 状態辞書のうち `.comfy_quant` で終わるキーを集め、層ごとに次の処理をする。
2. `_parse_marker`（:180-190）: U8 テンソルをバイト列にして JSON として読む。`format` が無いか、`int8_tensorwise`／`float8_e4m3fn` 以外ならエラーにする。
3. `{層}.weight`（I8）と `{層}.weight_scale` を取り出す（どちらかが無ければエラー）。
4. `TensorWiseInt8Format.params_from_marker`（:138-148）: `convrot` と `convrot_groupsize` を、最上位か入れ子の `params` から読む（既定 256）。ConvRot なのに `in_features % groupsize != 0` ならエラーにする。返す値は `is_weight=True, convrot, convrot_groupsize`。
5. `Params(scale=weight_scale, orig_dtype=bf16, orig_shape=qdata.shape, …)` を作り、`QuantizedTensor(qdata, "TensorWiseINT8Layout", params)` を `{層}.weight` に入れ直す。
6. Linear 側: `create_quantized_linear_shell` → `_build_linear`（:291-299）で、meta デバイス上の `ComfyKitchenLinear(nn.Linear)`（:21-36）を作り、その weight に QuantizedTensor を載せる。forward は入力を contiguous にして `F.linear` を呼ぶだけで、活性化の動的 int8 化と回転は comfy-kitchen 側の dispatch が行う。
7. `validate_environment`（:199-206）: `torch.version.cuda < 13.0` なら `ck.registry.disable("cuda")`。

**枝刈り版の時刻埋め込みの正確な式**（`minimax_h3_dit_comfy.py:34-45`・`:9-14`。ComfyUI の `comfy/ldm/minimax/model.py:742-747` も同じ振る舞い）

```
t      = 1 - sigma                         # H3 の時刻。t=1 が無ノイズ、t=0 が純ノイズ
G      = adaln_t_table.shape[0]            # 1025
pos    = clamp(t, 0, 1) * (G - 1)          # 0..1024 の実数位置。float32 で計算
i0     = min(floor(pos), G - 2)            # t=1.0 のときも最後の区間 [1023,1024] に留める
w      = pos - i0
e(t)   = table[i0] * (1 - w) + table[i0+1] * w      # torch.lerp。形状 [M, 8]（M = 異なる時刻の数）
mod    = adaln_proj.linear(e(t))           # SiLU をかけない（非枝刈り版は linear(silu(time_embedder(t)))）
```

- `adaln_proj.linear` は `[96768, 8]`＋bias（F16）。最終層は `final_layer.adaln_proj.linear` の `[10752, 8]`＋bias（F16。ヘッダで実測）。
- ComfyUI は枝刈り版の AdaLN を float32 で計算している（`model.py:492-493` の `adaln_dtype=torch.float32`・`apply_silu=False`）。

**状態辞書の変換表（Comfy-Org 形式 → 内部名）**

- DiT（Comfy の積み重ね形式）: DiffSynth はテンソル名を変えずにそのまま読む（接頭辞なしの場合）。違いは forward の解釈だけ。
  - `attn.qkv_proj.weight [21504, 5376]`: Comfy は行方向に `[q(7168); k(7168); v(7168)]`（`dit_comfy.py:20` の `view(total, 3, heads, head_dim)`）。MiniMax 公式の分割ファイルは**ヘッドごとの交互配置** `[head0: q k v, head1: q k v, …]`（`dit.py:139` の `view(total, heads, 3, head_dim)`）。diffusers の `single_file_utils.py:4319-4366` も同じことを述べ、2 つはファイルからは見分けられないとしている。
  - `mlp.fc1.weight [28672, 5376]`: `[gate(14336); value(14336)]`。出力は `silu(gate) * value`（`dit.py:160-162`）。diffusers の SwiGLU は `[value; gate]` を読むので、入れ替えている（`single_file_utils.py:4368-4371`）。
  - int8 の `weight_scale [N, 1]` は出力行ごとのスケールなので、行を `chunk(3)` で分けるときはスケールも同じ行で分ければよい（推測。ConvRot は入力の列方向にかかるので、行の並べ替えとは独立）。
  - diffusers の名前の対応（参考。`single_file_utils.py:4336-4350`）: `blocks.`→`transformer_blocks.`、`token_refiner.blocks.`→`token_refiner.refiner_blocks.`、`video_patch_proj`→`proj_in`、`audio_patch_proj`→`audio_proj_in`、`condition_proj`→`context_embedder`、`final_layer.norm`→`norm_out.norm`、`final_layer.adaln_proj.linear`→`norm_out.linear`、`final_layer.video_out`→`proj_out`、`final_layer.audio_out`→`audio_proj_out`、`q_norm`/`k_norm`→`norm_q`/`norm_k`、`out_proj`→`to_out.0`、`fc1`→`ff.net.0.proj`、`fc2`→`ff.net.2`。`rope.inv_freq` は捨てて、設定から計算し直す。`adaln_t_table` があると ValueError で拒否する（:4327-4332）。
- テキストエンコーダ（`MiniMaxH3TextEncoderComfyOrgStateDictConverter`、`state_dict_converters/minimax_h3_text_encoder.py:37-52`）:
  - `visual.*` → `model.visual.*`
  - `model.layers.N.*` → `model.language_model.layers.N.*`（N ≥ 50 は捨てる）
  - `model.embed_tokens.*` → `model.language_model.embed_tokens.*`
  - それ以外はそのまま。`lm_head`・`model.norm` はファイルに無い。内部では `language_model.norm = Identity` にしている（`minimax_h3_text_encoder.py:73`）。

### B. DiT の前向き計算

**構成**（設定値は公式 `transformer/config.json`。形状は Comfy 枝刈り int8 のヘッダで実測）

| 部品 | 形状・dtype | 中身 |
|---|---|---|
| `video_patch_proj` | [5376, 96] F32 | 映像の行（24ch × 1×2×2 = 96）→ 隠れ幅 |
| `audio_patch_proj` | [5376, 32] F32 | 音声の行（32ch）→ 隠れ幅 |
| `condition_proj` | [5376, 5120] BF16 | Qwen3-VL の 5120 → 5376 |
| `token_refiner.blocks.{0,1}` | BF16 | テキストだけの前置ノルム型 Transformer。AdaLN も RoPE も無い。self-attention はテキスト行の中だけ。その後に `final_norm` |
| `blocks.{0..49}` | qkv・out・fc1・fc2 が I8 | 下の構造 |
| `final_layer` | norm BF16・adaln F16・video_out [96,5376] F32・audio_out [32,5376] F32 | RMSNorm＋shift/scale の変調 → 出力 2 本 |
| `rope.inv_freq` | [16] F32 | `1/10000^(i/16)`, i=0..15 |
| `adaln_t_table` | [1025, 8] F32 | 枝刈り版の時刻曲線 |

**ブロック 1 つ**（`dit.py:214-234`、diffusers `transformer_minimax_h3.py:347-373`）

```
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_proj(t_emb)  # 各 [M*3, 5376]
h = RMSNorm1(x) * (1 + scale_msa[idx]) + shift_msa[idx]
x = x + gate_msa[idx] * Attn(h)
h = RMSNorm2(x) * (1 + scale_mlp[idx]) + shift_mlp[idx]
x = x + gate_mlp[idx] * MLP(h)
idx = (時刻の番号) * 3 + (モダリティタグ)     # タグは 0=映像, 1=テキスト, 2=音声
```

- **Attention**: qkv_proj 5376→21504、q/k/v は各 7168 = 56 ヘッド × 128。q と k にヘッドごとの RMSNorm（128 次元・eps 1e-5）をかけてから RoPE。softmax のスケールは 128^-0.5。出力 7168 → `out_proj` で 5376 に戻す。
- **隠れ幅が 5376 なのに attention の内部次元が 7168 である理由**: 【事実】ヘッド数とヘッド次元が隠れ幅と独立に決められていて、56×128 = 7168 > 5376（diffusers の docstring `transformer_minimax_h3.py:401-402`「`num_attention_heads * attention_head_dim` is *larger* than `hidden_size`」）。5376 = 42×128。qkv の出力 21504 は 3×7168。
- **MLP**: fc1 は 5376→28672（gate 14336 と value 14336）、SwiGLU、fc2 は 14336→5376。
- **norm1／norm2**: RMSNorm（学習される重みつき・eps 1e-5）。
- **96768 = 6 × 5376 × 3 の内訳**: 6 は `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`、3 はモダリティ（映像・テキスト・音声）。出力の並びは `view(M*3, 6*5376)` なので、行の順は「時刻 m ごとに [映像, テキスト, 音声]」、各行の中は 6 つの 5376 の順（`dit.py:178-183`）。重みの行で言えば、0〜32255 が映像、32256〜64511 がテキスト、64512〜96767 が音声で、それぞれの中が 6 分割されている。
- **最終層**: 10752 = 2 × 5376（shift, scale の順）× モダリティ 1。添字は時刻の番号だけ（`dit.py:237-252`、diffusers `:134-157`）。出力の 2 本は全行に計算して、映像の行と音声の行を選び出す。
- **出力の意味**: 「データ方向の速度」v で、x0 = x_t + σ·v。DiffSynth は符号を反転して返す（`pipelines/…:1079` の `return -v_video, -v_audio`）。ComfyUI も同じく反転する（`model.py:790`）。diffusers はそのまま返し、スケジューラ側で `x0 = sample + sigma*v` とする（`scheduling_minimax_h3.py:249`）。
- **token_refiner の役割**: `condition_proj` で 5376 にしたテキスト列だけに、2 層の通常の Transformer（RMSNorm 前置・self-attention・SwiGLU）をかけて、最後に RMSNorm する（`dit.py:186-211`）。時刻にも位置にも依存しないので、**ループの前に 1 回だけ計算すればよい**（推測。ComfyUI の `preprocess_text_embeds`（`model.py:518-522`）はこの前提で先に計算している）。

**位置符号化（RoPE）**

- 【事実】3 軸（t, h, w）で `inv_freq` の 16 本を共有する。角度は `[t·f(16), h·f(16), w·f(16)]` を 2 回つなげた 96 次元で、rotate_half（前半 48 と後半 48 の対）でヘッド次元 128 のうち先頭 96 を回す。残りの 32 次元は回さない（`dit.py:47-58, 84-102`、diffusers `:59-100`）。
- 座標の決め方（`pipelines/…:586-709`、diffusers `before_denoise.py:36-133, 308-373`。どちらも float64 で作る）:
  - **テキスト**: t = 0, 1, …, L−1。h = w = 0。
  - **映像（目標）**: t は L から始まり、潜在フレームの間隔は `5/3 × (1,4,4,4,4)` を周期として累積する（L, L+5/3, L+5/3+20/3, …）。h と w は、`np.linspace(left, left+ratio, dim/2, endpoint=False) × 32`。ここで `ratio = dim/√(H_lat·W_lat)`、`left = (1−ratio)/2`（縦横比で正規化した中央寄せの格子）。
  - **先頭フレームの条件行**: t = L（目標の最初の潜在フレームと同じ）。h と w は目標と同じ格子。末尾フレームの場合は、t = L + Σspan − 5/3（Σspan は numpy の pairwise 和で計算する）。
  - **音声**: t = L + k（k は音声潜在の番号。40 個/秒で、24fps×5/3 と同じ時計）。h = 0。w はチャンネル 0 なら w 格子の最小値、チャンネル 1 なら最大値。行はチャンネル順に並ぶ（ch0 の全時刻、続いて ch1）。

**系列の並びと各区間の長さ（FL2VA と T2V）**

```
[ text (L) | cond (K × (H/32)(W/32)) | audio (2 × A) | video (T' × (H/32)(W/32)) ]
L  = 提示文のトークン数（I2V では画像トークンを含む）
K  = キーフレームの数（T2V は 0、先頭フレームだけの I2V は 1）
A  = round(F / 24 × 40)                 # F は 17n+5 に揃えたフレーム数
T' = 5n + 2
```

- DiffSynth は合計を 64 の倍数に詰め物し、`cu_seqlens=[0, used, seq_len]` にしている（`pipelines/…:612-614, 708`）。diffusers は詰め物をしない（`transformer_minimax_h3.py:388-391`）。
- 各行の時刻: テキストと映像は t_v、音声は t_a、条件行は max(t_v, 0.999)。`torch.unique(sorted=True, return_inverse=True)` で M 個（1〜3 個）にまとめて、時刻の番号を作る（`pipelines/…:1011-1021`、diffusers `before_denoise.py:1198-1236`）。
- 1344×768・124 フレームの例: 映像 37×24×42 = 37,296 行、音声 414 行、テキストは数百行。I2V では条件行 1,008 行と、テキスト側の画像トークン約 1,010 行が加わる。

### C. attention の形（修正点 2）

- 【事実】diffusers: `transformer_minimax_h3.py:160-209` の `MiniMaxH3AttnProcessor.__call__`。q/k/v は (B, S, H, D) で、`dispatch_attention_fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=…)` に渡す（:196-205）。コメント（:193-195）に「packs one request into a single attention document, so the model passes no mask and every attention backend stays available」とある。`dispatch_attention_fn` の定義は `models/attention_dispatch.py:400`、標準の SDPA 経路は `:859-897`（(B,H,S,D) に並べ替えて SDPA を呼ぶ）。
- 【事実】attention は**テキスト・条件行・音声・映像のすべての間で全結合**（区間ごとの制限はない）。diffusers のクラスの説明（:382-385）に「Attention is full self-attention over that sequence; there is no cross-attention」とある。ComfyUI も `optimized_attention(…, mask=None)`（`model.py:200`）。token_refiner だけはテキスト行の中に閉じている（別の計算なので）。
- 【事実】DiffSynth の `_sdpa_varlen_attention`（`dit.py:70-81`）は `cu_seqlens` の区間ごとに SDPA を回す Python ループ。区間は `[0, used)` と詰め物 `[used, seq_len)` の 2 つだけなので、実データの部分は diffusers と数学的に同じ（全結合）。違いは、詰め物の分の計算、`.tolist()` による GPU 同期、区間ごとのコピー。
- **SageAttention と SDPA**: マスクなし・因果なし・ヘッド次元 128・bf16 の 1 本の系列なので、`sageattn(q,k,v, tensor_layout="HND")` も `F.scaled_dot_product_attention` も、そのまま差し込める形である（推測。diffusers は sage 系のバックエンドを公式に選べる。docs.comfy.org の手引きも KJ の Sage パッチで動くとしている）。スケールの既定（1/√128）も同じ。

### D. テキストエンコーダ

- **50 層目の後の正規化前の隠れ状態の取り方**:
  - DiffSynth: `Qwen3VLModel` を `num_hidden_layers=50` で構築し、`language_model.norm = Identity` にして、`last_hidden_state[0]` を取る（`minimax_h3_text_encoder.py:16-106`）。
  - diffusers: 64 層の `Qwen3VLForConditionalGeneration` の `.model` を `output_hidden_states=True` で呼び、`hidden_states[50]` を取る（`encoders.py:63-99`）。50 層以下に切り詰めたモデルを渡すと拒否する（:64-70。最後の hidden_state は正規化後になるため）。Comfy の TE ファイルは、メタデータの `output: unnormalized_hidden_after_layer_50` どおり DiffSynth と同じ方式。
  - Qwen3-VL の deepstack（視覚の特徴を浅い層に足し込む仕組み。`deepstack_visual_indexes [8,16,24]` の出力を言語側の先頭の層に加える）は 50 層の中で起きるので、50 層で切っても結果は変わらない（推測）。
- **テンプレート**: 【事実】chat template も system prompt も特殊トークンも**使わない**。`tokenizer(prompt, add_special_tokens=False)` で得たトークン列をそのまま使う（DiffSynth `:157-160`、diffusers `encoders.py:145, 194`、ComfyUI `text_encoders/minimax.py:3-17`）。
  - FL2VA の提示文は `"<Picture 1>: "` ＋ `<|vision_start|>` ＋ `<|image_pad|>`×N ＋ `<|vision_end|>` ＋ prompt（DiffSynth `:163-169`、diffusers `encoders.py:285-300`）。
  - モダリティタグは、ラベルの文字列が 1（テキスト）、視覚ブロック（start・pad・end を含む）が **0（映像）**。このタグは DiT の AdaLN の選択に使われる。
- **特殊トークン**: 【事実】公式の `tokenizer_config.json` の `additional_special_tokens` に `<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>`, `<|caption_end|>` がある。`tokenizer.json` 側の added_tokens は 151668 までしかない。transformers で読み込むと、これらが 151669〜151675 として追加される（ComfyUI は `MINIMAX_EXTRA_TOKENS` で同じ番号を明示している。`text_encoders/minimax.py:31-34`）。公式のプロンプト例は `<d>[English] …</d>` の形でせりふを書く。
- **トークン数の上限とパディング**: 【事実】DiffSynth と diffusers は、パディングも切り詰めもしない（attention_mask は全部 1）。`model_max_length` は 262144。ComfyUI は「テキストの区切りが 1 バッチに収まらないとエラー」にしている（`text_encoders/minimax.py` の `add_text`）。
- **視覚塔（`visual.*`）は使われる**: 【事実】FL2VA（I2V）では、キーフレームの画像を Qwen3-VL の image_processor（平均・標準偏差 0.5、patch 16、merge 2）に通し、視覚塔で埋め込んで提示文に差し込む（DiffSynth `pipelines/…:304-307`、diffusers `encoders.py:276-281`、ComfyUI `nodes_minimax_h3.py:147-158` が `clip.tokenize(prompt, images=images)` で同じ画像を渡す）。Comfy-Org の TE ファイルに 1.19GB の視覚塔が入っているのはこのため。T2V では使わない。
  - 画像はキャンバスの大きさ（W×H）に合わせてから TE に渡す。画像トークンの数は (H/16)(W/16)/4 で、768×1344 なら 1,008。
- **text_dim 5120 の流れ**: Qwen の hidden 5120 [L, 5120] → `condition_proj` で [L, 5376] → token_refiner 2 層 → final_norm → 系列の text 区間に入る。

### E. 条件付け（I2V の先頭フレーム）

- **前処理**:
  - キャンバス（指定がなければ）は、先頭画像の縦横比から、短辺 768・面積の上限 768×1344・32 の倍数に丸めて決める（diffusers `modular_pipeline.py:42-94`、`before_encoder.py:128-134`）。
  - 先頭画像はキャンバスへ**引き伸ばし**（LANCZOS、縦横比は保たない。`before_encoder.py:140-145`。ComfyUI も `_resize(…, "disabled")` で同じ）。末尾画像は縦横比を保った cover-crop。
- **VAE での潜在化**:
  - diffusers（`encoders.py:102-136`）: 画素を [0,1] にして ImageNet の平均と標準偏差で正規化する → 1 フレームだけなので、空間エンコーダだけを通す（時間方向の分割はしない。タイルあり）→ **事後分布からシード 42 の専用の生成器でサンプル**する → float16 に丸める → `(z − latents_mean)/latents_std`。
  - DiffSynth（`minimax_h3_video_vae.py:522-537`）: 事後分布の**平均**を使う（サンプルせず、fp16 にも丸めない）。この点は 2 つの実装で違う。diffusers は「released model の手順」と明記している。
- **ノイズの混ぜ方**: cond = 0.999·clean + 0.001·noise（`keyframe_noise_aug = 0.999`。diffusers `modular_pipeline.py:287-295`・`before_denoise.py:962-971`、DiffSynth `:440-466`）。乱数の取り方は実装ごとに違う。diffusers はリクエストの生成器から、映像のノイズより先に取る。DiffSynth はシードで (1,24,T'+K,h,w) を作って先頭を使う。ComfyUI は条件ごとにシードで CPU 生成器を作り直す（`model.py:533-546`）。
- **行と時刻**: patchify した 1 フレーム分の行（(H/32)(W/32) 行）を、系列の cond 区間に置く。その行の時刻は max(t_v, 0.999)、タグは 0（映像）。マスクやフラグのテンソルは**無い**。条件であることは、時刻（ほぼ無ノイズ）と RoPE の座標（t = L）だけで区別される。
- **ループの中での扱い**: 条件行は毎ステップ同じ値を入れ直し、更新しない（diffusers `denoise.py:229-234` は `[num_condition_video_rows:]` だけを step する。DiffSynth は出力から `cond_rows_count` 分を捨てる `:1074`）。目標の映像の先頭の潜在フレームは、置き換えずにモデルに生成させる。
- **末尾フレームがない場合**: cond 区間の 1 ブロックだけ（K=1・anchor "first"）。**T2V では cond 区間が無い**（K=0）。
- テキスト側: 前の D のとおり、画像を視覚塔に通した提示文になる。

### F. VAE

**映像 VAE**（公式 `vae/config.json`、diffusers `autoencoder_kl_minimax_h3.py`）

- 構成: エンコーダは因果的な 3D CNN（ch 128、倍率 (1,2,2,4,4,8)、空間の縮小 (2,2,2,2,1,1)、時間の縮小 (1,2,2,1,1,1)）で、空間 16 倍・時間 4 倍。潜在は 24ch。デコーダは ViT（36 層・32 ヘッド×64・レジスタトークン 4・RoPE の theta 100）。
- 画素の扱いは ImageNet の平均と標準偏差による正規化で、[−1,1] ではない（`:517-519`）。潜在は `latents_mean`／`latents_std` の 24 値で正規化する。
- **17n+5 の導出**（`:604-609, 776-853`）:
  - `clip_length=17`、時間の縮小 4。前置の詰め物 `(-17) % 4 = 3` で、1 チャンクは 20 フレーム、潜在は `ceil(17/4) = 5`。
  - エンコード: 17 の倍数まで末尾のフレームを繰り返して詰め、チャンクごとに 5 潜在を作り、最後に `token_drop=3` を捨てる。F = 17n+5 なら (n+1) チャンクで 5n+5 潜在になり、3 を捨てて **T' = 5n+2**。式は `T' = (F − 5)/17 × 5 + 2`（`modular_pipeline.py:116-130`、DiffSynth `:246`）。
  - デコード: 5+2 = 7 潜在の窓（`token_overlap = (-3) % 5 = 2`）で n 回解く。各窓の前半から 17 フレームを取り、窓の重なりの 5 フレーム（`frame_overlap = 2×4 − 3 = 5`）を線形に混ぜる。最後に重なりの 5 フレームを足すので、出力は **17n+5 フレーム**になる。つまり 17n+5 は、デコーダが余りなく出す長さである。
  - フレーム数の揃え方: `align_num_frames` は、`F % 17 == 5` になるまで 1 ずつ増やす（`modular_pipeline.py:97-113`）。
- **上限 345 フレームの理由**: 【事実】diffusers は、揃えた後の長さが 5 秒以上 15 秒以下であることを検査する（`before_denoise.py:399-412`。min_duration 5.0、max_duration 15.0）。17n+5 ≤ 360 となる最大は n=20 の 345 フレーム（14.375 秒）。n=21 の 362 フレームは 15.083 秒で弾かれる（:402-403 のコメント）。下限は n=7 の 124 フレーム。公式 README は「4–15 seconds」と書いていて、diffusers の下限 5 秒とは食い違う。ComfyUI のツールチップは「trained range is ~124-362」。

| n | フレーム | 秒 | 映像の潜在 T' | 音声の潜在 A（チャンネルごと） |
|---|---|---|---|---|
| 7 | 124 | 5.167 | 37 | 207 |
| 11 | 192 | 8.000 | 57 | 320 |
| 14 | 243 | 10.125 | 72 | 405 |
| 20 | 345 | 14.375 | 102 | 575 |

- **入出力の形**:
  - エンコードは [1,3,F,H,W]（ImageNet 正規化済み）→ moments [1,48,T',H/16,W/16] → (mean, logvar)。
  - デコードは [1,24,T',H/16,W/16]（正規化を戻したもの）→ [1,3,17n+5,H,W]（ImageNet 空間）→ `x·std + mean` → [0,1] に clamp。
- **タイル**: 【事実】公式は既定で**タイルあり**（タイル 256px・重なりの最小 64px・重なりを 16 刻みで広げて画面をちょうど覆う・線形の混合）。diffusers は「released frames are the blended-tile ones, so disabling tiling changes the output」としている（`:521-526, 656-677`）。公式の `FL2VA/video_vae/config.json` も `vae_encoder_tiling: 1`, `vae_decoder_tiling: 1`, `vae_tile_size: 256`, `vae_tile_overlap_min: 64`。
- **ComfyUI issue #15416（タイルの縞）**: 【事実】起票者が後から範囲を訂正している。縞（タイル 31.4、タイルなし 93.7 の誤差）が出るのは、**潜在 1 フレーム（T'=1、17n+5 の格子から外れた形）を解いたときだけ**。正しい長さの動画を解くのは clean だった（往復の誤差 3.92）。回避方法は「1 フレームの潜在は 2 つに複製して解き、最初の画素フレームだけを使う」。issue は 2026-10-17 の時点で open。
  - 自前実装への含意（推測）: 本番の解読は 17n+5 の長さだけを扱えば問題は出ない。プレビューなどで 1 潜在フレームだけを解くときは、2 つに複製する。
- 【事実】ComfyUI の int8 版の映像 VAE は、デコーダの 36 ブロックだけが int8（ノート §13.2）。fp16 版はヘッダの実測で、全 562 本が F16。

**音声 VAE**（公式 `audio_vae/config.json`、diffusers `autoencoder_kl_minimax_h3_audio.py`）

- DAC 型のエンコーダ（rates 2,4,4,5,5、hop 800）、因果 attention の射影（2048→32）、BigVGAN デコーダ（rates 5,5,2,2,2,2,2）。32kHz・モノラルの VAE を 2 チャンネル分、バッチ 2 として使う。
- 潜在は [2, 32, A]（チャンネル・特徴・時間）で、A = round(F/24 × 40)（40 潜在/秒。`modular_pipeline.py:133-147`）。32 値の `latents_mean`／`latents_std` で正規化する。
- デコード結果は A×800 サンプル。124 フレームなら 207×800 = 165,600 サンプル = 5.175 秒で、映像の 5.1667 秒より最大で半潜在（12.5ms）ぶん長い（推測。mux のときに映像の長さで揃える必要がある）。
- Comfy 版は weight_norm を合成済みなので、`weight_g/weight_v` を前提にした DiffSynth や diffusers の実装に Comfy 版を読ませるには変換が要る（推測。g = ‖W‖（出力チャンネルごと）、v = W とすれば元の重みに戻る）。

### G. サンプラ

- **時刻列**（DiffSynth `flow_match.py` の `set_timesteps_minimax_h3`、SGLang の cookbook でも同じ数え方）:
  - `base = linspace(1, 0, N+1)[:-1]`、`σ = s·b / (1 + (s−1)·b)`。s は映像が 12、音声が 3。モデルに渡す時刻は t = 1 − σ。
  - **N ステップ = N 回の評価**で、終点の 0 は含まない。
- **diffusers との数え方の違い**: 【事実】diffusers の `MiniMaxH3Scheduler.set_timesteps` は `linspace(1, 0, N)` で終点の 0 を含むので、N を指定しても評価は **N−1 回**になる（`scheduling_minimax_h3.py:111-153`、文書 `minimax_h3.md:77`）。SGLang の cookbook は「Reference implementations that count sigma points need N+1 to reproduce this」と書いている。
- **ステップ 20 のときの σ**（DiffSynth と SGLang の数え方。21 点・評価 20 回）:
  - 映像（s=12）: 1.0000 0.9956 0.9908 0.9855 0.9796 0.9730 0.9655 0.9571 0.9474 0.9362 0.9231 0.9076 0.8889 0.8660 0.8372 0.8000 0.7500 0.6792 0.5714 0.3871 → 0
  - 音声（s=3）: 1.0000 0.9828 0.9643 0.9444 0.9231 0.9000 0.8750 0.8478 0.8182 0.7857 0.7500 0.7105 0.6667 0.6176 0.5625 0.5000 0.4286 0.3462 0.2500 0.1364 → 0
- **Euler の更新**（eta=0）: `x0 = x_t + σ·v`、`x_{next} = r·x_t + (1−r)·x0`（r = σ_next/σ）で、`x + (σ − σ_next)·v` と同じ。diffusers は float32 で計算する（`scheduling_minimax_h3.py:249-256`）。
- **2 本の時刻列を 1 つのループで回す方法**: 同じ基底の格子の i 番目について、映像の σ_v[i] と音声の σ_a[i] を同時に使う。1 回の前向き計算で、映像の行は t_v、音声の行は t_a として扱う（AdaLN の添字で分かれる）。出力をそれぞれの σ 列で別々に Euler 更新する（DiffSynth `pipelines/…:184-198`、diffusers `denoise.py:229-240`）。
  - ComfyUI は映像の σ だけをサンプラに渡し、音声の σ を `time_shift_sigma` で閉じた式から導く（`model.py:36-39`）。base = σ_v / (s_v + σ_v(1−s_v)) → σ_a = s_a·base / (1 + (s_a−1)·base)。基底の格子が同じなら、2 本立てと同じになる。
- **seed と決定性**: 【事実】乱数の取り方は実装ごとに違い、同じ seed でも実装の間では一致しない。
  - diffusers: 1 つの生成器から、条件のノイズ → 映像 (1,24,T',h,w) fp32 → 音声 (2A,32) fp32 の順に取る（`before_denoise.py:861-887`）。
  - DiffSynth: 呼ぶたびにシードし直す。映像は bf16 で直接生成し、音声は (2,32,A)（`base_pipeline.py:183-188`、`pipelines/…:245-250`）。
- **ガイダンス**: (1) の論点 3 のとおり、CFG なし・1 回の前向き計算。
- 参考: ComfyUI の公式テンプレートは `res_multistep` サンプラ（2 次の多段法）と `simple` スケジューラで、ステップは 20（Turbo LoRA ありでは 8 または 4）。参照の実装（diffusers・SGLang・DiffSynth）は Euler で、既定は 50 ステップ。

### H. ComfyUI の判定の振る舞い（仕様として。コードは写していない）

**H3 と判定する条件**: `{接頭辞}video_patch_proj.weight` と `{接頭辞}audio_patch_proj.weight` の両方があること（`comfy/model_detection.py` の H3 の節）。

**構成を形状から導く規則**

| 項目 | 導き方 |
|---|---|
| num_layers | `blocks.N.` の N の数 |
| token_refiner_num_layers | `token_refiner.blocks.N.` の数 |
| hidden_size | `video_patch_proj.weight` の shape[0] |
| latents_dim | `final_layer.video_out.weight` の shape[0] ÷ 4（patch 1×2×2 を前提） |
| audio_latents_dim | `final_layer.audio_out.weight` の shape[0] |
| attention_head_dim | `blocks.0.attn.q_norm.weight` の shape[0] |
| num_attention_heads | `blocks.0.attn.qkv_proj.weight` の shape[0] ÷ (3 × head_dim) |
| ffn_hidden_size | `blocks.0.mlp.fc1.weight` の shape[0] ÷ 2 |
| text_dim | `condition_proj.weight` の shape[1] |
| 枝刈り版 | `adaln_t_table` があれば、格子数＝shape[0]、ランク（time_embed_dim）＝shape[1]。time_embedder は作らない。AdaLN の計算を float32 にし、SiLU をかけない |
| 非枝刈り版 | `time_embedder.proj_in.weight` の shape[1]＝周波数の次元、shape[0]＝中間の幅、`proj_out` の shape[0]＝time_embed_dim |
| rope_inv_freq_len | `rope.inv_freq` の shape[0]（値もファイルのものを使う） |
| VSA 版 | `blocks.0.attn.to_gate_compress.weight` があれば VSA。通常の計算では使わない |
| 上書き | `__metadata__["config"]` があれば、JSON の `["transformer"]` の中身で上書きする（`transformer` キーが無ければ何もしない） |
| PDD ヘッド | `video_out` の行数が out_features の倍数（n>1）なら、ヘッドの束として扱う（特殊な蒸留版向け。n=1 なら通常） |

- シフトの既定: `supported_models.py` の H3 の `sampling_settings` が shift 12.0・audio_shift 3.0。

**接頭辞の剥がし方**（`unet_prefix_from_state_dict`）: 候補は `model.diffusion_model.`・`model.model.`・`net.`。各候補で始まるキーの数を数え、最多の候補が 5 本を超えればそれを接頭辞とする。どれも 5 本以下なら `model.` とする（特殊な 2 例を除く）。H3 の判定は、この接頭辞を付けたキー名で行う。

**`convert_old_quants`**（`comfy/utils.py:1441-1499`）:
- メタデータに `_quantization_metadata` があれば、その JSON の `layers` の各項目を `{層}.comfy_quant`（JSON のバイト列の U8 テンソル）として状態辞書に書き足す。
- 無い場合でも、`scaled_fp8` テンソルがあれば旧式の fp8 とみなす。`.scale_weight` を `.weight_scale` に、`.scale_input` を `.input_scale` に改名し（値が 1.0 の input_scale は捨てる）、層ごとに `{"format":"float8_e4m3fn"}` を作って同じく書き足す。

**`comfy_quant` の読み方**（`comfy/ops.py:1236` 以降）:
- 層の `comfy_quant` を JSON として読む。読むキーは `format`（必須）・`full_precision_matrix_mult`（任意）。`int8_tensorwise` のときは、さらに `convrot`・`convrot_groupsize`（既定 256）を、最上位か入れ子の `params` から読む。**読むキーは format・convrot・convrot_groupsize の 3 つだけ、という前提には `full_precision_matrix_mult` が加わる**。それ以外のキー（`per_row`・`orig_dtype` など）は無視する。
- `int8_tensorwise` は `weight_scale` が必須（無ければエラー）。`comfy_quant` が無い層は、通常の浮動小数点の重みとして読む。

### I. 公式の推奨値（一次資料）

| 項目 | 値 | 出典 |
|---|---|---|
| 尺 | 4〜15 秒 | 公式 README:73。diffusers は 5〜15 秒で検査する（`modular_pipeline.py:243-251`）。既定は 124 フレーム |
| fps | 24 | README:76、diffusers `modular_pipeline.py:32` |
| 解像度 | 短辺 768（既定）。各辺は 32 の倍数。面積の上限 768×1344。縦横比は 1:4〜4:1 | README:75、diffusers `modular_pipeline.py:42-94, 231-236`・`before_denoise.py:185` |
| 音声 | 32kHz ステレオ | README:77 |
| shift | 映像 12・音声 3 | 公式 `scheduler/scheduler_config.json`・`audio_scheduler/…`・`FL2VA/model_index.json` の `sigma_shift_scales` |
| ステップ数 | **公式の模型カードには記載がない**。SGLang の cookbook の例・DiffSynth の既定は 50（N+1 点の数え方）。diffusers の既定は 50（N 点の数え方なので評価は 49 回）。ComfyUI の公式テンプレートは 20（`res_multistep`）で、手引きは「12〜16 で単純な場面は保つ、細部は約 50 まで改善」 | 上の各出典 |
| CFG | なし（CFG 蒸留済み） | README:191 |

---

## (3) 自前実装で間違えやすい点

1. **qkv の行の並び**: Comfy 形式は `[q;k;v]` の積み重ね、公式の分割ファイルはヘッドごとの交互配置。ファイルからは見分けられない。Comfy-Org の int8 を読むなら積み重ね形式に決め打ちする。
2. **fc1 の順**: `[gate; value]` で、silu をかけるのは前半。diffusers 方式（`[value; gate]`）に合わせて並べ替えてはいけない（自前で `chunk` するなら並べ替えは不要）。
3. **枝刈り版では SiLU をかけない**: `adaln_proj(lerp(table))` に silu を挟むと壊れる。非枝刈り版だけが `linear(silu(time_embedder(t)))`。最終層も同じ。
4. **時刻の向き**: AdaLN と表に渡すのは t = 1 − σ（1 が無ノイズ）。σ や σ×1000 を渡さない。表の添字の上限の clamp（`min(floor, G−2)`）を忘れると、t=1.0（条件行が t=0.999 から最終ステップで 1 に近づく場合など）で範囲外を読む。
5. **AdaLN の添字**: `時刻の番号 × 3 + タグ`。タグは 0=映像・1=テキスト・2=音声。**視覚ブロックのトークン（start・pad・end）はテキスト区間にあってもタグ 0**。最終層の添字は時刻の番号だけ。
6. **出力の符号**: モデルの出力は「データ方向の速度」（x0 = x_t + σv）。一般的な flow-match の Euler（x0 = x − σv）をそのまま使うと逆に進む。
7. **ステップ数の数え方**: diffusers は N 点（評価 N−1 回）、DiffSynth と SGLang は N+1 点（評価 N 回）。比べるときは揃える。
8. **映像と音声の時刻列は別**: 同じ添字 i で σ_v[i]（shift 12）と σ_a[i]（shift 3）を使う。音声に映像の σ を流用しない。
9. **RoPE の座標は float64 で作る**: 映像の時間の間隔は一様でない（5/3×(1,4,4,4,4)）。空間は縦横比で正規化して ×32。音声は w の両端に固定。テキストの長さ L が全メディアの時刻を押し出す。回す次元は 128 のうち先頭 96（rotate_half）で、残りの 32 は回さない。
10. **テキストエンコーダ**: 50 層目の後の**正規化前**。chat template も特殊トークンも付けない。`tokenizer_config.json` の追加特殊トークン（`<d>` など）を読み込む（`tokenizer.json` だけを使うと `<d>` が普通の BPE に割れる）。I2V では視覚塔を使い、画像トークンは DiT の系列にも入る（系列が約 1,000 行伸びる）。
11. **token_refiner はテキストの中だけ**: 時刻に依存しないので、ループの前に 1 回だけ計算してよい。
12. **条件行**: 0.999 のノイズ混ぜ・時刻は max(t_v, 0.999)・毎ステップ入れ直して更新しない・RoPE の t は L。映像の先頭の潜在フレームは置き換えない。
13. **VAE の画素の規約**: ImageNet の正規化（[−1,1] ではない）。潜在は 24 値と 32 値の mean/std で正規化する。キーフレームは 1 フレームとして空間エンコーダだけに通す（17 フレームに複製してはいけない）。
14. **VAE のタイルは既定でオン**（256／64）。潜在 1 フレームだけを解くと縞が出るので、2 つに複製する（#15416）。
15. **フレーム数**: 17n+5（124〜345）。T' = 5n+2、A = round(F/24×40)。音声は最大 12.5ms 長くなるので、mux のときに切る。
16. **Comfy の音声 VAE は weight_norm を合成済み**。Comfy の映像 VAE の `latents_mean/std` テンソルは F16 なので、値は config（F32）かメタデータの JSON から取るのが安全（推測）。
17. **精度の島**: patch_proj・出力の 2 本・時刻の部分は F32 で計算し、ブロックは bf16 にする（diffusers の `_keep_in_fp32_modules`、ComfyUI も同じ）。枝刈りの AdaLN（8→96768）は小さいので float32 で計算してよい。
18. **attention は 1 本の文書として全結合**。区間マスクは不要。詰め物を入れるなら、詰め物の行を実データの行から見えないようにする（DiffSynth は別区間にしている）。
19. **キーフレームのエンコード**: diffusers は「シード 42 でサンプル → fp16 に丸める」、DiffSynth は平均を使う。どちらに合わせるかを決めておく（diffusers が公式の手順と明記している）。

## (4) 不明点

- 公式の重み（F32）を F16 に丸めたものと Comfy の fp16 VAE の値が一致するかは、データ本体を落とさない制約のため未確認（ヘッダの名前・形状・バイト数の比・mean/std は一致）。確かめるなら、小さなテンソル（例: bias の数 KB）を Range で両方から取って比べる方法がある（データ本体の一部を取ることになるので、実施は判断を仰ぐ）。
- 公式の推奨ステップ数は、模型カードに一次の記載がない。50（SGLang・DiffSynth・diffusers の既定）と 20（ComfyUI の公式テンプレート）が並んでいる。
- 「ComfyUI の既定ワークフローに cfg 4 の例が複数ある」の出所は特定できなかった（公式の workflow_templates の 3 本と blueprint は、すべて BasicGuider）。
- 尺の下限は、公式 README が 4 秒、diffusers が 5 秒（124 フレーム）で食い違う。4 秒台の 17n+5 は 107 フレーム（4.458 秒）。
- 非枝刈り int8 の `__metadata__.config` が `{"transformer": {...}}` の入れ子かどうか（ComfyUI の上書きは入れ子の場合だけ効く）は、今回ヘッダを開いて確かめていない。
- ComfyUI の `text_encoders/minimax.py` で、テキストの区切りの長さの上限が実際にいくつかは追っていない。
- 枝刈りの表は FL2VA と Ref2VA で値が違う（ノート §13.2）。Hybrid の配布物の表がどちらと整合するかは、範囲外として調べていない。
- diffusers の `hidden_states[50]` と DiffSynth の「50 層で切って最終 norm を Identity にする」方式は、deepstack の注入も含めて数値的に一致するはずだが（推測）、実測はしていない。
