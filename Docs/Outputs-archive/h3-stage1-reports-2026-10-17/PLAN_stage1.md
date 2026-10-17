# MiniMax H3 対応・α版 第 1 弾の計画（台帳 §1-93）

作成 2026-10-17。監督が、調査 R1（venv スタック）・R2（モデル構造と参照実装）・R3（本製品の受け皿）の 3 報告（`scratchpad\h3\R1_stack.md`・`R2_model.md`・`R3_inventory.md`）とオーナー裁定 1〜21（`Docs/MINIMAX_H3_RESEARCH_NOTES.md` §10）を突き合わせて書き、敵対的レビュー（Opus。Critical 1・Major 6・Minor 14。§7）を反映した版。**この計画の了承は、裁定 20 の「ファイル群の取得と仮想環境の構築の 1 回の了承」と、第 1 弾の GPU 実測（§3.5 の G2〜G5）のまとめての了承を兼ねる。**

---

## 1. 結論（何を作り、何で合格とするか）

**第 1 弾の到達点**: 新しい仮想環境 `.venv-engine-h3` の中で、新しいワーカー `engine_h3/worker.py` を `--selftest-generate` で動かし、公式 FL2VA 枝刈り int8 ConvRot の重み一式から **T2V（文章から映像と音声）を 1 本**生成して mp4 に書き出す。API・アダプタ・記述子・操作パネル・インストーラは触らない（第 2 弾以降）。

**合格の条件（G5）**: 640×384・124 フレーム（17n+5 の最小 n=7）・24fps・20 ステップ・固定シードで、音声つき mp4 が 1 本できること。絵が破綻していないことは監督が確認し、オーナーの目視は第 2 弾（操作パネル）でまとめて行う。

**設計の骨格（裁定 16・17 を受けた形）**

| 部位 | 決めたこと | 根拠 |
|---|---|---|
| int8 の実行 | 重みは**素の int8 の `nn.Parameter`**・スケールはファイルどおり f32 `[N,1]` のバッファ・Linear の forward で **`comfy_kitchen.int8_linear`** を呼ぶ。量子化テンソル型（`QuantizedTensor`）は使わない | R1 2-2 追加: 量子化テンソルはラッパー型で本製品の先読みに乗らない。関数 API は同じ CUDA カーネルに届く |
| 経路の固定 | 起動時に `comfy_kitchen.disable_backend("triton")` と **`disable_backend("eager")`** の両方を切り、CUDA 経路に `int8_linear` が無ければロードを拒否。制約から外れた呼び出しは `NoCapableBackendError` で止まる（fail loud） | R1 2-3（issue #136 は triton 経路だけ）・レビュー M-1（eager は黙って受けて遅くなるだけで区別できない） |
| モデルの組み立て | DiT も TE も **`torch.device("meta")` の中で組み**、量子化層を `H3Int8Linear`（meta）に差し替え、テンソルを 1 本ずつ読んで `load_state_dict(assign=True)` で入れる。読み残しの meta テンソルが無いことを検査する | レビュー C-1: 普通に組むと DiT f32 約 77GB・TE bf16 約 49GB を確保して 64GB を超える |
| CPU オフロード | 既存の `engine/transformer/block_swap_service.py`＋`block_swap_prefetch.py` をそのまま使う（H3 の本体ブロック列の名前は `blocks` で候補名に当たる）。推論の囲みは **`engine_h3` 全経路 `torch.no_grad()`** | R3 項目 2。素のテンソルなら既存の形でそのまま乗る。囲みの規約は設計書 §5.8（§3-124 の実例） |
| 仮想環境 | torch 2.13.0+cu130・torchvision 0.28.0+cu130・comfy-kitchen 0.2.37・triton-windows 3.7.1.post27・sageattention 2.2.0 post6（cu130 版）・transformers 5.14.1・safetensors 0.8.0・**einops 0.8.2・av 18.1.0**。torchaudio は入れない | R1 結論＋レビュー m-9（移植元の映像 VAE が einops を使う。av は LTX 2.5 の venv で実績のある版） |
| 実装の土台 | DiffSynth-Studio の H3 部品を**移植**（Apache-2.0 の表示つき。約 2,000 行弱）。attention は diffusers 型（1 本の系列・マスクなし）で SDPA／SageAttention。パッケージとしては入れない | R2 論点 1: `import diffsynth` で core 全体と modelscope などが芋づる |
| 映像 VAE・音声 VAE | Comfy-Org の `minimax_h3_video_vae_fp16.safetensors`（計算は bf16）・`minimax_h3_audio_vae_fp32.safetensors`（計算は f32）。`latents_mean`／`std` は `__metadata__` の JSON（f32）から取る | R2 論点 2: 公式と名前・形状・mean/std が一致。音声は weight_norm 合成済み。精度はレビュー m-5 |
| ガイダンス | CFG なし・1 ステップ 1 回の前向き計算 | R2 論点 3: 公式 README・diffusers・DiffSynth・Comfy 公式雛形が一致 |
| ステップ数と shift | **ワーカーの `generate` は最初から `num_inference_steps`・`shift_video`・`shift_audio` を受け取る**（第 2 弾で操作パネルと Gradio に 3 本のスライダーを出すため。裁定 22）。第 1 弾のセルフテストの既定値は**ステップ 20・Euler**・shift 映像 12／音声 3・「N ステップ＝N 回評価」。**20 は監督の選択**で、Comfy 公式雛形の 20 は別のサンプラ（`res_multistep`）の値、参照実装の Euler の既定は 50。G5 で絵が甘ければ 50 で 1 本比べる。製品の既定値は第 2 弾の着手時にオーナーが検討 | 裁定 22（2026-10-17 差し戻し: UI への露出は第 2 弾。背景＝Turbo を融合した低ステップの配布物が多く、LoRA 未対応でも速く生成できる見込み）。レビュー m-11 |

---

## 2. 弾の分け方（α版の全体像。第 2 弾以降は着手時に別の計画で詰める）

| 弾 | 範囲 | 触る場所 |
|---|---|---|
| **第 1 弾（本計画）** | 仮想環境・重みの取得・`engine_h3/` と `h3_format.py` の新設・T2V セルフテスト 1 本・設計正本 | `engine_h3/`・`h3_format.py`（新設）・`Docs/MINIMAX_H3_DESIGN.md`（新設）・`Docs/MULTI_ENGINE_DESIGN.md` の訂正の引用・VERIFICATION_LOG §165 |
| 第 2 弾 | I2V（先頭フレーム＝視覚塔と条件行）のセルフテスト → アダプタ・系統登録・記述子・ヘッダ判定（`h3_format.py` をアプリ側の事前検査から呼ぶ）・API の制約（17n+5／24fps 固定／32 の倍数）・**API にステップ数と shift（映像・音声）の 3 項目**・操作パネル（ドロップダウンは自動追随・フレーム数のスナップ・快適上限マーカーの非表示・featureScope・**ステップ数／Shift（映像）／Shift（音声）の 3 本のスライダー＝既存の CFG スライダーのモックを差し替え。デザインは CFG スライダーを踏襲し、項目名・スライダー本体・数値を 1 行に並べる〔「幅」のように 3 行にまたがる形は避ける〕**）・**Gradio にも同じ 3 本**・MCP の最小対応・テスト・常駐（`cache_weights` 相当）の要否。**3 本の既定値は第 2 弾の着手時にオーナーが検討** | `services/engines/minimax_h3/`・`services/engines/__init__.py`・`services/model_registry.py`・`config.py`・`scripts/manifests/40-minimaxh3.json`・`api/models.py`・webui・`gradio_ui/`・`mcp_server/` |
| 第 3 弾 | `install-MiniMaxH3.bat`（venv 構築＋取得を 1 回で。UETrack が前例）・`Ensure-EngineVenv` の 3 回目の呼び出し・freeze・README の要件行（ドライバ R580 以上・Turing 以降・容量）・CLOSED | `scripts/install_ltx.ps1`・`scripts/install_model.ps1`・`README.md` |
| 以後 | §1-95 快適上限の較正（m×n ≥ 2³¹ の試験はここ）・§1-97 末尾と中間キーフレーム。**§1-96（ステップ数と shift の表示）は第 2 弾に吸収する**（了承後に台帳を直す） | — |

第 2 弾の「API の制約」は、設計書 §7 が「エンジン別の UI パラメータ制約は作らない」としてきた領域に踏み込む（R3 項目 7。8n+1 と 17n+5 の両方を満たす値は 73・209・345・481 の 4 つだけ）。**第 2 弾の計画で正面から扱う**。

---

## 3. 第 1 弾の作業項目

### 3.1 文書（コードより先に書く。了承後すぐ）

1. **`Docs/MINIMAX_H3_DESIGN.md` を新設**（設計正本）。中身: 位置づけ／**設計の裁定だけ**を移管（3・6・7・8・9・10・11・16・17・18・19。進め方の裁定 1・2・4・5・12・13・14・15・20・21 は研究ノート §10 に残し参照で結ぶ＝レビュー m-13）／3 段の推論／部品の一覧と `engine_h3/` のファイル構成／int8 の実行方式と block swap への乗せ方／meta での組み立て／ヘッダの判定（テンソル名の指紋）と吸収する揺れの範囲／フレーム規約（17n+5・T'=5n+2・A=round(F/24×40)）／定数／`load`・`generate` の分担（第 1 弾は load＝パスの検査とヘッダの判定だけ、generate が毎回組む）／進捗の段階名は LTX の語彙を借りる（単段＝`stage1_denoise`）／ワーカーのメッセージのキー名（§3.3）／tokenizer はカテゴリではなく記述子の `assets` で渡す予定／弾の分け方と門／未決事項。**裁定と書いた規則は【オーナー裁定】の印を付ける**。版の数字は書かず freeze を指す。
2. **`Docs/MULTI_ENGINE_DESIGN.md`** に 1 行ずつの訂正の引用（中身は設計正本の節を指すだけ）: §2.2・§2.4（テンソル名の指紋による判定を H3 で採る）・§5.3（venv の一覧・「全エンジンで torch を揃える【オーナー裁定】」の例外・容量が約 3.5〜4GB 増える）・§5.8（推論モードの表に `engine_h3`＝`no_grad`）・§9.2（推論パイプライン一式の行）。
3. **`Docs/MINIMAX_H3_RESEARCH_NOTES.md` §10** に**裁定 22 を追記**（ステップ数と shift の UI 露出は第 2 弾へ。3 本のスライダー・CFG スライダーのモックの差し替え・1 行のデザイン・既定値は第 2 弾の着手時。背景＝Turbo を融合した低ステップの配布物。裁定 18 のうち「画面に出さない」は取り下げ）し、設計の裁定を「設計正本 §x を見よ」の参照に置き換える（進め方の裁定は残す）。**台帳**は §1-96 を §1-93 の第 2 弾へ吸収して削除し、§1-93 の「α版の完結条件」の文面を直す（3 行目の最終更新にも 1 句）。
4. **`engine_h3/engine_h3-venv-pyproject.toml`**（手動再現用の依存の仕様。「なぜこの版か」を書く。版の正本は freeze）と、構築後の **`engine_h3/venv-engine-h3.freeze.txt`**。

### 3.2 仮想環境と重みの取得（監督がバックグラウンドで実行。コードと並行）

- **仮想環境**: 本番のピン一式（§1 の表。diffsynth と accelerate は入れない）で **dry-run をもう一度回してから**（GPU 不要。R1 の dry-run は diffsynth を含む組だった）、`uv venv .venv-engine-h3 --python 3.12` → `uv pip install --python .venv-engine-h3\Scripts\python.exe --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match <ピン一式>`。構築後に freeze を取り、容量を実測して記録する（R1 の見込みは 3.5〜4GB）。
- **重みの取得**（約 54GB。`.venv-engine` の `huggingface_hub` の CLI＝本製品のインストーラと同じ道具）。置き場は第 2 弾の記述子と同じ配置:
  - `models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors`（Comfy-Org/MiniMax-H3 `diffusion_models/`・20.97GB）
  - `models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors`（同 `text_encoders/`・27.14GB）
  - `models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors`（5.21GB）・`models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors`（0.61GB）
  - `models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer/`（MiniMaxAI/MiniMax-H3 `FL2VA/text_encoder/` の `tokenizer.json`・`tokenizer_config.json`・`vocab.json`・`merges.txt`・`preprocessor_config.json`・`config.json`。計約 11.5MB。`<d>` などの追加特殊トークンは `tokenizer_config.json` にしか無い）
  - 両リポジトリともゲート無し（2026-10-17 確認。Comfy-Org `e5eb578…`・公式 `42ed227…`）。開発機の空き約 133GB で足りる。

### 3.3 新設するコード（1 ファイル 1 所有者。Opus 3 体）

| ファイル | 役割 | 由来・再利用 | 担当 |
|---|---|---|---|
| **`h3_format.py`**（リポジトリ直下・torch 非依存） | `sft_quant_format.read_header`・`read_tensor_bytes` の上に: **H3 の判定（`video_patch_proj.weight`＋`audio_patch_proj.weight` の同時存在）**、接頭辞の剥がし（無し／`model.diffusion_model.`／`model.model.`／`net.`）、形状から構成を導く（層数・幅・ヘッド数・ffn・`adaln_t_table` の格子数 G とランク・枝刈りか否か）、`comfy_quant` の読み（`format`・`convrot`・`convrot_groupsize` だけ。他のキーは無視）、量子化層の一覧。**アプリ側（torch 無し）の事前検査から第 2 弾で呼ぶ**ので、`sft_quant_format.py`・`chain_math.py` と同じ置き場 | ComfyUI の振る舞いを仕様として自前で書く（R2 H）。レビュー M-5 | C3 |
| `engine_h3/__init__.py` | 空 | — | — |
| `engine_h3/sft_load.py` | `h3_format` の判定結果に従い、`engine/sft_quant/sft_reader.read_tensors` で**テンソルを 1 本ずつ seek＋readinto で読み**（mmap しない）、meta で組んだモデルへ `load_state_dict(assign=True)`。読み残しの meta が無いことを検査。知らないテンソルは警告ログを出して捨てる | R3 項目 1・4 | C3 |
| `engine_h3/quant_linear.py` | `H3Int8Linear(nn.Module)`: `weight` int8 `[N,K]` の Parameter（requires_grad=False）・`weight_scale` f32 `[N,1]` のバッファ・任意の bias。forward は `comfy_kitchen.int8_linear(x, weight, weight_scale, bias, out_dtype=bf16, convrot, convrot_groupsize)`。方式はモジュール属性に持つ。`require_cuda_int8()`（`"int8_linear" in ck.list_backends()["cuda"]["capabilities"]` を確かめ、`disable_backend("triton")`・`disable_backend("eager")`） | R1 2-2 追加 (b)(d)・レビュー M-1 | C1 |
| `engine_h3/dit.py` | DiT 本体（50 ブロック・token_refiner 2 ブロック・patch_proj・final_layer・RoPE）。**meta で組む**。**枝刈り版の AdaLN**（`t=1−σ`、`pos=clamp(t)·(G−1)`、`i0=min(floor,G−2)`、`lerp` → `adaln_proj.linear`、SiLU 無し、float32）。qkv は `[q;k;v]` の積み重ね、fc1 は `[gate;value]`。attention は 1 本の系列・マスク無しで `F.scaled_dot_product_attention`、`attention_backend="sage"` のときは `sageattn`。token_refiner はループの前に 1 回だけ。精度の島（patch_proj・出力 2 本・AdaLN は f32、ブロックは bf16） | DiffSynth `minimax_h3_dit.py`＋`_comfy.py` の移植。diffusers `transformer_minimax_h3.py` を照合先に | C1 |
| `engine_h3/rope.py` | 3 軸 RoPE。**座標は float64 で作り、角度と cos／sin は float32**（参照実装と同じ）。テキスト t=0..L−1・映像は L から 5/3×(1,4,4,4,4) 周期・空間は縦横比で正規化して ×32・音声は w の両端・回すのは 128 のうち先頭 96 | R2 B・レビュー m-7 | C1 |
| `engine_h3/video_vae.py` | 3D CNN エンコーダ＋ViT デコーダ（bf16）。ImageNet 正規化・24 値の mean/std・タイル既定 on（256／64）・17 の倍数の詰め物と `token_drop=3`・デコードは 7 潜在の窓で 17n+5 を出す。潜在 1 フレームだけ解くときは 2 つに複製（#15416） | DiffSynth `minimax_h3_video_vae.py` の移植（einops 使用） | C2 |
| `engine_h3/audio_vae.py` | DAC エンコーダ＋BigVGAN デコーダ（f32）。Comfy 版は weight_norm 合成済みなので普通の畳み込みの重みとして読む。潜在 `[2,32,A]`・32 値の mean/std・32kHz ステレオ | DiffSynth `minimax_h3_audio_vae.py` の移植（weight_g/v を外す） | C2 |
| `engine_h3/media_out.py` | PyAV で mp4。libx264・crf 19・yuv420p。**RGB→YUV の変換を BT.709 で明示**（`reformat(format="yuv420p", dst_colorspace=…, dst_color_range=…)` か自前の行列。タグだけでは swscale の既定 BT.601 で変換される＝レビュー m-4）。音声は s16 の AudioFrame → `AudioResampler`（fltp・stereo・32000）→ pts → フラッシュの順で AAC。音声は映像の尺に切り揃える（最大 12.5ms 長い）。`ltx_core`／`ltx_pipelines` は import しない | 自前（R3 項目 11） | C2 |
| `engine_h3/text_encoder.py` | **第 1 弾は `Qwen3VLTextModel`（文章だけ。視覚塔は組まない＝T2V では使わない。I2V の `Qwen3VLModel` は第 2 弾）**を meta で `num_hidden_layers=50`・`norm=Identity` で組み、Comfy 形式の名前を写す（`model.layers.N`→layers、`model.embed_tokens`。`visual.*` は第 1 弾では読まない）。量子化 350 本は `H3Int8Linear`。層オフロードは `engine/gemma/layer_offload_service.GemmaLayerOffloadService`（torch のみ・`.layers` を探す汎用実装。レビューで Qwen3-VL の層の署名と噛み合うことを確認済み）。設定は DiffSynth と同じ（`rope_scaling: {mrope_interleaved: True, mrope_section: [24,20,20]}`・`rope_theta 5000000`・head_dim 128）。トークナイズは chat template 無し・`add_special_tokens=False`・**追加特殊トークン 151669〜151675 が語彙に入ることをテストで固定**（入らなければ ComfyUI と同じく明示的に足す）。出力は 50 層後の正規化前 `[L,5120]` | DiffSynth `minimax_h3_text_encoder.py`＋変換表。レビュー M-2・M-3・m-12 | C3 |
| `engine_h3/scheduler.py` | **N・shift 映像・shift 音声を引数に取る**。`base=linspace(1,0,N+1)[:-1]`、`σ=s·b/(1+(s−1)·b)`、t=1−σ、Euler `x_next = x + (σ−σ_next)·v`（出力は x0=x+σv の向き）、f32。既定値（20／12／3）は定数として 1 箇所に置く | R2 G・裁定 22 | C3 |
| `engine_h3/pipeline_h3.py` | 3 段: (1) TE を組む→エンコード→解放 (2) DiT を組む（ブロックは CPU・それ以外は GPU）→`BlockSwapService(blocks_on_gpu, hold_arenas=True)` を `install`→先読み→ループ→解放 (3) 映像 VAE と音声 VAE で復号→mp4。系列 `[text(L) | audio(2A) | video(T'×(H/32)(W/32))]`（T2V は cond 区間なし）、時刻は映像と音声で別（`unique` で番号化・AdaLN の添字＝番号×3＋タグ〔0 映像・1 テキスト・2 音声〕）、ノイズは seed から `torch.Generator` で映像 `(1,24,T',h,w)` → 音声 `(2,32,A)` の順。`blocks_on_gpu` 既定 8。全経路 `torch.no_grad()`。段ごとの秒数と RSS を `phases` に記録 | DiffSynth `pipelines/minimax_h3_audio_video.py` の系列組み立て部分の移植 | C3 |
| `engine_h3/worker.py` | `engine25/worker.py` と同じ作法: `python -m engine_h3.worker`、stdin の JSON 行、stdout の `@@LTX@@`＋JSON、stderr のログ `[h3_worker]`、torch はモジュール最上位で import しない。**メッセージのキー名（第 2 弾の `_build_load_payload` がそのまま写せるように今決める）**: `load`＝`transformer_path`・`text_encoder_path`・`video_vae_path`・`audio_vae_path`・`tokenizer_dir`・`blocks_on_gpu`・`te_layers_on_gpu`・`attention_backend`・`block_swap_prefetch`・`deterministic`／`generate`＝`prompt`・`seed`・`width`・`height`・`num_frames`・`frame_rate`・**`num_inference_steps`・`shift_video`・`shift_audio`**（省略時は既定値）・`output_path`。イベント: `ready`〔`sage_available`〕・`progress`〔`stage`＝`encode`／**`stage1_denoise`**／`decode`・`index`・`total`〕・`done`〔`seed_used`・`peak_vram_mb`・`peak_vram_reserved_mb`・`rss_peak_gib`・`seconds`・`num_frames`・`size_bytes`・`attention_used`・`block_swap_prefetch_used`・`h3.phases`〕・`error`。`load` はパスの検査とヘッダの判定だけ。セルフテスト `--selftest-generate --transformer --text-encoder --video-vae --audio-vae --tokenizer-dir --output [--prompt --width --height --num-frames --seed **--steps --shift-video --shift-audio** --rounds --report --attention-backend --blocks-on-gpu --prefetch/--no-prefetch]` | engine25 の構造を写す。レビュー M-6・m-1・m-2・m-14・裁定 22 | C3 |

**書かないもの**: LoRA・Turbo・Ref2VA・Hybrid・視覚塔と条件行（I2V）・末尾／中間キーフレーム・連結・画角拡張・NAG／VSF・`keep_resident`・常駐。`comfy_quant` の旧式 `_quantization_metadata`・VSA の追加テンソル・PDD ヘッドは「警告して捨てる／拒否」。

**揺れの吸収（裁定 17 の修正点 4）**: 第 1 弾で**ただで取れる分だけ**＝接頭辞 4 通り・形状から構成（格子数とランクも）・`comfy_quant` の未知キーの無視・groupsize は 256 以外も `int8_linear` にそのまま渡す（融合されない経路で動く＝R1）・量子化されていない層は dtype のまま読む・知らないテンソルは警告して捨てる。試すのは公式ファイルだけ。コミュニティ版は第 2 弾以降。**Turbo を融合した低ステップの配布物**（裁定 22 の背景）も構造は同じ int8 ConvRot の safetensors なので、この吸収の範囲で読める見込み（第 2 弾のスライダーで少ないステップを指定する。実測は第 2 弾以降）。

### 3.4 テスト（CPU）

- **アプリ venv（torch 無し）で走るもの**: `tests/test_h3_format.py`＝合成ヘッダで H3 判定・接頭辞・枝刈り（G=1025／4097・rank 8／16）・LTX のヘッダは H3 でない・`comfy_quant` の未知キー無視。
- **`.venv-engine-h3` で走るもの**（`importorskip("torch")`＋`importorskip("comfy_kitchen")`。エンジン venv に pytest が無く `conftest.py` が FastAPI を import するので、**既存の engine venv 用 runner の形＝アプリ venv の site-packages を末尾に足し `--noconftest`** で回す）: 枝刈り AdaLN の補間（表の端で範囲外を読まない。I2V の条件行を想定した合成値 t=1.0 も）／フレーム規約の算術（124→37／207、345→102／575）／σ 列（N=20 の映像・音声の値）／時刻の番号 M（step 0 は 1・以後 2）／RoPE の座標（小さな格子で手計算）／TE の設定値（mrope_section・interleaved・theta・head_dim）／追加特殊トークンの番号（`<d>`→151669 … `<|caption_end|>`→151675）／状態辞書の名前の写し／`H3Int8Linear` と「`engine/sft_quant/dequant.py` で bf16 に戻して `F.linear`」の一致（GPU があるときだけ）／`engine_h3` 配下に `inference_mode` が無いこと／ワーカーの未知 op の扱い。

### 3.5 門（G 列。G2〜G5 は GPU を使う。**本計画の了承で第 1 弾の分をまとめて了承**）

| 門 | 内容 | GPU |
|---|---|---|
| G0 | 仮想環境の構築が成功し、`import torch, comfy_kitchen, transformers, av, einops` が通る。`torch.version.cuda` が 13.0。freeze と容量を記録 | 不要 |
| G1 | `ck.list_backends()["cuda"]` が `available=True` かつ `capabilities` に `int8_linear`（**合否はこれだけ**）。`probe_sage()` の結果は記録に回す | 不要（import のみ） |
| G2 | comfy-kitchen の煙試験: 合成の int8 重み `[28672,5376]` と x `[9.5k,5376]`（第 1 弾の系列の大きさ）で `int8_linear` が CUDA 経路で走り、eager の結果と末尾の行まで一致すること。**m=80,000（m×n ≥ 2³¹）は §1-95 へ送る** | 使う |
| G3 | block swap の噛み合わせ: `H3Int8Linear` 入りの小さなブロック列（50 個）に `BlockSwapService(blocks_on_gpu=8)` を `install`。**こだまが `"on"`**（eager は切ってあるので外れれば例外）、かつ「全ブロック常駐」「同期」「先読み」の 3 通り×2 パスで出力がビット単位で一致すること | 使う |
| G4 | **部品の数値照合**（正解データ源）: `outputs/h3-gate1/oracle/`（git 管理外）に DiffSynth の `minimax_h3_dit.py`＋`_comfy.py` を写し、`..core.attention.attention_forward`（SDPA）と `..core.gradient.gradient_checkpoint_forward`（素通し）のダミー 2 モジュールを置く。**正解側の Linear は `engine/sft_quant/dequant.py` で bf16 に戻した `nn.Linear`**（int8_convrot・groupsize 256 対応済み）。同じ重みファイルから**ブロック 1 個ずつ**を両実装に読み込み、同じ入力で forward して比べる（許容は相対誤差 1e-2 級＝活性化の量子化誤差を見込む。配線の誤りは O(1) で出る）。token_refiner・final_layer・AdaLN の表・RoPE の座標も同様。VAE はランダム潜在の復号を両実装で比べる（映像は `.weight` を付ける変換、音声は g/v への分解が正解側に要る）。TE は先頭 2 層と最後の 1 層を同じ方法で比べ、**層オフロード on と off で 50 層後の出力がビット単位で一致**すること | 使う（小さい） |
| G5 | **セルフテスト 1 本**: 640×384・124f・20 ステップ・seed 固定・`<d>[English] …</d>` を含む文章で T2V。mp4 に映像と音声が入り、`done` に VRAM ピーク・RSS ピーク（TE 読み込み後・解放後・DiT 読み込み後）・秒数・`phases`・`block_swap_prefetch_used="on"` が出ること。`--rounds 2` の同一シードの一致を記録（合否にはしない）。続けて `--attention-backend sage` を 1 本回して記録 | 使う（数分〜十数分の見込み） |

- 結果は `outputs/h3-gate1/`（git 管理外）に置き、要約を `Docs/VERIFICATION_LOG.md` §165 に書く。G5 が通ったら監督が mp4 を確認して報告する。**第 1 弾のクローズはオーナーの判断**。
- GPU を使う門は同時に 1 体だけ。

### 3.6 コミット（dev。監督が行う）

1. 文書（台帳の起票〔作業ツリーに置いてあるもの〕＋設計正本＋MULTI_ENGINE_DESIGN の訂正＋§10 の参照化）
2. 仮想環境の仕様と freeze
3. `h3_format.py`・`engine_h3/` 本体とテスト
4. 門の記録（VERIFICATION_LOG §165）
main への merge はオーナーの合図で `--no-ff`。

---

## 4. 担当と進め方（監督は監督専任）

| 役 | モデル | 範囲 |
|---|---|---|
| 監督 | — | 設計正本の指示と確認・dry-run・venv 構築と取得（バックグラウンド）・門の実行の統制・コミット |
| 文書係 | Sonnet | `MINIMAX_H3_DESIGN.md`・MULTI_ENGINE_DESIGN の訂正 5 か所・§10 の参照化・pyproject の仕様文 |
| C1 | Opus | `quant_linear.py`・`dit.py`・`rope.py`・G2〜G4 の試験スクリプトと oracle の shim |
| C2 | Opus | `video_vae.py`・`audio_vae.py`・`media_out.py` |
| C3 | Opus | `h3_format.py`・`sft_load.py`・`text_encoder.py`・`scheduler.py`・`pipeline_h3.py`・`worker.py`・テスト |
| 敵対的レビュー | Opus | 実装後のコード |

- 各担当には「設計に疑問を抱いたら止めてエスカレーション」「設計の勝手な変更禁止」「担当外ファイルの変更禁止」「`engine/`・`engine25/`・`services/`・`api/` は変更禁止」を毎回明記する。
- 共有部品（`engine/transformer/*`・`engine/sft_quant/*`・`engine/gemma/*`）は**変更しない**。噛み合わなければ `engine_h3/` 側で包む。どうしても変えるならオーナーに戻す。
- 順序: 文書係と dry-run → venv 構築・取得（背景） → C1・C2・C3 を並行 → C3 が結合 → G0〜G5 を順に → 記録 → コミット。

---

## 5. 間違えやすい点（担当への申し送り。R2 (3) とレビューから）

qkv は `[q;k;v]` 積み重ね／fc1 は `[gate;value]` で SiLU は前半／枝刈り版は SiLU を掛けない／AdaLN と表に渡す時刻は t=1−σ／表の添字は `min(floor,G−2)` で clamp／AdaLN の添字＝時刻の番号×3＋タグ／出力は x0=x+σv の向き（一般の flow-match と符号が逆）／N ステップ＝N 回評価／映像と音声の σ 列は別／RoPE の座標は float64・角度は float32・先頭 96 次元だけ回す／TE は 50 層目の後の正規化前・chat template 無し・追加特殊トークンを番号で確かめる／token_refiner は 1 回だけ／VAE は ImageNet 正規化・mean/std は `__metadata__` の f32 から／タイルは既定 on・1 潜在フレームは 2 つに複製／音声は最大 12.5ms 長いので切る／Comfy の音声 VAE は weight_norm 合成済み／精度の島は f32／`int8_linear` は重みとスケールが GPU 上にないと CUDA 経路に乗らない（eager を切ってあるので例外で分かる）／`weight_scale` はファイルどおり f32 `[N,1]`／全経路 `no_grad`／モデルは meta で組んで `assign=True`。

---

## 6. 危険と、分からない点

1. **性能**: 第三者の実測は 4060 Ti 16GB・864×480×124f・20 ステップで約 6 分（cu130）。本製品は block swap で毎ステップ約 19GB を転送するので PCIe の帯域が上限になる可能性がある（推測）。G5 で測る。遅すぎれば裁定 21。
2. **メインメモリ**: meta で組めば、山は TE 約 27GB → 解放 → DiT 約 21GB の逐次で約 27GB＋α。開発機 64GB は足りる。サブマシン 32GB はページファイル前提（第 3 弾の README）。
3. **block swap と `int8_linear`**: ソースの水準で噛み合う（R1・R3・レビュー (a)）。G3 で最初に確かめる。噛み合わなければ、量子化テンソル型を先読みの特別扱いに足す案（共有部品を触る）か、ブロック内で手で `.to()` する案に戻る。
4. **transformers 5.14.1 で `Qwen3VLTextModel(config)` を 50 層で組むこと**: 設定の読み替え（`rope_parameters`）はテストで固定値と照合し、G4 の層単位の照合で数値を確かめる。
5. **数値の正解データ源**は部品ごとの照合（G4）まで。動画 1 本全体をビット一致で比べる相手はない（diffusers は枝刈り版を拒否・seed の消費順が実装ごとに違う）。
6. **SageAttention post6 が torch 2.13 で動くか**は G5 の sage 1 本で確かめる（既定は sdpa）。
7. **ステップ 20・Euler は監督の選択**（§1）。G5 で絵が甘ければ 50 で 1 本比べる。
8. **サブマシンのドライバは未確認**（第 3 弾の README までに確かめる）。
9. **G4 の正解側は W8A16（活性化は bf16）**、本実装は W8A8。差は活性化の量子化誤差で、配線の誤りとは桁が違うので見分けられる見込み（推測）。

---

## 7. 敵対的レビューの結果と採否、オーナーの差し戻し

**オーナーの差し戻し（2026-10-17・1 件）**: ステップ数と shift の 3 本のスライダーを第 2 弾で操作パネルと Gradio に出す（§2 の第 2 弾・§1 の表・§3.1 の 3・§3.3 の `scheduler.py`／`worker.py` に反映。第 1 弾への影響はワーカーの `generate` が 3 項目を引数で受け取ることだけ）。

**敵対的レビュー**

Opus による事実確認レビュー（`C:\Users\phenotype nezu\.claude\plans\cached-doodling-hare-agent-a7460a6c4904ab860.md`）。Critical 1・Major 6・Minor 14。監督が現物で裏取り（こだまの値・親の段階名・移植元の import・comfy-kitchen の落ち先）したうえで、**全件採用**。ただし 2 点は部分採用:

- M-4「正解側の shim は約 1,100 行」: 正解側を量子化テンソル経路で作る場合の見積もりで、ダミーの `core` 2 モジュール＋`dequant.py` で bf16 に戻す案なら小さい。**提案された作り方（dequant 版の `nn.Linear` を正解に・許容 1e-2 級）は採用**し、見積もりだけ訂正。
- m-9 の `av` の版: 提案は「18.1.0 か 19.0.1 のどちらか」。LTX 2.5 の venv で実績のある **18.1.0** を採る。

採用した主なもの: C-1 meta での組み立て／M-1 eager も切る・こだま `"on"`・3 通りのビット一致／M-2 TE の設定値の照合と層単位の照合／M-3 特殊トークンの番号／M-5 判定を torch 非依存の `h3_format.py` に分離／M-6 段階名 `stage1_denoise`／m-1〜m-14（`done` のキー・セルフテストの引数とメッセージのキー名・`no_grad`・BT.709 の明示変換・VAE の精度・`[N,1]`・RoPE の float32・G−1／G−2・einops と av と dry-run の再実行・G1 の sage を任意に・ステップ 20 の注記・G2 の縮小と視覚塔の不読・文書の規約・load と generate の分担）。
