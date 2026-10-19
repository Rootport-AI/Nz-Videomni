# MiniMax H3: Turbo LoRA・FL2VA の連結・A2V の一次資料調査（R4）

- 調査日: 2026-10-19
- 方法: ウェブと一次資料とコードを読んだだけ。GPU・git・パッケージ導入は使っていない。LoRA のヘッダは HTTP Range（ファイルの先頭部分だけを取る要求）で読んだ。
- 読んだコードの版（ここに書く行番号はこの版のもの）
  - ComfyUI 本体: `Comfy-Org/ComfyUI` の `8f04fd6`（2026-10-18）
  - DiffSynth-Studio: `modelscope/DiffSynth-Studio` の `4e34969`（2026-10-16）
  - diffusers: `b800f82`（調査日時点の main）
- ライセンス: ComfyUI は GPL-3.0 なので仕様として読んだだけで、写していない。DiffSynth と diffusers も読んだだけ。

用語:
- **条件行（cond 行）**: 系列の中に、生成する映像・音声とは別に置く行。1 回だけ符号化して毎ステップ入れ直し、ノイズ除去はしない。
- **その場固定**: 生成する映像・音声の潜在の一部を元素材の潜在で置き換え、その行の時刻を無ノイズ側に止める方式。LTX の Chained や Inpainting と同じ系統の考え方。
- **denoise mask**: 行ごとに「残す（0）／作り直す（1）」を指定する印。

---

## 結論

### 問い 1. Turbo LoRA
- **事実**: Turbo は MiniMax 公式の配布物ではない。LightX2V（ModelTC）が配っている Apache-2.0 の LoRA で、Comfy-Org も同じファイルを転載している。変わる点は 3 つ。
  - ステップ数: 4 か 8。
  - shift（映像／音声）: 544p で学習した版は 12/3、768p 版は 6/3。
  - 強さ: alpha は版によって 8 と 128 があり、実際の倍率は 0.0625 と 1.0 になる。

  CFG（ガイダンスを効かせるための 2 回目の前向き計算）は、元のモデルが公式 README で「CFG 蒸留済み」とされているので、もともと無い。
- **事実（ヘッダの実測）**: ランクは 128。当たる層は、50 ブロックと token_refiner の 2 ブロックの qkv・out_proj・fc1・fc2 で、qkv は 3 本を束ねてランク 384 として持つ。**AdaLN には当たらない**ので、枝刈り版（AdaLN を表に畳んだ版）にもそのまま当たる。ファイルは 1,956,119,360 バイトで、1 ブロックあたり約 37.6MB。
- **事実**: 融合済みの配布物は多い（MATLOWAI の fused-turbo int8 ConvRot など。研究ノート §13.3 のとおり）。ComfyUI は、量子化された重みでも「読み込み時に融合」する。手順は「逆量子化→差分を足す→スケールを計算し直して再量子化」。
- **推測（VRAM）**: 融合なら、常に増える VRAM は 0。実行時に掛けるなら、LoRA を全部 GPU に置いて約 1.87GiB（約 1.96GB）。ブロックごとに出し入れするなら、1 ブロック約 38MB で足りる。中間の活性化は数十 MB 程度。計算量は、本体の行列積の約 5% 増える見込み。

### 問い 2. FL2VA の連結
- **事実（公式）**: 公式の FL2VA が受け付けるのは、画像 0〜2 枚の先頭フレームと末尾フレームだけ。SGLang・diffusers・DiffSynth は、どれも `frame_index` を 0 と -1 に限っている。
- **事実（ComfyUI 本体）**: `MiniMaxH3AddGuide` は、任意のフレームに「静止画・17k+5 フレームの短い動画・音声」を置ける。これは公式ではなく、コミュニティ（drozbay）の PR #15439 で、2026-08-13 に本体へ取り込まれた。仕組みは**条件行**で、ガイドを単独に符号化して、時間位置の座標だけを目的のフレームに合わせる。潜在の上書きではない。
- **事実（ComfyUI 本体）**: これとは別に、**その場固定**の仕組みとして denoise mask がある（PR #15375、2026-08-18 取り込み）。DiffSynth の Retake も同じ考え方。LTX の Chained に一番近いのはこちら。
- **事実**: Comfy 公式文書が勧める連結の仕方は「前のクリップの末尾 1 枚から I2V で続ける」。誤差は積み重なると明記されている。PR #15439 の作者は「既存の動画の先頭 22 フレームと音声を frame 0 のガイドに置くと、映像と音声の両方の続きが作られる」と書いている。
- **未確認**: 継ぎ目と音声のつながりの品質を数値で比べた報告は見つからなかった。Ref2VA は「参照」として受けるので、ピクセルどおりの連結には向かない（推測）。

### 問い 3. A2V（音声→映像）
- **事実（公式）**: 公式の機能一覧に A2V は無い。FL2VA には音声の入力が無い。Ref2VA は音声の参照を最大 3 本まで受けるが、Comfy の文書は「音声は声質を運ぶ。言葉は運ばない」と書いている。
- **事実（DiffSynth の Retake、2026-08-06。FL2VA の重みで動く）**: `retake_audio` に区間を指定しないで渡すと、文書の言う「audio-driven video generation」になる。`retake_video` を同じように渡すと V2A（映像→音声）になる。仕組みは、符号化した音声の潜在を、生成する音声の行にその場で入れ、時刻を 1.0（無ノイズ）に止めるもの。オーナーの見立てどおりだった。ComfyUI の audio 用 denoise mask でも同じことができる。
- **事実**: AddGuide の audio は、上の方式とは別物で、音声を別の条件行（時刻 1.0）として置く。
- **未確認**: 「Ref2VA のほうが A2V に向く」という主張には、一次資料の根拠が見つからなかった。リップシンク（口の動きと音声の同期）の報告は、すべてコミュニティの個別報告（下の 3.4）。

---

## 詳細

### 1. Turbo LoRA

**1.1 正体と配布元（事実）**
- MiniMax 公式の GitHub `MiniMax-AI/MiniMax-H3` の README とファイル一式には、Turbo の記述もファイルも無い。公式 README は「The released checkpoints are CFG-distilled Omni Transformer model weights.」と書いている。研究ノート §1 の「ガイダンス蒸留は要確認」は、公式としては蒸留済みが正しい。
- 配布元は `lightx2v/Minimax-h3-Turbo`（HF、Apache-2.0、2026-08-07 作成、最終更新 2026-10-16）で、仕様の表は `ModelTC/Minimax-H3-Turbo` の README にある。Comfy-Org の `loras/` フォルダの 3 本は、これの転載である（ヘッダのメタデータが同じ）。
- 仕様の表（ModelTC の README）:

| 版 | 対象 | 学習解像度 | 学習時の shift（映像／音声） | 蒸留のステップ数 | 推奨ステップ数 |
|---|---|---|---|---|---|
| FL2VA 4-step v0.1 | FL2VA／T2VA | 544p 混合 | 12 / 3 | 4 | 4 |
| FL2VA 8-step v1.0 | FL2VA／T2VA | 544p 混合 | 12 / 3 | 8 | 8 / 4 |
| FL2VA 4-step v1.0 768p | FL2VA／T2VA | 1344×768 | 6 / 3 | 4 | 4 |
| FL2VA 8-step v1.0 768p | FL2VA／T2VA | 1344×768 | 6 / 3 | 8 | 8 |
| Ref2VA 4-step v0.1 | Ref2VA | 544p 混合 | 12 / 3 | 4 | 4 |

- HF には、この表に載っていない v1.1・v1.2（4-step 768p。v1.1 は fp8 版もある）と、Ref2VA 8-step v1.0 768p も置かれている（ファイル一覧で確認。仕様は未確認）。
- サンプラーは euler が既定（ModelTC の `COMFYUI_SETUP_AND_INFERENCE.md`）。shift の決め方は「N 回の評価点を、shift をかける前の格子 `q_i=(N−i)/N` に取り、映像と音声にそれぞれの shift をかける」で、本製品の §3 と同じ形。
- Ref2VA の Turbo は、参照画像の大きさの決め方 `match`（生成する画角の面積に合わせる）で学習してある。推論でも `match` を使うよう推奨されている。

**1.2 アダプタの形（事実。HTTP Range でヘッダを実測）**
- 3 本（fl2v 8step v1.0、fl2v 4step 768p、ref2v 4step v0.1）とも同じ形で、テンソルは 624 個。
  - `diffusion_model.blocks.N.{attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2}`（50 ブロック）
  - `token_refiner.blocks.N.`の同じ 4 つ（2 ブロック）
  - それぞれに `lora_A`／`lora_B`／`alpha` がある。
- ランクは 128。qkv は `[384, 5376]`／`[21504, 384]` の形で、3 本を束ねている（メタデータに `qkv_fusion: block diagonal B; concat A; alpha multiplied by 3`）。
- メタデータに fc1 の並べ替えも書いてある（`swi_glu_mapping: Diffusers [value;gate] -> ComfyUI [gate;value]`）。
- **AdaLN・patch_proj・final_layer には当たらない。** Comfy の文書も「Turbo LoRAs have no adaln tensors and work on either build」と書いている（`tutorials/video/minimax/minimax-h3.mdx` の 43 行目）。
- 倍率: 8step v1.0 と ref2v 4step は `training_alpha 8.0`・`training_scale 0.0625`、4step 768p は `training_alpha 128.0`・`training_scale 1.0`。DiffSynth の例は、768p 版で `--lora-alpha 128` を指定している。
- 気づいた点: ref2v のメタデータの `base_model` が `minimax_h3_fl2va_bf16` になっている。雛形の書き写しとみられる（推測）。
- 大きさ: 1,956,119,360 バイト（BF16 が 416 個、F32 のスカラー alpha が 208 個）。1 ブロックあたり 18,808,832 パラメータで、bf16 なら約 37.6MB。

**1.3 融合済み配布物（事実）**
- HF には、MATLOWAI/minimax-h3-fused-turbo-int8-convrot（約 54 万 DL）、PulpCut/MiniMax-H3-Ref2VA-Turbo-INT8-ConvRot、Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI、sandpies/Wuli-MiniMax-H3-Turbo-ComfyUI などがある（HF の検索 API）。CivitAI 側の一覧は研究ノート §13.3。
- 別系統の少ステップ化もある。
  - DMAD 4-step LoRA: shift 12/2、サンプラー lcm＋simple（Comfy の文書 160 行目）。
  - OpenVDN の VDN-H3 8 steps（DiffSynth の文書の表）。

**1.4 VRAM と速度**
- 事実: ComfyUI は、量子化された重みに LoRA を掛けるとき「逆量子化→差分を足す→`requantize_from_float(scale="recalculate", stochastic_rounding=...)`」で融合し直す（`comfy/ops.py:320-345`、`comfy/model_patcher.py:920-950`）。Comfy の文書は「Turbo LoRAs shorten the schedule, not per-step time (merged at load)」と書いている（`minimax-h3.mdx` の 143 行目）。
- 推測: int8 ConvRot に融合すると、再量子化の誤差が乗る（ConvRot の回転を保ったまま再量子化できるかは、未確認）。
- 推測（実行時に掛ける場合）:
  - LoRA を全部 GPU に置くと +約 1.87GiB。
  - block swap と一緒にブロック単位で出し入れするなら、+約 38MB×（先読みの段数）。
  - 中間の活性化は、トークン数×384×2 バイト程度（3 万トークンで約 23MB）。
  - 計算量: 1 ブロックの本体の行列は約 3.56 億パラメータ、LoRA は約 1,880 万なので、約 5.3% 増える（bf16 で計算する分）。
- 品質の注記（Comfy の文書 139〜143 行目）:
  - 4-step では参照がほとんど効かない。
  - 細部（肌・布・髪）が滑らかになる。
  - 強さを 1 未満にして、ステップを増やすと元の品質に近づく。
  - 連結すると焼け（色や質感の劣化）が早く出る。
  - 8-step で映像の shift を 3 にするなど、shift を誤ると格子状の乱れが出る。

### 2. FL2VA の連結

**2.1 公式が受け付けるもの（事実）**
- 公式 README の表は「Supports zero, one, or two input images … First-frame / last-frame / first-and-last-frame」。Ref2VA は「画像 ≤9・動画 ≤3（各 2〜15 秒・合計 ≤15 秒）・音声 ≤3（同）・合計 ≤12」。
- 公式の再現スクリプト `scripts/readme/reproducible-768p-fl2va-request.sh` は `conditions: [{type:image, role:keyframe, frame_index:0}]` の形。SGLang の cookbook も 0 と -1 だけ。
- DiffSynth は `assert all(idx in (0, -1) ...)`（`diffsynth/pipelines/minimax_h3_audio_video.py:444`）。diffusers も keyframes は先頭と末尾だけを扱う。
- **中間フレームや動画の一部を条件にする機能は、公式には無い。**

**2.2 ComfyUI の `MiniMaxH3AddGuide`（事実。`comfy_extras/nodes_minimax_h3.py:166-240`）**
- 入力: `frame_idx`（負の値は末尾から数える）、`image`（1 枚、または 17k+5 フレームの短い動画。5 フレーム未満は先頭の 1 枚だけを使う。210 行目）、`audio`（同じ位置から、残りの尺に切り詰める。228〜234 行目）。ノードをつなげて複数置ける。
- 置き方は**条件行**。根拠は次のとおり。
  - `PackedLayout` が、ガイドごとに `cond`（映像）と `cond_audio`（音声）の区間を足す（`comfy/ldm/minimax/model.py:375-396`）。
  - 時間の座標は `cond_t = cursor + 5/3 × resolved_frame_index`（379 行目）。
  - 値は `0.999·clean + 0.001·noise`（映像）と、無ノイズのまま（音声。`AUDIO_COND_TIMESTEP = 1.0`。30〜33 行目、535〜562 行目）。
  - 時刻は `max(t_v, 0.999)`／`max(t_a, 1.0)`（681〜684 行目）。
  - 毎ステップ入れ直し、更新しない（`img_update`／`audio_update` が False）。
- 先頭と末尾の扱いも同じ経路を通る（`MiniMaxH3ImageToVideo` が `resolved_frame_index` を 0 と `frame_count−1` にする。149 行目と 154 行目）。つまり、本製品の I2V（条件行・0.999・毎ステップ入れ直し）の自然な一般化になっている。
- PR #15439（drozbay、2026-08-09 起票、08-13 取り込み）の本文の要点:
  - 「The model itself is capable of addressing guides by position on a continuous time axis」
  - 「feed the first 22 frames of an existing video plus its audio into one AddGuide at frame_idx 0 … the model generates the continuation of both streams」
  - 「put a still at frame 60 of 124」
  - 先頭と末尾の座標は従来と同じ。
- 未確認: 中間位置や短い動画をガイドに使う形を学習したかどうか。MiniMax は何も述べていない。

**2.3 その場固定（denoise mask と Retake。事実）**
- ComfyUI の PR #15375（drozbay、2026-08-18 取り込み）は、映像と音声の潜在に行単位の 0／1 の印を付けられるようにした。映像は 2×2 パッチ単位、音声は潜在フレーム単位。Comfy の文書は「inpainting, removal, or extension while holding the rest stable」と書いている。
- 残す行の扱い（`comfy/model_base.py` の MiniMaxH3、約 2295〜2330 行目と `model.py:686-707`）:
  - 時刻を `0.999`（映像）／`1.0`（音声）に止める。
  - 値は、元の潜在を `scale_latent_inpaint` で毎ステップ入れ直す。
- DiffSynth の Retake（`MiniMaxH3Unit_VideoRetakeEmbedder`／`AudioRetakeEmbedder`、366〜429 行目。`model_fn` の 986〜1017 行目）:
  - `latents = latents·mask + input·(1−mask)` で置き換え、残す行の時刻を 1.0 にする。
  - 映像は 17 フレームの VAE クリップ単位でしか区切れない（文書 139 行目「retaking any frame of a clip retakes the whole clip」）。
- LTX の Chained（潜在の末尾を凍結して使い回す）に**相当するのはこちら**。前のクリップの末尾 N フレームを、新しいクリップの冒頭に置いて mask を 0 にすれば、同じことになる。
- 推測: H3 の VAE は「先頭 5 フレーム→潜在 2 つ、その後 17 フレームごとに潜在 5 つ」という形をしている（`FRAME_PER_TOKEN = (1,4,4,4,4)`）。そのため、前のクリップの潜在の末尾を切り出してそのまま使うより、末尾のピクセルを符号化し直して冒頭に置くほうが、形が合う。

**2.4 連結の実例と品質（事実と未確認）**
- Comfy の公式文書（`minimax-h3.mdx`、最終更新 2026-10-18）の連結の節（129〜133 行目）の要点:
  - 「chain I2V continuations from each clip's last frame. Errors compound (faces lose texture first); turbo/distilled schedules show burn sooner.」
  - 雛形 `video_minimax_h3_i2v_continuation` は、プロンプトを空にして、前の末尾 1 枚を読む。
  - 2 本目以降に H3-Anti-Burn-LoRA を 0.5 で掛ける案が書かれている。
  - **公式の雛形は 1 枚で継いでいる。複数フレームでは継いでいない。**
- Comfy の公式雛形「Multiframe Reference」（`minimax-h3-multiframe.mdx`）は **Ref2VA の重み**＋AddGuide を連ねて、「video continuations and multi-shot scenes」を作る。
- コミュニティ:
  - ComfyUI-MiniMax-H3-LongMedia（vizart-vj、Apache-2.0、約 300 スター、最終 push 2026-09-29）の Segmented モード: 前の単位の末尾の重なり区間の映像潜在と音声潜在（`video_latent_tail`／`audio_latent_tail`）を、次の単位の keyframe として渡し、重なりは出力から取り除く（`continuation_context.py`、`docs/ARCHITECTURE_EN.md` の 49〜83 行目）。
  - PR #15439 の matlowai のコメント（2026-08-11）: 「the pinned keyframes are the seam itself」。参照動画を付けるとガイドの位置がずれる不具合も指摘された。
- 不具合: 継続ガイドの音声と、単独の音声参照を一緒に使うと、形が合わずに落ちる（issue #16289、2026-09-12、未解決。修正 PR #16750 も未取り込み）。
- **未確認**: 継ぎ目の映像の跳びや、音声のつながりを数値か目で比べた報告は見つからなかった。

**2.5 Ref2VA は連結に向くか（事実と推測）**
- 事実: Ref2VA の参照動画は、目的の映像より前の時間に置かれる（`cursor` が参照の長さだけ進む。`model.py:370-374, 398-438`）。参照として受けるだけで、目的のフレームに固定されるわけではない。公式の例では、参照動画の音声の扱いを、プロンプトの指示（`partially_copy`／`reference`）で決めている。
- 事実: Comfy の文書は「Reference shots need more steps … 4-step turbo can barely apply references」と書いている。
- 推測: 動きや人物を引き継ぐ目的なら Ref2VA は使える。ただし、画素の位置まで合わせる継ぎ目は、FL2VA（または Ref2VA）に AddGuide か denoise mask を足す方式のほうが確実。

### 3. A2V（音声→映像）と V2A

**3.1 公式（事実）**
- 公式のタスクは t2va・fl2va・ref2va の 3 つだけ。README には「audio is encoded solely by the H3-AudioVAE」とある。
- 公式の Ref2VA の例: 動画 1 本と音声 1 本を参照し、`<Audio 1>` を BGM として `partially_copy`、`<Audio 2>` を声質の参照として、唇を台詞に合わせる。音声を「その場で固定」する機能は、公式には無い。
- Comfy の文書（Reference to Video の節、217〜218 行目）: 「`use <Audio 1> exactly as it is` vs `follow the timbre of <Audio 2>`」「Audio carries timbre, not words.」

**3.2 音声を固定する A2V（事実）**
- DiffSynth の `MiniMax-H3-Retake.py`（2026-08-06 に #1567 で追加）。FL2VA の bf16 で、次の例を載せている。
  - 「Audio -> Video + Audio」
  - 「Audio + First frame -> Video + Audio」
  - 「Video -> Video + Audio」（V2A）
  - 区間を指定した部分の作り直し
- DiffSynth の文書 142 行目: 「Omitting it preserves the whole source audio, which turns `retake_audio` into audio-driven video generation.」
- 同 139 行目（映像側）: 「… which turns `retake_video` into video-driven audio generation.」
- 仕組みは「その場固定」。生成する音声の行に、符号化した潜在をそのまま入れ、時刻を 1.0 にする（DiffSynth の 990〜1017 行目）。オーナーの見立て「音声行に符号化済み潜在を入れて時刻を無ノイズに固定」と一致する。ComfyUI では audio の denoise mask で同じことができる（`audio_denoise_mask`、時刻は 1.0 に止める）。
- 別方式: AddGuide の audio は、音声を別の条件行に置く方式。目的の音声の行は、それとは別にノイズから作り直される。推測では、出力の音声は元の音声とビット単位で一致しない。LongMedia の lip_sync モードは、この方式で音声に合わせ、出力では元の音声に差し戻している（`docs/AUDIO_MODES_GUIDE_EN.md` の 136〜148 行目）。

**3.3 「Ref2VA のほうが A2V に向く」の根拠（未確認）**
- 一次資料には根拠が無い。検索要約の「Ref2VA is the only H3 route that takes audio for lip sync」は誤り。DiffSynth の Retake が、FL2VA の重みで音声を固定する A2V を公式の例として載せている。
- 反対の報告: HF の公式リポジトリのディスカッション #91（lorentaken、08-19）は、「Ref2VA＋音声だと人物が別人になる（顔の類似度の平均 0.07〜0.09）。FL2VA＋I2V なら 0.89」と報告した。返信は 1 件（rocky533、09-12）で、「Ref2VA は音声の有無にかかわらず人物を保つ」と反論している。MiniMax の回答は無い。

**3.4 リップシンクの品質（コミュニティ）**
- note「Long-form Music Lip-sync Method for Minimax-H3」（めぐチャンネル、2026-08-26）:
  - Ref2VA と diffusers で、「生成音声 latent をクリーン状態で固定」する Audio Lock を改良した「vocal_lock」を使う。
  - 「ほとんど完璧なリップシンク」で、ずれは 0 に改善した。口の開きと振幅の相関は 0.350。
  - 曲は場面ごとに切り出して渡し、生成後に尺を切りそろえてから連結する。継ぎ目の評価は無い。
- comfy.icu の「MiniMax H3 • LipSync Latent Setup」（LongMedia の一部）:
  - 音声の denoise を 0、映像の denoise を 0.35 が既定。
  - 「低すぎると唇がほとんど動かない」。音声とぶつかるときは 0.15〜0.25 にする。
- ComfyUI の issue #16387（2026-09-17、未解決）: 「v0.35 以降でリップシンクが使えない」（詳細は無い）。

---

## 本製品への含意（推測。設計判断はオーナーと監督に委ねる）
- 中間キーフレーム（§1-97）は、本製品の I2V の条件行（0.999・毎ステップ入れ直し）を、`cond_t = L + 5/3 × フレーム番号` に一般化するだけで載る見込み。ComfyUI の実装の形がその根拠。
- 連結は 2 案ある。
  - A 案: 末尾を短い動画のガイドにする（条件行。17k+5 フレーム）。
  - B 案: 冒頭を固定する（denoise mask。LTX の Chained に近い）。
- A2V と V2A は B 案と同じ部品（行ごとの時刻を固定する仕組み）で作れる。

## 出典

一次資料（公式）:
- MiniMax 公式 GitHub の README と再現スクリプト: https://github.com/MiniMax-AI/MiniMax-H3 （`README.md`、`scripts/readme/reproducible-768p-fl2va-request.sh`、`…-ref2va-request.sh`）
- SGLang cookbook: https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3

Turbo LoRA:
- 配布元: https://huggingface.co/lightx2v/Minimax-h3-Turbo
- 仕様の表: https://github.com/ModelTC/Minimax-H3-Turbo （`README.md`、`COMFYUI_SETUP_AND_INFERENCE.md`、`DIFFUSERS_SETUP_AND_INFERENCE.md`）
- 転載: https://huggingface.co/Comfy-Org/MiniMax-H3 （`loras/` の 3 本。ヘッダは 2026-10-19 に実測）

ComfyUI 本体のコード（`8f04fd6`）:
- `comfy_extras/nodes_minimax_h3.py`
- `comfy/ldm/minimax/model.py`
- `comfy/model_base.py`（class MiniMaxH3）
- `comfy/ops.py`
- `comfy/model_patcher.py`

ComfyUI の PR と issue:
- PR #15439: https://github.com/Comfy-Org/ComfyUI/pull/15439
- PR #15375: https://github.com/Comfy-Org/ComfyUI/pull/15375
- issue #16289／PR #16750／issue #16387

Comfy の公式文書: https://github.com/Comfy-Org/docs
- `tutorials/video/minimax/minimax-h3.mdx`
- `minimax-h3-native.mdx`
- `minimax-h3-multiframe.mdx`

DiffSynth-Studio（`4e34969`）: https://github.com/modelscope/DiffSynth-Studio
- `diffsynth/pipelines/minimax_h3_audio_video.py`
- `examples/minimax_h3/model_inference/MiniMax-H3-Retake.py`
- `MiniMax-H3-FL2VA-Turbo.py`
- `docs/en/Model_Details/MiniMax-H3.md`

diffusers: `src/diffusers/modular_pipelines/minimax_h3/`（`encoders.py`、`before_denoise.py`）

コミュニティ（日付は本文に記載）:
- LongMedia: https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia
- comfy.icu のノード解説: https://comfy.icu/node/MiniMaxH3LatentLabLipSyncSetup
- HF のディスカッション #91: https://huggingface.co/MiniMaxAI/MiniMax-H3/discussions/91
- note（めぐチャンネル）: https://note.com/ai_meg/n/ndf4ddc64e709
