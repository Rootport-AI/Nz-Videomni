# R1: H3 の音声エンコーダーと Retake（音声固定の A2V）の一次資料調査

調査日: 2026-10-25。読み取り専用で調べた。GPU は使っていない。

参照した版:
- DiffSynth-Studio（Apache-2.0）: コミット `4e34969`（2026-10-16）。該当ファイルは main（`60b709c`）と同じ内容であることを確かめた。
- diffusers（Apache-2.0）: main `aaf57d0`（2026-10-24）。
- ComfyUI（GPL-3.0）: master `19ae172`（2026-10-25）。**仕様を読むためだけに使った。コードは写していない。**
- ComfyUI-MiniMax-H3-LongMedia（Apache-2.0）: main `409e4cb`。

以下、DiffSynth のファイルは `DS:` と略す。
- `DS:vae` = `diffsynth/models/minimax_h3_audio_vae.py`
- `DS:pipe` = `diffsynth/pipelines/minimax_h3_audio_video.py`
- `DS:base` = `diffsynth/diffusion/base_pipeline.py`
- `DS:audio` = `diffsynth/utils/data/audio.py`
- `DS:ex` = `examples/minimax_h3/model_inference/MiniMax-H3-Retake.py`

diffusers のファイルは `DF:` と略す。
- `DF:vae` = `src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py`
- `DF:enc` = `src/diffusers/modular_pipelines/minimax_h3/encoders.py`
- `DF:benc` = `.../minimax_h3/before_encoder.py`

---

## 結論

1. **エンコーダーの正体**: 入力は生の波形（32 kHz・モノラル 1 チャンネル）。メルスペクトログラムではない。構成は 3 段。
   - DAC 系の畳み込みエンコーダー（`encoder.*`）。ストライドは 2・4・4・5・5 で、積は 800。
   - 因果注意（後ろを見ない注意）の射影ブロック（`pre_block.*`）。幅を 2048 から 32 に絞る。
   - `mean_proj` の 1×1 畳み込み。

   潜在には**平均（mean）をそのまま使う**。標本は取らない。`logs_proj` は推論では使われない。そのあと `(z − mean) / std` で正規化する。ステレオは、デコーダーと同じく「バッチ 2 のモノラル」として扱う。手元のファイルの 110＋22＋2 本のテンソルは、名前も形もこの構成と完全に一致した（ヘッダーで確認）。移植量は約 130〜170 行と見込む（推測）。
2. **波形の前処理**: DiffSynth は torchaudio で 32 kHz に変換する。モノラルは 2 チャンネルに複製し、3 チャンネル以上は先頭 2 つだけを使う。ピーク正規化はしない。全体を符号化してから、潜在を `A = round(F/24·40)` 個に切り詰める。足りない分は最後の潜在を繰り返して埋め、その部分は「作り直す」側に回す。**H3 用の仮想環境（venv）には torchaudio が無い**。入っているのは PyAV 18.1.0 だけ。
3. **Retake の仕組み**: 印（mask）は 1 が作り直し、0 が残す。残す行について、DiffSynth は次の 3 つを毎ステップ行う。
   - 入力を元の潜在に差し替える。
   - 時刻を 1.0（ノイズ無し）にする。
   - 速度を「元の潜在へ戻る向き」に差し替える。

   最終的な潜在は元の潜在そのものになる。映像の行は純ノイズから普通に作る。出力の音声は、元の波形ではなく、**VAE で復元した音声**。RoPE（位置の回転埋め込み）の座標は変わらない。値 1.0 は AdaLN（時刻で特徴を調整する層）用の、重複を除いた時刻の集合に加わる。
4. **部分的な固定**: 一次資料に前例がある。ComfyUI の H3 実装は 0〜1 の連続値の印を受け付ける。値 m の行は「ノイズの強さ m·σ の行」として毎ステップ扱われる。LongMedia の LipSync ノードの「音声 0／映像 0.35」はこの印の値そのもの。DiffSynth の Retake は 0／1 だけ。
5. **長さの規則**: 映像の長さ F に対し、`A·800` サンプルは映像の長さより 1/3 潜在（約 8.3 ミリ秒）長いことも、短いこともある。`audio_vae.py` の説明「最大で半潜在長い」は、短くなる場合が抜けている。それでも「音声の長さ ≥ 映像の長さ」ならば、潜在が足りなくなることは無い（DiffSynth の繰り返し埋めは起きない）。
6. **コミュニティの動き**: 2026-10-19 より後に、新しい一次報告は見つからなかった。ComfyUI の issue #16387 は未解決のまま（コメント 0）。

---

## 問い 1. 音声 VAE のエンコーダーの構造

### 1.1 構造（事実）

**全体**（`DS:vae` 412〜452 行目、`DF:vae` 536〜621 行目）
- `encoder_dim=64`、`encoder_rates=[2,4,4,5,5]`。
- `latent_dim = 64·2^5 = 2048`、`hop_length = prod = 800`（DS:vae 428〜430 行目）。
- 40 潜在／秒 @ 32 kHz（DF:vae 20 行目、500〜501 行目）。

**`encoder.block`**（DS:vae 343〜357 行目、DF:vae 268〜284 行目）。nn.Sequential で、次の順に並ぶ。

| 番号 | 中身 |
|---|---|
| 0 | Conv1d(1→64, k=7, pad=3) |
| 1〜5 | EncoderBlock。幅は順に 128・256・512・1024・2048、stride は順に 2・4・4・5・5 |
| 6 | Snake(2048) |
| 7 | Conv1d(2048→2048, k=3, pad=1) |

**EncoderBlock(dim, stride)**（DS:vae 322〜340 行目）。中身は次の順。
- ResidualUnit(dim/2, dilation=1)
- ResidualUnit(dim/2, dilation=3)
- ResidualUnit(dim/2, dilation=9)
- Snake(dim/2)
- Conv1d(dim/2→dim, k=2·stride, stride, pad=ceil(stride/2))

**ResidualUnit(d, dil)**（DS:vae 303〜319 行目）
- 中身: Snake → Conv1d(k=7, dilation=dil, pad=3·dil) → Snake → Conv1d(k=1)。
- 入力を足し戻す（長さが違えば中央を切り出す）。

**活性化の違い（重要）**
- エンコーダーの Snake（DF:vae 121〜130 行目、DS:vae 142〜158 行目の `beta=False` の場合）:
  - 式は `x + 1/(α+1e-9)·sin²(αx)`。
  - α の形は `[1,C,1]`。**指数を取らない生の値**で、α と β は共通。
- デコーダーの SnakeBeta（DF:vae 133〜148 行目）:
  - 対数空間の `[C]`。`exp` を取る。
  - 本製品の `audio_vae.py` 70〜81 行目と同じもの。
- エンコーダーには、アンチエイリアス（Activation1d、折り返し雑音を防ぐ上げ下げ）が**無い**。

**`pre_block` = AttnProjection(2048→32, heads=8, mlp_ratio=2)**（DS:vae 266〜300 行目、DF:vae 304〜390 行目）
- 式: `x = proj(norm3(x)) + attn(norm1(x))`、続いて `x = x + mlp(norm2(x))`。
- 入力は `[B, T, 2048]`（エンコーダー出力を転置したもの）。

**CausalAttention**
- `qkv = Linear(2048→6144, bias なし)`。そこに `q_bias`、`zero_k_bias`、`v_bias`（各 [2048]）を加える。
- 8 ヘッド、ヘッド幅 256。`is_causal=True`。
- 出力は**ヘッド方向に平均**する（連結ではない）。
- 残ったヘッド幅 256 を `adaptive_avg_pool1d` で 32 に縮め、`proj = Linear(32→32)` を掛ける（DS:vae 277〜284 行目、DF:vae 334〜338 行目）。

**GeGluMlp(32, 64)**（DS:vae 250〜263 行目）
- LayerNorm → `gelu_tanh(w0(x)) * w1(x)` → w2。

**射影の頭**
- `mean_proj = Conv1d(32→32, k=1)`。
- `logs_proj` も同じ形（DS:vae 431〜432 行目）。

### 1.2 ファイルのテンソル名との対応（事実。手元のヘッダーで確認）

`models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors` の中身。
- `encoder.block.0.weight [64,1,7]`
- `encoder.block.{1..5}.block.{0,1,2}.block.{0,2}.alpha [1,C,1]`
- `….block.{1,3}.weight/bias`
- `encoder.block.{1..5}.block.3.alpha`
- `encoder.block.{1..5}.block.4.weight`。形は [128,64,4]、[256,128,8]、[512,256,8]、[1024,512,10]、[2048,1024,10]（k=2·stride と一致）。
- `encoder.block.6.alpha [1,2048,1]`
- `encoder.block.7.weight [2048,2048,3]`

数え方。
- encoder: 2 ＋ 5×21 ＋ 3 ＝ 110 本。
- pre_block: norm1・norm2・norm3・mlp.norm で 8 本、qkv で 1 本、偏り 3 本、attn.proj 2 本、proj 2 本、w0・w1・w2 で 6 本。計 22 本。

形と本数がすべて一致した。

畳み込みはすべて単一の `weight` になっている。重み正規化の `weight_g`／`weight_v` は、あらかじめ畳み込まれた形。本製品のデコーダーの説明（`audio_vae.py` 19〜25 行目）と同じ扱いで読める。

**名前の注意**（`DS:` の `utils/state_dict_converters/minimax_h3_audio_vae.py` 1〜9 行目）: DiffSynth は `q_bias`／`v_bias`／`zero_k_bias` を `WarpedTensor` で包む。そのため名前の後ろに `.weight` を付け足す。素の `nn.Parameter` か buffer で受ければ、ファイルの名前のまま読める。diffusers は `zero_k_bias` を buffer にしている（DF:vae 361〜363 行目）。

### 1.3 符号化の手順（事実）

DiffSynth の `encode_audio`（DS:vae 442〜471 行目）は次の順に処理する。
- 入力 `[C, L]` を `unsqueeze(1)` して `[C, 1, L]` にする。**チャンネルがバッチになる**。
- 右側を 0 で埋めて、長さを 800 の倍数にする（`preprocess`、442〜445 行目）。
- `encoder` → `pre_block`（転置して通す）→ `mean_proj` の順に通す。
- `(latent − mean) / std` を計算する（float32）。

diffusers の `encode` も同じ（DF:vae 604〜621 行目）。説明文には「MiniMax-H3 always consumes the posterior **mean** … `logs_proj` is never evaluated」（DF:vae 590〜591 行目）とある。参照音声の符号化でも `posterior.mode()`、つまり平均を使う（DF:enc 759〜763 行目）。

正規化の統計量は、DiffSynth では定数として書かれている（DS:vae 408〜409 行目）。本製品は同じ値をメタデータから読んでいる（`audio_vae.py` 27〜30 行目、221〜226 行目）。

**精度**
- diffusers: 6 つのモジュールすべてを fp32 に固定する。理由として「bfloat16 では復元音が約 20 dB 小さくなる」と書いている（DF:vae 531〜534 行目）。
- DiffSynth の Retake: `encode_audio(..., dtype=pipe.torch_dtype)` で、例では bf16（DS:pipe 413 行目、DS:ex 7〜19 行目）。
- 推測: 本製品は fp32 で揃えるのが安全。

### 1.4 本製品のデコーダーとの差（事実）と移植の見込み（推測）

本製品の `audio_vae.py` の現状。
- `encoder.`、`pre_block.`、`mean_proj.`、`logs_proj.`、`latents_mean`、`latents_std` を読み捨てている（59〜67 行目）。
- `H3AudioVAE` は `dec_in_proj` と `decoder` だけを持つ（204〜240 行目）。

足りない部品:
- 生の Snake
- ResidualUnit
- EncoderBlock
- Encoder
- GeGluMlp
- CausalAttention（`F.scaled_dot_product_attention(is_causal=True)` で書ける）
- AttnProjection
- `mean_proj`
- `encode` メソッド（0 埋め・平均・正規化）

あわせて、`audio_vae_key_map` の読み捨てから `encoder.`／`pre_block.`／`mean_proj.` を外す。`logs_proj` は読み捨てたままでよい。

移植の行数の見込み（推測）:
- モジュール: 約 130〜170 行（説明文を含む）。
- 波形の読み込みと長さの調整: 約 50〜80 行。
- 雑音除去ループの固定処理: 約 40〜80 行。
- ほかに試験。

エンコーダーの重みは fp32 で約 0.1 億パラメータ規模（推測。最大は 2048×2048×3 の畳み込みと qkv の 6144×2048）。

---

## 問い 2. 波形の前処理

### 2.1 DiffSynth（事実）

**読み込み**
- 例は `read_audio(path)` を使う（DS:ex 34 行目）。中身は torchcodec の `AudioDecoder` で、元のサンプリングレートのまま読む（DS:audio 31〜52 行目、55〜87 行目）。

**Retake の音声側**（DS:pipe 405〜429 行目）
- 形が `[1,C,T]` なら `squeeze` する。
- `convert_to_stereo`: 1 チャンネルなら複製する（DS:audio 13〜20 行目）。
- `resample_waveform`: `torchaudio.functional.resample`。既定の設定で、レートが同じなら何もしない（DS:audio 23〜28 行目）。
- `waveform[:2]`: 3 チャンネル以上は先頭 2 つだけを使う。
- **全体を符号化する** → 長さが `A` を超えた潜在は切り捨てる（416〜417 行目）。足りないときは、最後の潜在を繰り返して埋める（418〜420 行目）。その埋めた区間の印は 1（作り直す）になる（427〜428 行目）。
- ピーク正規化・音量調整・切り抜きは**無い**。クランプ（値を [−1, 1] に収める処理）はデコーダーの出力にだけある（DS:vae 404 行目）。

**潜在の数**
- `audio_latent_t = round(num_frames/24·40)`（DS:pipe 248 行目）。本製品の `constants.py` 90〜92 行目と同じ。
- 目標の数は雑音の形から決まる（414 行目）。符号化した側の数とは**完全に一致させる**（切り詰めるか埋める）。

**秒から潜在への換算**
- `latent_fps = 32000/800 = 40`。区間は `floor(start·40)〜ceil(end·40)`（422〜426 行目）。

### 2.2 diffusers（事実。参照音声の経路で、Retake は無い）

`_normalize_audio_condition`（DF:benc 347〜385 行目）の処理。
- 3 チャンネル以上は拒否する。
- 元のレートのまま `num_frames/24` 秒に切り詰める（372 行目、452〜463 行目）。
- モノラルは `expand(2, …)` で複製する。
- `torchaudio.transforms.Resample` で 1 回だけ変換する。

ファイルからの読み込みは PyAV。元のレートで `fltp`（32 ビット浮動小数・チャンネル別の並び）にする（`references.py` 239〜264 行目）。

### 2.3 本製品の環境（事実）と含意（推測）

- 事実: `.venv-engine-h3/Lib/site-packages` には `av`（PyAV 18.1.0）がある。**torchaudio は無い**（LTX 系の 2 つの venv にはある）。本製品は出力の mux（映像と音声を 1 つのファイルにまとめる処理）で、すでに `av.audio.resampler.AudioResampler` を使っている（`engine_h3/media_out.py` 88 行目、171 行目）。
- 推測: パッケージを足さずに済ませるなら、PyAV の `AudioResampler(format="fltp", layout="stereo", rate=32000)` で、読み込みと同時に 32 kHz のステレオに変換するのが素直。ffmpeg の swresample と torchaudio の sinc 補間では数値がわずかに違う。品質への影響は小さいと見るが、**未確認**。
- 推測: PyAV の `layout="stereo"` の指定は、5.1 チャンネルなどを 2 チャンネルに混ぜる処理になる。DiffSynth の「先頭 2 チャンネルだけを使う」とは挙動が違う。

---

## 問い 3. Retake の仕組み（A2V と V2A）

### 3.1 DiffSynth（事実）

**印の意味**
- `mask` は「1 = regenerate（作り直す）」、0 = 残す（DS:pipe 390 行目、423 行目）。
- 区間を指定しないと、全部 0（全部残す）になる。

**準備**（音声）
- `input_latents_audio`（正規化済みの符号化潜在）と、`denoise_mask_audio [2,1,A]` を作る（429 行目）。

**残す行の入力**
- `model_fn` が呼ばれるたび、つまり**毎ステップ**、`audio = audio·mask + input·(1−mask)` で差し替える（989〜991 行目）。

**残す行の時刻**
- 時刻の向きは `t = 1 − timestep/1000` で、1 がノイズ無し（979〜980 行目）。
- 残す行は `1.0` に固定する（1016〜1017 行目）。0.999 ではない。映像側も 1.0（1014〜1015 行目）。
- これとは別に、条件行（キーフレーム）は `max(t, 0.999)`、参照音声は `max(t, 1.0)` になる（1019〜1020 行目）。

**AdaLN の時刻**
- 行ごとの時刻の配列を作り、`torch.unique(sorted=True)` で重複を除く（1011〜1021 行目）。値 1.0 は集合に入る。
- 文字の行は映像の時刻に従う（1011 行目）。

**RoPE**
- 位置は `PackedSequenceBuilder` が形だけから作る。引数に印は無い（838 行目）。**Retake では位置は変わらない。**

**状態の更新**
- `step()` は、残す行の予測を `return_to_timestep` の値、つまり `(x − clean)/σ` に置き換えて混ぜる（DS:base 217〜227 行目、`flow_match.py` 412〜418 行目）。
- その結果の更新式は `x_next = clean + (σ'/σ)(x − clean)` になる（`flow_match.py` 400〜410 行目からの式変形）。
- 純ノイズから始まった残す行の状態は、`(1−σ)·clean + σ·noise` の軌道をたどる。**最後のステップ（σ'=0）で clean に一致する。**
- 推測: DiT（拡散の本体の変換器）は残す行を毎回 clean として受け取る（990 行目）。なので「残す行の状態を最初から clean に保ち、更新しない」実装でも、DiT への入力と最終値は同じになる（bf16 の丸めを除く）。

**初期値**
- 映像も音声も純ノイズ（`NoiseInitializer`、245〜250 行目）。A2V の映像の行は純ノイズから普通に作る。
- 例 2 は FL2VA の I2V（先頭の 1 枚）と `retake_audio` の併用（DS:ex 46〜57 行目）。**FL2VA の重みで I2V と音声固定を一緒に使うのは、公式の例にある組み合わせ。**

**出力の音声**
- 最終の音声潜在を `decode_audio` で復元する（DS:pipe 205〜207 行目）。**元の波形は差し戻さない。**
- 対して LongMedia の lip_sync は、出力で元の音声に差し戻す（前回 R4 §3.2）。

**区間ありとの違い**
- 区間内だけ 1 になる（424〜426 行目）。
- 映像は 17 フレームの VAE クリップ単位で広がる（389〜393 行目、DS:ex 82〜88 行目）。
- 音声の短さを繰り返しで埋めた末尾は、常に 1 になる。

**例の条件**
- 20 ステップ。`cfg_scale` は指定なしで、既定の 1.0（DS:pipe 108 行目、DS:ex 35〜40 行目）。

### 3.2 ComfyUI（仕様として読んだ。GPL-3.0 なのでコードは写していない）

**印の粒度**
- 映像は 2×2 パッチ、音声は潜在フレーム単位。区画の中の最大値（amax）に寄せる。
- 値を 1/256 刻みに切り上げる（`comfy/model_base.py` 2270〜2287 行目）。
- 音声の印はチャンネル方向の最大値を取る。**左右で共通。**
- どこかに 1 未満の値があるときだけ、`denoise_mask`／`audio_denoise_mask` を DiT に渡す（2289〜2298 行目）。

**時刻**
- 映像の固定値は `VISUAL_COND_TIMESTEP = 0.999`、音声は `AUDIO_COND_TIMESTEP = 1.0`（`comfy/ldm/minimax/model.py` 32〜33 行目）。
- 値 m の行の時刻は `1 − m·σ` で、それぞれ 0.999／1.0 を上限にする（683〜707 行目）。m = 0 なら固定値になる。
- 行ごとの値も、重複を除いた時刻の集合に加える（709〜712 行目）。

**入れ直し**
- 汎用の `KSamplerX0Inpaint` が毎ステップ処理する（`comfy/samplers.py` 634〜643 行目）。
  - 入力は `x·m + inject·(1−m)`。
  - 出力（ノイズを除いた推定）は `out·m + clean·(1−m)`。
- H3 の `scale_latent_inpaint` は、映像には `0.999·clean + 0.001·noise` を、音声には clean を入れる（2303〜2327 行目）。
- DiT の速度出力には m を掛ける（`model.py` 637〜641 行目）。

---

## 問い 4. 部分的な固定（強さ）

### 4.1 前例（事実）

**DiffSynth の Retake**
- 印は 0／1 だけ（DS:pipe 390〜393 行目、423〜428 行目）。連続値の前例は無い。

**ComfyUI の H3**
- 連続値の印を正式に扱う。説明文は「mask value m puts a row at sigma = m * sigma_stream」（`model.py` 683〜684 行目）。
- 機械的には次のとおり。毎ステップ、その行は「clean を m·σ までノイズ化した値」として DiT に入る。
  - 入力 `m·x + (1−m)·clean` は、x が `(1−σ)clean + σn` の軌道にあるとき、`clean + mσ(n−clean)` に等しい（式変形）。
  - 時刻の印は `1−mσ`。
  - 出力は `m·out + (1−m)·clean` に混ぜる。
- 推測: その行の最終値は clean に戻り切らない。生成した音声は、m に応じて変わる。

**LongMedia の LipSync Latent Setup**（`nodes.py` 18614〜18649 行目、`latent_ops.py` 186〜209 行目）
- `replace_stream` で音声の流れを差し替える。
- `set_stream_denoise` で、流れごとに**一定値の noise_mask**（「0=preserve, 1=fully denoise」）を付ける。
- 既定値は `video_denoise=0.35`、`audio_denoise=0.0`。
- つまり「音声 0」は完全固定（時刻 1.0）。「映像 0.35」は、元の映像の潜在を 0.35·σ の強さで毎ステップ作り直すこと（上の ComfyUI の仕組み）。
- 推測: 映像 0.35 は、既存の映像の口元を直す「部分的な作り直し」の用途。純ノイズから作る A2V の設定ではない。

**汎用の前例**
- ComfyUI の DifferentialDiffusion（`comfy_extras/nodes_differential_diffusion.py` 39〜61 行目）。
- 進み具合の閾値（1 から 0 へ下がる）と印の値を比べ、ステップごとに 0／1 を切り替える。値 m の行は前半は固定され、終盤の m の割合だけ作り直される。
- H3 で使った報告は**未確認**。

### 4.2 強さ s（1.0 = 完全固定）の候補（推測。評価は未実測）

| 案 | 定義 | 前例 | 利点 | 欠点 |
|---|---|---|---|---|
| (a) ノイズ増し | 音声の行に `s·clean + (1−s)·noise` を置き、時刻 `max(t_a, s)` で毎ステップ固定する（キーフレームの条件行と同じ形）。 | H3 の条件行の `imgvid_cond_noise_aug=0.999`／`audio_cond_noise_aug=1.0`（DS:pipe 38〜39 行目、1019〜1020 行目）。値は属性で変えられる。0.999／1.0 以外の値で学習したかは**未確認**。 | 実装が最小（本製品にはキーフレームで同じ部品がある。`pipeline_h3.py` 257〜270 行目、`noisy_condition`）。音声の行は動かないので、「元の波形を mux する」と矛盾しない。 | s の意味は「モデルが音声をどれだけ信じるか」。s を下げても口の動きの自由度が増す保証は無い。s が 1.0 から離れると、学習時の分布から外れる。 |
| (b) 連続値の印（ComfyUI 式） | 印 m = 1−s。行の時刻は `1−mσ`、入力は `m·x+(1−m)·clean`、出力は `m·out+(1−m)·clean` に混ぜる。 | ComfyUI 本体の H3 実装と LongMedia（上記）。 | H3 での一次資料の前例がある。σ が下がるにつれて自然に clean に近づく。 | 生成した音声は元とずれる（mux で捨てるので無害だが、映像は「ずれた音声」に合わせて作られる）。ComfyUI の音声のスケール処理（`audio_scale`）など、細部を独自に起こす必要がある。 |
| (c) 途中から解放（SDEdit 型） | `σ > 1−s` の間は固定し、それ以降は普通に作り直す。解放する時点の状態は `(1−σ)clean+σn`。 | DifferentialDiffusion（汎用）。H3 での前例は**未確認**。 | 意味が「音声をどこまで作り直してよいか」で分かりやすい。 | 解放した後は音声が自由に変わるので、口の同期が崩れる恐れがある。製品の目的（元の音声に合わせる）と逆向き。 |

推測: 「元の波形を mux して映像だけを合わせる」という製品の目的に一番沿うのは (a)。前例の強さで選ぶなら (b)。どちらも、s = 1.0 では DiffSynth の Retake と同じになる。

---

## 問い 5. 「映像の長さが主・音声は映像以上・切り詰めのみ」に効く H3 の事情

**丸め**（事実と計算）
- `A = round(5F/3)`。F = 17n＋5 なので、`5F/3` の小数部は 0・1/3・2/3 のどれか。0.5 にはならないので、Python の偶数丸めは効かない。
- `A·800` サンプルと映像の長さ `F·32000/24` の差は、次の 3 通り。
  - n ≡ 1（mod 3）: ＋266.7 サンプル
  - n ≡ 0: −266.7 サンプル（約 8.3 ミリ秒）
  - n ≡ 2: 0

  例: F=22 は＋、F=56 は−、F=39 はちょうど。
- `audio_vae.py` 41〜43 行目の「up to half a latent longer」は、**短くなる場合（n ≡ 0）を書いていない**。
- `media_out.py` 154〜156 行目は `min(available, round(F·32000/24))` で切る（埋めはしない）。そのため n ≡ 0 では、今の T2V の出力でも、音声が映像より約 8.3 ミリ秒短い（計算からの事実。実ファイルでの確認は**未確認**）。

**A2V への含意**（推測）
- 元の波形を mux するなら、波形は映像の長さ以上ある。`round(F·32000/24)` で切ればちょうど揃い、上の差は消える。
- 符号化の側: 長さ L が `F·4000/3` 以上なら、`ceil(L/800) ≥ ceil(5F/3) ≥ A`。**潜在は必ず足りる**。DiffSynth の繰り返し埋め（作り直す印付き）は起きない。
- n ≡ 1 で、音声がちょうど映像の長さしかない場合、最後の潜在には最大 267 サンプルの 0 埋めが混ざる。
- 畳み込みは前後両方向を見る（DS:vae 306 行目の左右対称の埋め）。なので末尾の数潜在は、波形の切れ目の影響を受けうる。DiffSynth のように「手に入る分を（少し余分に）符号化してから潜在を切る」ほうが、末尾は安定すると見る。ただし、長い音源を全部符号化するのは無駄なので、`A·800` に 1 秒程度の余裕を足して切り出してから符号化する案を挙げる。

**ステレオ**（事実）
- `pack_audio` はチャンネル優先の並び（`DS:` の `models/minimax_h3_dit.py` 32〜35 行目）。本製品の `audio_to_rows`（ch0 の次に ch1）と一致する。
- モノラルの入力は複製するので、2 つのチャンネルは同じ潜在になる。

**開始位置**（事実）
- DiffSynth は波形の 0 サンプル目から符号化する。映像の 0 フレーム目とずらす処理は無い。

---

## 問い 6. コミュニティの動き（2026-10-19 より後）

- ComfyUI の issue #16387「The lip-sync feature for digital human MV in MiniMax H3 cannot be used in version 0.35 and later releases.」: **open**。作成は 2026-09-17 で、更新も同じ日。コメントは 0（GitHub API で 2026-10-25 に確認）。
- 2026-10-01 以降の ComfyUI の issue で「H3 lip」「minimax audio mask」を検索した。関係する新しいものは無かった。
- HF のディスカッション #91: コメントは 2 件のまま。最後は rocky533 の 09-12。rocky533 は「Ref2VA で、音声チャンネルを latent noise mask してリップシンクできる」外部ワークフローを挙げている。中身は**未確認**。
  - https://huggingface.co/MiniMaxAI/MiniMax-H3/discussions/91
- TheLocalLab の Patreon「FREE MiniMax H3 Audio Lip Sync ComfyUI Workflow」: 検索の要約によれば、ref2va を使う方式。本文は 403 で読めず、日付・方式とも**未確認**。
  - https://www.patreon.com/TheLocalLab/posts/free-minimax-h3-167084390
- FL2VA＋音声固定と Ref2VA の品質を数値で比べた 10-19 以降の報告は、見つからなかった。

---

## 出典

DiffSynth-Studio `4e34969`（https://github.com/modelscope/DiffSynth-Studio）:
- `diffsynth/models/minimax_h3_audio_vae.py`
- `diffsynth/pipelines/minimax_h3_audio_video.py`
- `diffsynth/diffusion/base_pipeline.py`
- `diffsynth/diffusion/flow_match.py`
- `diffsynth/utils/data/audio.py`
- `diffsynth/utils/state_dict_converters/minimax_h3_audio_vae.py`
- `diffsynth/models/minimax_h3_dit.py`
- `examples/minimax_h3/model_inference/MiniMax-H3-Retake.py`

diffusers `aaf57d0`:
- `src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py`
- `src/diffusers/modular_pipelines/minimax_h3/encoders.py`
- `src/diffusers/modular_pipelines/minimax_h3/before_encoder.py`
- `src/diffusers/modular_pipelines/minimax_h3/references.py`

ComfyUI `19ae172`（仕様として読んだだけ）:
- `comfy/model_base.py`（class MiniMaxH3、2191 行目〜）
- `comfy/ldm/minimax/model.py`
- `comfy/samplers.py`
- `comfy_extras/nodes_differential_diffusion.py`
- `comfy_extras/nodes_minimax_h3.py`

LongMedia `409e4cb`:
- `nodes.py`
- `latent_ops.py`
- `docs/AUDIO_MODES_GUIDE_EN.md`

本製品:
- `engine_h3/audio_vae.py`
- `engine_h3/pipeline_h3.py`（257〜270 行目、997〜1041 行目）
- `engine_h3/media_out.py`（154〜156 行目）
- `engine_h3/constants.py`（37 行目、90〜92 行目）
- `engine_h3/scheduler.py`（31〜50 行目）
- 重みファイルのヘッダー
- 前回の報告: `Docs/Outputs-archive/h3-roadmap-2026-10-19/R4_fl2va_ref2va_a2v.md`
