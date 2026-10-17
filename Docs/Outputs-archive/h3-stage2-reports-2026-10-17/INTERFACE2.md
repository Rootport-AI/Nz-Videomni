# engine_h3 第 2 弾 2a 後半（I2V と常駐）の境界の仕様 — 草稿

作成: 2026-10-17／担当 C1（監督の確認待ちの草稿）。計画は `Docs/Outputs-archive/h3-stage2-reports-2026-10-17/PLAN_stage2.md`（§3.1・§3.2・§8）、レビューは同 `REVIEW_plan2.md`（M-7・M-8・m-7〜m-12）。第 1 弾の境界は `Docs/Outputs-archive/h3-stage1-reports-2026-10-17/INTERFACE.md`。

**ここに書いた名前・形・dtype・式は、監督の確認後は担当が勝手に変えない。** 計画と食い違う箇所は無いつもりで書いた。計画が「担当が参照実装から写す」とした事項（§1〜§3）は、写した元の行を各項に付けた。

参照実装の所在（写しはすべて一時領域 `scratchpad\h3\src\`）:

- DiffSynth-Studio コミット `4e34969`（Apache-2.0）: `diffsynth/models/minimax_h3_text_encoder.py`（以下 DS-TE）・`diffsynth/pipelines/minimax_h3_audio_video.py`（以下 DS-P）・`diffsynth/models/minimax_h3_video_vae.py`（以下 DS-VAE）・`diffsynth/utils/state_dict_converters/minimax_h3_text_encoder.py`（以下 DS-CONV）
- diffusers コミット `b800f82`（Apache-2.0）: `src/diffusers/modular_pipelines/minimax_h3/encoders.py`（以下 DF-ENC）・`before_denoise.py`（以下 DF-BD）
- 本製品の仮想環境 `.venv-engine-h3` の transformers 5.14.1: `transformers/models/qwen3_vl/modeling_qwen3_vl.py`（以下 HF）

---

## 0. 確かめた事実（ヘッダと設定ファイル。GPU は使っていない）

| 事項 | 結果 | 確かめ方 |
|---|---|---|
| 視覚塔のテンソルの型 | **全部 BF16**（351 本・1.11GiB）。量子化されていない（`comfy_quant`／`weight_scale` は 0 本） | TE ファイル `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` のヘッダ。`visual.*` の名前と形は `Qwen3VLModel`（meta で組んだもの）の `visual.*` と 1 対 1 で過不足なし |
| 視覚塔の bf16 の重み（レビュー m-11） | 上のとおり TE ファイルの中にある。**取得は不要** | 同上 |
| プロセッサの設定 | 手元の `models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer/preprocessor_config.json` は、DiffSynth が読む公式 `MiniMaxAI/MiniMax-H3` の `FL2VA/processor/preprocessor_config.json` と**同一のファイル**（git の blob id `2ea84a4…` が一致）。tokenizer 一式（`tokenizer.json`・`tokenizer_config.json`・`vocab.json`・`merges.txt`）も同一。processor 側にだけある `chat_template.json` と `video_preprocessor_config.json` は I2V（先頭の画像 1 枚）では使わない | HF の API で公式フォルダの一覧と blob id を取り、手元のファイルの `git hash-object` と比べた（小さな JSON の読み取りだけ） |
| プロセッサの値 | `patch_size 16`・`merge_size 2`・`temporal_patch_size 2`・`size.shortest_edge 65536`（画素数の下限）・`size.longest_edge 16777216`（画素数の上限）・`image_mean = image_std = 0.5`（0〜1 を −1〜1 へ）・拡縮は BICUBIC・`do_rescale`（1/255） | `AutoImageProcessor.from_pretrained(<tokenizer フォルダ>)` を CPU で作って属性を読んだ（transformers 5.14.1 では `Qwen2VLImageProcessor` が返る） |
| 視覚のトークン ID | `<|vision_start|>` 151652・`<|image_pad|>` 151655・`<|vision_end|>` 151653（`config.json` の `vision_start_token_id`・`image_token_id`・`vision_end_token_id` と一致） | tokenizer の `convert_tokens_to_ids` |
| トークン数（実測） | 640×384 → 格子 `[1, 24, 40]`・240 トークン／1344×768 → `[1, 48, 84]`・1008 トークン／192×192 → `[1, 16, 16]`・64 トークン（下限 65536 画素に満たないので 256×256 へ拡大される） | 同上の画像プロセッサに乱数の画像を通した（CPU） |
| `Qwen3VLModel` の meta 上の非永続バッファ | `visual.rotary_pos_emb.inv_freq`（18・f32）・`language_model.rotary_emb.inv_freq`／`original_inv_freq`（64・f32）の 3 本。ファイルには無いので meta のまま残る → CPU で計算して入れる（§1.4） | meta で組んで `named_buffers()` を列挙 |

---

## 1. 視覚塔とテキストエンコーダ（`engine_h3/text_encoder.py`）

### 1.1 モデルの組み方

- `Qwen3VLTextModel` をやめて **`Qwen3VLModel`（文章＋視覚）を meta で組む**（T2V も I2V も同じ型。T2V では視覚塔を一度も読まない＝meta のまま呼ばれない）。設定は手元の `config.json`（DS-TE 20-71 の辞書と同じ値）を読み、`text_config.num_hidden_layers` を 50 に切る。`language_model.norm = nn.Identity()`（第 1 弾と同じく、50 層目の後の正規化前の隠れ状態を出す）。`_attn_implementation = "sdpa"`（文章側・視覚側とも）。
- 名前の写し（DS-CONV 30-45 の Comfy-Org 版に揃える）:
  - `model.layers.N.*`（N < 50）→ `language_model.layers.N.*`
  - `model.embed_tokens.*` → `language_model.embed_tokens.*`
  - `visual.*` → `visual.*`（そのまま）
  - `lm_head`・`model.norm`・N ≥ 50 の層は読まない
- 量子化層（350 本の `H3Int8Linear`）の名前も同じ写しで `language_model.layers.N.*` に付け替える。視覚塔は量子化層が無いので素の `nn.Linear`（bf16）のまま。

### 1.2 直接読み（2a 前半で実装済みの仕組みをそのまま広げる）

- 文章側の層: 前半で入れたフック（`install_layer_read_hooks`）を `language_model.layers` に掛ける。読む表は `te_read_plan` を新しい名前に合わせる。
- 埋め込み表（`language_model.embed_tokens`）: 計算の前に GPU へ読み、計算の後に meta へ戻す（前半と同じ）。
- **視覚塔（I2V のときだけ）**: 計算の前に `visual.*` 全部（1.11GiB・bf16）を GPU へ読み、計算の後に meta へ戻す。視覚塔の前向きは HF の `Qwen3VLModel.forward` の中の `get_image_features` が 1 回だけ呼ぶので、塔全体を先に置く（層ごとのフックにしない。小さいので不要）。
- **HF の `forward` は書き直さない**（レビュー M-7）。DeepStack（視覚塔の 8・16・24 層目の特徴を文章側の先頭 3 層の出力に足す処理。HF 849-858 `_deepstack_process`）と、画像トークンの 3 軸の位置（HF 927-1021 `get_rope_index`）は HF の中で起きる。文章側の層を読むフックは `Module.__call__` から呼ばれるので、DeepStack の足し込み（層の呼び出しの**後**に層の外で行われる）とは干渉しない。

### 1.3 前向きの呼び方（DS-TE 78-106・DF-ENC 72-91 と同じ）

```python
model(                                   # Qwen3VLModel
    input_ids=ids[None],                 # [1, L] long
    attention_mask=torch.ones_like(...),
    mm_token_type_ids=mm,                # [1, L]: <|image_pad|> の位置だけ 1、他は 0（DS-TE 88-92・HF processing_utils.create_mm_token_type_ids と同じ）
    pixel_values=pv.to(device, bf16),    # I2V のときだけ。[N_patch, 1536]（DS-P 314・DF-ENC 76 とも視覚塔の型 bf16 へ）
    image_grid_thw=grid.to(device),      # I2V のときだけ。[1, 3] long
    use_cache=False,
)
hidden = out.last_hidden_state[0]        # [L, 5120] bf16（50 層目の後・正規化前）
```

- transformers 5.14.1 は画像があるのに `mm_token_type_ids` が無いと例外を出す（HF 1117-1122）ので必須。
- `Qwen3VLModel.rope_deltas` はインスタンスに残るが、TE はジョブごとに組み直すので持ち越しは起きない（念のため前向きの前に `None` に戻す）。
- T2V では `pixel_values` も `mm_token_type_ids` も渡さない。このとき `Qwen3VLModel.forward` は埋め込みを引いて `language_model(inputs_embeds=..., position_ids=None)` を呼ぶだけで、位置は第 1 弾の `Qwen3VLTextModel` と同じ `arange` になる（HF 1233-1250・同 `Qwen3VLTextModel.forward` の `position_ids is None` の分岐）。**T2V の出力は第 1 弾とビット一致する見込み**で、2a-G2 の台本（`outputs/h3-gate2/g2_te_direct.py --mode compare`）と T2V の指紋で確かめる。

### 1.4 非永続バッファ

- 文章側 `language_model.rotary_emb`: 第 1 弾と同じく `compute_default_rope_parameters(config.text_config, device="cpu")` の f32。
- 視覚側 `visual.rotary_pos_emb.inv_freq`: `1.0 / (10000 ** (torch.arange(0, 36, 2, dtype=float32) / 36))`（HF `Qwen3VLVisionRotaryEmbedding.__init__` と同じ式。dim = 1152/16/2 = 36）を CPU で作る。読み込みの後に meta のまま残ったテンソルが無いことを、視覚塔を読んだ時点で検査する（T2V では視覚塔を検査の対象から外す）。

### 1.5 トークン列の書式（DS-TE 133-169・DF-ENC 282-299 の `fl2va`。両者は同じ）

```
"<Picture 1>: "  を tokenizer(…, add_special_tokens=False) で単独にトークン化   → タグ 1（文章）
[151652] + [151655] × N + [151653]                                         → タグ 0（映像）
プロンプトを tokenizer(…, add_special_tokens=False) で単独にトークン化         → タグ 1（文章）
```

- 3 つは**別々にトークン化して ID を連結する**（つなげた文字列をまとめてトークン化すると、`": "` の空白が次の語とくっつく。実測: `"<Picture 1>: "` 単独は `[21604, 3826, 220, 16, 26818, 220]`）。チャットテンプレートは使わない。
- 画像は 1 枚だけなので番号は常に `1`。
- `N` は画像プロセッサが返す `image_grid_thw` の積 ÷ `merge_size²`（= 4）。キャンバスが 32 の倍数で画素数が 65536〜16777216 のときは `N = (H/32) × (W/32)`。
- タグは TE の戻り値 `tags`（[L] long）に入れる。パイプラインの `modality_rows` は文章の行に TE のタグをそのまま写すので、視覚ブロックの行は AdaLN の映像（0）の行を使う（DS-P 838-849 `packed["token_tags"][text_pos] = text_token_tags`・DF-ENC 287-299 と同じ）。

### 1.6 視覚塔に渡す画像（計画 §3.1・レビュー m-10）

- **§2 のキャンバスに合わせた後の画像**（覆う拡縮＋中央切り抜き・LANCZOS の結果。PIL の RGB・キャンバスと同じ大きさ）を `processor.image_processor(images=[img], return_tensors="pt")` に渡す。DiffSynth は元画像を引き伸ばしてから渡す（DS-P 304-306）が、本製品は VAE と同じ切り抜き後の画像を渡す（計画の決めごと）。2a-G3 のトークン列の照合では、DiffSynth 側にも同じ切り抜き後の画像を渡す（PIL の `resize` は同じ大きさなら複写を返すので、DiffSynth の引き伸ばしは素通りになる）。
- プロセッサは手元の tokenizer フォルダから `AutoProcessor.from_pretrained(tokenizer_dir)` で作る（tokenizer も同じ物を使う。追加特殊トークン 7 語の検査は第 1 弾のまま）。
- `pixel_values` は f32 で返るので、視覚塔の型（bf16）へ変換して渡す（DS-P 314・DF-ENC 76・HF 1053 と同じ）。

---

## 2. 画像の読み込みとキャンバスへの合わせ方（新設 `engine_h3/image_cond.py`）

計画 §3.1 の決めごと（幾何は LTX の規約、補間は H3 の 3 実装の LANCZOS、H.264 の再圧縮はしない）を、式まで固定する。

```python
def load_canvas_image(path: str, width: int, height: int) -> PIL.Image.Image
    # 1. Image.open(path) → ImageOps.exif_transpose → convert("RGB")（アップロード時の正規化と同じ。RGBA は透明度を捨てる）
    # 2. scale = max(height / src_h, width / src_w)
    #    new_h = math.ceil(src_h * scale); new_w = math.ceil(src_w * scale)   … LTX の resize_and_center_crop と同じ丸め
    # 3. img.resize((new_w, new_h), Image.LANCZOS)                            … 同じ大きさなら PIL は複写を返す
    # 4. top = (new_h - height) // 2; left = (new_w - width) // 2
    #    img.crop((left, top, left + width, top + height))
    # 返り値はキャンバスちょうど（width × height）の RGB 画像。VAE と視覚塔の両方に渡す

def canvas_to_tensor(img) -> torch.Tensor
    # np.asarray(img, uint8) → torch f32 / 255.0 → [1, 3, 1, H, W]（0〜1。ImageNet の正規化は符号化器の中）
```

- 画像の存在・読めること・寸法の正しさは、テキストエンコーダより**前**に確かめる（読めない画像で 27GB を読ませない）。
- ImageNet の正規化（0〜1 を基準）は `encode` の中で行う。LTX の −1〜1 は持ち込まない（計画 §8）。

---

## 3. 条件行（`engine_h3/pipeline_h3.py`）

3 実装で一致する 4 点（R4a ①-5）を、本製品の形に固定する。

| 事項 | 決めごと | 写した元 |
|---|---|---|
| 系列の並び | `[text(L) | cond(grid) | audio(2A) | video(T'×grid)]`。`H3SeqLayout.make(cond_rows=grid)`。`grid = (H/32) × (W/32)` | DS-P 651-673 `_build_packed_fl2va` |
| 条件行の値 | `clean = patchify_video(z)`（z は §4 の符号化結果 `[1, 24, 1, H/16, W/16]` f32 → `[grid, 96]` f32）。`noise = torch.randn((grid, 96), generator=gen, dtype=torch.float32)`。`cond = KEYFRAME_NOISE_AUG * clean + (1.0 - KEYFRAME_NOISE_AUG) * noise`（f32。`KEYFRAME_NOISE_AUG = 0.999` は `constants.py` に既にある） | 式は DS-P 461-464・DF-BD 966-969（`scale_noise` = `t·sample + (1−t)·noise`）・ComfyUI と同じ形。行の形 [行数, 96]・f32 は ComfyUI の作り方 |
| 乱数の引き順 | 1 つの CPU 生成器（`torch.Generator("cpu").manual_seed(seed)`）から **映像の初期乱数 → 音声の初期乱数 → 条件行の乱数**（今の 2 本の後に 1 本足す）。**1 回だけ引き、全ステップで使い回す** | 計画 §3.1・レビュー m-8。T2V では 3 本目を引かないので T2V の指紋は変わらない |
| 毎ステップの入れ直し | `cond` は denoise の状態（`video`／`audio`）に入れず、毎ステップ `dit.embed_rows(text_refined, audio_rows, video_rows, cond_rows=cond)` に同じ値を渡す。DiT の出力は `final_rows` が `layout.video` と `layout.audio` だけを返すので、条件行は更新されない | DS-P 998-1003（毎回 `index_copy_`）・1074（条件行の出力を捨てる）、DF-BD 989-993 |
| 時刻 | 条件行の t = `max(t_video, KEYFRAME_NOISE_AUG)`（Python の float で求めて f32 の行に入れる）。`time_rows` に `t[layout.cond] = max(t_video, KEYFRAME_NOISE_AUG)` を足し、`unique` で番号を振る（M は最大 3 になる） | DS-P 1019・DF-BD 1261 |
| AdaLN のタグ | 条件行は映像（0）。`modality_rows` は今も cond を 0 にしている（変更なし） | DS-P 693-696 |
| RoPE | `build_coords(..., cond_frames=1)`。条件行の時間は t = L（L は視覚ブロックを含むテキストの全長）、縦横は映像の格子と同じ。G4 A で座標は照合済み | DS-P 678-689（`cond_t = float(text_len)`） |
| 条件行の埋め込み | `embed_rows` の `cond_rows` 引数（既存）。`video_patch_proj`（f32 の島）で射影してから bf16 | 既存の `dit.py` |

- DiffSynth は条件行を bf16 で混ぜ、乱数も別の作り方をする。**ビット一致は求めない**（計画 §3.1）。2a-G3 では並び・時刻の番号・RoPE の座標を float64 で完全一致させ、値は符号化の出力で比べる。

---

## 4. 映像 VAE を符号化器と復号器に分ける（`engine_h3/video_vae.py`。レビュー M-8・§1-99）

```python
class _H3VideoVAEBase(nn.Module)
    # 共通: latents_mean / latents_std（f32 [24]・__metadata__ の JSON から）・split_tiles・blend・_stitch・_pixel_stats・
    #       タイルの定数（256／64）・時間の定数（clip_length 17・token_drop 3 など）

class H3VideoEncoder(_H3VideoVAEBase)           # encoder（EncoderFCN3D）+ quant_conv
    @classmethod
    def from_file(cls, path, *, device, dtype=torch.float32) -> "H3VideoEncoder"
        # meta で組み、encoder.* と quant_conv.* だけを読み（ファイルは F16）、device へ移してから dtype へ 1 回だけ変換。
        # 島は無い（全部 dtype）。F16 → f32 は値が変わらない。dtype=bf16 は 2a-G3 で DiffSynth（bf16）とビット比較するためだけに使う
    def encode(self, video, *, tiled=True) -> Tensor
        # [1, 3, F, H, W] f32 0〜1 → [1, 24, T', H/16, W/16] f32（事後分布の平均・latents_mean/std で正規化済み）。
        # 今の H3VideoVAE.encode の本体をそのまま移す（F=1 は複製しない・タイルは常に 256／64）

class H3VideoDecoder(_H3VideoVAEBase)           # post_quant_conv + decoder（ViT3DDecoder）
    @classmethod
    def from_file(cls, path, *, device, dtype=torch.bfloat16) -> "H3VideoDecoder"
        # meta で組み、post_quant_conv.* と decoder.*（mask_token は捨てる）だけを読む。device へ移してから
        # 島（post_quant_conv・x_embedder・norm_out・各ブロックの norm1/norm2/scale1/scale2）を f32、他を dtype へ 1 回だけ変換（今の apply_precision と同じ）
    def decode(self, latents, *, tiled=True) -> Tensor
        # 2a 前半で入れた契約のまま: [1, 24, T', h, w] f32 → CPU の uint8 [F, H, W, 3]

def encoder_key_map(name) -> str | None   # encoder.* / quant_conv.* 以外は None
def decoder_key_map(name) -> str | None   # decoder.*（mask_token を除く）/ post_quant_conv.* 以外は None
```

- `H3VideoVAE`（全体の型）は**消す**（引き算）。テスト（`tests/test_h3_media.py` の `tiny_vae`）は `H3VideoEncoder` と `H3VideoDecoder` を小さい幅で組む形に書き換える。
- `sft_load.load_streaming` の「読み残しは例外」はそのまま効く（それぞれの型は自分のテンソルしか持たないので、読む範囲を名前の写しで絞れば足りる）。
- T2V は `H3VideoDecoder` だけを組む（符号化器 0.34GiB を読まない＝§1-99 の吸収）。I2V は符号化器を f32 で組み、1 枚を符号化して**すぐ捨てる**（常駐しない）。復号器の重み・島・変換の順は今と同じなので、**T2V の指紋は変わらない見込み**（2a-G4 の回帰で確かめる）。
- 符号化の場所: テキストエンコーダを解放した後、DiT を組む前（`h3.phases.image_encode`）。

---

## 5. ワーカーの受け口と `done`（`engine_h3/worker.py`。計画 §3.2 の表をキー名まで固定）

### 5.1 `generate` に足すキー（`_GENERATE_FIELDS` と黄金キーのテストに足す）

| キー | 形 | 無いとき | 検査 |
|---|---|---|---|
| `images` | `[{"path": str, "frame_idx": int}]` | `[]`（T2V） | **0 個か 1 個・`frame_idx == 0` だけ**。2 個以上・`frame_idx != 0`・`path` が無い・ファイルが無い／開けないは日本語の理由つきの `ValueError`（ジョブは `error`、プロセスは生き続ける）。要素の中の知らないキー（例: `strength`）は無視してログに 1 回出す（最上位の知らないキーと同じ扱い。アダプタは送らない） |
| `keep_resident` | bool | `False`（＝保持しているものを解放する。LTX と同じ契約） | — |
| `attention_backend` | `"sdpa"`／`"sage"` | load の値 | 今の `_resolve_attention` をジョブごとに呼ぶ（sage が無ければ `"sage->sdpa"`） |
| `block_swap_prefetch` | bool | load の値 | — |

- `H3GenerateRequest` に `images: tuple[H3ImageInput, ...] = ()`・`keep_resident: bool = False`・`attention_backend: str | None = None`・`block_swap_prefetch: bool | None = None` を足す（`H3ImageInput` は `path: str`・`frame_idx: int` の dataclass）。
- セルフテストに `--image P`（先頭フレーム 1 枚。`frame_idx` は 0 固定）と `--keep-resident` を足す。

### 5.2 `done` に足すキー（`GOLDEN_DONE_KEYS` の並び）

```
… "attention_used", "block_swap_prefetch_used", "keep_resident_used", "images_used", "h3"
```

- `keep_resident_used`: `"on"`／`"off"`（このジョブの終わりに DiT と復号器と音声 VAE を保持したか）。
- `images_used`: 0 か 1。
- `attention_used` はジョブごとの値になる（今は load の値）。
- `h3.phases` に、I2V のときだけ `image_encode`（`seconds`・`peak_vram_mb`・`peak_vram_reserved_mb`・`vision_tokens`・`canvas`・`source_size`）を足す。`h3` の最上位のキー（`phases`・`quant_backends`）は変えない。
- 進捗: `encode` の総数は T2V で 3（今のまま）、I2V で 4（4 つ目が画像の符号化）。

### 5.3 `load` から消すキー

- `te_layers_on_gpu`（2a 前半で消した。黄金キーも更新済み）。

---

## 6. 2a-G3 の台本で比べるもの（計画 §6 の具体化。台本は `outputs/h3-gate2/`）

| 比べるもの | 正解側 | 合否 |
|---|---|---|
| トークン列（書式込み）とタグ | DS-TE `presentation_fl2va` ＋ `image_token_counts`（同じ切り抜き後の画像・同じ tokenizer フォルダ） | ID とタグが完全一致 |
| 視覚塔＋TE の出力 `[L, 5120]` | DiffSynth の `MiniMaxH3TextEncoder` に同じ TE ファイルを DS-CONV の Comfy-Org 版で読ませる（文章側の量子化層は第 1 弾の G4-TE と同じく bf16 に戻した `nn.Linear`、視覚塔は同じ bf16） | 第 1 弾の G4-TE と同じ水準（int8 と bf16 の差。末尾の層で 3e-2 台） |
| 符号化（bf16 どうし） | DS-VAE `encode_video(..., dtype=bf16, process_image=True)` | ビット一致を期待。一致しなければ差の最大値を記録し、上限は G4 の水準（1e-2 台） |
| 符号化（f32 と bf16 の差） | 本製品の f32 と bf16 | 記録のみ |
| 条件行の並び・時刻の番号・RoPE の座標 | DS-P `_build_packed_fl2va`（第 1 弾の `oracle/diffsynth_h3/pipeline_coords.py` を使う） | float64 で完全一致 |

---

## 7. 決めていない点（監督の判断を仰ぎたいもの）

1. **`images` の要素の知らないキー**: §5.1 では「無視してログ」とした（最上位のキーと同じ扱い）。計画 §3.1 は「`images` は `{path, frame_idx}` だけ」としか書いていないので、拒否（`ValueError`）にするなら指示がほしい。
2. **視覚塔の検査の範囲**: T2V で視覚塔を meta のまま残すので、第 1 弾の「読み込みの後に meta が残っていないこと」の検査は、T2V では文章側だけ、I2V では視覚塔も含める形にする（§1.4）。設計正本 §6 の「meta のまま残ったテンソルが無いことを検査する」の言い回しとずれるので、文書係への申し送りが要る。
3. **`h3.phases.image_encode` の中身**: §5.2 の 6 項目で足りるか。

---

## §7 の裁定（監督・2026-10-17）

1. `images` の要素の知らないキーは「無視してログに 1 回出す」で確定。
2. 視覚塔の検査の範囲は §1.4 のとおり（T2V は文章側だけ・I2V は視覚塔も含める）で確定。設計正本 §6 の言い回しの修正は文書係へ申し送る。
3. `h3.phases.image_encode` は §5.2 の 6 項目で確定。

---

## 8. 常駐とジョブごとの設定（`engine_h3/pipeline_h3.py`・`worker.py`。計画 §3.2・レビュー M-1・M-9）

### 8.1 保持するもの・しないもの

| 物 | `keep_resident` が真のジョブの終わり | 置き場 |
|---|---|---|
| DiT（`H3DiT`） | 保持。**ブロックもブロック以外も全部 CPU へ戻す**（`move_module_tree(dit, cpu)`。同期経路が最後の窓のブロックを GPU に残すので、全テンソルについて GPU にあれば CPU へ移す。先読み経路は `teardown_prefetch()` で CPU の正本を指し直すので、移す物は無い） | CPU |
| 映像の復号器（`H3VideoDecoder`） | 保持。復号の後に `decoder.to("cpu")`（型と島はそのまま） | CPU |
| 音声 VAE（`H3AudioVAE`） | 保持。復号の後に `.to("cpu")` | CPU |
| 映像の符号化器（`H3VideoEncoder`） | **保持しない**（I2V のジョブごとに f32 で組んで捨てる） | — |
| テキストエンコーダ | 保持しない（直接読み。主メモリに載らない） | — |
| `BlockSwapService` | 今までどおりプロセスに 1 つ（ページ固定のプールと転送用ストリームはジョブをまたぐ） | — |

- **GPU には何も残さない**: 保持物を CPU へ戻した後、`_free_cuda()`（`gc.collect` と `empty_cache`）。合間の `torch.cuda.memory_allocated()` は 0 を目標にする（2a-G4 で測る）。
- **キーが無い（偽の）ジョブ**: 保持物があればそのジョブでは**使い回し**（読み込みを飛ばす）、ジョブの終わりに**解放**する（LTX と同じ「キーが真のときだけ保持し、キーが無いジョブで解放する」）。
- **失敗したジョブ**: 保持物を丸ごと解放する（どの段で止まったか分からない物を次のジョブに渡さない。次のジョブはファイルから読み直す）。
- **使い回すときの読み込みの省略**: DiT はヘッダの読み直し・meta での組み立て・ファイルからの読み込みを全部飛ばす。復号器と音声 VAE は `from_file` を飛ばして `.to(device)` だけ。`h3.phases.denoise.dit_source`・`h3.phases.decode.vae_source` に `"file"`／`"resident"` を記録する（`h3` の中なので黄金キーは変わらない）。

### 8.2 block swap の包みの剥がしと入れ直し

- `release_installed()` は包みを外さない（参照を手放すだけ）。H3 側に `_unpatch_block_swap(dit)` を書く: 各ブロックについて、共有部品の印 `_BLOCK_SWAP_ATTR`（`engine.transformer.block_swap_service` から import するだけ。共有部品は変えない）があれば、その値（元の `forward`）を `block.forward` に戻して印を消す。テンソルには触らない（LTX 2.5 の `_unpatch_block_swap`〔`engine25/gguf_transformer.py:1149-1177`〕と同じ考え）。
- 呼ぶ場所は 2 か所: **ジョブの終わり**（保持する DiT を包みの無い状態で置く）と、**`_place_dit` の直前**（念のため。包みが残っていれば剥がしてから `install`）。これで、先読み on → off → on と変わっても二重に包まない（同期経路の `_patch_block` は今の `forward` を元として退避するので、剥がさずに入れ直すと二重になる）。
- `install()` はブロックを CPU に置いてから包むので、保持していた DiT（全部 CPU）にそのまま掛けられる。ブロック以外は `_place_dit` が GPU へ移す（今と同じ）。

### 8.3 ジョブごとの `attention_backend`・`block_swap_prefetch`

- `H3GenerateRequest.attention_backend`／`block_swap_prefetch` が `None` なら load の値を使う。
- ワーカーはジョブの `attention_backend` を `_resolve_attention` に通し（sage が無ければ sdpa に落として `"sage->sdpa"`）、実際の値をパイプラインへ渡し、`done.attention_used` に返す。
- `block_swap_prefetch` は `_place_dit` が `service.prefetch_requested` に書く（`install` の前）。`done.block_swap_prefetch_used` は今までどおり `install` の判定（`"on"`／`"off"`／`"on->off"`）。

### 8.4 `done`

- `keep_resident_used`: このジョブの終わりに保持したなら `"on"`、そうでなければ `"off"`。
