# MiniMax H3 対応 α版・第 2 弾の計画（I2V・アダプタ・操作パネル・常駐）

作成: 2026-10-17（敵対的レビュー反映済み。採否は §10）。台帳 §1-93。設計正本 `Docs/MINIMAX_H3_DESIGN.md`・土台 `Docs/MULTI_ENGINE_DESIGN.md`。第 1 弾の記録は `Docs/VERIFICATION_LOG.md` §165、調査報告の保管は `Docs/Outputs-archive/h3-stage1-reports-2026-10-17/`。本セッションの調査報告（R4a／R4b＝I2V の符号化、R5a／R5b＝受け皿の棚卸し）とレビュー（REVIEW_plan2）は一時領域 `scratchpad\h3\` と plans フォルダにあり、了承後に `Docs/Outputs-archive/h3-stage2-reports-2026-10-17/` へ写す。

## 1. 結論（何を作り、何で合格とするか）

**第 2 弾の到達点**: AviUtl2 の操作パネルから MiniMax H3 の **T2V と I2V（先頭フレームのみ）** が生成できる。ステップ数と shift（映像・音声）の 3 本のスライダーが操作パネルと Gradio にあり、MCP からも値を渡せる。常駐（`keep_resident`）が H3 でも効く。大きな画角でも映像の復号が VRAM からこぼれない。快適上限マーカーは H3 では出ない。**LTX 2.3／2.5 の振る舞いと記録（`metadata.json`）は変わらない。**

**合格の条件**: 段ごとの門（§6）に全部合格し、最後にオーナーが操作パネルで T2V と I2V を目視して合格と判断すること。合格後に dev → main を `--no-ff` で merge する（裁定 23）。

**変えないもの**: `engine/`・`engine25/` のエンジン本体、block swap と先読みの共有部品、LTX の生成結果（H3 の T2V の指紋 `0da6111c…` を回帰の基準にする）。

**了承時に確認したい 3 点**（レビュー m-15・m-16・m-17）: ① §1-99（T2V で符号化器を読まない）は VAE の構造の変更で自然に片づくので第 2 弾へ吸収する（裁定 25 の小さな変更）。② H3 では Edit のサブタブが全部閉じるため、既存の規則により **Edit タブが丸ごと灰色**になる（中のサブタブの配置は見えない）。裁定 30 の「配置だけ決めておく」をこの形で満たしたと解釈した。③ NAG は「非表示」でなく**灰色**にする（非表示だと有効にしたまま解除できず全ジョブが 422 になるため）。Single の A2V カードは連結要求なので `chain` で閉じる（灰色）。

## 2. 段の分け方と門（裁定 23）

| 段 | 範囲 | GPU を使う門 | 了承 |
|---|---|---|---|
| **2a** ワーカー | 復号 VRAM → TE の直接読み（§1-98 を吸収・裁定 32）→ I2V（視覚塔＋条件行）→ 常駐とジョブごとの attention／先読み。この順で進める | 2a-G2〜G5 | 本計画の了承で 2a の分をまとめて（基準取りの小さな GPU 使用を含む） |
| **2b** バックエンド | アダプタ・系統登録・記述子・事前検査・API の系統別制約と 3 項目・`GET /models` の制約の配信・テスト | 2b-G2（API 経由の実生成） | 2a の門が済んだ時点で改めて |
| **2c** 画面 | 操作パネル（追随・スナップ・マーカー非表示・機能スコープ・3 本のスライダー・常駐トグルの文言）・Gradio・MCP・aux2 の再ビルドと配置 | 2c-G2（オーナーの目視） | 2b の門が済んだ時点で改めて |

- コードは 3 段を**並行**して書く（触るファイルが重ならない。§4）。GPU を使う門だけが順番に進む。GPU は同時に 1 体だけ。
- 常駐の方式はオーナーの裁定 32（2026-10-17）: **§1-98（TE を層ごとにファイルから GPU へ直接読む）を第 2 弾に戻し、常駐を α版から有りにする**。理由: DiT と VAE（約 26GiB）を常駐させたまま現行方式で TE（約 29GiB）を読むと主メモリの山が約 60GiB になり、64GB 機でもページファイルに溢れる見込みだから。
- MULTI_ENGINE_DESIGN §8.1 が「3 例目が来るまで保留」とした 3 点（汎用 Runner への分割・mock の共通化・カテゴリ集合の可変化）は、本弾でも**引き続き保留**する（`H3Runner(LTXRunner)`・mock 共有・既存の 4 語のカテゴリで足りるため）。

## 3. 設計の決めごと（裁定ではなく設計。レビュー済み）

### 3.1 I2V（ワーカー側）

調査 R4a／R4b の結論に基づく。

| 項目 | 決めごと | 根拠 |
|---|---|---|
| 符号化の方式 | 分布の**平均**（裁定 28） | ComfyUI・DiffSynth・既存 LTX・`engine_h3` の現状がすべて平均。diffusers だけが種 42 のサンプル |
| 符号化器の型 | **f32**。復号器は bf16 のまま | 現状の bf16 は ComfyUI の fp16 より仮数が 3 ビット少ない。先頭画像 1 枚なので費用は小さく、VAE ファイルは fp16 なので f32 への変換で値は変わらない |
| VAE の構造 | `H3VideoVAE` を**符号化器（`encoder`＋`quant_conv`）と復号器（`post_quant_conv`＋`decoder`）に分けて別々に組んで読める**ようにする。符号化器は I2V のジョブごとに f32 で組んで捨てる（常駐しない）。T2V では組まない（§1-99 を吸収） | 今の `from_file` は全体を 1 つの型にそろえ、読み残しを例外にする（レビュー M-8） |
| 画素の正規化 | ImageNet 方式（現状どおり） | 3 実装で共通。−1〜1 では合わない |
| タイル分割 | 空間 256／重なり 64・常時（現状どおり） | 3 実装で共通。diffusers は「切ると出力が変わる」と明記 |
| キャンバスへの合わせ方 | **縦横比を保って覆うように拡縮し、中央で切り抜く**。補間は **PIL の LANCZOS**。H.264 の再圧縮は**しない**。**視覚塔にも同じ切り抜き後の画像を渡す** | 幾何は本製品の規約（LTX と同じ。利用者から見た規則を 1 つにする）。補間は H3 の 3 実装と同じ。再圧縮は LTX 固有の工程 |
| 1 フレームの扱い | 複製しない（現状どおり） | 3 実装で共通 |
| 条件行 | 申し送りの 4 点（0.999 のノイズ混ぜ・時刻 max(t, 0.999)・RoPE の t=L・毎ステップ入れ直し）。混ぜる乱数は**ジョブの種の生成器から、映像→音声の初期乱数を引いた後に f32・行の形で 1 回引き、全ステップで使い回す**（引き順を後にすることで T2V の指紋は変わらない） | 4 点は 3 実装で一致。乱数の作り方は実装ごとに違いビット一致は不可能なので、照合は符号化の出力で取る |
| 視覚塔 | `Qwen3VLModel`（文章＋視覚）を使い、プロンプトの先頭に画像のトークン列を置く。HF の `forward` はそのまま使う（DeepStack と 3 軸の位置符号化が層のループの中にあるため。M-7）。**書式・視覚塔に渡す画像のプロセッサの設定・トークン数の決まり方は、担当が DiffSynth と diffusers から写して INTERFACE2 に書き、2a-G3 でトークン列ごと DiffSynth と一致させる** | R2 の「`<Picture 1>: <|vision_start|><|image_pad|>×N<|vision_end|>`」を出発点にする。第 1 弾は `Qwen3VLTextModel` で視覚塔を読んでいない |
| 強度（strength） | H3 には無い概念。**アダプタがワーカーへ送らない**（ワーカーの `images` は `{path, frame_idx}` だけ）。既定 0.8 以外はアダプタが拒否（§3.4） | 毎ジョブ警告が出る作りを避ける（m-7） |
| 画像の受け口 | `images: [{path, frame_idx}]`。α版は **0 個か 1 個・`frame_idx == 0` だけ**を受け、それ以外は理由つきの `ValueError` | アダプタからワーカーへの形を LTX と揃える（R4b） |

### 3.2 TE の直接読み・常駐・復号 VRAM・ジョブごとの設定（ワーカー側）

- **TE の直接読み（§1-98 の吸収）**: テキストエンコーダは 1 ジョブで 1 回しか前向き計算をしない（プロンプト 1 本。CFG も無い）ので、層を主メモリに置く意味がない。**HF の `forward` はそのまま使い、各層の前向きの直前にファイルから GPU へ読み、直後に捨てるフック**（`GemmaLayerOffloadService` の差し替え）にする。埋め込み表（約 1.5GB）と視覚塔は計算の前に GPU へ読み、計算後に捨てる。**`te_layers_on_gpu` は意味を失うので load とセルフテストの引数から外す**（引き算）。計算は変えないので、2a-G2 で第 1 弾の方式と**ビット一致**を要求する。期待する効果: 主メモリの山が DiT ぶん（約 20GiB）＋VAE へ下がる。所要時間は同程度（推測）。
- **常駐（`keep_resident`）**: LTX と同じ契約＝**キーが真のときだけ保持し、キーが無いジョブで解放する**（R5a A1）。保持するのは **DiT（ブロックは CPU の正本・ブロック以外も CPU へ戻す）・映像 VAE の復号器・音声 VAE（CPU）**。符号化器は保持しない（§3.1）。**ジョブの終わりに全ブロックとブロック以外を CPU へ戻し、GPU には何も残さない**（同期経路は最後の窓のブロックを GPU に残すので明示的に戻す。M-1）。block swap の包みは `release_installed()` では外れない（参照を手放すだけ）ので、**LTX 2.5 の `_unpatch_block_swap` と同じ「印から元の `forward` を戻す」剥がしを H3 側に書き**（共有部品は変えない）、次のジョブで入れ直す。`done` の `keep_resident_used`（`"on"`／`"off"`）は既存の `GenerationOutcome.keep_resident_used` 欄で運ぶ。主メモリの見込み: DiT 約 19.5GiB＋VAE 約 6GiB＋ページ固定プール（2×369MB）＋基礎で**作業セット約 30GiB・コミット量はそれより大きい可能性**（第 1 弾ではコミット量が作業セットより約 11GiB 多かった）。2a-G4 で実測し、64GB 機でページファイルに溢れない水準かを監督が判断する。
- **復号 VRAM**: `H3VideoVAE.decode` の戻り値を **CPU 上の uint8 `[F, H, W, 3]`** に変える。`_decode_temporal` で窓ごとに確定した片（`blend` の後・`parts.append` の時点）を、その場で正規化を戻し→**clamp → ×255 → round → uint8**（今の `frames_to_uint8` と同じ順。T2V の指紋を保つため）→CPU へ移す。`torch.cat` は CPU で行う。最後の重なりと `pad_tokens` の末尾の切り詰めも同じ契約に乗せる。`frames_to_uint8` は役目を失うので消す。`write_mp4` は元から CPU の uint8 を受けるので変えない。T'=1 の特例も同じ契約。音声の復号は窓なしのまま（§1-95 で測る）。
- **ジョブごとの設定**: `attention_backend` と `block_swap_prefetch` を **`generate` でもジョブごとに受ける**（load の値は既定値になる）。要求ではジョブごとの項目なので揃える（R5a D1）。先読みの on/off が常駐中に切り替わっても、上の「印から剥がして入れ直す」で包みが二重にならない。
- **`ready` は変えない**。`GET /status` の `block_swap_prefetch_available` はアプリ側が `blocks_on_gpu` の式から作る（M-11）。H3 のアダプタは 2.5 と同じく `DEFAULT_BLOCKS_ON_GPU = 8` を持つ。

#### ワーカーのメッセージ（設計正本 §8 からの差分。C1 と C3 の境界）

| メッセージ | 追加 | 削除 |
|---|---|---|
| `load` | — | `te_layers_on_gpu` |
| `generate` | `images: [{path, frame_idx}]`（0 か 1 個）・`keep_resident: bool`（無ければ解放）・`attention_backend`・`block_swap_prefetch`（無ければ load の値） | — |
| `done` | `keep_resident_used`（`"on"`／`"off"`）・`images_used`（個数）・`h3.phases` に符号化の秒数 | — |
| `ready` | — | — |

### 3.3 系統別の制約（API・記述子・`GET /models`）— 本弾の本丸

MULTI_ENGINE_DESIGN §7 は「エンジン別の UI パラメータ制約はまだ作らない【オーナー裁定】」とし、第 3 段階（UI 調整）を未着手としている。設計正本 §11 と裁定 23 により本弾で着手する。本計画の設計は次のとおり: **系統別の数値の規約は、系統が分かった時点で、その系統のアダプタが宣言した表で検査する。**

- **`GenerateRequest`（`api/models.py`）の検証は「どの系統でも成り立つ形の検査」だけにする**。`validate_ltx_constraints` のうち **64 の倍数（646-649）・8n+1（650-651）・distilled のステップ数 8 と guidance 1.0（657-665）の 3 種だけ**を LTX 系統の規則表へ移す。`crop_output`・キーフレームのスナップ・IC-LoRA／NAG／outpaint／inpaint／alpha_gen の検査、`num_frames` 9〜481・`frame_rate` 1〜60・寸法の下限の `Field` は**残す**。連結・Alpha Gen・inpaint など LTX しか使わない要求モデルの検証は**触らない**。`GenerateRequest` を作る他の経路（`api/generate_alpha.py:247` の内部要求・`GenerateChainRequest.to_clip_request`）は LTX 専用で系統の検査を通さない（実害なし。記録に残す）。
- **ステップ数と shift の項目**: `num_inference_steps` は**既存の項目**を使い、既定を `8` から **`None`（＝系統の既定値）** に変える。`shift_video: float | None = None`・`shift_audio: float | None = None` を足す。pydantic の範囲はステップ `ge=1, le=100`（`le` は残す。LTX 2.3 の two_stage_hq に 100 超が通らないように）・shift は正。**`None` の解決は 1 か所**: `validate_for_family` が系統の既定値（LTX は 8、H3 は 20／12／3）を**要求に書き戻す**。以後の読み手（mock の `max(1, steps)`・ジョブ開始ログの `%d`・2.3 の `num_steps`・2.5 の無視ログ・`metadata.json`・mp4 のレシピ）は常に整数を見るので、**LTX の記録は従来どおり `8`**（C-1）。`GET /config` の `defaults.num_inference_steps` は 8 のまま（系統共通の値であり、画面は使っていない）。
- **系統の規則表**: 各アダプタが `REQUEST_RULES`（仮称）を宣言する。LTX 2.3／2.5 は共通の 1 つ（`services/engines/ltx/request_rules.py`。8n+1・64 の倍数・distilled のときステップは 8・guidance 1.0・既定のステップ 8）。H3 は **17n+5 かつ 22 以上・32 の倍数・fps 24 固定・ステップ 1〜50（既定 20）・shift 映像 1〜20（既定 12）・shift 音声 1〜20（既定 3）**。キーフレームの枚数と強度は規則表でなく拒否表（§3.4。1 か所）。`frame_rate` の上下限と `num_frames` の上限は pydantic の `Field` と重なるので、**定数を 1 か所（規則表）に置き、`Field` はそれを参照する**。
- **検査の場所と 422 の形**: `api/generate.py` の冒頭、`engines.reject_unsupported(...)` の**後**に `engines.validate_for_family(active_engine_family, request)` を置く。違反は **422・符号 `VALIDATION_ERROR`・`message: "Request validation failed"`・`detail: [{"loc": ["body"], "msg": "Value error, <今の英文>", "type": "value_error"}]`** と一字一句決める（pydantic 由来の形を写す。既存のテストは本文の部分一致だけを見ている）。順序の変化（LTX 2.5 に `two_stage_hq` と 8n+1 違反を同時に送ると `FEATURE_UNSUPPORTED` が先に返る）は利用者に見える差として記録に残す。
- **`GET /models` の `base_models[]` に `constraints` を足す**。中身は規則表から機械的に作る（**正本は 1 か所＝アダプタの表**。操作パネル・Gradio・MCP・モックはこれを読む）。`frame_rate` は 1 つの形 `{min, max}` に統一し、画面は「min と max が同じなら欄を無効」と読む:

  ```json
  "constraints": {
    "num_frames": {"step": 17, "offset": 5, "min": 22, "max": 481},
    "dimension_multiple": 32,
    "frame_rate": {"min": 24, "max": 24},
    "sampling": {
      "num_inference_steps": {"min": 1, "max": 50, "step": 1,   "default": 20},
      "shift_video":         {"min": 1, "max": 20, "step": 0.1, "default": 12},
      "shift_audio":         {"min": 1, "max": 20, "step": 0.1, "default": 3}
    }
  }
  ```

  LTX は `{"num_frames": {"step": 8, "offset": 1, "min": 9, "max": 481}, "dimension_multiple": 64, "frame_rate": {"min": 1, "max": 60}, "sampling": null}`。`sampling` が `null` なら画面はスライダーを出さない（LTX の見た目は変わらない）。
- **LTX にとっての新項目の扱い**（4 分類表）: `shift_video`／`shift_audio` は 2.3 が「無視（宣言のみ）」、2.5 が「無視＋ログ」（それぞれの既存の作法。概念が無い項目＝`guidance_scale` の前例）。`num_inference_steps` は 2.3 が動作、2.5 が無視（現状どおり）。
- **`metadata.json`**: `GenerationOutcome` に `h3` 欄を足して `metadata["h3"]` へ（2.5 の `ltx25` 欄の前例）。中身はワーカーの `done` の `h3`（`phases`・`quant_backends`）と、最上位にある H3 固有の値（shift・コミット量）をアダプタがまとめたもの。ステップ数と shift は `request` にも残る（書き戻しのため）。

### 3.4 アダプタ・系統登録・事前検査・記述子（バックエンド側）

- **系統登録**: `services/engines/__init__.py` の `FAMILY_BY_ID` に `"h3"`、`FAMILY_DISPLAY` に `"MiniMax H3"`。`FAMILY_BY_KV` には**載せない**（H3 は safetensors の指紋で判定し GGUF を持たない）。`check_kv` は h3 では素通し。`config.py` に `engine_python_h3`（`engine_python_ltx25` と同じ形）。記述子・登録・アダプタは**同時に入れる**（記述子だけ先だと `GET /models` が落ちる）。
- **記述子 `scripts/manifests/40-minimaxh3.json`**（schema 2）: `id: "MiniMaxH3"`・`display_name: "MiniMax H3"`・`engine_family: "h3"`・カテゴリは既存の 4 語 `transformer`／`text_encoder`／`video_vae`／`audio`（操作パネルの型が 4 語で閉じているため。R5b E1）・`assets: {"tokenizer_dir": "MiniMaxH3/TextEncoder/qwen3vl_tokenizer"}`（LTX 2.3 の `gemma_root` と同じフォルダ方式。`downloads[].files[].path` を同じ値にそろえる）・`comfort` は**持たせない**・`downloads` は Comfy-Org/MiniMax-H3 の 4 本（`diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`・`text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors`・`vae/minimax_h3_video_vae_fp16.safetensors`・`vae/minimax_h3_audio_vae_fp32.safetensors`）と MiniMaxAI/MiniMax-H3 `FL2VA/text_encoder/` の tokenizer 6 ファイル（`"kind": "dir"`）。配置は第 1 弾で手動配置済みの `models/MiniMaxH3/{Weights,TextEncoder(+qwen3vl_tokenizer/),VAE}` に揃える。`downloads` の実行は第 3 弾のインストーラで検証する。
- **アダプタ `services/engines/h3/adapter.py`**: `H3Runner(LTXRunner)`＋`_RealBackendH3(_RealBackend)`。`SELECTION_FIELDS`・`REQUIRED_ASSETS = ("tokenizer_dir",)`・`check_kv`・4 分類表・`REQUEST_RULES`・`UNSUPPORTED_FEATURES`・`DEFAULT_BLOCKS_ON_GPU = 8`・`_build_load_payload`（設計正本 §8 のキー名から `te_layers_on_gpu` を除く）・`generate`（`_read_worker_events(one_stage=True)`、`done` → `GenerationOutcome(h3=..., keep_resident_used=...)`、**`crop_output` は LTX と同じアプリ側の切り抜きを呼ぶ**〔M-5〕）。mock は 2.3 の `_MockBackend` を共有し `_MOCK_BACKEND_LABEL` だけ変える（音声なしで足りる）。
- **4 分類表の方針**（全項目は C3 が `GenerateRequest` のスキーマから埋め、監査テストで完全性を固定する）:
  - 動作: `prompt`・`seed`・`width`・`height`・`num_frames`・`frame_rate`・`num_inference_steps`・`shift_video`・`shift_audio`・`conditioning_images`（先頭 1 枚）・`keep_resident`・`attention_backend`・`block_swap_prefetch`・`crop_output`・`embed_mp4_metadata`。
  - 拒否（`(項目, 機能名, 述語)`。述語は「既定値と違うか」が基本）: `loras → loras`・`reference_video_id → reference_video`・`nag_enabled → nag`・`pipeline → two_stage_hq`（2.5 と同じ名前）・`guidance_scale → guidance_scale`（1.0 以外。H3 に CFG は無い）・`outpaint → outpaint`・`inpaint → inpaint`・`alpha_gen → alpha_gen`・`prune_vaed → prune_vaed`・`keep_resident_embeddings → keep_resident_embeddings`・`conditioning_images → multi_keyframes`（2 枚以上または `frame_idx != 0`）・`conditioning_images → keyframe_strength`（強度が既定 0.8 以外）。**`conditioning_images` は「動作」でありつつ拒否の行を 2 つ持つ**（H3 の監査テストはこの重なりを明示的に許す）。拒否の文言は日本語で「MiniMax H3 は {機能} に対応していません」の形。`multi_keyframes` は「MiniMax H3 は先頭フレーム 1 枚だけに対応しています」。
  - 無視: `fused_gguf_dequant_kernel` など概念の無い項目（2.5 と同じく既定と違うときだけログ）。
  - 従属（`GOVERNED_FIELDS`）: `negative_prompt`・`neg_method`・`vsf_scale`・`nag_*` → `nag_enabled`、`conditioning_attention_strength`・`reference_video_strength` → `loras`。
  - **`UNSUPPORTED_FEATURES` の導出規則**: 単発の拒否表の機能名 ＋ `chain`（連結は `reject_chain` で丸ごと拒否）。`v2v`・`a2v`・`end_source`・`ic_lora`・`vsf`・`negative_prompt`・`retake` は公開しない（単発の要求に対応する項目が無い。M-4）。既存 2 系統の「もう一方の LTX を選べ」の文言は**触らない**。
- **事前検査** `services/model_registry.py`: `precheck_model_file` に系統（記述子）を渡し、h3 なら transformer → `inspect_h3_dit`、text_encoder → `inspect_h3_te`、VAE 2 本 → 既存の汎用検査（長さと JSON）。**`H3FormatError`（`ValueError`）を `MODEL_INCOMPATIBLE` に変換する**（今は捕まえず 500 になる。R5a C1）。`services/weight_class.py` は LTX 以外の系統では静かに `None` を返す。
- **進捗**: 単発の割合計算は `decode` 段で動かない（0.90 で止まって見える）。H3 の復号は全体の 1 割弱なので**このまま**。

### 3.5 操作パネル（webui）

場所は `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/webui/src/`（以下 `src/`）。方針は「**系統 id ではなく、サーバーが配る `constraints` と機能名で読む**」（`featureScope.ts` の既存の規則のまま）。

- **制約の配線**: `useBaseModels.ts` が `constraints` を写し、`AppShell.tsx` が `engineFamily` と同じ経路で Single へ渡す。無いときは LTX の値（現状の定数）を既定にして、Chained・Edit・バッチは**触らない**（H3 では閉じる）。
- **フレーム規約と解像度と fps**（Single と右クリックの Single 行き）: `paramUtils.ts` の `snapNumFrames`／`isValidNumFrames`／`isValidDimension` に格子（`step`・`offset`・`min`）と倍数を引数で渡せるようにし、省略時は LTX の値（呼び出し元を変えずに済む）。`DurationField` の `step`・下限、`SizeSlider` と数値入力の `step`、`GENERAL_MULTIPLE`、プリセットのスナップ、右クリックの事前入力（`AppShell.tsx` #9／#10・`SingleScreen.tsx`・`prefillSeed.ts`・`deriveDuration.ts`・`deriveGenerationParams.ts`）が `constraints` を読む。**fps は `frame_rate.min == max` なら欄を無効にしてその値を表示し、要求もその値**（事前入力の fps 方針より優先）。aux2 の C++ は変えない（本番経路に 8n+1 の前提が無い。R5b D2）。
- **3 本のスライダー**: `ReservedFields`（CFG のモック。`CommonGenerationFields.tsx:544`）を、`constraints.sampling` があるときだけ **`SamplingFields`** に差し替える（無いときは今のまま＝LTX の見た目は不変）。1 行＝項目名＋`<input type="range">`＋数値入力。既存の `FloatSliderField` に横並びの引数を足して 3 本とも賄う（ステップ数は `step=1` で整数化）。状態は `useGenerationForm` の `values`（既定値は `constraints.sampling.*.default`、ベースモデルが変わったら既定へ戻す）。`toGenerateRequest` は `sampling` があるときだけ 3 項目を**常に**送る（規則を 1 つにする）。Chained の `ReservedFields` は触らない。i18n: 「ステップ数」「Shift（映像）」「Shift（音声）」。
- **機能スコープ（裁定 30）**: `featureScope.ts` の `FEATURE_UI` に行を足し、各部品が閉じ方を決める（既存の作法）。**閉じた機能の項目は要求に載せない**（NAG は `nagRequestFields` を機能の可否で門にする。M-3）。

  | 機能名 | `UiTarget` | 閉じ方 | 部品 |
  |---|---|---|---|
  | `reference_video` | `createSection.referenceVideo`（新設） | **非表示**（IC-LoRA は LTX 固有） | `GenerationForm.tsx` の参照動画のアコーディオン |
  | `chain` | `mode.chained`（既存）＋ `createSection.a2v`（新設） | タブは灰色（既存）。**Single の A2V カードは灰色**（連結要求そのもの） | `ModeTabs.tsx`・`GenerationForm.tsx` の A2V カード |
  | `nag` | `shell.nag`（新設） | **灰色**＋要求に NAG の項目を載せない | `NagAccordion` |
  | `multi_keyframes` | `createSection.keyframesMulti`（新設） | **灰色**（`useKeyframes` の `maxItemsOverride: 1`。2 枚目以降の追加ボタンを無効） | `KeyframesPanel` |
  | `keyframe_strength` | `createSection.keyframeStrength`（新設） | **灰色** | キーフレームカードの強度 |
  | `retake`・`outpaint`・`inpaint`・`alpha_gen`・`prune_vaed`・`keep_resident_embeddings` | 既存 | 既存どおり | — |

  Edit の 4 サブタブが全部閉じるので、既存の `CONTAINER_TARGETS` により **H3 では Edit タブが丸ごと灰色**になる（Alpha Gen の閉じ方を変える必要はない。M-2）。Style LoRA は画面の部品ではなく文字列なので画面は変えず、送信時の 422 に任せる（MULTI_ENGINE §2.5）。`maxItemsOverride: 1` は追加を止めるだけで、LTX で 2 枚置いてから H3 へ切り替えた場合は送信時の 422（「先頭フレーム 1 枚だけ」の文言）で解決する（m-18）。右クリックの Single 行き（IC-LoRA・A2V・キーフレーム追加）は、画面側で閉じた部品に落ちるので今回は止めない。
- **快適上限マーカー**: 落ち先を外す条件は「**`limits.comfort_budgets` にその系統の表が無い**」に限る（表はあるが行が合わない LTX の場合は今までどおりスカラ値へ落ちる。M-6）。触るのは Single（`useGenerationForm.ts`）と右クリックの事前入力 3 か所。Chained・Retake は H3 では閉じるので触らない。マーカー無し＝`DurationField` の目盛りと警告が出ない・右クリックがフレーム数を快適上限で書き換えない（格子へのスナップだけ行う）。Settings の快適上限表と Outpainting は今のままで出ない。
- **Settings の常駐トグル**: 既存の「モデル骨格の常駐」を H3 でもそのまま使う（裁定 26）。説明文に「MiniMax H3 では DiT と VAE を保持します」の一文を足す（数値は書かない。正本は記録簿）。「先読みが有効なときだけ押せる」連動は**そのまま**。
- **モック `mockBridge.ts`**: `MOCK_BASE_MODELS` に `MiniMaxH3`（`constraints` つき）、`MOCK_UNSUPPORTED_FEATURES` に上の機能名、`isDefaultFilePresent` を系統ごとに分ける、`handleGenerate` の検証を `constraints` で行う。`comfort_budgets` には足さない（マーカー無しをモックでも再現）。
- **文書の整合**（C4 が担当）: `src/api/types.ts` の「`num_inference_steps` を UI に出さない」の注記と `FE/Docs/API_REFERENCE.md` §7・`FE/Mock/AVIUTL2_DESIGN_BRIEF.md` §5 を「LTX では出さない（distilled で固定）。MiniMax H3 ではステップ数と shift を出す（裁定 22）」に改める。DEVLOG §141。
- **ビルドと配置**: `npm run typecheck`・`npx vitest run --exclude "**/backend.integration.test.ts"`・`oxlint` → `build.ps1`（webui を aux2 に埋め込む）→ `deploy.ps1`（実機の Plugin フォルダと配布用コピーの 2 か所）。

### 3.6 Gradio と MCP

- **Gradio**: `GET /models` の `constraints` を読む。Single の飾りのステップ数／CFG スライダー（`ui.py:547-550`）を、選択中の系統に `sampling` があるときだけ**ステップ数／Shift（映像）／Shift（音声）の 3 本の操作できるスライダー**にし、送信に含める。無いときは今のまま（固定値を送る）。事前検査（`handlers.py:747-765`）とフレーム数のラベル（`i18n.py` の「(8n+1)」）を `constraints` から作る。Single の溢れ表（`/config` の `spill_free_frames` から作る LTX の表）は、系統に快適上限の表が無ければ出さない。Chain タブ（`876-880`）と `comfort.py` は触らない（m-20）。
- **MCP（最小対応）**: `submit_generate` に `num_inference_steps`・`shift_video`・`shift_audio` を任意引数で足す（`None` なら送らない＝系統の既定値）。`list_models` の返しに `constraints` を含め、説明文の「8n+1」を「ベースモデルの `constraints` に従う（LTX は 8n+1、MiniMax H3 は 17n+5）」に改める。`num_frames` の既定 49 は LTX の値のまま（H3 で省略すると 422 になることを説明文に書く）。`tests/test_mcp_tools_generate.py` の「ステップ数を出さない」固定は書き換える。

### 3.7 文書（了承後と完了時。文書係が担当）

- 了承後すぐ: 研究ノート §10 に裁定 32（常駐の方式）と、了承時の確認 3 点の結論を追記。台帳 §1-93 に第 2 弾の段分けと状態を書き、**§1-98 と §1-99 は第 2 弾へ吸収して削除**。
- 完了時: 設計正本 `MINIMAX_H3_DESIGN.md`（§2 に裁定 23〜32 の設計分を移管・§3 に I2V の系列と条件行・§4 に常駐と TE の直接読み・§8 にキー名の差分と `h3_format` の事前検査と VAE の分割・§9 に既定値と範囲・§11 に第 2 弾の門の結果・§12／§13 の更新）、`MULTI_ENGINE_DESIGN.md` §5.6（`constraints` と機能名の導出規則）・§7（第 3 段階に着手した旨と規則表の仕組み）・§8.1（3 例目の判断は引き続き保留）、`Videomni_Backend_Specification.md`（API の 3 項目・`constraints`・422 の順序）、`VERIFICATION_LOG.md` §166、台帳。操作パネル側の文書は C4（§3.5）。調査報告 R4a／R4b／R5a／R5b と本計画とレビューを `Docs/Outputs-archive/h3-stage2-reports-2026-10-17/` に保管。

## 4. 作業項目とファイル（担当ごと・1 ファイル 1 所有者）

### C1（Opus）— 2a ワーカー（`engine_h3/`・`tests/test_h3_engine.py`・`tests/test_h3_media.py`）。順序は固定

0. **基準取り（GPU・小）**: 改修前のコードで、同じプロンプトの TE の出力（T2V）を保存する（2a-G2 の正解側）。
1. **復号 VRAM**: `video_vae.py`（`decode` → CPU の uint8・窓ごとの変換・丸めの順）・`pipeline_h3.py`（`frames_to_uint8` を消す）。
2. **TE の直接読み**: `text_encoder.py`（HF の `forward` のまま層ごとにファイルから GPU へ読むフック。`GemmaLayerOffloadService` を外す）・`sft_load.py`（層単位の読み出し）・`worker.py`（`te_layers_on_gpu` を外す）。→ 2a-G2。
3. **I2V**: `engine_h3/image_cond.py`（新設。PIL で開く→覆う拡縮＋中央切り抜き・LANCZOS→f32 [0,1] `[1,3,1,H,W]`）・`video_vae.py`（符号化器と復号器の分割・符号化器の f32）・`text_encoder.py`（`Qwen3VLModel`。視覚塔とプロセッサ。画像があるときだけ組む）・`pipeline_h3.py`（条件行の組み立て・時刻・RoPE・乱数の引き順・毎ステップ入れ直し）・`worker.py`（`images` の受け口・`_GENERATE_FIELDS`・セルフテストの `--image`）・`constants.py`。実装の前に INTERFACE2（参照実装から写す事項）を書き、監督が読む。→ 2a-G3・2a-G4 の I2V。
4. **常駐とジョブごとの設定**: `pipeline_h3.py`（保持と解放・CPU への戻し・印による剥がしと入れ直し・符号化器は保持しない）・`worker.py`（`keep_resident`・`attention_backend`・`block_swap_prefetch` を `generate` で受ける）。→ 2a-G4 の常駐・2a-G5。
5. テスト: 黄金キーの更新・復号の契約（uint8）・画像の合わせ方・符号化の形と f32・常駐の状態遷移と剥がし（小さな重みで CPU）・直接読みの読み出し。
6. 門の台本: `outputs/h3-gate2/`（git 管理外）と `Docs/Outputs-archive/h3-gate2/`（台本と JSON）。

### C3（Opus）— 2b バックエンド（`api/`・`services/`・`scripts/manifests/`・`config.py`・`config.yaml`・`tests/`〔下の名指し分〕）

1. `api/models.py`（3 項目と既定の変更・LTX 固有の 3 種の検査の移動・`Field` の定数参照）・`services/engines/ltx/request_rules.py`（新設）・`services/engines/__init__.py`（登録と `validate_for_family`。既定値の書き戻し）・`api/generate.py`（呼び出し。`reject_unsupported` の後）・`api/models_registry.py`（`constraints`）・`api/errors.py`（必要なら）。
2. `services/engines/h3/adapter.py`（新設）・`services/engines/ltx/adapter.py` と `ltx25/adapter.py`（4 分類表に 2 項目・`GenerationOutcome` の `h3` 欄）・`services/pipeline_manager.py`（`metadata["h3"]`）・`services/model_registry.py`（事前検査の分岐と例外の変換）・`services/weight_class.py`・`config.py`・`config.yaml`（ローカル。`engine_python_h3`）。
3. `scripts/manifests/40-minimaxh3.json`。
4. テスト（所有）: `tests/test_engine_dispatch.py`・`test_comfort_budgets.py`（系統集合 ⊇ comfort 集合に緩める）・`test_base_model_contract.py`・`test_ltx25_adapter.py`・`test_ltx_adapter_audit.py`・`test_validation.py`・`test_model_swap_load.py`・**`tests/test_engine_h3_adapter.py`（新設。名前は C1 の `test_h3_*` と区別）**・`conftest.py`（3 系統の HTTP クライアント。H3 の DiT と TE のヘッダの偽物は `tests/test_h3_format.py` の作り方を使う）・`num_inference_steps: 8` を本体に入れている既存テスト群。`test_gradio_*`・`test_mcp_*` は触らず、API の変更で落ちるものの一覧を C5 に渡す。

### C4（Opus）— 2c 操作パネル（`AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/webui/` と同 `Docs/`・`Mock/`）

`src/api/types.ts`・`shell/useBaseModels.ts`・`shell/AppShell.tsx`・`shell/featureScope.ts`・`shell/comfortTable.ts` と Single 側の呼び出し・`shell/SettingsPanel.tsx`・`shell/NagAccordion.tsx`・`shell/nagSettings.ts`・`modes/single/{paramUtils,useGenerationForm,CommonGenerationFields,GenerationForm,SingleScreen,FloatSliderField,KeyframesPanel,KeyframeCard,useKeyframes,defaultConfig}.*`・`timeline/{deriveDuration,deriveGenerationParams,prefillSeed}.ts`・`bridge/mockBridge.ts`・`i18n/strings.ts`・対応するテスト・`Docs/API_REFERENCE.md` §7・`Mock/AVIUTL2_DESIGN_BRIEF.md` §5・DEVLOG §141。ビルドと配置は監督が `build.ps1`／`deploy.ps1` で行う（配布用コピーのコミットはオーナー）。

### C5（Opus）— Gradio と MCP（`gradio_ui/`・`mcp_server/`・`tests/test_gradio_*`・`tests/test_mcp_*`）

§3.6 のとおり。`constraints` の形と H3 の機能名は本計画で固定済みなので C3 と並行で書き始め、結合は C3 の 1 が済んでから。

### 文書係（Sonnet）— §3.7（バックエンド側の文書のみ）

## 5. テスト

- **ワーカー（C1）**: `.\.venv-engine-h3\Scripts\python.exe scripts\run_h3_pytest.py tests\test_h3_engine.py tests\test_h3_media.py`（GPU のものは `H3_GPU_TESTS=1`）。`tests\test_h3_format.py` はアプリ側の venv。
- **アプリ（C3・C5）**: `.\.venv\Scripts\python.exe -m pytest tests`（全件。mock が強制される）。
- **操作パネル（C4）**: `npm run typecheck`（`npx tsc --noEmit` は偽の合格）・`npx vitest run --exclude "**/backend.integration.test.ts"`（空いた機械で）・`oxlint`。
- 回帰の基準: H3 の T2V の指紋 `0da6111c…`（修正後・sdpa・seed 42）。LTX は既存のテスト群。

## 6. 門

| 門 | 内容 | GPU |
|---|---|---|
| **2a-G1** | ワーカーのテスト全件（新設分を含む）が通る。`engine_h3` 配下に `inference_mode` が無い | 不要 |
| **2a-G2** | **TE の直接読みの照合**: 改修前に保存した TE の出力（C1 の 0）と、直接読みの出力が**ビット一致**（T2V）。主メモリの山（コミット量・作業セット）を記録。I2V の TE（視覚塔込み）は 2a-G3 で DiffSynth と比べる | 使う |
| **2a-G3** | **I2V の部品の照合**（DiffSynth の写しを正解側に。視覚塔は同じ int8 ファイルを DiffSynth の経路で読ませる。bf16 の取得は不要）: 符号化は bf16 どうしでビット一致を期待し、一致しなければ差の最大値を記録して G4 の水準（1e-2 台）を上限にする／f32 との差を記録／視覚塔の出力は G4-TE と同じ水準／条件行の並び・時刻の番号・RoPE の座標は float64 で完全一致／トークン列（画像の書式込み。両側に同じ切り抜き後の画像）が一致 | 使う（小さい） |
| **2a-G4** | **I2V 1 本**: 640×384・124f・24fps・20 ステップ・seed 42・試験画像 1 枚。映像と音声の入った mp4。先頭フレームと入力画像（切り抜き後）の PSNR を記録（25dB 以上を目安。合否は監督の目視と合わせて判断）。**T2V の回帰**: 常駐なしで指紋 `0da6111c…` と一致。**常駐の照合**: 1 プロセスで **T2V（常駐 on・先読み on）→ I2V（常駐 on・先読み off）→ T2V（常駐 on・先読み on）→ T2V（キー無し）** を続け、各ジョブの後に `torch.cuda.memory_allocated()`・`memory_reserved()`・コミット量・作業セットを記録。合否: 指紋が常駐なしの同条件と一致／2 本目以降で DiT と VAE の読み込みが飛ぶ／**合間の allocated が 0**／キー無しの後にコミット量が保持前の水準へ戻る（戻らなければ記録して監督判断）／山（コミット量）は記録し、64GB 機でページファイルに溢れない水準かを監督が判断 | 使う |
| **2a-G5** | **映像の復号 VRAM**: 1344×768・345f（潜在 `[1, 24, 102, 48, 84]`）の乱数の潜在を復号だけ行い、VRAM の予約ピークが **12.8GB（16GB の 8 割。AviUtl2 のプレビューと同居する余裕）未満**で、出力が CPU の uint8 であること | 使う |
| **2b-G1** | アプリのテスト全件が通る。`GET /models` に H3 と `constraints` が出る。mock で T2V と I2V が HTTP 経由で通る。LTX 2.3／2.5 の 422 の形と文言が §3.3 のとおりで、`metadata.json` の `num_inference_steps` が LTX で `8` のまま | 不要 |
| **2b-G2** | **API 経由の実生成**: H3 を読み込み（事前検査が実ファイルで通る）、`POST /generate` で T2V と I2V を 1 本ずつ（MCP の自前クライアントか curl）。`metadata.json` に `h3` が入り、ステップ 20／12／3 が `done` にこだまする。常駐 on で 2 本目が速い | 使う |
| **2c-G1** | typecheck・vitest・oxlint・Gradio と MCP のテストが通る。モックで H3 を選ぶとスライダーが出て、Chained と Edit が灰色、参照動画が消え、A2V と NAG とキーフレームの追加と強度が灰色、マーカーが出ず、フレーム数が 17n+5 にスナップし fps が 24 固定になる | 不要 |
| **2c-G2** | `build.ps1` → `deploy.ps1` → AviUtl2 を起動して、**オーナーが操作パネルから T2V と I2V を生成して目視**（スライダー・常駐・グレーアウト・マーカー非表示を含む） | 使う |

## 7. 担当と進め方（監督は監督専任）

```text
了承 → 文書係（台帳・裁定 32・確認 3 点）
     → C1（2a）・C3（2b）・C4（2c）・C5（Gradio と MCP）を並行で起動
     → C1: 基準取り（GPU・小）→ 復号 VRAM → 直接読み → 2a-G1 → 2a-G2
            → INTERFACE2（視覚塔の書式など）→ 監督が確認 → I2V → 常駐 → 2a-G1 → 2a-G3 → G4 → G5
     → C3: 実装 → 2b-G1 → コードレビュー（Opus）→ 修正
     → C4: 実装 → typecheck と vitest → コードレビュー（Opus）→ 修正
     → C5: 実装 → テスト → コードレビュー（C3 と合わせて）
     → 2a の門が済んだら: 2b の GPU の了承 → 2b-G2
     → 2b が済んだら: 2c の GPU の了承 → build/deploy → 2c-G2（オーナーの目視）
     → コードレビュー（C1 の分。Opus）→ 修正 → 門の取り直し（指紋が変わったら切り分け）
     → 文書係（完了時の文書）→ コミット（dev・監督）→ オーナーの合図で main merge
```

- 了承の単位: 本計画の了承＝2a の GPU の了承（基準取りを含む）。2b・2c の GPU はそれぞれの段に入るときに改めて取る（裁定 23）。GPU は同時に 1 体だけ。
- 各担当には毎回「設計に疑問を抱いたら、根拠（実測値・エラー全文）を添えて止まってエスカレーションする権限」と「勝手な設計変更は禁止」を対で明記する。担当外のファイルは触らない。
- コミットの分け方（dev・監督が行う）: (1) 2a ワーカー (2) 2b バックエンド (3) 2c 操作パネル (4) Gradio と MCP (5) 文書。配布用の aux2 のコピーはオーナーが手動でコミットする。

## 8. 間違えやすい点（担当への申し送り）

- **視覚塔の入力は推測で書かない**。書式・プロセッサの設定（`preprocessor_config.json` は tokenizer のフォルダにある）・トークン数は DiffSynth と diffusers の実物から写し、2a-G3 でトークン列ごと一致させる。視覚塔のテンソルが int8 か bf16 かはヘッダで確かめてから組む。DiffSynth はキーフレームを引き伸ばしてからプロセッサに渡すが、本製品は切り抜き後の画像を VAE と視覚塔の両方に渡す。
- TE の直接読みは HF の `forward` を残す（DeepStack と 3 軸の位置符号化が層のループの中にある）。層のループを自前で書き直さない。
- 条件行の乱数は、映像→音声の初期乱数を引いた**後**に同じ生成器から引く（T2V の指紋を変えないため）。作り方は 3 実装で違い、ビット一致は期待しない。
- 画像の値域は **ImageNet 正規化（0〜1 を基に）**。LTX の −1〜1 を持ち込まない。LTX の公式関数（`load_image_conditioning`）は H3 の venv に無く、H.264 の再圧縮も H3 ではしない。
- `H3VideoVAE.encode` は第 1 弾で未検証（ソースの docstring が明記）。値の照合は 2a-G3 で初めて取る。
- 復号の uint8 化は `clamp → ×255 → round → uint8` の順を守る。混ぜ（`blend`）は正規化を戻す前に済ませる。
- `release_installed()` は包みを外さない。常駐の入れ直しは印から元の `forward` を戻す剥がし（2.5 の `_unpatch_block_swap` と同じ考え）を H3 側に書く。同期経路は最後の窓のブロックを GPU に残すので、ジョブの終わりに明示的に CPU へ戻す。
- `GenerateRequest` から LTX の検査を移すときは、**連結・Alpha Gen・inpaint の要求モデルの検査は動かさない**。移すのは `validate_ltx_constraints` の 3 種だけ。文言は英語のまま、`detail` の形は §3.3 のとおり。
- `num_inference_steps` の `None` は `validate_for_family` が系統の既定値に書き戻す。アダプタ・mock・ログは `None` を見ない前提で書く。
- 4 分類の監査テストは `GenerateRequest` の項目を足すと落ちる。2.3・2.5 の表に `shift_video`／`shift_audio` を「無視」で足す。
- 操作パネルの `paramUtils` は省略時の既定を LTX にして、Chained・Edit・バッチの呼び出し元を変えない。`comfortTable.ts` の `comfortFramesForBudget` は 8n+1 型しか表せないが、H3 はマーカーを出さないので触らない。
- `featureScope.ts` は「サーバーの機能名を知ってよい唯一のファイル」。系統 id の分岐を画面に書かない。閉じた機能の項目は要求に載せない。
- `isDefaultFilePresent`（モック）は LTX23 以外が全部 `ltx25Install` に連動している。3 件目を足すときに分ける。
- 右クリックの事前入力 3 か所は快適上限の落ち先でフレーム数を書き換えている。系統に表が無いときは格子へのスナップだけにする。
- `npx tsc --noEmit` は偽の合格。`npm run typecheck` を使う。vitest は空いた機械で回す。
- PYTHONUTF8・FastMCP の `Literal | None` など過去の落とし穴は記憶と VERIFICATION_LOG にある。

## 9. 危険と、分からない点

1. **API の検査の移動（本丸）**: 422 の形と文言を §3.3 で一字一句決めたが、Gradio の `format_api_error` と MCP の表示は `loc` と `msg` を連結するので、見え方の差があれば記録する。`num_inference_steps: 8` を本体に入れているテスト群（20 本以上）の個々の期待値は未読。2b-G1 の全件通過で担保する。
2. **TE の直接読み**: 計算は変えないので 2a-G2 でビット一致を要求できる。時間は同程度の見込み（推測）。
3. **常駐の主メモリ**: 作業セット約 30GiB の見込みは推計で、コミット量はそれより大きい。2a-G4 で実測し、64GB 機の余裕を確かめる。先読みの `teardown` 後に各ブロックの引数が CPU の正本を指すかは未確認（C1 が確かめる）。
4. **I2V の照合は部品単位**。DiffSynth の全パイプラインを回す正解は作らない。先頭フレームの PSNR は目安で、合否は目視と合わせて判断する。符号化の bf16 どうしのビット一致はタイルの切り方と演算順が同じ場合にだけ成り立つ。
5. **視覚塔**: テンソルの型・書式・解像度は未確認（§8）。int8 なら `H3Int8Linear` に乗る見込み。正解側は DiffSynth に同じ int8 ファイルを読ませる。
6. **操作パネルの格子の差し替え**: 右クリックの事前入力と `useGenerationForm` の呼び出しが多い（28 か所）。既定を LTX にして差分を小さくするが、vitest とオーナーの目視で確かめる。
7. **1 行のスライダー**: WebView2 の狭い幅では `.field-inline` が折り返す（420px 未満）。実機で確かめる。
8. **記述子の `downloads`**: Comfy-Org/MiniMax-H3 のゲートの有無と正式なファイル名は第 3 弾のインストーラで初めて検証する。本弾では配置済みのファイルで動かす。
9. **Gradio の i18n とテスト**: 3 本のスライダーの差し替えが `tests/test_gradio_*` に及ぶ範囲は未読。
10. **進捗の `decode` 段**: 復号の間は割合が 0.90 で止まって見える。今回は変えない。
11. **快適上限の配信表の網羅**: LTX の「表はあるが行が合わない」場合の頻度は未確認だが、今回はその経路を変えない。

## 10. 敵対的レビューの結果と採否

レビュー（Opus・`plans/shimmering-twirling-octopus-agent-abbda3b038a953c38.md`）: Critical 1・Major 13・Minor 22。監督がコードで裏取りした事実: mock の `max(1, steps)`（`ltx/adapter.py:1605`）・ジョブ開始ログの `%d`（`pipeline_manager.py:661-670`）・`GET /status` はアプリ側で計算（`pipeline_manager.py:302-334`）・pydantic の 422 の包み（`main.py:370-388`）・`validate_ltx_constraints` の切り分け（`api/models.py:644-700`）・2.5 の `_unpatch_block_swap`（`engine25/gguf_transformer.py:1149`）。

| 指摘 | 採否 | 反映 |
|---|---|---|
| C-1 ステップ数の既定 `None` が mock・ログ・記録を壊す | **採用** | §3.3「`None` の解決は 1 か所」（`validate_for_family` が書き戻す） |
| M-1 `release_installed()` は包みを外さない・GPU に残る | **採用** | §3.2 常駐（印による剥がし・CPU への戻し）・2a-G4 の allocated 0 |
| M-2 Alpha Gen の非表示化は H3 に効かず 2.3 を変える | **採用（削除）** | §3.5 から削除 |
| M-3 NAG を隠すと解除できず全ジョブ 422 | **採用（灰色＋項目を載せない）** | §3.5・了承時の確認 ③ |
| M-4 機能名の多くが要求の項目に無い | **採用（引き算）** | §3.4 の導出規則・§3.5 の表（`chain` で A2V を閉じる・`two_stage_hq`） |
| M-5 `crop_output` の扱いが無い | **採用** | §3.4 動作（LTX と同じ切り抜き） |
| M-6 快適上限の落ち先を外す条件が広い | **採用** | §3.5「その系統の表が無い」に限定 |
| M-7 TE の直接読みは HF の `forward` を残す | **採用** | §3.2・§8 |
| M-8 符号化器だけ f32／組まないは構造の変更が要る | **採用** | §3.1 VAE の構造（分割）・§1-99 の吸収は了承時に確認 |
| M-9 常駐の門が証明しない事項 | **採用** | 2a-G4 の並びと記録項目 |
| M-10 基準取りの順序 | **採用（順序を変更）** | C1 の順序を「基準取り→復号→直接読み→I2V→常駐」に |
| M-11 `ready` の項目は事実と違う | **採用（削除）** | §3.2 |
| M-12 422 の形の中身 | **採用** | §3.3 一字一句と順序 |
| M-13 テストと文書の所有の重なり | **採用** | §4（`test_engine_h3_adapter.py`・C5 の所有・FE 文書は C4） |
| m-1〜m-14・m-18〜m-22 | **採用** | 各節（引用の書き直し・3 例目の保留・2.3 の「宣言のみ」・`le=100`・定数の一本化・拒否表に一本化・強度を送らない・乱数の引き順・丸めの順・視覚塔へ同じ画像・int8 の正解側・許容の上限・2a-G5 の範囲と線・見込みの参照・422 の文言・キー差分の表・Gradio の溢れ表・文言に数値を書かない・`h3` 欄の中身） |
| m-15 §1-99 の吸収・m-16 Edit タブ丸ごと灰色・m-17 A2V と NAG の分類 | **了承時にオーナーへ確認** | §1 の「了承時に確認したい 3 点」 |
| B(2) `frame_rate.fixed` | **採用（`{min, max}` に統一）** | §3.3 |
| E-7 操作パネルが既定でも常に送る | **棄却** | 送る規則を 1 つに保つ方が単純。書き戻しがあっても害はない |
