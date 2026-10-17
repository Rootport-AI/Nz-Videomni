# MiniMax H3 設計 — 映像と音声を同時に作る第 3 のベースモデルの載せ方

## 本書の位置づけ

**生きた文書（MiniMax H3 系統の設計正本）。** ヘッダーのドロップダウンで選べる第 3 のベースモデル「MiniMax H3」（映像と音声を 1 つのモデルで同時に作る 33B の拡散モデル）を、本製品でどう動かすかを記述する。**本書には現在の姿だけを現在形で書く。** どういう順序でここへ辿り着いたか（調査の経緯・実装での訂正・敵対的レビューの指摘）は記録簿側が保持しているので、本書は繰り返さない。

- マルチエンジン機構（ベースモデルの切り替え、記述子、エンジン系統ごとのワーカーと仮想環境）の土台は [`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) である。本書はその上に載る H3 固有の取り決めだけを書く。
- 調査の記録は [`MINIMAX_H3_RESEARCH_NOTES.md`](MINIMAX_H3_RESEARCH_NOTES.md) である（**参考資料**。設計の正本ではない）。本書と記述が食い違った場合は本書を採る。
- 作業の入口は台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) の MiniMax H3 の項目（§1-93）である。件数や項目番号の一覧は本書に書かない。
- **現在の実装範囲**: 「第 1 弾＝ワーカーのセルフテスト（T2V を 1 本生成して mp4 に書き出す）」は完了している（記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §165）。アプリ側（API・アダプタ・記述子・操作パネル・インストーラ）はまだ無い。範囲の全体は §11。

本書のうち **【オーナー裁定】** と書いた規則は、オーナーの承認なしに変えない設計上の約束である。

### 関連文書

| 文書 | 関係 |
|------|------|
| [`PENDING_TASKS.md`](PENDING_TASKS.md) §1-93 | 作業の入口。弾ごとの計画と進み具合は台帳側にある |
| [`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) | 土台。記述子（§4）・エンジンアダプタ（§5.2）・仮想環境（§5.3）・推論モード（§5.8）・写像表（§9） |
| [`MINIMAX_H3_RESEARCH_NOTES.md`](MINIMAX_H3_RESEARCH_NOTES.md) | 調査の記録（参考資料）。**進め方の裁定**は同ノート §10。配布ファイルの実測は同 §13.2 |
| [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) | 実測と検証の正本（第 1 弾の門の結果） |
| `engine_h3/venv-engine-h3.freeze.txt` | 仮想環境の**版の正本**。本書は版の数字を書かない |
| `engine_h3/engine_h3-venv-pyproject.toml` | 仮想環境を手で作り直すときの依存の仕様（なぜその版かの説明つき） |

---

## 1. 結論（何をするのか）

1. MiniMax H3 を第 3 のエンジン系統（`h3`）として足す。**専用の仮想環境 `.venv-engine-h3`・専用のワーカー `engine_h3/worker.py`・専用のアダプタ**を新設する（MULTI_ENGINE_DESIGN §3.3 の「同居できない系統」パターン）。
2. 重みは公式系 FL2VA の**枝刈り（AdaLN の枝を表に畳んだ版）int8 ConvRot** の DiT と、**int8 ConvRot のテキストエンコーダ**、Comfy-Org 配布の映像 VAE と音声 VAE である。int8 の行列積は comfy-kitchen の CUDA 拡張に任せ、CPU オフロードは本製品の block swap と先読みに載せる。VRAM 16GB で動かす。
3. 推論は CFG なし・1 ステップ 1 回の前向き計算で、映像と音声を同じ系列の中で同時にノイズ除去する。ステップ数と shift はワーカーの引数で受け取る。

---

## 2. オーナー裁定

番号は研究ノート [`MINIMAX_H3_RESEARCH_NOTES.md`](MINIMAX_H3_RESEARCH_NOTES.md) §10 の裁定番号をそのまま使う。**進め方の裁定（作業の順序・了承の取り方・インストーラの名前など）は同ノート §10 に残してあり、本書には移していない。**

- **裁定 3 【オーナー裁定】**: 最初のタスク範囲は、FL2VA 重み 1 本での T2V（文章から動画）と I2V（画像から動画）の両対応とする。Ref2VA（参照素材からの生成）は後回しである。α版の I2V は先頭フレームだけを受け付ける。
- **裁定 6 【オーナー裁定】**: 重みの形式は int8 ConvRot とする。利用者が CivitAI などから落としたファイルをフォルダに置くだけで動かせることを理想とする。理由は、ファインチューンが大量に出回っており、コミュニティの主流が int8 ConvRot だからである。
- **裁定 7 【オーナー裁定】**: 配布は公開リポジトリからの直接ダウンロードとし、再ホストはしない（ライセンスに地域条項があるため）。
- **裁定 8 【オーナー裁定】**: テキストエンコーダは int8 ConvRot を使う。nvfp4 や bf16 は将来の拡張とする。
- **裁定 9 【オーナー裁定】**: 動画の尺は「快適上限を測って決める」方針に揃える。外部ガイドの 1344×768・345 フレームは正式目標にしない。
- **裁定 10 【オーナー裁定】**: LoRA は後回しとする。Turbo LoRA（少ないステップで生成するための蒸留 LoRA）への対応は Style LoRA への対応とほぼ同じ仕組みになるので、Style LoRA に対応するときに一緒に進める。
- **裁定 11 【オーナー裁定】**: H3 の仮想環境は、torch の版を LTX に揃えなくてよい。comfy-kitchen の CUDA 拡張が効く版を選ぶ。利用者側の条件（ドライバ R580 以上・ディスクの増加）は README に書く。
- **裁定 16 【オーナー裁定】**: int8 の実行は comfy-kitchen の CUDA 拡張に任せる。H3 の仮想環境の torch は、MiniMax H3 を動かすことに最も適した版を選ぶ。本製品はモデルごとに仮想環境を持てる設計なので、既存の torch の版にこだわらない。
- **裁定 17 【オーナー裁定】**: 実装の土台は DiffSynth-Studio（Apache-2.0）、diffusers は数値の正解データ源、ComfyUI（GPL-3.0）は仕様書として読むだけとする。次の 4 点を設計に織り込む。(1) int8 の実行は comfy-kitchen に任せる（既存の読み込み〔呼び出しごとに bf16 へ戻す・groupsize 256 のみ〕はヘッダの判定には使えても、H3 の実行経路には向かない）。(2) attention は DiffSynth の区間ごとの Python ループを使わず、diffusers の「1 リクエストを 1 つの系列に詰めてマスクなしで渡す」形を手本にして書き、本製品に導入済みの SageAttention を効かせる。(3) CPU オフロードは DiffSynth の仕組みではなく、本製品の block swap と先読みを接続する。(4) ローダは「揺れを吸収する」作りで書く（DiffSynth の変換表を出発点に、ComfyUI の判定の振る舞いを仕様として自前で書く）。
- **裁定 18 【オーナー裁定】**: 第 1 弾ではステップ数と shift を定数の既定値とし、ワーカーは引数で受け取る。画面に出すのは裁定 22 のとおり第 2 弾である。
- **裁定 19 【オーナー裁定】**: α版の I2V は先頭フレームのみとする。末尾キーフレームと中間キーフレームは台帳 §1-97 で扱う。
- **裁定 22 【オーナー裁定】**: ステップ数と shift（映像・音声）の 3 本のスライダーを、第 2 弾で操作パネルと Gradio に出す。既存の CFG スライダーのモックを差し替え、項目名・スライダー本体・数値を 1 行に並べる。既定値は第 2 弾の着手時に決める。背景は、Turbo を融合した低ステップの配布物が多く、LoRA 未対応でも速く生成できる見込みがあることである。

### 裁定ではなく設計として決めたこと（裁定 17 の論点）

- (a) DiffSynth は**移植**する（パッケージとしては入れない）。§8。
- (b) 映像 VAE は Comfy-Org の fp16 版、音声 VAE は同 fp32 版を使う。§8。
- (c) CFG は使わない。§3。
- (d) comfy-kitchen の triton 経路は切り、CUDA 経路に固定する。§5。

---

## 3. モデルの正体と推論の形

配布ファイルの実測は研究ノート §13.2 にある。ここでは推論の形だけを書く。

- **規模と構成**: DiT は約 33B パラメータ。隠れ幅 5376・50 ブロック・56 ヘッド×128 次元。このうち約 13B は AdaLN（時刻に応じて各層の変調量を決める枝）で、枝刈り版ではこの枝が表に畳まれている。
- **映像と音声を同時に作る**: 映像と音声（32kHz ステレオ）のトークンを 1 本の系列に詰め、同じ DiT で同時にノイズ除去する。
- **CFG なし【設計】**: モデルはガイダンス蒸留済みで、negative prompt も guidance scale も無い。1 ステップにつき前向き計算は 1 回である。
- **Rectified Flow の Euler**: モデルの出力は「データ方向の速度」v で、`x0 = x + σ·v` と読む（一般の flow-match と符号が逆）。更新は `x_next = x + (σ − σ_next)·v`。N ステップは N 回の評価である。
- **shift は 2 本立て**: σ の列は `base = linspace(1, 0, N+1)[:-1]`、`σ = s·base / (1 + (s−1)·base)` で作る。s は映像用と音声用の 2 つで、同じ添字 i について映像は σ_v[i]、音声は σ_a[i] を使い、出力もそれぞれの列で別々に更新する。モデルに渡す時刻は `t = 1 − σ`（1 が無ノイズ）。
- **フレームと解像度の規約**: 24fps 固定。フレーム数は **17n+5**。潜在フレーム数は `T' = 5n + 2`、音声の潜在数は `A = round(F / 24 × 40)`（例: 124 フレームで T'=37、A=207）。縦横は **32 の倍数**（映像 VAE の 16 倍圧縮とパッチ 2 の積）。短辺 768 が既定である。
- **系列の並び**: `[ text(L) | cond | audio(2A) | video(T'×(H/32)×(W/32)) ]`。cond は条件付けのキーフレーム行で、T2V では空である。attention は系列全体で全結合（区間ごとの制限もマスクも無い）。token_refiner（テキストだけを整える 2 ブロック）はテキスト行の中に閉じ、時刻にも位置にも依存しないので、ループの前に 1 回だけ計算する。
- **AdaLN の添字**: 各行が使う変調の行は `時刻の番号 × 3 + タグ`（タグは 0＝映像、1＝テキスト、2＝音声）。時刻の番号は、映像・テキスト・音声（と条件行）の時刻を `unique` でまとめた番号である。最終層の添字は時刻の番号だけである。
- **枝刈り版の AdaLN（表の補間）**: 時刻はスカラーなので、「時刻の埋め込み→AdaLN 射影」の写像を格子点の表（`adaln_t_table`、格子数 G・ランク R）に先計算してある。表の引き方は次の式で、SiLU は挟まない（float32）。

  ```text
  pos = clamp(t, 0, 1) × (G − 1)
  i0  = min(floor(pos), G − 2)        … 上端 t=1.0 でも範囲外を読まない
  w   = pos − i0
  e   = lerp(table[i0], table[i0+1], w)
  変調 = adaln_proj.linear(e)
  ```

- **RoPE（位置符号化）**: 3 軸（時間・縦・横）。座標は float64 で作り、角度と cos／sin は float32 で計算する。128 次元のうち先頭 96 次元だけを回す。映像の時間は一様でなく、`5/3 × (1,4,4,4,4)` の周期で進む。

---

## 4. 3 段の推論

一度に GPU（と主メモリ）に置くのは 1 段ぶんだけである。段が終わるたびに解放する。

| 段 | 内容 | GPU に置くもの | 解放するもの |
|---|---|---|---|
| 1. 条件付け（`encode`） | テキストエンコーダでプロンプトを埋め込みにする | テキストエンコーダの一部の層（`te_layers_on_gpu`。残りは CPU に置き、層ごとに出し入れする） | テキストエンコーダ全体 |
| 2. ノイズ除去（`stage1_denoise`） | DiT をステップ数だけ回す | ブロック以外（埋め込み・最終層など）と、block swap が選んだ一部のブロック（`blocks_on_gpu`） | DiT 全体 |
| 3. デコード（`decode`） | 映像 VAE と音声 VAE で復号し、mp4 に書く | 映像 VAE → 音声 VAE の順に 1 つずつ | VAE |

- 主メモリの山は、段 1 のテキストエンコーダを解放してから段 2 の DiT を読む逐次なので、両者が同時に載ることはない。ただし実測では、作業セット（実メモリに置いている量）の山が約 29GiB、コミット量（OS が確保を約束した量）の山が約 40GiB になる。テキストエンコーダを解放しても作業セットが下がらず、割り当て器が解放済みの領域を抱えているとみられる（推測）。32GB の機械はページファイルが前提になる（数値の出所は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §165.5）。
- 毎生成、段 1 から段 3 まで組み直す（常駐はしない）。

---

## 5. int8 ConvRot の実行方式

- **素の int8 `Parameter` ＋ f32 `[N,1]` のスケール**: 量子化された Linear は `H3Int8Linear` という自前のモジュールにする。重みは素の int8 の `nn.Parameter`（`requires_grad=False`）、スケールはファイルどおりの f32 `[出力次元, 1]` のバッファ、bias は任意である。`forward` は `comfy_kitchen.int8_linear` を呼ぶ。ConvRot（活性化を回転してから量子化する方式）の有無と groupsize は層ごとの記述（`comfy_quant`）から取る。
- **量子化テンソル型を使わない理由**: comfy-kitchen には重みをラッパー型のテンソルとして持つ量子化テンソル型もある。しかし、ラッパー型のテンソルは本製品の block swap と先読み（素のテンソルを前提に重みを出し入れする）に乗らない。関数 API の `int8_linear` は同じ CUDA カーネルに届くので、素のテンソルで足りる。
- **CUDA 経路に固定する**: 起動時に `comfy_kitchen.list_backends()["cuda"]` が使用可能で、能力に `int8_linear` を含むことを確かめ、満たさなければロードを拒否する（理由の文を返す）。そのうえで **triton と eager の両経路を切る**。
  - triton を切る理由: comfy-kitchen の triton 経路の int8 行列積は、トークン数×出力幅が 2³¹ 以上で出力の位置が int32 であふれ、プロセスごと落ちる既知の不具合がある（上流 issue #136）。CUDA 経路は出力の位置を int64 で計算しているので、この限界を持たない。
  - eager を切る理由: eager は制約から外れた呼び出しを黙って受けて遅くなるだけで、速度の低下から区別できない。切っておけば、CUDA 経路に乗らない呼び出しは `NoCapableBackendError` で止まる（**fail loud**）。
- **CUDA 経路に乗る条件**: 重みとスケールが GPU 上にあること。block swap が重みを GPU に載せた状態で呼ぶ限り満たされる。
- groupsize は 256 以外（例: 64）も、そのまま `int8_linear` に渡す。

---

## 6. 既存部品との接続

- **block swap と先読み**: 既存の `engine/transformer/block_swap_service.py` と `block_swap_prefetch.py` をそのまま使う。`BlockSwapService` はパイプラインに常駐し、ジョブごとには作らない。H3 の本体ブロック列の名前が `blocks` で、block swap が探す候補名に当たる。素の `nn.Module` の列なので、既存の形でそのまま乗る。共有部品は変更しない。
- **テキストエンコーダの層オフロード**: `engine/gemma/layer_offload_service.py` の `GemmaLayerOffloadService`（torch だけで書かれ、`.layers` を探す汎用実装）を使う。
- **推論の囲み【設計】**: `engine_h3` は**全経路 `torch.no_grad()`** で書く（`inference_mode` は 1 箇所も書かない）。規約と理由は MULTI_ENGINE_DESIGN §5.8。CPU テストで `engine_h3` 配下に `inference_mode` が現れないことを固定する。
- **精度の島の dtype**: 読み込みの直後に `H3DiT.normalize_dtypes()` で、精度の島の内外の dtype を揃える（公式ファイルでは AdaLN 射影の F16→F32 だけが変わる）。
- **meta で組んで `assign=True`【設計】**: DiT もテキストエンコーダも `torch.device("meta")` の中で組み、量子化層を `H3Int8Linear`（meta）に差し替え、テンソルを 1 本ずつ読んで `load_state_dict(assign=True)` で入れる。読み込みの後に meta のまま残ったテンソルが無いことを検査する。普通に組むと DiT が f32 で約 77GB、テキストエンコーダが bf16 で約 49GB を先に確保して主メモリ（64GB）を超えるので、この作りは必須である。

---

## 7. ヘッダの判定と吸収する揺れ

- **判定はテンソル名の指紋**: H3 の DiT は `{接頭辞}video_patch_proj.weight` と `{接頭辞}audio_patch_proj.weight` の**同時存在**で判定する。GGUF の KV のようなメタデータには頼らない（コミュニティ配布物は KV が偽装・欠落しており、safetensors の `__metadata__` も枝刈り版では空である）。MULTI_ENGINE_DESIGN §2.2・§2.4 の判定とは別の経路で、H3 系統は safetensors にこの指紋を使う。
- **吸収する揺れ（第 1 弾でただで取れる分）**:
  - 接頭辞 4 通り（無し／`model.diffusion_model.`／`model.model.`／`net.`）を剥がす。
  - 構成（層数・幅・ヘッド数・ffn・AdaLN 表の格子数 G とランク・枝刈りか否か）は、メタデータでなく**テンソルの形状**から導く。
  - `comfy_quant` は `format`・`convrot`・`convrot_groupsize` の 3 キーだけを読み、他のキーは無視する。
  - groupsize は 256 以外も通す。量子化されていない層は、読んだ後に島の外側を bf16、内側を f32 に揃える。
  - 知らないテンソルは警告ログを出して捨てる。
- **拒否するもの**: 非枝刈り版（`adaln_t_table` が無い）と、旧式の量子化記述（`_quantization_metadata`）、`int8_tensorwise` 以外の形式、VSA の追加テンソル、PDD ヘッドは、第 1 弾では理由の文つきで拒否する（または警告して捨てる）。試すのは公式のファイルだけで、コミュニティ版は第 2 弾以降である。
- **置き場**: 判定は torch に依存しない `h3_format.py`（リポジトリ直下。`sft_quant_format.py`・`chain_math.py` と同じ置き場）に書く。アプリ側（torch の無い仮想環境）の事前検査からも呼ぶためである。
- **FL2VA と Ref2VA はヘッダで区別できない**（テンソル名・形状が同じ）。記述子でファイル名を固定して縛る（第 2 弾）。

---

## 8. ファイル構成と境界

### 実装の土台

DiffSynth-Studio の H3 部品を**移植**する（Apache-2.0。移植したファイルの先頭に由来と表示を書く）。パッケージとしては入れない（`import diffsynth` が core 全体と modelscope などを芋づるに引き込むため）。ComfyUI（GPL-3.0）のコードは写さない。映像 VAE は Comfy-Org の `minimax_h3_video_vae_fp16.safetensors`（計算は bf16）、音声 VAE は同 `minimax_h3_audio_vae_fp32.safetensors`（計算は f32）である。`latents_mean`／`std` は `__metadata__` の JSON（f32）から取る。

### ファイル構成

| ファイル | 役割 |
|---|---|
| `h3_format.py`（リポジトリ直下・torch 非依存） | ヘッダの判定・接頭辞の剥がし・構成の導出・`comfy_quant` の読み・量子化層の一覧（§7） |
| `engine_h3/constants.py` | 定数と既定値（§9）。フレーム規約の算術 |
| `engine_h3/sft_load.py` | テンソルを 1 本ずつ読んで meta のモデルへ入れる（`assign=True`）。読み残しの検査 |
| `engine_h3/quant_linear.py` | `H3Int8Linear`・CUDA 経路の検査と固定・量子化層の差し替え（§5） |
| `engine_h3/dit.py` | DiT 本体・枝刈り版の AdaLN・attention（SDPA／SageAttention）・行の並べ替え |
| `engine_h3/rope.py` | 3 軸 RoPE |
| `engine_h3/text_encoder.py` | Qwen3-VL の文章側（50 層。正規化前の隠れ状態を出す）・層オフロード |
| `engine_h3/scheduler.py` | σ の列と Euler の 1 ステップ |
| `engine_h3/video_vae.py`・`engine_h3/audio_vae.py` | 映像 VAE・音声 VAE |
| `engine_h3/media_out.py` | mp4 の書き出し（映像 libx264、音声 AAC、色空間は BT.709 を明示して変換） |
| `engine_h3/pipeline_h3.py` | 3 段の組み立て（§4） |
| `scripts/run_h3_pytest.py` | エンジン用の仮想環境でテストを回す実行器 |
| `tests/test_h3_format.py`・`tests/test_h3_engine.py`・`tests/test_h3_media.py` | ヘッダの判定（アプリ側の仮想環境）／DiT・テキストエンコーダ・ワーカーの約束事（GPU を使う 2 件は既定で飛ばす）／映像 VAE・音声 VAE・mp4 書き出し（CPU） |
| `engine_h3/worker.py` | ワーカー本体。`engine25/worker.py` と同じ作法（`python -m engine_h3.worker`、stdin の JSON 行、stdout の `@@LTX@@`＋JSON、stderr のログ `[h3_worker]`）。torch はモジュール最上位で import せず、`load` の中で遅延 import する |

### 呼び出しの分担

`load` はパスの検査とヘッダの判定だけを行い、重みは読まない。`generate` が毎回、3 段を組み直す。

### ワーカーのメッセージ

第 2 弾のアダプタの `_build_load_payload` がそのまま写せるよう、キー名はここで固定する。

- **`load`**: `transformer_path`・`text_encoder_path`・`video_vae_path`・`audio_vae_path`・`tokenizer_dir`・`blocks_on_gpu`・`te_layers_on_gpu`・`attention_backend`・`block_swap_prefetch`・`deterministic`
- **`generate`**: `prompt`・`seed`・`width`・`height`・`num_frames`・`frame_rate`・`num_inference_steps`・`shift_video`・`shift_audio`・`output_path`（ステップ数と shift は省略可。省略時は既定値）
- **イベント**: `ready`（`sage_available`）／`progress`（`stage`・`index`・`total`）／`done`／`error`（`detail`）
- **進捗の段階名**: LTX の語彙を借りる。`encode`／`stage1_denoise`（単段のノイズ除去。H3 は 1 段なのでこの名前だけを使う）／`decode`。第 2 弾のアダプタは進捗の割合計算に `_read_worker_events(one_stage=True)` を使う。
- **`done` のキー**: `seed_used`・`sampler`・`num_inference_steps`・`shift_video`・`shift_audio`・`peak_vram_mb`・`peak_vram_reserved_mb`・`rss_peak_gib`・`seconds`・`num_frames`・`frame_rate`・`width`・`height`・`size_bytes`・`attention_used`・`block_swap_prefetch_used`・`commit_gib`・`commit_peak_gib`（コミット量）・`audio_samples`（切り揃え後）・`audio_samples_decoded`（復号した全量）・`h3`（段ごとの秒数と RSS を `phases` に、使われた量子化経路を `quant_backends` に持つ）

### 映像 VAE の特例

潜在フレームが 1 つ（T'=1）のときは、単独では復号が悪いので、5 つに複製して復号し、3 番目を取る。

### tokenizer の渡し方

テキストエンコーダの tokenizer は、記述子のカテゴリではなく**記述子の `assets`** で渡す予定である（第 2 弾）。`<d>` などの追加特殊トークンは `tokenizer_config.json` にしか無いので、`tokenizer.json` だけでは足りず、ファイル一式を 1 つのフォルダとして扱う。追加特殊トークン 7 語が語彙の 151669〜151675 に入ることを、ロード時に検査する（入らなければ明示的に足す）。

### セルフテスト

`python -m engine_h3.worker --selftest-generate` が、`load`／`generate` を合成メッセージで直接呼んで全イベントを記録し、mp4 の SHA-256 と `done` を JSON で書く。

---

## 9. 定数と既定値

- **値の正本は `engine_h3/constants.py`** である。本書は値を写さない。
- 置くもの: ステップ数・shift（映像・音声）の既定値、24fps、空間 32（1 トークン＝32px 四方）、VAE の圧縮率とクリップ長、音声の潜在レート・サンプルレート・ホップ長、モダリティのタグ（映像 0・テキスト 1・音声 2）、ログの接頭辞 `[h3_worker]`、`latent_frames(F)`・`audio_latents(F)`（17n+5 を外れた F は `ValueError`）。
- ステップ数と shift は、ワーカーの `generate` が引数で受け取り、省略時に既定値を使う。製品の既定値は第 2 弾の着手時に決める（裁定 22）。
- 解像度の規約: 縦横は 32 の倍数、フレーム数は 17n+5 かつ `MIN_NUM_FRAMES = 22` 以上（F ≥ 22。F=5 は拒否する）、フレームレートは 24 固定。違反は日本語の理由つきの `ValueError` で止める。アプリ側の API の制約は第 2 弾（§11）。

---

## 10. 仮想環境

- **名前**: `.venv-engine-h3`。MULTI_ENGINE_DESIGN §5.3 の一覧に並ぶ 3 つ目のエンジン用仮想環境である。
- **torch を LTX に揃えない理由【オーナー裁定：裁定 11・16】**: comfy-kitchen の CUDA 拡張は CUDA 13 系の torch で効く。拡張自体は torch の DLL にリンクしていないので、CUDA 13 系でさえあれば torch の版は問わない。そのうえで、SageAttention・triton-windows などの周辺の実績がある版を選ぶ。既存の 2 系統の torch（CUDA 12.8 系）のままでは CUDA 拡張が使えない。
- **利用者側の条件**: NVIDIA ドライバ **R580 以上**（CUDA 13 系の下限）。GPU は **Turing（RTX 20 系）以降**（CUDA 13 で Pascal 以前が削除された）。README に書く（第 3 弾）。
- **容量が増える**: torch の版が既存の仮想環境と違うため、uv のハードリンク共有（MULTI_ENGINE_DESIGN §5.3「容量は心配しなくてよい」）が効かず、この仮想環境ぶんの数 GB がそのまま増える。実測値は freeze と記録簿を見る。必要空き容量の正本は README のハードウェア要件表である。
- **入れないもの**: torchaudio（CUDA 13 の Windows 版が torch の新しい版と組めず、H3 の経路は使わない）、diffsynth（部品は移植した）。
- **版の正本は `engine_h3/venv-engine-h3.freeze.txt`** である。手で作り直すときの仕様と、各版を選んだ理由は `engine_h3/engine_h3-venv-pyproject.toml` に書く。

---

## 11. 弾の分け方と門

### 弾の分け方

| 弾 | 範囲 |
|---|---|
| 第 1 弾（完了。記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §165） | 仮想環境・重みの取得・`engine_h3/` と `h3_format.py`・T2V セルフテスト 1 本・本書 |
| 第 2 弾 | I2V（先頭フレーム）のセルフテスト → アダプタ・系統登録・記述子・ヘッダ判定（`h3_format.py` をアプリ側の事前検査から呼ぶ）・API の制約（17n+5・24fps 固定・32 の倍数）・API にステップ数と shift（映像・音声）の 3 項目・操作パネル（ドロップダウンは自動追随・フレーム数のスナップ・快適上限マーカーの非表示・`featureScope.ts`・ステップ数／Shift（映像）／Shift（音声）の 3 本のスライダー〔裁定 22〕）・Gradio の同じ 3 本・MCP の最小対応・テスト・常駐の要否 |
| 第 3 弾 | `install-MiniMaxH3.bat`（仮想環境の構築と取得を 1 回で）・`Ensure-EngineVenv` の 3 回目の呼び出し・freeze・README の要件行（ドライバ・GPU 世代・容量） |
| 以後 | 台帳 §1-95（快適上限の較正）・§1-97（末尾と中間キーフレーム） |

- 第 2 弾の「API の制約」は、MULTI_ENGINE_DESIGN §7 が「エンジン別の UI パラメータ制約は作らない」としてきた領域に踏み込む（8n+1 と 17n+5 の両方を満たす値は 73・209・345・481 の 4 つだけ）。第 2 弾の計画で正面から扱う。
- 快適上限は、較正が済むまで記述子に `comfort` の行を持たせず、操作パネルのマーカーを出さない。

### 門（第 1 弾）

| 門 | 内容 | GPU |
|---|---|---|
| G0 | 仮想環境の構築が成功し、主要パッケージが import できる。`torch.version.cuda` が 13.0。freeze と容量を記録 | 不要 |
| G1 | `list_backends()["cuda"]` が使用可能で、能力に `int8_linear` を含む（合否はこれだけ。SageAttention の可否は記録に回す） | 不要 |
| G2 | 合成の int8 重みと第 1 弾の系列の大きさの入力で、`int8_linear` が CUDA 経路で走り、eager の結果と末尾の行まで一致する | 使う |
| G3 | block swap の噛み合わせ: `H3Int8Linear` 入りの 50 個のブロック列に block swap を `install` し、CUDA 経路を示す印が `"on"`、かつ「全ブロック常駐」「同期」「先読み」の 3 通りで出力がビット単位で一致 | 使う |
| G4 | 部品の数値照合: DiffSynth の実装を正解側に、同じ重みから部品ごと（ブロック 1 個・token_refiner・最終層・AdaLN の表・RoPE・VAE・テキストエンコーダの先頭と末尾の層）に同じ入力で比べる。テキストエンコーダは層オフロードの on と off で出力がビット単位で一致 | 使う（小さい） |
| G5 | セルフテスト 1 本: 640×384・124 フレーム・24fps・20 ステップ・固定シードで、映像と音声の入った mp4 ができ、`done` に必要な数値が出る | 使う |

- GPU を使う門は同時に 1 体だけ回す。結果は記録簿に書く。第 1 弾のクローズはオーナーの判断である。オーナーの目視は第 2 弾の操作パネルでまとめて行う。

---

## 12. 書かないもの・後回し

第 1 弾では次を書かない。

- LoRA・Turbo（裁定 10）
- Ref2VA・Hybrid（FL2VA と Ref2VA を混ぜた配布物）
- 視覚塔と条件行（I2V。第 2 弾）
- 末尾キーフレームと中間キーフレーム（台帳 §1-97）
- 連結・画角拡張・NAG／VSF
- `keep_resident` と常駐

量子化の旧式の記述（`_quantization_metadata`）・VSA の追加テンソル・PDD ヘッドは、警告して捨てるか拒否する（§7）。

---

## 13. 未決事項

- **性能**: 第 1 弾の G5（640×384・124 フレーム・20 ステップ）は全体で約 140 秒だった（実測の表は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §165.5）。大きな画角では block swap の転送（毎ステップ約 19GB）が上限になる可能性がある（推測）。遅すぎて快適上限の較正に支障が出るなら、LoRA（Turbo）対応の優先順位を上げる（研究ノート §10 の裁定 21）。
- **SageAttention の既定**: 新しい仮想環境の torch で動き、ノイズ除去が約 25% 速かった（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §165.7）。既定を SDPA のままにするか sage にするかは第 2 弾で決める。
- **m×n ≥ 2³¹ の CUDA 経路**: ソースの水準では限界が無いが、実測は台帳 §1-95（快適上限の較正）で行う。
- **サブマシンのドライバ**: 未確認。第 3 弾の README までに確かめる。
- **メインメモリ 32GB 機**: コミット量の山が約 40GiB なので、ページファイル前提になる（§4）。README に書く（第 3 弾）。
- **テキストエンコーダの直接読み**: 層ごとにファイルから GPU へ直接読み、主メモリの山を DiT ぶんにする（案。第 2 弾の計画で扱う）。
- **大きな画角での復号の VRAM**: 復号結果を float32 のまま GPU に全部置くので、1344×768・345 フレームで 16GB を超える見込みである。窓ごとに uint8 にして CPU へ移す形へ直す。快適上限の較正（台帳 §1-95）の前に直す。
- **T2V で映像 VAE の符号化器を読まない**: 符号化器（約 0.34GiB）は I2V にしか要らない。
- **製品の既定のステップ数と shift**: 第 2 弾の着手時に決める（裁定 22）。
- **ヘッダの揺れのうち第 2 弾以降に回したもの**: コミュニティ版の読み込み、非枝刈り版、groupsize の混在への対応範囲。
