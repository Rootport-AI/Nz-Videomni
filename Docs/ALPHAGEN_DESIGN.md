# Alpha Gen（動画からマットを作る） — 普通の動画から、透過に使う白黒のマット動画を作る

## 本書の位置づけ

**生きた文書（Alpha Gen 機能の設計正本）。** Lightricks が公開した LTX 2.5 用の IC-LoRA「LTX AlphaGen」を使い、利用者の手元の普通の RGB 動画から、同じ大きさ・同じフレーム数の白黒の**マット**動画を作る機能を記述します。マットとは、白＝不透明・黒＝透明・灰＝半透明を明るさで表した動画で、元の動画に当てると背景が抜けます。**IC-LoRA**（参照動画つき LoRA）は、参照として渡した動画を条件にして生成を導く追加部品です。

- 作成: 2026-10-07
- 対象の台帳項目: [`PENDING_TASKS.md`](PENDING_TASKS.md) §1-83（第 1 弾＝バックエンド）。第 2 弾（操作パネル）は同 §1-85、第 3 弾（出口）は同 §1-86、LoRA ウェイトの配布は同 §1-87 です
- 状態: **第 1 弾の実装中**（第 2 弾・第 3 弾は未着手。ただし決まっている事項は本書 §9・§10 に収めてあります）

**本書には現在の設計だけを現在形で書きます。** 第 1 部「決まったこと」は、実装を担当する人が第 1 部だけを読めば着手できることを目標に書いてあります。第 2 部「なぜそう決めたか」は、一度否決した案を再提案しないための記録です。決定の時系列は残しません（それは [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) の役割です）。

### 関連文書

| 文書 | 関係 |
|------|------|
| [`LTX_ALPHAGEN_RESEARCH_NOTES.md`](LTX_ALPHAGEN_RESEARCH_NOTES.md) | 事前調査ノート。公式の仕様・ライセンス・AviUtl2 側の受け皿の棚卸しはここにあり、本書には書き写さない |
| [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §151 | **ゲート 0（go／no-go の簡易テスト）の記録。** 1 段原寸相当の負荷の実測・2 段経路でマットが出る条件・LoRA のヘッダの中身はここが正本 |
| [`Videomni_Backend_Specification.md`](../Videomni_Backend_Specification.md) §6 | API 契約の正本。本書 §4 は設計の説明で、フィールドの型・既定・エラーコードは仕様書が正本 |
| [`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) §9 | 快適上限の正本。Alpha Gen の暫定の予算もここに置く（本書 §6） |
| [`PENDING_TASKS.md`](PENDING_TASKS.md) | 作業の入口。§1-83・§1-85〜§1-87、§3-222（自前の入力プラグイン）、§3-223（快適上限の較正） |
| [`STORAGE_POLICY.md`](STORAGE_POLICY.md) | 保存領域の設計原則。本機能はジョブのフォルダに中間物と `matte.mkv` を書くので適用対象 |
| [`INPAINTING_DESIGN.md`](INPAINTING_DESIGN.md) | 前例。窓の切り出し・「元ファイルが寸法の正本」という規則・アプリ側の後処理は Inpainting と同じ形 |

---

# 第1部 決まったこと

## 1. 機能の定義

**Alpha Gen は「透過動画を新しく作る機能」ではありません。** 手元の動画を 1 本入れると、その動画の被写体を抜くためのマットを返します。髪・煙・ガラスのように手作業で抜きにくい縁を、灰色（半透明）として表せるのが特長です。

- **入力**: 利用者がアップロードした動画 1 本（`POST /upload/video` で預けたもの）。そのうちの 1 区間（**窓**）を対象にします。
- **出力**: 同じジョブのフォルダに 2 本を置きます。
  - `output.mp4` — プレビュー用。H.264・元の寸法・音声なし。
  - `matte.mkv` — 合成用の劣化しない灰色のマット。FFV1（可逆圧縮の動画形式）・gray（灰色 1 チャンネル）・全範囲（明るさ 0〜255）・元の寸法・音声なし。
- **対応するベースモデル**: LTX 2.5 だけです。LTX 2.3 を選んでいるときは 422 で断ります（§4.3）。

**モードは 2 つです。**

| モード | 中身 | 既定 |
|---|---|---|
| 1 段原寸モード | Stage-2 を飛ばし、Stage-1 を「出したい原寸・参照つき」で 1 回だけ走らせ、その潜在をそのままデコードする | **既定** |
| 軽量モード | 既存の 2 段経路を、参照つきでそのまま流す | 利用者が選んだときだけ |

**Stage-1／Stage-2** は本製品の LTX 2.5 の生成の 2 つの段です。普段の生成は、Stage-1 で半分の解像度の下描きを作り、Stage-2 で原寸に仕上げます。1 段原寸モードは、その Stage-1 を最初から原寸で走らせ、Stage-2 を行いません。

## 2. 利用者が決めること・固定するもの

### 2.1 利用者が決めること

| 項目 | 範囲 | 説明 |
|---|---|---|
| 入力動画 | — | アップロード済みの動画の ID |
| 窓の開始秒 | 0 以上 | 素材自身の時間軸での開始位置。既定 0 |
| フレーム数 | 9〜145・8n+1（8 の倍数＋1） | 窓の長さ。**推奨は 121 フレーム以下**（ゲート 0 で 1920×1088×121f の 1 段原寸が実用外だったため） |
| fps | 整数 1〜60 | 窓をこの fps で切り出す。小数の fps は受け付けない |
| seed | 整数（-1 は乱数） | 生成の種 |
| 軽量モード | 真偽 | 既定は偽（1 段原寸モード） |

あわせて、他のジョブと同じ高速化・常駐の設定（`attention_backend`・`keep_resident`・`keep_resident_embeddings` など）を**同じ名前・同じ意味で**受け取ります（§4.1）。これは Alpha Gen 用の新しいつまみではありません。

### 2.2 固定するもの（利用者は触らない）

- **プロンプトは空**（公式どおり）。API の内部では空白 1 文字を送ります（`GenerateRequest.prompt` が 1 文字以上を要求するため。ゲート 0 で空白 1 文字でも生成が進むことを確認済み）。
- **LoRA は `alpha-gen` を強さ 1.0** で 1 本だけ。
- **参照の強さは 1.0**（参照動画を条件としてどれだけ強く効かせるか。公式の推奨値）。
- **CFG は使いません**（蒸留版の LTX 2.5 は CFG〔プロンプトへの従い具合の制御〕を持たないため、既存の生成と同じく 1.0 固定）。
- 出力の寸法は元の動画の寸法です（幅・高さのつまみはありません）。

## 3. サーバーの処理

`POST /generate/alpha` を受けたサーバーは、次の順に処理します。生成そのもの（4 番）は既存のジョブの仕組み（`create_if_idle`→`run_job`）にそのまま乗ります。

1. **窓の切り出し。** アップロードされた動画から、開始秒とフレーム数で窓を切り出し、要求の fps へ変換して `_alpha_window.mp4` に書きます（正確なフレーム数・音声つき。`services/video_io.py` の `cut_window_mp4`）。この窓は第 3 弾の合成の元にもなります。
2. **元の寸法の測定。** 元の寸法は**切り出した窓を測って**決めます（アップロード元のファイルではなく）。受付の検査（§4.3 の d）で測ったアップロード元の寸法と比べ、窓の縦横が入れ替わっていたら（回転の情報を持つ素材）、ジョブを失敗にします（§11）。
3. **作業寸法とキャンバスの決定**（§3.1）。
4. **参照の作成**（§3.2）と**生成**（§3.3）。
5. **後処理**（§3.4）と**記録**（§3.5）。

### 3.1 作業寸法とキャンバス

**作業寸法**は、元の動画を縦横比を保って縮めた寸法です（縮める必要が無ければ元の寸法のまま）。**キャンバス**は、作業寸法を 64 の倍数へ切り上げた、実際にモデルへ渡す生成の寸法です。キャンバスの右と下には黒い余白が付きます。計算は `chain_math.py` の `alpha_gen_geometry(src_w, src_h, num_frames, budget)` が行い、`(作業幅, 作業高さ, キャンバス幅, キャンバス高さ)` を返します。

**トークン**はモデルが扱う処理の単位で、32×32 画素×8 フレームで 1 つです。式は次のとおりです。

```
潜在フレーム数 L = (フレーム数 − 1) ÷ 8 + 1
トークン数 = (キャンバス幅 ÷ 32) × (キャンバス高さ ÷ 32) × L
```

**予算 B**（何トークンまでなら快適か）は、1 段原寸モードなら快適上限の行の `alpha_gen_budget`、軽量モードなら同じ行の `single_budget` です（§6）。**予算はキャンバスのトークン数に当てます**（生成側だけを数え、参照側は数えません。参照の分は予算を半分にしてあることで見込んでいます。第 2 部 §16）。

1. 元の寸法のままのキャンバス（`ceil64(元の幅) × ceil64(元の高さ)`）が予算に収まるなら、縮めません。B が無い（`None`）ときも縮めません。
2. 収まらなければ、偶数の幅 w を元の幅から 2 ずつ下げ、高さを `h = 2 × int(w × 元の高さ ÷ 元の幅 ÷ 2 + 0.5)` で決め、キャンバス `(ceil64(w), ceil64(h))` が初めて予算に収まった (w, h) を作業寸法にします。
3. 余白は右に `キャンバス幅 − w`、下に `キャンバス高さ − h` です（どちらも 64 未満）。

**行の選び方**は既存の快適上限の規則と同じで、ベースモデルの行を上から見て、`requires` の鍵がすべて一致した最初の行を採ります（[`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) §1.1）。**例外として、重みの種別が判らない、または一致する行が無いときは、各行の値のうち最小のものを使います**（安全側に倒すため）。

**例**（4bit・B＝23,460）:

| 元の寸法×フレーム数 | 作業寸法 | キャンバス | キャンバスのトークン |
|---|---|---|---:|
| 1920×1088×81f | 縮めない（1920×1088） | 1920×1088 | 22,440 |
| 1280×768×121f | 縮めない（1280×768） | 1280×768 | 15,360 |
| 1920×1088×145f | 1468×832 | 1472×832 | 22,724 |

**例**（Q6_K・B＝21,672）: 1920×1088×81f → 作業 1808×1024・キャンバス 1856×1024（20,416 トークン）。

**公式の上限 1920×1088 を超える素材でも止めません。** 予算に収まるまで縮めて生成し、元の寸法へ引き伸ばして返します（精度が下がる旨の注意書きは第 2 弾の画面に出します。§9）。

### 3.2 参照の作成

窓を作業寸法へ縮め、右と下を**黒**で埋めてキャンバスの大きさにした動画を、参照 `alpha_reference.mp4` として作ります。内部の要求の `reference_video_id` はアップロードの ID のままで、実行時に参照動画のパスだけをこれに差し替えます。既存の `services/video_io.py` の `pad_green_mp4` に `scale`（縮める寸法）と `color`（余白の色）の引数を足して使います（新しい関数は作りません）。

- フィルタの形: `fps=…,scale=作業幅:作業高さ:flags=bicubic,format=rgb24,pad=キャンバス幅:キャンバス高さ:0:0:color=black,tpad=stop=-1:stop_mode=clone`
- 符号化は無劣化（`libx264rgb -crf 0` または FFV1）・音声なしです。

### 3.3 生成

- **1 段原寸モード**: ワーカーへの要求に `alpha_gen: {"mode": "one_stage"}` が載り、ワーカーは `engine25/alphagen25.py` の `run_alpha_gen` へ振り分けます（§5）。Stage-1 をキャンバスの原寸・参照つき（縮小率 1）で走らせ、その潜在をそのままデコードします。
- **軽量モード**: ワーカーへの要求に `alpha_gen` の鍵は載らず、既存の 2 段経路（IC-LoRA つきの普通の生成）がそのまま走ります。Stage-1 はキャンバスの半分の寸法です。

### 3.4 後処理（アプリ側）

エンジンが書いた `output.mp4` を `_alpha_engine.mp4` に改名し、`services/video_io.py` の `finalize_alpha_matte` が **1 回の ffmpeg** で次を行います。

1. 余白を切り落とす（`crop=作業幅:作業高さ:0:0`）。
2. 元の寸法へ引き伸ばす（bicubic）。
3. 2 本に分けて書く:
   - `matte.mkv`: 制限範囲（明るさ 16〜235）から全範囲（0〜255）へ伸ばし、gray にして FFV1（`-level 3`）で書く。
   - `output.mp4`: yuv420p の H.264（`-crf 18`・`+faststart`）で書く。

書き終えたら `_alpha_engine.mp4` は削除します（無劣化で大きいため）。`_alpha_window.mp4` と `alpha_reference.mp4` は記録として残します（[`STORAGE_POLICY.md`](STORAGE_POLICY.md) §1）。

### 3.5 記録

`metadata.json` に `alpha_gen` ブロックが加わります（鍵は §4.5）。配信の寸法（`output.resolution`）は元の寸法です。

## 4. API 契約

型・既定・エラーコードの正本は仕様書 [`Videomni_Backend_Specification.md`](../Videomni_Backend_Specification.md) §6 です。ここでは設計としての形を説明します。

### 4.1 `POST /api/v1/generate/alpha`

`POST /generate/chain` と同じく、凍結契約への加算として置く専用のエンドポイントです。認証は `/generate` と同じです。成功すると 202 でジョブの ID を返します。

本文 `AlphaGenRequest`:

| フィールド | 型 | 既定 | 制約 |
|---|---|---|---|
| `reference_video_id` | str | （必須） | 1 文字以上。`POST /upload/video` が返す ID |
| `window_start_sec` | float | `0.0` | 0 以上 |
| `num_frames` | int | （必須） | 9〜145・8n+1 |
| `frame_rate` | int | （必須） | 1〜60（整数） |
| `seed` | int | `-1` | — |
| `light_mode` | bool | `false` | 真なら軽量モード |

**素通しの項目**: `GenerateRequest` と同じ名前・同じ型・同じ既定の高速化・常駐の項目（`attention_backend`・`block_swap_prefetch`・`keep_resident`・`keep_resident_embeddings`・`fused_gguf_dequant_kernel`）を受け、内部の `GenerateRequest` にそのまま写します。LTX 2.5 では常駐の鍵が無いこと自体が「常駐を解放してよい」という指示なので、これを受けないと Alpha Gen のジョブのたびに利用者の常駐が外れてしまうためです。

サーバーは本文から次の内部の `GenerateRequest` を組み立てます: `prompt=" "`・`loras=[alpha-gen 強さ 1.0]`・`reference_video_id`（アップロードの ID のまま。実行時にパスだけ `alpha_reference.mp4` へ差し替える）・`width`/`height`＝キャンバス・`num_frames`・`frame_rate`・`seed`・素通しの項目・`alpha_gen=AlphaGenSpec`。

### 4.2 内部ブロック `GenerateRequest.alpha_gen`

`AlphaGenSpec` は、**サーバーだけが埋める内部の値**です（`inpaint` と同じ位置に置く加算のフィールド）。

| フィールド | 型 | 既定 | 意味 |
|---|---|---|---|
| `window_start_sec` | float | `0.0` | 窓の開始秒 |
| `one_stage` | bool | `true` | 真＝1 段原寸モード、偽＝軽量モード |
| `working_width` / `working_height` | int | （必須・2 以上） | 作業寸法 |
| `budget_tokens` | int \| null | `null` | 作業寸法を決めるのに使った予算 |

- **ファイルのパスと元の寸法は持たせません**（第 2 部 §13）。
- 形の検査（`validate_ltx_constraints`）は 2 つです: `reference_video_id` が必須／`width`・`height` が作業寸法を 64 の倍数へ切り上げた値と一致すること。
- **`POST /generate` にこのブロックが付いてきたら 422 `ALPHA_GEN_INVALID`** で断ります（第 2 部 §19）。検査の位置はエンジンの対応範囲の判定（`reject_unsupported`）の直後です。

### 4.3 検査の順と応答

`api/generate.py` の規則「エンジンの対応範囲を最初に見る」に合わせ、次の順に判定します。

| 順 | 条件 | 応答 |
|---|---|---|
| a | LTX 2.3 の判定: 有効なベースモデルの `unsupported_features` に `alpha_gen` がある | 422 `FEATURE_UNSUPPORTED`。文言は「LTX 2.3 は Alpha Gen に対応していません。ベースモデルに「LTX 2.5」を選んでください。」 |
| b | 本文の形が誤り（8n+1 でない・145 超・小数の fps など） | 422 `VALIDATION_ERROR`（8n+1 の文言は "num_frames must be 8n+1 (9-145)"） |
| c | 参照と LoRA: 参照動画が無い／`alpha-gen` が未登録・ファイルが無い | 404 `REFERENCE_VIDEO_NOT_FOUND`／`LORA_NOT_FOUND`。`alpha-gen` の登録が control・前処理なしでなければ 422 `ALPHA_GEN_INVALID`（§7） |
| d | 元寸: 元の動画が測れない・辺が奇数・辺が 256 画素未満 | 422 `ALPHA_GEN_INVALID` |
| e | 窓: 窓が素材に収まらない | 422 `ALPHA_GEN_INVALID`（Inpainting と同じ書式の説明） |
| f | 寸法の範囲: 作業寸法かキャンバスが `GenerateRequest` の範囲（幅 256〜4096・高さ 128〜4096）を外れる | 422 `ALPHA_GEN_INVALID` |
| g | 内部要求: 組み立てた内部の `GenerateRequest` が検証に落ちる | 422 `ALPHA_GEN_INVALID`（"internal request rejected: …"） |
| h | 409: モデルの読み込み中／別のジョブの実行中 | 既存の 409（`PIPELINE_LOADING`／`JOB_BUSY`） |
| i | `/generate` 側の拒否: `POST /generate` に `alpha_gen` が付いてきた | 422 `ALPHA_GEN_INVALID`（"alpha_gen is set by POST /generate/alpha only"） |

- **b（本文の形の検査）は pydantic が行い、ハンドラの前に走ります。** そのため LTX 2.3 でも、形が崩れた本文には a より先に `VALIDATION_ERROR` が返ります（`/generate` と同じ）。表の a が「最初」なのは、ハンドラの中の検査の順としてです。
- i は `POST /generate/alpha` ではなく `POST /generate` の側の検査です（位置は §4.2）。LTX 2.3 では先に `FEATURE_UNSUPPORTED` が返ります。
- 辺が偶数であることを求めるのは、出力の yuv420p（色を 2×2 画素ごとに持つ形式）の都合です。256 画素の下限は Inpainting の `INPAINT_MIN_SOURCE_SIDE` を流用しています。
- f を受付で断るのは、内部の `GenerateRequest` の検証に落ちると 500 になってしまうためです。g はその残りを 422 に包む受け皿です。
- 新しいエラーコードは `ALPHA_GEN_INVALID` の 1 つだけです。何が悪いかは `detail` が名指しします。

### 4.4 マットの取得: `GET /api/v1/jobs/{job_id}/matte`

`GET /jobs/{id}/video` と同じ形で `matte.mkv` を返します（`video/x-matroska`・ファイル名 `{job_id}_matte.mkv`）。ジョブが完了していなければ 409 `VIDEO_NOT_READY`、マットが無ければ 404 です。

ジョブの応答 `JobResponse` には **`matte: bool`**（既定 `false`）が加わります。`matte.mkv` が今あるかどうかを表し、`joined` と同じ形です。

### 4.5 `metadata.json` の `alpha_gen` ブロック

Alpha Gen のジョブにだけ現れます。

| 鍵 | 意味 |
|---|---|
| `mode` | `"one_stage"` または `"light"` |
| `source_video_id` | 入力動画の ID |
| `source_width` / `source_height` | 元の寸法（切り出した窓を測った値） |
| `source_fps` | 素材の fps |
| `resampled` | fps を変換したか |
| `window_start_sec` / `window_start_frame` | 窓の開始 |
| `window_written_frames` | 窓に書いたフレーム数 |
| `working_width` / `working_height` | 作業寸法 |
| `canvas_width` / `canvas_height` | キャンバス |
| `pad_right` / `pad_bottom` | 余白 |
| `scale` | 縮めた倍率 |
| `budget_tokens` | 使った予算 |
| `canvas_tokens` | キャンバスのトークン数 |
| `matte` | `"matte.mkv"` |
| `matte_codec` | `"ffv1/gray"` |

**`generation_mode` は既存の規則のまま `"t2v"` になります**（画像の条件が無いため）。Alpha Gen のジョブかどうか・どちらのモードかは、`metadata.alpha_gen.mode` を正としてください。`generation_mode` に新しい値は足しません。

## 5. エンジン（`engine25/alphagen25.py`）

1 段原寸モードは、新しいファイル `engine25/alphagen25.py` の `run_alpha_gen` が担います。手本は `engine25/inpaint25.py` の `run_inpaint` で、公式の `DistilledPipeline.__call__` を使わず、部品を直接呼びます。参照が無ければ `AlphaGenError` を出します（参照の無いマットは意味が無いため）。

手順:

1. `validate_geometry`（64 の倍数の規則のまま）。
2. `resolve_reference_downscale_factor`（`alpha-gen` のヘッダは縮小率 1）。
3. `stage.set_loras` → `begin_job` → 進捗の通知 → `vram.phases.clear()`。
4. `torch.no_grad()` の中で:
   - プロンプトを符号化する（`pipeline.prompt_encoder`）。
   - 原寸の形に合わせてタイルの設定を決める（`ensure_tiling_config`）。
   - **参照を原寸で**符号化する。`engine25/outpaint25.py` の `_encode_reference_conditionings` に、半分にしない引数とフレームが 0 枚なら例外にする引数を渡して共用します（写しは作りません）。
   - `stage.announce(STAGE_1_DENOISE, vram_phase="21_stage1_denoise")` のあと、Stage-1 を原寸・参照つきで 1 回走らせる。サンプラーは本製品の LTX 2.5 の Stage-1 と同じ ancestral（`chain25._stage1_sampler_kwargs`）。音声の文脈は公式どおり渡します。
   - 潜在を取り、メモリを片づけて `vram.reset()`。
   - デコードして `encode_video(..., audio=None, crf=0)` で書く（音声はデコードしません）。
5. `vram.record("40_job_end")` → `GenerationResult` を返す。

ワーカー（`engine25/worker.py`）は、要求に `inpaint`・`outpaint`・`alpha_gen` のどれがあるかで振り分けます（2 つ以上が同時にあれば assert で止めます）。完了の通知は素の生成と同じ形です。

**記録される段**（`metadata.json` の `ltx25.phases`）: `10_*`（文章理解部品が自分で記録）・`11_reference_encode`・`21_stage1_denoise`・`30_decode_encode`・`40_job_end`。**`22_*`（Stage-2）が無いことが、1 段で走った証拠です。**

**公式との差は 2 つです。**

| 点 | 公式 | 本製品 |
|---|---|---|
| Stage-2 を飛ばすときの寸法 | `skip_stage_2` は Stage-1 を**半分**で走らせ、半分のままデコードする | Stage-1 を**原寸**で走らせる（公式の 1 段を 2 倍の寸法で呼ぶのと同じ。第 2 部 §12） |
| サンプラー | `ICLoraPipeline` は Euler | ancestral（第 2 部 §15） |

## 6. 快適上限

- 快適上限の行（`ComfortRow`）に **`alpha_gen_budget`** を足します。正本はマニフェスト `scripts/manifests/20-ltx25.json` の `comfort.rows` で、`GET /config` の `limits.comfort_budgets` で配信されます。
- **値は暫定です**: 4bit 23,460・Q6_K 21,672・8bit 19,380（種別ごとの単発の線のちょうど半分）。LTX 2.3 の行は `None`（線なし）です。
- **既存の方針との関係**: 快適上限は「較正と裁定の済んでいない線は `None`＝線なし」が方針です（[`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) §9.4）。Alpha Gen は較正前ですが、**オーナー裁定で暫定値を置きます**（予算が無いと作業寸法を決められないため）。コードと文書の説明にも「暫定・オーナー裁定」と書きます。
- 軽量モードは既存の `single_budget` を使います。
- 較正は台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §3-223 で行います。数値の正本は [`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) §9.5 です。

## 7. `alpha-gen` の扱い

- **登録の場所**: `config.yaml` の `model.ic_loras` に文字列形式で 1 行（`models/LTX25/IC-LoRA/alpha-gen/` の LoRA ファイル）。参照を前処理しないので文字列形式です。配布用のひな型 `config.yaml.example` にも同じ行と「操作パネルと Gradio の一覧には出ない」注記を置きます。
- **操作パネルと Gradio の LoRA の一覧には出しません。** 操作パネルは画面側の隠し名簿 `UI_HIDDEN_CONTROL_LORA_NAMES`（`webui/src/lora/controlLoras.ts`。第 2 弾で 1 語足す）、Gradio は `gradio_ui/adapters.py` の `HIDDEN_ADAPTER_NAMES` で除きます。
- **`alpha-gen` は control・前処理なし（`preprocess` が `none`）で登録されている必要があります。** 違えば `POST /generate/alpha` は 422 で断ります（§4.3 の c）。
- **サーバー側に予約名の仕組みはありません。** `GET /loras` と `GET /config` の `ic_loras` には `alpha-gen` が出ます（第 2 部 §20）。
- **MCP には出ます**（`list_loras`）。名前が見えるだけで害はありません。
- LoRA の一覧はベースモデルに共通なので、LTX 2.3 を選んでいても名前は出ます。Alpha Gen を LTX 2.3 で頼むと §4.3 の a で断ります。

## 8. MCP

ツールは 2 つ増えて 25 本になります（`submit_alpha_gen`・`get_job_matte_path`）。`save_job_video` は新しいツールではなく、`which` に `"matte"` を加えた拡張です。

| ツール | 中身 |
|---|---|
| `submit_alpha_gen` | `POST /generate/alpha` を送る。引数は `reference_video_id`・`num_frames`・`frame_rate`・`window_start_sec`（既定 0.0）・`seed`（既定 -1）・`light_mode`（既定 false）と素通しの項目。ファイルのパスの引数は付けない（既存の submit 系と同じく ID だけ） |
| `get_job_matte_path` | ジョブの `matte.mkv` のパスを返す |
| `save_job_video(which="matte")` | 既存のツールの `which` に `"matte"` を足す（`"output"`・`"joined"`・`"matte"`） |

流れは `upload_video` → `submit_alpha_gen` → `wait_for_job` → `get_job_matte_path`（または `save_job_video(which="matte")`）です。

## 9. 第 2 弾（操作パネル）の決定事項

第 2 弾は台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §1-85 です。次の事項は決まっています。

### 9.1 置き場所と入力

- **Edit タブに「Alpha Gen」サブタブ**を置きます。
- **入力欄はドラッグ＆ドロップで受けます**（ブリッジ契約 v7 の `ui.resolveDroppedFiles` が前例）。タイムラインのオブジェクトの右クリックからも送れます。
- **1 対 1 でないオブジェクトは断ります。** タイムラインのフレームと素材のフレームが 1 対 1 に対応しない動画（再生速度が 100％でない・中間点がある・ループする）は、`guardMenuSelection` で次の警告文を出して受け付けません。
  > 再生速度100％・中間点なし・ループなしの動画のみに対応しています
- **アップロードは生成ボタンを押した時点**で行います。リボンの冒頭からスライダーの分だけを送ります。入力がスライダーより短ければフレーム数を自動で合わせ、トースト（画面の隅に数秒出る通知）で「フレーム数が自動調整されました」と知らせます。

### 9.2 fps とフレーム数

- **fps 欄は自動で埋め、編集できません。** 右クリックで送った動画はプロジェクトの fps、ドラッグ＆ドロップした動画はファイルの fps を入れます。
- **フレーム数はスライダー**です。8n+1 刻み・最大 145・初期値 81。説明に「推奨：121フレーム以下」と書き、秒数も表示します。
- **快適上限マーカー**（そのあたりから生成が重くなるという目安の線。§6 の予算から引く）は解像度に応じて動きますが、**スライダーの位置は動かしません**（助言だけで、値を勝手に変えない）。

### 9.3 注意書き（2 つ）

- 「1920*1088を超えるサイズでは、背景透過の精度が下がります」
- 「1280*768未満の小さな動画では、背景除去の精度が悪化します。」——**軽量モードは無効化しません**（注意書きを出すだけ）。

### 9.4 後処理カード

完成したジョブには次の 4 つの部品を持つカードを出します。

| 部品 | 働き |
|---|---|
| `🎞` | アルファつきの動画をタイムラインへ置く。元の音声を含める |
| `💾` | ProRes 4444（アルファを持てる動画形式）を作り、WebView2 の既定のダウンロードで保存する |
| `🪣` | クロマキー用に、背景を一色で塗った動画を、Join と同じ形の切り替えで表示する |
| `色` | `🪣` の塗り色のドロップダウン |

`色` の選択肢は次の 8 色で、既定は green です。名前は AviUtl2 の `data\Default\default.palette` のプリセット名と一致させます。

| 名前 | 値 |
|---|---|
| white | #FFFFFF |
| red | #FF0000 |
| yellow | #FFFF00 |
| green（既定） | #00FF00 |
| aqua | #00FFFF |
| blue | #0000FF |
| magenta | #FF00FF |
| black | #000000 |

### 9.5 画面の裏側

- **画面側の隠し名簿**: `UI_HIDDEN_CONTROL_LORA_NAMES` に `alpha-gen` を足します（§7）。
- **模擬サーバーの写し**: 操作パネルの模擬サーバー（`webui/src/bridge/mockBridge.ts`）の `MOCK_UNSUPPORTED_FEATURES` は本物の `unsupported_features` の写しなので、LTX 2.3 の側に `alpha_gen` を足します。
- **進捗バーの対応表**: 単発生成の対応表では Stage-1 が 0.06〜0.50 に当たるため、1 段原寸モードはデコードのあいだ 50％で止まって見えます（`services/engines/ltx/adapter.py`）。対応表をどうするかは第 2 弾で判断します。
- **高速化の鍵の送り方**: 他のジョブと同じく、Settings の高速化・常駐の設定を `POST /generate/alpha` の素通しの項目として送ります（§4.1）。

## 10. 第 3 弾（出口）の決定事項

第 3 弾は台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §1-86 です。次の事項は決まっています。

- **`🎞` と `💾` は、ProRes 4444 の MOV 1 本を共用します。** 合成の元は窓 `_alpha_window.mp4`（音声つき）と `matte.mkv` です。
- **挿入の経路にある拡張子 `.mp4` の決め打ち 3 箇所を手当てします**（場所は調査ノート [`LTX_ALPHAGEN_RESEARCH_NOTES.md`](LTX_ALPHAGEN_RESEARCH_NOTES.md) 3.2 節の「受け取る側」）。
- **最初の関門は実機の確認です。** 開発機の入力プラグイン L-SMASH Works（r1281）で、設定 `colorspace=0` のまま ProRes 4444 の MOV が透過つきで読めるかを、AviUtl2 の実機で確かめます。
- **無圧縮の RGBA の AVI は既定にしません**（大きすぎるため）。
- **自前の入力プラグイン**（エンコードせずに直接 AviUtl2 へ渡す方式）は台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §3-222 の研究課題です（第 2 部 §21）。

## 11. 第 1 弾に入れないもの

- 操作パネルの画面（第 2 弾。§9）。模擬サーバーの写しの更新・画面側の隠し名簿・進捗バーの対応表も第 2 弾です。
- 透過つきの動画・クロマキーの動画の作成とタイムラインへの挿入（第 3 弾。§10）。
- Gradio 画面（対象外。一覧から `alpha-gen` を除くだけ）。
- LoRA ウェイトの配布（第 3 弾の完成後に検討。台帳 §1-87）。
- 快適上限の較正（台帳 §3-223。第 1 弾は暫定値）。
- LTX 2.3 への対応（LoRA が LTX 2.5 用のため）。
- 回転情報つきの素材（スマートフォンの縦動画など）: 窓と元寸の縦横が入れ替わっていたら失敗にします（§3 の 2 番）。回転を考えた寸法の測り方は第 2 弾以降です。
- マットの品質の判定（第 1 弾の実機ゲートでは、位置ずれが無いことまでを見ます）。

---

# 第2部 なぜそう決めたか

## 12. 1 段原寸を自前で作った理由（公式の skip は半分の寸法）

公式の `skip_stage_2` は「Stage-1 を半分の寸法で走らせ、半分のままデコードする」ものです。原寸のマットは出ません。本製品の 1 段原寸モードは、**公式の 1 段を 2 倍の寸法で呼ぶのと同じ**ことを自前で行っています。負荷はゲート 0 の並び B（寸法を 2 倍で要求して Stage-1 を原寸で走らせた点）と同じで、16GB で 1280×768×185f・1920×1088×81f まで VRAM 溢れ（GPU の専用メモリに収まらず、遅いメインメモリ側へはみ出すこと）なしで通ることを確かめてあります（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §151.3）。

既存の `pipeline25.generate` は公式の `DistilledPipeline.__call__` を呼ぶだけで、Stage-1 は半分に固定されています。また、本製品が参照を差し込む仕組みは「Stage-1 かどうか」を寸法で判定するので、そのままでは 1 段にできません（§151.7 の 1）。そのため、Inpainting と同じく部品を直接呼ぶ新しいドライバを作りました。

## 13. ファイルのパスと元の寸法を要求に入れない理由

Inpainting の前例と同じです。**元のファイルが寸法の正本**で、要求にも寸法を書くと同じ事実の置き場が 2 つになり、食い違いを生みます。また、要求のフィールドからサーバーに任意のパスを読ませないためです。サーバーはアップロード済みの ID から窓を切り出し、その窓を自分で測ります。

## 14. キャンバスを 64 の倍数で足りるとした理由

参照つきの生成は、これまで 128 の倍数を求めてきました（`REFERENCE_RESOLUTION_INVALID`）。これは**縮小率 2 の IC-LoRA のための規則**です（参照を Stage-1 のさらに半分で読むため）。`alpha-gen` の縮小率は 1 なので、64 の倍数で足ります。64 なら `validate_geometry` も通ります。`POST /generate/alpha` は `/generate` の 128 の検査を通らないので、この規則は掛かりません。128 に揃えると余白が最大 127 画素になり、無駄なトークンが増えます。

## 15. サンプラーを ancestral にした理由

公式の `ICLoraPipeline` は Euler ですが、本製品の LTX 2.5 の Stage-1 は ancestral です。**ゲート 0 で出た良好なマットは、この設定で出ています。** 1 段原寸モードも同じ Stage-1 の作りにそろえました。マットの品質が悪ければ、最初に疑う点として記録してあります。

## 16. 予算を単発の線の半分にした理由

1 段原寸の負荷は「生成側＋参照側の合計トークン」で効く、という見立てがゲート 0 の実測と矛盾しませんでした（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §151.5 の H3）。縮小率 1 の参照は生成と同じ大きさなので、合計は生成側の 2 倍です。そこで**予算をキャンバス（生成側）に当て、値を単発の線の半分にしました**。こうすれば合計が単発の線に収まります。種別ごとの線（4bit・Q6_K・8bit）の半分を当てていますが、Q6_K と 8bit は測っていないので暫定です（台帳 §3-223）。

## 17. 余白を黒にした理由

マットでは黒が「透明＝背景」を意味します。余白は最後に切り落とす「何も写っていない場所」なので、参照の上でも背景と同じ黒にそろえました（余白が縁のマットに影響しにくい、という見込みです。実機ゲートで位置ずれと合わせて確かめます）。Inpainting・画角拡張の余白が緑なのは、描き足す場所の目印としてモデルに読ませるためで、目的が違います。

## 18. 「劣化しない」の意味（全範囲と制限範囲の量子化）

公式の `encode_video` は、BT.709 の制限範囲 yuv420p（明るさ 16〜235）に変換してから書きます。したがって **crf 0（無劣化）で書いても、明るさは 220 段階に量子化されています。** 後処理で全範囲（0〜255）へ伸ばすと、使われない値が飛び飛びに出ます。本書の「劣化しない」は「crf 19 の非可逆圧縮を避ける」という意味で、256 段階が全部使われるという意味ではありません。なお軽量モードの出力は既存の経路のまま crf 19 です。

## 19. `/generate` で `alpha_gen` を拒む理由

`alpha_gen` は、サーバーが自分で測った元の寸法から作業寸法を決めて埋める値です。`/generate` で外から受け付けると、測っていない寸法を信じることになり、余白の切り落としや引き伸ばしが食い違います。そのため受け付けずに断ります。判定はエンジンの対応範囲の判定の直後に置き、LTX 2.3 では先に `FEATURE_UNSUPPORTED` が返るようにしています。

## 20. `alpha-gen` を画面側で隠す理由

サーバーの `GET /loras` から除いても、`GET /config` の `ic_loras` には名前が残ります。操作パネルは `/loras` が読めない間 `/config` の `ic_loras` に戻るので、そこで名前が出てしまいます。前例の `in-outpainting` も画面側の隠し名簿で隠しているので、同じ仕組みにそろえました。サーバー側に予約名の仕組みを新しく作ると、前例と別の仕組みが 2 つ並ぶことになります。MCP の `list_loras` には名前が出ますが、見えるだけで害はありません。

## 21. 自前の入力プラグインを初版で採らなかった理由

AviUtl2 に透過つきの動画をエンコードせずに直接渡すには、自前の入力プラグインを作る方法があります（`register_input_plugin` は aux2 から呼べ、RGBA32・PA64・HF64 を受けられます）。しかし初版では、既にある L-SMASH Works で ProRes 4444 の MOV を読む方法を先に試します。新しいプラグインを作るより変更が小さく、本製品がタイムラインに動画を置く今の経路（動画ファイルのオブジェクト）をそのまま使えるためです。L-SMASH で透過が出ない場合や、エンコードの時間が問題になる場合に、台帳 §3-222 の研究課題として取り上げます。

## 22. 軽量モードを残す理由

ゲート 0 では、1 段原寸相当の 1920×1088×121f は VRAM 溢れで実用外でした。一方、今の 2 段経路のままでも、1920×1024×145f（Stage-1 は 960×512）は全体 218 秒で完走し、良好なマットが出ました（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §151.5）。大きな素材や長い窓を、縮めずに速く処理したいときの道として軽量モードを残します。ただし、2 段経路で Stage-1 が 640×384 未満になる小さな素材ではマットが出ないことも分かっています（同 §151.4）。止めはせず、第 2 弾の画面で注意書きを出します。

---

## 参照

- 事前調査: [`LTX_ALPHAGEN_RESEARCH_NOTES.md`](LTX_ALPHAGEN_RESEARCH_NOTES.md)
- ゲート 0: [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §151
- API 契約: [`Videomni_Backend_Specification.md`](../Videomni_Backend_Specification.md) §6（Alpha Gen は §6.3b）
- 快適上限: [`COMFORT_LIMIT_TABLE.md`](COMFORT_LIMIT_TABLE.md) §9.5
- 台帳: [`PENDING_TASKS.md`](PENDING_TASKS.md) §1-83・§1-85〜§1-87・§3-222・§3-223
