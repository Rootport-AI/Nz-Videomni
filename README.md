# LTX-AviUtl2-Bridge

LTX 2.3 動画生成モデルを **VRAM 16GB** のコンシューマーGPUで動かし、REST API として公開するバックエンドサーバー。検証用に Gradio UI(`/ui`) を同梱します。
最終的には AviUtl2 / DaVinci Resolve などの薄いフロントエンドから利用しますが、API は汎用設計です。

**Phase 1**（T2V + 最小I2V を同一MVP）の凍結 API を土台に、その後キーフレーム誘導・クリップ連結（`POST /generate/chain`）・
V2V（元動画からの継続生成）・A2V（音声から動画生成）・IC-LoRA／スタイルLoRA などを加算的に拡張しています。
API 契約・スキーマの詳細仕様は [`LTX23_Backend_Specification.md`](LTX23_Backend_Specification.md) を参照してください。

> **現状（2026-07-01）**: 実エンジンは **first-party の `engine/` パッケージ**（GGUF 量子化トランスフォーマー + block-swap +
> GGUF Gemma 逐次オフロード + DiT CPU 構築 + VAE タイリング）で、**RTX 4070 Ti SUPER 16GB 実機で 720p 級（1280×768→クロップ）
> の T2V/最小I2V 生成に成功**しています（`Docs/VERIFICATION_LOG.md`）。GPU / モデルウェイトが無い環境では自動的に **モック
> backend**（合成クリップ）へフォールバックし、API・ジョブ管理・Gradio・テストまで完全に疎通します。
>
> 公式 `ltx_pipelines` の safetensors ローダは本機(16GB/Windows)で native crash するため**不採用**で、GGUF + component-file
> 経路にしています。詳しい設計判断・実測は `Docs/VERIFICATION_LOG.md` と `engine/VENDOR_NOTICE.md` が一次情報です。

---

## 0. 環境分離ポリシー（最重要・最初に読む）

**このプロジェクトは PC のシステム Python 環境を一切汚しません。** Python 本体を含め、必要なものはすべて
プロジェクトディレクトリ配下（`.venv/`, `.venv-engine/`, `.python/`）に閉じ込めます（仕様書 2.5）。

- グローバル/システムの `pip install` は **禁止**。必ず `uv` + プロジェクトローカル venv。
- 環境変数（`PYTORCH_CUDA_ALLOC_CONF`, `UV_PYTHON_INSTALL_DIR`）は **そのプロセス内のみ**。永続化しない。
- 後片付けはこのディレクトリ（`.venv` / `.venv-engine` / `.python` 含む）を削除するだけで完全に元に戻ります。

### 2つの venv（重要）

このバックエンドは **2プロセス・2venv 構成**です。両者は別インタプリタで、共存させません。

| venv | 役割 | 主要依存 |
|------|------|----------|
| `./.venv` | FastAPI アプリ（`main.py`・API・ジョブ・Gradio・モック backend） | FastAPI / Pydantic / Pillow / ffmpeg 呼び出し。**torch は入れない** |
| `./.venv-engine` | 実エンジン worker（`engine/worker.py`） | **torch 2.9.1+cu128** + LTX 推論スタック（`ltx_core`/`ltx_pipelines`@`00dc53d` + `gguf`） |

アプリ(`./.venv`)は torch も LTX も import しません。実生成は `./.venv-engine` の python で
`python -m engine.worker` を **subprocess** として起動し、JSON-lines プロトコルで駆動します（下記アーキテクチャ参照）。
`.venv-engine` の依存スナップショットは [`engine/venv-engine.freeze.txt`](engine/venv-engine.freeze.txt) に凍結してあります
（`uv.lock` は fork 撤去時に失われたため）。

---

## 1. セットアップ

### 必要なもの
- Windows 10/11 x64
- [`uv`](https://docs.astral.sh/uv/)（パッケージ/Python管理）
- `ffmpeg`（PATH に通すこと。MP4エンコード・クロップに使用）
- 実モデル実行時のみ: NVIDIA CUDA GPU（VRAM 16GB 推奨・**CUDA 12.8/cu128 対応ドライバ**）

### アプリ venv（`./.venv`, torch 無し）

```powershell
# Python 本体もプロジェクト内に固定する（システムを汚さない）
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"

uv python install 3.12
uv venv --python 3.12 .venv
uv sync --extra dev
```

これで `.venv/`（アプリ仮想環境）と `.python/`（uv管理のPython 3.12）がプロジェクト内に作成されます。
`pyproject.toml` / [`requirements.txt`](requirements.txt) には FastAPI 側の依存のみ定義しています（`torch` は含みません）。
これだけで **モック backend** で API/UI/テストが動きます（GPU 不要）。

### エンジン venv（`./.venv-engine`, torch+cu128）と実モデル

実生成には別途 `./.venv-engine`（torch 2.9.1+cu128 + LTX 推論スタック）と GGUF/component モデル群が必要です。
`.venv-engine` は fork 撤去時に `./.venv-engine` へ退避済みで、依存は
[`engine/engine-venv-pyproject.toml`](engine/engine-venv-pyproject.toml)（`[tool.uv.sources]` に torch cu128 index と
`ltx-core`/`ltx-pipelines`/`diffusers` の git rev を記載）と
[`engine/venv-engine.freeze.txt`](engine/venv-engine.freeze.txt)（`name==version` の完全スナップショット）から再構築できます。
provenance と再現手順の詳細は [`engine/VENDOR_NOTICE.md`](engine/VENDOR_NOTICE.md) を参照してください。

必要なモデル（`config.yaml` の `model:` が参照。相対パスは PROJECT_ROOT 基準で絶対化）:

| 要素 | 既定パス | 概算 | 役割 |
|------|----------|------|------|
| GGUF transformer (Q4_K_M) | `models/ltx-2.3-gguf/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf` | ~17GB | 本番トランスフォーマー |
| GGUF Gemma (Q4_K_M) | `models/gemma-3-12b-it-gguf/gemma-3-12b-it-Q4_K_M.gguf` | ~7.3GB | text encoder（GPU 推論・逐次オフロード） |
| component VAE / audio / text-projection | `models/ltx-2.3-components/{vae,text_encoders}/*.safetensors` | ~3.9GB | 46GB モノリスを置換する小単体ファイル |
| spatial upsampler | `models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | ~0.95GB | 2段生成の x2 アップサンプラ |
| Gemma tokenizer dir (`gemma_root`) | `models/gemma-3-12b-it-tokenizer/` | ~40MB | tokenizer/preprocessor のみ（`tokenizer.model` 等）。**重みは含まない**（text encoder は上の GGUF Gemma が供給） |

> **削除済み（2026-07-01 の refactor）**: (1) 43GB モノリス `ltx-2.3-22b-distilled-1.1.safetensors` を物理削除。
> `config.model.checkpoint_path` はフィールドとしては残りますが **reference-only**（worker payload に載るが GGUF+component
> 経路では一切開かれない・rename test で実証済み）。(2) **22.7GB の QAT Gemma dir `models/gemma-3-12b-it-qat/` も物理削除**。
> Gemma を **text-only（`Gemma3ForCausalLM`・vision 無し）** で構築するよう作り替えたため（`engine/gemma/text_encoder_configurator.py`）、
> wheel が build 時に重みシャードを glob する必要が無くなり、`gemma_root` は上記 ~40MB の tokenizer-only dir で足ります。full-QAT
> baseline と出力バイト一致で検証済み（`Docs/VERIFICATION_LOG.md` §14）。
>
> 実行に本当に要るモデルは合計 **~28GB**（GGUF transformer + GGUF Gemma + components + upscaler + tokenizer dir。実測 `models/` 全体
> 28.15GB）で、ComfyUI の GGUF 16GB レシピと同等のフットプリントです。

backend の選択は `config.model.backend`（`auto`/`mock`/`real`）で行います。既定 `auto` は「`./.venv-engine` の python・
`engine/worker.py`・上記ロード対象ファイルが全て存在」すれば **real**、無ければ **mock** です
（[`services/ltx_runner.py`](services/ltx_runner.py) `_real_available`）。

### 追加の transformer GGUF / LoRA を配置する

**transformer GGUF**: `models/ltx-2.3-gguf/` **直下**に `.gguf` を置くだけで、ファイル名から自動認識され
UI/API のドロップダウンに列挙されます。サブフォルダに入れても再帰スキャンで拾われます（[`services/model_registry.py`](services/model_registry.py) の
`CATEGORY_SPECS["transformer"]`、`recursive=True`）。登録名はファイル名（拡張子除く）で、既定の登録名と
衝突する場合は親フォルダ名が `親フォルダ名__ファイル名` の形で前置されます。`config.yaml` の編集は不要です
（`model.transformers` への明示登録は、スキャンでは拾えないファイルを公開するための上書き用の代替手段です）。

> **既存インストールからの移行**: 従来の `install_ltx.ps1` は本番トランスフォーマー GGUF を
> `models/ltx-2.3-gguf/LTX-2.3-distilled-1.1/` というサブフォルダの中に配置していました。現在の公式配置は
> `models/ltx-2.3-gguf/` **直下**です（サブフォルダ配置自体は再帰スキャンで引き続き動作しますが、以後の
> 公式手順・ドキュメントの既定パスは直下を前提にします）。既にインストール済みの環境は、以下のいずれかで
> 揃えてください。
>
> - **PowerShell を再実行する**（推奨）: 最新の `scripts/install_ltx.ps1` は DL 後に自動でファイルを1階層
>   上へ移動し、空になったサブフォルダを削除します（既に直下にある場合はスキップされます）。
> - **手動で移動する**: 以下のコマンドでサブフォルダ内の `.gguf` を直下へ移し、空フォルダを削除します。
>
>   ```powershell
>   Move-Item "models\ltx-2.3-gguf\LTX-2.3-distilled-1.1\*.gguf" "models\ltx-2.3-gguf\"
>   Remove-Item "models\ltx-2.3-gguf\LTX-2.3-distilled-1.1" -Force
>   ```
>
>   `config.yaml` に `model.gguf_transformer_path` を明示的に指定している場合は、直下のパス
>   （既定値 `./models/ltx-2.3-gguf/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`）に合わせて書き換えてください。
>   指定していない場合はコード側の既定値が既に直下パスを指すため、`config.yaml` の編集は不要です。

選択は UI の「Models」設定タブのドロップダウン、または API `GET /models`（登録名の一覧確認）→
`POST /pipeline/load`（body `{"models": {"transformer": "<登録名>"}}`）で行います。選択が現在ロード中のものと
異なる場合のみワーカーが再構築されます。

GGUF の要件: (1) KVメタデータに `config`（モデル設定のJSON文字列）が埋め込まれていること、(2) テンソル名が
LTXネイティブの生キーであること、(3) `embeddings_connector` 層が非量子化（F32/BF16）であること。これらを
満たさない外部配布 GGUF はロードに失敗します（条件を満たすのは QuantStack 製、および自家製変換ツール
`Nz-GGUF-Converter-LTX23` の出力）。量子化タイプは既定の Q4_K_M に加え Q6_K / Q8_0 等にも対応します。

**LoRA**: `models/loras/` に `.safetensors` を置くと自動認識されます（`GET /loras` で一覧確認、
`POST /loras/reload` で明示再スキャン）。生成時は API の `loras: [{"name": ..., "strength": ...}]`、または
Gradio UI のプロンプト内 `<lora:名前:強度>` 記法で適用します（強度は 0〜2）。ComfyUI 形式
（`diffusion_model.` プレフィックス＋ `lora_A`/`lora_B`）に対応し、量子化 GGUF モデルにもそのまま適用できます
（実行時加算方式のため、モデル側の量子化と衝突しません）。

---

## 2. 起動

```powershell
# 推奨: 起動スクリプト（環境変数設定 → アプリ venv → main.py 起動 を一括）
./run.ps1

# もしくは直接
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
.\.venv\Scripts\python.exe main.py
```

`run.ps1` は **アプリ**（`./.venv` の `main.py`）を起動します。real backend が選ばれると、アプリが
`./.venv-engine\Scripts\python.exe -m engine.worker` を subprocess として自動 spawn します（手動起動は不要）。

起動後:
- UI: <http://127.0.0.1:18620/ui>
- API docs (Swagger): <http://127.0.0.1:18620/docs>
- Status: <http://127.0.0.1:18620/api/v1/status>

### CLI オプション

| オプション | 説明 |
|-----------|------|
| `--listen` | `0.0.0.0` にバインド（家庭内LAN公開。インターネット公開は非対応） |
| `--port <n>` | ポート変更（デフォルト 18620） |
| `--api-key <key>` | `Authorization: Bearer <key>` を要求 |
| `--allow-all-cors` | CORS 全許可（既定は localhost のみ） |
| `--config <path>` | `config.yaml` のパス指定 |
| `--te-offload` / `--no-te-offload` | Gemma text-encode の逐次 per-layer CPU オフロード（既定 ON） |
| `--dit-cpu-load` / `--no-dit-cpu-load` | DiT(transformer) を CPU 構築しブロックを GPU へストリーム（既定 ON。無効化するとロード時 ~16.9GB GPU スパイクが復活） |

```powershell
./run.ps1 --listen --port 19000
```

---

## 3. アーキテクチャ

**2プロセス分離**（凍結 API 層 + 実エンジン worker）です。

```text
[Gradio /ui]──HTTP──┐
                     ▼
┌──────────────────────────────────────────────┐
│  FastAPI アプリ  (./.venv, torch 無し)          │
│   api/ (router, models, status, generate, ...) │
│   services/ (job_store, upload_store,          │
│              pipeline_manager, video_io,       │
│              low_vram, gpu_info)               │
│   services/ltx_runner.py  ── 唯一の LTX 接点    │
│        ├─ _MockBackend  (合成クリップ・GPU不要) │
│        └─ _RealBackend  ── subprocess.Popen ──┐ │
└───────────────────────────────────────────────┼─┘
                                                 │  JSON-lines (@@LTX@@ frames)
                                                 ▼  stdin/stdout
                    ┌──────────────────────────────────────────┐
                    │ engine worker  (./.venv-engine, torch+cu128)│
                    │   python -m engine.worker                  │
                    │   engine/pipeline/fast_video_pipeline.py   │
                    │   engine/gguf/  (GGUF dequant/loader)      │
                    │   engine/gemma/ (GGUF Gemma + 層オフロード) │
                    │   engine/transformer/ (block-swap, dit-cpu)│
                    │   → output.mp4 を共有 output dir に直接書く │
                    └──────────────────────────────────────────┘
```

- **`engine/` は first-party**（project root 直下・git 追跡）。旧・同梱フォーク `vendor/LTX-Desktop-LOW-VRAM` は
  refactor（Stage 2b）で完全削除済み。`engine/` の由来と provenance は [`engine/VENDOR_NOTICE.md`](engine/VENDOR_NOTICE.md)。
  上流参照用の `vendor/LTX-2` は温存しています。
- **プロトコル**: アプリは real backend でも torch/LTX を import しません。`_RealBackend` が
  `subprocess.Popen([engine_python, "-u", "-m", "engine.worker"], cwd=<root>, env["PYTHONPATH"]=<root>)` で worker を常駐起動
  → `{"op":"load",...}` → `@@LTX@@{"event":"ready"}` → `{"op":"generate",...}` → `@@LTX@@{"event":"done",...}`。
  worker がモデルを **1度だけ**構築してジョブを使い回し、mp4 は worker が直接ディスクへ書きます（制御 JSON のみパイプを渡る）。
- **16GB 技術**（すべて `engine/` に実装・実測済み）: GGUF Q4_K_M transformer + block-swap（GPU 常駐 8 ブロック）+
  GGUF Gemma の逐次 per-layer CPU オフロード（`--te-offload`）+ DiT の CPU 構築（`--dit-cpu-load`）+ VAE タイリング +
  component-file 経路。512×320 で peak_vram ~9.2GB、720p（1280×768→crop）実証済み。
- **モック backend** は `./.venv` のみで動く合成クリップ生成で、テストと GPU 無し開発に使います（`GenerationOutcome.backend`
  だけが real と異なり、API/スキーマ/出力構造は同一）。

---

## 4. API 概要（`/api/v1`）

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/status` | サーバー/GPU/低VRAM/キュー状態（`vram_optimization` ブロックを含む） |
| GET | `/config` | 現在の設定 |
| POST | `/pipeline/load` / `/pipeline/unload` | パイプラインの明示ロード/アンロード |
| POST | `/upload/image` | 最小I2V用画像をアップロードし `image_id` を返す |
| POST | `/generate` | 生成ジョブ開始（即 `job_id` を返す, 202） |
| GET | `/jobs` / `/jobs/{id}` | ジョブ一覧 / 状態 |
| GET | `/jobs/{id}/video` | 完了動画(mp4)を取得 |
| DELETE | `/jobs/{id}` | 実行中ジョブのキャンセル(best-effort) / 完了ジョブの削除 |

### 凍結 API 契約（付録A・変更しない）
- 幅・高さは **64の倍数**（two-stage distilled は stage1 を半解像度で生成し x2 アップサンプルするため ÷64。
  検証は [`api/models.py`](api/models.py) の `GenerateRequest` validator）。
- フレーム数は **8n+1**（9, 17, 25, … 121）。尺 cap は **20s（481f=8×60+1）@24fps** まで許容
  （旧 257f/10.67s から緩和）。溢れ/低速/非実用は**クライアント UI 警告に委ねる**方針で、
  解像度別 spill-free フレーム数を `GET /api/v1/config` の `limits.spill_free_frames`
  （720p:257 / 1080p:153 / 1440p:81）に露出する。これを超えると shared へ溢れ ~2-4x 低速化
  （OOM せず）。1080p の長尺は非実用（~40分・commit リスク）のため **720p 生成＋外部 upscale** 推奨。
  閾値の正本は [`Docs/RESOLUTION_DURATION_CAPABILITY.md`](Docs/RESOLUTION_DURATION_CAPABILITY.md) §8.4/§8.6。
- Distilled は **8 steps / CFG=1.0** 固定。
- 最小I2V は **画像1枚・`frame_idx=0` 固定**（画像なし=T2V、1枚=I2V）。
- `crop_output` を指定すると、任意の非64サイズ（例 960×540, 1280×720）を中央クロップで得ます。

---

## 5. 16GB 向け生成テスト

出力は `outputs/{job_id}/output.mp4` と `outputs/{job_id}/metadata.json` に保存されます。
`peak_vram_mb` は `metadata.json` または `logs/ltx_worker.log` の `GENERATED_OK peak_vram_mb=` から取得できます
（jobs API 応答には含まれません）。

### smoke_test (T2V, 384x256 / 17 frames)

```bash
curl -X POST http://127.0.0.1:18620/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A red ball rolling on a white floor","width":384,"height":256,"num_frames":17,"num_inference_steps":8,"guidance_scale":1.0,"pipeline":"distilled","seed":42,"conditioning_images":[]}'
# -> {"job_id":"...","status":"queued",...}

curl http://127.0.0.1:18620/api/v1/jobs/<job_id>
curl http://127.0.0.1:18620/api/v1/jobs/<job_id>/video --output out.mp4
```

### minimal (T2V, 512x320 / 49 frames)

```powershell
# seed 固定で決定的に検証（別プロセスでもバイト一致することを実測済み）
$body = '{"prompt":"a calm ocean wave rolling onto a sandy beach at sunset, cinematic","width":512,"height":320,"num_frames":49,"num_inference_steps":8,"guidance_scale":1.0,"pipeline":"distilled","seed":12345,"conditioning_images":[]}'
# POST /api/v1/generate に body を送り、GET /api/v1/jobs/{id} で完了確認
Select-String -Path logs/ltx_worker.log -Pattern "GENERATED_OK|LOAD_FAILED|GENERATE_FAILED"
```

### 720p (1280x768 生成 → 任意で 1280x720 にクロップ)

```powershell
# crop_output を付けると worker が full-size を書き、ffmpeg 中央クロップで 1280x720 配信
$body = '{"prompt":"...","width":1280,"height":768,"num_frames":49,"num_inference_steps":8,"guidance_scale":1.0,"pipeline":"distilled","seed":12345,"crop_output":{"width":1280,"height":720},"conditioning_images":[]}'
```

### 最小I2V (512x320 / 49 frames + 画像1枚)

```bash
# 1) 画像アップロード
curl -X POST http://127.0.0.1:18620/api/v1/upload/image -F "file=@first_frame.png"
# -> {"image_id":"...",...}

# 2) 生成（conditioning_images に image_id を1件だけ・frame_idx=0）
curl -X POST http://127.0.0.1:18620/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The scene slowly comes alive","width":512,"height":320,"num_frames":49,"num_inference_steps":8,"guidance_scale":1.0,"pipeline":"distilled","seed":42,"conditioning_images":[{"image_id":"<id>","frame_idx":0,"strength":0.8}]}'
```

### worker 単体スモーク（engine を直接叩く場合）

```powershell
$env:PYTHONPATH = (Get-Location).Path
& ".\.venv-engine\Scripts\python.exe" -m engine.worker   # {"op":"load",...} を stdin へ → @@LTX@@{"event":"ready"}
```

### Gradio UI
`/ui` を開き、画像なしで「生成」→ T2V、画像1枚を指定して「生成」→ 最小I2V。

**A2V（音声から動画生成）**: Generate タブの「A2V（音声から動画生成）」アコーディオンに音声ファイルを添付する。`.wav` 推奨（添付すると音声長に収まる最大フレーム数を自動でFramesへ入力してくれる。他形式は自動調整の対象外でサーバー側チェックに委ねる）。画像でキャラクター等を固定したい場合は「キーフレーム画像」アコーディオンを使う（5スロットとも A2V と併用可）。**スタイルLoRA（画風・キャラクター系の軽量アダプタ。`<lora:...>` 記法）とは併用できる**（2026-07-11 に解禁。それ以前は排他だった）。ただし参照動画を必要とする control 系 IC-LoRA（参照動画から輪郭線 canny・骨格 pose 等を読み取って条件付けするアダプタ）は、チェーン生成が参照動画を持たない構造のため併用できず、指定すると送信前または `422 LORA_CONTROL_UNSUPPORTED_IN_CHAIN` で拒否される。

---

## 6. テスト

```powershell
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
.\.venv\Scripts\python.exe -m pytest -q
```

pytest は **アプリ venv（`./.venv`, torch 無し）** で動きます。`tests/conftest.py` が `model.backend="mock"` を強制するため、
GPU/モデル無しで T2V/I2V のバリデーション（64倍数・8n+1・複数画像・frame_idx≠0）とモックランナーによる生成疎通、
`GET /status` の `vram_optimization` 契約を検証します。

---

## 7. 制限事項（2026-07-11 現在）

- **`low_vram_mode=true` がデフォルト**。16GB 環境前提。`low_vram_mode=false` は高VRAM/クラウド用の任意検証で、16GB成功は保証しません。
- 同時実行は **1ジョブのみ**。実行中の新規 `POST /generate` は **409 Conflict**。
- ジョブ履歴は in-memory（再起動で消える）。`outputs/{job_id}/metadata.json` はディスクに残ります。
- **実行中ジョブのキャンセルは best-effort**。PyTorch 推論を安全に中断できないため、`running` のジョブは推論完了後に `cancelled` へ遷移します。まだ実行に移っていない `queued` のジョブは、`DELETE /jobs/{id}` で**即座に** `cancelled` になり単一ジョブガードが解放されます（2026-07-11 改修）。
- transformer は Q4_K_M 量子化のため、フル bf16 公式とビット一致ではありません（聴感・視感は良好）。
- **旧「未実装（Phase 2以降）」一覧の現状**（Phase 1 当時の一覧はその後の拡張で大半が実装済みになりました）:
  - **実装済み**: 複数キーフレームI2V（キーフレーム画像・最大5枚・任意 `frame_idx`）／V2V（元動画からの継続生成）／A2V（音声から動画生成）／クリップ連結（`POST /generate/chain`）／IC-LoRA・スタイルLoRA（`<lora:...>` 記法含む）／1080p の直接生成（`FHD_1080p` プリセット。なお「1080p アップスケール機能」としての提供は仕様書 §13.5 でスコープ削除）／簡易認証（`--api-key` 指定時の Bearer 認証・任意）。
  - **引き続き未実装**: 終了フレーム指定（キーフレームは末尾近傍まで＝`frame_idx` は `num_frames-8` にクランプされ、最終フレームちょうどの条件付けはできない）／本格的なジョブキュー（単一ユーザー想定のため「1ジョブ＋busy 409」を正式仕様とし、仕様書 §13.5 でスコープ削除）／AviUtl2 拡張フロントエンド（別リポジトリで開発・本リポジトリは汎用 REST API のまま）。

---

## ライセンス
Apache-2.0 を推奨（上流コード利用時の整合性を優先）。`engine/` は LTX-2 / LTX-Desktop 由来の派生コードを含むため、
再配布時は上流の帰属表示を保持してください（[`engine/VENDOR_NOTICE.md`](engine/VENDOR_NOTICE.md)）。モデルウェイトは
LTX-2 Community License に従ってください。
