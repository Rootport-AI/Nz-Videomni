# LTX-AviUtl2-Bridge

LTX 2.3 動画生成モデルを **VRAM 16GB** のコンシューマーGPUで動かし、REST API として公開するバックエンドサーバー。検証用に Gradio UI(`/ui`) を同梱します。
最終的には AviUtl2 / DaVinci Resolve などの薄いフロントエンドから利用しますが、API は汎用設計です。

これは **Phase 1**（T2V + 最小I2V を同一MVP）の実装です。詳細仕様は
[`LTX23_Backend_Specification_v04_Phase1_T2V_I2V.md`](LTX23_Backend_Specification_v04_Phase1_T2V_I2V.md) を参照してください。

> **Phase 1 の現状**: LTX 公式パイプライン呼び出しは [`services/ltx_runner.py`](services/ltx_runner.py) に隔離されており、
> 現在は **モック実装**（合成クリップを生成）です。GPU / モデルウェイトが無くても API・ジョブ管理・Gradio・テストまで完全に疎通します。
> 実モデルへの差し替えは下記「LTX 実体への差し替え (Step 7)」を参照。

---

## 0. 環境分離ポリシー（最重要・最初に読む）

**このプロジェクトは PC のシステム Python 環境を一切汚しません。** Python 本体を含め、必要なものはすべて
プロジェクトディレクトリ配下（`.venv/`, `.python/`）に閉じ込めます（仕様書 2.5）。

- グローバル/システムの `pip install` は **禁止**。必ず `uv` + プロジェクトローカル `.venv`。
- 環境変数（`PYTORCH_CUDA_ALLOC_CONF`, `UV_PYTHON_INSTALL_DIR`）は **そのプロセス内のみ**。永続化しない。
- 後片付けはこのディレクトリ（`.venv` / `.python` 含む）を削除するだけで完全に元に戻ります。

---

## 1. セットアップ

### 必要なもの
- Windows 10/11 x64
- [`uv`](https://docs.astral.sh/uv/)（パッケージ/Python管理）
- `ffmpeg`（PATH に通すこと。MP4エンコード・クロップに使用）
- 実モデル実行時のみ: NVIDIA CUDA GPU（VRAM 16GB 推奨）

### 初回セットアップ（PowerShell, プロジェクトルートで実行）

```powershell
# Python 本体もプロジェクト内に固定する（システムを汚さない）
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"

uv python install 3.12
uv venv --python 3.12 .venv
uv sync --extra dev
```

これで `.venv/`（仮想環境）と `.python/`（uv管理のPython 3.12）がプロジェクト内に作成されます。

> **注**: `pyproject.toml` には FastAPI 側の依存のみ定義しています（`torch` / `ltx-pipelines` は含みません）。
> 公式 LTX スタックは Step 7 で別途導入します。

---

## 2. 起動

```powershell
# 推奨: 起動スクリプト（環境変数設定 → venv → 起動 を一括）
./run.ps1

# もしくは直接
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
.\.venv\Scripts\python.exe main.py
```

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

```powershell
./run.ps1 --listen --port 19000
```

---

## 3. API 概要（`/api/v1`）

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/status` | サーバー/GPU/低VRAM/キュー状態 |
| GET | `/config` | 現在の設定 |
| POST | `/pipeline/load` / `/pipeline/unload` | パイプラインの明示ロード/アンロード |
| POST | `/upload/image` | 最小I2V用画像をアップロードし `image_id` を返す |
| POST | `/generate` | 生成ジョブ開始（即 `job_id` を返す, 202） |
| GET | `/jobs` / `/jobs/{id}` | ジョブ一覧 / 状態 |
| GET | `/jobs/{id}/video` | 完了動画(mp4)を取得 |
| DELETE | `/jobs/{id}` | 実行中ジョブのキャンセル(best-effort) / 完了ジョブの削除 |

### 制約（付録A）
- 幅・高さは **32の倍数**
- フレーム数は **8n+1**（9, 17, 25, … 121）
- Distilled は **8 steps / CFG=1.0** 固定
- 最小I2V は **画像1枚・`frame_idx=0` 固定**（画像なし=T2V、1枚=I2V）

---

## 4. 16GB 向け最小生成テスト

### curl: smoke_test (T2V, 384x224 / 17 frames)

```bash
curl -X POST http://127.0.0.1:18620/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A red ball rolling on a white floor","width":384,"height":224,"num_frames":17,"num_inference_steps":8,"guidance_scale":1.0,"pipeline":"distilled","seed":42,"conditioning_images":[]}'
# -> {"job_id":"...","status":"queued",...}

curl http://127.0.0.1:18620/api/v1/jobs/<job_id>
curl http://127.0.0.1:18620/api/v1/jobs/<job_id>/video --output out.mp4
```

### 最小I2V (phase1_default, 512x288 / 49 frames + 画像1枚)

```bash
# 1) 画像アップロード
curl -X POST http://127.0.0.1:18620/api/v1/upload/image -F "file=@first_frame.png"
# -> {"image_id":"...",...}

# 2) 生成（conditioning_images に image_id を入れる）
curl -X POST http://127.0.0.1:18620/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The scene slowly comes alive","width":512,"height":288,"num_frames":49,"num_inference_steps":8,"guidance_scale":1.0,"pipeline":"distilled","seed":42,"conditioning_images":[{"image_id":"<id>","frame_idx":0,"strength":0.8}]}'
```

### Gradio UI
`/ui` を開き、画像なしで「生成」→ T2V、画像1枚を指定して「生成」→ 最小I2V。

出力は `outputs/{job_id}/output.mp4` と `outputs/{job_id}/metadata.json` に保存されます。

---

## 5. テスト

```powershell
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
.\.venv\Scripts\python.exe -m pytest -q
```

バリデーション（32倍数・8n+1・複数画像・frame_idx≠0）と、モックランナーによる T2V/I2V 生成疎通を検証します。

---

## 6. 制限事項（Phase 1）

- **`low_vram_mode=true` がデフォルト**。16GB 環境前提。`low_vram_mode=false` は高VRAM/クラウド用の任意検証で、16GB成功は保証しません。
- 同時実行は **1ジョブのみ**。実行中の新規 `POST /generate` は **409 Conflict**。
- ジョブ履歴は in-memory（再起動で消える）。`outputs/{job_id}/metadata.json` はディスクに残ります。
- **キャンセルは best-effort**。PyTorch 推論を安全に中断できないため、推論完了後に `cancelled` へ遷移します。
- 未実装（Phase 2以降）: 複数キーフレームI2V / 終了フレーム / V2V / IC-LoRA / 1080pアップスケール / 本格ジョブキュー / 認証 / AviUtl2プラグイン。
- 現状の生成は **モック**（合成クリップ）。実モデルは Step 7 で差し替え。

---

## 7. LTX 実体への差し替え (Step 7)

実モデルで動かす際は、システムを汚さない原則を保ったまま以下を行います。

1. 公式リポジトリを隣接配置し、その場で（プロジェクト管理下の venv に）セットアップ:
   ```powershell
   git clone https://github.com/Lightricks/LTX-2.git
   cd LTX-2; uv sync --frozen   # ここでも .venv はこのディレクトリ専用
   ```
2. LTX-2.3 モデルウェイトを取得: <https://huggingface.co/Lightricks/LTX-2.3>（`ltx-2.3-22b-distilled`）。
3. [`services/ltx_runner.py`](services/ltx_runner.py) の `load()` / `_render_frames()` を、公式 `DistilledPipeline`
   （`QuantizationPolicy.fp8_cast()`, text encoder CPU offload, VAE tiling）呼び出しへ置換。
4. **API・スキーマ・ジョブ層・出力構造は変更しない。**

検証順序（付録B）: モデルロード → `smoke_test`(384x224/17) → `phase1_default`(512x288/49) →
`phase1_default` I2V → `phase1_target`(960x544/121, crop 960x540)。peak VRAM と生成時間は
`metadata.json` とログに記録されます。

---

## ライセンス
Apache-2.0 を推奨（上流コード利用時の整合性を優先）。モデルウェイトは LTX-2 Community License に従ってください。
