# LTX-AviUtl2-Bridge

LTX 2.3 動画生成モデルを **VRAM 16GB** のコンシューマーGPUで動かし、REST API として公開するバックエンドサーバー。検証用に Gradio UI(`/ui`) を同梱します。
最終的には AviUtl2 / DaVinci Resolve などの薄いフロントエンドから利用しますが、API は汎用設計です。

これは **Phase 1**（T2V + 最小I2V を同一MVP）の実装です。詳細仕様は
[`LTX23_Backend_Specification_v04_Phase1_T2V_I2V.md`](LTX23_Backend_Specification_v04_Phase1_T2V_I2V.md) を参照してください。

> **Phase 1 の現状**: LTX 公式パイプライン呼び出しは [`services/ltx_runner.py`](services/ltx_runner.py) に隔離されており、
> 我々の backend が呼ぶ既定はまだ **モック実装**（合成クリップ）です。GPU / モデルウェイトが無くても API・ジョブ管理・Gradio・テストまで完全に疎通します。
>
> ⚠️ **重要（2026-06-28 追記・本 README は一部 pre-pivot のまま）**: 実エンジンの方針は変わりました。公式 `ltx_pipelines` は
> 本機(16GB Windows)で native crash するため**不採用**で、現行は **低VRAMフォーク（GGUF transformer＋block-swap＋GGUF Q4 Gemma）**
> 方式に pivot し、**フォーク env スパイクで 16GB E2E T2V 生成に成功済み**です。我々の backend への取り込み（`_RealBackend` 実体化）は
> **Phase 5** で実施予定で、その完了後に本 §7「Step 7」と解像度制約（現行の正しい契約は **÷64**。L102 の「÷32」は旧記述）を含め
> README を全面改訂します。**現時点の正本**は [`Docs/NEXT_SESSION_HANDOFF.md`](Docs/NEXT_SESSION_HANDOFF.md) と
> [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) を参照してください。下記 §7 は旧（公式パイプライン）手順です。

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

> **注**: `pyproject.toml`（および pip 用の [`requirements.txt`](requirements.txt)）には FastAPI 側の依存のみ定義しています
> （`torch` / `ltx-pipelines` は含みません）。公式 LTX スタックは GPU 世代依存のため Step 7 の
> [`scripts/install_ltx.ps1`](scripts/install_ltx.ps1) で別途導入します。
>
> uv を使わない場合の pip フォールバック: `python -m venv .venv; .venv\Scripts\pip install -r requirements.txt`

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

## 7. LTX 2.3 のインストール & 実体への差し替え (Step 7)

実モデルで動かす際も、**システムを汚さない原則**（すべてプロジェクト内）を保ちます。
重いデータ（公式リポジトリ・モデルウェイト）は `.gitignore` 済みで **GitHub には絶対に入りません**
（`vendor/`, `models/*`, `*.safetensors` などを除外）。

> **Phase 1 の現状到達状況（2026-06-27 時点）**: 公式リポジトリの clone・LTX-2 専用 venv
> （`vendor/LTX-2/.venv`, **torch 2.9.1+cu128 で GPU 認識 OK**）・モデル3点（distilled 約46GB /
> spatial upsampler 約1GB / Gemma 3 12B 約25GB）の DL は**完了済み**で、`config.yaml` のパスは実ファイルと一致しています。
> 残るは **公式CLIでの実機検証**と、[`services/ltx_runner.py`](services/ltx_runner.py) の**モック→実体差し替え**です
> （ランナーは現状モックのまま）。

### 7.0 依存関係（公式LTX-2の固定値）

| 項目 | 値 | 備考 |
|------|----|------|
| Python | `>=3.10`（当方は 3.12） | `ltx-core` requires-python |
| PyTorch | `torch ~=2.7` + `torchaudio`（範囲指定。実体は **2.9.1+cu128**） | PyTorch **cu128（CUDA 12.8）** wheel index |
| NVIDIA ドライバ | **CUDA 12.8（cu128）対応の新しめのドライバ** | 古いと cu128 wheel が動かない |
| attention backend | **既定は SDPA**（xformers は任意の最適化。下表） | ここだけが世代差の調整点 |
| FP8 | `fp8-cast`（bf16 checkpoint用） | Ada は FP8 tensor core 対応 |

**必要なモデルは3点**（VAEはcheckpointに同梱で別途不要。現行公式 v1.1。合計 約70GB）

| 要素 | 入手元 | 概算 | 別途DL |
|------|--------|------|--------|
| distilled checkpoint（VAE同梱） | `ltx-2.3-22b-distilled-1.1.safetensors` | 約46GB | ○ |
| spatial upsampler | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 約1GB | ○ |
| text encoder (Gemma 3) | `google/gemma-3-12b-it-qat-q4_0-unquantized`（**gated**） | 約25GB | ○（`-WithGemma`） |

**attention backend（既定は SDPA。GPU世代別の任意最適化が下表）**

| GPU世代 | 代表例 (compute cap.) | バックエンド | スクリプト指定 |
|---------|----------------------|-------------|----------------|
| **Ada Lovelace** | RTX 40系 / L40 / L4 (sm_89) | **SDPA（既定。xformers は任意の最適化）** | `-GpuArch ada`（**既定・本機**） |
| Ampere | RTX 30系 / A100 / A6000 (sm_80/86) | **SDPA（既定。xformers は任意の最適化）** | `-GpuArch ampere` |
| Hopper | H100 / H200 (sm_90) | **SDPA（既定。xformers は任意の最適化）** | `-GpuArch hopper` |
| Blackwell | RTX 50系 / B200 (sm_100/120) | flash-attn-4（**本機では未検証**・他世代向け情報） | `-GpuArch blackwell` |

> **本機は Ada Lovelace 世代**で、既定の **SDPA**（Ada + torch 2.9 では実体が FlashAttention-2 カーネル）のままでOKです。
> xformers はあくまで任意の最適化（現状ビルドしていない）。別世代のユーザーは `-GpuArch` を変えるだけ（コード修正不要）。

### 7.1 HuggingFace 認証（Gemma 3 は gated・手作業で先に）

Gemma 3 はライセンス承認が必要な gated モデルです。**インストール前に一度だけ**認証します。
トークンはプロジェクト内 `hf_home/`（gitignore済み）に保存され、システムを汚しません。

```powershell
# (任意・推奨) uv/HF キャッシュもプロジェクト内へ隔離し、空き容量の大きいドライブへ
$env:UV_CACHE_DIR = "$PWD\.uv_cache"
$env:HF_HOME      = "$PWD\hf_home"

# ブラウザで一度だけ: ライセンス承認 + READトークン作成
#   https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
#   https://huggingface.co/settings/tokens
./scripts/hf_login.ps1          # hf auth login（トークン貼り付け）→ hf_home に保存
```

> 代替: ログインせず一時的に `$env:HF_TOKEN = "hf_..."` をセットしてもOK（プロセス内のみ）。

### 7.2 インストールスクリプト

```powershell
# 1) 公式LTX-2を vendor/LTX-2 に clone + uv sync + cu128 の torch 2.9.1 を導入、LTX-2.3重みを models/ltx-2.3 へ
#    （xformers はビルドしないため SDPA で動作。任意の最適化として §7.3 でビルド可）
./scripts/install_ltx.ps1                       # Ada Lovelace（本機）

# 別世代の例
./scripts/install_ltx.ps1 -GpuArch blackwell    # RTX 50系（flash-attn-4）

# Gemma text encoder も取得（事前に ./scripts/hf_login.ps1 で認証済みなら -HfToken 不要）
./scripts/install_ltx.ps1 -WithGemma

# clone と uv sync だけ（ダウンロードは後で）
./scripts/install_ltx.ps1 -SkipDownload
```

スクリプトの動作:
1. `vendor/LTX-2`（gitignore済み）へ公式リポジトリを clone
2. その中で `uv sync --frozen` で基本スタックを同期し、**cu128 インデックスから torch 2.9.1+cu128 を導入**
   → **LTX-2専用の `.venv`**（システム非汚染。Windows では `uv sync --frozen` だけだと `2.9.1+cpu` が入るため cu128 版を手動導入する）
   → **xformers はビルドしていないので SDPA で動作**（任意の最適化は §7.3）。Blackwell は `flash-attn-4`
3. `Lightricks/LTX-2.3` の重みを `models/ltx-2.3` へダウンロード
4. （`-WithGemma`時）gated な `google/gemma-3-12b-it-qat-q4_0-unquantized` を `models/gemma-3-12b-it-qat` へ
5. `config.yaml` の `model:` に貼る**実パスを表示**し `models/INSTALLED_PATHS.txt` に保存
   （`UV_CACHE_DIR`/`HF_HOME` 未設定なら、uv/HFキャッシュもプロジェクト内 `.uv_cache`/`hf_home` を既定使用）

> 将来的に GPU 世代の自動判定を追加予定ですが、現時点では `-GpuArch` の手動指定です。

> **重要**: `Lightricks/LTX-2.3` の正確なファイル名はリポジトリ更新で変わり得ます。
> ダウンロード対象が合わない場合は <https://huggingface.co/Lightricks/LTX-2.3/tree/main> を確認し、
> `-Include "<glob>"` で調整してください。

公式CLIの生成コマンド（参考）:
```bash
python -m ltx_pipelines.distilled \
  --checkpoint-path models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
  --spatial-upsampler-path models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root models/gemma-3-12b-it-qat \
  --quantization fp8-cast \
  --prompt "..." [--image first_frame.png] --output-path out.mp4
```

### 7.3 Windows: xformers をソースビルド（任意・現時点では未実施）

> **位置づけ**: これは **任意（Optional）の将来の最適化**であり、**現時点では未実施**です。
> 既定は SDPA（Ada + torch 2.9 では実体が FlashAttention-2）で動作し、`wheels/` は意図的に空のままです。
> 総生成時間の律速は CPU offload streaming の PCIe 転送であり、SDPA と xformers の差は全体の数%程度
> （詳細は [`Docs/note.md`](Docs/note.md)）。**速度をさらに追求したくなった場合のみ**本ビルドを行ってください。

LTX-2 が固定する xformers（cu128 devビルド）は **Linuxホイールのみ**で Windows に入りません。
そこで **torch 2.9.1 / cu128 / Windows / py3.12 / Ada(sm_89)** 向けに**自前ビルド**し、生成 wheel を
`wheels/`（**Git LFS**管理）に置きます。一度ビルドすれば、他のクローンは `install_ltx.ps1` が
その wheel を自動導入するだけで済みます。

> PyTorch 2.x の SDPA は xformers の memory-efficient/flash-2 を内包するため、wheel が無くても
> SDPA で問題なく動作します（既定）。xformers を使う場合のみ本ビルドが必要です。

**前提ツール（システムの開発ツール。Python隔離ルールには非抵触）**

⚠️ **CUDA 12.9 は Visual Studio 2026 の既定MSVC(v14.5x)を未サポート**（公式対応はCUDA 13.2+）。
VS 2026を使う場合は **MSVC v143（VS 2022相当, v14.44）コンポーネントを追加**し、ビルド時にそれを選ぶ
（`build_xformers.ps1` が `-vcvars_ver=14.44` で自動選択）。
**ランタイムは cu128（CUDA 12.8）だが、ビルド用 CUDA Toolkit は 12.9 を使う**（torch は cu128 ランタイム、
ビルドは CUDA 12.9 toolkit で minor 差の警告のみで通る想定。ランタイム=cu128 とビルドtoolkit=12.9 は別物）。

- VS 2026（IDE）に C++ を入れる場合（GUI推奨・確実）:
  1. Visual Studio Installer →「C++によるデスクトップ開発」ワークロードを選択
  2.「個別のコンポーネント」タブ → `v143` で検索 →
     **「MSVC v143 - VS 2022 C++ x64/x86 ビルド ツール (v14.44)」** と Windows 11 SDK をチェック
- winget で（IDEのワークロード追加。コンポーネントIDはGUIで確認推奨）:
  ```powershell
  winget install --id Microsoft.VisualStudio.2026.Community -e `
    --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Component.VC.v143.x86.x64 --includeRecommended"
  ```
- CUDA Toolkit 12.9（**ビルド用**。torch ランタイムは cu128 だが、ビルドは 12.9 toolkit で minor 差の警告のみ。`winget show` で正確な版を確認）:
  ```powershell
  winget show Nvidia.CUDA --versions
  winget install --id Nvidia.CUDA --version 12.9.1
  ```

**ビルド & 配布**
```powershell
./scripts/install_ltx.ps1 -SkipDownload     # 先に torch 2.9.1+cu128 を vendor/LTX-2/.venv へ
./scripts/build_xformers.ps1 -Install        # ビルド→wheels/へ出力→LTX venvへ導入（30–90分）

# 生成 wheel を Git LFS でコミット（他クローンはこれを再利用）
git add wheels/xformers-*.whl .gitattributes
git commit -m "Add prebuilt xformers wheel (torch2.9.1/cu128/win/py312, sm_89)"
```

ビルドスクリプトは MSVC（vcvars64）と CUDA 12.9 toolkit（ビルド用）を自動設定し、`TORCH_CUDA_ARCH_LIST=8.9` /
`--no-build-isolation` で LTX が pin する commit を、cu128 ランタイムの torch 2.9.1 向けにビルドします。

### 7.4 ランナーの差し替え

1. `install_ltx.ps1` が出力したパスを `config.yaml` の
   `model.checkpoint_path` / `spatial_upsampler_path` / `gemma_root` に設定。
2. [`services/ltx_runner.py`](services/ltx_runner.py) の `load()` / `_render_frames()` を、公式
   `DistilledPipeline`（`QuantizationPolicy.fp8_cast()` / text encoder CPU offload / VAE tiling）
   呼び出しへ置換（docstring にスケルトンあり）。
3. **API・スキーマ・ジョブ層・出力構造は変更しない。**

検証順序（付録B）: モデルロード → `smoke_test`(384x224/17) → `phase1_default`(512x288/49) →
`phase1_default` I2V → `phase1_target`(960x544/121, crop 960x540)。peak VRAM と生成時間は
`metadata.json` とログに記録されます。

---

## ライセンス
Apache-2.0 を推奨（上流コード利用時の整合性を優先）。モデルウェイトは LTX-2 Community License に従ってください。
