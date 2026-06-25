# LTX 2.3 動画生成バックエンド — 設計・仕様書（Codex実装向け修正版）

**プロジェクト名:** LTX-AviUtl2-Bridge  
**バージョン:** v0.4 Draft / AI-agent-ready / 16GB-first / T2V+minimal-I2V  
**作成日:** 2026-06-25  
**修整方針:** v0.3 Draft を、Phase 1でT2Vと最小I2Vを同一MVPとして実装できるよう再整理したもの  
**対象環境:** Windows 10/11 x64 / NVIDIA CUDA GPU / 主要開発環境 VRAM 16GB  

---

## 0. AIエージェントへの実装指示

この仕様書をAIコーディングエージェントに渡す場合は、まず **Phase 1のみ** を実装すること。  
本プロジェクトの主要開発マシンは **VRAM 16GB GPU** である。したがって、Phase 1から **Low VRAM mode をデフォルト有効** にし、16GB環境で毎回テストできる構成を最優先する。

Phase 1では **T2V と最小I2Vを同一MVPとして実装する**。  
画像なしで生成した場合はT2V、画像1枚を指定した場合は開始フレーム条件付きI2Vとして扱う。

Phase 2以降の高度な低VRAM最適化、複数キーフレームI2V、V2V、IC-LoRA、1080pアップスケール、本格ジョブキューはまだ実装しない。  
AviUtl2フロントエンドは、バックエンドAPIが安定してから、半ば独立した後続プロジェクトとして実装する。

### 0.1 Phase 1 のゴール

- `python main.py` で FastAPI + Gradio UI が起動する
- Low VRAM mode がデフォルトで有効になる
- `/ui` から画像なしでT2V生成できる
- `/ui` から画像1枚を指定して最小I2V生成できる
- `/api/v1/status` が JSON を返し、GPU情報、VRAM情報、Low VRAM設定を確認できる
- `/api/v1/upload/image` でI2V用画像をアップロードし、`image_id` を返せる
- `/api/v1/generate` は **最終仕様と同じくジョブIDを返す**
- Phase 1では内部実装を簡略化し、同時実行は1ジョブのみとする
- 生成中に別ジョブが投入された場合は `409 Conflict` を返す
- 生成結果は `outputs/{job_id}/output.mp4` に保存する
- 入力パラメータ、生成モード、実際に使った seed、生成時間、出力ファイル情報、peak VRAM、Low VRAM設定を `outputs/{job_id}/metadata.json` に保存する
- README にセットアップ手順、起動手順、制限事項、16GB向け最小生成テストを書く

### 0.2 Phase 1 で実装する Low VRAM baseline

Phase 1では、低VRAM対応を「後回し」にしない。  
ただし、壊れやすい独自パッチを一気に移植するのではなく、公式パイプラインやPyTorchの標準機能で扱いやすい範囲を **Low VRAM baseline** として実装する。

Phase 1で有効にするもの:

- FP8 checkpoint / FP8 transformer を優先して使う
- text encoder を可能な限りCPUへ退避する
- VAE tiling が公式APIで利用できる場合は有効にする
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` をREADMEで案内する
- 生成ごとに `torch.cuda.empty_cache()` 等の安全なメモリ解放を行う
- デフォルト生成サイズを小さくし、段階的に解像度・フレーム数を上げる

Phase 1で実装しないもの:

- BlockSwap の独自移植
- AttentionTile のグローバルモンキーパッチ
- 公式APIで利用できないVAE tilingの独自実装
- 複雑なCPU/GPUブロック転送制御
- 16GBでの `low_vram_mode=false` 成功保証

### 0.3 Phase 1 で実装しないもの

- AviUtl2 `.aux2` プラグイン
- DaVinci Resolve フロントエンド
- latent spatial upscaler による1080p化
- 複数キーフレームI2V
- 終了フレーム conditioning
- V2V
- IC-LoRA
- depth / pose / edge / motion tracking
- 動画アップロード
- 永続DB
- 複数ジョブのキューイング
- ユーザー認証
- インターネット公開

### 0.4 Phase 1 のI2V範囲

Phase 1で対応するI2Vは、以下の **最小I2V** のみとする。

- 入力画像は1枚のみ
- 開始フレーム conditioning のみ
- `frame_idx = 0` 固定
- `strength` のみ指定可能
- 対応形式は `png`, `jpg`, `jpeg`, `webp`
- 画像アップロード後は `uploads/{image_id}/input.png` に正規化保存する
- 複数画像、終了フレーム、任意フレーム指定、V2V、IC-LoRAはPhase 1では実装しない

### 0.5 実装時の重要ルール

- **【最重要・環境分離】PCのシステムPython環境を絶対に壊さない。** Python本体を含め、必要なものはすべてプロジェクトディレクトリ内の仮想環境にインストールする。グローバル/システムの `pip install`、システムPythonへのパッケージ追加、システム環境変数の恒久変更は禁止する。詳細は「2.5 環境分離ポリシー（最重要制約）」を参照。
- 外部API仕様は Phase 1 から最終形に寄せる。あとでAviUtl2/DaVinci Resolve側の通信仕様を壊さないため。
- LTX公式パイプライン呼び出しは `services/ltx_runner.py` に閉じ込める。API層やジョブ管理層に公式パッケージの細部を漏らさない。
- Low VRAM mode はグローバル設定で管理し、`GenerateRequest` には原則として含めない。
- 幅・高さは32の倍数のみ許可する。
- フレーム数は `8n+1` のみ許可する。
- Distilled pipeline のデフォルトは `num_inference_steps=8`、`guidance_scale=1.0` とする。
- Phase 1の最小疎通テストは `384x224 / 17 frames` とする。
- Phase 1のGradio/APIデフォルトは `512x288 / 49 frames` とする。
- Phase 1の目標プリセットは `960x544 / 121 frames / crop 960x540` とする。
- 画像なしならT2V、`conditioning_images` が1件なら最小I2Vとして処理する。
- 1080p相当の最終出力が必要な場合、内部では `1920x1088` のように32倍数へパディングし、最終エンコード時に `1920x1080` へクロップする。
- `low_vram_mode=false` は高VRAM環境・クラウド・将来検証用の任意オプションであり、16GB環境の受け入れ条件に含めない。

## 1. プロジェクト概要

### 1.1 目的

AviUtl2のプラグインから呼び出せるAI動画生成バックエンドサーバーを構築する。  
最終目標は、VRAM 16GBのコンシューマーGPUで、AviUtl2素材として使いやすい動画を生成し、REST APIとして公開することである。  
テスト・検証用にGradio UIを同梱する。

### 1.2 設計思想

- バックエンドが安定して動作することを最優先とする
- AviUtl2プラグインはHTTP APIを叩くだけの薄いクライアントとする
- APIを先に安定させ、将来的に別フロントエンドからも利用可能にする
- LTX公式コードベースとの結合部を薄いアダプタに閉じ込める
- 16GB VRAMで開発中に毎回検証できることを重視する
- Low VRAM baseline はPhase 1から有効化し、重い独自最適化のみPhase 2以降で段階的に追加する
- Phase 1ではT2Vと最小I2Vを同一MVPに含める
- AviUtl2/DaVinci Resolve等の編集ソフト連携は、バックエンドAPI安定後に薄いフロントエンドとして追加する

### 1.3 最終的な想定ワークフロー

Phase 1では、FastAPIバックエンドとGradio検証UIのみを実装する。

```text
[Gradio UI /ui]
        │
        │ HTTP REST
        ▼
[FastAPI バックエンド]
        │
        ├─ REST API
        ├─ Job Manager: Phase 1はin-memory single job
        ├─ Pipeline Manager: LTX 2.3 ロード・アンロード
        ├─ uploads/: I2V用画像保存
        └─ outputs/: 生成動画とメタデータ保存
```

AviUtl2フロントエンドはPhase 5で別プロジェクト相当として実装する。  
バックエンドAPIはAviUtl2専用にしない。将来的にDaVinci Resolve等の別フロントエンドからも同じREST APIを利用できるようにする。

### 1.4 参考リソース

| リソース | URL | 用途 |
|---------|-----|------|
| Lightricks/LTX-2 公式リポジトリ | https://github.com/Lightricks/LTX-2 | 公式 `ltx-pipelines` / `ltx-core` / `ltx-trainer` |
| LTX-2.3 モデル重み | https://huggingface.co/Lightricks/LTX-2.3 | モデルチェックポイントのダウンロード元 |
| LTX-Desktop 公式upstream | https://github.com/Lightricks/LTX-Desktop | ローカル実行UI・構成理解用 |
| LTX-Desktop-LOW-VRAM fork | https://github.com/Kandyman-iac/LTX-Desktop-LOW-VRAM-and-EXTRAS | Phase 2以降の低VRAM実装参考 |
| LTX PyTorch API Docs | https://docs.ltx.video/open-source-model/integration-tools/pytorch-api | 公式Python統合方針の確認 |
| AviUtl2 SDK 非公式ミラー | https://github.com/aviutl2/aviutl2_sdk_mirror | Phase 5の `.aux2` プラグイン実装参考 |

---

## 2. 実行環境

### 2.1 主要開発環境

本プロジェクトの主要開発環境は **VRAM 16GB GPU** とする。  
そのため、Phase 1から Low VRAM mode をデフォルト有効にし、低解像度・短尺の動画生成で疎通確認できることを最初の完了条件に含める。

| 項目 | Phase 1 必須/推奨 | 備考 |
|------|------------------|------|
| OS | Windows 10/11 x64 | 開発対象 |
| GPU | NVIDIA CUDA GPU / VRAM 16GB | 主要開発環境 |
| RAM | 32GB以上推奨 | CPU offload / block swap 検証に必要 |
| 空きディスク | 160GB以上推奨 | モデル、出力動画、中間ファイル用 |
| Python | 3.12系 | 公式LTX-2環境に合わせる |
| CUDA | 12.8系を優先 | PyTorch 2.7系との整合性を優先 |
| PyTorch | 2.7系を優先 | 公式LTX-2環境に合わせる |
| パッケージ管理 | uv 推奨 | 公式LTX-2のセットアップ方針に合わせる |
| 動画エンコード | ffmpeg を PATH に通す | MP4保存・クロップ用 |

### 2.2 GPUメモリ方針

16GB環境では、素の22B級パイプラインを `low_vram_mode=false` で安定動作させることを前提にしない。  
解像度やフレーム数を小さくしても、モデル重みやtext encoderがVRAMを大きく消費するため、最小動画でもOOMする可能性がある。

そのため、Phase 1では以下を標準方針とする。

- `low_vram_mode=true` をデフォルトにする
- FP8 checkpoint / FP8 transformer を優先する
- text encoder CPU offload を有効にする
- 公式APIで使えるVAE tiling等は有効にする
- 最初は `384x224 / 17 frames` で疎通確認する
- 次に `512x288 / 49 frames` をPhase 1標準プリセットにする
- 最後に `960x544 / 121 frames / crop 960x540` をPhase 1目標プリセットとして試す

`low_vram_mode=false` は、24GB以上のGPU、RTX 6000系、クラウドGPUなどで確認するための任意オプションとする。  
16GB環境で `low_vram_mode=false` が失敗しても、Phase 1の失敗とはみなさない。

### 2.3 セットアップ方針

Phase 1では `requirements.txt` よりも、公式LTX-2リポジトリの `uv sync --frozen` 方針を優先する。  
ただし、本プロジェクト自体は小さいFastAPIアプリとして構成し、LTX-2公式コードベースをサブモジュールまたは隣接ディレクトリとして参照する方式を許可する。

推奨ディレクトリ例:

```text
workspace/
├── LTX-2/                    # 公式リポジトリ。uv sync 済み
└── ltx-aviutl2-bridge/        # 本プロジェクト
```

Phase 1でCodexが迷った場合は、以下の優先順位で実装する。

1. 本プロジェクトのFastAPI/Gradio骨組みを作る
2. config.yaml に `low_vram_mode=true` の16GB向け設定を入れる
3. `services/ltx_runner.py` にLTX公式パイプライン呼び出しを閉じ込める
4. 公式LTX-2のサンプルに合わせて `ltx_runner.py` のみを調整する
5. APIスキーマやジョブ管理層は変更しない

### 2.4 CUDAメモリ断片化対策

READMEには、Windows PowerShell向けに以下の設定例を書く。

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python main.py
```

この設定は万能ではないが、CUDAメモリ断片化によるOOMを減らせる可能性がある。  
サーバー起動時にこの環境変数が未設定の場合、ログに警告を出してもよい。

### 2.5 環境分離ポリシー（最重要制約）

本プロジェクトでは、**開発マシンのシステムPython環境を一切汚さない・壊さないこと**を最優先の制約とする。  
Python本体を含め、必要なものはすべてプロジェクトディレクトリ配下の仮想環境にインストールする。

#### 禁止事項

- システム/グローバルの `pip install`（仮想環境を有効化していない状態でのインストール）
- システムにインストールされたPythonへのパッケージ追加・更新・削除
- `conda` のbase環境など、共有環境への直接インストール
- システム環境変数（ユーザー/システムの永続的な `PATH`、`PYTHONPATH` 等）の恒久的な書き換え
- 管理者権限を要するグローバルなツールインストール

#### 必須事項

- パッケージ管理は **uv** を使い、**プロジェクトローカルの `.venv`** に閉じ込める（`uv sync` / `uv run` はデフォルトでプロジェクト直下に `.venv` を作る）。
- **Python本体もプロジェクト内に閉じ込める。** uv管理のPythonを使い、インストール先をプロジェクト配下へ固定する。
  - 例（PowerShell、プロジェクトルートで実行）:
    ```powershell
    # uv が管理するPythonの配置先をプロジェクト内に固定
    $env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
    uv python install 3.12
    uv venv --python 3.12 .venv
    uv sync
    ```
- 環境変数（`PYTORCH_CUDA_ALLOC_CONF`、`UV_PYTHON_INSTALL_DIR` 等）は、**そのセッション/プロセス内のみ** で設定する。永続化しない。
- LTX-2公式リポジトリを隣接配置する場合も、その `.venv` をプロジェクト管理下のディレクトリに置き、システムへインストールしない。
- 起動スクリプト（例: `run.ps1`）を用意し、環境変数設定 → `.venv` 有効化 → 起動 を1コマンドで完結させ、システムを汚さない導線を標準とする。
- 後片付けは、プロジェクトディレクトリ（`.venv` / `.python` 等を含む）を削除するだけで完全に元に戻せる状態を維持する。

#### ディレクトリ配置（環境分離を反映）

```text
ltx-aviutl2-bridge/
├── .venv/        # プロジェクト専用の仮想環境（システムを汚さない）
├── .python/      # uv管理のPython本体（UV_PYTHON_INSTALL_DIRで固定）
├── LTX-2/        # 公式リポジトリ（隣接配置する場合も .venv はここ専用に）
└── ...           # 本プロジェクトのソース
```

READMEには、この環境分離手順を「セットアップ」の最初に明記する。

---

## 3. ネットワーク前提

### 3.1 デフォルト: localhost限定

本サーバーはローカルマシン上での実行を基本とする。インターネット公開は想定しない。

| 項目 | 設定 |
|------|------|
| バインドアドレス | `127.0.0.1` |
| デフォルトポート | `18620` |
| 通信 | HTTP |
| 認証 | なし |
| CORS | `http://127.0.0.1:*`, `http://localhost:*` のみ許可 |
| テレメトリ | 送信しない |
| ログ | ローカルファイルのみ |

起動例:

```bash
python main.py
```

### 3.2 `--listen` モード: 家庭内LAN公開

`--listen` を指定した場合のみ、バインドアドレスを `0.0.0.0` にする。

```bash
python main.py --listen
python main.py --listen --port 19000
```

`--listen` 時の動作:

- 起動時に警告を表示する
- ローカルIPアドレスと `/ui` のURLを表示する
- デフォルトではCORSを家庭内LANの想定範囲に限定する
- `--allow-all-cors` 指定時のみCORS全許可にする
- 任意で `--api-key` を指定できる
- `--api-key` 指定時は `Authorization: Bearer <api-key>` を要求する

注意:

- `--listen` は家庭内LAN用であり、インターネット公開はサポートしない
- ngrok、Cloudflare Tunnel、ポートフォワーディング等は利用者の自己責任とする

---

## 4. アーキテクチャ

### 4.1 全体構成

```text
┌──────────────────────┐
│  Gradio UI (/ui)     │──── HTTP REST ────┐
│  Phase 1検証用        │                    │
└──────────────────────┘                    │
                                            ▼
┌──────────────────────────────────────────────────────┐
│                 FastAPI サーバー                       │
│                                                      │
│  ┌─────────────┐  ┌──────────────────────────────┐   │
│  │ Gradio UI   │  │ REST API ルーター             │   │
│  │ /ui         │  │ /api/v1/...                  │   │
│  └─────────────┘  └──────────┬───────────────────┘   │
│                              │                        │
│  ┌───────────────────────────▼───────────────────┐   │
│  │ Upload Store                                  │   │
│  │ Phase 1: image upload for minimal I2V          │   │
│  └───────────────────────────┬───────────────────┘   │
│                              │                        │
│  ┌───────────────────────────▼───────────────────┐   │
│  │ Job Manager                                    │   │
│  │ Phase 1: in-memory single job                  │   │
│  │ Phase 3: asyncio.Queue + worker                │   │
│  └───────────────────────────┬───────────────────┘   │
│                              │                        │
│  ┌───────────────────────────▼───────────────────┐   │
│  │ Pipeline Manager                               │   │
│  │ - load/unload                                  │   │
│  │ - auto-load on first generation                │   │
│  │ - recovery after OOM                           │   │
│  └───────────────────────────┬───────────────────┘   │
│                              │                        │
│  ┌───────────────────────────▼───────────────────┐   │
│  │ services/ltx_runner.py                         │   │
│  │ Official LTX pipelines adapter                 │   │
│  │ T2V / minimal I2V normalization                │   │
│  └───────────────────────────┬───────────────────┘   │
│                              │                        │
│                     ┌────────▼────────┐               │
│                     │  GPU / CPU RAM  │               │
│                     └─────────────────┘               │
└──────────────────────────────────────────────────────┘

Phase 5以降でAviUtl2フロントエンドを追加する。追加後もバックエンドAPIは変更しない。
```

### 4.2 技術スタック

| レイヤー | 技術 | Phase | 理由 |
|---------|------|-------|------|
| Webフレームワーク | FastAPI | 1 | REST API、OpenAPI、自動docs、非同期対応 |
| テストUI | Gradio | 1 | Pythonだけで検証UIを作れる |
| 推論 | 公式 `ltx-pipelines` | 1 | LTX 2.3対応の中心 |
| ジョブ管理 | in-memory JobStore | 1 | 最小実装。1ジョブのみ |
| 画像アップロード | local UploadStore | 1 | 最小I2V用。画像1枚を保存・正規化 |
| ジョブキュー | `asyncio.Queue` | 3 | 複数ジョブ管理 |
| 動画エンコード | ffmpeg subprocess | 1 | MP4出力・クロップ・メタデータ処理 |
| 設定 | YAML + Pydantic Settings | 1 | 明示的な設定管理 |
| Low VRAM baseline | FP8 / CPU offload / VAE tiling可能なら有効 | 1 | 16GB開発環境での最小動作 |
| 高度な低VRAM | Attention tiling / BlockSwap / 独自VAE tiling | 2 | 960x544/121安定化・VRAM削減 |

### 4.3 ディレクトリ構成

```text
ltx-aviutl2-bridge/
├── main.py                         # FastAPI起動、CLI引数、Gradioマウント
├── config.py                       # config.yaml読み込み、設定クラス
├── gradio_ui.py                    # Gradio UI定義
├── requirements.txt                # FastAPI側の追加依存
├── README.md                       # セットアップ・起動手順
├── config.yaml                     # デフォルト設定
│
├── api/
│   ├── __init__.py
│   ├── router.py                   # APIルーター統合
│   ├── models.py                   # Pydanticスキーマ
│   ├── generate.py                 # POST /generate
│   ├── uploads.py                  # POST /upload/image
│   ├── jobs.py                     # GET /jobs, GET /jobs/{id}, video取得
│   ├── status.py                   # GET /status
│   └── pipeline.py                 # load/unload
│
├── services/
│   ├── __init__.py
│   ├── job_store.py                # Phase 1: in-memory job管理
│   ├── upload_store.py             # Phase 1: I2V用画像保存・正規化
│   ├── pipeline_manager.py         # load/unload/auto-load
│   ├── ltx_runner.py               # 公式ltx-pipelines呼び出しアダプタ
│   ├── video_io.py                 # ffmpeg encode/crop, metadata保存
│   ├── gpu_info.py                 # VRAM情報取得
│   ├── low_vram.py                 # Phase 1: FP8/CPU offload等のbaseline設定
│   └── vram/                       # Phase 2以降: 高度な独自最適化
│       ├── block_swap.py
│       ├── attention_tile.py
│       ├── vae_tile.py
│       └── fp8.py
│
├── outputs/
│   └── .gitkeep
├── uploads/
│   └── .gitkeep                    # Phase 1: I2V用画像保存
└── models/
    └── .gitkeep                    # 必要に応じてモデル配置
```

---

## 5. API設計

### 5.1 設計方針

- APIは Phase 1 から非同期ジョブ方式に統一する
- `POST /api/v1/generate` は即座に `job_id` を返す
- クライアントは `GET /api/v1/jobs/{job_id}` をポーリングする
- 動画取得は `GET /api/v1/jobs/{job_id}/video`
- Phase 1では同時実行1ジョブのみ。キューは持たない
- Phase 3で内部を `asyncio.Queue` に差し替えるが、外部APIは変えない

### 5.2 エンドポイント一覧

#### Phase 1で実装するエンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/api/v1/status` | サーバー状態、GPU情報、パイプライン状態 |
| GET | `/api/v1/config` | 現在の設定値を取得 |
| POST | `/api/v1/pipeline/load` | パイプラインを明示ロード |
| POST | `/api/v1/pipeline/unload` | パイプラインをアンロード |
| POST | `/api/v1/upload/image` | 最小I2V用画像をアップロードし、`image_id` を返す |
| POST | `/api/v1/generate` | 動画生成ジョブを開始し、ジョブIDを返す |
| GET | `/api/v1/jobs` | ジョブ一覧を返す |
| GET | `/api/v1/jobs/{job_id}` | ジョブ状態を返す |
| GET | `/api/v1/jobs/{job_id}/video` | 完了済み動画を返す |
| DELETE | `/api/v1/jobs/{job_id}` | 未開始/実行中ジョブのキャンセル、または結果削除 |

#### Phase 2以降で実装するエンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| PUT | `/api/v1/config` | 設定更新。Phase 1では未実装でもよい |
| GET | `/api/v1/jobs/{job_id}/preview` | 生成中プレビュー。実装可能なら追加 |
| POST | `/api/v1/upload/video` | V2V用動画アップロード |
| POST | `/api/v1/upload/image/keyframes` | 複数キーフレームI2V用。Phase 1では実装しない |

### 5.3 ステータスコード

| コード | 意味 |
|-------|------|
| 200 | 正常 |
| 201 | ジョブ作成成功 |
| 202 | ジョブ受付済み |
| 400 | リクエストパラメータ不正 |
| 401 | `--api-key` 有効時の認証失敗 |
| 404 | ジョブIDまたはファイルが存在しない |
| 409 | 実行中ジョブがある、またはパイプライン状態と要求が衝突 |
| 422 | Pydanticバリデーションエラー |
| 500 | 予期しない内部エラー |
| 503 | GPU OOM、モデルロード失敗、パイプライン実行失敗 |

---

## 6. Pydanticスキーマ

### 6.1 GenerateRequest

```python
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class CropOutput(BaseModel):
    width: int = Field(..., ge=32)
    height: int = Field(..., ge=32)

class ConditioningImage(BaseModel):
    image_id: str
    frame_idx: int = 0
    strength: float = Field(0.8, ge=0.0, le=1.0)
    crf: int | None = None

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = ""

    # LTXに渡す内部生成サイズ。必ず32の倍数。
    width: int = Field(512, ge=256, le=4096)
    height: int = Field(288, ge=128, le=4096)

    # 最終MP4のクロップサイズ。Noneならクロップしない。
    crop_output: CropOutput | None = None

    num_frames: int = Field(49, ge=9, le=257)
    frame_rate: float = Field(24.0, ge=1.0, le=60.0)
    num_inference_steps: int = Field(8, ge=1, le=100)
    guidance_scale: float = Field(1.0, ge=0.0, le=20.0)
    seed: int = -1
    pipeline: Literal["distilled", "two_stage_hq"] = "distilled"

    # 空配列ならT2V。1件ならPhase 1最小I2V。
    conditioning_images: list[ConditioningImage] = []

    @model_validator(mode="after")
    def validate_ltx_constraints(self):
        if self.width % 32 != 0:
            raise ValueError("width must be a multiple of 32")
        if self.height % 32 != 0:
            raise ValueError("height must be a multiple of 32")
        if (self.num_frames - 1) % 8 != 0:
            raise ValueError("num_frames must be 8n+1")
        if self.crop_output is not None:
            if self.crop_output.width > self.width:
                raise ValueError("crop_output.width must be <= width")
            if self.crop_output.height > self.height:
                raise ValueError("crop_output.height must be <= height")
        if self.pipeline == "distilled":
            if self.num_inference_steps != 8:
                raise ValueError("distilled pipeline requires num_inference_steps=8 in Phase 1")
            if self.guidance_scale != 1.0:
                raise ValueError("distilled pipeline requires guidance_scale=1.0 in Phase 1")

        # Phase 1最小I2V制約。
        if len(self.conditioning_images) > 1:
            raise ValueError("Phase 1 supports at most one conditioning image")
        if self.conditioning_images:
            image = self.conditioning_images[0]
            if image.frame_idx != 0:
                raise ValueError("Phase 1 supports only frame_idx=0 for I2V")
        return self
```

### 6.2 UploadImageResponse

```python
class UploadImageResponse(BaseModel):
    image_id: str
    original_filename: str
    stored_path: str
    width: int
    height: int
    content_type: str
```

Phase 1では、アップロード画像を `uploads/{image_id}/input.png` に正規化保存する。  
正規化時にEXIF orientationを反映し、RGBAはRGBへ変換する。

### 6.3 JobStatus

```python
from enum import Enum


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
```

Phase 1では `queued` は短時間しか使わない。ジョブ投入後、バックグラウンドタスク開始時に即 `running` へ移行する。

### 6.4 GenerateResponse

```python
class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
```

### 6.5 JobResult

```python
class JobResult(BaseModel):
    video_url: str
    duration_seconds: float
    resolution: str
    file_size_bytes: int
    generation_time_seconds: float
    seed_used: int
    output_path: str
    metadata_path: str
```

### 6.6 JobResponse

```python
class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    current_step: int | None
    total_steps: int | None
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None
    request: GenerateRequest
    result: JobResult | None
```

## 7. 主要リクエスト/レスポンス

### 7.1 POST `/api/v1/upload/image`

Phase 1で実装する。最小I2V用の画像をアップロードし、`image_id` を返す。

制約:

- 対応形式: `png`, `jpg`, `jpeg`, `webp`
- 最大ファイルサイズ: `config.yaml` の `upload.max_image_size_mb`
- 保存先: `uploads/{image_id}/input.png`
- ディレクトリトラバーサルを防止する
- `image_id` はUUIDとする

レスポンス例:

```json
{
  "image_id": "0f3d2c1b-9876-4321-aaaa-1234567890ab",
  "original_filename": "first_frame.png",
  "stored_path": "uploads/0f3d2c1b-9876-4321-aaaa-1234567890ab/input.png",
  "width": 1024,
  "height": 576,
  "content_type": "image/png"
}
```

### 7.2 POST `/api/v1/generate`

#### リクエスト例: T2V / Phase 1標準設定（16GB向け）

```json
{
  "prompt": "A flowing river in a forest at golden hour, cinematic, high detail",
  "negative_prompt": "blurry, low quality, distorted",
  "width": 512,
  "height": 288,
  "crop_output": null,
  "num_frames": 49,
  "frame_rate": 24.0,
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "seed": -1,
  "pipeline": "distilled",
  "conditioning_images": []
}
```

#### リクエスト例: 最小I2V / Phase 1対応

```json
{
  "prompt": "The character slowly turns their head, cinematic, high detail",
  "negative_prompt": "blurry, low quality, distorted",
  "width": 512,
  "height": 288,
  "crop_output": null,
  "num_frames": 49,
  "frame_rate": 24.0,
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "seed": 42,
  "pipeline": "distilled",
  "conditioning_images": [
    {
      "image_id": "0f3d2c1b-9876-4321-aaaa-1234567890ab",
      "frame_idx": 0,
      "strength": 0.8
    }
  ]
}
```

#### リクエスト例: 1080p最終出力用のパディング指定

Phase 4以降で使う想定。Phase 1ではVRAM不足の可能性が高いため、動作保証しない。

```json
{
  "prompt": "A futuristic city street at night, cinematic, high detail",
  "negative_prompt": "blurry, low quality, distorted",
  "width": 1920,
  "height": 1088,
  "crop_output": {
    "width": 1920,
    "height": 1080
  },
  "num_frames": 121,
  "frame_rate": 24.0,
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "seed": 42,
  "pipeline": "distilled",
  "conditioning_images": []
}
```

#### レスポンス例

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "created_at": "2026-06-25T12:00:00Z"
}
```

### 7.3 GET `/api/v1/jobs/{job_id}`

#### 実行中

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "running",
  "progress": 0.5,
  "current_step": 4,
  "total_steps": 8,
  "created_at": "2026-06-25T12:00:00Z",
  "started_at": "2026-06-25T12:00:02Z",
  "completed_at": null,
  "error": null,
  "request": {
    "prompt": "A flowing river in a forest at golden hour, cinematic, high detail",
    "negative_prompt": "blurry, low quality, distorted",
    "width": 512,
    "height": 288,
    "crop_output": null,
    "num_frames": 49,
    "frame_rate": 24.0,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": -1,
    "pipeline": "distilled",
    "conditioning_images": []
  },
  "result": null
}
```

#### 完了時

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "progress": 1.0,
  "current_step": 8,
  "total_steps": 8,
  "created_at": "2026-06-25T12:00:00Z",
  "started_at": "2026-06-25T12:00:02Z",
  "completed_at": "2026-06-25T12:03:10Z",
  "error": null,
  "request": {
    "prompt": "A flowing river in a forest at golden hour, cinematic, high detail",
    "negative_prompt": "blurry, low quality, distorted",
    "width": 512,
    "height": 288,
    "crop_output": null,
    "num_frames": 49,
    "frame_rate": 24.0,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": -1,
    "pipeline": "distilled",
    "conditioning_images": []
  },
  "result": {
    "video_url": "/api/v1/jobs/a1b2c3d4-e5f6-7890-abcd-ef1234567890/video",
    "duration_seconds": 2.04,
    "resolution": "512x288",
    "file_size_bytes": 4523008,
    "generation_time_seconds": 72.4,
    "seed_used": 42,
    "output_path": "outputs/a1b2c3d4-e5f6-7890-abcd-ef1234567890/output.mp4",
    "metadata_path": "outputs/a1b2c3d4-e5f6-7890-abcd-ef1234567890/metadata.json"
  }
}
```

### 7.4 GET `/api/v1/status`

```json
{
  "server": "running",
  "version": "0.4.0",
  "host": "127.0.0.1",
  "port": 18620,
  "pipeline_loaded": true,
  "pipeline_type": "distilled",
  "gpu": {
    "available": true,
    "name": "NVIDIA GeForce RTX 4070 Ti SUPER",
    "vram_total_mb": 16376,
    "vram_used_mb": 12800,
    "vram_free_mb": 3576
  },
  "vram_optimization": {
    "low_vram_mode": true,
    "low_vram_profile": "16gb_safe",
    "fp8_transformer": true,
    "cpu_offload_text_encoder": true,
    "vae_tiling": true,
    "attention_tiling": false,
    "block_swap": false,
    "low_vram_disabled_required": false
  },
  "queue": {
    "mode": "single_job_in_memory",
    "pending": 0,
    "running": 1,
    "completed": 5,
    "failed": 0
  }
}
```

## 8. ジョブ管理

### 8.1 Phase 1: in-memory single job

Phase 1ではDBや本格的なキューを使わない。  
`services/job_store.py` にメモリ上の辞書を持つ。

要件:

- `jobs: dict[str, JobRecord]` を保持する
- 実行中ジョブがある場合、新規 `POST /generate` は `409 Conflict`
- サーバー再起動でジョブ履歴は消えてよい
- `outputs/{job_id}/metadata.json` は残る
- `GET /jobs` はメモリ上に残っているジョブのみ返す
- 完了済みジョブの動画は `GET /jobs/{job_id}/video` で返す

### 8.2 ジョブ状態遷移

```text
[queued] → [running] → [completed]
                    └→ [failed]
                    └→ [cancelled]
```

Phase 1のキャンセルはベストエフォートでよい。  
PyTorch推論を安全に中断できない場合、以下の挙動でよい。

- `DELETE /jobs/{id}` を受け取ったら `cancel_requested=True` にする
- 推論完了後、キャンセル要求があった場合はステータスを `cancelled` にする
- 途中停止が実装できないことをREADMEに明記する

### 8.3 進捗更新

`services/ltx_runner.py` は、可能であればステップごとに `progress_callback(current_step, total_steps)` を呼ぶ。  
公式パイプラインから進捗が取れない場合は、Phase 1では以下の暫定値でよい。

- 生成開始: `progress=0.05`
- LTX推論中: `progress=0.10`
- ffmpegエンコード中: `progress=0.90`
- 完了: `progress=1.0`

Phase 3で正確なステップコールバックへ改善する。

---

## 9. Pipeline Manager

### 9.1 責務

`services/pipeline_manager.py` は以下を担当する。

- パイプラインのロード状態管理
- 初回生成時の自動ロード
- 明示的なロード/アンロードAPI
- OOM発生時の後始末
- GPUメモリ解放
- Low VRAM baseline設定の適用
- `low_vram_mode=false` が16GBでは必須でないことの扱い
- `ltx_runner.py` への実行委譲

### 9.2 ライフサイクル

```text
[unloaded]
    │ load()
    ▼
[loading]
    │ success
    ▼
[ready]
    │ generate()
    ▼
[running]
    │ success
    ▼
[ready]
    │ unload()
    ▼
[unloaded]
```

失敗時:

```text
[loading] or [running]
    │ exception / OOM
    ▼
[error]
    │ cleanup()
    ▼
[unloaded]
```

### 9.3 自動ロード方針

`POST /api/v1/generate` 時点でパイプライン未ロードなら、自動でロードする。  
ただし、ロードに失敗した場合は `503 Service Unavailable` を返し、ジョブは `failed` にする。

### 9.4 ltx_runner.py のインターフェース

公式 `ltx-pipelines` の実際のAPIは変わる可能性があるため、依存をこのファイルに閉じ込める。

```python
from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[int | None, int | None, float], None]

class LTXRunner:
    def __init__(self, config):
        self.config = config
        self.pipeline = None

    def load(self) -> None:
        """公式LTXパイプラインをロードする。"""
        ...

    def unload(self) -> None:
        """パイプラインを破棄し、GPUメモリを解放する。"""
        ...

    def generate(
        self,
        request: GenerateRequest,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """
        動画を生成し、最終MP4のPathを返す。
        request.conditioning_images が空ならT2V。
        request.conditioning_images が1件なら、その画像を開始フレーム条件として最小I2V。
        返すファイル名は output.mp4 とする。
        """
        ...
```

### 9.5 実装時の探索ルール

AIエージェントは、公式LTX-2リポジトリのサンプルコードを参照して `ltx_runner.py` を実装する。  
ただし、以下は変更しない。

- APIスキーマ
- `GenerateRequest`
- 出力ディレクトリ構造
- ジョブ状態レスポンス
- FastAPIエンドポイント名
- Phase 1のI2V制約（画像1枚、frame_idx=0）

公式パイプラインの戻り値が動画テンソル、フレーム列、ファイルパスのいずれであっても、`ltx_runner.py` 内で `output.mp4` に正規化する。

---

## 10. 出力ファイル仕様

### 10.1 ディレクトリ構造

```text
outputs/
└── {job_id}/
    ├── output.mp4
    ├── metadata.json
    ├── raw/                  # 任意。中間ファイル保存用
    └── preview.jpg           # Phase 3以降。任意
```

### 10.2 metadata.json

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "created_at": "2026-06-25T12:00:00Z",
  "started_at": "2026-06-25T12:00:02Z",
  "completed_at": "2026-06-25T12:03:10Z",
  "status": "completed",
  "request": {
    "prompt": "A flowing river in a forest at golden hour, cinematic, high detail",
    "negative_prompt": "blurry, low quality, distorted",
    "width": 512,
    "height": 288,
    "crop_output": null,
    "num_frames": 49,
    "frame_rate": 24.0,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": -1,
    "pipeline": "distilled",
    "conditioning_images": []
  },
  "generation_mode": "t2v",
  "seed_used": 42,
  "generation_time_seconds": 72.4,
  "output": {
    "path": "outputs/a1b2c3d4-e5f6-7890-abcd-ef1234567890/output.mp4",
    "resolution": "512x288",
    "duration_seconds": 2.04,
    "frame_rate": 24.0,
    "file_size_bytes": 4523008
  },
  "vram_optimization": {
    "low_vram_mode": true,
    "low_vram_profile": "16gb_safe",
    "fp8_transformer": true,
    "cpu_offload_text_encoder": true,
    "vae_tiling": true,
    "peak_vram_mb": 15120
  },
  "environment": {
    "python": "3.12.x",
    "torch": "2.7.x",
    "cuda": "12.8",
    "gpu": "NVIDIA GeForce RTX 4070 Ti SUPER"
  }
}
```

### 10.3 クロップ仕様

LTXに渡す内部生成サイズは32の倍数である必要がある。  
一方、AviUtl2素材としては `1920x1080` や `960x540` のような一般的な解像度が望ましい。  
そのため、`crop_output` が指定された場合は、ffmpegで中央クロップする。

例:

| 内部生成サイズ | 最終出力 | 用途 |
|---------------|----------|------|
| `960x544` | `960x540` | Phase 1検証用 |
| `1920x1088` | `1920x1080` | 1080p最終出力 |
| `1088x1920` | `1080x1920` | 縦長動画 |

ffmpeg例:

```bash
ffmpeg -i input.mp4 -vf "crop=960:540:(in_w-960)/2:(in_h-540)/2" -c:v libx264 -pix_fmt yuv420p output.mp4
```

---

## 11. config.yaml

```yaml
server:
  host: "127.0.0.1"
  port: 18620
  allow_all_cors: false
  api_key: null
  log_dir: "./logs"

model:
  # Phase 1では公式LTX-2リポジトリ/パッケージ側のロード方法に合わせる。
  checkpoint_dir: "./models"
  checkpoint_name: "ltx-2.3-22b-distilled"
  text_encoder: "google/gemma-2-2b-it"
  pipeline_type: "distilled"
  auto_load_on_generate: true
  reload_interval: 0

vram:
  # Phase 1から有効。主要開発環境がVRAM 16GBであるため。
  low_vram_mode: true
  low_vram_profile: "16gb_safe"

  # Phase 1 baseline。公式APIで利用できる範囲で有効化する。
  fp8_transformer: true
  cpu_offload_text_encoder: true
  vae_tiling: true

  # Phase 2以降の高度な独自最適化。Phase 1では無効。
  attention_tiling: false
  attention_tile_size: null
  block_swap: false
  block_swap_blocks_on_gpu: null

  # 高VRAM環境向けの任意検証用。16GBでの成功は保証しない。
  allow_disable_low_vram: true

generation_presets:
  smoke_test:
    width: 384
    height: 224
    crop_output: null
    num_frames: 17
  phase1_default:
    width: 512
    height: 288
    crop_output: null
    num_frames: 49
  phase1_target:
    width: 960
    height: 544
    crop_output:
      width: 960
      height: 540
    num_frames: 121

generation_defaults:
  # Gradio/APIの初期値。まず16GBで通しやすい軽量設定にする。
  width: 512
  height: 288
  crop_output: null
  num_frames: 49
  frame_rate: 24.0
  num_inference_steps: 8
  guidance_scale: 1.0
  seed: -1
  pipeline: "distilled"
  conditioning_images: []

upload:
  dir: "./uploads"
  max_image_size_mb: 20
  allowed_image_extensions: [".png", ".jpg", ".jpeg", ".webp"]
  normalize_to_png: true

limits:
  max_width: 1920
  max_height: 1088
  max_num_frames: 257
  max_conditioning_images_phase1: 1
  phase1_max_concurrent_jobs: 1
  low_vram_disabled_required: false

output:
  dir: "./outputs"
  format: "mp4"
  save_metadata_json: true
  keep_raw_frames: false
```

---

## 12. Gradio テストUI

### 12.1 Phase 1 機能

- プロンプト入力
- ネガティブプロンプト入力
- 任意の入力画像アップロード
- 画像strength指定。デフォルト `0.8`
- 幅・高さ入力。デフォルト `512x288`
- 最終クロップサイズ入力。デフォルト `null`
- フレーム数選択。デフォルト `49`
- プリセット選択: `smoke_test`, `phase1_default`, `phase1_target`
- Low VRAM設定表示。Phase 1ではデフォルト有効
- FPS指定。デフォルト `24`
- seed指定。`-1` でランダム
- 生成ボタン
- ジョブID表示
- 進捗表示
- 完了後の動画プレビュー
- `/api/v1/status` の簡易表示

### 12.2 UI動作

- 入力画像が空の場合、`conditioning_images=[]` でT2V生成する。
- 入力画像が指定された場合、Gradio UIは先に `POST /api/v1/upload/image` を呼ぶ。
- 画像アップロード成功後、返された `image_id` を `conditioning_images[0].image_id` に入れて `POST /api/v1/generate` を呼ぶ。
- Phase 1では `conditioning_images[0].frame_idx=0` 固定。
- Phase 1では `conditioning_images` は最大1件。

### 12.3 UI実装方針

Gradio UIは、Python関数内で直接LTXを呼ばない。  
必ず自分自身のREST APIを叩く。

理由:

- AviUtl2/DaVinci Resolve等の将来フロントエンドと同じ通信経路を検証できる
- API仕様の破綻を早期に見つけられる
- UIとバックエンド処理の結合を避けられる

Gradio側の処理:

```text
[Generate button]
    │
    ├─ input image がある場合のみ POST /api/v1/upload/image
    │      └─ image_id 取得
    │
    ├─ POST /api/v1/generate
    │      └─ job_id 取得
    │
    ├─ 1秒間隔で GET /api/v1/jobs/{job_id}
    │      └─ progress 更新
    │
    └─ completed になったら /api/v1/jobs/{job_id}/video を表示
```

### 12.4 FastAPIへのマウント

```python
import gradio as gr
from fastapi import FastAPI

app = FastAPI(title="LTX-AviUtl2-Bridge")
app.include_router(api_router, prefix="/api/v1")

gradio_app = gr.Blocks()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")
```

## 13. 低VRAM戦略（Phase 1から）

### 13.1 基本方針

Phase 1から `low_vram_mode=true` を標準にする。  
ただし、Phase 1では「確実に小さく動かす」ことを優先し、壊れやすい独自パッチは入れない。

Phase 1でのLow VRAM baseline:

1. FP8 checkpoint / FP8 transformer を優先する
2. text encoder CPU offload を有効にする
3. 公式APIでVAE tilingが使える場合は有効にする
4. 生成前後に安全なメモリ解放を行う
5. 生成プリセットを `smoke_test → phase1_default → phase1_target` の順に上げる
6. peak VRAM、生成時間、OOM有無をmetadataとログに記録する

Phase 2では、16GBで `960x544 / 121 frames / crop 960x540` をより安定させるため、以下の高度な最適化を段階導入する。

1. VAE tiling の独自調整
2. attention tiling
3. block swap
4. 生成ごとの自動リロード・断片化対策

### 13.2 検証プリセット

| プリセット | 内部サイズ | 最終出力 | フレーム数 | 位置づけ |
|------------|------------|----------|------------|----------|
| `smoke_test` | `384x224` | `384x224` | `17` | モデルロード後の最小疎通 |
| `phase1_default` | `512x288` | `512x288` | `49` | Phase 1の標準テスト |
| `phase1_target` | `960x544` | `960x540` | `121` | 16GBで狙う実用寄り目標 |
| `1080p_internal` | `1920x1088` | `1920x1080` | `121` | Phase 4以降。16GBでは必須にしない |

### 13.3 `low_vram_mode=false` の扱い

`low_vram_mode=false` は削除しない。  
ただし、これは高VRAM環境・クラウドGPU・将来の比較検証用であり、16GB開発環境で成功することを必須条件にしない。

16GB環境での検証順序:

1. `low_vram_mode=true` でモデルロード
2. `low_vram_mode=true` で `smoke_test`
3. `low_vram_mode=true` で `phase1_default`
4. `low_vram_mode=true` で `phase1_target`
5. 余力があれば `low_vram_mode=false` で `smoke_test`

5が失敗しても、Phase 1は失敗扱いにしない。

### 13.4 サービスの役割

#### LowVRAMService

Phase 1で追加する軽量サービス。  
`services/low_vram.py` に置き、以下を担当する。

- configからLow VRAM設定を読む
- FP8 checkpoint / FP8 transformer の利用を `ltx_runner.py` に伝える
- text encoder CPU offload の設定を `ltx_runner.py` に伝える
- VAE tiling が公式APIで使える場合に有効化する
- peak VRAM取得の補助
- 生成前後の安全なメモリ解放

#### FP8Service

Phase 1では、独自変換よりも公式FP8 checkpointや公式APIを優先する。  
独自変換が必要な場合はPhase 2以降に回す。

#### VAETileService

Phase 1では公式APIで利用できる場合のみ有効にする。  
公式APIで足りない場合の独自タイル合成はPhase 2以降に回す。

#### AttentionTileService

attention計算のクエリ長方向を分割する。  
PyTorch関数のグローバルモンキーパッチは壊れやすいため、Phase 1では実装しない。

#### BlockSwapService

transformerブロックの一部をCPU RAMへ退避し、必要なブロックだけGPUへ転送する。  
導入コストが高いため、Phase 2以降で最後に実装する。

### 13.5 ログに記録する項目

各生成で以下を記録する。

- `low_vram_mode`
- `low_vram_profile`
- `fp8_transformer`
- `cpu_offload_text_encoder`
- `vae_tiling`
- `attention_tiling`
- `block_swap`
- `peak_vram_mb`
- `generation_time_seconds`
- OOMが発生した場合の入力サイズとフレーム数

---

## 14. 1080p対応戦略（Phase 4以降）

### 14.1 方針

16GB VRAMで直接 `1920x1088` 生成するのは困難な可能性が高い。  
そのため、最終的には2段階方式を採用する。

1. Stage 1: `960x544` などの低解像度で生成
2. Stage 2: LTX公式の latent spatial upscaler で `1920x1088` へアップスケール
3. Stage 3: `1920x1080` へ中央クロップしてMP4出力

### 14.2 API上の扱い

Phase 4では `GenerateRequest` に以下を追加してよい。

```python
class UpscaleRequest(BaseModel):
    enabled: bool = False
    target_width: int = 1920
    target_height: int = 1088
    crop_width: int = 1920
    crop_height: int = 1080
```

Phase 1では追加しない。

---

## 15. AviUtl2プラグイン連携（Phase 5）

AviUtl2フロントエンドは、Phase 1の実装対象ではない。  
バックエンドAPIがT2V・最小I2V・ジョブ管理・出力保存まで安定してから、半ば独立した後続プロジェクトとして作成する。

### 15.1 役割分担

AviUtl2プラグインは薄いクライアントとする。

担当すること:

- プロンプト等の入力UI
- 必要に応じた画像アップロード
- `POST /api/v1/upload/image`
- `POST /api/v1/generate`
- `GET /api/v1/jobs/{job_id}` のポーリング
- 完了後のMP4取得
- タイムラインへの動画配置

担当しないこと:

- LTXモデルロード
- GPU管理
- 低VRAM最適化
- 動画生成処理
- ffmpegエンコード
- 複雑なI2V/V2V制御の直接実装

### 15.2 通信フロー

```text
[AviUtl2 .aux2プラグイン]
    │
    ├── 必要に応じて POST /api/v1/upload/image
    │       └── image_id取得
    │
    ├── POST /api/v1/generate
    │       └── job_id取得
    │
    ├── GET /api/v1/jobs/{job_id}
    │       └── 1秒間隔でポーリング
    │
    ├── GET /api/v1/jobs/{job_id}/video
    │       └── MP4取得
    │
    └── MP4をAviUtl2タイムラインへ配置
```

### 15.3 実装言語候補

| 言語 | 方式 | HTTP | 備考 |
|------|------|------|------|
| C++ | `.aux2` DLL | WinHTTP / libcurl | SDKに素直。開発コスト高め |
| C# | Native AOT DLL | HttpClient | HTTP実装が容易。AviUtl2との接続検証が必要 |

### 15.4 DaVinci Resolve等への横展開

バックエンドはAviUtl2専用にしない。  
DaVinci Resolve等の別フロントエンドは、同じREST APIを呼ぶ薄いクライアントとして実装する。

DaVinci Resolve向けの最初のMVPは、Python/Lua Scripting APIで以下のみを行う。

- localhostのバックエンドへ生成リクエストを送る
- ジョブ完了までポーリングする
- 完成したMP4をMedia Poolへ追加する
- 必要に応じて現在のTimelineへAppendする

OpenFX/OFXプラグイン化は初期MVPの範囲外とする。

## 16. 開発フェーズ

### Phase 1: 16GB-first 最小動作API + Gradio + 最小I2V

実装するもの:

- FastAPI起動
- Gradio UIマウント
- config.yaml読み込み
- `low_vram_mode=true` のデフォルト設定
- `services/low_vram.py` によるLow VRAM baseline設定
- `/api/v1/status`
- `/api/v1/config`
- `/api/v1/pipeline/load`
- `/api/v1/pipeline/unload`
- `/api/v1/upload/image`
- `/api/v1/generate`
- `/api/v1/jobs`
- `/api/v1/jobs/{job_id}`
- `/api/v1/jobs/{job_id}/video`
- in-memory single job管理
- upload_storeによる画像保存・正規化
- LTX 2.3 Distilled による小サイズT2V生成
- LTX 2.3 Distilled による最小I2V生成
- output.mp4保存
- metadata.json保存
- peak VRAM記録
- README作成

Phase 1のI2V制約:

- 入力画像は1枚のみ
- 開始フレーム conditioning のみ
- `frame_idx=0` 固定
- `strength` 指定のみ対応
- 複数キーフレーム、終了フレーム、V2V、IC-LoRAは実装しない

完了条件:

- `python main.py` で起動する
- `http://127.0.0.1:18620/ui` にアクセスできる
- `low_vram_mode=true` でモデルロードに成功する
- Gradio UIから画像なしでT2V生成できる
- Gradio UIから画像1枚ありで最小I2V生成できる
- `smoke_test` つまり `384x224 / 17 frames / 8 steps / CFG=1.0` でT2V生成できる
- `phase1_default` つまり `512x288 / 49 frames / 8 steps / CFG=1.0` でT2V生成できる
- `phase1_default` で最小I2V生成できる
- `outputs/{job_id}/output.mp4` が作成される
- `metadata.json` に `generation_mode`, `conditioning_images`, `peak_vram_mb`, `generation_time_seconds`, `seed_used`, `low_vram_mode` が保存される
- API経由でジョブ状態と動画取得ができる

目標条件:

- `phase1_target` つまり `960x544 / 121 frames / crop 960x540` のT2V生成を試す
- `phase1_target` の最小I2V生成を試す
- 成功/失敗にかかわらず、OOMやpeak VRAMをログに残す

任意条件:

- `low_vram_mode=false` で `smoke_test` を試す
- 失敗してもPhase 1の失敗とはみなさない

### Phase 2: 複雑なI2V/V2V・低VRAM高度化

実装するもの:

- 複数キーフレームI2V
- 終了フレーム conditioning
- 任意 `frame_idx` の画像conditioning
- V2V用動画アップロード
- IC-LoRA対応の調査・実装
- depth / pose / edge / motion tracking LoRA対応の調査・実装
- 公式APIで足りないVAE tilingの調整
- attention tiling
- block swap
- 各最適化のON/OFF設定
- VRAMプロファイリングログ
- OOM時の自動リカバリ
- N回生成ごとのパイプライン再ロード

完了条件:

- VRAM 16GB GPUで `phase1_target` が安定して生成できる
- 複数キーフレームI2VまたはV2Vの最小検証が通る
- `smoke_test`, `phase1_default`, `phase1_target` の検証ログがある
- 各最適化を個別にON/OFFして比較できる

### Phase 3: 本格ジョブキュー

実装するもの:

- `asyncio.Queue`
- 複数ジョブの待機
- キャンセル改善
- 正確な進捗コールバック
- 完了済みジョブ履歴の保持件数制限

完了条件:

- 複数ジョブを投入できる
- queued/running/completed/failed/cancelled が正しく返る
- Gradio UIの進捗表示が安定する

### Phase 4: 1080pアップスケール

実装するもの:

- latent spatial upscaler統合
- `960x544 → 1920x1088 → 1920x1080 crop`
- upscale有無のAPIオプション

完了条件:

- 1080p MP4が出力できる
- AviUtl2/DaVinci Resolve等に読み込める形式で保存される

### Phase 5: AviUtl2フロントエンド

実装するもの:

- `.aux2` プラグインプロトタイプ
- バックエンドURL設定
- プロンプト入力UI
- 画像アップロードUI
- ジョブ投入
- ポーリング
- MP4取得
- タイムライン配置

完了条件:

- AviUtl2上からT2V生成開始できる
- AviUtl2上から最小I2V生成開始できる
- 完了したMP4をタイムラインへ配置できる

## 17. ログとエラーハンドリング

### 17.1 ログ出力

ログは `logs/server.log` に出力する。

最低限記録するもの:

- サーバー起動時刻
- host / port / listen mode
- GPU名、VRAM総量
- パイプラインロード開始/完了/失敗
- ジョブID
- 入力解像度、フレーム数、seed
- 生成開始/完了/失敗
- 生成時間
- peak VRAM
- Low VRAM設定
- OOM発生時の例外

### 17.2 エラー形式

FastAPIの標準エラーに加えて、独自エラーでは以下を返す。

```json
{
  "error": {
    "code": "GPU_OOM",
    "message": "CUDA out of memory during generation",
    "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "detail": "Try smaller resolution or fewer frames."
  }
}
```

### 17.3 主なエラーコード

| code | HTTP | 意味 |
|------|------|------|
| `INVALID_RESOLUTION` | 400 | 幅・高さが32の倍数ではない |
| `INVALID_FRAME_COUNT` | 400 | フレーム数が8n+1ではない |
| `JOB_BUSY` | 409 | Phase 1で実行中ジョブがある |
| `UPLOAD_INVALID_TYPE` | 400 | 非対応の画像形式 |
| `UPLOAD_TOO_LARGE` | 400 | 画像ファイルサイズ超過 |
| `IMAGE_NOT_FOUND` | 404 | 指定された `image_id` が存在しない |
| `TOO_MANY_CONDITIONING_IMAGES` | 400 | Phase 1で複数画像が指定された |
| `JOB_NOT_FOUND` | 404 | ジョブが存在しない |
| `VIDEO_NOT_READY` | 409 | まだ動画が完成していない |
| `PIPELINE_LOAD_FAILED` | 503 | モデルロード失敗 |
| `GPU_OOM` | 503 | CUDA out of memory |
| `GENERATION_FAILED` | 503 | 推論中の一般エラー |

---

## 18. 受け入れテスト

### 18.1 API起動テスト

```bash
python main.py
curl http://127.0.0.1:18620/api/v1/status
```

期待:

- HTTP 200
- `server: running`
- `pipeline_loaded` がboolで返る

### 18.2 バリデーションテスト

幅が32の倍数でない場合:

```json
{
  "prompt": "test",
  "width": 1080,
  "height": 544,
  "num_frames": 121
}
```

期待:

- HTTP 422 または 400
- `width must be a multiple of 32` 相当のメッセージ

フレーム数が `8n+1` でない場合:

```json
{
  "prompt": "test",
  "width": 960,
  "height": 544,
  "num_frames": 120
}
```

期待:

- HTTP 422 または 400
- `num_frames must be 8n+1` 相当のメッセージ

Phase 1で複数画像が指定された場合:

```json
{
  "prompt": "test",
  "width": 512,
  "height": 288,
  "num_frames": 49,
  "conditioning_images": [
    {"image_id": "image-a", "frame_idx": 0, "strength": 0.8},
    {"image_id": "image-b", "frame_idx": 8, "strength": 0.8}
  ]
}
```

期待:

- HTTP 422 または 400
- `Phase 1 supports at most one conditioning image` 相当のメッセージ

Phase 1で `frame_idx=0` 以外が指定された場合:

```json
{
  "prompt": "test",
  "width": 512,
  "height": 288,
  "num_frames": 49,
  "conditioning_images": [
    {"image_id": "image-a", "frame_idx": 8, "strength": 0.8}
  ]
}
```

期待:

- HTTP 422 または 400
- `Phase 1 supports only frame_idx=0 for I2V` 相当のメッセージ

### 18.3 Low VRAM状態テスト

```bash
curl http://127.0.0.1:18620/api/v1/status
```

期待:

- HTTP 200
- `vram_optimization.low_vram_mode` が `true`
- `vram_optimization.low_vram_profile` が `16gb_safe`
- GPU名、総VRAM、空きVRAMが返る

### 18.4 画像アップロードテスト

期待:

- `POST /api/v1/upload/image` に `png`, `jpg`, `jpeg`, `webp` を送れる
- HTTP 200 または 201
- `image_id` が返る
- `uploads/{image_id}/input.png` が作成される
- 非対応形式は `UPLOAD_INVALID_TYPE`
- サイズ超過は `UPLOAD_TOO_LARGE`

### 18.5 最小T2V生成テスト: smoke_test

まず、16GB環境で最も軽い設定を検証する。

```json
{
  "prompt": "A red ball rolling on a white floor",
  "negative_prompt": "blurry, low quality",
  "width": 384,
  "height": 224,
  "crop_output": null,
  "num_frames": 17,
  "frame_rate": 24.0,
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "seed": 42,
  "pipeline": "distilled",
  "conditioning_images": []
}
```

期待:

- `POST /generate` が `job_id` を返す
- `GET /jobs/{id}` が running → completed へ遷移する
- `outputs/{job_id}/output.mp4` が作成される
- `metadata.json` が作成される
- `metadata.json` に `generation_mode=t2v`, `low_vram_mode=true`, `peak_vram_mb` が保存される
- `GET /jobs/{id}/video` でMP4が返る

### 18.6 Phase 1標準T2V生成テスト: phase1_default

```json
{
  "prompt": "A calm river flowing through a forest, cinematic, high detail",
  "negative_prompt": "blurry, low quality",
  "width": 512,
  "height": 288,
  "crop_output": null,
  "num_frames": 49,
  "frame_rate": 24.0,
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "seed": 42,
  "pipeline": "distilled",
  "conditioning_images": []
}
```

期待:

- 動画生成が完了する
- peak VRAMと生成時間が記録される

### 18.7 Phase 1標準I2V生成テスト: phase1_default + image

事前に `POST /api/v1/upload/image` で画像を1枚アップロードし、`image_id` を取得する。

```json
{
  "prompt": "The scene slowly comes alive, subtle camera movement, cinematic",
  "negative_prompt": "blurry, low quality",
  "width": 512,
  "height": 288,
  "crop_output": null,
  "num_frames": 49,
  "frame_rate": 24.0,
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "seed": 42,
  "pipeline": "distilled",
  "conditioning_images": [
    {
      "image_id": "<uploaded_image_id>",
      "frame_idx": 0,
      "strength": 0.8
    }
  ]
}
```

期待:

- 動画生成が完了する
- `metadata.json` に `generation_mode=i2v` が保存される
- `metadata.json` に使用した `image_id`, `frame_idx=0`, `strength=0.8` が保存される
- peak VRAMと生成時間が記録される

### 18.8 Phase 1目標生成テスト: phase1_target

```json
{
  "prompt": "A calm river flowing through a forest, cinematic, high detail",
  "negative_prompt": "blurry, low quality",
  "width": 960,
  "height": 544,
  "crop_output": {
    "width": 960,
    "height": 540
  },
  "num_frames": 121,
  "frame_rate": 24.0,
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "seed": 42,
  "pipeline": "distilled",
  "conditioning_images": []
}
```

期待:

- 成功した場合は `960x540` のMP4が保存される
- 失敗した場合も、OOM内容、peak VRAM、入力サイズがログに残る
- Phase 1では、このテストの失敗だけで実装失敗とはみなさない

### 18.9 Low VRAM disabled 任意テスト

高VRAM環境やクラウドGPUでのみ試す。

期待:

- `low_vram_mode=false` で `smoke_test` を試せる
- 16GB環境で失敗してもPhase 1失敗扱いにしない

### 18.10 Gradioテスト

期待:

- `/ui` が開く
- 画像なしでT2V生成開始できる
- 画像1枚ありで最小I2V生成開始できる
- ジョブIDが表示される
- 完了後に動画が表示される

## 19. ライセンス

| 対象 | ライセンス | 条件 |
|------|-----------|------|
| LTX-Desktop-LOW-VRAM 参考元 | Apache-2.0 | 改変・再配布・商用利用可。ライセンス同梱・変更箇所明記が必要 |
| LTX-Desktop 公式upstream | Apache-2.0 | 同上 |
| LTX-2公式コード | Apache-2.0想定。実際のリポジトリに従う | LICENSEを確認すること |
| LTX-2.3 モデル重み | LTX-2 Community License | モデルカードとLICENSEを確認すること |
| 本プロジェクト | Apache-2.0推奨 | 上流コード利用時の整合性を優先 |

---

## 付録A: LTX制約事項

- 幅・高さは32の倍数にする
- フレーム数は `8n+1` にする。例: `9, 17, 25, 33, ... 121`
- DistilledモデルはPhase 1では `8 steps / CFG=1.0` に固定する
- Phase 1のI2Vは画像1枚・開始フレーム・`frame_idx=0` のみ対応する
- プロンプトは英語推奨
- 1080pジャストのように32の倍数ではないサイズは、内部生成サイズをパディングし、最終出力時にクロップする

## 付録B: 16GB VRAM検証方針

16GB環境では、`low_vram_mode=true` を標準にする。  
`low_vram_mode=false` は高VRAM環境向けの任意検証であり、16GBで成功しなくてもよい。

検証順序:

1. モデルロードのみ
2. `smoke_test`: `384x224 / 17 frames`
3. `phase1_default`: `512x288 / 49 frames`
4. `phase1_default` I2V: `512x288 / 49 frames` + 画像1枚
5. `phase1_target`: `960x544 / 121 frames / crop 960x540`
6. 任意: `low_vram_mode=false` の `smoke_test`

記録する値:

| 項目 | 説明 |
|------|------|
| `peak_vram_mb` | 生成中のピークVRAM |
| `generation_time_seconds` | 生成時間 |
| `low_vram_mode` | 有効/無効 |
| `low_vram_profile` | 例: `16gb_safe` |
| `fp8_transformer` | FP8利用有無 |
| `cpu_offload_text_encoder` | text encoder CPU退避有無 |
| `vae_tiling` | VAE tiling有無 |
| `generation_mode` | `t2v` または `i2v` |
| `conditioning_images` | I2V時の `image_id`, `frame_idx`, `strength` |
| `oom_error` | OOM発生時のメッセージ |

実際のVRAM使用量は環境依存のため、必ずログとmetadataで確認する。

## 付録C: Codexへの最初の作業指示例

```text
この仕様書の Phase 1 のみを実装してください。
主要開発環境はVRAM 16GBなので、low_vram_mode=true をデフォルトにしてください。
Phase 1ではT2Vと最小I2Vを同一MVPとして実装してください。

まずディレクトリ構成、FastAPI、Pydanticスキーマ、in-memory JobStore、UploadStore、Gradio UIを作ってください。
次に services/low_vram.py と services/ltx_runner.py を作り、公式LTX-2.3 Distilled pipelineを16GB向け設定で呼び出してください。

API仕様、出力ディレクトリ構造、GenerateRequestのバリデーションは仕様書どおりに固定してください。
画像なしの場合は conditioning_images=[] としてT2V生成してください。
画像ありの場合は /api/v1/upload/image で画像を保存し、conditioning_images[0] に image_id, frame_idx=0, strength を入れて最小I2V生成してください。
Phase 1では conditioning_images は最大1件、frame_idx は0固定です。

Phase 1の必須テストは以下です。
- smoke_test T2V: 384x224 / 17 frames
- phase1_default T2V: 512x288 / 49 frames
- phase1_default I2V: 512x288 / 49 frames + 画像1枚

BlockSwap、AttentionTileの独自パッチ、複数キーフレームI2V、V2V、IC-LoRA、AviUtl2プラグイン、DaVinci Resolveフロントエンド、1080pアップスケールはまだ実装しないでください。
low_vram_mode=false は任意検証であり、16GB環境で失敗してもPhase 1失敗扱いにしないでください。
最後にREADMEへセットアップ手順、起動手順、16GB向け制限事項、T2V/I2Vの最小生成テストを書いてください。
```
