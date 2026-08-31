# Nz-Videomni バックエンド API リファレンス

最終更新: 2026-08-31 / 出典: 調査エージェント報告(2026-07-07)。エラーコード一覧・§3.12・§3.13は、モデル読み込み中の`/generate`・`/generate/chain`がジョブ作成前に同期で`409 PIPELINE_LOADING`を返すようになったことに伴う追記(2026-08-31。バックエンド`Videomni_Backend_Specification.md` §6.9(f)、実装は`DEVLOG.md` §98)。§12はフロントエンド実装完了(1.0.0-rc1)に伴う追記(2026-07-08)。§5.2・§12.4はバックエンドのchain拡張(clips24枠・chunked_upsample・chainへのloras/reference_video_id全面解禁)とポーリングタイムアウト延長に伴う追記・修正(2026-07-15)。§12.2は「未解消の穴」がN2/N3実装で解消済みになったことの追記(2026-07-17)。§5.1・§5.2・§7はNAG（Normalized Attention Guidance）解禁(バックエンドコミット2ae497b)に伴う追記(2026-07-28)。§10はバックエンド新設のMCPサーバー(`mcp_server/`、このAPIのもう一つのクライアント)に伴う追記(2026-07-28)。§5.1・§5.2・§7はVSF（Value Sign Flip。NAGに続く2つ目の非CFGネガティブプロンプト手法）の`neg_method`/`vsf_scale`追加とフロントエンド追随完了に伴う追記(2026-07-29)。§3.1・§5.1・§5.2・§7はAcceleration（生成の高速化。`attention_backend`＝SageAttentionのジョブ単位切替と、モックの`vae_mode`）の追加とフロントエンド追随・実機ゲート合格に伴う追記(2026-08-01)。§5.1・§5.2はモデル骨格のジョブ間キャッシュ(`keep_resident`。前処理の待ち時間を2本目以降から消す代わりにメインメモリを約20GB占有する切替)の追加とフロントエンド追随・エージェント実機ゲート合格に伴う追記(2026-08-03)。§5.1・§5.2・§7.1はGGUF逆量子化の1カーネル化(`fused_gguf_dequant_kernel`。生成結果を変えずに約17.5%短縮する切替)の追加と既定on反転に伴う追記と修正(2026-08-04)。§3.2・§5.2はRetake(`retake`)とStage-2クリップ長連動の窓上限を追記(2026-08-10)。§5.2・§5.4は長尺A2V（クリップ連結でのA2V＝音声から動画を生成する機能の複数クリップ解禁。`source_audio`の「clips=1本限定」撤廃）に伴う修正(2026-08-10)。§3.6・§5.2・§7はチェーン（`/generate/chain`）の参照動画（IC-LoRA）の複数クリップ解禁（長尺IC-LoRA。`reference_video_id`の「clips=1本限定」撤廃と`depth-control`専用の新エラーコード`LORA_DEPTH_CHAIN_UNSUPPORTED`追加、`GET /loras`への`preprocess`/`reference_downscale_factor`追加）に伴う追記(2026-08-11)。§2・§3.10・§5.2はEnd source（素材（末尾）。`/generate/chain`の`end_source`と、`POST /upload/video`応答の`frame_count`/`fps`）の追加に伴う追記(2026-08-16)。§5.2は同機能の**窓内モード**化（クリップ1本＝素材をクリップ自身の末尾として凍結し出力尺は変わらない・推奨／2本以上＝旧方式・非推奨）に伴う全面書き換え(2026-08-17)。§5.2は錨の固定強度`strength`（既定1.0・Stage-1のみソフト化可能・Stage-2は常にハード凍結）追加に伴う追記(2026-08-18)。§5.2・§5.4は同機能の逆順Chained化（クリップ2本以上＝`reverse`。出力尺はクリップ合計のまま・新422・旧方式は到達不能化）に伴う全面改稿(2026-08-18)。§5.2は同機能の第3弾（錨への素材音声の凍結。素材に音声トラックがあれば窓内モード・逆順Chainedいずれの錨クリップでも常にその音声を凍結し、`strength`は映像専用のつまみになった）に伴う追記(2026-08-18)。§3.2はSingleタブの賢い快適上限マーカー（`limits.single_comfort_token_budget`＝44880の新設。5つの高速化トグル全onのときだけ有効で、1つでもoffなら`spill_free_frames`が正）の追加に伴う追記(2026-08-18)。§3.5・§5.1・§5.2はLTX 2.5の高速化第3弾(SageAttention＝`attention_backend`)の開通に伴う追記・修正(2026-08-25。`unsupported_features`は6件へ、単発の422系は4個・連結の422系は5個へ。正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §77。**オーナーの目視ゲートG8は2026-08-25に合格し、その記録は同 §77.10**)。§5.1の`attention_backend`欄は、そのオーナー目視ゲートG8の合格(2026-08-25)を反映して更新した。§3.5・§5.1・§5.2・§5.4は撮り直し(Retake)と素材（末尾）(End source)のLTX 2.5開通に伴う修正(2026-08-26。`unsupported_features`は**4件**へ、連結の422系は**3個**へ〔全34件＝422系3／無視5／動作23／従属3〕。単発側の4分類は無変更。**連結生成のスキーマが表現できるモードは、これで全部LTX 2.5でも通る**。正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §78。**オーナーの目視ゲートG8は2026-08-30に合格＝同 §78.14**)。§3.5・§5.1は画角拡張(Outpainting)のLTX 2.5開通に伴う修正(2026-08-29。`unsupported_features`は**3件**へ、単発の422系は**3個**・動作は**17個**へ〔全28件＝422系3／無視5／動作17／従属3〕。連結側の4分類は無変更。**これで単発生成の側にも連結生成の側にも、断っている「モード」は1つも残っていない**。正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §79。**オーナーの目視ゲートG8′は2026-08-29に合格＝同 §79.11**)。§3.5・§5.1・§5.2は非CFGネガティブプロンプト(NAG／VSF)のLTX 2.5開通に伴う修正(2026-08-30。`unsupported_features`は**2件**へ、単発・連結とも422系は**2個**・無視は**2個**・動作は**24個／30個**・従属は**0個**へ〔単発 全28件＝422系2／無視2／動作24／従属0、連結 全34件＝422系2／無視2／動作30／従属0〕。**7つのフィールドが3つの分類から同時に「動作」へ移り、「従属」は両スキーマとも空になった**〔**空でも表は残す**〕。正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §80。**オーナーの目視ゲートG8は2026-08-30に合格＝同 §80.10**)。§3.2は快適上限マーカーのエンジン系統〔ベースモデルの世代〕別配信テーブル`limits.comfort_budgets`の新設に伴う追記(2026-08-31。従来の`single_comfort_token_budget`／`chain_comfort_token_budget`は表を持たない古いサーバー向けの互換値へ降格。線の正本はバックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §1.1、較正の記録は[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §84)。

関連ドキュメント: [SDK_REFERENCE.md](SDK_REFERENCE.md) ／ [WEB_RESEARCH.md](WEB_RESEARCH.md) ／ [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) ／ [BRIDGE_CONTRACT.md](BRIDGE_CONTRACT.md) ／ [DEVLOG.md](DEVLOG.md)

対象: `Nz-Videomni`（REST API）。**基本方針は凍結**（フロントエンド側の都合でバックエンドAPIを変えない）だが、**V2V結合(Join)機能の復活のために限定的な解除・拡張を行った実績がある**（`POST /jobs/{job_id}/join` の `source_tail_seconds`、応答の `trimmed_source_seconds`／`source_fps` など。本文§3.17・§3.18・§6に反映済み）。したがって「変更禁止」ではなく「必要が確定した箇所だけをオーナー承認のうえで拡張する」という運用である。本ドキュメントはフロントエンド(`.aux2`プラグイン)実装のための一次参照であり、`gradio_ui/api_client.py` ・ `gradio_ui/handlers.py` ・ `Videomni_Backend_Specification.md` ・ [`../Mock/AVIUTL2_DESIGN_BRIEF.md`](../Mock/AVIUTL2_DESIGN_BRIEF.md)（デザインブリーフの生きた正本・**必読**。モノレポ直下の`Docs/AVIUTL2_DESIGN_BRIEF.md`（凍結スナップショット。v2時点で凍結、オーナー決定2026-08-11）は別物）を根拠資料とする。

---

## 0. 全体像

- **プロトコル**: HTTP REST(JSON)のみ。**WebSocket / SSE は無し**。進捗取得は**1秒間隔のポーリング**。
- **2プロセス・2venv構成**(バックエンド `README.md` §0「環境分離ポリシー」の「2つの venv」)。
  - `./.venv`(torch無し): FastAPIアプリ本体。API・ジョブ管理・Gradio検証UI・モックbackend。
  - `./.venv-engine`(torch+cu128): 実推論worker。アプリが `subprocess` として自動spawnするため、**フロントエンドは engine を直接意識する必要はない**。
- **backend選択**: `config.model.backend` = `auto` / `mock` / `real`(`config.yaml`の`model.backend`)。`auto`はGPU+モデルがあれば`real`、無ければ`mock`(合成クリップ)。**モックでもAPI・スキーマ・出力構造は実物と同一**なので、フロントはbackendの別を意識せず開発できる。開発中のE2E確認は原則mockバックエンドで行う。
- 一次資料: `Videomni_Backend_Specification.md`(API契約詳細)、`Docs/RESOLUTION_DURATION_CAPABILITY.md`(解像度別性能実測)、[`../Mock/AVIUTL2_DESIGN_BRIEF.md`](../Mock/AVIUTL2_DESIGN_BRIEF.md)(本フロントエンド専用の設計ブリーフ。§5「変えてはいけない制約」、§4「性能の現実」が必読)。

## 1. 起動・接続

### 起動方法
- **標準（エンドユーザー向け）**: リポジトリ直下の `setup.bat` で導入し、`run.bat` で起動する（バックエンド `README.md` §1「かんたんインストール」・§2「起動」）。`run.bat` は `run.ps1` を呼ぶだけの薄いラッパー。
- **推奨（開発者向け）**: `run.ps1` を直接実行する。環境変数(`UV_PYTHON_INSTALL_DIR`, `PYTORCH_CUDA_ALLOC_CONF`)の設定と `tools/` の `PATH` 前置を済ませてから、末尾で `.venv\Scripts\python.exe main.py @Args` を実行する。
- 直接: `.\.venv\Scripts\python.exe main.py`。
- real backend選択時、アプリが `.venv-engine\Scripts\python.exe -u -m engine.worker` を自動spawn（バックエンド `README.md` §3「アーキテクチャ」）。フロント側からの手動起動は不要。

### 接続先
- **既定ホスト/ポート: `127.0.0.1:18620`**(`config.yaml`の`server.host`／`server.port`、`config.py::ServerConfig`)。
- APIベースパス: **`/api/v1`**(`main.py::build_app` の `app.include_router(api_router, prefix="/api/v1")`)。全エンドポイントは `http://127.0.0.1:18620/api/v1/...`。
- 付随ルート: `GET /` → `/ui` に307リダイレクト(`main.py::register_root_route`)、`/docs`(Swagger UI)、`/ui`(Gradio検証UI)、`/favicon.ico`。
- **Swagger(`/docs`)と OpenAPI(`/openapi.json`)が使える**ので、フロント開発時のスキーマ確認に有用。

### CLIオプション(`main.py::parse_args`、バックエンド `README.md` §2「CLI オプション」)

| オプション | 意味 |
|---|---|
| `--listen` | `0.0.0.0`にバインド(家庭内LAN公開) |
| `--port <n>` | ポート変更 |
| `--api-key <key>` | Bearer認証を要求 |
| `--allow-all-cors` | CORS全許可 |
| `--config <path>` | config.yamlパス指定 |
| `--te-offload`/`--no-te-offload`, `--dit-cpu-load`/`--no-dit-cpu-load` | VRAM最適化(フロント無関係) |

### 認証(`api/deps.py::require_auth`)
- `config.server.api_key`(既定 `null`)が設定されている場合のみ、**書き込み系エンドポイント**に `Authorization: Bearer <key>` ヘッダが必要。未設定なら認証不要。
- 認証が掛かるのは `require_auth` 依存を持つエンドポイントのみ = `/generate`, `/generate/chain`, `/upload/*`, `/pipeline/load`, `/pipeline/unload`, `/jobs/{id}/join`, `DELETE /jobs/{id}`。**GET系(status/config/job/loras/models)は認証不要**。
- 認証失敗は `401` + `{"error":{"code":"UNAUTHORIZED",...}}`。

### CORS(`main.py`の`LOCALHOST_CORS_REGEX`と`build_app`のCORSミドルウェア登録)
- 既定は `http://127.0.0.1` / `http://localhost`(任意ポート)のみ許可。WebView2からのCORS制約を回避するため、フロント設計はネイティブ側HTTPプロキシ経由を基本とする（詳細は [SDK_REFERENCE.md](SDK_REFERENCE.md) 参照）。

## 2. エラー形式(全エンドポイント共通)

`api/errors.py` により、全ドメインエラーは以下のエンベロープで返る(`main.py::register_exception_handlers`、`errors.py::APIError.to_envelope`):

```json
{ "error": { "code": "STRING_CODE", "message": "...", "job_id": "...(任意)", "detail": "...(任意)" } }
```

バリデーションエラー(422)は特別で(`main.py::register_exception_handlers`内の`_validation_handler`):
```json
{ "error": { "code": "VALIDATION_ERROR", "message": "Request validation failed",
  "detail": [ {"loc": [...], "msg": "...", "type": "..."} ] } }
```

### エラーコード一覧(`api/errors.py`)

| code | HTTP | 発生条件 |
|---|---|---|
| `JOB_BUSY` | 409 | 既にジョブ実行中(同時1ジョブ制約) |
| `UPLOAD_INVALID_TYPE` | 400 | 非対応画像/ファイル形式 |
| `UPLOAD_TOO_LARGE` | 400 | サイズ超過 |
| `IMAGE_NOT_FOUND` | 404 | image_id不明 |
| `REFERENCE_VIDEO_NOT_FOUND` / `SOURCE_VIDEO_NOT_FOUND` / `SOURCE_AUDIO_NOT_FOUND` / `END_SOURCE_NOT_FOUND` | 404 | アップロードIDが解決できない(`END_SOURCE_NOT_FOUND`は動画素材のみ、§5.2) |
| `SOURCE_VIDEO_TOO_SHORT` / `SOURCE_AUDIO_TOO_SHORT` / `END_SOURCE_TOO_SHORT` | 422 | 素材が短すぎる(`END_SOURCE_TOO_SHORT`は動画素材のみ、§5.2) |
| `LORA_NOT_FOUND` | 404 | 未登録LoRA名 |
| `LORA_REQUIRES_REFERENCE` | 422 | 制御LoRAに参照動画が無い |
| `REFERENCE_REQUIRES_CONTROL_LORA` | 422 | 参照動画が指定されているのに制御系LoRAが1本も無い（`LORA_REQUIRES_REFERENCE`の逆方向。単発・チェーン共通。2026-08-03新設） |
| `LORA_PREPROCESS_CONFLICT` | 400 | 参照動画1本に複数種の前処理要求 |
| `LORA_DEPTH_CHAIN_UNSUPPORTED` | 422 | チェーン（2クリップ以上）で`depth-control`アダプタを要求した（**2026-08-11新設**。深度前処理〔Video-Depth-Anything〕が全編一括設計でメモリに載らないための制限。depth以外の制御系〔canny/pose〕・参照系〔upscaler/deblur〕は多クリップで受理される。クリップ1本のチェーンと単発生成はdepth込みで従来どおり使える。旧`LORA_CONTROL_UNSUPPORTED_IN_CHAIN`〔2クリップ以上での制御系IC-LoRA全体拒否〕を置換） |
| `LORA_THUMBNAIL_NOT_FOUND` | 404 | サムネイル無し |
| `REFERENCE_RESOLUTION_INVALID` | 422 | 参照動画ジョブで width/height が128の倍数でない |
| `MODEL_NOT_FOUND`/`MODEL_FILE_MISSING`/`MODEL_INCOMPATIBLE` | 404/422/422 | モデル選択関連 |
| `JOB_NOT_FOUND` | 404 | job_id不明 |
| `VIDEO_NOT_READY` | 409 | 完了前に動画取得 |
| `JOB_NOT_JOINABLE`/`JOIN_FAILED`/`JOINED_NOT_READY` | 422/503/404 | V2V結合関連 |
| `PIPELINE_LOAD_FAILED` | 503 | モデルロード失敗 |
| `PIPELINE_LOADING` | 409 | 読み込み中のパイプラインへ重ねてロードを要求した。**2026-08-31から`/generate`・`/generate/chain`も、ジョブを作る前に同期でこれを返す**（バックエンド`Videomni_Backend_Specification.md` §6.9(f)。詳細は§3.12・§3.13） |
| `GPU_OOM` | 503 | 生成中VRAM不足 |
| `GENERATION_FAILED` | 503 | 生成失敗 |
| `UNAUTHORIZED` | 401 | APIキー不一致 |

## 3. 全エンドポイント仕様

ルーター登録: `api/router.py`。全て `/api/v1` プレフィックス配下。

### 3.1 `GET /status`(`api/status.py::get_status`)— 認証不要

サーバー・GPU・VRAM・キュー状態。

```json
{
  "server": "running", "version": "0.4.0",
  "host": "127.0.0.1", "port": 18620,
  "pipeline_loaded": false, "pipeline_type": null,
  "gpu": { "available": true, "name": "NVIDIA ...",
           "vram_total_mb": 16376, "vram_used_mb": 1234, "vram_free_mb": 15000 },
  "vram_optimization": { "low_vram_mode": true, "low_vram_profile": "16gb_safe",
           "fp8_transformer": true, "cpu_offload_text_encoder": true,
           "vae_tiling": true, "attention_tiling": false, "block_swap": false,
           "low_vram_disabled_required": false },
  "acceleration": { "attention_backends": ["sdpa", "sage"], "sage_available": true },
  "queue": { "mode": "single_job_in_memory",
             "pending": 0, "running": 0, "completed": 0, "failed": 0 }
}
```

- `gpu`ブロックの形は `services/gpu_info.py::get_gpu_info`。GPU無し/torch無しなら `available:false`, `vram_*_mb:0`。
  - **注意（既知・不具合ではない）**: 実機でも `gpu.available` は**常に`false`**になる。バックエンドは「アプリ用仮想環境（torch無し）＋エンジン用仮想環境（torch+CUDA）」の2プロセス構成で、`/status`を返すのは前者だから。実際のGPUはエンジン側のworkerプロセスが握っている。フロントはこの値をGPUの有無の判定に使わないこと。
- `acceleration`ブロック（**2026-07-31追加**、`services/pipeline_manager.py::acceleration_status_block`）。生成の高速化（Acceleration）機能の能力公開。凍結対象ではない。
  - `attention_backends`: `attention_backend`（§5.1）が受け付ける値の一覧。**実装のある項目だけ**が並ぶ。モックの`vae_mode`は載らない（§7参照）。
  - `sage_available`: SageAttention（注意機構を速くする外部カーネル）がサーバー側で使えるか。判定経路はサーバーの状態で変わる——mockなら常に`false`、パイプライン未ロードならエンジン用仮想環境のファイル存在チェック、ロード済みならworkerプロセス自身のimport判定（こちらが権威）。どちらの経路かは同じ応答の`pipeline_loaded`で判別できる。
  - **フロントの作法**: `false`のときは`sage`を選べないようにしてよい（現行の`SettingsPanel`はそうしている）。ただし**値が取れない・不明（`undefined`）のときは`sage`を封じないこと**——実際に使えなくてもサーバーが`sdpa`へ降格して完走するため、封じると無駄に機能を奪うことになる。`StatusResponse`側も`acceleration?`のoptionalとして扱う（`StatusResponse`は`/status`の部分型である点に注意）。
- **注意**: `pipeline_loaded` は「今ロード済みか」。real backendでは初回生成時に自動ロードされる(`config.yaml`の`model.auto_load_on_generate`、既定`true`)。
- フロントの起動シーケンスの起点はここ。疎通失敗＝「サーバー未起動」表示。

### 3.2 `GET /config`(`api/status.py::get_config`)— 認証不要

`config.py:AppConfig` 全体を `model_dump()` して返す。**フロントはここから制約値・プリセット・上限を動的に取得すべき（ハードコード禁止）**。特に:

- `generation_presets`(`config.yaml`のトップレベル同名キー): `smoke_test`/`minimal`/`small`/`standard_720p`/`FHD_1080p`/`WQHD_1440p` の6種。各 `width/height/crop_output/num_frames`。
- `generation_defaults`(`config.yaml`のトップレベル同名キー): UI初期値(width 512, height 320, num_frames 49, frame_rate 24.0, steps 8, guidance 1.0, seed -1)。
- `limits`(`config.yaml`のトップレベル同名キー、`config.py::LimitsConfig`): `max_width 1920`, `max_height 1088`（いずれも`config.py`のPydantic既定値。同梱`config.yaml`/`config.yaml.example`は4096）, `max_num_frames 481`, `max_conditioning_images 5`, `conditioning_frame_idx_multiple 8`, `conditioning_keyframe_grid_offset 1`, `phase1_max_concurrent_jobs 1`, `spill_free_frames`(解像度別の快適上限フレーム数), `v2v_context_frames_default/min/max`(73/25/145), `retake_window_min_frames`(73)/`retake_window_max_frames`(169、2026-08-10追加)、`comfort_budgets`(快適上限マーカーのエンジン系統別の配信テーブル、2026-08-31追加。下記参照)、`chain_comfort_token_budget`(40000、2026-08-12追加。下記参照)、`single_comfort_token_budget`(44880、2026-08-18追加。下記参照)、`end_context_frames_default/min/max`(72/8/136、2026-08-16追加。§5.2参照)。**上限169は`stage2_window="standard"`時の値**——`high_resolution`(潜在19)ではサーバーが`chain_math.retake_max_window_px(v_tile)`で実上限145まで検証する(公開値の169より狭い側に効く)。
- `upload`(`config.yaml`のトップレベル同名キー): 各種サイズ上限・許可拡張子。
- **`spill_free_frames`**(生成サイズの文字列 → 快適上限フレーム数の対応表)は「これを超えると2-4倍遅くなる(OOMせず)」の警告閾値。フロントで警告表示に使う。**`comfort_budgets`に一致する行が無いときのフォールバックがこれである**(下記)。**値そのものはここへ書かない**——実体は`config.yaml`の`limits.spill_free_frames`、説明の正本はバックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §付記(2026-08-31に再測定)。フロントは`webui/src/modes/single/spillUtils.ts`の`resolveSpillFreeFrames`(面積が最も近い鍵への丸め込み)を通して読む。
- **`comfort_budgets`**(2026-08-31追加)は「快適上限マーカーの線を、エンジン系統〔ベースモデルの世代〕ごとに配信する表」。**この鍵が来ているなら、`single_comfort_token_budget`／`chain_comfort_token_budget`ではなくこちらが正である。**

  ```jsonc
  "comfort_budgets": {
    "ltx":   { "spatial_factor": 32, "temporal_factor": 8,
               "rows": [ { "requires": { "attention_backend": "sage", "block_swap_prefetch": true,
                                         "keep_resident": true, "fused_gguf_dequant_kernel": true,
                                         "vae_mode": "prune_vaed" },
                           "single_budget": 44880, "chain_budget": 40000 } ] },
    "ltx25": { "spatial_factor": 32, "temporal_factor": 8,
               "rows": [ { "requires": {}, "single_budget": 44880, "chain_budget": 44880 } ] }
  }
  ```

  - **外側の鍵はエンジン系統のid**——`GET /models`の`base_models[].engine_family`(§3.5)と同じ語彙で、`"ltx"`＝LTX 2.3系、`"ltx25"`＝LTX 2.5系である。
  - **`requires`はサーバー語彙**（リクエストのフィールド名`attention_backend`・`block_swap_prefetch`・`keep_resident`・`fused_gguf_dequant_kernel`・`vae_mode`をそのまま使う）。フロントは実効の高速化設定を同じ語彙へ写してから照合する(`webui/src/shell/accelerationSettings.ts`の`effectiveAccelerationFields`。keep_residentはblock_swap_prefetchの実効状態で畳み込んだ後の値を使う)。
  - **照合は行の上から順で、全ての鍵が一致した最初の行を採る。** `requires`が空の行はどんな設定でも一致する。**フロントが知らない鍵を要求する行は不一致**として扱う（安全側へ倒す）。
  - **一致する行が無いときは`spill_free_frames`へ落ちる。これは異常ではなく正常系である。** `ltx`は**既定構成に当たる行を意図的に持っていない**——LTX 2.3の既定構成は復元(デコード)段での退避のため線がトークン数に対して単調にならず、1本の線では表せないからである(理由の正本はバックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §1.1)。**この表に「`requires`が空の`ltx`の行」を足してはならない。**
  - **`spatial_factor`／`temporal_factor`**は、トークン数の式`(幅÷spatial_factor) × (高さ÷spatial_factor) × 潜在フレーム数`と、そこからフレーム数を逆算する式`フレーム数 = temporal_factor × (潜在フレーム数の上限 − 1) + 1`に使う係数。将来のモデルで変わりうるので表に持たせてある。
  - **読み方はひとつに固定する**: 必ず`webui/src/shell/comfortTable.ts`の`resolveComfortRow`を通して読むこと。**この鍵を持たない古いサーバー向けには同ファイルの互換シムが働き**、従来どおり「5つの高速化トグルが全onのときだけ賢い線」という挙動になる(そのときの予算が下の2鍵である)。
  - **サーバーはこの表で一切の判定をしない**(拒否も丸めもしない)——助言専用の公開値である点は下の2鍵と同じ。
- **`chain_comfort_token_budget`**(既定`40000`、2026-08-12追加。**`comfort_budgets`を持たない古いサーバー向けの互換値**)は「stage-2(アップスケール工程)の1回分が快適に収まる注意トークン数の上限」。トークン数＝`(幅÷32) × (高さ÷32) × 潜在フレーム数`で、`chain_math.CHAIN_COMFORT_TOKEN_BUDGET`を単一の正本として写した値である。**サーバーはこの値で一切の判定をしない**(拒否も丸めもしない)——Chained画面が解像度スライダーに引く快適上限の目安線と、超過時の速度低下の注意書きにだけ使う、クライアントへの助言専用の公開値。VRAM容量の異なる機体では正しい値も変わるため配信化した(`config.yaml`を書き換えてサーバー再起動すれば線が動く)。**この鍵を持たない古いサーバーがあり得るので、フロントは必ず`webui/src/shell/tokenBudget.ts`の`resolveChainComfortBudget`を通して読むこと**——欠落・0・負・NaNのときは同ファイルのミラー定数へ落ちる。**なお`comfort_budgets`が来ている場合はそちらの行の`chain_budget`が優先され、この鍵は使われない。**`webui/src/modes/single/defaultConfig.ts`(オフライン時のフォールバック設定)の転記元でもある。
- **`single_comfort_token_budget`**(既定`44880`、2026-08-18追加。**`comfort_budgets`を持たない古いサーバー向けの互換値**)は「単発`/generate`(Create画面)1発が快適に収まる注意トークン数の上限」。式は`chain_comfort_token_budget`と同じ(`(幅÷32) × (高さ÷32) × 潜在フレーム数`)だが、**この鍵が使われるのは`comfort_budgets`が来ていないときだけ**である。そのとき互換シムは従来どおり「WebUIの5つの高速化トグル(sage・block_swap_prefetch・keep_resident・fused_gguf_dequant_kernel・vae_mode=prune_vaed)が全てonのときだけこの値からフレーム数マーカーの位置を逆算し、1つでもoffなら`spill_free_frames`へ落ちる」という振る舞いをする（フォールバックは`spill_free_frames`であって`chain_comfort_token_budget`ではない）。**`chain_comfort_token_budget`(Chained・stage-2の1窓分)とは別鍵・別値**で、単発は stage-2 タイル分割なしで全体を1パスで精製するため、チェーンの仕上げ工程1窓分とはワークロードが異なる。**サーバーはこの値で一切の判定をしない**(拒否も丸めもしない)——助言専用の公開値である点は`chain_comfort_token_budget`と同じ。44,880は2026-08-18の4段階・21ジョブ実機検証(3解像度×縦横両向き)で較正された値で、正本はバックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md)。**フロントは必ず`webui/src/shell/comfortTable.ts`の`resolveSingleComfortBudget`を通して読むこと**——欠落・0・負・NaNのときは同ファイルのミラー定数へ落ちる。
- `server`(`AppConfig.server`): `api_key`(設定済みならBearer認証に使う鍵。未設定は`null`)を含む。フロントは`webui/src/api/types.ts`の`AppConfig.server?.api_key`として型宣言しており、値そのものは表示せず**設定の有無だけ**をSettingsPanelのAPIキーバッジ(N13)判定に使う。
- `model`(`AppConfig.model`): `ic_loras`(canny/pose/depth/upscaler/deblur等のコントロールLoRA一覧。キー＝LoRA名)を含む。フロントは`webui/src/api/types.ts`の`AppConfig.model?.ic_loras`として型宣言しており、Create/Chain画面のコントロールLoRA選択ドロップダウン(N2)の選択肢生成に使う。

### 3.3 `POST /pipeline/load`(`api/pipeline.py::load_pipeline`)— 認証あり

明示的モデルロード。**ボディ省略可**。

- ボディ無し/空: 現在の選択(既定は全default)をロード。応答 `{"pipeline_loaded":bool, "pipeline_type":str, "state":str}`。
- ボディ `{"models": {"transformer":"名前", "text_encoder":"名前", ...}}`: モデルスワップ(worker再構築、数分かかる)。カテゴリは `transformer`/`text_encoder`/`video_vae`/`audio`。`"default"`は既定に戻す意味。
- 応答の`models`欄(`webui/src/api/types.ts`の`PipelineLoadResponse.models?: Record<ModelCategory, string>`): リクエストが**非空の`models`ブロックを含んだ場合のみ**応答に含まれ、確定した各カテゴリのactive選択をエコーする。ボディ省略/空でロードした場合(上記の1つ目のケース)はこの欄自体が応答に無い。
- ジョブ実行中は `409 JOB_BUSY`。ロード失敗は `503 PIPELINE_LOAD_FAILED`(フォールバック無し)。
- **ロードは遅い**ので httpx タイムアウトは600秒推奨(`gradio_ui/api_client.py`の`ApiClient.load_pipeline`／`ApiClient.load_pipeline_models`)。

### 3.4 `POST /pipeline/unload`(`api/pipeline.py::unload_pipeline`)— 認証あり

workerを停止しVRAM解放。ジョブ実行中は `409`。応答 `{"pipeline_loaded":false, "state":"unloaded"}`。

### 3.5 `GET /models`(`api/models_registry.py::list_models`)— 認証不要

カテゴリ別モデル一覧。

```json
{ "categories": {
    "transformer": { "default": "default", "active": "default",
       "entries": [ {"name":"default","path":"models/...","is_default":true,"exists":true,"source":"config"}, ... ] },
    "text_encoder": {...}, "video_vae": {...}, "audio": {...} } }
```

呼ぶたびディスク再スキャン。上の`categories`ブロックは**現在アクティブなベースモデル**のもので、形も並びも従来のまま(`services/model_registry.py`の`CATEGORIES`)。マルチエンジン土台(§3-97 P3a)で**加算**された層が次の2つである。

```jsonc
{
  "categories": { ... },            // 上のとおり。従来の利用者(gradio_ui)はここだけ読めばよい
  "active_base_model": "LTX23",
  "base_models": [
    { "id": "LTX23", "display_name": "LTX 2.3", "engine_family": "ltx",
      "active": true, "installed": true, "present": true, "missing_categories": [],
      "unsupported_features": [],
      "category_order": ["transformer", "text_encoder", "video_vae", "audio"],
      "categories": { ... } }
  ]
}
```

- **`category_order`(2026-08-20追加)**: そのベースモデルのカテゴリを**画面に並べる順**。正本は記述子`scripts/manifests/<base>.json`の`categories`のキー順で、既定は交換頻度順(動画モデル→テキストエンコーダ→動画VAE→音声モデル)である。
  - **配列で渡しているのは意図的である。** `categories`オブジェクトのキー順も同じ並びで送っているが、**JSONオブジェクトのキー順は転送を越えて保たれるとは限らない**——AviUtl2プラグイン経由のWebUIでは、サーバーが記述子順で送った応答が受け側でアルファベット順(`audio`/`text_encoder`/`transformer`/`video_vae`)になっていた実例がある(2026-08-20)。配列の要素順にはその曖昧さが無いので、**表示順はこの配列を読むこと**。キー順に依存してはならない。
- `installed`(全カテゴリの既定ファイルが実在) / `present`(1つ以上実在＝一部導入) / `missing_categories`(不足カテゴリ名)。「未導入」と「一部導入」を利用者に区別して見せるための2フィールドである。**未導入のものを選んだときは導入バッチを名指しして案内し、切替を中止して元の選択へ戻す**——文面の規約（バッチ名は`install-<記述子のid>.bat`。組み立ては`webui/src/shell/useBaseModels.ts`の`baseModelInstaller()`）と、その理由は[`MULTI_ENGINE_DESIGN.md`](../../../Docs/MULTI_ENGINE_DESIGN.md) §6.2 が正本。**この一覧は画面を開いたときにしか取り直さない**ので、案内には「開き直してから」を必ず含める。
- **`unsupported_features`(string[]、§3-98 P5で追加)**: そのベースモデルのエンジンが**扱えない機能の名前**。押しても必ず断られる操作を先回りで灰色にするために読む(強制するのは常にサーバー側の422 `FEATURE_UNSUPPORTED`)。**LTX 2.3 は空配列**、**LTX 2.5 は現在2件**(`two_stage_hq` / `prune_vaed`)である。**`chain` / `v2v` / `a2v` は 2026-08-23 に、`loras` / `reference_video` は 2026-08-24 に、`keep_resident` と `sage_attention` は 2026-08-25 に、`retake` と `end_source` は 2026-08-26 に、`outpaint` は 2026-08-29 にこの配列から外れた**——LTX 2.5 でもクリップ連結・V2V継続・A2V(SingleタブのA2V・長尺A2V・バッチA2Vを含む)・スタイルLoRA・IC-LoRA(参照動画による制御。長尺を含む)・モデル骨格の常駐(`keep_resident`。既定 off のオプトイン)・SageAttention(`attention_backend`。既定 `"sdpa"` のまま)・撮り直し(`retake`)・素材（末尾）(`end_source`)・画角拡張(`outpaint`)が動くようになったためで、宣言を残すと**動くタブやパネルを灰色にしてしまう**。**2026-08-26 に `retake` と `end_source` が、2026-08-29 に `outpaint` が外れたことで、モード名はこの配列から1つも無くなった**——いまここに並ぶ2件は、すべて単発生成の拒否表(`REJECT_TABLE`)から自動的に作られるもので、**どれもモードではなくエンジンが持っていない機能である**。**`nag` は 2026-08-30 に外れた**(非CFGネガティブプロンプトの開通)——**ただしこれで画面のグレーアウトは1つも変わらない。`nag` は下の3つの表のどこにも現れない語**で、NAG のトグルは `unsupported_features` を見ずに描かれていた(押せば422が返る、という関係だった)。**この配列が減ってもグレーアウトが変わらない項目がある**という区別の、2例目である。**フロントエンド側では、`retake` が外れると Edit タブの「撮り直し」サブタブ(とタイムラインの右クリックからそこへ入る導線)の灰色が解け、`end_source` が外れると Chained タブの「素材（末尾）」カードの灰色が解け、`outpaint` が外れると Edit タブの「画角拡張」サブタブの灰色が解ける。** **Edit タブはサブタブ単位で灰色にする**——タブ全体が灰色になるのは`retake`と`outpaint`の両方が並んでいるときだけで、**いまはどちらも並んでいないので、Edit タブの2つのサブタブは両方とも生きている**(`editSubTabsDisabledFor`。`chainPanelsDisabledFor`と同型の純関数で、サーバーの機能名`outpaint`とサブタブ名`outpainting`の読み替えもここだけで行う。**どちらのサブタブも灰色にしない状態が今の答えだが、この仕組みを消してはいけない**——将来また片方だけを断るベースモデルが載ったときに、そのまま効く)。**フィールドそのものを返さないバックエンドは「制限なし」として扱うこと**(省略＝全部だめ、ではない)。**ただし、設定画面の Attention の選択が LTX 2.5 でも押せるようになったのは、この配列から `sage_attention` が外れたからではない**——設定画面(`SettingsPanel`)は`unsupported_features`をそもそも読んでおらず、Attentionの可否は`GET /status`の`acceleration.sage_available`だけで決まる。押せるようになったのは、**2026-08-25にバックエンドがその真理値表からLTX 2.5の例外(恒久的に`false`を返す上書き)を削除した**ためである(バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §77.4)。**この配列が減ってもグレーアウトが変わらない項目がある**という区別に注意すること。正本はバックエンド[`Videomni_Backend_Specification.md`](../../../Videomni_Backend_Specification.md) §6.10(b)。

### 3.6 `GET /loras`(`api/loras.py::list_loras`)— 認証不要

```json
{ "loras": [ {"name":"Pixar_Toon", "kind":"style", "has_thumbnail":true, "exists":true, "source":"scan",
              "preprocess":null, "reference_downscale_factor":null}, ... ] }
```

- 行の形は `services/lora_registry.py::LoraEntryInfo.as_dict`。**パスは漏らさない**(nameのみ)。
- `kind`: `"style"`(画風・キャラ、参照動画不要) / `"control"`(制御、参照動画必須)。フロントでタブ分けに使う。
- `preprocess`(str \| null、**2026-08-11追加**): 制御系アダプタの前処理種別(`"canny"`/`"pose"`/`"depth"`等)。styleアダプタや前処理不要の参照系(`deblur`等)は`null`。フロントは**このフィールドの値が`"depth"`かどうか**でチェーン(2クリップ以上)を選ぶ操作を先回りブロックする(`LORA_DEPTH_CHAIN_UNSUPPORTED`、§2)。
- `reference_downscale_factor`(int \| null、**2026-08-11追加**): 参照動画の内部ダウンスケール倍率(`2`または`1`)。union-control系は`2`、`deblur`等のタイル化エンコード経路は`1`。
- 呼ぶたび `config.model.lora_dir`(既定 `./models/LTX23/StyleLoRA`)を再スキャン。

### 3.7 `POST /loras/reload`(`api/loras.py::reload_loras`)— 認証不要

明示再スキャン。応答 `{"total":N, "styles":n, "controls":m}`。「再読み込み」ボタン用。

### 3.8 `GET /loras/{name}/thumbnail`(`api/loras.py::get_lora_thumbnail`)— 認証不要

`<stem>.png` サムネイルを `image/png` で返す。無ければ `404 LORA_THUMBNAIL_NOT_FOUND`。**WebView2 の `<img src>` から直接叩ける**(絶対URL: `http://host:port/api/v1/loras/{name}/thumbnail`。CORS対象外のリソース取得なのでプロキシ不要)。

### 3.9 `POST /upload/image`(`api/uploads.py::upload_image`)— 認証あり

**multipart/form-data**、フィールド名 `file`。I2Vキーフレーム画像用。

- 応答 `UploadImageResponse`(`api/models.py::UploadImageResponse`): `{image_id, original_filename, stored_path, width, height, content_type}`。
- 返った `image_id` を `/generate` の `conditioning_images[].image_id` に渡す。
- 制約: `config.yaml`の`upload.max_image_size_mb`(20MB)／`upload.allowed_image_extensions`(`.png/.jpg/.jpeg/.webp`)／`upload.normalize_to_png`(既定でPNG正規化)。

### 3.10 `POST /upload/video`(`api/uploads.py::upload_video`)— 認証あり

multipart `file`。制御LoRAの参照動画 / V2V継続元 / 素材（末尾）。

- 応答 `UploadVideoResponse`(`api/models.py::UploadVideoResponse`): `{video_id, original_filename, stored_path, content_type, size_bytes, trimmed, frame_count, fps}`(`trimmed`は**2026-07-30追加**、`frame_count`/`fps`は**2026-08-16追加**。いずれも下記)。
- `video_id` を `/generate` の `reference_video_id`、`/generate/chain` の `source_video.video_id`、または同 `end_source.video_id` に渡す。
- 制約: max 200MB、`.mp4/.mov/.webm/.mkv`(`config.yaml`の`upload.max_video_size_mb`／`upload.allowed_video_extensions`)。再エンコードせず保存(下記のトリム引数を指定した場合を除く)。タイムアウト300秒推奨。

**任意のクエリ引数 `trim_start_sec` / `trim_duration_sec`(2026-07-30追加、V2Vリボン範囲トリム。台帳[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-59＝2026-08-01にクローズ済み。起票時は`PENDING_TASKS.md` §1-6)**

タイムライン上のリボンが元動画ファイルの一部しか占めていないとき、その範囲だけを切り出してアップロードするための引数。**クエリ文字列**で渡す(ボディではない)。**V2V継続元と IC-LoRA参照動画(`reference_video_id`)の両方が同じ引数を使う**(2026-08-01、フロントエンドが同じ判定関数`decideSourceTrim`を両経路へ適用するようになった。バックエンドは両者を区別しない)。

| 引数 | 型 | 既定 | 制約 |
|---|---|---|---|
| `trim_start_sec` | float \| null | `null` | **なし**(`Query(None)`に`ge=`/`le=`を付けていない) |
| `trim_duration_sec` | float \| null | `null` | 同上 |

- 意味: 保存する動画を `[trim_start_sec, trim_start_sec + trim_duration_sec)` の区間だけに切り詰める。切り出しは`services/video_io.py::cut_range_mp4`が担い、**ソースの実測fpsのままフレーム単位**で切る(リサンプルしない)。音声があれば同じ窓へ揃える。末尾を超える要求はエラーにせず残りだけを書く(クランプ)。
- **両方が揃って有限・`start >= 0`・`duration > 0` のときだけ**トリムが走る。それ以外(片方だけ指定・NaN/inf・負の開始・0以下の尺)は**黙って素通しし、従来どおり全体を保存する**。
- **設計判断: 不正値でも422にしない。** 制約を付けて422を返す設計にすると、それまで成功していたアップロードが失敗に変わる(クライアントが古い・値の導出にバグがある、のどちらでも起こり得る)。加えて`Docs/PENDING_TASKS.md` §3-32の「製品UIから422を起こす操作手段が無い」状態を壊してしまう。よって**このエンドポイントに新しい4xx/5xx経路は一切増えていない**。ffmpegが無い・コンテナが解析できない等の切り出し失敗も同様で、無傷の元アップロードを`trimmed: false`で返す。
- 応答 `trimmed`(bool): **2引数が指定され、かつ切り出しが実際に成功したときだけ`true`**。トリムを要求したのに`false`が返ったら「ID自体は使えるが指す中身は元動画全体」という意味なので、フロントエンド側はここでGenerateを止める(Chain=V2Vは`useChainForm`の`sourceTrimFailed`、Create=IC-LoRA参照動画は`useGenerationForm`の`referenceTrimFailed`〔2026-08-01追加〕。誤った区間から生成するより止めるほうが安全という判断)。
- **mkv/webm入力＋トリム時は保存が`input.mp4`へ正規化される**: 切り出し結果はmp4(libx264+AAC)なので、成功時は`uploads/videos/{video_id}/input.mp4`へ置き換わり、元の`input.mkv`は削除される(`path_for`が`glob("input.*")`で一意に解決できるよう`input.*`を常に1本に保つ設計)。したがって`content_type`は`video/mp4`、`size_bytes`は切り出し後の実サイズ、`stored_path`は`.../input.mp4`になる。トリムしなかった場合は従来どおり元の拡張子のまま保存される。
- **未指定時は従来リクエストとバイト単位で同一**: 2引数を送らなければ`cut_range_mp4`は呼ばれず、受信バイト列がそのまま保存される。ブリッジ側も`query`を省略するとURLが完全に不変になる(契約v10、[`BRIDGE_CONTRACT.md`](BRIDGE_CONTRACT.md) §4.5)。この等価性はバックエンドの`tests/test_upload_video_trim.py`とWebUIの`ChainScreen.prefill.test.tsx`の両方で機械的に固定してある。
- 検証記録: [`Nz-Videomni/Docs/VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §42。凍結契約表との関係は`Nz-Videomni/Videomni_Backend_Specification.md` §6.1の補足(本エンドポイントは凍結表に未掲載のADDITIVEエンドポイント)。

**任意のクエリ引数 `max_frames`(2026-08-11追加、長尺IC-LoRAの参照動画アップロード用。台帳[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-78)**

チェーンの総尺上限(`MAX_CHAIN_TOTAL_PIXEL_FRAMES = 11544`、§5.2)を超える参照動画をアップロードしようとした場合に、先頭をその値へ切り詰めるための引数。**int \| null、既定`null`**。実測フレーム数が指定値以下なら**一切変換しない**(再エンコードしないため画質は劣化しない)。超過したときだけ先頭を切り出す。`trim_start_sec`/`trim_duration_sec`と同様クエリ文字列で渡し、応答の`trimmed`は実際に切り詰めが発生したときだけ`true`になる。フロントエンドは右クリック#19(長尺IC-LoRA、[`RIGHTCLICK_REDESIGN_SPEC.md`](RIGHTCLICK_REDESIGN_SPEC.md) §3-4)経由のアップロードで`11544`を渡す。

**素材（末尾）のアップロードでは`max_frames=685`を渡す**(2026-08-16)。読み取るのは素材の先頭`context_frames+1`フレームだけなので、それ以上を`uploads/`へ残す理由がない——685は素材のフレームレートが生成側より高くても余裕がある値として選んだもので、窓内モード(§5.2)で実際に読むのは先頭9フレームだけになった今も**値は据え置いている**（下げても得るものが無く、フレームレートの高い素材を新たに弾く副作用だけが出るため）。**トリム(`trim_start_sec`/`trim_duration_sec`)と併用するときは`max_frames`が無視される既存仕様**があるため、フロントエンド側でトリムの秒数も`685/生成fps`でクランプしている。

**応答の `frame_count` / `fps`(2026-08-16追加。素材（末尾）が最低長9フレームを満たすかの判定に使う。台帳[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-82)**

| フィールド | 型 | 意味 |
|---|---|---|
| `frame_count` | int \| null | 保存された動画の実フレーム数 |
| `fps` | float \| null | 同じくフレームレート |

- **経路別の充填規則**: 無トリム・`max_frames`無し＝**両方`null`**（ffprobeの回数を増やさないため）／`max_frames`指定・未切り詰め＝`frame_count`は既存の実測値、`fps`は`probe_fps()`を1回だけ追加／切り詰め、または`max_frames`付きのトリムが成功＝`cut_range_mp4`の情報を流用（`max_frames`無しのトリムのみは両方`null`）／トリム失敗＝両方`null`。**通常のアップロード経路の挙動と所要時間は変わっていない。**
- クライアントはこの2値で**素材が最低長9フレーム（＝錨8＋因果VAEのプライマ1）を満たすか**を判定する（生成フレームレートへ保守側で換算）。窓内モード（§5.2）では錨が8フレーム固定なので、v2にあった「実測長から帯の長さを導出する」計算は無くなった。`null`のときは既知の尺（秒）からの推定へ、それも無ければ長さ不明として生成をブロックする——**黙って422を食わせないための段**である。

### 3.11 `POST /upload/audio`(`api/uploads.py::upload_audio`)— 認証あり

multipart `file`。A2V音声用。

- 応答 `UploadAudioResponse`(`api/models.py::UploadAudioResponse`): `{audio_id, original_filename, stored_path, content_type, size_bytes}`。
- `audio_id` を `/generate/chain` の `source_audio.audio_id` に渡す。
- 制約: max 50MB、`.wav/.mp3/.m4a/.aac/.flac/.ogg`(`config.yaml`の`upload.max_audio_size_mb`／`upload.allowed_audio_extensions`)。

### 3.12 `POST /generate`(`api/generate.py::generate`)— 認証あり — 中核

単一クリップ生成(T2V / I2V)。**非同期**。ボディ = GenerateRequest(§5参照)。

- 成功: **HTTP 202** + `{job_id, status:"queued", created_at}`(`api/models.py::GenerateResponse`)。
- 事前検証(同期): conditioning画像の存在(`IMAGE_NOT_FOUND`)、参照動画の存在と解像度(128の倍数でないと`REFERENCE_RESOLUTION_INVALID`)、LoRA名解決とkind判定(`LORA_REQUIRES_REFERENCE`/`REFERENCE_REQUIRES_CONTROL_LORA`/`LORA_PREPROCESS_CONFLICT`)。
- 実行中に別の生成を投げると `409 JOB_BUSY`(`api/generate.py::generate`の実行中ジョブガード)。
- 読み込み中は `409 PIPELINE_LOADING`(ジョブ作成前・単一ジョブガードより先に判定。2026-08-31追加)。

### 3.13 `POST /generate/chain`(`api/generate_chain.py::generate_chain`)— 認証あり

複数クリップ連結 / V2V継続 / A2V。ボディ = GenerateChainRequest(§5.2)。

- 成功: **202** + `{job_id, status, created_at, num_clips}`(`api/models.py::GenerateChainResponse`)。
- 事前検証: clip0画像存在、source_video/source_audioの404 + プリフライト(短すぎ→422)。
- 読み込み中は `409 PIPELINE_LOADING`(ジョブ作成前・`/generate`と同じ判定。2026-08-31追加)。

### 3.14 `GET /jobs`(`api/jobs.py::list_jobs`)— 認証不要

全ジョブ一覧 = `list[JobResponse]`(§6)。in-memory・再起動で消える。

### 3.15 `GET /jobs/{job_id}`(`api/jobs.py::get_job`)— 認証不要 — 進捗ポーリング用

1件の JobResponse(§6)。**フロントはこれを1秒間隔でポーリング**。不明IDは `404 JOB_NOT_FOUND`。

### 3.16 `GET /jobs/{job_id}/video`(`api/jobs.py::get_job_video`)— 認証不要

完成mp4をバイナリ(`video/mp4`, `filename={job_id}.mp4`)で返す。

- 未完了なら `409 VIDEO_NOT_READY`、出力欠落なら `404`。
- **これがフロントの主要な成果物取得口**。WebView2 なら `<video src>` に直接与えても、fetchでBlob化しても良い。

### 3.17 `POST /jobs/{job_id}/join`(`api/jobs.py::join_job`)— 認証あり

V2V結合(元動画+続きを1本化、音声クロスフェード付き)。**同期(200)・GPU不要・ffmpegのみ**。ボディ = JoinRequest(任意、`{}`可):

- `{audio_smoothing: bool(既定true), handle_crossfade_ms: int(既定300, 0-2000)}`(`api/models.py::JoinRequest`)。
- **`source_tail_seconds: float(既定5.0, 0=全長連結)`(2026-07-21・Join復活で追加)**: 元動画をフル尺で連結せず、末尾この秒数だけを切り出して続き動画とつなぐ「末尾トリム方式」。元動画がこの尺以下ならそのまま全長連結。長尺動画で再エンコード時間が元動画長に依存する問題を頭打ちにするための手当て。トリムはprobe後・normalize前にソース実測fpsで行い(リサンプル無し)、結果はatomic renameで`joined.mp4`へ差し替える。
- 応答 `{job_id, joined_path, join_mode, source_normalized, ...}`(`api/models.py::JoinResponse`)。**2026-07-21で応答に`trimmed_source_seconds: float`(実際に切り落とした元動画の秒数。フロントの🎞挿入位置計算に使う)と`source_fps: float|null`(元動画の実測fps)を追加**。
- V2Vジョブでないと `422 JOB_NOT_JOINABLE`、失敗 `503 JOIN_FAILED`。タイムアウト120秒推奨。
- **注**: `handle_crossfade_ms`は音声(`acrossfade`)にのみ効き、映像は常にハードカット。フロントUIのラベルも「音声クロスフェード」と表記する。

### 3.18 `GET /jobs/{job_id}/joined`(`api/jobs.py::get_job_joined`)— 認証不要

結合済みmp4を返す。join前は `404 JOINED_NOT_READY`。

### 3.19 `DELETE /jobs/{job_id}`(`api/jobs.py::delete_job`)— 認証あり

- 実行中: **best-effortキャンセル**(`cancel_requested=true`)。応答 `{job_id, cancel_requested:true, status}`。**推論を安全に中断できないため、完了後に`cancelled`へ遷移**(即停止ではない)。UIは「キャンセル要求済み」の表示が必要。
- 完了/終了済み: レコード削除+出力ディレクトリ削除。応答 `{job_id, deleted:true}`。

## 4. ジョブフロー(非同期・ポーリング)

既存Gradio UIの実装(`gradio_ui/api_client.py`の`ApiClient`、`gradio_ui/handlers.py`の`make_generate_handler`→`_poll_job_until_done`)が参考シーケンス。

**T2Vの基本フロー**:

1. (I2Vなら)`POST /upload/image` → `image_id` 取得。
2. `POST /generate` → **202** で `job_id` 即取得。
3. **1秒間隔で `GET /jobs/{job_id}` をポーリング**(`handlers.py::_poll_job_until_done`、既定interval=1.0秒、timeout=7200秒[120分、2026-07-14延長])。
   - `status=="running"`: `progress`(0.0-1.0)、`current_step/total_steps`、`stage`、`clip/clip_count`(chainのみ)で進捗表示。
   - `status=="completed"`: `GET /jobs/{job_id}/video` でmp4取得。
   - `status in ("failed","cancelled")`: `error` フィールド表示。
4. AviUtl2 タイムラインへ受け渡し(具体手順は [SDK_REFERENCE.md](SDK_REFERENCE.md) §7参照)。

**Chainフロー**: `POST /generate/chain` → 同じポーリングループ(`handlers.py::make_chain_handler`→`_poll_job_until_done`)。chainジョブは `clip`/`clip_count` で「今どのクリップを処理中か」を返す。

**進捗の `stage` 値**(`api/models.py::JobResponse`の`stage`、`gradio_ui/handlers.py`の`_STAGE_LABEL_KEYS`): `"encode"`/`"stage1"`/`"stage1_denoise"`/`"stage2_denoise"`/`"denoise"`/`"tile"`/`"decode"`。mockやqueued時は `null`。

**注意点**:

- `peak_vram_mb` は jobs API応答に含まれない(`metadata.json` か `logs/ltx_worker.log` のみ)。
- ジョブ履歴はin-memory(再起動で消滅)。ただし `outputs/{job_id}/metadata.json` はディスクに残る。

## 5. 生成リクエストスキーマ(パラメータ・制約)

### 5.1 GenerateRequest(`api/models.py::GenerateRequest`)— T2V/I2V共通

> **ベースモデルにLTX 2.5を選ぶと、本節のフィールドのうち2個が422になる**(2026-08-30現在)。断られるのは`pipeline`(`"distilled"`以外)／`vae_mode`(`"default"`以外)で、いずれも**既定値と違うときだけ**エラーコード`FEATURE_UNSUPPORTED`(422)になり、**最初の1件だけ**が名指しされる。判定はジョブ作成前・素材の404より先。**`loras`と`reference_video_id`(および2つの強度)は2026-08-24に、`keep_resident`と`attention_backend`は2026-08-25に、`outpaint`は2026-08-29に、**`nag_enabled`は2026-08-30に**422を抜けて動作側へ移った——`outpaint`は単発生成の側で「モード」を名指ししていた最後の1つである。** 別に**「断らないが黙って無視する」2個**(`guidance_scale`／`num_inference_steps`)があり、こちらはジョブが通常どおり走る(蒸留版2.5は工程数が固定で、CFGの概念も無いため。無視したことはバックエンドの`logs/server.log`に1行残る)——**ただしスキーマがこの2つを`pipeline="distilled"`のとき既定値に固定するので、LTX 2.5が受理しうるリクエストではこの行は出ない。** **フィールド単位の正本**はバックエンド[`Videomni_Backend_Specification.md`](../../../Videomni_Backend_Specification.md) §6.10(d)の4分類表(全28件＝422系2／無視2／動作24／従属0)。連結生成側は別の表で、§5.2の注記と同 §6.10(f)(全34件＝422系2／無視2／動作30／従属0)を見ること。**単発側のこの4分類は2026-08-26の改修では動かず(撮り直しと素材（末尾）は連結生成のスキーマにしか無いフィールドだからである)、2026-08-29の画角拡張の開通で動いた**——`outpaint`は逆に単発生成のスキーマにしか無い。**そして2026-08-30の非CFGネガティブプロンプトの開通では、単発側と連結側が同時に動いた**——7つのフィールド(`nag_enabled`／`negative_prompt`／`nag_scale`／`nag_tau`／`nag_alpha`／`neg_method`／`vsf_scale`)は両方のスキーマにあるためである。**この結果、422系の顔ぶれは単発側と連結側でまったく同じ2つになり、「従属」は両方とも空になった**(**空でも表そのものは残っている**——将来また422の背後に副パラメータができたときの置き場になるため)。
>
> **`nag_alpha`を0にしても「NAGなし」とビット単位で同じ絵にはならない**(負のプロンプトを1件足すぶんバックエンド内部のバッチが増え、行列積のカーネル選択が変わりうるため)。**切りたいときは`nag_enabled`を`false`にすること。**


| フィールド | 型/既定 | 制約 |
|---|---|---|
| `prompt` | str(必須) | 1-2000文字 |
| `negative_prompt` | str `""` | 最大2000文字。**2026-07-28、NAG経由で条件付き解禁**(§8参照。`nag_enabled`未指定時はUI非表示のまま) |
| `width` | int `512` | 256-4096、**64の倍数必須**(`api/models.py::GenerateRequest`のフィールドバリデータ) |
| `height` | int `320` | 128-4096、**64の倍数必須** |
| `crop_output` | `{width,height}` \| null | 各≥32、≤生成幅/高。最終mp4を中央クロップ(例1280×768生成→1280×720) |
| `num_frames` | int `49` | 9-481、**8n+1必須**(`api/models.py::GenerateRequest`のフィールドバリデータ) |
| `frame_rate` | float `24.0` | 1.0-60.0 |
| `num_inference_steps` | int `8` | **§8参照: distilledは8固定・変更不可(それ以外は422)** |
| `guidance_scale` | float `1.0` | **§8参照: distilledは1.0固定・変更不可** |
| `seed` | int `-1` | -1=ランダム |
| `pipeline` | `"distilled"` \| `"two_stage_hq"` | **§8参照: 既定distilled(実質distilledのみ)** |
| `conditioning_images` | list `[]` | **最大5枚**。空=T2V、1件以上=I2V |
| `loras` | list `[]` | 登録済みLoRA名参照(§5.3)。**2026-08-02、`LoraSpec`へ`audio_strength`(音声軸の適用強度)追加**(詳細は§5.3) |
| `reference_video_id` | str \| null | 制御LoRA用。単独指定不可(loras必須) |
| `conditioning_attention_strength` | float \| null | 0.0-1.0、loras必須 |
| `reference_video_strength` | float \| null | 0.0-1.0、loras必須 |
| `nag_enabled` | bool `false` | **2026-07-28追加**。NAG（Normalized Attention Guidance。CFGを使わずにネガティブプロンプトを効かせる手法）を使うか。`true`かつ`negative_prompt`が空文字だと**422** |
| `nag_scale` | float `11.0` | **2026-07-28追加**。1-20。NAGの効き具合の強さ |
| `nag_tau` | float `2.5` | **2026-07-28追加**。1-10。NAGの正規化しきい値 |
| `nag_alpha` | float `0.25` | **2026-07-28追加**。0-1。NAGの混合率 |
| `neg_method` | `"nag"` \| `"vsf"` `"nag"` | **2026-07-29追加**。非CFGネガティブプロンプトの方式選択。NAGは正負のattentionを2回計算して混ぜる方式、VSF（Value Sign Flip）は正負のコンテキストを連結し1回のattentionで済ませつつ負側のV（value）だけ符号反転×スケールする方式。`nag_enabled`がtrueのときのみ有効 |
| `vsf_scale` | float `1.5` | **2026-07-29追加**。0-10。VSFの効き具合の強さ(`neg_method="vsf"`時のみ参照)。実用域は1.5〜5、0を指定しても無効化はされない |
| `attention_backend` | `"sdpa"` \| `"sage"` `"sdpa"` | **2026-07-31追加**。attention（注意機構）の実装選択。`"sage"`はSageAttention 2.2.0（量子化で注意機構の計算そのものを速くする外部カーネル）。**720pで約1.17倍・1344×1728のi2vで約1.26倍**の高速化。**同一シードでも生成結果の細部が変わる**（数値精度の違い。構図は同じで質感やノイズの出方が変わる）。サーバーに未導入なら422にはならず`"sdpa"`へ降格して完走する。**LTX 2.5でも2026-08-25から使える**（**契約も実装も2.3と共有**しており、降格の規律も逐語で同じ。既定は両エンジンとも`"sdpa"`のまま）——実測は1280×768・2クリップ×121フレームの連結生成で**1.1017倍**。**効き目は形状に強く依存し、512×320級では効かないか、かえって遅くなる**（同一シードで細部が変わる点も2.5でそのまま当てはまる）。**この項目だけは高速化のうち唯一「絵が変わる」種類**なので、固定ベンチマークのSHA-256一致では回帰を判定できない。詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §43（2.3）・**§77（2.5開通。オーナー目視ゲートG8は2026-08-25に合格＝同 §77.10）** |
| `fused_gguf_dequant_kernel` | bool `true` | **2026-08-04追加**。GGUFのK量子化（Q4_K/Q5_K/Q6_K）の逆量子化（圧縮された重みを計算に使える形へ展開する処理）を、Tritonの1カーネルへまとめて速くする。**768p/257フレームの交互対比較で約17.5%短縮**（144.5秒→119.2秒）。**生成結果は変わらない**——同一シードならon/offでビット単位一致する（必須要件として検証済み）。Tritonが無い・カーネルが例外を出した・型ごとの初回自己検証で不一致だった、のいずれでも黙って従来実装へ降格し、生成そのものは落とさない。**既定は`true`**（2026-08-04、実機ゲートG1〜G8全項目合格を受けてオーナー承認のうえ`false`から反転）。既定と同じ値のときはリクエストにキーを載せないので、フロント側`accelerationRequestFields()`が`fused_gguf_dequant_kernel`を足すのは**OFFにしたときだけ**（`block_swap_prefetch`と同じ向き、`keep_resident`とは逆向き）。実際にどう扱われたかは`metadata.json`の`fused_gguf_dequant_kernel_used`（`"off"`／`"on"`／`"on->off"`。`"on->off"`は「要求したが実際には適用されなかった」）。詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §51。なお**このAPIは未知のフィールドを送っても422にはならない**ので、古いクライアントが送ってくる撤去済みフィールドは黙って無視される |
| `vae_mode` | `"default"` \| `"prune_vaed"` `"default"` | **2026-07-31追加・モック**（受理のみ・生成に一切影響しない）。§7参照。既存の`vram_optimization.vae_tiling`とは**無関係** |
| `keep_resident` | bool `false` | **2026-08-03追加**。モデルのCPU側「骨格」(GGUFから組み上げた重みの一覧とモジュール構造。DiT約16.5GB＋Gemma約8〜9GB)をジョブ間で保持して使い回し、2本目以降の前処理の待ち時間を消す。**HIT時の前処理は実測5.32秒**(ベースライン68.6〜75.0秒)。**生成結果は変わらない**——同一シードならon/offでビット単位一致し、GPUには何も常駐させないのでVRAMも不変。**メインメモリ64GB以上を推奨**(約20GBを常時占有)。既定`false`のときはリクエストにキーを載せない(フロント側`accelerationRequestFields()`はONのときだけ`keep_resident: true`を足す)。サーバー側で`dit_cpu_load=false`／`block_swap_prefetch=false`と併用された場合は警告のうえそのジョブだけ自動off、`gguf_per_layer_quant=false`との併用はジョブがエラーで停止する。実際にどう扱われたかは`metadata.json`の`keep_resident_used`(`"off"`／`"on"`／`"on->off"`)。詳細はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §48。**LTX 2.5では2026-08-25から使えるが、契約が同じだけで実装は別物である**——2.5が常駐させるのはGemma 4テキストエンコーダの状態辞書(state dict)ただ1つで、増分は実測**7.68GiB**(2.3の約20GBとは別物)。2.5側には併用制限も降格経路も無く、エコーは`"on"`／`"off"`の2値しか出ない(同 §76) |

- **`keep_resident`は`/status`に載らない（意図的）**: `attention_backend`の`acceleration.sage_available`のような可否フラグ（能力ゲート）は存在しない。このマシンに十分なメインメモリがあるかという利用者側の選択にすぎず、サーバーが「できる／できない」を答えられる性質のものではないため。したがってフロント側もボタンを封じず、常に操作できる。

**ConditioningImage**(`api/models.py::ConditioningImage`): `{image_id, frame_idx(既定0,≥0), strength(既定0.8,0.0-1.0), crf}`。

- `frame_idx==0`=先頭フレーム(latent-replace)。`>0`はサーバー側で 8n+1 グリッドへスナップ(`(f-1)//8*8+1`)+範囲クランプ。フロントは自然値を送ってよい。
- `generation_mode`: conditioning_imagesが空→`t2v`、あれば→`i2v`。

**凍結制約(緩和禁止)**:

- width/height ÷64、num_frames 8n+1、distilled=8step/CFG=1.0。
- 参照動画ジョブは width/height が**128の倍数**必須。
- 尺上限: 20秒(481f)@24fps。

### 5.2 GenerateChainRequest(`api/models.py::GenerateChainRequest`)

> **ベースモデルにLTX 2.5を選ぶと、本節のフィールドのうち2個が422になる**(2026-08-30現在)。断られるのは`pipeline`(`"distilled"`以外)／`vae_mode`(`"default"`以外)で、いずれも**既定値と違うときだけ**エラーコード`FEATURE_UNSUPPORTED`(422)になり、**最初の1件だけ**が名指しされる。判定はジョブ作成前・素材の404より先。**`source_video`と`source_audio`は2026-08-23に、`loras`と`reference_video_id`(および2つの強度)は2026-08-24に、`keep_resident`と`attention_backend`は2026-08-25に、`retake`と`end_source`は2026-08-26に、**`nag_enabled`(と`negative_prompt`／`nag_scale`／`nag_tau`／`nag_alpha`／`neg_method`／`vsf_scale`)は2026-08-30に**422や「無視」を抜けて動作側へ移った**ので、V2V継続・A2V・長尺A2V(複数クリップ)・Singleタブの`stage2_window="full_length"`・スタイルLoRA・IC-LoRA(クリップ別の参照窓＝長尺IC-LoRAを含む)・撮り直し・素材（末尾）はいずれもLTX 2.5で通る。ネガティブプロンプト(NAG／VSF)も含めてLTX 2.5で通る。**残る2個はいずれも連結生成のモードではなく、エンジンが持っていない機能を指すフィールドである**——**`POST /generate/chain`のスキーマが表現できるモードは、2026-08-26から全部LTX 2.5でも通る**(**オーナーの目視ゲートG8も2026-08-30に合格した**。バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §78.14)。**参照動画を添えるときは`width`/`height`が128の倍数であること**(系統に依存しない既存の検査)、**`depth-control`を2クリップ以上で使うと422**(`LORA_DEPTH_CHAIN_UNSUPPORTED`)という制約も従来どおり効く。**全34フィールドの4分類表(422系2／無視2／動作30／従属0)の正本はバックエンド[`Videomni_Backend_Specification.md`](../../../Videomni_Backend_Specification.md) §6.10(f)** で、先回りのグレーアウトに使う機能名は`GET /models`の`unsupported_features`(§3.5。現在2件)から取る。**連結生成では1本の`negative_prompt`が全クリップ・全ステージに効く**(この意味づけはLTX 2.3と同じである)。

共通: width/height/crop_output/frame_rate/num_inference_steps/guidance_scale/seed/pipeline は上と同じ制約。加えて:

| フィールド | 型/既定 | 制約 |
|---|---|---|
| `prompt` | str(必須) | 全体共通プロンプト |
| `overlap_frames` | int `3` | 1-8。クリップ間の重なり |
| `overlap_strength` | float `0.5` | 0.0-1.0。つなぎ目ブレンド強度 |
| `clips` | list(必須) | **1-24本**(2026-07-14に8本から拡張)。`source_video`/`source_audio`/`reference_video_id`/`retake`/`end_source`のいずれも無ければ最低2本 |
| `source_video` | `{video_id, context_frames}` \| null | V2V継続（素材（冒頭）＝start source） |
| `end_source` | `{video_id, image_id, context_frames, strength}` \| null | End source（素材（末尾）、2026-08-16追加。台帳[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-82）。**添付した動画・静止画へ繋がる動画を生成する**。`video_id`と`image_id`はどちらか一方（静止画はサーバー側で無音動画へ変換され、以降は1本の経路になる）。**`clips`が1本のときと2本以上のときで挙動が変わる**（1本＝窓内モード／2本以上＝逆順Chained。どちらも出力尺はクリップ合計／旧方式`internal_segment`は到達不能）。`strength`は2026-08-18追加。**LTX 2.5でも2026-08-26から使える**（モード判定・受理範囲・`metadata.json`の`end_source`ブロックはLTX 2.3と同一。**ただし推奨外の逆順Chainedは、LTX 2.5では音声の継ぎ目がさらに悪くなる**——下記**EndSourceSpec**の「エンジン系統による違い」を参照）。詳細は下記**EndSourceSpec** |
| `source_audio` | `{audio_id}` \| null | A2V（音声から動画を生成する機能）。source_videoと排他。**クリップ1〜24本**（2026-08-10に「clips=1本限定」を撤廃。台帳[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-74。記録当時は`PENDING_TASKS.md` §1-16で、2026-08-10のテーマ完結でクローズ移設）。連結タイムライン全体に対して**音声を1本だけ**添付し、各クリップが担当する音声潜在窓はサーバーが自動で割り当てる（`chain_math.audio_segment_windows`）。**クライアント側で音声を分割する必要はない**（分割ファイルを作らないのがフロントエンドの仕様でもある） |
| `loras` | list `[]` | 2026-07-03解禁・2026-07-11に全面解禁。登録済みLoRA名参照(§5.3)。チェーン全クリップ・全ステージに一律適用(クリップ別の強さ指定は無い)。**2026-08-02、`LoraSpec`へ`audio_strength`追加**(§5.3参照) |
| `chunked_upsample` | bool `false` | チャンク化アップサンプル(halo overlap＋CPU offload)へのopt-in。既定`false`=旧来の一括アップサンプル(VRAM消費が総尺に比例)。長尺/高解像度chainを16GBに収めるための2026-07-14追加機能 |
| `reference_video_id` | str \| null | 制御LoRA用参照動画(§5.1と同義、2026-07-11にchainへ解禁・alphaスコープ)。**loras必須・source_videoと排他**。**2026-08-11、「clips=1本限定」を撤廃し1〜24クリップへ拡張**(長尺IC-LoRA。台帳[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-78。記録当時は`PENDING_TASKS.md` §1-15)。長い参照動画を1本だけ添付すると、各クリップが担当する区間をサーバーが`chain_math.video_segment_windows`で自動的に切り出し、stage-1(低解像度で全体の動きを作る第1段階)にのみ注入する(stage-2には入らずVRAM天井は不変)。参照が生成の尺より短ければ、足りない分は参照なしで生成される(エラーにしない)。**`depth-control`のみ2クリップ以上で例外的に422**(`LORA_DEPTH_CHAIN_UNSUPPORTED`、§2)——深度前処理が全編一括設計でメモリに載らないため。フロント側は選択アダプタの`preprocess`(§3.6)が`"depth"`かどうかでチェーン2クリップ以上の組み合わせを先回りブロックする |
| `conditioning_attention_strength` | float \| null | 0.0-1.0、loras必須(§5.1と同義) |
| `reference_video_strength` | float \| null | 0.0-1.0、loras必須(§5.1と同義) |
| `negative_prompt` | str `""` | §5.1と同義。**2026-07-28、NAG経由で条件付き解禁** |
| `nag_enabled` | bool `false` | §5.1と同義(2026-07-28追加) |
| `nag_scale` | float `11.0` | §5.1と同義(2026-07-28追加) |
| `nag_tau` | float `2.5` | §5.1と同義(2026-07-28追加) |
| `nag_alpha` | float `0.25` | §5.1と同義(2026-07-28追加) |
| `neg_method` | `"nag"` \| `"vsf"` `"nag"` | §5.1と同義(2026-07-29追加) |
| `vsf_scale` | float `1.5` | §5.1と同義(2026-07-29追加) |
| `attention_backend` | `"sdpa"` \| `"sage"` `"sdpa"` | §5.1と同義(2026-07-31追加)。チェーン全クリップ・全ステージに一律適用される。**LTX 2.5でも2026-08-25から使える**——エンジンはクリップごと・タイルごとに変換器を組み直すが、その**組み直しのたびに前のぶんを剥がして貼り直す**ので、エコー`attention_used`は**チェーン全体の畳み込み**（どこか1回でも降格すれば`"sage->sdpa"`）になる。実測は1280×768・2クリップ×121フレームで119.5秒→108.5秒＝**1.1017倍**(同 §77.5) |
| `fused_gguf_dequant_kernel` | bool `true` | §5.1と同義(2026-08-04追加)。チェーン全クリップ・全ステージに一律適用される |
| `vae_mode` | `"default"` \| `"prune_vaed"` `"default"` | §5.1と同義(2026-07-31追加・モック) |
| `keep_resident` | bool `false` | §5.1と同義(2026-08-03追加)。`to_clip_request`経由で各クリップへ転記される。チェーンはジョブ内で既に骨格を償却しているため、効くのは**次のジョブ**からである。**LTX 2.5では2026-08-25から使えるが別実装**(テキストエンコーダの状態辞書のみ・増分は実測7.68GiB)で、連結生成でも1つの設定がチェーン全体に効き、構築はジョブあたり1回だけ。実測では2本目の連結生成が35.17秒→30.13秒(同 §76.5) |
| `stage2_window` | `"standard"` \| `"high_resolution"` \| `"full_length"` `"standard"` | Stage-2のクリップ長(潜在フレーム数)プリセット(2026-08-10追加、2026-08-12に`full_length`を追加)。`standard`=潜在22(既定)、`high_resolution`=潜在19。**`full_length`=潜在61(=481フレーム相当)・タイル数は必ず1**。Retake(`retake`、下記)の窓上限に連動し、`high_resolution`では上限が169→145フレームへ縮む(`chain_math.retake_max_window_px(v_tile)`)。`full_length`指定時は**クリップ数1・`source_audio`必須**の2条件をAPI層が422で強制する(`source_video`・`retake`との排他は既存のペア排他制約が先に効くため、`full_length`固有のチェックとしては到達しない)。快適上限は`limits.spill_free_frames`(単発生成用の実測テーブル)側で判定され、Singleの既存の警告帯(琥珀色バナー)がそのまま実態に一致する。Single・Batchのa2vリクエストビルダーが常にこの値を固定送出し、Chained・Retakeのフォームには選択肢として出さない |
| `retake` | `RetakeSpec` \| null | Retake(選択範囲の撮り直し、2026-08-10追加)。`{video_id, window_start_sec, head_px=25, tail_px=24, regenerate_audio=true}`。窓長の正は`clips[0].num_frames`(8n+1、73〜上限。上限は`stage2_window`に連動)。`source_video`・`source_audio`・`reference_video_id`・`clips[0].conditioning_images`と排他。`clips`はちょうど1本必須。**LTX 2.5でも2026-08-26から使える**——仕様（窓長の範囲・のりしろ・排他・`metadata.json`の`retake`ブロック）はエンジン系統に依存せず、LTX 2.3と同一である。**窓は121フレーム以上を推奨**（73フレームでは前後の帯25＋24を除くと自由に作り直せるのが24フレームしか残らず、出来上がりが元とほとんど変わらない。**LTX 2.3から引き継いだ性質**で2.5固有ではない）。**LTX 2.5でのオーナーの目視ゲートG8は2026-08-30に合格した**——AviUtl2の操作パネルからの実機生成で、プロンプトどおりに再生成されること・継ぎ目が分からないこと・「Picture and sound」で音声も生成されること・「Picture only (keep the original sound)」で音声はそのままであることの4点が確認されている（バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) **§78.14**、実測は同 §78） |

**ChainClip**(`api/models.py::ChainClip`): `{prompt(任意,上書き), num_frames(8n+1), conditioning_images}`。**clip 0のみ画像を持てる**。Retake使用時は`clips`がちょうど1本で、この`num_frames`がRetakeの窓長そのものになる。

**SourceVideoSpec**(`api/models.py::SourceVideoSpec`): `context_frames` は 8n+1、[25,145]、かつ `< clips[0].num_frames`。

**EndSourceSpec**(`api/models.py::EndSourceSpec`): **`clips`の本数で挙動が3つに分かれる**（2026-08-17に窓内モード追加、2026-08-18に逆順Chained追加）。判定は`chain_math.compute_chain_layout`が1箇所で行い、`metadata.json`の`end_source.mode`として公開する（リクエスト側にモードを指定するフィールドは無い）。フロントエンドが使うのは**窓内モード（クリップ1本）**と**逆順Chained（クリップ2本以上）**の2つで、**旧方式（`internal_segment`）はAPIから到達不能な死蔵コード**である（後方互換とテスト・切り戻し専用の内部引数`end_source_mode_override`でのみ選べる。通常のリクエストでは絶対に選ばれない）。**推奨は窓内モード（クリップ1本）のみ**（2026-08-18のオーナー裁定）。逆順Chained（クリップ2本以上）は受理されるが推奨外で、クリップの境目・末尾（錨直前）に品質劣化が出ることがあり、これは仕様として許容している（詳細は本節末尾）。

| | **窓内モード（クリップ1本）＝推奨** | **逆順Chained（クリップ2本以上）＝受理される（推奨外、2026-08-18〜）** | **旧方式（`internal_segment`）＝死蔵・到達不能** |
|---|---|---|---|
| 凍結フレームの置き場所 | **クリップ自身の末尾**（生成される窓の内側） | **最終クリップの末尾**（生成される窓の内側） | クリップ列の後ろに継ぎ足す内部区画 |
| 出力長 | **変わらない**（＝クリップの`num_frames`そのもの） | **変わらない**（＝クリップ合計） | クリップ合計＋`context_frames` |
| 生成順 | クリップ1本を1つの窓として生成 | **タイムラインの逆順**（最終クリップから先に生成し、過去側へリレーする） | タイムライン順（内部区画は最後） |
| 生成結果 | 素材へ自然に到達する（意味論的に近い素材の場合） | 機構としては成立（実機ゲートM1〜M7全PASS）。**オーナー目視・試聴済み**——クリップの境目・末尾にモーフが出ることがあり、品質劣化として仕様上許容（推奨外） | 素材へクロスフェードする既知問題 |
| UIからの到達 | Chainedタブの「素材（末尾）」カード | 同カード＋クリップを2本以上に増やす | 到達不可 |

**窓内モード**では、素材の先頭フレームがクリップ自身の末尾フレームとして凍結され、stage-1の**1つのデノイズ窓**の中に素材が同居する。注意機構から全区間に見えるので、生成は最初から素材へ向かって進む。凍結分は出力の内側にあるため**出力の長さは伸びない**。実装と実機実験（4ラウンド20ジョブ）の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §61である。

**逆順Chained**（`"reverse"`、2026-08-18追加。第2段階・バッチ2）は、複数クリップへEnd sourceを拡張する方式である。stage-1のみを依存順（タイムライン末尾から先頭へ）に生成し、各セグメントは自分より未来側のセグメントの頭を「のりしろ」として自分の尾に凍結する——正順Chainedの頭凍結（前のセグメントの尾を自分の頭に凍結）を鏡写しにした形で、**新しい凍結機構は増やしていない**（既存の`freeze_mask_values`が`tail_mask_value=None`のとき自動的に鏡写しソフト凍結になる仕組みをそのまま使う）。生成し終えた各セグメントの潜在は連結してから、既存の一括Stage-2へそのまま渡す（Stage-2は無改修）。設計の詳細はバックエンド[`CHAIN_STAGE2_RESEARCH_NOTES.md`](../../../Docs/CHAIN_STAGE2_RESEARCH_NOTES.md) §11、実装・機械検証・実機ゲートM1〜M7の正本は同[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §64である。

- **意味拡張1: `overlap_frames`（既存フィールド）**: 逆順Chainedの継ぎ目（セグメント間で共有される重なり潜在）の幅そのものとして使われる。窓内モードでは従来どおり`overlap_frames >= 2`が必須だが、**逆順Chainedはこの下限を免除され`overlap_frames = 1`が合法になる**（内部区画を追加しないため音声のりしろ予算の消費が素のチェーンと同一で、下限の根拠が消滅するため）。フロントエンドは「素材（末尾）×2本以上」への遷移で既定値を1へ自動的に切り替える（逆方向の遷移では従来既定へ復帰する）。
- **意味拡張2: `overlap_strength`（既存フィールド）**: 逆順Chainedの継ぎ目にも、正順Chainedの継ぎ目と同じつまみとして効く（鏡写しソフト凍結の強度）。
- **出力長の意味論**: 逆順Chainedは窓内モードと同じ恒等式（`total_px == clips_total_px`）で、**帯はクリップ合計の内側にある**。旧方式のような加算（`+ end_context_px`）は無い。フロントエンドの出力長計算（`computeOutputFrames`）は変更不要だった。
- **素材の先頭1フレームは出力に現れない**: アプリは`context_frames + 1`フレームを切り出し（静止画なら合成し）、その先頭を因果VAEのキーフレーム潜在に使う。出力の末尾に並ぶのは素材の2フレーム目以降である。したがって**素材は最低`context_frames + 1`フレーム必要**（錨8なら9フレーム。フロントエンドの下限もこの9フレーム）。
- **`strength`（0.0〜1.0、既定1.0、2026-08-18追加）**: 錨の固定強度。既定1.0は**従来と厳密同値**（Stage-1・Stage-2ともハード凍結）。1.0未満にすると**Stage-1だけ**マスク値が`1.0 - strength`へ緩み、素材への「なじみ方」が緩やかになる。**Stage-2は`strength`の値に関わらず常にハード凍結**なので、`strength`を下げても**配信される最終フレームは常に素材どおり**である——緩むのはStage-1の途中経過だけ。`overlap_strength`（生成物同士の継ぎ目のブレンド強度）とは別物で、あちらは生成セグメント間の継ぎ目、こちらは素材そのものへの継ぎ目を制御する。応答の`freeze_proof`（`end_source`メタデータ）では、`s2_video_tail`は常に0.0、`s1_video_tail`は`strength < 1.0`のとき非ゼロが正常（`s1_expected_zero`が判定基準を記録する）。フロントエンドは既定値でも常に明示送信する。実装・機械検証・実機ゲートの正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §63。
- **錨への素材音声の凍結（第3弾、2026-08-18追加）**: 素材（`video_id`側）に音声トラックがあれば、**窓内モード・逆順Chainedいずれの錨クリップでも常にその音声が凍結される**。新設のフィールドは無く、`EndSourceSpec`はv1から音声用のトグル・スライダーを持たない——「音声は素材からは取り込まれず独立して生成される」というv1〜第2弾の挙動そのものが変わった**既存フィールドの意味変更**である。**`strength`は映像専用のつまみになった**（音声のマスク値は`strength`の値に関わらず常に0.0）。凍結する音声潜在の本数（`n_end_a`）は、Retakeの尾側のりしろ凍結と同じスキャン規則（潜在の時間支持区間が帯の開始時刻以降にあるものを末尾から連続して数える）で決まり、単純な丸め計算ではない。素材ファイルの末尾基準でスライスするため、素材の潜在格子とタイムラインの格子の位相が一般には一致せず、**最大1潜在（40ミリ秒）の位置ずれ**が許容として残る。**フォールバック（自由生成）は「音声トラック無し／デコード不能」のときのみ**——`audio_frozen=false`・`audio_fallback_reason="no_audio"`になる。音声潜在が必要数に足りない構成（端数不足）はエラーにせず**取れた分だけ凍結する**（`audio_frozen=true`のまま`n_end_a_frozen`が`n_end_a`を下回る形で表れる）。**デジタル無音の検出はしない**——無音の素材は無音のまま凍結され、帯の入口で音が唐突に消える聴感もあり得るがそれは正常動作である。画像end source（静止画）はそもそも音声を持たないため本改修の対象外で、従来どおり音声は自由生成にフォールバックする。**配信される錨区間の音声は原波形のmuxではない**——A2Vや非再生成のRetakeが元の波形をそのまま貼り付けるのとは違い、本機能は**潜在を凍結**するため、配信音声は音声VAE＋ボコーダを1往復した音になる（cross-attentionを効かせるには潜在である必要があり、muxでは目的を達成できないため。同じ曲だが少しこもった音になるのが仕様）。切り戻しはバックエンドのモジュール定数`END_SOURCE_FREEZE_AUDIO`（既定`true`）で、`false`にすると本機能追加より前とバイト完全同一の出力に戻る。実装・機械検証・機械ゲートA1〜A9の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §65。
- **オーナー裁定（2026-08-18、テーマ完結）**: End sourceは**クリップ1本での使用を推奨**し、**複数クリップ（逆順Chained）は推奨外**の使い方と位置づける。実機ゲート後のオーナー目視・試聴で、複数クリップ時にクリップの境目・末尾（錨直前）へ系統的なモーフが出ることを確認しており、**この品質劣化は仕様として許容する**（根治にはモデル側の到着時刻拘束能力が要り、現行のLTX 2.3には無い）。UI上の警告文（フロントエンドのPENDING_TASKS.md §1-22で文言確定済み）はオーナーの次のUI改修バッチで実装する。品質を重視して複数クリップを終端付きで繋ぎたい場合は、End sourceをクリップ1本ずつ使い、生成物を次の素材にして過去へ遡って生成する**手動リレー**（AviUtl2タイムラインで組み合わせる。すべての継ぎ目が窓内モードの錨になる。音声錨の幅は映像の錨幅に比例し、錨8f〔UI既定〕なら音声7潜在≒約0.3秒・錨72fなら74潜在≒約3秒——音楽をしっかり引き継ぐには`context_frames`を大きくする必要があるが、UIは8固定のためAPI直叩きになる）が実用的な回避策になる。詳細・実験結果はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §64.7・§65.8、考察の正本は[`CHAIN_STAGE2_RESEARCH_NOTES.md`](../../../Docs/CHAIN_STAGE2_RESEARCH_NOTES.md) §11。
- **クリップ長169f/145fの品質警告は、クリップ1本（窓内モード）にのみ適用される**: `standard`（潜在22）なら169フレーム、`high_resolution`（潜在19）なら145フレームまでという「stage-2のタイル1枚に収めるのが望ましい」助言（422ではない）は、窓内モードの実験結果に基づくものであり、**複数クリップの逆順Chainedには適用されない**（フロントエンドの警告バナーも`clips.length === 1`のときだけ表示する）。サーバーは何も判定しないので、いずれのモードでもGenerateはブロックしない。
- **`overlap_frames >= 2` はモード依存**: 窓内モードでは従来どおり必須（サーバーはのりしろ1を422にする）。**逆順Chainedでは免除**（上記「意味拡張1」参照）。
- **排他**: `retake`（既に窓の両端を占有）・`source_audio`（映像だけを凍結するため音声と食い違う）・`reference_video_id`（制御アダプタのセグメント別条件付けと競合）とは、どのモードでも併用不可。**`source_video`との併用（＝冒頭と末尾を与えた補間）は、クリップ1本（窓内モード）に限り可能**。**クリップ2本以上との`source_video`併用は2026-08-18から422で拒否する**（新設422、下記参照。真ん中クリップが頭・尾の両方で凍結される、誰も走らせたことのない二重凍結の形になるため。Start＋End併用は次弾のスコープで、3本の表による設計はこの拡張を見越しているが未実装）。`clips[0].conditioning_images`との併用は可能。窓内モードでは凍結フレームがクリップの内側にあるため、極端に短いクリップでは頭と尾の凍結が窓を食い尽くして422になりうる（既定値では到達しない）。
- **新たに422になる構成（2026-08-18、受理範囲の後方非互換変更）**: いずれも新しいエラーコードは増やさず、422 `VALIDATION_ERROR`として返る。①`source_video`（素材（冒頭））×`end_source`×2クリップ以上（上記「排他」参照）。②最終クリップの潜在数が`kv + n_end_v`（のりしろ幅＋錨の潜在数）以下になる構成——逆向きに運ぶべき新規生成内容がゼロになるため、通るクリップ長を名指しした422で拒否する（既定値では到達しない）。**旧方式では通っていた構成が逆順Chainedでは422になる**点に注意（例: 最終クリップが極端に短い構成）。
- **のりしろ1×短いクリップ構成で既存の音声のりしろ枯渇422が新たに到達可能になる**: ③のりしろ1×短いクリップ構成では、既存の『音声のりしろ枯渇』422（`chain_math`の`sum_ka < n_join`）が逆順Chainedで初めて到達可能になる（窓内モードでは`kv >= 2`必須が覆い隠していた）。フロントエンドは`endSourceAudioOverlapBudget`ゲートで先回りブロックする。
- **`metadata.json`の新出キー（2026-08-18）**: `end_source.mode`に`"reverse"`が加わった。`end_source.generation_order`（幾何が宣言する生成順。3クリップなら`[2,1,0]`）、`end_source.stage1_order`（Stage-1ループが実際に生成した順の実測値。`generation_order`と一致することがエンジンの正しい実行の証明）、`end_source.stage1_freezes`（各セグメントの実行時凍結記録の配列。`{seg, fkv, ftv, fka, fta}`）が新出。
- **`metadata.json`の新出キー（第3弾・錨への素材音声の凍結、2026-08-18追加）**: `end_source.n_end_a`（錨区間が凍結する音声潜在の本数、幾何計算値）・`end_source.end_tile_bands_a`（音声版Stage-2タイル帯、幾何版）・`end_source.audio_frozen`（真偽値。音声つき素材ならtrue）・`end_source.audio_fallback_reason`（`null`／`"no_audio"`。partialはこのキーではなく`audio_frozen=true`＋`n_end_a_frozen < n_end_a`で表現する）・`end_source.n_end_a_frozen`（実際に凍結した音声潜在の本数。端数不足以外は`n_end_a`と一致）・`end_source.end_tile_bands_a_frozen`（実効タイル帯。端数不足で再計算されたときだけ`end_tile_bands_a`と異なる）・`end_source.end_fully_frozen_tiles_a`（音声帯が全数凍結されたタイルの番号一覧）が新出。`freeze_proof`へ`s1_audio_tail`/`s2_audio_tail`（無条件ゼロ期待。`s1_expected_zero`は映像専用の判定基準で音声には適用されない）が加わった。
- **エンジン系統による違い（2026-08-26追加）**: **`end_source`は2026-08-26からLTX 2.5でも使える。** モード判定（`chain_math.compute_chain_layout`が1箇所で行う）・受理範囲・`metadata.json`の`end_source`ブロックは**LTX 2.3と完全に同一**である（幾何計算が両系統から呼ばれる共有コードのため）。**上のオーナー裁定（クリップ1本を推奨、2本以上は推奨外）もそのまま引き継がれる。** ただし**推奨外の逆順Chainedは、LTX 2.5ではLTX 2.3より音声の継ぎ目が悪い**（**数値の正本はバックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §78.9①**。映像側はLTX 2.5でも問題なく、**推奨のクリップ1本〔窓内モード〕には影響しない**——窓内モードはセグメントが1つだけで、この差が生じる余地が無い）。**2026-08-30の目視・試聴ゲートG8で、オーナーはこの継ぎ目を実際に聴き取ったうえで「想定の範囲内なので合格」と裁定している**（裁定の正本は同 **§78.14(3)**）——**劣化は仕様として許容済みである。** したがって**フロントエンドとしては警告文の出し分けは不要**である——推奨外である旨の警告はすでにクリップ2本以上のときに出しており、エンジン別の文言を足すかどうかは台帳バックエンド[`PENDING_TASKS.md`](../../../Docs/PENDING_TASKS.md) §3-117 に起票してある（同項に残っているのは原因の究明だけである）。
- **旧方式（`internal_segment`）の契約**: 配信尺は `クリップ合計 − のりしろ + context_frames`（恒等式 `total_px == clips_total_px + end_context_px`）で、**`source_video`とは逆向き**（あちらは凍結した頭を配信尺から差し引く）。帯はstage-2のタイルを何枚またいでもよく（`ChainLayout.end_tile_bands`）、総尺上限`MAX_CHAIN_TOTAL_PIXEL_FRAMES`にも数えない。**2026-08-18から、通常のAPIリクエストではこの方式に到達できない**（クリップ2本以上は逆順Chainedへ切り替わったため）。実装はテスト・切り戻し専用として温存してあり、削除の判断は実機検証後にオーナーが行う。経緯は[`PENDING_TASKS_CLOSED.md`](../../../Docs/PENDING_TASKS_CLOSED.md) §3-82／バックエンド[`VERIFICATION_LOG.md`](../../../Docs/VERIFICATION_LOG.md) §60。

**総尺上限**: `MAX_CHAIN_TOTAL_PIXEL_FRAMES = 24×481 = 11544`(`api/models.py`の同名モジュール定数、2026-07-14に`8×481=3848`から拡張)。**`end_source`の帯（旧方式のみ発生）はこの上限に数えない**（2026-08-16）。窓内モード・逆順Chainedとも凍結分がクリップの内側にあるため、そもそも加算する帯が存在しない。

### 5.3 LoraSpec(`api/models.py::LoraSpec`)

`{name(サーバー登録名、パス不可), strength(既定1.0, 0<s≤2.0), audio_strength(既定null, 0≤s≤2.0)}`。`/`・`\`・`..` を含む名前は拒否。

- **`audio_strength`(2026-08-02追加)**: LoRAの適用強度を、映像軸(既存の`strength`)と音声軸(`audio_strength`)の2軸で独立に指定できるフィールド。**省略時(null)は音声側も`strength`の値にそのまま追従し、`audio_strength`を持たない従来のリクエストと完全同一の挙動になる**(後方互換)。範囲は0.0〜2.0で、既存の`strength`(`gt=0.0`、0は拒否)と異なり**0を許容する**——`0`を指定すると音声側の重みを一切適用しない(=適用そのものをスキップする)。これはLTX 2.3において、映像目的で訓練された画風LoRA(Style LoRA。追加学習した差分重みを本体モデルに足し込む仕組み)に含まれる音声側の差分重みが、生成音声を壊す(雑音・音割れ)というコミュニティの報告への対処。
  - **軸判定のルール**: 1本のLoRAが持つ多数の重みキーのうち、どのキーが音声軸でどのキーが映像軸かは、キーをドット分割して得たコンポーネント名が判定順のどのパターンと一致するかで決まる(`engine/gguf/ic_lora_common.py::classify_lora_axis`)。コンポーネント名`video_to_audio_attn`、および`audio_`で始まるコンポーネント名は音声軸。**`audio_to_video_attn`はコンポーネント名に`audio_`を含むが、映像ストリームへ書き込むクロス注意(音声を参照して映像を更新する層)なので映像軸として扱う特例**——クロス注意はソース側ではなく書き込み先のストリームで軸を判定する。`av_ca_v2a_gate_adaln_single`等のv2a/a2v変調項にも同種の特例があり、判定順の全体は[`LORA_AUDIO_STRENGTH_WORKORDER.md`](../../../Docs/LORA_AUDIO_STRENGTH_WORKORDER.md) §3の表を参照。それ以外のコンポーネント名は映像軸。
  - **音声関連の重みを持たないLoRA(例: 登録済みIC-LoRA、`Pixar_Toon`)では、`audio_strength`を指定しても効果がない(no-op)。** 分類対象となる音声軸のキー自体がLoRAに含まれていないため。
- **制御LoRA(kind=control)は `reference_video_id` 必須**。画風LoRA(kind=style)は不要。
- 既存UIはプロンプト内 `<lora:名前:映像の強さ:音声の強さ>` トークンをパースして `loras` 配列に変換(`gradio_ui/handlers.py::parse_prompt_loras`、映像側weight 0.05-2.0にクランプ)。**この記法はフロント側の慣習であり、API自体は `loras` 配列で受ける**。
  - **第3引数(音声の強さ、2026-08-02追加)**: 省略時は映像の強さにそのまま追従する。クランプ範囲は**0〜2.0**で、映像側の0.05〜2.0とは下限が異なる(音声側は0を許容)。
  - `<lora:名前::0>`(映像側の引数だけ省略)は**Gradio側とフロントエンド側で解釈が食い違うため使用しないこと**。Gradio側は`_LORA_TOKEN_RE`が数値限定マッチのため空の第2引数にはマッチせず、タグはLoRAとして解釈されずプロンプト本文にそのまま残る。フロントエンド側は`LORA_TAG_RE`が任意文字列にマッチするため空の第2引数でもタグとして認識され、`normalizeStrength`が既定値`1.0`を返して`{strength: 1.0, audio_strength: 0}`として送信される(映像側は既定強度で適用、音声側だけ0)。音声の強さだけを指定したい場合も映像の強さを明示すること。
  - 画風LoRAのカード(Library・Gradioギャラリーとも共通)をクリックしたときの既定挿入トークンは、2026-08-02以降 `<lora:名前:1.0:1.0>` (映像・音声とも1.0)になった。第3引数を最初から見せることで音声強度制御の存在に気づけるようにする狙いで、音声側1.0は映像追従時の実効値と数値が同一なので生成結果自体は変わらない。

### 5.4 生成モードまとめ

| モード | エンドポイント | 判定 |
|---|---|---|
| T2V | `/generate` | conditioning_images空 |
| I2V(最大5キーフレーム) | `/generate` | conditioning_images 1-5件 |
| クリップ連結 | `/generate/chain` | clips 2-24本(source系フィールドが無い場合の最低本数は2本) |
| V2V継続 | `/generate/chain` | source_video指定 |
| End source（素材（末尾）） | `/generate/chain` | end_source指定（1本＝窓内モード〔推奨〕／2本以上＝逆順Chained〔受理されるが推奨外、境目に品質劣化あり。LTX 2.5では音声の継ぎ目がさらに悪い。§5.2参照〕。どちらも出力尺はクリップ合計のまま変わらない。source_videoとの併用はクリップ1本のときのみ可能で、冒頭と末尾を与えた補間になる〔2本以上は422〕。旧方式`internal_segment`は通常のAPIリクエストからは到達不能。**LTX 2.5でも2026-08-26から使える**） |
| Retake（撮り直し） | `/generate/chain` | retake指定（`clips`はちょうど1本で、その`num_frames`が窓長そのもの。§5.2参照）。既存動画の窓の真ん中だけを作り直し、前後の「のりしろ」は凍結したまま返す。**LTX 2.5でも2026-08-26から使える**（窓は121フレーム以上を推奨） |
| A2V | `/generate/chain` | source_audio指定(クリップ1〜24本。音声はチェーン全体へ1本だけ添付し、窓割り当てはサーバー側) |
| 制御LoRA(canny/pose/depth/deblur) | `/generate` or `/chain` | loras(control)+reference_video_id |

`chain_math.py`: torch非依存の純Python幾何計算。latentフレーム数・stage2タイル・つなぎ目インデックスの単一の真実。フロントは直接使わないが、尺上限や8n+1の計算根拠。

## 6. JobResponse(進捗・結果スキーマ)

`GET /jobs/{id}` / `GET /jobs` の1件(`api/models.py::JobResponse`、`services/job_store.py::JobRecord.to_response`):

```json
{
  "job_id": "uuid", "status": "queued|running|completed|failed|cancelled",
  "progress": 0.0,
  "current_step": 3, "total_steps": 8,
  "stage": "stage1_denoise",
  "clip": 2, "clip_count": 4,
  "is_v2v": false, "joined": false,
  "created_at": "2026-06-25T12:00:00Z",
  "started_at": "...", "completed_at": "...",
  "error": "GENERATION_FAILED: ... (detail)",
  "request": { "...GenerateRequest全体..." : "" },
  "result": {
    "video_url": "/api/v1/jobs/{id}/video",
    "duration_seconds": 2.04, "resolution": "1280x720",
    "file_size_bytes": 123456, "generation_time_seconds": 168.3,
    "seed_used": 12345,
    "output_path": "outputs/{id}/output.mp4",
    "metadata_path": "outputs/{id}/metadata.json"
  }
}
```

- `resolution` は crop_output があればそれ、無ければ生成サイズ(`services/pipeline_manager.py::PipelineManager._finalize`)。
- タイムスタンプ形式: `YYYY-MM-DDTHH:MM:SSZ`(UTC)。
- **`is_v2v: bool`（2026-07-21・Join復活で追加）**: このジョブがv2v（`chain_request.source_video`由来）かどうか。フロントのJoinボタン表示ゲートはこのフラグのみで判定する。**`request`にはホワイトリスト（`to_clip_request`）で組むため`source_video`が構造的に含まれない**ので、v2v判定に`request`を覗いてはならない（実装以来Joinボタンが実機で表示されなかった旧バグの真因）。
- **`joined: bool`（2026-07-21で追加）**: `outputs/{id}/joined.mp4`が存在するか（＝Join済みか）。フロントは`GET /jobs/{id}/joined`の可否とJoin/Unjoin表示の初期値にこれを使う（ファイル削除で自然に`false`へ戻る）。

## 7. 未実装・未接続(フロント設計で重要)

> **UIに出してはいけない機能**
>
> [`../Mock/AVIUTL2_DESIGN_BRIEF.md`](../Mock/AVIUTL2_DESIGN_BRIEF.md) §5「変えてはいけない制約」の項目1が明示: 以下はAPIにフィールドはあるが engine に繋がっていないため、**UIコントロールとして表出させてはならない**（グレーアウト等で「場所のみ確保」は可）:
>
> - ~~ネガティブプロンプト(`negative_prompt`)~~ → **2026-07-28、NAG（Normalized Attention Guidance。CFGを使わずにネガティブプロンプトを効かせる手法）経由で条件付き解禁**（バックエンドコミット2ae497b）。CFGを迂回する専用の`nag_*`フィールド一式が実際にengineへ配線されたための解禁で、下のCFG/ステップ数/pipelineの禁止は変わらず有効（NAGはこれらを迂回する別経路のため）。フロント側の実装・UI詳細は`Docs/DEVLOG.md`の該当節、[`../Mock/AVIUTL2_DESIGN_BRIEF.md`](../Mock/AVIUTL2_DESIGN_BRIEF.md) §5・§11参照。**2026-07-29には方式選択が`neg_method`で解禁され、VSF（Value Sign Flip。正負のコンテキストを連結し1回のattentionで済ませつつ負側のVだけ符号反転×スケールする、NAGに続く2つ目の非CFGネガティブプロンプト手法）も選べるようになった**（`vsf_scale`、0-10、既定1.5）。フロントはNAG/VSFの方式をラジオで選び、方式に応じたスライダー(NAG: scale/tau/alpha、VSF: vsf_scale)を出し分ける。
> - CFG/ガイダンス強度スライダー(`guidance_scale`): distilled固定1.0。**引き続き禁止**
> - 品質モード切替(`pipeline: two_stage_hq`): 実質distilledのみ。**引き続き禁止**
> - 生成ステップ数変更(`num_inference_steps`): 8固定。**引き続き禁止**

### 7.1 モックのみ＝「出すが無効化する」機能（2026-07-31追加）

上の「UIに出してはいけない機能」とは**別のカテゴリ**として、**UIに出すが常に操作できない状態にしておく**フィールドがある。生成の高速化（Acceleration）の項目のうち、実装がないものがこれにあたる。**モックとして残っているのは`vae_mode`の1件**である。

| フィールド | UIでの見え方 | 実体 |
|---|---|---|
| `vae_mode` | Default/PruneVAEDの疑似ラジオを**表示するが常時disabled** | 枝刈り版のVAEデコーダ（映像の復元処理の軽量版）。**未実装** |

**なぜ隠さずに出すのか**: 「これから何ができるようになるか」を利用者に見せる意図があり、オーナー確定の方針である。バックエンド側もリクエストとしては受理する（Pydanticのフィールドとして存在する）。

**「表示だけで実際は効かない」罠を避けるための約束事**（過去に`fp8_transformer`が「statusには出るがworkerの挙動を変えない」状態で長く残った反省から、今回は次の3点を構造で担保している）:

1. **サーバーの能力公開には載せない**。`GET /status`の`acceleration.attention_backends`（§3.1）に並ぶのは実装のある選択肢だけで、モックの`vae_mode`はどこにも現れない。
2. **workerへは渡さない**。ただし`metadata.json`と`GET /jobs`の`request`エコーには**現れる**（`model_dump()`をそのまま出す既存の作法。`pipeline: "two_stage_hq"`と同じ前例）。「ジョブ記録に残る」ことと「効く」ことは別である点に注意。
3. **実際に使われた方式は必ず記録される**。実装のある`attention_backend`については、生成後に`outputs/{job_id}/metadata.json`の`attention_used`（`"sdpa"` / `"sage"` / `"sage->sdpa"`）で実効値を確認できる。フロント側の実機ゲートもログではなくこの値で判定した。

**フロントが送る値**: **モックの`vae_mode`はこのWebUIから一切送らない**（バックエンドが受理するだけで何もしないフィールドを送る意味がないため。型の上では設定オブジェクトに載せてあり、「決して送らない」というルール自体がテストで固定できる形にしてある）。送るのは実装のある4フィールド（`attention_backend`・`block_swap_prefetch`・`keep_resident`・`fused_gguf_dequant_kernel`）だけで、それぞれ**サーバー既定から変わったときにだけ**リクエストへ加算される（`shell/accelerationSettings.ts`の`accelerationRequestFields()`が単一の集約点。すべて既定なら`{}`を返すのでリクエストJSONは従来とバイト等価）。永続化は`localStorage`の`nzvideomni.acceleration`キー（保存するのはこの4フィールドに対応する選択で、モックの`vae_mode`は保存しない）。

その他の設計上の注意:

- キャンセルは即時でない(完了後に`cancelled`へ遷移)。UIは「キャンセル要求済み」表示が必要。
- 同時1ジョブ。実行中の新規投入は409。「混雑中」表示が必要（設計上は「予約」して現ジョブ完了後に自動送信する扱いを推奨。詳細は [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) M3参照）。
  - **追補（2026-07-17・予約機構は廃止）**: この「予約（現ジョブ完了後に自動送信）」方式はフロントエンドから取り除きました。実行中は生成ボタン自体をビジー状態で押せないように無効化する方式に変更しています（同時1ジョブという制約は変わらず、上の409応答の仕様もそのまま）。まれに状態の反映が遅れて409が返った場合は、生成ボタンのすぐ下にエラーを表示し、もう一度押せば再送信できます。
- ジョブ履歴はin-memory(再起動で消滅)。
- README冒頭のPhase1制限記述は古い(V2V/A2V/IC-LoRA/複数キーフレームは実装済み)。実装状況は各`api/*.py`が正。

## 8. 性能・UX前提

- **ロードは遅い**: real backend初回生成時にworker spawn+モデル構築(~28GB)。数分規模。「モデル読込中」の扱いが必要。
- **VRAM**: 16GB前提。512×320でpeak~9.2GB、720p実証済み。
- **生成時間の実測**(一次資料 `Docs/RESOLUTION_DURATION_CAPABILITY.md`):

  | 解像度 | 5秒生成 | 快適上限 | 上限での時間 |
  |---|---|---|---|
  | 720p(1280×768) | 約2.8分 | 約10.7秒(257f) | 約4.5分 |
  | 1080p(1920×1088) | 約4.3分 | 約6.3秒(153f) | 約5.7分 |
  | 1440p(2560×1472) | 約11分 | 約3.3秒(81f) | 約5.6分 |

  **上表の「快適上限」は2026-07-01時点の実測値である。** 現在の値は`GET /config`の`limits.spill_free_frames`(実体は`config.yaml`、説明の正本はバックエンド[`COMFORT_LIMIT_TABLE.md`](../../../Docs/COMFORT_LIMIT_TABLE.md) §付記。2026-08-31に再測定)を参照のこと——**フロントは必ず配信値を読み、この表の数字を焼き込まないこと。** 超過で2-4倍低速化(OOMせず)。フロントは警告表示(禁止ではなく注意喚起)。

- サーバー未起動/落ちている状態を「サーバー未起動」と表示。`GET /status` の疎通確認が起点。

## 9. テスト内のAPI利用例(`tests/`)

- `tests/test_smoke.py` — 基本疎通。
- `tests/test_join_api.py` — V2V join の使用例。
- `tests/test_ic_lora_api.py` / `tests/test_loras_endpoint.py` — LoRAエンドポイント。
- `tests/test_validation.py` — GenerateRequestバリデーション。
- `tests/test_gradio_*.py` — 既存UIハンドラ(呼び出しシーケンスの実例)。
- `tests/conftest.py` が `model.backend="mock"` を強制。
- curl 具体例はバックエンド `README.md` §5「16GB 向け生成テスト」。

## 10. 参考実装

- **`gradio_ui/api_client.py`** — 全REST呼び出しの集約実装。エンドポイント別のタイムアウト設定(load=600秒、upload/video=300秒、join=120秒等)の実例として最重要。
- **`gradio_ui/handlers.py`の`_poll_job_until_done`** — ポーリングループの実装(interval=1.0秒、timeout=7200秒[120分、2026-07-14延長])。`stage`値の扱い、chainの`clip`/`clip_count`表示、プロンプト内`<lora:name:strength>`タグのパース(`parse_prompt_loras`)も参照。
- **`mcp_server/`**(2026-07-28新設) — このAPIのもう一つのクライアント。Claude CodeなどMCP(Model Context Protocol)クライアントから、Web操作パネル相当の22ツールでバックエンドを直接操作できる(AviUtl2タイムライン連携は対象外)。フロントエンド(`.aux2`)実装への影響は無い別系統のクライアントだが、参考として挙げておく。詳細はバックエンド`README.md` §8、`Docs/MCP_SERVER_DESIGN.md`参照。

## 11. 推奨実装フローまとめ

1. 起動時 `GET /api/v1/status` で疎通・GPU・混雑確認。失敗=「サーバー未起動」。
2. `GET /api/v1/config` でプリセット・制約・警告閾値を動的取得(ハードコードしない)。
3. `/upload/*`(multipart, field=`file`)→ `/generate`(202)→ 1秒ポーリング → `/jobs/{id}/video`。
4. LoRA: `GET /loras` + サムネは `<img>`直参照。
5. §7の未接続機能はUIに出さない。
6. 数分待ち前提 → 進捗可視化、混雑(409)、キャンセル非即時、spill超過警告を設計に織り込む。
7. api_key設定時のみ書き込み系にBearer。
8. 最重要参考実装: `gradio_ui/api_client.py`(全REST集約)と `gradio_ui/handlers.py`の`_poll_job_until_done`(ポーリングループ)。

## 12. 付録: フロントエンド開発中に発見された挙動

> 本節は2026-07-07〜08のフロントエンド(`.aux2`)実装中に判明した、本文(§1-11、2026-07-07時点の調査)には無かった挙動の追記。本文は改変していない。詳細な経緯は[DEVLOG.md](DEVLOG.md) §4を参照。この時点ではバックエンドのコード自体は変更していない(凍結方針のまま)。なお凍結方針は絶対の禁止ではなく、後の2026-07-21にV2V結合(Join)復活のための限定解除・拡張が入っている(§3.17・§3.18・§6)。

### 12.1 mockバックエンドはchainジョブの`clip`/`clip_count`を常に`null`で返す

§6のJobResponse例は`"clip": 2, "clip_count": 4`のように記載しているが、これは**実バックエンド(`real`)の挙動**を示したものである。`config.model.backend=mock`で動作するmockバックエンド(`services/ltx_runner.py::_MockBackend.generate_chain`)は`progress_callback(None, None, progress)`のみを呼び出し、`clip`/`clip_count`キーワード引数を一切渡さない。`services/pipeline_manager.py`のchain用`on_progress`は`clip is not None`のときにしか`job.clip`/`job.clip_count`を更新しないため、**mockバックエンドで確認する限り`GET /jobs/{id}`のchainジョブは`clip: null, clip_count: null`のまま**である。フロントの進捗UIはmock検証時にこれを前提に設計する必要がある(実バックエンドでは`clip`/`clip_count`が入る想定でUIを組んでおくこと)。

### 12.2 `reference_video_id`(§5.1)のフロント送信配線が完成(2026-07-08、右クリック生成の配線セッション)

§5.1に記載の`reference_video_id`(IC-LoRA参照動画、`/generate`専用、参照動画ジョブは width/height 128の倍数必須)は、**バックエンドAPI仕様としては1.0.0-rc1の時点で既に記載済み**であり、本節での追記は仕様値の変更ではない。1.0.0-rc1時点ではCreate画面のフォーム(`webui/src/modes/create/useGenerationForm.ts`)がこのフィールドを組み立てておらず、右クリックメニューの「IC-LoRA」アクションはM6の`menuRouting`がChain画面(`/generate/chain`)へ誤って送る配線になっていた(chainには`reference_video_id`が存在しないため破綻する不整合。[TIMELINE_ALPHA_REQUIREMENTS.md](TIMELINE_ALPHA_REQUIREMENTS.md) 第7-3-A参照)。

今回のセッションで是正した内容(フロント側の配線のみ。バックエンド仕様・制約値は無変更):

- `menuRouting`の行き先をChain→Createへ付け替え。
- Createフォームに参照動画アップロードUI(Chainの`useSourceUpload`を再利用)と、リクエスト型`GenerateRequest`(`webui/src/api/types.ts`)への`reference_video_id?: string | null`追加、`toGenerateRequest`への合流を実装。
- IC-LoRA有効時はwidth/heightを128の倍数に丸めるようフォーム側で対応(§5.1の「参照動画ジョブは128の倍数必須」制約への追従)。

**未解消の穴(次段)**: §5.1のとおり`reference_video_id`は単独指定不可でcontrol LoRA(`loras`)同伴が必須だが、今回のフロント配線はこの同伴をUI側で強制していない。ユーザーが参照動画のみを設定してcontrol LoRAを指定しないまま送信すると422になり得る(バックエンド仕様・制約値は無変更。[DEVLOG.md](DEVLOG.md) §7.7参照)。

**追補(2026-07-15、N2/N3で解消済み)**: 上記の穴は`WEBVIEW2_PARITY_BACKLOG.md`のN2(IC-LoRAコントロールLoRAの選択ドロップダウン)・N3(Create画面のIC-LoRA強度スライダー)の実装セッションで解消した。Create画面は現在、参照動画とcontrol LoRA(`loras`)を双方向に強制するゲート(`isValid`が「参照動画があるのに`loras`が空」「control LoRAを選んだのに参照動画が無い」の両方で送信をブロック)を実装しており、上記422の穴は塞がっている。また`conditioning_attention_strength`／`reference_video_strength`の強度スライダー2種も、Chain画面だけでなくCreate画面(通常の`/generate`)からも送信できるようになっている。詳細は[DEVLOG.md](DEVLOG.md) §11、[`WEBVIEW2_PARITY_BACKLOG.md`](WEBVIEW2_PARITY_BACKLOG.md) N2/N3を参照。

**フロント型定義の非対称(解消済み)**: 直前の追補(2026-07-15、N2/N3で解消済み)のとおり、この段落が指摘していた非対称は同セッションで解消されている。現在の`webui/src/api/types.ts`の`GenerateRequest`型は`reference_video_id`に加え`conditioning_attention_strength`/`reference_video_strength`の2フィールドも宣言済み(`types.ts`の`GenerateRequest`)で、Create画面のフォーム(`toGenerateRequest`)から単発生成側でも送信できる。`GenerateChainRequest`(§12.4)側の対応する2フィールドと合わせ、Create/Chain両経路で対称になっている。

詳細は[DEVLOG.md](DEVLOG.md) §7.4を参照。

### 12.3 `overlap_frames`にはclip側の`num_frames`に依存する未文書化の上限がある

§5.2は`overlap_frames`の制約を「int `3`(既定)、1-8」としか記載していないが、実際には**各クリップのstage-1 video latentフレーム数未満**という追加制約がある(`chain_math.py::compute_chain_layout`: `v_latent_frames(num_frames) ≈ (num_frames-1)//8+1`。`overlap_frames >= min(v_latent_frames(c.num_frames) for c in clips)`の場合、`"overlap_frames (K_v=...) must be < every clip's stage-1 latent frames [...]"`という例外で拒否される)。`num_frames`が小さいクリップ(例: 9〜16フレーム、stage-1 latentは2〜3)を含むchainでは、`overlap_frames`の既定値3でもこの制約に抵触し得る。フロント側は「1-8」の範囲チェックだけでなく、各clipの`num_frames`から`overlap_frames`の実効上限を動的に計算しUIで警告/クランプすることが望ましい。

### 12.4 chainの拡張(clips24枠・chunked_upsample・loras/reference_video_id全面解禁)とポーリングタイムアウト延長(2026-07-15)

バックエンド側で2026-07-03〜07-14にかけて`GenerateChainRequest`(§5.2)が段階的に拡張された。本節はその反映であり、この拡張はフロントエンド側の改修ではない(凍結方針のもとでバックエンド側が行った拡張を、フロントが追随して記録したもの)。その後2026-07-21には、V2V結合(Join)復活のためにフロント側の要件からバックエンドAPIを限定的に拡張した実績もある(§3.17・§3.18・§6)。

- **`clips`上限を8本→24本に拡張**(2026-07-14、`api/models.py::GenerateChainRequest`の`clips`)。同時に総尺上限`MAX_CHAIN_TOTAL_PIXEL_FRAMES`も`8×481=3848`→`24×481=11544`ピクセルフレームへ拡張(`api/models.py`の`MAX_CHAIN_TOTAL_PIXEL_FRAMES`)。
- **`loras`をchainへ解禁**(2026-07-03に受理開始、2026-07-11に全面解禁)。§5.3の挙動(プロンプト内`<lora:name:strength>`タグのパース)は`/generate`と`/generate/chain`の両方に等しく適用される。全クリップ・全ステージへ一律の強さで適用され、クリップ別の強さ指定は無い(`api/models.py::GenerateChainRequest`の`loras`)。
- **`reference_video_id`/`conditioning_attention_strength`/`reference_video_strength`をchainへ解禁**(2026-07-11、owner決定によりalphaスコープ)。§5.1の同名フィールドと型・範囲は同一だが、chain側は追加制約として**loras必須・clips=1本限定・source_videoと排他**が掛かる(`api/models.py::GenerateChainRequest.validate_chain_constraints`)。
- **`chunked_upsample`(bool, 既定`false`)を新設**(2026-07-14)。組み立て済みのstage-1タイムラインをチャンク単位(halo overlap＋CPU offload)でアップサンプルする経路へのopt-inで、既定`false`は旧来の一括GPUアップサンプル(総尺に比例してVRAMを消費)のまま据え置く。長尺/高解像度chainを16GBに収めるための機能(`api/models.py::GenerateChainRequest`の`chunked_upsample`)。**フロントはこのフィールドを省略せず常に明示送信すること**(WebUI既定値`true`)。省略すると恒久的に旧来の一括アップサンプル経路にフォールバックする。
- **ポーリングタイムアウトを3600秒→7200秒(120分)に延長**(2026-07-14、`gradio_ui/handlers.py`の`_poll_job_until_done`の`timeout_s`既定値・`_resolve_poll`のフォールバック他)。§4・§10の記述を合わせて更新した。

`webui/src/api/types.ts`の`GenerateChainRequest`は本節の内容を反映済み(2026-07-15)。
