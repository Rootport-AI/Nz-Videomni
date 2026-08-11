# LTX 2.3 バックエンド仕様書

LTX 2.3 動画生成 REST API バックエンド（16GB VRAM 向け・2プロセス/2venv・FastAPI + Gradio）

| 項目 | 値 |
|------|----|
| 版 | **v0.5.5** |
| 日付 | **2026-07-28**（v0.5 本体は 2026-07-02。以後の更新は §0.1 の改訂履歴を参照） |
| 前版 | `LTX23_Backend_Specification_v04_Phase1_T2V_I2V.md`（v04・全面改訂の元。本書で置換） |

## 目次

- §0 この文書について
- §1 プロジェクト概要とゴール
- §2 実行環境と環境分離
- §3 ネットワークと起動
- §4 アーキテクチャ
- §5 モデル構成（実行 ~28GB・取得 ~30GB）
- §6 【凍結】API 契約 ← 本書が正本
- §7 ジョブ管理と Pipeline / Runner
- §8 出力ファイル
- §9 低VRAM戦略（Phase 1 実装済み・既定 ON）
- §10 解像度×尺の能力と音声
- §11 config.yaml
- §12 Gradio 検証UI
- §12b MCPサーバー（AIエージェント連携）
- §13 開発フェーズとロードマップ
- §14 AviUtl2 / DaVinci Resolve 連携（Phase 2 = ゴール）
- §15 ログ・エラーハンドリング
- §16 受け入れテスト
- §17 ライセンスと provenance
- 付録A LTX 制約早見表
- 付録B 用語・SSOT ドキュメント地図

---

## §0 この文書について

### 0.1 版メタ

| 項目 | 値 |
|------|----|
| 版 | **v0.5.10** |
| 日付 | **2026-08-11** |
| 対象 | LTX 2.3 動画生成 REST API バックエンド（16GB VRAM 向け・2プロセス/2venv・FastAPI + Gradio） |
| 前版 | `LTX23_Backend_Specification_v04_Phase1_T2V_I2V.md`（v04・全面改訂の元） |

本書は前版 v04 の**全面改訂版**である。v04 は当初計画（公式 `ltx_pipelines` safetensors ローダ + fp8-cast + cu129 + xformers 前提）の章立てを引きずっていたため、本 v0.5 では **実装現実に合わせて章立てから書き直した**。

#### 改訂履歴

| 版 | 日付 | 内容 |
|----|------|------|
| v0.5 | 2026-07-02 | v04 からの全面改訂（章立てを実装現実に合わせて書き直し）。 |
| v0.5.1 | 2026-07-26 | α版インストール導線の整備に追随して **10 箇所**を更新。**§0.3**（SSOT 地図に `config.yaml.example` を追加し、`config.yaml` を git 追跡外にしたこと＝clone 直後には存在しないことを明記）／**§2.4**（`setup.bat` / `run.bat` はラッパーで環境変数を設定しないこと、`tools/uv` と `tools/ffmpeg/bin` をプロセスの `PATH` 先頭へ足すこと）／**§2.5**（`setup.bat` → `scripts/setup.ps1` → `install_ltx.ps1` の導線、取得元の 3 リポジトリ化、冪等の粒度＝venv 再同期とモデルガード）／**§3.3**（`run.bat`、`run.ps1` のリポジトリ直下固定、`uv sync` を行わない設計、起動バナー）／**§4.4**（ディレクトリ構成に `setup.bat` / `run.bat` / `config.yaml.example` / `.au2pkg.zip` を反映）／**§5 の章見出しと目次**（「モデル構成」→「モデル構成（実行 ~28GB・取得 ~30GB）」＝実行に要る量と取得量の区別を見出しに出した）／**§5.1**（`INSTALLED_PATHS.txt` が 9 行であることの内訳、および §5.1b を含まない旨の明示）／**§5.1b＝完全新設**（インストーラが追加取得する IC-LoRA 2 点・DWPose 前処理器 2 点、取得総量 31,889,519,494 B、検証表が 14 項目である理由）／**§5.4**（GPU アーキ自動判定と `wheels/` プリビルド wheel 自動導入の**廃止**、`build_xformers.ps1` は手動ツールとして存置）／**§11.2**（`model.ic_loras` の行を追加＝§5.1b が参照している登録の本体）。**同日の第 2 次敵対的レビューによる訂正**: §2.5 の冪等ガードの記述を実装（`-Check` によるディレクトリ単位の独立判定）に合わせて全面的に書き直し（旧記述は廃止済みの `-CheckDir`＋`MinBytes` 合計方式＝Gemma のデッドロックを再発させる誤りだった）、§5.2 と付録B に「43GB / 46GB は同一ファイル」の表記注記を追加、本履歴表自体の記載漏れ（§5 見出し・§5.1・§5.1b・§5.4）を補完。正本はフロントエンド側 `Docs/PENDING_TASKS_CLOSED.md` §3-36・§3-37。 |
| v0.5.2 | 2026-07-27 | サブマシンでの実機検証（`Docs/NEXT_SESSION_HANDOFF.md`・`README.md` §7）の反映と、実装との乖離を潰す収束修正。**§2.1**（ffmpeg は「PATH に通す」ではなく `scripts/setup.ps1` が `tools/ffmpeg` へ取り込む＝§2.4 の PATH 前置で解決される）／**§2.5**（`setup.ps1` の事前チェック 3 種＝空き容量・ページファイル・GPU がいずれも警告のみであること、および `run.ps1` の二重起動ガードの存在を追記）／**§3.3**（URL を控えに入れる処理は現行の `run.ps1` に存在しない＝利用者に見せる URL は `main.py` の起動バナーが唯一の正本、という実装に合わせて訂正）／**§4.4**（ディレクトリ構成のルートフォルダ名を実際の `Nz-LTX23-backend/` へ訂正）。正本はフロントエンド側 `Docs/PENDING_TASKS_CLOSED.md` §3-38。 |
| v0.5.3 | 2026-07-28 | NAG（Normalized Attention Guidance＝非CFGネガティブプロンプト機能）の追加を反映。**§6.2**（`GenerateRequest` に `nag_enabled`/`nag_scale`/`nag_tau`/`nag_alpha` の4フィールドを追加し `negative_prompt` に `max_length=2000` を付与、凍結制約に相互検証を追加。`GenerateChainRequest` にも同一フィールドが存在する旨を補足）／**§12.2**（共有プロンプト直下の Negative Prompt アコーディオンを追記）。凍結API契約（§6）への追加は、2026-07-21 の V2V Join 拡張（`d22706e`）と同じく既存フィールドの意味変更を伴わない加算のみで、実装・機械検証・実機ゲート状況の正本は `Docs/VERIFICATION_LOG.md` §38。 |
| v0.5.4 | 2026-07-28 | 同日の第2コミット（`2f8e85b`）で追加された **MCPサーバー**（`mcp_server/`。Claude Code 等の MCP クライアントからバックエンドを操作する22ツール）を反映。**§12b＝完全新設**（位置づけ・22ツールの概要・依存関係・実機ゲート状況へのポインタ）／**§0.3 の SSOT 地図**（`Docs/MCP_SERVER_DESIGN.md` を追加）／**§4.5**（アプリ venv の技術スタック表に `mcp` パッケージを追加）／**付録B.1**（用語集に MCP のエントリを追加）。凍結 API 契約（§6）そのものへの変更は無い（MCPサーバーは既存 `/api/v1/*` を叩く追加のクライアントであり、REST API 契約は無改修）。詳細な設計判断は `Docs/MCP_SERVER_DESIGN.md`、利用者向け説明は `README.md` §8、機械検証・実機ゲート状況は `Docs/VERIFICATION_LOG.md` §39 が正本。 |
| v0.5.5 | 2026-07-28 | 同日の依存整理バッチ（未使用パッケージ削除・`uv sync --extra dev` 常時化・`checkpoint_path` 後始末）を反映し、`checkpoint_path` が `config.model` から削除された事実に追随して**5箇所の記述を訂正**した。**§4.3**（`checkpoint_path` はそもそも対応する config フィールドが無くなり、worker payload への値は `services/ltx_runner.py` が直値の `""` をハードコードする、という実装に合わせて訂正）／**§5.1**（`models/INSTALLED_PATHS.txt` の行数を「9 行」→「**8 行**」に訂正。`checkpoint_path (ref-only)` の行は `config.yaml` からの削除に伴い消えた）／**§5.2**（`config.model.checkpoint_path` は「フィールドとしては残る」ではなく**削除済み**、worker payload には `services/ltx_runner.py` のハードコードとして残るのみ、と訂正）／**§5.5**（`checkpoint_path` は「config に残るが reference-only」ではなく**削除済み**、と訂正）／**§11.2**（`config.yaml` の実値表から `checkpoint_path` の行を削除。現行 `config.yaml` にこのキーは存在しない）。凍結 API 契約（§6）への変更は無い（worker プロトコルの `checkpoint_path` フィールド自体は存続し、値がハードコード化されただけ）。実装・機械検証の詳細は `Docs/VERIFICATION_LOG.md` §40 が正本。 |
| v0.5.6 | 2026-08-01 | Acceleration（生成の高速化）機能の追加を反映。**§4.5**（エンジン venv の技術スタック表に `sageattention` 2.2.0 と `triton-windows` を追加）／**§5.4**（章題を「attention バックエンド」のまま、既定は SDPA・ジョブ単位で SageAttention へ切替可能という現行仕様へ書き直し。xformers/flash-attn を導入しない方針そのものは不変）／**§6.2**（`GenerateRequest` に `attention_backend`／`fused_gguf_dequant_gemm`／`vae_mode` の3フィールドを追加。後者2つは受理のみでエンジン未消費の**モック**である旨を明記。`GenerateChainRequest` にも同一フィールドが存在する旨を補足）／**§6.5b＝新設**（`GET /status` の非凍結ブロック `acceleration`＝`attention_backends` と `sage_available` の真理値表）／**§6.6**（`metadata.json` トップレベルへの `attention_used` 追加）。凍結 API 契約（§6）への追加は既存フィールドの意味変更を伴わない加算のみで、既定値のジョブは worker ペイロードがバイト同一のまま。実装・機械検証・実機ゲート状況の正本は `Docs/VERIFICATION_LOG.md` §43、利用者向け説明は `README.md`「生成の高速化（Acceleration）」節。 |
| v0.5.7 | 2026-08-03 | `keep_resident`（モデルCPU骨格のジョブ間キャッシュ）の製品化を反映。**§6.2**（`GenerateRequest` に `keep_resident` を追加。`GenerateChainRequest` にも同一フィールドが存在すること、既定のジョブは worker ペイロードがバイト同一のままであること、`GET /status` には意図的に載せないことを補足で明記）／**§9.2**（本番既定の env 一覧から `LTX_KEEP_RESIDENT` を削除し、per-job フィールドへ移行した旨へ書き換え。あわせて 2026-06-30 当時の「keep=1 は native crash する」という記述に、現構成では修正済みで製品化されたという追記を添えた）。凍結 API 契約（§6）への追加は既存フィールドの意味変更を伴わない加算のみで、既定値のジョブは worker ペイロードがバイト同一のまま。実装・機械検証・実機ゲート状況の正本は `Docs/VERIFICATION_LOG.md` §48、利用者向け説明は `README.md`「モデル骨格の常駐（`keep_resident`）」節。 |
| v0.5.7b | 2026-08-03 | IC-LoRA Depth（深度制御）・Deblur（ぼけ除去）の追加を反映。**§5.1b**（Deblur 1 点・VDA 深度前処理器 2 点を取得対象に追加し、検証表を 14 → **16 項目**へ）／**§11.2**（`model.ic_loras` を `depth-control`・`deblur` を含む **5 エントリ**へ）。凍結 API 契約（§6）への変更は無い（登録済みアダプタ名が増えただけで、`loras[].name` の解決方式は不変）。仕様・設計判断の正本は `Docs/ICLORA_DEPTH_DEBLUR_WORKORDER.md`、実測と検証記録は `Docs/VERIFICATION_LOG.md` §49。 |
| v0.5.8 | 2026-08-04 | `fused_gguf_dequant_kernel`（GGUF 逆量子化の Triton 1カーネル化）の追加と、旧モック `fused_gguf_dequant_gemm` の完全撤去を反映。**§6.2**（`GenerateRequest` の表から `fused_gguf_dequant_gemm` の行を削除し、実装のある `fused_gguf_dequant_kernel`〔既定 `true`〕の行を追加。補足の「モック2件」を「モック1件（`vae_mode`）」へ書き換え、既定 `true` のフィールドが2つになったことに伴う worker ペイロードの記述も更新）／**§6.5b**（`/status` にモックを載せない理由の記述を1件構成へ更新）／**§6.6**（`metadata.json` トップレベルに `fused_gguf_dequant_kernel_used` を追加）。凍結 API 契約（§6）への変更は、加算1件と**受理のみで生成に一切影響しなかったモック1件の削除**である（pydantic の `extra=ignore` により、旧フィールドを送る古いクライアントがあっても API は壊れない）。既定 `true` への反転は実機ゲート G1〜G8 全項目合格を条件にオーナーが承認した（`block_swap_prefetch` の前例と同じ手順）。実装・機械検証・実機ゲート状況の正本は `Docs/VERIFICATION_LOG.md` §51、利用者向け説明は `README.md`「生成の高速化（Acceleration）」節。 |
| v0.5.9 | 2026-08-05 | **PrunaVAED（枝刈り版の映像VAEデコーダ）の実装**を反映。2026-07-31 から Acceleration 区画で唯一モックのまま残っていた `vae_mode` が実機能になり、**モックは1件も無くなった**。**§6.2**（`GenerateRequest` の `vae_mode` の行を「モック」から実装済みの記述へ全面書き換え。降格が環境ではなく重みファイルの有無で決まること・既定が恒久 off であること・実測値〔768p/257f で −12.54秒／`peak_vram_reserved_mb` −2,635MB／PSNR 36.06dB／SSIM 0.9854〕を明記）／**§6.2 の Acceleration 補足**（「モック1件」の記述を撤去し、モックの作法そのものは将来の枠のために記述として残す。`vae_mode` の降格セマンティクスと恒久 off の項を追加）／**§6.5b**（`/status` の acceleration ブロックに `vae_mode` を載せない理由を「モックだから」から「環境依存の可否ではなく、しかもジョブ単位で変わる判定だから」へ差し替え）／**§13.x の Gradio Settings タブ記述**（プレースホルダ表記の撤去）。表示名を **PruneVAED → PrunaVAED** へ訂正した（上流の正式名称の誤記だったため）が、**API のフィールド値 `"prune_vaed"` は既存の外部コントラクトなので改名していない**。凍結 API 契約（§6）への変更は無く、既定 `"default"` のジョブの worker ペイロードはバイト同一のままである（既定反転を行わないため鍵集合が増えない）。あわせて**MCP サーバーのツール引数へ `vae_mode` を公開**した（§12b。実機ゲート G1〜G7 の合格を待ってからの最終ステップ＝G9 として実施し合格。`submit_generate` / `submit_chain` の両方に列挙値2つで現れ、**ツール本数22は不変**。設計判断は `Docs/MCP_SERVER_DESIGN.md` D12）。実装・機械検証・実機ゲート G1〜G7＋G9 の正本は `Docs/VERIFICATION_LOG.md` §52、仕様と設計判断の正本は `Docs/PRUNAVAED_WORKORDER.md`、利用者向け説明は `README.md`「生成の高速化（Acceleration）」節。 |
| v0.5.10 | 2026-08-11 | **長尺IC-LoRA（クリップ別の参照動画）の実装**を反映。チェーン（`POST /generate/chain`）の `reference_video_id` が2026-07-11の解禁時点で持っていた「clips=1本限定」を撤廃し、**長い参照動画を1本だけ添付すると各クリップが担当する区間をサーバーが自動で切り出してstage-1にのみ注入する**方式へ拡張した。**§6.2**（`GenerateChainRequest` の`reference_video_id`系3フィールドの補足段落を新設し、`chain_math.video_segment_windows`の幾何・stage-1限定・参照不足時の穏当な劣化を明記）／**§6.8**（`LORA_CONTROL_UNSUPPORTED_IN_CHAIN`〔422、2クリップ以上での制御系IC-LoRA拒否〕を削除し、深度前処理〔Video-Depth-Anything〕がメモリに載らないことに由来する `LORA_DEPTH_CHAIN_UNSUPPORTED`〔422、2クリップ以上での`depth-control`拒否のみ〕へ置換。canny/pose等の他アダプタは多クリップで受理されるようになった）。あわせて `GET /loras` の各行へ `preprocess`／`reference_downscale_factor` を追加し、`metadata.json` へ `ic_lora` ブロック（`reference_segment_windows`等）を追加した（いずれも凍結表未掲載のADDITIVE要素）。凍結 API 契約（§6）への変更は、エラーコード1件の置換（他クライアントへの実害なし——旧コードはこの状況でしか出ず、出なくなっただけ）と、フィールドの制約緩和（1クリップ限定→1〜24クリップ）のみで、キーの追加・型変更は無い。実装・機械検証・実機ゲートG1〜G10の正本は `Docs/VERIFICATION_LOG.md` §57、利用者向け説明は `README.md`「Gradio UI」節、フロントエンド側の実装記録はフロントエンド`Docs/DEVLOG.md` §72・§73。 |

### 0.2 スコープ

本書は次の2点を扱う。

1. **Phase 1 として実装済みのバックエンド仕様**（T2V + 最小I2V + native 音声 + 16GB fit + 720p 級 + マルチジョブ）。凍結 API 契約（§6）はこの文書が正本。
2. **ゴール（AviUtl2/DaVinci Resolve 連携）への道筋**を、粗い粒度・広い視点で示すロードマップ（§13/§14）。

実装細部（エンジン内部・解像度×尺の網羅・実測秒数/GB/SHA）は各 Docs が担う。本書はそれらを重複させず、代表値＋ポインタに留める。

### 0.3 読み方と SSOT（Single Source of Truth）地図

**本文は実装現実を正とする。** 記述と実装が食い違ったら、実装（コード・config.yaml・各 Docs の実測）が正しい。数値を更新する場合は、本文ではなく**該当 Docs を直す**（本書は代表値を引くだけ）。

正本の分担は次の通り。

| 正本（SSOT） | 何の正本か |
|--------------|-----------|
| **本書 §6** | 凍結 API 契約・スキーマ・enum・エラーコード（この文書が正本） |
| `README.md` | 起動・2venv・セットアップ導線・API 概要・生成テスト手順 |
| `config.yaml` | 実行時設定の実値（presets / limits / vram ノブ / モデルパス）。**git 追跡外**（2026-07-26〜） |
| `config.yaml.example` | 配布されるひな型。**リポジトリに入っているのはこちらだけ**で、`config.yaml` は `setup.bat` / `run.bat` がここから複製する |
| `Docs/VERIFICATION_LOG.md` | 実機検証の全経緯・実測 peak_vram/秒数・SHA256 バイト一致・設計判断の根拠 |
| `Docs/MCP_SERVER_DESIGN.md` | MCPサーバー（`mcp_server/`）の設計判断（ツール分割・非同期化・エラー翻訳・`.mcp.json` 生成方式 等） |
| `Docs/ICLORA_DEPTH_DEBLUR_WORKORDER.md` | IC-LoRA Depth（深度制御）・Deblur（ぼけ除去）の仕様・設計判断 |
| `Docs/ACCELERATION_RESEARCH_NOTES.md` | 生成高速化の候補整理と採否判断 |
| `Docs/RESOLUTION_DURATION_CAPABILITY.md` | 解像度×尺の能力（spill-free 閾値・生成時間・den2 推定式・UI 含意）の正本 |
| `Docs/STORAGE_POLICY.md` | 保存領域（`outputs/` / `uploads/`）の方針と実構造。「Outputs は宝物、Uploads は事実上の一時ファイル置き場」という設計原則・ID の紐づき・ディスク整理ルールの正本 |
| `Docs/NEXT_SESSION_HANDOFF.md` | プロジェクトのゴール像・フェーズ別ロードマップ・設計対話の決定・削除スコープ |
| `Docs/LTX23_REFERENCE.md` | LTX-2/2.3 の一般知識（VAE 32×圧縮・2段パイプライン・÷64 の由来・VRAM スケーリング） |
| `engine/VENDOR_NOTICE.md` | `engine/` の由来・provenance・依存再現手順・ライセンス帰属 |

> **乖離時の原則**: 本文の数値（秒数・GB・上限フレーム等）は**代表値**であり、正確な網羅・最新実測は上表の該当 Docs が正。乖離を見つけたら Docs を正とし、Docs 側を更新すること。

> **`config.yaml` の SSOT としての読み方（2026-07-26 更新）**: `config.yaml` は 2026-07-26 に **git の追跡から外した**。したがって **`git clone` した直後のリポジトリに `config.yaml` は存在しない**。本書や README が「`config.yaml` の値」と書いているとき、リポジトリ上でその値を確認できる場所は `config.yaml.example` である。設定の**実値**の正本は各利用者の手元の `config.yaml`、**配布される既定値**の正本は `config.yaml.example` という二段構えになる。両者は同期させること（片方だけを編集すると、既定値と実値が静かに食い違う）。

---

## §1 プロジェクト概要とゴール

### 1.1 目的

**VRAM 16GB のコンシューマー GPU で LTX 2.3 の動画生成を動かし、REST API として公開するバックエンドサーバー**を構築する。生成した動画は、AviUtl2 / DaVinci Resolve などの**薄いフロントエンド**から利用する。

最終ゴールは **AviUtl2 拡張機能からこのバックエンドを使って動画を生成できること**（§14・Phase 2）である。ただし REST API は AviUtl2 専用にはしない。汎用設計とし、DaVinci Resolve 等の別フロントエンドからも同じ API を叩けるようにする。

検証・デモ用に Gradio UI（`/ui`・§12）を同梱する。この UI 自体が薄いクライアントの一例で、LTX を直接呼ばず自身の REST API を叩く。

### 1.2 設計思想

- **バックエンドの安定動作を最優先**とする。まず「動くツール」を確実に得る。
- **フロントエンドは HTTP API を叩くだけの薄いクライアント**とする（モデルロード・GPU 管理・低VRAM 最適化・ffmpeg エンコードはすべてバックエンド側）。
- **API を先に安定させ、汎用に保つ**。LTX 公式コードとの結合部は薄いアダプタ（`services/ltx_runner.py`・§7/§9）に閉じ込める。
- **16GB VRAM で開発中に毎回検証できる**ことを重視する。Low VRAM baseline は Phase 1 から有効。

### 1.3 early-integration（早く統合して育てる）

本プロジェクトの開発方針は「機能を全部固めてから統合する」ではなく、**「早く統合して、使いながら育てる」**である（`Docs/NEXT_SESSION_HANDOFF.md`「設計対話の決定」）。

具体的には、Phase 1 の最小バックエンドが 16GB で T2V/最小I2V/720p/マルチジョブまで通った時点で、次は機能拡張ではなく **AviUtl2 統合（Phase 2＝ゴール）へ進む**。長尺化・高度な条件付け（クリップ連結・IC-LoRA・V2V 等）は、統合して実運用しながら Phase 3 以降で育てる。ロードマップの詳細は §13、統合の役割分担は §14。

---

## §2 実行環境と環境分離

### 2.1 主要開発環境

| 項目 | 値 | 備考 |
|------|----|----|
| OS | Windows 10/11 x64 | 開発対象 |
| GPU | NVIDIA CUDA GPU / VRAM 16GB 推奨 | 実モデル実行時のみ必須。検証環境＝RTX 4070 Ti SUPER 16GB |
| RAM | 32GB 以上推奨 | block-swap / CPU オフロードの commit（仮想メモリ）予約に必要（§10 参照） |
| Python | 3.12 系 | uv 管理・プロジェクト内に固定 |
| CUDA | **12.8（cu128）** | engine 側 torch の wheel index |
| PyTorch | **`torch 2.9.1+cu128`** | engine 側実体（`engine/venv-engine.freeze.txt`） |
| NVIDIA ドライバ | CUDA 12.8 対応版 | cu128 wheel 実行要件。Blackwell は R570+ |
| パッケージ管理 | **uv** | Python 本体もプロジェクト内へ |
| 動画エンコード | ffmpeg（`scripts/setup.ps1` が `tools/ffmpeg` へ取り込む） | MP4 保存・クロップ用。`run.ps1` が `tools/ffmpeg/bin` をプロセスの `PATH` 先頭へ足すため、システムの `PATH` へ通す必要はない（§2.4） |

### 2.2 2プロセス・2venv 構成（概要）

このバックエンドは **2プロセス・2venv 構成**である（詳細なアーキテクチャは §4）。

| venv | 役割 | 主要依存 |
|------|------|----------|
| `./.venv` | FastAPI アプリ（`main.py`・API・ジョブ・Gradio・mock backend） | FastAPI / Pydantic / Pillow / ffmpeg 呼び出し。**torch は入れない** |
| `./.venv-engine` | 実エンジン worker（`engine/worker.py`） | **torch 2.9.1+cu128** + LTX 推論スタック（`ltx_core`/`ltx_pipelines` @ `00dc53d` + `gguf`） |

両者は別インタプリタで、同一インタプリタに共存させない。アプリ側は torch も LTX も import せず、実生成は `./.venv-engine` の python で `python -m engine.worker` を subprocess として起動する（詳細は §4）。

### 2.3 環境分離ポリシー（最重要制約）

**このプロジェクトは開発マシンのシステム Python 環境を一切汚さない・壊さない。** これは本プロジェクトの最優先制約であり、維持し続ける。

- **Python 本体を含め**、必要なものはすべてプロジェクトディレクトリ配下（`.venv/`, `.venv-engine/`, `.python/`）に閉じ込める。
- グローバル/システムの `pip install` は**禁止**。必ず `uv` + プロジェクトローカル venv。
- システム環境変数（永続的な `PATH`・`PYTHONPATH` 等）の恒久的な書き換えを禁止する。
- 後片付けは**このディレクトリ（`.venv` / `.venv-engine` / `.python` を含む）を削除するだけ**で完全に元に戻せる状態を維持する。

### 2.4 プロセススコープの環境変数

環境変数はすべて**そのプロセス内のみ**で設定し、永続化しない（`run.ps1` / `scripts/setup.ps1` / `scripts/install_ltx.ps1` が設定する）。`setup.bat` / `run.bat` は `.ps1` を呼ぶだけのラッパーで、環境変数の設定は行わない。

あわせて `scripts/setup.ps1` と `run.ps1` は、`tools/uv` と `tools/ffmpeg/bin` を**そのプロセスの `PATH` の先頭**に足す（システムの `PATH` は書き換えない）。これにより `services/video_io.py` の `shutil.which` がプロジェクト内の `ffmpeg` / `ffprobe` を解決する（Python 側のコードは不変）。

| 変数 | 用途 |
|------|------|
| `UV_PYTHON_INSTALL_DIR = <root>\.python` | uv 管理の Python 本体をプロジェクト内に固定 |
| `PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True` | CUDA メモリ断片化による OOM 低減（未設定時は起動ログに警告） |
| `UV_CACHE_DIR = <root>\.uv_cache`（install 時） | uv ダウンロードキャッシュをプロジェクト内へ |
| `HF_HOME = <root>\hf_home`（install 時） | HuggingFace キャッシュをプロジェクト内へ |

### 2.5 インストール導線

セットアップの実体は `scripts/install_ltx.ps1`（冪等クリーンインストーラ）が担う。両 venv 構築 → 現行 ~30GB モデルセットのダウンロード → PASS/MISSING 検証表 → `models/INSTALLED_PATHS.txt` 再生成、を冪等（再実行安全）に行う。手順とモデル内訳は **§5** / `README.md` §1 を参照（§16 は受け入れテスト）。

**モデルの取得元は 3 リポジトリ（2026-08-03 更新）**: `Rootport/Nz-LTX23-weights`（LTX 本体 5 点＋IC-LoRA 3 点＋VDA 深度前処理器 2 点＝計 10 ファイル・25,219,475,913 B）／`Rootport/Nz-Gemma3-12B`（11 ファイル・7,339,810,357 B）／`Rootport/Nz-DWPose`（前処理器 2 点・352,756,773 B）。5 回のダウンロード呼び出しで合計 23 ファイル・32,912,043,043 B（約 30.7GiB）。3 つとも Public かつ非 Gated で、HuggingFace のアカウント・トークンは一切不要。各リポジトリの内部構造は本プロジェクトの `models/` 配下と 1 対 1 で一致させてあるため、いずれも `models/` へそのまま展開され、後処理（平坦化・リネーム）は発生しない。`Rootport/Nz-Sulphur2`（自家変換 GGUF）は同じアカウントにあるがインストーラの取得対象ではない。

エンドユーザーの入口は **`setup.bat`（ダブルクリック）→ `scripts/setup.ps1` → `install_ltx.ps1`** である。`setup.ps1` は前提ツール（`uv` / `ffmpeg`+`ffprobe`）を `tools/` 配下へ取り込み、`config.yaml` が無ければ `config.yaml.example` から複製し、`logs/setup_<日時>.log` へ記録を残したうえで `install_ltx.ps1` を `&` で呼ぶ（ドットソースは禁止＝`install_ltx.ps1` の `exit` が呼び出し元ごと落とすため）。想定利用者が PowerShell を自分で開けないことを前提とした導線であり、`.bat` は純 ASCII・CRLF・末尾 `pause`、日本語のメッセージはすべて `.ps1` 側に置く。

`setup.ps1` は本処理の前に**事前チェックを 3 種**（インストール先ドライブの空き容量／ページファイルの設定／NVIDIA GPU の有無）行うが、**いずれも警告を出すだけでセットアップは止めない**（環境の自動判定で利用者の作業を止めない方針）。起動側にも同種のガードがあり、`run.ps1` は待ち受けポートが既に使われていれば「すでに起動しています」と案内して `exit 0` する＝**二重起動ガード**（詳細は §3.3）。

**冪等の粒度（2026-07-26 更新）**: 「既存物は無条件 SKIP」ではない。

- **アプリ venv `./.venv`**: `uv sync` は毎回走る（冪等だが SKIP はしない）。
- **エンジン venv `./.venv-engine`**: **ピン留めされた依存が前回適用時から変わっていないときだけ SKIP** する。判定は `engine/venv-engine.freeze.txt` の本文と、`install_ltx.ps1` 内にハードコードされた 3 つの git リビジョン（`diffusers` / `ltx-core` / `ltx-pipelines`）を連結した SHA-256 で行い、適用済みの値を `.venv-engine/.nz-engine-state` に記録する（`.gitignore` 配下＝追跡外・venv と運命を共にする）。このファイルは**完了マーカーを兼ねる**ため、記録が無い場合は「再適用する」側へ倒す（中断で半端に残った venv と、この仕組み以前に作られた venv を救うため）。`-ResolveLatest` を通った場合はマーカーを書かない（凍結スタックではないため）。この再同期があってはじめて「`git pull` → `setup.bat` 再実行」という更新手順が成立する。
- **モデル群**: サイズによる冪等ガードで SKIP するが、判定は **ディレクトリごとに独立**である。`Invoke-ModelDownload` の `-Check` はそのリポジトリが展開する各ディレクトリについて `@{ Dir = "<パス>"; Min = <バイト数> }` を受け取り、**全エントリがそれぞれ自分の `Min` を満たしたときにだけ SKIP** する（1 つでも下回れば、そのリポジトリのダウンロードが走る）。判定対象を展開先（3 件とも `models`）から分離しているのはこの `Dir` である。**複数ディレクトリの合計を単一のしきい値と比べてはならない**（旧 `-CheckDir` ＋ `MinBytes` 方式・2026-07-26 廃止）。合計方式では大きなファイル 1 つが丸ごと欠けた兄弟ディレクトリを覆い隠す。実際 Gemma で発現し、7.3GB の GGUF だけで合計しきい値を超えるためトークナイザ dir が全欠落でも SKIP → 検証表 `gemma_root` MISSING → exit 1 → 再実行しても変わらない、という復旧不能のデッドロックになった。
- **`Min` の取り方**: 真の正当性ゲートは `install_ltx.ps1` step 6 のファイル単位検証表（**16 項目**）である。したがって各 `Min` は「**そのディレクトリ**の合計 − **そのディレクトリ内で step 6 の検証表が個別にゲートしている最小ファイル**」より大きく、かつそのディレクトリの想定合計より少し小さく置く。これで「検証表が MISSING にできる欠落は必ずガードも割る」という対応が成立し、上記デッドロックが構造的に起きなくなる。検証表が個別に見ていないファイル（例: `gemma_root` はディレクトリ全体を 20MB で 1 行として見るだけ）は、欠けても MISSING にならないため捕捉できる必要はない（捕捉しようとすると 35 バイト幅のしきい値になり破綻する）。なお `models/ltx-2.3-gguf/` はユーザーが自家変換 GGUF を置くためサイズ判定が原理的に緩くなるが、これはサイズガードでは解けず検証表が受け持つ。

---

## §3 ネットワークと起動

### 3.1 バインドと公開範囲

既定では **`127.0.0.1:18620`（localhost 限定）**にバインドする（`config.server.host` / `port`）。外部からは到達できない。

`--listen` を付けると **`0.0.0.0`（家庭内 LAN 公開）**にバインドする。この場合サーバーは LAN 上の他機から到達可能になる。**インターネットへの公開は非サポート**（設計上、単一ユーザー・家庭内利用を想定）。

### 3.2 起動フラグ（`main.py` 準拠）

`config.yaml` の値に対し、CLI フラグが上書きする。

| フラグ | 説明 |
|--------|------|
| `--listen` | `0.0.0.0` にバインド（家庭内 LAN 公開。インターネット公開は非対応） |
| `--port <n>` | ポート変更（デフォルト **18620**） |
| `--api-key <key>` | `Authorization: Bearer <key>` を要求 |
| `--allow-all-cors` | CORS 全許可（既定は localhost オリジンのみ許可＝`^http://(127\.0\.0\.1\|localhost)(:\d+)?$`） |
| `--config <path>` | `config.yaml` のパス指定 |
| `--te-offload` / `--no-te-offload` | Gemma text-encode の逐次 per-layer CPU オフロード（既定 ON・§9/§11） |
| `--dit-cpu-load` / `--no-dit-cpu-load` | DiT(transformer) を CPU 構築しブロックを GPU へストリーム（既定 ON。無効化するとロード時 ~16.9GB の GPU スパイクが復活） |

### 3.3 起動方法

エンドユーザー向けの標準手順は **`run.bat`（ダブルクリック）**である。`run.bat` は実行ポリシーを回避してリポジトリ直下の `run.ps1` を呼ぶだけの薄いラッパーで、引数はそのまま透過する（`"%~dp0run.ps1" %*`）。**`run.ps1` はリポジトリ直下に置くこと**（`$PSScriptRoot` を基準に `.python` / `.venv` / `main.py` を解決するため、`scripts/` へ移すと全て 1 階層ずれる）。開発時は `run.ps1` を直接実行してよい。

```powershell
./run.ps1                 # localhost 限定・port 18620
./run.ps1 --listen        # 0.0.0.0（家庭内 LAN）
./run.ps1 --port 19000
```

`run.ps1` は環境変数設定 → `tools/` を `PATH` 先頭へ → `config.yaml` 不在時の `.example` からの複製 → アプリ venv で `main.py` 起動、を行う。**依存の再同期（`uv sync` 等）は行わない**（起動を軽く保つための設計判断。更新は「`git pull` → `setup.bat` 再実行」＝§2.5）。`.venv` が無い場合は `setup.bat` を促して `exit 1` する（`throw` は使わない＝`$ErrorActionPreference = "Stop"` 下では案内ブロックへ到達しないため）。ポートが既に使われている場合は「すでに起動しています」と案内して `exit 0` する（`setup.bat` への誤誘導を避けるため）。**利用者に見せる URL の正本は `main.py` の起動バナー**であり（CLI フラグ適用後のポートを知っているのはそこだけ）、`run.ps1` は URL を一切表示しない。

`run.ps1` は**アプリ**（`./.venv` の `main.py`）だけを起動する。real backend が選ばれると、アプリが `./.venv-engine\Scripts\python.exe -m engine.worker` を subprocess として自動 spawn する（worker の手動起動は不要）。起動後のエンドポイント:

- UI: `http://127.0.0.1:18620/ui`
- API docs (Swagger): `http://127.0.0.1:18620/docs`
- Status: `http://127.0.0.1:18620/api/v1/status`

---

## §4 アーキテクチャ

本バックエンドは **2プロセス・2venv 構成**である。凍結された API 層（`./.venv`, torch 無し）と、実 LTX 推論を担うエンジン worker（`./.venv-engine`, torch 2.9.1+cu128）を分離し、両者を小さな JSON-lines プロトコルで結ぶ。この分離により「API・ジョブ管理・Gradio・pytest は torch 無しでも常に疎通し、実生成は隔離された別インタプリタで動く」という契約が成立する。

### 4.1 2プロセス図

```text
[Gradio /ui]──HTTP──┐
                     ▼
┌────────────────────────────────────────────────┐
│  FastAPI アプリ  (./.venv, torch 無し)           │
│   api/       (router / models / status / ...)   │
│   services/  (job_store / upload_store /        │
│               pipeline_manager / video_io /     │
│               low_vram / gpu_info)              │
│   services/ltx_runner.py  ── 唯一の LTX 接点     │
│        ├─ _MockBackend  (合成クリップ・GPU不要)  │
│        └─ _RealBackend  ── subprocess.Popen ──┐ │
└───────────────────────────────────────────────┼─┘
                                                 │  JSON-lines（@@LTX@@ フレーム）
                                                 ▼  stdin / stdout（mp4 はパイプを通らない）
                    ┌──────────────────────────────────────────────┐
                    │ engine worker (./.venv-engine, torch 2.9.1+cu128)│
                    │   python -m engine.worker                       │
                    │   engine/pipeline/fast_video_pipeline.py        │
                    │   engine/gguf/        (GGUF dequant / loader)   │
                    │   engine/gemma/       (GGUF Gemma + 層オフロード)│
                    │   engine/transformer/ (block-swap / dit-cpu)    │
                    │   → output.mp4 を共有 outputs/ dir へ直接書く    │
                    └──────────────────────────────────────────────┘
```

| venv | 役割 | 主要依存 |
|------|------|----------|
| `./.venv` | FastAPI アプリ（`main.py` / API / ジョブ / Gradio / mock backend） | fastapi / uvicorn / gradio / pydantic / pillow。**torch は入れない** |
| `./.venv-engine` | 実エンジン worker（`engine/worker.py`） | **torch 2.9.1+cu128** + LTX 推論スタック（`ltx_core` / `ltx_pipelines` @rev `00dc53d` + `gguf`） |

アプリ（`./.venv`）は real backend を選んでも torch も `ltx_*` も import しない。実生成は `./.venv-engine` の python で `python -m engine.worker` を **subprocess** として常駐起動し、プロトコルで駆動する。

### 4.2 プロセス間通信（JSON-lines / `@@LTX@@` プロトコル）

`services/ltx_runner.py` の `_RealBackend` が、`subprocess.Popen([engine_python, "-u", "-m", "engine.worker"], cwd=<project_root>, env["PYTHONPATH"]=<project_root>)` で worker を **1度だけ**起動する。以降は stdin にリクエスト JSON（1行1オブジェクト）を書き、stdout から `@@LTX@@` プレフィクス付きの JSON イベント行を読む（プレフィクスの無い行 = ライブラリ/tqdm のノイズは読み捨てる）。worker はモデルを**1度だけ**構築してジョブを使い回し、**mp4 は worker が `output_path` へ直接ディスク書き込み**する。パイプを渡るのは小さな制御 JSON のみである。

**op（親 → worker, `engine/worker.py` のプロトコル）:**

| op | 主なフィールド | 意味 |
|----|----------------|------|
| `load` | `checkpoint_path`, `gemma_root`, `upsampler_path`, `gguf_transformer_path`, `gguf_gemma_path`, `gguf_per_layer_quant`, `block_swap_blocks_on_gpu`, `vae_spatial_tile_size`, `vae_temporal_tile_size`, `component_video_vae_path`, `component_audio_vae_path`, `component_text_projection_path` | パイプラインを1度だけ構築 |
| `generate` | `prompt`, `seed`, `height`, `width`, `num_frames`, `frame_rate`, `num_steps`, `images:[{path,frame_idx,strength}]`, `output_path` | 1本生成し `output_path` へ mp4 を書く |
| `shutdown` | （なし） | best-effort 解放 → `exit 0` |

**event（worker → 親, `@@LTX@@<compact-json>`）:**

| event | フィールド | 意味 |
|-------|-----------|------|
| `ready` | （なし） | `load` 完了。パイプライン構築成功 |
| `done` | `seed_used`, `peak_vram_mb` | 生成完了。`peak_vram_mb` は `torch.max_memory_allocated` 換算 MB |
| `error` | `detail`（repr + traceback 末尾 ~2000 字） | 失敗。`load` 時は worker が `exit 1`、`generate` 時は serving 継続 |

補足:
- **seed は親（アプリ側）で解決**する（`request.seed >= 0` ならそのまま、`-1` なら乱数）。`done.seed_used` は worker から返るが決定性の基準は親が握る。
- stderr は**パイプでなくログファイル** `logs/ltx_worker.log` へ流す（stderr をパイプすると、worker が大量の tqdm/log を吐く間に親が stdout でブロックしてデッドロックしうるため）。`GENERATED_OK peak_vram_mb=` 等の診断行はこのログに出る。
- `load` タイムアウトは 600 秒。タイムアウトすると watchdog スレッドが worker を kill し、ブロック中の readline を返させる。
- `crop_output` 指定時は、worker にフルサイズ mp4 を一時ファイル（`_full.mp4`）へ書かせ、アプリ側 ffmpeg ヘルパ（`services/video_io.crop_mp4`）が中央クロップして `output.mp4` を作る（→ §10 も参照）。

### 4.3 backend 選択（auto / mock / real）

backend は `config.model.backend`（`auto` / `mock` / `real`, 既定 `auto`）で決まる（`_select_backend`）。

- `mock` → 常に MockBackend。
- `real` → 常に RealBackend（後述の材料が揃わなければ `RuntimeError`）。
- `auto`（既定）→ `_real_available()` が真なら real、偽なら mock。

`_real_available()` は **torch / `ltx_*` を一切 import せずに**、real GGUF + component-file 経路が実際に開くファイル群の存在だけを確認する（失敗は握り潰して `False` → auto は mock へフォールバック、かつ torch 無しの `.venv` で `import services.ltx_runner` が安全に保たれる）。確認する材料:

| 材料 | 対応 config フィールド |
|------|------------------------|
| エンジン python | `model.engine_python` |
| worker スクリプト | `model.engine_dir`/`worker.py` の存在 |
| Gemma tokenizer dir | `model.gemma_root` |
| spatial upsampler | `model.spatial_upsampler_path` |
| GGUF transformer | `model.gguf_transformer_path` |
| GGUF Gemma | `model.gguf_gemma_path` |
| component VAE / audio / text-projection | `model.component_video_vae_path` / `component_audio_vae_path` / `component_text_projection_path` |

`checkpoint_path`（43GB モノリス。本書中の「46GB モノリス」と**同一ファイル**＝§5.2 の表記注記）は2026-07-28に`config.model`から削除済みで、そもそも対応する config フィールドが無い。`services/ltx_runner.py`が worker payload へ渡す値は直値の`""`にハードコードされており（`DistilledPipeline`構築のシグネチャを満たすためだけの存在）、GGUF + component 経路では一切開かれないため、ここでは**あえてゲートしない**（§5 参照）。逆に `gemma_root` は tokenizer/processor の module_ops をこの dir から読むため load-bearing で、欠けていればアプリ層で fail-fast させる。

**MockBackend の用途**: GPU / モデルウェイトの無い環境（開発・CI・pytest）向けの合成クリップ生成。`tests/conftest.py` が `model.backend="mock"` を強制する。API・スキーマ・出力構造（`outputs/{job_id}/output.mp4` + `metadata.json`）は real と同一で、`GenerationOutcome.backend` の値だけが異なる（mock=`"mock"`, real=`"ltx-distilled"`）。

### 4.4 ディレクトリ構成

```text
Nz-LTX23-backend/
├─ setup.bat / run.bat     エンドユーザー向け入口（純 ASCII・CRLF・末尾 pause）
├─ run.ps1                 起動本体（**直下固定**・$PSScriptRoot 依存）
├─ AviUtl2-Plugin/         NzLTX23.aux2（フロントエンド配布物・**git 追跡**）
├─ main.py                 アプリ起動（./.venv, FastAPI）
├─ gradio_ui/              検証用 /ui（ui.py / handlers.py / presets.py / i18n.py ほか）
├─ mcp_server/             MCPサーバー（§12b・./.venv で動くクライアント層）
├─ chain_math.py           チェーンのフレーム／音声 latent 計算（アプリとエンジンで共用）
├─ config.py               設定ローダ（相対パスは PROJECT_ROOT 基準で絶対化）
├─ config.yaml             実設定（**git 追跡外**・config.yaml.example から複製される）
├─ config.yaml.example     配布されるひな型（config.yaml と同期させること）
├─ api/                    router / models(スキーマ) / status / generate ...
├─ services/               job_store / upload_store / pipeline_manager /
│                          video_io / low_vram / gpu_info /
│                          ltx_runner.py（唯一の LTX 接点・mock/real backend）
├─ engine/                 first-party 実エンジン（./.venv-engine で実行）
│   ├─ worker.py           常駐 worker（python -m engine.worker）
│   ├─ pipeline/           fast_video_pipeline.py（低VRAM機構の配線点）
│   ├─ gguf/               GGUF dequant / loader サービス
│   ├─ gemma/              GGUF Gemma + 逐次層オフロード + text-only configurator
│   ├─ transformer/        block_swap / dit_cpu_load / sage_attention / nag / vsf サービス
│   ├─ preprocess/         参照動画の制御信号を起こす前処理（canny / dwpose / depth）
│   ├─ VENDOR_NOTICE.md    provenance
│   └─ venv-engine.freeze.txt / engine-venv-pyproject.toml（venv 再現）
├─ scripts/                setup.ps1（setup.bat の本体）/ install_ltx.ps1 /
│                          build_xformers.ps1 ...
├─ tools/                  scripts/setup.ps1 が取り込む前提ツール（uv / ffmpeg・**git 追跡外**）
├─ tests/                  pytest（mock 強制・torch 無し）
├─ models/                 GGUF/component 群＋IC-LoRA＋前処理器（取得 ~30GB, §5）
├─ outputs/                outputs/{job_id}/output.mp4 + metadata.json
└─ vendor/LTX-2            上流 LTX-2 クローン（reference only・非実行）
```

`engine/` は **first-party**（project root 直下・git 追跡）。旧・同梱フォーク `vendor/LTX-Desktop-LOW-VRAM` は de-fork リファクタ（§9・VERIFICATION_LOG §13）で完全削除済み。`vendor/LTX-2` は上流参照用に温存する（実行経路では読まれない）。

### 4.5 技術スタック

| 層 | 主なパッケージ / バージョン | 出所 |
|----|------------------------------|------|
| アプリ（`./.venv`） | fastapi / uvicorn / gradio / pydantic / pillow / **`mcp`（`>=1.28,<2`。2026-07-28追加・MCPサーバー用main依存＝§12b）**（+ ffmpeg 呼び出し）。**torch 無し** | `pyproject.toml` / `requirements.txt` |
| エンジン（`./.venv-engine`） | **torch 2.9.1+cu128**（+ torchaudio 0.24.1+cu128 / torchvision）、`ltx-core` / `ltx-pipelines` @git rev `00dc53d`、`diffusers` @git rev（`engine/venv-engine.freeze.txt` ヘッダが正）、`gguf`、`transformers`、**`sageattention` 2.2.0＋`triton-windows` 3.5.1.post24（2026-07-31追加・Acceleration 機能用＝§5.4）** | `engine/venv-engine.freeze.txt`（完全スナップショット）/ `engine/engine-venv-pyproject.toml`（`[tool.uv.sources]` に cu128 index と git rev） |

`ltx-core` / `ltx-pipelines` / `diffusers` は **git direct-url インストール**（PyPI ではない）。Windows の PyPI 版 torch は CPU 専用のため、cu128 インデックスから `torch/torchaudio/torchvision == *+cu128` を明示インストールする（`Docs/note.md`）。provenance と再現手順の一次情報は `engine/VENDOR_NOTICE.md`。

### 4.6 設計 pivot の記録

当初の設計は **公式 `ltx_pipelines` の `DistilledPipeline` + block streaming（`--offload cpu`）+ `fp8-cast`** を 16GB 実現の本命としていた。しかし本機（RTX 4070 Ti SUPER 16GB / Windows）では、公式 safetensors ローダが 22B チェックポイントのバルクロード中に **native crash（traceback 無し）** するため不採用となった。

そのため現行は **first-party `engine/` パッケージによる GGUF Q4_K_M + component-file 経路**へ pivot している:
- transformer は GGUF Q4_K_M で VRAM に圧縮常駐（block-swap でストリーム）。
- Gemma text encoder も GGUF Q4_K_M（逐次層オフロード）。
- VAE/audio/text-projection は 46GB モノリスでなく小単体 component ファイルから読む。

判断根拠は「16GB / Windows で公式ローダが落ちる」という実機事実に尽きる。詳細な経緯・A/B 実測・SHA256 バイト一致検証は `Docs/NEXT_SESSION_HANDOFF.md` と `Docs/VERIFICATION_LOG.md`、engine の由来は `engine/VENDOR_NOTICE.md` を一次情報とする。なお `Docs/note.md` は旧・公式ローダ前提のノートだが、「torch 2.9.1+cu128 / attention は SDPA（Ada では FlashAttention-2）で十分 / xformers は任意 / 16GB の律速は重み転送(PCIe)」という結論部分は現行でも有効である（**2026-07-31 補足**: この「SDPA で十分」は既定の話として引き続き正しいが、2026-07-31 に SageAttention をジョブ単位で選べるようにした結果、attention 側にも 720p で 1.17 倍・二段目単体で 1.56 倍という実測の伸びしろがあることが判明した。詳細は §5.4）。

---

## §5 モデル構成（実行 ~28GB・取得 ~30GB）

### 5.1 実行に本当に要る構成

本番経路（GGUF + component-file）が実際にロードするのは以下の要素で、合計 **~28GB**（実測 28.15GiB）である。相対パスは `config.model` が保持し、PROJECT_ROOT 基準で `config._abs` が絶対化する。厳密なファイル別サイズは `models/INSTALLED_PATHS.txt` を正とする。同ファイルが列挙するのは **8 行**で、内訳は本表の 5 要素を展開したもの（component ファイルが video VAE / audio VAE / text-projection の 3 行に分かれる）**7 行**＋ `engine_python`（`./.venv-engine/Scripts/python.exe`）である（2026-07-28、`checkpoint_path`の行は`config.yaml`からの削除に伴い消えた・§5.5）。**§5.1b の IC-LoRA / 前処理器は含まない。**

| 要素 | 既定パス | 概算 | 役割 |
|------|----------|------|------|
| GGUF transformer (Q4_K_M) | `models/ltx-2.3-gguf/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf` | ~17GB | 本番 DiT トランスフォーマー（VRAM 圧縮常駐） |
| GGUF Gemma (Q4_K_M) | `models/gemma-3-12b-it-gguf/gemma-3-12b-it-Q4_K_M.gguf` | ~7.3GB | text encoder（GPU 推論・逐次層オフロード） |
| component VAE / audio / text-projection | `models/ltx-2.3-components/{vae,text_encoders}/*.safetensors` | ~3.9GB | 46GB モノリスを置換する小単体ファイル |
| spatial upsampler | `models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | ~0.95GB | 二段生成の x2 アップサンプラ |
| Gemma tokenizer dir（`gemma_root`） | `models/gemma-3-12b-it-tokenizer/` | ~40MB | tokenizer/processor のみ（`tokenizer.model` + `preprocessor_config.json`）。**重みは含まない** |

このフットプリントは ComfyUI の GGUF 16GB レシピと同等である（VERIFICATION_LOG §10.4 / §14.1）。

### 5.1b インストーラが追加で取得するもの（IC-LoRA / 前処理器）

上表は `_real_available()` が実在を確認する「生成の中核」であり、これが揃えば real backend は成立する。一方 `scripts/install_ltx.ps1` は、2026-07-26 以降これに加えて次の 4 種類も取得する（**以前は上流から手で置く前提の、インストーラ管理外のファイルだった**）。下 2 行は 2026-08-03 の depth-control / deblur 追加分である（正本 `Docs/ICLORA_DEPTH_DEBLUR_WORKORDER.md`）。合わせて取得総量は **~33GB（30.7GiB）** になる。

| 要素 | 既定パス | 概算 | 取得元 | 役割 |
|------|----------|------|--------|------|
| IC-LoRA 2 点 | `models/ltx-2.3-ic-lora/pixel-spatial-upscaler/…-x2-0.9.safetensors`／`models/ltx-2.3-ic-lora/union-control/…-union-control-ref0.5.safetensors` | 1.22GiB | `Rootport/Nz-LTX23-weights` | `config.yaml` `model.ic_loras` の 4 エントリの実体。union-control の 1 ファイルを `canny-control` / `pose-control` / `depth-control` の 3 名で公開している（§11 / IC-LoRA Phase C / `Docs/ICLORA_DEPTH_DEBLUR_WORKORDER.md`） |
| DWPose 前処理器 2 点 | `models/preprocessors/yolox_l.torchscript.pt`／`models/preprocessors/dw-ll_ucoco_384_bs5.torchscript.pt` | 0.33GiB | `Rootport/Nz-DWPose` | `pose-control` の前処理（`engine/preprocess/dwpose.py` が自ファイル位置からの絶対パスで両方を読むため、配置は変更不可） |
| IC-LoRA Deblur 1 点（2026-08-03 追加） | `models/ltx-2.3-ic-lora-deblur/ltx-2.3-22b-ic-lora-deblur-0.9.safetensors` | 906,071,437 B | `Rootport/Nz-LTX23-weights` | `deblur` エントリの実体。**前処理不要**（ぼけた参照動画をそのまま渡す）なので `config.yaml` では**文字列形式**で登録する。メタデータの `reference_downscale_factor="1"`＝参照を縮小せずに（縮小係数1で）条件付けに使うため、既存アダプタ（係数 2）より stage-1 の VRAM を食う |
| VDA 深度前処理器 2 点（2026-08-03 追加） | `models/preprocessors-vda/video_depth_anything_vits.pth`／同 `LICENSE` | 116,452,112 B | `Rootport/Nz-LTX23-weights` | `depth-control` の前処理（Video-Depth-Anything Small。`engine/preprocess/depth.py` が自ファイル位置からの絶対パスで読むため配置は変更不可。dwpose.py と同じ作法）。**`LICENSE` は Apache-2.0 の全文**で、重みリポジトリ内の他ファイル（LTX-2 Community Licence）とはライセンスが異なるため `.pth` と必ず一緒に運ぶ |

**これらは「無くても mock に落ちない」ため、検証を省くと壊れ方が分かりにくい**: `_real_available()` は見ておらず、`config.yaml` はすべての `ic_loras` エントリを無条件に登録し、`gradio_ui/adapters.py` は登録が空でも同じ名前を静的フォールバックで並べる。したがって欠けていても UI にはアダプタ名が出て、選んだ瞬間に 404 になる。この失敗を前倒しで名指しするため、install_ltx.ps1 step 6 の検証表はこの 6 ファイルを含む **16 項目**になっている（x4 アップスケーラ版は未登録＝リポジトリにも置かない）。

> **新規の重みを既存ディレクトリの「子」ではなく「兄弟」に置いている理由**: インストーラのスキップ判定は Check ディレクトリの**再帰的サイズ合計**である。Deblur の 906MB を既存の `models/ltx-2.3-ic-lora/` の**中**に置くと、union-control ファイル（654MB）を失っているマシンでも合計が既存の Min 値を上回ってしまい、ダウンロードがスキップされて「union-control MISSING」が永久に解消しなくなる（Gemma tokenizer で起きた事故の逆方向の再発）。`models/preprocessors-vda` を `models/preprocessors` の兄弟にしているのも同じ理由で、子（`models/preprocessors/vda/`）にしていたら VDA の 116MB が DWPose 側の判定を水増しして、欠けた `dw-ll_ucoco`（135MB）を覆い隠していた。詳細は `Docs/VERIFICATION_LOG.md` §49.6。

### 5.2 削除済みの重量物

de-fork（§13）と QAT 回収（§13）のリファクタで、実行に不要な以下を**物理削除**した:

- **43GB モノリス** `ltx-2.3-22b-distilled-1.1.safetensors`。`config.model.checkpoint_path` フィールドは2026-07-28に削除済み（死んだ設定と確認済み・`Docs/VERIFICATION_LOG.md` §40）。worker payload には引き続き載る（`services/ltx_runner.py`が`DistilledPipeline`のシグネチャを満たすため直値`""`をハードコード）が、GGUF + component 経路では一切開かれず、存在チェックすら課さない（rename test で実証・VERIFICATION_LOG §13.3）。
- **22.7GB の QAT Gemma dir** `models/gemma-3-12b-it-qat/`。Gemma を text-only 化（下記 5.3）したため wheel が build 時に重みシャードを glob する必要が無くなり、`gemma_root` は ~40MB の tokenizer-only dir で足りる。

これにより `models/` は 50.9GB → ~28.15GB（VERIFICATION_LOG §14.1）。

> **「43GB モノリス」と「46GB モノリス」の表記について（本書全体に適用）**: 本書はこのファイルを箇所により「43GB モノリス」とも「46GB モノリス」とも書いている（§4.3・§4.6・§5.1・§5.2・§5.5・§7.6・§9.1・§9.2・§11.2・付録B）。**両方とも同一の 1 ファイルを指し、どちらの表記も誤りではない。** 実測 **46,139,885,414 バイト**で、10 進 GB では 46.1GB、2 進 GiB では 42.97GiB（≒43GiB）になるだけである（同じ混在が `config.py` にもある）。**この表記についての正本は本節である。**

### 5.3 Gemma の text-only 化

LTX の text encoder（`GemmaTextEncoder.precompute`）は `language_model` の hidden_states しか使わないのに、wheel は vision_tower + multi_modal_projector 込みのフル・マルチモーダル `Gemma3ForConditionalGeneration` を構築していた。Q4_K_M GGUF は言語モデル重みしか持たないため、この構成では vision が meta デバイス上の死蔵重みとなり、`model.device`（= 最初の param の device）が meta に落ちて `precompute` が meta 上に `input_ids` を作り **crash** する。

`engine/gemma/text_encoder_configurator.py` は wheel の configurator を差し替え、**text-only `Gemma3ForCausalLM`（vision 無し）** を構築する（wheel はフォークしない）。最初の param が言語 embedding になるため `model.device` が自然に解決し、hidden states は multimodal build と**バイト一致**する（VERIFICATION_LOG §14.3、T2V/最小I2V の SHA256 が full-QAT baseline と完全一致）。vision を落として失う機能は `enhance_i2v`（既定 OFF・Phase1 外）のみである。

### 5.4 attention バックエンド

既定は **PyTorch SDPA**（全アーキ共通）。Ada Lovelace + torch 2.9 では SDPA の実体は **FlashAttention-2** カーネルであり、16GB の律速は attention でなく重み転送（PCIe）であるため、これで十分である（`Docs/note.md` の結論サマリ）。**xformers は任意**の最適化で、Windows では自動導入されず、必要ならソースビルドする（`scripts/build_xformers.ps1`）。

**SageAttention によるジョブ単位の切替（2026-07-31追加）**: 上記の既定は変えないまま、**ジョブごとに attention の実装を選べる**ようにした（`GenerateRequest` / `GenerateChainRequest` の `attention_backend`＝`"sdpa"`（既定）/ `"sage"`。§6.2）。`"sage"` を選んだジョブでは、engine 側の `engine/transformer/sage_attention_service.py` が transformer の **48 ブロック × 6 種（`attn1`／`attn2`／`audio_attn1`／`audio_attn2`／`audio_to_video_attn`／`video_to_audio_attn`）＝ 288 モジュール**の `attention_function` を SageAttention 2.2.0 の実装へ差し替える。サーバー再起動・パイプライン再ロードは不要。

- **降格の規律**: sageattention が導入されていない環境で `"sage"` を指定しても **422 にはせず、警告1行を出して SDPA で完走する**。NAG（§6.2 の相互検証）のような fail-loud と規律をあえて変えているのは、本機能が生成結果の意味ではなく**速度**のオプションだからである。マスク付きの attention 呼び出し（IC-LoRA の `conditioning_attention_strength < 1.0` 経路）も SDPA へ自動フォールバックし、カーネル例外が出た場合はそのジョブ内で SDPA に固定する。
- **実効値の記録**: 実際に使われた方式は `metadata.json` トップレベルの `attention_used`（`"sdpa"` / `"sage"` / `"sage->sdpa"`）に残る（§6.6）。利用可否は `GET /status` の `acceleration.sage_available`（§6.5b）。
- **依存**: `sageattention` 2.2.0（woct0rdho 版 Windows wheel を直リンク pin）＋ `triton-windows==3.5.1.post24` を `engine/venv-engine.freeze.txt` に載せ、インストーラが標準で導入する。`sageattention` は `engine/engine-venv-pyproject.toml` の `dependencies` にも `[tool.uv.sources]` にも**載せない**——`-ResolveLatest` は未検証の新しい torch を解決する経路で、torch 2.9.1 固定 ABI の wheel と非互換になるため。**`-ResolveLatest` を使った環境では sage の動作は保証外**である。2026-07-28 の依存整理（`Docs/VERIFICATION_LOG.md` §40.1）で削除した旧世代 `sageattention` 1.0.6 とは別物で、今回は実消費者があるための再導入である（同 §43.3）。
- **実測と設計判断の正本**: `Docs/VERIFICATION_LOG.md` §43。利用者向け説明は `README.md`「生成の高速化（Acceleration）」節。

インストーラ `scripts/install_ltx.ps1` は「全アーキで SDPA、xformers/flash-attn は自動導入しない」方針である。かつては GPU アーキを `nvidia-smi` で自動判定し、ada/ampere/hopper については `wheels/` にプリビルド wheel があればそれを導入する分岐を持っていたが、**この自動導入は 2026-07-26 に廃止した**（全アーキ SDPA 固定である以上、アーキごとに導入物を変える理由が無く、判定の失敗・誤判定が事故の種になるだけであるため）。`scripts/build_xformers.ps1` は、xformers を自分でビルドしたいユーザーのための**手動ツール**として引き続き `scripts/` に残す（インストーラからは呼ばれない）。

### 5.5 reference-only パスの位置づけ

`ltx_repo_dir`（`vendor/LTX-2` 上流クローン）は config に残るが **reference-only** で、GGUF + component 経路では読まれない。`checkpoint_path`（43GB モノリスへの旧参照パス）は2026-07-28に`config.model`から削除済みで、現在は`services/ltx_runner.py`が worker payload へ直値の`""`をハードコードして渡すのみ（`DistilledPipeline` 構築のシグネチャを満たすためだけで存在チェック無し）。`fast_video_pipeline.py` の fail-fast アサートが「component/GGUF ソースが全て揃っていること」を build 前に要求するため、モノリスへサイレントにフォールバックすることは無い。

---

## §6 【凍結】API 契約

本章は Phase 1 の外部 API 契約を凍結・自己完結で定義する正本である。将来のフロントエンド（AviUtl2 / DaVinci Resolve 等）や後続 Phase が破壊されないよう、ここに記した識別子・パス・enum 値・JSON キー・バリデータは**そのまま実装（`api/models.py`, `api/*.py`, `services/*`）と一致する**。値・フィールド名は英語のままコードに合わせ、変更しない。

全エンドポイントは `/api/v1` プレフィックス配下にマウントされる（`main.py` の `app.include_router(api_router, prefix="/api/v1")`）。エラー時は共通エンベロープ（§6.8）で返る。

### 6.1 エンドポイント一覧

Phase 1 で**実在する**全エンドポイント。認証は `server.api_key`（= `--api-key`）が設定されている時だけ `Authorization: Bearer <api-key>` を要求する（`api/deps.py::require_auth`）。api_key 未設定なら「認証要」列のエンドポイントも認証なしで通る。

| メソッド | パス | 役割 | 認証 | 主なステータス |
|---------|------|------|:---:|---------------|
| GET | `/api/v1/status` | サーバー状態・GPU 情報・パイプライン状態・VRAM 最適化・キュー | 不要 | 200 |
| GET | `/api/v1/config` | 現在の実効設定（`AppConfig.model_dump()`）を返す | 不要 | 200 |
| POST | `/api/v1/pipeline/load` | パイプラインを明示ロード | 要 | 200 / 503(PIPELINE_LOAD_FAILED) |
| POST | `/api/v1/pipeline/unload` | パイプラインをアンロード | 要 | 200 / 409(JOB_BUSY) |
| POST | `/api/v1/upload/image` | 最小 I2V 用画像をアップロードし `image_id` を返す | 要 | 200 / 400(UPLOAD_INVALID_TYPE, UPLOAD_TOO_LARGE) |
| POST | `/api/v1/generate` | 生成ジョブを開始し job_id を返す | 要 | **202** / 404(IMAGE_NOT_FOUND) / 409(JOB_BUSY) / 422(VALIDATION_ERROR) |
| GET | `/api/v1/jobs` | メモリ上のジョブ一覧を返す | 不要 | 200 |
| GET | `/api/v1/jobs/{job_id}` | ジョブ状態を返す | 不要 | 200 / 404(JOB_NOT_FOUND) |
| GET | `/api/v1/jobs/{job_id}/video` | 完了済み動画（`video/mp4`）を返す | 不要 | 200 / 404(JOB_NOT_FOUND) / 409(VIDEO_NOT_READY) |
| POST | `/api/v1/jobs/{job_id}/join` | 完了済みV2Vジョブの継続動画（`output.mp4`）をアップロード元のソース動画へサーバー側で結合し `joined.mp4` を生成（ffmpegのみ・GPU不要・単一ジョブガードから独立） | 要 | 200 / 404(JOB_NOT_FOUND, SOURCE_VIDEO_NOT_FOUND) / 409(VIDEO_NOT_READY) / 422(VALIDATION_ERROR, JOB_NOT_JOINABLE) / 503(JOIN_FAILED) |
| GET | `/api/v1/jobs/{job_id}/joined` | 結合済み動画（`joined.mp4`）を返す | 不要 | 200 / 404(JOB_NOT_FOUND, JOINED_NOT_READY) |
| DELETE | `/api/v1/jobs/{job_id}` | 実行中はキャンセル要求、終了済みは結果削除 | 要 | 200 / 404(JOB_NOT_FOUND) |

補足（実装どおり）:
- `POST /generate` は成功時 **202 Accepted**（`status_code=202`）。レスポンス `status` は `queued`。
- 認証が必要なのは `require_auth` 依存を持つ 6 経路のみ: `pipeline/load`, `pipeline/unload`, `upload/image`, `generate`, `jobs/{job_id}/join`, `DELETE /jobs/{job_id}`。`status` / `config` / `jobs` 系 GET（`jobs/{job_id}/joined` を含む）は認証不要。
- api_key 設定時に Bearer 不一致/欠落 → `401 UNAUTHORIZED`。
- `POST /generate` は投入時にまず `conditioning_images` の各 `image_id` の実在を検証（`upload_store.path_for` が `IMAGE_NOT_FOUND`=404 を送出）、次に単一ジョブガードで 409。
- `POST /pipeline/unload` は実行中ジョブがあると `409 JOB_BUSY`（detail="cannot unload while a job is running"）。
- `POST /upload/video`（`api/uploads.py::upload_video`）も本表に未掲載の ADDITIVE エンドポイント（Phase B の IC-LoRA 参照動画／V2V 継続元アップロード）で、**2026-07-30 に任意のクエリ引数 `trim_start_sec` / `trim_duration_sec`（いずれも `float | None`、既定 `None`）を加算した**——アップロードした動画のうち `[trim_start_sec, trim_start_sec + trim_duration_sec)` の区間だけを残してサーバー側で切り出す（フロントエンドのタイムライン上でリボンが元動画の一部しか占めていないときに、その範囲だけを冒頭クリップにするための機能）。作法は `source_tail_seconds` と同じ「凍結表外エンドポイントへの追加専用拡張」で、**2引数とも未指定なら旧リクエストとバイト単位で同一**（`services/video_upload_store.py` の切り出し経路そのものが走らず、受信バイト列がそのまま保存される）。`Query()` に `ge=`／`le=` を意図的に付けておらず、**片方だけ指定・NaN／inf・負の開始・0以下の尺・ffmpeg 失敗はすべて 422 や 500 にせず「トリムせずそのまま保存」へ穏当に劣化する**（本引数の加算で新しいエラー応答が生まれないことを保証する設計）。レスポンス `UploadVideoResponse` には `trimmed`（bool, 既定 `False`, `2026-07-30追加`——2引数が指定され、かつ切り出しが実際に成功したときだけ `True`）を加算した。**この2引数は V2V 継続元だけでなく IC-LoRA 参照動画（`reference_video_id`）のアップロードにも同じクエリのまま使われる**（`2026-08-01`——フロントエンドが同じ判定関数で両方の経路にトリムを適用するようになったため。バックエンドは両者を区別せず、`POST /upload/video` は1本のままである）。実装・機械検証・実機ゲートの記録は `Docs/VERIFICATION_LOG.md` §42。
- `POST /jobs/{job_id}/join`・`GET /jobs/{job_id}/joined` は V2V（video-to-video 継続）専用の ADDITIVE エンドポイントで、V2V 継続機能そのものの実装時（§24）に新設され、**2026-07-21 に凍結の限定解除（オーナー承認・コミット `d22706e`）でリクエスト/レスポンスが拡張された**。リクエスト `JoinRequest` は `audio_smoothing`（bool, 既定 `true`＝クロスフェード）・`handle_crossfade_ms`（int, 既定 `300`, `0`〜`2000`）・`source_tail_seconds`（float, 既定 `5.0`, `2026-07-21追加`——結合前にソース動画の末尾 `N` 秒だけを残す tail-keep トリム。`0` はソースを全長のまま結合）。レスポンス `JoinResponse` は `job_id`・`joined_path`（結合後 mp4 のパス）・`join_mode`・`source_normalized`・`source_lufs`・`continuation_lufs_before`・`fade_ms_applied`・`handle_crossfade_ms_applied`・`handle_context_seconds`・`loudness_matched`・`trimmed_source_seconds`（float, `2026-07-21追加`——tail-keep で削られた秒数。挿入位置計算に使う）・`source_fps`（float \| null, `2026-07-21追加`——ソースの実測fps）を返す。ボディ省略（またはPOST時ボディ無し）は既定値でのスムーズ結合になる。

### 6.2 GenerateRequest 全文

`api/models.py::GenerateRequest`。フィールド・型・デフォルト・制約は下表のとおり（Pydantic `Field` の実値）。

| フィールド | 型 | デフォルト | 制約（Field） | 説明 |
|-----------|----|-----------|--------------|------|
| `prompt` | str | （必須） | `min_length=1, max_length=2000` | 生成プロンプト |
| `negative_prompt` | str | `""` | `max_length=2000`（**2026-07-28追加**。従来はno-opだったため上限が無かった） | ネガティブプロンプト |
| `width` | int | `512` | `ge=256, le=4096` | 生成幅。**64 の倍数必須**（下記バリデータ） |
| `height` | int | `320` | `ge=128, le=4096` | 生成高。**64 の倍数必須** |
| `crop_output` | CropOutput \| null | `null` | — | 最終 MP4 のクロップサイズ。null ならクロップなし |
| `num_frames` | int | `49` | `ge=9, le=481` | フレーム数。**8n+1 必須**。上限 481=20s@24fps |
| `frame_rate` | float | `24.0` | `ge=1.0, le=60.0` | フレームレート |
| `num_inference_steps` | int | `8` | `ge=1, le=100` | 推論ステップ。distilled では 8 固定 |
| `guidance_scale` | float | `1.0` | `ge=0.0, le=20.0` | CFG。distilled では 1.0 固定 |
| `seed` | int | `-1` | — | `-1` は乱数シード（親プロセスで解決） |
| `pipeline` | `Literal["distilled","two_stage_hq"]` | `"distilled"` | enum | パイプライン種別 |
| `conditioning_images` | list[ConditioningImage] | `[]` | — | 空=T2V、1 件=最小 I2V |
| `nag_enabled` | bool | `false` | — | **2026-07-28追加**。NAG（Normalized Attention Guidance＝CFGを使わずcross-attention出力レベルでネガティブプロンプトを効かせる非CFG手法）の有効化 |
| `nag_scale` | float | `11.0` | `ge=1.0, le=20.0` | **2026-07-28追加**。外挿の強さ |
| `nag_tau` | float | `2.5` | `ge=1.0, le=10.0` | **2026-07-28追加**。ノルム頭打ち上限 |
| `nag_alpha` | float | `0.25` | `ge=0.0, le=1.0` | **2026-07-28追加**。正出力とのブレンド比率 |
| `attention_backend` | `Literal["sdpa","sage"]` | `"sdpa"` | enum | **2026-07-31追加**。attention（注意機構）の実装選択。`"sage"` は SageAttention 2.2.0（§5.4）。**実装あり**。sageattention 未導入の環境では 422 にせず `"sdpa"` へ降格して完走する |
| `block_swap_prefetch` | bool | `true` | — | **2026-08-02追加**。block swap（transformer のブロックを CPU⇔GPU 間で出し入れする既定の省VRAM機構）の転送を、計算とは別の CUDA stream で先回りさせて隠す先読み機能。**実装あり**。`attention_backend` と違い転送方式のみを変えるため、同一シードなら off/on で出力がビット単位一致する。block swap 自体が無効な設定（`vram.block_swap=false` / `block_swap_blocks_on_gpu=0` / 全ブロック数以上）では黙って no-op になる。詳細は `Docs/VERIFICATION_LOG.md` §44 |
| `keep_resident` | bool | `false` | — | **2026-08-03追加**。モデルのCPU側「骨格」（GGUF から組み上げた state dict とモジュールツリー。DiT 約16.5GB＋Gemma 約8〜9GB）をジョブ間で保持して使い回し、2本目以降の前処理固定費を消す per-job フィールド（ジョブごとのリクエスト項目）。**実装あり**。GPU には何も常駐させないため VRAM プロファイルは不変で、同一シードなら off/on で出力がビット単位一致する（HIT 時の前処理は実測 5.32 秒。ベースラインは 68.6〜75.0 秒）。**メモリ 64GB 以上を推奨**（約20GB を常時占有する）。`gguf_per_layer_quant=false` との併用はジョブがエラーで停止し（bf16 融合経路の in-place な LoRA 融合がキャッシュを汚染するため）、`dit_cpu_load=false` / `block_swap_prefetch=false` との併用は警告のうえ自動 off になる。実際に効いたかは `metadata.json` の `keep_resident_used`（`"off"` / `"on"` / `"on->off"`）で確認する。詳細は `Docs/VERIFICATION_LOG.md` §48 |
| `fused_gguf_dequant_kernel` | bool | `true` | — | **2026-08-04追加**。GGUF の K量子化（Q4_K / Q5_K / Q6_K）の逆量子化を Triton の1カーネルへ融合する高速化。**実装あり**。`attention_backend` と違い展開の手順しか変えないため、同一シードなら off/on で出力がビット単位一致する（必須要件として検証済み）。Triton 不在・カーネル例外・型ごとの初回自己検証の不一致のいずれでも黙って従来実装へ降格し、生成は落とさない。実際に効いたかは `metadata.json` の `fused_gguf_dequant_kernel_used`（`"off"` / `"on"` / `"on->off"`）で確認する。実測は 768p/257f で約17.5%短縮。既定は実機ゲート G1〜G8 合格を条件に 2026-08-04 に `false` から反転した。詳細は `Docs/VERIFICATION_LOG.md` §51。**同日、受理のみで生成に影響しなかった旧モック `fused_gguf_dequant_gemm` は完全撤去した**（オーナー裁定。pydantic の `extra=ignore` により旧クライアントが送っても API は壊れない） |
| `vae_mode` | `Literal["default","prune_vaed"]` | `"default"` | enum | **2026-07-31追加（モックとして）・2026-08-05に実装**。映像VAEデコーダの選択。`"prune_vaed"` は PrunaVAED（Pruna AI 公開の枝刈り＋蒸留済みデコーダを当方で ltx-core native 形式へ変換した単体ファイル約690MB。`models/ltx-2.3-components/vae/prunavaed/PrunaVAED-decoder-bf16.safetensors`）。**実装あり**。`attention_backend="sage"` と同じく**出力の絵が変わる**（ビット一致しない）系の切替で、**既定は恒久 off**（ゲート合格後の既定反転は行わない。オーナー確定事項）。降格は環境ではなく**重みファイルの有無**で決まり、無ければ 422 にせず既定デコーダで完走する（存在確認はジョブ単位なので、サーバーを動かしたままファイルを着脱しても追随する）。実際に効いたかは `metadata.json` の `vae_mode_used`（`"off"` / `"on"` / `"on->off"`）で確認する。実測は 768p/257f で **−12.54秒（−10.5%）**・`peak_vram_reserved_mb` **−2,635MB**・PSNR 36.06dB／SSIM 0.9854（輝度）。詳細は `Docs/VERIFICATION_LOG.md` §52。**フィールド値 `"prune_vaed"` は上流名称の誤記（正しくは PrunaVAED）に由来するが、既存の外部コントラクトなので改名しない**（表示名のみ訂正した）。既存の `vram_optimization.vae_tiling`（VRAM 節約のタイル分割）とは**無関係** |

凍結制約（`model_validator(mode="after") validate_ltx_constraints`、順序どおり）:

1. `width % 64 != 0` → `ValueError("width must be a multiple of 64")`
2. `height % 64 != 0` → `ValueError("height must be a multiple of 64")`
3. `(num_frames - 1) % 8 != 0` → `ValueError("num_frames must be 8n+1")`
4. `crop_output` 指定時: `crop_output.width > width` / `crop_output.height > height` はそれぞれ ValueError（クロップは生成サイズ以下）
5. `pipeline == "distilled"` のとき: `num_inference_steps != 8` → ValueError、`guidance_scale != 1.0` → ValueError（Phase 1 の distilled は 8 step / CFG=1.0 固定）
6. `len(conditioning_images) > 1` → `ValueError("Phase 1 supports at most one conditioning image")`（最小 I2V は 1 枚まで）
7. `conditioning_images` が 1 件のとき `frame_idx != 0` → `ValueError("Phase 1 supports only frame_idx=0 for I2V")`
8. `nag_enabled` が true かつ `negative_prompt.strip()` が空 → `ValueError("nag_enabled requires a non-empty negative_prompt")`（**2026-07-28追加**）

> **NAGフィールドの補足（2026-07-28追加）**: 上記4フィールドは `GenerateRequest` に加えて `GenerateChainRequest`（`POST /generate/chain`。本書は§6ではPhase 1の単発生成のみを扱うため独立のスキーマ表は持たない）にも同一の名前・型・デフォルト・制約で存在し、`to_clip_request` 経由で `ClipGenerateRequest` へ転記される。詳細な設計判断（式の規約・非対称設計の根拠・実装箇所一覧・実機ゲート）は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §38 を正本とする。

> **Acceleration フィールドの補足（2026-07-31追加、2026-08-02・2026-08-04・2026-08-05追記）**: 上記4フィールド（`attention_backend` / `block_swap_prefetch` / `fused_gguf_dequant_kernel` / `vae_mode`）も NAG と同じく `GenerateChainRequest` に同一の名前・型・デフォルトで存在し、`to_clip_request` 経由で `ClipGenerateRequest` へ転記される。
>
> - **モックは1件も残っていない（2026-08-05）**。最後まで残っていた `vae_mode` が同日に実装され、Acceleration の全項目がエンジンまで到達するようになった。**2026-08-04 までは `fused_gguf_dequant_gemm` というもう1件のモックがあったが、同名の実装が入るわけではないため完全撤去し、実装のある `fused_gguf_dequant_kernel` が UI 上の位置だけを引き継いだ**（`Docs/VERIFICATION_LOG.md` §51.1）。モックの作法そのもの（受理はするが worker ペイロードと `GET /status` には出さず、`metadata.json` と `GET /jobs` の `request` エコーには現れる。`pipeline: "two_stage_hq"` が現存の前例）は、将来また実装前の枠を置くときのために本節の記述として残す。
> - **`vae_mode` の降格セマンティクス（2026-08-05）**: 他の3フィールドと同じ3値規約（`"off"` / `"on"` / `"on->off"`）だが、**降格の判定材料が環境ではなく「重みファイルが存在するか」である**点だけが違う。`vae_mode="prune_vaed"` を指定したジョブは、`config.model.component_video_vae_pruned_path` が指すファイルが無ければ **422 にもエラーにもならず、既定デコーダで完走して `vae_mode_used="on->off"` を記録する**。存在確認は**パイプライン構築時ではなくジョブ単位**で行うため、サーバーを動かしたままファイルを退避／復帰させると、ワーカーを再起動せずに `"on->off"` ↔ `"on"` が追随する（実測は `Docs/VERIFICATION_LOG.md` §52.7）。
> - **既定値のジョブの worker ペイロード**: いずれのフィールドも「**サーバー既定と異なる値のときだけキーを加算する**」方式である。既定が `"sdpa"` の `attention_backend` と既定 `false` の `keep_resident` は既定のジョブでキーが増えず、既定 `true` の `block_swap_prefetch` と `fused_gguf_dequant_kernel` は既定のジョブでも `true` として載る（`Docs/VERIFICATION_LOG.md` §43.4／§44.4／§51.5 の完全一致テスト群が固定している。**このキー集合が増えてよいのは「実機ゲート合格後の既定反転」のときだけ**である）。
> - **`block_swap_prefetch` と `fused_gguf_dequant_kernel` の既定は `true`**（`attention_backend` とは既定値の向きが逆）。いずれも実機ゲート（ビット一致＋VRAM）合格を条件に、開発時の既定 `false` から反転したものである（前者は 2026-08-02、経緯は `Docs/VERIFICATION_LOG.md` §44.7／後者は 2026-08-04、経緯は同 §51.5）。
> - **`vae_mode` の既定は恒久的に `"default"`（off）である**。`block_swap_prefetch` / `fused_gguf_dequant_kernel` のような「実機ゲート合格後の既定反転」ステップは**置かない**（オーナー確定事項 2026-08-05）。出力の絵が変わる機能であり、利用者が意図して選ぶべきものだからである。したがって既定のジョブでは worker ペイロードにキーが載らず、**凍結してある既定リクエストの鍵集合は本フィールドの実装によって1つも増えていない**。
> - 詳細な設計判断・実装箇所一覧・実機ゲート・実測値は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §43（`attention_backend`）・§44（`block_swap_prefetch`）・§51（`fused_gguf_dequant_kernel`）・§52（`vae_mode`）を正本とする。`vae_mode` の仕様と設計判断の正本は [`Docs/PRUNAVAED_WORKORDER.md`](Docs/PRUNAVAED_WORKORDER.md)。

> **`keep_resident` の補足（2026-08-03追加）**: 本フィールドも NAG・Acceleration の各フィールドと同じく `GenerateChainRequest` に同一の名前・型・デフォルトで存在し、`to_clip_request` 経由で `ClipGenerateRequest` へ転記される。既定（`false`）のジョブは worker ペイロードのキーが1つも増えない（`true` のときだけ加算する方式）。**`GET /status` には意図的に載せていない**——`/status` は「サーバーが実際にできること」を書く場所であり、`keep_resident` は `sage_available` のような可否判定（能力ゲート）を持たないためである（このマシンに十分なメモリがあるかという利用者側の選択にすぎない）。旧来の環境変数 `LTX_KEEP_RESIDENT` は**撤去済み**で、真実源は本フィールドへ一本化されている（§9.2、[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §48）。

> **チェーンの参照動画（長尺IC-LoRA）の補足（2026-08-11追加）**: `GenerateChainRequest` の `reference_video_id`／`conditioning_attention_strength`／`reference_video_strength`（§6.2の3フィールドと同義。§6.3ではなく`GenerateChainRequest`固有のため本節で補足する）は、**2026-07-11の解禁時点では「clips=1本限定」だったが、2026-08-11に多クリップへ拡張された**。長い参照動画を1本だけ添付すると、`chain_math.video_segment_windows`（開始＝8×そのクリップのグローバルな潜在開始位置／長さ＝そのクリップのフレーム数）に従って各クリップが担当する区間をサーバーが自動で切り出し、stage-1（低解像度で全体の動きを作る第1段階）にのみ注入する。stage-2 には入らないため VRAM の天井は悪化しない。隣り合う窓は `8kv−7` ピクセルフレーム重なるため、つなぎ目の参照映像が前後のクリップで自動的に一致する。参照が生成の尺より短ければ、足りないぶんのクリップは参照なしで生成される（422にはしない）。**`depth-control` のみ2クリップ以上で明示422**（`LORA_DEPTH_CHAIN_UNSUPPORTED`、§6.8）。あわせて `GET /loras`（`api/loras.py::list_loras`。凍結表未掲載のADDITIVEエンドポイント、詳細はフロントエンド`Docs/API_REFERENCE.md` §3.6）が返す各アダプタの行に `preprocess`（前処理種別。`depth`かどうかの判定に使う）と `reference_downscale_factor`（参照動画の内部ダウンスケール倍率。2または1）を追加し、`metadata.json`（§6.6）のトップレベルに `ic_lora` ブロック（`reference_segment_windows`＝各クリップが実際に参照した区間の配列 等）を追加した。詳細な設計判断・実測・実機ゲートG1〜G10の正本は[`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §57。

- `width`/`height` は two-stage distilled が stage-1 を半解像度で生成し 2x アップサンプルするため **64 の倍数**（32 からの意図的な厳格化。960x540 等の非 64 表示サイズは `crop_output` で得る）。
- バリデータ失敗はすべて 422（`VALIDATION_ERROR` エンベロープ、§6.8）。
- 派生プロパティ `generation_mode` は `conditioning_images` が空なら `"t2v"`、あれば `"i2v"`（フィールドではなく `@property`）。

### 6.3 その他スキーマ

すべて `api/models.py` の実フィールド。

**CropOutput**
| フィールド | 型 | 制約 |
|-----------|----|------|
| `width` | int | `ge=32` |
| `height` | int | `ge=32` |

**ConditioningImage**
| フィールド | 型 | デフォルト | 制約 |
|-----------|----|-----------|------|
| `image_id` | str | （必須） | — |
| `frame_idx` | int | `0` | Phase 1 は 0 固定（バリデータ） |
| `strength` | float | `0.8` | `ge=0.0, le=1.0` |
| `crf` | int \| null | `null` | — |

> 注: `crf` はスキーマに存在するが、real backend の I2V 送信ペイロードでは **落とされる**（`ltx_runner._RealBackend.generate` はフォークの `ImageConditioningInput` に crf が無いため `path/frame_idx/strength` のみ送る）。

**UploadImageResponse**
| フィールド | 型 |
|-----------|----|
| `image_id` | str |
| `original_filename` | str |
| `stored_path` | str（`uploads/{image_id}/input.png`） |
| `width` | int |
| `height` | int |
| `content_type` | str |

**JobStatus**（`str, Enum`）: `queued` / `running` / `completed` / `failed` / `cancelled`。

**GenerateResponse**
| フィールド | 型 |
|-----------|----|
| `job_id` | str |
| `status` | JobStatus |
| `created_at` | str（`YYYY-MM-DDTHH:MM:SSZ`） |

**JobResult**
| フィールド | 型 | 値の由来 |
|-----------|----|---------|
| `video_url` | str | `/api/v1/jobs/{job_id}/video` |
| `duration_seconds` | float | `round(num_frames / frame_rate, 3)` |
| `resolution` | str | `"{w}x{h}"`（crop 指定時は crop サイズ） |
| `file_size_bytes` | int | output.mp4 のサイズ |
| `generation_time_seconds` | float | `round(elapsed, 2)` |
| `seed_used` | int | 実際に使われたシード |
| `output_path` | str | `outputs/{job_id}/output.mp4` |
| `metadata_path` | str | `outputs/{job_id}/metadata.json` |

**JobResponse**
| フィールド | 型 |
|-----------|----|
| `job_id` | str |
| `status` | JobStatus |
| `progress` | float |
| `current_step` | int \| null |
| `total_steps` | int \| null |
| `is_v2v` | bool |
| `joined` | bool |
| `created_at` | str |
| `started_at` | str \| null |
| `completed_at` | str \| null |
| `error` | str \| null |
| `request` | GenerateRequest |
| `result` | JobResult \| null |

> `is_v2v` / `joined` は**2026-07-21追加（V2V Join復活・コミット `d22706e`）**。`is_v2v` は `chain_request` に `source_video` があるV2Vジョブ（＝ `POST /jobs/{job_id}/join` の対象になり得るジョブ）かどうか、`joined` はサーバー側の連結済み動画 `outputs/{job_id}/joined.mp4` が現存するか（＝ join 実行済みで未削除か）を表す。

### 6.4 代表的な Req/Res 例（JSON）

**(a) T2V generate リクエスト**
```json
{
  "prompt": "A flowing river in a forest at golden hour, cinematic, high detail",
  "negative_prompt": "blurry, low quality, distorted",
  "width": 512,
  "height": 320,
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
レスポンス（202）:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "created_at": "2026-06-25T12:00:00Z"
}
```

**(b) I2V（upload → generate）**

まず `POST /api/v1/upload/image`（multipart `file=@start.png`）:
```json
{
  "image_id": "0f3d2c1b-9876-4321-aaaa-1234567890ab",
  "original_filename": "start.png",
  "stored_path": "uploads/0f3d2c1b-9876-4321-aaaa-1234567890ab/input.png",
  "width": 512,
  "height": 320,
  "content_type": "image/png"
}
```
次に `POST /api/v1/generate`:
```json
{
  "prompt": "The character slowly turns their head, cinematic, high detail",
  "negative_prompt": "blurry, low quality, distorted",
  "width": 512,
  "height": 320,
  "crop_output": null,
  "num_frames": 49,
  "frame_rate": 24.0,
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "seed": 42,
  "pipeline": "distilled",
  "conditioning_images": [
    { "image_id": "0f3d2c1b-9876-4321-aaaa-1234567890ab", "frame_idx": 0, "strength": 0.8 }
  ]
}
```

**(c) GET /api/v1/jobs/{job_id}（完了時）**
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
  "request": { "...": "GenerateRequest 全文" },
  "result": {
    "video_url": "/api/v1/jobs/a1b2c3d4-e5f6-7890-abcd-ef1234567890/video",
    "duration_seconds": 2.042,
    "resolution": "512x320",
    "file_size_bytes": 4523008,
    "generation_time_seconds": 72.4,
    "seed_used": 42,
    "output_path": "outputs/a1b2c3d4-e5f6-7890-abcd-ef1234567890/output.mp4",
    "metadata_path": "outputs/a1b2c3d4-e5f6-7890-abcd-ef1234567890/metadata.json"
  }
}
```

> 実行中の `progress` は real backend では `0.05 → 0.90 → 1.0` の粗い区切り（`current_step`/`total_steps` は `null`）。mock backend はステップ進捗を細かく刻み `current_step`/`total_steps` を数値で埋める。

### 6.5 GET /status の `vram_optimization`（凍結キー集合）

`vram_optimization` ブロックは `services/low_vram.py::LowVramSettings.status_block()` が生成する。正本は同ファイルの `_STATUS_KEYS` タプル（7 キー）＋末尾 1 キー。**この 8 キー・この順序が契約**であり、内部専用フラグ（`block_swap_blocks_on_gpu`, `vae_*_tile_size`, `te_offload_text_encoder`, `dit_cpu_load`）は `status_block()` のフィルタで**混入しない**。

キー（生成順）:

| キー | 値の由来（config.vram / limits） | 意味 |
|-----|-------------------------------|------|
| `low_vram_mode` | `vram.low_vram_mode` | 低 VRAM モード |
| `low_vram_profile` | `vram.low_vram_profile` | プロファイル名（例 `16gb_safe`） |
| `fp8_transformer` | `vram.fp8_transformer` | 表示専用（下記注） |
| `cpu_offload_text_encoder` | `vram.cpu_offload_text_encoder` | 表示専用（下記注） |
| `vae_tiling` | `vram.vae_tiling` | VAE タイリング |
| `attention_tiling` | `vram.attention_tiling` | Attention タイリング（Phase 1 は false） |
| `block_swap` | `vram.block_swap` | ブロックスワップ |
| `low_vram_disabled_required` | `limits.low_vram_disabled_required` | status_block 引数として付与 |

例（実効 config.yaml の値を反映）:
```json
"vram_optimization": {
  "low_vram_mode": true,
  "low_vram_profile": "16gb_safe",
  "fp8_transformer": true,
  "cpu_offload_text_encoder": true,
  "vae_tiling": true,
  "attention_tiling": false,
  "block_swap": true,
  "low_vram_disabled_required": false
}
```

**表示専用の明記（実装の実態）**: `fp8_transformer` と `cpu_offload_text_encoder` は status / metadata 契約に出力されるが、**worker ペイロードには伝播しない**。fp8 はランタイムの `device_supports_fp8` 自動判定で選ばれ、CPU text-encode オフロードは別フィールド `te_offload_text_encoder`（環境変数 `LTX_TE_OFFLOAD`）で駆動される。契約互換のため 2 キーは削除せず保持しているが、値そのものは worker 挙動を変えない。

（GET /status 全体の形は `api/status.py` を参照。`gpu` ブロックは `gpu_info.get_gpu_info()`、`queue` は `mode="single_job_in_memory"` に `job_store.counts()`=`pending/running/completed/failed` を展開。`version`="0.4.0"（API 実装のバージョン文字列であり、本仕様書の版 v0.5 とは別物）。）

> **`gpu.available` が常に `false` になる件（既知・仕様どおり）**: `GET /status` の `gpu` ブロックは、アプリ用仮想環境（`./.venv`、torch 無し）から見た値を返すため、実機でも `available: false` になる。2プロセス／2仮想環境という構成（§2）に由来する既存の挙動で、実際の GPU はエンジン側 worker プロセスが握っている。紛らわしいが不具合ではない。

### 6.5b GET /status の `acceleration`（**非凍結**・2026-07-31追加）

`services/pipeline_manager.py::PipelineManager.acceleration_status_block()` が生成するトップレベルブロック。§6.5 の `vram_optimization`（凍結キー集合）とは別物で、**こちらは凍結対象ではない**（`vram_optimization` には一切触れていない）。

```json
"acceleration": {
  "attention_backends": ["sdpa", "sage"],
  "sage_available": true,
  "block_swap_prefetch_available": true
}
```

- `attention_backends`: `attention_backend`（§6.2）が受け付ける値の一覧。**実装のある項目だけ**を並べる。**`vae_mode` は 2026-08-05 に実装されたが、この acceleration ブロックには意図的に載せていない**（`keep_resident` と `fused_gguf_dequant_kernel` と同じ理由——環境依存の可否ではないためである）。`vae_mode` は「サーバーの環境で使えるか」ではなく「枝刈りデコーダのファイルがその瞬間に存在するか」で決まり、しかも判定はジョブ単位である。**起動時に1回答える `/status` に載せると、実態と食い違う値を返しうる**（`Docs/VERIFICATION_LOG.md` §52.7 のとおり、サーバーを動かしたままファイルを着脱できる）。実際にどちらで生成されたかは `metadata.json` の `vae_mode_used` で確認する、というのがこのフィールドの正しい観測経路である。
- `block_swap_prefetch_available`（**2026-08-02追加**）: 先読み block swap（`block_swap_prefetch`、§6.2）が実際に効く構成かどうか。判定式は実ゲートと完全同一で `not runner.is_mock and int(low_vram.block_swap_blocks_on_gpu or 8) > 0`（`services/pipeline_manager.py::_block_swap_prefetch_available`）。§6.5 の凍結 `vram_optimization` ブロックには一切触れていない。`block_swap_blocks_on_gpu=0` を `or 8` により実質8として扱う式は本項のために新設したものではなく、`services/ltx_runner.py:1086` に既存の式をそのまま流用している。詳細は `Docs/VERIFICATION_LOG.md` §44.1・§44.8。
- `sage_available`: SageAttention が使えるかどうか。サーバーの状態によって判定経路が変わる（**真理値表**。この表の置き場が `acceleration_status_block()` である）:

| サーバーの状態 | `sage_available` の由来 |
|---|---|
| mock backend | 常に `false`（エンジンが存在しない） |
| パイプライン未ロード | エンジン用仮想環境の site-packages に `sageattention/` と `triton/` が両方あるかの**ファイル存在チェック** |
| パイプライン ロード済み | **worker プロセス自身の import 判定**（こちらが権威。DLL ロード失敗や ABI 不一致まで捕捉できる） |
| ロード失敗 / unload 後 | ファイル存在チェック（生きた worker が無いため） |

2つの経路は「import できる**はず**か」と「実際に使うプロセスで import **できる**か」という別々の問いへの答えであり、どちらの値かは同じ応答の `pipeline_loaded` を見れば判別できる。そのため判定経路を示す `sage_source` のような追加フィールドは設けていない。

**クライアント側の作法**: `sage_available` が `false` でも `attention_backend="sage"` を送ること自体は許される（サーバーが `"sdpa"` へ降格して完走する）。値が取れない・不明な場合に UI 側で `sage` を封じる必要はない。

### 6.6 metadata.json スキーマ

`services/pipeline_manager.py::_write_metadata` が書き出す（`output.save_metadata_json` が true のとき）。実フィールド:

| キー | 値 |
|-----|----|
| `job_id` | str |
| `created_at` | str |
| `started_at` | str |
| `completed_at` | str（書き出し時刻 `now_iso()`） |
| `status` | 固定 `"completed"` |
| `request` | `GenerateRequest.model_dump()` |
| `generation_mode` | `"t2v"` / `"i2v"`（outcome 由来） |
| `seed_used` | int |
| `attention_used` | str \| null（**2026-07-31追加**。実際に使われた attention の実装＝`"sdpa"` / `"sage"` / `"sage->sdpa"`。`seed_used` とまったく同じ経路〔worker の完了イベント → outcome → メタデータ〕で書き出される。mock backend や旧 worker では `null`） |
| `block_swap_prefetch_used` | str \| null（**2026-08-02追加**。実際に効いた先読み block swap の状態＝`"off"` / `"on"` / `"on->off"`〔on を要求したが block swap 未インストール・pinned 確保失敗などで同期経路へ降格〕。`attention_used` と同じ経路で書き出される。mock backend や旧 worker では `null`） |
| `keep_resident_used` | str \| null（**2026-08-03追加**。モデル骨格の常駐が実際に効いたか＝`"off"` / `"on"` / `"on->off"`〔on を要求したが `dit_cpu_load=false` / `block_swap_prefetch=false` との併用で自動 off になった〕。`attention_used` と同じ経路で書き出される。mock backend や旧 worker では `null`。詳細は `Docs/VERIFICATION_LOG.md` §48） |
| `fused_gguf_dequant_kernel_used` | str \| null（**2026-08-04追加**。GGUF 逆量子化の1カーネル化が実際に効いたか＝`"off"` / `"on"` / `"on->off"`〔on を要求したが実際には適用されなかった。Triton 不在・カーネル例外での降格・型ごとの初回自己検証の不一致・対象テンソル0件のいずれか〕。`attention_used` と同じ経路で書き出される。mock backend や旧 worker では `null`。実機ゲートの判定基準もこのフィールドである。詳細は `Docs/VERIFICATION_LOG.md` §51） |
| `peak_vram_reserved_mb` | int \| null（**2026-08-02追加**。`torch.cuda.max_memory_reserved` 換算 MB。既存の `vram_optimization.peak_vram_mb`〔`max_memory_allocated`〕はstream別プール分断・reserved増を検知できないため、先読み block swap のVRAMリスクを見る指標として加算した。既存フィールドは置換していない） |
| `generation_time_seconds` | `round(elapsed, 2)` |
| `backend` | outcome.backend（mock は `"mock"`、real は `"ltx-distilled"`） |
| `output` | `{path, resolution, duration_seconds, frame_rate, file_size_bytes}` |
| `vram_optimization` | `LowVramSettings.metadata_block(peak_vram_mb=...)`（6 キー、下記） |
| `environment` | `{python, torch, cuda, gpu, platform}` |

`vram_optimization`（metadata 版、`metadata_block`）の 6 キー — status 版とは**別集合**である点に注意:
`low_vram_mode`, `low_vram_profile`, `fp8_transformer`, `cpu_offload_text_encoder`, `vae_tiling`, `peak_vram_mb`。

`environment` は `platform` を含む（`sys.platform`）。torch/cuda は app venv に torch が無ければ `null`。

例:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "created_at": "2026-06-25T12:00:00Z",
  "started_at": "2026-06-25T12:00:02Z",
  "completed_at": "2026-06-25T12:03:10Z",
  "status": "completed",
  "request": { "...": "GenerateRequest.model_dump()" },
  "generation_mode": "t2v",
  "seed_used": 42,
  "generation_time_seconds": 72.4,
  "backend": "ltx-distilled",
  "output": {
    "path": "outputs/a1b2c3d4-e5f6-7890-abcd-ef1234567890/output.mp4",
    "resolution": "512x320",
    "duration_seconds": 2.042,
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
    "torch": "2.9.1+cu128",
    "cuda": "12.8",
    "gpu": "NVIDIA GeForce RTX 4070 Ti SUPER",
    "platform": "win32"
  }
}
```

### 6.7 制約と limits

`config.yaml` の `limits`（`config.py::LimitsConfig`）実値:

| キー | 値 | 意味 |
|-----|----|------|
| `max_width` | `1920` | 生成幅の運用上限（Pydantic の `le=4096` とは別の運用リミット） |
| `max_height` | `1088` | 生成高の運用上限 |
| `max_num_frames` | `481` | 20s@24fps（481=8×60+1） |
| `max_conditioning_images` | `5` | I2V キーフレーム画像は最大5枚（Phase 3 で 1→5 に拡張） |
| `phase1_max_concurrent_jobs` | `1` | 単一ジョブ |
| `low_vram_disabled_required` | `false` | status に反映 |
| `spill_free_frames` | 下記マップ | 解像度別「溢れない」フレーム数（クライアント UI 警告用） |

`spill_free_frames`（キーは `"WxH"` 生成サイズ文字列）:
```yaml
"1280x768": 257    # 720p（実用快適上限）
"1920x1088": 153   # 1080p
"2560x1472": 81    # 1440p
```
API は 481f まで受理するが、この値を超えると shared メモリへ溢れて低速化（OOM はしない）。クライアント UI が警告する用途。実測根拠は `Docs/RESOLUTION_DURATION_CAPABILITY.md §8.4/§8.6` を参照（数値網羅表は複製しない）。

### 6.8 エラーコード

すべてのドメインエラーは `api/errors.py::APIError` として送出され、`main.py` の例外ハンドラでエンベロープに変換される:
```json
{ "error": { "code": "...", "message": "...", "job_id": "...", "detail": "..." } }
```
（`job_id`・`detail` は非 None のときだけ含まれる。）

実在するエラーコードと HTTP ステータス（`api/errors.py` のファクトリ 27 件と、ハンドラ側で生成される 2 件）:

| code | HTTP | 送出条件 |
|------|:---:|---------|
| `JOB_BUSY` | 409 | 実行中ジョブがある（generate）/ 実行中の unload |
| `UPLOAD_INVALID_TYPE` | 400 | 未対応画像形式・デコード不能・filename 欠落 |
| `UPLOAD_TOO_LARGE` | 400 | `upload.max_image_size_mb` 超過 |
| `IMAGE_NOT_FOUND` | 404 | `image_id` が存在しない |
| `REFERENCE_VIDEO_NOT_FOUND` | 404 | `reference_video_id` が存在しない（IC-LoRA の参照動画） |
| `SOURCE_VIDEO_NOT_FOUND` | 404 | `source_video.video_id` が存在しない（V2V 継続の元動画。参照動画と区別するための独立コード） |
| `SOURCE_VIDEO_TOO_SHORT` | 422 | V2V 継続の元動画のフレーム数が `context_frames` に足りない（GPU 作業の前に拒否） |
| `SOURCE_AUDIO_NOT_FOUND` | 404 | `source_audio.audio_id` が存在しない（A2V の元音声） |
| `SOURCE_AUDIO_TOO_SHORT` | 422 | A2V の元音声がチェーンのタイムラインより短い（音声は切り詰めのみで水増ししない） |
| `JOB_NOT_JOINABLE` | 422 | `POST /jobs/{id}/join` の対象が V2V 継続ジョブではない（metadata に `v2v` ブロックが無い） |
| `JOIN_FAILED` | 503 | サーバー側 ffmpeg の結合（正規化・クロスフェード・連結）が失敗 |
| `JOINED_NOT_READY` | 404 | `joined.mp4` の生成前に `GET /jobs/{id}/joined` を呼んだ |
| `LORA_NOT_FOUND` | 404 | 未登録の IC-LoRA アダプタ名 |
| `MODEL_NOT_FOUND` | 404 | 未登録のモデル名（またはカテゴリ不明）。生パスは受けない |
| `MODEL_FILE_MISSING` | 422 | 登録名は在るが重みファイルがディスク上に無い |
| `MODEL_INCOMPATIBLE` | 422 | 選択ファイルが互換性の事前チェックに失敗（拡張子違い・GGUF でない・safetensors ヘッダ破損） |
| `LORA_REQUIRES_REFERENCE` | 422 | 制御系 IC-LoRA を `reference_video_id` 無しで要求した |
| `REFERENCE_REQUIRES_CONTROL_LORA` | 422 | `reference_video_id` を渡したが、要求アダプタに制御系が1つも無い（逆方向チェック） |
| `LORA_DEPTH_CHAIN_UNSUPPORTED` | 422 | 2クリップ以上のチェーンで `depth-control`（深度制御）アダプタを要求した。深度マップを作る前処理が全編一括設計でメモリに載らないため、depth系のみ多クリップ非対応（**2026-08-11・長尺IC-LoRA実装で `LORA_CONTROL_UNSUPPORTED_IN_CHAIN` を置換**。他の制御系〔canny/pose〕・参照系〔upscaler/deblur〕アダプタは多クリップで受理される。クリップ1本のチェーンと単発生成は depth 込みで従来どおり使える） |
| `LORA_THUMBNAIL_NOT_FOUND` | 404 | サムネイル（`<stem>.png`）を持たないアダプタ、または未知のアダプタ名 |
| `LORA_PREPROCESS_CONFLICT` | 400 | 1本の参照動画に対して2種類以上の制御前処理を要求した（canny と pose の同時指定など） |
| `REFERENCE_RESOLUTION_INVALID` | 422 | 参照動画を使うジョブで `width`/`height` が 128 の倍数でない |
| `JOB_NOT_FOUND` | 404 | ジョブ未存在 / video 実体なし |
| `VIDEO_NOT_READY` | 409 | ジョブが completed 前に video 要求 |
| `PIPELINE_LOAD_FAILED` | 503 | パイプラインロード失敗 |
| `GPU_OOM` | 503 | 生成中の CUDA OOM |
| `GENERATION_FAILED` | 503 | 生成中の非 OOM 例外 |
| `UNAUTHORIZED` | 401 | api_key 設定時の Bearer 不一致/欠落（`deps.require_auth`） |
| `VALIDATION_ERROR` | 422 | Pydantic バリデーション失敗（`main.py` の RequestValidationError ハンドラ。`detail` は `loc/msg/type` の配列） |

> `GPU_OOM` / `GENERATION_FAILED` はジョブ実行スレッド（`run_job`）内で `job.error` 文字列（`"{code}: {message} ({detail})"`）として記録される。ジョブは `failed` で終了し、HTTP としては後続の `GET /jobs/{job_id}` の `error` フィールドで観測される。

関連: §7（ジョブ管理）, §11（config.yaml）, §4（backend 選択）, §10（解像度・尺の能力 = Docs/RESOLUTION_DURATION_CAPABILITY.md）。

---

## §7 ジョブ管理と Pipeline / Runner

本章は Phase 1 のジョブ実行モデルと、`PipelineManager` / `LTXRunner` の責務分担・実契約を定義する。実装は `services/job_store.py`, `services/pipeline_manager.py`, `services/ltx_runner.py`。

### 7.1 In-memory single job

- ジョブ履歴は `JobStore._jobs: dict[str, JobRecord]`（メモリのみ、DB もキューも無い）。**サーバー再起動でジョブ履歴は消える**が、`outputs/{job_id}/metadata.json`（と output.mp4）はディスクに残る。
- `job_id` は UUID4（`uuid.uuid4()`）。
- 単一ジョブ制約: `POST /generate` は `JobStore.create_if_idle()` で**アトミックに**予約する。既に active（`queued` または `running`）なジョブがあれば `None` を返し、ハンドラは `409 JOB_BUSY` を送出。
- `POST /pipeline/unload` も active ジョブがあれば `409 JOB_BUSY`。
- `GET /jobs` はメモリ上に残るジョブのみを返す。

### 7.2 状態遷移

`JobStatus`: `queued → running → completed`、または `running → failed` / `cancelled`。

```
[queued] → [running] → [completed]
                    └→ [failed]
                    └→ [cancelled]
```

- 投入直後は `queued`。バックグラウンドタスク（`run_job`）開始で即 `running`（`started_at` を記録）。
- `queued` は短命（`create_if_idle` 直後〜スレッド開始まで）。
- `DELETE /jobs/{job_id}` の挙動と応答形はジョブ状態で異なる（実装＝`api/jobs.py::delete_job`）:
  - **`running` ジョブ＝キャンセルはベストエフォート**: `record.cancel_requested = True` を立て、`{"job_id": ..., "cancel_requested": true, "status": ...}` を返す（推論は安全に中断できない）。生成が終わったとき `cancel_requested` が立っていれば最終ステータスを `cancelled` にする。
  - **`queued` ジョブ＝即時キャンセル（2026-07-11 追加・queued 詰まり対策・VERIFICATION_LOG §33）**: まだ実行に移っていないので、ベストエフォートではなく**即座に `cancelled` へ遷移**させ、単一ジョブガードを解放する。応答は `{"job_id": ..., "cancelled": true, "status": "cancelled"}` ＝**running 時とキー名が異なる**（`cancel_requested` ではなく `cancelled`）ため、API クライアント実装は取り違えに注意。`JobStore` は `_lock` 配下の CAS（compare-and-swap）ヘルパー（`start_job` / `cancel_if_queued`）で queued→running への昇格とキャンセル操作を相互排他化しており、昇格とキャンセルが競合しても取りこぼさない（`cancelled` になったジョブが `running` へ復活することはない）。これにより従来サーバー再起動でしか抜けられなかった「queued のまま無言で固まる」状態を GUI／API から解消できる。
  - **終了済みジョブ＝削除**: レコード削除＋ `outputs/{job_id}` ディレクトリ削除で `{"job_id": ..., "deleted": true}` を返す。

### 7.3 進捗コールバック

`run_job` が `on_progress(step, total, progress)` を runner に渡し、`JobRecord.current_step/total_steps/progress` を更新する。`GET /jobs/{job_id}` がこれを返す。real backend は `0.05 / 0.90 / 1.0`（step/total は None）、mock backend は step 単位で細かく更新。

### 7.4 PipelineManager と LTXRunner の責務分担

**PipelineManager**（`services/pipeline_manager.py`）:
- パイプラインの load / unload / auto-load ライフサイクルを所有（内部 state: `unloaded/loading/ready/running/error`）。
- `run_job(job)`: バックグラウンドスレッドで 1 ジョブを終端まで実行する。runner 未ロードかつ `model.auto_load_on_generate` が true なら自動ロード（false なら `PIPELINE_LOAD_FAILED`）。conditioning 画像パスを `upload_store.path_for` で解決し、`runner.generate(...)` に委譲。
- 完了後 `_finalize` で `JobResult` を組み、`_write_metadata` で **metadata.json を書くのは PipelineManager**（runner ではない）。
- 例外処理: OOM（`_is_oom` 判定）は `GPU_OOM` + `_cleanup_after_error()`（unload + memory cleanup）、それ以外は `GENERATION_FAILED`。いずれもジョブは `failed`。

**LTXRunner**（`services/ltx_runner.py`）: LTX 内部を知る唯一のファサード。mock / real の 2 backend を隠蔽。

### 7.5 LTXRunner の実契約

コンストラクタ: `LTXRunner(config, low_vram)`。公開プロパティ `loaded` / `pipeline_type` / `pipeline`（back-compat、real では常に None）。メソッド `load()` / `unload()` / `generate(...)`。

`generate` の実シグネチャ:
```python
generate(
    request: GenerateRequest,
    output_dir: Path,
    progress_callback: ProgressCallback | None = None,
    conditioning_image_paths: list[Path] | None = None,
) -> GenerationOutcome
```

戻り値 `GenerationOutcome`（dataclass）:
| フィールド | 型 | 内容 |
|-----------|----|------|
| `output_path` | Path | 常に `output_dir / "output.mp4"` |
| `seed_used` | int | 実使用シード（`seed=-1` は**親プロセス**で乱数解決し決定性を担保） |
| `peak_vram_mb` | int \| None | ピーク VRAM（real は worker が報告、無ければ None） |
| `generation_mode` | str | `"t2v"` / `"i2v"` |
| `backend` | str | mock=`"mock"` / real=`"ltx-distilled"`（`MOCK_BACKEND` / `REAL_BACKEND`） |

- `output_path` は必ず `output_dir/"output.mp4"`。`crop_output` 指定時、real backend は worker にフルサイズを `_full.mp4` へ書かせ、ffmpeg で中央クロップして `output.mp4` を生成する。
- metadata.json は runner の責務ではなく PipelineManager が書く。

### 7.6 backend 選択（概要）

`LTXRunner._select_backend()` が `config.model.backend`（`auto` / `mock` / `real`）で分岐:
- `mock` → 常に `_MockBackend`（GPU・重み不要）。
- `real` → `_real_available()` が真でなければ `RuntimeError`、真なら `_RealBackend`。
- `auto`（既定）→ `_real_available()` が真なら real、偽なら mock。

`_real_available()` の判定材料（すべて存在必須。存在チェックする load-bearing ファイル群）: `engine_python`, `gemma_root`（tokenizer-only）, `spatial_upsampler_path`, `gguf_transformer_path`, `gguf_gemma_path`, `component_video_vae_path`, `component_audio_vae_path`, `component_text_projection_path`、および `engine_dir/worker.py`。※ `checkpoint_path`（43GB モノリス）は reference-only のため**存在チェックしない**（worker payload に載るが GGUF+component 経路では開かれない）。詳細は §4 に譲る。

関連: §4（backend 実装詳細）, §6（API・metadata 契約）, §11（config.yaml）。

---

## §8 出力ファイル

### 8.1 ディレクトリ構造

生成結果はジョブごとに `outputs/{job_id}/` 配下へ保存する（`config.output.dir` 既定 `./outputs`）。

```text
outputs/
└── {job_id}/
    ├── output.mp4       # H.264 / yuv420p。音声ありジョブは AAC を mux
    └── metadata.json    # 生成メタデータ（save_metadata_json: true のとき）
```

- ジョブ履歴は in-memory（サーバー再起動で消える）。一方 `outputs/{job_id}/metadata.json` はディスクに残る。
- `metadata.json` のスキーマ自体は **§6.6** を正本とする（本章では重複させない）。`peak_vram_mb` は `metadata.json` の `vram_optimization` ブロック、または `logs/ltx_worker.log` の `GENERATED_OK peak_vram_mb=` から取得できる（jobs API 応答には含まれない）。

### 8.2 クロップ仕様（`services/video_io.py` 準拠）

LTX 2.3 の two-stage distilled は生成サイズが **64 の倍数**でなければならない（付録A・§6.7）。そのため、AviUtl2/DaVinci で扱いやすい非 64 サイズ（例 1280×720, 960×540）を得るには、**内部では ÷64 サイズで生成し、ffmpeg で中央クロップ**して配信する。

- リクエストの `crop_output`（`{width, height}`）を指定すると、real backend は full-size の mp4 を書き、`crop_mp4()` が ffmpeg の `crop` フィルタ（中央基準＝`crop=W:H:(in_w-W)/2:(in_h-H)/2`）で再エンコードして最終表示サイズを配信する。
- クロップ再エンコードは映像（`libx264`/`yuv420p`）と**音声（`-c:a copy` で AAC を保持）**を両方 map する。
- 代表例:

| 生成サイズ（÷64） | クロップ後（表示） |
|-------------------|--------------------|
| 960×576 | 960×540 |
| 1280×768 | 1280×720（720p。上下 24px を中央クロップ） |

---

## §9 低VRAM戦略（Phase 1 実装済み・既定 ON）

以下の機構はすべて first-party `engine/` に実装済みで、**Phase 1 で既定 ON** である。16GB 実機での効果は A/B で実測済み（代表値 + Docs ポインタを併記。網羅表は複製せず各 Docs を正本とする）。エンジン内部の詳細アルゴリズムは `engine/` コードと `engine/VENDOR_NOTICE.md` を参照。

### 9.1 実装済み機構一覧

| 機構 | 役割 | 代表効果（実測） | Docs |
|------|------|------------------|------|
| **block-swap**（GPU 常駐 8 / 全 48 ブロック） | DiT の 48 ブロックを CPU 常駐にし、denoise 中は N ブロックだけ GPU にストリーム | 常駐ベースライン ~5GB に圧縮（`block_swap_blocks_on_gpu=8`） | VERIFICATION_LOG §10 |
| **GGUF Q4_K_M 量子化**（transformer） | 22B DiT を GGUF Q4_K_M で VRAM 圧縮常駐、per-layer 逐次 dequant | ~17GB ファイルのまま VRAM 常駐（bf16 44GB を回避） | — |
| **GGUF Q4_K_M Gemma** | 24GB bf16 Gemma を ~7.3GB Q4_K_M で圧縮常駐 | フル bf16 が 16GB を溢れるのを回避 | — |
| **te-offload**（Gemma 逐次 per-layer CPU オフロード） | Gemma decoder 48 層を CPU 常駐にし encode 中に 2 層ずつ GPU へストリーム（compute は GPU） | encode peak **15,839 → 10,540 MB（−33%）**・encode 時 shared 溢れ消失・速度ペナルティ無し・出力バイト一致 | VERIFICATION_LOG §11 |
| **dit-cpu-load**（DiT を CPU 構築） | transformer を直接 CPU RAM に構築し非ブロックのみ GPU へ。ロード時の一過性 GPU スパイクを除去 | transformer-load peak **15,815 → 1,444 MB（−14.4GB）**・load 時 shared 溢れ消失・出力 SHA256 一致 | VERIFICATION_LOG §12 |
| **VAE tiling**（空間 512px / 時間 64f） | VAE decode をタイル化し解像度・尺に非依存化 | VAE decode ~1.9GB 固定（解像度・尺非依存） | RESOLUTION_DURATION §2 |
| **component-files 経路**（Path B） | VAE/audio/text-projection を 46GB モノリスでなく小単体ファイルから読む | マルチジョブ連続で commit 枯渇 native crash を回避 | VERIFICATION_LOG §10.3 |

**2つのピークと天井**: 16GB の天井は **逐次に現れる 2 つのピークの max** で決まる。peak ①（Gemma text-encode）を te-offload が、peak ②のうち transformer-LOAD スパイク成分を dit-cpu-load がそれぞれ下げ、両者の効果で 512×320 級では全体ジョブ天井が **16,944 → ~9.2GB** に低下する（VERIFICATION_LOG §12.5）。残る可変軸は高トークン時の denoise stage2 のみ（→ §10）。

### 9.2 本番既定の組合せ

本番 worker 起動時、`services/ltx_runner.py` は以下の env を子プロセスに設定する:

- `LTX_KEEP_RESIDENT` は **2026-08-02 に撤去済み**（この env はもう設定されないし、設定しても読まれない）。モデル骨格のジョブ間常駐は、環境変数ではなく `POST /generate`・`POST /generate/chain` の per-job フィールド（ジョブごとのリクエスト項目）`keep_resident`（既定 `false`）で指定する。§6.2 と [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §48 を参照。
- `LTX_COMPONENT_FILES=1`（`config.vram.use_component_files=true` に連動 / comp=1）。
- `LTX_TE_OFFLOAD=1` / `LTX_DIT_CPU_LOAD=1`（既定 ON、`--no-te-offload` / `--no-dit-cpu-load` で無効化）。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` / `TORCH_COMPILE_DISABLE=1` / `PYTHONUNBUFFERED=1`。

`keep=0`（ジョブ毎再 materialize）と `comp=1`（Path B）の組合せが本番既定である理由は、**マルチジョブ連続生成での commit（仮想メモリ）枯渇回避**にある。`keep=1` は 720p の Gemma text-encode 中に out-of-place な `.to(cuda)` で瞬間二重在が発生し native crash する（VERIFICATION_LOG §10.2）。`comp=0` は毎ジョブ 46GB モノリスを再 materialize して job3 で commit 枯渇 crash（同 §10.3）。`comp=1/keep=0` は 46GB モノリスを使わず commit を束縛し、T2V・I2V ともマルチジョブ連続 + 音声で PASS 済（同 §10.3 / §10.7）。

> **上段の keep=1 に関する記述は 2026-06-30 時点の判断である（2026-08-02 追記）。** その後 Gemma レイヤーオフロード導入後の構成で再検証し、native crash の原因だったデバイス移動の不具合を修正したうえで、`keep_resident` を per-job フィールドとして製品化した（既定は引き続き off）。現在の正しい理解は「本番既定は off のまま・利用者がジョブ単位で on にできる・on 時はメモリ 64GB 以上を推奨」であり、詳細は [`Docs/VERIFICATION_LOG.md`](Docs/VERIFICATION_LOG.md) §47・§48 を正本とする。

### 9.3 表示専用フィールド（worker へ非伝播）

`fp8_transformer` と `cpu_offload_text_encoder` は **凍結 API 契約フィールド**（`GET /status` の `vram_optimization` ブロックと `metadata.json` に露出）だが、**worker へは伝播しない・表示専用**である（§6.5 と整合）。実際の挙動は別経路で駆動する:
- fp8 は runtime の `device_supports_fp8` 自動判定（`use_fp8 = use_fp8_transformer or device_supports_fp8(device)`）。
- CPU text-encode オフロードは別フィールド `te_offload_text_encoder`（env `LTX_TE_OFFLOAD`）で駆動。

これらは `services/low_vram.py` の凍結 `_STATUS_KEYS` 契約を満たすために保持されているが、実低VRAM機構の ON/OFF は上記 §9.2 の内部ノブが握る。

### 9.4 low_vram_mode と高VRAM検証

`low_vram_mode=true` が Phase 1 の既定（16GB 環境前提）。`low_vram_mode=false` は高VRAM/クラウド環境向けの**任意検証**であり、16GB での成功は保証しない。**16GB で `low_vram_mode=false` が失敗しても Phase 1 の失敗ではない**（README §7）。`allow_disable_low_vram: true`（config）は無効化を許容するフラグに過ぎない。

---

## §10 解像度×尺の能力と音声

数値は**代表値のみ**を引く。網羅表・推定式・全スイープ結果は複製せず、正本を `Docs/RESOLUTION_DURATION_CAPABILITY.md`（以下 RES-DUR）とする。乖離時は RES-DUR を正とする。

### 10.1 実測代表値（Acceleration 全オフ時の基準値。16GB / RTX 4070 Ti SUPER・本番既定 comp=1/keep=0/bs=8/vae512-64）

| 解像度（生成÷64） | 尺 | 生成時間 | 全体 peak | 出所 |
|-------------------|----|----------|-----------|------|
| 720p（1280×768） | 121f（5s） | **~167–171s** | peak ~16.9GB（全体天井） | RES-DUR §0 / VERIFICATION_LOG §10.1 |
| 1080p（1920×1088） | 121f（5s） | **262s（4.3分）** | — | RES-DUR §3.2 |

- 上表は **Acceleration（生成の高速化）を全オフにしたときの基準値**である。**現行の既定（先読み block swap on・GGUF 逆量子化の1カーネル化 on）では合計3割前後短くなる**（`Docs/VERIFICATION_LOG.md` §44／§51）。
- **生成サイズは必ず ÷64**（two-stage distilled の契約）。表示解像度へは `crop_output` の中央クロップで得る（例 1280×768 → 1280×720）。crop は既定 OFF。
- 実測の秒数・GB は RES-DUR の代表値であり、細部は同 Docs を参照。

### 10.2 溢れない尺の上限（spill-free frames）

VRAM が溢れ始めない最長尺は解像度別に異なり、`GET /api/v1/config` の `limits.spill_free_frames` に露出する（RES-DUR §8.4 / §8.6 が正本）:

| 表示解像度 | 生成サイズ | 溢れない最長（実測） |
|------------|-----------|----------------------|
| **720p** | 1280×768 | **257f（API 上限 10.67s・clean）**。物理溢れ境界は 257<x<321f |
| **1080p** | 1920×1088 | **~153f（~6.3s）** |
| **1440p** | 2560×1472 | **~81f（~3.3s）** |

これを超えると WDDM shared へ溢れて **~2–4x 低速化**（OOM せず完走）。天井付近の生成時間は解像度に依らず ~334–344秒（~5.7分）に収束する（RES-DUR §8.6）。720p は API `num_frames≤257` が物理溢れ境界より手前で効くため、実運用では 257f が溢れない最長となる。

### 10.3 1080p 長尺は非実用 → 720p 生成 + 外部アップスケール推奨

1080p の長尺は生成時間が**非線形に**伸びる（spill による PCIe 帯域律速ページング。RES-DUR §8.2 で、フレーム増に対する時間比が 1.22 → 3.20 に膨張することを実証）。20s 級は ~40分に達し commit リスクもあるため非実用である。したがって **720p で生成し、外部アップスケーラで 1080p 化する**運用を推奨する。

### 10.4 native joint audio（Phase 1）

LTX-2.3 の **native joint audio** は 16GB 実機で正常動作する（VERIFICATION_LOG §9.8 = 音声 Phase 1 の最終結論）:
- **AAC / 48kHz / stereo** を distilled pipeline が自動生成し、mp4 に mux する。**効果音・音楽・発話（セリフ）**を 16GB で生成可能（内容はプロンプト依存・ベストエフォート）。発話テストで "Hello" が明瞭に聞こえることを確認済。
- **crop 後も音声を保持**する（`crop_mp4` が `-map 0:v -map 0:a? -c:a copy` でストリームコピー・無劣化）。
- decode 区間の VRAM は dedicated ~2.7GB / shared ~0.5GB で、denoise（~14–15GB）・全体ピークより**大幅に低く**、音声デコードは 16GB のボトルネックではない。
- I2V マルチジョブ + 音声連続も本番既定（comp=1/keep=0）で PASS 済（VERIFICATION_LOG §10.7）。

---

## §11 config.yaml

本章は現行 `config.yaml` の全セクションを実値どおりに記す。各フィールドは `config.py` の Pydantic 型（`ServerConfig` / `ModelConfig` / `VramConfig` / `GenerationDefaults` / `LimitsConfig` / `UploadConfig` / `OutputConfig`）に対応する。

> **実ファイルとの乖離時は `config.yaml` が正**。パスは相対のとき `AppConfig._abs()` が PROJECT_ROOT 基準で絶対化する。

### 11.1 server
| キー | 実値 | 説明 |
|-----|------|------|
| `host` | `"127.0.0.1"` | バインドアドレス（`--listen` で 0.0.0.0） |
| `port` | `18620` | ポート（`--port` で上書き可） |
| `allow_all_cors` | `false` | 既定は localhost 限定 CORS（`--allow-all-cors` で全許可） |
| `api_key` | `null` | Bearer 認証キー（設定時のみ認証要求。`--api-key` で上書き） |
| `log_dir` | `"./logs"` | ログ出力先 |

### 11.2 model
| キー | 実値 | 説明 |
|-----|------|------|
| `checkpoint_dir` | `"./models"` | モデルルート |
| `checkpoint_name` | `"ltx-2.3-22b-distilled-1.1"` | チェックポイント名（表示用） |
| `text_encoder` | `"google/gemma-3-12b-it-qat-q4_0-unquantized"` | テキストエンコーダ識別子（表示用） |
| `pipeline_type` | `"distilled"` | `pipeline_type` プロパティの元 |
| `auto_load_on_generate` | `true` | 初回 generate で自動ロード |
| `reload_interval` | `0` | 再ロード間隔（0=無効） |
| `ltx_repo_dir` | `"./vendor/LTX-2"` | reference-only（上流クローン） |
| `spatial_upsampler_path` | `"./models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"` | 空間 2x アップサンプラ（load-bearing） |
| `gemma_root` | `"./models/gemma-3-12b-it-tokenizer"` | **tokenizer-only ~40MB**。DistilledPipeline を `gemma_root=None` で構築し重み glob をバイパス、engine は tokenizer/processor の module_ops のみ読む（load-bearing、存在チェックあり） |
| `backend` | `"auto"` | `auto` \| `mock` \| `real`（auto: GPU+モデル有→real、無→mock） |
| `gguf_transformer_path` | `"./models/ltx-2.3-gguf/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"` | GGUF 量子化トランスフォーマー（16GB レシピ Q4_K_M） |
| `gguf_gemma_path` | `"./models/gemma-3-12b-it-gguf/gemma-3-12b-it-Q4_K_M.gguf"` | GGUF Gemma（Q4_K_M） |
| `engine_dir` | `"./engine"` | 自前 engine パッケージ（`python -m engine.worker`） |
| `engine_python` | `"./.venv-engine/Scripts/python.exe"` | engine worker を回す専用 venv インタプリタ（torch+cu128） |
| `gguf_per_layer_quant` | `true` | GGUF 逐次レイヤー量子化 |
| `component_video_vae_path` | `"./models/ltx-2.3-components/vae/LTX23_video_vae_bf16.safetensors"` | 単体 VIDEO VAE（load-bearing） |
| `component_audio_vae_path` | `"./models/ltx-2.3-components/vae/LTX23_audio_vae_bf16.safetensors"` | 単体 AUDIO VAE/vocoder（load-bearing） |
| `component_text_projection_path` | `"./models/ltx-2.3-components/text_encoders/ltx-2.3_text_projection_bf16.safetensors"` | 単体 text projection（load-bearing） |
| `ic_loras` | `pixel-spatial-upscaler-x2` / `canny-control` / `pose-control` / `depth-control` / `deblur` の 5 エントリ | IC-LoRA アダプタの**名前 → パス**登録。API の `GenerateRequest.loras[].name` はここに登録された**名前でのみ**解決する（生パスは受けない）。値は文字列（＝`preprocess: none`・Phase B 互換）または `{ path, preprocess }` マップ。`canny-control`（`preprocess: canny`）・`pose-control`（`preprocess: dwpose`）・`depth-control`（`preprocess: depth`・2026-08-03 追加）は**同一の union-control ファイル**を 3 つの論理名で公開したもの。`deblur`（2026-08-03 追加）は前処理不要のため**文字列形式**で登録する。**セクション不在＝`loras` 要求は全て拒否（fail loud）**。実体ファイルの取得と検証は §5.1b（`_real_available()` は見ないため、欠けると UI に名前は出るのに選択時 404 になる）。depth-control / deblur の正本は `Docs/ICLORA_DEPTH_DEBLUR_WORKORDER.md`（2026-08-04にG3・G4合格でテーマ完結） |

> **2026-07-28**: 本表はかつて`checkpoint_path`（43GBモノリスへの reference-only パス）の行を含んでいたが、`config.model`から削除済みのため本表から除去した（§5.2・§5.5・`Docs/VERIFICATION_LOG.md` §40）。worker payload 自体には`checkpoint_path`キーが残るが、値は`services/ltx_runner.py`が直値`""`をハードコードするため、config側に対応するフィールドは無い。

### 11.3 vram
| キー | 実値 | 説明 |
|-----|------|------|
| `low_vram_mode` | `true` | 低 VRAM モード |
| `low_vram_profile` | `"16gb_safe"` | プロファイル名 |
| `fp8_transformer` | `true` | **凍結 API 契約・表示専用**（status/metadata に出るが worker 非伝播。実 fp8 は device_supports_fp8 自動判定） |
| `cpu_offload_text_encoder` | `true` | **凍結 API 契約・表示専用**（同上。実オフロードは `te_offload_text_encoder` が駆動） |
| `te_offload_text_encoder` | `true` | 内部専用。GGUF Gemma の逐次レイヤー CPU オフロード（encode peak ~15GB→数 GB）。`LTX_TE_OFFLOAD` で worker へ |
| `dit_cpu_load` | `true` | 内部専用。DiT を CPU 構築し非ブロックのみ GPU へ（ロード時 ~16.9GB スパイク除去）。`LTX_DIT_CPU_LOAD` で worker へ |
| `vae_tiling` | `true` | VAE タイリング |
| `attention_tiling` | `false` | Phase 2+ 用。Phase 1 は無効 |
| `attention_tile_size` | `null` | 同上 |
| `block_swap` | `true` | ブロックスワップ |
| `block_swap_blocks_on_gpu` | `8` | GPU 常駐ブロック数（内部 knob、status 非出力） |
| `vae_spatial_tile_size` | `512` | real engine の VAE 空間タイルサイズ（0=engine 既定） |
| `vae_temporal_tile_size` | `64` | real engine の VAE 時間タイルサイズ |
| `use_component_files` | `true` | 単体 component ファイル経路を使う（`LTX_COMPONENT_FILES`）。commit 枯渇クラッシュ回避に必須 |
| `allow_disable_low_vram` | `true` | 高 VRAM 環境向け任意検証用（16GB 成功は非保証） |

> `block_swap` は現行 config.yaml で `true`。status の `vram_optimization.block_swap` はこの値をそのまま反映する。

### 11.4 generation_presets
生成サイズ（width/height）は必ず 64 の倍数。最終表示サイズは crop_output で。

| プリセット | width | height | crop_output | num_frames |
|-----------|------:|------:|-------------|----------:|
| `smoke_test` | 384 | 256 | null | 17 |
| `minimal` | 512 | 320 | null | 49 |
| `small` | 960 | 576 | `{960, 540}` | 121 |
| `standard_720p` | 1280 | 768 | `{1280, 720}` | 257 |
| `FHD_1080p` | 1920 | 1088 | `{1920, 1080}` | 153 |
| `WQHD_1440p` | 2560 | 1472 | `{2560, 1440}` | 81 |

> `standard_720p` / `FHD_1080p` / `WQHD_1440p` の num_frames は `limits.spill_free_frames`（§11.7）の解像度別快適上限（spill-free 実測値）と一致させてある。

### 11.5 generation_defaults
Gradio / API の初期値。
| キー | 実値 |
|-----|------|
| `width` | `512` |
| `height` | `320` |
| `crop_output` | `null` |
| `num_frames` | `49` |
| `frame_rate` | `24.0` |
| `num_inference_steps` | `8` |
| `guidance_scale` | `1.0` |
| `seed` | `-1` |
| `pipeline` | `"distilled"` |
| `conditioning_images` | `[]` |

> 注: `config.py::GenerationDefaults` のコード上のデフォルト `height` は `288` だが、**実 config.yaml は `320`**。ロード時は config.yaml が勝つ。

### 11.6 upload
| キー | 実値 | 説明 |
|-----|------|------|
| `dir` | `"./uploads"` | アップロード保存先 |
| `max_image_size_mb` | `20` | 画像最大サイズ（超過で UPLOAD_TOO_LARGE） |
| `allowed_image_extensions` | `[".png", ".jpg", ".jpeg", ".webp"]` | 対応拡張子 |
| `normalize_to_png` | `true` | PNG 正規化（EXIF orientation 反映・RGB 変換） |

### 11.7 limits
§6.7 の表と同一（`max_width=1920`, `max_height=1088`, `max_num_frames=481`, `max_conditioning_images=5`, `phase1_max_concurrent_jobs=1`, `low_vram_disabled_required=false`, `spill_free_frames`={"1280x768":257,"1920x1088":153,"2560x1472":81}）。

### 11.8 output
| キー | 実値 | 説明 |
|-----|------|------|
| `dir` | `"./outputs"` | 出力先 |
| `format` | `"mp4"` | 出力形式 |
| `save_metadata_json` | `true` | metadata.json を書くか |
| `keep_raw_frames` | `false` | 生フレーム保持 |

関連: §6（API・metadata・limits 契約）, §7（Runner が読む vram 設定）, §4（backend 選択が読む model パス群）。

---

## §12 Gradio 検証UI

検証・デモ用の Gradio UI を FastAPI アプリの **`/ui`** にマウントする（`main.mount_gradio`。`gradio_ui.build_ui`）。環境変数 `LTX_DISABLE_GRADIO` を設定すると無効化できる（テストはこれで headless 化）。

### 12.1 設計方針

**UI は LTX を直接呼ばない。** 将来のフロントエンド（AviUtl2, DaVinci Resolve）と同じく、**自身の REST API（`/api/v1/*`）を HTTP で叩く**薄いクライアントである。

```text
[optional] POST /api/v1/upload/image  -> image_id
POST       /api/v1/generate            -> job_id
poll GET   /api/v1/jobs/{job_id}       -> progress
GET        /api/v1/jobs/{job_id}/video -> mp4
```

### 12.2 機能（`gradio_ui/` 準拠）

タブ構成は **Generate | Clip Chain | Style LoRA | Jobs | Settings**。上段の共通バーは **server status 表示 + Refresh のみ**（旧 Load Model / Unload ボタンは廃止。モデルのロードは Settings→Models、アンロードは Settings→Danger zone に一本化）。**プロンプト**入力はタブの上に常時表示され、Generate / Clip Chain 双方で共用する。

**共有 Negative Prompt アコーディオン（2026-07-28追加）**: 共有プロンプトの直下・`Tabs` の外に「Negative Prompt」アコーディオンがあり、Generate/Clip Chain 両タブから見える単一の入力群を提供する。テキスト欄（既定値 `"blurry, low quality, distorted"`）は「non-CFG Negative」チェックボックスがONのときだけ編集可能で、OFFの間は非活性のまま既定文字列を保持する。「NAG / Other」ラジオ（Other選択時は即座にNAGへ戻るフォールバック＋トースト）と、nag_scale/nag_tau/nag_alphaの3スライダーを持つ。詳細な設計判断は `Docs/VERIFICATION_LOG.md` §38、APIフィールドは§6.2を参照。

**Generate タブ**:

- **negative_prompt** 入力 / **入力画像アップロード**（任意・1枚 → 最小I2V。`image strength` スライダ付き。frame_idx=0 固定。画像なし → T2V）/ キーフレーム画像・IC-LoRA の各アコーディオン。
- **width / height**（各 ×64）、**num_frames**（8n+1）、**frame_rate** の3つは `gr.Number` の**サーバー側 `minimum` を撤去**してある。手入力の途中（例: "80" と打つ途中の "8"）で最小値未満エラーが飛ぶ Gradio の不具合を根治するための措置で、代わりにページロード時の JS（`demo.load(js=...)`）が各入力の HTML `min` 属性を後付けする。動作としては**手入力は自由**、スピナー矢印は width/height が **64刻み**、num_frames が **8n+1刻み**でスナップする（サーバー側の 8n+1・÷64 検証は従来どおり有効）。
- **num_frames / Duration / frame_rate は1つのパネルに統合**（`gr.Group` で横3カラム）。中央カラムは入力欄を持たず、num_frames と frame_rate から算出した **Duration（"N.NNs"）をアクセントカラーで常時表示**する読み取り専用の要約。
- **preset** ドロップダウン（`GET /config` の `generation_presets`＝§11.4 の6種を解像度・フレーム数のラベル付きで列挙。選択で width/height/num_frames/crop を一括反映）。
- **crop width / height**（0=none。両方 >0 のとき `crop_output` を送る）/ **seed**（-1=random）。
- **キーフレーム画像アコーディオン**: 固定5スロット（I2Vの多段誘導）。**A2V（音声から動画生成）と併用時も5枚すべて配線済み**で、`frame_idx>0` はサーバー側で 8n+1 グリッドへスナップ＋動画尺内にクランプされる（`frame_idx=0` は開始フレーム扱い）。
- **A2V（音声から動画生成）アコーディオン**: 音声ファイルを添付すると、内部的には 1 クリップのチェーン生成（`POST /generate/chain` + `source_audio`）として送信する。**スタイルLoRA（画風・キャラクター系。`<lora:...>` 記法）とは併用できる**（2026-07-11 に解禁。`GenerateChainRequest.loras` の加算＝VERIFICATION_LOG §32。それ以前は排他だった）。**参照動画を要する control 系 IC-LoRA（canny／pose 等）も、`clips` がちょうど1つのチェーン（A2V を含む）に限り併用できる**（α版・2026-07-11 に解禁＝VERIFICATION_LOG §34。詳細は §13.4b の追記を参照）。`clips` が2つ以上のチェーン（Clip Chain タブでの複数クリップ連結）についても、2026-08-11に長尺IC-LoRAとして多クリップ解禁した（`chain_math.video_segment_windows` による自動区間切り出し・stage-1のみ注入・§6.2／§6.8／VERIFICATION_LOG §57）。深度前処理（Video-Depth-Anything）がメモリに載らない `depth-control` アダプタのみ、2クリップ以上で `422 LORA_DEPTH_CHAIN_UNSUPPORTED` で拒否される（他の制御系・参照系アダプタは多クリップで受理される）。
  - **音声長の事前チェック**: 添付が `.wav` の場合、送信前にクライアント側で長さを測定し、その設定（フレーム数・fps）が必要とする秒数に足りなければ、必要秒数を明示して送信を拒否する（API 呼び出しゼロ）。`.wav` 以外（mp3/m4a等）はクライアント側で測定できないためこのチェックをスキップし、サーバーの `422 SOURCE_AUDIO_TOO_SHORT` に委ねる（このエラーもヒント付きで表示される）。
  - **Frames の自動調整**: `.wav` を添付すると、その音声長に収まる最大の 8n+1 値を自動計算して num_frames へ入力し、トースト通知で知らせる（既存の値は上書きされる）。計算式は `((floor(音声秒数×fps)-1)//8)*8+1` を起点に、音声側の latent フレーム数（`chain_math.audio_latents_required`）で検算しながら 8 刻みで縮小し、最終的に `[9, 481]` へクランプする。`.wav` 以外の添付・クリア時は何もしない。
- **生成中はボタンをグレーアウト**: 「生成」ボタンはクリック直後に無効化され、ラベルが "Generating..." / "生成中…" に切り替わる。生成完了・失敗いずれの場合も必ずボタンが再有効化・ラベル復帰する（click→無効化→生成→復元 の3段イベント連鎖。Clip Chain タブの「Generate chain」ボタンも同じ挙動）。

**Clip Chain タブ**:

- モード切替は **None / V2V の2択**（A2V は Generate タブの上記アコーディオンへ移設済み）。
- Quality mode 直下に **Preset** ドロップダウン。選択すると解像度・crop と各クリップの推奨フレーム数（解像度別の快適上限）を全スロットへ一括自動入力する。
- width / height の `minimum` 撤去・JS での `min` 属性付与、および「生成中はボタンをグレーアウト」は Generate タブと同じ仕組みを共有する。

**Settings タブ**: 言語/テーマ・接続情報・ポーリング設定・サーバー config ビューアに加え、**Models** セクション（カテゴリ別ドロップダウン＋Load。`models\ltx-2.3-gguf` 直下に GGUF を置くと自動認識される旨のフォルダ案内つき。`default` 選択肢は実ファイル名を併記した `default — <ファイル名>` 表示）と **Danger zone**（Unload 等・チェックボックスで解錠）を持つ。あわせて **Acceleration（生成の高速化）** 区画があり、**5項目すべて実装済み**の切替——Fused GGUF Dequantization Kernel（GGUF 逆量子化の1カーネル化）／attention（`sdpa`・`sage`）／Block-swap prefetch（先読み block swap）／モデル骨格の常駐（`keep_resident`）／VAE（Default・PrunaVAED。**2026-08-05 に実装**。既定は Default で恒久的に反転しない）——を並べる。**常時グレーアウトのプレースホルダは 2026-08-05 をもって1件も無くなった**。**値はジョブ単位でリクエストに載る**（サーバーの再起動もパイプラインの再読み込みも要らない。フィールドは §6.2 を参照）。

**共通**: 上段バーの server status は `GET /api/v1/status` を叩き、GPU 名・空き VRAM・**low_vram_mode / profile** を表示。ジョブ進捗を 1 秒間隔でポーリングし、完了後に mp4 を取得してプレビュー表示。distilled は **8 steps / CFG=1.0** 固定で送る。

---

## §12b MCPサーバー（AIエージェント連携）

**MCP（Model Context Protocol。AIエージェントが外部ツールを呼び出すための標準規格）** サーバーを `mcp_server/` パッケージとして新設した（2026-07-28・コミット `2f8e85b`）。Claude Code などの MCP クライアントから、Gradio UI（§12）と同等の操作をツール呼び出しとして行える。

### 12b.1 位置づけ（アーキテクチャへの影響なし）

`mcp_server/` は Gradio UI と同じ**クライアント層**に属する。§4 の「2プロセス・2venv」構成そのものは変わらない——MCPサーバーは `./.venv`（torch 無し）の中で動く追加のプロセスで、既存の `/api/v1/*` を `httpx` 経由で叩くだけであり、バックエンド自身は起動しない（起動確認は `backend_status` ツールが行う）。凍結 API 契約（§6）への変更も無い。

### 12b.2 ツールの概要

**22個のツール**を6カテゴリ（system 6 / uploads 3 / generate 2 / jobs 7 / outputs 3 / batch 1）で公開する。全ツールは `async def` で実装され、ブロッキングI/O（ファイルコピー・wav走査等）は `anyio.to_thread.run_sync` で逃がす（MCP SDK 1.28 は同期ツールをイベントループ上で直接呼ぶため）。生成物は base64 埋め込みではなく常にローカル絶対パスで返す。ツール名の完全な一覧・1行説明・典型ワークフロー（T2V/I2V/A2Vバッチ）は `README.md` §8 を参照（本書では重複させない）。設計判断（依存を main 化した理由・非同期化の根拠・エラー封筒の翻訳規則・`.mcp.json` 絶対パス生成方式 等、D1〜D11）は `Docs/MCP_SERVER_DESIGN.md` が正本。

### 12b.3 依存とセットアップ導線

`mcp>=1.28,<2` は `pyproject.toml` / `requirements.txt` の**main依存**として追加した（`dev` extra ではない。エンドユーザーの `uv sync` にそのまま含まれる、§2.5/§4.5）。`scripts/setup.ps1` がセットアップ完了時に、`.venv\Scripts\python.exe` への絶対パスを埋め込んだ `.mcp.json` をリポジトリ直下へ自動生成する（マシン固有パスのため `.gitignore` 済み・git 追跡外）。

### 12b.4 テストと実機ゲート

テストは `tests/test_mcp_*.py`（7ファイル）が `./.venv` で完結し、GPU・実バックエンドプロセスを必要としない（`httpx.ASGITransport` でモック FastAPI アプリに直結）。機械検証・実機ゲート（承認フロー・22ツール表示・T2V/I2V/A2Vバッチ等の実操作）の実施記録は `Docs/VERIFICATION_LOG.md` §39 が正本（本書執筆時点でのゲート状況もそちらを参照。本書では都度古くなる実施状況を複製しない）。

---

## §13 開発フェーズとロードマップ

> 本章は `Docs/NEXT_SESSION_HANDOFF.md`「設計対話の決定」「フェーズ別ロードマップ」の **early-integration 是正**を反映する。旧 v04 §16 の Phase 2〜5 枠は、以下の削除・再分類・格上げで置き換わる。

### 13.1 方針: early-integration

「機能を固めてから統合」ではなく、**「早く統合して、使いながら育てる」**（§1.3）。Phase 1 の最小バックエンドが 16GB で通った時点で、次のゴールは機能拡張ではなく **AviUtl2 統合（Phase 2）** である。

### 13.2 Phase 1 — 最小バックエンド（**実装済み**）

Phase 1 ＝ 凍結 REST API を持つ最小バックエンド。以下は **done**（実機検証済み。詳細は `Docs/VERIFICATION_LOG.md`）。

- 凍結 REST API（÷64 解像度・8n+1 フレーム・T2V/最小I2V・`GET /status` の `vram_optimization` 契約・`metadata.json` スキーマ・limits/presets）。
- 単一ジョブ管理（in-memory・実行中の新規 `POST /generate` は 409）。
- **Low VRAM baseline + 16GB fit**（下記「16GB fit の意味」）。
- **720p 級**（1280×768 二段生成 → 任意で 1280×720 クロップ）の T2V/最小I2V。
- **native 音声**（joint 生成・AAC mux・crop での音声保持）。
- **マルチジョブ**（単一ユーザーの逐次連続生成。I2V＋音声の連続も両経路 PASS）。
- ~28GB モデル構成（GGUF transformer + GGUF Gemma + component ファイル + upscaler + tokenizer dir）。
- mock backend による GPU 無し pytest（§16）。

> **「16GB fit」の意味**: ＝**ハード OOM しない**（溢れても system RAM へページングで完走）の意。「全工程が dedicated 16GB 内」は別軸（denoise stage2 の解像度×尺スケーリング）であり、`Docs/RESOLUTION_DURATION_CAPABILITY.md` の spill-free 閾値（付録A）に従う。

**再分類（旧 Phase 2 → done）**: block-swap / VAE tiling / `--te-offload`（§9） / `--dit-cpu-load`（§9） / component-files といった低VRAM 高度化は、16GB fit のため**既に Phase 1 で実装済み**。旧 v04 が「Phase 2＝低VRAM 高度化」としていたものの実体はこれで、**done** に再分類する。

### 13.3 Phase 2 — AviUtl2 拡張機能 統合（**ゴール・早期統合**）

**このプロジェクトの最終目的**（§14）。汎用 REST API は既に安定しているため、機能を足す前に**まず AviUtl2 拡張から本 API を叩いて「動くツール」を得る**。統合しながら Phase 3 以降で育てる。

### 13.4 Phase 3 — LTX-Desktop 生成パリティ（育てる）

**Phase 3 の北極星＝公式 [LTX-Desktop](https://github.com/Lightricks/LTX-Desktop)（Lightricks・Apache-2.0・LTX-2.3 と同時リリース）の「AI 生成機能」パリティ。** 動画編集・エンコード・タイムライン配置は AviUtl2 が担うので、拡張機能側は **LTX-Desktop が持つ生成系機能をすべて出せる**ことを目標にする（編集系は対象外）。我々のバックエンドは元々 LTX-Desktop の低VRAM フォーク（*Kandyman-iac* fork）由来で、アーキも同型（FastAPI backend ＋ 別フロント）＝**フロントを AviUtl2 拡張に差し替え、バックエンドの露出を LTX-Desktop に揃える**構図。

> **重要な調査結論（2026-07-02・`Docs/NEXT_SESSION_WORKORDER.md` に詳細）**: LTX-Desktop の生成機能の**大半は下層（我々の `engine`／凍結 wheel `ltx_core`/`ltx_pipelines`）が既に対応済み**で、露出を塞いでいるのは**我々の Phase 1 凍結 API だけ**。キーフレーム／first+last／任意 frame_idx／複数条件／strength は wheel の `VideoConditionByKeyframeIndex`/`VideoConditionByLatentIndex`/`VideoConditionByReferenceLatent` が既にサポートし、`combined_image_conditionings` が frame_idx で自動振り分けする。

**Phase 3 の作業（優先順）:**

1. **凍結 API の「解凍」＝条件付けの露出（★次セッションの主作業・低リスク）**: `api/models.py` の 2 検証（`len(conditioning_images) ≤ 1`／`frame_idx == 0` 強制）と `services/ltx_runner.py` の frame_idx ハードコードを緩和し、**多キーフレーム・first+last ブックエンド・任意 frame_idx・複数条件・per-item strength** を露出する。これで長尺化の土台（クリップ連結・キーフレーム制御）が現物化する。engine 内で完結する見込み（新パイプライン不要）。制約＝frame_idx は8の倍数／`[0, num_frames)`・8n+1 フレーム不変・latent 1枚≒pixel 8枚。**要確認事項＝二段パイプラインで条件付けが Stage 1/Stage 2 双方に効くか**（LTX-2 は upscale 段で再注入しないと詳細が失われると報告）。契約変更なので `GET /status`・limits・`metadata.json`・mock pytest も併せて更新。
2. **クリップ連結（生成プリミティブ）**: 上記 API の上で、ブックエンド I2V（前クリップ終了＝次クリップ開始の共有境界）と自己回帰 extend を提供。タイムライン配置は AviUtl2 側。プロンプトは「グローバル基底＋クリップ毎 override」を UX 指針とする（text-only プロンプト伝播は可）。
3. **Gap Fill ／ Retake（大規模・後続セッション）**: LTX-Desktop の連続性プリミティブ。Gap Fill＝近傍フレーム条件の間埋め、Retake＝`TemporalRegionMask` による領域再生成（`RetakePipeline`）。**規模が大きいので次セッションでは着手しない。**
4. **その他パリティ項目（段階的）**: 生成キュー（逐次・cancel）／延長尺（〜30s）／プロンプト強化（**text-only 版のみ**＝T2V 用）／STG・sigma schedule・denoise loop・negative・seed lock 等の露出／空間アップスケーラのユーザー操作露出／LoRA 再導入（de-fork で削除済のため）／attention tiling 再導入。

> **→ 進捗追記（2026-07-11 時点）**: 上記 1（多キーフレーム・任意 frame_idx・複数条件・per-item strength の露出）と 2（クリップ連結＝`POST /generate/chain`）は**実装済み**。4 のうち「LoRA 再導入」も**実装済み**（IC-LoRA＋画風/キャラクターのスタイル LoRA。`GET /loras`／`<lora:...>` 記法／チェーンへの `loras` 加算＝VERIFICATION_LOG §32）。本節は起票当時のロードマップ記録として残す。
>
> **→ 進捗追記（2026-08-10 時点）**: 3のRetakeは2026-08-09〜10に実装済み（`Docs/VERIFICATION_LOG.md` §55系）。Gap Fillは未着手。

### 13.4b Phase 4 — 高度な条件付け（IC-LoRA / V2V）

**LTX-Desktop 自身が UI で提供していない**（モデルは可能だがアプリ未提供・フォークも roadmap 止まり）ため、生成パリティの対象外として **Phase 4 に切り出す**。

- **IC-LoRA**（Union Control / Motion Track / Pose / Camera / Detailer / HDR / Lip-Dub 等）
- **V2V**（`ICLoraPipeline` 経由）
- **audio-to-video（A2Vid）／時間アップスケーラ**等の別パイプライン系も、必要になった時点で Phase 4 で検討。

> **→ 進捗追記（2026-07-11 時点）**: 本節の主要項目は**実装済み**＝IC-LoRA（canny／pose 等の control 系＋strength 可変・VERIFICATION_LOG §21/§28）・V2V（`source_video` によるチェーン継続生成・§24）・audio-to-video（`source_audio`・§25）。さらに A2V＋LoRA の併用も 2026-07-11 に解禁した（`GenerateChainRequest.loras`・§32。参照動画を要する control 系のみ `LORA_CONTROL_UNSUPPORTED_IN_CHAIN` で拒否）。時間アップスケーラは未実装のまま。本節は起票当時のフェーズ分類の記録として残す。
>
> **→ 追加進捗（2026-07-11・α版・VERIFICATION_LOG §34）**: 上記「参照動画を要する control 系のみ拒否」の制約は、**`clips` がちょうど1つのチェーン（A2V を含む）に限り**解禁された。`GenerateChainRequest` に単発 `GenerateRequest` と同型・同バリデーションの3フィールド（`reference_video_id`／`conditioning_attention_strength`／`reference_video_strength`、いずれも optional）を加算し、`clips` が1のときだけ受理する（2以上のチェーンは従来どおり `422 LORA_CONTROL_UNSUPPORTED_IN_CHAIN`）。意味論は単発生成と同じ「reference latent は stage 1 のクリップ0にのみ注入し、stage 2 のタイルには注入しない」＝チェーン全体（stage1+stage2の全区間）へ一様に効く点は変わらない。新設の 422 群: `REFERENCE_RESOLUTION_INVALID`（幅・高さが128の倍数でない）／`LORA_REQUIRES_REFERENCE`（control系＋clips=1＋参照動画無し）。なお `LORA_PREPROCESS_CONFLICT`（preprocess 種別が2種以上混在）は**既存の 400** であって本群には含まれない。`reference_video_id` は `source_video`（V2V 継続）と排他（422）。既知の制約: pydantic のスキーマ検証がエンドポイントより先に走るため `clips>=2` ＋ `reference_video_id` の複合誤設定はコード付きでない汎用 `VALIDATION_ERROR` になる。GPU 実機の目視ゲート（単発生成との一致確認・A2V 音声との共存確認）は次セッションへ持ち越し。

### 13.4c 将来課題（現行の開発計画からは除外）

- **VLM（vision）再導入**: `enhance_i2v`（入力画像を見たプロンプト補強）や「近傍フレームを見た Gap Fill プロンプト提案」は Gemma の vision が必要。我々は QAT 回収で **text-only Gemma 化**（vision 除去・22.7GB 削減、`Docs/VERIFICATION_LOG.md §14`）しており、vision 再導入はこの最適化を巻き戻す。**今回の開発計画からは外す**。将来これらの機能が要件化したときに、別途 vision を再導入する判断を行う（＝本項は将来課題としての記録）。text-only で可能なプロンプト強化（T2V 用の言い換え）は Phase 3 の範囲内。

### 13.5 削除スコープ（旧 v04 から外すもの・理由付き）

| 旧要素 | 判断 | 理由 |
|--------|------|------|
| **1080p アップスケール「機能」**（旧 Phase 4） | **削除** | ユーザーが外部ツールで行う前提を、当時のエージェントが「本システムに AI upscale を組み込む」と誤解して機能化していた。1080p を LTX 内部の二段目で出すのは 16GB の壁であり、**外部の専用アップスケーラ推奨**（`Docs/LTX23_REFERENCE.md` §5/§6）。**※注意**: 内部の二段 upsampler（spatial upscaler x2）は distilled 生成そのものの仕組みなので**残す**。削除するのは「機能として提供する 1080p アップスケール」だけ。 |
| **多人数インフラ**（本格ジョブキュー / 認証必須 / インターネット公開 / 永続 DB） | **削除** | **単一ユーザー想定**のため不要。現状の「1ジョブ＋busy 409」が正しい設計。`--api-key`（§3）は任意の付加であって必須ではない。 |

---

## §14 AviUtl2 / DaVinci Resolve 連携（Phase 2 = ゴール）

AviUtl2 拡張機能との連携は**本プロジェクトの最終目的**であり、early-integration 方針により **Phase 2＝ゴール**へ格上げする（旧 v04 では Phase 5 相当）。バックエンド API が T2V・最小I2V・ジョブ管理・出力保存まで安定した今、統合に着手できる。

### 14.1 役割分担

フロントエンド（AviUtl2 拡張 / DaVinci スクリプト）は**薄いクライアント**とする。

**担当すること**: プロンプト等の入力 UI／必要に応じた画像アップロード（`POST /api/v1/upload/image`）／`POST /api/v1/generate`／`GET /api/v1/jobs/{job_id}` のポーリング／完了 MP4 の取得／タイムラインへの配置。

**担当しないこと**: LTX モデルロード／GPU 管理／低VRAM 最適化／動画生成処理／ffmpeg エンコード／複雑な I2V/V2V 制御の直接実装（すべてバックエンド側）。

### 14.2 通信フロー

```text
[AviUtl2 拡張 / DaVinci スクリプト]
    │
    ├── （任意）POST /api/v1/upload/image  → image_id 取得
    ├── POST /api/v1/generate              → job_id 取得
    ├── GET  /api/v1/jobs/{job_id}         → 1秒間隔でポーリング
    ├── GET  /api/v1/jobs/{job_id}/video   → MP4 取得
    └── MP4 をタイムライン / Media Pool へ配置
```

### 14.3 実装言語候補

| 言語 | 方式 | HTTP | 備考 |
|------|------|------|------|
| C++ | `.aux2` DLL | WinHTTP / libcurl | AviUtl2 SDK に素直。開発コスト高め |
| C# | Native AOT DLL | HttpClient | HTTP 実装が容易。AviUtl2 との接続検証が必要 |

### 14.4 DaVinci Resolve 等への横展開

バックエンドは AviUtl2 専用にしない。DaVinci Resolve 等は同じ REST API を叩く薄いクライアントとして実装する。DaVinci の最初の MVP は Python/Lua Scripting API で「localhost のバックエンドへ生成リクエスト → 完了までポーリング → 完成 MP4 を Media Pool へ追加 →（任意で）Timeline へ Append」のみを行う。OpenFX/OFX プラグイン化は初期 MVP の範囲外。

---

## §15 ログ・エラーハンドリング

### 15.1 ログ出力

アプリログは `logs/server.log`、実エンジン worker のログは `logs/ltx_worker.log` に出力する（`config.server.log_dir` 既定 `./logs`）。最低限、次を記録する。

- サーバー起動時刻 / host / port / listen mode
- GPU 名・VRAM 総量（GPU 情報）
- パイプラインのロード開始/完了/失敗
- ジョブ ID / 入力解像度・num_frames・seed
- 生成の開始/完了/失敗・生成時間
- **peak VRAM**（`ltx_worker.log` の `GENERATED_OK peak_vram_mb=`）
- Low VRAM 設定
- OOM 発生時の例外

### 15.2 エラー形式

独自エラーは次のエンベロープで返す（`main.register_exception_handlers`）。

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

- `code`（機械可読の識別子）/ `message`（人向け説明）は必須。`job_id`（該当ジョブ）/ `detail`（補足）は文脈に応じて付く。
- Pydantic のリクエスト検証失敗は FastAPI の 422 を、同じ `error` エンベロープ形（`code="VALIDATION_ERROR"` + `detail` に loc/msg/type のリスト）でラップして返す。

### 15.3 エラーコード

エラーコードと HTTP ステータスの**表は §6.8 が正本**（重複させない）。本章では運用面のみ補足する。

- `JOB_BUSY`（409）: 実行中ジョブがある状態での新規 `POST /generate`。単一ジョブ設計（§13.2）の正常な busy 応答。
- `GPU_OOM`（503）: 16GB を超える設定での OOM。UI 側は解像度/フレーム数を下げる案内を出す（spill-free 閾値は 付録A / `Docs/RESOLUTION_DURATION_CAPABILITY.md`）。
- `IMAGE_NOT_FOUND`（404）: `conditioning_images` の `image_id` が未アップロード。

---

## §16 受け入れテスト

受け入れは **(A) 16GB 実機での生成順序**と **(B) GPU 無しの mock pytest** の 2 経路。

### 16.1 16GB 実機の生成順序

`config.yaml` の `generation_presets` 実値に沿って、軽い順に確認する。

| 手順 | preset / 内容 | width×height | num_frames | crop_output | 種別 |
|------|---------------|--------------|-----------|-------------|------|
| 1 | `smoke_test` | 384×256 | 17 | なし | T2V |
| 2 | `minimal` | 512×320 | 49 | なし | T2V |
| 3 | I2V（`minimal` + 画像1枚） | 512×320 | 49 | なし | 最小I2V（`frame_idx=0`） |
| 4 | `small` | 960×576 | 121 | **960×540** | T2V（+ 任意で I2V） |

期待される具体的な生成秒数・peak_vram は `Docs/RESOLUTION_DURATION_CAPABILITY.md` および `Docs/VERIFICATION_LOG.md` が正本（本書は代表値のみ）。手動 API 例は `README.md` §5 を参照。720p は 1280×768 生成 → `crop_output={1280×720}` で確認する。

### 16.2 mock pytest（GPU 無し）

`./.venv`（torch 無し）で pytest を実行する。`tests/conftest.py` が `model.backend="mock"` を強制するため、GPU/モデル無しで API・ジョブ管理・Gradio・スキーマまで疎通する。

```powershell
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
.\.venv\Scripts\python.exe -m pytest -q
```

検証観点（`tests/test_validation.py` / `tests/test_smoke.py`）:

- **÷64**: width/height が 64 の倍数でないと 422（例 544=÷32 だが÷64 でない → reject）。
- **8n+1**: num_frames が 8n+1 でないと 422。**cap=481**（=8×60+1=20s@24fps）を超えると 422。481 は許容、480（8n+1 でない）は 422。
- **conditioning ≤ 1**: `conditioning_images` が 2 枚以上で 422。
- **frame_idx=0**: conditioning の `frame_idx` が 0 以外で 422。
- **distilled 固定値**: `num_inference_steps=8` 以外で 422。
- **生成疎通**: mock runner で T2V/I2V が完了し、`metadata.json` の `generation_mode` / `seed_used` / `vram_optimization.low_vram_mode` / `peak_vram_mb` が入る。
- **busy 409**: 実行中ジョブがある状態で新規 generate は 409（`JOB_BUSY`）。
- **`GET /status`**: `vram_optimization` の `low_vram_mode=true` / `low_vram_profile="16gb_safe"` 契約。

---

## §17 ライセンスと provenance

- **本プロジェクト**: **Apache-2.0 を推奨**（上流コード利用時の整合性を優先）。
- **`engine/`**: LTX-2（`ltx_core` / `ltx_pipelines`）および LTX-Desktop 由来の派生コードを含む。由来・依存再現手順・帰属の一次情報は **`engine/VENDOR_NOTICE.md`**。再配布時は上流の帰属表示を保持すること。
- **モデルウェイト**: **LTX-2 Community License** に従う。

---

## 付録A LTX 制約早見表

| 制約 | 値 | 由来 / ポインタ |
|------|----|------------------|
| 生成サイズ（width/height） | **64 の倍数** | two-stage distilled が Stage1 を半解像度で生成し x2 アップサンプルするため÷64（÷32 由来は `Docs/LTX23_REFERENCE.md` §3）。検証は §6.7・`api/models.py` |
| フレーム数 | **8n+1**（9, 17, 25, … 481） | 上限 cap=**481=20s@24fps**（旧 257 から緩和）。§6.7 |
| distilled ステップ / CFG | **8 steps / CFG=1.0** 固定 | §6.7 |
| 最小I2V | **画像1枚・`frame_idx=0` 固定**（画像なし=T2V） | §6.7 |
| 非64 表示サイズ | **`crop_output` で中央クロップ**（例 1280×768→720, 960×576→540） | §8.2 |
| 解像度別 spill-free フレーム（16GB 実測・代表値） | 720p(1280×768):**257** / 1080p(1920×1088):**153** / 1440p(2560×1472):**81** | 超えると shared へ溢れ ~2-4x 低速（OOM せず）。`GET /api/v1/config` の `limits.spill_free_frames`。正本＝`Docs/RESOLUTION_DURATION_CAPABILITY.md` §8.4/§8.6 |
| 解像度×尺の実用上限 | 解像度別に §10 / §6.7 を参照 | 1080p 長尺は非実用（~40分・commit リスク）→ **720p 生成 + 外部 upscale 推奨** |

---

## 付録B 用語・SSOT ドキュメント地図

### B.1 用語集

| 用語 | 1行定義 |
|------|---------|
| **T2V** | Text-to-Video。プロンプトのみから動画生成（画像なし）。 |
| **I2V（最小I2V）** | Image-to-Video。開始フレーム1枚（`frame_idx=0`）を条件に動画生成。Phase 1 は 1 枚・開始フレームのみ。 |
| **distilled** | LTX-2.3 の蒸留パイプライン。8 steps / CFG=1.0 固定、内部 two-stage（Stage1 半解像度8step → x2 upscale → Stage2 4step）。 |
| **two-stage / spatial upsampler** | distilled が最終解像度を作るための内部2段。x2 spatial upscaler（別チェックポイント）を使う。この二段目は生成の仕組みであり「1080p 機能」とは別（§13.5）。 |
| **block-swap** | transformer のブロックを GPU 常駐（既定 8）とし残りを退避、重み VRAM を削る手法。 |
| **te-offload**（`--te-offload`） | Gemma text-encoder を逐次 per-layer で CPU オフロードし encode ピーク VRAM を下げる（既定 ON・§9）。 |
| **dit-cpu-load**（`--dit-cpu-load`） | DiT(transformer) を CPU で構築しブロックのみ GPU へストリーム。ロード時の ~16.9GB GPU スパイクを除去（既定 ON・§9）。 |
| **component-files** | VAE / audio / text-projection を 46GB モノリスでなく小単体 safetensors から読む経路（`use_component_files: true`）。マルチジョブの commit 枯渇を防ぐ。 |
| **43GB モノリス / 46GB モノリス** | **同一の 1 ファイル** `ltx-2.3-22b-distilled-1.1.safetensors`（実測 46,139,885,414 B＝46.1GB＝42.97GiB）。本書は箇所により両方の表記を使うが指すものは同じで、どちらも誤りではない（§5.2 の表記注記・`README.md` §1 の削除済みブロック注記）。物理削除済で、旧参照パス`checkpoint_path`も2026-07-28に`config.model`から削除済み（現在は`services/ltx_runner.py`がworker payloadへ直値`""`をハードコード）。 |
| **spill / spill-free** | 生成が dedicated 16GB を超えて system RAM（shared）へ溢れること。溢れると ~2-4x 低速化（OOM はしない）。溢れない上限が spill-free frames。 |
| **commit** | Windows の仮想メモリ予約（物理 RAM + ページファイル）。ディスク使用量ではない。連続生成で枯渇すると native crash しうる（component-files で束縛）。 |
| **GGUF / Q4_K_M** | 量子化重みフォーマット。本番の transformer / Gemma とも GGUF Q4_K_M。 |
| **mock backend** | GPU/モデル無しで合成クリップを返す backend。API/スキーマ/出力構造は real と同一で、テスト・GPU 無し開発に使う。 |
| **MCP（Model Context Protocol）** | AIエージェントが外部ツールを呼び出すための標準規格。本プロジェクトは `mcp_server/` パッケージで22個のツールを公開する（§12b）。 |

### B.2 SSOT ドキュメント地図

§0.3 の SSOT 地図と整合。どの Docs が何の正本かは §0.3 の表を参照（本書 §6＝凍結 API 契約 / `config.yaml`＝設定実値 / `VERIFICATION_LOG`＝実測・検証 / `RESOLUTION_DURATION_CAPABILITY`＝解像度×尺の能力 / `NEXT_SESSION_HANDOFF`＝ゴール・ロードマップ / `LTX23_REFERENCE`＝LTX 一般知識 / `engine/VENDOR_NOTICE`＝provenance / `MCP_SERVER_DESIGN`＝MCPサーバーの設計判断（§12b）/ `ICLORA_DEPTH_DEBLUR_WORKORDER`＝IC-LoRA Depth・Deblur の仕様・設計判断 / `ACCELERATION_RESEARCH_NOTES`＝生成高速化の候補整理と採否判断 / `README`＝起動・導線）。
