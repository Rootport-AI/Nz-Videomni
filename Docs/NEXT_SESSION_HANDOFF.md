# 次セッション引き継ぎ書 — LTX 2.3 を 16GB で動かす

---

## ▶▶▶▶▶▶▶ 最新ステータス（2026-07-03 セッション末・**次セッションはこのブロックだけ読めば現在地が分かる**）

> **本ブロックが最新の正本。** これ以降の▶節（2026-07-03昼以前・2026-07-01以前）はすべて歴史記録。食い違ったら本ブロックが正。

### 現在地（1分サマリ）

| 項目 | 状態 |
|---|---|
| リポジトリ | main＝`a578c83`（IC-LoRA Phase Aマージ済・**ローカル先行＝push未**）／branch `feature/ic-lora-phase-b`＝Phase B一式（engine機構+API+docs）で main から先行・**全ゲートPASS済み・マージ判断待ち** |
| IC-LoRA Phase A | ✅完了・mainマージ済（スパイク＝bf16融合経路。正本=[`IC_LORA_PHASE_A_STATUS.md`](IC_LORA_PHASE_A_STATUS.md)・VERIFICATION_LOG §20） |
| IC-LoRA Phase B | ✅完了（**forward時GPU LoRA適用**＝per-layer-quant本番経路・VRAM増ゼロ・ジョブ毎切替可＋**API露出**＝`loras`/`reference_video_id`/`POST /upload/video`。G1〜G5全PASS・基準SHA=`735a6de9…272`。正本=[`IC_LORA_PHASE_B_STATUS.md`](IC_LORA_PHASE_B_STATUS.md)・VERIFICATION_LOG §21） |
| Phase 3 スライス1（キーフレーム誘導） | ✅完了・main入り済・**目視受容済み**（挙動=「途中キーフレームは磁石・間の遷移は自由領域でプロンプト支配」をユーザーが仕様として受容 2026-07-03。VERIFICATION_LOG §17.9-17.10） |
| Phase 3 スライス2（クリップ連結） | ✅完了・main入り済・**目視/試聴全PASS**（720p級2セグでは継ぎ目不可視まで実証。継ぎ目以外のbacklog6件=非ブロッキング。正本=[`PHASE3_CLIP_CONCAT_STATUS.md`](PHASE3_CLIP_CONCAT_STATUS.md)） |
| 目視ゲート | **全クローズ**（低解像度6本＋高解像度3本・2026-07-03。成果物一覧=`outputs/visual_review/README.md`） |
| ユーザー向け機能解説 | [`FEATURE_GUIDE_KEYFRAMES_AND_ICLORA.md`](FEATURE_GUIDE_KEYFRAMES_AND_ICLORA.md)（キーフレーム誘導とIC-LoRA 2系統の平易な解説） |

### 次セッションの入口（この順で）

1. **ユーザーに2点確認**: ①main の push（`git push origin main`・監督のpushは権限拒否されるためユーザー実施） ②`feature/ic-lora-phase-b` の main マージ。
2. **Phase C スコープ確定**: **最優先候補（ユーザー決定 2026-07-03）＝IC-LoRA制御系アダプタ（Pose/Union）対応＋DWPose等の外部プリプロセッサ段の新設**（「動き=完全トレース・内容=置換」の実現。現実装は参照系=生動画入力のみ・DWPose等は不存在とコード確認済み）。他の候補=VERIFICATION_LOG §21.8（keep=1トグル検証／oracle照合／x4登録／denoise+20-26%最適化／バリデーション緩和／Gradio UI露出）。
3. 入口ドキュメント: [`IC_LORA_PHASE_B_STATUS.md`](IC_LORA_PHASE_B_STATUS.md)（アーキテクチャ要点「勝手に最適化しない」注意あり）→ VERIFICATION_LOG §21。

### やらないこと（スコープ外・混同注意）

- **プロンプトの練り込み＝リリース後のユーザー作業であり開発スコープ外**（ユーザー指示 2026-07-03）。開発中の生成は検証用の定型プロンプト（下記運用ルール）を使うだけでよい。プロンプト研究のタスク化・実験はしない。
- Gap Fill の前倒し＝しない（multikey の遷移挙動はユーザー受容済み）。Gap Fill／Retake は Phase 2（AviUtl2統合）後に要件逆算。
- Phase 2（AviUtl2統合）＝ユーザーのプラグイン開発環境整備待ちで**ブロック中**。
- 1080pアップスケール機能・多人数インフラ＝削除済みスコープ（spec §13.5）。

### 運用ルール（最新の正・次セッションも適用）

- **目視検証**: 720p級（1280×768）以上＋**映画トレイラー風プロンプト**＋「賑やかな町＋セリフ」題材（512×320級は顔溶けで判断不能・「CM風」は廃止・波/静的部屋はNG）。
- **人間向けの説明**: プロジェクト内部の略語・造語禁止。ただし専門用語の過剰な言い換えも逆効果＝**「普通のまともな日本語」**で書く。判断を仰ぐ前に機能説明を届ける。解説文書はOpusサブエージェントに執筆させ、監督はレビュー。
- **サブエージェント**: Opus以下を使用（**Fable5禁止**）。長時間GPU実験は「完了待ちで停止」しがち＝**能動ポーリング監視を指示**する。非破壊原則・異常時は続行せず報告。
- 客観PASS（byte-match・pytest・メトリクス）とユーザー目視ゲートを混同しない。実験前に仮説→Web/コードで裏取り。
- **SHA照合の注意**: IC-LoRA付き出力の基準SHA=`735a6de9…272`。旧`outputs/ic_lora_phaseA/spike.mp4`（`8e10aa59…`）は旧コードの出力＝照合に使わない。

### 2026-07-03 セッションの主な記録ポインタ（詳細は各正本・ここには書かない）

- Phase B設計根拠・リサーチ＝[`IC_LORA_PHASE_B_WORKORDER.md`](IC_LORA_PHASE_B_WORKORDER.md)／ゲート数値＝VERIFICATION_LOG §21。
- 目視結果（低解像度6本＋高解像度3本・受容判断の経緯・「遷移はプロンプト支配」の分析）＝VERIFICATION_LOG §17.9-17.10・各STATUS doc。
- クリップ連結の新規backlog 3件（背景歪み/ワイプ/看板・継ぎ目以外・720p短尺では再発せず）＝[`PHASE3_CLIP_CONCAT_STATUS.md`](PHASE3_CLIP_CONCAT_STATUS.md) backlog#4-6。
- IC-LoRAアップスケールの人物同一性＝参照解像度が主レバー（640×384参照で「同じ人種の別の役者」程度まで改善・ユーザーはLTX 2.3の性能限界=仕様として受容）＝[`IC_LORA_PHASE_B_STATUS.md`](IC_LORA_PHASE_B_STATUS.md)。

---

## ▶▶▶▶▶▶ 現状ステータス（2026-07-01・**次セッションはまずここを読む**）

> **この節が最新の正本サマリ。** 以下の各▶節は古い順に温存した歴史記録なので、食い違ったら**本節が正**。
> **次の一手は下の「次の一手メニュー」から、担当者＋ユーザーで優先順位を議論して決める前提**（規定の順序は無い）。

### いま完了していること（全体像）
- **de-fork リファクタ済**: エンジンは first-party の **`engine/`** パッケージ（旧・同梱フォークは削除・上流 `vendor/LTX-2` のみ温存）。
  起動＝`python -m engine.worker`（別プロセス・別venv）。詳細＝§13 / `engine/VENDOR_NOTICE.md`。
- **720p 達成**: 1280×768 二段を本番 API 経路で完走→任意で 1280×720 クロップ（~167–171秒 / RTX 4070 Ti SUPER 16GB）。連続
  マルチジョブも commit 枯渇せず PASS（レシピ＝comp=1 / keep_resident=0 / bs=8 / vae 512-64）。詳細＝§10。
- **load/encode の 16GB fit 済**: `--te-offload`（Gemma encode ピーク低減・既定 ON, §11）＋ `--dit-cpu-load`（transformer load
  スパイク除去・既定 ON, §12）で、512×320 の whole-job ceiling 16,944→~9.2GB。残るのは den2 の解像度・尺スケーリング軸のみ。
- **音声 Phase1 済**: native joint audio が 16GB で効果音/音楽/発話を生成・AAC mux・crop 音声保持を実機確認（§9.8）。
- **reuse-crash（残課題A）解決済**: BlockSwapService の per-job transformer 保持リークを keep-latest clear＋between-job gc で修正、
  6ジョブ実機 PASS・出力バイト一致（§9.7）。
- **★今回 QAT 回収済（2026-07-01）**: Gemma を **text-only（`Gemma3ForCausalLM`）** 化し、22.7GB の QAT dir を物理削除。
  gemma_root は ~40MB tokenizer-only dir（`models/gemma-3-12b-it-tokenizer/`）。**models/ 50.9GB→28.15GB**。full-QAT baseline と
  **バイト完全一致**（T2V `23844b4e…6bb7bf` / 最小I2V `a511eda4…c217`・3経路）・peak_vram 8440・退行なし。詳細＝**§14** ／
  `Docs/QAT_RECLAMATION_RESEARCH.md` 冒頭 ✅RESOLVED バナー。commit `93696b4`→`694ca54`→main マージ `826e76f`。
- **★今回 I2V マルチジョブ＋音声 検証済（2026-07-02）＝§10.5【最重要】クローズ**: production default（comp=1/keep=0）連続 I2V＋音声を直接ハーネス（I2V×4 @384）＋本番API（I2V×3 @512×320）両経路で PASS（両経路 peak_vram 一致）。詳細＝**VERIFICATION_LOG §10.7**。
- **★今回 install スクリプト全面書き換え（2026-07-02）**: `scripts/install_ltx.ps1` を冪等クリーンインストーラ化（両venv・現行~28GBセットのみDL[リポID暗号確認済]・GpuArch自動判定・PASS/MISSING表・INSTALLED_PATHS再生成）＋`build_xformers.ps1` の CUDA_PATH/.venv-engine/12.8 修正。本機で冪等スキップ実行検証（exit 0・全PASS・smoke 7）。**※要確認**: attention 実装が上記 install 互換メモ③「xformers/flash-attn は足さない（SDPA維持）」と不整合（下記参照）。**本セッションの全変更は未コミット（作業ツリー）**。

### 凍結してある契約・構成（壊さない）
- **凍結 API 契約（不変）**: ÷64 解像度（`api/models.py`）・8n+1 フレーム・T2V/最小I2V・`GET /status` の `vram_optimization`
  （`services/low_vram.py` `_STATUS_KEYS`）・`metadata.json` スキーマ・limits/generation_presets。
- **2プロセス・2venv 構成**: app＝`./.venv`（torch 無し・FastAPI/Gradio/mock backend）／engine＝`./.venv-engine`（torch 2.9.1+cu128＋
  `ltx_core`/`ltx_pipelines`@`00dc53d`＋`gguf`）。両者を同一インタプリタで共存させない。
- **本番 env 要点**: `LTX_KEEP_RESIDENT=0` 既定（keep=1 だと 720p の Gemma 移動で native crash・§10.2）／`use_component_files: true`
  （Path B・46GB モノリス非経由で commit 束縛）／`te_offload_text_encoder`・`dit_cpu_load` 既定 ON。設定は `config.yaml` が正。
- **本番モデル（実行に要る ~28GB）**: GGUF transformer(Q4_K_M ~17GB)＋GGUF Gemma(Q4_K_M ~7.3GB)＋component VAE/audio/projection
  (~3.9GB)＋spatial upsampler(~0.95GB)＋tokenizer-only gemma_root(~40MB)。43GB モノリス・QAT dir は**削除済**。

### リポジトリ状態
- branch `main`。**QAT 回収〜spec 全面改訂（`LTX23_Backend_Specification.md` v0.5）＋Phase 1〜3 チェックリストまで、2026-07-02 にユーザーが commit＆push 済**（`origin/main` 同期済）。
- **未コミット（作業ツリー）**: 本セッションで整理した引き継ぎ文書＝[`NEXT_SESSION_WORKORDER.md`](NEXT_SESSION_WORKORDER.md) 新規＋本書の当該追記。次のコミットで拾う。

---

## ✅ Phase 1〜3 やることリスト（チェックリスト・2026-07-02）

> Phase 構造は spec `LTX23_Backend_Specification.md` §13（early-integration 是正）と本書「設計対話の決定」が正。
> `[x]`＝実装/検証済（done）、`[ ]`＝未了。数値・検証の一次情報は各 §ポインタ（VERIFICATION_LOG / RESOLUTION_DURATION）。

### Phase 1 ＝ 最小バックエンド（T2V＋最小I2V＋音声＋16GB fit＋720p）

**コア（done）**
- [x] 2プロセス・2venv アーキ（app `./.venv` torch無し／engine `./.venv-engine` torch2.9.1+cu128）
- [x] subprocess worker ＋ JSON-lines(`@@LTX@@`) プロトコル
- [x] 凍結 REST API（10 エンドポイント・全 `/api/v1`）
- [x] Pydantic スキーマ＆バリデータ（÷64／8n+1／distilled 8step・CFG1.0／最小I2V frame_idx0／conditioning≤1）
- [x] 単一ジョブ管理（in-memory・実行中は busy 409）
- [x] backend 選択 auto/mock/real（`_real_available`）
- [x] T2V 生成
- [x] 最小 I2V（画像1枚・frame_idx0）
- [x] native joint audio（AAC/48kHz/stereo・crop 後も保持）＝§9.8
- [x] 16GB fit 一式: block-swap(8/48)／GGUF Q4_K_M transformer／GGUF Q4_K_M Gemma／te-offload／dit-cpu-load／VAE tiling／component-files
- [x] 720p 実証（1280×768→crop 1280×720・~167–171s）＝§10
- [x] マルチジョブ連続（comp=1/keep=0・T2V×3 ＆ I2V×3＋音声 両経路 PASS）＝§10.7
- [x] ~28GB モデル構成（GGUF transformer＋GGUF Gemma＋component＋upsampler＋tokenizer-only gemma_root）
- [x] QAT 回収（text-only Gemma・22.7GB dir 削除・byte 一致）＝§14
- [x] de-fork（engine first-party 化）＝§13／43GB モノリス削除（checkpoint_path は reference-only 化）
- [x] mock backend ＋ pytest（GPU 無し疎通）
- [x] install_ltx.ps1 冪等クリーンインストーラ化／build_xformers.ps1 是正
- [x] Gradio 検証 UI 実装（`/ui`）
- [x] **spec 全面改訂（`LTX23_Backend_Specification.md` v0.5）＝旧残(c) 完了**（2026-07-02・README 参照更新・旧 v04 削除・commit/push 済）

**残（Phase 2 前後で片付ける・優先度はユーザー判断）**
> **次セッションで着手する 4 件（Gradio 手動確認を除く d/e/keep/掃除）は詳細ワークオーダー [`NEXT_SESSION_WORKORDER.md`](NEXT_SESSION_WORKORDER.md) を必ず読む**（根本原因・該当コード行・byte-match ゲート・要ユーザー確認点を整理済）。
- [ ] (a) Gradio GUI **手動動作確認**（`/ui` で T2V/I2V/720p/crop トグルの end-to-end 目視）＝小・実 backend＋GPU ※今回スコープ外
- [x] (d) dead-code 整理 ＝ **done（2026-07-02・byte-match PASS）**。`_SkipGemmaLMSDOps`＋defensive strip／`_read_target_vocab_from_header(path)` 未使用引数／`attention_tile_size`・`loras` no-op 署名パラメータ（＋orphan `_ORIG_GEMMA_LM_PREFIX`/`LoraEntry` import）を除去。T2V `23844b4e…`／I2V `a511eda4…` バイト一致・peak_vram 8440・pytest 16 passed。commit `be15887`→`473ac85`→`fa5dd83`→`bc5b3c1`（branch `chore/phase1-residual-cleanup`）。詳細＝`VERIFICATION_LOG §16`・WORKORDER①
- [x] (e) load/encode の一時 shared 溢れ最適化 ＝ **done（ユーザー確認 2026-07-02）**。§11 te-offload／§12 dit-cpu-load で load/encode の一時溢れは実質解消。残る shared 溢れは高トークン denoise stage2＝**解像度×尺の能力限界（バグでない・`RESOLUTION_DURATION_CAPABILITY.md §8` が正本）**であり、本タスクの対象外。
- [x] **keep=1 常駐モードの新設 ＝ 調査完了につき CLOSE（2026-07-02）**＝当初構想（フル GPU 常駐で gen 漸増解消）は 16GB で原理的 non-viable・利得も既存経路（dit-cpu-load/te-offload＋OS RAM キャッシュ）で捕捉済み。既定 keep=0 不変。**詳細＝`VERIFICATION_LOG.md §15`（仮説 H1–H4 検証・コード読解＋Web リサーチ）**。将来 Phase3 の長尺連結で漸増が実害化したら GPU 常駐でなく §15.4「CPU 正本温存＋層ストリーミング」で再着手・WORKORDER③(closed)
- [~] 開発ゴミ掃除 ＝ **一部done・残はユーザー手動キュレーションへ委譲（2026-07-02 決定）**。①`.claude/settings.json` の `Bash(git clean *)` 許可は**削除済**（commit `efb406e`・誤発火リスク断ち）。②`outputs/`（~127MB）は**生成時間・VRAM 溢れの一次情報**でドキュメントの根拠のため**今回は削除せず**、後日ユーザーが手動整理。uploads/・logs/ も同性質（`logs/ltx_worker.log`＝peak_vram 一次ソース [README §216]）で保全。③`__pycache__`(108)/`.pytest_cache` は純粋な再生成物で任意消去可（`find . -type d -name __pycache__ -not -path './.venv*' -not -path './.uv_cache*'` 等）。`.venv`/`.uv_cache` 等の環境本体は掃除対象外・WORKORDER④

### Phase 2 ＝ AviUtl2 拡張機能 統合（ゴール・早期統合）　※未着手

- [ ] 実装言語決定（C++ `.aux2` DLL ／ C# Native AOT DLL）
- [ ] AviUtl2 拡張の HTTP クライアント実装（`/api/v1/*` を叩く薄いクライアント）
- [ ] upload → generate → poll(jobs) → video 取得 の疎通
- [ ] 生成 MP4 のタイムライン配置
- [ ] （前提）Phase 1 (a) Gradio 手動確認で現物を固めてから着手
- [ ] （横展開・任意）DaVinci Resolve スクリプト MVP（localhost 生成 → Media Pool 追加 →（任意）Timeline Append）

### Phase 3 ＝ LTX-Desktop 生成パリティ（2026-07-02 再編・spec §13.4 が正）

> **北極星＝公式 LTX-Desktop（Lightricks）の「AI 生成機能」パリティ**。編集/エンコード/タイムラインは AviUtl2 が担う。調査で判明＝**LTX-Desktop 生成機能の大半は下層（engine/wheel）が既に対応済みで、塞いでいるのは我々の凍結 API だけ**。

- [x] **★スライス1＝凍結 API の「解凍」（条件付け露出）＝実装＋客観検証 PASS・main merge 済（2026-07-02・merge `7f31935`・push 済）。目視品質のみ PENDING**: 多キーフレーム・first+last ブックエンド・任意 frame_idx・複数条件・per-item strength・cap5 を露出。**engine 内で完結**したが「API 表層のみ／engine 不可触」の当初想定は**誤り**で、実際は engine の条件付け経路に**公式ハイブリッド（idx0=置換 / idx>0=guide）を monkeypatch で自前再現**する必要があった（インストール済み wheel に `combined_image_conditionings` が無い・wheel 更新は回避）。frame_idx は公式 `8n+1` latent 格子へスナップ（ComfyUI `LTXVAddGuide` 準拠）。回帰 byte-match（T2V/単一 I2V バイト一致）＋新経路スモーク（bookend 0/41・multikey3 0/17/41 完走）＋VRAM 8440MB（多キーフレームでもデルタ0）全 PASS。**残＝目視品質判断（ユーザー）＝[`PHASE3_KEYFRAME_VISUAL_VERIFICATION.md`](PHASE3_KEYFRAME_VISUAL_VERIFICATION.md)。merge 済ゆえ不足時は追いコミットで調整（strength/グリッド）**。正本＝[`VERIFICATION_LOG.md` §17](VERIFICATION_LOG.md)。commit `1602245`→`5033385`→`d7a56b1`→docs `0854f8d`→merge `7f31935`。num_pixel_frames／reference-video は今回スコープ外（将来）。
  - **⚠️ 2026-07-03 訂正**: 目視は**まだ有効に実施されていない**。監督が提示した `outputs/phase3_multikey_smoke/bookend|multikey3/output.mp4` は自動スモークで**全キーフレームに同一の合成画像**を使い無効。**有効な目視＝`run_visual.py` を異なる実画像で**（→`visual_bookend/`・`visual_multikey/`）。
- [x] **★スライス2＝クリップ連結（生成プリミティブ）＝masked AV-latent 連結で再実装・main merge・push 済（ユーザー最終目視/試聴のみ PENDING・2026-07-03）**: 元 branch `feature/phase3-clip-concat`（commit `760a283`→`9fb7111`→`5e73e47`→`aebcd08`→`c0ed582`→`d359e4a`→`622dd81`（同日中に旧latent-extend方式を全面置換）→`194ce44`→`eb5f7ac`）→ **main へ merge `2cc4cac`**。同日中に旧 `_EXTEND` monkeypatch 機構も撤去（`de587b4`→cleanup merge `1ab5e0c`・byte-match検証済）。旧方式は境界フレーム目視で hard cut・音声断絶が確定していたが、原因（per-clip decodeでcausal VAEリセット再トリガー＋音声独立生成）を踏まえ「1本の連続AV latentを組み立てて1回だけdecode」する構造に作り直し。境界連続性ハーネスを既知hard-cutへ較正済み＋GPU spikeでユーザー都度目視/試聴PASS＋実機Chain A/Bで全映像junction continuous。**残＝Chain A/B本番出力そのものの最終目視/試聴（ユーザー・パス/フレーム番号は冒頭「最新ステータス」表）**。**正本＝[`PHASE3_CLIP_CONCAT_STATUS.md`](PHASE3_CLIP_CONCAT_STATUS.md)**。設計記録＝[`PHASE3_CLIP_CONCAT_DESIGN.md`](PHASE3_CLIP_CONCAT_DESIGN.md)（採用アーキテクチャnote付）・[`PHASE3_CLIP_CONCAT_WORKORDER.md`](PHASE3_CLIP_CONCAT_WORKORDER.md)・[`VERIFICATION_LOG.md` §19](VERIFICATION_LOG.md)（§18=旧方式FAILの記録）。
- [ ] **Gap Fill／Retake**（＝LTX-Desktop の連続性プリミティブ・**大規模ゆえ次セッションでは着手しない**）。Retake=`TemporalRegionMask`/`RetakePipeline`、Gap Fill=近傍条件の間埋め。
- [ ] その他パリティ（段階的）: 生成キュー／延長尺(〜30s)／text-only プロンプト強化／STG・sigma schedule・denoise loop・negative・seed lock 露出／空間アップスケーラのユーザー操作露出／**LoRA・attention tiling 再導入**（de-fork で削除済）。

### Phase 4 ＝ 高度な条件付け（LTX-Desktop 未提供・パリティ対象外）※IC-LoRA Pixel-Spatial-Upscaler x2 = Phase A スパイク PASS 済（branch `feature/ic-lora-phase-a`・冒頭「最新ステータス」ブロック参照）。他アダプタ・本実装は未着手
- [ ] IC-LoRA（Union/Motion Track/Pose/Camera/Detailer/HDR/Lip-Dub）
- [ ] V2V（`ICLoraPipeline` 経由）・audio-to-video（A2Vid）・時間アップスケーラ

### 将来課題（現行計画から除外・spec §13.4c）
- [ ] **VLM(vision) 再導入**＝enhance_i2v・フレームを見た Gap Fill 提案。QAT text-only 化 [[qat-reclamation-textonly-gemma]] を巻き戻すため今回計画外。要件化時に別途判断。text-only プロンプト強化は Phase 3 内。

> **やらない（削除済みスコープ）**: ①1080p アップスケール「機能」（＝外部ツール推奨。ただし内部二段 upsampler は生成の仕組みなので残す）／②多人数インフラ（本格ジョブキュー・認証必須・インターネット公開・永続 DB＝単一ユーザー想定で不要）。詳細は spec §13.5。

---

## ★次セッション着手予定 ＋ 設計対話の決定（2026-07-02・最優先で読む）

> 本セッション終盤の「spec 全面改訂」に向けたユーザー対話の結論。**次セッションはここから。** 下の「フェーズ別ロードマップ」の Phase 2/5 枠は本節で **early-integration に是正**されている（本節が正）。

### 次セッションの着手 ＝ Phase 1 残 4 件（Gradio 手動確認を除く）
> **▶ 2026-07-02 最新更新**: **spec 全面改訂（`LTX23_Backend_Specification.md` v0.5）は完了・push 済**（旧残(c) クローズ）。
> **次セッションの着手＝Phase 1 残の 4 件＝①dead-code 整理 ②shared 溢れ最適化(e) ③keep_resident 恒久化 ④開発ゴミ掃除**（Gradio 手動確認は今回スコープ外）。
> **着手前に必ず [`NEXT_SESSION_WORKORDER.md`](NEXT_SESSION_WORKORDER.md) を読む**（各タスクの根本原因・該当コード行・byte-match ゲート・“誤認防止サマリ”＝特に **(e) は §11/§12 で実質解消済み＝新規実装不要**、③は vendor wheel 凍結で in-place 化に制約・計測前提、を整理済）。
> ↓下記「install＋マルチジョブ」「spec 全面改訂の方針」は完了/背景として温存（歴史記録）。
- **✅ 完了（2026-07-02・commit `3c008c7`・push無し）**:
  - **install スクリプト全面書き換え**: `scripts/install_ltx.ps1` を冪等クリーンインストーラ化（両venv・現行~28GBのみDL[リポID暗号確認]・GpuArch自動判定・PASS/MISSING表・INSTALLED_PATHS再生成）＋`build_xformers.ps1` 是正（CUDA_PATH/.venv-engine/12.8）。本機で冪等スキップ実行 exit0/全PASS・mock smoke 7。
  - **Phase 1 残「マルチジョブ」＝I2V＋音声 連続を両経路で実機 PASS**（直接ハーネス I2V×4 @384＋本番API I2V×3 @512×320・**VERIFICATION_LOG §10.7**）。※マルチジョブ＝単一ユーザー逐次連続（複数人同時ではない＝削除済スコープ）。**Phase 2 クリップ連結の前提クリア。**

### install 互換メモ（調査済 2026-07-02・再調査不要）
- **stack は世代跨ぎで可搬**: `torch 2.9.1+cu128`(stable) が arch_list に **sm_80/86(Ampere)・sm_89(Ada)・sm_100/sm_120(Blackwell)** を同梱／attention＝**SDPA**（xformers/flash-attn 未使用）／GGUF dequant＝**pure-torch**／fp8 本番未使用 → **Ampere/Ada/Blackwell 現 pin のまま動く見込み**（nightly も source build も不要）。ユーザーはビルド済みコンポーネント不要。
- **install に書く gotcha 3点**: ①**torch は必ず cu128 index から**（素の `pip install torch` は CPU/旧CUDA→Blackwell "no kernel image"）②`install_ltx.ps1` は **2026-07-02 に全面書き換え済＝現行の正**（旧記述「stale・torch2.7/cu129/xformers/flash-attn-4」はもう当てはまらない。README §1 手順を自動化＋本メモ③準拠＝SDPA・xformers/flash-attn 自動導入せず・Blackwell は SDPA 固定・xformers は wheel 在れば任意） ③**Blackwell は R570+ ドライバのみ**・**xformers/flash-attn/sageattention は足さない**（SDPA 維持・足すと逆に詰まる）。導通1行: `python -c "import torch;print(torch.cuda.is_available(),torch.cuda.get_arch_list())"`。

### spec 全面改訂の方針（合意済・未着手＝Phase 1 残(c)・spec はバックアップ済ゆえ自由に改訂可）
- **Phase 構造を early-integration に是正**: **Phase 1**(最小バックエンド, ほぼ done) → **Phase 2 ＝ AviUtl2 拡張機能（＝ゴール・まず「動くツール」を得る）** → **Phase 3+ ＝ 育てる**（長尺クリップ連結 → IC-LoRA/V2V/プロンプト強化・希望次第）。方針＝「機能を固めてから統合」でなく「**早く統合して使いながら育てる**」。
- **削除**: ①**1080p アップスケール「機能」**（＝ユーザーが外部ツールで行う想定を当時のエージェントが「本システムに AI upscale を組み込む」と誤解して機能化していた → 削除。**※内部二段 upsampler は生成そのものの仕組みなので残す**）②**多人数インフラ**（同時ジョブキュー/認証/インターネット公開＝**単一ユーザーなので不要**。現状の「1ジョブ＋busy 409」は正しい設計）。
- **再分類**: 旧 Phase 2 の低VRAM最適化（block-swap/VAE tiling/te-offload/dit-cpu-load/component-files）は 16GB fit のため**既に実装済 → done**（＝「Phase 2 に実装済みのものがある」の正体）。
- **spec の役割**: 設計の正本＋ゴール（AviUtl2 で LTX 2.3 を動かす拡張機能）への道筋を、**粗い粒度・広い視点**で書く。実装細部は handoff/VERIFICATION_LOG/RESOLUTION_DURATION が担う（粒度が違う）。

---

## フェーズ別ロードマップ（2026-07-01・旧枠・上の「設計対話の決定」が正）

> **▶ 2026-07-02 更新**: 下記 Phase 2/5 の枠組みは上の「設計対話の決定」で **early-integration（Phase 2＝AviUtl2 統合）**に是正され、1080p-upscale「機能」・多人数インフラは削除。以下は spec 全面改訂で置換予定の旧枠（Phase 1 の done/残の記述は引き続き有効）。
> **開発フェーズの正本サマリ（旧枠）。** 出典＝spec `LTX23_Backend_Specification_v04…md §0.1/0.2/0.3/1.3/5.2`・`config.yaml` vram 節・`VERIFICATION_LOG §9.8`。
> **自然な順（規定はしない）**: 長尺動画への道＝Phase 2 の「クリップ連結（複数キーフレーム/終了フレーム I2V）」。その**Phase 1 側の前提が下記 (b) I2V 連続＋音声の検証**。→「(b) を片付けてから Phase 2 連結」が筋（優先順位は担当者＋ユーザーで決定）。

### Phase 1 ＝ 最小バックエンド（T2V＋最小I2V＋音声＋16GB fit＋720p）
- **✅ done**: 凍結 REST API（÷64・8n+1・T2V/最小I2V）・単一ジョブ管理・Low VRAM baseline・**720p(1280×768→crop720) T2V/I2V**・**音声 joint 生成**（§9.8 PASS）・~28GB モデル構成・mock pytest。16GB fit（下記注記の意味で）。
- **⚠️ 残課題（Phase 2 前に）**:
  - **(b) I2V マルチジョブ＋音声連続生成の検証【✅ 2026-07-02 PASS＝VERIFICATION_LOG §10.7】** — 直接ハーネス I2V×4 @384＋本番API I2V×3 @512×320 の両経路 PASS（commit 86%・installed_tf=1・VRAM定常~9.5GB・全出力 AAC・両経路の peak_vram 一致）。keep=0 の gen 漸増は4本で ~115→134s（許容範囲）。**Phase 2 クリップ連結の前提クリア。**
  - (a) 検証用 **Gradio GUI 手動確認**（UI で T2V/I2V/720p/crop トグルが出せるか）。
  - (c) **README/spec の全面改訂**（spec 本文は pre-pivot のまま・~90KB）。
  - (d) **dead-code 整理**（text-only 化で no-op 化した `_SkipGemmaLMSDOps` 等・意図的温存分）。
  - (e) **load/encode の一時 shared 溢れ**（~2–2.5GB）。
- **注記「16GB fit」の意味**: ＝**ハード OOM しない**（溢れ＝system RAM へページングで激遅だが完走）の意。「全工程が dedicated 16GB 内」は**未達**（既知・別軸＝den2 の解像度×尺スケーリング。§8.4/§8.6）。

### Phase 2 以降 ＝ spec §0.3「Phase 1 で実装しない」群
- **最適化（高解像度・長尺・品質）**: attention tiling・FFN チャンキング・latent spatial upscaler で 1080p 化・Gemma Q6_K 品質バンプ。
- **高度な条件付け（＝長尺の本命）**: IC-LoRA（depth/pose/edge/canny/参照動画）・V2V・**複数キーフレーム I2V／終了フレーム conditioning（＝クリップ連結で長尺）**・プロンプト強化（enhance_i2v・要 vision 再導入 [[qat-reclamation-textonly-gemma]]）。
- **本番インフラ**: 本格ジョブキュー（Phase 1 は in-memory 単一）・永続 DB・認証・インターネット公開・動画アップロード。
- **Phase 2 エンドポイント**（spec §5.2）: `PUT /api/v1/config`・`GET /api/v1/jobs/{id}/preview`・`POST /api/v1/upload/video`・`POST /api/v1/upload/image/keyframes`。

### Phase 5 ＝ AviUtl2 統合（本来の最終目的・別プロジェクト）
- spec §0.1/1.3: **API 安定後の「半ば独立した後続プロジェクト」**。REST API は AviUtl2 専用にしない（DaVinci Resolve 等の別フロントエンドも想定）。＝Phase 2 ではない。

> **↓下記「次の一手メニュー」は上記フェーズの具体タスク（順序規定なし）。**

---

## 次の一手メニュー（**順序は規定しない**・担当者＋ユーザーで優先順位を議論して決める）

> **次セッションはこのメニューから優先順位を議論して選ぶところから再開する。** 以下は「◯◯せよ」という指示ではなく、
> 各候補に「何を・なぜ・依存/前提・粗い規模感」を添えた**選択肢**。相互依存に注意して順序はその場で決める。

- **検証用 Gradio GUI の手動動作確認**
  - 何を: `/ui`（Gradio）で 720p / crop トグル / T2V / 最小I2V を人が実際に出せるか確認。
  - なぜ: API は疎通済みだが GUI 経由の end-to-end は未確認。AviUtl2 統合前の最短の現物確認。
  - 依存/前提: 実 backend＋GPU。規模＝小（生成数本＋目視）。

- **（本来の目的）AviUtl2 拡張機能 ↔ 本 API 統合**
  - 何を: AviUtl2 拡張から本バックエンド（FastAPI, port 18620, `/api/v1/*`）を叩く連携を作る。
  - なぜ: このプロジェクトの最終目的。API は汎用設計で既に安定。
  - 依存/前提: 上の Gradio 手動確認で現物が固まっていると安全。規模＝大（別プロジェクト級・仕様書 0 章の想定）。

- **keep_resident=1 を 720p でも使える恒久最適化**
  - 何を: Gemma の out-of-place `.to(cuda)` 移動を in-place 化／移動前 CPU 解放し、keep=1 と 720p を両立。
  - なぜ: keep=0 は毎ジョブ再 materialize で gen 時間が漸増（720p で +24%）。長尺連結・速度が要件化した時の本命（§9.7 の高速 flat 経路）。
  - 依存/前提: 独立（信頼性ブロッカーではない・現状 keep=0 で安定）。規模＝中。

- **I2V マルチジョブ＋音声連続生成の検証**
  - 何を: 新デフォルト（comp=1 / keep=0）で **I2V の連続生成・音声付き**を実測（今回の実測は T2V マルチジョブのみ）。
  - なぜ: 「5秒クリップを繋いで長尺」（終了フレーム→次の開始フレームの I2V 連結）の前提。連結は I2V 連続なのでここが未検証だと着手できない。
  - 依存/前提: 独立。keep=0 の gen 時間漸増が実本数（4本以上）で許容範囲かの再計測も兼ねる。規模＝中。

- **README / spec の全面改訂** ✅**完了（2026-07-02・push 済）**＝`LTX23_Backend_Specification.md` v0.5 に全面改訂・旧 v04 削除・README 参照更新。以下は当時の記述（歴史）。
  - 何を: `README.md` と `LTX23_Backend_Specification_v04…md` を現アーキで整理。
  - なぜ: 両者とも de-fork Stage 5 で post-refactor 注記＋要点訂正は入れたが、spec 本文は依然 pre-pivot 構成（fp8-cast/公式パイプライン
    前提の章立て）。QAT 回収でモデル構成も変わった（tokenizer-only gemma_root）。**現状に反する箇所は本セッションで最小是正済**（下記
    「今セッションで是正した doc」）だが、章立てレベルの全面改訂は未了。
  - 依存/前提: 独立。規模＝中〜大（spec は ~90KB）。

- **dead-code 整理**
  - 何を: text-only 化で no-op 化した `_SkipGemmaLMSDOps`（`engine/gemma/gguf_quant_service.py:201`）・未使用 `path` 引数
    `_read_target_vocab_from_header`（同 :234）・その他 de-fork の残滓を整理。
  - なぜ: 保守性。ただし belt-and-suspenders / signature 互換で**意図的に温存**したものなので、消す前に §14.4 の意図を確認。
  - 依存/前提: **コード編集（本タスクはドキュメントのみ）**。規模＝小。byte-match ゲートで退行確認。

- **開発ゴミ掃除（方法ごと次セッションへ委譲・ユーザー指示 2026-06-30）**
  - 何を: `outputs/` のテスト出力・未追跡の診断スクリプト（`_gpu_mem_sampler.ps1` 系のログ等）の整理。
  - なぜ: 公開前の後片付け。**掃除の方法自体をユーザーと相談してから**（何を残すか判断が要る）。
  - 依存/前提: 独立。規模＝小〜中。

- **`origin/main` への push** ✅**完了（2026-07-02・ユーザー実施）**＝spec 改訂までを commit＆push・`origin/main` 同期済。※本セッション整理の WORKORDER＋handoff 追記は未コミット（次コミットで拾う）。
  - 何を: ローカル先行 commit 群を push。
  - なぜ: リモート同期。
  - 依存/前提: ドキュメント整備の確認＋commit が先。規模＝小。

### 今セッションで是正した doc（QAT 回収の反映・上記メニューの前提）
- `Docs/VERIFICATION_LOG.md`: **§14 を追加**（text-only Gemma・byte-match 3経路・device override 経緯・peak_vram 8440・commit）。
- 本書冒頭: 本「現状ステータス」＋「次の一手メニュー」を追加（以降の▶節＝歴史記録は温存）。
- `README.md` / `LTX23_Backend_Specification_v04…md`: QAT dir → tokenizer-only gemma_root＋text-only Gemma への**最小是正**
  （全面改訂はメニュー候補）。
- `Docs/QAT_RECLAMATION_RESEARCH.md` は既に ✅RESOLVED バナー付きで最新（追加編集不要）。
- `config.yaml` の `gemma_root` インラインコメントも監督が是正済（「後工程で回収予定」→「回収済＝物理削除」）＝VERIFICATION_LOG §14.5 と一致。

---

## ▶▶▶▶▶ post-refactor ステータス（2026-07-01・branch `refactor/engine-firstparty-cleanup`・最初に読む）

**de-fork リファクタが完了した。** このハンドオフの**以降の記述の多くは de-fork 前（2026-06-30）に書かれており、
`vendor/LTX-Desktop-LOW-VRAM/backend/`・`_ltx_worker.py`・「フォーク env / フォーク venv」を前提とする部分は陳腐化している。**
現行の正確な構成は `README.md` / `engine/` / `config.yaml` / `engine/VENDOR_NOTICE.md` を参照すること（歴史記録として本書の旧節は温存）。

### 何が変わったか
- **エンジンを first-party 化**: 旧・同梱フォーク `vendor/LTX-Desktop-LOW-VRAM` の backend ソースを **`engine/`**（project root
  直下・git 追跡）へ採用・再編。`engine/{worker,api_types,lora_types,pipeline,gguf,gemma,transformer}`。**アルゴリズムは不変**
  （配置・import・パッケージ名・デッド枝削除のみ）。旧フォークツリー（frontend/electron 含む）は**完全削除**。上流 `vendor/LTX-2` は温存。
  - 起動は `python -m engine.worker`（`cwd=root`, `PYTHONPATH=root`）。`_ltx_worker.py` は `engine/worker.py` に相当。
- **torch venv を移設**: 旧フォーク配下の `.venv` → **`./.venv-engine`**（gitignore）。app は torch 無し `./.venv`。config は
  `engine_python: "./.venv-engine/Scripts/python.exe"` / `engine_dir: "./engine"`（相対）。
- **43GB モノリス `ltx-2.3-22b-distilled-1.1.safetensors` を物理削除**（rename test で「GGUF+component 経路は非 open」を実証）。
  `checkpoint_path` は reference-only フィールドとして温存。
  - **✅ QAT Gemma dir 22.7GB は回収済（2026-07-01・text-only Gemma 化）**。当初は「wheel が build 時に tokenizer.model+model*.safetensors を
    glob するため construction-required で温存」だったが、**Gemma を text-only（`Gemma3ForCausalLM`・vision 無し）で構築**するよう作り替え、
    vision 構造ごと不要化。gemma_root は ~40MB の tokenizer-only dir（`models/gemma-3-12b-it-tokenizer/`）に差し替え、QAT dir は物理削除
    （models/ 50.9GB→28.15GB）。full-QAT baseline とバイト一致・peak_vram 微減・退行なし。詳細＝**§14** ／`Docs/QAT_RECLAMATION_RESEARCH.md`
    冒頭 ✅RESOLVED バナー、commit `93696b4`→`694ca54`→**main へ `--no-ff` マージ済 `826e76f`**（branch `refactor/qat-reclamation`）。
- **設定後始末**: 未使用 `quantization` を削除。`fp8_transformer`/`cpu_offload_text_encoder` は**凍結 `GET /status` 契約**のため保持
  （worker 非伝播）。`uv.lock` 消失を `engine/venv-engine.freeze.txt`＋`engine/engine-venv-pyproject.toml` で穴埋め。
- **凍結 API 契約は不変**: ÷64 解像度・8n+1 フレーム・T2V/最小I2V・`GET /status` の `vram_optimization`・`metadata.json` スキーマ・
  limits/presets はすべて保全。

### 検証（全段 SHA256 一致）
Stage 0–4 で「出力 mp4 の SHA256 がリファクタ前ベースラインとバイト完全一致」を各段で確認（挙動不変・別プロセス再起動でも決定的）。
baseline(seed=12345, "a calm ocean wave rolling onto a sandy beach at sunset, cinematic"): 512×320/49f=`23844b4e…6bb7bf`、
1280×768/49f=`4feea65f…3768da`、peak_vram_mb=9164。詳細は `Docs/VERIFICATION_LOG.md` §13。

### commit 列（`refactor/engine-firstparty-cleanup`）
`d0d3df5` Stage 1（フォーク delete-set 削除）→ `1bf4163` Stage 2a（engine/ へ relocate）→ `c6ff5a4` Stage 2b（venv 移設＋フォークツリー削除）
→ `4639008` Stage 3a（デッド枝刈り＋モノリス fail-fast＋存在チェック再配線）→ `35c3be5` Stage 4（config 後始末・reproducibility 復元）。
Stage 5（本ドキュメント改訂）は未 commit（監督確認待ち）。

---

## ▶▶▶▶ （歴史）720p 達成 & 連続生成の commit 枯渇を解決（2026-06-30 後半）

> **【超過】この節の「最初に読む／次は◯◯」は当時のもの。最新の現状と次の一手は冒頭「現状ステータス」＋「次の一手メニュー」が正。**
> 720p・連続生成の技術記録として温存。

**＝残課題C（720p スケールアップ）完了。本来の機能（16GB で 720p 動画生成）が、先行事例と遜色ないマシンスペックで実現した。** （当時の次の一手＝検証用 Gradio GUI 手動確認 →（本来の目的）AviUtl2 統合。→現在はメニュー化。）

### 何ができるようになったか（実機検証済）
- **720p 級（1280×768 を生成→任意で 1280×720 にクロップ）の T2V が本番 API 経路で完走**。生成 ~167–171秒（RTX 4070 Ti SUPER 16GB）。job `684393c6`（crop なし 1280×768 配信）/ `5a3540ba`（crop 1280×720）で ffprobe 確認。OOM なし。
- **連続生成（マルチジョブ）が commit 枯渇クラッシュせず完走**。720p×3本連続・384×256×3本連続いずれも全 PASS（実測）。→ 将来の「5秒クリップを繋いで 20秒動画」（終了フレーム→次の開始フレームの I2V 連結）の土台ができた。
- **✅ `--te-offload`（既定 ON）＝Gemma text-encode ピーク低減 DONE・A/B 実機検証済（2026-07-01）**。機構＝逐次 per-layer ストリーミング（GGUF Gemma 48 層を CPU 常駐→2層ずつ GPU へ・compute は GPU）。encode-phase Dedicated ピークを 15,839→10,540 MB（−33%）に下げ encode 時の shared 溢れを消去・wall-clock ペナルティ無し・出力 OFF とバイト一致（512×320/49f A/B）。720p ON スモークも完走（168.2s）。**toggle＝`--no-te-offload` で無効化**。詳細＝VERIFICATION_LOG §11、commit `96f41c4`。**★スコープ注意：te-offload 自体は全体ジョブ `peak_vram_mb`（両モード 16,944）に触れない**（残課題B の Gemma encode 側＝peak ① だけ解消）。**↳ transformer load 側＝peak ② は ✅ `--dit-cpu-load`（既定 ON, 2026-07-01）で別途 RESOLVED**（transformer ブロックを直接 CPU 構築し full-GPU materialization を経由しない＝transformer-load Dedicated 15.8→1.4GB・512×320 で whole-job ceiling 16,944→~9.2GB・出力バイト一致。VERIFICATION_LOG §12、後述【B】）。

### 確定した本番デフォルト設定（2点・いずれも検証済・コミット対象）
1. **`services/ltx_runner.py`** worker env に `env.setdefault("LTX_KEEP_RESIDENT","0")` ＝ **keep_resident OFF が既定**（明示 env は尊重）。理由＝keep_resident=True だと 720p で Gemma を CPU ビルド→out-of-place `.to(cuda)` 移動する瞬間二重在が 16GB マージンを超え **native crash**（traceback 無し、text-encode 中）。=0 で直接 cuda ビルドになり回避。`LTXFastVideoPipeline.create()` の既定は False（`ltx_fast_video_pipeline.py:75`）。
2. **`config.yaml`** vram に `use_component_files: true`（Path B）＋ `block_swap_blocks_on_gpu: 8` / `vae_spatial_tile_size: 512` / `vae_temporal_tile_size: 64`。理由＝comp=1 が VAE/audio/projection を 46GB モノリスでなく小ファイルから読み、**毎ジョブ再 materialize の commit を束縛**。これが連続生成の鍵。

### なぜこの設定か（実測根拠・GPU＋system commit 両監視）
- **comp=0/keep=0（旧既定）はマルチジョブで残課題A の機構（§8 commit 枯渇）が再発**＝3本連続@384 の job3 で commit **99.2%（135GB）→ Gemma 再構築で native crash**。§9.7 のリーク修正では治らない別機構（毎ジョブ再 materialize）と実証。
- **comp=1 にすると 46GB モノリスを使わず commit が ~90%（102GB）に束縛**＝384/720p とも3本連続 PASS（crash 消滅・commit はジョブ間で累積せず横ばい）。
- 留意（非致命）：keep=0 は毎回再 materialize するため **gen 時間が漸増**（720p で 168→209秒/3本、+24%。アロケータ断片化等）。commit は横ばいなので暴走しない。**長尺連結を多数本回す段で問題化したら、keep=1 を維持したまま Gemma 移動を in-place 化する恒久最適化に着手**（§9.7 の高速 flat 経路＝keep=1 と 720p を両立させる）。

### マシンスペック（公開時の見積り・先行事例と比較）
- 「commit ~102GB」は **ディスクでなく仮想メモリ予約（物理RAM＋ページファイル）** のピーク。先行事例 ComfyUI 16GB レシピも「RAM32GB＋swap64GB」相当（§91）＝**ほぼ同等で極端でない**（旧懸念の ~200GB は 46GB モノリス＋二重 materialize の悪い経路の話で、comp=1 で回避済）。本機は RAM64GB＋pagefile48GB＝commit 上限 ~112GB で 90% 着地。
- **実行に本当に要るモデルは ~28GB**（GGUF transformer 16.5＋GGUF Gemma 6.8＋components 3.85＋upscaler 0.93＋設定）＝ComfyUI GGUF 構成（~25–30GB）と同等。~~現状 `models/` は 93.83GB あるが、**モノリス 43GB＋qat 重み 22.7GB（計 ~66GB）は今の comp=1/GGUF 経路では不要候補**~~ → **【是正 2026-07-01】モノリス 43GB は de-fork（§13.3）で、qat 重み 22.7GB は今回の QAT 回収（§14）で共に削除済。`models/` は現在 28.15GB で先行事例並みが達成済。**

### 次のマイルストーン
1. **検証用 Gradio GUI でユーザーが手動動作確認**（UI が 720p/crop トグル/T2V/I2V を出せるか要確認）。
2. （本来の目的）**AviUtl2 拡張機能 → このバックエンド API（FastAPI, port 18620, `/api/v1/*`）統合**。

### 残課題（ディレクトリ掃除を除く・優先度はユーザー判断）
> ※開発段階のゴミ（`outputs/` のテスト出力、未追跡の診断スクリプト等）の**ディレクトリ掃除は、専任の次セッションに方法ごと委ねる**ため、ここには残課題として載せない（ユーザー指示・2026-06-30）。

- **【B】load/encode の一時 shared 溢れ → ✅ load/encode フェーズは 16GB 内達成（2026-07-01）**。(i) **Gemma encode 側（peak ①）は `--te-offload`（既定 ON）で解消**（encode Dedicated 15.9→10.5GB・encode 時 shared 溢れ 0、VERIFICATION_LOG §11）。(ii) **transformer load 側（peak ②）は `--dit-cpu-load`（既定 ON）で解消**＝従来 block-swap が一旦 full-GPU ロード（peak max_alloc 16,944）→退避していたのを、transformer ブロックを**直接 CPU 構築**し full-GPU materialization を経由しないよう変更（transformer-load Dedicated 15.8→1.4GB・load 時 shared 溢れ消失・出力バイト一致。**512×320 で whole-job ceiling 16,944→~9.2GB**、VERIFICATION_LOG §12）。**↳ これで load/encode フェーズ（peak ①＝te-offload・peak ②の load-spike＝dit-cpu-load）は 16GB 内に収まった。** 残るニュアンス＝**大解像度・長尺では denoise stage2 が依然天井へ近づきうる**（これは固定 load スパイクとは別軸＝den2 のトークン数スケーリング。`RESOLUTION_DURATION_CAPABILITY.md §2` の den2 表は不変）。詳細は下記【残課題B の本丸】＋ VERIFICATION_LOG §7.9/§11/§12。
- **【最適化】keep_resident=1 を 720p で使えるようにする恒久修正**＝Gemma の out-of-place `.to(cuda)` 移動を in-place 化／CPU 側を移動前解放。成功すれば keep=1（§9.7 で 6ジョブ flat・高速・gen 時間漸増なし・commit より低い）と 720p を両立。長尺連結や速度が要件化した時の本命。
- **【残課題B の本丸＝peak ② 低減】→ ✅ RESOLVED（`--dit-cpu-load` 既定 ON, 2026-07-01）**。transformer load 段の一時 shared 溢れ（block-swap の full-GPU ロード→退避）を、transformer ブロックの**直接 CPU 構築**（full-GPU materialization 非経由）で除去した＝**全体ジョブ天井を下げるレバーが効いた**：512×320 で whole-job ceiling（torch `max_memory_allocated`）**16,944→~9.2GB**、transformer-load Dedicated 15.8→1.4GB、出力バイト一致、wall-clock 退行なし（VERIFICATION_LOG §12）。Gemma 側（peak ①）は te-offload 済。**↳ 残るのは「den2（denoise stage2）の解像度・尺スケーリング」軸のみ**＝固定 load スパイクではなく高トークン時に天井へ近づく可変成分（`RESOLUTION_DURATION_CAPABILITY.md §2`・VERIFICATION_LOG §12.5 のスコープ注意）。これは別課題で、load/encode の「真の 16GB fit」自体は達成済。
- **【任意・te-offload チューニング露出】**`GemmaLayerOffloadService(layers_on_gpu=2)` は現状ハードコード（`config.yaml` 未露出）。1 に下げると encode ピークさらに低下／3-4 で速度トレード。露出は任意の将来作業（VERIFICATION_LOG §11.6）。
- **【連続生成の上限確認】**keep=0 での gen 時間漸増が、連結機能の実本数（5s→20s＝4本以上）で許容範囲かを、その実装時により長い連続で再計測。
- **【公開フットプリント削減】**comp=1/GGUF 経路で **モノリス 43GB＋qat 重み 22.7GB が実行時に本当に開かれないか**をコード/ログで検証→不要なら required から外す（DL させない/削除可に）。落とせばディスク要求 ~28GB で先行事例並み。[[no-large-pagefile-disk-requirement]] の達成。
- **【✅ 検証済 2026-07-02＝VERIFICATION_LOG §10.7】**新デフォルト（comp=1/keep=0）での **I2V マルチジョブ・音声付き連続生成を実機 PASS**（直接ハーネス I2V×4 @384＋本番API I2V×3 @512×320 の両経路）。連結機能の前提クリア。
- **【D】README / `LTX23_Backend_Specification_v04…` の全面改訂**＝pre-pivot のまま。アーキテクチャが固まった今が改訂の好機（残課題サマリ末尾参照）。
- **【真の目的】AviUtl2 拡張機能との統合**（上記マイルストーン2）。

### 一次情報のポインタ
- **解像度×尺の能力リファレンス＝Docs/RESOLUTION_DURATION_CAPABILITY.md**（16GB でどの解像度/尺まで作れるか・den2 推定式・生成時間・AviUtl2 UI 含意の正本）。API の尺 cap は §8.4/§8.6 を根拠に **257f→481f(20s)** へ緩和済み（溢れ/低速はクライアント警告に委ねる方針。解像度別 spill-free = 720p:257 / 1080p:153 / 1440p:81 を `/api/v1/config` の `limits.spill_free_frames` に露出。1080p 長尺は非実用/commit リスク → 720p 生成＋外部 upscale 推奨）。
- 技術記録＝**VERIFICATION_LOG §10**（720p 二段達成・keep_resident 原因/修正・component_files マルチジョブ修正・commit/ディスク実測・§10.6 解像度×尺スイープ）。
- スケールアップ調査の正本＝Docs/SCALEUP_16GB_RESEARCH.md（✅達成バナー追記済）。
- 設定知識＝Docs/LTX23_REFERENCE.md。memory `[[720p-16gb-verified]]` / `[[ltx-bridge-project]]`。

**（以下は本セクションより前の歴史的経緯。残課題A〜D の元記述・Phase 5 配線記録等は参照用に温存。最新は上記 ▶▶▶▶ が正。）**

---

> **【超過】以下の「最終更新 2026-06-30」ブロックは歴史記録。当時「次の一手」に挙げた qat-drop(24GB削減) も 720p スケールアップ
> （残課題C）も**いずれも✅完了済**（QAT 回収＝§14／720p＝§10）。現状と次の一手は冒頭「現状ステータス」＋「次の一手メニュー」が正。**

最終更新: 2026-06-30（**残課題A＝per-job リークを診断確定→修正→6ジョブ実機検証 PASS。フォーク backend を版管理化（commit 済）。音声 Phase 1 実機確認 PASS（✅DONE）。当時の次の一手＝qat-drop(24GB削減)／720p スケールアップ（残課題C）＝いずれも現在✅完了**）/ 想定読者: 次セッションのエージェント

> **（歴史）当時このブロックが最新・正本でした。まず §0 と「★残課題サマリ」を…と案内していたが、現在の最新・正本は冒頭「現状ステータス」節。**
> **【リポジトリ状態 2026-06-30】** フォーク `vendor/LTX-Desktop-LOW-VRAM/backend/` の **.py ソース 169ファイルが版管理対象になった**（commit `4ea5c4b` "Vender"、origin/main 同期済）。`.gitignore` を negation ブロックに変更し backend ソースのみ追跡（`.venv`/weights/フロントエンド/`uv.lock` は除外維持）。フォーク自身の `.git` は **`.git_fork_disabled` にリネームして無効化**（境界を外しファイル追跡を可能にするため・42MB・削除可）。出典は `vendor/LTX-Desktop-LOW-VRAM/backend/VENDOR_NOTICE.md`。**＝以後エンジン改変は通常の git で commit できる**（以前は vendor/ 丸ごと ignore でエンジン全体が版管理外だった）。
> 設計判断の根拠と実機検証の全経緯は [VERIFICATION_LOG.md](VERIFICATION_LOG.md)（**最新＝§9.7（per-job リーク診断確定＋修正＋検証 PASS＝残課題A 完全解決）。次セッションはまず §0 末尾の ▶▶▶ と §9.7 を読め**、§9.6＝Path B component-files、
> §6＝Phase 5(A) 配線、§5＝Gemma GGUF）が一次情報。実装計画は `~/.claude/plans/frolicking-tumbling-hennessy.md`。
> 要約は memory `[[ltx-bridge-project]]` / `[[ltx-desktop-lowvram-fork]]`。⚠️ 古い記述が残るドキュメント（後述 §6）に惑わされないこと。

---

## 次セッション開始点（2026-06-30 更新）

- **完了**: 音声 Phase 1 実機確認 **✅PASS**（VERIFICATION_LOG §9.8）。16GB 高解像度スケールアップの調査を **`Docs/SCALEUP_16GB_RESEARCH.md`** に集約（正本）。残課題A（worker 再利用 crash）も解決済（§9.7）。
- **最初に読むべき**: **`Docs/SCALEUP_16GB_RESEARCH.md`**（720pレバー棚卸し＋コミュニティ実証レシピ＋真の難所3点）。要点＝**720pレバーの 4/5 は我々のフォークに既に配線済・既定OFF**（二段パイプライン / attention tiling=`attention_tile_size` / VAE 空間タイル / VAE 時間タイル / block-swap）。**未実装は FFN チャンキングのみで、それは主に長尺向け＝1280×720 の空間スケールには非必須**。
- **次の一手（推奨A・GPU不要から着手）＝720pスケールアップ**:
  - **Step 1（読解のみ・GPU不要）**: 公式 `ti2vid_two_stages.py`（`vendor/LTX-2/packages/ltx-pipelines/.../ti2vid_two_stages.py`）を精読し、(1) 難所①「stage1→stage2 の VRAM 遷移＝両段を同時保持するか／段間で latent 保存＋モデルアンロードが要るか」、(2) `LTXFastVideoPipeline.create()` の VRAM ノブ（`attention_tile_size` / `vae_spatial_tile_size` / `vae_temporal_tile_size` / `block_swap_blocks_on_gpu`）が **両ステージに効くか** を確定する。
  - **Step 2**: SCALEUP doc §5 の推奨レシピで **計装つき 1 本だけ生成**（dedicated＋shared 両監視・既存 `_gpu_mem_sampler.ps1` 併走・フォーク venv のみ）→ 収まるか／どこでスパイクするか実測。
  - **Step 3（必要時のみ）**: block-swap 深度調整 or 段間アンロード実装。FFN チャンキングは将来の長尺対応で初めて検討。
- **真の難所3点（先行事例に答えが無い・SCALEUP §4）**: (1) 段間(stage1→stage2)遷移スパイク (2) 22B の block-swap 深度 (3) tiling＋GGUF＋block-swap の共存性。
- **代替B**: qat-drop（24GB 削減）。
- **目標解像度＝1280×720 で充分**（より大きいサイズ・長尺の情報は SCALEUP doc 付録に保全）。
- **作業原則の念押し**: 監督役＝編集/テスト/生成はサブエージェントへ委譲、本体 Python 不可触・生成はフォーク venv のみ、**いきなり 720p 生成や闇雲なスイープをしない**（先行事例レシピを写す＋最小再現確認＝[[research-prior-art-first]]）。

---

## 0. いまどこにいるか
- **目標＝LTX-2.3 を 16GB VRAM(RTX 4070 Ti SUPER/Win/torch2.9.1+cu128)で動かし FastAPI で公開**。Phase 1＝T2V＋最小I2V。
- **✅ Phase 5(A) 配線完了・実機検証済**：凍結 REST API が実エンジン（GGUF 低VRAMパイプライン）で生成。`services/ltx_runner.py`
  の `_RealBackend`＝**Approach W（常駐サブプロセスワーカー `_ltx_worker.py`）**。詳細 §3／VERIFICATION_LOG §6。
- **✅ Phase 5(B)＝denoise 工程の VRAM 低減 完了・実装済（2026-06-29）**：真因＝本機 Windows で `expandable_segments` が
  no-op のため、block-swap の transformer ロード後に空きセグメント ~16GB がアロケータに居座り denoise 中 shared へ ~3.9GB
  溢れ（激遅化）。対策＝`_ltx_worker.py` で**各 denoise 直前に `gc.collect(); torch.cuda.empty_cache()`**（monkeypatch）。
  結果＝denoise shared 3969→**742MB(ambient)**・dedicated ~6.4GB・**生成 −46%**、T2V/初回I2V 実機検証・mock pytest 13 passed・
  凍結境界不変。**※「denoise 工程の持続的溢れ解消」であって「全工程 16GB 内」ではない**（load/encode は依然 ~2–2.5GB shared
  一時溢れ＝下記【残課題B】）。詳細＝VERIFICATION_LOG §7。
- **⚠️ 別件の production ブロッカー（Phase 5B 起因でない既存バグ#5系）＝worker 再利用 crash**：常駐 worker の**2ジョブ目以降**で
  native access violation（exit 139）。詳細・修正方針は下記「★残課題サマリ【残課題A】」。
- **▶（超過 → 最新は §0 末尾の ▶▶▶／VERIFICATION_LOG §9.6）旧・起点メモ＝Stage 3**：**残課題A の真因が判明し理解が一新された。最新は
  [VERIFICATION_LOG.md](VERIFICATION_LOG.md) §8（特に §8.3 真因再フレーム・§8.8 commit 確定・§8.9 過剰commit究明・§8.10 Stage1・
  §8.11 Stage3 設計）。** 要点：**再利用 crash は「safetensors mmap ハンドルのバグ」ではなく「Windows commit（仮想メモリ）
  枯渇」**（per-job の全 submodel 再構築が ~70GB のモデルを毎ジョブ re-mmap＋re-materialize＝~125GB commit スパイク→
  上限到達で native 0xC0000005）。**VRAM レバー（empty_cache 等）も非mmap 化も的外れと実証済**。**次の作業＝Stage 3 ＝
  load-once/keep-resident（フォークの未配線 `StateDictRegistry` を配線）を §8.11 の設計ブリーフどおり実装・検証**。
  Stage 1（捨てる bf16 Gemma 24GB を読まずスキップ）は実装済・併存。詳細・着手手順は下記【残課題A】（更新済）と §8.11。
- **▶▶（超過 → 最新は §0 末尾の ▶▶▶／VERIFICATION_LOG §9.6）§9 方針メモ＝Path B 決定**：Stage 3（StateDictRegistry）は実装したが **job1 warmup が commit 152GB/99% で crash**（§8.11 が留保したリスクが顕在化）。リサーチで**真の解決＝モノリス廃止＝コンポーネント・ファイル分離（Path B）**と確定：VAE/audio/projection を 46GB モノリス＋24GB qat でなく **ComfyUI 同様の小単体ファイル**（Video VAE 1.45GB／Audio VAE 365MB／text projection 2.31GB）から読む＝commit bounded・RAM~32GB・巨大ページファイル/DL 不要・凍結API＋パイプライン無改変。**音声は既にフォークが mp4 へ mux 済みと判明→Phase 1 へ格上げ**（`crop_mp4` の音声保持修正だけ要）。唯一の新規実装＝connector 供給（方式選択中）。詳細・理由・棄却した代替（pread/公式streaming/ComfyUI移行）は **§9** が正本。
- **▶▶▶ 最新の正本・次セッションの起点（2026-06-30 最終・最重要）＝VERIFICATION_LOG §9.7：残課題A（worker 再利用 crash／per-job リーク）は完全解決**。経緯：Path B（component-files）が §9.6 で warmup crash を解消したが job5 で別の per-job リークが露呈（commit・VRAM が generate 毎に単調増加→160.5GB/100% native crash、`block_swap_service.py:86`）。§9.7 で **read-only 並列診断→非破壊・計装run で按分実測→標的2パッチ→6ジョブ実機検証 PASS** により解決：
  - **真因＝`BlockSwapService._installed_transformers` が毎 generate の transformer を append し続け解放しない**（仮説a 確定）。registry は HIT・キャッシュ汚染なし（仮説b/c・in-place .to 説は実測で棄却）。計装で `installed_transformers` 1→2→3→4 を直接観測。leaked transformer の CPU 退避ブロック=commit +18GB/job、GPU 窓=VRAM 床 +1GB/job、で両軸を単一原因に統合。
  - **修正（フォーク製品コード・凍結境界外・persistent）**：①`block_swap_service.py install()` で append 前に `_installed_transformers.clear()`（keep-latest）②`_ltx_worker.py _do_generate()` の `_emit("done")` 直後に `gc.collect(); torch.cuda.empty_cache()`。**`model_ledger.py` の in-place `.to` 改修は不要**（キャッシュ非汚染を実測確認＝vendored `.venv` 不触）。
  - **検証 PASS**：6ジョブ全完走（旧 job5 crash 点突破）、`installed_transformers` 1 で一定、torch 床・denoise peak・commit すべて平坦、**出力 mp4 が修正前と SHA256 バイト一致**（計算不変）、mock 13 緑。詳細＝§9.7。
  - **音声 Phase 1 の実機確認は ✅DONE（2026-06-30 PASS、VERIFICATION_LOG §9.8 参照）**＝native joint audio が 16GB で効果音/音楽/発話を生成・AAC mux・decode VRAM 余裕（dedicated ~2.7GB）・crop 音声保持を確認済。**次の一手＝qat-drop(24GB削減)／720p スケールアップ（残課題C）**。gate＝env `LTX_COMPONENT_FILES=1`＋`LTX_KEEP_RESIDENT=1`。

## ★残課題サマリ（次セッション向け・2026-06-29 時点）

Phase 5(B)＝**denoise 工程の VRAM 低減は達成・実装済**（empty_cache fix／§7.5・§7.8、shared 計測で検証）。
以下がオープン項目。優先度はユーザー判断。

### 【残課題A】worker 再利用の native crash → ✅ 解決済（2026-06-30・VERIFICATION_LOG §9.7）

> **✅ 【2026-06-30 解決】残課題A は完全解決した。** 真因＝`BlockSwapService._installed_transformers` が毎 generate の transformer を保持し続け解放しない（registry HIT・キャッシュ非汚染を計装で実証、仮説b/c は棄却）。修正2点（フォーク製品コード・凍結境界外・版管理済）＝① `block_swap_service.py install()` で append 前に `_installed_transformers.clear()`（keep-latest）② `_ltx_worker.py _do_generate()` の `_emit("done")` 直後に `gc.collect(); torch.cuda.empty_cache()`。6ジョブ実機検証 PASS（旧 job5 crash 点突破・`installed_transformers` 1 で一定・床/denoise/commit 平坦・出力 mp4 が修正前と SHA256 バイト一致・mock 13 緑）。詳細＝VERIFICATION_LOG §9.7。**以下（旧）は診断確定前の歴史的記録。**
>
> **【2026-06-29 全面更新・正本＝VERIFICATION_LOG §8】** 以下の「真因＝mmap 反復ロードの native 破損」「修正方向＝非mmap」は
> **誤診で棄却済み**。実機計測（§8.8）で **真因＝Windows commit（仮想メモリ）枯渇**と確定した。要旨：
> - distilled は各 generate で **全 submodel（Gemma/VAE/transformer）を毎回ディスクから再構築**。Windows では safetensors の
>   mmap が「読む量に関係なくファイルサイズ分の commit を予約」＋get_tensor の materialize で **モデルサイズの ~2×** を commit。
>   per-job で ~70GB のモデルが **~125GB commit スパイク**になり、commit 上限（~146→自動拡大160GB）到達で native 0xC0000005。
>   **VRAM は無関係（crash 時 dedicated 9.5GB）／site は非決定的（safetensors VAE / GGUF dequant 等）＝ヒープ破損でなく commit 枯渇**。
> - **試行と棄却**：①完全 non-mmap whole-file→46GB で MemoryError・FAIL（§8.1）。②Gemma CPU キャッシュ→commit を足して逆効果・FAIL（§8.2）。
>   ③**Stage 1＝捨てる bf16 Gemma 24GB を読まずスキップ→実装済・数値不変・commit −10〜16GB だが job3 でまだ crash＝不十分**（§8.10）。
> - **★次セッションの作業＝Stage 3：load-once/keep-resident**。フォークに**存在するが未配線**の `StateDictRegistry`（`registry.py:49-84`）を
>   配線し、submodel を warmup で1度だけロード→全ジョブ HIT 再利用＝per-job 再 materialize を消し commit を steady **~62GB** に。
>   **配線は `ltx_fast_video_pipeline.py:137`（DistilledPipeline 構築直後・service install 前）に ~4行＋Gemma を `_target_device()`
>   準拠（CPU build）にする1修正**。設計・commit試算・VRAM中立・GGUF互換・要確認点は **VERIFICATION_LOG §8.11** に詳細。
>   検証＝reuse_loop（CACHE=0 EC=1 BS=8）＋commit/VRAM サンプラで全5ジョブ完走・commit steady・出力不変・mock 13。
>   **ユーザー作業原則**：実装前に計画提示→承認、計測はサブエージェントに「終了まで block＋同一ターン報告（Monitor 禁止）」を課す。
> - **現コード状態（known-good＋Stage1）**：worker は pre-denoise empty_cache fix のみ＋Stage1 の bf16 Gemma skip。GGUF `copy=True` 温存。
>   `services/gemma_sd_cache.py`・`services/sft_loader_safe_patch.py` は dormant（未配線）。mock pytest 13 passed・凍結境界不変。
>
> 以下（旧）は歴史的記録。

- **症状**：常駐 worker（Approach W）の **2ジョブ目以降の generate 冒頭で Windows native access violation
  （exit 139 / 0xC0000005）**。worker が無言で死亡→`pipeline_manager._cleanup_after_error→unload→次ジョブで再ロード`で
  復帰するが当該ジョブは失敗。warm 運用で断続的に再発（単発/初回は常に成功）。
- **Phase 5(B) 起因ではない**：対照実験（fix 有/無）で **empty_cache fix と無関係**と確定（両方 job2 で crash）＝既存の
  **バグ#5（access-violation）系**。Phase 5A の「reuse 両 PASS」は非決定的な幸運だった。
- **真因**：distilled が各 generate で Gemma テキストエンコーダを `del→全再構築`（VRAM 節約のため）。毎回 **base Gemma
  safetensors を再 mmap ＋ GGUF Gemma を再 dequant**。この反復重ロードが本機 Windows で断続的に native 破損。crash site
  は複数・非決定的：`sft_loader.py:36`（mmap read→`torch storage __getitem__`）と `gguf_quant_service.py:402 _dequant_q6_k`。
- **失敗した試み**：`sft_loader_safe_patch`（copy=True eager 化）配線→**効かず撤去済**（fault は上流の mmap materialize＝
  `get_tensor`、かつ GGUF dequant site は非カバー）。詳細 §7.7。
- **修正方向（優先順位を補正）**：
  1. **［最優先・小・先行事例準拠＝未試行］完全 non-mmap ロード**。コミュニティの実証済み回避策は **mmap をやめる**こと
     （ComfyUI `--disable-mmap`／#13220）。我々の safe-patch は `safe_open`+`get_tensor`（＝mmap のまま）を残し `.to(copy=True)`
     だけ変えたので**効かなかった**（fault は上流の `get_tensor` の mmap 実体化）。→ **safetensors を `safetensors.torch.load_file`
     （非mmap / `backend="pread"`）化**＋**GGUF dequant も mmap 非依存化/計算前コピー**（2つ目の crash site `_dequant_q6_k` は
     city96 #444/#416 と同系の mmap エイリアス疑い）。両 site を non-mmap 化して reuse loop で再検証。
  2. **［大・代替］** 1 で消えなければ **per-job 再構築を廃し Gemma をジョブ間キャッシュ**（VRAM 常駐は 16GB fit と非両立 →
     CPU RAM 側キャッシュ・encode 時のみ GPU）。
  - **要 research**：ComfyUI `--disable-mmap` 実装、フォーク backend が Windows 再利用でどう振る舞うか（Linux 前提か／
    per-request で何を再ロードするか）、公式 CLI が一発実行ゆえ無症状なだけか。**この crash は Windows 固有**
    （safetensors #164 の mmap ハンドル寿命）で、公式/フォークが「解決した」のではなく発症条件（Windows＋同一プロセス反復
    mmap）を踏んでいないだけ、が現時点の理解。詳細 §7.6/§7.7。

### 【残課題B】load/encode の一時 shared 溢れ（「全工程 16GB 内」は未達）
- **症状**：denoise は ambient だが、**Gemma encode（shared ~2.0GB）と transformer load（shared ~2.5GB）は依然 16GB を
  一時超過**（dedicated ~15.9GB＋shared）。ユーザー目視と一致。global shared ピークは 3.9GB→2.5GB に減ったがゼロではない。
- **原因**：transformer load＝block-swap が**一旦 full-GPU ロード(peak 16.9GB)→退避**。Gemma encode は ~15.1GB で本質的にタイト。
- **修正方向**：transformer ブロックを**直接 CPU ロード**（full-GPU を経由しない）。Gemma 側は残課題A の load-path 作り替えと
  地続き。詳細 §7.9。

### 【残課題C】スケールアップ 1280×768→720p クロップ（本来の機能目標）→ ✅ 完了（2026-06-30 後半・冒頭 ▶▶▶▶ と VERIFICATION_LOG §10 が最新）
> ✅ **達成**：1280×768/121f を本番 API で完走（~167–171秒・16GB）、crop で 1280×720 配信、連続3本も commit 枯渇せず PASS。レシピ＝comp=1（Path B）＋keep_resident=0＋bs=8＋vae 512/64。真の難所3点の実測結果は冒頭 ▶▶▶▶／§10 参照。以下（旧）は着手前の調査メモ。
- → 詳細調査は Docs/SCALEUP_16GB_RESEARCH.md に集約（レバー棚卸し＋コミュニティレシピ＋真の難所3点）
- ★訂正(2026-06-30)：以下「修正方向」の **「attention tiling 配線（未配線）」前提は古い**。SCALEUP doc で判明＝attention tiling は既に実装済（`attention_tile_service.py`・`attention_tile_size`、既定OFF）。VAE 空間/時間タイル・二段・block-swap も配線済。**未実装は FFN チャンキングのみ（主に長尺向け・1280×720 には非必須）**。真の難所は移植でなく (1)段間遷移スパイク (2)22B block-swap 深度 (3)tiling＋GGUF＋block-swap 共存。**目標は 1280×720 で充分**。次の一手の手順は本書冒頭「次セッション開始点」＋ SCALEUP doc §5/§6 を参照。
- 384x256 の denoise は解消済だが **production ターゲットは 720p**。高解像度では denoise の**アクティベーションが支配的**に
  なる（384x256 で効かなかった活性軸技法が高解像度で本命）。
- **修正方向**：**attention tiling 配線**（`attention_tile_size`＝フォークに実装済・我々の worker/config で未配線）＋
  **FFN チャンキング移植**（RandomInternetPreson `ChunkedFFN`→`ltx_core/.../feed_forward.py:FeedForward.forward`）。高解像度
  VAE decode には VAE tiling も。empty_cache fix は高解像度でも有効。
- **要・新規診断**：1280×768 で改めてどの軸が支配的かを §7 と同じ手法（perf-counter dedicated/shared フェーズ別）で計測して
  から技法選択。詳細 §7.1、LTX23_REFERENCE §5。

### 【残課題D】後回し（低優先）
- README / `LTX23_Backend_Specification_v04…` は pre-pivot のまま＝全面改訂は**スケールアップ完了後**まで保留。
- 他の `non_blocking=True` 箇所（`fuse_loras.py`／`gemma_gguf_quant_service.py` の owned-tensor 転送）は mmap 非依存で低リスク＝据え置き。
- 品質バンプ：Gemma Q6_K（忠実度↑、encode に余裕あれば）。

**相互関係/推奨着手順（一例）**：A と B は「Gemma/transformer のロード経路」で重なる（A の load-path 作り替えが B も部分解消）。
C は独立の機能拡張。**A（信頼性ブロッカー）→ B → C（720p）** が一例だが優先度はユーザー判断。

---

## 1. アーキテクチャ（二層・これを壊さない）
- **凍結層（不変の資産）**: REST API 形（`api/models.py`・**÷64 解像度契約**）、ジョブ管理、出力 `outputs/{job_id}/output.mp4`+`metadata.json`、`LTXRunner.generate(...)→GenerationOutcome` 契約、**mock backend**（torch 無し app `.venv` のテスト用・温存）。
- **エンジン層（実証済・Phase 5(A) で取り込み済）**: フォーク `vendor/LTX-Desktop-LOW-VRAM/backend/` の `LTXFastVideoPipeline`。公式 `DistilledPipeline`(ltx_core@00dc53d) をラップし、(a) GGUF transformer + block-swap、(b) **GGUF Q4_K_M Gemma**、(c) VAE/attention tiling を配線。常駐 worker `_ltx_worker.py` 経由で `_RealBackend` から呼ばれる（§3）。

## 2. いま整っている資産（実機確認済）
- **フォーク env**: `vendor/LTX-Desktop-LOW-VRAM/backend/.venv`（LTX-2 を git rev `00dc53d` に pin＝ModelLedger API。torch 2.9.1+cu128 / gguf / 全 import OK）。
- **モデル**:
  - transformer GGUF: `models/ltx-2.3-gguf/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`（17.76GB）
  - **Gemma GGUF**: `models/gemma-3-12b-it-gguf/gemma-3-12b-it-Q4_K_M.gguf`（7.30GB, `ggml-org/gemma-3-12b-it-GGUF`, ungated, arch gemma3, 量子化 {F32,Q4_K,Q6_K}）
  - safetensors（VAE/audio源・projection源・tokenizer/config・vision）: `models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors`, `…-spatial-upscaler-x2-1.1.safetensors`, `models/gemma-3-12b-it-qat/`
- **実装済（フォーク側）**:
  - 新規 `vendor/.../backend/services/gemma_gguf_quant_service.py` — GGUF Q4 Gemma を per-layer dequant で GPU 推論。`gguf_quant_service.py` の bit-exact カーネル再利用＋city96 キーremap＋RMSNorm−1補正＋GGUF Gemma×safetensors projection の merge＋embed CPU offload。
  - `vendor/.../backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py` に `gguf_gemma_path` 引数＋`_install_gemma_gguf`（`cpu_text_encode` と排他）。
- **実装済（我々の backend 側）**: ÷64 契約（`api/models.py`）、`services/ltx_runner.py` の facade＋`_MockBackend`(温存)＋`_RealBackend`(**Phase 5(A) で Approach W 常駐ワーカーへ実体化済＝§3**)、`video_io.crop_mp4`、`config.model.backend`(auto/mock/real)＋`config.model.{gguf_*,fork_*}`。mock pytest 13 passed。

## 3. Phase 5(A)：フォークエンジンを我々の backend へ取り込む

> ## ✅ 完了 (Phase 5A) — 2026-06-28
> **凍結 REST API が実エンジンで生成するようになった。** `services/ltx_runner.py` の `_RealBackend` を
> **Approach W（常駐サブプロセスワーカー）** で実体化。公式 `DistilledPipeline`（本機で native crash）は使わず、
> フォーク `vendor/LTX-Desktop-LOW-VRAM/backend/_ltx_worker.py` を **フォーク env** で常駐起動し、JSON-lines
> プロトコル（`@@LTX@@` フレーム）でジョブを受けてエンジンが `output.mp4` を共有 output dir に直接書く。app `.venv`
> は torch/engine を一切 import しない。**凍結境界は不変・mock pytest 13 passed**。実機検証＝384x256/9 の T2V/I2V
> 完走（VERIFICATION_LOG §6）。実装計画 `~/.claude/plans/a-witty-kazoo.md`。
> **次＝Phase 5(B)（このセクション末尾「次＝Phase 5(B)」を参照）。**
>
> **なぜ Approach W か**（旧ハンドオフの「`services/lowvram/` へコピー」案でも in-process 案でもなく）:
> エンジンは torch＋ltx_core@00dc53d＋gguf を要し、それらは**フォーク env にしか無い**。さらに app と
> フォークは**双方とも `services` という同名トップレベルパッケージ**を持ち、フォーク側 `__init__` が ~18 の
> torch 依存モジュールを eager import する → **同一インタプリタで共存不可**。二プロセス分離なら衝突を完全に
> 回避でき（rename 不要）、**検証済みのフォークコードを無改変で**使える。

**目的**: 我々の FastAPI が、この実証済みパイプラインを実際に配信できるようにする。**凍結層は不変**。

（以下は Phase 5(A) で実際にどう配線したか＝seam の正確な記録。次セッションが実装を追う地図として残す。
詳細は計画書 `a-witty-kazoo.md`、検証は VERIFICATION_LOG §6。）
1. **エンジンの置き場所**: フォークの `services/{fast_video_pipeline, gemma_gguf_quant_service, gguf_quant_service, block_swap_service, ltx_pipeline_common}.py` 等を我々の `services/lowvram/` に取り込む（or フォークを実行基盤として import）。**重要・実行環境**: 我々の backend は **フォーク env(00dc53d) で動かす**必要がある（1.1.6 は ModelLedger 無し＋crash）。足りない依存（pyyaml/pillow/httpx/python-multipart 等）は同 env に追加。gradio は別プロセス（hub 依存衝突回避）。
2. **`services/ltx_runner.py` の `_RealBackend` を作り替え**: 公式 `DistilledPipeline` 直叩き（crash）をやめ、`LTXFastVideoPipeline.create(... gguf_transformer_path=, gguf_gemma_path=, block_swap_blocks_on_gpu=, vae_*_tile_size=)` → `.generate(prompt, seed, height, width, num_frames, frame_rate, images, output_path, num_steps=8)` を呼ぶ。circular import 回避の warm import（`import ltx_core.loader`）を初期化に。
3. **config 追加**: `model.gguf_transformer_path`、`model.gguf_gemma_path`、`vram.block_swap_blocks_on_gpu`、VAE tile sizes 等。`services/low_vram.py` のマッピング更新。`backend` 名は `ltx-distilled-gguf` 等に。
4. **出力**: `LTXFastVideoPipeline.generate(output_path=...)` が mp4 まで書く。`crop_output` 指定時は既存 `video_io.crop_mp4` で後段クロップ。
5. **検証**: ①mock pytest（app `.venv`）が緑のまま ②real smoke 384x256/9 T2V → mp4・peak VRAM フェーズ別・時間 を実測（フォーク env）③phase1_default→最小I2V→phase1_target。落ちたら**勝手に直さず原因分析して報告**。

## 3b. Phase 5 精密ポインタ（backend 凍結契約マップ・2026-06-28 調査済）
次セッションが即着手できるよう、我々の backend 側の正確な契約と差し替え箇所を記す（行番号は目安、シンボルで追うこと）。

**差し替える対象＝`services/ltx_runner.py` の `_RealBackend` のみ**（`load()`/`generate()`/`unload()`, おおよそ L327–500）。現行は公式 `DistilledPipeline` を import・呼び出し（crash する死に筋）。ここを `LTXFastVideoPipeline.create/generate` 呼びに置換。

**変えてはいけない凍結境界**:
- **契約シグネチャ**（`ltx_runner.py`）:
  `LTXRunner.generate(request: GenerateRequest, output_dir: Path, progress_callback: ProgressCallback|None, conditioning_image_paths: list[Path]|None) -> GenerationOutcome`
  - `ProgressCallback = Callable[[int|None,int|None,float],None]`（real は `(None,None,progress)` で良い）
  - `GenerationOutcome(output_path: Path, seed_used: int, peak_vram_mb: int|None, generation_mode: str, backend: str)`。`output_path` は必ず `output_dir/"output.mp4"`。`generation_mode` は `request.generation_mode`（i2v/t2v）。`backend` 文字列は新値（例 `ltx-distilled-gguf`）にして良い。
- **唯一の呼び出し元**＝`services/pipeline_manager.py`（~L135）。`output_dir = config.output_dir/job_id`、`conditioning_image_paths` は `upload_store.path_for(image_id)` で解決済み絶対パス。`metadata.json` は **pipeline_manager 側が `_finalize()` で書く**（runner は `output.mp4` だけ書く）。`outcome.peak_vram_mb` は `low_vram.metadata_block()` に渡る。
- **解像度/フレーム検証**＝`api/models.py`（`GenerateRequest` validator）: **width/height ÷64**、`(num_frames-1)%8==0`、distilled は `num_inference_steps==8`/`guidance_scale==1.0`、I2V は最大1枚・`frame_idx==0`。
- **I2V 型**: `ConditioningImage(image_id, frame_idx=0, strength, crf|None)` → 現行は公式 `ImageConditioningInput(path, frame_idx, strength, crf=33既定)` に変換。フォーク engine 側の画像条件型に合わせて変換し直すこと（フォーク `api_types.ImageConditioningInput`／`generate` の image 引数仕様を次セッションで再確認＝この探索は中断済）。
- **video_io**: `encode_frames_to_mp4(frames, output_path, frame_rate, crop=None,...)`／`crop_mp4(in,out,w,h)`／`save_metadata(path,dict)`。crop は `request.crop_output` 指定時に既存 `crop_mp4` で後段クロップ。
- **テスト固定**: `tests/conftest.py` が `model.backend="mock"` を強制＋`LTX_DISABLE_GRADIO=1`。**mock 経路と pytest は緑のまま**にする（torch 無し app `.venv` で回る）。

**backend 選択 & 可用性**: `_select_backend()` が `config.model.backend`（auto/mock/real）で分岐。`_real_available()` は現状 **`import ltx_pipelines`＋`torch.cuda`＋モデルパス存在**で判定 → **Phase 5 では「フォーク engine が import 可能か＋GGUF パス存在」に更新**が要る（公式 1.1.6 ではなくフォーク env で動かすため）。

**low_vram knob**（`services/low_vram.py` `LowVramSettings`）: 現在 engine に効くのは `low_vram_mode`/`fp8_transformer`/`vae_tiling` のみ。`block_swap`/`attention_tiling`/`cpu_offload_text_encoder` は **定義済みだが未配線** → Phase 5 で `block_swap_blocks_on_gpu` 等として `LTXFastVideoPipeline.create` に渡す配線を追加。

**config 追加（Phase 5）**: `model.gguf_transformer_path`／`model.gguf_gemma_path`／`vram.block_swap_blocks_on_gpu`／VAE tile sizes 等。パス解決は既存 `config._abs`(PROJECT_ROOT 基準) を踏襲。

**フォーク engine 呼び出しの実証済み雛形**: `outputs/phase4_gguf_gemma/run_t2v_bs8.py`（384x256/9 が通った `create(...)+generate(...)` の実値一式）と `vendor/.../backend/_spike_gguf_min.py`。`create()` の全引数は `ltx_fast_video_pipeline.py` 参照（`gguf_transformer_path`/`gguf_gemma_path`/`block_swap_blocks_on_gpu`/`vae_*_tile_size`/`cpu_text_encode`(排他) 等）。Phase 5(A) ではこの呼び出しを **常駐ワーカー `_ltx_worker.py`** に内包した（worker がモデルを1度だけ構築しジョブを使い回す）。

## 3c. （歴史的）Phase 5(B) 起点メモ＝当時の仮説

> **【超過 2026-06-29】このセクションは Phase 5(B) 着手時の仮説メモ（歴史的）。** denoise の shared 溢れは
> **解決済**（§0／VERIFICATION_LOG §7.5）＝真因は当時の想定（block_swap 深度等）ではなく Windows アロケータ居座りで、
> 対策は denoise 直前 `empty_cache`。**現状と次の一手は §0「★残課題サマリ」を参照**。以下は当時の記録。
Phase 5(A) で「配線」は終わったが、**"16GB に本当に収まる" はまだ未達**。denoise フェーズが（384x256 でも）shared メモリへ溢れている（ユーザーがタスクマネージャで目視確認、VERIFICATION_LOG §6/§5 の "denoise shared ~3.6GB @bs8" と整合）。WDDM が溢れ分を system RAM へページングするため**ハード OOM はしないが激遅化**する。これが残った主タスク。

- **ユーザー方針＝これは「Phase 5(B) スケールアップ（1280×768 → crop 720p）」と一つの同じ仕事**。スケールアップ用の denoise-stage VRAM 技法（block_swap 深度を bs4/bs2 へ／VAE・attention タイリング／解像度依存の sequential・streaming）を適用すれば、**現状の 384x256 の shared 溢れも一緒に解消**する見込み。
- **着手の起点＝先行事例の denoise-stage 技法を調査・複製**（独自発明しない、[[research-prior-art-first]]）: フォークの `vendor/.../backend/services/block_swap_service.py` 等、および ComfyUI-GGUF / ComfyUI カスタムノード・ワークフローの解像度対応技法。これらは**まだ我々の backend に反映していない**。
- 計測は perf-counter の "Shared Usage" サンプリングで（§5。worker の `max_memory_allocated` では溢れを検知できない＝§4）。
- **重要・過去ログの読み替え**: VERIFICATION_LOG §2.3 で **bs4 が crash** したのは、当時 Gemma が bf16 で ~17.7GB を共有メモリへ溢れさせ、その圧迫下で遅延ローダが access violation した（バグ#5）ため。**この前提は Phase 4/5(A) の GGUF Q4 Gemma 化で既に解消済**＝「bs4 は落ちる」は**もう当てはまらない**。よって Phase 5(B) は bs8→bs6→bs4→bs2 を**クリーンに再計測**してよい（各深度で dedicated/shared 両ピーク＋wall-clock を §5 のサンプラで測り、溢れ消失と速度のバランス点を探す）。

## 4. （超過）旧・既知の残課題リスト

> **【超過 2026-06-29】この §4 は古い。現行の残課題は冒頭「★残課題サマリ」（A=worker 再利用 crash／B=load/encode
> 一時溢れ／C=720p スケールアップ／D=後回し）と VERIFICATION_LOG §7 を正とする。** 以下は歴史的参照（denoise spill は解決済）。
- **★【最優先・真の 16GB fit】denoise フェーズの shared 溢れ**。これはもはや「block_swap の小ノブ」の後回し課題ではなく、**残った主タスク**であり、上記 **Phase 5(B) スケールアップ（1280×768→crop）と同一の作業**（§3c）。384x256 でもユーザーがタスクマネージャで shared 溢れを目視確認済。**注意: worker が報告する `torch.cuda.max_memory_allocated`（T2V 16913 / I2V 17989 MB）は dedicated と WDDM shared を区別できず、この溢れを検知できない** → 検知には perf-counter の "Shared Usage" 直接サンプリングが要る（§5）。フォーク `block_swap_service.py` 等・ComfyUI-GGUF の denoise-stage 技法を複製して詰める。
- **品質バンプ（任意・将来）**: Gemma を Q6_K に上げると忠実度↑（KL的に Q4 の3倍正確）。同じローダで GGUF 差し替え1つ、encode フェーズに余裕あり（Q6 ~9.6GB→encode ~12GB）。今は実績ある Q4 で完走しているので必要時に。
- **未使用経路の地雷**: フォークの音声/IC-LoRA/dev-HQ は未検証。使う時に個別検証（[[ltx-desktop-lowvram-fork]] 留意点参照）。

## 5. 検証・計測の道具
- **★VRAM 実測（テスト担当サブエージェントへの必須手順）**: 実生成の検証では **専用GPUメモリ(dedicated)だけでなく共有GPUメモリ(WDDM shared)も必ず監視する**。`torch.cuda.max_memory_allocated`（＝worker 報告 `peak_vram_mb`）は dedicated/shared を区別できず **WDDM 共有溢れを見逃す**（Phase 5(A) で「16GBに収まった」と誤判定した原因＝§4/§6.4）。
  - **使う道具＝既存サンプラ `vendor/LTX-Desktop-LOW-VRAM/backend/_gpu_mem_sampler.ps1`**（タスクマネージャと同一ソースの perf-counter `\GPU Adapter Memory(*)\Dedicated Usage` と `\…\Shared Usage` を全アダプタ最大で既定0.5s間隔サンプリング→ `timestamp,dedicated_MB,shared_MB` を CSV 出力。引数 `-OutFile/-IntervalSec/-MaxSeconds`）。
  - **手順**: 生成の直前に**バックグラウンド起動**（例 `powershell -ExecutionPolicy Bypass -File <repo>/vendor/LTX-Desktop-LOW-VRAM/backend/_gpu_mem_sampler.ps1 -OutFile outputs/gpu_mem_<tag>.log -IntervalSec 0.5`）→ 生成後に停止 → ログの **dedicated_MB と shared_MB の両ピークを必ず報告**し、worker のフェーズ print（Gemma encode / denoise / VAE）と wall-clock で突き合わせて溢れフェーズを特定。`torch` 側の `max_memory_allocated` は補助としてのみ併用。過去ログ例: `outputs/gpu_mem_*.log`（§2.2 / §2.4 に読み方）。
- Gemma 単体の正しさ検証スクリプト: `outputs/verify_gemma_gguf/`（dequant bit-exact / キー一致 / norm−1 / forward 49層有限 / bf16 cosine）。
- E2E スパイク: `outputs/phase4_gguf_gemma/`（runner・perf log・出力 mp4・contact_grid・GIF）。フォーク最小呼び出し雛形 `vendor/.../backend/_spike_gguf_min.py`。

## 6. ドキュメントの正本/古い注意（重要）
> **【更新 2026-07-01】この §6 は Phase 5(A) 当時の版。最新の「正本」ポインタは以下に更新済:**
> - **現状の正本＝本書冒頭「現状ステータス」節＋「次の一手メニュー」**（この §6 より上位）。
> - **技術検証の正本＝[VERIFICATION_LOG.md](VERIFICATION_LOG.md)**：最新は **§14（QAT 回収・text-only Gemma）**／§13（de-fork）／
>   §12（dit-cpu-load）／§11（te-offload）／§10（720p・連続生成）／§9.7（reuse-crash 解決）。
> - **README.md / `LTX23_Backend_Specification_v04…md` は post-refactor 反映済**（de-fork Stage 5＋本セッションの QAT 是正）＝
>   下記「古い」欄の「pre-pivot のまま」はもう当てはまらない（章立て全面改訂だけが未了＝メニュー候補）。
>
> 以下（旧）は Phase 5(A) 当時の記述。歴史参照用に温存。
- **正本**: 本書（handoff）／[VERIFICATION_LOG.md](VERIFICATION_LOG.md)（**§6＝Phase 5(A) 配線・実機検証が最新**、§5＝Gemma GGUF）／計画書 `a-witty-kazoo.md`／memory。本書と VERIFICATION_LOG は **Phase 5(A) の状態を反映済**。
- **LTX 2.3 一般リファレンス（参照URL付き基礎知識）**: [LTX23_REFERENCE.md](LTX23_REFERENCE.md) — 解像度契約(÷32/÷64・2段)・VAE 32×圧縮とトークン数・VRAMスケーリング(重み支配)・16GBレシピ・720pの作り方(1280×768→crop)。タスク非依存の事実集。
- **古い・歴史的（鵜呑み禁止）**:
  - [DESIGN_COMPARISON_and_direction.md](DESIGN_COMPARISON_and_direction.md): fp4_mixed 推奨だったが**不採用**（ComfyUI密結合/cu130前提）。GGUF Q4 採用が結論。比較表の枠組みは有効。
  - [note.md](note.md): 公式 offload+fp8 **戦略は破棄**（本機で native crash）。ただし torch2.9.1/cu128・SDPA(=FlashAttn-2)・xformers任意 の**事実は有効**。
  - `README.md` / `LTX23_Backend_Specification_v04_…md`: **pre-pivot のまま**（公式 DistilledPipeline＋fp8-cast＋xformers、ランナー=モック、一部 ÷32 表記）。**現行の正しい解像度契約は ÷64**（`api/models.py`）。engine の真実は本書/VERIFICATION_LOG。README/spec の全面改訂は **Phase 5(B) 完了後まで保留**（VRAM の真の 16GB fit が実現＝実エンジンの実像が確定してから）。Phase 5(A) で配線は済んだが、VRAM fit が未達のうちは実像が未確定のため。

## 7. 作業原則（ユーザー）
- コード着手前に**実装計画を提示して合意**（plan mode 段階承認）。**テストが落ちたら勝手に直さず原因分析して報告**。
- 編集・テスト実行・生成・DL は**サブエージェントに委譲し、本体は監督**。環境隔離厳守（システム Python を汚さない／全てプロジェクト内 `.venv`/`.python`/`hf_home`/`.uv_cache`）。
- **先行事例のソースを複製し独自発明しない**（[[research-prior-art-first]]）。大きな pivot は相談してから。
