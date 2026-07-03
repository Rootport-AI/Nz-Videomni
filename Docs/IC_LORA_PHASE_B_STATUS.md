# IC-LoRA Phase B — 現状ステータス（成立・全ゲートG1〜G5 PASS・残作業=mainマージ判断のみ）

- 更新: 2026-07-03（branch `feature/ic-lora-phase-b`・base=main merge `a578c83`）
- 併読: [`VERIFICATION_LOG.md` §21](VERIFICATION_LOG.md)（全ゲート詳細数値）／[`IC_LORA_PHASE_B_WORKORDER.md`](IC_LORA_PHASE_B_WORKORDER.md)（設計・リサーチ根拠）／[`IC_LORA_PHASE_A_STATUS.md`](IC_LORA_PHASE_A_STATUS.md)（前段スパイク）

---

## ⚠️ 一行結論

**成立**: IC-LoRAの本実装＝**per-layer-quant本番経路でのforward時GPU LoRA適用**（ComfyUI-GGUF実証の「dequant時ウェイトパッチ」方式）＋**API露出**（`loras`パラメータ＋`POST /upload/video`）が完成し、全ゲートG1〜G5 PASS。Phase Aのbf16融合ペナルティ（RAM 54-57GB・約3倍遅・VRAM+3.6GB）は本経路では**すべて解消**（VRAMピークはLoRA無しと同一の8440MB・attach 0.02〜0.3秒）。bf16融合経路はユーザー指示どおり選択可能なまま温存。**残る「作業」はmainへのマージ判断のみ**。

## 何ができるようになったか

- **本番経路（GGUF Q4_K_M per-layer-quant・32GB RAM級で動く構成）でIC-LoRAが使える**。ベース量子化バイトは決して変異しない（deltaは毎forwardの使い捨てdequantテンソルへfp32計算・1回キャストで加算）→ LoRA off時は既存経路と完全同一・`StateDictRegistry`キャッシュ汚染も構造的に無い。
- **ジョブ毎のLoRA切替**: `generate(ic_loras=…, ic_reference=…)`（明示`[]`=クリーンdetach）。再ロード不要（Phase Aのfuse方式では原理的に不可能だった）。
- **REST APIから利用可能**: `POST /api/v1/upload/video`で参照動画→`video_id`、`POST /generate`に `loras:[{name:"pixel-spatial-upscaler-x2", strength:1.0}]`＋`reference_video_id`。アダプタ名は`config.yaml` `model.ic_loras`のサーバー側レジストリで解決（クライアントからパス指定は不可・fail-loud 404）。凍結API契約は純加算のみで不変。

## ゲート結果サマリ（詳細＝VERIFICATION_LOG §21）

| ゲート | 内容 | 結果 |
|---|---|---|
| G1 | 回帰byte-match（LoRA off・本番per-layer経路） | **PASS**（T2V/I2V基準SHA完全一致・peak_vram 8440不変・pytest 58 passed/1 skipped） |
| G2 | 新経路 vs bf16融合のspike照合 | **PASS**（**byte完全一致** `735a6de9…272`＝Phase B基準SHAに再ピン。旧`spike.mp4`との不一致は`016f442`以前のstale baselineと根本原因特定・480層delta CPU/GPU 0 ULP一致） |
| G3 | 非汚染トグル（同一プロセス lora→なし→lora） | **PASS**（「なし」=base.mp4完全一致・4経路すべて`735a6de9…`に収束） |
| G4 | VRAM/速度 | **PASS**（全体ピーク8440.9MB=LoRA無しと同一・bf16融合比2.2倍速/denoise−3.4GB・attach 0.02–0.3s。**rank64のdenoise増は実測+20〜26%**＝許容judged） |
| G5 | API e2e実機（upload→generate→取得） | **PASS**（115.8s完走・**API経路出力もbyte一致** `735a6de9…`・metadata `ic_lora`ブロック・凍結`/status`不変・偽video_id→404） |

## commit（branch `feature/ic-lora-phase-b`）

| commit | 内容 |
|---|---|
| `b805ae1` | feat(engine): forward-time GPU LoRA on per-layer-quant path（`ic_lora_common.py`新設・`ggml_linear_forward`パッチ・provider・generate毎切替） |
| `fbef799` | feat(api): loras param + POST /upload/video（レジストリ・video store・5ホップ配線・mock対応・テスト+11） |

## アーキテクチャ要点（Phase C以降の担当者向け）

- 機構＝**dequant時ウェイトパッチ**（ComfyUI-GGUFと同型・Web/ソース裏取り済み）。delta式はPhase A fuse（`loader_service._fuse_ic_loras`）と**演算順序まで同一**に保ってあり、これがbyte-parity検証（G2）の根拠。**勝手に「最適化」しないこと**。
- A/B因子は対象`nn.Linear`の**非persistentバッファ**（block-swapの`.to()`移動に自動相乗り・state_dict非汚染）。attach/detachは`engine/gguf/ic_lora_common.py`。
- bf16融合経路（`gguf_per_layer_quant=False`）は開発/パリティ照合用に温存（選択スイッチ＝config/loadのper_layer_quant）。
- 新アダプタの追加＝`config.yaml` `model.ic_loras`に1行（＋参照動画が要るかの検証要件確認）。x4はファイル配置済み・未登録（スコープ外）。

## 未了・Phase C候補（詳細=VERIFICATION_LOG §21.8）

keep_resident=1でのトグル検証／`ICLoraPipeline` oracle照合／x4・他アダプタ登録／denoise+20〜26%の最適化（必要時）／参照不要アダプタ向けのバリデーション緩和／Gradio UI露出。

## Pending（ユーザー）

1. **mainマージ判断**（branch `feature/ic-lora-phase-b`・2コミット先行・全ゲートPASS）。
2. main の push（Phase Aマージ `a578c83` 以降ローカル先行・監督のpushは権限拒否のため手動）。
3. 目視（fix-later承認済み・非ブロッカー）: Phase A持ち越し5本＋Phase B出力（`outputs/ic_lora_phaseA/phaseB/api_smoke.mp4`＝API経由のx2アップスケール結果）。
4. HFトークン無効化（前セッションからの宿題）。
