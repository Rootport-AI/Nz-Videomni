# IC-LoRA Phase A スパイク — 現状ステータス（成立・**✅全消化済み＝歴史記録**）

> **⚠️ 2026-07-03 更新: 本書のPending項目はすべて消化済み**（mainマージ=`a578c83` 実行済・Phase B本実装=branch `feature/ic-lora-phase-b` で完了・目視も全消化）。**最新の正本は [`IC_LORA_PHASE_B_STATUS.md`](IC_LORA_PHASE_B_STATUS.md) と [`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md) 冒頭ブロック**。本書はPhase Aスパイクの技術記録として温存。

- 更新: 2026-07-03（branch `feature/ic-lora-phase-a`・[`PHASE3_NEXT_WORK_SURVEY.md`](PHASE3_NEXT_WORK_SURVEY.md) を受けた同日実装）
- 併読: [`VERIFICATION_LOG.md` §20](VERIFICATION_LOG.md)（本スパイクの全ゲート詳細数値）／[`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md)（引き継ぎ）／[`PHASE3_NEXT_WORK_SURVEY.md`](PHASE3_NEXT_WORK_SURVEY.md)（着手前サーベイ・入口Doc）

---

## ⚠️ 一行結論

**成立**: IC-LoRA（Lightricks公式 Pixel-Spatial-Upscaler x2 アダプタ）をGGUF Q4_K_M本番経路の**bf16サブパス**にfuse-at-load配線し、参照動画条件付け（`VideoConditionByReferenceLatent`）と合わせて実機スパイクをPASSさせた。回帰（LoRA off時のper-layer-quant本番経路）はbyte-match完全一致・pytest 41 green。**唯一の未消化「作業」はmainへのマージ実行**（branch `feature/ic-lora-phase-a` は main 未マージ・4コミット先行・次セッション冒頭でユーザー一言確認して実行）。目視サインオフ（spike.mp4 vs base.mp4）は**fix-later方針でユーザー承認済み**＝非ブロッカー。

## 背景・スコープ

- 仕様 `LTX23_Backend_Specification.md` §13.4b は IC-LoRA をLTX-Desktopパリティ対象外＝Phase 4としていたが、ユーザー最優先関心のため本スパイクとして前倒し着手（ユーザー承認済み・[`PHASE3_NEXT_WORK_SURVEY.md`](PHASE3_NEXT_WORK_SURVEY.md) §4 参照）。
- 最初に載せるアダプタ＝**Pixel-Spatial-Upscaler**（サーベイ推奨どおり）。x2/x4のうち本スパイクは**x2**を使用。
- 目的は「研究ファースト・段階導入」の最初のステップ＝**spike**（LoRAキー形式確認→fuse-at-load配線→RAM/VRAM/トークン数/生成時間の計測→LoRA無効時のbyte-match完全一致確認）。基本配線（`loras`パラメータのAPI露出）とPhase Bの本実装はスコープ外。

## 実行環境

i7-13700（**AVX2のみ・AVX512-BF16/AMX無し**）／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`。

## 何を作ったか

- **`engine/gguf/loader_service.py`**:
  - (a-0) bf16パスのdequantを`quant_service`の忠実カーネルへ委譲するよう変更（旧・簡易実装のQ4_K/Q6_Kカーネルは数値的に誤りだった。合成Q8_0/F32での等価性検証＝ALL_OK）。
  - `GGUFStateDictLoader.ic_loras`（`(path, strength)` のリスト）→ `_fuse_ic_loras`: wheel の `SafetensorsStateDictLoader` ＋ `LTXV_LORA_COMFY_RENAMING_MAP` でLoRA safetensorsをロードし、キーごとに**fp32行列積でdelta fuseをin-place適用**（数学的には wheel の `_fuse_delta_with_bfloat16` と同一・丸め回数が1回少ないのみ。合成データでfp64参照比bf16-ULPオーダーと検証済み）。マッチしたdeltaが0件なら大声でWARN。
  - **wheelの`transformer_builder.loras`は使わない**（使うとwheelがpathを無視する我々のローダー経由でGGUFを再読込してしまう既知の落とし穴・ドキュメント化済み）。
- **`engine/pipeline/fast_video_pipeline.py`**: `ic_loras`/`ic_reference`をキーワード専用引数として追加（デフォルトは不活性）。以下をfail-loudにフェイルする: LoRA指定＋`per_layer_quant=True`の組み合わせ、LoRAインストール失敗（safetensorsへの無言フォールバックなし）、`reference_downscale_factor<=1`。`VideoConditionByReferenceLatent`をstage1のみに既存のhybrid-conditioning monkeypatch経由で追加。ステージ判別＝`cond height == full_height//2`（ステージ間で異なる唯一のkwarg）。
- **ハーネス `outputs/ic_lora_phaseA/run_spike.py`**（untracked・`outputs/`はgitignore対象）: モード `parity`/`base`/`spike`/`toggle`。psutilでRSSサンプリング、ステージごとのVRAMマーク、`VideoConditionByReferenceLatent.apply_to`のmonkeypatchでトークン計測。`device_supports_fp8`をFalseにパッチし純bf16計測に固定。

## アダプタ

- HF `Lightricks/LTX-2.3-22b-IC-LoRA-Pixel-Spatial-Upscaler`（gated=auto・ユーザーがライセンス承諾済み）。
- 配置: `models/ltx-2.3-ic-lora/pixel-spatial-upscaler/`、x2/x4とも 654,465,286 bytes。
- x2メタデータ検証済み: `reference_downscale_factor=2`・`reference_spatial_scale_factor=2`・`model_version=2.3`。960キー＝lora_A 480＋lora_B 480、すべて`diffusion_model.`プレフィックス、BF16、rank 64＝wheelの`apply_loras`期待値＋COMFYリネームマップと厳密一致。

## commit（branch `feature/ic-lora-phase-a`）

| commit | 内容 |
|---|---|
| `bbcd82f` | feat(engine): spike wiring — bf16-path faithful dequant + engine-side LoRA fuse + reference conditioning |
| `016f442` | perf(engine): fuse IC-LoRA deltas in fp32 — 19min→33s on AVX2-only CPUs |

## ゲート結果サマリ（詳細数値は VERIFICATION_LOG §20）

| ゲート | 内容 | 結果 |
|---|---|---|
| 1. 回帰byte-match | 本番per-layer経路・LoRA off | **PASS**（T2V/I2V sha完全一致・peak_vram 8440MB不変・pytest 41 passed） |
| 2. パリティ（a-0検証） | per_layer_quant True/False・LoRA無し | **PASS**（SHA256完全一致）。bf16パスはTrue経路の約3倍遅・VRAM+3.6GB |
| 3. スパイク本番実行 | x2 LoRA fuse＋参照条件付け・1024x640/25f | **COMPLETED**（VRAM溢れなし・トークン+25%を確認） |
| 4. トグル（on/off切替コスト） | keep_resident=0前提の全リビルド | **PASS**（fp32修正後 load+fuse=104s・うちfuse≈33s。no_lora出力=base.mp4と完全同一SHA＝非汚染性の証明） |
| 5. タイミング異常の根本原因調査 | Web検証 | AVX512-BF16/AMX非搭載CPUでのtorch CPU bf16 matmul fallbackが原因と確認 |

## 未了・Phase Bへの持ち越し（詳細は挙げるのみ・計画はPhase B側）

- 32GB RAM級マシン向けの本番機構: bf16 full-dequant fuseはスパイク専用（RAM 54-57GB消費）。per-layer-quant経路＋GPU forward-time LoRA適用（ComfyUI実証パターン）か、事前fuse済みチェックポイント派生か、要選定。
- wheelの`ICLoraPipeline`をoracleとした公式パリティ照合（公式はstage1のみLoRA適用・我々は単一ledgerゆえ両ステージにfuse＝挙動差分は文書化済みだがoracle未照合）。
- keep-resident運用との整合: in-place fuseがキャッシュ済みbaseを変異させるため、`StateDictRegistry`下でのLoRAトグルは設計要（リビルド vs デュアルキャッシュ）。
- API/UI露出（`engine/api_types.py`のIcLoraスキーマは存在するが未配線）。
- x4バリアント・他アダプタ（In-Outpainting/Deblur、`PHASE3_NEXT_WORK_SURVEY.md` §6準拠）。
- **ユーザー目視ゲート**: spike.mp4 vs base.mp4（送付済み・回答PENDING）＋前セッションから持ち越しの目視4本（`NEXT_SESSION_HANDOFF.md`参照）。

## Pending 項目（次セッション/ユーザー）

1. **mainマージ**（ユーザー一言確認→実行。branch `feature/ic-lora-phase-a` は4コミット先行・未マージ）。
2. Phase B着手（承認済み・新セッションで。上記「未了」項目からのスコープ確定）。
3. 目視（fix-later承認済み・非ブロッカー）: `outputs/ic_lora_phaseA/spike.mp4` vs `outputs/ic_lora_phaseA/base.mp4` の比較＋前セッション持ち越しの目視4本（`NEXT_SESSION_HANDOFF.md`参照）。
