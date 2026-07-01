# 設計比較と次セッション方針 — 現状コード vs LTX-Desktop-LOW-VRAM fork

> # ⚠️ SUPERSEDED (2026-07-01)
> **本文は de-fork リファクタ*前*の比較・方針記録。** フォークエンジンの採用/Phase B は既に**完了**した
> （低VRAMエンジンを first-party `engine/` へ自前化・`vendor/LTX-Desktop-LOW-VRAM` 削除・merge `a857a3b`）。
> 以下 §1–7 は当時の検討記録であり、**現行の作業指示ではない**（「fork engine を採用し Phase B を完遂せよ」は
> 読み違い＝もう済んでいる）。現状は `README.md` / `Docs/NEXT_SESSION_HANDOFF.md` / `Docs/QAT_RECLAMATION_RESEARCH.md`
> を参照。本文は歴史記録として温存する。

> ⚠️ **状態（2026-06-28 後日追記）**: 本書の「§5 推奨＝fp4_mixed Gemma」は**不採用**となった。fp4_mixed は ComfyUI 密結合
> （`comfy-kitchen` 必須・高速 FP4 カーネルは torch cu130+ 前提、本機は cu128）と判明し、代わりに **GGUF Q4_K_M Gemma を
> 我々の bit-exact dequant エンジンで GPU 推論**する方式を採用→**16GB E2E 達成済み**。最新の正本は
> [NEXT_SESSION_HANDOFF.md](NEXT_SESSION_HANDOFF.md) と [VERIFICATION_LOG.md](VERIFICATION_LOG.md) §5。
> 本書の比較表（現状コード vs fork の層分け）と「engine 採用は必須」という結論は引き続き有効。

作成: 2026-06-28 / 目的: 「現状のカスタムコードを残すか、fork に寄せて再設計するか」の判断材料。
このセッションは**分析・考察のみ**（実装は次セッション）。詳細な実機検証は [VERIFICATION_LOG.md](VERIFICATION_LOG.md)。

## 1. 比較表（現状コード vs fork）

| 軸 | 現状の我々のコード | LTX-Desktop-LOW-VRAM fork (Kandyman) |
|---|---|---|
| **engine** | 公式 `DistilledPipeline` 直叩き（薄いアダプタ） | `LTXFastVideoPipeline`(公式ラップ)＋`DistilledNativePipeline`(手組み) |
| **T2V / 最小I2V** | 配線済だが **real は未実行**※ | 実装済・**bs8 で有意映像を実生成（実証済）** |
| フル I2V（複数条件画像/任意frame_idx） | API契約でブロック | 実装済 |
| **音声生成** | **未対応**（`_audio` を破棄） | ★**LTX-2 native joint audio 実装済**（+外部TTS/Foley, 多くWSL依存） |
| **IC-LoRA** | **未対応**（`loras=[]` 固定） | ★**実装済**（canny/depth/pose 条件抽出＋pipeline） |
| two-stage HQ | enum受けのみ・未配線 | dev pipeline 実装済（ただし後述 16GB 不適） |
| upscaler / prompt enhance | 暗黙ON / OFF固定 | 実装済 / 複数経路 |
| **16GB transformer** | 公式 `OffloadMode.CPU`+fp8 依存（**fork の block_swap 等は未ポート**） | **GGUF per-layer + block_swap + attn/VAE tiling**（我々が dequant 4件＋loader を修正・実証） |
| **16GB Gemma** | 公式委譲（丸ごとGPU=溢れる） | CPU bf16(=この機で地雷)/fp8-cast。**FP4量子化ロード経路なし** |
| **実際に動くか** | ★**real 未実行**（venv に ltx_pipelines 未install→mock fallback） | ★**実生成で有意映像**（我々の de-risk 成果は全てここに存在） |
| API/job/output 層 | ★**きれいな凍結契約**（AviUtl2/Resolve 向け・÷64・ポータブル） | アプリ寄り handler 群（WSL 依存多・凍結契約なし） |
| 保守性・脆さ | 薄く小さい（公式依存・カスタム少） | 低VRAM中核**テスト皆無**／GGUF dequant 誤実装が2系統併存（zero-tensor fallback含む）／ltx_core 内部APIへの**monkeypatch 密結合**／Windows crash 場当たり対策多数 |

※現状コードの real 経路は **一度も実行されていない**（ltx_pipelines が venv 未install、`_real_available()` が False→mock）。「16GB で動いた」のは**全て vendor の fork 側**での実証。

## 2. 重要な気づき（二択の前提がズレている）

- **両者は競合ではなく“層”が違う**。我々の**きれいな凍結API/job/output/config 層**は資産（残すべき）。fork は**実際に動くengine＋将来必要な音声・IC-LoRA**を持つ。
- 当初計画の **Phase B＝「fork engine を我々の API の裏に移植」** が、まさにこの統合だった。つまり進むべきは「全カスタム維持」でも「ComfyUI へ破壊的再設計」でもなく、**我々のAPI ＋ fork engine（我々の修正込み）＋ FP4-Gemma（コミュニティ手法）** という、**当初意図の完遂**。
- 現状コードの real 経路（公式 DistilledPipeline）は、**この16GB機では transformer safetensors ロードで crash する死に筋**。fork の GGUF がそれを回避。→ **fork engine の採用は事実上必須**。

## 3. 16GB 先行事例の確定レシピ（コミュニティ標準）

- **transformer**: fp8 safetensors（25GB）＋ ComfyUI Sequential Offloading＋`--reserve-vram`、**または** GGUF（Q3_K_M 14.7GB / Q4_K_M 17.8GB）。我々は後者(GGUF+block_swap)で実証済。
- **★Gemma = `gemma_3_12B_it_fp4_mixed.safetensors`(9.5GB, ~90%FP4) を GPU で**。これが**16-24GB カードの標準**。RTX 4070 Ti SUPER の実走報告(note.com)も**この FP4 ファイル使用**で video+audio 成功。
  - ComfyUI では `LTXAVTextEncoderLoader` ノードで単一ファイルとしてロード（公式HF/Gemmaローダは ComfyUI で壊れていると報告 #106）。**別途 text projection** `ltx-2.3_text_projection_bf16.safetensors` が要る。
  - CPU encode は「数秒遅いが OOM 回避」の**フォールバック**として存在（我々が試した道＝主流ではない）。
- **16GB の本質は「量子化＋逐次オフロード」**。FP4 Gemma 単体では不十分で、**量子化transformer＋逐次ロード/オフロード＋システムRAM≥32GB** の組み合わせで成立。

## 4. ユーザー仮説の判定（FP4 Gemma GPU で局所修正・破壊的再設計不要）

**支持される。ただし1点のニュアンス。**
- FP4 Gemma を GPU に載せるのは**確定で標準解**。テキストエンコーダ以降（transformer denoise→VAE）は我々も実証済なので、**つまずきは「Gemma をどう載せるか」一点**で、そこに標準解がある＝**破壊的再設計は不要**。
- ニュアンス: 「FP4 Gemma に替えるだけ」では完結せず、**逐次ロード/オフロードの枠組みの一部**として効く。ただし我々の fork engine は既に transformer 側の逐次機構（GGUF+block_swap）と Gemma の encode 後解放を持つので、**残る新規実装は「FP4 Gemma のロード経路」だけ**＝局所的。
- **fork に FP4 Gemma 経路は無い**ので、そこは**コミュニティ手法を複製**して新設する（独自発明しない）。

## 5. 推奨：次セッションの方針（案）

1. **engine 採用**: 現状の real 経路（公式 DistilledPipeline）を捨て、**fork の `LTXFastVideoPipeline`（我々の dequant/loader 修正込み）を我々の凍結APIの裏に据える**（当初 Phase B の完遂）。`services/lowvram/` へ取り込み or fork を実行基盤に。
2. **Gemma = FP4 GPU**（CPU bf16 を撤回）: `gemma_3_12B_it_fp4_mixed.safetensors`＋`ltx-2.3_text_projection_bf16.safetensors` を DL し、**ComfyUI `LTXAVTextEncoderLoader` がどう読むかを精読して複製**、ltx_core の text_encoder 経路に差し込む。← 次セッション最初の調査ポイント（fp4_mixed が HF/torchao で読めるか、ComfyUI 専用フォーマットか）。
3. **16GB 収容の最終確認**: Gemma(9.5GB GPU・encode後解放)→ transformer denoise を block_swap 深度で詰める。T2V で完走・有意映像・peak<16GB を実測。
4. **将来**: 音声(native joint audio)・IC-LoRA は fork に土台があるので、API/契約を段階拡張して露出。

## 6. 留意（fork の脆さ）
fork の低VRAM中核はテスト皆無・monkeypatch 密結合・dequant 誤実装併存。**我々が実際に使う経路（per-layer dequant・block_swap・distilled fast・sft_loader patch）は既に de-risk 済**だが、音声/IC-LoRA/dev-HQ など未使用経路には地雷が残る前提で、使う時に個別検証する。dev(HQ) は cpu_text_encode 非対応で 16GB 不適（FP4 Gemma 化で改善余地はあるが要検証）。

## 7. 参照URL（次セッションで精読すべき先行事例）

### ユーザー提示（リポジトリ立ち上げ意図の原典系）
- https://wavespeed.ai/blog/posts/ltx-2-3-comfyui-setup-two-stage-pipeline/ — 2段パイプライン構成・CPU-encode フォールバック・sequential offload
- https://note.com/automate_nahito/n/n3acd161c7d98?hl=en — ★**RTX 4070 Ti SUPER 実走報告（最重要・一次）**。FP4 Gemma 使用で video+audio 成功・`--reserve-vram 5`
- https://github.com/RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management — FFN チャンク分割（長尺向け・24GB対象）。**Gemma管理ではない**＝今回の主目的とは直交

### 調査で発見（16GB レシピ／FP4 Gemma の一次）
- https://ltxworkflow.com/models — 「16GB Best」モデルセットとファイル名・サイズ（fp8 transformer 25.2GB / `gemma_3_12B_it_fp4_mixed.safetensors` 9.5GB / `ltx-2.3_text_projection_bf16.safetensors` / TAE VAE）
- https://docs.comfy.org/tutorials/video/ltx/ltx-2-3 — 公式 ComfyUI チュートリアル（**fp4_mixed Gemma 推奨**）
- https://github.com/Lightricks/LTX-2/issues/106 — ★**`LTXAVTextEncoderLoader` で FP4 Gemma を単一ファイルロード**（公式 HF/Gemma ローダは ComfyUI で壊れていると報告）。次セッションのGemmaロード複製の起点
- https://github.com/Lightricks/ComfyUI-LTXVideo — 公式 ComfyUI ノード（`gemma_encoder.py`・テキストエンコーダローダの実装を精読）
- https://github.com/city96/ComfyUI-GGUF — GGUF ローダ（PR #402 で Gemma-3 GGUF 対応）。GGUF Gemma を採る場合の参照
- https://github.com/wildminder/awesome-ltx2 — GGUF transformer 量子化サイズ（Q3_K_M 14.7GB / Q4_K_M 17.8GB）
- https://github.com/Lightricks/ComfyUI-LTXVideo/issues/303 — FP4 以前（フル bf16 Gemma で OOM）の文脈
- https://github.com/huggingface/transformers/issues/36822 — Gemma-3 で fp16 は空出力バグ（CPU/dtype 注意点）

### 量子化 Gemma 重みの所在（次セッションで DL 先を確定）
- `gemma_3_12B_it_fp4_mixed.safetensors`（~9.5GB, ~90%FP4）＋ `ltx-2.3_text_projection_bf16.safetensors` … 配布元 HF リポジトリは ltxworkflow.com/models／公式チュートリアル経由で要特定（**fp8 代替**: GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn）。
- **次セッション初手の調査ポイント**: fp4_mixed が HF/torchao で素直に読めるか、ComfyUI 専用ロード（`LTXAVTextEncoderLoader`）が前提か。後者なら ComfyUI-LTXVideo のロードコードを複製して ltx_core 経路へ差し込む。
