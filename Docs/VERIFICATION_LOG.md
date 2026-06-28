# 検証ログ — LTX 2.3 を 16GB VRAM で動かす（フォーク移植 Path A）

> 目的: 作業中に行った検証の結果（バグ・修正・実測値・未解決点）を**時系列で累積**する生きた記録。
> 切りのいいタイミングで更新する。設計根拠は [note.md](note.md)、引継ぎは [NEXT_SESSION_HANDOFF.md](NEXT_SESSION_HANDOFF.md)、
> env構築は [../outputs/forkenv_build_log.md](../outputs/forkenv_build_log.md)、計画は `~/.claude/plans/fuzzy-inventing-key.md`。

本機: RTX 4070 Ti SUPER (16GB) / System RAM 63.8GB / Windows 11 / torch 2.9.1+cu128。

---

## 1. フォーク低VRAMエンジンの検証（de-risk スパイク, 2026-06-28〜）

実証の乗り物 = フォークの `LTXFastVideoPipeline`（`vendor/LTX-Desktop-LOW-VRAM/backend/services/`）。
GGUF per-layer quant + block-swap + VAE tiling を公式 `DistilledPipeline` に配線。GGUF = Q4_K_M(17.76GB)。

### 1.1 発見したバグと修正（全て `services/gguf_quant_service.py`）

フォークの **per-layer GGUF dequant 経路に集中**（フォーク作者が最もテストしていない箇所と推定。
block-swap / パイプライン組み立ては健全）。

| # | バグ | 真因 | 修正 | 状態 |
|---|---|---|---|---|
| 1 | `apply_sd_ops` import 失敗→raw key フォールバック | rev 00dc53d に `apply_sd_ops` 不在（`SDOps.apply_to_key/apply_to_key_value` のみ） | デッド remap ブランチ削除。GGUF raw key は全4186モデルキーと完全一致（余分258は audio経路で無害）と meta LTXModel 照合で検証済 | ✅ 適用済 |
| 2-A | GGUF重みが一切ロードされず meta float16 のまま | `weight` buffer を `persistent=False` で登録 → `load_state_dict` から除外 | `persistent=True`（検証: "All keys matched"） | ✅ 適用済 |
| 2-B(.to) | `_float_shape` AttributeError | `GGMLQuantizedTensor` subclass が torch演算で型は保つが Python属性を剥がす。`.to()` override が属性無しで返却 | moved に `_ggml_type`/`_float_shape` 再付与 | ✅ 適用済 |
| 2-B(.view) | 同上（forward 経路） | forward の `raw = w.view(torch.uint8)` が同じ属性剥がし→override `numel()` が落ちる | `w.as_subclass(torch.Tensor).view(torch.uint8)` で override 非経由に。属性は `w` 本体から明示引数で dequant に渡すため raw 側不要 | ✅ 適用済（ユーザー承認） |
| 3 | **偽 CUDA OOM「4096 GiB 確保」** | `_dequant_q{4,5,6}_k` の `d`/`dmin` が `(n,2)→(n,1)` の **2D**（コメントは `(n,)` を意図）。`d[:,None]` が `(n,1,1)` になり `sc_rep(n,256)` との乗算が **`(n,n,256)` の n×n 外積に膨張** | 実在3型（Q4_K/Q5_K/Q6_K）の `d`/`dmin` を `.reshape(-1)` で 1D 化 | ✅ 適用済（OOM消失・形状一致） |
| 4 | **dequant 数値がノイズ（rel≈1.0–1.6）** | 手書き K-quant カーネルの **GGML レイアウト解釈が誤り**: nibble の lo/hi 配置順＋6-bit packed scale/min のサブブロック割当が不正 | **B案採用**: `gguf.quants` を torch/GPU 忠実移植（`_dequant_q{4,5,6}_k` 全面置換＋`_q_get_scale_min`新設）。検証 **18/18 bit-exact (max_abs=0)** | ✅ 適用・検証済 |
| 5 | **native access violation**（safetensors ロード） | **真因精密特定**: `sft_loader.py:36` の `f.get_tensor(name).to(device, non_blocking=True, copy=False)` を **CUDA初期化後に mmap-backed safetensors へ実行**。PyTorch公式が「CUDA後の `.to(non_blocking=True)` は同期なしでデータ破損/不安定」と明記。`copy=False` は mmap エイリアスを返し悪化。これが**当初pivot crash・bs4間欠・CPU-Gemma決定的crash の同一根本** | **C-精密版**: ltx_core ローダの `non_blocking/copy=False` を除去（`copy=True`／`load_file(pread)`）する**可搬 monkeypatch** を我々のコードから適用 | 🔄 適用中 |

**進捗の意義**: 修正 1/2-A/2-B/3 により crash/OOM は解消し dequant 実演算まで到達。だが**数値検証で
カーネル自体の値がノイズと判明**（バグ#4）。形状だけ直して生成を回す前に数値検証を入れた判断が奏功
（「クラッシュしないのにノイズ出力」を事前排除）。

### 1.3 数値検証の結果（fork カーネル vs `gguf.quants.dequantize`, 実在3型×各3 tensor）
- **全 9 ケース FAIL**（rel 1.10〜1.62）。検証スクリプト: `vendor/.../backend/_verify_dequant.py`。
- この GGUF の実在量子化型 = **Q4_K(1242) / Q6_K(322) / Q5_K(68)** のみ（F32/BF16 は float branch）。
  Q2_K/Q3_K/IQ4 等は不在 → 対象外（据え置き）。

### 1.4 バグ#4 の修正方針（ユーザー承認待ち・トレードオフ）
フォークの「per-layer 圧縮 in VRAM ＋ GPU on-the-fly dequant」という低VRAMの肝を、どう正しくするか:
- **A) gguf リファレンスを forward 毎に呼ぶ**: 正しいが numpy/CPU 実装を毎 forward・毎 Linear で実行 →
  CPU往復だらけで**深刻な速度低下**。非推奨。
- **B) GPU dequant カーネルを正しく置換（推奨）**: 既知の正しい torch K-quant dequant（gguf/ComfyUI-GGUF
  系譜＝このフォークの出自）を移植。圧縮 in VRAM ＋ 高速 GPU dequant を維持。実装コスト中・correctness は
  リファレンス準拠で担保。
- **C) ロード時に1回だけ正しく dequant → bf16 を CPU 常駐 → bf16 block-swap**: 正しく単純だが、22B bf16 ≈
  44GB を CPU RAM（空き63.8GB）に置き、PCIe 転送量が約2倍。圧縮 in VRAM の利点は失う。

> **教訓**: フォークの価値（GGUF ロードで safetensors crash 回避 ＋ block-swap）は健在だが、**K-quant
> dequant カーネルは「実証済みコード」ではなく壊れていた**。ここは正しいリファレンス実装へ置換が必要。
> → **解決済**: B案で `gguf.quants` を torch/GPU 忠実移植、18/18 bit-exact。

### 1.5 生成スパイク結果（bs8 完走・数値妥当性つき, 2026-06-28）
- **bs8 のみ End-to-End 完走**: 384x256/9/8steps、生成 **228.3s**、出力 `outputs/forkenv_spike/min_bs8.mp4`。
- ffprobe: `h264 384x256 9frames 8fps` 一致。映像妥当: 全9f luma~157（黒/白でない）・フレーム間 156.7→158.6 と滑らか
  → **ノイズでも真っ黒でもない有意な映像**。bit-exact dequant とあわせ出力は正常。
- **✅ Path A の中核 premise（GGUF で crash 回避しつつ正しい映像生成）は実証完了**。

### 1.2 数値正しさの方針（過剰検証を避ける）
- 手書きK-quantカーネルがデモでバグっていた以上、形状を直して「動く」だけでは**出力ノイズ化のリスク**。
- ただし目標は「16GBで動かす」こと。**Q4_K_M GGUF に実在する量子化型だけ**を修正＋`gguf`ライブラリの
  リファレンス dequantize とサンプルtensorで数値突き合わせ（型ごと数秒・軽量）。使われない型は対象外。

---

## 2. VRAM 挙動の観察（重要・未解決）

### 2.1 人間によるタスクマネージャ目視（2026-06-28, bs8 試行中）
- **専用GPUメモリ(16GB)使い切り＋共有GPUメモリ 10〜20GB 使用**のタイミングあり（＝合計26〜36GBがGPU
  アドレス空間に載った）。
- **解釈**: 何か大きな重みが丸ごとGPUに載っている。Windows WDDM は専用VRAM超過分を**ハードOOMせず
  共有メモリ（システムRAM）へ溢れさせる**ため、crashせず激遅化する。**Forkの低VRAM機構（Gemma offload /
  transformer block-swap）がこの環境で期待通り効いていない疑い**。
- **候補原因**（未切り分け）: ① Gemma-3-12b(bf16 ≈24GB) を丸ごとGPUにロードしてエンコード（24>16→約8GB溢れ、
  観察と数値整合）② GGUF 17.76GB を block-swap でCPU退避する前に一旦GPUに丸ごとロード。
- **注意**: 形状バグ(#3)の偽OOMは*一瞬の不可能確保→即失敗*なので、この**継続的な10〜20GB溢れは説明しない**＝
  別シグナル。形状バグ解消後のクリーンな計測でフェーズを切り分ける必要がある。

### 2.2 計装方針（フェーズ別＋共有メモリ直接監視）
- Windows perf counter を**直接**サンプリング（タスクマネージャと同ソース・低コスト）:
  `Get-Counter "\GPU Adapter Memory(*)\Dedicated Usage"` と `"\...\Shared Usage"`。サンプラ: `_gpu_mem_sampler.ps1`。
- `torch.cuda.max_memory_allocated()` のみでは WDDM 共有溢れを見逃す（=「16GBに収まった」と誤認）ため併用。

### 2.3 実測結果（perf-counter 直接監視, 2026-06-28）— ★戦略的に決定的
| blocks_on_gpu | dedicated peak | **shared peak（WDDM溢れ）** | 結果 | 生成時間 |
|---|---|---|---|---|
| 8 | 15923 MB（≈16GB張り付き） | **17792 MB** | GENERATED_OK（有意映像） | 228.3s |
| 4 (1) | 15957 MB | 17718 MB | CRASH access violation @VAE decoder load（denoise後） | — |
| 4 (2) | 506 MB（早期crash） | 785 MB | CRASH access violation @Gemma load（denoise前） | — |
| 2 | — | — | 未実施（bs4不安定のため停止） | — |

**結論（実機確定）**:
- **専用16GB張り付き＋共有~17.7GB溢れ＝合計~33.7GB を system RAM 経由ページング**。だから 228s と遅い
  （~15s/it = thrashing）。`max_memory_allocated`(torch) の 31772MB はクランプ＋断片化で不正確、実態は perf-counter。
- **bs8↔bs4 で溢れ量も速度もほぼ同一 → block_swap 深度は支配的レバーではない**。食っているのは block-swap
  作業集合ではなく、**Gemma-3-12b(bf16 unquantized ≈24GB) ＋ VAE ＋ per-layer dequant bf16中間体**。
- **バグ#5 はメモリ圧迫起因**（共有thrashing下で safetensors 遅延ロードが access violation）。bs8 は通過、
  bs4 は再発＝非決定的。**＝遅さもクラッシュも、根本は「16GBに収まっていない」こと**。footprint を削れば両方解消の見込み。

### 2.4 フェーズ別実測の確定結果（analysis-only, 2026-06-28）
bs8 完走ログ（`gpu_mem_bs8.log`＋スパイクのフェーズ print）を wall-clock 突き合わせ:

| フェーズ | dedicated | **shared** | 判定 |
|---|---|---|---|
| **Gemma テキストエンコード** | 15.9GB 張り付き | **0.77→17.79GB（線形上昇しピーク）** | ★**溢れの主因（100%限局）** |
| Gemma 解放(`cleanup_memory`) | 15.9→2.5GB 急落 | 17.79→0.76GB 急落 | 解放署名（持ち越し無し） |
| transformer denoise(8 steps) | 15.88–15.90GB 張り付き | **~3.9GB（安定・小）** | ほぼ16GB内。溢れ小 |
| VAE decode + upsampler | ~15.9GB | ~3.9GB→崩落 | tiling はこの寸法で no-op |

**コード根拠（確定）**:
- **Gemma = 丸ごと bf16 GPU ロード**。フォークの `LTXFastVideoPipeline` が構築する公式 `DistilledPipeline`(rev 00dc53d)
  には **`offload_mode` 引数が無い**。`ModelLedger.text_encoder()` → `SingleGPUModelBuilder.build` →
  `load_sd(device=cuda)` で **全 state_dict を一括 GPU ロード**。`models/gemma-3-12b-it-qat/` は **bf16 24.4GB・非量子化**
  （"qat" だが保存重みはフル bf16）。→ 24GB を 16GB GPU に載せ ~17.7GB を WDDM 共有へ溢れ。
- **旧 rev の退行**: 公式 1.1.6 にあった Gemma 層ストリーム（`_DEFAULT_GPU_SLOTS=2`）が、フォーク pin の rev 00dc53d
  の `SingleGPUModelBuilder` には**無い**。フォークは transformer の crash は解決したが Gemma メモリは退行。
- **block_swap は効いている**が transformer フェーズのみ作用＝溢れ(Gemma)に無関係。**VAE tiling は本寸法で no-op**。

### 2.5 レバー評価（確定・主因=Gemma に効くもの）
| レバー | 有効性 |
|---|---|
| Gemma を CPU でエンコード（24GB を system RAM・空き63.8GB、埋め込みのみ GPU へ） | **○ 最単純・GPU圧迫ゼロ**。一度きりの encode が CPU で遅くなるが許容圏か（要実測） |
| Gemma を層ストリーム化（block_swap 相当を Gemma に・実装要） | **○ GPU 速度維持の本命**。ただし一括ロード peak を避けるにはローダ介入が必要 |
| Gemma を 4bit 量子化重みに差し替え（~7GB） | **○ だが要ローダ対応**（ltx_core の Gemma ローダは bf16 safetensors 前提） |
| VAE tiling 強制 / GGUF Q3_K_M | **× 主因(Gemma)に効かない**。本寸法では効果ゼロ/限定的 |

**含意**: Gemma フェーズさえ 16GB に収めれば denoise(shared 3.9GB)/VAE はほぼ収まる → 完走が現実的。
バグ#5(native crash)も Gemma 周辺の thrashing 下で発生していたため、Gemma 溢れ解消で再発確率低下の見込み。

### 2.6 WEB リサーチ結果（Gemma 低VRAM運用の定石, 2026-06-28・出典付き）
- **CPU text-encode は標準パターン**（Wan/HunyuanVideo/diffusers `enable_model_cpu_offload`）。「1回エンコード→
  エンコーダ退避→埋め込みのみ GPU 常駐」。LTX-2.3 でも 12GB カードで実証報告（WaveSpeed）。→ **A案は妥当**。
- **地雷1: CPU の bf16 は激遅**（AVX512_BF16/AMX 非搭載の消費者CPUで最悪~1000倍。PyTorch中核開発者報告）。
  → **CPUエンコードは fp32 で**（Gemma ~48GB RAM 常駐、本機 63.8GB で収まるが余裕小）。所要 ~1–2分/回 のオーダー。
  `torch.autocast`(AMP) は重み fp32 保持で RAM 倍化 → **使わない**。
- **地雷2/代替: コミュニティ本命は「量子化 Gemma を GPU」**（GGUF Q4_K_M ~6–7GB / fp8 / fp4。city96 PR#402、
  公式ComfyUIは fp4_mixed 推奨）。GPU速度維持で 16GB 内。**ただし ComfyUI 前提**で、我々の ltx_core スタックでは
  ローダ統合が要る＝A案より実装コスト高 → **将来の最適化候補（C案）**。
- transformer GGUF: 16GB実用は Q3_K_M(14.7GB) or Q4_K_S(16.7GB)。Q4_K_M(17.8GB) は単体16GB超でエンコーダ同時
  常駐不可。RAM ≥32GB がコミュニティ総意。
- **方針**: Phase 1 は **A案（CPU-encode・fp32）** で確実に 16GB 達成 → 動いてから C案（量子化Gemma GPU）で高速化。

### 2.7 A案 実装設計（調査完了・read-only, 2026-06-28）
- **ltx_core Gemma 経路**: `ModelLedger.text_encoder()` = `text_encoder_builder.build(device=cuda, dtype=bf16).to(cuda)`。
  Gemma は HF `Gemma3ForConditionalGeneration` だが **`from_pretrained` 非経由**（meta構築＋汎用 `load_sd`＋
  `load_state_dict(assign=True)`＋独自 key-remap）。`__call__` 冒頭で encode→`del text_encoder; cleanup_memory()` で即解放。
  埋め込み出力 `GemmaEncoderOutput(video/audio_encoding, mask)` は数MB級。
- **A案 最小変更点（フォーク1ファイル `ltx_fast_video_pipeline.py`）**: `model_ledger.text_encoder` を、
  `text_encoder_builder.build(device=cpu, dtype=fp32)` で組み（末尾の `.to(cuda)` を回避）、薄いラッパで
  `encode_text` 呼び出し時に **CPU fp32 で Gemma forward → 出力埋め込みのみ `.to(cuda, bf16)`** して返す。
  transformer/VAE/upsampler は不変。block_swap/GGUF ラップと非衝突。
- **dtype = fp32 確定**: 本機 CPU = **Intel i7-13700（AVX2のみ・AVX512_BF16/AMX 無し）** → CPU bf16 はエミュレートで
  桁違いに遅い。fp32＋oneDNN で実行。`torch.autocast('cpu')` は使わない（重み fp32 保持・RAM倍化回避）。
- **★RAM 注意**: Gemma fp32 ≈ **48.8GB** vs 空き **47.2GB**（僅か超過）。緩和: (i) シャード逐次 fp32 化で
  ロードピーク抑制（bf16+fp32 同時 ~73GB を回避）、(ii) text-encode に不要なら vision_tower 除外、
  (iii) 重いアプリを閉じて空き RAM 確保（総 63.8GB）。
- **コスト**: 生成あたり ~2–3分 の一度きり CPU エンコード時間増（GPU 生成本体 228s に上乗せ）。
- **C案（量子化Gemma GPU）は容易でない**: `from_pretrained` 非経由のため bnb-4bit 統合は経路作り替えが必要。
  将来の高速化本命だが今は A案優先。

### 2.10 ★方針転換: CPU-projection 撤回 → 量子化Gemma GPU推論（先行事例を複製）, 2026-06-28
- **CPU-encode(bf16) は撤回**。理由: ① この CPU(i7-13700, AVX2のみ・AVX512_BF16なし)では bf16 はエミュレートで遅く、
  かつ transformers の bf16→fp32 暗黙アップキャストで 48GB 膨張→スワップ thrashing の疑い（生成開始から十数分でも
  encode 終わらず＝フェルミ推定の「最大~1分」を大幅超過＝異常）。② **CPU-projection は先行事例が乏しく**（我々の
  検索ツールでは発見できず）、結局**手探りの独自実装**になっていた＝地雷を踏む構造。**fp32 は RAM 48GB で不可**。
- **教訓（ユーザー指摘）**: 調査不足のまま独自実装を進めるのは危険。**コミュニティに多数の先行事例がある
  「量子化モデルの GPU 推論」に切り替え、検索スニペットでなく先行事例の“ソースコード”を直接複製する。**
- **採用方針**: Gemma を **量子化(fp8 e4m3fn / fp4_mixed / GGUF Q4, ~6-7GB)で GPU に載せて推論**（16GB に収まる）。
  de-risk 済の **GGUF transformer＋block_swap＋dequant修正(bit-exact)＋sft_loader安全patch は維持**。変えるのは
  Gemma エンコーダの load 戦略のみ。複製元: Lightricks 公式 **ComfyUI-LTXVideo `gemma_encoder.py`**（`from_pretrained`）、
  city96/ComfyUI-GGUF、量子化Gemma 配布(GitMylo fp8 / 公式 fp4_mixed)。**先行事例ソースを精読してレシピ確定 → 複製。**
- bs8 で denoise は依然 ~4GB shared spill（GEN_PEAK 16901MB）→ block_swap 深度 bs4/bs2 で別途詰める（小課題）。

### 2.8 決定: A案・bf16 安全側（fp32 ではなく bf16）, 2026-06-28 ※2.10 で撤回
- **採用 = Gemma を CPU に bf16 ネイティブ重み(~24.4GB)で載せエンコード**（量子化ではない・半精度のまま）。
  空き RAM 47.2GB に余裕で収まり **OOM リスクなし・アプリを閉じる不要・ローダ改変不要**。fp32(48.8GB) の RAM 逼迫を回避。
- 代償: CPU bf16 はエミュレートで遅い → **一度きりのエンコードが数分**（Phase1 は正しさ優先で許容）。
- 「量子化Gemmaを CPU で」は不採用: bnb は CUDA 専用・GGUF は別ランタイム要で**かえって複雑**。bf16 だけで OOM 回避は達成。
- 実装はフォーク1ファイル（`ltx_fast_video_pipeline.py`）に `cpu_text_encode` ラッパ追加。

### 2.9 sft_loader crash の根治（WEB調査で精密特定, 2026-06-28）
- **CPU-Gemma は妥当・CUDA必須は否定**（WaveSpeed 実走 / ComfyUI CPUログ / HF transformers CPU forward /
  diffusers offload で多重裏付け）。crash は「CPUで動かない」のではなく**ローダ実装ミス**。
- **真因 = `sft_loader` の `.to("cpu", non_blocking=True, copy=False)` を CUDA 初期化後に mmap-backed safetensors へ**。
  PyTorch 公式が CUDA後 `.to(non_blocking=True)` の同期なし破損を明記。`copy=False` は mmap エイリアス化で悪化。
  ComfyUI は同種問題に対し `non_blocking`/`copy=False` を**使わず** `copy=True` eager を採用（堅牢パターンの裏付け）。
  **当初 pivot crash・bs4間欠・今回の CPU-Gemma決定的crash は全て同根**。
- **採用 = C-精密版（可搬 monkeypatch）**: ltx_core ローダの該当 `.to()` から `non_blocking=True`/`copy=False` を除去
  （→ `copy=True`、必要なら `safetensors.torch.load_file(path, device, backend="pread")` で no-mmap eager 化）。
  我々のコードから runtime patch（`.venv` 生編集なし・env再構築で消えず・in-repo）。de-risk は spike 冒頭、Phase B は
  `services/lowvram` 初期化に。**bs4 の VAE crash も同根のため同時解消の見込み**。
- **E（HF from_pretrained 迂回）は不採用**: LTX は Gemma 全49層 hidden states＋LTX側 projection
  (`text_embedding_projection.aggregate_embed.weight [3840,188160]`, stock Gemma に無し) を要し、迂回は再実装増。
- 実装→スパイクで「CPU-Gemma load 成功」「Gemma 17.7GB 溢れ解消」「denoise を bs深度で 16GB 完全収容」「出力妥当性」を実測中。

---

## 3. 環境・クリーン状態（参考）
- アイドルVRAM ~770MiB（占有は全て C+G デスクトップアプリ。compute プロセス無し）。スパイク後リーク無し。
- フォーク env: `vendor/LTX-Desktop-LOW-VRAM/backend/.venv`（torch 2.9.1+cu128 / gguf / ltx_core@00dc53d）。

---

## 4. 未解決・次アクション
- [x] バグ#1〜#4 修正・数値検証 18/18 bit-exact・bs8 完走（有意映像）＝**Path A 中核 premise 実証完了**。
- [ ] **footprint を 16GB 内に収める**（block_swap ではなく Gemma/VAE が支配）。まず `gpu_mem_bs8.log`＋
      フェーズログ突き合わせで 17.7GB 溢れの主因フェーズを特定（analysis-only, 新規実行なし）。
- [ ] レバー選択（§2.4）: Gemma q4化/層ストリーム（最有力）／VAE tiling／(補助)Q3_K_M。**ユーザー判断待ち**。
- [ ] バグ#5（遅延ローダ access violation）: footprint 削減で解消するか再確認（メモリ圧迫起因の仮説検証）。
- [ ] 上記で 16GB 内・共有溢れ無し完走を確認後: フォークエンジンを `services/lowvram/` へ取り込み、
      `_RealBackend` 差し替え（Phase B）。

---

## 5. ★Gemma 量子化マイルストーン — 16GB E2E 達成（2026-06-28, fp4撤回→GGUF Q4採用）

**方針**: fp4_mixed(NVFP4) は ComfyUI 密結合（`comfy-kitchen` 必須・高速カーネルは torch cu130+ 前提、本機 cu128 は遅い fallback）と判明し**撤回**。代わりに **GGUF Q4_K_M Gemma を我々の bit-exact dequant エンジンで GPU 推論**（先行事例＝WaveSpeed が 12–16GB で Q4 採用／キーremapは city96/ComfyUI-GGUF `loader.py` を複製）。Q4 にした理由は実績優先（Q6 は KL 的に良いが LTX エンコーダ実績が無く独自最適化になるため後回し）。

**実装**: 新規 `vendor/LTX-Desktop-LOW-VRAM/backend/services/gemma_gguf_quant_service.py`。
- `gguf_quant_service.py` の dequant カーネル/`GGMLQuantizedTensor`/per-layer forward を再利用。
- net-new（city96 複製）: `GEMMA3_SD_MAP`+`sd_map_replace`（gemma3固有=QKノルム/二重FFNノルム）、`gemma3_norm_corrections`（RMSNorm の llama.cpp 焼込 (1+w) を −1 補正）、embed/norm の bf16 据え置き、Gemma3 用 module-ops matcher。llama_permute 不使用。
- **MERGE**: base safetensors（feature_extractor projection=video[4096,188160]/audio[2048,188160]＋connectors＋vision）に GGUF Gemma 重みを overlay。projection は GGUF に無く LTX checkpoint から従来通り。
- 配線: `ltx_fast_video_pipeline.py` に `gguf_gemma_path`/`_install_gemma_gguf`（`cpu_text_encode` と排他）。
- DL: `ggml-org/gemma-3-12b-it-GGUF` `gemma-3-12b-it-Q4_K_M.gguf`(7.30GB, ungated, arch gemma3, 量子化型 {F32,Q4_K,Q6_K})→`models/gemma-3-12b-it-gguf/`。

**検証（生成前ゲート）**: Gemma 実テンソルで dequant **bit-exact(max_abs=0)**、キー remap missing/unexpected=0、norm −1 を1回だけ適用、forward 49層・全有限、**bf16 リファレンスと cosine=1.0009(video)/0.99991(audio)**＝「crash しないのにノイズ」を排除。

**修正したバグ**: ①base bf16 Gemma を strip 前に GPU ロードし OOM → base を CPU ロード後に strip。②GGUF vocab 262144 vs config 262208 → ゼロパディング（target_vocab を base から動的取得）。

**VRAM 最適化（エンコーダを 16GB に収容）**: 真因＝(a)1BパラQ4_K embedding の **GPU dequant が一時 21GB** → CPU dequant 化、(b)`logits_to_keep=1`（encode は logits 破棄）、(c)embed_tokens を **CPU offload**（lookup を CPU、結果のみ GPU）。lm_head は元々 tie 済（前ゲートの「1.9GB無駄」は検証スクリプトの二重計上だった）。結果: build peak **13.1GB** / forward 込み peak **15.1GB**（~1.3GB 余裕）、336量子化バッファ、forward std 維持。

**Phase 4 実生成（perf-counter フェーズ別実測）**: 384x256/9/8steps T2V **完走175.7s**、出力 `outputs/phase4_gguf_gemma/t2v_bs8.mp4`（h264 384x256 9f 8fps）。
| フェーズ | dedicated | shared | 判定 |
|---|---|---|---|
| **Gemma encode** | 15.9GB | **~0（ambient 435MB のみ）** | ★**17.7GB溢れ解消** |
| denoise(stage1+2, bs8) | 15.97GB | **3.6GB** | 既知の block_swap ノブ（Gemma 無関係。bs4/bs2 で詰める） |
| VAE | 6.3GB | ~0 | 余裕 |
出力妥当: luma 170.6→175.1 滑らか・**プロンプト追従（赤い車/海岸/夕日）**。

**結論**: 「16GB で LTX-2.3 を動かす」の支配的ボトルネック（Gemma bf16 溢れ）は**解消**。残課題は denoise の shared 3.6GB（block_swap 深度の小課題）のみ。次=Phase 5（`services/lowvram/` 取込み＋`_RealBackend` 差替え、要承認）。計画書 `~/.claude/plans/nifty-beaming-puzzle.md`。
