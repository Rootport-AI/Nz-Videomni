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
- [x] フォークエンジンの取り込み＋`_RealBackend` 差し替え＝**Phase 5(A) 完了**（§6）。ただし方式は当初想定の
      `services/lowvram/` への取込みではなく **Approach W＝フォークを常駐サブプロセスワーカー `_ltx_worker.py`
      で呼ぶ**（venv 分離＋`services` 名前衝突回避）。**残＝真の 16GB 収容（denoise の共有溢れ解消）＝Phase 5(B)＝§6.4／handoff §3c**。

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

**結論**: 「16GB で LTX-2.3 を動かす」の支配的ボトルネック（Gemma bf16 溢れ）は**解消**。残課題は denoise の shared ~3.6GB。計画書 `~/.claude/plans/nifty-beaming-puzzle.md`（＝§5＝Phase 4 の計画・歴史的参照）。※その後 **Phase 5(A) を Approach W（常駐サブプロセスワーカー）で実施・実機検証済＝§6**（当初の "`services/lowvram/` へ取込み" 案は不採用）。なお denoise の shared ~3.6GB は「小課題」ではなく、ユーザー目視（384x256 でも溢れ）を踏まえ **真の 16GB fit の中核**として **Phase 5(B)** で扱う（§6.4／handoff §3c）。

---

## 6. Phase 5(A) 配線・実機検証 — 凍結API が実エンジンで動いた (Approach W, 2026-06-28)

§5 の実証済みエンジンを、我々の**凍結 REST API** から実際に配信できるようにした（"配線 first"）。凍結境界は不変のまま、`services/ltx_runner.py` の `_RealBackend`（従来は公式 `DistilledPipeline` 直叩き＝本機で native crash する死に筋）を実体化。計画書 `~/.claude/plans/a-witty-kazoo.md`。

### 6.1 アーキテクチャ＝Approach W（常駐サブプロセスワーカー）＋採用理由
- **構成**: `_RealBackend` は、フォーク `vendor/LTX-Desktop-LOW-VRAM/backend/_ltx_worker.py` を**フォーク env** で**常駐プロセス**として起動する。worker はモデルを**1度だけ**構築し、以後ジョブを JSON-lines プロトコル（行は `@@LTX@@` でフレーミング）で受け付ける。エンジンが `output.mp4` を共有 output dir へ直接書く。**app `.venv` は torch/engine を一切 import しない**。
- **なぜ Approach W か**（旧ハンドオフの「`services/lowvram/` へコピー」案でも in-process 案でもなく）:
  - エンジンは torch＋ltx_core@`00dc53d`＋gguf を要し、それらは**フォーク env にしか無い**（app `.venv` には torch すら無い）。
  - さらに app とフォークは**双方ともトップレベルに `services` という同名パッケージ**を持ち、フォーク側 `__init__` が ~18 の torch 依存モジュールを eager import する → **同一インタプリタで共存不可**（import 衝突）。
  - 二プロセス分離なら衝突を完全に回避でき（**rename 不要**）、**検証済みのフォークコードを無改変で**使える。

### 6.2 ファイル変更（凍結境界は不変）
- **`services/ltx_runner.py`**: `_RealBackend` を作り替え（公式 DistilledPipeline → 常駐 worker spawn＋JSON-lines プロトコル）。`_real_available()` も変更＝**ltx_pipelines/torch を import しない**。代わりに `fork_python`＋worker スクリプト＋モデルパス5個の存在のみをチェック。
- **新規 `vendor/LTX-Desktop-LOW-VRAM/backend/_ltx_worker.py`**: モデルを1度構築して常駐し、ジョブを使い回す worker 本体。
- **`config.py`**: `model.{gguf_transformer_path, gguf_gemma_path, fork_backend_dir, fork_python, gguf_per_layer_quant}` ＋ `vram.{vae_spatial_tile_size, vae_temporal_tile_size}` を追加。
- **`services/low_vram.py`**: 新ノブをマッピング。`status_block()`/`metadata_block()` のキー形は**凍結のまま**（新しい内部フィールドが GET /status 契約へ漏れないよう、明示的な **7-key タプル**へ射影）。
- **凍結境界 intact**: `pipeline_manager.py` / `api/models.py` / `main.py` / `run.ps1` / `tests/` / `_MockBackend` / `LTXRunner` public / `GenerationOutcome` / `ProgressCallback` は**すべて不変**。**mock pytest は緑のまま（13 passed）**。

### 6.3 実機検証（実 `LTXRunner.generate` 経路, 16GB GPU でサブエージェントが実行）
| 検証 | 結果 | dims/frames | wall | backend / mode |
|---|---|---|---|---|
| T2V 384x256/9 | **PASS** | h264 384x256 9f | **~207s**（初回 worker spawn＋モデルロード込み） | `ltx-distilled` / t2v |
| 最小 I2V 384x256/9 | **PASS** | h264 384x256 9f | **~214s** | `ltx-distilled` / i2v |

- **常駐ワーカー再利用を実証**: worker ログに `PIPELINE_CREATED_OK` が**ちょうど1回**、`GENERATED_OK` が**2回**＝モデルは1度だけロードし両ジョブを使い回した。
- **OOM リカバリ経路（コード確認で確定）**: worker エラーに "out of memory" を含む場合、`pipeline_manager._is_oom` → `_cleanup_after_error()` → `runner.unload()`（worker を kill）→ 次ジョブで fresh に再ロード。

### 6.4 VRAM の正確な状況（訂正）
**★ "16GB に収まった" とは書かないこと。** 重要な訂正:
- worker が報告する `peak_vram_mb`＝`torch.cuda.max_memory_allocated` は **16913（T2V）/ 17989（I2V）MB**。**この指標は dedicated（専用VRAM）と WDDM shared（共有メモリ）を区別できず、したがって shared への溢れを検知できない。**
- **ユーザーが Windows タスクマネージャで目視確認**: **384x256 でも、denoise ステージで相当量のデータが SHARED GPU メモリへ溢れる**（WDDM が system RAM へページング）。これは §5 の "denoise shared ~3.6GB @bs8" と整合する既知挙動。**ハード OOM しないのは WDDM が溢れ分を system RAM へ逃がすからで（＝遅さの源）、"真の 16GB fit" はまだ未達。**
- 溢れの検知には perf-counter の **"Shared Usage" サンプリング**が要る（§2.2／§5）。`max_memory_allocated` だけでは見逃す。
- **残作業の再フレーム（ユーザー方針）**: 「真に 16GB に収める」＝この denoise-stage の shared 溢れを解消することであり、**Phase 5(B) スケールアップ（1280×768 → crop 720p）と一つの同じ仕事**。フォーク（`block_swap_service.py` 等）と ComfyUI カスタムノード/ワークフローが持つ denoise-stage の VRAM 技法（より深い block_swap 深度 bs4/bs2、VAE/attention タイリング、解像度依存の sequential/streaming）を**我々の backend はまだ反映していない**。これらスケールアップ技法を適用すれば、**現状の 384x256 の shared 溢れも一緒に解消**する見込み。Phase 5(B) はこれら実証済み denoise-stage 技法をフォーク／ComfyUI-GGUF から**調査・複製**することから始める。

**次=Phase 5(B)**（真の 16GB fit＝denoise shared 溢れ解消＝1280×768 スケールアップ）。計画書 `~/.claude/plans/a-witty-kazoo.md`、引継ぎ [NEXT_SESSION_HANDOFF.md](NEXT_SESSION_HANDOFF.md) §3c/§4。

---

## 7. Phase 5(B) — 384x256 denoise 溢れの真因特定と修正確定（調査→診断→実測, 2026-06-28〜29）

§6.4 の「denoise が 384x256 でも shared へ ~3.6GB 溢れる」を、**調査（ソース）→診断（フェーズ別4指標）→対策の実測確認**の順で解決した（計画書 `~/.claude/plans/frolicking-tumbling-hennessy.md`）。**実コード・凍結層は不変**、全計測は throwaway スパイク（`outputs/phase5b_diag/`）＋ `_gpu_mem_sampler.ps1`。

### 7.1 ステップ1＝先行事例調査の結論（当初4候補の希望判定）
- **384x256 では当初候補はほぼ無効**：~190 トークンで attention/FFN 活性は単桁〜数十MB＝3.6GB を活性で説明不能。
  - FFN チャンキング／attention tiling は**高解像度（スケールアップ）専用のレバー**。attention tiling は worker 未配線（`attention_tile_size` 未送）。VAE tiling は denoise 無関係。block-swap 深度は ~0.8GB の小利得のみ。
  - FFN チャンキング複製元・変換を特定（RandomInternetPreson `tensor_parallel_v3.py` の `ChunkedFFN.forward`／`num_chunks=8`／`torch.cat(dim=1)`、フック点 `ltx_core/.../feed_forward.py:FeedForward.forward`）＝スケールアップ局面で利用。
- ComfyUI #11726 の「活性支配」は **1080p アップスケール段**の OOM＝384x256 とは別レジーム。

---

## 10. ★720p 達成 & 連続生成の commit 枯渇を解決（2026-06-30 後半）＝残課題C 完了

本セッションで残課題C（720p スケールアップ）を達成し、さらに新デフォルトがマルチジョブ連続生成で commit 枯渇しないことを実測確定した。GPU(dedicated/shared)＋system commit を両監視。

### 10.1 720p 二段の実機到達
- **本番経路は公式 `DistilledPipeline` の二段**（stage1=半解像度 W/2,H/2 生成→spatial upscaler×2→stage2=フル解像度 refine）をフォーク `LTXFastVideoPipeline` がラップ。段間は `gpu_model` が各段終了時にモデルを `meta` 退避で即解放＝**両段同時 GPU 滞在なし**（doc の「難所①段間スパイク」は構造的に緩和済）。VRAM ノブ（block_swap/vae tile/fp8/GGUF）は両段に適用。
- **計装1本（直接スクリプト `vendor/.../backend/_run720p_instrumented.py`、雛形=diag_sweep.py＋本番ワーカーの pre-denoise empty_cache fix を再現）**：1280×768/121f/8step を **169秒**で完走。段境界 max_alloc＝denoise stage1 ~5.8GB / stage2 ~7.7GB（大余裕）。唯一のスパイクは `before denoise stage1`（reserved 17.4GB→empty_cache で 1.5GB に即解放）。
- **本番 API 経路**：`POST /api/v1/generate` で 1280×768/121f を **167–171秒**完走、crop_output で 1280×720 配信。ffprobe で解像度/121f/5.04s 確認（job `684393c6`=768 無 crop, `5a3540ba`=720 crop）。

### 10.2 keep_resident=True が 720p 単発を native crash させる（原因＋修正）
- **症状**：本番 API（keep_resident=True 既定）が 720p の Gemma text-encode 中に native crash（traceback 無し、dedicated ~15.86GB で即死）。直接スクリプト（keep 未指定＝既定 False）は同条件で完走。
- **原因（read-only コード調査・確信度高）**：keep_resident=True→`ModelLedger._target_device()`=CPU（`model_ledger.py:178-182`）→GGUF Gemma を **CPU ビルド→`model._apply(t.to(cuda))` で out-of-place にモデル全体を GPU コピー**（`gemma_gguf_quant_service.py:1135-1138`）＝CPU重み＋新規GPU重みの瞬間二重在が薄い16GBマージンを超過。keep=False は cuda 直接ビルドで二重在なし。`create()` 既定 False（`ltx_fast_video_pipeline.py:75`）。`embed_tokens.weight` の "Uninitialized parameters" は正常ログ（red herring）。
- **検証**：`LTX_KEEP_RESIDENT=0`（env は `ltx_runner.py:486` `env=dict(os.environ)` で worker へ伝播）で本番 API が完走（171.5秒、ded 15977/shr 2589）＝主因確定。
- **修正（採用）**：`services/ltx_runner.py` worker env に `env.setdefault("LTX_KEEP_RESIDENT","0")`（明示 env 尊重）。

### 10.3 マルチジョブ連続：comp=0 は commit 枯渇再発、comp=1 で解消（実測）
keep=0 は毎ジョブ全 submodel を再 materialize するため、§8 の commit 枯渇機構が再発しうる。3本連続で検証：

| 設定 | 結果 | committed ピーク | commit% | 備考 |
|---|---|---|---|---|
| comp=0/keep=0（旧既定）@384×256 | **job3 で native crash** | 135GB | **99.2%** | Gemma 再構築点で死＝§8.10 を §9.7 修正後も再現（別機構＝毎ジョブ再materialize） |
| comp=1/keep=0 @384×256 | **3本全完走** | 102GB | 89.5% | 46GB モノリス不使用で commit 束縛 |
| comp=1/keep=0 @1280×768/121f | **3本全完走** | 102.6GB | 90.3% | 実ターゲットで PASS・commit はジョブ間で累積せず横ばい |
- **採用**：`config.yaml` vram `use_component_files: true`（Path B）。＋`block_swap_blocks_on_gpu:8`/`vae_spatial:512`/`vae_temporal:64` 明示。
- 留意（非致命）：keep=0 は gen 時間が漸増（720p で 168→181→209秒/3本、+24%）。commit 横ばいゆえ暴走せず＝アロケータ断片化等の性能ナンス。長尺連結を多数本回す段で問題化したら keep=1＋Gemma 移動 in-place 化の恒久最適化（§9.7 の高速 flat 経路と 720p を両立）。

### 10.4 マシンスペック（公開時見積り・先行事例比較）
- **commit（仮想メモリ＝物理RAM＋ページファイル）ピーク ~102GB**。本機 RAM64GB＋pagefile48GB＝上限~112GB で 90% 着地。先行事例 ComfyUI 16GB レシピも「RAM32GB＋swap64GB」相当（§91）＝**ほぼ同等・極端でない**。旧懸念の ~200GB は 46GB モノリス＋二重 materialize の悪い経路で、comp=1 で回避済。keep=0 が commit をやや押し上げ（keep=1 の §9.7 実測は ~93GB flat）。
- **ディスク（モデル）**：実行に要るのは ~28GB（GGUF transformer 16.54＋GGUF Gemma 6.8＋components 3.85＋upscaler 0.93＋設定）＝ComfyUI GGUF 構成（~25–30GB）と同等。現状 `models/` は 93.83GB だが **モノリス 42.98GB＋qat 重み 22.74GB（計 ~66GB）は comp=1/GGUF 経路では不要候補**（要・読み込み検証→落とせば公開フットプリント先行事例並み＝[[no-large-pagefile-disk-requirement]] 達成）。

### 10.5 状態と残課題（掃除を除く）
- **残課題C 完了。次＝Gradio 手動検証→AviUtl2 拡張統合**（本来の目的）。
- 未解決：【B】load/encode 一時 shared 溢れ ~2.6GB（§7.9）／【最適化】keep=1 を 720p で使う Gemma 移動 in-place 化／【連続上限】keep=0 の gen 漸増を連結実本数で再計測／【footprint】モノリス＋qat 不使用の確定→削減／【✅済 §10.7（2026-07-02 PASS）】comp=1/keep=0 での I2V マルチジョブ・音声連続／【D】README/spec 改訂。
- mock pytest 13 passed 維持（config 変更後）。アーティファクト＝`outputs/run720p*`・`outputs/multijob_*`（テスト出力・サンプラ CSV・段境界 JSON）。
- ※ディレクトリ掃除は専任の次セッションへ委譲（残課題から除外・ユーザー指示）。

### 10.6 解像度×尺の上限スイープ（2026-06-30 後半・1080p/1440p/4K）
直接ハーネス `scratchpad/gen_one.py`（本番同等＝comp=1/keep=0/bs8/vae512-64、API の÷64・解像度制限をバイパス）で、GPU+commit+段境界 VRAM を採取しながら昇順生成。
- **天井＝ロード時 Gemma encode max_alloc 15,098MB（解像度・尺に完全非依存の固定費）**。生成（denoise/upsample/VAE）は天井に対し余裕＝ユーザー仮説どおり。
- **denoise stage2 max_alloc ≈ 5,085MB + 0.175MB/token**（token=(W/32)(H/32)(1+(F-1)/8)。4点フィット＝1080p/1440p/4K の25f＋720p/121f アンカー。高トークン側で実測+3%上振れ）。**VAE decode は ~1.9GB 横ばい**（空間/時間タイル512/64で解像度・尺非依存）。

| 解像度(生成÷64) | 25f(≈1s) | **5秒(121f)** | 確認した実用上限 | 備考 |
|---|---|---|---|---|
| 1080p 1920×1088 | ✅ den2 6.5GB | **✅ den2 10.8GB / 4.3分** | ~9秒 | 余裕 |
| 1440p 2560×1472 | ✅ den2 7.6GB | **✅ den2 15.9GB / 11分** | ~5秒 | 限界点（den2が天井超→shared溢れ2.8GB・激遅化） |
| 4K 3840×2176 | ✅ den2 10.8GB | ❌（推定~49f=2s が上限） | ~2秒（49f,den2 15.5GB,8.7分） | — |
- **結論：16GB で 5秒動画は 1080p・1440p で実現可能、4K は ~2秒まで。** den2 が天井（~15.1GB）を超えると WDDM shared へ溢れて完走するが激遅。表示解像度へは crop（1088→1080 等）。AviUtl2 UI の解像度/尺の上限はこの実測表が根拠。アーティファクト＝`outputs/bigres_*`・`outputs/verify_*`（mp4＋sampler CSV＋段境界 JSON）。

#### 10.6.1 1080p 長尺の追い込み（音声付き・発話プロンプト）
1080p(1920×1088)で 9s/10s/11s を昇順生成（joint audio 自動・AAC/48kHz/stereo）:
| 尺 | フレーム | 結果 | den2 max_alloc | shared peak | commit% | 時間 |
|---|---|---|---|---|---|---|
| 9秒 | 217f | ✅ OK | 15,539MB | 2,462MB | 58% | 9.1分 |
| 10秒 | 241f | ✅ OK | 16,815MB | 3,892MB | 63% | 11.3分 |
| 11秒 | 265f | ✅ OK | 18,093MB | 5,506MB | 60% | 14.9分 |
- **den2 は物理 VRAM(16,376MB)を超えても完走**＝超過分は WDDM shared(commit 余裕60%)へ。**ハードOOMの壁は未到達**。1080p の上限は OOM でなく **速度（shared paging で ~+3〜4分/秒）**で決まる＝11秒は確実、12秒以降は1本20分超で実用性低下。
- 高トークン側フィット `den2 ≈ 10,796 + 0.194×(token − 32,640)` が ±1% で的中（9s予測15,555/実測15,539、10s予測16,745/実測16,815、11s予測17,934/実測18,093）。
- 音声は `LTXFastVideoPipeline.create()/generate()` 既定で生成（gen_one.py 直接経路でも AAC track 確認）。発話内容はモデル生成のベストエフォート（要・聴取確認）。アーティファクト＝`outputs/long_1080p_{9s,10s,11s}/`。

### 7.2 ステップ2 診断 A＝フェーズ別4指標（bs8・baseline, expandable_segments:True）
torch `allocated/reserved/max_alloc`（スパイク内）＋ perf-counter `dedicated/shared`（サンプラ）を境界ごとに突き合わせ:

| フェーズ | alloc | reserved | max_alloc | perf_ded | perf_shr |
|---|---|---|---|---|---|
| idle | 0 | 2 | 0 | 590 | 693 |
| Gemma encode+cleanup 後 | 21.1 | **104** | 15102 | 15867 | — |
| denoise stage1 前 | 1448.8 | **17452** | 16913 | 15952 | 3788 |
| denoise stage1 後 | 1705.6 | **18732** | 5159 | — | 3848 |
| 全体ピーク | — | — | — | 15988 | **3969** |

**判定**：
- **H1（Gemma 未解放）棄却**：encode+cleanup 後 alloc=21MB/reserved=104MB＝torch は Gemma を完全解放。
- **H3（実常駐）棄却**：denoise の live は alloc ~1.4–1.7GB、stage1 の max_alloc も 5.2GB のみ。
- **H2 確定（Windows 固有）**：denoise の live 5.2GB に対し **reserved 18.7GB（3.6倍）＝アロケータ断片化/居座り**。
  起点は **Gemma 解放後（reserved 104）→ denoise 前（reserved 17452）の急増＝transformer ロード時**。block-swap が
  GGUF transformer(16.5GB) を一旦 GPU に丸ごと載せ（peak 16913）→ブロックを CPU 退避するが、**`expandable_segments`
  が本機 Windows で no-op（`UserWarning: not supported on this platform`）のため空いた ~16GB セグメントがキャッシュに
  居座り**、16GB カードを超えて WDDM shared へ ~3.8GB 溢れる。

### 7.3 ステップ2 対策スイープ（denoise shared ピークで評価, baseline=3969MB）
| 構成 | reserved（fix後） | denoise ded ピーク | **denoise shared ピーク** | 生成時間 | 溢れ解消 |
|---|---|---|---|---|---|
| baseline | 17452→18732 | 15988 | 3969 | 210.9s | ✗ |
| `max_split_size_mb:512` | 18732 | 15904 | 3785 | 188.0s | ✗（≈baseline） |
| `max_split_size_mb:256` | 18732 | 15917 | 3743 | 188.9s | ✗（≈baseline） |
| `gc_threshold:0.6,max_split:256` | 18732 | 15917 | 3774 | 190.5s | ✗（≈baseline） |
| `backend:cudaMallocAsync` | 16736 | 15954 | 1805 | 124.6s | △（大幅減・残る） |
| **EC＝denoise 直前 `empty_cache()`** | **17452→1508** | **6438** | **742（ambient）** | **114.9s** | **✓ 完全消失** |
| block_swap=4 | 18732 | 15904 | 3869 | 201.4s | ✗（≈baseline） |

- 出力は**全構成バイト単位同一**（43706B, h264 384x256 9f, seed=10 決定論）＝メモリ挙動のみ変化・画は不変。
- **`max_split`/`gc_threshold` は Windows で無効（死枝）**。**block-swap 深度はレバーではない**（bs4≈baseline＝溢れは断片化で
  あり常駐重みでない裏付け）。`cudaMallocAsync` は改善するが単体では溢れ残存（任意の補助）。
- **診断 B の教訓**：当初の `empty_cache()` テストは Gemma 直後（reserved 既に 104）で無効だった＝**置き場所が誤り**。
  正しくは **transformer ロード後・denoise 直前**。
- **計測運用の改善（恒久原則化）**：C1 が baseline と同値になった時点で死枝と判断・停止すべきだった
  （max_split で約15分浪費）→ サブエージェントに「異常値/baseline 一致で即報告・続行判断」プロトコルを課す（memory `[[delegation-early-stop-protocol]]`）。

### 7.4 結論と確定した対策
- **真の対策＝transformer ロード後・最初の denoise 直前に `gc.collect(); torch.cuda.empty_cache()` を1回**。
  reserved 17452→1508MB に即落ち（alloc 1448 不変＝解放分は空セグメント）→ denoise が dedicated **6438MB に収容**、
  shared **742MB＝ambient**、**生成 210.9→114.9s（−46%）**。**＝denoise 工程の持続的 shared 溢れ（激遅化の主因）を解消（shared も計測確認）。※「全工程 16GB 内」ではない＝Gemma encode/transformer load は依然 ~2–2.5GB を shared へ一時溢れ（§7.9）**。
- **実装先（次＝ステップ3, 要承認）**：`create()` は遅延ロード＝transformer ロードは初回 `generate()` 内で起きるため、
  real worker `vendor/.../backend/_ltx_worker.py` の bootstrap monkeypatch で `ltx_pipelines.distilled.denoise_audio_video`
  をラップし初回 denoise 前に `empty_cache()` を入れるのが凍結境界を侵さずクリーン（worker は既に同種 monkeypatch 流儀）。
  併用候補：子 env に `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`（任意）。`max_split`/block-swap 深度は不採用。
- **スケールアップへの留保**：本fixは「居座り解放」であり高解像度でも有効だが、**1280×768 では denoise の活性自体が
  増える**ため、本fix単体では不足し §7.1 の活性軸技法（attention tiling／FFN チャンキング）が別途必要になる見込み。
- **アーティファクト**：`outputs/phase5b_diag/`（`diag_sweep.py`、`phase_timeline_{C1..C4,EC,BS4}.json`、
  `gpu_mem_{...}.log`、`timing_{...}.txt`、`diag_{...}.mp4`）。crash なし。

### 7.5 ステップ3 実装＋実機検証（empty_cache を worker へ実装, 2026-06-29）
- **実装**：`_ltx_worker.py` に `import gc` ＋ `ltx_pipelines.distilled.denoise_audio_video` のラッパ
  （各 denoise 直前に `gc.collect(); torch.cuda.empty_cache()`）。**凍結境界・status/metadata 不変・mock pytest 13 passed**。
- **実機検証（凍結 `LTXRunner.generate` 経路, perf-counter）**：
  - **T2V 384x256/9 PASS**：denoise shared **~997MB(ambient)**・dedicated 6446MB・wall 121s ＝ EC スパイク再現。ffprobe h264 384x256 9f。
  - **I2V 384x256/9（fresh worker）PASS**：denoise shared **~963–988MB(ambient)**・dedicated ~6.1–6.8GB・wall 118.9s。ffprobe OK。
- **評価＝採用**：denoise 溢れ解消・−42〜46%・crash 中立。**384x256 の denoise 持続溢れ解消・実装済み（shared 検証済。ただし load/encode の一時 shared 溢れ ~2–2.5GB は残＝§7.9）**。

### 7.6 ★別件で判明した production ブロッカー：worker 再利用（2ジョブ目）の native crash（bug#5 系・Phase 5B 起因ではない）
- **症状**：常駐 worker の **2ジョブ目以降の generate 冒頭で access violation（exit 139）**。最初の実機検証で reused worker の T2V→I2V の I2V が無言で死亡。
- **対照実験（`reuse_loop.py`：1 パイプラインを再利用し `T2V,I2V,I2V,T2V,I2V` をループ・fix 有/無）で empty_cache を完全 exonerate**：
  fix 有(2a)・無(2b) **両方が job2(I2V) の同一命令で crash**（job1 は両方 pass）。→ empty_cache は原因でも悪化要因でもない。
- **真因（faulthandler C-level traceback）＝per-job の Gemma テキストエンコーダ再ロード**：
  `gemma_gguf_quant_service.py:286`（base Gemma safetensors を CPU 再ロード）→ `sft_loader.py:36` の
  `f.get_tensor(name).to(device, non_blocking=True, copy=False)` → torch storage `__getitem__` で access violation。
  **§1.1 バグ#5（mmap-backed safetensors への CUDA 後 `non_blocking=True,copy=False`）と同一系**。distilled は各 generate で
  text encoder を `del→rebuild` するため、**再利用 worker は2ジョブ目で必ず Gemma を再 mmap ロード→このバグを踏む**。
  Phase 5A の「reused T2V→I2V 両 PASS」は非決定的な幸運（本対照では baseline も job2 で crash）。**OOM ではない**（crash 時 shared ambient）。
- **位置づけ**：Phase 5(B)（denoise VRAM）とは**独立の別タスク**だが、Approach W（常駐 worker・多ジョブ）の **production 信頼性ブロッカー**。
- **修正方向（未着手・要承認）**：(i) base Gemma state_dict をジョブ間でキャッシュ/常駐し再 mmap を避ける、または
  (ii) §1.1 バグ#5 で設計済の「`non_blocking`/`copy=False` 除去の可搬 monkeypatch」を
  **`gemma_gguf_quant_service` の base ローダ経路にも**適用（バグ#5 は「🔄適用中」のままで本経路は未カバーだった疑い）。
- アーティファクト：`outputs/phase5b_diag/reuse_loop.py`、`gpu_mem_reuse_{A2a,B2b}.log`、`reuse_{A2a,B2b}.stderr.log`（faulthandler 全文）。

### 7.7 ステップ4 試行＝失敗：sft_loader safe-patch では再利用 crash は消えない（真因はより深い, 2026-06-29）
既存 `sft_loader_safe_patch`（`SafetensorsStateDictLoader.load` を CPU 先読み＋`copy=True` 化）を `_ltx_worker.py` に
配線（`apply_sft_loader_safe_patch()`・mock pytest 13 passed・empty_cache fix 併存・worker ログに install 確認）。
だが **5ジョブ reuse テストは job1 で crash・未完走＝FAIL**。
- **faulthandler（patch 適用下でも crash）**：fault は patch 後の `.to(copy=True)` ではなく、**上流の
  `f.get_tensor(name)` が mmap を materialize する `torch/storage.py:470 __getitem__` で発生**。patch は転送 semantics を
  変えるだけで **mmap read 自体の fault を防げない**。Gemma base load は元々 `device="cpu"`（`gemma_gguf_quant_service.py:286`）で
  `non_blocking` は既に no-op、patch の実効デルタ（copy=True）はこの crash に無関係。
- **crash site は複数・非決定的**：safetensors mmap read（`sft_loader`）だけでなく **GGUF dequant
  `gguf_quant_service.py:402 _dequant_q6_k`** でも発生。job index も 1/2/3 とばらつく（バグ#5 access-violation 系）。
  **共通項＝per-job の Gemma テキストエンコーダ全再構築**（`distilled.py:96 __call__`→`patched_text_encoder`→
  base safetensors 再 mmap＋GGUF Gemma 再 dequant、毎 `_run_inference`）。この反復重ロードが断続的に native 破損。
- **未試行のローダ修正(b)** `safetensors.torch.load_file(..., backend="pread")`（mmap 不使用）は mmap read site には
  効く見込みだが **GGUF dequant site はカバーしない**ため単独では不十分。
- **真の修正方向（再評価・要研究＋計画）＝per-job 再構築をやめ Gemma をジョブ間キャッシュ**（両 crash site を同時除去）。
  ただし **VRAM 常駐は 16GB fit と非両立**なので、ロード済み Gemma を **CPU RAM にキャッシュ**し encode 時のみ GPU へ、の形が要る。
- **empty_cache denoise fix は影響なし＝維持**（reuse でも denoise 到達時は毎回 reserved 17452→1508 等・shared ambient 再確認）。
- worker の safe-patch 配線は **fix にならないため撤去済み（ユーザー決定 2026-06-29）**。worker は検証済み empty_cache fix のみ
  保持（`import gc`＋denoise monkeypatch 健在、mock pytest 13 passed）。`services/sft_loader_safe_patch.py` 自体は温存
  （将来の本修正で再利用可）。
- アーティファクト：`reuse_*.stderr.log`（faulthandler 全文 3 site）、`gpu_mem_step4_reuse.log`。

### 7.8 Phase 5(B) クローズ（2026-06-29）
- **Phase 5(B)＝denoise VRAM 低減：完了**。成果＝`_ltx_worker.py` の pre-denoise `empty_cache` monkeypatch（§7.5）。
  384x256 で denoise shared 溢れ消失（3969→742MB ambient）・生成 −46%・T2V/初回I2V 実機検証・mock pytest 13 passed・
  凍結境界不変。384x256 で **denoise 工程の持続的 shared 溢れを解消**（shared 検証済）。**※全工程 16GB 内ではない＝load/encode の一時溢れ ~2–2.5GB は残（§7.9）**。
- **次セッションの2大オープン項目**：(1) **再利用 crash の本修正**＝per-job の Gemma テキストエンコーダ全再構築を廃し
  **ジョブ間 CPU キャッシュ**（§7.6/§7.7、両 crash site 除去・VRAM 非両立に注意、要 research＋計画）。(2) **スケールアップ
  1280×768→720p クロップ**（活性軸＝attention tiling 配線＋FFN チャンキング移植、§7.1）。empty_cache fix は高解像度でも有効。

### 7.9 ★訂正：fit は「denoise 工程」限定。load/encode は依然 shared へ ~2–2.5GB 一時溢れ（2026-06-29 ユーザー指摘で精査）
EC run を perf-counter `shared` とフェーズ境界で時間相関（`gpu_mem_EC.log`＋`phase_timeline_EC.json`、`empty_cache` fix 適用下）:
| フェーズ | dedicated_max (MB) | shared_max (MB) |
|---|---|---|
| Gemma encode | 15874 | 2039 |
| transformer load | 15948 | **2457（全体ピーク）** |
| **denoise stage1** | 6272 | **742（ambient）** |
| **denoise stage2** | 6438 | **741（ambient）** |
| VAE/done | 6437 | 740 |

- **検証は dedicated 単独ではなく shared も記録していた**（ゆえに denoise=ambient と言える）＝計測は片手落ちではない。
- だが **empty_cache fix が解消したのは denoise 工程の持続的 shared 溢れ（baseline 3969→742MB）のみ**。
  **Gemma encode（shared ~2.0GB）と transformer load（shared ~2.5GB）の各フェーズは依然 dedicated ~15.9GB＋shared
  ~2–2.5GB＝16GB を一時超過**（ユーザーのタスクマネージャ目視と一致）。global shared ピークは baseline 3969→2457 に
  下がったが **ゼロではない**。
- よって **「真の16GB fit（全工程で共有ゼロ）」は未達**。本 fix の正しい主張＝**denoise 工程（生成時間の大半・激遅化の
  主因）の持続的溢れを解消し、denoise を ~6.4GB dedicated／shared ambient に収めた**（−46%）。
- **残課題（load/encode の一時 spill）**：transformer は block-swap が**一旦 full-GPU ロード(peak max_alloc 16913)→退避**
  するため load 時に 16GB 超過（直接 CPU ロードにすれば回避可）。Gemma encode は ~15.1GB で本質的にタイト（§5）。
  いずれも「再利用 crash 修正（Gemma ロード経路の作り替え）」「スケールアップ」と併せて扱うのが自然。

---

## 8. 残課題A（worker 再利用 crash）の本修正トライ → ★真因の再フレーム（2026-06-29 後半）

§7.6/§7.7 の「再利用 crash（exit-139）」の本修正に着手。2つの修正案を実機検証で**いずれも棄却**し、その過程で **crash の真因理解が「safetensors mmap ハンドルのバグ」から「Windows WDDM の near-full-VRAM/commit 圧迫由来の native 0xC0000005」へ大きく転換**した。**この §8 が残課題A に関する最新理解の正本**（§1.1 バグ#5・§7.6/§7.7 の「mmap が真因」という旧フレームを**上書き**する）。実コード・凍結層は不変、検証は throwaway スパイク（`outputs/phase5b_diag/`）。計測は `_gpu_mem_sampler.ps1`（dedicated/shared）＋ reuse_loop の `@@JOB@@`/marks。

### 8.1 試行1＝完全 non-mmap（whole-file `load(bytes)`）→ 棄却
- ユーザー方針「safetensors と GGUF dequant の両方を完全 non-mmap 化」に沿い、ComfyUI `--disable-mmap` 流の
  **ファイル全体を RAM へ読み `safetensors.torch.load(bytes)`** を `sft_loader_safe_patch.py` に実装（`safe_open`/`get_tensor` 不使用）。
  GGUF read も `np.array(tensor.data, copy=True)` で owned 化。
- **実機 reuse_loop で FAIL（job1 で死亡）**。真因＝**同じ `SafetensorsStateDictLoader.load` が 46GB の distilled
  チェックポイント（VAE/video_encoder）も読む共有ローダ**で、whole-file 読みが (a) **46GB を Python bytes 化→MemoryError**、
  (b) `load(bytes)` がバッファを zero-copy alias→`del data` で **pyo3 panic「deallocated bytearray … exported buffers」**。
  → **whole-file 非mmap はこのコードベースと構造的に非両立**。撤去し known-good へ復帰。
- 付随確定：installed **safetensors は 0.7.0**（`backend="pread"` は 0.8.0 以降＝**当該方針の前提が誤り**）。

### 8.2 試行2＝Gemma ジョブ間 CPU キャッシュ（Approach A）→ 棄却・むしろ逆効果
- `GemmaGGUFQuantStateDictLoader.load` を worker monkeypatch でラップし、merged Gemma state_dict を **CPU RAM に単一キャッシュ**
  （job1=MISS で CPU build→保存、job2+=HIT で disk/dequant skip→per-job `.to(GPU)`）。`services/gemma_sd_cache.py` に
  importable 化し worker＋reuse_loop（`CACHE=1`）が同一コードを適用。mock pytest 13 passed。
- **実機 FAIL かつ逆効果**：`CACHE=1` は **job1 で即 crash**（safetensors VAE load `sft_loader.py:36`）。対照の
  `CACHE=0`(known-good) は **job1・job2 PASS、job3 で crash**（GGUF dequant `gguf_quant_service.py:406 _dequant_q6_k`）。
  → cache は crash を**早めた**（~12GB の常駐 CPU キャッシュ＋開いた mmap が **committed memory を押し上げた**疑い＝§8.3 機構）。撤去し known-good へ復帰。

### 8.3 ★真因の再フレーム（WEB 調査＋実機データで確定）：mmap バグではなく VRAM/commit 圧迫
- **crash は非決定的**：site が run ごとに変わる（job1 で safetensors VAE / job3 で GGUF dequant）、job 番号もバラバラ。
  かつ **GGUF read は既に owned copy（非mmap）なのに `_dequant_q6_k` で落ちる**＝**mmap エイリアスでは説明不能**。
- **旧フレーム棄却（先行事例で裏取り）**：safetensors **#164 はファイルロック問題で crash ではなく、PR #166 で解決済**。
  ComfyUI **`--disable-mmap` は 0xC0000005 を直さない**と明示報告（#13220/#10896）。「バグ#5＝mmap ハンドル」は**誤診**。
- **確定した機構（PROVEN・出典）**：Windows **WDDM** では near-full VRAM での確保/転送が **clean な CUDA OOM ではなく
  native 0xC0000005** に化ける。pytorch **#178892**（16GB/WDDM、maintainer が **WDDM TDR** を指摘、**「VRAM が変動する
  小モデルで頻発」＝per-job 再構築 churn と符合**）、ComfyUI **#8298**（PyTorch は使用量の **1.5–1.7倍を commit**し、
  **物理 RAM が空いていても committed が上限到達で crash**）。crash は **(i) VRAM レール(TDR)** か **(ii) commit 上限枯渇** の
  どちらか（両者とも同じ 0xC0000005）。

### 8.4 本機環境の実測（proven レバーの突き合わせ）
| 項目 | 実測 | 判定 |
|---|---|---|
| NVIDIA ドライバ | **581.57 / CUDA 13.0**（cu128 要件 ≥570.65） | ✅ 十分。**除外** |
| ページファイル | C: 17GB(system) ＋ S: Initial 32GB/Max 64GB（計 ~81GB）。**S: PeakUsage 65533＝上限張り付き** | △ 大きいが Initial=32・**過去に commit 逼迫の痕跡** |
| RAM | 64GB（アイドル free 53GB） | 潤沢 |
- research の定番 proven fix のうち**ドライバは空振り**、**ページファイルは大きいが commit 逼迫の痕跡あり**（#8298 機構と符合）。

### 8.5 レバー scorecard（先行事例の実証強度・2026-06-29）
- **PROVEN/cheap**：ドライバ≥570.65（本機✅）、ページファイル拡大（commit 枯渇に効く・要再起動）。
- **PLAUSIBLE・最高 payoff**：**per-job の全 submodel 再構築をやめ、一度 CPU RAM に載せ GPU へストリーム**（CPU offload）。
  trigger が「再ロード×圧迫」＋ComfyUI Dynamic VRAM の設計が支持。ただし *この native crash を消した* 直接事例は無し（gap）。
  ※ Approach A の失敗は「やり方」次第＝commit を増やさない形が必須。
- **UNVERIFIED**：**load 前 `empty_cache` で 0xC0000005 を防いだ先行事例はゼロ**（OOM/断片化の文脈のみ）。補助どまり。
- **CONTESTED**：アロケータ（`--disable-cuda-malloc`/native は #9505 で効くが #9084 で逆に crash 誘発／cudaMallocAsync は
  我々の denoise では spill 減）。load フェーズの勝者は不明＝**本機で実測要**。`expandable_segments` は Windows で no-op。

### 8.6 現在のコード状態（known-good）
- worker `_ltx_worker.py`：**pre-denoise `empty_cache` monkeypatch のみ**（Phase 5B fix）。gemma cache も sft patch も**未配線**。
- 温存（dormant）：`services/gemma_sd_cache.py`、`services/sft_loader_safe_patch.py`（現状 non-mmap 版・未配線）。
- 残す軽微改善：GGUF read の `np.array(..., copy=True)`（無害）。reuse_loop の `CACHE=1`/`SAFE=1` トグル（既定 off）。
- mock pytest 13 passed・凍結境界不変。

### 8.7 次アクション（要・計測先行）
- **commit+VRAM 切り分け run**：reuse_loop を回しつつ `\Memory\Committed Bytes`/`Commit Limit` と VRAM dedicated/shared を
  同時サンプリングし、**crash 瞬間に commit が上限到達しているか**を見て **(i)VRAM レール vs (ii)commit 枯渇** を確定。
  → commit 枯渇なら**ページファイル拡大**、VRAM レールなら**churn 低減＋アロケータ実測**。empty_cache は機構確定後の補助。
- アーティファクト：`outputs/phase5b_diag/`（`run_cacheA.*`/`run_knowngood.*`/`gpu_mem_*.log`/`reuse_marks_*.json`/`reuse_loop.py`）。

### 8.8 ★commit+VRAM 切り分け診断＝crash は commit（仮想メモリ）枯渇で確定（2026-06-29）
instrumented run（`CACHE=0 EC=1 BS=8`、VRAM サンプラ＋新規 `_commit_sampler.ps1` で `\Memory\Committed Bytes`/`Commit Limit` を 0.5s 記録）。
job1,2 PASS、**job3 で crash**（faulthandler＝GGUF dequant `gguf_quant_service.py:406 _dequant_q6_k` ← `_load_gguf_gemma`
← per-job Gemma 再構築 `distilled.py:96`）。

| crash 瞬間(12:18:54–12:19:05) | 値 | 上限 | % |
|---|---|---|---|
| **Committed（仮想メモリ）** | **159,467 MB** | 160,329 MB | **99.46%＝枯渇** |
| Dedicated VRAM | 9,562 MB | 16,384 | 58%（**無関係**） |
| Shared VRAM | 444 MB | — | ambient |

- **commit の per-job 軌跡（leak ではなく transient スパイク）**：idle ~30GB(20%) → **各 per-job 再構築で ~125GB の commit
  スパイク**（job1 127GB/88.8% 生存、job2 148GB/**99.3% 危機一髪**＝OS が pagefile 拡張で回避、job3 155.7GB/**99.5%**→
  拡張 race 敗北で crash）→ 各ジョブ後に解放（job1後37%/job2後47%/crash後20%）。**累積 leak ではない**。
- run 中に **commit limit が 146→160GB へ OS 自動拡大**（pagefile を能動拡張するも job3 で間に合わず）。
- VRAM レール(16,049MB)は **job2 denoise 中に一瞬**触れただけ＝crash とは無関係。
- 機構＝**ComfyUI #8298**「PyTorch は使用量の 1.5–1.7倍を over-commit、物理 RAM/VRAM が空でも committed が上限到達で
  native 0xC0000005」と一致。**ディスク 70GB のモデルに対し ~125GB commit＝~1.8倍の過剰**。
- **★結論：crash は commit（仮想メモリ）枯渇。VRAM 系レバー（empty_cache 等）は的外れ。** §8.3 の「圧迫由来」を commit 側に精密化。
- **★ユーザー方針（次の調査）**：200GB ページファイル要求はリリース不可、先行事例（低VRAMフォーク/ComfyUI）はそんな空き容量を
  要さない＝**我々のコードが過剰 commit している**と見るべき。「**なぜ per-job 再構築が ~125GB commit するか**」を
  **仮説→WEB/コード裏取り→実験スコープ確定**の順で解明（手当たり次第の実験は禁止）。ページファイル拡大は band-aid として保留。
- 本機ディスク：C: 20GB空き / S: 184GB / D: 932GB / F: 2.4TB（大ページファイルは F:/D: なら可能だが band-aid）。
- アーティファクト：`outputs/phase5b_diag/`（`run_commitdiag.*`/`commit_mem_commitdiag.log`/`gpu_mem_commitdiag.log`/`_commit_sampler.ps1`）。
- **計測委譲の教訓（再発）**：バックグラウンドのテスト担当が Monitor を armed して return し最終相関を上げない事象が3回。
  対策＝長い run は「プロセス終了まで自実行内で block＋同一ターンでレポート」を課す、or サブエージェントは run/サンプラ起動のみ・
  相関は監督がログから実施（今回有効だったパターン）。`[[delegation-early-stop-protocol]]` に追補。

### 8.9 ★過剰 commit の真因究明（仮説→WEB/コード裏取り, 2026-06-29）＝我々のコードが過剰 materialize
**根本機構**（InvokeAI #7563 / ComfyUI #8298）：Windows は全 committed 仮想メモリに物理 backing（RAM/pagefile）必須。
safetensors ロードは (1) **mmap がファイルサイズ分の commit を予約**（safe_open 時点・重み読む前から）＋(2) **get_tensor が
anonymous コピーを materialize＝もう一度ファイルサイズ分** → **ピークでモデルサイズの 2×**（mmap 参照が切れるまで共存）。
＋CUDA/WDDM sysmem fallback の commit 上乗せ。→ **ディスク 70GB × 2× × per-job 全 submodel 再構築 ＝ ~125GB**。
先行事例が 200GB を要さないのは、この 2× と per-job 再構築を回避しているから。

**我々の具体的な過剰 commit（コード読解・file:line・深刻度順）**：
1. **★最大の無駄＝bf16 Gemma を毎ジョブ読んで即削除**：`gemma_gguf_quant_service.py:286` の base ロードが key-ops
   （`encoder_configurator.py:116` `AV_GEMMA_TEXT_ENCODER_KEY_OPS`）で **`language_model.*`（~24GB bf16 Gemma 本体）を
   materialize** → **L311-312 で削除→GGUF で置換**。我々は GGUF Gemma を使うのに使わない bf16 を毎ジョブ読んで捨てている。
   **読まずにスキップ可・数値不変**＝最も明白なバグ。
2. **per-job 再構築 churn**：`ModelLedger`＝`DummyRegistry`（無キャッシュ）で毎 generate 全 submodel 再構築。フォーク自身の
   `StateDictRegistry`（load once/keep resident）が**存在するのに未配線**。
3. **我々が足した `copy=True`（§8 GGUF mmap 回避用）が 17.76GB transformer＋7.3GB Gemma GGUF を毎ジョブ完全 anonymous commit**
   （皮肉にも先の修正が commit を悪化）。
4. **H5**：video_encoder＋transformer が両 stage 共存（`distilled.py` L188-189 まで解放されず）＝commit ピークが max でなく sum。

**commit 削減の先行事例レバー（VRAM でなく commit に効く）**：`backend="pread"`（非mmap, 要 safetensors≥0.8.0・本機 0.7.0）／
**load-once-resident（StateDictRegistry 配線）**／low_cpu_mem_usage(accelerate meta-device, PR#10604)／direct-to-CUDA ロード／
NVIDIA "Prefer No Sysmem Fallback"。**★罠：cpu_offload・VAE tiling・fp8 は VRAM のみ削減で commit には無効**（lever を誤らない）。

**実験スコープ（段階・各段で commit 計測・過剰実装回避・要承認）**：
- **Stage 1**：base read で削除される `language_model.*`（bf16 ~24GB）を**読み込み時点でスキップ**（数値不変）。最優先・低リスク。
- **Stage 2（必要なら）**：mmap 2× 削減＝`copy=True` 見直し／direct-to-device／pread。
- **Stage 3（なお必要なら）**：`StateDictRegistry` 配線で load-once/keep-resident（spike を sum→max）。
- 出典：InvokeAI #7563（根本）, ComfyUI #8298/#2288, safetensors `backend=pread`, diffusers low_cpu_mem_usage(PR#10604), MS WDDM commit。

### 8.10 Stage 1 実装＋commit 計測（2026-06-29）：実改善だが不十分
- **実装**：`gemma_gguf_quant_service.py` で base ロードの sd_ops を `_SkipGemmaLMSDOps` でラップし、削除される `language_model.model.*`
  （bf16 ~24GB、qat 5シャード由来）を**読み込み時点でスキップ**。target_vocab=262208 はヘッダ読みで保存。**skip-set≡delete-set を実データで証明
  （survivors 701 が new/old 完全一致）＝数値不変**。mock pytest 13 passed・scope は GGUF base ロードのみ。
- **計測（reuse_loop CACHE=0 EC=1 BS=8、commit+VRAM サンプラ）**：

  | job | Stage1 commit ピーク | pre-fix baseline | 差 |
  |---|---|---|---|
  | 1 t2v | 117.5GB/80.1% | 127GB/88.8% | −9.5GB |
  | 2 i2v | 132.2GB/92.2% | 148GB/99.3% | −15.8GB |
  | 3 i2v | 153.6GB/99.3%→**CRASH** | 155.7GB/99.5%→CRASH | −2.1GB |

- **結果：per-job commit スパイクは確かに低下（−10〜16GB）＝bf16 Gemma reload が主要因の1つと確認。だが job3 でまた crash**
  （commit 99.3%、limit が 146→160GB 自動拡大しても追いつかず）。crash site は今回 **VAE/video_encoder の safetensors load
  `sft_loader.py:36`**（非決定的だが常に commit 枯渇）。VRAM は無関係（crash 時 commit が要因）。
- **含意**：1ピース削っても、**per-job 再構築が ~70GB を毎ジョブ re-mmap＋re-materialize（mmap は読む量に関係なくファイル
  サイズ分 commit を予約＝InvokeAI #7563）**という構造が支配的。決定的 fix は **per-job 再構築自体を無くす load-once/keep-resident
  （Stage 3、フォークの未配線 `StateDictRegistry`）**。Stage 2(pread) は mmap 予約の半分を消すが safetensors 0.8.0 要・部分的。
- **計測委譲の改善が奏功**：「プロセス終了まで block＋同一ターン報告・Monitor 禁止」を課したテスト担当が正常に相関報告（3連続失敗を解消）。
- アーティファクト：`outputs/phase5b_diag/`（`run_stage1.*`/`commit_mem_stage1.log`/`gpu_mem_stage1.log`/`reuse_marks_stage1.json`）。

### 8.11 ★Stage 3 設計ブリーフ（load-once/keep-resident via StateDictRegistry, 2026-06-29・read-only 調査・★次セッションの実装対象）
- **機構**：`StateDictRegistry`（`registry.py:49-84`）は **CPU の生 state_dict をキャッシュ**（key＝resolved paths＋sd_ops.name）。
  HIT 時は `SingleGPUModelBuilder.load_sd`（`single_gpu_model_builder.py:71-75`）が `model_loader.load` を**呼ばない**＝
  disk read＋materialize（mmap/get_tensor/GGUF `np.array(copy=True)`）が**完全スキップ**＝per-job commit スパイクの源を断つ。
  現状 `DummyRegistry`（無キャッシュ）が既定で、どの pipeline も `registry=` を渡していないだけ（**未配線**・実装は存在し export 済）。
- **配線 seam（lib 編集不要・推奨）**：`ltx_fast_video_pipeline.py:137`（DistilledPipeline 構築直後・**service install 前**）で
  `reg=StateDictRegistry(); self.pipeline.model_ledger.registry=reg; self.pipeline.model_ledger.build_model_builders()`。
  `_target_device()`（`model_ledger.py:178-182`）が自動で CPU に切替→transformer/VAE/audio/upsampler が CPU-resident キャッシュ化。
- **必須の追加修正1点（Gemma）**：`patched_text_encoder`（`gemma_gguf_quant_service.py:903-906`）は `device=ledger_device=GPU` 直書きで
  `_target_device()` を無視→そのままだと Gemma キャッシュが **GPU テンソルを pin→16GB fit 破壊**。registry 有効時は **CPU build** へ変更要。
- **commit 試算**：steady CPU-resident ≈ transformer 17.8GB＋Gemma ~11GB＋VAE/audio/upsampler ~3GB ＋ idle ~30GB ≈ **~62GB**
  （上限 146GB の十分下）。per-job スパイク（~125GB）は **job1 warmup の一度きり**になり、job2+ は HIT で平坦化。
- **以前の Gemma cache 失敗（§8.2）との差**：あれは Gemma だけ常駐（+12GB）で他 submodel は毎ジョブ再 materialize＝スパイクに上積み（悪化）。
  **full registry は `_target_device` 系の全 load を一括置換**＝残留 materialize なし。
- **VRAM 中立**：HIT は CPU dict を返し build の `.to(cuda)` は out-of-place＝毎 generate に新規 GPU 配置・cache CPU テンソル不変。
  block_swap/`del transformer;del video_encoder`（distilled.py:188-189）/empty_cache 不変。fork は registry 再利用を**既に設計**
  （`apply_loras` の `destination_sd` ガード `single_gpu_model_builder.py:113`・`_target_device` CPU/GPU 切替が証拠）。
- **GGUF/quant 互換**：`GGMLQuantizedTensor` は CPU キャッシュ可（`.to` out-of-place で subclass/`_ggml_type`/`_float_shape` 保持）。
  cache dict の in-place 変異なし（dtype=None は read のみ／LoRA 経路は registry 非Dummy 時 `destination_sd=None` で fresh dict）。
- **十分性**：steady は decisive（job3 crash の源＝VAE/transformer 反復 reload が HIT 化）。**warmup（job1）は一度だけ ~125GB スパイク残**
  （job1 は 88.8% で通過実績＝多分十分、足りなければ pread/sequential empty_cache を併用）。
- **要・小確認（実装時に計測で潰す）**：①registry 有効時に Gemma cache が CPU か ②job2 で cache テンソルの device/attrs 不変か ③warmup スパイクが上限内か ④reuse_loop で全5ジョブ完走・commit が steady ~62GB・出力不変・mock 13。
- **実装規模**：`ltx_fast_video_pipeline.__init__` に ~4行＋Gemma を `_target_device()` 準拠にする1修正。Stage 1（bf16 Gemma skip）は warmup 軽減として併存。

---

## 9. ★残課題A の解決方針確定 — モノリス廃止＝コンポーネント・ファイル分離（Path B）＋音声 Phase 1 格上げ（2026-06-29 さらに後半）

§8.11 の Stage 3（StateDictRegistry 配線）を実装し実機検証したところ、**steady は bounded だが job1 warmup で commit 152.4GB/99.27% に達し native crash**（video_encoder の safetensors load）。§8.11 が留保していた「warmup が上限内か」が**否**と判明。これを受け、Web×コードの多角リサーチ（3エージェント・出典付き）で**真の解決方針が確定**した。**この §9 が残課題A の最新・正本**（§8 の「Stage 3 単独で解決」見込みを上書き）。

### 9.1 なぜ現状は ~200GB のディスク空きを要求してしまうか（200GB 制約の root cause）
- 我々は VAE/audio/projection を **46GB のモノリシック distilled チェックポイント**から、Gemma 基底を **24GB の qat safetensors** から読む。Windows の safetensors **mmap は読む量に関係なくファイルサイズ分の commit（仮想メモリ）を予約**（InvokeAI #7563）＋get_tensor で materialize＝**2×**。per-job 再構築（DummyRegistry）が毎ジョブ ~70GB を re-mmap＋re-materialize → **commit ~125–152GB スパイク** → 上限到達で exit-139（§8.8/§8.9/§8.10、Stage3 warmup も同 152GB）。
- これを crash させず通すには commit 上限（RAM＋ページファイル）が ~152GB 超必要 → 64GB RAM なら **~90–128GB のページファイル＝~200GB 近いディスク空き**を要求。**ユーザー方針によりリリース不可**（memory [[no-large-pagefile-disk-requirement]]）。**VRAM は無関係**（crash 時 dedicated 9–16GB・shared ambient）。

### 9.2 なぜモノリス廃止＝小分けファイルへ切替えるか（Path B 採用理由）
- **proven な ComfyUI 16GB レシピは RAM≥32GB・巨大ページファイル無しで動く**。理由＝**小さな単体ファイルだけを読む**（GGUF/fp8 transformer＋FP4/GGUF Gemma＋単体 Video VAE 1.45GB＋Audio VAE 365MB＋text projection 2.31GB）。**46GB モノリスを一切開かない**ので mmap 予約が小さく commit が bounded。
- リサーチ（ファイルヘッダ実測）で、これら**単体ファイルは実在・DL 可（Kijai/LTX2.3_comfy, Comfy-Org/ltx-2）・ltx_core のキーとほぼ一致**（audio VAE/vocoder/projection は完全一致、video VAE のみ `vae.` プレフィックス SDOps が要る）。**モノリス＋qat から読む全テンソルに単体の代替先があり、モノリスは回避可能**。
- 切替えは**ローダ（ModelLedger の各 builder の読み込み元）差し替えのみ**で、**凍結 FastAPI も推論パイプライン（GGUF transformer＋block_swap＋Gemma＋denoise）も無改変**。結果＝commit bounded・~30GB フットプリント（ComfyUI 同等）・**巨大 DL/ページファイル不要**・**DL も ~5GB の小ファイルが 70GB のモノリスを置換して縮む**。
- **棄却した代替**：①pread 非mmap（safetensors 0.8.0 が3週間前リリースで未枯れ・効果は RSS 実測のみで Windows committed-bytes 未確認・当の InvokeAI も採らずページファイルに逃げた・46GB DL を残す＝製品的に劣る）②公式 layer-streaming（pin rev に不在・GGUF と非互換で GGUF を捨てる羽目・`--offload cpu` は 36GB を pinned host RAM に常駐で commit 悪化）③ComfyUI 本体へ移行（凍結 API/可搬性が無い）。**＝大 pivot は不要、ローダのファイルソーシングだけ proven に寄せる**のが正解。
- **唯一の新規実装＝connector 供給**：text encoder が要求する `{video,audio}_embeddings_connector.*`（258テンソル）は単体 projection ファイルに無く、**transformer ファイル（＝我々の GGUF）内**にある。供給方式は別途調査・選択肢提示の上で決定（実装前に承認）。

### 9.3 なぜ音声生成を Phase 1 に格上げするか
- 音声経路をエンド・ツー・エンド調査した結果、**フォークは既に音声を joint 生成し output.mp4 へ mux 済み**だと判明：DistilledPipeline が `video/audio_context` 二系統で常時 audio を生成し `Audio(waveform, sampling_rate)` を返す → `LTXFastVideoPipeline.generate()` が audio VAE+vocoder で decode し **PyAV で AAC トラックとして mp4 に mux**。worker が書く mp4 に**既に音声が入っている**。**ネイティブ音声は純 torch・WSL 非依存**（WSL 依存は別物の外部 TTS/Foley）。
- 音声が失われるのは**我々の `crop_mp4` 再エンコードだけ**（`-c:a`/`-map` 欠落）。crop 無しの出力は**今日すでに音声付き**。
- 有効化の労力＝**LIGHT**：音声 VAE 系 builder を単体ファイルに向ける（Path B で実質タダ・`per_channel_statistics` キー同梱だけ要確認）＋`crop_mp4` に音声保持（~1行）＋任意の非破壊メタ追記。**最大コストは VRAM/品質検証**（本機 16GB で未検証＝フォークの未踏路。decode VRAM・metallic アーティファクト有無を実測）。
- Path B のローダ刷新が**どのみち音声 builder を触る**ので、後で再着手するより**今 Phase 1 に畳む**方が安い＝格上げ。

### 9.4 確定した次の作業（Path B ＋ 音声 Phase 1）
1. 小ファイル DL（~4GB、46GB＋24GB モノリス置換）。2. ModelLedger ソース差し替え（VAE/audio/projection→単体）。3. **connector 供給（方式は選択肢提示→承認）**。4. `crop_mp4` 音声保持。5. config 追加。6. Stage 3 registry は実装済（モノリス除去で warmup が bounded 化）。検証＝mock 13／reuse_loop で commit bounded・全5完走・出力等価＋音声／VRAM fit。副次＝24GB qat は vision_tower/multi_modal_projector 専用で T2V/最小I2V では未使用の公算→落とせれば commit 追加削減（要確認）。**ページファイル拡大／逐次 warmup は不要に（モノリス除去が根本解）**。

### 9.5 connector 供給の事実調査と決定（Option A＝GGUF transformer から読む, 2026-06-29）
モノリス廃止で text encoder に再供給が要る survivors＝**258個の `embeddings_processor.{video,audio}_connector.*` ＋ 4個の `feature_extractor.{video,audio}_aggregate_embed.*`**（`gemma_gguf_quant_service.py:38-44,184-187`）。read-only 調査の事実：
- **258 connector は我々の GGUF transformer 内に F32/BF16（非量子化）で存在**（video=73 F32+56 BF16、audio 同）→ 再供給は**コピー＋bf16 キャストのみ・dequant 不要**（`_dequant_to_bf16` の float 分岐で対応可）。
- connector は **GemmaTextEncoder の `embeddings_processor` に attach**（`encoder_configurator.py:111-114`、encode 時・transient）。transformer モデル `LTXModel` に connector submodule は無く、**transformer GGUF ローダは 258 を読むが捨てている**（`gguf_quant_service.py:557-594`、孤児リード）＝再供給は重複でなく唯一の生きた供給先。
- bf16 抽出時 ~3.85GB。**エコシステムに 258 connector の単体ファイルは無い**（`..._embeddings_connectors.safetensors` は中身が 4 aggregate_embed のみ＝上流 ComfyUI commit f266b8d で connector を diffusion model 側へ移動）。**proven スタックは connector を transformer ファイル内に保持＝我々の GGUF と同じ供給元**。
- 4 aggregate_embed は単体 projection `ltx-2.3_text_projection_bf16.safetensors`(2.31GB) が供給。
- **決定＝Option A**（ユーザー承認 2026-06-29）：connector を **transformer GGUF から読み**（先行事例と同じ供給元）、キー remap（`{video,audio}_embeddings_connector.*`→`embeddings_processor.{video,audio}_connector.*`）して text-encoder マージへ注入。理由＝(1) prior-art の供給元に倣う、(2) de-risk 済みパイプラインが今使うのと同一テンソルを**読み出し元だけ**替える＝挙動不変で正しさのリスク最小（適用箇所＝text encoder は無改変）。出力等価を検証で担保。
- 棄却：B（単体ファイル抽出＝前例なき成果物の新造・正しさは A と同じで部品増）、C（モノリス残置＝目的に反）、D（エコシステム単体ファイルは connector を含まず）。
- 参照：`encoder_configurator.py:111-114`／`gguf_quant_service.py:557-594`／`gemma_gguf_quant_service.py:38-44,184-187,614-620`。

### 9.6 Path B 実装＋Phase 4 実機結果（component-files が warmup crash 解消・per-job リーク露呈, 2026-06-29 さらに後半）
**実装（Phase 0-3, gate=`use_component_files`／env `LTX_COMPONENT_FILES`）**：
- 小ファイル DL（`models/ltx-2.3-components/`・project-local hf_home）：video VAE `vae/LTX23_video_vae_bf16.safetensors`(1.45GB)／audio VAE+vocoder `vae/LTX23_audio_vae_bf16.safetensors`(365MB)／text projection `text_encoders/ltx-2.3_text_projection_bf16.safetensors`(2.31GB)。
- `_install_component_sources()`（`ltx_fast_video_pipeline.py`、registry 配線後・GGUF install 前）：VAE builder→video VAE＋`vae.` 前置 SDOps チェーン、audio/vocoder builder→audio VAE（remap 不要）。
- text encoder（`gemma_gguf_quant_service.py` 改修・Option A）：4 aggregate_embed を projection ファイルから、258 connector を GGUF transformer から `model.diffusion_model.*_embeddings_connector.*`＋bf16 で注入。
- `crop_mp4`（`services/video_io.py`）：`-map 0:v -map 0:a? -c:a copy` で音声保持（Phase 1 音声）。
- config 追加：`model.component_{video_vae,audio_vae,text_projection}_path`＋`vram.use_component_files`。worker は env `LTX_COMPONENT_FILES`（既定0）。reuse_loop ハーネスも配線。
- **等価性（GPU なし実証）**：VAE/audio ビルダ最終キー＝モノリスと完全一致／**connector 258/258 ビット一致（max abs diff 0.0）**＋aggregate_embed 4/4 一致／マージ後キー完全一致（701）／mock pytest 13 passed。**use_component_files=ON でモノリス(46GB)はどのビルダからも開かれない**ことを検証。OFF 経路は不変。

**Phase 4 実機（reuse_loop, `LTX_KEEP_RESIDENT=1 LTX_COMPONENT_FILES=1 EC=1 BS=8`, 5 jobs, 終了まで block＋同一ターン報告）**：
| job | 結果 |
|---|---|
| 1 t2v | PASS 111.9s |
| 2 i2v | PASS 95.6s |
| 3 i2v | PASS 111.5s |
| 4 t2v | PASS 132.0s |
| 5 i2v | **CRASH**（native, `block_swap_service.py:86` `module.to` via `patched_transformer`） |
- denoise dedicated VRAM が **15.1→18.3GB と毎ジョブ単調増加**、commit が **160.5GB/100%**（上限 ~160GB 自動拡大）まで**ジョブ毎にラチェット**→job5 で枯渇 crash。
- **結論**：**component-files が warmup commit-exhaustion を解消（以前 job1 即死＝0 generates → 今回 4 実生成）＝Path B の中核は成功**。だが**別の per-job リーク**が露呈：commit・VRAM が generate ごとに増加（ディスク再 mmap でなく per-generate 確保の解放漏れ）。**registry（Stage 3）だけでは commit が平坦化しない**ことも判明。
- progression（reuse 検証3パス）：Stage3(registry のみ)→Gemma device bug 発見・修正／Stage3b→job1 で 46GB モノリス mmap crash 152GB／**PhaseB(component-files)→4 generates・job5 で per-job リーク crash 160GB**。
- **リーク root-cause は未確定（前セッションでコード read が permission バグで拒否され診断ブロック）**。仮説：(a) block_swap pinned バッファ蓄積（crash site 一致・有力）、(b) registry 非HIT で毎ジョブ再 materialize、(c) per-generate GPU モデル解放漏れ。**次セッションの最優先＝この per-job リークの診断→修正**。
- 残：per-job リーク（最優先）／qat-drop（24GB・分析上 T2V/最小I2V で安全・未実装・`_return_model` meta 短絡対応要）／音声 ffprobe 確認・VAE ビット一致確認（未取得・ログ/4 mp4 はディスク保全）。
- アーティファクト：`outputs/phase5b_diag/`（`commit_mem_phaseB.log`/`gpu_mem_phaseB.log`/`reuse_phaseB.stderr.log`/`reuse_marks_PHASEB.json`/`reuse_PHASEB_job{1..4}_*.mp4`）。

### 9.7 ★per-job リークの診断確定＋修正＋実機検証 PASS（2026-06-30）＝残課題A 完全解決
§9.6 が残した per-job リークを、**read-only 並列診断（仮説 a/b/c 切り分け）→ 非破壊・計装run で按分実測 → 標的2パッチ → 6ジョブ実機検証**の順で解決した。**この §9.7 が残課題A の最終結論**。

**診断（4並列 read-only サブエージェント＋ログ法医学）**：
- **registry（Stage 3 keep-resident）は HIT している＝仮説b は棄却**。builders は `single_gpu_model_builder.load_sd`→`registry.get` を通り、キーは `sha256(resolved_paths + sd_ops.name)` で安定、ジョブ間で clear されない（`registry.py:49-84`）。
- **transient 解放漏れの素朴版＝仮説c も否定**。denoise working-set は毎ジョブ一定（torch 床にのみ蓄積）。
- **真因＝block_swap 保持（仮説a）**。`BlockSwapService` は常駐単一インスタンスで、`install()` が毎 generate で再構築される transformer を `_installed_transformers` に append し（`block_swap_service.py:92`）`uninstall()` を一度も呼ばない。`patched_transformer()`（`ltx_fast_video_pipeline.py:485-490`）が毎ジョブ新 transformer を建てる（`model_ledger.transformer()` は明示的に非キャッシュ）ため、末尾 `del transformer` はこの list 参照のため効かず、過去ジョブ transformer が GC されない。

**計装run で按分を実測確定（非破壊 monkeypatch、`reuse_loop_diag.py`、KEEP_RESIDENT=1 COMPONENT_FILES=1 EC=1 BS=8、4ジョブ）**：
- `len(_installed_transformers)` が **1→2→3→4**＝毎ジョブ +1 で累積（リークの直接証拠）。
- registry キャッシュ7エントリは **全ジョブ通じて安定・全て CPU のまま**＝**in-place `.to` がキャッシュを CUDA 汚染する説（当初の C 仮説）は REFUTE**。`.to`/`_apply` は `param.data` を新テンソルに差し替えるだけでキャッシュ本体の CPU テンソルは温存される。→ **`model_ledger.py` の in-place `.to` を out-of-place 化するパッチは不要**（vendored `.venv` を触らずに済む）。
- registry MISS は warmup の7回のみ・job1-4 は 100% HIT。CPU キャッシュ総量 ~32.3GB を一度だけ add。
- **両軸の整合**：leaked transformer は自身の block を保持し、block_swap が大半を `block.to("cpu")` で退避＝**新規 CPU コピー ~18-20GB（§9.6 の commit +18GB/job）＋GPU 窓 ~1GB（VRAM 床 +~1076MB/job）**。両者は同一の leaked transformer で説明でき、`installed_transformers` 1→2→3→4 が決定的証拠。

**適用パッチ（2点・いずれもフォーク製品コード＝`vendor/.../backend/`、`.venv` 配下でない＝持続的・凍結境界外）**：
1. **`services/block_swap_service.py` `install()`**：`self._installed_transformers.append(transformer)`（旧 L92）の直前に `self._installed_transformers.clear()` を追加（keep-latest 化）。前ジョブ transformer（pipeline が既に `del` 済）を list から外して GC 可能にする。`uninstall()` は本経路で未使用なので clear は安全。
2. **`_ltx_worker.py` `_do_generate()`**：`_emit("done", ...)`（L195）直後に `gc.collect(); torch.cuda.empty_cache()` を追加。leaked transformer は `swapped_forward` クロージャの参照循環を持つため、参照を外しても回収には gc が要る（`gc`/`torch` は既存 import）。
- ※ これらは vendored（gitignore 配下）に在る。vendor/ 再生成時は本 §9.7 の通り再適用すること。

**実機検証 PASS（6ジョブ、旧 job5 crash 点を越える、`DIAG_TAG=VERIFY` 同条件）**：
| 指標 | 修正前 | 修正後 |
|---|---|---|
| 完走 | job5 で native crash | **全6 pass・exit 0** |
| `installed_transformers` | 1→2→3→4 | **1 で一定** |
| torch alloc 床 @JOB DONE | +~1076MB/job | **平坦 1082→1085MB（ジッタのみ）** |
| denoise dedicated peak | 15.1→18.3GB | **~16.18GB 定常** |
| commit @JOB DONE | +18GB/job→160.5GB/100% crash | **平坦 ~82%（93GB 前後）** |
| registry キャッシュ | — | 7エントリ全 CPU 安定・MISS は warmup の7のみ |
| 出力 mp4（job1-4） | — | **PHASEB と SHA256 バイト一致**（計算不変・メモリ寿命のみ修正） |

mock pytest 13 passed（app `.venv`・凍結経路不変）。アーティファクト：`outputs/phase5b_diag/`（`reuse_loop_diag.py`／`diag_probe_{LEAKDIAG,VERIFY}.json`／`reg_events_*.json`／`reuse_marks_VERIFY.json`／`commit_mem_VERIFY.log`／`gpu_mem_VERIFY.log`／`reuse_VERIFY_job{1..6}_*.mp4`）。

**残課題A（worker 再利用 crash）はこれで完全解決**。残るオープン項目＝音声 Phase 1 の実機確認（ffprobe で AAC トラック有無・decode VRAM・metallic アーティファクト）／qat-drop(24GB)／720p スケールアップ（残課題C）。

### 9.8 音声 Phase 1 実機確認（2026-06-30 PASS）

§9.3 で Phase 1 へ格上げした native joint audio を、本機 16GB で実機確認した。**結論＝PASS**。この §9.8 が音声 Phase 1 の最終結論。

**結論**：LTX-2.3 の native joint audio は 16GB 実機で正常動作する。効果音・音楽・**セリフ(発話)**のすべてを生成でき、AAC として mp4 に mux され、16GB に余裕で収まる。**Phase 1 audio は PASS**。

**AAC 実在（客観・ffprobe）**：保全済み `outputs/phase5b_diag/reuse_VERIFY_job{1..6}.mp4` の**全6本に AAC/48kHz/ステレオ音声トラック**が存在。音声長 1.090s vs 映像長 1.125s（1フレーム差でほぼ一致）。映像は H.264 384×256/9f。

**crop 音声保持**：`services/video_io.py` の `crop_mp4`（L105-123、`-map 0:v -map 0:a? -c:a copy`）で job6 を 320×192 にクロップ → AAC がストリームコピーで**無劣化保持**されることを ffprobe で確認。

**decode VRAM（実測）**：単発ジョブの decode 区間 peak ＝ dedicated ~2.7GB / shared ~0.5GB。6ジョブ VERIFY 実行では decode ~6.3GB（次ジョブのプリロード混入）。いずれも denoise（~14.3〜15.3GB）・全体ピーク（モデルロード時 ~15.9GB）より**大幅に低く、音声デコードは VRAM のボトルネックでない**。16GB 安全。

**主観確認（ユーザー）**：
- T2V job6＝赤い車＋楽しげな音楽＋タイヤ/エンジン音（内容追従）。
- I2V job2＝映像はほぼ同じだが別の音楽/エンジン音（画像条件付き経路でも joint audio が機能・音声は毎回生成される本物）。
- 新規 T2V 発話テスト `outputs/phase5b_diag/reuse_hello_job1_t2v.mp4`（プロンプトで人物が "Hello" と発話）→ ユーザーが再生し**明瞭に "Hello" と聞こえることを確認**。

**注意/限界**：
- (a) GPU サンプラ `_gpu_mem_sampler.ps1` は Get-Counter ベースで実効間隔 ~2秒（`-IntervalSec 0.25` を渡しても短縮されない）。decode 窓 ~2秒に有効サンプルが実質1点のみで、2秒未満の瞬間ピークは未捕捉。ただし decode の ~2.7GB という余裕（16GB まで +13GB 超）から、この限界は結論に影響しない。真の 0.25s 採取が要れば NVML/`nvidia-smi --loop-ms` 方式への置換が必要（未実施）。
- (b) 検証は 384×256/~1.1秒の小クリップのみ。高解像度・長尺の音声挙動はスコープ外＝**残課題C（スケールアップ）**。
- (c) transformer は Q4_K_M 量子化でフル bf16 公式とビット一致ではないが聴感良好。

---

### 10.7 comp=1/keep=0 I2V 連続＋音声 実機検証 PASS（2026-07-02）＝§10.5【未検証】クローズ

§10.3 は comp=1/keep=0 マルチジョブを **T2V のみ**実測、§9.7/§9.8 の I2V 含む reuse は **keep=1**（再利用crash診断経路）だった。本節は残る穴＝**production default（keep=0/comp=1/bs8/vae512-64/te-offload/dit-cpu-load）での連続 I2V＋音声**を、①直接ハーネス と ②本番API経路 の両方で実測し **PASS**。

**手法**：① `outputs/phaseB_i2v_audio/i2v_audio_loop.py`（`reuse_loop_diag.py` を first-party `engine/` へ再ポイント。create() は `engine/worker.py` を鏡写し＝gemma_root=tokenizer-only, vae512/64, comp=1, keep=0, te_offload/dit_cpu_load=env）で連続 I2V×4 @384×256/9f、commit+GPU サンプラ併走。② `run.ps1` 起動の本番 FastAPI に `POST /api/v1/generate` で I2V×3 @512×320/49f を逐次投入（各 completed を待って次＝single-job 409 guard 経由）。両者 keep_resident=0（既定）・音声は distilled pipeline が自動生成。

**① 直接ハーネス I2V×4 @384×256/9f**：

| job | result | wall | denoise dedicated peak | installed_transformers | 出力 |
|---|---|---|---|---|---|
| 1 | pass | 115.2s | 8,441MB | 1 | aac48k + h264 384×256/9f |
| 2 | pass | 128.8s | 9,527MB | 1 | 同上 |
| 3 | pass | 134.6s | 9,523MB | 1 | 同上 |
| 4 | pass | 132.4s | 9,520MB | 1 | 同上 |

- PIPELINE_CREATED_OK→4/4 pass→RUN_DONE_ALL_JOBS、harness exit 0。
- **commit peak 86.0%**（committed 87.9GB / limit 102GB）＝§9.7(82%)/§10.3(89.5%)と整合、95%未満で横ばい（ジョブ間累積なし）。
- **installed_transformers=1 を全4ジョブ維持**（§9.7 リーク修正の不変条件＝リーク無し）。denoise dedicated peak は job1=8.4GB(初回)以降 ~9.5GB 定常（漸増なし）、GPU memory.used max 12.0GB（<16GB）。
- 全4本 ffprobe＝aac/48kHz + h264 384×256/9f（音声≈53フレーム/1.1s、§9.8 と同傾向）。
- 注1：wall は `CUDA_LAUNCH_BLOCKING=1` 込みで本番速度ではない（native fault pinpoint 用）。注2：`_diag_probe` の cache-device 監査は first-party の DummyRegistry に `_state_dicts` が無く N/A（無害）。リーク判定は installed_transformers で直接取得済。注3：初回ランは launcher の `2>&1｜Tee`＋`$ErrorAction=Stop` が torch の pynvml FutureWarning(stderr) で中断＝PS5.1 の既知落とし穴。`Start-Process` の OS レベル redirect に修正して再走（エンジンは無関係）。

**② 本番API I2V×3 @512×320/49f（run.ps1・REAL worker 自動選択）**：

| job_id | HTTP | status | gen | 出力 |
|---|---|---|---|---|
| ca6515ea | 202 | completed | 129.2s | aac48k/2ch + h264 512×320/49f |
| 65e87d28 | 202 | completed | 120.5s | 同上 |
| 7727dd03 | 202 | completed | 133.0s | 同上 |

- `logs/ltx_worker.log`：`GENERATED_OK peak_vram_mb=8440 / 9525 / 9522`＝①の denoise peak とほぼ一致＝**両経路の強い相互検証**。GENERATE_FAILED/access-violation/OOM 無し（ログ内の唯一の GENERATE_FAILED は過去セッション由来）。
- app層＝`pipeline_manager` の between-job `safe_memory_cleanup()`＋metadata書出し＋single-job 409 guard＋常駐worker再利用（@@LTX@@）を通し3本連続クリーン。後片付けでポート解放・python残留なし。
- client側の一過性エラー（job1&2 の PS5.1 `Invoke-WebRequest` -1、job3初回の curl クォート由来 422）は駆動側の問題でサーバ実欠陥ではない（422 は入力検証が正しく効いた証拠）。

**結論**：§10.5 の【未検証】comp=1/keep=0 I2V マルチジョブ・音声連続 を **CLOSE**。Phase 2「5秒クリップの I2V 連結（終了フレーム→次開始フレーム）」の前提が両経路で満たされた。アーティファクト＝`outputs/phaseB_i2v_audio/`（`i2v_audio_loop.py`・`run_i2v_audio.ps1`・`reuse_I2VAUD_job{1..4}_i2v.mp4`・`diag_probe`/`reuse_marks`/`commit_mem`/`gpu_mem`_I2VAUD.*）／本番出力＝`outputs/<job_id>/output.mp4`。

---

## 11. ★`--te-offload`（Gemma text-encode ピーク低減）の実装＋A/B 実機検証 PASS（2026-07-01）

残課題B（load/encode の一時 shared 溢れ／§7.9）のうち **Gemma text-encode ピーク（peak ①）** を下げる機能 `--te-offload` を実装し、A/B で実機検証した。**この §11 が te-offload の最終結論。**

### 11.1 機能と機構
- **`--te-offload` 起動フラグ（既定 ON、`--no-te-offload` で無効化）**。Gemma テキストエンコーダを CPU オフロード・モードで動かす。
- **機構＝逐次 per-layer ストリーミング（full-CPU-encode ではない）**：GGUF 量子化された Gemma decoder 48 層を **CPU 常駐**のまま保持し、text-encode forward 中に GPU へ **2層ずつストリーム**する。**compute は GPU に留まる**。実証済み BlockSwapService のスライディングウィンドウ方式を Gemma の decoder 層へ適応し、GGML バッファ対応化（GGUF 量子化重みは parameter でなく buffer）したもの。
- **monolith-free / component-file 不変条件を保持**：ストリームするのは GGUF 量子化層のみで、**24GB bf16 モノリスの再 materialize は無い**（以前棄却した「full CPU encode」経路を採らなかった理由＝§2.10/§8 の commit 枯渇回避）。

### 11.2 配線（次セッションが追えるよう entrypoint を記録）
`main.py`（argparse `BooleanOptionalAction --te-offload`）→ `config.vram.te_offload_text_encoder`（既定 True・`config.py`＋`config.yaml`）→ `services/low_vram.py` の内部ノブ（凍結 `_STATUS_KEYS` には**入れない**）→ env `LTX_TE_OFFLOAD`（1/0、`services/ltx_runner.py` で設定）→ `vendor/LTX-Desktop-LOW-VRAM/backend/_ltx_worker.py` で読取 → `LTXFastVideoPipeline.create(te_offload_text_encoder=)` → `_install_gemma_gguf(te_offload=)` → `GemmaGGUFQuantLoaderService(layer_offload=)` ＋ 新規 `vendor/LTX-Desktop-LOW-VRAM/backend/services/gemma_layer_offload_service.py`（`GemmaLayerOffloadService`, `layers_on_gpu=2`）。

### 11.3 A/B 実機検証（512×320 / 49f / 8 steps / seed=12345・distilled）
| 指標 | OFF（`--no-te-offload`） | ON（既定） |
|---|---|---|
| encode-phase Dedicated peak | 15,839 MB | **10,540 MB**（粗 ~2s サンプラ＝下限） |
| encode-phase WDDM Shared spill | 1,616 MB | **0 MB（baseline）** |
| 全体 worker `peak_vram_mb` | 16,944 | 16,944（**不変**） |
| wall-clock | 135.1s | 117.1s |
| 出力 | valid mp4 | valid mp4・**OFF とバイト一致** |

- **encode ピーク低減 −5,299 MB（−33%）／encode 時の shared 溢れ消失／wall-clock ペナルティ無し／出力同一。**

### 11.4 720p ON スモーク
- 1280×768 / 121f / seed=12345：完走・valid mp4（~1.26MB）・**168.2s**・`peak_vram_mb` 16,944・encode-phase Dedicated ~10,467 MB / spill 無し。
- **→ encode ピークが解像度・フレーム数に非依存である**ことを確認（§5/RESOLUTION_DURATION_CAPABILITY §1 の「天井＝固定費」と整合、その固定費を te-offload が下げた形）。

### 11.5 ★スコープ注意（次セッションが誤解しないよう明記）

- **（2026-07-27追記）以後の将来項目の管理は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md)へ一本化した。** 本節の「peak ② の低減は将来課題」は同書**§4-3**（VRAMの残レバー3件）の①にあたる。以下の実測記録は不変。
- **全体ジョブ `peak_vram_mb` は両モードとも 16,944 で不変**＝**16GB の天井は denoise / transformer-load 段（peak ②）で決まり、`--te-offload` はそこに触れない**。te-offload が下げるのは **Gemma text-encode ピーク（peak ①）** のみで、encode 時の shared 溢れを消すだけ。**peak ② の低減は将来課題**（§7.9 の「transformer を直接 CPU ロード」と地続き）。2つのピークは逐次で、その max がジョブ天井を決めるという前セッションの所見（§7.9）どおり。

### 11.6 チューニングレバー

- **（2026-07-27追記）以後の将来項目の管理は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md)へ一本化した。** 本節の`layers_on_gpu`未露出は同書**§4-3**の②にあたる。以下の実測記録は不変。
- `GemmaLayerOffloadService(layers_on_gpu=2)`：**1 に下げると encode ピークがさらに下がる**／3-4 に上げるとピークと速度をトレード。**現状ハードコード**（`config.yaml` 未露出）。露出は任意の将来課題。

### 11.7 テスト
- mock pytest **13 passed**（`tests/conftest.py` の `_make_args` フィクスチャに `te_offload=None` を追加＝parse_args 既定をミラー。GET /status `vram_optimization` 契約テストも緑）。

**commit**：`96f41c4`（main・push 済）。

---

## 12. ★`--dit-cpu-load`（transformer ロード時 GPU スパイク除去）の実装＋A/B 実機検証 PASS（2026-07-01）

残課題B（load/encode の一時 shared 溢れ／§7.9）のうち、§11 で残した **transformer（DiT）ロード段（peak ②）の一過性 GPU スパイク** を除去する機能 `--dit-cpu-load` を実装し、A/B で実機検証した。**この §12 が dit-cpu-load の最終結論。**

### 12.1 機能と機構
- **`--dit-cpu-load` 起動フラグ（既定 ON、`--no-dit-cpu-load` で無効化）**。transformer ロード時の一過性 ~16.9GB GPU スパイクを除去する。
- **従来の挙動**：block-swap は、凍結 `ModelLedger.transformer()` が **48 個の GGUF transformer ブロックを一旦すべて GPU に materialize（~16.9GB）してから** CPU へ退避していた。この「一旦 full-GPU ロード→退避」がジョブ天井（peak ②）を作っていた（§7.9）。
- **新挙動**：transformer を **直接 CPU RAM 上に構築**し、**block 以外の submodule のみ** GPU へ移す。ブロックは CPU に留め、既存の `BlockSwapService` が denoise 窓ごとに CPU↔GPU ストリーム（**この機構は不変**）。**compute は GPU に留まり・出力はバイト単位で同一**。full-model の GPU materialization が無くなるためスパイクが消える。
- **実装＝独立サービス＋最小配線**（§11 の `--te-offload` と同じ流儀）：新規 `vendor/LTX-Desktop-LOW-VRAM/backend/services/dit_cpu_load_service.py`（`DitCpuLoadService`）＋最小配線。
  - **build-on-CPU 機構**：`ltx_fast_video_pipeline.py::_install_block_swap` の `patched_transformer` 内で、凍結 `ModelLedger.transformer()` 呼び出しの周囲だけ `ledger.device = torch.device("cpu")` に一時設定（try/finally で復元）。これにより `_target_device()` も末尾の `.to(self.device)` も CPU に解決され、**full-model の GPU materialization（＝スパイク）が起きない**。`.venv` 凍結の `ltx_core`/`ltx_pipelines` は**改変しない**。
  - **`DitCpuLoadService`**：non-block の leaf テンソルのみを、モジュール単位の `recurse=False` walk（`_parameters`/`_buffers`）で GPU へ移す。LTXModel に直付けの `scale_shift_table`/`audio_scale_shift_table` を拾い、block サブツリーは決して巻き込まない。GGML uint8 buffer は `tensor.to`（`GGMLQuantizedTensor.to()` を尊重）で移動。block の同定は `BlockSwapService._get_blocks()` を単一ソースとして用い、ストリーミングフックと食い違わない。**ガード**：`blocks_on_gpu >= total`（swap 実質 OFF）なら全部を GPU に移す（OFF 等価）。

### 12.2 配線（次セッションが追えるよう entrypoint を記録）
`--te-offload` と同一の経路をミラー：
`main.py`（argparse `BooleanOptionalAction --dit-cpu-load`・既定 None ＋ `build_app` override）→ `config.py` `VramConfig.dit_cpu_load: bool = True` → `config.yaml` `vram.dit_cpu_load: true` → `services/low_vram.py`（`LowVramSettings.dit_cpu_load`・`build_low_vram_settings` でコピー・凍結 `_STATUS_KEYS` には**入れない**）→ `services/ltx_runner.py` で env `LTX_DIT_CPU_LOAD`（1/0）→ `vendor/LTX-Desktop-LOW-VRAM/backend/_ltx_worker.py` で読取（既定 "1"）→ `LTXFastVideoPipeline.create(dit_cpu_load=)`。`tests/conftest.py::_make_args` に `dit_cpu_load=None` を追加。`run.ps1` にフラグを記載。

### 12.3 A/B 実機検証（512×320 / 49f / 8 steps / seed=12345・distilled）
`dit_cpu_load` のみを切り替え、他のノブは本番同等（GGUF transformer+Gemma、block_swap_blocks_on_gpu=8、vae 512/64、component_files ON、keep_resident OFF、te_offload ON）。
| 指標 | OFF（`--no-dit-cpu-load`） | ON（既定） |
|---|---|---|
| transformer-load Dedicated peak | 15,815 MB | **1,444 MB**（−14.4 GB） |
| denoise Dedicated peak | 6,558 MB | 6,558 MB（不変） |
| 全体 Shared peak | 2,612 MB | **419 MB**（溢れ消失） |
| wall-clock | 109.40 s | 102.41 s（退行なし） |
| 出力 SHA256 | 9E058F6E…（113,329 B） | 9E058F6E…（**同一・バイト一致**） |

- **transformer-load ピーク低減 −14.4 GB／load 時の shared 溢れ消失／wall-clock ペナルティ無し／出力同一。**
- **裏付け（torch `max_memory_allocated`）**：OFF は denoise 直前に **16,944 MB** へ跳ね上がる（＝§7.9 で記録したジョブ天井＝peak ②）。ON はその跳ね上がりが消え、te-offload 済 text-encode ピーク **9,165 MB** を一度も超えない。⇒ **このサイズでは全体ジョブ天井が 16,944 → ~9.2GB に低下**（peak ② が除去され、残る天井 = peak ① text-encode＝既に te-offload で低減済）。

### 12.4 720p ON スモーク
- 1280×768 / 121f / 8 steps / seed=12345、`dit_cpu_load=True`（既定）：完走・OOM 無し・**156.0 s**・Dedicated peak 14,611 MB（16 GB 内）・Shared 427 MB（溢れ無し）・valid mp4（1280×768/121f・音声あり・5.04 s・1.20 MB）。

### 12.5 ★スコープ注意（次セッションが誤解しないよう明記）
- 除去したのは **peak ② のうち「transformer-LOAD materialization スパイク」成分のみ**。**denoise stage2（peak ② の denoise 成分）はトークン数に比例して依然成長する**（`RESOLUTION_DURATION_CAPABILITY.md §2` の den2 推定表は**不変**）。大解像度・長尺では den2 が天井へ近づく軸が別途残る。
- 一方このサイズ（512×320）では、load スパイクが消えたことで **全体ジョブ天井が 16,944 → ~9.2GB** に下がった（peak ② 除去・残る天井は peak ① text-encode）。**§11 の te-offload（peak ① 低減）と本 §12（peak ② の load-spike 除去）で、load/encode フェーズは 16GB 内に収まる**。残る可変軸は高トークン時の denoise stage2 のみ。

### 12.6 テスト
- mock pytest **13 passed**（app `.venv`・torch 非依存・凍結経路不変。`tests/conftest.py::_make_args` に `dit_cpu_load=None` を追加＝parse_args 既定をミラー）。

---

## 13. ★de-fork リファクタ（Stage 0–4）検証サマリ（2026-07-01・branch `refactor/engine-firstparty-cleanup`）

同梱フォークからの脱却（first-party `engine/` 化）と 43GB モノリス削除を、各段で「出力 mp4 の SHA256 がリファクタ前
ベースラインとバイト完全一致」を確認しながら実施した。**挙動不変（決定性あり・別プロセス再起動でも一致）**。

### 13.1 ベースライン（Stage 0）
- 固定 seed=12345 / prompt "a calm ocean wave rolling onto a sandy beach at sunset, cinematic" / distilled / 8 steps。
- **512×320/49f SHA256=`23844b4e…6bb7bf`**、**1280×768/49f SHA256=`4feea65f…3768da`**、**peak_vram_mb=9164**。
- worker kill→再起動を挟む2回生成でバイト一致を実測 → 以降のゲートを「output.mp4 SHA256 一致 + ffprobe(dims/fps/codec) +
  peak_vram_mb」に確定。

### 13.2 各段の結果（全段 SHA256 一致 PASS）
| Stage | commit | 内容 | 検証 |
|---|---|---|---|
| 1 | `d0d3df5` | フォーク未使用 142 ファイル削除（`services/__init__.py` 空化→`interfaces.py`削除→delete-set、`LoraEntry`→`lora_types.py`） | SHA256 一致・mock pytest 緑 |
| 2a | `1bf4163` | keep-set を `engine/{worker,api_types,lora_types,pipeline,gguf,gemma,transformer}` へ git mv＋import 全置換。worker 起動を `python -m engine.worker`（cwd=root, PYTHONPATH=root）化。grep ゲート「engine 内に非engine first-party import ゼロ」 | SHA256 一致・grep ゼロ |
| 2b | `c6ff5a4` | torch venv を `./.venv-engine` へ move。`vendor/LTX-Desktop-LOW-VRAM/` 完全削除（frontend/electron 含む）。`VENDOR_NOTICE.md`/pyproject を `engine/` へ退避。`vendor/LTX-2`（上流）温存 | SHA256 一致（削除後含め計3回）・worker スモーク ready |
| 3a | `4639008` | 残デッド枝刈り（cpu_text_encode 枝・pre-quantized FP8 枝）。`use_component_files` else 分岐削除。モノリス/QAT 存在チェックを rename test 結果に合わせ再配線＋component/GGUF 必須アサート | SHA256 一致・T2V/最小I2V 完走 |
| 4 | `35c3be5` | 未使用 `quantization` 削除。`fp8_transformer`/`cpu_offload_text_encoder` は凍結 `GET /status` 契約のため保持（worker 非伝播の注記化）。`checkpoint_path`(reference-only)/`gemma_root`(construction-required) 注記。`uv.lock` 消失を `engine/venv-engine.freeze.txt`＋`engine/engine-venv-pyproject.toml` で穴埋め | mock pytest 13 passed |

### 13.3 モノリス/QAT の rename test（Stage 3）
- **43GB モノリス `ltx-2.3-22b-distilled-1.1.safetensors`**: リネーム退避しても load 通過＋生成 SHA256 一致 →
  **GGUF+component 経路は非 open と実証** → 物理削除（~43GB 回収）。`checkpoint_path` はフィールドとして温存（worker payload
  の DistilledPipeline シグネチャ用・非 open）。
- **QAT Gemma dir `models/gemma-3-12b-it-qat/`**: wheel の `ModelLedger.build_model_builders()` が build 時に
  `tokenizer.model`+`model*.safetensors` を glob するため **削除不可（construction-required）** → 温存。重みは runtime に
  読まれない（GGUF Gemma が供給）。`services/ltx_runner.py` は `gemma_root` を存在必須（fail-fast）、`checkpoint_path` を
  非存在許容（reference-only）に再配線済み。

### 13.4 凍結 API 契約の保全
÷64 解像度（`api/models.py`）・8n+1 フレーム・T2V/最小I2V・`GET /status` の `vram_optimization`（`services/low_vram.py`
`_STATUS_KEYS`）・`metadata.json` スキーマ・limits/generation_presets はすべて**不変**。Stage 5（ドキュメント改訂）でも
これらは変更していない。

### 13.5 Stage 5（ドキュメント改訂・本エントリ含む）
`README.md` 全面改訂（2venv・engine/ アーキ・subprocess worker・GGUF+component・凍結 API・16GB 検証コマンド）／
`LTX23_Backend_Specification_v04…md` の陳腐化章に post-refactor 注記＋「÷32」→「÷64」統一（凍結契約は保全）／
`Docs/NEXT_SESSION_HANDOFF.md` 冒頭に post-refactor ステータス節を追加（歴史記録は温存）。**未 commit**（監督確認待ち）。

---

## 14. ★QAT gemma_root 22.7GB 回収（text-only Gemma 化）検証 PASS（2026-07-01・branch `refactor/qat-reclamation`）

de-fork リファクタ（§13）では「wheel が build 時に `tokenizer.model`+`model*.safetensors` を glob するため QAT dir は
construction-required で削除不可」として温存していた（§13.3）。本エントリはその **前提を根本から覆し 22.7GB を回収**した記録。
**解＝Gemma を text-only（`Gemma3ForCausalLM`・vision 無し）で構築**。full-QAT baseline と **出力 mp4 バイト完全一致**を確認済み。

> **本節は、§8/§9.7/§10.5/§10.6 で "残課題" として挙がっていた「qat-drop（24GB 削減）」および「models/ 93.83GB→フットプリント削減」を CLOSED にする**（実測 models/ 28.15GB）。旧節のそれらの記述は当時の時系列ログとして温存（追記型ログのため個別書換えせず、本注記で closed を明示）。

### 14.1 根因と解（なぜ QAT dir が要らなくなったか）
- **根因**: LTX の text encoder（`GemmaTextEncoder.precompute`）は `language_model` の hidden_states しか使わないのに、wheel は
  **vision_tower＋multi_modal_projector 込みのフルのマルチモーダル `Gemma3ForConditionalGeneration`** を構築していた。vision は
  最初から死蔵重みで、QAT shard#1（vision/mm_projector 439テンソル）がそれを供給し「construction-required」を作っていた。
- **解**: 新規 `engine/gemma/text_encoder_configurator.py` で **text-only `Gemma3ForCausalLM` を構築**（我々の seam で wheel の
  `model_class_configurator` を差し替え・**wheel フォーク不要**）。vision 構造そのものが消え、shard を glob/`safe_open` する
  必要も無くなった。gemma_root は ~40MB の **tokenizer-only dir**（`models/gemma-3-12b-it-tokenizer/`）に差し替え、QAT dir は
  **物理削除**。**models/ 50.9GB→28.15GB**（実測 30,227,833,513 bytes・GGUF 先行事例並み）。
- 事前調査＝`Docs/QAT_RECLAMATION_RESEARCH.md`（冒頭 ✅RESOLVED バナー）。当初案（案B=loader パッチ／案A=vision 抽出）は
  「マルチモーダルを保ったまま重みロードを避ける」前提だったが、実装時に上記のより深い根因が判明し不要化した。

### 14.2 途中の学び（device 周り・非自明）
1. **素朴な shardless 化は crash**: vision を meta 残置すると `self.model.device`（=最初の param の device, transformers
   `get_parameter_device`）が meta → `precompute` が input_ids を meta 上に生成 → crash。
2. **text-only では別の device 衝突**: CPU-offload した embed（text-only の最初の param）で `.device` が cpu → `precompute` が
   attention_mask を cpu 生成 → cuda の hidden_states と `feature_extractor.py:78`（`torch.where`）で衝突。
3. **修正＝`.device` を compute device に override**: `_ComputeDeviceGemma3ForCausalLM` サブクラスで `.device` を
   **final-norm（`model.norm.weight`）の device** として報告（＝`_cpu_embed_forward` の compute_dev と同一・常に GPU 常駐）。
   マルチモーダル baseline は vision(cuda) が**偶然の device アンカー**だった＝それを text-only 用に明示復元。embed の実 device
   （CPU offload）は不変ゆえ **numerics はバイト不変**。
- スコープ確認（text-only で失うもの）: IC-LoRA・i2v 画像条件・audio・enhance_t2v は全て VAE/text ベースで **vision 非依存**。
  唯一 `enhance_i2v`（入力画像を VLM に見せる画像プロンプト補強）だけ vision 使用だが `enhance_prompt` は既定 OFF・未配線・
  Phase1 外（ComfyUI-LTXVideo も同機能を Florence-2 で代替）。ComfyUI 先行事例も text-only Gemma。将来 enhance_i2v を使うなら
  vision 再導入が別途必要。

### 14.3 実機検証（全て byte-match・3経路）
- 固定 seed=12345 / prompt "a calm ocean wave rolling onto a sandy beach at sunset, cinematic" / distilled / 8 steps。
- **T2V 512×320/49f SHA256=`23844b4eebd107ccba8c5534eb65bab86575cca0b9050cb6c7e680a4506bb7bf`** ＝ §13 の full-QAT baseline
  `23844b4e…6bb7bf` と**完全一致**。
- **最小I2V SHA256=`a511eda431cf0d0942cee97fa130f45e55fc3236833cbf9ea743ea7f4715c217`** ＝ full-QAT baseline と**完全一致**。
- **3経路**: ① QAT dir 在（tokenizer-only 差替前）の T2V ② 同 I2V ③ QAT dir 不在（tokenizer-only dir へ rename）の T2V＝
  rename-test。いずれも上記 SHA と一致 → **重みバイトは元から非寄与＝vision 未使用が実証**され、QAT dir 物理削除を確定。
- **peak_vram_mb=8440**（§13 baseline 9164 より微減・退行なし）。mock pytest **13 passed**。

### 14.4 温存した意図的 no-op（dead-code 候補）→ ✅ §16 で除去済（2026-07-02）
- `engine/gemma/gguf_quant_service.py:201` `_SkipGemmaLMSDOps` … text-only 化で shard が消え no-op だが belt-and-suspenders で温存していた。
- 同 `:234` `_read_target_vocab_from_header(path)` の `path` 引数 … 未使用だが signature 互換のため温存していた。
- **↳ これらは §16 の dead-code 整理で byte-match ゲート付きで除去済み。**

### 14.5 commit
- `93696b4` feat(engine): text-only Gemma text encoder（実装＋回収）→ `694ca54` docs+chore（コメント polish＋QAT 調査 docs）→
  main へ `--no-ff` マージ `826e76f`。branch `refactor/qat-reclamation`。**config.yaml / config.py / services/ltx_runner.py の
  gemma_root 注記も併せて更新済**（`gemma_root: ./models/gemma-3-12b-it-tokenizer`）。

---

## 15. ★keep=1 常駐モード新設 ＝ 調査完了につき CLOSE（2026-07-02・コード読解＋Web リサーチ）

> ※この「keep=1 は non-viable」という結論は §47 のバグ修正と §48 の製品化で覆った。現在の正本は §48 である。

**結論: 当初構想の keep=1（＝GGUF モデルをジョブ間 GPU 常駐させ「毎ジョブ再ビルドによる gen 時間漸増」を消す）は、16GB では原理的に non-viable。しかも狙った利得は既存経路（`--dit-cpu-load` / `--te-offload` ＋ OS の RAM/mmap キャッシュ）で概ね捕捉済み。よって本タスクは「新規実装せず・調査結論を記録して close」とする。** 現行の既定 `LTX_KEEP_RESIDENT=0`（keep=0）は不変。将来の必要が生じたら §15.4 の道筋（GPU 常駐ではなく層ストリーミング）で再着手する。

> 本節は HANDOFF「次の一手メニュー」「Phase 1 やることリスト」の keep_resident 恒久化項目と、`NEXT_SESSION_WORKORDER.md` タスク③ をクローズする。

### 15.1 コード読解で確定した機構（read-only・確信度高）
- keep=1 経路: `engine/pipeline/fast_video_pipeline.py:170-173` が `StateDictRegistry`（**凍結 wheel `ltx_core`**）を有効化 → `ModelLedger._target_device()`（**凍結 wheel `ltx_pipelines/utils/model_ledger.py`**）が CPU を返す → `engine/gemma/gguf_quant_service.py:1280` が CPU ビルド → `:1305-1313` で `model._apply(lambda t: t.to(cuda))` により GPU へ **out-of-place** コピー。
- `GGMLQuantizedTensor.to()`（`engine/gguf/quant_service.py:471-487・first-party`）は subclass メタ（`_ggml_type`/`_float_shape`）を保つため意図的に out-of-place（reconstruct）。
- **移動の駆動部（`_target_device` / `StateDictRegistry`）は凍結 wheel 側**＝我々の seam から単純に in-place へ差し替えられない。
- te-offload ON 時、移動 lambda は **CPU/meta テンソルをスキップ**（`:1304-1309`, `t if t.device.type in ("meta","cpu") else t.to(...)`）＝全層バルク移動を部分的に回避する分岐が既にある。

### 15.2 Web リサーチによる仮説検証（一次情報つき）
- **H1 CONFIRMED**: 低VRAM コミュニティ（ComfyUI native / ComfyUI-GGUF / kijai wrappers）で「モデルを GPU 常駐」は **HIGH_VRAM＝24GB+ 専用**。8–16GB は「**CPU RAM を正本キャッシュ→毎ジョブ/毎レイヤー GPU へストリーミング**」が標準（ComfyUI NORMAL/LOW_VRAM）。我々の keep=1 は「CPU 正本」という常駐位置は既に正しく、欠陥は **バルク `.to(cuda)` だけ**。
  - city96/ComfyUI-GGUF は `GGMLTensor`（＝我々の `GGMLQuantizedTensor` と同型）を CPU/mmap 常駐させ `patch_weight_to_device` / `forward_ggml_cast_weights` で**層単位 JIT 移動**、バルク移動は一切しない。推論後は GPU 側を解放し CPU 正本のみ残す。
  - ComfyUI "Dynamic VRAM"（2025）は**まさにこの double-residency spike の解消**が目的で、解は「virtual allocation＋層単位 fault」＝ストリーミングであって GPU 常駐ではない。LTX-2 native でも sampler 間でテンソル未解放だと "doubling memory" で OOM する事例（ComfyUI issue #11726）＝我々の観測と同型。
- **H2 一部REFUTED＋一部CONFIRMED**: `nn.Module._apply` は param を**1個ずつ**移動し（`param.data = fn(param)` or `torch.utils.swap_tensors`）、瞬間オーバーヘッドは**約1パラメータ分**でモデル全体の 2× ではない → §10.2 の「out-of-place 移動が二重在を生む」という因果説明は**不正確**。真の spike 源は「**CPU 正本を保持したまま GPU にもフルコピーを持つ定常状態の加算メモリ**」（CUDA context だけで 1–2GB、HF accelerate も同旨）。CPU↔CUDA 間に真の zero-copy 移動は存在しない（`swap_tensors` も dest 確保は避けられない）。
- **H3/H4 CONFIRMED**: {CPU キャッシュ保持・単一コピー・subclass メタ保持} は**論理的に同時成立不可**。「CPU キャッシュ保持」が「単一コピー」を禁ずる。ピークを下げる唯一のレバーは「**GPU へ移す集合を小さくし残りを層ストリーミング**」＝我々の `--dit-cpu-load` / per-layer offload が既に実装している方式。ComfyUI-GGUF も同じ理由でフル常駐設計を採らない。

### 15.3 close 判断の根拠（利得が既に捕捉済み）
- 消したい「gen 時間漸増」は実測 720p T2V×3 で 168→209s（+24%）・I2V×4@384 で ~115→134s（+~20%）＝**実在するが bounded・crash せず**（§10.3/§10.7）。
- リサーチにより、この漸増の**大半（disk→RAM の cold load ~20–30s）は OS の RAM/mmap キャッシュが既に吸収**。GPU 常駐が追加で節約するのは RAM→GPU コピー分のみで、その代償が double-alloc crash＝**16GB では割に合わない**（H1/H2）。
- 残る漸増主因はアロケータ断片化（§10 で特定・§7.3 で `max_split`/`gc_threshold` は Windows 死枝と実証）＋毎ジョブ dequant。これらは GPU 常駐では解けない別軸。
- ∴ **当初構想の keep=1 は「16GB で誤ったターゲット」**。単一ユーザー逐次運用（クリップ連結含む）では現行 keep=0 で十分（§10.7 で I2V×4＋音声 連続 PASS 済）。

### 15.4 将来やるなら（Phase 3・GPU 常駐ではなく層ストリーミング）

**（2026-07-27追記）以後の将来項目の管理は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md)へ一本化した。** 本節のopt-in層ストリーミングは同書**§4-3**の③にあたる。本節の調査結論・実測記録そのものは不変。
もし長尺連結で漸増が実害化したら、**opt-in の「CPU 正本温存＋GPU は限定サブセット＋残りを層ストリーミング」**（＝ComfyUI-GGUF / HF accelerate 方式）で再着手する。これは既存の `--dit-cpu-load` / te-offload 経路の延長で、フル GPU 常駐（crash 源）を避ける唯一の 16GB fit 経路。断片化緩和（`expandable_segments` 等）も別レバーとして併検討。**いずれも計測前提**（`torch.cuda.max_memory_allocated` ＋ WDDM Dedicated/Shared ＋ committed bytes）。

### 15.5 一次情報
- コード: `engine/pipeline/fast_video_pipeline.py:170-173` / `engine/gemma/gguf_quant_service.py:1280,1304-1313` / `engine/gguf/quant_service.py:471-487`。
- Web（主要）: DeepWiki ComfyUI-GGUF 3.1/3.3・ComfyUI model loading 2.3／blog.comfy.org "Dynamic VRAM"／kijai WanVideoWrapper block-swap（DeepWiki 6.2）／ComfyUI issues #11726・#12330／PyTorch docs `nn.Module.to`/`_apply`・`torch.utils.swap_tensors`／HF accelerate big-model-inference。
- 関連: §10.2（keep=1 720p native crash の初出観測・te-offload 導入前）・§10.3/§10.7（keep=0 連続 PASS）・§11（te-offload）・§12（dit-cpu-load）・memory `[[720p-16gb-verified]]`。

---

## 16. ★dead-code 整理（text-only 化・de-fork の残 no-op 除去）検証 PASS（2026-07-02・branch `chore/phase1-residual-cleanup`）

**§14.4 で「belt-and-suspenders / signature 互換で温存」としていた no-op と、de-fork で機能が消えた署名パラメータを、全て byte-match ゲート付きで除去した。** T2V/最小I2V とも §14.3 baseline と**バイト完全一致**・pytest 緑・退行なし。

### 16.1 除去内容（4 コミット・各コミット後に T2V byte-match ゲート）
| commit | 対象 | 内容 |
|---|---|---|
| `be15887` | #1+#3 | `engine/gemma/gguf_quant_service.py`: `_SkipGemmaLMSDOps` クラス全削除＋その唯一の instantiation を素の `sd_ops` に戻す／defensive strip ループ削除（`n_stripped`/`n_base`・ログ行も整理）／orphan `_ORIG_GEMMA_LM_PREFIX` 定数も除去。#1 と #3 は依存ペアのため 1 コミット。 |
| `473ac85` | #2 | 同ファイル: `_read_target_vocab_from_header` の未使用 `path` 引数を def・唯一の呼び出し元(`:399`)・docstring から除去。 |
| `fa5dd83` | #4 | `engine/pipeline/fast_video_pipeline.py`: no-op 署名パラメータ `attention_tile_size`/`loras` を `create()`/`__init__` 署名・create() forwarding・`self._attention_tile_size` 代入から除去（`DistilledPipeline(..., loras=[])` の定数は保持）。NOTE docstring 更新＋未使用の `LoraEntry` import 除去。**呼び出し元ゼロを事前確認**（唯一の caller `engine/worker.py:131` は明示指定せず）。 |
| `bc5b3c1` | #5 | `engine/gemma/layer_offload_service.py`: text-only `Gemma3ForCausalLM` を反映する docstring/コメント明確化（削除でなく更新）。`text_encoder_configurator.py` は multimodal 対比が意図的に正しいため不変。cosmetic・ゲート不要。 |

### 16.2 退行確認ゲート（§13.2/§14.3 と同一手法）
- baseline（seed=12345 / "a calm ocean wave…" / distilled 8 steps / 512×320 / 49f）:
  - **T2V SHA256 `23844b4e…6bb7bf`** … 各コミット後（be15887/473ac85/fa5dd83）で計測、**全て一致 PASS**。
  - **最小I2V SHA256 `a511eda4…c217`** … 最終確認で**一致 PASS**。
- **peak_vram_mb=8440**（§14.3 と同値・退行なし）。**mock pytest 16 passed**（`LTX_DISABLE_GRADIO=1 ./.venv/Scripts/python.exe -m pytest -q`）。
- 触っていない DO-NOT-TOUCH: RMSNorm `+1` 補正（city96 verbatim・必須）／`multi_modal_guider_factory_denoising_func`（video+audio 用・text-only でも使用）。

### 16.3 事前調査
read-only 調査で 5 候補（#1-#5）を検証し、#4 の呼び出し元ゼロ（worker/config/API/services 全走査）・#1/#3 の依存関係・DO-NOT-TOUCH 2 件を確定してから着手（アノマリー無し）。WORKORDER① の削除計画に準拠。

---

## 17. ★Phase 3 スライス1「凍結 API の解凍」＝多キーフレーム条件付け 実装＋客観検証 PASS（2026-07-02・branch `feature/phase3-api-unfreeze-conditioning`・**目視品質は PENDING**）

**凍結 API を解凍し、多キーフレーム／任意 frame_idx／first+last ブックエンド／per-item strength／件数上限5 を露出した。** 客観検証（回帰 byte-match＋新経路の機能/VRAM スモーク）は全 PASS。**目視品質判断（受け入れ基準の目視項目）はユーザー外出のため後日**（レシピ＝`PHASE3_KEYFRAME_VISUAL_VERIFICATION.md`）。**本ブランチは未 merge・未 push。**

### 17.1 ★重要な前提訂正（WORKORDER の誤りを実機が是正）
- WORKORDER §2/§3 は「下層は対応済み・`combined_image_conditionings` が frame_idx==0→LatentIndex / >0→KeyframeIndex に振り分け・**engine 不可触で API 表層のみ**」と想定していた。**これは誤り**だった。原因＝当該調査が**vendor ミラー（`vendor/LTX-2`＝新しい別リビジョン）の `distilled.py` を読み、実際に動く first-party `engine/`＋インストール済み wheel を読んでいなかった**。
- **実機の真実**（GPU クラッシュ＋再調査で確定）: engine は**インストール済み wheel の `ltx_pipelines.distilled.DistilledPipeline`** を呼ぶ。これは全画像を module-global `image_conditionings_by_replacing_latent` → `VideoConditionByLatentIndex(latent_idx=img.frame_idx)` で処理し、**ピクセル frame_idx を latent インデックスと誤用**（÷8 変換なし）。`combined_image_conditionings` は**インストール済み wheel に存在しない**（vendor ミラーのみ）。
- frame_idx==0 は両インデックス空間で 0 に一致するため単一 I2V では露見せず、byte-match も通っていた（バグ潜伏）。

### 17.2 スモークが捕捉した実欠陥（"成功前提" の前に客観チェックを走らせた成果）
- mock 緑＋回帰 byte-match PASS の後、**新経路の客観スモーク**（frame_idx=48／num_frames=49）が **denoise 到達前にクラッシュ**:
  `RuntimeError: expanded size (0) must match existing size (40) at latent_cond.py:40 (VideoConditionByLatentIndex.apply_to)`。
  latent 7 枚（idx 0..6）に対し latent_idx=48 が溢れ、対象スライスが size-0 に。OOM ではない論理エラー。

### 17.3 修正＝公式ハイブリッド（wheel 更新なし）＝ LTX-Desktop パリティ
- 先行事例調査（LTX-Desktop OSS＋ComfyUI-LTXVideo `LTXVAddGuide`＋LTX-2 公式）で確定: 公式機構は**ハイブリッド** `frame_idx==0→VideoConditionByLatentIndex(ハード置換)` / `>0→VideoConditionByKeyframeIndex(guide 追記・ピクセル RoPE オフセット・÷8 なし)`。LTX-Desktop は条件付け数学を持たず LTX-2 に委譲＝**「本家の実装」＝我々が持つ LTX-2 コアそのもの**。
- **インストール済み wheel に部品が全て存在**: `image_conditionings_by_replacing_latent`（置換）・`image_conditionings_by_adding_guiding_latent`（guide 追記, `VideoConditionByKeyframeIndex` を構築）・両クラス。欠けるのは 30 行のルーターのみ → **自前で再現**。
- 実装（commit `5033385`, `engine/pipeline/fast_video_pipeline.py` `_run_inference` の既存 monkeypatch ブロック内）: `_orig_replace` を捕捉し、`_hybrid_image_conditionings` を定義（images を frame_idx で分割→ idx==0 は `_orig_replace`、idx>0 は `_orig_add_guide` に委譲→ list 連結）。module-global `image_conditionings_by_replacing_latent` を try 前に rebind、finally で復元（既存の sigma/euler patch と同一パターン＝LOAD_GLOBAL）。**両 stage が自動的にハイブリッドを拾う（Stage 2 再注入維持）**。**÷8 変換は一切追加せず**（クラッシュ根因は「置換経路にピクセル idx」＝idx>0 を guide 経路へ回せば解消）。
- **設計不変条件**: idx==0 画像＋T2V は `_orig_replace` に素通し＝**byte 同一性維持**。`api/models.py` のスナップ（8 の倍数）・`services/ltx_runner.py` は不変。
- 非本番の一段経路 `engine/pipeline/common.py` `DistilledNativePipeline`（`worker.py` 参照ゼロ）は同根バグを持つが NOTE コメントのみ（本番化時にハイブリッド必須と明記）。

### 17.4 API 表層の変更（commit `1602245`・先行）
- `api/models.py`: 検証 `len>1 拒否`＋`frame_idx≠0 拒否` を撤廃 →`cap>5 拒否`＋各 frame_idx を**最寄りの 8 の倍数へスナップ＋[0, num_frames-1] クランプ**（`round(f/8)*8`）。`frame_idx: Field(0, ge=0)`。docstring/コメントを「conditioning は Phase-3 解凍・他は凍結維持」に更新。
- `services/ltx_runner.py`: 全画像を `zip(conditioning_images, cond_paths)` でパススルー（スナップ済 frame_idx＋strength・crf は従来通りドロップ）。
- `config.py`/`config.yaml`: `max_conditioning_images_phase1(=1)` → `max_conditioning_images(=5)`＋`conditioning_frame_idx_multiple(=8)`（/config で広告）。
- テスト（`tests/test_validation.py`）: cap=6 拒否／5件・3件受理（実アップロード）／frame_idx=10 受理＋純モデル snap 数学（10→8, 24→24, 100→48 クランプ, 0→0）。**mock 19 passed**。

### 17.5 客観検証結果（全 PASS・real backend `ltx-distilled`）
- **回帰 byte-match**（不変経路が壊れていないか＝ハイブリッド idx==0 分岐の忠実性）:
  - T2V `23844b4e…6bb7bf` 一致 ✅／最小 I2V `a511eda4…c217` 一致 ✅（peak_vram 8440・§14.3/§16 と同値）。**回帰なし**。
- **新経路スモーク**（`outputs/phase3_multikey_smoke/run_smoke.py`）:
  - bookend（first@0 s0.8＋last@48 s0.7）: 完走 ✅・peak_vram **8440**・99.8s・512×320/49f。
  - multikey3（@0/24/48）: 完走 ✅・peak_vram **8440**・96.6s・512×320/49f。
  - 以前のクラッシュ消失。**VRAM は単一画像と同一（多キーフレームのデルタ 0）・16GB に ~48% 余裕・溢れ/OOM なし**。

### 17.6 未了・要フォロー（次セッション/ユーザー）
- **【目視・ユーザー】受け入れ基準の目視項目**: ①末尾 frame_idx が実際に末尾を固定するか ②first+last が両端尊重の中間補間か（strength 調整込み）③多キーフレームが各指定位置を反映するか。レシピ＝`PHASE3_KEYFRAME_VISUAL_VERIFICATION.md`。
- **【パリティ改良】frame_idx グリッド `8n+1`**: ✅**実施済み（§17.8・commit `d7a56b1`）**＝公式 ComfyUI/LTX-2 格子へ整合。
- **【merge 判断・ユーザー】**: 目視 OK 後に `feature/phase3-api-unfreeze-conditioning`（commit `1602245`→`5033385`）を main へ。
- **num_pixel_frames／reference-video 条件付けは今回スコープ外**（将来）。

### 17.7 監督ノート（プロセス学習）
- 教訓: **go/no-go 調査に vendor ミラーではなく engine 実経路を読ませるべきだった**。前提の裏取りは「実際に import/実行されるコード」で行う。スモークを "成功前提" の前に走らせたことで、目視不可の欠陥（クラッシュ）を早期捕捉できた（[[dont-overanchor-on-context]]／[[delegation-early-stop-protocol]] の実践）。

### 17.8 ★frame_idx グリッドを公式 `8n+1` へ整合（2026-07-02・commit `d7a56b1`・ユーザー指示「公式寄りで NG リスクをさらに下げる」）
- **動機**: engine の guide 経路（`VideoConditionByKeyframeIndex`）は `frame_idx` を**生のピクセル RoPE オフセット**（`positions[:,0] += frame_idx`・÷8 なし・`keyframe_cond.py:43`）として使う。ゆえにキーフレームは latent フレーム開始ピクセル＝`8n+1` 格子に載せるのが公式（ComfyUI `LTXVAddGuide.get_latent_index`＝`(f-1)//8*8+1`）。旧「8 の倍数」は最大 7px off-grid ＝微妙な劣化要因。
- **変更**（`api/models.py` validator）: idx==0 は**不変**（`continue`＝先頭フレーム latent-replace・byte 安全）。idx>0 は `snapped=(f-1)//8*8+1; frame_idx=max(1, min(snapped, num_frames-8))`（num_frames=8m+1 の最終 latent 開始＝num_frames-8。49→41）。`config.py`/`config.yaml` の広告は `conditioning_frame_idx_multiple=8`＋`conditioning_keyframe_grid_offset=1` に更新。テスト（snap 数学）更新・**mock 19 passed**。**設計＝UI は自然値（48 等）を送り、サーバーが公式位置（41）へ整える**（＝当初ユーザーが望んだサーバー側スナップの完成形）。
- **GPU 再検証（real `ltx-distilled`）**: **回帰 byte-match 維持**（T2V `23844b4e…`／単一 I2V `a511eda4…` 一致・idx==0 無傷）。**スモーク新位置で完走**＝bookend 0/**41**（48→41）・multikey3 0/**17**/**41**（24→17, 48→41）・両者 512×320/49f・**peak_vram 8440**（単一画像と同値・16GB に ~51% 余裕・溢れ/OOM なし）。
- **注記**: マシン上に ComfyUI `nodes_lt.py` 実ソースが無く式のバイトレベル確認は未（wheel の生ピクセル semantics ＋先の Web 調査の `get_latent_index` 記述と整合的なので採用）。もし目視で旧グリッドの方が良ければ当該 3 行を戻すだけで比較可。
- commit 列（branch `feature/phase3-api-unfreeze-conditioning`・未 merge）: `1602245`(API 表層・当初 8 の倍数)→`5033385`(engine ハイブリッド)→`d7a56b1`(**8n+1 整合＝最新**)。

### 17.9 ★ユーザー目視結果（2026-07-03・実画像版 visual_bookend / visual_multikey）
- **bookend＝PASS**: 20〜36フレームで浜辺→町並みへ綺麗なクロスフェード（明白な境界線なしとユーザー確認）。
- **multikey＝観察2特性は機構どおりと確認・受容判断は保留**: ユーザー観察＝16-21fで海→室内女性のクロスフェード（＝中間guideのソフト誘導によるブレンド・§17.8のf17）／38→39fで連続性なく町歩き女性へ切替（＝最終latentスロット41のguideに近傍連続性制約が無く、最終スロット境界で急遷移）。**どちらも実装機構から予測される文書化済み挙動でありバグではない**（コード読解での機構確認＝guide は追加クリーントークンのみ・補間制約なし）。
- **受容判断が保留になった理由（重要なプロセス学習）**: 監督が「f17ゴースト・静止保持→急遷移の補間特性の受容可否」という**内部略語のまま判断を求め、機能自体の説明をしなかった**ため、ユーザーから「これが何をする機能なのか知らない。知らないものは受容も拒絶もできない」と正当な指摘。→ **平易な機能解説 [`FEATURE_GUIDE_KEYFRAMES_AND_ICLORA.md`](FEATURE_GUIDE_KEYFRAMES_AND_ICLORA.md) を作成して提示し、そのうえで再度判断を仰ぐ**フローに変更。以後、人間向け説明は略語・造語禁止＋機能説明が先。
- **検証運用の是正（ユーザー総評）**: 512×320級は顔溶け・背景ぼやけで品質判断不能→**目視検証は720p級＋映画トレイラー風プロンプト**で行う（「CM風」はネット上の低品質TVCMに引きずられる可能性があるため廃止）。同日中に高解像度再生成（multikey/IC-LoRA/chain）を実施。

### 17.10 ★高解像度multikey再実行（1280×768・job `52e99266`）＝設定検証と挙動差の考察（2026-07-03）
- **ユーザー観察（1始まりのフレーム番号）**: 海は1-2fのみ→3fで不連続にビデオ会議風の女性へ（3fは右下からのグラデーションでほぼ真っ白→10fにかけて白が抜ける）→11-29f室内の女性→30-41fホワイトクロスフェード→42fから町を歩く女性。**旧runで観察された「最終キーフレーム直前のハードカット」は見られない**。ユーザーから「生成設定が間違っていた可能性」の指摘→検証実施。
- **設定検証＝正しい**: metadata.json（一次情報）で conditioning 3枚＝frame_idx 0(0.9)/17(0.8)/41(0.7)・49f・seed12345 を確認。アップロード画像3枚を直接目視し、frame0=海・frame17=室内の女性（ウェブカメラ風＝ユーザーの「ビデオ会議風」と一致）・frame41=町の女性、と対応も正しい（ハッシュ差はアップロード経路のリサイズ/再エンコードによるもの・内容は同一）。
- **変数はプロンプト**: 旧512×320 run＝"a calm ocean wave rolling onto a sandy beach at sunset, cinematic"（＝**先頭キーフレームと同内容**）。本run＝トレイラー風「町を歩く女性」（＝**最終キーフレームと同内容**）。
- **考察＝機構と整合**: キーフレーム3点は全て所定位置で履行されている（1始まりで1-2f=海・11-29fが18f[=0始まり17]をカバー・42f[=0始まり41]=町）。ガイド機構が保証するのは「各キーフレーム位置への引き寄せ」のみで、**間の遷移の時刻・スタイルは自由領域＝プロンプトが支配的レバー**。旧runはプロンプトが海を支えたため海が~16fまで持続し、最終キーフレームがプロンプトと衝突して「保持→終端で急切替」になった。本runはプロンプトが町を支持するため海は2fで放棄され（ハード固定はframe0の1フレームのみ）、遷移はホワイトフェード（トレイラー的な演出スタイル＝プロンプトの"movie trailer"起因と推定・INFERRED）で終端キーフレームに滑らかに合流した。**＝「終端直前ハードカット」は機構の定数ではなく、旧runのプロンプト×キーフレーム緊張関係の産物**。監督の事前フレーミング（38f前後で急遷移が再現される想定）はこの点で不正確だった。
- **高解像度チェーン（job `634f2da5`）の実在検証**: metadata＝`kind: chain`・num_clips 2（73f+73f・2部構成プロンプト）・overlap 3/strength 0.5・masked_av_latent_concat・seam junction=72（0始まり）・総129f・264s・peak 9578MB。**本物の2セグメント連結であり、ユーザーは継ぎ目を発見できなかった**（=720p級＋トレイラー風での連結品質の実証）。
- **✅multikey受容判断＝受容（ユーザー 2026-07-03）**: FEATURE_GUIDE 提示＋本節の「遷移スタイルはプロンプトで操縦できる」分析を踏まえ、**「途中のキーフレームは磁石として引き寄せる（間の遷移は自由領域・プロンプト支配）」という現挙動を仕様として受容**。ユーザーコメント「プロンプトの練りがいがありそうだ」。なめらかな中間補間（Gap Fill前倒し）は優先課題化しない。**＝Phase 3 スライス1の目視ゲート完全クローズ**。

---

## 18. ★Phase 3 スライス2「クリップ連結」＝配管 客観 PASS だが**映像連続性は FAIL**（2026-07-03・branch `feature/phase3-clip-concat`・**機能未達**）

> **この節が本スライスの正本記録。** 実装/設計の詳細は [`PHASE3_CLIP_CONCAT_STATUS.md`](PHASE3_CLIP_CONCAT_STATUS.md)（現状正本）／[`PHASE3_CLIP_CONCAT_DESIGN.md`](PHASE3_CLIP_CONCAT_DESIGN.md)。

### 18.1 やったこと（配管）
研究（native extend は ComfyUI ノード層・wheel 非搭載）→スコープ厚版確定→スパイク（stage1 latent seed+mask 凍結）→engine 配線（`9fb7111`）→chain エンドポイント+オーケストレーション+concat（`5e73e47`）。branch `feature/phase3-clip-concat`・未 merge・未 push。

### 18.2 客観 PASS（＝配管が動く）
- 回帰 byte-match: T2V `23844b4e…`／I2V `a511eda4…` 不変（extend keys 不在時 payload byte 一致）。
- スパイク: overlap 凍結 max_abs_diff=0.0・新フレーム生成あり・arm/disarm リーク無し（normal→extend→normal 一致）。
- mock pytest 31（19+chain 12）・VRAM 9526–9990MB<16GB・4clip で carry が clip0→1→2→3 伝播。

### 18.3 ★目視 FAIL（＝機能の本体が未達・ユーザー 2026-07-03）
- 2clip は frame24→25、4clip は 49→50/89→90/129→130 で**hard cut・音声も断絶**。クリップ内は滑らかだが**境界で完全不連続**（各クリップが「同じ被写体の独立シーン」）。
- 根本原因（read-only 調査2本＋監督訂正）: ①**二段ミスマッチ**（carry=stage1 低解像度 latent／表示=stage2 refine後・refine は各clip独立）②**carry 弱すぎ短すぎ**（K=2/soft0.5・ComfyUI は overlap~24・単段）③**overlap を concat で trim 破棄**④**音声は各clip独立生成**⑤**AdaIN/CF/stage2 凍結が未実装**。→ 現アーキでは連続性は原理的に出ない。
- ※初回 agent 分析の「stage2 は fresh noise で stage1 を無視」は**誤り**。`distilled.py:155-185` で stage2 は `initial_video_latent=upscale(stage1)` を refine（監督訂正）。

### 18.4 ★監督ノート（検証手法の欠陥＝最重要教訓）
- **テンソル一致・byte-match・VRAM は "配管が動く/壊れない" の検証であって、"映像が連続して見えるか" を全く検証していなかった。** スパイクの "GO"・各フェーズの "PASS" は配管の PASS に過ぎず、監督は**デコード後の境界フレームを一度も目視していなかった**。
- **是正（次セッションの前提）**: 実装より先に「境界フレームを目視/画素差分する連続性検証ハーネス」を用意する（方針 D）。生成系タスクでは客観 PASS ≠ 目視 OK。§17.7 の「実経路で裏取り」に加え、**"最終成果物（映像/音声）そのものを見る"** をゲートに含める。

### 18.5 次アクション
詳しい症状分析＋修正計画立案は次セッション（ユーザー指示）。方針候補＝B(素朴 I2V 連結)／C(長い単一クリップ)／A(本格 latent-extend 修復)／D(連続性検証先行)＝[`PHASE3_CLIP_CONCAT_STATUS.md`](PHASE3_CLIP_CONCAT_STATUS.md)。

---

## 19. ★Phase 3 スライス2「クリップ連結」再実装＝masked AV-latent concatenation で実機PASS（2026-07-03・同branch・§18の直接続き）

> 本節は §18 の「次アクション」を受けた同日中の再実装＋検証の記録。**正本は [`PHASE3_CLIP_CONCAT_STATUS.md`](PHASE3_CLIP_CONCAT_STATUS.md)**（現状サマリ）。本節はゲートごとの詳細数値。

### 19.1 WP1＝境界連続性検証ハーネスの構築＋既知bad アーティファクトへの較正（§18.4の是正の実行）

- 新規: `services/video_io.py` に `frame_count`（ffprobe `-count_frames`）／`extract_frame_at`（ffmpeg `select` フィルタ・フレーム番号ベース）／`extract_last_frame`。新規テスト `tests/test_video_io.py`。
- 新規: `outputs/phase3_clip_concat_spike/verify_boundaries.py`＝境界フレームのMAD（mean absolute diff）比較（video）＋30ms窓RMSジャンプ比較（audio）。しきい値＝**video: 同クリップ内ベースラインMADの3倍**／**audio: 同ベースラインRMSジャンプの6倍**。
- **較正（実装より先に実施・方針D）**: §18 で目視確認済みの旧 hard-cut アーティファクト（`outputs/b3c8bc8f-…`2clip・`outputs/e9e1363e-…`4clip）を本ハーネスで再解析。
  - 全ユーザー確認済み hard cut を検出: video比 **9–11倍**、audio比 **8.8–17.5倍**（いずれも閾値を大きく超過）。
  - intra-clip（境界でない）サンプル **11件で偽陽性 0**。
  - 旧記録の junction 座標に off-by-one を発見・修正: 4clipの真の0-based junctionは **48/88/128**（旧記録は49/89/129）。
- **結論**: ハーネスは「ユーザーの目視判断」と一致する形で hard cut を検出できることを既知データで確認済み＝以降のGPU spikeの"continuous"判定の裏付けとして使用可能と判断。

### 19.2 WP2＝read-only wheel 深掘り（root cause 再確認・修正方針の裏付け）

- 旧 hard cut の本質: **per-clip VAE decode が causal VAE の「先頭 latent フレーム=1pixel」リセットを境界ごとに再トリガーする**＋**音声が各clip完全独立生成**。
- position はデノイズ呼び出しごとに再構築される一方、**VAE decodeはposition-freeかつ既に時間タイル化済み** → 「1本の連続 stage1 AV latent を組み立てる→1回のstage2→1回のdecode」にすれば構造的にシームレスになるという設計判断の裏付けを得た。
- 音声 latentも同じ mask機構（mask respected + 毎ステップ再ピン）を持つ。audio rate=25 latent/s。
- **両モダリティともRoPEに20秒天井**（audio positionは秒単位・`max_pos=[20]`）→ stage2は20秒超チェーンでは時間タイル化必須（位置を各タイルで0再スタートさせることで天井内に収める）。
- wheel 非改変で完結（engineがwheel関数を直接呼ぶのみ、monkeypatch不要）。

### 19.3 先行事例WEB調査（github Lightricks/ComfyUI-LTXVideo コード確認済み）

- 公式 extend/looping サンプラーは **video-only**（AV latentを渡すと明示的に raise）。
- 公式の音声連続性機構は `LTXVSetAudioVideoMaskByTime`（モダリティ別 preserve mask＋線形ランプ）＋標準サンプラー。継ぎ目は `LinearOverlapLatentTransition` クロスフェード。
- last-frame I2V チェーン（方針B相当）は構造上 audio を運べない＝**音声連続性を必須としたユーザー要件のため不採用**と判断。

### 19.4 WP3＝GPU spike（`outputs/phase3_clip_concat_spike/` 配下）

| spike | 設定 | 客観結果 | 目視/試聴 |
|---|---|---|---|
| **S1**（`s1_chain_spike.py`） | 2-seg・K_v=3・overlap strength=0.5 | 全 junction continuous・VRAM 8.85GB | ユーザー目視 **PASS** |
| **S2**（`s2_tiled_spike.py`） | 4-seg・529px/22.04s・stage2タイル22latent/overlap4（hard-freeze後blend）・音声も同一タイルで結合 | 30プローブ全 continuous | ユーザー：タイル継ぎ目そのものに破綻なし。継ぎ目**直後**の subtle drift 2件を発見（171→172フレーム付近の雲、470→471フレーム付近の波）＝チューニング backlog |
| **S3**（`s3_speech_spike.py`） | 発話プロンプト | 音声メトリクス・スペクトログラムとも continuous | 「口が境界で約0.25秒閉じて再度開く」アーティファクトを `s3_standalone_seg0.py`（standalone比較）で分離＝チェーン由来（standalone単体には不在）。strength 0.5/0.8/1.0スイープ（`s3b_speech_sweep.py`）で strength は原因でないと確認。おそらく全タイムラインstage2が2発話区間をポーズ挟みで再調停する挙動。ユーザーは視聴時に気づかず＝v1として受容、backlog記録 |

**実装上の load-bearing な発見**:
- `@torch.inference_mode` 必須（無いとautograd蓄積でOOM）。
- 音声overlap長は捕捉したlatentの形状から shape-consistent に逆算する必要がある（`round(K*200/fps)` のような固定式では不可）。
- 全セグメント・全タイルを通して **transformerインスタンスは1つ**を使い回す。

### 19.5 WP4＝本番配線（commit `aebcd08`／`c0ed582`／`d359e4a`／`622dd81`）

- 新規 `chain_math.py`（リポジトリ直下・app/engine両venvでimport可能・torch非依存の純Pythonジオメトリ単一情報源）。
- 新規 `engine/pipeline/chain_pipeline.py`（`run_chain`＝S1+S2 spikeの本番移植）。
- `engine/worker.py`／`engine/api_types.py`：新 `generate_chain` op。
- `services/ltx_runner.py`（real+mock）／`services/pipeline_manager.py`（`run_chain_job`＝単一呼び出しへ再構成）。
- `api/models.py`：`overlap_frames` 既定=K_v=3、`MAX_CHAIN_TOTAL_PIXEL_FRAMES = 8×481`。
- 旧 `_EXTEND` monkeypatch機構（`9fb7111`由来）は**残置・未使用**（削除はユーザー判断で保留）。
- キャンセルはジョブ境界のみ（チェーン全体が単一の atomic worker op＝ユーザー承認済みの逸脱）。

**回帰ゲート**: byte-match T2V `23844b4e…`／I2V `a511eda4…` 不変・pytest 41 green。

**実機（real backend `ltx-distilled`）**:
- **Chain A**（2×73f・512×320）: 147s・peak_vram 8475MB・全junction continuous（`outputs/a1459043-1177-48ec-9689-a12dbb540dd5/`）。
- **Chain B**（4×145f・512×320・22.04s）: 302s・peak_vram 9564MB・映像junction 7件すべてcontinuous（`outputs/0e20e9aa-4ba2-4aa6-89bf-7e62fa74dff6/`）。
- **Chain B の音声フラグ調査（J=456）**: `boundary_metrics.json` junction 456＝audio ratio **12.63**（閾値6超過・`verdict: DISCONTINUOUS`）、同junctionの video ratio=**0.84**（閾値3内・continuous）。境界時刻 t≈19.04s。同チェーンの**非境界地点**で25–46倍のジャンプが観測される一方、この境界は12.6倍（非境界より低い）。同一ジオメトリはS2 spikeでpassしている。境界は無音→発話のonsetに一致＝**ハーネスのspeech-onset偽陽性の疑いと暫定判断**（映像は同境界でcontinuous）。**確定判断はユーザーの実試聴PENDING**。

### 19.6 検証手法の是正（実践結果）

§18.4の教訓（客観PASS≠映像連続性）を本セッション全体で徹底: ①ハーネスを実装より先に構築し既知データで較正 ②各GPU spikeでユーザーの目視/試聴を都度求め、メトリクスのみで進めなかった ③本番配線後もChain A/Bの**最終ユーザー目視/試聴はPENDINGのまま明記**（配管PASSで完了扱いにしない）。

### 19.7 未了・次セッション

- **【最優先・ユーザー】** Chain A/B 出力（パス上記）の最終目視/試聴。特にChain Bの音声 t≈19.0s 付近（J=456偽陽性疑いの確定）。
- チューニングbacklog（タイル継ぎ目後drift・発話ポーズ・harness speech-onset偽陽性）はv1受容済みだが将来の磨き候補。
- 旧 `_EXTEND` monkeypatch 機構の削除判断（ユーザー保留）。
- スライス1キーフレームの目視は別件で引き続きPENDING（`PHASE3_KEYFRAME_VISUAL_VERIFICATION.md`）。

---

## 20. ★IC-LoRA Phase A スパイク＝bf16パス忠実dequant＋engine側fuse＋参照条件付けで実機成立（2026-07-03・branch `feature/ic-lora-phase-a`）

> 本節は [`PHASE3_NEXT_WORK_SURVEY.md`](PHASE3_NEXT_WORK_SURVEY.md) を受けた同日実装の記録。**正本は [`IC_LORA_PHASE_A_STATUS.md`](IC_LORA_PHASE_A_STATUS.md)**（現状サマリ）。本節はゲートごとの詳細数値。

本機: i7-13700（**AVX2のみ・AVX512-BF16/AMX無し**）／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`。

### 20.1 実装（commit `bbcd82f`→`016f442`）

- `engine/gguf/loader_service.py`:
  - (a-0) bf16パスのdequantを`quant_service`の忠実カーネルへ委譲するよう変更。**旧・簡易実装のQ4_K/Q6Kカーネルは数値的に誤りだった**（合成Q8_0/F32での等価性検証＝ALL_OK）。
  - `GGUFStateDictLoader.ic_loras`（`(path, strength)`リスト）→`_fuse_ic_loras`: wheelの`SafetensorsStateDictLoader`＋`LTXV_LORA_COMFY_RENAMING_MAP`でLoRA safetensorsをロードし、キーごとにfp32行列積でdelta融合をin-place適用。wheelの`_fuse_delta_with_bfloat16`と数学的に同一（丸め回数が1回少ないのみ、合成データでfp64参照比bf16-ULPオーダーと検証済み）。マッチdelta0件は大声WARN。**wheelの`transformer_builder.loras`は不使用**（使うとwheelがpath無視の当ローダー経由でGGUFを再読込する既知の落とし穴）。
  - `016f442`（perf）: 当初 wheel の `apply_loras`（bf16 matmul）を呼んでいたが AVX2-only CPU で**19.3分**要した。engine側で fp32 の per-key fuse ループに置換 → **33秒**に短縮（下記20.5）。
- `engine/pipeline/fast_video_pipeline.py`: `ic_loras`/`ic_reference`をキーワード専用引数として追加（デフォルト不活性）。fail-loud条件: LoRA指定＋`per_layer_quant=True`併用、LoRAインストール失敗（safetensorsへの無言フォールバック無し）、`reference_downscale_factor<=1`。`VideoConditionByReferenceLatent`はstage1のみ既存hybrid-conditioning monkeypatch経由で追加。ステージ判別＝`cond height == full_height//2`（ステージ間で唯一異なるkwarg）。
- diff規模: `bbcd82f`＝`loader_service.py` 258行変更／`fast_video_pipeline.py` 139行追加（計246 insertions/151 deletions）。`016f442`＝`loader_service.py` 84行変更（58 insertions/26 deletions）。
- ハーネス `outputs/ic_lora_phaseA/run_spike.py`（untracked・`outputs/`はgitignore）: モード`parity`/`base`/`spike`/`toggle`。psutil RSSサンプリング・ステージ別VRAMマーク・`VideoConditionByReferenceLatent.apply_to`monkeypatchによるトークン計測。`device_supports_fp8`をFalseに固定パッチし純bf16計測。プロンプトは目視検証題材指針どおり「賑やかな町を歩く女性が『LTX 2.3!』と言う」CM風。

### 20.2 アダプタ

- HF `Lightricks/LTX-2.3-22b-IC-LoRA-Pixel-Spatial-Upscaler`（gated=auto・ユーザーがライセンス承諾済み）。
- `models/ltx-2.3-ic-lora/pixel-spatial-upscaler/` に x2/x4 とも 654,465,286 bytes で配置。
- x2メタデータ検証: `reference_downscale_factor=2`・`reference_spatial_scale_factor=2`・`model_version=2.3`。960キー＝lora_A 480＋lora_B 480、すべて`diffusion_model.`プレフィックス、BF16、rank 64＝wheelの`apply_loras`期待値＋COMFYリネームマップと厳密一致。

### 20.3 ゲート1＝回帰byte-match（本番per-layer経路・LoRA off）

**PASS**: T2V sha `23844b4e…6bb7bf`／I2V sha `a511eda4…15c217`、既存ピン止めベースラインと一致・peak VRAM 8440MB不変。pytest **41 passed**（`bbcd82f`適用後・`016f442`適用後の各時点で再検証）。

### 20.4 ゲート2＝パリティゲート（a-0検証・LoRA無し）

512×320/49f/8steps/seed12345、`per_layer_quant=True` vs `False`で比較（`result_parity.json`）。

| 指標 | per_layer_quant=True | per_layer_quant=False（bf16パス） |
|---|---|---|
| SHA256 | `9a65e4b1…` | `9a65e4b1…`（**完全一致**） |
| bytes | 235193 | 235193 |
| generate_s | 94.24 | 282.65（約3倍遅） |
| denoise最終 max_alloc | 5332.8MB | 8954.7MB（+3.6GB） |

RAM peak 55.9GB。**verdict: MATCH**。

### 20.5 ゲート3＝スパイク本番実行（x2 LoRA fuse＋参照条件付け）

条件: target 1024×640/25f、参照＝base clip 512×320/25f（`base.mp4`、sha `3a2a87a2…`、gen 103.4s）。`result_spike.json`。

- 出力: `spike.mp4`、sha `8e10aa59…`、378415 bytes。**COMPLETED**（VRAM溢れ無し）。
- gen 1435.2s（うちload+fuseがfp32修正前で約20分を占めた個体＝下記トグルの`with_lora`計測より前の実行）。

VRAMピーク（`max_alloc_MB`）:

| ステージ | 値(MB) |
|---|---|
| encode_text後 | 8440.9 |
| denoise stage1後 | 8915.0 |
| denoise stage2後 | 9102.2 |
| vae_decode_video後 | 2364.2 |

RAM peak 56.7GB（system 65.3/65.3GB＝ほぼ飽和）。

トークン計測（`VideoConditionByReferenceLatent.apply_to`monkeypatch）: 参照latent shape `[1,128,4,5,8]`、downscale_factor=2、strength=1.0。stage1 latentトークン数 640 → 参照concat後 **800**（**+25%、2倍ではない**＝参照はdownscale_factor=2で縮小されているため）。

### 20.6 ゲート4＝トグルゲート（LoRA on/offの切替コスト、keep_resident=0のため切替=フルリビルド）

`result_toggle.json`。fp32修正（`016f442`）**前後**で2回計測。

| 指標 | 修正前 | 修正後 |
|---|---|---|
| with_lora generate_s | 1455.0 | 323.2 |
| no_lora generate_s | 294.9 | 295.5 |
| 差分（≒load+fuse時間） | fuse ≈ 19.3分 | load+fuse 104s vs no_lora load 71s → **fuse ≈ 33s** |

- `no_lora`出力SHA: 両回とも `3a2a87a2…` で完全一致、かつ`base.mp4`のSHAとも一致＝**決定性＋LoRA非汚染性の証明**。
- `with_lora`出力SHA: 修正前後で異なる（`016f442`は丸め回数1回分の差＝想定どおり）。`with_lora` vs `no_lora`のSHAが異なる＝**fuseが実際に適用されている挙動的証拠**。

### 20.7 ゲート5＝タイミング異常の根本原因調査（Web検証）

- torch CPU bf16 matmulはAVX512-BF16/AMX非搭載CPUで高速カーネルを持たず、fp32変換フォールバックに落ちる（コミュニティ実測で約54倍遅化の報告・Intel執筆のPyTorchチューニングDocsが機構を裏付け・ComfyUI PR#3649も同様の傾向を裏付け）。
- コンスーマCPU事情: AMXはサーバー専用。Intel第12〜14世代コンスーマ機はAVX-512自体非搭載。AMD Zen4以降はAVX512-BF16搭載。
- コミュニティの16GB級LTX-2生成時間の通念と照合し、当機のdenoise（~3.3分 @1024×640/25f 二段）は標準〜良好な範囲と確認。RTX 5090はDiT推論で当機の約2〜2.3倍速の目安。

### 20.8 Phase B 未了・持ち越し（詳細は挙げるのみ）

- 32GB RAM級マシン向け本番機構: bf16 full-dequant fuseはスパイク専用（RAM 54-57GB）。候補＝per-layer-quant経路＋GPU forward-time LoRA適用（ComfyUI実証パターン）vs 事前fuse済みチェックポイント派生。
- wheelの`ICLoraPipeline`をoracleとした公式パリティ照合（stage1のみLoRAの公式挙動 vs 当実装の両ステージfuse＝乖離は文書化済みだがUpscaler用途では挙動的に問題無し・oracle未照合）。
- keep-resident運用との整合: in-place fuseがキャッシュ済みbaseを変異させるため、`StateDictRegistry`下でのLoRAトグルは設計要（リビルド vs デュアルキャッシュ）。
- API/UI露出（`engine/api_types.py`のIcLoraスキーマは存在するが未配線）。
- x4バリアント・他アダプタ（In-Outpainting/Deblur、`PHASE3_NEXT_WORK_SURVEY.md` §6準拠）。
- `spike.mp4` vs `base.mp4`の最終目視は**fix-later方針でユーザー承認済み・非ブロッカー**（回答到着次第、必要なら追いコミット対応）＋前セッションから持ち越しの目視4本（`NEXT_SESSION_HANDOFF.md`参照）。**残る「作業」＝mainへのマージ実行**（branch `feature/ic-lora-phase-a` 未マージ・4コミット先行）。→ **✅マージ実行済 2026-07-03（merge `a578c83`）・Phase B本実装＝§21**

## 21. ★IC-LoRA Phase B＝forward時GPU LoRA適用（per-layer-quant経路）＋API露出 実装・全ゲートPASS（2026-07-03・branch `feature/ic-lora-phase-b`）

> **正本＝[`IC_LORA_PHASE_B_STATUS.md`](IC_LORA_PHASE_B_STATUS.md)**（現状サマリ）／設計・ゲート定義＝[`IC_LORA_PHASE_B_WORKORDER.md`](IC_LORA_PHASE_B_WORKORDER.md)。本節はゲートごとの詳細数値。
> base＝main merge `a578c83`（Phase A取り込み）。commit `b805ae1`（engine機構）→`fbef799`（API露出）。

本機: i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`。

### 21.1 事前リサーチ（実装前の裏取り・仮説→確認）

- **ComfyUI-GGUF（city96）のLoRA機構をソース確認**: `GGMLLayer.get_weight()`がdequant直後の行列へfp32 delta（`strength·(alpha/rank)·B@A`）を加算・**毎forward再計算・キャッシュ無し・量子化バイト不変**（=「dequant時ウェイトパッチ」方式）。ComfyUI coreの`calculate_weight`は`intermediate_dtype=fp32`でdeltaを計算し weight dtypeへ1回キャスト＝Phase A `016f442`のfp32 fuse判断と同型。事前fuse/再量子化・deltaキャッシュはメインライン不採用（不可逆量子化誤差・トグル喪失のため）。
- **side-path方式（出力側低ランク加算）は不採用と判定**: per-layer経路はどのみち毎forwardで全重みを実体化するためVRAM利得ゼロ・G2のbyte-match検証レバー喪失・動画のトークン数域ではdelta計算の方が安い。
- **自エンジンのフック点をコード読解で確定**: `ggml_linear_forward`（`quant_service.py:625-643`・`types.MethodType`で各Linearにバインド）。block-swapはブロックを`.to()`移動するだけ（モジュール差し替え無し）→Linearに載せた非persistentバッファはswapを自動で生き残る。

### 21.2 実装（commit `b805ae1`→`fbef799`）

- **`engine/gguf/ic_lora_common.py`（新規）**: safetensorsロード＋`LTXV_LORA_COMFY_RENAMING_MAP`＋lora_A/Bペアリングの共通化（`load_ic_lora_pairs`・bf16融合経路と共用）。`attach_ic_loras`＝LoRA prefix→`nn.Linear`解決（X0Modelの`velocity_model.`ラップはLTXModelノード起点で回避）・A/Bを**非persistentバッファ**登録（state_dict非汚染・`.to()`移動対象）。shape不一致raise・0マッチ大声WARN。`detach_ic_loras`＝完全除去。
- **`quant_service.py`**: `ggml_linear_forward`にLoRA分岐。量子化weight→**毎回新規のdequantテンソルへ** `delta=matmul(B.float()*strength, A.float())`→`.to(bf16.dtype)`加算（**式・演算順序ともPhase A fuseと同一**・G2要件）。float weight→out-of-place（保存バッファ不変）。LoRA無し時は属性読み1回のみ＝既存パス完全同一。複数LoRA＝逐次加算。
- **`fast_video_pipeline.py`**: 旧「per_layer＋LoRAでraise」ガードを`ic_loras_provider`配線に置換（transformerビルド毎に現ジョブのLoRAをattach）。`generate(*, ic_loras=None, ic_reference=None)`でジョブ毎切替（明示`[]`＝detach・`None`＝create時デフォルト）。**bf16融合経路（`gguf_per_layer_quant=False`）は無傷温存**（ユーザー指示）。
- **API露出（`fbef799`）**: `POST /api/v1/upload/video`新設（画像uploadと同型）。`GenerateRequest`に追加（凍結契約へ純加算）: `loras:[{name,strength(0<s≤2)}]`（**サーバー側レジストリ名のみ**・パス形式拒否）＋`reference_video_id`（lorasと全か無か）。レジストリ＝`config.yaml` `model.ic_loras`（現登録=`pixel-spatial-upscaler-x2`のみ）。配線=pipeline_manager→ltx_runner→worker generate op（**LoRA無しでも明示`[]`送信**=G3のclean-detach保証）。metadata＝LoRAジョブのみ追加`ic_lora`ブロック。`GET /status`凍結キー不変。

### 21.3 G1＝回帰byte-match（LoRA off・本番per-layer経路）

**PASS**: 本番runner経路でT2V `23844b4e…6bb7bf`／I2V `a511eda4…15c217` 完全一致・peak_vram **8440**不変。pytest **47 passed/1 skipped**（Stage 1後）→**58 passed/1 skipped**（Stage 2後・+11本のAPIテスト）。

### 21.4 G2＝新経路 vs bf16融合のbyte照合（**基準SHA再ピン**）

spike同条件（1024×640/25f・x2 strength1.0・参照条件付け・seed12345）で：

- per-layer forward時LoRA（`spike_pl`）＝ **`735a6de97d2deb56a781c66307849924a8ac25b51f0591fbddab07a78875e272`**
- 現行コミットのbf16融合（`spike_false_current`）＝ **同一SHA（byte完全一致）** → **PASS**
- 旧アーカイブ`spike.mp4`（`8e10aa59…`）との不一致は**stale baseline**（`016f442`以前のbf16-matmul fuse生成物・§20.6の「丸め1回差」記録と整合）と根本原因特定。フレーム差分PSNR 26.2dB＝拡散カスケードによる増幅で説明済み・ロジック欠陥ではない。
- 数値証拠: **全480層でdeltaのCPU/GPU計算が0 ULP一致**（bf16キャスト後bit一致）・fused weightサンプル16層0差・LoRA無しres_parity（1024×640）両経路SHA一致（`13227dfe…`）・LoRAあり参照無し（`lora_noref`）両経路SHA一致（`575671ff…`）。
- **Phase B基準SHA＝`735a6de9…272`に再ピン**（監督判断・旧`spike.mp4`は温存）。

### 21.5 G3＝非汚染トグル（同一プロセス・per_layer=True）

**PASS**: lora1→nolora→lora2 連続実行で、nolora出力＝`base.mp4`（`3a2a87a2…`）**完全一致**・lora1＝lora2＝`735a6de9…`相互一致。create時デフォルト／generate上書き×2／bf16融合の**4経路すべてが`735a6de9…`に収束**＝決定性＋detach健全性の証明。

### 21.6 G4＝VRAM／速度（1024×640/25f・§20.5と同一計測）

| 経路（出力は全て`735a6de9`で同一） | gen時間 | denoiseピーク | 全体ピーク | load/attach |
|---|---|---|---|---|
| per-layer forward時LoRA | **124.7s** | 5679MB | **8440.9MB** | attach **0.02–0.3s** |
| bf16融合（現行） | 273.5s | 9102MB | 9102MB | fuse≈33s+load |
| LoRA無しper-layerベースライン | 103.4s | 5564MB | 8440.9MB | — |

- **Phase Aのbf16ペナルティ（+3.6GB・約3倍遅）は解消**（対bf16: 2.2倍速・全体ピーク−661MB・denoise−3.4GB）。
- **全体ピークはLoRA無しと同一（8440.9MB）**＝A/B 654MB＋一時deltaはencode天井の下に収まる。
- **rank64のdenoise時間増は実測+20〜26%**（毎forwardのB@A再計算。ワークオーダーの「<1%」は楽観的すぎた＝理論flopsでなく実効。それでも融合方式より圧倒的に安い）。run-to-run分散±20s程度あり。

### 21.7 G5＝API e2e実機スモーク（HEAD `fbef799`）

**PASS（ボーナス＝APIパス出力もbyte一致）**: 本番サーバー起動→`POST /upload/video`（base.mp4・保存はbyte同一＝再エンコード無し）→`loras=[{pixel-spatial-upscaler-x2, 1.0}]`＋`reference_video_id`でgenerate→**115.8sで完走・出力SHA＝`735a6de9…272`（ハーネス基準とbyte完全一致）**。workerログ`ic_loras=1 ic_reference=yes`・`peak_vram_mb=8440`。metadata`ic_lora`ブロック有・`GET /status`凍結8キー不変。negativeケース＝偽`reference_video_id`→**404 REFERENCE_VIDEO_NOT_FOUND**・ジョブ非生成。
- 副次的知見: 本番は`fp8_transformer:true`（Ada）で`QuantizationPolicy.fp8_cast()`が入るが、ハーネス（fp8オフ固定）とbyte一致＝**fp8-castポリシーはGGUF per-layer経路では実質no-op**であることの挙動的証明。

### 21.8 未了・Phase C候補への持ち越し（挙げるのみ）

- keep_resident=1下のLoRAトグル（機構は非汚染なので障害無し・検証のみ未実施）。
- wheel `ICLoraPipeline` oracle照合（stage1のみ適用 vs 両ステージ適用・未照合のまま）。
- x4バリアント（ファイルは配置済み・レジストリ未登録）・他アダプタ（In-Outpainting/Deblur）。
- rank64 denoise +20〜26%の最適化（必要になったら: deltaの層内キャッシュ等・現状は許容と判断）。
- loras⇔reference_video_id全か無か制約は「参照必須アダプタしか無い」前提＝参照不要アダプタ導入時にアダプタ別メタデータ駆動へ緩和。
- Gradio UI露出（Phase 1残(a)と合流）。
- ユーザー目視: Phase A持ち越し5本＋Phase B出力（`outputs/ic_lora_phaseA/phaseB/api_smoke.mp4`等）＝fix-later方針継続。

## 22. ★IC-LoRA Phase C＝制御系アダプタ（Union-Control）＋engine内前処理段（canny／DWPose）実装・客観ゲート全PASS（2026-07-04・branch `feature/ic-lora-phase-c`・**G5目視はユーザー受容待ち**）

> **正本＝[`IC_LORA_PHASE_C_STATUS.md`](IC_LORA_PHASE_C_STATUS.md)**（現状サマリ）／設計・ゲート定義・リサーチ根拠＝[`IC_LORA_PHASE_C_WORKORDER.md`](IC_LORA_PHASE_C_WORKORDER.md)。本節はゲートごとの詳細数値。
> base＝main `1a3dfec`（Phase B＋Phase C準備docs取り込み済）。commit `5a9de32`（スライス1）→`80a909c`（スライス2）→`91dc485`（÷128バリデーション）→`058e862`（スライス3）。

本機: i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`。

### 22.1 実装（commit `5a9de32`→`80a909c`→`91dc485`→`058e862`）

- **スライス1（`5a9de32`）＝レジストリ・config拡張**: `config.py` に `IcLoraEntry{path, preprocess: none|canny|dwpose}`・`ic_loras: dict[str, str | IcLoraEntry]`（文字列値=後方互換=`preprocess none`）。`lora_registry.resolve` は `(path, strength, preprocess)` を返す（辞書/文字列両受理）。`api/generate.py` に複数preprocess種の競合検出 400 `LORA_PREPROCESS_CONFLICT`・`reference_payload` に `preprocess` を追加。tests 10→19・pytest **66→69**。
- **スライス2（`80a909c`）＝engine前処理段（canny）**: 新設 `engine/preprocess/`（`FrameProcessor` Protocol・`CannyProcessor`＝フォーク `apply_canny` 忠実移植: 64pad→`cv2.Canny(gray,100,200)`→crop→3ch・cv2ドライバでFPS/フレーム数/寸法保存・`get_processor` レジストリ）。worker: `preprocess != none` で制御動画 `control_<kind>.mp4`（ジョブ出力ディレクトリ）へ変換し `ic_reference` を差し替え。**`none` 経路は完全無変更**（cv2遅延import）。metadata.json `ic_lora.loras[]` に `preprocess` 加算。E2E＝job `eb00afe2`（canny-control 512×256/49f）**126.4s**・PREPROCESS canny 129f **0.95s**・peak_vram **9267**。**発見: factor=2参照ジョブは出力÷128必須**（512×320はVAE encodeでeinops fail・512×256 PASS）。
- **÷128バリデーション（`91dc485`）**: 参照付きジョブで width/height%128≠0 → 422 `REFERENCE_RESOLUTION_INVALID`。参照無しジョブは無影響（回帰テスト付き）。
- **スライス3（`058e862`）＝DWPose前処理**: `engine/preprocess/dwpose.py`＝G0-bハーネス／フォーク `DWPosePipeline` のverbatim移植（YOLOX検出→DWPose SimCCバッチ5→OpenPose18点remap→body/hand/face骨格描画）。**遅延ロード＋`preprocess_video` 完了後に `release()` でGPU解放**（ワークオーダーの「プロセス内キャッシュ・再ロードなし」から意図的逸脱＝監督承認: 再ロード+5〜7s/ジョブと引き換えにdenoise中の355MB常駐を排除）。E2E＝job `7591f01f`（pose-control 512×256/49f）**144.5s**・PREPROCESS dwpose 129f **16.81s**（モデルロード込み）・peak_vram **8445**（Phase B帯下限）。

### 22.2 G0-b＝DWPoseスループット実測（GPUスモーク・PASS）

- 入力＝`outputs/visual_review/09_hires_chain_2seg_1280x768.mp4`（1280×768・129f→300フレームに循環・ウォームアップ20f）。
- **13.34fps end-to-end**（計算のみ14.0fps）＝**混雑ワーストケース**（~13人/フレーム・DWPoseはバッチ5で~3バッチ/フレーム）。121f≈**9.1s**・257f≈**19.3s**。
- VRAM: モデル常駐 **354.7MB**・ループピーク **482MB**(alloc)/617MB(reserved)・`del`+`empty_cache()` で **8.5MB** まで完全解放。
- 段階別ms/フレーム: decode 1.34／yolox 13.87(18%)／**dwpose 53.04(71%・律速)**／draw 4.51／encode 3.12。
- **判定**: 前処理は生成時間の~5%・VRAM競合なし＝実用。**スライス4（キャッシュ）前倒し・rtmlibフォールバック共に不要**と監督判断。

### 22.3 G1＝回帰byte-match（LoRA off・本番per-layer経路）

**PASS**: Job A（T2V基準512×320/49f/seed12345/"a calm ocean wave rolling onto a sandy beach at sunset, cinematic"・LoRA無し）SHA＝**`23844b4eebd107ccba8c5534eb65bab86575cca0b9050cb6c7e680a4506bb7bf`** 基準完全一致・peak **8440**（Phase B記録一致）・102.8s。Job B（最小I2V基準）SHA＝**`a511eda431cf0d0942cee97fa130f45e55fc3236833cbf9ea743ea7f4715c217`** 基準完全一致・peak **9525**（§10.4歴史値と同型）・118.2s。pytest **69 passed/1 skipped**。

### 22.4 G2＝非汚染トグル（pose→なし→canny→なし・同一プロセス）

**PASS**: pose（Job C: `8e914661`・148.7s・peak 9525）→なし（Job D: `49109d77`・**SHA=Job A完全一致**）→canny（Job E: `d6daa625`・137.2s・peak 9522）→なし（Job F: `c20fe645`・**SHA=Job A完全一致**）＝attach/detach・前処理の非汚染確認。

### 22.5 G3＝VRAM／速度

- 制御ジョブ peak **9525／9522 ≦ 天井9527**。D/F の peak 9543／9535 はプロセス2ジョブ目以降の既知パターン（VERIFICATION_LOG既存記録「8440/9525/9522」と同型・出力byte一致ゆえ機能退行なし）。
- PREPROCESS実測: dwpose **16.60s**（≈7.8fps・ロード込み）／canny **1.12s** @129f。

### 22.6 G4＝API e2e実機

**PASS**: metadata（`preprocess="dwpose"`・`reference_video_id` 記録）✓・GET video HTTP **200** ✓・偽video_id→**404 REFERENCE_VIDEO_NOT_FOUND** ✓・512×320+参照→**422 REFERENCE_RESOLUTION_INVALID** ✓。

### 22.7 G5＝目視 → ✅**PASS＝ユーザー受容（2026-07-04）**

- **Job P**＝pose-control 1280×768/121f/seed12345（`408fd361`・219.7s・peak **8548**・PREPROCESS dwpose 18.15s・AAC音声あり）／**Job Q**＝canny-control 同パラメータ（`8c1b5a9c`・196.0s・peak **9541**＝LoRA無しD/F(9543/9535)と同じ~9.5GB帯のジッタ・OOMなし）。
- プロンプト＝映画トレイラー風「老船長が港町を歩く」（**制御種は非言及**＝シーン記述のみ・追補R4のComfyUI公式ワークフロー流儀・セリフ入り）。
- 成果物＝`outputs/visual_review/10_〜13_`（pose出力／骨格／canny出力／エッジ・README追記済み）。
- **監督の事前目視所見（受容判断ではない）**: 参照の歩行動作・カメラ・群衆構図を維持して別キャラクター（老船長）へ置換成立。canny版は参照の街並み構造をより強く保持・pose版は背景自由度が高い（制御タイプの性質どおり）。**受容判断はユーザー**（G1〜G4=客観PASS／G5=目視ゲート＝別物）。
- **✅ユーザー受容（2026-07-04・起床後レビュー）**: `10_〜13_` を目視し「動き維持で内容置換」の成立を受容。÷128制約・DWPose release()逸脱も併せて了承。**Phase C全ゲートクローズ→mainマージ・push実施（ユーザー指示）**。

## 23. ★検証用Gradio GUIのバックエンド追いつき＝4タブ再構築・客観ゲート全PASS（2026-07-04・branch `feature/gradio-gui-catchup`・**目視ゲートは未実施＝OPEN**）

> **正本＝本節＋引き継ぎ書 `NEXT_SESSION_HANDOFF.md` 最新ブロック。** 旧GUI（`gradio_ui.py` 単一ファイル186行・Phase 1初期機能のまま）を、Phase 3／IC-LoRA Phase B/C で増えたバックエンド機能を検証できる4タブ構成へ作り替えた。GUIは凍結APIの薄いクライアントに徹する方針（AviUtl2統合の予行）。
> base＝main `8bdf90a`（Phase C マージ済）。**このブランチはまだ main へ未マージ**（push/マージはユーザー承認待ち）。

本機: i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11。

### 23.1 実装（コミット列・順序）

まず**HTMLモックアップ（アーティファクト）で仕様をユーザーと合意してから**実装に着手した（勝手に画面を盛らない方針）。ユーザーからの仕様修正3点を反映済み＝(1)既定言語は英語・日本語はSettingsタブから切替、(2)ダークモード既定、(3)Generateボタンは右カラムの操作パネルへ移動。

- **S1（`1c5b146`）＝土台**: 4タブ・上部ステータスバー・`ApiClient`・i18n `LABELS`（英語既定＋日本語）・ダークテーマ既定（`mount_gradio_app` の `js=` 経由）。
- **S2（`feea189`）＝プリセット**: `GET /config` からプリセット取得。`config.yaml` に `standard_720p`（1280×768生成→crop 1280×720／257フレーム）を追加。spill-free（VRAM溢れ回避）の警告表示。
- **S3（`4c2c356`）＝多キーフレーム**: 5スロット固定のアコーディオン式キーフレーム条件付け。
- **S4（`b4a09e1`）＝IC-LoRA参照動画**: 参照動画アコーディオン・アダプタ選択ドロップダウン（`/config` から取得）・÷128／拡張子／サイズの事前チェック・`format_api_error`（15コード）。
- **S5（`6f00ebc`）＝クリップ連結タブ**: Clip Chainタブ・8クリップスロット・overlap／総フレーム数の事前チェック（サーバー側と同じ `chain_math` を流用＝独自算術を発明しない）。
- **リファクタ（`88f7c1a`）**: `gradio_ui.py` 単一ファイル → `gradio_ui/` パッケージ（9モジュール）。**挙動変更ゼロ**（リファクタ前後で pytest 131 passed が同一であることを確認）。
- **S6（`f766f88`）＝Jobs／Settings**: Jobsタブ・Settingsタブ・実行中の言語切替（ラベルレジストリ経由で127出力を一括更新）・ポーリング設定・危険操作ゾーン（danger zone）。

### 23.2 完全性監査（Opusサブエージェント）

仕様とAPI表層全体を突き合わせる完全性監査を実施し、9件の指摘を反映。うちユーザー判断で見送った2件：

- **(a) `two_stage_hq` オプション**: 画面には出すが**無効化し「バックエンド対応待ち」の注記**を付けた。理由＝`services/ltx_runner.py:849-862` の payload が pipeline／guidance_scale をエンジンへ渡しておらず、常にdistilled経路になるため。バックエンド側の対応は将来項目として記録。
- **(b) `GET /jobs/{id}/metadata` エンドポイント**（GUIで peak_vram_mb／backend を表示する用途）: **新規エンドポイントを足さない方針**により意図的に追加せず。将来項目。

### 23.3 客観ゲート（PASS）

- **pytest**: **147 passed / 1 skipped**（skipはappのvenvにおける既存の `ltx_core` import skip＝新規ではない）。
- **統合スモーク（モックバックエンド）**: `--config` で `model.backend=mock`・port 8765 で起動。
  - `GET /ui` → **200**・ダークモードjs入り・**213** Gradioコンポーネント（タブ含む）。
  - `POST /generate` → completed → video/mp4 **8094 bytes**。
  - `POST /generate/chain`（2クリップ）→ completed → video/mp4 **17750 bytes**。

### 23.4 目視ゲート＝**未実施（OPEN）**

- **未実施**: 実サーバー起動での目視確認（4タブ全操作・言語／テーマ切替の見た目・720p生成の音声付き再生をGUI内で）。客観PASS≠目視ゲートの規律に従い、ユーザー目視は別ゲートとして残す。
- **未実施**: 実バックエンドe2e（720pプリセット・多キーフレーム・pose-controlの÷128違反／遵守・クリップ連結）。
- **未実施**: push／main マージ（ユーザー承認要）。

### 23.5 持ち越し（将来項目）

**（2026-07-27追記）以後の将来項目の管理は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md)へ一本化した。** 1点目（`two_stage_hq`の実効化）は同書**§3-2**、残る4点は同書**§4-6**（バックエンド同梱Gradio UIの残4件）にあたる。以下の記録は不変。

- バックエンドでの pipeline／guidance_scale 消費（`two_stage_hq` の実効化）。
- `GET /jobs/{id}/metadata`（新規エンドポイント不追加の方針で見送り）。
- `gr.BrowserState` による言語／テーマの永続化（固定secret＋実機検証が必要）。
- `gr.render` による動的なキーフレーム／クリップ行（現状は固定スロット）。
- Settings のデフォルト negative prompt 設定（S6では死んだUIを避けるため意図的に見送り）。

### 23.6 得られた知見（記録）

- **gradio 6 で theme／js／css の指定場所が移動**: `gr.Blocks()` から `mount_gradio_app()`／`launch()` へ。
- **`build_ui` は uvicorn がlistenする前に走る**ため、`/config` はビルド時に取得できない → `demo.load`（`demo.load` 内）で取得する。
- **`gr.I18n` はブラウザロケール依存のみ**＝実行中の言語切替はラベルレジストリ＋`gr.update` 一括更新で実現。
- **クリップ連結の総フレーム数上限はサーバーと同じ `chain_math` を流用**して事前チェック（独自算術を発明しない）。

---

## 24. ★video-to-video 継続（アップロード動画の「続き」生成）＝設計→実装→実機e2e 全客観ゲートPASS（2026-07-04・branch `feature/v2v-continuation`・**G3目視はユーザー未実施＝OPEN**）

> **正本＝本節＋設計書 [`V2V_CONTINUATION_DESIGN.md`](V2V_CONTINUATION_DESIGN.md)（ユーザー合意済み4決定含む）。一次情報＝`outputs/v2v_spike/SPIKE_REPORT.md`（S0）・`outputs/v2v_e2e/E2E_REPORT.md`（S3）・`outputs/v2v_e2e/S4_fix_verify/`（バグ修正検証）。**
> base＝main `8bdf90a`＋GUIマージ `86a3fcf`。**push／main マージはユーザー承認待ち。**

本機: i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11。

### 24.1 何を作ったか（1分）

`POST /generate/chain` に optional `source_video: {video_id, context_frames=73}` を追加（凍結APIの加算的拡張・省略時byte同一）。アップロード動画（`POST /upload/video` 再利用）の末尾 context（既定73px≈3秒）を VAE エンコードし、クリップ連結の carry+freeze 機構に凍結ヘッドとして注入して続きを生成。出力は**新規部分のみ**（context はサーバー側でフレーム正確にトリム・音声は30msフェードインのクリックガード付き）。fps 不一致は app 側 ffmpeg で自動リサンプル。音声トラックも latent 化して凍結継続（音声なし源は自由生成へフォールバック）。

### 24.2 ゲート実績（コミット列: 設計`cf99448`→S1 `e7d497c`/`2830ede`/`5482225`→S2 `33fae6d`/`73dc20f`/`44facdd`/`a89963c`→レビュー反映`092d37f`→docs`0e8b1a7`→チェーンバグ修正`e8557cb`）

- **G0 スパイク（GO）**: stage2 ヘッド凍結の A/B が決着＝**variant B（源末尾のフル解像度 tiled 再エンコードで stage2 tile0 先頭を mask=0 ハード凍結）必須**。A は継ぎ目で色調が跳ね FAIL（MAD 4.00×/輝度6.62×）、B は 1.00×。h264-crf35 の汚い源でも連続（圧縮アーティファクト混入は観測されず）。audio_encoder 初ロード 45.7MB。
- **G1 回帰**: T2V `23844b4e…6bb7bf`／I2V `a511eda4…c217` **byte 完全一致**（S1 実装後と `e8557cb` 修正後の2回証明）。等長チェーンは修正前後で SHA `f706057a…0ea1` **バイト一致**＝修正は等長経路の no-op を証明。pytest **168 passed / 1 skipped**（基準147→+15 S2＋+3 レビュー＋+3 バグ回帰）。
- **G2 新経路（実機・REST 経由＝完全消化）**: 720p（1280×768・context73＋新規144f）完走・継ぎ目連続（video 1.18×）・worker peak 10202MB（瞬間 nvidia-smi 15.9GB＝既知の Gemma encode 一過性・16GB内）。多クリップ継続（等長・不等長とも）・30fps→24fps 実リサンプル・422/404 異常系全PASS。
- **★既存バグ発掘・修正（`e8557cb`）**: クリップ長が不揃いのチェーンは V2V 以前からクラッシュする潜在バグ（stage1 carry の init を `torch.zeros_like(prev)`＝前セグメント長で確保）。過去検証が全て等長だったため潜伏。現セグメントの latent 形状から確保する修正＋回帰テスト3件。音声 carry 側の防御要否は網羅スキャン（結合900万件超）で「既存バリデータで到達不能」を確認し追加せず。
- **VRAM**: 生成ピークは既存チェーン同族（640×384で8.4–9.5GB・720pで10.2GB）。V2V 固有の追加コストは実質ゼロ（源エンコードは tiled 化で ピーク非支配・スパイクの未タイル比 -1066MB）。

### 24.3 設計の要点・制約（実装で確定）

- **context_frames 上限=145px**（8n+1・下限25）。理由＝凍結ヘッドは stage2 時間タイル（22 latent）内に収まる必要（variant B は tile0 のみ凍結・理論上限169px）。不変条件は `chain_math.py` の early-raise＋config ガードテストで二重化（レビュー Finding 1 反映）。
- metadata.json に加算的 `v2v` ブロック（source_video_id・source_fps・resampled・freeze_ka・audio_head_frozen・トリム幾何）。`GET /config` の limits に `v2v_context_frames_default/min/max` 露出。
- チェーン要求に `loras` フィールドは存在しないため IC-LoRA 併用は構造的に不可（将来チェーンに loras を足す時に 422 を設けること）。

### 24.4 目視ゲート G3＝**ユーザー未実施（OPEN）**・素材準備済み

- **本命（720p・音声付き・源+継続の結合）**: `outputs/v2v_e2e/E2E-A/G3_candidate_joined_source_plus_continuation.mp4`（源=visual_review 09 の 720p 級＋6秒継続・継ぎ目はフレーム128/129）。継続単体＝`outputs/v2v_e2e/E2E-A/continuation_only_1280x768_144f.mp4`・境界モンタージュ＝同 `boundary_report/boundary_montage_128.png`。
- 補助: 多クリップ＝`outputs/v2v_e2e/E2E-B/joined_source_plus_multiclip.mp4`・リサンプル＝`outputs/v2v_e2e/E2E-C/joined_source_plus_continuation.mp4`・スパイクの A/B 比較（variant A の色跳ね現物）＝`outputs/v2v_spike/runs/A_clean|B_clean/joined_source_plus_continuation.mp4`。
- **判定観点**: ①継ぎ目の自然さ（映像） ②色/明度ドリフト ③音声の継ぎ目（クライアント側で源と結合した時の微小クリックが許容か。ハーネス値は 3.5–6.7×・映像は全て連続）。

### 24.5 持ち越し（将来項目）

**（2026-07-27追記）以後の将来項目の管理は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md)へ一本化した。** stage2の音声タイル継ぎ目とfpsリサンプルの全体変換は同書**§4-7**（V2V／チェーンの音声まわりの残件3点）にあたる。1点目（音声継ぎ目のクリック根治）は§24.7のv1.1で対処済みで、残った浅い凹みが§4-7の③。以下の記録は不変。

- 音声継ぎ目のクリック根治（源の実音声とデコード音声のノイズフロア差）: v1 は 30ms フェードで緩和・G3 試聴の結果次第で「サーバー側結合出力＋真のクロスフェード」オプションを検討。
- stage2 の音声タイル継ぎ目（既知・チェーン由来 backlog と同族・V2V 固有ではない）。
- fps リサンプルが全体変換（末尾だけの部分変換に最適化可能・単一ユーザーでは実害小）。VFR 源は未ストレステスト。
- mock の音声数値は 0 固定（実バックエンドと差異あり・docstring 記載済み）。

### 24.6 得られた知見

**（2026-07-27追記）以後の将来項目の管理は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md)へ一本化した。** 本節3点目の`RetakePipeline`在中の知見は、同書**§4-12**（Retake・Inpaint）の「土台」根拠として引用されている。以下の知見の記録は不変。

- **実写・圧縮素材の VAE latent 混入リスクは杞憂だった**（h264-crf35 でも継ぎ目連続・色ドリフトむしろ最小）。
- **causal VAE の先頭 latent 規約**: 末尾 kv 個だけの切り出し注入は先頭アンカー規約とずれる → **context 丸ごと凍結ヘッド**にすれば規約が一致（ComfyUI extend と同運用）。
- **pin `00dc53d` に `RetakePipeline`（動画→initial latent＋mask 部分 denoise）と `tiled_encode` は在る**が、新 vendor ツリーの `video_latent_from_file`/`VideoConditionByMask` は無い（流用不可・チェーンの freeze mask 方式で代替）。
- 検証ハーネスの再利用が効いた: Phase 3 の境界ハーネス（既知 hard-cut 較正済み）を V2V の junction 判定にそのまま流用。

### 24.7 ★G3 試聴 → v1.1（音声継ぎ改善＋VRAM 是正）（2026-07-04 同日後半・同 branch）

> 一次情報: 音声接続リサーチ＝[`V2V_AUDIO_JOIN_RESEARCH.md`](V2V_AUDIO_JOIN_RESEARCH.md)（§4=ハンドル実装）・E2E 追補＝`outputs/v2v_e2e/E2E_REPORT.md`（E2E-A2 節）・VRAM 修正検証＝`outputs/v2v_encode_fix/`・試聴素材＝`outputs/v2v_e2e/E2E-A2|A3|A4/`。用語: 以後「元動画」（旧表記「源動画」と同義）。

**G3 初回試聴（ユーザー・2026-07-04）**: 映像=**完璧**（128→129f の継ぎ目視認不能）。音声=**FAIL**（①音楽がブツ切れ ②同じセリフの反復）。比較用の不採用 variant A は「雰囲気だけ似た別動画の接続」と評され、variant B 採用の正しさを裏付け。

**根本原因（調査で確定）**: ①セリフ反復=検証素材の継続プロンプトが元動画の発話済みセリフを指示していた（素材の不手際・コードのバグではない）②音楽ブツ切れ=クロスフェード機構なしのハード連結＋「音楽を継続させる条件付けは構造的に存在しない」（公式ホスト API も同方式・プロンプトに音楽記述が皆無）。**クリップ連結には本問題は存在しない**（単一ジョブ内の音声は 1 本の連続 latent＋1 回デコード＝貼り合わせ点が無い。`concat_mp4s` の本番呼び出し元も無し）。問題は「別々に生成したファイルの結合面」全般に付随する。

**v1.1 の実施内容（コミット列 `b95a38c`→`933b57a`→`b2c20ee`→`501c5ca`）**:

1. **R0 リサーチ（ユーザー指摘で追加・実装前）**: 音声接合の先行事例調査 → クロスフェード方針は整合・equal-power（`qsin`）/音楽向け 300–500ms/LUFS 整合を確定（[`V2V_AUDIO_JOIN_RESEARCH.md`](V2V_AUDIO_JOIN_RESEARCH.md)）。
2. **`join_v2v` ヘルパー（`933b57a`）**: フェードペア（400ms qsin）＋2パス loudnorm。実素材で継ぎ目 audio ratio 3.52×→**0.29×**。副次発見=生成音声は元動画より 8.6dB 大きかった（ラウドネス整合の価値）。
3. **G3v2 再生成**: セリフ無し＋音楽明示プロンプト・context 最大化（元動画 129f 全量。※監督指示の 145 は元動画長超過で不可・事前検証 422 が正しく拘束）。継ぎ目 audio 0.61×。**ユーザー判定: 谷はあるが音楽は連続・用途を絞れば実用域＝v1 として温存決定**（壊さない）。
4. **谷の平坦化（ユーザー方針: 案出し→先行事例→高性能×低難易度なら試す）**: リサーチ裁定=**ハンドル方式**（エンジンがトリムで捨てている context 領域音声を流用した真の重ね合わせ equal-power クロスフェード）が勝者・位相打ち消しリスクは低〜中（「同じ曲の別テイク編集」として音響編集の定石が明示的に容認・qsin は重ね合わせ時に equal-power）。フェード変種 4 種の計測で**谷の一部（~85ms）は生成内容そのものに由来**（フェードゼロでも存在）と判明。
5. **ハンドル実装（`501c5ca`・オプトイン・現行既定不変）**: エンジンがトリム前全長音声を `<stem>_audio_handle.wav` サイドカー出力（配信 mp4 バイト不変を SHA で実証）＋`join_v2v(handle_audio=…, handle_crossfade_ms=300)`。**ほぼ無音の谷（深さ 0・~710ms）は消滅**、残余はクロスフェード窓内の浅い凹み（深さ 0.42・150ms 版で幅 105ms ≒ 内容由来の下限 85ms）。数値=[`V2V_AUDIO_JOIN_RESEARCH.md`](V2V_AUDIO_JOIN_RESEARCH.md) §4。
6. **★VRAM 是正（`b2c20ee`・ユーザーの「VRAM に載せ続ける必要あるの?」指摘が起点）**: `_encode_source_heads` が context ピクセル全量を GPU 常駐させていた → フレーム毎に GPU 前処理→即 CPU 退避（数値完全一致設計・`tiled_encode` は CPU 入力対応）。**共有 GPU メモリ溢れ ~12.5GB→実質ゼロ・壁時間 584s→279s（2.09×）・出力バイト完全一致**。真因の教訓=**WDDM はエンコード段の一過性超過で torch のページを共有メモリへ降格し、降格ページは戻らないため後続 stage-1 全体が共有メモリ実行になる**（「stage-1 で溢れた」ように見えた正体）。720p/257f は修正後 spill-free（E2E-A2 時の「V2V 実用天井は 217f」暫定判断は撤回・257f まで OK）。
7. **付随修正（GUI・`b95a38c`）**: Settings の spill-free 表が初回 `/config` 取得失敗時に無言で空のまま（=「無限読み込み」に見える）→ 失敗の可視化（警告・英日）＋`gr.Timer` 自動再試行（最大5回）。

**回帰**: T2V/I2V byte-match は各実装後に維持（`23844b4e…`/`a511eda4…`）。同一シード V2V 出力も VRAM 修正・サイドカー追加の前後でバイト一致。pytest **191 passed / 1 skipped**。

**将来項目（記録）**: ①**音声スムージングの UI チェックボックス**（ユーザー要望 2026-07-04）: 将来の GUI V2V 露出時に「結合出力」機能へ ON/OFF を付ける（ON=ハンドル有なら真クロスフェード/無ければフェードペア・OFF=ハード連結。API/エンジン不変＝結合はクライアント側の関心事）②残余の浅い凹みのさらなる平坦化（ノイズフロア整合等）③モデル管理（A1111 風）=[`MODEL_MANAGEMENT_FUTURE_WORKORDER.md`](MODEL_MANAGEMENT_FUTURE_WORKORDER.md)。

> **（2026-07-27追記）以後の将来項目の管理は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md)へ一本化した。** 上記①（音声スムージングのUIチェックボックス）はGUI露出とともに実装済み、②（残余の浅い凹みの平坦化）は同書**§4-7**の③、③（モデル管理）は2026-07-05に実装完了（§26）。③のリンク先だった`MODEL_MANAGEMENT_FUTURE_WORKORDER.md`は2026-07-27の文書整理で削除しており、現在の正本は[`MODEL_MANAGEMENT_DESIGN.md`](MODEL_MANAGEMENT_DESIGN.md)である。

**✅G3 最終試聴 PASS（ユーザー・2026-07-05）**: G3v4（音楽プロンプト継続×ハンドル真クロスフェード・`outputs/v2v_e2e/E2E-A4/`）で「音の繋ぎ目はかなり滑らかになった。自然音やスローテンポの EDM ならまず繋ぎ目に気付かない。音楽や会話の途中なら気づくが、それは現在の生成 AI の性能の限界」＝**合格**。参考: G3v4 の計測は下請けエージェントが生 wav から独立再計算しても完全一致（二重検証済み）。**V2V の目視/試聴ゲートは全クローズ** → push/main マージへ（ユーザー事前決定の条件成立）。

---

## 25. ★audio-to-video（A2V）＝アップロード音声に合わせた動画生成＝設計→実装→実機e2e 全客観ゲートPASS＋**G3試聴（ユーザー・条件付き受容＝マージ承認 2026-07-05）**（branch `feature/a2v`）

> **正本＝本節＋設計書 [`A2V_DESIGN.md`](A2V_DESIGN.md)（ユーザー合意済み Q1-Q4/Q8 含む）。一次情報＝`outputs/a2v_spike/SPIKE_REPORT.md`（G0）・`outputs/a2v_e2e/E2E_REPORT.md`（S3）。outputs は git 管理外のため数値は本節へ転記。**
> base＝main（V2V マージ済 merge `18296b2`）。**G3試聴の結果「おおむね満足」＝ユーザーが push／main マージを承認（2026-07-05）。リップシンク品質の深掘りは §25.5 参照（追加検証・継続）。**

本機: i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11。

### 25.1 何を作ったか（1分）

`POST /generate/chain` に optional `source_audio: {audio_id}` を追加（凍結APIの加算的拡張・省略時byte同一）＋ `POST /upload/audio` を新設。アップロード音声を VAE で latent 化し、AV 結合 latent の**音声側を全長ハード凍結**して動画側だけを denoise する。LTX-2 の動画／音声双方向クロスモーダル注意でリップシンクは生成の中で成立。機構は V2V の凍結 mask を流用（`mask_value=0.0`）＝新規開発なし。出力は**元波形をそのまま mux**（vocoder 不使用）。v1 スコープ＝1クリップのみ・A2V×V2V 排他・トリミング非露出・短い音声は 422・`conditioning_images` 併用可。

### 25.2 ゲート実績（コミット列: 設計docs `83150ca`→S0記録 `7a93259`→S1エンジン `b59d0fb`→G1記録 `30b61c9`→S2 API `e37a9ef`→S2記録 `8cc793e`）

- **G0 スパイク（GO）**: 独立プローブ `outputs/a2v_spike/probe_a2v.py`（n=1・704×448・121f・seed1234 固定・main/wheel 不可触）。①shape 整合・crash 無し（stage1 音声凍結 max drift=**0.000e+00**＝ハード凍結が厳密に成立） ②torch ピーク **8858.7MB**／nvidia-smi dedicated ピーク **~11.2GB**・**共有溢れ無し** ③出力音声==入力 wav（Pearson r=**1.0000/0.9999**・差は AAC 損失のみ） ④**同一 seed・異なる2音声→全フレーム平均絶対差 3.94%**（per-frame MAD 9.22–12.70・**全フレーム非ゼロ**）・目視で口の形／頭部姿勢が相違＝**蒸留経路でも凍結音声がクロスアテンション経由で動画を駆動**（modality_scale 摂動なしで成立・fallback 不要＝§1.4 の最大リスク解消）。1本 ~122秒。素材＝System.Speech TTS 16kHz stereo。**技術知見: 音声 VAE エンコーダは stereo 入力必須（conv_in=[128,2,3,3]）・mux の `_write_audio` も stereo 前提** → mono→stereo 複製の正規化を S1 で実装。
- **G1 回帰（全PASS）**: HEAD `b59d0fb` 時点・本番 runner 経路。T2V `23844b4e…6bb7bf`／I2V `a511eda4…c217`／チェーン no-source `f706057a…0ea1`（640×384・clips[121,121]・seed12345・793973 bytes）すべて基準 SHA-256 と**バイト完全一致**。pytest **199 passed / 1 skipped**（基準191＋chain_math 新規8）。peak_vram_mb=**8440**（§22/§24 と同値）。チェーン最終 VAE デコード瞬間 ~14.9GB（16GB 内・OOM/spill なし）。
- **G2 スモーク（mock＋実機・全PASS）**:
  - **mock**（実 uvicorn）: upload 200→chain 202→completed→a2v メタ正・negative（2クリップ 422／不在 404／A2V+V2V 422）。pytest **212 passed / 1 skipped**（＋`test_a2v_chain.py` 13本）。
  - **実機**（RTX 4070 Ti SUPER・`main.py --port 18620`・config 不変）: TTS 11.19s wav→upload→chain（704×448/121f）→completed **117.1s**。a2v メタ（a_total=126・muxed_original_waveform・vocoder_skipped・source_audio_id）正。CHAIN_OK peak_vram_mb=**8440**・nvidia-smi dedicated peak **14371MB**・shared 平坦（spill なし）。音声一致 Pearson r=**1.0000**・RMS 比 0.9989。negative 実機4件（A2V+V2V 422／不在 404／2クリップ 422／短音声 422 `SOURCE_AUDIO_TOO_SHORT`）全緑。

### 25.3 G3 候補生成（試聴は OPEN）・720p 級

720p 級＝1280×768 生成→crop 1280×720（`standard_720p` プリセット）・121f・REST 経由。全 run で OOM/spill なし。

| ケース | job | 時間 | worker peak | nvidia-smi peak | 音声 r | 出力 |
|---|---|---|---|---|---|---|
| A（セリフ・男声TTSトレイラーナレーション） | `0bab3830` | 175.14s | 9519MB | 14885MB | 0.9999 | `outputs/a2v_g3/caseA_speech_1280x720_121f.mp4` |
| B（音楽のみ・過去LTX生成物のオーケストラ音声流用） | `93fffd3b` | 185.84s | 9525MB | 14715MB | 0.9999 | `outputs/a2v_g3/caseB_music_1280x720_121f.mp4` |

- **客観PASS≠目視ゲート: G3 試聴判定はユーザー OPEN**（ケースA＝リップシンク本丸／ケースB＝音楽）。

### 25.4 正本・成果物

- 設計正本＝[`A2V_DESIGN.md`](A2V_DESIGN.md)（Q1-Q4/Q8 合意済み）。レポート＝`outputs/a2v_spike/SPIKE_REPORT.md`（G0）・`outputs/a2v_e2e/E2E_REPORT.md`（S3）＝outputs は git 管理外。

### 25.5 ★G3 試聴結果（ユーザー・2026-07-05）＝条件付き受容・リップシンク品質の追加検証へ

**総評: おおむね満足 → push／main マージ承認。** 個別判定:

| 素材 | 判定 |
|---|---|
| スパイク run_A（704×448・女声Zira・静的クローズアップ） | **かなり正確にリップシンク** |
| スパイク run_B（704×448・男声David早口） | 意図は見えるが一致は弱い |
| G3 ケースA（720p・男声トレイラーナレ・賑やかな町＋歩行） | 意図は見えるが一致は弱い |
| G3 ケースB（720p・音楽のみ） | 大きな問題なし。セリフ終了と同時にカメラがパンして女性が画面外へ＝「音と一致させようとしている」ことは分かる |

**ユーザー考察（仮説・モデル性質由来の可能性）**: ①女声の方がリップシンクが得意 ②当該男声音源がたまたま LTX 2.3 が解釈しにくい波形だった ③背景の書き込み・人物の移動が増えるとリップシンクが弱くなる（スパイク＝静的クローズアップとの差）。→ **バグではなくモデル性質の可能性が高い**との見立て。

**フォローアップ（発注済み・2026-07-05）**: (a) コミュニティ報告の広域リサーチ（LTX 2.3 の A2V/リップシンクの使い勝手） (b) 仮説切り分けの追加検証動画 2〜3本（声質×背景複雑度のマトリクス） (c) Stage1 クロスモーダル摂動ガイダンス（上流 `a2v_guidance_scale`）差し込み案の平易な解説→採否判断。結果は本節に追記する。

#### 25.5.1 フォローアップ (a) コミュニティリサーチ結果（2026-07-05・正本=[`A2V_LIPSYNC_COMMUNITY_RESEARCH.md`](A2V_LIPSYNC_COMMUNITY_RESEARCH.md)）

- **仮説①（女声が得意）＝✗支持されない（むしろ逆報告あり）**。我々の Zira>David は性別でなく「David の早口・明瞭度・波形品質」の交絡で説明する方が整合的。
- **仮説②（音源の波形が苦手）＝○支持（複数報告）**。本質は「音声の明瞭さ・プロソディ・感情整合」。**動画尺 5〜6 秒未満で同期不発の報告あり**（今回の検証は 5.04 秒＝該当しうる）。推奨=先頭 0.2〜0.5s 無音・10 秒以上の尺・明瞭な発話。
- **仮説③＝○「動き・広い画角・複数話者」の形で公式が明言**（arXiv 2601.03233 Limitations・公式 prompt guide）。「賑やかな町×歩行×ナレ」が弱いのは**既知のモデル性質どおり＝実装バグではない可能性が高い**。「複雑背景」単独の一次資料は無し（真のギャップ）。
- 実践ノブ上位: ①クローズアップ・単一話者・プロンプトに `lip sync` 明示（期待度高） ②音声前処理（無音マージン・10s+・明瞭 TTS）（中〜高） ③ComfyUI 系はモーション制御ガイド off（中）。
- `a2v_guidance_scale`＝**上流既定 3.0**（1.0 で無効）。蒸留 vs dev のリップシンク差は信頼データ無し。

#### 25.5.2 フォローアップ (b) 仮説切り分けマトリクス生成（2026-07-05・seed 4242 固定・パラメータ既存 caseA と同一・一覧=`outputs/a2v_g3/HYPOTHESIS_MATRIX.md`）

| セル | ケース | job | 時間 | worker peak | 出力 |
|---|---|---|---|---|---|
| 複雑シーン×女声 | **caseC** | `c26a313f` | 194.2s | 8562MB | `outputs/a2v_g3/caseC_town_woman_1280x720_121f.mp4` |
| 単純シーン×男声（caseA と同一 wav） | **caseD** | `495fdf8a` | 185.7s | 9524MB | `outputs/a2v_g3/caseD_headshot_man_1280x720_121f.mp4` |
| 単純シーン×女声（スパイク run_A の 720p 再現） | **caseE** | `072a1ed6` | 214.2s | 9526MB | `outputs/a2v_g3/caseE_headshot_woman_1280x720_121f.mp4` |

- 既存 caseA（複雑×男声）＋新 3 本で 2×2 の 4 隅が 720p・同一 seed で完備（＋低解像スパイク run_A）。音声一致 r≈1.0・全 run spill/OOM なし。
- 注: 3 本は比較可能性のため**改善ノブ適用前**の条件（`lip sync` 明示・10s+ 尺などは未適用）。

#### 25.5.3 ★マトリクス試聴結果（ユーザー・2026-07-05）＝**シーン要因で確定・モデル性質と結論**

| セル | 判定 |
|---|---|
| caseC（複雑シーン×女声） | 意図は感じるが**かなり大きくズレる** |
| caseD（クローズアップ×男声・caseA と同一 wav） | **完璧なリップシンク** |
| caseE（クローズアップ×女声） | **完璧なリップシンク** |
| （参考）caseA（複雑シーン×男声・試聴済み） | 弱い |

**結論**: 決定要因は**シーンの複雑さ・人物の動き・画角**（仮説③）。同一の男声 wav がクローズアップで完璧＝仮説②（音源）は実用上消え、caseC の劣化で仮説①（声質）も消えた。**公式 Limitations（動き・広角で口の追従が落ちる）と完全に整合＝実装ミスではなく LTX 2.3 のモデル性質で確定（ユーザー・監督合意）**。720p でもクローズアップなら完璧が再現（解像度は無関係）。

**運用への帰結**: A2V のリップシンクは「クローズアップ・単一話者・動き控えめ」の構図で使う機能として案内する（GUI 露出時のガイド文言・プロンプト例に反映すること）。複雑シーンでの強化が将来必要になった場合の選択肢＝Stage1 クロスモーダル摂動ガイダンス差し込み（生成 ~2倍・上流既定 3.0・蒸留との相性未知）は**未採用のまま選択肢として保留**（§25.5 フォローアップ (c)）。本フォローアップはクローズ。

---

## 26. ★GUI への V2V/A2V 露出＋モデル管理ドロップダウン＋コンソールログ改修＝3子並行オーケストレーション実装＋G3 目視ゲート＋フィードバック対応（2026-07-05）

> **正本＝本節。** 設計正本＝モデル管理=[`MODEL_MANAGEMENT_DESIGN.md`](MODEL_MANAGEMENT_DESIGN.md)／結合API=[`mockups/JOIN_API_PROPOSAL.md`](mockups/JOIN_API_PROPOSAL.md)／GUI モック=[`mockups/GUI_V2V_A2V_MOCK.html`](mockups/GUI_V2V_A2V_MOCK.html)。前提＝§25 完結（A2V main マージ済）。本セッションは **Fable5 親＋Opus 子の並行オーケストレーション**（フェーズ1＝3子・フェーズ2＝1子）で実施。GPU 所有権は常に1つ＝親が逐次貸与（並行 GPU 実行なし・従来規律維持）。
> 本機: i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11。

### 26.1 フェーズ1＝3子並行（各自 worktree ブランチ→親が順にマージ）

**題目C＝コンソールログ改修**（正本の調査＝[`CONSOLE_LOG_FORGE_NEO_RESEARCH.md`](CONSOLE_LOG_FORGE_NEO_RESEARCH.md)）
- 内容: httpx のポーリング行抑制＋worker logging 土台（従来無出力だった engine 側 logger 群の可視化）＋ジョブ開始/終了・ステージ進捗・完了サマリ（peak_vram＋所要時間）のコンソール出力。
- コミット列: `afc66aa`→`a3d7f3b`→`199e8da`→**merge `b77a812`**。
- **S3 実機（GPU 窓）**: byte-match 3系統完全一致（T2V `23844b4e…`／I2V `a511eda4…`／チェーン `f706057a…`）・peak_vram **8440** 再現・server.log の httpx ポーリング行 **68%→0行**。

**題目B＝モデル管理ドロップダウン**（正本の設計＝[`MODEL_MANAGEMENT_DESIGN.md`](MODEL_MANAGEMENT_DESIGN.md)）
- 内容: 4カテゴリレジストリ（transformer／text_encoder／video_vae／audio＝音声 VAE＋vocoder は物理1ファイル）＋`GET /models`＋`POST /pipeline/load` の optional モデル指定（省略時＝golden スナップショットで byte 同一固定）＋Settings タブ GUI（IC-LoRA ドロップダウンの型を踏襲）。
- コミット列: `8c762b9`→`4613b7e`→`a062274`→`e3e3601`→`42f36b2`→（rebase 後 `5ad77d3`）→merge 済み。
- **S4 実機（GPU 窓）**: 別名二重登録（default-alias）で swap→同 seed 生成→**SHA 完全一致×3**（切替配線の実証）・未知名 404・後片付け済み。**実代替モデルの DL はしない方針（ユーザー決定・検証後に余計な登録を削除）**。

**題目A＝GUI V2V/A2V 露出＋サーバー側結合 API**（正本の提案＝[`mockups/JOIN_API_PROPOSAL.md`](mockups/JOIN_API_PROPOSAL.md)）
- 内容: ユーザー承認モック（[`mockups/GUI_V2V_A2V_MOCK.html`](mockups/GUI_V2V_A2V_MOCK.html)）→`POST /jobs/{id}/join`＋`GET /jobs/{id}/joined` 新設（`services/join_manager.py`・解像度/fps 不一致時の正規化＝**setsar 必須の発見**を含む）→Clip Chain タブに生成モード Radio（なし／V2V 継続／A2V 音声駆動＝構造的排他）＋A2V ガイド（クローズアップ・単一話者・動き控えめ＝§25.5.3 反映）。
- コミット列: `13b5501`→`6fbe763`→`214c26f`（rebase 後 `dce6fc9`→`8b08b5c`→`c7fce67`）→**merge `f999374`**。
- **実機 e2e（GPU 窓）**: V2V **190.5s**／peak **9889**・join **3.2s**（handle_crossfade・正規化発動・尺誤差 **0.000**）・A2V **185.2s**／音声一致 r=**1.0000**。

**本セッションのユーザー決定**: 音声スムージング＝サーバー側結合 API 方式・切替検証＝別名二重登録のみ・モデル UI＝Settings タブ内。

### 26.2 G3 目視ゲート（ユーザー・2026-07-05）

| 素材 | 判定 |
|---|---|
| A2V（GUI 経由・joined 前の単発） | **完璧（リップシンク含む）** |
| i2v 操作/出力 | **完璧** |
| A2V 音量 +3dB | 違和感なし＝**研究課題へ格下げ（クローズ）** |
| V2V 結合版（子A 素材） | **FAIL＝音楽ブツ切り＋新規部分の内容ジャンプ** |

- **V2V FAIL の原因分析（確定）**: 子A の素材が §24.7 の教訓（元動画と同じシーンを継続するプロンプト・context 最大化）に反していた＝**コード退行ではない**（後述 26.3 の F6/再検証で二重実証）。
- 積み残し症状: GUI 進捗「3% (step None/None)」凍結＋uvicorn アクセスログ洪水＝26.3 で対処。

### 26.3 フェーズ2＝G3 フィードバック対応（子F＝Opus1人・branch `fix/g3-feedback`）

- **F1** uvicorn ジョブポーリング GET の選択的抑制（カスタム log_config＋Filter・POST/エラー/video 系は残す）。
- **F2** ステップ単位進捗: wheel の `ltx_pipelines.utils.samplers.tqdm` を `engine/progress_shim.py` のシムに実行時差し替え（wheel 無改変・数値非干渉）＋チェーンに encode イベント＋単発経路 `_RealBackend.generate` の progress 受理ループ化＋コンソール n/N・it/s 整形。
- **F3** GUI 進捗整形（step None 時は % のみ＋工程名ラベル・`JobResponse` に optional `stage` 加算）。
- **F4** V2V 継続の作法ガイド（同一シーン継続を記述・セリフ再指示禁止・音楽継続明示・参照フレーム数最大化）。
- **F5** クロスフェード長: 既定 150→**300ms**＋GUI ドロップダウン 150/300/500。
- コミット列: `fc9ce2d`→`33f1be4`→`3bd6100`→`2a955ac`→`821db57`→**merge `b8d9236`**。pytest **343 passed / 1 skipped**（本セッション累計: 212→343）。
- **GPU 窓④実機**: **byte-match×3 完全一致**（T2V/I2V＋**V2V 再検証 output.mp4 が §24.7 G3v4 合格素材 `b7a2fa21…8aa7` とバイト一致**＝シム非干渉＋コード退行なしの二重実証）・単発ステップ進捗の実機動作（step1/8→8/8→stage2 1/3→3/3）・uvicorn ポーリング行 **0**。
- **ユーザー再ゲート（2026-07-05）**: joined.mp4＝完璧・ログ＝完璧・GUI 進捗表示＝完璧→**3件クローズ**。

### 26.4 ★次セッション持ち越し＝GUI 経由のクリップ連結＋結合の症状（分析・仮説まで本セッション実施・調査/修正は次セッション）

**ユーザー症状（GUI 操作時）**: ①1クリップ目だけ生成され結合版が作られない ②2クリップ目が作られない ③「Create joined version」で 503＋ffmpeg エラー `[loudnorm] Value -inf for parameter 'I' out of range [-70 - -5]`。

**親の一次調査（確定事実）**:
- 対象 job `03c0a691`（wtA outputs）＝**正常 completed**（313.8s・clips=2・total 225f・new_frames_px=152・source_had_audio=true・audio_handle 出力済み・source_fps=16→24 リサンプル）。
- 元動画（upload `ed7d8b41`）の音声は**デジタル無音**（AAC 2kb/s・volumedetect mean_volume **-91.0 dB**）。

**仮説（次セッションで検証）**:
- **H1（③の本命・確度高）**: `join_v2v` の2パス loudnorm が無音音声の実測 I=**-inf** をそのまま第2パスに渡し ffmpeg が拒否→FFmpegError→503。`source_had_audio=true`（ストリーム存在）だが無音のため、音声なしフォールバック（video_only）も発動しない盲点。**対処方向＝実測が -inf/範囲外なら loudness 整合をスキップ（loudness_matched=false）してクロスフェードのみ実施**。
- **H2（①は UX と③の複合）**: 結合版は自動生成ではなく手動ボタン（設計どおり）＋クリックすると③で失敗＝「作られない」に見える。
- **H3（②は仕様と期待のギャップの可能性）**: metadata 上クリップ2は生成済み（new_frames_px=152）。V2V では参照フレーム数 73 がクリップ1の頭を置換するため新規部分＝152f≈6.3s（2×121f≈10s ではない）。出力が想定より短い/クリップ境界が滑らかで区別できないことが「2クリップ目が無い」に見えた可能性。GUI 側で「新規部分の長さ＝クリップ合計−参照フレーム数」の説明が不足。※**実物の目視確認は未実施＝次セッションで出力内容を確認してから確定**。
- 副次: 16fps→24fps リサンプルは正常動作。**無音音声の 16fps 素材という入力条件そのものがこれまでの検証マトリクスに無かった（新しい入力クラス）**。

---

## 27. ★GUI 経由クリップ連結＋結合の症状解消＝無音音声素材の V2V 結合 503 修正（2026-07-05）

> **正本＝本節。** §26.4 の持ち越し（仮説 H1/H2/H3）の検証と対処。作業ブランチ＝`fix/v2v-join-silent-loudnorm`（worktree・push／main マージはユーザー承認待ち）。GPU 生成を伴わない修正のため、実装・単体検証は ffmpeg/CPU のみで完結（実機 e2e は親が別途実施・後述 27.6）。

### 27.1 症状（§26.4 からの持ち越し）

GUI の V2V クリップ連結で「Create joined version」を押すと **HTTP 503**＋ffmpeg エラー `[loudnorm] Value -inf for parameter 'I' out of range [-70 - -5]`。素材＝ユーザーの元動画（upload `ed7d8b41`・1280x720・16fps・音声ストリームは存在するがデジタル無音＝volumedetect mean_volume **-91.0 dB**）。

### 27.2 確定原因＝H1（2パス loudnorm の -inf 無ガード）

`services/video_io.py::join_v2v` の音量整合（ラウドネスマッチ）は2パス方式＝第1パスで両クリップの実測ラウドネス（LUFS）を測り、第2パスの `loudnorm` フィルタに目標値・実測値として展開する。**音声ストリームが存在するがデジタル無音のとき、第1パスの実測 `input_i` が `-inf` になり、それを無ガードで `loudnorm=I=…:measured_I=…` に展開するため、ffmpeg が範囲外（`I` の許容範囲 [-70, -5]）として拒否**→FFmpegError→503。

- 該当箇所（修正前）: with_audio フェードペア経路と handle 真クロスフェード経路の両方（同じ展開ロジック）。
- 盲点: `has_audio_stream` は**ストリームの有無しか見ない**ため、`source_had_audio=true` のまま音声なしフォールバック（video_only）に落ちず、必ず loudnorm 展開に進んでいた。

### 27.3 仮説検証の経緯（修正前に再現→修正→同手順成功）

1. **修正前の再現（実素材）**: 無音 16fps 元動画を継続クリップの 1280x768・24fps へ正規化（`normalize_clip`・`join_manager.py` と同じ判定）→ `join_v2v` 実行→ **FFmpegError 再現**。エラー本文＝`[Parsed_loudnorm_2] Value -inf for parameter 'I' out of range [-70 - -5]` / `Error applying option 'I' to filter 'loudnorm': Result too large`。仮説 H1 と完全一致。
2. 修正（27.4）を実装。
3. **同一手順を再実行→成功**: 517 フレーム（正規化後 source 365f＋新規 152f）・音声 AAC あり・`loudness_matched=false`・`loudness_skip_reason` 設定・警告ログ `source input_i='-inf', continuation input_i='-8.32'`。

### 27.4 対処＝実測値が使えないときは音量整合をスキップ（クロスフェードのみ実施）

`services/video_io.py` に検査ヘルパー `_loudnorm_stats_usable(stats: dict, *, as_target: bool) -> bool`（553行〜）を新設し、第1パスの実測値（`input_i`/`input_tp`/`input_lra`/`input_thresh`/`target_offset`）を float 化・**非有限（-inf/NaN）または範囲外なら不合格**とする。範囲＝`input_i` が第2パスの目標 `I=` に渡る側（元動画）は ffmpeg の許容 **[-70, -5]**・実測 `measured_*` として渡る側（継続/ハンドル）は **[-99, 0]**。

- with_audio フェードペア経路（812行〜）と handle 真クロスフェード経路（764行〜）の両方で、**どちらか一方でも不合格なら loudnorm フィルタを組まずスキップ**: `loudness_matched=false` のまま・LUFS 欄は null のまま・クロスフェード（フェードペア／acrossfade）はそのまま実施。内部 info に `loudness_skip_reason` を記録し、`logger.warning`（`ltx.video_io`）で実測生値を出力。
- **クランプ方式（ffmpeg-normalize が採る -inf→-99 置換）を不採用にした理由**: 先行事例の ffmpeg-normalize は「固定目標へ正規化する」ツールなので実測側のクランプで足りるが、我々は「**元動画の実測 LUFS を目標 `I=` に渡して継続側を合わせる**」整合であり、無音（-inf→-99 相当）に合わせると**継続クリップの音声を無音近くまで潰してしまう**。無音素材に音量を「合わせる」こと自体が無意味なので、スキップが正しい。
- **凍結 API は無変更**: `api/models.py::JoinResponse` に手を入れず、`loudness_skip_reason` は info 内部キーに留めた（`join_manager.py` は明示キーしか拾わないためレスポンスに漏れない）。

### 27.5 テスト＝anullsrc フィクスチャ＋3本追加（pytest 346 passed / 1 skipped）

- `tests/test_video_io.py::_make_v2v_clip` に `silent: bool = False` を追加（sine の代わりに `anullsrc`＝ストリームあり・信号ゼロ。既存呼び出しの挙動は不変）。
- 追加 3 本（実 ffmpeg 使用・既存作法）: ①無音 source×有音 continuation（フェードペア） ②有音 source×無音 continuation ③handle 経路×無音 source。いずれも**例外なく成功・`loudness_matched=false`・LUFS 欄 null・`loudness_skip_reason` あり・フレーム数＝source+continuation・音声ストリームあり**を確認。
- pytest **346 passed / 1 skipped**（343+1 → +3）。既存の有音経路テスト（`test_join_v2v_loudness_match_changes_continuation_level` 等）は緑のまま＝**正常値は従来どおり素通しで整合が掛かる（非退行の証拠）**。

### 27.6 実機 e2e（親が別エージェントで実施・PASS・2026-07-05）

新ジョブ **`e4a34bf5-941f-4ed5-a9cc-ed6d82783964`**: 無音 16fps 素材（ユーザー症状素材のコピー）をアップロード→V2V チェーン生成（2×121f・context 73・1280x768・**312.66s**・peak_vram_mb **9889**）→ `POST /jobs/{id}/join` が **HTTP 200（503 解消）**。

- join レスポンス: `join_mode=handle_crossfade`・`handle_crossfade_ms_applied=300`・`loudness_matched=false`・`source_lufs=null`・**`loudness_skip_reason` はレスポンスに非露出＝凍結 API 無変更の実証**。
- joined.mp4: **21.54 秒・517 フレーム**（正規化 source 365f＋新規 152f）・1280x768 24fps・音声 AAC あり。
- server.log:1841 に `WARNING ltx.video_io: join_v2v handle: skipping loudness match (source input_i='-inf', handle input_i='-8.08')`。**503／ERROR／Traceback なし**。

### 27.7 H2/H3 の結論

- **H2（①「結合版が作られない」）＝手動ボタン＋503 失敗の複合で確定**。結合版は設計どおり手動ボタンで作るものであり、クリックすると③の 503 で失敗していたため「作られない」に見えた。**コード修正（H1）で解消**。
- **H3（②「2クリップ目が無い」）＝仕様と期待のギャップで確定**。metadata 上クリップ2は生成済み（new_frames_px=152）。V2V では参照フレーム数（73）がクリップ1の頭を置換するため、**出力は新規部分のみ＝152f≈6.3 秒**（2×121f≈10 秒ではない）＝仕様どおり。対処＝GUI の V2V 説明文（`gradio_ui/i18n.py` の `v2v_cap_panel`・日英両方）に「出力の長さ≈（総フレーム数−参照フレーム数）÷24 秒」の説明と例を追記。
- **ユーザー目視ゲート＝OPEN**: ①output.mp4（6.3 秒）にクリップ2の内容が含まれるかの実物目視（H3 の最終確定） ②joined.mp4（`outputs/e4a34bf5-…/joined.mp4`）の目視試聴。→ **✅クローズ（§27.9・2026-07-05 夜「問題なし」）**

### 27.8 コミット列・状態

- コミット列（branch `fix/v2v-join-silent-loudnorm`）: `c7ad906`（fix: video_io ガード＋スキップ＋警告ログ）→`6ba83ae`（test: anullsrc フィクスチャ＋3本）→`1422543`（docs(i18n): v2v_cap_panel 尺説明）→`2322ac8`（docs: §27＋handoff）→`5557dcb`（fix(gui): トースト警告・§27.9）→docs 追記コミット。
- 凍結 API 無変更・wheel 無改変・GPU 経路無変更（ffmpeg フィルタ組み立てのみ）。**push／main マージ＝ユーザー承認待ち**。

### 27.9 ユーザー実機 GUI 再検証の顛末＋事前チェックのトースト警告化（2026-07-05 夜）

**ユーザー目視ゲート 2 件＝クローズ**: `03c0a691` の output.mp4（新規部分 6.3 秒）と `e4a34bf5` の joined.mp4 をユーザーが確認し「問題なし」→ **H3＝仕様で最終確定**。

**GUI 再検証で新事象→原因確定（コード退行ではない）**:
- 症状: V2V モードで「Generate chain」を押しても生成が始まらないように見える（Gradio のイベントはサーバーに届くがバックエンド API は呼ばれない）。
- 調査: ①ハンドラの全早期 return 経路の精読（配線・モード Radio の内部値・ボタン有効状態に問題なし） ②git 履歴の突き合わせ（20:05 に成功した時点のコードと現 main で「Generate chain」まわりの配線・比較値は機能的に同一＝**退行なし**）。
- **原因（ユーザー確認で確定）＝共有プロンプト（Prompt (shared)）が空欄**のまま押したため、事前チェックの最初の関門で早期 return。メッセージは `chain_progress` テキスト欄に 1 行出るだけで、見落として「何も起きない」と解釈された。
- プロンプトを入れた再操作: チェーン job **`9b4a184d`**（2×121f・225f・314.5s・peak_vram_mb **9889**）＝ログの `chain stage-1 denoise [1/2]`／`[2/2]` どおり**両クリップ生成成功**→「Create joined version」→join **HTTP 200**→joined.mp4 が **GUI 内の結合版プレーヤーでプレビュー再生まで全工程成功**（この join は音あり素材＝音量整合は通常経路）。「クリップ2が見えない」＝クリップは 1 本に連続デコードされる設計（§19）＋出力は新規部分のみ表示（§27.7 H3）の仕様どおり。
- 副次の確認: 22:43 の実行は修正ブランチ未マージの main 上だったが、音あり素材のため join は成功（無音素材対策の本修正は引き続きブランチ側にある）。

**対処（ユーザー指示）＝事前チェック拒否の可視化（コミット `5557dcb`）**:
- `gradio_ui/handlers.py` に `_precheck_reject(message)` ヘルパーを新設し、`generate_chain` の**バックエンド API 呼び出し前の早期 return 全 22 箇所**（プロンプト空・寸法・fps・クリップ数・フレーム数・V2V/A2V 各チェック・総尺ジオメトリ）で、従来のテキスト欄表示に加えて **`gr.Warning` のトースト警告**（同一のローカライズ済み文字列）を発火。
- 裏取り: 使用中の gradio 6.19.0 で `gr.Warning` は raise しない通知 API（キュー有効イベント内＝黄色いトースト表示・イベント外＝`warnings.warn` に無害フォールバック）。generator ハンドラは必ずキュー経由のため実 UI では確実にトーストになる。
- スコープ: `generate_chain` のみ（Generate タブ・join ハンドラは不変）。新規 i18n 文字列なし・yield 構造不変。
- テスト: 共有プロンプト空で「API 呼び出しゼロ＋`gr.Warning` ちょうど 1 回＋トースト文字列＝テキスト欄文字列」を検証する 1 本を追加。pytest **347 passed / 1 skipped**（346+1 → +1）。
- **残 OPEN**: main マージ後にユーザーが空プロンプトで押してトーストが出ることの実機確認。

### 27.10 チェーン進捗のクリップ位置表示（ユーザー要望・2026-07-06 未明）

**要望の本質（ユーザー整理）**: 不便さの真因は「今いくつのクリップまで生成が終わったのか」「本当に全クリップ生成されたのか」が GUI やログから読めないこと。クリップごとの VAE デコードプレビュー（ComfyUI 式）は**大半を捨てるデータをデコードする無駄なので不採用（ユーザー判断）**。文字情報で表示する。

**配線調査（実装前）**: worker のステップ進捗イベントは F2 の tqdm シム由来の `outer_index`/`outer_total`（stage-1 のセグメント＝クリップ位置）を既に運んでいたが、`services/ltx_runner.py::_read_worker_events` がコンソールラベル `[1/2]` と進捗率補間に使うだけで **progress コールバックへ渡す時点で捨てていた**（＝欠落点はサーバー側の1箇所のみ・engine/wheel 改変不要）。

**実装（コミット `8bfd559`＝サーバー側・`338519f`＝GUI 側）**:
- `api/models.py::JobResponse` に **optional 加算** `clip: int | None`・`clip_count: int | None`（既定 None・凍結 API の加算的変更＝F3 の `stage` と同型。単発生成・キュー中・mock・旧 worker では None のまま）。`services/job_store.py::JobRecord` 経由で配管。
- `_read_worker_events`: chain の stage-1 系イベントに限り `clip=`/`clip_count=` キーワードを追加（per-step＝処理中クリップの 1 始まり番号・粗い per-segment＝完了したクリップ番号）。**既知のときだけ渡す**ため他イベント・単発経路の呼び出し形は完全不変。stage-2 のタイル位置はクリップと誤認しないようガード。
- chain の `on_progress` は clip 到着時のみ保持し、stage-2／デコード中も最後の値を維持（「クリップ N/N」＝全クリップが stage-1 通過済み、の意味づけ）。
- GUI: 進捗テキストを「生成中… 20% (step 3/8) — デノイズ中 (stage 1) — クリップ 1/2」形式（i18n 日英・新キー `msg_clip_progress`/`msg_all_clips_done`）。完了行に「全 N クリップ処理済み」。clip 情報なしのジョブは従来表示と完全同一。

**検証**:
- テスト 7 本追加（stage-1 のみ clip が届く／stage-2 タイル位置は clip にしない／単発では渡らない／JobRecord→JobResponse 配管／GUI 整形 日英／完了行の有無で従来文字列不変）。pytest **354 passed / 1 skipped**（347+1 → +7）。
- **実機 e2e PASS（2026-07-06 未明・job `fa6b37d3`）**: 通常 2 クリップチェーン（各 49f・1280x768・seed 42）で 1 秒間隔ポーリング→ `clip` が **None（encode）→1/2（stage-1 前半）→2/2（stage-1 後半）→2/2 維持（stage-2／デコード／completed）** と設計どおり遷移。従来のコンソール行（`chain stage-1 denoise [1/2]` 等）不変・ERROR/Traceback なし・**peak_vram_mb 8440＝チェーン回帰基準値と一致**。
- **残 OPEN**: main マージ後、ユーザーが GUI で「クリップ n/N」表示を実機確認。あわせてクリップ1/2 に別々の個別プロンプトを入れて生成し、出力後半で内容が切り替わること＝クリップ2の実在を目視確認する手順を案内済み。

---

## 28. ★IC-LoRA 制御の効き具合（strength）可変化＝`conditioning_attention_strength`＋`reference_video_strength`（2026-07-06・branch `feature/ic-lora-strength`・**目視ゲート／マージはユーザー承認待ち**）

> **正本＝本節。** [`IC_LORA_PHASE_C_STATUS.md`](IC_LORA_PHASE_C_STATUS.md) の「strength=1.0固定」制約（既知の制約）とスコープ外節「strength可変化」を実装で解消した回。凍結 API の加算的変更（optional フィールド・**省略時 byte 同一**の定型ゲート）で3層（API／エンジン／GUI）に配線。
> commit `2a2bd06`（S3=GUI）→`3e5f367`（S1=API/payload/metadata）→`e2039f6`（S2=engine 配線＋unit）。**push／main マージはユーザー承認待ち**。

本機: i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`。

### 28.1 リサーチ結論＝「制御にどれだけ従わせるか」のノブは3つ（コード読解＋Web・2系統一致）

コードの実配線と公式ドキュメントを突き合わせ、「制御追従度」を左右するパラメータを3つに切り分けた。

| ノブ | 実体 | 従来状態 | 出典（照合済み） |
|---|---|---|---|
| ①参照条件付け strength | `VideoConditionByReferenceLatent.strength`（`denoise_mask = 1 − s`） | `ltx_runner.py` で **1.0 固定**送信（worker→pipeline は既に可変対応済み） | 公式 tutorial（ltxworkflow.com IC-LoRA guide）の「<1.0 で reference が pop／bleed-through」警告は**このノブ**の話 |
| ②`conditioning_attention_strength` | noisy↔reference トークン間 attention への **log-space 加算バイアス** | 上流 `ICLoraPipeline` 専用で我々の `DistilledPipeline` 経路には**未配線**だった | Lightricks 公式 docs（docs.ltx.video）が「control signal にどれだけ厳密に従うか」のノブと明記・0〜1・既定 1.0・0.5=バランス・depth 系 0.6 推奨・**アーティファクト警告なし** |
| ③LoRA アダプタ強度 | B@A スケール（per-layer） | **既に全層配線済み**（API 0〜2.0＋GUI Adapter strength スライダー）＝今回不変 | Phase B G0-c |

- **ユーザー決定**: ①と②の両方を可変化・既定値＝公式既定の **1.0**（省略時 byte 同一）。**②が本命ノブ**（アーティファクト警告なし・追従度を素直に緩める）。
- **上流 parity（省略時 byte 同一の構造的裏付け）**: `iclora_utils.py:122-140` は strength<1.0 のときだけ `ConditioningItemAttentionStrengthWrapper` で包み、1.0 ではラッパー自体を作らない。したがって「1.0＝no-wrapper」は上流 verbatim で成立する。

### 28.2 S0 スパイク（GPU 実測・懸念潰し・pose 1280×768/121f/seed12345）

②のラッパーが挟むマスク付き attention で SDPA が math フォールバックへ落ちて VRAM が跳ねないか、を実測で潰した。

- **カーネル選択**: マスク付き video self-attention（q=(1,32,4800,128)・mask=(1,1,4800,4800) bf16）は **cuDNN カーネル選択**（スコア行列を実体化しない）。math フォールバックは一度も発生せず。
- **ラッパーの VRAM 増≈ゼロ**: フレッシュワーカーでラッパーあり(0.6)=peak **8544** vs なし=**8548** ＝差は誤差帯。**緩和策（EFFICIENT 強制）不要**と判断。
- **ベースライン健全性 PASS**: Job A（②省略）peak **8548** ＝ Phase C Job P 基準と完全一致。

### 28.3 実装（3層・凍結 API 加算的変更）

- **API（`api/models.py`）**: `GenerateRequest` に optional 2フィールド `conditioning_attention_strength: float|None (ge=0, le=1)`・`reference_video_strength: float|None (ge=0, le=1)`。`loras` なしで指定→**422 VALIDATION_ERROR**（新エラーコードなし）。
- **ペイロード（`services/ltx_runner.py`）**: `reference_video.strength` ＝省略時 1.0（従来 byte 同一）／指定時その値。`attention_strength` キーは**指定時のみ splice**（省略時はキー自体なし）。
- **metadata（`services/pipeline_manager.py`）**: 各フィールド非 None のときのみ `ic_lora` ブロックに記録。
- **エンジン（`engine/worker.py`＋`engine/pipeline/fast_video_pipeline.py`）**: `attention_strength` を payload から parse→`ic_attention_strength` として配線→`_reference_conditioning_for_stage` で **<1.0 のときのみ** `ConditioningItemAttentionStrengthWrapper` で包む（上流 verbatim・**IC-LoRA 重みパッチ機構は不可触**のまま）。参照 strength（①）は worker→pipeline が既に可変対応済みだったため**送信側の固定解除のみ**。
- **GUI（`gradio_ui/`）**: Generate タブ IC-LoRA アコーディオンにスライダー2本（0.0〜1.0 step 0.05 既定 1.0）。「制御追従度 / Control adherence」（info: 1.0=制御信号に厳密に従う・下げるほど自由に解釈・推奨 0.5〜0.7）、「参照強度 / Reference strength」（info: 通常 1.0 のまま・1.0 未満は参照映像が滲み込むことがある＝公式の注意）。**値<1.0 のときのみキー送出**＝既定操作は API リクエストも従来形。EN/JA i18n。

### 28.4 検証ゲート（全 PASS・実 API 経由・一次ソース＝`peak_vram_mb`）

| ゲート | job | 結果 |
|---|---|---|
| G1 T2V 基準 | `05bee2d0…` 512×320/49f | SHA `23844b4eebd107ccba8c5534eb65bab86575cca0b9050cb6c7e680a4506bb7bf` **完全一致**・peak **8440**・104.8s |
| G2 IC-LoRA 基準（両フィールド省略） | `c7dcd1e1…` 1024×640/25f upscaler | SHA `735a6de97d2deb56a781c66307849924a8ac25b51f0591fbddab07a78875e272` **完全一致**・peak **9525**・128.0s |
| G3 明示 1.0/1.0 | `8d838f27…` | G2 と SHA **完全一致**＝「1.0＝no-wrapper」の e2e 実証・peak **9534** |
| E2E-A attention 0.6 | `2f9a1e41…` pose 1280×768/121f | 完走・metadata 両所に 0.6 記録・peak **9537**（≤9541 帯）・215.9s |
| E2E-B 参照 0.8 | `a62d8b74…` | 完走・metadata 記録・**attention キー正しく不在**・peak **9542**・215.6s |
| pytest | — | **365 passed / 1 skipped**（旧基準 354→+11 新テスト＝S1 6本 API/payload・S2 2本 engine・S3 3本 GUI） |

- G2/G3 の peak が Phase B 期の 8440 と違うのは**同一ワーカー連続実行のアロケータ状態持ち越し**（SHA 一致が計算不変を証明）。OOM／WDDM 共有溢れなし。

### 28.5 OPEN（ユーザー宿題・次回）

1. **目視ゲート（ユーザー）**: `outputs/visual_review/` — **#10 vs #14**（同プロンプト／seed／参照の追従度 1.0 vs 0.6・**本命ペア**）・**#15**（参照強度 0.8・滲み／ポップスルーの実態確認）・**#16 vs #17**（スパイク由来の補助ペア・別プロンプト）。README に見どころ追記済み。→ **✅ユーザー目視確認 OK（2026-07-06・「参照動画の動きを反映した生成」を確認・受容）**。
2. **push／main マージ**: ユーザー承認待ち（branch `feature/ic-lora-strength`）。
3. 前回からの持ち越し: GUI 実機確認 3 点（黄トースト／クリップ n/N／クリップ別プロンプト）＝ユーザーが後日実施。
4. negative／CFG／pipeline は worker 未配線＝GUI 露出禁止（継続）。

## 29. ★画風／キャラクター LoRA 対応 S0「D スパイク」＝キー互換 GO/NO-GO 判定＝**GO**（2026-07-06・branch `feature/style-lora`・両トラック合格＋ユーザー目視確認済み）

> **正本＝本節。** 画風／キャラクター LoRA 対応（[`STYLE_LORA_WORKORDER.md`](STYLE_LORA_WORKORDER.md)）の最初のゲート「D. キー互換スパイク」の検証記録。CivitAI（外部配布サイト）が配っている実物の LoRA 2 本が、既存の IC-LoRA 適用機構でそのまま使えるか（追加のキーマップなしで読み込めるか）の GO/NO-GO 判定。
> スパイクは**コード・config を一切変更せずに** GO 判定へ到達（一時パッチなし）。本セッションはコード・設定を触っていない。

本機: i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`。

### 29.1 対象 LoRA と素性（2 本とも sd-scripts 系学習）

| ファイル | サイズ | rank | `ss_network_alpha` | 用途 | 発動方法 |
|---|---|---|---|---|---|
| `models/loras/Pixar_Toon.safetensors` | 336MB | 32 | 16 | 画風（Pixar 風トゥーン） | トリガーワード `P1x4r` |
| `models/loras/LTX-2.3-Henshin.safetensors` | 816MB | 64 | 32 | 変身エフェクト | プロンプトテンプレート方式 |

- 両者とも sd-scripts 系（`networks.lora_ltx2`）で学習。全キーが `diffusion_model.` プレフィックス ＋ `lora_A.weight`／`lora_B.weight`・データ型 BF16・`.alpha` テンソルは**持たず**（alpha はメタデータ `ss_network_alpha` のみ）。
- alpha/rank は両方とも **0.5**（Pixar=16/32・Henshin=32/64）。

### 29.2 Track A＝attach 検証（生成なし・GPU 不使用）

LoRA の各テンソルが本番モデルのどのモジュールに対応付くか（attach）を、実重みなしで確かめた。

- **方法**: `torch.device("meta")` 下で `LTXModelConfigurator.from_config()`（GGUF メタデータから本番と同一 config）を使い、モジュール木の構造だけを構築。そこへ公式 `attach_ic_loras()` を直接実行。attach 処理はモジュール名と Linear 層の in/out features しか参照しないため、実重みをロードせずに解決可否を判定できるのが根拠。
- **結果**: Pixar＝**576/576 ペア**解決・Henshin＝**1152/1152 ペア**解決。shape mismatch **0 件**・「0 マッチ WARN」なし。Pixar の `to_gate_logits`／`ff.net.0.proj`／`ff.net.2`、Henshin の `audio_attn1/2`・`audio_to_video_attn`・`video_to_audio_attn` を含む全モジュールが `named_modules()` に実在して解決（モデル＝AudioVideo・48 層・gated attention）。
- **含意**: 上流 `LTXV_LORA_COMFY_RENAMING_MAP`（`diffusion_model.` プレフィックス剥がし）＋ `lora_A`/`lora_B` ペアリングという既存経路のまま、**追加キーマップは不要**。

### 29.3 Track B＝実生成検証（API バイパス・runner 直接駆動・pose なしの純 T2V）

- **方法**: `services.ltx_runner.LTXRunner` を直接駆動する使い捨てドライバ（scratchpad・**コミットせず**）。`lora_paths=[(絶対パス, strength, "none")]`／`reference_video_path=None` の純 T2V。現行本番設定（distilled 8step／CFG1.0・`keep_resident=0`・comp=1）・seed=12345 固定・1 回ロードで 6 本を直列生成。
- **結果表**（一次ソース＝`ltx_worker.log` の `peak_vram_mb`）:

| ラン | 解像度／尺 | LoRA（strength） | 生成時間 | `peak_vram_mb` | attach 数 |
|---|---|---|---|---|---|
| smoke_pixar-1.0 | 512×320／49f | Pixar（1.0） | 115.2s | 8440 | 576 |
| baseline | 1280×768／121f | なし | 179.7s | 9532 | 0 |
| pixar-1.0 | 1280×768／121f | Pixar（1.0） | 195.1s | 9518 | 576 |
| pixar-0.5 | 1280×768／121f | Pixar（0.5） | 191.5s | 9532 | 576 |
| henshin-1.0 | 1280×768／121f | Henshin（1.0） | 192.3s | 9532 | 1152 |
| henshin-0.5 | 1280×768／121f | Henshin（0.5） | 193.6s | 9539 | 1152 |

- **VRAM**: LoRA のオーバーヘッドは実質ゼロ（baseline 9532MB と同水準・rank64／1152 バッファでも増分なし）。512×320 スモークは 8440MB ＝ LoRA なし基準値と一致。OOM／WDDM 共有溢れなし。
- **目視**: baseline＝実写調 ／ pixar-1.0＝明確なフル 3D CGI トゥーン調 ／ pixar-0.5＝自然寄りの 3D アニメ調（同一 seed で段階差＝strength 可変の実効を確認）／ henshin-1.0＝実写調のまま腰周りに金色の光の渦（変身 VFX）が発現。**ユーザー本人が代表フレーム 4 枚を目視し「完璧」と受容（2026-07-06）**。※動画本体の最終目視ゲートは実装完了後に別途実施。
- **attach ログ引用**: 「IC-LoRA Pixar_Toon.safetensors: 576 Linear(s) attached for forward-time apply (strength=1.000)」等・「0 マッチ WARN」なし。

### 29.4 alpha スケールの裏付けと設計決定（ユーザー承認済み）

- 両 LoRA は `.alpha` テンソルを持たず、メタデータ `ss_network_alpha` のみ（両方 alpha/rank＝0.5）。既存の delta 式（`strength × B@A`）は alpha を読まない。
- **ユーザー決定**: alpha/rank を LoRA 名前解決層（`services/lora_registry.py` の `resolve()`）で strength に自動乗算する（Forge 系互換・weight 1.0＝学習が想定したとおりの効き）。不可触の重みパッチ機構は無改造。既存 IC-LoRA は alpha メタなし→係数 1.0→既存挙動は不変。
- **併せて確定した設計判断**: GUI は分離する（制御 LoRA＝従来のアダプタドロップダウン ／ 画風 LoRA＝プロンプト内 `<lora:名前:weight>` コマンド ＋ 新「Style LoRA」タブ）。

### 29.5 結論と次アクション

- **結論＝GO**（Track A／Track B 両方合格 ＋ ユーザー目視受容・2026-07-06）。
- ブランチ: `feature/style-lora`。LoRA 実物は `models/loras/` に配置（git 管理外）。
- **次**: S1（ディレクトリスキャン＋kind 判定＋alpha 畳み込み＋all-or-nothing 緩和＋`GET /loras` 等の加算 API）→ S2（GUI）。

## 30. ★画風／キャラクター LoRA 対応 S1（バックエンド）＋S2（GUI）実装と回帰ゲート＝全 PASS（2026-07-06・branch `feature/style-lora`・**目視ゲート✅クローズ・main マージ＆push 済**）

> **正本＝本節。** §29 の S0「D スパイク」＝GO を受け、[`STYLE_LORA_WORKORDER.md`](STYLE_LORA_WORKORDER.md) の要件3点（ディレクトリ配置＋リロード／プロンプト内 `<lora:名前:weight>` コマンド／GUI「Style LoRA」サムネイルタブ）を S1＝バックエンド、S2＝GUI の2スライスで実装した回。凍結 API の加算的変更（新設エンドポイントと optional 緩和・**トークン無し／フィールド省略時は従来と byte 同一**の定型ゲート）を守り、不可触の LoRA 重みパッチ機構には触れていない。
> commit `bcdde05`（S1＝バックエンド）→`591d4be`（S2＝GUI）＋docs。**push／main マージ＝✅済（merge `f39f22f`・ユーザー承認 2026-07-06）。最終目視ゲート＝✅クローズ（30.5 参照・スパイク動画 6 本をユーザー受容）。**

本機: i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`。

### 30.1 S1＝バックエンド（commit `bcdde05`）

- **ディレクトリスキャン＋登録制のマージ（`services/lora_registry.py` 拡張）**: `rescan()` を新設。`config.yaml` の登録（authoritative）と `config.model.lora_dir="./models/loras"` のスキャン結果をマージする。名前が衝突したら config 側を勝たせ、スキャン側は親ディレクトリ名を付けて退避する。
- **kind 判定**: safetensors メタに `reference_downscale_factor` があるか、または preprocess が `none` 以外なら **control**（制御 LoRA＝参照動画が要る）、それ以外を **style**（画風／キャラ）と判定する。
- **alpha 自動畳み込み**: scale＝alpha/rank（メタ `ss_network_alpha`／`ss_network_dim` から算出・メタ無しは 1.0）を `resolve()` が返す strength に自動乗算する（§29.4 のユーザー承認済み設計・不可触機構の外側で完結）。
- **all-or-nothing 緩和（`api/models.py`）**: 「loras を指定したら `reference_video_id` 必須」という all-or-nothing 検証を撤廃（逆向き＝`reference_video_id`／strength 系を指定したら loras 必須、は維持）。かわりに `api/generate.py` の endpoint 層で kind を判定し、**kind＝control かつ参照動画なし** のときに **422 `LORA_REQUIRES_REFERENCE`** を返す。
- **新設 API（`api/loras.py`）**: `GET /loras`（毎回 rescan して一覧）／`POST /loras/reload`／`GET /loras/{name}/thumbnail`（`<stem>.png` を併置・無ければ 404）。
- **既存 IC-LoRA への無影響を事前確認**: 既存の IC-LoRA 実ファイル（upscaler x2／x4・union-control）は `ss_network_alpha` を持たないため scale＝1.0＝挙動不変。alpha 畳み込みが既存経路に影響しないことを事前に確認済み。

### 30.2 S1 回帰ゲート（全 PASS）

- **pytest**: **391 passed / 1 skipped**（§28 の基準 365+1 → +26＝`test_lora_registry` 19 本＋`test_loras_endpoint` 6 本＋意図的なテスト更新分 +1）。
- **SHA byte 一致 3/3**（alpha 畳み込みが既存経路に無影響であることの実証を含む）:

| 経路 | SHA256 |
|---|---|
| T2V 基準 | `23844b4eebd107ccba8c5534eb65bab86575cca0b9050cb6c7e680a4506bb7bf` |
| 最小 I2V | `a511eda431cf0d0942cee97fa130f45e55fc3236833cbf9ea743ea7f4715c217` |
| IC-LoRA 付き | `735a6de97d2deb56a781c66307849924a8ac25b51f0591fbddab07a78875e272` |

- **peak_vram_mb**（一次ソース＝`ltx_worker.log`）: T2V＝**8440**（基準一致）／I2V＝**9525**／IC-LoRA＝**9525**。
- **新 API スモーク**: `GET /loras`＝実物 2 本（style・スキャン由来）＋config 3 種（control・登録由来）を返す。`POST /loras/reload`＝`{total:5, styles:2, controls:3}`。thumbnail＝png 無しで 404。style 単独 generate＝202→completed（worker 実効 strength **0.5**＝1.0×α）。control 単独＝**422**。

### 30.3 S2＝GUI（commit `591d4be`）

- **プロンプト内コマンドのパース（`gradio_ui/handlers.py`）**: `<lora:名前:weight>` を解釈（`re.IGNORECASE`・weight 省略時 1.0・0.05〜2.0 に clamp して超過時は警告）。名前は `GET /loras` の名前集合に大文字小文字を無視して解決し、**未知の名前は警告して送信を中止**（アップロード前配置を促す設計＝副作用ゼロ）。同名は後勝ちでマージ、トークンはプロンプトから除去して空白を畳む。**トークンが無いときは `GET /loras` を発行せず、payload は従来と byte 同一。**
- **「Style LoRA」タブ（`gradio_ui/ui.py`）**: 5 番目のタブに `gr.Gallery`（kind＝style のみ表示・control 3 種は除外）を置き、サムネイルは `GET /loras/{name}/thumbnail`（無しはプレースホルダ）。Reload ボタン＝`POST /loras/reload`。サムネイルをクリックすると Generate タブの prompt 末尾に `<lora:名前:1.0>` を追記する。i18n EN／JA 11 キー。
- **pytest**: **418 passed / 1 skipped**（S1 の 391+1 → +27・赤化ゼロ）。

### 30.4 S2 E2E（実サーバー＋実 GUI ハンドラ経路・全 PASS）

- **入力**: プロンプト「…cheering crowd `<lora:pixar_toon:0.8>` P1x4r pixar style character」／512×320・49f・seed 12345。
- **送信 payload**: prompt からトークン除去済み・`loras=[{"name":"Pixar_Toon","strength":0.8}]`（小文字入力 `pixar_toon` → 正準名 `Pixar_Toon` に解決）。
- **`ltx_worker.log`**: 「IC-LoRA Pixar_Toon.safetensors: 576 Linear(s) attached for forward-time apply (strength=0.400)」（0.8×0.5＝0.4＝GUI 指定 weight × alpha 畳み込み）。
- **結果**: completed 112.84s・**peak_vram_mb 8440**（基準一致）。gallery＝style 2 件のみ（control 3 種は除外）。

### 30.5 目視ゲート＝✅クローズ（2026-07-06）

1. **最終目視ゲート（ユーザー）＝✅OK（2026-07-06）**: スパイク動画 6 本（`dspike_out` の baseline／pixar 1.0・0.5／henshin 1.0・0.5／smoke）をユーザーが目視し「問題ない・完璧」と受容。GUI 実機操作（Style LoRA タブのサムネイル選択→プロンプト欄へのタグ挿入）は §31 のプロンプト欄一本化 GUI 確認で併せて OK（クリック挿入先は一本化後の上部下書き欄）。
2. **push／main マージ＝✅済**（ユーザー承認 2026-07-06・merge `f39f22f`）。
3. 前回からの持ち越し: GUI 実機確認 3 点（黄トースト／クリップ n/N／クリップ別プロンプト）＝**この承認とは別・ユーザーが後日実施（継続）**。
4. negative／CFG は worker 未配線＝機能的に無効（CFG＝1 固定）。**値の送出自体は禁止しない**（ネガティブ欄は現状 payload に送信）。§31 で GUI 上は `interactive=False` のグレーアウト表示（将来の非蒸留対応モック）へ。

## 31. ★GUI プロンプト欄の一本化 ＋ ネガティブ欄グレーアウト 実装＝客観ゲート PASS＋目視ゲート✅クローズ（2026-07-06・branch `feature/prompt-unification`・commit `fd35877`・**main マージ＆push 済**）

> **正本＝本節。** メインのプロンプト入力欄が Generate（`prompt`）と Clip Chain（`chain_prompt`＝共通ベース）の2箇所に分かれていたのを、タブ外・タブ群の上に置く**単一「下書き」欄**へ統合した回。Forge Neo 風（LoRA 選択→上部欄にタグが載る）。入口＝[`PROMPT_UNIFICATION_WORKORDER.md`](PROMPT_UNIFICATION_WORKORDER.md)。**GUI のみ（API／engine／services 不可触）**。
> commit `fd35877`（変更4ファイル）。**push／main マージ＝✅済（ユーザー承認 2026-07-06）。最終目視／実機ゲート＝✅クローズ（31.4 参照・実機4点をユーザー受容）。**

本機: i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`。（**本節は GUI 改修のため GPU 計測・SHA 新規測定は該当なし＝既存 SHA 不変。**）

### 31.1 方式（設計判断＝「下書き欄を唯一の本物にする」）

- 「隠し欄＋クリック時にコピー」ではなく、**下書き欄そのものを唯一の本物**にして両生成ハンドラへ**直結**する方式を採った。
- 根拠: Gradio は `inputs=` に並べたコンポーネントの値をクリック時に**atomic に受け取る**ため、コピー順序の取りこぼしや残留プロンプトが**構造的に発生しない**。ユーザーの当初案（下書き欄）を本セッションでブラッシュアップして合意した設計。

### 31.2 実装（commit `fd35877`・変更4ファイル）

- **`gradio_ui/ui.py`**: 上部共通バー（status／refresh／load／unload の Row）直後・`with gr.Tabs()` の直前に**単一の3行 Textbox（下書き欄）**を新設。旧 `prompt`／`chain_prompt` は削除。`generate_btn.click`・`chain_generate_btn.click`・`style_gallery.select` を下書き欄へ直結（`handlers.py` のシグネチャは変更なし）。
- **`gradio_ui/handlers.py`**: `make_chain_handler` で共通プロンプト中の `<lora:...>` を除去し `gr.Warning`（連結は LoRA 未対応＝研究課題）を出す。**トークンが無いときは `send_prompt=prompt` として payload byte 同一を維持**（`_LORA_TOKEN_RE` を再利用・パース仕様は不変）。各クリップの**個別プロンプトは除去対象外**。
- **ネガティブ欄**（`negative`／`chain_negative`）を `interactive=False`＋info 注記（蒸留 CFG＝1 で無効）へ。値・送信ペイロード・キーは不変＝回帰 SHA 不変。欄は**将来の非蒸留対応モックとして残置**（削除しない）。
- **`gradio_ui/i18n.py`**: `lbl_prompt` 文言更新・`info_negative`／`chain_lora_ignored` 新設（EN／JA）・`lbl_prompt_shared`／`ph_prompt2` の対を削除（EN／JA 計4）。

### 31.3 客観ゲート（PASS）

- **pytest**: **422 passed / 1 skipped**（§30 の基準 418+1 → 新規4件・赤化ゼロ・app venv・mock transport）。
- **`build_ui()` スモーク**: Blocks グラフが正常に組み上がることを確認。下書き欄は `lbl_prompt` が1回だけ登録され、旧 `lbl_prompt_shared` は消滅していることを確認。
- **payload byte 同一**: トークン無しプロンプトで送信ペイロードが従来と byte 同一であることを新規テストで担保（Generate 側の既存担保＋連結側の担保を追加）。

### 31.4 目視／実機ゲート＝✅クローズ（2026-07-06）

1. **最終目視／実機ゲート（ユーザー）＝✅OK（2026-07-06）**: ①上部に下書き欄が1つ・全タブで表示 ②Style LoRA タブでサムネイル選択→上部欄にタグが載る ③連結で LoRA タグ除去の警告 ④ネガティブ欄グレーアウト、をユーザーが実機確認し「問題ない」と受容。
2. **push／main マージ＝✅済**（ユーザー承認 2026-07-06）。
3. **スコープ外／研究課題**: 連結タブで LoRA を実際に効かせるバックエンド改修（`GenerateChainRequest.loras` 加算＋worker 配線）＝将来の研究課題。**→ §32 で実施済み。**

---

## 32. ★A2V＋LoRA併用の解禁＝`GenerateChainRequest.loras` 追加＋stale LoRA持ち越し修正込み＝客観ゲート全PASS・独立レビュー must-fix/should-fix ゼロ・**目視ゲート✅クローズ＝ユーザー受容**（2026-07-11）

> **正本＝本節。** NEXT_SESSION_HANDOFF.md 冒頭ブロックの引き継ぎ課題「A2V＋LoRA併用の解禁」（§31.4 の3で「将来の研究課題」としていたものが今回昇格）を実装した回。A2V（音声から動画を生成する機能）とスタイル／キャラクター LoRA（少量データで画風・キャラクターを追加学習した軽量アダプタ）を同時に使えるようにした。設計論点2点はユーザー決定済み（32.1）。**目視ゲート＝✅クローズ（32.5参照・原因究明の経緯＝トリガーワード欠落）。** ログ拡充・GUIバグ2件修正・JobStore CAS化は§33を参照。**コミット未実施（作業ツリーの変更のまま・push はユーザー判断待ち）。**

本機: i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`。（**32.1〜32.4は配線・スキーマ・GUI の客観ゲート。GPU 実生成によるLoRA効果の目視確認は32.5で実施＝✅クローズ。**）

### 32.1 設計論点の決着（ユーザー決定）

前回（§31.4 の3・NEXT_SESSION_HANDOFF.md）で未決だった2点を本セッションでユーザーが決定した:

1. **LoRA 強度はチェーン全体で共通**。クリップ（連結される動画の1区間）ごとに強度を変える機能は v1 では見送り（`GenerateChainRequest.loras` はリクエスト全体で1本のリストのみ）。
2. **reference 動画付き IC-LoRA（参照動画から輪郭線 canny・骨格 pose などを読み取って条件付けする「control系」アダプタ）はチェーン非対応**。チェーンは `reference_video_id` を持たない構造のため、control系アダプタが指定されたら API 側でジョブ予約前に **422**（新設エラーコード `LORA_CONTROL_UNSUPPORTED_IN_CHAIN`）で拒否する。
3. GUI は Generate タブ（単発生成・A2V の窓口）と Clip Chain タブ（クリップ連結）の**両方**で解禁する。

### 32.2 実装（加算的変更5箇所＋GUI）

既存の凍結 API 契約への加算的変更（省略時は従来と byte 同一）:

- **`api/models.py`**: `GenerateChainRequest` に `loras: list[LoraSpec] = Field(default_factory=list)` を追加（`GenerateRequest.loras` と同じ型・同じバリデーション）。
- **`api/generate_chain.py`**: ジョブ予約前に指定 LoRA 名をすべて解決（未知/欠落は404）し、control系アダプタが含まれていれば `lora_control_unsupported_in_chain()`（422）で拒否。`api/errors.py` に同エラーファクトリを新設。
- **`services/pipeline_manager.py`**: `run_chain_job` 内で `lora_registry.resolve(...)` を呼び `lora_paths` を解決（単発 `run_generation` の既存箇所を踏襲）。
- **`services/ltx_runner.py`**: `generate_chain`（`LTXRunner`／`_MockBackend`／`_RealBackend` 全て）に `lora_paths` 引数を追加。`_RealBackend` は非空のときのみ worker ペイロードへ `loras` ブロックを追加（空なら従来と byte 同一）。
- **`engine/worker.py`**: `_do_generate_chain` で `msg.get("loras", [])` を明示的に解析し `ic_loras` として `run_chain` へ渡す（キー欠落時も空リストとして扱い、必ず解析を通す）。
- **`engine/pipeline/chain_pipeline.py run_chain`（`engine/pipeline/fast_video_pipeline.py generate_chain` 経由）**: `ic_loras` 引数を追加し、**`ledger.transformer()` を呼び出す前に必ず `pipe._set_ic_job(list(ic_loras or []), None, 1.0)` を呼んで明示的にセット/クリア**する。**stale（前のジョブの LoRA 適用状態が次のジョブに残留すること）防止の修正を兼ねる**: 従来チェーン側はこの呼び出しが一切無く、直前の単発 `generate()` 由来の LoRA がそのままチェーンの denoise に持ち越される恐れがあった。今回、LoRA 無指定のチェーンでも空リストで明示的にクリアするようにしたことで、この persistence 問題を解消した。
- **GUI（`gradio_ui/handlers.py`）**: A2V×LoRA の相互排他プリチェックを撤去。**reference 動画付き control アダプタ＋A2V の組み合わせのみ**、新文言 `a2v_control_lora_unsupported` で事前拒否する。単発 generate 経路の LoRA 合成ロジックを `_combine_generate_loras()` として関数化し、A2V 経路（`use_adapter=False` で呼ぶ）でも同じロジックを共用。Clip Chain タブは、これまでプロンプト内 `<lora:...>` トークンを除去して警告（`chain_lora_ignored`）していたのを撤去し、トークンを解決して `loras` として送出するように変更。
- **`gradio_ui/i18n.py`**: EN／JA 両方を更新。`gen_a2v_conflict_lora`・`chain_lora_ignored` を削除、`gen_a2v_note` を新仕様に書き換え、`a2v_control_lora_unsupported`・`apierr_LORA_CONTROL_UNSUPPORTED_IN_CHAIN` を新設。

### 32.3 客観ゲート（PASS）

- **pytest**: **500 passed / 1 skipped**（本セッション開始時点の基準 485+1 → +15・退行ゼロ。§31 時点の基準 422+1 からは、間に挟まる「2026-07-11 Gradio WebGUI 大規模改修」セッション分＝+63 を経て 485+1 に到達済み〔同セッションの詳細は VERIFICATION_LOG に節を持たず `NEXT_SESSION_HANDOFF.md` のみに記録〕。うち本セッションの新規分＝新規ファイル `tests/test_chain_lora.py` 13件＋既存 `tests/test_gradio_handlers.py` の更新・新規分）。
- **新規 `tests/test_chain_lora.py`（13件）のカバレッジ**: (a) スキーマ＝`loras` 既定空・spec 受理・パス様の名前は拒否　(b) エンドポイント＝チェーンへの control アダプタは422／未知名は404／style アダプタは完走　(c) 配線＝mock backend の e2e でリクエストダンプに `loras` が載ること、実バックエンドでは worker ペイロードに `loras` ブロックが追加される（空なら追加されない）こと　(d) stale クリア＝`chain_pipeline.run_chain` が transformer ビルド前に `pipe._set_ic_job` を（このチェーンの loras か、クリア用の空リストで）呼ぶこと。**mock backend には実重みが無いため forward 時の実際の LoRA 効果そのものはこのテストの対象外**＝(d) は配線を GPU 無しで担保するのみで、実際の重み適用効果は §32.5 のユーザー立ち会いGPU確認に委ねる。
- **`tests/test_gradio_handlers.py` 更新**: A2V＋LoRA トークン併用時にチェーンペイロードへ `loras` が載ること／未知トークンでゼロ API 呼び出しのまま中止すること／control アダプタ＋A2V の新拒否文言／Clip Chain 共有プロンプトの `<lora:...>` トークンが除去ではなく解決・送出されることを確認。

### 32.4 独立レビュー（PASS・must-fix/should-fix ゼロ）

- 独立レビューを実施。**must-fix（必須修正）・should-fix（推奨修正）はゼロ件**。
- **nit（軽微な指摘）1件**: control系アダプタの名前をプロンプトに `<lora:...>` として手打ちした場合、音声または画像を1回アップロード消費したあとサーバー側の422（`LORA_CONTROL_UNSUPPORTED_IN_CHAIN`）で拒否される（拒否メッセージ自体は正しくローカライズされる）。**Style ギャラリーのサムネイルをクリックする通常の操作では発生しない**（クリック挿入は style のみを対象にした一覧から選ぶため）。既知の軽微事項として記録し、対応は見送り。

### 32.5 目視ゲート＝✅クローズ（ユーザー受容・2026-07-11）

客観ゲート（32.3・32.4）はすべて PASS していたが、ユーザーが GPU 実機で32.1〜32.4の4点（チェーン全体適用・stale 非残留・Clip Chain 適用・省略時回帰なし）を確認する過程で、当初「LoRA が効いていないのでは」という報告があった。原因を2段階で調査し、以下の経緯で解明・クローズした。

1. **調査(a)＝ログ面の誤解**: コンソールにはそもそも LoRA 適用ログが出力されない仕様だった（後述§33.1でログ拡充を実施）。物証として `logs/ltx_worker.log` の「N Linear(s) attached」行と `outputs/<job_id>/metadata.json` を確認したところ、**全ジョブで LoRA は当初から正常に適用されていた**ことが判明した。
2. **調査(b)＝本質的な原因＝トリガーワード（LoRA を発動させるための合言葉となる特殊な単語）の欠落**: 使用していた Pixar_Toon LoRA は「P1x4r pixar style character」等のトリガー語をプロンプトに含めないと画風が変化しない設計であり、過去にスパイクテストで成功していた事例は全てトリガー語入りだった。報告のあった生成ではトリガー語が入っていなかったため、LoRA 自体は適用されていても画風変化が視覚的に確認できなかった。
3. 併せて、alpha／rank 正規化（Pixar_Toon は alpha16／dim32＝×0.5 のため実効強度が指定値の半分になる仕様）と seed の影響も、体感的な効き方のばらつき要因として確認した。
4. **トリガー語を入れた実機テストでユーザーが成功を確認**し、32.1〜32.4 の4点の確認事項も含めて A2V＋LoRA 併用を受容。**目視ゲート✅クローズ**。

なお、この調査を機にオーナーから追加要望が3件出ており（ログ拡充・GUI バグ2件の修正）、これらは§33で別途実装・クローズ済み。git commit／push は本節・§33とも時点では未実施（作業ツリーの変更のまま・ユーザー判断待ち）。

## 33. ★ジョブ起動ログ拡充＋GUIバグ2件修正（ref_video 無効化・queued 詰まり解消）＋JobStore CAS化＝客観ゲート全PASS・独立レビュー2巡（must-fix残存なし）（2026-07-11）

> **正本＝本節。** §32.5 の目視ゲート調査（トリガーワード欠落の解明）を機に、オーナーから追加要望が出た3件を実装した回: 「ジョブ開始時のログに実 seed／ベース GGUF／LoRA／プロンプトを出す」「adapter=None のまま参照動画をアップロードするとサイレント無視される」「queued で詰まると無言のままハングしサーバー再起動が必要になる」。バックエンド改修のため、計画→実装→独立レビュー→指摘への追修正→再検証、のフルサイクルを実施。**コミット未実施（作業ツリーの変更のまま・push はユーザー判断待ち）。**

本機: 同上（i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`）。

### 33.1 ジョブ開始ログの拡充（オーナー要望）

- コンソール（`ltx.pipeline` ロガー）に、ジョブ開始時点で以下を出力するようにした。
  - 実際に使われる seed（`-1` 指定＝ランダム決定の場合でも、解決後の実値を表示。**表示値＝使用値の一致を保証**）。
  - ベースとなる transformer GGUF のファイル名。
  - 適用 LoRA 名（要求 strength／実効 strength の両方を併記。LoRA 無しの場合は `loras=none` と明示）。
  - プロンプト先頭200字（改行はエスケープして1行に収める）。
- 目的＝§32.5 で判明した「LoRA 適用状況がログから読み取れず、効果の有無を切り分けにくい」問題への恒久対応。今後同種の問い合わせがあっても、ログのみで seed・LoRA 適用状況を即座に確認できるようにした。
- pytest 件数の増分（503→の一部）はこの節の新規／更新テストを含む。

### 33.2 GUIバグ2件の修正（計画→実装→独立レビュー→追修正→再検証まで完了）

1. **adapter=None＋参照動画のサイレント無視**: LoRA アダプタ未選択（None）のまま参照動画をアップロードしても警告なく無視される不具合。`gradio_ui/ui.py` で参照動画（ref_video）欄を初期状態で非活性化し、adapter 選択の `change` イベントで interactive を連動トグルするように修正。None へ切り替えた際はアップロード済みの動画も自動クリアする。
2. **queued 詰まりで無言ハング＆サーバー再起動が必須**: ジョブが queued 状態のまま進まなくなっても、GUI は何も表示せずユーザーはサーバー再起動以外に手段が無かった。以下3点で対応した。
   - ポーリング表示に queued 状態を追加し、30秒を超えて解消しない場合は「Jobs タブからキャンセルしてください」と誘導する文言を表示（i18n 新設キー `msg_queued`／`msg_queued_stuck`、EN／JA）。
   - `DELETE /jobs/{id}` が queued 状態のジョブに対しても即座に `cancelled` へ遷移させ、詰まりを解放できるようにした（従来は running 以降のみ対応）。
   - `JobStore` に `_lock` 配下の CAS（compare-and-swap）ヘルパー（`start_job`／`cancel_if_queued`）を新設し、`run_job`／`run_chain_job` の queued→running 昇格とキャンセル操作を相互排他化。独立レビュー指摘 S1（昇格とキャンセルの間の TOCTOU＝競合状態）を解消。
   - 実バックエンド使用時はジョブ実行を専用 daemon スレッドへ切り出し（mock バックエンドは従来通り BackgroundTasks のまま）。`Thread.start()` 失敗時はジョブを failed へ遷移させガードを解放（独立レビュー指摘 S2）。AnyIO のスレッドリミッタを200へ拡大。

### 33.3 客観ゲート（PASS）

- **pytest**: **511 passed / 1 skipped**（§32時点の基準 500+1 → +11・退行ゼロ）。

### 33.4 独立レビュー（2巡・must-fix残存なし）

- **1巡目**: should-fix（推奨修正）2件を指摘（S1＝queued 昇格とキャンセル操作の TOCTOU、S2＝`Thread.start()` 失敗時にジョブが queued のまま取り残される）。
- **追修正**: 33.2 記載の CAS ヘルパー導入（S1対応）と `Thread.start()` 失敗時の failed フォールバック（S2対応）を実装。
- **2巡目（再検証）**: must-fix／should-fix とも残存なし。クローズ。

3f29776としてコミット・push済み（2026-07-11）。

---

## 34. ★reference動画付きIC-LoRAのchain対応＝α版（clips=1限定）で解禁＝客観ゲート全PASS・独立レビュー must-fix ゼロ・mock E2E PASS・**GPU実機目視ゲート✅完了・ユーザー受容（2026-07-12）**（2026-07-11）

> **正本＝本節。** NEXT_SESSION_HANDOFF.md 冒頭ブロックの引き継ぎ課題「reference 付き IC-LoRA のチェーン対応」（§32.1 の2で「チェーン非対応・422で拒否」としていたスコープを、オーナー決定によりα版として昇格）を実装した回。**`clips` がちょうど1つのチェーン（A2V を含む）に限り**、reference動画付き control 系 IC-LoRA（参照動画から輪郭線 canny・骨格 pose 等を読み取って条件付けするアダプタ）を許可する。`clips` が2つ以上のチェーンは従来どおり 422（`LORA_CONTROL_UNSUPPORTED_IN_CHAIN`）で拒否を維持。**客観ゲート（pytest・独立レビュー・mock E2E）はすべて PASS。GPU 実機の目視ゲートは2026-07-12にオーナー立ち会いのもと実施し完了・受容済み（詳細＝34.7）。7c50c32としてコミット・push済み（2026-07-12）。**

本機: 同上（i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`）。

### 34.1 要件確定（オーナー決定・§32.1 の2〔NEXT_SESSION_HANDOFF.md §32歴史ブロックの残課題2〕の課題を実装）

- **α版スコープ**: `clips` がちょうど1つのチェーンに限り、reference 動画付き IC-LoRA（control系）を許可する。`clips` が2つ以上のチェーンは従来どおり 422（`LORA_CONTROL_UNSUPPORTED_IN_CHAIN`）。
- **根拠**: 主眼のユースケースは A2V（音声から動画を生成する機能）との併用であり、A2V は内部的に「1クリップのチェーン」として実行されるため、clips=1 限定で目的を満たせる。
- **クリップ毎の参照動画入力**（Clip 2 に専用の参照動画を与える方式）は研究課題へ格下げ（対象外）。

### 34.2 実装（加算的変更）

既存の凍結 API 契約への加算的変更（省略時は従来と byte 同一）:

- **`api/models.py`**: `GenerateChainRequest`（280行目〜）に `reference_video_id: str | None = None`／`conditioning_attention_strength: float | None = Field(None, ge=0.0, le=1.0)`／`reference_video_strength: float | None = Field(None, ge=0.0, le=1.0)` を追加（355-357行目。`GenerateRequest` の同名フィールドと同型・同バリデーション）。`validate_chain_constraints`（360行目〜）に検証を追加: クリップ数下限判定（397-406行目）に `self.reference_video_id is None` の条件を足し、reference 指定時は clips=1 を許容／`reference_video_id` かつ `loras` 空 → 422（479-482行目）／`reference_video_id` かつ `len(clips) != 1` → 422「requires exactly 1 clip in v1」（484-485行目）／`conditioning_attention_strength`・`reference_video_strength` 単独指定かつ `loras` 空 → 422（487-495行目）／`reference_video_id` と `source_video` の併用 → 422（502-507行目、V2V の凍結ヘッドとの競合を理由に排他）。
- **`api/generate_chain.py`**: `generate_chain()`（37行目〜）内、ジョブ予約前に `reference_video_id` の実在確認＋÷128 検証（76-82行目・未知IDなら404、幅高が128の倍数でなければ422 `REFERENCE_RESOLUTION_INVALID`）。LoRA 解決ループ（97-112行目）で `preprocess_kinds` を集計するよう拡張し、control系アダプタが含まれる場合: `len(request.clips) != 1` なら従来どおり422 `LORA_CONTROL_UNSUPPORTED_IN_CHAIN`（108行目）、clips=1 かつ `reference_video_id is None` なら422 `LORA_REQUIRES_REFERENCE`（109-110行目、単発 `/generate` と同一の判定）。preprocess 種別が2種以上なら422 `LORA_PREPROCESS_CONFLICT`（111-112行目）。`api/errors.py::lora_control_unsupported_in_chain` の docstring／エラーメッセージを「chains carry no reference_video_id」から「chain with clips >= 2」に更新（送出条件が変わったため）。
- **`services/pipeline_manager.py`**: `run_chain_job`（550行目付近）で `chain.reference_video_id` を `video_upload_store.path_for()` で実パスへ解決し `reference_video_path` として `ltx_runner.generate_chain(...)` へ渡す（619行目）。無指定なら `None`（従来と byte 同一の経路）。
- **`services/ltx_runner.py`**: `LTXRunner.generate_chain`／`_MockBackend.generate_chain`／`_RealBackend.generate_chain` の3箇所に `reference_video_path: Path | None = None` 引数を追加。mock は無視するのみ。`_RealBackend`（1383-1400行目付近）は `reference_video_path` が非 None のときだけ worker ペイロードに `reference_video` ブロック（`{path, strength, preprocess}` ＋ `conditioning_attention_strength` 指定時のみ `attention_strength`）を追加。単発 `/generate` の `reference_payload` 組み立てをそのまま踏襲。
- **`engine/worker.py`**: 単発 `_do_generate` にあった reference 解析ロジック（旧: 関数内インライン）を `_resolve_ic_reference(ref, output_path) -> (ic_reference, attn_strength)`（263-307行目）として関数抽出。**単発側の挙動は逐語不変**（抽出のみ、ロジック変更なし）。`_do_generate`（310行目〜）と `_do_generate_chain`（451行目付近）の両方がこのヘルパーを呼ぶよう変更し、`_do_generate_chain` は結果を `run_chain(..., ic_reference=ic_reference, ic_attention_strength=ic_attn)`（483-484行目）へ渡す。
- **`engine/pipeline/chain_pipeline.py run_chain`**（`engine/pipeline/fast_video_pipeline.py generate_chain` 経由）: `ic_reference`／`ic_attention_strength` 引数を追加（418-419行目）。`ledger.transformer()` 構築前の `pipe._set_ic_job(...)` 呼び出し（577行目）に両値を渡す（無指定時は従来どおり `(None, 1.0)` の stale クリア）。**stage1のクリップ0（V2V 継続でない非分岐時のみ）**で `pipe._reference_conditioning_for_stage(...)`（651-667行目）を呼び、reference latent を conditioning に追加。半解像度（`height//2`/`width//2`）の cond_kwargs は単発側と同一のビルダーを再利用。**stage2 のタイルには一切注入しない**（単発生成と同じ「stage1のみ」の意味論）。
- **GUI（`gradio_ui/handlers.py`）**: A2V 送信前の「adapter が None 以外なら拒否」チェックを撤去（旧: 402行目付近）。`use_adapter` が真のときは既存の `_combine_generate_loras(use_adapter, adapter, adapter_strength, prompt_loras)`（569-570行目）で単発経路と同一ロジックにより `loras` を合成し、chain_payload へ `reference_video_id`（574行目）と、`control_adherence`／`reference_strength` が1.0未満のときのみ `conditioning_attention_strength`／`reference_video_strength`（576-581行目、単発経路 623-635行目と同じ「1.0未満のみ送出」規約）を追加。adapter 選択済み＋参照動画未指定は既存の `msg_ref_video_required` により従来どおり事前拒否（変更なし）。
- **`gradio_ui/i18n.py`**: デッドキーとなった `a2v_control_lora_unsupported` を EN/JA から削除。`gen_a2v_note` を「control 系アダプタも併用可」に書き換え。`apierr_LORA_CONTROL_UNSUPPORTED_IN_CHAIN` の文言を「チェーン（clips>=2）では使えない」旨に正確化（旧文言は条件を明示していなかった）。

### 34.3 客観ゲート（PASS）

- **pytest**: **526 passed / 0 failed / 1 skipped**（§33 時点の基準 511+1 → +15・退行ゼロ。既存 skip 1件のみ）。新規ファイル `tests/test_chain_reference.py`（14件）＝スキーマ検証（reference＋loras空／reference＋clips≠1／strength系＋loras空／reference＋source_video排他／未知ID404／÷128違反422）・エンドポイント（clips=1+control+reference→受理、clips=1+control+reference無し→`LORA_REQUIRES_REFERENCE`）・worker ペイロード疎通（reference_video ブロックの有無・attention_strength 省略時の挙動）・`run_chain` が `_set_ic_job` へ ic_reference/attention_strength を渡すこと・stage1のクリップ0にのみ reference conditioning が1回だけ注入されること・A2V+control+reference の mock 完走をカバー。既存 `tests/test_chain_lora.py` は control アダプタ拒否テストの docstring／条件を「clips>=2」に更新。`tests/test_gradio_handlers.py` は旧「A2V+adapter は無条件拒否」テストを「reference 未指定のみ拒否（既存 `msg_ref_video_required` 経由）」に置き換え、A2V+control+reference がチェーンペイロードへ正しく配線されることを確認する新規テストを追加。

### 34.4 独立レビュー（PASS・must-fix ゼロ）

- 実装に関与していないレビュアーによる独立レビューを実施。**must-fix（必須修正）はゼロ件**。

### 34.5 mock E2E（実サーバー起動・port 18901・隔離 config）

- 実サーバー（mock backend）を隔離環境（scratchpad 上の専用 config・別ポート 18901）で起動し、以下4パターンを確認:
  - `clips=1` ＋ control アダプタ ＋ `reference_video_id` ＋ `source_audio`（A2V）→ 202 → `completed`。`outputs/{job_id}/metadata.json` に `reference_video_id`／`loras` が記録されていることを確認。
  - `clips=2` ＋ control アダプタ → 422 `LORA_CONTROL_UNSUPPORTED_IN_CHAIN`。
  - `clips=1` ＋ control アダプタ ＋ `reference_video_id` 無し → 422 `LORA_REQUIRES_REFERENCE`。
  - 従来どおりの A2V（`loras` 無し）→ `completed`（回帰なし）。

### 34.6 既知事項（ドキュメント記録のみ・対応は見送り）

1. **precedence**: pydantic のスキーマ検証がエンドポイントより先に走るため、`clips>=2` ＋ `reference_video_id` の複合誤設定は、コード付きの `LORA_CONTROL_UNSUPPORTED_IN_CHAIN` ではなく汎用の `VALIDATION_ERROR`（422）になる。
2. `reference_video_id` ＋ スタイル系 LoRA のみ（control 系を含まない）はスキーマ上は受理されるが、worker 側の `_set_ic_job` で `RuntimeError`（ジョブ失敗）になる。単発 `/generate` の既存挙動をそのまま写像したものであり、GUI からは到達不能（アダプタ欄が control 系のみを選択肢に持つため）＝直接 API 経由のみ。早期422化は将来の改善余地として残す。
   - **→ ✅ 解消（2026-08-03）**: IC-LoRA Depth／Deblur 追加（§49）で新設した **`REFERENCE_REQUIRES_CONTROL_LORA`（422）**により、この組み合わせは**単発・チェーンの両方でリクエスト時点で拒否**されるようになった。「アップロードもジョブ開始も済んだあとにワーカー内で落ちる」経路は消えている。**解消は §49.7 の G2（mock 通し）で実機確認済み**（項目5）。ここで挙げていた「早期422化は将来の改善余地」は**クローズ**である。
3. `GET /jobs/{id}` の応答の `request` ブロックは chain 固有フィールド（`source_audio`／`loras`／`reference_video_id` 等）を載せない（`JobResponse.request` が `to_clip_request` 経由の `GenerateRequest` 形のため）。§30 の `loras` 追加時からの既存の表現上の制約であり、今回の退行ではない。正式な記録は `metadata.json` 側。
4. **既知の無害事象（GPU実機目視ゲート中に観測・2026-07-12）**: 2本目のジョブ完了直後にサーバーログへ `ERROR asyncio: ... ConnectionResetError [WinError 10054]`（`_ProactorBasePipeTransport._call_connection_lost`）が1回出力された。これは Windows の asyncio proactor がクライアント（ブラウザ）側の強制切断を後処理する際の既知の無害なノイズであり、直後に再接続し以降のジョブも正常完走した。機能影響なし・対応不要（オーナー判断で無視と決定）。

### 34.7 GPU実機目視ゲート＝✅完了・ユーザー受容（2026-07-12）

客観ゲート（34.3〜34.5）に続き、オーナー立ち会いのもと GPU 実機で目視ゲートを実施した。GUI の Generate タブから音声 wav＋参照動画＋control 系アダプタ（reference動画付き IC-LoRA）で生成を実行し、成功を確認した。

**実績**: 解像度 1280×768・201 frames・8 steps・所要時間 約300秒/本・ピーク VRAM 9241〜9537MB。内訳＝pose-control（strength=1）×3本＋canny-control（strength=1）×1本、いずれも `completed`。ジョブ開始ログに `loras=pose-control(strength=1)` 等の配線物証を確認（本節「ジョブ起動ログの拡充」機能がここでも有効に機能した）。

オーナーが実際の生成結果を受容＝**A2V＋reference付きIC-LoRA併用のGPU実機目視ゲートはクローズ**。

観測された既知の無害事象は34.6の4に追記済み（asyncio ConnectionResetError・機能影響なし・対応不要）。

7c50c32としてコミット・push済み（2026-07-12）。

---

## 35. GUIスピナー退行の原因特定と修正＝`demo.load` 3箇所へ`show_progress="hidden"`＝オーナー実機確認✅完了・**§35クローズ**（2026-07-12）

> **正本＝本節。** §34.7 の GPU 実機目視ゲートに着手する前提として、実機（実ブラウザ）で GUI を触ったところ、§34 の本題（reference動画付きIC-LoRAのchain対応）とは**別系統**の表示不具合が発覚した。目視ゲート自体を進める前に本節の修正を先行させる。

本機: 同上（i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`）。

### 35.1 症状（実機確認・2026-07-12）

- Generate タブの **Control adapter**（参照動画欄を含む一帯）が、ページ読み込み直後から永久スピナー状態のまま固まり、参照動画のアップロード欄が操作不能（クリックが素通りしない）。
- Settings タブでも同様の症状: `server_config_json`／`spill_table`、および Models セクションの4本のモデル管理ドロップダウン（transformer／text_encoder／video_vae／audio）が同じく操作不能。**影響箇所は計7箇所。**
- **3種のブラウザ（Chrome／Edge／Firefox、実機・別環境）で再現**。単一ブラウザ／拡張機能起因ではない。

### 35.2 調査結論

- **退行コミットの特定**: `1e18dea`（WebGUI 大規模改修）**単独**が原因。それ以前の `fd35877`（§31・プロンプト欄一本化）・`ab274ad` は実ブラウザで動作クリーンであることを確認した。オーナーが 2026-07-08 に確認した GUI 生成成功時の出力（`outputs/1a9aee65-...`）とも整合する＝それ以降に踏んだコミットのうち `1e18dea` のみが新規に該当。
- **機構の特定**: 複数の `demo.load` イベントが同一ページロードで同時発火する際、Gradio 6.19 のクライアント側 status-tracker（コンポーネントごとの進捗オーバーレイ管理）が pending のまま残ってしまう既知動作。pending のまま残った status-tracker のオーバーレイ（`z-index:30`／`pointer-events:auto`）が、折りたたまれたアコーディオンや非アクティブなタブの内側にある部品を覆い、クリックを奪う。
  - **因果の精密化**: 退行コミット直前の `ab274ad` にも、同じ3つの `demo.load` と閉じたアコーディオン内の部品は**既に存在していた**のに症状は出なかった（実ブラウザ確認済み）。`1e18dea` は4本目の `demo.load(None, js=_min_attr_js)` の追加と大量の部品追加（`gen_a2v_accordion`・Duration パネル・`chain_preset` 等）で構成を拡大しており、この**構成拡大によって、潜在していたクライアント側のレース条件が顕在化した**というのが正確な因果である。切り分け実験では js ロード単独の無効化では解消しなかったため、単一要素の追加ではなく構成拡大そのものが引き金と結論した。
  - バックエンド側は正常完走しておりサーバーエラーは無し。ブラウザの JS 例外もゼロ（コンソールクリーン）。
  - **除外した仮説**: `config_retry_timer`（`gr.Timer`）／`_min_attr_js`（`demo.load(None, js=...)`）／`CUSTOM_CSS` を個別に無効化してもこの症状に変化は無く、これらは主因ではないと判断した。

### 35.3 修正（`gradio_ui/ui.py`・変更3箇所）

以下3箇所の `demo.load(...)` に `show_progress="hidden"` を追加した。背景処理そのもの（fetch内容・outputs）は変更していない＝挙動は「進捗オーバーレイを出さない」点のみ変わる。

1. **`on_page_load`**（設定・アダプタ・spill_table 等の起動時フェッチ）。
2. **`refresh_model_dropdowns`（warn=False）**（Models セクションの起動時ドロップダウン更新）。
3. **`load_style_gallery`（warn=False）**（Style LoRA タブの起動時ギャラリー読み込み）。

なお 3 の `load_style_gallery` への付与は、観測された症状（35.1 の adapter＋Settings 6箇所＝計7箇所）に含まれていたからではなく、**同クラスの起動時バックグラウンド取得への予防的統一**である。

**対象外**（変更していないもの・理由）:

- `config_retry_timer.tick`（`gr.Timer` ハンドラ）: Gradio の `Timer.tick` はリスナー個別既定で `show_progress="hidden"` のため対象外。**出典＝実行時実体** `.venv/Lib/site-packages/gradio/events.py` 1330-1334行の `tick = EventListener(..., show_progress="hidden")`。自動生成の型スタブ（`gradio/timer.pyi`）は既定を `"full"` と表示するが、これはリスナー個別既定を反映しないスタブ側の不正確さであり、実行時の既定は hidden が正（将来この食い違いを見て「対象漏れ」と誤解しないこと）。副次的な理由として、`config_retry_timer` は `gr.Timer(3.0, active=False)` で**非アクティブ起動**（`ui.py` 189行付近）であり、設定取得が失敗してリトライに入るまで発火しない＝そもそも起動時の複数 `demo.load` 同時発火シナリオの外にある。
- `model_refresh_btn.click`／`style_reload_btn.click` 等の**ユーザー操作イベント**: 明示的な操作への応答なので進捗表示は維持（意図的に変更しない）。
- `config_state.change`: 対象外（デリバリー先の挙動と無関係）。
- `demo.load(None, js=_min_attr_js)`: `fn=None` のクライアント専用フックであり status-tracker を生成しないため対象外。

### 35.4 検証結果

- [x] **pytest** ✅ **526 passed / 0 failed / 1 skipped**（junit 集計）＝ベースライン（§34 の 526+1）からの退行ゼロ。
- [x] **CDP実ブラウザでの再検証** ✅ 以下すべて確認:
  - ページロード後、アクティブな status-tracker が **0個**。
  - IC-LoRA アコーディオン展開後 **60秒監視**でスピナー出現無し。`elementFromPoint` が INPUT 本体を返し（オーバーレイに覆われていない）、実クリックで Control adapter の選択肢4項目が開く。
  - Settings タブ展開後も status-tracker **0個**（`server_config_json`／`spill_table`／モデル管理ドロップダウン×4 とも操作可能）。
  - Style LoRA タブのギャラリーが正常描画。
  - mock バックエンドでの Generate ボタン実行時、進捗表示は**従来どおり出現し完了後に復元**（ユーザー操作イベントの進捗表示維持＝意図どおり）。
- [x] **オーナー実機確認**✅（2026-07-12）: `run.ps1` でサーバー起動 → `/ui` を開いたところ Control adapter が即座に表示され操作可能であることを確認。副次効果として「以前よりGUIが開くまでの時間が短縮され軽快になった」という体感改善も確認された。

---

## 36. ★バッチA2V機能（就寝中一括生成のGUI機能）＝実装完了・客観ゲート全PASS（pytest 585 passed / 1 skipped）・独立レビュー must-fix解消済み・**GPU実機目視ゲート✅合格（オーナー確認済み）・実機フィードバック修正5点＋第2次フィードバック対応6点も実装済み・§36クローズ**（2026-07-12）

> **正本＝本節。** [`BATCH_A2V_WORKORDER.md`](BATCH_A2V_WORKORDER.md)（§4の未決5論点の決着経緯・GPU実機ゲート結果・オーナー決定2件・修正5点・第2次フィードバック対応6点を含む設計正本）の実装回。ゆっくり実況・VOICEROID実況（音声合成ソフトによるキャラクター実況動画）制作者向けに、音声素材100〜200個ぶんの i2v（画像から動画を生成する機能）口パクアニメーションを就寝中に一括生成するGUI機能。CSVマニフェストの共通仕様（WebView2フロントエンド＝AviUtl2拡張・Reactベースのフロントエンドとの相互運用の正本）は新設[`BATCH_A2V_CSV_SPEC.md`](BATCH_A2V_CSV_SPEC.md)に切り出した。**客観ゲート（pytest・独立レビュー・フルスタックテスト）に加え、GPU実機での目視ゲートもオーナー立ち会いで実施し合格した（§36.6）。オーナー判断待ちだった2件も決着し（§36.7）、実機フィードバックに基づく修正5点（§36.8）・第2次フィードバック対応6点（§36.9）も実装・検証済み。**

本機: 同上（i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`）。

### 36.1 §4の未決5論点の決着（オーナー承認・実装時決定）

1. **§4.1 Replace切替時の挙動**: オーナー原案（切替時に共通プロンプトを全行へ転記＋Undo必須）ではなく、**対案の非破壊方式を採用**。行のテキストは常に「個別プロンプトのみ」を保持し、送信時にAdd（共通の末尾に追記）／Replace（行が共通を置換。行が空なら共通にフォールバック）というグローバル切替に応じて合成する。個別プロンプトが空の行を下敷き編集したいニーズには「共通をこの行にコピー」ボタンで対応。
2. **§4.2 Prompt列の編集方式**: §3の折衷案（Prompt列のみセル直接編集を`.edit`イベントで許可し、選択式が必要なimage列は行選択→下部編集パネルの`gr.Dropdown`経由）を採用。既存Jobsタブ（`jobs_table`/`on_job_select`）と同型の実装パターンを踏襲。
3. **§4.3 Skipの閾値**: 有力案どおり「快適上限は警告のみで表示し、Skip自体は481フレーム基準にする」で確定。解像度別の快適上限（VRAMあふれの無い目安）超過は警告のみでSkipしない。
4. **§4.4 バッチAPIの実装層**: 第一候補（GUI側ループが既存の`POST /generate/chain`を1件ずつ叩く・API側無改修）を採用。ただし、ループ自体はGradioの画面イベント内ではなく**サーバープロセス内のバックグラウンドスレッド（`BatchRunner`）**に持たせた。Gradioの画面イベント（ジェネレータ）はブラウザのSSE接続が切れると打ち切られる（Gradio 6.19の`queueing.py`の既知動作）ため、ブラウザに依存しないスレッドをループの実行主体にする設計判断をした。画面は2秒間隔の`gr.Timer`が`BatchRunner`の状態を読み取って表を再描画するだけの読み取り専用クライアントであり、ブラウザを閉じても・タブがスリープしても生成は継続する。
5. **§4.5 CSVマニフェストの形式**: 10列・UTF-8 BOM付き・CRLF改行・音声フォルダ直下に`batch_a2v_manifest.csv`として配置、と確定。WebView2フロントエンドとの共通仕様として、詳細は[`BATCH_A2V_CSV_SPEC.md`](BATCH_A2V_CSV_SPEC.md)に正本化した。

### 36.2 実装

- **`gradio_ui/manifest.py`（新規）**: CSV純ロジック（`gradio`・スレッド・HTTPクライアントいずれにも非依存）。10列のCSVフォーマット（`CSV_FIELDS`）・`read_manifest`/`write_manifest_atomic`（原子的置換＋ロック時5回リトライ＋autosave退避）・`scan_wav_folder`（更新日時順スキャン・481フレーム超過/非wavのSkip判定）・`merge_rows`（既存CSVとのマージ4規則・Done保護）・`resolve_output_dir`/`unique_output_name`（出力先解決・重複回避の連番付与）を実装。詳細な列契約は[`BATCH_A2V_CSV_SPEC.md`](BATCH_A2V_CSV_SPEC.md)。
- **`gradio_ui/batch.py`（新規）**: `BatchRunner`（サーバープロセス内のデーモンスレッドで動く実行本体・プロセス全体で1インスタンスのシングルトン）。`BatchSnapshot`（バッチ開始時点のGenerateタブの全設定を凍結したデータクラス）を受け取り、行を`queue`順に1件ずつ処理する。各行につき: 音声アップロード（毎回）→画像アップロード（絶対パスをキーにランごとキャッシュ）→参照動画アップロード（ランに1回のみ）→`build_a2v_chain_payload`でペイロード構築→`POST /generate/chain`（409＝サーバービジー時は3回まで3秒間隔でリトライ）→`GET /jobs/{id}`をポーリングし、成功時は出力フォルダへコピー＋一意なファイル名を採番、失敗時は`Failed`として次の行へ進む。1行のあらゆる例外はその行を`Failed`にするだけで全体を止めない設計（try/exceptで各行を隔離）。停止要求（`request_stop`）は次以降の投入を止め、実行中の1件はベストエフォートでキャンセルを試みる（間に合えば`Waiting`へ巻き戻り、間に合わなければ完走）。
- **`gradio_ui/handlers.py`**: 既存の単発A2V送信ロジックから`build_a2v_chain_payload`を関数抽出し、`BatchRunner`と単発A2Vハンドラの両方から呼ぶ共通関数にした。**単発A2V送信のペイロードはバイト等価**（抽出のみでロジック変更なし）。
- **`gradio_ui/ui.py`**: Generateタブの A2V アコーディオンの下に「Batch A2V」アコーディオンを新設・配線。Enableチェック（既定オフ）オンでGenerateボタンが「Start a2v batch」に変わり、単発音声欄・Frames欄がグレーアウトする。表（`gr.Dataframe`・7列表示＝queue/wav/duration/image/prompt/stat/output、Prompt列のみ`interactive`でセル編集可）・行選択→下部編集パネル（画像ドロップダウン・「共通をこの行にコピー」・「この行を再生成」ボタン）・2秒間隔の`gr.Timer`（`BatchRunner`が動いている間だけactive）でのポーリング再描画、を実装。「Set audios」ボタンはバッチ実行中はロックされ、警告を出して何も変更しない。
- **`gradio_ui/i18n.py`**: EN/JA文言48キー追加（アコーディオン見出し・列見出し・状態表示の絵文字＋ラベル・各種警告文言など）。

### 36.3 独立レビュー（実装非関与）

- **must-fix（必須修正）1件**: 個別画像を指定した行の画像パス解決処理に不具合があり修正済み（画像フォルダが未設定のときのフォールバック解決を含め、`gradio_ui/batch.py::_resolve_image_path`で音声フォルダへのフォールバックを明示）。
- **should-fix（推奨修正）3件のうち2件を修正済み**: ①バッチ完了後の一時動画ファイル（`fetch_video`が`%TEMP%`へ落とす中間ファイル）の掃除漏れ→`finally`節で確実に削除するよう修正（200件回すバッチで一時ファイルが積み上がるのを防止）。②バッチ実行中に「Set audios」を押すと実行中の状態と表がズレる問題→`BatchRunner.state != STATE_IDLE`の間は「Set audios」・行編集・画像割当・再生成ボタンいずれも警告を出してロックするよう修正。
- **残り1件（オーナー判断待ち）**: サーバー側`outputs`/`uploads`フォルダが自動掃除されない点。200件規模のバッチを繰り返すとディスクを消費し続けるが、自動削除は`outputs/{job_id}/metadata.json`に記録されたseed情報も一緒に消してしまうトレードオフがあるため、対応の要否はオーナー判断待ちとして記録のみとした。

### 36.4 客観ゲート（PASS）

- **pytest**: **574 passed / 1 skipped**（§35時点の基準526+1=527件 → +48・退行ゼロ。junit集計で`tests=575, failures=0, errors=0, skipped=1`を確認）。新規テストファイル3本＝`tests/test_gradio_batch_manifest.py`（20件・CSV純ロジック＝スキャン/マージ/原子的書き込み/出力名解決）・`tests/test_gradio_batch_runner.py`（16件・`BatchRunner`のライフサイクル/リトライ/停止/クラッシュ耐性をmock HTTPトランスポートで検証）・`tests/test_gradio_batch_fullstack.py`（実サーバー・mockバックエンド経由で失敗継続・再開・出力ファイルの付番・Add/Replaceの実送信を検証）＋既存`tests/test_gradio_handlers.py`の更新（`build_a2v_chain_payload`抽出後の単発A2V経路が従来どおりであることを確認）。
- **独立レビュー**: 36.3のとおりmust-fix対応済み・should-fix2/3対応済み・残り1件は既知事項として記録。

### 36.5 既知の制約（ドキュメント記録のみ・対応は見送り、または将来課題）

1. 行の個別プロンプトに`<lora:名前:強度>`トークンは書けない（解析されず文字列として残る）。LoRA（少量データで画風・キャラクターを追加学習した軽量アダプタ）を指定したい場合は共通プロンプト側かStyle LoRAの仕組みで行う。将来、要望があれば対応を検討する研究課題として位置づける。
2. シード欄が固定値だと全行が同一シードになる（同じセリフを撮り直す「ガチャ」運用では-1=ランダム推奨）。実際に使われたシードはサーバーのジョブ記録（`outputs/{job_id}/metadata.json`）に残るため、後から確認は可能。
3. バッチ実行中は表の編集・「Set audios」の再実行がロックされる（36.3参照。意図的な制約）。
4. サーバー側`outputs`/`uploads`フォルダの自動掃除は未実装（36.3参照。オーナー判断待ち）。
5. 生成中の1件を即時中断する手段は従来通り無い（Phase 1からの既知制約。ベストエフォートのキャンセルのみ）。

### 36.6 GPU実機目視ゲート＝合格（オーナー確認済み・2026-07-12）

客観ゲート（36.3・36.4）に続き、実際にGPU上で複数音声を就寝中運用相当（一括投入・ブラウザを閉じても継続・翌朝の再開・出力ファイルの付番・Add/Replaceモードの見た目上の効果）で確認する目視ゲートを、オーナー立ち会いのもと実施した。**合格。**

実ログで確認された動作:

- 3行バッチ（うち1行はSkip判定）で2行連続生成→就寝運用相当の完走。
- ガチャ再生成（同じセリフの撮り直し運用）で`_2`〜`_4`の自動付番。既存mp4は無傷のまま。
- 停止ボタン押下で「実行中の1件は完走させ、次の行は投入せず停止する」という設計どおりの挙動（サーバーログに`stop requested, halting before row 2`を確認）。
- 再開時、Done／Skip行を正しくスキップして続きから再開。
- 日本語ファイル名の音声で正常動作（文字コード起因の不具合なし）。
- 解像度変更（1280x768→896x1152）が`BatchSnapshot`へ正しく反映される。

**オーナーの評価＝「事前に期待していた要件はいずれも満たしている」。**

### 36.7 オーナー決定2件（36.3・36.5で判断待ちとしていた事項の決着・2026-07-12）

1. **サーバー側`outputs`/`uploads`フォルダの自動削除方針（36.3の残り1件・36.5の4）**: **ユーザーの手動削除で運用する**と決定。UI側に削除機能は追加しない。理由＝ローカル専用ツールであり、ユーザーは日常的にエクスプローラーで`\outputs`を直接操作するため、直接操作のほうが便利という判断。`metadata.json`のseed記録が自動削除で消えるトレードオフを踏まえた判断でもある。
2. **行の個別プロンプトへの`<lora:名前:強度>`トークン対応（36.5の1）**: **将来の研究課題へ格下げ**。ユーザーから要望が出た時点で検討する。現状どおりLoRAは共通プロンプト側／バッチ開始時の`BatchSnapshot`（Generateタブ設定のスナップショット）で全行共通のまま。

### 36.8 実機フィードバックに基づく修正5点（実装済み・pytest 574 passed / 1 skipped＝退行ゼロ・2026-07-12）

GPU実機目視ゲートの過程でオーナーから寄せられた使い勝手フィードバックを、同一セッション内で実装・検証まで完了した。

1. **バッチ実行中のGenerateボタン制御**: 実行中はボタン表示が「Batching a2v...」に変わり押下不可になり、完了で「Start a2v batch」に復元される。誤操作による二重投入を防止する。i18nキー`batch_running`を新設。
2. **Batch A2Vアコーディオン内の並び替え**: 実際の操作順に合わせて再配置。Enableパネル（`gr.Group`にEnable／音声フォルダ／画像フォルダ／出力先〔Auto/Custom〕を集約）→ Prompt mode（Add/Replace）はSet audiosボタンの直上 → Set audios → 表 → 表の直下にmax durationとサマリ（Done/Failed/Skip/Waiting）を横並びで集約 → 行編集パネル（画像ドロップダウン・行情報）→ 共通プロンプトコピー → 行再生成 → 停止ボタン。
3. **表の列名短縮**: `queue#`→「#」、`Duration`→「dur.」（表示崩れ対策）。
4. **Prompt列の初期幅確保**: `gr.Dataframe`の`column_widths`でPrompt列に41%を割り当て、Set audios直後からセルをクリックして編集しやすくした。
5. **コンソールログの静音化**: uvicornのアクセスログから`/ui/gradio_api/`を含む行（Gradioの内部イベント通信。2秒間隔の表更新タイマー等が出す大量ログ）を除外するフィルタを`main.py`に追加（既存の`JobPollingAccessFilter`と同方式）。GradioがStarletteの古い定数を使うことで出る`StarletteDeprecationWarning`もメッセージ・カテゴリ限定で抑制した。`/api/v1/*`のアクセスログと生成進捗ログは従来どおり表示され、表更新タイマーの間隔（2秒）自体は変更なし。

**検証**: pytest再実行で**574 passed / 1 skipped**（退行ゼロ）を再確認。GPU実機での目視確認もオーナー自身が実施し、上記36.6の合格評価に含まれる。

### 36.9 第2次フィードバック対応6点（実装済み・pytest 585 passed / 1 skipped＝退行ゼロ・2026-07-12）

36.8の修正5点をオーナーが実機で使い込んだ結果、さらに6点のフィードバックが寄せられ、同一セッション内で実装・検証まで完了した。

1. **表の横スクロール復活**: 列幅を固定ピクセル（#=50px／音声=220px／dur.=70px／画像=140px／Prompt=480px／状態=90px／出力=220px、合計1270px）に変更した。Gradioの列幅は割合指定だとビューポート幅に常に収まる仕様のため、以前はPrompt列が実質的に潰れて折り返しが多発していた。固定ピクセル化により、合計幅がパネル幅を超えると表に横スクロールバーが出るようになった。セル内の折り返し自体は禁止していない（480pxを超える長文プロンプトは折り返して表示される）。
2. **表示位置の調整**: max durationとサマリ（Done/Failed/Skip/Waiting）の行を「Set audios」ボタンと表の間（表の直上）に移動した。
3. **Start押下時のSkip再判定**: バッチ開始時に、処理対象の全行についてフレーム数とSkip判定を、その時点のフレームレートで再計算するようにした。効果は次の2点: (a) 手動でWaitingに戻した上限超過行も、開始時に確実にSkipへ再マークされる。(b) 「Set audios」後にfpsや解像度を変更しても、フレーム数は開始時の設定で計算し直される（**注意**: 行のframes値を手で編集していても、開始時にこの再計算で上書きされる）。判定・再計算がすべて通ってからCSVへ書き込みバッチを開始する設計にしたため、開始不可のときは表もCSVも一切変化しない。
4. **Regenerateガード**: Skip行を選んで「この行を再生成」を押しても、無条件にWaitingへ戻すのではなく、理由付きの警告を出して止めるようにした。
5. **フールプルーフ（開始前チェック・オーナー承認済みの判定表）**: 検査対象は処理対象行（Waiting／Failed／クラッシュ遺残）のみ。
   - **プロンプト**: 共通プロンプトが入力済みなら開始可。共通プロンプトが空の場合、Addモードなら開始不可（「共通プロンプトを入力するかReplaceに切り替えて」の旨を案内）。Replaceモードなら、処理対象行のプロンプトが全部埋まっていれば開始可、1行でも空なら開始不可（空の行数を表示する）。
   - **画像**: 共通キーフレーム画像の**1枚目**（Keyframe imagesのスロット1・先頭フレーム0。Useチェックがオン かつ 画像ありが条件）が設定済みなら開始可。未設定の場合（スロット2以降だけ設定されている場合を含む）、処理対象行が全部個別画像指定になっていれば開始可、"Shared"のままの行が1つでも残っていれば開始不可（2026-07-13に判定強化。§36.10参照）。
   - 狙いは、入力忘れという単純ミスで一部の行だけ走ってFailedが量産される事態を防ぐこと。全行に個別入力を済ませた使い方（Replace＋全行個別プロンプト、全行個別画像）は従来どおり問題なく開始できる。
6. **テスト**: 既存574件＋新規11件＝**585 passed / 1 skipped**。フルスタックテストのうち「音声が短すぎてサーバー側422になるシナリオ」を検証していた1本は、Start押下時のフレーム数再計算により、そのエラー自体が到達不能になった。テストの狙い（再開機能の検証）は維持したまま、失敗ベクタを「存在しない個別画像を指定した行」に差し替えて対応した。

**検証**: pytest再実行で**585 passed / 1 skipped**（§36.8時点の574+1から+11・退行ゼロ）を確認。

**§36クローズ（GPU実機目視ゲート合格・オーナー決定2件決着・修正5点＋第2次フィードバック対応6点実装済み・2026-07-12）。** 残タスクはgit commitのみ（作業ツリーの変更のまま・オーナー指示待ち）。

### 36.10 画像判定の強化（実装済み・pytest 589 tests / 588 passed / 1 skipped＝退行ゼロ・2026-07-13）

36.9の5.で実装した画像判定を、オーナー承認済み仕様にもとづき「共通キーフレーム画像が1枚でもあればOK」から「**1枚目**（Keyframe imagesのスロット1・先頭フレーム0。Useチェックがオン かつ 画像ありが条件）が入っているか」に厳密化した。

- **確認結果**: `gradio_ui/ui.py`のdispatch（Batch開始ハンドラ内、`shared_images`収集ループ）はもともと`if en and img:`（Useチェックがオン かつ 画像あり）のスロットしか`shared_images`へ集めない。したがって「Useチェックがオンか」という条件は本増分以前から担保済みであり、今回の変更対象外。スロット1はframe位置0固定、スロット2以降はframe>0で使うのが運用上の前提のため、「shared_imagesに`frame_idx==0`のエントリが存在する」ことをもって「1枚目が入力されている」と判定できる。
- **実装**: `gradio_ui/batch.py::_validate`の画像分岐1を、「`snapshot.shared_images`が非空」から「`snapshot.shared_images`に`frame_idx == 0`のエントリが含まれる」判定に変更（`has_first_keyframe = any(int(frame_idx) == 0 for _p, frame_idx, _s in snapshot.shared_images)`）。含まれない場合（空、またはスロット2以降のみ）は従来どおり、処理対象行が全部個別画像指定ならOK、"Shared"行が残っていれば開始不可。ブランケット＋条件分岐の構造・理由コード（`"shared keyframe image(s) required: ..."`）は変更なし（`_batch_start_reason`のマッピングも無改修で追従）。個別画像ファイルの実在チェックは今回も追加していない（オーナー決定）。
- **i18n**: `gradio_ui/i18n.py`の`batch_msg_no_common_image`（キー名不変・en/ja）を「1枚目」が要件だと分かる文言に更新。
- **テスト**: `tests/test_gradio_batch_runner.py`に3件追加——①`shared_images`がスロット2以降のみ（frame_idx>0のみ）＋Sharedの対象行あり→start失敗・chain呼び出しゼロ・CSV未生成（`test_start_rejects_shared_row_when_only_non_first_slot_set`）。②`shared_images`にframe_idx==0のエントリあり→開始成功の回帰（`test_start_allows_shared_row_when_first_slot_set`）。③1枚目なし＋対象行すべて個別画像→開始成功の回帰（`test_start_allows_no_first_slot_when_all_targets_have_own_image`）。既存の`shared_images`利用テストはすべてframe_idx=0のエントリを使っていたため追従不要だった。
- **検証**: `.venv\Scripts\python.exe -m pytest -q -p no:warnings`をjunitxml集計で確認——`tests="589" failures="0" errors="0" skipped="1"`（§36.9時点の585+1=586から+3＝退行ゼロ）。

Docs更新: `README.md`（Batch A2V節の開始前チェック説明）・`BATCH_A2V_WORKORDER.md`（§画像判定の強化・2.5節）・`NEXT_SESSION_HANDOFF.md`（最新ステータスブロック＋画像判定の強化サブ節）・本節。

---

## 37. ★V2V Join機能（末尾トリム方式）の実機検証＋IC-LoRA×A2V併用のreal GPU生成完走＝オーナー実機ゲート全項目合格（2026-07-21）

> **正本＝本節。** V2V Join復活（API拡張＝コミット `d22706e`、本書はコード非対象なので詳細は `LTX23_Backend_Specification.md` §6.1／§6.3、フロント側の実装増分はI1〜I6）の、オーナー立ち会いによるrealバックエンド・実GPUでの実機検証記録。詳細な経緯・修正差分・実機ログはフロント側 `Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md` §3-27〜§3-29・`Nz-LTX23-frontend-AviUtl2/Docs/DEVLOG.md` §44〜§45を参照。

本機: 同上（i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11）。

### 37.1 V2V Join機能（末尾トリム方式）の実機検証＝全項目合格

realバックエンド・実GPUでオーナーが以下を確認し、**全項目合格**:

- Joinボタンの表示（実装以来初の実機表示）。
- トリム2択（末尾を残す秒数の選択）→ Join実行。
- プレビューの差し替え（結合後の joined.mp4 への切り替え）。
- 位置付き🎞挿入＋仮オブジェクトの自動削除。
- Unjoin／再Join（上書き）。
- joined.mp4 を手動削除した状態からの復帰。

### 37.2 fps注意文の誤情報バグ発覚→真因特定→同日中に修正・再検証合格

実機検証の過程で、fps注意文が「project 24 / video 24なのに did not match」という誤情報を表示する不具合が発覚した。

- **真因**: バックエンドのJoin処理自体は全ケースで成功していた（正規化・トリム・atomic rename とも仕様どおり動作）。不具合の所在はフロント側の文言設計——正規化（`JoinResponse.source_normalized`）は解像度差だけでも発火する（fps差の有無を問わない）仕様なのに、注意文はfps不一致を前提とした文言しか出さず、しかもソース側の実測fpsを表示していなかった。
- **修正**: フロント側で注意文を発火理由別の3部品（ソースfps明示の文言／fps非言及の正規化文言／プロジェクトfps推奨行）へ分岐する形に改修し、同日中に再検証まで完了した。バックエンド側は無改修。詳細は `PENDING_TASKS_CLOSED.md` §3-29・`DEVLOG.md` §45。

### 37.3 IC-LoRA×A2V併用のreal GPU生成完走

音声＋参照動画＋制御LoRAを同時に添付したチェーン生成を実GPUで完走させ、ポーズ制御（参照動画由来）と音声由来の動きが両立した動画が出力されることを確認した。同日合格。

### 37.4 参照

詳細な経緯・実機ログ・フロント側の修正差分はフロント側 `Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md` §3-27〜§3-29・`Nz-LTX23-frontend-AviUtl2/Docs/DEVLOG.md` §44〜§45を参照。バックエンド側のV2V継続初回実装は本ログ§24、2026-07-21のAPI拡張（`is_v2v`／`joined`／`source_tail_seconds`／`trimmed_source_seconds`／`source_fps`）はコミット `d22706e`（`LTX23_Backend_Specification.md` §6.1／§6.3に反映済み）。

---

## 38. ★NAG（非CFGネガティブプロンプト）機能＝実装完了・機械検証（selfcheck・pytest）全PASS・**実機ゲート全項目合格（2026-07-29 オーナー実機確認で完了）**（2026-07-28実装／2026-07-29実機ゲート完了）

> **正本＝本節。** 蒸留版 LTX 2.3 は CFG（Classifier-Free Guidance。正負2パスのdenoiseでネガティブプロンプトを効かせる従来手法）が `guidance_scale=1.0` に凍結されているため、従来型のネガティブプロンプトはこれまで完全な no-op だった。NAG（Normalized Attention Guidance, arXiv:2505.21179）は、cross-attention（テキストと映像/音声の対応を取る注意機構）の出力レベルで正プロンプト出力と負プロンプト出力を外挿・正規化・ブレンドすることで、CFGの2パス化なしに1パスのままネガティブプロンプトを効かせる非CFG手法。適用範囲は単発Generate（`POST /generate`）・Clip Chain（`POST /generate/chain`）・バッチA2V（内部的に行ごとに`/generate/chain`を叩く）の全経路。Wave 0〜3（エンジンコア→エンジン配線→API/runner→GUI）を段階的に実装し、本節（Wave 4）で実機ゲート前の最終ドキュメント化を行う。

### 38.1 決定事項

1. **式の規約**: 先行実装 [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) の `LTX2_NAG`（`nodes/ltxv_nodes.py`）実コードを取得・精読し、その規約に完全準拠する設計にした。正プロンプト出力を Z⁺、負プロンプト出力を Z⁻ とすると、外挿式は `Z̃ = s・Z⁺ − (s−1)・Z⁻`（`s`=nag_scale）。以降、L1ノルム比 `R = ‖Z̃‖₁ / ‖Z⁺‖₁` に対して `min(R, τ)/R`（`τ`=nag_tau）でノルムを頭打ちにし、`α`（nag_alpha）で正出力とブレンドしてから、per-head ゲート（`2・sigmoid(to_gate_logits(x))`）→ `to_out` という順で結合する（結合はゲートより前）。既定値 `scale=11.0 / tau=2.5 / alpha=0.25` も KJNodes の既定値をそのまま踏襲した。
2. **非対称設計の根拠**: 実装時の敵対的レビュー（Opusサブエージェントによる17件指摘）で「negative側にもAdaLN変調を掛けて対称化すべき」という指摘が上がったが、採用しなかった。理由は、本番で使う3種のGGUF量子化モデルすべてが `cross_attention_adaln: true` をメタデータに埋め込んでおり、実際の推論では positive context が毎ステップ `context*(1+scale_kv)+shift_kv` の変調を受けてからattentionに渡る一方、KJNodes参照実装のコードを直接照合したところ、そちらも同じ非対称（positiveは変調済み・NAGのnegative contextは生のまま）で動いていることを確認したため。既定値11.0/2.5/0.25はこの非対称の下でチューニングされた値であり、対称化すると参照実装から意味的に乖離してしまう。指摘は2体の独立検証エージェント（ローカルコード確認＋KJNodes実物のWeb取得）で裏取りした上で不採用と判断した。
3. **空ネガティブは422**: `nag_enabled=True` かつ `negative_prompt` が空（または空白のみ）の場合、両リクエストモデル（`GenerateRequest`/`GenerateChainRequest`）のバリデータが `ValueError("nag_enabled requires a non-empty negative_prompt")` を送出し、`422 VALIDATION_ERROR` として返る。GUI側にも同内容のprecheckトースト（`nag_msg_negative_required`）があり、API呼び出し自体を行わずに止める。
4. **UIルール**: 共有プロンプト欄の直下（Generate/Clip Chain両タブの外）に「Negative Prompt」アコーディオンを新設し一本化した。中の入力欄（`negative`、既定値`"blurry, low quality, distorted"`）は「non-CFG Negative」チェックボックスがONのときだけ編集可能になり、OFFの間は非活性（グレーアウト）表示のまま既定文字列が残る。「NAG / Other」ラジオはOtherを選ぶと即座にNAGへ復帰しトースト通知が出る簡易フォールバック（Other方式の実体は未実装。§38.5参照）。
5. **無効時バイト一致の構造保証**: `NagState`が「未要求」の場合、`NagService.install()`は先頭のbool判定1回で即returnし、cross-attentionのforwardへのパッチを一切当てない。したがって`nag_enabled=False`のジョブは、エンジン内部の挙動がNAG導入前とバイト単位で完全に一致する（パッチ自体が存在しないため、恒等短絡のような数値的な作り込みに依存しない構造的な保証）。同様にworkerペイロードも無効時は`payload["nag"]`キー自体が存在しない加算方式で、既存のkey-order契約テストが無改修で通っている。

### 38.2 実装箇所一覧

- **`engine/transformer/nag_service.py`（新規）**: NAGコア。`NagParams`（negative_prompt/scale/tau/alpha の frozen dataclass）、`NagState`（1ジョブ分のミュータブル状態）、`encode_negative(text_encoder, prompt)`（negativeを1回エンコードし映像・音声両方のcontextを返す共通ヘルパ）、`nag_combine(z_pos, z_neg, scale, tau, alpha)`（§38.1の式。冒頭に`alpha==0.0 or scale==1.0`の恒等短絡）、`_make_nag_forward`（cross-attention forwardの等価展開パッチ。pe/mask/perturbationが非Noneなど想定外の呼び出しはRuntimeErrorでfail-loud）、`NagService(state_provider).install(transformer)`（未要求なら即return 0、要求済みでcontext未設定ならRuntimeError）。
- **`engine/transformer/nag_selfcheck.py`（新規）**: `.venv-engine`のpythonで直接実行するエンジンvenv用の自己検証（pytest非収集）。6項目は§38.3参照。
- **`engine/pipeline/fast_video_pipeline.py`**: `__init__`末尾で`_install_nag()`を無条件に呼び`ledger.transformer`をラップ（block-swap/GGUFラップの後、最後に足す）。`_set_nag_job`（IC-LoRAの`_set_ic_job`と同型）。`_run_inference`の`try:`開始位置を既存4本のモジュールグローバル差し替え区間の前に前倒しし、`encode_text`パッチを含む計5本すべてを`finally`復元の傘に入れた（既存の潜在欠陥＝パッチ適用中の例外でグローバルが復元されない問題も同時に解消）。`_make_nag_encode_text`がpositiveを素通ししつつ同じ生きたtext_encoderでnegativeを1回追加エンコードする。`generate`/`generate_chain`のシグネチャ末尾に`nag: NagParams | None = None`を追加し、両方とも`try/finally: self._nag.reset()`。
- **`engine/pipeline/chain_pipeline.py`**: `run_chain`に`nag=None`引数を追加。positiveエンコードループ直後・`del text_encoder`直前（text_encoder生存中）でnegativeエンコードを実施。stage-1/2とも同一transformer・同一negativeを共用。
- **`engine/worker.py`**: `_resolve_nag(msg)`でNagParamsを復元し、`_do_generate`/`_do_generate_chain`から`_PIPE.generate(...)`/`generate_chain(...)`へ渡す。ログ行に`nag=on/off`を出力。
- **`api/models.py`**: `GenerateRequest`/`GenerateChainRequest`双方に`nag_enabled: bool = False`・`nag_scale: float = Field(11.0, ge=1.0, le=20.0)`・`nag_tau: float = Field(2.5, ge=1.0, le=10.0)`・`nag_alpha: float = Field(0.25, ge=0.0, le=1.0)`を追加。`negative_prompt`に`max_length=2000`（promptと同じ上限。これまでno-opだったため上限が無かった）。バリデータに§38.1の3の相互検証。`to_clip_request`（本番経路＝`services/job_store.py`の`create_chain_if_idle`が毎回呼ぶ）に4フィールドを転記。
- **`services/ltx_runner.py`**: `_RealBackend.generate`/`generate_chain`のペイロード末尾に、有効時のみ`payload["nag"] = {negative_prompt, scale, tau, alpha}`を加算（無効時はキー自体が存在しない）。
- **`gradio_ui/i18n.py`**: `nag_accordion`/`nag_note`/`nag_enable`/`nag_lbl_method`/`nag_method_nag`/`nag_method_other`/`nag_msg_fallback`/`nag_lbl_scale`/`nag_lbl_tau`/`nag_lbl_alpha`/`nag_msg_negative_required`をen/ja両方に新設。既存`info_negative`の文言差し替え。
- **`gradio_ui/ui.py`**: 旧・タブ内グレーアウトNegative Prompt欄（Generate/Chain両方）を撤去し、共有プロンプト直下・Tabsの外に共通アコーディオンを新設。`on_nag_enable_toggle`（interactive切替のみ）・`on_nag_method_change`（"nag"以外を選ぶと即NAGへ復帰＋トースト）ハンドラを追加。
- **`gradio_ui/handlers.py`**: `build_a2v_chain_payload`・単発`generate`ハンドラ・chainハンドラそれぞれの末尾にnag 4値を追加し、有効時のみペイロードへ加算。

### 38.3 機械検証の結果

**エンジンvenvでのselfcheck（`.venv-engine`のpythonで`python -m engine.transformer.nag_selfcheck`を実行）＝6/6 PASS**（本節作成時に再実行し確認済み）:

```
[PASS] nag_combine matches KJNodes reference (bf16+fp32, both clamp branches)
[PASS] identity short-circuit (alpha=0 / scale=1) is the same z_pos object
[PASS] zero-norm degenerate inputs produce no NaN/Inf
[PASS] real BasicAVTransformerBlock: OFF no-op, ON matches reference, fail-loud shapes
[PASS] NagService.install() on fake transformer with NAG off is a no-op
[PASS] NagService.install() raises when requested but negative context not encoded

6/6 checks passed
```

4番目の「real BasicAVTransformerBlock」チェックは、モックではなく実物の`BasicAVTransformerBlock`を`cross_attention_adaln=True`・`apply_gated_attention=True`（＝本番3モデルの実際の構成）でCPU上に小サイズ構築し、(a) NAG OFFではパッチ0件・出力がビット一致、(b) NAG ONでは「positiveは変調済み・negativeは生」という非対称を踏まえた手書き参照計算と一致、(c) 結合順（NAG→ゲート→to_out）が正しいこと、の3点を検証している。

**アプリvenvでのpytest（`.venv\Scripts\python.exe -m pytest -q`）＝658 passed / 6 skipped**（skipはいずれも`torch`未導入によるエンジン系テストの収集スキップ＝アプリvenvに元々torchを入れない設計のための既知スキップで、NAG関連ではない）。Wave 0〜3合計で新規テスト+37件程度を追加した一方、**既存テストの改修は「positional `_chain_args`ヘルパ」と「i18nキー一覧」の2件に限定**した。これは狙って達成した性質で、NAGが既存の生成経路に対して純粋な加算的拡張（無効時は挙動もペイロードも従来と不変）であることの、テストスイート側からの裏付けになっている。

### 38.4 実機ゲート表（2026-07-29 オーナー実機確認で全項目合格）

以下は承認済み計画書の「実機検証チェックリスト」を転記したもの。**2026-07-29、オーナーの実機確認により全項目合格した。**

| ゲート | 内容 | 合格条件 | 状態 |
|---|---|---|---|
| G0（最重要） | 回帰: NAG OFFで単発T2V/I2V/A2V/chain 2clips/バッチ2行 | 出力mp4のSHA256が変更前とバイト一致・peak VRAMも同値 | ✅ 合格（2026-07-29 オーナー実機確認。単発T2V/I2Vは下記2026-07-28追記のとおり機械検証でも裏付け済み） |
| G1 | ログ | `NAG installed on <実測数> cross-attention modules ...`（期待96）が1ジョブ1回出力される（chainでも1回） | ✅ 合格（2026-07-29 オーナー実機確認） |
| G2 | 恒等 | alpha=0 / scale=1でOFFと一致（恒等短絡によりビット一致が期待値。cuBLAS差ならPSNR≥50dB許容＋要因記録） | ✅ 合格（2026-07-29 オーナー実機確認） |
| G3（オーナー目視） | 効果 | 同一seedでOFF/ONのSHA256が異なり、negativeの概念が抑制され、破綻がない（破綻時はalpha 0.25→0.15、scale 11→5で再確認） | ✅ 合格（2026-07-28） |
| G4 | 音声到達 | 音声寄りnegativeで音声トラックが変化することを聴取確認 | ✅ 合格（2026-07-29 オーナー実機確認） |
| G5 | コスト | VRAM増分ピーク（z_neg＋z_gの2テンソル分、768p stage-2タイルで約+350MB目安・16GB内）と時間増（cross-attentionは倍だがself-attention支配のため全体数%〜15%程度の見込み）を実測。バッチA2Vは行ごとにGemmaロード＋negativeエンコードが加算されるため行あたりの時間増も実測 | ✅ 合格（2026-07-29 オーナー実機確認） |
| G6 | 経路網羅 | ONで単発T2V/I2V/A2V/chain/chain+V2V/chain+IC-LoRA/バッチ/chunked_upsampleすべて完走 | ✅ 合格（2026-07-29 オーナー実機確認） |
| G7 | 併用非干渉 | IC-LoRA＋NAG、block-swap小窓＋NAG、GGUF既定経路のいずれも問題なく完走 | ✅ 合格（2026-07-29 オーナー実機確認） |
| V-UI | 目視6項目 | 共有アコーディオンが両タブから見える／旧2欄消滅／チェックOFFグレーアウト・ONで解除／Other→NAG復帰トースト／言語切替追従／有効＋空negativeはトーストのみでジョブ不発 | ✅ 合格（2026-07-29 オーナー実機確認） |

**2026-07-28追記（G3合格）**: オーナーがGradio経由2ジョブ（`423d19ca-1228-4621-a48f-09862fdcc5a0`／`74c4a11f-66ab-4b5e-ae30-f158ddea9920`）・AviUtl2経由2ジョブ（`2637bcc9-edd0-484d-b49e-a97787cc5f3c`／`77f028ce-45b0-464f-9b53-9b0d3b32238c`）でOFF/ON比較を実施し、NAG=Enableでnegative_promptに書いた内容が出力から目に見えて減ることを確認した（G3合格）。G0/G1/G2/G4〜G7・V-UIは引き続き未実施。

**2026-07-28追記（G0部分合格、§40の依存整理バッチ実機回帰と同時実施）**: §40.6の実GPU SHA回帰（MCP経由、job `b1645c1d-bdfa-441d-922e-39ba90ae90e1`＝T2V／job `3d231fc1-4cad-49d8-9026-8ddf45419f11`＝I2V）は、依存整理バッチ（§40.1〜40.3）だけでなくNAG機能そのものについても「OFF時は出力に一切触れない」ことの裏付けになる。両ジョブとも基準SHA256・peak_vram_mbと完全一致し、T2Vのリクエストエコーで`nag_enabled=false`を確認済み。**したがってG0は単発T2V／単発I2Vの2経路に限り部分合格とする。A2V・chain 2clips・バッチ2行のG0、およびG1/G2/G4〜G7・V-UIは引き続き未実施のまま**（過大評価を避けるため、G0行は「部分合格」表記に留め全合格とはしない）。

**2026-07-29追記（残ゲート全項目合格・オーナー実機確認完了）**: オーナーが2026-07-29に実機確認を行い、「結果は合格」と確認した。これを受けて、G0（A2V・chain 2clips・バッチ2行を含む残り経路）・G1・G2・G4・G5・G6・G7・V-UIの残り全項目を合格とした。個別ゲートのSHA256・VRAM増分などの数値は本追記の時点では記録していない（必要であれば別途実測記録を追加する）。

**§38は2026-07-29、オーナーの実機検証完了（全ゲート合格）を受けてクローズする。**

---

## 39. ★MCPサーバー（`mcp_server/`）＝実装完了・機械検証（pytest）全PASS・**実機検証をエージェント経由で全項目実施（2026-08-04）。残るはClaude Code本体のUI（承認ダイアログ・`/mcp`画面）の目視のみ**（2026-07-28実装／2026-08-04実機検証）

> **正本＝本節。** Claude Code 等の MCP（Model Context Protocol）クライアントから、Web の操作パネルと同等の操作をできるようにする `mcp_server/` パッケージ（22ツール）の実装記録。設計判断の詳細は [`MCP_SERVER_DESIGN.md`](MCP_SERVER_DESIGN.md)、利用者向け説明は [`../README.md`](../README.md) 「AIエージェント連携（MCPサーバー）」節を参照。実装計画そのものはオーナーのプランファイル（リポジトリ外）が正本で、Wave 0〜6（依存追加＋骨組み→system完成→uploads/submit_generate→jobs 7本→submit_chain→outputs→batch_planning）を順に実装し、本節（Wave 7）でドキュメント化と最終機械検証を行った。

### 39.1 実装物

- `mcp_server/`（新設・7ファイル＋`tools/`サブパッケージ6ファイル＝計13ファイル）: `__init__.py` / `__main__.py`（`Path(__file__)`起点でsys.path自己解決）/ `settings.py`（`LTX_MCP_*`環境変数→`config.yaml`の順で解決）/ `client.py`（`BackendClient`＝httpx.AsyncClient＋エラー翻訳）/ `params.py` / `paths.py` / `batch_planning.py` / `server.py`（`build_server()`＋日本語instructions）/ `tools/{system,uploads,generate,jobs,outputs,batch}.py`。
- `pyproject.toml` / `requirements.txt`: `mcp>=1.28,<2` をmain依存として追記。
- `scripts/setup.ps1`: セットアップ完了時に絶対パス入り `.mcp.json` を生成する `New-McpJson` 関数を追加。`.gitignore` に `.mcp.json` を追加（マシン固有の絶対パスを含むためコミット対象外）。
- `tests/conftest.py`: `_build_app(tmp_path)` を抽出し `mcp_app` フィクスチャを追加（既存 `client` フィクスチャの外形は不変・15ファイルが依存する契約を維持）。
- `tests/test_mcp_*.py`（新規7ファイル）: `test_mcp_registration.py` / `test_mcp_tools_system.py` / `test_mcp_errors.py` / `test_mcp_tools_generate.py` / `test_mcp_tools_jobs.py` / `test_mcp_outputs.py` / `test_mcp_batch_planning.py`。
- `gradio_ui/handlers.py` / `gradio_ui/manifest.py`: 写経元であることを示すコメントを追加（ロジック自体は無変更）。

### 39.2 pytest推移（Wave別・退行ゼロ）

Wave 0〜6を順に実装するにつれ、アプリvenv（`.venv\Scripts\python.exe -m pytest -q`）のテスト総数は次のように増加した（NAG機能完了時点の658 passed/6 skippedを起点に、MCPサーバーのテストのみを加算した）。

```
658 → 675 → 699 → 736
```

W7（本節作成時点）で最終確認した結果は **736 tests / 730 passed / 6 skipped / 0 failed / 0 errors**（`--junitxml`集計）。skipの6件はいずれも `torch` 未導入によるエンジン系テストの収集スキップで、アプリvenvに元々torchを入れない設計のための既知スキップ（NAGの回でも同数出ており、MCP関連ではない）。既存テストへの改修は最小限（`tests/conftest.py`の`_build_app`抽出のみ）で、新規7ファイルはすべて加算的に追加された。

### 39.3 stdout清浄性の確認

MCPの `stdio` トランスポートはJSON-RPCを標準出力に流すため、`mcp_server/`配下のどこかで意図せず標準出力に書き込みが発生すると、プロトコル全体が即座に壊れる。この清浄性を次のコマンドで機械的に確認した。

```powershell
.venv\Scripts\python.exe -m mcp_server < NUL
```

標準入力を即座に閉じることでサーバーが起動直後に終了する状態を作り、標準出力バイト数を計測したところ **0 バイト**（終了コード0）だった。ドキュメント作成による変更後に再実行しても結果は変わらず、退行が無いことを確認した。

### 39.4 22ツール登録の確認

`tests/test_mcp_registration.py::test_tool_name_set_matches_expected` が、`mcp.shared.memory.create_connected_server_and_client_session` 経由の実MCPプロトコル往復で `mcp.list_tools()` を呼び、返ってきたツール名の集合が `EXPECTED_TOOLS`（22個・system 6/uploads 3/generate 2/jobs 7/outputs 3/batch 1）と厳密一致することをアサートしている。同ファイルの別テストは、全ツールに空でない説明文があること、`dict[str, Any]`を返すツールの結果が`{"result": ...}`でラップされず`structuredContent`にそのまま入ること（FastMCP 1.28の構造化出力の仕様どおり）も検証済み。

### 39.5 実機実叩き（Claude Codeからの操作）＝未実施 → **§39.6で実施済み（Claude Code UIの承認導線を除く）**

**本節作成時点で、Claude Code など実際のMCPクライアントからの操作は一つも行っていない。** ここまでの検証はすべてモックバックエンド（`httpx.ASGITransport`）を使ったオフラインのpytestであり、実際のバックエンドプロセス（`run.bat`）と実際のMCPクライアントを組み合わせた実機ゲートは、オーナーの確認待ちである。

以下は承認済み計画書「W7」の実機検証チェックリストを転記したもの。**全項目未実施**（2026-07-28時点。**2026-08-04に承認ゲートを除く全項目を実施＝§39.6**）。

| ゲート | 内容 | 状態 |
|---|---|---|
| 承認ゲート | リポジトリフォルダをClaude Codeで開く→ワークスペース信頼確認→プロジェクトスコープMCPサーバーの承認（⏸ Pending approval）が出て、承認すると解除される | ⬜ 未実施（**エージェントからは検証不能**＝§39.6） |
| 22ツール表示 | `/mcp` コマンドで `nz-ltx23` サーバーと22個のツールが一覧表示される | 暫定✅ 2026-07-28（下記注記参照）→ **§39.6で再実施済み** |
| T2V submit+poll | `submit_generate`（画像なし）→`wait_for_job`を繰り返し呼んで完了確認 | 暫定✅ 2026-07-28（job `b1645c1d-bdfa-441d-922e-39ba90ae90e1`、§40.6参照）→ **§39.6で再実施済み** |
| I2V | `upload_image`→`submit_generate`（`conditioning_images`指定）→完了確認 | 暫定✅ 2026-07-28（job `3d231fc1-4cad-49d8-9026-8ddf45419f11`、§40.6参照）→ **§39.6で再実施済み** |
| A2Vバッチ1行 | `plan_a2v_batch`→1行分`upload_audio`→`submit_chain`→`wait_for_job`→`save_job_video` | ⬜ 未実施 → **§39.6で実施済み** |
| join | V2V継続ジョブに対して`join_job`を呼び、`joined.mp4`が生成される | ⬜ 未実施 → **§39.6で実施済み** |
| purge dry_run | `purge_terminal_jobs(dry_run=true)`が実際には何も削除せず対象一覧のみ返す | ⬜ 未実施 → **§39.6で実施済み** |
| JOB_BUSY挙動 | ジョブ実行中に別の`submit_generate`/`submit_chain`を呼ぶと409 JOB_BUSY相当のエラーがエージェントに伝わる | 暫定✅ 2026-07-28（I2V実行中に409「`JOB_BUSY: A job is already running (Phase 1 allows one concurrent job)`」をライブ確認）→ **§39.6で再実施済み** |
| api_key設定時の再起動 | `--api-key`指定でバックエンドを起動した状態でMCP経由の操作がBearer認証込みで通る | ⬜ 未実施 → **§39.6で実施済み** |

**2026-07-28追記（「暫定✅」の位置づけ）**: 上表の「暫定✅」は、オーナー承認済みのエージェントがMCPクライアント（`mcp==1.28.1`、stdioトランスポート、実バックエンド＝ltx-distilled・RTX 4070 Ti SUPER）を自前のスクリプトから直接叩いて確認したもので、**Claude Code本体のUI（承認ダイアログ・`/mcp`コマンドの実画面）を経由した確認ではない**。オーナー判断により「MCP経由で実バックエンドを回せることが確認できれば暫定合格と見做す」ため✅としているが、Claude Code UIの承認導線そのものの実地確認はまだ済んでいない。**未実施のまま残る項目**: 承認ゲート（Claude Code UI経由のPending approval導線）、A2Vバッチ1行、join、purge dry_run、api_key設定時の再起動、submit_chain単体、cancel/delete、save_job_video、load/unload_pipeline、upload_video/upload_audio、list_jobs。

**2026-08-04追記**: 直前の段落が挙げた「未実施のまま残る項目」11件のうち、**承認ゲート（Claude Code UI経由のPending approval導線）を除く10件を §39.6 で実施し、全項目合格**した。§39.6 の実施方法もこの段落と同じ「エージェントが自前スクリプトからMCPクライアントを叩く」経路であり、**Claude Code本体のUIを経由した確認ではない点も変わらない**（同じ限界を引き継ぐ）。

**§39は実機ゲート未実施の状態でクローズしない。** → **2026-08-04に §39.6 で実施済み**。残るのはClaude Code本体のUIでしか確認できない承認導線1件だけで、これはエージェントからは原理的に検証できない（オーナーが実際にClaude Codeでリポジトリを開いたときに確認する項目）。

### 39.6 実機検証（2026-08-04・オーナーの実施指示による）

オーナーの指示により、稼働中の実バックエンド（`127.0.0.1:18620`・ltx-distilled・RTX 4070 Ti SUPER・**GGUF逆量子化の1カーネル化＝`fused_gguf_dequant_kernel` を既定onへ反転した後のコード**、§51）に対して、§39.5に残っていた未実施項目を検証担当エージェントが実施した。

実施方法は §39.5 の「暫定✅」と同じ経路で、**MCPの stdio トランスポートで `.venv\Scripts\python.exe -m mcp_server` を子プロセスとして起動し、`mcp` パッケージのクライアントから実プロトコル往復（`initialize` → `tools/list` → `tools/call`）でツールを呼ぶ**。検証スクリプトはリポジトリ外の作業フォルダに置いたので、リポジトリには何も足していない。生成条件はいずれも軽量（384×256・17〜33フレーム・24fps＝`generation_presets.smoke_test` 相当）にしてGPUの占有を最小限にした。

#### 検証結果＝実施した全項目が合格

| 項目 | 使ったツール | 結果 |
|---|---|---|
| 22ツール列挙 | プロトコルの `tools/list` | ✅ ちょうど22個。説明文が空のツールは0件（§39.4のpytestと同じ集合を実プロセスでも確認） |
| stdout清浄性の再確認 | `python -m mcp_server < NUL` | ✅ 標準出力**0バイト**・終了コード0（§39.3から退行なし。§40以降の改修が入った現行コードでも維持） |
| 状態系5本 | `backend_status` / `get_config` / `list_models` / `list_loras` / `list_jobs` | ✅ 実サーバーの応答をそのまま返す（`pipeline_loaded: true`、`acceleration.sage_available: true`、`block_swap_prefetch_available: true`、モデル台帳4カテゴリ、ジョブ一覧と件数集計） |
| T2V submit+poll（既定値） | `submit_generate` → `wait_for_job` | ✅ job `a1a85190-a62f-41e2-8e4e-9358326198a3`（70.8秒・出力23,262バイト）。メタデータのechoは **`fused_gguf_dequant_kernel_used="on"`**（既定on＝§51.5の反転がMCP経路にも効いている） |
| 新引数を明示offで指定 | 同上（`fused_gguf_dequant_kernel=false`） | ✅ job `616590bc-d616-4911-8368-d1a2716c2a37`。echoは **`"off"`**。同一シード（1234）の出力`output.mp4`は **既定onのjobとSHA256完全一致**＝§51の「ビット単位で不変」をMCP経路からも再確認 |
| I2V | `upload_image` → `submit_generate`（`conditioning_images`） | ✅ job `7da5f00d-0c21-4615-849c-64a59151ee09`（53.7秒）。384×256のPNGをアップロードしフレーム0に条件付け |
| JOB_BUSY挙動 | ジョブ実行中に `submit_generate` | ✅ `JOB_BUSY: A job is already running (Phase 1 allows one concurrent job)` がツールエラーとしてクライアントへ伝わる |
| `wait_for_job` のタイムアウト分岐 | `wait_for_job(timeout_sec=45)` | ✅ 45秒で `timed_out: true` ＋ hint「同じ引数でもう一度呼んでください」を返し、**エラーにせず**同じ引数の再呼び出しで続きから待てる（全生成ジョブで2回以上この分岐を通った） |
| 出力の取得と保存 | `get_job_video_path` / `save_job_video` | ✅ 絶対パス・`exists: true`・実サイズを返し、任意フォルダへコピーできる（`no_clobber` の既定で連番回避） |
| `submit_chain` 単体＋V2V | `upload_video` → `submit_chain(source_video_id=…)` | ✅ job `c6487fb9-e0d8-4026-9f9e-3cf14e81ef1a`（78.4秒）。既存の生成物をアップロードし、`context_frames=25` / `clips[0].num_frames=33` で継続生成 |
| join | `join_job` → `get_joined_video_path` → `save_job_video(which="joined")` | ✅ `joined.mp4` が生成された（199,683バイト・`join_mode="handle_crossfade"`・`finished: true`）。`get_joined_video_path` は `joined: true` を返す |
| A2Vバッチ1行 | `plan_a2v_batch` → `upload_audio` → `submit_chain(source_audio_id=…)` → `wait_for_job` → `save_job_video` | ✅ job `8df6d50a-ba37-4159-9d64-54d6643c5927`（78.9秒）。計画→アップロード→投入→待機→保存の一連が計画書どおりの部品接続で通った |
| バッチ計画の走査規約 | `plan_a2v_batch`（HTTP不使用） | ✅ mtime昇順・`image_dir` の同stem画像の自動紐付け・`skip_reason` の2種（21秒のwavが `over-481f`、mp3が `wav-only-alpha`）をいずれも仕様どおり確認。`counts` は total 3 / plannable 1 / skipped 2 |
| cancel | `cancel_job` | ✅ running中の呼び出しで `cancel_requested: true` ＋ ベストエフォートである旨の note。**実際にもジョブは最終的に `cancelled` になった**。終端後に再度呼ぶと `JOB_ALREADY_TERMINAL: … — delete_job を使ってください` で拒否 |
| delete | `delete_job` | ✅ `deleted: true`、以後 `job_status` は `JOB_NOT_FOUND`（対象はこの検証で作った使い捨てジョブ `4c1e0d81-c98f-471b-a700-eb872404f702`のみ） |
| purge dry_run | `purge_terminal_jobs(dry_run=true)` | ✅ `attempted: 5 / deleted: 0 / dry_run: true` を返すだけで、**ジョブ件数も `outputs/` のフォルダも一切変化なし**（前後の `list_jobs` が同一） |
| load / unload | `load_pipeline` / `unload_pipeline` | ✅ 読み込み済みでの `load_pipeline` は POST せず `no_op: true`。`unload_pipeline` で `pipeline_loaded: false` / `state: "unloaded"`、再 `load_pipeline` で `state: "ready"`（**検証前の「読み込み済み」状態へ復帰させて終了**） |
| エラー系8件 | 各種 | ✅ 下表 |
| api_key設定時の動作 | 別プロセス＋Bearer | ✅ 下記 |
| 承認ゲート（Claude Code UI） | — | ⬜ **検証不能**（下記「検証できなかった項目」） |

#### エラー系8件＝すべて意図した1行メッセージへ翻訳される

計画D7の「非2xxはすべて `CODE: message — detail` の単一 `ToolError` にする」が実経路でも成立することを確認した。サーバー側で弾かれるものと、POSTの前にMCP側で弾くものの両方を含む。

| 入力 | 返ってきたエラー |
|---|---|
| `num_frames=20`（8n+1でない） | `VALIDATION_ERROR: Request validation failed — [… 'num_frames must be 8n+1' …]` |
| `width=300`（64の倍数でない） | `VALIDATION_ERROR: … 'width must be a multiple of 64' …` |
| `crop_width` だけ指定 | `CROP_SIZE_INCOMPLETE: crop_width と crop_height は両方指定するか、両方省略してください`（**POST前にMCP側が拒否**） |
| 存在しないjob_id | `JOB_NOT_FOUND: job not found: …` |
| 存在しない画像ファイル | `FILE_NOT_FOUND: …`（**アップロード前のローカル事前チェック**） |
| 未完了ジョブの `save_job_video` | `VIDEO_NOT_FOUND: … が見つかりません (which=output、ジョブが未完了か join未実行の可能性があります)` |
| 存在しないwavフォルダ | `WAV_DIR_NOT_FOUND: …` |
| クリップ1件だけの通常チェーン | `VALIDATION_ERROR: … 'chain requires at least 2 clips' …` |

#### api_key（Bearer認証）の検証方法と結果

**オーナーの稼働中サーバーは止めずに**、別ポートで2つ目のバックエンドを一時起動して確認した（`--config` に `port: 18621`・`log_dir` を作業フォルダに変えた設定の写しを渡し、`--api-key testkey123` 付きで起動）。MCPサーバー側は `LTX_MCP_BASE_URL` / `LTX_MCP_API_KEY` の環境変数でそこへ向けた（§39.1の設定解決の経路そのもの）。確認後、この2つ目のプロセスは停止し、稼働中サーバーが無事であることも確認済み。

- **`/status` は認証の対象外**である点が実測でわかった。`require_auth`（`api/deps.py`）が付いているのは生成・ジョブ・アップロード・パイプラインの各ルートだけで、`GET /status` には付いていない。したがって `backend_status` だけではBearerの成否を判定できない。そこで**保護されたルートを使う** `upload_image`（`POST /upload/image`）で判定した。
- 正しい鍵: `upload_image` が成功（`image_id` が返る）。`backend_status` の `api_key_configured` も `true`。
- 誤った鍵（`wrong-key`）: `UNAUTHORIZED: Invalid or missing API key` に翻訳されて返る。
- 空文字の鍵（`LTX_MCP_API_KEY=""`）: `api_key_configured` は `false`（設定解決の「環境変数が空なら鍵なし扱い」が効いている）、保護ルートは `UNAUTHORIZED`。**期待どおりの挙動**。

#### 新規発見＝実バックエンド稼働中はpytestが1件だけ落ちる（環境依存・コードの退行ではない）

検証のついでにMCP関連のpytest（7ファイル・98件）を回したところ、**`tests/test_mcp_registration.py::test_backend_status_structured_content_not_wrapped_and_reachable_false` の1件だけが失敗**した（他97件はPASS）。原因はこのテストが「バックエンドが**起動していない**こと」を前提に `reachable: false` をアサートしている点にあり、実バックエンドが既定ポートで動いていると `reachable: true` になって落ちる。`LTX_MCP_BASE_URL=http://127.0.0.1:18999`（未使用ポート）を与えて同じテストを回すとPASSするので、**コードの退行ではなく実行環境の前提のずれ**と確定した。§39.2の「736 tests 全PASS」も実バックエンド未起動時の計測である（§49.5と同じ注記）。テスト側を環境非依存にする改修は本検証の範囲外なので、事実の記録にとどめる。

#### 検証できなかった項目（粉飾せずに残す）

- **承認ゲート（Claude Code UI経由のPending approval導線）**: エージェントからは**検証不能**。Claude Codeでリポジトリフォルダを開いたときのワークスペース信頼確認と、プロジェクトスコープMCPサーバーの承認ダイアログは、Claude Code本体のUI操作でしか発生しない。同じ理由で `/mcp` コマンドの実画面での22ツール表示も未確認のまま（**ツール集合そのものは上表のとおりプロトコル越しに22個ちょうどで確認済み**なので、残るのは画面表示の目視のみ）。§39.5の「暫定✅」がもともと抱えていた限界と同一で、今回もそこは動いていない。
- **`.mcp.json` を使った実際のClaude Code起動**: 上と同じ理由で未確認。ただし `.mcp.json` の中身（絶対パスの `python.exe` ＋ `-m mcp_server` ＋ `PYTHONUTF8=1`）と等価なコマンドラインで子プロセスを起動して全ツールを叩いているので、**設定ファイルが指す起動方法そのものは動作する**ことは言える。

#### この検証で `outputs/` に残ったジョブ

いずれも軽量な検証用の生成物で、削除して構わない（オーナー判断）。`purge_terminal_jobs` の dry_run はこの5件を対象として数えた。

| job_id | 種別 | 備考 |
|---|---|---|
| `a1a85190-a62f-41e2-8e4e-9358326198a3` | t2v | `fused_gguf_dequant_kernel_used="on"`（既定） |
| `616590bc-d616-4911-8368-d1a2716c2a37` | t2v | 同 `"off"`。上と出力SHA256一致 |
| `7da5f00d-0c21-4615-849c-64a59151ee09` | i2v | 条件付け画像あり |
| `c6487fb9-e0d8-4026-9f9e-3cf14e81ef1a` | chain（V2V） | `joined.mp4` も同フォルダにある |
| `8df6d50a-ba37-4159-9d64-54d6643c5927` | chain（A2V） | 検証用の合成wav（1.3秒）から生成 |

なお `cancel` / `delete` の検証に使ったジョブ `4c1e0d81-c98f-471b-a700-eb872404f702` は `delete_job` で記録ごと削除済み（出力フォルダも残っていない）。

#### 注意記録

- 今回の2本のT2V（fused on 70.8秒 / off 92.0秒）は**速度比較として読んではいけない**。単発1本ずつの計測で交互対比較になっておらず、直前のジョブの残留状態にも影響される。速度の正本は§51.4の交互対比較である。
- V2Vの `source_video.context_frames` は **25以上・8n+1・`clips[0].num_frames` 未満**という3条件があるため、17フレームの短い動画は継続元にできない（`context_frames must be >= 25` で弾かれる）。今回は既存の長めの生成物を継続元にした。この制約はツールの説明文には「8n+1・既定73」までしか書かれていないので、短い動画で試すと最初の1回は必ずエラーになる。

---

## 40. ★依存整理バッチ（未使用パッケージの削除＋`uv sync --extra dev`常時化＋`checkpoint_path`後始末）＝実装完了・機械検証全PASS・**実機SHA回帰 全項目合格（2026-07-28）**

> **正本＝本節。** フロントエンド側の起票・経緯は [`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §1-5・§3-25・§3-26（本節との対応関係もそちらに明記）。着手はNAG（§38）の実機ゲート完了後の別バッチとして実施した（同時実施だとSHA不一致が出た場合にNAGと依存削除のどちらが原因か切り分けられなくなるため）。

### 40.1 未使用依存パッケージの削除（`.venv-engine`）

`.venv-engine`の依存表に残っていた、2026-06-30のフォーク取り込み（`4ea5c4b`）由来で消費側コードが翌日の大掃除（`d0d3df5`）で削除済みの「フォーク由来の化石」を棚卸しし、確認済み未使用10件＋追加調査で特定した確実な孤児9件＝**計19エントリ**を`engine/engine-venv-pyproject.toml`と`engine/venv-engine.freeze.txt`から削除した。

**確認済み未使用10件（内訳）**: `sageattention` 1.0.6（77KB）／`triton-windows` 3.6.0（129MB）／`imageio`＋`imageio-ffmpeg`（計約85MB）／`peft`（2.3MB）／`fastapi`・`uvicorn`・`python-multipart`のエンジンvenv側重複コピー（計約1MB）／`pynvml`／`ftfy`。

**追加の確実な孤児9件**: 同じ棚卸しの過程で、上記10件とは別の観点（コードから一度もimportされない・消費側が既に削除済み）で特定した9パッケージ。残り9件は上記10件と同じ基準（全ソースgrepで参照ゼロを確認）で選定した。

**sageattentionも削除する理由**: 入っているのは旧世代1.0.6で、将来の高速推論モードは現行世代（SageAttention 2系）の新規選定になるため温存価値がない。フォーク元のコード自体に「有効化すると2倍遅くなる原因を調査中（既定OFF）」と記されていた。

> **2026-07-31追記（`sageattention`／`triton-windows` の再導入）**: 上記19エントリのうちこの2つは、Acceleration（生成の高速化）機能の実装にともなって**エンジン用仮想環境へ戻した**（§43.3）。**当時の判断と矛盾はしない。** 2026-07-28にここで削除したのは旧世代の `sageattention` **1.0.6**＝コードから一度も import されない死重依存であり、再導入したのは現行世代の **2.2.0**（torch 2.9.1+cu128 向けのビルド済み wheel を直リンクで固定）で、`engine/transformer/sage_attention_service.py` という**実際の消費者がある**。「実消費者のない依存は置かない」という本節の基準はそのまま維持されている。上の一文が予告していた「将来の高速推論モードは現行世代の新規選定になる」という見立てが、そのとおりに実現した形である。

**保留（削除しない）**: `sentencepiece`／`protobuf`の2つ。Gemmaトークナイザのフォールバック経路で使われる可能性を静的解析で否定しきれず、壊れたときの症状（トークナイザロード失敗）が致命的なわりに節約が小さい（2026-07-28オーナー決定）。

**サイズ・エントリ数の変化**: `engine/venv-engine.freeze.txt`のエントリ数は**69→50**。`.venv-engine`の実ディスク使用量は約230MB縮小（**4.98GB→4.77GB**）。

### 40.2 `uv sync`が「任意の依存」を黙って刈り取る問題への対処

アプリ用仮想環境（`./.venv`）で、`uv sync`（`--extra dev`無し）を実行すると`dev` extra（`pytest`・推移的依存の`iniconfig`／`pluggy`）が**警告なく削除される**仕様が確認されていた（`uv sync`は要求された状態へ環境を合わせにいく道具で、明示しないextraは「入っていてはいけないもの」とみなして刈り取るため）。

**対処（実施済み）**: `scripts/install_ltx.ps1`のアプリvenv同期ステップを常に`uv sync --extra dev`にした。従来は`-RunSmoke`指定時だけ追加で`uv sync --extra dev`を走らせる特別扱いだったが、ステップ本体が常時`--extra dev`になったことでこの特別扱いは冗長になったため撤去した。数MBの追加でエンドユーザー環境にも`pytest`が入るが、「`git pull`後に`setup.bat`を再実行するとテストが消える」罠を仕組みで塞ぐことを優先した（実害の実績: 2026-07-28のNAG実装作業でもこの罠を踏み、手動`uv sync --extra dev`での復旧が必要だった）。

### 40.3 `checkpoint_path`の後始末

`checkpoint_path`（物理削除済みの43GBモノリスを指す参照専用キー）を`config.py`の`ModelConfig`・`config.yaml`・`config.yaml.example`の3箇所から完全に削除した。

調査の結果、`config.py`のコメントにあった「`DistilledPipeline`のシグネチャのために保持している」という制約は外せると確定した。`ModelLedger.build_model_builders`が要求するのは「`None`でない文字列」であることだけで、その文字列が実際に開かれることは無い（GGUF＋componentファイル経路では一切参照されない）。そのため`services/ltx_runner.py`側で、worker payloadへ渡す`checkpoint_path`を直値の`""`にハードコードするよう変更し（証跡コメントをコード上に残した）、設定削除の前後でworker payloadがバイト同一であることを確認した。

### 40.4 機械検証の結果

- **エンジンimport確認**: `.venv-engine`再構築後、`engine`パッケージ一式のimportが正常に通ることを確認。
- **`engine.transformer.nag_selfcheck`**: 6/6 PASS（§38.3と同一の自己検証。依存削除後も退行なし）。
- **アプリvenv pytest**: 736 tests / 730 passed / 6 skipped（§39.2と同数。退行ゼロ）。
- **`uv sync --extra dev`の存続確認**: セットアップ経路を再実行しても`dev` extra（pytest/iniconfig/pluggy）が消えないことを確認（§40.2の対処が機能している）。

### 40.5 判明した2点（NOTE）

1. **インストーラのfreeze適用は加算的（additive）**: `install_ltx.ps1`のfreeze同期は「無いものを入れる」動作であり、「入っているものを消す」動作ではない。したがって、この変更より前に`.venv-engine`を作った既存ユーザーの環境からは、削除対象の19パッケージは自動では消えない。実害の無い残留物として残るのみで、新規インストール環境は最初からクリーンな状態になる。
2. **`torchvision`は要調査のまま保留**: メタデータの依存グラフ上は必須の要求元（requirer）が見当たらないが、明示依存かつ既存の記載であるため今回の削除対象には含めなかった。将来あらためて確認する対象として記録する。

### 40.6 オーナー実機チェックリスト（実GPU SHA回帰・全項目合格）

依存削除・`uv sync`変更・`checkpoint_path`削除のいずれも「出力に触れないはずの変更」であることを、実GPUでのバイト一致回帰で最終確認する。

| # | 内容 | 合格条件 | 状態 |
|---|---|---|---|
| 1 | T2V回帰 | 既存の基準seed/paramsで生成し、出力mp4のSHA256が変更前の基準と一致・peak VRAMも同値 | ✅ 合格（2026-07-28。job `b1645c1d-bdfa-441d-922e-39ba90ae90e1`、SHA256 `23844b4eebd107ccba8c5534eb65bab86575cca0b9050cb6c7e680a4506bb7bf`＝基準と完全一致、peak_vram_mb **8440**＝基準一致） |
| 2 | 最小I2V回帰 | 同上（最小I2Vの基準ケースで実施） | ✅ 合格（2026-07-28。job `3d231fc1-4cad-49d8-9026-8ddf45419f11`、SHA256 `a511eda431cf0d0942cee97fa130f45e55fc3236833cbf9ea743ea7f4715c217`＝基準と完全一致、peak_vram_mb **9525**＝基準一致） |
| 3 | 合格後の後始末 | 1・2が合格したら、フロントエンド側台帳[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §1-5／§3-25／§3-26をクローズする | ✅ 実施済み（2026-07-28、本節作成と同日にフロントエンド側台帳をクローズ。[`PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md)へ移設済み） |

**2026-07-28追記（実GPU SHA回帰PASS）**: オーナー承認のもと、エージェントがMCPサーバー（`mcp_server/`）経由（`mcp==1.28.1`のstdioクライアントスクリプト、実バックエンド＝ltx-distilled、GPU＝RTX 4070 Ti SUPER）で実施。Claude Code UIの承認フローは通していない（この点は§39の暫定注記を参照）が、生成そのものは実プロセス・実GPUを通しており、出力SHA256とpeak_vram_mbの一致という合否基準に照らして**正式PASS**として扱う。

- **T2V**: job `b1645c1d-bdfa-441d-922e-39ba90ae90e1`。パラメータ＝基準（prompt "a calm ocean wave..."／512×320／49f／8steps／seed12345）。リクエストエコーで`nag_enabled=false`を確認（§38 NAG機能の非干渉も同時に裏付け）。出力SHA256は§13.1以来の基準値と完全一致・peak_vram_mbも基準一致。
- **最小I2V**: job `3d231fc1-4cad-49d8-9026-8ddf45419f11`。条件画像は`outputs/qat_reclaim_baseline/cond_image_512x320.png`をmanifestとバイト一致確認したうえで`upload_image`ツール経由でアップロードしたもの。出力SHA256・peak_vram_mbともに基準一致。
- **副産物（JOB_BUSY実地確認）**: I2Vジョブ実行中に別ジョブ投入を試みたところ、ライブで409「`JOB_BUSY: A job is already running (Phase 1 allows one concurrent job)`」を受信・確認（§39のJOB_BUSY行の実地裏付けを兼ねる）。
- **この回帰が持つ副次的な意味**: これらの基準SHA自体はNAG（§38）実装より前から存在する値のため、今回のバイト一致は「依存整理（§40.1〜40.3）が出力に触れていないこと」だけでなく、「単発T2V／最小I2Vにおいて NAG OFF 時の出力がバイト不変であること」も同時に裏付けている（＝§38.4 G0 の単発T2V／I2V分に相当。A2V／chain／バッチのG0は未実施のまま）。

**§40は実機SHA回帰が未実施の状態でクローズしない。** →**2026-07-28時点でチェックリスト1〜3すべて合格・完了。** 依存整理バッチ（§40.1〜40.3）の実機確認はこれで完結する。

> **2026-07-31追記（§43との関係）**: 2026-07-31の Acceleration 機能で `sageattention`／`triton-windows` の2件を再導入した（§40.1の追記・§43.3）。この依存構成の変更に対する実機確認は、**本節と同じ形のSHA回帰を繰り返すのではなく、§43 の G9（インストーラを再適用して完走し `sage_available: true` になること）で吸収・代替する**。再導入した2パッケージは既定のジョブでは import すらされない位置にあり、既定経路が従来とバイト同一であることは §43.4 のペイロード完全一致テスト群が構造的に固定しているためである（判断の根拠は §43.7）。

---

## 41. ★VSF（Value Sign Flip）による非CFGネガティブプロンプト第2方式＝実装完了・機械検証（selfcheck・pytest）全PASS・**全ゲート合格・テーマ完結（2026-07-29）**（実用域はscale1.5〜5。AdaLNモードのデバッグスイッチは§41.10で撤去済み。フロントエンド目視ゲートまでの完結は§41.11）

> **正本＝本節。** NAG（§38）に続く2つ目の非CFGネガティブプロンプト手法。台帳 `PENDING_TASKS.md` §3-46で調査・起票していたもので、着手条件（NAGの実機確認全合格）は2026-07-29に成立し、同日中に実装・機械検証・実機ゲート（機械検証分）まで完了した。オーナー承認済みの計画書（`reactive-weaving-umbrella.md`、Opus調査2本＋敵対的レビュー2ラウンドを経た最終版）に沿ってWave 0（chain基準SHA取得）→Wave 1（エンジン）→Wave 2（API/MCP）→Wave 3（Gradio UI）→Wave 4（敵対的コードレビュー＋全テスト）→Wave 5（MCP経由の実機テスト）→Wave 6（本節・ドキュメント）の順で実施した。範囲はバックエンド＋Gradio UI（React側フロントエンドは対象外・未着手）。

### 41.1 設計根拠

1. **中核式**（arXiv:2508.10931）: `Z = softmax(Q·[K⁺;K⁻]ᵀ/√d) · [V⁺; −α·V⁻]`。正負のコンテキストを連結し**1回のattention**で処理する点がNAG（正負2回計算して外挿・混合）と根本的に異なる。負側のV（value）だけを−α倍（符号反転×スケール）する。softmax分母を正負で共有すること自体が「正側の減衰＋負側の符号反転」の二重作用を生む本質で、NAGのような正規化・ゲート機構は設計上存在しない。
2. **最大のリスクは質量不均衡**: attn2（cross-attention）に届く正コンテキストは常に(B, 1024, 4096)（実トークン＋学習済みlearnable register。実トークンは先頭詰め）。負側は実トークン数Nにスライスすると通常十数トークンしかなく、負側が奪えるsoftmax質量mは素朴には1%オーダーになりうる。「実装は成功するが効かない」が最大の失敗様式と想定し、対策として①負側softmax質量mの直接ログによる効き検出②`vsf_scale`の上限を論文実装の10から100へ拡大（出力の負項は−α·m·V̄⁻なので、mが小さくてもαで一次補正できる）の2点を設計に組み込んだ。
3. **効き検出はmログ、動画PSNRは使わない**: NAG実測でON/OFFペアがPSNR 15.5dB／12.0dB（ほぼ別動画相当）だったことから、拡散過程は微小摂動でも軌道が発散し閾値を置けないと判断済み（§38関連の実測）。VSFでも同じ理由でPSNRを合否判定に使わず、負側softmax質量mの実測ログを一次指標とした（目視比較の物差しとしてPSNR値は併記する）。
4. **β（負側ロジットへの加算バイアス）は第1弾から除外**: 論文の第2ノブだが、一次効果はα（scale）で代替可能・floatマスクをSDPAに渡すとflashカーネルが外れ性能/VRAMが未知・全レイヤー貫通のフィールド増という3点から見送った。mの実測後、αで不足と判明した場合に限り第2弾として追加する設計（全フィールド末尾追加運用のため後付けコストはゼロ）。
5. **負側スライスは無条件・エンコード時に実施**: パディング位置には学習済みregisterが実データとして詰まっており、反転混入は明確に有害。トークナイザの重み合計から実トークン数Nを求め、本番エンコードと厳密一致することを確認済み。スライスはエンコード地点（方式を知っている場所）で行い、状態には**スライス済みテンソルをそのまま**格納する。音声connectorも同一構造のため`audio_attn2`にも同じスライスを適用する。
6. **AdaLN非対称は第一級の実験対象**: 正コンテキストのみ`apply_cross_attention_adaln`で毎ステップ変調される。論文はこの論点に無言のため、`raw`（負は生のまま。NAGと同じ非対称）／`modulated`（負も同じ変調を通す）／`v_scale`（Kは生のままVのみ変調係数でスケール補正）の3モードを実装しデバッグ用ラジオとして公開、実機A/Bで判断材料を得る方針にした。`apply_cross_attention_adaln`はforward時にLOAD_GLOBAL解決されるモジュールグローバルのため、パッチ窓はdenoise全体を覆う必要がある（モデル構築呼び出しだけを囲むとパッチが一度も呼ばれない失敗様式）。
7. **既存NAGインフラを流用**: `NagState`・`encode_negative`・`_cross_attn_modules`・`install`・per-headゲート＋`to_out`テール・fail-loudガードを共有。`q_norm`/`k_norm`はRMSNorm（位置独立）のため射影後個別正規化→連結は厳密に等価。attention実体はSDPA（xformers/flash_attn不在を確認済み）でK/V長≠Q長を許容する。
8. **OFF時無害はパッチ0件の構造保証**（NAGと同じ設計思想）。`vsf_scale=0`によるビット一致は原理的に不成立のため、そのようなゲートは作らない。

### 41.2 API（3フィールド）

`GenerateRequest`/`GenerateChainRequest`両方＋`to_clip_request`転記＋MCP `submit_generate`/`submit_chain`末尾に以下3フィールドを追加した。

- `neg_method: Literal["nag","vsf"] = "nag"`（既定はNAGのまま。既存挙動は不変）
- `vsf_scale: float`（範囲0〜100・既定1.5。論文実装の上限10ではなく100に拡大——1024:Nの質量不均衡下で不足しうるため。UIに「0でも無効化にはならない」と明記）
- `vsf_adaln: Literal["raw","modulated","v_scale"] = "raw"`（デバッグ用と明記）
- `nag_enabled`は非CFGネガのマスタートグルとして維持し、空negativeの422バリデータは無改修で両方式をカバーする。
- 見送った候補: `vsf_offset`（β。第2弾へ延期）、`vsf_slice_pos`（正側register除去は学習済みインタフェースの9割を消すOOD操作で結果が解釈不能。同じ問いにはmログが上位互換で答える）。

### 41.3 実装箇所一覧

- **`engine/transformer/vsf_service.py`（新規）**: `VsfParams`（negative_prompt/scale/adaln_modeのfrozen dataclass。未知adaln_modeはfail-loud）、`_make_vsf_forward`（NAGと同じfail-loudガード・同じテール。連結1回attention本体はマスクを渡さない設計）、負側softmax質量mの平均をINFOログに出力する仕組み（N/Lk/shift_kv.shapeも同時ログ）、`adaln_stash_window()`（contextmanager。`apply_cross_attention_adaln`を退避→係数スタッシュ付きラッパへ差し替え→finallyで復元＋delattr。開くのはVSF要求時かつmode≠rawのときのみ）。importは`vsf_service→nag_service`の一方向のみ（循環import回避）。
- **`engine/transformer/nag_service.py`（拡張）**: `NagState._params`の型をunion化（`NagParams | VsfParams`）。`encode_negative`にオプション引数`slice_to_real_tokens: bool = False`を追加（既存呼び出しは無変更で挙動不変）。
- **`engine/pipeline/fast_video_pipeline.py`**: `_install_nag`ラップ内で`isinstance(params, VsfParams)`により`VsfService`/`NagService`を1行分岐。単発生成の`self.pipeline(...)`実行（denoise全体）を`with adaln_stash_window():`で覆う。`compile_transformer`の非互換コメントの主語を「NAG」→「NAG/VSF」に更新。
- **`engine/pipeline/chain_pipeline.py`**: `run_chain`の`ledger.transformer()`取得から全セグメントのdenoise・アップサンプル完了までを同じ`adaln_stash_window()`で覆う（`_run_inference`の既存グローバル5本傘には手を入れない）。
- **`engine/worker.py`**: `_resolve_nag`が`nag.method`欠落時`"nag"`にフォールバック。ログを`neg=off|nag|vsf`に拡張。
- **`api/models.py`**: 上記3フィールドを`GenerateRequest`/`GenerateChainRequest`双方に追加し、`to_clip_request`へも転記（漏れるとチェーン500になるため重点確認済み）。
- **`services/ltx_runner.py`**: `payload["nag"]`に`method`/`vsf_scale`/`vsf_adaln`を加算（単発・chain両方、`if nag_enabled:`の内側のためOFF時はバイト不変）。
- **`mcp_server/tools/generate.py`**: `submit_generate`/`submit_chain`のシグネチャ**末尾**に3引数を追加（既存テストが全位置引数のため途中挿入を避けた）。日本語`Args:` docstring（inputSchema生成元）も同時更新。
- **`gradio_ui/ui.py`**: 方式ラジオ「Other」→「VSF」に変更。NAG用3スライダとVSF用グループ（`vsf_scale`スライダ・デバッグ用Accordion内にAdaLNラジオ3択）を`on_nag_method_change`のvisible出し分けで切替。言語切替再構築にも対応。
- **`gradio_ui/handlers.py`**: 単発`generate`は末尾追加、chain `generate_chain`は契約どおり`nag_alpha`と`src_audio`のあいだに挿入。payload加算箇所は`if nag_enabled:`ブロック内に方式に関わらず常に`neg_method`/`vsf_scale`/`vsf_adaln`を追記。
- **`gradio_ui/batch.py`**: `BatchSnapshot`へキーワード追加＋転記。
- **`gradio_ui/i18n.py`**: en/ja両方に`nag_method_vsf`/`vsf_lbl_scale`/`vsf_lbl_adaln`等を追加、廃止キー`nag_method_other`/`nag_msg_fallback`を削除。
- **`engine/transformer/vsf_selfcheck.py`（新規）**: `.venv-engine`用のエンジンvenvセルフチェック（5項目、41.4参照）。

### 41.4 機械検証の結果

**エンジンvenvでのselfcheck**: `vsf_selfcheck` 5/5 PASS（連結1回attentionが手書き参照式と一致／エンコード時スライスの正当性／AdaLN 3モードが実物`BasicAVTransformerBlock`経由で各々参照式と一致・モンキーパッチが実際に呼ばれることまで検証／OFF時パッチ0件＋`attn.forward`と`apply_cross_attention_adaln`グローバルのidentity検査／fail-loud＝形状ガード・未知adaln_mode・`shift_kv`shape[1]!=1）。`nag_selfcheck`回帰6/6も維持（退行なし）。

**アプリvenvでのpytest**: 763 passed / 6 skipped（skipは既存の`torch`未導入によるエンジン系テストの収集スキップ、VSF関連ではない）。意図的に更新した既存テストは計画どおり4本のみ（フォールバック→visible出し分けテスト、choices期待値、i18nキー一覧、payload末尾契約`[-4:]`→`[-7:]`）で、それ以外は加算的拡張。

**その他**: `.gitignore`の`tools/`パターンが`mcp_server/tools/`（MCPツール実装一式）をgitから不可視にしていた致命欠陥を本Wave中に発見・`/tools/`へ修正済み（MCP実装セッション由来の先行欠陥。VSF自体のバグではない）。**オーナーへ: 次回コミット時に`git add mcp_server/tools/`が必要。**

### 41.5 実機ゲート表（2026-07-29・親エージェントがMCP経由で実施）

| ゲート | 内容 | 合格条件 | 状態 |
|---|---|---|---|
| G0（単発） | 回帰: 単発T2V/I2V | 出力SHA256が既存基準（§40.6）と一致 | ✅ 合格（T2V=`23844b4e…`一致・最小I2V=`a511eda4…`一致） |
| G0（chain OFF） | chain（chunked_upsample=false）の基準一致 | Wave 0で取得した基準SHA `E9E809C0…`と一致 | ✅ 合格 |
| G0（chain chunked） | chain（chunked_upsample=true）の新旧エンジン比較 | 旧エンジンと新エンジンで同一SHA | ✅ 合格（`D37AF079…`同値。chain SHA基準は途中で条件不一致騒ぎがあったが、真因はMCP `submit_chain`の`chunked_upsample`既定がtrue（操作パネル準拠の意図的設計）でWave 0基準（既定false）と条件が違ったこと。退行ではない） |
| VSF疎通 | `neg_method="vsf"`小サイズ1本 | 完走＋エコー`neg_method=="vsf"`＋ログ`VSF installed on 96 cross-attention modules`が1回＋mログ確認 | ✅ 合格（raw時m実測: video約0.8〜1.2%・audio約0.5%。N_neg=8, L_k=1032, 正コンテキスト(1,1024,4096)。質量不均衡の事前予測が的中。mはscale非依存＝理論どおり） |
| AdaLN 3モード | 単発raw/modulated/v_scale各1本＋chain modulated1本 | 完走＋全SHA相互に異なる（実効あり） | ✅ 合格（modulatedはmを約4倍に増加。block1 video 0.84%→3.5%。stashログ`shift_kv=(1,1,4096)`でshape[1]==1も実証） |
| 経路網羅 | chain 2clip・A2V（audio_source=yes）・NAG回帰（neg=nag） | 全て完走・NAG回帰にVSFログが出ない | ✅ 合格 |
| NAG対VSF効き比較 | オーナー指定の基準セット2組でOFF/NAG/VSF比較生成 | 完走・比較セット収集 | ✅ 合格（下記41.6参照） |
| 目視評価 | 上記成果物をオーナーが目視し効き具合・scale適正値・AdaLNモードを判断 | オーナー確認 | ✅ 合格（2026-07-29 完結。既定=raw・実用域scale1.5〜5・上限10縮退済み・全経路成立） |

### 41.6 目視評価セット

比較用の成果物一式をHugging Faceの非公開datasetへアップロード済み: https://huggingface.co/datasets/Rootport/Nz-LTX23-vsf-eval-20260729 （比較セット2組=OFF/NAG/VSF s5/VSF s15、scale梯子1.5/5/15/40、AdaLNモード比較、経路確認、README比較表付き）。オーナーが外出先から確認できるようREADMEに条件（seed・方式・パラメータ・m値・NAGの参考PSNR 15.5/12.0dBを物差しに併記）の比較表を添えている。

性能面の気づきとして、scale=15の1280×768出力はファイルサイズ約42MB（NAG版約6MB）と突出しており、高scaleでの高周波成分増加（ノイズ／破綻の可能性）の兆候として目視で要確認と記録した。

### 41.7 教訓

- **chain系のSHA基準比較は、比較対象のchunked_upsampleの値を必ず明記すること。** MCP `submit_chain`の既定値（true）とWave 0基準取得時の生HTTP既定値（false）が食い違っていたために「G0不一致」と見えた騒ぎが発生したが、条件を揃えたら一致し、退行ではないことが判明した。今後同種の基準比較を行う際は、比較対象のリクエスト全条件（既定値を含む）を明記する。
- **`.gitignore`のパターンは、後から追加したディレクトリと衝突しないか定期的に確認すること。** `tools/`という広すぎるパターンが、後発の`mcp_server/tools/`を不可視化していた。ワイルドカードに近いignoreパターンを書くときは、将来同名のサブディレクトリが生まれる可能性を考慮する。

### 41.8 残タスク（オーナー）

1. ~~HFの動画を目視して効き具合・scale適正値・AdaLNモードを判断する。~~ → 2026-07-29に第1ラウンド実施済み。結果は41.9参照。AdaLNモードはscale5での再実験待ち。
2. ~~判断後に既定値を確定し、デバッグスイッチ（AdaLNラジオ等）を縮退するかどうかを決める（別途承認が必要）。~~ → `vsf_scale`のAPI上限100→10への縮退は決定済み（41.9参照）。AdaLNラジオ自体は縮退第3弾で撤去し、決着した（41.10参照）。
3. βノブ（`vsf_offset`）は、mの実測を踏まえてもαだけでは不足すると判明した場合に限り第2弾として起票する。
4. コミット（`git add mcp_server/tools/`を含む。§41.4のgitignore修正参照）。
5. フロントエンド（React）追随は別セッションで行う（本Waveの範囲外）。

### 41.9 目視評価の結果（2026-07-29 オーナー実施・追記）

HFの非公開dataset（41.6のリンク）をオーナーが実際に目視した結果、部分合格（scale次第で効果あり・高scaleは非実用）という実態が判明した。全滅でも全面合格でもない。

**セット1（雨の街）**: OFF・NAGは従来どおり合格。VSF scale5は「室内で会話する男女」の映像に変化した——正プロンプトからの意味ドリフトで、メタデータでプロンプト自体が正しいことは確認済み。scale15はノイズだらけで収束せず、非実用と判断した。

**セット2（図書館）**: OFF・NAG合格。**VSF scale5はネガティブプロンプトの排除効果が明確に確認でき、合格。** 画像認識でも半袖→長袖・長い黒髪→短い非黒髪への変化を確認した。ただし場面が図書館から屋内家庭風へ、人物も年配の男女へドリフトしている。scale15はセット1と同様にノイズ崩壊した。

**scaleの梯子（1.5／5／15／40）**: 意味のある動画として成立していたのはscale1.5と5のみ（いずれもかなり高品質）。15以上はノイズ崩壊で非実用。scale5では「プロンプトに無い女性が海岸に出現する」というドリフトの兆候も見られた。

**AdaLN 3モード比較（raw／modulated／v_scale）**: 比較に使ったscale15がすでに崩壊領域だったため、3モードいずれも崩壊した映像となり、モード間の優劣は**判定不能**だった。scale5で撮り直して再実験することが決定した（実施中）。

**分析と結論**:
- VSFは予想以上に敏感な技術であることが実測で判明した。実用レンジは**scale1.5〜5**に収まる。
- scale5ではネガの排除力は明確な一方、正プロンプトへの忠実度低下（意味ドリフト）という代償を伴う。これは論文が主張する性格（排除力はNAGより強いが、正忠実度はNAGが上）どおりの結果であり、VSFの設計上の特性として想定内。
- scale15以上は8ステップ蒸留で収束しない。理屈としては、VSFにはNAGのような再正規化機構が無いため、共有softmaxに大きな負のV（value）を混ぜるほど出力ノルムが崩れる。論文の実証レンジ（≤10・実用値は1.5〜1.7）を超えた領域はやはり使えないという今回の実測結果と整合する。
- 41.1で立てた仮説「負側softmax質量m≈1%だから、scaleを大きくして一次補正できる」は、**scaleを上げるより先に忠実度の崩壊が起こるため成立しないこと**が実測で確定した。質量不均衡の懸念は「効かない」ではなく「（scale1.5〜5の範囲で）軽く効く」方向に収まったことになる。

**決定事項**:
1. AdaLN 3モードはscale5で再生成し、判定をやり直す（実施中）。
2. `vsf_scale`のAPI上限を100→**10**へ縮退する（別担当が実施中。既定値1.5は不変）。

**2026-07-29 追記（第2ラウンド・完結）**

**AdaLN 3モードのscale 5再実験の判定（完結）**:
- 3モードとも収束・高品質。ドリフトの強さは raw（海岸＋女性1人）＜ modulated／v_scale（海岸で会話する女性2人。modulatedは透かし風ロゴも出現）。
- modulatedが負側の注意質量mを約4倍にするログ実測とドリフトの強さが整合。
- エージェントの1フレーム目画像認識とオーナーの動画目視が全件一致（「短い動画なら1フレーム画像認識でそこそこ信頼できる」という運用知見も得た）。
- **判定: 既定は raw を維持**（ドリフト最小。NAGと同じ非対称構成がVSFでも最も素直だった）。→ デバッグスイッチ縮退の第3弾（`vsf_adaln`の撤去）へ進むことをオーナーが決定。

**path再実験（scale 5 raw）の結果**:
- 初回のscale 15版は崩壊域だったため配線証明にしかならず、chain 2クリップとA2V（音声駆動）を実用域scale 5で取り直した。
- 両経路とも収束し一貫した映像として成立＝実用域での経路成立を確認。A2Vのリップシンクもオーナー目視で暫定合格。
- 新知見: **両経路とも2Dアニメ調へスタイルが転じた**（chainは夕日の海上で料理する3人のアニメキャラ＝海・夕日の意味核は保持。A2Vは会話する2人のアニメ調女性）。ネガティブの blurry / low quality / distorted から遠ざかる圧が、ノイズやボケの少ないフラットなアニメ表現へ押し出した可能性がある。scale 5のドリフトは「実写→アニメの様式転換」として現れることがある。
- HFデータセットに `path_chain2clip_s5_raw.mp4`／`path_a2v_s5_raw.mp4`／`adaln_s5_modulated.mp4`／`adaln_s5_v_scale.mp4` を追加済み（README更新済み）。

**目視ゲートの最終状態**: §41.5のゲート表の目視行を「✅ 合格（2026-07-29 完結。既定=raw・実用域scale1.5〜5・上限10縮退済み・全経路成立）」に更新。

### 41.10 デバッグスイッチ縮退第3弾（vsf_adaln撤去・2026-07-29完結）

41.9でAdaLNモードの判定が「既定rawを維持・ドリフト最小」で完結したことを受け、オーナー決定によりデバッグ実験用スイッチ`vsf_adaln`を全レイヤー（engine／API／runner／MCP／Gradio UI／i18n／テスト）から撤去した（引き算の原則）。実験の過程そのものは歴史記録としてコミット`f2124e1`に保存済みのため、コード上は消えても経緯は追跡可能。

**縮退後のVSF API**: `neg_method`（`"nag"|"vsf"`）＋`vsf_scale`（0〜10。上限100→10への縮退は41.9で決定済み）の2フィールドに確定した。`VsfParams`もnegative_prompt＋scaleの2フィールドのみとなった。正味570行削減（+328/-898・22ファイル）。

**機械検証**:
- pytest 753 passed（vsf_adaln関連4テスト削除により757→753。新規の失敗・スキップ増なし）。
- `vsf_selfcheck` 5/5（AdaLNモード検査は対象自体が消滅したため削除）。
- `nag_selfcheck` 6/6（NAG側は無影響であることを確認）。
- `chain_pipeline.py`のインデント復元箇所は、空白無視diff（`git diff -w`相当）でロジック混入がゼロであることを確認済み。

**実機再検証（MCP経由）**:
- G0回帰: 単発T2V基準SHA `23844b4e…`が縮退前と一致。
- G0回帰: chain（chunked_upsample=false）基準SHA `E9E809C0…`が縮退前と一致。
- **VSF scale5疎通は、縮退前の同条件ジョブとSHA-256が1ビット一致（`CD9024EF…`）**——`vsf_adaln`撤去がraw計算経路を一切変えていないことをバイト単位で証明した。
- mログ（負側softmax質量）も従来値どおり（video約0.8〜1.2%）を確認。

以上により、VSFテーマは実装・目視評価・デバッグスイッチ縮退まで完結した。残るのはフロントエンド（React／AviUtl2連携UI）の追随のみ（台帳は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-46）。

### 41.11 フロントエンド目視ゲート合格・テーマ完結（2026-07-29）

オーナーがAviUtl2の操作パネルからVSFの実機テストを実施し、全件を合格と判定した。エージェントも3ジョブの冒頭フレームを画像認識で確認し、オーナーの判定と一致した。

- `1a710694`（基準・ネガなし・seed 100652868）: 黒いワンピースの女性、背景に建物群。
- `f88bce4a`（VSF scale1.5・同一seed・ネガ「blurry, low quality, distorted, watermark, text, black outfits, buildings.」）: 服が水色系に変わり、背景の建物が消えた。正プロンプト（雨の公園を歩く女性・木・芝・濡れ）の要素は全て健在。
- `556c35bd`（VSF scale1.5・品質系ネガのみ・seed -1）: 高品質でプロンプトに完全準拠。

同一seedでの比較により、ネガティブプロンプトの排除力と正プロンプトへの忠実度保持が両立することを実証し、既定scale 1.5の妥当性を裏付けた。

これをもってVSFテーマ（バックエンド実装・Gradio UI・デバッグスイッチ縮退・フロントエンド追随・目視ゲート）は**全クローズ**とした。台帳は[`PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md) §3-55（フロントエンド側の完結記録）。

**将来の再訪条件（オーナー決定）**: コミュニティからWan等の動画生成AIにおけるVSFのscaleベストプラクティスの報告が出てきたとき、LTX 2.3への応用可否と既定値1.5の見直しを検討する。現状の1.5は問題ないと判断している。

## 42. ★`POST /upload/video` のリボン範囲トリム（`trim_start_sec` / `trim_duration_sec` 加算＋`cut_range_mp4` 新設）＝実装完了・機械検証（pytest）全PASS・**実機ゲート全項目合格（2026-08-01 オーナー実機確認で完了）**（2026-07-30実装／2026-08-01実機ゲート完了）

> **正本＝本節**（バックエンド側の検証記録）。機能全体の作業指示書はフロントエンド側の[`V2V_RIBBON_TRIM_WORKORDER.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/V2V_RIBBON_TRIM_WORKORDER.md)、台帳は同[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §1-6。AviUtl2のタイムラインに置いた動画オブジェクト（リボン）が元動画ファイルの一部しか占めていないとき、その範囲だけを切り出したmp4をV2Vの冒頭クリップにするための**バックエンド側の受け口**を実装した。凍結表（`LTX23_Backend_Specification.md` §6.1）外エンドポイントへの追加専用拡張であり、`source_tail_seconds`（§37のJoin拡張）と同じ作法に従う。

### 42.1 実装箇所（4ファイル）

- **`services/video_io.py`**: `cut_range_mp4(src, out, start_sec, duration_sec)` を新設した。既存の `cut_tail_mp4` のクローンだが**リサンプルを一切しない**点が決定的に異なる——ユーザーがタイムラインで見ている素材そのものを渡したいので、ソースの実測fps（`probe_fps`）をそのまま `-r` に据え、切り出す窓だけを変える。窓の指定は時間シークではなく `select='between(n\,start\,end)'` のフレーム番号選択で行い（VFR素材でもずれない既存流儀）、`start_frame = round(start_sec * source_fps)` / `n = round(duration_sec * source_fps)` / `end = min(total-1, start+n-1)` と**フレーム空間で完結**させている。音声があれば `atrim=start=…:end=…` で同じ窓へ揃えてAAC再エンコードする。戻り値は `{source_fps, total_frames, start_frame, end_frame, written_frames}`。`FFmpegError` を送出するのは①fpsが計測できない②尺が0フレームに丸まる③開始位置がソース末尾以降——の3ケースのみで、**末尾を超える要求は失敗ではなくクランプ**（残りだけを書く）。
  - 単位に関する意図的なヘッジ: 公開シグネチャは**秒**だが内部で即フレームへ変換しているため、実機調査の結果「AviUtl2の`再生位置`はフレーム単位だった」と判明した場合でも、`start_frame`/`num_frames` を受け取る引数を足して換算を短絡させるだけで済み、選択ロジックは動かさなくてよい（関数のdocstringにも明記した）。
- **`services/video_upload_store.py`**: `VideoUploadStore.save()` にキーワード引数 `trim_start_sec` / `trim_duration_sec` を追加した。判定は純ヘルパ `_trim_window()` に閉じ込め、**「使える窓」でなければ `None` を返して従来の保存経路へ素通しする**——片方だけ指定・非数値・NaN／inf・`start < 0`・`duration <= 0` がすべてここで吸収される。使える窓があるときも、**まず受信バイト列を `input{ext}` へそのまま書いてから**兄弟の一時ファイル（`_input.tmp.mp4`。`_` 始まりなので `path_for` の `glob("input.*")` に決して掛からない）へ切り出し、成功したときだけ `os.replace` で `input.mp4` へ差し替える。したがって`FFmpegError`／`OSError` はすべて「無傷の元アップロードを `trimmed=False` で返す」へ縮退する。mkv/webm等を入力にトリムした場合は保存が `input.mp4` へ正規化され（元の `input.mkv` は削除。`path_for` が一意になるよう `input.*` を1本に保つ）、`content_type` も `video/mp4` に、`size_bytes` は切り出し後の実サイズに差し替わる。
- **`api/uploads.py`**: `upload_video` に `trim_start_sec: float | None = Query(None)` / `trim_duration_sec: float | None = Query(None)` を追加した。**`ge=`／`le=` を意図的に付けていない**——範囲外や非有限の値でバリデーションエラー（422）を新設してしまうと、それまで成功していたリクエストが失敗に変わるため、判定はストア側に委ねて「黙って素通し」に統一する意図をコメントで明記している。あわせて `context.video_upload_store.save` の呼び出しを `run_in_threadpool` 経由へ変更した（トリム経路はffmpegへshell outするので、そのままawaitしないとイベントループ＝他の全APIが切り出しのあいだ止まる）。
- **`api/models.py`**: `UploadVideoResponse` に `trimmed: bool = False` を加算した。既存クライアントは無視するだけで済み、トリム引数なしのアップロードは常に `False` を返す。

### 42.2 機械検証の結果（pytest 776 passed / 6 skipped）

**アプリvenvでのpytest**: `776 passed, 6 skipped`（skipは既存の`torch`未導入によるエンジン系テストの収集スキップで、本件とは無関係）。§41.10時点の753件から23件増（新規2ファイル分）で、既存テストの削除・書き換えはゼロ＝完全に加算的な拡張である。

**`tests/test_video_io.py`（`cut_range_mp4` の単体、5テスト＋パラメータ展開3ケース）**:

| 検証内容 | 合格条件と結果 |
|---|---|
| フレーム精度 | 10fps・6色フレームの素材で `[0.2s, +0.3s)` を要求 → `start_frame=2` / `end_frame=4` / `written_frames=3`、`frame_count(out)==3`。さらに**出力の各フレームを`extract_frame_at`で取り出して平均色を突き合わせ、元素材のフレーム2・3・4であることを色で確認**（「3フレーム書けた」だけでなく「正しい3フレームを書いた」ことの検証） |
| 末尾超過のクランプ | 6フレーム素材のフレーム4から10秒を要求 → 失敗せず残り2フレーム（`end_frame=5`）を書く |
| 非正の窓 | `duration_sec` が `0.0` / `-1.0` / `0.01`（10fpsで0フレームに丸まる）の3通りで `FFmpegError` |
| 開始位置がソース末尾以降 | `start_sec=100.0` で `FFmpegError` |
| ソースfpsの保存 | 30fps素材から0.5秒を切り出し → `source_fps≈30`・**出力の実測fpsも≈30**（リサンプルしていないこと）・`frame_count==15` |
| 音声窓の一致 | 24fps・2.0秒・440Hzトーン付き素材から `[0.5s, +1.0s)` → `start_frame=12` / `end_frame=35`、`frame_count==24`、音声ストリーム存置、かつ**出力の実測尺が1.0秒±0.15**（音声が丸ごと通っていない＝同じ窓へ切られていること） |

**`tests/test_upload_video_trim.py`（エンドポイント契約、6テスト＋パラメータ展開10ケース）**:

- **無トリム時の`cut_range_mp4`未呼び出し＋バイト等価**: `cut_range_mp4` を「呼ばれたら `AssertionError`」のスタンドインへ差し替えた状態でクエリなしPOST → 200・`trimmed=False`・`size_bytes==len(payload)`、かつ**保存ファイルが投稿バイト列と完全一致**。「トリム引数を足したがOFF経路は1バイトも変わっていない」ことを機械的に固定している。
- **トリム引数の素通し確認**: `trim_start_sec=1.0&trim_duration_sec=2.0` → 呼び出し1回・引数が `(1.0, 2.0)` のまま到達・入力は `input.mp4`・出力先の名前が `_` 始まり、応答は `trimmed=True`・`stored_path` が `/input.mp4` 終わり、ディレクトリ内に残るのは `input.mp4` 1本のみ（一時ファイルが片付いている）。
- **片方だけ指定**: `trim_start_sec` のみ／`trim_duration_sec` のみの2ケースとも、`cut_range_mp4` は呼ばれず200・`trimmed=False`・バイト等価。
- **ffmpeg失敗のフォールバック**: 一時ファイルを半端に書いてから `FFmpegError` を投げるスタブで、200・`trimmed=False`・**元アップロードが無傷**・`input.*` が1本・一時ファイルが除去済み。
- **nan / inf / 負値 / 0以下の素通し**: `(nan,2.0)` `(1.0,nan)` `(inf,2.0)` `(1.0,inf)` `(-1.0,2.0)` `(1.0,0.0)` `(1.0,-2.0)` の7ケースで `cut_range_mp4` に**到達しない**ことをスタブのアサーションで固定し、いずれも200・`trimmed=False`・バイト等価。8ケース目の `(1e30, 1e30)`（有限だが荒唐無稽）だけは意図どおりffmpegまで到達し、そこでの失敗が200・`trimmed=False` へ縮退する。
- **mkv入力の正規化**: `clip.mkv` をトリム付きでPOST → `trimmed=True`・`content_type` が `video/mp4`・`stored_path` が `/input.mp4`・ディレクトリの `input.*` は1本のみ・`path_for` の拡張子が `.mp4`。
- **実ffmpegでのE2E**: 10fps・10フレームの実mp4を生成して `[0.2s, +0.3s)` をPOST → 保存ファイルの `frame_count` が実測3・`size_bytes` が実ファイルサイズと一致・元ファイルとサイズが異なる（本当に切られている）。続けて `(1e30, 1.0)` をPOSTし、ffmpeg内で失敗して200・`trimmed=False`・**保存バイト列が投稿バイト列と一致**することも同じテスト内で確認している。

### 42.3 実機ゲート（**全項目合格・クローズ（2026-08-01）**）

**2026-08-01のオーナー実機確認で、フロントエンドの[`REAL_BACKEND_CHECKLIST.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/REAL_BACKEND_CHECKLIST.md) §4.12の全5項目が合格した。** クローズ記録はフロントエンドの[`PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md) **§3-59**（※同書§3-58はAcceleration＝本書§43なので混同しないこと）。

- **①実機採取・解析**: AviUtl2の`動画ファイル`エフェクトの`再生位置`項目の生値は`開始,終了,再生範囲,0`という**4フィールドのCSV**で、先頭2値は**素材時間軸の秒**（プロジェクトfpsに依存しない）と確定した。切り出し尺は`min(リボンの秒数, 終了−開始, 素材の残り)`の3項クランプになった。確定事実の全文はフロントエンドの[`V2V_RIBBON_TRIM_WORKORDER.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/V2V_RIBBON_TRIM_WORKORDER.md) §4。採取用の調査コードは解析完了後に全撤去済み（採取依頼書[`V2V_TRIM_PROBE_GUIDE.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/V2V_TRIM_PROBE_GUIDE.md)は歴史記録になった）。
- **②非退行**（2026-07-31）: リボン＝素材全体のとき従来と同一（無トリム時バイト等価）。§42.2で機械的に固定していた性質が実機でも確認された。
- **③短尺ゲート**: 短すぎるリボンではクライアント側でGenerateが止まり案内が出る。**本エンドポイントに422を新設していない設計（§42.4の教訓）と合わせて、製品UIから422を起こす経路は存在しないままである。**
- **④トリム後Joinの中身連続性**（本改修の決め手）: 末尾を削ったリボンのV2Vで「切られた位置から続き」が生成される。改修前は`source_tail_seconds`が残す末尾と生成の文脈が**元動画ファイルそのものの末尾**になっていた。
- **⑤IC-LoRA参照動画の範囲反映**（2026-08-01に追加した拡張分）: `reference_video_id`も同じ`POST /upload/video`＋同じトリムクエリを通るため、**バックエンドは無改修**で拡張が完結した（フロントエンドの呼び出し側の配線のみ）。リボンでクリップした区間が正しく参照されることを実機で確認した。

### 42.4 教訓

- **既存エンドポイントへ任意引数を足すときは、「不正値でも新しいエラー応答を作らない」を先に決めてから`Query()`を書くこと。** `ge=0` のような一見自然な制約を付けると、それまで200だったリクエストが422になり得る（クライアントが古い・値の導出にバグがある、のどちらでも起こる）。本件は制約をゼロにして判定をストア側の純ヘルパへ移し、「使えない窓＝トリムしない」に一本化した。
- **ffmpegへshell outする処理を`async def`の中で直接呼ばないこと。** `run_in_threadpool` を挟まないと、切り出しのあいだイベントループが止まり、進捗ポーリングを含む他の全APIが無応答になる。同期I/Oを足すときは呼び出し側の非同期性を必ず確認する。

---

## 43. ★Acceleration（生成の高速化）機能＝SageAttention 2.2.0 のジョブ単位切替＋モック2項目＝実装完了・機械検証（selfcheck・pytest・型検査・doctest）全PASS・**実機ゲート全項目合格（2026-08-01 オーナー実機確認で完了）**（2026-07-31実装／2026-08-01実機ゲート完了）

> **正本＝本節。** 操作パネル（AviUtl2連携UI）とGradio UIの設定画面に「Acceleration（生成の高速化）」という区画を新設し、生成そのものを速くする切替を3項目ぶん置いた。3項目のうち**実装があるのは attention（注意機構）の実装選択だけ**で、残る2項目は将来の実装枠として置いた**モック（受理はするが効果が無い）**である。attention の選択肢は `sdpa`（PyTorch標準の実装。既定）と `sage`（[SageAttention 2.2.0](https://github.com/thu-ml/SageAttention)＝量子化を使って注意機構の計算そのものを速くする外部カーネル）の2つ。着手条件は2026-07-31の実機スパイク（本節43.5）で「本環境で安全に動き、720pの生成が実測1.17倍速くなり、VRAMは増えない」ことを確認できたことで、オーナー承認済みの計画書（`wise-waddling-dragon.md`、敵対的レビュー1ラウンドと調査エージェント10件の裏取りを経た rev.2）に沿って STEP 1（エンジン）→ STEP 2（APIスキーマ・runner・能力公開）→ STEP 3（Gradio UI）→ STEP 4（MCP公開）→ STEP 5（インストーラ標準同梱）→ STEP 6（フロントエンド）→ STEP 7（本節・ドキュメント）の順で実施した。

### 43.1 決定事項

1. **既定は `sdpa` のまま据え置く**。理由は3つある。①**アップデートで生成結果を黙って変えない**——`sage` は数値精度が異なるため、同じシードを指定しても生成結果の細部が変わる（構図は同じで、細かな質感やノイズの出方が変わる）。既定を差し替えると、利用者が「昨日と同じ設定なのに絵が違う」という説明のつかない体験をすることになる。②`sdpa` は常に正しい参照実装であり、比較の基準として動かさない価値がある。③切替はUIの1クリックで済み、その選択はブラウザ側に保存されるため、速度を取りたい利用者が払うコストが小さい。
2. **切替の単位はジョブ**。サーバーの再起動もパイプラインの再読み込みも要らない。リクエスト（`GenerateRequest` / `GenerateChainRequest`）のフィールド `attention_backend` を毎回のジョブが持ち、worker がそのジョブの実行直前に注意機構を差し替える。設定ファイルや起動オプションでの固定は導入していない。
3. **モック2項目は「受理するが効かない」ことを構造で担保する**。`fused_gguf_dequant_gemm`（GGUFの逆量子化と行列積を1つの計算に融合する案）と `vae_mode`（映像を復元するVAEの実装選択。`prune_vaed` は枝刈り版デコーダ）はリクエストとしては受け取るが、**workerへ渡すペイロードにも `GET /status` にも一切載せない**。UI側は常に無効（グレーアウト）表示で、押しても何も起こらない。過去に `fp8_transformer` が「表示はあるが挙動を変えない」状態で長く残った反省から、**実際に使われた方式を後から検証できる仕組み**（下記5）を同時に入れた。`vae_mode` は既存の `vae_tiling`（VRAM節約のためにVAEをタイル分割する設定）とは無関係で、コード上のコメントにもその旨を明記した。
4. **MCPには実装のある1項目だけを公開する**。`submit_generate` / `submit_chain` の引数に `attention_backend` を追加し、モック2件は追加しない（実装のない切替をエージェントに見せても意味がないため。`two_stage_hq` の `pipeline` と同じ考え方）。
5. **実際に使われた方式をメタデータに記録する**。worker の完了イベントに `attention_used` を載せ、`seed_used` とまったく同じ経路で `outputs/{job_id}/metadata.json` のトップレベルへ書き出す。値は `"sdpa"` / `"sage"` / `"sage->sdpa"`（sage を要求したが利用不可で降格した）の3種。**実機ゲートの合否判定はログではなくこの値を根拠にした。**
6. **sage が使えないときはジョブを落とさず降格する**。NAG（§38）や VSF（§41）は「要求したのに条件が揃わなければ 422 で止める」fail-loud の規律だが、Acceleration は**速度の最適化**であって生成結果の意味を変える機能ではないため、規律をあえて変えている。sageattention が導入されていない環境で `sage` を指定しても、警告を1行出して `sdpa` で完走する（この理由はコード上のコメントにも残した）。

### 43.2 実装箇所一覧

- **`engine/transformer/sage_attention_service.py`（新規）**: sage の本体。`SageState`（そのジョブで要求された backend 文字列だけを持つ）、`probe_sage()`（sageattention が実際に import できるかを1回だけ確かめてモジュール変数へキャッシュする。失敗した import を Python 自身はキャッシュしないため必須）、`SageAttentionService.install()`（transformer の各ブロックが持つ `attention_function` を sage 版へ差し替える）。**差し替え対象は48ブロック×6種（`attn1`／`attn2`／`audio_attn1`／`audio_attn2`／`audio_to_video_attn`／`video_to_audio_attn`）＝288モジュール**（NAG が96なのは cross-attention だけを対象にするためで、数が違うのは正しい）。head_dim が sage の対応外であるといった**静的に判定できる条件はインストール時に判定し、対象外のモジュールにはラップ自体を張らない**。実行時に見るのは「マスク付きの呼び出しかどうか」と dtype/device だけにした。
- **`engine/transformer/sage_selfcheck.py`（新規）**: エンジン用仮想環境（`.venv-engine`）のpythonで直接実行する自己検証（pytestからは収集されない）。3項目＝①`sdpa` と `sage` の出力が数値的に一致すること（コサイン類似度 ≥0.999）②フォールバック行列（マスク付き・非対応dtype等でSDPAへ戻ること）③install件数が288であること。NAG/VSF の selfcheck のような大型のものにはせず、G0の単一成果物として必要な3点に絞った。
- **`engine/pipeline/fast_video_pipeline.py`**: `_install_nag()` の直後に `_install_sage()` を追加（毎回のビルドで `ledger.transformer` をラップする）。`_set_sage_job()` は `_set_nag_job` と同じ位置に置くため**例外を投げない実装**にし、`finally` でリセットする。**インストールの順序は結果に影響しない**——NAG は `attn.forward` を、sage は `attn.attention_function` を差し替えるので、触る属性が独立しているため。
- **`engine/pipeline/chain_pipeline.py` / `engine/worker.py`**: chain 側は `generate_chain()` で設定する（NAG が `run_chain` の中で設定しているのは負プロンプトのエンコード順序の制約によるもので、この非対称は相互参照コメントで固定した）。worker には `_resolve_attention()`（未知の値は fail-loud、sage が使えなければ降格）、ジョブ開始ログへの `attn=` 表示、`ready` イベントへの `sage_available` 付与、`done` イベントへの `attention_used` 付与を入れた。**利用可否のプローブはパイプライン構築より前に `try/except BaseException` で完全に囲んで実行し、結果をキャッシュする**（DLLの読み込み失敗やABI不一致が起きてもworkerの起動そのものは絶対に落とさないため）。
- **`api/models.py`**: `GenerateRequest` / `GenerateChainRequest` の両方に `attention_backend`（`"sdpa"` / `"sage"`、既定 `"sdpa"`）・`fused_gguf_dequant_gemm`（bool、既定 `false`）・`vae_mode`（`"default"` / `"prune_vaed"`、既定 `"default"`）の3フィールドを追加し、`to_clip_request()` にも3つとも転記した（転記漏れの実害はジョブ記録の表示欠落だが、再現性のためのメタデータが正しくなくなるので必須）。
- **`services/ltx_runner.py`**: `attention_backend` が `"sdpa"` でないときだけ worker ペイロードへ加算する（既定のジョブはペイロードのキーが1つも増えない＝従来とバイト同一）。加えて、engine用仮想環境の site-packages に `sageattention/` と `triton/` が両方あるかを見るファイル存在チェック（`sage_available`、1回だけ評価してキャッシュ）、worker の `ready` からの受領、`done` の `attention_used` の中継を実装した。
- **`services/pipeline_manager.py`**: `acceleration_status_block()` を新設し、利用可否の真理値表（mock時は `false`／未ロード時はファイル存在チェック／ロード成功時はworker自身の import 判定／ロード失敗・解放後はファイル存在チェック）をこの1箇所に集約した。凍結済みの `vram_optimization` には一切触れていない。metadata.json への `attention_used` の書き出しもここ。
- **`api/status.py`**: `GET /status` のトップレベルに `"acceleration": {"attention_backends": ["sdpa","sage"], "sage_available": bool}` を追加した。判定経路を示す `sage_source` のような追加フィールドは設けていない（同じ応答の `pipeline_loaded` を見ればどちらの経路の値かが分かるため）。
- **`mcp_server/tools/generate.py`**: `submit_generate` / `submit_chain` の引数末尾に `attention_backend` を追加。日本語のdocstring（MCPのツール定義の生成元）に「同一シードでも生成結果の細部が変わる」旨と、`backend_status` の `acceleration.sage_available` で利用可否を確認できることを明記した。
- **`gradio_ui/i18n.py` / `ui.py` / `handlers.py` / `batch.py`**: Settings タブの Behavior と Server config viewer のあいだに Acceleration 区画を新設（①モックのチェックボックス〔無効〕②attention のラジオ③モックのラジオ〔無効〕）。ペイロードへの加算は単発・chain・`build_a2v_chain_payload` の3経路に入れた。**Gradio のバッチはハンドラの引数ではなくスナップショット（`BatchSnapshot`）経由で設定を受け取る**ため、そこへの配線も忘れずに行った（ここが漏れるとバッチだけ永久に `sdpa` のまま、という「表示だけ」の罠の再演になる）。
- **インストーラ関連（`scripts/install_ltx.ps1` / `engine/venv-engine.freeze.txt` / `engine/engine-venv-pyproject.toml`）**: 43.3の「依存の再導入」を参照。

### 43.3 依存の標準同梱（2026-07-28に削除したものの再導入）

sage を使うには外部パッケージが要るため、**`sageattention` 2.2.0 と `triton-windows` 3.5.1.post24 をエンジン用仮想環境の標準同梱に戻した**。

- `sageattention` は [woct0rdho 版の Windows 用ビルド済み wheel](https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/) を**直リンクで固定**して導入する（`sageattention-2.2.0+cu128torch2.9.1.post6-cp310-abi3-win_amd64.whl`）。torch 2.9.1+cu128 に合わせてビルドされた ABI 固定の wheel であるため、`engine/engine-venv-pyproject.toml` の `dependencies` にも `[tool.uv.sources]` にも**載せていない**——`-ResolveLatest`（依存を最新へ解決し直すオプション）は未検証の新しい torch を引く経路であり、この wheel とは互換にならないため。**`-ResolveLatest` を使った環境では sage の動作は保証外**である旨をコメントに明記した。`triton-windows` のみ `dependencies` に載せている（バージョンの固定は freeze 側が持つ）。
- `triton-windows` は TinyCC と ptxas を同梱しており、**エンドユーザーに Visual Studio の導入を要求しない**ことを確認済み。
- **§40.1 で削除した19エントリのうち2つを戻したことになるが、当時の判断と矛盾しない。** §40.1 が削除したのは旧世代の `sageattention` **1.0.6**（フォーク由来で、コードから一度も import されていない死重依存）であり、今回入れるのは現行世代の 2.2.0 で、`engine/transformer/sage_attention_service.py` という**実際の消費者がある**。「実消費者のない依存は置かない」という当時の基準はそのまま守られている。同旨の追記を §40.1 にも入れた。
- **ハッシュが変わるため、導入済みの環境では次回の `setup.bat` 実行時に freeze の再適用が1回走る**（`.venv-engine/.nz-engine-state` との突き合わせによる冪等ガード）。パッケージの差分自体はほぼ無いため、実測では監査（audit）で止まる短時間の処理になる。「即座に終了」ではない点に注意。

### 43.4 機械検証の結果

- **エンジン用仮想環境の selfcheck**: `sage_selfcheck` **3/3 PASS**（数値パリティ cos≥0.999／フォールバック行列／install件数288）。
- **バックエンドの pytest**（アプリ用仮想環境）: **817 passed / 6 skipped**。§42.2 のベースライン776件に新規41本を加算したもので、既存テストの削除はゼロ。skip の6件は従来どおり torch 未導入によるエンジン系テストの収集スキップで、本件とは無関係。
- **フロントエンドの型検査**: `npm run typecheck`（`tsc -b`）**0エラー**。
- **フロントエンドの vitest**: **1585 passed / 10 skipped**（直前の1545件から+40本。前回値の出典はフロントエンド [`DEVLOG.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/DEVLOG.md) §54.9）。
- **ネイティブ（`.aux2`）の doctest**: **267ケース全PASS**。

新規テストの重点は「**既定のジョブでは何も増えない**」ことの固定に置いた。worker ペイロードの完全一致断言（`build_a2v_chain_payload` の完全dict一致1本＋キー順序断言5本）と、MCP側の `set(body.keys()) ==` の完全一致断言が既存のトリップワイヤとして張られており、既定時にキーが1つも増えないことをこれらが無改修のまま通ることで担保している。加えて `to_clip_request()` の3フィールド転記を直接検査するテストと、モック2件が worker ペイロードにも `/status` にも現れないことを証明するテストを置いた。

### 43.5 実機ゲート表（2026-07-31〜2026-08-01・全項目合格）

生成テストはMCP経由でエージェントが実施し、目視はオーナーが実施した（既存の運用どおり）。

| ゲート | 内容 | 合格条件 | 状態 |
|---|---|---|---|
| G0 | selfcheck | エンジン用仮想環境で `sage_selfcheck` が全PASS（数値パリティ・フォールバック行列・install件数288） | ✅ 合格（3/3） |
| G1 | pytest 回帰 | ベースライン776 passed / 6 skipped を下回らず、新規テストも全緑 | ✅ 合格（**817 passed / 6 skipped**） |
| G2 | 能力公開（ロード前） | `GET /status` に `acceleration` があり `sage_available: true`、凍結済みの `vram_optimization` のキー集合が不変 | ✅ 合格 |
| G3 | 能力公開（ロード後） | パイプライン読み込み後は worker 自身の判定値へ切り替わり、workerログにも記録される | ✅ 合格 |
| G4 | 単発 sage（Gradio） | 720p固定シードで成功し、`metadata.json` の `attention_used` が `"sage"`、`sdpa` 比1.15倍以上、VRAM同等、差は細部のみ | ✅ 合格（下記の実測。**平均1.167倍**・peak VRAM差0.08%以内・同一シードでPSNR約27〜28dB） |
| G5 | IC-LoRA のフォールバック | control系IC-LoRA＋`conditioning_attention_strength=0.6`＋sage で成功し、マスク経路のフォールバックがログに残り、品質は同等 | ✅ 合格（`attention_used="sage"`、マスクフォールバックのINFOログがジョブ内でちょうど1回） |
| G5.5 | sage × NAG | NAG有効＋sage で成功・品質同等 | ✅ 合格 |
| G5.6 | sage × VSF | `neg_method="vsf"`＋sage で成功・品質同等 | ✅ 合格 |
| G6 | chain ＋ sage | 2クリップ以上の720pが成功し、継ぎ目に破綻がない | ✅ 合格（2クリップ×121フレーム・720p） |
| G7 | 未導入時の縮退 | sageattention を一時退避 → `sage_available: false`、API直叩きで `sage` を指定しても `sdpa` へ降格して完走し、メタデータが `"sage->sdpa"` になる（検証後に復元） | ✅ 合格 |
| G8 | フロント目視（AviUtl2実機） | 区画の表示・日英の文言・モック2件のグレーアウト・sage での生成・選択の保持 | ✅ 合格（2026-08-01 オーナー実機確認。下記の実測を含む） |
| G9 | インストーラ | `.nz-engine-state` を削除して再実行し、wheel の直リンク取得と freeze の再適用が1回走って完走する | ✅ 合格（監査止まりで完走・wheel直リンクは HTTP 200 応答） |

**スパイク（2026-07-31・実装着手前の可否判定。RTX 4070 Ti SUPER）**: 720p（1280×768・257フレーム）の生成が end-to-end で **261.4秒 → 224.3秒（1.17倍）**、2段目（stage2）は **32.76 → 20.95 秒/ステップ（1.56倍）**。VRAMのピークは同一。

**G4の本計測**: `sdpa` と `sage` を交互に流した対比較3組で **1.143 / 1.171 / 1.188（平均1.167倍）**。peak VRAM の差は0.08%以内。同一シードでの `sdpa` 版と `sage` 版のPSNRは約27〜28dBで、構図は同一・細部のみが異なる。

**オーナー実機（2026-08-01）**: i2v（画像からの動画生成）＋NAG、1344×1728・153フレームで **460.63秒 → 366.85秒（1.26倍）**。Settings の3項目の見た目・文言も合格。

### 43.6 計測手順の注意（**これを外すと偽のFAILが出る**）

速度の比較をやり直すときは、次の2点を必ず守ること。**単純に「1本目に `sdpa`、2本目に `sage`」を流して比べると 1.09倍程度にしか見えず、合格基準（1.15倍以上）を割って偽のFAILになる。**

1. **worker プロセスの初回ジョブだけが約8%速い。** 原因は特定していないが再現性のある挙動で、比較の1本目に有利な下駄を履かせてしまう。したがって **`sdpa` と `sage` を交互に流す対比較**（sdpa→sage→sdpa→sage…）を行い、隣り合う組どうしで比べること。
2. **`sage` の計測は2ジョブ目以降で行う。** sage は内部で triton を使い、その初回だけ JIT コンパイル（実行時のカーネル生成）が走るため、1本目には無関係な時間が乗る。

### 43.7 §40（依存整理バッチ）との関係

§40.6 の実機SHA回帰は「依存構成を変えたときに出力がバイト単位で変わらないことを確かめる」ためのチェックリストだったが、**今回の依存の再導入（43.3）については、本節の G9（インストーラの再適用が完走し、`sage_available: true` になること）で吸収・代替する**。理由は、今回追加した2パッケージが「既定のジョブでは import すらされない」位置にあり（`attention_backend != "sdpa"` のときだけ触る）、既定経路のバイト不変性は §43.4 の pytest 側のペイロード完全一致テスト群が構造的に固定しているため、実GPUでのSHA再取得は同じ事実を高いコストで二重に確かめるだけになるからである。§40 側にも同旨の追記を入れた。

### 43.8 既知の無関係な事象（記録のみ）

`GET /status` の `gpu` ブロックが常に `available: false` を返す。これはアプリ用仮想環境（`./.venv`）にCUDA版のtorchを入れない**2プロセス／2仮想環境という構成そのものに由来する既存の挙動**で、今回の改修とは一切関係がない。ただし Acceleration の検証中に `/status` を何度も読むことになり、「GPUが見えていないから sage も効いていないのでは」と紛らわしいため、無関係であることをここに明記しておく（実際のGPU情報はエンジン側のworkerプロセスが持っている）。

### 43.9 残タスク

1. コミットはオーナー判断（本節作成時点で未コミット）。
2. モック2項目（fused GGUF dequant + GEMM／PruneVAED）の実装は将来課題として起票済み（フロントエンド側台帳 [`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-49・§3-50）。
3. `sage` を既定にするかどうかの再検討は、フィールドでの安定実績が溜まってからの判断事項として起票済み（同 §4-22。現状は再現性を優先して `sdpa` 既定）。

## 44. ★Acceleration第2弾（先読みblock swap＝`block_swap_prefetch`）＝CPU⇔GPUブロックスワップの転送を計算の裏へ隠す先読み機構＝実装完了・機械検証（selfcheck・pytest・型検査）全PASS・**実機ゲート全項目合格・オーナー目視ゲート合格・オーナー実機検証完了（2026-08-02）**（2026-08-01〜02実装／実機ゲート・オーナー目視ゲート・オーナー実機検証すべて完了。**テーマ完結**）

> **正本＝本節。** block swap（既存の低VRAM機構。48ブロック中8個常駐のスライディングウィンドウで transformer の重みを CPU⇔GPU 間に出し入れする）の毎 forward パスの転送を、計算とは別の CUDA stream で先回りさせて隠す「先読み」を追加した。あわせて GPU→CPU の退避コピーを廃止する（重みは推論中に一切変化しないため、CPU 側の正本を保持して GPU 側は捨てるだけでよい）。**攻撃対象は転送方式のみで生成アルゴリズムは一切変えない**ため、§43 の `sage`（数値精度が変わり同一シードでも絵が変わる）とは違い、**同一シードなら off/on でビット単位一致が期待値**になる。承認済みプラン・詳細設計・敵対的レビュー（`prefetch_plan_review_findings.md`、CRITICAL 3件・MAJOR 7件・MINOR 5件）を統合した実装用正本 [`BLOCKSWAP_PREFETCH_WORKORDER.md`](BLOCKSWAP_PREFETCH_WORKORDER.md) に沿って、S0（`inference_mode`×`_make_subclass`可否スパイク）→S1（バックエンド核心）→S2（API配線）→S3（実機ゲート G1〜G7）→S4（既定 on 反転＋G8）→S5（Gradio UI）→S6（フロントエンド）→S7（本節・ドキュメント）の順で実施した。

### 44.1 決定事項

1. **転送の高速化は2本立て**。①**GPU→CPU 退避コピーの廃止**（重みは推論中不変とテスト実証済み。CPU 側の正本を保持し、ウィンドウから外れたブロックは GPU 側を捨てるだけで復元は正本への付け替えのみ、D2H はゼロになる）。②**pinned メモリのステージング（2枠）＋専用転送 stream＋CUDA event による先読み**（H2D 転送を計算カーネルの裏に隠す）。既存の同期スワップ本体 `_patch_block`（`block_swap_service.py:153-218`）は**1文字も変えず**、prefetch 要求時だけ新設の `_patch_block_prefetch` を張る二択構成にした。
2. **arena（先読みしたブロックの GPU 側連続領域）は計算 stream 上で確保し、転送 stream 上では確保しない**。原案（転送 stream 上で確保＋`record_stream`）は敵対的レビューで CRITICAL 判定を受けた——PyTorch 2.9 の CUDACachingAllocator は stream を第1キーにしており、転送 stream 所有の解放済みブロックは計算 stream の割り当てに再利用されない。stage1 最終パス終了時に約3.0〜3.3GB が転送 stream 側プールに滞留したまま spatial upsampler（パイプライン最大の VRAM 山）を迎え、16GB 機で共有メモリスピルを誘発する。修正版は arena を計算 stream 上で確保し、確保完了イベントを転送 stream が `wait_event` してから H2D を発行する（WAR ハザードもこれで解消し `record_stream` は不要になる）。同期点はこれを含めて**4点のみ**（S1: 枠再利用前の host wait／S1b: 転送 stream の alloc 完了待ち／S2: 計算 stream の転送完了待ち／S4: install/teardown 時の `xfer.synchronize()`）。
3. **常駐ウィンドウは非循環のまま**（現行と同一の `W(idx) = {j | idx <= j < min(idx+bs, total)}`）。循環させると stage1→stage2 遷移時（spatial upsampler 直前）に常時8ブロック（約2.7GB）が残留し OOM を誘発しうるため。非循環を維持する代償は「パス先頭ブロック0の転送待ち約16ms/pass」のみ（11パスで約0.18秒＝全体の0.07%）で、払う価値がないと判断した。
4. **off 時は現行挙動を完全温存する**（オーナー確定事項）。退避コピーの廃止も on 時のみ適用し、off のジョブで発生する追加処理は `prefetch_requested` の bool 参照1回のみ。A/B 比較のベースラインを保護し、`sage`/NAG/VSF が守ってきた「OFF はバイト同一」の規律を踏襲した。
5. **`/status` の利用可否判定は実ゲートと完全同一の式にする**（`not runner.is_mock and int(low_vram.block_swap_blocks_on_gpu or 8) > 0`）。原案は表示専用の `low_vram.block_swap`（bool）を見る設計だったが、real 経路のどこからも読まれておらず既定構成では `false` になる不整合が敵対的レビューで確定したため修正した（§44.8 にこの `or 8` の既存挙動そのものについても記録する）。
6. **fused GGUF dequant+GEMM は本テーマの実測マイクロベンチにより no-go でクローズ**（§44.3）。フロントエンド側台帳 [`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-49 が正本（本節は根拠データの置き場）。UI の disabled トグルは撤去せず現状維持というオーナー裁定は §43 から不変。
7. **既定値は実機ゲート合格（ビット一致＋VRAM）を条件に on へ反転する**（S4）。開発中は off で実装し、ゲート全PASS を確認してから反転した。詳細な経緯は §44.7。
8. **VRAM 追加は先読み分+1ブロック（≈370MB）まで許容する**（快適フレーム上限より高速化を優先するというオーナー確定事項）。判定は `peak_vram_reserved_mb`（`torch.cuda.max_memory_reserved`）を主指標にする——既存の `peak_vram_mb`（`max_memory_allocated`）は本改修のリスク（stream別プール分断・reserved 増）を原理的に検知できないため、加算的に追加した。

### 44.2 実装箇所一覧

- **`engine/transformer/block_swap_prefetch.py`（新規）**: `PrefetchEngine` 本体。CPU 正本のスナップショット・レイアウト計算（512B アライン）・pinned ステージング・専用転送 stream・event 管理・arena 確保/解放を持つ。`GGMLQuantizedTensor`（GGUF 圧縮重み）はメタ（`_ggml_type`/`_float_shape`）を保持したまま生バイトとして転送し、GPU 側で `_make_subclass` により再構成する（S0 スパイクで `inference_mode()` 下での可否を実機検証済み・成功）。
- **`engine/transformer/block_swap_prefetch_selfcheck.py`（新規）**: `.venv-engine` で直接実行する自己検証（pytest では収集しない。engine 用仮想環境に pytest が無いため）。14項目（§44.4）。
- **`engine/transformer/block_swap_service.py`**: 既存の `_patch_block`（同期スワップ本体）は無改変。`prefetch_requested`/`last_prefetch_used`/`_patch_block_prefetch`/`teardown_prefetch()` を追加し、prefetch 要求時だけ `PrefetchEngine` を張る二択分岐にした。pinned プールと転送 stream はサービス常駐インスタンスがジョブ跨ぎで保持し再利用する（grow-only）。
- **`engine/pipeline/fast_video_pipeline.py`**: `_set_block_swap_prefetch_job()`/`_reset_block_swap_prefetch_job()`（`_set_sage_job` と同じ per-job set/finally-reset 規律）。ジョブ終了の `finally` で `teardown_prefetch()`（in-flight 転送の完走待ち＋前ジョブの CPU 正本・arena 参照の解放）を呼び、次ジョブの `install()` 冒頭にも冪等な安全網として同じ呼び出しを置いた（`VERIFICATION_LOG.md` §9.6 の旧リークと同形の再発を防ぐため）。
- **`api/models.py`**: `/generate`・`/generate/chain` に `block_swap_prefetch: bool`（S4で既定 `True` へ反転）を追加、`to_clip_request()` へ転記。
- **`services/ltx_runner.py`**: worker ペイロードへ既定と異なるときだけ加算（キー順は `attention_backend` の後）。`GenerationOutcome` に `block_swap_prefetch_used`（`"off"`/`"on"`/`"on->off"`）と `peak_vram_reserved_mb` を加算。
- **`engine/worker.py`**: `_resolve_block_swap_prefetch`（キー欠落 → `False`）、`_block_swap_prefetch_used`／`_peak_vram_reserved_mb`（`torch.cuda.max_memory_reserved` 換算）を追加し `done` イベントへ加算。
- **`services/pipeline_manager.py`**: `/status` の `acceleration` ブロックへ `block_swap_prefetch_available` を追加（§44.1-5 の式）。metadata.json（`_write_metadata`／`_write_chain_metadata`）へ `block_swap_prefetch_used`・`peak_vram_reserved_mb` を追加。
- **`mcp_server/tools/generate.py`**: `submit_generate`／`submit_chain` の引数末尾へ素通し追加。
- **`gradio_ui/ui.py`／`i18n.py`／`handlers.py`／`batch.py`**: Acceleration 区画へチェックボックスを追加（`attention_backend` と同じ reg/配線/i18n パターン）。**fused GGUF の disabled チェックボックスは変更しない**（オーナー確定事項）。
- **フロントエンド**: `webui/src/shell/accelerationSettings.ts`（`blockSwapPrefetch: boolean` 追加、localStorage を JSON 形式へ移行）・`useAccelerationSettings.ts`（localStorage 書き出しは `useEffect` 側。render 中の副作用にしない）・`SettingsPanel.tsx`（トグル追加。disabled は利用不可側〔On〕のボタンのみ）・`i18n/strings.ts`・`api/types.ts`。

### 44.3 設計判断の根拠（マイクロベンチ、2026-08-01実測・RTX 4070 Ti SUPER）

- **転送帯域実測**: pageable H2D 14.0GB/s／pageable D2H 9.1GB/s／pinned H2D 23.4GB/s／pinned D2H 25.4GB/s／CPU内 pageable→pinned 27.1GB/s。別 stream の pinned H2D と GEMM のオーバーラップはほぼ完全（同時実行≒max）——これが「先読みで隠せる」ことの実測根拠。
- **1パス（48ブロック）あたりの転送（改修前）**: H2D 47×340MB÷14.0GB/s ≈ 1.14s ＋ D2H 47×340MB÷9.1GB/s ≈ 1.76s ＝**約2.9s**。768p/257f は全体270s・11パスで転送**約32s**（11〜13%）。
- **改修後の見積り**: D2H は構造的にゼロ（退避コピー廃止）。H2D は pinned 23.4GB/s で 0.68s/pass に短縮した上でほぼ完全に計算の裏へ隠れる。露出するのは cold start とパス先頭の約46ms/pass のみ（11パスで約0.5秒＝0.19%）。期待値: **270s → 約236〜240s（1.13〜1.15倍）**。ゲート合格基準は保守的に**1.08倍以上**とした。
- **fused GGUF dequant+GEMM 不採用の実測根拠**（§3-49 no-go クローズの根拠データ。生データは `scratchpad/prefetch_gate_results.md` 末尾「マイクロベンチ」節）: 逆量子化（dequant）の1パス合計は**1.63秒**＝stage2実測1ステップ32.76秒の**4.97%**（Q6_K 層混在を補正しても5〜7%）。dequant+linear を1カーネルへ融合した場合と、事前 dequant 済み linear との差は 4096×4096 層の実測で**1.954ms（13.2%）のみ**——支配項は融合できる dequant コストではなく、block swap の CPU⇔GPU 転送（1パス約2.9秒）側にあることが判明し、こちらは本テーマで解消したため、融合カーネルへの投資対効果が立たないと判断した。

### 44.4 機械検証の結果

- **エンジン用仮想環境の selfcheck**（`block_swap_prefetch_selfcheck.py`、新規）: **14項目（C1〜C14）全PASS**——出力ビット一致（off/on）、CPU 正本の不変性、常駐数上限 `<= blocks_on_gpu+2`、発行スケジュール（sync miss はパス先頭のみ）、GGML メタ保持、pinned 確保失敗時のフォールバック、`blocks_on_gpu>=total` の早期return、ジョブ跨ぎのリーク無し、stream 安全性の負荷テスト（S1b を意図的にスキップするネガティブケースで破綻を確認＝S1b の必要性を実証）、例外後の再 install、レイアウト計算、スロット列挙、発行スケジュールの純関数境界、共有テンソル検出の14点。
- **バックエンドの pytest**（アプリ用仮想環境）: 全緑（§43.4 のベースライン 817 passed / 6 skipped を下回らず、既定リクエストでは worker ペイロードのキーが1つも増えないことをペイロード完全一致テスト群が固定した状態のまま新規分もすべて通過）。
- **フロントエンドの型検査**: `npm run typecheck`（`tsc -b`）**0エラー**。
- **フロントエンドの vitest／ネイティブ doctest**: 全緑・不変（本件によるケース数の増減は次回のフロントエンド側記録〔`DEVLOG.md`〕を正本とする）。

### 44.5 実機ゲート表（2026-08-01〜02・全項目合格）

環境: RTX 4070 Ti SUPER 16GB、real backend、`attention_backend="sdpa"` 固定（sage との要因混在を避ける）、シード 424242 固定、同一 i2v キーフレーム。§43.6 の交互対比較プロトコルを遵守（worker 初回ジョブの偏りは G1 で吸収済み）。

| # | 内容 | 合格条件 | 実測 | 判定 |
|---|---|---|---|---|
| G1 | 768p/121f ビット一致 | 全フレーム一致 | mp4 の SHA256 完全一致 | ✅ PASS |
| G2 | 768p/257f 交互対比較3組 | 平均8%以上短縮 | 平均**14.71%**短縮（15.94% / 17.03% / 11.15%） | ✅ PASS |
| G3 | 1088p/153f 1組 | 短縮確認 | **13.39%**短縮 | ✅ PASS |
| G4 | Style LoRA付きビット一致 | 出力ビット一致 | SHA256 完全一致 | ✅ PASS |
| G5 | G2のVRAM | `peak_vram_reserved_mb` 差+400MB以内 | 最大+0MB（2組はマイナス） | ✅ PASS |
| G6 | chain 2クリップ | 完走＋一致＋短縮傾向 | 完走・SHA一致・**23.15%**短縮 | ✅ PASS |
| G7 | バッチ経路スモーク | metadata反映 | MCP `submit_generate` 経由で `used=on` | ✅ PASS |
| G8 | 既定on反転後スモーク | 明示指定なしで on 動作 | 明示なしで `used=on`／明示 `false` で `used=off`／両者 SHA 一致 | ✅ PASS |

**全ジョブ生数値**（`gen秒`は生成所要時間、`peak_vram_mb`は `max_memory_allocated`、`peak_vram_reserved_mb`は `max_memory_reserved`）:

| label | gen秒 | prefetch_used | peak_vram_mb | peak_vram_reserved_mb | SHA256[:16] |
|---|---|---|---|---|---|
| G1_off | 180.03 | off | 8535 | 13808 | 6c3c9be7eaca42d9 |
| G1_on | 179.02 | on | 9524 | 13796 | 6c3c9be7eaca42d9 |
| G2_off_1 | 333.02 | off | 10636 | 13926 | 8dc87eef9c60c9b5 |
| G2_on_1 | 279.92 | on | 10614 | 13918 | 8dc87eef9c60c9b5 |
| G2_off_2 | 321.89 | off | 10638 | 13920 | 8dc87eef9c60c9b5 |
| G2_on_2 | 267.08 | on | 10614 | 13920 | 8dc87eef9c60c9b5 |
| G2_off_3 | 307.65 | off | 10640 | 13924 | 8dc87eef9c60c9b5 |
| G2_on_3 | 273.34 | on | 10617 | 13918 | 8dc87eef9c60c9b5 |
| G3_off | 378.87 | off | 12242 | 13664 | 37739d5e637be100 |
| G3_on | 328.13 | on | 12221 | 14088 | 37739d5e637be100 |
| G4_off | 228.42 | off | 9263 | 13816 | 6927e8947bcf6fff |
| G4_on | 173.70 | on | 9531 | 13812 | 6927e8947bcf6fff |
| G6_off | 206.09 | off | 9296 | 10848 | 159798ebdd635cf8 |
| G6_on | 158.37 | on | 9561 | 11078 | 159798ebdd635cf8 |
| G7(MCP) | 101.91 | on | — | 10020 | — |
| G8_default(49f) | 162.86 | on | — | — | 15eef6b9…aa13a |
| G8_explicit_off | 217.38 | off | — | — | 15eef6b9…aa13a（一致） |

失敗・破損・OOM・traceback は全88パス中0件。先読み統計は全88パスで sync miss がパス先頭ブロック0の1回のみ（設計どおり）、pinned 枠待ちは2〜6ms/パス（768p）。pinned プールは2×253.8MB、CPU 正本は11440MB（48ブロック、最大ブロック253.8MB）。LoRA 適用時は11776MB/260.8MB（LoRA A/B バッファが arena へ自動的に取り込まれることを実測確認）。nvidia-smi 10秒間隔83点の実測ピークは14896MiB/16376MiB（91%）で物理 VRAM 内、共有 GPU メモリ溢れなし。

**G6_off 実行中に発生したサーバー停止1回**は、検証エージェントが起動したバックグラウンドタスクの寿命切れ（親シェル終了）が原因と切り分け済みで、製品不具合ではない（workerログに traceback／CUDAエラー／OOM なし、走っていたのは prefetch=off の無変更経路、デタッチ起動で立て直し完走）。

### 44.6 計測手順の注意（§43.6を踏襲・追加事項あり）

速度の比較は §43.6 と同じ2点（worker 初回ジョブだけ約8%速い＝必ず交互対比較で1本目は捨てる／`attention_backend="sdpa"` 固定で sage と要因を混ぜない）を守った上で実施した。本節で追加する注意は次の1点。

- **ビット一致（SHA256）が off/on 比較の判定基準としてそのまま使える**。§43 の `sage` は数値精度が変わるため PSNR（約27〜28dB）で「構図は同じ・細部は違う」ことを確認する方式だったが、本改修は**転送方式しか変えていない**ため、同一シードの off と on は理論上まったく同じ計算を行う。したがって「ビット単位で完全一致するか」がそのまま合否判定になり、実際に G1・G4・G6・G8 の全ゲートで SHA256 完全一致を確認した。**不一致が出た場合は速度低下より深刻な stream 同期漏れの疑いとして即 FAIL・原因究明**とする規律（設計時点からの方針どおり）。

### 44.7 既定 on 反転の経緯とペイロード方式の修正

- **既定値の反転はオーナー確定事項どおり「実機ゲート合格を条件」に行った**（S4）。§44.5 の G1〜G7 が全PASS したことを受けて `api/models.py` の `/generate`・`/generate/chain` を `block_swap_prefetch: bool = True` へ反転し、`gradio_ui`（`BLOCK_SWAP_PREFETCH_DEFAULT=True`）・フロントエンド（`BLOCK_SWAP_PREFETCH_SERVER_DEFAULT=true`）も揃えた上で G8（既定 on 反転後スモーク）を実施し、明示指定なしで `used=on`、明示 `false` 指定で `used=off`、両者の出力 SHA が一致することを確認した。
- **既定反転で顕在化した事故経路をS4追補で修正した**: `gradio_ui/handlers.py` の3箇所のペイロード構築は、実装当初「値が `True` のときだけ送信する」規律（既定 `False` の頃はこれで足りていた）のままだった。既定を `True` へ反転すると、この規律のままでは**利用者が明示的に `False`（off）を選んでも、`False` はペイロードへ一切送られなくなり、サーバー既定の `True` が黙って適用されてしまう**——「明示 off が届かない」事故経路になる。S4 でこの3箇所を「**サーバー既定と異なる値のときだけ明示送信する**」規律へ修正した（フロントエンドの `accelerationRequestFields()` は当初からこの規律で実装済みだったため対象外）。同型の不整合は MCP ツール（`submit_generate`／`submit_chain`）にもあり、同時に修正した。`services/ltx_runner.py`（server→worker の中継）は解決済みの値を読む方式のため変更不要だった。

### 44.8 注意記録（既知の癖・見かけ上の差分）

- **G3（1088p/153f）の `peak_vram_reserved_mb` は 13664→14088（+424MB）**で、単体では §44.1-8 の「+1ブロック（≈370MB）まで許容」をわずかに超える。ただし**実害なしと判断**した根拠は、この絶対値（14088MB）が 768p/257f（13918〜13926MB）より**低い**ことである。1088p/153f は活性値ピークが小さいぶん、先読み arena の増分がそのまま `reserved` に顕在化しただけで、VRAM 天井を押し上げてはいない。
- **G1（121f）の `peak_vram_mb`（allocated）+989MB は見かけ上の差**であり、257f（G2/G6 相当）ではむしろ allocated/reserved ともに on が微減する。121f 固有の現象で、フレーム数が少なく活性値ピークが小さいぶん先読み arena の増分が相対的に露出しただけである。
- **`block_swap_blocks_on_gpu=0` は `or 8` により実質8として扱われる**（`services/ltx_runner.py:1086` 由来の既存の式）。これは**本テーマで導入したものではなく既存挙動**であり、`/status` の `block_swap_prefetch_available` の判定式（§44.1-5）もこの既存の式にあえて揃えた——実ゲート（実際に block swap が効くか）と表示が食い違わないようにするためで、`0` を「本当に0」として扱いたい場合の是非そのものは本テーマのスコープ外である。

### 44.9 §43（Acceleration第1弾）との関係

本節は Acceleration 区画の2つ目の実装項目である。§43 の `attention_backend`（sage）は**計算精度を変えて速くする**方式でビット一致を捨てる代わりに1.17〜1.26倍を得たのに対し、本節の `block_swap_prefetch` は**転送方式だけを変えて速くする**方式でビット一致を保ったまま768p/257fで平均1.17倍（14.71%短縮）を得た。両者は独立した切替（`attention_backend`・`block_swap_prefetch`）であり、G4（sdpa固定でのビット一致ゲート）以外では併用の実機ゲートを本テーマ単独では行っていなかったが、2026-08-02のオーナー実機検証（§44.11）で `sage`＋prefetch on の1点計測を実施し、併用時も速度がさらに短縮しVRAMは悪化しないことを確認した。

### 44.10 残タスク（すべて解消・テーマ完結）

1. **オーナー目視ゲート（Gradio UI／フロントエンド Settings > Acceleration の見た目）は解消済み**。2026-08-02、Gradio と AviUtl2 プラグイン両方の Settings > Acceleration に「先読みblock swap」トグルが表示されることをオーナー本人が確認し、合格判定した（§44.11）。
2. **コミットは解消済み**。両リポジトリ（本リポジトリ・フロントエンド `Nz-LTX23-frontend-AviUtl2`）ともオーナーが手動コミット・プッシュ済み（2026-08-02）。
3. **sage との併用時の速度・VRAM 計測（§44.9）は解消済み**。2026-08-02のオーナー実機検証（§44.11）で `sage`＋`block_swap_prefetch` 併用の1点計測を実施し、`sdpa`＋prefetch on 比でさらに14.6%短縮、VRAM は同水準（増加なし）と確認した。G2/G3 相当の交互対比較3組を伴う網羅計測ではないが、併用が問題なく動作し速度・VRAM とも悪化しないことは実機で裏付けられたため、追検証の必要が生じるまではこれで足りると判断する。

**以上により本テーマ（Acceleration第2弾＝先読みblock swap）は完結した。**

### 44.11 オーナー実機検証＋目視ゲート合格（2026-08-02）

**目視ゲート**: Gradio と AviUtl2 プラグインの両方で、Settings > Acceleration 区画に「先読みblock swap」トグルが表示されることをオーナー本人が確認し、合格判定した。両リポジトリ（本リポジトリ・フロントエンド `Nz-LTX23-frontend-AviUtl2`）ともオーナーが手動でコミット・プッシュ済み（2026-08-02）。

**実機検証環境**: AviUtl2実機、1088p（1920×1088）・153フレーム・i2v、同一プロンプト／同一キーフレーム／シード `1373009257` 固定。worker再起動直後に実施した。

| ジョブID | 設定 | 生成時間 | `peak_vram_reserved_mb` |
|---|---|---|---|
| `adcb2a99-9a98-4207-8763-6cec6c827654` | sdpa＋prefetch off | 340.75秒 | 13686 |
| `0102215f-9f99-4ff8-8603-e9aa7a516567` | sdpa＋prefetch on | 330.84秒 | 14114 |
| `2b54dd08-6aa2-4bf3-a76a-bb38f8bda837` | sage＋prefetch on | 282.68秒 | 14102 |

4本目のジョブ（`7a6b1228-…`）は投入直後にキャンセル／失敗した空フォルダであり、分析対象外とした。

**ビット一致**: off/on の出力mp4はSHA256完全一致（`CFEE3AAC…C56E4F`）。§44.6で述べた「転送方式のみの変更でありoff/onは理論上ビット単位で一致する」という設計上の期待値を、エージェントによるAPI/MCP経由の計測（G1・G4・G6・G8）だけでなく、**AviUtl2実運用フロー（シード -1 採番→前ジョブからの引き継ぎ）を通した状態でも**裏付けた。

**初回ジョブ偏りの解釈**: off実測340.75秒はworker再起動後の最初のジョブであり、§43.6に記録した「workerプロセスの初回ジョブだけ約8%速い」偏りの影響を受けている可能性が高いため、この値をそのまま真のoffベースラインとして扱うことはできない。一方、on実測330.84秒はゲートG3（§44.5、同じ1088p/153f）のon実測328.13秒と1%以内で一致しており、on側は初回偏りとは無関係に実力値を正しく再現していると判定できる。そこで真の短縮率は、初回偏りの影響を受けていないG3のoff実測378.87秒を基準に算出すると、`(378.87-330.84)/378.87 ≈ 12.7%`＝**約13%短縮**となる。

**sage併用の積み上げ**: sdpa＋prefetch on（330.84秒）を基準に、sage＋prefetch on（282.68秒）は`(330.84-282.68)/330.84 ≈ 14.6%`のさらなる短縮となった。真のoffベースライン（G3のoff実測378.87秒）と比べると、2機能（`block_swap_prefetch`＋`sage`）を合計で適用した場合の短縮率は`(378.87-282.68)/378.87 ≈ 25.4%`＝**約25%短縮**になる。

**VRAMの再現性**: reserved の増分は off→on で `14114-13686=428MB`。ゲートG3で観測した増分（+424MB、§44.8）とほぼ同値（1%未満の差）で再現しており、実運用フローでも増分の傾向がぶれないことを確認した。sage併用時のreserved（14102MB）もsdpa＋prefetch on（14114MB）とほぼ同水準で、sage自体はVRAM使用量にほとんど影響しない。今回の絶対値（13686〜14114MB）は、共有メモリスピルの兆候なく完走しており、§44.5で安全性を確認済みのG2（768p/257f、約13.9GB）・G3（同じ1088p/153f、13664〜14088MB）の水準と整合する。

**位置づけ**: 本項の実機検証とその直前の目視ゲート合格をもって、§44.10の残タスク3点はすべて解消し、Acceleration第2弾（先読みblock swap）はテーマとして完結した。

## 45. ★Style LoRA音声強度制御（`audio_strength`）＝実装完了・機械検証（pytest・型検査・vitest）全PASS・**実機A/Bゲート全項目合格・オーナー実機確認で完了（2026-08-02）**（2026-08-02実装・実機ゲート完了。**テーマ完結**）

> LTX 2.3でStyle LoRA（画風・キャラクターLoRA。追加学習した差分重みを本体モデルへ足し込む仕組み）適用時に音声が壊れる（雑音・音割れ）というコミュニティ報告への対処として、LoRAごとに音声側の適用強度を映像側と独立制御できる`audio_strength`をバックエンドAPI＋MCP（Model Context Protocol）＋Gradio（検証用UI）＋AviUtl2フロントエンドへ一気通貫で実装した。正本は[`LORA_AUDIO_STRENGTH_WORKORDER.md`](LORA_AUDIO_STRENGTH_WORKORDER.md)（分類ルール・スキップ設計の理由・実装ファイル一覧）、API利用者向けの仕様はフロントエンド[`API_REFERENCE.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/API_REFERENCE.md) §5.3、実装の経緯はフロントエンド[`DEVLOG.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/DEVLOG.md) §57。

### 45.1 実装後の自動ゲート結果（2026-08-02実測）

- **バックエンドpytest**（アプリ用仮想環境`.venv`）: **874 passed / 6 skipped**。skipの6件はいずれも従来どおり`torch`未導入によるエンジン系テストの収集スキップ（`test_ic_lora_engine_conditioning.py`1件・`test_ic_lora_forward.py`1件・`test_chain_lora.py`2件・`test_chain_reference.py`2件の計6件）で、既存テストの削除はゼロ。
- **torch必須テスト**: 上記skip対象のうち、本テーマで新設した`tests/test_ic_lora_forward.py`（分類6ケース・None追従・2タプル受理・`audio_strength=0`でのattachスキップとforward結果のno-LoRAとのビット同一・全ミュート時にWARNが出ないことのcaplog確認、計**13件**）を`.venv-engine`（torch入りのエンジン用仮想環境）で実行し、**13 passed**を確認した。実行は`--noconftest`（アプリ用の`tests/conftest.py`が`fastapi`等のアプリ依存を要求し、エンジン用仮想環境には無いため）を付けた`pytest`で行った。
- **フロントエンド**: `npm run typecheck`（`tsc -b`）**0エラー**。`npm run test`（vitest）**1656 passed / 10 skipped（109ファイル）**（skipの10件は`backend.integration.test.ts`が実バックエンド未起動時に自動スキップする既存分で、本テーマとは無関係）。既存テストの削除はゼロ。

### 45.2 オーナー実機A/Bゲート＝全項目合格（2026-08-02実施）

本節は§45.1の機械検証に続き、オーナー本人がreal環境・実GPUで実施したA/B比較の結果記録。以下は実施前に準備した手順で、実施後に得られた結果を末尾に追記した。

**準備**: `audio_strength`が実際に効くことを確認するには、**音声側の重みキーを持つLoRA**を使うこと。実測で確認できたのは以下3本（網羅ではない）：`DR34ML4Y_LT3X_V3`／`LTX-2.3-Henshin`／`LTX2.3-MysticXXX`。**登録済みIC-LoRA（当時3本・現在5本）と`Pixar_Toon`は音声キーがゼロのためno-op**（`audio_strength`を指定しても何も起きない）で、本ゲートの検証には使えない。

**A/B比較（本機能の目的）**: 音声キーを持つLoRAのいずれか1本を用い、同一シード・同一設定で以下を比較する。

- A: `<lora:名前:0.8>`（従来どおり、音声側も0.8が適用される）
- B: `<lora:名前:0.8:0>`（音声側だけ0にスキップ）

Bで音割れが消えているか確認する。**映像はA/Bでビット一致しないのが正常**である（音声潜在が`audio_to_video_attn`経由で映像側へ還流するため。構図・画風は保たれるが細部は変わりうる）。

**回帰C（後方互換ゲート）**: `audio_strength`を指定しない従来どおりのタグ（例`<lora:名前:0.8>`のみ、または`loras`配列に`audio_strength`キーを含めないリクエスト）の出力が、本改修**前**のビルドとビット同一であることを確認する。

**分類が実際に効いたことの確認**: 生成時のエンジンログに`muted=N linears`（`N>0`。bf16融合経路では`muted=N keys`）が出ることを確認する。**`N=0`の場合は分類が空振りしている（音声側キーを1つも掴めていない）ことを意味するので、その場合は結果を待たずに報告すること。**

- [x] A/B比較: Bで音割れが消える
- [x] 回帰C: オーナー判断で省略（合格扱い。理由は下記結果を参照）
- [x] エンジンログに`muted=N linears`（N>0）が出る（出力先はコンソールでなく`logs/ltx_worker.log`。下記結果を参照）
- **結果・所見（2026-08-02・オーナー実施・主観評価。同一シード・同一プロンプト・同一先頭キーフレームで比較。ベースモデルは`Sulphur2 base`、使用LoRAは`LTX2_3_NSFW_furry_concat_v2`——上記「準備」の3本リストには含まれないが、別途音声側の重みキーを確認済みのアダプター）**:
  - **LoRAなし（コントロール）**: やや音質低め。`Sulphur2 base`はもともと音割れ・雑音が載りやすい傾向がある。
  - **動画1.0・音声1.0**: スタイル・動きは明確に変化する一方、音質はさらに悪化した。
  - **動画1.0・音声0.0**: 雑音・音割れが明らかに減少した。加えて、動きが音声1.0時より向上し、音質もLoRAなしのコントロールより向上するという副次的な発見があった（原因は考察の余地があり、音声側の未学習な差分がクロス注意——`audio_to_video_attn`——経由で映像側の時間整合にも悪影響を与えていた可能性が考えられる）。
  - **回帰確認**: オーナー判断で省略した。改修前実装のほうが正しいと信じる理由がないため、ビット同一の突き合わせまでは行わなかったが、両ビルドともLoRAが期待どおりに効く挙動であることを確認しており、合格扱いとする。
  - **`muted=N linears`ログ**: 想定と異なり、コンソールには出ない設計だった。生成workerサブプロセスのstderrは`logs/ltx_worker.log`へ直接リダイレクトされ（`services/ltx_runner.py:1203`の`stderr=log_fh`）、コンソールに出るのはアプリプロセス側の`loras=`行のみである。実際に`logs/ltx_worker.log`を確認したところ、複数ジョブ分にわたって`IC-LoRA LTX2_3_NSFW_furry_concat_v2.safetensors: 672 Linear(s) attached for forward-time apply (strength=1.000, audio_strength=0.000, muted=672 linears)`が記録されており、分類が音声側Linear 672本を正しく掴んでいたことを確認できた。合格。
  - **判定**: 機能検証合格。テーマの実機ゲート完了。

### 45.3 UIフィードバック対応（2026-08-02再デプロイ）

オーナーの実機フィードバックを受け、LoRAチップの並び順を「名前・🎥動画強度・−＋・ミュートトグル・音声強度常時表示」に改善し、同日中に再デプロイした。**この新チップデザインは同日中にオーナーが実機で目視確認し、承認された（目視合格・2026-08-02）。**

ミュートトグルの挙動: 🔇=音声強度0.0、🔊解除=1.0固定復帰（元の追従状態には戻らない）。中間値の指定はタグの手編集のみ。

---

## 46. ★ジョブ毎の再マテリアライズ削減（フロントエンド`PENDING_TASKS.md` §1-9）の一点計測＝前処理固定費の内訳分解・計測完了（2026-08-02）

> 全ジョブ先頭の前処理固定費（従来「74〜90秒」と呼んでいたもの）の内訳を、コード変更ゼロのログ打刻方式で分解した記録。CPU骨格キャッシュ（`LTX_KEEP_RESIDENT=1`）の設計判断の入力となる基礎データ。

### 46.1 計測方法

- ワーカーのログ`logs/ltx_worker.log`は無バッファ書き出し（`python -u`＋`flush=True`）だがタイムスタンプを持たない。そこで外部のPowerShellプロセスで`Get-Content -Wait`によるtailを行い、各行の到着時刻を打刻した別ファイルを作って、既存のフェーズ境界ログ行の時刻差から分解した。**リポジトリのコードは1行も変更していない**。
- 境界行: T0=`generating ...`（ジョブ開始）／T1=`Gemma GGUF module_ops: patched 336 ...`（Gemma骨格開始）／T2=`Gemma GGUF per-layer quant active: ...`（同終了）／T3=`GGUF quant-load from ...gguf`（DiT骨格開始。ロガー名`engine.gguf.quant_service:`で絞る——Gemma側にも同名行があるため）／T4=`BlockSwap prefetch ready: ...`（DiT骨格終了）。
- 条件: t2v（画像なし）、プロンプト・seed=12345固定、attention=sdpa・先読みblock swap有効（いずれもサーバー既定）。ワーカー起動直後の1本目は既知の初回偏り（§43.6）があるため捨てジョブとして除外。検算は打刻ログと`logs/server.log`（タイムスタンプ付き）の突き合わせで±0.33秒以内、4分解の合計と前処理合計の差は±0.01秒。

### 46.2 結果（2026-08-02実測・5本すべて成功）

| 条件 | job_id | 全体 | 前処理正味(T0→T4) | Gemma骨格 | Gemma forward＋VAEエンコーダ骨格 | DiT骨格 | その他 |
|---|---|---|---|---|---|---|---|
| 捨て 768p/257f | 8dff6369 | 244.2秒 | 55.96秒 | 27.94 | 5.61 | 21.32 | 1.10 |
| ① 1088p/153f | 9ca6731f | 321.8秒 | 63.90秒 | 34.44 | 6.61 | 21.76 | 1.10 |
| ① 1088p/153f | bb06a85e | 324.8秒 | 68.42秒 | 36.06 | 9.11 | 22.16 | 1.10 |
| ② 768p/257f | 16565746 | 272.5秒 | 84.55秒 | 37.57 | 22.15 | 23.72 | 1.10 |
| ② 768p/257f | 0eba42e3 | 262.6秒 | 74.11秒 | 32.14 | 16.62 | 24.25 | 1.10 |

条件別平均（捨てジョブ除く）: ①1088p/153f=前処理66.2秒（Gemma骨格35.3／forward7.9／DiT骨格22.0）、②768p/257f=前処理79.3秒（Gemma骨格34.9／forward19.4／DiT骨格24.0）。

### 46.3 わかったこと

1. **骨格再構築（Gemma骨格＋DiT骨格）は毎ジョブ計49〜61秒で、前処理の72〜88%を占める。** 解像度・フレーム数にほぼ依存しない。CPU骨格キャッシュで消せる見込み量は毎ジョブ約50〜60秒（全体比で1088p約17%・768p約21%）で、§1-9起票時のフェルミ推定（55〜75秒）と整合する。
2. **従来の「74〜90秒」はdenoiseの1ステップ目込みの数字だった。** 境界に使える`BlockSwap prefetch pass 1`行や`server.log`の`stage-1 denoise started`行はいずれもdenoise 1ステップ目の完了時に出るため、従来の計り方には7〜9秒（1ステップ分）が混入していた。前処理の正味は56〜85秒。
3. **Gemma forwardは同一プロンプトなのに5.6〜22.2秒と実行順に沿って増加した**（Gemma骨格にも同傾向）。仮説はDiTのCPU側マスター（11.4GB）の積み上がりによるページキャッシュ追い出しだが、実行順と条件が交絡しており未確定。骨格キャッシュの対象外のため§1-9の設計判断には影響しない。
4. ピークVRAMは①12.2GB／②10.6GB。ワーカーログに異常行なし。

- **正本**: 本テーマの台帳はフロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §1-9（同節に本計測を踏まえた訂正——`services/gemma_sd_cache.py`は削除済みで後継の`StateDictRegistry`＋`keep_resident_weights`配線が本流に実在すること——を2026-08-02に追記済み）。

---

## 47. ★keep_resident=1×TE offload併用時のGemmaデバイス配置バグ修正＝実装完了・機械検証（pytest）全PASS・**スパイク再実験でキャッシュ効果を実証（前処理9〜15秒へ短縮・出力ビット一致）。ただし1088pでdenoise減速の副作用を観測（2026-08-02）**

> §46の一点計測に続く§1-9のデバッグ段階。`LTX_KEEP_RESIDENT=1`（wheel側`StateDictRegistry`によるCPU骨格のジョブ間キャッシュ）を有効にすると、TE offload併用時にGemmaが全CPU実行（text-encode 553秒）→device mismatch例外死するバグを修正し、スパイク再実験でキャッシュの効果と副作用を実測した。プランは敵対的レビュー（BLOCKER 1・MAJOR 4・MINOR 11、うちテスト設計のBLOCKERはtorch 2.9.1実機再現で裏取り）を経て確定したもの。

### 47.1 バグの実体と修正

- **原因**: `engine/gemma/gguf_quant_service.py`のCPUビルド後GPU移動が「テンソルの現在デバイス」を代理指標に『loaderが意図的にCPUへ残したdecoder 48層』を識別していた。registry有効時は全テンソルがCPUビルドされるため代理指標が崩れ、全テンソルが移動をスキップされる（`compute_device=cpu`でGemma全体がCPU実行）。
- **修正**: 代理述語を捨て、DiT側の前例（`engine/transformer/dit_cpu_load_service.py`の「ブロック集合サブツリーを除外したleaf走査」）と同型の**除外集合方式**へ置換。除外集合（decoder 48層）は`GemmaLayerOffloadService`の**同一インスタンス・同一root**から取得し、offload installと食い違えない構造にした。if/else分岐は統一形にし、layer_offload無効時は除外集合が空＝従来のelse枝と同じ終着状態。安全装置は「除外サブツリー外にCPU残留leafがあれば名前を列挙して即死する」事後条件検査1本に集約。`held_embed_cpu`のキャッシュHIT時暗黙契約（初回MISSの副作用が常駐loaderに残る前提）もfail-loudガード化。**変更は`gguf_quant_service.py`1ファイルに集約**（+218/−43行）で、本体は`build_device != ledger_device`の内側＝**keep=0（本番既定）では1命令も実行されない**。
- **キャッシュ非汚染の根拠**: paramは`p.data`の再束縛（`load_state_dict(assign=True)`が新規Parameterで包むためキャッシュ実体に届かない）、bufferはスロット再束縛（`assign=True`でキャッシュ実体そのものが刺さっているため`b.data=`は禁止）。この非対称は実測（§47.3のG9ビット一致）でも裏付けられた。

### 47.2 機械検証（2026-08-02実測）

- 新規`tests/test_gemma_keep_resident_move.py`: **5 passed**（`.venv-engine`・`--noconftest`。CUDA有り環境のため実移動テストも実走行）。選択ロジックを純関数`_leaf_tensors_to_move`に分離し、除外・metaスキップ・直付けparamの選択規則をCPU-onlyで検証する設計（当初案の「metaへ移動して観測」はtorch 2.9.1で`p.data = p.data.to("meta")`がRuntimeErrorになることを実機確認して廃止）。
- mock回帰（`.venv`）: **873 passed / 4 skipped / 1 failed**。失敗1件は`test_backend_status_...`で、**ポート18620でバックエンドが稼働中だと必ず落ちる環境依存テスト**（修正を`git stash`した状態でも同一失敗を確認済み＝本改修と無関係）。

### 47.3 スパイク再実験（2026-08-02・環境変数のみ・計7本）

keep=0対照1本（G9参照兼用）→keep=1で捨て768p+計測4本→復帰確認1本。ゲート結果:

| ゲート | 結果 |
|---|---|
| G1 完走 | ✅ 5/5・crash 0 |
| G2 compute_device=cuda:0 | ✅ 全ジョブ |
| G3 2本目以降quant-load行なし（HIT実証） | ✅ |
| G4 Gemma骨格≤5秒 | ✅ 1.6〜4.6秒（基準32〜37秒） |
| G5 T2→T4 ①≤15/②≤25秒 | ✅ ①6.5〜9.1/②5.6〜6.1秒 |
| G6 前処理正味 ①≤30/②≤40秒 | ✅ ①9.7〜15.3/②9.3秒（基準66.2/79.3秒） |
| G7 コミット | ⚠️ 相対PASS（ジョブ間増分0.04〜0.20GB・単調増加なし）／**絶対FAIL**（ピーク110.1GB＝上限114.69GBの96.0%、keep=0比+20.9GB） |
| G8 ピークVRAM | ✅ 基準と±3MB |
| G9 出力SHA256 | ✅ **5本すべてkeep=0とビット一致**（§46のジョブ出力とも一致） |
| G10 復帰実証 | ✅ 環境変数撤去後、既定経路のログに復帰・新設行0件 |

### 47.4 判明したトレードオフ（本テーマの続行判断に直結）

- **前処理は大勝**: 2本目以降のT0→T4が①66.2→9.7〜15.3秒、②79.3→9.3秒（毎ジョブ約51〜70秒短縮。見込み50〜60秒を上回る）。
- **しかし全体時間は条件依存**: ②768p/257fは267.6→222秒（**−45秒・−17%**）だが、①1088p/153fは323.3→340秒（**+16秒悪化**）。原因はdenoise工程の減速で、①のstage2ステップが43.6→60〜62秒（+37%）。VRAMは±3MBで不変のため、**ホストRAM側のコミット圧（96%）でblock swapのCPUマスター11.4GBがページアウトされている**と推定。キャッシュ常駐約20GB（ワーカーPrivateBytes 41.6→62.9GB）が原因。
- **訂正（2026-08-02・オーナーによる実験解釈の確定）**: 上記の1088p減速の計測は、**オーナーが裏で重量級の並行作業（ブラウザでの配信視聴・AviUtl2での動画編集・別の画像生成アプリの常駐等）を行っていた状態**でのもので、物理メモリの取り合いという機序の推定を含めて条件が汚れていた。正しい読みはむしろ「**その状態ですら768p/257fは全体−45秒（−17%）で速度低下ゼロ**」というポジティブな結果である。よって1088pのdenoise減速は現時点では判断材料にせず、§1-9の製品化（configノブ・Settings UI）へ予定どおり進む。実装後の通常運用（生成中は他の重量級作業を控える）でも減速が頻発する場合の対策は、フロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) **§3-51**（メインメモリ不足時の速度低下を賢く避ける機構）として起票した。
- 既知の残リスク（次段階の必須要件として引き継ぎ）: `gguf_per_layer_quant=False`時のIC-LoRA融合（`engine/gguf/loader_service.py`の`weight.add_()`）はstate dictをin-place変異させるため、keep=1と併用すると融合済み重みが永続キャッシュされる。次段階では排他制御が必須。
- **現状**: 本番既定（keep=0）は無変更・バックエンドは既定状態で稼働中。**追記（2026-08-03）**: 本節の修正コードを含む実装は backend `c67f860`／frontend `fdc4fc8` でコミット済みである（未コミットで残っているのは当日の文書追記のみ）。

## 48. ★keep_resident（モデルCPU骨格のジョブ間キャッシュ）の製品化＝`/generate`・`/generate/chain` の per-job フィールド化＋3段ガード＝実装完了・機械検証全PASS・**実機ゲート R1〜R7・R9（エージェント担当分）全項目合格（2026-08-02〜03）。R8 はオーナー目視の別枠・本命の高速化はオーナー実機確認済み（2026-08-03）・細目のオーナー目視ゲート残**

> **正本＝本節。** §46（前処理固定費の内訳分解）→§47（keep=1×TE offload 併用時の Gemma デバイス配置バグ修正とスパイク再実験）に続く、フロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §1-9 の最終段階＝製品化である。§47 まで環境変数 `LTX_KEEP_RESIDENT` によるワーカー全体の設定として実験していたものを、**`/generate`・`/generate/chain` の `keep_resident: bool`（既定 `false`）という per-job フィールド（ジョブごとのリクエスト項目）**へ作り替え、Gradio・MCP・AviUtl2 フロントエンドの3クライアントすべてに露出した。中身は DiT 約16.5GB＋Gemma 約8〜9GB の**CPU 側の「骨格」**（GGUF から組み上げた state dict とモジュールツリー）をジョブ間で保持して使い回す仕組みで、GPU には何も常駐させないため VRAM プロファイルは不変・生成結果はビット単位で不変である。効くのは毎ジョブ先頭の前処理固定費だけで、**HIT 時の前処理は実測 5.32 秒**（ベースライン 68.6〜75.0 秒）まで落ちる。

### 48.1 決定事項（設計判断）

1. **wheel の `build_model_builders()` は呼ばない。** registry を差し替えるのに一番素直に見えるのはビルダー群を作り直す（`build_model_builders()` を呼ぶ）方法だが、これは `*_builder` をゼロから作り直すため、その後に install 群（component-files 経路・GGUF ローダ・Gemma）が書き込んだ `model_path` / `model_loader` / `model_sd_ops` / `module_ops` が**丸ごと消える**。46GB モノリスが経路に戻り GGUF ローダが外れるという、実行時にはログにすら現れない静かな退行になる。
2. **`_swap_registry(enabled)` が `ledger.registry` と8ビルダーを `dataclasses.replace` で同時に差し替える。** 対象は `transformer_builder` / `vae_decoder_builder` / `vae_encoder_builder` / `audio_encoder_builder` / `audio_decoder_builder` / `vocoder_builder` / `upsampler_builder` / `text_encoder_builder` の8つ（すべて `_builder` 付きの実名。`engine/pipeline/fast_video_pipeline.py` の `_LEDGER_BUILDER_ATTRS` が正）。`replace` は他のフィールドを構造上そのまま引き継ぐので、install 群が書き込んだ内容は保存される。registry を掴んだクロージャは存在せず（block swap・NAG・sage の各ラッパも Gemma サービスのラッパも、ビルドのたびに `ledger` / `ledger.*_builder` を読み直す）、差し替えは**次のビルドから即座に効く**。
3. **8属性は存在必須（assert）。** `getattr(..., None)` で「無ければ黙って飛ばす」書き方は意図的に採らない。属性名の打ち間違いや wheel 側のリネームが起きたとき、その書き方だと「そのサブモデルだけキャッシュ無しで CPU ビルドされる」＝**遅くなり、かつメモリも食い、しかもログに何も出ない**という最悪の壊れ方をするためである。
4. **OFF 時は `clear()` ＋ `gc.collect()` ＋残留ログ。** ここを通らないと約20GB の CPU 骨格が居座り続ける。`clear()` は state dict の参照を落とすだけなので、循環参照（block swap の `swapped_forward` クロージャ等）を確実に回収するために `gc` を1回回す。回収しきれない残留は「次の transformer ビルドまで残りうるもの」としてログに明示する（§48.8）。
5. **arm（有効化／無効化）は `generate()`／`generate_chain()` の最外で行い、引数は `keep_resident: bool | None = None`。** `None` は「触らない＝現在の状態を維持」を意味する。これは §47 までのスパイクスクリプト（create 時に `keep_resident_weights=True` を張って直接 `generate()` を呼ぶ書き方）との互換のためで、既存の呼び出し側が壊れない。
6. **create 時の `keep_resident_weights=True` は廃止し、install 群の後の `_swap_registry(True)` へ一本化した。** create 時点で張ると install 群より前に registry が刺さり、1 の問題と同じ経路をもう一本作ってしまう。実装を1本にするため、ワーカーは常に `keep_resident_weights=False` でパイプラインを作り、以後は per-job の arm だけで制御する。
7. **ジョブ終了時にリセットしない。** これは `_set_nag_job` / `_set_sage_job` / `_set_block_swap_prefetch_job` に対する唯一の非対称である——**キャッシュが次のジョブまで残ることそのものが機能**だからである。解放されるのは、後続のジョブが明示的に `keep_resident=False` を要求したとき（＝リクエストでフィールドを省略した場合がまさにこれ）か、ワーカーが死んだときだけである。
8. **モデル切替時のキャッシュ無効化は、`reload()` のワーカー kill によって構造的に保証される。** `services/pipeline_manager.py` の `reload()` はワーカープロセスごと落とすため、モデルを切り替えたあとに古い重みがこのキャッシュから配られることは原理的に起こりえない。§1-9 が要件に挙げていた「キャッシュの無効化条件」は、新しい仕掛けを足さずにこの既存の構造で満たされている。
9. **arm は決して例外を投げない。** `_set_keep_resident_job` は `generate()` の try/finally の**外側**で走るため、ここで落ちると本来走れるジョブが死ぬ。切替に失敗しても「キャッシュされないだけで生成は走る」ほうが良いので、失敗はログに落として続行し、状態フラグは実際に成功した切替でのみ進める（次のジョブが同じ遷移を再試行する）。
10. **RAM 監視・自動降格は作らない**（オーナー判断・2026-08-02）。「メモリが減ってきたら自動でキャッシュを捨てる」類の機構は、フロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) **§3-51** へ先送りした。本節のスコープは「利用者が明示的に on/off する」までである。

### 48.2 実装箇所一覧

- **`engine/pipeline/fast_video_pipeline.py`**: `_swap_registry(enabled)`（本体）・`_set_keep_resident_job(enabled)`（arm）・`_keep_resident_enabled` / `_keep_resident_registry` の2状態・`_LEDGER_BUILDER_ATTRS`（8属性）。`generate()`／`generate_chain()` に `keep_resident: bool | None = None` を追加。
- **`api/models.py`**: `KEEP_RESIDENT_DEFAULT = False` と、`/generate`・`/generate/chain` の `keep_resident: bool`。`to_clip_request()` へ転記。
- **`services/ltx_runner.py`**: worker ペイロードへ**既定と異なるとき（＝`True` のとき）だけ**加算。`GenerationOutcome` に `keep_resident_used` を追加。**env の `setdefault("LTX_KEEP_RESIDENT", ...)` は削除**（§48.7）。
- **`engine/worker.py`**: `_resolve_keep_resident(msg, bs_prefetch)`（キー欠落 → `False`。3段ガードもここ＝§48.3）と `_keep_resident_used(msg, effective)`（`"off"` / `"on"` / `"on->off"`）。**env 読みは `False` 固定へ**。パイプライン生成は常に `keep_resident_weights=False`。
- **`services/pipeline_manager.py`**: metadata.json（`_write_metadata`／`_write_chain_metadata`）へ `keep_resident_used` を追加。**`GET /status` には載せない**（§48.7 末尾）。
- **`mcp_server/tools/generate.py`**: `submit_generate`／`submit_chain` の引数へ追加。既定と異なるときだけペイロードへ載せる（§44.7 で確立した規律に揃えた）。
- **`gradio_ui/handlers.py`／`batch.py`／`ui.py`／`i18n.py`**: Settings タブの Acceleration 区画へチェックボックスを追加。バッチは実行開始時のスナップショットに含める。
- **フロントエンド**（`Nz-LTX23-frontend-AviUtl2`）: `webui/src/shell/accelerationSettings.ts`（`keepResident: boolean`・`KEEP_RESIDENT_SERVER_DEFAULT=false`・`accelerationRequestFields()` が ON のときだけ `keep_resident` を載せる）・`useAccelerationSettings.ts`（localStorage 永続化）・`SettingsPanel.tsx`（2ボタントグル＋ON のときだけ出るヒント）・`i18n/strings.ts`（en/ja）・`api/types.ts`。詳細はフロントエンド[`DEVLOG.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/DEVLOG.md) §58。

### 48.3 ガード表（`engine/worker.py::_resolve_keep_resident`）

`keep_resident=1` と噛み合わない設定を、ワーカー側で3段に分けて止める。**正しさに関わるものだけを 422 相当のエラーにし、メモリの都合にすぎないものは警告して自動 off にする**という切り分けである。自動 off は metadata.json の `keep_resident_used` に `"on->off"` として残り、ログには必ず理由を明示する。

| # | 条件 | 挙動 | 理由 |
|---|---|---|---|
| **G-A** | `gguf_per_layer_quant=False` × `keep_resident=1` | **`RuntimeError`（ジョブを止める）** | bf16 融合経路の IC-LoRA 適用は state dict を in-place で書き換える（`engine/gguf/loader_service.py` の `weight.add_()`）。永続キャッシュと併用すると**融合済みの重みがキャッシュに焼き付いて次のジョブへ漏れる**＝速度ではなく**正しさ**の問題なので、黙って降格せず止める。 |
| **G-B** | `dit_cpu_load=False` × `keep_resident=1` | **warn ＋ auto-off**（`"on->off"`） | block swap の退避コピーとキャッシュが同居して CPU 側に重みが二重化する。生成結果は変わらないのでジョブは完走させ、キャッシュだけ諦める。 |
| **G-C** | `block_swap_prefetch=False` × `keep_resident=1` | **warn ＋ auto-off**（`"on->off"`） | 同期スワップは毎ステップ GPU→CPU の退避コピーを行うため、キャッシュと合わせて実測 +11.4GB の二重化になりコミット（仮想メモリ）を圧迫する。先読み block swap（§44）は退避コピーを構造的に廃止しているので、そちらが on ならこの問題は起きない。 |

### 48.4 機械検証の結果

- **バックエンドの pytest（モック・アプリ venv）**: **899 passed**（**実バックエンド稼働中**の計測）。失敗1件は `test_backend_status_...` で、**ポート 18620 でバックエンドが稼働中だと必ず落ちる既知の環境依存テスト**（§47.2 と同一。本改修と無関係であることは確認済み）。
- **エンジン用仮想環境の新規テスト**（`.venv-engine` ＋ `--noconftest`）: **27 passed**。内訳は `tests/test_gemma_keep_resident_move.py` **5**（§47 の Gemma 移動選択ロジック）／`tests/test_registry_swap.py` **14**（8属性の同時差し替え・`replace` による他フィールド保存・属性欠落時の assert・OFF 時の `clear()`＋`gc`）／`tests/test_worker_keep_resident_resolve.py` **8**（キー欠落＝off・G-A の raise・G-B/G-C の auto-off と `"on->off"` 表記）。
- **フロントエンドの型検査**: `npm run typecheck`（`tsc -b`）**0エラー**。
- **フロントエンドの vitest**: **1675 passed**（**実バックエンド稼働中**の計測）。失敗7件は実バックエンドを起動しているときだけ走る `backend.integration.test.ts` の統合テストで、これも既知の環境依存である。
- **フロントエンドの lint**: **0**（エラーなし）。

### 48.5 実機ゲート表（2026-08-02 深夜〜08-03 未明・R1〜R7・R9〔エージェント担当分〕全項目合格。R8 はオーナー目視の別枠）

環境: RTX 4070 Ti SUPER 16GB／System RAM 64GB、real backend。**清浄環境**（SD 系プロセスをはじめ重量級の並行作業なし。§47.4 の計測が汚れていた反省を踏まえ、開始前に確認した）。**全10ジョブ完走・crash 0**。

| # | 内容 | 合格条件 | 実測 | 判定 |
|---|---|---|---|---|
| R1 | ON 2本＋計測用1本の連続投入 | 1本目 MISS → 2本目以降 HIT | 2本目以降は `quant-load` 行が消失、metadata は `"on"`。**HIT 時の前処理 5.32 秒**（ベースライン 68.6〜75.0 秒） | ✅ PASS |
| R2 | ON 4本の出力一致 | 全本 SHA256 一致 | 4本すべて `FE549395…AA02D5`（§47 の参照値と一致） | ✅ PASS |
| R3 | フィールドを省略したジョブ（＝OFF）でキャッシュが解放されるか | `"off"`＋解放ログ＋RAM 返却＋出力一致 | metadata `"off"`・解放ログ出力・**PrivateBytes 49.98GB → 30.54GB（19.4GB 解放）**・SHA256 一致（キャッシュを使わなくてもビット一致） | ✅ PASS |
| R4 | ON → OFF → ON の再ウォームアップ | MISS が復活し再びキャッシュされる | MISS 復活（前処理 74.98 秒）・SHA256 一致 | ✅ PASS |
| R5a | G-C ガード（`block_swap_prefetch=false` × ON） | 完走＋`"on->off"`＋理由付き警告 | 完走・`keep_resident_used="on->off"`・理由を明示した WARNING | ✅ PASS |
| R5b | G-B ガード（`dit_cpu_load=false` × ON） | 同上 | 完走・`"on->off"`・理由付き WARNING（`config.yaml` の一時編集で再現させ、**バイト一致で復元済み**） | ✅ PASS |
| R6 | chain（`/generate/chain`）で ON | 完走＋chain metadata に反映 | 完走・chain の metadata に `"on"`（`to_clip_request` の転記を確認） | ✅ PASS |
| R7 | コミット（仮想メモリ）のピーク | 記録のみ（閾値判定なし） | ピーク **99.75GB / 114.69GB＝87.0%**。ON/OFF のアイドル差 **+19.0GB** | ✅ PASS（記録） |
| R9 | MCP 経由 | スキーマに `keep_resident` があり完走する | 自前の stdio クライアントで `submit_generate` の引数スキーマを確認＋完走・metadata `"on"`・**HIT 時の前処理 7.82 秒** | ✅ PASS |

- **R1 のベースライン 68.6〜75.0 秒の出所**: 本ゲート中に発生した MISS ジョブ（1本目と R4 の再ウォームアップ）の実測である。§46.2 の 66.2／79.3 秒とは**計測条件が別**（あちらは §46 の一点計測ジョブの条件別平均）なので、両者を同じ数列として並べて比較してはならない。
- **R7 で §47.3 の G7 絶対基準（コミット 90% 閾値）を引き継がず「記録のみ」としたのは**、①当時の 96% が並行重量作業下の汚れた計測だったこと（§47.4 の訂正）、②RAM を見て自動で振る舞いを変える機構は作らないというオーナー方針（2026-08-02・§48.1-10）——の2点による。閾値による合否判定を置くと、作らないと決めた自動判定の代わりを人手で運用することになるためである。
- **Gradio 同梱 UI のトグルは mock テストのみ**である（実機での操作確認は行っていない。Gradio 側の実機確認はフロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §4-6 の Gradio 残件と同じ扱い）。
- **R7 のピークの読み方**: 87.0% という数字は **G-C の auto-off が起きたジョブの最中**に記録されたもので、`clear()` 直後のコミット返却の遅れと、キャッシュを捨てたことによる再ビルドとが一過性に重なった瞬間である。定常状態の値ではないため、上限に対する余裕の評価にそのまま使ってはならない。ON/OFF のアイドル差 +19.0GB のほうが、キャッシュが常時占有する量として実態に近い。
- **R8 は本表に無い**。R8 はフロントエンドの実機目視ゲートで、**オーナーが行う残ゲート**である（§48.9）。
- **MCP の Claude Code UI 経由の実機確認は別枠のまま**である。§39.5 の既存方針（エージェントが自前スクリプトで MCP クライアントを叩いた確認は「暫定✅」とし、Claude Code 本体の UI＝承認ダイアログや `/mcp` の実画面は別途オーナーが通す）を変更していない。R9 は前者に相当する。

### 48.6 メモリの実測値まとめ

| 観測点 | 値 |
|---|---|
| キャッシュが占有する CPU RAM（ワーカー PrivateBytes の ON/OFF 差） | **約19.4GB**（R3 の解放実測 49.98GB → 30.54GB） |
| 同上（アイドル時の ON/OFF 差） | **+19.0GB**（R7） |
| コミットのピーク | **99.75GB / 114.69GB＝87.0%**（R7。G-C auto-off ジョブ中の一過性） |
| ピーク VRAM | **不変**（GPU には何も常駐させない設計。§47.3 の G8 で ±3MB を確認済み） |
| HIT 時の前処理 | **5.32 秒**（R1）／**7.82 秒**（R9・MCP 経由）。ベースラインは 68.6〜75.0 秒 |

- **名目サイズ（DiT 約16.5GB＋Gemma 約8〜9GB＝計25〜27GB）と実測19.4GB の差について**: 19.4GB は**ワーカープロセスの PrivateBytes の ON/OFF 差の実測**であり、モデルの名目サイズの合計ではない。骨格が保持するのは量子化済み state dict とモジュールツリーであること、また OS のページキャッシュや共有マップに落ちる分がプライベート領域に計上されないことから、実測が名目合計を下回る。利用者向けの案内はこの実測（約20GB）を採る。

この実測をもって、利用者向けの案内は「**メモリ 64GB 以上を推奨（約20GB を常時占有します）。生成結果は変わりません**」に統一した（フロントエンドのヒント文・Gradio・MCP のツール説明・`README.md` すべて同じ趣旨）。

- **UI 文言の「約70秒→約10秒」という丸めの根拠**: HIT 時の前処理は本ゲートで 5.32 秒（R1）／7.82 秒（R9）、§47.3 のスパイクでは 9.3〜15.3 秒だった。利用者向けの文言はこの幅の**上側を含むように保守的に丸めて**「約70秒→約10秒」としてある（実測の最良値である 5.32 秒をそのまま謳わない）。

### 48.7 env 経路（`LTX_KEEP_RESIDENT`）撤去の移行メモ

**`LTX_KEEP_RESIDENT` は撤去した。真実源（single source of truth）は per-job フィールドへ一本化されている。**

- `services/ltx_runner.py` がワーカーの env へ入れていた `setdefault("LTX_KEEP_RESIDENT", "0")` を削除した。
- `engine/worker.py` の env 読みは `False` 固定へ置き換えた（パイプライン生成時の `keep_resident_weights` は常に `False`）。
- **旧手順の `$env:LTX_KEEP_RESIDENT="1"` は無効である。**設定しても何も起こらない。今後は `POST /generate`・`POST /generate/chain` の `keep_resident` フィールド（既定 `false`）で指定する。Gradio・MCP・AviUtl2 フロントエンドからは Settings の Acceleration 区画のトグルで切り替える。
- 本ログ内の各節の実験条件・環境記述（**§9・§10・§15・§20〜§36 など**。§10.2・§46・§47 を含む）に出てくる `LTX_KEEP_RESIDENT` の記述は、当時の実験手順を記録した歴史であり書き換えていない。同様の理由で `Docs/RESOLUTION_DURATION_CAPABILITY.md`・`Docs/SCALEUP_16GB_RESEARCH.md`・`Docs/NEXT_SESSION_HANDOFF.md`・`Docs/NEXT_SESSION_WORKORDER.md`・`Docs/IC_LORA_PHASE_A_STATUS.md` の該当箇所には本文を書き換えず注記を1行だけ添えた。
- **`GET /status` には意図的に載せていない。** `/status` は「サーバーが実際にできること」を書く場所であり、`keep_resident` は能力ゲート（使える／使えないの判定）を持たない——このマシンに RAM があるかどうかという利用者側の選択にすぎないためである。`attention_backend` の `sage_available` や `block_swap_prefetch_available` のような可否フラグは存在せず、クライアント側もボタンを封じない。

### 48.8 既知の残留・注意記録

- **OFF 時のコミット返却には遅れがある。** `clear()` ＋ `gc.collect()` を通した直後でも、OS がコミットを返すまでには間があり、その最中に次のジョブが再ビルドを始めると一過性にピークが立つ（R7 の 87.0% がまさにこれ）。異常ではない。
- **OFF 時に完全にはゼロにならない残留がある。** 実測で残りうるのは、先読み block swap が ON のときの DiT 分（次の `install()` まで）・pinned プール・`held_embed_cpu` **約1.9GB** の3つ。解放ログ自体にこの3点を書き出してあるので、RAM の減り方が期待と違うときはまずログを見ればよい。
- **G-B の再現には `config.yaml` の一時編集が必要だった。** `dit_cpu_load` はリクエストから切れないためで、検証後に**バイト一致で復元済み**であることを確認している。
- **LoRA 併用 × keep=1 の実機確認は未実施である。** R1〜R9 はすべて素の T2V で、LoRA を付けたジョブは一度も流していない。G-A ガードが止めるのは `gguf_per_layer_quant=False`（bf16 融合経路）だけで、**通常の per-layer quant 経路の LoRA は毎ジョブ attach/detach されキャッシュを汚染しない**——設計上は安全だが、実測はまだ取っていない。実機確認の項目はフロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §2-1 に起票した。
- **`keep_resident_used` は3値**（`"off"` / `"on"` / `"on->off"`）。`"on->off"` は G-B / G-C の auto-off が起きたことを意味し、`metadata.json` に残る。速度が期待どおり出ないときは、まずここを見ると原因が切り分けられる。

### 48.9 残るオーナーゲート（R8＝フロントエンド実機目視）

以下はオーナー本人の目視・操作でしか判定できないため、本テーマは**細目のオーナー目視ゲート待ち**の状態である。ただし**本命にあたる 7（実 GPU での高速化）は 2026-08-03 にオーナーが実機で確認して合格**しており、残っているのは見た目・文言・永続化・4経路といった細目である。フロントエンド側のチェックリストは[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §2-1／§2-2 に「何を操作して確認するか → どうなれば合格か」の形で起票済み。

1. Settings > Acceleration に「モデル骨格の常駐（ジョブ間キャッシュ）」トグルが表示される。※フロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) **§1-10**（prefetch 連動グレーアウト）を実装したあとは合格条件が変わる（先読み block swap が off のときだけグレーアウトしているのが正しい姿になる）。
2. **既定が off** である（初回起動時・localStorage が空のとき）。
3. **ON にしたときだけ**ヒント文（「メモリ64GB以上を推奨…生成結果は変わりません」）が出る。
4. localStorage に永続化され、AviUtl2 を再起動しても選択が保たれる。
5. **4経路すべて**（Create／Chain／バッチA2V／バッチi2v-long）で、ON のときリクエストに `keep_resident` キーが載る。
6. 日本語・英語の両方で文言が自然である。

### 48.10 オーナー実機の三者併用交互対比較（768p/257f、2026-08-03記録）

オーナーが768p/257fで attention・block_swap_prefetch・keep_resident の交互対比較を実機実施した（実施日時2026-08-02 22:18〜22:42・UTC、metadata.jsonの`created_at`で確認）。4ジョブとも1280x768・257フレーム・t2v・同一プロンプト・8ステップ・seed 1687351733・LoRAなし・distilledで条件を揃えている。

| attention | block_swap_prefetch | keep_resident | 生成時間 | ジョブID（先頭8桁） |
|---|---|---|---|---|
| sdpa | off | off | 257.31秒 | 12f0fa3d |
| sage | on | off | 202.04秒 | fe6db822 |
| sage | on | on（1回目＝キャッシュ MISS・構築） | 228.63秒 | b152172b |
| sage | on | on（2回目＝HIT） | 165.0秒 | 263c64b4 |

- 全部offに対し、フルスタック定常（HIT）は約36%短縮。try1の+26.6秒は骨格キャッシュ構築の入場料である（`keep_resident_used="on"`・ガード降格なし）。
- sageの3ジョブの`output.mp4`はSHA-256完全一致（`926CE1BD8CA627F8...`）——§44（prefetch）・§48（keep_resident）で個別に確認済みのビット一致性が、768p/257fでの三者併用でも保たれることをオーナー実機で裏付けた。
- これは768p/257fにおける attention・prefetch・keep_resident 三者併用の初の実測である。

この165.0秒を分母にした§3-49（fused GGUF dequant+GEMM）の再検算: 逆量子化の見た目の割合は約11%に育つが、融合で実際に削れるBF16往復分は約2.3秒=約1.4%のままで、クローズ判断は不変（2026-08-03ディスカッション）。
7. 実 GPU で、2本目以降の生成が体感で速くなる（前処理の待ちが消える）。→ **✅ 合格（2026-08-03・オーナー実機確認）。「生成が大幅に高速化すること」を確認。**

**フォローアップ（2026-08-03 オーナー指示）**: prefetch=off 時の UI 連動グレーアウトをフロントエンド台帳[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) **§1-10** として起票した。§48.3 の G-C（`block_swap_prefetch=False` × `keep_resident=1` → warn ＋ auto-off）はサーバー側の安全装置として正しく働いているが、利用者から見ると「トグルを On にしたのに効かない」という見え方になりうる。そこでフロントエンド側で、先読み block swap が off のあいだは骨格常駐トグルを自動的に off にしてグレーアウト（操作不能）にし、prefetch を on に戻すと解除する。**本節のガードそのものは変更しない**（UI の見せ方だけの改修である）。

**追補（2026-08-03ディスカッション・attn2 K/Vキャッシュ案の検討）**: テキストcross-attention（attn2）のK/V射影はプロンプト不変ゆえ全11パスでキャッシュ可能ではないかという案を検討した（成立すればto_k/to_vのper-layer逆量子化スキップ込みで約2秒/ジョブ・ビット一致の見込みだった）。判定は**no-go**。本番3種のGGUF（distilled/Sulphur/10Eros）すべてがメタデータで`cross_attention_adaln: true`を設定していることをGGUFバイナリから直接読んで実測確認した（wheelの`model_configurator.py`はキー欠落時のみFalseを返すのみ）。このためattn2へ渡るコンテキストは各ブロックの`prompt_scale_shift_table`＋タイムステップ由来のscale/shiftで毎ステップ変調され、射影入力がパスごとに異なるため素朴なキャッシュは数学的に不成立である。変種（蒸留スケジュールのシグマ固定を利用した同一プロンプトのジョブ間キャッシュ）も、キャッシュ実体約2.2GB（11パス×48ブロック×2射影×256トークン×4096次元）に対し利得約2秒で見送った。位置づけは§3-49（fused GGUF dequant+GEMM）と同様、「さらなる高速化が必要になったとき」の再訪候補ですらなく、前提条件（AdaLN変調）が崩れない限り不成立という記録である。

---

## 49. ★IC-LoRA Depth（深度制御）・Deblur（ぼけ除去）の追加＝実装完了・機械検証（pytest・自己診断・型検査・vitest）全PASS・**G0〜G4・G-fix1〜G-fix3の全ゲート合格（2026-08-04）。テーマ完結（残：ヒント文言の表現修正のみ）**（2026-08-03実装・2026-08-04全ゲート合格）

> **正本＝[`ICLORA_DEPTH_DEBLUR_WORKORDER.md`](ICLORA_DEPTH_DEBLUR_WORKORDER.md)**（経緯・確定方針・公式ワークフロー解読結果・実装内容・残ゲートの定義）。本節はその**実測値の記録**である。フロントエンド台帳[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §4-8 に残っていた未対応IC-LoRA 4種のうち、Depth と Deblur の2種を製品化した。前処理器は **Video-Depth-Anything Small（vits・Apache-2.0）**、Deblur は Lightricks 公式 `LTX-2.3-22b-IC-LoRA-Deblur`。**G0・G1・G2・自動テスト・再ホスト検証・Deblur OOM修正の完走確認・参照動画ローダーのreserved膨張根治（第3修正・§49.10）に加え、G-fix3（Deblur品質目視）とG4（既存canny/pose/upscalerの回帰）も2026-08-04にオーナー実機で合格し、全ゲートが揃った（§49.11）。残るのはヒント文言（不正確な「出力と同じ解像度で処理」という表現の削除）の修正のみである。第3修正・G-fix3・G4に対応する実装・テスト差分のコミット・プッシュは未実施。デプロイも未実施。** → **実施済み（2026-08-04、`e273052`〜`9701fbe`）。ヒント文言の修正を含め、コミット・プッシュ・webuiの再ビルド／再デプロイまで完了している。**

### 49.1 一次資料の実測（公式ワークフローJSONの直接解読・Deblurのヘッダ実測）

**公式ComfyUIワークフロー `LTX-2.3_ICLoRA_Union_Control_Distilled.json`** をJSONとして直接解読し、深度前処理ノードのパラメータ実値を読み出した（推測なしの転記）。

| 項目 | 実値 | 本実装 |
|---|---|---|
| モデル種別 | `vits`（Small） | 同一 |
| `input_size` | `518` | 同一 |
| `max_res` | `960` | 同一 |
| 精度 | `fp32`（autocast無効） | 同一 |
| 出力形式 | グレースケール（近＝白） | 同一（クリップ全体でのmin-max正規化1回） |

**留保**: 同ワークフローで実際に配線が完成しているのは **canny 経路のみ**で、depth と pose のノードは「例示」の位置づけである。したがって上表は「公式が depth に使うと示している前処理器とその値」の一次資料としては有効だが、「公式が動作保証している完成品の depth 経路」ではない。**G1 で公式実装との数値比較を合格条件に採らなかった理由の一つがこれである**（§49.3 末尾）。

**Deblurアダプタの safetensors ヘッダ実測**（`ltx-2.3-22b-ic-lora-deblur-0.9.safetensors`・906,071,437バイト）:

| キー | 実測 | 帰結 |
|---|---|---|
| `reference_downscale_factor` | **実在・値 `"1"`** | 既存のメタデータキー判定がそのまま `kind=control` へ自動分類する。**configスキーマの拡張は不要**と確定 |
| `reference_temporal_scale_factor` | **無し** | 動作中wheelが時間係数を適用できない制約に抵触しない。方針の再検討は不要 |

値が **1**（参照を縮小せずに条件付けに使う）であることが、エンジン側の縮小係数ガード緩和（§49.4）が必要になった直接の理由であり、同時に Deblur の VRAM 増（stage-1 の総トークン数が、**参照なしを1として、縮小係数2で1.25倍・係数1で2.0倍＝既存の制御系比で約1.6倍**）の理由でもある。**VRAM の実測は G3 で行う（未実施）。** → **§49.11で実測済み（ジョブ全体ピーク13.9GB）。**

### 49.2 G0 前処理単体スモーク（2026-08-03実測・エンジン用仮想環境＋GPU）

エンジン用仮想環境（**torch 2.9.1／numpy 2.4、xformers・decord は無し**）で、**追加インストールを一切せずに動く**ことを確認した。

| 観測点 | 実測 |
|---|---|
| xformers不在フォールバックの実体 | **素のsoftmax実体化**（N×Nのスコア行列を丸ごと確保する上流実装）。1920×1088で **reserved 14.7GB** |
| SDPAへ差し替えた場合 | **約4.4GB**（reserved）。**速度も向上** |
| 差し替えによる数値差 | **fp16の許容誤差の1/20以下** |
| 本実装のVRAM | **約4GB固定** |
| 本実装のスループット | **1280×768で約15fps** |
| `easydict` | キーワード引数の入れ物としてしか使われていない → 素の `dict` へ置換（エンジン用仮想環境に未導入のため） |

- **この実測を受けて SDPA 化を「任意の最適化」ではなく必須として実装に組み込んだ**（`vda/video_depth_anything/dinov2_layers/attention.py` と `vda/video_depth_anything/motion_module/attention.py` の2箇所）。後者は上流が `baddbmm` の `alpha` としてスケールを掛けており、それが必ずしも `1/sqrt(head_dim)` ではないため、SDPA へは `scale=self.scale` を**明示的に**渡してある。前者は SDPA が同じ `1/sqrt(head_dim)` を内部で掛けるので明示の `q * self.scale` を落とした。
- あわせて xformers の import ガードを3ファイルから削除した。ガードが握る `except ImportError` 分岐は素の `print(...)` を実行するもので、**ワーカーの標準出力はフレーム化されたプロトコル通信路**であるため、そこへ文字列が漏れると通信が壊れる。削除後に残る経路は、xformers が無いときに上流が元々通っていた経路そのもの＝本スモークが実測した構成である。

### 49.3 G1 深度制御信号の性質ゲート＝全項目PASS（2026-08-03実測）

`engine/preprocess/depth_g1_gate.py`（本テーマで新設）で実施。**本番と同じ経路**（`get_processor("depth")` ＋ `driver.preprocess_video`）を1回走らせ、mp4へ書き出す直前のフレームを取り出して測る作りなので、測定対象は実ジョブが生成するものと同一である。条件は**実クリップ 1280×720・41フレーム**。

| # | 内容 | 合格条件 | 実測 | 判定 |
|---|---|---|---|---|
| (a) | 近い被写体が白か（近／遠の矩形を指定して平均輝度を比較） | 近 > 遠 ＋ 余裕 | 近 **184.07** ／ 遠 **15.49** | ✅ PASS |
| (b) | 静止画素のフリッカ（元動画がほぼ動いていない画素だけで測ったフレーム間の深度の揺れ） | 閾値 **2.0** 以下 | 平均 **0.561** ／ p95 **0.693** | ✅ PASS |
| (c) | mp4書き出し往復の階調劣化（バンディング） | PSNR 閾値 **6.0** 以上 | **35.67dB** | ✅ PASS |
| — | 速度 | 記録のみ | 41フレームを **4.35秒** | ✅ PASS（記録） |

- **(b) が「静止画素だけ」で測る理由**: 連続フレームの生の差分は、フリッカと**本物の動き**を混同する。元動画側の差分が小さい画素（`--static-delta` 未満）に限れば、そこは深度も動いてはならないので、フリッカだけを取り出せる。
- **(c) の閾値 6.0 の出所**: ドライバの既存エンコーダ（`mp4v` fourcc。canny/pose が既に通っている経路）の実測から較正した値である。**この項目が捕まえるのはコーデック由来の一般的な劣化ではなく、階調の崩壊**である。生き残った異なるグレー階調の数を数えるバンディング指標も併せて出している。
- **(a) は矩形を指定しない場合、報告のみで合否判定しない**（絵としての最終判断はオーナーの目視＝G3に委ねる設計）。
- **公式実装との数値比較を G1 に採らなかった理由**: ①比較には別環境の構築が必要、②§49.1 のとおり公式ワークフローの depth 経路は配線が完成しておらず、突き合わせ先として信頼できる基準にならない——の2点。最終的な絵の正しさは既存の `outputs/visual_review/` 運用に従い G3 のオーナー目視で判定する。

### 49.4 実装箇所一覧（設計判断の要点つき）

- **`engine/pipeline/fast_video_pipeline.py`**: 縮小係数の読み取りを**先頭1本のみ → 全LoRA走査**へ。wheel の読み取り関数は「1と宣言されている」場合と「キー自体が無い」場合の**両方で1を返す**ため、`safetensors` のヘッダを別途開いて**キーの存在そのもの**を確かめ、**キーを持つLoRAだけが投票**する方式にした。宣言値が2種類以上あれば矛盾としてエラー（**1と2の混在も矛盾**——参照動画は1つの解像度で1回だけ読まれるので、片方のアダプタに訓練時と違うスケールの参照を黙って食わせることになる）。**どのLoRAもキーを持たなければエラー**。あわせて `factor <= 1` 拒否を **`factor >= 1` 許容**へ緩め、下流の `assert` を**明示的な検証と例外**へ置換した（`python -O` では assert が消えるため）。割り切れ判定には手を触れていない（係数1では元々発火しない）。チェーン側は同メソッドを共用するため独自修正は不要。
- **`api/errors.py`・`api/generate.py`・`api/generate_chain.py`**: 新エラー **`REFERENCE_REQUIRES_CONTROL_LORA`（422）**。ガード緩和で消える「参照動画＋画風LoRAのみ」の誤用検出を、**API層の逆方向チェック**として作り直した（**単発・チェーンの両方**）。従来はこの誤用を縮小係数ガードが偶然弾いていたが、それは**アップロードもジョブ開始も済んだあとで落ちる**という不親切な失敗の仕方だった。フロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) **§4-11 の①**を本テーマの副産物として解消したことになる。
- **`engine/preprocess/vda/`（新規）**: Video-Depth-Anything の**推論コアのみ**をベンダリング（git追跡対象。`vendor/` は `.gitignore` 除外のため使えない）。Apache-2.0 の `LICENSE` 全文と、**改変点の完全な一覧を含む `README.md`** を同梱。クラス名・モジュール構成・属性名・コンストラクタ引数は**一切不変**（`strict=True` で読み込むため）。改変は4点のみ＝①相対import化（`from utils.util import ...` は `sys.path` 上の別の `utils` へ黙って結びつく危険があった）②`easydict` 除去 ③SDPA化2箇所（§49.2）④xformers import ガード削除3ファイル（§49.2）。上流の `run.py` / `app.py` / `benchmark/` / `loss/` / `utils/dc_utils.py`（`decord`・`matplotlib`・`imageio` を引き込む）は取り込んでいない。`.pth` は git 外・インストーラ経由。
- **`engine/preprocess/depth.py`（新規）**: `DepthProcessor`。公式 `infer_video_depth`（**32フレームの移動窓・10フレームの重なり・窓どうしの整合処理**）をそのまま呼ぶ薄いラッパー。重いimport（torchvision＋DINOv2スタック）は `_ensure_loaded()` 内に**遅延**（DWPose と同じ作法。トップレベルのimport失敗が canny/pose を道連れにするのを防ぐ）。ジョブごとにロードし、制御動画を書き終えたら `release()` でGPUから追い出す（16GBぎりぎりの生成デノイズ中に深度の重みを残さない）。
- **`engine/preprocess/base.py`・`driver.py`・`__init__.py`**: 新プロトコル **`VideoProcessor`（クリップ単位）** を `FrameProcessor`（フレーム単位）と並置し、ドライバがどちらを持つかで分岐する。深度は時間方向の移動窓とクリップ全体の正規化を使うため1枚ずつでは処理できない。**`frame_cap`**（デコード打ち切り）を追加——深度は与えられた分をすべて正規化に使うので、生成で使わないフレームまで処理すると時間を無駄にするうえ**グレーの割り当てレンジがずれる**。
- **`engine/worker.py`**: `_preprocess_frame_cap(msg)`。**単発＝`num_frames`／チェーン＝`clips[0]["num_frames"]`**（参照条件は先頭クリップのstage-1にしか付かないため）。キーが無ければ `None`（全デコード＝従来どおり）。**`frame_cap` は `preprocess == "depth"` のときだけ渡す**ので、**canny/dwpose の経路はバイト不変**であり、ログ行も `cap=` の部分は該当するときにしか付かない（Phase C で記録した実行ログと文字単位で一致し続ける）。
- **`config.yaml.example`・`config.py`・`services/lora_registry.py`**: `depth-control`（**union-control のファイルを共用**・`preprocess: depth`）と `deblur`（前処理不要のため**文字列形式**でパスのみ）の2エントリ登録。`IcLoraEntry.preprocess` のリテラル型に `"depth"` 追加。**Gradio の静的フォールバック一覧には手を触れていない**（`/config` 不達時だけの死に枝と確認済み）。
- **`gradio_ui/adapters.py`・`i18n.py`・`ui.py`**: `ADAPTER_FRIENDLY` に2件、英日それぞれ**ヒント3種**（アスペクト比／Depth の推奨値／Deblur の書式とVRAM）。**アダプタ選択に連動して出し分ける仕組みは既存に無く、そのためだけの新UI機構は足していない**（既存の `note_ref128` と同じ常時表示の注記）。
- **フロントエンド**（`Nz-LTX23-frontend-AviUtl2`）: `webui/src/i18n/strings.ts`（英日の同内容ヒント3種）／`GenerationForm.tsx`（参照動画セクションに3行）／`api/types.ts`（`"depth"`）／`bridge/mockBridge.ts`（`depth-control` マップ形式＋`deblur` 文字列形式のフィクスチャ）。**選択肢そのものは `/config` 経由で自動的に増える。** 詳細は[`DEVLOG.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/DEVLOG.md) §59。

**値の自動セットはしない**（オーナー決定）。深度の推奨 0.6 は**ヒント文の表示のみ**である。またヒント文は、推奨 0.6 の対象が **②`conditioning_attention_strength`（制御追従度）**であり **③LoRAアダプタ強度は 1.0 のまま**であるという区別を明記している（③を下げると参照が滲み込む＝bleed-through と公式が警告している）。3つのノブの切り分けは **§28.1** で照合済みで、0.6 の出典は Lightricks 公式ドキュメントの記載である。

### 49.5 自動テストの結果（2026-08-03実測・いずれも**実バックエンド未起動時**の計測）

| 対象 | 結果 | 着手前 |
|---|---|---|
| アプリ用仮想環境の pytest | **910 passed / 9 skipped** | 900 passed |
| エンジン用仮想環境（`tests/test_ic_lora_engine_conditioning.py` ＋ forward 系・`--noconftest`） | **28 passed** | 15 passed |
| `engine/preprocess/preprocess_selfcheck.py`（前処理の自己診断） | **11 / 11 PASS** | 新規 |
| Gradio 系テスト | **277 緑** | — |
| フロントエンド vitest | **1673 緑** | — |
| フロントエンド `npm run typecheck`（`tsc -b`） | **0エラー** | — |

- **`preprocess_selfcheck.py` を pytest ではなくスクリプトにした理由**: エンジン用仮想環境には `fastapi` が無いため `tests/conftest.py` を収集できず、アプリ用仮想環境には `cv2` も `torch` も無い。`block_swap_prefetch_selfcheck.py` と同じ切り分けである。GPU も重みも不要で、ドライバは偽のプロセッサで、`DepthProcessor` は純粋な配列変換ヘルパだけを叩く。11項目の内容は、`get_processor` の解決と不正値の拒否／深度だけが `VideoProcessor` であること（canny/dwpose の誤ルーティングが起こりえないこと）／クリップ単位分岐が全フレームを1回で順序どおり渡すこと／`frame_cap` の先頭切り出し／`frame_cap=None` の従来動作／FPSと解像度の保存／両分岐＋失敗時の `release()` 実行／フレーム数不一致の loud fail／`_inference_size` の 960 境界と偶数化／`_to_control_frames` の近＝白・クリップ全体正規化・3チャンネル同一・元解像度復元／チェックポイントの探索先の一致。
- **フロントエンド `npm run typecheck`（`tsc -b`）を使っている**のは既存の規律どおりで、`npx tsc --noEmit -p .` は偽合格するため使わない。

### 49.6 再ホストの検証（2026-08-03）

`Rootport/Nz-LTX23-weights`（既存の公開・非gatedリポジトリ）へ2ディレクトリを追加。**上流とのSHA-256一致・匿名（ログイン無し）でのダウンロード成功・README／NOTICE の更新**をいずれも確認済み。

| ディレクトリ | 内容 | サイズ |
|---|---|---|
| `ltx-2.3-ic-lora-deblur/` | `ltx-2.3-22b-ic-lora-deblur-0.9.safetensors` | 906,071,437バイト |
| `preprocessors-vda/` | `video_depth_anything_vits.pth` ＋ `LICENSE`（Apache-2.0 全文） | 116,452,112バイト（2ファイル計） |

`LICENSE` を同梱しているのは、これが重みリポジトリ内の他ファイル（LTX-2 Community Licence）と**異なるライセンス**であり、`.pth` と必ず一緒に運ばれる必要があるためである。

**`scripts/install_ltx.ps1`**: 独立したダウンロード呼び出しを2本追加し、検証表（`$required`）を **14項目 → 16項目**へ拡張した。

> **兄弟ディレクトリ方式を採った理由（容量チェックの構造的な罠）**: インストーラのスキップ判定は Check ディレクトリの**再帰的サイズ合計**である。Deblur の906MBを既存の `models/ltx-2.3-ic-lora/` の**中**に置くと、同ディレクトリの合計は 1,308,930,638 → 2,215,002,075 になる。すると **654MB の union-control ファイルを失っているマシンでも合計 1,560,536,723 となり、既存の Min 値 1,000,000,000 を上回ってスキップ**してしまう。以後は再実行のたびに「union-control MISSING」が出続け、しかも自力では直せない恒久的な行き詰まりになる。これは Gemma tokenizer で実際に起きた事故の**逆方向の再発**である（あちらは大きなファイルが不在の兄弟**ディレクトリ**を覆い隠した。こちらは新しいファイルが不在の兄弟**ファイル**を覆い隠す）。新しい Check ディレクトリは自分の中身だけで測られるので、この事故が構造的に起こりえない。
>
> **命名も僅差で助かっている**: `models/preprocessors-vda` は DWPose の `models/preprocessors` の**兄弟**であって子ではないため、互いのサイズ合計が混ざらない。もし `models/preprocessors/vda/` にしていたら、その116MBが DWPose の判定を水増しし、欠けた135MBの `dw-ll_ucoco` を覆い隠していた。ダウンロード対象を絞る glob も `preprocessors/*` は `preprocessors-vda/...` に一致しないため混線しない。
>
> **Min 値**: `ltx-2.3-ic-lora-deblur` = **900,000,000**（唯一のファイルが検証表の対象なので、失えば 0 に落ちて再ダウンロードが走る）／`preprocessors-vda` = **110,000,000**（`LICENSE` の 11,356バイトは検証表の対象外＝欠けても MISSING にならないため、**LICENSE の有無で判定が変わらない**値を選んだ。両方あり＝116,452,112でSKIP、LICENSEのみ欠損＝116,440,756でもSKIP、`.pth` 欠損＝11,356でダウンロード）。**疑似環境で3ケースの実測トレースと反例テストを実施済み。**

### 49.7 G2 mock 通し＝全項目PASS（2026-08-03実施）

**実施方法**は §23.3／§34.5 の前例に倣った——scratchpad へコピーした**隔離 config**（`backend: "mock"`・**ポート 18902**・出力先とアップロード先も scratchpad へ隔離）で**実 uvicorn を起動**し、HTTP 経由で叩く。**リポジトリ側のファイルおよび `outputs/`・`uploads/` への書き込みはゼロ**であることを確認済みである。

| # | 内容 | 合格条件 | 実測 | 判定 |
|---|---|---|---|---|
| 1 | `depth-control` ＋ 参照動画 ＋ `conditioning_attention_strength=0.6` の**単発生成** | 202 → `completed`・metadata に正記録 | 202 → `completed`。`metadata.json` に `preprocess: depth`／参照ID／`0.6` が正しく記録された | ✅ PASS |
| 2 | `deblur` ＋ 参照動画の**単発生成** | 202 → `completed` | 202 → `completed` | ✅ PASS |
| 3 | **1クリップ chain**（Create 画面の A2V 相当。`clips[0]["num_frames"]` 側の分岐を通る） | 202 → `completed` | 202 → `completed`（`num_frames=25`） | ✅ PASS |
| 4 | **2クリップ chain ＋ 制御系** | 422 で拒否（既存仕様の維持） | **422 `LORA_CONTROL_UNSUPPORTED_IN_CHAIN`** | ✅ PASS |
| 5 | 参照動画 ＋ **画風系 LoRA のみ** | 422 で拒否（§49.4 の新設チェック） | **単発・chain の両方で 422 `REFERENCE_REQUIRES_CONTROL_LORA`** | ✅ PASS |
| 6 | `/config`・`/loras` への新2件の出現 | 2件が出現し、kind が正しい | 両エンドポイントに出現。**`deblur` は実メタデータ由来で `kind: control`**（§49.1 の `reference_downscale_factor="1"` による自動分類が実サーバー上でも効いていることの確認） | ✅ PASS |

- **項目3 の `num_frames=25` について**: 17 で投げると **422** になるが、これは `overlap_frames=3` との**既存の境界バリデーション**が正しく働いた結果であり、**本改修のバグではない**。25 で正常に完走する。
- **項目5 は §34.6 の既知事項#2 を解消したことの実機確認である。** 従来「`reference_video_id` ＋ 画風系 LoRA のみ」はスキーマ上受理されてワーカー内の `_set_ic_job` で `RuntimeError` になっていた（早期422化は将来の改善余地として残されていた）。§49.4 で新設した `REFERENCE_REQUIRES_CONTROL_LORA` によりリクエスト時点の 422 になったことを、実サーバーで確認した。§34.6 側にも解消を追記済み。
- **`control_depth.mp4` の実生成確認は G2 の対象外である。** **mock エンジンは設計上、前処理を実行しない**（`_MockBackend` は `lora_paths`／`reference_video_path` を受理はするが無視する）。深度前処理が実際に動いて制御動画が書き出されることの確認は **G3 の対象**として残る（前処理そのものの性質は §49.3 の G1 で実測済み）。

### 49.8 残ゲート（**G3・G4とも全項目合格。テーマ完結**）— G3／G4

定義の正本は[`ICLORA_DEPTH_DEBLUR_WORKORDER.md`](ICLORA_DEPTH_DEBLUR_WORKORDER.md) §7。実測・進捗は本節以降（§49.9〜§49.11）に記録する。

| ゲート | 内容 | 状態 |
|---|---|---|
| **G3** オーナー実機 real | ①`depth-control` 通常経路を1本以上（制御追従度 0.6 を出発点に、元の動きや構図を保ったまま内容が置き換わることを目視）②`depth-control` の A2V 経由を1本以上 ③`deblur` を1本以上（**デフォーカスぼけ**の素材で。§49.1 のプロンプト2段構成）④**Deblur の VRAM 実測**——`reference_downscale_factor=1` のため stage-1 の総トークン数が、**参照なしを1として、縮小係数2で1.25倍・係数1で2.0倍（既存の制御系比で約1.6倍）**になる。スピルの起きない解像度・フレーム数の表と突き合わせ、**ヒント文の VRAM 記述を確定させる**（現在は具体値のない固定文言）。**あわせて `control_depth.mp4` が実際に書き出されることの確認もここに含まれる**（G2 は mock のため前処理が走らない） | ✅ **全項目合格**——①②は2026-08-03合格。③はOOM原因究明のうえ修正・完走確認済みで、**目視によるG-fix3判定も2026-08-04合格**（§49.11）。④VRAM実測は完了・reserved膨張は第3修正で根治（SHA完全一致確認済み。§49.10）・ヒント文はピーク13.9GBのため**警告不要**とオーナー裁定（§49.11。文言修正自体は別途残作業） |
| **G4** 既存アダプタの回帰 | `canny-control`・`pose-control`・`pixel-spatial-upscaler-x2` が本改修前と同一に動くこと。**縮小係数2の経路がバイト不変**であることを確認する（§49.4 でメタデータ読み取りを全走査に変え、ドライバに `frame_cap` を足しているため） | ✅ **合格（2026-08-04）**。基準3本（`24c67fd`）と現行3本（`e273052`）がSHA-256完全一致（§49.11） |

**デプロイは未実施である。** 初期実装のコミット・プッシュ済み（backend `fd6d43f` / frontend `d375028`、2026-08-03）。**Deblur OOM修正はコミット済み（backend `e45a27b`）。§49.10の参照動画ローダー修正（第3修正）と§49.11のG-fix3・G4に対応する実装・テスト差分は、機械検証・実機ゲートまで完了しているが、コミット・プッシュは未実施**（`git status`にエンジンパイプライン・テストの差分が残っている）。 → **実施済み（2026-08-04、`e273052`〜`9701fbe`）。コミット・プッシュ済みで、webuiも再ビルド・再デプロイ済み。**

### 49.9 Deblur参照エンコードOOMの原因究明と修正（2026-08-03）

**経緯**: §49.8のG3でDeblur（係数1）1280x768/257fがOOM（ジョブ`92a95e94`）した件を調査した。落下地点は**参照動画のVAEエンコード**（非タイル・`torch.cuda.empty_cache()`の保護外）と特定した。

**第1修正（不合格）**: 係数1のときだけ参照動画のVAEエンコードをタイル化し、あわせて事前クリーンアップを入れた。しかしG-fix1の再走（1280x768/257f・ジョブ`447084b7`）は`tiled_encode`実行中にOOMし**不合格**だった。この失敗を解析したところ真因が判明した——**torch 2.9.1のbf16 Conv3dは、入力がcontiguousレイアウトのときim2colフォールバックを取り**、「入力ch×27×出力体積×2B」というサイズの一時行列を確保する（実測がオフライン予測と比率1.00で一致）。タイル化だけでは、タイルがcontiguousレイアウトを経由する限りこの一時行列の確保を避けられない。

**第2修正（合格）**: 係数1のときだけ、参照エンコーダのConv3d重み**42本**を`channels_last_3d`へ切り替え、cuDNNの直接カーネル経路（im2colを経由しない）へ載せた。`finally`節で**元のcontiguousへ復元**する——同一ジョブのstage2で行うキーフレーム再エンコードがビット一致を要求するためである。復元後にも`empty_cache()`によるクリーンアップを入れてある。この変換により非ビット一致となるのは**係数1の参照潜在のみ**（rel_rms 1.2e-2／cos 0.99993）で、実測は許容範囲内である。

**ジョブ台帳**（すべてseed 424242・同一プロンプト）:

| ジョブID | 位置づけ | 条件 | 結果 | SHA |
|---|---|---|---|---|
| `c322130d` | G-fix0（修正前基準・非タイル） | 896x512/121f | 122.93秒で完走 | `180a5f57...` |
| `4202415b` | 中間（第1修正のみ・タイルcontiguous） | 896x512/121f | 完走。**記録上無効（最終比較には使わない）・出力ファイルは保持** | `651e7cc6...` |
| `447084b7` | 第1修正のみでのG-fix1 | 1280x768/257f | `tiled_encode`実行中にOOMで**不合格**（この失敗の解析でim2col真因が確定） | — |
| `bda91a2b` | G-fix1再走（第2修正込み・合格） | 1280x768/257f | **270.02秒で完走** | `9a8a047d...` |
| `d486cbee` | 小条件最終A/B | 896x512/121f | 143.22秒で完走（`c322130d`との目視比較用ペア） | `31f077d8...` |
| `b2bec008` | chain経由1クリップ+deblur | 1クリップ・deblur | **136.13秒で完走**（chain配線・復元→stage2再利用の統合確認。復元失敗ログゼロ） | — |

**実測（`bda91a2b`）**: 参照エンコード区間はallocatedが**1450→peak 3006MB**（差分約1.56GB＝オフライン予測1.2GB＋動画テンソル0.36GBとほぼ一致）。変換したConv3dは42本。

**未解決の観察（要追跡・調査中）**: `bda91a2b`（大条件）の参照エンコード中、**reservedが39,846MBまで膨張**した（実確保3GBの13倍）。Windowsは`expandable_segments`に非対応であり、36タイルの形状ばらつきでセグメントが蓄積し、WDDM経由でホストへ溢れつつ完走したと見られる。オーナーのタスクマネージャ観察（専用GPU天井→共有GPUメモリ上昇→回収→健全化）と整合する。小条件（`d486cbee`）はreserved 2,780MBで問題ない。参考として、旧OOM（`92a95e94`）時の別観察は、専用15.7/16GB・共有40.3GB・RAM100%（`keep_resident=on`の19.4GB常駐も寄与）だった。

> **訂正（§49.10）**: 上記の「断片化／タイル形状ばらつき」という推定は誤りだった。真因は参照動画ローダー（wheel の `load_video_conditioning`）がフレームを1枚ずつGPU上で`torch.cat`連結する実装であることによる、確保総量がフレーム数の2乗に比例する膨張だった。第3修正（CPU組み立て版への置換）で根治し、SHA完全一致まで確認済み。詳細は§49.10。

### 49.10 参照動画ローダーの2乗則膨張の特定と根治（第3修正、2026-08-04）

**真因確定**: §49.9末尾の「未解決の観察」として残っていたreserved 39.8GB膨張の犯人は、第2修正（channels_last_3d切替）が扱ったタイル処理やレイアウトではなく、**wheel側の`load_video_conditioning`がフレームを1枚ずつGPU上で`torch.cat`連結する実装**だった。連結を重ねるたびに確保総量が増えていくため、**確保総量はフレーム数の2乗に比例**する。640×384×257フレームでの理論値46.4GBは、オフライン実測46.8GBとほぼ一致した。**Deblur固有の問題ではなく**、縮小係数2（既存の canny/pose 等）でも本番条件で+2.2GBの膨張を実測しており、Windows環境では共有メモリへ溢れ続けて青天井になる性質を持つ。§49.9の「断片化・タイル形状ばらつき」という旧推定は誤りだったため、ここで訂正する。

**修正内容**: `_load_video_conditioning_cpu`を`engine/pipeline/common.py`へ公開名`load_video_conditioning_cpu`として移設し、chain側に重複していた同等の定義を削除して両パイプライン（fast／chain）で共用する形にした。fast側の参照読み込みをこのCPU組み立て版へ差し替え、縮小係数2以上の枝でのみ`.to(device)`を1回追加する。テストはmetaデバイス方式の転送検証を1本追加し、エンジンvenv **32 passed**・アプリvenvは全緑（既知の環境依存1件を除く）。

**ゲート実測（2026-08-04、すべてseed 424242）**:

- **G-3rd-α（オフライン）**: ローダー置換前後の潜在は、縮小係数1・係数2の両方で`torch.equal`=True（ビット完全一致）。ローダー区間のreservedは、係数1でwheel版47,488MB→CPU版660MB、係数2で4,810MB→648MBへ縮小した。
- **G-3rd-0（実装前基準）**: upscaler＋参照896×512/121fがジョブ`fecff551`（SHA `8630ca90...`）で完走。小条件deblurの再実行はジョブ`df8831da`でSHAが`d486cbee`と完全一致し、再現性n=2が成立してSHAゲートが有効化された。
- **G-3rd-1（修正後）**: 3本ともSHA完全一致——upscaler=ジョブ`edfdc50f`（`8630ca90...`一致）、小条件deblur=ジョブ`eceea948`（`31f077d8...`一致）、大条件deblur 1280×768/257f=ジョブ`d4bedb69`（`9a8a047d...`一致、249.70秒で完走）。大条件のジョブ全体peak reservedは**39,846MB→13,932MB**に縮小し、参照エンコード区間のreserved増分は**38,162MB→1,348MB**に縮小した。

**ゲート判定の注記（較正ミスの記録）**: 計画時点の数値目標「区間reserved増分が数十MB級／1GB未満」は、オフラインの**ローダー単体**計測を根拠にした較正ミスだった。本番ログの計測区間はローダーとエンコード本体の両方を含み、エンコード本体が正当に2.2〜3.2GBの実確保を使うため、reservedがその約1.1倍に収まる現状はアロケータとして健全である（病理だった「実確保の13倍」という乖離が1.13倍へ正常化した）。よって**G-3rd-1は合格**と判定する。残存の主因はエンコード本体の正常な作業領域と、既知のスコープ外項目（縮小係数2 untiledのim2col問題。[`ACCELERATION_RESEARCH_NOTES.md`](ACCELERATION_RESEARCH_NOTES.md)記載済み）である。

**付随の解消**: `peak_vram_reserved_mb`の計測汚染も解消し、大条件の13,932MBが実態を表す値になった。

**残作業**: 第3修正自体のコミット・プッシュは未実施（`git status`に`engine/pipeline/common.py`・`chain_pipeline.py`・`fast_video_pipeline.py`・テストの差分が残っている）。**目視によるG-fix3判定とG4（既存アダプタの回帰）は§49.11で実施し、いずれも合格した。** デプロイは引き続き未実施。 → **実施済み（2026-08-04、`e273052`〜`9701fbe`）。第3修正・G-fix3・G4に対応する差分はコミット・プッシュ済みで、webuiも再ビルド・再デプロイ済み。**

**テスト**: エンジンvenv **31 passed**（新規：channels_last変換／contiguous復元ヘルパーの往復ビット一致を確かめるCPUテスト）。アプリvenvは全緑（既知の環境依存1件を除く）。

**注記**: `StateDictRegistry`がGPU常駐に変わる変更が入った場合、本修正（Conv3d重みのレイアウト切替と復元）の安全性——重み参照が他ジョブ・他経路と分離されていること——を再検証すること。

### 49.11 G4（既存アダプタの回帰）合格・G-fix3（Deblur品質目視）合格（2026-08-04）

**G4＝既存 canny/pose/upscaler の回帰・バイト不変**: テーマ着手前コミット `24c67fd` を基準に、共通条件（896×512/121f・シード424242・8ステップ・sage/prefetch on・resident off・同一プロンプト・参照動画バイト一致〔機械照合済み〕）で基準3本を生成し、現行コミット `e273052` で同一条件のまま3本を再生成して突き合わせた。**3本ともSHA-256完全一致。**

| アダプタ | 基準（`24c67fd`）ジョブID | 現行（`e273052`）ジョブID | 判定 |
|---|---|---|---|
| canny-control | `d770eceb` | `6c1b866e` | ✅ SHA-256完全一致 |
| pose-control | `70c3d305` | `e809ead7` | ✅ SHA-256完全一致 |
| pixel-spatial-upscaler-x2 | `fb87dce7` | `77462587` | ✅ SHA-256完全一致 |

**手順**: `config.yaml`のdepth／deblurエントリを一時退避 → `24c67fd`をcheckoutして基準3本を生成 → mainへ復帰・config復元 → 現行3本を再生成 → SHA突き合わせ。

**G-fix3＝Deblur品質目視・合格**: A/Bペア（`c322130d`対`eceea948`、同一シード）で視覚差なし。タイル継ぎ目のちらつきは知覚不能レベル。大条件`d4bedb69`もデブラー効果が正しく出ており品質問題なし（いずれもオーナー主観評価）。

**ヒント文言の方針確定**: Deblurのピーク実測13.9GB（既存アダプタ比+約0.4GB）は16GB推奨環境に収まるため、**VRAM警告は不要**とオーナーが裁定した。ヒント文中の不正確な「出力と同じ解像度で処理」という表現の削除は、別途実施する残作業として残る。

**本節をもって §49 の G0〜G4・G-fix1〜G-fix3 はすべて合格し、本テーマは完結する。** 残るのはヒント文言の表現修正と、§49.10（第3修正）・本節（G-fix3・G4）に対応する実装・テスト差分のコミット・プッシュ・デプロイのみである。 → **実施済み（2026-08-04、`e273052`〜`9701fbe`）。第3修正・G-fix3・G4に対応する差分はコミット・プッシュ済みで、webuiも再ビルド・再デプロイ済み。**

---

## 50. ★GGUF逆量子化の分解計測（fused dequantカーネルの採否判断）＝Phase 0マイクロベンチ＋Phase 2実機プローブ計測 完了・**判定＝GO**（2026-08-04）

> **正本＝本節。** 逆量子化（dequantization。GGUFファイルの中で圧縮された形で持っている重みを、計算に使える形式へ展開する処理）が1ジョブ中で何秒使っているのかを、マイクロベンチ（小さな処理を単体で繰り返し測るベンチマーク）と実機計測の両方から確定させた記録である。台帳[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-49 の再訪条件が成立したことを受けたもので、[`ACCELERATION_RESEARCH_NOTES.md`](ACCELERATION_RESEARCH_NOTES.md) が「次の一歩」として掲げていた分解計測の実施にあたる。**結論は GO**——逆量子化の1カーネル化で**1ジョブあたり約20.1秒（現ベースライン約146秒の約14%）**の短縮が見込め、GO基準（3.0秒）を大きく上回った。実装は別プランの承認後に行う。本計測のために一時的に入れた計測プローブ（probe。処理時間を記録するためだけの計測用コード）は**撤去済み**で、リポジトリのコードは計測前と1バイトも違わない状態に戻してある。

### 50.1 目的と経緯

- **再訪条件の成立**: §3-49（fused GGUF dequant+GEMM）は2026-08-01に「削減できる絶対量が約2.3秒しかない」としてクローズしていた。§48.10で三者併用フルスタックが165秒まで縮み、以後は1〜2秒級の改善を積み上げる段階に入ったため、**2026-08-04にオーナー裁定で再訪**した。
- **今回の対象は「GEMM融合」ではない**。文字通りの fused dequant+GEMM（逆量子化と行列積〔GEMM。General Matrix Multiply、行列同士の掛け算〕を1つのカーネル〔GPU上で走る処理のかたまり〕に融合する案）は、行列積側を自作カーネルで置き換えることになり cuBLAS に速度で勝てないため **no-go のまま**である。今回測ったのは、その手前にある「**逆量子化処理そのものを1カーネルにまとめる**」という、より小さく安全な案の採否である。
- 現状の逆量子化は、PyTorchの通常演算（eager実行）を積み重ねた実装になっており、1回の呼び出しで **18〜33個ものカーネル**を起動し、そのたびに中間データをGPUメモリへ書き出しては読み直している。これを1つのカーネルにまとめれば、中間データの往復も、カーネル起動の回数も、まとめて消える。

### 50.2 Phase 0＝マイクロベンチ（逆量子化カーネル単体の天井測定）

融合後にどれだけ速くなるかの**下限**を、実際に測って確定させた。手書きカーネルを書く前に `torch.compile`（PyTorchの自動融合コンパイラ）で同じ計算を融合させ、その速度を「手書きカーネルなら最低でもこれくらいは出る」という下限値として使う方法である。

- 環境: RTX 4070 Ti SUPER・torch 2.9.1+cu128・triton-windows 3.5.1。ベンチプロセスでは `TORCH_COMPILE_DISABLE` を外している（本番はこれを1に固定しているため、後述のとおり本番実装では `torch.compile` は使えない）。
- 入力は毎回別バッファへ回して作業セットを約150MB（L2キャッシュ48MBの3倍超）にし、実際の逐層逆量子化と同じ「キャッシュに載っていない重みを読む」条件を再現した。
- 実測したメモリ帯域（DRAMの読み書き速度の上限）は **611.9 GB/s**（カタログ値672 GB/sの91%）。以下の「理論天井」はこの実測値を基準にしている。

**形状の分布**（`LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`）: 量子化テンソルは全1632本がひとつの `transformer_blocks.N` の塔の中にあり（4096次元＝映像ストリーム、2048次元＝音声ストリーム。1パスで両方が動く）、**1632本すべてがDiTの1回のforward（順伝播）ごとに逆量子化される**。

| 量子化形式 | テンソル数 | 要素数 | 量子化要素に占める割合 |
|---|---|---|---|
| Q4_K | 1242 | 12,182,224,896 | 65.7% |
| Q6_K | 322 | 5,595,201,536 | 30.2% |
| Q5_K | 68 | 772,931,584 | 4.2% |
| **合計** | **1632** | **18,550,358,016** | 100% |

**代表形状の実測**（50回の中央値・CUDAイベント計測。「カーネル数」は1回の呼び出しで起動されるカーネルの個数）:

| 形式 | 形状 | eager ms | カーネル数 | 融合後 ms | カーネル数 | 速度比 R |
|---|---|---|---|---|---|---|
| Q4_K | (32, 2048) | 0.286 | 27 | 0.106 | 2 | **×2.7** |
| Q4_K | (32, 4096) | 0.705 | 27 | 0.070 | 2 | **×10.0** |
| Q4_K | (2048, 2048) | 0.368 | 27 | 0.069 | 2 | **×5.3** |
| Q4_K | (2048, 4096) | 0.569 | 27 | 0.067 | 2 | **×8.5** |
| Q4_K | (4096, 4096) | 1.397 | 27 | 0.080 | 2 | **×17.4** |
| Q4_K | (16384, 4096) | 5.427 | 27 | 0.286 | 2 | **×19.0** |
| Q5_K | (32, 2048) | 0.410 | 33 | 0.068 | 2 | **×6.1** |
| Q5_K | (32, 4096) | 0.417 | 33 | 0.068 | 2 | **×6.2** |
| Q5_K | (2048, 2048) | 0.430 | 33 | 0.069 | 2 | **×6.2** |
| Q5_K | (2048, 4096) | 1.711 | 33 | 0.099 | 2 | **×17.3** |
| Q5_K | (4096, 4096) | 2.324 | 33 | 0.074 | 2 | **×31.6** |
| Q5_K | (16384, 4096) | 9.104 | 33 | 0.300 | 2 | **×30.3** |
| Q6_K | (2048, 2048) | 0.312 | 18 | 0.062 | 2 | **×5.0** |
| Q6_K | (2048, 4096) | 1.577 | 18 | 0.068 | 2 | **×23.1** |
| Q6_K | (4096, 4096) | 2.159 | 18 | 0.086 | 2 | **×25.0** |
| Q6_K | (4096, 16384) | 8.651 | 18 | 0.332 | 2 | **×26.1** |

**実際の呼び出し回数で重み付けした速度比**（1パス1632回の内訳をそのまま重みにしたもの）:

| 形式 | 1パスあたり呼び出し数 | eager ms/パス | 融合後 ms/パス | R |
|---|---|---|---|---|
| Q4_K | 1242 | 1131.3 | 104.9 | ×10.79 |
| Q6_K | 322 | 712.3 | 34.9 | ×20.41 |
| Q5_K | 68 | 111.9 | 5.9 | ×18.83 |
| **合計** | **1632** | **1955.4** | **145.7** | **R = ×13.42** |

- **加重速度比 R = 13.42。** 絶対値では、逆量子化は現状 **1パスあたり1.96秒**のGPU時間を使っており、融合すれば146ミリ秒＝約1.81秒の短縮になる。
- **理論天井**: すべての呼び出しが611.9 GB/sの帯域上限で、カーネル起動の待ち時間ゼロで動いたとすると1パス80.2ミリ秒＝**R_max = ×24.4**。`torch.compile` はすでにこの天井の55%に達しており、大きな形状では帯域上限の94〜98%（例: Q4_K (16384,4096) で601 GB/s／上限612 GB/s）に到達している＝**メモリ帯域の理論限界に達している**状態である。残る差はすべて小さい形状のカーネル起動の下限（1回あたり0.05〜0.07ミリ秒）によるもので、手書きカーネルが `torch.compile` を大きく上回る余地は小さい。つまり R = 13.42 は**堅実な下限**として使える。
- **数値の一致**: 16構成すべてで融合版の出力は eager と**ビット単位で完全一致**（bf16のビットパターンの相違0個、最大絶対差0.0、NaN 0個）。`view(torch.float16)` による型の読み替えやニブル（4ビット）シフトの連鎖を Inductor が壊さないことを確認した。
- **初回コンパイルの費用**: モデルが必要とするのは16通りの（形式, 形状）の組み合わせで、キャッシュが空の状態からの合計コンパイル時間は **11.1秒**（初回1.2秒・中央値0.6秒・最大1.2秒）。2回目以降はディスクキャッシュが効いてほぼ0秒。手書きカーネルなら0秒である。
- **`max-autotune` は採らない**: より攻めた最適化モードである `max-autotune` は、この処理では一貫して**遅かった**（加重 R が 13.42 → 9.25 に低下）。CUDA graph trees が有効になり呼び出しごとに入力コピーのカーネルが1つ増えるためで、入力バッファを回している本ワークロードではそのコピーは純粋な無駄になる。
- **Windows固有の不具合を1件踏んだ**（将来 `torch.compile` を本番経路で検討する場合の申し送り）: `mode="max-autotune"` は当初**すべての形状**で `OverflowError: Python int too large to convert to C long` を出して失敗した。発生元は `torch/_inductor/runtime/static_cuda_launcher.py:244`（`_StaticCudaLauncher._launch_kernel`）で、64ビット値をC言語の `long`（Windowsでは32ビット）へ入れようとしたことによる。`torch._inductor.config.use_static_cuda_launcher = False` で回避した。通常モードの `torch.compile` はこの問題の影響を受けない。

### 50.3 Phase 2＝実機計測の方法（一時プローブ・撤去済み）

「1パス1.96秒」が実際のジョブでどれだけの合計になるのかを、本番と同じ経路で測った。

- **計測プローブの設計**: `engine/profiling/dequant_probe.py`（一時ファイル）を新設し、`engine/worker.py` から**環境変数 `LTX_PROFILE_DEQUANT` が設定されているときだけ import する**5行を追加した。未設定なら import すらされないため、通常運用の経路には一切の影響がない。
- **走行中に同期を挟まない**: CUDAイベント（GPU上に打つ時刻の印）を**40,000個まとめて事前確保**しておき、計測中は印を打つだけ・時間の読み出しはジョブ完了後にまとめて行う設計にした。走行の途中でCPUとGPUを待ち合わせる（同期する）処理を入れると、それ自体が計測を歪めるためである。実際に使ったイベントは36,884個で、プールの枯渇は起きていない。
- **計測対象**: (a) 逆量子化1回ごとのGPU時間とCPU側の発行時間、(b) 逐層量子化の linear（全結合層）forward 全体の同、(c) フェーズ（テキストエンコード／stage1／アップサンプル／stage2／VAEデコード／動画エンコード）ごとの壁時計時間。GEMM（行列積本体）の時間は「linear全体 − 逆量子化」の差として導出した。
- **条件**: 1280×768・257フレーム・t2v・8ステップ・seed 1687351733・LoRAなし・distilled・attention=sage・block_swap_prefetch=on・keep_resident=on。§48.10 の三者併用フルスタック（165.0秒の行）と**同一条件・同一プロンプト**である。
- **撤去**: 計測完了後、`engine/worker.py` の5行と `engine/profiling/` ディレクトリを削除した。撤去後の `git diff engine/worker.py` は**差分ゼロ**である。

**ジョブ表**（6本。J0〜J3がプローブあり、J4・J5がプローブなしの対照）:

| ラベル | ジョブID | プローブ | 骨格キャッシュ | 生成時間 | 出力SHA-256 |
|---|---|---|---|---|---|
| J0 | `0943db83-a964-4ed3-9951-6a2ecaa8e77c` | あり | MISS（構築） | 198.73秒 | ✅一致 |
| J1 | `7ddac942-1258-4778-86fb-f0de04fc9dc7` | あり | HIT | 147.60秒 | ✅一致 |
| J2 | `6f98fc1e-f315-4794-b8b2-285dfaf943cc` | あり | HIT | 145.49秒 | ✅一致 |
| J3 | `9a01ec9e-0c3f-489f-9cb1-5a156bf105f3` | あり | HIT | 145.25秒 | ✅一致 |
| J4 | `c8945dfc-19f0-4608-acfc-a76706582d70` | なし | MISS（構築） | 195.56秒 | ✅一致 |
| J5 | `adba0643-eeb9-49a0-8b82-4a70f9b9425e` | なし | HIT | 146.52秒 | ✅一致 |

出力フォルダは `outputs/<ジョブID>/` である。SHA-256は6本すべてが `926CE1BD8CA627F8637141FCDFE0E5147A34F9E7582C58D47B5543E0AA817B87` で完全一致した（§48.10の `263c64b4` と同じ値＝**プローブは出力に影響していない**）。

### 50.4 Phase 2の結果（本命数値）

代表値は HIT 3本の中央値である **J1** を採る（逆量子化GPU合計で J1=21.77／J2=21.83／J3=21.67 秒の中央値）。

| 指標 | 実測 |
|---|---|
| **逆量子化のGPU合計 D** | **21.77秒/ジョブ**（呼び出し18,288回） |
| うち Q4_K | 12.71秒（13,950回） |
| うち Q6_K | 7.82秒（3,590回） |
| うち Q5_K | 1.23秒（748回） |
| フェーズ別: テキストエンコード | 0.97秒（336回） |
| フェーズ別: stage1 denoise | 15.14秒（13,056回） |
| フェーズ別: stage2 denoise | 5.66秒（4,896回） |
| linear forward 全体のGPU時間 | 71.74秒 |
| うちGEMM（差として導出） | 49.97秒 |
| ジョブ全体の壁時計 | 147.60秒 |

**フェーズ別の壁時計**（同 J1）:

| フェーズ | 秒数 |
|---|---|
| テキストエンコード（Gemma） | 3.81 |
| stage1 denoise | 47.37 |
| アップサンプル | 0.34 |
| stage2 denoise | 55.11 |
| 音声VAEデコード | 0.17 |
| 映像VAEデコード（7回） | 33.07 |
| 動画エンコード（ファイル書き出し） | 36.77 |

- **読み方の注意**: 映像VAEデコードの33.07秒は動画エンコードの36.77秒の**内側**に含まれる（デコードした分から順に書き出す構造のため、単純に足すと二重計上になる）。フェーズの合計が壁時計を超えて見えるのはこのためである。
- **検出されたパス数は12**（パス0＝Gemmaのテキストエンコード336回、パス1〜11＝DiTのforward 11回×1632回）。11パス＝stage1の8ステップ＋stage2の3ステップで、`1632 × 11 + 336 = 18,288` と呼び出し数が完全に一致する。
- **denoise部分だけを取ると**、逆量子化は 15.14+5.66 = **20.80秒**を11パスで使っており、1パスあたり **1.89秒**。Phase 0のマイクロベンチが予測した1.96秒/パスと**3.6%の差**で一致した（別々の方法による相互検証）。

### 50.5 副次発見(1)＝CPU側の発行時間がGPU実行時間を大きく上回る

今回もっとも重要な発見である。

| 指標 | GPU実行時間 | CPU側の発行時間 |
|---|---|---|
| 逆量子化 | 21.77秒 | **93.19秒** |
| linear forward 全体 | 71.74秒 | **94.67秒** |

- 逆量子化はGPU上で21.8秒しか動いていないのに、**CPU側（Pythonの実行とカーネル発行）に93.2秒**かかっている。1回の呼び出しで18〜33個のカーネルを発行するため、GPUが計算する時間よりもCPUが「命令を出す」時間のほうが長くなっている。
- さらに linear forward 全体のCPU発行時間94.67秒は、denoise 2ステージの壁時計（47.37+55.11＝**102.48秒**）のほぼ全部を占める。**denoiseはGPUの計算力ではなく、CPU側のカーネル発行で律速している**というのが実態である（Windows の WDDM ドライバモデルはカーネル発行のオーバーヘッドが大きく、Phase 0の小形状で見えた「起動律速」と同じ現象がジョブ規模で現れている）。
- したがって1カーネル化は、GPU時間21.8秒を削るだけでなく、**CPU発行の93.2秒側も同時に削る**ことになる。カーネル発行の回数は18〜33回から2回へ落ちるため、CPU側の削減率はGPU側と同等かそれ以上になる可能性が高い。**期待削減20.1秒は上振れしうる**、というのが本発見の意味である（本節の判定はこの上振れを一切当てにせず、GPU時間だけで計算している）。

### 50.6 副次発見(2)＝ベースラインが§48.10より約19秒速い

今回のHITジョブは **145.25〜147.60秒**で、§48.10に記録された同一条件の165.0秒より**約19秒速い**。

- プローブの有無とは無関係である。プローブなしの対照 J5 も **146.52秒**で同じ水準だった。
- 推定原因は**計測環境の清浄さ**である。§47.4でオーナー自身が確定させたとおり、§48.10の計測時はブラウザでの配信視聴・AviUtl2での動画編集・別の画像生成アプリの常駐といった重量級の並行作業が走っていた。今回はエージェント単独の清浄な環境で走らせている。
- **判定への影響**: 期待削減の絶対量（20.1秒）は分母によらないが、割合表示は分母次第で変わる。本節では保守的に**今回の実測ベースライン146秒を分母**として「約14%」と記載している（165秒を分母にすれば約12%）。

### 50.7 検証ゲートの結果

| ゲート | 基準 | 結果 |
|---|---|---|
| G-A 出力の同一性 | 6本すべてSHA-256一致 | ✅ 全6本が `926CE1BD8CA627F8637141FCDFE0E5147A34F9E7582C58D47B5543E0AA817B87`（§48.10の `263c64b4` とも一致） |
| G-B プローブの計測侵襲 | 生成時間の差が1%以内 | ✅ **0.41秒＝0.28%**（プローブあり HIT 3本の平均146.11秒 対 プローブなし J5 の146.52秒）。しかも符号は**プローブありのほうが速い**＝完全にノイズの範囲内 |
| G-C 相互検証 | 2つの独立な方法の値が一致 | ✅ マイクロベンチ外挿 1.96秒/パス 対 実機実測 1.89秒/パス（差3.6%） |
| G-D イベントプールの健全性 | 枯渇しないこと | ✅ 40,000個中36,884個使用・枯渇0 |

### 50.8 判定＝GO

期待削減 S を、Phase 2の実測 D と Phase 0の加重速度比 R から計算する。

```
S = D × (1 − 1/R) = 21.77 × (1 − 1/13.42) = 20.15 秒
```

- **S ≈ 20.1秒／ジョブ**（現ベースライン約146秒の**約14%**）。**GO基準の3.0秒を6.7倍上回る**ため、判定は **GO** である。
- この数字は次の点でいずれも保守的である: (a) `torch.compile` の速度を手書きカーネルの下限として使っている、(b) §50.5のCPU発行時間の削減を一切勘定に入れていない、(c) 分母に今回の速いベースライン146秒を採っている。
- **実装方式は Triton による自作カーネル**とする。本番のワーカーは `TORCH_COMPILE_DISABLE=1` を強制しているため（`engine/worker.py:119` の `os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")`、および `services/ltx_runner.py:1166` の `env["TORCH_COMPILE_DISABLE"] = "1"`）、`torch.compile` をそのまま本番経路へ持ち込むことはできない。Triton なら初回コンパイル11.1秒の入場料も不要になる。
- **実装は本節の範囲外**である。別プランを立て、承認を得てから着手すること。

### 50.9 生データ

計測の一次データを本節に取り込んでおく（§44.3で外部ファイルの生データを失った教訓による。**外部ファイル参照にはしない**）。

**(a) Phase 0＝実測メモリ帯域（`clone()` の読み書き）**

| サイズ | GB/s |
|---|---|
| 8 MB | 1280.3（L2に載る外れ値。DRAMの値ではない） |
| 32 MB | 642.5 |
| 128 MB | 612.5 |
| 512 MB | 611.9 |

**(b) Phase 0＝形状ごとの詳細**（`amp` は「最小限必要なバイト数の何倍を実際に動かしているか」。eager が中間データをどれだけ無駄に往復させているかの指標）

| 形式 | 形状 | 要素数 | 最小転送量 | eager ms | eager GB/s | amp | eagerカーネル数 | CPU発行 ms | 融合 ms | 融合 GB/s | 帯域上限比 | R | コンパイル秒 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q4_K | (32, 2048) | 65,536 | 0.2 MB | 0.286 | 0.6 | 1041× | 27 | 0.288 | 0.106 | 1.6 | 0% | ×2.7 | 1.2 |
| Q4_K | (32, 4096) | 131,072 | 0.3 MB | 0.705 | 0.5 | 1285× | 27 | 0.313 | 0.070 | 4.8 | 1% | ×10.0 | 0.6 |
| Q4_K | (2048, 2048) | 4,194,304 | 10.7 MB | 0.368 | 29.2 | 21× | 27 | 0.348 | 0.069 | 155.9 | 25% | ×5.3 | 0.6 |
| Q4_K | (2048, 4096) | 8,388,608 | 21.5 MB | 0.569 | 37.8 | 16× | 27 | 0.562 | 0.067 | 321.9 | 53% | ×8.5 | 0.6 |
| Q4_K | (4096, 4096) | 16,777,216 | 43.0 MB | 1.397 | 30.8 | 20× | 27 | 1.359 | 0.080 | 534.7 | 87% | ×17.4 | 0.6 |
| Q4_K | (16384, 4096) | 67,108,864 | 172.0 MB | 5.427 | 31.7 | 19× | 27 | 5.191 | 0.286 | 601.0 | 98% | ×19.0 | 0.6 |
| Q5_K | (32, 2048) | 65,536 | 0.2 MB | 0.410 | 0.4 | 1425× | 33 | 0.382 | 0.068 | 2.6 | 0% | ×6.1 | 0.6 |
| Q5_K | (32, 4096) | 131,072 | 0.4 MB | 0.417 | 0.8 | 724× | 33 | 0.398 | 0.068 | 5.2 | 1% | ×6.2 | 0.6 |
| Q5_K | (2048, 2048) | 4,194,304 | 11.3 MB | 0.430 | 26.2 | 23× | 33 | 0.430 | 0.069 | 162.5 | 27% | ×6.2 | 0.6 |
| Q5_K | (2048, 4096) | 8,388,608 | 22.5 MB | 1.711 | 13.2 | 46× | 33 | 0.934 | 0.099 | 228.1 | 37% | ×17.3 | 0.6 |
| Q5_K | (4096, 4096) | 16,777,216 | 45.1 MB | 2.324 | 19.4 | 32× | 33 | 2.232 | 0.074 | 612.6 | 100% | ×31.6 | 0.7 |
| Q5_K | (16384, 4096) | 67,108,864 | 180.4 MB | 9.104 | 19.8 | 31× | 33 | 8.684 | 0.300 | 600.9 | 98% | ×30.3 | 0.6 |
| Q6_K | (2048, 2048) | 4,194,304 | 11.8 MB | 0.312 | 37.9 | 16× | 18 | 0.312 | 0.062 | 191.0 | 31% | ×5.0 | 0.9 |
| Q6_K | (2048, 4096) | 8,388,608 | 23.7 MB | 1.577 | 15.0 | 41× | 18 | 1.543 | 0.068 | 346.5 | 57% | ×23.1 | 0.7 |
| Q6_K | (4096, 4096) | 16,777,216 | 47.3 MB | 2.159 | 21.9 | 28× | 18 | 2.087 | 0.086 | 547.8 | 90% | ×25.0 | 0.7 |
| Q6_K | (4096, 16384) | 67,108,864 | 189.3 MB | 8.651 | 21.9 | 28× | 18 | 8.322 | 0.332 | 570.2 | 93% | ×26.1 | 0.7 |

**(c) Phase 0＝`max-autotune`（主要形状のみ・不採用の根拠）**

| 形式 | 形状 | 通常モード ms | max-autotune ms | カーネル数 | コンパイル秒 | ビット一致 |
|---|---|---|---|---|---|---|
| Q4_K | (2048, 2048) | 0.069 | 0.081 | 3 | 0.9 | ✅ |
| Q4_K | (4096, 4096) | 0.080 | 0.084 | 3 | 0.9 | ✅ |
| Q4_K | (16384, 4096) | 0.286 | 0.415 | 4 | 1.4 | ✅ |
| Q5_K | (2048, 2048) | 0.069 | 0.078 | 3 | 1.1 | ✅ |
| Q5_K | (4096, 4096) | 0.074 | 0.081 | 3 | 1.0 | ✅ |
| Q5_K | (16384, 4096) | 0.300 | 0.465 | 5 | 1.1 | ✅ |
| Q6_K | (2048, 2048) | 0.062 | 0.116 | 3 | 1.7 | ✅ |
| Q6_K | (4096, 4096) | 0.086 | 0.240 | 3 | 1.6 | ✅ |
| Q6_K | (4096, 16384) | 0.332 | 1.008 | 5 | 3.0 | ✅ |

**(d) Phase 0＝支配的な形状の内訳**（1パス1632本の内訳。「本数×形状」）

- Q4_K: 414×(2048,2048)、276×(4096,4096)、138×(32,4096)、138×(32,2048)、92×(2048,4096)、46×ずつ (16384,4096)/(8192,2048)/(2048,8192)/(4096,2048)
- Q6_K: 138×(2048,2048)、92×(4096,4096)、46×(4096,16384)、46×(2048,4096)
- Q5_K: 68本が細かく分散

**(e) Phase 2＝ジョブごとの主要数値**（単位はミリ秒。カッコ内は呼び出し回数）

| 指標 | J0（MISS） | J1（HIT・代表） | J2（HIT） | J3（HIT） |
|---|---|---|---|---|
| 壁時計（プローブ計測） | 194,838 | 147,596 | 145,479 | 145,241 |
| 逆量子化GPU合計 | 29,084.1（18,289） | **21,765.2（18,288）** | 21,832.0（18,288） | 21,667.6（18,288） |
| 逆量子化CPU発行合計 | 100,284.8 | **93,186.0** | 93,299.2 | 93,226.5 |
| ├ Q4_K（型12） | 12,820.6（13,950） | 12,713.6（13,950） | 12,789.3（13,950） | 12,648.4（13,950） |
| ├ Q6_K（型14） | 15,037.2（3,591） | 7,824.0（3,590） | 7,803.4（3,590） | 7,799.1（3,590） |
| └ Q5_K（型13） | 1,226.4（748） | 1,227.6（748） | 1,239.2（748） | 1,220.0（748） |
| ├ フェーズ: テキストエンコード | 970.7（336） | 966.2（336） | 970.1（336） | 973.7（336） |
| ├ フェーズ: stage1 | 15,192.3（13,056） | 15,140.4（13,056） | 15,239.6（13,056） | 15,056.9（13,056） |
| └ フェーズ: stage2 | 5,695.1（4,896） | 5,658.6（4,896） | 5,622.3（4,896） | 5,637.0（4,896） |
| linear forward GPU合計 | 71,919.5（18,596） | 71,737.7（18,596） | 71,871.5（18,596） | 71,700.2（18,596） |
| linear forward CPU発行合計 | 94,753.6 | 94,671.9 | 94,789.5 | 94,678.7 |
| GEMM（linear − 逆量子化） | 42,835.3 | **49,972.6** | 50,039.5 | 50,032.6 |
| 検出パス数 | 12 | 12 | 12 | 12 |
| イベント使用数／プール | 36,885／40,000 | 36,884／40,000 | 36,884／40,000 | 36,884／40,000 |

J0はキャッシュMISS（骨格構築）を含む1本目で、骨格構築中の逆量子化が「パス0」の外側（`other` 分類、7,226ms）に計上されている。Q6_KのGPU時間がJ0だけ突出して大きい（15,037ms 対 7,8xx ms）のは、この骨格構築時の初回逆量子化が同じカウンタに混ざるためである。**代表値には使わない**。

**(f) Phase 2＝J1のパスごとの逆量子化GPU時間**（単位ミリ秒。パス0＝Gemma、パス1〜8＝stage1、パス9〜11＝stage2）

| パス | 呼び出し | GPU ms | CPU発行 ms |
|---|---|---|---|
| 0 | 336 | 966.2 | 929.8 |
| 1 | 1632 | 2011.9 | 5144.2 |
| 2 | 1632 | 1889.3 | 5045.1 |
| 3 | 1632 | 1926.9 | 5088.8 |
| 4 | 1632 | 1877.0 | 5037.2 |
| 5 | 1632 | 1878.3 | 5036.9 |
| 6 | 1632 | 1849.9 | 5017.4 |
| 7 | 1632 | 1864.4 | 5027.1 |
| 8 | 1632 | 1842.7 | 5016.7 |
| 9 | 1632 | 1897.1 | 17310.9 |
| 10 | 1632 | 1890.3 | 17299.0 |
| 11 | 1632 | 1871.1 | 17233.0 |

stage2（パス9〜11）でCPU発行時間だけが3倍以上に跳ねているのは、stage2の解像度が上がりGEMM本体の実行が長くなるため、その裏でCPUが次のカーネルを発行し終えても待たされる時間が計上されるためである（GPU時間は各パスほぼ一定＝逆量子化の仕事量は解像度に依存しない）。

**(g) Phase 2＝条件とメタデータ**（J1の `metadata.json` より）

1280×768・257フレーム・24fps・t2v・8ステップ・seed 1687351733（`seed_used` も同値）・LoRAなし・distilled・`attention_used="sage"`・`block_swap_prefetch_used="on"`・`keep_resident_used="on"`・`peak_vram_reserved_mb=13902`・`generation_time_seconds=147.6`。プロンプトは§48.10と同一（`263c64b4` のもの）。

### 50.10 残課題

1. **台帳§3-49の書き換え**（オーナー判断後）→ **実施済み（2026-08-04）**。旧§3-49（fused GGUF、再訪条件つきクローズ）はフロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §1-11「Fused GGUF Dequantization Kernel（GGUF逆量子化の1カーネル化）」として起票し直した（本節のGO判定を受けたもの・未着手・プランモード承認待ち）。クローズ経緯はフロントエンド[`PENDING_TASKS_CLOSED.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS_CLOSED.md) §3-61に保存済み。
2. **Triton実装は別プラン** → **実施済み（2026-08-04）。実装・機械検証・実機ゲート・既定値反転の正本は §51 である。**
3. **UIトグルの名称案**: 「Fused GGUF Dequantization Kernel」。既存の `fused_gguf_dequant_gemm`（`metadata.json` に残っている旧名のフィールド）とは意味が異なるため、名前を流用しないこと → **その方針どおり別名 `fused_gguf_dequant_kernel` で実装し、旧 `fused_gguf_dequant_gemm` のほうは2026-08-04のオーナー裁定で完全撤去した（§51.1）。**
4. **プローブは撤去済み**。再計測が必要になった場合は本節50.3の設計（イベント事前確保・走行中同期なし・環境変数ガード）をそのまま再実装すればよい。

---

## 51. ★Acceleration第5弾（GGUF逆量子化の1カーネル化＝`fused_gguf_dequant_kernel`）＝Tritonカーネル3本による逆量子化の融合＝実装完了・機械検証（selfcheck・pytest・型検査・vitest）全PASS・**実機ゲートG1〜G8全項目合格・既定onへ反転済み（オーナー承認2026-08-04）**（2026-08-04実装／同日ゲート合格・既定反転。残：オーナー目視のみ）

> **正本＝本節。** §50の分解計測で「逆量子化（dequantization。GGUFファイルの中で圧縮された形で持っている重みを、計算に使える形式へ展開する処理）が1ジョブあたり実測21.77秒のGPU時間を使っている」ことが確定し、判定がGOになったのを受けて実装したものである。やったことは一言でいえば「**これまで18〜33個の細かいGPU処理に分かれていた展開作業を、量子化形式ごとに1個の自作GPU処理（Tritonカーネル）へまとめた**」。**生成結果は1ビットも変わらない**（現行実装とのビット単位一致を必須要件として設計・検証している）ことが、この機能の設計上の中心的な約束である。台帳はフロントエンド[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §1-11。

### 51.1 実装サマリ

- **新規モジュール3本**（`engine/gguf/`）
  - `dequant_triton.py`（234行）: 製品側の窓口。`enabled()` / `dequant()` / `set_job()` / `reset_job()` / `last_used()` の5つだけを公開し、状態（要求の有無・降格の掛かり方・呼び出し回数・型ごとの自己検証済みフラグ）をこのモジュール内に閉じ込めている。Tritonの読み込みは遅延（実際に必要になるまでimportしない）で、失敗したという事実もキャッシュする。
  - `dequant_triton_kernels.py`（314行）: Q4_K・Q5_K・Q6_Kの3本のTritonカーネル本体。
  - `dequant_triton_selfcheck.py`（698行）: 自己検証スクリプト（C1〜C10。§51.2）。
- **既存コードへの差し込みは1箇所だけ**: `engine/gguf/quant_service.py` の `dequantize_ggml_tensor` に「Tritonで試す→戻り値が `None` なら従来実装へ落ちる」という分岐を入れた（`:47` のimportと `:128-130`）。K量子化3形式・GPU上のテンソル・出力がbf16、という3条件すべてを満たすときだけ新経路に入る。GemmaのembeddingのようにCPU上で展開されるものは条件で弾かれ、従来どおりに動く。
- **落ちない設計**: Tritonが無い環境・カーネルが例外を出した場合・後述の自己検証で数値が一致しなかった場合、いずれも**黙って従来実装へ降格し、生成そのものは絶対に止めない**。警告はジョブにつき1回だけ出す。
- **型ごとの初回自己検証**: プロセス内で各量子化形式が初めて呼ばれたとき1回だけ、同じ入力を従来実装でも計算してビット単位で突き合わせる（費用は1ミリ秒未満）。「例外は拾えるが、間違った数値は拾えない」という穴を塞ぐための仕掛けである。
- **切替はジョブ単位で独立**: APIフィールド `fused_gguf_dequant_kernel`、実際に効いたかの記録は `metadata.json` の `fused_gguf_dequant_kernel_used`（`"off"` / `"on"` / `"on->off"`。`"on->off"` は「要求したが実際には適用されなかった」）。他のAcceleration項目（`attention_backend`・`block_swap_prefetch`・`keep_resident`）とは互いに独立で、どれか一つの状態に依存しない。
- **旧 `fused_gguf_dequant_gemm` は完全撤去した**（オーナー裁定2026-08-04）。受理するだけで生成に何の影響も与えていなかったモック項目で、API（`api/models.py`）・Gradio UI・i18nの2キー・フロントエンド（設定パネルのJSX・型・テスト）から削除し、新しいトグルが画面上の同じ位置を引き継いだ。**2026-08-01の「無効化したまま残す」という裁定（§43・§44.1-6）は、この裁定で上書きされた**。pydanticの既定が `extra=ignore` のため、古いクライアントが旧フィールドを送ってきてもAPIは壊れない。モック作法そのものは `vae_mode` が残るので失われない。

### 51.2 STEP1＝自己検証（selfcheck）の結果＝C1〜C10全PASS

`.venv-engine\Scripts\python.exe -m engine.gguf.dequant_triton_selfcheck` で実行する（engine用の仮想環境にはpytestが無いため、テストモジュールではなく単体スクリプトの形にしてある。sage・block swap prefetchと同じ作法）。**10項目すべてPASS・skipゼロ・終了コード0**。

| 番号 | 何を証明するか | 結果 |
|---|---|---|
| C1 | 実際に同梱している2つのGGUFファイルから（形式, 形状）の組み合わせを**機械的に全列挙**し、総当たりでビット一致を確認 | ✅ **31形状すべて一致**（Q4_K 14／Q5_K 10／Q6_K 7。最大は Gemma の Q6_K 262144×3840） |
| C2 | 手で組み立てた境界値ブロック（6ビットのスケール梱包 0x00/0x0F/0x3F/**0x80**/**0xFF**、qhの全ビット位置、Q6_Kの符号付きスケールの0x7F/0x80跨ぎ、fp16のゼロ・非正規化数・最小正規化数・最大正規化数） | ✅ 一致 |
| C3 | CPU上のテンソルはTritonに渡らず、従来実装から無変更で戻る | ✅ |
| C4 | IC-LoRAの `bf16 += delta` という上書き加算がオン／オフで同一。要素数が256の倍数でない形状（大きなバッファの一部を切り出した形）も含む | ✅ |
| C5 | カーネルに例外を注入すると、10回の呼び出しすべてが従来実装の結果と一致し、降格の記録（latch）は1回・警告も1回だけ | ✅ |
| C6 | 機能を要求したが対象テンソルが1本も無かったジョブは `"on->off"` と記録する | ✅ |
| C7 | 降格の記録が次のジョブへ持ち越されない | ✅ |
| C8 | 1ジョブ分の形状を連続で流しても新規コンパイルが発生しない | ✅ |
| C9 | 先読みblock swapのアリーナ（512バイト境界に整列した大きな連続領域の途中を切り出したテンソル）でもC1と同じ結果 | ✅ |
| C10 | `enable_fp_fusion=True`（浮動小数点の積和融合を許す設定）にした対照実験の記録 | ✅ 3形式ともビット一致。ただし**保険として製品では `False` を明示したまま維持**する |

**検証力の裏取り（変異テスト）**: 「全部PASSするのは、テストが甘いからではないか」を確かめるため、わざとカーネルを壊して自己検証が落ちることを確認した。Q6_Kの符号拡張（スケールのバイトを符号付きとして読む処理）を落とすと**10項目中6項目がFAIL**する。テストは実際に間違いを検出できる。

**速度（カーネル単体）**: 加重速度比 **R = 18.89**（1パスあたり eager 1823.7ミリ秒 → 融合後 96.6ミリ秒）。§50.2で `torch.compile` を下限の目安として測った R = 13.42 を上回った。

**`BLOCKS_PER_PROG` の凍結**: 1つのプログラムが担当するブロック数を4/8/16/32でスイープし、**8** で凍結した（`dequant_triton_kernels.py:89`）。`triton.autotune` は使っていない。

**初回コンパイル（JIT）の実費**: キャッシュが空の状態からの合計 **0.81秒**（Q4_K 483ミリ秒／Q5_K 184ミリ秒／Q6_K 147ミリ秒）。キャッシュが効いた2回目以降は0.12秒。形状に依存しない設計（ブロック数を実行時引数にした）にしてあるため、**1プロセスにつき1形式1回しかコンパイルが起きない**（`do_not_specialize` を明示）。Tritonのキャッシュ置き場は既定のまま（`~/.triton/cache`）で、環境変数による設定は追加していない。§50.2で `torch.compile` について記録した「16通りで合計11.1秒」という入場料が、Tritonの手書きカーネルでは0.81秒に下がったことになる。

### 51.3 テストゲート

| 対象 | コマンド | 結果 |
|---|---|---|
| バックエンド本体 | `.venv\Scripts\python.exe -m pytest -q` | ✅ **942件中941件PASS**。唯一のFAILは `test_mcp_registration.py::test_backend_status_structured_content_not_wrapped_and_reachable_false` で、これは「バックエンドに到達できないこと」を前提にするテストであり、実機ゲートのためにサーバーを起動したままにしていた環境要因である（コード起因ではない） |
| エンジン仮想環境 | `.venv-engine\Scripts\python.exe -m pytest --noconftest tests\test_worker_fused_dequant_resolve.py` | ✅ **9件PASS** |
| フロントエンド型検査 | `npm run typecheck`（＝`tsc -b`） | ✅ エラーなし |
| フロントエンド単体テスト | `npm run test`（vitest） | ✅ **1700件PASS**（109ファイル）。別枠の `backend.integration.test.ts`（10件）は実機バックエンドが同じポートを占有している状態では走らせられないため対象外 |

**新設した回帰テストのうち要点**: ①単発・chain・バッチA2Vの**3経路すべてでキーがワイヤーに載る**ことを、バックエンド側（`test_gradio_batch_runner.py`）とフロントエンド側（`fusedGgufDequantKernel.paths.test.ts`）の両方で押さえた。バッチA2Vだけは配線漏れが他のすべての自動テストをすり抜ける経路なので、専用の1本を置いてある。②chainの入力リストは末尾から数える添字で受け渡す作りのため、末尾の並び順を固定する回帰テストを更新した。

### 51.4 実機ゲート G1〜G8（全項目合格）

共通条件は§50.3と同一である（1280×768・257フレーム・t2v・8ステップ・seed `1687351733`・LoRAなし・distilled・`attention_backend=sage`・`block_swap_prefetch=on`・`keep_resident=on`）。既知の基準SHA-256は `926CE1BD8CA627F8637141FCDFE0E5147A34F9E7582C58D47B5543E0AA817B87`（§48.10・§50.3と同じ値）。

| ゲート | 合格基準 | 結果 |
|---|---|---|
| G1 ビット一致 | OFF 1本・ON 1本の両方が既知SHAと完全一致 | ✅ A1（off）・A2（on）とも一致 |
| G2 速度 | O,F,O,F,O,F,O の交互7本でONの中央値がOFFの中央値より5秒以上短い・SHA全一致 | ✅ **25.28秒短縮（約17.5%）**・7本すべてSHA一致 |
| G3 echo | 全ジョブのmetadataでON側が `"on"`・OFF側が `"off"`。`"on->off"` が1本でも出たらFAIL | ✅ 該当なし |
| G4 IC-LoRA併用 | canny-control＋参照映像あり、OFF→ONの1対でSHA完全一致＋ON側 `"on"` | ✅ SHA一致・**64.89秒→38.40秒** |
| G5 降格証明 | sdpa条件で、Tritonのディレクトリを一時的に退避した状態でONを要求 → 完走・両者のSHAが互いに一致・`"on->off"`・警告は1本だけ | ✅（詳細は下記） |
| G6 先読みblock swap off併用 | 完走・`"on"`・prefetch offの既知SHAと一致 | ✅ **233.70秒→207.45秒** |
| G7 chain | 2クリップのchainをONで完走・`"on"` | ✅ 34.24秒 |
| G8 バッチ | バッチA2V 1行をONで実行・完走・metadataのechoが `"on"` | ✅ 25.81秒 |

**全ジョブ表**（`generation_time_seconds` は `metadata.json` の値。「骨格」は `keep_resident` のキャッシュがHITしたかMISSしたか）

| ラベル | ジョブID | 条件 | 生成時間 | echo | attention | prefetch | keep_resident | peak reserved MB | 骨格 | 出力SHA-256 |
|---|---|---|---|---|---|---|---|---|---|---|
| A0（捨て） | `f76a699e-565a-454a-9da5-a6871ca38e67` | off | 198.74秒 | off | sage | on | on | 13912 | MISS | `926CE1BD…B87` |
| A1 | `68da98a8-4b6a-48b6-a20e-b4b3409763d2` | off | **148.96秒** | off | sage | on | on | 13902 | HIT | `926CE1BD…B87` |
| A2 | `048d6f34-091e-4bed-97ee-c96ae2b4276d` | **on** | **120.06秒** | on | sage | on | on | 13914 | HIT | `926CE1BD…B87` |
| A3 | `6407eaea-68f2-4d5d-96ca-b321b638bcd0` | off | **144.51秒** | off | sage | on | on | 13916 | HIT | `926CE1BD…B87` |
| A4 | `e2c878e1-a773-40b1-8061-8649661994f8` | **on** | **119.12秒** | on | sage | on | on | 13914 | HIT | `926CE1BD…B87` |
| A5 | `bbb2cef1-58cf-41f5-9911-8a7509166912` | off | **144.38秒** | off | sage | on | on | 13916 | HIT | `926CE1BD…B87` |
| A6 | `032a923f-e4aa-4535-a361-c6127ce74ee7` | **on** | **119.23秒** | on | sage | on | on | 13914 | HIT | `926CE1BD…B87` |
| G4 off | `9f2227a8-ac45-4afb-9b25-36d8392fa211` | off・IC-LoRA | 64.89秒 | off | sage | on | on | 12380 | HIT | `CC88642E…05C` |
| G4 on | `96e2e3f9-a9ba-491e-9b90-76714aa149a5` | **on**・IC-LoRA | **38.40秒** | on | sage | on | on | 12380 | HIT | `CC88642E…05C` |
| G6 off | `ae52d070-be49-4b5c-be38-94bdd3a82da2` | off・prefetch off | 233.70秒 | off | sage | off | on→off | 13932 | MISS | `926CE1BD…B87` |
| G6 on | `c0b48394-201e-46b0-bd00-d65dd3676cee` | **on**・prefetch off | **207.45秒** | on | sage | off | on→off | 13920 | MISS | `926CE1BD…B87` |
| G5 off | `da673d87-eee7-4c10-a4f1-dd6edf7b4a8f` | off・sdpa・Tritonあり | 271.72秒 | off | sdpa | on | on | 13914 | MISS | `73B629D9…754` |
| G5 on | `ec6bfcea-408b-4133-b76d-ef8418bbe695` | on要求・sdpa・**Triton退避** | 233.89秒 | **on→off** | sdpa | on | on | 13912 | MISS | `73B629D9…754` |
| G7 | `4baa2d38-c2b6-49a5-a857-5df61ca726ce` | **on**・2クリップchain | 34.24秒 | on | sage | on | on | 12292 | HIT | `78B07CA1…643` |
| G8 | `09461cfd-8524-4a7d-8617-a58e191a33cd` | **on**・バッチA2V 1行 | 25.81秒 | on | sage | on | on | 11896 | HIT | `17F4C1B6…678` |

SHA-256の全桁: `926CE1BD8CA627F8637141FCDFE0E5147A34F9E7582C58D47B5543E0AA817B87`（768p/257f t2v・sage）、`73B629D972F2A05A80C95F9F196B04810372C96280766D3478F7C3B03E76E754`（同・sdpa）、`CC88642E55663C17B8BA983859C42A220837026745DCF12ACC9D8A162D3AA05C`（G4のIC-LoRA条件）、`78B07CA17974FF0277DDD231798F4472022BE387703A502B57B46D67E9087643`（G7のchain）、`17F4C1B69DBD4280DD9C2071342D98802719FB7FAFF949417B5ECD759960B678`（G8のバッチA2V）。

**G2の統計（本命の速度数値）**

| 条件 | 3本の実測 | 中央値 |
|---|---|---|
| OFF（A1・A3・A5） | 148.96 / 144.51 / 144.38 秒 | **144.51秒** |
| ON（A2・A4・A6） | 120.06 / 119.12 / 119.23 秒 | **119.23秒** |

- 差は **25.28秒＝約17.5%短縮**。合格基準（5秒以上）を大きく上回った。
- §50.8の期待値（GPU時間だけから計算した約20.1秒）を**5秒ほど上回っている**。これは§50.5で予告したとおりで、1カーネル化はGPUの計算時間だけでなく**CPU側がカーネルを発行する時間（実測93.2秒）も同時に削る**ため、上振れが実際に現れた形である。
- 交互（off→on→off→on…）で7本走らせたのは、速度の比較だけでなく**状態がジョブをまたいで漏れていないことの検証**を兼ねている。7本すべてでechoが要求どおりに切り替わり、出力SHAも全一致した。

**G5（降格証明）の詳細**: sageはTritonに依存するため、Tritonを取り除いた状態でsage条件のゲートは実施できない。そこで**attention=sdpa条件**で行った。`.venv-engine` の `triton` ディレクトリを一時的に別名へ退避してサーバーを起動し、ONを要求した1本が完走することを確認したうえで、必ず元に戻す手順にしてある（実施後の復旧も確認済み）。結果は、①完走、②echoが `"on->off"`、③出力SHAが同条件のOFF（Tritonあり）と**互いに完全一致**（`73B629D9…754`）、④ワーカーログの該当警告は**2本のみ**で、その内訳は「Tritonのカーネルを読み込めなかった」という診断1本と「このジョブでは無効にして従来実装へ落ちる。生成は続行、遅くなるだけ」という降格通知1本である（同じ警告が繰り返し出ないことの確認）。生ログの該当行:

```
[ltx_worker] engine.gguf.dequant_triton: Triton dequant kernels unavailable: ModuleNotFoundError: No module named 'triton'
[ltx_worker] engine.gguf.dequant_triton: Fused GGUF dequant kernel disabled for this job (Triton kernels could not be imported) - falling back to the eager PyTorch dequant. The job continues, only slower.
```

**G6の補足**: `keep_resident_used` が `"on→off"` になっているのは、`block_swap_prefetch=off` との併用時に骨格キャッシュを自動的に降格させる既存のガード（§48の `_resolve_keep_resident`、メインメモリの二重確保を避けるためのもの）が想定どおり働いた結果である。本テーマとは無関係の既知挙動で、OFF/ON両方に等しくかかっている。

**VRAM**: `peak_vram_reserved_mb`（予約量）は off/on で**実質不変**（A1 13902 対 A2 13914 など、揺らぎの範囲）。一方 `peak_vram_mb`（実確保量）は **10617MB → 10230MB＝約386MB減**した。中間データの往復が消えたぶんだけ確保量が減っており、VRAMの観点でも悪化はない。

### 51.5 既定値の反転（オーナー承認2026-08-04）と最終検証

**裁定**: 実装計画では「STEP2の段階では既定off（既定のリクエストのキー集合を1つも増やさない純粋な加算にするため）とし、実機ゲート全PASSを確認したうえでオーナー承認を得てからonへ反転する」と定めていた。G1〜G8が全項目合格したことを受け、**2026-08-04にオーナーが既定onを承認**した。§44のS4（`block_swap_prefetch` の既定on反転）とまったく同じ前例に従う。

**反転した3箇所**（3つを同じ変更でまとめて動かすのが規律）:

1. `api/models.py` の `FUSED_GGUF_DEQUANT_KERNEL_DEFAULT = True`（`/generate`・`/generate/chain` の2つのFieldとMCPツールのimport元を兼ねる正本）
2. `gradio_ui/handlers.py` のミラー定数（Gradioはサーバーと同居していてもHTTP越しのクライアントなので、importせず自前のミラーを持つ）
3. フロントエンド `webui/src/shell/accelerationSettings.ts` の `FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT = true`

**送信規律の向きが反転する点に注意**: クライアント側（Gradio・MCP・フロントエンド）はいずれも「**サーバー既定と異なるときだけ明示的に送る**」という規律で書いてあるため、反転前は「onのときだけ送る」だったものが、反転後は「**offのときだけ送る**」になる。この規律で書いてあったからこそ、反転にあたって送信ロジック自体は1行も変えずに済んだ（§44.7が「値がTrueのときだけ送る」実装で踏んだ事故＝明示offが届かない、という落とし穴を最初から避けてある）。なお `services/ltx_runner.py`（サーバー→ワーカーの中継）だけは「解決済みの値がTrueなら送る」方式のままでよい（既定onになった結果、既定のジョブでもワーカーへキーが載るようになった）。凍結してあった「既定リクエストのキー集合」テスト2箇所は、この反転にあわせて `fused_gguf_dequant_kernel` を追加した——**このキー集合が増えてよいのは「実機ゲート合格後の既定反転」のときだけ**であり、そのことをテスト本体のコメントに明記してある（現在この例外に該当するのは `block_swap_prefetch` と `fused_gguf_dequant_kernel` の2つだけ）。

**最終検証（実機1本）**: 反転後のコードでサーバーを再起動し、**Accelerationのフィールドを一切載せない既定のペイロード**（プロンプト・解像度・フレーム数・シードのみ。§50.3と同一条件）で1本生成した。

| 項目 | 結果 |
|---|---|
| ジョブID | `9e87c5f2-4380-4bd7-a47a-70161b47845f` |
| 送信したペイロード | `prompt` / `width` / `height` / `num_frames` / `frame_rate` / `num_inference_steps` / `guidance_scale` / `seed` / `pipeline` の9項目のみ（Accelerationのフィールドは1つも送っていない） |
| 生成時間 | 200.22秒（骨格キャッシュはMISS＝サーバー再起動直後の1本目） |
| `fused_gguf_dequant_kernel_used` | ✅ **`"on"`**（＝**既定onが実機で効いていることの証明**） |
| `attention_used` | ✅ `"sdpa"`（既定のまま） |
| `keep_resident_used` / `block_swap_prefetch_used` | `"off"` / `"on"`（それぞれの既定どおり） |
| `peak_vram_reserved_mb` | 13922（`peak_vram_mb` は 10229） |
| 出力SHA-256 | ✅ `73B629D972F2A05A80C95F9F196B04810372C96280766D3478F7C3B03E76E754` |
| `metadata.json` の `request` エコー | `fused_gguf_dequant_kernel: true`（送っていないのにtrueで記録される＝サーバー既定が反映されている）。旧 `fused_gguf_dequant_gemm` は**現れない**（撤去済みの確認） |
| 判定 | ✅ **合格**（完走・echo `"on"`・SHAが基準値と完全一致） |

比較の基準値はG5のOFF側（`da673d87`、sdpa・Tritonあり）である。sdpa条件でのビット一致の基準として、同じSHA-256（`73B629D9…754`）になることを合格条件にした。参考までに生成時間は 271.72秒 → 200.22秒（どちらも骨格キャッシュMISSの1本目）で、**71.5秒＝約26%短縮**しているが、この2本は `keep_resident` の指定が異なる（前者on・後者は既定のoff）ため、速度の正式な数値は交互対比較で測ったG2の**約17.5%**を採る。

### 51.6 残課題

1. **オーナー目視ゲートのみ**。確認していただきたいのは次の4点である。
   - 設定タブのAcceleration区画に「Fused GGUF Dequantization Kernel（日本語表示では「GGUF逆量子化の1カーネル化」）」の切替が、**既定でOn（有効）** の状態で表示されること。日本語／英語の切替で文言が正しく変わること。
   - 旧項目「Fused GGUF dequant + GEMM」（常時グレーアウトされていたモック）が**消えていること**。
   - On/Offを切り替えると表示が追随し、アプリを閉じて開き直しても選んだ状態が残ること（localStorageへの保存）。
   - その状態で生成が完走し、体感で速くなっていること。
2. **コミットはオーナーの手動作業**（本リポジトリ・フロントエンド `Nz-LTX23-frontend-AviUtl2` の両方）。
