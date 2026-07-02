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
- **全体ジョブ `peak_vram_mb` は両モードとも 16,944 で不変**＝**16GB の天井は denoise / transformer-load 段（peak ②）で決まり、`--te-offload` はそこに触れない**。te-offload が下げるのは **Gemma text-encode ピーク（peak ①）** のみで、encode 時の shared 溢れを消すだけ。**peak ② の低減は将来課題**（§7.9 の「transformer を直接 CPU ロード」と地続き）。2つのピークは逐次で、その max がジョブ天井を決めるという前セッションの所見（§7.9）どおり。

### 11.6 チューニングレバー
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
