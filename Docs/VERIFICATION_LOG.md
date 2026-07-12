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

- 音声継ぎ目のクリック根治（源の実音声とデコード音声のノイズフロア差）: v1 は 30ms フェードで緩和・G3 試聴の結果次第で「サーバー側結合出力＋真のクロスフェード」オプションを検討。
- stage2 の音声タイル継ぎ目（既知・チェーン由来 backlog と同族・V2V 固有ではない）。
- fps リサンプルが全体変換（末尾だけの部分変換に最適化可能・単一ユーザーでは実害小）。VFR 源は未ストレステスト。
- mock の音声数値は 0 固定（実バックエンドと差異あり・docstring 記載済み）。

### 24.6 得られた知見

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

git commit／push は本節時点でも未実施（作業ツリーの変更のまま・ユーザー判断待ち）。次セッションの筆頭課題はコミット／プッシュ（オーナー判断）。

---

## 34. ★reference動画付きIC-LoRAのchain対応＝α版（clips=1限定）で解禁＝客観ゲート全PASS・独立レビュー must-fix ゼロ・mock E2E PASS・**GPU実機目視ゲート✅完了・ユーザー受容（2026-07-12）**（2026-07-11）

> **正本＝本節。** NEXT_SESSION_HANDOFF.md 冒頭ブロックの引き継ぎ課題「reference 付き IC-LoRA のチェーン対応」（§32.1 の2で「チェーン非対応・422で拒否」としていたスコープを、オーナー決定によりα版として昇格）を実装した回。**`clips` がちょうど1つのチェーン（A2V を含む）に限り**、reference動画付き control 系 IC-LoRA（参照動画から輪郭線 canny・骨格 pose 等を読み取って条件付けするアダプタ）を許可する。`clips` が2つ以上のチェーンは従来どおり 422（`LORA_CONTROL_UNSUPPORTED_IN_CHAIN`）で拒否を維持。**客観ゲート（pytest・独立レビュー・mock E2E）はすべて PASS。GPU 実機の目視ゲートは2026-07-12にオーナー立ち会いのもと実施し完了・受容済み（詳細＝34.7）。コミット未実施（作業ツリーの変更のまま・push はユーザー判断待ち）。**

本機: 同上（i7-13700／RTX 4070 Ti SUPER 16GB／System RAM 64GB／Windows 11／`LTX_KEEP_RESIDENT=0`）。

### 34.1 要件確定（オーナー決定・§32.9〔NEXT_SESSION_HANDOFF.md 99-108行〕の課題を実装）

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
3. `GET /jobs/{id}` の応答の `request` ブロックは chain 固有フィールド（`source_audio`／`loras`／`reference_video_id` 等）を載せない（`JobResponse.request` が `to_clip_request` 経由の `GenerateRequest` 形のため）。§30 の `loras` 追加時からの既存の表現上の制約であり、今回の退行ではない。正式な記録は `metadata.json` 側。
4. **既知の無害事象（GPU実機目視ゲート中に観測・2026-07-12）**: 2本目のジョブ完了直後にサーバーログへ `ERROR asyncio: ... ConnectionResetError [WinError 10054]`（`_ProactorBasePipeTransport._call_connection_lost`）が1回出力された。これは Windows の asyncio proactor がクライアント（ブラウザ）側の強制切断を後処理する際の既知の無害なノイズであり、直後に再接続し以降のジョブも正常完走した。機能影響なし・対応不要（オーナー判断で無視と決定）。

### 34.7 GPU実機目視ゲート＝✅完了・ユーザー受容（2026-07-12）

客観ゲート（34.3〜34.5）に続き、オーナー立ち会いのもと GPU 実機で目視ゲートを実施した。GUI の Generate タブから音声 wav＋参照動画＋control 系アダプタ（reference動画付き IC-LoRA）で生成を実行し、成功を確認した。

**実績**: 解像度 1280×768・201 frames・8 steps・所要時間 約300秒/本・ピーク VRAM 9241〜9537MB。内訳＝pose-control（strength=1）×3本＋canny-control（strength=1）×1本、いずれも `completed`。ジョブ開始ログに `loras=pose-control(strength=1)` 等の配線物証を確認（本節「ジョブ起動ログの拡充」機能がここでも有効に機能した）。

オーナーが実際の生成結果を受容＝**A2V＋reference付きIC-LoRA併用のGPU実機目視ゲートはクローズ**。

観測された既知の無害事象は34.6の4に追記済み（asyncio ConnectionResetError・機能影響なし・対応不要）。

git commit／push は本節時点でも未実施（作業ツリーの変更のまま・ユーザー判断待ち）。残る筆頭課題はコミット／プッシュ（オーナー判断）のみ。

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

**§35クローズ（オーナー実機確認完了・2026-07-12）。** git commit／push は本節時点でも未実施（作業ツリーの変更のまま・ユーザー判断待ち）。
