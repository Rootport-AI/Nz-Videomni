# 開発ノート — LTX 2.3 を 16GB VRAM で動かす

> ⚠️ **状態（2026-06-28 後日追記）**: 本書の中心戦略「公式 `ltx_pipelines` の block streaming(`--offload cpu`)＋`fp8-cast`」は
> **破棄**された（本機 16GB Windows で 22B safetensors バルクロード中に native crash）。現行は**フォークの GGUF transformer
> ＋block-swap＋GGUF Q4 Gemma**方式で 16GB E2E 達成済み。最新の正本は [NEXT_SESSION_HANDOFF.md](NEXT_SESSION_HANDOFF.md) /
> [VERIFICATION_LOG.md](VERIFICATION_LOG.md)。**ただし以下の事実は今も有効**: torch 2.9.1+cu128、attention は SDPA(=Ada では
> FlashAttention-2)で十分、xformers は任意、16GB の律速は重み転送(PCIe)。

最終更新: 2026-06-26

本機: RTX 4070 Ti SUPER (16GB, Ada Lovelace / sm_89) / System RAM 63.8GB / PCIe 4.0 ×16

---

## 結論サマリー

- **attention は SDPA で進める。** Ada + torch 2.9 の SDPA は実体が **FlashAttention-2** カーネル。
- **xformers は "さらなる速さが欲しくなったときの Optional な工夫" と位置づける。** 今はビルドしない。
- 16GB で 22B を動かす本命は **block streaming (`--offload cpu`) + `fp8-cast`**。
- 速度の律速は **offload streaming の PCIe 転送**であり、attention バックエンドではない。

---

## torch / CUDA / xformers の事実確認（前回前提の訂正）

| 論点 | 前回の前提（誤） | 一次情報による事実 | 根拠 |
|---|---|---|---|
| torch | 2.7 固定 | **2.9.1**（`~=2.7` は `>=2.7,<3.0` の範囲指定。公式 lock も 2.9.1 に解決） | `vendor/LTX-2` の `ltx-core/pyproject.toml`, `uv.lock` |
| CUDA ビルド | cu129 | **cu128**（公式 README が `torch 2.9.1+cu128` を検証済みと明記。cu129 に 2.9.1 の Windows wheel は存在しない） | `vendor/LTX-2/README.md` L80 |
| xformers | 必須（→ソースビルド） | **任意の最適化**（`uv sync --extra xformers`。SDPA で動く） | README "Optimization Tips", `ltx-core` optional-deps |

- Windows の PyPI 版 torch は **CPU 専用**のため、`uv sync --frozen` だけでは `2.9.1+cpu` が入る。
  → cu128 インデックスから `torch / torchaudio / torchvision == 2.9.1+cu128 / 0.24.1+cu128` を上書きインストールする。
- xformers の lock pin: `0.0.33+5d4b92a5.d20251029`（commit `5d4b92a5`、torch 2.9.1 依存、wheel は **Linux 専用**）。Windows で使う場合のみソースビルドが必要。

---

## 16GB 実現の仕組み（公式コードで確認）

- `ltx_pipelines` の全パイプライン（distilled 含む）が `--offload {none,cpu,disk}` を持つ。
- `offload != NONE` のとき、**22B transformer も Gemma 12B エンコーダも `StreamingModelBuilder` で1層ずつ GPU へストリーム**（常駐は 2 層のみ。`_DEFAULT_GPU_SLOTS = 2`）。
  - `blocks.py` `PromptEncoder`（L496–522）: Gemma 用 streaming builder（`blocks_attr="model.model.language_model.layers"`）。
  - `blocks.py`（L235–243）: **block streaming は `bf16` と `fp8_cast` に対応**。`fp8-scaled-mm`（TensorRT-LLM）のみ非対応。
    → **`--offload cpu` + `--quantization fp8-cast` は両立可能**。
- メモリはステージング: Gemma でエンコード → 解放 → transformer（ストリーム）→ 解放 → VAE decode（タイル）。

---

## フェルミ推定: SDPA vs xformers の速度差はどのオーダーか

### 前提
- 22B distilled, fp8-cast（重み ≈ 1 byte/param ≈ **22GB**）, `--offload cpu`。
- DistilledPipeline: stage1 約8 step + stage2 約4 step ≈ **12 フォワードパス**（SimpleDenoiser, B=1）。
- PCIe 4.0 ×16 実効スループット ≈ **24 GB/s**。

### 支配項 = 重みの PCIe 転送
- 1 パス = 22GB / 24GB/s ≈ **0.9 秒**。
- 12 パス → transformer 転送だけで **約 11 秒**。
- + Gemma エンコード（1回, 約1–2秒）+ VAE/upsampler decode（タイル, 数–十数秒）+ overhead
  → **総生成時間 ≈ 30〜90 秒/短尺クリップ**（オーダー）。

### attention 計算の占める分（= xformers が効く範囲）
- 12 パスの GPU 計算合計 ≈ 1〜6 秒、うち attention は 2〜4 割 ≈ **0.5〜2.5 秒**。
- xformers vs SDPA の差は ±15% 程度 → **全体で ±0.1〜0.4 秒（数%）**。

### 答え
- **総生成時間のオーダー: 数十秒〜数分。** 主因は offload streaming の PCIe 壁（16GB では不可避、xformers/SDPA で同一）。
- **xformers vs SDPA の差: サブ秒〜数秒（全体の数%）。「秒→分」にはならない。**
- xformers は重み転送も FFN/linear 層も速くしない（attention の行列積のみ）ので、offload 構成ではボトルネックに効かない。
- **fp8-cast は転送量を半減**（bf16 なら 44GB/パス ≈ 22秒）させるため、xformers より fp8-cast の方が速度に効く。
- 長尺・高解像（960x544/121）では attention 比率が上がるが、その領域こそ FlashAttention(=SDPA) の本領。やはり xformers ≈ SDPA。

### 根拠となった事実
- LTX-core `attention.py`: `PytorchAttention` は `sdpa_kernel(priority)` 経由で SDPA。Ada 既定優先順位は FLASH 先頭 → 実体は FlashAttention-2。
- PyTorch SDPA の 2 カーネル = `sdpa_flash`(FlashAttention-2) と `sdpa_mem_eff`(xformers の memory-efficient カーネル内蔵)。
- 実測: 「xformers ≈ pytorch_sdp」「FlashAttention-2 は xformers cutlass の約2倍速」ケースあり。「xformers 必須」は PyTorch 2.0 以前の常識。
- offload 実測: 1層転送 240ms vs 計算 0.1ms（RTX 4090, PCIe4.0×16）。Flux で CPU offload により 6.7s→21.5s。

---

## 進め方（SDPA 先行）

1. LTX venv の torch を **cu128 版に差し替え**（`2.9.1+cu128` 一式）→ `torch.cuda.is_available()` 確認。
2. モデル DL: distilled 46GB + spatial upsampler 1GB + Gemma 25GB（≈72GB / 空き 147GB）。
3. **公式 CLI で 16GB 実機検証**: `python -m ltx_pipelines.distilled --offload cpu --quantization fp8-cast ...`
   → peak VRAM・生成時間を実測。
4. `services/ltx_runner.py` を `DistilledPipeline(offload_mode=CPU, quantization=fp8_cast)` 呼び出しへ。
5. **実測で attention が律速かつ差が大きい場合のみ** xformers ソースビルドを検討（commit `5d4b92a5`, cu128 torch 向け, 30–90分）。

### xformers を将来ビルドする場合のメモ
- 前提ツールは導入済み: CUDA Toolkit 12.9 (nvcc V12.9.86), VS 2026 + MSVC v143 (v14.44)。
- torch は cu128(CUDA 12.8 ランタイム)、ビルドは CUDA 12.9 toolkit → minor 差の警告のみで通る想定。
- `scripts/build_xformers.ps1` の既定値（`$XformersRef=5d4b92a5` / `$CudaVersion=12.9` / `$VcvarsVer=14.44` / `$Arch=8.9`）は妥当。
