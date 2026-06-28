# 次セッション引き継ぎ書 — LTX 2.3 を 16GB で動かす (Path A: 低VRAMフォーク方式の移植)

最終更新: 2026-06-28 / 想定読者: 次セッションでコード改修を行うエージェント

> ⚠️ **方針更新（2026-06-28 後半）**: 本書のうち「Gemma の扱い」「移植の細部」は更新されました。最新の方針・比較・参照URL
> は **[DESIGN_COMPARISON_and_direction.md](DESIGN_COMPARISON_and_direction.md)**、実機検証の経緯は **[VERIFICATION_LOG.md](VERIFICATION_LOG.md)** を参照（こちらが最新の真実）。
> 要点: フォークの GGUF transformer は実証済（dequant 4件＋loader を修正、有意映像を生成）。残課題は Gemma を
> **FP4 量子化で GPU 推論**（コミュニティ標準）に載せる一点＝局所修正で 16GB 達成見込み。CPU bf16 エンコードは
> この機(i7-13700/AVX2)で地雷につき撤回。本書の env 手順・凍結API・検証手順は引き続き有効。

> このセッションの成果 = **土台づくりと実証**。コード改修（フォーク手法を我々の backend へ移植）は**このドキュメントを起点に次セッションで行う**。
> 全経緯は計画書 `~/.claude/plans/playful-napping-scone.md`、要約は memory `[[ltx-bridge-project]]` / `[[ltx-desktop-lowvram-fork]]`。

---

## 0. 背景（なぜこの方針か）— 3行
- 公式 `ltx_pipelines` 1.1.6 は本機(RTX 4070 Ti SUPER 16GB / Windows / torch 2.9.1+cu128)で **22B safetensors バルクロード中に native crash**（access violation。VRAM容量問題ではない。`--offload cpu/disk` でも同じ）。実測確定済み。
- → **低VRAMフォーク** `Kandyman-iac/LTX-Desktop-LOW-VRAM-and-EXTRAS`(Apache-2.0) の **GGUF transformer + block-swap** 方式へ pivot（ユーザー承認＝Path A）。clone 済: `vendor/LTX-Desktop-LOW-VRAM/`。
- **✅ 本セッションで実証**: フォーク方式は GGUF transformer + VAE/Gemma を **crash せずロードできる**（`PIPELINE_CREATED_OK`）＝公式 crash を回避。**最大リスクは解消済み**。

## 1. いま整っている資産
- **フォーク clone**: `vendor/LTX-Desktop-LOW-VRAM/`（gitignored）。中核は `backend/services/`。
- **フォーク env（構築済・動作確認済）**: `vendor/LTX-Desktop-LOW-VRAM/backend/.venv`
  - LTX-2 を git rev **`00dc53d3f81c405932f9f16d9c57557de411e702`** に pin（= `ModelLedger` API。我々の 1.1.6 とは非互換）。
  - torch **2.9.1+cu128 / cuda True**、`gguf`、`ltx_core@00dc53d`、`ltx_pipelines.utils.ModelLedger`、fork services 全て import 成功。
- **GGUF transformer（DL済）**: `models/ltx-2.3-gguf/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`（17.76GB, GGUF v3, `config` KV あり）。
- **safetensors（既存）**: `models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors`(VAE/audio/vocoder 源として使用。transformer は GGUF), `models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors`, `models/gemma-3-12b-it-qat/`。
- **我々の backend 側で実装済（前段）**: ÷64 解像度契約（`api/models.py`）、`services/ltx_runner.py` の facade＋`_MockBackend`(温存)＋`_RealBackend`(現状は公式 DistilledPipeline 呼び＝要置換)、`video_io.crop_mp4`、`config.model.backend`(auto/mock/real)。mock pytest **13 passed**。
- 旧 env `vendor/LTX-2/.venv`(1.1.6, crashする側) は**不要＝削除可**。

## 2. 環境セットアップ手順（将来の install スクリプトの素・検証済みコマンド）
プロジェクトルートで、環境変数は**プロジェクト内に隔離**（システムを汚さない）:
```bash
export UV_PYTHON_INSTALL_DIR="$PWD/.python"   # PowerShell: $env:UV_PYTHON_INSTALL_DIR="$PWD\.python"
export UV_CACHE_DIR="$PWD/.uv_cache"
export HF_HOME="$PWD/hf_home"

# 1) フォーク env 構築（managed python 3.12 を明示。PATH の WindowsApps python3.exe を避けるため --python 必須）
uv python install 3.12
( cd vendor/LTX-Desktop-LOW-VRAM/backend && uv sync --python 3.12 )

# 2) torch を cu128(GPU) に上書き（Windows の PyPI torch は CPU 版になるため）
( cd vendor/LTX-Desktop-LOW-VRAM/backend && uv pip install --python .venv/Scripts/python.exe \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 )

# 3) GGUF transformer 取得（QuantStack。LTX-2 community license）
./.venv/Scripts/python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('QuantStack/LTX-2.3-GGUF','LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf', local_dir='./models/ltx-2.3-gguf')"
```
（生コマンドログ: `outputs/forkenv_build_log.md`。失敗→成功の両方を記録済み。最初の `uv sync` は WindowsApps の python3.exe stub を掴んで失敗したので **`--python 3.12` 必須**。）

## 3. 生成の中核 API（移植のターゲット）と判明した gotcha
- 中核 = **`services/fast_video_pipeline/ltx_fast_video_pipeline.py` の `LTXFastVideoPipeline`**。
  - `LTXFastVideoPipeline.create(checkpoint_path, gemma_root, upsampler_path, device, block_swap_blocks_on_gpu=, gguf_transformer_path=, gguf_per_layer_quant=True, vae_spatial_tile_size=, vae_temporal_tile_size=)`
    - 内部で 公式 `DistilledPipeline` 構築 → `model_ledger` に `GGUFQuantLoaderService.install()`(per-layer dequant) ＋ `BlockSwapService.install()` ＋ VAE tiling を配線。
  - `.generate(prompt, seed, height, width, num_frames, frame_rate, images, output_path, num_steps=8)` → mp4。
  - 動く最小呼び出しの雛形 = `vendor/LTX-Desktop-LOW-VRAM/backend/_spike_gguf_min.py`（このセッションで作成。throwaway）。
- **gotcha 1 — circular import（回避策あり）**: cold で `from ltx_core.quantization import QuantizationPolicy` すると ltx_core@00dc53d の `quantization.fp8_cast ↔ loader.fuse_loras` が循環 import で落ちる。
  - **回避**: quantization を使う前に `import ltx_core.loader`（または `ltx_pipelines.distilled`）を先に import して warm する。フォーク本体は通常起動でこれらを先に import するため表面化しない。
- **gotcha 2 — VAE tile**: `vae_temporal_tile_size` は temporal overlap(=24) より大きく（≥32）。`0` で library 既定(64/24)。小さすぎると `ValueError: Overlap must be less than tile size`。
- **解像度**: `height/width` は ÷64（two-stage distilled。stage-1 が半分）。`num_frames` = 8n+1。最小実証 dims = 256x384 / 9（fork warmup と同）。

## 4. 次セッションで直すべき「フォークのバグ」（生成完走を阻む。crux=ロードは通る）
本セッションの実証で `PIPELINE_CREATED_OK`（ロードは crash 回避）まで到達後、生成(`generate`)で以下に当たった。**いずれも fork コード × ltx_core@00dc53d の不整合で、修正対象**:
1. **`apply_sd_ops` が `ltx_core.loader.sd_ops` に無い** → `gguf_quant_service.py` が期待。警告で "using raw keys" にフォールバック中（GGUF tensor key と model の key 対応がズレる懸念）。→ key remap を別手段で実装 or sd_ops 呼びを当該 rev に合わせる。
2. **`GGMLQuantizedTensor` に `_float_shape` 属性が無い** → `gguf_quant_service.py:470` の `numel()` が `math.prod(self._float_shape)` で `AttributeError`（torch Tensor subclass の custom 属性が `.to()` 等で喪失する典型）。→ subclass の属性伝播（`__tensor_flatten__`/`__torch_function__` で復元、または numel を float_shape 非依存に）を修正。
3. **⚠️ VRAM がタイト**: 上記 load 中に **peak ~15.9GB / 16GB**。block_swap 深度（`block_swap_blocks_on_gpu` を 8→小さく）、CPU 経由ロード、または GGUF を **Q3_K_M(~14.7GB)** に落とす等で要調整。実生成完走時の peak は未測。

## 5. 移植方針（我々の backend へ。凍結レイヤーは不変）
- **凍結（変更しない）**: REST API 形（`api/models.py`）、ジョブ管理、出力レイアウト `outputs/{job_id}/output.mp4`+`metadata.json`、`LTXRunner.generate(...)→GenerationOutcome` 契約。**mock backend も温存**（torch 無し app `.venv` のテスト用）。
- **`services/ltx_runner.py` の `_RealBackend` を作り替え**: 公式 `DistilledPipeline` 直叩き（=crash する現状）をやめ、**フォークの `LTXFastVideoPipeline` 相当**（GGUF per-layer + block-swap + VAE tiling）を呼ぶ。
  - フォークの `block_swap_service.py` / `gguf_quant_service.py` / `fast_video_pipeline/` / `ltx_pipeline_common.py` を `services/lowvram/` に取り込む（or vendor から import）。**上記バグ2件を修正してから**。
  - circular import 回避の warm import を runner の実バックエンド初期化に入れる。
- **重要・実行環境**: 我々の backend（FastAPI）は **フォーク env(00dc53d) で動かす**必要がある（1.1.6 では ModelLedger 無し＋crash）。フォーク env には既に fastapi/uvicorn 等あり。我々の追加依存（pyyaml/pillow/httpx/python-multipart 等）が足りなければ同 env に追加。**gradio は別プロセス**（hub 依存衝突回避。既出）。
- **config 追加**: `model.gguf_transformer_path`、`vram.block_swap_blocks_on_gpu`、`vram.gguf_per_layer_quant`、VAE tile sizes。`services/low_vram.py` のマッピング更新。
- **出力**: `LTXFastVideoPipeline.generate(output_path=...)` が mp4 まで書く（内部 `encode_video`）。`crop_output` 指定時は既存 `video_io.crop_mp4` で後段クロップ。`backend="ltx-distilled-gguf"` 等に。

## 6. 検証手順（移植後。落ちたら分析報告し、修正はユーザー承認後）
1. mock pytest（app `.venv`）が緑のまま。
2. real smoke: 384x256/9（or 17）T2V → mp4 生成・**peak VRAM<16GB**・時間 を実測。
3. phase1_default(512x320/49) → 最小I2V(画像1枚, frame_idx=0) → phase1_target(960x576/121, crop 960x540)。
4. Gradio は app `.venv` から別プロセスで API を叩いて疎通（任意）。

## 7. 参照
- 計画書（全経緯）: `~/.claude/plans/playful-napping-scone.md`
- env 生ログ: `outputs/forkenv_build_log.md` / `outputs/forkenv_build.log`
- 実証スパイク: `vendor/LTX-Desktop-LOW-VRAM/backend/_spike_gguf_min.py` / `outputs/forkenv_spike.log`
- フォーク中核: `vendor/LTX-Desktop-LOW-VRAM/backend/services/{fast_video_pipeline,gguf_quant_service,block_swap_service,ltx_pipeline_common}.py`
- 我々の凍結 API: `api/models.py` / runner: `services/ltx_runner.py`
- ユーザー方針: plan mode で段階承認 / **テストが落ちたら勝手に直さず原因分析して報告** / 編集・テスト・生成は**サブエージェントに委譲**し本体は監督 / 環境隔離厳守 / ÷64 契約。
