# 次セッション引き継ぎ書 — LTX 2.3 を 16GB で動かす

最終更新: 2026-06-28（Phase 5(A) 配線・実機検証 完了後に改訂）/ 想定読者: 次セッションのエージェント

> **このドキュメントが現時点の最新・正本（handoff）です。** 設計判断の根拠と実機検証の全経緯は
> [VERIFICATION_LOG.md](VERIFICATION_LOG.md)（最新 §6＝Phase 5(A) 配線・実機検証、§5＝Gemma GGUF マイルストーン）が一次情報。
> 実装計画は `~/.claude/plans/a-witty-kazoo.md`。要約は memory `[[ltx-bridge-project]]` / `[[ltx-desktop-lowvram-fork]]`。
> ⚠️ 古い記述が残るドキュメント（後述 §6）に惑わされないこと。

---

## 0. いまどこにいるか（3行）
- **目標＝LTX-2.3 を 16GB VRAM(RTX 4070 Ti SUPER/Win/torch2.9.1+cu128)で動かし FastAPI で公開**。Phase 1＝T2V＋最小I2V。
- **✅ Phase 5(A) 配線完了・実機検証済**: 凍結 REST API が実エンジン（実証済 GGUF 低VRAMパイプライン）で実際に生成するようになった。`services/ltx_runner.py` の `_RealBackend` を **Approach W（常駐サブプロセスワーカー）** で実体化し、384x256/9 の T2V／最小I2V を実 `LTXRunner.generate` 経路で完走。詳細 §3。
- **⚠️ ただし "16GB に収まった" わけではない**: Phase 4 スパイクは E2E 完走し **Gemma** の共有溢れ(17.7GB→~0)は解決したが、**denoise フェーズは依然 shared メモリへ溢れる**（384x256 でもユーザーがタスクマネージャで目視確認）。**真の 16GB fit はまだ未達**。これが次の主タスク＝Phase 5(B)（§3 末尾・§4・VERIFICATION_LOG §6）。実装計画 `~/.claude/plans/a-witty-kazoo.md`。

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

## 3c. ★次＝Phase 5(B)：真の 16GB fit（denoise の shared 溢れ解消）＝1280×768 スケールアップと同一作業
Phase 5(A) で「配線」は終わったが、**"16GB に本当に収まる" はまだ未達**。denoise フェーズが（384x256 でも）shared メモリへ溢れている（ユーザーがタスクマネージャで目視確認、VERIFICATION_LOG §6/§5 の "denoise shared ~3.6GB @bs8" と整合）。WDDM が溢れ分を system RAM へページングするため**ハード OOM はしないが激遅化**する。これが残った主タスク。

- **ユーザー方針＝これは「Phase 5(B) スケールアップ（1280×768 → crop 720p）」と一つの同じ仕事**。スケールアップ用の denoise-stage VRAM 技法（block_swap 深度を bs4/bs2 へ／VAE・attention タイリング／解像度依存の sequential・streaming）を適用すれば、**現状の 384x256 の shared 溢れも一緒に解消**する見込み。
- **着手の起点＝先行事例の denoise-stage 技法を調査・複製**（独自発明しない、[[research-prior-art-first]]）: フォークの `vendor/.../backend/services/block_swap_service.py` 等、および ComfyUI-GGUF / ComfyUI カスタムノード・ワークフローの解像度対応技法。これらは**まだ我々の backend に反映していない**。
- 計測は perf-counter の "Shared Usage" サンプリングで（§5。worker の `max_memory_allocated` では溢れを検知できない＝§4）。

## 4. 既知の残課題（Phase 5(B) の中核 ＋ 後で）
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
