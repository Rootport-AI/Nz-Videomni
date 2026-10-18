# R3 棚卸し: MiniMax H3（3 つ目のエンジン系統）を足すときに触る場所・新設する場所・再利用できる部品

調査日 2026-10-17。読むだけ（編集なし・GPU 不使用）。パスはリポジトリ `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni` からの相対。行番号は調査時点のもの。

---

## (1) 結論の表

### 第 1 弾（ワーカーのセルフテストで 1 本生成）

| 区分 | 対象 | 根拠・注記 |
|---|---|---|
| **新設** | H3 用ワーカーのパッケージ（`engine25/` と同格の兄弟。worker.py＋パイプライン＋モデル定義） | `engine25/worker.py` の作法（stdin/stdout の JSON 行・`@@LTX@@`・`--selftest-generate`）を写す形が前例 |
| **新設** | H3 用の仮想環境と freeze ファイル（＋手動の再現手順書にあたる `*-venv-pyproject.toml`） | 前例 `engine25/venv-engine-ltx25.freeze.txt`・`engine25/engine25-venv-pyproject.toml`。第 1 弾は手作業で作ってよいかは監督判断（インストーラは後の弾） |
| **新設** | mp4＋音声の書き出し部品 | LTX 2.5 は `ltx_pipelines.utils.media_io.encode_video`（PyAV・libx264 crf 19・yuv420p）に依存しており H3 の venv では使えない（項目 11） |
| **新設** | テキストエンコーダ（Qwen3-VL 32B を 50 層に切った int8 ConvRot）・映像 VAE（デコーダの一部が int8 ConvRot）・音声 VAE の読み込み | 既存の Gemma 系の読み込みは ltx_core 前提で使えない |
| **再利用（そのまま import）** | `sft_quant_format.py`（`read_header`・`layer_schemes`・`parse_metadata`・`SCHEME_TABLE`） | 標準ライブラリだけ。`layer_schemes(path, header, "")` は接頭辞なしの H3 ファイルにそのまま使える見込み（`inspect` は不可。項目 4） |
| **再利用（そのまま import）** | `engine/sft_quant/dequant.py`（`dequantize`・`normalize_aux`・`hadamard`）・`engine/sft_quant/sft_reader.py`（`read_tensors`） | torch＋`sft_quant_format` だけ。int8 ConvRot は groupsize 256 のみ（枝刈り版は全層 256 なので通る） |
| **再利用（関数単位で import）** | `engine/sft_quant/quant_service.py` の `_quant_linear_forward`・`_patch_model_for_quant`・`_assert_quant_only_in_linears`・`load_connector_bf16` | モジュールの最上位は torch＋`engine.gguf.ic_lora_common`（torch のみ）。`SftQuantStateDictLoader.load`・`sft_transformer_sd_ops`・`_make_quant_module_ops`・`SftQuantLoaderService.install` は関数内で ltx_core を import するので H3 では使えない |
| **再利用（そのまま import）** | `engine/transformer/block_swap_service.py`（`BlockSwapService`）・`engine/transformer/block_swap_prefetch.py` | 最上位の依存は torch と `engine.gguf.quant_service`（→ torch のみ。triton・gguf・numpy・ltx_core は関数内の遅延 import）。H3 の本体ブロックは `blocks` という名前で、`_get_blocks` の候補名に含まれる |
| **再利用（そのまま import）** | `engine/gemma/layer_offload_service.py`（`GemmaLayerOffloadService`） | torch のみ。`.layers` を持つ ModuleList を探す汎用実装。LTX 2.5 の `Ltx25GemmaLayerOffloader` は ltx_core 依存のモジュール内にあり import できない |
| **再利用（任意）** | `engine/transformer/sage_attention_service.py` の `probe_sage` | 最上位は torch のみ。sageattention は遅延 import。H3 の注意機構への差し込み（`_attention_modules`）は ltx_core 依存 |
| **触らない** | API（`api/*`）・アダプタ（`services/engines/*`）・記述子（`scripts/manifests/*`）・フロントエンド・インストーラ・`config.py` | 第 1 弾の裁定どおり。ただし「記述子だけ先に置く」と起動や `GET /models` が壊れる（項目 5・9）ので、記述子は後の弾でアダプタと同時に入れる必要がある |

### 後の弾で触るもの

| 区域 | 触る場所 | 規模の見込み |
|---|---|---|
| 系統の登録 | `services/engines/__init__.py` の `FAMILY_BY_ID`（50 行）・`FAMILY_DISPLAY`（58 行）。`FAMILY_BY_KV`（43 行）は H3 の疑似 KV の扱い次第 | 小 |
| ヘッダの判定 | `services/model_registry.py` の `precheck_model_file`（118 行）→ `_precheck_sft_transformer`（200 行）は系統を知らずに `sft_quant_format.inspect` を呼ぶ。H3 のファイルはここで 422 になる（項目 4） | 中（系統別の分岐を足す設計判断が要る） |
| 記述子 | `scripts/manifests/` に新ファイル（例 `40-…json`） | 小。ただしテストの連鎖あり（項目 9） |
| アダプタ | `services/engines/<h3>/adapter.py` 新設 | 中（最小構成の見積もりは項目 6） |
| 設定 | `config.py` の `ModelConfig` に `engine_python_<h3>`（`engine_python_ltx25` は 122 行） | 小 |
| API の制約 | `api/models.py` の 8n+1・64 の倍数・fps の検証は系統を知らない pydantic 検証（項目 7） | 大（エンジン別制約の仕組みが要る。設計書 §7 で「スコープ外」とした領域） |
| 快適上限マーカー非表示 | フロントエンド `shell/comfortTable.ts`・`modes/single/useGenerationForm.ts`・`modes/chained/useChainForm.ts`、Gradio `gradio_ui/comfort.py` | 記述子から `comfort` を省くだけでは**非表示にならない**（項目 5） |
| 操作パネル | `featureScope.ts`・`mockBridge.ts`・フレーム数スナップ（37 ファイルに 8n+1 の記述） | 大 |
| インストーラ | `scripts/install_ltx.ps1`（`Ensure-EngineVenv` の 3 回目の呼び出し＋検証表の行）・`scripts/install_model.ps1`・`install-MiniMaxH3.bat` | 中（UETrack が前例） |
| MCP・Gradio | `mcp_server/params.py`・`mcp_server/tools/generate.py`・`mcp_server/batch_planning.py`・`mcp_server/server.py`・`gradio_ui/handlers.py`・`gradio_ui/i18n.py` に 8n+1 の記述 | 中 |
| 文書 | `Docs/MULTI_ENGINE_DESIGN.md` §5.3 の「全エンジンで torch のバージョンを揃える【オーナー裁定】」は H3 の裁定 11（torch を揃えない）と食い違う。§9.1 の「mock は ltx アダプタ内」も H3 で再判断対象 | 小 |

---

## (2) 項目 1〜11 の事実

### 項目 1. ワーカーの作法（`engine25/worker.py`・2448 行）

- **起動**: アプリ側が `[engine_python, "-u", "-m", "engine25.worker"]` を cwd＝リポジトリ直下で起動（`services/engines/ltx/adapter.py:2062` 付近の `subprocess.Popen`）。環境変数は `_RealBackend25._build_child_env`（`services/engines/ltx25/adapter.py:629`）で `TORCH_COMPILE_DISABLE=1`・`PYTHONUNBUFFERED=1`・`PYTHONPATH=<root>` だけ。ワーカー自身も `ROOT=parents[1]` を `sys.path.insert(0)`＋`os.chdir(ROOT)`（`engine25/worker.py:220-222`）、`TORCH_COMPILE_DISABLE` を torch import 前に設定（同 212）。torch はモジュール最上位で import しない（`load` 内で遅延 import）。
- **プロトコル**: 親→子は 1 行 1 JSON の `op`。`load`／`generate`／`generate_chain`／`shutdown`（`main` は 1620 行）。load 前の未知 op は無視（ログのみ）、load 後の未知 op は `error` イベントを返して生存（1678 行）。load の失敗は `error`＋exit 1、generate の失敗は `error` を返して生存。
- **子→親**: 標準出力の各行は `@@LTX@@` ＋ 区切り詰めの JSON（`PREFIX` 227 行・`_emit` 256 行）。`ready`（`sampler`・`sage_available`）・`progress`（`stage`・`index`・`total`、連結時のみ `outer_index`・`outer_total`。`_emit_progress` 293 行）・`done`・`error`（`detail`＝repr＋トレースバックの先頭 1200 字＋末尾 1600 字。`_detail` 262 行）。進捗の段階名は 2.3 と同じ語彙（`encode`／`stage1_denoise`／`stage2_denoise`／`decode`）で、親側の割合計算（`_progress_frac`・`_read_worker_events` 2695 行。単段用の `one_stage` 引数あり）がこの名前に依存する。
- **ログ**: 人向けは標準エラーのみ（`[ltx25_worker]` 接頭辞。`_configure_worker_logging` 230 行・`_log` 251 行）。親が標準エラーを `logs/ltx25_worker.log` に追記でつなぐ（`_LOG_NAME`。ltx25 adapter 617 行）。
- **load の流れ**（`_do_load` 329 行）: パスの存在確認（`_LOAD_PATH_FIELDS` 279 行）→ `probe_sage` → パイプラインを組む → `LOAD_OK` をログ → `ready`。
- **`done` の中身**（1117-1150 行付近）: `seed_used`・`sampler`・`peak_vram_mb`・`peak_vram_reserved_mb`（MiB）・`rss_peak_gib`・`seconds`・`num_frames`・`encode_fps`・`size_bytes`・高速化のこだま（`block_swap_prefetch_used`・`fused_gguf_dequant_kernel_used`・`keep_resident_used` ほか）・`vae_mode_used`・`ltx25`（`encode_fps`/`video_chunks`/`tiling`/`size_bytes`/`phases`）。
- **mp4 と音声**: ワーカーが `output_path` に直接書く。LTX 2.5 は `engine25/pipeline25.py:1707` の `encode_video`（`ltx_pipelines.utils.media_io.encode`：PyAV `av.open`・`libx264`・crf 19・yuv420p・音声は `ltx_core.color.audio_mux`）。ワーカーは ffmpeg を呼ばない（クロップは app 側の ffmpeg 後処理）。
- **セルフテスト**: `python -m engine25.worker --selftest-generate --transformer … --text-encoder … --video-vae … --audio-vae … --spatial-upsampler … --output out.mp4 [--width 320 --height 192 --num-frames 25 --frame-rate 24 --seed 42 --rounds N --report path …]`（`_add_load_arguments` 1690・`_load_message` 1714・`_selftest_generate` 1965・入口 2442）。`_do_load`／`_do_generate` を合成メッセージで直接呼び、`_emit` を差し替えて全イベントを記録、mp4 の sha256 と `done` をそのまま JSON レポートに入れる。`--rounds 2` で同一シード再現性を比較。
- **`engine25/` が `engine/` から import している共有モジュール**: `engine.gguf.quant_service`・`engine.gguf.ic_lora_common`・`engine.gguf.dequant_triton`・`engine.sft_quant.quant_service`・`engine.transformer.block_swap_service`・`engine.transformer.sage_attention_service`・`engine.transformer.nag_service`・`engine.transformer.vsf_service`・`engine.transformer.nag_selfcheck`・`engine.transformer.sage_selfcheck`・`engine.inpaint.canvas`・`engine.outpaint.canvas`・`engine.outpaint.pyramid_blend`・`engine.preprocess`、リポジトリ直下の `sft_quant_format`・`chain_math`。`engine/__init__.py`・`engine/gguf/__init__.py`・`engine/transformer/__init__.py`・`engine/sft_quant/__init__.py` は import を持たない（副作用なし）。

**H3 の venv が共有部品を import するときに必須になるパッケージ**

| モジュール | モジュール最上位で必須 | 関数内で遅延 import されるもの（使う関数次第） |
|---|---|---|
| `sft_quant_format`・`chain_math` | なし（標準ライブラリ） | — |
| `engine.sft_quant.dequant`・`engine.sft_quant.sft_reader` | torch | — |
| `engine.sft_quant.quant_service` | torch（＋`engine.gguf.ic_lora_common`＝torch） | ltx_core（`SftQuantStateDictLoader.load` 116・`sft_transformer_sd_ops` 197・`_make_quant_module_ops` 350・`install` 408） |
| `engine.transformer.block_swap_service` | torch | `block_swap_prefetch`（先読み要求時のみ） |
| `engine.transformer.block_swap_prefetch` | torch（＋`engine.gguf.quant_service`→`dequant_triton`・`ic_lora_common`＝torch） | triton（`dequant_triton_kernels` は失敗時に縮退）・gguf・numpy・ltx_core（いずれも gguf の読み込み関数内） |
| `engine.gemma.layer_offload_service` | torch | — |
| `engine.transformer.sage_attention_service` | torch | sageattention（`_sage_callable` 118）・ltx_core（`_attention_modules` 455） |
| `engine.inpaint.canvas`・`engine.outpaint.canvas` | なし／torch は関数内 | — |
| `engine.preprocess` | 各前処理器（cv2 など。今回未精査） | — |

結論として、block swap・先読み・int8 ConvRot の逆量子化・Linear の差し替え・層オフロードは **torch だけで import できる**。safetensors パッケージも不要（`sft_reader` は seek＋readinto の自前読み）。ただし前提として H3 の venv に `engine/` 側の文法（Python 3.12 前提の記法）と torch の API（`torch.cuda.Stream`・`pin_memory`・`float8` dtype の参照）が通ることが要る。`engine/sft_quant/quant_service.py:54` は `torch.float8_e4m3fn` を最上位で参照するので、torch の版によらず存在するが、版を変えるなら確認対象。

### 項目 2. block swap と先読み

- **`BlockSwapService(blocks_on_gpu, device, hold_arenas=False)`**（`engine/transformer/block_swap_service.py:49`）。`install(transformer)`（97 行）は冪等。`blocks_on_gpu==0` で無効、`>= 総数` で何もしない。全ブロックを `block.to("cpu")` してから、各ブロックの `forward` をラッパーに差し替える（フックではなく `block.forward` の属性差し替え。元の forward は属性 `_BLOCK_SWAP_ATTR` に退避）。
- **ブロック列の探し方**（`_get_blocks` 195 行）: `transformer` と `transformer.velocity_model` の順に、属性名 `transformer_blocks`／`blocks`／`layers`／`model_blocks` の最初に見つかったもの。無ければ子の中で要素数 4 超の `nn.ModuleList`。H3 は最上位に `blocks.0..49`（＋`token_refiner.blocks.0..1`）なので `blocks` で本体 50 ブロックが取れる見込み。
- **前向きの呼び方**: ラッパーは引数と kwargs のテンソルを再帰的に GPU へ移してから元の forward を呼ぶ（`_move_to_device`）。各ブロックは「直前のブロックの出力を次のブロックへ渡す」逐次呼び出しを前提にし、窓は `[idx, idx+blocks_on_gpu)`。
- **同期経路**（`_patch_block` 221 行）: `blk.to(device)`・取りこぼし対策の `p.data = p.data.to(device)`（264 行）・追い出しは `prev.to("cpu")`。パラメータの device で判定（`blk.parameters()`）。**バッファだけのブロックは判定に掛からない**（2.3 の Gemma 層オフロードが別実装になった理由。`engine/gemma/layer_offload_service.py` 冒頭の説明）。
- **先読み経路**（`_build_prefetch_engine` 292 行・`PrefetchEngine` は `block_swap_prefetch.py:302`）: 準備の失敗は例外を捕まえて同期経路へ縮退（こだまは `on->off`）。
  - ページ固定プール `PinnedStagingPool`（234 行）＝`torch.empty(nbytes, dtype=uint8, pin_memory=True)`（173 行）の 2 枠（`_NUM_STAGING_SLOTS=2`）。最大ブロックの大きさで確保。
  - 転送は専用の `torch.cuda.Stream` 上で `arena.copy_(pinned[...], non_blocking=True)`（600 行）。ホスト側は `pinned[off:off+n].copy_(src_u8)`（580 行）。同期は `Event.record`／`wait_event`／`synchronize`。**`record_stream` は使わない**（説明 32-40 行）。
  - アリーナの輪（`hold_arenas=True`、LTX 2.5 だけが使う。`engine25/gguf_transformer.py:642`）は `blocks_on_gpu+2` 個の uint8 の GPU バッファ。
  - 1 ブロックの全テンソルを 1 本の uint8 アリーナに 512 バイト境界で詰める（`_plan` 419 行）。列挙は各モジュールの `_parameters`／`_buffers` を直接たどる（`_enumerate_slots` 282 行）。
  - GPU 側の再構築は **`raw.view(dtype).view(shape)` の素のテンソル**（`_build_views` 642 行）。取り付けは パラメータなら `mod._parameters[name].data = dev_t`、バッファなら `mod._buffers[name] = dev_t`（`_install_views` 646-651 行）。解放は CPU 側の正本へ同じ方法で付け替えるだけで転送なし（`_release` 653 行）。
- **機構が前提にしている性質**（テンソル サブクラスが乗るかの判断材料）:
  1. 各テンソルが CPU 上にあり（`_plan` 434 行で検査）、`tensor.data`（パラメータ）または本体（バッファ）に対して `.is_contiguous()`／`.contiguous()`・`.numel()`・`.element_size()`・`.reshape(-1).view(torch.uint8)`・`.data_ptr()` が素直に効くこと。
  2. 型の分岐は **`GGMLQuantizedTensor` だけ**（`isinstance` 438・493 行。`block_swap_prefetch.py:133` で import）。それ以外は「素のテンソル（dtype と形）」として扱われ、GPU 側では素のテンソルとして作り直される。つまり**それ以外のサブクラスは往復でクラスが失われる**（`_validate` は素のテンソルなら dtype と形しか見ない）。`GGMLQuantizedTensor` は `torch.Tensor._make_subclass` で作る非ラッパー型サブクラス（`engine/gguf/quant_service.py:299-317`）で、専用の 3 引数コンストラクタで復元している。
  3. パラメータの差し替えは `.data =`（Parameter オブジェクトの同一性を保つ）。`set_data` は型の互換性を要求する（`_release` 660-669 行の注記に「incompatible tensor type」で落ちる実例）。
  4. 同期経路は `Module.to(device)` と `p.data = p.data.to(device)`。
- **既存の量子化 safetensors の乗せ方**（`engine/sft_quant/quant_service.py` 冒頭 25-31 行・`_patch_model_for_quant` 296 行）: 量子化層の `weight` を **保存形のまま（int8 の素のテンソル）`requires_grad=False` の `nn.Parameter`** にし、`weight_scale` などは**永続バッファ**として登録、`load_state_dict(strict=False, assign=True)` で埋める。Linear の forward を `_quant_linear_forward`（274 行）へ差し替え、呼ぶたびに `dequantize(scheme, w, self._buffers, x.dtype)` で bf16 に戻して `F.linear`。量子化の情報はテンソルではなく **モジュールの属性 `_sft_scheme`** に置く。だから全部が素のテンソルのままで block swap／先読み／常駐キャッシュに乗る。
- **`engine25/gguf_transformer.py` の配線**: `Ltx25DiffusionStage.__init__` で `BlockSwapService(blocks_on_gpu, device, hold_arenas=True)`（632-644 行）。`_place_transformer`（1070 行）がブロック以外を `_move_module_tree`（570 行。パラメータは新しい `nn.Parameter` に作り直し、バッファは `.to()`）で GPU へ置き、ブロックは CPU に残して `_unpatch_block_swap`→`ensure_block_swap_installed`（1113 行）。ブロック列は `transformer_blocks_of`（559 行。`velocity_model.transformer_blocks`）。先読みの要求は `set_block_swap_prefetch`（788 行）、片付けは `teardown_block_swap_prefetch`（828 行。ジョブの finally）。GPU に置く数の既定は `DEFAULT_BLOCKS_ON_GPU = 8`（ltx25 adapter 120 行、ペイロードの `blocks_on_gpu`）。
- **H3 のブロックの大きさ（ヘッダの実測値からの計算）**: 量子化 4 本の int8 本体で約 0.385GB／ブロック（19.27GB÷50）。先読みのページ固定プールは 2 枠で約 0.77GB、アリーナの輪は `(blocks_on_gpu+2)×0.39GB`。`_quant_linear_forward` の逆量子化は最大層（fc1 [28672,5376]）で float32 の中間が約 0.62GB×2 前後と bf16 の 0.31GB（`dequant.py` の説明「float32 2 個分」による推計。実測ではない）。

### 項目 3. テキストエンコーダの層オフロードの前例

- **LTX 2.5（`te_layers_on_gpu`）**: `engine25/gguf_gemma4.py:531` の `Ltx25GemmaLayerOffloader`。層以外（埋め込み表など）を GPU へ、`lm_head` は CPU のまま、先頭 `layers_on_gpu` 層を常駐、残りは forward を包んで「入れる→実行→出す」（`_streamed` 625 行。`_move_module_tree` で毎回 CPU⇄GPU、非同期なし）。冪等の印は属性。既定は 0（1 層ずつ）。**モジュールが最上位で ltx_core（`ltxcore_compat`）を import するので H3 からは import できない**。
- **LTX 2.3（`--te-offload`）**: アプリが環境変数 `LTX_TE_OFFLOAD` を渡し（`services/engines/ltx/adapter.py:1998`）、ワーカーが `te_offload_text_encoder` に変換（`engine/worker.py:438`）。実体は `engine/gemma/layer_offload_service.py:62` の `GemmaLayerOffloadService(layers_on_gpu, compute_device)`。`install(model)` が `.layers` の `nn.ModuleList`（要素数 4 超）を探し（`_get_layers` 177 行。`model`／`model.model`／`language_model` を順に見て、無ければ最大の ModuleList）、各層の forward を窓方式（`layer.to(device)`・`layer.to("cpu")` を無条件に呼ぶ。`_patch_layer` 271 行）に差し替える。torch のみに依存。
- **再利用できる部分**: `GemmaLayerOffloadService` は Gemma 固有の名前を持つが、探索は汎用（Qwen3 系の `model.layers` でも候補に入る見込み）。もう 1 つの道は `BlockSwapService`／先読みを TE の層列にも使うことだが、TE に先読みを使った前例はない（両エンジンとも TE は同期の素朴な出し入れ）。

### 項目 4. ヘッダの判定の経路

- **`sft_quant_format.py`**（644 行・torch 非依存）: `read_header`（262）・`parse_metadata`（346。`__metadata__` の各値を JSON として解釈）・`detect_prefix`（371。`model.diffusion_model.transformer_blocks.0.` か `transformer_blocks.0.` の有無で決め、どちらも無ければ `QuantFormatError`）・`inspect(path, expected_blocks=48)`（387）・`layer_schemes`（452）・`SCHEME_TABLE`／`SCHEMES`（132／140。`fp8`・`fp8_scaled`・`int8`・`int8_convrot`・`w4a8`）・ConvRot の groupsize は 256 のみ（157 行・`_scheme_of` 内の検査）。
- **`inspect` の段**: ① ヘッダ ② `__metadata__.config` に `transformer` 必須 ③ 接頭辞＋`transformer_blocks` がちょうど 48 個 ④ 層ごとの方式 ⑤ 量子化層が 1 本以上 ⑥ `_embeddings_connector.` のテンソルが必須。**②③⑥ が LTX 固有**。④（`layer_schemes`）は汎用で、H3 枝刈り版の dtype（BF16・F32・F16・I8・U8）と印（`int8_tensorwise`・`convrot: true`・groupsize 256）はすべて受理範囲。
- **`services/model_registry.py`**: `precheck_model_file`（118）は拡張子で分岐。`.safetensors` で `category == "transformer"` なら **系統を見ずに** `_precheck_sft_transformer`（200）→ `sft_quant_format.inspect`。合格すると **`{"general.architecture": "ltxv", "model_version": __metadata__.model_version}` を固定で作る**（216 行）。これが safetensors の「疑似 KV」。transformer 以外の `.safetensors` は長さと JSON の妥当性だけ（`_precheck_safetensors` 223）。
- **`services/engines/__init__.py`**: `FAMILY_BY_KV = {("ltxv","2.3"):"ltx", ("ltxv","2.5"):"ltx25"}`（43）。`check_kv`（106）は transformer だけ、`family_by_kv` が引けて記述子の系統と違えば 422（文面は「ltxv {version}（…）です」と ltxv を前提）。その後アダプタの `check_kv` に委ねる。`_require_family`（74）は未知の系統で `RuntimeError`。
- **`services/gguf_kv.py`**: GGUF の KV を読むだけ。H3（safetensors）には関係しない。
- **`services/weight_class.py`**: `GET /models` の `transformer_weight_class` を決める。safetensors は `inspect(...).layers` を数える（99 行）。失敗は INFO ログ＋`None`。
- **呼び出し元**: `api/pipeline.py:132` が `precheck_model_file` → `engines.check_kv`（145）。**ベースモデルが変わらず名前が `default` のカテゴリは precheck を飛ばす**（121 行）。ベースモデルを切り替えたときは全カテゴリを明示パスで通す。
- **H3 のファイルを今の経路に通した場合の追跡**:
  1. H3 を選んで `POST /pipeline/load`（ベースモデル切り替え）→ transformer の `.safetensors` → `_precheck_sft_transformer` → `inspect` の ② で「`__metadata__` に config がありません（ComfyUI でも LTX として組めない形です）」の `QuantFormatError` → **422 `MODEL_INCOMPATIBLE`**。系統の照合（`check_kv`）まで届かない。（非枝刈り版は `__metadata__.config` を持つが `transformer` キーが無い見込みなので同じく ② で落ちる。未検証）
  2. 仮に ② を越えても ③ `detect_prefix` が `transformer_blocks.0.` を見つけられず 422。
  3. text_encoder・video_vae・audio は `.safetensors` の汎用検査なので通る。
  4. 逆向き（LTX の safetensors を H3 のベースモデルで選ぶ）は `inspect` に合格して `ltxv 2.x` の疑似 KV が出来、`family_by_kv` が `ltx`／`ltx25` を返すので系統不一致の 422 になる（`FAMILY_DISPLAY` に H3 があれば文面も成立）。
  5. H3 のファイルを LTX のフォルダに置いた場合は 1 と同じ 422（LTX 側では「動いてしまわない」が成立。ただし文面は「H3 を選んでください」にはならない）。
  6. `GET /models` の `transformer_weight_class` は `None`（INFO ログ。`lru_cache` で再判定はファイル更新時のみ）。

### 項目 5. 記述子（manifest）

- **schema 2 の必須**（`services/base_models.py` `_parse_descriptor`）: `id`・`display_name`・`engine_family`（文字列）・`categories`（空でない object）。**任意**: `assets`（object、値は models_dir 相対）・`default_selection`（キーは宣言済みカテゴリ）・`comfort`（object であることだけ検査、中身は `api/context.py:188` の `build_comfort_budgets` が pydantic で検査）。`downloads`／`migrate` は Python 側では読まない。schema 2 かつ `engine_family` があるものだけがベースモデル。読み順はファイル名順（先頭が既定）。
- **`categories` の各欄**（`_parse_category`）: `scan`（必須・非空・相対パス）・`extensions`（必須・`.` 始まり）・`recursive`（既定 false）・`name_hint`（`video`／`audio` のみ想定。`_OPPOSITE_HINT` が 2 語の辞書なので他の語は KeyError。`services/model_registry.py:75`）・`default_file`。カテゴリ名の集合は自由だが、`services/model_registry.py:63` の `CATEGORIES = ("transformer","text_encoder","video_vae","audio")` が `PipelineManager.active_models` の形を固定している（`services/pipeline_manager.py:217`）。
- **`comfort` が無いとき**: `build_comfort_budgets` はその系統の行を作らない（202 行）。`GET /config` の `limits.comfort_budgets` に H3 のキーが無い。
- **フロントエンドの振る舞い**: `shell/comfortTable.ts` の `resolveComfortRow` は系統のプロファイルが無いと `null` を返す（181-189 行付近）。`modes/single/useGenerationForm.ts:699-717` は `null` のとき **`resolveSpillFreeFrames(config.limits.spill_free_frames, …)`（`config.yaml:179` の LTX の実測表）へ落ちて線を描く**。`modes/chained/useChainForm.ts:2475` は `chain_comfort_token_budget` へ落ちる。`shell/outpaintBudget.ts:33-35` だけは `null`（線なし）。Gradio も `gradio_ui/comfort.py:109-113` でスカラ値へ落ちる。**→ 記述子から `comfort` を省くだけでは快適上限マーカーは非表示にならず、LTX の値で描かれる**。加えて `tests/test_comfort_budgets.py:69` が「`comfort_budgets` のキー集合＝`FAMILY_BY_ID`」を固定しており、`comfort` 無しの系統を登録するとこのテストが落ちる。
- **`downloads` の処理**（`scripts/install_ltx.ps1` の手順 6）: 項目ごとに `files[].path`／`min` で既得判定 → `models\.dl\<id>-<n>\` へ `.venv-engine` の `huggingface_hub`（0.36.2）を `-m huggingface_hub.cli.hf download <repo> --include <全パターン> --local-dir <stage>` で取得（5 回まで再試行）→ 取得物を `map` の最長一致で `models/` へ移動（どの `map.from` にも当たらないファイルは例外で停止）→ 再判定 → 段階置き場を削除。`include` は fnmatch（`*` は `/` をまたぐ）。
  - **`Comfy-Org/MiniMax-H3` の `diffusion_models/…` のような下位ディレクトリからの取得は今の形で書ける**: `repo` は任意の公開リポジトリ名でよく（検証は空でないことだけ）、`include` に完全なファイルパス（例 `diffusion_models/<file>.safetensors`）、`map` に `{from:"diffusion_models", to:"<H3>/Weights"}` を書けばよい。ディレクトリ単位の `*` を使うと同じフォルダの bf16 版など全部を取る点に注意。
  - リビジョン（コミット）の固定欄は無い（常に main）。トークンも想定していない（公開・非ゲートが前提。`Comfy-Org/MiniMax-H3` がゲート付きかは未確認）。
  - 形の検証（`Test-ManifestShape` 303・`Test-BaseModelShape` 374）: `engine_family` は `^[a-z0-9_]+$`、`default_file` と `assets` の値は同じ記述子の `downloads[].files[].path` のどれかと一致しなければ失敗（downloads が空なら免除）、`files[].path` は同じ項目の `map.to` の下でなければ失敗、`migrate` はキーが無いと PowerShell の `@($null)` で失敗するので空配列でも書く（`30-uetrack.json` の注記）。
- **`migrate`**: 旧配置からの移動表。H3 には旧配置が無いので空配列。
- **`assets`**: 選び替え対象外の固定ファイル。アダプタの `REQUIRED_ASSETS` に挙げたものだけが `_real_available` の門になる（`services/engines/ltx/adapter.py:902`）。

### 項目 6. ディスパッチ

- **`services/pipeline_manager.py`**: 起動時は `runner_class_for(descriptor.engine_family)`（182 行。記述子が無いときは `LTXRunner`）。切り替えは `_point_runner_at`（563 行）が**型の完全一致**で比べ、系統が変われば旧ワーカーを unload してから新しい Runner を作る。`active_engine_family`（251 行）を `api/generate.py:79`・`api/generate_chain.py:58`・`api/generate_alpha.py:153` が使う。
- **`config.py`**: `ModelConfig.engine_python`（112）・`engine_python_ltx25`（122）。3 つ目は同じ形で 1 項目を足す形（アダプタの `_engine_python_value` が読む）。`EngineComfortProfile`（285）・`LimitsConfig.comfort_budgets`（428 付近）。
- **ltx25 アダプタの骨格**（`services/engines/ltx25/adapter.py`・1452 行）:
  - モジュール定数: `SELECTION_FIELDS`（75）・`REQUIRED_ASSETS`（100）・`SUPPORTED_MODEL_VERSIONS`（105）・`DEFAULT_BLOCKS_ON_GPU`（120）・`REJECT_TABLE`（140）・`IGNORED_FIELDS`（153）・`HONOURED_FIELDS`（233）・`GOVERNED_FIELDS`（296）・`CHAIN_REJECT_TABLE`（318）・`CHAIN_IGNORED_FIELDS`（329）・`CHAIN_HONOURED_FIELDS`（396）・`CHAIN_GOVERNED_FIELDS`（459）・`UNSUPPORTED_FEATURES`（467）。関数 `reject_unsupported`（472）・`reject_chain`（497）・`_log_ignored`（523）・`check_kv`（551）。
  - `_RealBackend25(_RealBackend)`（604）: クラス属性 `SELECTION_FIELDS`・`REQUIRED_ASSETS`・`_WORKER_MODULE="engine25.worker"`・`_LOG_NAME`・`_ENGINE_DIR_VALUE="./engine25"`・`_ENGINE_PYTHON_LABEL`。メソッド `_engine_python_value`・`_build_child_env`・`_log_load_start`・`_build_load_payload`（673。キー集合と順序はゴールデン スナップショットで固定）・`generate`（730〜1108）・`generate_chain`（1109〜1434）。
  - `LTX25Runner(LTXRunner)`（1435）: `_REAL_BACKEND_CLS`・`_MOCK_BACKEND_CLS`（2.3 の `_MockBackend` を再利用）・`_MOCK_BACKEND_LABEL` だけ。
  - **load ペイロード**: `op`・`transformer_path`・`text_encoder_path`・`video_vae_path`・`audio_vae_path`・`spatial_upsampler_path`・`blocks_on_gpu`・`cache_weights`・`deterministic`。**generate ペイロード**（ワーカー側の説明 `engine25/worker.py:52`）: `prompt`・`seed`・`width`・`height`・`num_frames`・`frame_rate`・`output_path`・`images`・`loras`（常に存在）・`reference_video`・追加ブロック（`outpaint`／`inpaint`／`alpha_gen`）・高速化のつまみ。
  - **2.3 アダプタとの共有**（50-61 行の import）: `_MockBackend`・`_RealBackend`（起動・`_send`・`_read_event`・`_read_worker_events`・unload・`_require_*`）・`GenerationOutcome`・`LTXRunner`（記述子の解決・mock/real の選択・`_real_available`・sage の探査）・`LTX_ARCHITECTURE`・`_lora_payload_entry`・`_minor_version`・`_resolve_reference_preprocess`・`resolve_seed`。
- **最小のアダプタに要るもの**（見積もり）: `SELECTION_FIELDS`・`REQUIRED_ASSETS`（空でよい見込み）・`check_kv`・`UNSUPPORTED_FEATURES`＋`reject_unsupported`／`reject_chain`（未対応の機能を 422 にする表）・`_RealBackend` の子（上記クラス属性 6 個＋`_engine_python_value`・`_build_child_env`・`_build_load_payload`・`generate`）・Runner の子（3 属性）。`generate_chain` は全面拒否なら不要（`reject_chain` で止まる）。mock は 2.3 の `_MockBackend` が 8n+1・64 の倍数の LTX 規約で絵を作る（`services/engines/ltx/adapter.py:1060`）ので、そのまま継承すると H3 の規約と食い違う。
- **1452 行のうち H3 で要らない部分の見積もり**: `generate_chain` 約 325 行・`generate` 内の outpaint／inpaint／alpha_gen／LoRA／参照動画の分岐（380 行の大半）・2 種類 × 4 分類の表（約 330 行）の中身。最小アダプタは 150〜300 行程度と見込む（実装していないので推計）。
- **注意**: `_real_available`（`services/engines/ltx/adapter.py:902`）は `<engine_dir>/worker.py` の存在を門にしているので、H3 のワーカーは `worker.py` という名前でパッケージ直下に置く前提になる。進捗の割合は段階名（`stage1_denoise` など）に依存しており、単段の生成は `one_stage` の経路（Alpha Gen 用）が前例。

### 項目 7. API 層の制約（後の弾の大きさの見積もり用）

- **8n+1**（`api/models.py`）: 201（`frame_idx` のスナップ `(raw-1)//8*8+1`）・430（inpaint の `num_frames`）・650（`GenerateRequest.num_frames`）・867（`source_video.context_frames`）・1468（`clips[i].num_frames`）。関連して 8 の倍数の検査が 937・942（retake の `head_px`／`tail_px`）・1146（end_source の `context_frames`）。
- **64 の倍数**: 646・648（`GenerateRequest`）・1336・1338（`GenerateChainRequest`）・`ceil64`（779-780、Alpha Gen）。
- **fps**: `frame_rate: float = Field(24.0, ge=1.0, le=60.0)`（592・1225）。既定 24。`chain_math.a_frames_for_px(pixel_frames, fps)`（446）が音声潜在＝毎秒 25 の規約で使う。
- **範囲**: `num_frames` は 9〜481（591・822）、inpaint は 9〜145（416）。`width` 256〜4096・`height` 128〜4096。
- **これらは系統を知らない pydantic 検証**で、アダプタの `reject_unsupported` より前に走る。参考事実として、8n+1 と 17n+5 の両方を満たすのは 9〜481 の範囲で **73・209・345・481** の 4 値、64 の倍数は 32 の倍数の部分集合なので、今の API のままでも H3 の規約を満たす要求は作れる（fps 24 固定の強制は別途必要）。
- **4 分類の監査テスト**: `tests/test_ltx25_adapter.py:3118`（単発の未分類検出）・3161（連結）、`tests/test_ltx_adapter_audit.py:68`。`GenerateRequest`／`GenerateChainRequest` にフィールドを足すと既存 2 系統の監査が落ち、表の更新が強制される。H3 のアダプタが表を持つなら同形の監査テストを新設する形になる。
- **フロントエンド**: `paramUtils.ts:45`（`snapNumFrames`）・`:53`（`isValidNumFrames` が `(value-1)%8===0`）。8n+1 の記述は 37 ファイル。`temporal_factor` の利用は `shell/comfortTable.ts`（`tf*(n-1)+1` で画素フレームへ戻す式＝8n+1 型の格子を前提）ほか 10 ファイル。`mockBridge.ts:316-317` に `MOCK_BASE_MODELS`（LTX23・LTX25 の 2 件）、1588・1650 に 8n+1 の検証。`featureScope.ts`（329 行）は `unsupported_features` の機能名から閉じる UI 部品を引く宣言表で、系統 id には依存しない。
- **MCP・Gradio**: 8n+1 の記述が `mcp_server/{params.py, server.py, batch_planning.py, tools/generate.py}`・`gradio_ui/{handlers.py, i18n.py}` にある。

### 項目 8. インストーラ

- **`Ensure-EngineVenv`**（`scripts/install_ltx.ps1:1396`）: 必須 `-VenvPath`・`-PythonPath`・`-StateFile`・`-FreezeFile`・`-Label`、任意 `-DirectPins`・`-DirectPinArgs`・`-DirectPinsLabel`・**`-IndexUrl`（既定 cu128）**・`-StackLabel`・`-IgnoreSkipVenv`。`uv venv --python 3.12` → `Invoke-EngineFreezeApply`（1328。(a) 直接ピンを `uv pip install`、(b) freeze から直接ピンの行を除いて `--index $IndexUrl --index-strategy unsafe-best-match -r`）→ 状態ファイル `<venv>/.nz-engine-state` に freeze＋直接ピンのハッシュ（`Get-EngineStateHash` 1289）。ハッシュが変われば再適用。
- **LTX 2.5 の呼び出し**（1460 行付近）: `-DirectPins $ltx25DirectPins`（torch・torchaudio・ltx-core・ltx-pipelines・sageattention）と `-DirectPinArgs $ltx25UvArgs`（1256。`--no-sources --index cu128 --index-strategy unsafe-best-match` を**固定文字列**で持つ）。
- **cu130 への変更可否**: (b) 段は `-IndexUrl` 引数で変えられる。(a) 段の index は `-DirectPinArgs` に渡す配列次第なので、H3 用に別の配列を渡せば変えられる。どちらもコードの関数側の変更は不要。
- **`-BaseModel`**（141 行）: 既定 `LTX23, Preprocessors`。記述子の `id` で取得と検証の対象を絞る（大文字小文字は無視）。
- **`setup.ps1`**: `install_ltx.ps1` を `-BaseModel` 指定なしで呼び、`.venv`・`.venv-engine`・`.venv-engine-ltx25` を必ず作る。
- **`scripts/install_model.ps1`**: `install-<ID>.bat` の共通本体。記述子の受け付けは「`engine_family` あり」または「`opt_in: true`」（`Get-BaseModelPlan` 79）。`Test-SetupDone`（159）が uv・`.venv`・`.venv-engine`・`.venv-engine-ltx25` の存在と `huggingface_hub` の起動を確認。最後に `install_ltx.ps1 -BaseModel <ID> -SkipVenv -SkipMigrate`（377）。
- **`install-LTX25.bat`**: 取得のみ（venv を作らない）。**`install-UETrack.bat`**: 同じ本体を呼ぶが、`install_ltx.ps1:1485` の `if ($foundIds -contains 'UETrack') { Ensure-EngineVenv … -IndexUrl cpu -IgnoreSkipVenv }` により **`-SkipVenv` を越えて `.venv-utils` を作る**。検証表にも UETrack 選択時だけ venv とワーカーの行を足す（1783 行）。
- **`install-MiniMaxH3.bat`（venv の構築＋取得を 1 回で）に近い前例は UETrack**（任意導入の venv を `-IgnoreSkipVenv` で作る形）。違いは、UETrack が CPU 版 torch・直接ピンなし・ベースモデルでない（`opt_in`）のに対し、H3 は CUDA 版 torch・直接ピン（comfy-kitchen など）・ベースモデル（`engine_family` あり）である点。取得は `.venv-engine`（LTX 2.3 の venv）の `huggingface_hub` に依存するので、setup.bat 済みが前提のまま。

### 項目 9. テスト（3 つ目の系統を足したときに落ちるもの）

| テスト | 何を固定しているか | 落ちる条件 |
|---|---|---|
| `tests/test_engine_dispatch.py:139` | `set(FAMILY_BY_ID) == set(FAMILY_DISPLAY)` | 片方だけ足す |
| `tests/test_engine_dispatch.py:141` | `FAMILY_BY_KV` の完全一致（2 行） | `FAMILY_BY_KV` に H3 の行を足す |
| `tests/test_engine_dispatch.py:206-210` | 系統ごとの `unsupported_features` の値 | 既存系統の値を変えた場合 |
| `tests/test_engine_dispatch.py:233-235` | 出荷の記述子ごとに `categories` と その系統の `SELECTION_FIELDS` の一致 | H3 の記述子とアダプタのずれ／アダプタ未登録（`FAMILY_BY_ID` に無いと KeyError） |
| `tests/test_comfort_budgets.py:69` | `comfort_budgets` のキー集合＝`FAMILY_BY_ID` | **`comfort` を持たない系統を登録した時点で落ちる** |
| `tests/test_base_model_contract.py:56-58` | 出荷記述子のカテゴリ順が `transformer, text_encoder, video_vae, audio` | H3 のカテゴリ名か順序が違う |
| `tests/test_base_model_contract.py:84-89` | LTX23・LTX25 の transformer 拡張子 | 影響なし（名指し） |
| `tests/test_ltx25_adapter.py`・`tests/test_ltx_adapter_audit.py` | 4 分類の網羅監査 | `GenerateRequest` 等にフィールドを足した場合 |
| `tests/test_model_swap_load.py` | 既存系統のロードペイロードのゴールデン | 既存アダプタを触った場合のみ |

- `tests/conftest.py` は一時ディレクトリに記述子を書き（`write_manifests` 117 行・`base_model_descriptor` 80 行）、`model.backend = "mock"`（156 行）・`tracking.backend = "mock"` を強制する。API 系のテストの大半は出荷の記述子を読まないので、H3 の記述子追加では落ちない。出荷の記述子を読むのは `test_base_model_contract`・`test_comfort_budgets`・`test_engine_dispatch`・`test_model_swap_load`（317 行。LTX23 名指し）など。
- ワーカーの単体テストは各エンジンの venv で `pytest.importorskip("torch")`＋`importorskip("ltx_core")` を付けて走らせる形（`tests/test_worker_ltx25_unknown_op.py:28-29`）。アプリの venv ではスキップされる。
- `install_ltx.ps1` の `Test-ManifestShape` は出荷の全記述子を毎回検証する（`-BaseModel` に関係なく）。H3 の記述子の形が崩れていると setup.bat も止まる。

### 項目 10. モデルの置き場

- 配置はベースモデル優先 `models/<BaseModel>/<Category>/…`（`install_ltx.ps1` 冒頭 .NOTES）。LTX 2.5 は `models/LTX25/{Weights, TextEncoder, VAE, Upscaler, IC-LoRA}`。各フォルダに `put_*_here.txt` の案内ファイル。
- `models/INSTALLED_PATHS.txt` は `install_ltx.ps1` の手順 7 が記述子の `files[]` から再生成する（ラベル＝`files[].label`）。
- `config.yaml` の `model.*` に重みのパスは持たない（`config.py` 95-103 行の説明。記述子の `default_file` と `assets` が正本）。`models_dir`・`manifest_dir` だけ設定に残る。
- 走査（`services/model_registry.py` `_scan_category` 448）: `recursive` が false なら `iterdir`（直下だけ）、true なら `rglob`。`.` で始まる部分を含むパスは除外。拡張子と `name_hint` で絞る。既定ファイルと同じ実体は `default` 以外の名前を作らない。

### 項目 11. ワーカーの出力の慣例

- ジョブの出力は `outputs/<job_id>/output.mp4`＋`metadata.json`（`services/pipeline_manager.py:653`・1556・1590）。`metadata.json` を書くのはアプリ側（`_write_metadata` 1855。`services/video_io.save_metadata` 1759）。ワーカーは `output_path` に mp4 を 1 本書き、メタ情報は `done` イベントで返す。
- `services/video_io.py` はアプリ側の ffmpeg 部品（`encode_frames_to_mp4` 37・`crop_mp4` 99・`embed_comment_tag` 1786 など）。**エンジン側は `services` を import しない**（アプリと推論の両方にトップレベルの `services` があり衝突する。設計書 §5.3）。
- セルフテストの成果物は、過去の実測では `outputs/<作業名>/`（例 `outputs/alphagen-gate0`・`outputs/ltx25-accel-gate`・`outputs/comfort-calib-<日付>`）に mp4 とレポート JSON を置いている。`--report` で JSON を書き、標準出力にも同じ JSON を出す。
- mp4 の書き方の前例は LTX 2.5 の `encode_video`（PyAV・libx264・crf 19・yuv420p・BT.709・音声多重）。プロジェクトには `tools/ffmpeg/bin` もある（`install_model.ps1` が PATH に足す）。

---

## (3) 危険な点と不明点

1. **量子化テンソルを block swap に乗せる方法**: 先読み（`block_swap_prefetch.py`）は `GGMLQuantizedTensor` だけを特別扱いし、それ以外は「素のテンソル」として `reshape(-1).view(torch.uint8)` で詰め、GPU 側も素のテンソルとして作り直す。comfy-kitchen の量子化テンソルが**ラッパー型のサブクラス**（内部に int8 本体とスケールを持つ型）なら、(a) `view(uint8)` が通らず `prepare()` が例外 → 黙って同期経路へ縮退（こだま `on->off`）、または (b) 通っても往復でクラスが失われる、のどちらかになる見込み（推測。comfy-kitchen の実物はこの機械に無く未確認）。同期経路の `p.data = p.data.to(device)` も型の互換を要求する。既存の int8 ConvRot は「素の int8 パラメータ＋スケールのバッファ＋Linear の forward 差し替え」という形で乗っている事実がある。どちらの形にするかは設計判断で、共有モジュールを変える場合は LTX 2.3／2.5 の先読み（ゴールデンの一致検証あり）に波及する。
2. **共有モジュールの依存**: 必要な部品は torch だけで import できるが、`engine/sft_quant/quant_service.py` の「組み立て」側（`install`・ローダー）は ltx_core 前提で使えない。`engine/transformer/sage_attention_service.py` の差し込みも ltx_core の `Attention` 型前提。torch の版を LTX と変える（裁定 11）と、共有部品は**2 つの torch 版で動く**ことが新たな前提になる（今は両 venv とも 2.9.1+cu128）。`block_swap_prefetch.py` の説明にある Windows のアロケータの挙動（`expandable_segments` が無効）も torch 2.9.1 での実測。
3. **groupsize**: `dequant.py:44` と `sft_quant_format.py:157` は ConvRot の groupsize 256 だけを受け付ける。非枝刈り版（AdaLN 射影が 64）とコミュニティの 64 版は通らない（研究ノート §13.7 と同じ）。
4. **`inspect` の転用不可**: ② config・③ 48 ブロック・⑥ connector の 3 段が LTX 固有。`precheck_model_file` に系統の情報が渡っていない（`descriptor` は拡張子の門にだけ使われる）ので、H3 用の判定を足すには呼び出しの形の変更が要る。疑似 KV（`general.architecture`）を何にするか、`FAMILY_BY_KV` に載せるかは未決。
5. **快適上限マーカーの非表示**: `comfort` を省いてもフロントエンドと Gradio は LTX のスカラ値・`spill_free_frames` へ落ちて線を描く。さらに `tests/test_comfort_budgets.py:69` が「系統の集合＝comfort の集合」を固定しているので、`comfort` 無しの登録はテストの変更を伴う。
6. **テストの連鎖**: 記述子だけを先に置くと、`GET /models` が `engines.unsupported_features` → `_require_family` で `RuntimeError`（未知の系統）になり、`test_engine_dispatch.py:233` も KeyError で落ちる。記述子・`FAMILY_BY_ID`・`FAMILY_DISPLAY`・アダプタの `SELECTION_FIELDS` は同時に入れる必要がある。
7. **API の検証が系統を知らない**: 8n+1・64 の倍数・fps の検証は pydantic の段で系統を問わず走る。H3 の 17n+5 を受けるには、検証を系統別にする仕組みが要る（設計書 §7 で「スコープ外」とされた領域に踏み込む）。重なる値は 73・209・345・481 だけ。
8. **mock**: `_MockBackend` は LTX の規約（8n+1・`chain_math`）で絵を作る。H3 のアダプタが継承すると規約が食い違う（設計書 §9.1 の注記どおり再判断が要る）。
9. **インストーラ**: 取得は `.venv-engine` の `huggingface_hub` 0.36.2 頼み（1.x では単一 `--include` が無視される既知の落とし穴）。`Comfy-Org/MiniMax-H3` のゲートの有無・リビジョン固定欄が無いこと・音声 VAE の実ファイル名（`name_hint` の `audio` に合うか）は未確認。
10. **文書の食い違い**: `Docs/MULTI_ENGINE_DESIGN.md` §5.3 の「全エンジンで torch のバージョンを揃える【オーナー裁定】」と、H3 の裁定 11（揃えなくてよい）が併存している。
11. **ワーカーの出口**: LTX 2.5 の mp4 書き出しは ltx_pipelines／ltx_core に依存しているため H3 では使えない。PyAV を H3 の venv に入れるか、ffmpeg を呼ぶかは未決。
12. **未確認（推測を含む）**: comfy-kitchen の量子化テンソルの型の性質（ラッパー型か・`.to()`／`set_data`／`view` の挙動）、Qwen3-VL の層構造が `GemmaLayerOffloadService._get_layers` の探索に掛かるか、H3 の各ブロックの forward が「1 ブロックずつ順に呼ぶ」形か（block swap の窓はこれを前提にする）。
