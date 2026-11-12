# MiniMax H3 の fp8・w4a8 の GPU の門（G1〜G6）の台本

計画 `imperative-crafting-lightning.md` §5.2 の G1〜G6 を、利用者と同じ経路（MCP〔Model Context Protocol。外部の道具からアプリを操作する仕組み〕→ アプリのサーバー → エンジン）で順に回す台本です。G7（事前検査で拒否した理由が操作パネルの Settings に出ること）は手で確かめます。組み立ては Inpainting の門 `Docs/Outputs-archive/h3-inpaint-2026-11-11/h3_inpaint_gate_api.py` の写しです。

## 回し方（リポジトリの根で。Git Bash。アプリの仮想環境の python）

```bash
export PYTHONUTF8=1
G=outputs/h3-fp8-w4a8-gate-2026-11-12/h3_quant_gate_api.py
.venv/Scripts/python.exe $G --dry-run          # 計画を表示するだけ。サーバーも MCP も起こさない。GPU に触れない
.venv/Scripts/python.exe $G                    # G1 → G6 を順に（サーバーの起動は 1 回）
.venv/Scripts/python.exe $G --only fp8         # 1 条件だけ（int8・fp8・w4a8_kijai・w4a8_eros・int8_oldmarker・fp8_turbo）
.venv/Scripts/python.exe $G --post-only        # runs/ にある結果から検査・並び画像・summary.md・hf-round17/ を作り直す
```

本番の前に、オーナーの了承と GPU の空きを確かめてください。台本自身も次の場合は起動を断ります。

- ポート 18620 で何かが待ち受けている（オーナーのサーバー）
- ワーカー（`engine_h3.worker` など）のプロセスが残っている
- `nvidia-smi` に別の python／uvicorn が出ている、または GPU の使用量が 3,072 MiB を超えている
- `runs/<条件>/` に前の結果（`output.mp4` か `result.json`）がある（上書きしない。別の場所へ移してから回す）

`config.yaml` は読むだけです。ポート・ログ・出力・アップロード・状態ファイルの置き場を `_rt/` に向けた一時の設定（`_rt/config.h3quantgate.yaml`）でサーバーをポート 18699 に起こし、終わったら（失敗や中断でも）プロセスの木ごと止めます。

## 条件

共通: T2V（文章から動画）・1280×768・73 枚・種 1234・プロンプトはスパイク（`outputs/h3-fp8-w4a8-spike-2026-11-12/h3_quant_spike.py` の `DEFAULT_PROMPT`）と同じ・注意の実装 sage・VRAM 16GB mode on・先読み on・常駐 off。条件ごとに `unload_pipeline` → `load_pipeline` で重みを載せ替えてから生成します。

| 門 | 条件 | 重み（`models/MiniMaxH3/Weights/`） | ステップ | shift（映像／音声） | LoRA |
|---|---|---|---|---|---|
| G1 | `int8` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors`（回帰の基準） | 20 | 12／3 | なし |
| G2 | `fp8` | `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | 20 | 12／3 | なし |
| G3 | `w4a8_kijai` | `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | 20 | 12／3 | なし |
| G4 | `w4a8_eros` | `10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors` | 8 | 10／3 | なし |
| G5 | `int8_oldmarker` | `h3ErosMax_beta5_3185154.safetensors`（旧式の印。読み込めて完走すること） | 8 | 10／3 | なし |
| G6 | `fp8_turbo` | G2 と同じ fp8 | 4 | 12／3 | Turbo（`minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16`・強さ 1.0） |

G6 の shift: 研究ノート `Docs/MINIMAX_H3_RESEARCH_NOTES.md` には FL2V 768p の Turbo LoRA の shift の推奨がありません（835 行目は Ref2VA 用の Turbo で 4 ステップ・12／3）。これまでの Turbo LoRA の実行（記録簿 `Docs/VERIFICATION_LOG.md` の 22308 行目・22565 行目）と同じ 12／3 にしました。

## 記録するもの

条件ごとに `runs/<条件>/` へ `output.mp4`・`metadata.json`・`result.json`・`request.json`・サーバーとワーカーのログの切り出し・`sheet.png`（フレーム 0・36・72）。

- 読み込みの秒（`load_pipeline` から読み込み完了まで）・ジョブの秒・秒／ステップ（サーバーのログの `denoise k/N steps (..., Ts elapsed)` から。スパイクと同じ定義）・VRAM の山（`h3.phases.denoise.peak_vram_mb`／`peak_vram_reserved_mb`）・`h3.quant_backends`
- 読み込みの報告（`fp8_mode` と形式ごとの層の数）: API にも `metadata.json` にも出ないので、ワーカーのログ `_rt/logs/h3_worker.log` の `[h3_worker] LOAD_OK {json}` の行（`engine_h3/worker.py` の `_do_load` が書く `build_report`）から読みます。`fp8: N layer(s), path '<方式>'` の行も拾います。

自己検査（`result.json` の `post.checks`）: ジョブが completed・出力が 1280×768・73 枚・音声あり・使った重みのファイル名・ステップと shift・`h3.quant_backends.cuda` に新しい能力の欄（`w4a8_int8_linear`・`quantize_per_tensor_fp8`）があり、その形式に要るものが真・`LOAD_OK` の行がある・`fp8_mode`（G2・G6 は `scaled`、ほかは null）・形式ごとの層の数がファイルのヘッダー（`h3_format.inspect_h3_dit`）から数えた数と一致・G6 は LoRA が適用された記録（`h3.loras`）・G2〜G6 は `MODEL_INCOMPATIBLE` が出ていない。

## 出力

- `summary.md`・`summary.json`: 全条件の数値の表と検査の結果
- `hf-round17/`: `h3_<条件>_1280x768_<ステップ>steps_seed1234.mp4`・`sheet_<条件>.png`・`sheet_all.png`・`README.md`（日本語の骨組み。「所見」は空欄）。Hugging Face の `Rootport/Nz-LTX23-eval-clips` の `round17-h3-quant-product/` へは監督が上げます（台本は上げません）
- `logs/`: 実行ごとのログ・MCP の呼び出しの記録・MCP サーバーの標準エラー
- `_rt/`: 一時の設定・状態・サーバーとワーカーのログ・アプリの出力とアップロード

終了コード: 0＝回した条件がすべて completed で後処理まで済んだ（検査の合否は記録に書くだけ）、1＝前の確かめ・準備で止まった、2＝引数の誤り、3＝ジョブの失敗・拒否・待ちきれない。
