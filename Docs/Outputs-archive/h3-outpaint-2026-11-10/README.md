# MiniMax H3 の Outpainting の GPU の門（エンジン直接の台本）

`h3_outpaint_gate.py` は、MiniMax H3 の Outpainting（元の動画の外側へキャンバスを広げて描き足す機能。台帳 `Docs/PENDING_TASKS.md` §1-125）の弾 1 の GPU の門 G0〜G4 を回す台本です。門の設計は実装計画（`rosy-jingling-stonebraker.md` の §3「弾 1」と §7「GPU の門」）、仕組みは設計正本 `Docs/MINIMAX_H3_DESIGN.md` §3.1 です。

この台本はサーバーも MCP も通さず、エンジンの仮想環境で `engine_h3.pipeline_h3.H3Pipeline` を直接呼びます（前例は `Docs/Outputs-archive/h3-chain-2026-10-29/c_chain.py`）。製品コードは変えません。この保管庫には台本・この説明書・結果の JSON（`summary.json`。実行後に監督が置く）だけを置き、動画・素材・ログは置きません。数値はここに書きません（数値の正本は記録簿 `Docs/VERIFICATION_LOG.md`）。

## 目的

- **G0**: キャンバスごと符号化したとき、余白の詰め物が元の領域の潜在をどれだけ汚すかを測ります。設計（元の映像を外接矩形で単独に符号化して、キャンバスの潜在へ貼る）の根拠の実測です。
- **G1〜G4**: トークン単位の固定で生成した出力について、保持矩形（元の映像を置いた領域）の画質、継ぎ目の立ち方、記録の中身、所要秒数、VRAM の山を取ります。合否は付けず、数値を取るだけです。最終の判断はオーナーの目視です。

## 前提

- GPU を使う実行は、毎回オーナーの了承を得てから行います。`--prepare` と `--dry-run` は GPU を隠して（`CUDA_VISIBLE_DEVICES=-1`）CPU だけで動きます。
- バックエンド（サーバー）と AviUtl2 は閉じておきます。この台本は自分のプロセスでモデルを読み込みます。
- 使う python はエンジンの仮想環境 `.venv-engine-h3`。リポジトリの根を `PYTHONPATH` に入れ、`PYTHONUTF8=1` を設定します。
- ffmpeg と ffprobe は `<repo>\tools\ffmpeg\bin\` のものを使います（無ければ PATH）。
- 元の素材は `outputs\h3-retake-gates-2026-11-08\_material\H3-Ref-1080p-30s.mp4`（1920×1080・24fps・30 秒・ステレオ 44.1kHz。Retake の門で作ったもの）。

## コマンド

cmd から（リポジトリの根で）:

```
set PYTHONPATH=%CD%
set PYTHONUTF8=1
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --prepare
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --dry-run
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --gate G0
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --gate G1,G2,G3,G4
```

Windows PowerShell 5.1 から:

```
Set-Location "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni"
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONUTF8 = "1"
& ".\.venv-engine-h3\Scripts\python.exe" ".\Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py" --dry-run
& ".\.venv-engine-h3\Scripts\python.exe" ".\Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py" --gate G0
& ".\.venv-engine-h3\Scripts\python.exe" ".\Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py" --gate G1,G2,G3,G4
```

G0 と生成の門は、別々のプロセスで回すことを勧めます。同じプロセスで回すと、G0 の VAE が使った領域がキャッシュに残った状態で生成が始まり、生成の VRAM の山が単独のときと変わりうるためです（台本は G0 の前に cuBLAS の作業領域を先に確保し、G0 の後に解放しますが、断片化までは戻せません）。

`--dry-run` は素材が無ければ先に `--prepare` と同じことをし、幾何の計算（固定トークン数）・各本の要求の中身・`validate_request` の検査（元動画と LoRA のファイルの有無とヘッダ）・出力先のフォルダの作成までを行って、表と要求を表示します。モデルは読みません。

## 引数

| 引数 | 既定 | 意味 |
|---|---|---|
| `--gate` | `G0,G1,G2,G3,G4` | 回す門（カンマ区切り）。`E1`・`E2` は既定に入らず、明示したときだけ回す（下の「実験 E1」「実験 E2」） |
| `--prepare` | 切 | 素材の切り出しと対照を作って終わる（CPU） |
| `--dry-run` | 切 | モデルを読まずに幾何と要求の一覧を出す（CPU） |
| `--seed` | 1234 | 種（全本で同じ） |
| `--prompt` | 下の「プロンプト」 | 全本で同じ文。負のプロンプトは無し |
| `--steps` | 4 | ステップ数（Turbo LoRA の 4 ステップ） |
| `--attention` | E2 は `sage`、G0〜G4・E1 は `sdpa` | 注意の実装（`sdpa`／`sage`）。指定すると全部の本がそれになる。各本の要求に渡し、読み込み時には最初の本の値を渡す |
| `--no-lora` | 切 | Turbo LoRA を付けない（そのときは `--steps 20` などを合わせて指定する） |
| `--vram16-off` | 切 | VRAM 16GB mode を切る（既定は on。アプリが H3 で送る既定と同じ） |
| `--out-dir` | `outputs\h3-outpaint-gates-2026-11-10` | 出力の置き場（git の追跡外） |

### 固定の条件

| 項目 | 値 | 根拠 |
|---|---|---|
| DiT の重み | `models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors`（FL2VA） | 前例 `c_chain.py` の `PATHS` |
| Turbo LoRA | `models/MiniMaxH3/LoRA/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors`・強度 0.8 | 置き場は `scripts/manifests/40-minimaxh3.json` の `lora_dir`。名前と強度は前例 `h3-vsf-2026-11-09/h3_vsf_gate.py` の T 系列と `h3-chain-a2v-2026-11-09/h3_chain_a2v_gate.py` の GA3T。エンジンへは `engine_h3/worker.py` と同じく `H3LoraInput(path=<絶対パス>, strength)` で渡す |
| ステップ・shift | 4 ステップ・shift 映像 12／音声 3 | 前例と同じ（Turbo でも shift は据え置き） |
| 読み込み | `blocks_on_gpu=8`・先読み on・`deterministic=False` | `H3LoadOptions` の既定（前例 `c_chain.py` と同じ） |
| 常駐 | `keep_resident=True`。読み込みは 1 回、6 本を順に回し、最後に `close()` | 前例 `c_chain.py` |
| フレーム数 | 124（17×7＋5）・24fps | |

### プロンプト

既定の文は「A wide shot of a young woman in a black-and-white maid costume talking cheerfully to the camera in a spacious pastel room decorated with sweets and flowers; natural head and hand movement, steady camera.」です。素材（Retake の門の素材と同じ人物と部屋）に合わせ、外側に部屋が広がる構図を指す語（wide shot・spacious）を足しました。

## 素材（`--prepare`）

`outputs\h3-outpaint-gates-2026-11-10\_material\` に次を作ります。

| ファイル | 内容 |
|---|---|
| `src_1280x720_124f.mp4` | 元の素材の 2.0 秒（フレーム 48）から 124 枚、1280×720 へ縮小（lanczos）。libx264 crf 12・yuv420p・24fps CFR、音声は PCM（断られたら AAC 192k）。製品の窓の切り出し `services/video_io.py` の `cut_window_mp4` と同じ作法に、縮小だけを足したもの。ffprobe の `-count_frames` で 124 枚・1280×720 でなければ止まる |
| `control_write_mp4_1280x720_124f.mp4` | 対照。上の切り出しを `read_source_frames` で読み、出力と同じ `engine_h3.media_out.write_mp4` で書き直したもの（音声なし）。出力 mp4 の保持矩形はこの書き直しと同じ再符号化を一度経ているので、比べる相手にする |
| `prepare.json` | ffmpeg のコマンド、ffprobe の結果、`read_outpaint_source` を s=1.0 で読んだ結果（窓の全長のサンプル数がそろうこと）、対照と切り出しの PSNR |

奇数 pad の本も同じ素材を使います。既にある素材は作り直しません（無いものだけ作り、`prepare.json` に追記します）。実験 E1 の素材は下の「実験 E1」です。

## 各門の条件

キャンバスは元の寸法（1280×720）＋pad。なじませ幅は計画 §2.2 の式 `feather_px = floor(r × max(W_c, H_c) / 64 + 0.5)`。

| 本（フォルダ名） | 門 | キャンバス | pad 左／右／上／下 | r（なじませ px） | 音声の Strength s | 注入係数 | 目的 |
|---|---|---|---|---|---|---|---|
| `G0` | G0 | 1920×1088 | 320／320／184／184 | — | — | — | VAE の往復だけ（下の表） |
| `G1_main` | G1 | 1920×1088 | 320／320／184／184 | 1（30） | 0.5 | 1.0 | 主条件。G2 の r=1 と G4 の s=0.5 を兼ねる |
| `G1_odd` | G1 | 1312×736 | 13／19／8／8 | 1（21） | 0.5 | 1.0 | 奇数 pad・32 に乗らない例 |
| `G2_r0` | G2 | 1920×1088 | 320／320／184／184 | 0（0） | 0.5 | 1.0 | 硬い貼り付け |
| `G2_r3` | G2 | 1920×1088 | 320／320／184／184 | 3（90） | 0.5 | 1.0 | 広いなじませ |
| `G3_inject0999` | G3 | 1920×1088 | 320／320／184／184 | 1（30） | 0.5 | 0.999 | 対照。`engine_h3.outpaint_h3.OUTPAINT_INJECT_CLEAN` を `generate()` の前に差し替え、本の後に 1.0 へ戻す |
| `G4_s1` | G4 | 1920×1088 | 320／320／184／184 | 1（30） | 1.0 | 1.0 | 元の波形が mp4 に入る（`audio_output: "original"`） |

G0 の符号化の 4 通り（幾何は G1 の主条件と同じ。124 枚全部を 17 フレームの塊の規則で `encode_clips(tiled=True)` に通し、復号器は bf16 の `H3VideoDecoder`）:

| 名前 | 符号化するもの | 復号するもの |
|---|---|---|
| `single` | 外接矩形 [320,160)-[1600,928) を `aligned_frames`（すき間は鏡写し）で組んだ 1280×768 | その潜在全部 |
| `canvas_imagenet` | 1920×1088 のキャンバスに元を置き、余白を (124,116,104) で埋めたもの | 外接矩形に当たる潜在だけを切り出したもの |
| `canvas_gray128` | 同じく余白を (128,128,128) で埋めたもの | 同上 |
| `control_source` | 元の 1280×720 をそのまま | その潜在全部（VAE の往復の下限） |

## 記録するもの

G0:

- 潜在の差: `single` を基準に、`canvas_*` との相対 L2（全チャンネル・全潜在フレーム）。固定トークンに当たる潜在マス全体（`fixed`）、生成トークンに 8 近傍で接する固定トークン（`fixed_edge`）、その内側（`fixed_inner`）、外接矩形全体（`whole_box`）に分けて出す。
- 復号した保持矩形の、切り出しに対する PSNR（`keep_db`）、保持矩形の 4 辺から内側 32 画素の帯（`edge32_db`）、その内側（`interior_db`）、その差（`edge_minus_interior_db`）。
- 符号化と復号の秒数と VRAM の山。復号した 4 本は `G0\<名前>.mp4` に書く（目視用）。

G1〜G4（各本）:

- 保持矩形の PSNR: 出力 mp4 の保持矩形と対照（`control_write_mp4_...`）の PSNR（`keep_psnr_vs_control_db`）、なじませの重みが 1 の芯の画素だけの PSNR（`keep_core_psnr_vs_control_db`）、切り出しそのものとの PSNR（`*_vs_cut_db`）。読み込みはエンジンと同じ `engine_h3.source_tail.read_source_frames`（PyAV）。
- 境界の勾配比（`boundary_gradient_ratio`）: 保持矩形の辺をまたぐ画素差の平均を、その両側 8 画素の近傍の差の平均で割った値。左右の縦の境界と、上下の横の境界を別々に出す（`left`・`right`・`top`・`bottom`・`vertical_max`・`horizontal_max`）。前例 `outputs/ltx25-outpaint-gate/analyze_w2.py` の `boundary_gradient_ratio`（縦の境界だけ）と同じ量に、横の境界を足したもの。関数は import せず台本に書いた。1.0 は近傍と見分けがつかず、大きいほど継ぎ目が立つ。近傍の差が 0 の辺は `null`。
- pad の帯の標準偏差（`pad_band_std`。8 フレームおき）: 平らな帯は絵が出ていない目安。
- エンジンの記録 `h3.outpaint`（固定トークン数・`audio_output`・`inject_clean` など）、復号の段の `audio_vae_source`、所要秒数（`seconds`）、VRAM の山（`peak_vram_mb`・`peak_vram_reserved_mb`）、主メモリの山。
- 固定トークンの自己検査はエンジンの中で行われ、外れると本が失敗として記録されます。

失敗した本はエラーの全文を記録して次の本へ進みます。終了コードは 0（全部完走）、1（失敗した本か門がある）、2（引数・素材・CUDA の誤り）。

## 出力の置き場

すべて `outputs\h3-outpaint-gates-2026-11-10\`（git の追跡外）。

| ファイル | 内容 |
|---|---|
| `_material\` | 素材・対照・`prepare.json`（上の「素材」） |
| `summary.json` | 全体の記録（下の「`summary.json` の鍵」）。本ごとに書き直す |
| `summary.md` | G0 と各本の表 |
| `load.log` | 読み込みのログ |
| `G0\result.json`・`G0\g0.log`・`G0\<名前>.mp4` | G0 の記録・ログ・復号した 4 本 |
| `<本>\output.mp4` | 出力 |
| `<本>\result.json` | `generate()` の戻り値そのまま |
| `<本>\run.log` | その本のエンジンのログと進捗 |

監督は実行後に `summary.json`（と必要なら `G0\result.json`・各本の `result.json`）をこの保管庫へ写します。

## `summary.json` の鍵

| 鍵 | 意味 |
|---|---|
| `script`・`started_at`・`finished_at`・`exit_code`・`args`・`gates` | 台本、時刻、終了コード、受けた引数、回した門 |
| `conditions` | 重み・LoRA と強度・ステップ・shift・種・注意の実装・VRAM 16GB mode・読み込みの値・常駐・フレーム数・プロンプト |
| `material` | 元の素材と切り出しの ffprobe、ffmpeg のコマンド、音声の符号化、開始と終わりのフレーム、s=1.0 での読み込みの結果 |
| `control` | 対照のパスと、対照と切り出しの PSNR |
| `g0` | G0 の記録: `geometry`・`fixed_tokens_in_box`・`latent_shapes`・`latent_rel_l2`・`psnr_vs_cut`・秒数・VRAM の山・`status`・失敗時の `error`／`traceback` |
| `load` | `build_report`（`H3Pipeline.load()` の戻り値）・`seconds`・失敗時の `error` |
| `runs[]` | 本ごとの記録（下の表） |
| `error`・`traceback` | 台本全体が例外か中断で止まったときだけ |

`runs[]` の鍵:

| 鍵 | 意味 |
|---|---|
| `label`・`gate`・`purpose`・`pads`・`canvas`・`r`・`feather_px`・`audio_strength`・`inject_clean` | 本の条件 |
| `geometry` | 外接矩形・トークンの格子・固定／生成トークン数・潜在への貼り位置・潜在フレーム数（`compute_outpaint_geometry` から） |
| `request` | `H3GenerateRequest` の中身 |
| `status`・`error`・`traceback` | `done`／`failed`／`skipped`（読み込み失敗）／`done+measure_failed`（生成は済み・測定で失敗） |
| `outpaint_record`・`loras_record`・`decode_record` | エンジンの記録 `h3.outpaint`・`h3.loras`・復号の段の音声の欄 |
| `seconds`・`wall_seconds`・`peak_vram_mb`・`peak_vram_reserved_mb`・`rss_peak_gib`・`commit_peak_gib` | エンジンの所要秒数、測定を含む壁時計の秒数、VRAM と主メモリの山 |
| `metrics` | 上の「記録するもの」の G1〜G4 の量と、出力の ffprobe |

## 注意（追跡外のもの）

- `outputs\` は丸ごと git の追跡外（`.gitignore` の `outputs/*`）。保管庫の mp4 と `_rt\` も追跡外（`.gitignore` の `Docs/Outputs-archive/**/*.mp4` と `Docs/Outputs-archive/**/_rt/`）。この保管庫で追跡するのは台本・この説明書・監督が写す結果の JSON だけ。
- 判定器の前例 `outputs/ltx25-outpaint-gate/analyze_w2.py` も追跡外です。この台本は同じ量を自分で計算するので、前例が無くても動きます。
- 生成は 6 本です（主条件の 1 本を G1・G2 の r=1・G4 の s=0.5 で共用するため）。実験 E1 の 2 本は別です。
- `summary.json` が既にあれば、台本はそれを読み込んで足します（今回回さない本と G0 の記録は残し、同じ名前の本は今回のもので置き換える。前回までの実行の時刻・引数・読み込みの記録は `previous_invocations` に残す。各本の条件は `runs[].conditions`）。`summary.json` に G0 が無く `G0
esult.json` があれば、それを `g0` として取り込みます。

## 実験 E1（素材の縁の暗い行を落とす）

- 目的: オーナーの目視で、保持矩形の上下の境界に細い線が見えました。元の素材（1080p）と切り出し 1280×720 の上端 1 行と下端 2 行が暗く（左右には無し）、出力の線の位置（y=184 と y=902〜903）がその行と一致したためです。素材の暗い行を落とせば線が消えるかを確かめます。
- 素材（`--prepare` で作る。既存の切り出しは作り直さない）: `_material\src_1280x716_124f_trim2.mp4` は切り出し `src_1280x720_124f.mp4` から ffmpeg の `crop=1280:716:0:2` で上 2 行と下 2 行を落としたもの（libx264 crf 12・yuv420p・24fps CFR、音声はそのまま写す）。ffprobe で 124 枚・1280×716 でなければ止まります。対照は `_material\control_write_mp4_1280x716_124f_trim2.mp4`（trim2 を `write_mp4` で書き直したもの）。`prepare.json` の `material_trim2`・`control_trim2` に記録します。
- 条件: 主条件と同じ 1920×1088 のキャンバス・Turbo 4 ステップ・s=0.5・注入 1.0・種 1234。pad は左 320・右 320・上 186・下 186（716＋372＝1088）。外接矩形と固定トークン数は主条件と同じ（[320,160)-[1600,928)・固定 880）。

| 本（フォルダ名） | なじませ r（px） |
|---|---|
| `E1_trim_r1` | 1（30） |
| `E1_trim_r0` | 0（0） |

- PSNR は trim2 の対照と比べ、勾配比は pad 186 の位置の辺で測ります。出力の置き場と記録の形は他の本と同じ（`<本>\output.mp4`・`result.json`・`run.log`、`summary.json` の `runs[]`）。
- コマンド（リポジトリの根で。cmd）:

```
set PYTHONPATH=%CD%
set PYTHONUTF8=1
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --prepare
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --dry-run --gate E1
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --gate E1
```

- `make_outpaint_sheet.py` は E1 の 2 本も拾い、保持矩形の枠を 1280×716・pad 186 の位置に描きます。

## 実験 E2（汎用。別の元動画で確かめる）

- 目的: 切り落としの代わりに、四辺に線の無い別の元動画で、保持矩形の境界の線が出るかを確かめます（素材は監督が選ぶ）。素材・pad・なじませを引数で与える汎用の門です。
- 素材: `--material` の動画を `--prepare` と同じ作法（select の窓・atrim・libx264 crf 12・yuv420p・24fps CFR・音声は PCM、断られたら AAC 192k）で、24fps・ちょうど `--frames` 枚に切り出します。素材が 24fps でなければ先に 24fps に揃えてから窓を取ります（`cut_window_mp4` と同じ順）。`--scale W:H` を渡すと、縦横比を保って W:H を覆うまで lanczos で縮め、中央で W:H に切り抜きます（W・H は偶数）。渡さなければ、奇数の辺だけ 1 画素を切り落として偶数にします。
  - できるもの: `_material\<label>_<W>x<H>_<F>f.mp4`（切り出し）と `_material\<label>_<W>x<H>_<F>f_control.mp4`（`write_mp4` で書き直した対照）。ffprobe で枚数と寸法を確かめ、合わなければ止まります。`prepare.json` の `e2:<label>` に記録し、同じ引数で既にあれば作り直しません。
  - E2 の素材は `--dry-run` でも GPU の実行でも、無ければその場で作ります（CPU）。`--prepare --gate E2 ...` で素材だけを作ることもできます。
- 条件: キャンバスは素材＋pad（`--canvas` を渡すと一致を確かめる。32 の倍数でなければ止まる）。`--feathers` の r ごとに 1 本、本の名前は `<label>_r<r>`。Turbo 4 ステップ・注入 1.0・種 1234 は他の本と同じ。注意の実装の既定は `sage`（時短。オーナー了承）。音声の Strength は `--audio-strength`（既定 0.5。音声の無い素材では 0 を渡す）。
- 引数:

| 引数 | 既定 | 意味 |
|---|---|---|
| `--material` | （E2 で必須） | 元の動画 |
| `--pads` | （E2 で必須） | pad の 左,右,上,下 |
| `--canvas` | 素材＋pad | キャンバスの 幅,高さ（指定すれば一致を確かめる） |
| `--feathers` | `1,0` | なじませの段数 r の列。1 つにつき 1 本 |
| `--label` | `E2` | 本の名前と素材のファイル名の接頭辞 |
| `--frames` | 124 | フレーム数（17n+5） |
| `--start-sec` | 0 | 窓の開始秒（開始フレームは `round(秒×24)`） |
| `--scale` | なし | `W:H`。覆うまで縮めて中央で切り抜く |
| `--audio-strength` | 0.5 | 音声の Strength s |

- 記録: PSNR はその本の素材の対照と比べ、勾配比はその本の pad の位置で測ります。出力・記録の形は他の本と同じ（`<本>\output.mp4`・`result.json`・`run.log`、`summary.json` の `runs[]` に足す。`runs[].conditions` に注意の実装とフレーム数が入る）。
- コマンド（リポジトリの根で。cmd）:

```
set PYTHONPATH=%CD%
set PYTHONUTF8=1
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --dry-run --gate E2 --material <元の動画> --pads 320,320,184,184
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate.py --gate E2 --material <元の動画> --pads 320,320,184,184
```

- `make_outpaint_sheet.py` は `summary.json` にある E2 の本も拾います。保持矩形の枠（pad と元の寸法）は本ごとの `result.json` の `h3.outpaint` から読み、元の動画の写しを `source_<本>.mp4` として置きます。

## 遠隔目視のフォルダの作り方

門の実行後に、各本の動画・一覧の画像・説明書を 1 つのフォルダにまとめて、Hugging Face（非公開データセット）で外出先から見られるようにします。`make_outpaint_sheet.py` が読んで書くだけで、GPU もサーバーも使いません（アプリの仮想環境 `.venv` の python と、`tools\ffmpeg\bin\ffmpeg.exe`、Pillow を使います）。

```
.venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\make_outpaint_sheet.py --hf-dir outputs\h3-outpaint-gates-2026-11-10\hf-round13
```

- 読むもの: `summary.json`（`runs[]` の条件・`metrics`・`seconds`・`peak_vram_mb`、`conditions.seed`。無ければ各本の `result.json` と台本と同じ既定の条件）、`<本>\output.mp4`、`G0\<名前>.mp4`・`G0\result.json`、`_material\src_1280x720_124f.mp4`。
- 書くもの: 各本の mp4（`<本>_<幅>x<高さ>_r<なじませ段数>_s<音声Strength>_seed<seed>.mp4`）、`source_1280x720_124f.mp4`、`G0_<名前>.mp4`、`sheet_all.png`・`sheet_<本>.png`・`sheet_G0.png`（先頭・中央・末尾の 3 コマ。保持矩形に 1 画素のマゼンタの枠）、`README.md`（条件の表と数値の表）。動画の無い本は飛ばし、表に「未実行」と書きます。何度回してもよく、既存のファイルは消しません。
- 引数: `--no-sheets`（画像を作らない）、`--out-root`（入力の根の差し替え。試走用）、`--hf-dir`（書き先。既定は `<out-root>\hf-round13`）。
- 上げるのは前例と同じ書式です（`.venv\Scripts\python.exe -m huggingface_hub.cli.hf upload Rootport/Nz-LTX23-eval-clips <フォルダ> round13-h3-outpaint --repo-type dataset`）。上げる前にオーナーの了承を得ます。
