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

## 弾 2 の GPU の門 G5・G6（API 経由の台本 `h3_outpaint_gate_api.py`）

`h3_outpaint_gate_api.py` は、弾 2（API・アプリ層・配信・操作パネル・MCP）の後の GPU の門 G5（快適上限の点）と G6（20 ステップの目視用）を、**MCP サーバー → バックエンド → エンジン**の経路で回す台本です。門の設計は実装計画（`rosy-jingling-stonebraker.md` の §2.6・§3「弾 2」・§7）です。製品コードは import も変更もしません。部品は前例から写しました。MCP の stdio クライアントは Retake の門の台本（`../h3-retake-2026-11-08/h3_retake_gate.py`。経路は記録簿 §39.6 と同じ）、別ポートの一時サーバー（`_rt`）は VSF の較正の台本（`../h3-vsf-calib-2026-11-09/h3_vsf_calib.py`）、保持矩形の PSNR と境界の勾配比は弾 1 の `h3_outpaint_gate.py` の `measure_run` です（G6 だけ、エンジンの仮想環境の子プロセスで呼びます）。退避の測り方（WDDM の共有メモリの標本と、server.log による窓の区切り）は較正台 `outputs/comfort-calib-h3-chain-2026-11-08/` の `vram_sampler2.ps1` と `calib.py` を写しました。数値はここに書きません（数値の正本は記録簿 `Docs/VERIFICATION_LOG.md`）。

### 実際に回した本と結果の置き場（2026-11-10）

- 回したのは `G5_56f`（規則 v3″ の基準点）・`G5_362f`・`G5_379f`・`G6_20steps` の 4 本です（17:57〜18:59。1 本ごとに新しいサーバーのプロセス）。結果は記録簿 §205.19（G5）・§205.20（G6）にあります。
- **G5 の点を 345／362 から 362／379 に変えました。** 下のコマンドや表の例の 345・362 は、快適上限表 §17.4 の較正（FL2VA・K=1）の c と U です。Outpainting には条件行が無く、text にもキーフレームの行が入らないので、同じ枚数でも S が小さくなります。そこで、S が配信の線 `single_rows_budget` と同じ値になる 362 を線の上の点に、その 1 段上の 379 を線を越えた点にしました（快適上限表 §20.3・§20.4）。台本の `--prepare-frames` の既定（`345,362`）は変えていません。345 枚の素材は作りましたが、回していません。
- **379 枚の判定**: 台本は「判定なし（ジョブが failed）」と出し、終了コードは 3 でした。サーバーが失敗を書いた直後に取った状態で `error` がまだ null だったためで（記録簿 §205.23）、記録簿では規則 ① の OOM と判定しています。
- **この保管庫に写したもの**: `summary_api.json`・`summary_api.md`・`run_G5_G6.console.log`（根）と、`runs_api/<本>/`（`summary_row.json`・`request.json`・`server_log_excerpt.txt`・`worker_log_excerpt.txt`、完走した本の `metadata.json`、G5 の `wddm.csv`、G6 の `measure_request.json`・`measure_result.json`）。動画・`_rt\`・`logs_api\` は写していません（git の追跡外の `outputs\h3-outpaint-gates-2026-11-10\` に残る）。
- G6 の動画・元の素材・一覧の画像は、遠隔目視のフォルダ（`round13-h3-outpaint/`）に足しました（記録簿 §205.20）。

### 前提

- GPU を使う実行は、毎回オーナーの了承を得てから行います。`--prepare` と `--dry-run` は GPU もサーバーも使いません。
- 台本を動かす python はアプリの仮想環境 `.venv`（`mcp`・`yaml` が入っている）。`PYTHONUTF8=1` を設定します。G6 の測定の子プロセスだけ、エンジンの仮想環境 `.venv-engine-h3` の python を使います（CPU だけ。`CUDA_VISIBLE_DEVICES=-1`）。
- 台本は自分で一時サーバーを起こします。`config.yaml` は読むだけで、`server.port`（既定 18699）・`server.log_dir`・`output.dir`・`upload.dir`・`state_file` の 5 つの鍵を `outputs\h3-outpaint-gates-2026-11-10\_rt\` に向けた一時 yaml（`config.h3outpaintapi.yaml`）と、MiniMax H3 の 4 カテゴリを `default` にした一時 state（`state.h3outpaintapi.json`）を書きます。オーナーの `config.yaml`・`state.json`・`logs\`・`outputs\<ジョブ>`・`uploads\` は書き換えません。
- MCP サーバーには `LTX_MCP_BASE_URL=http://127.0.0.1:<ポート>`・`LTX_MCP_CONFIG=<一時 yaml>`・`LTX_MCP_OUTPUT_DIR=_rt\outputs` を渡します（`get_job_video_path` が一時の出力先を指すように）。
- 前処理（AviUtl2 の扱い・GPU の使用量・孤児の `engine.worker` が無いこと）は監督が確かめます。G5 の線の較正（快適上限表 §17.1）は AviUtl2 と操作パネルを開いたまま測ったので、cap（VRAM 16GB mode の割り当て器の上限）が同居プロセスで動きます。台本は cap を `metadata.json` から記録するだけです。

### コマンド

リポジトリの根で（cmd）:

```
set PYTHONUTF8=1
.venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate_api.py --prepare
.venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate_api.py --gate G5 --frames 345 --dry-run
.venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate_api.py --gate G5 --frames 362 --dry-run
.venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate_api.py --gate G6 --dry-run
.venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate_api.py --gate G5 --frames 345
.venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate_api.py --gate G5 --frames 362
.venv\Scripts\python.exe Docs\Outputs-archive\h3-outpaint-2026-11-10\h3_outpaint_gate_api.py --gate G6
```

1 回の起動で 1 本を回し、既定では終わりにパイプラインを降ろしてサーバーをプロセスの木ごと止めます。G5 の 2 本はそれぞれ新しいサーバーのプロセスで回ることになり、前の本の VRAM の断片化を持ち越しません。続けて回したいときは `--keep-server`（次の回で同じポートのサーバーを使い回す）と `--skip-load`（`load_pipeline` を呼ばず、H3 が載っていることだけ確かめる）を使います。

### 引数

| 引数 | 既定 | 意味 |
|---|---|---|
| `--gate` | （必須。`--prepare` だけなら省略可） | `G5` か `G6` |
| `--frames` | （G5 で必須） | G5 のフレーム数（17n+5）。G6 は 124 固定 |
| `--prepare` | 切 | 素材を作って終わる（ffmpeg。CPU） |
| `--prepare-frames` | `345,362` | `--prepare` で作る G5 の枚数（カンマ区切り）。`--gate G5 --frames N --prepare` なら N も足す |
| `--dry-run` | 切 | サーバーを起こさず、素材の有無・一時サーバーの起こし方・MCP の環境・送る引数・送らない引数・S の予測を表示して終わる |
| `--skip-load` | 切 | `load_pipeline` を呼ばない |
| `--keep-server` | 切 | 終わりに `unload_pipeline` もサーバーの停止もしない |
| `--port` | 18699 | 一時サーバーのポート |
| `--label` | `G5_<枚数>f`／`G6_20steps` | 本の名前＝保存先のフォルダ名 |
| `--prompt` | G5 は較正の文、G6 は弾 1 の文 | プロンプトの差し替え |
| `--idle-sec` | 15 | G5: 標本を始めてから送るまでの静かな区間（記録だけ） |
| `--load-timeout-sec` | 1200 | H3 の読み込みを待つ上限 |
| `--job-timeout-sec` | 7200 | 1 本を待つ上限 |

### 素材（`--prepare`。`outputs\h3-outpaint-gates-2026-11-10\_material\`）

| ファイル | 内容 |
|---|---|
| `G5_src_1280x720_345f.mp4`・`G5_src_1280x720_362f.mp4` | `outputs\h3-retake-gates-2026-11-08\_material\H3-Ref-1080p-30s.mp4`（1920×1080・24fps・720 枚・AAC 44.1kHz）の先頭から 345 枚と 362 枚を 1280×720 へ縮めたもの。作法は弾 1 の素材と同じ（select の窓・atrim・lanczos・libx264 crf 12・yuv420p・24fps CFR・音声は PCM、断られたら AAC 192k）。ffprobe で枚数・寸法・音声を確かめ、合わなければ止まる |
| `G6_src_1344x768_124f.mp4` | `outputs\h3-vsf-calib-2026-11-09\hf-round12\L_s0_1344x768_seed42.mp4`（1344×768・24fps・124 枚・AAC 32kHz・四辺に線なし）の写し（再符号化しない）。枚数・寸法・fps・音声を確かめる |
| `prepare_api.json` | ffmpeg のコマンドと ffprobe の結果 |

既にある素材は作り直しません。元の 30 秒の素材は 15 秒を 2 回つないだものなので、362 枚の切り出しはつなぎ目（360 枚目）をまたぎ、音声が映像より約 0.02 秒短くなります。Strength 0.5 は読めた分だけ使う規則（`engine_h3.outpaint_h3.read_outpaint_source`）なので止まりません（1.0 なら全長が要るので止まります）。サーバーは受け取った動画から、さらに先頭 `num_frames` 枚を `_outpaint_source.mp4` へ切り出し直します（`services/video_io.cut_window_mp4`）。

### 各門の条件

共通: キャンバス 1920×1088・音声の Strength 0.5・なじませは送らない（サーバーの既定）・Turbo LoRA なし・`frame_rate` 24。`vram16_mode`・`keep_resident`・`block_swap_prefetch`・`shift_*` は送らず、サーバーの既定（on・off・on・12／3）に任せる。元動画の ID は `reference_video_id` で渡す（H3 では MCP が `outpaint.source_video_id` へ移す）。

| 本 | 素材 | pad 左／右／上／下 | フレーム | ステップ | 注意の実装 | 種 | プロンプト |
|---|---|---|---|---|---|---|---|
| `G5_56f`・`G5_362f`・`G5_379f`（実際に回した本。上の「実際に回した本」） | `G5_src_1280x720_<F>f.mp4` | 320／320／184／184 | 56・362・379 | 4 | sage | 12345 | 較正の文（`outputs\h3-vram16-gates\f0_prompt.txt`。text 30 行） |
| `G6_20steps` | `G6_src_1344x768_124f.mp4` | 288／288／160／160 | 124 | 20 | sage | 1234 | 弾 1 の既定の文（同じ素材・同じ pad・同じ種の `E2c_r1` と比べられるように） |

G5 の条件は快適上限表 §17.1 に合わせました（FL2VA の重み・VRAM 16GB mode on・先読み on・`keep_resident` off・LoRA なし・sage・種 12345・同じ文）。ステップ数は §17.1 の走査と同じく 4 です（VRAM の山はステップ数に依らない）。`--frames 56` で回すと規則 v3″ の基準点（較正の `b1080` と同じ長さ）になり、② 退避の判定の比べる相手として使われます（素材は `--prepare --prepare-frames 56` で作る）。

### 記録するもの

- 全部の本: `submit_generate` に渡した引数（`request.json`）、`metadata.json` の写し、ジョブの成否・`generation_time_seconds`・壁時計の秒、`h3.phases.denoise` の `peak_vram_mb`・`peak_vram_reserved_mb`・`sequence`（`sequence.total`＝S）・`vram16`（`cap_mib`・`num_alloc_retries`・`num_ooms`・`inactive_split_peak_mib` ほか）・`block_swap_prefetch_used`、`h3.outpaint` とその 12 鍵の照合（`outpaint_keys_check`。欠けた鍵・余分な鍵）、サーバーとワーカーのログの切れ端。
- G5: S の予測（快適上限表 §17.3 の式。Outpainting は cond 0・キーフレームなし・参照なしで、固定行は行数を増やさない）と線 219,516（`single_rows_budget`）との差、ジョブの間の WDDM の共有メモリの標本（`wddm.csv`）、server.log で区切った `encode`・`denoise`・`decode` の窓ごとの共有メモリの中央値と山、送る前の静かな区間の値。
- G5 の判定（`judgement`）は快適上限表 §17.2 の規則 v3″ の写しです。快適＝ジョブが `completed` で、① OOM（`job.error` が `GPU_OOM:` で始まる。段は最初のトレースバックから `denoise`／`encode`／`decode`／`other`）、② 退避（`denoise` と `decode` の窓で、標本が 10 以上あり、共有メモリの中央値が基準点〔G5 の 56 枚〕の同じ窓より +300MB 以上）、③ 先読みの降格（`block_swap_prefetch_used` が on でない）のどれにも当たらない。基準点が無いか標本が足りない窓は「② は未判定」と書きます。
- G6: 保持矩形の PSNR（対照＝サーバーの切り出し `_outpaint_source.mp4` を `engine_h3.media_out.write_mp4` で書き直したもの、と切り出しそのもの）・なじませの芯の PSNR・境界の勾配比（左右上下）・pad の帯の標準偏差。量の定義は上の「記録するもの」の G1〜G4 と同じで、関数も同じ（弾 1 の `measure_run`）です。

### 出力の置き場

すべて `outputs\h3-outpaint-gates-2026-11-10\`（git の追跡外）。

| ファイル | 内容 |
|---|---|
| `<本>\request.json` | `submit_generate` に渡した引数 |
| `<本>\metadata.json` | ジョブの `metadata.json` の写し（失敗したジョブには無い） |
| `<本>\summary_row.json` | その本の記録（`summary_api.json` の `runs[]` の 1 行と同じ） |
| `<本>\<job_id>.mp4` | 出力（`save_job_video` の既定のファイル名） |
| `<本>\_outpaint_source.mp4`・`_control_write_mp4.mp4` | サーバーの切り出しの写しと、G6 の対照 |
| `<本>\wddm.csv`・`worker_log_excerpt.txt`・`server_log_excerpt.txt` | G5 の共有メモリの標本、ログの切れ端 |
| `<本>\measure_request.json`・`measure_result.json` | G6 の測定の子プロセスへの依頼と結果 |
| `summary_api.json`・`summary_api.md` | 全体の記録（同じ名前の本は今回のもので置き換える。起動ごとの記録は `invocations[]`）と表 |
| `logs_api\` | MCP の呼び出しの控え（起動ごと）と MCP サーバーの標準エラー |
| `_rt\` | 一時サーバーの設定・状態・ログ・出力・アップロード |

監督は実行後に、`summary_api.json`・`summary_api.md`・`run_G5_G6.console.log` と各本の記録（`request.json`・`metadata.json`・`summary_row.json`・ログの切れ端・G5 の `wddm.csv`・G6 の測定の依頼と結果）を、この保管庫の根と `runs_api/<本>/` へ写します（mp4・`_rt\`・`logs_api\` は写さない）。

### 終了コード

0＝ジョブが `completed`（G5 で OOM になったときも、門の記録として 0）。3＝OOM 以外の失敗・送信の拒否・アップロードの失敗・待ちきれない・測定の失敗。1＝準備の段で止まった（素材が無い・サーバーが起きない・道具の引数名が違う・バックエンドに届かない・読み込みの失敗・例外か中断）。
