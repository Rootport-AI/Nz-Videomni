# MiniMax H3 の Retake の GPU の門（MCP 経由の台本）

`h3_retake_gate.py` は、MiniMax H3 の Retake（入力動画の指定した区間だけを描き直す機能）の GPU の門を、MCP サーバー経由で回す台本です（台帳 `PENDING_TASKS.md` §1-123。門の設計と結果は記録簿 `Docs/VERIFICATION_LOG.md` §200.7、仕組みは設計正本 `Docs/MINIMAX_H3_DESIGN.md` §3.4）。この保管庫には台本・説明書・各条件の `request.json`／`metadata.json`／`summary` の写しを置き、動画・素材・ログは写しません。この台本は製品コードを import しません。MCP サーバー（`.venv\Scripts\python.exe -m mcp_server`）を stdio の子プロセスとして起こし、実際のプロトコル（`initialize → tools/list → tools/call`）で道具を呼びます。経路は記録簿 `Docs/VERIFICATION_LOG.md` §39.6 と同じです。結果の数値はここには書きません（数値の正本は記録簿 §200）。

## 前提

- バックエンドが起動済みで、`GET /api/v1/status` が 200 を返すこと（起動の手順と前処理〔AviUtl2 を閉じる・GPU の使用量が約 2,000MiB 未満・孤児の `engine.worker` が無い〕は計画 §7 の 3）。この台本はバックエンドを起動も停止もしません。
- 台本を動かす python は、アプリの仮想環境 `Nz-Videomni\.venv`（`mcp==1.28.1` が入っている）。MCP サーバーも既定でこの仮想環境の python で起こします。
- 標準出力は UTF-8 です。`PYTHONUTF8=1` を設定して起動してください（台本の中でも標準出力を UTF-8 に直しますが、子プロセスの MCP サーバーにも同じ値を渡すためです）。
- `--material` のパスと保存先 `--out-dir` は、**MCP サーバーが動くマシン上のパス**です（`upload_video` の `file_path` と `save_job_video` の `dest_dir` は MCP サーバーの側で開かれます）。この台本は MCP サーバーを同じマシンの子プロセスとして起こすので、手元のパスがそのまま使えます。
- `metadata.json` と `_retake_window.mp4` は、`get_job_video_path` が返す `outputs\<job_id>\output.mp4` と同じフォルダから読みます（MCP サーバーの `config.yaml` の出力先。`LTX_MCP_OUTPUT_DIR` で上書きされていればそちら）。
- ffmpeg と ffprobe は `<repo>\tools\ffmpeg\bin\` のものを使います（無ければ PATH）。

## 起動のしかた

Windows PowerShell 5.1 から（パスに空白が入りうるので引用符をつけます）:

```
$env:PYTHONUTF8 = "1"
& "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\.venv\Scripts\python.exe" "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\Docs\Outputs-archive\h3-retake-2026-11-08\h3_retake_gate.py" --material "S:\path\to\talking.mp4" --start-sec 2.0
```

送る引数だけを確かめる（MCP サーバーもバックエンドも使わない。素材の縦横は ffprobe で読む）:

```
& "...\.venv\Scripts\python.exe" "...\h3-retake-2026-11-08\h3_retake_gate.py" --material "S:\path\to\talking.mp4" --start-sec 2.0 --dry-run
```

計画 §7 の追加の 2 本と、長い窓の Strength 0.5 の 1 本の、実際に回した形（`--material` と `--out-dir` は一次資料の置き場の例。`<主条件と同じプロンプト>` は主条件の `s0\request.json` の `prompt`）:

```
# 音声の無い素材で s=0（audio_fix=None の経路。プロンプトは台本の既定）
... --material "S:\path\to\silent.mp4" --start-sec 1.0 --strengths 0 --label silent-s0 --out-dir "...\outputs\h3-retake-gates-2026-11-08\extra-silent" --skip-load
# 長い窓（481 フレーム・解像度は小さめ。素材は主条件の 15 秒を 2 回つないだ 30 秒）
... --material "...\outputs\h3-retake-gates-2026-11-08\_material\H3-Ref-1080p-30s.mp4" --start-sec 2.0 --frames 481 --width 640 --height 384 --prompt "<主条件と同じプロンプト>" --strengths 0 --label long481-s0 --out-dir "...\outputs\h3-retake-gates-2026-11-08\extra-long" --skip-load
# 長い窓を Strength 0.5 で（--skip-load は付けず、台本が H3 を読み込んだ）
... --material "...\outputs\h3-retake-gates-2026-11-08\_material\H3-Ref-1080p-30s.mp4" --start-sec 2.0 --frames 481 --width 640 --height 384 --prompt "<主条件と同じプロンプト>" --strengths 0.5 --label long481-s0.5 --out-dir "...\outputs\h3-retake-gates-2026-11-08\extra-long-s0.5"
```

## 引数

| 引数 | 既定 | 意味 |
|---|---|---|
| `--material` | （必須） | 話し声のある素材の mp4 |
| `--start-sec` | （必須） | 窓の開始秒（0 以上）。窓の開始フレームは製品と同じく `round(開始秒×24)` |
| `--frames` | 124 | 窓のフレーム数（17n+5・73〜481）。`clips[0].num_frames` として送る |
| `--width`・`--height` | 下の「解像度の既定」 | 32 の倍数。両方指定するか両方省略する |
| `--seed` | 1234 | 種 |
| `--prompt` | 話している人の短い英文 | 共通のプロンプト。クリップ側の `prompt` は送らない（送ると共通の文の後ろに追記されるため） |
| `--strengths` | `0,0.5,1.0` | `retake_audio_strength` の並び（カンマ区切り・0〜1）。1 つにつき 1 条件 |
| `--base-url` | `LTX_MCP_BASE_URL`、無ければ `http://127.0.0.1:18620` | バックエンドの URL。MCP サーバーへ `LTX_MCP_BASE_URL` として渡す |
| `--transformer` | `default` | `load_pipeline` の `models={"transformer": ...}`。`default` は FL2VA の重み |
| `--attention` | `sdpa` | `attention_backend`（`sdpa`／`sage`） |
| `--out-dir` | `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-retake-gates-2026-11-08` | 記録の置き場（git 管理外） |
| `--label` | `s<strength>`（例 `s0`・`s0.5`・`s1.0`） | 条件の名前＝保存先のフォルダ名。strength が 1 つならそのまま、複数なら `<label>-s<strength>` |
| `--skip-load` | 切 | `load_pipeline` を呼ばない。H3 が載っていなければ止まる |
| `--dry-run` | 切 | 送る引数を表示して終わる |
| `--load-timeout-sec` | 900 | 読み込みを待つ上限（ベースモデルの切り替えは数分かかる） |
| `--job-timeout-sec` | 3600 | 1 条件のジョブを待つ上限。超えたら残りの条件は送らない（ジョブはバックエンドで続くため、送っても JOB_BUSY になる） |
| `--repo` | `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni` | MCP サーバーの作業フォルダ |
| `--mcp-python` | `<repo>\.venv\Scripts\python.exe` | MCP サーバーを起こす python |

### 解像度の既定

素材の縦横を ffprobe で読み、**長辺を 1280 にして、短辺を縦横比から求めて 32 の倍数へ切り上げます**。16:9 の素材なら 1280×736、縦長の 9:16 なら 736×1280 です。切り上げにするのは、製品が入力動画を「縦横比を保って画面を覆うまで拡大し、中央を切り抜く」形で取り込むためで、切り上げなら削られるのは長辺側の端の数画素だけになります。1280×736・124 フレームは快適上限（快適上限表 §1.1）の内側です。1920×1088 などで回すときは `--width`／`--height` で指定してください。

## 流れ

1. MCP サーバーを起こし、`tools/list` の `inputSchema` で、台本が渡す引数名がすべてあるかを照合します（`load_pipeline`：`base_model`・`models`・`wait_sec`／`upload_video`：`file_path`・`max_frames`／`submit_chain`：`prompt`・`clips`・`width`・`height`・`frame_rate`・`seed`・`attention_backend`・`retake_video_id`・`retake_window_start_sec`・`retake_audio_strength`／`wait_for_job`：`job_id`・`timeout_sec`・`poll_interval_sec`／`save_job_video`：`job_id`・`dest_dir`・`no_clobber`・`which`／`get_job_video_path`：`job_id`）。1 つでも無ければ終了コード 1 で止まります。
2. `backend_status` で到達を確かめます。
3. `load_pipeline(base_model="MiniMaxH3", models={"transformer": <--transformer>}, wait_sec=45)` を 1 回呼び、応答待ちが時間切れ（`finished: false`）なら、`backend_status` を 10 秒おきに見て `pipeline_loaded` が真・`base_model` が `MiniMaxH3`・`state` が `ready` になるまで待ちます。`load_pipeline` を繰り返し呼ばないのは、読み込み中に呼ぶと 409 PIPELINE_LOADING で断られるためです。読み込み後に `list_models` の `transformer_role`（`fl2va`／`ref2va`）を記録します（止めはしません）。
4. `upload_video(file_path=<素材>, max_frames=100000)` を 1 回だけ呼びます。`max_frames` を渡したときだけ `frame_count` と `fps` が実測されて返るので、尺を測るためだけに十分大きな値を渡しています（小さい値だと素材が切り詰められて保存されます）。
5. strength ごとに `submit_chain` を送ります。H3 では `retake_head_px`／`retake_tail_px` を渡しません（のりしろは 22 固定でサーバーが決める。22 以外は 422）。`retake_regenerate_audio` は道具の既定（真）のままで、H3 は真を断りません。`frame_rate` は 24、`clips` は 1 本です。そのあと `wait_for_job(timeout_sec=45, poll_interval_sec=5)` を繰り返します（`timed_out: true` は失敗ではなく、同じ引数でもう一度呼ぶ）。
6. 完了したら `save_job_video(job_id, dest_dir=<out-dir>\<label>, no_clobber=True, which="output")` で保存し、`outputs\<job_id>\metadata.json` を `<out-dir>\<label>\metadata.json` に写します。
7. 記録だけ取ります（判定は付けません）。`metadata.retake` の鍵、出力のフレーム数（ffprobe の `-count_frames`）、出力の頭 22 フレームと尾 22 フレームの、窓の切り出しとの PSNR。比べる相手はジョブのフォルダの `_retake_window.mp4`（製品が窓を切り出したもの）で、無ければ台本が素材から同じ規則（24fps に揃えてから `round(開始秒×24)` 番目から `--frames` 枚）で `<out-dir>\_window_ref\window_from_material.mp4` に切り出します。参照は製品の取り込みと同じ形（覆うまで拡大→中央を切り抜き・LANCZOS）に縮めてから比べます。製品の拡大は切り上げ、ffmpeg は四捨五入なので、1 画素ずれることがあります。
8. 条件が 1 つ終わるたびに `<out-dir>\summary.json` と `<out-dir>\summary.md` を書き直します。

失敗の扱い: ジョブが `failed`／`cancelled` なら `error` を記録して次の条件へ進みます。`submit_chain` が断られたとき（422 など）も、エラーの全文を記録して次へ進みます。道具のエラーと転送の失敗は、すべて全文が `summary.json` の `mcp.calls` に残ります。

## できるもの

| ファイル | 内容 |
|---|---|
| `<out-dir>\summary.json` | 全体の記録（下の「JSON の欄」） |
| `<out-dir>\summary.md` | 条件ごとの表 |
| `<out-dir>\mcp_server.stderr.log` | MCP サーバーの標準エラー（追記） |
| `<out-dir>\<label>\<job_id>.mp4` | 出力（`save_job_video` の既定のファイル名） |
| `<out-dir>\<label>\metadata.json` | ジョブの `metadata.json` の写し |
| `<out-dir>\<label>\request.json` | `submit_chain` に渡した引数（`arguments`）。サーバーが受けた本文（書き戻し後）は `metadata.json` の `request` |
| `<out-dir>\_window_ref\window_from_material.mp4` | `_retake_window.mp4` が無かったときだけ作る参照 |

## JSON の欄（`summary.json`）

| 欄 | 意味 |
|---|---|
| `script`・`started_at`・`finished_at`・`exit_code`・`args` | 台本の名前、開始と終了の時刻、終了コード、受けた引数 |
| `material` | 素材のパスと ffprobe の結果（縦横・fps・音声の有無と長さ） |
| `geometry` | 窓の開始秒と開始フレーム、フレーム数、幅と高さとその決め方（`size_source`）、種、注意の計算、fps、のりしろの幅。アップロード後は素材の尺（`upload_seconds`）と窓の終わり（`window_end_sec`） |
| `warnings` | 送る前に気づいた注意（32 の倍数でない・17n+5 でない・素材に音声が無い）。止めはしない |
| `mcp` | MCP サーバーの起こし方（`python`・`cwd`・`LTX_MCP_BASE_URL`・`stderr_log`）、`server_info`、`calls`（全呼び出しの道具名・引数・秒・成否・エラー全文。`wait_for_job` と `backend_status` は応答の本体を省く） |
| `tool_schema_check` | 引数名の照合の結果。`ok`・`problems`（止まった理由）・`warnings`・`submit_chain_properties`（`submit_chain` の引数名の一覧）・`submit_chain_required` |
| `status_before` | 最初の `backend_status` |
| `load` | `requested`（送った引数）、`responses`（`load_pipeline` の応答かエラー）、`status_polls`・`last_status`・`seconds`、止まったときの `error`。`--skip-load` なら `skipped` |
| `models_after_load` | `list_models` の MiniMaxH3 の項目、`transformer_role`・`transformer_weight_class` |
| `upload` | `upload_video` の応答（`video_id`・`frame_count`・`fps`・`stored_path` など） |
| `conditions[]` | 条件ごとの記録（下の表） |
| `error`・`traceback` | 台本全体が例外か中断で止まったときだけ |

`conditions[]` の欄:

| 欄 | 意味 |
|---|---|
| `label`・`strength`・`dir`・`started_at`・`finished_at` | 条件の名前、`retake_audio_strength`、保存先、時刻 |
| `tool_args` | `submit_chain` に渡した引数（`request.json` と同じ） |
| `status` | ジョブの最後の状態（`completed`／`failed`／`cancelled`）。台本の側の失敗は `submit_error`（送信が断られた）・`wait_gave_up`（待ちの上限か転送の失敗が続いた）・`script_error` |
| `error` | 失敗の全文（ジョブの `error`、道具のエラー、例外） |
| `submit_response`・`job_id` | `submit_chain` の応答とジョブ番号 |
| `wait` | `calls`（`wait_for_job` を呼んだ回数）・`seconds`・`samples`（状態・進捗・段・ステップが変わった時点）・`gave_up` |
| `job_final` | 最後の `wait_for_job` の応答（`JobResponse` の欄一式） |
| `saved`・`job_video_path` | `save_job_video` と `get_job_video_path` の応答 |
| `metadata_path`・`metadata_missing` | 写した `metadata.json` の元のパス、無かったときの説明 |
| `metadata_summary` | `metadata.json` の `backend`・`attention_used`・`vram16_mode_used`・`generation_time_seconds`・`seed`・`models` |
| `metadata_retake` | `metadata.json` の `retake` そのまま |
| `retake_picks` | `metadata.retake` から表に出す鍵（`audio_strength`・`muxed_original_waveform`・`decoded_frames_px`・`source_had_audio`・`audio_frozen`・`n_head_v`・`n_tail_v`・`n_head_a`・`n_tail_a`・`window_start_frame`・`written_frames`・`upload_has_audio`・`source_fps`・`resampled`）。無い鍵は `null` |
| `output_probe` | 保存した出力の ffprobe（`video.nb_read_frames` が復号して数えたフレーム数） |
| `window_ref`・`window_ref_probe`・`window_fallback` | PSNR の参照の種類（`job_retake_window`＝製品の切り出し／`material_cut_by_script`＝台本の切り出し／`none`）とパス、その ffprobe、台本が切り出したときのコマンドと結果 |
| `psnr` | `band_frames`（22）、`output_frames`・`ref_frames`（比べた両者のフレーム数）、`head`・`tail`。それぞれ `out_frames`・`ref_frames`（比べた区間。[始, 終)。尾は各自の最後の 22 枚）、`frames_compared`、`per_frame_psnr_avg`（フレームごとの平均 PSNR。一致したフレームは無限大）、`summary`（ffmpeg の y・u・v・average・min・max）、`min_psnr_avg`、`identical_frames`。失敗したときは `error` |

`summary.md` の表の列は `conditions[]` の欄から取ったもので、「頭 PSNR 平均／最小」は `psnr.head.summary.avg` と `psnr.head.min_psnr_avg` です。

## 終了コード

0＝全条件のジョブが `completed`。3＝どれかの条件が失敗した（ほかの条件は回した）。1＝準備の段で止まった（素材が無い・道具の引数名が違う・バックエンドに届かない・読み込みの失敗・アップロードの失敗・例外か中断）。

## 注意

- **途中で取り消せません。** 台本を Ctrl+C で止めても、送ったジョブはバックエンドで走り続けます（連結のジョブには協調的な停止が無い）。
- 台本は後片付け（`unload_pipeline`・バックエンドの停止）をしません。計画 §7 の「終了」の手順で別に行います。
- 1 条件の待ちが上限を超えたら、残りの条件は送らずに終わります（同時 1 ジョブのため）。
- `no_clobber=True` なので、同じ `--out-dir` と `--label` で回し直すと、出力は `<job_id>.mp4` の別名（ジョブ番号が違うので通常は衝突しない）、`metadata.json` と `request.json` は上書きです。回し直すときは `--label` を変えるか、別の `--out-dir` にしてください。

## この保管庫の中身

2026-11-08 の GPU の門の一次資料（`outputs/h3-retake-gates-2026-11-08/`。git 管理外）からの写しです。結果の数値はここには書きません（数値の正本は記録簿 §200.7）。

| ファイル | 内容 |
|---|---|
| `h3_retake_gate.py` | 門を回した台本 |
| `README.md` | この説明書 |
| `s0\request.json`・`s0\metadata.json` | 主条件・Strength 0 の、`submit_chain` に渡した引数とジョブの `metadata.json` |
| `s0.5\request.json`・`s0.5\metadata.json` | 主条件・Strength 0.5 の同じもの |
| `s1.0\request.json`・`s1.0\metadata.json` | 主条件・Strength 1.0 の同じもの |
| `silent-s0\request.json`・`silent-s0\metadata.json` | 音声の無い素材・Strength 0 の同じもの |
| `long481-s0\request.json`・`long481-s0\metadata.json` | 長い窓（481 フレーム）・Strength 0 の同じもの |
| `long481-s0.5\request.json`・`long481-s0.5\metadata.json` | 長い窓（481 フレーム）・Strength 0.5 の同じもの（追加の門。記録簿 §200.9） |
| `summary_main.json`・`summary_main.md` | 主条件の 3 本の `summary.json` と `summary.md` |
| `summary_silent.json`・`summary_silent.md` | 音声の無い素材の 1 本の同じもの |
| `summary_long.json`・`summary_long.md` | 長い窓の 1 本の同じもの |
| `summary_long_s0.5.json`・`summary_long_s0.5.md` | 長い窓・Strength 0.5 の 1 本の同じもの |
