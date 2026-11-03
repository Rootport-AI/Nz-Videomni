> **このファイルは実機の `outputs/comfort-calib-h3-ref-2026-11-03/README.md`（git追跡外）のスナップショットである（2026-11-03複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

## この複写について（2026-11-03 追記）

台帳 §1-122（MiniMax H3 の Reference つきの快適上限の較正）の一次記録のうち、文字の記録だけをここへ写しました。原本は git 管理外の `outputs/comfort-calib-h3-ref-2026-11-03/` にあります。結論と数値は記録簿 [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §194（Docs の中での数値の正本）、表と結論は快適上限表 [`../../COMFORT_LIMIT_TABLE.md`](../../COMFORT_LIMIT_TABLE.md) §18 を読んでください。

**写したもの**（`.md` と `.txt` の先頭には複写の断り書きを 1 行足しました。`kind_weights.py` と `mem_history/harness_copy.diff` は中身のままです）:

| ファイル | 中身 |
|---|---|
| `README.md` | 較正台の使い方（本ファイル。下の区切り線より後が元の README） |
| `RESULTS.md` | 数値の一次記録（全点の表・軸ごとと種類ごとの帯・OOM の内訳・仕組みの要約・留保・工程） |
| `MECHANISM_ANALYSIS.md` | コードと既存の記録から読んだ仕組みの考察（式・行の種類ごとの張量・OOM の場所） |
| `progress.md` | 点ごとの時系列 |
| `report_batchA.txt`・`report_batchB.txt`・`report_batchC.txt` | 担当の報告の表 |
| `harness_diff.txt` | 前回の較正台からの差分 |
| `git_state_start.txt` | 開始時の HEAD・未コミットの変更の数・GPU の使用量 |
| `check_results_output.txt` | `RESULTS.md` の表と `run.json` を機械で突き合わせた出力 |
| `kind_weights.py` | 種類ごとの帯の計算 |
| `mem_history/MEM_HISTORY_REPORT.md` | メモリ確保の履歴の測定（判定・セグメントの表・時系列） |
| `mem_history/harness_copy.diff` | 計測台の写しと原本の差分（OOM の瞬間のスナップショットを書く observer ほか） |
| `mem_history/analysis/r1080_277_oom.md`・`r1080_277_step1.md`・`r1080_277_boundary.md` | 判定に使った r1080_277 の集計（OOM の瞬間・ステップの境目）と時系列 |
| `mem_history/analysis/i1080_243_p9_boundary.md` | 再現しなかった参考の 1 本の時系列 |
| `mem_history/analysis/dryrun_selftest.md` | 集計の台本の試験 |

**写さなかったもの**（どれも原本の `outputs/comfort-calib-h3-ref-2026-11-03/` にあります）:

- 点ごとの記録 `runs/<label>/`（`run.json`・`job.json`・`request.json`・`server_slice.log`・`wddm.csv` ほか）と、履歴の測定の採取物 `mem_history/runs/`（`snapshot_oom.pickle` などのスナップショット・`oom_observer.json`・`run.json`）。
- `mem_history/analysis/i1080_243_p9_oom.md`（約 3.5MB。再現しなかった参考の 1 本の全セグメントの集計。要点は `mem_history/MEM_HISTORY_REPORT.md` §4）と `mem_history/analysis/*.json`。
- `mem_history/README_archive.md` と `mem_history/measure_h3_vram.orig.py`（git 管理の [`../h3-vram16-2026-10-18/`](../h3-vram16-2026-10-18/) の `README.md`・`measure_h3_vram.py` と同じ中身）、計測台の写し `mem_history/measure_h3_vram.py`（原本との差は上の `harness_copy.diff`）、`mem_history/` の集計の台本と前後の確認の記録。
- 較正台のコード（`calib.py` ほか）・素材 `materials/`・ログ（`drive2_*.log`・`commit_log.csv`・`preflight_*`・`rp_*`・`run_p_*`・`cool_*` ほか）・`plans/`・`RESULTS_template.md`・`make_results_ref.py`・`check_results.py`。
- 計画の正本はセッションの計画ファイル `C:\Users\phenotype nezu\.claude\plans\delegated-marinating-patterson.md`（git 管理外）です。

---

# comfort-calib-h3-ref-2026-11-03（台帳 §1-122・MiniMax H3 の参照つきの快適上限）較正台

- 計画の正本は `C:\Users\phenotype nezu\.claude\plans\delegated-marinating-patterson.md`（§1 参照が足す行・§2.2 判定規則 v3″・§2.3 点と予測・§2.5 較正台の拡張・§2.6 開始の門）。この README は較正台の使い方だけを書く。
- 結果は `RESULTS.md`、点ごとの記録は `progress.md`（どちらも計測が始まってから作る）。
- 準備した日は 2026-11-03。この時点で計測は 0 本（GPU は使っていない）。

## 1. 前回の較正台との差分

複製元は `outputs/comfort-calib-h3-2026-11-01/`（読むだけ）。持ち込んだのはコードだけ（`calib.py`・`mkplan.py`・`make_results.py`・`batch_table.py`・`vram_sampler2.ps1`・`commit_monitor.ps1`・`drive2.sh`・`commit_table.py`・`digest.py`・`selftest/`・`mk.sh`・`rp.sh`・`cool.sh`・`wait.sh`・`sum.py`）と、素材 5 点の写し（`materials/`）。`keyframe_f0.png` は持ち込んでいない（キーフレームなし。裁定 3）。

変えた箇所（全文は `harness_diff.txt`。番号は計画 §2.5 の作業項目）:

- 1 固定パス: `drive2.sh`・`mk.sh`・`rp.sh`・`wait.sh`・`cool.sh` の `cd` 先、`drive2.sh` の `CSV`、`commit_table.py`・`digest.py` の `D` を本フォルダに。`commit_monitor.ps1` は固定パスなし（`-Out` 引数）。`drive2.sh` は前の担当が書きかけたときに改行が CRLF になっていたので LF に戻した（CRLF のままだと bash の `cd` が失敗する）。
- 2 計画ファイル: 点に `references: [{kind, source_path, slot, max_sec?}]`（kind は製品と同じ `picture`／`video`／`audio`、`source_path` は本フォルダからの相対パス）。`mkplan.py`／`mk.sh` に `--ref kind:path:slot`（繰り返し可）。`--keyframes` の既定を 0 にし、K=0 ではキーフレームの欄を書かない。`check_point_h3()` の検査: 種類ごとの枚数（画像 9・動画 3・音声 3）と合計 12、種類ごとの slot の範囲、同じ種類で slot の重複なし、ファイルの存在、画像に `max_sec` なし、参照つきの点は Ref2VA の重み、計画ファイルの `predict` が較正台の計算と一致。キーフレームがある点は誤り（要求に `conditioning_images` を載せないので、載せたつもりの点が別物を測るのを防ぐ）。
- 3 要求: `build_request_h3()`。送る鍵は prompt・width・height・num_frames・frame_rate・seed・attention_backend・block_swap_prefetch・num_inference_steps・loras・references の 11 個（前回の `conditioning_images` を外し `references` を足した）。参照のない点は `references: []`（製品の既定と同じ）。`max_sec` は計画に書いたときだけ送る（今回は送らない）。参照の素材は種類と sha256 ごとに 1 回だけ `POST /upload/image|video|audio` で上げ、`reference_uploads.json` に覚えて使い回す（画像 ×n の点も `image_id` は 1 つ）。応答の鍵は `image_id`／`video_id`／`audio_id`。送信が 404 `REFERENCE_NOT_FOUND` で返ったら 1 回だけ上げ直して再送する。`run.json` に `request_sent`・`reference_uploads_used`。
- 4 行数の予測: `engine_h3/reference.py` の算術（`_round32`〔Python の `round`＝偶数丸め〕・`match_size`・`video_canvas`・`video_frame_count`・`frames_for_seconds`）と `qwen_pairs` の組の規則を `calib.py` に移植（較正台の Python には torch も av も無い）。素材の幅・高さ・秒は `tools/ffmpeg/bin/ffprobe.exe` で読む（動画の 24 fps の枚数＝`round(秒×24)`）。text の行は、アプリの venv にある `tokenizers` で製品の `tokenizer.json` を読み、ラベル（`<Picture n>: `・`<Video n>: `・`<Audio n>: `・`<x.x seconds>`）とプロンプトを製品と同じく 1 つずつ数える（視覚ブロックは行数＋開始と終了の 2）。`token_counts_h3()` の `predict: {video, cond, ref_video, ref_audio, audio, text_estimate, vc, vcr, S_estimate}` を計画ファイルと `run.json` に書く。正確な行数は完了点の `h3.phases.denoise.sequence` から取る。
- 5 OOM の点の S: `h3_worker.log` の `encode: N text tokens (with k reference(s))` の行から text を取り、ほかの行は式で足す（`S_source: "worker_log"`）。行が無いときだけ `predict.text_estimate`（`"estimate"`）。完了点は `"metadata"`。`run.json` の `h3.axes`（vc・vcr・S・各行・予測との差）。
- 6 集計: `sum.py`・`make_results.py`（「行数の軸」と「参照の符号化の段と TE の段」の 2 表を追加）・`calib.py report`（列を追加）・`digest.py`（vcr／S／text の列と送った参照）・`commit_table.py`（下の 10）。`cap_mib`・OOM の場所・`reference_encode` と `encode` の山と秒・`te_read.tokens` を出す。
- 7 照合: `used_check_h3()` に `references`（`metadata.json` の最上位 `references` の `pictures/videos/audios` と `inputs[]` の kind・slot・id・max_sec を、送った要求〔id は `reference_uploads.json` から〕と突き合わせる。参照のない点は「最上位に `references` が無い」が正しい）。`images_used` は 0、`transformer_used` は `minimax_h3_ref2va_pruned_int8_convrot`。`run.json` の `reference_inputs_resolved` に、報告された id を `reference_uploads.json` で引いた元ファイルを写す。`worker_conditions` に `generate:` の行の `references=[...]`（`generate_references`）と `encode: N text tokens` の行を写す。
- 8 復帰: `drive2.sh` は前回のまま。OOM のあとの `load` は `TRANSFORMER` の既定を `minimax_h3_ref2va_pruned_int8_convrot` にした（`GET /models` の `categories.transformer.entries[].name` にあることを確認済み）。
- 9 窓: server.log の窓の切り出し（`STAGE_START_RE`・`STAGE_PROG_RE`）は `(\d+) steps` で、`(4 steps)` を固定していない。参照つきの encode は `started (4 steps)` 〜 `5/4`、参照なし K=0 は `started (3 steps)` 〜 `4/3` で、どちらも同じ規則で切れる（自己試験 H(8) の最後で確認）。
- 10 主メモリ: `commit_table.py` に、点の `encode` の窓と `denoise` の窓（`run.json` の `v3dprime.windows.*.start_iso`〜`end_iso`）の中の `commit_log.csv` の山の 2 列を足した。終わりの行が無い窓（denoise の途中の OOM）は点の壁時計の終わりで閉じる。
- 11 preflight: 利用者の既定の attention を sage に（裁定 1）。警告は sage の点でステップ数と seed の 2 つ、sdpa の点（b1080x）は attention を足して 3 つ。
- 12 判定規則 v3″・窓・基準点の扱いは変えていない。
- OOM の場所（計画 §2.2）: `job.error` の traceback の「最初の traceback の最後のフレーム」を `v3dprime.oom_location.last` に、段を `stage`（`denoise`／`encode`／`reference_encode`／`decode`／`other`）に、線に数えるかを `counts_for_line`（denoise のときだけ真）に書く。前回の OOM 点 s1080_277 の `error` から `lowvram.py:178`・denoise を読めることを確かめた。

## 2. 固定した条件と素材

- 全点 Ref2VA（`minimax_h3_ref2va_pruned_int8_convrot`）・K=0・seed 12345・24 fps・ステップ 4（`mk.sh` で 20 も可）・attention は b1080x だけ sdpa、ほかは sage・先読み on。
- 送らない鍵: LTX の加速 3 個・`keep_resident`（＝off）・`vram16_mode`（＝on）・`shift_video`／`shift_audio`（＝12／3）・`conditioning_images`・参照の `max_sec`。
- プロンプト: `outputs/h3-vram16-gates/f0_prompt.txt`（30 行）。
- 素材（`materials/`。元は `uploads/h3-ref-calibration/`。sha256 は元と一致）:

| ファイル | 中身（ffprobe） | sha256 | 使う点 |
|---|---|---|---|
| `large-image.jpg` | 1600×2057 | `cf8e040d0d3ef9222455fe8e1ce94bde571fe04cf34adf32e0712e51f92fee90` | i1080_*（slot 1〜n） |
| `H3-Ref-1080p-3s.mp4` | 1920×1080・24 fps・73 枚（3.042 秒）・音声 3.019 秒 | `8b15a3165c2ccf2eb301413261d30258b515459d1f18ec8887de9020a409a517` | v1080_* |
| `H3-Ref-1080p-15s.mp4` | 1920×1080・24 fps・360 枚（15.0 秒）・音声 14.977 秒 | `3d136a1dafbff1a959b9dfdb52d54177d9b2538767a5056da573d05c3ed3a428` | v720_*（動画）・a1080_260_15s（音声。`/upload/audio` に動画の容器のまま） |
| `H3-Ref-720p-15s.mp4` | 1280×720・24 fps・360 枚・音声 14.977 秒 | `3cb73ee0f5c71110ddb3a96d990e3e5326fd26974bd9a28024ff21451b5b2ad8` | 使わない（計画 §5） |
| `H3-Ref-720p-3s.mp4` | 1280×720・24 fps・73 枚・音声 3.019 秒 | `c3a99d63701ead07894d2a1bf99345b0882f4a00a94606c2ff01bc8a246b6719` | 使わない |

- 参照の行（較正台の計算。自己試験 H(9) で固定）: 画像は 1920×1088 で 1280×1632 → 2,040 行・text 2,048（札 6＋2,040＋2）。3 秒の動画は画布 1344×768・73 枚・潜在 22・22,176 行・4 組・text 4,070（札 6＋4×1,016）。15 秒の動画は生成 277 で 277 枚・潜在 82・82,656 行・12 組、345 で 345 枚・潜在 102・102,816 行・15 組（2 桁秒の札は 7 行）。音声は 14.977 秒 → 600 潜在・1,200 行・text は札の 5 行だけ。
- `metadata.json` の鍵の置き場所（実ジョブ `outputs/2c160a44…/metadata.json` で確認）: 行数は `h3.phases.denoise.sequence.{text, cond, audio, video, ref_video, ref_audio, total}`（参照なしのジョブには `ref_video`／`ref_audio` の鍵が無い→0 と読む）、参照の段は `h3.phases.reference_encode.{seconds, peak_vram_mb, peak_vram_reserved_mb, commit_process_peak_gib, references[].{kind, slot, source_frames, used_frames, latent_t, pairs, rows, seconds}}`、TE は `h3.phases.encode.{peak_vram_mb, te_read.tokens, te_read.references[].{pairs, vision_tokens}}`、照合は最上位の `references.{pictures, videos, audios, inputs[].{kind, slot, id, max_sec}}`（参照なしのジョブには鍵が無い）と `models.selection.transformer.name`。

## 3. 打つ順の手順

`server start --pid`・`luid`・`.ps1` は監督が打つ（計画 §2.6 P0）。

1. 監督: `calib.py server start --pid 13976`・`calib.py luid`・`commit_monitor.ps1 -Out commit_log.csv` を起動。`git_state_start.txt`。
2. 担当: 引き継ぎの 3 点確認（`GET /api/v1/status` の待ち行列・直前の `drive2_*.log` の最終行・`bash.exe` のコマンドライン）。
3. 担当: `calib.py load --base-model MiniMaxH3 --transformer minimax_h3_ref2va_pruned_int8_convrot` → `GET /models` の `transformer_role` が `"ref2va"` → `calib.py idle-calib --minutes 3`。
4. 点ごとに（`plans/` に計画 §2.3 の 17 本を作って preflight 済み。`preflight_<label>.rc` も 0）: `plans/p_<label>.json` を目で確認 → `bash cool.sh` → `bash rp.sh <label> > rp_<label>.txt 2>&1`（裏）→ `bash wait.sh <label>`（繰り返す）→ `progress.md`。
   - 順: warm → b1080 → r1080_277 → i1080_277_p1 →（i1080_277_p1 が完走したときだけ r1080_294）→ i1080_260_p2 → i1080_260_p5 → i1080_243_p9 → v1080_226_3s → v1080_243_3s → b720 → v720_277_15s →（OOM なら v720_260_15s、快適なら v720_294_15s）→ a1080_260_15s → v720_345_15s → b1080x → x1080_* → *_20。
   - 監督が途中で作る点の例（sdpa の照合と 20 ステップ）:
     - `bash mk.sh x1080_260_p2 MiniMaxH3 minimax_h3_ref2va_pruned_int8_convrot sdpa 1920 1088 260 4 0 b1080x --ref picture:materials/large-image.jpg:1 --ref picture:materials/large-image.jpg:2`
     - `bash mk.sh i1080_260_p2_20 MiniMaxH3 minimax_h3_ref2va_pruned_int8_convrot sage 1920 1088 260 20 0 b1080 --ref picture:materials/large-image.jpg:1 --ref picture:materials/large-image.jpg:2`
     - 動画は `--ref video:materials/H3-Ref-1080p-3s.mp4:1`、音声は `--ref audio:materials/H3-Ref-1080p-15s.mp4:1`。
5. 集計: `make_results.py`・`calib.py report`・`commit_table.py`・`digest.py`・`sum.py <label>`。
6. 終わり: `calib.py restore-state` → `state.json` と `original_state.json` の突き合わせ → `POST /api/v1/pipeline/unload` → 3 点確認。

## 4. 自己試験（2026-11-03・GPU なし・サーバーへは `GET /models` と `GET /status` だけ）

- `selftest/parser_selftest.py`: 281 項目・失敗 0（`selftest/parser_selftest_output.txt`）。節 A〜H(3) は前回のまま。節 H(4) 以降を今回用に書き直した: 要求の鍵と参照の id の使い回し・preflight の誤りと警告・時間切れ・照合（`references`・Ref2VA・`images_used` 0）・移植した算術と実ジョブ `2c160a44`（used_frames 39・latent 12・pairs 2・rows 11,520・text 1,958・S 34,842）・全 17 点の `predict`・ワーカーのログの 2 行の読み取り・OOM の場所・`analyse_h3` の判定と軸（OOM の点の `S_source` が `worker_log`／`estimate`）・参照なしの点の `(3 steps)` の窓・節 I（`failed` で `error` が空のときの取り直し。偽の応答で）。環境変数 `H3_SELFTEST_DUMP=<フォルダ>` を付けると、作った偽の `run.json` をそのフォルダへ書く（集計スクリプトの試験用。本フォルダには書かない）。それで `sum.py`・`make_results.py`・`digest.py`・`commit_table.py` が落ちないことを確かめた。
- `selftest/analyse_selftest.py`: 43 項目・失敗 0（LTX の分）。
- `calib.py preflight`（`plans/` の 19 本。計画 §2.3 の 17 本＋2026-11-03 に足した r1080_260・i1080_226_p9）: 失敗 0（`selftest/preflight_output.txt`）。
- `calib.py run --dry-run`（19 本）: `dryrun/*.request.json`・`selftest/dryrun_output.txt`。`selftest/dryrun_validate.py`: 19 本とも製品の検証（`GenerateRequest`・`reject_unsupported`・`validate_for_family("h3")`）を通る（`selftest/dryrun_validate_output.txt`）。参照の id は仮の値に差し替えるので、アップロードの存在の検査（`check_references_exist`）は通していない。

## 5. 落とし穴

- `wait.sh` は 570 秒で戻る。1 点 16〜23 分（20 ステップは 70〜80 分）なので何度も呼ぶ。
- OOM の点は `metadata.json` が無い。S は `h3.axes`（`S_source` を見る）。段の山は WDDM の `encode` の窓・`commit_table.py` の窓の列・ワーカーのログの TE の行から取る。OOM のあとは `drive2.sh` が復帰を挟む（`load` は Ref2VA）。
- OOM の場所が `denoise` 以外（`encode`・`reference_encode`）の点は線の判定から外して別に報告する（`oom_location.counts_for_line` が偽）。
- `reference_uploads.json` の id は `uploads/` が掃除されると無効になる。そのときは送信が 404 `REFERENCE_NOT_FOUND` で返り、較正台が 1 回だけ上げ直す。上げ直しでも駄目なら `submit_failed`。
- 音声の参照は `/upload/audio` に mp4 をそのまま上げる（製品が受け付ける。音声の無い・無音のファイルは 422）。
- 参照の行の予測は ffprobe の秒から 24 fps の枚数を出す（`round(秒×24)`）。素材を替えたら自己試験 H(9) の値も見直す。
- text の予測に `tokenizers` を使う。読めないと preflight が誤りにする。
- 計画 §2.3 の表の S（見積）は text を 1 組あたり約 8 で丸めているので、較正台の `predict` と最大 75 行ずれる点がある（i1080_260_p2・v1080_*・v720_*・a1080_260_15s）。比べるのは較正台の値。
- `mkplan.py` は既にある計画ファイルと使用済みのラベルを拒む。作り直すときはファイルを消してから。2 回目の点は `_r2`。
- サーバーは `status=failed` を先に立てて `error` を後で書くので、失敗を見た直後の `error` が空のことがある（r1080_277 で発生。前回の較正も同じ: `Docs\VERIFICATION_LOG.md` §192.7 の 1）。`poll_job` は `failed` で `error` が空なら 2 秒おきに最長 60 秒取り直し、それでも空なら `run.json` に `error_pending: true` を残す。r1080_277 は `job.json` を取り直し（元は `job.json.before_refetch`、取り直す前の `run.json` は `run.json.before_refetch`）、`reanalyse --label r1080_277` で OOM（denoise・`lowvram.py:178`）に直した（2026-11-03）。
- UTF-8（`PYTHONIOENCODING=utf-8`・`PYTHONUTF8=1`）。`calib.py` は CRLF のまま（前回と同じ）。`server stop` は打たない。
