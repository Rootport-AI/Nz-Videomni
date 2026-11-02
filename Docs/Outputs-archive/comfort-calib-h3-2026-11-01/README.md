> **このファイルは実機の `outputs/comfort-calib-h3-2026-11-01/README.md`（git追跡外）のスナップショットである（2026-11-02複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# comfort-calib-h3-2026-11-01（台帳 §1-95・MiniMax H3 の快適上限）較正台

- 計画の正本は `C:\Users\phenotype nezu\.claude\plans\snug-booping-sphinx.md`（§2.2 判定規則 v3″・§2.4 較正台の拡張・§2.5 腕と順序）。この README は較正台の使い方だけを書く。
- 結果は `RESULTS.md`、点ごとの記録は `progress.md`（どちらも計測が始まってから作る）。
- 準備した日は 2026-11-01。この時点で計測は 0 本。

## 1. 複製元との差分

複製元は `outputs/comfort-calib-2026-10-04/`。持ち込んだのはコードだけ（`calib.py`・`mkplan.py`・`make_results.py`・`batch_table.py`・`vram_sampler2.ps1`・`drive2.sh`・`commit_table.py`・`digest.py`・`selftest/`・`mk.sh`・`rp.sh`・`cool.sh`・`wait.sh`・`sum.py`）。`one.sh` は複製元に無かったので持ち込んでいない。`runs/`・`manifest.jsonl`・`plans/`（中身）・`calib/`・`original_state.json`・`luid.json`・`server.pid`・`commit_log.csv`・ログ・`RESULTS.md`・`progress.md` は持ち込んでいない。`commit_monitor.ps1` は監督が置く。

変えた箇所（全文は `harness_diff.txt`。番号は計画 §2.4 の作業項目）:

- 1 固定パス: `drive2.sh`・`mk.sh`・`rp.sh`・`wait.sh`・`cool.sh` の `cd` 先、`drive2.sh` の `CSV`、`commit_table.py`・`digest.py` の定数 `D` を本フォルダに。
- 2 系統: `calib.py` の `load`／`restore-state` の `--base-model` に `MiniMaxH3`。`engine_tag_of()` が `ltx23`／`ltx25`／`h3` の 3 値。`WORKER_LOGS` に `h3`（`logs/h3_worker.log`）。
- 3 格子と規則: H3 の点は `check_point_h3()` で検査（32 の倍数・17n+5・22〜481・24 fps・ステップ 1〜50・キーフレーム 1〜10 枚・重複なし・0〜F−1・強度なし・画像ファイルの存在・計画の式どおりの `frame_idx`）。LTX の 8n+1・401 フレーム・LTX23／LTX25 限定・「single は LoRA を持たない」は H3 には掛からない。`token_counts_h3()`（潜在 `(F−5)//17*5+2`・映像トークン・条件行・映像＋条件行・音声行）。
- 4 要求: `build_request_h3()`。送る鍵は prompt・width・height・num_frames・frame_rate・seed・attention_backend・block_swap_prefetch・num_inference_steps・loras・conditioning_images（`{image_id, frame_idx}`。強度なし）の 11 個だけ。キーフレーム画像は `POST /upload/image` で 1 回上げ、`reference_uploads.json` に覚えて使い回す（送信が `IMAGE_NOT_FOUND` で返ったら 1 回だけ上げ直して再送）。
- 5 窓: `analyse()` は H3 の点を `analyse_h3()` へ回す。窓は server.log の `generate text encode`（`encode`・記録だけ）・`generate stage-1 denoise`（`denoise`）・`generate VAE decode`（`decode`）の「started の行〜最後の行」（`h3_stage_window()`。既存の `stage_groups()` の始点と終点）。1 ステップの秒はログ側が `経過秒 ÷ N`（0 ステップ目の行が計算の前に出るため。LTX の `÷ (N−1)` ではない）、metadata 側が `(denoise.seconds − dit.read_seconds) ÷ steps`。`stage2`・`restore` は H3 では作らない。
- 6 判定: v3″ を `run.json` の `v3dprime`／`verdict_v3dprime` に書く（`oom`・`oom_vram16`・`prefetch_degraded`・窓ごとの `delta_mb`・`decidable`・`plateau`・`reasons`・`comfortable`・基準点の `baseline_health`）。判定の値は `baseline`／`smoke`／`comfortable`／`uncomfortable`／`undecidable`／`oom`／（失敗のときは job の状態）。
- 7 照合: `used_check_h3()`。鍵は `metadata.json` の実際の置き場所から読む（`h3_used_from_meta()`）。None は不一致に数える。照合する鍵は vram16_mode_used・attention_used・keep_resident_used・images_used・loras・num_inference_steps・dit_source・transformer_used。`block_swap_prefetch_used` は照合に入れない（監督裁定 2026-11-01: 照合は「送った条件が守られたか」、先読みの降格は「結果」なので v3″ の `prefetch_degraded` だけで扱う）。完了しなかったジョブは `ok: null`（対象外）。`h3.loras` が無い＝LoRA なし、名前は拡張子を外して比べる。
- 7 の補足（監督裁定 2026-11-01）: どの点も `run.json` の `worker_conditions` に、ジョブ自身の `generate:` の行（seed・幾何・attention・prefetch・vram16・枚数・LoRA）と、送信の時点での直前のロードの行（`h3_worker.log` の `load: transformer_path OK …` のパスと `LOAD_OK` の options）を写す。送った要求は `request_sent`（＝`runs/<label>/request.json`）。OOM の点でも条件を後から読める。
- 8 復帰: `drive2.sh` が各点の前に `GET /status` の `pipeline_loaded` を見て、false なら `calib.py load` → `idle-calib --minutes 1` → 捨てラン（`$WARM_PLAN`。既定 `plans/p_warm.json` を `--label-suffix _rec<日時>` で）→ 冷却 150 秒。
- 9 時間切れ: `calib.py run --job-timeout auto`（4 ステップ以下 7,200 秒・それより多いと 10,800 秒）。`drive2.sh` は `auto` を渡す（夜 2 から点の行は環境変数 `JOB_TIMEOUT` で変えられる。§5）。
- 10 2 回目の点: ラベルを `_r2` にする（`mk.sh` のラベルは自由。`mkplan.py` は使用済みのラベルを拒む）。`report` はラベルごとに最新 1 件のまま。
- 11 preflight: 利用者の既定と違う指定を `warnings` に出す（attention が sdpa でない・ステップが 20 でない・seed を固定している・LoRA を付けている）。
- 12 `mkplan.py`／`mk.sh`: H3 の引数（下の §3）。
- 13 集計: `sum.py`・`make_results.py`（`h3` の節）・`calib.py report`（H3 の表）に、映像＋条件行・S・1 ステップの秒・OOM の種別・降格・`cap_mib`・窓ごとの基準比を出す。`commit_table.py`・`digest.py` は H3 の点で v3″ の判定と denoise／decode の基準比を出す。
- 追加: `calib.py run --dry-run`（サーバーなしで要求を `dryrun/<label>.request.json` に書く）・`--label-suffix`。`selftest/dryrun_validate.py`（`dryrun/` の要求を製品の検証〔`GenerateRequest`・H3 アダプタの `reject_unsupported`・`validate_for_family("h3")`〕にオフラインで通す）。

## 2. 固定した条件

- キーフレーム画像: `keyframe_f0.png`（本フォルダ）。F0 の画像 `uploads/71c540dc-970b-470e-aa0e-856dce76dfdc/input.png`（`outputs/h3-vram16-gates/g4.sh` の `--image`）の写し。896×1152（縦長）。sha256 `2c9ec216185440c91aa4fd804b0a403dcccd4fb3c587277e624c126eb99e698a`。`uploads/` は一時置き場なので、消えても困らないように較正台の中へ写した。K=10 の点は同じ 1 枚を 10 か所に置く（`image_id` 1 つ）。
- プロンプト: `outputs/h3-vram16-gates/f0_prompt.txt` を `mkplan.py` が読んで計画ファイルに書き込む。
- 送らない鍵: LTX の加速 3 個（`fused_gguf_dequant_kernel`・`vae_mode`・`keep_resident_embeddings`）・`keep_resident`（＝off）・`vram16_mode`（＝on）・`shift_video`／`shift_audio`（＝12／3）。
- seed 12345・24 fps・先読み on。
- `metadata.json` の鍵の置き場所（実データ `outputs/6a0b72f3…/metadata.json` で確認）: `peak_vram_mb` は `vram_optimization` の下、`commit_peak_gib`・`images_used`・`num_inference_steps`・`loras` は `h3` の下、DiT の読み込みの秒は `h3.phases.denoise.dit.read_seconds`、`attention_used` など `*_used` と `peak_vram_reserved_mb` は最上位。
- preflight の警告はどの点も 3 つ（attention・ステップ数・seed）。**Turbo の点だけ 4 つ**（LoRA が加わる。監督裁定で可）。
- 画像が縦長でもよい理由（監督裁定）: 製品はキーフレームを出力の大きさを覆うように拡縮して中央で切り抜くので、メモリには効かない。画像の読み取り部に入るのは先頭と末尾の 2 枚だけ。K=10 で同じ 1 枚を 10 か所に置くのも可。

## 3. 打つ順の手順（雛形）

GPU を使う手順（`load`・`idle-calib`・`rp.sh`・`run`）はオーナーの了承のあとだけ。`server start --pid`・`luid`・`.ps1` は監督が打つ。

1. 監督: `calib.py server start --pid <PID>`（`original_state.json` を記録）・`calib.py luid`・`commit_monitor.ps1` を起動。
2. 担当: 引き継ぎの 3 点確認（`GET /api/v1/status` の待ち行列・直前の `drive2_*.log` の最終行・`bash.exe` のコマンドライン）。
3. 担当: `calib.py load --base-model MiniMaxH3 --transformer default` → `calib.py idle-calib --minutes 3`。開始時の選択は `original_state.json` のとおり。**`--transformer default` は必須**（開始時の選択と違うことがある。省くと別の重みを測る）。
4. 点ごとに: `bash mk.sh <label> MiniMaxH3 default <sage|sdpa> <W> <H> <frames> <steps> <K> [turbo] <基準点|FIRST|SMOKE>` → `plans/p_<label>.json` を目で確認 → `bash cool.sh` → `bash rp.sh <label> > rp_<label>.txt 2>&1`（裏）→ `bash wait.sh <label>`（必要なだけ繰り返す）→ `progress.md`。
   - 例: `bash mk.sh warm MiniMaxH3 default sage 640 384 56 4 1 SMOKE`（煙の点と復帰用の捨てラン）
   - 例: `bash mk.sh b720 MiniMaxH3 default sage 1280 768 56 4 1 FIRST`（基準点）
   - 例: `bash mk.sh s720_481 MiniMaxH3 default sage 1280 768 481 4 1 b720`
   - 例: `bash mk.sh k720_345 MiniMaxH3 default sage 1280 768 345 4 10 b720k`
   - 例: `bash mk.sh t720_turbo MiniMaxH3 default sage 1280 768 345 4 1 turbo b720`
5. 集計: `make_results.py`・`calib.py report`・`commit_table.py`・`digest.py`。
6. 終わり: `calib.py restore-state`（`original_state.json` に記録した開始時の選択へ戻す）→ `state.json` と `original_state.json` の突き合わせ → `POST /api/v1/pipeline/unload` → 3 点確認。

昼 3（2026-11-02 10:30〜11:46。追加の 1 点 s1440_141_20＝2560×1472×141・sage・20 ステップ・K=1・基準点は夜 2 の b1440・夜 2 と同じサーバープロセス）の記録ファイル: `git_state_start_night3.txt`・`git_state_end_night3.txt`（HEAD `05ee80b`）・`load_night3_output.txt`・`idle_night3_output.txt`・`restore_night3_output.txt`・`plans/p_s1440_141_20.json`・`preflight_s1440_141_20_output.txt`・`rp_s1440_141_20.txt`・`run_p_s1440_141_20_output.txt`・`drive2_s1440.log`（末尾の 3 行）・`runs/s1440_141_20/`・`commit_log.csv`（10:30 以降の行。監視は 11:46 に停止）・`progress.md`（11-02 10:31〜11:45 の行）。名前は `night3` だが昼に取った。夜 1・夜 2 の記録ファイルは `RESULTS.md` §0 の出典の行。

`plans/` には見本として `p_warm`・`p_b720`・`p_s720_481`・`p_b720k`・`p_k720_345`・`p_t720_turbo` を置いた（preflight 済み）。計画どおりの値ならそのまま使える。作り直すときは同じ名前のファイルを消してから `mk.sh` を打つ（`mkplan.py` は既にあるファイルを上書きしない）。

## 4. 自己試験（2026-11-01・GPU なし）

- `selftest/parser_selftest.py`: 183 項目・失敗 0（`selftest/parser_selftest_output.txt`）。節 A〜G は複製元のまま。節 H を H3 用に書き直した（手で作った server.log の 3 窓・実ログのジョブ `6a0b72f3` の 3 窓と `metadata.json` の `sequence`・トークン・要求の鍵・preflight の誤りと警告・時間切れ・照合・`analyse_h3` の判定 9 通り・ロードの行の読み取り・OOM の点の `worker_conditions`）。
- `selftest/analyse_selftest.py`: 43 項目・失敗 0（LTX の分。変更なしで通る）。
- `calib.py preflight`（`plans/` の 6 本）: 失敗 0（`selftest/preflight_output.txt`）。
- `calib.py run --dry-run`（6 本）: `dryrun/*.request.json`・`selftest/dryrun_output.txt`。`selftest/dryrun_validate.py`: 6 本とも製品の検証を通る（`selftest/dryrun_validate_output.txt`）。

## 5. 落とし穴

- `plans/` は最初の `mk.sh` の前にあること（作ってある）。
- `wait.sh` は 570 秒で戻る。H3 の 1 点は 15〜20 分（20 ステップは 50〜75 分）なので、1 回のシェル呼び出しに 1 回ずつ何度も呼ぶ。
- OOM の点は `metadata.json` が無い。S は同じプロンプトの完了点の text 行数から計算する（`run.json` の `h3.sequence_estimate`）。OOM のあとはパイプラインが unload されるので、次の点の前に `drive2.sh` が復帰を挟む（`recover_output.txt`・`run_warm_rec*_output.txt`）。
- 先読みが降格した点（`block_swap_prefetch_used` が on でない）は v3″ の `prefetch_degraded` で非快適に数える。照合（`used_matches_request`）には出ない。
- 復帰の `idle-calib --minutes 1` は `calib/calibration.json` を上書きする（v3″ は基準点との比較なので判定には効かない。効くのは記録だけの v1／v2 の列）。
- 復帰の捨てランは `drive2.sh` のコミット 95% の停止の検査の対象外（検査は計測点の間だけ）。
- `h3.loras` は LoRA を付けたジョブにだけ書かれる。名前はファイル名（`.safetensors` 付き）なので、照合は拡張子を外して比べる。
- サーバーの直前の H3 ジョブ（server.log 2026-10-31 23:38）は `PulpCut_minimax_h3_fl2va_pruned_turbo_int8_convrot` で走っていた。`load --transformer default` を省くと別の重みを測る。照合の `transformer_used` で検出できる。
- 点の時間切れは `drive2.sh` が環境変数 `JOB_TIMEOUT`（秒。既定 `auto`）で受ける（2026-11-01 夜 2。sdpa・20 ステップは `auto` の 10,800 秒に収まらない）。例: `JOB_TIMEOUT=28800 bash rp.sh <label>`。復帰の捨てランは `auto` のまま。
- UTF-8（`PYTHONIOENCODING=utf-8`・`PYTHONUTF8=1`）。`server stop` は打たない。切り替え後に `reanalyse` を打たない。
