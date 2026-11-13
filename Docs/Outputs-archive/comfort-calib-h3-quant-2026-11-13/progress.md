> 複写（スナップショット）: 原本 `outputs/comfort-calib-h3-quant-2026-11-13/progress.md`（git 管理外）・複写日 2026-11-13

# progress（台帳 §1-285 H3 形式別の快適上限・2026-11-13）

## P0（09:29〜09:30）
- 09:29:11 git_state_start.txt: HEAD fcccdcf・porcelain は ` M Docs/PENDING_TASKS.md` の 1 件・branch dev。
- /status: running・unloaded・キュー空（completed 0・failed 0）。サーバー: powershell 22936（run.ps1）→ python 17004（.venv main.py）→ python 10512。
- 09:29:21 server start --pid 22936 rc 0（「既に 18620 で待ち受け中・2 つ目は起こさない」。server.pid 22936）。original_state.json の MiniMaxH3 transformer = h3ErosMax_beta5_3178732_int8_TURBO_Hybrid（復元先・期待どおり）。
- 09:29:30 commit_monitor.ps1 を Start-Process（非表示）で起動、PowerShell PID 9048。commit_log.csv は 09:29:33 から約 3 秒おきに更新。
- 09:29:47 luid rc 0: luid_0x00000000_0x0000f1c2_phys_0（786.6MB）。nvidia-smi 待機時 780MiB（08:35 実測 835MiB より 55MiB 少ない。記録のみ）。主メモリのコミット 18.41／138.08GiB（13.3%）。
- 7 ファイル（drive2.sh・mk.sh・rp.sh・wait.sh・cool.sh・commit_table.py・digest.py）の `2026-11-04` は全て 0 件。selftest の直す前の結果は selftest_before.txt（門にしない）。
- engine.worker／engine_h3 の孤児なし。
- P0 合格 → 弾 I へ。

# 弾 I（int8 対照。transformer=default）
- 09:30:15 git_state_load_int8.txt（fcccdcf・PENDING_TASKS 1 件）。09:30:15 load MiniMaxH3/default rc 0（8.3 秒。応答・/models の active とも default）。transformer_path = models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors（20,970,379,616 バイト）。
- load_int8.txt（LOAD_OK）: quant_layers 200・quant_layers_by_format {"int8_tensorwise": 200}・fp8_mode null・quant_specs ["int8_tensorwise/convrot=True/g256"]。§5 の期待どおり。
- 09:31〜09:34 idle-calib 3 分 rc 0: dedicated 中央値 1,031.8MB・shared 中央値 164.4MB・182 標本。読み込み後の nvidia-smi 1,021MiB。
- preflight p_warm__default rc 0（警告はステップ 4・seed 固定のみ）。最初の点なので 09:34:16〜09:36:46 手で 150 秒待った。

| ラベル | 形 | 判定 | 開始〜終了 | S（predict／実測） | text | cap | 予約の山（denoise／decode） | 割り当ての山（denoise／decode） | retries | inactive_split | 1 ステップ秒（meta／log） | 所要秒 | used_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm__default | 640×384×56・K=1 | smoke（completed） | 09:36:51〜09:37:50 | 4,784／4,784 | 278 | 14,023 | 3,848／5,382 | 3,816／5,062 | 0 | 92 | 1.867／1.675 | 58 | 真 |
| b1080_int8 | 1920×1088×56・K=1 FIRST | baseline（completed。denoise 窓の共有メモリ中央値 1,175.4MB・n=79） | 09:40:26〜09:42:51 | 38,984／38,984 | 2,078 | 13,999 | 5,866／10,486 | 4,340／8,143 | 0 | 616 | 19.865／19.675 | 144 | 真 |
| s1080_362_int8 | 1920×1088×362・K=1（基準 b1080_int8） | **OOM**（failed・verdict oom・reasons oom_vram16）。denoise（線に数える）・`*.py:147 block_forward`（`o = torch.empty((s, inner), …)`。traceback は途中が切られている）。1/4 ステップ目を 352.9 秒で完走し 2 ステップ目で OOM。Tried to allocate 2.99 GiB・3.89 GiB free・13.67 GiB allowed・allocated 6.94 GiB・reserved but unallocated 3.84 GiB | 09:45:27〜09:52:02 | 223,604（predict）／223,604（worker_log） | 2,078 | —（失敗点は記録なし） | — | — | — | — | —／352.9 | 395 | —（失敗点） |

- 対照の合否（§5）: OOM の文の 4 つの数（要求 2.99・割り当て 6.94・予約済みの空き 3.84・許可 13.67GiB）と free 3.89GiB は 11-04 の s1080_362（runs/s1080_362/run.json・server_slice.log）と全て一致。OOM の場所（denoise・block_forward の `o` の確保・2 ステップ目）と 1 ステップ目の秒（352.9／11-04 354.4）も同じ → 環境は同じ。traceback の行番号だけ違う（11-04: pipeline_h3.py:1580／:1625・*.py:143、今回: :1966／:2025・*.py:147。e85d52f→fcccdcf のコードの行の移動）。
- error は空でなかった（§192.7 の取り直しは不要。s1080_362_int8_error.txt は作っていない）。
- 復帰: drive2.sh は 1 回の呼び出しで計画 1 本だけを回し、recover() は「次の点の前に pipeline_loaded=false なら」動く作り（11-04 も復帰は次の点 s1080_328 の前に走った）なので、今回は復帰は走っていない。09:52 の /status: pipeline_loaded false・unloaded・キュー空（completed 2・failed 1）。nvidia-smi 774MiB。主メモリのコミット 18.02／138.08GiB（13.1%）。孤児なし（python 17004・10512 のみ）。commit_monitor（PID 9048）は動かしたまま。
- 弾 I 終了。ここで止めて監督へ報告。

# 弾 W（Kijai w4a8。transformer=minimax_h3_fl2va_pruned_w4a8_mixed。差し替え後の較正台・環境変数なし）
- 監督が calib.py・drive2.sh・selftest/parser_selftest.py を差分入りに差し替え済み（元は *_before_quant.*）。
- 09:55:11 git_state_load_w4a8.txt（fcccdcf・PENDING_TASKS 1 件）。load 前の /status: unloaded・キュー空。09:55:11 load rc 0（3.1 秒。応答・/models の active とも minimax_h3_fl2va_pruned_w4a8_mixed）。transformer_path …/Weights/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors（12,540,858,008 バイト）。
- load_w4a8.txt（LOAD_OK）: quant_layers 200・quant_layers_by_format {"asym_w4a8_int8": 200}・fp8_mode null・quant_specs ["asym_w4a8_int8/convrot=True/g256/G16"]。期待どおり。
- 09:55〜09:58 idle-calib 3 分 rc 0: dedicated 中央値 1,031.9MB・shared 162.5MB・182 標本。nvidia-smi 1,023MiB。preflight（煙）rc 0（警告 2 件のみ）。

| ラベル | 形 | 判定 | 開始〜終了 | S（predict／実測） | cap | 予約の山（denoise／decode） | 割り当ての山（denoise／decode） | 余裕（cap−予約 denoise） | retries | inactive_split | 1 ステップ秒（meta／log） | 所要秒 | Δ共有 denoise／decode | 降格 | used_ok | load_ok_dit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm__minimax_h3_fl2va_pruned_w4a8_mixed | 640×384×56・K=1 | smoke（completed） | 09:58:40〜09:59:22 | 4,784／4,784 | 14,023 | 2,888／5,374 | 2,846／5,062 | — | 0 | 117 | 2.013／1.825 | 42 | —（denoise 中央値 663.9） | なし | 真 | asym_w4a8_int8 200・fp8_mode null |
| b1080_w4a8 | 1920×1088×56・K=1 FIRST | baseline（completed）。**ただし共有メモリの標本が欠けて基準点として使えない（下の事故）** | 10:01:59〜10:04:19 | 38,984／38,984 | 13,999 | 5,052／10,486 | 3,465／8,143 | 8,947 | 0 | 578 | 21.545／21.4 | 140 | 標本なし（denoise n=0・decode n=0・encode n=8） | なし | 真 | asym_w4a8_int8 200・fp8_mode null |

- **10:00〜10:02 担当の誤り（二重起動）**: b1080_w4a8 の起動を Bash の `&` で出したあと、Git Bash の `ps` に出てこないのを「死んだ」と誤って判断し、同じ点をもう 1 本起動した。実際には 1 本目（pt.sh PID 17236 → rp.sh → drive2.sh → calib.py run PID 22388）は生きていた。
  - サーバーは同時 1 ジョブなので、2 本目は 10:02:01 に `SUBMIT FAILED 409 JOB_BUSY` で終わった（GPU のジョブは 1 本だけ。job 3477effc…）。
  - しかし 2 本目は同じ `runs/b1080_w4a8/` に書き、終わり際に `stop_sampler.txt` を置いたので、1 本目の WDDM 標本（wddm.csv）が 10:02:02 で止まった（10 行）。そのため denoise・decode の窓の共有メモリ中央値が無い（n=0）。基準点として、木の点の退避判定（Δ共有）に使えない。
  - 1 本目の run.json（completed・baseline）は最後に書かれたので、2 本目の submit_failed は上書きされている。manifest.jsonl には submit_failed の行と completed の行が両方ある。drive2_b1080.log にも 2 本分の行がある。rp_b1080_w4a8.txt と run_p_b1080_w4a8_output.txt には 2 本分の出力が混ざっている。
  - cap・予約・割り当て・ステップの秒・S・load_ok_dit は 1 本目のジョブの値で、正しい見込み。
- 10:04 ここで止めて監督へ報告（b1080_w4a8 を取り直す方法の指示待ち）。/status: pipeline_loaded true・ready（w4a8 読み込み済み）・キュー空（completed 4・failed 1）。

## 取り直し（監督の指示: 案 B。事故の runs/b1080_w4a8/・rp_b1080_w4a8.txt・manifest の行はそのまま残す）
- 起動の規律: run_in_background で rp.sh を 1 本だけ。起動の後に Get-CimInstance で rp.sh／calib.py run が 1 本だけであることを確認（10:07 確認: rp.sh 1700・calib.py run 13136 の 1 本）。

| ラベル | 形 | 判定 | 開始〜終了 | S（predict／実測） | cap | 予約の山（denoise／decode） | 割り当ての山（denoise／decode） | 余裕 | retries | inactive_split | 1 ステップ秒（meta／log） | 所要秒 | 共有中央値 denoise／decode（n） | 降格 | used_ok | load_ok_dit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1080_w4a8_r2 | 1920×1088×56・K=1 FIRST | baseline（completed）。木 1080 の基準点 | 10:06:56〜10:09:17 | 38,984／38,984 | 13,999 | 5,052／10,486 | 3,465／8,143 | 8,947 | 0 | 578 | 21.5／21.325 | 141 | 662.5（85）／661.5（34） | なし | 真 | asym_w4a8_int8 200・fp8_mode null |
- 事故の b1080_w4a8 と cap・予約・割り当て・retries・inactive_split が全て一致（1 本目のジョブの値は正しかった）。共有メモリの中央値は int8 の基準点（1,175.4）より約 510MB 低い（記録のみ）。

## 木 1080（基準点 b1080_w4a8_r2）
| ラベル | 判定 | 開始〜終了 | S（predict／実測） | OOM の段・場所 | OOM の文（要求・free・許可・割り当て・予約済みの空き） | 何ステップ目 | 1 ステップ秒（log） | 所要秒 | Δ共有 denoise（n） | load_ok_dit |
|---|---|---|---|---|---|---|---|---|---|---|
| s1080_345_w4a8 | **OOM**（failed・oom_vram16） | 10:11:55〜10:23:34 | 213,348／213,348（worker_log） | denoise（線に数える）・**sage の中**: `sageattention/quant.py:275` `v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), …)`（int8 の 362 は `o` の確保だった） | 418.00 MiB・1.33 GiB・13.67 GiB・11.88 GiB・1.42 GiB | 3 ステップ目（1/4・2/4 を完走） | 333.9 | 699 | −19.9（665） | asym_w4a8_int8 200・fp8_mode null |
- §4.2 の予測（快適・余裕 340MiB）は外れ。計画の読み「Kijai の 345 が OOM なら a が +5,372 より大きい（xq 2 本の瞬間が山に乗る）」に当たる側。木に従い s1080_328_w4a8 へ（次の rp の頭で recover が走る）。
| s1080_328_w4a8 | **comfortable**（completed）。recover（10:26:08 load w4a8 rc 0・LOAD_OK asym_w4a8_int8 200・暖機 warm__…_rec1113102714 completed・冷却 150 秒）の後 | 10:30:24〜10:54:19 | 203,092／203,092 | — | — | 4/4 完走 | 305.1（meta 305.285） | 1,435 | −11.3（1,216）・decode −3.7（192） | asym_w4a8_int8 200・fp8_mode null |
  - s1080_328_w4a8 の数: cap 13,999・予約の山 denoise 13,444／decode 10,540・割り当ての山 12,240／8,261・余裕 555・retries 16・inactive_split 4,230・降格なし・used_ok 真。参考 int8（11-04 s1080_328）: 予約 12,964・割り当て 12,169・余裕 1,035・retries 9・297.5 秒／ステップ。
- **木 1080 の結論: c=328（S 203,092）・U=345（S 213,348。OOM 1 回・denoise・sage の中）**。int8（c=345・U=362）より 1 段下。快適の回数 1（_r2 は弾 Z）。

## 1440
| ラベル | 判定 | 開始〜終了 | S（predict／実測） | cap | 予約の山（denoise／decode） | 割り当ての山 | 余裕 | retries | inactive_split | 1 ステップ秒（meta／log） | 所要秒 | 共有中央値 denoise／decode（n） | 降格 | used_ok | load_ok_dit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1440_w4a8 | baseline（completed） | 10:56:56〜11:01:43 | 70,144／70,144 | 13,999 | 8,178／14,800 | 5,097／10,768 | 5,821 | 0 | 1,119 | 51.288／51.1 | 287 | 662.3（204）／650.1（56） | なし | 真 | asym_w4a8_int8 200・fp8_mode null |
  - 参考 int8（11-04 b1440）: 予約 8,214／14,800・割り当て 5,709／10,768・48.2 秒／ステップ。

| ラベル（木 1440・基準点 b1440_w4a8） | 判定 | 開始〜終了 | S（predict／実測） | OOM の段・場所 | OOM の文（要求・free・許可・割り当て・予約済みの空き） | 何ステップ目 | 1 ステップ秒（log） | 所要秒 | Δ共有 denoise（n） | load_ok_dit |
|---|---|---|---|---|---|---|---|---|---|---|
| s1440_192_w4a8 | **OOM**（failed・oom_vram16） | 11:04:18〜11:10:46 | 217,798／217,798（worker_log） | denoise（線に数える）・**w4a8 の線形層の活性の量子化**: `comfy_kitchen/backends/cuda/__init__.py:1604 quantize_int8_rowwise_convrot64` の `q_2d = torch.empty(…)`（どの線形層かは traceback の切り詰めで不明） | 1.46 GiB・2.13 GiB・13.67 GiB・10.28 GiB・2.23 GiB | 2 ステップ目（1/4 を完走） | 346.6 | 387 | −8.8（346） | asym_w4a8_int8 200・fp8_mode null |
- §4.2 の予測（境目 +39MiB）は OOM の側。木に従い s1440_175_w4a8 へ（次の rp の頭で recover）。
| s1440_175_w4a8 | **comfortable**（completed）。recover（11:13:19 load w4a8 rc 0・LOAD_OK asym_w4a8_int8 200・暖機 warm__…_rec1113111425 completed・冷却 150 秒）の後 | 11:17:37〜11:40:39 | 199,342／199,342 | — | — | 4/4 完走 | 294.55（meta 294.773） | 1,382 | −14.7（1,175）・decode −0.5（176） | asym_w4a8_int8 200・fp8_mode null |
  - s1440_175_w4a8 の数: cap 13,999・予約の山 denoise 13,838／decode 14,846・割り当ての山 12,058／10,861・余裕 161・retries 13・inactive_split 4,068・降格なし・used_ok 真。
- **木 1440 の結論: c=175（S 199,342）・U=192（S 217,798。OOM 1 回・denoise・w4a8 の活性の量子化）**。int8（c=192・U=209）より 1 段下。快適の回数 1（_r2 は弾 Z）。
- **弾 W のまとめ**: 快適の最大 S は 203,092（1080×328）、OOM の最小 S は 213,348（1080×345）。1440 は 199,342 快適・217,798 OOM で、帯は 1080 と矛盾しない（線は S 203,092〜213,348 の間）。int8 の線（213,348 快適・217,798 OOM〔1440×192 は int8 で快適〕・223,604 OOM）より下。
- 11:41 弾 W 終了。/status: pipeline_loaded true・ready・w4a8 読み込み済み（/models の active minimax_h3_fl2va_pruned_w4a8_mixed）・キュー空（completed 10・failed 3）。ここで止めて監督へ報告。

# 弾 Z-①（Kijai の c の _r2。読み込みは w4a8 のまま）
| ラベル | 判定 | 開始〜終了 | S | cap | 予約の山（denoise／decode） | 割り当ての山 | 余裕 | retries | inactive_split | 1 ステップ秒（log） | 所要秒 | Δ共有 denoise／decode | 降格 | 1 回目との差 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1080_328_w4a8_r2（基準 b1080_w4a8_r2） | **comfortable** | 11:44:53〜12:08:49 | 203,092／203,092 | 13,999 | 13,444／10,540 | 12,240／8,261 | 555 | 16 | 4,230 | 305.575 | 1,437 | −31.9（1,219）／−13.1（192） | なし | 予約・割り当て・retries・inactive_split 全て一致（差 0） |
| s1440_175_w4a8_r2（基準 b1440_w4a8） | **comfortable** | 12:11:26〜12:34:28 | 199,342／199,342 | 13,999 | 13,838／14,846 | 12,058／10,861 | 161 | 13 | 4,068 | 294.675 | 1,382 | −4.4（1,175）／+3.3（177） | なし | 予約・割り当て・retries・inactive_split 全て一致（差 0） |
- 弾 Z-① 合格: Kijai の c は 1080=328・1440=175 とも 2 回快適（快適の回数 2）。→ 弾 F へ。

# 弾 F（fp8。transformer=minimax_h3_fl2va_pruned_fp8_scaled）
- 12:34:49 git_state_load_fp8.txt（fcccdcf・PENDING_TASKS 1 件）。load 前の /status: ready（w4a8）・キュー空。12:34:50 load rc 0（3.9 秒。応答・/models の active とも minimax_h3_fl2va_pruned_fp8_scaled）。transformer_path …/Weights/minimax_h3_fl2va_pruned_fp8_scaled.safetensors（20,958,205,608 バイト）。
- load_fp8.txt（LOAD_OK）: quant_layers 200・quant_layers_by_format {"float8_e4m3fn": 200}・fp8_mode "scaled"・quant_specs ["float8_e4m3fn/convrot=False/g256"]。期待どおり。
- 12:35〜12:38 idle-calib 3 分 rc 0: dedicated 中央値 1,029.2MB・shared 145.0MB・182 標本。nvidia-smi 1,022MiB。preflight（煙）rc 0（警告 2 件のみ）。

| ラベル | 判定 | 開始〜終了 | S（predict／実測） | cap | 予約の山（denoise／decode） | 割り当ての山 | 余裕 | retries | inactive_split | 1 ステップ秒（meta／log） | 所要秒 | 共有中央値・Δ denoise／decode（n） | 降格 | used_ok | load_ok_dit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm__minimax_h3_fl2va_pruned_fp8_scaled | smoke（completed） | 12:38:30〜12:39:16 | 4,784／4,784 | 14,023 | 3,848／5,382 | 3,809／5,062 | — | 0 | 149 | 2.545／2.35 | 46 | 1,186.8（9）／1,186.9（8） | なし | 真 | float8_e4m3fn 200・fp8_mode scaled |
| b1080_fp8 | baseline（completed） | 12:41:54〜12:44:42 | 38,984／38,984 | 13,999 | 6,014／10,486 | 4,481／8,143 | 7,985 | 0 | 679 | 25.352／25.15 | 169 | 1,166.7（100）／1,158.0（34） | なし | 真 | float8_e4m3fn 200・fp8_mode scaled |
  - 参考 int8（今回 b1080_int8）: 予約 5,866／10,486・割り当て 4,340／8,143・19.675 秒／ステップ。

## 木 1080（fp8・基準点 b1080_fp8）
| ラベル | 判定 | 開始〜終了 | S | cap | 予約の山（denoise／decode） | 割り当ての山 | 余裕 | retries | inactive_split | 1 ステップ秒（meta／log） | 所要秒 | Δ共有 denoise／decode（n） | 降格 | used_ok | load_ok_dit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1080_345_fp8 | **comfortable** | 12:47:20〜13:14:55 | 213,348／213,348 | 13,999 | 13,486／10,548 | 12,661／8,271 | 513 | 9 | 4,786 | 355.155／354.95 | 1,655 | −3.7（1,415）／+12.0（203） | なし | 真 | float8_e4m3fn 200・fp8_mode scaled |
  - int8（11-04 s1080_345）: 予約 13,486／10,548・割り当て 12,669／8,271・余裕 513・retries 9・inactive 4,778・326.05 秒。予約の山は int8 と同じ（+0MiB。§4.2 の +150MiB 以上ではない）。割り当ては −8MiB。→ 木に従い s1080_362_fp8 へ。
| s1080_362_fp8 | **OOM**（failed・oom_vram16） | 13:17:32〜13:24:40 | 223,604／223,604（worker_log） | — | — | — | — | — | — | —／384.5 | 427 | −1.5（383） | — | — | float8_e4m3fn 200・fp8_mode scaled |
  - OOM: denoise（線に数える）・`*.py:147 block_forward` の `o = torch.empty((s, inner), …)`（int8 の 362 と同じ場所）・2 ステップ目（1/4 を 384.5 秒で完走）。要求 2.99 GiB・free 3.89 GiB・許可 13.67 GiB・割り当て 6.93 GiB・予約済みの空き 3.85 GiB（int8: 2.99・3.89・13.67・6.94・3.84。割り当てと予約済みの空きが 0.01 GiB ずつ違う）。
- **木 1080（fp8）の結論: c=345（S 213,348）・U=362（S 223,604。OOM 1 回・denoise・`o` の確保）**。int8 と同じ。快適の回数 1（_r2 は弾 Z-②で打たない）。

## 1440（fp8）
| ラベル | 判定 | 開始〜終了 | S | cap | 予約の山（denoise／decode） | 割り当ての山 | 余裕 | retries | inactive_split | 1 ステップ秒（meta／log） | 所要秒 | 共有中央値・Δ denoise／decode（n） | 降格 | used_ok | load_ok_dit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1440_fp8 | baseline（completed）。recover（13:27:14 load fp8 rc 0・LOAD_OK float8_e4m3fn 200・scaled・暖機 warm__…fp8_scaled_rec1113132820 completed・冷却 150 秒）の後 | 13:31:46〜13:37:11 | 70,144／70,144 | 13,999 | 8,362／14,800 | 5,811／10,768 | 5,637 | 0 | 1,252 | 58.207／58.0 | 325 | 1,176.5（232）／1,175.7（56） | なし | 真 | float8_e4m3fn 200・fp8_mode scaled |
  - 参考 int8（11-04 b1440）: 予約 8,214／14,800・割り当て 5,709／10,768・48.2 秒。
| s1440_192_fp8（木 1440 初手・基準 b1440_fp8） | **comfortable** | 13:39:48〜14:08:12 | 217,798／217,798 | 13,999 | 13,764／14,858 | 12,898／10,874 | 235 | 9 | 4,912 | 367.975／367.775 | 1,703 | −35.4（1,466）／−12.4（194） | なし | 真 | float8_e4m3fn 200・fp8_mode scaled |
  - int8（11-04 s1440_192）: 予約 13,764／14,858・割り当て 12,905／10,874・余裕 235・retries 9・inactive 4,905・338.1 秒。予約は同じ・割り当て −7MiB。→ 木に従い s1440_209_fp8 へ。
| s1440_209_fp8 | **OOM**（failed・oom_vram16） | 14:10:49〜14:11:30 | 236,254／236,254（worker_log） | — | — | — | — | — | — | — | 40 | 標本なし（denoise n=0） | — | — | float8_e4m3fn 200・fp8_mode scaled |
  - OOM: denoise（counts_for_line 真）・**sage の中** `sageattention/quant.py:275` の `v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), …)`・1 ステップ目（完走したステップなし。所要 40 秒）。要求 462.00 MiB・free 1.29 GiB・許可 13.67 GiB・割り当て 12.79 GiB・予約済みの空き 560.07 MiB。int8（11-04 s1440_209）: 462.00 MiB・1.22 GiB・13.67・12.80 GiB・552.25 MiB・同じく 1 ステップ目・40.1 秒。
- **木 1440（fp8）の結論: c=192（S 217,798）・U=209（S 236,254。OOM 1 回・denoise・sage の中）**。int8 と同じ。快適の回数 1（_r2 は打たない）。
- **弾 F のまとめ**: fp8 の c・U は 1080・1440 とも int8 と同じ。c の点の予約の山は int8 と MiB 単位で同じ（13,486・13,764）、割り当ての山は −8／−7MiB、retries 9 も同じ。1 ステップの秒は 1.09 倍（345: 354.95／326.05）・1.09 倍（192: 367.775／338.125）。
- 14:12 弾 F 終了。ここで止めて監督へ報告。

# 弾 Z-②・③は打たない（監督の判断）

# 弾 E（Eros。transformer=10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized。345 の 1 点だけ）
- 14:14:04 git_state_load_eros.txt（fcccdcf・PENDING_TASKS 1 件）。load 前は unloaded。14:14:04 load rc 0（3.1 秒。応答・/models の active とも 10Eros_…）。transformer_path …/Weights/10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors（13,997,668,774 バイト）。
- load_eros.txt（LOAD_OK）: quant_layers 200・quant_layers_by_format {"asym_w4a8_int8": 180, "int8_tensorwise": 20}・fp8_mode null・quant_specs [asym_w4a8_int8 G16/G4/G8、int8_tensorwise]。期待どおり。
- 14:14〜14:17 idle-calib 3 分 rc 0: dedicated 中央値 1,042.2MB・shared 162.3MB・182 標本。nvidia-smi 1,033MiB。preflight（煙）rc 0（警告 2 件のみ）。

| ラベル | 判定 | 開始〜終了 | S | cap | 予約の山（denoise／decode） | 割り当ての山 | 余裕 | retries | inactive_split | 1 ステップ秒（meta／log） | 所要秒 | 共有中央値・Δ denoise／decode（n） | 降格 | used_ok | load_ok_dit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm__10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized | smoke（completed） | 14:17:41〜14:18:23 | 4,784／4,784 | 14,023 | 3,848／5,382 | 3,816／5,062 | — | 0 | 112 | 1.988／1.75 | 42 | 1,178.2（7）／1,178.2（7） | なし | 真 | asym_w4a8_int8 180・int8_tensorwise 20・fp8_mode null |
| b1080_eros | baseline（completed） | 14:21:01〜14:23:23 | 38,984／38,984 | 13,999 | 5,866／10,486 | 4,436／8,143 | 8,133 | 0 | 718 | 21.378／21.175 | 142 | 1,175.0（85）／1,174.7（34） | なし | 真 | asym_w4a8_int8 180・int8_tensorwise 20・fp8_mode null |
  - 予約の山: 煙 3,848（int8 3,848・Kijai 2,888）、基準点 5,866（int8 5,866・Kijai 5,052）→ どちらも int8 と同じ値。
| s1080_345_eros（基準 b1080_eros） | **OOM**（failed・oom_vram16） | 14:26:00〜14:26:40 | 213,348／213,348（worker_log） | — | — | — | — | — | — | — | 40 | 標本なし（denoise n=0） | — | — | asym_w4a8_int8 180・int8_tensorwise 20・fp8_mode null |
  - OOM: denoise（counts_for_line 真）・**sage の中** `sageattention/quant.py:275` の `v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), …)`・1 ステップ目（完走したステップなし。TE 15.5 秒の後すぐ）。要求 418.00 MiB・free 1.02 GiB・許可 13.67 GiB・割り当て 12.83 GiB・予約済みの空き 799.40 MiB。
- **弾 E の結論（1 点の裁定）**: 345 が OOM → 「Eros の線は 213,348 より下。量は未定」。指示どおり打ち止め（328・362 は打たない）。

# 復元（14:27）
- 14:27:17 restore-state rc 0: MiniMaxH3／h3ErosMax_beta5_3178732_int8_TURBO_Hybrid を load（3.0 秒。応答・/models の active 一致・LOAD_OK int8_tensorwise 200・fp8_mode null・transformer_path …/h3ErosMax_beta5_3178732_int8_TURBO_Hybrid.safetensors 20,970,414,464 バイト）。state.json の selections.MiniMaxH3.transformer を確認。
- POST /api/v1/pipeline/unload → {"pipeline_loaded":false,"state":"unloaded"}（unload_output.txt）。
- state.json と original_state.json: active_base_model（MiniMaxH3）・selections（LTX23・LTX25・MiniMaxH3 の 4 鍵ずつ）が全て一致。違う鍵は記録用の鍵だけ（state.json の schema、original_state.json の captured_at・server_shell_pid・server_python_pids）。
- /status: running・unloaded・キュー空（completed 20・failed 6）。
- commit_monitor（PID 9048）を Stop-Process で停止・消えたことを確認。commit_log.csv は残した（最後の行 14:27:36・18.79／138.08GiB・13.6%）。
- 孤児なし（python 17004・10512 のみ）。GPU の待機使用量 778MiB。git: fcccdcf・PENDING_TASKS 1 件。
- **計測完了（2026-11-13 14:27:41）**。

# 弾 X（計画外の 2 点。オーナー了承済みと監督から指示）
- 14:56:44 commit_monitor を再起動（PowerShell PID 47848。commit_log.csv に追記・約 3 秒おき）。14:57:02 git_state_x.txt（fcccdcf・PENDING_TASKS 1 件）。nvidia-smi 待機 777MiB。GPU を使う他のプロセス: デスクトップの C+G（NVIDIA Overlay・Teams・Discord・Slack・VS Code など）のみ。AviUtl2 なし・engine.worker なし。/status unloaded・キュー空。server start は打っていない。

## X-1（int8 の 2560×1472×175）
- 14:57:02 load default rc 0（応答・/models の active 一致）。load_int8_x.txt: {"int8_tensorwise": 200}・fp8_mode null。idle-calib 3 分 rc 0（dedicated 中央値 1,037.3MB・shared 146.1MB）。
- 煙は calib.py run --label-suffix _x で直接（runs/warm__default_x）。b1440_int8 の前は cool.sh が drive2 のログしか見ないので手で 150 秒待った（15:01:16〜15:03:46）。

| ラベル | 判定 | 開始〜終了 | S | cap | 予約の山（denoise／decode） | 割り当ての山 | 余裕 | retries | inactive_split | 1 ステップ秒（meta／log） | 所要秒 | 共有中央値・Δ denoise／decode（n） | 降格 | used_ok | load_ok_dit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm__default_x | smoke（completed） | 15:00:20〜15:01:14 | 4,784／4,784 | 14,023 | 3,848／5,382 | 3,816／5,062 | — | 0 | 92 | 1.875／1.675 | 54 | 1,167.0（7）／1,167.0（8） | なし | 真 | int8_tensorwise 200 |
| b1440_int8 | baseline（completed） | 15:03:50〜15:08:37 | 70,144／70,144 | 14,001 | 8,214／14,800 | 5,709／10,768 | 5,787 | 0 | 1,233 | 48.513／48.325 | 287 | 1,192.0（193）／1,180.4（57） | なし | 真 | int8_tensorwise 200 |
| s1440_175_int8（基準 b1440_int8） | **comfortable** | 15:11:13〜15:33:53 | 199,342／199,342 | 14,001 | 12,700／14,846 | 12,007／10,861 | 1,301 | 9 | 4,396 | 286.517／286.3 | 1,360 | Δ −26.1（1,142）／−18.3（176） | なし | 真 | int8_tensorwise 200 |
  - Kijai s1440_175_w4a8 と並べる: 予約 13,838（int8 より +1,138）・割り当て 12,058（+51）・retries 13（int8 9）・inactive_split 4,068（int8 4,396）・294.55 秒（int8 286.3）。
  - 予約の山 12,700 は、監督の 2 つの見込み（予約の読み 約 13,384・割り当ての読み 約 13,786）のどちらとも合わない（どちらより 684／1,086MiB 低い）。

## X-2（Eros の 1920×1088×328）
- 15:34:28 git_state_load_eros_x.txt（fcccdcf・PENDING_TASKS 1 件）→ load 10Eros_… rc 0（応答・/models の active 一致）。load_eros_x.txt: {"asym_w4a8_int8": 180, "int8_tensorwise": 20}・fp8_mode null。idle-calib 3 分 rc 0（dedicated 1,039.1MB・shared 160.3MB）。煙は --label-suffix _x で直接（completed）。s1080_328_eros の前は手で 150 秒待った（15:38:28〜15:40:58）。

| ラベル | 判定 | 開始〜終了 | S | cap | 予約の山（denoise／decode） | 割り当ての山 | 余裕 | retries | inactive_split | 1 ステップ秒（log） | 所要秒 | Δ共有 denoise（n） | 降格 | used_ok | load_ok_dit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm__10Eros…_x | smoke（completed） | 15:37:42〜15:38:26 | 4,784／4,784 | 14,023 | 3,848／5,382 | 3,816／5,062 | — | 0 | 112 | 1.75 | 44 | —（1,192.0・n=7） | なし | 真 | asym_w4a8_int8 180・int8_tensorwise 20 |
| s1080_328_eros（基準 b1080_eros） | **OOM**（failed・oom_vram16） | 15:41:01〜15:46:44 | 203,092／203,092（worker_log） | — | — | — | — | — | — | 304.7 | 343 | −32.8（303） | — | — | asym_w4a8_int8 180・int8_tensorwise 20 |
  - OOM: denoise（線に数える）・**w4a8 の線形層の活性の量子化** `comfy_kitchen/backends/cuda/__init__.py:1604 quantize_int8_rowwise_convrot64` の `q_2d = torch.empty(…)`（Kijai の s1440_192 と同じ場所）・2 ステップ目（1/4 を 304.7 秒で完走）。要求 1.02 GiB・free 1.96 GiB・許可 13.67 GiB・割り当て 11.40 GiB・予約済みの空き 1.26 GiB。
  - → 「Eros の線は 203,092 より下」。指示どおり打ち止め（311・345 再測は打たない）。

## 復元（弾 X の後・15:47）
- 15:47:18 restore-state rc 0（h3ErosMax_beta5_3178732_int8_TURBO_Hybrid を load・応答と /models の active 一致・LOAD_OK int8_tensorwise 200・state.json 確認）→ POST /pipeline/unload → {"pipeline_loaded":false,"state":"unloaded"}（unload_x_output.txt）。
- state.json と original_state.json: active_base_model・selections 全一致。違う鍵は記録用だけ（state.json の schema、original_state.json の captured_at・server_shell_pid・server_python_pids）。
- /status: running・unloaded・キュー空（completed 24・failed 7）。commit_monitor（PID 47848）停止・消滅確認。commit_log.csv は残した（最後の行 15:47:36・18.48／138.08GiB・13.4%）。孤児なし（python 17004・10512 のみ）。GPU 待機 773MiB。git fcccdcf・PENDING_TASKS 1 件。
- **弾 X 完了（2026-11-13 15:47:39）**。
