> **このファイルは実機の `outputs/comfort-calib-h3-chain-2026-11-08/progress.md`（git追跡外）のスナップショットである（2026-11-08複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# progress（台帳 §1-276 MiniMax H3 の Chained の快適上限の較正・2026-11-08 夜）

- 夜のあいだに計測担当が書いた時系列の記録は無い（監督が直接打った）。この記録は、較正台担当が翌朝に作業場所のログから組み立てたもの。出典は行ごとに書く: `drive2_<label>.log`（点の開始・終了・コミット）、`manifest.jsonl`・`runs/<label>/run.json`（点の時刻と値）、`recover_output.txt`（自動復帰）、`commit_log.csv`（主メモリのコミット量、3 秒おき）、`original_state.json`・`luid.json`・`git_state_start.txt`（開始時の状態）。
- 点の値の表は `RESULTS.md` §4（台本 `analysis/collect.py` の出力）。ここには時刻と出来事だけを書く。

## 開始（01:37〜01:41）

- 01:37:37 `original_state.json`: 有効な系統 MiniMaxH3・重み `minimax_h3_ref2va_pruned_int8_convrot`・サーバーのシェル PID 45972・python 50876。
- 01:37:39 `luid.json`: `luid_0x00000000_0x0000f2a0_phys_0`（999.2MB）。
- 01:37:53 `commit_log.csv` の最初の行（24.73／138.36GiB・17.9%）。
- 01:38:05 `git_state_start.txt`: HEAD `091d2aa`。未コミットの変更は操作パネル（`webui/src`）の 5 ファイルと未追跡の 1 ファイル（`ChainedScreen.h3Comfort.test.tsx`）。サーバー側のファイルの変更は無い。
- 読み込みと idle-calib（3 分）の結果ファイル `calib/calibration.json` は 03:01 の自動復帰の idle-calib（1 分）で上書きされている。最初の値は各点の `calibration_used` にだけ残る（共有メモリの p95 213.1MB・専有の p95 1,247.7MB。warm〜A-U の 5 点）。

## 点の順（`drive2_<label>.log`。時刻は driver の run／done の行）

| 順 | ラベル | run | done | 結果 | コミットの山 |
|--:|:--|:--|:--|:--|:--|
| 1 | warm | 01:41:52 | 01:43:03 | rc 0・smoke | 54.39GiB（39.3%） |
| 2 | smoke_c640 | 01:45:36 | 01:46:59 | rc 0・smoke。`calib.py report` がクリップごとの表を出した（`report.md`） | 55.55GiB（40.2%） |
| 3 | cb1080 | 01:49:31 | 01:53:53 | rc 0・baseline | 59.21GiB（42.8%） |
| 4 | A1 | 01:56:23 | 02:49:11 | rc 0・comfortable | **66.78GiB（48.3%）**。夜の山（`commit_log.csv` 02:06:00） |
| 5 | A-U | 02:51:41 | 02:58:16 | rc 0（台本は完走）・点は oom（`oom_vram16:clip0`・原因 denoise） | 63.79GiB（46.1%） |
| — | 自動復帰 | 03:00:47 | 03:02:58 | 次の点 cb1440 の前に `pipeline_loaded=false` を検出 → load（8.0 秒）→ idle-calib 1 分 → `warm_rec1108030159` rc 0 → 冷却 150 秒（`drive2_cb1440.log`・`recover_output.txt`） | — |
| 6 | cb1440 | 03:05:28 | 03:13:59 | rc 0・baseline | 58.30GiB（42.1%） |
| 7 | B1 | 03:16:30 | 04:07:19 | rc 0・comfortable | 63.85GiB（46.1%） |
| 8 | A1_r2 | 04:09:49 | 05:02:01 | rc 0・comfortable（A1 が快適だったので計画 §4.2 の順 7 は A1 #2） | 63.81GiB（46.1%） |
| 9 | B1_r2 | 05:04:37 | 05:55:22 | rc 0・comfortable | 63.86GiB（46.2%） |
| 10 | cb720 | 05:57:53 | 06:00:04 | rc 0・baseline | 53.76GiB（38.9%） |
| 11 | C1 | 06:02:35 | 06:27:57 | rc 0・comfortable | 61.43GiB（44.4%） |
| 12 | b1080 | 06:30:28 | 06:32:53 | rc 0・baseline（Single） | 54.35GiB（39.3%） |
| 13 | D-K3 | 06:35:24 | 07:02:21 | rc 0・comfortable | 62.06GiB（44.9%） |
| 14 | D-K3_r2 | 07:04:51 | 07:31:48 | rc 0・comfortable | 62.11GiB（44.9%） |
| 15 | E | 07:34:19 | 08:11:22 | rc 0・comfortable | 63.64GiB（46.0%） |
| 16 | cb1080v | 08:13:52 | 08:18:40 | rc 0・baseline（元動画のアップロード 08:13:53） | 55.57GiB（40.2%） |
| 17 | F | 08:21:11 | 09:15:28 | rc 0・comfortable | 66.73GiB（48.2%） |

- A0・A0_r2・A0_328・B1_175 は打っていない（A1・B1 が崩れなかったので計画 §4.2 の分岐に入らなかった）。予備の B-U・D-K4 も打っていない。
- 止める条件（smoke の解析の失敗・cap が帯の外・コミット 95%）には一度も当たっていない。全点の cap は 13,999 か 14,025MiB（帯 13,999〜14,033 の内）。

## 終了（09:15〜）

- 09:16:19 `commit_log.csv` の最後の行（24.09GiB・17.4%）。
- 復元: `state.json`（リポジトリ直下）の `active_base_model` と 3 系統の `selections` は `original_state.json` と全一致（較正台担当が 2026-11-08 朝にファイルで確認）。restore-state の出力・unload・`/status` の確認はこのフォルダにファイルが無い（監督の観測: restore-state 全一致・unload 済み）。
