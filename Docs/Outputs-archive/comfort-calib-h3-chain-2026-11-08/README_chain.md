> **このファイルは実機の `outputs/comfort-calib-h3-chain-2026-11-08/README_chain.md`（git追跡外）のスナップショットである（2026-11-08複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# MiniMax H3 の連結生成（Chained）の快適上限の較正台（台帳 §1-276）

計画の正本は `C:\Users\phenotype nezu\.claude\plans\cozy-gliding-pudding.md` の §4（点の一覧と実行順は §4.2、判定は §4.1 の「判定（v3″ の連結版）」、止める条件は §4.3）。ここには台本の使い方と、11-04 の較正台（`outputs/comfort-calib-h3-2026-11-04-te/`。読むだけ）からの差分だけを書く。

## 結論（準備の状態。2026-11-08 夜の前・GPU は使っていない）

- 計画 §4.1 の S0〜S5 は全段の受け入れ条件を満たした（各段の出力は `stage_logs/`、最終の出力は `selftest/*_output.txt`）。
- §4.2 の全点の計画ファイル（`plans/p_<label>.json`。25 本）は事前点検（preflight）がすべて rc 0。予測の行数 S は計画 §4.2 の表とすべて一致（`selftest/plan_table_output.txt`）。E のクリップ 1 は実測のトークナイザで 217,167（計画の表の値と同じ。操作パネルの上界式の 217,177 ではない）。
- Single の経路（腕 D）は変えていない。11-04 の Single の計画 12 本（`plans_te/`）の事前点検と送信内容の下書き（dry-run）が 11-04 の出力と一致することを各段で確かめた（`selftest/single_regress.py`）。

## 1 点を回す手順

11-04 と同じ（README_te の「点の順」の手順をこのフォルダで行う）。各点は次の 4 手。

1. `plans/p_<label>.json` を目で確認する（形・K・基準点・予測の S）。
2. `bash cool.sh`（前の点の終わりから 150 秒待つ）。
3. `bash rp.sh <label> > rp_<label>.txt 2>&1`（裏で走らせる。事前点検の rc が 0 でない点は打たない。OOM の後は drive2.sh が自動で読み直す〔`plans/p_warm.json` を使う〕）。
4. `bash wait.sh <label>`（最大 570 秒。点が長いときは繰り返す）→ `progress.md` に 1 行。

`rp.sh` の最後に `sum.py` が出す連結の点の要約は、クリップごとに 1 行（行数の大きいクリップが先頭）と、窓ごとの判定（`clip<i>.encode`〔記録だけ〕・`clip<i>.denoise`・`decode`）。全点の表は `calib.py report`（連結はクリップごとの表）、記録用の表は `make_results.py --h3`。

### 点の名前と基準点の対応

| 腕 | 基準点 | 点 |
| :-- | :-- | :-- |
| A（1920×1088） | `cb1080` | `A1`・`A1_r2`（2 回目）・`A-U`・`A0`・`A0_r2`・崩れたとき用 `A0_328`・`A0_328_r2` |
| B（2560×1472） | `cb1440` | `B1`・`B1_r2`・予備 `B-U`・崩れたとき用 `B1_175`・`B1_175_r2` |
| C・E（1280×768） | `cb720` | `C1`・`E`（クリップ 1 に 15 秒の参照動画 `materials/H3-Ref-1080p-15s.mp4`。11-04 の v720 の素材） |
| D（Single） | `b1080`（Ref2VA・K=1・56） | `D-K3`・`D-K3_r2`・予備 `D-K4` |
| F（入力動画の続き） | `cb1080v` | `F`（元動画は E と同じ素材） |
| — | — | 煙の点 `smoke_c640`、自動復帰用 `warm`（11-04 の `p_warm.json` と同じ中身） |

重みは全点 Ref2VA（`minimax_h3_ref2va_pruned_int8_convrot`）・sage・4 ステップ・seed 12345。2 回目の点は名前を変える（同じ名前で回すと `runs/<label>/` が上書きされる）。

### 判定の読み方

- 点の判定は `verdict_v3dprime`（快適＝`comfortable`）。非快適の理由は `v3dprime.reasons`（`oom:clip1`・`plateau:clip1.denoise`・`plateau:decode`・`prefetch_degraded:clip1` など）、原因の段は `v3dprime.causes`（`denoise` は線の 2 択に使う。`decode` は使わず §4-49 の材料）。
- クリップの番号は 0 始まり。ログの `[i/n]`（server.log）と `chain: clip i/n`（h3_worker.log）は 1 始まりなので、クリップ番号＝ログの番号−1。OOM のクリップは `v3dprime.oom_clip.clip`（復号の段の OOM はクリップなし）。
- 窓は基準点の同じ窓と比べる（`clip1.denoise` 同士、`decode` 同士）。

## 夜の運転（監督が打つ。GPU の了承の後）

`P=../../.venv/Scripts/python.exe`、作業場所はこのフォルダ。

1. オーナーが AviUtl2 と操作パネルを 11-04 と同じ形で開き、サーバー（run.bat）を起動する。
2. `$P calib.py server start --pid <PID>` → `$P calib.py luid` → `powershell -NoProfile -ExecutionPolicy Bypass -File commit_monitor.ps1 -Out commit_log.csv`（裏で走らせ続ける）。
3. `$P calib.py load --base-model MiniMaxH3 --transformer minimax_h3_ref2va_pruned_int8_convrot` → `$P calib.py idle-calib --minutes 3`。
4. `warm`（1 点を回す手順）。
5. `smoke_c640` → `$P calib.py report` で連結の表がクリップごとに出ることを確かめる。出なければその夜は止める。
6. 計画 §4.2 の順: `cb1080` → `A1` → `A-U` → `cb1440` → `B1` →（A1 が快適なら `A1_r2`、OOM なら `A0`）→ `B1_r2` →（A1 が OOM のときだけ `A0_r2`）→ `cb720` → `C1` → `b1080` → `D-K3` → `D-K3_r2` → `E` → `cb1080v` → `F`。予備は `B-U`・`D-K4`。崩れたときは計画 §4.2「崩れたときの規則」（A1 のクリップ 0 が OOM なら `A0` の代わりに `A0_328`・`A0_328_r2`／B1 のクリップ 1 が OOM なら `B1_175`・`B1_175_r2`）。
7. 毎点: 全クリップの cap（`clip <i>: ... cap_mib=`）が 13,999〜14,033MiB の帯の内か。

## 復元（README_te と同じ）

`$P calib.py restore-state`（`state.json` と `original_state.json` が全一致）→ `curl -s -X POST http://127.0.0.1:18620/api/v1/pipeline/unload` → `curl -s http://127.0.0.1:18620/api/v1/status` が unloaded・キュー空 → commit_monitor を止める → 自分の台本・待ちループの残りがない。

## 11-04 の較正台からの差分

差分の全文は `calib_chain.diff`（11-04 の `-te` フォルダとの `diff -ruN`。行末の CR は無視）。

| ファイル | 変更 |
| :-- | :-- |
| `drive2.sh`・`mk.sh`・`rp.sh`・`wait.sh`・`cool.sh`・`commit_table.py`・`digest.py` | 固定パスをこのフォルダへ（S0） |
| `calib.py` | 連結の点の関数一式（2632〜3489 行の新しい節）。入口は共有の振り分け 4 箇所（`token_counts`・`build_request`・`check_point`・`analyse`）に `is_h3_chain` の 2 行ずつ、`cmd_run` の 4 箇所（見出しの行・アップロード・再アップロード 1 回・結果の行）、`cmd_report` の表の振り分け。Single の関数（`token_counts_h3`・`build_request_h3`・`check_point_h3`・`analyse_h3`・`used_check_h3`・`h3_report_cells`・`h3_stage_window`）は 1 バイトも変えていない（連結はクリップごとにこれらを呼ぶ） |
| `mkplan.py` | `--kind chain` の H3 版（`--clips`・`--keyframes` はクリップ 0 の分・`--clip-ref i:kind:path:slot`・`--source-video`）。308 行の拒否を外した |
| `mkc.sh`（新規） | 連結の点を作って事前点検する（`mk.sh` は D 用に残した） |
| `sum.py`・`digest.py`・`make_results.py`・`commit_table.py` | 連結の行（クリップごと）。Single の行は前と同じ |
| `selftest/dryrun_validate.py` | 連結の本文を `GenerateChainRequest` → `h3_adapter.reject_chain` → `validate_chain_for_family("h3")` の順に通す。点検するフォルダを引数で選べる |
| `selftest/parser_selftest.py` | 新しい節 J（連結の 5 窓を server.log の実ジョブ e8f0951d で・番号の 1 始まり・実ジョブの metadata との行数・送信内容・事前点検の拒否）。節 H は 11-03 の参照の較正台の計画ファイルを写した `selftest/plans_fixture_h3ref/` を読むようにした（11-04 の写しは既に節 H の途中で KeyError 'r1080_277' で止まっていた） |
| `selftest/analyse_selftest.py` | CASE 4（連結の偽記録で 5 例: 全部良い→快適／クリップ 1 の OOM→番号 1／クリップ 1 の先読み off→非快適／復号の退避→原因 decode／クリップ 1 のノイズ除去の退避） |
| `selftest/single_regress.py`（新規） | Single の 12 本（`plans_te/`）が 11-04 と同じであることの点検 |
| `selftest/run_chain_mock.py`（新規） | サーバーなしで `cmd_run` の連結の経路（アップロード・再送・結果の行）と Single の 1 点を通す |
| `selftest/plan_table.py`（新規） | 全点の予測 S を計画 §4.2 の表と突き合わせる |
| `plans/`・`plans_te/` | 11-04 の Single 12 本は `plans_te/` へ移した（この較正台の D の基準点も `b1080` という名前で、11-04 の `p_b1080.json` は FL2VA の点なので同じ場所に置けない）。`plans/p_warm.json` は 11-04 と同じ中身（drive2.sh の自動復帰が読む） |

持ち込まなかったもの: 11-04 の `runs/`・`manifest.jsonl`・`commit_log.csv`・`calib/`・`reference_uploads.json`・`luid.json`・`original_state.json`・各点の出力。アップロードは夜に取り直す（1 ファイル 1 回で再利用）。

## 留意

- 本物の `phases.clips` の鍵の形と worker log の書式は、GPU なしでは過去の実ジョブ（`outputs/e5894843-*`・`31c8a99a-*`・`63138ec0-*`・`3ab502e9-*`・`7eea02e1-*` の metadata.json、server.log の e8f0951d）でしか確かめていない。夜の煙の点で確かめる（計画 §4.1）。
- 連結の 2 本目の `chain text encode 1/4 steps (-0.01 step/s, ...)` は速さが負で、共有の行の読み取り（`STAGE_PROG_RE`）が読まない。窓の境はノイズ除去の行で決まるので判定に影響はない（encode は記録だけ）。
- `keep_resident` は送らない（連結 API の既定 false）。照合ではクリップ 0 の `dit_source` が "file"、クリップ 1 以降が "resident"、`keep_resident_used` が "off" を期待する。

## 夜の実績（2026-11-08）

- 開始 01:37（開始時の状態の記録・LUID・コミットの記録）、最初の点 warm 01:41:55、最後の点 F の終了 09:15:25。結果は `RESULTS.md`、時系列は `progress.md`。
- 打った順: warm → smoke_c640 → cb1080 → A1 → A-U（クリップ 0 で OOM。cb1440 の前に drive2.sh が自動復帰 `warm_rec1108030159`）→ cb1440 → B1 → A1_r2（A1 が快適だったので A0 ではなく 2 回目）→ B1_r2 → cb720 → C1 → b1080 → D-K3 → D-K3_r2 → E → cb1080v → F。
- 復元: `state.json` の選択は `original_state.json` と全一致（ファイルで確認）。restore-state・unload は監督が実行（出力ファイルはこのフォルダに無い）。
- 打っていないもの: 予備の B-U・D-K4、崩れたとき用の A0・A0_r2・A0_328・A0_328_r2・B1_175・B1_175_r2。
