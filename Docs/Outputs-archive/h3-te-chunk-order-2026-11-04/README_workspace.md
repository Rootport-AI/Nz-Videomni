> **このファイルは実機の `outputs/h3-te-equal-chunks-2026-11-04/README.md`（git追跡外）のスナップショットである（2026-11-05複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# 台帳 §1-275 の門（TE の注意の塊の等分）の作業場所

計画の正本は `C:\Users\phenotype nezu\.claude\plans\smooth-watching-lobster.md` の §5（合否は §5.4、止める条件は §9）。この README は台本の使い方だけを書く。

## 第 2 版（塊の境目は変えず、最も高い塊を最初に計算）の流れ（計画 §5.0。2026-11-04 夜）

第 1 版（塊の等分）は S3-c の t2 で修正前とビットで一致せず止めた。その結果は `v1_equal_split/` に移した（一覧は `v1_equal_split/MOVED.txt`）。修正前の基準（`pre_*.pt`・`pre_*.json`・`pre_i1080.*`・`logs/s1*`）は第 2 版でもそのまま使うので、ここに残した。

| 段 | 担当 | 台本 | 本体のコード |
|---|---|---|---|
| S0′ 前確認 | GPU 計測担当 | `s0_precheck.sh`（patch は `fix_v2.patch`・`fix_v2.patch.sha256.txt`） | 修正前 |
| S2′ 適用 | 監督 | `git apply fix_v2.patch`・本体で H3 venv の pytest | 修正後になる |
| S3-a → S3-c → S3-d → S3-e → `compare_gates.py` | GPU 計測担当（監督の合図の後） | 下の表と同じ台本 | 修正後 |
| S4 アプリ venv の pytest 全体 | 監督 | — | 修正後 |

- S1 は流さない（第 1 版の S1 の基準を使う）。
- S3-a（T4）は「順番を変えた製品の出力と stock SDPA の全行一致」を直接確かめる門。
- S3-c 画像 9 枚（pictures9）の TE 段の予約の見込みは約 6.6〜7.1GB（修正前の 8,914MiB から下がる）。動画 15 秒（video15）は再現性の対照として流す。
- 合否表は同じ `compare_gates.py` で `gates.md` に出す。

## 段の順（第 1 版。参考）

| 段 | 担当 | 台本 | 本体のコード |
|---|---|---|---|
| S0 前確認 | GPU 計測担当 | `s0_precheck.sh` | 修正前 |
| S1-a・S1-b・S1-c 修正前の基準 | GPU 計測担当 | `s1a_t2_pictures9_save.sh` → `s1b_t2_video15_save.sh` → `s1c_worker_i1080_pre.sh` | 修正前 |
| **ここで止まって監督へ報告** | | | |
| S2 適用 | 監督 | `git apply --check fix.patch` → `git apply fix.patch`、sha256 の記録、本体で H3 venv の pytest | 修正後になる |
| S3-a〜S3-e 修正後の門 | GPU 計測担当（監督の合図の後） | `s3a_t4_all.sh` → `s3c_t2_compare.sh` → `s3d_worker_i1080_post.sh` → `s3e_g5_fingerprint.sh` → `compare_gates.py` | 修正後 |
| S4 アプリ venv の pytest 全体 | 監督 | — | 修正後 |

- どの台本も Git Bash から `bash <台本>` で呼ぶ（場所はどこからでもよい。台本が本体のフォルダへ移る）。1 条件＝1 プロセス。
- 段の間は AviUtl2 とサーバーを閉じたまま。S1・S3 の間は主メモリのコミットを記録する: `powershell -NoProfile -ExecutionPolicy Bypass -File S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\comfort-calib-h3-2026-11-04\commit_monitor.ps1 -Out S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\h3-te-equal-chunks-2026-11-04\commit_log.csv`（裏で起動し、段の後に止める）。
- S1-c・S3-d は起動から数分のうちに `logs/<台本名>.console.log` に `encode: 18462 text tokens` が出るのを確かめる。違う数なら打ち切って報告（§9）。

## 台本の役割

| ファイル | 役割 |
|---|---|
| `common.sh` | 各台本が読み込む共通部分。パス（bash 側 `/s/...`、Python 側には `cygpath -w` で作る `S:\...` の絶対パス）、環境変数（`PYTHONPATH=<本体>`・`PYTHONUTF8=1`・`PYTHONDONTWRITEBYTECODE=1`・`PYTHONUNBUFFERED=1`）、`run_logged`（実行時刻・コマンド・`text_encoder.py` の状態・終了コードを `logs/` に残す）、S1-c と S3-d が共有する worker の引数 `i1080_cmd` |
| `s0_precheck.sh` | GPU の計算プロセス・aviutl2・HEAD・porcelain・`text_encoder.py` が HEAD と同じか・日本語のプロンプトが Python へ化けずに届くか・`fix.patch` の sha256・`git apply --check`（読むだけ）。結果は `git_state_pre.txt` |
| `s1a_t2_pictures9_save.sh`・`s1b_t2_video15_save.sh` | t2 の製品の TE（`--mode product --cap`）で修正前の隠れ状態を `pre_<refs>.pt` に保存。報告は `pre_<refs>.json` |
| `s1c_worker_i1080_pre.sh`・`s3d_worker_i1080_post.sh` | worker 直呼びで i1080 の TE の形（1920×1088・画像 9 枚・N＝18,462、56 フレーム・sdpa・`--vram16`・1 ラウンド）。`<pre|post>_i1080.mp4`・`.json`・`.mp4.sha256.txt` |
| `s3a_t4_all.sh` | T4 を 13 本（N＝513〜6,148）。`t4_n<N>.json`。最後に `summarize_t4.py` の表を `t4_summary.md` へ |
| `s3c_t2_compare.sh` | t2 を修正後のコードで流し、S1 の `.pt` と比べる。`post_<refs>.json` |
| `s3e_g5_fingerprint.sh` | G5 の T2V の指紋（`--vram16` の 1 本）。`g5_on.*`。期待値との比較を `g5_on.mp4.sha256.txt` |
| `summarize_t4.py` | T4 の 13 本の `product.equal`・`differing_row_count` の一覧 |
| `compare_gates.py` | §5.4 の合否表を `gates.md` に（T4・t2・worker の前後・G5） |

## 判定に使わない欄

- T4 の `tail_rows_of_last_block`・`fold_check`・`position_check` は旧規則（端数の併合）用の欄。`position_check` は 4 行の塊を測るので必ず不一致になる（記録簿 §182.6）。判定は `product.equal` と `product.differing_row_count` だけ。
- worker の TE 段の秒数は記録だけ（AviUtl2 を閉じて流すので溢れの有無が修正前でも変わりうる）。
- t2 の `num_alloc_retries`・`decoder_peak_reserved_mb`・encode 秒は前後を並べて記録するだけ。

## 置き場所

結果はすべてこのフォルダ直下（json・pt・mp4・`git_state_pre.txt`・`t4_summary.md`・`gates.md`）。コンソールと実行の記録は `logs/<名前>.console.log`・`logs/<名前>.meta.log`、1 行ずつの一覧は `logs/runs.log`。`coding/`・`fix.patch`・`fix.patch.sha256.txt` はコーディング担当の納品物（触らない）。

準備の段（GPU なし）で `t4_dryrun_cpu.json`・`logs/t4_dryrun_cpu.*` を作った（`--n 70 --block 16 --cpu`。鍵の名前の確認用で、門ではない。CPU では一致しないのが普通）。
