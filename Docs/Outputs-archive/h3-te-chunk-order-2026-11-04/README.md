# 台帳 1-275（MiniMax H3 の TE 段の VRAM 溢れの回避）の門の記録の複写（2026-11-05）

TE〔テキストエンコーダー〕の注意の塊の**計算順**を「最も高い塊を最初に」に変えた修正（第 2 版）の門の記録と、その前に試して不採用にした第 1 版（塊の等分）の記録のうち、文字の記録だけをここへ写しました。**結論と数値の正本は記録簿 [`../../VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §197** です。裁定は設計正本 [`../../MINIMAX_H3_DESIGN.md`](../../MINIMAX_H3_DESIGN.md) の裁定 93、台帳の記録は [`../../PENDING_TASKS_CLOSED.md`](../../PENDING_TASKS_CLOSED.md) 1-275 にあります。再較正の記録の複写は [`../comfort-calib-h3-2026-11-04-te/`](../comfort-calib-h3-2026-11-04-te/) です。

## 元のフォルダの名前について

一次記録は、git 管理外の **`outputs/h3-te-equal-chunks-2026-11-04/`** にあります。名前は第 1 版（塊の等分）のときに付けたもので、中身は第 1 版と第 2 版の両方です。ログの中の参照を壊さないため、ディスクの名前は変えていません。この複写は中身に合わせて `h3-te-chunk-order-2026-11-04` の名前にしました。下の表のパスは、元のフォルダの中のパスと同じです（`README_workspace.md` だけは元の `README.md` の名前を変えて写しました）。

## 第 1 版と第 2 版の経緯

1. **第 1 版（塊の等分）**: 塊の高さを等しく割り直しました（高さ 256〜512 行）。合成入力の T4（製品の注意の関数と PyTorch 標準の注意を全行で比べる台本）は 13 本とも一致しましたが、実モデルの TE の隠れ状態を修正前と比べる t2 で一致しませんでした（数値は記録簿 §197.2）。本体は `git apply -R` で修正前に戻し、記録は `v1_equal_split/` に移しました（一覧と sha256 は `v1_equal_split/MOVED.txt`）。
2. **第 2 版（最も高い塊を最初に計算）**: オーナー裁定（2026-11-04 深夜）で、塊の境目と高さは変えず、計算の順番だけを変える方針に差し替えました。修正前の基準（`pre_*`）は第 1 版のときに取ったものをそのまま使いました。門はすべて合格です（記録簿 §197.4・§197.5）。

## 写したもの

`.md` と `.txt` の先頭には、複写の断り書きを 1 行足しました。その後ろは元のファイルのままです。ただし `*.sha256.txt` は `sha256sum -c` で照合できるよう、断り書きを足さずに中身のまま写しました。`.patch`・`.py`・`.sh`・`.json` も中身のままです。

| ファイル | 中身 |
|---|---|
| `README.md` | 本ファイル（経緯・写したもの・写さなかったもの・正本の所在） |
| `RESULTS.md` | 数値の集計と仕組みの考察（結論・門の表・第 1 版の記録・再較正・考察・留保・所在） |
| `README_workspace.md` | 元の作業場所の説明（台本の使い方。元の名前は `README.md`。見出しの「等分」は第 1 版のときの名残） |
| `gates.md` | 合否表（`compare_gates.py` の出力。T4・t2・worker の前後・G5） |
| `t4_summary.md` | T4 の 13 本の一覧（`summarize_t4.py` の出力） |
| `fix_v2.patch`・`fix_v2.patch.sha256.txt` | 第 2 版の変更（`engine_h3/text_encoder.py` と `tests/test_h3_engine.py`）とその sha256 |
| `git_state_pre.txt`・`git_state_post_apply.txt` | 第 2 版の前確認（S0′）と適用の後（S2′）の git の状態 |
| `pre_pictures9.json`・`pre_video15.json`・`pre_i1080.json`・`pre_i1080.mp4.sha256.txt` | 修正前の基準の報告（第 1 版の S1 で取得） |
| `post_pictures9.json`・`post_video15.json`・`post_i1080.json`・`post_i1080.mp4.sha256.txt`・`g5_on.mp4.sha256.txt` | 第 2 版の門の報告 |
| `common.sh`・`s0_precheck.sh`・`s1*.sh`・`s3*.sh`・`summarize_t4.py`・`compare_gates.py` | 門の台本 |
| `analysis/` | 集計の台本と出力（`collect_te.py`・`check_progress_te.py`・`check_gates_te.py` とその出力、`points_te.json`） |
| `v1_equal_split/fix.patch`・`fix.patch.sha256.txt` | 第 1 版（塊の等分）の変更とその sha256（不採用） |
| `v1_equal_split/post_pictures9.json`・`post_video15.json`・`t4_summary.md` | 第 1 版の門の報告（t2 の不一致の記録） |
| `v1_equal_split/MOVED.txt` | 第 1 版の結果を移した一覧と sha256 |

## 写さなかったもの

どれも原本の `outputs/h3-te-equal-chunks-2026-11-04/` にあります。

- 修正前の基準の隠れ状態 `pre_pictures9.pt`・`pre_video15.pt`、生成した動画 `pre_i1080.mp4`・`post_i1080.mp4`・`g5_on.mp4`（指紋は `*.sha256.txt` に写した）。
- T4 の各点の報告 `t4_n<N>.json`（一覧は `t4_summary.md`）、G5 の報告 `g5_on.json`、準備の `t4_dryrun_cpu.json`。
- 実行の記録 `logs/`（各段の `*.console.log`・`*.meta.log`・`runs.log`・`s2_applied_diff_v2.patch`・pytest の出力）、主メモリのコミットの記録 `commit_log.csv`、コーディング担当の納品物 `coding/`。
- 第 1 版の `v1_equal_split/` のうち、`coding/`・`logs/`・`git_state_*.txt`・`t4_n<N>.json`。
