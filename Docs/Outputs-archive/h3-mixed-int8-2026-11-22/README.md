# MiniMax H3 の混在 int8 の重み（2026-11-22）の出典

台帳 `Docs/PENDING_TASKS.md` §1-290（一部のブロックを BF16 のまま残した int8 の重みと、adaln_proj まで int8 にした重みを動かす修正）の出典となる調査報告・計画・実機の確認の記録を置く保管庫。調査日と実行日はいずれも 2026-11-22。結論と数値の正本は記録簿 `Docs/VERIFICATION_LOG.md` §221、決めたことは設計正本 `Docs/MINIMAX_H3_DESIGN.md` の裁定 105 にある（ここには書かない）。

## 調査報告と計画

調査時のセッション限りの作業ファイルを、中身を変えずに写したもの。文中の「同じフォルダの `*.py`・`*.txt`」（調査の台本と生の出力）は写していない。報告の中の X・Y・Z・R は条件の記号で、X＝`s3_raw_int8`、Y＝`Sulphur-3_Ref2VA-hybrid_mid_int8_convrot_keepB0-1-49`、Z＝`10Eros_Max_h3_8stepTURBO_beta7_int8HQ`、R＝公式の枝刈り版 Ref2VA の int8。

- `A_code_inventory.md` — 製品の H3 の重みの検査・差し替え・VRAM 16GB mode の経路を、Sulphur-3 の int8 版 2 本（X・Y）に照らしてコードで棚卸しした報告。
- `A2_adaln_and_noconvrot.md` — int8 の adaln_proj の扱いと、ConvRot なしの int8 の経路を、本製品と ComfyUI のコードで調べた報告。
- `B_header_survey.md` — X・Y と、公式の重み・10Eros の 2 本など手元の H3 の重み 9 本のヘッダ（ファイル先頭の目録部分）の実測と、製品の検査器にかけた結果。
- `E_adaln_numeric.md` — int8 の adaln_proj の数値照合（CPU だけ。修正の前）。
- `D_vram_fermi.md` — BF16 のブロックを含む重みの VRAM のフェルミ推定（実測なし）。
- `計画.md` — 承認済みの監督の計画（コードの変更 2 か所・採らなかった案・確認の段取り）。
- `F_marker_inventory.md` — 快適上限マーカーを重みの形式ごとに切り替える仕組み（判定・配り方・操作パネルの経路・線の値の作り方・暫定の線の候補と印の位置・影響）をコードで棚卸しした報告。続きの別件（台帳 `Docs/PENDING_TASKS.md` §1-292。結論と数値の正本は記録簿 §222、決めたことは設計正本の裁定 106）の出典。文中の「同じフォルダの `marker.py`・`hdr.py`」（印の位置の計算とヘッダの読み取りの台本）は写していない。
- `計画_暫定の快適上限の線.md` — 上と同じ別件の、承認済みの監督の計画（BF16 のブロックを含む重みに、概算による暫定の線を出す）。計画の表の「2560×1472 の従来の印 209／192」は誤りで、正しくは 192／192（記録簿 §222.5）。

## 実機の確認（GPU の門）の写し

MCP 経由で製品を通して回した 6 条件の記録。元のフォルダは git の追跡外の `outputs/h3-mixed-int8-gate-2026-11-22/`（動画・ログ・条件ごとの `metadata.json` と並び画像はそちらにだけある）。条件の記号は、R0＝公式の int8 ConvRot（修正の前）、R1＝R0 と同じ条件（修正の後）、X・Y・Z＝上の 3 本（修正の後）、YL＝Y に Turbo LoRA を当てたもの（修正の後）。

- `h3_mixed_int8_gate_api.py` — 門の台本（アプリの仮想環境の python で、一時サーバーを立てて MCP から回す。`outputs/h3-fp8-w4a8-gate-2026-11-12/h3_quant_gate_api.py` の写しから作った）。回すときは元のフォルダのものを使う。
- `summary.md` — 6 条件の数値と自己検査の結果。
- `summary.json` — 同じ内容の JSON（実行の記録を含む）。
- `runs/<条件>/result.json` — 条件ごとの記録（`R0`・`R1`・`X`・`Y`・`Z`・`YL`）。
- `sheet_all.png` — 6 条件の並び画像。

JSON の写しでは、GPU の使用状況の確認で取った起動中のアプリの一覧（`nvidia_smi_compute_apps`）とプロセスの起動行（`cmdline`）を、個人情報の保護のため `(removed for privacy)` に差し替えた。生の値は元のフォルダにある。
