# MiniMax H3 の fp8 と 4 ビット系の重み（2026-11-12）の出典

台帳 `Docs/PENDING_TASKS.md` §1-283 の出典となる調査報告と、それをもとにした比較資料を置く保管庫。調査日はいずれも 2026-11-12。

- `R_A_読み込み経路の棚卸し.md` — 本製品の H3 の重みの読み込み経路を、コード（dev `abc8eea`）で棚卸しした報告。Eros の beta5 の 2 本が拒否される理由、fp8 と 4 ビットが今の経路のどこで止まるか、LTX 側の前例を扱う。
- `R_B_配布形式の調査.md` — 出回っている fp8 と 4 ビット系（w4a8 など）の配布形式と、ComfyUI と comfy-kitchen が扱う形式の一覧を、ウェブと重みのヘッダで調べた報告。
- `R_C_枝刈り版の流通と検証用ファイル.md` — 枝刈り版と非枝刈り版の流通の割合、検証用にダウンロードするファイルの正確な名前と大きさとハッシュ、Ampere（RTX 30 系）での対応を調べた報告。
- `R_D_カーネルと復元の数式と層の条件.md` — comfy-kitchen のカーネルと w4a8 の復元の数式、LTX 側の前例、H3 の新しい層が満たすべき条件、重みのメモリの計算を、コードで調べた報告。
- `R_E_速度と品質の裏取り.md` — fp8 と w4a8 の速度と品質の数字、ComfyUI の動き方、comfy-kitchen のライセンスと版、ComfyUI 以外の実行環境を、ウェブと一次資料で調べた報告。
- `実行方式の比較と推奨.md` — R_A〜R_E をもとに、fp8 と w4a8 の実行方式の候補を比べ、監督役の推奨をまとめた資料。オーナーが読んで方式を選ぶための文書。

英語の原文は調査時のセッション限りの作業ファイルで、保管していない。ここにある日本語版を保管用の正本とする。

## スパイク（2026-11-12）の計画と結果の写し

製品コードを変えずに台本の中だけで fp8 と w4a8 の試作の層を差し込んだ GPU 実験の記録。結論と数値の正本は記録簿 `Docs/VERIFICATION_LOG.md` §210、決めたことは設計正本 `Docs/MINIMAX_H3_DESIGN.md` の裁定 101。元のフォルダは git の追跡外の `outputs/h3-fp8-w4a8-spike-2026-11-12/`（動画とログはそちらにだけある）。

- `スパイク計画.md` — 承認されたスパイクの計画（仮説 H1〜H6・方法・合否の基準）。冒頭の状態の行は承認前の書きぶりのまま。
- `DESIGN_spike.md` — 台本の実装担当向けの設計（英語）。S1 の参照の作り方を直した CORRECTION の注記を含む。
- `h3_quant_spike.py` — スパイクの台本（`.venv-engine-h3` でプロセスの中から動かす）。
- `dry_run.json` — 乾き試し（GPU なし）の記録。4 本の重みの検査・鍵の突き合わせ・dtype の監査。
- `s4_result.json` — S4（LTX 側の判定器 `sft_quant_format.layer_schemes` を 3 本に当てた結果）。
- `s1_result.attempt1.json` — S1（層単位の一致）の 1 回目。fp8 の fc2 で参照と食い違った版。
- `s1_result.json` — S1 の再実行（参照を直した版。全件合格）と S1'（使われた計算部品）。
- `s2_result.json` — S2（ブロック 10 の層ごとの速度と一時メモリ、1 ステップの見積り）。
- `s3_all.json` — S3（通しの生成）の 5 条件の終了状態の一覧。
- `s3_int8_result.json`・`s3_fp8_result.json`・`s3_w4a8_kijai_result.json`・`s3_w4a8_eros_result.json`・`s3_fp8_dequant_result.json` — S3 の条件ごとの記録（条件・検査の要約・段ごとの時間と VRAM・呼び出し回数と自己検査）。
- `s3_summary.md` — S3 の 5 条件の表（英語の見出し）。
- `hf-round16_README.md` — Hugging Face の `round16-h3-quant/` に上げた README（オーナーの視聴の判定の行を含む）。
- `sheet_all.png` — 5 条件の並び画像（行が条件、列がフレーム 0・36・72）。

結果の JSON の写しでは、GPU の使用状況の確認で取った起動中のアプリの一覧（`gpu_check.nvidia_smi_compute_apps`）を個人情報の保護のため取り除き、利用者名を含むパスは `<user>` に置き換えた。生の値は `outputs/h3-fp8-w4a8-spike-2026-11-12/`（git 無視）にある。

## 製品実装（2026-11-12）の計画と GPU の門の写し

スパイクの後に製品のコードへ載せた段（弾 1・弾 2）の計画と、MCP 経由で製品を通した GPU の門（G1〜G6 と G5b の 7 条件）の記録。結論と数値の正本は記録簿 `Docs/VERIFICATION_LOG.md` §211、仕組みの正本は設計正本 `Docs/MINIMAX_H3_DESIGN.md` §5・§7。元のフォルダは git の追跡外の `outputs/h3-fp8-w4a8-gate-2026-11-12/`（動画とログ・条件ごとの `metadata.json` はそちらにだけある）。

- `製品実装の計画.md` — 監督の製品実装の計画（検出・層・読み込みのときの判定・Settings・検証・文書）。冒頭の状態の行は承認前の書きぶりのまま。RTX 20 系の w4a8 の拒否をサブ検証機で確かめるとした §5.2 の一文は誤りで、訂正は記録簿 §211.2。
- `h3_quant_gate_api.py` — GPU の門の台本（アプリの仮想環境の python で、一時サーバーを立てて MCP から回す）。
- `門の台本README.md` — 台本の回し方・条件・記録するもの・自己検査の説明（元の `README.md`）。
- `gate_summary.md` — 7 条件の数値の表と、条件ごとの自己検査の結果（元の `summary.md`）。
- `gate_summary.json` — 同じ内容の JSON（元の `summary.json`。実行の記録を含む）。
- `gate_runs/<条件>/result.json` — 条件ごとの記録（`int8`・`fp8`・`w4a8_kijai`・`w4a8_eros`・`int8_oldmarker`・`int8_oldmarker_s20`・`fp8_turbo`）。
- `hf-round17_README.md` — Hugging Face の `round17-h3-quant-product/` に上げた README（監督の所見を含む）。
- `gate_sheet_all.png` — 7 条件の並び画像（行が条件、列がフレーム 0・36・72）。

`gate_summary.json` の写しでは、GPU の使用状況の確認で取った起動中のアプリの一覧（`nvidia_smi_compute_apps`）とプロセスの起動行（`cmdline`）を個人情報の保護のため `(removed for privacy)` に差し替え、利用者名を含むパスは `<user>` に置き換えた。生の値は元のフォルダにある。
