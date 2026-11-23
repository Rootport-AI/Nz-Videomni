# MiniMax H3 の高速な復号器 Sol-H3-VAE の実験（2026-11-23）の出典

Sol-H3-VAE（MiniMax H3 の動画VAE の復号器を 24 層に蒸留した、復号専用の軽い復号器）を、製品のコードを変えずに実験用の台本で動かし、NVIDIA の基準の実装との照合・画質の比較・速さと VRAM の計測を行った実験の、台本と記録の写しを置く保管庫です。台帳には無い、オーナーの直接の指示によるテーマです。結論・数値・オーナーの判定の正本は記録簿 `Docs/VERIFICATION_LOG.md` §228、製品にするときの裁定と仕組みの正本は設計正本 `Docs/MINIMAX_H3_DESIGN.md` の裁定 108 と §6.8 にあり、ここには書きません。

**これは 2026-11-23 の時点の写しです。** 元のフォルダは git の追跡外の `outputs/h3-sol-spike-2026-11-23/` で、正本は常にそちらです。回すときは元のフォルダの台本を使ってください（使い方は `SCRIPT_README.md` と `h3_sol_spike.py` の冒頭の説明にあります。GPU を使う項目は、その都度オーナーの了承を得てから回します）。

## 台本（中身を変えずに写したもの）

- `h3_sol_spike.py` — 実験の本体。項目ごとに単独で実行できます（GPU を使わない確認 `--dry-run`、GPU を使う前の確認 `--gate`、関数の確認 `--selfcheck`、NVIDIA の実装との照合 `--verify`、潜在の生成 `--gen`、画質の比較 `--compare`、速さと VRAM の計測 `--bench`）。
- `sol_decoder.py` — 実験用の Sol の復号器。構成 A・B・C と、照合用の fp32 の経路を持ちます。NVIDIA の Apache-2.0 の実装を読んで書いたもので、ComfyUI と Unsloth のコードは写していません。
- `sol_metrics.py` — 画質の指標（PSNR・SSIM・128 画素の格子の位置での縁の強さ・ちらつき）と、拡大の比較画像を作る部分。
- `wddm_sampler.ps1` — 共有 GPU メモリの採取（快適上限の較正の台本と同じもの）。
- `_summ_cmp.py`・`_summ_gen.py` — 記録（`runs/compare_*`・`runs/gen_*` の `result.json`）から一覧を画面に出すだけの補助。
- `SCRIPT_README.md` — 元のフォルダの `README.md` の写し（ファイルの役割・実行の順・出力の場所）。

## 記録の写し

- `conditions_table.md` — 潜在を生成した条件と、操作パネルの既定との突き合わせの表。
- `dry_run.json` — GPU を使わない確認の結果（12 項目）。
- `runs/<項目>/result.json` — 項目ごとの記録。項目は `gate`（GPU を使う前の確認）、`selfcheck`（関数の確認）、`verify`（照合）、`gen_a`〜`gen_e`（潜在の生成）、`compare_a`〜`compare_e`（画質の比較）、`bench`（速さと VRAM の 33 点）です。
- `ref_nvidia/NOTICE.md` — 照合の基準に使った NVIDIA の実装の出どころ（リポジトリ・パス・SHA-256・ライセンスの表示）の注記。

写したファイルは、元のファイルとバイト単位で同じです。台本は、GPU を使っているプロセスの数だけを記録し、記録に書くパスから利用者の名前を消す作りです。写す前に、利用者名・利用者のフォルダーを含むパス・メールアドレスが含まれていないことを確かめました（`h3_sol_spike.py` には、名前を消す処理の試験のための架空の名前の例があるだけです）。

## 写さなかったもの

- **NVIDIA のコードの写し**（`ref_nvidia/decoder.py`）は写していません。照合の基準として実行するためだけに元のフォルダに置いたもので、出どころは `ref_nvidia/NOTICE.md` にあります。
- 潜在（`runs/gen_*/latent.safetensors`）、動画（mp4）、画像（拡大の比較画像）、ログ（`*.log`・`*_console.log`）、共有 GPU メモリの採取の生の記録（`runs/bench/wddm_*.csv`）は写していません。大きすぎるか、リポジトリに入れない慣行の物です。元のフォルダにだけあります。
- d の素材（縦長のイラスト）の画像は写していません。
