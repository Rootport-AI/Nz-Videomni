# Sol-H3-VAE の実験（2026-11-23）

MiniMax H3 の動画VAE の復号器を 24 層に蒸留した Sol-H3-VAE（以下 Sol）を、製品のコードを変えずに試作して、NVIDIA の基準の実装との照合、画質の比較、速度と VRAM の計測を行うための台本です。計画の正本は、オーナーが了承した計画のファイル（`parsed-greeting-pancake-of-328841ff-…md`）です。

## ファイル

| ファイル | 役割 |
|---|---|
| `sol_decoder.py` | 実験用の Sol の復号器（台本から独立したモジュール）。正規化済みの潜在 `[1, 24, T, h, w]` を受け、CPU の uint8 `[F, H, W, 3]` を返します。構成の記号 A・B・C（計画 5.4）と照合用の F32 を持ちます |
| `sol_metrics.py` | 画質の指標（PSNR・SSIM・128 画素の格子の位置での縁の強さ・ちらつき）と、拡大の比較画像 |
| `h3_sol_spike.py` | 台本の本体。項目ごとに単独で実行できます |
| `ref_nvidia/decoder.py` | 照合の基準（NVIDIA の実装の写し。出どころとライセンスは `ref_nvidia/NOTICE.md`） |
| `wddm_sampler.ps1` | 共有 GPU メモリの採取（快適上限の較正の台本 `comfort-calib-h3-2026-11-04` の `vram_sampler2.ps1` と同じもの） |

## 実行の順

リポジトリの最上位（`Nz-Videomni`）で、Git Bash から実行します。

```
PY=./.venv-engine-h3/Scripts/python.exe; S=outputs/h3-sol-spike-2026-11-23/h3_sol_spike.py
PYTHONPATH=$PWD PYTHONUTF8=1 $PY $S --dry-run
```

| 順 | 指定 | GPU | 中身 |
|---|---|---|---|
| 1 | `--dry-run` | 使わない（GPU を見えなくして動きます） | ファイルの有無、生成の条件の表、重みのキーの対応、小さな乱数の復号器での NVIDIA の実装との一致、位置の符号化の式、INT8 の線形層の引数の意味、差し替えの出し入れ |
| 2 | `--gate` | 使わない | GPU を使う前の確認だけを行います（製品のサーバーの待ち受け、ほかのプロセスの GPU の使用、残っているワーカー） |
| 3 | `--selfcheck` | 使う（数十秒） | `comfy_kitchen` の関数（`rms_rope_split_half_`・融合の引数つきの `int8_linear`・`int8_attention`）と、torch で書いた同じ式との比較 |
| 4 | `--verify` | 使う | 計画 5.2 の照合。既定は乱数の潜在。`--latent a` で生成した潜在から切り出せます。合格は 80dB 以上 |
| 5 | `--gen a`（b・c・d・e も 1 つずつ） | 使う | 計画 5.3。製品のエンジンで生成して潜在を保存します |
| 6 | `--compare a b c d e` | 使う | 計画 5.4・5.5。標準の復号器と構成 A・B・C で復号して比べます |
| 7 | `--bench` | 使う | 計画 5.6。速度と VRAM。途中で止まっても、もう一度同じ指定で続きから再開します（`--force` でやり直し） |

- GPU を使う項目は、どれも最初に `--gate` と同じ確認を行い、引っかかったら何も読み込まずに終了します（終了コード 2）。
- 3 の合格が、4 に進む条件です。4 は、3 の記録（`runs/selfcheck/result.json`）が合格でなければ実行せずに終了します。3 は、本番の構成 A・B・C が使う CUDA 版の `comfy_kitchen` の関数を確かめる唯一の項目です（`int8_attention` が使えない GPU では不合格になり、構成 C は回せません）。
- 4 は 5 より先に回します（計画の順。乱数の潜在で動きます）。4 は、fp32 の経路と基準の一致度が 80dB 以上で、あわせて構成 A と基準の一致度が目安の下限 40dB 以上のときに合格です。不合格なら、5 以降に進まずに原因を調べます（計画 5.2）。
- 7 は既定で潜在 a を並べて使うので、5 の a が先に要ります。乱数の潜在で測るときは `--latent-source random` を付けます。
- 生成の注意機構は製品の既定（sdpa）です。`--attention sage` で変えられます（変えたときは記録に残ります）。

## 出力の場所

`runs/` の下に、項目ごとのフォルダーができます。

| フォルダー | 中身 |
|---|---|
| `dry_run.json`・`dry_run.log`・`conditions_table.md` | 空回しの記録と、生成の条件の表 |
| `runs/gate/` | `--gate` の記録 |
| `runs/selfcheck/` | 関数の比較 |
| `runs/verify/` | 照合の結果（`result.json` の `compare_vs_ref`） |
| `runs/gen_a/` など | 潜在（`latent.safetensors`）、製品の出力の動画（`output.mp4`）、記録 |
| `runs/compare_a/` など | 復号器ごとの動画（映像だけ）、拡大の比較画像、指標 |
| `runs/bench/` | 速度と VRAM（`result.json` に点ごとに書き足し）、共有 GPU メモリの採取の記録 |

どの項目の `result.json` にも、条件、版、git の状態、時間、VRAM、合否が入ります。記録には、利用者の名前を含むパスを書かないようにしてあります。
