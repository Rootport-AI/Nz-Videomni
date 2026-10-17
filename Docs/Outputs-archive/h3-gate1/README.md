# MiniMax H3 第 1 弾 門 G2〜G4 の台本と結果（DiT・int8 Linear・RoPE）

作成: 2026-10-17／コード担当 C1。`outputs/h3-gate1/` は git の管理外なので、門をやり直せるように台本と結果をここに複写した。記録の正本は `Docs/VERIFICATION_LOG.md` §165。

## 結果の要約

実行機は RTX 4070 Ti SUPER（sm_89）、仮想環境は `.venv-engine-h3`（torch 2.13.0+cu130・comfy-kitchen 0.2.37）。しきい値は監督の承認済みの値。

| 門 | 台本 | 合否 | 主な数値 | 秒 |
|---|---|---|---|---|
| G2 | `g2_int8_smoke.py` | 合格 | 量子化前に戻した重みとの相対誤差 0.97e-2（しきい値 2e-2）。末尾 64 行も 0.97e-2。行数 9537（最後のタイルが半端）も同じ | 2.4 |
| G3 | `g3_blockswap.py` | 合格 | 先読みの報告値が `"on"`。全ブロック常駐・同期・先読みの 3 通り×2 回の出力がビット単位で一致 | 2.4 |
| G4 | `g4_compare.py` | 合格 | RoPE の座標・AdaLN の表は完全一致。2 ブロックの通し計算は 0.74e-2（映像）／0.44e-2（音声）。全 50 ブロックの配線の差は 0.0（ビット一致）、int8 経路の差は最大 4.1e-2（ブロック 39）・中央値 2.6e-2（しきい値 5e-2） | 22.3 |

結果の全文は `g2_result.json`・`g3_result.json`・`g4_result.json`・`g4_stdout.log`。

## 置いてあるもの

- `g2_int8_smoke.py`: `comfy_kitchen.int8_linear` を `H3Int8Linear` 経由で CUDA 経路に流す煙試験。合成の int8 重み [28672, 5376] と入力 [9600, 5376] を使い、`engine/sft_quant/dequant.py` で bf16 に戻した重みの f32 計算と比べる。
- `g3_blockswap.py`: `H3Int8Linear` を 2 本持つ小さなブロック 50 個を、本製品の `BlockSwapService(blocks_on_gpu=8, hold_arenas=True)` に載せる試験。
- `g4_compare.py`: 本実装の DiT の部品と、DiffSynth-Studio の写し（正解）を比べる。比べるのは A 座標・B AdaLN の表・C テキストの前処理と埋め込みと最終層・D 2 ブロックの通し計算・E 全 50 ブロック（1 ブロックずつ GPU に載せる）。
- `oracle/diffsynth_h3/`: 正解側を import するための置き物。
  - `core/attention.py`（SDPA の代役）・`core/gradient.py`（素通し）・各 `__init__.py`。
  - `pipeline_coords.py`: DiffSynth-Studio `diffsynth/pipelines/minimax_h3_audio_video.py` の 593〜598 行・607〜709 行の写し（Apache-2.0）。

## DiffSynth の写し 2 本の取得（ここには複写していない）

DiffSynth-Studio（https://github.com/modelscope/DiffSynth-Studio、Apache-2.0）のコミット `4e34969` から、次の 2 本を取り出して `oracle/diffsynth_h3/models/` に置く。中身は書き換えない。

- `diffsynth/models/minimax_h3_dit.py`
- `diffsynth/models/minimax_h3_dit_comfy.py`

```
git clone https://github.com/modelscope/DiffSynth-Studio
cd DiffSynth-Studio
git show 4e34969:diffsynth/models/minimax_h3_dit.py       > <置き場>/oracle/diffsynth_h3/models/minimax_h3_dit.py
git show 4e34969:diffsynth/models/minimax_h3_dit_comfy.py > <置き場>/oracle/diffsynth_h3/models/minimax_h3_dit_comfy.py
```

## 実行の手順と条件

1. このフォルダの中身を `outputs/h3-gate1/` に複写する。台本は自分の場所から 2 つ上をリポジトリ直下とみなすので、`outputs/h3-gate1/` に置かないと動かない。
2. 上の手順で DiffSynth の写し 2 本を `outputs/h3-gate1/oracle/diffsynth_h3/models/` に置く。
3. リポジトリ直下で、PYTHONPATH をリポジトリ直下にして実行する。

```
set PYTHONPATH=<リポジトリ直下>
.venv-engine-h3\Scripts\python.exe outputs\h3-gate1\g2_int8_smoke.py
.venv-engine-h3\Scripts\python.exe outputs\h3-gate1\g3_blockswap.py
.venv-engine-h3\Scripts\python.exe outputs\h3-gate1\g4_compare.py   （--blocks 0-9 で範囲指定・--skip-e2e で D を省く）
```

- どれも GPU を使う。同時に動かすのは 1 本だけにする。G4 の VRAM のピークは約 6.2GB。
- G4 は重みのファイル `models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors` を読む。
- 各台本は既定で、自分と同じフォルダに `g*_result.json` を書く（`--report` で変えられる）。合格なら終了コード 0。
- しきい値は各台本の先頭の定数（G2 `REL_TOL`、G4 `WIRING_TOL`・`QUANT_TOL`・`E2E_TOL`・`PART_TOL`）。

## 数値の読み方

- G2 の約 1e-2 は、活性化を int8 にすることの誤差。乱数の入力では、理論上の見積もりが約 0.9e-2 になる。
- G4 の E の「配線の差」は、同じ量子化前の重みを使って本実装と正解を比べたもの。0.0 なら、qkv と fc1 の分け方・AdaLN の添字・RoPE が正解と同じ計算になっている。「int8 経路の差」は、それに活性化の int8 化の誤差を加えたもの。どちらも残差の増分（出力 − 入力）で測っている。

## 追加で保管したもの（監督が複写。2026-10-17）

- `g0_g1.json`: G0（import の疎通・CUDA 13.0）と G1（comfy-kitchen の CUDA 経路に `int8_linear`・`probe_sage`）の結果。
- `g4_te_compare.py`・`g4_te_result.json`: テキストエンコーダの数値照合（層 0／1／49 の int8 と bf16 版の差・層オフロード on/off のビット一致・トークン列の一致）。
- `g5_t2v.json`（修正前の基準・2 ラウンド sha256 一致 `2a09ee8e…`）・`g5_t2v_sage.json`（sage）・`g5_t2v_after_review.json`（コードレビュー後の新基準 `0da6111c…`）・`g5_bisect_old_vae_cast.py`／`.json`（sha256 の差が映像 VAE の精度の島の型変換の順番だけに由来することの切り分け）。
- mp4 とログは `outputs/h3-gate1/`（git 管理外）にだけ置く。記録の正本は `Docs/VERIFICATION_LOG.md` §165。
