# Sulphur-3 の int8 版 2 本のヘッダ実測（2026-11-22）

記号: **X** = `s3_raw_int8.safetensors`、**Y** = `Sulphur-3_Ref2VA-hybrid_mid_int8_convrot_keepB0-1-49.safetensors`、**R** = `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（公式・枝刈り版 Ref2VA）、**F** = `minimax_h3_fl2va_pruned_int8_convrot.safetensors`（公式・FL2VA）。
すべて `models\MiniMaxH3\Weights\` 配下。読み取りはヘッダと必要なテンソルのバイト範囲だけ（seek）。GPU・torch は使っていない。

生データ: 同じフォルダの `survey1.json`（全 9 本のヘッダ集計）、`show2.txt`（ブロック別の表と鍵の差）、`inspect_run.txt`（製品の検査器の結果）、`values.txt`・`adaln.txt`（値の実測）。台本は `survey1.py`・`show1.py`・`show2.py`・`inspect_run.py`・`values.py`・`adaln.py`。

## 1. 全体

| ファイル | サイズ(byte) | ヘッダ長 | テンソル数 | dtype の内訳 | comfy_quant | `__metadata__` |
|---|---|---|---|---|---|---|
| X | 20,948,314,532 | 131,344 | 1032 | F32 256 / BF16 225 / F16 51 / I8 250 / U8 250 | 250 | `_quantization_metadata` のみ |
| Y | 22,104,811,854 | 136,632 | 1002 | F32 241 / BF16 240 / F16 51 / I8 235 / U8 235 | 235 | `_quantization_metadata` のみ |
| R | 20,970,379,616 | 95,416 | 932 | F32 210 / BF16 220 / F16 102 / I8 200 / U8 200 | 200 | 無し |
| F | 20,970,379,616 | 95,416 | 932 | R と同じ | 200 | 無し |
| PulpCut（FL2VA turbo） | 20,970,379,854 | 95,654 | 932 | R と同じ | 200 | base_model など 4 鍵 |
| pinkcherry v18 | 21,056,259,240 | 8,388,608 | 932 | F32 261 / BF16 220 / I8 200 / U8 200 / F16 51 | 200 | 学習情報多数 |
| h3ErosMax beta5（旧式の印） | 20,970,414,464 | 96,280 | 734 | F32 210 / BF16 324 / I8 200 | 0 | `_quantization_metadata`・format・converted_by |
| 10Eros w4a8 14gb | 13,997,668,774 | 161,984 | 1294 | BF16 334 / I8 200 / F32 380 / U8 200 / F8_E4M3 180 | 200 | 変換器情報多数 |
| 10Eros 8stepTURBO int8HQ | 22,893,150,024 | 116,328 | 892 | BF16 352 / I8 180 / F32 180 / U8 180 | 180 | 変換器情報多数 |

`_quantization_metadata`（X・Y）: `{"format_version": "1.0", "layers": {...}}`。
- X: layers 250 件、全件 `{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16"}`（convrot の鍵が無い）。
- Y: layers 235 件。188 件 `{"convrot": true, "convrot_groupsize": 256, "format": "int8_tensorwise", "orig_dtype": "torch.bfloat16"}`、47 件 `{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16"}`（adaln_proj.linear の 47 層）。

## 2. 名前の体系

- 接頭辞: 9 本とも無し（`model.diffusion_model.` などは付いていない）。
- 本体ブロック `blocks.0`〜`blocks.49`（50 個、連続）、`token_refiner.blocks.0`〜`1`（2 個）。9 本とも同じ。

ブロックごとの表（dtype, shape）。N/A は鍵が無いこと。

| テンソル（blocks.N.） | X（全ブロック同じ） | Y ブロック 0・1・49 | Y ブロック 2〜48 | R（全ブロック同じ） |
|---|---|---|---|---|
| adaln_proj.linear.weight | I8 [96768, 8] | BF16 [96768, 8] | I8 [96768, 8] | F16 [96768, 8] |
| adaln_proj.linear.weight_scale | F32 [96768, 1] | N/A | F32 [96768, 1] | N/A |
| adaln_proj.linear.comfy_quant | U8 [78] | N/A | U8 [78] | N/A |
| adaln_proj.linear.bias | F16 [96768] | F16 [96768] | F16 [96768] | F16 [96768] |
| attn.qkv_proj.weight | I8 [21504, 5376] | BF16 [21504, 5376] | I8 [21504, 5376] | I8 [21504, 5376] |
| attn.qkv_proj.weight_scale | F32 [21504, 1] | N/A | F32 [21504, 1] | F32 [21504, 1] |
| attn.qkv_proj.comfy_quant | U8 [78] | N/A | U8 [121] | U8 [72] |
| attn.out_proj.weight | I8 [5376, 7168] | BF16 | I8 | I8 |
| attn.out_proj.weight_scale | F32 [5376, 1] | N/A | F32 [5376, 1] | F32 [5376, 1] |
| mlp.fc1.weight | I8 [28672, 5376] | BF16 | I8 | I8 |
| mlp.fc1.weight_scale | F32 [28672, 1] | N/A | F32 [28672, 1] | F32 [28672, 1] |
| mlp.fc2.weight | I8 [5376, 14336] | BF16 | I8 | I8 |
| mlp.fc2.weight_scale | F32 [5376, 1] | N/A | F32 [5376, 1] | F32 [5376, 1] |
| attn.q_norm / k_norm.weight | BF16 [128] | BF16 [128] | BF16 [128] | BF16 [128] |
| norm1 / norm2.weight | BF16 [5376] | BF16 [5376] | BF16 [5376] | BF16 [5376] |

（ブロック 0・1・10・47・48・49 を並べた生の表は `show2.txt`。X と R はブロック番号によらず同じ構成、Y はブロック 0・1・49 だけが BF16。）
線形層の bias はブロック内では adaln_proj.linear.bias のみ（qkv・out_proj・fc1・fc2 に bias は無い）。bias の総数は 9 本とも 56。

## 3. 量子化の印（comfy_quant の中身、ファイルから復号）

| ファイル | 件数 | 中身（件数） |
|---|---|---|
| X | 250 | `{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "per_row": true}` ×250 |
| Y | 235 | `{..."convrot": true, "convrot_groupsize": 256, "per_row": true, "orig_dtype"...}` ×188 ／ `{"format": "int8_tensorwise", "orig_dtype": "torch.bfloat16", "per_row": true}` ×47（adaln） |
| R・F・PulpCut・pinkcherry | 200 | `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}` |
| 10Eros int8HQ | 180 | `{convrot: true, groupsize 256, int8_tensorwise, per_row: true}` |
| 10Eros w4a8 | 200 | w4a8 group_size 16 ×82・8 ×53・4 ×45、int8（convrot, per_row）×20 |
| h3ErosMax | 0 | （メタデータのみ: convrot true・256・int8 ×200） |

メタデータ式と層ごとの印の一致（X・Y）: 対象の層の集合は完全一致（X 250/250、Y 235/235）。中身は「層ごとの印にだけ `"per_row": true` がある」点だけが違い、それ以外の鍵と値は同じ。製品の検査器は `per_row` を読まないので、判定上は同じ。
`convrot` は X の全層で鍵自体が無い（検査器の既定は False）。`group_size`／`groupsize`・`full_precision_matrix_mult` の鍵は X・Y のどこにも無い。

## 4. 重みの dtype の配置

- X: qkv・out_proj・fc1・fc2・**adaln_proj.linear** の 5 種 × 50 ブロック = 250 層が I8。BF16 の線形層はブロック内には無い（norm と q/k_norm のみ BF16）。
- Y: ブロック **0・1・49 の 3 個**（「0〜1 と 49」と「0・1・49」は同じ集合。2〜48 は全部 I8）で、qkv・out_proj・fc1・fc2・adaln_proj.linear の 5 層が BF16。残り 47 ブロックは 5 層とも I8（adaln だけ convrot 無し）。
- R・F・PulpCut: 4 種 × 50 = 200 層が I8。adaln_proj.linear は F16（pinkcherry は F32、h3ErosMax・10Eros は BF16）。**adaln_proj.linear を I8 にしているのは 9 本中 X と Y だけ**。
- 10Eros int8HQ: ブロック 0・1・47・48・49 の 4 層が BF16（Y と同じく「端を BF16 で残す」形）。
- token_refiner（2 ブロック）は X・Y・R とも全部 BF16。

## 5. 倍率（weight_scale）

形: X・Y・R・F すべて F32 [N, 1]（行ごと）。スカラーや [N] は無い。X・Y は adaln 用に [96768, 1] が加わる。
ブロック 10 の qkv の weight_scale（21504 行）:

| | min | max | mean | median |
|---|---|---|---|---|
| X | 0.001669 | **0.042569** | 0.004353 | 0.004029 |
| Y | 0.001453 | 0.007358 | 0.003511 | 0.003347 |
| R | 0.001448 | 0.007377 | 0.003512 | 0.003346 |
| F | 0.001429 | 0.007317 | 0.003511 | 0.003346 |

ブロック 10 の adaln の weight_scale は X と Y で完全に同じ値（min 0.000245・max 0.02925・mean 0.003779）。

## 6. ブロック以外のテンソル

鍵の集合は X・Y・R で同じ（16 個: adaln_t_table・rope.inv_freq・video/audio_patch_proj・condition_proj・final_layer.*・token_refiner.final_norm）。`adaln_basis`・`adaln_mean`・`time_embedder.*` は X・Y・R・F のいずれにも無い（`adaln_basis`・`adaln_mean` は h3ErosMax と 10Eros w4a8 にだけある）。adaln_t_table は 3 本とも F32 [1025, 8] で、枝刈り版の形。
dtype の違い（X・Y 共通、R との差）:

| テンソル | X・Y | R |
|---|---|---|
| video_patch_proj.weight [5376, 96] | BF16 | F32 |
| audio_patch_proj.weight [5376, 32] | BF16 | F32 |
| final_layer.video_out.weight [96, 5376] | BF16 | F32 |
| final_layer.audio_out.weight [32, 5376] | BF16 | F32 |
| final_layer.adaln_proj.linear.weight [10752, 8] | BF16 | F16 |

（bias 類・condition_proj・norm・rope.inv_freq・adaln_t_table は dtype も shape も同じ。）

## 7. 鍵の集合の差

- X にだけあって R に無い: `blocks.N.adaln_proj.linear.weight_scale` ×50、`blocks.N.adaln_proj.linear.comfy_quant` ×50（計 100）。R にだけある鍵: 0。
- Y にだけあって R に無い: 上の 2 種 ×47（ブロック 2〜48）。R にだけあって Y に無い: ブロック 0・1・49 の qkv・out_proj・fc1・fc2 の weight_scale と comfy_quant（8 種 ×3 = 24）。
- X にだけあって Y に無い: ブロック 0・1・49 の 5 層の weight_scale と comfy_quant（30）。Y にだけある鍵: 0。
- R と F: 差なし（鍵の集合が同一）。

## 8. 製品の検査器（`Nz-Videomni\h3_format.py` の `inspect_h3_dit`）

コードで確認したこと: `h3_format` と `sft_quant_format` の import は標準ライブラリだけ（torch・numpy・safetensors を読まない）。検査はヘッダと comfy_quant の小さなバイト列を seek で読むだけ。実行は app 側の `.venv`（torch 無し）で、実行の前後で `sys.modules` に torch が無いことも確かめた。

結果（9 本とも受理。構成は全部 hidden 5376・50 ブロック・refiner 2・56 heads・head_dim 128・ffn 14336・text 5120・latents 24・audio 32・grid 1025・rank 8・inv_freq 16・接頭辞なし）:

| ファイル | 判定 | 量子化層 | 形式ごと | unknown | quant_class |
|---|---|---|---|---|---|
| X | 受理 | 250 | int8・convrot=False・256 ×250（adaln 50 を含む） | なし | int8 |
| Y | 受理 | 235 | int8・convrot=True・256 ×188 ／ int8・convrot=False ×47（adaln） | なし | int8 |
| R・F・PulpCut・pinkcherry | 受理 | 200 | int8・convrot=True・256 ×200 | なし | int8 |
| h3ErosMax | 受理 | 200 | 同上 | adaln_basis・adaln_mean | int8 |
| 10Eros w4a8 | 受理 | 200 | w4a8 g16 ×82・g8 ×53・g4 ×45（codebook あり）、int8 convrot ×20 | adaln_basis・adaln_mean | w4a8 |
| 10Eros int8HQ | 受理 | 180 | int8 convrot ×180 | なし | int8 |

## 9. 値の実測（ブロック 10 の qkv、I8 を 1 層ぶん全部 = 115,605,504 個）

| | min | max | std | 0 の割合 | 尖度 | 各行の最大絶対値が 127 の行 | -128 | \|q\|≤8 | \|q\|≥64 |
|---|---|---|---|---|---|---|---|---|---|
| X | -127 | 127 | 27.72 | 1.64% | 3.75 | 100% | 0 | 27.1% | 3.0% |
| Y | -127 | 127 | 33.05 | 1.22% | 3.06 | 100% | 0 | 20.5% | 5.5% |
| R | -127 | 127 | 33.05 | 1.22% | 3.06 | 100% | 0 | 20.5% | 5.5% |

倍率を掛けた値（先頭 4096 行）の相関:
- ブロック 10 qkv: Y–R **0.9997**、X–R **0.062**、X–Y **0.062**。標準偏差は 3 本とも 0.1189 でほぼ同じ。尖度は X 4.47、Y・R 3.69。
- ブロック 0 qkv（先頭 2048 行。Y はここが BF16）: X（倍率を掛けた値）–Y（BF16）**0.9999**（相対誤差 1.6%）、R–Y 0.062、X–R 0.062。尖度は X・Y 5.51、R 4.75。

AdaLN（adaln_t_table と adaln_proj）:
- adaln_t_table: X と Y は完全に同一、R とは違う。列ごとの相関（X 対 R）は列 0〜3 が +0.996〜1.0、列 4〜7 が −0.991〜−0.9997（符号が反転）。列のノルムは X・R でほぼ同じ（7.08・2.09・0.75・0.47・0.047・0.015・0.004・0.002）。
- adaln_proj.linear.weight の列ごとの相関（X 対 R）も列 0〜3 が正、列 4〜7 が負。両方を掛けた変調量（table @ W^T + bias、65 格子点）は X 対 R で相関 0.9996（ブロック 0）・0.999（ブロック 10）・0.9999（ブロック 49）、相対差 3.0%・4.7%・1.9%。final_layer は 0.9926。
- ブロック 0・49 の変調量: Y（BF16）と X（I8）の相対差 0.09%・0.08%。
- その他: rope.inv_freq は X・Y・R で同一。video_out・video_patch_proj・norm1・adaln bias は X–R の相関 0.999 以上。

公式との構造上の差（事実のみ）:
1. X の印には convrot の鍵が無い（全 250 層 convrot=False と扱われる）。R・F・Y の本体 4 層は convrot=True・256。
2. X の I8 の値は、convrot=True の Y・R とは相関 0.06 しか無く、BF16 の Y（回転していない重み）とは相関 0.9999。尖度も回転していない BF16 と同じ。
3. X の倍率は最大 0.0426 で、Y・R（最大 0.0074）の約 5.8 倍の行がある。平均は X のほうが約 24% 大きい。
4. X・Y は adaln_proj.linear を int8（convrot なし・K=8 の行ごと倍率）にしている。他の 7 本はどれも浮動小数のまま。
5. X・Y の patch_proj・final_layer の出力 2 本・final_layer.adaln は BF16（R は F32／F16）。
6. adaln_t_table と adaln_proj の列 4〜7 の符号が R と逆（積はそろっている）。
7. 印に `per_row: true` と `orig_dtype` があるが、検査器はどちらも読まない。

**推測（未検証）**:
- X の値は印（convrot なし）と食い違っていないように見える。原因は印と中身の食い違いよりも、エンジンが convrot=False の int8 層や adaln_proj の int8 層を実際にどう処理するかの側にあるかもしれない。
- Y の `'Linear' object has no attribute 'shares_input'` について。`engine_h3\lowvram.py:150` が `qkv.shares_input(h)` を呼んでいて、shares_input を持つのは量子化用の H3Int8Linear・H3Fp8Linear・H3W4A8Linear だけ。Y のブロック 0・1・49 は qkv が BF16 なので、ここで普通の Linear に当たっている可能性がある。10Eros int8HQ（ブロック 0・1・47〜49 が BF16）も VRAM 16GB 設定では同じ経路に入りうる。
