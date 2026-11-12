# MiniMax H3: fp8 と 4 ビット系の配布形式の調査（調査 B）

- 調査日: 2026-11-12
- 方法: ウェブと一次資料を読んだだけ（読み取り専用）。GPU・git・パッケージ導入は使っていない。重みファイルのヘッダは HTTP Range（ファイルの先頭部分だけを取る要求）で、ヘッダと小さな `comfy_quant` のテンソルだけを読んだ。
- 読んだコードの版
  - ComfyUI 本体: `Comfy-Org/ComfyUI` の master（2026-11-12 時点の写し）
  - comfy-kitchen: `Comfy-Org/comfy-kitchen` の main（0.2.41。2026-11-12 時点の写し）
- ヘッダの書き出し（`h_*.txt`・`hdr_*.txt`）、スクリプト（`tools/`）、上記 2 つのリポジトリの写し（`src/`）は、調査時の作業フォルダにあったもので、保管していない。
- 印の読み方: 【一次資料で確認】は一次資料で確かめたこと（以下 [P]）、【推測】は推し量ったこと（以下 [I]）。

用語:
- **量子化の印（`comfy_quant`）**: 量子化（重みの数値を少ないビット数で表して容量を減らすこと）した層ごとに、その層の形式を書いた小さなテンソル `<層の名前>.comfy_quant`。中身は JSON の文字列を uint8 の並びにしたもの。
- **旧式のメタデータ（`__metadata__._quantization_metadata`）**: 層ごとのテンソルの代わりに、safetensors のファイル先頭にある目録（ヘッダ）の `__metadata__` 欄へ、全層の形式をまとめて JSON で書く方式。
- **枝刈り版**: AdaLN（時刻に応じて各層の出力を調整する仕組み）の枝を、先に計算した表 `adaln_t_table` に畳み込んで小さくした版。非枝刈り版は、時刻の埋め込みの枝をそのまま持つ。
- **fp8**: 8 ビットの浮動小数点数。ComfyUI では `float8_e4m3fn`（指数 4 ビット）と `float8_e5m2`（指数 5 ビット）がある。
- **w4a8**: 重みを 4 ビット、計算時の入力（活性化）を 8 ビットで扱う方式。ComfyUI の形式名は `asym_w4a8_int8`。同様に w6a8 は重み 6 ビット、w4a4 は重みと活性化がともに 4 ビット。
- **nvfp4**: NVIDIA の 4 ビット浮動小数点の形式。
- **カーネル**: GPU の上で実際に計算を行う中核の処理（計算核）。
- **Ampere／Ada／Blackwell**: NVIDIA の GPU の世代。それぞれ RTX 30 系／RTX 40 系／RTX 50 系。本文の「SM」や「compute capability（計算能力）」は、GPU の世代を表す NVIDIA の版番号である（数字が大きいほど新しい）。

## 要点

- 実際に出回っている fp8 は、`comfy_quant` に `float8_e4m3fn` と書いた、倍率つき（scaled）のもの。単純に型を変えただけのものと、旧式の `scaled_fp8`＋`.scale_weight` の書き方は、H3 では見つからなかった。
- 「4 ビット」と呼ばれるものは、ほとんどが `asym_w4a8_int8`。CivitAI は 14 GB の h3ErosMax の w4a8 に「fp8」の札をつけている（コメント欄に「札が違う。これは w4a8」という指摘がある）。
- 量子化の設定の持たせ方は 2 通りある。層ごとの `<layer>.comfy_quant`（JSON を uint8 で持つテンソル）と、ヘッダの `__metadata__["_quantization_metadata"]`（JSON の対応表）である。Kijai の `w4a8_mixed` と h3ErosMax の beta5 の int8 は、ヘッダ式だけを持つ。両方あるとき、ComfyUI はヘッダの側でテンソルの側を上書きする。
- コミュニティの int8 のファイルは、`weight_scale` を [N,1]（行ごと）にして `"per_row": true` を足している。ComfyUI はこの余分なキーを無視する。

## 問 1 fp8

Comfy-Org/MiniMax-H3 の `diffusion_models/`（fl2va と ref2va）[P]:

- bf16 66.3 GB
- int8_convrot 34 GB
- pruned_bf16 40.2 GB
- pruned_fp8_scaled 21 GB
- pruned_int8_convrot 21 GB
- pruned_w6a8 16 GB

テキストエンコーダ [P]: bf16 51.5 GB、int8_convrot 27.1 GB、nvfp4_awq 15.7 GB（モデルカードの説明: nvfp4_awq のエンコーダは Blackwell を必要としない。cu130 では int8_convrot を勧め、それ以外では fp8_scaled を勧める）。

`pruned_fp8_scaled` のヘッダ [P]:

- F8_E4M3 の `weight` が 200 本。それぞれに F32 の倍率 1 つの `weight_scale` と、`{"format":"float8_e4m3fn"}` の `comfy_quant` がつく。
- F32 の `input_scale` が 150 本（qkv、out、fc1）。
- 50 本の `mlp.fc2` はすべて `"full_precision_matrix_mult": true` を足していて、`input_scale` を持たない。
- token_refiner は BF16。`adaln_proj` は F16 [96768,8]。`adaln_t_table` がある。

CivitAI の「pruned fp8 scaled」（Liant）は同じファイルを置き直したもの。joeygambino の一式にも float8_e4m3fn がある。https://civarchive.com/models/2857809?modelVersionId=3227890

e5m2: 見つからなかった。joeygambino によると、行列積のところで失敗する。

## 問 2 「4 ビット」

- `asym_w4a8_int8`（よくある形）[P]:
  - Kijai の `*_pruned_w4a8_mixed` 12.5／11.8 GB。200 層すべてが `{"format":"asym_w4a8_int8","group_size":16,"convrot":true,"convrot_groupsize":256}` で、ヘッダ式のみ（https://huggingface.co/Kijai/MiniMax-H3-experimental）。
  - TenStrip の h3ErosMax `*_w4a8_14gb_optimized`: w4a8 の層が 180（group size は 16／8／4、`weight_codebook` [16]（符号表））と、行ごとの int8 の端の層が 20。作成は「h3_w4a8_converter」。
  - joeygambino の w4a8。
- `convrot_w4a4`: Abiray の `pruned_mixed_int4_int8_convrot` だけ（w4a4 が 117 層で `linear_dtype:"int8"`、int8 が 83 層）。joeygambino によると、w4a4 の出力は目に見えて崩れる。
- `nvfp4`: lilcheaty、Abiray、joeygambino。`weight` は U8 [N,K/2]、`weight_scale` は F8 [N,K/16]、`weight_scale_2` は F32 の値 1 つ。そのまま動かすには SM 10 以上が要る。それより古い GPU では元の精度に戻して計算する（「emulated＝模擬」）。
- H3 では見つからなかったもの: Nunchaku の SVDQuant、nf4、torchao の int4、mxfp4。ComfyUI は SVDQuant と AWQ W4A16 を登録していない。
- 必要な最低の GPU [P]: int8 と convrot_w4a4 は SM 7.5。asym_w4a8 と w6a8 は SM 8.0。fp8 は SM 8.9。nvfp4 と mxfp8 は SM 10。
- カーネル [P]: convrot_w4a4 は eager（PyTorch の素の計算）／CUDA／HIP で、Triton（GPU のカーネルを書く仕組み）はない。w4a8 は eager／CUDA／Triton（kitchen の PR #90、2026-08-06）。ComfyUI は、PyTorch が cu130 以上でないと kitchen の CUDA の経路を切る。Triton は `--enable-triton-backend` をつけたときだけ使う。
- 各形式に対応した ComfyUI の版 [P]: nvfp4 は v0.8.0、mxfp8 は v0.18.0、int8_tensorwise は v0.27.0、convrot_w4a4 は v0.28.0、H3 への対応は v0.30.0、asym_w4a8_int8 は v0.31.0、w6a8_int8 は v0.38.0。

## 問 3 形式の一覧（ComfyUI の QUANTIZATION.md、comfy/quant_ops.py、comfy/ops.py の 1360-1480 行あたり）

| 形式 | weight | 倍率と設定 |
|---|---|---|
| float8_e4m3fn／e5m2 | [N,K] | `weight_scale` は値 1 つ。`input_scale` は任意 |
| nvfp4 | uint8 [N,K/2]、16 の倍数に詰め物 | `weight_scale` は F8 で 16 個ごとの区画。`weight_scale_2` は全体に 1 つ。`input_scale` と `pre_quant_scale` は任意 |
| mxfp8 | F8 [N,K] | `weight_scale` は E8M0（32 個ごとの区画） |
| int8_tensorwise | int8 [N,K] | `weight_scale` は値 1 つか [N,1]。`convrot`、`convrot_groupsize`（256） |
| convrot_w4a4 | int8 [N,K/2] | `weight_scale` は行ごと。`convrot_groupsize`。`linear_dtype` は int4／int8。量子化のグループは 64 |
| asym_w4a8_int8 | int8 [N,K/2] | `weight_s_rel` は F8 [N,K/G]、`weight_s_channel` は F32 [N]、`weight_codebook` [16] は任意。`group_size`（16） |
| w6a8_int8 | int8 [N,3K/4] | 倍率は上と同じ。符号表はない |

設定は入れ子の `params` の中に置かれることもある。ComfyUI のリポジトリには量子化を行う道具がない。ファイル名の末尾の語は慣習にすぎない。

## 問 4 量子化した層への LoRA [P]

すべてを GPU に載せる場合（`model_patcher.patch_weight_to_device`）: `convert_weight` が元の精度に戻し、`calculate_weight` が浮動小数点で LoRA を足し、`set_weight` → `requantize_from_float(scale="recalculate", stochastic_rounding)` で同じ形式に量子化し直す。VRAM を節約する場合: `LowVramPatch` が、計算のたびに元の精度に戻して LoRA を当てる。bf16 の脇道（別枝）は使わない。コミュニティの Turbo 入りの重みは、bf16 で合成してから量子化している。PulpCut の合成では、AdaLN の LoRA の 51 組が抜け落ちた。

## 問 5 「Hybrid」

h3ErosMax: Ref2VA と FL2VA／T2V を 1 本の重みにまとめたもの。「TURBO-hybrid」はさらに 2 つの Turbo LoRA を焼き込んでいる（約 4.2 GB 節約）。DaSiWa の Hybrid も同じ意味。混合精度という意味ではない。

h3ErosMax の精度（HF の TenStrip/10Eros-Max）: bf16（フル 66 GB、枝刈り版 40 GB）、int8／int8_convrot（約 21 GB）、int8HQ（約 23 GB）、w4a8（14 GB）。本物の fp8 や nvfp4 はない。CivitAI のページには量子化の方法が書いていないが、ヘッダには作成の道具として「Star Ultimate Model Converter」（beta5 の int8）と「h3_w4a8_converter」（w4a8、int8HQ）の名がある。

pinkcherry v18Hybrid: 見つからなかった。あったのは「PinkCherry fl2va v1_final (turbo+pruned+int8 convrot)」の置き直しだけ。PulpCut: Turbo を合成した int8 ConvRot で、H3ddle（macOS 向け）用。その中の 1 種は入力側を主とする並び（input-major）で、ふつうの読み込み処理とは互換がない。

## 問 6 知られている問題と数字

RTX 5090（joeygambino）の 1 ステップあたりの秒数（s/it）: nvfp4 12.7、int8 13.5、w4a8 14.05、fp8 14.8、Q8_0 の GGUF 19.4。RTX 4090: w4a8 17.4 に対して fp8 31.0（fp8 は重みが VRAM に収まらず主記憶へ逃がすオフロードが起きている）。4070 12 GB: fp8 で 544p・5 秒の Turbo に約 105 秒。ほかの人は fp8 が int8 より遅いと言っている。12〜16 GB での int8／fp8／int4 の比較は見つからなかった。

ComfyUI の issue: #17098 は RTX 5060 Ti で int8_convrot だとノイズが出て、fp8_scaled ではきれい（未解決）。#15563 は v0.32 の bf16 で黒いフレーム。#15524 は int8 ConvRot の VAE で黒いフレーム。fp8 に固有の黒いフレームの報告はない。

## 出典

- https://huggingface.co/Comfy-Org/MiniMax-H3
- https://huggingface.co/Kijai/MiniMax-H3-experimental
- https://huggingface.co/TenStrip/10Eros-Max
- https://civarchive.com/models/2851079?modelVersionId=3294059
- https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4
- https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot
- https://huggingface.co/PulpCut/MiniMax-H3-Turbo-INT8-ConvRot
- https://github.com/Comfy-Org/comfy-kitchen
