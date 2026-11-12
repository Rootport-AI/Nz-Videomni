# MiniMax H3: fp8_scaled と w4a8 の速度・品質・動き方の裏取り（調査 E）

- 調査日: 2026-11-12
- 方法: ウェブと一次資料を読んだだけ（読み取り専用）。GPU・git・パッケージ導入は使っていない。
- 読んだコードの版（ここに書く行番号はこの版のもの）
  - ComfyUI 本体: `Comfy-Org/ComfyUI` の master（2026-11-12 時点の写し）。仕様を確かめるためだけに読み、コードは写していない。
  - comfy-kitchen: `Comfy-Org/comfy-kitchen` の main（0.2.41。2026-11-12 時点の写し）。`ck_fp8.py`、`ck_w4a8_int8.py` などの行番号はこれ。
  - 写しは調査 B のときに取ったもので、調査時の作業フォルダにあった。補助の書き出しも同じく、保管していない。
- 印の読み方: 【一次資料で確認】は一次資料で確かめたこと（以下 [P]）、【推測】は推し量ったこと（以下 [I]）。
- 方式の呼び名（調査 D と同じ）: 方式 A＝専用のカーネルをそのまま使う。方式 B＝計算のたびに重みを bf16 に戻して、ふつうの行列積で計算する。方式 C＝読み込みのときに int8 ConvRot の重みへ変換しておく。

用語:
- **量子化**: 重みの数値を少ないビット数で表して容量を減らすこと。
- **fp8**: 8 ビットの浮動小数点数。ComfyUI では `float8_e4m3fn`（指数 4 ビット）と `float8_e5m2`（指数 5 ビット）がある。fp8_scaled は、fp8 の重みに倍率（`weight_scale`）を添えた形式。
- **w4a8**: 重みを 4 ビット、計算時の入力（活性化）を 8 ビットで扱う方式。ComfyUI の形式名は `asym_w4a8_int8`。
- **nvfp4**: NVIDIA の 4 ビット浮動小数点の形式。
- **カーネル**: GPU の上で実際に計算を行う中核の処理（計算核）。
- **Ampere／Ada／Blackwell**: NVIDIA の GPU の世代。それぞれ RTX 30 系／RTX 40 系／RTX 50 系。「SM」は GPU の世代を表す NVIDIA の版番号である（Ampere は 8.0／8.6、Ada は 8.9）。
- **ConvRot**: 重みを 256 列ずつの組に分け、アダマール行列（回転の働きをする行列）で回してから int8 にする方式。
- **行列積（GEMM）**: 重みの表と入力の表を掛け合わせる計算。
- **活性化**: 層に入ってくる入力の値（前の層の出力）。
- **逆量子化**: 量子化した重みに倍率を掛けて、元の数に戻すこと。
- **二重量子化**: 一度量子化した重みを、別の形式でもう一度量子化すること。
- **相対誤差**: 誤差の大きさを、元の値の大きさで割ったもの。relL2 は、全体の誤差の二乗和の平方根を、元の値の二乗和の平方根で割った値。
- **s/it**: 1 ステップ（1 回の繰り返し）にかかる秒数。
- **TFLOPS／TOPS**: 1 秒に何兆回の計算ができるかの目安。浮動小数点は TFLOPS、整数は TOPS と書く。
- **`_scaled_mm`**: PyTorch の、fp8 どうしで行列積を行う関数（`torch._scaled_mm`）。
- **`input_scale`**: fp8 のファイルに入っている、活性化を fp8 に量子化するための固定の倍率。

## 問 1 速度

### a. Ada での fp8 の `_scaled_mm`

- [P] テンソル全体で倍率 1 つの方式は、SM 8.9 で cuBLASLt（NVIDIA の行列積ライブラリ）を通して動く。https://github.com/pytorch/pytorch/issues/153555
- [P] 行ごとの倍率の方式も、SM 8.9 で CUTLASS のカーネルで動く（PR #145676）。https://github.com/pytorch/pytorch/pull/145676
  - 未解決の不具合: 行ごとの方式は SM89 で、CUDA graph（GPU の処理をまとめて記録して再生する仕組み）の再生が止まることがある。https://github.com/pytorch/pytorch/issues/198399
  - 行ごとの方式は torch 2.10〜2.12 で、出力の型を誤って fp16 にする。https://github.com/unslothai/unsloth/pull/15791
- [P] RTX 4090 の正方行列の行列積（TFLOPS）: 2048 で fp16 142／fp8 243、4096 で fp16 165／fp8 323 → fp8 は fp16 の約 2 倍。https://github.com/triton-lang/triton/issues/5583
- [P] L4（SM89）で行ごとの倍率の大きな層: `_scaled_mm` が 255〜300 µs、逆量子化してからの行列積が 1180 µs。層ごとの相対誤差は `_scaled_mm` で 0.0218、逆量子化の経路で 0.0022。https://github.com/unslothai/unsloth/pull/15791
- [P] `use_fast_accum`（足し合わせを速いが粗い方法で行う設定）: comfy-kitchen の `scaled_mm_v2` は `F.scaled_mm` に False を渡す。`torch._scaled_mm` の既定も False。https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/scaled_mm_v2.py （113-145 行）
- [P] 固定の `input_scale` の注意点: 活性化は fp8 に直す前に ±448×倍率 で切り詰められる → 外れ値は頭打ちになる。https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/eager/quantization.py （45-59 行）

### b. w4a8 の速度

- [P] PR #90:「int8 の約 1.09 倍の速さ」、重みの relL2 は約 0.073（nvfp4 は約 0.094）。GPU ごとの表はない。https://github.com/Comfy-Org/comfy-kitchen/pull/90
- [P] CUDA の経路は、int4 を int8 に区切りながら復元してから、INT8 の行列積を行う。L2 キャッシュ（GPU の中の小さく速い記憶）の小さい Ampere では一度に全部を復元し、3080 で M=512 のとき最大 20% 速い。https://github.com/Comfy-Org/comfy-kitchen/pull/215
- [P] PR #255（0.2.39）は、入力の活性化の処理を量子化の処理に一体化した:「MiniMax-H3 is slightly faster」（MiniMax-H3 はわずかに速くなる）。https://github.com/Comfy-Org/comfy-kitchen/pull/255
- [P] 利用者の報告: RTX 3090 で w4a8、10 本の連結ショットで約 45 s/it。3090 では int8 と w4a8 は専用の計算で動き、fp8／nvfp4／mxfp8 は模擬の計算（emulated）になる。3090 での fp8 の時間はない。https://civarchive.com/models/2857809?modelVersionId=3227890
- [P、引用] 3090 で int8_convrot: 9.0 s/it、sage（SageAttention。注意機構を速くする部品）を使うと 6.5 s/it（864×480、124 フレーム）。https://smeltcore.com/recipes/minimax-h3-on-rtx-3080-ti-12-gb-video-audio-in-comfyui-on-ampere-sm-86/
- 3080／4070／4080 で int8・w4a8・fp8 を比べた表は見つからなかった。https://smeltcore.com/recipes/minimax-h3-on-rtx-4080-16-gb-ada-and-the-only-number-this-card-has/

### c. 計算のたびに逆量子化する費用

- [P] いちばん近い代わりの数字: 4060 Ti 16 GB で int8_convrot を、comfy-kitchen の CUDA の経路を切って（torch cu129）動かすと 770 秒、入れると 355 秒 → 2.17 倍。https://note.com/tank_ai/n/nab26edd4ab96?hl=en
- 「同じ Ampere の GPU で、fp8 を模擬の計算で動かした場合と int8 の比較」の直接の数字はない。
- [P] RTX A6000（SM 8.6）、H3 の QKV の形の行列: bf16 148.8 ms、int8 ConvRot の既定の設定 144.3 ms（1.03 倍）、int8 で CUTLASS の設定を最もよく合わせたもの 約 84〜89 ms。https://github.com/Comfy-Org/comfy-kitchen/issues/98 。Ampere の区切り方（タイル）の修正 PR #237 はまだ取り込まれていない。https://github.com/Comfy-Org/comfy-kitchen/pull/237
- [I] 仕様表（第三者の資料）:
  - RTX 3080 10 GB: BF16 59.5 TFLOPS（密行列）、INT8 238 TOPS（4 倍）。https://hmc-tech.com/gpu/nvidia-geforce-rtx-3080-10gb
  - RTX 4070 Ti SUPER: FP16（FP32 で足し合わせ）88.2 TFLOPS、FP8 176.4、INT8 352.8。https://hmc-tech.com/gpus/nvidia-geforce-rtx-4070-ti-super-vs-nvidia-geforce-rtx-4080-super ／ https://flopper.io/gpu/nvidia-geforce-rtx-4070-ti-super-16gb/spec-sheet.pdf
  - NVIDIA の公式の Ampere の資料: https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf

## 問 2 品質

### a. 重みの誤差

- [P] int8 ConvRot は相対誤差 約 0.85%、fp8 は約 2.8%（Unsloth が ComfyUI のファイルを読んで測ったもの）。https://github.com/unslothai/unsloth/pull/12765
- [P] 埋め込みの表: int8 ConvRot 約 0.8%、fp8_scaled 約 2.6%。https://github.com/Comfy-Org/ComfyUI/pull/15035
- [P、個人の感想] 見た目: CivitAI の作者は、w4a8 と int8 で同じ欠点が出ると見ている。あるコメントの書き手は、fp8 が「int8 よりわずかによい」と感じている。https://civarchive.com/models/2857809?modelVersionId=3227890

### b. 二重量子化

- [P] int8 ConvRot → w4a8 の relL2 は 0.07368。fp16 から直接 w4a8 にすると 0.07311（+0.8%。合成したガウス分布のデータだけでの測定）。https://github.com/0xDELUXA/comfy-int8-convrot-to-w4a8
- [P] LoRA を掛けた後の再量子化で ConvRot の設定が落ちて、出力が劣化していた。PR #14650（シードつきの確率的な丸め）で直った。https://github.com/Comfy-Org/ComfyUI/issues/14642 ／ https://github.com/Comfy-Org/ComfyUI/pull/14650
- [P] 未解決: LoRA を掛けた int8 ConvRot のモデルで、サーバーを動かし続けたまま再実行すると壊れる。https://github.com/Comfy-Org/ComfyUI/issues/16731
- [I] fp8 → int8 の再量子化を調べたものはない。

### c. 知られている不具合

- [P] #136: Triton の INT8 の行列積は、m×n が 2^31 以上になると int32 の位置の計算が溢れて落ちる。修正 PR #172 はまだ取り込まれていない → Triton の経路は危険。https://github.com/Comfy-Org/comfy-kitchen/issues/136 ／ https://github.com/Comfy-Org/comfy-kitchen/pull/172
- [P] #243: INT8 の行列積は、次元が 4 の倍数でなければならない。https://github.com/Comfy-Org/comfy-kitchen/issues/243
- [P] #289: `int8_linear` は割り算を bf16 で行う（誤差が増える。fp32 で割れば費用なしで直る）。https://github.com/Comfy-Org/comfy-kitchen/issues/289
- [P] #196: Windows で、モデルにもう一度入ると int8 ConvRot が止まる。https://github.com/Comfy-Org/comfy-kitchen/issues/196
- [P] ComfyUI の PR #15308（w4a8）: レビューで「`_correction` が読み込まれない」「切り替えの失敗が見えにくい」と指摘されたが、確かめられていない。https://github.com/Comfy-Org/ComfyUI/pull/15308
- w4a8 に固有の誤った出力の報告はない。Windows での `_scaled_mm` の不具合の報告もない。
- [P] ComfyUI は Windows で fp8 を使うのに torch 2.4 以上を求める（`model_management.py` 2104-2106 行）。

## 問 3 ComfyUI の動き方（写しの `ops.py`、`quant_ops.py`）[P]

https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ops.py

1. Ampere（SM 8.6）: `supports_fp8_compute` が False（`model_management.py` 2093-2100 行）→ fp8 の形式は無効（`ops.py` 1966-1968 行）→ 読み込み時に `module._full_precision_mm = True` にする（1384 行）。
2. そういう層の計算: `_use_quantized` が False（1622-1627 行）→ 活性化は量子化しない。**`input_scale` は無視される。** fp8 の重みは計算のたびに逆量子化され（`weight.dequantize()`、1597-1598 行）、`orig_dtype = compute_dtype`（1465 行あたり）＝ bf16（または fp16）になる。あとはふつうの線形層の計算。
3. Ada で `full_precision_matrix_mult` の印がある層: 同じ経路（1381-1383 行）。`_scaled_mm` は使わない。
4. それ以外の Ada の層: `input_scale` があれば固定の倍率で、なければその場の最大値（動的な amax）で活性化を量子化し、`_scaled_mm` を呼ぶ。失敗したら逆量子化に切り替える（`ck_fp8.py` 155-169 行）。
5. Ampere での w4a8: 専用の計算で動く。この形式は SM 8.0 以上を求める（`ck_w4a8_int8.py` 159 行）。無効になるのは int8 の計算ができないときだけ（`ops.py` 1969-1972 行）。torch が cu130 以上なら CUDA の経路、そうでなければ eager（PyTorch の素の計算）（`quant_ops.py` 22-28 行）。Triton は `--enable-triton-backend` を付けない限り使わない（33-41 行）。Windows の cu130 では、既定で CUDA のカーネルになる。

## 問 4 comfy-kitchen のライセンスと版

- [P] ライセンスは Apache-2.0。https://github.com/Comfy-Org/comfy-kitchen/blob/main/LICENSE
- [P] PyPI の公開: 0.2.37（10-02）、0.2.38（10-17）、0.2.39（10-18）、0.2.40（10-22）、0.2.41（11-05）。ComfyUI の master は 0.2.41 に固定している。https://pypi.org/project/comfy-kitchen/ ／ https://github.com/Comfy-Org/ComfyUI/blob/master/requirements.txt
- [P] 0.2.37 → 0.2.41 の int8／w4a8 の変更: #234 Ampere の int8 の注意機構の最適化、#251 ConvRot の量子化処理での SiTU 活性化、#255 W4A8／W6A8 の量子化処理への入力活性化の一体化、#259 AMD での int8 の性能、#301 CUDA のブロック量子化の高速化と RMS-RoPE の一体化。ほかに sm100 の Windows 向けビルドと新しい演算。https://github.com/Comfy-Org/comfy-kitchen/commits/main
- [I] API を壊す変更の記述は見当たらない。上げても安全というのは推測である。

## 問 5 ComfyUI 以外の実行環境

- [P] SGLang-Diffusion: Comfy の fp8／MXFP8 を読み、層ごとに固定か動的かを選ぶ。INT8 ConvRot は CC 9.0／10.x／12.x では自前の JIT（実行時に組み立てる）演算、それ以外（4090 を含む）では comfy-kitchen。**W4A8／W4A4 には comfy-kitchen が必要。** https://lmsysorg.mintlify.app/cookbook/diffusion/MiniMax/MiniMax-H3
- [P] Unsloth Studio: fp8 は読み込み時に逆量子化し、`input_scale` は無視する。int8 は torchao か自前の部品で扱う。https://github.com/unslothai/unsloth/pull/12765
- [P] InvokeAI: Comfy の int8 ConvRot を、回転を戻して読む。https://github.com/invoke-ai/InvokeAI/pull/10186
- [P] DiffSynth-Studio: 自前の NF4。https://github.com/MiniMax-AI/awesome-minimax-h3-integration
- [I] LightX2V、vLLM-Omni、H3ddle が w4a8 や fp8 を読める根拠は見つからなかった。

## 要点

- 方式 B は、ComfyUI 自身の Ampere での動き方と同じ: 計算のたびに逆量子化し、`input_scale` を無視する。
- 方式 C は、誤差が約 +0.8% 増える（測定は合成データでの int8 → w4a8 の 1 件だけ）。fp8 の重みの誤差 2.8% から始めるので、品質は fp8 の水準が上限になる [I]。
- 方式 A をとるなら、Triton の INT8 の経路（#136）は避けること。Ampere での int8 の伸びは、区切り方（タイル）の調整（#98、#237）次第である。
