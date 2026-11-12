# MiniMax H3: comfy-kitchen のカーネル、復元の数式、LTX 側の前例、H3 の層の条件、メモリ（調査 D）

- 調査日: 2026-11-12
- 方法: ソースコードを読んだだけ（読み取り専用）。GPU・テスト・本製品のファイルの編集は使っていない。復元（4 ビットや int8 の重みを元の数に戻すこと）の数式は、Apache-2.0 の comfy-kitchen だけから導いた。ComfyUI からは導いていない。
- 読んだコードの版（ここに書く行番号はこの版のもの）
  - comfy-kitchen 0.2.37: `.venv-engine-h3\Lib\site-packages\comfy_kitchen` に入っているもの（以下 CK）。特に断りのない comfy-kitchen の行番号はこれ。
  - comfy-kitchen 0.2.41: `Comfy-Org/comfy-kitchen` の main の写し（2026-11-12 時点）。§1 f の `ck_w4a8_packed.py` はこちら。
  - ComfyUI 本体: `Comfy-Org/ComfyUI` の master の写し（2026-11-12 時点）。仕様を確かめるためだけに読んだ。ライセンスが GPL-3.0 なので、コードは写していない。`ops.py`、`model_management.py`、`quant_ops.py` の行番号はこれ。
  - 本製品のバックエンド: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni` の dev ブランチ `abc8eea`（2026-11-12）。`h3_format.py`、`quant_linear.py`、`lowvram.py`、`lora.py`、`dit.py`、`sft_load.py`、`pipeline_h3.py`、`block_swap_prefetch.py`、`engine/sft_quant` 以下の行番号はこれ。
- 補助の資料（表 `tables.md`、数式 `w4a8_math_reference.md`、メモリの計算 `mem.py`・`mem_out.txt`）は調査時の作業フォルダにあったもので、保管していない。中身はこの文書の §1 a の関数の一覧、§1 b の数式、§2 の拒否の一覧、§4 の表に移した。
- 印の読み方: 【一次資料で確認】は一次資料（ここではコードそのもの）で確かめたこと（以下 [P]）、【推測】は推し量ったこと（以下 [I]）。原文で「fact」とされた事柄は [P]、「Inference」とされた事柄は [I] と書く。
- 方式の呼び名（原文のまま）: 方式 A＝専用のカーネルをそのまま使う。方式 B＝計算のたびに重みを bf16 に戻して、ふつうの行列積で計算する。方式 C＝読み込みのときに int8 ConvRot の重みへ変換しておく。

用語:
- **量子化の印（`comfy_quant`）**: 量子化（重みの数値を少ないビット数で表して容量を減らすこと）した層ごとに、その層の形式を書いた小さなテンソル `<層の名前>.comfy_quant`。中身は JSON の文字列を uint8 の並びにしたもの。
- **旧式のメタデータ（`__metadata__._quantization_metadata`）**: 層ごとのテンソルの代わりに、safetensors のファイル先頭にある目録（ヘッダ）の `__metadata__` 欄へ、全層の形式をまとめて JSON で書く方式。
- **枝刈り版**: AdaLN（時刻に応じて各層の出力を調整する仕組み）の枝を、先に計算した表 `adaln_t_table` に畳み込んで小さくした版。非枝刈り版は、時刻の埋め込みの枝をそのまま持つ。
- **fp8**: 8 ビットの浮動小数点数。ComfyUI では `float8_e4m3fn`（指数 4 ビット）と `float8_e5m2`（指数 5 ビット）がある。
- **w4a8**: 重みを 4 ビット、計算時の入力（活性化）を 8 ビットで扱う方式。ComfyUI の形式名は `asym_w4a8_int8`。同様に w6a8 は重み 6 ビット。
- **カーネル**: GPU の上で実際に計算を行う中核の処理（計算核）。
- **Ampere／Ada**: NVIDIA の GPU の世代。それぞれ RTX 30 系／RTX 40 系。Turing は RTX 20 系。「SM」や「CC（compute capability、計算能力）」は、GPU の世代を表す NVIDIA の版番号である（数字が大きいほど新しい。Turing は 7.5、Ampere は 8.0／8.6、Ada は 8.9）。
- **ConvRot**: 重みを 256 列ずつの組に分け、アダマール行列（+1 と −1 だけでできた、回転の働きをする行列）で回してから int8 にする方式。保存される重みは回した後のもの。
- **行列積（GEMM）**: 重みの表と入力の表を掛け合わせる計算。
- **活性化**: 層に入ってくる入力の値（前の層の出力）。
- **逆量子化**: 量子化した重みに倍率を掛けて、元の数（bf16 や fp32）に戻すこと。
- **再量子化**: いったん元の数に戻した重みを、別の形式で量子化し直すこと。
- **eager**: PyTorch の素の計算で処理する経路。専用のカーネルを使わない。
- **cuBLASLt**: NVIDIA の行列積ライブラリ。
- **VRAM 16GB mode**: 本製品の、重い層を行の範囲に分けて少しずつ計算する仕組み（`forward_rows`）。
- **block swap**: 本製品の、ブロックの重みを主記憶と VRAM の間で出し入れする仕組み。

## 1. comfy-kitchen（入っている 0.2.37）

### a. 入口となる関数 [P]

- 公開されている関数: `int8_linear`（`__init__.py:997`）、`w4a8_int8_linear`、`dequantize_w4a8_int8_weight`、`quantize_w4a8_int8_weight`（`tensor/w4a8_int8.py:31-127`）。ほかに `quantize_per_tensor_fp8`／`dequantize_per_tensor_fp8`、int8 のテンソル全体／行ごとの量子化関数。
- CUDA の `w4a8_int8_linear` は、最低 CC 8.0 で登録される（`cuda/__init__.py:4213-4248`）。cuBLASLt の有無には左右されない。一方 `int8_linear` は cuBLASLt があるときだけ登録される（4463 行）。
- **関数として呼べる fp8 の行列積はない。** fp8 は `TensorCoreFP8Layout` の QuantizedTensor（量子化した重みを包む PyTorch のテンソル）を通してしか使えない（`tensor/fp8.py:133-169`。最低の SM は (8,9)、39 行）。入力と重みの両方が fp8 の QuantizedTensor である必要があり（144 行）、`scaled_mm_v2`（`scaled_mm_v2.py:101-157`。中身は `F.scaled_mm`／`torch._scaled_mm`）を呼ぶ。RuntimeError が出たら、逆量子化して `F.linear` で計算する（167-169 行）。
- [I] 方式 A で fp8 を扱うなら、Ada で `scaled_mm_v2`／`torch._scaled_mm` を直接呼ぶことになる。Ampere 向けの fp8 のカーネルはない。

関数の一覧（原文の表 T1 を箇条書きに直したもの）[P]:

- `int8_linear(x, weight, weight_scale, bias=None, out_dtype=None, convrot=False, convrot_groupsize=256, input_act=None, ..., residual=None, residual_scale=None)`
  - 定義: `__init__.py:997-1054`（登録の仕組み）。公開関数の一覧は `__init__.py` の `__all__`（57-137 行）。
  - 経路: CUDA は cuBLASLt が見つかるときだけ（`cuda/__init__.py:4463-4487`、最低 CC 7.5）。eager は `quantization.py:985-1084`（`torch._int_mm`）。triton もある。
- `w4a8_int8_linear(x, qdata, s_rel, s_channel, codebook=None, correction=None, bias=None, group_size=16, convrot_groupsize=256, out_dtype=bf16)`
  - 定義: `tensor/w4a8_int8.py:101-127`（登録の仕組み）。
  - 経路: CUDA は `cuda/__init__.py:2358-2545`。条件は 4213-4248 行（qdata は 2 次元の int8、s_rel は 2 次元の F8 か F32、s_channel は 1 次元の F32、codebook は 1 次元の F32、最低 CC 8.0、cuBLASLt には左右されない）。eager は `eager/w4a8_int8.py:659-710`。triton は `backends/triton/w4a8_int8.py`。
- `dequantize_w4a8_int8_weight(qdata, s_rel, s_channel, codebook=None, correction=None, group_size=16, convrot_groupsize=256, output_dtype=bf16)`
  - 定義: `tensor/w4a8_int8.py:76-98`。
  - 経路: CUDA は 2327-2355 行（専用の復元と回転）、条件は 4182-4211 行（CC 8.0）。eager は 623-656 行。
- `quantize_w4a8_int8_weight(weight, group_size=16, convrot_groupsize=256, symmetric=True, scale_dtype=f8, codebook=True, codebook_tensor=None, stochastic_rounding=0, bits=4, scale_search=True)`
  - 定義: `tensor/w4a8_int8.py:31-73`。経路: CUDA は 1508-1560 行（CC 8.0）、eager は 517-549 行。
- `quantize_int8_tensorwise`／`quantize_int8_rowwise`／`dequantize_int8_simple`: `__init__.py:925-948`。eager・CUDA・triton。
- `quantize_per_tensor_fp8`／`dequantize_per_tensor_fp8(x, scale, dtype)`: `__init__.py:387-422`（torch.ops）。eager は `quantization.py:45-65`、ほかに CUDA・triton。
- 公開されていない `quantize_int8_convrot_weight(weight, group_size, stochastic_rounding=0)`: eager は `quantization.py:899-907`、独自の演算 `comfy_kitchen::quantize_int8_convrot_weight` として 1121 行。CUDA は 1700-1740 行（条件は 4146-4156 行、CC の欄なし）。`tensor/int8.py:123-128` と同じように `registry.get_implementation("quantize_int8_convrot_weight")` で取り出すか、経路の関数を直接読み込んで呼ぶ。
- 公開されていない `rotate_int8_convrot_weight(weight, group_size)`: eager は 910-913 行、CUDA は 1406-1416 行（高速アダマール変換のカーネル）。
- 公開されていない `dequantize_int8_convrot_weight[_dtype]`: eager は 916-926 行。`torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype`。
- fp8 の行列積: 関数としての入口はない。上に書いた `TensorCoreFP8Layout`（`tensor/fp8.py:23-169`、最低の SM (8,9) は 39 行）→ `scaled_mm_v2`（`scaled_mm_v2.py:101-157`。`torch.nn.functional.scaled_mm` があればそれ、なければ `torch._scaled_mm`）だけ。RuntimeError か TypeError のときはログを出して、逆量子化＋`F.linear` に切り替える（`fp8.py:167-169`）。入力と重みの両方が、すでに fp8 の QuantizedTensor になっている必要がある（`fp8.py:144-145`）。この仕組みは活性化を自分では量子化しない。

### b. w4a8 の数式 [P]

- 取り出し（`eager/w4a8_int8.py:80-93`）: uint8 として読み、番号は 0〜15。偶数番目の列は 1 バイトの下位 4 ビット、奇数番目は上位 4 ビット。
- 復元（552-567 行）: 符号帳（codebook。16 段階の値の表）があれば `v = codebook[c]`、なければ `c - 8`。それに組の倍率を掛けて丸め、`i8 = clamp(round_half_even(v * s_rel[n, k//G]), ±127)` とする。
- 倍率を掛ける（606-620 行）: `W_rot = i8 * s_channel[n]`（あれば `correction[k//G, n]` を足す）。
- 回転を戻す（656 行）: `W = W_rot · R`。R は H を斜めに並べたブロック対角の行列で、H は H4 を 4 回クロネッカー積したものを 16 で割ったもの（`tensor/int8_utils.py:11-38`）。H は対称で、2 回掛けると元に戻る。
- 保存されている重みは回した後のもの。カーネルは活性化を回して（x·R）、行ごとの倍率 `s_channel` で int8 の行列積を行う（696-710 行が `int8_linear(x, i8, s_channel, convrot=True)` を呼ぶ）。
- 「asym」は仕組みの名前（クラス名）にすぎない。非対称（ゼロ点つき）の変種は `correction` の枝（323-333 行）で、対象のファイルには `*_correction` がないので対称である。
- G（組の大きさ）は 4、8、16、32… を取れる（584-585 行）。
- **要点: w4a8 の層は、重みを `i8`、倍率 `weight_scale` を `s_channel[:,None]` とした int8 ConvRot の層と、数学的にまったく同じである。**

数式のまとめ（原文の `w4a8_math_reference.md` から。出典の行は CK 0.2.37）:

```
記号: W = 本来の重み [N,K]。G = 組の大きさ。
H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
H  = kron^4(H4) / sqrt(256)      … 256×256。対称・直交で H = H^T = H^-1（int8_utils.py:24-36）
R  = blockdiag(H, ..., H)         … K 方向に K/256 個

ConvRot（int8 の ConvRot と、w4a8 のすべてのファイル）
  保存する重み:   W_rot = W R^T = W R            （_rotate_weight、int8_utils.py:46-55）
  実行時の活性化: x_rot = x R                    （_rotate_activation、int8_utils.py:58-74）
  x_rot W_rot^T = x R R^T W^T = x W^T            … 行列積の後で回転を戻す必要はない
                                                    （int8_linear(convrot=True): quantization.py:1036-1042。
                                                     その後 x_rot を行ごとに int8 へ量子化）
  保存した重みの回転を戻す: W = W_rot R          （dequantize_int8_convrot_weight、quantization.py:916-919）

int8_tensorwise: q は int8 [N,K]、倍率 s は F32 のスカラーか [N,1]
  W_rot_hat[n,k] = q[n,k] * s[n]。convrot なら W_hat = W_rot_hat R、そうでなければ W_hat = W_rot_hat

asym_w4a8_int8 の保存形式
  qdata      int8 [N, K/2]   … 行 n のバイト j の 0-3 ビットが列 2j、4-7 ビットが列 2j+1
                                 （_pack_codes 69-77、_unpack_codes 80-93。uint8 として読むので符号なし 0..15）
  s_rel      [N, K/G]        float8_e4m3fn（または F32）… 組の倍率（行の倍率に対する比）
  s_ch       [N]             F32 … 行（チャンネル）の倍率 = 組の倍率を掛けた段階値の絶対値の最大 / 127
  codebook   [16]            F32（なくてもよい）… [-1,1] の中の 16 段階の値
                                 （Lloyd-Max 法の段階値。eager 19-22 の固定の表か、合わせて求めたもの）
  correction [K/G, N]        （なくてもよい）… 非対称（ゼロ点つき）の変種だけ
  ビット数は幅から決まる: K = s_rel.shape[1]*G、bits = 8*列数/K ∈ {4,6}（_w4a8_geometry 54-66）

int8 の目盛りへの復元（_dequant_int4_grouped_to_int8、eager 552-567）
  c[n,k] ∈ 0..15
  v[n,k] = codebook[c]   （符号帳があるとき）
         = c - 8          （ないとき。符号つきの一様な 4 ビット。565 行）
  i8[n,k] = clamp(round_half_even(v[n,k] * float(s_rel[n, k//G])), -127, 127)
回した基底での重み（606-620）:  W_rot_hat[n,k] = i8[n,k] * s_ch[n]（+ correction[k//G, n]）
元の基底（656）:                W_hat = W_rot_hat R

線形層（eager 696-710。CUDA も一体化したカーネルで同じことをする）
  y = int8_linear(x, i8, s_ch, bias, convrot=True, convrot_groupsize=256)
⇒ correction のない w4a8 の層は、劣化なしで H3Int8Linear の入力に変換できる:
   weight = i8（int8 [N,K]）、weight_scale = s_ch.view(N,1)、convrot=True、組の大きさ 256
```

- 補足 [P]: G の検査は「G が 4 以上で 16 を割り切る、または 16 で割り切れる」を受け付ける（eager 584-585、`tensor/w4a8_int8.py:160-171`）。`convrot_groupsize` は K を割り切らなければならない（ここでは 256）。既定の量子化関数（対称＋符号帳）は `correction` を作らない。非対称の変種は量子化関数の `symmetric=False` の枝（eager 323-333）で、番号は符号なし、`(c-8)×組の倍率` で復元し、`correction`＝「8×組の倍率＋組の最小値」を、`s_channel` を掛けた後に、回した基底で組ごとに足す。対象のファイルには `*_correction` のキーがないので、形式は対称である。
- 原文には、この数式を PyTorch で書いた短い参照コード（実行はしていない）もあった。4 ビットの取り出し、`c-8` か符号帳での復元、組の倍率、丸め（`torch.round` は偶数への丸め）、行の倍率、回転の戻しの順で、上の数式と同じ内容である。fp8 の逆量子化は `(w_f8.float() * weight_scale.float()).to(torch.bfloat16)` で、LTX 側の `dequant.py:97` と同じ。参照コードの 1 層あたりの一時的なメモリは、int32 の番号・f32 の値・f32 の回転後の値の 3 つ分（要素あたり各 4 バイト）。LTX 側の `dequant.py:111-134` は `index_select` とその場での演算で、f32 の 2 つ分ほどに抑えている。

### c. int8 ConvRot [P]

- 保存される重みは `W·R`。`int8_linear(convrot=True)` が x を回す（`eager/quantization.py:1036-1042`）。
- `per_row` の印は誰も読まない（CK も ComfyUI の `ops.py:1407-1419` も）。意味を持つのは倍率の要素の数（1 か N）だけ（`quantization.py:1031-1035`）。

### d. ComfyUI での fp8（仕様の確認のみ）[P]

- Ampere: CC 8.9 未満では `supports_fp8_compute` が False（`model_management.py:2083-2108`）。fp8 の形式は無効になり、`_full_precision_mm` になる（`ops.py:1384-1385, 1967-1969`）。中身は `weight.dequantize()` の後に bf16 の `F.linear`（`ops.py:1599-1600`）で、方式 B と同じ。
- `full_precision_matrix_mult` の印も同じ経路にする（`ops.py:1381-1383`）。
- Ada: 入力は固定の `input_scale` で量子化する（`ops.py:1660-1665`）。`input_scale` がなければ倍率は **1.0** になる（`quant_ops.py:107-108`）。fc2 に full precision（元の精度での計算）の印がついている理由は、これで説明がつく。

### e. 方式 C の道具 [P]

- `quantize_int8_convrot_weight(weight, group_size, stochastic_rounding=0)` は、`TensorWiseINT8Layout.quantize(convrot=True, per_channel=True)` が呼び出す先である（`tensor/int8.py:117-128`）。ComfyUI の `requantize_from_float` も `requantize_kwargs` を通してここに行き着く。`__all__` には入っていない。
- eager は `quantization.py:899-907`（独自の演算は 1121 行）で、CPU でも GPU でも動く。CUDA の一体化した版は `cuda/__init__.py:1700-1740`。
- `stochastic_rounding=0` なら偶数への丸めで、結果は毎回同じ（決定的）。
- 注意: 回転は重みの型のまま計算される（`int8_utils.py:53`）→ fp32 で渡すこと。
- [I] CPU の eager と CUDA で、結果がビット単位まで一致するとは限らない。
- ライセンスは Apache-2.0（NOTICE つき）。

### f. 配布物（wheel）[P]

- `cp312-abi3-win_amd64`。`.pyd` の中の文字列に sm_75、sm_80、sm_89、sm_120f がある（文書にある Windows の既定 `75-virtual;80;89;120f` と同じ）→ Ampere と Ada に対応。w4a8 のカーネルの記号はすべてある。
- **注意:** `cuda.w4a8_int8_linear` は、`cutlass_int8_dequant` が False を返すと、黙って `eager_w4a8_int8_linear`（直接読み込んだもの）に切り替わる（`cuda/__init__.py:2530-2542`）。`disable_backend("eager")` でも止まらない → 本製品の `require_cuda_int8` の「遅い経路に黙って落ちず、はっきり失敗させる」規則が、そのままでは効かない。
- w4a8 のカーネルは x を自分で量子化する。量子化済みの x を受け取るカーネル `w4a8_codebook_gemm_chunked`（2447 行）は内部用である。
- 0.2.41 では、詰め方の選択肢「mma_packed」が加わった（`ck_w4a8_packed.py:51-62`）。0.2.37 にはない。

### §1 から言えること

- 方式 A: w4a8 は CC 8.0 以上で専用のカーネルが使える。fp8 の専用の計算は Ada だけ → Ampere では fp8 に方式 B か C が要る。
- 方式 C: w4a8 では劣化がなく簡単（`i8` に復元して `s_channel` をそのまま使う）だが、4 ビットで減らしたメモリが元に戻る。fp8 では丸めがもう一度加わる。
- 方式 B: いつでも可能。

## 2. LTX 側の前例（`engine/sft_quant`）[P]

- a. 計算のたびに逆量子化する: `_quant_linear_forward`（`quant_service.py:274-293`）。ConvRot は重みの側で戻す（`dequant.py:102-110, 130-132`）: 256 列の組ごとに `(q·s) @ H`。w4a8 の数式（111-134 行）は CK の eager と同じだが、G=16 に決め打ちで、**符号帳が必須**（`index_select`、121 行）。
- b. `dequant.py` が読み込むのは torch と `sft_quant_format`（`sft_reader` は必要になったときだけ）。`engine/__init__.py` は空。engine_h3 はすでに `engine.sft_quant.sft_reader` を読み込んでいる（`sft_load.py:38`）→ `dequant.py` は H3 の venv から読み込める。`quant_service.py` は `ltx_core` が要る（116、197、350 行）→ 使い回せない。
- c. `input_scale` は読み飛ばす（`sft_quant_format.py:143`、`quant_service.py:126`）。`full_precision_matrix_mult` は無視する（592 行）。計算はすべて bf16。
- d. `inspect()` は LTX 専用。`layer_schemes(path, header, prefix)` は汎用で、層 → 方式（fp8、fp8_scaled、int8、int8_convrot、w4a8）を返す。付属のテンソルの規則は `SCHEME_TABLE`（121-138 行）。`h3_format` は量子化の判定をこれに任せられる。H3 に固有で残る検査は、枝刈り版の表、接頭辞（`model.model.`、`net.`）、`weight_shape()` を使う形の表、付属のテンソルと `input_scale` まで広げた `_DIT_KNOWN` の正規表現。`layer_schemes` は G=8／4 を拒否し（545-550 行）、符号帳のない w4a8 を拒否し（562-565 行）、`convrot_groupsize` が 256 でないものを拒否する。メタデータだけの印を受け付け（メタデータが優先）、接頭辞は何でも受け付ける。

§2 から言えること: `dequant.py` は方式 B に使い回せるが、G が 16 以外のものと、符号帳のないものへの対応が要る。

### 検証用の 3 種類のファイルに当たる拒否（原文の表 T2 を、形式ごとの箇条書きに直したもの）[P]

fp8_scaled（Comfy-Org の枝刈り版）:

- `h3_format.py:359-374` の形の表（[N,K] を期待）: 通る。
- `h3_format.py:245-259` の「はぐれ」の検査（I8 でない重みに `weight_scale` や `comfy_quant`）: **拒否**。
- `h3_format.py:269-274`（層ごとの `.comfy_quant` テンソルが必要。メタデータだけの印は不可）: 通る。
- `h3_format.py:105-109` の `ACCEPTED_FORMATS = int8_tensorwise`: **拒否**。
- `h3_format.py:164-183` の `_DIT_KNOWN`: `input_scale` が「未知」として `dit_key_map_for`（`pipeline_h3.py:561-574`）で捨てられる（meta のままのバッファが残ることになる）。
- `sft_quant_format.layer_schemes`: 「fp8_scaled」として受け付ける（`weight_scale` は `()` か `(1,)`）。`input_scale` は読み飛ばし（143、561 行）、`full_precision_matrix_mult` は無視（592 行）。
- メタデータだけの印: 受け付ける。テンソルよりメタデータが優先（596-609 行）。

asym_w4a8_int8:

- 形の表: **拒否**（詰め込んだ [N,K/2]）。
- `h3_format.py:267-268`（I8 の層には `weight_scale` が必要）: **拒否**（代わりに `weight_s_rel`／`weight_s_channel` を持つ）。
- `h3_format.py:269-274`: ヘッダだけに印がある Kijai のファイルは**拒否**。
- `ACCEPTED_FORMATS`: **拒否**。
- `_DIT_KNOWN`: `weight_s_rel`／`weight_s_channel`／`weight_codebook` が「未知」として捨てられる（fp8 と同じく meta のまま残る）。
- `sft_quant_format` で group_size が 16 でない: G=8／4 の層は**拒否**（545-550 行）。
- `sft_quant_format` で `weight_codebook` がない: **拒否**（「missing aux」、562-565 行。付属のテンソルの組が `_W4A8_AUX`〔125-129 行〕と一致しなければならない）。
- `sft_quant_format` で `convrot_groupsize` が 256 でない: **拒否**（540-544 行）。
- メタデータだけの印: 受け付ける。

int8_tensorwise の行ごとの倍率（per_row）:

- 形の表、はぐれの検査、`ACCEPTED_FORMATS`: 通る。
- `h3_format.py:269-274`: ヘッダだけに印があるファイルは**拒否**。
- `h3_format.py:276-280`（倍率は F32 [N,1]）: 通る（倍率がスカラーの int8 なら拒否される）。
- `sft_quant_format` で `convrot_groupsize` が 256 でない: **拒否**。`per_row` の印は無視（読むのは format／convrot／convrot_groupsize／group_size だけ）。メタデータだけの印は受け付ける。

接頭辞について: `layer_schemes` はどんな接頭辞でも受け付ける。`inspect()` は LTX 専用（`__metadata__.config` が必要〔402-410 行〕、`transformer_blocks`〔371-384／413-420 行〕、connectors〔430-432 行〕）。

## 3. H3 の新しい線形層のクラスが満たすべき条件 [P]

- 求められる口: `forward(x)`（bf16 で出す）。`forward_rows(x, n0, n1, lora_xa=None)` は呼ばれるたびに行を切り出し、LoRA も自分で掛ける（`quant_linear.py:207-248`）。qkv では `shares_input`、`quantize_input`、`forward_rows_q` が条件なしに呼ばれる（`lowvram.py:148-152`）。LoRA には `in_features`／`out_features` が要る（`lora.py:193-199`。isinstance の検査は `H3Int8Linear` か `nn.Linear` だけを対象にする）。
- int8 や [N,K] を前提にしている場所: 形の表 `h3_format.py:359-379`、I8 だけを数える `_quant_layers`（245-280 行）。`swap_in_int8_linears` は必ず H3Int8Linear を作る（`quant_linear.py:497-536`）。`normalize_dtypes` が飛ばすのは H3Int8Linear だけ（`dit.py:499`）。float8 は浮動小数点として数えられるので、**新しいクラスも飛ばす対象にしないと、fp8 の重みと F32 の `s_channel` が bf16 に変えられてしまう。**
- 読み込み: `load_state_dict(assign=True)` はファイルの型を保つ（U8 の `s_rel` は U8 のままなので `.view` が要る）。宣言した形（`[N,K/2]`、`[N,K/G]`、`()`）は厳密に守られる。`_DIT_KNOWN` が知らない付属のキーは捨てられ、その後の「まだ meta のまま」の検査で失敗する（`sft_load.py:98-105`）。
- block swap の置き場: パラメータとバッファをすべて、`view(uint8)` でふつうのテンソルとして詰める（`block_swap_prefetch.py:284-301, 455-460, 644`）。0 次元の fp8 のスカラーは問題ない。連続していないものは一度だけ連続にされる。パラメータは `.data=` でつなぎ直され、バッファは差し替えられる → forward は呼ばれるたびにテンソルを引き直さなければならない。
- `adaln_t_table`: BF16 の表は BF16 のまま読まれ、`normalize_dtypes` が F32 に変える（`dit.py:501`、`mod is self`）。値は正確に保たれる。

§3 から言えること: どの形式でも `forward_rows` と、`normalize_dtypes` から外す指定が要る。方式 B では VRAM 16GB mode のとき、fc1／fc2 を行の区切りごとに 1 回ずつ逆量子化する。区切りの数は ⌈s/4096⌉（`lowvram.py:188`）。

## 4. メモリ（計算による。`mem_out.txt`）。線形層だけ [P]

| 形式 | 1 ブロック | 50 ブロック |
|---|---|---|
| int8 | 367.7 MiB | 17.96 GiB |
| fp8 | 367.5 MiB | 17.94 GiB |
| w4a8（G16） | 207.0 MiB | 10.11 GiB |
| w4a8（G8） | 229.9 MiB | 11.23 GiB |
| w4a8（G4） | 275.9 MiB | 13.47 GiB |
| bf16 | — | 35.89 GiB |

- 1 ブロックの線形層は 4 本: `attn.qkv_proj`（N 21504 × K 5376）、`attn.out_proj`（5376 × 7168）、`mlp.fc1`（28672 × 5376）、`mlp.fc2`（5376 × 14336）。要素の数は 1 ブロックで 385,351,680、50 ブロックで 19,267,584,000。
- 10 進の GB で書くと、50 ブロックで int8 19.28 GB、fp8 19.27 GB、w4a8 G16 10.85 GB（G8 12.05、G4 14.46）。
- 方式 B での、層ごとの一時的な bf16 の写し: fc1 294.0 MiB、qkv 220.5 MiB、fc2 147.0 MiB、out_proj 73.5 MiB。LTX 側の逆量子化の最大は f32 の写し 2 つ分ほど: fc1 で約 1,176 MiB、qkv で約 882 MiB。

§4 から言えること: w4a8 に方式 C を使うと int8 と同じ大きさになる。方式 A と B は約 10 GiB を保つ。

## 実験でしか答えの出ない問い

1. H3 の形の行列で、`cutlass_int8_dequant` は Ampere／Ada で True を返すか。それとも w4a8 は黙って eager に落ちるか。
2. `codebook=None` のときの CUDA の w4a8 の復元は、eager と同じ一様な `c-8` か、組み込みの表か。eager と突き合わせて確かめる。
3. H3 の大きさでの方式 A・B・C の速さ。VRAM 16GB mode の区切りによる倍数も含めて。
4. fp8 に方式 C を使ったときの品質: fp8 → int8 の再量子化の誤差と、`input_scale` を無視する影響。
5. CPU の eager と CUDA の `quantize_int8_convrot_weight` は、ビット単位で一致し、毎回同じ結果になるか。
6. h3ErosMax の w4a8 の層について、`_quantization_metadata` の中身（調査時の書き出しが途中で切れていた）。
7. torch 2.13 の `F.scaled_mm` は、行を切り出して転置した fp8 の重み（`forward_rows`）を、写しを作らずに受け付けるか。
