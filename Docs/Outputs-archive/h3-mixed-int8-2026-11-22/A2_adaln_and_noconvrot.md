# A2: I8 の adaln_proj の扱いと、ConvRot なしの int8 の計算経路

調べた版: HEAD `c773dd6d`。作業は読むことだけで、ほかにしたのは WEB の取得 3 件。「推測」と書いたもの以外は、コードか取得した原文で確かめた事実。

## 結論

**I8 の `adaln_proj` は、製品の中で倍率を掛けずに計算される。** 差し替えでは `H3Int8Linear` になる。しかし `H3AdaLNProj.forward` は `self.linear.weight` を直接読み、`.to(float32)` で float32 にしてから `F.linear` に渡す。そのため int8 の生の符号（−127〜127）がそのまま重みとして掛かり、`weight_scale` は使われない。bias は正しく足される。

この結果、X では 50 ブロックすべてで、Y ではブロック 2〜48 で、AdaLN の変調量が出力の行ごとに約 1/倍率 倍に膨らむ。値の大きさは推測。ブロック状のノイズの最有力の原因と見る（推測）。

この経路は製品で一度も通っていない。公式の重みの adaln_proj は F16 だからである。

---

## 調査 1: I8 の adaln_proj

### a. 定義と差し替え
- `H3DiTBlock.adaln_proj` は `H3AdaLNProj(rank, hidden, expand=6, modality_num=3)`（`engine_h3/dit.py:360`）。
- その中身は `self.linear = nn.Linear(rank, expand*hidden*modality_num, bias=True, dtype=torch.float32)`（:314）。形は `nn.Linear(8, 96768, bias=True)`。
- 印の付いた層の名前は `blocks.N.adaln_proj.linear`。
  - 検査器はブロックの 4 層だけに印を限らない。`_read_markers` は `.weight` を持つ層ならどれでも印を読む（`h3_format.py:345-348`）。
  - `_check_int8` が見るのは、I8 `[96768,8]` の重みと、F32 `[96768,1]` の倍率だけ（:405-412）。
  - 形の表の `adaln_proj.linear.weight: (18*hidden, rank)`（:579）とも合う。
  - 補助テンソルの `weight_scale` と `comfy_quant` は `_quant_aux` で知っている名前に入る（:497-499, 592）。
  - したがって X は「量子化層 250」で受理される。
- `swap_in_int8_linears`（`quant_linear.py:611-651`）は `cfg.quant` に載った名前を順に処理し、親モジュールと末端の名前で差し替える。
  - `adaln_proj.linear` は `nn.Linear` なので、`H3Int8Linear` に差し替わる。
  - 形式が int8 ならバイアスつきの層も断らない（:658-665）。断るのは fp8 と w4a8 だけ（:668-669）。

### b. bias
- `H3Int8Linear` は bias を持てる。dtype は `bias_dtype_of(name)` が返す、ファイルの bias の dtype（`quant_linear.py:237-242`、`pipeline_h3.py:1583-1585`）。
- `forward` と `forward_rows` は、bias を bf16 にして `int8_linear` に渡す（:253-255, 289-293）。
- 読み込みでは、`adaln_proj.linear.bias` の名前が `H3Int8Linear.bias` にそのまま当たる。`assign=True` で代入され、読み捨てもエラーも起きない（`sft_load.py:90`）。
- `normalize_dtypes` は `H3QuantLinear` を飛ばす（`dit.py:499-500`）。
  - `f32_modules` には `m.linear` の H3Int8Linear が入る（:496）が、先に飛ばされるので何もしない。
  - そのため bias はファイルの dtype のまま残る。

### c. 差し替わらない場合
該当しない（a のとおり、必ず差し替わる）。

### d. forward での使い方
- `H3AdaLNProj.forward`（`dit.py:316-322`）の中身。
  ```
  w = self.linear.weight.to(torch.float32)
  b = self.linear.bias.to(torch.float32)
  x = F.linear(t_emb.to(torch.float32), w, b)
  ```
- **モジュールとして呼ばずに、重みを直接読んでいる。**
  - `H3Int8Linear.weight` は int8 の Parameter（`quant_linear.py:228-231`）で、`.to(float32)` は生の符号を float32 にするだけ。
  - `weight_scale` には触れない。
  - そのため int8 のカーネルも、入力の量子化も通らない。
- 呼び出し元:
  - off の経路: `dit.py:375`
  - VRAM 16GB mode: `lowvram.py:135`
  - 最終層: `dit.py:608`、`lowvram.py:240`
  - いずれも同じ `forward` を通る。
- 入力 `t_emb` は `adaln_embed` が返す値で、`adaln_t_table` を線形補間した float32 の `[M,8]`（`dit.py:580-592`）。M は時刻の数で 1〜3。
- 出力は float32。続く `_modulate` と `_gate` も float32 で計算する（:325-345）。

### e. X での変調量の見立て
- 事実から言えること: 出力の行 r は `(w_r·t)/s_r + b_r` になる。s_r はファイルの `weight_scale[r]`。
- 推測（行ごとの最大値で量子化したと仮定した場合）: s_r ≈ max|w_r|/127。その場合、shift・scale・gate の重み側の項は行ごとに 127/max|w_r| 倍になり、bias の項だけが正しい大きさで残る。
- 倍率の値はファイルの数値を見ないと決まらないので、何倍かは未確認。
- 推測: すべてのブロックで、`x*(1+scale)+shift` と `x + gate*other` が桁違いに崩れる。それでもエラーは出ず、完走してノイズになる。
- Y もブロック 2〜48 の adaln が I8 なので、ブロック 0・1・49 の qkv の不具合（前回の報告）を直しても、同じ崩れが残る（推測）。

### f. ComfyUI の計算

出典（いずれも 2026-11-22 に取得）:
- https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/comfy/ldm/minimax/model.py
- https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/comfy/ops.py
- https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/comfy/quant_ops.py

取得は、ページを要約して答える取得の道具で行った。引用と行の目安はその道具が返したものなので、行番号は概算。

- `AdalnProj`（model.py）:
  - `self.linear = operations.Linear(t_dim, expand*hidden*modalities, bias=True, dtype=...)`
  - `forward` は `x = self.linear(silu(t_emb) if self.apply_silu else t_emb)`。
  - **モジュールとして呼んでいる。** 重みの直接読み、`cast_bias_weight`、`dequantize` は使っていない。
  - 枝刈り版（「curve」形式）では `apply_silu=False`、`adaln_dtype=float32`。
- 量子化つきの Linear（ops.py、概算 1,190〜1,500 行）:
  - 印から `format` を読む。`int8_tensorwise` の分岐では `convrot` を、層の記述にあればそこから、無ければ `params` から読む。どちらにも無ければ False。groupsize の既定は 256。
  - `per_row` は読まない。
  - 計算の dtype は入力の dtype に従う（`compute_dtype = input.dtype`）。
  - 入力を量子化するかどうかは `QUANT_ALGOS[...]["quantize_input"]` で決まり、int8 では False（quant_ops.py）。
  - 最後に `F.linear(input, QuantizedTensor の重み, bias)` を呼ぶ。
- comfy-kitchen 側の受け手 `_handle_int8_linear`（ローカルの `comfy_kitchen/tensor/int8.py:270-294`）:
  ```
  torch.ops.comfy_kitchen.int8_linear(input.contiguous(), qdata.contiguous(), weight_scale, bias, out_dtype=入力の dtype, convrot, convrot_groupsize)
  ```
- まとめると、ComfyUI は adaln を int8 のカーネルで計算する。8 次元の float32 の入力は、カーネルの中で行ごとに int8 へ量子化され、倍率が掛かり、出力は float32 になる（m は 1〜3 なので、m=1・2 は GEMV の分岐。CUDA 側の `__init__.py:2169-2188`）。

---

## 調査 2: ConvRot なしの int8

### a. `convrot` と groupsize を使う全箇所（`engine_h3` 全体を grep した結果）
- `H3Int8Linear.forward`: `quant_linear.py:262-263`。旗をそのまま渡すだけ。
- `H3Int8Linear.forward_rows`: :300-301。同じく旗をそのまま渡すだけ。重みの行は呼ぶたびに切り出す。
- `shares_input`: :321。`not convrot` または groupsize が 256 でなければ False。
- `quantize_input`: :344。`shares_input` が真のときだけ呼ばれる。
- `forward_rows_q`: convrot を使わない。`shares_input` が真のときだけ呼ばれる。
- `add_lora` と `lora.apply`: convrot を使わない。LoRA は生の入力を読む（`lora.py:11`）。
- `extra_repr`（:370）と読み込みの記録（`pipeline_h3.py:833`）: 表示だけ。
- 製品のコードには、回転行列（Hadamard）の用意、groupsize で割り切れることの前提、`weight_tiled`、回転を鍵にしたキャッシュは無い（grep で 0 件）。

`convrot=False` のときに実行される行:
- `forward`: :246-252 → :256 の `int8_linear(..., convrot=False)`。
- CUDA 側の `int8_linear`（`backends/cuda/__init__.py`）:
  1. :2061 の `_fused_convrot_ok` は False になる。
  2. :2091-2126 の m=1 の分岐は、K が 2560 を超えるので通らない（ブロックの K は 5376 以上）。
  3. :2155-2165 で `_C.quantize_int8_rowwise` が入力を回転せず、行ごとに量子化する（倍率は `[m,1]`）。
  4. :2169-2188 の m≤2 の GEMV か、:2190 以降の CUTLASS・cuBLAS の行列積で、行ごとの重みの倍率を掛ける。

α 版（`7a7ee81d`）から、旗はそのまま渡していた（`git show 7a7ee81d:engine_h3/quant_linear.py` の :118-119）。

### b. 分岐（`lowvram.py:150-154`）
```
if qkv.shares_input(h):
    proj = functools.partial(qkv.forward_rows_q, *qkv.quantize_input(h), lora_xa=xa)
else:
    proj = functools.partial(qkv.forward_rows, h, lora_xa=xa)
```
- 下の `forward_rows` の経路の実績:
  - `_probe_shared_quant` が失敗したときと、計算能力が 8.0 未満のとき（`quant_linear.py:511-529`）も、同じこの経路を通る。
  - 共有が入る前（`4d9d73a8` より前）は、int8 の qkv はすべてこの経路で、VRAM 16GB mode の門 A-G2・A-G3b は公式の ConvRot つきで合格した（記録簿 §169。前回の報告での確認による）。
  - w4a8 は常にこの経路（記録簿 §211.6 の G3・G4）。
- ConvRot なしの重みで、この経路を実際の GPU で回した実績は無い。

### c. 呼び方の比較
| 項目 | ComfyUI | 製品 |
|---|---|---|
| 呼ぶ関数 | `torch.ops.comfy_kitchen.int8_linear(input, qdata, weight_scale, bias, 出力 dtype の番号, convrot, groupsize)` | `_ck().int8_linear(x2d, weight, weight_scale, bias, out_dtype=bf16, convrot, groupsize)` |
| `weight_tiled` | 0.2.37 にこの引数は無い | 同じ |
| 入力の dtype | 入力の dtype のまま | bf16 にする（`quant_linear.py:248-252`） |
| 出力の dtype | 入力の dtype | bf16 に固定 |
| bias | そのまま渡す | bf16 にする |

- ブロックの 4 層の入力は bf16 なので、両者は一致すると見る（推測）。
- 違いが出るのは、入力が float32 の adaln のような層だけ。

### d. 「ConvRot を前提にする」と決めた記述
- 設計正本の裁定 6「重みの形式は int8 ConvRot とする」（`MINIMAX_H3_DESIGN.md:46`。研究ノート :250 と同じ）。
  - これは受ける重みの形式を選んだもので、のちに裁定 101 で更新された（:47）。
  - エンジンの計算を ConvRot つきに限るとは書いていない。
- 設計正本 §5（:1237）には「ConvRot の有無と groupsize は層ごとの記述から取る」とある。
- 文書で ConvRot つきを条件にしているのは、VRAM 16GB mode の qkv の入力量子化の共有（:1244）だけ。
- `git log --all --grep=convrot` にも、エンジン全体を ConvRot 前提にした決定は見当たらない。
- 推測: オーナーの記憶は、裁定 6 と、実機で回した重みがすべて ConvRot つきだったことに由来する。
