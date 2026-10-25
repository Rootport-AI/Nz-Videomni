# P1 台帳 §1-105 ①「QKV の活性化量子化の共有」の実装計画（MiniMax H3・VRAM 16GB mode）

起草日 2026-10-25。対象はブランチ dev の作業ツリー。本書は計画だけで、コードは 1 行も変えていない（読んだのはリポジトリのファイル・`.venv-engine-h3` の comfy-kitchen 0.2.37 のソース・既存の計測記録だけ。GPU は使っていない）。出発点は `R1_investigation.md` の「実装の形の案（最小）」、短縮の実測は `R2_measurement.md`。

本文では、確かめた事実には根拠（ファイルと行）を添え、推測には「（推測）」と書く。

---

## 結論

1. **変えるのはエンジンの 4 ファイルとテストだけ**（合計で本体 約 +175 行・テスト 約 +130 行）。
   - `engine_h3/quant_linear.py`（約 +150 行）: 起動時の検査 `_probe_shared_quant`、共有してよいかの判定 `shares_input`、1 回だけの量子化 `quantize_input`、量子化済みの入力での行列積 `forward_rows_q`、`int8_linear` の量子化より後の部分を写した補助関数 `_int8_gemm_prequantized`。
   - `engine_h3/lowvram.py`（約 +8／−5 行）: 137〜153 行のヘッド群のループの前で 1 回だけ量子化し、直後に `h` を捨てる。
   - `engine_h3/lora.py`（約 +4／−3 行）: `apply` が `x` を見ずに行数を得られるようにする（`xa` の行数から導く）。
   - `engine_h3/pipeline_h3.py`（約 +4 行）: ロードのログに検査の結果を 1 行、`done` の `h3.phases.denoise.vram16` に 1 項目。
2. **数値は現行の on と全桁同じになる見込み**（論法は §3）。同じ量子化の核を同じ入力と同じ引数で 1 回呼び、その結果を、現行と同じ行列積の核に同じ形で 21 回渡す。量子化の核が決定的なことは R2 で実測済み（2 回の結果が `torch.equal`）。
3. **速度**: 1 ステップの短縮の上限は 1.447 秒（R2 の実測。I2V 896×1152×124 フレーム・S＝39,764）。20 ステップで約 29 秒、Turbo の 4 ステップで約 5.8 秒。
4. **VRAM**: 注意の区間の山は約 0.5U（約 204 MiB）下がる。ただし**ノイズ除去全体の割り当てのピークは変わらない可能性が高い**（推測）。設計正本 §6.5「山の内訳」の概算では FFN の区間（約 5.3U）が注意（約 5U）より高く、今回の変更は FFN の区間に触れないため。門の条件は「下がること」ではなく「上がらないこと」にする（§4）。
5. **共有の経路に乗れないとき（comfy-kitchen の内部が見つからない・Turing・形が条件の外）は、現行の `forward_rows` にそのまま戻す。** ロードは拒否しない（裁定 1）。
6. 監督に決めてほしい点は §11 に 10 件ある。中でも急ぐのは、(1) 行数の渡し方を「`xa` から導く」にしてよいか（裁定 3 の字義との関係）、(2) `done` に 1 項目足すか（足すなら仕様書の版をどうするか）、(3) 検査を `require_cuda_int8` の中から呼んでよいか、の 3 つ。

---

## 1. 変更の一覧

行番号は 2026-10-25 時点の dev の作業ツリー。

### 1.1 `engine_h3/quant_linear.py`（253 行 → 約 400 行）

| 置く場所 | 足す・変えるもの | 中身 |
|---|---|---|
| 冒頭の docstring（1〜30 行） | 段落を 1 つ追加（約 +8 行） | 「VRAM 16GB mode の qkv だけは、`int8_linear` を量子化と行列積の 2 段に分けて呼ぶ（設計正本 §5 の例外。台帳 §1-105 ①）。量子化は comfy-kitchen の `backends.cuda.quantize_int8_rowwise_convrot64`、行列積は `int8_linear` の量子化より後の部分を写した `_int8_gemm_prequantized`。どちらも comfy-kitchen 0.2.37 の非公開の内部なので、起動時に記号と引数の並びを調べ、無ければ現行の経路に戻す」 |
| 41 行の `_ck_module = None` の下 | モジュールの状態 2 つ（+3 行） | `_SHARED_CB = None`（検査に合格したときの `comfy_kitchen.backends.cuda` モジュール。None なら共有しない）と `_SHARED_REASON = "not probed"`（不合格の理由。合格なら None） |
| `forward_rows` の後（164 行の後・166 行の `extra_repr` の前） | `shares_input(self, x) -> bool`（約 +18 行） | §2.2 の条件をすべて満たすときだけ真。`_SHARED_CB is None` なら即座に偽 |
| 同上 | `quantize_input(self, x) -> tuple[Tensor, Tensor]`（約 +10 行） | `forward_rows` の 140〜144 行と同じ整形（2 次元化・bf16 化・`contiguous()`）をしてから `_SHARED_CB.quantize_int8_rowwise_convrot64(x2d, int(self.spec.convrot_groupsize))` を呼び、`(xq int8 [M, K], xs f32 [M, 1])` を返す |
| 同上 | `forward_rows_q(self, xq, xs, n0, n1, lora_xa=None) -> Tensor`（約 +25 行） | `forward_rows` と同じく、重み・スケール・bias を**毎回** `[n0:n1]` で切る（block swap が `.data` を差し替えるので、切片を持たない。145〜149 行と同じ bias の扱い）。`_int8_gemm_prequantized(xq, xs, self.weight[n0:n1], self.weight_scale[n0:n1], bias, torch.bfloat16)` を呼ぶ。戻り値は 2 次元 `[M, n1 − n0]`（呼び出し側の `h` はもともと 2 次元なので、`forward_rows` の `reshape(*lead, …)` と同じ形）。LoRA が付いていれば `lora.apply(self, None, out, n0, n1, xa=lora_xa)`。LoRA が付いているのに `lora_xa` が None なら `RuntimeError`（`x` が無いので計算できない。lowvram は必ず渡すので、起きれば配線の誤り） |
| `require_cuda_int8` の末尾（210 行と 211 行の間） | `_probe_shared_quant()` の呼び出し（+1 行） | 検査は例外を投げない。`require_cuda_int8` の戻り値と拒否の条件は変えない |
| `require_cuda_int8` の後（212 行の後） | `_probe_shared_quant() -> None`（約 +45 行）・`shared_quant_status() -> tuple[bool, str \| None]`（+4 行） | §2.4。`_SHARED_CB` と `_SHARED_REASON` を設定する |
| 同上 | `_int8_gemm_prequantized(xq, xs, weight, weight_scale, bias, out_dtype) -> Tensor`（約 +40 行） | §2.1 の写し |

- `forward`・`forward_rows`・`swap_in_int8_linears` は 1 行も変えない（off の経路と、共有に乗れないときの戻り先）。

### 1.2 `engine_h3/lowvram.py`（194 行 → 約 200 行）

137〜153 行を次の形にする（`functools` の import を 1 行足す）。

```python
    qkv = attn.qkv_proj
    xa = _lora.xa_for(qkv, h)   # LoRA's f32(h) @ A^T once per block; None without LoRA (before h goes)
    if qkv.shares_input(h):
        # one quantization of h for the 21 slices (台帳 §1-105 ①); h is not needed after it
        proj = functools.partial(qkv.forward_rows_q, *qkv.quantize_input(h), lora_xa=xa)
    else:
        proj = functools.partial(qkv.forward_rows, h, lora_xa=xa)
    del h   # shared: freed here (the int8 copy is half its size); otherwise the partial keeps it, as before
    for h0 in range(0, heads, plan.head_group):
        h1 = min(h0 + plan.head_group, heads)
        g = h1 - h0
        q = proj(0 * inner + h0 * d, 0 * inner + h1 * d).view(s, g, d)
        k = proj(1 * inner + h0 * d, 1 * inner + h1 * d).view(s, g, d)
        v = proj(2 * inner + h0 * d, 2 * inner + h1 * d).view(s, g, d)
        ...（144〜152 行は変えない）
    del proj, xa
```

- 戻り先（`shares_input` が偽）の寿命は現行と同じ。`h` は `partial` が持ち、ループの後の `del proj` で放される（現行の 153 行の `del h, xa` と同じ位置）。
- `lambda` で包まない理由: `h` を参照する入れ子の関数があると、`h` がセルの変数になって寿命が読みにくくなる。`functools.partial` は束ねた値だけを持つ。
- 冒頭の docstring（6〜13 行の「Nothing here is new maths」と 16〜19 行の注意の箇条）を直す（§8）。

### 1.3 `engine_h3/lora.py`（316 行 → 約 317 行）

`apply`（283〜316 行）で、行数を `x` ではなく、`xa` があれば `xa[0].shape[0]` から取る。

```python
    m = xa[0].shape[0] if xa is not None else x.reshape(-1, x.shape[-1]).shape[0]
    x2d = None if xa is not None else x.reshape(-1, x.shape[-1])
    ...
    if out2d.shape[1] != n1 - n0 or out2d.shape[0] != m: raise ...（300 行の文言の x2d.shape[0] を m に）
    for r0, r1 in row_chunks(m, _C.VRAM16_ROW_CHUNK):（304 行）
```

- **新しい引数を足さない**。`xa` は `xa_for` が `[M, r]` で作るので、行数は既存の値から導ける（「既存の値から導けるものに新しいフィールドを足さない」）。オーナー裁定 3 の「行数を別に渡すように直す」の字義（引数 `rows` を足す）と違うので、監督の確認が要る（§11 の 1）。
- `xa` が None のとき（off の経路の hook・LoRA の付いた `forward`）は今と同じ。`xa` があるときの算術は変わらない（`x2d` は `xa` があると使われていない。306 行）。
- docstring（20〜23 行と 286〜287 行）に「`forward_rows_q` は `x` を持たないので None を渡し、`xa` を必ず渡す」を足す。

### 1.4 `engine_h3/pipeline_h3.py`（1,175 行 → 約 1,179 行）

| 置く場所 | 中身 |
|---|---|
| 519〜520 行（`backends = require_cuda_int8()` の直後） | `ok, why = shared_quant_status()` を取り、`self._log(...)` で 1 行（文言は §5）。ロードは続ける |
| 861〜864 行（`_vram16_begin` の `record` の辞書） | `"qkv_quant_shared": shared_quant_status()[0]` を 1 項目（足すかどうかは §5・§11 の 2） |

### 1.5 テスト `tests/test_h3_engine.py`（約 +130 行）

§6 に詳しく書く。CPU テスト 6 本（約 +95 行。1590 行付近の偽物の隣と、1716 行付近の VRAM 16GB mode のテストの後）と GPU テスト 2 本（約 +35 行。1884 行の `test_gpu_forward_rows_join_equals_forward` の後）。

### 1.6 触らないもの（裁定 5）

`engine_h3/dit.py`（off の経路。読むだけ）・`engine/`（共有部品）・`out_proj`／`fc1`／`fc2` の呼び方・`engine_h3/constants.py`（定数は増やさない）・ワーカー・アプリ側・操作パネル・利用者向けの設定。

---

## 2. 補助関数が写す範囲と、共有に乗る条件

対象は `.venv-engine-h3/Lib/site-packages/comfy_kitchen/backends/cuda/__init__.py`（以下 `cb`）の `int8_linear`（2035〜2289 行）。

### 2.1 `int8_linear` の行と補助関数の対応

H3 の qkv の呼び出しは `forward_rows` の 150〜158 行のとおり、`x`＝bf16 の `[S, 5376]`（連続）・`weight`＝int8 の `[1024, 5376]`（連続なビュー）・`weight_scale`＝f32 の `[1024, 1]`・`bias`＝None（H3 の qkv は bias なし。`dit.py:229`）・`out_dtype`＝bf16・`convrot=True`・`convrot_groupsize=256`・`input_act`／`residual` は既定の None。

| `int8_linear` の行 | 中身 | 補助関数での扱い |
|---|---|---|
| 2050〜2051 | `residual` の検査 | 写さない（`residual` を渡さない） |
| 2052 | `x_2d` の整形 | `quantize_input` の整形（`forward_rows` の 140〜144 行と同じ）で済んでいる |
| 2053〜2054 | `weight` を連続に | **写す**（切片は連続なので実際には何もしない） |
| 2055 | `stream_ptr`（現在のストリーム） | **写す**（`xq.device` で取る。量子化と行列積は同じストリームに積まれる） |
| 2060〜2070 | `k_act`・`_fused_convrot_ok`・`input_act` の適用 | 判定は `shares_input` へ（§2.2）。`input_act` は None なので適用は起きない |
| 2072〜2075 | `m`・`k`・`n` と形の assert | **写す** |
| 2077〜2078 | `out_dtype`・`output_dtype_code`（`cb.DTYPE_TO_CODE`） | **写す** |
| 2079〜2089 | `is_2d_output`・`fused_residual_done`・`_finish` | 写さない。`residual` が None なので `_finish` は `out` をそのまま返す（`backends/_activations.py:77-78`）。入力は 2 次元なので形も変わらない |
| 2091〜2126 | m＝1 の専用の核（`int8_linear_m1`） | 写さない。`shares_input` が m > 2 を求めるので届かない |
| 2129〜2149 | ConvRot の融合核での量子化 | **`quantize_input` が受け持つ**（§3 の対応表）。2150〜2156 の別の量子化（融合核に乗らないとき）は、`shares_input` が融合核の条件を求めるので届かない |
| 2157〜2167 | ConvRot なしの量子化 | 届かない（convrot を求める） |
| 2169〜2188 | m ≤ 2 の GEMV（`int8_gemv_dequant`） | 写さない。m > 2 を求める |
| 2190〜2193 | `out`・`_int8_weight_scale_arg`・`bias_arg`（`_gemm_vector_arg`／`_empty_cuda_tensor`） | **写す**（bias が無ければ `_empty_cuda_tensor`） |
| 2195〜2209 | Turing の融合核（`_prefer_turing_fused_int8` かつ `_cuda_device_should_use_turing_kernels`） | 写さない。`shares_input` が `_cuda_device_supports_cutlass_int8_dequant`（compute capability の major ≥ 8。`cb:321-331`）を求めるので、Turing（7.5）は届かない |
| 2211〜2222 | `used_cutlass`・`_prefer_cublas_int8_fallback`（0.2.37 では常に False。`cb:534-540`）・`_DISABLE_CUTLASS_INT8`（環境変数 `COMFY_KITCHEN_DISABLE_CUTLASS`。`cb:1635`）・CUTLASS の対応 | **写す**（常に False の関数も呼ぶ。版が変わって中身が変わっても同じ判断になるように） |
| 2223 | `ws_cutlass` | **写す** |
| 2224〜2244 | 残差の融合（`cutlass_int8_dequant_residual`） | 写さない（`residual` が None で `resid_2d` が None になり、この枝は走らない） |
| 2245〜2255 | `_C.cutlass_int8_dequant(xq, weight, xs, ws, bias_arg, out, code, stream)` | **写す**（引数の並びも同じ） |
| 2256〜2287 | CUTLASS が偽を返したときの cuBLAS（`_C.cublas_gemm_int8` → `_C.dequantize_int8_linear`） | **写す。ただし Turing の詰め物（2258〜2265 の `use_turing_padding` が真の枝）は写さない**（Turing は届かない）。`padded_n = n` の枝だけ |
| 2289 | `return _finish(out)` | `return out` |

- 補助関数の中で使う `cb` の名前: `_C`・`DTYPE_TO_CODE`・`_int8_weight_scale_arg`・`_gemm_vector_arg`・`_empty_cuda_tensor`・`_wrap_for_dlpack`・`_prefer_cublas_int8_fallback`・`_DISABLE_CUTLASS_INT8`・`_cuda_device_supports_cutlass_int8_dequant`・`get_cublas_workspace`。`_C` の関数は `cutlass_int8_dequant`・`cublas_gemm_int8`・`dequantize_int8_linear`。
- `_DISABLE_CUTLASS_INT8` は呼ぶたびに `cb` のモジュール変数を読む（`int8_linear` も同じく呼ぶたびに読む）。値を補助関数の側に写し取らない。
- 補助関数には、写した元の行番号をコメントで付ける（版を上げたときに見比べるため）。

### 2.2 `shares_input` の条件（すべて真のときだけ共有）

| # | 条件 | 対応する `int8_linear` の行 | 判定に使う名前 |
|---|---|---|---|
| 1 | 起動時の検査に合格している | — | `_SHARED_CB is not None` |
| 2 | `self.spec.convrot` が真で、`self.spec.convrot_groupsize == 256` | 2062〜2063 | — |
| 3 | `x` が 2 次元で、最後の次元 K が `self.in_features` に等しく、`K % 256 == 0` かつ `256 <= K <= cb._CONVROT_FUSED_MAX_K` | 2064〜2065 | `cb._CONVROT_FUSED_MAX_K`（16,384） |
| 4 | 共有メモリが足りる | 2066 | `cb._convrot_fused_shared_memory_fits(x2d, K, 256)` |
| 5 | 行数 m > 2 | 2091（m＝1）・2169（m ≤ 2） | — |
| 6 | CUTLASS の int8 に対応する GPU（major ≥ 8。Turing を外す） | 2195〜2197・2220・2258 | `cb._cuda_device_supports_cutlass_int8_dequant(x2d)`（CPU のテンソルなら偽を返す。`cb:322-323`） |
| 7 | 重みとスケールが `x` と同じデバイスにあり、重みが int8・スケールが f32 | 登録の制約（`cb:4464-4486`） | — |

- #7 の理由: 共有の経路は comfy-kitchen の登録（`registry.get_implementation`）を通らないので、設計正本 §5 の「CUDA 経路に乗らない呼び出しは `NoCapableBackendError` で止まる（fail loud）」が効かない。重みが CPU に残っているような呼び出しは、共有せずに現行の `forward_rows` へ戻し、そこで今と同じく止まるようにする。
- x の dtype は条件にしない。`quantize_input` が `forward_rows` と同じく bf16 に直す。`h` はもともと bf16（`lowvram.py:128`。`_modulate` の出力）。
- **戻す分岐の場所は `lowvram.block_forward` の 1 か所だけ**（§1.2 の `if qkv.shares_input(h)`）。ブロックごとに 1 回判定する。`forward_rows_q` の中では判定しない（量子化の後に戻ろうとしても `h` はもう無い）。

### 2.3 判定が実際に偽になる場面（推測を含む）

- Turing（RTX 20 系）: #6 で偽。現行の経路のまま（遅い経路にも速くもならない）。
- comfy-kitchen の版が変わって記号や引数の並びが違う: #1 で偽。
- 製品の H3 で #2〜#5・#7 が偽になる場面は無いと見られる（推測。K は 5,376 で固定、S は数千行以上、qkv は全層 `convrot=True, g256`〔研究ノート §13.2〕、block swap は forward の前に重みを GPU に載せる）。

### 2.4 起動時の検査 `_probe_shared_quant`

`require_cuda_int8` の末尾から呼ぶ（triton・eager を切った後）。例外は投げず、`_SHARED_CB` と `_SHARED_REASON` を設定するだけ。調べる順:

1. `import comfy_kitchen.backends.cuda as cb`。`cb._EXT_AVAILABLE` が真で `cb._C` が None でない。
2. `cb` の属性が全部ある: `quantize_int8_rowwise_convrot64`・`_CONVROT_FUSED_MAX_K`・`_convrot_fused_shared_memory_fits`・`_cuda_device_supports_cutlass_int8_dequant`・`_prefer_cublas_int8_fallback`・`_DISABLE_CUTLASS_INT8`・`_int8_weight_scale_arg`・`_gemm_vector_arg`・`_empty_cuda_tensor`・`_wrap_for_dlpack`・`DTYPE_TO_CODE`・`get_cublas_workspace`。
3. `cb._C` の属性が全部ある: `quantize_int8_rowwise_convrot64`（Python の包みが中で呼ぶもの）・`cutlass_int8_dequant`・`cublas_gemm_int8`・`dequantize_int8_linear`。
4. **引数の並び**:
   - Python の関数は `inspect.signature` の引数名の並びを固定値と比べる。`quantize_int8_rowwise_convrot64` は `(weight_2d, group_size, stochastic_rounding, input_act, input_act_weight, input_act_eps)`（`cb:1588-1595`。確認済み）。`_convrot_fused_shared_memory_fits` は `(x, k, group_size)`（`cb:455`）、`_prefer_cublas_int8_fallback` は `(m, n, k, device_index)`（`cb:534-539`）、`_int8_weight_scale_arg` は `(weight_scale, device)`（`cb:528`）、`_gemm_vector_arg` は `(v, device, dtype)`（`cb:519`）。
   - 拡張 `_C` の関数は nanobind の docstring の 1 行目（`名前(引数: 型, …) -> 戻り値`）から引数名を取り出して固定値と比べる。`quantize_int8_rowwise_convrot64` は `(input, output, scales, group_size, stochastic, act_code, seed, act_weight, act_eps, stream_ptr)`、`cutlass_int8_dequant` は `(a, b, xs, ws, bias, d, out_dtype_code, stream_ptr)`（いずれも R1 が CPU だけの import で確認）。**`cublas_gemm_int8` と `dequantize_int8_linear` の引数名は未確認**。実装の担当が CPU だけの import（GPU を使わない）で docstring を読み、固定値にする。docstring に引数名が無い（`arg0` などの）場合は、引数の個数だけを比べる（監督に報告）。
5. デバイス: `torch.cuda.get_device_capability(torch.cuda.current_device())` の major ≥ 8（記録 §5 の値を「この GPU で共有の経路が使える」にするため。各呼び出しの #6 とは別に、ロード時の 1 回）。

- 1〜4 は CUDA に触れない関数 `_shared_quant_missing(cb) -> str | None`（足りない名前を返す）に分け、CPU テストから実物の `cb` で呼べるようにする（§6 の C1）。
- 不合格の理由（`_SHARED_REASON`）は、最初に見つかった不足の名前を含む短い英文（例: `"comfy_kitchen.backends.cuda._C.cutlass_int8_dequant is missing"`・`"signature of quantize_int8_rowwise_convrot64 is (…), expected (…)"`・`"compute capability 7.5 < 8.0"`）。

**`require_cuda_int8` の「中から」呼ぶ理由**: 門の台本 `g2_block_equal.py` と GPU テスト（`_gpu_int8_linear`）は `require_cuda_int8()` だけを呼ぶ。検査を `pipeline_h3.py` の側に置くと、台本を変えないかぎり、g2 は共有しない戻り先（＝現行の経路）を比べることになり、門が素通りしてしまう。中から呼べば、台本を変えずに共有の経路が門にかかる。オーナー裁定 1 の「`require_cuda_int8` の隣」をこう解してよいかは §11 の 3。

---

## 3. 数値等価の根拠

### 3.1 論法

1. 現行の on は、1 ブロックで `int8_linear(h, W[n0:n1], ws[n0:n1], None, out_dtype=bf16, convrot=True, convrot_groupsize=256)` を 21 回呼ぶ。各回の中で、(a) `h` を量子化し（2138〜2149 行の `_C.quantize_int8_rowwise_convrot64`）、(b) その結果と重みの切片で行列積をする（2245〜2255 行の `_C.cutlass_int8_dequant`）。
2. (a) の入力は 21 回とも同じ `h`、引数も同じ（量子化は重みに関わらない。R1「ConvRot の回転は層に依存しない」）。量子化の核は決定的（R2 §1。S＝39,764 と 89,388 で 2 回の結果が `torch.equal`）。**したがって 21 回の (a) の結果は全部同じ値**で、1 回だけ計算して使い回しても (b) に渡る値は変わらない。
3. 共有の経路の (a) は、Python の包み `cb.quantize_int8_rowwise_convrot64(x2d, 256)` を通して同じ `_C` の核を呼ぶ。引数は次の表のとおり `int8_linear` の中の呼び出しと 1 つずつ同じになる（確認済み）。

   | `_C.quantize_int8_rowwise_convrot64` の引数 | `int8_linear` の中（2138〜2149 行） | 包み（1604〜1621 行。`stochastic_rounding=0`・`input_act=None` の既定） |
   |---|---|---|
   | input | `x_2d`（bf16・連続） | `weight_2d`＝同じ `x2d` |
   | output | `torch.empty((m, k), int8)` | `torch.empty((M, K // _input_act_width(None)), int8)`＝`(M, K)`（`_activations.py:39-41` で幅は 1） |
   | scales | `torch.empty((m, 1), f32)` | `torch.empty((M, 1), f32)` |
   | group_size | `convrot_groupsize`＝256 | 256 |
   | stochastic | `False` | `0 is not None and 0 > 0`＝`False` |
   | act_code | `_input_act_code(None)` | `_input_act_code(None)` |
   | seed | `0` | `int(0 or 0)`＝`0` |
   | act_weight | `_act_weight_arg(None, None, dev, x_2d.dtype)`＝空の bf16 | `_act_weight_arg(None, None, dev, weight_2d.dtype)`＝空の bf16（2025〜2030 行） |
   | act_eps | `float(0.0)` | `float(0.0)` |
   | stream_ptr | 現在のストリーム | 現在のストリーム |

4. (b) は §2.1 の表のとおり、`int8_linear` の 2190〜2287 行を、同じ関数・同じ引数の並びで写す。形は現行と同じ（M＝S・N＝1,024・K＝5,376）、重みは同じ連続な切片（`self.weight[n0:n1]`）、スケールは同じ `[1024, 1]` の切片（`_int8_weight_scale_arg` はそのまま返し、`ws_cutlass` も同じテンソル。`numel == n` のため）。行列積の核の選び方（CUTLASS か cuBLAS か）は `int8_linear` と同じ関数で同じ値から決まる。
5. 1〜4 から、各切片の出力は現行の `forward_rows` の出力とビット一致する。LoRA の加算は `xa` と出力だけを使い（`lora.py:302-316`）、`xa` は量子化の前に同じ `h` から計算する（§1.2 で `xa_for` を量子化より前に置く）ので、これも変わらない。

### 3.2 崩れうる点と押さえ方

| 崩れうる点 | 押さえ方 |
|---|---|
| 写し損ね（行列積の側で別の核や別の引数に入る） | GPU テスト G1（§6）で「共有して 21 切片を結合したもの＝`forward`」を CUTLASS と cuBLAS の両方で、実際の行数 39,764 で確かめる。端から端は g3 の指紋・g2 の 50 ブロック・腕 N の mp4 の md5（§7） |
| 包みの既定値の違い（`stochastic_rounding` など） | §3.1 の 3 の表で確認済み。版が変わったときに備え、起動時の検査で包みの引数名の並びを固定（§2.4 の 4）。既定値そのものは検査しない（§11 の 4 で版の一致を足すかを問う） |
| `x2d` の整形の違い（dtype・連続性） | `quantize_input` は `forward_rows` の 140〜144 行をそのまま使う（関数に切り出さず同じ 4 行を書く。`forward_rows` は変えない） |
| `int8_linear` が `weight` に `.contiguous()` をかける・`weight_scale` を f32 に直す | 写す（2053〜2054・2191 行）。入力が条件を満たすので実際は何もしない |
| ポインタの揃え方（CUTLASS の中で核の選び方に効くかもしれない。推測） | `xq` は割り当て器の新しい確保で、現行の `x_qdata` と同じく少なくとも 512 バイト境界に揃う（推測。PyTorch の割り当て器の一般的な性質）。重みの切片の先頭は n0 × 5,376 バイトずれ、現行と同じ |
| 21 回目までに `xq` が書き換わる | `xq` を書き込む処理は無い（`cutlass_int8_dequant` と cuBLAS は入力を読むだけ）。同じストリームなので、量子化の終わりより前に行列積が読むことも無い |
| 量子化の核が非決定的になる版 | R2 の確認は 0.2.37 だけ。版が変われば起動時の検査で共有を止める（§9）ことに頼る |
| sage の 2 ブロック（22・40） | 共有は qkv の値を変えないので、sage に渡る q・k・v は現行の on と同じ。差は前回と同じ値のまま残る見込み（g2 で確かめる） |

---

## 4. VRAM

### 4.1 寿命

| もの | 大きさ（S＝39,764） | 生まれる | 消える |
|---|---|---|---|
| `h`（bf16 `[S, 5376]`） | 1U ≈ 408 MiB | 128 行（変調の後） | **共有のとき: `quantize_input` の直後の `del h`**（§1.2）。戻り先のとき: ループの後の `del proj`（現行の 153 行と同じ位置） |
| `xq`（int8 `[S, 5376]`） | 0.5U ≈ 204 MiB | `quantize_input` | ループの後の `del proj`（`partial` が持つ） |
| `xs`（f32 `[S, 1]`） | 0.16 MB | 同上 | 同上 |
| LoRA の `xa`（f32 `[S, r]` × 本数） | Turbo（rank 384＋128）で約 78 MB | 量子化の前 | ループの後（現行どおり） |

- 量子化の瞬間だけ `h` と `xq` が同時に生きる。そのときの生存は `x`（1）＋`h`（1）＋`xq`（0.5）＋`o`（1.33）＝約 3.8U で、注意の区間の山（約 5U）より低い。新しい山は作らない。
- 現行でも `int8_linear` の各呼び出しの中で同じ大きさの `x_qdata` を一時的に確保していた。共有後は 21 回の確保・解放が 1 回になる。

### 4.2 山の変化（設計正本 §6.5「山の内訳」の表）

| 区間 | 今の on の概算 | 変更後の on の概算 | 理由 |
|---|---|---|---|
| 注意（山） | 約 5U（`x`・`h`・`o`＋群の q・k・v・RoPE・sage の内部） | **約 4.5U**（`x`・`xq`〔0.5〕・`o`＋同じ） | `h`（1U）が `xq`（0.5U）に代わる |
| その他の行 | 変わらない | 変わらない | FFN（約 5.3U）を含め、`h` は注意の区間の外では生きていない |

- **割り当てのピーク（実測 4,411 MiB）は変わらない可能性が高い**（推測）。上の表の概算では FFN の区間（約 5.3U）が注意（約 5U）より高く、注意を 0.5U 下げても全体の山は FFN のまま。ただし表は概算で、どちらが実際の山かは測っていない（設計正本 §6.5 の「実測との照合」は全体で約 5.4U と書くだけ）。注意が実際の山なら、最大で約 204 MiB 下がる。
- 門の条件は「割り当て／予約のピークが前を上回らないこと」（§7）。下がったかどうかは記録するだけにする。
- `inactive_split`（返されない空きセグメント）のピークは、確保の回数が変わるので動きうる（推測。向きは分からない）。記録する。

---

## 5. 記録とログ

### 5.1 `done` の `h3.phases.denoise.vram16` に 1 項目足す案（推奨）

- 名前 `qkv_quant_shared`、型 `bool`。値は起動時の検査の合否（§2.4。デバイスの確認を含む）。
- 足す理由: 共有の経路に乗ったかどうかは出力からは分からない（数値は同じ）。速度の差が出ないジョブを後で調べるとき、記録の他に手がかりが無い。既存の値（`quant_backends` は能力の一覧だけ）からは導けない。
- 足さない場合の代わり: ロードのログ 1 行だけ。ログはジョブの記録（`metadata.json`）に残らない。
- 意味の限界: ロード時の判定で、ブロックごとの判定（§2.2 の #2〜#5・#7）は数えない。製品の H3 でブロックごとに偽になる場面は無いと見ている（§2.3。推測）。数えるには計数の仕組みが要り、機構が増えるので採らない。
- アプリ側にこの欄を読むコードは無い（`services/`・`api/`・`mcp_server/`・`webui/src` を検索して 0 件）。仕様書 §6.6 の `h3` 欄の説明に名前を足す（§8）。
- 鍵の一覧を固定したテストは無い（`tests/test_h3_engine.py:1759, 1940` は個々の値を見るだけ）。

### 5.2 ログの文言（ロード時に 1 行）

- 合格: `int8: qkv quantization shared across head groups (comfy-kitchen internals found)`
- 不合格: `int8: qkv quantization NOT shared (<理由>); VRAM 16GB mode quantizes the qkv input once per head-group slice, as before`
- ワーカーの既存のログが英語なので英語にする（`pipeline_h3.py:508` など）。

---

## 6. テスト

### 6.1 CPU テスト（`tests/test_h3_engine.py`。`scripts/run_h3_pytest.py` で回す）

| # | 名前（案） | 中身 | 置く場所 |
|---|---|---|---|
| C1 | `test_shared_quant_symbols_of_the_pinned_comfy_kitchen` | 実物の `comfy_kitchen.backends.cuda` を import し、`_shared_quant_missing(cb)` が None を返す（0.2.37 の記号と引数の並びの固定）。CUDA の関数は呼ばない（拡張の読み込みは CPU だけで済む。R1 で確認） | 1590 行付近（偽の `_ck` の隣） |
| C2 | `test_shared_quant_missing_names_the_symbol` | 偽の `cb`（`types.SimpleNamespace`）から 1 つずつ記号を抜く・`_C` の docstring の引数名を 1 つ変える → 返す理由にその名前が入る。`_probe_shared_quant` は例外を投げず `_SHARED_CB` が None・`shared_quant_status()` が `(False, 理由)` | 同上 |
| C3 | `test_shares_input_conditions` | 偽の `cb`（`_CONVROT_FUSED_MAX_K=16384`・`_convrot_fused_shared_memory_fits`・`_cuda_device_supports_cutlass_int8_dequant` を真偽で差し替えられるもの）を `_SHARED_CB` に入れ、§2.2 の #1〜#7 を 1 つずつ崩すと偽、全部満たすと真（`convrot=False`・groupsize 64・K＝128・K＝16,640・m＝2・CUTLASS 非対応・重みが別のデバイス） | 同上 |
| C4 | `test_vram16_block_forward_shared_matches_off`（既存の `test_vram16_block_forward_matches_off` を `shared` の真偽で parametrize してもよい） | `shares_input` を真に、`quantize_input` を「`(x2d.to(float64), ones[M,1] f64)` を返す偽物」に、`quant_linear._int8_gemm_prequantized` を「`(xq * xs) @ (w * ws)ᵀ` を float64 で計算して `out_dtype` に 1 回丸める偽物」（既存の `_int8_linear_as_float64` と同じ算術）に差し替え、on と off がビット一致。`forward_rows` が 1 度も呼ばれず、`quantize_input` が 1 ブロック 1 回、`forward_rows_q` が 1 ブロック 3 × 群の数だけ呼ばれる。**`quantize_input` に渡った `h` の `weakref` が、最初の `forward_rows_q` の呼び出しの時点で None**（`h` を捨てたことの構造の確認） | 1672 行の後 |
| C5 | `test_lora_vram16_shared_matches_off` | C4 の差し替えのうえで、既存の `test_lora_vram16_matches_off`（2231 行）と同じ 3 通りの parametrize。on と off がビット一致、`xa_for` は 1 ブロック 1 回、`apply` は `x=None`・`xa` ありで 3 × 群 × ブロック回 | 2276 行の後 |
| C6 | `test_vram16_falls_back_when_not_shared` | `_SHARED_CB` が None（既定）のとき、`quantize_input`・`forward_rows_q` が呼ばれず `forward_rows` が 3 × 群 × ブロック回（既存のテストは全部この分岐を通るので、回数の確認だけ） | 1716 行の後 |

- 偽の量子化を float64 にする理由: 本物の int8 の丸めを真似ると、off（`_int8_linear_as_float64`。量子化しない）と一致しない。CPU テストが確かめるのは配線（切り方・LoRA の行・`h` の寿命）で、数値の等価は GPU テストと門で確かめる。
- 既存のテストは変えない。`_fake_int8` は `_ck` だけを差し替えるので、`_SHARED_CB` が None のまま（＝戻り先）で今と同じく通る。
- モジュールの状態（`_SHARED_CB`）は `monkeypatch.setattr` で差し替え、テストの後に戻す。

### 6.2 GPU テスト（`H3_GPU_TESTS=1` のときだけ）

| # | 名前（案） | 中身 |
|---|---|---|
| G1 | `test_gpu_shared_quant_join_equals_forward`（parametrize: `bias` 真偽 × `cutlass` 真偽 × `m` 4096・39,764） | `_gpu_int8_linear(3 * 7168, 5376, convrot=True, bias=…)`（1858 行）。`shared_quant_status()[0]` が真、`lin.shares_input(x)` が真。`xq, xs = lin.quantize_input(x)` の後、21 切片の `forward_rows_q` の結合が `lin(x)` と `torch.equal`、各切片が `forward_rows` とも `torch.equal`。`cutlass=False` は `monkeypatch.setattr(cb, "_DISABLE_CUTLASS_INT8", True)` で、`int8_linear` と補助関数の両方を cuBLAS に回す（両方とも呼ぶたびにモジュール変数を読む） |
| G2 | `test_gpu_quantize_input_is_the_kernel_int8_linear_runs` | `quantize_input(x)` を 2 回 → 2 回とも `torch.equal`（R2 の決定性を製品の形でテストに固定）。行列積を通さずに確かめられる範囲はここまで |

- G1 の m＝39,764 の入力は bf16 で約 427 MB、重みは約 115 MB。開発機の 16 GB に収まる。

---

## 7. 門（弾の分け方）

前回（`VERIFICATION_LOG.md` §169.3・§176）に倣う。GPU を使う門は監督の了承を取ってから、同時に 1 体だけで回す。

### 7.0 実装の前

| 門 | 内容 | 合格 |
|---|---|---|
| P-R | 本計画の敵対的レビュー（Opus。記憶の「承認ゲートの教訓」） | 指摘を採否の表にし、Critical が残らない |

### 7.1 弾 A（エンジン＋テスト）と機械の門

| 門 | 内容 | 合格の条件 |
|---|---|---|
| A-1 | CPU テスト全件（`.venv-engine-h3\Scripts\python.exe scripts\run_h3_pytest.py`） | 失敗 0。件数は §176.3 の 215 件合格・14 件スキップに C1〜C6 の分を足した数（parametrize で増える）。スキップの増分は G1・G2 の分だけ |
| A-2 | アプリの仮想環境の pytest 全件 | 失敗 0（§176.3 では 3,553 件合格・67 件スキップ。件数は変わらない見込み） |
| A-3 | 実装差分のコードレビュー（Opus） | Critical 0。Major は全件対処 |
| A-4 | GPU テスト（`H3_GPU_TESTS=1` で `tests/test_h3_engine.py` 全件） | 失敗 0。G1・G2 を含む。ログに `int8: qkv quantization shared …` が出る |
| A-5 | g3 の指紋（`outputs/h3-vram16-gates/g3.sh` を写した台本。sdpa・640×384・124 フレーム・20 ステップ・seed 42。off と on） | **off も on も sha256 が `0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9` に全桁一致**。on の report の `h3.phases.denoise.vram16.qkv_quant_shared` が `true`（§5 を採った場合） |
| A-6 | 製品の形の 50 ブロック比較（`g2_block_equal.py` を変えずに使う。**`--report` を必ず新しいパスにする**。既定の出力先は前回の `g2_block_equal.json` で、上書きすると比べる相手が消える） | **sdpa 50/50 ビット一致。sage は 48/50 で、違うのはブロック 22 と 40 だけ、その差の値が前回の `g2_block_equal.json` と全桁同じ**: ブロック 22 は `max_abs` 256.0・`elements_differ` 115,388,441・`rel_out` 0.00659228097488087、ブロック 40 は 1792.0・6,974,901・0.0030670819750614436。あわせて、台本を回す前に `shared_quant_status()` が `(True, None)` であることを、台本を包む小さなスクリプト（`runpy` で g2 を呼ぶ前に表示と assert をする。git 管理外）で確かめる |
| A-7 | LoRA の門の取り直し（`Docs/Outputs-archive/h3-lora-2026-10-24/gates/gates.sh` の写し。出力先だけ変える） | on の側の **g2・g4b・g5b・g5d・g5f・g5g の sha256 が前回と全桁一致**（前回の値の正本は同じフォルダの `gates/*.json`。先頭 16 桁は `0da6111cdca853ce`・`0caa05dcde6e48a8`・`c05bca7bcbbc803a`・`a9df669270612d50`・`f48a4c2a064fc861`・`c05bca7bcbbc803a`）。off の側は経路を変えていないので回さない（回すかは §11 の 6） |
| A-8 | 前後の速度比較（`outputs/h3-speedup-2026-10-24/arm_n3.sh` の写し。I2V 896×1152×124・sage・常駐 on・決定性 on・先読み on・VRAM 16GB mode on・3 ステップ。計測台 `measure_h3_speedup.py` の写しに、`H3Int8Linear.shares_input` を常に偽にする `--no-qkv-share` を足す〔計測台の中だけの差し替え。製品にはつまみを足さない〕。「前」＝`--no-qkv-share`、「後」＝無し。前→後→前→後の 4 回） | (1) 各回の各ステップの dmon の受信の中央値を `step_rx.py` で出し、**外部の負荷のあるステップ（受信の中央値が 300 MB/s 超）を比較から外す**。外したせいで前後それぞれ 2 ステップ未満になったら、その回を取り直す。(2) 負荷の無い 2・3 ステップ目の平均で、**後が前より 1.0 秒以上速い**（上限の見込みは 1.447 秒。下回る場合は監督に報告）。(3) **生成物の mp4 の md5 が 4 回とも同じで、R2 の `eb2a53ae336e4eb99dc6ef8ddcad6bb7` に一致**（＝旧 on と新 on が製品の形・sage・3 ステップで端から端まで一致。台本を変えずに取れる追加の確認） |
| A-9 | VRAM（A-8 の 4 回の記録から） | 割り当て／予約のピーク（ノイズ除去）が、後で前を上回らない（前の実測は 4,411／7,262 MiB）。退避・OOM・再試行 0。`inactive_split` のピークは記録だけ |

- A-8 の「前」を同じ時間帯に取る理由: R2 §2.4 のとおり、別のプロセスの GPU の負荷が断続的にあり、ステップを約 1.5 秒遅くする。これは期待する差（約 1.45 秒）と同じ大きさなので、日をまたいだ比較（R2 の N3 の 21.641／21.684 秒）だけでは判定できない。N3 の値は参考として並べる。
- 長尺（T2V 1280×768×311・S＝89,388）の速度は門にしない（見込みは 1 ステップ約 3.1 秒の短縮。R2 §1）。回すかは §11 の 9。

### 7.2 オーナーの目視

エンジンだけの変更なので、操作パネル（aux2）の配置は要らない。バックエンドの再起動（ワーカーがエンジンのコードを読み直す）だけで実機に効く。

| 門 | 内容 | 合格 |
|---|---|---|
| O-1 | 操作パネルから H3 の I2V を 1 本（768p 以上・20 ステップ・VRAM 16GB mode on〔既定〕） | 完走。`metadata.json` の `h3.phases.denoise.vram16.qkv_quant_shared` が `true`。1 ステップの秒数が前回の同じ条件より短い（記録だけ） |
| O-2 | Turbo LoRA で 1 本（768p 以上・4 ステップ） | 完走。絵に異常が無い |

- 数値は全桁同じになる設計なので、目視は「製品の経路で動くこと」の確認で、品質の比較ではない（記憶の「H3 の目視は 768p 以上」に従う）。

### 7.3 弾 C（文書）

§8 の一覧。コミットは監督が dev で行う（エンジン＋テスト／文書・保管の 2 つ）。

---

## 8. 文書の更新一覧

| 文書 | 箇所 | 中身 |
|---|---|---|
| `Docs/MINIMAX_H3_DESIGN.md` | §5（169〜177 行）の末尾 | **例外の加筆**: 「VRAM 16GB mode の qkv だけは、関数 API の `int8_linear` に任せず、comfy-kitchen 0.2.37 の非公開の内部（量子化の `backends.cuda.quantize_int8_rowwise_convrot64` と、`int8_linear` の量子化より後の部分を写した補助関数）で、量子化を 1 ブロック 1 回にする（台帳 §1-105 ①・オーナー裁定 2026-10-25）。起動時に記号と引数の並びを調べ、無ければ現行の経路に戻す（ロードは拒否しない）。この経路は登録の制約を通らないので、重みが GPU に無い呼び出しは共有せずに `int8_linear` に回し、fail loud を保つ」。数値の正本は VERIFICATION_LOG §177 を参照させる |
| 同上 | §6.5「何をするか」の表（199〜208 行） | 行を 1 つ足す（9. qkv の量子化の共有）。表の前の「次の 8 つ」を直す。#2「norm 後の入力は最後のヘッド群の後に捨てる」を「量子化の直後に捨てる（共有に乗れないときは最後のヘッド群の後）」に |
| 同上 | §6.5「置き方」（215 行・217 行） | **例外の加筆**: `lowvram.py` の「新しい計算を書かない」に「例外: qkv の入力の量子化を 1 回にする分岐（計算そのものは comfy-kitchen の同じ核）」。`forward_rows` の項に `shares_input`・`quantize_input`・`forward_rows_q` を足す |
| 同上 | §6.5「山の内訳」の表（258〜267 行）の「注意（山）」の行 | on の概算を「約 4.5U」に、生きているものを「`x`・量子化済みの `h`（int8・0.5U）・`o`…」に。表の後の「on の山は約 5U」を「約 5.3U（FFN）」とするかは、A-9 の実測で決める |
| 同上 | §6.5「範囲と既知の限界」の末尾（278 行の「21 回」の行） | 「量子化を 1 回にした（台帳 §1-105 ①。短縮の実測は VERIFICATION_LOG §177）。Turing と、comfy-kitchen の内部が見つからないときは 21 回のまま」に書き換え |
| 同上 | §6.6「接続」（300 行） | `forward_rows_q` も `apply` を呼ぶこと（`x` は None・`xa` は必須） |
| 同上 | §8 のファイル構成の表（378 行の `quant_linear.py` の行） | 「VRAM 16GB mode の qkv の量子化の共有（§5 の例外）」を足す |
| 同上 | §11 の弾の表 | 行を足すか、「以後」の行のままにするか（前例は VRAM 16GB mode とキーフレームで行を足している）。足すなら 1 行で、記録は VERIFICATION_LOG §177 を参照 |
| `Docs/MINIMAX_H3_RESEARCH_NOTES.md` | §14.1 の ① の箇条の後 | 結果を 1〜2 行: 「① は実装した（1 ステップ約 1.4 秒の短縮。数値は全桁同じ）。② と ③ は実測で見送り（GPU の空きが 20 ステップで約 1.1 秒・ほぼ 0）。④ は未着手。数値の正本は VERIFICATION_LOG §177」。数値は書き写さず参照に（記憶の「文書の正本 1 箇所原則」） |
| `Docs/VERIFICATION_LOG.md` | **§177（新しい節）**（最新は §176 で、§176.9 に「次は §177」とある。確認済み） | 結論・範囲・R1／R2 の要点（②③の見送りの根拠の実測を含む）・門 A-1〜A-9 と O-1・O-2 の結果・レビューの採否・コミット・教訓・次の番号 |
| `Docs/PENDING_TASKS.md` §1-105（33 行） | 下の 2 案 | |
| `Docs/PENDING_TASKS_CLOSED.md` | 末尾に 3-254 | 次の番号は **3-254**（最新は 3-251。3-252・3-253 は台帳 §3 の生きた番号なので飛ばす。§176.9・台帳 134〜141 行で確認済み） |
| `Videomni_Backend_Specification.md` | 改訂履歴（155 行が最新の v0.5.99）と 39 行の版の欄、§6.6 の `h3` 欄の説明（1323 行の「VRAM 16GB mode のジョブでは `phases.denoise.vram16` に…」） | §5 の項目を足すなら、1323 行の列挙に「qkv の量子化を共有したか（`qkv_quant_shared`）」を足し、履歴に 1 行。加算の記録の追加で版を上げるか（v0.5.100 か）、「文言の現行化（版は上げない）」にするかは §11 の 2。項目を足さないなら仕様書は変えない |
| コード内コメント | `lowvram.py` の冒頭（6〜13 行・16〜19 行）・`quant_linear.py` の冒頭（1〜30 行）と `forward_rows` の docstring（123〜139 行。「activation is quantized per row」の段落に共有の経路への参照）・`lora.py` の冒頭（20〜23 行）と `apply` の docstring | §1 のとおり。**`dit.py` の docstring（63〜69 行の VRAM 16GB mode の段落）は量子化の回数に触れていないので変えない**（確認済み） |
| 保管 | `Docs/Outputs-archive/h3-speedup-2026-10-24/` | R1・R2・本計画・レビュー・門の台本と JSON の写し（前例どおり。mp4 と trace は大きいので除く） |

**台帳 §1-105 の扱い（監督が選ぶ）**

- **案 A（§1-105 全体をクローズ）**: §1-105 を CLOSED 3-254 へ移し、台帳から削除する。①は完了、②③は実測で見送り（理由は VERIFICATION_LOG §177）。④（QKV の不要なコピー・レイアウト変換の削減。条件付き）は §3 に新しい項目として起票し直す（番号は §3 の次の空き番号。3-254 は CLOSED で使うので 3-255 の見込み。要確認）。台帳には「出典: §1-105 の ④」とだけ書く（台帳に歴史を書かない）。
- **案 B（④ を §1-105 に残す）**: §1-105 の本文を ④ だけに書き直す（①②③の経緯は台帳に書かず、VERIFICATION_LOG §177 を参照）。クローズは ④ の結論が出たとき。CLOSED の番号は今は使わない。
- 本計画の見立て: ④ は「既存設計が strided な Q／K を不採用としている」ので数値と VRAM の再検証が前提で、第 1 弾（数値等価）の残りというより別の調査に近い。案 A のほうが台帳の項目の意味（第 1 弾）と合う（判断は監督）。

---

## 9. 危険と対処

| 危険 | 対処 |
|---|---|
| comfy-kitchen の版を上げたとき、記号・引数・中の経路の分岐が変わる | 版は `engine_h3/venv-engine-h3.freeze.txt:50` で 0.2.37 に固定。上げるときは起動時の検査が名前と並びの変化を拾って共有を止める（ロードは続く）。**ただし、名前と並びが同じまま `int8_linear` の中の経路の分岐が変わった場合は検査で拾えない**（例: CUTLASS の前に新しい枝が入る）。版を上げるときは、補助関数のコメントの元の行番号と見比べ、GPU テスト G1 と g3 を回すことを、設計正本 §5 の例外の段落に書いておく。版の一致そのものを検査に加えるかは §11 の 4 |
| 記号が無い（古い・新しい comfy-kitchen、拡張の読み込みの失敗） | 検査で `_SHARED_CB` が None → 現行の経路。ログに理由が 1 行、`qkv_quant_shared` が `false` |
| block swap が `.data` を差し替える | `forward_rows_q` も切片を持たず、毎回 `self.weight[n0:n1]` から切る（`forward_rows` の docstring 128〜131 行と同じ理由）。量子化済みの入力 `xq` は活性化で、重みではないので差し替えの影響を受けない。共有の寿命は 1 ブロックの中に閉じる（ブロックをまたいで持たない） |
| LoRA との組み合わせ | `xa` は量子化の前に同じ `h` から作る（§1.2）。`apply` は `xa` があると `x` を使っていない（`lora.py:302-306`）。C5 と A-7 で確かめる |
| Turbo LoRA（rank 384＋128） | 経路は他の LoRA と同じ。`xa` が大きい（約 78 MB）が寿命は今と同じ。A-7 の g4b（Turbo・640×384・on）で確かめる。短縮は S に比例するので、640×384（S≈9,300）では 1 ステップ約 0.34 秒にとどまる（推測。R2 の 1.447 秒 × 9,335／39,764） |
| Ampere（RTX 30 系）で未確認 | 判定 #6 は Ampere（8.x）で真になり、共有の経路に乗る。`int8_linear` も Ampere で同じ CUTLASS の枝に入るので、等価の論法は GPU によらない。ただし実測は開発機（Ada）だけ。サブマシン（RTX 3080 Laptop）で G1 と g3 を回すかは §11 の 7 |
| Turing（RTX 20 系） | 判定 #6 で共有しない。現行のまま（変わらない） |
| 環境変数 `COMFY_KITCHEN_DISABLE_CUTLASS=1` | 補助関数も `int8_linear` と同じく cuBLAS に回る（モジュール変数を呼ぶたびに読む）。G1 の `cutlass=False` で確かめる |
| 量子化の核が非決定的な版・GPU | R2 で確認したのは 0.2.37・Ada だけ。非決定的なら現行の on も off と一致しないはずで、前回の A-G2 が成り立たない。版を上げたときの G2 で拾う |
| 速度の比較が外部の GPU の負荷で汚れる | A-8 の手順（`step_rx.py` で負荷のあるステップを外す・前後を交互に回す） |
| `g2_block_equal.py` の既定の出力先で前回の基準を上書きする | A-6 で `--report` を必ず指定（計画に明記） |

---

## 10. 担当の分け方の案

- **コーディング 1 体（Opus）で足りる**。本体の変更は約 175 行で、4 ファイルが 1 本の流れ（検査 → 判定 → 量子化 → 行列積 → LoRA の行数 → 記録）でつながっている。分けると `quant_linear.py` と `lowvram.py` と `lora.py` の間の約束（`forward_rows_q` の引数・`apply` の `x=None`）を 2 体で合わせる手間のほうが大きい。
- 1 ファイル 1 所有者の原則は 1 体なら自動的に満たす。テスト（`tests/test_h3_engine.py`）も同じ担当に持たせる（偽物の差し替え先の名前が本体の名前と一体のため）。
- 別にするのは次の 3 つ: (1) 計画の敵対的レビュー（Opus・読むだけ）、(2) 実装差分のレビュー（Opus・読むだけ）、(3) GPU の門の実行（監督の了承の後、1 体。台本は git 管理外の新しいフォルダ `outputs/h3-qkv-share-gates/` に置く）。文書（弾 C）は門の後に 1 体。
- 実装の担当が最初にすること: CPU だけの import で `_C.cublas_gemm_int8.__doc__` と `_C.dequantize_int8_linear.__doc__` を読み、§2.4 の固定値を埋める（GPU を使わない）。

---

## 11. 未決・監督への質問

1. **行数の渡し方**: 本計画は `lora.apply` に引数を足さず、`xa[0].shape[0]` から行数を取る（§1.3）。オーナー裁定 3 の「行数を別に渡すように直す」の字義（引数 `rows` を足す）と違う。「既存の値から導けるものに新しいフィールドを足さない」に沿う形として、これでよいか。
2. **記録の項目**: `done` の `h3.phases.denoise.vram16` に `qkv_quant_shared: bool`（ロード時の検査の合否）を足すか（推奨は足す。§5）。足すなら仕様書は v0.5.100 として履歴に 1 行足すか、「文言の現行化（版は上げない）」で済ませるか。
3. **検査の呼び出し位置**: `require_cuda_int8` の末尾から `_probe_shared_quant` を呼ぶ（§2.4。門の台本と GPU テストが変更なしで共有の経路にかかるため）。裁定 1 の「隣」をこう解してよいか。
4. **版の一致を検査に加えるか**: 記号と引数の並びが同じでも、`int8_linear` の中の分岐が変われば写しは古くなる（§9）。`importlib.metadata.version("comfy-kitchen") == "0.2.37"` を検査の条件に足すと、版を上げたとき自動で共有が止まり、再確認するまで現行の経路で動く。足すと「版を上げたら共有を有効にするのに 1 行の変更が要る」。足すかどうか。
5. **速度比較の「前」**: 計測台（git 管理外）の中で `shares_input` を偽に差し替えて、同じ時間帯に前後を交互に取る（§7 の A-8）でよいか。R2 の N3 だけを「前」にする案もあるが、外部の負荷の大きさが期待する差と同じなので推奨しない。
6. **LoRA の門の範囲**: on の側 6 本（g2・g4b・g5b・g5d・g5f・g5g）だけで足りるか。off の側（経路は不変）と、門 3（アプリ経由の常駐の清潔さ。共有はブロックの中に閉じ、ジョブをまたぐ状態を持たない）も回すか。
7. **Ampere（サブマシン）**: G1 と g3 をサブマシンで回すか。回さない場合、設計正本 §6.5 の「RTX 30 系は未確認」の記述に共有の経路も含まれることを書く。
8. **台帳**: 案 A（§1-105 全体をクローズ・④ を §3 に起票し直す）か案 B（④ を §1-105 に残す）か（§8）。
9. **長尺の確認**: T2V 1280×768×311（S＝89,388）で前後を 1 回ずつ測るか（見込みは 1 ステップ約 3.1 秒の短縮。門にはしない）。
10. **山の内訳の表の直し方**: A-9 でピークが変わらなかった場合、設計正本 §6.5 の「on の山は約 5U」を「山は FFN の区間（約 5.3U）」と直すか（今回の実測で裏付けられるのは「注意を下げても全体の山は動かなかった」までで、FFN が山だとは直接は測れない）。
