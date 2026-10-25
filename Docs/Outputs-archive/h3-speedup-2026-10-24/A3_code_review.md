# A-3 差分のコードレビュー（台帳 §1-105 ①「QKV の活性化量子化の共有」）

対象: 作業ツリーの未コミットの変更 5 ファイル（ブランチ dev）。照合先: 承認済みの計画 `snazzy-marinating-turtle.md`（正本）、`P1_plan_qkv_share.md`、`P1_review.md`、comfy-kitchen 0.2.37 の `backends/cuda/__init__.py`。

## 総評

- **Critical 0 件・Major 0 件・minor 5 件**（ほかに記録だけの事項 2 件）。
- **GPU の門（A-4 以降）に進んでよい**と判断します。minor はどれも数値と経路の選び方に影響しません。直すなら、どれも 1〜数行で済みます。
- 最重要の観点（写しの正確さ）: `_int8_gemm_prequantized` を `int8_linear` の 2190〜2289 行と 1 行ずつ突き合わせました。`shares_input` が真の条件のもとでは、同じ核に同じ引数で入ります。別の核に入る余地や、別の引数で呼ぶ余地は見つかりませんでした（下の「確認できた良い点」1〜3）。

## 指摘（重い順）

### 1. minor: `shares_input` は bias の置き場所を見ていない（docstring の書き方が実際より強い）

- **該当箇所**: `engine_h3/quant_linear.py` 250〜276 行（`shares_input`。最後の行で `self.weight` と `self.weight_scale` のデバイスだけを比べている）。docstring の 257〜258 行には「a call it would refuse goes to `forward_rows` and fails there」とある。
- **事実**: 登録の制約 `constraints["int8_linear"]` は `default_devices=cuda_devices` で、bias も含めたすべてのテンソル引数に効く（`backends/cuda/__init__.py` 4464〜4486 行）。共有の経路では、bias が CPU にあっても `_gemm_vector_arg` が黙って GPU へ写す（同 519〜525 行）。止まらずに遅くなるだけ。
- **影響**: 製品では起きない。H3 の qkv は `bias=False`（`engine_h3/dit.py:229`）。計画の条件 ⑦ も重みとスケールだけを挙げており、計画どおりではある。G1 は bias ありで回るが、その bias は GPU 上にある。
- **最小の対処案**: 条件は足さない。docstring を「the weight and its scale must already be on `x`'s device (the qkv has no bias)」のように、実際に守っている範囲の書き方に直す。
- **代替案**: 最後の行に `and (self.bias is None or self.bias.device == x.device)` を足す（1 条件増える。C3 にも 1 行要る）。

### 2. minor: `lowvram.py` の冒頭 docstring の「What is cut」が `forward_rows` だけを挙げている

- **該当箇所**: `engine_h3/lowvram.py` 21〜22 行（「projected per head group straight from the rows of the stacked `[q; k; v]` weight (`H3Int8Linear.forward_rows`)」）。
- **事実**: 共有が有効なときは `forward_rows_q` が使われる。同じ docstring の 9〜14 行の例外の段落とは食い違わないが、ここだけ書き換えが漏れている。
- **影響**: 読む人が少し迷うだけ。
- **最小の対処案**: 括弧の中を `(H3Int8Linear.forward_rows, or forward_rows_q on the input quantized once)` に直す。

### 3. minor: C5 は `h` が解放されたことを確かめていない

- **該当箇所**: `tests/test_h3_engine.py` の `test_lora_vram16_shared_matches_off`（`calls, _h_gone = _count_qkv_calls(...)`。`_h_gone` を使っていない）。
- **事実**: `h` を量子化の後まで握り続ける危険があるのは、`xa_for(qkv, h)` が走る LoRA ありの経路のほう。C4（LoRA なし）は `h_gone` を確かめているが、C5 は確かめていない。現在のコードでは `xa_for` が `h` を持ち続けない（`engine_h3/lora.py:261-282`。戻り値は新しく確保した `xa` だけ）ので、いまは問題がない。
- **影響**: 将来 `xa_for` が `h` の view を返すように変わると、VRAM の山が黙って戻る。そのときに CPU テストで気づけない。
- **最小の対処案**: C5 の末尾に `assert all(_h_gone)` を 1 行足す（名前も `h_gone` に）。

### 4. minor: `_SHARED_CB_NAMES` の 14 個のうち 9 個は `_SHARED_CB_SIGNATURES` のキーと重なる

- **該当箇所**: `engine_h3/quant_linear.py` 70〜97 行。
- **事実**: 関数 9 個は、両方の表に名前が載っている。存在の検査（`hasattr`）と引数の並びの検査は別のループで回る（380〜381 行と 390〜393 行）。
- **影響**: 版を上げるときに直す箇所が 2 つになる。数値と挙動への影響はない。追加の行数（約 +281 行）の大半は 4 つの表と、元の行番号つきの写しが占める。精度を落とさずに削れるのは、この重なりの約 9 行くらい。
- **最小の対処案**: 現状のままにする（読みやすく、固定値が目で照合できる）。
- **代替案**: `_SHARED_CB_NAMES` を関数以外の 5 個（`_CONVROT_FUSED_MAX_K`・`_DISABLE_CUTLASS_INT8`・`DTYPE_TO_CODE`・`_EXT_AVAILABLE`・`_C`）だけにする。関数の存在は、引数の並びのループの中で `getattr(cb, name, None) is None` として調べる（分岐の数は同じ）。

### 5. minor（意見）: `_probe_shared_quant` の `except Exception` の範囲

- **該当箇所**: `engine_h3/quant_linear.py` 396〜413 行。
- **事実**: try の中身は import、`_shared_quant_missing`、`get_device_capability` の 3 つ。`_shared_quant_missing` 自体の書き間違い（例: `AttributeError`）も「共有しない」に変わる。ただし理由には `repr(exc)` が入り、`pipeline_h3.py:520-525` でログに 1 行出る。
- **影響の評価**: 許容できると考えます。理由は 3 つあります。(1) C1 が `_shared_quant_missing(cb)` を try の外で直接呼ぶので、検査関数の誤りは CPU テストで例外として表に出る。(2) A-5 の包みの台本が `shared_quant_status() == (True, None)` を assert する。(3) 記録の `qkv_quant_shared` が false になり、O-1 の `metadata.json` でも見える。黙って失われるのは「速さ」だけで、数値は現行の経路のまま。
- **最小の対処案**: 変えない。
- **代替案**: try を `import` と `get_device_capability` だけに狭め、`_shared_quant_missing` は外に出す（書き間違いはロードを止める）。これは「ロードは拒否しない」という裁定 1 と緊張関係にあるので、勧めません。

### 記録だけ（対処は不要）

- **R-1**: 設計正本 `Docs/MINIMAX_H3_DESIGN.md` 278 行の「qkv の量子化が 1 ブロックで 3 × 7 群の 21 回走る」は古くなる。計画の「文書の更新（弾 C）」に入っているので、ここでは触れない。`engine_h3/` 配下を grep したところ、「21 回」の類の古い記述は残っていない。
- **R-2**: GPU テスト（`_gpu_int8_linear` が `require_cuda_int8()` を呼ぶ。`tests/test_h3_engine.py:2031`）の後は `_SHARED_CB` が本物の `cb` のまま残る。triton と eager を切ったままにする既存の作法と同じ。C6 は `None` を明示しているので影響を受けない。ほかの既存の vram16 の CPU テストは小さなモデルが `convrot=False` なので、本物の `cb` が残っていても条件 ② で偽になる。A-4 の全件 162 件合格もこれと矛盾しない。

## 観点ごとの確認

1. **写しの正確さ**: 下の「確認できた良い点」1〜3 を参照。問題なし。
2. **`quantize_input`**: 公開されていない Python の包み `quantize_int8_rowwise_convrot64(x2d, 256)` は、既定値（`stochastic_rounding=0`・`input_act=None`・`input_act_eps=0.0`）から `_C.quantize_int8_rowwise_convrot64(x, q, s, 256, False, _input_act_code(None), 0, _empty_cuda_tensor(dev, bf16), 0.0, stream)` を呼ぶ（1588〜1625 行）。`int8_linear` の 2134〜2149 行（`False`・`_input_act_code(None)`・`0`・`_act_weight_arg(None, None, dev, x_2d.dtype)` は同じ空テンソル・`float(0.0)`）と一致する。出力の形 `(m, k)` int8 と `(m, 1)` f32 も同じ。整形（2 次元化・bf16 への変換・`contiguous()`）は `forward_rows` の 222〜225 行と同じで、`int8_linear` の 2052 行は 2 次元で連続のテンソルをそのまま使う。
3. **`shares_input` の 7 条件**: ①〜⑦ は計画どおり。② と ③ と ④ は `_fused_convrot_ok`（2061〜2067 行）と同じ。`_convrot_fused_shared_memory_fits` は m と k とデバイスだけで決まり、dtype によらない（415〜459 行）ので、bf16 に変える前の `x2d` を渡しても同じ答えになる。⑤ m > 2 で、m＝1 の核（2091〜2126 行）と m ≤ 2 の GEMV（2169〜2188 行）を外している。⑥ `major >= 8`（321〜332 行）で、Turing の判定 `== (7, 5)`（289〜295 行）を外している。そのため 2195〜2209 行と 2258〜2263 行には届かない。残差は渡していない。常に真になる条件は書かれていない。
4. **`lowvram.py` の `h` の寿命**: `xa_for` が先に走る（144 行）→ `shares_input` → `quantize_input(h)` の戻り値は新しく確保した `xq` と `xs` だけ → `del h`（150 行）。それより後に `h` を参照する行はない（ループの変数は `h0` と `h1`）。`h` は `_rows_into` のラムダに捕まっていない（ラムダが持つのは `n1`）。戻り先の分岐では partial が `h` を持ち、`del proj`（166 行）で解放する。旧コードの `del h, xa` と同じ位置。例外のときもフレームが `proj` を持つだけで、旧コードと同じ。ループの間は `h`（bf16）の代わりに `xq`（int8）が残るので、この区間の常駐は半分になる。
5. **`lora.py` の `apply`**: `xa` があるときの行数は、以前の `x2d.shape[0]` と等しい（`xa_for` が同じ `x` から `[M, r]` を作る。lora.py 272〜278 行）。形の検査、`row_chunks(m, …)`、行の塊の切り方は変わらない。`xa` がないとき（off の hook と `forward`）は、`x2d` の作り方も `m` も以前と同じ。`x2d` は `xa` がないときの分岐（`xr = x2d[r0:r1]`）でしか使われない。
6. **`pipeline_h3.py`**: `shared_quant_status()` は `require_cuda_int8()` の直後に呼ばれる（519〜525 行）。ログの文言は計画の 2 通りと完全に一致する。`record` の `"qkv_quant_shared": shared_quant_status()[0]` も計画どおり（871 行）。
7. **検査の固定値**: CPU の import で実物と照合した（`.venv-engine-h3\Scripts\python.exe`。CUDA の拡張は読み込まれたが、核の実行も確保もしていない）。結果: `importlib.metadata.version("comfy-kitchen") == "0.2.37"`、`_shared_quant_missing(cb) is None`、`_C` の 4 関数の docstring の 1 行目がすべて固定値と一致、Python 関数 9 つの引数の並びがすべて一致、`comfy_kitchen.__version__` は存在しない（計画の記述どおり）。入口で `None, "not probed"` に戻している（399 行）。
8. **テスト**: C1〜C6 と G1 は計画の意図を満たす。件数は A-0 の 129 件合格・14 件スキップから A-1 の 140 件合格・22 件スキップへ。増分は合格 11 件（C1・C2・C3・C4 の shared 側 4・C6・C5 の 3）とスキップ 8 件（G1）で勘定が合う。A-4 は 162 件合格で、G1 の 8 通りすべてを含む。C4 の weakref は、偽の量子化が `x2d.to(float64)` という新しいテンソルを返して引数を持たないので、意味のある確認になっている。`shares_input` を「常に真」に差し替えた影響: 本物の `shares_input` が CPU で走るのは C3（全条件）と C6・既存の vram16 テスト（条件 ① のところまで）。条件 ② 以降を本物の `cb` で通るのは G1 と A-5 だけ。この整理でよいと考えます。② 以降の条件は CUDA のテンソルでしか真にならず、CPU では偽物でしか確かめられないうえ、A-5 が `forward_rows` 0 回を数えて製品の形で裏を取るため。既存のテストの変更は、C4 として計画された `test_vram16_block_forward_matches_off` への `shared` の parametrize の追加だけ（`shared=False` 側が元のテスト）。
9. **触ってはならないもの**: `git status --short` で変更があるのは 5 ファイルだけ。`dit.py`・`constants.py`・`engine/` に差分はない。`forward`・`forward_rows`・`swap_in_int8_linears` は差分の中で 1 行も変わっていない。`require_cuda_int8` の戻り値と、ロードを拒否する条件も変わっていない。CRLF: 5 ファイルとも作業ツリーの全行が CRLF（行数と CR の数が一致）。
10. **コメント**: 英語で、既存の作法（台帳の番号を書く、理由を書く）に合っている。写した行の元の行番号を、2053・2055・2072〜2075・2078・2190〜2193・2211〜2223・2246〜2289 行について確かめた。すべて正しい。古い記述が残っていたのは指摘 2 だけ。
11. **行数**: 約 +281 行の内訳は、表 4 つ（約 60 行）、モジュールの docstring（19 行）、写し（約 75 行）、残りは検査と 3 メソッド。過剰とまでは言えない。削れるのは指摘 4 の約 9 行くらいで、`_C` の docstring の固定値は全文のまま残すべき（計画の minor 4 の裁定）。

## 未確認の懸念

- **ストリーム**: 量子化（`quantize_input`）と行列積（`forward_rows_q`）が別々の呼び出しになったので、両者の間で現在のストリームが変わると順序の保証が変わる。`block_forward` の中でストリームを切り替える行は見当たらない。ブロックの入れ替えの先読みは別のストリームで重みを写すが、計算は現在のストリームで走る。ただし、先読みの実装（`engine/transformer/block_swap_prefetch.py`）までは読んでいない。A-5 と A-8 の md5 の一致で、実地に確かめられる。
- **CUTLASS の核の選び方とポインタの揃え**: `xq` は `int8_linear` の中と同じく新しく確保した連続テンソルで、重みの切片は両方の経路で同じ。だから同じ核に入るはずだが、核の内部の選び方までは確かめていない。G1 の 8 通りが `torch.equal` で合格しているので、実地の裏は取れている。

## 確認できた良い点

1. **写し（2190〜2289 行）**: `out` の確保、`_int8_weight_scale_arg(weight_scale, dev)`、bias の `_gemm_vector_arg(bias, dev, out_dtype)` と空テンソル `_empty_cuda_tensor(dev, out_dtype)`、`used_cutlass` の判定（`_prefer_cublas_int8_fallback(m, n, k, xq.get_device())`・`cb._DISABLE_CUTLASS_INT8`・`_cuda_device_supports_cutlass_int8_dequant(xq)`）、`ws_cutlass` の式、`_C.cutlass_int8_dequant` の 8 引数の並び、`False` のときの `cublas_gemm_int8`（`padded_n = n`・`get_cublas_workspace()`）、`dequantize_int8_linear` の 7 引数の並び。どれも元と同じで、`_wrap_for_dlpack` もすべての引数に付いている。`_DISABLE_CUTLASS_INT8` は呼ぶたびに `cb` のモジュール変数を読むので、元（モジュールのグローバル変数を呼ぶたびに読む）と同じく、G1 の `monkeypatch` が両方の経路に効く。
2. **前処理の行（2053〜2078 行）**: 重みの `contiguous()`、`stream_ptr = torch.cuda.current_stream(dev).cuda_stream`、m と k、形の assert、`DTYPE_TO_CODE[out_dtype]` も写されている。`out_dtype` は `forward_rows` と同じく常に `torch.bfloat16`。`_finish` は、出力が 2 次元で残差もないので恒等（`_activations.apply_residual` 77〜78 行）。
3. **登録を通らないことの埋め合わせ**: 制約（x の dtype と次元・重みは int8 で 2 次元・スケールは f32・capability (7,5) 以上・CUDA のデバイス）のうち、重みとスケールのデバイスを ⑦、capability を ⑥ で守っている。dtype は `H3Int8Linear` の構造上いつも満たされる。
4. 検査関数を CUDA に触れない純関数（`_shared_quant_missing`）と、状態を持つ部分（`_probe_shared_quant`）に分けている。そのおかげで C1 が実物の `cb` で、C2 が偽物で、それぞれ素直に書けている。
5. `lowvram.py` の変更は、戻す分岐が 1 か所でブロックごとに 1 回だけ判定する。`h` の解放の位置が 1 行で読める。
