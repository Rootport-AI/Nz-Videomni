# 敵対的コードレビュー A-G6: VRAM 16GB mode（弾 A・エンジン）

対象: `Nz-Videomni` dev の未コミットの作業ツリー（`engine_h3/constants.py`・`dit.py`・`pipeline_h3.py`・`quant_linear.py`・`worker.py`・`tests/test_h3_engine.py`・新規 `engine_h3/lowvram.py`）。設計の正本は `PLAN_vram16.md` §1〜§3。
実行したもの: `run_h3_pytest.py tests\test_h3_engine.py -q`（`PYTHONUTF8=1`）は全件合格、GPU のテスト 2 件は飛ばされた（従来の 2 件。新しい GPU テストは無い）。GPU・git の変更はしていない。

## 結論

**Critical 0・Major 1・minor 12（うち 4 件は記録だけで直さなくてよい）。**

総評: 裁定 45（off の経路に触れない）と裁定 46（既存の関数を範囲ごとに呼ぶ）は、コードを読んだ範囲では守られている。off のとき、テンソルの演算・dtype・形・順序が変わる行は 1 つも無い。足されたのは Python の属性の代入と分岐、整数の計算、同じ値の再代入、`raise` で投げ直すだけの `except` だけ。on の経路は `_modulate`／`_gate`／`block.mlp`／`_attention`／`apply_rope`／`_linear_f32`／`fl.norm` を書き写さずに呼んでいて、受け皿への書き込みで dtype が変わるのは最終層の bf16→f32（値は変わらない）だけ。退避・上限・OOM の包み直し・常駐数の毎回の代入も計画 §3.3 どおり。担当が自分で決めた点（観点 8）はどれも設計の範囲内で、裁定 49 に反する設計変更は無い。

穴は 1 つ。**`forward_rows` の実際のコードを実行するテストがどこにも無い**（CPU のテストは偽物に差し替えており、計画 §3.5 の GPU テストは書かれていない）。A-G2 の前に足す必要がある。ほかは小さな文言・テストの補強・記録の注意。

---

## 指摘

### M1【Major】`forward_rows` の実際のコードがどのテストでも実行されない（計画 §3.5 の GPU テストが無い）

- 場所: `tests/test_h3_engine.py:1309-1320`（`_int8_rows_as_float64`・`_fake_int8`）、`engine_h3/quant_linear.py:123-154`
- 何が問題か: CPU の等価テストは `H3Int8Linear.forward_rows` を**メソッドごと**テスト側の偽物に差し替えている。そのため、本物の `forward_rows` が持つ `self.weight[n0:n1]`・`self.weight_scale[n0:n1]`・`bias[n0:n1]` の切り出し、bf16 への変換、`reshape(*lead, n1 - n0)` は一度も実行されない。テストが確かめているのは「テストが書いた偽物の切り方」だけ。計画 §3.5 の GPU テスト（`forward_rows` の結合と `forward` の `torch.equal`、`forward(x[r])` の縦結合と `forward(x)` の `torch.equal`〔m＝実寸・4,096・129、(n,k) 4 種〕、`k.mean` の切り出し比較）も、`H3_GPU_TESTS` の付いた新しいテストとしては 1 本も無い（`grep` で確認。GPU で飛ばされたのは従来の 2 件だけ）。`forward_rows` の docstring は「checked on the GPU, gate A-G2」と書いているが、その検査はまだ存在しない。
- 根拠: `_fake_int8` は `monkeypatch.setattr(H3Int8Linear, "forward_rows", _int8_rows_as_float64)`。`H3_GPU_TESTS` が出てくるのは 25 行目（説明）と 690 行目（従来のテスト）だけ。
- 直し方の案:
  (a) CPU: 偽物はメソッドではなく `quant_linear._ck` の側で 1 つ用意する（`int8_linear(x2d, w, s, b, ...)` を float64 で計算して bf16 に戻す関数を持つ小さな名前空間）。こうすると本物の `forward` と `forward_rows` が両方ともそのまま実行され、切り出しと `reshape` もテストされる。`_int8_as_float64`／`_int8_rows_as_float64` の 2 つが 1 つになるので、テストのコードも減る。
  (b) GPU: 計画 §3.5 の 3 本を `H3_GPU_TESTS=1` で足す（A-G2 で使うもの。実行は監督の了承の後）。
- 観点: 2（ビット一致の裏付け）

### m1【minor】`row_chunks` の docstring の「chunk / 2 より多い」が不正確

- 場所: `engine_h3/lowvram.py:58-60`
- 何が問題か: S＝4,097・塊 4,096 だと塊は 2,049 行と 2,048 行になる。2,048 は chunk/2 と**等しい**ので、「more than chunk / 2 rows」は成り立たない。実害は無い（計画が求める「端も 129 行以上」は満たしている）。
- 直し方の案: 「at least ``chunk // 2`` rows」に直す。
- 観点: 2

### m2【minor】`blocks_on_gpu_for` の `num_blocks=50` の既定値は、ブロック数の正本が 2 つになる

- 場所: `engine_h3/lowvram.py:42-43, 77`
- 何が問題か: 本番の呼び出し（`pipeline_h3.py`）は `len(dit.blocks)` を渡しているので、既定値の 50（`_NUM_BLOCKS`）を使うのはテストだけ。ブロック数の正本は `cfg.num_layers` なので、モジュールに 50 を書いておくと正本が 2 つになる。計画の「≥ 50」をブロック数に置き換えたこと自体は、設計の範囲内の改善。
- 直し方の案: `num_blocks` をキーワード専用の必須引数にして `_NUM_BLOCKS` を消す。テストは `num_blocks=50` を明示して渡す。
- 観点: 8（コードの質）

### m3【minor】ヘッドを群に分けた注意計算が `H3Attention.forward` と一致することを、直接は確かめていない

- 場所: `tests/test_h3_engine.py` の `test_vram16_block_forward_matches_off`
- 何が問題か: 計画 §3.5 は「ヘッド 4・群 3 で on の attention が `H3Attention.forward` と `torch.equal`」。実装はブロックの出力どうしの比較で代えている。行やヘッドの位置を取り違えるような構造の誤りなら、差が大きいのでこれで十分に捕まる。ただ、`o` が最終桁で 1 だけずれたときは、`out_proj`（float64 で計算して bf16 に丸める）と `_gate`（bf16 に丸める）の丸めで消えることがある（推測）。CPU では同じ計算を同じデータに行うので、実際に差が出る心配は小さい。
- 直し方の案: `attn.out_proj` の入力を両方の経路で控えて（`out_proj.forward` を包んで入力を記録するだけ）、`o` どうしを `torch.equal` で比べる 1 行を足す。安く済み、計画の文言どおりになる。
- 観点: 2・6

### m4【minor】「on のジョブで減らした常駐数が次の off のジョブに残らない」ことを確かめるテストが無い

- 場所: `engine_h3/pipeline_h3.py:687`（`service.blocks_on_gpu = n`）、テストの側
- 何が問題か: 毎回代入するのは、常駐数の持ち越しを防ぐためにある（計画 #8）。しかし、常駐の遷移のテストは系列が短いので `blocks_on_gpu_for` が読み込み時の値をそのまま返し、持ち越しの場面が起きない。いまのコードは正しい（`install` は `self.blocks_on_gpu` を読む。`block_swap_service.py:233, 317`）。
- 直し方の案: `_tiny_pipe_and_dit(blocks_on_gpu=2)` で `_place_dit(dit, blocks_on_gpu=1)` → `_place_dit(dit)` と呼び、`service.blocks_on_gpu == 2` を確かめる数行のテスト。
- 観点: 1・6

### m5【minor】`_vram16_finish`（上限を 1.0 に戻す・例外を出さない）がテストされていない

- 場所: `engine_h3/pipeline_h3.py:711-723`
- 何が問題か: CPU のテストでは `cap_set=False` と `counters0=None` なので、上限を戻す行と差分を取る行が実行されない。
- 直し方の案: `torch.cuda.set_per_process_memory_fraction`／`memory_stats` を monkeypatch して `_vram16_finish` を直接呼ぶ。(i) 1.0 で呼ばれること、(ii) 差分が記録されること、(iii) `memory_stats` が例外を投げてもログに書くだけで外へ出ないこと、の 3 点を確かめる小さなテスト。
- 観点: 6

### m6【minor】退避の後に `empty_cache` を呼ばないため、約 1.47GB が割り当て器のキャッシュに残る（記録の読み方に注意）

- 場所: `engine_h3/pipeline_h3.py:799-800`・`_evict_text_modules`
- 何が問題か: token_refiner と `condition_proj` を GPU から外しても、解放された領域は割り当て器のキャッシュに残る。重みの区画は 1 つあたり約 231MB 以下なので、U（約 408MB）のような大きな活性化はそこに収まらない。そのため新しい区画が要り、上限に当たった時点で「キャッシュを全部返して再試行」が起き、それが `num_alloc_retries` に 1 回以上数えられる見込み（推測。割り当て器の内部の挙動は読んでいない）。数値には影響しないが、A-G4 で「再試行の回数＝メモリの逼迫」と読むと誤る。空いた区画（§168 の原因）を自分で作り出していることにもなる。
- 直し方の案: どちらか。(a) cuda のときだけ退避の直後に `torch.cuda.empty_cache()` を 1 回呼ぶ（同期は 1 ジョブに 1 回だけ。数値は変わらない）。(b) 呼ばない理由を docstring に 1 文書き、A-G4 で再試行の回数を読むときの注記にする。担当が決めた点で、計画はどちらとも言っていない。監督の判断で。
- 観点: 3・8

### m7【minor・記録だけ】カーネルが投げる `RuntimeError("CUDA error: out of memory")` は案内文に包まれない

- 場所: `engine_h3/pipeline_h3.py:874-877`
- 何が問題か: 包み直すのは `torch.OutOfMemoryError` だけ。cuBLAS や CUTLASS、sage が自分で確保に失敗した場合は、普通の `RuntimeError` として上がってくることがある（推測）。そのときは案内文が付かない。ただしアプリ側の `_is_oom` は文言の「out of memory」で GPU_OOM に分類するので、503 になってワーカーを unload する動きは変わらない。
- 直し方の案: 計画どおりなので直さなくてよい。設計正本 §13 の「既知の限界」に 1 行書く程度。
- 観点: 3

### m8【minor】群の出力を書き込むとき、`[S, g*D]` の一時領域が 1 つ余計にできる

- 場所: `engine_h3/lowvram.py:153`
- 何が問題か: `out` は転置したビューで連続していないので、`out.reshape(s, g * d)` がコピーを作る（S＝39,764・群 8 で約 81MB）。続く `copy_` でもう一度書き込む。
- 直し方の案: `o[:, h0 * d:h1 * d].view(s, g, d).copy_(out)`。スライスの形は (inner, 1) の刻みなので、`view` で最後の次元を (g, d) に割れる。値はまったく同じコピーなのでビット一致は変わらない。山の区間（sage の呼び出しの中）の外なので、効果は小さい。
- 観点: 2・7

### m9【minor】`H3DiT.forward` の型注釈が `tokens: torch.Tensor` のまま

- 場所: `engine_h3/dit.py:547`
- 何が問題か: on の経路は `tokens=None` で呼ぶ。
- 直し方の案: `tokens: torch.Tensor | None` に直す（実行には影響しない）。
- 観点: 7

### m10【minor】`_denoise` の on のときの準備（約 20 行）が `try` の中に直接書かれている

- 場所: `engine_h3/pipeline_h3.py:777-797`
- 何が問題か: 片付けの側は `_vram16_finish` にまとめてあるのに、準備の側は `_denoise` の中に直接書かれていて、off の読み手が毎回この塊を読み飛ばすことになる。誤りではない。
- 直し方の案（任意）: `_vram16_begin(dev, seq_rows, n_blocks, dit) -> (plan, entry, counters0, cap_set, n_blocks)` にまとめると、`_denoise` の中は `if req.vram16_mode:` の 1 行の分岐と呼び出しだけになる。監督の判断で。
- 観点: 7

### m11【minor・記録だけ】上限のせいで先読みの受け皿が確保できないと、同期のやり方に黙って切り替わる

- 場所: 既存の `engine/transformer/block_swap_service.py:300-327`（`except Exception` で同期のやり方へ切り替える）
- 何が問題か: 上限の下で受け皿 10 本の確保が OOM になると、既存の仕組みが例外を握りつぶし、そのジョブは同期のやり方で走る。VRAM が共有メモリへ溢れるわけではない（裁定 47 の「黙って退避を許す経路」ではない）が、遅くなる。
- 直し方の案: コードは変えない。A-G4 と A-G5 で `block_swap_prefetch_used` が `"on"` であることを確かめる項目を足す。
- 観点: 3

### m12【minor・記録だけ】改行コードの混在は無い

- 場所: 全ファイル
- 何が問題か: バイト単位で数えた。1 つのファイルの中で混ざっているものは無い（`quant_linear.py` だけ全行 CRLF、ほかは作業コピーも全行 LF。インデックスは LF）。`git diff` の「LF will be replaced by CRLF」は `core.autocrlf=true` による警告で、コミット時に LF に揃えられる。
- 直し方の案: 対応は要らない。
- 観点: 7

### 補足（指摘ではない・記録だけ）: `forward_rows` が `forward` の本体のほぼ写しであること

`forward` を共通の補助関数に書き換えると off の経路に触れる（裁定 45）。約 15 行の重複は、この裁定を守るための代償として妥当と判断する。直さない。

---

## 観点ごとの確認（問題が無かったもの）

1. **裁定 45**
   - `dit.py`: off のときは `if self.vram16 is not None` が偽になり、`tokens is None` も偽になる。毎回の `block.vram16 = None` は Python の属性の代入にすぎない（`nn.Module.__setattr__` はテンソルでもモジュールでもない値をインスタンスの辞書に入れる）。`final_rows(..., vram16=None)` は従来の行をそのまま通る。
   - `_place_dit`: `n_blocks = int(self.options.blocks_on_gpu)` を渡すので `n` は従来と同じ値。`options` はパイプラインを作るときに固定され、`load` で作り直されない（`pipeline_h3.py:371-395`）。`service.blocks_on_gpu = n` は作ったときと同じ値の再代入で、結果は「最初の値のまま」と同じ。
   - `except torch.OutOfMemoryError`: `vram16 is None` なら、引数なしの `raise` で同じ例外の型・文言のまま投げ直す。
   - `_denoise` の 3 行: off のときは `box = None` で、`dit(tokens, ..., vram16=None, tokens_box=None)`。
   - `seq_rows` は整数の計算だけ。
   - セルフテストの `generate` に `vram16_mode: False` が増えるが、指紋は mp4 のバイト列の sha256 で、ワーカーは mp4 にメタデータを書かない（`media_out` に該当なし）。A-G3 の指紋には影響しない。
2. **裁定 46**
   - qkv の行の範囲 `part*inner + h0*D`〜`part*inner + h1*D` は、積み重ねた `[q; k; v]` の `view(s, 3, H, D)` と一致する。
   - `o` の列 `h0*D:h1*D` は off の `reshape(s, H*D)` の列の並び（h*D + d）と一致する。
   - `adaln_index[r0:r1]` は行ごとの `index_select` なので、全体で計算したものと同じ値になる。
   - 受け皿は `empty_like`（bf16）で、書き込む値も bf16（`_modulate`／`_gate` は x の dtype、`block.mlp` は bf16）。
   - 最終層は bf16→f32 の `copy_` のあと `_linear_f32` を 1 回呼ぶ。`_linear_f32` の中の `x.to(f32)` は f32 に対して何もしない。
   - `row_chunks` は等分で、端の塊は chunk//2 行以上。
   - 補足: off では q が strided のビュー（刻みは 3·inner）、on では連続したテンソルとして `q_norm`（RMSNorm）に入る。CPU では値が一致することを確認した。GPU で縮約の分け方が変わるかどうかは推測の域を出ないので、A-G2 で判定する。
3. **§3.3**
   - 上限を課すのは `try` の内側・`_place_dit` の前で、cuda のときだけ。`finally` の先頭で `_vram16_finish` を呼び、例外は外へ出さない。
   - S の式はレイアウトの合計と一致する（`refine_text` は行数を変えない）。
   - 退避は、常駐のジョブなら CPU、そうでなければ meta。
   - キー無しのジョブが常駐の DiT を使ったとき: `finally` の `elif` で `_resident` から外れ、`generate` の `finally` の `release_resident()` で捨てられるので、meta に落としてよい。テストの J3 → J4（`len(builds) == 2`）でも確認されている。
   - 常駐のジョブの次のジョブ: `_place_dit` の `move_module_tree(skip=ブロック)` が 2 つのモジュールを GPU に戻す。
   - OOM: 案内文に「(CUDA out of memory)」が入り、`from exc` で元の例外をつなぎ、「解像度」を使って「画角」を使っていない。文言は計画の文と完全に一致する。
   - `entry["vram16"]` は失敗したときも書かれる。
   - 埋め込みの参照: `_denoise` の局所変数は None、箱は pop で空、forward の引数は `del`。`_call_impl` が持つのは None。swap の包みの args はブロック 0 の呼び出しの間だけ。ブロック 0 のあとに解放されることは weakref のテストで確認されている。
4. **`forward_rows`**: 毎回 `self.weight` から切っている。[N,1] の行のスライスは連続。bias は切ってから bf16 にする。`spec` は `forward` と同じ。`reshape(*lead, n1-n0)`。本番では n0 が 128 の倍数なので、重みの先頭の位置は 16 バイト境界に揃う。
5. **worker**: `_GENERATE_FIELDS`、`_or("vram16_mode", False, bool)`（`keep_resident` と同じ作法）、`--vram16`（`BooleanOptionalAction`）、golden の末尾、docstring（43・66・79 行）、ログ行。すべて計画どおり。
6. **テスト**: weakref・spy（`move_module_tree`／`drop_to_meta`）・off が `lowvram` を呼ばないこと・OOM の文言・golden・parametrize（常駐の遷移・OOM・分割単位 4 通り）・偽の int8 が float64 であること。いずれもある。足りないものは M1・m3・m4・m5。冗長なテストは見当たらない。
7. **`lowvram.py`**: 196 行で、`_rows_into` によって 4 か所の塊のループが 1 行ずつになり、読みやすい。重複は無い。docstring は現状を正しく言っており（m1 を除く）、「inference_mode」の語は `engine_h3` のどこにも無い。`dit.py` の分岐は最小限（片側だけの遅延 import）。
8. **計画からの逸脱**: `num_blocks` 引数（m2）、群のループを `block_forward` の中に置いたこと（計画 §3.2 の手順 3 そのもの）、`_evict_text_modules`・`_vram16_finish`・`_vram16_counters`・`VRAM16_OOM_MESSAGE`・記録のキー（`text_modules` と `inactive_split_peak_mib` を足した）、`empty_cache` を呼ばないこと（m6）。どれも実装の詳細で、裁定 49 の「設計の変更」には当たらない。

---

## 一覧（監督へ）

| 番号 | 重さ | 1 行要約 |
|---|---|---|
| M1 | Major | `forward_rows` の本物のコードを実行するテストが無い（CPU は偽物に差し替え・計画 §3.5 の GPU テスト 3 本が無い）→ `_ck` の側に偽物を置く＋GPU テストを足す |
| m1 | minor | `row_chunks` の docstring「chunk/2 より多い」は誤り（4,097→2,048 行）→「chunk//2 行以上」 |
| m2 | minor | `num_blocks=50` の既定値はブロック数の正本が 2 つになる → 必須引数にする |
| m3 | minor | 注意計算の一致はブロック単位の比較だけ（丸めで消える可能性）→ `out_proj` の入力どうしを比べる |
| m4 | minor | 常駐数が次のジョブへ持ち越されないことのテストが無い |
| m5 | minor | `_vram16_finish`（上限を戻す・例外を出さない）のテストが無い |
| m6 | minor | 退避の後に `empty_cache` を呼ばず、再試行の回数が増えて見える見込み（推測）→ 1 回呼ぶか注記にする |
| m7 | minor（記録） | カーネル由来の RuntimeError の OOM には案内文が付かない（GPU_OOM の分類は効く） |
| m8 | minor | `out.reshape` が余計なコピーを作る → `o[...].view(s,g,d).copy_(out)` |
| m9 | minor | `tokens` の型注釈を `Tensor | None` に |
| m10 | minor（任意） | on のときの準備の約 20 行を `_vram16_begin` にまとめると `_denoise` が読みやすい |
| m11 | minor（記録） | 上限の下で先読みの受け皿が確保できないと、同期のやり方へ黙って切り替わる → A-G4 で `block_swap_prefetch_used` を確かめる |
| m12 | minor（記録） | 改行コードの混在は無い（対応は要らない） |
