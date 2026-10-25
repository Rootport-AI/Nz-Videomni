# R1 台帳 §1-105 ①〜③ の調査（MiniMax H3 の 1 ステップの高速化・第 1 弾：数値等価）

調査日 2026-10-24。対象はブランチ dev の作業ツリー。GPU を使う処理は一切行っていない（コード読解・既存の計測記録の読み直し・CPU だけの import による関数の署名の確認のみ）。

## 結論

1. **① QKV の活性化量子化の共有は有力。** 現状の VRAM 16GB mode は、1 ブロックで同じ入力 `h`（bf16 `[S, 5376]`）を 21 回、ConvRot の回転込みで int8 に量子化している（コードで確定）。これを 1 回にすると、1 ステップで 1,000 回（20 回 × 50 ブロック）の量子化が消える。1 回あたりの時間は、メモリ帯域から見た下限が約 0.95 ms、既存の実測（A-G5 のヘッド群 8／14／28）から逆算した上限が約 1.6 ms。**20 ステップで約 19〜32 秒、Turbo の 4 ステップで約 4〜6 秒の短縮**の見込み（概算。実測は別途）。§14.6 の「15 秒以上＝有力」に当たる。ただし **comfy-kitchen 0.2.37 には「量子化だけ」「量子化済みの入力で行列積だけ」の公開 API が無い**。バックエンドの内部関数（`comfy_kitchen/backends/cuda/__init__.py` の `quantize_int8_rowwise_convrot64` と、拡張モジュール `_C.cutlass_int8_dequant`）を直接使う形になる。設計正本 §5 の「関数 API の `int8_linear` に任せる」という方針から外れるので、オーナーの裁定が要る。
2. **② ステップ間の cyclic prefetch は見送りが妥当。** ステップの頭でブロック 0 が毎回「冷えた状態から同期で」読み込まれることはコードで確定した。ただし GPU が待つのは 1 ステップあたり約 45 ms（見積もり）で、**20 ステップで約 0.9 秒、4 ステップで約 0.14 秒**にとどまる。しかも共有部品（`engine/transformer/block_swap_prefetch.py`）の変更が必須で、H3 のパイプラインだけでは実現できない（理由は ② の節）。
3. **③ CPU の正本から pinned staging への先読みも見送りが妥当。** メモリコピー（memcpy）は計算を流すのと同じスレッドで、同期で走っている（コードで確定）。しかしステップの途中では GPU が使用率ほぼ 100% で働き続けており（既存の記録）、CPU は GPU より先を走っている。したがって GPU の待ちはステップの頭にしか生じない。その分は ② の約 45 ms と同じものなので、③ 単独で取れるのは最大で **20 ステップあたり約 0.55 秒**。

---

## ① QKV の活性化量子化の共有

### 確認済みの事実

**現状の量子化の回数（1 ブロック・VRAM 16GB mode）**

- `engine_h3/lowvram.py:138-143`: 56 ヘッドを 8 ヘッドずつの 7 群に分け、群ごとに `attn.qkv_proj.forward_rows(h, n0, n1)` を q・k・v の 3 回呼ぶ。毎回、入力 `h` の全行（S 行）を渡している。
- `engine_h3/quant_linear.py:140-158`（`forward_rows`）: 毎回 `x2d`（`h` そのもの）を `comfy_kitchen.int8_linear` に渡す。重みは `self.weight[n0:n1]`（1,024 行）で切る。
- `comfy_kitchen/backends/cuda/__init__.py:2035`（`int8_linear` の CUDA 実装）: 活性化の量子化は**この関数の中**で毎回行われる。K＝5,376 では、2061-2067 行の条件 `_fused_convrot_ok`（convrot が真・groupsize が 256・K が 256 の倍数・256 ≤ K ≤ 16,384・共有メモリの要求 54,272 バイトが上限に収まる）が真になる。そのため 2132-2149 行の `_C.quantize_int8_rowwise_convrot64`（ConvRot の回転と行ごとの int8 量子化を 1 つのカーネルで行う）が走る。その後 2219-2289 行で、量子化済みの入力と int8 の重みの行列積（CUTLASS の `_C.cutlass_int8_dequant`。これが偽を返したときは cuBLAS の `_C.cublas_gemm_int8` と `_C.dequantize_int8_linear`）が走る。
- したがって、**1 ブロックで同じ `h` の量子化が 3 × 7＝21 回走る**。設計正本 §6.5 の末尾の記述どおり。
- **他の int8 層には重複が無い**:
  - `out_proj` は `o` の全体に 1 回（`lowvram.py:154`）。
  - `fc1`・`fc2` は、`block.mlp(h2[r0:r1])` を行の塊ごとに呼ぶ（`lowvram.py:167`。`dit.py:259-261`）。塊は互いに重ならない行の範囲なので、同じ入力の量子化は重複しない。
  - 1 ブロックの量子化の回数は、qkv 21 回＋out_proj 1 回＋fc1 10 回＋fc2 10 回（S＝39,764 は 4,096 行の塊 10 個）。共有で消えるのは qkv の 20 回だけ。
- off の経路は qkv を 1 回で計算するので量子化も 1 回（`dit.py:240`）。本件は on の経路だけの話。

**comfy-kitchen 0.2.37 の API（`.venv-engine-h3/Lib/site-packages/comfy_kitchen/`。版は `engine_h3/venv-engine-h3.freeze.txt:50` で `comfy-kitchen==0.2.37` に固定）**

| 区分 | 場所と名前 | 署名 | 公開か |
|---|---|---|---|
| 量子化＋行列積（現行） | `__init__.py:997` `int8_linear` | `(x, weight, weight_scale, bias=None, out_dtype=None, convrot=False, convrot_groupsize=256, input_act=None, input_act_weight=None, input_act_eps=0.0, residual=None, residual_scale=None) -> Tensor` | 公開（`__all__`） |
| 回転なしの行ごとの量子化 | `__init__.py:936` `quantize_int8_rowwise` | `(x, stochastic_rounding=0) -> (Tensor, Tensor)` | 公開。ただし ConvRot の回転をしないので H3 には使えない |
| ConvRot の回転＋行ごとの量子化 | `backends/cuda/__init__.py:1588` `quantize_int8_rowwise_convrot64` | `(weight_2d, group_size, stochastic_rounding=0, input_act=None, input_act_weight=None, input_act_eps=0.0) -> (q_2d int8 [M,K], scales_2d f32 [M,1])` | **非公開**。バックエンドのモジュールの `__all__`（80 行）には入っているが、レジストリの制約表（`_build_constraints`）に登録が無く、トップレベルからも出ていない。引数名が `weight_2d` なのは、もとは重みの量子化用に書かれたためと見られる（推測） |
| 同上（別の核） | `backends/cuda/__init__.py:1379` `quantize_int8_rowwise_convrot` | `(x_2d, group_size, stochastic_rounding=0)` | 非公開。K＝5,376 の現行の経路では使われない |
| 量子化済みの入力での行列積 | 拡張 `_C.cutlass_int8_dequant` | `(a: int8[M,K], b: int8[N,K], xs: f32, ws: f32, bias, d: [M,N], out_dtype_code, stream_ptr) -> bool`（docstring: "INT8 GEMM + fused rowwise x colwise dequant + bias via CUTLASS; false -> fall back to cuBLAS"） | **非公開**（拡張の生の関数） |
| 同上の Python の包み | `backends/cuda/__init__.py:941` `_int4_linear_via_int8_values` | `(x_int8, weight_int8, x_scale, weight_scale, bias, out_dtype)` | 非公開（先頭が下線）。INT4 の代替経路用で、スケールの形の渡し方が `int8_linear` と少し違う（`weight_scale` を `[N]` に平らにする） |
| 参考 | `__init__.py:951` `mm_int8` | `(a, b) -> int32` | `__all__` に無い。中身は eager の `torch._int_mm` で、別の核。逆量子化の丸めの順序が CUTLASS の後処理と同じ保証が無いので不適 |

- `QuantizedTensor`（`tensor/int8.py`）の経路も、入力が量子化済みなら**いったん逆量子化してから** `int8_linear` に渡す（`tensor/int8.py:277-279`）。量子化済みの入力を受け付ける公開の経路は無い。
- 拡張の生の関数の docstring は CPU だけの import で確認した（`quantize_int8_rowwise_convrot64(input, output, scales, group_size, stochastic, act_code, seed, act_weight, act_eps, stream_ptr)`）。
- **版の間での変化は未確認**（手元にあるのは 0.2.37 だけで、上流の履歴は読んでいない）。非公開の関数なので、版を上げるたびに変わりうると考えるべき。

**ConvRot の回転は層に依存しない**

- 回転は活性化の側に、groupsize ごとのアダマール行列で掛ける（`int8_linear` の docstring「online activation rotation」。eager の参照実装は `backends/eager/quantization.py:1036-1042`）。行列は `tensor/int8_utils.py:11-38` の `_build_hadamard(size)` が作る「正規化した正則アダマール行列」で、**groupsize だけで決まる**。重み（層）ごとの乱数や置換は無い。重みのほうはファイルの時点で `W @ H^T` に回転済み（`quant_linear.py:25-29`）。
- 量子化を決めるのは入力 `x`・groupsize・`input_act`（H3 では None）・確率的丸め（H3 では無し）だけで、重みは関わらない（`int8_linear` の 2132-2149 行）。
- q・k・v は**同じ 1 つの層** `attn.qkv_proj` の重みの行（`[q; k; v]` を積んだもの。`dit.py:229`）で、記述子は層ごとに 1 つ。しかもファイルの 200 本の int8 層の記述子はすべて `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`（研究ノート §13.2。実行時の `LOAD_OK` も `quant_specs: ["int8_tensorwise/convrot=True/g256"]`。保管庫 `gates/runs/G5a_h8_r4096_b4/stdout.log`）。
- したがって **q・k・v の 21 回の呼び出しは、量子化の段では入力も引数も完全に同じ**。

### 数値がビット一致する根拠

- 共有する形は「同じ関数（`_C.quantize_int8_rowwise_convrot64`）を同じ入力（`h` の全行）に同じ引数で 1 回だけ呼び、その結果（int8 の `[S, 5376]` と f32 の `[S, 1]`）を 21 回の行列積に渡す」もの。行の切り方は変えない（今も毎回 S 行の全体を量子化している）ので、「行の切り方で丸めが変わる」問題は起きない。
- 行列積の側は、現行と同じ `_C.cutlass_int8_dequant` に、同じ形（M＝39,764・N＝1,024・K＝5,376）で、同じ重みの切片（`self.weight[n0:n1]`。連続したビュー）を渡す。核の選び方は変わらない。
- 残る前提は「量子化の核が同じ入力に対して毎回同じ値を返すこと（決定的であること）」。核のソースは手元に無い（`.pyd` だけ）ので**コードでは未確認**。ただし次の 2 つが状況証拠になる。(a) 現行の on の経路は 21 回の独立した量子化の結果を使い、off（1 回の量子化）と sdpa で 50/50 ブロックがビット一致した（VERIFICATION_LOG §169.3 の A-G2）。(b) GPU テスト `tests/test_h3_engine.py:1876`（`forward_rows` の結合と `forward` の一致）も合格している。量子化が呼ぶたびに揺れるなら、これらは一致しない。
- 注意: 共有の補助関数は `int8_linear` の 2169-2290 行（m ≤ 2 の GEMV・Turing 専用の経路・CUTLASS・cuBLAS への退避）を**そっくり写す**必要がある。写し損ねると、別の核に入って一致が崩れる。安全策は「検証した経路の条件（convrot・g256・K の範囲・共有メモリが収まる・m > 2・Turing でない）を満たすときだけ共有し、それ以外は現行の `forward_rows` に戻す」こと。

### 見積もり（前提つき。実測ではない）

- 前提: GPU は RTX 4070 Ti SUPER。メモリ帯域は公称 672 GB/s（GDDR6X 21 Gbps × 256 bit）。L2 は公称 48 MB（未確認）で、入力 427 MB より十分小さいので、毎回 DRAM から読むとみなす。
- 量子化 1 回の処理量: 要素数 S × K＝39,764 × 5,376＝2.138 億。読み（bf16）427.5 MB＋書き（int8）213.8 MB＋スケール 0.16 MB で、計 641.5 MB。回転（256 点の高速アダマール変換）は 1 要素あたり十数回の演算で約 3 GFLOP。FP32 の公称性能（約 44 TFLOPS）なら 0.1 ms 未満なので、**メモリ律速**。
- 1 回の時間: 帯域 100% で 0.95 ms（下限）。実効 85% なら約 1.1 ms。
- 既存の実測との照合（推測を含む）: A-G5 のヘッド群 8→14→28 では、1 ブロックの量子化の回数が 21→12→6 に減り、2 ステップ目が 21.74→20.99→20.54 秒になった。差をすべて量子化のせいとみなすと、1 回あたり 0.75 秒 ÷ 450 回＝1.67 ms、1.20 秒 ÷ 750 回＝1.60 ms で、2 つの値がよく揃う。実際には sage の呼び出しの大きさや行列積の N の違いも効いているはずなので、**1.6 ms は量子化 1 回の上限**とみる。
- 1 ステップの短縮: 1,000 回 × 0.95〜1.6 ms＝**0.95〜1.6 秒**（1 ステップ約 22 秒の 4〜7%）。
- **20 ステップで約 19〜32 秒、Turbo の 4 ステップで約 3.8〜6.4 秒。**
- 系列が長いほど比例して増える（T2V 1280×768×311 は S＝89,388 で、1 ステップあたり約 2.1〜3.6 秒）。
- 設計正本 §6.5 の末尾は「実測の 1 ステップ（約 22 秒）に収まっている」と書いているが、上の見積もりどおりなら、この 21 回はヘッド群 28 の速さの大部分を説明する。その場合、**① を入れるとヘッド群の拡大（§3-245）の残りの得が小さくなる**（推測）。

### VRAM の増減（見積もり）

- 共有した量子化済みの入力は int8 の `[S, 5376]`＝213.8 MB（約 204 MiB＝0.5U）とスケール 0.16 MB。7 群のループの間、生かしておく。
- 現行でも呼び出しのたびに同じ大きさを一時的に確保しているが、各呼び出しの中で解放されるので、注意の山（sage の内部）には載っていない。
- (a) `h` を残す場合: 注意の山が **+0.5U（約 +204 MiB）**。割り当てのピーク 4,411 MiB が約 4,615 MiB になる見込み。
- (b) 量子化の直後に `h`（bf16 `[S, 5376]`＝1U）を捨てる場合: 山は **差し引き −0.5U（約 −204 MiB）**。LoRA のジョブでも、`lora.xa_for` は量子化の前に計算済みで、`lora.apply` は `xa` を渡されたときは `x` の行数しか見ない（`engine_h3/lora.py:295-306`）。行数を別に渡すように直せば、`h` を捨てられる。
- (b) を採れば、快適上限には中立か、わずかに有利（推測）。

### 実装の形の案（最小）

1. `engine_h3/quant_linear.py`
   - `H3Int8Linear.shares_input(x2d) -> bool`: 検証した経路の条件（`spec.convrot`・groupsize 256・CUDA・bf16・M > 2・K が 256 の倍数かつ `_CONVROT_FUSED_MAX_K` 以下・`_convrot_fused_shared_memory_fits`・Turing でない）を満たすか。
   - `H3Int8Linear.quantize_input(x) -> (xq, xs)`: `comfy_kitchen.backends.cuda.quantize_int8_rowwise_convrot64(x2d, 256)` を呼ぶ。`int8_linear` の 2132-2149 行と同じ `_C` の呼び出しになる（引数も一致することを確認した）。
   - `H3Int8Linear.forward_rows_q(xq, xs, n0, n1, rows_for_lora=None, lora_xa=None)`: 重み・スケール・bias を毎回 `[n0:n1]` で切る（block swap が `.data` を差し替えるので、切片を保持しない。今の `forward_rows` と同じ理由）。そのうえで、`int8_linear` の 2169-2290 行を写した補助関数 `_int8_gemm_prequantized(...)` を呼び、LoRA があれば今と同じく `lora.apply` を足す。
   - 起動時の検査（`require_cuda_int8` の隣）: 非公開の記号が在ること（`_C.quantize_int8_rowwise_convrot64`・`_C.cutlass_int8_dequant`・`_C.cublas_gemm_int8`・`_C.dequantize_int8_linear` と補助関数）を確かめる。無ければ現行の経路に戻す（または拒否する。どちらにするかは要裁定）。
2. `engine_h3/lowvram.py` の 137-153 行: ループの前で `shares_input` が真なら `hq, hs = attn.qkv_proj.quantize_input(h)` を 1 回だけ行い、`del h`（案 (b)）。ループの中の 3 本を `forward_rows_q(hq, hs, …, lora_xa=xa)` にする。偽なら今のまま。
3. テスト
   - CPU テスト（`tests/test_h3_engine.py:1551-1590` の偽の `_ck`）に、量子化と行列積に分けた偽物（float64 の参照）を足し、on と off の一致の検査を保つ。
   - 非公開の記号の存在と、引数の並び（docstring）を固定するテストを足す。
   - GPU テスト（1876 行の隣）に「共有した量子化で 21 切片を結合したもの＝`forward`」を足す。
4. 触らないもの: off の経路（`dit.py`）・共有部品・`out_proj`／`fc1`／`fc2`。
5. 設計正本 §6.5「置き方」の「`lowvram.py` は新しい計算を書かない」との関係: 核は同じだが、`int8_linear` を 2 段に分けた補助関数が新しく入る。§5 の「関数 API の `int8_linear` に任せる」からも外れる。**オーナーの裁定事項。**

### 公開 API だけで済ませる代替案（参考・非推奨）

- 群ごとに q・k・v の重みの行を 1 本の `[3072, 5376]` に寄せ集めて、`int8_linear` を 1 群 1 回呼ぶ（21 回→7 回）。短縮は ① の 14/20（20 ステップで約 13〜22 秒）。
- ただし出力 `[S, 3072]` から切り出した q・k・v は連続でない。sdpa に連続でない v を渡すのは既知の不正アクセスの境界に触れ（VERIFICATION_LOG §168.3 T6）、設計の「採らないもの」（strided な渡し）に当たる。避けるには連続化のコピーが要り、得が減る。
- 寄せ集めのコピーも毎ステップ要る（1 群 16.5 MB）。

### 計測の設計（GPU の使用は監督が別途了承を取る）

- **M0（最小・数分）**: `[39764, 5376]` の bf16 の乱数に `quantize_int8_rowwise_convrot64` を 100 回かけ、`torch.cuda.Event` で 1 回あたりの時間を測る。同じ形の `int8_linear`（N＝1,024）も測り、量子化と行列積の比を出す。見積もりの 0.95〜1.6 ms を直接確かめる。
- **M1（製品の条件・2〜3 ステップ）**: 後述の共通の計測台で torch.profiler を 1 ステップだけ掛け、量子化の核の合計時間と回数（期待値は 1 ステップで 2,100＋50＋500＋500 回）を出す。qkv の分の合計 × 20/21 が ① の短縮の上限の実測値になる。
- **M2（実装後の A/B）**: 現行と共有版を同じ条件（I2V 896×1152×124・sage・VRAM 16GB mode on・常駐 on・決定性 on・3 ステップ）で回し、2・3 ステップ目の秒数と、割り当て／予約のピークを比べる。案 (a) と (b) の両方を測る。

### 合格の門との噛み合わせ

- `g2_block_equal.py`: on と off をブロックごとに比べる台本なので、**台本を変えずにそのまま乗る**。sdpa は 50/50 のビット一致、sage は前回と同じ 2 ブロック（22・40）だけが違い、その差の数値（最大絶対差 256／1,792 など）が前回の `g2_block_equal.json` と全桁で同じであること、を合格条件にする。
- 「旧 on と新 on のビット一致」を直接見たいなら、台本に比較を 1 本足す必要がある（旧 `block_forward` の写しを台本の中に持つ小さな変更）。
- `g3.sh`（sdpa・640×384・124 フレーム・20 ステップの指紋 `0da6111c…`。on と off）: 台本の変更なしで乗る。
- LoRA の門（VERIFICATION_LOG §176。off と on の mp4 の sha256 の一致）: LoRA の経路（`h` の寿命・`lora.apply` への行数の渡し方）に触れるので、取り直す。

### §14.6 に照らした暫定の見立て

**有力（20 ステップで 15 秒以上の見込み）。** 改修は 2 ファイル＋テストで小さい。最大の負担は、非公開の内部を使うことによる保守の危険（comfy-kitchen の版を上げるたびに確認が要る）。版の固定と、記号の存在を見る検査で抑えられる。4 ステップの Turbo でも約 4〜6 秒。

---

## ② ステップ間の cyclic prefetch

### 確認済みの事実

- VRAM 16GB mode の常駐数は `min(n_load, 4)`（`engine_h3/lowvram.py:75-86`、`VRAM16_BLOCKS_ON_GPU = 4`、`constants.py:62`）。受け皿の輪は `min(50, 4 + 2)`＝6 本（`block_swap_prefetch.py:405`）。H3 は `hold_arenas=True`（`pipeline_h3.py:821`）。実際の記録にも「pinned staging pool = 2 x 371.1 MB」「ring of 6 x 371.1 MB (held)」「window 4」とある（保管庫 `gates/runs/G5a_h8_r4096_b4/stdout.log`）。
- 窓は輪にならない: `_window(idx) = range(idx, min(idx + 4, 50))`（`block_swap_prefetch.py:178-185`）。窓はパスの末尾で縮み、ブロック 46 で 49 を読み込んだのが最後の発行になる。`_CYCLIC = False` は定義だけで、どこからも参照されていない（160 行）。
- 次のパス（次のステップ）の頭では、まず残っていたブロック（49）を CPU の正本に戻す（`on_block_forward` の 529-540 行）。ブロック 0 は状態に無いので、ミスとして**同期で発行**する（544-546 行。`_issue` は CPU 側の memcpy を同期で行う）。続けて窓の末尾の 1・2 を発行し（551-557 行。1 回に最大 2 本）、そのあとでようやく計算のストリームがブロック 0 の H2D〔ホストから GPU への転送〕を待つ（562 行）。ブロック 0 の計算はその後に積まれる。
- 受け皿の輪に前のステップのブロック 0 の中身は残っていない（輪は 6 本で 50 ブロックを回すので、上書き済み）。記録でも毎パス「50 issued, 1 sync miss(es)」（`G5a_h8_r4096_b4/stdout.log`、`g3_on.log`）。ミスはブロック 0 の 1 回だけ。
- ステップの頭には、もう 1 つ同期点がある: `pipeline_h3.py:1008` の `time_rows(..., device=dev)` の中の `torch.unique(t, …, return_inverse=True)`（266-270 行）。CUDA の `torch.unique` は出力の大きさを CPU に読み戻すので、GPU との同期になる（torch の実装の一般知識。このリポジトリのコードでの確認ではない）。その結果、ステップの頭では GPU の仕事の列が空になり、上の「ブロック 0 の同期の発行」がそのまま GPU の空き時間になる。

### GPU の待ち時間の見積もり（前提つき）

- 前提: ブロック 1 本＝371.1 MiB＝389 MB。CPU 内の pageable→pinned のコピーは 27.1 GB/s、pinned の H2D は 23.4 GB/s（VERIFICATION_LOG §44.3。同じ開発機で、LTX のブロックで測った値）。CPU 側のコピー 1 本 14.4 ms、H2D 1 本 16.6 ms。
- ステップの頭の時間線（ブロック 0 の発行から）:
  - コピー(0) が 0〜14.4 ms。H2D(0) が 14.4〜31.0 ms。
  - コピー(1) が 14.4〜28.7 ms。
  - 発行(2) は pinned の区画 A の空きを待ち（H2D(0) が終わる 31.0 ms まで）、コピー(2) が 31.0〜45.4 ms。
  - その後に S2 の待ちとブロック 0 の計算が積まれる。
- **GPU の空きは 1 ステップあたり約 45 ms**（＋前処理の数 ms）。LTX の §44.3 の見積もり「パス先頭の約 46 ms」とも合う。
- 1 ステップ目の頭は前のステップが無いので、cyclic でも消せない。20 ステップで消せる境目は 19 回。
- **短縮の上限: 20 ステップで約 0.86 秒（19 × 45 ms）、4 ステップで約 0.14 秒（3 × 45 ms）。** CPU 側のコピーが 2 倍遅くても約 1.7 秒。
- 参考: ブロックの計算は 1 本約 440 ms（22 秒 ÷ 50）。ステップの途中の切り替えでは、H2D はブロック 3 本分の余裕をもって終わるので、待ちは生じない（`on_block_forward` と S1b の順序からの推論。③ の節の観測とも合う）。

### 実装の形の案

- **共有部品の変更が必須。** H3 のパイプラインから次ステップの先読みを依頼しても、次のパスの頭の「残りを全部戻す」処理（`block_swap_prefetch.py:539-540`）が、先読みしたブロック 0〜2 を消してしまう。依頼の入口（エンジンは `BlockSwapService._prefetch_engine` という非公開の属性）も無い。
- 最小の形:
  - `PrefetchEngine` に既定で切れた設定（例: `cyclic_head`）を足し、H3 だけがそれを入れる。
  - 入れたときは、窓が末尾にかかったら（idx ≥ 47）、`j mod 50` で 0〜2 を発行する。
  - パスの頭の解放は「新しい窓に入っていないものだけ」にする。
  - 窓が折り返しても常駐は最大 5 本で、輪の 6 本に収まる（idx 47 で {46 は解放待ち・47・48・49・0} など）。
  - 最後のステップで無駄に先読みした 0〜2 は、`teardown()` の S4〔転送の流れの同期〕と解放で片づく（既存の流れ）。
- モジュールの docstring（118-120 行）が求める「cyclic にするならノイズ除去の最後に明示的な release_all が必須」は、H3 では `_denoise` の `finally` の `teardown_prefetch` が満たす。
- LTX への影響: 設定が既定で切れていれば振る舞いは変わらない。ただし共有のコードが変わるので、`block_swap_prefetch_selfcheck.py`（GPU）と LTX の先読み on の指紋の取り直しが要る。
- 設計正本 §6「共有部品は変更しない」と衝突する。

### 計測の設計

後述の共通の計測台の「(b) ステップの境目の GPU の空き」で、`on_block_forward(0)` の前後の CUDA イベントの差を各ステップで測る。期待値は約 45 ms。100 ms を大きく超えるなら見積もりの前提（CPU 側のコピーの速さ）が崩れているので、見立てを見直す。

### 合格の門との噛み合わせ

- `g2_block_equal.py` は block swap を通らないので対象外。
- `g3.sh` は先読み on・常駐 4 で走るので（`g3_on.log` に「BlockSwap prefetch pass」の行がある）、台本の変更なしで指紋 `0da6111c…` の門に乗る。
- 加えて、製品の形（I2V 896×1152×124・sage・決定性 on・3 ステップ）で、変更前と変更後の mp4 の sha256 の一致を見る。これは新しい比較だが、台本は `measure_h3_vram.py` の出力をそのまま使える。
- LTX は上記の selfcheck と指紋を取り直す。

### §14.6 に照らした暫定の見立て

**見送り（20 ステップで約 0.9 秒＝「1〜2 秒は原則として見送る」）。** 共有部品に触る負担と、輪の再利用の危険（先読みの途中のブロックが再利用された区画を指すと、出力が黙って変わる。docstring 112-116 行の記録）に見合わない。Turbo の 4 ステップでは約 0.14 秒。

---

## ③ CPU の正本から pinned staging への先読み

### 確認済みの事実

- CPU の正本から pinned の区画へのコピーは、`PrefetchEngine._issue`（`block_swap_prefetch.py:570`）の 581-582 行（`pinned[off:off + nbytes].copy_(src_u8)` をテンソルごとに）で行う。
- 呼ばれる場所は `on_block_forward`（517 行）で、これはブロックの forward の包み（`block_swap_service.py:355-361`）の先頭から呼ばれる。つまり**計算を GPU に積むのと同じ Python のスレッドで、同期で**走る。
- 区画は 2 つ（`_NUM_STAGING_SLOTS = 2`。147 行）で、使い回す前に、その区画を読んでいた H2D の完了を CPU が待つ（`_acquire_slot` の 615-624 行。`evt.synchronize()`）。
- ブロックの forward の中に、CPU と GPU を同期させる呼び出しは見当たらない。
  - 読んだ範囲: `lowvram.block_forward`・`dit._modulate`／`_gate`／`_attention`・`H3Int8Linear.forward(_rows)`・comfy-kitchen の `int8_linear`（DLPack を `stream=-1` で渡して同期しない。`backends/cuda/__init__.py:543-560`）・`lora.apply`。
  - SageAttention 2.2.0 の Python 部分にも `.item()`／`synchronize`／`.cpu()` は無い（grep で確認）。
  - 同期点は、ステップの頭の `torch.unique`（② の節）と、計測台の progress の割り込みの `torch.cuda.synchronize()` だけ。

### 観測（既存の記録の読み直し）

- 保管庫 `gates/runs/G5a_h8_r4096_b4/dmon.txt`（nvidia-smi dmon。1 秒刻み）: ノイズ除去の間、SM の使用率はステップの境目（計測の割り込み 1.41 秒を含む）以外ずっと 100%。VERIFICATION_LOG §169.7 の 7 も「GPU 使用率 99%」と書いている。
- 同じ回の「265.9 ms waiting on pinned slots」（1 パスの合計）。CPU が区画の空きを待つ時間は短い。
- 解釈（推測）: CPU は GPU より数ブロック先を走っており、1 ブロックあたり約 14 ms の CPU 側のコピーは、GPU に積まれた仕事の陰に隠れている。区画の待ちが短いのに CPU が先行しすぎないのは、CUDA の起動の列が満杯になって起動の呼び出しで止まっているから、と見られる（未確認。計測 M1 の profiler で `cudaLaunchKernel` の CPU 側の長さを見れば分かる）。
- dmon は 1 秒刻みなので、数 ms の空きは見えない。ステップの途中の空きを 0 と断定するのは、M1 で確かめてから。

### 見積もり

- ステップの途中: GPU の待ちは約 0（上の観測による）。
- ステップの頭: ② の約 45 ms のうち、CPU 側のコピーが道筋に乗っているのは、コピー(0)・(1)・(2) の約 43 ms。これを別のスレッドで前のステップのうちに済ませても、H2D(0) の約 17 ms は残る。取れるのは 1 ステップ約 29 ms。
- **上限: 20 ステップで約 0.55 秒（19 × 29 ms）、4 ステップで約 0.09 秒。** しかもこれは ② の約 45 ms の一部で、② と足し合わせられない。

### 実装の形の案と危険

- 案: `_issue` を「区画を取ってコピーする段」と「H2D を積む段」に分ける。前者を専用のスレッドで次のブロックぶん先に走らせ、後者は計算のスレッドで、前者の完了（`threading.Event`）を待ってから積む。
- 危険:
  1. 区画の WAR〔読み終わる前に上書きする〕: スレッドも `_acquire_slot` の待ちを守る必要がある。区画は 2 つしか無いので、先行できる量も小さい。
  2. スレッド安全性: `_state`・`_next_slot`・`_slot_event` は今は単一スレッドを前提にしており、ロックが要る。
  3. torch の CPU コピーは内部で OpenMP の並列を使うので、計算のスレッドの起動と CPU を取り合う。
  4. ステップの頭で効かせるには、結局 ② と同じ折り返しの発行が要る。
- いずれにせよ**共有部品の変更**で、LTX も同じコードを通る。

### 計測の設計

後述の共通の計測台の「(c) ブロック切り替えの GPU の空き」（idx ≥ 1 の合計）と、`_issue` の中の CPU 側のコピーの時間（`time.perf_counter`）、およびコピーを始めた時点で計算のストリームがまだ動いていたか（`torch.cuda.current_stream().query()` が偽か。同期しない）を記録する。動いていたなら、そのコピーは隠れている。

### §14.6 に照らした暫定の見立て

**見送り（20 ステップで約 0.55 秒以下、しかも ② と重複）。** Turbo の 4 ステップでは約 0.09 秒。

---

## 共通の計測の設計（監督が GPU の使用の了承を取ってから回す）

**置き場と台本**: 新しいフォルダ `outputs/h3-speedup-2026-10-24/` に、`outputs/h3-vram-2026-10-18/measure_h3_vram.py` の写しを置き、フラグを足す（元の台本は変えない）。製品のコードは変えず、計測台の中の差し替え（monkeypatch）だけで行う。この作法は前回と同じ。

**条件**: I2V 896×1152×124・sage・VRAM 16GB mode on（`--vram16`）・常駐 on（`--keep-resident`）・決定性 on（`--deterministic`）・先読み on・`--blocks-on-gpu 8`（on で 4 になる）・3 ステップ・seed 42・画像とプロンプトは `g5a.sh` と同じ。腕の間の冷まし方は `arm.sh` と同じ。

**腕 E（イベント。軽い。壁時計にほぼ影響しない）**

- `PrefetchEngine.on_block_forward` を包み、呼ぶ前と後に計算のストリームへ `torch.cuda.Event(enable_timing=True)` を記録する（`pre[i]`・`post[i]`）。
  - 計算のストリーム上で両者の間にあるのは S2 の待ちだけ。だから `elapsed(pre[i], post[i])` は「ブロック i に入るときに GPU が止まっていた時間」になる。S2 の待ちと、CPU が遅れて GPU の列が空になった時間の両方を含む。
- `pipeline_h3.time_rows` を包み、入口で同期の後にイベント `head` を記録する。
  - `elapsed(head, post[0])` が**(b) ステップの境目の GPU の空き**になる。計測台の割り込み（同期とメモリの採取）を含まない。
- (c) は Σ_{i≥1} `elapsed(pre[i], post[i])`。
- `_issue`／`_acquire_slot` も包み、CPU 側のコピーの時間と区画の待ちの時間を記録する。あわせて、コピーを始めた時点の `current_stream().query()` の値を残す。
- 集計は progress の割り込み（もともと `synchronize` している）の中で行う。

**腕 P（torch.profiler。重い。帰属を見るため）**

- 割り込みの index 1 で `torch.profiler.profile(activities=[CPU, CUDA], record_shapes=False, with_stack=False)` を始め、index 2 で止める。2 ステップ目だけを記録する。
- `record_function` の名札を付ける: `swap.on_block_forward[i]`・`swap.issue[i]`・`swap.acquire_slot`・`step.time_rows`・`dit.forward`・`qkv.forward_rows`。
- 出力は `trace_step2.json`（Chrome trace）と、核の名前ごとの回数と合計時間の CSV。
- **(a) 量子化の合計**: ConvRot の量子化の核（名前は初回の trace で確かめる。"convrot" や "quant" を含むはず。未確認）の回数と合計時間。期待する回数は 1 ステップで qkv 1,050＋out_proj 50＋fc1 500＋fc2 500。qkv の分 × 20/21 が ① の短縮の上限の実測値になる。
- **(b)／(c) の検算**: 計算のストリームの核の時間の隙間を並べ、直前の CPU の名札で境目か切り替えかに振り分ける。
- profiler の負担で核の時間はやや伸びるので、秒数そのものは腕 E と腕 N の値を正とする。

**腕 N（何も足さない基準）**: 前回の `G5a_h8_r4096_b4` と同じ条件の素の回。2・3 ステップ目の秒数を、腕 E と比べて計測の負担を確かめる。

**M0**（① の節）: 量子化 1 回の単体のマイクロベンチ（数分）。

**判定の使い方**: (a) が見積もり（1 ステップ 0.95〜1.6 秒）に入れば ① を進める。(b) が約 45 ms、(c) がほぼ 0 なら、②・③ の見送りを確定する。

---

## 監督が判断に使う一覧表

| 件 | 期待短縮の上限（20 ステップ） | 同（Turbo 4 ステップ） | 触るファイル | 共有部品への影響 | 数値等価の根拠 | 不確かさ |
|---|---|---|---|---|---|---|
| ① QKV の量子化の共有 | **約 19〜32 秒**（1 ステップ 0.95〜1.6 秒） | 約 3.8〜6.4 秒 | `engine_h3/quant_linear.py`・`engine_h3/lowvram.py`・`tests/test_h3_engine.py` | なし（ただし comfy-kitchen 0.2.37 の非公開の内部を使う） | 同じ核・同じ入力・同じ引数で 1 回量子化し、結果を再利用するだけ。行列積の核と形も現行と同じ。量子化の核が決定的であることは状況証拠（A-G2・GPU テスト）による | 1 回の時間（下限は帯域、上限は A-G5 からの逆算）。非公開 API の版の間での安定性は未確認。VRAM は案 (a) で +約 204 MiB、案 (b) で −約 204 MiB |
| ② ステップ間の cyclic prefetch | 約 0.86 秒（19 境目 × 約 45 ms） | 約 0.14 秒 | `engine/transformer/block_swap_prefetch.py`（必須）・`engine/transformer/block_swap_service.py`（設定の受け渡し）・`engine_h3/pipeline_h3.py`（設定を入れる） | **あり**（LTX と共有。既定で切った設定にしても selfcheck と指紋の取り直しが要る） | 転送の時機だけを変え、計算は変えない。危険は区画の再利用の誤り（黙って出力が変わる） | CPU 側のコピーの速さは LTX で測った値の流用。ステップの境目の空きは見積もりで未実測 |
| ③ CPU→pinned の先読み | 約 0.55 秒以下（② と重複） | 約 0.09 秒 | `engine/transformer/block_swap_prefetch.py`（スレッド化） | **あり**（同上。スレッド安全性の作り直し） | 転送の準備の時機だけを変える。危険は区画の競合とスレッドの競合 | ステップの途中の空き＝0 は、dmon（1 秒刻み）と区画の待ちの短さからの推論で未実測 |

---

## 調査中に感じた「設計の前提が崩れている」点

1. **③ の前提（オーナーのメモの「pack／memcpy は同期処理なので GPU が待つ」）は、ステップの途中では成り立っていない見込み。** memcpy は確かに同期だが、GPU はほぼ 100% 働いていて、CPU が先を走っている。GPU が待つのはステップの頭だけで、それは ② と同じ約 45 ms。
2. **設計正本 §6.5 末尾の「qkv の量子化 21 回は 1 ステップ約 22 秒に収まっている」は、影響を小さく書きすぎている可能性がある。** 見積もりでは 1 ステップの 4〜7%（約 1〜1.6 秒）。A-G5 のヘッド群 28 の速さ（約 5.5%）の大部分がこれで説明できそうで、そうなら **① の後ではヘッド群の拡大（§3-245）の得が小さくなる**。§3-245 の再評価の前提が変わりうる。
3. **台帳 §1-105 の前提「comfy-kitchen が量子化済みの入力の行列積を公開 API で受け付けるか」の答えは「受け付けない」。** ① は非公開の内部（`backends/cuda` の `quantize_int8_rowwise_convrot64`、拡張の `_C.cutlass_int8_dequant` ほか）を使う形になり、設計正本 §5 の「関数 API の `int8_linear` に任せる（同じ CUDA の核に届くので素のテンソルで足りる）」と、§6.5 の「`lowvram.py` は新しい計算を書かない」から外れる。オーナーの裁定が要る。
4. **② と ③ は、設計正本 §6 の「共有部品は変更しない」と衝突する。** H3 のパイプラインだけでは実現できない（次のパスの頭の解放が先読みを消すため）。
5. 小さな事実: ステップの頭の `time_rows` の `torch.unique`（CUDA）は、毎ステップ CPU と GPU を同期させる（torch の実装の一般知識。コードでは未確認）。ステップの境目の空きはこれと「ブロック 0 の同期の発行」の組み合わせで生じる。
