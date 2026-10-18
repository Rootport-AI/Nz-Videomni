# R2: comfy-kitchen の int8_linear と注意計算の作業メモリ（調査報告）

調査日 2026-10-18。読み取り専用。コードは実行していない（torch も comfy_kitchen も import していない）。
以下で「事実」はソースの行か公式の記述に基づく。「推測」はそこからの計算や推論である。
バイト数は 10 進の GB（10^9）で書く。N はトークン数（行数）、K は入力幅、M は出力幅。

## 結論

1. **int8_linear 1 回で新しく確保されるメモリは、N・K + 4N + 2・N・M バイトである。** 内訳は、int8 に量子化した活性化 `[N,K]`、行ごとのスケール（f32 `[N,1]`）、bf16 の出力 `[N,M]` の 3 つ。H3 の DiT の 4 種の層（K＝5376／7168／14336）は、どれも回転と量子化を 1 つのカーネルで行う経路に入る。このため、回転後の活性化を bf16 や f32 の `[N,K]` として丸ごと作ることはない。行列積は CUTLASS〔NVIDIA の行列積ライブラリ〕のカーネルで、逆量子化を出力の書き込みに融合している。int32 や f32 の `[N,M]` の中間結果は作られず、bf16 が直接書かれる。
   - 支配的なのは出力 2・N・M である。fc1（M＝28672）は N＝37,000 で 2.12 GB、N＝100,000 で 5.73 GB。量子化の写し N・K は、最大の fc2（K＝14336）でも 0.53 GB／1.43 GB にとどまる。
   - このほかに CUTLASS の stream-K〔1 つの出力タイルの計算を複数のブロックで分け合う方式〕の作業領域がある。PyTorch のアロケータの外で `cudaMalloc` される。大きさは SM〔GPU の演算ユニット〕の数に比例し、N には比例しない（推測：数 MB〜十数 MB）。
2. **行（N）方向のチャンク化は、0.2.37 にも 0.2.38 にも無い。** 分けたいなら呼び出し側で分けるしかない。
3. **int32 の限界**: CUDA 経路の量子化カーネルと CUTLASS の出力の書き込みは、位置を int64 で計算している。issue #136（triton 経路）の限界は CUDA 経路には無い、という本製品の理解はソースで裏付けられた。1 点、出力の「バッチ方向の刻み」だけが int の掛け算になっている。ただしバッチの番号は常に 0 なので、実害は無いと推測する。
4. **SDPA〔PyTorch 標準の注意計算〕**: 本製品の torch 2.13.0+cu130（Windows 版）は **flash attention を含まずにビルドされている**（DLL の中の文字列で確認）。Ada（sm_89）と Ampere（sm_86）では、既定の優先順は flash → mem-efficient → math → cuDNN である。flash が無いので、**mem-efficient〔メモリ節約型の注意計算〕が選ばれる**。math〔素直な実装〕に落ちる条件には当てはまらない。mem-efficient の作業メモリは出力 `[N,56,128]` bf16（14,336・N バイト）だけで、N に比例する。勾配を取らないので logsumexp〔softmax の正規化項〕は確保されない。head_dim 128 なので f32 の累積用の領域も要らない。
   - **重要な発見（推測、ソースを読んだ水準）**: mem-efficient のカーネルは、キーと値の行の位置を **int32 の掛け算**で計算している（`iter_key_start * v_strideM`）。本製品の v は qkv の出力の view〔値を写さない見え方の変更〕なので、行の刻みが 21,504 要素ある。このため **N が約 99,968 を超えるとこの掛け算があふれる**。調べている範囲（N≈37,000〜100,000）の上端にあたる。
5. **SageAttention**: 作業メモリは N に比例し、N² のものは無い。ただし本製品の呼び出し方では、`.contiguous()` による q・k・v の bf16 の写し 3 本などが加わる。追加分は合計で SDPA の約 6.5 倍（約 93,000・N バイト）になる。長い系列での既知の限界として、第三者の解説が「ヘッド数×系列長×ヘッド次元 ≥ 2^31 で不正なメモリアクセス」と報告している。H3 では N ≈ 299,600 にあたり、今回の範囲よりずっと上である。

---

## 1. int8_linear の CUDA 経路の中身

### 1-1. 呼び出し側が渡しているもの（事実）

- `engine_h3/quant_linear.py:101-121` の `H3Int8Linear.forward` は次のように呼ぶ。活性化を `[N,K]` の 2 次元の bf16 にして `.contiguous()` する（すでに連続なら写さない）。重みは int8 `[M,K]`、スケールは f32 `[M,1]`（`:88-92`）、bias は無し、`out_dtype=bf16`、`convrot=True`、groupsize は層ごとの記述から取る。
- DiT の量子化層は 1 ブロックに 4 本ある（`engine_h3/dit.py:205-262`、隠れ幅 5376、56 ヘッド×128 次元、ffn 14336 は `h3_format.py:147-152`）。

| 層 | 重みの形 [M,K] | 入力 |
|---|---|---|
| attn.qkv_proj | [21504, 5376] | 変調後の h |
| attn.out_proj | [5376, 7168] | 注意計算の出力を reshape したもの |
| mlp.fc1 | [28672, 5376] | 変調後の h |
| mlp.fc2 | [5376, 14336] | silu(gate)·value |

- 依頼文にあった「FFN は [28672,5376] と [5376,28672]」は誤りで、fc2 は [5376, **14336**] である。fc1 の出力は [gate; value] の 2 等分で、SwiGLU〔ゲート付きの活性化〕は呼び出し側が PyTorch で計算している（`dit.py:249-250`）。comfy-kitchen の `input_act="swiglu"` 融合は使っていない。
- 「AdaLN 射影だけ groupsize 64」は非枝刈り版の話である（`Docs/Outputs-archive/h3-stage1-reports-2026-10-17/R1_stack.md:108`）。本製品が扱う枝刈り版では、AdaLN の射影は f32 の普通の Linear で、int8 ではない（`dit.py:265-291`）。

### 1-2. 経路の選ばれ方（事実。行番号はインストール済みの `.venv-engine-h3\Lib\site-packages\comfy_kitchen\backends\cuda\__init__.py`）

1. `int8_linear`（`:2035`）は、まず融合 ConvRot の可否を判定する（`:2061-2067`）。条件は、groupsize＝256、K が 256 の倍数、256 ≤ K ≤ 16384（`_CONVROT_FUSED_MAX_K`、`:1631`）、共有メモリの必要量 (K+8192)×4 バイトが上限未満（`:418-458`）の 4 つ。
   - 推測（計算）: 共有メモリの必要量は、K＝5376 で 54,272 B、7168 で 61,440 B、14336 で 90,112 B。sm_86／sm_89 の上限 101,376 B に収まるので、**4 層とも融合経路に入る**。
   - 補足: 0.2.37 の `_should_use_convrot_fused_kernel`（`:469-476`）には「5120 < K < 8192 を除く」という帯がある。しかしこれは融合経路に入らなかった場合の次の候補（`:2150`）でしか使われない。K＝5376／7168 は先の判定（`:2133`）で融合経路に入るので、この帯は関係しない。
2. 融合経路（`:2133-2147`）では、`x_qdata = torch.empty((m,k), int8)`（`:2134`）と `x_scale = torch.empty((m,1), f32)`（`:2135`）を確保し、`_C.quantize_int8_rowwise_convrot64` を 1 回呼ぶ。
   - カーネル（上流 v0.2.37 `backends/cuda/ops/int8_linear.cu:1205-1240, 1609-1700`）は 1 行を 1 ブロックで処理する。行全体を f32 で**共有メモリ**に置き（`row_buf = smem`、`tmp = smem + K`）、256 ずつアダマール回転〔直交変換で外れ値を均す〕をかける。そのうえで行の absmax〔絶対値の最大〕から行ごとのスケールを求め、int8 を書く。**回転後の活性化はグローバルメモリに出ない。**
3. M ≥ 3 なので、GEMV〔行列とベクトルの積〕の特別経路（`:2169`）には入らない。`out = torch.empty((m,n), out_dtype)`（`:2190`）を確保する。`weight_scale` は f32・連続・同じデバイスなので、そのまま使う（`:528-531`）。`ws_cutlass` も numel＝M なので写さない（`:2223`）。
4. sm_80 以上では `_C.cutlass_int8_dequant`（`:2246`）を呼ぶ。上流 `cutlass_gemm_int8.cu:1-14, 60-118` によれば、これは int8×int8→int32 の累積をレジスタ内で行う。そのうえで、エピローグ〔行列積の最後の書き出し段〕で `acc × x_scale[m] × w_scale[n] + bias[n]` を計算し、**出力の dtype（bf16）で直接書く**。int32 や f32 の `[N,M]` は作らない。
5. CUTLASS がどの設定でも起動できなかったときだけ、cuBLAS の int8 行列積に落ちる（`:2259-2296`）。このときは `out_int32 = torch.empty((m,n), int32)`（`:2269`）、つまり **4・N・M バイト**が追加される。
   - 推測: CUTLASS の起動は、選んだ設定が失敗しても予備の一覧（`cutlass_gemm_int8.cu:240-253`。作業領域が要らない 64×128 の通常タイルを含む）を順に試す。このため、sm_86／89 で cuBLAS まで落ちることは事実上無いと考える。作業領域の `cudaMalloc` が失敗した場合も、通常タイルで成功するはずである。

### 1-3. 問い 1 の (a)〜(e)

- (a) 回転後の活性化を `[N,K]` で丸ごと作るか → **作らない**（融合経路）。ただし融合経路に入らない形では作る。K > 16384 の層や groupsize 64 の層がそれにあたり、本製品ではテキストエンコーダの down_proj（K＝25600）が該当する（R1_stack.md の記述）。その経路（`:2156` → `:1689-1696` → `tensor/int8_utils.py:58-72`）は、`torch.matmul` で **x と同じ dtype（bf16）の `[N,K]`** を作る。
- (b) int8 の `[N,K]` の写し → **作る**。N・K バイト。
- (c) スケールの形 → f32 `[N,1]`、4N バイト（行ごと。グループごとではない）。
- (d) 出力 → **bf16 の `[N,M]` を直接書く**。2・N・M バイト。f32 を経由しない。
- (e) 生存区間 → `x_qdata` と `x_scale` は関数の終わりで解放される。`out` は戻り値として呼び出し側に渡る。量子化の時点では、入力 x（呼び出し側の持ち物）と `x_qdata` が生きている。行列積の時点では、x・`x_qdata`・`out` が同時に生きる（x は呼び出し側が参照を持つため）。

**1 回の呼び出しの式（融合経路、本製品の 4 層）**

```
新たに確保する量 = N·K（int8 の活性化） + 4·N（スケール） + 2·N·M（bf16 の出力） + W_sk
同時に生きる最大 = 入力 2·N·K（bf16。呼び出し側） + N·K + 4·N + 2·N·M + W_sk
W_sk: CUTLASS stream-K の作業領域。PyTorch の外で cudaMalloc し、ストリームごとに 1 本を使い回す。
      N に比例しない（推測：SM 数 × タイル 1 枚ぶん程度、数 MB〜十数 MB）
```

融合されない経路（K > 16384 など）では、これに 2・N・K（回転後の bf16）が量子化の間だけ加わる。cuBLAS に落ちた場合は、行列積の間だけ 4・N・M（int32）が加わる。

**数値（推測・計算）**

| 層 | N·K（int8） N=37,000 / 100,000 | 2·N·M（出力） N=37,000 / 100,000 |
|---|---|---|
| qkv_proj | 0.20 / 0.54 GB | 1.59 / 4.30 GB |
| out_proj | 0.27 / 0.72 GB | 0.40 / 1.08 GB |
| fc1 | 0.20 / 0.54 GB | 2.12 / 5.73 GB |
| fc2 | 0.53 / 1.43 GB | 0.40 / 1.08 GB |

### 1-4. 参考：呼び出し側（本製品）の生存区間（事実はコード、量は推測・計算）

int8_linear の外側にあるため本題ではないが、仮説を立てる材料として書いておく。

- MLP（`dit.py:249-250`）: `gate, value = self.fc1(x).chunk(2)` で、fc1 の出力 57,344・N バイトは、MLP の forward が終わるまで生きる。`F.silu(gate) * value` は silu の一時（28,672・N）と積（28,672・N）を作る。このため、**掛け算の瞬間は fc1 の出力＋silu＋積 ＝ 114,688・N バイト**（N＝37,000 で 4.24 GB、100,000 で 11.47 GB）になる。これに、MLP の入力 h（10,752・N）と残差 x などが加わる。fc2 の呼び出し中は、fc1 の出力＋積＋int8 の写し＋出力 ＝ 111,104・N バイト。
- 注意（`dit.py:226-236`）: qkv の出力 43,008・N バイトは、v がその view なので注意計算の間ずっと生きる。q と k は RMSNorm と RoPE の後の新しいテンソルで、それぞれ 14,336・N。
- 推測: ブロック内でいちばん高い山は MLP の SwiGLU のところで、int8_linear の中ではない。

## 2. 内部のチャンク化

- 事実: 0.2.37 の `int8_linear`（`:2035-2296`）に、行（N）方向を分けて処理する仕組みは無い。量子化カーネルは「1 行に 1 ブロック」のグリッドを 1 回起動するだけである（上流 `int8_linear.cu:1658`）。
- 事実: 同じファイルには `COMFY_KITCHEN_INT4_INT8_WEIGHT_CHUNK_N` などのチャンク化の設定があるが（`:243-245`）、これは W4A8／INT4 の重みを列方向に分けるもので、int8_linear には関係しない。
- 事実: 0.2.38 で int8_linear に入った変更は、M＝1 の GEMV に残差を融合したことなどで、大きな N 向けの分割は無い（v0.2.37…main の差分）。
- したがって、分けたいなら呼び出し側で行を区切って呼ぶしかない（設計の提案ではなく、事実の帰結として書く）。

## 3. CUDA 経路の int32 の限界

- 事実: 量子化カーネルは行の先頭位置を int64 で計算している（上流 v0.2.37 `int8_linear.cu:1229` の `const int64_t row_offset = static_cast<int64_t>(row) * K;`）。グリッドの大きさは `num_rows`（x 方向。上限は 2^31−1）。
- 事実: CUTLASS の出力の書き込みは `VisitorAuxStore<..., cute::Stride<int64_t, _1, int64_t>>`（`cutlass_gemm_int8.cu:96`）で、行の刻みは int64 として扱われる。M・N・K は `launch_cutlass_int8_dequant(..., int64_t M, int64_t N, int64_t K, ...)` で受け、内部では int に変換している（個々の値は 2^31 未満なので問題ない）。
- 事実（残っている int の掛け算）: 本製品が通る残差なしの経路 `run_strided`（`cutlass_gemm_int8.cu:116`）は、出力の 3 番目の刻み（バッチ方向）を `M * output_stride` として **int どうしの掛け算**で作っている。fc1 では N ≥ 74,899 であふれる。
  - 推測: この刻みにかかるバッチの番号は常に 0（バッチ数 1）なので、計算される位置は変わらず、実害は無いと考える。厳密には C++ の未定義動作である。残差ありの版（`:183-184`）は `int64_t(M) * N` で正しく書かれている。
- 事実: issue #136 は 2026-10-18 時点でも open、コメントは 0 件（GitHub API）。
- 結論: 本製品の「CUDA 経路は #136 の限界を持たない」という理解は、ソースを読んだ水準で裏付けられた。

## 4. 上流の動向（0.2.37 より後）

- v0.2.38 のタグが 2026-10-17 に出ている（`42081bf9`）。v0.2.37 からのコミットは 9 件。int8 に関係する変更は次のとおり。
  - **CUTLASS の作業領域の確保方法が変わった**（`cutlass_gemm_common.cuh`）。0.2.37 は「ストリームごとに 1 本を `cudaMalloc` し、プロセスが終わるまで保持（大きい要求が来たら作り直す）」だった。0.2.38 は「起動のたびに `cudaMallocAsync`、終わったら `cudaFreeAsync`」になった。どちらも PyTorch のキャッシュアロケータの外である。
  - #251 で SiTU 活性化の融合が追加された。M＝1 の GEMV にも入力活性化と残差を融合した。
  - 大きな N 向けのチャンク化、活性化の量子化のメモリ削減、ConvRot の扱いの変更は**無い**。
- 関係しそうな未マージの PR と issue:
  - #219「int8 GEMM に indexed gate と残差を融合」、#221「RMSNorm＋変調＋ConvRot 量子化の融合」、#222「SM120 の H3 の QKV/RMS/RoPE の融合」。どれも MiniMax H3 向けで、SM120（Blackwell）に限られる。中間テンソル（norm や変調の bf16）の実体化を無くすもので、トークン数 32,700〜90,461 で計測している。https://github.com/Comfy-Org/comfy-kitchen/pull/219 ・ /221 ・ /222
  - #237「sm86 の int8 GEMM のタイル選択の調整」。H3 の形を含む 45 形で計測し、大きな M では stream-K の設定 13 が最良としている。https://github.com/Comfy-Org/comfy-kitchen/pull/237
  - #196（open）「Windows/NVIDIA で int8 ConvRot のモデルを読み直すと詰まる。VRAM が高止まりし、int8_linear の一時バッファの確保で OOM することがある」。16 GB のカード、torch 2.13.0+cu130、comfy-kitchen 0.2.35 での報告。https://github.com/Comfy-Org/comfy-kitchen/issues/196
  - #136（open）は前節のとおり。

## 5. SDPA のカーネル選択

### 5-1. 選ばれるカーネル（事実）

- この torch のビルドには flash attention が無い。
  - `.venv-engine-h3\Lib\site-packages\torch\lib\torch_cuda.dll` に `USE_FLASH_ATTENTION was not enabled for build.` と `Torch was not compiled with flash attention.` の文字列がある。後者は `sdp_utils.cpp:920-925` の `#ifndef USE_FLASH_ATTENTION` の枝にだけある文である。
  - flash のカーネル名（`flash_fwd_kernel`）は 0 件、mem-efficient のカーネル名（`fmha_cutlassF_*`、例 `fmha_cutlassF_bf16_aligned_64x128_rf_sm80`）はある。
  - ソースは torch の git_version `cf30153c` の `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`。
  - 実行時に確かめるなら `torch.backends.cuda.is_flash_attention_available()` を使う（今回は実行していない）。
- 既定の優先順は flash → efficient → math → cuDNN（`aten/src/ATen/Context.h:480-485`）。cuDNN を先頭にするのは sm_90／sm_100 系だけで（`sdp_utils.cpp:79-108`）、sm_86／89 には当てはまらない。cuDNN は math より後なので、ここでは選ばれない。
- mem-efficient の条件（`sdp_utils.cpp:972-1040`）: sm_50〜sm_121、sm_80 以上なら bf16 可、head_dim と 8 の倍数の揃え、最後の次元の刻みが 1、マスクなしで可。本製品の q・k・v（`dit.py:191-199`。`[1,56,S,128]` の転置した view）はこれを満たすと読める。
  - 推測: したがって **mem-efficient が選ばれ、math には落ちない**。
- 事実: 起動するカーネルは、bf16・sm80 系の 3 種（64×64 rf、64×128 rf、32×128 gmem。`mem_eff_attention/kernels/cutlassF.h:26-30`）から、head_dim ≤ kMaxK で共有メモリが収まる最初のものである（`attention.cu:1674-1715`）。
  - 推測: head_dim 128 なので 64×128 rf（kMaxK＝128）が選ばれる。

### 5-2. 作業メモリ（事実＋計算）

- 出力 `res = at::empty({B,M,H,Kv})`（bf16、`attention.cu:1717-1720`）＝ 2・N・56・128 ＝ **14,336・N バイト**（N＝37,000 で 0.53 GB、100,000 で 1.43 GB）。
- logsumexp は、勾配が要るときだけ N ぶん確保される（`attention.cpp:634-638`、`attention.cu:1724-1728`）。本製品は `no_grad` なので大きさは 0。
- 累積用の f32 の領域（`output_accum`、4・N・H・D）は `kNeedsOutputAccumulatorBuffer` のときだけ確保される（`attention.cu:1737-1745`）。これは「1 回で値を処理しきれない」カーネルの場合で（`kernel_forward.h:112, 117-119`）、64×128 rf（kMaxK 128 ≤ kKeysPerBlock 128）では確保されない。
  - 推測: 共有メモリが足りず 32×128 gmem に落ちた場合だけ、+28,672・N バイトになる。
- 出力は `[B,S,H,D]` の並びで返り、本製品の `out.squeeze(0).transpose(0,1)` で `[S,H,D]` の連続になる。このため、その後の reshape で写しは生じない（推測。ストライドを追った結果）。
- **N² の領域は無い。** 作業メモリは N に比例する。
- 参考（math に落ちた場合）: スコアを `[1,56,N,N]` で実体化する。bf16 なら N＝37,000 で約 153 GB、f32 に上げるなら約 307 GB で、どの GPU でも即座に溢れる（推測・計算）。

### 5-3. 長い系列での int32 のあふれ（推測。ソースを読んだ水準）

- `kernel_forward.h`（同じ git_version）は、キーと値の位置を `p.key_ptr + iter_key_start * p.k_strideM`（`:764`）と `p.value_ptr + iter_key_start * p.v_strideM`（`:720`）で計算している。`iter_key_start` も `*_strideM` も `int32_t` である（`:166-168, 700`）。クエリ側の `(q_start + query_start) * q_strideM`（`:250`）も同じ形をしている。出力側だけは `int64_t(...)` に変換している（`:255`）。
- 本製品の刻みは次のとおり。
  - q・k は RoPE の後の連続テンソル（`rope.py:175-184` の `torch.cat`）なので、刻みは 56×128＝7,168。あふれるのは N ≈ 299,593 から。
  - **v は qkv の出力の view**（`dit.py:229-230` の `view(s,3,H,D).unbind(1)`）なので、刻みは 3×7,168＝21,504（`attention.cu:1777-1779` の `ASSIGN_CHECK_OVERFLOW` は刻みそのものが int32 に入るかしか見ない）。キーの位置は 128 刻みなので、**N が 99,968 を超えると `iter_key_start × 21,504` が 2^31 を超える**。
- 推測: 超えた場合、値の読み出し位置がおかしくなる。結果は不正なメモリアクセスか、黙って誤った値になる。符号付きのあふれは未定義動作なので、どちらになるかは決まらない。
- GitHub と検索では、これに当たる PyTorch の issue は見つけられなかった。

## 6. SageAttention

### 6-1. 経路（事実。`.venv-engine-h3\Lib\site-packages\sageattention\core.py`、版 2.2.0+cu130torch2.10.0andhigher.post6）

- sm_86 → `sageattn_qk_int8_pv_fp16_cuda(pv_accum_dtype="fp32")`（`:160-161`）。
- sm_89 → `sageattn_qk_int8_pv_fp8_cuda`。CUDA ≥ 12.8 では `pv_accum_dtype="fp32+fp16"`（SageAttention2++）（`:162-168`）。torch.version.cuda＝13.0 なのでこちらになる。
- 本製品の呼び出し（`dit.py:196`）は `qh.contiguous(), kh.contiguous(), vh.contiguous()` を `tensor_layout="HND"` で渡す。

### 6-2. 1 回の作業メモリ（E ＝ N×56×128 要素。量は推測・計算）

| 何か | 大きさ | 根拠 |
|---|---|---|
| q・k・v の `.contiguous()` の写し（`[S,H,D]` を `[1,H,S,D]` に転置した view なので必ず写す） | 3×2E バイト | `dit.py:191-196` |
| k の平均（smooth_k） | 小さい（`[1,H,1,D]`） | `core.py:551` 付近 |
| q_int8・k_int8 | 2E バイト（スケールは小さい） | `quant.py:154-171` |
| 出力 o（bf16） | 2E バイト | `core.py:574` 付近 |
| sm_86: `v.to(fp16)` | 2E バイト | `core.py:584` 付近 |
| sm_89: 転置して 64 にそろえた v（bf16）＋ fp8 の v | 2E＋1E バイト（前者は関数の終わりで解放） | `quant.py:274-287` |
| 戻った後の `transpose` → `reshape` の写し（HND の出力を `[S,H·D]` にするため） | 2E バイト | `dit.py:199, 236` |

- 合計の追加は、sm_89 で約 13E バイト ＝ 約 93,000・N バイト（N＝37,000 で 3.45 GB、100,000 で 9.3 GB）。一度に全部が生きるわけではないが、上限の目安になる。sm_86 もほぼ同じ。
- SDPA（mem-efficient）の追加は 2E バイトなので、**本製品の呼び出し方では Sage のほうが一時メモリが多い**。N² の領域はどちらにも無い。
- 長い系列での既知の問題: 第三者の解説（RunComfy のガイド）が「SageAttention のカーネルは位置を 32 bit で計算するので、ヘッド数×系列長×ヘッド次元 が 2^31 を超えると不正なメモリアクセス（OOM ではない）。全版に共通」と書いている。https://comfyui-guides.runcomfy.com/ultimate-comfyui-how-tos-a-runcomfy-guide/long-video-generation-fails-with-cuda-illegal-memory-access-sage-attention-length-limit
  - 推測・計算: H3 では 2^31/7,168 ≈ **299,593 トークン**で、今回の範囲（≤ 100,000）より十分上である。上流（thu-ml/SageAttention）の issue では裏付けを見つけられなかった。カーネルはビルド済みの `.pyd` で、ソースでは確かめていない。

## 確認できなかった点

- CUTLASS stream-K の作業領域（`Gemm::get_workspace_size`）の正確な大きさ。CUTLASS 本体のソースまでは追っていない。「N に比例しない・SM 数に比例する」は、スウィズル〔ブロックの割り当て方〕の実装（`cutlass_gemm_common.cuh` の `sk_blocks ≤ avail_sms`）から推した値である。
- 本製品の形（M ≥ 37,000）で実際に選ばれる CUTLASS の設定の番号。式（`choose_fused_int8_config`）では設定 13（128×256 の stream-K）になる見込みだが、実行していない。
- インストール済みの `_C.abi3.pyd` が上流のタグ v0.2.37 のソースと同じかどうか。Python 側は一致を前提に読んだ。.cu の行番号は GitHub の v0.2.37 のもの。
- この torch で flash が本当に使えないことの実行時の確認（`is_flash_attention_available()`）。DLL の文字列による判断にとどまる。
- mem-efficient が 64×128 rf を選ぶこと（共有メモリが sm_86／89 の上限に収まること）の確認。
- math 経路が bf16 を f32 に上げて計算するかどうか（量の見積もりにしか影響しない）。
- mem-efficient の int32 のあふれ（§5-3）が実際に不具合として出るか、上流で報告や修正があるか。
- SageAttention の 2^31 の限界の一次資料（カーネルのソースか上流の issue）。

## 参照

- comfy-kitchen: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\.venv-engine-h3\Lib\site-packages\comfy_kitchen\backends\cuda\__init__.py`、`...\comfy_kitchen\tensor\int8_utils.py`、`...\comfy_kitchen\backends\_activations.py`
- 上流の .cu（v0.2.37）: https://github.com/Comfy-Org/comfy-kitchen/tree/v0.2.37/comfy_kitchen/backends/cuda/ops （`int8_linear.cu`・`cutlass_gemm_int8.cu`・`cutlass_gemm_common.cuh`）。差分: https://github.com/Comfy-Org/comfy-kitchen/compare/v0.2.37...main
- torch（git `cf30153c4c131c8164ee7798e5022d810682e2cb`）: `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`・`cuda/attention.cu`・`attention.cpp`・`cuda/mem_eff_attention/kernel_forward.h`・`kernels/cutlassF.h`・`aten/src/ATen/Context.h`
- 本製品: `engine_h3/quant_linear.py`・`engine_h3/dit.py`・`engine_h3/rope.py`・`h3_format.py`・`Docs/MINIMAX_H3_DESIGN.md §5`・`Docs/Outputs-archive/h3-stage1-reports-2026-10-17/R1_stack.md`
