# I3 MiniMax H3 の対応 GPU 世代（調査報告・2026-10-18）

調査は読むだけで行った。GPU は使っていない。`.venv-engine-h3` の Python で呼んだのは `torch._C._cuda_getArchFlags()` の 1 回だけで、呼んだ後に `torch.cuda.is_initialized()` が False であることを確かめた。sageattention と comfy_kitchen は import していない（理由は §2-2 の注）。拡張モジュールに入っている GPU 向け機械語の一覧は、CUDA 12.9 の `cuobjdump --list-elf / --list-ptx` で読んだ。

## ① 結論の表

| 部品（版） | Turing sm_75（RTX 20） | Ampere sm_80／86（RTX 30） | Ada sm_89（RTX 40） | Blackwell sm_120（RTX 50） | 根拠 | 確度 |
|---|---|---|---|---|---|---|
| torch 2.13.0+cu130 | 動く（sm_75 の機械語あり） | 動く（sm_80・sm_86） | 動く（sm_86 の機械語がそのまま動く） | 動く（sm_120） | 導入済み torch のビルド対象一覧 `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`。CUDA 13 は Turing より前の世代を削除 | 高（事実） |
| ドライバ | R580 以上 | 同左 | 同左 | 同左 | NVIDIA の CUDA 13.0 リリースノート（表 3「>= 580」）・comfy-kitchen の README「r580+」 | 高（事実） |
| comfy-kitchen 0.2.37 の CUDA 拡張・`int8_linear` | **動く（登録条件は sm_75 以上）。ただし遅い経路**: 回転と量子化の融合は共有メモリの上限で外れる・CUTLASS の int8 GEMM は使わない（Turing 専用の融合カーネルか cuBLAS の int8） | **速い経路**: 融合カーネル＋CUTLASS の int8 GEMM（sm_86 は sm_80 の機械語で動く） | **速い経路**（同左・実測済み） | 機械語あり（sm_120）・速い経路のはず | `_C.abi3.pyd` の機械語 `sm_75/80/89/120`・PTX `sm_89/120f`、`backends/cuda/__init__.py` の制約 `min_compute_capability=(7, 5)` と分岐 | Turing＝コード上は事実・速度と実動は未検証／Ampere＝コード上は事実・実動は未検証 |
| GTX 16 系（Turing・テンソルコアなし） | Turing 専用カーネルから除外され cuBLAS の int8 だけ | — | — | — | `_NVIDIA_16_SERIES`（1660・1650・1630・T500 など）の除外 | 事実（実用性は VRAM 的に論外） |
| SageAttention 2.2.0 post6 cu130 | **使えない見込み**: sm_75 は triton 経路に回る。Triton 3.3 以降は Turing の行列演算コアを使わず、bf16 も Turing では非対応。H3 は bf16 | 使える（`_qattn_sm80` の機械語 sm_80） | 使える（`_qattn_sm89`） | 使える（`_qattn_sm89` に sm_120 の機械語） | `core.py` の世代分岐・各 `.pyd` の機械語・triton-windows の README | Ampere 以降＝高／Turing＝推測（高め） |
| triton-windows 3.7.1.post27 | comfy-kitchen には不要（import は try で守られている）。**SageAttention の import に必須** | 同左 | 同左 | 同左 | `comfy_kitchen/backends/triton/__init__.py:47-95`、`sageattention/core.py:20-27` | 高（事実） |

まとめ: **動作の下限は Turing（CUDA 13 と comfy-kitchen の両方が sm_75 を受け付ける）。速い int8 の経路と SageAttention は Ampere 以降。実測は Ada だけ。** 設計正本 §10 の「ドライバ R580 以上・Turing 以降」は、下限の書き方としては正しい。

## ② 各部品の事実

### 2-1. comfy-kitchen 0.2.37

事実:
- 導入済みの wheel は `comfy_kitchen-0.2.37-cp312-abi3-win_amd64.whl`（dist-info の WHEEL のタグ `cp312-abi3-win_amd64`）。PyPI の 0.2.37 には、ほかに cp310・cp311 の win_amd64、cp312-abi3 の win_arm64（7.6MB）、純 Python の `py3-none-any` がある（https://pypi.org/pypi/comfy-kitchen/0.2.37/json）。
- `backends/cuda/_C.abi3.pyd`（25,742,848 バイト）の中身（cuobjdump）: 機械語（cubin）は **sm_75・sm_80・sm_89・sm_120** が 32 本ずつ、PTX は **sm_89・sm_120f** が 32 本ずつ。README は Windows の既定のビルド対象を `75-virtual;80;89;120f` としていて、75 は PTX だけのように読める。しかし配布の wheel には **sm_75 の機械語が入っている**（研究ノート §13.4 の「`75-virtual`」は README の既定値であって、配布物の中身ではない）。
- sm_86（RTX 30 系）向けの機械語は無い。sm_80 の機械語は同じ major の sm_86 でそのまま動く（CUDA のバイナリ互換の規則）。
- 要件（METADATA の README「Requirements」）: Python ≥3.10・PyTorch ≥2.7・CUDA ランタイム ≥13.0・「Pre-built wheels require NVIDIA Driver r580+」。
- `int8_linear` の CUDA 版の登録（`backends/cuda/__init__.py:4463-4486`）: `_CUBLASLT_AVAILABLE` が真のときだけ登録し、`min_compute_capability=(7, 5)`。**Turing でも CUDA 版が選ばれる。**
- 世代による分岐（同ファイル）:
  - CUTLASS の int8 GEMM と逆量子化の融合は major ≥ 8 のときだけ（`_cuda_device_supports_cutlass_int8_dequant`・`:319-330`）。Ampere・Ada・Blackwell が対象。
  - Turing（capability が (7, 5)）で、GTX 16 系ではない機種は、`m ≤ 128` または `n ≥ 2k` の形のとき Turing 専用の融合カーネル `_C.cutlass_turing_int8_dequant` を使う（`:2196-2209`）。それ以外の形は cuBLAS の int8 GEMM と別カーネルの逆量子化を使い、Turing では K と N を詰め物で揃える（`:2256-2290`）。
  - GTX 16 系（`_NVIDIA_16_SERIES`・`:247-`）は Turing 専用カーネルから外される（`:310-313`）。
  - 回転と量子化の融合カーネル（groupsize 256）の条件の 1 つは「共有メモリの必要量 (K+8192)×4 バイトがブロックあたりの上限未満」（`:418-467`・`:2061-2067`）。上限は **Turing だけ `shared_memory_per_block`（opt-in ではない値）** を使い、それ以外は `shared_memory_per_block_optin` を使う（`:403-415`）。
- GitHub のリリースページにはリリースノートが無い（「There aren't any releases here」・https://github.com/Comfy-Org/comfy-kitchen/releases）。直近 100 件のコミットで世代に関わるものは `152d52a`（2026-10-16「Optimize int8 attention for ampere (#234)」・0.2.37 より後）・`178050f`（09-30「Further optimize int8 GEMM」）・`36ba0d4`（09-28「fix: decline fused DeltaNet kernels below SM75」）。

推測（計算）:
- Turing の `shared_memory_per_block` は 48KB（49,152 バイト）のはず。H3 の DiT の K（5376・7168・14336）で必要量は 54,272／61,440／90,112 バイトで、**どれも上限を超える**。だから Turing では DiT の全線形層で回転と量子化が融合されず、「bf16 の回転の行列積 → 量子化」の 2 段になる。さらに行列積も、fc1（n ≥ 2k）以外は CUTLASS ではなく cuBLAS の int8 になる。Ada と比べて明確に遅くなる見込み（どれだけ遅いかは測っていない）。
- Ampere の sm_86 は opt-in 上限が 101,376 バイトで、上の 3 つの K はすべて収まる。sm_80（A100）は上限がもっと大きい。つまり Ampere は Ada と同じ「融合カーネル＋CUTLASS」の経路に乗るはず。ただし **CUTLASS の int8 は sm_80 の機械語で sm_86 を動かす形**なので、実機で動くことと速度は未検証。

R1_stack の記述の出典について: 「Ada/Ampere では回転と量子化の融合カーネル」は、R1_stack §2-2（105-106 行）と (d)（233-240 行）が comfy-kitchen のソース（`backends/cuda/__init__.py:2060-2066`・`:418-458`・`:2228-2262`）を読み、共有メモリの上限 101,376 バイトを当てはめて計算した推測である。外部の文書の記述ではない。今回ソースを読み直し、分岐の条件は R1 の記述どおりであることを確かめた。Turing が外れる理由（opt-in ではない上限を使う）は R1 には無く、今回新しく分かったこと。

### 2-2. SageAttention 2.2.0+cu130torch2.10.0andhigher.post6

事実:
- 導入済みの wheel のタグは `cp310-abi3-win_amd64`（WHEEL）・`Requires-Python: >=3.10`。Python 3.10 以上で共通に使える。
- torch の範囲: ファイル名が「torch2.10.0andhigher」。post4 のリリースノートに「along with ABI3, it uses libtorch stable ABI」「the wheel supports PyTorch >= 2.9」（post4 の版の話）とある（https://github.com/woct0rdho/SageAttention/releases）。post6 のノートは「An out-of-bound issue is fixed, which caused black or noise outputs.」だけで、版の範囲は書いていない。torch 2.13 での動作は第 1 弾の自己検査で確かめた（研究ノートと記録による。今回は再確認していない）。
- CUDA: ファイル名の `cu130`（CUDA 13.0 でビルド）。
- 世代ごとの振り分け（`sageattention/core.py:157-180`）: sm75 → `sageattn_qk_int8_pv_fp16_triton`（triton のカーネル）、sm80/86/87 → `_qattn_sm80`、sm89 → `_qattn_sm89`、sm90 → `_qattn_sm90`、sm100/120/121 → `_qattn_sm89` の関数。それ以外は `ValueError`。
- 各拡張の機械語（cuobjdump）: `_qattn_sm80.pyd`＝sm_80、`_qattn_sm89.pyd`＝sm_89・sm_120、`_qattn_sm90.pyd`＝sm_90、`_fused.pyd`＝sm_80・sm_89・sm_90・sm_120。**sm_75 の機械語はどれにも無い。**
- triton-windows の README（https://github.com/woct0rdho/triton-windows）: 「GTX 16xx/RTX 20xx (Turing architecture, sm75)」は「officially supported by Triton <= 3.2」「Support for Turing has been dropped since Triton 3.3」、「bf16 (also known as bfloat16) on Turing are not supported by the official Triton」。fork として Turing の bf16 を足したという記述は無い。根拠の triton-lang PR #5066 は「[BACKEND] Deprecate MMAv1; fallback to FMA for computeCapability < 80」で、保守担当は「best effort and won't be tested on our side」と書いている（https://github.com/triton-lang/triton/pull/5066）。
- 本製品の H3 は DiT を bf16 で組む（`engine_h3/dit.py:217-305`）。attention の既定は sdpa で、sage は指定したときだけ使う（`engine_h3/worker.py:454`）。sage が使えないときは sdpa に落ちて `attention_used="sage->sdpa"` と報告する（`worker.py:186-199`・`:333`）。

推測:
- Turing では、sage を指定すると triton 3.7 の経路に入る。行列演算コアを使わない FMA の経路で、しかも bf16 は非対応なので、**カーネルの実行で失敗するか、極端に遅くなる**見込み。この失敗は import の段階では起きないので、本製品の import による判定（`probe_sage()`）では拾えない。Turing では「sage は使えない（sdpa を使う）」と書くのが安全。
- 注: `sageattention/core.py:71` は import の時点で `torch.cuda.get_device_capability()` を全機器について呼ぶ。つまり **sageattention の import は CUDA を初期化する**（今回 import しなかった理由）。本製品の import による判定も、判定の時点で CUDA の文脈を作っていることになる。害があるかどうかは調べていない。

### 2-3. triton-windows 3.7.1.post27

事実:
- comfy-kitchen は triton を必須としない。`backends/triton/__init__.py:44-95` は `import triton` を try で囲み、失敗したら `_TRITON_AVAILABLE=False` にして登録しないだけ。METADATA の依存も空（extra の `cublas` に `nvidia-cublas>=13.0.0` があるだけ）。
- torch 2.13.0+cu130 の Windows 版の METADATA に triton の依存は無い。`engine_h3/*.py` に `torch.compile` と `import triton` は無い。
- **SageAttention は triton を必須とする**。`sageattention/__init__.py` → `core.py:20-27` が `sageattention/triton/*.py` を無条件に import し、その各ファイルが `import triton` を行う。triton が無いと `from sageattention import sageattn` が ImportError になり、本製品は sdpa に落ちる。
- triton-windows の README: CUDA 12 以上が要る（wheel に最小限の CUDA ツールチェーンを同梱）、vcredist（Visual C++ 再頒布可能パッケージ）が必須、「make sure you have the latest GPU driver」。

結論: triton-windows は **sage を残すなら必要、sage をやめるなら要らない**。H3 自体（comfy-kitchen の CUDA 拡張の経路）には要らない。README の要件行には書かなくてよい（インストーラが入れる部品で、利用者が用意するものではない。vcredist は既存の環境と同じ要件）。

### 2-4. CUDA 13 系の torch（2.13.0+cu130）

事実:
- NVIDIA の CUDA Toolkit のリリースノート（https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html）: 表 2 は CUDA 13.0 を R580 系列に対応づけ、表 3 は 13.x の minor version compatibility を「>= 580」としている。表の下に「Existing CUDA 13.x applications run on drivers >=580 under CUDA minor version compatibility.」とある。Windows と Linux で下限は同じ。
- 同じノートの cuFFT 13.0 の節に「Removed support for Maxwell, Pascal, and Volta GPUs, corresponding to compute capabilities earlier than Turing.」、cuSPARSE 13.0 の節に「Dropped support for pre-Turing architectures (Maxwell, Volta, and Pascal)」とある。
- 導入済みの torch 2.13.0+cu130（`torch.version.cuda`＝13.0）のビルド対象（`torch._C._cuda_getArchFlags()`）は `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`（PTX は無い）。sm_89（Ada）は sm_86 の機械語で動き、Pascal（sm_61）以前は入っていない。
- PyTorch 側の対応表の一次資料は今回は開いていない。上のビルド対象の一覧で代えた（配布物そのものなので、表より確か）。

## ③ 文言案（1 案）

README の H3 の動作要件の行として:

> - **NVIDIA ドライバ**: R580 以上（CUDA 13 系を使うため。これより古いドライバでは動きません）
> - **GPU**: RTX 20 シリーズ（Turing 世代）以降。GTX 10 シリーズ以前は CUDA 13 の対象外のため動きません。
>   - 動作を確かめたのは RTX 40 シリーズ（Ada 世代・RTX 4070 Ti SUPER）だけです。RTX 30 シリーズ（Ampere 世代）と RTX 50 シリーズ（Blackwell 世代）は、同じ速い計算経路に乗る作りですが、まだ確かめていません。
>   - RTX 20 シリーズは、int8 の計算が遅い経路になり、SageAttention も使えません（標準の注意機構で動きます）。動くかどうかも未確認です。

（書き方の理由: 「下限＝動く可能性がある範囲」と「確かめた範囲」と「遅くなる範囲」を分ける。Turing を下限に書くのは CUDA 13 と comfy-kitchen の登録条件という事実に基づくが、実動の裏付けは無いので「未確認」を併記する。）

## ④ 未確認

- Turing で H3 が最後まで動くか（bf16 の行列積・回転の行列積・sdpa の bf16 が sm_75 でどの実装になるか。とくに sdpa は、flash と memory-efficient の bf16 版が sm_80 以上を要求していれば math 版に落ち、H3 の系列長では VRAM が足りなくなるおそれがある＝推測。torch の sm_75 での bf16 の扱いの一次資料は見ていない）。
- Ampere（sm_86）で CUTLASS の int8（sm_80 の機械語）と融合カーネルが実際に選ばれ、正しく速く動くか。Ampere 16GB の第三者の実測は研究ノート §13.5 の時点で見つかっていない（今回は再検索していない）。
- Blackwell（sm_120）での実動（comfy-kitchen・sage・torch とも機械語はある）。
- Turing の `shared_memory_per_block` が 48KB であること（一般的な値。実機で読んでいない）。
- comfy-kitchen の Turing 専用カーネルが入った版（GitHub の直近 100 件のコミットの範囲には無い）。
- PyTorch 公式の cu130 の対応世代表（配布物のビルド対象で代えた）。
