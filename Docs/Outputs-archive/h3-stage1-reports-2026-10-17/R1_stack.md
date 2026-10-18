# R1 調査報告: MiniMax H3 専用 venv のスタック（版の組み合わせ）

調査日: 2026-10-17。調査担当 R1。何もインストールしていない（作ったのはスクラッチ領域の捨て venv `scratchpad\h3\venv-probe` だけで、そこでは解決の試行 `--dry-run` しかしていない）。本製品のファイルと `.venv*` には触れていない。GPU は使っていない。

表記: 【事実】は出典つきで確認したこと。【推測】は根拠からの推論で、実機で確かめていないこと。

---

## (1) 結論

**第 1 候補は torch 2.13.0+cu130 の組み合わせ。** 決め手は comfy-kitchen の CUDA 拡張の性質にある。CUDA 拡張 `_C.abi3.pyd` は torch の DLL を一切リンクしていない。静的に依存しているのは `nvcuda.dll`（ドライバ）と `python3.dll` だけで、cuBLASLt は実行時に `cublasLt64_13.dll` を探して読み込む。だから、**CUDA 13 系の torch でさえあれば、torch の版は問わない**。そうなると torch の版を決めるのは周辺（SageAttention・triton-windows・flash-attn）の実績であり、H3 での第三者の実測（§13.5）と SageAttention の作者の発言の両方が 2.13 を指している。

### 第 1 候補（推奨）

| パッケージ | 版 | wheel の URL | 根拠 |
|---|---|---|---|
| Python | 3.12（uv 管理） | — | 既存 2 系統と同じ。comfy-kitchen の Windows wheel は `cp312-abi3`。 |
| torch | **2.13.0+cu130** | https://download-r2.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-win_amd64.whl （index の upload-time 2026-07-08T20:26:27Z・1,915,519,202 バイト） | 第三者が H3 int8 ConvRot で cu129 比 2.17 倍の実測を出した版（研究ノート §13.5）。SageAttention の作者が「torch 2.13 なら triton-windows 3.7」と案内している（woct0rdho/SageAttention issue #101、2026-08-17）。flash-attn の Windows wheel もある。 |
| torchvision | 0.28.0+cu130 | https://download-r2.pytorch.org/whl/cu130/torchvision-0.28.0%2Bcu130-cp312-cp312-win_amd64.whl （2026-07-08） | torch 2.13.0 の対。DiffSynth が依存に持つ。 |
| torchaudio | **入れない** | — | cu130 の index にある Windows cp312 の torchaudio は 2.11.0 が最後。2.12 以降の wheel は存在しない（§2-1）。DiffSynth の H3 の経路は torchaudio を import しない。 |
| comfy-kitchen | 0.2.37 | https://files.pythonhosted.org/packages/bf/0a/96c1141180bdc4dce2664cd9316390069c66c6f03e79963a17f252b4b730/comfy_kitchen-0.2.37-cp312-abi3-win_amd64.whl （2026-10-02・sha256 31e1b593…fdb8） | PyPI の最新。`dependencies = []`。 |
| triton-windows | 3.7.1.post27 | https://files.pythonhosted.org/packages/76/30/325b420efd0047e119679c646a9a410db216069800ec009fae3da26c69a3/triton_windows-3.7.1.post27-cp312-cp312-win_amd64.whl （2026-06-21） | torch 2.13.0 の Linux 版は `triton==3.7.1` を要求する（PyPI のメタデータ）。SageAttention の作者の案内とも一致する。 |
| sageattention | 2.2.0+cu130torch2.10.0andhigher.post6 | https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl （2026-07-17・正式版） | 既存 venv の post6 と同じリリースの cu130 版。libtorch の安定 ABI（版をまたいで使える C++ の境界）でビルドされ、torch 2.10 以上で使える。 |
| transformers | 5.14.1 | https://files.pythonhosted.org/packages/…/transformers-5.14.1-py3-none-any.whl （2026-07-16） | 既存の LTX 2.5 venv の freeze と同じ版。Qwen3-VL は 4.57.0 から入っている。最新の 5.20.1 は公開から 1 日しか経っていない（2026-10-16）。 |
| safetensors | 0.8.0 | PyPI | 最新で、既存 freeze とも同じ。transformers 5.14.1 は 0.8.0 以上を要求する。 |
| diffsynth | 2.1.9（任意） | https://files.pythonhosted.org/packages/16/91/d394550f1622c69259c413ec7ead80f9e407373933571d605e084b917472/diffsynth-2.1.9-py3-none-any.whl （2026-10-09） | PyPI 版にも H3 一式（`minimax_h3_dit_comfy.py`・`core/quant/backends/comfy_kitchen.py` など）が入っていることを wheel の中身で確かめた。引き込む依存は §2-6。 |
| accelerate／einops／av | 解決結果のまま（1.15.0／0.8.2／19.0.1） | PyPI | 既存 freeze の av は 18.1.0。揃えたいなら固定する。 |

uv の解決（dry-run）は成功し、62 パッケージになった（§2-7）。

### 第 2 候補（最新の安定版）

| パッケージ | 版 | URL・公開日 |
|---|---|---|
| torch | 2.14.1+cu130 | https://download-r2.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-win_amd64.whl （2026-09-30T20:25:17Z・1,990,608,063 バイト） |
| torchvision | 0.29.1+cu130 | https://download-r2.pytorch.org/whl/cu130/torchvision-0.29.1%2Bcu130-cp312-cp312-win_amd64.whl （2026-09-30） |
| triton-windows | 3.8.0.post29 | https://files.pythonhosted.org/packages/ec/bc/b027bba13beb00cd4993a33b7fa89a84ef3a42ecd05d8a0e346526a7dc80/triton_windows-3.8.0.post29-cp312-cp312-win_amd64.whl （2026-09-28）。torch 2.14.1 の Linux 版は `triton~=3.8.0` を要求する。 |
| その他 | 第 1 候補と同じ | SageAttention post6 の torch2.10.0andhigher 版は、torch 2.14 より前（7 月）の公開。安定 ABI の設計上は使えるはずだが、2.14 で動いたという公開の報告は見つけていない。 |

第 2 候補の利点は、patch 版（2.14.1）が出ていて新しいこと。欠点は、周辺の実績が薄いこと。どちらも dry-run は成功した。

---

## (2) 項目 1〜8 の調査結果

### 2-1. torch

- 【事実】`https://download.pytorch.org/whl/cu130/torch/` にある Windows x64・cp312 の torch は、2.9.1／2.10.0／2.11.0／2.12.0／2.12.1／2.13.0／2.14.0／2.14.1（2026-10-17 取得）。cu132 の index にも 2.12.0〜2.14.1 がある。cu133 の index は空。
- 【事実】PyPI での公開日は次のとおり。2.12.0 が 2026-05-13、2.12.1 が 06-17、2.13.0 が 07-08、2.14.0 が 09-02、2.14.1 が 09-30。2.13 の patch 版（2.13.1）は存在しない。
- 【事実】torchvision の cu130・cp312・Windows は 0.24.1〜0.29.1。torchaudio の cu130・cp312・Windows は 2.9.0／2.9.1／2.10.0／2.11.0 で終わっており、cu132 の index には Windows 版がない。PyPI の torchaudio も最新は 2.11.0。→ torch 2.12 以降と組める torchaudio はない。
- 【事実】cu130 の Windows wheel は CUDA ランタイムを同梱している。torch 2.14.1+cu130 の wheel の中央ディレクトリを HTTP Range で読み、`torch/lib/cudart64_13.dll`・`cublas64_13.dll`・`cublasLt64_13.dll`・`nvrtc64_130_0.dll`・`cudnn64_9.dll` などがあることを確かめた。uv の解決結果にも nvidia-* の別パッケージは出てこない。
- 【事実・ドライバの最低版】NVIDIA の CUDA Toolkit リリースノート（https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html 、2026-10-17 取得）の表 3 によると、CUDA 13.x はマイナー版互換で **ドライバ 580 以上**なら動く。対応表は 13.0＝R580、13.1＝R590、13.2＝R595、13.3＝R610、13.4＝R615。cu130 の wheel を使う限り、R580 以上で足りる。
- 【事実】同じリリースノートに「Maxwell・Pascal・Volta（Turing より前、compute capability 7.5 未満）のサポートを削除」とある。→ H3 の venv は GTX 10xx などの Pascal 世代では動かない。ここが既存 2 系統（cu128）と違う。
- 【事実】Linux の PyPI 版 torch は 2.11 から CUDA 13.0 が既定になっている（2.11.0 のメタデータに `cuda-toolkit[...]==13.0.2`）。

### 2-2. comfy-kitchen

**版と wheel**
- 【事実】PyPI の最新は 0.2.37（2026-10-02）。その前は 0.2.36（09-29）・0.2.35（09-17）・0.2.34（09-15）。GitHub の main は 2026-10-14 まで進んでいるが、まだ公開されていない。
- 【事実】Windows の CUDA wheel は `comfy_kitchen-0.2.37-cp312-abi3-win_amd64.whl`（48.8MB。Python 3.12 以上で共通の abi3）。中には CUDA 版 `backends/cuda/_C.abi3.pyd`（25.7MB）と AMD 用 `backends/hip/_C.abi3.pyd`（133MB）の 2 つが入っている。cp310 と cp311 には個別の wheel があり、ほかに純 Python の `py3-none-any` もある。
- 【事実】README の Requirements: PyTorch ≥2.7.0、CUDA ランタイム ≥13.0、配布 wheel はドライバ r580 以上、Windows の既定アーキテクチャは `75-virtual;80;89;120f`。`pyproject.toml` は `dependencies = []`、extra `cublas` は `nvidia-cublas>=13.0.0`。torch の上限の指定はない。
- 【事実】`_C.abi3.pyd` のバイト列に出てくる DLL 名は `KERNEL32.dll`・`nvcuda.dll`・`python3.dll`・`cublasLt64_13.dll`・`cublasLt64.dll` だけで、torch と cudart は出てこない。cuBLASLt は `cublaslt_runtime.h:137-143` の `LoadLibraryA` で実行時に読み込まれる。読み込みの前に、`backends/cuda/__init__.py:94-120` が `nvidia.cu13`、無ければ `torch.__path__` の下から `cublasLt64` を探して `os.add_dll_directory` に足す。
- 【推測】以上から、CUDA 拡張が効く条件は「ドライバ R580 以上」と「`cublasLt64_13.dll` が見つかる（＝torch cu13x か `nvidia-cublas` 13 系が入っている）」の 2 つで、torch の版には依存しない。sm_86（RTX 3080 mobile）は sm_80 の機械語がそのまま動く世代なので、`80` が入っていれば足りる。

**実行時の判定に使える公開 API**
- 【事実】`comfy_kitchen.list_backends()`（`__init__.py:1090`）は `{"cuda": {"available", "disabled", "unavailable_reason", "capabilities"}, ...}` を返す。CUDA 拡張が読めなかったときは、`available=False` で、理由が `unavailable_reason` に入る（`backends/cuda/__init__.py:4536-4550`）。
- 【事実・重要】CUDA 拡張が読めても、cuBLASLt が見つからなければ、cuda バックエンドの能力に `int8_linear` は登録されない（`backends/cuda/__init__.py:4463` の `if _CUBLASLT_AVAILABLE:`）。そこで判定は **`"int8_linear" in ck.list_backends()["cuda"]["capabilities"]`** にするのが正確。`available` だけでは足りない。
- 【事実】この他の公開 API: `disable_backend(name)`・`enable_backend`・`set_backend_priority`・`use_backend(name)`（コンテキストマネージャ）・`int8_attention_is_available()`。

**CUDA 拡張が無いときの振る舞い**
- 【事実】選ぶ順は `cuda → triton → eager` で、各バックエンドが宣言した制約（dtype・次元・compute capability）を事前に照合し、通ったものを使う（README の「Automatic Backend Selection」、`registry.py:25`）。triton は `triton` を import できて CUDA があれば無条件に登録される（`backends/triton/__init__.py:343-359`）。triton 版の `int8_linear` は SM 8.0 以上が条件（同 `:239-250`）。eager 版は `torch._int_mm` を使う。
- 【推測・重要】cu128 の venv に triton-windows を入れた構成で comfy-kitchen を使うと、`int8_linear` は **triton 経路に落ちる**。そして issue #136 の事故はその経路で起きる（§2-3）。ComfyUI 本体が `torch.version.cuda < 13` で comfy-kitchen の CUDA バックエンドを切り、triton も既定で切っている（研究ノート §13.4）のは、これを避けるためと読める。

**int8 tensorwise＋ConvRot の最小手順（ソースの該当箇所）**

手順 A は、量子化テンソルを経由する方法。DiffSynth が使っているのはこちら。
```python
import json, torch
from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
m = json.loads(bytes(sd[p + ".comfy_quant"].tolist()))           # {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
w = sd[p + ".weight"]                                              # I8 [N,K]
params = TensorWiseINT8Layout.Params(
    scale=sd[p + ".weight_scale"],                                 # F32 [N,1]
    orig_dtype=torch.bfloat16, orig_shape=tuple(w.shape),
    is_weight=True, convrot=m.get("convrot", False),
    convrot_groupsize=m.get("convrot_groupsize", 256))
qw = QuantizedTensor(w, "TensorWiseINT8Layout", params)
y = torch.nn.functional.linear(x_bf16, qw, bias)                   # → torch.ops.comfy_kitchen.int8_linear
```
- `Params` のフィールドは `tensor/int8.py:67-77` にある（scale・orig_dtype・orig_shape は `tensor/base.py:40-49` から引き継ぐ）。`QuantizedTensor(qdata, layout名, params)` は `tensor/base.py:157-180`。
- `F.linear` が分解されて届く `aten.linear`・`aten.t`・`aten.mm`・`aten.addmm` は、すべて `tensor/int8.py:248-375` で受け取られ、`torch.ops.comfy_kitchen.int8_linear` に渡る（カスタム op の定義は `backends/eager/quantization.py:1230`。中でバックエンドを選ぶ）。
- DiffSynth の組み立て方も同じ形だった（`diffsynth/core/quant/backends/comfy_kitchen.py:236-255` の `unflatten_state_dict`）。

手順 B は、関数を直接呼ぶ方法。量子化テンソルの部分クラスを使わない。
```python
import comfy_kitchen as ck
y = ck.int8_linear(x_bf16, w_i8, w_scale_f32, bias, out_dtype=torch.bfloat16,
                   convrot=True, convrot_groupsize=256)              # __init__.py:997-1059
```
- 【推測】本製品の block swap と噛み合わせるなら、手順 B が扱いやすい。重みが普通の int8 テンソルと f32 テンソルのままなので、既存のバイト単位の置き場（`engine/transformer/block_swap_prefetch.py`）にそのまま載る。

**仕様の細部**
- 活性化の dtype: CUDA 版の制約は x が f32・f16・bf16 のいずれかで 2 次元以上、weight が int8 の 2 次元、weight_scale が **float32**、out_dtype が f32・f16・bf16（`backends/cuda/__init__.py:4464-4488`）。bf16 は受け付ける。out_dtype の既定は bf16（`__init__.py:1046`）。
- 動的な活性化の int8 化: 行（トークン）ごとに absmax を取り、scale＝absmax/127（f32、行ごと）とする。最近接に丸めて [-128,127] に収める（`backends/cuda/ops/int8_linear.cu:185-230`）。ConvRot のときは、先に 256 要素ずつのグループでアダマール回転をかける。
- groupsize 256 の融合カーネル（回転と量子化を 1 回で行う）が使われる条件は、`backends/cuda/__init__.py:2060-2066`。convrot が真、groupsize＝256、K が 256 の倍数、256 ≤ K ≤ 16384（`_CONVROT_FUSED_MAX_K`、`:1631`）、共有メモリの必要量が上限未満（`:418-458`。K が特別な値でなければ (K+8192)×4 バイト）、の 5 つ。
  - 【推測・計算】H3 の DiT の 4 種の K は 5376（qkv・fc1）・7168（out_proj）・14336（fc2。swiglu 後）で、どれも条件を満たす。必要な共有メモリは最大でも fc2 の 90,112 バイトで、Ada・Ampere の上限 101,376 バイトに収まる。
  - 【推測・計算】テキストエンコーダ（Qwen3-VL 32B）の down_proj は K＝25600 で、上限 16384 を超える。融合されない経路（回転の行列積の後に量子化。`:2147-2150`）になり、遅くはなるが動く。
  - 非枝刈り版の AdaLN（groupsize 64）も融合されない経路になる。
- 行列積: SM 8.0 以上では CUTLASS の int8 GEMM と逆量子化の融合を使い、駄目なら cuBLAS の int8 と別カーネルの逆量子化に落ちる（`:2228-2296`）。重みスケールの `[N,1]` は要素数 N の縦ベクトルとしてそのまま通る（`:2236`）。
- 重みの移動操作（`tensor/base.py`）:
  - `.to(device)`／`._to_copy` は処理されるが、`non_blocking` は無視されて同期コピーになる（`_handle_to`、`:417-436`。`qt._qdata.to(device=…)` に引数を渡していない）。
  - `copy_` は、コピー元も同じ型の量子化テンソルである場合に限り処理され、`non_blocking` も効く（`:453-466`）。ただし、`non_blocking` を位置引数で渡すと `False` でも真として扱われる（`len(args) >= 3` の判定）。
  - **`pin_memory` は処理表に無い**ので、汎用の後戻り経路で逆量子化され、bf16 の普通のテンソル（容量 2 倍）に化ける（`:347-372`。`_DISPATCH_TABLE` は `:484-495`）。`is_pinned()` は中身の int8 テンソルに委ねている（`:269`）。
  - `.data =` の差し替えに要る `_has_compatible_shallow_copy_type` は True を返す（`:494`）。
  - 【推測】block swap は、量子化テンソルの外側ではなく、中身（`_qdata` と `_params.scale`）か、手順 B の普通のテンソルを pin して出し入れするのが安全。

### 2-3. issue #136 の影響範囲

- 【事実】issue #136「INT8 GEMM kernels index the output in int32: illegal memory access once m*n >= 2**31 (MiniMax H3 at high resolution x length)」は 2026-08-24 に立ち、2026-10-17 時点で未解決（open）。原因は、**triton** の 2 つのカーネル `_int8_matmul_dequant_kernel` と `_int8_matmul_dequant_per_row_kernel`（`backends/triton/quantization.py`）が出力の位置を int32 で計算していること。m（トークン数）×n（出力幅）が 2³¹ 以上になると不正なメモリアクセスで落ち、プロセスごと死ぬ。報告者の環境は torch 2.10.0+cu128・triton-windows 3.6・`--enable-triton-backend`。
- 【事実】v0.2.37 でも、2026-10-17 時点の main でも直っていない（`quantization.py:904-905, 986-987` が `offs_am = (pid_m * block_m + tl.arange(0, block_m)) % m` のまま）。
- 【事実・CUDA 経路で起きない根拠（ソースの水準）】
  - CUTLASS の int8 GEMM の出力ストライドは int64 で宣言されている（`backends/cuda/ops/cutlass_gemm_int8.cu:94, 160, 183-184` の `cute::Stride<int64_t, _1, int64_t>`、`{int64_t(N), _1{}, int64_t(M) * N}`）。M・N・K の引数も int64（`:344-411`）。
  - 活性化の量子化と逆量子化のカーネルは、行の先頭位置と要素の位置を int64 で計算している（`int8_linear.cu:198, 215, 303, 356, 524-561` など）。
  - 【推測】以上から、CUDA 拡張の経路は #136 の int32 の限界を持たない。ただし実際に m×n ≥ 2³¹ の形を CUDA 経路に通した試験はしていないので、「言い切り」はソースを読んだ水準にとどまる。
- 【推測・計算】H3 での境目。最も幅の広い fc1（n＝28672）が最初に越える。m ≥ 2³¹/28672 ≈ 74,899 トークン。映像のトークンは 1 潜在フレームあたり (W/32)×(H/32) で、潜在フレームは (F−1)/4+1。1280×736 なら 920 トークン×82 潜在フレーム（325 フレーム）、1344×768 なら 1008 トークン×75 潜在フレーム（約 297 フレーム）あたりで越える。issue の報告（1344×768・294 フレームで落ちる）とほぼ合う。音声トークンなどが同じ列に詰められていれば、境目はさらに少し手前になる。
- 【推測・対策】triton 経路に落ちない形にすることが対策になる。CUDA 拡張が `int8_linear` を持っていることを起動時に確かめる（§2-2 の判定式）。持っていなければ、`ck.disable_backend("triton")` で eager に落とす（遅いが落ちない）か、H3 を使えなくする。

### 2-4. attention のバックエンド

- 【事実】SageAttention（woct0rdho の Windows ビルド）の cu130 向けは、v2.2.0-windows.post6（2026-07-17、正式版）に `sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl`（16.7MB）がある。post4 以降は libtorch の安定 ABI でビルドされ、「torch X 以上」で使える（post4 のリリースノート）。post7（2026-10-09）はプレリリースで、AMD 用だけ。
- 【事実】作者は issue #101（2026-08-17）で、torch 2.13 の利用者に triton-windows 3.7 を案内している。issue #91（2026-05）では torch 2.12＋CUDA 13 での自前ビルドの失敗が報告され、修正された。
- 【事実・代替】flash-attn 2.8.3 の Windows 向けビルド済み wheel が mjun0812/flash-attention-prebuild-wheels にある。v0.9.52（2026-07-26）に `flash_attn-2.8.3+cu130torch2.13-cp312-cp312-win_amd64.whl`、v0.10.2（2026-09-08）に `…+cu130torch2.14-…` がある。研究ノート §13.5 の RTX 4080 の報告は、この系統（FlashAttention 2.8.3＋torch 2.13+cu130）。torch 標準の SDPA は常に使える。
- 【事実】comfy-kitchen 自体にも int8 の attention（`ck.int8_attention`、判定は `ck.int8_attention_is_available()`、`sage_attention.py:135-148`）がある。
- 【事実・本製品の判定】`sage_available` は `engine/transformer/sage_attention_service.py:146-175` の `probe_sage()` が出す。中身は `_sage_callable()`（`:99-`）で `from sageattention import sageattn` を行い、関数の引数の形を確かめるだけ。例外は BaseException まで捕まえて False にする。カーネルは実行しない。engine25 も同じ関数を import している（`engine25/worker.py:344, 367`）。
- 【推測】新しい venv でも、この判定は同じ意味で成り立つ。不一致の wheel の典型的な失敗（issue #101 の `DLL load failed while importing _fused`）は import の時点で起きるので、拾える。ただし、import は通ってもカーネルの実行で失敗する種類（GPU 世代・head_dim 不一致など）は拾えない。H3 の head_dim は 128 で、SageAttention の対応範囲に入る。

### 2-5. triton-windows

- 【事実】PyPI の triton-windows の cp312 の最新は 3.8.0.post29（2026-09-28）。3.7.1.post27（06-21）と 3.7.0.post26（05-14）もある。
- 【事実】torch と triton の対応は、Linux 版 torch のメタデータでは 2.11→`triton==3.6.0`、2.12.1→`3.7.1`、2.13.0→`3.7.1`、2.14.1→`~=3.8.0`。triton-windows の README の対応表は 2026-02 の更新で止まっていて、2.10→3.6 までしか載っていない。
- 【注意】comfy-kitchen は triton を import できれば triton バックエンドを登録する。issue #136 があるので、H3 の venv に triton-windows を入れるなら、起動時に CUDA 拡張の `int8_linear` があることを確かめるのが前提になる（§2-3）。SageAttention は triton を実行時の JIT（その場でのカーネル生成）にも使うので、入れておくほうが無難。

### 2-6. transformers／safetensors／DiffSynth の依存

- 【事実】Qwen3-VL（`models/qwen3_vl/modeling_qwen3_vl.py`）は transformers v4.57.0（2025-10-03）で入った。v4.56.2 には無い。→ **下限は 4.57.0**。DiffSynth の `minimax_h3_text_encoder.py:2, 20-72` は `Qwen3VLConfig(**{... "num_hidden_layers": num_retained_layers ...})` と `Qwen3VLModel(config)` で組み立てている（`transformers_version: "4.57.0.dev0"`）。
- 【事実】transformers 5 系は `rope_scaling` を `rope_parameters` に読み替える互換処理を持つ（v5.20.1 の `modeling_rope_utils.py:753-754`）。Qwen3-VL の設定は `mrope_section` と `mrope_interleaved` を検証の対象から外している（`configuration_qwen3_vl.py:82`）。
- 【事実】transformers 5.14.1 の必須依存は huggingface-hub<2.0,>=1.5.0・tokenizers<=0.23.0,>=0.22.0・safetensors>=0.8.0・typer など。torch の版の上限はない。
- 【事実】DiffSynth（PyPI の `diffsynth` 2.1.9、2026-10-09）の必須依存は torch>=2.0.0・torchvision・transformers・imageio[ffmpeg]・safetensors・einops・modelscope・ftfy・pandas・accelerate・peft。comfy-kitchen は extra `quant` に入っている（同じ extra に bitsandbytes・entropack[cuda13]・torchao>=0.16 も入っているので、extra ごと入れるのは勧めない）。
- 【事実】uv の解決で、diffsynth を加えると増えるパッケージは 19 個だった（diffsynth・modelscope・modelscope-hub・pandas・peft・imageio・imageio-ffmpeg・ftfy・cryptography・cffi・requests など）。重いのは imageio-ffmpeg（31MB。ffmpeg の実行ファイルを同梱）くらい。deepspeed や bitsandbytes のような重いものは入らない。
- 【事実】`import diffsynth` は `diffsynth/core/__init__.py` から、loader（`modelscope`・`huggingface_hub`）と data（imageio）を import する。→ `--no-deps` で入れると import できない。必要なファイルだけを持ってくる（Apache-2.0）か、依存ごと固定するかを選ぶことになる。
- 【事実】DiffSynth の H3 の各ファイル（`models/minimax_h3_*.py`・`pipelines/minimax_h3_audio_video.py`・`utils/data/minimax_h3.py`）に torchaudio・torchcodec・librosa の import はない。

### 2-7. uv による解決の確認

- 捨て venv は `uv venv …\scratchpad\h3\venv-probe --python 3.12` で作った（CPython 3.12.13、uv 0.6.4）。
- コマンドは `uv pip install --dry-run --python <probe>\Scripts\python.exe --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match -r <ピン一覧>`。uv 0.6.4 は `--dry-run` に対応していた。download.pytorch.org の index はメタデータを別に配っている（PEP 658 の `data-dist-info-metadata`）ので、2GB の wheel を落とさずに解決できる。
- 結果（入力と出力は `scratchpad\h3\probe\*.txt` と `*.out`）:

| 組 | 中身 | 結果 |
|---|---|---|
| B2（第 1 候補） | torch 2.13.0+cu130・torchvision 0.28.0+cu130・comfy-kitchen 0.2.37・triton-windows 3.7.1.post27・sage post6 cu130・transformers 5.14.1・safetensors 0.8.0・accelerate・einops・av・diffsynth 2.1.9 | 成功・62 パッケージ。huggingface-hub 1.33.0・tokenizers 0.22.2・numpy 2.5.4 |
| A2（第 2 候補） | 上の torch・torchvision・triton を 2.14.1／0.29.1／3.8.0.post29 に変えたもの | 成功・62 パッケージ |
| A／B | transformers 5.20.1 の版 | 成功・63 パッケージ（huggingface-hub 2.2.0 になる） |
| C | A から diffsynth を除いたもの | 成功・44 パッケージ |

- `uv pip compile --python-platform windows` でも B2 を解決できた（`probe\B2.lock.txt`、199 行。ハッシュは付けていない）。
- B2 の全解決結果: accelerate 1.15.0, annotated-doc 0.0.5, anyio 4.15.1, av 19.0.1, certifi 2026.7.22, cffi 2.1.1, charset-normalizer 3.5.2, click 8.5.0, colorama 0.4.6, comfy-kitchen 0.2.37, cryptography 50.0.2, diffsynth 2.1.9, einops 0.8.2, filelock 4.2.0, fsspec 2026.9.0, ftfy 6.3.1, h11 0.16.0, hf-xet 1.7.0, httpcore 1.0.9, httpx 0.28.1, huggingface-hub 1.33.0, idna 3.20, imageio 2.38.1, imageio-ffmpeg 0.6.0, jinja2 3.1.6, markdown-it-py 4.2.0, markupsafe 3.0.4, mdurl 0.1.2, modelscope 1.40.2, modelscope-hub 0.4.6, mpmath 1.3.0, networkx 3.7, numpy 2.5.4, packaging 26.3, pandas 3.0.6, peft 0.21.2, pillow 12.3.0, psutil 7.2.2, pycparser 3.11, pygments 2.21.0, python-dateutil 2.9.0.post0, pyyaml 6.0.3, regex 2026.9.29, requests 2.34.2, rich 15.0.0, safetensors 0.8.0, sageattention（URL）, setuptools 84.0.0, shellingham 1.5.4, six 1.17.0, sympy 1.14.0, tokenizers 0.22.2, torch 2.13.0+cu130, torchvision 0.28.0+cu130, tqdm 4.70.1, transformers 5.14.1, triton-windows 3.7.1.post27, typer 0.27.3, typing-extensions 4.16.0, tzdata 2026.5, urllib3 2.8.0, wcwidth 0.9.2。
- 本番の手順に移すときは、既存の `Ensure-EngineVenv` と同じ形（freeze ファイル＋直接ピン＋`--index …/cu130 --index-strategy unsafe-best-match`）で書ける。【推測】`--no-sources` などの細部は、既存の `$ltx25UvArgs` に合わせる。

### 2-8. README に書く利用者側の要件

- **ドライバ: R580 以上**（CUDA 13.x のマイナー版互換の下限）。既存の README の推奨（R570 以上）より上がる。開発機の 610.88 は満たしている。サブマシンは未確認。
- **GPU: Turing（RTX 20 系、sm_75）以降**。CUDA 13 で Maxwell・Pascal・Volta が削除されたため。int8 の高速な行列積（CUTLASS）は Ampere（sm_80）以上で、Turing では別のカーネルか cuBLAS で動く。
- **ディスク**:
  - 【事実】torch 2.13.0+cu130 の wheel は 1.92GB、2.14.1+cu130 は 1.99GB（圧縮時）。参考までに cu128 の 2.9.1 は 2.86GB。既存の `.venv-engine-ltx25` の `torch` フォルダは展開後 4,350MB、venv 全体で 5,112MB（`du` で測定）。
  - 【事実】torch 以外の PyPI の wheel は合計 239MB（圧縮時。大きい順に triton-windows 53MB・comfy-kitchen 49MB・imageio-ffmpeg 31MB・av 28MB）。
  - 【推測】展開後は torch が約 3.0GB（cu128 の圧縮比 1.52 を当てはめた値）、その他が約 0.6GB（comfy-kitchen は AMD 用の pyd 133MB を含むので展開後に大きくなる）で、**新しい venv は約 3.5〜4GB**。uv のキャッシュは `.uv_cache`（プロジェクト内。`install_ltx.ps1:154`）にあって venv と同じドライブなので、venv はキャッシュへのハードリンクになり、二重には数えない。既存の venv とは torch が違うので共有は効かず、この 3.5〜4GB がそのまま増える。
- vcredist（Visual C++ 再頒布可能パッケージ）: triton-windows の README が必須としている。既存の venv も triton-windows を使っているので、新しい要件ではない。

### 2-2 追加（監督からの追加確認。block swap の先読みとの噛み合わせ）

**結論**: 量子化テンソルの型（`QuantizedTensor`）は、本製品の先読みの「素のテンソルとしてバイト列に詰める」処理と噛み合わない。一方で、**素の int8 の重み・F32 のスケール・bf16 の活性化をそのまま受け取る関数 `comfy_kitchen.int8_linear` が公開 API としてある**。中で選ばれる CUDA カーネルは、量子化テンソル経由のときと同じものになる。よって、「重みは素の int8 の nn.Parameter、スケールはバッファ、Linear の forward を差し替える」という既存の形のまま、comfy-kitchen のカーネルだけを借りられる。

**(a) `QuantizedTensor` の型と、各操作の振る舞い**
- 【事実】**ラッパー型のサブクラス**である。`__new__` が `torch.Tensor._make_wrapper_subclass(cls, params.orig_shape, device=qdata.device, dtype=params.orig_dtype)` を返し、int8 の本体は `self._qdata`、スケールなどは `self._params`（凍結した dataclass）に属性として持つ（`tensor/base.py:157-180`）。外から見える形と dtype は **元の形・元の dtype（bf16）** で、int8 ではない。`__torch_dispatch__` は `:346-367` にあり、まず汎用の処理表（`_DISPATCH_TABLE`、`:484-495`）を引き、次に層の種類ごとの表を引き、どちらにも無ければ**逆量子化してから元の演算を行う**（`_dequant_and_fallback`、`:369-372`）。`__torch_function__` は定義していない（ライブラリ全体に該当する文字列がない）。テストの `tests/test_tensor.py:364-371`（`qt + 1.0`）と `:1101-1108`（`reshape`）は、後戻り経路の結果が普通のテンソル（量子化テンソルではない）になることを確かめている。
- 先読みが使う操作ごとの振る舞い（【事実】はソースの分岐、【推測】はそこからの推論）:

| 操作 | 振る舞い | 根拠 |
|---|---|---|
| `p.data`（取得） | `aten.detach` → `_handle_detach` で量子化テンソルのまま返る | `base.py:409-410, 485` |
| `.reshape(-1)` | 処理表に無い → **逆量子化した bf16 の普通のテンソル**になる | `:347-372`。テスト `test_tensor.py:1101-1108` |
| `.view(torch.uint8)` | 処理表に無い（`aten.view.dtype`）→ 同じく逆量子化した bf16 のバイト列になる【推測】 | `:347-372` |
| `.data_ptr()` | **int8 本体のポインタ**を返す（上書き） | `:266-267` |
| `.numel()`・`.element_size()` | 上書きなし → **論理上の形（N×K）と bf16 の 2 バイト**を返す【推測】。int8 本体の実バイト数（N×K×1）とは合わない | `:157-169`（外側の形と dtype の宣言） |
| `.nbytes` | 上書きで int8 本体のバイト数を返す | `:201-209` |
| `.to(device)` | `_handle_to`：本体とスケールを移した新しい量子化テンソルを返す。`non_blocking` は捨てられて同期になる | `:417-436` |
| `module.to(device)` | torch 2.13 の `Module._apply` は、ラッパー型（`is_traceable_wrapper_subclass`）なら `swap_tensors` に切り替える（torch v2.13.0 `torch/nn/modules/module.py:953-975`）。`__tensor_flatten__`／`__tensor_unflatten__` は実装済み（`base.py:313-344`）なので、この経路で移る【推測】 | 同左 |
| `p.data = 量子化テンソル` | `aten._has_compatible_shallow_copy_type` は True を返すので通る【推測】 | `:494` |
| `copy_` | コピー元も同じ型の量子化テンソルなら、本体とスケールを `copy_` する（`non_blocking` は効く）。コピー元が普通のテンソルなら `TypeError` | `:453-466` |
| `pin_memory()` | 処理表に無い → 逆量子化した bf16 が pin される（容量 2 倍で量子化も失われる）【推測】。`is_pinned()` は本体に委ねる | `:269, 347-372` |

- 【推測・帰結】本製品の先読み（`tensor.data.reshape(-1).view(torch.uint8)`・`data_ptr()`・`numel()*element_size()`）に量子化テンソルを渡すと、良くて「bf16 に戻された重みを 2 倍の容量で運ぶ」、悪ければ「バイト数（bf16 換算）と実体（int8 のポインタ）が食い違う」。どちらの場合も使えない。先読みの特別扱いに `QuantizedTensor` を足す手もあるが、(b) の方法なら特別扱いそのものが要らない。

**(b) 量子化テンソルを使わない、関数レベルの公開 API**

| 関数 | 引数 → 戻り値 | 場所 | 用途 |
|---|---|---|---|
| **`comfy_kitchen.int8_linear`** | `(x, weight, weight_scale, bias=None, out_dtype=None, convrot=False, convrot_groupsize=256, input_act=None, input_act_weight=None, input_act_eps=0.0, residual=None, residual_scale=None)` → `Tensor [..., N]`（out_dtype の既定は bf16） | `__init__.py:997-1059`（公開。`__all__` に入っている） | **本命**。素の int8 の重み [N,K]・F32 のスケール（スカラーか [N] か [N,1]）・bf16 の x [...,K] を受け取り、活性化の回転・動的な int8 化・int8 の行列積・スケールの戻しを 1 回で行う。量子化テンソルの経路（`tensor/int8.py:287-295`）が最終的に呼んでいる関数と同じ（registry で `int8_linear` の実装を選ぶ） |
| `torch.ops.comfy_kitchen.int8_linear` | `(x, weight, weight_scale, bias, output_dtype_code, convrot, convrot_groupsize, input_act, input_act_weight, input_act_eps, residual, residual_scale)` → Tensor。dtype の番号は f32＝0・f16＝1・bf16＝2 | `backends/eager/quantization.py:1230-1268`（fake 実装つき） | 上と同じ処理を torch のカスタム op として呼ぶ形。`torch.compile` に通すならこちら |
| `comfy_kitchen.quantize_int8_rowwise` | `(x, stochastic_rounding=0)` → `(int8 [M,K], f32 [M,1])` | `__init__.py:936-943` | 行ごとの動的 int8 化だけ（回転なし） |
| `comfy_kitchen.mm_int8` | `(a, b)` → `a @ b` | `__init__.py:951-953` | eager の `torch._int_mm` だけ。CUDA 拡張は使わない |
| （非公開）`comfy_kitchen.backends.cuda.quantize_int8_rowwise_convrot` | `(x_2d, group_size, stochastic_rounding=0)` → `(int8, f32 [M,1])` | `backends/cuda/__init__.py:1379-` | 回転と量子化の融合カーネル（256 のみ） |
| （非公開）`…cuda.quantize_int8_rowwise_convrot64` | `(x_2d, group_size, stochastic_rounding=0, input_act=None, …)` → 同上 | `:1588-` | `int8_linear` の中で実際に使われる融合版 |
| （非公開）`…cuda.quantize_and_rotate_rowwise` | `(x, h, group_size, stochastic_rounding=0)` → 同上 | `:1689-1697` | 回転の行列積の後に量子化（融合されない経路） |

- int8×int8 の行列積にスケールを戻す処理だけを取り出した公開関数はない。CUTLASS の `_C.cutlass_int8_dequant` などは `_C` の内部関数で、呼べはするが公開されていない。→ **`int8_linear` を丸ごと使う**のが、公開 API の範囲で取れる形になる。
- 【事実・注意点】
  - CUDA 版の制約は、引数のテンソルごとに機器（cuda）を確かめる（`constraints.py:170-212` と `:216-248`）。**重みが CPU にあると cuda バックエンドは選ばれず**、triton か eager に落ちる。eager は重みを `weight.to(device=x.device)` で毎回 GPU に移す（`backends/eager/quantization.py:1028`）。→ forward の時点で、重みとスケールが GPU 上にそろっている必要がある。先読みが「本体が届く前に view を差し込み、イベントで待つ」設計なら、この条件は満たせる。
  - スケールが f32 で同じ機器・連続なら、そのまま使われる。そうでなければ、毎回 `.to(device, float32).reshape(-1)` がかかる（`backends/cuda/__init__.py:528-531`）。→ スケールのバッファもブロックと一緒に GPU へ運ぶ。
  - カーネルは `torch.cuda.current_stream(x.device)` に積まれる（`:2056`）。既存の先読みの「転送用のストリーム → イベント → 計算用のストリーム」と同じ考え方で扱える。
  - `int8_linear` はリストに挙げた引数しか制約を確かめない。`input_act`・`residual` は CUDA 版の制約には無いが、CUDA 版の実装はどちらも受け付ける（`:2035-2071`）。H3 の fc2 の前の swiglu（`input_act="swiglu"`）は融合カーネルに畳み込める【事実・docstring `__init__.py:1021-1027`】。ただし、DiffSynth が H3 で使っているかは未確認。

**(c) 「regular」型のアダマール行列 H と回転の手順**
- 【事実】`comfy_kitchen/tensor/int8_utils.py:11-41` の `_build_hadamard(size)`。size は 4 のべき乗に限る。`H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]` から始め、`torch.kron(h, H4)` を size に達するまで繰り返し（256 なら H4⊗H4⊗H4⊗H4）、最後に `/ sqrt(size)` で正規化する。`backends/eager/convrot_w4a4.py:15` にも同名の関数がある。
- 【推測・計算】H4 は対称行列なので、そのクロネッカー積の H も対称で、H＝Hᵀ、H·Hᵀ＝I。
- 【事実】重みの回転（保存時）は `_rotate_weight`：入力次元を group_size ずつのグループに分けて `W_grouped @ Hᵀ`（`int8_utils.py:44-56`）。その後、行ごとに int8 化する（`backends/eager/quantization.py:899-913` の `quantize_int8_convrot_weight`）。→ 保存されている重みは「回転済みの W·Hᵀ を出力チャネルごとに int8 化したもの」で、`weight_scale [N,1]` はその行ごとのスケール。
- 【事実】活性化の回転（実行時）は `_rotate_activation`：同じくグループに分けて `x_grouped @ H`（`int8_utils.py:58-74`）。eager の `int8_linear` は `_build_hadamard` と `_rotate_activation` を呼んでから量子化する（`backends/eager/quantization.py:1036-1042`）。CUDA 版は、条件が合えば融合カーネル `quantize_int8_rowwise_convrot64`、合わなければ `_build_hadamard` と `quantize_and_rotate_rowwise` を使う（`backends/cuda/__init__.py:2119-2150`）。
- 【推測・計算】(x·H)·(W·Hᵀ)ᵀ ＝ x·H·H·Wᵀ ＝ x·Wᵀ（H が対称かつ直交なので H·H＝I）。保存形式の重みをそのまま使い、活性化にだけ x·H を掛ければよい。逆回転は、bf16 に戻す（逆量子化）ときにだけ要る（`dequantize_int8_convrot_weight`、`backends/eager/quantization.py:916-`）。

**(d) CUDA 拡張の経路が選ばれる条件と、Ada（sm_89）で選ばれるカーネル**
- 【事実】登録の条件（すべて満たすこと）: CUDA 拡張 `_C` が読めた（`backends/cuda/__init__.py:124-157`）、`torch.cuda.is_available()`（`:4542`）、cuBLASLt（`cublasLt64_13.dll`）が読めた（`_CUBLASLT_AVAILABLE`、`:232` と `:4463`）。
- 【事実】呼び出しごとの制約（`:4464-4488`）: compute capability 7.5 以上・x は f32／f16／bf16 で 2 次元以上で cuda 上・weight は int8 の 2 次元で cuda 上・weight_scale は float32 で cuda 上・bias は f32／f16／bf16・out_dtype は f32／f16／bf16・convrot は bool・convrot_groupsize は int。groupsize の値と K の割り切れは、登録の条件には入らない（実装の中で経路を分ける）。
- 【事実】実装の中の分岐（`int8_linear`、`:2035-2296`）と、sm_89 で通る経路:
  1. m＝1（1 行）で、convrot・groupsize 256・K が 256 の倍数・256≤K≤16384・共有メモリに収まる場合 → `_C.int8_linear_m1`（回転・量子化・GEMV を融合、`:2084-2128`）。H3 の推論ではほぼ通らない。
  2. 活性化の量子化: convrot・groupsize 256・K が 256 の倍数・256≤K≤16384・共有メモリに収まる場合 → 融合カーネル `_C.quantize_int8_rowwise_convrot64`（`:2132-2146`）。それ以外の convrot → `_should_use_convrot_fused_kernel`（K≤5120 か K≥8192）なら `quantize_int8_rowwise_convrot`、さらに外れれば回転の行列積の後に `quantize_int8_rowwise`（`:2147-2153`）。convrot でなければ `_C.quantize_int8_rowwise`（`:2154-2165`）。
  3. m が 1 か 2 の小さい場合 → `_C.int8_gemv_dequant`（`:2167-2186`）。
  4. Turing（sm_75）用の分岐は、`_cuda_device_should_use_turing_kernels` が真のときだけ（`:2196-2210`）。sm_89 では通らない。
  5. **sm_89（と sm_86）での本線** → `_cuda_device_supports_cutlass_int8_dequant`（major≥8、`:321-332`）が真なので、**CUTLASS の int8 GEMM と逆量子化（行スケール×列スケール＋bias）の融合** `_C.cutlass_int8_dequant`（`:2228-2262`）を使う。residual が出力 dtype で 16 バイト境界にそろっていれば、`_C.cutlass_int8_dequant_residual` で残差の加算も畳み込む。タイルの構成は `cutlass_int8_selected_config(M,N,K)` が形から選ぶ（`ops/cutlass_gemm_int8.cu:193-229, 344`）。`_prefer_cublas_int8_fallback` は常に False（`:534-541`）。
  6. CUTLASS が失敗したとき、または環境変数 `COMFY_KITCHEN_DISABLE_CUTLASS=1` のとき → cuBLAS の int8 GEMM（int32 で累積）の後に別カーネル `dequantize_int8_linear`（`:2263-2294`）。
- 【推測・計算】H3 の DiT の 4 種の線形層（K＝5376／7168／14336）は、sm_89 でも sm_86 でも「2 の融合カーネル＋5 の CUTLASS 融合」の組み合わせになる（共有メモリの見積もりは §2-2 の本文）。テキストエンコーダの down_proj（K＝25600）は、2 の融合が外れて「回転の行列積＋量子化＋CUTLASS」になる。

---

## (3) 不確かな点と、実機で確かめないと分からない点

1. **comfy-kitchen の CUDA 拡張が実際に `int8_linear` を登録するか**。torch 2.13.0+cu130 の venv で `ck.list_backends()["cuda"]` を見て、`available=True` かつ capabilities に `int8_linear` があることを確かめる（GPU 不要、import だけで済む）。
2. **CUDA 経路で m×n ≥ 2³¹ が本当に通るか**。ソースでは int64 だが、試験はしていない。合成テンソルで m＝80,000・n＝28672・k＝5376 を 1 回流せば決着する（GPU 使用・要了承）。出力だけで 4.6GB あるので、16GB 機では行の分割が要るかもしれない。
3. **SageAttention post6（torch2.10.0andhigher）が torch 2.13.0（第 2 候補なら 2.14.1）で import でき、正しく計算するか**。本製品の `probe_sage()` と、自己検査（`engine25.sage_selfcheck25` 相当）で確かめる。
4. **ドライバ R580 未満の利用者の扱い**。サブマシンのドライバの版が未確認。R580 未満だと torch cu130 自体が CUDA を初期化できない（【推測】`torch.cuda.is_available()` が False になる）。
5. **block swap との噛み合わせ**。量子化テンソルの各操作の振る舞い（`pin_memory` と `reshape`／`view` が逆量子化に化けること、`numel`・`element_size` が bf16 換算になること、`.to(non_blocking=True)` が同期になること）は、ソースから読んだ結論で、実行はしていない。推奨する形（素の int8 の Parameter＋スケールのバッファ＋forward で `ck.int8_linear` を呼ぶ）で、既存の先読みにそのまま載ることと、CUDA 経路が選ばれること（重みが GPU 上にあるとき）を、小さな試作で確かめる必要がある。
6. **融合カーネルが実際に選ばれているか**。共有メモリの上限（Ada・Ampere とも 101,376 バイトのはず）は `torch.cuda.get_device_properties().shared_memory_per_block_optin` で確かめる。テキストエンコーダの down_proj（K＝25600）が融合されない経路で実用的な速度か。
7. **torch 2.13 と 2.14 の速度差**。第三者の報告は 2.13 だけで、2.14 の H3 での実測は見つからなかった。
8. **transformers 5.14.1 で DiffSynth の Qwen3-VL の組み立てがそのまま通るか**。DiffSynth が試験している transformers の版は明記されていない（pyproject は版を指定していない）。
9. **torchaudio が無いことの影響**。本製品の H3 の経路で音声の読み書きに何を使うか（av か、既存の mp4 書き出しか）は、設計の段階で決める。
10. **ディスクの見込み（約 3.5〜4GB）は推測**。実際に 1 回構築して測る必要がある。
11. comfy-kitchen は活発に更新されている（0.2.34→0.2.37 が 3 週間）。#136 の修正や Qwen 系の高速化（main の 10-13・10-14 のコミット）が次の版で入る可能性がある。版を固定したうえで、上げる時期を決める運用になる。
