# MiniMax H3 推論高速化・長尺化 事前調査メモ（オーナー共有・2026-10-19）

オーナーが別の調査で得た情報を、台帳 §1 の起票（§1-103〜）の出典として写したもの。内容は共有された原文のまま（監督は一次資料で裏取りしていない。各テーマの計画の冒頭で確かめる）。

## 1. 推論高速化

対象は、Nz-Videomniの現行MiniMax H3 INT8 ConvRot＋Low VRAM経路。Turbo LoRAによるstep数削減とは別に、**1 stepあたりの処理時間を短縮する**ことを目的とする。

現行のBlock Swap Prefetchは、pinned staging、非同期H2D、専用CUDA stream、arena ring、D2H eviction削除まで実装済みであり、大幅な追加改善余地は小さい。次は以下の5項目を検討する。

| 優先度 | 技術 | 要点 |
|---|---|---|
| **1** | **QKVのINT8 activation量子化の共有** | 最初に検討する。Low VRAM経路ではQ/K/Vをhead groupごとに`forward_rows()`しており、同じ入力activationに対するrow-wise INT8量子化・ConvRot前処理が繰り返されている可能性がある。activation preparationを1回だけ行い、その量子化済み入力を複数のQ/K/V GEMMで再利用できないか調査する。数学的なモデル構造は変更しない。 |
| **2** | **step間 cyclic block prefetch** | 現行Block Swap Prefetchは各denoise stepの末尾でwindowが縮み、次stepのblock 0はcold startになる。step Nの終盤でstep N+1のblock 0/1を先読みするcyclic prefetchを検討する。計算結果には影響しない。 |
| **3** | **head group 8→28/56** | Low VRAM経路では56 attention headsを現在8 headsずつ処理している。実測では28 heads/groupが8 heads/groupより約5.5%高速だった。一方でVRAM peakは増えるため、生成可能な解像度・尺とのトレードオフになる。56＝分割なしも試験候補。安全側の既定値8は維持してもよい。 |
| **4** | **CPU→pinned stagingの先読み** | 現行prefetchはpinned RAM→VRAMを非同期化しているが、CPU master→pinned stagingのpack/memcpyは同期処理。GPUがblock Nを計算中にCPU threadでblock N+1をpinned bufferへ準備できれば、host側転送も隠せる可能性がある。 |
| **5** | **RMSNorm / RoPE / SwiGLU等のkernel fusion** | GEMM以外のkernel launchと中間tensor read/writeを削減する。特にQ/KのRMSNorm＋RoPE、FFNのSwiGLUが候補。Comfy-Kitchenには`rms_rope`系のfused kernelが既に存在する。大幅な高速化より数%単位の積み重ねを狙う。 |

### 参考URL

- Nz-Videomni Low VRAM実装: https://github.com/Rootport-AI/Nz-Videomni/blob/main/engine_h3/lowvram.py
- Nz-Videomni Block Swap Prefetch: https://github.com/Rootport-AI/Nz-Videomni/blob/main/engine/transformer/block_swap_prefetch.py
- Nz-Videomni H3定数・head group設定: https://github.com/Rootport-AI/Nz-Videomni/blob/main/engine_h3/constants.py
- Comfy-Kitchen INT8実装（Activationはdynamic per-row INT8 quantization。QKV activation共有を検討する際の主要参照先）: https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8.py
- Comfy-Kitchen fused kernels（`rms_rope`、`apply_rope`等の公開API）: https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/__init__.py

### 当面の検討順

**QKV activation量子化共有 → cyclic prefetch → CPU→pinned先読み → kernel fusion**を優先する。head group拡大は既に高速化効果が確認できているが、VRAMとの交換条件になるため、独立したFast profile候補として扱う。これらはいずれもTurbo LoRAによるstep数削減と基本的に直交するため、将来的に併用できる。

## 2. MiniMax H3長尺化

MiniMax H3には、LTXのような「Continuation専用checkpoint」が最初から存在するわけではない。しかし、**生成済みvideo/audio latentをdecode→encodeせず保持し、次区間のconditioningまたは固定prefixとして再利用する**ことで、LTX Chainedに近い潜在空間での動画連結が可能になっている。重要なのは、「H3の重みにContinuation能力が追加された」のではなく、**推論エンジン側が既存AV latentの一部を保存・再配置・固定できるようになった**という点である。

### 2.1 H3 Continuum

H3 Continuumは、MiniMax H3で長尺動画を生成するComfyUI系実装。現在はFL2VA / Ref2VAの双方を扱えるが、**latent continuation自体はRef2VA専用ではない**。FL2VAでもNative Masked Continuationが実証されている。

基本構造: Clip A → 生成済みVideo / Audio latentの末尾を保存 → Clip Bの冒頭へ配置 → 引き継いだlatentを保護 → それ以降だけ新しくdenoise → 重複区間をtrim → Clip A + Clip B。

Native Masked方式では、引き継いだVideo / Audio prefixをdenoise対象から外せるため、最終フレーム1枚だけをI2V入力する方式とは異なり、**動き・質感・音声の時間的連続性をlatent単位で維持できる**。FL2VAでも利用可能なので、Nz-VideomniでChainedを実装するためにRef2VAを先に完成させる必要はない。Ref2VAを併用する利点は別にあり、各chunkで人物・衣装・背景・動画・音声referenceを再投入することで、長いchainにおけるsemantic driftを抑えられる。役割分担: 短期的な連続性 = AV latent continuation／長期的な一貫性 = Ref2VA reference。

- H3 Continuum: https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum
- CHANGELOG（FL2VA Native Masked continuationのGPU検証やFL2VA / Ref2VA対応状況）: https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum/blob/main/CHANGELOG.md
- User Guide（Continuation Backend、Masked AV、Reference Context等）: https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum/blob/main/docs/USER_GUIDE.md

### 2.2 vLLM-Omni Long Video

vLLM-OmniにもMiniMax H3専用のLong Video Continuation実装がある。こちらは現在**Ref2VA専用**であり、H3 ContinuumのNative Masked prefixとは内部方式が少し異なる。一定長のwindowをoverlapさせながら生成し、前windowの末尾video/audio latentを次windowへ渡し、新しく生成されたsuffixだけを累積latentへ追加する。前windowのlatent tailを**次windowの固定target prefixとしてmaskするのではなく、追加のlatent guide conditioningとして投入する**。さらに各windowへ`temporal_offset`を与え、RoPE上の時間座標を動画全体で連続させる（動画全体を一つのglobal temporal timelineとして扱う）。

Nz-Videomniでは、まずH3 Continuum型のFL2VA latent continuationを実装し、その後Ref2VA対応時にvLLM-Omni型の**rolling window + latent tail guide + global temporal RoPE**を研究する価値がある。

- vLLM-Omni MiniMax H3 recipe: https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md
- MiniMax H3 continuation実装（window分割、overlap、AV latent tail、`temporal_offset`）: https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/minimax_h3/continuation.py

## 結論

推論高速化では、**QKVのINT8 activation量子化共有**を最優先の調査対象とする。長尺化では、**Ref2VAを待たずFL2VAでH3 Continuum型latent continuationを実装可能**。その後Ref2VAを追加し、latent tailによる短期連続性とreferenceによる長期一貫性を組み合わせる。さらに長尺化する場合は、vLLM-Omniの**rolling window＋global temporal RoPE**を次段階の研究対象とする。
