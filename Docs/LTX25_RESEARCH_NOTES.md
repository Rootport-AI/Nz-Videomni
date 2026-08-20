# LTX 2.5 事前調査ノート（参考資料）

## 本書の位置づけ

**オーナーによる2026-08-20共有の事前調査である。マルチエンジン設計（ヘッダーのドロップダウンによるベースモデル切り替え）の議論より前に行われた調査であり、設計資料ではなく参考資料として収蔵する。**

**設計の正本は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md)である。** 両者の記述が食い違った場合は設計正本を採ること。本書は「LTX 2.5 とはどういうモデルで、LTX 2.3 と何がどれだけ違うのか」という技術的な下調べであり、実装の進め方・責任分界・UI仕様については本書ではなく設計正本を参照する。

- 収蔵日: 2026-08-20
- 収蔵時の扱い: **本文は内容を改変せずそのまま収める**（整形は見出しレベルの調整など最小限）。数値の見積り（再利用率のパーセンテージ等）は調査時点の概算であり、実装で裏取りされたものではない。
- 起票先: [`PENDING_TASKS.md`](PENDING_TASKS.md) §3-98

---

## 結論

LTX 2.5対応は実現可能で、プロジェクトの全面書き換えは不要です。

LTX 2.3と2.5は、joint audio-video DiT、latent構造、`8n+1`フレーム規約、8-step＋3-stepの二段生成、AV latentの結合・分離、I2V conditioningなど、推論パイプラインの約70〜78%が共通しています。

AviUtl2プラグイン、WebUI、API、ジョブ管理、動画I/O、chainのvideo geometryは大部分を再利用できます。新規対応が必要なのは、主にモデル境界と低VRAM実行部分です。

## 使用する公式重み

変換元には、これを使います。

```text
ltx-2.5-22b-distilled-transformer-bf16.safetensors
```

これはQ4ではなく、非量子化BF16の8-step蒸留済みTransformerです。

以下は使用しません。

```text
ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
```

こちらはINT8＋ConvRot量子化済みで、Q4ではありません。GGUFへ再量子化するには不向きです。

変換経路は次の形です。

```text
公式LTX 2.5 BF16 distilled Transformer
→ LTX 2.5対応版Nz-GGUF converter
→ 独自GGUF Q4
```

## Nz-GGUF converterの改造

現在のコンバータから再利用できるものは、

- safetensorsの一tensor単位ストリーミング
- BF16/F16/F32読み込み
- GGUF writer
- Q4_K/Q5_K/Q6_K量子化kernel
- shape変換
- 構造検証の基盤

です。

LTX 2.5向けに作り直すものは、

- tensor key／prefixの判定
- 2.5専用typemap
- block数・shape・architecture metadata
- Q4化するtensorと高精度維持tensorの分類
- GGUF metadata
- backend loaderとのkey／shape一致
- 2.5 Transformer model shell

です。

つまりコンバータそのものを作り直すのではなく、LTX 2.5用profileと検証規則を追加します。

## 推論パイプライン

LTX 2.5でも基本構造は維持されます。

```text
Gemma 4 text encode
→ video/audio latent生成
→ AV latent結合
→ stage 1 joint AV denoise（8 steps）
→ AV latent分離
→ video latent x2 upscale
→ AV latent再結合
→ stage 2 refinement（3 steps）
→ video/audio latent分離
→ tiled video decode
→ audio decode
→ mux
```

LTX 2.3から変わる主要部分は、

- Gemma 3からLTX向けGemma 4＋projectionへ変更
- all-in-one checkpointからsplit componentsへ変更
- 通常CFGからvideo/audio Dual CFGへ変更
- EulerからEuler ancestralへ変更
- video decoderをConv VAE／DiffVAEへ変更
- 2.5 checkpoint metadataに基づくモデル構築

です。

Transformerは完全な別アーキテクチャとして書き直すのではなく、2.5 checkpoint schemaへloaderとadapterを対応させます。VAEとGemmaは新しいadapterへ置換します。

## プロジェクトの再利用範囲

概算では次の程度を再利用できます。

| 領域 | 再利用見込み |
|---|---:|
| AviUtl2プラグイン／HTTP／timeline配置 | 90〜100% |
| API／ジョブ／成果物管理 | 85〜95% |
| React UI／進捗監視 | 75〜90% |
| model registry | 65〜80% |
| video chain geometry | 60〜80% |
| two-stage orchestration | 55〜75% |
| GGUF converter基盤 | 50〜70% |
| sampler／guider | 25〜50% |
| Gemma text encoder | 10〜30% |
| DiffVAE | 10〜25% |

製品全体では、およそ55〜70%を直接または軽微な変更で再利用できます。

## VRAM 16GB向け実行方式

最初から低VRAM専用の`LTX25LowVramPipeline`を新設します。

```text
LTX25LowVramPipeline
├─ Gemma 4 layer streaming
├─ Q4 Transformer CPU resident
├─ DiT block swap
├─ per-layer GGUF dequant
├─ stage-boundary release
├─ latent upscaler
├─ tiled/chunked video decode
└─ sequential audio decode
```

GPUにはモデル全体を置きません。

### Text encode

- Gemma 4全体はCPUに保持
- 現在処理する1〜2層だけGPUへ転送
- encode完了後はGemmaをGPUから解放
- video/audio conditioningだけ保持

### Stage 1／Stage 2

- Q4 Transformer全体はCPUに保持
- 現在実行するDiT blockだけGPUへ転送
- Linearごとに必要なweightだけ一時的にBF16へdequant
- inactive blockはCPUへ戻す
- Q4全体やBF16 state dictをGPUへ展開しない

### VAE decode

- denoise完了後、DiTをGPUから完全に退避
- 初期実装は軽量Conv VAEを使用
- 空間tile＋時間chunk単位でdecode
- 全フレームをGPU上へ同時展開しない
- 基本生成が安定した後、同じinterfaceへDiffVAEを追加

### Audio

- video decode終了後にaudio VAE／vocoderをロード
- video VAEとaudio decoderを同時常駐させない

要するに、

> LTX 2.3版の外周と低VRAM設計を維持し、公式BF16重みからLTX 2.5専用GGUFを生成する。Gemma 4、2.5 Transformer adapter、Conv VAE／DiffVAE adapterだけを新設し、最初からCPU resident・block swap・layer streaming・tiled decodeで動かす。

これが、このプロジェクトをLTX 2.5対応させてVRAM 16GBで動かすための最短ルートです。

## 参考文献

- [LTX-2.5公式モデルページ](https://huggingface.co/Lightricks/LTX-2.5) — BF16／INT8 ConvRot／NVFP4、Gemma 4、VAEなどの公式配布物。
- [LTX-2 v1.2.0リリース](https://github.com/Lightricks/LTX-2/releases/tag/v1.2.0) — LTX 2.5対応、split checkpoint、Gemma 4、DiffVAEなどの変更点。
- [LTX-2公式コア・アーキテクチャ](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core) — joint AV DiT、latent構造、空間・時間圧縮規約。
- [LTX公式Two-Stage Generation](https://docs.ltx.io/open-source-model/usage-guides/two-stage-generation) — 8-step生成、latent upscale、3-step refinementの公式構成。
- [LTX公式ComfyUI統合ガイド](https://docs.ltx.io/open-source-model/integration-tools/comfy-ui) — LTX 2.5のノード構成とcomponent配置。
- [LTX 2.3公式ComfyUI T2V workflow](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_ltx2_3_t2v.json) — 2.3の8＋3-step推論グラフ。
- [LTX 2.5公式ComfyUI T2V workflow](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_ltx2_5_t2v.json) — 2.5のsplit Transformer、Gemma 4、Dual CFG、DiffVAE構成。
- [LTX 2.3公式ComfyUI I2V workflow](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_ltx2_3_i2v.json) — stage 1／2の画像conditioning再注入。
- [LTX 2.5公式ComfyUI I2V workflow](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_ltx2_5_i2v.json) — 2.3とのI2V契約比較用。
- [LTX公式最適化ガイド](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/optimization.md) — offload、FP8/NVFP4、DiffVAE chunking、stage間メモリ解放。
- [ComfyUI-LTXVideo low-VRAM loader](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/low_vram_loaders.py) — componentを順次ロードして同時常駐を避ける公式実装。
- [LTX公式ComfyUIカスタムノード](https://docs.ltx.io/open-source-model/integration-tools/ltx-comfy-ui-nodes) — conditioning保存、Gemma API encode、空間・時間tiled VAE decode。
- [ComfyUIのLTX Transformer実装](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/lightricks/model.py) — LTXモデルのロード、block構造、forward実装。
- [Comfy公式LTX量子化資料](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/quantization/ltx2.md) — LTXの量子化対象、除外tensor、構造検証の参考資料。
- [Comfy公式INT8 Tensorwise形式](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/int8_tensorwise.md) — INT8 ConvRotとGGUF Q4が別形式であることの根拠。
- [Comfy公式ConvRot実装](https://github.com/Comfy-Org/comfy-quants/blob/main/src/comfy_quants/formats/convrot.py) — Hadamard rotationとINT8量子化処理。
- [Comfy公式INT8 ConvRot変換ツール](https://github.com/Comfy-Org/comfy-model-tools/blob/main/quant_int8_convrot.py) — dense BF16からINT8 ConvRotを生成する公式実装。
