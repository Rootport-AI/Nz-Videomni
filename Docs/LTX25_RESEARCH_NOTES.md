# LTX 2.5 事前調査ノート（参考資料）

## 本書の位置づけ

**オーナーによる2026-08-20共有の事前調査である。マルチエンジン設計（ヘッダーのドロップダウンによるベースモデル切り替え）の議論より前に行われた調査であり、設計資料ではなく参考資料として収蔵する。**

**設計の正本は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md)である。** 両者の記述が食い違った場合は設計正本を採ること。本書は「LTX 2.5 とはどういうモデルで、LTX 2.3 と何がどれだけ違うのか」という技術的な下調べであり、実装の進め方・責任分界・UI仕様については本書ではなく設計正本を参照する。

- 収蔵日: 2026-08-20
- 収蔵時の扱い: **本文は内容を改変せずそのまま収める**（整形は見出しレベルの調整など最小限）。数値の見積り（再利用率のパーセンテージ等）は調査時点の概算であり、実装で裏取りされたものではない。
- 起票先: [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-98（2026-08-22にv1のスコープでクローズし、[`PENDING_TASKS.md`](PENDING_TASKS.md)から移設した。後続として引き取った§3-102も2026-09-01にクローズして同書§3-102になり、拡散デコーダ版の映像VAEの採否は[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-33へ移った。**生きている後続課題の一覧は同書§3が唯一の正本である**——本書には件数も項目番号も書かない）

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

> **2026-08-21更新:** 本節は2026-08-20時点では「これから作り直すもの」として書かれているが、その後の改造は完了し、2026-08-21付でtransformerのGGUF再変換・全検証合格まで達成している。詳細は本書末尾の「2026-08-21追加調査」の1節を参照。

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

---

## 2026-08-21追加調査

2026-08-20収蔵の上記本文に続き、2026-08-21に追加で実施した調査の結果を本節に記す。上記本文は無改変のまま収めており、本節はそれを踏まえた追加の事実確認という位置づけである。既存記述と矛盾する箇所には「2026-08-21更新:」を付した注記を該当箇所（「Nz-GGUF converterの改造」節末尾）にも挿入している。

### 1. transformerのGGUF再変換が完了

結論から言うと、2026-08-21にtransformerのGGUF再変換が完了し、全検証に合格した。従来から判明していた不具合、すなわちconnector（audio/video embeddingsをtransformerへ橋渡しする部分）の258テンソルが欠落し、本来あるべきテンソル数に対し4,091テンソルしか含まれていない不良GGUFが生成されていた問題を修正したスクリプトで、再変換を実施した結果である。

出力先は`Nz-GGUF-Converter-LTX23\output\LTX-2.5-22B-distilled-transformer.gguf`で、サイズは14,738,670,368バイト、SHA-256は`4ead2a7dbae374639794b717517a3d1938bb8a16629957467e617ee99f22cf98`。

実測のテンソル総数は4,349（transformer本体4,091＋audio/video embeddings connectorが各129で計258）。型内訳はQ4_K（4ビット量子化、Kブロック方式）が1,658、BF16（16ビット浮動小数点）が2,401、F32（32ビット浮動小数点）が290。connector 258本については、入力safetensorsとの生バイト一致検証に2回（変換時の検証と、それとは独立したself-verify時の検証）合格している。pytestは124件全て合格。所要時間は約68分だった。

旧不良GGUFはオーナーが撤去済みで、変換スクリプトの修正自体もオーナーが手動でコミット・プッシュ済みである。本エージェントはDocs以外への書き込み権限を持たないため、コード側の変更確認は行わずオーナーからの報告事実として本節に記録するのみとする。

### 2. 公式パイプラインが実際にロードするファイル構成

結論として、公式実装コード（GitHub Lightricks/LTX-2の`ltx-pipelines`パッケージ）を確認したところ、実際にロードされるファイルは用途によって必要最小限まで絞り込めることが分かった。これは実装コードを直接読んで確認した事実であり、推測を含まない。

中核となるローダー`ModelPaths`が受け取る枠は、transformer・text encoder・video VAE・audio VAE・duration head（生成尺を予測するヘッド）の5つのみである。アップスケーラ（低解像度latentを2倍に引き伸ばすモデル）はこの5枠には含まれず、パイプラインごとのコンストラクタ引数として別扱いになっている。

単一ステージ生成（`ti2vid_one_stage`）は、transformer・text encoder・video VAE・audio VAEの4種だけで成立する。

一方、2段階系のパイプライン（distilled／two-stages／DFR〈Diffusion Fidelity Rendering、半解像度でまず生成し2倍にアップサンプルしてから本番解像度で再デノイズする方式〉）では、spatial upscaler（空間方向のアップスケーラ）が必須引数になる。これはLTX 2.3のstage-2の考え方をそのまま踏襲している。temporal upscaler（時間方向のアップスケーラ）は`temporal_upsample_rounds`が0より大きいときだけ必要になるオプション扱いである。

duration headは`--num-frames`を明示すれば不要になるオプション。distilled transformerを直接使う構成であればdistilled LoRAも不要。プロンプトエンハンサー（プロンプトを事前に拡張する仕組み）は既定でOFFのオプションで、使う場合のみ別途Gemmaディレクトリが必要になる。

text encoderファイル（gemma4-12b-with-proj）は、tokenizer・config・processor・text projectionを全部同梱した1ファイル完結構成であることが公式のinstallation.mdに明記されており、`--gemma-root`のような別ディレクトリ指定は不要。config情報は各safetensorsファイルの`__metadata__`にJSONとして埋め込まれており（LTX 2.3と同じ方式）、別途config.jsonを用意する必要もない。

video VAEは拡散デコーダー版とConv版の2択があるが、どちらもencoder＋decoderを同梱しておりi2v入力のエンコードに使える。デコーダ種別はファイル内metadataの`vae._class_name`から自動判別される。拡散デコーダー版はnatten（近傍注意を計算する専用ライブラリ）の使用が推奨され、反復的なデノイズステップを伴う。

主な出典はGitHub Lightricks/LTX-2の以下のファイル群である。`packages/ltx-pipelines/src/ltx_pipelines/utils/model_paths.py`、`utils/args.py`、`ti2vid_one_stage.py`、`ti2vid_two_stages.py`、`dfr_pipeline.py`、`distilled.py`、`packages/ltx-core/README.md`、`ltx_core/loader/sft_loader.py`、`docs/installation.md`（いずれも https://github.com/Lightricks/LTX-2 配下）。

### 3. text encoder（gemma4-12b-with-proj）の内部構造

結論として、公式のbf16版text encoderファイルのヘッダをHTTP Range取得で直接確認し、内部構造を確定させた。これは実測にもとづく事実であり、推測を含まない。

総テンソル数は686（BF16が681、U8が5）。`__metadata__`に含まれるのは`format`と`gemma_config`（JSON、3,583バイト）のみで、`gemma_config`内には`architectures: ["Gemma4UnifiedForConditionalGeneration"]`、`gemma_version: "gemma4-12b-ltx-v1"`という記述がある。

バックボーンは48層、hidden次元3840、head数16（KVヘッドは8）、head_dim 256、vocabサイズ262,144。sliding_attention（近傍のみを見る注意機構）の層とfull_attention（全トークンを見る注意機構）の層が交互に並び、full層はindex 5, 11, ... , 47の8層のみ。このfull層8つはk（key）とv（value）の重みを共有しておりv_projが存在しない（1層あたり13テンソル。他の40層は1層あたり14テンソル）。各層には`layer_scalar`というゲート（層の出力を調整する係数）がある。RoPE（回転位置埋め込み）のthetaはsliding層で10000、full層で1,000,000、かつfull層はpartial_rotary_factor=0.25。

条件付け信号（テキストをどう画像・音声生成に反映させるかの信号）の取り出し方が特殊で、通常のように最終層の出力だけを使うのではなく、49個のhidden_states（48層の出力＋埋め込み層の出力）をトークンごとにRMSNorm（Root Mean Square Normalization、正規化手法の一種）してからリスケールして結合し、`text_embedding_projection.video_aggregate_embed`（形状[4096, 188160]）と`audio_aggregate_embed`（形状[2048, 188160]）で射影する。188160という数字は3840×49であり、49個のhidden_statesを結合した次元に一致する。この構造は公式ltx-coreの`feature_extractor.py`内の`FeatureExtractorV2`で確認した。

ほかに`audio_projector.embedding_projection`（[3840,640]）、`multi_modal_projector.embedding_projection`（[3840,3840]）、vision tower（画像入力を処理する部分）9テンソルが同梱されている。

さらに、tokenizer等の付属データがU8型のテンソルとして埋め込まれている点も特徴的である。`tokenizer_json`（32.2MB）、`hf_asset__chat_template.jinja`など4点がこの方式で格納されている。

この構造が持つ含意は、llama.cpp（GGUF量子化・推論の標準実装）の標準gemma4対応はテキスト専用のvanilla構成向けであり、上記の49層抽出＋射影という構造を変換できないという点である。つまりGGUF化には自前の変換ロジックと、49層のhidden states抽出・射影を行う自前の推論経路の両方が必要になる。

### 4. GGUF化の既存事例と信頼性評価

結論として、gemma4-12b-with-projを既にGGUF化した事例は存在するが、そのまま流用するのは非推奨である。

`elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF`（Q5_K_M、9.51GB）というHugging Face上の配布物があり、テンソル会計（686個）は実ヘッダの数値と一致しており実際に変換処理を行った形跡はある。ただし投稿者が無名のアップローダーであり第三者による検証も存在しないため、自前実装への流用は避けるべきである。一方で、自前で変換した結果を独立に突き合わせる「オラクル（正解データ）」としての参照価値はある。

そのほか、`joeygambino/LTX-2.5-Quantized`にはComfyUI向けのw4a8版（10.6GB）が存在する。`realrebelai/LTX-2.5_GGUFs`はtransformerのみを含んでおり、text encoderへの導線はLTX 2.3時代の記述をそのまま流用しているため実質的に壊れている。

### 5. 「16GB VRAM問題」の原因切り分け

結論として、LTX 2.5をVRAM 16GBで動かすことを阻む構造上の困難は見当たらない。16GBという壁が語られる原因は、モデル自体の構造ではなく、エコシステム（周辺ツール群）側の未対応にあるというのが本調査の見立てである。

ComfyUI-LTXVideoのIssue #303で語られている「16GB VRAM問題」は、旧世代（無印LTX-2＋標準Gemma 3の組み合わせ）についての報告であり、CPUオフロードなどの回避策が提示された結果、報告者自身がその日のうちにissueを自己クローズしている。これは構造的な困難の証拠にはならない。

ComfyUIでgemma4が動かない原因は、ComfyUI本体側の配管バグである。Comfy-Org/ComfyUI Issue #15595によれば、検出済みのGEMMA_4_12B型がCLIPTypeのLTXV向けハードコードによって握りつぶされている。加えて、ComfyUI-GGUF側のアーキテクチャ・ホワイトリストにgemma4がまだ登録されていない。

量子化そのものを阻む構造的な証拠はない。公式がすでにint8-convrot版（15.4GB）を配布しており、品質劣化の報告も見当たらない。さらに、Lightricks公式のデスクトップアプリはlayer streaming（層単位でのメモリ転送）を実装しており、VRAM閾値を32GBから16GBへ引き下げ済みであることがLTX-Desktop Issue #16で公式スタッフにより回答されている。なお、公式が「エンコーダーの差し替えは品質劣化を招く」と発言している箇所（HF Discussion #34）は、別モデルへの交換についての話であり、同一モデルを量子化することについては否定していない。

49層のhidden statesを結合する構造が増やすのはactivationメモリ（推論時の中間結果が占めるメモリ）であり、512トークンの入力で200MB弱程度である。これは重みの量子化とは独立した話であり、量子化の可否には影響しない。

以上から、自前でのGGUF量子化と、LTX 2.3で実績のあるレイヤー単位の逆量子化ローダーをLTX 2.5へ適用することについて、構造的なリスクは低いと判断できる。

### 6. LTX 2.3資産の出自の確認とLTX 2.5への含意

結論として、現行LTX 2.3で使っている資産は、すべて既存の第三者配布物の流用であり、Lightricks公式のsafetensorsから自前で抽出したものはない。この事実は正本である`Nz-HF-Rehost\manifest.json`のupstream_repo欄で確認した。

内訳は、transformer GGUFがQuantStack/LTX-2.3-GGUF、Gemma 3 GGUFがggml-org/gemma-3-12b-it-GGUF、VAE・audio VAE・text projectionがKijai/LTX2.3_comfy、tokenizerがgoogle/gemma-3-12b-it-qat-q4_0-unquantized。自前で行った変換はPrunaVAEDのキーリネームのみである。

LTX 2.5では、公式が最初からコンポーネントを分割配布しているため、Kijai氏のような第三者による抽出物への依存が不要になる。一方でGemma 4はLTX専用にファインチューンされたモデルであるため、LTX 2.3のように汎用のGGUF配布物をそのまま流用することができない。つまり自前での量子化が必須になる。これがLTX 2.3との間の最大の条件差である。

### 7. 残課題

再ホスト（自前で作った量子化版をHugging Face等へ配布すること）を検討する場合、「LTX-2.x Community License」内のfine-tune transfer条項（派生モデルの再配布に関する条項）が量子化版の再配布に適用されるかどうかは、本文を未確認のため未確定である。再ホストを行う前に、一次ソースのLICENSE本文を直接確認する必要がある。

Gemma 4 text encoderの量子化設計についても、未決の論点が2つ残っている。1つは量子化対象の分類で、embed_tokens（[262144×3840]の巨大な埋め込みテーブル）と、同じく巨大なaggregate_embed×2（video/audio用の射影行列）を量子化するか、それとも精度維持のため温存するかという判断。もう1つはGGUFのテンソルキー形式で、LTX 2.3で採用したllama.cpp形式を踏襲するか、それともtransformerと同じ生のstate_dictキー方式に統一するかという判断である。

### 8. テキストエンコーダーGGUF変換の完了と成果物の所在

結論から言うと、2026-08-21にtext encoder（gemma4-12b-with-proj）のGGUF変換も完了し、全検証に合格した。7節末尾に残課題として書いた「量子化対象の分類」「テンソルキー形式」の2論点は、この変換で裁定・実装済みである。

出力先は`Nz-GGUF-Converter-LTX23\output\LTX-2.5-gemma4-12b-text-encoder-Q4_K_M.gguf`で、サイズは9,231,374,624バイト、SHA-256は`4e69e4a33065c856e039fb4c1b78adffae582c4da14b633224136a7e7e30fb90`。実測のテンソル総数は686で、型内訳はQ4_K（4ビット量子化）が328、Q6_K（6ビット量子化）が2、BF16（16ビット浮動小数点）が351、I8（8ビット整数、後述のsidecar用）が5。self-verifyとE6（K量子化の数値検証）は全合格した。

量子化ポリシーはオーナー裁定を経て次のとおり確定した。40層のsliding-attention層と8層のfull-attention層が持つ`nn.Linear`の重み328本（1層あたり7種、full層は`v_proj`を共有するため実質40×7＋8×6ではなく、7名の対象名を持つ層すべてから拾える328本）をQ4_Kにする。`aggregate_embed`（video用[4096, 188160]・audio用[2048, 188160]の2本）はQ6_Kにする——実測rel-RMSEがQ4_Kで0.0656、Q6_Kで0.0173となり、Q6_Kを採用した。`embed_tokens`（[262144, 3840]の埋め込みテーブル）はK量子化の条件を満たすがBF16のまま温存する——バックエンド側が埋め込みを読み込み時に必ずBF16へ全展開する実装のため、量子化してもVRAM削減効果がゼロで、リスクだけが残るという判断による。残りのnorm系・`layer_scalar`・biases・projector・vision towerはBF16のまま。

transformer側のGGUFの所在も再掲しておく。`output\LTX-2.5-22B-distilled-transformer.gguf`（14,738,670,368バイト、1節参照）。**このサイズとSHA-256はKVメタデータを足す前の出力のものである**——実際に配布・配置されている現物の値は10節の表が正本。

**両GGUFとも`models\LTX25\`へは未配置である。** `models\LTX25\Weights\`フォルダは実在するが中身は空で、`TextEncoder\`・`VAE\`フォルダはまだ作成されていない。配置作業はオーナーが行う。**（この段落は2026-08-21時点の記述で、いまは失効している——重みは10節のとおり再ホスト済みで、配置は`install-LTX25.bat`が行う。）**

残る4本の重み（safetensors形式）はGGUF化せず、そのまま使う方針である。`Nz-GGUF-Converter-LTX23\.artifacts\official-ltx25\`配下に検証済みでダウンロード済みになっている。

| ファイル | サイズ（バイト） | SHA-256 |
|---|---:|---|
| `vae\ltx-2.5-video-vae-conv-bf16.safetensors` | 1,452,269,922 | `685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d` |
| `vae\ltx-2.5-video-vae-bf16.safetensors` | 1,472,223,346 | `847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3` |
| `vae\ltx-2.5-audio-vae-bf16.safetensors` | 364,866,540 | `c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5` |
| `latent_upscale_models\ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 995,778,752 | `eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8` |

このSHA一覧の正本は`.artifacts\official-ltx25\download-5files-report.json`である。

### 9. 次セッション実装者への技術申し送り

- **text encoder GGUFの消費契約の正本**: converter側の[`LTX25_CONVERSION_MODE_SPEC.md`](../../Nz-GGUF-Converter-LTX23/Docs/LTX25_CONVERSION_MODE_SPEC.md)のgemma4-ltx25節（Consumer handoff段落）である。読み込み側の実装はそちらを一次情報として参照すること。
- **I8 sidecarの読み込み側復元手順**: GGUF内のI8型テンソル5本（`tokenizer_json`等、3節参照）は、元のUTF-8バイト列をint8として格納したものである。I8はgguf-py writerにU8型が無いための代替であり、値としてはbyte-identicalなpassthroughにすぎない。読み込み側はint8のまま解釈せず、uint8として再解釈してからバイト列を復元する必要がある（例: `np.asarray(tensor.data).view(np.uint8).tobytes()`）。復元したバイト列をJSONパースまたはテキストとして使う。
- **公式コードのローカル固定コミット**: `Nz-GGUF-Converter-LTX23\.artifacts\official-ltx25\official-code-400fd310\`に、Lightricks/LTX-2のコミット`400fd31054597515f47125691032c04b1c3ee24e`をチェックアウト済みで置いてある。オフラインで公式実装を参照できる。生キー→モジュールツリーの公式対応表として使える関数は、`packages\ltx-core\src\ltx_core\text_encoders\gemma\gemma_assets.py`の`_flatten_gemma4_unified_keys_for_comfy(key: str) -> str`（314行目）である。これはHFの`gemma4_unified`キーをComfyUIのLTXAV TEモジュール名へ変換する関数で、呼び出し元の`build_text_encoder_tensors_from_gemma_root(...)`（335行目）が`comfy_flat_lm_keys: bool = True`という既定Trueの引数でこれを制御する。49層のhidden states結合は同ディレクトリ`feature_extractor.py`の`FeatureExtractorV2`、aggregate出力幅4096/2048の由来は`encoders\encoder_configurator.py`の`_create_feature_extractor`で確認できる。
- **バックエンド側の具体的変更点**: `services\engines\ltx\adapter.py`の88行目、`SUPPORTED_MODEL_VERSIONS: frozenset[str] = frozenset({"2.3"})`に`"2.5"`を加えるゲート解除が必要（同ファイル147行目の`check_kv`がこの定数で422を出している）。加えて`scripts\manifests\20-ltx25.json`は現状`text_encoder`／`video_vae`／`audio`の各カテゴリで`scan`・`extensions`のみ定義済みで`default_file`が未記入のまま（`transformer`だけ`default_file`が入っている）。カテゴリ構成が確定した今、この3カテゴリの`default_file`を埋める作業が必要。
- **16GB VRAM運用方式は未裁定の設計判断である**。本ノート「VRAM 16GB向け実行方式」節（本文）で示した「Gemma 4層streaming＋Q4 Transformer CPU常駐＋per-layer dequant」案と、LTX 2.3で実績のある「量子化のままVRAM常駐＋Linear単位のforward時逆量子化」（`engine\gemma\gguf_quant_service.py`方式。converter側spec文書の「Backend observation」節が参照する`quant_service.py`と同系統）の2案が併存しており、まだどちらを採るか決まっていない。今回のtext encoder GGUF（9.2GB）は後者の方式を想定した量子化構成で作られている——`aggregate_embed`がQ6_Kの`nn.Linear`として量子化されたままVRAM常駐できる構成になっているのは、この想定を反映したものである。**どちらの方式を採るかは、LTX 2.5推論エンジンの実装セッションでの裁定事項として残っている。**

### 参考文献（2026-08-21追加分）

- GitHub Lightricks/LTX-2 `packages/ltx-pipelines/src/ltx_pipelines/utils/model_paths.py` — ModelPathsの5枠構成（transformer／text encoder／video VAE／audio VAE／duration head）の根拠。
- 同 `utils/args.py`、`ti2vid_one_stage.py`、`ti2vid_two_stages.py`、`dfr_pipeline.py`、`distilled.py` — 単一ステージ／2段階系パイプラインの必須引数・オプション引数の実装根拠。
- 同 `packages/ltx-core/README.md`、`ltx_core/loader/sft_loader.py` — safetensorsの`__metadata__`埋め込みconfig方式の根拠。
- 同 `docs/installation.md` — text encoderファイルが1ファイル完結構成であることの公式明記箇所。
- [Lightricks/LTX-2.5公式モデルページ](https://huggingface.co/Lightricks/LTX-2.5) — gemma4-12b-with-proj bf16 safetensorsのヘッダをHTTP Range取得で直接確認（実測。3節の根拠）。
- [elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF](https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF) — 既存GGUF化事例（Q5_K_M、9.51GB）。突合オラクルとしての参照価値はあるが流用は非推奨。
- [joeygambino/LTX-2.5-Quantized](https://huggingface.co/joeygambino/LTX-2.5-Quantized) — ComfyUI向けw4a8量子化版（10.6GB）。
- [realrebelai/LTX-2.5_GGUFs](https://huggingface.co/realrebelai/LTX-2.5_GGUFs) — transformerのみ。text encoder導線はLTX 2.3時代の記述の流用で実質的に破綻。
- [ComfyUI-LTXVideo Issue #303](https://github.com/Lightricks/ComfyUI-LTXVideo/issues/303) — 旧世代（無印LTX-2＋Gemma 3）の16GB VRAM報告。回避策提示により報告者が自己クローズ。
- [Comfy-Org/ComfyUI Issue #15595](https://github.com/Comfy-Org/ComfyUI/issues/15595) — GEMMA_4_12B型がCLIPTypeのLTXVハードコードで握りつぶされる配管バグ。
- [Lightricks/LTX-Desktop Issue #16](https://github.com/Lightricks/LTX-Desktop/issues/16) — 公式デスクトップアプリのlayer streaming実装によりVRAM閾値が32GBから16GBへ引き下げられたことの公式スタッフ回答。
- [Lightricks/LTX-2.5 Hugging Face Discussion #34](https://huggingface.co/Lightricks/LTX-2.5/discussions/34) — 「エンコーダー差し替えは品質劣化」発言。別モデル交換の話であり同一モデルの量子化を否定するものではない。

### 10. HuggingFace再ホストの完了と、インストーラ連携の申し送り（2026-08-23）

結論から言うと、2026-08-23付でLTX 2.5の重みのHuggingFace再ホストが完了した。8節・9節で「`models\LTX25\`へは未配置」「オーナー環境にのみ存在」と書いていた重みが、以下の2つの新規リポジトリでPublic・非Gated（HuggingFaceのアカウント登録もトークンも不要でダウンロードできる状態）として公開されている。台帳の正本は`Nz-HF-Rehost\manifest.json`であり、本節はそこからの抜粋である。

**`Rootport/Nz-LTX25-weights`**（動画生成本体・VAE・アップスケーラー）

| path_in_repo | サイズ（バイト） | SHA-256 |
|---|---:|---|
| `Weights/LTX-2.5-22B-distilled-transformer.gguf` | 14,738,670,464 | `c26473d2d5c3eb61012f60769ac4dccdb8bb77935678206c0148fff351ba060d` |
| `VAE/ltx-2.5-video-vae-conv-bf16.safetensors` | 1,452,269,922 | `685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d` |
| `VAE/ltx-2.5-audio-vae-bf16.safetensors` | 364,866,540 | `c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5` |
| `Upscaler/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 995,778,752 | `eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8` |

**`Rootport/Nz-Gemma4-12B-LTX25`**（LTX専用ファインチューン版Gemma 4のテキストエンコーダー）

| path_in_repo | サイズ（バイト） | SHA-256 |
|---|---:|---|
| `TextEncoder/LTX-2.5-gemma4-12b-text-encoder-Q4_K_M.gguf` | 9,231,374,624 | `4e69e4a33065c856e039fb4c1b78adffae582c4da14b633224136a7e7e30fb90` |

**ライセンス構成**: 2リポジトリともLTX-2.x Community License（2026-08-11版）を同梱する。加えて`Nz-Gemma4-12B-LTX25`は、基底モデル（Gemma 4）がApache License 2.0であるため、`LICENSE-APACHE-2.0`とNOTICE.mdによる帰属表記を同梱ファイルとして別途添えている。テキストエンコーダーだけリポジトリを分けているのは、この由来ライセンスの違いが理由である（`Nz-DWPose`をApache-2.0のために分けたのと同じ考え方）。

**配布していないもの**: `.assets.safetensors`（GGUFの中身からバックエンドが起動時に自動生成する付帯資産。§3-98実装で判明した仕組み。9節参照）と、拡散デコーダー版VAE（`ltx-2.5-video-vae-bf16.safetensors`。エンジンが未接続のため配布対象から外した）、および台帳メタデータの`.manifest.json`は、いずれもこの2リポジトリに含まれていない。

#### `scripts\manifests\20-ltx25.json`の`downloads[]`へ追加すべきエントリ案

**この項の内容は2026-08-23時点の案である**（この2エントリは2026-08-30に投入済みで、いまの正本は`20-ltx25.json`の現物である。節末の追記を参照）。当時の`20-ltx25.json`は`downloads: []`のままだった（`categories`・`assets`・`default_selection`は§3-98 Phase 4で確定済み）。`10-ltx23.json`の実書式に倣うと、次の2エントリを追加すればよい。

```json
{
  "name": "LTX 2.5 weights + VAE + upscaler (4 files)",
  "repo": "Rootport/Nz-LTX25-weights",
  "include": ["Weights/*", "VAE/*", "Upscaler/*"],
  "map": [
    { "from": "Weights", "to": "LTX25/Weights" },
    { "from": "VAE", "to": "LTX25/VAE" },
    { "from": "Upscaler", "to": "LTX25/Upscaler" }
  ],
  "files": [
    {
      "path": "LTX25/Weights/LTX-2.5-22B-distilled-transformer.gguf",
      "min": 14100000000,
      "label": "gguf_transformer",
      "key": "transformer",
      "_min": "公式14,738,670,464Bの1ファイル(約4%下)"
    },
    {
      "path": "LTX25/VAE/ltx-2.5-video-vae-conv-bf16.safetensors",
      "min": 1400000000,
      "label": "component_video_vae",
      "key": "video_vae",
      "_min": "公式1,452,269,922Bの1ファイル(約4%下)"
    },
    {
      "path": "LTX25/VAE/ltx-2.5-audio-vae-bf16.safetensors",
      "min": 350000000,
      "label": "component_audio_vae",
      "key": "audio",
      "_min": "公式364,866,540Bの1ファイル(約4%下)"
    },
    {
      "path": "LTX25/Upscaler/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
      "min": 900000000,
      "label": "spatial_upsampler",
      "_min": "公式995,778,752Bの1ファイル(約10%下)"
    }
  ]
},
{
  "name": "Gemma 4-12B text encoder for LTX 2.5 (1 file)",
  "repo": "Rootport/Nz-Gemma4-12B-LTX25",
  "include": ["TextEncoder/*"],
  "map": [
    { "from": "TextEncoder", "to": "LTX25/TextEncoder" }
  ],
  "files": [
    {
      "path": "LTX25/TextEncoder/LTX-2.5-gemma4-12b-text-encoder-Q4_K_M.gguf",
      "min": 8850000000,
      "label": "gguf_gemma4",
      "key": "text_encoder",
      "_min": "公式9,231,374,624Bの1ファイル(約4%下)"
    }
  ]
}
```

`categories.*.default_file`4件（`LTX25/Weights/LTX-2.5-22B-distilled-transformer.gguf`・`LTX25/TextEncoder/LTX-2.5-gemma4-12b-text-encoder-Q4_K_M.gguf`・`LTX25/VAE/ltx-2.5-video-vae-conv-bf16.safetensors`・`LTX25/VAE/ltx-2.5-audio-vae-bf16.safetensors`）と`assets.spatial_upsampler_path`（`LTX25/Upscaler/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`）は、いずれも上のどちらかの`files[].path`と文字列一致している。`min`は`10-ltx23.json`の慣例（公式サイズの約4〜10%下）に倣った概算であり、実配置後の実測に合わせて微調整してよい。`key`は`install_ltx.ps1`の`models/INSTALLED_PATHS.txt`再生成部のコメントが「`categories`／`assets`の呼称そのもの」と説明している値に倣ったが、必須項目ではない（spatial_upsamplerの`files[]`行に`key`を付けていないのは、`10-ltx23.json`の同種エントリ（Upscaler）も`key`を持たない先例に倣ったもの）。

#### `install_ltx.ps1`の制約（配線前に必ず踏まえること）

実物を読んで確認した事実のみを記す。**引く場所は関数名で書く**——初出時は行番号で書いていたが、その後の改修（`-BaseModel`の新設など）で全て失効したためである。

① **`downloads[]`の各エントリは`name`／`repo`／`include`／`map`／`files`が全て必須**で、欠けると`Test-ManifestShape`が`throw`する。加えて、`files[]`の各行の`path`は同じエントリの`map[].to`のいずれかの配下でなければならず、外れていると「remapが絶対にそこへ置けないのでguardが満たされ得ない」という文言で`throw`する（どちらも同じ`Test-ManifestShape`の中）。

② **`downloads`が非空になった瞬間から、descriptor drift検査が有効化される。** `categories.*.default_file`4件と`assets.spatial_upsampler_path`のすべてが、同じmanifestファイル内の`downloads[].files[].path`のどれかと文字列一致していないと`throw`する（`Test-BaseModelShape`の中。有効化条件はその関数が冒頭で作る`$hasDownloads`）。上記のJSON案では2つの`downloads`エントリを**同一の`20-ltx25.json`内**に追加するため、`$filePaths`の集計はその2エントリ分を合算した上で行われ、5件の一致要求はまとめて成立する。

③ **ステージング領域（`models\.dl\...`）に落ちた全ファイルが`map`に回収されなければならない。** ダウンロード後、`map`のどのエントリにも属さないファイルが1つでも残っていると、「`map`を広げるか`include`を絞れ」という文言で`throw`する（ダウンロード段のremapループ。`Resolve-MapTarget`が`$null`を返した場合）。

④ **`include`は`<dir>/*`形式で、リポジトリ直下のLICENSE・NOTICE.md・README.md等はどの`<dir>/*`にもマッチしない。** これはダウンロード段の見出しコメント（glob semanticsの注記）で明示的に説明されている仕様であり、上記のJSON案の`include`（`"Weights/*"`等）はこの仕様に沿って書いてある——カード類がステージングへ紛れ込んでremapで`throw`されるのを、この境界がそもそも防いでいる。

⑤ **単一`--include`に複数パターンを並べる現行記法はhuggingface_hub 0.36系専用である。** スクリプト自身の警告コメント（同じくダウンロード段の見出し。`nargs="*"`の注記）によれば、1.20.1系では「`--include`は無視する」という警告とともに黙ってファイルを取りこぼす（実際に空間アップスケーラが消える形で再現済みとコメントにある）。`$hfExe`は`"$ProjectRoot\.venv-engine\Scripts\hf.exe"`に固定されており、これはLTX 2.3側の仮想環境である。`.venv-engine`のhuggingface_hubは0.36.2固定（`engine\venv-engine.freeze.txt`実測）なので、現行記法のままで問題は起きない設計になっている。

　ただし**開発機ではこの`.venv-engine\Scripts\hf.exe`自体が壊れている**——2026-08-19のモノレポ改名（`Nz-LTX23-backend`→`Nz-Videomni`）より前（実測タイムスタンプ2026-07-08）に作られたuvシムで、埋め込みパスが旧ディレクトリ名を指している。これは`Nz-HF-Rehost\README.md`冒頭の2026-08-23追記ブロックで報告されている「旧`Nz-LTX23-backend\.venv\Scripts\hf.exe`が改名で壊れたシムのため使えない」現象と同型の問題である。開発機で`setup.bat`／`install_ltx.ps1`のダウンロード検証を行う際、もし`.venv-engine`側を作り直さずLTX 2.5用の`.venv-engine-ltx25`（huggingface_hub **1.28.0**固定、`engine25\venv-engine-ltx25.freeze.txt`60行目実測）のhf.exeへ差し替えて検証するなら、1.20系以降の挙動（②の警告どおり複数パターンが無視される）に当たるため、単一`--include`複数パターンの記法から`--include`を繰り返す記法へ変更する必要がある——コード修正はDocs担当の本エージェントの権限外であり、次セッションでの判断事項として残す。

**この`20-ltx25.json`の編集と`install_ltx.ps1`（`setup.bat`経由）の動作検証は、次セッション（LTX 2.5推論エンジン担当）の作業とする。** 本節はJSONエントリ案と踏まえるべき制約を申し送るのみで、コード変更は一切行っていない。

> **2026-08-30追記（上の本文は当時のまま）**: **上の2エントリは`scripts/manifests/20-ltx25.json`へ投入した。正本はその現物であり、上の案文はもう最新ではない。**
>
> 投入できるようになったのは、`scripts/install_ltx.ps1`に**「どの記述子をこのRUNが担当するか」を決める`-BaseModel`**が入ったからである。その既定値は「`setup.bat`が同梱する集合」＝LTX 2.3と共用前処理器なので、`downloads[]`を埋めても`setup.bat`はLTX 2.5を取りにいかない。導線は`install-LTX25.bat` → `scripts/install_model.ps1` → `install_ltx.ps1`。設計正本は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §6.2、実装と実測の記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-111 と[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §82。
>
> **案と現物の差は2点である。**
>
> 1. **空間アップスケーラの行に`key`を付けた**（`"key": "spatial_upsampler"`）。`key`を持つ行だけが`models/INSTALLED_PATHS.txt`に載るので、付けないとこの1本だけが一覧から落ちる。
> 2. **上の案文が「`10-ltx23.json`のUpscaler行も`key`を持たない」と書いているのは事実誤認である。** 現物の`10-ltx23.json`は該当行に`key`を持っている（裏取り済み）。したがって「先例に倣って付けない」という理由は成り立たない。
>
> **⑤（`--include`の記法）の申し送りも決着した。** ダウンロードは`.venv-engine`側の`hf.exe`（huggingface_hub 0.36.2）で走らせる設計のままとし、記法は変更していない——実ダウンロードを伴う検証はサブマシンで行い、5ファイルとも取得できている（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §82）。**開発機の`hf.exe`が改名で壊れている事実はいまも有効**で、開発機ではこれを「ダウンロードを必ず失敗させる故障注入器」として使い、失敗案内とステージング残存の確認に充てている（同 §82.7）。**なお`hf download`は固定回数まで再試行する**（回数と待ち時間の正本は[`Videomni_Backend_Specification.md`](../Videomni_Backend_Specification.md) §2.5。リマップは`.cache`を消さないので再開情報が残り、取得済みのファイルは取り直さない）。**`hf-xet`もfreezeで固定済み**（2026-08-30・[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-128）。
>
> なお`_note`も書き換えてあり、**`downloads`は`install-LTX25.bat`専用であること**・`install_ltx.ps1`の`-BaseModel`既定値のおかげで`setup.bat`はここに到達しないこと・拡散デコーダ版VAEと`*.assets.safetensors`は配布対象外であることを、現物のコメントとして残してある。**この節に書いた5ファイルのサイズとSHA-256の表は、いまも正本のままである。**

### 11. Stage-2の窓サイズの実測と2.5向け窓プリセット案／LTX 2.3のLoRA・IC-LoRAを2.5へ流用できるか（2026-08-23）

本節は、Chained（クリップ連結）のLTX 2.5移植とSingle版IC-LoRAへ着手する前に行った実験の**設計への持ち帰り**をまとめたものである。**実測値・全ラン表・当てはめ式の正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §71**であり、本節はそこから設計判断に効く部分だけを抜いている。実験中に製品コードは1行も変えていない。

#### (a) Stage-2の窓サイズ

**窓は解像度別に決めるのではなく、単一のトークン予算で決めればよい。** LTX 2.5のStage-2が使うVRAMは、縦横の細かさに使おうが時間の長さに使おうが、**トークン数さえ同じなら同じ**だった（48,960トークンを1280×768／401フレームと1920×1088／185フレームの2通りで測って13.571 GiBと13.554 GiB＝差0.1%以内。44,880〜45,120でも同様）。したがって**解像度ごとに窓の実験をやり直す必要は無い。**

**快適に回る線は約47,000トークン**である。最後に快適だったのが47,040トークン（1280×768／385フレーム）、最初に劣化したのが48,960トークン（両解像度とも）、完全に破綻したのが53,040トークン（1920×1088／201フレーム）だった。LTX 2.3の単発生成で使っている線44,880は、2.5でも安全側の値として通用する（2.5のほうが約5%余裕がある）。

**予算の値そのものは、次のテーマの実装計画で決めること。** 本節は候補と、その候補を選んだときに何が起きるかを示すだけである。以下は「安全側に2.3と同じ考え方で置くなら」という例として**44,000前後**を仮に当てた場合の早見であり、**決定ではない**。

窓の値は自由には選べず、2つの制約がかかる（LTX 2.3で確立した規則がそのまま2.5にも当てはまる。[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md)）。

1. **音声との丸め誤差をゼロに保つには、前進（窓の潜在フレーム数 − のり代）が3の倍数でなければならない。** 前進A潜在フレームは映像で8Aピクセルフレーム＝A÷3秒にあたり、音声側は毎秒25潜在フレームなので、25A÷3が整数である必要があるためである。**のり代が4でも7でも、この条件を満たす窓は「3で割って1余る数」＝…, 19, 22, …, 40, 43, 46, 49 で同じ一覧になる。**
2. **位置埋め込みの天井から、窓は20秒以内。** `positional_embedding_max_pos = [20, 2048, 2048]`（`audio_positional_embedding_max_pos = [20]`も同様）がLTX 2.5のGGUFヘッダでも同じ値であることを確認済みで、潜在60（480フレーム＝20.000秒）が上限、潜在61（481フレーム＝20.04秒）は既に超える。

| 解像度 | 空間のマス目 | 予算44,000での潜在上限 | 条件を満たす最大の窓 | 前進（のり代4／のり代7） | トークン | 実測の裏付け |
|---|---:|---:|---:|---|---:|---|
| 1280×768 | 960 | 45（44,000÷960＝45.8） | **43** | 39＝13.000秒／36＝12.000秒 | 41,280 | 45,120も47,040も快適だった |
| 1920×1088 | 2,040 | 21（44,000÷2,040＝21.6） | **19** | 15＝5.000秒／12＝4.000秒 | 38,760 | 38,760は快適。**ただし下の注意を読むこと** |
| 2560×1472 | 3,680 | 11 | 10 | 6＝2.000秒／3＝1.000秒 | 36,800 | **未実測**（今回の対象外） |

**予算を44,000に置くと、1920×1088の窓が22から19へ落ちる。** これは実測と噛み合わない——1920×1088・窓22は**ちょうど44,880トークン**で、実測では快適だったからである（確保12.600 GiB・予約14.096 GiB・専有ピーク15,750.2MB・Stage-2窓内の共有メモリ135.4MB＝アイドル水準・秒／トークンは最小点比+5%。3軸のいずれも不成立）。つまり**予算を44,880に置けば1080pの窓22がちょうど収まり、44,000に切り下げると19へ後退する**という、境界のすぐ内側での綱引きになっている。参考までに予算44,880での早見は、1280×768が窓46（前進42＝14.000秒・44,160トークン）、1920×1088が窓22（前進18＝6.000秒・44,880トークン）である。**どちらを採るかは次テーマの実装計画で決める。**

**「19フレームモード」はLTX 2.5では不要である。** LTX 2.3では1920×1088に窓22を当てると溢れたので、手動で選ぶ窓19（`high_resolution`プリセット・のり代7）を後付けした。LTX 2.5では窓22が快適に通るので、この後付けを移植する理由は無い。**ただし44,880は線（約47,000）の95%で、余裕は5%しかない。** Chained移植で窓あたりに追加コスト（条件付けの持ち込み等）が乗れば簡単に食い潰されるので、移植後に1920×1088で1本は実測すること。

> **【2026-08-24 追記】この「移植後に1920×1088で1本は実測すること」という宿題は済んでいる。** 2026-08-23のChained移植では**同等化を優先してLTX 2.3の窓の値（`standard`＝潜在22・`high_resolution`＝潜在19）をそのまま持ち込んだ**ので、本節の「19フレームモードは不要」という結論は製品には反映していない。そのうえで1920×1088の実測は取ってあり、2クリップ×169フレームのChained（確保ピーク13.2 GiB）も、169フレームのV2V継続（ベンチマークB8）も、16GBの回避策を1段も使わずに完走している（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §72・§73）。**窓の作り直しは引き続き未着手の判断事項**で、台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-102の判断材料(3)に申し送ってある。

**チャンク化アップサンプルは、Chainedでは必須になる。** 空間アップサンプラ（Stage-1とStage-2のあいだで潜在を縦横2倍に引き上げる部品）は**連結後の全長を一度に処理する**ため、窓分割の恩恵をまったく受けない（`ltx_core/model/upsampler/model.py`の`upsample_video`にチャンク化も`channels_last`も無い）。実測した増え方は完全な直線で、`確保GiB ＝ 5.738×10⁻⁵ × トークン ＋ 1.545`（R²=1.0000）。確保を13.0 GiB以内に抑える条件で逆算すると**上限は約199,600トークン**で、長さに直すと次のとおりである。

| 解像度 | 上限の潜在フレーム | 上限のフレーム数 | 24fpsでの長さ |
|---|---:|---:|---:|
| 1280×768 | 208 | 約1,663 | **約69秒** |
| 1920×1088 | 98 | 約781 | **約32秒** |
| 2560×1472 | 54 | 約433 | 約18秒 |

**1080pで32秒、720pで69秒を超える連結からチャンク化が要る。** 2560×1472にいたっては、1本481フレーム（20秒）の単独クリップですらアップサンプラを通らない。一方、**単発生成（1クリップ・最大481フレーム）ではチャンク化は不要**である——1080pの最長ケースでも8.687 GiB・2.81秒で終わり、同じランのStage-2のほうが先に破綻している。**LTX 2.3のときのような緊急性は無い**点も重要で、2.3ではim2col経路で非線形に爆発したが、2.5は1,000トークンあたり57MiBの素直な直線である。つまりチャンク化は「壊れるから要る」のではなく「長さに比例するから要る」——時間方向に切って繋ぐ単純な形で足りるはずである（重なりの設計は品質確認が要る）。

**Windowsではメモリ不足で落ちない。遅くなるだけである。** 1280×768／481フレーム（58,560トークン）はStage-2の予約ピークが17.756 GiBとカードの物理容量15.99 GiBを超えたにもかかわらず**完走した**（Stage-2は371.11秒＝当てはめ式の予測157秒の2.4倍、共有メモリへ2,822.7MBを退避、PCIe受信の中央値21,800MB/秒）。WDDMが超過分をシステムメモリで裏打ちしてしまうためである。**したがって「メモリ不足の例外を捕まえて安全な設定へ落とす」という設計は成立しない**——捕まえる例外が飛んでこない。利用者から見ると、快適上限を超えた指定は「失敗」ではなく「異様に長い待ち時間」として現れるので、**快適上限マーカーや警告で事前に止める以外に手が無い。**

**スピル判定の作法も2.5向けに更新が要る。** ①**判定はStage-2の時間窓の中だけ**を見ること——復号の局面では幾何によらず毎回600MB前後を共有メモリへ置く（害の無い定常的な帯）ので、ジョブ全体を見た「共有メモリ>0＝溢れ」は常に真になって意味をなさない。②**PCIe軸は中央値でしか使えない**——LTX 2.5は8ブロック常駐のブロックスワップで動くため健全なStage-2でも常に転送が乗り、最大値は毎回10〜13GB/秒に達する。中央値なら快適時2〜6MB/秒・破綻時12,974〜21,800MB/秒と3桁以上離れるので判別できる。実務上は共有メモリの持続的な上振れと秒／トークンの2軸で足りる。

#### (b) LTX 2.3のLoRA・IC-LoRAをLTX 2.5へ流用できるか

**結論: 流用できる（go）。ただし強度は上げる必要がある。** 3段の実験がすべて合格した。

- **A（鍵と形状の照合）**: LTX 2.5の蒸留トランスフォーマーを`meta`デバイスに組み立ててLoRA 14本と突き合わせた。IC-LoRA 5本・Style LoRA 7本の計12本が**形状不一致0・未解決鍵0**。とくに2.3の蒸留LoRA 1本はペア数1,660が2.5側の`nn.Linear`の総数1,660と完全に一致し、**LTX 2.3と2.5のLinearの顔ぶれは名前・形状ともに完全に同一**であることが分かった。2.5固有の`keyframes_abs_pos_embedding`は`nn.Linear`ではなく`nn.Parameter`なのでLoRAの射程外、撤去された映像側FFバイアスもLoRAが触らないので、衝突も欠落も起きない。
- **B（重みの転移）**: `Pixar_Toon`（2.3用Style LoRA）を強度1.0で当てると、フレーム差分の平均絶対値（MAD）が34.84＝フルスケールの13.7%動いた。付け外しは可逆で、外したあとの出力は一度もLoRAを触っていない状態と**1バイト違わなかった**（mp4・復号映像・復号音声の3種のSHA-256がすべて一致）。VRAMも6.743→6.744 GiBとほぼ動かない。
- **C（参照動画の条件付け経路）**: Deblur IC-LoRAに参照動画（ぼかした動画）を渡す経路が2.5でも成立した。**同じseedで参照を渡すかどうかだけを変えると、元動画との距離が19.23→9.08＝53%減**になる。参照はStage-1に1,536トークンを足し、Stage-1のVRAMは3.490→3.795 GiB（**+0.305 GiB**）だが、**ジョブ全体のピークは6.75 GiBで変わらない**（全体の天井はテキストエンコーダーの固定床が握る）。

**いちばん実装に効く発見は、強度である。** Deblurの復元を鮮鋭度（ラプラシアン分散）で測ると、ぼかし入力2.76に対し**強度1.0では35.32＝元動画（75.95）の46%止まり**で中途半端だった。**強度1.3にすると103.65＝元動画の136%**まで戻り、しかも参照への追従はほとんど変わらない（MAD 9.08→9.73）。つまり**強度は「どれだけ復元するか」を制御し、「どこまで参照に従うか」はほぼ独立**である。LTX社の担当者が「2.3のLoRAを2.5で使うときは強度を上げよ」と薦めている件は、これで実測の裏が取れた。**既定強度を1.0のまま2.5へ持ち込むと「効きが弱い」と見える公算が高い**ので、ベースモデルごとに既定強度を分ける設計を検討すること（ただし最適値が1.3で固定なのか、素材やIC-LoRAの種類で変わるのかは今回の1点では決まらない）。

なお**鮮鋭度だけを見る評価は誤る**。参照を渡さない対照は鮮鋭度146.47と比較5本中の最高値を出したが、構図は元動画からMAD 19.23も離れた別場面だった。Deblur LoRAの重みは参照の有無に関わらず高周波を強く押し上げるので、**鮮鋭度は必ず構図の一致と対にして見ること。**

**kohya形式のLoRA 2本は、2.5以前にLTX 2.3のエンジンが読めていない。** `LTX2.3-MysticXXX`と`SynthPussy_01_rank32`は`lora_down`／`lora_up`／`alpha`という方言で書かれており、`engine/gguf/ic_lora_common.py`のローダーは`.lora_A.weight`を固定で探すためペアが1組も作られない。**これは2.5の互換性の問題ではなく、2.3でも同じ状態（警告は出るが実質何も起こらない＝無音の空振り）である。** 手当ては[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-108として起票した。

> **【2026-09-02 追記（上の本文は当時のまま）】kohya形式は読み込み対応した。** いまのローダーはA/B形式とkohya形式（鍵がドットで区切られているもの）の両方を読み、`alpha ÷ rank`を読み込み時にB側へ畳み込む。台帳の記録は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-108で、[`PENDING_TASKS.md`](PENDING_TASKS.md)側は欠番である。撤回した過去の設計と実測は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §88。

**実装時の注意が3つある。** ①**ジョブ末尾の`detach`は必須**——LoRAのバッファは非永続バッファとしてLinearへ付き、公式の`dispose()`を生き延びることを実測で確認した（Stage-2の構築時点でStage-1のバッファが残っていた）。しかもモデルの器は使い回されるので、明示的な`detach_ic_loras`だけが「前のジョブのLoRAを引きずらないこと」を保証する。②**既定強度のベースモデル別分離を検討する**（上記）。③**`attach`はブロックの配置より前に置く**——LoRAバッファは`module.to(device)`に付いて動くので、ブロックスワップがブロックをCPUへ移したあとでは取り残される。LTX 2.3側の正本（`engine/gguf/quant_service.py:789-791`）と同じ規則で、`engine25`では`_build_transformer`の「構築→X0Model→配置」のうち**X0Modelの直後・配置の直前**が挿入位置になる。

**この節の数値の詳細（全ラン表・当てはめ式・A表14本・B/Cの全指標・オーナー目視欄）は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §71にある。**

---

## §3-98 v1実装完了（2026-08-22）— 調査時の想定と、実際にやってみた結果の差分

本節は**この調査ノートの本文を訂正するものではない**。本書は2026-08-20〜21当時の下調べを当時のまま収蔵する参考資料であり、そこは動かさない。ここに書くのは、**実装が終わってみて「調査時の想定と違ったところ」がどこだったか**という短い覚書である。実測値と検証の経緯は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §69が正本で、設計の正本は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md)。

| 調査時の想定 | 実装の結果 |
|---|---|
| **9節**「16GB VRAM運用方式は未裁定」。Gemma 4の層streaming案と、LTX 2.3実績の「量子化のままVRAM常駐」案が併存 | **両方を場所ごとに使い分ける形で決着した。** transformerは常駐＋層ごとの逆量子化＋ブロックスワップ、テキストエンコーダは層を1枚ずつ流す（既定はGPU常駐0層）。**そして、そもそも16GBは苦しくなかった**——本番規模（768×512／121フレーム）でのピークは**7.00 GiB**にすぎず、回避策の手順を1段も使っていない |
| **5節**「16GBを阻む構造上の困難は見当たらない」（調査時点では見立て） | **見立ては正しかった。** 実測でピークは予算の半分以下。ただし**関門の場所は予想と違った**——最大のピークはノイズ除去でもVAEデコードでもなく、**テキストエンコード（7.0 GiB）**である |
| **9節**「バックエンド側の変更点は `SUPPORTED_MODEL_VERSIONS` に `"2.5"` を足すゲート解除が中心」 | **この見立ては外れた。** LTX 2.5は既存の推論スタックの中では動かないため（公式パッケージが5か月古く、`transformers`のバージョンも同居できない）、**専用の仮想環境とワーカーを持つ別のエンジン系統 `ltx25` として新設**した。**LTX 2.3側のこの定数は `{"2.3"}` のまま触っていない** |
| **2節**「text encoderは1ファイル完結構成で、`--gemma-root` のような別ディレクトリ指定は不要」 | **正しかったが、GGUFを渡すと成立しない**という落とし穴があった。公式のパイプラインは重みの差し替え口に触れるより前にトークナイザ等をそのファイルから読むため、GGUFのパスを渡すとパイプラインの生成そのものが落ちる。**対策は「公式が探しているファイルを、GGUFの中身から作って隣に置く」方式**（付帯資産だけの約31MBのsafetensors）で、これが機能した。おかげで**公式のプロンプトエンコーダをフォークせずに済んだ**——差し替えたのは公開引数から注入できる1点だけである |
| **3節** 49層の隠れ状態を結合して巨大な行列で射影する構造（`aggregate_embed`） | 構造の理解は正しかった。ただし**素直に展開すると映像側だけで確保ピークが24.2 GiBへ跳ねる**ことが実測で分かった。出力行を刻んで「一部を展開→掛ける→捨てる」形にすると**1.76 GiB**まで落ちる。**刻み幅は調整つまみではなく固定定数として扱う必要がある**（刻んでも数学的には等価だが、ビット単位では等価にならないため。決定性の前提になっている） |
| **8節** テキストエンコーダのGGUFは「量子化のままVRAM常駐」方式を想定した構成（`aggregate_embed` をQ6_Kで量子化） | **その想定どおりに使えた。** 量子化ポリシーの変更は不要だった |
| **7節** 再ホスト時の LTX-2.x Community License の fine-tune transfer 条項が未確認 | **未確認のまま残っている。** 再ホストする場合は着手前に一次ソースのLICENSE本文を読むこと |
| CPUストリーミングなどの追加の回避策が要るかもしれない | **不要だった。** GPUに載せるブロック数は既定の8のまま、VAEデコードのタイル設定も明示せず（自動タイル化が働いた）、解像度の上限も掛けていない |

**調査時に想定していなかった発見**が2つある。どちらも実装中に実測で判明したもので、詳細は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §69.19にある。

1. **音声VAE（音声の圧縮展開器）のボコーダーが非決定性の源だった。** 同じ潜在表現を3回デコードすると3通りの波形になる。cuDNNを決定的アルゴリズムへ固定すると根治し、速度コストは実測誤差以下だったので、既定ONにした。
2. **LTX 2.5ワーカーの常駐RAMは約25 GiB。** stage-2で14.7GBを読み直さないための重み保持による。64GBのマシンでは問題にならないが、**動作要件としてRAM側に書くべき事実**である。**この「約25 GiB」は2026-08-25に更新された**——計測の標本点を増やしたところ、テキストエンコーダの組み立て中は単発で25.89GiB・連結生成で26.33GiBまで上がることが分かった（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §76.2(3)と、その訂正である§76.8(3)を参照すること）。

---

## §3-102 第3段（スタイルLoRA・IC-LoRA）実装完了（2026-08-24）— 11節(b)の「実装時の注意」がどう決着したか

本節も**この調査ノートの本文を訂正するものではない**。11節(b) は 2026-08-23 の前提実験の記録で、そこは動かさない。ここに書くのは、**その結論を実際に実装してみてどうなったか**という短い覚書である。実測と検証の経緯は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §74が正本で、設計の正本は[`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) §5.6・§8.5、連結生成まわりの方式差分は[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) 12節である。

| 11節(b) の記述 | 実装の結果 |
|---|---|
| ①**ジョブ末尾の `detach` は必須** | **そのとおりだが、より強い形で解決した。** ジョブ末尾の取り外しは残しつつ、**取り付けの箇所そのものを「LoRAがあれば付ける／無ければ外す」の対**にした。連結生成は1つのジョブでモデルを何度も組み直すので、毎回どちらかを必ず通るこの形なら**取り残しの芽が構造的に消える**。実測でも、LoRA無しの出力が「一度もLoRAを触っていない別プロセスの出力」と1バイト違わないことを確認している（§74.3(b)） |
| ②**既定強度のベースモデル別分離を検討する** | **今回は分離せず、LTX 2.3 と同じ 1.0 に据え置いた**（同等化のスコープでは既定を動かさない）。強めたい利用者は画面のLoRAタグで `1.3` のように指定できる。**分離するかどうかの判断は台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-102 の課題として生きている** |
| ③**`attach` はブロックの配置より前に置く** | **そのとおりの位置で成立した**（`engine25/gguf_transformer.py` の `_build_transformer` 内、X0Model 生成とブロック配置のあいだ）。取り付け・取り外しの数も実測で一致している（スタイルLoRAで576層） |
| **強度1.0では復元が46%止まり**という実測 | **実装では既定を変えていない**ので、この性質はそのまま残る。README §7.1 とVERIFICATION_LOG §74.9(1) に「効きが弱いと感じたら強度を上げる」という案内として記載した |

**調査時に触れていなかった、実装で決まった差分が4つある。**

1. **LoRAは Stage-1・Stage-2 の両方に適用する。** 公式の IC-LoRA パイプラインは Stage-1 にだけ適用するが、**LTX 2.3 が両方に適用しているので、同等化を優先してこちらに揃えた。** 承知のうえの差である。
2. **参照アイテムのノイズ側の扱いが 2.3 と実装レベルで違う。** LTX 2.5 の公式部品は**ノイズ側をゼロで埋める**（2.3 は「ここは参照だ」という印を立てる方式）。公式の仕様なのでそのまま使い、意図せず変わったら気づけるよう照合検査で固定した。
3. **前処理（canny／dwpose／depth）は LTX 2.3 と同じコードを共有する。** 2.5 用の仮想環境に依存を2行（`opencv-python-headless` と `torchvision`）足すだけで動き、**同じ素材から両エンジンが書いた制御動画は SHA-256 まで一致した**（§74.7(f)）。エンジンごとに前処理を二重に持つ必要は無い。
4. **LTX 2.5 用の仮想環境での実行手順**（`engine/` 側のヘルプ文は LTX 2.3 用のまま据え置いてある）:

```text
.venv-engine-ltx25\Scripts\python.exe -m engine.preprocess.preprocess_selfcheck
.venv-engine-ltx25\Scripts\python.exe -m engine.preprocess.depth_g1_gate --video <参照動画> --frames 121
```

**なお 11節(b) が名指しした kohya 形式のLoRAが読めない件は、今回も直していない**（台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-108として生きている）。**音声側の強度（`audio_strength`）も、音声軸の重みを持つLoRAが1本も無いため 2.5 でも実効を確認できていない**（§74.3(c)）。

> **【2026-09-02 追記（上の本文は当時のまま）】kohya形式の件は片付いた。** 2026-09-02にローダーが読み込み対応し、§3-108は[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-108としてクローズしている（[`PENDING_TASKS.md`](PENDING_TASKS.md)側は欠番。詳細は11節(b)末尾の追記と[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §88）。`audio_strength`の実効が未確認である点は、いまも変わっていない。
