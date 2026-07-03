# Phase C 統合判断（監督・2026-07-03）

R1（公式仕様・WEB）/ R2（前処理器エコシステム・WEB）/ R3（コード考古学・リポジトリ）の3調査を統合した監督決定。
ワークオーダー執筆者はこの決定に従うこと。3調査の生レポートは本ファイル末尾に全文添付。

## 仮説判定（H1〜H5）

| 仮説 | 判定 | 根拠 |
|---|---|---|
| H1: 制御系も同じ参照動画latent-append機構 | **裏付け** | wheel実コード（ic_lora.py `_create_conditionings`→`VideoConditionByReferenceLatent`）＋削除済みフォークも制御動画を`video_conditioning=[(path,strength)]`でそのまま同経路に投入していた＋ComfyUIノード構成も同型。R1のFLAG-2（README記載の`VideoConditionByKeyframeIndex`との名称食い違い）はR3のwheel実コード精読で解消（実経路はReferenceLatent。READMEは不正確または画像keyframe側の記述） |
| H2: 制御動画は外部前処理器で特定視覚形式に事前レンダリング必須 | **強く裏付け** | 公式trainerの`compute_reference.py`が「ユーザーが制御動画を用意する」設計と明記（default=Canny）。wheelに前処理コードは一切なし（grep済み）。Pose=DWPose系のOpenPose風カラー骨格が慣行（公式docsもOpenPose/MediaPipe/DWPoseを列挙）。ただし線色の厳密規約は公式文書に無し→標準実装（フォーク/ComfyUIと同系の描画）を使うことでリスク回避 |
| H3: 前処理器はWindows+ローカルvenvで両立 | **裏付け（想定より好条件）** | 第一候補=フォーク実証済みの**TorchScript版DWPose**（`hr16/DWPose-TorchScript-BatchSize5` 135MB + `yolox_l.torchscript.pt` 218MB）→ `.venv-engine`の既存torchだけで動き**追加pipインストール不要**。第二候補=rtmlib（torch非依存・onnxruntimeのみ・Apache-2.0）。Canny=cv2（.venv-engineに導入済み・追加なし）。Depth=transformers DPT（導入済み） |
| H4: 2.3用制御アダプタがHFで入手可能 | **修正のうえ裏付け** | ⚠️LTX-2.3-22bに単体Pose/Canny/Depthは存在しない（単体は旧19b世代のみ・2.3への流用は「効果ゼロ」報告あり）。**2.3の制御系=Union-Control（Canny+Depth+Pose統合・654,465,352 bytes・`Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control`）とMotion-Track-Controlの2本**。フォークのダウンロード定義にも同一ファイルがあり整合 |
| H5: 凍結API純加算で載せられる | **裏付け（新フィールドすら不要の設計を採用）** | 既存`/upload/video`＋`reference_video_id`＋`loras[].name`をそのまま転用。制御タイプはレジストリ側メタデータで表現（下記） |

## 矛盾の解消

1. **`ref0.5`の意味**: R1=「downscale factor 2（参照を出力の0.5倍で内部使用）」/ R3フォーク考古学=「デフォルト参照強度の可能性」と解釈が割れた。公式docsの「64で割り切れること（=factor 2×32）」という制約記述はR1説を支持。ただし**確定は現物のsafetensorsメタデータ（`reference_downscale_factor`キー）を読むこと**＝実装セッションのGate 0に設定。factor=2なら既存の`factor<=1`ガード（fast_video_pipeline.py:282-290）はそのまま通り、ガード緩和は不要になる。factor=1ならR3の指摘どおりメタデータ駆動の緩和が必要（wheel側はfactor=1を正常系として扱うことをR3が確認済み＝緩和自体は安全）。
2. **前処理器の選定**: R2推奨=rtmlib（新依存onnxruntime-gpu、CUDA12.8とのバージョンpin注意）vs R3発掘=TorchScript版DWPose（追加依存ゼロ・フォークで実証済み）。**監督決定=TorchScript版を第一候補**（理由: 依存追加ゼロ・同一マシンで動いていた実証・onnxruntime-gpuのCUDA/cuDNN整合リスク回避）。rtmlibはTorchScript版の精度/速度に問題が出た場合のフォールバックとして記載。

## 設計決定

1. **ターゲットアダプタ = LTX-2.3-22b Union-Control 一本**（654MB）。Motion-Track/In-Outpainting/Deblur等はPhase D以降。
2. **Phase Cの制御タイプ = canny + pose の2種**。
   - canny: cv2.Canny（公式default・追加依存ゼロ）→ **E2E配線の疎通・検証を最小コストで先行**させる持ち上げ用スライス
   - pose: DWPose TorchScript（本命=「動き維持で内容置換」）
   - depth: スコープ外（Phase D。フォークのMidas実装は古く、公式スタッフはDepthCrafter言及・R2はVideo-Depth-Anything推奨と選定が割れており追加リサーチ要）
3. **API形状 = レジストリ駆動・リクエストスキーマ無変更**。`config.yaml model.ic_loras` を `name → {path, preprocess}` 形式に拡張し、論理アダプタ名で前処理を切替:
   - `pose-control: {path: <union>, preprocess: dwpose}`
   - `canny-control: {path: <union>, preprocess: canny}`
   - `pixel-spatial-upscaler-x2: {path: <既存>, preprocess: none}`（後方互換: 文字列値も従来どおり受理= preprocess none）
   - クライアントは従来どおり `loras:[{name:"pose-control"}]` + `reference_video_id`（生動画をアップロード→サーバー側で前処理）。凍結API契約は完全に不変。
4. **前処理の実行場所とタイミング = engine worker（.venv-engine）内・generate時**。動画→動画のフレーム単位変換モジュールを新設（フォークのProtocol設計をレシピとして参照・コードは書き直し）。前処理済み動画のキャッシュ（video_id×preprocess種別キー）はオプションスライス（初版はキャッシュ無しでよい・実測で遅ければ追加）。
5. **strength = 1.0固定を維持**（公式tutorial「1.0未満はreferenceが滲む」・既存配線の固定1.0と整合）。可変化はスコープ外。
6. **制約の強制**: 制御動画は出力と同尺前提（wheelはframe_capで切り詰めるのみ・FPS変換なし）→ 検証/警告をどこに入れるか実装時に determine。解像度は既存の64グリッド処理と整合確認。
7. **モデルファイル取得**: Union-Control LoRA（654MB）＋DWPose TorchScript 2ファイル（計~353MB）。配置先は `models/ltx-2.3-ic-lora/union-control/` と `models/preprocessors/`（新設）。ダウンロードは実装セッションで実施（本セッションではしない）。

## Gate 0（実装セッション冒頭の検証・実装より先）

- G0-a: Union-Control LoRAをダウンロードし、safetensorsメタデータの`reference_downscale_factor`を読む → 2なら既存ガード無変更/1ならメタデータ駆動緩和スライスを有効化
- G0-b: DWPose TorchScriptの実測スモークテスト（数百フレームの骨格レンダ・fps/VRAM実測。R2のスループット確度は低〜中のため）
- G0-c: LoRAキー構造が既存`ic_lora_common.py`のattach機構（lora_A/lora_B命名・ComfyUIリネームマップ）でそのまま解決できるか確認

## ゲート（Phase B同型＋制御系固有）

- G1: 回帰byte-match（LoRA off・T2V/I2V基準SHA不変・pytest全通過）
- G2: 非汚染トグル（pose→なし→canny→なし で「なし」がbase一致）
- G3: VRAM/速度（前処理込みピークが既存天井を超えない・前処理時間の実測記録）
- G4: API e2e（生動画upload→pose-control generate→取得・メタデータに制御情報記録）
- G5: 目視（720p級・映画トレイラー風・賑やかな題材＝visual-verification-content-guideline準拠。「動き維持で内容置換」の成立確認: 例=踊る人物の参照→別キャラクター化）
- 制御系固有: 前処理出力の形式検証（骨格動画のサンプルフレームをユーザー目視に供する）

## スコープ外（Phase Cではやらない）

- depth・Motion-Track・In-Outpainting・Deblur・Ingredients・LipDub等の他アダプタ
- 19b世代アダプタの流用（効果ゼロ報告・非対応）
- strength可変化・conditioning_attention_mask露出
- 前処理キャッシュ（オプションスライス扱い・初版不要）
- Gradio UI露出

---

# 追補 R4（2026-07-04・ユーザー質問起点）: ComfyUIは「単体アダプタ不在」をどう扱っているか

ユーザーからの「ComfyUIのIC-LoRAカスタムノード＆ワークフローではアダプタ不在の問題をどう解決しているか」という質問を受けた追加調査。結論＝**ComfyUIエコシステムでは「不在」は問題として扱われておらず、公式の答えは一貫して「2.3ではUnion-Controlを使う」**。

- 公式リポジトリ `Lightricks/ComfyUI-LTXVideo` の `example_workflows/2.3/` に **`LTX-2.3_ICLoRA_Union_Control_Distilled.json`** が存在（JSON実体を確認）。構造:
  - **同じUnion LoRAを常にロード**（`LTXICLoRALoaderModelOnly`→`LTXAddVideoICLoRAGuide`でguide latent注入）
  - **前処理ノードを3系統すべて内蔵**: `DWPreprocessor`（pose・comfyui_controlnet_aux由来のDWPose）／`CannyEdgePreprocessor`／`VideoDepthAnythingProcess`（depth）
  - 制御種の切替＝「どの前処理出力をguideに流すか」**だけ**。LoRAは共通。19bファイルへの参照は一切なし
  - **プロンプトは制御種に言及しない**（実JSONのpromptはシーン内容の記述のみ。"pose"/"canny"等の制御語なし）
  - 出典: https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows/2.3 ／ https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/example_workflows/2.3/LTX-2.3_ICLoRA_Union_Control_Distilled.json
- 19b→22bのLoRAキー変換ノード・移行ツールは**存在しない**。公式開発者ガイド系記述は「2.0系LoRAはVAE/latent空間/パラメータ規模（→22B）の変更により2.3では動作しない・移行パス無し・要再学習」と明言（https://ltx.io/blog/using-lora-adapters ）。「エラーなく効果ゼロ」報告と整合（サイレントno-op化の内部機序のみ推論）。
- コミュニティの対応も「リマップ」ではなく**22bネイティブ再学習**（例: CivitAI「Cameraman IC-LoRA for LTX2.3 22B」＝カメラ制御。pose単体の2.3再学習は発見できず）。

**Phase C設計への含意（すべて既存設計を強化する方向）**: ①Union一本＋前処理切替という当方設計は公式ワークフローと同型＝正当性確認。②pose前処理にDWPose（controlnet_aux系）を使う選択も公式ワークフローと一致。③将来のdepth対応はVideoDepthAnything（R2推奨と一致）が公式採用済み＝Phase Dの前処理器選定はほぼ確定。④プロンプト規約=制御種を書かずシーン記述のみ（Phase C目視検証時のプロンプト設計に反映）。

---

# 添付: R1生レポート（公式仕様）

（※以下、R1の全文をそのまま保持）

## ⚠️仮説矛盾 / 重要フラグ

**FLAG-1(世代の食い違い / H4に部分矛盾):** 専用の Pose / Canny / Depth control adapter は LTX-2-19b 世代のみ存在し、LTX-2.3-22b 版の単体 Pose/Canny/Depth は存在しない。LTX-2.3-22b の control 系は Union-Control(Canny+Depth+Pose 統合) と Motion-Track-Control の2本に集約。コミュニティ報告(HF discussion, user PGCRYPT)で「旧 19b の depth LoRA を LTX-2.3 に適用してもエラーは出ないが視覚効果がゼロ」との指摘あり。
- 出典: https://huggingface.co/Lightricks/LTX-2.3/discussions/3

**FLAG-2(機構クラス名の食い違い):** 公式 ltx-pipelines README は ICLoraPipeline が VideoConditionByKeyframeIndex(ltx-core)を使うと記述（実コードとの食い違い→R3で解消済み: 実経路はVideoConditionByReferenceLatent）。ICLoraPipeline は distilled model 専用との明記あり。
- 出典: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md

**FLAG-3(reference_downscale_factor):** LTX-2.3 の Union / Motion-Track は factor=2(`ref0.5`)。19b の pose/canny/depth はファイル名に ref サフィックス無し→factor=1 と推定。adapter ごとに metadata から factor を読む実装(ComfyUI の LTXICLoRALoaderModelOnly が正にこれをやる)が必須。

## Q1. 存在する control-type IC-LoRA

### A. LTX-2-19b 世代(旧・我々のbaseとは非互換の可能性大)
- Lightricks/LTX-2-19b-IC-LoRA-Pose-Control / -Canny-Control / -Depth-Control / -Union-Control / -Detailer

### B. LTX-2.3-22b 世代(我々の base に一致)
| Repo | filename | 用途 | ref factor |
|---|---|---|---|
| Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control | ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors(654 MB) | Canny+Depth+Pose 統合 | 2 |
| Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control | ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors | 疎な軌跡で物体動作制御 | 2 |
| Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting / -Deblur / -Ingredients / -LipDub / -HDR | — | 各種 | — |

Camera-Control 系(Dolly/Jib/Static)は control-IC-LoRA ではなく通常 style LoRA(制御信号動画を取らない)。
旧 ltxv-0.9.7 DEV in-context LoRA(LTX-Video-Trainer, 0.9.x 系)は別世代・非互換。

## Q2. 各 control 信号の期待フォーマットと前処理ツール

大前提(H2確認): 公式 trainer は制御信号動画を内部生成しない。ユーザーが compute_reference.py(LTX-2 trainer)/compute_condition.py で事前レンダリングした control 動画を用意する設計。default 実装は Canny、depth/pose は「関数を差し替えて自作せよ」と明記。
- 出典: https://raw.githubusercontent.com/Lightricks/LTX-2/main/packages/ltx-trainer/docs/dataset-preparation.md

| 信号 | 期待フォーマット | プリプロセッサ | 確度 |
|---|---|---|---|
| Pose | 17-18 keypoints の骨格線オーバーレイ | 公式docsはOpenPose/MediaPipe Pose/DWPoseを列挙。ComfyUI実務標準・公式チュートリアルはDWPoseを明記 | 高 |
| Depth | 全フレームで深度レンジ一貫の深度マップ動画 | DepthCrafter(Lightricksスタッフ言及)。pre-rendered depthも"totally valid" | 高 |
| Canny | エッジ太さ一定のエッジ動画 | OpenCV Canny(compute_reference.pyのdefault)。Lightricks/Canny-Control-Dataset が基準 | 高 |
| Motion-Track | 色付きスプライン/円のトレイルで動作パス描画 | point tracking(SpatialTrackerV2)またはComfyUIのtrajectory drawing node | 中 |
| Union | 上記のいずれか(または複数同時)を同フォーマットで | 各単体と同じ | 中 |

注意: pose の正確な線/関節の色規約は公式未明文化("skeleton visualization with lines connecting keypoints"止まり)。DWPose の標準カラー骨格が慣行。
- 出典: https://docs.ltx.video/open-source-model/usage-guides/ic-lo-ra 、https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/discussions/1 、https://ltxworkflow.com/resources/tutorials/ic-lora-ltx-2-3-complete-guide

## Q3. 機構
- control adapter も「reference 動画→VAE encode→small latent を denoise トークン列に append」系。ComfyUI: LTXICLoRALoaderModelOnly(LoRAロード+metadata から downscale factor 抽出)、LTXAddVideoICLoRAGuide(small latent を guide として付加、Frame index 既定 0、Strength 既定 1.0)。
- LoRA 適用ステージが stage-1 限定かはREADMEに明記なし(wheel実コードでは stage-1 のみ=R3/Phase B実装で確認済み)。

## Q4. 実用制約
- control 信号解像度 = 生成解像度。生成動画の解像度は reference 動画の解像度に一致。
- ref0.5 モデル: reference の width/height は共に 64 で割り切れること(= downscale factor 2 × 32)。内部で half size 化。
- FPS: control FPS = 生成 FPS。25fps 標準、docs 推奨 704×1216 @ 24-30 FPS。
- 長さ: control 動画は生成と positionally aligned = 同尺前提。
- Strength: 既定 1.0 を維持推奨。1.0 未満は reference が pop/bleed-through する(tutorial の明確な警告)。

## Q5. ライセンス
- LTX-2 Community License。学術無料・商用も ARR $10M 未満は無料。$10M 超は要商用契約。
- 出典: https://github.com/Lightricks/LTX-2/blob/main/LICENSE

## 確度評価(抜粋)
- Q1列挙=高 / Q2前処理=中〜高(骨格の厳密カラー規約は断定不可) / Q3機構=中(クラス名はR3で実コード確認済み) / Q4制約=中〜高 / Q5=高

---

# 添付: R2生レポート要約（前処理器エコシステム）

- 環境両立性=青信号。「⚠️環境両立性に赤信号」フラグ無し。
- rtmlib(Tau-J): pip install rtmlib。numpy/opencv/onnxruntimeのみ・torch依存ゼロ。DWPose(133-kpt)内蔵。`to_openpose=True`でOpenPose風カラー骨格描画(body18虹色+手+顔)。Apache-2.0。0.0.15(2026-02)活発。GPUはonnxruntime-gpu別途(CUDA12.8→1.20〜1.26帯にpin。1.19未満=cuDNN8系NG・1.27以降=CUDA12廃止予定)。
- easy-dwpose: pip・torch+PILのみ。次点。
- 本家IDEA-Research/DWPose main=mmcvビルド地獄で除外。Sapiens=CC-BY-NC(非商用)で除外。mediapipe=33kptトポロジ非互換で低適合。
- ComfyUI comfyui_controlnet_aux は yolox_l.onnx + dw-ll_ucoco_384.onnx(onnxruntime)を使用=Windows実績最大。
- スループット/VRAM: DWPose固有の公開ベンチ見つからず(確度低〜中)。モデルは計~350MB級・GPU onnxruntimeなら120-240フレームは1分未満と推定。実測スモークテスト推奨。
- Depth(参考): Video-Depth-Anything(時間一貫性のある動画深度・CVPR2025)が per-frame DA-V2 より適。Canny: cv2.Canny で追加依存ゼロ。
- 総合推奨: rtmlib > easy-dwpose > comfyui_controlnet_aux流用 > controlnet-aux > mediapipe。
（※監督決定によりPhase C第一候補はフォーク実証済みTorchScript版DWPose・rtmlibはフォールバック）

---

# 添付: R3生レポート（コード考古学・全文骨子）

## ⚠️要設計注意
fast_video_pipeline.py は reference_downscale_factor <= 1 を意図的にハードエラーで拒否(L282-290 `_set_ic_job`、L543-544 `assert scale > 1`)。stage判別式 `cond_height == full_height // 2`(L536)はscale非依存で無罪。wheel側(ic_lora.py L354-360、reference_video_cond.py L64-66)は factor=1 を正常系として扱う(`if scale != 1:`ガード・RoPE補正も`!= 1`時のみ)。ブロッカーは自前実装側のみ。L551-557のdownscale算術は現状scale>1前提なので緩和時は`if scale != 1:`ガード追加要。

## T1. 削除済みフォーク(d0d3df5^ に現存)
- vendor/LTX-Desktop-LOW-VRAM/backend/ 配下: _routes/ic_lora.py、handlers/ic_lora_handler.py、services/depth_processor_pipeline/(midas_dpt_pipeline)、services/pose_processor_pipeline/(dw_pose_pipeline)、services/ic_lora_pipeline/、state/conditioning_cache.py、tests/
- Canny: cv2.Canny(gray, 100, 200) + 64の倍数へpad/crop + 1ch→3ch展開(video_processor_impl.py:44)
- Depth: Intel/dpt-hybrid-midas(transformers DPTForDepthEstimation)fp16→bilinear復元→min-max正規化→COLORMAP_INFERNO疑似カラー
- Pose: DWPosePipeline = YOLOX person detector(torch.jit.load TorchScript) + DWPose keypoint(TorchScript・simcc)。純torch+cv2+numpy・onnxruntime不要。OpenPose準拠のbody/hand/face skeleton描画(limb_seq/color table)
- 投入方法: 前処理で動画丸ごと制御動画mp4化→ wheel ICLoraPipeline の video_conditioning=[(path, strength)] へそのまま(=現行と同一機構)
- handlers の match が "canny"/"depth" のみで pose のハンドラ配線は未接続だった可能性
- ConditioningCache: (video_path, conditioning_type)キーで前処理済み動画キャッシュ
- モデル配布定義(runtime_config/model_download_specs.py):
  - ic_lora: ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors 654,465,352 bytes repo=Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control
  - depth_processor: Intel/dpt-hybrid-midas(folder ~500MB)
  - person_detector: yolox_l.torchscript.pt 217,697,649 bytes repo=hr16/yolox-onnx
  - pose_processor: dw-ll_ucoco_384_bs5.torchscript.pt 135,059,124 bytes repo=hr16/DWPose-TorchScript-BatchSize5

## T2. wheel再確認
- ICLoraPipeline.__call__: images(必須)、video_conditioning: list[tuple[str,float]](必須)、conditioning_attention_strength: float=1.0、conditioning_attention_mask: Tensor|None=None、skip_stage_2: bool=False
- _create_conditionings(L354-360): scale=1 は正常系(`scale != 1 and (height % scale ...)` が false→バリデーションスキップ・height//1=そのまま)
- load_video_conditioning(media_io.py:102-115): PyAV逐次デコード+frame_capで先頭から切り詰め(FPS変換なし・フレーム伸縮なし)+resize_and_center_crop+[-1,1]正規化
- wheelに前処理の痕跡なし(docstringに"control signals such as depth maps, human pose, or image edges"の言及のみ)

## T3. 自前拡張ポイント
- api/uploads.py:40 POST /upload/video(.mp4/.mov/.webm/.mkv、max_video_size_mb、UUID保存・traversal対策済み)→制御用に転用可・専用経路不要
- api/models.py:143-152 loras⇔reference_video_id all-or-nothing検証(文言がupscaler前提=一般化要)。LoraSpec(L39-59)=name+strengthのみ
- services/lora_registry.py + config.py:85 ic_loras: dict[str,str](フラットマップ)→メタデータ付きエントリへ拡張要
- pipeline_manager.py:147-155→ltx_runner.py:815-833(reference strength固定1.0)→engine/worker.py:198-222→fast_video_pipeline generate
- venv: .venv(サーバー)=cv2/transformersなし。.venv-engine=torch+cv2(opencv_python_headless 4.13)+transformers 4.57.6→canny/depth/poseの3種すべて追加pipなしでengine worker内に実装可能。前処理は必ずengine worker側
