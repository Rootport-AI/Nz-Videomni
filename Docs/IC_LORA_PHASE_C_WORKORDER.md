# IC-LoRA Phase C ワークオーダー — Union-Control 制御系アダプタ対応＋engine内前処理段の新設

- 作成: 2026-07-03（base = main・Phase B マージ済み `cfddd77`・作業branch `feature/ic-lora-phase-c-research`）
- 併読: [`IC_LORA_PHASE_B_STATUS.md`](IC_LORA_PHASE_B_STATUS.md)（前提状態・完成物）／[`IC_LORA_PHASE_B_WORKORDER.md`](IC_LORA_PHASE_B_WORKORDER.md)（構成の踏襲元）／[`VERIFICATION_LOG.md` §21](VERIFICATION_LOG.md)
- 正本: [`IC_LORA_PHASE_C_RESEARCH.md`](IC_LORA_PHASE_C_RESEARCH.md)（監督の統合判断＋R1/R2/R3リサーチ全文）。**本ワークオーダーはその決定を具体化したもの。設計変更は監督判断を仰ぐこと。**

---

## 1. 目的と背景

Phase Bまでで、IC-LoRAアダプタを本番の量子化経路（GGUF Q4_K_M）に forward 時ウェイトパッチとして載せる機構と、`loras`パラメータ＋参照動画アップロードのREST APIが完成した。ただし実装・検証したのは**参照系アダプタ**（Pixel-Spatial-Upscaler）のみで、これは生の参照動画を入力に「解像度・ディテールを盛る」用途であり、看板機能である「動きを維持したまま内容を差し替える」（例: 踊る人物の映像を別のキャラクターに置き換える）は担えない。

Phase Cは、その「動き維持で内容置換」を実現する。具体的には LTX-2.3-22b の **Union-Control** アダプタ（Canny/Depth/Pose統合の制御系IC-LoRA）に対応し、あわせて生の動画を制御信号（エッジ動画・骨格動画）へ変換する**前処理段を engine worker 内に新設**する。ユーザーは従来どおり生の動画をアップロードするだけで、サーバー側がエッジ抽出や姿勢推定を行って制御信号に変換し、その動きに沿った新しい映像を生成する。Phase Cで扱う制御タイプは **canny（エッジ）** と **pose（姿勢・骨格）** の2種。

## 2. リサーチ根拠

3調査（R1公式仕様WEB／R2前処理器エコシステムWEB／R3リポジトリ考古学）の統合。判定と区別は synthesis のとおり。

| 仮説 | 判定 | 根拠（区別: 公式声明／コミュニティ慣行／推測） |
|---|---|---|
| H1: 制御系も参照系と同じ latent-append 機構 | **裏付け** | wheel実コード `ic_lora.py _create_conditionings`→`VideoConditionByReferenceLatent`（**公式コード**）＋削除済みフォーク・ComfyUIノードも同型。R1のFLAG-2（README記載 `VideoConditionByKeyframeIndex` との名称食い違い）はR3のwheel実コード精読で解消＝実経路は ReferenceLatent |
| H2: 制御動画は外部前処理器で特定視覚形式に事前レンダリング必須 | **強く裏付け** | 公式trainer `compute_reference.py` が「ユーザーが制御動画を用意」設計と明記・default=Canny（**公式声明**）。wheelに前処理コードは皆無（grep確認）。Pose=DWPose風カラー骨格が慣行（**コミュニティ慣行**）。線色の厳密規約は公式文書に無く標準実装で回避 |
| H3: 前処理器がWindows+ローカルvenvで両立 | **裏付け（好条件）** | 第一候補=フォーク実証済み **TorchScript版DWPose**（`hr16/DWPose-TorchScript-BatchSize5` 135MB ＋ `yolox_l.torchscript.pt` 218MB）→ `.venv-engine`の既存torchのみで**追加pipなし**。Canny=cv2（導入済み）。フォールバック=rtmlib（onnxruntimeのみ・Apache-2.0） |
| H4: 2.3用制御アダプタがHFで入手可能 | **修正のうえ裏付け** | ⚠️LTX-2.3-22bに単体 Pose/Canny/Depth は**存在しない**（単体は旧19b世代のみ・2.3流用は効果ゼロ報告）。2.3の制御系＝**Union-Control**（Canny+Depth+Pose統合・654,465,352 bytes・`Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control`）とMotion-Track-Controlの2本。フォークのダウンロード定義とも整合 |
| H5: 凍結API純加算で載る | **裏付け（新フィールド不要）** | 既存 `POST /upload/video`＋`reference_video_id`＋`loras[].name` をそのまま転用。制御タイプはレジストリ側メタデータで表現 |

主要出典:
- 世代の食い違い（19b専用の効果ゼロ報告）: https://huggingface.co/Lightricks/LTX-2.3/discussions/3
- ICLoraPipeline機構・README: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md
- 前処理はユーザー責務（default=Canny）: https://raw.githubusercontent.com/Lightricks/LTX-2/main/packages/ltx-trainer/docs/dataset-preparation.md
- Pose/Canny/Depthの期待形式: https://docs.ltx.video/open-source-model/usage-guides/ic-lo-ra
- Union-Control形式ディスカッション: https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/discussions/1
- チュートリアル（DWPose・strength 1.0）: https://ltxworkflow.com/resources/tutorials/ic-lora-ltx-2-3-complete-guide
- ライセンス（Community License）: https://github.com/Lightricks/LTX-2/blob/main/LICENSE
- **追補R4（2026-07-04）**: ComfyUI公式の `LTX-2.3_ICLoRA_Union_Control_Distilled.json` が「同一Union LoRA＋前処理ノード（DWPose/Canny/VideoDepthAnything）切替」構成であることをJSON実体で確認＝本設計と同型。**プロンプトは制御種に言及せずシーン記述のみ**（G5目視のプロンプト設計に適用）。詳細=[`IC_LORA_PHASE_C_RESEARCH.md`](IC_LORA_PHASE_C_RESEARCH.md) 追補R4。

**⚠️割れている点を隠さない — `ref0.5` の意味**: R1=「downscale factor 2（参照を出力の0.5倍で内部使用）」／R3フォーク考古学=「デフォルト参照強度の可能性」と解釈が割れた。公式docsの「参照の width/height は 64 で割り切れること（= factor 2 × VAE 32×）」記述はR1（factor=2）を支持するが、**確定はGate 0で現物のsafetensorsメタデータを読む**まで保留。

## 3. 設計（監督決定の具体化）

1. **ターゲットアダプタ = LTX-2.3-22b Union-Control 一本**（654MB）。1つのアダプタが canny/pose/depth すべてを担い、どの制御信号を入れるかは前処理側で決まる。Motion-Track・In-Outpainting・Deblur等は Phase D以降。

2. **制御タイプ = canny + pose の2種**。
   - **canny**: `cv2.Canny`（公式default・追加依存ゼロ）。E2E配線の疎通・検証を最小コストで通すための**先行スライス**。
   - **pose**: DWPose TorchScript（本命=「動き維持で内容置換」）。YOLOX person detector（`torch.jit.load`）＋DWPose keypoint（TorchScript・simcc）で OpenPose準拠の骨格を描画。
   - depth はスコープ外（Phase D。前処理器の選定が未収束）。

3. **API形状 = レジストリ駆動・リクエストスキーマ完全無変更**。凍結API契約は不変。クライアントは Phase B と同一で `loras:[{name:"pose-control"}]` ＋ `reference_video_id`（生動画をアップロード）。制御タイプは**論理アダプタ名 → 前処理種別**のマッピングで切替。
   - `config.py:85` の `ic_loras: dict[str, str]` を **`dict[str, str | IcLoraEntry]`** へ拡張（`IcLoraEntry = {path: str, preprocess: "none"|"canny"|"dwpose"}`）。**後方互換 = 文字列値は従来どおり受理**（`preprocess=none` とみなす）。
   - `config.yaml:42-43` の拡張例:
     ```yaml
     ic_loras:
       # 後方互換: 文字列値 = preprocess none（Phase B のまま動く）
       pixel-spatial-upscaler-x2: "./models/ltx-2.3-ic-lora/pixel-spatial-upscaler/ltx-2.3-22b-ic-lora-pixel-spatial-upscaler-x2-0.9.safetensors"
       # Phase C: 同一 Union-Control ファイルを2つの論理名で公開し preprocess で切替
       canny-control:
         path: "./models/ltx-2.3-ic-lora/union-control/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
         preprocess: canny
       pose-control:
         path: "./models/ltx-2.3-ic-lora/union-control/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
         preprocess: dwpose
     ```
   - `services/lora_registry.py`: `resolve()` は現在 `(path, strength)` を返す（L33-51・値は文字列前提 L28）。**`(path, strength, preprocess)` を返すよう拡張**し、辞書値・文字列値の両方を受理。`resolve` を呼ぶ上流（`services/pipeline_manager.py`）で `preprocess` を payload まで運ぶ。

4. **前処理の実行場所・タイミング = engine worker（`.venv-engine`）内・generate時**。サーバー venv（`.venv`）には cv2/torch が無いため前処理は必ず engine 側。
   - **挿入点 = `engine/worker.py` の `_do_generate`（L198-204）**。ここで `reference_video` の生パスから `ic_reference` を組み立てている。Phase Cは、`preprocess != none` の場合に**生の参照動画 → 制御動画（mp4）へ変換してから** `ic_reference` のパスに差し替える一段を挟む。`preprocess` 種別は payload の `reference_video` ブロックに新フィールドとして運ぶ（`services/ltx_runner.py:816-820` の `reference_payload` に `preprocess` を追加）。
   - **前処理モジュール = 新設 `engine/preprocess/`**（フォークの `video_processor_impl.py` のProtocol設計をレシピとして参照するが**コードは書き直す**）。動画→動画のフレーム単位変換。フォークと同型の設計:
     - `FrameProcessor` Protocol: `process(frame_bgr: np.ndarray) -> np.ndarray`（1フレーム入力→制御信号1フレーム出力）。
     - `CannyProcessor`: `cv2.Canny(gray, 100, 200)` → 1ch→3ch展開（フォーク `video_processor_impl.py:44` の値を踏襲）。64の倍数へのpad/crop整合は生成解像度側で担保されるため、前処理は入力寸法を保つのみ。
     - `DwposeProcessor`: YOLOX（TorchScript）で人物検出 → DWPose（TorchScript・simcc）でkeypoint → OpenPose準拠の limb_seq/color table で骨格描画。モデルは初回generate時にロードしプロセス内キャッシュ（毎ジョブ再ロードしない）。
     - ドライバ: PyAV等で逐次デコード → 各フレームを `FrameProcessor.process` → mp4へエンコード。FPS・尺は入力を保存（wheel が frame_cap で切り詰めるため前処理側でのFPS変換は不要）。
   - **前処理済み動画のキャッシュ（video_id × preprocess種別キー）はオプションスライス**。初版はキャッシュ無しでよい（実測で遅ければ追加）。

5. **strength = 1.0固定を維持**。公式tutorialが「1.0未満は reference が pop/bleed-through する」と警告。`ltx_runner.py:817` の固定 `strength: 1.0` を制御系でもそのまま使う。可変化はスコープ外。

6. **同尺・64グリッド制約**。制御動画は出力と positionally aligned = 同尺前提（wheel `load_video_conditioning` は `frame_cap` で先頭から切り詰めるのみ・フレーム伸縮なし）。解像度は既存の `÷64` バリデータ（`api/models.py:98-101`）と整合。制御動画の尺が生成尺より短い場合の警告をどこに入れるかは実装時に determine（前処理ドライバでフレーム数を計測しWARNログが最小）。

7. **モデルファイル配置**（ダウンロードは実装セッションで実施・本ワークオーダー執筆時点では未取得）:
   - Union-Control LoRA 654MB → `models/ltx-2.3-ic-lora/union-control/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`（repo `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control`）
   - DWPose 2ファイル → `models/preprocessors/`（新設）: `yolox_l.torchscript.pt` 217,697,649 bytes（repo `hr16/yolox-onnx`）＋ `dw-ll_ucoco_384_bs5.torchscript.pt` 135,059,124 bytes（repo `hr16/DWPose-TorchScript-BatchSize5`）

## 4. Gate 0（実装前検証・最重要・実装コードより先に消化する）

制御アダプタと前処理器は「未読の現物」に依存する箇所があるため、**実装前に3点を実測で確定**する。

- **G0-a: Union-Control LoRAのsafetensorsメタデータ実読** → **✅完了（2026-07-04）**: メタデータは `reference_downscale_factor: "2"`・`model_version: "2.3.0"`。wheel `_read_lora_reference_downscale_factor` の戻り値=**2**（既知良好のUpscaler x2でも同リーダーで2を確認＝リーダー妥当性も再確認）。→ **factor=2ブランチ確定＝既存ガード（`engine/pipeline/fast_video_pipeline.py:271-290`・L544・L551-557）は無変更でそのまま通る。緩和スライス不要**。ファイルは `models/ltx-2.3-ic-lora/union-control/` に配置済み（654,465,352 bytes=フォーク配布定義と完全一致）。
- **G0-b: DWPose TorchScriptのスループット実測スモークテスト**（**未消化・実装セッションで実施**）。数百フレームの骨格レンダを実行し fps/VRAMを実測（R2はDWPose固有ベンチを発見できず確度低〜中）。生成本体のVRAM天井と前処理ピークが競合しないことを確認。TorchScriptモデル2ファイルは `models/preprocessors/` にダウンロード済み（yolox_l 217,697,649 bytes／dw-ll_ucoco_384_bs5 135,059,124 bytes=いずれも期待値一致）。
- **G0-c: LoRAキー構造の解決性確認** → **✅完了（2026-07-04）**: `load_ic_lora_pairs`（CPUのみ）でUnion-ControlとUpscaler x2の両方をロードし比較。**ペア済みprefix 480個・集合として完全同一・rank=64・shape一致**（lora_A (64,4096) / lora_B (4096,64) bf16）。Upscalerは Phase B G1/G2 で transformer への解決実証済みのため、Union-Control も `attach_ic_loras` でそのまま解決可能（0マッチWARNは出ない）。**キー正規化・リネームマップの変更不要**。

## 5. 実装スライス（順序付き）

**canny疎通を先行させ、pose を後段に**（poseは前処理が重くGate 0-b依存のため）。

1. **スライス1: レジストリ・config拡張（前処理なし配線）**
   - 対象: `config.py:85`（型拡張＋`IcLoraEntry`）／`config.yaml:42-43`（辞書エントリ追加）／`services/lora_registry.py:33-51`（`resolve`が`preprocess`を返す・辞書/文字列両受理）／`services/pipeline_manager.py`（`preprocess`を下流へ）／`services/ltx_runner.py:816-820`（`reference_payload`に`preprocess`）
   - 完了条件: 既存Phase B pytest（`tests/test_ic_lora_api.py`）green維持・文字列値エントリが従来どおり解決・辞書値エントリが `preprocess` 込みで payload まで到達（mock backendで確認）。

2. **スライス2: engine前処理段（canny）**
   - 対象: 新設 `engine/preprocess/`（`FrameProcessor` Protocol・`CannyProcessor`・video→videoドライバ）／`engine/worker.py:198-204`（`preprocess != none` で生動画→制御動画変換を挿入し `ic_reference` を差し替え）
   - 完了条件: 生動画upload→`canny-control` generate→完走。制御動画（エッジmp4）が中間生成物として得られ、サンプルフレーム目視でエッジが妥当。

3. **スライス3: DWPose前処理**
   - 対象: `engine/preprocess/`（`DwposeProcessor`・TorchScriptモデルのロード＆プロセス内キャッシュ・OpenPose骨格描画）／`models/preprocessors/` 配置
   - 完了条件: `pose-control` generate完走。骨格mp4のサンプルフレーム目視で関節・肢が妥当。G0-bのスループットが実用範囲。

4. **スライス4（オプション）: 前処理キャッシュ**
   - 対象: `(video_id, preprocess種別)` キーの前処理済み動画キャッシュ。実測で前処理が律速な場合のみ。

## 6. 検証ゲート（Phase B同型＋制御系固有）

| # | 内容 | 期待 |
|---|---|---|
| G1 | 回帰byte-match（LoRA off・本番per-layer経路） | T2V/I2V基準SHA完全一致・peak_vram不変・pytest全green（Phase B G1と同値） |
| G2 | 非汚染トグル（pose→なし→canny→なし） | 「なし」ジョブが base出力とSHA一致＝attach/detach漏れ・前処理の副作用なし |
| G3 | VRAM/速度 | 前処理込みピークが既存天井（8440MB級）を超えない・前処理時間を実測ログ（DWPoseのfps/VRAM記録） |
| G4 | API e2e | 生動画upload→`pose-control` generate→取得完走・metadataに制御情報（アダプタ名・preprocess種別）記録・偽video_id→404 |
| G5 | 目視（720p級・映画トレイラー風・賑やかな題材） | 「動き維持で内容置換」の成立確認（例: 踊る人物の参照→別キャラクター化）。目視条件は 1280×768 以上・波/静的部屋NG・賑やかな町＋セリフ（[`PHASE3_KEYFRAME_VISUAL_VERIFICATION.md`](PHASE3_KEYFRAME_VISUAL_VERIFICATION.md) と同基準） |
| 制御系固有 | 前処理出力の形式検証 | 骨格動画・エッジ動画のサンプルフレームをユーザー目視に供する（骨格の関節・色が妥当か） |

- G1〜G4は客観（SHA/メトリクス主体）。G5＋制御系固有ゲートは目視。目視題材は賑やかな町＋セリフ系。
- 実GPU実行はワーカーログ `logs/ltx_worker.log` の `peak_vram_mb`（`engine/worker.py:230` の `GENERATED_OK`）を一次ソースとする。

## 7. スコープ外（Phase Cではやらない）

- depth・Motion-Track・In-Outpainting・Deblur・Ingredients・LipDub・HDR等の他アダプタ
- 19b世代アダプタの流用（効果ゼロ報告・非対応）
- strength可変化・`conditioning_attention_mask` 露出
- 前処理キャッシュ（スライス4＝オプション扱い・初版不要）
- Gradio UI露出（APIのみ）
- rtmlib への切替（TorchScript版DWPoseが精度/速度で問題を出した場合のフォールバックとしてのみ記載）

## 8. リスクと未確定事項

- ~~`ref0.5` の意味が未確定~~ → **✅解消（2026-07-04・G0-a）**: 現物メタデータで `reference_downscale_factor=2` を確認。`ref0.5`=「参照を出力の0.5倍で内部使用」（R1解釈が的中）。既存ガード無変更。
- **骨格の線/関節の色規約が公式未明文化**（"skeleton visualization with lines connecting keypoints" 止まり）。DWPoseの標準カラー骨格（フォーク/ComfyUIと同系の描画）を採用してリスク回避するが、Union-Controlが期待する厳密な色と食い違う可能性はゼロではない。G5目視＋制御系固有ゲートで検出する。※追補R4で公式2.3ワークフローが `DWPreprocessor`（controlnet_aux系DWPose）を使うことを確認済み＝同系描画なら食い違いリスクは小。
- **DWPoseスループット未実測**（R2確度低〜中）。Gate 0-b で確定。遅ければキャッシュ（スライス4）で緩和。
- **19b非互換**: 誤って19b世代の単体アダプタを使うと「エラーは出ないが効果ゼロ」。config登録は 2.3-22b Union-Control のみに限定する。
- **onnxruntime-gpuの整合リスク（フォールバック採用時）**: rtmlibへ切替える場合、CUDA 12.8 に対し onnxruntime-gpu を 1.20〜1.26 帯にpin要（1.19未満=cuDNN8系NG・1.27以降=CUDA12廃止予定）。第一候補のTorchScript版なら追加依存ゼロでこのリスクを回避。
