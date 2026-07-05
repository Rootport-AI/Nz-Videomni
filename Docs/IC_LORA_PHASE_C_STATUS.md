# IC-LoRA Phase C — 現状ステータス（✅完了・全ゲートPASS＝G5ユーザー受容済み・mainマージ済 2026-07-04）

- 更新: 2026-07-04（branch `feature/ic-lora-phase-c`・base=main `1a3dfec`）
- 併読: [`VERIFICATION_LOG.md` §22](VERIFICATION_LOG.md)（全ゲート詳細数値）／[`IC_LORA_PHASE_C_WORKORDER.md`](IC_LORA_PHASE_C_WORKORDER.md)（設計・ゲート定義・リサーチ根拠）／[`IC_LORA_PHASE_B_STATUS.md`](IC_LORA_PHASE_B_STATUS.md)（前段=参照系アダプタ）

---

## ⚠️ 一行結論

**成立**: 制御系IC-LoRA（**LTX-2.3-22b Union-Control** 一本）＋**engine内前処理段の新設**（canny＝エッジ抽出／pose＝DWPose骨格）が完成し、看板機能「動きを維持したまま内容を差し替える」を実装した。ユーザーは従来どおり生の動画をアップロードするだけで、サーバー側が制御信号へ変換して生成する。客観ゲート（G0-bスループット実測・G1回帰byte-match・G2非汚染トグル・G3 VRAM/速度・G4 API e2e）は全PASS。前処理は生成時間の~5%・VRAM天井（Phase B帯8440〜9527MB）を超えない。**G5もユーザー受容済み（2026-07-04）＝Phase C完了・mainマージ済**。

## 何ができるようになったか

- **制御系アダプタが使える**: `canny-control`（エッジ）と `pose-control`（DWPose骨格）の2論理名。いずれも同一のUnion-Control safetensors（654MB）を指し、**どの制御信号を入れるかは前処理種別（レジストリ側メタデータ）で切替**。Phase Bで完成したforward時GPU LoRA適用機構をそのまま使う（機構は無変更）。
- **engine内前処理段**: サーバーは生の参照動画を受け取り、engine worker（`.venv-engine`）内で**制御動画（`control_<kind>.mp4`）へ変換してから**IC-LoRA参照に差し替える。ユーザーはエッジ抽出や姿勢推定を意識しなくてよい。前処理は動画→動画のフレーム単位変換（cv2ドライバでFPS・フレーム数・寸法を保存）。
- **API形状はPhase Bと完全同一**: 凍結API契約は不変・リクエストスキーマ無変更。クライアントは `POST /upload/video` で参照動画→`video_id`、`loras:[{name:"pose-control"}]`＋`reference_video_id`。制御タイプはサーバー側レジストリの論理名で解決。
- **前処理なし経路（`preprocess=none`）は完全無変更**: 文字列値エントリ（Phase Bの `pixel-spatial-upscaler-x2` 等）は従来どおり動く（後方互換）。

## ゲート結果サマリ（詳細＝VERIFICATION_LOG §22）

| ゲート | 内容 | 結果 |
|---|---|---|
| G0-b | DWPoseスループット実測（GPUスモーク） | **PASS**（1280×768で13.34fps end-to-end・混雑ワーストケース~13人/フレーム。前処理は生成時間の~5%・モデル常駐354.7MB・ループピーク482MB→del+empty_cacheで8.5MBまで完全解放＝denoise天井と非競合。スライス4キャッシュ／rtmlibフォールバック共に不要と監督判断） |
| G1 | 回帰byte-match（LoRA off・本番per-layer経路） | **PASS**（T2V基準SHA `23844b4e…6bb7bf` 完全一致・peak 8440／最小I2V `a511eda4…15c217` 完全一致・peak 9525。pytest 69 passed/1 skipped） |
| G2 | 非汚染トグル（pose→なし→canny→なし） | **PASS**（「なし」2本がJob A基準SHAと完全一致＝attach/detach漏れ・前処理の副作用なし） |
| G3 | VRAM/速度 | **PASS**（制御ジョブpeak 9525/9522≦天井9527。前処理時間実測: dwpose 16.60s(≈7.8fps・ロード込み)／canny 1.12s @129f） |
| G4 | API e2e実機 | **PASS**（metadata に `preprocess`・`reference_video_id` 記録・GET video 200・偽video_id→404・512×320+参照→422 REFERENCE_RESOLUTION_INVALID） |
| G5 | 目視（720p級・映画トレイラー風） | **PASS＝ユーザー受容（2026-07-04）**（成果物=`outputs/visual_review/10_〜13_`・pose/canny両方で「動き維持で内容置換」成立を受容） |

## commit（branch `feature/ic-lora-phase-c`・base=main `1a3dfec`）

| commit | 内容 |
|---|---|
| `5a9de32` | スライス1: レジストリ・config拡張（`IcLoraEntry{path, preprocess}`・`resolve`が`preprocess`返却・後方互換文字列値・`LORA_PREPROCESS_CONFLICT` 400・tests 10→19／pytest 66→69） |
| `80a909c` | スライス2: engine前処理段（canny）＋worker挿入（`FrameProcessor` Protocol・`CannyProcessor`・cv2ドライバ・E2E PASS） |
| `91dc485` | ÷128事前バリデーション（参照付きジョブで width/height%128≠0 → 422 `REFERENCE_RESOLUTION_INVALID`） |
| `058e862` | スライス3: DWPose前処理（`DwposeProcessor`・TorchScript移植・遅延ロード＋release()でGPU解放） |

## アーキテクチャ要点（Phase D以降の担当者向け）

- **制御機構＝Phase Bの流用**: forward時GPU LoRA適用（per-layer-quant経路）は無変更。Union-ControlのLoRAキーはPhase B G0-cで解決性確認済み（480ペア・rank64・Upscaler x2と集合同一）。**IC-LoRA機構を勝手に触らないこと**（Phase Bの注意を継承）。
- **前処理段＝新設 `engine/preprocess/`**: `FrameProcessor` Protocol＋`get_processor`レジストリ。`CannyProcessor`＝フォーク `apply_canny` 忠実移植（64pad→`cv2.Canny(gray,100,200)`→crop→3ch）。`dwpose.py`＝G0-bハーネス／フォーク `DWPosePipeline` のverbatim移植（YOLOX検出→DWPose SimCCバッチ5→OpenPose18点remap→body/hand/face骨格描画）。**コードは書き直しでなくverbatim移植**（挙動同一性を優先）。
- **DWPoseはジョブ毎再ロード＋release()**: ワークオーダーの「プロセス内キャッシュ・再ロードなし」から**意図的逸脱（監督承認）**。再ロード+5〜7s/ジョブと引き換えに、denoise中の355MB常駐を排除（VRAM天井を最優先）。
- **前処理の挿入点**: worker で `preprocess != none` の場合、生の参照動画をジョブ出力ディレクトリの `control_<kind>.mp4` へ変換し `ic_reference` を差し替え。`none` 経路はcv2遅延importで完全無変更。metadata.json の `ic_lora.loras[]` に `preprocess` を加算。
- **新制御タイプの追加**: `engine/preprocess/` に `FrameProcessor` を1つ実装＋`get_processor`に登録＋`config.yaml`に論理名エントリ1つ。

## 既知の制約

- **÷128制約**: 全登録アダプタが `reference_downscale_factor=2` ＝参照は出力解像度の半分でVAEの64格子に載る。**参照付きジョブは出力 width/height が128で割り切れないと必ず失敗する**（512×320はVAE encodeでeinops fail・512×256はPASS）。API層に422事前バリデーション（`REFERENCE_RESOLUTION_INVALID`）を追加して検出。参照無しジョブは無影響。
- **前処理種の競合禁止**: 1ジョブで複数の異なる `preprocess` 種を混在させると 400 `LORA_PREPROCESS_CONFLICT`。
- **strength=1.0固定**: 公式tutorial警告（1.0未満はreferenceのpop/bleed-through）に従い可変化はしない。 → **（2026-07-06 注記）実装済み＝VERIFICATION_LOG §28**（`conditioning_attention_strength`＋`reference_video_strength` を optional 加算・省略時 1.0 で byte 不変）。ここでの警告はノブ①（参照 strength）の話で、本命ノブ②（attention strength）は公式 docs でアーティファクト警告なしと判明。
- **前処理は逐次デコード**: キャッシュ無し（スライス4）。G0-bで前処理が生成時間の~5%と実測されたためキャッシュは不要と判断。

## ✅ G5成果物（客観準備完了・ユーザー目視待ち）

- **Job P**＝pose-control 1280×768/121f/seed12345（`408fd361`・219.7s・peak 8548・PREPROCESS dwpose 18.15s・AAC音声あり）／**Job Q**＝canny-control 同パラメータ（`8c1b5a9c`・196.0s・peak 9541）。
- プロンプト＝映画トレイラー風「老船長が港町を歩く」（**制御種は非言及**＝シーン記述のみ・追補R4のComfyUI公式ワークフロー流儀に準拠・セリフ入り）。
- 成果物＝`outputs/visual_review/10_〜13_`（pose出力／骨格／canny出力／エッジ・README追記済み）。
- **監督の事前目視所見（受容判断ではない）**: 参照の歩行動作・カメラ・群衆構図を維持して別キャラクター（老船長）へ置換成立。canny版は参照の街並み構造をより強く保持・pose版は背景自由度が高い（制御タイプの性質どおり）。**受容判断はユーザー**。

## スコープ外（Phase Cではやらない・Phase D以降）

- depth・Motion-Track・In-Outpainting・Deblur等の他アダプタ
- 19b世代アダプタの流用（効果ゼロ報告・非対応）
- strength可変化・`conditioning_attention_mask` 露出 → **（2026-07-06 注記）strength可変化は実装済み＝VERIFICATION_LOG §28**（`conditioning_attention_strength`＋`reference_video_strength`・省略時 1.0 不変）。`conditioning_attention_mask` の露出は引き続きスコープ外。
- 前処理キャッシュ（スライス4＝G0-bでキャッシュ不要と判断）
- rtmlibへの切替（TorchScript版DWPoseで問題が出た場合のフォールバックとしてのみ記載・G0-bで不要判断）
- Gradio UI露出（APIのみ）

## Pending（ユーザー）→ ✅全消化（2026-07-04）

1. **G5目視受容判断** → ✅**ユーザー受容（2026-07-04）**。`outputs/visual_review/10_〜13_` を目視し「動き維持で内容置換」の成立を受容。÷128制約・DWPose解放方針（release()逸脱）も併せて了承。**Phase C全ゲートクローズ**。
2. **mainマージ判断** → ✅**受容を受けてmainへマージ・push済（2026-07-04・ユーザー指示）**。
