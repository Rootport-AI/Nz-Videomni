# LTX 2.3 リファレンス知識（基礎・参照URL付き）

作成: 2026-06-28 / 性格: **特定タスクに紐づかない LTX-2/LTX-2.3 の一般知識**。解像度契約・2段パイプライン・VRAM スケーリング・16GB レシピを、出典とともに記録する。実機検証の経緯は [VERIFICATION_LOG.md](VERIFICATION_LOG.md)、現状の方針と残課題は [PENDING_TASKS.md](PENDING_TASKS.md)。

> 表記: **[公式]**=Lightricks 公式ドキュメント/コード/論文で確認 / **[実測]**=コミュニティ実測値 / **[推定]**=論理的帰結・フェルミ推定（明記）。

---

## 1. モデルの上限と構成 [公式]
- **構成**: 22B、**動画 ~14B + 音声 ~5B の dual-stream**（bidirectional cross-modal attention）。flow-matching/velocity 予測 DiT。音声は joint（同一 forward）で生成、ほぼ固定費（解像度では増えない）。
- **解像度上限**: 最大 **4K** を謳うが、**単発ネイティブ4K ではなく**、半解像度生成→2段目で **latent を空間+時間タイル分割しながらアップスケール**して到達。
- **尺**: ネイティブ **~10秒**（LTX-2.3 は ~20秒を広告）。ただし高解像ほど早く破綻（[実測] 704×1280 portrait は ~12-13秒で崩壊）。
- **fps**: 24 / 25 / 48 / 50。
- **変種**: `dev`(フル bf16, ~42-44GB, 学習可)／`dev-fp8`／`distilled`(8+4 step, 高速)／GGUF コミュニティ量子化／spatial upscaler x2・temporal upscaler x2。

## 2. VAE 圧縮とトークン数 [公式＋推定]
- **VAE 圧縮 = 32×(幅) × 32×(高) × 8×(時間)、潜在 128ch、transformer patch=1**。公式 LTX-Video 論文（VAE 32×32×8, 1:8192 px/token）＋ ltx-core README の実例 `[B,3,33,512,512] → [B,128,5,16,16]`（512/16=32×, 33→5=8×）で確認。
  - ※「8×8×8」とする二次情報は**誤り**（自分の挙げた実例とも矛盾）。**32×空間が正**。
- **トークン数** = `(W/32) × (H/32) × (1 + (T-1)/8)`。
  | 解像度 / フレーム | トークン |
  |---|---|
  | 384×256 / 9 | ~190 |
  | 768×512 / 121 | ~6,100 |
  | 1280×704 / 121（5秒） | ~14,000 |
  | 1280×704 / 241（10秒） | ~27,000 |
  | 1920×1088 / 121 | ~32,600 |
- **含意**: 32×圧縮のおかげで 720p/5-10秒でもトークンは1.4〜2.7万と**小さい** → アクティベーションは数GB規模に収まる（§4 と整合）。

## 3. 解像度契約：2段パイプラインと ÷32/÷64 [公式＋推定]
- **distilled は常に2段（two-stage）**：`generate(width,height)` は**最終出力解像度**。内部で **Stage1 を半分(W/2,H/2)で8ステップ生成 → 2倍 spatial upscaler → Stage2 を最終解像度で4ステップ再デノイズ → VAE decode**。[公式] `DistilledPipeline`（"8 steps stage 1, 4 steps stage 2"）。
  - **spatial upscaler は別チェックポイント**（`ltx-2.3-spatial-upscaler-x2`）で、README に "Required for current two-stage pipeline" ＝必須。
  - 単発 `TI2VidOneStagePipeline` は存在するが **"primarily for educational purposes"** かつ dev/full 向け。**高速 distilled では単発は選べない**。
- **÷32 は公式のハード制約**（VAE 32×空間圧縮。モデルカード "divisible by 32"）。フレームは **8n+1**。
- **÷64 は2段の論理的帰結 [推定]**：Stage1 が半分で走り、その半分も ÷32 でneed → 最終は ÷64。公式コード `assert_resolution`(two-stage 分岐) に入るが、散文ドキュメントには ÷32 までしか明記なし。ComfyUI も散文は ÷32 だが two-stage モードでは実質 ÷64。
- **720p の作り方**：1280×720 は **÷32 ですらない**（720/32=22.5）ので native 生成不可。**1280×768（÷64）で生成 → 上下12pxセンタークロップで 720 化**が定石（1280×704 は <720 でパディングが要るため劣る）。

## 4. VRAM スケーリング：支配項は「重み」 [実測中心]
- **LTX-2 の VRAM は重みが支配的、アクティベーションは小さい**。[実測] Kotonia: peak はほぼ解像度非依存（fp8_cast で 57.9→59.1 GiB）、**アクティベーションは ~7GiB**。fp8_cast で 22B が 40→24 GiB（cold）。
- [実測] Apatero: 720p/72f ≈ **14GB(fp16) / 9GB(fp8)**。480p/120f ≈ 12/7GB。
- **コスト3軸は独立**（重み／アクティベーション／VAEデコード）。16GB に収めるには軸ごとに別の道具を積む（§5）。
- **非オフロード可能なスパイク = 2段目アップスケール/refine**。[実測] 16GB OOM 報告（ComfyUI #11726）はほぼ **1080p へのアップスケール**段で発生（重みは退避できても、この段の中間/活性は退避できない）。720p ターゲットなら Stage2 も ~14k トークンで収まる。

## 5. 16GB で大きく/長くするテクニック [公式/実測/推定混在]
| 技術 | 削る対象 | 長尺に効く？ | メモ |
|---|---|---|---|
| GGUF/fp8 量子化 ＋ block-swap / sequential offload / `--reserve-vram` | **重み** | △ | 支配項。我々が採用（GGUF transformer＋GGUF Gemma） |
| **FFN チャンキング**（RandomInternetPreson ノード） | **アクティベーション** | ◎ | FFN中間を系列分割。数学的に等価。"81→800+フレーム"・~8-10倍削減 [実測on author] |
| SDPA / FlashAttention | attention 活性 | ◎ | O(N²)→O(N)。我々は SDPA 既定 |
| **tiled VAE decode**（空間+時間） | **VAEデコード** | ○ | 末端スパイク対策。2048²で 56→8GB [実測] |
| 2段パイプライン（半解像度→アップスケール） | 全体のピーク平準化 | △ | 公式推奨。ただし**アップスケール段が16GBの最大OOM地点** |
| recursive extend / windowed denoise | — | ◎ | peak が窓サイズで決まり総尺に依存しない。LTX は Kijai NativeLooping 等（実験的） |
| 外部の専用アップスケーラ AI（720p→1080p） | — | — | LTX 内蔵 Stage2 を高解像で走らせず、別パス/別VRAM に逃がす。**16GBで1080pを得る推奨ルート** |

## 6. 16GB 実用性の見立て [実測＋推定]
- **720p / 5-10秒は 16GB で実現可能**。[実測] note.com（RTX 4070 Ti SUPER 16GB）が **1280×720 / 121f(5秒) / 音声付き** を `--reserve-vram 5` で ~4分生成成功。
- **1080p を LTX 内部（2段目）で出すのは 16GB の壁**。→ 実用上は **LTX で 720pクラス生成 → 外部の動画アップスケーラ AI で 1080p 化**の分業が筋。
- 我々のアーキテクチャ（GGUF transformer＋GGUF Gemma＋block-swap）は**コミュニティ標準の 16GB レシピと一致**。残る詰めは block_swap 深度・tiled VAE・（長尺なら）FFN チャンキング/extend。

---

## 参照URL
**公式（Lightricks）**
- LTX-2 メイン README: https://github.com/Lightricks/LTX-2/blob/main/README.md
- ltx-pipelines README（2段/1段/distilled の定義）: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md
- LTX-2.3 モデルカード（÷32, 8n+1）: https://huggingface.co/Lightricks/LTX-2.3
- spatial upscaler x2 チェックポイント: https://huggingface.co/Lightricks/LTX-2.3
- LTX-Video 論文（VAE 32×32×8, full spatiotemporal attention）: https://arxiv.org/abs/2501.00103
- LTX-2 論文: https://arxiv.org/pdf/2601.03233
- システム要件（公式は 32GB+ 推奨）: https://docs.ltx.video/open-source-model/getting-started/system-requirements

**VRAM 実測/解析**
- Kotonia fp8_cast 計測（重み支配・活性~7GiB）: https://kotonia.ai/en/articles/ltx2-22b-fp8-cast-quantization/
- Apatero 8GB 最適化（解像度×フレーム別 fp16/fp8 表）: https://apatero.com/blog/ltx-2-8gb-vram-optimization-complete-guide-2025
- ComfyUI #11726（16GB・2段目アップスケールで OOM トレース）: https://github.com/Comfy-Org/ComfyUI/issues/11726

**16GB レシピ/実績**
- note.com（RTX 4070 Ti SUPER で 720p+音声 成功）: https://note.com/automate_nahito/n/n8d751be0bc76
- WaveSpeed 2段セットアップ: https://wavespeed.ai/blog/posts/ltx-2-3-comfyui-setup-two-stage-pipeline/
- WaveSpeed VRAM ティア: https://wavespeed.ai/blog/posts/blog-ltx-2-vram-requirements/
- NVIDIA RTX 動画生成ガイド（÷32, 704/1088 推奨）: https://www.nvidia.com/en-us/geforce/news/rtx-ai-video-generation-guide/

**低VRAM技術**
- FFN チャンキング: https://github.com/RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management
- city96 ComfyUI-GGUF（GGUF ローダ）: https://github.com/city96/ComfyUI-GGUF
- ComfyUI-LTXVideo メモリ管理（tiled VAE 数値）: https://deepwiki.com/Lightricks/ComfyUI-LTXVideo/4.3-memory-management
- 長尺 windowed/extend（Kijai NativeLooping, 実験的）: https://deepwiki.com/kijai/ComfyUI-NativeLooping_testing/4-example-workflow:-ltx-2.3-long-video-generation
