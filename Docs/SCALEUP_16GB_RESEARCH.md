# 16GB 高解像度スケールアップ調査 — VRAM 16GB で高解像度生成を実現するための調査結果集

> ## ✅ 達成（2026-06-30 後半）
> **このスケールアップ（残課題C＝720p）は完了した。** 1280×768/121f を本番 API で ~167–171秒で完走（16GB・OOM なし）、crop で 1280×720 配信、**連続3本も commit 枯渇せず PASS**。採用レシピ＝**use_component_files=true（Path B）＋ LTX_KEEP_RESIDENT=0 ＋ block_swap_blocks_on_gpu=8 ＋ vae_spatial_tile_size=512 ＋ vae_temporal_tile_size=64**。
> 真の難所3点の実測結論：①段間遷移＝フォークは meta 退避で両段同時滞在なし＝非問題化。②block-swap 深度＝bs=8 で denoise ~6–8GB の大余裕（深掘り不要）。③共存性＝tiling＋GGUF＋block-swap＋component-files が 720p で同時に正常動作を実機確認。
> 詳細・原因分析（keep_resident の Gemma out-of-place 移動 crash、comp=1 による commit 束縛、マシンスペック比較）は **VERIFICATION_LOG §10** と **NEXT_SESSION_HANDOFF.md 冒頭 ▶▶▶▶**。以下（本バナー以降）は着手前の調査記録（有効・参照用）。
>
> ※`LTX_KEEP_RESIDENT` は当時の手順。2026-08-02 に環境変数の経路は撤去され、現在は API の `keep_resident` フィールド（`POST /generate`・`POST /generate/chain`。既定 `false`＝上記レシピの keep=0 と同じ状態）で指定する（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §48）。

作成日: 2026-06-30

**位置づけ**: VRAM 16GB で高解像度生成を実現するための調査結果集（先行事例レシピ＋我々のコードのレバー棚卸し）。**16GB対応の最大の難所に関する正本**。

**目標**: 最終目標は **1280×720 まで生成できれば充分**。より大きいサイズ（1080p/長尺）の情報も将来用に付録として保全（破棄しない）。

**関連**: [VERIFICATION_LOG.md](VERIFICATION_LOG.md)（§9.8 音声Phase1）、[NEXT_SESSION_HANDOFF.md](NEXT_SESSION_HANDOFF.md)（残課題C＝720pスケールアップ）、[LTX23_REFERENCE.md](LTX23_REFERENCE.md)。

---

## 1. 結論（要約）

- 720pレバーの **4/5 は我々のフォークに既に配線済み・既定OFF**。欠けているのは FFN チャンキングのみで、それは主に長尺（シーケンス長支配）向けで **1280×720 の空間スケールには必須でない**。
- よって 720p は「大規模な移植」ではなく **「既存ノブを正しい値でONにして再現確認する」タスク**。
- ただし先行事例にも答えが無い“真の難所”が3点残る（§4）。

---

## 2. レバー棚卸し（我々のコードに何が実装済みか）

| レバー | 実装状況 | パラメータ名 | 既定値 | ファイル:行 | 役割 |
|---|---|---|---|---|---|
| 二段パイプライン（stage1=height//2,width//2 → ×2 spatial upsampler） | ✅実装済 | `stage1_height`, `stage2_height` | stage1=1/2 | `distilled.py:116-117`, `ti2vid_two_stages.py:137-144,76` | 高解像度の本体戦略・直接720p生成を回避 |
| Attention tiling（SDPA を `F.scaled_dot_product_attention` グローバルパッチでタイル化、クエリ長>tileで作動、マスク/causal/短シーケンスはフォールバック） | ✅配線済・OFF | `attention_tile_size` | 0 | `attention_tile_service.py:40-41,63-93`, `ltx_fast_video_pipeline.py:239` | 推奨値256/512/1024/2048。高解像度で爆発するattention活性を抑える要レバー |
| VAE 空間タイル | ✅配線済・OFF | `vae_spatial_tile_size` | 0=ライブラリ既定768px | `ltx_pipeline_common.py:36`, `ltx_fast_video_pipeline.py:776` | `SpatialTilingConfig(768px, overlap64px)`、最小64px・32倍数 |
| VAE 時間タイル | ✅配線済・OFF | `vae_temporal_tile_size` | 0=ライブラリ既定80フレーム | `ltx_pipeline_common.py:43`, `:777` | `TemporalTilingConfig(80f, overlap24f)`、最小16f・8倍数 |
| Block-swap | ✅配線済 | `block_swap_blocks_on_gpu` | 0(OFF=全48ブロックGPU常駐) | `block_swap_service.py:51-53`, `ltx_fast_video_pipeline.py:64` | 全48ブロック中N個のみGPU、残りCPU-RAM入替。blocks_on_gpu=20で約8-10GB削減（worker内コメント） |
| FFN チャンキング | ❌未実装 | — | — | `ltx_core/.../feed_forward.py`は`Sequential(GELUApprox,Identity,Linear)`で分割なし | 主に長尺向け、720p空間スケールには非必須。**将来項目としての管理は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-44へ移設（2026-07-27）** |

**generate() シグネチャ**: `prompt, seed, height, width, num_frames, frame_rate, images, output_path, num_steps` ＋ `sigma_schedule`（distilled/linear/linear_quadratic/beta）/ `denoising_loop`（euler/gradient_estimating/res2s）/ STG。**注意: worker（`_ltx_worker.py`）のgenerateプロトコルは標準引数のみ受け、VRAMレバーはload(create)時に固定**。

**重要な訂正**: 旧メモリの「attention tiling／FFNチャンキングを移植」という前提は半分古い。**attention tiling は SDPA パッチとして実装済み**。未実装は FFN チャンキングのみ。

---

## 3. コミュニティ実証レシピ（出典付き・信頼度明記）

| 解像度 | フレーム | VRAM | 主要設定 | モデル世代 | 信頼度 | 出典 |
|---|---|---|---|---|---|---|
| 1280×768 final(2段) | — | 12–16GB | Stage1=640×384→Stage2アップサンプル, Q4 GGUF | LTX-2.3 | 広く再現(手順記事) | WaveSpeed |
| 1280×704 | ~25秒(24fps) | 16GB | Model Memory Usage Factor Override | LTX-2 | 単発(コミュニティ) | Kijai LTXV2_comfy discussions |
| 1536×864 | ~18秒(30fps) | 16GB | 同上 | LTX-2 | 単発 | 同上 |
| 1280×736(720p) 10秒 | — | 6GB | RTX3050+44GB RAM, <15分 | LTX-2 | 単発(極端例) | Kijai discussions |
| 960×544 | — | 8GB | 成功報告 | LTX-2 | 単発 | 同上 |
| 832×480以下 | — | 8GB | Q4 distill GGUF + Gemma Q2_K, --novram | LTX-2 | 単発(8GBは「no promises」) | 同上 |

（長尺/大解像度は§付録Aへ）

**VAE tiled decode 公式デフォルト**（信頼度高, `Lightricks/ComfyUI-LTXVideo/tiled_vae_decode.py`）: 空間タイル数 default 4（1-8）/ 空間 overlap 1 latentフレーム / temporal_tile_length default 16 latentフレーム / 時間 overlap 1。推奨スタート＝空間 4×4 ＋ overlap1 ＋ 時間 16/overlap1、OOM時タイル数増。

**attention**: SDPAはO(n²)materializeで長シーケンスOOMしやすい。FlashAttention/SageAttention（`--use-sage-attention`）はタイル化O(n)で高解像度activation削減に有利（コミュニティ実証、8GBでLTX-2稼働報告あり）。

**block-swap実数**（注意: 多くは旧LTX-Video/Wan/汎用記事由来、22Bのレイヤ数に紐づく確定値ではない）: 一般則＝OOM消えるまで増やす最大~40、12GB向け記事で35(8-10GB狙い)/24(10-12GB狙い)。

---

## 4. 我々だけが確かめる必要がある“真の難所”（先行事例に答えが無い）

1. **段間(stage1→stage2)の遷移スパイク** ＝ 16GB OOMの実測ポイント（Comfy-Org #11726: 1080p>200fが2段目遷移でpeak14.9GB→OOM、ヘッドルームほぼ0）。回避策＝**段間でlatent保存＋モデル完全アンロード→再ロード**（サンプラー分割）。我々の二段経路が両段を同時保持するか要確認＝最大の未知。
2. **22Bのblock-swap深度** ＝ 外部に確定値なし。自前で詰める唯一の値（全48ブロック）。
3. **共存性** ＝ tiling＋GGUF＋block-swap同時動作は、配線済みでも高解像度で同時検証された記録なし。ChunkedFFNも「GGUF/block-swap併用の記載なし・16GB単GPU実績なし(24GBのみ)」。

---

## 5. 推奨レシピ（1280×720狙い・合成）

- 二段で1280×720（stage1=640×360相当→×2）。※解像度は÷32/÷64契約に合わせ調整要（例: 1280×768生成→720pクロップ、既存presets参照）。
- VAEタイルON（空間512px / 時間64f 程度から、公式既定は768px/80f）。
- `attention_tile_size=512`（クエリ長>512でタイル化）。
- block-swap深め（`block_swap_blocks_on_gpu` を小さく＝より多くCPU退避、OOM消えるまで段階調整）。
- GGUF Q4_K_M transformer ＋ GGUF Gemma はそのまま（既存）。Gemmaはtext pass後CPUオフロード。
- 任意: SageAttention系へ寄せる（activation削減、ただし我々環境での導入可否は未検証）。

---

## 6. 推奨の進め方（規律ある最小手順）

1. **読解のみ（GPU不要）**: `ti2vid_two_stages.py`精読→段間VRAM遷移の扱い・create()ノブが両段に効くか・latent保存+アンロードが要るか確定。
2. **計装つき1回生成**: 上記レシピでdedicated＋shared採取しながら1本だけ→収まるか/どこでスパイクするか実測。
3. **必要時のみ**: block-swap深度調整、段間アンロード実装。FFNチャンキングは将来の長尺対応で初めて検討。

---

## 付録A. より大きいサイズ・長尺の情報（将来用・破棄しない）

- 1920×1088/800f ~16.5GB・900f ~18.5GB / V3単GPU `ffn_chunks=16` / LTX-2(22B系) / 単発(作者ベンチRTX4090) / RandomInternetPreson
- 1080p>200f / 16GBでstage2遷移OOM(peak14.9GB) / Comfy-Org #11726
- **ChunkedFFN**（RandomInternetPreson, LTX-2専用）＝**将来項目としての管理は[`PENDING_TASKS.md`](../../Nz-LTX23-frontend-AviUtl2/Docs/PENDING_TASKS.md) §3-44へ移設した（2026-07-27）**。機構の要旨（FFNのhidden4倍展開の中間テンソルをシーケンス分割して最大~8x削減）と参考値（`ffn_chunks`: 600f→8 / 800f→12-16 / 900f+→16-24。実績24GBのみ・GGUF/block-swap併用の記載なし）も同項に転記済み。出典URLは付録Bに残す。
- 16GB起動フラグ実例（手順記事）: `--use-sage-attention --novram --cache-none --disable-smart-memory --preview-method taesd`、RAM32GB+swap64GB(swappiness=6)。`--novram`がモデルスワップ時スパイク平滑化。

---

## 付録B. 出典一覧

- RandomInternetPreson ChunkedFFN: https://github.com/RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management
- 公式 tiled VAE decode: https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/tiled_vae_decode.py
- 16GB OOM実測 issue: https://github.com/Comfy-Org/ComfyUI/issues/11726
- WaveSpeed 2段パイプライン手順: https://wavespeed.ai/blog/posts/ltx-2-3-comfyui-setup-two-stage-pipeline/
- 低VRAM報告: https://huggingface.co/Kijai/LTXV2_comfy/discussions/32
- LTX-2 GGUF VRAM階層ガイド: https://dev.to/gary_yan_86eb77d35e0070f5/how-to-install-and-configure-ltx-2-gguf-models-in-comfyui-complete-2026-guide-1d3m
- SageAttention issue: https://github.com/Lightricks/ComfyUI-LTXVideo/issues/421
