# Phase 3 スライス2「クリップ連結（生成プリミティブ）」設計 — リサーチ結論＋実装計画

> ## ⚠️ 2026-07-03 訂正バナー（最初に読む）
> **本文書の「スパイク✅GO」「フェーズ2/3✅ 全ゲートPASS」は "配管が動く" の意味であり、映像の連続性は達成していない。** ユーザー目視（2026-07-03）でクリップ境界が hard cut・音声も断絶＝**連結は機能未達**と判明。監督が境界フレームの実映像を目視せず配管チェックのみで PASS 判定していた検証手法の欠陥が原因。**現状・根本原因・修正方針の正本は [`PHASE3_CLIP_CONCAT_STATUS.md`](PHASE3_CLIP_CONCAT_STATUS.md)。** 以下の設計/計画は「latent-extend という方針で配管を組んだ記録」として温存するが、そのまま完成扱いにしないこと。

- 作成: 2026-07-02（監督＋ユーザー合意）
- 上位: [`PHASE3_CLIP_CONCAT_WORKORDER.md`](PHASE3_CLIP_CONCAT_WORKORDER.md)（§1 リサーチ指示・§2 スコープ分岐）／[`VERIFICATION_LOG.md` §17](VERIFICATION_LOG.md)（スライス1）
- スコープ確定（ユーザー・2026-07-02）: **最初から厚版（latent 連続）**。薄版（pixel bookend）は decode→再encode 往復で latent 文脈が劣化するため不採用。

> 鉄則: システム Python 不可触（全 project-local venv）。仮説→裏取り（WEB/コード読解）→スコープ確定→実験。engine 改変・契約変更はユーザー チェックポイント。回帰 byte-match をゲートに使う。

---

## 1. リサーチ結論（4 subagent＝コード読解×3＋WEB×2 で裏取り・file:line＋URL）

### 1.1 native extend の所在
- インストール済み wheel（`.venv-engine` の `ltx_core`/`ltx_pipelines` v1.0.0）に **native な自己回帰 extend パイプラインは無い**。KeyframeInterpolation＝補間・Retake＝マスク領域再生成・IC-LoRA＝参照制御であり、いずれも「継続/extend」ではない。
- LTX の正しい長尺 extend は **ComfyUI ノード層**（`LTXVLoopingSampler`/`LTXVExtendSampler`）に在り pip パッケージには無い＝**我々が orchestration を自前再実装**する構図（スライス1 と同じ「部品は wheel・厚い層は自前」）。製品版「Extend」（LTX Studio）はこの hosted ラッパー。

### 1.2 機構（latent レベル）
前クリップ末尾 latent（~2–3 latent frame ≈ 15–17 pixel）を overlap 種→新パス先頭に注入→線形クロスフェードでシーム除去→AdaIN で clip-0 統計へ合わせドリフト抑制→negative-index アンカーで長期記憶。温度圧縮 time factor=8（latent 1 ≈ pixel 8）。

### 1.3 二段 distilled の latent 引き継ぎ（gating 論点・確定）
- `DistilledPipeline`（`distilled.py`）は **Stage1＝ノイズからのフル生成（低解像度・`initial_video_latent` 無し）** → **Stage2＝stage1 の upscale ドラフトを init に部分ノイズを足して refine（`initial_video_latent=upscaled`, `noise_scale=stage_2_sigmas[0]`）**。
- **seed は Stage1 に入れる**のが正（Stage2 は refine 段で、非overlap をゼロ埋めすると refine が壊れる＝当初 subagent 案の誤りを訂正）。Stage1 に前クリップの stage1 空間 tail を注入すれば continuity が既存二段フローを自然伝播する。
- **持ち回すのは stage1(lowres) tail latent**。

### 1.4 メカニクスの決定的裏取り（コード）＝スパイクは GO
`.venv-engine` 実コードで確認（file:line）:
- `GaussianNoiser`（`ltx_core/components/noisers.py:23-35`）は **denoise_mask を尊重**: `latent = noise*(mask*noise_scale) + latent_initial*(1-mask*noise_scale)`。**mask=0 の overlap は noise が足されず初期値を保持**。
- denoise loop（`ltx_pipelines/utils/samplers.py:60` ＋ `helpers.py:201-203 post_process_latent`）が **毎ステップ mask=0 を clean_latent へ再ピン**。
- `create_initial_state`（`ltx_core/tools.py:117`）で `clean_latent = initial_latent.clone()`＝注入 tail がブレンド標的。
- attention は full（`attention_mask=None`）＝**新フレーム（mask=1）は凍結 overlap に文脈参照する**。
- 制約: `tools.py:106-109` の shape assert＝**initial_latent は target 形状に完全一致必須**（tail を full フレーム数へ pad する）。custom mask は現状 all-1s 固定＝**注入口を monkeypatch で足す必要**（スライス1 流儀）。

### 1.5 WEB 判定＝GO-WITH-MODIFICATION（先行事例・公式ソース）
- メカニズムは公式 `LTXVExtendSampler`/looping と一致（overlap 種＋線形ブレンド＋AdaIN）。mask 規約（0=keep clean/1=generate）も公式 core-API ガイドと一致。
- **★修正: hard-freeze しない**。公式 `temporal_overlap_cond_strength` 既定=**0.5**。strength=1.0（mask=0 完全凍結）は frozen/flicker アーティファクトを招くと複数公式ソースが報告（ltx-desktop#41・warble blog）→ **overlap は soft（mask≈0.3–0.5 相当）でパラメータ化**。
- **8n+1 格子厳守**（overlap 長・seam index）＋ overlap→生成境界の **RoPE 位置を連続**に。
- ドリフト対策＝**AdaIN**（factor 0.1–0.3）＋ join の **LinearOverlapLatentTransition** 線形ブレンド。
- 主要 URL: LTX-2 core-API guide（HF spaces Lightricks/ltx-2）・looping_sampler.md・ComfyUI-LTXVideo latents.py・deepwiki 5.3-latent-operations・LTXVAddGuide docs・ltx-desktop#41。

---

## 2. スパイク設計（メカニクス検証・実装の第一歩）

**目的**: §1.4 の「mask=0 で overlap 凍結＋mask=1 で文脈参照生成」が **我々の実 engine 経路**で成立するかを決定的に確認（NO-GO なら厚版を再設計）。

- **形態**: engine/ を改変せず、**.venv-engine 内で engine パイプラインを直接インスタンス化する独立プローブ script**（可逆・main 不可触）。runtime monkeypatch で (a) stage1 に `initial_video_latent`（padded tail）を渡す・(b) custom denoise_mask（overlap=0/新=1）を注入・(c) stage1 tail latent を捕捉。
- **あえて hard-freeze（mask=0）**＝メカニズムの二値検証（soft 化は品質チューニングゆえ本実装フェーズで目視と共に導入）。
- **検証項目**: ①極小解像度で clip1 生成→stage1 tail 捕捉 ②clip2 を tail＋mask で stage1 seed→shape/crash なし完走 ③overlap フレーム ≈ carried tail（凍結が効く）＋新フレームは生成される ④VRAM 16GB fit ⑤**回帰＝extend 不使用の既定 T2V/I2V が byte-match 不変**（`outputs/qat_reclaim_verify_textonly/run_verify.py`・T2V `23844b4e…`／I2V `a511eda4…`）。
- **早期停止**: 異常・shape assert・overlap 不一致・byte-match 破れ を検知したら即報告＋判断を仰ぐ（走り切らない）。

---

## 3. フェーズ計画（v1＝スパイク→全段階実装）

1. **スパイク**（§2・GO/NO-GO 確定）
2. **engine 本実装**: stage1 seed 配線＋custom denoise_mask 注入（monkeypatch）＋stage1 tail 永続化（`torch.save`／worker プロトコル入出力）。
3. **連続性処理**: overlap strength パラメータ化（既定 0.5・soft）＋AdaIN（factor 0.1–0.3）＋線形クロスフェード（最終 latent・VAE decode 前・unpatchify [B,C,F,H,W]）。8n+1 格子＋RoPE 連続。
4. **API**: 新 `POST /api/v1/generate/chain`（global 基底プロンプト＋clip 毎 override・backend が逐次ループ＋tail スレッド＋concat→単一mp4）。**凍結単一 generate 契約は不変・旧クライアント互換（加算的）**。concat/mux は app 側 ffmpeg（既存音声 mux と同経路・app venv に torch 不要）。
5. **検証**: 回帰 byte-match＋新機能の目視（①2クリップ境界破綻なし ②3本以上でドリフト許容内 ③プロンプト伝播 ④VRAM fit＋gen 時間漸増を実本数で再計測）＋mock pytest 緑。

---

## 4. engine 触点（本実装フェーズ・スパイク確定後）
- `engine/pipeline/fast_video_pipeline.py` `_run_inference`（既存 monkeypatch 群に seed/mask/tail 捕捉を追加）
- `ltx_pipelines/utils/helpers.py` `denoise_audio_video`／`noise_video_state`（custom mask 注入口・wheel は不可触ゆえ engine 側 wrapper/monkeypatch）
- `ltx_pipelines/distilled.py`（stage1 の initial_video_latent 配線・tail 捕捉／wheel 不可触ゆえ engine から monkeypatch）
- `engine/worker.py`＋`engine/api_types.py`（generate op に `prev_clip_latent_path`/`overlap_frames`/`overlap_strength`/`adain_ref_latent_path` 等を追加・done に `carry_latent_path`）
- app 側: `api/`（新 chain エンドポイント）・`services/`（逐次オーケストレーション＋concat）
