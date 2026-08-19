# audio-to-video（A2V）— アップロード音声に合わせた動画生成 設計＋実装計画

- 作成: 2026-07-05（監督）。**ステータス: ✅完結 — S0-S3 完了（G0 GO／G1 全PASS／G2 mock+実機 PASS）＋G3 試聴=条件付き受容→main マージ・push 済（merge `d4a0cb2`・2026-07-05）**。リップシンクの弱さはシーン要因（動き・画角）＝モデル性質で確定（判定と運用指針=VERIFICATION_LOG §25.5）。本書は歴史記録＝以後の変更は加えない。
- 入口: [`A2V_ENTRY.md`](A2V_ENTRY.md)（次セッション導線）・機能リサーチ原典 [`FEATURE_RESEARCH_2026-07-04.md`](FEATURE_RESEARCH_2026-07-04.md) C節
- 直近の成功例（様式の手本）: [`V2V_CONTINUATION_DESIGN.md`](V2V_CONTINUATION_DESIGN.md)（§2.2 幾何・§3 ゲート表・スライス進捗ログ）
- 裏取り: 上流 pin `00dc53d` の `A2VidPipelineTwoStage` 読解＋我々のチェーン機構（V2V で main マージ済 merge `18296b2`）の再確認（2026-07-05・本書 §1・file:line）
- 鉄則: システム Python 不可触・凍結 API は加算的拡張のみ・wheel 不可触・実験前に仮説→裏取り・回帰 byte-match をゲートに使う・GPU 計測は dedicated+shared 両監視

> **2026-08-10 追記（長尺A2V＝台帳 §1-16）**: v1 の「クリップちょうど1個」制約は撤廃した。1本のアップロード音声が 1〜24 クリップの連結タイムライン全体を駆動する。音声ファイルを分割する必要はなく、`chain_math.audio_segment_windows` がステージ1の各セグメントへ「その区間が担当する音声窓」を割り当てる（隣り合う窓は、映像の連結と同じ K_a だけ重なる）。エンジン側では `_denoise_av_with_carry` に `audio_mask_value` を追加し、**音声窓だけをハード凍結（0.0）して映像の継ぎ目は従来どおり overlap_strength でブレンドする**ようにした。以下の本文は 2026-07-05 時点の設計記録であり、単クリップ前提の記述には該当箇所ごとに追記を入れてある。検証記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §56。

---

## 0. 一行サマリ

**ユーザーがアップロードした音声（波形）を VAE で latent 化し、AV 結合 latent のうち音声側を「全長ハード凍結」して動画側だけを denoise する。** LTX-2 の動画14B／音声5B 双方向クロスモーダル注意により、リップシンクは生成の中で成立する見込み。機構は V2V（`/generate/chain` の外部 latent 注入＋凍結 mask）を**流用**し、音声を「全長 freeze・mask_value=0.0」の極限で凍結するだけ＝denoise 機構の新開発は不要。新規性は「音声の入力口の配線」と「音声凍結窓の幾何」のみ。**最大の未知数**＝我々は蒸留経路のみで上流 Stage1 の modality_scale 摂動を持たないため、素の双方向クロスアテンションでリップシンクが成立するかは G0 スパイクで実証する。

---

## 1. リサーチ結論（コード裏取り済みの事実・file:line）

### 1.1 上流の手本 `A2VidPipelineTwoStage`（機構の一次参照）

- **本体**: `.uv_cache\git-v0\checkouts\821c13058d1842d3\00dc53d\packages\ltx-pipelines\src\ltx_pipelines\a2vid_two_stage.py:41`。`audio_path` を受け取り、まずこのファイルが公式 A2V 二段パイプラインの手本（V2V における `retake.py` に相当）。
- **音声経路**: `decode_audio_from_file`（:118）＝PyAV で audio stream を探索するのみ＝wav/mp3 等の**音声のみファイルも読める**。→ `vae_encode_audio`（波形→mel→VAE）→ **25 latent/秒**（`AudioLatentShape.from_duration`）→ **動画尺へ切り詰めるだけ**（:121・**パディング無し**）。
- **凍結の実体**: `denoise_video_only`（`ltx_pipelines/utils/helpers.py:476-524`）＝音声の `denoise_mask = torch.zeros_like(...)`・`noise_scale = 0.0`（:512）。毎ステップ `post_process_latent(denoised*mask + clean*(1-mask))` で clean latent へ置き戻す＝**完全凍結**。**Stage1・Stage2 とも凍結**。
- **⚠ 誤記訂正（A2V_ENTRY.md §1.1）**: 「`ltx_pipelines/utils/blocks.py` の `ModalitySpec(frozen=True)`」は**実在しない**（`blocks.py` 無し・`ModalitySpec` クラス無し）。**音声凍結の実体は上記 `denoise_video_only` の denoise_mask 全ゼロ方式**（helpers.py:476-524）。本書が正・A2V_ENTRY.md には訂正注記済み。
- **出力音声**: 元波形をそのまま mux（:246-248「preserve fidelity」コメント）。**vocoder は使わない**。
- **ガイダンス構造**: Stage1＝非蒸留フルステップ＋ `a2v_guidance_scale`（＝ `MultiModalGuider` の modality_scale＝クロスモーダル摂動ガイダンス。CLI ヘルプに「リップシンク品質に効く」と明記）。Stage2＝2x アップサンプル＋蒸留 LoRA＋固定 4 シグマ＋ガイダンス無し。
- **兄弟 `retake.py` との差**: retake は「切り詰め＋ゼロパディング両対応」かつ vocoder デコード出力。A2V は「**切り詰めのみ**」かつ「**原波形 mux**」。

### 1.2 我々のコードベースに部品が揃っている（V2V で配線済み・無改造で流用可）

- **音声デコード→VAE エンコード→latent**: `engine/pipeline/chain_pipeline.py` `_encode_source_heads`（:236-317・音声エンコードの実配線）＝ `decode_audio_from_file`→`ltx_core.model.audio_vae.encode_audio`→latent 切り出し。`ledger.audio_encoder()` のロードは実測 **45.7MB**（VRAM 上の心配なし）。encode 後は即解放する WDDM 定石を V2V で確立済み（`_encode_source_heads` :346-349 と同一呼出型）。
- **全長ハード凍結が無改造で成立**: `_denoise_av_with_carry`（同ファイル :139-213）は `freeze_ka=全長`・`mask_value=0.0` ＋ `initial_audio_latent` を渡すだけで**全長ハード凍結**になる（監督が直接確認済み・:193-197）。V2V の stage2 variant B（:593, :595）で mask=0.0 のハード凍結を**実績投入済み**＝上流 `denoise_video_only` と等価。
- **音声ジオメトリの単一情報源**: `chain_math.py`（`AUDIO_LATENTS_PER_SEC=25.0`・`a_frames_for_px(px, fps)` :67-69）。
- **音声出力/結合資産**: エンジンの `encode_video_output`（`common.py:65-80`）で映像＋音声を mux。V2V の audio_handle／`join_v2v`（`services/video_io.py`）経路は A2V では**通らない**（原波形をそのまま mux するため）。

### 1.3 単発 `/generate` は A2V 不可・chain 経路一択

- 単発 `/generate` は wheel 内 `DistilledPipeline` を直呼びし（`fast_video_pipeline.py:667`）**音声凍結フックが無い**。A2V は外部 latent 注入＋凍結 mask を持つ `/generate/chain` 経路でのみ実現できる。

### 1.4 最大リスク（G0 スパイクで実証する）

- 我々は**蒸留経路のみ**＝上流 Stage1 の modality_scale 摂動（クロスモーダル・ガイダンス）が**無い**。素の双方向クロスアテンションだけでリップシンクが成立するかは**未知数**。→ G0 スパイク④「同一 seed・異なる 2 音声を与えて動画が変化するか」で音声の影響力を実証する。弱ければ NO-GO でユーザー相談（§4 Q3）。

---

## 2. 設計

### 2.1 API 表層（凍結契約の加算的拡張・V2V 前例に従う）

- **`POST /upload/audio` を新設**（`services/audio_upload_store.py` を新設＝`video_upload_store.py` のクローン）:
  - 保存先 `uploads/audios/{id}/input{ext}`・許可拡張子 `.wav/.mp3/.m4a/.aac/.flac/.ogg`・`max_audio_size_mb`（config デフォルト付き）。無検査保存（コーデック検査は preflight で ffprobe が担当）。
  - `api/uploads.py` に `POST /upload/audio` を追加（`POST /upload/video` と同型）。
- **`POST /generate/chain` に optional `source_audio` を追加**（省略時 byte 同一＝V2V `source_video` 前例）:

```jsonc
{
  // 既存フィールドは全て不変…
  "source_audio": {                 // ★新規 optional・既定 null
    "audio_id": "<POST /upload/audio の id>"
  },
  "clips": [ { "prompt": "...", "num_frames": 121 } ]   // source_audio 有り時は 1 クリップから可
}
```

> 2026-08-10 追記: `clips` は `source_audio` 有りでも **1〜24 個**（上限は通常のチェーンと同じ）。音声は常に1本で、タイムライン全体を覆う。

- `api/models.py` に `SourceAudioSpec{audio_id}` ＋ `GenerateChainRequest.source_audio` を追加（`SourceVideoSpec` :191／:282 前例）。
- **新エンドポイントは作らない**（Q1）。省略時は byte 同一。
- バリデータ（すべて 422 事前検出・加算的）:
  - `source_audio` 有り時のみ `clips` の min を 1 に緩和（無し時は従来通り＝旧クライアント不変）。**2026-08-10 追記: 上限側の「ちょうど1個」制約は撤廃済み（1〜24 個）。**
  - **`source_audio` と `source_video` は排他**（422・V2V と A2V を同時に混ぜない・v1 スコープ）。
  - `source_audio` と `conditioning_images` の**併用は許可**（Q2）。
  - エラーコード追加: `SOURCE_AUDIO_NOT_FOUND`（404・`audio_id` 不在）／`SOURCE_AUDIO_TOO_SHORT`（422・音声がタイムライン尺に足りない＝**動画尺が主・切り詰めのみ**の帰結）。
- v1 では**トリミング指定（`audio_start_time`／`audio_max_duration`）は非露出**（Q2）。

### 2.2 幾何（chain_math を単一情報源として拡張・構造変更なし）

- 論理タイムライン＝動画尺が主。音声 latent は動画尺に合わせて**切り詰め**、足りなければ 422（上流 a2vid :121 準拠・パディング無し）。
- `chain_math.py` に**純関数を加算**（app／engine の幾何一致の単一情報源）:
  - `audio_latents_required(clip_frames, fps)`＝当該クリップに必要な音声 latent 数（`a_frames_for_px` :67-69 を流用）。
  - `audio_segment_windows(layout)`＝凍結すべき音声窓（v1 は 1 クリップなので全長 1 窓）。
- `compute_chain_layout` の**構造は変更しない**（音声窓の算出を純関数として足すのみ）。

> **2026-08-10 追記（長尺A2V の本線）**: `audio_segment_windows` はもともとクリップ数に依存しない書き方になっており、これが長尺 A2V の中核である。セグメント `i` の窓は `(sum(seg_audio[:i]) - sum(ka_list[:i]), seg_audio[i])`＝グローバル音声タイムライン上の開始位置と長さで、①先頭窓は 0 から始まり ②末尾窓は `a_total` ちょうどで終わり ③隣接窓は `ka_list[i]`（映像の連結に使うのと同じ音声のり代）だけ重なる。つまり窓の集合はアップロード音声 latent を過不足なくタイル張りする。数値例（[121,121,121] @24fps・K_v=2）は `tests/test_a2v_chain.py::test_a2v_segment_windows_tile_the_global_timeline` に書いてある。
>
> あわせて `audio_latents_required` は `compute_chain_layout` を呼ぶのをやめ、`a_total`（`seg_latent`→`f_total`→`total_px`→`a_frames_for_px`）を直接計算するようにした。`a_total` はステージ2の窓（`v_tile`／`v_adv`）に依存しないので、長さの事前検査がステージ2タイルの検算を巻き込む理由がない。巻き込んでいたせいで、**リクエスト側の窓なら通る構成が事前検査で例外になり 500 になる穴**（実測: 321フレーム1クリップ・23.976fps）が単クリップ A2V にも残っていた。これも同時に塞がった。

### 2.3 エンジン（chain_pipeline の加算的拡張）

- `chain_pipeline.py` に `AudioSourceSpec{path}` を追加（`SourceSpec` :61 とは**別型**）＋ `run_chain(..., audio_source=None)` を追加。
- **音声エンコードは一度だけ**: `_encode_source_heads`（:346-349）と同一呼出型で `decode_audio_from_file`→`encode_audio`→latent。encode 後は即解放（WDDM 定石・audio_encoder 実測 45.7MB）。
- **Stage1／Stage2 両方**で該当窓を `freeze_ka=全長`・`mask_value=0.0` で凍結（§1.2＝上流 `denoise_video_only` の Stage1/Stage2 両凍結と等価）。
  - **2026-08-10 追記**: ステージ1では `mask_value=0.0` ではなく `audio_mask_value=0.0` を渡す形に変えた。`_denoise_av_with_carry` は凍結の強さを「映像の先頭／映像の末尾／音声の先頭／音声の末尾」の4本に分けて解決するようになっており（純関数 `chain_math.freeze_mask_values`）、`audio_mask_value` 省略時は4本とも `mask_value` ＝従来と1ビットも変わらない。複数クリップでは2本目以降の映像先頭が `freeze_kv = K_v > 0` になるため、単一の `mask_value=0.0` を使い続けると**映像の継ぎ目まで完全凍結され、overlap_strength が無言で無効化される**。ステージ2は元から全経路 `mask_value=0.0`（ハード凍結）なので変更していない。
- **出力音声**: vocoder をスキップし、**原波形を** `n_samples = round(total_px / fps * sr)` **に切って** `encode_video_output`（`common.py:65-80`）へ渡して mux（Q8・上流 :246-248 準拠）。**トリム／audio_handle 経路は通らない**。
- worker プロトコル: `generate_chain` op に optional `audio_source` ブロックを追加（V2V 前例と同型）。

### 2.4 app 側（services / metadata / mock）

- `services/pipeline_manager.py` に `preflight_source_audio` を追加＝ **ffprobe で音声尺 ≥ タイムライン尺**を事前検査（V2V の ffprobe preflight :202 前例）。不足時は `SOURCE_AUDIO_TOO_SHORT`（422）。
- `services/ltx_runner.py` `_RealBackend` payload に `audio_source` ブロックを追加（V2V payload :1051-1057 前例）。`_MockBackend`（:367）に A2V 対応（GPU 無し pytest で API 契約を固定）。
- `metadata.json` に**加算的 `a2v` ブロック**（`source_audio_id`・音声尺・切り詰め有無等）。既存キーは不変。

### 2.5 引き継ぐ制約・スコープ外（v1）

- ÷64・8n+1・総フレーム上限・16GB 天井・SDPA 一択・torch 2.9.1+cu128 固定・2プロセス/2venv・巨大ディスク要求禁止＝すべて不変。
- **蒸留パイプライン＝CFG/negative 不可**（worker 未配線・露出禁止・継続）。
- **スコープ外（v1）の4点**（A2V＋V2V同時指定〔422で排他〕／vocoder音声の返却／`audio_start_time`・`audio_max_duration`の露出／`modality_scale`の露出）は、2026-07-27の整理で[`PENDING_TASKS.md`](PENDING_TASKS.md) §4-9へ移設した。**以後の管理は同書で行う。** いずれも凍結APIへの加算的拡張として後から足せる形は保たれている。
  - **2026-08-10 更新**: 5点目だった「複数クリップA2Vの音声窓割り」は台帳 §1-16（長尺A2V）として**実装済み**のため、このリストから外した。予告どおり凍結APIへの加算的拡張のみで足りている（新しいリクエストフィールドはゼロ・削除したのはバリデータのガード1本）。
- 同じくv1でスコープ外としていた**GUI露出**（音声スムージングON/OFFチェックボックス要件＝[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §24.7将来項目①）は、その後実装済み。

---

## 3. 検証計画（ゲート）

| ゲート | 内容 | 合否基準 |
|---|---|---|
| **G0 スパイク（GPU・最優先）** | 独立プローブ script（main 不可触・`outputs/a2v_spike/`）: TTS wav → latent 化 → n=1 chain で**全長凍結** → 生成 → 原波形 mux | ① shape 整合／crash 無し ② VRAM ≤16GB ③ 出力音声 == 入力音声 ④ **同一 seed・異なる 2 音声 → 動画が変化する**。④ が弱ければ **NO-GO → ユーザー相談** |
| G1 回帰 | T2V `23844b4e…`／I2V `a511eda4…` byte-match・既存チェーン／V2V 同一シード byte 一致・pytest **191 passed / 1 skipped** 維持 | バイト完全一致・緑 |
| G2 新経路スモーク | mock ＋実機で `source_audio` 付き chain 完走・422／404／排他（source_audio×source_video）の全網羅 | 全 PASS |
| G3 試聴ゲート（ユーザー） | ①セリフ音声（Windows 標準 TTS `System.Speech` で生成した wav）＋②音楽のみ（過去 LTX 生成物の音声トラック流用）の 2 ケース・**720p 級（1280×768）**・**映画トレイラー風（賑やかな町＋話者）** | **リップシンク・音声保全をユーザーが受容** |

- **early-stop**: shape assert・VRAM 超過・音声不保持を検知したら走り切らず即報告→判断を仰ぐ。
- GPU 計測: dedicated+shared 両監視・**WDDM ページ降格に注意**（一過性の超過で降格したページは戻らず後続が共有メモリ実行になる・V2V で実証）。一次ソース＝`logs/ltx_worker.log` の `peak_vram_mb`。

---

## 実装スライス（V2V 前例に従う）— 進捗ログ（S0 以降を追記）

1. **S0 スパイク**: ✅**GO（2026-07-05・G0 4基準すべて PASS）**。独立プローブ `outputs/a2v_spike/probe_a2v.py`（n=1・704×448・121f・seed固定・main/wheel 不可触）で実証。①shape 整合・crash 無し（stage1 音声凍結の max drift=**0.000e+00**＝ハード凍結が厳密に成立）②VRAM: torch ピーク **8858.7MB**／nvidia-smi dedicated ピーク **~11.2GB**・**共有溢れ無し** ③出力音声==入力 wav（Pearson r=1.0000/0.9999・差は AAC 損失のみ）④**同一 seed・異なる2音声→全フレーム平均絶対差 3.94%（per-frame MAD 9.22–12.70・全フレーム非ゼロ）**、目視で口の形・頭部姿勢が明確に相違＝**蒸留経路でも凍結音声がクロスアテンション経由で動画を駆動**（§1.4 の最大リスク解消・fallback 不要）。1本 ~122秒。詳細＝`outputs/a2v_spike/SPIKE_REPORT.md`（outputs は git 管理外のため数値は本書へ転記）。**技術知見: 音声 VAE エンコーダは stereo(2ch)入力必須（conv_in=[128,2,3,3]）・mux の `_write_audio` も stereo 前提** → S1 では mono 入力を stereo へ複製する正規化が必要。
2. **S1 コア（エンジン側）**: ✅**完了（2026-07-05・commit `b59d0fb`・G1 全 PASS）**。`chain_pipeline.py` に `AudioSourceSpec`／`run_chain(audio_source=)`＝音声を一度だけエンコード（mono→stereo 複製正規化・encoder 即解放）→Stage1 各セグメント＋Stage2 各タイルの音声窓を `mask_value=0.0` でハード凍結→vocoder スキップ・原波形を動画尺へ切って**ストリーム mux**（lazy デコードチャンクをそのまま `encode_video_output` へ＝WDDM 定石維持。監督レビューで実体化を差し戻し修正済み）。`chain_math.py` に `audio_latents_required`／`audio_segment_windows`（app/engine 幾何の単一情報源・ユニットテスト 8 本）。`worker.py` が payload `audio_source:{path}` を受理し `done.chain.a2v` を返す。**既知の制限（v1 設計どおり）**: n>1 の A2V は `_denoise_av_with_carry` の mask_value が映像 carry と共有のため未対応（API 層で n=1 を強制・コード内に文書化）。**〔2026-08-10 解消済み — 台帳 §1-16。`audio_mask_value` を足して映像と音声の凍結強度を分離し、API のガードを撤廃した。冒頭の追記を参照。〕** **G1 回帰（GPU 実生成・2026-07-05）**: T2V `23844b4e…6bb7bf`／I2V `a511eda4…c217`／チェーン no-source `f706057a…0ea1` すべて基準 SHA-256 と**バイト完全一致**・pytest **199 passed/1 skipped**（基準 191+新規 8）・peak_vram_mb=8440（§22/§24 記録と同値）・OOM/spill なし。
3. **S2 API/テスト**: ✅**完了（2026-07-05・commit `e37a9ef`・G2 mock 側 PASS）**。`services/audio_upload_store.py`（video store クローン・`uploads/audios/{id}`）・`POST /upload/audio`＋`UploadAudioResponse`・`SourceAudioSpec{audio_id}`＋バリデータ（**A2V×V2V 排他 422・clips ちょうど1つ 422・source_audio 有り時のみ床緩和**・conditioning_images 併用可）・`SOURCE_AUDIO_NOT_FOUND(404)`／`SOURCE_AUDIO_TOO_SHORT(422)`・`preflight_source_audio`（ffprobe 尺 vs `chain_math.audio_latents_required`=単一情報源・ffprobe 不能時はエンジンの a2v_avail ガードが最終防壁）・runner payload `audio_source:{path}`・mock backend が実エンジンと同一キー構成の `chain.a2v` を合成・metadata に加算的 `a2v` ブロック（+`source_audio_id` 来歴）。**pytest 212 passed/1 skipped**（199+新規13）。**G2 mock スモーク PASS**（実 uvicorn: upload 200→chain 202→completed→a2v メタ正・2クリップ422／不在404／A2V+V2V 422 全確認）。G2 実機側は S3 で消化。
4. **S3 実機 e2e ＋ G3 素材**: ✅**完了（2026-07-05・commit `8cc793e`・G2 実機側 PASS）**。REST 経由実機（RTX 4070 Ti SUPER・`main.py --port 18620`・config 不変）で source_audio 付き chain 完走: TTS 11.19s wav→upload→chain（704×448/121f）→completed **117.1s**・peak_vram_mb=**8440**・nvidia-smi dedicated peak 14371MB・shared 平坦（spill なし）・音声 Pearson r=1.0000/RMS 比 0.9989・a2v メタ正。negative 実機**4件**（A2V+V2V 422／不在 404／2クリップ 422／短音声 422 `SOURCE_AUDIO_TOO_SHORT`）全緑。**G3 候補生成（720p 級＝1280×768 生成→crop 1280×720・121f・映画トレイラー風）**: ケースA（セリフ・男声TTSナレーション）=job `0bab3830`・175.14s・worker peak 9519MB・nvidia-smi 14885MB・音声 r=0.9999→`outputs/a2v_g3/caseA_speech_1280x720_121f.mp4`／ケースB（音楽のみ・過去 LTX 生成物の音声流用）=job `93fffd3b`・185.84s・worker peak 9525MB・nvidia-smi 14715MB・音声 r=0.9999→`outputs/a2v_g3/caseB_music_1280x720_121f.mp4`。全 run で OOM/spill なし。**残 OPEN は G3 試聴（ユーザー・客観PASS≠目視ゲート）**。数値正本＝VERIFICATION_LOG §25／`outputs/a2v_e2e/E2E_REPORT.md`。

- 各スライスでコミット。**push／マージはユーザー承認待ち**。

---

## 4. 設計論点のユーザー決定（✅2026-07-05 合意済み）

- **Q1 API 形**: ✅ **`POST /upload/audio` 新設 ＋ `/generate/chain` に optional `source_audio{audio_id}`**（新生成エンドポイントは作らない・省略時 byte 同一）。
- **Q2 v1 スコープ**: ✅ **クリップ 1 つのみ／A2V×V2V 排他（422）／トリミング指定は非露出／音声不足＝422（動画尺が主・切り詰めのみ）／`conditioning_images` 併用は許可**。
- **Q3 G0 で音声の影響が弱い場合**: ✅ **G0 結果を見てユーザーと相談**（fallback a＝modality_scale 差し込みは事前承認しない／fallback b＝上流パイプライン直呼びは制約変更のユーザー署名が前提）。
- **Q4 G3 検証素材**: ✅ **Windows 標準 TTS（`System.Speech`）でセリフ wav 生成 ＋ 音楽ケースは過去 LTX 生成物の音声トラック流用**。
- **Q8 出力音声**: ✅ **元波形 mux（上流準拠・vocoder 不使用）**。
