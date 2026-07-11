# video-to-video 継続（アップロード動画の「続き」生成）設計 — リサーチ結論＋実装計画ドラフト

> **（歴史記録・2026-07-06 注記）✅実装完結・mainマージ済（merge `18296b2`・2026-07-05）。** 実測/ゲート/コミット列の正本＝VERIFICATION_LOG §24（音声知見=§24.7）。本書は設計根拠として温存。

- 作成: 2026-07-04（監督）。**ステータス: ✅完結（実装・mainマージ済 2026-07-05）**（原文: ✅ユーザー合意済み 2026-07-04 → 実装フェーズへ）
- 入口: [`FEATURE_RESEARCH_2026-07-04.md`](FEATURE_RESEARCH_2026-07-04.md) B節（ユーザー決定: 本セッションの本題）
- 裏取り: コード読解サブエージェント×3＋上流(vendor/pin)読解×1＋Web先行事例×1（2026-07-04・本書 §1）
- 鉄則: システム Python 不可触・凍結 API は加算的拡張のみ・実験前に仮説→裏取り・回帰 byte-match をゲートに使う

---

## 0. 一行サマリ

**アップロード動画の末尾（context・既定 73 ピクセルフレーム ≈ 3秒）を VAE エンコードして凍結 latent 先頭として注入し、既存のクリップ連結（masked AV-latent chain）機構でその続きを denoise する。** 上流 pin `00dc53d` の `RetakePipeline` が「実動画→initial latent＋mask 部分 denoise」の実証例であり、機構部品はすべて確認済み。新規性は「入力口の配線」のみで、denoise 機構の新開発は不要。

---

## 1. リサーチ結論（コード裏取り済みの事実）

### 1.1 既存チェーン機構は「外部 latent 注入」に構造的に開いている

- `engine/pipeline/chain_pipeline.py` の `_denoise_av_with_carry`（:122-196）は `initial_video_latent` / `initial_audio_latent` を **plain `torch.Tensor | None`** で受け、shape assert 以外に出所を問わない。clip i>0 は「前セグメント stage1 latent の末尾 kv フレームを `init_v` の先頭に置き、freeze mask（`denoise_mask` を `1-overlap_strength` に）で毎ステップ re-pin」する（:324-331, :163-168）。
- **ハードコードされているのは orchestration だけ**: `run_chain`（:223-236）・worker `generate_chain` op（`engine/worker.py:271-297`）・`services/ltx_runner.py:950-961` のペイロード・`api/models.py` の `GenerateChainRequest` のどこにも「外部動画/latent を渡す口」が無い。＝**足すのは入力口の配線のみ**。

### 1.2 動画→latent の VAE エンコードは実績あり（IC-LoRA 参照経路）

- `engine/pipeline/fast_video_pipeline.py:559-567`: `load_video_conditioning`（PyAV デコード→resize+center-crop→[-1,1] 正規化）→ `video_encoder(video)`＝**実 VAE encode が本番稼働中**（IC-LoRA 参照条件付け・stage1 のみ）。
- pin `00dc53d` の `retake.py` に `_encode_video_for_retake` / `_encode_audio_for_retake`（:55-101）が存在＝**「実動画ファイル→initial_video_latent／音声波形→initial_audio_latent」の上流実証例**。音声側の部品（`decode_audio_from_file`・`encode_audio`・`waveform_to_mel`）も pin に確認済み。
- `ledger.audio_encoder()` ビルダーは存在するが現在未使用（`model_ledger.py:247-253`）＝音声エンコーダの**初ロード**になる（VRAM 実測が必要）。
- **注意（pin に無いもの）**: 新しい vendor ツリーにある `video_latent_from_file`／`_conform_latent_length`／`VideoConditionByMask` は **pin に存在しない**。凍結ヘッドの注入は wheel ヘルパーでなく**既存チェーンの freeze mask 方式**（1.1）で行う（wheel 不可触・実績経路）。

### 1.3 先行事例（Web）の要点

- 公式ホスト API の Extend: **context（源動画の参照尺）最小 73 フレーム（≈3秒@24fps）・最大 20 秒**、context＋新規尺 ≤ 505 フレーム。**音声は「源に音声があれば新規部分の音声を（源に整合するよう）再生成」**。
- ComfyUI コミュニティの extend は「末尾 N フレームを VAE エンコードして凍結ガイドにする inpainting」＝本設計と同型。**出力は「overlap＋新規」で、overlap 部分は連結前に削除する**運用。**音声継続は community では未解決**（我々のチェーン機構は音声 freeze を既に持つ＝優位点）。
- 既知の落とし穴: 継ぎ目後の**明度/彩度ドリフト**・暗背景での半透明オーバーレイ（コミュニティ報告・未解決）→ スパイクの検証項目に含める。h264 圧縮アーティファクトの latent 混入は「もっともらしいが未確認」→ スパイクで実写素材を使って観察。

### 1.4 現状の欠落（実装で埋めるもの）

- アップロード動画の**音声トラックはどこでも読まれていない**（`video_upload_store` は無検査保存・IC-LoRA 経路は video stream のみ）。
- **fps 整合はどこにも無い**（インストール済み `decode_video_from_file` は生デコード順で `frame_cap` 切り詰めのみ・タイムスタンプ無視）。
- アップロード時の検証は拡張子・非空・サイズ上限のみ（コーデック/解像度/尺/fps は未検証）。

---

## 2. 設計

### 2.1 API 表層（凍結契約の加算的拡張・チェーン前例に従う）

**`POST /generate/chain` に optional フィールドを追加**（省略時 byte 同一＝IC-LoRA Phase B 前例）:

```jsonc
{
  // 既存フィールドは全て不変…
  "source_video": {              // ★新規 optional・既定 null
    "video_id": "<POST /upload/video の id>",   // 既存アップロード機構を再利用
    "context_frames": 73         // 源末尾の参照尺(ピクセルフレーム・8n+1)・既定73
  },
  "clips": [ { "prompt": "...", "num_frames": 121 } ]   // source_video 有り時は 1 クリップから可
}
```

- **新エンドポイントは作らない**。継続は意味的に「セグメント -1 が外部動画であるチェーン」そのものであり、多クリップ継続（源→続き1→続き2…）が追加コストゼロで手に入る。
- バリデータ（すべて 422 事前検出・加算的）:
  - `source_video` 有り時のみ `clips` の min を 1 に緩和（無し時は従来通り min 2＝旧クライアント不変）。
  - `source_video` と `clips[0].conditioning_images` は排他（先頭 latent は源が占有するため）。
  - `context_frames` は 8n+1・下限 25（≈1秒）・**上限 145（実装で確定・当初ドラフトの「481」は誤り）**。理由＝凍結ヘッド n_ctx_v は stage2 の時間タイル（`STAGE2_V_TILE`=22 latent）内に収まる必要があり（variant B のハード凍結は tile0 にしか掛からない・レビュー Finding 1）、理論安全上限は 169px、運用上限は保守的に 145px（`limits.v2v_context_frames_max`・chain_math 側にも不変条件 raise とガードテストあり）。源動画の実フレーム数 ≥ context_frames（app 側 ffprobe で事前検査・不足時 422）。
  - `source_video.video_id` 不在は 404（`reference_video_not_found` 前例に従い新エラーコード `SOURCE_VIDEO_NOT_FOUND` を追加）。
  - IC-LoRA（`loras`）との併用は v1 スコープ外＝当面 422（将来解禁の余地は残す）。
    - **→ 2026-07-11 更新（NEXT_SESSION_HANDOFF 冒頭ブロック／VERIFICATION_LOG §32）**: その後 `GenerateChainRequest.loras` が加算され、`/generate/chain` 全体で LoRA を扱えるようになった（当初「`loras` フィールドが無いので併用 422 は構造的に不要」＝下記 S2 注記の状況も、フィールド追加で変化した）。現在の拒否対象は「参照動画を要する control 系 IC-LoRA」に限られ（`LORA_CONTROL_UNSUPPORTED_IN_CHAIN`）、V2V は `source_video`（参照動画ではない）で成立するため、**画風・キャラクター系のスタイル LoRA は V2V チェーンと併用可能**。本節の「当面 422」は起票当時の設計判断の記録として残す。
- `GET /config` の `limits` に `v2v_context_frames_default` / `v2v_context_frames_max` 等を追加（`spill_free_frames` 前例＝config.py にデフォルト付きで足すだけで自動露出）。

### 2.2 幾何（chain_math を単一情報源として拡張）

- 論理タイムライン＝ **[凍結 context 部] + [新規生成部]**。clip0 の stage1 latent 先頭 `n_ctx = (context_frames-1)/8 + 1` 個を源動画由来の凍結 latent が占める。
- **kv=3 の従来 carry ではなく context 全体（既定 10 latent）を凍結ヘッドにする**。理由: 末尾 kv 個だけを切り出して timeline 先頭に置くと、causal VAE の「先頭 latent＝1 ピクセルフレーム anchor」規約と源 latent（steady-state・8 フレーム分）の意味がずれる。context 丸ごとなら「エンコード時の先頭 latent が timeline 先頭に座る」ため規約が一致し、ComfyUI extend の実運用（overlap＋新規を生成→overlap を刈る）とも同型。
- 凍結強度は既存 `overlap_strength`（既定 0.5・soft）を流用。クリップ間 join は従来の `overlap_frames`/`ka_list` のまま不変。
- 音声: 源の音声トラックを `decode_audio_from_file`→`encode_audio` で latent 化し、context 相当の先頭 `n_ctx_a = round(context_frames/fps*25)` 個を凍結ヘッドに（映像と対称・チェーンの音声 freeze 実績経路）。**源に音声が無ければ freeze_ka=0 で自由生成**。
- 出力トリム: decode 後のテンソルを engine 内でスライスし（フレーム正確・再エンコード無し）、**context 相当のピクセルフレームと音声サンプルを刈ってから mux**。＝**出力 mp4 は「新規部分のみ」**。metadata に境界情報を記録。

### 2.3 エンジン（chain_pipeline の加算的拡張）

1. `run_chain(..., source: SourceSpec | None)` を追加。`SourceSpec = {path, context_frames}`。
2. 源のエンコード（stage1 用）: `load_video_conditioning(path, h/2, w/2, frame_cap=context_frames)` → `ledger.video_encoder()` → 先頭 `n_ctx` latent。音声も同様に latent 化。エンコーダは使用後即解放（`del`＝retake.py 前例）。
3. clip0 の stage1 を「i>0 と同じ carry 分岐」で実行: `init_v[:, :, :n_ctx] = 源latent`・`freeze_kv=n_ctx`・mask=`1-overlap_strength`。
4. stage2: 源ヘッド区間の扱いは**スパイクで A/B**（§3 S2）: (a) 現行どおり tile join のみ freeze、(b) 源 context をフル解像度でも別途エンコードして tile0 先頭を hard-freeze（mv=0・tile join と同じ型）。junction 忠実度が高い方を採用。
5. decode→トリム→mux（1回だけ decode の現行構造は不変）。
6. worker プロトコル: `generate_chain` op に optional `source` ブロックを追加。`done` イベントの `chain` dict に `v2v` サブ dict（context_frames・trimmed_px 等）を追加。

### 2.4 app 側（services / metadata / mock）

- `services/pipeline_manager.py`: `source_video` 有り時に `video_upload_store.path_for()` で解決（IC-LoRA と同じ 404 パターン）→ **ffprobe で fps/フレーム数を事前検査**。
- **fps 整合（新規・要ユーザー決定 §4-Q3）**: 源 fps ≠ request `frame_rate` の場合、app 側 ffmpeg で末尾 context 区間だけを `frame_rate` に**リサンプルした一時 mp4** を作って engine に渡す（推奨）。または 422。
- metadata.json: 加算的 `v2v` ブロック（`source_video_id`・`context_frames`・`source_fps`・`resampled`・境界フレーム番号）。既存キーは不変。
- mock backend: `_MockBackend` に同型の source 対応（chain_math の幾何で合成出力）→ GPU 無し pytest で API 契約を固定。

### 2.5 引き継ぐ制約・スコープ外

- ÷64・8n+1・総フレーム上限（`MAX_CHAIN_TOTAL_PIXEL_FRAMES=3848`）・16GB 天井・SDPA・torch pin はすべて不変。÷128 は不要（IC-LoRA 固有の制約であり V2V には掛からない）。
- 解像度不一致の源は既存 `resize_and_center_crop` で吸収（IC-LoRA 前例＝黙って合わせる）。
- スコープ外: Retake（部分再生成）・audio-to-video（次の機能・ただし本実装の `encode_audio` 配線が下地になる）・源動画との server-side 結合 mp4（v1 はクライアント側で並べる想定・将来 `concat_mp4s` で追加可能）。

---

## 3. 検証計画（ゲート）

| ゲート | 内容 | 合否基準 |
|---|---|---|
| **G0 スパイク（GPU・最優先）** | 独立プローブ script（main 不可触）: 実写 mp4（音声付き）の末尾 73f を encode→凍結ヘッド注入→続き生成→源との junction を境界ハーネス（`verify_boundaries.py`・較正済み）で計測＋色ドリフト観察 | junction continuous・shape/crash 無し・VRAM ≤16GB（audio_encoder 初ロード分込み）。S2=stage2 ヘッド freeze の A/B もここで |
| G1 回帰 | T2V `23844b4e…`／I2V `a511eda4…`／既存チェーンの byte-match 不変・pytest 全緑 | バイト完全一致 |
| G2 新経路スモーク | mock＋実機で source 付きチェーン完走・422/404 系全網羅 | 全 PASS |
| G3 目視ゲート（ユーザー） | **720p 級・映画トレイラー風・賑やかな町＋セリフ**の実写または高品質生成素材を源にした継続を、源→継続と並べて再生 | 継ぎ目・色ドリフト・音声連続をユーザーが受容 |

- スパイク素材: 高品質な既存生成物（`outputs/visual_review/` の 720p 級）＝「きれいな源」と、実写・h264 圧縮素材＝「汚い源」の両方で junction を比較（§1.3 の未確認リスクの検証）。
- 失敗時の early-stop: shape assert・junction 不連続・VRAM 超過を検知したら走り切らず報告→判断を仰ぐ。

## 実装スライス（チェーン実装の前例に従う）— 進捗（2026-07-04）

1. **S0**: ✅**GO**（`outputs/v2v_spike/SPIKE_REPORT.md`）。**variant B（stage2 tile0 ヘッドのフル解像度ハード凍結）が必須と確定**＝variant A は継ぎ目で色調が跳ね G0 不合格（MAD 4.00×）、B は 1.00×。h264 強圧縮源でも継ぎ目連続（圧縮アーティファクト混入リスクは観測されず）。音声はノイズフロア差の微小クリック→30ms フェードインガードで対処（最終判断は G3 試聴）。audio_encoder 初ロード 45.7MB。
2. **S1**: ✅完了（commit `e7d497c`→`2830ede`→`5482225`）。tiled_encode 採用で VRAM はスパイク比 -1066MB・**720p 完走 15,817MB**。G1a pytest 147 緑・G1b T2V/I2V byte-match 完全一致・no-source チェーン回帰一致。
3. **S2**: ✅完了（commit `33fae6d`→`73dc20f`→`44facdd`→`a89963c`）。pytest 162 緑・mock 実サーバースモーク PASS（30fps 源→24fps 自動リサンプル含む）。注記: チェーン要求に `loras` フィールドは元々存在せず「IC-LoRA 併用 422」は構造的に不要だった。
4. **レビュー反映**: ✅完了（commit `59c8ee4`・Opus レビュー Finding 1 MAJOR=タイル適合不変条件を chain_math に early-raise＋設定ガードテスト、Finding 2=音声アンダーフリーズ警告＋`audio_head_frozen` キー、nits）。pytest 165 緑・数値経路不変。
5. **S3**: ✅完了（`outputs/v2v_e2e/E2E_REPORT.md`）。REST 経由実機 e2e＝720p PASS（G3候補生成済み）・等長多クリップ PASS・実 fps リサンプル PASS・422/404 全PASS。**副産物＝既存チェーンの潜伏バグ発掘**（クリップ長不揃いでクラッシュ・V2V 以前から存在）→ 修正 `e8557cb`（等長経路は修正前後バイト一致で無害証明・不等長の実機再現→修正確認・pytest 168 緑）。**G2 完全消化。残る OPEN は G3（ユーザー目視）のみ**＝素材と判定観点は VERIFICATION_LOG §24.4。

---

## 4. 設計論点のユーザー決定（✅2026-07-04 合意済み）

- **Q1 API 形**: ✅ **`/generate/chain` への optional `source_video` 追加**（新エンドポイントは作らない）。
- **Q2 出力範囲**: ✅ **新規部分のみ**（context はサーバー側でフレーム正確に刈る・境界情報は metadata）。
- **Q3 fps 不一致**: ✅ **app 側で自動リサンプル**（末尾 context 区間のみ・metadata に変換有無を記録）。
- **Q4 音声**: ✅ **源の音声末尾も latent 化して凍結継続**（源に音声が無ければ自由生成へ自動フォールバック）。
