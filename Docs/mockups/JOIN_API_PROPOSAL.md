# V2V 結合エンドポイント提案（`POST /jobs/{job_id}/join`）— 実装前ドラフト

> **✅ 実装済み（2026-07-05・merge `f999374`）。本書は実装前ドラフトで歴史記録。** `POST /jobs/{id}/join`＋`GET /jobs/{id}/joined` は `services/join_manager.py` に実装され G3 再ゲートでクローズ済み。**実装との差分**: (1) `handle_crossfade_ms` 既定は本書の 150→**300ms に変更済み**（G3 フィードバック F5・GUI ドロップダウン 150/300/500 追加）。(2) OFF＝ハードカット（§4・R4）は API のみ実装し **GUI には露出しない**（ON／結合しないの 2 択で足りるとの判断）。(3) §3・R3 の解像度/fps 不一致正規化は **setsar 必須の発見**を含めて実装。正本＝VERIFICATION_LOG §26（§26.4 に無音音声 loudnorm=-inf の次セッション持ち越しあり）。

- 作成: 2026-07-05（子A・S0-3）。**ステータス: ✅実装済み（当初＝提案・親の承認待ち）。**
- 目的: V2V 出力（新規部分のみの `output.mp4`）を元動画とサーバー側で結合し、音声の継ぎ目をクロスフェードで滑らかにした完成版 `joined.mp4` を得る。凍結 API への**加算的**新設（既存エンドポイントは byte 不変）。
- ユーザー決定: 「サーバー側の結合エンドポイントを凍結 API に加算的に新設する」。VERIFICATION_LOG §24.7 将来項目①（音声スムージング UI）の受け皿。
- 前提の裏取り（すべて file:line）: 部品はすべて実在し、無改造で組み合わせられる。以下 §1 で確認。

---

## 1. 使う素材はすべて既に揃っている（裏取り済み）

| 素材 | 実体 | 場所（file:line） |
|---|---|---|
| 元動画 | uploads ストアの実ファイル | `services/video_upload_store.py:79` `path_for(video_id)` |
| ジョブ出力 mp4（新規部分のみ） | `outputs/{job_id}/output.mp4` | `api/jobs.py:38`・`services/pipeline_manager.py:486` |
| 音声ハンドル・サイドカー | `outputs/{job_id}/output_audio_handle.wav`（トリム前の全長タイムライン音声） | `engine/pipeline/chain_pipeline.py:835,839`・mock は `services/ltx_runner.py:453` |
| 結合ヘルパー | `join_v2v(source, continuation, out, *, audio_fade_ms, loudness_match, handle_audio, handle_crossfade_ms)` | `services/video_io.py:549-` |
| 結合に必要な来歴 | `metadata.json` の `v2v` ブロック（`source_video_id` / `audio_handle_filename` / `handle_context_seconds` / `source_had_audio`） | `services/pipeline_manager.py:356-360,499-501`・`services/ltx_runner.py:458-478` |

**サイドカー wav の存在条件（確認結果）**: `chain_pipeline.py:826-839` は `source is not None`（＝V2V）かつ音声トラック有りの経路でのみ `<stem>_audio_handle.wav` を書き出す。`metadata.json.v2v.audio_handle_filename`（実機・mock 共通キー）にファイル名が入る。**存在するかは `source_had_audio` と `audio_handle_filename != null` で判定できる**（元動画に音声が無ければハンドルは無く、フェードペアにフォールバックする）。

**join_v2v の 2 モードは実装済み（`services/video_io.py:598-711`）**:
- **ハンドル無し（`handle_audio=None`）＝フェードペア**: 元動画末尾を fade-out・継続先頭を fade-in（`qsin`＝equal-power・既定 400ms）＋2パス loudnorm。継ぎ目に「谷」が残る（VERIFICATION_LOG §24.7）。
- **ハンドル有り（`handle_audio=<sidecar>`）＝真クロスフェード**: 継ぎ目より前の context 区間で元動画音声とハンドルを `acrossfade=qsin`。継ぎ目自体はハンドルの連続波形（フェード無し）＝**ほぼ無音の谷が消える**。継ぎ目オフセットは `handle_context_seconds = handle_dur − continuation_dur` で導出（`video_io.py:669`）。

---

## 2. エンドポイント形状（加算的・凍結 API 不変）

### 2.1 結合を実行 — `POST /jobs/{job_id}/join`

```jsonc
// Request body（全フィールド optional・省略時は既定で動く）
{
  "audio_smoothing": true,        // 既定 true。true=クロスフェード（ハンドル有れば真クロスフェード/無ければフェードペア）。
                                  //           false=ハードカット結合（音声フェード無し・concat_mp4s 相当）。
  "handle_crossfade_ms": 150      // 既定 150。ハンドル真クロスフェード時のみ有効（フェードペア時は audio_fade_ms=400 固定）。
}
```

```jsonc
// Response 200（同期・GPU 不要・ffmpeg 数秒）
{
  "job_id": "0bab3830-...",
  "joined_path": "outputs/0bab3830-.../joined.mp4",
  "join_mode": "handle_crossfade",   // handle_crossfade | fade_pair | hard_concat | video_only
  "source_lufs": -18.3,
  "handle_context_seconds": 5.375,
  "handle_crossfade_ms_applied": 150,
  "loudness_matched": true
}
```

- 結合結果は `outputs/{job_id}/joined.mp4` に書き出す（ジョブの出力ディレクトリ内・DELETE /jobs でまとめて消える）。`join_v2v` が返す dict（`video_io.py:647-655` の `info`）をそのままレスポンスに載せる。
- **同期 200 を推奨**: `join_v2v` は ffmpeg（loudnorm 2 パス＋再エンコード）のみで GPU 不要・短尺なら数秒。FastAPI は同期エンドポイントをスレッドプールで実行するためイベントループを塞がない。単一ジョブ GPU ガード（`create_chain_if_idle`）とは無関係＝生成中でも結合できる。<span>（代替＝BackgroundTasks 化して `joined_status` をポーリング。長尺で ffmpeg が伸びる場合の昇格経路として温存。）</span>

### 2.2 結合結果を取得 — `GET /jobs/{job_id}/joined`

- `api/jobs.py:31` の `GET /jobs/{job_id}/video`（`output.mp4` を返す `FileResponse`）と同型で、`joined.mp4` を返す新規ルート。既存 `/video` は**一切変更しない**（output.mp4 のまま）。
- `joined.mp4` 未生成なら 404（`JOINED_NOT_READY` でも可・下記 §3）。

---

## 3. エラーコード（既存 `apierr` 体系に加算・`api/errors.py` 前例）

| コード | HTTP | 意味 | 実装 |
|---|---|---|---|
| `JOB_NOT_FOUND` | 404 | ジョブ ID 不在 | **既存流用**（`errors.py:139`） |
| `VIDEO_NOT_READY` | 409 | ジョブ未完了（結合対象の output.mp4 が無い） | **既存流用**（`errors.py:143`） |
| `SOURCE_VIDEO_NOT_FOUND` | 404 | 結合に要る元動画 upload が purge 済み | **既存流用**（`errors.py:68`） |
| `JOB_NOT_JOINABLE` | 422 | 対象が V2V ジョブでない（`metadata.v2v` 無し＝通常チェーン/A2V/単発） | **新規**（`source_video_too_short` 型の factory を追加） |
| `JOIN_FAILED` | 503 | ffmpeg 失敗（`FFmpegError`） | **新規**（`generation_failed` 型・detail に ffmpeg stderr 要約） |
| `JOINED_NOT_READY` | 404 | `GET /joined` を結合前に呼んだ | **新規**（任意・404 で代替可） |

- 新規コードは `api/errors.py` に factory を足すだけ（`source_audio_not_found` 等と同型）。GUI 側は既存 `format_api_error`（`gradio_ui/formatting.py:49`）が `apierr_<code>` ラベルを引くので、`i18n.py` に日英ヒントを 3 個足せば自動で露出する（`apierr_*` 追加のみ・既存 15 コードは不変）。

---

## 4. クロスフェード ON/OFF の意味（§24.7 準拠の整理）

VERIFICATION_LOG §24.7 将来項目①の定義に合わせる:

| UI（音声スムージング） | `audio_smoothing` | ハンドル有無 | join_v2v 呼び出し | `join_mode` | 音の継ぎ目 |
|---|---|---|---|---|---|
| **ON** | `true` | サイドカー有り | `handle_audio=<sidecar>, handle_crossfade_ms=150` | `handle_crossfade` | ほぼ無音の谷が消える・残るは窓内の浅い凹み（深さ0.42・幅105ms）|
| **ON** | `true` | サイドカー無し（元動画に音声なし等） | `handle_audio=None`（フェードペア 400ms） | `fade_pair` | 継ぎ目に谷（depth≈0・~710ms）|
| **OFF** | `false` | — | ハードカット（`concat_mp4s` 相当・音声フェード無し） | `hard_concat` | 段差がそのまま聞こえうる |

- 既定 ON・150ms は §24.7 §4 の計測（`V2V_AUDIO_JOIN_RESEARCH.md §4`）で「凹み幅を 245→105ms へ半減」＝内容由来の下限 85ms に最接近した実績値。
- **数値・意味の正本は `V2V_AUDIO_JOIN_RESEARCH.md`（§2 フェードペア・§4 ハンドル）と VERIFICATION_LOG §24.7**。本エンドポイントは既存 `join_v2v` の 2 モードに OFF（ハードカット）を足して REST から叩けるようにするだけで、**音声処理そのものは新開発しない**。
- <span>要判断（親へ）: OFF＝ハードカットを本当に露出するか。§24.7 の定義には有るが、実務では「ON（既定）／OFF は結合版を作らない」の 2 択で足りる可能性。OFF=ハードカットは比較検証用の色が濃い。</span>

---

## 5. 既存エンドポイント不変の担保 & テスト計画

### 5.1 不変の担保
- **新規ルートのみ追加**（`POST /jobs/{job_id}/join`・`GET /jobs/{job_id}/joined`）。`/generate`・`/generate/chain`・`/upload/*`・`/jobs`・`/jobs/{id}`・`/jobs/{id}/video`・`/jobs/{id}`(DELETE) は 1 行も触らない。
- `join_v2v`（`video_io.py:549`）は既に本番在。`handle_audio=None` 時は byte 同一（docstring 明言・`video_io.py:625`）。追加するのは「REST の口」と「`concat_mp4s` によるハードカット分岐」のみ。
- 出力は `joined.mp4`（別名）。`output.mp4` を上書きしないので `/video` の返り値は不変。
- エラーは新コード追加のみ。既存 15 コードのラベル・envelope は不変（`errors.py:31` `to_envelope`）。

### 5.2 テスト計画（GPU 不要＝子A のスコープで完結可能）
1. **回帰（既存不変）**: T2V `23844b4e…` / I2V `a511eda4…` / no-source チェーン `f706057a…` の byte-match と pytest 212+1 を維持（新規ルート追加が既存を壊さないこと）。
2. **`join_v2v` 単体**（`tests/test_video_io.py:386-` 前例）: 合成 mp4＋合成ハンドル wav で `handle_crossfade` / `fade_pair` / `hard_concat` / `video_only`（音声無し）の 4 分岐・出力尺 assert・`join_mode` の値を確認（既にハンドル系テストは存在＝拡張のみ）。
3. **API 統合（MockTransport 型 + 実 uvicorn mock）**: mock backend は既に `v2v` メタ＋`output_audio_handle.wav` プレースホルダを出す（`ltx_runner.py:453,476`）。この上で:
   - V2V mock ジョブ完了 → `POST /jobs/{id}/join`（ON）→ 200・`joined.mp4` 生成・`join_mode` 正・`GET /joined` 200。
   - `audio_smoothing:false` → `hard_concat`。
   - negative: 通常チェーン/A2V ジョブへ join → `JOB_NOT_JOINABLE` 422／未完了ジョブ → `VIDEO_NOT_READY` 409／不在ジョブ → `JOB_NOT_FOUND` 404／`GET /joined` を join 前 → 404。
   - mock のプレースホルダハンドルは**合成無音**（`ltx_runner.py` docstring）なので、真クロスフェードの音質検証は G3 実機扱い＝子A では契約（存在・キー・mode・尺）のみ固定。
4. **GUI ハンドラ単体**（`gradio_ui` MockTransport）: 「結合版を作成」ボタン → `api.join_job(job_id, ...)` → 成功パス・エラーパス（`format_api_error` に新コードが流れる）。

---

## 6. リスク・未決事項

- **R1 同期 vs 非同期**: 同期 200 を推奨（GPU 不要・短尺数秒）。長尺 V2V（257f＋元動画）で loudnorm 2 パス＋再エンコードが数十秒に伸びる可能性。60s タイムアウト内なら同期で十分だが、伸びるなら BackgroundTasks 化＋`joined_status` ポーリングへ昇格（設計は温存）。**要親判断**。
- **R2 元動画 upload の寿命**: 結合は元動画の実ファイルに依存する。upload ストアが TTL/purge を持つ場合、生成から時間が経つと `SOURCE_VIDEO_NOT_FOUND`。GUI は結合を「生成直後に促す」導線が安全。
- **R3 解像度/fps 一致要件**: `join_v2v` は元動画と継続の解像度・fps 一致を要求し不一致で `FFmpegError`（`video_io.py:632,639`）。V2V は app 側で元動画末尾を request fps にリサンプル済み（`pipeline_manager.py:353`）だが、**元動画フル尺の fps／解像度が出力と一致するとは限らない**（context 末尾だけ揃えている）。→ 結合前に app 側で元動画を出力解像度・fps に正規化する一時 mp4 を作る必要があるか要確認。**これは実装 S1 での検証項目**（G0 相当のスモークで実素材確認）。
- **R4 OFF＝ハードカットの要否**: §4 の通り、露出するか 2 択（ON／結合版を作らない）で足りるか。**要親判断**。
- **R5 A2V との関係**: A2V 出力は元波形をそのまま mux 済み・結合面が無い（`A2V_DESIGN §2.4`）＝**join 非対象**。A2V ジョブへの join は `JOB_NOT_JOINABLE`（422）で弾く。

---

## 7. まとめ（実装の最小差分見取り）

- `api/jobs.py`: `POST /jobs/{id}/join`＋`GET /jobs/{id}/joined` の 2 ルート追加。
- `api/models.py`: `JoinRequest{audio_smoothing, handle_crossfade_ms}`＋`JoinResponse` 追加（optional・加算的）。
- `api/errors.py`: `job_not_joinable`（422）・`join_failed`（503）・（任意）`joined_not_ready`（404）factory 追加。
- `services/pipeline_manager.py`（または新 `services/join_manager.py`）: metadata から `source_video_id`/`audio_handle_filename` を読み、`video_upload_store.path_for` で元動画解決 →（R3 の正規化）→ `join_v2v` 呼び出し → `joined.mp4` 書き出し。
- `services/video_io.py`: OFF（ハードカット）分岐のみ追加（`concat_mp4s` 流用・音声フェード無し）。既存 2 モードは不変。
- `gradio_ui/`: `api_client.join_job`／ボタンハンドラ／`i18n` に `apierr_*` 3 個。
- テスト: §5.2。すべて GPU 不要（mock）＝**子A スコープで完結可能**。
