# R5 実装差分レビュー: MiniMax H3 A2V のアプリ層と操作パネル（計画 §2.5〜§2.8・§4 G2）

- 対象: バックエンド `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`（dev・未コミット）。操作パネル `webui` は同じ git リポジトリ（トップは同じ `Nz-Videomni`）。`engine_h3/` は R4 で済んでいるので見ていない。
- 読むだけのレビュー。確認のために次を流した（いずれも GPU は使わず、実バックエンドも起動していない）。
  - `.venv` の pytest: `test_h3_a2v_api.py`・`test_engine_h3_adapter.py`・`test_ltx_adapter_audit.py`・`test_ltx25_adapter.py`・`test_engine_dispatch.py`・`test_mcp_tools_generate.py`・`test_mcp_tools_system.py`。全件合格。
  - `npx vitest run --exclude "**/backend.integration.test.ts"`: 変更・新規の 8 ファイル、194 件すべて合格。
  - `npm run typecheck`: 合格。
- 結論: **Critical 0・Major 0・Minor 9・情報 6**。計画からの逸脱はない。加算だけの変更であること（LTX の constraints・`/generate/chain`・`generation_mode`）も守られている。F3 の観察（273 フレーム）は試験の環境でしか起きず、実際の流れでは起きない。仮に起きても、画面の格子の検査で Generate が押せなくなるだけで、422 には届かない（§3 M1）。

## 一覧（重い順）

| # | 重さ | 場所 | 要旨 | 修正案 |
|---|---|---|---|---|
| M1 | Minor | FE `useGenerationForm.ts:1142-1161`、`SingleScreen.sourceAudio.test.tsx:101-109` | wav の提案値に、LTX の格子を前提にした上限（快適上限・`a2vSeedMaxFrames`）を後から掛けている。H3 の格子に戻す処理が無い。試験は 277 を打ち直す回り道でこれを避けており、コメントの原因の書き方も誤っている | 上限を掛けた後に `floorToGrid(suggested, frameGrid)` を 1 行足す。試験は `engineFamily="h3"` を渡し、回り道を消す |
| M2 | Minor | FE `bridge/mockBridge.ts:1747-1762` | 模擬サーバーの LTX の 422 の `detail` が、実サーバーの `H3_ONLY_REFUSALS["source_audio"]` と違う（「MiniMax H3 を選んでください」となっていて、Chained への案内が無い） | `detail` を実サーバーの文と同じにする |
| M3 | Minor | `services/pipeline_manager.py:1336-1349` | 同じ関数の中で、`detail` が英語（音声の流れが無いとき）と日本語（短いとき）に分かれている。チェーンの前例は英語 | どちらかの言語にそろえる（H3 の文言の前例に合わせるなら両方とも日本語） |
| M4 | Minor | `mcp_server/tools/generate.py:605-609`、`Docs/MCP_SERVER_DESIGN.md` D30 | 強度を省いたとき MCP が 1.0 を自分で書き込む。D27・D28 の「MCP は既定値を持たない」という規律から外れている（理由は D30 に書いてある） | 規律に合わせるなら、強度が `None` のときは `strength` を送らない（サーバーの既定 1.0 が使われる）。今のままにするなら手直しは不要 |
| M5 | Minor | `tests/test_h3_a2v_api.py:157-164` | `test_real_payload_without_source_audio_is_the_golden` が、既存の `test_engine_h3_adapter.py:560` `test_generate_payload_t2v_golden` と同じことを確かめている（§3-238 の試験の量の懸念に当たる） | 新しい方を消す。前の試験の `keys` の比較だけで、golden が保たれていることは示せている |
| M6 | Minor | `tests/test_h3_a2v_api.py` 全体 | ffprobe／ffmpeg が無いときに飛ばす印が無い（前例は `test_retake_api.py:187` の `skipif(not _HAS_FFMPEG)`）。ffprobe が無い環境では、`has_audio_stream` が False になり 422 で落ちる | ffprobe に頼る試験に、前例と同じ `skipif` を付ける |
| M7 | Minor | `tests/test_h3_a2v_api.py` | 二つの分岐に試験が無い。①音声の流れが無いアップロード（動画だけの mp4 など）が 422 `SOURCE_AUDIO_TOO_SHORT` になること。②長さが測れないときに通すこと | ①を 1 本足す。②は `probe_duration` を monkeypatch で None にして 202 を見る 1 本 |
| M8 | Minor | FE `SingleScreen.sourceAudio.test.tsx:113-116` | 「チェーン専用の項目が無い」ことの確認が `clips` だけ。計画 §2.7 の本文の条件（画像の strength を省くこと、LoRA の `audio_strength` を外すこと）を見ていない | `body.conditioning_images` の各要素に `strength` が無いこと、`body` に `overlap_frames`・`stage2_window` が無いことを足す（キーフレームが無いならその旨のコメント） |
| M9 | Minor | `services/pipeline_manager.py:1316-1350`・`716-730`、`h3/adapter.py:683` | 前の検査が返す ffprobe の事実を `api/generate.py:115` が捨てている。1 本の A2V ジョブで ffprobe が 6 回呼ばれる（前の検査 3 回、`run_job` 2 回、模擬サーバー 1 回） | 戻り値を使わないなら `-> None` にして説明文を直す（チェーン版と同じ形）。呼び出しの回数は実害が小さいので、そのままでよい |
| I1 | 情報 | `api/errors.py:101-104` | `SOURCE_AUDIO_TOO_SHORT` の `message` が変わったので、`/generate/chain` の 422 の文も変わる（code は同じ）。依頼にあった「中立な文」にはなっている | D1 の仕様書の更新で、チェーンの 422 の文も新しい文として書く |
| I2 | 情報 | `pipeline_manager.py:1936` | `request` の書き出し（`model_dump`）の都合で、普通のジョブの metadata にも `"source_audio": null` が増える。`vram16_mode`・`alpha_gen` と同じ扱いで、前例どおり | 不要 |
| I3 | 情報 | `pipeline_manager.py:1333-1337` | 音声の流れが無いときと ffprobe が無いとき（`has_audio_stream` が False）も 422 `SOURCE_AUDIO_TOO_SHORT` になる。計画 §2.5 の指定どおりで、チェーンの `preflight_source_audio` と同じ動き。依頼の項目 2 への答えとして「計画が許している」 | 不要 |
| I4 | 情報 | FE `strings.ts:1675`・`3306` | カードの `note` が「単一の distilled クリップを使用」のまま。H3 の経路では正しくない。計画では見た目を変えないことになっているので範囲外 | オーナーの判断に任せる（直すなら D1 か次の弾で） |
| I5 | 情報 | FE `SingleScreen.tsx:1029`・`AppShell.tsx:1543` | 送り先を決めるのに `a2vRouteFor` を使っておらず、`singleSourceAudio !== null` を直接見ている。元になるデータは同じ（`activeConstraints` → `constraints`）なので、結果は一致する | 不要（揃えたいなら `a2vRouteFor(...) === "generate"` を使う） |
| I6 | 情報 | `Docs/PENDING_TASKS.md` | 担当表に無い変更として §3-256 が足されている（音声が約 8.3 ミリ秒短くなる件）。計画 §5 に「§3-256 はそのまま」とあり、監督が起票したものと読める | 監督がコミットに含めるかを決める |

## 1. 加算だけの変更か・バイト単位で同じか（項目 1）

- LTX の constraints: `request_rules.py:127-149` は、`source_audio_strength` が None のときキーを出さない。`test_engine_h3_adapter.py:305-306`・`707-708` は LTX の constraints を辞書で完全一致させて確かめており、合格している。H3 だけが `prompt_token_limit` の後ろに `source_audio` を持つ。
- 普通の `/generate`: ワーカーへのペイロードは `h3/adapter.py:563-567` で、`source_audio` を値があるときだけ `output_path` の直前に入れる。`test_h3_a2v_api.py:157-164` と既存の golden で、順序が変わらないことを確かめてある。metadata は、`outcome.h3` に `a2v` が無ければ今日と同じ値を `metadata["h3"]` に書く（`pipeline_manager.py:2032-2040`）。`a2v` キーは、出所かエンジンの記録のどちらかがあるときだけ書く（2048-2053）。
- `/generate/chain`: `generate_chain.py` には差分が無い。`api/models.py` の変更は `to_clip_request` の説明文だけ（1776-1784）。`SourceAudioSpec` も変わっていない。
- `generation_mode`: 変わっていない。キーフレームと音声を併用したジョブは `i2v` になる（`test_h3_a2v_api.py:117`）。

## 2. エラーの順番と形（項目 2）

- `api/generate.py:79-117` の順番は、系統による拒否 → `validate_for_family` → `alpha_gen` → キーフレームの存在 → 音声の 404 → 前の検査の 422 で、指定どおり。404 は `audio_upload_store.path_for` の APIError を `source_audio_not_found` に包む形で、`generate_chain.py:80-83` と同じ。
- LTX 2.3・2.5 とも `REJECT_TABLE` に 1 行ずつ足され（`ltx/adapter.py:143-145`、`ltx25/adapter.py:147-149`）、文は共有の `H3_ONLY_REFUSALS`（`ltx/adapter.py:153`）から出す。2.5 は 52 行目でこれを読み込み、534-535 行目で使っている。「もう一方の LTX を選んで」という文は出ない。両方の系統に対して、`test_h3_a2v_api.py:121-132` が `detail` の完全一致で確かめている。
- `SOURCE_AUDIO_TOO_SHORT` を「音声の流れが無い」ときにも使うのは、計画が許している（I3）。

## 3. pipeline_manager（項目 3）

- `run_job`（716-730・998-999）: `**runner_extra` は値があるときだけ `source_audio_path` を渡すので、LTX のランナーの引数の形は変わらない。
- `a2v_provenance` は `_finalize`（1880・1911）を通って `_write_metadata` に渡る。エンジン側のキー 10 個（模擬サーバーの `mock` を含む）とアプリ側のキー 3 個は重ならない。重ならないことは `test_h3_a2v_api.py:65-66` で確かめてある。
- 許容の幅: アプリは `duration + 0.001 < F/fps` で断る。エンジンは `required = round(F·32000/fps)`（`pipeline_h3.py:665`）に対して 32 サンプル（32 kHz で 1 ミリ秒）まで 0 で埋める（`source_audio.py:36`・121）。両者の幅は合っている。ffprobe のコンテナの長さと、実際に復号できたサンプル数の差は、エンジン側の歯止めが受け止める。
- 長さが測れないとき（`duration is None`）は通す（1340）。ffprobe が無いときは、その前の `has_audio_stream` で 422 になる。チェーンの前例と同じ（I3）。
- metadata にパスは入らない。アプリ側は ID・強度・ffprobe の事実だけを書き、エンジン側のキーにもパスは無い。
- M3・M9 は一覧のとおり。

## 4. H3 のアダプタ（項目 4）

- `HONOURED_FIELDS` に `source_audio` が足された（250-252）。`REQUEST_RULES.source_audio_strength = SamplingRange(0.0, 1.0, 0.05, 1.0)`（156-158）。
- `H3Runner.generate`（706-738）の引数は `LTXRunner.generate`（`ltx/adapter.py:810-822`）と同じ並びで、最後に `source_audio_path` が加わっている。本体も同じで、バックエンドの読み込みと委譲をしている。
- `_RealBackendH3` は、`request.source_audio` があるのにパスが無ければ `RuntimeError` を出す（527-537）。強度は `request.source_audio.strength` から読む。`keep_resident` は最後に置かれ、`test_h3_a2v_api.py:152-154` で順序を確かめてある。
- `_MockBackendH3(_MockBackend)` なので、`isinstance` の判定は保たれる。登録の試験は `issubclass` で書き直されているが、弱めてはいない。同じクラスかどうか（`is`）も見ている（`test_engine_h3_adapter.py:271-273`）。模擬サーバーは `muxed_audio_samples: None` を返し、音声を重ねたとは言っていない。`channels_used` の語は、エンジンの `source_audio.py:112-117` と同じ。
- 進み具合: `audio_encode` は `done.h3.phases` の段の名前としてだけ現れ、進み具合の段にはならない（説明文 36-41、worker.py:73）。進み具合の表示は変わらない。

## 5. MCP（項目 5）

- 引数は 2 つ（189-190）。強度だけを渡すと HTTP より前に `ToolError`（517-522）。入れ子は `{"audio_id", "strength"}`（605-609）。`source_audio_id` が無ければキーごと出ない。これらは `test_mcp_tools_generate.py:2088-2117` の 4 本で確かめてある。
- 説明文: H3 専用であること、LTX は `submit_chain` を使うこと、1.0 未満では出力の音声がモデルのものになること、長さの規則、が書かれている。`server.py:50-52` も「連結は不可・単発 A2V は可」に直っている。D30 の文は計画と一致し、D 番号も D29 の次で、見出しの範囲も「D1〜D30」に直っている。M4 だけが規律の面で異なる。

## 6. 操作パネル（項目 6）

- 送り先: `SingleScreen.tsx:1038-1052`。`/generate` の経路では、普通の単発の流れと同じく LoRA の音声の強度を外し、画像の strength を省いた本文に `source_audio: {audio_id, strength}` を足す。チェーン専用の項目は付かない。強度は本文に入るので、同じ設定の判定（`submissionFingerprint(buildSubmission())`）にも自然に含まれる。
- 能力で決めているか: `a2vRouteFor`・`singleSectionsDisabledFor`（`featureScope.ts:406-428`）と `AppShell.tsx:1543` は規約の有無だけを見ていて、エンジン名を読む箇所は無い。規約は `resolveGenerationConstraints` に 1 行で入り（`generationConstraints.ts:95`）、LTX の既定値は `null`（74）。
- `audioFit.ts`: 3.886 秒 × 24 = 93.26 → 93 → (93−5)/17 の切り捨ては 5 → 90。最小値・最大値・誤差の吸収（1e-6 フレーム）も正しい。最大値が格子の上に無い場合も、切り捨てで格子に乗る。画面側の検査は、サーバーの許容（1 ミリ秒）より最大 0.024 フレームだけ厳しい。安全側に寄っているので問題は無い。`useGenerationForm` は規約があるときだけこれを使い（1083-1100・1184-1193）、無ければ従来の `snapNumFrames(suggestFramesForAudio(...))` を使う。
- 強度: 既定はサーバーの値（無ければ 1.0）。設定するときは範囲に収め、規約が無ければ何もしない。系統を切り替えると既定に戻る（784-792・814）。試験は `useGenerationForm.h3.test.ts:222-239`。
- スライダー: `KeyframeCard.tsx:250-262` と同じ作り（`kfc-field`・`kfc-field-label`・`kfc-hint`、`toFixed(2)`）。min・max・step はサーバーの値。`singleClipNote` は `/generate` の経路では出ない（`GenerationForm.tsx:737`）。文言は英語と日本語の両方にあり、中身も正しい（`strings.ts:1689-1691`・`3314-3316`）。
- 模擬サーバー: H3 の constraints（376-378）、LTX 2.3・2.5 の一覧の末尾（439-441・511-512）、LTX での 422（1747-1762）。文の食い違いは M2。
- **F3 の観察（273）を追った結果**:
  - 右クリック #7 の種は `resolvePrefillSeed` → `computeTargetNumFrames` で決まる。素材の長さが無ければ `null` を返し（`deriveDuration.ts` の materialClampedToCeiling）、その場合 `a2vSeedMaxFrames` は設定されない。初期値も `useGenerationForm.ts:761-766` で、その時点の格子（17n+5）にそろえられる。だから、273 は「種」からは来ない。
  - 273 の出どころは、wav を測った後の W8 の上限 `spillThresholdFrames` である。試験は `engineFamily` を渡さないので、`comfortProfileMissing` が false になる。すると、LTX の古い表の `spill_free_frames["1280x768"] = 273`（`defaultConfig.ts:58`）が引かれ、提案値の 464 が 273 に切り下げられる。コード中のコメント（1142-1143・1148-1150）は「上限はどれも 8n+1 の上にある」と LTX を前提にしている。
  - 実際の流れでは、`engineFamily="h3"` が渡り、H3 には快適上限の表の項目が無い（模擬サーバー 340 行目。`scripts/manifests/` に H3 の `comfort` は無い）。このため上限は null になり、提案値は格子の上に残る。H3 の素材の長さから決まる種も、H3 の格子で切り捨てられ、上限が無いので格子の上にある。
  - 仮に格子から外れた値になっても、`validityReasons` の `numFramesOffGrid`（1251）で Generate が押せなくなる。サーバーの 422 には届かない。
  - 結論: 実際の流れで H3 の利用者が格子から外れたフレーム数を送ることはできない。ただし前提が暗黙なので、M1 のとおり、上限を掛けた後（1161 の `setValues` の直前）で格子にそろえ直すのが安全である。

## 7. 試験（項目 7）

- 完全一致の試験は、更新後もすべて一覧の完全一致か長さのまま残っていて、弱めたものは無い。`test_ltx_adapter_audit.py:161`（タプルの完全一致）、`test_ltx25_api_guard.py:1662-1664`（`len == 4` と集合の一致）、`test_mcp_tools_system.py:335-340`、`test_engine_dispatch.py:228`、`useBaseModels.test.ts:245・262・398`、`mockBridge.test.ts:1002・1013`。
- G2 の項目と `test_h3_a2v_api.py` の対応:

  | G2 の項目 | 試験 |
  |---|---|
  | mock で A2V を 1 本流す | 52-61・63-75 |
  | metadata の形（`h3` が出ない） | 75 |
  | 404 | 98-102 |
  | 短い音声の 422 | 86-95（ジョブが増えないことも確認） |
  | LTX の 422 と H3 専用の文 | 121-132 |
  | キーフレームと音声の併用 | 105-117 |
  | ペイロードの位置 | 143-154 |
  | 普通のジョブが変わらない | 135-140 |

  mp4 ができたことは直接は見ておらず、`completed` で代わりにしている。前例の試験と同じ程度。
- 中身の無い試験は無い。重複は M5、抜けは M6・M7。
- 操作パネルの試験は、送り先・本文・スライダーの表示と値・既定への戻り、という振る舞いを見ている。実装の細部に寄っているのは `data-testid` を使う点くらいで、許容の範囲。M1 の回り道（`waitFor` の中で値を打ち直す）は壊れやすく、15 秒の待ち時間の原因にもなっている。

## 8. 後始末（項目 8）

- `git status` で担当表に無いのは `Docs/PENDING_TASKS.md` だけ（I6）。`scripts/h3_a2v_g0_roundtrip.py` と `tests/test_h3_engine.py` はエンジン担当の表にある。
- デバッグ用の出力の残りは無い。改行コードの警告（LF→CRLF）が出ているファイルもあるが、差分の行数は小さく、改行コードだけの差分にはなっていない（`request_rules.py` は全行 LF）。
- コメントの書き方は周りと同じ。H3 の `RuntimeError` は日本語で、H3 の前例に合っている。
- `test_h3_a2v_api.py` が `test_engine_h3_adapter` から補助関数を読み込んでいる。動くが、試験の間に依存ができる。前例があるなら問題は無い。
