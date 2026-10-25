# R3 計画の事実確認レビュー（H3 の A2V・§1-103）

- 対象: `C:\Users\phenotype nezu\.claude\plans\cosmic-giggling-bubble.md`（以下「計画」）。補助: R1・R2。
- 方法: 読み取りのみ。GPU・インストール・状態を変える git 操作はしていない。行番号はすべて実物で確かめた。
- 注記: 依頼の保存先（scratchpad の `h3a2v\R3_plan_review.md`）は、このセッションがプランモードで書き込み先がこのファイルに限られたため、ここに保存した。

## 0. 指摘の一覧（重い順）

| # | 重さ | 要旨 |
|---|---|---|
| 1 | Major | §2.1 の整理した式 `v_s = m²v + (1−m²)(clean−x)/σ` は「inject = clean」（音声）のときだけ正しい。映像（inject = 0.999·clean + 0.001·n4）では項が 1 つ足りない。式の正本にそのまま載る |
| 2 | Major | LTX の拒否行に `H3_ONLY_REFUSALS` への追加が書かれていない。無いと 2.3 は「LTX 2.5 を選んで」、2.5 は「LTX 2.3 を選んで」と誤った案内を返し、しかも既存の監査テストは通ってしまう |
| 3 | Major | 「監査テストはそのまま通る」（§0）は誤り。LTX の `unsupported_features` を完全一致で見ている試験が 6 か所以上あり、うち 3 ファイルは担当表に無い |
| 4 | Major | PyAV の変換器: 末尾を出し切る `resample(None)` が書かれていない。加えて、入力がすでに 32 kHz・fltp のとき（32 kHz の AAC など）は素通しになり、`resample(None)` が `[None]` を返す |
| 5 | Major | 操作パネルの模擬サーバー（`bridge/mockBridge.ts`）が H3 の `constraints.source_audio` を公開しないので、新しい経路を模擬サーバーで一度も通せない。LTX の機能名の写しもずれる |
| 6 | Minor | 事前検査の許容 0.001 秒とエンジンの厳密な検査（`N < keep_samples`）が食い違う。最大 32 サンプル足りない音声が 422 を抜け、ジョブの失敗になる |
| 7 | Minor | §2.3 の数値例が誤り。124,352 サンプルは 124,800 に 0 埋めされ、潜在は 156 個（計画は 125,600・157 個） |
| 8 | Minor | §2.2 の表の 1016 行は `am.rows` を無条件に読んでいる。マスクが None のとき落ちる |
| 9 | Minor | `entry["mask"]` を無条件に書くと、A2V でないジョブの `h3.phases.denoise` も変わる |
| 10 | Minor | G1 ②「混在マスクで m=1 の行がビット一致」は、既存の小さな実物の DiT では成り立たない（注意で行が混ざる） |
| 11 | Minor | 映像マスクの組み立てと乱数の引き方の仕様が噛み合わない（`_denoise` の外で作れない）。ComfyUI は注入用の雑音に初期雑音を使い回しているはず（要確認） |
| 12 | Minor | G3-0 は、作りの上で同じ経路を 2 回流すだけになり、GPU の費用に見合わない |
| 13 | Minor | G0 の判定: 波形の相関はずれ（遅延）を探す前提が書かれていない。主の基準と予備の基準の関係もあいまい |
| 14 | Minor | 担当の割り当ての漏れと重なり: ワーカーの試験は `tests/test_h3_engine.py` にあり、E2 の担当ファイルと重なる。`_finalize` の引数と TypeScript の `GenerateRequest` 型も表に無い |
| 15 | Minor | 細かい抜け: ペイロードのキーの順（`keep_resident` は末尾）、`zero_k_bias` は永続の buffer にすること、`has_audio_stream` の検査、`singleClipNote` の文言、文書の置き場所 |
| 16 | Minor（過剰設計） | `source_audio_rules.py`（新モジュール）と取り次ぎ関数。前例どおり `RequestRules` の任意の項目 1 つで足りる。公開する `latents_per_second`・`sample_rate` は操作パネルでは使われない |

Critical はなし。指定された確認点 1〜10 への答えは §2 にまとめた。

## 1. 指摘の詳細

### 1. 映像では整理した式が成り立たない（Major）
- **主張**: §2.1 の 5 で、`v_s = (x0_b − x)/σ` を整理すると `m²·v + (1−m²)·(clean − x)/σ` になる、としている。
- **根拠（式の展開）**: x̃ = m·x + (1−m)·inject。x0̂ = x̃ + σ·m·v。x0_b = m·x0̂ + (1−m)·clean。これを展開すると次のとおり。
  - x0_b = m²x + σm²v + (1−m)(m·inject + clean)
  - よって v_s = m²v + (1−m²)(clean−x)/σ + **m(1−m)(inject − clean)/σ**
  - 最後の項は inject = clean（音声）のときだけ 0 になる。映像では inject − clean = 0.001(n4 − clean) なので 0 にならない。
- **影響**: V1 は映像のマスクを出さないので、製品には影響しない。ただし、この式は MINIMAX_H3_DESIGN §3.1 の正本になり、§1-106 で映像にも使う予定。G1 ⑤（`(x0_b−x)/σ` との一致）を映像でも試せば落ちる。
- **直し方**: `state_velocity` は整理した形を使わず、`(x0_b − x)/σ` をそのまま計算する。式の正本には一般形を書き、音声（inject = clean）のときに整理できる、と添える。G1 ⑤ は映像と音声の両方で試す。

### 2. `H3_ONLY_REFUSALS` への追加が無い（Major）
- **根拠**: `services/engines/ltx/adapter.py:148-150` の `H3_ONLY_REFUSALS` と、`reject_unsupported` の 284-295 行。表に無い機能名のときは「LTX 2.3は…対応していません…『LTX 2.5』を選んでください」という汎用の文言になる。2.5 側（`ltx25/adapter.py:529-535`）は同じ表を共有し、無ければ「LTX 2.3 を選んで」と返す。
- **見落としやすい点**: `tests/test_ltx_adapter_audit.py:139-143` は、`_H3_ONLY` に無い機能名なら「detail に field 名と "LTX 2.5" を含む」ことしか見ていない。そのため誤った文言でも試験は通る。
- **直し方**: 計画の文言（「単発の A2V は MiniMax H3 専用です。LTX の A2V は Chained の音声で使えます」）を `H3_ONLY_REFUSALS["source_audio"]` に置くと、§2.5 に明記する。VRAM 16GB mode の前例と同じ形。

### 3. 「監査テストはそのまま通る」は誤り（Major）
LTX の `UNSUPPORTED_FEATURES` は `REJECT_TABLE` の機能名の列から作られる（`ltx/adapter.py:241-243`、`ltx25/adapter.py:477-479`）。そのため行を足すと、LTX 2.3 と 2.5 の公開リストに `source_audio` が増える。次の試験が落ちる。

| ファイル:行 | 中身 | 計画の担当 |
|---|---|---|
| `tests/test_ltx_adapter_audit.py:160` | LTX 2.3 の一覧と完全一致 | T1 |
| `tests/test_ltx_adapter_audit.py:129-130` | `REQUEST_OVERRIDES_23` が拒否行をすべて覆う | T1 |
| `tests/test_ltx25_adapter.py:513` | `REQUEST_OVERRIDES` と拒否表が一致 | T1 |
| `tests/test_ltx25_api_guard.py:512`、`1620-1624` | 同上と、`/models` の LTX 2.3 の一覧 | T1 |
| `tests/test_engine_dispatch.py:224-228` | `unsupported_features("ltx")` と完全一致 | **担当なし** |
| `tests/test_mcp_tools_system.py:334-338` | MCP〔AI に道具を渡す仕組み〕の `list_models` の LTX 2.3 の一覧 | **担当なし**（M1 は `test_mcp_tools_generate.py` だけ） |
| `tests/test_engine_h3_adapter.py:300-304` | H3 の constraints と完全一致（`source_audio` のキーが増える） | T1 のファイルだが、計画は 73-81・271 しか挙げていない |
| FE〔フロントエンド〕の `shell/useBaseModels.test.ts:245`・`262`・`398`、`bridge/mockBridge.test.ts:1001` | 模擬サーバーの LTX の一覧（#5 で模擬サーバーを直すと連動する） | **担当なし** |

- **直し方**: §0 の一文を「次の試験の期待値を更新する」に改め、上の表を担当に割り振る（1 ファイル 1 担当の規則に合わせる）。73-81 行の分類の試験は `issubclass` と関係ない。`HONOURED_FIELDS` に足せば通る。`issubclass` に直すのは 271 行だけ。

### 4. PyAV の末尾の出し切りと素通し（Major）
- **根拠**: `.venv-engine-h3/Lib/site-packages/av/audio/resampler.py`（PyAV 18.1.0。`engine_h3/venv-engine-h3.freeze.txt:46`）。
  - `layout=None` は正しい呼び方。最初のフレームの並びを引き継ぐ（67-69 行）。
  - 入力の形式・並び・レートがすべて出力と同じなら素通しになる（72-79 行）。素通しの後に `resample(None)` を呼ぶと `[frame]`、つまり `[None]` を返す（59-60 行）。32 kHz の AAC（復号すると fltp）が当てはまる。
  - 素通しでないときは、`resample(None)` で出し切らないと、変換器が抱えた末尾の数十サンプルが失われる。前例の `media_out.py:173` は出し切っている。
- **影響**: 音声がちょうど映像の長さのとき、末尾が欠けて `N < keep_samples` に引っかかる。32 kHz の AAC では、出し切りを素直に書くと `None` を扱って落ちる。
- **直し方**: `load_source_waveform` は、読み込みのあとに必ず `resample(None)` を呼び、`None` の要素を捨てる。試験は E1 の CPU 試験で 2 種類用意する。①44.1 kHz の wav で、出し切りの有無でサンプル数が変わること。②32 kHz の fltp の入力（素通し）で落ちないこと。途中で形式が変わる入力は `ValueError`（124-125 行）になる。日本語の理由をつけて包む。

### 5. 模擬サーバーの公開内容（Major）
- **根拠**: `webui/src/bridge/mockBridge.ts:362-377`（系統ごとの constraints。H3 に `source_audio` が無い）。379-391 行には「サーバーの一覧をそのまま写す」という規則が書かれている。`handleGenerate`（1671-1759）は知らないキーを通すので、本文そのものは受かる。
- **影響**: 計画の F2 は「`/generate` の本文をそのまま受けるか」だけを確かめる。しかし規約が無いと経路は `null` のままで、カードは灰色のまま。模擬サーバーでは新しい経路を一度も通せない。LTX の `unsupported_features` の写しも本物とずれる。
- **直し方**: F2 の担当に次の 3 つを足す。H3 の constraints に `source_audio` を足す。LTX 2.3・2.5 の一覧に `source_audio` を足す。LTX が有効なとき `source_audio` 付きの `/generate` を 422 FEATURE_UNSUPPORTED にする。連動する FE の試験は #3 の表のとおり。

### 6. 長さの許容の食い違い（Minor）
- **根拠**: 事前検査は `duration + 0.001 < F/fps` で 422 にする。1 ミリ秒は 32 サンプル。エンジンは `N < keep_samples` で止める（厳密）。
- **影響**: 1 ミリ秒以内で短い音声は 422 を抜け、ワーカーでジョブの失敗になる。テキストエンコーダーの前で止まるので費用は小さいが、オーナー裁定 5 の「足りなければ 422」と食い違う。
- **直し方**（どちらか 1 つ）:
  - 事前検査の許容を 1 サンプル程度（ffprobe の丸め分）にする。
  - エンジンの歯止めも同じ 32 サンプルの許容にする。mux〔映像と音声を 1 ファイルにまとめる〕は `audio_keep_samples` の min で切るだけなので、埋めは起きない。
  - 操作パネルの `singleAudioPrecheck`（`F/fps ≤ sec`）はより厳しい側なので、問題ない。

### 7. §2.3 の数値例（Minor）
124,352 / 800 = 155.44 なので、0 埋めの先は 124,800、潜在は 156 個。先頭の 150 個を使う結論は変わらない。文書に写す前に直す。

### 8. 1016 行の書き方（Minor）
表の `masked_row_times(t_a, sig_a[i], am.rows, 1.0)` は無条件に読まれる。`audio_t=None if am is None else …` と明記する。そうすれば「None の経路は今日のコードそのもの」（G1 ⑦）が字面でも保たれる。あわせて `time_rows` の中では、`t[layout.cond] = max(...)` を新しい代入の**後**に残す。現状は 267-268 行の順で、条件行が最後に上書きしている。

### 9. `entry["mask"]`（Minor）
`phases.data` は `h3.phases` として metadata に出る。LoRA の前例（`pipeline_h3.py:634-635`「LoRA-off payload is unchanged」）に合わせ、マスクがあるときだけ書く。

### 10. G1 ② の前提（Minor）
既存の CPU 試験台（`tests/test_h3_engine.py:1936-1957`）は、本物の `_denoise` を小さな本物の DiT で回す。注意層で行どうしが混ざるので、m<1 の行があれば m=1 の行の出力も変わる。G1 ② は要素ごとの部品（`dit_input`・`state_velocity`・`masked_row_times`）の試験に限ると書き直す。マスクが None の経路のビット一致（G1 ①）は、この台でそのまま試せる。

### 11. 映像マスクの組み立て（Minor）
`video_mask(clean, per_frame, *, grid, noise)` は雑音を受け取る。しかし雑音は `_denoise` の中で 997-1003 行の後に引かれる。`_denoise(..., video_mask=StreamMask)` の形では外で組めない。
- 直し方: `_denoise` は「行ごとの値と clean」を受け、中で組む、と決める。
- 加えて要確認の点: ComfyUI の `KSampler` は `model_k.noise = noise`（初期雑音）を `scale_latent_inpaint` に渡すはずで、4 本目の乱数は引かない。R1 に該当の行の記載が無いので、E2 の着手前に R1 の担当に確かめてもらう。初期雑音を使い回す形なら、乱数の引き順の議論そのものが要らなくなる。

### 12. G3-0 の価値（Minor）
- Strength 0 は m=1 なのでマスクは None になり、符号化も飛ばす（§2.2 ②）。違いは「波形を読み込むこと」と「進捗の総数」だけで、DiT・復号の経路は `source_audio` 無しと同じコードを通る。そのビット一致は G1 ①（CPU）とペイロードの試験で保証できる。
- 直し方（どちらか）:
  - G3-0 を外す。
  - GPU で確かめる価値があるのは、変更前と変更後の比較のほう。同じ種の I2V（`source_audio` 無し）を main と dev で 1 本ずつ流し、PCM〔音声の生の値〕と映像の一致を見る。これに置き換える。

### 13. G0 の判定（Minor）
- 符号化器と復号器の畳み込みの埋め方で、数サンプルの遅延が出うる。遅延 0 だけの相関では過小に出る。±800 サンプルの範囲で最大の相関を取ると決める。
- 判定は「RMS〔音量の実効値〕±1.5 dB かつ（r ≥ 0.8 または STFT〔短時間の周波数分解〕の振幅の相関 ≥ 0.95）」と、1 行の式で書く。
- ±1.5 dB は妥当。diffusers は bf16 で約 20 dB 小さくなると書いており、精度の誤りはこれで捕まる。
- G3 の「r ≥ 0.99」も同じ扱いにする。AAC には先頭の遅延（priming）があるので、遅延を探してから比べる。

### 14. 担当の割り当て（Minor）
- `_generate_request` などワーカーの試験は `tests/test_h3_engine.py` に 11 か所ある（`test_worker_vram16_mode_key` 1960 行など）。W1 の「既存テスト」はこのファイルで、E2 の担当と重なる。W1 用に `tests/test_h3_worker_a2v.py`（新）を作るか、E2 に回す。
- アプリ側の `a2v` の出所（`source_audio_id`・`strength`・ffprobe の事実）を `_write_metadata` に渡すには、`_finalize`（`pipeline_manager.py:1802-`）の引数が増える。inpaint の前例と同じく `a2v_provenance=` を `run_job` で作る、と書く。
- FE の `api/types.ts` の `GenerateRequest`（170 行）に `source_audio?` が無い。`npm run typecheck` で気づけるが、表に書く。

### 15. 細かい抜け（Minor）
- `test_engine_h3_adapter.py:608` は「`keep_resident` が末尾」を見ている。`source_audio` は `output_path` の前か、`keep_resident` より前に置くと決める。
- H3 の「読まれている」試験（`test_engine_h3_adapter.py:105-107`）は、`_RealBackendH3.generate` の中に `request.source_audio` の字面を求める。強度を `request.source_audio.strength` から読めば満たせる。
- `zero_k_bias` を buffer にするときは永続（`persistent=True`）にする。非永続だと、`load_state_dict(strict=False)` が「余分」として警告だけ出して捨てる（`sft_load.py:90-97`）。その後 meta のまま残り、105 行で止まる。
- 事前検査にも、前例（`pipeline_manager.py:1265-1268`）の `has_audio_stream` を入れる。
- FE の `singleClipNote`（`i18n/strings.ts:1688`・`3310`「長い音声は Chained の long a2v が使えます」）は H3 では誤り。経路が `generate` のときは出さない。
- 文書の置き場所: FE の DEVLOG と API_REFERENCE は `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/` にある。計画の「FE」の定義（`webui\src\`）から読むと誤った場所になる。

### 16. 過剰設計（Minor）
- `source_audio_rules.py` と `engines.source_audio_rules(family)` は、新しいモジュールと取り次ぎ関数が 1 組ずつ増える。前例の `prompt_token_limit`（§1-111）は `RequestRules` の項目 1 つで済ませた（`ltx/request_rules.py:115`、136）。
- 公開する中身のうち、操作パネルが読むのは「あるか」と `strength` だけ。`latents_per_second` と `sample_rate` は使われない。`required_latents` の使い手も計画に無い。
- 直し方: `RequestRules` に `source_audio_strength: SamplingRange | None = None` を足す。値があるときだけ `constraints()` に `"source_audio": {"strength": …}` を出す。こうすれば LTX の constraints は 1 バイトも変わらない（計画の目的は保たれる）。

## 2. 指定された確認点への答え

1. **式**: 音声（inject = clean）では正しい。m=1 なら v_s = 1·v + 0·(…) = v で、`euler_step` に同じ値が渡る（(clean−x)/σ が有限である限り、浮動小数でも完全に一致）。m=0 なら更新は x + (σ−σ')(clean−x)/σ = clean + (σ'/σ)(x−clean) で正しい。σ が 0 で割られることは無い。`sigma_schedule`（`scheduler.py:39-41`）は評価する σ の最小が shift·(1/n)/(1+(shift−1)/n) > 0 で、0 は末尾の行き先にしか現れない。符号は `x0 = x + σv`、`euler_step = x + (σ−σ')v`（16-20 行、44-52 行）で計画と合う。x0̂ = x̃ + σ·m·v は「雑音の強さ mσ の行」の正しい推定にもなっている。映像だけ #1 の項が足りない。出力を clean と混ぜるのは ComfyUI と同じ（R1 §3.2）。
2. **None の経路**: 1020・1038-1041・1046 は「None ならそのまま返す」で本当に何もしない処理にできる。1016 は #8 の書き方の注意だけ。乱数は、映像のマスクが無ければ今日と同じ順で引かれる。進捗の総数を変えても、`_progress_frac` は `encode` を常に 0.06 に写す（`ltx/adapter.py:523-524`）。A2V でないジョブの進捗の割合は変わらない。注意点は `entry["mask"]` だけ（#9）。
3. **符号化器**: 計画の部品の一覧は R1 §1 と合う。矛盾は無い。統計量は、復号器と同じく `__metadata__` から読む（`audio_vae.py:27-30`、221-226）。そのため key map で統計量のテンソルを捨てても、`encode` の正規化はできる（計画に一言添えると誤読が減る）。「134 本がちょうど 1 回ずつ」は実装できる。重複は `load_streaming` が止める（79-80 行）。`LoadReport.loaded == 134` と、`skipped` が「logs_proj 2 本＋復号器＋統計量 2 本」と一致することを確かめればよい。余分なキーは警告だけなので、`skipped` も確かめる必要がある。
4. **PyAV**: #4 のとおり。
5. **アプリ層**: 実現できる。`LTXRunner.generate`（806-845）は引数を名前つきで渡すので、`H3Runner.generate` を上書きして `source_audio_path` を足すのが素直。模擬バックエンドの子クラスにしても、`is_mock`（1108 行の `isinstance`）は保たれる。同一性を見ている試験は `test_engine_h3_adapter.py:271` だけ（`test_ltx25_adapter.py:3388-3389` は LTX 側なので影響しない）。`_write_metadata`（1855-）、`h3` の扱い（1959-1960）、inpaint の前例（1969-1973）は計画の記述どおり。`_h3_block`（`h3/adapter.py:365-370`）は `done.h3` を平らに混ぜるので、`a2v` は `outcome.h3["a2v"]` に来る。取り出す方針と合う。
6. **API**: `validate_for_family` は `source_audio` に触れない。H3 の分類は `HONOURED_FIELDS` に足せば揃う。LTX は #2 と #3。Gradio は知らない機能名も、知らない constraints のキーも捨てる（`gradio_ui/feature_scope.py:77-86`、`gradio_ui/constraints.py:67-90`）。影響なし。
7. **操作パネル**: 規約は `GET /models` → `useBaseModels` → `baseModels.activeConstraints` → `SingleScreen` の `constraints`（`AppShell.tsx:1705`）で届く。`generationConstraints.ts` の `resolveGenerationConstraints` に 1 行足せば足りる。抜けている層は模擬サーバーだけ（#5）。`AppShell.tsx` の呼び出し元は 1540 行と 1706 行（計画の「未確認」は確認済み）。
8. **長さ**: `constants.py:34-36`、`90-92`、`media_out.py:154-156` と計画は合う。n≡1 で音声がちょうど映像の長さのときを例で確かめた。F=22 では keep = 29,333、A = 37、37×800 = 29,600。符号化すると 37 個で足り、最後の潜在に 267 サンプルの 0 が混ざる。境目の 1 ずれは無い。許容の食い違いは #6。
9. **番号**: VERIFICATION_LOG の最後は §177（18403 行）なので、次は §178 で正しい。CLOSED の最後は 3-254。台帳 §3 に 3-255 と 3-256 が残っているので、次は 3-257 で正しい。FE の DEVLOG の最後は §148（5493 行）なので、次は §149 で正しい（置き場所は #15）。
10. **門**: #12 と #13。

## 3. 確かめた主なファイル（いずれも `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\` 以下）
`engine_h3/scheduler.py`、`engine_h3/pipeline_h3.py`（150-280、560-750、900-1150）、`engine_h3/audio_vae.py`、`engine_h3/sft_load.py`、`engine_h3/media_out.py`、`engine_h3/worker.py`、`engine_h3/dit.py`、`engine_h3/constants.py`、`.venv-engine-h3/Lib/site-packages/av/audio/resampler.py`、`services/engines/__init__.py`、`services/engines/h3/adapter.py`、`services/engines/ltx/adapter.py`、`services/engines/ltx25/adapter.py`、`services/engines/ltx/request_rules.py`、`services/pipeline_manager.py`、`api/generate.py`、`api/models.py`、`gradio_ui/*.py`、各 tests、FE の `shell/featureScope.ts`・`shell/generationConstraints.ts`・`modes/single/SingleScreen.tsx`・`modes/single/useGenerationForm.ts`・`api/types.ts`・`bridge/mockBridge.ts`。
