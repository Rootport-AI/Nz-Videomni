# R4 実装差分のレビュー（弾 1〜2・エンジン側）— MiniMax H3 の A2V

- 対象: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`（ブランチ dev・未コミット）の `engine_h3/` の差分、新規 `engine_h3/denoise_mask.py`・`engine_h3/source_audio.py`・`tests/test_h3_denoise_mask.py`・`tests/test_h3_audio_encoder.py`・`tests/test_h3_worker_a2v.py`・`scripts/h3_a2v_g0_roundtrip.py`、`tests/test_h3_engine.py` の差分。
- 照合先: 計画 `cosmic-giggling-bubble.md` §2.1〜§2.4・§4 G0/G1、R1、DiffSynth の写し（`scratchpad\h3a2v\src\ds_diffsynth_models_minimax_h3_audio_vae.py`）。
- 読み取りのみ。CPU の試験だけ流した: `scripts/run_h3_pytest.py` で `tests/test_h3_denoise_mask.py`・`tests/test_h3_worker_a2v.py`・`tests/test_h3_audio_encoder.py`（実重みの試験を含む）と、`tests/test_h3_engine.py -k "a2v or mask or time_rows or golden or source_audio"` は全部通った。G0 の記録 `outputs/h3_a2v_g0/g0.json` は合格（loaded 134、meta に残った重み 0、RMS の比 +0.02 dB、r=0.979（遅延 0）、STFT の相関 0.993、潜在 [2, 32, 156]）。

## 指摘の一覧（重い順）

| # | 重さ | 要旨 | 場所 |
|---|---|---|---|
| 1 | Minor | A2V でないジョブの `decode` の段にも `audio_output: "decoded"` が増える（metadata の形が変わる） | `engine_h3/pipeline_h3.py:1318` |
| 2 | Minor | ワーカーの説明文が `audio_encode` を進捗の段の名前として書いている（実際は `encode` の 1 ステップ）。`encode` の総数の説明も古い | `engine_h3/worker.py:68-72` |
| 3 | Minor | `encode` の総数を 2 か所で別々に計算している | `engine_h3/pipeline_h3.py:674` と `:782` |
| 4 | Minor | 改行コードがファイルごとにばらばら（もとは CRLF だった 3 本が LF になり、新規ファイルも LF と CRLF が混在） | 下記 |
| 5 | Minor | 試験の代役の説明文が古い（「ほかの担当のモジュールなので、まだ無くてよい」） | `tests/test_h3_engine.py`（`_a2v_fakes` の説明文） |
| 6 | Minor | G0 の台本が検証用のファイルを `outputs/` に書く（保存領域の方針との関係を確認） | `scripts/h3_a2v_g0_roundtrip.py:45` |
| 7 | 所見 | 計画の担当表に無いファイルの変更と、T1 の試験がまだ変わっていないこと | `git status` |

Critical と Major に当たるものは見つからなかった。確認の観点 1〜7 は、上の Minor を除けばどれも計画どおり。

---

## 観点ごとの確認結果

### 1. マスクが無いときにビットが一致すること

- `_denoise`（`pipeline_h3.py:1046-1225`）: 乱数を引く順序（`:1143-1147`）は変わっていない。マスクを組むのは引いた後で（`:1151-1162`）、`audio_fix`／`video_fix` が None なら何もしない。`dit_input`・`state_velocity`・`finish` は、マスクが None なら受け取ったオブジェクトをそのまま返す（`denoise_mask.py:122-123`、`133-134`、`145-146`。試験は `is` で確かめている）。新しいテンソルの演算、dtype や device の変更、乱数の引き出しは無い。足されたのは `denoise_mask` の import だけ。
- `time_rows`（`:295-320`）: キーワード引数が None なら、今日と同じ順序で同じ演算をする。条件行の代入は今日どおり最後。試験 G1 ⑦（`test_time_rows_without_masks_is_the_function_of_before`）は、凍結した写しと `torch.equal` で比べ、dtype も確かめている。
- `generate`: 新しく呼ぶのは `audio_latents(...)`（`:654`、Python の整数計算だけ）と、dict の操作だけ。`_encode` の総数は、A2V でなければ今日と同じ値になる（`3 + int(has_cond)`）。`_encode_image` は `(4, encode_total)` で呼ばれ、I2V では `encode_total` が 4 なので、進捗は今日と同じ `("encode", 4, 4)` になる。アダプタは `encode` を常に 0.06 に写すので、進捗の割合にも影響しない。
- `_decode`: `source_wave` が None のときの音声 VAE の経路は、字下げが 1 段深くなっただけでコードは同じ（`:1304-1317`）。ただし指摘 1 のとおり、metadata のキーが 1 つ増える。
- **指摘 1（Minor）**: `entry["audio_output"] = "decoded"`（`:1318`）を、A2V でないジョブにも書いている。T2V／I2V の `done.h3.phases.decode` にキーが 1 つ増え、そのまま metadata に入る。計画 §2.2 は `entry["mask"]` について「A2V でないジョブの metadata を変えない」と決めており、`_decode` で記録するのは `source_wave` があるときだけと読める。この値を読む箇所は app にも試験の golden にも無い（`grep` で確認）ので実害は無いが、`a2v["audio_output"]` は `source is not None` のときしか使わない。直し方: `:1318` を消し、`generate` 側で `a2v["audio_output"] = entry.get("audio_output", "decoded")` とする。`test_a2v_tiny_job_audio_output_switch` の `plain[...]["decode"]["audio_output"] == "decoded"` は「キーが無い」ことを確かめる形に変える。

### 2. マスクの式（`denoise_mask.py`）

- `dit_input` は `m·x + (1−m)·inject`（`:124`）。`state_velocity` は x̃ → x0̂ = x̃ + σ·(m·v) → x0_b = m·x0̂ + (1−m)·clean → `(x0_b − x)/σ` を展開せずにそのまま計算している（`:135-140`）。`finish` は `torch.where(zero, clean, x)`（`:147`）。`masked_row_times` は m<1 の行が `min(1 − m·σ, cap)`、m=1 の行が流れの t（`:150-155`）。上限は映像 0.999、音声 1.0（`:40-41`）。どれも計画 §2.1 と一致する。
- 行の並び: 音声は `values.repeat(channels)` で左チャンネル（ch0）の後に右（ch1）、映像は `repeat_interleave(grid)`。試験は `audio_to_rows`／`patchify_video` に m を通して突き合わせている（`test_h3_denoise_mask.py:50-77`）。
- 映像の inject: 初期雑音（`noise=video`。`pipeline_h3.py:1162`）を組み立てのときに 1 回だけ使う。この時点の `video` は引いたばかりの雑音で、`euler_step` はその場で書き換えない（`scheduler.py:46-53`）。新しい乱数は引かない。
- 数値はすべて fp32。σ で割る箇所が評価する σ_i は `i < steps` なので常に正（`sigma_schedule` の linspace は端の 0 を落としている）。
- 全部 1 のマスクは組み立ての時点で None にしている（`:88-89`、`:110-111`）。値の検査（個数・有限・0〜1）もある。

### 3. 符号化器（`audio_vae.py`）

- 構造は DiffSynth の `Snake`／`ResidualUnit`／`EncoderBlock`／`Encoder`／`GeGluMlp`／`CausalAttention`／`AttnProjection` と一つずつ対応する。Snake は α を生のまま使い（`[1,C,1]`、exp は取らない）、β=α の分岐を `1/(α+1e-9)` で再現している。DAC のストライドは 2・4・4・5・5、最後の畳み込みは d_latent=2048 に出す。注意はヘッド方向に平均してから `adaptive_avg_pool1d(32)` にかけ、因果マスクをかける。`encode` は右側を 800 の倍数まで 0 で埋め、平均（mean）だけを使い、`(z−mean)/std` で正規化し、fp32 で通す。統計量は `__metadata__` から読む（`from_file`）。
- `zero_k_bias` は永続の buffer（`register_buffer(..., persistent=True)`）。実重みの試験では meta のまま残る重みが 0（`test_real_encoder_loads_134_tensors...`）。state_dict の名前と形がファイルと完全に一致することも試験で確かめている。
- key map: 1 つの表（`_FILE_PARTS`＋`_STAT_KEYS`）から 2 つの map を作っている。`audio_vae_key_map` の振る舞いは前と同じ（落とすもの: `encoder.`・`pre_block.`・`mean_proj.`・`logs_proj.`・統計量。表に無い名前は素通し）。`H3AudioVAE` のクラス本体には差分が無い（差分は先頭の説明文・key map・末尾の追加だけ）。
- ComfyUI のコードは含まれていない。ComfyUI に由来するのは、説明文に出てくる定数の名前だけ。

### 4. `source_audio.py`

- 出し切り: 読み込みのループの後で必ず `resample(None)` を呼ぶ（`:69`）。途中で読み込みを止めたときも呼ぶ。None の要素は捨てている（`:51`）。素通しになる入力（32 kHz・fltp）も試験で確かめている（`:201-242`）。
- チャンネル: 1 チャンネルは左右に複製、3 チャンネル以上は先頭の 2 つ（`:109-117`）。長さ: 足りないのが 32 サンプル以内なら 0 で埋め、それ以上なら日本語の `ValueError`（`:121-130`）。`read_samples` に達したらループを止める（`:67-68`）。途中で形式が変わる入力は、日本語の理由をつけて包み直す。
- 後始末: `with container:` で閉じているので、どの経路でもコンテナは閉じる。開けないファイルは日本語の `ValueError` にしている。
- `read_samples = A·800 + 32000`、`keep_samples = round(F·32000/24)` で、`ceil(N/800) ≥ A` が成り立つ（F·5/3 の小数部が 0・1/3・2/3 のどの場合も確かめた。0 埋めの許容 32 サンプルを入れても成り立つ）。そのため、`_encode_audio` の形の検査（`:856`）に正しい入力で引っかかることは無い。

### 5. メモリの扱い

- `_encode_audio`（`:834-864`）は、ジョブごとに符号化器を組み、`finally` で `del enc` と `_free_cuda()` を行う。これは DiT を組む前（`_denoise` より前）。常駐させるものは何も増えていない。
- `_decode` に `source_wave` を渡したときは、`H3AudioVAE` を組まず、常駐している物にも触らない（`:1298-1301`。試験では組み立ての回数を数えている）。常駐していた物は、`generate` の `finally` で今日どおり解放される。
- 波形（CPU、数百 KB）と clean（device、2×32×A）は、ノイズ除去の間も保持される。どちらも小さい。

### 6. ワーカー

- `_source_audio` は、オブジェクトであること、`path` が空でない文字列であること、`strength` が有限の数値であること（bool は断る）を検査する。省略や null は 1.0 として扱う。範囲とファイルの有無は `validate_request` に任せている。知らない内側のキーは `source_audio.<key>` としてログに 1 回出す。`done` は `result` をそのまま出すので、`h3.a2v` も素通しになる。セルフテストの `--source-audio`／`--audio-strength` は、指定したときだけキーを末尾に足す。
- golden の試験（`test_golden_generate_message_keys`）は、A2V でないセルフテストのメッセージのキーの順序を、`GOLDEN_GENERATE_KEYS` で今も完全一致で確かめている。`_GENERATE_FIELDS` には任意のキーとして `source_audio` を足している。
- **指摘 2（Minor）**: `worker.py:68` の `stage: encode / [audio_encode] / stage1_denoise / decode` は誤り。`audio_encode` は `done.h3.phases` の段の名前で、進捗の段として出るのは `encode` だけ（アダプタの説明文 `services/engines/h3/adapter.py:37-40` が正しい）。続く `:70-72` の「key frame があれば 4、無ければ 3」も、A2V で 1 増えることを書いていない。直し方: `:68` を `stage: encode / stage1_denoise / decode` に戻し、`:70-72` を「``encode`` は 3 ステップ、key frame があれば +1、音声を符号化するなら +1」とする。

### 7. 試験（G1 ①〜⑧）

| 門 | 試験 | 判定 |
|---|---|---|
| ① | `test_denoise_all_ones_mask_is_the_unmasked_run_bit_for_bit`（本物の `_denoise` を小さな DiT で走らせ `torch.equal` で比べる）、`test_a2v_tiny_job_audio_output_switch`（strength 0 で `torch.equal`） | 合う |
| ② | `test_dit_input_keeps_m1_rows_and_injects_m0_rows`、`test_masked_row_times_per_row_and_caps` | 合う |
| ③ | `test_denoise_kept_rows_feed_inject_and_end_clean`（音声は毎ステップの入力と最終値を `torch.equal` で、映像の inject は allclose（1e-6）で比べる）、`test_m0_rows_follow_return_to_timestep_and_finish_exactly` | 合う（映像の inject は試験の中で別に計算した値と比べるので allclose が妥当。計画もビット一致は求めていない） |
| ④ | `test_time_rows_unique_set_with_masks`（M=6）、`test_denoise_with_more_than_three_times_passes_adaln_and_vram16`（M>3、VRAM 16GB mode の on と off を `torch.equal` で比べる） | 合う |
| ⑤ | `test_state_velocity_matches_the_definition_audio_and_video`（音声と映像の両方を、定義どおりの式と展開形の両方に対して float64 で比べる。inject−clean の項が効いていることも確かめる） | 合う |
| ⑥ | `test_video_mask_rows_are_t_then_h_then_w` | 合う |
| ⑦ | `test_time_rows_without_masks_is_the_function_of_before` | 合う |
| ⑧ | `tests/test_h3_audio_encoder.py` の 44.1 kHz の出し切り、素通し（フレームとファイルの両方）、1 ミリ秒の 0 埋めと拒否、上限、3 チャンネル、形式の変化 | 合う |

代役のせいで中身の無い試験になっているものは無い。`_a2v_stage_job` は TE・VAE・ローダーを代役に置き換えるが、それで確かめるのは進捗・段・引数の受け渡しだけで、式は確かめていない。式は小さな本物の DiT の試験が受け持っている。

- **指摘 5（Minor）**: `_a2v_fakes` の説明文に「another coder's modules: they need not exist for these tests」とあるが、モジュールはもう存在する。直し方: 「試験を速く・重み無しで回すための代役」に書き換える。

### 8. 書き方と後始末

- 利用者が目にするメッセージ（`validate_request`、`source_audio`、ワーカー）は日本語。`denoise_mask` の `ValueError` と `_encode_audio` の `RuntimeError` は英語だが、前段の検査があるので通常は起きない内部の検査で、周りの `image encode: ...` と同じ書き方に揃っている。デバッグ用の print は残っていない（G0 の台本の print は出力として意図したもの）。
- **指摘 3（Minor）**: `encode` の総数を `generate`（`:674` の `encode_total`）と `_encode`（`:782`）で別々に計算している。式は同じだが、正本が 2 つある。直し方: `_encode` に `total` を引数で渡し（`_encode_image`／`_encode_audio` と同じ形）、`:782` の計算を消す。
- **指摘 4（Minor）**: 作業ツリーの改行コード。`engine_h3/` の他のファイルは CRLF（`git ls-files --eol` で 15 本）なのに、`engine_h3/audio_vae.py`・`engine_h3/pipeline_h3.py`・`tests/test_h3_engine.py` は全行が LF になった（変更前は CRLF。`git diff` が「LF will be replaced by CRLF」と警告している）。新規ファイルも、`denoise_mask.py`・`test_h3_denoise_mask.py`・`test_h3_worker_a2v.py` は LF、`source_audio.py`・`test_h3_audio_encoder.py`・`h3_a2v_g0_roundtrip.py` は CRLF と混在している。`core.autocrlf=true` でインデックスには LF で入るので、リポジトリの中身には影響しない。作業ツリーで揃えたいなら、コミットの後に `git checkout -- <file>` を実行すれば CRLF に戻る（監督の判断。読み取りのみのレビューなので実行していない）。
- **指摘 6（Minor）**: G0 の台本は `outputs/h3_a2v_g0/` に wav と json を書く。保存領域の方針（Outputs は利用者の成果物）と照らして、置き場がこれでよいかを監督が確認する。変えるなら `--out` の引数を足し、既定をスクラッチや検証用のフォルダにする。
- **所見 7**: `git status` には、計画 §3 の担当表に無い次のファイルの変更が出ている。`api/errors.py`（`source_audio_too_short` の説明文と文言を、単発にも当てはまる形に直したもの。A1 の範囲の近く）、`Docs/PENDING_TASKS.md`（3-256 の起票。計画 §5 が「そのまま」と書いている項目）、操作パネルの試験 `GenerationForm.accordion.test.tsx`・`GenerationForm.referenceVideo.test.tsx`・`AppShell.minimaxH3.test.tsx`・`useGenerationForm.h3.test.ts`（表では `useGenerationForm.test.ts`）・新規 `SingleScreen.sourceAudio.test.tsx`。また、T1 の担当ファイル（`tests/test_engine_h3_adapter.py`・`tests/test_ltx_adapter_audit.py`・`tests/test_ltx25_adapter.py`・`tests/test_ltx25_api_guard.py`・`tests/test_engine_dispatch.py`・新規 `tests/test_h3_a2v_api.py`）にはまだ変更が無い。完全一致の期待値を更新しないと、G2 で失敗する見込み。エンジン側の担当（弾 1〜2）のファイルの外への変更は見つからなかった。
