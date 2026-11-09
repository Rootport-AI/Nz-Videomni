# MiniMax H3 の連結生成にクリップごとの A2V を足した門

MiniMax H3 の連結生成（Chained。複数のクリップを潜在表現の段階で継ぎ目なくつなぐ生成）に、クリップごとの A2V（音声から映像を作る機能。要求の `clips[i].source_audio`）を足した実装を、GPU で確かめる門 GA0〜GA3・GA3T の台本と記録の置き場です（台帳 `PENDING_TASKS.md` §1-120）。門の条件と基準の正本は計画の §6.12、契約の形は計画の §3 と §6.6 です。結果の全部は各フォルダの `result.json` と `summary.json` に残し、この説明書には合否と要点の値だけを書きます。

## フォルダの中身

| ファイル・フォルダ | 役目 |
|---|---|
| `preconditions.py` | 回す前と後の点検（git の状態・GPU の使用量・python のプロセス・ワーカーの残り・AviUtl2 の有無）。`preconditions_before.txt`／`preconditions_after.txt` に書く。`h3-vsf-chain-2026-11-09` の同名の台本の写しで、動きは同じ |
| `h3_chain_a2v_gate.py` | 門 GA0〜GA3 を、別のポートに起こした実際のバックエンドへ HTTP で直接送る台本。送る本文と期待値を見るだけの `--dry-run` もある |
| `ga0\`〜`ga3\`・`ga3t\` | 各門の `result.json`・`request.json`（送った本文）・`metadata.json`（サーバーが書いたものの写し）・ワーカーのログの抜き書き（`worker_log_excerpt.txt`） |
| `summary.json` | 台本が回した全ジョブの条件・秒・指紋（sha256）・合否 |
| `_rt\` | 一時サーバーの設定・状態・ログ・出力・アップロードの置き場。動画（`output.mp4`）はここの `outputs\<job_id>\` に置いたままにし、保管庫の各門のフォルダには写さない。**保管しない**（終わったら消してよい。git に入れない） |

## 門と基準

共通の条件（GA0〜GA2）: FL2VA の重み（登録名 `default`）・640×384・seed 42・sdpa・20 ステップ・shift 12/3・動画への生成条件の書き込みなし（`embed_mp4_metadata=false`）・連結 3 クリップ・GPU に残すブロック 8・LoRA なし・切り抜きなし。プロンプトは過去の門 G-C0 と同じ組み立て（`h3-chain-2026-10-29\prompts_c.json`。クリップ 0 は共通の文だけ、クリップ 1・2 は追記の文）。

音声は `uploads\h3-a2v-development\ref-voice-patra001.wav`（30.9 秒）を一時サーバーの `POST /api/v1/upload/audio` で 1 回だけ上げ、返った `audio_id` を GA1〜GA3 の全クリップで使います。各クリップの `offset_frames` は、そのクリップの連結の時間軸上の開始フレーム S_i（S_0 = 0、S_i = S_{i-1} + F_{i-1} − 22）です。

どれかの門が合格しなければ（要求の拒否・ジョブの失敗・基準の不合格）、台本はそこで止まり、次の門へは進みません。拒否されたときは、返ってきた本文の全文が `result.json` の `submit` に残ります。

### GA0: 不変（`--gate GA0`）

G-C0 の要求本文（`h3-chain-gates-2026-10-29\g_c0.json` の `request_body`）をそのまま送ります。124 フレームのクリップ 3 本。

| 確かめること | 期待 | 結果 |
|---|---|---|
| 動画の指紋 | G-C0 の `4a98be0884ae7129e34f58578a3fe7ca3117ca5a764545cbac404e12a70cbec5`（`--expect-sha` を付けたときはその値）と全桁一致 | |
| 各クリップの潜在の指紋（`metadata.h3.clips[i].latents_sha256`） | `g_c0.json` の値と 3 本とも全桁一致 | |
| 条件（`metadata.json`） | `backend`＝`minimax-h3`・FL2VA・sdpa・20 ステップ・shift 12/3・種 42／43／44・クリップ 124×3・総フレーム 328・640×384・プロンプトの指紋が G-C0 と同じ・LoRA と切り抜きとネガティブなし・書き込みなし | |
| A2V の記録 | `metadata.a2v` も `metadata.h3.clips[i].a2v` も無い | |
| サーバーが受けた本文 | 合否に使う欄が G-C0 の本文と同じ | |

### GA1: Strength 0 は効き目なし（`--gate GA1`）

GA0 と同じ本文の各クリップに `source_audio = {audio_id, strength: 0.0, offset_frames: S_i}`（S = 0・102・204）を足します。

| 確かめること | 期待 | 結果 |
|---|---|---|
| 動画の指紋 | GA0 と同じ基準（G-C0 の値）と全桁一致 | |
| 各クリップの潜在の指紋 | `g_c0.json` の値と 3 本とも全桁一致 | |
| `metadata.a2v.clips[i].encoded_samples` | 3 本とも 0 | |
| `metadata.a2v.clips[i]` の形 | 3 本ともあり、`strength` 0.0・`offset_frames` 0／102／204・`mask_value` 1.0・`start_sample` 0／136000／272000・`required_samples` 165333・`audio_output` "decoded"・`source_audio_id` が上げた音声 | |
| 条件 | GA0 の条件の行と同じ（プロンプト・種・本文の突き合わせは除く） | |

同じ回の GA0 の指紋との一致も `result.json` に記録します（`sha256_equals_ga0_this_run`・`latents_equal_ga0_this_run`）。

### GA2: Strength 1.0・継ぎ目が格子に乗らない形（`--gate GA2`）

クリップは [124, 141, 124]・全クリップ `strength: 1.0`・`offset_frames` 0・102・221。継ぎ目 S_1 = 102 は 3 の倍数で音声の潜在の格子に乗り、S_2 = 221 は乗りません。

| 確かめること | 期待 | 結果 |
|---|---|---|
| ジョブ | 完了する | |
| `metadata.h3.clips[i].head_equals_prev_tail` | クリップ 1・2 で真 | |
| `metadata.a2v.clips[i].mask_value` | 3 本とも 0 | |
| `metadata.a2v.clips[i].audio_output` | 3 本とも "decoded" | |
| 総フレーム（`metadata.chain.total_frames`） | 345 | |
| `metadata.a2v.clips[i].start_sample` | 0・136000・294400（台本の計算 `round(O×5/3)×800` と、製品の `h3_chain_math.clip_audio_start_sample` の値の両方と一致） | |
| `metadata.a2v.clips[i]` の形 | `required_samples` 165333／188000／165333 ほか、GA1 と同じ鍵 | |

### GA3: Strength 0.5・目視用（`--gate GA3`）

209 フレームのクリップ 3 本（583 フレーム＝24.3 秒）・1344×768・sage・`strength: 0.5`・`offset_frames` 0・187・374・20 ステップ・shift 12/3。最後のクリップが使う音声の終わりは 24.283 秒で、音声の長さ（30.9 秒）に収まります。プロンプトは共通の文だけで、台詞のタグ（`<d>`）は入れません（既定の文は台本の `GA3_PROMPT_DEFAULT`。`--ga3-prompt` で替えられる）。対照の生成はありません。

台本が見るのは完走と記録の形だけです。口の合い方と継ぎ目の合否はオーナーの目視で付けます。

| 確かめること・記録すること | 期待 | 結果 |
|---|---|---|
| ジョブ | 完了する | |
| 条件 | sage・1344×768・クリップ 209×3・総フレーム 583・20 ステップ・shift 12/3 | |
| `metadata.a2v.clips[i]` の形 | 3 本ともあり、`mask_value` 0.5・`start_sample` 0／249600／498400・`required_samples` 278667・`audio_output` "decoded" | |
| `head_equals_prev_tail` | クリップ 1・2 で真 | |
| VRAM の山（記録だけ） | `metadata.json` の予約の山・段ごとの山（音声の符号化の段 `audio_encode` を含む）、`nvidia-smi` の使用量の最大（3 秒おき） | |
| 所要（記録だけ） | 生成の秒数・クリップごと段ごとの秒数 | |
| 主記憶（記録だけ） | `metadata.h3` の常駐の山（`rss_peak_gib`）とコミットの山（`commit_peak_gib`） | |
| 目視 | オーナー | |

### GA3T: GA3 と同じ条件で Turbo 4 ステップ・目視用（`--gate GA3T`）

GA3 と同じ本文（プロンプト・1344×768・sage・クリップ 209×3・`strength: 0.5`・`offset_frames` 0・187・374・shift 12/3）に、Turbo LoRA と 4 ステップを足します。LoRA の指定は前例 `h3-vsf-2026-11-09\h3_vsf_gate.py` の T 系列（Turbo 4 ステップ）の要求本文の写しで、`loras: [{"name": "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16", "strength": 0.8}]`・`num_inference_steps: 4` です（名前は `models\MiniMaxH3\LoRA\` のファイル名から拡張子を除いたもの）。GA3 の本文との違いはこの 2 つの欄だけです。結果は `ga3t\` に書きます。

| 確かめること・記録すること | 期待 | 結果 |
|---|---|---|
| ジョブ | 完了する | |
| 条件 | GA3 の条件の行と同じ。ただしステップは 4（shift 12/3 はそのまま） | |
| LoRA | `metadata.request.loras` が Turbo 0.8 の 1 本だけ・`metadata.h3.loras` に `minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors` があり `pairs_applied` が 0 より大きい | |
| `metadata.a2v.clips[i]` の形・`head_equals_prev_tail` | GA3 と同じ | |
| VRAM の山・所要・主記憶（記録だけ） | GA3 と同じ鍵 | |
| 目視 | オーナー（GA3 の 20 ステップと見比べる） | |

## 台本の使い方

すべてリポジトリ直下（`S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`）から回します。GPU を使う前に、その都度オーナーの了承を得ます。AviUtl2 は閉じておきます。

1. 送る本文と期待値を見る（GPU もサーバーも使わない）: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-chain-a2v-2026-11-09\h3_chain_a2v_gate.py --gate all --dry-run`
   GA0 の `body_equals_g_c0_request_body` と、各門の `offsets_equal_plan`・`math_crosscheck.equals_script` が真であることを確かめます。
2. 前の点検: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-chain-a2v-2026-11-09\preconditions.py before`
   終了コード 2 は AviUtl2 が動いている、3 はワーカーのプロセスが残っている、という意味です（台本は何も止めません）。
3. 門を回す: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-chain-a2v-2026-11-09\h3_chain_a2v_gate.py --gate all`
   1 つずつ回すときは `--gate GA0` のように選びます。続けて同じサーバーを使うときは `--keep-server` を付けます。
4. 後の点検: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-chain-a2v-2026-11-09\preconditions.py after`

`h3_chain_a2v_gate.py` の引数:

| 引数 | 既定 | 意味 |
|---|---|---|
| `--gate` | （必須） | `GA0`・`GA1`・`GA2`・`GA3`・`GA3T`・`all`（all は GA0 → GA1 → GA2 → GA3 → GA3T を 1 つのサーバーで続けて回し、不合格で止まる） |
| `--label` | `run` | 結果の JSON に残す名前（英数字と `.`・`_`・`-` だけ） |
| `--expect-sha` | G-C0 の値 | GA0 と GA1 で比べる動画の指紋 |
| `--audio` | `uploads\h3-a2v-development\ref-voice-patra001.wav` | GA1〜GA3 で上げる音声 |
| `--ga3-prompt` | 台本の `GA3_PROMPT_DEFAULT` | GA3 と GA3T の共通のプロンプト |
| `--port` | 18699 | 一時サーバーのポート。普段のサーバー（18620）とは別 |
| `--keep-server` | 切 | 終わりに H3 を降ろさず、サーバーも止めない。次の回は同じポートのサーバーを使い回す（一時の設定ファイルの名前で起こしたサーバーだけ。別のサーバーが応答していたら止まる） |
| `--load-timeout-sec` | 1200 | H3 の読み込みを待つ上限 |
| `--job-timeout-sec` | 7200 | 1 本のジョブを待つ上限。超えたらそこで止まる |
| `--dry-run` | 切 | 送る本文と期待値を表示して終わる |

台本の動き:

- `config.yaml` は読むだけです。ポート・ログの置き場・出力の置き場・アップロードの置き場・状態ファイルの 5 つを `_rt\` に向け、`vram.block_swap: true` と `vram.block_swap_blocks_on_gpu: 8` を書き込んだ一時の設定ファイル（`_rt\config.h3chaina2v.yaml`）と、MiniMax H3 を選んだ状態ファイル（`_rt\state.h3chaina2v.json`）を作り、`.venv\Scripts\python.exe main.py --config _rt\config.h3chaina2v.yaml` で起こします。起動の環境は普段の起動（`run.ps1`）に合わせ、`tools\uv` と `tools\ffmpeg\bin` を PATH の先頭に足します。オーナーの `state.json`・`logs\`・`outputs\`・`uploads\` には書きません。
- 要求は MCP サーバーを通さず、`POST /api/v1/generate/chain` へ直接送ります。MCP の `submit_chain` のクリップの引数には `source_audio` が無いためです（計画 §6.11）。G-C0 も HTTP で直接送っていたので、GA0 は G-C0 と同じ本文になります。`config.yaml` に `server.api_key` があれば `Authorization` の見出しを付けます。
- H3 の読み込みは `POST /api/v1/pipeline/load {"base_model": "MiniMaxH3", "models": {"transformer": "default"}}` を 1 回だけ呼び、あとは `GET /api/v1/status` を 10 秒おきに見て待ちます。ジョブは `GET /api/v1/jobs/{id}` を 10 秒おきに見ます。
- ワーカーのログは `_rt\logs\h3_worker.log` の、ジョブの前後の差分を読みます。
- 終了コード: 0＝回した門がすべて合格（GA3 は記録の形の基準が合格）、3＝どれかが不合格か失敗（そこで止まる）、1＝準備の段で止まった。最後に門ごとの合否と落ちた基準を短く表示します。
- サーバーの止め方: H3 を降ろし（`POST /api/v1/pipeline/unload`）、サーバーを止め、止まらなければプロセスの木ごと強制終了します。最後にサーバーの子孫（ワーカー）が残っていれば、それも止めます。

## 実行の記録

（まだ回していない）
