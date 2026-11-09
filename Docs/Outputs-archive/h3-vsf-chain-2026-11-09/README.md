# MiniMax H3 のネガティブプロンプト（VSF）を連結生成へつなぐ門

MiniMax H3 のネガティブプロンプトを VSF（Value Sign Flip。負のプロンプトの行の値の符号を反転し、1 回の注意の計算で負の要素を遠ざける方式）で連結生成（Chained。複数のクリップを潜在表現の段階で継ぎ目なくつなぐ生成）と Retake（入力動画の指定した区間だけを描き直す機能）の経路へつないだことを、機械で確かめる門 GC0〜GC3 の台本と記録の置き場です（台帳 `PENDING_TASKS.md` §1-127 の続き）。単発の生成へ載せた第 1 弾の門と目視の実験は `Docs\Outputs-archive\h3-vsf-2026-11-09\` にあります。結果の全部は各フォルダの `result.json` と `summary.json` に残し、この説明書には合否と要点の値だけを書きます（記録簿 `Docs/VERIFICATION_LOG.md` §202.7 にも同じ値があります）。

## フォルダの中身

| ファイル・フォルダ | 役目 |
|---|---|
| `preconditions.py` | 回す前と後の点検（git の状態・GPU の使用量・python のプロセス・ワーカーの残り・AviUtl2 の有無）。`preconditions_before.txt`／`preconditions_after.txt` に書く。第 1 弾の同名の台本の写しで、動きは同じ |
| `preconditions_before.txt` | 前の点検の結果。2 回目（配線の後の回の前。2026-11-09 09:26・`a44ed19`・GPU 743MiB・AviUtl2 とワーカーなし）で上書きされている。`preconditions_after.txt` は作っていない |
| `h3_vsf_chain_gate.py` | 門 GC0〜GC3 を、別のポートに起こした実際のバックエンドへ MCP サーバー経由で送る台本。GPU を使わない自己検査（`--selfcheck`）と、送る引数を見るだけの `--dry-run` もある |
| `gc0\before\`・`gc0\after\` | 門 GC0 の `result.json`・`request.json`（MCP の道具に渡した引数）・`metadata.json`・ワーカーのログの抜き書き（`worker_log_excerpt.txt`）・動画。名前は `--label` の値（配線の前が `before`、後が `after`） |
| `gc1\`・`gc2\`・`gc3\` | 各門の `result.json` と、ジョブごとのフォルダ（`gc1\vsf\`・`gc2\retake\`・`gc3\off\`）の `request.json`・`metadata.json`・ログの抜き書き・動画 |
| `summary.json` | 台本が回した全ジョブの条件・秒・指紋（sha256）・合否。回すたびに、同じ名前のジョブは上書き、他は残る |
| `logs\` | MCP の呼び出しの控え（全文）と MCP サーバーの標準エラー |
| `_rt\` | 一時サーバーの設定・状態・ログ・出力・アップロードと、コンソールの控え（`gc0_before.console.log`・`all_after.console.log`）の置き場。**保管しない**（終わったら消してよい。git に入れない） |

動画（`*.mp4`）は git に入れません。

## 実行の記録

すべて 2026-11-09、開発機で回しました（記録簿 §201.7 と同じ機械）。

| 回 | コード | 時刻 | 門 | 結果 |
|---|---|---|---|---|
| 1 回目（`--mode gc0 --label before`） | 配線の前（dev `f71ec2e`。コードは `33de4d5` と同じ） | 09:00〜09:07 | GC0 | 合格（終了コード 0） |
| 2 回目（`--mode all --label after`） | 配線の後（dev `a44ed19`） | 09:27〜09:43 | GC0・GC1・GC2・GC3 | すべて合格（終了コード 0） |

- 待ちの途中で、MCP の `wait_for_job` が `BACKEND_UNREACHABLE` を返す回がありました（`summary.json` の `wait.samples`。GC0 before で 4 回、GC0 after で 6 回、GC1・GC2 で各 1 回、GC3 で 2 回）。どのジョブも止まらずに完了しました。原因は追っていません。

## 門と合否

### GC0: ネガティブなしの連結生成の回帰（`--mode gc0`）

条件は過去の門 G-C0（`Docs\Outputs-archive\h3-chain-gates-2026-10-29\`。記録簿 `Docs/VERIFICATION_LOG.md` §184.6）と同じです。FL2VA の重み（登録名 `default`）・640×384・124 フレームのクリップ 3 本・プロンプトは `h3-chain-2026-10-29\prompts_c.json` から G-C0 と同じ式で組み立て（クリップ 0 は共通の文だけ、クリップ 1・2 は追記の文）・sdpa・20 ステップ・shift 12/3・VRAM 16GB mode on・先読み on・GPU に残すブロック 8・LoRA なし・seed 42・動画への生成条件の書き込みなし・切り抜きなし。

MCP の `submit_chain` にはステップ数・shift・VRAM 16GB mode の引数が無いので、台本はこの 3 つを送らず、H3 の既定（20・12/3・on）で動いたことを `metadata.json` で確かめます。G-C0 も VRAM 16GB mode は送らずに既定で回していました。

配線の前（`--label before`）と後（`--label after`）の 2 回回して比べます。

| 確かめること | 期待 | before | after |
|---|---|---|---|
| 動画の指紋 | G-C0 の `4a98be0884ae7129e34f58578a3fe7ca3117ca5a764545cbac404e12a70cbec5`（`--expect-sha` を付けたときはその値） | 合格（全桁一致） | 合格（全桁一致） |
| 条件（`metadata.json`） | `backend`＝`minimax-h3`・FL2VA・sdpa・VRAM 16GB mode on・先読み on・20 ステップ・shift 12/3・種 42／43／44・クリップ 3 本・プロンプトの指紋が G-C0 と同じ・LoRA と切り抜きとネガティブなし・書き込みなし | 合格 | 合格 |
| サーバーが受けた本文 | 合否に使う欄（プロンプト・クリップ・画角・ステップ・shift・VRAM 16GB mode など）が G-C0 の本文と同じ | 合格（違いは H3 が無視する `chunked_upsample` だけ） | 合格（違いは同じく `chunked_upsample` だけ） |
| 各クリップの潜在の指紋（記録だけ） | G-C0 の値との一致を記録する | 3 本とも一致 | 3 本とも一致 |
| 生成の秒数・VRAM の予約の山（記録だけ） | — | 406.5 秒・5,382MiB | 384.4 秒・5,382MiB（ログは `neg=off`） |

### GC1: VSF つきの連結生成（`--mode gc1`）

2 クリップ・512×288・56 フレームずつ・seed 42・sdpa・VRAM 16GB mode on・20 ステップ・shift 12/3・LoRA なし・負のプロンプト `blurry, low quality`・倍率 3・`keep_resident=true`。

| 確かめること | 期待 | 合否 |
|---|---|---|
| ジョブ | 完了する | 合格（117.1 秒・VRAM の予約の山 5,240MiB・sha256 `3f0ad33230225b0e0ff6d26db559da267e7e06bf97b7b6fc74d93e2a3ce0771c`） |
| `metadata.h3.clips[0].vsf`・`clips[1].vsf` | `rows`・`scale`・`m_neg`・`m_text` の 4 つがあり、`scale`＝3.0 | 合格（クリップ 0: rows [41, 46]・`m_neg` 0.0043637565・`m_text` 0.0107666263／クリップ 1: rows [54, 59]・`m_neg` 0.0064656506・`m_text` 0.0199875041） |
| `metadata.h3.phases.clips[0].encode` | `te_read_negative` と `negative_tokens` がある | 合格（`negative_tokens` 5・テキストエンコーダーの読み込み 9.8 秒） |
| `metadata.h3.phases.clips[1].encode` | どちらも無い（負のプロンプトの符号化はジョブに 1 回だけ） | 合格 |
| ワーカーのログ | 連結生成の行（`generate_chain:`）に `neg=vsf(scale=3.0)` | 合格 |
| 同じ連結生成を `neg_method="nag"` で送る | 断られる。文に `FEATURE_UNSUPPORTED` と機能名 `nag` と「VSF 方式だけ」 | 合格 |
| `list_models` の MiniMax H3 の `unsupported_features` | `nag` が残っている | 合格 |

### GC2: VSF つきの Retake（`--mode gc2`）

素材は GC0 の出力（無ければ GC1 の出力。`--material` で替えられる）。窓は 73 フレーム・0.5 秒から（GC1 の出力の 90 フレームにも収まる長さ）、画角は素材に合わせて 640×384（GC1 の出力なら 512×288）。負のプロンプト `blurry, low quality`・倍率 3・`keep_resident=true`。のりしろ（22 固定）と音声の強さ（既定 0.5）は渡しません。

| 確かめること | 期待 | 合否 |
|---|---|---|
| ジョブ | 完了する | 合格（素材は GC0 after の出力・640×384・110.6 秒・VRAM の予約の山 5,398MiB・sha256 `5e8b897f6772201931f4a65700d00cbebd6a9000b8e995556ce0b0c9a19bc6a3`） |
| Retake として動いたこと | `metadata.retake` か `metadata.request.retake` がある | 合格（窓 73・頭と尾 22・`audio_strength` 0.5・元の音声あり） |
| `metadata.h3.clips[0].vsf` | 4 つの鍵があり、`scale`＝3.0 | 合格（rows [41, 46]・`m_neg` 0.0049774484・`m_text` 0.0126818049） |
| ワーカーのログ | 連結生成の行に `neg=vsf(scale=3.0)` | 合格 |

### GC3: VSF のジョブの直後に持ち越しが無いこと（`--mode gc3`）

同じサーバーで VSF のジョブ（`keep_resident=true`）を回した直後に、GC0 と同じ要求を送ります。`--mode all` では直前のジョブは GC2 の Retake（これも VSF・`keep_resident=true`）、GC1 の後に単独で回すなら GC1 です（`result.json` の `preceding_vsf_job_this_run` に残ります）。

| 確かめること | 期待 | 合否 |
|---|---|---|
| 動画の指紋 | GC0 と同じ基準（G-C0 の値か `--expect-sha`）。同じ回の GC0 との一致も記録する | 合格（基準と同じ回の GC0 の両方に一致。直前の VSF のジョブは GC2。378.9 秒・VRAM の予約の山 5,382MiB） |
| GC0 と同じ条件の基準 | GC0 の表と同じ | 合格 |
| `metadata.h3.clips[i].vsf` | どのクリップにも無い | 合格（最上位の `h3.vsf` も無い） |
| ワーカーのログ | 連結生成の行に `neg=off` | 合格 |

## 台本の使い方

すべてリポジトリ直下（`S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`）から回します。GPU を使う前に、その都度オーナーへ知らせます。AviUtl2 は閉じておきます。

1. 自己検査（GPU もサーバーも使わない）: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-chain-2026-11-09\h3_vsf_chain_gate.py --selfcheck`
   最後の行が `SELFCHECK PASS` なら、プロンプトの組み立て・G-C0 の本文との突き合わせ・判定の関数が想定どおりです。
2. 前の点検: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-chain-2026-11-09\preconditions.py before`
   終了コード 2 は AviUtl2 が動いている、3 はワーカーのプロセスが残っている、という意味です（台本は何も止めません）。
3. 配線の前のコードで GC0: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-chain-2026-11-09\h3_vsf_chain_gate.py --mode gc0 --label before`
4. 配線の後のコードで、GC0 から GC3 まで続けて: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-chain-2026-11-09\h3_vsf_chain_gate.py --mode all --label after`
   G-C0 の指紋が今のコードで再現しなかったとき（手順 3 の GC0 が指紋だけで落ちたとき）は、手順 3 の値を基準にして `--expect-sha <手順 3 の sha256>` を付けます。
   門を 1 つずつ回すときは `--keep-server` を付けると、次の回が同じサーバーと読み込んだ H3 をそのまま使います（GC3 は GC1 か GC2 の直後に、同じサーバーで回します）。
5. 後の点検: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-chain-2026-11-09\preconditions.py after`

送る引数を見るだけなら `--dry-run` を付けます（サーバーも MCP サーバーも起こしません）。

`h3_vsf_chain_gate.py` の引数:

| 引数 | 既定 | 意味 |
|---|---|---|
| `--mode` | （必須） | `gc0`・`gc1`・`gc2`・`gc3`・`all`（all は gc0 → gc1 → gc2 → gc3 を 1 つのサーバーで続けて回す） |
| `--label` | `run` | 結果の JSON に残す名前。GC0 は `gc0\<label>\` に書く（英数字と `.`・`_`・`-` だけ） |
| `--expect-sha` | G-C0 の値 | GC0 と GC3 で比べる動画の指紋 |
| `--material` | GC0 の出力 → GC1 の出力 | GC2 の素材の動画 |
| `--retake-frames` | 73 | GC2 の窓のフレーム数（17n+5・73〜481） |
| `--retake-start-sec` | 0.5 | GC2 の窓の開始秒 |
| `--retake-width`・`--retake-height` | 素材に合わせる | GC2 の画角（両方指定するか両方省略する） |
| `--port` | 18699 | 一時サーバーのポート。普段のサーバー（18620）とは別 |
| `--keep-server` | 切 | 終わりに H3 を降ろさず、サーバーも止めない。次の回は同じポートのサーバーを使い回す（一時の設定ファイルの名前で起こしたサーバーだけ。別のサーバーが応答していたら止まる） |
| `--load-timeout-sec` | 1200 | H3 の読み込みを待つ上限 |
| `--job-timeout-sec` | 3600 | 1 本のジョブを待つ上限。超えたら残りの門は回さない |
| `--dry-run` | 切 | 送る引数を表示して終わる |
| `--selfcheck` | 切 | GPU もサーバーも使わない自己検査をして終わる |

台本の動き:

- `config.yaml` は読むだけです。ポート・ログの置き場・出力の置き場・アップロードの置き場・状態ファイルの 5 つを `_rt\` に向け、さらに `vram.block_swap: true` と `vram.block_swap_blocks_on_gpu: 8`（GPU に残すブロックの数。G-C0 を回したときの `config.yaml` の値）を書き込んだ一時の設定ファイル（`_rt\config.h3vsfchain.yaml`）と、MiniMax H3 を選んだ状態ファイル（`_rt\state.h3vsfchain.json`）を作り、`.venv\Scripts\python.exe main.py --config _rt\config.h3vsfchain.yaml` で起こします。先読み（block swap prefetch）は設定ファイルの鍵ではなく要求ごとの引数なので、`block_swap_prefetch=true` を毎回送ります。オーナーの `state.json`・`logs\`・`outputs\` には書きません。
- 起動の環境は普段の起動（`run.ps1`）に合わせ、`tools\uv` と `tools\ffmpeg\bin` を PATH の先頭に足します。
- MCP サーバーには `LTX_MCP_BASE_URL`（一時サーバーの URL。`/api/v1` を含まない形）・`LTX_MCP_CONFIG`（一時の設定ファイル）・`LTX_MCP_OUTPUT_DIR`（`_rt\outputs`）・`PYTHONUTF8=1` を渡します。
- 起動の後、`tools/list` の引数の定義で、台本が渡す引数名がすべてあるかを照合します（無ければ終了コード 1）。`submit_chain` にステップ数・shift・VRAM 16GB mode の引数が足されていたら、記録だけ残します（台本は渡しません）。
- H3 の読み込みは `load_pipeline(base_model="MiniMaxH3", models={"transformer": "default"})` を 1 回だけ呼び、あとは `backend_status` を 10 秒おきに見て待ちます。
- ワーカーのログは `_rt\logs\h3_worker.log` の、ジョブの前後の差分を読みます（ジョブのフォルダの `worker_log_excerpt.txt`）。
- 1 つの門が失敗しても先へ進みます。待ちきれなかったジョブがあれば、残りの門は回しません。
- 終了コード: 0＝回した門がすべて合格、3＝どれかが不合格か失敗、1＝準備の段で止まった。最後に門ごとの合否と落ちた基準を短く表示します。
- サーバーの止め方: H3 を降ろし（`unload_pipeline`）、サーバーを止め、止まらなければプロセスの木ごと強制終了します。最後にサーバーの子孫（ワーカー）が残っていれば、それも止めます。
