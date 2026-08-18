# MCPサーバー設計判断の記録（`mcp_server/`）

> **本書の位置づけ**: `mcp_server/` パッケージ（Claude Code 等の MCP クライアントからバックエンドを操作するための22ツール）の設計判断を記録する。実装計画そのもの（Wave分割・敵対的レビュー原文）はオーナーのプランファイル（リポジトリ外）が正本で、本書はその決定事項をリポジトリ内に定着させたもの。利用者向けの使い方は [`README.md`](../README.md) 「AIエージェント連携（MCPサーバー）」節、機械検証の実行記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) の該当節を参照。

## 1. 何を作ったか

**MCP（Model Context Protocol。AIエージェントが外部ツールを呼び出すための標準規格）** サーバーを `mcp_server/` に新設し、既存の FastAPI バックエンド（`/api/v1/*`）を **22個のツール**として公開した。Web の操作パネル（`gradio_ui/`）が使える操作は一通りツール化してあり、パネルと同等の操作性が MCP 経由でも成立する（AviUtl2 のタイムライン連携は対象外——これはフロントエンド側の責務であり、本パッケージは扱わない）。

## 2. 主要な設計判断（D1〜D14）

以下は実装計画で決定した設計判断の要約。番号は計画書の通し番号に対応する。

| # | 決定 | 理由 |
|---|---|---|
| D1 | `mcp>=1.28,<2` を **main依存**として `pyproject.toml` / `requirements.txt` の両方に追記 | エンドユーザーに追加手順なしで届けるため。`extra` 依存にすると `uv sync`（`dev` extra無し）で刈り取られてしまう。依存衝突は事前にPyPI実データで確認済み（starlette 1.3.1が mcp/sse-starlette/gradio 6.19 の制約を同時に満たす） |
| D2 | `mcp_server/` は `services/` / `gradio_ui/` / `main.py` を **import禁止**。importして良いのは `config.py` と `chain_math.py` のみ | 両方とも純Python・print無しを事前確認済み。`gradio_ui` は `__init__.py` 経由で最終的に重い `gradio` を import してしまい、stdout汚染のリスクもある |
| D3 | 全ツールを `async def ... -> dict[str, Any]` で統一。ブロッキングI/O（ファイルコピー・wavスキャン等）は `anyio.to_thread.run_sync` で包む | MCP SDK 1.28 は同期ツールをイベントループで直接呼び出し、スレッドへ逃さないことを実ソースで確認した。イベントループを塞がないためには自前でスレッドへ逃がす必要がある |
| D4 | `submit_generate` / `submit_chain` は submit（登録して即 `job_id` を返す）＋ `wait_for_job`（ポーリング）の2段構成を徹底。`wait_for_job` は `timeout_sec` を上限45秒にクランプし、タイムアウトはエラーではなく `timed_out: true` ＋その時点の `progress`/`stage` を返す | 生成は数十秒〜数分かかるため、1回の呼び出しをブロックさせ続けるのはMCPクライアント側のUXとして不適切。エージェントに「投げて→待つ」を繰り返させる設計にした |
| D5 | `/pipeline/load` と `/jobs/{id}/join` は `httpx.ReadTimeout` を `{"finished": false, "hint": ...}` に翻訳する（エラーにしない） | バックエンドのこれら2エンドポイントは同期処理で、クライアント側のソケットがタイムアウトしてもサーバー側の処理は継続して完走する。タイムアウト＝失敗ではないため、状態を再確認するヒントを返す設計にした |
| D6 | 生成された動画は常にローカル絶対パスで返す。`save_job_video` で任意フォルダへコピー（`no_clobber` で連番回避）。**base64埋め込みは禁止** | 動画ファイルは数MB〜数十MBあり、MCPのレスポンスに埋め込むとコンテキストを圧迫する。ローカルツール前提のため絶対パスで十分 |
| D7 | エラーは `{"error": {code, message, detail}}` 封筒を `ToolError("CODE: message — detail")` に一元翻訳。接続不能時は「run.batで起動してください」という日本語文言に翻訳。`backend_status` だけは例外を投げず `reachable: false` を返す | エージェントが読める形の一貫したエラーメッセージにするため。`backend_status` は「疎通確認」自体が目的のツールなので、疎通不可を例外にせず結果として返す設計にした |
| D8 | 隠しフィールド（`pipeline` / `num_inference_steps` / `guidance_scale` / `crf` 等、パネルにも出していない検証専用フィールド）はツールのスキーマに出さない。逆にパネル未配線だがAPIとして合法な機能（join の `audio_smoothing=false` 等）は公開する | パネルと同等の操作性を目指す一方、パネルが意図的に隠している実験用フィールドまで晒すと、エージェントが誤って触ってしまうリスクがある（**チェーンの参照動画〔長尺IC-LoRA〕はD8策定時点〔2026-07-28〕ではパネル未配線の例だったが、2026-08-11に`ChainReferencePanel`としてパネル配線済みになった**。ツール側は当初からD8の方針どおり公開済みで、変更は不要だった） |
| D9 | `cancel_job` / `delete_job` は実行前にGETで状態を確認し、意図（キャンセルしたい／削除したい）と実際の状態が食い違えばエラーにする。破壊的な操作には `destructiveHint` を付け、`purge_terminal_jobs` には `dry_run` を用意する | `DELETE /jobs/{id}` は「実行中ならキャンセル・終端済みなら削除（出力フォルダのrmtree含む）」という**1エンドポイント2動作**の設計になっている。この二面性を暗黙に踏ませず、エージェントが意図しない削除をしないためのガード |
| D10 | `list_jobs` はプロンプト全文などを含まない要約射影を返す。全文が要る場合は `job_status` を使う | 全ジョブぶんのプロンプト全文（最大2000字）を毎回返すとトークンを大きく消費するため |
| D11 | docstring・`instructions` は日本語（ツール名・引数名は英語のまま）。同時1ジョブ制約（409 JOB_BUSY）を `instructions` と両submitツールのdocstringの両方に明記する | エージェントの学習・利用者の可読性のため日本語で統一。409を連発させないための注意書きを、サーバー起動時の案内文と個々のツール説明の両方に重ねて置いた |
| D12（2026-08-05追加） | **Acceleration（生成の高速化）の5フィールドは全て公開する**。`submit_generate` / `submit_chain` の両方に、`attention_backend`（`"sdpa"` / `"sage"`・既定 `"sdpa"`）・`block_swap_prefetch`（bool・既定 `true`）・`keep_resident`（bool・既定 `false`）・`fused_gguf_dequant_kernel`（bool・既定 `true`）・`vae_mode`（`"default"` / `"prune_vaed"`・既定 `"default"`）を出す | 当初 `vae_mode` だけは D8 の趣旨（実体の無いフィールドをエージェントに触らせない）に沿って**モックである間は除外**していたが、2026-08-05 に実機能へ転換したため除外理由が消滅した。同日のオーナー裁定（「外出先から操作したいときに便利なので公開まで進みたい」）で公開へ転じ、実装と実機ゲート G1〜G7 の合格を待ってから最終ステップとして実施した。**ツールの本数は22本のまま不変**（フィールドの追加であってツールの追加ではない）。検証記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §52.10-b、設計の正本は [`PRUNAVAED_WORKORDER.md`](PRUNAVAED_WORKORDER.md) §6.3 |
| D13（2026-08-17追加） | **`submit_chain` に end source（素材（末尾））の引数を公開する**——`end_source_video_id` / `end_source_image_id` / `end_source_context_frames`（8の倍数・既定72）。2つのIDの同時指定は POST 前に `END_SOURCE_XOR_VIOLATION` として弾く。docstring と `INSTRUCTIONS` には**「1件＝窓内モード／2件以上＝逆順Chained。出力の長さはどちらもクリップ合計」**と明記する（**2026-08-18注**: 引数は4引数（`end_source_strength`を追加）、文言も逆順Chainedに合わせて更新済み。詳細はD14） | 2026-08-16 時点では「生成結果が素材へクロスフェードする」既知問題のため**あえて送信できないようにしていた**が、翌日の窓内モード（`in_window`）実装でその問題が解消し、除外理由が消滅した。実機実験4ラウンド20ジョブは、まさにこの3引数を使って MCP の実プロトコル経由で実施している（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §61）。**ツールの本数は22本のまま不変**（フィールドの追加であってツールの追加ではない） |
| D14（2026-08-18追加） | **`submit_chain` の end source 引数を4引数（`end_source_strength`を2026-08-18に追加）へ拡張し、`INSTRUCTIONS` と docstring を逆順Chained対応へ更新する**。文言は「end_source はクリップ何件でも使える。クリップ1件のときは窓内モード、2件以上のときは逆順Chained（最後のクリップから順に生成し、各クリップが1つ後ろのクリップの冒頭を自分の末尾として引き継ぐ）になる。出力の長さはどちらの場合もクリップの合計」へ改めた。D13の「2件以上は非推奨」という案内は撤去した | 同日の逆順Chained実装（第2段階・バッチ2）でクリップ2本以上も推奨経路になったため、D13時点の「非推奨」案内は実態と食い違うようになった。`end_source_strength`（バッチ1で先行追加）も同日にツール引数へ公開した。**ツールの本数は22本のまま不変**（フィールドの追加であってツールの追加ではない）。実装・機械検証・実機ゲートの正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §63（strength）・§64（逆順Chained） |

## 3. `.mcp.json` 絶対パス生成方式の経緯

計画時点の敵対的レビューで、当初案（`${CLAUDE_PROJECT_DIR:-.}` を使う相対パス的な解決）が実運用で壊れることが判明した。既定値 `.` はセッションの起動時cwd基準になるため、次の2パターンで破綻する。

1. 開発機でこのリポジトリの**親フォルダ**をワークスペースとして開いた場合、`.` が指すディレクトリが本リポジトリと一致しない。
2. エンドユーザーが `setup.bat` 実行前（＝`.venv` がまだ存在しない状態）で先にMCPクライアントを設定してしまった場合、そもそも解決すべき実体がない。

そこで、**`scripts/setup.ps1` がセットアップ完了時に、このマシンの `.venv\Scripts\python.exe` への絶対パスを埋め込んだ `.mcp.json` を生成する**方式に変更した。`.venv\Scripts\python.exe` の実在を確認できた場合のみ書き出し、まだ無ければ警告を出してスキップする（セットアップ途中の呼び出しに対して安全）。生成された `.mcp.json` はマシン固有の絶対パスを含むため **`.gitignore` に追加**し、リポジトリにはコミットしない。

これに合わせて `mcp_server/__main__.py` は `Path(__file__)` 起点で `sys.path` を自己解決する（`config.py` の `PROJECT_ROOT` と同じ手法）。これにより `PYTHONPATH` 環境変数への依存を無くし、`.mcp.json` の `args: ["-m", "mcp_server"]` だけで `config` / `chain_math` のトップレベルモジュールを解決できるようにしてある。

## 4. `reload_loras` を削減した理由

計画段階の一覧には `reload_loras`（パネルの `POST /loras/reload` に対応するツール）が含まれていたが、実装前に引き算として削除した。理由は、**`GET /loras`（`list_loras` ツール）自体が呼び出しのたびにディスクを再スキャンする**実装になっている（`api/loras.py`）ため、明示的な再スキャンツールは実質的に冗長だったため。この引き算により、ツール総数は当初想定の23から **22** になった。

## 5. 全ツールを async にした理由

MCP SDK（`mcp>=1.28,<2`）の実ソースを確認したところ、**同期関数として登録したツールは、サーバーのイベントループ上で直接呼び出される**（内部で `anyio.to_thread` 等へ自動的に逃がす仕組みは無い）ことが分かった。これは、1つのツール呼び出し中にブロッキングI/O（ファイルコピー・wavファイルの走査等）が発生すると、サーバープロセス全体（他のリクエストの処理も含む）が止まってしまうことを意味する。

そのため、**全22ツールを `async def` として実装**し、内部でブロッキングI/Oが必要な箇所（`save_job_video` の `shutil.copy2`、`plan_a2v_batch` のフォルダ走査等）は個別に `anyio.to_thread.run_sync` でスレッドへ逃がす方式に統一した。HTTPコールはすべて `httpx.AsyncClient` を使うため、この点は自然に非同期になっている。

## 6. stdout禁止の理由

MCPの `stdio` トランスポート（本サーバーが使っている接続方式）は、**JSON-RPCメッセージをプロセスの標準出力（stdout）に流す**。したがって、`mcp_server/` 配下のどこかで意図せず `print()` が呼ばれる、あるいはimportしたモジュールがstdoutに何かを書き出すと、そのバイト列がJSON-RPCストリームに混入し、以降の全メッセージが壊れる（クライアント側からは原因不明のプロトコルエラーとして観測される）。

これを避けるため:
- **`mcp_server/` 配下のどのファイルにも `print()` を書かない**という規約を明文化した（`server.py` の冒頭docstringに明記）。
- ロギングは `logging.basicConfig(stream=sys.stderr, ...)` で明示的に **stderr へ**向けている。
- import範囲を `config.py` / `chain_math.py` に限定した（§2のD2）ことで、重いモジュール（特に `gradio` を引き込む `gradio_ui`）が持ち込むかもしれない出力リスクを構造的に排除した。
- W0（最初のWave）のチェックポイントで `echo "" | python -m mcp_server` の stdout が空であることを機械的に確認する手順を必須にした。

## 7. 出力パスをローカル `config.yaml` から導出する理由

生成された動画の実パスは `config.output_dir/{job_id}/output.mp4`（`api/jobs.py`）、join済み動画は `config.output_dir/{job_id}/joined.mp4`（`services/join_manager.py`）が正本である。

一見、`GET /jobs/{id}` のレスポンス（`JobResult.output_path`）を使えば良さそうに見えるが、これは**相対パスのハードコード**であり信用できない。また `GET /config` のレスポンスにも `output_dir` は含まれない。そのため `mcp_server/paths.py` は**HTTPレスポンスを一切見ず**、`mcp_server.client.BackendClient.output_dir`（= `Settings.output_dir` = ローカルの `config.load_config().output_dir`）だけを入力にパスを組み立てる純関数として実装した。これはMCPサーバーがバックエンドと**同じマシン上で、同じ `config.yaml` を読める**という前提（stdioトランスポートの性質上、両者は常に同一マシン上にある）に立脚した設計である。

## 8. 22ツール一覧

| カテゴリ | 本数 | ツール名 |
|---|---|---|
| system | 6 | `backend_status` / `get_config` / `list_models` / `load_pipeline` / `unload_pipeline` / `list_loras` |
| uploads | 3 | `upload_image` / `upload_video` / `upload_audio` |
| generate | 2 | `submit_generate` / `submit_chain` |
| jobs | 7 | `job_status` / `list_jobs` / `wait_for_job` / `cancel_job` / `delete_job` / `purge_terminal_jobs` / `join_job` |
| outputs | 3 | `get_job_video_path` / `get_joined_video_path` / `save_job_video` |
| batch | 1 | `plan_a2v_batch` |

一覧の1行説明は [`README.md`](../README.md) 「AIエージェント連携（MCPサーバー）」節のツール一覧表を参照。ツール名の集合は `tests/test_mcp_registration.py::EXPECTED_TOOLS` が厳密アサートで固定している。

## 9. 写経（`batch_planning.py`）のパリティテスト方針

`plan_a2v_batch` ツールは、パネルの Batch A2V 機能（`gradio_ui/manifest.py::scan_wav_folder`・`gradio_ui/handlers.py::suggest_frames_for_audio`）と同じ規約でフォルダを走査する必要がある。しかし `gradio_ui` パッケージは `__init__.py` 経由で重い `gradio` を import してしまう（起動コスト・stdout汚染リスク）ため、`mcp_server/` から直接importすることを避け、代わりに **`mcp_server/batch_planning.py` へロジックを1対1で写経**した。

写経元は2箇所:
- `gradio_ui/handlers.py::suggest_frames_for_audio`（フレーム数提案。stdlib `wave` で長さを取得し、`chain_math.audio_latents_required` と突き合わせて8刻みで縮める）。
- `gradio_ui/manifest.py::scan_wav_folder` の走査規約（全音声拡張子を候補にし、`manifest`/`autosave`/`*.tmp` を除外、mtime昇順、非wav・読めないwavは可視Skip行 `skip_reason="wav-only-alpha"`、481フレーム超は `"over-481f"`）。

写経の乖離を防ぐため、`tests/test_mcp_batch_planning.py` が本家 `gradio_ui.handlers.suggest_frames_for_audio` との**総当たりパリティテスト**（多数の秒数・fps値の組み合わせで両実装の出力を突き合わせる）で固定している。さらに写経元の2ファイル（`gradio_ui/handlers.py` / `gradio_ui/manifest.py`）側にも「MCPサーバー側に写経あり・変更時は両方＋パリティテストを更新」というコメントを追加してあり、将来どちらかを変更する開発者が反対側の存在に気づける設計にした。

画像フォルダの同stemマッチング（`plan_rows` の `image_dir` 引数）は、パネルには存在しないMCP専用の追加規約である。パネルの Batch A2V は行ごとに手動でドロップダウンから画像を選ぶ方式のため、自動マッチングに相当する既存ロジックがそもそも存在しない。エージェントが非対話で計画を立てられるよう、ここで新設した。

## 10. テスト構成（7ファイル・オフライン）

すべて `.venv`（torch無しのアプリ venv）で完結し、GPU・実バックエンドプロセスを必要としない。結合テストは `httpx.ASGITransport` でモックFastAPIアプリに直結し、非同期テストは追加pytestプラグインなしで `anyio.run` を使う。

| ファイル | 主な検証内容 |
|---|---|
| `test_mcp_registration.py` | 登録ツール数22・名前集合の厳密一致・全ツールに空でない説明文があること・MCPプロトコル経由の1往復（`structuredContent` が `{"result": ...}` でラップされずそのまま返ること） |
| `test_mcp_tools_system.py` | system系6ツール（疎通・設定取得・モデル一覧・パイプライン読み込み/解放・LoRA一覧） |
| `test_mcp_errors.py` | エラー封筒の `ToolError` 翻訳・接続不能時の日本語文言・`ReadTimeout` の非翻訳（D5） |
| `test_mcp_tools_generate.py` | アップロード3本＋`submit_generate`/`submit_chain` のペイロード契約（隠しフィールド不在・None/空を送らない・XOR事前弾き等） |
| `test_mcp_tools_jobs.py` | jobs系7ツール（`wait_for_job` のクランプとtimed_out契約・`cancel_job`/`delete_job` の状態ガード・`purge_terminal_jobs` のdry_run） |
| `test_mcp_outputs.py` | outputs系3ツール（パス導出の回帰・`no_clobber` 連番・`to_thread` 化） |
| `test_mcp_batch_planning.py` | `plan_a2v_batch` の走査規約（mtime昇順・Skip行可視・481f超判定）と、本家 `suggest_frames_for_audio` との総当たりパリティ |

## 11. 参照

- 利用者向けの使い方: [`README.md`](../README.md) 「AIエージェント連携（MCPサーバー）」節
- 機械検証の実行記録: [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md)（本パッケージの節）
- 次セッション引き継ぎ: [`NEXT_SESSION_HANDOFF.md`](NEXT_SESSION_HANDOFF.md)
