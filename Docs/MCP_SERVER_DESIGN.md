# MCPサーバー設計判断の記録（`mcp_server/`）

> **本書の位置づけ**: `mcp_server/` パッケージ（Claude Code 等の MCP クライアントからバックエンドを操作するための22ツール）の設計判断を記録する。実装計画そのもの（Wave分割・敵対的レビュー原文）はオーナーのプランファイル（リポジトリ外）が正本で、本書はその決定事項をリポジトリ内に定着させたもの。利用者向けの使い方は [`README.md`](../README.md) 「AIエージェント連携（MCPサーバー）」節、機械検証の実行記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) の該当節を参照。

## 1. 何を作ったか

**MCP（Model Context Protocol。AIエージェントが外部ツールを呼び出すための標準規格）** サーバーを `mcp_server/` に新設し、既存の FastAPI バックエンド（`/api/v1/*`）を **22個のツール**として公開した。Web の操作パネル（`gradio_ui/`）が使える操作は一通りツール化してあり、パネルと同等の操作性が MCP 経由でも成立する（AviUtl2 のタイムライン連携は対象外——これはフロントエンド側の責務であり、本パッケージは扱わない）。

## 2. 主要な設計判断（D1〜D21）

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
| D14（2026-08-18追加） | **`submit_chain` の end source 引数を4引数（`end_source_strength`を2026-08-18に追加）へ拡張し、`INSTRUCTIONS` と docstring を逆順Chained対応へ更新する**。文言は「end_source はクリップ何件でも使える。クリップ1件のときは窓内モード、2件以上のときは逆順Chained（最後のクリップから順に生成し、各クリップが1つ後ろのクリップの冒頭を自分の末尾として引き継ぐ）になる。出力の長さはどちらの場合もクリップの合計」へ改めた。D13の「2件以上は非推奨」という案内は撤去した | 同日の逆順Chained実装（第2段階・バッチ2）でクリップ2本以上も推奨経路になったため、D13時点の「非推奨」案内は実態と食い違うようになった。`end_source_strength`（バッチ1で先行追加）も同日にツール引数へ公開した。**ツールの本数は22本のまま不変**（フィールドの追加であってツールの追加ではない）。実装・機械検証・実機ゲートの正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §63（strength）・§64（逆順Chained）。**【2026-08-22 追記】この「2件以上も推奨経路になった」という前提は、同じ2026-08-18の目視・試聴ゲートを経たオーナー裁定「End sourceはクリップ1本で使うのが推奨、複数クリップは推奨外」によって失効している**（裁定の正本は[`CHAIN_STAGE2_RESEARCH_NOTES.md`](CHAIN_STAGE2_RESEARCH_NOTES.md) §11末尾・[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-84）。**機能としては引き続きクリップ何件でも受理する**ので引数の公開は不変だが、`INSTRUCTIONS`・docstring の文言を裁定に合わせるかどうかは未判断である |
| D15（2026-08-22追加・**2026-08-23改訂**） | **ベースモデル軸（`base_model`）は `load_pipeline` の引数として公開する**（**ツール本数は22本のまま不変**）。`list_models` は当初から `active_base_model` と `base_models[]`（id / display_name / engine_family / active / installed / present / missing_categories / unsupported_features / category_order / categories）を**透過して返していた**——docstringが説明していなかっただけで、射影も加工も足していない（この透過そのものを契約として文書化した）。切替の可否判定はサーバー側の409/422に委ね、MCP側に事前ガードを二重に置かない（ジョブ実行中409 JOB_BUSY／未知id 404 MODEL_NOT_FOUND／重み未導入422 MODEL_FILE_MISSING／系統の食い違い422 MODEL_INCOMPATIBLE／読み込み中409 PIPELINE_LOADING）。`destructiveHint` も付けない | **【改訂前の判断（2026-08-22）】ベースモデル軸はMCPツールに出さない**——LTX 2.5対応（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-98）のv1スコープ判断で、エージェントが切り替えるとワーカーが載せ替わり、同時1ジョブ制約と相まって他の利用者の作業を止めうるため、D9（破壊的な操作にガードを置く）と同じ考え方でv1では露出しないほうを選んでいた。**この判断は2026-08-23のオーナー裁定で上書きされた**——「Chained移植・IC-LoRA検証の実験をMCP経由で回せるように、先にMCPを開通させる」。ガードを二重化しないのは、サーバー側が既に全ての拒否理由を封筒で返しており（`_raise_for_error` が `ToolError` へ一元翻訳する）、MCP側の事前チェックは**同じ判断を2箇所で持つ＝食い違う余地を作る**だけだからである。実機往復の記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §70 |
| D16（2026-08-23追加・2026-08-24改訂・**2026-08-25再改訂〔高速化第2弾〕**・**2026-08-25再々改訂〔高速化第3弾〕**） | **LTX 2.5 で使えるようになった機能を、`INSTRUCTIONS` と `submit_chain` / `submit_generate` の docstring の文言だけで反映する**（**ツール本数は22本のまま不変**・引数の増減も無し）。**2026-08-25 現在の文言**は「LTX 2.5 では submit_chain は連結生成そのものに加えて `source_video_id`（V2V継続）・`source_audio_id`（A2V。複数クリップにまたがる長尺A2Vも含む）・`loras`（スタイルLoRA）・`reference_video_id`（IC-LoRA。クリップ別の参照窓＝長尺IC-LoRAも含む）が使えるが、**使えないのは `end_source_*`（素材（末尾））／ `nag_enabled` ／ `vae_mode` の3つである**。**`keep_resident`（モデル骨格の常駐）は 2026-08-25 から LTX 2.5 でも使える**（既定 off のまま。ただし LTX 2.3 とは常駐する中身が違い、2.5 が抱えるのはテキストエンコーダの重みだけで約 7.7GiB である）。**同じ 2026-08-25、`attention_backend`（SageAttention）も LTX 2.5 で使えるようになった**（既定 `"sdpa"` のまま。チェーンの全クリップ・全ステージへ一律に効き、エコーはチェーン全体の畳み込み。**`"sage"` を選ぶと同じシードでも生成結果の細部が変わる**）」である（2026-08-23版では `reference_video_id` と `loras` も、2026-08-24版では `keep_resident` も、2026-08-25 の第2弾時点の版では `attention_backend` も使えない側に挙げていた） | 引数は最初から全部公開してあり、変わったのは**サーバーが受理する範囲**だけなので、MCP側で足すものは無い（D15と同じく、可否判定はサーバーの422に委ねてMCPに二重のガードを置かない）。それでも文言を直したのは、**エージェントは docstring を読んで「試すまでもない」と判断する**ためで、古い案内を残すと使える機能が使われないまま終わる。**`submit_chain` は `stage2_window` を送らない**（サーバー既定の `standard` で走る）——これはLTX 2.3の頃からで2.5対応で失われた機能ではなく、引数を足すかどうかは同等化のスコープ外である（[`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-102 の判断材料(5)）。実装は連結生成側が `33d28df`、V2V・A2V側が `74baeea`、**スタイルLoRA・IC-LoRA側が `41b5bb4`**、**モデル骨格の常駐側はアプリ層の開通が `69b58ec`（C2）・説明文の書き換えが `f01004e`（C3）**、**SageAttention 側はアプリ層の開通が `cb97613`（C2）・説明文の書き換えが `8176668`（C3）**である（**2026-08-25 の再改訂では、常駐する中身が LTX 2.3 と別物であることを 1 文添えた**——同名別実装の誤読が第2弾の最大のリスクだったため。**同日の再々改訂では、`"sage"` を選ぶと絵の細部が変わることを 1 文添えた**——エージェントが「速くなるだけ」と読んで既定のように使うのを防ぐためで、**この点はオーナーの目視ゲートでも 2026-08-25 に合格が出ている**〔同 §77.10〕）。実測の正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §72.5・§73.5・**§74.6**・**§76**・**§77**、オーナーの実機確認は同 §73.10 |
| D17（2026-08-26追加） | **撮り直し（Retake）と素材（末尾）（End source）の LTX 2.5 開通を、`INSTRUCTIONS` と `submit_chain` の docstring の文言だけで反映する**（**ツール本数は22本のまま不変**・引数の増減も無し）。**`end_source_video_id` / `end_source_image_id` は 2026-08-26 から LTX 2.5 でも使え、submit_chain がまだ断るのは `nag_enabled` / `vae_mode` の2つだけになった。** あわせて、推奨外の逆順Chained（クリップ2件以上）について**「LTX 2.5 では音声の継ぎ目がさらに悪くなる（2.3 は6箇所とも連続、2.5 は 4/6 が不連続。映像側は 2.5 でも 6/6 連続）」**という一文を加えた——**推奨のクリップ1件〔窓内モード〕には影響しない**ことも同時に書いている | **この2箇所は、開通の時点で失効していた**——実機ゲートG5で「LTX 2.5 では素材（末尾）は使えない」と書いたままの記述が`mcp_server/server.py`（`INSTRUCTIONS` の LTX 2.5 の段落）と `mcp_server/tools/generate.py`（`submit_chain` の docstring）の2箇所に見つかった。**実装計画は「MCPに失効記述は無い・検証済み」と記録していたが、これは誤りだった**——**「文言だけの反映」で済む段でも、失効の有無は毎回grepで確かめること。** **撮り直しは範囲外である**——`submit_chain` に retake の引数は **LTX 2.3 にも存在しない**ので、同等化すべきずれがそもそも無い（引数の追加は台帳[`PENDING_TASKS.md`](PENDING_TASKS.md) §3-115 へ起票した）。実装・ゲートの正本は[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §78（MCP経由の実機確認は同 §78.8 の R2）。 |
| D18（2026-08-30追加） | **非CFGネガティブプロンプト（NAG／VSF）の LTX 2.5 開通を、`INSTRUCTIONS` と `submit_generate` / `submit_chain` の docstring の文言だけで反映する**（**ツール本数は22本のまま不変**・引数の増減も無し）。**`nag_enabled` は 2026-08-30 から LTX 2.5 でも使え、`submit_generate` も `submit_chain` も、まだ断るのは `vae_mode` の1つだけになった**（`negative_prompt` / `nag_scale` / `nag_tau` / `nag_alpha` / `neg_method` / `vsf_scale` も同時に効くようになった）。あわせて**「`nag_alpha=0` にしても『NAG なし』とビット単位で同じ絵にはならないので、無効化は `nag_enabled=False` で行うこと」**という一文を両方へ加えた——**これはエージェントが取り違えやすい点である**（0 にすれば無効、と読むのが自然だが、実際には経路が変わる）。 | **この4箇所（`server.py` の `INSTRUCTIONS` 2箇所と、`generate.py` の docstring 2箇所）は、LLM クライアントに提示される製品の説明文でありながら、テストが1件も無い。** 文言が古いままだと、エージェントは「LTX 2.5 ではネガティブプロンプトが使えない」と信じて**使える機能を提案しなくなる**——サーバーが 422 を返すわけではないので、**誰も気づかない種類の劣化**である。したがって**LTX 2.5 の対応範囲が動くたびに、この4箇所を人手で確認するチェックリストが唯一の防波堤**であり、D16・D17 と同じ運用を本項でも踏襲する。実測は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §80 |
| D19（2026-09-01追加） | **`submit_chain` に撮り直し（Retake）の引数を公開する**——`retake_video_id` / `retake_window_start_sec` / `retake_head_px`（8n+1・既定25）/ `retake_tail_px`（8の倍数・既定24）/ `retake_regenerate_audio`（既定 `true`）の**5引数を末尾に追加**し、`retake_video_id` があるときだけ `retake` ネストを丸ごと足す（**5キーは既定値のままでも全部送る**——`RetakeSpec` 側の既定と同じ値なのでボディの意味は変わらず、ネストの形が呼び出しごとに揺れないほうがメタデータの突き合わせが読みやすい。`end_source` と同じ作法）。**POST前の `ToolError` は2本だけ足す**: `retake_video_id` があるのに `retake_window_start_sec` が無いときの `RETAKE_WINDOW_START_REQUIRED` と、撮り直しでないのに補助引数だけ非既定のときの `RETAKE_ARGS_WITHOUT_VIDEO_ID`。docstring と `INSTRUCTIONS` には**「窓の長さを決めるのは `clips[0].num_frames` ただ1つで、長さを表す第2の引数は存在しない」**と、**「窓開始秒の下調べには `upload_video(max_frames=...)` を使う」**（D21）を明記する。**ツール本数は22本のまま不変**（引数の追加であってツールの追加ではない） | **本表 D17 の「撮り直しは範囲外である」という裁定を、本項が解除する**（D17 本文は当時の判断の履歴として不改変。あちらは「LTX 2.3 にも引数が無いので同等化すべきずれが無い」という文言反映スコープの話であり、引数そのものを足す作業は同項が台帳 §3-115 へ起票していた）。API 側（`api/models.py` の `RetakeSpec`）は前から実装済みで、MCP は引数を素通しするだけである。**追加した検証を「指定が黙って消える／意図しない値に化ける」型の2本に絞ったのは、D15 と同じ理由**——幾何条件（8n+1・8の倍数・窓長 [73, 169]）と排他はサーバーの422が既に全て返しており、MCP 側に同じ判断を二重に持つのは食い違う余地を作るだけだからである。`retake_window_start_sec` にだけ既定値を置かずエラーにしたのは 2026-09-01 のオーナー裁定で、**忘れたときに黙って 0.0 になって「動画の頭」を作り直してしまう**のが最悪の失敗だからである（既存の `CROP_SIZE_INCOMPLETE` と同じカテゴリ）。出典は台帳 [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-115（2026-09-01にクローズ済み。[`PENDING_TASKS.md`](PENDING_TASKS.md) 側は欠番）、検証記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §86 |
| D20（2026-09-01追加） | **`submit_generate` に画角拡張（Outpainting）の引数を公開する**——`outpaint_pad_left` / `_right` / `_top` / `_bottom`（各既定0）/ `outpaint_blend_dilation_stage1`（既定5）/ `outpaint_freeze_source_audio`（既定 `true`）の**6引数を末尾に追加**し、**4辺の合計が0より大きいときだけ** `outpaint` ネストを足す（4辺すべて0なら通常の生成のままで、ボディは1バイトも変わらない）。**`blend_dilation_stage2` は引数にしない**——stage 1 が既定（5）ならこの2キーは両方とも送らず、非既定のときだけ公式ワークフローと同じ 5:2 の比で `_outpaint_stage2_from_stage1(r)`（`r<=0→0`・他は `max(1, round(r*2/5))`）が追従させる。**`outpaint` を送るとき、`loras` に `in-outpainting` が無ければ `{name: "in-outpainting", strength: 1.0}` を自動で注入する**（**重複判定は name の一致だけ**で行い、strength は見ない）。POST前の `ToolError` は `OUTPAINT_PADS_ALL_ZERO` の1本だけ（4辺すべて0なのに blend / freeze だけ非既定＝指定が黙って無視される形）。docstring と `INSTRUCTIONS` には幾何条件4つ（キャンバスは128の倍数／残す領域＝元動画の解像度／残す領域は縦横とも256px以上／元動画のフレーム数 ≥ `num_frames`）と、**`in-outpainting` が未導入の環境では404になるので事前に `list_loras` で確認すること**、**`upload_video` の応答は解像度を返さないのでキャンバスの逆算はローカルの ffprobe 等で行うこと**を明記する。**ツール本数は22本のまま不変** | API 側（`api/models.py` の `OutpaintSpec` と `GenerateRequest` の検証）は前から実装済みで、MCP は引数を素通しするだけである。**5:2 の比の正本は操作パネル**（`webui/src/modes/edit/outpaintGeometry.ts::stage2FromStage1`）で、そこから Python へ移植した——**Python の `round` は偶数丸め、JavaScript の `Math.round` は 0.5 切り上げだが、`r` が整数なら `r*2/5` の小数部は 0/0.2/0.4/0.6/0.8 のいずれかにしかならず、ちょうど .5 になる `r` が存在しないため両者の差は出ない**（この根拠をコードのコメントとテストの両方に残した）。既定（5）のとき2キーとも省略するのはパネルと同じ後方互換の規律である。**LoRA の自動注入は 2026-09-01 のオーナー裁定**で、操作パネルがこの LoRA を固定でピン留めしているのと同じ振る舞いを再現したもの——サーバーは `outpaint` に preprocess の無い control 系アダプタをちょうど1本要求するので、エージェントに毎回書かせても忘れて422になるだけだからである。出典は台帳 [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-122（2026-09-01にクローズ済み。[`PENDING_TASKS.md`](PENDING_TASKS.md) 側は欠番）、検証記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §86 |
| D21（2026-09-01追加） | **`upload_video` に `max_frames: int \| None = None` を足し、既存のクエリ引数（`api/uploads.py` の `max_frames: int \| None = Query(None)`）へ素通しする**。渡したときだけ応答の `frame_count` / `fps` が実測値になり、渡さなければクエリごと送らない（従来のアップロードのリクエストは1バイトも変わらない）。docstring に**「`max_frames` を渡すと尺と fps が実測されて返る。撮り直しの窓開始秒（D19）の下調べに使え」**と、**「`max_frames` は先頭Nフレームだけ残す切り詰め引数でもあるので、測るだけのときは元の尺より確実に大きい値を渡すこと」**を明記する。**ツール本数は22本のまま不変**（引数の追加であってツールの追加ではない） | 当初のスコープ判断は「docstring で『アップロード前にローカルで ffprobe を使え』と案内するだけ」だったが、敵対的レビューで**実 API の `upload_video` は `max_frames` クエリで `frame_count` / `fps` を返せる**ことが判明し（裏取り済み）、2026-09-01 のオーナー裁定で拡張することにした。これで**撮り直しの下調べは MCP 内で完結する**（バックエンド無改修）。**解像度（width / height）だけは今も返せない**ので、画角拡張（D20）のキャンバス逆算は ffprobe 案内のままである。出典は台帳 [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) §3-115（2026-09-01にクローズ済み。[`PENDING_TASKS.md`](PENDING_TASKS.md) 側は欠番）、検証記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §86 |

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
- `gradio_ui/manifest.py::scan_wav_folder` の走査規約（全音声拡張子を候補にし、`manifest`/`autosave`/`*.tmp` を除外、mtime昇順、非wav・読めないwavは可視Skip行 `skip_reason="wav-only-alpha"`、実効上限 `min(max_frames, 481)` 超は `"over-cap"`）。理由コードと上限の規約の正本は [`BATCH_A2V_CSV_SPEC.md`](BATCH_A2V_CSV_SPEC.md)。

写経の乖離を防ぐため、`tests/test_mcp_batch_planning.py` が本家 `gradio_ui.handlers.suggest_frames_for_audio` との**総当たりパリティテスト**（多数の秒数・fps値の組み合わせで両実装の出力を突き合わせる）で固定している。さらに写経元の2ファイル（`gradio_ui/handlers.py` / `gradio_ui/manifest.py`）側にも「MCPサーバー側に写経あり・変更時は両方＋パリティテストを更新」というコメントを追加してあり、将来どちらかを変更する開発者が反対側の存在に気づける設計にした。

画像フォルダの同stemマッチング（`plan_rows` の `image_dir` 引数）は、パネルには存在しないMCP専用の追加規約である。パネルの Batch A2V は行ごとに手動でドロップダウンから画像を選ぶ方式のため、自動マッチングに相当する既存ロジックがそもそも存在しない。エージェントが非対話で計画を立てられるよう、ここで新設した。

## 10. テスト構成（7ファイル・オフライン）

すべて `.venv`（torch無しのアプリ venv）で完結し、GPU・実バックエンドプロセスを必要としない。結合テストは `httpx.ASGITransport` でモックFastAPIアプリに直結し、非同期テストは追加pytestプラグインなしで `anyio.run` を使う。

| ファイル | 主な検証内容 |
|---|---|
| `test_mcp_registration.py` | 登録ツール数22・名前集合の厳密一致・全ツールに空でない説明文があること・MCPプロトコル経由の1往復（`structuredContent` が `{"result": ...}` でラップされずそのまま返ること） |
| `test_mcp_tools_system.py` | system系6ツール（疎通・設定取得・モデル一覧・パイプライン読み込み/解放・LoRA一覧）と**ベースモデル軸**（`load_pipeline` の `base_model` がbodyへ載ること・指定時にno_opの近道を通らないこと・`two_family_client` を使った実アプリでの切替と `list_models` の透過） |
| `test_mcp_errors.py` | エラー封筒の `ToolError` 翻訳・接続不能時の日本語文言・`ReadTimeout` の非翻訳（D5） |
| `test_mcp_tools_generate.py` | アップロード3本＋`submit_generate`/`submit_chain` のペイロード契約（隠しフィールド不在・None/空を送らない・XOR事前弾き等）。**撮り直し（D19）と画角拡張（D20）**では、`retake` / `outpaint` ネストの厳密一致・既定値のみのボディにこの2キーが現れないこと（トリップワイヤ）・POST前 `ToolError` 3本がHTTP呼び出しゼロで上がること・`in-outpainting` の自動注入と非重複・5:2追従の全域一致・`inputSchema` への露出。**`upload_video` の `max_frames`（D21）**はクエリ透過と `frame_count`/`fps` の返却 |
| `test_mcp_tools_jobs.py` | jobs系7ツール（`wait_for_job` のクランプとtimed_out契約・`cancel_job`/`delete_job` の状態ガード・`purge_terminal_jobs` のdry_run） |
| `test_mcp_outputs.py` | outputs系3ツール（パス導出の回帰・`no_clobber` 連番・`to_thread` 化） |
| `test_mcp_batch_planning.py` | `plan_a2v_batch` の走査規約（mtime昇順・Skip行可視・上限超過判定〔`min(max_frames, 481)`〕）と、本家 `suggest_frames_for_audio` との総当たりパリティ |

## 11. 参照

- 利用者向けの使い方: [`README.md`](../README.md) 「AIエージェント連携（MCPサーバー）」節
- 機械検証の実行記録: [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md)（本パッケージの節）
- セッションの入口（課題台帳）: [`PENDING_TASKS.md`](PENDING_TASKS.md)
