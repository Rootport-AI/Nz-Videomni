# B-G4 敵対的コードレビュー（アプリ側＋操作パネル）

対象: Nz-Videomni dev の未コミット差分のうちアプリ側（`api/`・`services/`・`gradio_ui/`・`mcp_server/`・対応テスト・仕様書・MULTI_ENGINE・MCP_SERVER_DESIGN）と、操作パネル（`webui/src/**`・API_REFERENCE・DEVLOG）。`engine_h3/*` と `tests/test_h3_engine.py` は対象外（読んだのは、鍵名と OOM の文言の照合に要る範囲だけ）。読み取りのみで、テストは実行していない。

## 結論

- **Critical 0／Major 1／minor 9**。
- 総評: 契約は端から端まで一致している。鍵名は API の `vram16_mode`（`bool|None`、既定 None）→ `flag_defaults` → H3 アダプタが毎回 `bool` で送る → ワーカー `_GENERATE_FIELDS`（既定 False）→ `done.vram16_mode_used`（"on"/"off"）→ `GenerationOutcome` → `_write_metadata` の順につながっている。`fill_defaults` は `is None` のときだけ書き戻すので、False が True に化けることはない。LTX の 2 つの表は `is True` で判定するので、None と False は通る。422 の文言は H3 専用の文で、もう片方の LTX へ誘導しない。操作パネル・Gradio・MCP はどれも「既定と違うときだけ送る」で統一され、LTX では行を隠して既定の ON へ書き戻すので、LTX の要求に鍵は載らない。`tests/test_ltx_runner_payload.py` は差分なしで、LTX のワーカーペイロードのゴールデンは不変（裁定 45 に影響なし）。§6.10(d) の件数は AST で数えて `GenerateRequest` が 35 項目、2.5 の拒否表が 3 行で、表と一致する。
- 過剰設計: `H3_ONLY_REFUSALS`（1 項目の辞書と、2 つの `reject_unsupported` の 1 分岐）は最小限で妥当。拒否表の 3 つ組の形を変えるより影響が小さい。`flag_defaults` も既存の `sampling` と同じ作法で、新しい抽象は増えていない。
- 直すべきは Major 1 件（「503」の記述）で、残りは文書とコメントの正確さが中心。

## 指摘

### 1. Major — 「503 `GPU_OOM` の `detail`」はクライアントからは見えない（C2 アプリ）
- 場所: `mcp_server/tools/generate.py` の docstring（「ジョブが 503 ``GPU_OOM`` で失敗し、案内…が ``detail`` に入ります」）、`mcp_server/server.py` の `INSTRUCTIONS`（「GPU_OOM で失敗して detail に案内が入ります」）、`Docs/MCP_SERVER_DESIGN.md` D28、`Videomni_Backend_Specification.md` §6.2 の `vram16_mode` の行と履歴 v0.5.93。
- 問題: ジョブの失敗は非同期で起きるので、HTTP の 503 はどこにも返らない。`services/pipeline_manager.py:1028-1036` は `job.error = f"{code}: {message} ({detail})"` の 1 本の文字列にまとめるだけで、`GET /jobs/{id}` に `detail` という欄は無い。MCP のエージェントは説明文に従って行動するので、「503」や「`detail` を見よ」は誤った手がかりになる。
- 根拠: `pipeline_manager.py` の except 節。操作パネルの `JobCard.test` 自身が `error: "GPU_OOM: … (detail)"` という文字列の形で確かめていることとも矛盾する。API_REFERENCE §5.1 は「ジョブは`GPU_OOM`で失敗し、`detail`に…」と 503 を書いていない。
- 直し方: 「ジョブが失敗し、`get_job` の `error` が `GPU_OOM: …（解像度かフレーム数を下げるか、VRAM 16GB mode を切ってください）` になる」と、見える形で書く。4 か所とも同じ文にそろえる。

### 2. minor — OOM のテストがワーカーからの中継の形を通っていない（C2）
- 場所: `tests/test_engine_h3_adapter.py` `test_vram16_oom_guidance_reaches_the_job_error_as_gpu_oom`。
- 問題: `pm.runner.generate = boom` で、アプリ側から RuntimeError を直接投げている。実際には、ワーカーが `_emit("error", detail=_detail(exc))`（`repr(exc)` とトレースバック。最大約 2,800 字）を送り、アダプタ `h3/adapter.py:466` が `RuntimeError(event["detail"])` に包み直す。現状でも案内文は先頭の repr に残り、"CUDA out of memory" で `_is_oom` が当たるので、挙動は正しい。ただ、計画の「案内文を含む RuntimeError → 503 GPU_OOM」をこのテストで固定できているのは、アプリ内の半分だけになる。
- 直し方（任意）: 既存の `_read_worker_events` の差し替えを使い、`{"event":"error","detail":"RuntimeError('<案内文>')\nTraceback…"}` を返す 1 本にすると、端から端まで固定できる。注入の仕組みは新たに作らない。

### 3. minor — `bool(request.vram16_mode)` が None を OFF に変える（C2）
- 場所: `services/engines/h3/adapter.py:448`。
- 問題: `validate_for_family` を通らない経路で None が届くと、H3 の既定（ON）とは逆の OFF が黙って送られる。いまの経路（`api/generate.py:90`）は必ず書き戻すので、実害は無い。
- 直し方（任意）: `request.vram16_mode is not False`、または `VRAM16_MODE_DEFAULT_H3 if request.vram16_mode is None else request.vram16_mode`。書き方を変えなくても、None のまま届いたときの扱いをコメントで明示すれば足りる。

### 4. minor — 履歴 v0.5.93 の「既定の応答・記録…は変わらない」が事実と違う（C2）
- 場所: 仕様書 v0.5.93 の行。
- 問題: LTX でも次の 2 つが変わる。(1) `GET /models` の `unsupported_features` に `vram16_mode` が加わる。(2) `metadata.json`（と mp4 に埋め込む JSON）に `request.vram16_mode: null` と `vram16_mode_used: null` が加わる（`_write_metadata` の `req.model_dump()`）。どちらも加算なので無害だが、「変わらない」と言い切るのは誤り。
- 直し方: 「LTX のワーカーペイロードは不変。応答と記録には加算だけ（`unsupported_features` に 1 件、metadata に null の 2 鍵）」と書き直す。

### 5. minor — 仕様書 §6.10(b) の例「現在 2 件」が古いまま（C2）
- 場所: `Videomni_Backend_Specification.md:1602-1604` の jsonc の例（`"two_stage_hq", "prune_vaed"` で「現在 2 件」）。同じ節の 1611〜1612 行も、LTX 2.3 を 2 件と書いたまま。
- 問題: いまはどちらの LTX も 3 件。計画の触る節の一覧に §6.10(b) が入っていなかったための漏れ。
- 直し方: 件数を書かずに「正本は拒否表」と参照する形にする（文書の正本 1 箇所の原則）。あるいは 1 行の訂正行で `vram16_mode` を足す。

### 6. minor — Gradio のミラー定数のコメントが MCP について誤っている（C2）
- 場所: `gradio_ui/handlers.py`（`VRAM16_MODE_DEFAULT_H3` の上のコメント「the MCP tools import the canonical constant itself」）。
- 問題: MCP は定数を import しない。既定は None で送らないので、`api/models.py` のコメントと D28 もそう書いている。同じ変更の中でコメント同士が食い違っている。
- 直し方: 「MCP は値を持たない（既定 None＝送らない）。説明文だけが既定 on と書く」に直す。

### 7. minor — Gradio と操作パネルの注記が食い違い、「同じ内容」というコメントも誤り（C2／C3）
- 場所: `gradio_ui/i18n.py`（コメント「Same content as the WebUI's note」）、`webui/src/i18n/strings.ts` の `accelVram16Note`（ja「ノイズ除去の活性化を分割して…」）。
- 問題: Gradio は「計算を分割」と書き、OOM の案内と「既定はon」も含む。操作パネルは「活性化を分割」とだけ書く。コメントの言う「同じ内容」になっていない。また操作パネルの ja の「活性化」は説明の無い専門語で、オーナー向けの文体の規則にかかる。計画 §5 の指定どおりの文ではあるので、直すかどうかは監督の判断になる。
- 直し方: Gradio のコメントを「趣旨は同じ（Gradio は OOM の案内まで含む）」に直す。操作パネルの ja は「ノイズ除去の計算を分割して…」にそろえるのが無難（API_REFERENCE は「活性化（計算の途中で一時的に作られる大きなテンソル）」と説明を付けている）。

### 8. minor — API_REFERENCE の「7 フィールド共通」と、連結の本文に載りうる経路（C3）
- 場所: `Docs/API_REFERENCE.md` §5.1 の「高速化フィールドの送信規約（7 フィールド共通）…Single／Chained／バッチ／Edit の各経路はその結果を展開」。実装は `chainUtils.ts:951`・`buildA2vChainPayload.ts:269`・`useRetakeForm.ts:649` が `...accelerationRequestFields(...)` を展開している。
- 問題: 本文は 7 つとも全経路で同じ扱いだと読めるが、`vram16_mode` は `/generate` だけの項目で、型も連結には足していない。実行時は、保存値が OFF のまま LTX で起動し、`/models` が届いて書き戻すまでの短い間に連結を送ると、連結の本文に `vram16_mode: false` が載りうる（スプレッドには余分なプロパティの検査が働かない）。バックエンドの連結スキーマは余分な鍵を無視するので（`extra` の指定なし。`test_validation.py` が `model_fields` に無いことを固定）、実害は無い。担当の判断 2（連結の型には足さず、実行時に載っても放置）は妥当。
- 直し方: 送信規約の行に「`vram16_mode` は `/generate` の経路だけ。連結の経路に載っても、サーバーは無視する」と 1 文足す。コードの変更は不要。Alpha Gen は鍵を選んで渡しているので対象外（`useAlphaGenForm.ts:366`）。

### 9. minor — DEVLOG §142 が、リポジトリに無い作業メモを参照している（C3）
- 場所: `Docs/DEVLOG.md` §142.1「計画はバックエンドの作業メモ `PLAN_vram16.md` §5」と、§142.4「未コミット…」。
- 問題: `PLAN_vram16.md` は scratchpad にあり、リポジトリにも保管庫にも無いので、読み手は辿れない。§142.4 はコミット後に古くなる。
- 直し方: 参照先を設計正本の `Docs/MINIMAX_H3_DESIGN.md`（弾 C で書く節）か台帳 §1-101 に替える。§142.4 はコミットのときに更新する。

### 10. minor — `AppShell.featureScope.test` の LTX 2.3 の判定が名前で読みにくい（C3）
- 場所: `webui/src/shell/AppShell.featureScope.test.tsx` の新しい 2 本。どちらも `renderApp(AS_LTX25)` で始め、それを LTX 2.3 として扱っている。
- 問題: 既存の作法どおり（`AS_LTX25` は「2.5 を選べるモック」で、起動時は 2.3）なので正しい。ただ、2 本目の題が「on LTX 2.3」なのに、本文に 2.3 が有効になっていることを確かめる行が無い。
- 直し方（任意）: `expect(select.value).toBe("LTX23")` を 1 行足す。

## 確認して問題なしとした点（抜粋）

- `UNSUPPORTED_FEATURES` は拒否表の順（2.3＝`keep_resident_embeddings, alpha_gen, vram16_mode`、2.5＝`two_stage_hq, prune_vaed, vram16_mode`）で、完全一致のテスト（`test_ltx_adapter_audit`・`test_engine_dispatch`・`test_ltx25_api_guard`・`test_mcp_tools_system`・`test_ltx25_adapter`）も更新済み。
- H3 の `GOLDEN_GENERATE_KEYS_H3` は `block_swap_prefetch` の直後に `vram16_mode` があり、2 つのゴールデンで True／False の両方を確かめている。`_DONE_FACTS` は変えていない。
- Gradio: 隠した部品の戻し先は True で、「OFF のときだけ送る」と整合している。`_gated_components` と `FEATURE_UI` は同じ順で、末尾に追加されている。入力は `shift_audio` の直後で、連結と共有する後ろの並びは不変（負の添字は動かない。バッチの位置引数のテストは 46 個に更新済み）。ミラー定数の値が API と一致することもテストで固定している。
- MCP: 既定 None で、`is not None` のときだけ送る。省略時に鍵が無いことをテストで固定。`submit_chain` には出していない。
- 操作パネル: `accelerationRequestFields` は False のときだけ鍵を足す。`resets: "vram16Mode"` → `withServerDefaults` は既定の true へ戻す。`readStoredAcceleration` は古い文字列形式・鍵なし・不正値の 3 つとも既定の true になる（テストあり）。`useBatchForm` の OR-list への追加と、Batch i2v のテストもある。`JobCard.test` は `detail` 込みの表示を確かめている。
- `engine_h3/pipeline_h3.py:353` の `VRAM16_OOM_MESSAGE` は「解像度かフレーム数を下げるか、VRAM 16GB mode を切ってください」で、裁定 47 と一致し、「画角」は使っていない。"CUDA out of memory" を含むので `_is_oom` が当たる。
