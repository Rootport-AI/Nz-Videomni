# E2_app.md — H3 の `vram16_mode`（仮）を足すときの、アプリ側の触る場所と前例

パスはすべて `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni` からの相対です。事実だけを書き、設計の提案は入れていません。

## 0. 先に知っておくべき事実

- **`config.yaml` の区画名は `low_vram` ではなく `vram:`** です（`config.py` の `VramConfig`、`config.yaml.example:80-108`）。`low_vram` は `services/low_vram.py` の変数名です。
- **`GenerateRequest` に `model_config` はありません**（`api/models.py` を grep しても出ません）。pydantic の既定なので、知らない項目は黙って捨てられます。
- **機能の有無を伝える仕組みは「拒否リスト方式（ネガティブ）」だけ**です。
  - 正本は各アダプタの `REJECT_TABLE` で、`UNSUPPORTED_FEATURES` はそこから導出されます。
  - 記述子 `scripts/manifests/40-minimaxh3.json` には機能の欄がありません（`unsupported`・`feature` の grep は 0 件）。
  - 「この系統だけにある」を肯定形で書く欄は、`constraints.sampling` だけです。
- **前例どうしで向きが違います。**
  - `keep_resident` / `attention_backend` / `block_swap_prefetch` は全系統で動く項目です。
  - H3 専用に近い前例は 2 つあります。
    - `shift_video` / `shift_audio`: 既定が `None` で、LTX では「無視＋ログ」です。
    - `keep_resident_embeddings`: LTX 2.5 専用で、他系統は 422 で断ります。

## 1. API の要求モデル

| 項目 | 場所 | 前例の形 | 備考 |
|---|---|---|---|
| 既定値の定数 | `api/models.py:113-164` | `BLOCK_SWAP_PREFETCH_DEFAULT=True`(124)、`KEEP_RESIDENT_DEFAULT=False`(134)、`KEEP_RESIDENT_EMBEDDINGS_DEFAULT=False`(143) | 各定数のコメントに「変えるときは api の正本、gradio のミラー、MCP の import 元の 3 点を同時に」とある。`attention_backend` に定数はなく、`Literal["sdpa","sage"]="sdpa"`(495) と直書き |
| 項目定義 | `api/models.py:476-592`（`GenerateRequest` の Acceleration 区画） | `attention_backend`(495)、`block_swap_prefetch`(510)、`keep_resident`(535)、`keep_resident_embeddings`(556)、`fused_gguf_dequant_kernel`(568)、`vae_mode`(584)、`embed_mp4_metadata`(592) | 各項目に日本語の長いコメント。`*_used` が `metadata.json` に載る旨と、`GET /status` に載せない理由を書く作法 |
| H3 用の `None` 既定の前例 | `api/models.py:610-617` | `num_inference_steps`、`shift_video`、`shift_audio` は `Field(None, ...)`。コメントは「`None` は系統の既定値で、`validate_for_family` が書き戻す」 | LTX には概念がないので、そのまま無視される |
| 検証(422) | `api/models.py:668-801`（`validate_ltx_constraints`） | Acceleration 項目には検証が 1 つもない。型エラーだけ pydantic が 422 にする（例: `tests/test_validation.py:804` の `"keep_resident": "not-a-bool"`） | 他の項目との相互検証があるのは NAG・LoRA・outpaint・inpaint・alpha_gen のみ |
| 系統別の規則表 | `services/engines/ltx/request_rules.py`（`RequestRules`）、`services/engines/h3/adapter.py:125`（`REQUEST_RULES`） | 規則表は数値の規約専用（フレーム格子・倍数・fps・`sampling`）。真偽の機能スイッチを持つ欄はない | `violation()` の検査順は width、height、frames、範囲、fps、sampling、extra |
| 機能の可否(FEATURE_UNSUPPORTED) | `api/generate.py:69-95` | 順序は `engines.reject_unsupported` が先、`engines.validate_for_family` が後、その後に alpha_gen、画像、LoRA | 「機能の 422」は「数値の 422」より先に返る |
| 連結用スキーマ | `api/models.py:1209-1226`（`GenerateChainRequest`）、`to_clip_request`(1697-1757。kwargs は 1739-1742) | 加速項目はここにも同名で存在し、`to_clip_request` に転記が要る。落としても検証は通るが、記録が既定値の嘘になる | H3 は `reject_chain` で連結を丸ごと断る。連結側に足す必要は、事実としては無い（`num_inference_steps` や `shift_*` も連結側には無い） |
| Alpha Gen | `api/models.py:408-437`、`api/generate_alpha.py:245-259` | 加速項目を素通しでコピーする（`keep_resident` が 2.5 では解放要求になるため） | Alpha Gen は LTX 2.5 専用なので、H3 項目は対象外 |

テストの更新点は次のとおりです。

- `tests/test_validation.py:613-817`: 既定値、連結での存在、`to_clip_request` の転記を確かめています。連結側に足すなら同型のテストが要ります。
- `tests/test_engine_h3_adapter.py:246-253`: `test_schema_bounds_read_the_rules_constants` が `shift_*` の既定 `None` を固定しています。
- `request` の丸ごとの鍵集合を固定するゴールデンは、grep で見つかりませんでした。

## 2. アダプタ(H3)と、LTX 側で要ること

`services/engines/h3/adapter.py` の該当行は次のとおりです。

| 要素 | 行 | 前例の形 |
|---|---|---|
| `REJECT_TABLE` | 166 | `(field, feature, 既定と違うか)` の 3 つ組。判定は常に「既定と違うか」 |
| `IGNORED_FIELDS` | 183 | 現状 `fused_gguf_dequant_kernel` のみ。`_log_ignored`(264) が、既定と違う値のときだけ 1 行ログを出す |
| `HONOURED_FIELDS` | 191 | `keep_resident`、`attention_backend`、`block_swap_prefetch`、`shift_*` などが載っている |
| `GOVERNED_FIELDS` | 213 | 従属項目（NAG 系など） |
| `UNSUPPORTED_FEATURES` | 232 | `REJECT_TABLE` の機能名（重複除去）に `("chain","retake")` を足したもの |
| `_build_load_payload` | 348 | キー集合と順序が契約で、ゴールデンで固定。`blocks_on_gpu = self.low_vram.block_swap_blocks_on_gpu or DEFAULT_BLOCKS_ON_GPU(=8)`。`attention_backend` と `block_swap_prefetch` は `GenerateRequest.model_fields[...].default` を送る（ワーカーの既定値） |
| `generate` の payload | 429-448 | 先に `attention_backend: request.attention_backend`(442) と `block_swap_prefetch: bool(...)` を**毎回**入れる（コメント: 「不在はロード値の意味になる」）。`keep_resident` は `if request.keep_resident:` のときだけ `True` を最後に追加(448)（不在が解放要求） |
| `done` の読み取り | 482-497 | `GenerationOutcome(attention_used=event.get("attention_used"), block_swap_prefetch_used=..., keep_resident_used=..., peak_vram_reserved_mb=..., h3=_h3_block(event))` |
| `_DONE_FACTS` / `_h3_block` | 285 / 298 | `done` の最上位から `h3` 欄へ集める鍵の一覧。ここに足すと `metadata["h3"]` に出る |

H3 ワーカー側の取り決めは次のとおりです（アプリ側の変更ではないが、境界の事実）。

- `engine_h3/worker.py:156-160` の `_GENERATE_FIELDS` に載っていない鍵は「無視」としてログに出ます。
- `blocks_on_gpu` はロード時だけ読まれます（`worker.py:235`）。生成メッセージの許可リストにはありません。

LTX 側で「H3 だけの項目」をどう扱うかは、3 つの前例のどれかになります。

| 前例 | LTX 2.3(`services/engines/ltx/adapter.py`) | LTX 2.5(`services/engines/ltx25/adapter.py`) |
|---|---|---|
| (a) 無視＋ログ（`shift_*`） | `IGNORED_FIELDS`(153-170)。**宣言のみでログは出さない**。`generate` が `request.<f>` を読まないことを監査テストが確かめる | `IGNORED_FIELDS`(154)。`_log_ignored`(532) がログを出す（既定と違うときだけ） |
| (b) 422 で断る（`keep_resident_embeddings`） | `REJECT_TABLE`(137) に 1 行。`UNSUPPORTED_FEATURES`(232) はそこから導出され、メッセージは「LTX 2.3は…対応していません」で LTX 2.5 を案内する | `REJECT_TABLE`(141) に 1 行。`UNSUPPORTED_FEATURES`(472) は `REJECT_TABLE` と `CHAIN_REJECT_TABLE` から導出される |
| (c) 動作 | `HONOURED_FIELDS`(176) | `HONOURED_FIELDS`(238) |

LTX 側で必ず要るのは次の 2 点です。

- **分類は必須です。** `GenerateRequest` の全項目は、3 つのアダプタ（H3・LTX 2.3・LTX 2.5）の 4 表のどれかにちょうど 1 つ属する必要があります。これは機械検証されていて、足し忘れるとテストが落ちます。
- **連結側は別表です。** 連結スキーマに項目を足さなければ、LTX 2.5 の `CHAIN_*` 表（323、334、401、464）は要りません。

テストの更新点は次のとおりです。

- 分類の網羅監査は 3 か所にあります。
  - `tests/test_engine_h3_adapter.py:69-98`（H3 は「`conditioning_images` だけ 2 表に重なってよい」）
  - `tests/test_ltx_adapter_audit.py:67-82`
  - `tests/test_ltx25_adapter.py:3109-3133`（連結側は 3153-3180）
- H3 の `test_honoured_fields_are_read_by_generate`(100) は、honoured にしたら `generate` のソースに `request.<field>` の文字列が必要、と要求します。ignored は逆に、読んではいけません。
- 項目を `REJECT_TABLE` に足す場合の更新点は次のとおりです。
  - H3: `REQUEST_OVERRIDES_H3`(112-148) に例を足す。`test_the_fixture_covers_every_refusal`(150) は「機能名の集合が一致」を要求する。`test_unsupported_features_is_the_reject_table_plus_chain_and_retake`(181) の完全一致タプルも更新する。
  - LTX 2.3 を 422 にする場合:
    - `REQUEST_OVERRIDES_23`（`tests/test_ltx_adapter_audit.py:38`）に例を足す。
    - `tests/test_ltx_adapter_audit.py:144` の `== ("keep_resident_embeddings","alpha_gen")` を更新する。
    - `tests/test_engine_dispatch.py:222` を更新する。
    - `tests/test_ltx25_api_guard.py:1609` を更新する。
    - `tests/test_mcp_tools_system.py:313` を更新する。
    - `tests/test_ltx25_adapter.py:576` の近辺を確認する。
  - LTX 2.5 を 422 にする場合:
    - `REQUEST_OVERRIDES`(`test_ltx25_adapter.py:272`) と、502 の集合一致を更新する。
    - `test_ltx25_adapter.py:612` の `len(ltx25.UNSUPPORTED_FEATURES) == 2` と、`tests/test_ltx25_api_guard.py:1655` の `len(features) == 2` を更新する。
- ペイロードのゴールデンは次のとおりです。
  - H3: `GOLDEN_LOAD_KEYS_H3`(350)、`GOLDEN_GENERATE_KEYS_H3`(447)、`test_generate_payload_t2v_golden`(471)、`..._i2v_keep_resident_and_per_job_acceleration`(501)。
  - `DONE_EVENT`(404): `*_used` を足すときはここに追加する。`outcome.h3` の完全一致(488) は `_DONE_FACTS` を変えたときだけ動く。
  - LTX のペイロード(`tests/test_ltx_runner_payload.py:351` の `test_default_payload_key_set_is_unchanged_by_acceleration`) は、LTX には送らないので変わらない。
- 備考: LTX 2.3 の `_RealBackend.generate` は `attention_backend`・`block_swap_prefetch`・`keep_resident` を `ltx/adapter.py:2280-2299` で「サーバー既定と違うときだけ鍵を送る」作法で扱います。H3 の「毎回送る」とは違います。

## 3. `GET /models`・`unsupported_features`・`constraints`

| 要素 | 場所 | 事実 |
|---|---|---|
| 公開 | `api/models_registry.py:134-142` | `unsupported_features` は `engines.unsupported_features(family)`、`constraints` は `engines.request_constraints(family)` |
| 取り次ぎ | `services/engines/__init__.py:165-212` | アダプタの `UNSUPPORTED_FEATURES`、`REQUEST_RULES`、`reject_unsupported` を `getattr` で読むだけ。宣言のない系統は「制限なし」 |
| **正本はどちらか** | `Docs/MULTI_ENGINE_DESIGN.md` §5.6 (1) | 「判断の主体はエンジンアダプタ 1 箇所」。**記述子ではない**。記述子は機能の欄を持たない |
| 規則表から作る `constraints` | `RequestRules.constraints()` | キー集合は `num_frames`、`dimension_multiple`、`frame_rate`、`sampling` の 4 つで固定。新しい真偽の欄は、ここには存在しない |
| 画面側の読み方 | Gradio: `gradio_ui/feature_scope.py:29-50`、WebUI: `featureScope.ts:130` の `FEATURE_UI` | 機能名が `unsupported_features` に**あれば閉じる**。知らない名前は無視。肯定形の機能を伝えるのは `constraints.sampling` だけ（`gradio_ui/constraints.py`） |

新しい機能名を足す前例（`keep_resident_embeddings`）の手順は次の 3 つです。

1. アダプタの `REJECT_TABLE` に行を足す。
2. `FEATURE_UI` と `RESET_VALUES` に行を足す。
3. `ui.py` の `_gated_components` に部品を足す（足し忘れると構築時に KeyError）。

`GATED_CONTROLS` の件数に依存するテストは `tests/test_gradio_models.py:398`、`457`、`510` です。

テストの更新点は次のとおりです。

- `tests/test_engine_h3_adapter.py:558`（`GET /models` の H3 欄）と `:239`（`H3_CONSTRAINTS`／`LTX_CONSTRAINTS`）。
- `constraints` に新しい鍵を足すなら、`test_constraints_blocks_are_the_plans_json` と `tests/test_gradio_constraints.py` の `H3_CONSTRAINTS`／`LTX_CONSTRAINTS` が変わります。

## 4. 設定(`config.yaml`)と「状態の 3 層」

| 要素 | 場所 | 事実 |
|---|---|---|
| 設定の型 | `config.py:176-199`（`VramConfig`） | `block_swap_blocks_on_gpu: int | None`。コメントは「worker 非伝播（表示専用）」と「内部 knob」の別を明記 |
| 例 | `config.yaml.example:80-108` | `block_swap: true`、`block_swap_blocks_on_gpu: 8` |
| 変換 | `services/low_vram.py:27-86` | `LowVramSettings` に `block_swap_blocks_on_gpu` を持つ。`_STATUS_KEYS`（`GET /status`）に出るのは 7 キーに固定で、内部 knob は出さない契約。`metadata_block`(`vram_optimization`) も固定キー |
| H3 の読み方 | `h3/adapter.py:366` | `self.low_vram.block_swap_blocks_on_gpu or DEFAULT_BLOCKS_ON_GPU(8)` を**ロード時の 1 回だけ**ペイロードに入れる。ジョブごとには動かない |
| 利用可否 | `services/pipeline_manager.py:321` | `_block_swap_prefetch_available` は `int(blocks_on_gpu or 8) > 0`。H3 でも同式 |
| 3 層憲章 | `Docs/MULTI_ENGINE_DESIGN.md` §5.4 | `config.yaml` はユーザーの上書き・好み・VRAM のつまみ（サーバーは読むだけ）。`state.json` は最後のベースモデル。ブラウザ localStorage に表示言語・テーマ・`.acceleration` など。ジョブごとの値は要求に載る |
| 快適上限の語彙 | `config.py:257-275`（`ComfortRow.requires`） | 要求の項目名（`attention_backend`、`block_swap_prefetch`、`keep_resident` など）で行を引く。H3 には `comfort` がない（記述子 `_note`） |

テストの更新点は次のとおりです。

- 設定値側を変えるなら `tests/test_engine_h3_adapter.py` の `test_load_payload_golden`（`blocks_on_gpu: 8`）。
- `config` のゴールデンは、本調査では grep していません。

## 5. Gradio

| 要素 | 場所 | 前例の形 |
|---|---|---|
| 部品 | `gradio_ui/ui.py:1231`（`attention_backend` の `gr.Radio`）、`1241`（`accel_prefetch` の `gr.Checkbox`）、`1249`（`accel_keep_resident`）、`1268`（`accel_keep_resident_embeddings`、`visible=False` で作り、`/models` の取得後に出す） | ラベルと説明は `L("accel_lbl_*")`、`L("accel_info_*")`。`reg(...)` で言語切替に登録 |
| 文言 | `gradio_ui/i18n.py` EN 側(約 100-130)、JA 側(約 668-682) | `accel_lbl_*` と `accel_info_*` が各言語に必要。`tests/test_gradio_ui.py:732-1002` が EN の鍵で部品を引く |
| 要求への載せ方（単発） | `gradio_ui/handlers.py:1032-1065` | `attention != "sdpa"`、`prefetch != DEFAULT`、`keep_resident != DEFAULT` の**「既定と違うときだけ」**方式。既定と同じ値は送らない |
| 既定のミラー | `gradio_ui/handlers.py:52-95` | HTTP 越しのため `api.models` を import せず、自前のミラー定数を持つ |
| 引数の追加位置 | `handlers.py:698-734` | キーワード専用で末尾に追加する作法（位置引数をずらさない）。最後が `constraints`、`num_inference_steps`、`shift_video`、`shift_audio` |
| 入力配線 | `ui.py:1736-1745`(入力リスト)、`1528-1570`(`dispatch`)、`1660-1690`(バッチのスナップ)、`2218-2250`(`chain_dispatch` は `args[-8]` などの**負の添字が末尾の長さに固定**、`tests/test_gradio_ui.py` が順序を固定) | 単発のみに足せば、連結の負添字はずれない（H3 は連結を断る） |
| H3 での表示切替 | `gradio_ui/constraints.py`（`sampling_payload`、`single_scope_updates`）、`ui.py:446-449`（`gen_scope_state`）、`1460-1472`（`on_gen_scope_change`）、`610-623`（`shift_video` は `visible=False` で作る） | `constraints.sampling` が null でなければ 3 本のスライダーを出し、CFG を隠す。`gen_scope_state.change` で更新。サーバーが `sampling` を返さなければ LTX 扱い |
| `unsupported_features` による非表示 | `ui.py:2514-2584` | `hidden_controls(active_unsupported_features(models_json))` で隠し、`RESET_VALUES` の値へ戻す（見えない部品も値を送り続けるため） |
| バッチ | `gradio_ui/batch.py:258-264`(`BatchSnapshot`)、`618-623` | 連結（A2V）用のスナップショットのみ。H3 は連結を断るので、単発項目は対象外 |

テストは `tests/test_gradio_handlers.py`（139 件ヒット）、`tests/test_gradio_ui.py`、`tests/test_gradio_models.py:376-510`、`tests/test_gradio_constraints.py`、`tests/test_gradio_batch_runner.py` です。

## 6. MCP

| 要素 | 場所 | 事実 |
|---|---|---|
| 引数 | `mcp_server/tools/generate.py:167-171` | `attention_backend: str="sdpa"`、`block_swap_prefetch: bool=BLOCK_SWAP_PREFETCH_DEFAULT`、`keep_resident: bool=KEEP_RESIDENT_DEFAULT`、`keep_resident_embeddings`、`fused_gguf_dequant_kernel`。H3 用の前例は 180-182 の `num_inference_steps`、`shift_video`、`shift_audio`（`None` 既定） |
| 既定の取得 | 61-65 | `api.models` から `*_DEFAULT` を import（Gradio と違い、MCP は同一リポジトリなので import する） |
| ペイロード | 501-536 | `if x != DEFAULT: payload[x]=x`（Acceleration は定数との比較で、「True のとき送る」にしない旨のコメント付き）。H3 の 3 項目は `if x is not None:` で 486-491 |
| 説明文 | モジュール docstring 1-58、`submit_generate` の docstring 190-262・338-450 | 引数ごとの日本語の説明。H3 専用の項目は「LTX では〜」と系統別の挙動を書く |
| サーバー説明 | `mcp_server/server.py:20-100`(`INSTRUCTIONS`) | 系統別の注意を書く文。`keep_resident` の系統別の中身も記述 |
| 連結 | `submit_chain` 608-1019 | H3 は連結不可なので出さない（docstring に明記） |
| ツール本数 | `tests/test_mcp_registration.py` の `EXPECTED_TOOLS` | 25 本。引数の追加ではツール本数は変わらない（D27 の前例） |

テストは `tests/test_mcp_tools_generate.py` です。

- `:338`、`:392`: 既定のとき鍵が増えないことの固定。既定の鍵集合 `{"prompt","width","height","num_frames","frame_rate","seed"}` は、既定のまま送らない作法なら変わらない。
- `:582-660`: 入力スキーマの検査。`:648` は `num_inference_steps` などが任意で既定 `None` であることを固定。
- `:2063-2064`: `embed_mp4_metadata` の既定の一致。同型の一致テストを足すなら、ここが書式の前例。
- 設計表: `Docs/MCP_SERVER_DESIGN.md` の D12、D22、D27（引数追加の前例。D27 は 2026-10-18）。

## 7. `metadata.json` と mp4 への埋め込み

| 要素 | 場所 | 事実 |
|---|---|---|
| `*_used` の書き出し | `services/pipeline_manager.py:1860-1893`（`_write_metadata`） | `attention_used`、`block_swap_prefetch_used`、`keep_resident_used`、`keep_resident_embeddings_used`、`fused_gguf_dequant_kernel_used`、`vae_mode_used`、`peak_vram_reserved_mb` が常に存在する鍵（mock では `None`） |
| 連結の書き出し | `pipeline_manager.py:1633-1653`、1571-1574 | 同じ鍵を連結用にも書く。連結の `request` は `chain.model_dump()` なので、連結スキーマに項目を足さない限り `_write_chain_metadata` は無関係 |
| 新しい `*_used` を足す場合の経路 | 3 段 | `GenerationOutcome` に項目を足す（`ltx/adapter.py:558-640`）→ H3 の `generate` で `event.get(...)`（`h3/adapter.py:482`）→ `_write_metadata` に鍵を足す |
| `h3` 欄 | `pipeline_manager.py:1955-1956`、`h3/adapter.py:285-301` | `outcome.h3` があれば `metadata["h3"]` に丸ごと入る。`done` の最上位から `_DONE_FACTS` の鍵を集め、`event["h3"]` の中身を重ねる。**`*_used` を `h3` 欄に載せるなら、アプリ側の変更は `_DONE_FACTS` への追加だけ**で済む |
| `request` の写し | `_write_metadata` の `"request": req.model_dump()` | 新しい項目は自動で `metadata.json` の `request` に載る（コードの追加は不要） |
| mp4 への埋め込み | `pipeline_manager.py:1798、1986、1989` | `metadata.json` と同じ JSON 文字列を `comment` タグに入れる（`video_io.recipe_text`）。**`*_used` も `h3` も `request` も、コード変更なしに mp4 に入る** |

## 8. 仕様書・設計書の該当節

| 文書 | 節 | 内容 |
|---|---|---|
| `Videomni_Backend_Specification.md` | §6.2（694-820）。Acceleration 表 720-725、補足 771-782 | 項目ごとに 1 行の表。`keep_resident_embeddings` の補足(782) が「連結に同名で存在する」旨を書く形 |
| 同上 | §6.2b（821-866） | (c) 規則表、(e) `constraints` のスキーマ、(f) H3 の `unsupported_features` の列挙（`keep_resident_embeddings` まで列挙）。**ここに「H3 の閉じた機能」の一覧があり、名前を足したら更新が要る** |
| 同上 | §6.5b（1218-1247） | `GET /status` の `acceleration` は非凍結。`keep_resident` は載せない旨 |
| 同上 | §6.6（1249-1277） | `metadata.json` の `*_used` の表。`h3` 欄は §6.2b (g) |
| 同上 | §6.10 (d)（1623-1638） | LTX 2.5 の 4 分類の件数表（全 34 件、422 系 2・無視 4・動作 28）。**`GenerateRequest` に項目を足すと件数が変わる**（`shift_*` のときに改訂した前例が見出しに残る）。1636 に「以後項目を足す人は 2 つのアダプタの両方で分類が必要」とある |
| 同上 | §11.3（1960-1976） | `vram` の設定表。`block_swap_blocks_on_gpu`（内部 knob、`status` 非出力）の行 |
| 同上 | §12.2（2058-2092） | Gradio の仕様。ステップ数・Shift の行(約 2074) は `constraints.sampling` の記述。Acceleration 節は v0.5.62 で追随 |
| `Docs/MULTI_ENGINE_DESIGN.md` | §5.2(298)、§5.4(345)、§5.6(396-456)、§7(571)、§9.2(893) | §5.6 (1)〜(4) が「正本はアダプタ」「強制はサーバー」「先回りは `GET /models`」「`constraints` は規則表から」。H3 の 4 分類の正本は `h3/adapter.py` の表そのもので、文書へは写さない(§5.6 冒頭) |
| `Docs/MCP_SERVER_DESIGN.md` | D27 | MCP に H3 用の引数を足した前例 |
| 他 | `README.md`（keep_resident の表 746）、`AviUtl2-Plugin\...\Docs\API_REFERENCE.md`、`webui/src/api/types.ts:230-256`（`keep_resident?: boolean`、「`true` のときだけ現れる」）、`webui/src/shell/featureScope.ts` | WebUI 側の写し（本棚卸しの対象外だが、同期が要る先） |

## 9. 要点

1. `config.yaml` の区画は `vram:` です。H3 の `blocks_on_gpu` はロード時 1 回だけ読まれ、ジョブごとには動きません（`h3/adapter.py:366`）。
2. 機能の有無の正本は**アダプタの `REJECT_TABLE`** です。記述子（`40-minimaxh3.json`）に機能の欄はありません。
3. 公開の仕組みは「拒否リスト」だけです。肯定形で系統の機能を伝える枠は `constraints.sampling` だけです。
4. `GenerateRequest` に項目を足すと、H3・LTX 2.3・LTX 2.5 の 3 アダプタの 4 表への分類が必須です。足し忘れは `tests/test_engine_h3_adapter.py:69`、`test_ltx_adapter_audit.py:67`、`test_ltx25_adapter.py:3109` が落とします。
5. LTX 側の扱いは前例が 3 つあります。`shift_*` と同じ無視（2.3 はログなし、2.5 はログあり）、`keep_resident_embeddings` と同じ 422、動作です。422 にすると `UNSUPPORTED_FEATURES` の完全一致テスト 5 か所と、2.5 の「件数 2」の固定 2 か所が動きます。
6. H3 の payload は `attention_backend` と `block_swap_prefetch` を毎回送り、`keep_resident` は真のときだけ最後に付けます。新項目をどちらにするかで、`GOLDEN_GENERATE_KEYS_H3` と 2 つのゴールデンテスト（471、501）が変わります。
7. H3 ワーカーは `_GENERATE_FIELDS`（`worker.py:156`）にない鍵を「無視」としてログに出すので、ワーカー側の対応が要ります。
8. `*_used` を足す経路は `GenerationOutcome` → H3 の `generate` の `event.get` → `_write_metadata` の 3 段です。`h3` 欄に載せるだけなら `_DONE_FACTS` への追加だけで済みます。
9. mp4 の埋め込みは `metadata.json` の全文を入れるので、新しい `*_used` も `h3` も `request` も、コード変更なしに入ります。
10. Gradio の作法は、部品を単発の Generate タブにだけ足し、要求には「既定と違うときだけ」載せ、ミラー定数を持ち、`handlers.py` の引数は末尾のキーワード専用で追加する、というものです。連結の負の添字は固定されているので、単発のみなら影響しません。
11. Gradio の非表示の前例は 2 つで、`sampling` のスライダー（`constraints` 駆動）と `keep_resident_embeddings`（`unsupported_features` 駆動で、`RESET_VALUES` と `_gated_components` が要る）です。
12. MCP は `submit_generate` に引数を足し（H3 用の前例は `None` 既定の `shift_*`）、`server.py` の `INSTRUCTIONS` と docstring を更新します。ツール本数は 25 のままです。
13. 仕様書で更新が要る箇所は、§6.2 の表、§6.2b (f)、§6.6、§6.10 (d) の件数表、§12.2、`MULTI_ENGINE_DESIGN` §5.6 です。