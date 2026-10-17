# R5a バックエンド棚卸し（MiniMax H3 第 2 弾の計画用）

調査日 2026-10-17。読むだけ（編集なし・GPU 不使用・テスト不実行）。対象は `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`（dev、HEAD `ca6f047`）。パスはリポジトリからの相対、行番号は調査時点。R3 は `Docs/Outputs-archive/h3-stage1-reports-2026-10-17/R3_inventory.md` を指す。「事実」はコードを読んで確かめたこと、「推測」はそうでないこと。

---

## ① 結論の要約

**A. 常駐の既存機構**
- 要求の項目名は `keep_resident`（`api/models.py:522`、既定 False）。アダプタは「オンのときだけ `keep_resident: True` を generate に足す」形で、**キーが無いこと自体が解放の指示**になっている（2.3・2.5 共通）。2.5 が主メモリに抱えるのはテキストエンコーダの重みだけ（DiT の重みは load の `cache_weights: True` で常に保持）。2.3 は全部品の骨格を抱え、先読みが off なら自動で off に落とす。
- 常駐の状態は `GET /status`・`GET /models` に出ない（`metadata.json` の `keep_resident_used` だけ）。ベースモデル切り替えは系統が変わればワーカーのプロセスごと終了、同じ系統でも unload するので、どちらも常駐分は消える。
- H3 のワーカーには常駐の経路が無く、毎ジョブ「組む→読む→解放」を 3 段で繰り返す（`engine_h3/pipeline_h3.py` の `_encode`／`_build_dit`／`_place_dit`／`_denoise` の finally／`_decode`）。手が入る場所は特定した（詳細 A3）。

**B. API の制約を系統別にする土台**
- `GenerateRequest` に**ベースモデルや系統を表す項目は無い**。pydantic の検証が外の状態（ロード中のモデル）を読む前例も無い（`ValidationInfo`・`context=` の使用ゼロ）。検証は本体の受け取り時点で走り、その後に初めて `engines.reject_unsupported(active_engine_family, …)` が呼ばれる（`api/generate.py:79`）。
- **重大**: `num_inference_steps` は既に `GenerateRequest` にある（`api/models.py:593`、既定 8・1〜100）。ただし `pipeline == "distilled"`（既定）のとき **8 以外を 422 で拒否する**検証がある（`api/models.py:657-661`、連結は 1346-1349）。このままでは H3 にステップ 20 を送れない。新設が要るのは shift の 2 本だけで、ステップ数は既存項目の扱いの変更になる。
- 422 は 2 系統ある。pydantic 由来は `VALIDATION_ERROR`（英語の `msg`、`main.py:370-388`）、アダプタ由来は `FEATURE_UNSUPPORTED`（英日併記の message＋日本語の detail、`api/errors.py:623`）。アダプタの拒否表は「既定値と違うか」の真偽だけを見る作りで、数値の規約違反（17n+5 など）を返した前例は無い。エンドポイントで数値規約を独自の符号で返す前例は `REFERENCE_RESOLUTION_INVALID`（`api/errors.py:572`・`api/generate.py:103`）がある。

**C. 事前検査と記述子**
- `precheck_model_file`（`services/model_registry.py:118`）・`_precheck_sft_transformer`（200）・疑似 KV（216）は R3 のとおり。`h3_format.py` には `inspect_h3_dit`（293）と `inspect_h3_te`（432）があり、どちらも `(Header, 設定のデータクラス)` を返し、拒否は日本語の `H3FormatError(ValueError)`。**`precheck_model_file` は `APIError` と `OSError` しか捕まえない**ので、そこから `inspect_h3_*` を呼んで `H3FormatError` が漏れると 500 になる（事実。変換が要る）。VAE 用の判定関数は無い。
- 記述子の `assets` はフォルダも取れる（LTX 2.3 の `gemma_root` がトークナイザのフォルダで、`downloads` 側は `"kind": "dir"` の行）。tokenizer の受け皿の前例そのもの。`30-uetrack.json` は schema 1 の取得専用記述子で、**「LTX でない系統のベースモデル」の前例にはならない**。
- 系統の 3 辞書・`engine_python_ltx25`（`config.py:122`）は R3 のとおり。`config.yaml.example` には `engine_python*` の行が無い（既定は `config.py` だけ）。

**D. アダプタの骨格**
- `services/engines/ltx25/adapter.py` の定数とクラスの行は R3 と完全に一致。Alpha Gen は `_read_worker_events(…, one_stage=…)`（`ltx25/adapter.py:1022-1027`）で、単発の割合計算は `stage1_denoise` を 0.06〜0.90 に広げる（`ltx/adapter.py:478-482`）。`decode` 段は単発では割合を動かさない（None）。
- `_real_available`（`ltx/adapter.py:902`）は「エンジンの python・各カテゴリの `default_file`・`REQUIRED_ASSETS`・`<engine_dir>/worker.py`」の存在で決まる。
- mock の単発生成は**フレーム数・fps をそのまま使い、8n+1 を自分では前提にしない**（`_render_frames`、`ltx/adapter.py:1587`）。規約の強制は pydantic 側にある。H3 の 17n+5・24fps の絵を作るために mock 自体に触る必要は、単発に限れば無い（R3 の記述より範囲が狭い）。

**E. Gradio と MCP**
- Gradio のステップ数と CFG のスライダーは `gradio_ui/ui.py:547-550`（Single）と `876-880`（Chain）。**どちらも `interactive=False` の飾りで、送信にはつながっていない**（送信は `handlers.py:604-605`・`991-992`・`1495-1496` の固定値 8／1.0）。Gradio の機能スコープ（`gradio_ui/feature_scope.py`）は**隠して既定値へ戻す**方式。
- MCP は `submit_generate` が `num_inference_steps` を意図的に出していない（`mcp_server/tools/generate.py:4-9`）。ベースモデルは `load_pipeline(base_model=…)` で選ぶ。MCP 側に検証は無いので、API が 17n+5・32 の倍数を通せば T2V／I2V は引数そのままで届く。ただし `num_frames` の既定 49 は 17n+5 ではない。

**F. 復号 VRAM**
- 映像の復号は「7 潜在フレームの時間窓 × 256px の空間タイル」で、**各窓の結果を float32 のまま GPU に溜め、最後に `torch.cat` で 1 本にする**（`engine_h3/video_vae.py:620` `_decode_temporal`）。その後 `decode`（601）で正規化を戻し、パイプラインの `frames_to_uint8`（`pipeline_h3.py:182`）が GPU 上で並べ替え・float 化・255 倍・uint8 化してから CPU へ移す。`write_mp4`（`media_out.py:81`）は CPU の uint8 `[F,H,W,3]` を要求する。音声は窓なしで一度に復号し、`.float().cpu()` で CPU へ。
- 既存テスト `tests/test_h3_media.py:52-69` が「`decode` の戻り値は float32・[0,1]・形 `(1,3,F,16,16)`」を固定している。

**G. 落ちるテストと実行方法**
- R3 の一覧の行はほぼ一致（ずれは `test_base_model_contract.py` の拡張子のテストが 84-89 → 87-94 の 1 件）。R3 に無い連鎖として、ステップ数の固定を緩めると落ちる・意味が変わるテスト（`test_ltx25_adapter.py:802`、`test_validation.py` ほか）と、H3 側の黄金キー（`test_h3_engine.py:575-589`）がある。
- 実行は、アプリ側 `.\.venv\Scripts\python.exe -m pytest tests\<file>`（`pyproject.toml` の `testpaths=["tests"]`・`-q`）、エンジン側 `.\.venv-engine-h3\Scripts\python.exe scripts\run_h3_pytest.py tests\test_h3_engine.py tests\test_h3_media.py`（`--noconftest` 付き。GPU のテストは `H3_GPU_TESTS=1` でのみ動く）。`tests/test_h3_format.py` は torch 不要でアプリ側の環境で動く。

---

## ② 詳細

### A. 常駐の既存機構

#### A1. 全経路（設定値 → 要求 → アダプタ → ワーカー）

| 段 | 事実 | 場所 |
|---|---|---|
| 操作パネルの表示 | ラベルは「モデル骨格の常駐（ジョブ間キャッシュ）」。行は Settings の Acceleration 区画 | `webui/src/i18n/strings.ts:3520`、`webui/src/shell/SettingsPanel.tsx:467-481` |
| 操作パネルの送信 | `accelerationRequestFields` が、オンのときだけ `keep_resident: true` を足す。**先読み block swap（`block_swap_prefetch`）が有効なときだけオンとみなす**（`keepResidentEffective`）。先読みが無効なら行が押せない | `webui/src/shell/accelerationSettings.ts:453-458`・`553-575`、`SettingsPanel.tsx:195-196`・`469-470` |
| 隣の行の前例 | 「Embeddings processor の常駐（LTX 2.5）」の行は、選択中のベースモデルの `unsupported_features` に `keep_resident_embeddings` があると**非表示**になる | `SettingsPanel.tsx:491-528` |
| API の項目 | `GenerateRequest.keep_resident: bool = KEEP_RESIDENT_DEFAULT`（False）。連結 `GenerateChainRequest`（1203）、Alpha Gen の要求（423）にも同名で素通し | `api/models.py:121`・`522`・`1203`・`423` |
| 2.3 アダプタ | `request.keep_resident` が真のときだけ generate に `keep_resident: True` を足す（連結も同様） | `services/engines/ltx/adapter.py:2278-2279`・`2629-2630` |
| 2.5 アダプタ | 同じ足し方。`keep_resident_embeddings` も別キーで同様 | `services/engines/ltx25/adapter.py:878-879`・`1012-1013`・`1343-1344` |
| 2.5 の load | `cache_weights: True` を**常に**送る（アプリに設定は無い）。DiT の重みを主メモリのレジストリに抱え、2 段目で読み直さないため | `ltx25/adapter.py:713-720` |
| 2.5 ワーカー | `_resolve_keep_resident`：キーが無ければ False＝**前のジョブが抱えたものを解放する指示** | `engine25/worker.py:561-581` |
| 2.5 の保持物 | `Ltx25Pipeline.set_acceleration_job`（`engine25/pipeline25.py:1275`）が `_swap_keep_resident`（880）を呼び、Gemma 4 テキストエンコーダの state dict を**主メモリ（CPU）**に残すかを切り替える。オフに変わった時点でレジストリから取り出して `del`＋`gc.collect()`。`keep_resident_embeddings` は埋め込み処理器の state dict を同じ関数で別に扱う。自動で off に落とす経路は無く、こだまは "on"/"off" の 2 値 | `engine25/pipeline25.py:846-950`・`1362-1386`・`1476` |
| 2.3 ワーカー | `_resolve_keep_resident(msg, bs_prefetch)`：全部品の骨格を抱える。`dit_cpu_load=False`（G-B）と先読み off（G-C）で自動的に off（`"on->off"`）。キーが無いジョブで解放 | `engine/worker.py:692-` |
| 結果の記録 | `done` の `keep_resident_used` を `GenerationOutcome` 経由で `metadata.json` へ | `ltx25/adapter.py:1071`、`services/pipeline_manager.py:1883` |

系統ごとの違い（事実）: 名前と約束（オンのときだけ送る・無いと解放・こだまを返す）は同じで、中身が別物。2.3 は全部品の骨格（主メモリ）と先読み依存のガード、2.5 はテキストエンコーダの state dict 1 本（主メモリ）でガードなし。VRAM に何かを常駐させる実装はどちらにも無い。

#### A2. 状態 API と切り替え時の解放

- `GET /status` の `acceleration` は `attention_backends`・`sage_available`・`block_swap_prefetch_available` の 3 つだけ（`services/pipeline_manager.py:279-311`）。常駐は「メモリ量の問題でサーバーには判定できない」ので載せない方針がコメントに明記されている（`api/models.py:513-514`）。`GET /models`（`api/models_registry.py:99-160`）にも常駐の項目は無い。
- 系統をまたぐ切り替え: `_point_runner_at`（`pipeline_manager.py:563`）が旧 Runner の `unload()` を呼ぶ → `_RealBackend.unload`（`ltx/adapter.py:2106`）が `shutdown` を送って待ち、最後は terminate／kill。**プロセスが消えるので、常駐分は OS に返る**。
- 同じ系統内の切り替え: `LTXRunner.set_descriptor`（`ltx/adapter.py:693`）が `unload()`＋バックエンド破棄。これもプロセス終了。
- 明示の解放: `POST /pipeline/unload` → `PipelineManager.unload`（`pipeline_manager.py:492`）。

#### A3. H3 に常駐を入れる場合に手が入る場所（場所の特定まで）

第 1 弾のパイプラインの段ごとの組み立てと解放（すべて `engine_h3/pipeline_h3.py`）:

| 段 | 組むもの | 解放 |
|---|---|---|
| `generate`（406） | `validate_request` → `torch.no_grad()` の中で 3 段 → finally で `_free_cuda()` | — |
| `_encode`（453） | 毎回 `H3TextEncoder(...)` を作り `te.load()`（重みを読む）→ `te.encode()` | finally で `te.release()`＋`del te`＋`_free_cuda()` |
| `_build_dit`（479） | 毎回ヘッダを読み直し（`inspect_h3_dit`）、meta 上で `H3DiT` を組み `swap_in_int8_linears` → `load_streaming`（ファイルから 1 本ずつ）→ `normalize_dtypes` | — |
| `_place_dit`（511） | ブロック以外を `move_module_tree` で GPU へ。ブロックは CPU のまま `BlockSwapService.install`。`blocks_on_gpu <= 0 または >= 50` なら DiT 全体を GPU へ | — |
| `_denoise`（538） | ステップのループ | finally で `teardown_prefetch()`（アリーナの輪＝GPU 側の作業領域を解放）→ `release_installed()` → `del dit` → `_free_cuda()` |
| `_decode`（631） | `H3VideoVAE.from_file(..., device=GPU, bf16)` → 復号 → `del`。続いて `H3AudioVAE.from_file(..., device=GPU)` → 復号 → `del` | 各 finally で `del`＋`_free_cuda()` |
| `close`（669） | — | 常駐の `BlockSwapService` を片付ける（shutdown 時） |

- `BlockSwapService` は 1 プロセスに 1 つで既に常駐（`pipeline_h3.py:330-333`・`527-528`）。ページ固定のプール（2 枠、コメントでは 2×369MB）と転送用ストリームはジョブをまたいで残る。`hold_arenas=True`。アリーナの輪はジョブの終わりの `teardown_prefetch` で毎回捨てる（`engine/transformer/block_swap_service.py:175-193`）。
- ワーカーの受け口: `_GENERATE_FIELDS`（`engine_h3/worker.py:143-146`）に無いキーは「無視したキー」としてログに出すだけ。`keep_resident` を受けるにはここと `_generate_request`（244）・`H3GenerateRequest`（`pipeline_h3.py:106`）の拡張が要る。黄金キーのテスト（`tests/test_h3_engine.py:579-582`・`610-615`）がキー集合を固定している。`done` に `keep_resident_used` を足すなら `GOLDEN_DONE_KEYS`（583-588）も。
- 同じ DiT を次のジョブで使い回すときの落とし穴（事実）: 同期経路の `_patch_block` は `block.forward` をそのまま「元の forward」として退避する（`block_swap_service.py:228-229`）ので、**同じブロックに 2 回 `install` すると包みが二重になる**。先読み経路は退避済みの印を優先するので二重にならない（342-346）。2.5 はこれを `_unpatch_block_swap`（`engine25/gguf_transformer.py:1149`）と `ensure_block_swap_installed`（1113）で処理している（使い回しの前例）。
- 推測: 「DiT を主メモリに保持」は、`_build_dit` の結果を保持して `_denoise` の finally の `del dit` を避ける形になるが、`_place_dit` がブロック以外（埋め込み・最終層など）を GPU に置くので、ジョブの合間にそれを CPU へ戻すかどうかが別に決まらないと、VRAM に一部が残る。VAE も `from_file` が GPU へ移してからキャストするので、主メモリに保持するなら置き場の扱いを決める必要がある。

### B. API の制約を系統別にする土台

#### B1. 検証の時点で系統を知れるか

- `GenerateRequest`（`api/models.py:435`）にベースモデル id・系統の項目は無い（`base_model`／`engine_family` の項目なし。コメント内の言及だけ）。
- `validate_ltx_constraints`（`api/models.py:644-786`、`@model_validator(mode="after")`）は自分の項目だけを見る。リポジトリの `api`・`services`・`mcp_server`・`gradio_ui` に `ValidationInfo`・`info.context`・`model_validate(..., context=...)` の使用は無い（事実。外の状態を読む前例なし）。
- 順序（事実）: FastAPI が本体を `GenerateRequest` に変換する時点で pydantic の検証が走り、失敗すると `RequestValidationError` → `main.py:370-388` が `{"error": {"code": "VALIDATION_ERROR", "message": "Request validation failed", "detail": [{loc, msg, type}]}}` の 422 を返す。エンドポイントに入って最初の処理が `engines.reject_unsupported(context.pipeline_manager.active_engine_family, request)`（`api/generate.py:79`）。系統は `runner.descriptor.engine_family`（`pipeline_manager.py:251-261`）。
- R3 の行（確認済み・ずれなし）: スナップ 201・inpaint 416/430・`num_frames` 591・fps 592・64 の倍数 646/648・8n+1 650・822・867・937/942・1146・1225・1336/1338・1468。
- **R3 に無い事実**: 同じ検証の中に「`pipeline == "distilled"` なら `num_inference_steps == 8` かつ `guidance_scale == 1.0`」がある（`api/models.py:657-664`、連結 `1345-1352`）。`pipeline` の既定は `"distilled"`（598）。`num_inference_steps` は既存で `Field(8, ge=1, le=100)`（593）。
- 先頭フレームのスナップ `_normalize_conditioning_images`（166-218）は `frame_idx > 0` を 8n+1 格子へ寄せる。`frame_idx == 0`（先頭フレーム）はそのまま。H3 の α版は先頭フレームだけなので、0 なら影響しない（事実：0 は触らない）。
- アダプタ側の 422: `REJECT_TABLE` は `(項目名, 機能名, 既定値と違うかの関数)` の並び（`ltx25/adapter.py:140-143`、`ltx/adapter.py:136-140`）。`reject_unsupported` が最初の違反を `feature_unsupported(feature, detail=日本語)` で返す（`ltx25/adapter.py:472-494`、`ltx/adapter.py:229-248`）。**数値の規約違反を返す行は無い**（すべて「既定値か否か」）。`UNSUPPORTED_FEATURES` は拒否表から導出され `GET /models` に出る。
- 宣言を持たない系統は素通り（`services/engines/__init__.py:157-208`、`getattr(..., None)`）。

#### B2. 既存 422 の文言

| 出どころ | 符号 | 言語 | 例 |
|---|---|---|---|
| pydantic | `VALIDATION_ERROR` | 英語（`msg`） | `"num_frames must be 8n+1"`・`"width must be a multiple of 64"`・`"distilled pipeline requires num_inference_steps=8"` |
| アダプタの拒否表 | `FEATURE_UNSUPPORTED` | message は英日併記、detail は日本語 | `"LTX 2.5は{feature}に対応していません(リクエストの{field}が既定値ではありません)。この機能を使うにはベースモデルに「LTX 2.3」を選んでください。"`（`ltx25/adapter.py:488-494`）。**どちらの系統も「もう一方の LTX を選べ」と書いている** |
| エンドポイントの数値検査 | 専用の符号（例 `REFERENCE_RESOLUTION_INVALID`） | 英語の message＋detail に値 | `api/errors.py:572-584` |
| 事前検査 | `MODEL_INCOMPATIBLE` | 英語の message＋日本語の detail | `api/errors.py:253`、系統不一致の文面は `services/engines/__init__.py:131-141`（`"このtransformerはltxv {version}（…）です"` と ltxv を前提） |

#### B3. 新項目を足したときの監査テストと表の書式

- 表の書式（事実）:
  - 拒否: `REJECT_TABLE` に `("field", "feature", lambda r: r.field != 既定)` を 1 行。
  - 無視＋ログ: `IGNORED_FIELDS = {"field": "理由"}`。2.5 は `_log_ignored` が既定と違うときだけ 1 行ログ（`ltx25/adapter.py:523`）。2.3 は宣言だけでログなし（`ltx/adapter.py:142-158`）。
  - 動作: `HONOURED_FIELDS` の frozenset に名前を足す。
  - 従属: `GOVERNED_FIELDS = {"field": "上位の field"}`（今は両系統とも空）。
  - 連結は `CHAIN_*` の 4 表が別にある（2.5 のみ監査あり）。
- 落ちるテスト（事実）:
  - `tests/test_ltx25_adapter.py:3113-3124`（単発の未分類検出）・`3156-3167`（連結）・`3137-3143`／`3178-3182`（表が実在しない項目を名指ししない）・`459-462`（拒否表と無視表が実在の項目だけ）・`502`（`REQUEST_OVERRIDES` が拒否表と一致）・`3304-3314`（動作の項目が `generate` のソースに `request.<field>` として現れる）。
  - `tests/test_ltx_adapter_audit.py:64-80`（2.3 の 4 表＝スキーマ）・`105-121`（動作の項目は読まれ、無視の項目は読まれない）・`125-126`（拒否表＝`REQUEST_OVERRIDES_23`）・`129-136`（拒否の detail に `"LTX 2.5"` が含まれる）・`143-145`（`UNSUPPORTED_FEATURES` の値が完全一致）・`154-168`（HTTP 経由の拒否）。
  - `tests/test_engine_dispatch.py:206-209`（2.3 の `unsupported_features` が完全一致）。
- 「LTX にとって使えない項目を受け取ったとき」の前例（事実）:
  - 2.5 専用の項目を 2.3 が受けた場合 → **拒否（422）**: `keep_resident_embeddings`・`alpha_gen`（`ltx/adapter.py:136-140`）。`unsupported_features` に載り、操作パネルは該当行を非表示・灰色にする。
  - 概念自体が無い項目 → **無視＋ログ**: 2.5 の `guidance_scale`・`num_inference_steps`（`ltx25/adapter.py:153-156`）、2.3 の `guidance_scale`・`pipeline`（`ltx/adapter.py:150-158`）。
  - 2.3 は `num_inference_steps` を**動作**に分類し、ワーカーへ `num_steps` として送る（`ltx/adapter.py:176-177`・`2236`）。今は pydantic が 8 に縛っているので実害がない（事実）。縛りを緩めると 2.3 に 8 以外が届く（推測ではなく、コード上そうなる）。
  - `tests/test_ltx25_adapter.py:802-830` は「2.5 が受理する要求では `num_inference_steps` が既定にしかなり得ない（distilled の縛りのため）」を前提に書かれている。

### C. 事前検査と記述子

#### C1. 事前検査

- `precheck_model_file`（`services/model_registry.py:118`）: 記述子の拡張子の門 → 拡張子が `.gguf`／`.safetensors` か（`_STRUCTURAL_CHECKS` 105）→ `.gguf` は `_precheck_gguf`（185。KV を読む）、`.safetensors` で `category == "transformer"` は `_precheck_sft_transformer`（200）、それ以外の `.safetensors` は `_precheck_safetensors`（223。長さと JSON の妥当性だけ）。例外は `APIError` を素通し、`OSError` を `MODEL_INCOMPATIBLE` に変換（176-181）。**それ以外の例外（`ValueError` を継承する `H3FormatError` を含む）は捕まえない**。
- 呼び出し元 `api/pipeline.py:121`（同じベースモデルで `default` のカテゴリは飛ばす）・`132`（precheck）・`145`（`engines.check_kv`）は R3 のとおり。
- LTX の各カテゴリの検査（事実）: transformer＝GGUF なら KV、safetensors なら `sft_quant_format.inspect`（LTX 固有の段を含む）。text_encoder・video_vae・audio＝GGUF なら KV を読むだけ（`check_kv` は transformer 以外を素通し、`services/engines/__init__.py:126-143`）、safetensors なら長さと JSON だけ。
- `h3_format.py` の関数（事実）:

| 関数 | 行 | 返り値／振る舞い |
|---|---|---|
| `check_model_paths(files, dirs)` | 61 | 存在確認。空は `ValueError`、無いと `FileNotFoundError`。ログ行のリストを返す |
| `parse_comfy_quant(raw)` | 86 | `QuantSpec(format, convrot, convrot_groupsize)` |
| `detect_h3_prefix(names)` | 127 | 接頭辞（4 通り `H3_PREFIXES` 45） |
| `is_h3_dit_header(header)` | 138 | bool（`video_patch_proj` と `audio_patch_proj` の同時存在） |
| `inspect_h3_dit(path)` | 293 | `(Header, H3DiTConfig)`。LTX の DiT なら「LTX の DiT です（MiniMax H3 の DiT ではありません）」で拒否（297-299） |
| `inspect_h3_te(path)` | 432 | `(Header, H3TEConfig)`。50 層ごとの形も検査 |
| VAE 用 | なし | ワーカーは `sft_quant_format.read_header` だけ（`pipeline_h3.py:357-358`） |

- 拒否の型は `H3FormatError(ValueError)`（50）。ヘッダが壊れているときは `QuantFormatError` を包み直す（`_read_header` 196-200）。`_quant_layers`（236）はヘッダの外の小さな `comfy_quant` バイト列も読む（1 ファイルハンドルで seek）。
- `transformer_weight_class`（`api/models_registry.py:79` → `services/weight_class.py:78` `classify_transformer`）は safetensors に `sft_quant_format.inspect` を使うので、H3 の DiT では失敗して `None`（INFO ログ）。R3 のとおり。

#### C2. 記述子

- schema の解析（`services/base_models.py`）: `assets` は「文字列の値を持つ object」（186-193）で、値は models_dir 相対の安全なパス。`comfort` は object かどうかだけ（210-212）。`_parse_category`（127）・`_parse_descriptor`（172）。
- `assets` の実例（事実）: `20-ltx25.json` は `spatial_upsampler_path` 1 件。`10-ltx23.json:33-38` は `gemma_root: "LTX23/TextEncoder/tokenizer"`（**フォルダ**）ほか。2.3 の `REQUIRED_ASSETS`（`ltx/adapter.py:85-`）に `gemma_root` があり、`_real_available` と `_require_asset`（1784）は `.exists()` で判定するのでフォルダでも通る。`downloads` 側は `"kind": "dir"` の `files` 行（`10-ltx23.json:143-150`）で受けている。インストーラは `assets` の値が `downloads[].files[].path` のどれかと一致することを要求する（`scripts/install_ltx.ps1:429-437`）。
- `comfort` を省いたとき: `GET /models` 自体に comfort は出ない。`GET /config` の `limits.comfort_budgets` に系統のキーが無くなる（`api/context.py:188-226`）。フロントエンドと Gradio が LTX の値へ落ちる点は R3 項目 5 のとおり（Gradio は `gradio_ui/comfort.py:106-121` で確認済み）。
- `downloads` の実例（`20-ltx25.json`）: `{"name", "repo": "Rootport/Nz-LTX25-weights", "include": ["Weights/*", "VAE/*", "Upscaler/*"], "map": [{"from": "Weights", "to": "LTX25/Weights"}, …], "files": [{"path", "min", "label", "key", "_min"}]}`。`migrate: []`。
- 機械にある実物（事実）: `models/MiniMaxH3/` に `Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors`、`TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors`、`TextEncoder/qwen3vl_tokenizer/`（config.json・merges.txt・preprocessor_config.json・tokenizer.json・tokenizer_config.json・vocab.json）、`VAE/minimax_h3_video_vae_fp16.safetensors`、`VAE/minimax_h3_audio_vae_fp32.safetensors`。VAE の 2 本は名前に `video`／`audio` を含むので、`name_hint` の仕組み（`model_registry.py:246-251`）でそのまま振り分けられる見込み（事実：文字列の包含判定）。
- `30-uetrack.json` は `schema: 1`・`opt_in: true`・`downloads` と `migrate` だけ。`categories`・`engine_family` を持たないのでベースモデルではない（事実）。
- ベースモデル記述子のカテゴリ順を固定するテスト（`tests/test_base_model_contract.py:56-58`）と、系統のアダプタの `SELECTION_FIELDS` と一致させるテスト（`tests/test_engine_dispatch.py:229-235`）は R3 のとおり。

#### C3. 系統の 3 辞書と設定

- `services/engines/__init__.py`: `FAMILY_BY_KV` 43・`FAMILY_BY_ID` 50・`FAMILY_DISPLAY` 58・`_require_family` 74・`check_kv` 106・`runner_class_for` 146・`unsupported_features` 157・`reject_unsupported` 167・`reject_chain` 189（R3 と一致）。
- `config.py`: `ModelConfig`（80）・`engine_dir`（108）・`engine_python`（112）・`engine_python_ltx25`（122）。`config.yaml.example` には `engine_python*` の行は**無い**（`model:` 節は `backend`・`manifest_dir`・`models_dir`・`ic_loras` など）。同ファイルの既定値の節に `num_inference_steps: 8`・`guidance_scale: 1.0`（160-161）がある。

### D. アダプタの骨格

#### D1. ltx25 アダプタの行（R3 と完全一致）

`SELECTION_FIELDS` 75・`REQUIRED_ASSETS` 100・`SUPPORTED_MODEL_VERSIONS` 105・`DEFAULT_BLOCKS_ON_GPU` 120・`REJECT_TABLE` 140・`IGNORED_FIELDS` 153・`HONOURED_FIELDS` 233・`GOVERNED_FIELDS` 296・`CHAIN_REJECT_TABLE` 318・`CHAIN_IGNORED_FIELDS` 329・`CHAIN_HONOURED_FIELDS` 396・`CHAIN_GOVERNED_FIELDS` 459・`UNSUPPORTED_FEATURES` 467・`reject_unsupported` 472・`reject_chain` 497・`_log_ignored` 523・`check_kv` 551・`_RealBackend25` 604・`_build_child_env` 629・`_build_load_payload` 673・`generate` 730・`generate_chain` 1109・`LTX25Runner` 1435。

- Alpha Gen の進捗: `generate` の中で `self._read_worker_events(progress_callback, chain=False, prefix="generate", one_stage=request.alpha_gen is not None and request.alpha_gen.one_stage)`（`ltx25/adapter.py:1022-1027`）。割合の関数 `_progress_frac`（`ltx/adapter.py:431-482`）は単発で `encode`→0.06、`stage1_denoise`→0.06＋0.84×進み（`one_stage` のとき）、`decode` は None（割合を動かさない。直前の値を保つ）。`done` を受けた後にアダプタが 0.90 と 1.0 を送る（`ltx25/adapter.py:1047-1049`）。
- `_read_worker_events`（`ltx/adapter.py:2695`）は `progress` 以外の最初のイベントを返す。`_STEP_STAGES` に `stage1_denoise` が入っている（375）。
- `GenerationOutcome`（`ltx/adapter.py:544-625`）は H3 の `done` の固有キー（`num_inference_steps`・`shift_*`・`commit_*`・`h3`）を受ける欄を持たない。2.5 は自分の固有情報を `ltx25: dict` 欄で丸ごと中継し、`_write_metadata`（`pipeline_manager.py:1855`、`1950-1951`）が `metadata["ltx25"]` に書く（前例）。`metadata.json` の `request` は `req.model_dump()`（同 1866）なので、要求に足した項目は全系統のジョブに既定値で載る（事実）。
- `_real_available`（`ltx/adapter.py:902-971`）: エンジンの python（`_engine_python_value`）・各カテゴリの `default_file`・`REQUIRED_ASSETS`・`<_engine_dir_value>/worker.py` の存在。欠けたら欠けたものを WARNING に列挙して mock へ落ちる（`backend: auto` のとき。`_select_backend` 876）。
- `_RealBackend.load`（`ltx/adapter.py:2020-2104`）は汎用: `_engine_python_value`・`_engine_dir_value`・`_build_load_payload`・`_build_child_env`・`_LOG_NAME`・`_WORKER_MODULE` を差し替えれば流用でき、`ready` の `sage_available` を拾う。
- H3 ワーカーの load と generate のキー（`engine_h3/worker.py:22-41`・`139-146`）: load は `transformer_path`・`text_encoder_path`・`video_vae_path`・`audio_vae_path`・`tokenizer_dir`（必須）＋任意の `blocks_on_gpu`・`te_layers_on_gpu`・`attention_backend`・`block_swap_prefetch`・`deterministic`。**H3 では `attention_backend` と `block_swap_prefetch` が load 時の値**（1 プロセス 1 値。`_ATTENTION` 158、`H3LoadOptions` `pipeline_h3.py:84-103`）で、LTX のように generate ごとに切り替わらない（事実）。`GenerateRequest` 側ではこの 2 つはジョブごとの項目。
- **I2V の受け口は H3 ワーカーにまだ無い**（事実）: `_GENERATE_FIELDS` に画像のキーが無く、`_denoise` は `cond_rows=0`・`cond_frames=0` 固定（`pipeline_h3.py:567-572`）。`H3VideoVAE.encode`（`video_vae.py:567`）はあるが「第 1 弾では未使用・未検証」と明記。

#### D2. mock

- `_MockBackend.generate`（`ltx/adapter.py:1125-1265`）→ `_render_frames`（1587）: 幅・高さ・`num_frames`・`frame_rate` を要求からそのまま使う。I2V は先頭の画像を引き伸ばして少しずつ寄る、T2V はグラデーション、どちらも球を描く。`num_inference_steps` は進捗の刻みにだけ使う。8n+1 の算術は単発の mock には無い（連結の mock は `chain_math` を使う。R3 はこの区別をしていない）。
- mock の出力は `video_io.encode_frames_to_mp4`（アプリ側の ffmpeg）で、**音声は付かない**（事実：呼び出しに音声の引数なし）。
- mock は Runner のクラス属性 `_MOCK_BACKEND_CLS`・`_MOCK_BACKEND_LABEL` で決まる（`ltx/adapter.py:896-900`）。2.3 と 2.5 が同じクラスを共有することをテストが固定（`tests/test_ltx25_adapter.py:3322-3328`）。
- H3 の規約で mock の絵を作るのに触る箇所（事実に基づく見込み）: 単発なら mock 本体は不要。要るのは「17n+5・32 の倍数・24fps の要求が pydantic を通ること」と、H3 の Runner に `_MOCK_BACKEND_LABEL` を与えること。

### E. Gradio と MCP

#### E1. Gradio

- スライダー: `gradio_ui/ui.py:547-550`（Single：ステップ数 1〜50 既定 8、CFG 1.0〜12.0 既定 1.0、どちらも `interactive=False`）、`876-880`（Chain：同形）。下の `cap_lock` の説明文（551）。送信側は固定値（`handlers.py:604-605`・`991-992`・`1495-1496`）。スライダーの値をどの入力リストも参照していない（`steps`／`cfg`／`chain_steps`／`chain_cfg` の参照は定義行だけ）。
- 8n+1・64 の倍数・fps の記述: `handlers.py:135`（`_snap_frame_rate`）・`189-207`（音声長からの 8n+1 の提案）・`747-765`（Single の事前検査：64 の倍数、9〜481 の 8n+1）・`1234`（Chain の 64）・`1319-1338`（Chain の 8n+1）、`ui.py:2335-2350`（数値欄の `min` 属性：幅高さ 64、フレーム 9 を基準にする JS）・`ui.py:1460`、`validation.py:78`、`i18n.py:143`／`689`（ラベル「フレーム数 (8n+1)」）・`158`／`703`・`299`／`831`ほか。
- 快適上限: Chain の Stage-2 窓のラベル用 `resolve_chain_budget`（`gradio_ui/comfort.py:93-121`）は表が無い系統ではスカラ値へ落ちる。Single の表は `/config` の `limits.spill_free_frames` から作る（`ui.py:101` `build_spill_rows`、`presets.py:64-73`）。
- 機能スコープ: `gradio_ui/feature_scope.py` の `FEATURE_UI`（`prune_vaed`→`accel_vae`、`keep_resident_embeddings`→`accel_keep_resident_embeddings`）。**隠す**＋`RESET_VALUES` で既定値へ戻す（送信値が残らないように）。未知の機能名は無視。適用は `ui.py:2430-2434`。操作パネルの `featureScope.ts` は灰色化（R3 項目 7）なので、2 つの画面で方式が違う（事実）。
- ベースモデルの選択肢は `gradio_ui/adapters.py:104` `build_base_model_choices`（`GET /models` の `base_models[]` から自動）。

#### E2. MCP

- `mcp_server/params.py`（59 行）は引数の形だけで検証なし。8n+1 は `ConditioningImageArg` の説明文（24）だけ。
- `mcp_server/tools/generate.py`: `submit_generate`（141）は `num_inference_steps`・`guidance_scale`・`pipeline` を**意図的に出さない**（冒頭 4-9 の説明：distilled で固定値に縛られているため）。`num_frames: int = 49`（152）、`frame_rate: float = 24.0`（153、`_snap_frame_rate` 103 で整数へ丸める）、幅 512・高さ 320 が既定。説明文に「`num_frames` は 8n+1（9, 17, 25, ... 481）」（223・287）。常駐は `keep_resident`（164、既定と違うときだけ送る 463-464）。
- ベースモデルの選び方: `mcp_server/tools/system.py` の `list_models`（75）と `load_pipeline(base_model=...)`（111-185）。サーバーの案内文 `mcp_server/server.py:26-29` が `"LTX25"` を例に挙げる。
- 「最小対応」で T2V／I2V を通すのに要るもの（事実からの見込み）: API が 17n+5・32 の倍数・ステップの既定を受けられるようになれば、MCP は引数をそのまま送るので追加の引数は不要。ただし既定の `num_frames=49` は 17n+5 でないので、利用者が値を指定しないと 422 になる。説明文の 8n+1 の記述は H3 では誤り。ステップ数・shift を MCP に出すかは分岐点（④）。
- MCP のテスト: `tests/test_mcp_tools_generate.py`（ペイロードの組み立て）・`tests/test_mcp_registration.py:26-62`（ツール名の集合）。

### F. 復号 VRAM

#### F1. 復号の経路（`engine_h3/video_vae.py`）

| 関数 | 行 | 何をするか | dtype と置き場 |
|---|---|---|---|
| `H3VideoVAE.from_file` | 423 | meta で組み、ファイルから読み、GPU へ移して bf16 にキャスト（島は float32） | 重み GPU |
| `decode` | 601 | 潜在（float32）に `std`・`mean` を戻す → `_decode_temporal`（T'=1 は特例）→ ImageNet 正規化を戻して `clamp_(0,1)` | 戻り値 float32 `[1,3,F,H,W]`・GPU |
| `_decode_temporal` | 620 | 5＋2 潜在の時間窓を回す。各窓の `_decode_clip` の結果を `parts` に溜め、前の窓の重なり（5 フレーム）と線形に混ぜる（`blend`）。最後に `torch.cat(parts, dim=2)` | 各窓 float32・GPU。**全窓を GPU に溜めてから連結** |
| `_decode_clip` | 539 | 1 窓を 256px のタイルに分けて `decoder` を呼び、`.float()` してから `_stitch` | float32・GPU |
| `_stitch` | 506 | タイルを混ぜて行ごと・全体を `torch.cat` | float32・GPU |
| `blend` | 493 | 重なりの線形クロスフェード | 入力と同じ |

- パイプライン側（`engine_h3/pipeline_h3.py`）: `_decode`（631）が `vae.decode(video_lat, tiled=True)`（`video_lat` は denoise の結果で GPU・float32）→ `frames_to_uint8(pixels)`（182）＝`pixels[0].permute(1,2,3,0).float().clamp(0,1)`→`×255`→`round`→`uint8`→`.cpu()`。**uint8 化は GPU 上で、CPU へ移るのは最後**。
- `write_mp4`（`engine_h3/media_out.py:81`）: `frames` は uint8 `[F,H,W,3]`・CPU 必須、`audio` は float32 `[2,N]`・CPU 必須（違えば `ValueError`）。内部で 16 フレームずつ BT.709 の YUV へ変換（`_CONVERT_BATCH` 50、`rgb_to_i420_bt709` 53）。
- 音声: `H3AudioVAE.from_file`（`audio_vae.py:217`。GPU・float32）→ `decode`（232）が全潜在を一度に BigVGAN へ通す（窓なし）→ パイプラインで `.float().cpu()`（`pipeline_h3.py:651`）。
- 窓の構造上の事実: 各窓の 1 つ目の片（j==0）は、前の窓の重なりと混ぜた時点で確定する（`_decode_temporal` 内 `parts.append(piece)`）。2 つ目の片（重なり）は次の窓まで保持される。17n+5 の要求では `pad_tokens` は 0（T'+3＝5n+5 が 5 の倍数）なので末尾の切り詰めは起きない（算術上の事実）。正規化の戻し（`pixels*ps+pm`）と `clamp` は今は連結後に一括。
- 大きさの推測（実測ではない）: 1344×768・345 フレームの float32 画素は約 4.3GB。`torch.cat` で一時的に 2 倍、`frames_to_uint8` の `permute(...).float().clamp` と `×255` でさらに float32 の写しが作られるので、16GB を超えうる（設計書 §13 の記述と整合）。
- 「窓ごとに uint8 にして CPU へ移す」で触る関数の一覧（場所の特定のみ）: `H3VideoVAE.decode`（601）・`H3VideoVAE._decode_temporal`（620）・（窓の中の `_decode_clip` 539／`_stitch` 506 は窓 1 つ分の大きさなので触らずに済む可能性。推測）・`pipeline_h3.frames_to_uint8`（182）・`H3Pipeline._decode`（631）。`write_mp4` は既に CPU の uint8 を受けるので変更不要の見込み（事実：入力の契約がそうなっている）。T'=1 の特例（`decode` 内 615-617）も同じ戻り値の契約に乗っている。

#### F2. 既存テストが固定していること（`tests/test_h3_media.py`）

- `test_decode_frame_count_and_windows`（52-69）: `decode` の戻り値の形 `(1,3,F,16,16)`、窓の数（22→1、124→7、345→20）、各窓 7 潜在、**`out.dtype == torch.float32`**、値が [0,1]。
- `test_decode_single_latent_gives_one_frame`（71-83）: 形 `(1,3,1,32,32)` と、5 潜在の 1 窓で復号すること。
- `test_decode_two_latents_refused`（86-88）。
- `test_islands_float32_without_bf16_round_trip`（125-149）: 島の dtype。
- `test_audio_decode_length`（155-169）: 音声の戻り値 float32。
- 配置（CPU か GPU か）を固定するテストは無い（すべて CPU で回る）。`write_mp4` のテスト（209-275）は CPU の uint8 を渡す。
- `tests/test_h3_engine.py:618-645`（`test_golden_done_keys`）は `_decode` を差し替えている（`lambda req, v, a, entry: 1234`）ので、`_decode` の引数の形を変えると影響する。

### G. 落ちるテストと実行方法

#### G1. R3 の一覧の確認

| R3 の記載 | 今 | 判定 |
|---|---|---|
| `test_engine_dispatch.py:139`・`141` | 138-141（`test_the_three_family_tables_agree`、`FAMILY_BY_KV` 完全一致は 141） | 確認済み |
| `test_engine_dispatch.py:206-210` | 206-209 | 確認済み |
| `test_engine_dispatch.py:233-235` | 229-235（`test_shipped_descriptors_match_their_familys_selection_fields`） | 確認済み |
| `test_comfort_budgets.py:69` | 64-69（`test_family_id_set_matches_engine_registry`） | 確認済み |
| `test_base_model_contract.py:56-58` | 56-58 | 確認済み |
| `test_base_model_contract.py:84-89`（拡張子） | **87-94**（`test_shipped_transformer_extensions_accept_fp8_safetensors`）。84-89 は LTX23 の assets のテスト | ずれあり（影響なし。名指しのテスト） |
| `test_model_swap_load.py:317` | 314-317（`_shipped_ltx23`） | 確認済み |
| `conftest.py` 80・117・156 | 同じ | 確認済み |
| `test_worker_ltx25_unknown_op.py:28-29` | 同じ | 確認済み |

#### G2. 第 2 弾で追加・変更が要りそうなテストファイル（事実に基づく一覧。要否の判断は計画で）

- 系統の登録で落ちる: `tests/test_engine_dispatch.py`（138-141・206-209・229-235）、`tests/test_comfort_budgets.py`（64-69。`comfort` を持たない系統の登録で落ちる）、`tests/test_base_model_contract.py`（56-58。カテゴリの順）。
- 要求の項目を足す・縛りを緩めると動く: `tests/test_ltx25_adapter.py`（459-462・502・802-830・3113-3182・3304-3314）、`tests/test_ltx_adapter_audit.py`（64-168）、`tests/test_validation.py`（8n+1・64 の文言、14-50）、`tests/test_gradio_handlers.py`（1315-1322・1783-1792。サーバーの 422 の文言を表示する部分）、`tests/test_mcp_tools_generate.py`、`tests/test_chain.py`（296・302）、`tests/test_smoke.py` ほか `num_inference_steps: 8` を本体に入れているテスト群（`tests/test_smoke.py:167` ほか多数）。
- H3 のワーカーを触ると動く: `tests/test_h3_engine.py`（黄金キー 575-589・610-645）、`tests/test_h3_media.py`（52-69）。
- 新設の候補（前例のある形）: H3 アダプタのテスト（4 分類の監査は `test_ltx_adapter_audit.py` が雛形）、H3 の load ペイロードの黄金（`test_model_swap_load.py` が雛形）、3 系統の HTTP 往復（`conftest.py:247-288` の `two_family_client`（248） が雛形。ベースモデル切り替え時は precheck が走るので、H3 の DiT のヘッダの偽物が要る。`tests/test_h3_engine.py` の `_write_sft`／`_tiny_dit_tensors` と `tests/test_h3_format.py` に作り方がある）。

#### G3. 実行方法

- アプリ側: `.\.venv\Scripts\python.exe -m pytest tests\<ファイル>`。`pyproject.toml:32-34` で `testpaths = ["tests"]`・`addopts = "-q"`。`tests/conftest.py` が mock を強制（156・172）。
- エンジン側（H3）: `.\.venv-engine-h3\Scripts\python.exe scripts\run_h3_pytest.py tests\test_h3_engine.py tests\test_h3_media.py`。`scripts/run_h3_pytest.py` はアプリの venv の site-packages を後ろに足して pytest を借り、`--noconftest -p no:cacheprovider` で回す。引数なしなら `tests/test_h3_engine.py` だけ。`test_h3_engine.py` は `torch` と `comfy_kitchen` が無ければ丸ごと飛ぶ（38-39）。GPU のテストは `H3_GPU_TESTS=1` かつ CUDA ありのときだけ（442-445）。
- `tests/test_h3_format.py` は torch を import しないので、アプリ側の venv で回る。

---

## ③ R3 からずれていた点

1. `tests/test_base_model_contract.py` の拡張子のテストは 84-89 ではなく 87-94（`test_shipped_transformer_extensions_accept_fp8_safetensors`）。
2. `PipelineManager.active_models` の形を固定している行は 217 ではなく 216（`services/pipeline_manager.py:215-217`）。実質同じ。
3. R3 項目 6・(3)-8 は「mock が LTX の規約（8n+1・`chain_math`）で絵を作る」としているが、**単発の mock は 8n+1 を自分では使わない**（要求の値をそのまま使う）。8n+1 の強制は pydantic 側。`chain_math` を使うのは連結の mock だけ。
4. R3 項目 7 は API の制約として 8n+1・64・fps・範囲を挙げているが、**`pipeline == "distilled"` のときステップ数 8・CFG 1.0 を強制する検証**（`api/models.py:657-664`・`1345-1352`）が抜けている。ステップ数を API に出す第 2 弾ではこれが最初の障害になる。
5. R3 には、H3 ワーカーの `attention_backend`・`block_swap_prefetch` が load 時の値（1 プロセス 1 値）であることの記載が無い（第 1 弾の実装で決まった事実。`engine_h3/worker.py:158`・`pipeline_h3.py:84-103`）。
6. R3 項目 4 の「H3 用の判定を足すには呼び出しの形の変更が要る」に加えて、`precheck_model_file` が `H3FormatError`（`ValueError`）を捕まえないので、変換を足さないと 500 になる。

それ以外の行番号（`ltx25/adapter.py` の全定数・関数、`ltx/adapter.py` の 902・1060・2062・2695、`services/engines/__init__.py` の全行、`model_registry.py` の 63・75・118・200・216、`api/pipeline.py` の 121・132・145、`api/models.py` の制約の全行、`config.py` 112・122、`api/context.py:188`、`chain_math.py:446`、`pipeline_manager.py` の 182・251・563・1855）はすべて確認済みで一致。フロントエンドの行番号（R3 項目 5・7）は本調査の範囲外のため再確認していない。

---

## ④ 計画で決めなければならない分岐点（選択肢の列挙のみ）

1. **ステップ数の項目の扱い**: (a) 既存の `num_inference_steps` を H3 でも使い、distilled の「8 固定」の検証を何らかの形で緩める／(b) H3 専用の別名の項目を新設する。(a) の場合、2.3 は今「動作」に分類して `num_steps` としてワーカーへ送っているので、緩めると 2.3 に 8 以外が届く。2.3 の扱いを「8 以外は拒否」「無視」「そのまま送る」のどれにするか。範囲（今は 1〜100、裁定は 1〜50）と既定（今は 8、H3 は 20）をどう両立させるか。
2. **shift 2 本の LTX での扱い**: 無視＋ログ（2.5 の `guidance_scale` の前例）／拒否（2.3 の `keep_resident_embeddings` の前例。`unsupported_features` に載り画面が閉じる）。連結の要求（`GenerateChainRequest`）・Alpha Gen の要求にも足すか（足せば連結の監査も動く）。
3. **系統別の数値規約（17n+5・32 の倍数・24fps・22 フレーム以上）をどこで判定するか**: (a) pydantic の検証を「どの系統でも通りうる値」に広げ、系統ごとの判定をエンドポイントの冒頭（`reject_unsupported` の直後）かアダプタに置く／(b) 要求に系統（またはベースモデル id）を持たせて pydantic で分岐／(c) pydantic の context で現在の系統を渡す（前例なし）。その際の 422 の符号（`VALIDATION_ERROR`／`FEATURE_UNSUPPORTED`／新しい符号）と文言の言語。LTX 側の既存の英語の文言・テスト（`test_validation.py` など）を変えずに済ませるか。
4. **8n+1 の格子へのスナップ**（`_normalize_conditioning_images`）: H3 の I2V は先頭フレーム（0）だけなので触らずに済むか、H3 では 0 以外を拒否するか。
5. **アダプタの拒否文言**: 「もう一方の LTX を選べ」の文言（2.3・2.5）を、3 系統になった後もそのままにするか。H3 側の拒否文言の向き。`check_kv` の「ltxv {version}」前提の文面と、`FAMILY_BY_KV` に H3 を載せるか（疑似 KV を何にするか）。
6. **事前検査で H3 をどう判定するか**: (a) `precheck_model_file` に系統（または記述子）を渡して分岐／(b) 拡張子と記述子だけで分け、H3 の DiT とテキストエンコーダに `inspect_h3_dit`／`inspect_h3_te` を呼ぶ／(c) 事前検査は汎用のままでワーカーの load に任せる。`H3FormatError` を `MODEL_INCOMPATIBLE` に変換する場所。VAE（H3 用の判定関数なし）とテキストエンコーダを事前検査の対象にするか。
7. **tokenizer の渡し方**: 記述子の `assets`（2.3 の `gemma_root` と同じフォルダ方式、`downloads` は `"kind": "dir"`）を `REQUIRED_ASSETS` に入れて `_real_available` の門にするか。
8. **`attention_backend`・`block_swap_prefetch` の扱いの差**: 要求ではジョブごと、H3 では load 時。(a) 値が変わったらワーカーを作り直す／(b) H3 では load 時の設定値（設定ファイルや既定）を使い、要求の値は無視＋ログ／(c) H3 のワーカーをジョブごとに受けられるよう直す。
9. **常駐の中身と既定**: 「DiT と VAE を主メモリに保持し、テキストエンコーダは毎回読む」を、既存の `keep_resident` の同名の項目に載せるか（2.3 と 2.5 も同名で中身が別、という前例どおり）、別名にするか。ブロック以外の GPU 常駐分・VAE の置き場をジョブの合間にどうするか。同じ DiT に `install` し直すときの二重包みの対策（2.5 の `_unpatch_block_swap` に倣うか）。操作パネルが常駐を「先読みが有効なときだけ」送る規則（`keepResidentEffective`）を H3 でも適用してよいか。こだま `keep_resident_used` を `done` に足すか。
10. **常駐の解放の契約**: 既存どおり「キーが無ければ解放」にするか。ベースモデル切り替え時はプロセス終了で自然に解放されるので追加の処理は不要か。
11. **`done` の H3 固有の値（`num_inference_steps`・`shift_*`・`commit_*`・`h3.phases`）を `metadata.json` に載せる経路**: `GenerationOutcome` に `h3` 欄を足す（2.5 の `ltx25` 欄の前例）か、汎用の欄にするか。
12. **I2V の受け口のキー名**: LTX と同じ `images`（`{path, frame_idx, strength}` のリスト）にするか、H3 専用の形にするか。設計書 §8 の generate のキー一覧にはまだ無い。
13. **mock**: H3 も 2.3 の `_MockBackend` を共有して label だけ変えるか（共有を固定するテスト `test_ltx25_adapter.py:3322-3328` は ltx／ltx25 だけを見ている）、音声なしの mock で足りるか。
14. **快適上限マーカーを出さない方法**: `comfort` を省いたうえで、`test_comfort_budgets.py:64-69` を変えるか、空の `comfort` を持たせるか、フロントエンドと Gradio の「表が無いときはスカラ値へ落ちる」挙動を系統で止めるか。
15. **Gradio の 3 本のスライダー**: 既存の飾りのスライダー（`ui.py:547-550`）を H3 選択時だけ差し替えるか、常設するか。Gradio の機能スコープは「隠す」方式なので、LTX 選択時に shift を隠す形にするか。Chain タブ側（876-880）を触るか。
16. **MCP の最小対応の範囲**: 説明文の 8n+1 の修正と `num_frames` の既定だけにするか、ステップ数と shift を引数に出すか（出さないと MCP からは既定値でしか動かない）。
17. **復号の直し方の範囲**: `decode` の戻り値の契約（今は GPU 上の float32 `[1,3,F,H,W]`、テストが float32 を固定）を変えるか、新しい関数を足して既存の `decode` を残すか。音声の復号（窓なし）も対象に含めるか。
18. **進捗の `decode` 段**: 単発の割合計算は `decode` で動かない（0.90 で止まって見える）。H3 は復号が長いので、そのままにするか。

---

## ⑤ 未確認事項

- フロントエンド（`webui/`）の R3 記載の行番号（`comfortTable.ts`・`useGenerationForm.ts`・`paramUtils.ts`・`mockBridge.ts`・`featureScope.ts`）は再確認していない。
- 2.3 ワーカーが `keep_resident` で抱える骨格の正確な範囲（`engine/worker.py` の `_resolve_keep_resident` の先、`StateDictRegistry` の中身）は説明文の範囲でしか読んでいない。
- H3 の DiT・VAE を主メモリに保持したときの主メモリと VRAM の量は未測定。
- 1344×768・345 フレームの復号の VRAM の山は推計のみ（実測なし）。音声の BigVGAN を長い尺で一度に通すときの中間の VRAM も未測定。
- `GenerateRequest.num_inference_steps` の縛りを緩めたときに落ちるテストの網羅（`num_inference_steps` を含むテストファイルは 20 本以上あり、個々の期待値までは読んでいない）。
- `api/generate.py` の `reject_unsupported` より後の処理（参照動画・LoRA・inpaint など）のうち、H3 で到達しうるものの全洗い出し。
- `Comfy-Org/MiniMax-H3` リポジトリのゲートの有無と、`downloads` に書くファイル名の正式なもの（R3 から据え置き）。
- Gradio で 3 本のスライダーを差し替えたときの i18n・テスト（`tests/test_gradio_*`）への影響範囲。
