# MiniMax H3 第 2 弾のあとの文書の棚卸し（2026-10-18・dev `cc38a67` 時点）

対象: リポジトリ `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`（ブランチ dev。main からの差分 `10f1674`〜`f67e01f`＋`cc38a67` aux2）。読むだけで、リポジトリは編集していない。
事実の根拠は生きたコード（`engine_h3/`・`services/engines/h3/adapter.py`・`services/engines/ltx/adapter.py`・`services/engines/ltx25/adapter.py`・`services/engines/__init__.py`・`services/model_registry.py`・`services/weight_class.py`・`gradio_ui/constraints.py`・`mcp_server/tools/generate.py`・`mcp_server/server.py`・`webui/src/shell/featureScope.ts`・`scripts/manifests/40-minimaxh3.json`・`config.py`）。行番号は dev の現物で数えた（直し担当は引用文で探すこと）。

---

## ① 要約

**高（この文を信じて作業すると間違える）**
- MCP の設計書 D8 が「`num_inference_steps` はツールに出さない」のまま。実物は `submit_generate` に `num_inference_steps`・`shift_video`・`shift_audio` がある（D-001）。
- MULTI_ENGINE_DESIGN §7 が見出しも本文も「エンジン別の数値の制約は作らない【オーナー裁定】」「現在の 2 系統は LTX 系なので必要ない」のまま。訂正の注記だけが挟まっている（D-002）。
- MULTI_ENGINE_DESIGN の「§5.6 (4)」（`constraints` の形と H3 の `unsupported_features` の導出規則）が §5.6 ではなく §6.5 の後ろに置かれており、各所の「§5.6 (4)」参照が空振りする（D-003）。
- MULTI_ENGINE_DESIGN §5.3 が「全エンジンで torch の版を揃える【オーナー裁定】」を本文に残し、H3 の例外と `.venv-engine-h3` は訂正の注記だけ（D-004）。
- 仕様書 §6.10(d) の LTX 2.5 の 4 分類の件数（全 32・無視 2）が古い。shift の 2 項目が増えて全 34・無視 4（D-005）。
- 台帳 §1-95 の前提「較正の前に大きな画角での復号の VRAM を直す」は第 2 弾で済んでいる（D-006）。
- 操作パネル側 API_REFERENCE §5.1 の `GenerateRequest` 表が、64 の倍数・8n+1 を `api/models.py` の検査と書き、`num_inference_steps` を「既定 8・変えると 422」とし、`shift_video`／`shift_audio` が無い。§3.5 にも `constraints` が無い（D-007）。

**中（紛らわしい・不足）**: 仕様書の事前検査の段落が H3 の分岐を書いていない（D-008）／仕様書の venv・設定キー・ディレクトリ・用語集が 2 系統のまま（D-009）／仕様書の Gradio の段落が半分だけ更新（D-010）／付録 A と README の「検証は `api/models.py`」（D-011）／H3 の 4 分類が仕様書のどこにも無い（D-012）／`transformer_weight_class` が H3 では常に `null` であることが書かれていない（D-013）／COMFORT_LIMIT_TABLE の「行が無いときの落ち先」が H3（表そのものが無い系統）に当たらない（D-014）／MULTI_ENGINE の §3・§4・§5.2・§9 が 2 系統の書きぶりのまま（D-015）／MULTI_ENGINE の訂正注記の畳み込み（D-016）／設計正本 §13・§11 が §1-100・§1-101 と第 2 弾の実測を指していない（D-017・D-018）／記録 §166.2 の「再試行中」（D-019）／台帳 §1-93 の「4〜15 秒」（D-020）／DEVLOG §141.5 の「未コミット」（D-021）／右クリック設計書の快適上限による尺の決め方（D-022）／研究ノート §13.7 の `install-H3.bat` と TE の層オフロード（D-023）／台帳 §1-93 の未来形（D-024）。

**低（言い回し）**: 研究ノート §10・§13.8 の「移管予定」「計画のセッションで決める」（D-025・D-026）、設計正本の未来形と「短辺 768 が既定」（D-027）、保管 README の題と手順書（D-028）、API_REFERENCE の最終更新日（D-029）、MODEL_MANAGEMENT_DESIGN の注記（D-030）、仕様書の連結の 8n+1 の一般化（D-031）。

---

## ② 一覧

### D-001【高】MCP_SERVER_DESIGN.md §2（判断の表）D8 — ステップ数を「出さない」のまま
- 場所: `Docs/MCP_SERVER_DESIGN.md` 22 行目 D8。
- 古い文: 「隠しフィールド（`pipeline` / `num_inference_steps` / `guidance_scale` / `crf` 等、パネルにも出していない検証専用フィールド）はツールのスキーマに出さない」
- 正しい事実: `mcp_server/tools/generate.py` の `submit_generate` に `num_inference_steps: int | None = None`・`shift_video`・`shift_audio` があり、`None` なら送らない（479〜行）。`mcp_server/server.py` の `INSTRUCTIONS` も「H3 では … num_inference_steps / shift_video / shift_audio（省略で既定 20 / 12 / 3）が選べます」「フレーム数・画角・fps の決まりは … base_models[].constraints が正本」と書く。記録は VERIFICATION_LOG §166（コミット `cdaa51c`）。D 表に第 2 弾の行（D27 相当）が無い。
- 差し替え案: D8 の括弧から `num_inference_steps` を外し「`num_inference_steps` は MiniMax H3 のために公開した（D27）」と一言。D 表の末尾に D27「`submit_generate` に `num_inference_steps`・`shift_video`・`shift_audio` を足す（`None` は送らない＝系統の既定値）。`INSTRUCTIONS` のフレーム規約の説明を `constraints` 参照に改める。ツール本数は不変」。§9 のテスト表の `test_mcp_tools_generate.py` の「隠しフィールド不在」も「`pipeline`／`guidance_scale` の不在、3 項目は `None` のとき不在」に。

### D-002【高】MULTI_ENGINE_DESIGN.md §7 と §5.6 冒頭 — 「エンジン別の数値の制約は作らない」が本文のまま
- 場所: `Docs/MULTI_ENGINE_DESIGN.md` 571 行目の見出し「## 7. スコープ外 — エンジン別のUIパラメータ制約 【オーナー裁定】」、573 行目「**エンジンごとに違うパラメータ制約（…）をUIが切り替える機能は、まだ作らない。**」、577 行目「現在の2系統はどちらも LTX 系なので、フレーム数の `8n+1` 規約…を共有しており、必要になっていない。」、575 行目と 586 行目の【2026-10-17 訂正】の段落。§5.6 の冒頭（398 行目）「§7が「エンジン別のUIパラメータ制約は作らない」と決めているのはパラメータの**刻みや倍数**の話であって」。
- 正しい事実: 系統別の数値の規約は実装済み。`services/engines/__init__.py` の `validate_for_family` が `reject_unsupported` の後に規則表で検査し、`GET /models` の `base_models[].constraints` で配る。操作パネル（`generationConstraints.ts`）・Gradio（`gradio_ui/constraints.py`）・MCP がこれを読む。第 3 段階は「着手」（同 594 行目）。
- 差し替え案: 見出しを「## 7. エンジン別の数値の規約（フレーム数の刻み・解像度の倍数・fps・ステップ数）」に。本文は現在形で「系統が分かった時点で、その系統の規則表で検査する（`validate_for_family`）。違反は pydantic と同じ形の 422。`GenerateRequest` は系統を問わない形の検査だけ。`null` の既定は同じ関数が系統の既定値へ書き戻す。配り方は §5.6 (4)」。「現在の2系統は…必要になっていない」の段落と、`chain_math.py` の定数が散らばっていた話（訂正の段落を含む）は「LTX の規則は `services/engines/ltx/request_rules.py` に集約し、`GenerateRequest` の `Field` もそこを参照する」の 1 文へ畳む。§5.6 冒頭の「§7が…作らないと決めている」は「刻みや倍数は §7 の規則表、機能の有無は本節」に。オーナー裁定の扱いは ③ J-6。

### D-003【高】MULTI_ENGINE_DESIGN.md「(4) 系統別の数値の規約は…」の置き場所
- 場所: 同書 551〜569 行目。`### 6.5 ヘッダーの状態バッジ` の後ろ、`---` の直後、`## 7.` の直前にある。
- 何が問題か: §7（575 行目）・仕様書 §6.2b（811・851 行目）・同 (f) が「§5.6 (4)」を参照しているが、§5.6 には (1)〜(3) しか無い。§6 の中に浮いているので、§5.6 を読んだ人は `constraints` の形と H3 の `unsupported_features` の導出規則（`chain`・`retake` を足す理由、`v2v` などを公開しない理由）を見落とす。
- 差し替え案: この塊（`constraints` の JSON 例・`frame_rate` の min==max・`sampling` の null・導出規則・「閉じた機能の項目は要求に載せない」）を §5.6 の (3) の直後へ移す。§6 の末尾の `---` は 1 本に戻す。

### D-004【高】MULTI_ENGINE_DESIGN.md §5.3 — torch を揃える方針と venv 一覧
- 場所: 同書 §5.3 のコードブロック（`.venv` / `.venv-engine` / `.venv-engine-ltx25` / `.venv-wan`）、表（ltx／ltx25 の 2 列）、「インストーラ側は…`Ensure-EngineVenv`…を**2回呼ぶ**」、340 行目「**方針: 全エンジンで torch のバージョンを揃える 【オーナー裁定】。**」、342 行目の「訂正（2026-10-17）」注記。§3.2（「専用のワーカープロセス（`engine/worker.py`・`engine25/worker.py`）」「専用の Python 仮想環境（`.venv-engine`・`.venv-engine-ltx25`）」）。
- 正しい事実: `.venv-engine-h3` が 3 つ目のエンジン用 venv（`config.py` 128 行目 `engine_python_h3 = "./.venv-engine-h3/Scripts/python.exe"`）。torch は LTX に揃えない（裁定 11・16。設計正本 MINIMAX_H3_DESIGN §10。CUDA 13 系）。設計正本 §10 は「MULTI_ENGINE_DESIGN §5.3 の一覧に並ぶ 3 つ目」と書いているが一覧に無い。`Ensure-EngineVenv` の 3 回目は第 3 弾（未実装）なので「2 回」は今は正しい。
- 差し替え案: コードブロックに `.venv-engine-h3 … h3 エンジン系統（MiniMax H3）。CUDA 13 系の torch（版は engine_h3/venv-engine-h3.freeze.txt）` を足す（版の数字は書かない）。方針の段落は「同じ torch の版で足りる系統は揃える（uv のハードリンク共有が効く）。揃えられない系統は、容量が増えることを承知でオーナー判断を仰ぐ。MiniMax H3 がその例で、裁定 11・16 により揃えない（理由と容量は MINIMAX_H3_DESIGN §10）」へ現在形で畳み、訂正注記は消す。表は 3 列目 h3 を足すか、「h3 は MINIMAX_H3_DESIGN §10」の 1 行に。§3.2 の 2 つの括弧にも h3 を足す。

### D-005【高】仕様書 §6.10(d)（と MULTI_ENGINE §5.6 の 4 分類の表）— LTX 2.5 の「無視＋ログ」の件数
- 場所: `Videomni_Backend_Specification.md` 1607 行目「#### (d) `GenerateRequest` の各フィールドの扱い（4 分類・全 32 件。…422 系 2・無視 2・動作 28・従属 0…）」、1615 行目の表「**無視＋ログ** | 2 | `guidance_scale`／`num_inference_steps`」。`Docs/MULTI_ENGINE_DESIGN.md` §5.6 の 4 分類の表「**2026-08-24以降ここに残っているのは1種類だけ**——蒸留版2.5にはCFG…もステップ数の概念も無い、という理由の項目である（`guidance_scale`／`num_inference_steps`）」。
- 正しい事実: `GenerateRequest` に `shift_video`・`shift_audio` が増え、`services/engines/ltx25/adapter.py` の `IGNORED_FIELDS` は `guidance_scale`・`num_inference_steps`・`shift_video`・`shift_audio` の 4 件（理由は「the distilled schedule has no shift knob」）。全 34 件・422 系 2・無視 4・動作 28・従属 0。MULTI_ENGINE §5.6 は「件数の正本は仕様書 §6.10」と書いているので、仕様書側が正本として誤っている。
- 差し替え案: 仕様書 (d) の見出しを「全 34 件。…2026-10-18 に `shift_video`／`shift_audio`（MiniMax H3 の項目）の新設で無視が 2 つ増え、422 系 2・無視 4・動作 28・従属 0」とし、表の無視の行に 2 項目を足して理由を「蒸留版のスケジュールに shift の概念が無い」と添える。MULTI_ENGINE §5.6 は件数と項目名を書かず「無視＋ログに残るのは、エンジンに概念が無い項目（CFG・ステップ数・shift）。内訳は仕様書 §6.10(d)」へ。README の同趣旨の文は ④ R-4。

### D-006【高】PENDING_TASKS.md §1-95 の前提 — 済んだ工事が前提に残る
- 場所: `Docs/PENDING_TASKS.md` §1-95「**前提**: …較正の前に、大きな画角での復号の VRAM（[`VERIFICATION_LOG.md`] §165.7）を直す。」
- 正しい事実: 第 2 弾の 2a で済んでいる。復号は窓ごとに CPU の uint8 を返す（設計正本 §8「映像 VAE の特例」）。2a-G5 で 1344×768・345 フレームの復号の VRAM 予約ピーク 8.01GB（線 12.8GB）に合格（VERIFICATION_LOG §166.3）。
- 差し替え案: 「較正の前に…を直す。」を消し、「復号の VRAM は第 2 弾で直した（VERIFICATION_LOG §166.3 の 2a-G5）。ノイズ除去の段の VRAM の溢れは §1-101 で扱う」に。§1-101 との順序は「§1-101 の後か同時（オーナーが決める）」が既に §1-101 側にあるので、§1-95 にも「§1-101 との順序はオーナーが決める」を 1 句。

### D-007【高】操作パネル側 API_REFERENCE.md §5.1・§3.5 — `GenerateRequest` の表と `GET /models`
- 場所: `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/API_REFERENCE.md`
  - 489〜494 行目の表: 「`width` | int `512` | 256-4096、**64の倍数必須**(`api/models.py::GenerateRequest`のフィールドバリデータ)」「`num_frames` | int `49` | 9-481、**8n+1必須**(`api/models.py::GenerateRequest`のフィールドバリデータ)」「`num_inference_steps` | int `8` | **§8参照: distilledは8固定・変更不可(それ以外は422)**」。`shift_video`・`shift_audio` の行が無い。`frame_rate` は「1.0-60.0」だけ。
  - 556 行目の「**凍結制約(緩和禁止)**: width/height ÷64、num_frames 8n+1、distilled=8step/CFG=1.0。」
  - 228〜233 行目の `GET /models` の例と 243 行目以降の説明に `constraints` が無い。243 行目の `unsupported_features` の説明は LTX の件数だけ。
- 正しい事実: 64 の倍数・8n+1・distilled の検査は LTX の規則表（`services/engines/ltx/request_rules.py`）へ移り、`validate_for_family` が `reject_unsupported` の後に検査する（仕様書 §6.2b）。`num_inference_steps` の既定は `null`（系統の既定値。LTX 8・H3 20）。`shift_video`／`shift_audio` は `float | null`・`gt=0`。H3 は 17n+5（22〜481）・32 の倍数・24fps 固定。`base_models[].constraints` がある（仕様書 §6.2b (e)）。H3 の `unsupported_features` は 14 語（仕様書 §6.2b (f)）。
- 差し替え案: 表の 3 行を「LTX は 64 の倍数／8n+1（系統の規則表。値はベースモデルの `constraints`）」「`num_inference_steps` | int \| null `null` | null＝系統の既定値。LTX は distilled で 8 固定、MiniMax H3 は 1〜50」に。`shift_video`・`shift_audio` の 2 行を足す（MiniMax H3 用・LTX では無視）。556 行目は「LTX の規則: …。系統ごとの値は `GET /models` の `constraints`」。§3.5 の例に `"constraints": {…}` を 1 つ（LTX の値）足し、説明に「`constraints`（2026-10-18 追加）: 形と意味の正本は仕様書 §6.2b (e)。操作パネルは `shell/generationConstraints.ts` で読む」。`unsupported_features` の説明は件数を消し「各系統の内訳は仕様書 §6.10 と §6.2b (f)」へ。冒頭の最終更新日は D-029。

### D-008【中】仕様書の事前検査の段落 — H3 の分岐が無い
- 場所: `Videomni_Backend_Specification.md` 1459 行目「**事前チェックは 2 段**である。①`precheck_model_file`（拡張子・GGUF/safetensors のヘッダ健全性 …）、②KV の判定（transformer カテゴリのみ。GGUF の KV メタデータで裁く…）」、1460 行目「**transformer に `.safetensors` が選ばれたときは、①が量子化 safetensors（fp8／int8）の受け入れ検査になる。**…検査の本体は `sft_quant_format.inspect`」。
- 正しい事実: `services/model_registry.py` の `precheck_model_file` は記述子の系統が `h3` なら `_precheck_h3` へ行き、transformer を `h3_format.inspect_h3_dit`、text_encoder を `inspect_h3_te`、VAE 2 本を汎用の safetensors 検査にかける。`H3FormatError` は 422 `MODEL_INCOMPATIBLE`。H3 の `check_kv` は何もしない（KV が無い）。H3 の int8 ConvRot は `sft_quant_format` を通らない。設計正本 MINIMAX_H3_DESIGN §7「アプリ側の事前検査」。
- 差し替え案: 1460 行目の直後に 1 項「**MiniMax H3（系統 `h3`）は別の経路である。** ①は記述子の系統を見て、DiT を `h3_format.inspect_h3_dit`、テキストエンコーダを `inspect_h3_te`（テンソル名の指紋と量子化の記述）にかけ、VAE は汎用の safetensors 検査。不合格は 422 `MODEL_INCOMPATIBLE`。②（KV）は H3 では行わない。判定の規則の正本は Docs/MINIMAX_H3_DESIGN.md §7」。1460 行目の頭に「LTX の系統で」を足す。

### D-009【中】仕様書の venv・設定・構成・用語集が 2 系統のまま
- 場所（`Videomni_Backend_Specification.md`）: 246 行目と 382 行目の venv の表（`./.venv-engine-ltx25` まで）、347 行目「エンジン worker（LTX 2.3 は `./.venv-engine`、LTX 2.5 は `./.venv-engine-ltx25`。…」、373 行目付近の構成図、430 行目と 1753 行目の比較表（「エンジン python | `model.engine_python` | `model.engine_python_ltx25`」）、487〜492 行目のディレクトリ図（`engine_h3/`・`h3_format.py` が無い。「manifests/ ベースモデル記述子（10-ltx23.json / 20-ltx25.json）」）、§11.2 の設定の表（1929 行目 `engine_python_ltx25` の後に `engine_python_h3` が無い）、2383 行目の用語集「現在は `"ltx"`〔LTX 2.3〕と `"ltx25"`〔LTX 2.5〕の2系統」。
- 正しい事実: `config.py` に `engine_python_h3: str = "./.venv-engine-h3/Scripts/python.exe"`。`services/engines/__init__.py` の `FAMILY_BY_ID` は `ltx`・`ltx25`・`h3`。記述子は `40-minimaxh3.json` が増えた。`engine_h3/`・`h3_format.py`（直下）がある。
- 差し替え案: 用語集を「現在は `"ltx"`〔LTX 2.3〕・`"ltx25"`〔LTX 2.5〕・`"h3"`〔MiniMax H3〕。一覧の正本は `services/engines/__init__.py` の `FAMILY_BY_ID`」（件数は書かない）。§11.2 に `engine_python_h3` の行（「系統 `h3` の worker を回す専用 venv。torch の版は LTX と違う。Docs/MINIMAX_H3_DESIGN.md §10」）。venv の表・ディレクトリ図に 1 行ずつ（版の数字は書かず freeze を指す）。比較表 2 つは列を足すか「H3 は Docs/MINIMAX_H3_DESIGN.md §8」の 1 行。インストールは第 3 弾なので「導入は `install-MiniMaxH3.bat`（未実装）」を添えるかは直し担当の判断で、書くなら未来形にせず「導入手順は README（第 3 弾で記載）」程度に。

### D-010【中】仕様書 §12.2 Gradio の Generate タブ — 半分だけ更新
- 場所: 同書 2049 行目「**width / height**（各 ×64）、**num_frames**（ベースモデルの `constraints` に従う。LTX は 8n+1、MiniMax H3 は 17n+5）、… スピナー矢印は width/height が **64刻み**、num_frames が **8n+1刻み**でスナップする（サーバー側の 8n+1・÷64 検証は従来どおり有効）。」。同節にステップ数と shift の 3 本のスライダーの記述が無い。
- 正しい事実: `gradio_ui/constraints.py` の `single_scope_updates` が width／height の `step` を `dimension_multiple`（LTX 64・H3 32）、frames の `step` を格子の刻み（8・17）にし、ラベルも `8n+1`／`17n+5` を出す。`sampling` がある系統ではステップ数・Shift（映像）・Shift（音声）の 3 本が操作でき、CFG の飾りのスライダーと「8／1.0 固定」の注記を隠す。`sampling` が無い系統では `num_inference_steps` を送らない（`sampling_payload`）。
- 差し替え案: 「width / height・num_frames の刻みとラベルはベースモデルの `constraints` に従う（LTX は 64 の倍数・8n+1、MiniMax H3 は 32 の倍数・17n+5）。サーバー側の検査は系統の規則表（§6.2b）」。1 項足す「**ステップ数・Shift（映像）・Shift（音声）**: `constraints.sampling` がある系統（MiniMax H3）だけ操作でき、既定はサーバーの値。無い系統（LTX）はステップ 8・CFG 1.0 の固定表示で、ステップ数を送らない」。

### D-011【中】仕様書 付録 A の「検証は §6.7・`api/models.py`」
- 場所: 同書 2351 行目「生成サイズ（width/height）**【LTX の規則】** | **64 の倍数** | … 検証は §6.7・`api/models.py`」（2352・2353 行目も「§6.7」）。
- 正しい事実: LTX の 64 の倍数・8n+1・distilled の検査は `services/engines/ltx/request_rules.py` の規則表で、`services/engines/__init__.py` の `validate_for_family` が呼ぶ（§6.2b）。`api/models.py` の `Field` はその定数を参照するだけ。
- 差し替え案: 3 行とも「検証は系統の規則表（§6.2b。`services/engines/ltx/request_rules.py`）」。§6.7 が今も参照先として正しいかは直し担当が確かめる（§6.7 は快適上限の節で、検査の節ではない可能性が高い）。

### D-012【中】仕様書に MiniMax H3 の 4 分類の表が無い
- 場所: 仕様書 §6.2b (f) は `unsupported_features` だけ。MULTI_ENGINE §5.6「**どのフィールドがどの分類かの正本は [仕様書] §6.10 である**」。§6.10 は LTX 2.5 だけ。
- 正しい事実: `services/engines/h3/adapter.py` に 4 分類がある。422 系 12 行（`REJECT_TABLE`。`conditioning_images` が 2 行）、無視＋ログ 1（`fused_gguf_dequant_kernel`）、動作 15（`HONOURED_FIELDS`。`crop_output`・`embed_mp4_metadata` を含む）、従属 8（`GOVERNED_FIELDS`。NAG の副パラメータ 6 と参照強度 2）。`conditioning_images` は動作と 422 の両方に載る（監査テストが許す重なり）。`reject_chain` が連結を丸ごと断る。
- 差し替え案: ③ J-5 で置き場を決めてから。仕様書に §6.11「MiniMax H3（系統 `h3`）の対応範囲」を新設し、4 分類の表（件数は見出しに 1 回だけ）・連結は丸ごと 422・`unsupported_features` は §6.2b (f) を指す、とするのが MULTI_ENGINE §5.6 の「正本は仕様書」と整合する。

### D-013【中】`transformer_weight_class` が H3 では常に `null` であることが書かれていない
- 場所: 仕様書 1501 行目「**`transformer_weight_class`**…は `base_models[]` の各要素に必ず入り、選択中の transformer の重みの種別を `"4bit"`／`"8bit"`／`"q6k"`／`null` で名乗る。…**値が入るのは有効なベースモデル（`active: true`）だけ**」。操作パネル側 API_REFERENCE 241 行目（同趣旨。「8bit＝fp8・fp8_scaled・int8・int8_convrot」）。
- 正しい事実: `services/weight_class.py` の `classify_transformer(path, engine_family=…)` は LTX 以外の系統ではファイルを開かず `None` を返す（94 行目）。`api/models_registry.py` の `transformer_weight_class` が記述子の系統を渡す。したがって H3 の int8 ConvRot の DiT は、有効でも `null`（API_REFERENCE の「int8_convrot は 8bit」を読むと H3 も 8bit と誤解する）。
- 差し替え案: 両方に 1 文「LTX 以外の系統（MiniMax H3）は、有効でも `null`（快適上限の表を持たないため、判定しない）」。§1-95 で表を作るときの扱いは ③ J-4。

### D-014【中】COMFORT_LIMIT_TABLE.md §1.1 — 系統の表そのものが無いときの振る舞い
- 場所: `Docs/COMFORT_LIMIT_TABLE.md` 12 行目「**どの行にも当てはまらないときは線を引かず、単発生成はレガシー表`spill_free_frames`…へ、連結生成は互換値 40,000…へ落ちる**——これは異常ではなく、設計どおりの正常な経路である。」、23 行目「正本はマニフェストの `comfort`（`scripts/manifests/10-ltx23.json`・`20-ltx25.json`。…）」、§1.4 の 6 項「連結の 40,000 は、一致する行が無いときの落ち先も兼ねる」。本書に MiniMax H3 の言及が 1 つも無い。
- 正しい事実: H3 の記述子 `40-minimaxh3.json` は `comfort` を持たない（台帳 §1-95 まで）。配信される `comfort_budgets` に系統の項目が無いとき、操作パネルは「線が無い」と読みマーカーを出さない（`webui/src/shell/comfortTable.ts` の `comfortProfileMissing`。DEVLOG §141.2）。右クリックの事前入力も快適上限で尺を書き換えず、格子へのスナップだけ（同）。Gradio も表が無ければ溢れ警告と Settings の最大フレーム数の表を出さない（コミット `cdaa51c`）。`spill_free_frames` へは落ちない。
- 差し替え案: 12 行目の後に 1 文「**系統の項目そのものが表に無いとき（記述子が `comfort` を持たない系統。いまは MiniMax H3）は、マーカーも溢れ警告も出さず、レガシー表へも落ちない**（較正が済むまで線を出さないため。台帳 §1-95）」。23 行目は記述子名の列挙をやめ「各ベースモデルの記述子（`scripts/manifests/*.json`）の `comfort`」に。

### D-015【中】MULTI_ENGINE_DESIGN.md の 2 系統の書きぶり（§1・§3・§4・§5.2・§6.2・§9.1）
- 場所と古い文:
  - §1（40 行目）「**LTX 2.5 がこれに当たり**、将来の Wan なども同じ扱いになる。」
  - §3.3 の図（LTX 2.3／LTX 2.5／Wan 2.2〔将来〕）と「**現在は2つのベースモデルがそれぞれ別の系統に載っており、1対1になっている。**」、表「（LTX 2.5・将来の Wan 等）」。
  - §4.2「**ベースモデル記述子**（`10-ltx23.json`・`20-ltx25.json`）」。
  - §4.3 補足「**`assets` の内容は系統ごとに違う。** LTX 2.5 側の `assets` は空間アップスケーラ1つだけである」（H3 は `tokenizer_dir` のフォルダ 1 つ）。
  - §4.4（255 行目）「（`services/engines/ltx/adapter.py`・`services/engines/ltx25/adapter.py`）…**カテゴリ名（キー）は両系統とも同じ4種**」。
  - §5.2（305 行目）「**分割の要否は3例目（Wan 等）が来たときに改めて判断する**（§8.1）。」— §8.1 は「3例目（MiniMax H3）が載ったが、この3点は引き続き保留した」と書いており食い違う。
  - §6.2 の注記「**3本目以降のベースモデルは、バッチ1本と記述子1本で足せる。**」— 既存の系統で読める重みの話。H3 のように別系統なら venv・ワーカー・アダプタも要る（§3.3 の表）。
  - §9.1 の注記（889 行目）「mock バックエンドは…`ltx` アダプタの中に置いたまま…真に別規約の系統（Wan 等）が来たときに、改めて判断する。」— H3 は別規約だが LTX 2.3 の mock を自分の名札で共有している（`H3Runner._MOCK_BACKEND_CLS = _MockBackend`、`MOCK_BACKEND_H3 = "mock-h3"`）。
- 正しい事実: 系統は `ltx`・`ltx25`・`h3`（`FAMILY_BY_ID`）。H3 は §3.3 の「同居できない系統」で、`H3Runner(LTXRunner)`・mock は共有・カテゴリは同じ 4 語（`services/engines/h3/adapter.py` の冒頭説明）。
- 差し替え案: 一覧は件数と固有名を減らし「正本は `FAMILY_BY_ID`」と指す。§3.3 の図に `MiniMax H3 → h3 （engine_h3/ + .venv-engine-h3）` を足し、「1対1」の文は「現在は各ベースモデルが別の系統に載っている」。§5.2 と §9.1 の「3例目（Wan 等）が来たときに判断」は「3例目の MiniMax H3 でも分割と mock の共通化は要らなかった（§8.1）。次の系統で改めて判断する」。§6.2 は「同じ系統で読める重みなら」を前に足す。§4.3 に「MiniMax H3 の `assets` は Qwen3-VL の tokenizer のフォルダ（`tokenizer_dir`）1 つ」。

### D-016【中】MULTI_ENGINE_DESIGN.md の「訂正（2026-10-17）」注記を本文へ畳む（設計正本の規則）
- 場所: §2.2 の末尾 2 か所（「> **訂正（2026-10-17）**: H3 系統は safetensors にテンソル名の指紋で判定する」、同「上の規約（両キーを必ず持つ）は及ばない」）、§5.3（D-004）、§5.8 の表の下「> **訂正（2026-10-17）**: 推論モードの表に `h3`（MiniMax H3）が加わる。`engine_h3` は**全経路 `torch.no_grad()`**」、§7 の【2026-10-17 訂正】2 か所（D-002）、§8.1 の「【2026-10-17】」、§9.2 の表の下「> **訂正（2026-10-17）**: 「推論パイプライン一式」の行に H3 の系統が加わる。」
- 何が問題か: 本書は「現在の姿だけを現在形で」と冒頭で宣言している。§9.2 は H3 の行（`H3 の推論パイプライン一式`）を表に足したのに訂正注記も残り、二重になっている。§9.2 の「低VRAM制御のつまみ」行は LTX の 2 系統だけ（H3 は `blocks_on_gpu` と、ジョブごとの `attention_backend`・`block_swap_prefetch`。TE は直接読み）。§5.8 の見出しは「`engine25` は全経路 `no_grad`。2.3 は全経路 `inference_mode`」のまま。
- 差し替え案: §5.8 の表に `h3` の行（「全経路 `torch.no_grad()`」「CPU テストで `engine_h3` 配下に `inference_mode` が無いことを固定」）を足して注記を消し、見出しを「推論モードの規約 — 2.3 だけが `inference_mode`、他は `no_grad`」へ。§2.2 は「H3 系統は KV を持たず、事前検査でテンソル名の指紋を見る（MINIMAX_H3_DESIGN §7）。本節の 2 キーの規約は GGUF と LTX の safetensors に適用する」を本文に 1 文。§9.2 の訂正注記は削除し、低VRAMの行に H3 を 1 句。§8.1 の「【2026-10-17】」は日付の札を外して現在形に。

### D-017【中】MINIMAX_H3_DESIGN.md §13「性能」— 第 2 弾の観察と §1-101 を指していない
- 場所: `Docs/MINIMAX_H3_DESIGN.md` 322 行目「**性能**: 第 1 弾の G5（640×384・124 フレーム・20 ステップ）は全体で約 140 秒だった（…§165.5）。大きな画角では block swap の転送（毎ステップ約 19GB）が上限になる可能性がある（推測）。遅すぎて快適上限の較正に支障が出るなら、LoRA（Turbo）対応の優先順位を上げる（研究ノート §10 の裁定 21）。」
- 正しい事実: 第 2 弾の目視で I2V 896×1152・124 フレームのノイズ除去が 1 ステップ約 210 秒・共有 GPU メモリ 3.7GB の溢れ（VERIFICATION_LOG §166.9）。台帳 §1-101（VRAM 16GB 最適化・β版以降）が起票済みで、出典に本書 §13 を挙げている。設計正本は数値を書かない約束（§166.8「設計の正本は…数値は書かず、本節を参照する」）なのに §13 には「約 140 秒」「毎ステップ約 19GB」がある。
- 差し替え案: 「**性能**: 大きな画角ではノイズ除去の段で VRAM が溢れ、1 ステップが大きく遅くなる（実測は VERIFICATION_LOG §166.9）。最適化は台帳 §1-101。遅すぎて快適上限の較正（§1-95）に支障が出るなら、LoRA（Turbo）対応の優先順位を上げる（研究ノート §10 の裁定 21）」。数値は消す。

### D-018【中】MINIMAX_H3_DESIGN.md §11 — 済んだ弾の門の表と「以後」の列挙
- 場所: §11「門（第 1 弾）」G4「…テキストエンコーダは層オフロードの on と off で出力がビット単位で一致」、同 G1〜G5 と「門（第 2 弾）」の表全体。「弾の分け方」の表「以後 | 台帳 §1-95（快適上限の較正）・§1-97（末尾と中間キーフレーム）」。
- 何が問題か: G4 の「層オフロード」は第 2 弾で廃止された仕組み（§4「`te_layers_on_gpu` という設定は無い」と本書内で食い違う）。済んだ弾の門の文言は経緯（記録簿 §165・§166 が持つ）で、2a-G4 の文言は監督が「常駐による上乗せが 0」に改めた（§166.3）のに本書は「合間の GPU の割り当て」のまま。「以後」は §1-100・§1-101 が抜けている（規則 (b)）。
- 差し替え案: 門の表 2 つは「門の内容と結果は VERIFICATION_LOG §165.5（第 1 弾）・§166.3〜§166.5（第 2 弾）」の 1 行へ置き換える（残すなら G4 に「当時の方式」と明記し、2a-G4 の文言を直す）。「以後」は「台帳 §1 の MiniMax H3 の項目」と台帳を指す。

### D-019【中】VERIFICATION_LOG.md §166.2 — 2b-G2 が「再試行中」のまま
- 場所: `Docs/VERIFICATION_LOG.md` §166.2「GPU を使う 2b-G2 は、担当のバックエンド起動が自動許可の判定で拒否されたため、同一プロセスで API を叩く方式に替えて再試行中。」
- 正しい事実: §166.4 のとおり、オーナーが `run.bat` で起動したサーバーに対して監督が PowerShell から台本 `run_2b_g2.py`（MCP サーバーを stdio で叩く）を実行して合格（2026-10-18 早朝）。「同一プロセスで API を叩く方式」は採っていない。
- 差し替え案: 「GPU を使う 2b-G2 は、サブエージェントによるサーバー起動が自動許可の判定で拒否されたため、オーナーが `run.bat` で起動したサーバーに監督が台本を実行した（§166.4）。」

### D-020【中】PENDING_TASKS.md §1-93「裁定と前提」の尺
- 場所: §1-93「前提の工事＝フレーム規約 17n+5・24fps 固定・4〜15 秒の系統別制約（同 §9）」。
- 正しい事実: 実装の規則表は 17n+5・22〜481 フレーム（約 0.9〜20 秒）・32 の倍数・24fps 固定（`services/engines/h3/adapter.py` の `REQUEST_RULES`）。尺は「快適上限を測って決める」（裁定 9）で、4〜15 秒の制約は入れていない。前提の工事そのものも第 2 弾で済んだ。
- 差し替え案: 「前提の工事（系統別の数値の規約・専用の仮想環境）は第 2 弾までに済んだ。規約の正本は MINIMAX_H3_DESIGN §9」。

### D-021【中】操作パネル側 DEVLOG.md §141.4・§141.5 — 状態が古い
- 場所: `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/DEVLOG.md` §141.5「実装済み・未コミット。ビルド（`build.ps1`）と配置（`deploy.ps1`）、オーナーの目視（2c-G2）は監督が行います。」、§141.4「1 行のスライダーが WebView2 の狭い幅でどう折り返すかは、実機の目視で確かめます。」
- 正しい事実: コミット `b2a07e7`。ビルドと配置は 2026-10-18 未明に済み、2c-G2 はオーナーの目視で合格（狭い幅での折り返しも合格。VERIFICATION_LOG §166.9）。目視で見つかった 2 点はバックエンド台帳 §1-100。
- 差し替え案: §141.5 を「コミット済み（`b2a07e7`）。ビルドと配置済み。オーナーの目視（2c-G2）に合格（バックエンド VERIFICATION_LOG §166.9）。目視で見つかった 2 点（Fused GGUF の行・A2V カードとバッチ A2V の閉じ方）はバックエンド台帳 §1-100」。§141.4 の折り返しの行は「目視で合格」に。DEVLOG は記録なので、追記の形（「> 更新（2026-10-18）」）でもよい。

### D-022【中】操作パネル側 RIGHTCLICK_REDESIGN_SPEC.md §5-12 — 快適上限で尺を決める方式
- 場所: `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/RIGHTCLICK_REDESIGN_SPEC.md` §5-12「**`comfortCeiling`（快適上限）**: …解像度から`AppConfig.limits.spill_free_frames`…を引き、その値へ**常に**…合わせます」、「**`materialClampedToCeiling`**: …素材自身の実尺…を8n+1グリッドへ**切り下げ**」。
- 正しい事実: 快適上限の表に系統の項目が無いベースモデル（MiniMax H3）では、右クリックの事前入力は尺を快適上限で書き換えず、格子（`constraints.num_frames`。H3 は 17n+5）へのスナップだけ行う（DEVLOG §141.2・§141.3。`timeline/deriveDuration.ts`・`prefillSeed.ts`・`AppShell.tsx` の #9／#10）。
- 差し替え案: §5-12 の冒頭の 2026-08-31 追記の下に「> 追記（2026-10-18）: 快適上限の表に系統の項目が無いベースモデル（いまは MiniMax H3）では、`comfortCeiling` も `materialClampedToCeiling` も快適上限を使わず、ベースモデルの `constraints` の格子へのスナップだけ行う。格子（8n+1 など）は `constraints.num_frames` から取る（操作パネル DEVLOG §141）」。

### D-023【中】MINIMAX_H3_RESEARCH_NOTES.md §13.7（本製品の現物との照合）
- 場所: §13.7 の 6「記述子（manifest）・`install-H3.bat`・専用の仮想環境・…」、同 3「**テキストエンコーダの層オフロードにも前例がある。** LTX 2.3 の `--te-offload` と LTX 2.5 の `te_layers_on_gpu`。」
- 正しい事実: インストーラの名前は `install-MiniMaxH3.bat`（裁定 15「`install-H3.bat` ではない」。`install-<記述子の id>.bat` の規約。id は `MiniMaxH3`）。H3 のテキストエンコーダは既存の層オフロードを使わず、層ごとにファイルから GPU へ直接読む（裁定 32。設計正本 §4・§6「`GemmaLayerOffloadService` は使わない」）。第 3 弾でインストーラを作る人がここを読むと名前を誤る。
- 差し替え案: 6 の直後に「> 訂正（2026-10-17）: 名前は `install-MiniMaxH3.bat`（裁定 15）」、3 の直後に「> 更新（2026-10-18）: H3 では既存の層オフロードを使わず、層ごとにファイルから直接読む（裁定 32。設計正本 §4）」。研究ノートは参考資料なので、本ノートの作法（訂正の引用）に合わせる。

### D-024【中】PENDING_TASKS.md §1-93 の未来形と決着済みの論点
- 場所: §1-93「**α版の完結条件**: …ステップ数と shift（映像・音声）は第 1 弾では既定値で固定し、第 2 弾で操作パネルと Gradio に 3 本のスライダーとして出す（裁定 22）。」、「**概要**: …ヘッダの揺れ（…§13.3）を α版でどこまで吸収するかは計画のセッションで決める。」、§1 冒頭の太字「第 2 弾（…）は目視合格・main への merge 待ち（オーナーの合図で行う）。§1-95 と §1-97 は…」（§1-100・§1-101 に触れていない）。
- 正しい事実: 3 本のスライダーは第 2 弾で出した（§166）。ヘッダの揺れの吸収範囲は設計正本 §7 で決まった（接頭辞 4 通り・groupsize 256 以外・形状から構成を導く。コミュニティ版・非枝刈り版は後回し）。§1 冒頭は merge の後に必ず古くなる。
- 差し替え案: 完結条件から「第 1 弾では…第 2 弾で…出す（裁定 22）」を消し「ステップ数と shift の 3 本のスライダーは第 2 弾で出した」か削除。概要の揺れの文は「吸収する範囲は MINIMAX_H3_DESIGN §7」。§1 冒頭は merge したら「第 2 弾は完了（main merge 済み）。次は第 3 弾（`install-MiniMaxH3.bat`・README）。§1-100 は第 3 弾の前後どちらでもよい。§1-95・§1-97・§1-101 は α版の完結後」へ（merge と同じコミットで直す）。

### D-025【低】MINIMAX_H3_RESEARCH_NOTES.md §10 の見出しの注記と裁定 22・25 の行き先
- 場所: §10 冒頭「H3固有の設計正本が未執筆のため、暫定的に本書へ記録する。**設計正本の執筆後はそちらへ移管し、本書の本節は参照に置き換えること。**」（直後の行は「設計の裁定は設計正本 §2 に移管済み」で食い違う）。裁定 22「既定値は第 2 弾の着手時に決める。」。裁定 25「…根本策と、T2V で映像 VAE の符号化器を読まない省略は α版の後に回し、台帳 §1-98・§1-99 に起票した。」。「2026-10-17 の裁定（第 2 弾の着手時・追記）」の前文「設計に関わる裁定は、第 2 弾の計画の了承後に設計正本 §2 へ移管する。」
- 正しい事実: 設計正本は 2026-10-17 に新設済み。既定値は裁定 24 で 20／12／3。§1-98・§1-99 は裁定 32・33 で第 2 弾へ吸収して削除。移管は済んだ（§2・§4・§8・§9）。
- 差し替え案: 冒頭 2 文を「設計に関わる裁定は設計正本 [`MINIMAX_H3_DESIGN.md`] §2 に移管済み。本節は進め方の裁定と、裁定の原文の記録として残す」の 1 文に。裁定 22 の末尾に「→ 既定値は裁定 24」、裁定 25 の末尾に「→ 2 点とも裁定 32・33 で第 2 弾へ戻した（§1-98・§1-99 は削除）」。前文は「設計に関わる裁定は設計正本へ移管した（各行の → 印）」。

### D-026【低】MINIMAX_H3_RESEARCH_NOTES.md §13.8 の「計画のセッションで書く／決める」
- 場所: §13.8「**これは設計ではなく、範囲のメモである。** 設計正本は計画のセッションで書く。」、箇条「揺れ（13.3）を α版でどこまで吸収するかは計画のセッションで決める。」
- 正しい事実: 設計正本 `MINIMAX_H3_DESIGN.md` があり、揺れの範囲は同 §7 で決まった。
- 差し替え案: 「設計正本は `MINIMAX_H3_DESIGN.md`」「吸収する範囲は設計正本 §7」。

### D-027【低】MINIMAX_H3_DESIGN.md の未来形・経緯の言い回し
- 場所と古い文: §1「1. MiniMax H3 を第 3 のエンジン系統（`h3`）として足す。…新設する」／§2 裁定 18「画面に出すのは裁定 22 のとおり第 2 弾である。」／裁定 22「…3 本のスライダーを、第 2 弾で操作パネルと Gradio に出す。」／§7「**吸収する揺れ（第 1 弾でただで取れる分）**」／§3「短辺 768 が既定である。」
- 正しい事実と案: §1 は「載せている」「置いてある」の現在形に。裁定 18・22 は裁定の原文なので、裁定 18 は「（画面に出すのは裁定 22）」程度に縮め、22 は「操作パネルと Gradio に出す」（「第 2 弾で」を外す）。§7 は「吸収する揺れ」だけに。「短辺 768 が既定」はモデルの学習上の既定（研究ノート §2「短辺768pxが既定（学習canvasは1344×768）」）で、本製品の既定の画角ではない。「モデルの学習上の既定は短辺 768（本製品の既定の画角ではない）」に。

### D-028【低】保管フォルダ `Docs/Outputs-archive/h3-gate2/` の README と手順書
- 場所: `README.md` の題「# MiniMax H3 第 2 弾 2a（ワーカー）の門の台本と結果」（表に 2b-G2 の行がある）。`README_2b_g2.md`「手順は「バックエンドを起動 → 台本を 1 本実行 → バックエンドを止める → `state.json` を戻す」の 4 つ。」
- 正しい事実: 2b-G2 の台本と結果も置いてある。実施では、サーバーは止めず `state.json` も戻していない（選択中のベースモデルは MiniMaxH3 のまま。VERIFICATION_LOG §166.4「後片付け」）。
- 差し替え案: 題を「MiniMax H3 第 2 弾の門（2a と 2b-G2）の台本と結果」。手順書の結論の下に「> 実施の記録（2026-10-18）: 起動はオーナーの `run.bat`。終了後もサーバーは止めず、`state.json` は戻していない（VERIFICATION_LOG §166.4）」。

### D-029【低】操作パネル側 API_REFERENCE.md の最終更新日と `comfort_budgets` の鍵の語彙
- 場所: 3 行目「最終更新: 2026-10-15（Alpha Gen 第3弾…）」（第 2 弾で §7 を直したのに日付が進んでいない）。182 行目「**外側の鍵はエンジン系統のid**——…`"ltx"`＝LTX 2.3系、`"ltx25"`＝LTX 2.5系である。」
- 案: 最終更新に「2026-10-18（MiniMax H3 第 2 弾: §7 のステップ数の行。D-007 を直すなら §3.5・§5.1 も）」。182 行目に「`h3`（MiniMax H3）は表に項目が無い＝マーカーを出さない（COMFORT_LIMIT_TABLE §1.1）」。

### D-030【低】MODEL_MANAGEMENT_DESIGN.md の注記（歴史記録の文書）
- 場所: 冒頭の注記（2026-09-27）「**`GET /models` の行には重みの種別（GGUF／fp8／int8 など）を示す項目は無い**——種別を名乗らせるのは [`PENDING_TASKS.md`] §1-31 の予定である。」。§5.1「3. **拡張子/カテゴリ整合**: transformer/text_encoder は `.gguf`、vae/audio は `.safetensors` 必須。」
- 正しい事実: §1-31 は 2026-10-06 に完了し `base_models[].transformer_weight_class` がある（CLOSED 3-216）。カテゴリと拡張子は記述子の `categories[].extensions` が決め、H3 は transformer・text_encoder とも `.safetensors`。事前検査は系統で分かれる（D-008）。
- 案: 本書は「以後は歴史記録」と宣言しているので本文は直さず、冒頭の注記に 1 行「> 注記（2026-10-18）: 種別は `transformer_weight_class`（CLOSED 3-216）。拡張子はベースモデルの記述子の `extensions` が決め、MiniMax H3 は text_encoder も `.safetensors`。事前検査の分岐は仕様書の `POST /pipeline/load` の節」。第 2 弾が原因ではない古さを含むので優先度は低。

### D-031【低】仕様書 §6.10(f) の連結の 8n+1「系統に依存しない」
- 場所: 仕様書 1660 行目「- **総尺上限・`8n+1` のフレーム規約・クリップ本数（1〜24）は系統に依存しない**（`chain_math.py` の共有計算がアプリとエンジンの両方から呼ばれる）。LTX 2.5 でも同じ値が効く。」
- 正しい事実: LTX の 2 系統で共通。MiniMax H3 は連結を丸ごと 422（`reject_chain`）。
- 案: 「LTX の 2 系統で共通である（MiniMax H3 は連結そのものを持たない）」。

---

## ③ 判断が要るもの（文書どうし・文書とコードの食い違い）

### J-1 メインメモリ 32GB 機の扱い
- 食い違い: 設計正本 §13「**メインメモリ 32GB 機**: コミット量の山が 32GB を超えるので、ページファイル前提になる（§4…）。README に書く（第 3 弾）。」と、記録 §166.7「64GB 機では作業セット 29GiB で溢れない。**32GB 機は引き続き対象外**。」。研究ノート裁定 1「サブマシンは32GB。どちらもページファイル利用可」、裁定 26「推奨スペックは VRAM 16GB・主メモリ 64GB…足りない機械への対応は後から考える」。
- 選択肢: (a) 「32GB 機は α版の対象外（推奨 64GB。裁定 26）」に統一し、設計正本 §13 を直す。(b) 「32GB 機はページファイル前提で動く（遅い・SSD 書き込みあり）。常駐は非推奨」に統一し、§166.7 の「対象外」を記録簿の追記で訂正。(c) 第 3 弾の README を書くまで保留し、設計正本 §13 に「対象外か、ページファイル前提かはオーナーが第 3 弾で決める」と書く。第 3 弾の README の要件行に直結するので、第 3 弾の計画の前に決めるのがよい。

### J-2 Single の A2V カードの閉じ方（裁定 35 と台帳 §1-100）
- 食い違い: 設計正本 §2 裁定 35【オーナー裁定】「Single の A2V カードは連結要求そのものなので `chain` で閉じる（灰色）」と、台帳 §1-100（オーナーの目視の所感）「Single の A2V カードと Batch A2V は、H3 では灰色でなく非表示にする」。コードは今は灰色（`featureScope.ts` の `chain` の行。閉じ方の見た目は各画面が決める）。
- 選択肢: (a) §1-100 を実装するときに裁定 35 を改める（それまでは今のまま。設計正本に「§1-100 で非表示へ変える予定」とは書かない＝未来形を避ける）。(b) 今のうちに裁定 35 に「オーナーの所感で非表示へ変える（台帳 §1-100）」と注記する。オーナー裁定の文言を変えるのでオーナーの確認が要る。

### J-3 研究ノートの裁定 5「インストーラーを先に作る」
- 食い違い: §10 裁定 5（2026-09-06）「推論エンジンはまずSingle用のみを作る。インストーラーを先に作る」。2026-10-16 の追記は「9 月の裁定 1・3・5 は変更しない（維持とみなす。監督の判断）」、2026-10-17 の追記も「裁定 1〜12 は変更しない」。しかし裁定 14・23 と台帳 §1-93 はインストーラを第 3 弾（最後）に置き、実際もそう進んだ。
- 選択肢: (a) 裁定 5 の後半は裁定 14（α版の完結条件・順序）で置き換わったと注記する（オーナー確認）。(b) 第 3 弾の計画で「インストーラ先行」の趣旨（インストール作業の検証を省く）をどう扱うかを議題にする。

### J-4 MiniMax H3 の快適上限の表と `weight_class`（台帳 §1-95 の設計の穴）
- 食い違い: 仕様書 1346 行目「**全行に `weight_class` が必須**で、種別なしの汎用行や上の 3 値以外の値は起動時のエラーになる」。一方 `services/weight_class.py` は LTX 以外の系統で常に `None` を返す（D-013）。このままでは §1-95 で H3 の記述子に `comfort` の行を足しても、`weight_class` が `null` なので操作パネルの照合でどの行にも当たらない。
- 選択肢: (a) §1-95 で H3 の int8 ConvRot を `"8bit"` と判定するよう `classify_transformer` を広げる。(b) H3 は種別を 1 つしか持たないので、系統ごとに「種別なしの行」を許す規則に変える（起動時の検査を緩める）。(c) H3 用に別の種別名（例 `"int8_convrot"`）を足す。どれも §1-95 の計画で決めればよいが、台帳 §1-95 に「`weight_class` の扱いを決める」を 1 行足しておかないと、次の担当が表だけ足して効かない事故になる。

### J-5 H3 の 4 分類の正本の置き場
- 食い違い: MULTI_ENGINE §5.6「どのフィールドがどの分類かの正本は仕様書 §6.10」。仕様書 §6.10 は LTX 2.5 だけで、H3 は §6.2b (f) の `unsupported_features` しか無い（D-012）。
- 選択肢: (a) 仕様書に §6.11（MiniMax H3 の対応範囲。4 分類の表）を新設する。(b) 「H3 の 4 分類の正本は `services/engines/h3/adapter.py` の表そのもの」と MULTI_ENGINE §5.6 に書く（文書へ写さない）。(b) は文書の正本 1 箇所の原則に沿うが、LTX 2.5 と扱いが割れる。

### J-6 MULTI_ENGINE §7 の【オーナー裁定】の解除の根拠
- 食い違い: §7 の訂正注記「同日の MiniMax H3（3例目の系統）の第2弾で、**オーナーの裁定により**この保留は解除され」。研究ノート §10 の裁定 13〜35 に、この保留の解除を直接言う番号が無い（裁定 22・24 のスライダーと、計画の了承がそれに当たると読める）。
- 選択肢: (a) 根拠を「第 2 弾の計画の了承（2026-10-17。計画 PLAN_stage2 §3.3）」と書く。(b) オーナーに確認して裁定番号を振る（研究ノート §10 に追記）。§7 の見出しから【オーナー裁定】を外すかどうかも、この判断に従う。

### J-7（小）台帳 §1-95 の題の「非蒸留」
- 食い違い: 台帳 §1-95「MiniMax H3 **非蒸留** int8 ConvRot の快適上限の較正」。設計正本 §3「モデルはガイダンス蒸留済み」。研究ノート §13.9 は「ガイダンス蒸留済み」を要確認としている。題の「非蒸留」はステップ蒸留（Turbo）を使わないという意味と読めるが、ガイダンス蒸留と取り違えやすい。
- 選択肢: (a) 題を「MiniMax H3（Turbo なし・int8 ConvRot）の快適上限の較正」に。(b) そのまま。

---

## ④ README で誤りになった箇所（第 3 弾へ申し送り。H3 の追記はしない）

`README.md`（リポジトリ直下）。いずれも「今ある記述が誤りになった」もの。H3 を書き足すことは第 3 弾の仕事なので、ここでは誤りだけを挙げる。

- **R-1** 601 行目（§「凍結 API 契約（付録A・変更しない）」）「幅・高さは **64の倍数**（…、検証は [`api/models.py`] の `GenerateRequest` validator）。」— 検証は系統の規則表（`services/engines/ltx/request_rules.py`。`validate_for_family`）へ移った。64 の倍数は LTX の規則で、MiniMax H3 は 32 の倍数。
- **R-2** 602〜603 行目「フレーム数は **8n+1**（9, 17, 25, … 121）。尺 cap は **20s（481f=8×60+1）@24fps** …」「Distilled は **8 steps / CFG=1.0** 固定。」— 8n+1・ステップ 8 固定は LTX の規則になった（仕様書 付録 A は【LTX の規則】の札を付けた）。MiniMax H3 は 17n+5（22〜481）・24fps 固定・ステップ数 1〜50。節の題「凍結 API 契約（変更しない）」のままだと、系統ごとに違うことと食い違う。
- **R-3** 604 行目「`frame_idx` は `0`（開始フレーム）か 8n+1 で、それ以外の値を送っても 422 にはならず、8n+1 グリッドへ丸めて…」— MiniMax H3 では `frame_idx` が 0 以外、または 2 枚以上なら 422 `FEATURE_UNSUPPORTED`（`multi_keyframes`）。強度も既定 0.8 以外は 422（`keyframe_strength`）。
- **R-4** 834 行目「**指定しても断らず、黙って無視して生成を続ける項目もあります**——`guidance_scale` と `num_inference_steps` の2つです。…（ただし**この2つは、LTX 2.5 が受け付けるリクエストでは既定値から動かせないので、実際にこの行が出ることはありません**）」— LTX 2.5 の無視の項目は `shift_video`・`shift_audio` を加えて 4 つ（D-005）。shift は LTX 2.5 でも既定（`null`）以外の値を送れる（例: MCP の `submit_generate` の `shift_video`）ので、「実際にこの行が出ることはない」は誤り。
- **R-5**（誤りに近い）576 行目「**Base model（ベースモデル）**: …（LTX 2.3 ／ LTX 2.5 の切り替え）」— ドロップダウンには MiniMax H3 も並ぶ（未導入なら導入を促す案内が出る）。177 行目・306 行目の「venv を分けているのは、LTX 2.3 と LTX 2.5 が…」「**2つに分かれているのは**…」も、`.venv-engine-h3` を作った機械では 3 つになる（導入手順は第 3 弾）。340 行目の python の設定の表に `model.engine_python_h3` が無い。これらは第 3 弾の追記で一緒に直すのが自然。

---

## 付記（リポジトリ外・参考）

- 監督の自動メモリ `MEMORY.md` の MiniMax H3 の行は「次＝第 2 弾の計画（別セッション・プランモード）」のままで、第 2 弾の完了（2c-G2 合格・merge 待ち）が反映されていない。リポジトリの文書ではないので一覧には入れていない。
