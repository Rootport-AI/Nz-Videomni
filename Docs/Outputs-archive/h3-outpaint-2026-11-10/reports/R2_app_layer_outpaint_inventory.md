# R2 アプリ層・MCP・操作パネルの棚卸し — MiniMax H3 の Outpainting（台帳 §1-125）

調査日: 2026-11-10 / リポジトリ: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`（ブランチ dev・HEAD e78f13a）/ 読み取りのみ（編集・テスト実行・GPU なし）

## 結論（先に）

1. **H3 で Outpainting を通すには、拒否表の `outpaint` の行を外すだけでは足りない。** LTX の Outpainting は元動画を `reference_video_id`（IC-LoRA〔制御用の追加学習重み〕の参照動画の欄）で受けており、この欄は H3 では 4 か所で断られる。①H3 の拒否表の `reference_video` の行、②スキーマ「`reference_video_id` には LoRA が 1 本以上要る」、③受付「参照動画には control 系の LoRA が要る」、④受付「参照動画つきは縦横 128 の倍数」。**元動画をどの欄で受けるかは API の形の判断なので、要エスカレーション（§5 の E1）。**
2. **系統に依存しない部品は多い。** 拒否表から配信・操作パネルへ自動で伝わる仕組み（`unsupported_features` → `featureScope.ts` → サブタブ／右クリックの門）、ffprobe による保持矩形と元動画の解像度の照合、フレーム数の照合、排他の検査、緑のキャンバスを作る関数（`pad_green_mp4`。色は引数で変えられる）、mock の幾何の印、パッド入力の UI がそのまま使える。
3. **LTX 固有の数字は 3 層に直書きされている。** 128／256／5:2／`in-outpainting`／8n+1／トークン式の ÷8 が、サーバー（`api/models.py`・`api/generate.py`）、操作パネル（`outpaintGeometry.ts`・`useOutpaintForm.ts`・`i18n/strings.ts`）、MCP（`mcp_server/tools/generate.py`）の 3 か所にある。H3 の Retake と同じく、配信の `constraints` に寄せるのが型になる。
4. **快適上限の線**: H3 の固定行は行の数を増やさない（設計正本 §3.1「性質」、快適上限表 §19.3）。固定行のマスクで割り当ての山は約 160MiB 上がったが、境界は動かなかった（§19.4）。この 2 つは、Single の線（S＝219,516）を共用できるという読みの根拠になる。ただし、①キャンバスの寸法での元動画の VAE〔画像を潜在へ圧縮する符号化器〕の符号化、②空間のマスク、の 2 つは測っていない。`config.py` には「新しい系統の `outpaint_budget` は較正と裁定が済むまで None のままにする」という規則（J1）がある。H3 の Edit 系（Retake）には、いまマーカーを出していない（表 §1.1）。

---

## (1) 箇所ごとの表

分類: **A**＝系統に依存せず H3 でそのまま使える／**B**＝LTX 固有で、H3 では迂回か置き換えが要る／**C**＝H3 のアダプタ・配信・モック・テストで新しい経路が要る。
フロントエンドのパスは `F/` ＝ `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/webui/src/` と略す。

### 1-1. スキーマ `api/models.py`

| path:line | 何をしているか | 分類 | 備考 |
|---|---|---|---|
| `api/models.py:57-76` | `OUTPAINT_MIN_KEEP_SIDE = 256`。根拠はラプラシアンピラミッドのブレンドのマスクの膨張（r×キャンバス長辺÷64） | B | 理由が LTX のブレンドだけに依る。engine 側にも同じ値の二重化がある（`engine/outpaint/canvas.py`。コメントの記述） |
| `api/models.py:230-276` `OutpaintSpec` | pad 4 本（0〜4096）・`blend_dilation_stage1/2`（既定 5/2、0〜15）・`freeze_source_audio`（既定 true） | pad＝A／blend＝B／freeze＝A（意味）・C（H3 の実装） | docstring が「公式 LTX-2.3 ICLoRA Outpaint ワークフローの再現」と明記。幾何の約束（width/height＝最終キャンバス、保持矩形＝元動画の解像度）は系統に依存しない |
| `api/models.py:719` | `GenerateRequest.outpaint: OutpaintSpec \| None` | A | |
| `api/models.py:775-779` | `reference_video_id` があるのに `loras` が空なら ValueError | **B（H3 では障害）** | H3 には IC-LoRA が無いので、元動画を `reference_video_id` で受けるとここで落ちる。全系統で効くスキーマの検査 |
| `api/models.py:808-814` | outpaint → `reference_video_id` が必須 | A（欄の選択は E1） | |
| `api/models.py:815-819` | pad の合計が 0 なら拒否 | A | |
| `api/models.py:820-824` | `conditioning_images` と排他 | A | H3 はキーフレームを受けるが、元動画が絵を決めるので排他のままでよい（推測） |
| `api/models.py:825-829` | `crop_output` と排他 | A | |
| `api/models.py:830-838` | 保持矩形の辺 ≥ 256 | B | 系統で分けるなら、この検査を `REQUEST_RULES` の側（`validate_for_family`）へ移す必要がある |
| `api/models.py:851-855` | inpaint と outpaint は排他 | A | |
| `api/models.py:93-117` `INPAINT_CANVAS_MULTIPLE=128`・`round_up_128` | inpaint の格子。outpaint は直接は使わない | — | 参考 |

### 1-2. 受付 `api/generate.py`・`api/errors.py`

| path:line | 何をしているか | 分類 | 備考 |
|---|---|---|---|
| `api/generate.py:109` | `engines.reject_unsupported(active_family, request)`。拒否表を最初に引く | A | ここで H3 の `outpaint`／`reference_video` の行が効く |
| `api/generate.py:121` | `engines.validate_for_family`（系統の数値の規則。H3 は 17n+5・32 の倍数・24fps） | A | outpaint の width/height（キャンバス）にもそのまま効く |
| `api/generate.py:162-170` | 参照動画が存在すること（404）と、**縦横が 128 の倍数**（`reference_resolution_invalid`） | **B（H3 では障害）** | 系統を見ていない。理由は「control アダプタの縮小係数 2」で、LTX の IC-LoRA の事情 |
| `api/generate.py:172-195` | LoRA の解決（404）。control 系なのに参照が無ければ `LORA_REQUIRES_REFERENCE`、参照があるのに control が無ければ `REFERENCE_REQUIRES_CONTROL_LORA`、前処理の衝突 | **B（H3 では障害）** | H3 には control 系の LoRA が無いので、参照動画の欄を使う限り 192-193 行で必ず 422 になる |
| `api/generate.py:197-203` | outpaint で前処理つきの LoRA があれば `OUTPAINT_PREPROCESS_CONFLICT` | B | H3 の LoRA（Style／Turbo）は前処理が "none" のはずなので、通っても害は無い（推測。H3 の登録簿の `preprocess` の値は確かめていない） |
| `api/generate.py:206-214` | ffprobe で元動画の解像度を測り、保持矩形と照合（`OUTPAINT_SOURCE_MISMATCH`） | A | 系統を見ていない |
| `api/generate.py:215-220` | 元動画のフレーム数 ≥ `num_frames`（`OUTPAINT_SOURCE_TOO_SHORT`） | A | 根拠の文言は「ブレンドがフレームを対にする」で LTX 寄りだが、H3 でも元の潜在を固定するのに要る |
| 系統の判定の場所 | **outpaint のブロックの中では系統を見ていない**。系統が効くのは 109・121 行の 2 回の取り次ぎだけ | — | 設計の規律（系統の事実はアダプタに置き、エンドポイントで分岐しない。`services/engines/__init__.py:207-226`） |
| `api/errors.py:451-463` | `OUTPAINT_PREPROCESS_CONFLICT`（文言が In-Outpainting IC-LoRA に触れる） | B | |
| `api/errors.py:466-482` | `OUTPAINT_SOURCE_MISMATCH` | A | |
| `api/errors.py:485-500` | `OUTPAINT_SOURCE_TOO_SHORT` | A（文言は LTX 寄り） | |
| `api/errors.py:358`・`:373`・`:660` | `lora_requires_reference`・`reference_requires_control_lora`・`reference_resolution_invalid` | B | 上の障害の 422 を作る関数 |
| `api/errors.py:738` | `feature_unsupported` | A | |

### 1-3. 映像の加工 `services/video_io.py`

| path:line | 何をしているか | 分類 | 備考 |
|---|---|---|---|
| `services/video_io.py:143-148` | `OUTPAINT_GREEN_HEX = "0x66FF00"`（In-Outpainting IC-LoRA が学習した緑） | B | |
| `services/video_io.py:151-253` `pad_green_mp4` | 元動画をキャンバスに置き、帯を塗り、可逆 RGB でちょうど `num_frames` 枚・映像だけの mp4 を書く。`scale` と `color` の引数がある（Alpha Gen は黒を使う） | **A（部品）**／緑の既定は B | H3 のキャンバス作りにも使える。fps をそろえる・枚数を固定する・可逆、の 3 点は H3 でも効く。帯の色（緑・黒・端の複製など）は §1-125 の「前提」どおり実験で決める |
| `services/video_io.py:486-` `fill_mask_green_mp4` | Inpainting のマスクの緑塗り | B | 参考 |
| `services/video_io.py:1026` `cut_window_mp4` | 窓の切り出し（Retake・Inpainting が使う） | A | LTX の Outpainting は元動画を先頭から使い、窓は切らない。H3 でも同じなら不要 |
| `services/video_io.py:1314`・`:1342` | `probe_resolution`・`frame_count` | A | |

### 1-4. ジョブの実行 `services/pipeline_manager.py`

| path:line | 何をしているか | 分類 | 備考 |
|---|---|---|---|
| `services/pipeline_manager.py:746-750` | `reference_video_id` を参照動画のパスに解決 | A／E1 | |
| `services/pipeline_manager.py:790-820` | outpaint の分岐: `outpaint_source_path` に元の upload を入れ、`output_dir/outpaint_canvas.mp4` に緑のキャンバスを作り、**`reference_video_path` をキャンバスへ差し替える**（IC-LoRA の経路を変えずに済ませる工夫） | **B**（差し替え）／キャンバス作りは A | H3 にはキャンバスを受ける IC-LoRA の経路が無い。キャンバスを別の引数で渡すか、H3 では差し替えないか、が要る（C）。系統は見ていない |
| `services/pipeline_manager.py:1044-1057` | `runner.generate(..., reference_video_path=…, outpaint_source_path=…, **runner_extra)` | A | H3Runner の `generate` は `outpaint_source_path` をすでに受けて backend へ渡す |
| `services/pipeline_manager.py:2216-2230` `_write_metadata` | `metadata.json` に **`outpaint` 専用の塊は無い**。outpaint の指定は `request`（`req.model_dump()`）の写しにだけ残る | A | LTX 2.5 のエンジンの事実は `ltx25` の塊（`outpaint25._ltx25_block`）に入る |
| `services/pipeline_manager.py:2316-2340` | H3 の `outcome.h3` → `metadata["h3"]`（`a2v`・`references` は最上位へ持ち上げる） | C | H3 の Outpainting の記録（マスクの行数・自己検査など）を足すなら、ここに `a2v` と同じ形の持ち上げが要るか判断する |
| `services/pipeline_manager.py:2395-2403` `_embed_recipe` | metadata の文字列を mp4 の comment タグへ埋める | A | 系統に依らない |

### 1-5. LTX のアダプタ・規則表

| path:line | 何をしているか | 分類 | 備考 |
|---|---|---|---|
| `services/engines/ltx/request_rules.py:1-30` | 「outpaint のスキーマの交差検査は全系統の規則としてスキーマに残す」と明記 | 参考 | 256 の下限を系統に寄せるなら、この方針の見直しになる |
| `services/engines/ltx/request_rules.py:186-225` `RequestRules.constraints()` | `GET /models` の `constraints` を作る。`source_audio`・`reference`・`chain`・`retake` は「値を持つ系統だけ鍵を出す」型 | **C（型）** | H3 の Outpainting に固有の数（格子・保持の下限など）を配るなら、ここに `outpaint_*` の欄を 1 つ足す（`retake_audio_strength` と同じ形） |
| `services/engines/ltx/adapter.py:224` | `HONOURED_FIELDS` に `outpaint` | B | |
| `services/engines/ltx/adapter.py:873-900` | `outpaint_source_path` の docstring（元の upload。音声だけ読む） | B | |
| `services/engines/ltx/adapter.py:2410-2430` | ワーカー本文 `outpaint: {source_path, canvas_width, canvas_height, pad_left/right/top/bottom, blend_dilation_stage1/2, freeze_source_audio}`。鍵があればワーカーは `generate_outpaint` へ行く | B | 鍵の名前（pad・canvas・source_path・freeze_source_audio）は H3 でもそのまま流用できる（「1 つの機能に 1 つのアプリ側の契約」という LTX 2.5 の方針。`ltx25/adapter.py:222-225`） |
| `services/engines/ltx/adapter.py:1246-1268`・`:1746-1760` | **mock**: width/height（＝キャンバス）で描き、保持矩形を緑の枠（102,255,0）で描く。元動画は開かない | A | 系統を見ていない。H3 の mock（`_MockBackendH3.generate` → `super().generate`）もこの枠を描く |
| `services/engines/ltx25/adapter.py:216-225`・`:271-274`・`:985-1000` | 2.3 と同じ鍵の `outpaint` 本文 → `engine25.outpaint25.run_outpaint` | B | |

### 1-6. H3 のアダプタ `services/engines/h3/adapter.py`

| path:line | 何をしているか | 分類 | 備考 |
|---|---|---|---|
| `:259` | `REJECT_TABLE` の `("reference_video_id", "reference_video", …)` | **B／C（障害）** | 元動画を `reference_video_id` で受ける場合、述語を `r.reference_video_id is not None and r.outpaint is None` のように狭める必要がある。そうすると監査テストの「従属の統治者は拒否表にある」（`GOVERNED_FIELDS` の 2 本の IC-LoRA の強度が `reference_video_id` に従属）と「重複は許可の 2 件だけ」（`_ALLOWED_OVERLAP`）に触れる → E1 |
| `:264` | `("outpaint", "outpaint", lambda r: r.outpaint is not None)` | C | 外すと `UNSUPPORTED_FEATURES`（`:450-455`、表から導く）から `outpaint` が自動で消える |
| `:279-318` `HONOURED_FIELDS` | 動作する欄の一覧 | C | `outpaint` を足す。監査 `test_honoured_fields_are_read_by_generate` は `_RealBackendH3.generate` の源に `request.outpaint` の文字列があることを要求する |
| `:326-332` `GOVERNED_FIELDS` | IC-LoRA の 2 つの強度は `reference_video_id` に従属 | 参考 | E1 の選び方で扱いが変わる |
| `:438-455` | `UNSUPPORTED_FEATURES` の docstring（「Outpainting／Inpainting／Alpha Gen は閉じる」） | C | 文言の更新 |
| `:156-206` `REQUEST_RULES` | 17n+5・32 の倍数・24fps・サンプリング 3 本・`retake_audio_strength` など | C | Outpainting の配信の欄を足すならここ（Retake の型） |
| `:828-1000` `_RealBackendH3.generate` | 引数に `outpaint_source_path` があるが、docstring（`:857-860`）は「拒否表で断るので値は来ない」と書く。本文に `outpaint` の鍵は無い | C | 本文に `outpaint` の鍵を足す（LTX と同じ鍵名にするかは設計の判断）。`reject_unsupported(request)` を後ろ盾として呼び直している（`:865`） |
| `:1279-1303` `_MockBackendH3.generate` | `super().generate` で LTX の mock の絵（保持矩形の枠つき）を作る。`a2v`／`references` の記録を足す | C | Outpainting の記録（`h3.outpaint` 相当）を mock でも足すかどうか |
| `:1554-1587` `H3Runner.generate` | `outpaint_source_path` を backend へ渡す | A | すでに配線済み |
| `:585-612` `validate_chain` 内の Retake | のりしろは「省略なら 22 を書き戻し、明示の 22 以外は 422」 | 参考（型） | Outpainting で H3 固有の値（ブレンドの段数など）を断るときの型 |

### 1-7. 監査テスト（`tests/`）

| path:line | 主張 | 分類 | 備考 |
|---|---|---|---|
| `tests/test_engine_h3_adapter.py:63-104` | 4 枚の表（拒否・無視・動作・従属）を合わせると `GenerateRequest.model_fields` とちょうど一致する。重複は `conditioning_images`・`nag_enabled` だけ。従属の統治者は必ず拒否表にある | C | `outpaint` を拒否表から動作へ移すのは 1 行の移動で済む。`reference_video_id` を両方に入れると `_ALLOWED_OVERLAP` を変えることになる |
| `tests/test_engine_h3_adapter.py:107-111` | 動作の欄は `_RealBackendH3.generate` の源に `request.<欄>` で現れる | C | |
| `tests/test_engine_h3_adapter.py:121-150`・`:156-157` | `REQUEST_OVERRIDES_H3["outpaint"]`（`reference_video_id`＋`loras`＋`width=768`）。拒否表の全行を網羅しているか検査 | C | 行を外したら、この項目を消す |
| `tests/test_engine_h3_adapter.py:210-233` | `UNSUPPORTED_FEATURES` の並びを丸ごと照合（`"outpaint"` を含む） | C | |
| `tests/test_engine_h3_adapter.py:1289-1290` | 両方の拒否表の機能名 ⊆ `UNSUPPORTED_FEATURES` | A | |
| `tests/test_engine_dispatch.py:243-246` | LTX 2.5 の features に `outpaint` が無い | A | |
| `tests/test_comfort_budgets.py:144-163` | `outpaint_budget` は ltx=42240・ltx25=46080、行には持たない、既定は None | 参考 | H3 の値を足すと、ここに H3 の 1 行が増える |
| `tests/test_gradio_constraints.py:249` | H3 のプロファイルは `outpaint_budget: None` | 参考 | |
| `tests/test_outpaint_api.py`（58 件）・`tests/test_ltx25_outpaint.py`・`tests/test_ltx25_api_guard.py`・`tests/test_ltx25_adapter.py`・`tests/test_outpaint_canvas.py`・`tests/test_outpaint_pyramid_blend.py` | LTX の Outpainting の受付・本文・engine の検査 | B | `test_outpaint_api.py` は H3 を扱っていない（grep で h3/minimax が 0 件） |
| `tests/test_mcp_tools_generate.py:1648-1879` | MCP の outpaint の本文・5:2・LoRA の自動注入・入力の型 | B | |

### 1-8. 快適上限の配信

| path:line | 何をしているか | 分類 | 備考 |
|---|---|---|---|
| `config.py:295-353` `EngineComfortProfile` | `outpaint_budget: int \| None = None`、`single_rows_budget: int \| None`。docstring に **J1**: 新しい系統は較正と裁定が済むまで `outpaint_budget` を None のままにする（「先例を写さない」） | 参考 | H3 で Single の線を共用するなら、新しい裁定と、どの鍵で配るか（`outpaint_budget` は LTX のトークン式の数）の判断が要る |
| `scripts/manifests/10-ltx23.json:47-50`・`20-ltx25.json:45-48` | `comfort.outpaint_budget` 42240／46080 | B | 正本は快適上限表 §9 |
| `scripts/manifests/40-minimaxh3.json:46` | `"comfort": {"single_rows_budget": 219516}` だけ | 参考 | `config.yaml`／`config.yaml.example` には H3 の comfort の値は無い（`config.yaml.example:67-72` は LTX の in-outpainting LoRA の登録だけ。表 §9.4 によると `config.yaml` の `limits.comfort_budgets` は無視される） |
| `F/shell/outpaintBudget.ts:33-36` `resolveOutpaintComfortBudget` | `limits.comfort_budgets[系統].outpaint_budget` を引く。null なら警告を出さない | A（H3 では null になり、何も出ない） | |

### 1-9. モック

| 箇所 | 何をしているか | 分類 | 備考 |
|---|---|---|---|
| バックエンドの mock（`ltx/adapter.py:1746-1760`） | 1-5 を参照。系統を見ずにキャンバスの寸法で描き、保持矩形の枠を描く | A | アプリ層（`pipeline_manager.py:800-820`）は mock でもキャンバスを ffmpeg で作る |
| `F/bridge/mockBridge.ts:587-600` | H3 の `MOCK_UNSUPPORTED_FEATURES` に `"outpaint"` がある（592 行） | C | Retake の型（§163.7）では、ここから外して `constraints` を足した |
| `F/bridge/mockBridge.ts:367-408` `MOCK_CONSTRAINTS_BY_FAMILY.h3` | H3 の配信の写し（`retake` まで） | C | outpaint の配信の欄を作るならここにも写す |
| `F/bridge/mockBridge.ts:57-60`・`:245` | `/loras` に `in-outpainting` を**系統によらず**出す | B | 実サーバーは H3 の一覧に LTX の IC-LoRA を出さない（`services/engines/__init__.py:189-205` `lora_scope_for`）。mock は実物と食い違う |
| `F/bridge/mockBridge.ts:1832-` `handleGenerate` | 単発の `/generate` は個別の検査だけで、outpaint の幾何も系統も見ずに完走する | A | |

### 1-10. MCP（`mcp_server/`。AI エージェント向けの道具の口）

| path:line | 何をしているか | 分類 | 備考 |
|---|---|---|---|
| `mcp_server/tools/generate.py:80-88` | 既定の 5、`_OUTPAINT_LORA_NAME = "in-outpainting"`、強度 1.0 | B | |
| `mcp_server/tools/generate.py:97-108` `_outpaint_stage2_from_stage1` | 5:2 の追従 | B | |
| `mcp_server/tools/generate.py:179-184` | `submit_generate` の 6 引数（pad 4 本・`outpaint_blend_dilation_stage1`・`outpaint_freeze_source_audio`） | pad＝A／blend＝B／freeze＝A（意味） | |
| `mcp_server/tools/generate.py:290-324`・`:453-466` | docstring: 「幾何条件 4 つ（128 の倍数・保持矩形＝元動画・256 以上・フレーム数）」「in-outpainting が必要で自動で追加」 | B | 系統ごとの書き分けが要る（D33 と同じ型） |
| `mcp_server/tools/generate.py:504-518` | `OUTPAINT_PADS_ALL_ZERO`（POST の前に弾く） | A | |
| `mcp_server/tools/generate.py:619-652` | `outpaint` の入れ子を組み、**`in-outpainting` を系統によらず自動で注入**する | **B（H3 では障害）** | MCP は読み込み中の系統を見ていない（`generate.py` に `engine_family` の参照なし）。H3 では LoRA の一覧の範囲に `in-outpainting` が無いので、注入すると 404 になる（推測。`lora_registry.py:307-324` は範囲ごとに解決する作りだが、H3 で 404 になることは試していない） |
| `mcp_server/server.py:105`・`:203-223` | `INSTRUCTIONS` の Outpainting の節（128・256・in-outpainting・5:2） | B | |
| `mcp_server/tools/uploads.py:111` | `upload_video` の docstring（「画角拡張の元動画も」） | A | |
| `Docs/MCP_SERVER_DESIGN.md:34` **D20** | 6 引数と LoRA の自動注入（オーナー裁定）、`blend_dilation_stage2` は引数にしない | B | H3 の扱いは新しい D 番号で裁く（D33 が Retake の前例） |
| `Docs/MCP_SERVER_DESIGN.md:47` **D33** | Retake の引数を系統によらない形に（既定 `None`＝送らない、サーバーが系統の既定を使う） | 型 | |

### 1-11. 操作パネル（`webui/`）

| path:line | 何をしているか | 分類 | 備考 |
|---|---|---|---|
| `F/modes/edit/outpaintGeometry.ts:47` `CANVAS_MULTIPLE = 128` | キャンバスの格子 | B | H3 は `constraints.dimension_multiple`（32）。ただし LTX の配信は 64 で、Outpainting の 128 は参照動画の規則から来る。**`dimensionMultiple` をそのまま読むと LTX が 64 に緩む** → 分岐の鍵が要る（§3） |
| `:52` `MIN_INNER_SIDE = 256` | 保持矩形の下限 | B | |
| `:59` `MAX_CANVAS_SIDE = 4096` | キャンバスの上限（スキーマの le=4096 と同じ） | A | |
| `:75` `PAD_SLIDER_MAX = 220`・`:83` `MAX_PAD = 4096` | スライダーと数値ボックスの上限 | A | 操作感の数で、系統に依らない |
| `:95-153` | `ZERO_PADS`・`canvasSize`・`clampPad`・`totalPad`・`centerPads` | A | |
| `:160` `MIN_NUM_FRAMES = 9`・`:172-180` `maxNumFrames`（8n+1） | フレーム数の格子 | B | H3 は 17n+5・下限 22（`constraints.frameGrid`・`minNumFrames`） |
| `:183-209` `latentFrameCount`（÷8）・`comfortTokenEstimate`・`isOverComfortBudget` | LTX のトークン式 | B | H3 の式は `F/shell/h3SequenceRows.ts` の `h3SequenceRows`／`h3ComfortFrames` |
| `:225-265` `BLEND_DILATION_*`・`DEFAULT_*`（5/2）・`featherWidthPx`・`stage2FromStage1` | マスクブラー | B | |
| `:316-370` `outpaintReasons` | 生成できない理由: 8n+1（`(frames-1)%8`）・128 の格子・4096・256・`loraMissing` | B（8n+1・128・256・LoRA）／A（プロンプト・素材の状態・4096・padsZero） | |
| `F/modes/edit/useOutpaintForm.ts:10`・`:46`・`:408-411` | `in-outpainting` をピン留め。`GET /loras` に control 種として無ければ `loraMissing` | B | H3 の一覧に `in-outpainting` は出ないので、H3 では必ずブロックされる |
| `:16`・`:228-240` | fps は `snapFrameRate`（自由・整数） | B | H3 は 24 固定（`fixedFrameRate(constraints)`） |
| `:389-403` | フレーム数の上限・スナップ（`snapNumFrames(value, MIN_NUM_FRAMES=9, …)`。8n+1） | B | |
| `:434-438` | 快適上限: `resolveOutpaintComfortBudget` → `comfortTokenEstimate` | B（式）／A（null で消える） | |
| `:439-471` `buildRequest` | `reference_video_id: sourceId`・`loras: [in-outpainting]`・`outpaint{pads, blend（既定なら省く）, freeze_source_audio: true}`・`accelerationRequestFields` | 参照＝E1／LoRA＝B／blend＝B／pads・freeze・加速＝A | サンプリング（steps・shift）を送っていない → H3 では C（Retake は `samplingRequestFields` を足した） |
| `F/modes/edit/OutpaintingPanel.tsx:86`・`:150-171` | マスクブラーのスライダーと、その説明（緑が残るときの案内） | B | |
| `F/modes/edit/OutpaintingPanel.tsx:101-104` | **LTX のレガシー表 `spill_free_frames` で目盛りと警告を出す**（`FALLBACK_APP_CONFIG` から。系統を見ていない） | B | 快適上限表 §1.1: H3 は `spill_free_frames` へ落ちない規則。H3 で開くと LTX の目盛りが出てしまう |
| `F/modes/edit/OutpaintingPanel.tsx:240-241` | 快適上限の警告の帯 | A（線が null なら出ない） | |
| `F/modes/edit/OutpaintingPanel.tsx:257-320` | パッドのスライダーと数値ボックス | A | |
| `F/modes/edit/OutpaintPreview.tsx:39-118` | キャンバス・元動画・ブレンドの帯の図 | A（図）／B（帯） | `blendBandPx` を 0 にすれば帯は描かれない（`:59`） |
| `F/modes/edit/editReasonMessages.ts:49-80` `buildOutpaintReasonMessages` | 理由コード → 文言（`loraName`・`minInnerSide` を引数で受ける） | B（一部） | Retake は格子を受ける関数へ直した（§163.4） |
| `F/i18n/strings.ts:296`・`:314`・`:357-367`（英）／`:2709`・`:2713`・`:2742-2749`（日） | 「128 の倍数」「緑が残る場合はマスクブラーを強める」が直書き | B | |
| `F/i18n/strings.ts:133-134`・`:2614-2615` | `unavailableOnBaseModel.outpainting`（案内の一行の文言） | A | |
| `F/shell/featureScope.ts:192` | `outpaint: { disables: ["editSubTab.outpainting"] }` | A | 配信から名前が消えれば自動で開く（Retake のとき、この表は変えていない。§163.6） |
| `F/shell/featureScope.ts:426-451` `editSubTabsDisabledFor` | サブタブごとの灰色 | A | |
| `F/shell/AppShell.tsx:134-148` `EDIT_INTENT_SUB_TABS`・`:748-761` | **閉じたサブタブへの右クリックを「案内の一行で止める」実装の場所**（`outpaint → outpainting`） | A | 名前が配信から消えれば自動で通る |
| `F/modes/edit/EditScreen.tsx:213-214` | 右クリックの意図 `outpaint` の初期サブタブ（閉じていれば他へ） | A | |
| `F/modes/edit/EditScreen.tsx:249-258` | `useOutpaintForm` には **`constraints` を渡していない**（Retake には渡す。`:283-288`） | C | Retake と同じく `constraints={baseModels.activeConstraints}`・`loraAudioHidden` を渡す配線が要る |
| `F/lora/controlLoras.ts:42`・`:54-72` | `OUTPAINT_LORA_NAME`、ほかのドロップダウンから隠す名前 | B | |
| `F/timeline/menuRouting.ts:304-305` `outpaintVideo` | 右クリック #15 → Edit・意図 `outpaint`・選択が必要 | A | |
| `native/src/plugin.cpp:369-399`・`:509` | 右クリックの項目 `outpaintVideo` | A | |
| `F/shell/generationConstraints.ts:35-90` | `GenerationConstraints`（`frameGrid`・`dimensionMultiple`・`frameRate`・`sampling`・`retake` …）と `resolveGenerationConstraints` | C | Outpainting の欄を配るなら、ここに `outpaint` を 1 つ足す（`retake` と同じ形） |
| `F/modes/single/defaultConfig.ts:93-142`・`F/bridge/mockBridge.ts:139-189` | オフラインのときの `outpaint_budget` の写し（ltx／ltx25） | 参考 | |

操作パネルの関連テスト（ファイル名に outpaint を含む、または outpaint を 5 件以上扱うもの）: `F/modes/edit/OutpaintingPanel.test.tsx`・`outpaintGeometry.test.ts`・`EditScreen.test.tsx`（79 件）・`F/shell/outpaintBudget.test.ts`・`featureScope.test.ts`（46 件。`:420-454` がサブタブの表）・`AppShell.featureScope.test.tsx`・`AppShell.minimaxH3.test.tsx`（`:207-215`・`:258`: H3 では Outpainting が灰色で、右クリックは案内で止まる）・`F/App.editRoute.test.tsx`・`App.editTab.test.tsx`・`F/timeline/menuRouting.test.ts`・`menuSelection.test.ts`・`F/bridge/mockBridge.test.ts`・`F/lora/controlLoras.test.ts`・`AppShell.controlLora.test.tsx`・`F/shell/comfortTable.test.ts`・`F/modes/edit/EditScreen.retake.test.tsx`・`useRetakeForm.test.tsx`（型の見本）。

---

## (2) LTX 固有の定数と文字列の一覧

| 値 | 意味 | 置き場（同じ値の写しをすべて） |
|---|---|---|
| **128** | キャンバスの格子（参照動画つきの規則。control アダプタの縮小係数 2 のため） | `api/generate.py:169`（`reference_resolution_invalid`）・`F/modes/edit/outpaintGeometry.ts:47`・`F/i18n/strings.ts:296,357-358,2709,2742-2743`・`mcp_server/tools/generate.py:302,320`・`mcp_server/server.py:213,220`・仕様書 §6.2 補足・設計ノート `OUTPAINTING_DESIGN_NOTES.md` §3-7 |
| **256** | 保持矩形の下限（ブレンドの膨張） | `api/models.py:76`・`engine/outpaint/canvas.py`（二重化。コメントの記述）・`F/.../outpaintGeometry.ts:52`・`F/modes/edit/editReasonMessages.ts:22`・`mcp_server/tools/generate.py:304`・`mcp_server/server.py:213`・仕様書 §6.2 の 12 |
| **4096** | pad 1 本とキャンバスの辺の上限 | `api/models.py:251-254`（pad の le）・`:658-659`（スキーマの width/height の le）・`F/.../outpaintGeometry.ts:59,83`・`F/i18n/strings.ts:2744-2745` |
| **5／2**（0〜15） | ブレンドの膨張の段数（stage1／stage2）と 5:2 の追従 | `api/models.py:274-275`・`F/.../outpaintGeometry.ts:225-265`・`mcp_server/tools/generate.py:83,97-108`・`F/.../useOutpaintForm.ts:441-445` |
| **÷64** | 膨張の実寸（段数×長辺÷64） | `F/.../outpaintGeometry.ts:248`（`featherWidthPx`）・`mcp_server/tools/generate.py:318` |
| **`#66FF00`（102,255,0）** | 緑の目印の色 | `services/video_io.py:148,581`・`services/engines/ltx/adapter.py:1757`（mock の枠） |
| **`in-outpainting`** | ピン留め／自動注入する IC-LoRA の名前 | `F/lora/controlLoras.ts:42`・`mcp_server/tools/generate.py:87`・`config.yaml.example:67-72`・`scripts/manifests/10-ltx23.json:172-195`・`F/bridge/mockBridge.ts:60,245`・`F/i18n/strings.ts`（`loraMissing`） |
| **強度 1.0** | 上の LoRA の強度 | `F/.../useOutpaintForm.ts:46`・`mcp_server/tools/generate.py:88` |
| **8n+1・下限 9** | LTX のフレームの格子 | `F/.../outpaintGeometry.ts:160,172-186,321-327`・`F/.../useOutpaintForm.ts:402`（`snapNumFrames(…, MIN_NUM_FRAMES)`） |
| **÷32×÷32×潜在（÷8）** | LTX のトークン式 | `F/.../outpaintGeometry.ts:183-209` |
| **42,240／46,080** | `outpaint_budget` | `scripts/manifests/10-ltx23.json:50`・`20-ltx25.json:48`・`F/modes/single/defaultConfig.ts:130,142`・`F/bridge/mockBridge.ts:175,187`・快適上限表 §9 |
| `spill_free_frames` | LTX のレガシーの目盛り | `F/modes/edit/OutpaintingPanel.tsx:101-104` |
| `freeze_source_audio: true` | 元の音声を固定（公式のノード 5392） | `api/models.py:281`・`F/.../useOutpaintForm.ts:465`・`mcp_server/tools/generate.py:184,636-637` |

H3 の側の対応する値（参考）: 縦横の倍数 32（`services/engines/h3/adapter.py:126`）、17n+5・下限 22（`:116-121`）、24fps 固定（`:128`）、1 トークン＝32×32 画素（VAE 16×パッチ 2）。

---

## (3) H3 の Retake を動作へ移したときの型と、Outpainting で同じ型が使える箇所

### 3-1. Retake の変更点（設計正本 `Docs/MINIMAX_H3_DESIGN.md` §3.4、フロントエンド `Docs/DEVLOG.md` §163、MCP D33）

| 層 | Retake でしたこと | Outpainting で同じ型が使えるか |
|---|---|---|
| 拒否表 | `CHAIN_REJECT_TABLE` から `retake` の行を外し、`CHAIN_HONOURED_FIELDS` へ入れた。`unsupported_features` は自動で消える | **使える**（`REJECT_TABLE:264` の行を外し `HONOURED_FIELDS` へ）。ただし `reference_video` の行（`:259`）が残る問題は Retake には無かった（Retake は専用の欄 `retake.video_id` で元動画を受ける） → E1 |
| 系統の値の書き戻し | のりしろは「省略なら 22、明示の 22 以外は 422」、`regenerate_audio=false` は 422、`audio_strength` は省略なら配信の既定 | **使える**: `blend_dilation_*` のように H3 で意味の無い欄を断るか、黙って無視するか（`IGNORED_FIELDS`）。既定 5/2 は省略と見分けにくい（スキーマの既定が 5/2 で `model_fields_set` を使う必要がある。Retake の head_px/tail_px と同じ技法） |
| LTX 側の欄の拒否 | LTX は `audio_strength` を拒否表ではなく系統の規則（`chain_rules.py`）の 1 行で断った（拒否表の「断る名前は必ず配信する」の不変条件を守るため） | H3 専用の欄を足すなら同じ型 |
| 配信 `constraints` | `RequestRules.retake_audio_strength` → `constraints["retake"]`（H3 だけ鍵を出す）。窓の下限・上限・のりしろは既存の値から読めるので足さなかった | **使える**: 鍵の有無が操作パネルの分岐を兼ねる（`h3 = rules.retake !== null`）。Outpainting でも `constraints.outpaint` の有無で 1 か所だけ分岐し、格子（32 か 128 か）・保持の下限などを配信から読む形になる。足す欄は最小にする（Retake の「過剰設計の指摘」の前例。§163.6） |
| 操作パネル | `useRetakeForm` が `constraints` を読む（格子・上限・縦横の倍数・固定 fps・サンプリング）。ベースモデルの切り替えで値を丸め直す。H3 では音声の Strength のスライダーを出し、Stage-2 の欄を隠す。文言を「格子を受ける関数」に直す | **使える**: `useOutpaintForm` に `constraints` を通す（`EditScreen.tsx:249` の配線）。8n+1・128・fps・サンプリングを配信から読む。H3 ではマスクブラーと LoRA のピン留めを隠す（または置き換える） |
| 右クリック | `EDIT_INTENT_SUB_TABS`（`AppShell.tsx:142-148`）。閉じたサブタブは案内の一行で止める | **変更不要**（名前が配信から消えれば通る） |
| 表 `FEATURE_UI` | 変えていない | **変更不要** |
| モック | `mockBridge.ts` の H3 の `unsupported_features` から `retake` を外し、`constraints.retake` を足した | **同じ作業が要る**（`:592` と `:367-408`）。加えて `/loras` の `in-outpainting` を系統で分けるかどうか |
| MCP | D33: 系統の既定を MCP が持たず、`None` なら送らない。docstring と `INSTRUCTIONS` を系統ごとに書き分ける | **使える**: LoRA の自動注入（D20）を系統で分ける必要がある。MCP は系統を知らないので、①サーバーが H3 で注入を受け流す、②MCP が `list_models` を引く、③注入をやめる、のどれかを裁く → E4 |
| Gradio | 画面がもともと無いので触っていない | 同じ（見送りは確認済み） |
| 監査テスト | 4 枚の表の監査を維持し、fixture を更新 | 同じ |

---

## (4) 快適上限の線の扱いと根拠

**事実（文書とコードから）**

1. H3 の線は総行数 S（DiT〔ノイズ除去の本体〕に入る行の合計）の 1 本で、219,516（`40-minimaxh3.json:46`、快適上限表 §1.1・§17.5 (a)）。境界は VRAM 16GB mode の上限での OOM（GPU メモリ不足）で、退避は出ていない（§17.5 (d)）。
2. **固定行は行を増やさない**: 「RoPE の座標は変えない。行の数も増えないので、系列の長さも変わらない」（設計正本 §3.1「性質」）。「重なり 22 フレームは、クリップ自身の映像・音声の行の先頭を固定行にするだけで、行を増やしません」（表 §19.3）。
3. **固定行は境界を動かさなかった**: 固定行のマスクで割り当ての山は約 160MiB 上がったが、予約の山は r1080_362 より 12MiB 低く、cap も下がらなかった。固定行つきのクリップは線ちょうどの 219,516 で 2 回快適だった（表 §19.4・§19.5 (a)）。V2V の点 F（入力動画の符号化つき）も 219,516 で快適（§19.4。1 回だけ）。
4. LTX で Outpainting の線を Single と分けたのは、Outpainting に固有の負荷（素材の VAE の符号化とブレンド・IC-LoRA の参照の行）があったため。拡張だけを外した対照でも、それを実測で確かめた（表 §9.1・§9.4）。H3 の計画（台帳 §1-125）には IC-LoRA の参照の行もピラミッドのブレンドも無い。
5. `config.py:309-322` の **J1**: 新しい系統の `outpaint_budget` は「較正と裁定が済むまで None」で、先例を写してはならない。
6. H3 の Edit 系には、いまマーカーを出していない（表 §1.1「Edit 系と Settings の表には出さない」。Retake も出していない）。

**H3 で Outpainting の線を Single と共用できるか**

- 共用を支える根拠: 事実 2・3・4。H3 の Outpainting は「キャンバスの寸法の生成＋元の領域の行をマスク 0 で固定」なので、S はキャンバスの寸法での T2V と同じ式になる（cond 0、参照 0）。固定行のマスクは連結で実測済みで、境界を動かさなかった。
- 共用の前に測っていないこと: ①キャンバスの寸法での元動画の VAE の符号化（Retake・V2V は生成の寸法で符号化しており、段も `encode` で、線を決めるノイズ除去の段の外にある）。②空間のマスク（`[T', 縦, 横]`。§1-124 で実装する予定で、まだ無い）の山の増え方。マスク自体は f32 で S 行 1 本ぶんなので小さいと見られる（推測）。③音声の固定（`freeze_source_audio` 相当）は A2V・Retake で実測済みの仕組み。
- 判断: 「共用できる見込みは高いが、J1 により、較正（最低限、線の直下と直上の 2 点）とオーナーの裁定が要る」。**どの鍵で配るか**も決める必要がある。`outpaint_budget` は LTX のトークン式の数なので H3 の S の線をそこに入れると式が合わない。案は (a) `single_rows_budget` を Outpainting にも当てる裁定を出す（新しい鍵は作らない。Chained の共用の前例 §19 と同じ形）、(b) 操作パネルが H3 では `h3SequenceRows` で数える → E5。

---

## (5) 未確認の点と要エスカレーション

**未確認（今回の調査で確かめていない事実）**

- H3 の LoRA の範囲で `in-outpainting` の解決が 404 になること（`lora_registry.py` の範囲の作りからの推測）。
- H3 の登録簿の LoRA（Style／Turbo）の `preprocess` の値が "none" であること（`OUTPAINT_PREPROCESS_CONFLICT` を素通りするか）。
- `engine/outpaint/canvas.py` の 256 の二重化（コメントの記述を見ただけで、ファイルは開いていない）。
- フロントエンド `Docs/API_REFERENCE.md` の Outpainting の表の正確な節番号（grep で 13 か所を見ただけ。`:89`・`:168-191`・`:575-587`・`:720` 付近）。
- `engine_h3/` には `outpaint` の語が 1 件も無い（エンジンは未着手）。空間のマスク（§1-124）も未実装。

**要エスカレーション**

- **E1（API の形）**: H3 の Outpainting の元動画をどの欄で受けるか。
  - (i) LTX と同じ `reference_video_id`。この場合は 4 か所を系統で緩める: `REJECT_TABLE:259` の述語、`api/models.py:775-779`、`api/generate.py:169`（128）、`api/generate.py:192-193`（control 系の LoRA）。後ろの 3 か所は系統を知らない層なので、アダプタへ移すか例外を書くことになり、「エンドポイントで系統の分岐をしない」規律（`services/engines/__init__.py:207-226`）に触れる。監査テストの `_ALLOWED_OVERLAP`・`GOVERNED_FIELDS` にも触れる。
  - (ii) `OutpaintSpec` に元動画の欄を足す（Retake の `retake.video_id` の前例）。LTX の契約は足すだけで済むが、LTX にも二つ目の入口ができる。
  - オーナーの方針（ベースモデルに最適な設計が主）からは (ii) 寄りと読めるが、契約の判断なので裁定が要る。
- **E2（256 の下限・128 の格子の置き場）**: スキーマにある 256 の下限（全系統）を系統の規則へ移すか。H3 の格子（32、トークン境界に合わせて pad も 32 の倍数にするか）をどう決めるか。保持矩形＝元動画の解像度という約束のままだと、元動画が 32 の倍数でなければ境目のトークンが元と新の両方にまたがる。
- **E3（`blend_dilation_*`・`freeze_source_audio` の H3 での意味）**: H3 にはピラミッドのブレンドが無い。拒否・無視・別の意味に読み替える、のどれにするか。台帳 §1-125 の「前提」は脱緑ブレンドの類いを実験で決めるとしている。`freeze_source_audio` は A2V／Retake の音声マスク（Strength）と同じものに当たる可能性がある（Strength の連続値にするか真偽のままにするか）。
- **E4（MCP の LoRA の自動注入）**: D20 は系統を見ずに `in-outpainting` を注入する。H3 では障害になる。どう分けるか（新しい D 番号）。
- **E5（快適上限）**: §4 の較正と、配る鍵の裁定（J1）。
- **E6（順序）**: 台帳 §1-125 の「留意」は「§1-124（Inpainting）の後に着手する」「マスクの細粒化は §1-124 で実装し Outpainting が流用する」と書いている。今回 Outpainting を先にするなら、空間のマスクを Outpainting で先に作ることになる。着手の順はその都度オーナーが決める規則なので、確認が要る。
- **E7（LoRA のピン留め）**: H3 の Outpainting で、利用者の LoRA（Turbo・Style）を選べるようにするか（LTX は `in-outpainting` 1 本の固定。Retake はスタイル LoRA を許した前例がある。DEVLOG §103）。

---

## (6) 推測（事実と分けて書く）

- H3 の Outpainting の本文の鍵は、LTX と同じ名前（`outpaint: {source_path, canvas_width, canvas_height, pad_*, freeze_source_audio}`）にすると、ワーカーのログを比べやすい（LTX 2.5 の「1 つの機能に 1 つのアプリ側の契約」の方針に沿う）。`blend_dilation_*` は H3 には送らない形になりそう。
- キャンバスは `pad_green_mp4(color=…)` で作れる。ただし H3 の VAE の空間の畳み込みは境目をまたいで見るので、帯の色が元の領域の境目のトークンの潜在に滲む。固定行は元動画そのものを（キャンバスに置かずに）符号化した潜在から取り、キャンバスは位置合わせにだけ使う、という分け方もありうる。これはエンジン（R2 の範囲外）の判断。
- 操作パネルの分岐は `constraints.outpaint` の有無の 1 か所に集められる。LTX は配信を変えずに今の 128・256・5:2・LoRA のピン留めのまま、H3 は配信の格子・17n+5・24fps・サンプリングで描き、マスクブラーと LoRA の行を隠す。
- 操作パネルの `OutpaintingPanel.tsx:101-104` の `spill_free_frames` の目盛りは、H3 では消す必要がある（快適上限表 §1.1 の規則）。消し方は Single の「rows-line の系統では spill へ落ちない」判定（`single_rows_budget` が null でないか）を流用するのが自然。
- `metadata.json` には、LTX と同じく `request` の写しに outpaint が残る。H3 のエンジンの事実（固定したトークンの数・自己検査など）は `h3` の塊に入れれば、アプリ層の持ち上げ（`pipeline_manager.py:2316-2340`）は不要。

---

## 文書の棚卸し（場所の列挙だけ。更新は後日）

| 文書 | outpaint を系統別に記している場所 |
|---|---|
| `Videomni_Backend_Specification.md` | §6.2 の表 `:764`、補足 12（5 検査）`:787-792`、補足の H3 の参照動画 `:785`、§6.2b (f) H3 の `unsupported_features` `:960`、§6.3 `OutpaintSpec` `:1053-1075`（「エンジン系統による違い」`:1063`）、§6.10（LTX 2.5 の画角拡張）`:1742`、出力のファイル `outpaint_canvas.mp4` `:1970`、変更履歴 v0.5.38〜40 `:89-91` |
| `Docs/MINIMAX_H3_DESIGN.md` | 裁定 30 `:60`・裁定 34 と更新 `:65-66`・§3.1「粒度」`:402-405`・§3.4「操作パネル」`:834`・§12「書かないもの」`:1290`（画角拡張） |
| `Docs/MULTI_ENGINE_DESIGN.md` | `unsupported_features` の変遷 `:413-417`・`:743`、推論モードの表 `:500-506`・`:518`、「続き: 画角拡張（2026-08-29）」`:847-855` |
| `Docs/MCP_SERVER_DESIGN.md` | D20 `:34`、テストの表 `:123`、D33（Retake の型）`:47` |
| `Docs/COMFORT_LIMIT_TABLE.md` | §1.1 `:25`・`:105`（留保 4）、§4.9 の留保 `:334-335`、§9（全体）`:388-438`、§9.4 の Inpainting の流用 `:435`、§10.4 `:504` |
| `README.md` | `:344`（重みの表）・`:409`・`:520`・`:641`（Edit タブ）・`:990-995`（H3 でできないこと）・`:1117`（MCP） |
| フロントエンド `Docs/API_REFERENCE.md` | `:89`・`:168-191`（`outpaint_budget`）・`:575-587`・`:720` |
| フロントエンド `Docs/OUTPAINTING_DESIGN_NOTES.md` | 全体が LTX の設計。§3-3（Inpainting とのマスクの共通性）・§3-7（128 の継承）・§4-5（快適上限）・§5-A〜5-D |
| フロントエンド `Docs/DEVLOG.md` | §63・§64・§66・§67・§75.2・§92・§93・§94・§95・§108・§110・§111（Outpainting）、§163（H3 の Retake の型） |
| フロントエンド `Docs/RIGHTCLICK_REDESIGN_SPEC.md`・`Docs/BRIDGE_CONTRACT.md` | #15 `:134`・`:171-176`・`:362`・`:570`／`BRIDGE_CONTRACT.md:334` |
| `Docs/PENDING_TASKS.md` | §1-124 `:31-36`・§1-125 `:38-42`・§3-59（Outpainting の連結対応）`:226-228` |
