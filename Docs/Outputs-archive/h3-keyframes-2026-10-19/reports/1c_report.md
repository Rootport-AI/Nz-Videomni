# 1c（Gradio と MCP）の実装報告

## 結論
計画の §3.4・§3.5・§4.3 を実装しました。バックエンドのテストは全件合格です（3,449 件合格・65 件スキップ・失敗 0。1a 時点の 3,447 件に、今回足した 2 件が加わった数です）。コミットはしていません。GPU も使っていません。

## 変更したファイル（`git diff --stat`、担当範囲だけ）
```
 gradio_ui/feature_scope.py       | 15 +++++++++++++--
 gradio_ui/handlers.py            | 13 +++++++++++--
 gradio_ui/i18n.py                |  6 ++++--
 gradio_ui/presets.py             |  6 ++++++
 gradio_ui/ui.py                  | 13 ++++++++++---
 mcp_server/params.py             | 12 ++++++++----   （diff --stat を取った後に docstring を 1 行直したので、今の数はこの値）
 mcp_server/server.py             |  3 ++-
 mcp_server/tools/generate.py     |  6 ++++--
 tests/test_gradio_handlers.py    |  5 ++++-
 tests/test_gradio_models.py      | 34 ++++++++++++++++++++++++++++++++++
 tests/test_mcp_tools_generate.py | 14 ++++++++++----
```

## Gradio
- `presets.py`: `KF_STRENGTH_DEFAULT = 0.8` を 1 か所に置きました。参照しているのは `ui.py` の 2 本のスライダー（キーフレームの各スロットと、連結のクリップ 1。どちらも 0.8 の写しだったため）、`handlers.py` の連結の予備値（:1485）、`feature_scope.RESET_VALUES` です。
- `feature_scope.py`: `FEATURE_UI["keyframe_strength"]` を `kf_strength_1`〜`kf_strength_{KF_MAX_SLOTS}` にしました。`RESET_VALUES` は各スロットを `KF_STRENGTH_DEFAULT` に戻します。型は `str | bool | float` に広げました。
- `ui.py`: `_gated_components` に `kf_strength_<n>` として各スロットのスライダー（`kf_slots[n-1][3]`）を登録しました。隠す処理と値の戻しは `refresh_model_dropdowns` が受け持ちます。この関数は既存の 3 つの経路、つまり更新ボタン（`model_refresh_btn.click`）・ページの読み込み・モデル読み込み後の再取得から呼ばれます。経路は既存のテストのコメントとコードで確かめました。
- `handlers.py`（単発の生成、:905 付近）: スロットの強度が `KF_STRENGTH_DEFAULT` と同じなら、送るデータに `strength` を入れません。違うときだけ入れます。アップロード用のタプルはそのままです（送るかどうかは送る直前の 1 か所で決めています）。連結と A2V の経路（:1485、LTX 専用）は今までどおり常に送ります。`batch.py` は変えていません。
- `constraints.py` には触っていません（`keyframes` は足していません）。

## MCP
- `params.py`: `strength: float | None = None` に変えました。docstring も直しました。
- `tools/generate.py`（submit_generate の `conditioning_images` の説明）と `server.py`（:48 の H3 の説明）から「先頭フレーム 1 枚」「strength は既定のまま」を消し、系統別の説明に書き換えました。`model_dump` はそのままなので、省略すると null を送ります。
- grep で見つけた残りの「8n+1」は、どれもフレーム数か参照フレーム数の話で、キーフレームの位置とは関係がないので変えていません。`multi_keyframes` は MCP 側にはありませんでした。generate.py :259 と :745-750 の説明には H3 の話が無い（連結は LTX 専用）ので、そのままです。

## 新しい文言
- Gradio の `cap_kf_grid`
  - 英語: "Frame positions are snapped server-side by the base model's rule (LTX: 0 and 8n+1; MiniMax H3: any frame from 0 to the last)."
  - 日本語: 「フレーム位置はサーバ側でベースモデルの規則に合わせられます（LTX は 0 と 8n+1、MiniMax H3 は 0 から最後までの任意のフレーム）。」
- ui.py のコメント（英語）: "the server snaps each position by the loaded base model's rule: LTX 0 and 8n+1, MiniMax H3 any frame 0..F-1"
- MCP の params.py（日本語）: 「``frame_idx`` はサーバー側でベースモデルの規則に寄せられる（0 は開始フレーム。LTX はそれ以外を 8n+1 へスナップ・クランプ、MiniMax H3 は 0〜``num_frames``−1 の任意のフレームをそのまま使い、それより後ろは最後のフレームに寄せる。どちらも最大 10 枚）。``strength`` は省略（``None``）でベースモデルの既定になる（LTX は 0.8）。MiniMax H3 には強度が無く、値を指定すると 422 FEATURE_UNSUPPORTED（``keyframe_strength``）になる。」
- MCP の generate.py: 「``frame_idx`` はベースモデルの規則に寄せられる（LTX は 0 と 8n+1、MiniMax H3 は 0〜``num_frames``−1 の任意のフレーム）。``strength`` は省略でベースモデルの既定（LTX は 0.8）。MiniMax H3 には強度が無く、指定すると 422 FEATURE_UNSUPPORTED。」
- MCP の server.py: 「H3 は単発の T2V と I2V（キーフレームは最大 10 枚・frame_idx は 0〜num_frames−1 の任意のフレーム・strength は指定しない）だけに対応し、…」

## テスト
- 隠す処理のテストは `tests/test_gradio_models.py` に置きました。指示書では `test_gradio_ui.py` でしたが、既存の隠す処理と値の戻しのテスト（vram16 など）と、補助関数の `_refresh_with`・`_gated_update` がこのファイルにあるため、そちらに揃えました。足したのは次の 2 件です。
  - H3（`unsupported_features` に `keyframe_strength`）では 10 本とも隠れ、0.8 に戻る。
  - LTX では 10 本とも見え、値は書き換えない。
- `test_gradio_handlers.py`: :359-363 の期待値を変えました。0.8 のスロットには `strength` の鍵が無く、0.6 と 0.4 のスロットにはある、という形です。既存の 1 件で両方を確かめられるので、新しいテストは足していません（計画の m13「任意のテストは足さない」に従いました）。
- `test_mcp_tools_generate.py`: submit_generate と submit_chain の既存のテストに、strength を省略した画像を 1 枚ずつ足しました。省略した画像は `"strength": None`、明示した画像はその値（0.9）で送られることを確かめています。古い H3 の文言を固定しているテストはありませんでした（grep で確認）。
- `test_gradio_batch_runner.py` は変える必要がありませんでした（全件合格）。
- 実行方法は `.venv\Scripts\python -m pytest tests -rs` です。結果は「3449 passed, 65 skipped, 63 warnings in 319.54s」。全文は `C:\Users\PHENOT~1\AppData\Local\Temp\claude\s--OriginalApps-12-Nz-LTX23-AviUtl2\6080e139-5d66-42d3-a48f-feb0e61dd152\scratchpad\kf197\1c\pytest_full_1c.txt` にあります。
  - 補足: pyproject の addopts が `-q` なので、`-q` を重ねると最後の件数の行が出ません。今回は `-q` を付けずに実行しました。

## エスカレーションと未対応
- 設計への疑問はありません。
- 判断を仰ぎたい点が 1 つあります。MCP の `list_models` の説明（`mcp_server/tools/system.py` :100-107）は `constraints` の中身を項目ごとに挙げていますが、1a で増えた `keyframes`（`{step, offset}`。LTX は `{8, 1}`、H3 は `{1, 0}`）が載っていません。計画の対象ファイルに入っていなかったので、書き足していません。足すかどうかは監督の判断に任せます。
- `feature_scope.py` は、これまで標準ライブラリ以外を読み込んでいませんでした。今回 `.presets` を読み込むようにしたので、gradio と config も読み込まれるようになりました。循環 import は無いことを確かめ、テストも全件合格しています。
