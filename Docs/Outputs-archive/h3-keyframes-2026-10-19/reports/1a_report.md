# 1a（バックエンド）実装報告 — 台帳 §1-97 第 1 弾

## 結論

計画 §2.1〜2.4・§3.1・§3.2・§4.1・§4.2・§7.2 を実装しました。コミットはしていません。GPU は使っていません（GPU のテストはスキップのまま）。テストは全部合格です。

- アプリ venv の pytest 全件: **3,447 件合格・65 件スキップ・失敗 0**（`PATH` の先頭に `tools\ffmpeg\bin`、`-o addopts=""`）。出力は `scratchpad\kf197\pytest_full.txt`。
- エンジン venv（`scripts/run_h3_pytest.py tests/test_h3_engine.py tests/test_h3_media.py tests/test_h3_format.py`・`PYTHONUTF8=1`）: **165 件合格・11 件スキップ**（スキップは全部 GPU のテスト）。

## float64 の総当たりの結果と、残した式

- F = 17n+5（22〜481 の 28 通り）× L = 0〜8192 の 229,404 組を調べました。
- `L + (5/3)(F−1)` が正解データの式 `(float(L) + span) − 5/3` と全桁一致したのは **141,203 組**。一致しない組が 88,201 組あります（例: F=22・L=0 で、正解データの式は 35.00000000000001、単純な式は 35.0）。
- このため **F−1 だけを特別に扱う分岐は残しました**（引き算はしていません）。`keyframe_t` の F−1 は正解データと同じ組み方・同じ順序の式で、全 229,404 組で全桁一致することをテストで確かめています（`test_keyframe_t_last_frame_is_the_oracle_expression_bit_for_bit`。一致の件数は表示し、`matched < total` も確かめる）。

## 変更したファイル（バックエンドのみ。`git diff --stat` の該当部分）

```
 api/generate.py                        |   3 +-
 api/models.py                          |  92 +++----
 config.py                              |   6 -
 config.yaml.example                    |   5 -
 engine/pipeline/chain_pipeline.py      |   2 +-
 engine/pipeline/fast_video_pipeline.py |   4 +-
 engine_h3/audio_vae.py                 |   2 +-   (docstring だけ)
 engine_h3/dit.py                       |   5 +-   (docstring だけ)
 engine_h3/image_cond.py                |   6 +-
 engine_h3/pipeline_h3.py               | 175 +++++++++----
 engine_h3/rope.py                      |  65 +++--
 engine_h3/text_encoder.py              |  74 +++---
 engine_h3/video_vae.py                 |   6 +-   (docstring だけ)
 engine_h3/worker.py                    |  46 +++-
 services/engines/__init__.py           |   8 +
 services/engines/h3/adapter.py         |  41 ++-
 services/engines/ltx/adapter.py        |   6 +-   (コメントだけ)
 services/engines/ltx/request_rules.py  |  85 ++++++-
 tests/test_a2v_chain.py                |   9 +-
 tests/test_chain.py                    |  22 +-
 tests/test_engine_h3_adapter.py        |  60 +++--
 tests/test_h3_engine.py                | 280 ++++++++++++++++++++-
 tests/test_ltx25_adapter.py            |  21 ++
 tests/test_smoke.py                    |   4 +-
 tests/test_validation.py               | 132 ++++++++--
```

`config.yaml`（git の管理外）からも 2 つのキーとコメントを消しました。`lowvram.py` は変えていません（直す単数形がありませんでした）。作業ツリーにある `AviUtl2-Plugin/...` の差分は 1b の担当者のもので、私は触っていません。

## 中身の要点

- **RequestRules**: 3 項目を足しました（`keyframe_step=1`・`keyframe_offset=0`・`keyframe_strength_default=None`）。`constraints()` は `"keyframes": {"step", "offset"}` を公開します。`normalize_keyframes` は計画 §2.1 の式どおりで、`frame_idx` と既定の strength を書き戻し、違反があれば文言を返します。LTX の規則は 8／1／0.8 です。
- **validate_for_family**: `fill_defaults` → `violation` → `normalize_keyframes` の順にしました。文言が返れば `request_validation_failed` です。
- **api/models.py**: strength を `float | None = None` にしました。`_normalize_conditioning_images` は削除し、2 つの validator には枚数の上限の検査だけを残しました。連結の側は `LTX_REQUEST_RULES.normalize_keyframes` を呼びます。循環 import は起きません（`request_rules.py` が api を import するのは型検査のときだけです）。
- **H3 アダプタ**: `_STRENGTH_DEFAULT`・`_multi_keyframes`・`multi_keyframes` の行を削除し、strength は「指定されていれば拒否」にしました。payload と `images_used` は変えていません。
- **engine_h3**:
  - `build_coords(cond_frame_idx=…)` と `keyframe_t` を追加しました。範囲外と重複は ValueError です。
  - `validate_request` は 1〜N 枚を受け付けます（位置は 0〜F−1・重複なし・パスが空でない）。
  - `generate` は画像を frame_idx の昇順に並べ（安定ソート）、全部の画像を読み込みます。画像読み取り部（vision tower）に渡すのは 0 と F−1 の画像だけです。
  - 進捗の分母は「キーフレームが 1 枚以上なら 4、無ければ 3」です。
  - `_encode_image` は画像の符号化器を 1 度だけ組み、1 枚ずつ符号化してから連結します。
  - ノイズは小さな関数 `noisy_condition` に分けました。`[grid, 96]` を 1 度だけ引き、`repeat(K, 1)` で K 枚に繰り返します。K=1 では今の式と全桁一致することをテストで確かめています。
  - `source_size` はリストになりました。
- **text_encoder**: `presentation(..., image_tokens: Sequence[int])`・`vision_inputs(images)`（処理器の呼び出しは 1 回）・`encode(prompt, images=())`・`read_report` に `vision_tokens_per_image` を足しました。
- **worker**:
  - `--keyframe FRAME=PATH`（複数可）を足しました。`--image` は frame 0 の別名として残しています。
  - 形の誤りと位置の重複は、読み込みの前に `_selftest_images` で拒否します。
- **計画に無い小さな追加**:
  - `_encode` に渡す画像は、位置引数ではなくキーワード（`images=`）で渡しています。既存テストの差し替え（`lambda prompt, phases, **kw`）がそのまま使えるようにするためです。
  - `audio_vae.py`・`video_vae.py` の docstring にあった「先頭フレームだけ」「1 枚」という単数形も直しました。

## 新しい 422 の文言など（正確な文字列）

- H3 で strength を指定したとき（FEATURE_UNSUPPORTED の detail）: 「MiniMax H3 にはキーフレームの強度がありません（strength は指定しないでください）。」
- 位置の重複（VALIDATION_ERROR。文言は今のまま）:
  - `conditioning_images: frame_idx {a} and {b} both snap to frame {n} (positions must be distinct)`
  - `two conditioning images at frame_idx 0 (the start frame can only be set once)`
  - 連結では先頭に `clips[i]: ` が付きます。
- 枚数の上限（文言は今のまま）:
  - `at most 10 conditioning images are supported`
  - 連結では `clips[i]: at most 10 conditioning images are supported`
- エンジン内の文言（ワーカーの error として出るもの）:
  - 「キーフレームの位置は 0〜{F−1} にしてください（frame_idx=…）」
  - 「キーフレームの位置が重複しています（frame_idx=…）」
  - 「キーフレームの画像のパスが空です（frame_idx=…）」
  - image_cond: 「キーフレームの画像が見つかりません／を読めません／の大きさが不正です」
  - CLI: 「--keyframe は FRAME=PATH の形で渡してください（…）」「キーフレームの位置が重複しています（frame_idx=[…]。--image は --keyframe 0=PATH と同じです）」

## 追加・変更したテスト（計画 §4.1・§4.2）

- **test_h3_engine**:
  - 総当たり／`keyframe_t` の先頭・中間・エラー／`build_coords` の 3 画角 × 7 通りの位置の組とエラー
  - presentation の 2 枚
  - validate_request
  - ワーカーの複数枚と `--keyframe`・重複の拒否
  - ノイズ（K=1 が今の式と全桁一致・K=3 で各区間が同じノイズ）
  - 本物の `_encode` と `_encode_image` を通す generate（昇順・画像読み取り部には 0 と F−1 だけ・ブロックの並び・進捗の分母 4／T2V は 3・末尾だけ・中間だけ）
  - 小さな DiT で本物の `_denoise` を K=2 で回す（cond の行数 8・形が合わなければ拒否）
- **test_validation**:
  - 寄せの計算を `LTX_REQUEST_RULES.normalize_keyframes` を直接呼ぶ形にしました（期待値は不変。F=9 も足しました）。
  - 重複の文言、H3 の表、「スキーマでは寄せず validate_for_family で寄る」、「数値の違反が寄せより先」、strength を省略したとき（LTX は 0.8・明示した値はそのまま・H3 は None）、連結の clips[0] が 0.8 になること。
- **test_engine_h3_adapter**: multi_keyframes を削除し、strength 0.8 も拒否されることを確かめます。文言・一覧・`keyframes` の制約・3 枚の payload・mock の I2V（0 と 30→21、strength は null）。
- **test_ltx25_adapter**（連結で strength を省略すると 0.8 が届く）、**test_chain**（HTTP で 10→9・0.8 が記録される）、**test_a2v_chain**・**test_smoke**（strength を省略しても metadata には 0.8）。
- 計画 §7.2 の確認: `validate_for_family` を通さずに GenerateRequest を LTX の runner へ渡すテストで、strength を省略しているものはありませんでした。単発・Alpha・連結の 3 つの経路では、None は LTX のアダプタに届きません。

## エスカレーション・やり残し

- 設計への疑問はありません。設計も変えていません。
- Gradio と MCP のテスト（1c の担当範囲）は、今の状態でも全部合格しています。1c が Gradio の「0.8 なら鍵を送らない」を入れたときは、`test_gradio_handlers.py:359-363` の期待値をそちらで直す必要があります。
- 正解データと GPU で照合する 1-G2 の台本は本担当の範囲外のため、作っていません。
