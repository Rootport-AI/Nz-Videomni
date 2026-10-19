# 1-G5 コードレビュー — 台帳 §1-97 第 1 弾（末尾キーフレーム）

対象: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni` の未コミットの作業ツリー（62 ファイル）。計画 `scalable-floating-thimble.md`、実装報告 1a／1b／1c、門の準備の報告 g2_prep と突き合わせた。ファイルは一切変えていない。GPU も使っていない。

## 結論

- **C（挙動の誤り・契約の破れ・全桁一致の経路の後退）: 0 件**
- **M（門や利用者が当たる穴）: 1 件**
- **m（分かりやすさ・死んだコード・古いコメント）: 6 件**

T2V（画像なし）と先頭だけの I2V（画像から動画を作る生成）は、コードを読んだ範囲では次の 4 つが変更前と同じです。

- トークン列
- 乱数の取り出しの順（映像 → 音声 → 条件行 1 回）
- 条件行の値
- RoPE（回転位置符号化）の座標と時間の番号

指紋での最終確認は 1-G3 に任せます。

## 自分で回した検査（証拠）

- アプリの venv で pytest を回した: `test_validation`・`test_engine_h3_adapter`・`test_gradio_models`・`test_gradio_handlers`・`test_mcp_tools_generate`・`test_chain` の 594 件が合格。
- エンジンの venv（`scripts/run_h3_pytest.py`、CUDA は見えない状態）で `tests/test_h3_engine.py` のキーフレーム関連（`-k "keyframe or coords or noise or presentation or validation or worker or denoise or t2v_encode or last_only"`）が全件合格。
- フロント: `npm run typecheck` はエラー 0。vitest は `keyframeGrid`・`keyframeUtils`・`generationConstraints`・`KeyframesPanel`・`featureScope`・`mockBridge.h3` の 6 ファイル・129 件が合格。

---

## C（該当なし）

全桁一致の確認で読んだところ（観点 1）:

- **乱数**: `noisy_condition`（`engine_h3/pipeline_h3.py:223-238`）は 3 回目の取り出しで `torch.randn((frame_rows, 96), generator=gen, dtype=float32)` を 1 回だけ引く。旧コードの `torch.randn(tuple(cond_clean.shape), ...)` と同じ形・同じ dtype・同じ順。混ぜ方の式も旧と同じ順 `KEYFRAME_NOISE_AUG * clean + (1.0 - KEYFRAME_NOISE_AUG) * noise`。
  - K=1 のときに増えた処理は 2 つで、どちらも値を変えないコピーです。
    - `.repeat(1, 1)`
    - `_encode_image` の `torch.cat([1 塊])`
  - T2V では `cond_clean is None` なので、3 回目の取り出し自体が起きない（`:922-924`）。
- **条件行の値**: `_encode_image`（`:644-671`）は、旧と同じ `enc.encode(canvas_to_tensor(image).to(device), tiled=True)` → `patchify_video(z.float())`。形の検査が `del enc` の前に移っただけ。`H3VideoEncoder.encode` は呼び出しの間で状態（因果畳み込みの記憶）を持たないので、1 枚ずつ符号化しても先頭 1 枚の結果は旧と同じ。
- **RoPE**: `keyframe_t(0, …)` は `float(L)` を返し、旧の `cond[:, 0] = L` と同じ値（`engine_h3/rope.py:104-105`）。
  - T2V（`cond_frame_idx=()`）では `keyframe_t` を呼ばない。新しく足した `latent_frames` の 5n+2 検査も T2V の経路には入らない。
  - F−1 の式（`:108-112`）は、正解データの `_temporal_position_span`＋`_build_packed_fl2va`（`pipeline_coords.py:38-42, 89-95`）と同じ組み方・同じ順序。
- **トークン列**: `presentation(tok, prompt, ())` は旧の `image_tokens=None` と同じ列を返す。1 枚なら旧と同じく `<Picture 1>` を付ける（`text_encoder.py:310-331`）。`vision_inputs([image])` は旧の `image_processor(images=[image])` と同じ呼び出し。`mm_token_type_ids` は変わっていない。
  - `read_report` に `vision_tokens_per_image` が増えた。これは記録用の鍵が 1 つ増えるだけで、計算には関わらない。
- **dtype**: 変わった箇所は無い。`engine_h3` に `torch.inference_mode` は無い（grep で 0 件）。`lowvram.py` は差分なし。

---

## M

### M1 ベースモデルを H3 から LTX へ切り替えると、格子から外れたピンがそのまま残り、サーバーが黙って位置をずらす

- **コードの動き**:
  - 操作パネルの格子は `SingleScreen.tsx:960-961`（と `:754-755`）で `constraints.keyframeGrid` から毎回読む。
  - 一方、既にあるカードの `frameIdx` を新しい格子へ寄せ直す処理はどこにも無い。
  - 生成を止める安全装置 `findOverflowFrameIdxs`（`keyframeGrid.ts:202-210`）が見るのは「最後に置ける位置より後ろか」だけ。
- **何が困るか**:
  - H3 で 0・61・123 に置いてから LTX 2.3 に切り替えると、次のようになる。
    - 123 は範囲外として「生成」が止まる。これは正しい動き。
    - 61 はタイムラインでは 61 と表示されたまま送られ、サーバーが 57 に寄せる。
  - 61 と 62 のように、同じ格子点に寄る 2 枚があると 422（重複）になる。
  - 計画 §5 の「目視の前にオーナーへ伝える画面の挙動」3 点には、この件が入っていない。1-G7 の「LTX は 8n+1 の格子が今のまま」を確かめる目視で、オーナーが当たりうる。
- **最小の対処**（どちらか）:
  - (a) 設計を変えずに済ませるなら、§5 の事前共有に 4 点目として「H3 で置いた格子外のピンは、LTX へ切り替えると送信時にサーバーが 8n+1 へ寄せる（表示は寄せない）」を書き足す。
  - (b) 直すなら、`SingleScreen` に `keyframeGrid` の変化を見る effect（画面の値が変わったときに走る処理）を 1 つ置き、各カードを `resolveLanding` で寄せ直す。こちらは監督判断。

---

## m

### m1 MCP の `list_models` の説明に `constraints.keyframes` が無い

- **箇所**: `mcp_server/tools/system.py:100-107`。`constraints` の項目を `num_frames`・`dimension_multiple`・`frame_rate`・`sampling` と 1 つずつ挙げているが、今回増えた `keyframes`（`{step, offset}`。LTX は `{8, 1}`、H3 は `{1, 0}`）が無い（1c の報告の未決事項。確認済み）。
- **何が困るか**: MCP（AI から操作するための窓口）の利用者は、H3 で位置を寄せずに使えることを一覧から知る手段が無い。
- **最小の対処**: `sampling` の説明の後ろに「``keyframes``（キーフレームを置ける位置の格子 ``{step, offset}``。LTX は ``{8, 1}``＝0 と 8n+1、MiniMax H3 は ``{1, 0}``＝任意のフレーム）」を 1 文足す。

### m2 H3 アダプタの REJECT_TABLE の説明が古い前提のまま

- **箇所**: `services/engines/h3/adapter.py:155-157`。「Every predicate tests "differs from the default" (the frontend sends the whole schema)」とある。
- **何が困るか**: `_keyframe_strength`（`:150-152`）は今は「指定されているか」（`is not None`）を見る。操作パネルも強度の鍵を外して送るようになったので、「全項目を送る」という前提の説明は H3 の強度については当たらない（既定が None なので、結果は同じ）。
- **最小の対処**: 「(the default of `strength` is `None`: any value is refused)」の趣旨を 1 句足す。

### m3 H3 アダプタの冒頭の説明にキーフレームの格子が無い

- **箇所**: `services/engines/h3/adapter.py:15-17`。数値の規則として「17n+5 frames from 22, a 32 multiple, 24 fps fixed, 1..50 steps, the two shifts, and flag_defaults」を挙げている。
- **何が困るか**: 今回足した `keyframe_step=1`・`keyframe_offset=0`（`:137-141`）が一覧に無く、一覧の役目を果たしていない。
- **最小の対処**: 「keyframes on any frame (step 1, offset 0)」を足す。

### m4 フロントの `snapFrameIdx` とサーバーの寄せ方が、offset 0・step 2 以上の格子では食い違う（今の系統では起きない）

- **箇所**: `keyframeUtils.ts:53-56` は `floor((f−offset)/multiple)*multiple+offset` だけで、下限を設けていない。サーバーの `normalize_keyframes`（`request_rules.py:209-216`）は `max(first, …)`（first は offset 0 なら step）で下限を設ける。
- **何が困るか**: 格子が (s≥2, 0) の系統ができると、1〜s−1 は画面の予告では 0 になり、サーバーでは s になる。今の系統（8/1 と 1/0）では起きない（1b の留意 5）。
- **最小の対処**: 今は何もしない。将来その系統を足すときのために、`snapFrameIdx` の docstring に「offset 0 で step≥2 の格子は未対応（サーバーは step に寄せる）」と 1 行書く。

### m5 セルフテストの `--keyframe` の誤りがトレースバックで終わる

- **箇所**: `engine_h3/worker.py:548`。`_selftest_images(args)` の `ValueError` を捕まえていない。
- **何が困るか**: 開発者用の CLI（コマンドラインの道具）なので実害は無い。ただ、文言が Python のトレースバックの中に埋もれる。
- **最小の対処**: そのままでよい。直すなら `parser.error(str(exc))` に通す。

### m6 `H3LoadOptions` の説明が、中間だけのジョブを考えていない

- **箇所**: `engine_h3/pipeline_h3.py:125-128`。「for I2V, the vision tower's patch embedding (Conv3d)」とある。
- **何が困るか**: 中間だけのジョブは I2V でも画像読み取り部（vision tower）を読まない。説明の厳密さだけの問題。
- **最小の対処**: 「for I2V」を「when a first / last key frame goes to the vision tower」にする。

---

## 観点ごとの確認結果（要点）

### 2. エンジンの一般化

- 昇順の安定ソート（`pipeline_h3.py:525`）、画像読み取り部へ渡すのは 0 と F−1 だけ（`:539-540`）、ラベルは渡す順（`text_encoder.py:322-324`）。
  - 末尾だけのジョブは `<Picture 1>`。
  - 中間だけのジョブは `images=[]` で、T2V と同じトークン列になり、画像読み取り部を読まない（`text_encoder.py:499-507, 520`）。
- `H3VideoEncoder` は 1 度だけ組む。条件行は昇順に連結する。
- `build_coords` の重複検査と範囲検査は `keyframe_t` の中にある。
- `_denoise` の形の検査（`:870-877`）、`entry["sequence"]["cond"] = cond_rows_n`（K×grid）。
- `H3SeqLayout.make(cond_rows=cond_rows_n)`。
- 進捗の分母: `has_cond` が真なら 4、偽なら 3（`:620-623`）。
- `images_used = len(req.images)` は不変（`:595`）。
- `validate_request` は次の 3 つを拒否する（`:192-202`）。
  - 範囲（0〜F−1）の外
  - 位置の重複
  - 空のパス
- CLI の `--image` と `--keyframe 0=…` の重複は拒否される（`worker.py:515-533`）。
- 1 枚を前提にした残りかすを探した:
  - `source_size` はリストになり、計画どおり。ただし保管庫の `run_2b_g2.py` の `i2v_crop_path` が偽になることは g2_prep の手順書に書いてある。
  - `cond_frames`・単数のログ文言・docstring の残りは無い（grep で確認）。

### 3. API と services

- `normalize_keyframes`（`request_rules.py:184-235`）は、(8,1) で旧の `max(1, min((raw−1)//8*8+1, F−8))` と同じ値。
  - F=9 では first＝last＝1。
  - 0 は 0 のまま。
  - 文言 2 種と `where` の接頭辞も旧と同じ。
- `validate_for_family` の順は `fill_defaults` → `violation` → `normalize_keyframes`（`services/engines/__init__.py:241-246`）。3 系統とも `REQUEST_RULES` を宣言しているので、素通りする経路は無い。
- 連結の validator は、枚数の上限（旧と同じ文字列）の後に `LTX_REQUEST_RULES.normalize_keyframes(..., where=f"clips[{i}]: ")` を呼び、文言が返れば `ValueError` を投げる（`api/models.py:1484-1495`）。
- strength に None が入っても、LTX のアダプタには届かない。経路ごとに確かめた。
  - 単発は `api/generate.py:91` で寄せる。
  - Alpha は画像なし（`generate_alpha.py:247-278`）。
  - 連結は pydantic の中で寄せ、`to_clip_request`（`api/models.py:1747`）が同じオブジェクトを使う。
  - バッチ・i2v-long・Retake は連結を送るか、H3 では閉じている。
  - サーバーで `GenerateRequest` をほかに組む場所は無い（grep で確認）。
- H3 側:
  - H3 の判定は `is not None` で、文言は計画どおり。
  - `REJECT_TABLE` の `conditioning_images` は 1 行だけになった。
  - `UNSUPPORTED_FEATURES` から `multi_keyframes` が消えた。
  - `_ALLOWED_OVERLAP` はそのまま成り立つ。
- config の 2 キーは、`config.py`・`config.yaml.example`・`config.yaml`（git 管理外）のどこからも消えている。git 管理下で読む箇所は無い（`outputs/` の古い較正台の yaml は extra='ignore' で無害）。
- `constraints()["keyframes"]` を公開している。`GET /models` の完全一致テストも更新済み。

### 4. フロントエンド

- 削除: `keyframesMulti`・`multiUnavailable`・`conditioning_frame_idx_multiple`・`conditioning_keyframe_grid_offset`・`strengthDisabled` は git の管理下に残っていない（`mockBridge.h3.test.ts:46` の「含まないこと」の検査だけが残る）。
- `validPositions` の修正（`keyframeGrid.ts:42-45`）。
- `snapFrameIdx` の新しい式は (8,1) で 1〜481 の全部が旧式と同じで、テストで確かめている。
- `omitStrength` は鍵そのものを外す。
- `SingleScreen` は 2 か所とも `keyframeGrid` を読み、`sectionsUnavailable.keyframeStrength` のときは `/generate` の本文だけで strength を外す（A2V＝連結は H3 では閉じている）。
- `KeyframeCard` は行そのものを描かない。
- `useDurationShrinkGuard.ts`・`KeyframeTimeline.tsx`・`GenerationForm.tsx` の変更は、`config.limits.*` を指していた props の説明コメントの書き換えだけで、挙動は変わらない。
- mock の制約は系統ごとに `keyframes` を持つ。

### 5. Gradio

- 10 本のスライダーを `kf_strength_<n>` として `_gated_components` に登録した（`ui.py:2543-2545`）。
- `FEATURE_UI`・`RESET_VALUES` の形は揃っている。
- 送る側は 0.8 と等しければ鍵を送らない（`handlers.py:905-911`）。`gr.Slider(value=0.8)` の初期値と、隠したときの戻し値は Python の `0.8` そのものなので、等しいかどうかの比較は成り立つ。
  - 利用者が LTX で動かした結果が 0.8000000000000002 のような値になれば鍵を送るが、LTX は受けるので無害。
  - H3 では隠すと 0.8 に戻すので、送らない。
- 連結のクリップ 1 は、今までどおり常に strength を送る（LTX 専用）。
- 文言は英語・日本語とも直した。
- `feature_scope.py` が `presets`（gradio と config を読み込む）を import するようになった。
  - これを読み込むのは `ui.py`（既に gradio を読む）とテストだけなので、起動時の負担は増えない。
  - 軽くする手としては、`config.MAX_CONDITIONING_IMAGES` と 0.8 を `feature_scope.py` に置く方法がある。ただ、0.8 の写しが 2 か所になり、計画 m11（0.8 の写しの一本化）に反する。今の形でよいと判断した。

### 6. MCP

- `strength: float | None = None`。
- `model_dump` は `null` を送り、サーバーの `float | None` がそれを受ける。テストは省略したもの（null）と明示したもの（0.9）の両方を確かめている。
- `server.py`・`tools/generate.py` の H3 の説明は現行化済み。`system.py` は m1。

### 7. テスト

- 総当たりは F（28 通り）× L（0〜8192）の二重の繰り返しで、組ごとに `keyframe_t == oracle` を確かめ、最後に `total == 28*8193` も確かめる（`test_h3_engine.py:229-253`）。実装が正解データと全桁一致することを全組で確かめている。
- K=3 のノイズのテスト（`:1127-1156`）は、次の 3 つを `torch.equal` で確かめている。
  - 各区間が「1 区間だけで引いた結果」と同じこと
  - clean が 0 なら 3 区間のノイズが同じになること
  - K=1 が旧式と全桁一致すること
- `_keyframe_job` 系のテストは、本物の `_encode`／`_encode_image` を通している。灰色の濃さで画像を見分け、次の 4 つを確かめている。
  - 昇順に並ぶこと
  - 画像読み取り部には端の 2 枚だけが渡ること
  - 進捗の分母
  - `source_size`
- `test_denoise_runs_with_several_key_frame_blocks` は、本物の `_denoise` を K=2 で回している。
- 「偶然に通っている」旧の期待値は見当たらなかった。H3 の拒否文言の検査は `startswith("MiniMax H3 は")` から `startswith("MiniMax H3 ")` に緩んだ。新しい文言の全文は別のテストで固定しているので問題ない。

### 8. 古いコメント

grep で「先頭フレーム」「first frame」「multi_keyframes」「_normalize_conditioning_images」「cond_frames=」を探した。残っているのは正しい文脈のものだけだった（例: 先頭→末尾の順、`--image` は先頭の別名）。指摘は m2・m3・m6 だけ。

---

## 9. 計画との差

### 計画にあって未実装のもの

- 文書（計画 §3.6）。担当は 1d で、門の結果の後に書く段取り。計画どおり。
- MCP の `system.py`（m1）。計画の対象一覧に入っていなかった。

### 計画に無い変更（どれも無害と判断）

- `engine_h3/audio_vae.py`・`video_vae.py` の docstring の単数形を直した。コメントだけ。
- ノイズの混ぜ方を小さな関数 `noisy_condition` に切り出した。式と乱数の取り出しは旧と同じで、K=1 の全桁一致をテストで固定できるようになった。
- `_encode` に画像をキーワード（`images=`）で渡す形にした。既存の差し替え（`**kw`）を壊さないため。
- CLI の `--keyframe` の誤りを、読み込みの前に検査するようにした（m5）。
- Gradio の連結クリップ 1 のスライダーの既定を、0.8 の写しから定数 `KF_STRENGTH_DEFAULT` に変えた。値は同じ。
- `keyframeUtils.test.ts` の (4,2) 格子の期待値を 13→14 から 13→10 に変えた。計画が指定したサーバーと同じ式の結果で、実際の系統（8/1・1/0）では値は変わらない。
- Gradio の隠す処理のテストを `test_gradio_models.py` に置いた（計画では `test_gradio_ui.py`）。既存の補助関数があるファイルに揃えた。
- `_denoise` の形の誤りの文言にキーフレームの位置を足した。

---

## 判定

C は無いので、1-G5 は**合格**として次の門（1-G6 の配置・1-G2〜1-G4 の GPU の照合）へ進めてよいと判断します。

- 全桁一致の約束（T2V と先頭だけの I2V）は、乱数・値・座標・トークン列のどれも、読んだ範囲では保たれています。最終の確認は 1-G3 の指紋です。
- M1 は、1-G7 の目視の前に、オーナーへ伝える画面の挙動として書き足せば足ります（対処 a）。直すかどうか（対処 b）は監督の判断です。
- m1〜m3・m6 は、1d の文書の現行化と一緒に直すと手間が少なくて済みます。

## 正しく実装されていると確かめた計画の項目

- §2.1 RequestRules の 3 項目・`constraints()["keyframes"]`・`normalize_keyframes` の式と文言・`validate_for_family` の順序・連結での LTX の規則の直接呼び出し・枚数の上限を 2 か所に残すこと・`_normalize_conditioning_images` の削除・config の 2 キーの削除
- §2.2 strength の省略（LTX は 0.8 を書き戻し、H3 は null）・H3 の判定と文言・`_STRENGTH_DEFAULT`／`_multi_keyframes`／REJECT の行の削除・`images_used` を改名しないこと
- §2.3 昇順・画像読み取り部の規則・RoPE の 3 分岐（F−1 は正解データと同じ組み方）・ノイズの `repeat(K,1)`・値と時刻と印・進捗の分母・`dit.py`／`lowvram.py` のコード不変
- §2.4 T2V と先頭だけの I2V の全桁一致（コードを読んで確認。指紋は 1-G3）・LTX の寄せる値と 0.8
- §3.1 rope／pipeline_h3／text_encoder／image_cond／worker（`--keyframe` と重複の拒否）／dit の docstring
- §3.2 api/models・request_rules・engines/__init__・h3/adapter・config 3 ファイル・api/generate のコメント・古いコメント 4 か所の現行化
- §3.3 フロントの削除 7 項目・置き換え 9 項目（types・generationConstraints・SingleScreen の 2 か所と送信本文・テスト 4 ファイル・validPositions・snapFrameIdx・omitStrength・KeyframeCard・mock の制約）
- §3.4 Gradio（定数 1 か所・10 本の登録・既定と同じなら送らない・文言の英語と日本語・`constraints.py` は不変・バッチは不変）
- §3.5 MCP（params・generate・server の説明）
- §4.1〜§4.4 のテスト（総当たり・座標・presentation・validate_request・ノイズ・進捗・worker・API／アダプタ／検証・Gradio／MCP・vitest）
