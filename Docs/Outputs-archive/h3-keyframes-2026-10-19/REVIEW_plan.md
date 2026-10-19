# レビュー: 台帳 §1-97 実装計画（scalable-floating-thimble.md）

作成 2026-10-19。読み取りだけで行った（GPU・テストの実行なし）。
注: 依頼された保存先は scratchpad の `kf197\REVIEW_plan.md` だったが、このセッションはプランモードで、書き込めるのはこのファイルだけだった。内容は同じものを返答にも載せる。

用語（初出のみ）: RoPE＝回転位置埋め込み（行ごとの時間・縦・横の座標）／VAE＝画像と潜在の変換器／TE＝テキストエンコーダ（画像読み取り部 vision tower を含む）／F＝画素フレーム数／T'＝潜在フレーム数／L＝文章の長さ（視覚トークンを含む）／CLI＝コマンド行の入口。

重さの記号: C＝計画のとおりに作ると誤動作かビルドの失敗になる／M＝実装中か門で表に出る抜け／m＝明確さ・軽微。

---

## C（計画のとおりだとビルドが壊れる）

### C1. `GenerationForm.tsx:280` が削除の一覧から漏れている
- 計画の記述（§3.3 削除）: `SingleSectionsDisabled.keyframesMulti`（featureScope.ts:367-368, 380）と、`KeyframesPanel.tsx:22-31` の `multiUnavailable` を消す。
- コードの実際: `webui/src/modes/single/GenerationForm.tsx:280` に `multiUnavailable={sectionsUnavailable?.keyframesMulti}` があり、この 2 つを両方とも参照している。2 つを消すと型検査（`npm run typecheck`）で落ちる。`featureScope.ts:108-111` の説明文も `multi_keyframes` を前提にしている。
- 最小の直し: §3.3 の削除の一覧に `GenerationForm.tsx:280` の 1 行と `featureScope.ts:108-111` の説明文を足す。

### C2. 「fixture 6 ファイルの同じ 2 行」のうち 4 ファイルは limits のキーの行ではなく、部品に値を渡す行
- 計画の記述（§3.3 削除）: `GenerationForm.accordion.test.tsx:42-43`・`GenerationForm.referenceVideo.test.tsx:34-35`・`GenerationForm.sourceAudio.test.tsx:28-29`・`KeyframesPanel.test.tsx:20-21` の「同じ 2 行」を消す。
- コードの実際: これらは `frameIdxMultiple={FALLBACK_APP_CONFIG.limits.conditioning_frame_idx_multiple}` と `gridOffset={…}`（KeyframesPanel.test.tsx は `const MULTIPLE = …`／`const OFFSET = …`）で、必須の部品の値（`KeyframesPanelProps.frameIdxMultiple: number`、KeyframesPanel.tsx:15-17）を渡している。消すと必須の値が欠けて型検査で落ちる。設定の写しとして消すのが正しいのは `EditScreen.inpaint.test.tsx:160-161`・`EditScreen.retake.test.tsx:130-131` の 2 ファイルだけ。
- 最小の直し: 4 ファイルは「消す」から「置き換える」へ移す（`LTX_GENERATION_CONSTRAINTS.keyframeGrid.step／offset`、または 8／1）。

---

## M（実装中か門で表に出る抜け）

### M1. 中間だけのジョブで、encode の段の進捗の分母が 3 のまま 4/4 が出る
- 計画の記述（§3.1）: `_encode(prompt, phases, images=())`（進捗の段数は「1 枚以上なら 4」のまま）。§2.3 では、`_encode` に渡すのは画像読み取り部へ入る画像（frame 0 と F−1）だけ。
- コードの実際: `engine_h3/pipeline_h3.py:570` は `total = 3 if image is None else 4` で、`_encode` が受けた画像で分母を決める。一方 `_encode_image`（:609）は無条件に `_progress("encode", 4, 4)` を出す。中間だけのジョブでは、画像読み取り部への画像は 0 枚なのに VAE の符号化は走るので、分母 3 の後に 4/4 が届く。
- 最小の直し: `_encode` に「条件行の画像が 1 枚以上あるか」を別に渡す（例: `total = 4 if has_cond else 3`）。worker.py:64-65 の docstring（「T2V は 3 段、I2V は 4 段」）もこの規則に合わせて書く。

### M2. `constraints()` の完全一致を固定しているテストが、§4.2 の一覧に入っていない
- 計画の記述（§4.2）: `constraints()["keyframes"] == {"step":1,"offset":0}` の確認を足す。
- コードの実際: `tests/test_engine_h3_adapter.py:221-242`（`H3_CONSTRAINTS`／`LTX_CONSTRAINTS` の辞書と `test_constraints_blocks_are_the_plans_json`）と `:585-586`（`GET /models` の `constraints == LTX_CONSTRAINTS`）は、辞書をまるごと比べている。`keyframes` を足すと全部落ちる。
- 最小の直し: §4.2 に「この 2 つの辞書に `keyframes` を足す」を書く。

### M3. `images_used` → `keyframes_used` の改名で、次のテーマの計測台が壊れる。同じ情報はもう記録にある
- 計画の記述（§2.3 記録・§7.7）: done の `images_used` を `keyframes_used: [frame_idx…]` に置き換え、`_DONE_FACTS` も同じ名前にする。
- コードの実際: この名前を読むところが計画の外にある。
  - `Docs/Outputs-archive/h3-vram16-2026-10-18/measure_h3_vram.py:730-733`（`done.images_used` が想定と違うと不一致として扱う。§1-95 の快適上限の較正で使う計測台）と、同じ処理の `h3-vram-2026-10-18/measure_h3_vram.py:693-696`
  - `h3-gate2/run_2b_g2.py:220, 433, 464`
  - `h3-gate2/g4_i2v_resident.py:119`（1-G3 の土台になる台本）
  - 設計正本 `MINIMAX_H3_DESIGN.md:325`（done のキーの一覧）
  
  位置の一覧は、metadata.json の `request.conditioning_images[].frame_idx` に、サーバーが寄せた後の値で既に残る（エンジンは位置を動かさない）。
- 最小の直し（引き算）: `images_used`（枚数）はそのまま残し、§2.3 の「記録」の項を消す。それでもエンジン側に位置を残したいなら、置き換えずに足す。

### M4. 1-G3 の指紋の条件が足りない。基準を取っていない条件が 2 つある
- 計画の記述（§5 1-G3）: セルフテストで先頭 I2V `77e29687…`（§166 J1 の条件）。`--vram16` を on にしても同じ。API 経由（deterministic true）でも 1 本ずつ。
- コードと記録の実際:
  - J1 はセルフテストの CLI ではない。`Docs/Outputs-archive/h3-gate2/g4_i2v_resident.py` を同じプロセスで動かしたもので、条件は G5 のプロンプト（黄色いレインコート）・画像 `outputs/h3-gate2/i2v_input.png`・そのジョブだけ先読み off・常駐のキー無し・sdpa・vram16 off（同ファイル :8-14, :73, :100-105）。
  - 画像 `i2v_input.png` は git の管理外にしかない（`outputs/h3-gate1/g5_t2v_after_review.mp4` の先頭フレーム。Outputs-archive/h3-gate2/README.md）。
  - セルフテストで同じ条件にするには、`--prompt <G5>`・`--image outputs/h3-gate2/i2v_input.png`・`--no-prefetch` が要る（worker.py:448-462 の既定は先読み on・既定のプロンプト）。
  - I2V の指紋を取った記録は J1 と J3（どちらも先読み off・vram16 off）だけ。vram16 on の I2V（§169 で全体の指紋を比べたのは T2V だけ）と、API 経由の I2V（H3 の既定は vram16 on・先読み on・アップロードの保管を通る。2b-G2 の I2V は正方形の別の画像を使っていて、指紋の記録が無い）は、一度も指紋を取っていない。一致しなかったとき、回帰なのか条件の違いなのかを見分けられない。
- 最小の直し: 1-G3 の表に J1 の条件をそのまま書く（画像のパス・プロンプト・`--no-prefetch`）。vram16 on と API 経由の 2 つは、変更の前の main で先に基準を取る。それができないなら「`77e29687` と一致」を合否から外して「記録」にする。M3 で改名する場合は、g4 台本の `done["images_used"]` も直す。

### M5. 1-G2 の土台にする台本は「延長」では済まず、正解側のファイルと 2 枚目の試験画像が決まっていない
- 計画の記述（§5 1-G2）: `g3_i2v_parts.py` の延長の台本で、[先頭＋末尾] と [末尾だけ] を照合する。
- コードの実際:
  - 正解側（DiffSynth の写し）の `minimax_h3_text_encoder.py`／`minimax_h3_video_vae.py` は `outputs/h3-gate2/oracle/diffsynth_h3/models/` だけにある。Outputs-archive の同じ場所には `__init__.py` しかない（README に「複写していない」とある）。
  - 台本は 1 枚の形を直接呼んでいる。`te.vision_inputs(canvas)`（:185）・`te.encode(PROMPT, image=canvas)`（:186）・`build_coords(..., cond_frames=1)`（:309）・時刻の照合の `img_pos[:grid_rows]`（:325）。どれも計画で引数が変わるので、書き直しになる。
  - 正解側が 2 枚に対応していることは確かめた。`image_token_counts(processor, images)`（:109-113）、`presentation_fl2va(..., counts のリスト)`（:163-169）、`_build_packed_fl2va(..., keyframe_indices=[0,-1] または [-1])`（pipeline_coords.py:90-95）。
  - 末尾に使う 2 枚目の画像が計画に書かれていない。
- 最小の直し: 1-G2 の項に、正解側の置き場所（`outputs/h3-gate2/oracle/`）、試験画像 2 枚（例: `g5_t2v_after_review.mp4` の先頭と末尾のフレーム）、書き直す呼び出しの一覧を書く。

---

## m（明確さ・軽微）

- **m1. LTX の監査テストは変わらない。** §4.2 の「監査（`test_ltx_adapter_audit.py`・`test_ltx25_api_guard.py`）: 重なりの行数の期待値」について。この 2 ファイルに、重なりや `multi_keyframes` の期待値は無い（grep で確認。ヒットは test_ltx_adapter_audit.py:146 の vram16 だけ）。LTX の表は変わらないので、この項は消してよい。
- **m2. Gradio のバッチが H3 に届くかは、今の時点で決まる。** `gradio_ui/batch.py:704-709` は `POST /generate/chain` を送り、H3 は `reject_chain` で連結の入口を丸ごと断る（h3/adapter.py の `reject_chain`）。届かないので、§3.4 の最後の項と §7.9 の「確かめる」は「届かない（理由はこれ）」と書き換えられる。
- **m3. Gradio の `LTX_CONSTRAINTS` に `keyframes` を足す項（§3.4）は要らない（過剰設計）。** Gradio は位置を寄せない。`normalize_constraints`（constraints.py:67-）も `keyframes` を読まない。足すだけだと、H3 の scope にも LTX の格子が写る（`test_gradio_constraints.py:102, 263` は配信された辞書そのものと比べている）。項ごと消す。
- **m4. 計画の外に、古くなる文言やコメントが残る（現行化の対象）。**
  - コメント: `services/engines/ltx/adapter.py:2208`、`engine/pipeline/fast_video_pipeline.py:1587`、`engine/pipeline/chain_pipeline.py:315`、`tests/test_chain.py:343`（どれも `_normalize_conditioning_images` に触れている）
  - 画面の文言: `gradio_ui/i18n.py:174` の `cap_kf_grid`「8n+1 に寄せる」（H3 の画面にも出る）と `gradio_ui/ui.py:633-637`
  - docstring: `h3/adapter.py` の generate（「先頭フレームの I2V」）、`mcp_server/params.py:22-27`
  - 文書: README の「できないこと」の表の 2 行（README.md:976-977）を消すか書き換える。設計正本 §2 の裁定 3 の最後の文（「α版の I2V は先頭フレームだけ」）に更新の行を付ける。
- **m5. 422 の順序の変化は 1 つでなく 2 つある。** 重複のエラーは FEATURE_UNSUPPORTED の後になるだけでなく、系統の数値の違反（8n+1・17n+5）の後にもなる（validate_for_family の中で violation が先）。仕様書にはこの両方を書く。
- **m6. H3 の操作パネルの「追加」は中央に置く。** `nextAddPosition`（keyframeGrid.ts:150-179）は、最後のピンと F−1 の中点に置く。H3 の 2 枚目は中間になり、末尾にするにはドラッグが要る。尺を延ばすと、元の F−1 にあったピンは中間のフレームになり、黙って画像読み取り部から外れる。仕様どおりだが、1-G7 の目視の前にオーナーへ伝えておくとよい。
- **m7. 1 フレーム刻みでは、左端の近くがドラッグで選びにくい。** `SNAP_ZERO_MIN_PX = 14`（KeyframeTimeline.tsx:52）なので、幅 600px・481 フレームでは 1〜11 番あたりのフレームがドラッグでは 0 に吸われる（数値の入力と矢印キーでは置ける）。:124 は `offset` を「位置 1」とみなしているので、offset が 0 だと最小の幅に落ちる。目盛りは描いていないので、480 本の目盛りの問題は起きない。
- **m8. 行番号のずれ。** `validate_request` のテストは test_h3_engine.py の :781-790 と :974-983 にある（:151・:196 ではない）。H3SeqLayout の docstring は dit.py:113-119。
- **m9. `omitStrength` は鍵を消す形で書く。** `tsconfig.app.json:22` で `exactOptionalPropertyTypes: true` になっている。`{...ci, strength: undefined}` は型エラーになるので、分割代入で鍵そのものを外す。
- **m10. Gradio の 0.8 の写しを 1 つにまとめる。** `presets.py` に `KF_STRENGTH_DEFAULT` を足すなら、`ui.py:654`（`value=0.8`）・`handlers.py:1477`・`RESET_VALUES` もそれを使う。ほかに、`tests/test_gradio_handlers.py:359-363` の LTX の期待値（0.8 の鍵がある）は「鍵が無い」に変わる。§4.3 にこの書き換えを入れる。
- **m11. 式の書き方を正解側に揃えて明記する。**
  - 中間の位置は `float(L) + (5.0/3.0)*n`（ComfyUI と同じく掛け算が先）。
  - 末尾の span は、正解側と同じ組み方（`np.ones(T')*5/3` の後に刻み 5 で ×(1,4,4,4,4)、それから `np.sum`）でないと全桁一致しない。§2.3 に一言書く。
  - 総当たりのテストが Outputs-archive の正解側を import するのは、新しい依存になる。3 行の式を写して持つ方法もある（任意）。
- **m12. CLI で `--image` と `--keyframe 0=…` を両方渡したときの扱いを決める**（重複として拒否するか、`--image` を優先するか）。
- **m13. バッチの枠が閉じることは、関数のテストで既に固定されている**（featureScope.test.ts:197-207）。H3 の fixture で足すテストは任意。

---

## 総評

設計の骨格は正しい。エンジンの一般化は、K=1（先頭 1 枚）のときにトークン列・乱数の取り出し・条件行・RoPE 座標が今と同じになる。末尾の式と F の逆算は、正解側と一致している。LTX の寄せる式・境界・文言は、今の関数と同じ値を返す。強度を省略可にしても、LTX のアダプタに None が届く経路は無い。計画の要点は、どれもコードで裏が取れた。

直すべき点は 3 種類ある。
- フロントの削除の一覧の漏れと取り違え（C1・C2）。型検査ですぐ見つかるが、そのまま作ると落ちる。
- 中間だけのジョブの進捗の分母（M1）と、完全一致を固定しているテストの見落とし（M2）。
- 記録の改名（M3）と門の条件（M4・M5）。どれも門で初めて表に出る。特に `images_used` の改名は、次のテーマの計測台を壊すうえに、同じ情報が metadata に既にある。引き算（改名しない）を勧める。

1-G3 の I2V の指紋は、J1 の条件（画像・プロンプト・先読み off）を明記し、vram16 on と API 経由は変更前に基準を取るか、合否から外すべきである。以上を直せば実装に進めると判断する。

---

## 確かめて正しかった主張

1. 末尾の時刻の式と足す順序: 正解側は `(float(text_len) + temporal_span) − 5/3`（pipeline_coords.py:95）。span は float64 の numpy 配列の `.sum()`（:38-42）。計画 §0・§2.3 と一致する。数学的には span(T') − 5/3 = 5/3×(F−1)（17n+5 では 1 周期 5 潜在の幅の和が 17）。
2. F の逆算 `F = (T'−2)//5×17 + 5` は、`latent_frames` の T' = (F−5)//17×5+2（constants.py:67-79）の逆として正しい。F の範囲 22〜481 は 28 通り。
3. `normalize_keyframes` の式は、LTX（8,1）と F＝8n+1 のとき `max(1, min((raw−1)//8*8+1, F−8))`（api/models.py:224）と同じ値になる。F＝9 のときも first＝last＝1 になる。文言 2 種（:229-238）も引き継げる。H3（1,0）では 1..F−1 がそのまま通り、F 以上は F−1 に寄る。
4. 連結の検査（api/models.py:1509 の 8n+1 → :1527）は、寄せる前に 8n+1 を確かめている。`request_rules.py` はモジュールの読み込み時に api を import しない（:29-39）ので、循環は無い。
5. LTX の strength の書き戻しは、すべての経路を覆う。
   - 単発は api/generate.py:90。
   - Alpha は generate_alpha.py:278（画像なし）。
   - 連結は pydantic の検査の中で寄せる。`to_clip_request`（models.py:1756-1781）と job_store.py:171 は、寄せた後のオブジェクトを使い回す。
   - Gradio のバッチと操作パネルのバッチ（i2v 行を含む）、Batch I2V-Long、Retake は、連結の入口を通るか、H3 では画面が閉じる。
   - `ci.strength` を読むのは LTX の実際のバックエンドの 4 か所（ltx/adapter.py:2214, 2533・ltx25/adapter.py:822, 1228）だけで、模擬のバックエンドは読まない。
6. `reject_unsupported`（generate.py:79）は `validate_for_family`（:90）より先に走る。アダプタの generate の中でもう一度呼ぶ検査（h3/adapter.py の generate の冒頭）も、H3 では None を見る。
7. 重なりの監査は `_ALLOWED_OVERLAP = {"conditioning_images"}` のまま成り立つ（`keyframe_strength` の行が残るため）。
8. `LimitsConfig` は pydantic 既定の extra='ignore' で、config.py に `model_config` は無い。2 つのキーを読むのはバックエンドでは config.py だけ（Gradio・MCP・テストも読まない）。
9. フロントの格子の直し 2 点は、（1,0）で必要十分である。
   - `validPositions` は今のままだと 0 が二重に入る。
   - `maxPlaceablePosition`＝F−1 になり、サーバーの `last` と同じ式。
   - `findOverflowFrameIdxs`・`snapToPlaceableGrid`（KeyframeCard.tsx:66-72）・`stepPin`・`resolveLanding` は問題ない。
   - 時間の帯は目盛りを描かず、F−1 では描けない区間も現れない。
10. `constraints` の部品の値（SingleScreen.tsx:114, 196）は、:761 と :967 の位置で使える。
11. H3 ではバッチの枠が丸ごと閉じる（AppShell.tsx:1660 → featureScope の `chain` → `createSection.batchA2v`）。Batch I2V-Long は Chained の画面の中にあり（ChainedScreen.tsx:1220）、Edit のタブも全部閉じる。
12. Gradio の隠す仕組み（ui.py:2535-2604）は、部品を登録するだけで 10 本に効く。`refresh_model_dropdowns` は更新ボタン・ページの読み込み・読み込み後の再取得の 3 か所から呼ばれる。
13. MCP で `"strength": null` を送っても、`float | None = Field(None, ge, le)` は None を受ける（範囲の検査は None に掛からない）。MCP のテストで 0.8 を固定しているものは無い（:198-206・:1349-1362 は 0.9 を明示している）。
14. T2V の指紋 `0da6111c…` は、黄色いレインコートのプロンプトが条件（VERIFICATION_LOG の §165 の訂正行・§167.6）。I2V の `77e29687…` は §166 2a-G4 の J1・J3。
15. エンジンの部品は条件行の枚数に依存しない（K に依らない）。
    - `H3SeqLayout.make`・`embed_rows`・`time_rows`・`modality_rows`・DiT の出力の切り出し・lowvram は、cond を一塊として扱う。
    - ノイズの `repeat(K,1)` は K=1 で今と同じ値になる。
    - `_encode_image` の `(1,24,1)` の検査は 1 枚ずつで成り立つ。
    - L（視覚トークンを含む）が RoPE の原点になる点は、正解側の text_len と同じ定義で、中間の画像を画像読み取り部に入れない規則とも矛盾しない（中間の画像は L を増やさない）。
16. 番号: 次は CLOSED 3-241・VERIFICATION_LOG §170・DEVLOG §143（フロントの DEVLOG の最後は §142）・設計正本の裁定は 54 から（最後は 53）。README の凍結区間の境界は 134 行目の `# Nz-Videomniの詳細`。
