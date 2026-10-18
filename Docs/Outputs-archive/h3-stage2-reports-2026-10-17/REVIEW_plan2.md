# REVIEW_plan2 — MiniMax H3 第 2 弾の計画の事実確認レビュー

対象: `C:\Users\phenotype nezu\.claude\plans\shimmering-twirling-octopus.md`（以下「計画」）。読むだけで実施。GPU・テスト・インストールは使っていない。
根拠の行番号は 2026-10-17 時点の作業ツリー（`S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`）。「事実」はコードで確かめたこと、「推測」はそうでないこと、「未確認」は確かめられなかったこと。

置き場についての注記: 依頼の置き場（scratchpad の `h3\REVIEW_plan2.md`）は、このセッションの計画モードの制約（書けるのはこのファイルだけ）により書けなかった。本ファイルを写して使ってほしい。

---

## 0. 要旨

- Critical 1 件、Major 13 件、Minor 22 件。
- 計画の事実は大筋で正しい（`GenerationOutcome.ltx25` の前例、`precheck_model_file` が `H3FormatError` を捕まえず 500 になる点、`_decode_temporal` の窓構造で「窓ごとに CPU へ」が成り立つ点、Gradio のスライダーが飾りである点、`useKeyframes` の `maxItemsOverride` は実在する点）。
- 誤り・穴が集中しているのは次の 5 か所: (1) `num_inference_steps` の既定を `None` にしたときの波及（Critical）、(2) block swap の入れ直しと常駐の解放の契約（`release_installed()` は包みを外さない）、(3) 機能名の表と UI の閉じ方（Alpha Gen の非表示化が LTX 2.3 だけを変える・NAG を隠すと解除できない・`UNSUPPORTED_FEATURES` を拒否表から導出できない名前が多い）、(4) 快適上限の落ち先の外し方が LTX の「行が合わない」場合も巻き込む、(5) 門の段取り（2a-G2 の基準取りの時点、2a-G4 が証明しない事項）。

---

## 1. Critical

### C-1. `num_inference_steps` の既定を `None` にすると、LTX の mock・ログ・`metadata.json` が壊れる／変わる（計画 §3.3「新しい項目」・§8 の該当行）

- 事実:
  - LTX の mock の描画が `steps = max(1, request.num_inference_steps)`（`services/engines/ltx/adapter.py:1605`）。`None` だと `TypeError`。mock は 2.3・2.5 で共有され（`test_ltx25_adapter.py:3322-3328`）、計画ではさらに H3 も共有する。操作パネルは `num_inference_steps` を送らない（`webui/src/api/types.ts:9`）、MCP も送らない（`mcp_server/tools/generate.py:4-9`）、Alpha Gen の内部要求も指定しない（`api/generate_alpha.py:247-269`）。したがって mock を使う全テストと mock 運転のジョブが落ちる。
  - `PipelineManager` のジョブ開始ログが `"steps=%d"` に `job.request.num_inference_steps` を渡す（`services/pipeline_manager.py:661-670`）。`None` だとロギングの例外表示（`--- Logging error ---`）が毎ジョブ出る。
  - LTX 2.3 の進捗が `progress_callback(0, request.num_inference_steps, 0.05)` など（`ltx/adapter.py:1175・1237・1256`）。
  - `metadata.json` の `request` は `req.model_dump()`（R5a D1 が `pipeline_manager.py:1866` を確認）なので、LTX のジョブでも `num_inference_steps: null` が記録される。mp4 に埋め込むレシピも同じ（§3-164）。これは「LTX 2.3／2.5 の振る舞いは変わらない」（計画 §1）に反する。
  - LTX 2.5 の無視＋ログは「既定値と違うときだけ名指しする」（`ltx25/adapter.py` の `_log_ignored`、仕様書 1594 行）。既定が `None` になると、Gradio が固定で送る `8`（`gradio_ui/handlers.py:604`）が「既定と違う」扱いになり、2.5 のジョブ毎にログ行が出る。`test_ltx25_adapter.py:802-830` の前提も崩れる。
  - `GET /config` の `defaults.num_inference_steps` は `config.py:218` で 8 のまま残り、要求の既定（`None`）と食い違う。
- 計画は「2.3 のアダプタは `None` を 8 として `num_steps` に送る」（§8）しか手当てしていない。このままでは 2b-G1（アプリのテスト全件）に合格できず、LTX の記録内容も変わる。
- 直し方の提案: `None` の解決を 1 か所に寄せる。`validate_for_family` で系統の既定値（LTX は 8、H3 は 20）を要求に書き戻し、以後の読み手（mock・ログ・アダプタ・`metadata.json`）は常に整数を見るようにする。こうすれば LTX の `metadata.json` は従来どおり `8`、2.5 の無視ログの比較も従来どおり。書き戻しを採らない場合は、上の読み手を全部列挙して担当 C3 に渡す。`GET /config` の既定値の扱い（系統共通の 8 のままでよいか）も決めておく。

---

## 2. Major

### M-1. `release_installed()` は block swap の包みを外さない（計画 §3.2「常駐」・§8）

- 事実: `BlockSwapService.release_installed()` は「touches no tensor and no device; it only lets go of the reference」で、`_installed_transformers.clear()` だけ（`engine/transformer/block_swap_service.py:166-173`）。`block.forward` は戻さない。包みを外すのは LTX 2.5 の `_unpatch_block_swap`（`engine25/gguf_transformer.py:1149-1177`。`_BLOCK_SWAP_ATTR` の印から元の `forward` を戻して印を消す）だけ。
- 現状の H3 も毎ジョブ `release_installed()` を呼んでいる（`engine_h3/pipeline_h3.py:623`）が、DiT を `del` するので問題が表に出ていないだけ。
- 常駐で同じ DiT を使い回すと次が起きる（コードからの帰結）:
  - 同期経路どうし: `_patch_block` が包み済みの `forward` を「元」として退避し、二重に包む（`block_swap_service.py:228-229`）。
  - 計画 §3.2 で `block_swap_prefetch` をジョブごとにするので、「先読みのジョブ → 同期のジョブ」の順が起こりうる。同期の `_patch_block` は印を見ずに今の `forward`（前のジョブの先読みの包み。そのエンジンは `teardown` 済み）を元として退避し、毎ブロックで破棄済みのエンジンの `on_block_forward` を呼ぶ。操作パネルは先読み無効なら常駐を送らないので通常の画面操作では起きにくいが、API／MCP からは起こせる。
- 加えて、同期経路はジョブの終わりに最後の窓のブロック（`blocks_on_gpu` 個）を GPU に残す（追い出しは `idx-1` だけ。`block_swap_service.py:266-273`）。今は `del dit` で消えるが、常駐では「ジョブの合間に GPU には何も残さない」（計画 §3.2）が成り立たない。先読み経路の `teardown` が全ブロックを CPU の正本に戻すかは未確認（`block_swap_prefetch.py:691-720` は `_release` を呼ぶが中身は読んでいない）。
- 直し方の提案: 計画の選択肢から `release_installed()` を外し、「`_unpatch_block_swap` と同じ印ベースの剥がしを H3 側に書く（共有部品は変えない）」に一本化する。ジョブの終わりに全ブロックとブロック以外を CPU へ戻すことを明記し、門で `torch.cuda.memory_allocated()` がジョブの合間に 0 であることを測る（M-11）。

### M-2. Alpha Gen の「非表示化」は H3 には効かず、LTX 2.3 の見た目だけを変える（計画 §3.5 の表の `alpha_gen` 行）

- 事実: LTX 2.3 は `alpha_gen` を `UNSUPPORTED_FEATURES` に載せている（`services/engines/ltx/adapter.py:136-140`。`featureScope.ts` の説明も「It is LIVE: LTX 2.3 publishes it」）。計画が「2.3 の現状を確認して」と書いた点は確認済みで、灰色→非表示に変えると LTX 2.3 の Edit タブから Alpha Gen が消える。
- 一方 H3 では Retake・Outpainting・Inpainting・Alpha Gen の 4 つが全部閉じるので、`CONTAINER_TARGETS` により Edit タブごと閉じ、Edit 画面はマウントされない（`featureScope.ts` の `editSubTabsDisabledFor` の説明「the screen this feeds is not mounted at all」）。H3 で Alpha Gen を隠すか灰色にするかは、画面に現れない。
- 直し方の提案: `alpha_gen` 行の変更と `EditSubTabs.tsx` の改修を計画から削る（引き算）。裁定 30 の「AlphaGen は非表示」は Edit タブが閉じることで満たされる、と計画に書く。将来 H3 が Edit の一部を持つ段で改めて決める。

### M-3. NAG のアコーディオンを隠すと、同じセッション内で有効にした NAG を解除できず、H3 の全ジョブが 422 になる（計画 §3.5 の `nag`・`vsf` 行）

- 事実: NAG の `enabled` は起動時は必ず `false`（永続化しない。`shell/useNagSettings.ts:47-61`）が、セッション内では状態として残る。送信は `nagRequestFields` が `enabled` のときだけ `negative_prompt`・`nag_enabled` などを足す（`shell/nagSettings.ts:158-169`）。LTX で NAG を有効にしてから H3 に切り替えると、アコーディオンが隠れたまま `nag_enabled: true` が送られ続ける。`featureScope.ts` の `FeatureUiEntry.resets` の説明がまさにこの事故を警告しているが、`resets` は `AccelerationSettings` のキーしか取れない。
- 直し方の提案: 次のどちらかを計画で決める。(a) 隠すのをやめ灰色にする（解除の操作は残るか要確認）。(b) 隠すなら、H3 では `nag.enabled` を `false` に戻す手当てを `AppShell` 側に足す（`resets` の型を広げるか、別の 1 行）。いずれも「既存の作法に例外を足す」ので、オーナーの好み（UI の規則は単純に）に照らして (a) が素直と考える（推測）。

### M-4. 拒否する機能名の多くは `GenerateRequest` に対応する項目が無く、「拒否表から導出」と監査テストに合わない（計画 §3.4「拒否する機能名」・§3.5 の表）

- 事実:
  - 既存 2 系統の `UNSUPPORTED_FEATURES` は単発の `REJECT_TABLE` の機能名の列そのもの（`ltx/adapter.py:222-224`、`ltx25/adapter.py:462-469`）。監査テストは「表が実在しない項目を名指ししない」ことを固定している（R5a B3 が挙げる `test_ltx25_adapter.py:459-462`・`3137-3143`、`test_ltx_adapter_audit.py:64-80`）。
  - 計画の名前のうち `chain`・`v2v`・`a2v`・`end_source`・`retake` は連結要求の項目（または連結そのもの）で、単発の `GenerateRequest` には無い。`ic_lora` は要求の形からは決まらない（制御か画風かは登録簿の種別。`api/generate.py:114-126`）。`nag` と `vsf` は同じ `nag_enabled` を見る 2 つの名前になる。`multi_keyframes`・`keyframe_strength` は同じ `conditioning_images` を見る。
  - Single の A2V カードは `POST /generate/chain` を使う（R5b A-3: 「A2V が付いていると `buildA2vRequest` 経由で `POST /generate/chain` に変わる」）。
- 直し方の提案（引き算）:
  - `v2v`・`end_source` を H3 の公開名から外す。Chained タブは `chain` で丸ごと閉じるので、パネル単位の名前は何も閉じない。
  - `a2v` も外し、`FEATURE_UI` の `chain` 行の `disables` に Single の A2V カードを足す（Single の A2V は連結要求そのものなので意味が通る。LTX 2 系統は `chain` を公開しないので影響なし）。
  - `ic_lora` を外し、Single の参照動画アコーディオンは `reference_video` だけで閉じる。
  - `nag`・`vsf`・`negative_prompt` は 1 つ（`nag_enabled` を見る行）にまとめ、`negative_prompt`・`neg_method`・`vsf_scale`・`nag_*` は `GOVERNED_FIELDS` に置く（`negative_prompt` は `enabled` のときしか送られない。M-3 の事実）。
  - そのうえで「`UNSUPPORTED_FEATURES` ＝ 単発の拒否表の機能名 ＋ `chain`（連結は丸ごと拒否）」のように導出の規則を 1 行で決め、監査テストの期待も合わせる。
  - LTX 2.5 は `pipeline` の拒否を機能名 `two_stage_hq` で公開している（`ltx25/adapter.py:140-143`）。H3 も同じ名前にそろえる（計画は `pipeline`）。

### M-5. `crop_output` の扱いが計画に無い（計画 §3.4「動作する項目」と 4 分類表）

- 事実: `crop_output` は LTX のアダプタの `generate` の中でアプリ側が切り抜く（`ltx/adapter.py:164-166`・`1240`・`2182-2184`・`2379`）。操作パネルの Single は出力の切り抜き（N1）を持ち、プリセットにも `crop_output` が入る（`webui/src/bridge/mockBridge.ts:64-93`、`batchRunner.ts:110`）。H3 のアダプタは新しく `generate` を書くので、何もしなければ切り抜きは黙って無視される。4 分類の監査は全項目の分類を要求するので、どの表に置くかも決まっていないと 2b-G1 で落ちる。
- 直し方の提案: 「動作」にして LTX と同じ後処理（全サイズを一時ファイルに書かせて切り抜く）を H3 の `generate` でも呼ぶか、「拒否」にして画面の切り抜きを閉じるかを決める。前者なら共通の切り抜き関数を呼ぶだけで済む見込み（推測）。あわせて `embed_mp4_metadata`（アプリ側の汎用処理）・`fused_gguf_dequant_kernel`・`conditioning_attention_strength`・`reference_video_strength`（`loras` に従属）・NAG の従属項目の分類も計画の表に全部書く。

### M-6. 快適上限の落ち先を「系統が分からないときだけ残す」にすると、LTX の「行が合わない」場合も落ち先を失う（計画 §3.5「快適上限マーカー」）

- 事実: `resolveComfortRow` が `null` を返すのは 3 通り（`webui/src/shell/comfortTable.ts:169-220`）: 互換の分岐で合わない、系統の表が無い（H3）、**表はあるがどの行も合わない**。3 つ目は LTX でも起こり、説明に「when unknown (`""`/omitted) it is left out, so every typed row misses and the caller falls back exactly as for an unmatched configuration」とある（重みの種別が判定できない・高速化の組み合わせが表に無いなど）。Single は `??` でスカラの表へ落ちる（`useGenerationForm.ts:702-721`）。
- 計画どおりに「系統が分かっているなら落ちない」とすると、LTX の 3 つ目の場合にマーカーと右クリックの尺の合わせ込みが消える。実際にその場合が起きる頻度（配信表が全組み合わせを覆うか）は未確認。
- 直し方の提案: 落ち先を外す条件を「`limits.comfort_budgets` に**その系統の表が無い**」に限定する（呼び出し側で `table[engineFamily]` の有無を見るか、`resolveComfortRow` が「表が無い」を区別して返す）。Chained・Retake は H3 では閉じるので触らない（E-6）。

### M-7. TE の直接読みは HF の前向き計算の中に層の読み込みを差し込む形にしないと、I2V の DeepStack と多次元 RoPE を壊す（計画 §3.2「TE の直接読み」）

- 事実: 計画が使う `Qwen3VLModel`（`.venv-engine-h3` の transformers）は、視覚塔の中間特徴を言語モデルの先頭の層に足し込む DeepStack を層のループの中で行う（`modeling_qwen3_vl.py:835-839` `_deepstack_process`）。画像トークンの位置は `get_rope_index`（同 927）が画像の格子から 3 軸で作る。DiffSynth も 1 回の `pipe.text_encoder(**kwargs)` に `pixel_values`・`image_grid_thw` を渡している（scratchpad `src/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:296-319`）。
- 「1 ジョブ 1 回の前向き計算」は事実（DiffSynth の I2V も TE の呼び出しは 1 回。CFG も無い）。「層を主メモリに置く意味がない」も成り立つ。ただし層のループを自前で書き直すと、上の 2 点を写し損ねる危険がある。
- 直し方の提案: 実装の形を計画に 1 行で決める。「HF の `forward` はそのまま使い、各層の前向きの直前にファイルから GPU へ読み、直後に捨てるフック（`GemmaLayerOffloadService` の差し替え）にする」。埋め込み表（約 1.5GB）と視覚塔は計算の前に GPU へ読み、計算後に捨てる。2a-G2 のビット一致は T2V だけでなく I2V（画像 1 枚）でも取る（I2V の基準は直接読みでない方式で一度取る必要がある。M-10）。

### M-8. 「符号化器だけ f32」と「T2V で符号化器を組まない」は、今の `H3VideoVAE.from_file` の構造では作れない（計画 §3.1「符号化器の型」・§3.2「§1-99」）

- 事実: `from_file` は符号化器と復号器を含むモデル全体を meta で組み、`load_streaming` で読み、`model.to(device)` してから `apply_precision(dtype)` で全体を 1 つの型にそろえる（島だけ f32。`engine_h3/video_vae.py:422-470`）。`load_streaming` は読み終わりに meta のまま残ったテンソルがあれば例外にする（`engine_h3/sft_load.py:98-105`）。したがって符号化器だけ読まない・別の型にする、には構造の変更が要る。`quant_conv`（符号化器側）は島に入っていない（`float32_islands`、同 444-449）。
- 常駐の観点でも未決: 常駐する VAE に符号化器を含めるか、I2V のジョブごとに組むか。
- 直し方の提案: 「符号化器（`encoder`＋`quant_conv`）と復号器（`post_quant_conv`＋`decoder`）を別々に組んで読めるようにし、符号化器は I2V のジョブごとに f32 で組んで捨てる（常駐しない）。常駐するのは復号器だけ」と決める。§1-99 は裁定 25 で α版の後に回されていたので、吸収することをオーナーに一言確認する（m-15）。

### M-9. 常駐の照合（2a-G4）が証明しない事項が多い（計画 §6 2a-G4）

- 「T2V → T2V（`keep_resident: true`）→ T2V（キー無し）で 3 本の指紋が一致」で証明できるのは「先読み on・同じ構成の T2V で、保持した DiT と VAE を使い回しても出力が変わらない」ことだけ。
- 証明しない事項（事実からの帰結）:
  - ジョブの合間に GPU に何も残らないこと（M-1 の同期経路の最後の窓）。
  - キー無しのジョブで主メモリが実際に返ること（コミット量の低下）。第 1 弾ではテキストエンコーダを解放しても作業セットが下がらなかった（設計正本 §4）。
  - 先読み on と off を交互にしたときの包みの付け直し（M-1）。
  - 常駐中の I2V（復号器だけ保持・符号化器はそのジョブで組む。M-8）。
  - 1 本目の `keep_resident` の値（計画の文面では 1 本目に `true` があるのかが曖昧。2 本目で「読み込みが飛ぶ」ためには 1 本目も `true` が要る）。
- 直し方の提案: 並びを「T2V（常駐 on・先読み on）→ I2V（常駐 on・先読み off）→ T2V（常駐 on・先読み on）→ T2V（キー無し）」にし、各ジョブの後に `torch.cuda.memory_allocated()` と `memory_reserved()`、コミット量を記録する。合否は「指紋が常駐なしの同条件と一致」「合間の allocated が 0」「キー無しの後にコミット量が保持前の水準へ戻る（戻らなければ記録して監督判断）」。主メモリの山が見込み（約 30〜35GiB）を超えたときの扱い（合否か記録か）も書く。

### M-10. 2a-G2 の「第 1 弾の方式を削る前に 1 回取る」は、計画の流れ（実装 → 2a-G1 → GPU の門）と順序が合わない（計画 §6 2a-G2・§7）

- 事実: 計画 §7 では C1 が「実装 → 2a-G1 → 2a-G2〜G5」と進む。基準（今の層オフロードの出力）を取るには、実装の途中（`GemmaLayerOffloadService` を外す前）に GPU を使う必要がある。さらに C1 の順序は「1. I2V（`text_encoder.py` を `Qwen3VLModel` に変える）→ 2. 直接読み」なので、基準を第 1 弾のコードで取るのか、I2V の改修後のコードで取るのかが決まっていない。
- 直し方の提案: 基準は dev の現 HEAD（第 1 弾のコード）を別の作業ツリーで動かして取る、と明記する（GPU を途中で使わずに済み、削った後でも取り直せる）。あるいは C1 の順序を「直接読み → I2V」に入れ替え、直接読みの前に基準を取る。I2V の TE の基準（M-7）は直接読みでない方式を一時的に残して取るか、DiffSynth の写しで取るかを決める。

### M-11. 計画 §3.2 の「`ready` に `block_swap_prefetch_available: true` を足す（`GET /status` が読む）」は事実と違う（計画 §3.2「ジョブごとの設定」）

- 事実: `GET /status` の `acceleration.block_swap_prefetch_available` はアプリ側で `self.low_vram.block_swap_blocks_on_gpu or 8 > 0` から作る（`services/pipeline_manager.py:302-334`）。ワーカーの `ready` は読んでいない。LTX 2.5 のワーカーもこのキーを出していない（`engine25` に該当なし）。
- 直し方の提案: この項目を削る（引き算）。H3 のアダプタの `blocks_on_gpu` を同じ式（または 2.5 の `DEFAULT_BLOCKS_ON_GPU` の前例）で作る、とだけ書く。計画 §3.4 の「`blocks_on_gpu` 相当（あれば）」もここで確定させる。

### M-12. 422 の「形と文言を保つ」の中身が足りない（計画 §3.3「検査の場所」・§9-1）

- 事実: 移す前の検査は `model_validator(mode="after")` の `ValueError`（`api/models.py:644-665`）。pydantic v2 と FastAPI の組では、この種の誤りは `loc` が `["body"]`、`msg` が `"Value error, num_frames must be 8n+1"`（接頭辞つき）、`type` が `"value_error"` になる（pydantic の既定の振る舞い。手元で実行しての確認はしていない）。`main.py:370-388` がそれを `{loc, msg, type}` に詰める。既存のテストは `r.text` の部分一致（`"multiple of 64"`・`"8n+1"`・`"num_inference_steps=8"`）だけを見ている（`tests/test_validation.py:14-50・246-252`）ので、接頭辞や `loc` が変わってもテストは通るが、Gradio の `format_api_error` の表示（`loc` と `msg` を連結。`tests/test_gradio_handlers.py:1311-1321`）や MCP の表示は変わりうる。
- 順序の変化: 今は 8n+1／64 の違反が `reject_unsupported` より先に返る。移した後は `reject_unsupported` の後（計画は「隣」としか書いていない）。LTX 2.5 に `pipeline: two_stage_hq` と 8n+1 違反を同時に送ると、`VALIDATION_ERROR` から `FEATURE_UNSUPPORTED` に変わる。
- 他の経路: `GenerateRequest` を作るのは `/generate` だけではない。`api/generate_alpha.py:247` の内部要求と `GenerateChainRequest.to_clip_request`（`api/models.py:1714`）がある。前者は `AlphaGenRequest` 側の 8n+1（`api/models.py:427-432`）と `ceil64` で実害は無い見込みだが、計画に書いておく。
- 直し方の提案: 422 の `detail` は `{"loc": ["body"], "msg": "Value error, <今の英文>", "type": "value_error"}` と一字一句決め、`validate_for_family` は `reject_unsupported` の**後**に置くと明記する（順序の変化は利用者に見える差として記録に残す）。

### M-13. テストファイルの所有が重なる（計画 §4・§7）

- 事実: C1 の所有は `tests/test_h3_*.py`、C3 は新設の `tests/test_h3_adapter.py` を作る（同じ名前の型に入る）。C3 の項目 4 に `test_mcp_*`・`test_gradio_*`（C5 と調整）があり、C5 も「対応するテスト」を持つ。「1 ファイル 1 所有者」（計画 §4 の見出し）に反する。C4 と文書係も `FE/Docs/API_REFERENCE.md` §7 と DEVLOG を両方挙げている（計画 §3.5 末・§3.7・§4 C4）。
- 直し方の提案: C3 の新設テストを `tests/test_engine_h3_adapter.py` などに改名するか、C1 の所有を「`tests/test_h3_engine.py`・`tests/test_h3_media.py`」と名指しにする。`test_mcp_*`・`test_gradio_*` は C5 だけの所有にし、C3 は API の変更で落ちるものの一覧を C5 に渡す。FE の文書は C4 か文書係のどちらか一方にする。

---

## 3. Minor

1. **計画 §3.3 の引用**: MULTI_ENGINE_DESIGN §7（`Docs/MULTI_ENGINE_DESIGN.md:551-574`）は「まだ作らない【オーナー裁定】」と「第 3 段階＝UI 調整（未着手）」を書いているだけで、計画が鉤括弧で引く「系統別の数値の規約は…アダプタが宣言した表で検査する」という文は無い。引用でなく計画の設計として書く。§7 が裁定である点は、設計正本 §11（第 2 弾で正面から扱う）と裁定 23 を根拠に挙げておく。
2. **3 例目で判断する事項**: MULTI_ENGINE_DESIGN §8.1 は、汎用 `EngineRunner` への分割（§5.2）・mock の共通化（§9.1）・カテゴリ集合の可変化（§4.1）を「3 例目が来るまで保留」としている。H3 が 3 例目なので、計画（`H3Runner(LTXRunner)`・mock 共有・4 語のカテゴリ）が「引き続き保留」を選んだことを 1 行で書く。
3. **2.3 の「無視＋ログ」**: 2.3 の `IGNORED_FIELDS` は宣言だけでログを出さない（`ltx/adapter.py:142-149`「DECLARATIVE ONLY」）。計画 §3.3 の「2.3・2.5 とも無視＋ログ」は 2.3 では「無視（宣言のみ）」。
4. **pydantic の上限**: 「ステップ 1 以上」として `le=100` を外すと、LTX 2.3 の `two_stage_hq` に 100 超が通る。`le=100` は残してよい（H3 の 50 はその内側）。
5. **`frame_rate` の範囲の二重化**: LTX の規則表に「fps 1〜60」を書くと pydantic の `ge=1, le=60`（`api/models.py:592`）と同じ値が 2 か所になる。`num_frames.max: 481` も pydantic の `le=481` と重なる。`constraints` は規則表から作るなら、pydantic 側の値を規則表の定数から取るか、`constraints` の該当値を pydantic の `Field` から読む。
6. **`multi_keyframes` の二重の検査**: H3 の `REQUEST_RULES`（0 か 1 枚・`frame_idx == 0`）と `REJECT_TABLE` の `multi_keyframes` が同じ条件を 2 か所で持ち、違反時の符号も 2 通り（`VALIDATION_ERROR`／`FEATURE_UNSUPPORTED`）になる。拒否表の 1 か所にする。なお `_normalize_conditioning_images` が先に 8n+1 の格子へ寄せる（`api/models.py:190-216`）ので、`frame_idx: 5` は `1` に書き換わってから拒否される（文言に出る値が送った値と違う）。
7. **強度の警告が毎ジョブ出る**: `ConditioningImage.strength` の既定は 0.8 で、アダプタは既定以外を拒否する。ワーカーが「受け取っても無視して警告ログ」だと、I2V の全ジョブで警告が出る。ワーカーへ `strength` を送らないか、警告を出さないかのどちらかにする。
8. **条件行の乱数の引き順**: 「ジョブの種から f32・行の形で 1 回作る」の引き順を決めておく。今は 1 つの CPU 生成器から映像→音声の順に引く（`engine_h3/pipeline_h3.py:575-579`）。その後に同じ生成器から引けば T2V の指紋は変わらず、同じ種の I2V と T2V で映像と音声のノイズが一致する。
9. **復号の uint8 化の丸め**: ビット一致（T2V の指紋 `0da6111c…`）を保つには、今の `frames_to_uint8` と同じく `clamp → ×255 → round → uint8` の順を守る（`pipeline_h3.py:182-185`）。計画 §3.2 は `round` を書いていない。重なりの混ぜ（`blend`）は正規化の前に済ませる必要があり、`_decode_temporal` で `parts.append(piece)` する時点（`video_vae.py:642-647`）は混ぜた後なので、その片を変換して CPU へ送るのは成り立つ（事実）。最後の `overlap` と、`pad_tokens` があるときの末尾の切り詰め（同 649-658）も同じ契約に乗せる。
10. **視覚塔へ渡す画像**: DiffSynth はキーフレームを `img.resize((width, height), Image.LANCZOS)`（縦横比を保たない引き伸ばし）してからプロセッサに渡す（scratchpad `src/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:304-306`）。計画は VAE 側を「覆う拡縮＋中央切り抜き」にする。視覚塔にも同じ切り抜き後の画像を渡すこと、2a-G3 のトークン列の照合では両側に同じ切り抜き後の画像を入れることを INTERFACE2 の前に決めておく。
11. **2a-G3 の正解の材料**: 「視覚塔の出力は int8 と bf16 版で G4-TE と同じ水準」には視覚塔の bf16 の重みが要る。第 1 弾の G4-TE は層 0／1／49 の bf16 を使った（`Docs/VERIFICATION_LOG.md:17027`）が、視覚塔の bf16 の重みが手元にあるかは未確認。取得が要るなら、了承の対象（GPU でなくダウンロード）として計画に書く。
12. **2a-G3 の符号化のビット一致**: DiffSynth の写しと「bf16 どうしでビット一致」は、タイルの切り方と演算順が完全に同じでないと成り立たない。一致しなかったときの許容（差の最大値の上限など）を先に決めておく。
13. **2a-G5 の範囲**: 映像の復号だけを測る門なので、計画 §1 の「大きな画角でも復号が VRAM からこぼれない」は「映像の復号は」に絞る（音声の窓なし復号は §1-95 で測ると計画自身が書いている）。合格の線「予約ピーク 16GB 未満」は、AviUtl2 のプレビューと同居する余裕（専用 VRAM の何割か）で決めた方が実用に近い（推測）。
14. **2a-G4 の見込みの参照先**: 「主メモリの山が 2a-G2 の見込み（約 30〜35GiB）」とあるが、見込みは §3.2 にあり 2a-G2 には無い。コミット量と作業セットのどちらで判定するかも書く（第 1 弾はコミット量の山 40GiB・作業セットの山 29GiB。設計正本 §4）。
15. **§1-99 の吸収**: 裁定 25 は §1-98 と §1-99 を α版の後に回した。裁定 32 が戻したのは §1-98 だけ。§1-99 の吸収は小さいが裁定の変更なので、オーナーに一言確認する。
16. **Edit タブ丸ごと灰色と裁定 30**: 裁定 30 は Inpainting・Outpainting を「グレーアウトして配置だけ決めておく」とする。Edit タブごと閉じると中のサブタブは見えない（画面がマウントされない）。計画は「合致する」と判断しているが、オーナーの意図（サブタブの配置が見えること）と一致するかは確認を勧める。
17. **A2V と NAG を「LTX 固有」として非表示にする判断**: 裁定 30 の列挙（IC-LoRA・Alpha Gen・Stage-2 窓）に A2V と NAG は無い。計画の解釈なので、オーナーに確認する（M-3 の結論しだいで NAG は灰色が素直）。
18. **`maxItemsOverride: 1` の限界**: 追加ボタンを止めるだけで、(a) 残る 1 枚の `frame_idx` は自由に変えられる、(b) LTX で 2 枚以上置いた後に H3 へ切り替えると、既存のカードは残る（`useKeyframes.ts:136-148`）。オーナーの好み（フールプルーフを作らず 422 の文言で解決）に沿うなら、送信時の 422 の文言（「MiniMax H3 は先頭フレーム 1 枚だけに対応しています」）を整えることを計画に書く。
19. **設計正本 §8 のキー名との食い違い**: 設計正本 §8 の `load` のキーには `te_layers_on_gpu` があり、`generate` のキーには `images`・`keep_resident`・`attention_backend`・`block_swap_prefetch` が無い（`Docs/MINIMAX_H3_DESIGN.md:170-176`）。C1 と C3 が並行で書くので、境界のキー一覧（追加と削除）を計画 §3.1〜3.2 に表で固定しておく。`ready` のキー（M-11）も同じ。
20. **Gradio の快適上限**: `gradio_ui/comfort.py` の `resolve_chain_budget`（93-120）は Chain の Stage-2 窓のラベルにしか使われていない。H3 では Chain が使えない（422）ので、ここを変える必要は無い（引き算）。一方、Gradio の Single の溢れ表は `/config` の `limits.spill_free_frames`（LTX の実測）から作る（R5a E1）ので、H3 でも LTX の表が出る。こちらをどうするか（出したままにするか）を決める。
21. **Settings の説明文の数値**: 「MiniMax H3 では DiT と VAE 約 26GB」を画面の文言に直書きすると、正本 1 か所の原則（数値は記録簿が正本）とずれる可能性がある。実測後に書くか、数値を書かない文にする。
22. **`metadata["h3"]` に入れるもの**: 計画は `h3` 欄に「ステップ・shift・段ごとの秒数・コミット量・`keep_resident_used`」を入れるとするが、ワーカーの `done` ではステップ・shift・コミット量・`keep_resident_used` は最上位のキーで、`h3` の中身は `phases` と `quant_backends` だけ（`engine_h3/worker.py:50-54`、`pipeline_h3.py:429-449`）。`keep_resident_used` は既存の `GenerationOutcome.keep_resident_used` 欄（`ltx/adapter.py:574`）で運べる。C-1 の書き戻しを採れば、ステップと shift は `metadata.json` の `request` に残るので、操作パネルが「既定でも常に送る」（計画 §3.5）必要も無くなる（引き算）。

---

## 4. 観点ごとの確認結果（事実として一致した点を含む）

### A. 事実の確認

| 観点 | 結果 |
|---|---|
| (1) `validate_ltx_constraints` の切り分け | 移す 3 種（64 の倍数 646-649・8n+1 650-651・distilled の 2 行 657-665）と、残す部分（`crop_output` 652-656、スナップ 668、IC-LoRA・NAG・outpaint・inpaint・alpha_gen 670-785）の切り分けは正しい。連結（`GenerateChainRequest` 1334-1352・1469）は別の検査で、計画どおり触らない。ただし C-1・M-12 |
| (2) `GenerationOutcome` と `metadata["ltx25"]` | 前例は計画どおり（`ltx/adapter.py:611`、`pipeline_manager.py:1950-1951`）。中身の割り振りは m-22 |
| (3) block swap の二重包みと `release_installed()` | 二重包みは事実。`release_installed()` は包みを外さない（M-1） |
| (4) `decode`／`_decode_temporal` の窓 | 「窓ごとに CPU へ」は成り立つ（m-9） |
| (5) `FEATURE_UI` と `CONTAINER_TARGETS` | 表の振る舞いは計画の理解どおり。Alpha Gen の行の扱いは M-2、名前の導出は M-4 |
| (6) `resolveComfortRow` の落ち先 | M-6 |
| (7) `useKeyframes` の `maxItemsOverride` | 実在（`useKeyframes.ts:18-25・136-148`、Chained が `useChainForm.ts:1458` で使用）。限界は m-18 |
| (8) Gradio と MCP | スライダーは `interactive=False` の飾り（R5a E1）、送信は固定値（`handlers.py:604・991・1495`）。MCP は `num_inference_steps` を意図的に出さず、テストがそれを固定（`tests/test_mcp_tools_generate.py:641・1357`。C5 で書き換えが要る） |

### B. 設計の穴（要約）

(1) C-1・M-12。(2) `constraints` の形: `frame_rate` が LTX は `{min,max}`、H3 は `{fixed}` と 2 つの形になる。H3 も `{"min": 24, "max": 24}` とし、画面は「min と max が同じなら欄を無効」と読めば型が 1 つで済む（引き算）。`dimension_multiple` と `num_frames` の格子は過不足なし。(3) M-1・M-8・M-9。主メモリの見込み（DiT 19.5GiB〔`pipeline_h3.py:88-89`〕＋VAE＋ピン留めのプール 2×369MB＋基礎）の足し算は筋が通るが、第 1 弾のコミット量が作業セットより約 11GiB 多かった事実（設計正本 §4）を考えると、コミット量では 35GiB を超える可能性がある（推測）。(4) M-7。`te_layers_on_gpu` を外して困るアプリ側の経路は無い（H3 のアダプタはまだ無い）。外すのはワーカーの `H3LoadOptions`・`_do_load`・セルフテストの `--te-layers-on-gpu`・黄金キーのテスト・設計正本 §4／§8 の記述。(5) m-8・m-10・M-8。(6) M-2・M-3・M-4、m-16・m-17。(7) M-6。(8) ジョブごとの `attention_backend` は副作用なし（ワーカーの `_resolve_attention` をジョブごとに呼ぶだけ）。`block_swap_prefetch` は常駐と組むと M-1。(9) `H3FormatError` の変換は計画どおりで正しい（`services/model_registry.py:176-181` は `APIError`・`OSError` だけを捕まえる）。(10) カテゴリ 4 語・`assets.tokenizer_dir`（2.3 の `gemma_root` と同じフォルダ方式）・`downloads` の `"kind": "dir"` は前例どおり。インストーラは `assets` の値が `downloads[].files[].path` のどれかと一致することを要求する（R5a C2、`scripts/install_ltx.ps1:429-437`）ので、tokenizer の行の `path` は `MiniMaxH3/TextEncoder/qwen3vl_tokenizer` にそろえる。

### C. 門の妥当性（要約）

- 2a-G2: M-10。I2V の TE の照合は M-7。
- 2a-G3: m-10・m-11・m-12。条件行の並び・時刻の番号・RoPE を float64 で完全一致させる部分は、DiffSynth の写しが手元にある（scratchpad `src/DiffSynth-Studio`）ので材料は実在する。
- 2a-G4: M-9。T2V の回帰は、TE の直接読み・`Qwen3VLModel` への切り替え・復号の変更（m-9 の丸めを守れば）のいずれも計算を変えない設計なので、指紋の一致を要求できる。
- 2a-G5: 「乱数の潜在を `H3VideoVAE.decode` に通すだけ」の試験は、潜在の形 `[1, 24, T', 48, 84]`（1344×768・345 フレームで T'=102）を作れば書ける。m-13。
- 2b-G1・2b-G2・2c-G1・2c-G2: 証明する内容は明確。2b-G1 は C-1 を直さないと通らない。

### D. 順序と並行

- M-13（テストと FE 文書の所有の重なり）。
- C3 の `_build_load_payload`・`generate` のキーは C1 のワーカーの受け口に依存する（m-19）。C5 と C4 は `constraints` の形と H3 の機能名の一覧に依存する（計画の JSON と機能名の表で先に固定されているので並行は可能。ただし M-4 で名前を変えるなら先に確定させる）。

### E. 過剰設計の疑い（採否は監督の判断）

1. `ready.block_swap_prefetch_available`（M-11）— 削れる。
2. `frame_rate.fixed`（B(2)）— `min == max` で足りる。
3. 機能名 `ic_lora`・`v2v`・`end_source`・`a2v`・`vsf`・`negative_prompt` と UI の `prompt.negative`（M-4）— 既存の名前と `chain` で足りる。
4. Alpha Gen の非表示化と `EditSubTabs.tsx` の改修（M-2）— H3 では画面に現れないので削れる。
5. `multi_keyframes` の規則表と拒否表の二重化（m-6）、強度の警告（m-7）— 1 か所に。
6. Chained・Retake・Gradio の `comfort.py` の落ち先の変更（M-6・m-20）— H3 では閉じる画面なので不要。
7. 操作パネルが既定値でも 3 項目を常に送る（m-22）— サーバーが既定値を書き戻せば不要。

---

## 5. 確認できなかった点（未確認）

- pydantic v2 のモデル検証の誤りの `loc`・`msg` の正確な文字列（M-12）は、手元で実行して確かめていない（テストの実行は禁止のため）。
- 先読みの `PrefetchEngine.teardown` の後に、各ブロックの引数が CPU の正本を指しているか（`_release` の中身）。
- 配信中の快適上限表が LTX の全ての高速化の組み合わせと重みの種別を覆うか（M-6 の影響範囲）。
- 視覚塔の bf16 の重みが手元にあるか（m-11）。視覚塔のテンソルが int8 か bf16 か（計画も未確認と明記）。
- 常駐時の主メモリの実量（計画も門で実測と明記）。
