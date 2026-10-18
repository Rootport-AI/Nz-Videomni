# MiniMax H3 推論エンジン第 1 弾 コードの事実確認レビュー（REVIEW_code1）

- 日付: 2026-10-17
- 対象: `h3_format.py`・`engine_h3/` 一式・`tests/test_h3_format.py`・`tests/test_h3_engine.py`・`engine_h3/engine_h3-venv-pyproject.toml`・`engine_h3/venv-engine-h3.freeze.txt`
- 方法: 読むだけ。ファイルの編集・GPU の使用・インストールはしていない。CPU で走らせた小さな検査は 4 つ（下の「CPU 検査」）。

---

## (1) 総評

1. 致命的（Critical）な欠陥は見つからなかった。数値の規約（σ の列・Euler の向き・時刻の番号・RoPE・AdaLN の端）と、ワーカーの約束事（イベントの形・stdout の清潔さ）は、参照実装と前例のとおりに組まれている。
2. 重い指摘（Major）は 5 件。性格は 2 種類ある。1 つは**今の門の条件（640×384・124 フレーム）では表に出ない資源の山**で、主メモリの山が設計の想定を約 2.5GiB 上回ることと、大きな画角での復号の VRAM である。もう 1 つは**検証と保守の穴**で、テキストエンコーダ（TE）の数値照合が無いこと、テストの実行手段と門の台本が版管理の外にあること、VAE と mp4 書き出しのテストが 0 本であることである。
3. TE の解放漏れは見つからなかった。小型模型で確かめると、参照は gc 後に消える。RSS（作業セット＝プロセスが実メモリに置いている量）が下がらないのは割り当て器が解放済みの領域を抱えているためと見るのが自然である（推測。根拠は M1）。

### CPU 検査（`.venv-engine-h3`・`CUDA_VISIBLE_DEVICES=""`・`-B`）

| 検査 | 結果 |
|---|---|
| `tests/test_h3_engine.py`（`outputs/h3-gate1/run_h3_pytest.py` 経由） | 30 件合格・2 件スキップ（GPU 用） |
| `tests/test_h3_format.py`（アプリ側の `.venv`） | 31 件合格 |
| 小型の `Qwen3VLTextModel`（8 層）に `GemmaLayerOffloadService` を入れて前向き計算し、参照を落として gc | 模型と層の弱参照は消えた（参照の取り残しなし） |
| 小型の `H3VideoVAE` で F=5／22／124／345 を検証して復号 | 22→22 フレーム・窓 1、124→124・窓 7、345→345・窓 20。**F=5 は `validate_request` を通るのに復号で失敗**（m1） |

---

## (2) 指摘の一覧（重さの順）

所有者の記号: C1 = quant_linear／dit／rope、C2 = video_vae／audio_vae／media_out、C3 = その他。

### Major

#### M1. 主メモリの山が設計の想定を上回る。段 1 の後も RSS が下がらない

- **指摘**: 設計書 §4 は「段 1 の TE を解放してから段 2 の DiT を読むので、約 27GB に少しの上乗せで収まる。両者が同時に主メモリに載ることはない」と書く。G5 の実測は次のとおりである。
  - TE を解放した直後の RSS は 23.88GiB で、解放前の 23.55GiB から下がっていない。
  - DiT を読んだ後の RSS は 28.38GiB（2 ラウンド目は 29.65GiB）。
  - プロセスの山は 29.11〜29.65GiB。
  - 復号の終わりには 2.69GiB まで落ちる。

  参照の取り残しではない。CPU 検査で、TE と同じ組み方（meta で組む → `GemmaLayerOffloadService.install` → 前向き計算 → `release` に相当する手順）をすると、模型は gc 後に回収された。`phases` と `build_report` とログにもテンソルは入っていない。
  一方、torch の `c10.dll` には mimalloc（Microsoft 製のメモリ割り当て器）の文字列が埋め込まれている（観測）。解放済みの領域を割り当て器が抱え、後で OS へ返していると見ると、次の 3 つの観測がすべて説明できる（推測。要実測）。
  - 解放後も RSS が下がらない。
  - DiT を読んでも RSS は +4.5GiB しか増えない。DiT の fc1（1 本 154MB）は TE の最大テンソル（131MB）の空き穴に入らないので、そのぶんは新しく確保される。
  - 復号中に 2.69GiB へ落ちる。

  利用者への影響は次のとおりである。
  - **32GB の機械**: 山の 29.6GiB に OS・AviUtl2・アプリのサーバーが加わるので、ページアウトがほぼ確実に起きる。
  - **16GB の機械**: TE の CPU 側だけで 22.6GiB あり、ページファイルが前提になる。
- **根拠**: `outputs/h3-gate1/g5_t2v.json` の `h3.phases`（encode の `rss_after_load_gib` 23.55・`rss_after_release_gib` 23.88、denoise の `rss_after_load_gib` 28.38／2 ラウンド目 29.65、decode の `rss_end_gib` 2.69）。`Docs/MINIMAX_H3_DESIGN.md:97`。`engine_h3/pipeline_h3.py:171-213`（`_rss_gib` は WorkingSetSize だけを返す。構造体にはすでにある `PagefileUsage`＝コミット量を捨てている）。
- **直し方の案**:
  1. `_rss_gib` で `PagefileUsage`／`PeakPagefileUsage` も記録する（構造体は定義済みで、1 行足すだけ）。作業セットは OS の刈り込みで小さく見えることがあり、コミット量のほうが「主メモリを本当に使っているか」の答えになる。
  2. 割り当て器の保持が原因かどうかを、mimalloc の環境変数（解放を即時にする系統）を付けた 1 本で確かめる（推測なので、まず計測で確定する）。
  3. 根本策として、TE は各層を 1 回しか通らないので、CPU に全層を持たず、層ごとにファイルから GPU へ直接読む。今も TE の読み込みに 12.9 秒かけているので、所要時間は大きく変わらない見込み（推測）。これで山は DiT ぶん（約 20GiB）まで下がる。
  4. 設計書 §4 の文を実測に合わせて直す。
- **所有者**: C3（pipeline_h3・text_encoder）

#### M2. 大きな画角では、復号結果を float32 のまま GPU に全部置くので VRAM が足りなくなる見込み

- **指摘**: 復号の流れでは、float32 の大きな配列が何本も同時に GPU に載る。
  - `_decode_temporal` が窓ごとの復号結果を float32 で `parts` にためる。
  - `torch.cat` でもう 1 本を作る。
  - `pixels * ps + pm` でさらに 1 本。
  - `frames_to_uint8` が GPU 上で clamp・×255・round を行い、それぞれ新しい配列を作る。
  - 映像 VAE 本体（4.85GiB）は、`vae.decode` の間ずっと GPU にある。

  1344×768・345 フレームを計算すると、復号結果 1 本が 3×345×768×1344×4 バイト＝約 4.3GB になり、VAE と合わせて 13〜17GB に達する。16GB の GPU では OOM（GPU のメモリ不足で処理が落ちること）になる見込みである（計算による。実測ではない）。G5 の 640×384・124 フレームでは 1 本 0.37GB なので表に出ない。第 2 弾の快適上限の較正（どこまでの画角と尺が快適に動くかの計測）で最初に当たる壁になる可能性が高い。
- **根拠**: `engine_h3/video_vae.py:596-597`・`614-630`。`engine_h3/pipeline_h3.py:161-164`・`578-585`。
- **直し方の案**: 窓ごとに「ImageNet の逆正規化 → clamp → uint8 化 → CPU へ移す」を済ませ、GPU に float32 の全長を作らない。重なりの混ぜ合わせ（クロスフェード）に要るのは直前の窓の 5 フレームだけである。`frames_to_uint8` は CPU 側で行うか、窓ごとに分ける。
- **所有者**: C2（video_vae）・C3（pipeline の `frames_to_uint8`）

#### M3. TE の出力を参照実装と照合した門が無い

- **指摘**: G4 は DiT の部品ごとの照合で、TE は範囲外である。TE には間違えやすい点が集まっている。
  - int8 ConvRot の Qwen3-VL。
  - transformers 5 での rope 設定の読み替え。
  - 50 層目の後・正規化の前の隠れ状態を取ること。
  - 追加の特殊トークン。
  - R2 (4) の未確認事項「deepstack を含めて数値的に一致するはず（推測）」。

  それなのに、確かめているのは設定値（`check_rope`）とトークン番号と名前の対応だけで、出力の数値は誰とも比べていない。TE の誤りは画が壊れる形ではなく「プロンプトの効きが弱い」形で出るので、1 本の目視では見逃しやすい。
- **根拠**: `outputs/h3-gate1/g4_result.json`（DiT のみ）。`g0_g1.json`。`tests/test_h3_engine.py` の TE のテスト（設定と名前だけ）。`R2_model.md` (4) の最終項。
- **直し方の案**: G4 E（ブロックごとの照合）と同じ要領で、TE の数層について 2 つを比べる。
  1. int8 の経路と、bf16 に戻した重みでの経路を、層ごとに比べる。
  2. 同じプロンプトで、DiffSynth の `export_text_embedding`、または diffusers の `hidden_states[50]` と最終出力を比べる。CPU の float32 で数層だけでもよい。
- **所有者**: C3（text_encoder）

#### M4. テストの実行手段と門の台本が版管理の外にある

- **指摘**: `.gitignore` の 21 行目が `outputs/*` を除外している。そのため、`tests/test_h3_engine.py` の docstring が実行方法として指す `outputs/h3-gate1/run_h3_pytest.py` も、門の証拠である `g2_int8_smoke.py`・`g3_blockswap.py`・`g4_compare.py` も、clone した先には存在しない。半年後に保守する人は、テストを書いてあるとおりに動かせないし、門をやり直すこともできない。
- **根拠**: `.gitignore:21-22`。`tests/test_h3_engine.py:3-7`。`git status` では `engine_h3/`・`tests/test_h3_*.py` が未追跡で、`outputs/` は出てこない。
- **直し方の案**: 前例（`Docs/Outputs-archive/start-end-bridge-2026-09-07/implA_engine_runner/run_ltx25_pytest.py`）に倣い、テストの実行器を版管理下に置く（`tests/` か `scripts/`、または `Docs/Outputs-archive/` の下）。門の台本も同じ場所に保管し、docstring の参照先を直す。
- **所有者**: C3

#### M5. 映像 VAE・音声 VAE・mp4 書き出しの CPU テストが 1 本も無い

- **指摘**: C2 の 3 ファイルは G5 の通し実行でしか確かめられていない。次の点はどれも、小型の模型を使えば CPU で数秒で固定できる。
  - 17n+5 ⇔ T' の復号の算術（窓の数・フレーム数・詰め物の切り落とし）。
  - `split_tiles` の被覆（タイルが画面をちょうど覆うこと）。
  - BT.709 の変換（既知の色で Y／Cb／Cr を確かめる）。
  - 音声の切り揃えと、書き出しに失敗したときの片づけ。
  - 1 フレームの復号。

  実際、今回の CPU 検査でこの種の確認をしたら、F=5 の不一致（m1）がすぐに見つかった。
- **根拠**: `tests/test_h3_engine.py` に `video_vae`・`audio_vae`・`media_out` の参照が 1 つも無い。`video_vae.py:391-393` は「CPU テストが小さな復号器を作れるように」引数を用意しているのに、そのテストが無い。
- **直し方の案**: 小型の `H3VideoVAE`（`ch=32`・`decoder_layers=1` で組める。CPU 検査で確認済み）で F＝22／124／345 → 同じフレーム数・窓 1／7／20 を固定する。あわせて、`split_tiles` の被覆、`rgb_to_i420_bt709` の既知色、`write_mp4` の音声の切り揃え（`round(F×32000/24)`）と失敗時の削除を固定する。
- **所有者**: C2

### Minor

#### m1. F=5 は検証を通り、TE と DiT を回し切った後の復号で落ちる

- **指摘**: `latent_frames(5)` は T'=2 を返すので `validate_request` を通る。ところが `_decode_temporal` では窓の数が 0 になり、`ValueError` が出る。約 2 分の計算が無駄になり、出る文も「潜在フレーム数が短すぎる」で、利用者には分かりにくい。
- **根拠**: `engine_h3/constants.py:47`（`f < 5` だけを拒否）。`video_vae.py:607-611`。CPU 検査で再現した。
- **直し方の案**: `latent_frames` か `validate_request` で n ≥ 1（F ≥ 22）を要求する。R2 (3)-15 の範囲（124〜345）に合わせるかどうかは第 2 弾の API の制約として決める。
- **所有者**: C3

#### m2. 映像 VAE の「float32 の島」が、bf16 を一度通ってから float32 に戻されている

- **指摘**: 「島」とは、精度を落とすと結果が変わりやすいので float32 で計算する小さな部品群のことである。`from_file` の手順は、`model.to(dtype=bf16)` で全体を一度 bf16 にしてから、`_restore_float32_islands` で島を `.float()` に戻すという順になっている。このため F16 のファイル値が bf16 で丸められ、仮数の 3 ビットが失われる（相対誤差は約 0.2%）。docstring の「float32 に保つ」と食い違う。島には norm の重み・`scale1`／`scale2`・`x_embedder`・`post_quant_conv` が含まれる。
- **根拠**: `video_vae.py:436-455`。
- **直し方の案**: 島を先に `.float()` にしてから、島以外だけを bf16 に変換する（またはモジュールごとに変換先の型を選ぶ）。
- **所有者**: C2

#### m3. T2V でも映像 VAE の符号化器（encoder 0.34GiB）を読み、GPU に載せている

- **指摘**: 第 1 弾は復号しか使わないのに、`encoder.*` と `quant_conv` を毎ジョブ読み込み、GPU に置いている。音声 VAE は復号器だけを組んでいるので、扱いがそろっていない。
- **根拠**: `video_vae.py:406-407`・`89-95`。ヘッダの実測では decoder 4.514GiB・encoder 0.336GiB。
- **直し方の案**: 復号専用で組む選択肢を足し、`encoder.*`・`quant_conv.*` を key_map で None にする（I2V の第 2 弾で切り替える）。
- **所有者**: C2

#### m4. `BlockSwapService` をジョブごとに作り直している（前例は常駐）

- **指摘**: 固定ページ（ページアウトされない主メモリ）の受け皿 `PinnedStagingPool`（2×369MB）と、転送用の CUDA ストリームが毎ジョブ作り直しになる。受け皿の docstring には「ジョブごとに作らない。作り直しを繰り返すと 7 本目で先読みが効かなくなる」と明記されている。`engine25` はサービスを常駐させている。torch のホスト側キャッシュが同じ大きさの領域を再利用するので、実害は小さい見込み（推測）。
- **根拠**: `pipeline_h3.py:472`。`engine/transformer/block_swap_prefetch.py:234-243`。`engine25/gguf_transformer.py` の `Ltx25DiffusionStage.__init__`。
- **直し方の案**: `H3Pipeline` にサービスを 1 つ持たせ、毎ジョブは `prefetch_requested` → `install` → `teardown_prefetch` → `release_installed` だけを行う。
- **所有者**: C3

#### m5. ジョブが失敗した後、CUDA のキャッシュを空けない

- **指摘**: `generate` に `finally` の `_free_cuda()` が無い。失敗したジョブで予約された VRAM は、次のジョブの TE 段まで保持されたままになる。待機中に AviUtl2 のプレビューと VRAM を取り合う。
- **根拠**: `pipeline_h3.py:358-393`。`worker.py:327-333`。
- **直し方の案**: `generate` 全体を `try/finally: _free_cuda()` で包むか、ワーカーの GENERATE_FAILED の後に gc と `empty_cache` を行う。
- **所有者**: C3

#### m6. 量子化されていない層の dtype の揺れを吸収できない

- **指摘**: `assign=True` はファイルの dtype をそのまま使う。refiner の `qkv_proj`／`out_proj`／`fc1`／`fc2` は bf16 の素の `nn.Linear` なので、F16 や F32 のファイルが来ると、TE 段の後に dtype の不一致で落ちる。norm の重みが F32 だと、後続の計算に f32 が流れ込む。`h3_format` は dtype を検査しない。`dit.py` の docstring の「結果は読み込んだ dtype に依存しない」は、島についてだけ正しい。今の配布ファイルは BF16 なので、表には出ない（ヘッダで確認した）。
- **根拠**: `dit.py:43-49`・`213-216`・`331-336`。`sft_load.py:90`。`Docs/MINIMAX_H3_DESIGN.md` §7「量子化されていない層は保存された dtype のまま読む」。
- **直し方の案**: 読み込み後に島以外の浮動小数のテンソルを bf16 に変換する。または `h3_format` で dtype を検査し、理由の文つきで拒否する。
- **所有者**: C1（dit）・C3（sft_load・h3_format）

#### m7. load の数値オプションに明示的な null が来ると、load が失敗する

- **指摘**: `int(msg.get("blocks_on_gpu", 既定))` に `None` が来ると TypeError になる。`attention_backend: null` は文字列 `"None"` になり、未知の値として拒否される。`generate` 側は `_or` で null を既定値として扱うので、扱いが食い違う。
- **根拠**: `worker.py:157`・`207-214`・`236-238`。
- **直し方の案**: load 側も `_or` と同じ扱いにする。または INTERFACE に「キーを省くか値を入れる（null は禁止）」と書く。
- **所有者**: C3

#### m8. 進捗の割り当て: 第 2 弾のアダプタは `one_stage=True` を渡す必要がある

- **指摘**: 親の `_progress_frac` は、単発の生成では `stage1_denoise` を 0.06〜0.50 に割り当て、`decode` には None を返す。H3 には 2 段目が無いので、進捗バーは 0.50 で止まり、復号の 12 秒の後に 0.90 へ飛ぶ。Alpha Gen の前例と同じく `one_stage=True` が要る。`index`／`total` の形（0..N）は前提に合っている。
- **根拠**: `services/engines/ltx/adapter.py:476-482`。G5 の progress の並び。
- **直し方の案**: INTERFACE と設計書 §8 に明記する。
- **所有者**: C3（第 2 弾）

#### m9. `deterministic` は cudnn のフラグ 2 つを立てるだけで、名前が約束しすぎている

- **指摘**: DiT には畳み込みが無いので、DiT の結果には効かない（効くのは VAE だけ）。G5 はこのオプション無しでも sha256 が一致した。第 1 弾の範囲では過剰な設計にあたる。
- **根拠**: `pipeline_h3.py:318-320`。
- **直し方の案**: 削る。残すなら「VAE の cudnn だけ」と docstring に書く。
- **所有者**: C3

#### m10. pyproject の説明が事実と食い違う

- **指摘**: einops の理由として「the ported video VAE uses it」と書いてあるが、`engine_h3` では使っていない（`video_vae.py:10-11` は置き換えたと書いている）。逆に、`numpy` は `rope.py` と `sft_reader` が直接使うのに依存表に無く、transformers 経由の間接的な依存に頼っている。
- **根拠**: `engine_h3/engine_h3-venv-pyproject.toml`（WHY THESE VERSIONS の einops の項）。`rope.py:31`。
- **直し方の案**: einops の項を「未使用（削除候補）」に直し、numpy を直接の依存として明記する（または freeze の注記に書く）。
- **所有者**: C3

#### m11. 起動完了ログ（LOAD_OK）が `ensure_ascii=False` で、同じファイルの「STDERR は ASCII のみ」と食い違う

- **指摘**: 日本語環境では stderr が cp932 で書かれ、親はログを UTF-8 として読む。そのため、パスに日本語を含むとログが文字化けする。
- **根拠**: `worker.py:161`（規則）・`228`（LOAD_OK）。`services/engines/ltx/adapter.py:2060`（ログを UTF-8 で開く）。
- **直し方の案**: `ensure_ascii=True` にする。
- **所有者**: C3

#### m12. `write_mp4` の finally で `container.close()` が例外を出すと、部分ファイルが残る

- **指摘**: close が例外を出すと unlink が実行されない。元の例外も上書きされる。
- **根拠**: `media_out.py:137-140`。
- **直し方の案**: close を try/except で包み、unlink を必ず実行する。
- **所有者**: C2

#### m13. 同じ処理と同じ許容値の重複

- **指摘**: パスの存在確認が 2 か所にある（`worker._do_load`／`H3Pipeline.load`）。attention の許容値の列挙が 3 か所にある（`worker._resolve_attention`・`pipeline_h3.ATTENTION_BACKENDS`・`dit._ATTENTION_BACKENDS`）。出力フォルダの作成も 2 か所にある（`_decode` と `write_mp4`）。
- **直し方の案**: 正本を 1 か所（constants）にして参照する。
- **所有者**: C3

#### m14. 第 1 弾では使われないコード（死蔵）

- **指摘**: 次のものは T2V では呼ばれない。
  - `H3VideoVAE.encode`・`_encode_clip`・`EncoderFCN3D`・`quant_conv`
  - `build_coords(cond_frames=1)`
  - `embed_rows(cond_rows=…)`
  - `KEYFRAME_NOISE_AUG`
  - `H3SeqLayout.cond`

  このうち I2V の座標だけは G4 A（small_i2v・portrait_i2v）で照合済みである。符号化の経路は一度も実行されていない。
- **直し方の案**: 第 2 弾まで残すなら、各所に「第 1 弾では未使用・未検証」と書く。
- **所有者**: C2・C3

#### m15. テストの副作用と、約束事の固定テストの欠如

- **指摘**:
  - `engine_h3.worker` を import すると、import の時点で `os.chdir(ROOT)` と「ルートロガーに INFO のハンドラを足す」処理が走り、pytest のセッション全体に及ぶ。
  - load／generate／done のキー集合を固定するテストが無い。前例は `tests/test_ltx25_adapter.py` のゴールデン（正解の写しとの比較）である。第 2 弾の接続者が頼れる固定点が無い。
- **直し方の案**: キー集合のスナップショットテストを足す。import の副作用は `importlib` で隔離するか、テスト側で元に戻す。
- **所有者**: C3

#### m16. 0 の意味がオプションごとに逆向き・音声の報告値の名前

- **指摘**:
  - `te_layers_on_gpu=0` は実際には 1 層（`max(1, …)`）になる。`blocks_on_gpu<=0` は全常駐（19.5GiB）になる。
  - `entry["audio_samples"]` は切り揃える前の 165,600 で、mp4 に入るのは 165,333 である。
- **根拠**: `text_encoder.py:247`。`pipeline_h3.py:465-468`・`599`。
- **直し方の案**: docstring と INTERFACE に意味を書く。報告値は切った後の値にするか、名前を変える。
- **所有者**: C3

---

## (3) 確認したが問題なしと判断した点

- **数値の規約**（R2 (3)。G4 で機械的に確かめていない項目）:
  - σ の列は N+1 点の `[:-1]` に終点 0 を足した形で、N 回評価する（DiffSynth の `set_timesteps_minimax_h3` と同じ）。
  - Euler は `x + (σ − σ_next)v` で、DiffSynth の「出力の符号を反転してから一般の更新式」と同値である（`minimax_h3_audio_video.py:1079`・`flow_match.py:409`）。
  - 時刻は t = 1 − σ。映像と音声は同じ添字 i で別々の列を使う。テキスト行は映像の t を使う（参照 1010-1011 と同じ）。
  - `unique` による番号付けは、G5 の `m_per_step` が 1 → 2 で実測どおり。
  - token_refiner はループの前に 1 回だけ計算する。条件行は空。
  - 精度の島（patch_proj・出力の 2 本・AdaLN）は float32。
  - RoPE は座標が float64、角度が float32、掛け算はブロックの dtype（DiffSynth と同じ）。
  - AdaLN の表の添字は G−2 で上限を抑えている。
  - `validate_request` は 32 の倍数・17n+5・24fps を要求する（例外は m1 だけ）。
- **ヘッダの判定**:
  - 接頭辞は完全一致で調べるので、誤検出は起きない。2 つの接頭辞に当てはまる場合は拒否する。
  - 壊れたヘッダや巨大なヘッダは、`read_header` の上限検査と範囲検査で拒否される。
  - `comfy_quant` の入れ子 `params`・余分なキー・groupsize の省略は、テストで固定されている。
  - 非枝刈り版と LTX のファイルは、理由の文つきで拒否する。
  - torch に依存しない。アプリの venv で import しても `torch`／`numpy` が読み込まれないことを確認した。
- **読み込み**:
  - `assign=True` なので複製は作られない。辞書の山はモデルの大きさに等しい（DiT 19.53GiB・TE 24.17GiB）。
  - meta のまま残ったテンソル（非永続のバッファを含む）を検査する。不足は拒否、余分はログに出して捨てる。
  - `read_tensors` はジェネレータ内の with でファイルを閉じる。
- **block swap**:
  - DiT の `blocks` が拾われる。ブロック以外（AdaLN の表・rope・refiner・patch_proj・final_layer、計 32 テンソル）は GPU に置く。
  - `prefetch_requested` は install の前に立てている。
  - finally での片づけの順は、`teardown_prefetch` → `release_installed` → `del` → gc。
  - 先読みの arena の環（ジョブ中に使い回す GPU 上の受け皿）は decode の前の finally で返るので、`hold_arenas=True` でも decode の山には積まれない。
  - 同期の経路でも先読みの経路でも、int8 の重みとスケールは GPU に乗る。ブロックの引数はテンソルだけを位置で渡しているので、`_move_to_device` に適合する。
  - CUDA 経路の出力の位置計算は int64 である（comfy-kitchen `int8_linear.cu:529-535`）。設計書 §5 の記述と一致する。
- **ワーカーの約束事**:
  - `engine25/worker.py` と同じ形である（ready／progress／done／error、未知の op の扱い、load 失敗で exit 1、shutdown、EOF）。
  - stdout は ASCII の JSON だけで、毎回 flush する。G5 の stdout に約束事の外の行は無い（末尾のセルフテスト報告を除く）。
  - 親は ASCII の JSON を送り、`PYTHONUNBUFFERED` を設定している。
- **VAE と mp4**:
  - T'=37 で窓は 7 つ。タイルの分割は 384 → `[0,128]`（重なり 128）、1344 → 7 枚。DiffSynth の混ぜ合わせ方を忠実に写している。
  - 音声の切り揃えは `round(F×32000/24)`。
  - G5 の mp4 は BT.709／制限範囲のタグが 1／1／1／1 で、音声 5.167 秒・映像 5.1667 秒だった。
- **TE の解放**: CPU 検査で参照の取り残しが無いことを確認した。`release` は `uninstall` を呼ばない（呼ぶと全層が GPU に戻るため）。この判断は正しい。
- **ライセンス**:
  - 移植した 7 ファイル（dit・rope・scheduler・pipeline_h3・text_encoder・video_vae・audio_vae）の先頭に、由来・commit・Apache-2.0・変更点が書いてある。
  - ComfyUI のコードは写していない。
  - リポジトリ自体が Apache-2.0 である。
- **freeze と pyproject**: dependencies の 8 本は、すべて freeze と同じ版である。
- **テストの安定性**: 時間や乱数に依存するところは無い（乱数はシードを固定している）。GPU テストは `H3_GPU_TESTS=1` を付けたときだけ動く。CPU テストで `inference_mode` が書かれていないことを固定している。
