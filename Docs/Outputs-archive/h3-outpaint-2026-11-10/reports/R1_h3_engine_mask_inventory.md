# R1 MiniMax H3 エンジンのマスク部品の棚卸し（台帳 §1-125 Outpainting の下調べ）

調査日 2026-11-10・ブランチ dev・読み取りだけ（テスト・GPU は使っていない）。行番号は調査時点の作業ツリーのもの。

## 結論（先に要点）

- トークン単位のマスク（`[T', 縦, 横]`）にしたときに**式が変わる箇所は無い**。中身を変える必要があるのは**組み立て関数 `video_mask` だけ**。`dit_input`・`state_velocity`・`finish`・`masked_row_times`・`time_rows`・DiT・VRAM 16GB mode・RoPE〔位置の回転埋め込み〕は、行ごとに違う時刻をもともと受けられる作りになっている。
- 渡し口で変わるのは 3 か所。`_denoise` の `video_mask` 呼び出しの引数、単発の `generate()` が `video_fix` を渡していないこと、チェーン側の固定行の組み立てと自己検査。
- 既存に**無い**部品は 2 つ。(1) 元の映像を縮小せずキャンバスの任意の位置へ置き、余白を足す読み方。既存の読み方は 3 本とも「拡縮する」読み方。(2) 空間の固定領域を `torch.equal` で確かめる自己検査。
- 映像 VAE〔映像の圧縮器〕の符号化器は GroupNorm〔チャンネルの組ごとに平均と分散で正規化する層〕を**空間のタイル全体**でかける。そのため、余白の画素は畳み込みの受容野（片側約 220 画素）を越えて、**同じ 256 画素タイルの中の潜在すべて**に（弱く）効く。復号器は窓の中の全結合の注意〔ViT〕なので、固定した潜在を復号した画素も、同じタイルの中の生成された潜在の影響を受ける。
- マスクで増えるメモリは 1080p・345 フレームで約 4MB。快適上限を決める系列の行数は変わらない。キャンバスが大きくなること自体が行数を増やす。
- 台帳 §1-124（Inpainting）に「マスクの細粒化はここで実装し、Outpainting（§1-125）が流用する」とある（`Docs/PENDING_TASKS.md:35`）。設計書 §12 は「画角拡張」をまだ「書かないもの」に挙げている（`Docs/MINIMAX_H3_DESIGN.md:1290`）。

---

## (1) 事実

### a. `engine_h3/denoise_mask.py`

- **状態の形**: 映像の clean は `[1, C, T', H/16, W/16]`（C=24）。`video_mask` が 5 次元でバッチ 1 であることを確かめる（`engine_h3/denoise_mask.py:111-112`）。ノイズは同じ形でなければならない（`:113-114`）。音声は `[2, 32, A]`（`:86-88`）。
- **`m` のブロードキャストの形**: 映像は `values.view(1, 1, frames, 1, 1)`（`:126`）、音声は `view(1, 1, steps)`（`:93`）。データクラスの説明は `:52-53`。
- **`rows` の並び**: 映像は `values.repeat_interleave(grid)`（`:127`）で、潜在フレームごとに同じ値を `grid` 回並べる。並びは t→h→w で、`patchify_video` と同じ順（`engine_h3/dit.py:165-177`。permute `(0,2,4,6,1,3,5,7)` で t, h, w の順に並ぶ）。この対応は既存のテストが確かめている（`tests/test_h3_denoise_mask.py:64-77`。`mask.m.shape == (1,1,3,1,1)` も検査している）。
- **`video_mask` の引数**: `(clean, per_frame, *, grid, noise, inject_clean=None)`（`:97-99`）。`per_frame` は `_values` で `reshape(-1)` され、要素数が `T'` でなければ `ValueError`（`:68-77`、`:120`）。全部が 1 なら None（`:121-122`）。
- **inject の作り方**: `inject = c·clean32 + (1−c)·noise32`（`:123-125`）。c の既定は `VIDEO_INJECT_CLEAN`＝0.999（`:45`、`:115`）。連結・V2V・Retake は 1.0 を渡す（`engine_h3/chain_h3.py:54`、`:624`）。noise はジョブの初期の映像乱数そのもの（`engine_h3/pipeline_h3.py:1578`、`:1609-1611`）で、新しい乱数は引かない。
- **残りの関数はブロードキャストだけで動く**:
  - `dit_input`: `m·x + (1−m)·inject`（`:131-135`）
  - `state_velocity`: 式のとおり 1〜5 を順に計算する（`:138-151`）
  - `finish`: `torch.where(mask.zero, clean, x)`（`:154-158`。`zero` は `m == 0`、`:128`）
  - `masked_row_times`: `rows` の形を問わない 1 次元の演算（`:161-166`）

  したがって `m` が `[1,1,T',H/16,W/16]` になり、`rows` が `[T'·gh·gw]` のトークンごとの値になっても、この 4 関数の**コードは変えなくてよい**（形のブロードキャストの算術による判断。実行はしていない）。
- **粒度の正本の記述**: 「映像の 2×2 パッチ単位にするには、マスクを `[T', 縦, 横]` で持ち、パッチごとの最大値に寄せる処理と、トークン単位の時刻が要る。式は同じ」（`Docs/MINIMAX_H3_DESIGN.md:402-405`）。ここでの「縦横」は潜在（16 画素）の格子で、状態に掛ける `m` は潜在の解像度、`rows`（時刻）はトークン（32 画素）の解像度になる。

### b. `engine_h3/pipeline_h3.py` の `_denoise`

- **型と受け渡し**: `audio_fix`／`video_fix` は `(clean, 値の列)` の組（`engine_h3/pipeline_h3.py:1440`、説明 `:1443-1450`）。乱数を引いた後で、形を確かめてからマスクを組む（`:1598-1611`）。映像の呼び出しは `video_mask(clean_v.to(dev), video_fix[1], grid=gh * gw, noise=video, inject_clean=video_inject_clean)`（`:1609-1611`）。
- **単発の `generate()` は `video_fix` を渡さない**: `_denoise(..., audio_fix=audio_fix, refs=ref_rows, ...)` のみ（`:855-860`）。`video_fix` を渡しているのは連結（`engine_h3/chain_h3.py:619-627`）だけ。
- **`time_rows`**: 全行の時刻 `t [S]` を f32 で作り、映像の行を `video_t` で丸ごと置き換え、最後に条件行と参照行の時刻を入れて、`torch.unique(sorted=True, return_inverse=True)` で番号を振る（`:378-412`）。映像の `video_t` の長さは `layout.video`（`T'·gh·gw`）と一致する必要がある。いまでも `rows` は `T'·grid` 個なので、トークン単位にしても長さは同じ。
- **DiT への渡し方**: `t_unique [M]` と `time_index [S]`（`:1632-1636`、`:1652-1655`）。DiT の中で `adaln_index = time_index·3 + modality`（`engine_h3/dit.py:670-672`）になり、各ブロックが `index_select` で行ごとの変調を引く（`engine_h3/dit.py:325-346`）。最終層も `time_index[rows]`（`engine_h3/dit.py:611-614`）。dit.py の冒頭の説明は「マスクの行が自分の時刻を足すので M は 3 を超えうる。ここに M に依存するものは無い」（`engine_h3/dit.py:25-30`）。
- **0/1 の二値のマスクなら時刻の種類は増えない（算術）**: m=1 の行は流れの時刻 `t_v` のまま（`denoise_mask.py:166`）。m=0 の行は `min(1 − 0·σ, 0.999) = 0.999`（`:165`）。0.999 は条件行の `max(t_v, 0.999)`（`pipeline_h3.py:403`）と同じ値なので、`unique` の集合に新しく加わる値は、条件行が無ければ高々 1 つ（0.999）、あれば 0 個。トークン単位でも二値なら Retake と同じ数になる。0〜1 の間の値（境界をぼかすなど）を使うと、値の種類ごとに時刻が 1 つ増える。増えても、AdaLN〔時刻に応じて各層の変調を決める枝〕の射影は `[M·3, 6·5376]` f32 で、M＝100 でも 1 ブロックあたり約 39MB。行ごとの一時テンソルは `[S, hidden]` で M によらない（`engine_h3/dit.py:316-322`、`:325-337`）。
- **`video_inject_clean` の入口**: `_denoise` の引数（`:1440`）から `video_mask(inject_clean=...)` へ渡る（`:1610`）。記録には、呼び出し側が値を選んだときだけ `entry["mask"]["video"]["inject_clean"]` を書く（`:1699-1702`）。
- **VRAM 16GB mode**: `row_chunks` は行数 S だけで等分する（`engine_h3/lowvram.py:62-81`）。各塊は `adaln_index[r0:r1]` を切り出して `_modulate`／`_gate` に渡す（`:137-138`、`:178-179`、`:184-185`、`:191-192`）。最終層も `time_index[rows]` を塊ごとに切る（`:232-239`）。注意はヘッド群ごとに系列全体を見る（`:154-172`）。**「映像の行は全部同じ時刻」を前提にしたコードは無い。**
- **RoPE（`engine_h3/rope.py`）**: `build_coords` は `grid_h`・`grid_w` から座標を作り、映像の行は `frame[None]`（全フレーム共通の縦横）と、潜在フレームごとの時間（`engine_h3/rope.py:296-299`）。縦横の座標は面積で正規化している。比 `latent_dim / sqrt(lh·lw)`、`linspace(left, left+ratio, latent_dim//2, endpoint=False)·32`（`:49-69`）。トークンの間隔は算術で `64 / sqrt(lh·lw)` になり、キャンバスが大きいほど細かくなる。**コードの変更は要らない**（任意の大きさを受ける。マスクにも依存しない。設計書 `Docs/MINIMAX_H3_DESIGN.md:393`「RoPE の座標は変えない」）。ただし、元の領域のトークンの座標は、元の大きさで生成したときの座標とは違う値になる（下の (5) 推測）。

### c. `engine_h3/chain_h3.py` と読み込み

- **固定行の定型**:
  - `carry_fix`: 0 埋めの clean `[1, 24, T', lh, lw]` に頭の k 個を写し、値の列は `[0]*k + [1]*(T'−k)`（`engine_h3/chain_h3.py:129-152`）
  - `retake_fix`: 符号化した窓の潜在そのものを clean にし、値の列は「頭 0・真ん中 1・尾 0」（`:155-183`）。音声は長さ A の 0 の列に `[0, te_a)` を埋め、範囲ごとに値を決める（`:172-183`）

  どちらも CPU の f32 で、値の列は Python の list。m=1 の行の clean の値は式の中で 0 倍され、使われない（`denoise_mask.py:135`、`:150`）。
- **`_read_retake_window`**（`:309-343`）: 映像は `read_source_tail(path, width, height, window)`。音声はトラックがあれば `load_source_waveform`。s=1.0 は全長が必須、それ未満は読めた分だけ（`:330-342`）。
- **`read_source_tail`**（`engine_h3/source_tail.py:22-61`）: 全フレームに `cover_crop`（**覆うように拡縮して中央を切り抜く**）をかけ、`np.stack` で uint8 `[F, H, W, 3]` を返す（`:51-52`、`:61`）。フレーム数がちょうどでなければ止める（`:55-60`）。
- **既存の読み方に「任意の位置へ置く」「余白を足す」ものは無い**:
  - `image_cond.cover_crop`（`engine_h3/image_cond.py:54-68`）は拡縮と中央の切り抜き。コメントに「never letterboxing」とある（`:5`）。
  - `load_canvas_image` も同じ（`:29-51`）。
  - Ref2VA の `reference.load_reference_image` は切り抜きなしの拡縮だけ（`engine_h3/reference.py:121-145`、`match_size` `:77-83`）。
  - `read_reference_video` は `video_canvas` の大きさへ拡縮する（`:186-203` 付近）。

  `pad`・`paste`・`letterbox` を `engine_h3/` で探しても該当は無い（`media_out.py:162` の「no padding」だけ）。
- **`_encode_source_tail`**（`engine_h3/pipeline_h3.py:1208-1256`）:
  - 入力は CPU の uint8 `[F, H, W, 3]`（キャンバスの大きさ）。
  - 符号化器をこの段のためだけに組み、`encode_clips(frames_u8, device, tiled=True)` で 17 フレームずつ GPU へ送る（`:1227-1236`、`engine_h3/video_vae.py:615-649`）。
  - 出力の形 `(1, 24, T')` を確かめて CPU の f32 に移す（`:1229-1234`）。
  - 音声は `freeze_ka = min(round(F·5/3), N // 800)` 個を符号化する（`:1239-1247`）。
- **時間方向の因果性**: 時間方向の詰め物は前に 0 を `2·p_t` 枚だけ置き、後ろには置かない（`engine_h3/video_vae.py:179-197`）。17 フレームの塊は `_encode_clip` を塊ごとに別々に呼ぶ（`:641-646`）ので、塊をまたぐ依存は無い。GroupNorm は時間を batch に畳むので、フレームをまたがない（`:200-207`）。
- **`_decode`**（`engine_h3/pipeline_h3.py:1743-1844`）:
  - `trim_head>0` のときだけ先頭を切り、音声ハンドルを書く（`:1803-1824`）。
  - `source_wave` が与えられると、音声 VAE を組まずにその波形を mp4 に入れる（`:1785-1787`）。記録は `audio_vae_source: "skipped"`・`audio_samples_source`（`:1786`、`:1834-1835`）。
  - 単発の A2V は Strength 1.0 のときだけ `source_wave = wave`、記録は `audio_output: "original"`（`:866-871`）。
  - Retake は `muxed_original_waveform` のとき `source_wave=src_wave, trim_head=0`（`engine_h3/chain_h3.py:713-719`）。
- **自己検査の定型**:
  - 連結の頭: `torch.equal(video[:, :, :k], prev_video[:, :, -k:])`（`:660-665`）
  - Retake: `_check_retake_fixed` が頭・尾の時間の帯を `torch.equal` で比べ、外れれば `RuntimeError`（`:346-364`、呼び出し `:666-668`）

  どれも**時間の範囲（スライス）**で比べている。空間の領域を比べる形は無い。

### d. `engine_h3/video_vae.py`（符号化器と復号器）

- **空間の詰め物の種類**: `EncoderFCN3D` の既定は `padding_mode="reflect"`（`engine_h3/video_vae.py:249-251`）。`H3VideoEncoder` は引数を渡さずに組むので reflect（`:568`）。`BaseConv3d` は空間を `F.pad(..., mode=pad_mode)`（reflect）、時間を 0 で埋める（`:191-197`）。`Downsample3D` は右と下に 1 を reflect で足してから、詰め物なしでストライド 2（`:214-224`）。
- **1 トークン＝32 画素の根拠**: 空間の縮小は `space_down=(2,2,2,2,1,1)` の積で 16（`:557`、`self.vae_ratio = math.prod(space_down)` `:419`）。DiT のパッチは `(1, 2, 2)`（`engine_h3/dit.py:102-103`）。定数は `SPATIAL_FACTOR = 32 # VAE 16 x patch 2`（`engine_h3/constants.py:29`）。
- **空間タイル（256／64）**:
  - 既定は `tile_size=256, tile_overlap_min=64`（`:561`）。
  - `split_tiles` は、256 画素のタイルを最小の枚数で並べ、重なりを 16 刻みで広げて長さにぴったり合わせる（`:480-496`）。長さが 256 以下ならタイル 1 枚（`:483-484`）。
  - 符号化はタイルごとに `encoder` と `quant_conv` を通し、潜在の空間で重なり÷16 の幅を線形に混ぜる（`:580-591`、`_stitch` `:512-528`、`blend` `:498-509`）。
  - 復号は潜在をタイルに切り、画素の空間で混ぜる（`:678-692`）。
- **縦横の制約**: 符号化器が求めるのは 16 の倍数だけ（`:603-604`、`:633-634`）。DiT は潜在が 2 で割り切れること、つまり 32 の倍数（`engine_h3/dit.py:172-173`）。要求の検査も 32 の倍数だけ（`engine_h3/pipeline_h3.py:260-262`）。`split_tiles` はタイル・重なり・開始位置がすべて 16 の倍数なので、16 の倍数の長さならぴったり覆う（算術）。**64 や 128 の倍数の制約は見当たらない。**
- **符号化器の受容野（算術。推測と明記）**: カーネル 3 の畳み込みを入力側から足し上げると、受容野は全幅 **441 画素**、片側**約 220 画素**（潜在で約 13.8 個、トークンで約 6.9 個）になる。足し上げの内訳は次のとおり。
  - `conv_in` で +2
  - 各段にはブロック 2 個×畳み込み 2 枚（`:257-265`）がある。段 0〜3 は縮小（`:266-270`）の前に、そのときの刻みで 4 枚×2 を足す。段ごとの刻みは 1・2・4・8 で、縮小のたびに刻みが倍になる。
  - 段 4・5 は刻み 16 のまま縮小しない（`space_down`・`time_down` が 1。`:266`）
  - `conv_out` で +32

  ```text
  r = 1 → 3 → 11 → 13 → 29 → 33 → 65 → 73 → 137 → 153 → 281 → 409 → 441
  ```

  `Downsample3D` の非対称の詰め物で、中心は右と下へわずかにずれる。
- **ただし実効の影響範囲はタイル全体（構造の事実と算術）**: GroupNorm は `(b·t, c, h, w)` に並べ替えて、空間 h×w 全体で統計を取る（`TemporalIsolatedSpatialGroupNorm`、`:200-211`）。どの ResnetBlock にも入っている（`:232-233`、`:245-246`）。タイルで符号化するときは統計がタイルごとになる（`:584-588`）。したがって**同じタイルにある余白の画素は、そのタイルの潜在すべてに統計を通して効く**。どこまで効くかは、受容野（片側約 220 画素）ではなくタイルの範囲（256 画素）で決まり、重なりの帯の潜在は 2 枚のタイルの合併（最大でおよそ 256＋256−64＝448 画素）から影響を受けうる（算術）。どれだけ大きく効くかは測っていない。
- **復号器の受容野（構造の事実と算術）**: `ViT3DDecoder` は潜在の 1 ボクセルを 1 トークンにして、窓の中で全結合の自己注意を 36 層かける（`:341-399`）。窓は時間 7 潜在（5＋重なり 2。`:425-428`、`:719-760`）、空間はタイル 256 画素＝16 潜在、重なり 64 画素以上を画素の空間で線形に混ぜる（`:678-692`）。したがって、固定した潜在を復号した画素は、**同じ復号タイル（と時間窓）にある生成された潜在すべての影響**を受ける。境界からの距離で言えば、タイルの中でおよそ 256 画素、重なりの帯では合併の 448 画素程度まで（算術による推測）。タイルの区切りはキャンバスの大きさだけで決まり、元の領域の置き場所とは無関係（`:681-682`）。

### e. `engine_h3/worker.py`

- **ジョブの種類**: `op` は `load`／`generate`／`generate_chain`／`shutdown`（`engine_h3/worker.py:847-886`）。振り分けは `main` の `if op == "generate"`（`:867`）と `if op == "generate_chain"`（`:876`）。知らない op は `error`（`:885-886`）。
- **本文の鍵**:
  - `generate` で読む鍵は `_GENERATE_FIELDS`（`:260-264`）で、`images`・`source_audio`・`references`・`nag` などを含む。
  - `generate_chain` は `_CHAIN_FIELDS`（`:279-281`）。上の鍵から `num_frames`／`images`／`source_audio`／`references` を除き、`clips`・`source`・`retake` を足す。
  - クリップの鍵は `_CHAIN_CLIP_FIELDS`（`:285`）、`source` は `{path, context_frames}`（`:287`）、`retake` は `{path, audio_strength}`（`:289`）、`source_audio` は `{path, strength}`（`:270`）。
- **解析関数**:
  - `_source_audio`（`:431-456`）
  - `_references`（`:492-529`）
  - `_chain_source`（`:692-713`）
  - `_chain_retake`（`:715-736`）
  - `_chain_clips`（`:631-668`）
  - `_generate_request`（`:532-583`）
  - `_do_generate`（`:586-628`）
  - `_do_generate_chain`（`:747-795`。`H3ChainSpec(..., source=, retake=)` を組んで `run_chain_h3` を呼ぶ `:778`、`:793`）
- **`done` の構造**: 単発は `seed_used`〜`images_used` と `h3: {phases, quant_backends, [loras], [a2v], [references], [vsf]}`（説明 `:122-137`、組み立て `engine_h3/pipeline_h3.py:891-917`）。連結はこれに `chain` が足される（`:138-156`、組み立て `engine_h3/chain_h3.py:751-811`）。Retake は `chain.retake` に幾何の 8 鍵と実行時の 5 鍵（`:776-790`）。
- **進捗の段の名前**: `encode`／`stage1_denoise`／`decode`（`engine_h3/worker.py:107-121`）。`encode` の総数は `encode_steps`（`engine_h3/pipeline_h3.py:433-438`）で決まり、連結では入力動画（V2V・Retake の窓）の符号化もキーフレームの 1 段に数える（`engine_h3/chain_h3.py:548-552`）。`done.h3.phases` の段の名前は `encode`・`image_encode`・`source_encode`・`reference_encode`・`audio_encode`・`denoise`・`decode`。
- **新しいジョブの形を足すときに類似の入口がある場所（列挙だけ）**:
  1. `engine_h3/worker.py:260-289`: 受ける鍵の集合（`_GENERATE_FIELDS`・`_CHAIN_FIELDS`・`_RETAKE_FIELDS` など）
  2. `engine_h3/worker.py:692-736`: `_chain_source`／`_chain_retake`（入れ子の鍵の解析の型）
  3. `engine_h3/worker.py:747-795`: `_do_generate_chain`（`H3ChainSpec` を組む）。`:586-628` の `_do_generate`（単発）
  4. `engine_h3/worker.py:867-886`: op の振り分け
  5. `engine_h3/chain_h3.py:94-119`: `H3ChainRetake`・`H3ChainSpec` のデータクラス
  6. `engine_h3/chain_h3.py:235-270`: `_check_spec`（排他の検査）
  7. `engine_h3/chain_h3.py:465-470`（窓の事前読み込み）、`:587-600`（`source_encode` の段と固定行）、`:666-668`（自己検査）、`:713-719`（復号）、`:776-790`（記録）: Retake の分岐の並び
  8. `engine_h3/pipeline_h3.py:219-249`（`H3GenerateRequest`）、`:256-342`（`validate_request`）、`:727-917`（単発の `generate`）
  9. `h3_chain_math.py`（`compute_h3_chain_layout(..., retake=True)`。アプリ・mock・エンジンの共通の幾何。`h3_chain_math.py:40-51`、`:164-167`）
  10. `services/engines/h3/adapter.py:386-401`（ワーカーへの `retake` の組み立て）、`:586-607`（Retake の書き戻しと 422）

### f. 音声を全部固定したとき（Strength 1.0）の経路

- 単発の A2V: `source_wave = wave if (wave is not None and strength == 1.0)`（`engine_h3/pipeline_h3.py:866`）から `_decode(source_wave=...)` に渡り、音声 VAE を飛ばして（`:1785-1787`）`write_mp4(audio=wav)` へ進む（`:1827`）。記録は `audio_output: "original"`（`:870`）。
- 書き出しは `_write_audio` で、`audio_keep_samples` により映像の長さ `round(F·32000/24)` で切る（`engine_h3/media_out.py:161-163`、`:220-224`）。そのあと int16 にし、PyAV の AAC で**再符号化する**（`:224-238`、ストリームは `add_stream("aac", rate=sample_rate)` `:125-129`）。
- 元の波形は読み込みの時点で PyAV の `AudioResampler(rate=32000)` で 32kHz に変換されている（`engine_h3/source_audio.py:6`）。したがって「original」の音声は、元のファイルの音声のビット単位の写しではない（32kHz への変換と AAC の再符号化を経る）。
- Retake の s＝1.0 も同じ `source_wave` の経路（`engine_h3/chain_h3.py:717-718`）。
- 連結のクリップごとの A2V は、元の波形を入れない（`engine_h3/chain_h3.py:532`）。

### g. マスク経路のメモリ（算術）

例として 1920×1088・345 フレーム（快適上限表の 1080 K=1 c=345 の近く。記憶の索引から引いた値）を取る。T'＝102、潜在 68×120、トークン 34×60。

| テンソル | 形 | f32 の大きさ | Retake がすでに持っているか |
|---|---|---|---|
| clean（デバイス） | `[1,24,102,68,120]` | 約 76.2 MiB | 持っている（`pipeline_h3.py:1610` の `clean_v.to(dev)`） |
| inject（デバイス） | 同上 | 約 76.2 MiB | 持っている（`denoise_mask.py:125`。c＝1.0 でも新しいテンソル） |
| video_fix の clean（CPU） | 同上 | 約 76.2 MiB | 持っている（`chain_h3.py:170`。自己検査のため `retake_fixed` に残る `:598`） |
| m（現行） | `[1,1,102,1,1]` | 408 B | — |
| m（トークン単位・潜在の解像度） | `[1,1,102,68,120]` | 約 3.2 MiB | **新しく増える** |
| zero（bool） | 同上 | 約 0.8 MiB | 新しく増える（現行は 102 B） |
| rows | `[102·34·60]` = 208,080 | 約 0.8 MiB | 同じ大きさ（現行でも `T'·grid` 個） |
| `masked_row_times`・`time_rows` の一時テンソル | `[rows]`・`[S]` | 各 1MB 未満 | 同じ |

- 増えるのは**約 4MB**。比べると、DiT の活性化 1 本 `[S, 5376]` bf16 は約 2.25GB（S≈209,500）で、その約 0.2%。
- `dit_input`／`state_velocity` の一時テンソルは状態の形（約 76MiB）で、いまと変わらない。
- 快適上限は系列の行数で表され、マスクは行の数を変えない（`Docs/MINIMAX_H3_DESIGN.md:393`）。**マスクそのものは快適上限に影響しない**（算術）。
- 影響するのは**キャンバスが大きくなること**で、出力の大きさの T2V と同じ行数になる。単発のジョブに clean・inject（合わせて約 152MiB）が加わるのは、Retake と同じ量。
- CPU 側では、キャンバスの uint8 フレーム `[345,1088,1920,3]` が約 2.16GB で、`np.stack` で倍の山になる（Retake の既知の限界と同じ性質。`Docs/MINIMAX_H3_DESIGN.md:842`）。

### h. 既存のテスト（実行はしていない）

- **CPU でマスクと連結を見るファイル**:
  - `tests/test_h3_denoise_mask.py`（式。行の並び t→h→w `:64-77`、`time_rows` と `unique` `:204`、`inject_clean` `:238-266`）
  - `tests/test_h3_chain.py`（小さな DiT で本物の `_denoise` を回す `:480-560`、`retake_fix` `:1056-1084`、Retake の自己検査 `:1251-1297`、本物の `_denoise` での Retake `:1320`）
  - `tests/test_h3_engine.py`（小さな DiT `_tiny_denoiser` と `_clean_latents`。`tests/test_h3_chain.py:481` が import する）
  - `tests/test_h3_source_tail.py`（`cover_crop`・`read_source_tail`・`cut_head`）
  - `tests/test_h3_media.py`（映像 VAE・mp4）
  - `tests/test_h3_worker_chain.py`
  - `tests/test_h3_chain_math.py`（幾何）
- **仮想環境とコマンド**: エンジン用の `.venv-engine-h3` で `scripts/run_h3_pytest.py` を使う（`scripts/run_h3_pytest.py:1-12`。アプリの venv の site-packages を後ろに足し、`--noconftest` で回す）。例は `.venv-engine-h3\Scripts\python.exe scripts\run_h3_pytest.py tests\test_h3_chain.py`（`tests/test_h3_chain.py:3-5`）。
- 記録簿の直近の慣例は 7 ファイルを一緒に流す形（`Docs/VERIFICATION_LOG.md:22019`。`test_h3_chain`・`test_h3_worker_chain`・`test_h3_chain_math`・`test_h3_engine`・`test_h3_denoise_mask`・`test_h3_source_tail`・`test_h3_media`）。
- アプリ側の conftest が要る API のテスト（`test_h3_*_api.py`）は、このランナーでは `fixture 'three_family_client' not found` になる（`Docs/VERIFICATION_LOG.md:21002`）。これらはアプリの venv で流す（`.venv\Scripts\python.exe -m pytest`。`Docs/VERIFICATION_LOG.md:21945`）。GPU のテストは `H3_GPU_TESTS=1` のときだけ回る（`Docs/MINIMAX_H3_DESIGN.md` §8 の表）。

---

## (2) トークン単位のマスクに最小限必要な変更点（箇所と理由だけ）

1. **`engine_h3/denoise_mask.py` の `video_mask`（`:97-128`）**: 値の列に `[T']` に加えて `[T', 縦, 横]` も受け、2 つを作り分ける。
   - 状態に掛ける `m` は `[1,1,T',H/16,W/16]`。トークン単位で受けたなら縦横に 2 倍に広げる。
   - 時刻に使う `rows` はトークン単位の t→h→w。潜在の単位で受けたなら 2×2 ごとの最大値に寄せる。設計書 §3.1「粒度」の指示。
   - 理由: いまは `_values` が要素数 `T'` を強制し（`:120`）、`view(1,1,T',1,1)` と `repeat_interleave(grid)` で潜在フレーム単位に固定されている。
   - 既存の `[T']` の入力の挙動（`m` の形 `(1,1,T',1,1)`）を変えなければ、既存のテスト（`tests/test_h3_denoise_mask.py:69`）はそのまま通る。
2. **`engine_h3/pipeline_h3.py` の `_denoise` の `video_mask` の呼び出し（`:1609-1611`）**: 今は `grid=gh*gw` の積しか渡していない。縦横の格子（`gh`, `gw`）が要るなら、引数を変えるか、組み立て関数の側で clean の形から求める。形の検査（`:1607-1608`）はそのままでよい。
3. **単発の `generate()`（`engine_h3/pipeline_h3.py:727-917`）**: `video_fix` を `_denoise` に渡す経路が無い（`:855-860`）。Outpainting を単発のジョブにするなら、ここに映像の固定行の受け渡しが要る。連結の形（`run_chain_h3`）に乗せるなら、ここは変えない。どちらにするかは要エスカレーション（下）。
4. **固定行の組み立て（`engine_h3/chain_h3.py` の `retake_fix`／`carry_fix` の隣）**: 空間の値の列を組む関数が要る。
5. **自己検査（`engine_h3/chain_h3.py:346-364` の型）**: 空間の領域で `torch.equal` を取る形が要る。いまは時間のスライスだけ。
6. **説明文と記録**: `_denoise` の説明（`:1443-1445`「one mask value per latent frame」）、`denoise_mask.py` の冒頭（`:30-31`「Granularity (V1)」）、`entry["mask"]` の記録（`:1695-1698`。行数・最小・最大だけなので形は変わらない）。

変えなくてよいもの（根拠は (1) a・b）: `dit_input`・`state_velocity`・`finish`・`masked_row_times`・`time_rows`・`dit.py` の AdaLN と最終層・`lowvram.py`・`rope.py`・`H3SeqLayout`。

## (3) Outpainting のジョブがエンジンに求める部品

| 部品 | 既存で足りるか | 根拠 |
|---|---|---|
| 状態の式（入力・時刻・推定・混ぜ・更新・最後の置き換え） | 足りる | (1) a |
| 行ごとに違う時刻の DiT への受け渡し（`unique`・AdaLN の添字・最終層） | 足りる | (1) b |
| VRAM 16GB mode の行の塊 | 足りる | `lowvram.py:62-81`、`:137-192`、`:232-239` |
| RoPE（大きなキャンバス） | 足りる（コードは変えない） | `rope.py:49-69`、`:296-299` |
| トークン単位の値を受けるマスクの組み立て | **無い** | `denoise_mask.py:120`、`:126-127`（§1-124 で作る予定。`PENDING_TASKS.md:35`） |
| 元の映像を縮小せずキャンバスの任意の位置へ置き、余白を足す読み方 | **無い** | 既存は `cover_crop`（拡縮と中央の切り抜き）・`match_size`（拡縮）・`video_canvas`（拡縮）だけ（(1) c） |
| 映像のフレームを潜在へ符号化（任意の 16 の倍数・17 フレームの塊・タイル） | 足りる | `video_vae.py:615-649`、`pipeline_h3.py:1208-1256` |
| 音声の符号化と全部固定（値 0） | 足りる | `pipeline_h3.py:1174-1206`。`audio_mask` は値の列を受ける（`denoise_mask.py:80-94`） |
| 元の波形を mp4 へ入れる（音声 VAE を飛ばす） | 足りる | `pipeline_h3.py:1785-1787`、`chain_h3.py:717-718` |
| 窓まるごと 1 本の復号（切り落としなし） | 足りる | `_decode(trim_head=0)`（`chain_h3.py:716-719`） |
| 固定行の注入の係数（0.999／1.0） | 足りる（引数で選べる） | `denoise_mask.py:115-118`、`pipeline_h3.py:1440` |
| 空間の固定領域の自己検査 | **無い**（時間の型だけ） | `chain_h3.py:346-364`、`:660-665` |
| ジョブの入口（ワーカーの鍵・データクラス・検査） | **無い**（Retake の型が手本） | (1) e の列挙 |
| 記録（`done` の鍵） | **無い**（Retake の `chain.retake` が手本） | `chain_h3.py:776-790` |

## (4) 動かさないと分からない点（すべて未確認）

- トークン単位のマスクで、H3 が境界をまたいで自然につなぐか。前例は台帳 §1-124・§1-125 に「見つけていない」とある（`PENDING_TASKS.md:33`、`:41`）。
- 余白の画素の埋め方（黒・灰・緑・反射など）が元の領域の潜在をどれだけ変えるか。GroupNorm がタイル全体に効くので、元の領域の端だけでなく、同じ 256 画素タイルの中の潜在が少しずつ動く。LTX で要った緑の埋めと脱緑ブレンドの類いが要るかは、台帳が「実験で決める」としている（`PENDING_TASKS.md:41`）。
- 元の領域の潜在を「キャンバス全体を符号化して取る」場合と「元の映像だけを符号化してキャンバスの潜在へ写す」場合で、復号した元の領域の画素の差がどれだけあるか。
- 固定した潜在を復号した画素が、隣の生成領域の潜在からどれだけ影響を受けるか（復号器はタイルの中で全結合の注意）。境界から内側へ何画素まで目に見える差が出るか。
- VAE の往復による元の領域の劣化（境界から遠くても、復号した画素は元の画素と一致しない）。Retake の記録簿では、帯の画素の差を「記録だけ」取った（`Docs/MINIMAX_H3_DESIGN.md:813`）。
- 映像の固定行の係数を 0.999 にするか 1.0 にするかで、結果がどう違うか。
- 音声を全部固定（値 0、Strength 1.0 相当）にしたとき、単発の A2V で見えた「硬い動き」（`Docs/MINIMAX_H3_DESIGN.md:451`）が Outpainting でも出るか。元の領域の口は固定されるので、条件が違う。
- 大きなキャンバスでの快適上限は、同じ大きさの T2V と同じでよいか。clean と inject の約 152MiB（1080p・345 フレーム）は Retake でも同じだけ加わるが、Retake の条件での快適上限を測った記録は今回探していない。
- 元の大きさ・置く位置が 32 画素の格子に乗らないとき（例: 高さ 720 を 1088 の中央に置くと上の余白は 184 画素で、16 でも 32 でも割り切れない）の扱い。

## (5) 推測

- **RoPE の座標の変化（推測）**: 縦横の座標は面積で正規化されるので、元の領域のトークンは、元の大きさで生成したときより細かい間隔の座標になる。H3 はいろいろな解像度で学習しているので、大きなキャンバスの普通の座標として扱われ、問題にはならないと見ている（確かめていない）。
- **符号化器の影響範囲（推測）**: 畳み込みだけなら片側約 220 画素（トークン約 7 個）だが、GroupNorm の統計を通じて同じタイル全体（256 画素。重なりの帯では約 448 画素）に効く。その大きさは、タイルの中で余白が占める割合と、余白の画素の値の分布しだいと見ている。
- **復号器の影響範囲（推測）**: 窓の全結合の注意なので、固定した領域の画素も同じ復号タイルの中の生成された潜在から影響を受ける。境界の近くほど強いとは限らない（距離で減衰する仕組みは RoPE の相対位置だけ）。
- **時刻の種類（算術に基づく推測）**: 二値のマスクなら `unique` の値は Retake と同じ数で、AdaLN の費用も同じ。

## 要エスカレーション（推測で埋めずに監督へ）

1. **ジョブの形**: 単発の `generate()` に映像の固定行を通すか（`pipeline_h3.py:855-860` は今は渡さない）、連結の 1 本のクリップ（Retake と同じ `run_chain_h3`）に乗せるか。連結に乗せると `CHAIN_INJECT_CLEAN`＝1.0（裁定 84-2）が自動で付き、単発に乗せると既定の 0.999 になる。どちらの係数を使うかは裁定 84-2 の範囲の外。
2. **元の領域の潜在の作り方**: キャンバス全体（余白つき）を符号化するか、元の映像だけを符号化して写すか。前者は余白の埋め方が元の領域の潜在に効く（GroupNorm）。後者は元の大きさと置く位置が 16（トークンをそろえるなら 32）の倍数であることが要る。
3. **格子に乗らない境界の扱い**: 固定するトークンを元の領域の内側へ寄せるのか（端の部分的なトークンは生成）、元の大きさ・位置を 32 の倍数に制限するのか。
4. **文書の正本**: 設計書 §12 は「画角拡張」を「書かないもの・後回し」に挙げている（`Docs/MINIMAX_H3_DESIGN.md:1290`）。§3.1「V1 で公開するもの・しないもの」は、映像のマスクとパッチ単位のマスクを「公開しない」としている（`:443-447`）。着手するときはこの記述を更新する必要がある。また、細粒化は §1-124 が先と台帳が定めている（`PENDING_TASKS.md:35`、`:42`）。
