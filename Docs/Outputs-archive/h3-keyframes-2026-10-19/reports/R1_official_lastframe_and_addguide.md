# R1: 公式の末尾フレーム条件と ComfyUI の任意フレームのガイド（MiniMax H3）

## 要約（12 行）
1. 公式系の 3 実装（DiffSynth・diffusers・ComfyUI）とも、末尾の画像は先頭の画像と**同じ手順**で符号化する。1 枚の画像として VAE に通し、同じ正規化をかけ、同じ 0.999 の混ぜ方をし、条件行の時刻も同じ max(t, 0.999)、種別の印も映像（0）。違うのは RoPE の時間座標だけ。
2. 末尾の画像も Qwen3-VL の視覚部に**入る**。並びは「`<Picture 1>: `＋先頭の視覚ブロック、`<Picture 2>: `＋末尾の視覚ブロック、その後にプロンプト」。画像が末尾 1 枚だけのときは、その画像が `<Picture 1>` になる。
3. 系列の中の条件行は「先頭のブロック → 末尾のブロック」の順に並ぶ（どの実装も入力した順に詰める）。
4. 実装ごとの差は 3 つ。(a) 末尾の画像の画角合わせ: diffusers と ComfyUI は縦横比を保って中央を切り抜き、DiffSynth は先頭と同じく引き伸ばす。(b) 条件に混ぜるノイズ: DiffSynth と ComfyUI は全部の条件で同じノイズを使い、diffusers は条件ごとに別のノイズを引く。(c) VAE の出力: DiffSynth と ComfyUI は平均を使い、diffusers は seed 42 で標本を取って fp16 に丸める。
5. 公式のプロンプト文例集は、FL2VA のプロンプトの 1 行目に「Picture 1 は 0.00 秒、Picture 2 は S.SS 秒に合わせる」という指示文を書くよう定めている。これは利用者が書くプロンプトの本文の一部で、どの実装もコードでは自動挿入しない。
6. ComfyUI の `MiniMaxH3AddGuide` の時間座標は `cursor + 5/3 × フレーム番号`。フレーム番号は 0〜F−1 のどの整数でもよい（負の値は末尾から数える）。潜在の境界には丸めない。
7. AddGuide の画像は**条件行だけ**に入り、視覚部には入らない。`<Picture N>` も付かない。公式文書がそう明記している。
8. AddGuide に strength（強さ）の入力は無い。文書も「denoise は V2V のようにはガイドの強さを変えない」と書いている。
9. AddGuide の条件行の値は先頭／末尾と全く同じで、0.999 の混ぜ方・max(t, 0.999)・印 0。複数のガイドは、つないだ順に、I2V ノードの先頭／末尾の後ろへ足される。
10. AddGuide で frame 0 や F−1 に置くと、時間座標は I2V ノードの先頭／末尾と同じになる。ただし、画角合わせ（frame 0 でも中央を切り抜く）と、視覚部に入らない点が違う。
11. 中間キーフレームの品質を一次資料で評価した報告は見つからなかった。あるのは、作者の「frame 60 に静止画を置けばその絵を通らせる」という記述と、Comfy の公式雛形が Ref2VA の重みを使っていることと、販売業者のブログの経験則だけ。
12. 公式（MiniMax・DiffSynth・diffusers）が受け付けるのは先頭と末尾だけで、中間フレームは ComfyUI だけの拡張。

## 前提と読んだ版
- 読んだ版:
  - DiffSynth-Studio `4e34969`
  - diffusers `b800f82`
  - ComfyUI `8f04fd6`
  - Comfy-Org/docs `main`（2026-10-19 時点）
  - HF の `MiniMaxAI/MiniMax-H3` `main`
- 行番号は、上の版の生ファイル（raw）を取得して数えた。
- コードは写していない。処理の中身は言葉と短い式で書く。
- 用語:
  - **VAE**（変分オートエンコーダ）: 画像と潜在を相互に変換する部品。
  - **RoPE**（回転位置埋め込み）: 行ごとの時間と縦横の座標を注意に与える仕組み。
  - **AdaLN**（適応的レイヤー正規化）: 行ごとの時刻と種別の印で、層の出力を変調する仕組み。
  - **条件行**: 系列に置く、ノイズ除去しない行。
  - **L**: 文章の長さ（視覚ブロックを含む）。
  - **F**: 画素フレーム数。
  - **t**: その時点の映像の時刻。1 が無ノイズ側。
- 時間座標の式 `cond_t = L + 5/3·F − 5/3`（末尾）は、監督が確認済みとのことなので、検証し直していない。

## 1. 公式の FL2VA（DiffSynth）

### 1(a) VAE の符号化（事実）
- 先頭も末尾も同じ処理を通る。キーフレームの一覧を順に回し、1 枚ずつ次の処理をする（`diffsynth/pipelines/minimax_h3_audio_video.py:446-452`）。
  - 縦横を生成する画角へ LANCZOS（画像の縮小拡大の方式の一つ）で**引き伸ばす**。
  - 0〜1 の画素にする。
  - `encode_video(..., process_image=True)` に通す。
  - 2×2 のパッチに並べ替える。
- `encode_video` の中身（`diffsynth/models/minimax_h3_video_vae.py:522-537`）:
  - 画素を ImageNet の平均と標準偏差で正規化する。
  - 分布の**平均**（moments の前半）を取る。
  - 1 潜在フレームに切り詰める。
  - 潜在の平均と標準偏差で正規化する。
- 先頭と末尾で処理が分かれる箇所は無い。

### 1(b) 視覚部とプロンプトの並び（事実）
- キーフレームがあると、全部を同じ大きさに引き伸ばしてから Qwen3-VL の画像処理器に渡す（同ファイル `:304-307`）。
- 並べ方（`diffsynth/models/minimax_h3_text_encoder.py:163-169`）:
  - 画像ごとに、番号を 1 から付けた `"<Picture i>: "`（文章の印 1）と、`<|vision_start|>`＋`<|image_pad|>`×N＋`<|vision_end|>`（3 つとも映像の印 0。`:131-152`）を順に置く。
  - 最後にプロンプトを置く（印 1）。
- 順番は、利用者が渡した `keyframes` の順。公式の例は `keyframes=[first_frame, last_frame], keyframe_indices=[0, -1]`（`examples/minimax_h3/model_inference/MiniMax-H3-FL2VA.py:49`）なので、`<Picture 1>` が先頭、`<Picture 2>` が末尾になる。
- 文書は `keyframe_indices` を「0（先頭）か −1（末尾）で、keyframes と 1 対 1」と書いている（`docs/en/Model_Details/MiniMax-H3.md:115-116`）。

### 1(c) 条件行の順番（事実）
- 系列の並びは `[text | cond | audio | video | pad]`。条件行は、キーフレーム i ごとに 1 フレーム分（`frame_rows` 行）を、入力の順に詰める（`minimax_h3_audio_video.py:651-690`）。
- 条件行の値の並びも同じ順（`:464-466`、`:998-1003`）。

### 1(d) 時刻と種別の印（事実）
- 条件行の AdaLN の時刻は、先頭も末尾も一律に `max(t_video, imgvid_cond_noise_aug)`（`:1011`）。既定値は 0.999（`:38`、`:970`）。
- 種別の印は、条件行も生成する映像の行も 0（`:705`）。
- 先頭と末尾で違うのは、RoPE の時間座標だけ。先頭は `text_len`、末尾は `text_len + temporal_span − 5/3`（`:680-683`）。縦横の座標は、どちらも生成する映像と同じ格子（`:685`）。

### 1(e) ノイズの混ぜ方（事実）
- `0.999·clean + 0.001·noise`。ノイズは、`seed` から引いた潜在の、時間方向の 1 枚目だけを切り出したもの（`:459-465`）。
- **全部のキーフレームが同じノイズを共有する**（同じ `noise_rows` を全部に足している）。

### 1(f) その他の違い（事実）
- 「末尾」を表す印、別のパッチ射影、マスクは無い。条件行も生成する映像も、同じ `img_pos` の列として同じ埋め込みを通る（`:671-673`、`:1000-1003`）。
- インデックスは 0 と −1 しか受け付けない（`:444`）。

## 2. diffusers（モジュラー版）

### 2(a) VAE の符号化（事実）
- 入力は `image`（先頭）と `last_image`（末尾）。この順に一覧を作り、印 `"first"`／`"last"` を付ける（`modular_pipelines/minimax_h3/before_encoder.py:83-88, 120-124`）。
- 画角合わせが先頭と末尾で**違う**。
  - 先頭（一覧の 0 番目）は引き伸ばす。
  - 2 枚目（末尾）は縦横比を保って拡大し、中央を切り抜く（`:138-161`）。注記によると、中央の求め方は `(src − 目標) // 2` で、MiniMax の計算に合わせている。
  - 末尾 1 枚だけのときは、それが 0 番目なので引き伸ばす（推測。コードの分岐から読める）。
- 符号化の手順（`encoders.py:100-130`）は、先頭と末尾で同じ。
  - 画素を ImageNet で正規化する。
  - 分布から、**seed 42 の別の乱数で標本を取る**。
  - fp16 に丸めてから、潜在を正規化する。
  - 注記は「これが公開モデルの手順で、再現に必要」と書いている（`:106-110`）。seed 42 は `modular_pipeline.py:279-286`。
- これは DiffSynth と ComfyUI の「平均を使う」と違う。どちらが MiniMax 本家の推論コードと一致するかは未確認。

### 2(b) 視覚部とプロンプト（事実）
- 切り抜いた後の画像を、一覧の順に Qwen3-VL の画像処理器へ渡す（`encoders.py:277-280`）。
- 並べ方は DiffSynth と同じ（`:286-300`）。
  - `"<Picture {index+1}>: "`（印 1）
  - 視覚ブロック（start・pad・end の全部が印 0）
  - プロンプト（印 1）
- 視覚ブロックの印 0 は、AdaLN の変調の切り替えに使われる（注記 `:282-284`）。

### 2(c) 条件行の順番（事実）
- 印の並び（`keyframe_anchors`）の順に、1 フレームずつ詰める（`before_denoise.py:309-341`）。

### 2(d) 時刻と時間座標（事実）
- 条件行の時刻は `max(timestep, 0.999)`（`before_denoise.py:1261`、`modular_pipeline.py:288-295`）。
- 時間座標:
  - 先頭は `num_text_tokens`。
  - 末尾は「文章の長さ＋潜在フレームの幅の合計 − 5/3」（`before_denoise.py:327-339`）。
  - 合計は numpy の pairwise 加算（数を 2 つずつ足し合わせる方式）で取る。注記によると、順に足す方式とは潜在 16 フレーム以上で最下位の 1 ビットがずれるため、本家に合わせて pairwise を使っている。

### 2(e) ノイズ（事実）
- `x = 0.999·clean + 0.001·noise`（`before_denoise.py:965-970`、式は `schedulers/scheduling_minimax_h3.py:193-221`）。
- ただしノイズは、**条件ごとに 1 回ずつ**、リクエストの乱数生成器から引く。だから先頭と末尾でノイズが違う。DiffSynth・ComfyUI との差になる。

### 2(f) その他（事実）
- 末尾だけの印は無い。違うのは、時間座標と、画角合わせの方式だけ。
- 印は `"first"` と `"last"` 以外を受け付けない（`before_denoise.py:340`）。

## 3. ComfyUI `MiniMaxH3ImageToVideo`（先頭＋末尾）

### 3(a) VAE の符号化（事実）
- 先頭は `_resize(..., "disabled")` で引き伸ばし、末尾は `"center"` で縦横比を保って中央を切り抜く（`comfy_extras/nodes_minimax_h3.py:146-155`）。Comfy の文書も「first_frame は縦横比を保たずに引き伸ばす。last_frame は中央を切り抜く」と書いている（`tutorials/video/minimax/minimax-h3-native.mdx:142`）。
- 符号化は、どちらも `vae.encode`（`nodes_minimax_h3.py:160-161`）。中身は次のとおり（`comfy/ldm/minimax/vae.py:792-808`）。
  - 1 フレームなら空間の符号化器だけを通す。
  - **平均**を取る。
  - 潜在の平均と標準偏差で正規化する。

### 3(b) 視覚部とプロンプト（事実）
- 同じノードの中で、画像の一覧（先頭、末尾の順）を `clip.tokenize(prompt, images=images)` に渡す（`nodes_minimax_h3.py:157-158`）。別のノードは使わない。
- 並べ方（`comfy/text_encoders/minimax.py:193-197`。先頭の注記 `:1-19` にも図がある）:
  - 画像ごとに `"<Picture i>: "` と視覚ブロック
  - 最後にプロンプト
- 視覚ブロックは、前後の start／end も含めて印 0（`:75-82`）。
- 末尾 1 枚だけのときは、一覧に 1 枚しか無いので `<Picture 1>` になる（コードから読める）。

### 3(c)〜(e) 順番・時刻・ノイズ（事実）
- 条件行の順番は、`minimax_keyframes` の一覧の順（先頭 → 末尾）。`PackedLayout` が、文章の直後に、その順に詰める（`comfy/ldm/minimax/model.py:375-396`、`comfy/model_base.py:2234-2238`）。
- 時間座標は `cursor + 5/3 × resolved_frame_index`（`model.py:379`）。インデックスは、先頭が 0、末尾が `frame_count − 1`（`nodes_minimax_h3.py:149, 154`）。
- 時刻は、条件行の区間がまとめて `max(t_v, 0.999)`（`model.py:32, 681-684`）。印は `cond` の区間が 0（`model.py:713`）。
- ノイズは `0.999·r + 0.001·noise`。注記に「どの条件も**同じ乱数の流れを最初から引き直す**（意図的）」とあり、ノイズは全条件で同じ（`model.py:535-549`）。seed はサンプラーの seed（`model_base.py:2248`）。

### 3(f) その他（事実）
- 末尾だけの特別な扱いは無い。違うのは、時間座標と画角合わせだけ。

## 4. ComfyUI `MiniMaxH3AddGuide`（任意のフレーム）

### 4(a) 時間座標の規則（事実）
- 負の `frame_idx` は `frame_count + frame_idx` に直す（`nodes_minimax_h3.py:213`）。
- 範囲の外ならエラーにする。条件は、0 未満、または「位置＋ガイドの長さ ＞ frame_count」（`:214-218`）。
- 時間座標は `cursor + 5/3 × resolved_frame_index`（`model.py:379`）。潜在の境界への丸めや切り詰めは**無い**。
- 生成する映像の潜在 k の時間座標は「原点＋5/3×(1,4,4,4,4,…) の手前までの累積和」（`model.py:95-102`）。
  - 計算すると、潜在 1 は画素フレーム 1 の時刻、潜在 2 はフレーム 5 の時刻になる。つまり各潜在は、自分の受け持ち区間の先頭フレームの時刻に座る。
  - だから中間のガイドは、多くの場合、潜在の行の時刻と時刻の**あいだ**に座る（推測ではなく、式からの計算）。
- 時間座標の原点 `cursor` は、参照ブロックの分だけ進む。そのため、Ref2VA と一緒に使っても、ガイドは目的のフレームに対応したまま（`model.py:367-370`）。PR #15439 で matlowai が「参照があるとガイドがずれる」と指摘し（2026-08-11）、それを直した形が取り込まれている（https://github.com/Comfy-Org/ComfyUI/pull/15439#issuecomment-5250032129）。

### 4(b) 受け付けるフレーム（事実）
- 0〜F−1 の**どの整数でも**よい。格子への制限は無い（`nodes_minimax_h3.py:184-186, 213-218`）。
- 制限があるのは、短い動画をガイドにするときの**長さ**だけ。17k+5 フレームに切り詰め、5 未満なら 1 枚にする（`:201-209`）。

### 4(c) 視覚部とプロンプト（事実）
- AddGuide は、条件（conditioning）の `minimax_keyframes` に足すだけで、文章の符号化器には触れない（`nodes_minimax_h3.py:220-240`）。
- 公式文書は「Add Guide にだけつないだ画像は、決まったフレームに構図を留めるが、文章の符号化器からは見えない。プロンプトで `<Picture N>` として見せたいなら、ref_images の枠にもつなぐ」と明記している（`tutorials/video/minimax/minimax-h3-multiframe.mdx:74`）。
- つまり、**条件行だけ**の仕組み。

### 4(d) 強さ（事実）
- 入力は `positive`・`vae`・`audio_vae`・`latent`・`image`・`audio`・`frame_idx` だけで、strength は無い（`nodes_minimax_h3.py:177-187`）。
- 文書は「これは条件付けで、V2V ではない。denoise は V2V のようにはガイドの強さを変えない」と書いている（`minimax-h3-native.mdx:244`）。
- 効き方の唯一の数値は、全体共通の `visual_cond_noise_aug`（既定 0.999）だけ（`model.py:537, 679`）。

### 4(e) 時刻と印（事実）
- 先頭／末尾と同じ `cond` の区間なので、`max(t_v, 0.999)`・印 0（`model.py:681-684, 713`）。
- 音声のガイドは `cond_audio` の区間で、`max(t_a, 1.0)`・印 2。

### 4(f) 複数のガイドの順番（事実と推測）
- 事実: 既にある一覧の**後ろへ足す**（`nodes_minimax_h3.py:236-238`）。だから系列の中では「I2V ノードの先頭／末尾 → AddGuide をつないだ順」に並ぶ。
- 事実: ノイズは全条件で同じ（`model.py:540`）。
- 推測: 時間の位置は RoPE だけが運び、系列の中での位置は注意の結果に効かない（全部の行がお互いを見る注意で、行の位置による区別が無いため）。だから、並べる順番は数値の丸め程度にしか効かない見込み。未検証。

### 4(g) frame 0／F−1 に置いたとき（事実）
- 時間座標は I2V ノードと同じ。値（0.999）、時刻、印も同じ。
- 違いは 2 つ。
  - 画角合わせ: AddGuide は frame 0 でも中央を切り抜く（`nodes_minimax_h3.py:222`）。I2V ノードの先頭は引き伸ばす。
  - 視覚部: AddGuide は視覚部に入らず、`<Picture N>` も付かない。
- PR の本文は「先頭／末尾の座標は従来と同じ」と書いている（PR #15439 本文）。

## 5. 中間キーフレームの品質に関する証拠
- PR #15439 の本文（drozbay、2026-08-09）: 「モデル自体が連続した時間軸の位置でガイドを扱える」「124 フレームの 60 番目に静止画を置けば、その絵を通らせる」。ただし、評価の数値は無い。根拠も、作者の主張と添付の動画だけ（https://github.com/Comfy-Org/ComfyUI/pull/15439）。
- 同じ PR の matlowai（2026-08-11）は、位置のずれの不具合を計測した。
  - 動画の参照があると、frame 0 とガイドの差が 33.91 だった。直すと 5.47 になった。
  - これは参照と組み合わせたときの位置の話で、FL2VA だけの中間フレームの品質を評価したものではない。
  - 試したのは画像と動画の参照だけで、音声は試していないと明記している（上の URL のコメント）。
- Comfy の公式雛形「Multiframe Reference」は、**Ref2VA の重み**（`minimax_h3_ref2va_pruned_int8_convrot`）＋AddGuide を連ねる形（`minimax-h3-multiframe.mdx:7, 38`）。FL2VA の重みで中間フレームを使う公式の雛形は見つからなかった。
- 公式文書（`minimax-h3-native.mdx:242`）の例: 「先頭 22 フレームを frame 0 に置くと、両方のストリームの続きができる」「frame 60 の静止画で、その姿勢を通らせる」。どの重みで試したかは書いていない。
- 販売業者のブログ（Seedance、Emma Chen、2026-09-04。https://www.seedance.tv/blog/minimax-h3-multi-keyframe-workflow）の経験則:
  - 「両立しない 2 つの目標を近くに置くと、橋渡しの時間が足りない」
  - 「どの画像も終点になると、動作の代わりに画像の間で変形（morph）する」
  - 「ディゾルブ（溶け合うような切り替え）のように見える」
  - 重みの種類は書いていない。一次資料ではない。
- MiniMax 本家と、ComfyUI の issue 検索（AddGuide MiniMax）では、中間フレームの品質（ちらつき・流れ）の報告は**見つからなかった**。
- MiniMax が中間フレームを学習したかどうかの記述も見つからなかった。公式 README の FL2VA の説明は「画像 0〜2 枚。先頭・末尾・先頭と末尾」だけ（`MiniMaxAI/MiniMax-H3` の `README.md:74`）。

## 6. 先頭・末尾・中間の差分表

| 項目 | 先頭（実装済み） | 末尾 | 中間（ComfyUI の拡張のみ） |
|---|---|---|---|
| RoPE の時間座標 | L（事実。DiffSynth `:681`） | L＋5/3·(F−1)（事実。DiffSynth `:683`、ComfyUI `model.py:379`） | L＋5/3·n。n は 0〜F−1 の整数で丸め無し（事実。`model.py:379`、`nodes:213-218`） |
| AdaLN の時刻 | max(t, 0.999)（事実） | 同じ（事実。DiffSynth `:1011`、diffusers `:1261`、ComfyUI `:681-684`） | 同じ（事実。ComfyUI `:681-684`） |
| 種別の印 | 0（事実） | 0（事実。DiffSynth `:705`、ComfyUI `:713`） | 0（事実。ComfyUI `:713`） |
| VAE の符号化 | 1 枚・平均・ImageNet の正規化・潜在の正規化（事実） | 先頭と同じ手順（事実。3 実装）。diffusers だけは標本＋fp16 で、これは先頭も同じ（事実） | 先頭と同じ（事実。`vae.encode` を共有） |
| 画角合わせ | 引き伸ばし（事実。3 実装） | DiffSynth は引き伸ばし。diffusers と ComfyUI は中央の切り抜き（事実）。末尾 1 枚だけなら diffusers は引き伸ばし（推測） | 中央の切り抜き（事実。`nodes:222`） |
| 視覚部への入力 | 入る（事実） | 入る（事実。3 実装） | 入らない（事実。`multiframe.mdx:74`） |
| プロンプトの並べ方 | `<Picture 1>: `＋視覚ブロック＋プロンプト（事実） | `<Picture 1>: `先頭→`<Picture 2>: `末尾→プロンプト。末尾 1 枚だけなら `<Picture 1>`（事実。DiffSynth `te:163-169` ほか） | ラベル無し（事実） |
| プロンプトの本文 | 公式の文例: 「at 0.00 seconds … <Picture 1> … is fully referenced.」（事実。`VIDEO_PROMPT_WRITING_GUIDE_base_en.md:19`） | 「Picture 1 … 0.00-second mark …; Picture 2 … S.SS-second mark」（事実。同 `:25`）。末尾のみなら同 `:31` | 公式の文例は無い（事実）。Ref2VA の雛形は ref_images と併用（事実） |
| 条件ブロックの順番 | 文章の直後の 1 番目（事実） | 先頭の次（事実。3 実装とも入力の順） | 先頭／末尾の後ろに、つないだ順（事実。`nodes:236-238`）。順番の影響はほぼ無い（推測） |
| ノイズの定数 | 0.999·clean＋0.001·noise（事実） | 同じ定数（事実）。ノイズは DiffSynth と ComfyUI で先頭と共有、diffusers で別に引く（事実） | 同じ定数・同じノイズを共有（事実。`model.py:540`） |
| 強さの入力 | 無い（事実） | 無い（事実） | 無い。denoise も効かない（事実。`native.mdx:244`） |

補足（事実）: 本製品の `Nz-Videomni` の中を `fully referenced|aligns with` で検索した（`Outputs-archive` は除外）。見つかったのは `"<Picture 1>: "` のラベルだけで、上の指示文を自動で入れるコードは無い。

## 出典 URL
- DiffSynth: https://github.com/modelscope/DiffSynth-Studio/tree/4e34969
  - `diffsynth/pipelines/minimax_h3_audio_video.py`
  - `diffsynth/models/minimax_h3_text_encoder.py`
  - `diffsynth/models/minimax_h3_video_vae.py`
  - `examples/minimax_h3/model_inference/MiniMax-H3-FL2VA.py`
  - `docs/en/Model_Details/MiniMax-H3.md`
- diffusers: https://github.com/huggingface/diffusers/tree/b800f82
  - `src/diffusers/modular_pipelines/minimax_h3/` の `before_encoder.py`・`encoders.py`・`before_denoise.py`・`modular_pipeline.py`
  - `src/diffusers/schedulers/scheduling_minimax_h3.py`
- ComfyUI: https://github.com/Comfy-Org/ComfyUI/tree/8f04fd6
  - `comfy_extras/nodes_minimax_h3.py`
  - `comfy/ldm/minimax/model.py`
  - `comfy/ldm/minimax/vae.py`
  - `comfy/text_encoders/minimax.py`
  - `comfy/model_base.py`
- PR #15439: https://github.com/Comfy-Org/ComfyUI/pull/15439
- Comfy の公式文書: https://github.com/Comfy-Org/docs （`tutorials/video/minimax/minimax-h3-native.mdx`、`minimax-h3-multiframe.mdx`）
- MiniMax 公式: https://huggingface.co/MiniMaxAI/MiniMax-H3 （`README.md`、`docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`）
- Seedance のブログ（一次資料ではない）: https://www.seedance.tv/blog/minimax-h3-multi-keyframe-workflow
