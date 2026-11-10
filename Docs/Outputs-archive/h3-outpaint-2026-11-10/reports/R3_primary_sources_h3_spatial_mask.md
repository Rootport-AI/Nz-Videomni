# R3 一次資料の調査: H3 の空間マスクと Outpainting（キャンバスの拡張）

調査日: 2026-11-10。調査員: R3。リポジトリ内のファイルは読んだだけで、編集していない。
ダウンロードしたコードは作業場の `dl_*` フォルダに置いて読んだだけで、実行していない。

## 0. 結論（先に）

1. **H3 の生態系には、Outpainting の前例が 2 つある。** どちらも 2026-11-09 時点の WanGP（独自ライセンス）にある。
   - ① FL2VA の重みで、固定する行の時刻を止める方式（WanGP の呼び名は「Grouped Rows」）。本製品の方式と同じ系統である。
   - ② Alibaba PAI の Fun ControlNet の修復の枝を使う方式。
   - ①の作り方は「**元の映像だけを符号化し、潜在 0 のキャンバスへ貼る**」で、仮説 H2 の「候補」そのものである。余白の幅は 32px の倍数に丸める。
   - ただし WanGP の Outpainting は、最後に元の画素を貼り戻していない。文書は「元の領域はわずかに変わりうる」と書く。
   - 継ぎ目を測った記録や画質を比べた記録は見つからなかった。
2. **ComfyUI 本体の H3 の denoise mask は、空間方向も受ける**（一次資料で確認）。
   - 映像は 2×2 パッチ（1 トークン＝32px 四方）ごとに最大値へ寄せ、行ごとの時刻に使う。
   - vLLM-Omni（Apache-2.0）も同じ形の空間マスクを持つ。
   - LongMedia（Apache-2.0）と drozbay（GPL-3.0）には、空間の修復（inpainting）のノードがある。
   - DiffSynth の Retake は、計算の部分は空間のマスクも通せる形だが、入口は時間方向だけである。
3. **H3 の映像 VAE〔映像を潜在へ圧縮する符号化器と復号器〕の符号化器は、畳み込みと GroupNorm〔チャネルの組ごとに平均と分散で正規化する層〕でできている。** 空間の詰め物は `reflect`（鏡写し）で、タイル（256px・重なり 64px 以上）ごとに独立に符号化する。
   - 畳み込みの理論上の受容野は、潜在 1 マスの外側へ**左・上に 205px、右・下に 220px**である（推測。算術は §2.3）。
   - ただし GroupNorm は 1 枚のタイルの全画素から統計量を取る。そのため、**キャンバスごと符号化すると、詰め物はそのタイル全体の潜在に効く**（推測。コードの構造からの導出）。仮説 H2 の「境界のトークンにだけ影響する」は、この点で成り立たない可能性が高い。
4. **復号器は ViT〔画像を小片の列として扱う変換器〕型で、タイルの中は全結合の注意である。** そのため、固定した領域の復号結果は、同じタイルの中の生成した潜在からも影響を受ける。VAE はもともと非可逆でもある。
   - よって、元の画素とのビット一致は起こらない（推測。構造からの導出）。仮説 H3 を支える。
   - 画素の貼り戻しやなじませは、コミュニティの修復の流れでよく使われている。drozbay の Subject Uncrop（ぼかした縁で貼る）、ComfyUI の VACE の雛形（ImageCompositeMasked）、WanGP の Remake（少し広げてぼかした縁）である。
5. **硬い固定の落とし穴（仮説 H4）のうち、H3 で実際に報告されたもの:**
   - 固定行で継ぎ目にちらつきが出た（ttulttul。行ごとの時刻を入れる前の時期で、時間方向）。
   - マスクを渡すと 16px 周期の格子が出た（ComfyUI #15981。速度から x0 への換算の誤りで、#15988 で修正済み）。
   - 画素単位のマスクがトークンの格子とずれると乱れる（PR #15375 の説明）。
   - 修復で、カメラの動きや位置が保たれなかった（#16516 の書き込み）。
   - **空間の Outpainting の継ぎ目を名指しした報告は見つからなかった。**

---

## 1. H3 の生態系での空間マスクの前例（事実）

### 1.1 ComfyUI 本体（GPL-3.0。仕様として読んだだけで、コードは写していない）

版: commit `61e524bdba7a4e193b45c2697f1af83c1beb86d6`（2026-11-09）。ファイルは `comfy/model_base.py`・`comfy/ldm/minimax/model.py`・`comfy_extras/nodes_minimax_h3.py`。

- 【一次資料で確認】**denoise mask は空間方向も受ける。**
  - `model_base.py` の H3 の処理（2363〜2426 行付近）は、画素の格子（潜在の解像度）のマスクを受け取る。映像は DiT の 2×2 パッチごとに**最大値**へ寄せ、元の解像度へ戻す。端数は複製で埋める。
  - 値は 1/256 刻みに切り上げる。音声は潜在 1 個単位である。
  - `model.py` の `mask_row_values`（77〜86 行）が、マスクを `[T, H, W]` から 2×2 パッチ単位の行の値に変える。行の時刻は `1 − m·σ` を映像 0.999 で頭打ちにする（685〜705 行）。
  - 意味は研究ノート §16.6 の記述と同じで、空間方向のマスクも同じ式で扱う。
- 【一次資料で確認】`scale_latent_inpaint`（2402〜2426 行）は、固定の領域に `0.999·clean + 0.001·noise` を入れ直す。最大値へ寄せたトークンの格子のマスクと、元のマスクとの差の分だけ、状態の x を混ぜ戻す。
  - 意味は「2×2 パッチの中に生成の部分が 1 マスでもあれば、そのトークンは生成として扱う」になる。
- 【一次資料で確認】**PR #15375**（drozbay が作成。2026-08-07 に作られ、2026-08-18 にマージ）: https://github.com/Comfy-Org/ComfyUI/pull/15375
  - 説明文: 「Video masks snap to the 2x2 latent patch grid, audio masks to whole latent frames」。初めは 0/1 に丸めていたが、後のコミットで連続値を許した。
  - 素朴な画素単位のマスクは「will not match the timesteps assigned to the 2x2 latent grids unless by coincidence」で乱れを生むので、モデルのコードの側で寄せるように直した、とある。
  - 修復の例として drozbay のワークフロー（`droz_MiniMaxH3_PerRowMasking_Example_v2.json`）がある。**Outpainting と継ぎ目への言及は無い。**
- 【一次資料で確認】**Issue #15981**（2026-08-30 に作られ、2026-09-08 に閉じた。PR #15988 で修正）: https://github.com/Comfy-Org/ComfyUI/issues/15981
  - 症状は「passing any denoise_mask to MiniMax-H3 overlays a repeating grid on the output」。FFT〔周波数の分析〕で 16.0×16.0px の山が出た。一様なマスクでも出る。
  - poorpaper の分析は、DiT の後で、速度から x0 への換算が全体の σ を使っているのが原因、というもの。行ごとの σ（または行ごとの速度の掛け直し）が要る。
  - 本製品の §3.1 の式 3「`x0̂ = x̃ + σ·(m·v)`」は、この掛け直しを含む形である（推測。式を突き合わせた判断）。
- 【一次資料で確認】**Issue #16516**（2026-09-24 に作られ、未解決）: https://github.com/Comfy-Org/ComfyUI/issues/16516
  - drozbay の修復の例では、元の映像を別に符号化し、Set Latent Noise Mask を通している。
  - Jonseed の報告: 元の映像を Reference-to-Video の参照にもつながないと、参照への忠実さが下がる。
  - **dream-1ab（2026-09-27）: 「H3 inpainting did not preserve the camera movement, position, and motion」**。
  - 継ぎ目・画素の合成・Outpainting への言及は無い。
- 【一次資料で確認】**MiniMax H3 Fun ControlNet の適用ノード**（`MiniMaxH3FunControlNetApply`。`nodes_minimax_h3.py` の 420〜500 行と 585〜640 行）。
  - `mask`（1＝作り直す）と `source_video` を受ける。マスクの外の画素だけを残した映像を符号化し、制御の枝に渡す。
  - `inpaint_post_norm` の重みでは、穴を ImageNet の平均の色（正規化すると 0 になる中間の灰色）で埋める。コードの注釈は「holes at mid-gray instead of black」。
  - 旧版の重みでは、穴を黒（0）にする。
  - これは制御の枝への入力であり、denoise mask による固定ではない。
  - 公式文書の説明: 「regions marked `1` are regenerated … while the rest of the frame stays fixed」（https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fun-controlnet 。日付は記載なし。0.35.0 以降）。**Outpainting と画素の合成は書かれていない。**

### 1.2 Alibaba PAI の Fun ControlNet（VideoX-Fun のコードは Apache-2.0、重みは H3 Community License）

- 【一次資料で確認】モデルカード:
  - https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union （旧版）
  - https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0
  - 修復では、制御の入力を 49 チャネル（潜在＋マスクした潜在＋マスク）に広げる。
  - 制御の枝は 50 ブロックのうち 0・10・20・30・40 の 5 か所に付く。
  - 2.0 は「masked pixels are zeroed *after* the ImageNet normalization (holes sit at 0 / mid-gray)」の方式である。旧方式は「holes landed near −2 / extreme dark」で、2.0 の方式は「improves how filled regions blend with kept regions」とある。
  - **Outpainting は書かれていない。**
- 【一次資料で確認】VideoX-Fun の学習コード `scripts/minimax_h3_fun/train_control.py`（commit `e8c314fbfa494d0429bbef605c9f6112b79a7ead`、1660〜1675 行の注釈）: https://github.com/aigc-apps/VideoX-Fun
  - 原文（要旨）は「Wan 2.1's recipe: zero the masked pixels after the normalization … instead of the legacy pre-normalization zero that lands near -2 as an extreme dark signal and **contaminates the kept regions through the VAE's receptive field**」。
  - **詰め物が VAE の受容野を通して残す領域を汚す、と作者自身が書いている。** これが仮説 H2 の直接の根拠になる。
- 【一次資料で確認】学習のマスクの種類（`videox_fun/data/utils.py` の `get_random_mask`）は 10 種類ある。画面の中の長方形・全面・時間の区間・楕円・円・乱数などで、**画面の縁を帯状に開ける Outpainting 型のマスクは無い**（長方形が端に接することはある）。
- 【一次資料で確認】Issue #509（2026-08-25）: https://github.com/aigc-apps/VideoX-Fun/issues/509 。協力者の返信は「This version's inpainting isn't very good」（旧版について）。

### 1.3 WanGP（deepbeepmeep/WanGP。**WanGP Community License 2.0**。GitHub の判定は NOASSERTION で、OSI〔オープンソースの認定団体〕の承認ライセンスではない。仕様として読むだけにし、コードは写さない）

版: main `5b63ae9d4c0e8578929c95c4c5e09162f4520d93`（2026-11-09）。ファイルは `models/minimax_h3/minimax_h3_handler.py`・`pipeline.py`・`constants.py`。

- 【一次資料で確認】**H3 の Outpainting（「Spatial Outpainting」）がある。** 経路は 2 つ。
  - (a) **Fun ControlNet の経路**: 元の長方形を見える文脈として渡し、新しい縁を自動でマスクして生成する。説明文は「The original area can still change slightly」「the original rectangle is guided, not copied pixel for pixel」（handler 261〜265・286 行）。
  - (b) **FL2VA の経路**: 「MiniMax H3 outpainting requires Mask Denoising Mode to be set to Grouped Rows」（handler 1251 行）。
    - Grouped Rows の説明は「conditioning timestep for fixed rows; denoising timestep for editable rows」で、既定値である（constants 20 行）。本製品の固定行と同じ系統である（推測。説明文とコードからの判断）。
    - もう 1 つの選択肢 Shared Timestep は「同じ時刻で固定と生成の行を進める」方式である。
- 【一次資料で確認】**元の映像の符号化**（pipeline 270〜283 行の `_encode_video_source`）: Outpainting のときは、**元の長方形だけを VAE で符号化し、ゼロで埋めたキャンバスの潜在に、(上端÷16, 左端÷16) の位置で貼る。** 仮説 H2 の「候補」そのものである。
- 【一次資料で確認】余白は 32px の倍数に丸める（`quantize_margins=32`。pipeline 256 行・handler 1055 行）。マスクは 2×2 パッチ単位の最大値に寄せる（`_snap_video_mask_to_patch_cells`、219〜222 行）。
- 【一次資料で確認】固定する行の入れ直しは、元の潜在とノイズを `1 − 0.999`（＝0.001）の重みで混ぜる（`_reinject_video_source` と 1575〜1577 行）。ComfyUI と同じ値である。
- 【一次資料で確認】**画素の貼り戻し**: `_keep_source_outside_mask`（305〜325 行）が、マスクを短辺の 1.5% だけ広げてから 2 回ぼかし、元の画素を戻す。注釈は「the rest of the picture stays exact」。
  - ただし呼ぶのは Remake のモード（1957 行）だけで、**Outpainting の経路では呼んでいない**（この版では）。
  - 別に、与えたキーフレームの画素を復号の後に戻す処理がある（1953 行。注釈は「restore exact supplied pixels after the VAE round trip」）。VAE を往復すると画素が一致しないことを、作者は前提にしている。
- 【一次資料で確認】constants 13〜15 行の注釈: 潜在の受け渡しの実験は「worsened the first outpainting join」なので、既定では切っている（時間方向の継ぎ目の話）。
- 【一次資料で確認】説明文（handler 129 行）: 「Use a lower masking strength (<0.75) to facilitate continuity with masked areas」。固定を緩めると、マスクの内外がつながりやすい、という作者の助言である。計測の記録は無い。
- 継ぎ目・色ずれ・画質の比較の記録は**見つけられなかった**。探した範囲は README とこの 3 ファイル。

### 1.4 その他のコミュニティ実装

- 【一次資料で確認】**vLLM-Omni**（Apache-2.0。commit `813618b5`。`vllm_omni/diffusion/models/minimax_h3/latent_mask.py`、PR #7465 は 2026-09-20 にマージ）: https://github.com/vllm-project/vllm-omni/pull/7465
  - 潜在の格子の大きさの映像マスク `[T', H, W]` を受ける。DiT 用には 2×2 の最大値で切り上げ、復元用には**マスの単位の精度**を保つ。最後に x0 をマスの単位で clean と混ぜる。
  - 意味は ComfyUI に合わせている（0＝残す・1＝作り直す・端数は連続に混ぜる）。
  - 説明文は、画質は自動検査の範囲外と書く。**Outpainting と継ぎ目への言及は無い。**
- 【一次資料で確認】**LongMedia**（Apache-2.0。`409e4cb0`。`nodes.py` 18653 行の `MiniMaxH3LatentLabVideoInpaint`、`latent_ops.py` 430〜470 行）:
  - ComfyUI の MASK を三線形補間で潜在の格子に写し、内側と外側の強さを別々に与えて `noise_mask` にする。
  - `_crop_pad_stream`（277〜300 行）は、ある潜在を、0 で埋めた別の大きさの潜在へ**中央合わせで貼る**。用途は AV〔映像と音声〕の合成で、Outpainting という名前は付いていない。
  - 文書に修復や Outpainting の記述は見つからなかった。
- 【一次資料で確認】**drozbay/MaskVidExperiments**（GPL-3.0。`c32ed8c1`）: https://github.com/drozbay/MaskVidExperiments
  - 「Mask To Latent Space」は、H3 の 2×2 トークンの格子とフレームの周期に合わせて、画素のマスクを潜在に寄せる。
  - 「Subject Uncrop」は、処理した切り抜きを**ぼかした縁で元の画面に貼り戻す**（「pixel-exact when the crop size was not changed」）。
  - README は、継ぎ目を見せないための貼り戻しを主題にしている。Outpainting は無い。
- 【一次資料で確認】**DiffSynth-Studio**（Apache-2.0。`fdad57bd`、2026-11-09。`diffsynth/pipelines/minimax_h3_audio_video.py`）:
  - Retake のマスクは潜在フレームの単位で作り、空間の全面へ広げる（366〜394 行）。
  - モデルの関数（986〜1017 行）は、マスクをパッチに並べて先頭の要素を取る。0 の行の時刻を 1.0 にする。
  - 計算の部分は空間のマスクも通せる形だが、2×2 の最大値ではなく左上のマスを取る（推測。`patchify_video(mask)[:, 0]` の読み）。**入口は時間方向だけ。**
  - 新しく `minimax_h3_controlnet.py`（Fun ControlNet の受け口。`control_in_dim=49`）が入った。
- 【一次資料で確認】**MiniMax 公式**（GitHub の `MiniMax-AI/MiniMax-H3`、`199b7d0e`、2026-10-28。コードの SPDX は Apache-2.0、重みは H3 Community License）:
  - README と Ref2VA のプロンプト指針（`.claude/skills/h3-prompt-writing/references/ref-en.txt`）に「video editing」「video continuation」の作業の種類があるが、参照の動画の言葉による編集である。
  - **inpainting・outpainting・空間マスク・キャンバスの拡張の記述は見つけられなかった。**
- 【未確認】H3 Continuum（MIT）・SGLang・Hugging Face の公式モデルカードの空間マスクは、今回は調べ直していない。研究ノート §16 の範囲では、Continuum は時間方向の固定だけである。
- 【一次資料で確認】PR jmanhype/wangp-dspy #201（2026-09-26）: https://github.com/jmanhype/wangp-dspy/pull/201
  - その時点の WanGP の H3 の処理で、Outpainting の項目が注釈で外されていた、という記録だけがある。
  - 11-09 の WanGP では有効になっている（1.3）ので、9 月末から 11 月の間に入ったとみられる（推測）。

---

## 2. H3 の映像 VAE の構造

出典（いずれも一次資料で確認）:

- 公式 `FL2VA/video_vae/source/config.json`・`config.json`・`vae_cnn.py`・`norm.py`・`klvae.py`（`MiniMax-AI/MiniMax-H3` の `199b7d0e`。コードは Apache-2.0）。
- DiffSynth `diffsynth/models/minimax_h3_video_vae.py`（`fdad57bd`。Apache-2.0）。
- 本製品の `engine_h3/video_vae.py`（読んだだけ）。

### 2.1 事実

| 項目 | 値 | 出典 |
|---|---|---|
| 空間の圧縮 | 16 倍（`space_down = [2,2,2,2,1,1]`、`vae_ratio 16`） | 公式 config |
| 時間の圧縮 | 4 倍（`time_down = [1,2,2,1,1,1]`）。17 フレームの塊ごとに符号化して潜在 5 個、末尾の端数 5 フレームは潜在 2 個（`vae_clip_length 17`・`vae_token_drop 3`） | 公式 config・研究ノート §16.2 |
| 潜在のチャネル | 24 | 公式 config |
| DiT のパッチ | [1,2,2]（1 トークン＝潜在 2×2＝32px 四方） | 研究ノート §13.2 |
| 符号化器 | 3D 畳み込みの FCN〔全部が畳み込みの網〕。`ch 128`・`ch_mult [1,2,2,4,4,8]`・各段に ResNet ブロック 2 個（3×3×3 の畳み込み 2 本）。入口と出口の畳み込みも 3×3×3 | 公式 config・`vae_cnn.py` |
| 空間の詰め物 | **`reflect`（鏡写し）**。ダウンサンプル（3×3・ストライド 2）は**右と下にだけ 1 画素**を鏡写しで足す（非対称） | `source/config.json` の `padding_mode`、`vae_cnn.py` 34〜80 行 |
| 正規化 | **GroupNorm 32 組。時間方向は混ぜず、1 フレーム（潜在の時刻）ごとに空間の全画素で統計量を取る**（`use_t_isolated_gn: true`） | `norm.py` 187〜253 行、DiffSynth 76〜86 行 |
| 時間の因果性 | 符号化器は因果（`causal_encoder: true`。過去の側だけを 0 で埋める）。17 フレームの塊の中に閉じる | config・研究ノート §16.2 |
| 復号器 | **ViT 型**（`use_vit_decoder`・36 層・32 ヘッド×64・RoPE〔位置の回転埋め込み〕）。`causal_decoder: false`。タイルの中では全部のトークンが互いに注意する。登録トークン 4 個 | config・DiffSynth `ViT3DDecoder` |
| 入力の正規化 | ImageNet の平均と標準偏差（`pixel_norm_type: imagenet`）。**ImageNet の平均の色の画素は 0 になる** | config・DiffSynth `VAEProcessor` |
| タイル | 符号化・復号とも空間のタイル（`vae_tile_size 256`・`vae_tile_overlap_min 64`、`vae_encoder_tiling 1`・`vae_decoder_tiling 1`）。タイルは独立に処理し、重なりを線形に混ぜる（符号化は潜在で 4 マス以上、復号は画素で 64px 以上）。時間方向の復号は 7 潜在の窓 | 公式 config・`klvae.py` 192〜250 行。本製品も同じ（`video_vae.py` 52・498〜543 行） |

### 2.2 タイルの区切りの規則（事実）

- 長さ L が 256 以下ならタイルは 1 枚。
- それより長ければ、重なりを 64px 以上にしたうえで、必要な最少の枚数に区切る。余りは 16px 刻みで、重なりに順番に足す（`split_tiles`）。
- したがって**タイルの区切りはキャンバスの大きさで決まり、元の映像だけを符号化したときとキャンバスごと符号化したときでは区切りが違う。**

### 2.3 空間の受容野の見積もり（推測。算術の過程を示す）

最後の潜在 1 マス（番号 n、画素 [16n, 16n+15] を受け持つ）から入力へ逆にたどる。3×3 の畳み込みは片側 1 マスずつ広げ、ストライド 2 のダウンサンプルは「出力 i ← 入力 [2i, 2i+2]」（右と下にだけ詰め物があるため）。各段の数字は、その解像度での区間である。

- 1/16 解像度: `conv_out` で [n−1, n+1] → 段 5 の 4 本で [n−5, n+5] → 段 4 の 4 本で [n−9, n+9]
- ダウンサンプル 3 → 1/8 解像度で [2n−18, 2n+20]。段 3 の 4 本で [2n−22, 2n+24]
- ダウンサンプル 2 → 1/4 解像度で [4n−44, 4n+50]。段 2 で [4n−48, 4n+54]
- ダウンサンプル 1 → 1/2 解像度で [8n−96, 8n+110]。段 1 で [8n−100, 8n+114]
- ダウンサンプル 0 → 画素で [16n−200, 16n+230]。段 0 の 4 本で [16n−204, 16n+234]。`conv_in` で **[16n−205, 16n+235]**

**理論上の受容野は幅 441px で、受け持ちのマスの外へ左・上に 205px、右・下に 220px**（潜在でおよそ 13 マス、トークンでおよそ 6〜7 個）。

注意が 4 つある。

1. 理論上の上限である。実際に効く範囲は中心に寄った釣鐘型で、ずっと狭いのが普通である。
2. タイル（256px）は受容野（441px）より小さい。そのため畳み込みの受容野は実際には**タイルの中で切れ**、タイルの端では鏡写しの詰め物がその先の代わりになる。
3. **GroupNorm は、1 枚のタイル・1 つの潜在の時刻の全画素から平均と分散を取る。** タイルに詰め物が入ると、そのタイルの全部の潜在が（大きさは別として）変わる。符号化器の各 ResNet ブロックに GroupNorm が 2 個あり、出口にも 1 個ある。
4. タイルの重なりの線形の混ぜ合わせ（潜在で 4 マス以上）により、隣のタイルの影響が混ぜ合わせの幅だけ入る。

時間方向は因果で、17 フレームの塊の中に閉じる（研究ノート §16.2）。空間の Outpainting では、全フレームで同じ位置に詰め物が入るので、時間方向の受容野は問題にならない（推測）。

### 2.4 復号の受容野（推測。構造からの導出）

- 復号器はタイル（256px＝潜在 16 マス。重なりは 64px 以上）と 7 潜在の窓ごとに、全結合の注意で復号する。そのため、ある画素の値は**同じタイル・同じ窓のすべての潜在**に依存しうる。
- 固定した領域（潜在は clean とビット一致）であっても、同じタイルに生成した潜在があれば復号結果は変わりうる。そもそも VAE は非可逆なので、clean の潜在だけでも元の画素には戻らない。
- 参考（一次資料で確認）: ComfyUI issue #15548「MiniMax H3 i2v output is a grid of independently-decoded square tiles」（2026-08-12）。タイルの独立な復号が見た目に出た事例である。中身は今回は読んでいない。

---

## 3. 他の動画 DiT での潜在マスクの Outpainting の実例

| 実装 | 広げた領域の詰め物（画素） | 広げた領域の潜在 | 固定の方式 | 最後の画素の貼り戻し | 出典・印 |
|---|---|---|---|---|---|
| **Wan VACE**（Apache-2.0。`ali-vilab/VACE` の `48eb44f1`） | **灰色**（`video_outpainting_anno.MASK_COLOR = 'gray'`。`vace/configs/video_preproccess.py` 195〜204 行）。符号化の前に、マスクの内側を 0（[−1,1] の範囲での 0＝中間の灰色）にした「inactive」と、外側を 0 にした「reactive」を**別々に符号化**する（`wan_vace.py` 137〜142 行） | 全面をノイズから生成（固定しない） | 固定ではなく、文脈の枝で導く（元の領域も作り直す） | 生成の関数には無い（`wan_vace.py` の `decode_latent` まで）。`MASK_BLUR` を使うと、マスクを元の領域へ 2×blur だけ食い込ませる（`outpainting.py` 71〜79 行。既定は 0） | 一次資料で確認 |
| ComfyUI の Wan2.1 VACE 修復の雛形 | — | — | — | **ImageCompositeMasked で元の映像へ合成する**（雛形の説明「crucial for blending the inpainted regions with the original video」） | 二次資料（https://comfy.org/hub/workflows/wan2-1-vace-inpainting-a1bb55d904c9/ ） |
| H3 Fun ControlNet（VideoX-Fun・ComfyUI） | **ImageNet の平均の色**（正規化後に 0）。2.0 で黒から変えた | 全面を生成 | 制御の枝で導く | 書かれていない | 一次資料で確認（1.1・1.2） |
| **WanGP の H3 FL2VA の Outpainting** | 使わない（元を単独で符号化する） | **潜在 0 のキャンバスに元の潜在を貼る** | 固定する行の時刻を止める（Grouped Rows） | Outpainting では無い（Remake のモードには、縁をぼかした貼り戻しがある） | 一次資料で確認（1.3） |
| LTX の潜在による条件付け | 本プロジェクトの既知事項（IC-LoRA・緑のキャンバス・脱緑ブレンド）。今回は調べ直していない | — | — | — | 未確認（今回） |
| HunyuanVideo・CogVideoX | 公式の Outpainting は見つけられなかった。CogVideoX-Fun（VideoX-Fun）の修復の旧方式は「正規化の前に 0（黒）」で、VideoX-Fun 自身が Wan 2.1 の方式（正規化の後に 0）へ移した（1.2 の注釈） | — | — | — | 一次資料で確認（VideoX-Fun の注釈の範囲だけ） |

- **「元を単独で符号化して潜在を貼る」やり方**: WanGP の H3 の実装（1.3）と、LongMedia の中央合わせの貼り付け（1.4。用途は Outpainting ではない）で見つけた。広く知られた慣行と言える証拠は**見つけられなかった**。VACE・Fun ControlNet は「キャンバスに置いて灰色で埋め、まとめて符号化する」方式である。
- **画像の拡散での既知の事実**（一次資料で確認）:
  - Blended Latent Diffusion（Avrahami ほか、arXiv 2206.02779）は「the inherent inability of this LDM to accurately reconstruct images」に対して、最適化による解決を提案している。潜在で背景を固定しても、VAE を通すと背景は元に戻らない、という問題意識である。
  - RePaint（Lugmayr ほか、arXiv 2201.09865）の再サンプリングの節は、既知の領域を毎ステップ差し替える方式は「is performed without considering the generated parts of the image」であり、「cannot correct mistakes that lead to disharmonious boundaries in the subsequent steps」と書く。

---

## 4. 硬いトークン固定の落とし穴（事実）

| 報告 | 方向 | 内容 | 状態 | 印 |
|---|---|---|---|---|
| ComfyUI #15981 | 空間・時間の両方 | マスクを渡すと 16px の格子が出る。原因は速度から x0 への換算の σ | #15988 で修正済み | 一次資料で確認 |
| ComfyUI PR #15375 の説明 | 空間 | 画素単位のマスクが 2×2 トークンの時刻とずれると乱れる。モデルの側で寄せて解決 | 解決 | 一次資料で確認 |
| ComfyUI #16516 | 空間（修復） | 修復で、カメラの動き・位置・運動が保たれない（dream-1ab）。参照につながないと忠実さが下がる | 未解決 | 一次資料で確認 |
| ttulttul（MIT。研究ノート §16.12） | 時間 | 潜在を硬くつなぐと継ぎ目でちらつく。線形の混ぜ合わせでは「visual confetti」が残る。行ごとの時刻を入れる前の時期 | — | 研究ノートの一次資料の記録 |
| H3 Continuum #13（研究ノート §16.3） | 時間 | クリップを重ねると焼ける。Masked 方式でも 60 秒で劣化 | 未解決 | 同上 |
| WanGP の説明文 | 空間 | 「lower masking strength (<0.75) to facilitate continuity with masked areas」。潜在の受け渡しは「worsened the first outpainting join」 | 助言だけで計測なし | 一次資料で確認 |
| VideoX-Fun #509 | 空間 | 旧版の修復は「isn't very good」 | 2.0 で改良 | 一次資料で確認 |
| 本製品の A2V の目視（設計正本 §3.1） | 音声の全固定 | Strength 1.0 で口の動きが硬い。仮説は「厳密な固定が残りの創造性を引き下げる」「片方の流れだけが先に完成している」 | 確かめていない | 本製品の記録 |
| RePaint | 画像 | 差し替えだけでは境界が調和しない | — | 一次資料で確認 |

- **H3 の空間の Outpainting で、継ぎ目・内容の不一致・時刻が混ざることによる劣化を名指しした報告は見つけられなかった。**
- 探した範囲:
  - GitHub の issue の検索: `repo:Comfy-Org/ComfyUI minimax mask`・`H3 inpaint`・`minimax h3 outpaint`・`minimax h3 inpaint`
  - ComfyUI の H3 の PR と issue の上記
  - VideoX-Fun・WanGP・LongMedia・vLLM-Omni・drozbay・DiffSynth のコードと文書
  - ウェブ検索（「MiniMax H3 outpainting」ほか）
- Reddit・Discord・CivitAI の記事は調べていない。

---

## 5. 推測（事実からの導出。確かめていない）

1. **生成の領域（m＝1）の clean の値は、本製品の式では使われない。** 設計正本 §3.1 の式 1・4 で m＝1 なら `x̃ = x`・`x0_b = x0̂` になり、ループの後の置き換えも m＝0 だけである。
   - したがって、広げた領域の潜在に何を入れても（0 でも乱数でも）、DiT の計算は変わらない。
   - 詰め物の選び方が効くのは、**キャンバスごと符号化するときに、符号化器を通して固定の領域の clean を変える経路だけ**である。
   - 「元を単独で符号化して貼る」なら、詰め物の問題そのものが消える。
2. **キャンバスごと符号化すると、詰め物の影響は「境界のトークン」に限らず、詰め物を含むタイルの全部の潜在に及びうる**（GroupNorm の統計量のため）。
   - 大きさは、詰め物が ImageNet の平均の色（正規化後に 0）のときが最も小さいと見込まれる。VideoX-Fun が黒から灰色へ移した理由と同じ。
   - 境界の近く（最大で約 13 潜在マス、タイルの中に限る）は、畳み込みを通してさらに強く変わる。
   - 単独で符号化する方式では、元の映像の端は鏡写しの詰め物で符号化される。そのため、端のトークンは「元の映像の外側が鏡写しで続く」と読んだ潜在になる。生成の側との意味のずれは、端の 1 トークン列あたりに出るかもしれない。
3. **単独で符号化すると、タイルの区切りが元の映像の大きさで決まる。** そのため、同じ元の映像でも、キャンバスごと符号化したときとは内部の潜在も少し違う。どちらが「正しい」というものではなく、単独の符号化は元の映像だけから決まる、という利点がある。
4. **位置合わせ**: 貼る位置は 16px（潜在 1 マス）の倍数であれば潜在に乗る。トークンの境目に揃えるには **32px の倍数**が要る（WanGP は 32 に丸める）。
   - 32 の倍数でないと、境目のトークンは最大値に寄せられて生成の側になる。そのトークンの中の元の画素の部分も作り直される。
   - 本製品のキャンバスはもともと 32 の倍数（設計正本 §3）なので、余白も 32 の倍数にすれば元の映像も 32 の倍数になる。
5. **貼り戻しは要る**（仮説 H3）。VAE は非可逆で、復号器はタイルの中で全結合なので、固定の領域も画素は変わる。
   - なじませの要否: 貼り戻すと、境目の内側は元の画素、外側は生成の画素になり、色や明るさの差が線として出うる。
   - 前例はどれもぼかした縁を使う。drozbay はぼかした縁、WanGP の Remake は「1.5% 広げて 2 回ぼかす」、VACE は `MASK_BLUR` で元の領域へ食い込ませる。
   - 幅の決め手は見つからなかった。

---

## 6. 仮説ごとの整理

### H2（固定の領域の clean は元の符号化。詰め物は受容野を通して境界のトークンにだけ効く。単独で符号化して貼るのも候補）

- **賛成**
  - 詰め物が VAE の受容野を通して残す領域を汚す、という作者の注釈がある（VideoX-Fun。1.2）。
  - 「単独で符号化して潜在 0 のキャンバスに貼る」は WanGP の H3 FL2VA の Outpainting の実装そのものである（1.3）。
  - 符号化器は畳み込みなので、空間の影響は有限である（2.3）。
- **反対（部分的）**
  - 「境界のトークンにだけ」は成り立たない可能性が高い。GroupNorm がタイル全体から統計量を取り、タイルは独立に符号化される（2.1・2.3。推測）。
  - 畳み込みの理論上の届く範囲も約 13 潜在マスあり、「境界の 1 列」よりずっと広い（推測）。
- **証拠なし**
  - 詰め物の種類ごとの差を測った記録は見つからなかった。
  - 単独の符号化とキャンバスごとの符号化を比べた記録も見つからなかった。
  - 潜在の空間で 0 を入れるか乱数を入れるかは、本製品の式では結果に効かない（5.1。推測）。

### H3（復号の受容野により固定の領域はビット一致しない。貼り戻しが要る。なじませの要否は未知）

- **賛成**
  - 復号器は ViT 型で、タイルの中は全結合である（2.1）。
  - VAE を往復すると画素が一致しないので、与えた画素を復号の後に戻す処理が WanGP にある（1.3）。
  - 背景は元に戻らないと Blended Latent Diffusion が書く（3）。
  - 前例はどれも、ぼかした縁で貼り戻している（drozbay・ComfyUI の VACE の雛形・WanGP の Remake）。
- **反対**
  - WanGP の Outpainting は貼り戻しをしていない。元の領域が少し変わることを「guided, not copied」として受け入れている（1.3）。
  - 貼り戻しは必須ではなく、設計の選択であることを示す。
- **証拠なし**
  - なじませの幅や方式を、H3 で比べた記録は見つからなかった。

### H4（硬い固定の継ぎ目・元の内容の無視などの報告）

- **賛成（問題があるという報告）**
  - 修復でカメラの動きや位置が保たれない（#16516）。
  - 固定を緩めると連続性が上がるという作者の助言がある（WanGP）。
  - 時間方向では、ちらつきや焼けの報告がある（ttulttul・Continuum #13）。
  - 換算の誤りによる格子は修正済みである（#15981）。
  - 一般の拡散でも、境界が調和しない問題が知られている（RePaint）。
  - 本製品の A2V にも、固定 1.0 で硬くなるという観察がある。
- **反対（硬い固定で実用になっているという材料）**
  - ComfyUI 本体・vLLM-Omni・LongMedia・drozbay・WanGP が、空間のマスクで行の時刻を止める方式を採用している。WanGP は、これを既定（Grouped Rows）にし、Outpainting の必須条件にしている。
  - LongMedia は、時間方向で、キーフレームより固定行のほうがよかったとして移った（研究ノート §16.4）。
- **証拠なし**
  - 空間の Outpainting での継ぎ目の報告は見つからなかった（探した範囲は §4）。

---

## 7. 出典一覧（取得はすべて 2026-11-10）

| 出典 | 版 | ライセンス |
|---|---|---|
| Comfy-Org/ComfyUI `comfy/model_base.py`・`comfy/ldm/minimax/model.py`・`comfy_extras/nodes_minimax_h3.py` | `61e524bdba7a4e193b45c2697f1af83c1beb86d6`（2026-11-09） | GPL-3.0（仕様として読んだだけ） |
| ComfyUI PR #15375・issue #15981・#16516・#15548 の題名 | 2026-08〜09 | 同上（issue の本文） |
| docs.comfy.org の minimax-h3-fun-controlnet | 日付の記載なし | — |
| MiniMax-AI/MiniMax-H3（GitHub）`FL2VA/video_vae/*`・README・`ref-en.txt` | `199b7d0e1fd95e7b3eafa8c15eca692534142427`（2026-10-28） | コードは Apache-2.0（SPDX）、重みは H3 Community License |
| modelscope/DiffSynth-Studio `minimax_h3_video_vae.py`・`minimax_h3_controlnet.py`・`pipelines/minimax_h3_audio_video.py` | `fdad57bd94d3fbac8a72c44920dd0a23b719ebb9`（2026-11-09） | Apache-2.0 |
| aigc-apps/VideoX-Fun `scripts/minimax_h3_fun/train_control.py`・`examples/minimax_h3_fun/predict_v2v_control_inpaint.py`・`videox_fun/data/utils.py`、issue #509 | `e8c314fbfa494d0429bbef605c9f6112b79a7ead` | Apache-2.0 |
| HF alibaba-pai/MiniMax-H3-Fun-Controlnet-Union・-2.0 | 日付の記載なし | H3 Community License |
| deepbeepmeep/WanGP `models/minimax_h3/{minimax_h3_handler,pipeline,constants}.py`・README | main `5b63ae9d4c0e8578929c95c4c5e09162f4520d93`（2026-11-09） | WanGP Community License 2.0（OSI 外。仕様として読んだだけ） |
| vllm-project/vllm-omni `latent_mask.py`・PR #7465 | `813618b5a22b9106d324aed5057fa14c295eddc6`。PR は 2026-09-20 にマージ | Apache-2.0 |
| vizart-vj/ComfyUI-MiniMax-H3-LongMedia `nodes.py`・`latent_ops.py`・文書 | `409e4cb03c85cf455bf195d5185d106c0d821487`（2026-09-29） | Apache-2.0 |
| drozbay/MaskVidExperiments README | `c32ed8c17e6fe892a174ebf25fe98b1317de97fb` | GPL-3.0 |
| ali-vilab/VACE `vace/annotators/outpainting.py`・`vace/models/wan/wan_vace.py`・`vace/configs/video_preproccess.py` | `48eb44f1c4be87cc65a98bff985a26976841e9f3` | Apache-2.0 |
| jmanhype/wangp-dspy PR #201 | 2026-09-26 | 不明 |
| Blended Latent Diffusion arXiv 2206.02779・RePaint arXiv 2201.09865 | — | 論文 |
| 本製品 `engine_h3/video_vae.py`、Docs の `MINIMAX_H3_DESIGN.md` §3・§3.1、`MINIMAX_H3_RESEARCH_NOTES.md` §13・§15・§16 | 2026-11-10 の作業ツリー | — |
