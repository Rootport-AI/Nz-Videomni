# R4 一次資料の調査: Outpainting で元の映像の縁に出る細い線（先行事例と原因の候補）

調査日: 2026-11-10。調査員: R4。リポジトリ内のファイルは読んだだけで、編集していない。
ダウンロードしたコードは作業場の `dl_*` フォルダに置いて読んだだけで、実行していない。
既存の調査（研究ノート §17・R3 報告）と重なる事実は、ここでは繰り返さず「§17.x」で参照する。

用語:
- **固定トークン**: マスク値 m=0 で、毎ステップ元の潜在を注入し、行の時刻を 0.999 に止めるトークン（1 トークン＝潜在 2×2＝32 画素四方）。
- **縁の潜在行**: 元の映像の長方形のうち、広げた領域（帯）に面した最も外側の潜在 1 マス（16 画素）の行または列。
- **貼り戻し**: 復号した画素の上に元の画素を戻す処理。

---

## 0. 結論

1. **同じ症状の報告は 1 件見つかった。しかも原因の特定と対策まで書かれている。** TwoAbove/ComfyUI-H3VideoOutpaint（MIT。H3 の FL2VA で、元の映像だけを符号化して 0 のキャンバスに貼り、denoise mask で固定する。本製品とほぼ同じ方式）の README とコードの注釈が、次のように書いている。
   - 「The outermost latent row on each band-facing edge is left free: the VAE folds whatever lies past the source edge into that row, and pinning it produces an image border at the seam.」
   - コードの注釈: 「pinning a band-facing edge row makes the model render that context (a border, or replicated rows)」。
   - 対策: 帯に面した縁の潜在行を**固定しない**（生成に回す）。そのうえで、復号の後に元の画素をその行の幅で smoothstep のなじませで貼り戻し、継ぎ目でフレームごとに RGB の段差を合わせる。
   - 2026-09-08 のコミット `59b51ef` で入った。作者は 1280×720→1280×960 などで端から端まで動かしたと、ComfyUI の PR #16184 に書いている。
   - **ただし、線を数値で測った記録や、固定する／しないの比較の画像は公開されていない。** 主張は作者の観察と説明だけである（一次資料。検証はない）。
2. **原因は「固定した縁のトークンの中身」とする説が最も有力である（候補 (e)）。** 根拠は次の 4 つ。
   - (i) 上の TwoAbove の記述。
   - (ii) VideoX-Fun の作者が、穴の詰め物が「VAE の受容野を通して残す領域を汚す」と書いていること（§17.2）。
   - (iii) arXiv 2602.14157 が、潜在のマスクをそのまま縮小すると元の輪郭が潜在に漏れて結果に残り、マスクを境界で少し広げると直ると報告していること。
   - (iv) 本製品の症状が、なじませの幅に依存しないこと。
   - 本製品の符号化器の詰め物は鏡写し（§17.3）なので、縁の潜在行は「外側に鏡像が続く」絵を符号化している。これを固定すると、生成のトークンとの間に「元の映像の縁」という情報が残り続けると考えられる（推測）。
3. **時刻の不連続（候補 (a)）を主因とする報告は見つからなかった。** ComfyUI・WanGP・vLLM-Omni・LongMedia・MMH3Tools・dkackman はどれも硬い 0/1 のマスクで行の時刻を止めており、時刻の段差を線の原因と名指しした資料は無い。
   - TwoAbove の対策は境界を 1 行内側へずらすだけで、時刻の段差は残る。それでも線が消えたと書いているので、(a) より (e) を支える（推測。作者の観察からの導出）。
4. **注入にノイズを足すかどうか（候補 (b)）が線に効くという資料は無い。** ComfyUI と WanGP の 0.999·clean＋0.001·noise は、時刻 0.999 という札と中身をそろえるための値と読める。WanGP の「masking strength <0.75」は**固定を最後のステップで外す**仕組みで、空間のなじませの仕組みではない（§3.2）。
5. **角が丸くなる理由を書いた資料は見つからなかった**（候補 (f)）。
6. **試す価値の高い対策の順位**（根拠の強い順。どれも本製品で計測していない）:
   1. **帯に面した縁のトークン 1 列（32 画素）を固定から外し、復号の後に元の画素をその幅のなじませで貼り戻す。** 前例は TwoAbove（縁の潜在行を外す）、dkackman（生成のマスクを既定で 1 マス広げる）、arXiv 2602.14157（マスクを境界で少し広げる）、LTX 公式（`spatial_radius` でマスクを広げる）。本製品のトークンは 32 画素で、元の映像の位置が 32 の倍数にそろっていれば、外す単位は 32 画素になる。貼り戻しの外側の 32 画素は生成された画素とのなじませに使う。
   2. **継ぎ目でフレームごとに RGB の段差を合わせる**（TwoAbove の `_match_band_tone`）。1 で線が消えた後に色の段差が残るときに試す。
   3. **2 段で作る**: 1 回目の出力の画素に元の画素を貼り、キャンバスごと符号化し直してから、もう一度固定して短く生成する（または復号だけする）。縁のトークンが鏡像ではなく本物の隣を見た潜在になる。前例は TwoAbove の「生成した潜在を VAE で往復させ、固定する入力をいつも符号化器の出した潜在にする」（窓の受け渡しの用途）。空間の縁に使った前例は無い（推測）。
   4. **最後の数ステップだけ固定を外す**（WanGP の masking strength の形）。4 ステップでは 0.75 が「最後の 1 ステップだけ外す」になる。元の領域の全体が描き直されるが、貼り戻しで忠実さは戻る。線に効いた記録は無い。
   5. 連続値のマスクで内側へなじませる（MMH3Tools の `feather`）。作者自身が、今の ComfyUI では連続値の帯が継ぎ目を荒らした観察を書き、0 以外の値は疑えと注意している。優先度は低い。

---

## 1. 報告の一覧

| # | 出典 | 日付 | 症状 | 原因の特定 | 対策 | 種別 |
|---|---|---|---|---|---|---|
| 1 | TwoAbove/ComfyUI-H3VideoOutpaint（MIT）の README と `nodes.py` の `_pinned_span` の注釈、`tests/test_nodes.py` | コミット `59b51ef`（2026-09-08） | 帯に面した縁の潜在行を固定すると、継ぎ目に**画像の枠（border）**が出る。または元の行が繰り返される | あり（作者の説明）: 符号化器が元の縁の外側にあるもの（単独の符号化では詰め物）を縁の潜在行に折り込み、固定すると、モデルがその文脈を描く | 縁の潜在行を固定しない。復号の後に元の画素を smoothstep のなじませで貼り戻す。継ぎ目でフレームごとの RGB オフセットを合わせ、帯の中へ徐々に戻す。列ごとの補正は縞になるので、フレームごとに 1 つの補正にする | 一次（作者の記述）。計測の公開は無い |
| 2 | ComfyUI PR #16184（TwoAbove。未マージ・open） | 2026-09-08 作成 | — | — | 元の長方形を、位置を指定したキーフレームの行としても渡す（固定と同時に参照の行で見せる）。1254×720→1248×960、1280×720→1280×960 で端から端まで動かした | 一次 |
| 3 | T8mars/comfyui-minimax-h3-audio-T8（GPL-3.0。文書だけを読んだ）の `docs/OUTPAINT_SOURCE_MODES.md`・`docs/H3_VIDEO_OUTPAINT_PLAN.md` | `ee569da`（2026-11-09）。記述は 2026-09-06〜09-09 | `preserve_source`（元の画素の貼り戻し）では「visible border seams may remain」。中文版は「硬い貼り戻しで輪郭が途切れる」と書く。継ぎ目の修正版も「visibly discontinuous」のままだった。キャンバスに黒い帯の入った入力では「black source-boundary bars」 | 特定していない（「not proof of a model-only cause」） | 既定を `joint_decode`（元の領域も含めてキャンバス全体を復号した画素をそのまま出す）に変えた。元の細部は変わりうる。参照実装は TwoAbove の `27df0ff`（縁の行を外す前の版） | 一次（作者の記述）。GPL のため文書だけを読んだ |
| 4 | ckinpdx/ComfyUI-MMH3Tools（MIT）の `MMH3OutpaintLatent` の注釈と CHANGELOG 0.32.0 | 2026-08-07 | 余白を画素で詰めて符号化すると「black-edge artefact」が出る。黒や灰色の詰め物は 0 でない潜在になり、モデルは「何かがある」と読んで残そうとする | あり（作者の説明） | 余白は潜在 0。元の領域の内側へ `feather`（既定 64 画素＝4 マス）で連続値のマスクを当て、元の外側の帯を部分的に描き直して継ぎ目を隠す。輪郭が見えるなら「広めに広げて切り戻す」 | 一次（作者の記述） |
| 5 | MMH3Tools の CHANGELOG 0.72.2・0.73.0・0.76.0 | 2026-08-13〜 | 時間方向の重なりで連続値のなじませ（`feather_latents`）を使うと、継ぎ目が**ノイズっぽく**なった。0 にすると消えた | 初めは「時刻と中身の対応がずれる」と説明したが、後で ComfyUI 本体を読み直して撤回した。原因は不明 | なじませを撤去した。`MMH3OutpaintLatent` の空間のなじませは残したが、「0 以外の値は疑え」と注記した | 一次（作者の観察） |
| 6 | ComfyUI PR #15375 の drozbay のレビューへの返信 | 2026-08-15 | 画素の単位のマスクでは、2×2 トークンの時刻と混ぜたノイズがずれ、乱れが出る（動画が添付されている） | あり | マスクを 2×2 の最大値に寄せる（§17.2） | 一次 |
| 7 | ComfyUI issue #17429 | 2026-10-29・open | Sol-H3-VAE の復号器で、128 画素の格子（注意の窓）に沿って細い直線の継ぎ目が出る。滑らかな肌で見える | あり（窓の境目）。元の VAE では出ない | 未解決 | 一次。**本製品の症状とは別の現象**（境界の位置ではなく格子の位置に出る） |
| 8 | dkackman/diffusers-workflow #1077（Apache-2.0） | 2026-10-31 | 症状の報告ではない | — | H3 の部分編集で、生成のマスクを 2×2 の格子へ最大値で寄せた後、既定で 1 マス（`mask_grow=1`）広げる。画素の貼り戻しはしない | 一次（設計の記述） |
| 9 | arXiv 2602.14157（Ghorbel ほか。v2 2026-03-30）の付録 B | 2026 | 画素のマスクをそのまま潜在へ縮小すると、元の輪郭（青いシャツの縁）が潜在に漏れて結果に残る。LTX（空間 32 倍）で Wan2.1（8 倍）より目立つ | あり: 符号化器の縮小のせいで、マスクの外の情報が境界の潜在に入る | マスクの台を境界で少し広げる（「at the cost of slightly altering the observed regions around the mask edges」） | 一次（論文）。inpainting の例で、outpainting ではない |
| 10 | arXiv 2512.05198（Bradbury・Zhong。2025-12-04） | 2025 | 縮小したマスクで VAE の潜在を線形に混ぜると、マスクの継ぎ目に大きな乱れと色ずれが出る | あり: 今の VAE は小片を超えた広い文脈を潜在に持つ | 画素で合成したのと同じになる潜在の合成器を学習する | 一次（論文の要旨だけを読んだ） |
| 11 | LTX 公式の in/outpainting の文書 | 日付なし | 「Tight mask leaves artifacts」「Boundary seam visible」 | 書かれていない | `LTXVDilateVideoMask` の `spatial_radius` でマスクを広げる。`LTXVLaplacianPyramidBlend` のなじませ幅で元と合成する | 一次（公式文書） |
| 12 | vrgamegirl19/comfyui-vrgamedevgirl PR #216 | 2026-09-25 マージ | H3 の 2 段の高解像化（タイル）で、なじませの幅がタイルの重なり（128 画素）以上だと「frozen seam anchor」を消してしまい、タイルの格子の線がにじむ | あり | なじませの幅を 64 画素に下げる | 一次。タイルの継ぎ目の話で、Outpainting ではない |
| 13 | fal.ai の `minimax/h3-max/reframe-video`、fuser.studio の H3 Reframe | 日付なし | fuser の頁に「The original center can be subtly re-rendered」とある | — | — | 二次。H3 Max は閉じたモデルで、作りは公開されていない |
| 14 | WanGP issue #2036（LTX-2 の Outpainting） | 2026-07-26 | 黒を「ここを作る」の印に使う方式では、縁の暗い素材で帯が黒のまま残る | あり | 明示した 0/1 のマスクとラプラシアンのなじませの方式に替える | 一次。別のモデル・別の仕組み |

**どこにも見つからなかったこと**: 「角の丸い長方形」の形の線の報告。

---

## 2. 原因の候補ごとの裏付け

### (a) 固定トークンと生成トークンの時刻の不連続（注意と AdaLN〔時刻ごとの変調〕を通して、境界の生成トークンに枠を描かせる）
- **裏付けなし。** 時刻の段差を線の原因として名指しした資料は無かった。
- 反対の材料（推測を含む）:
  - ComfyUI・WanGP・vLLM-Omni・LongMedia・MMH3Tools・dkackman は、どれも硬い 0/1 の固定で時刻を止めている（§17.2）。
  - TwoAbove の対策は固定の境界を 1 行内側へずらすだけで、時刻の段差そのものは残る。それでも枠が消えたと作者は書いている。時刻の段差が主因なら、枠は 1 行内側へ移るだけのはずである（推測）。
- 関連: MMH3Tools の時間方向の観察（#5）では、**連続値**の時刻の帯が継ぎ目を荒らした。硬い段差ではなく中間の値の話である。
- 「モデルが内側を貼り付けた写真と読む」という言い方は資料に無かった。ただし TwoAbove の「the model render that context (a border, …)」はこれに近い（モデルが枠を描く）。TwoAbove はその引き金を時刻ではなく、縁のトークンの中身（折り込まれた外側の文脈）に置いている。

### (b) 注入にノイズを足すかどうか（ComfyUI の 0.999·clean＋0.001·noise と本製品の 1.0）
- **裏付けなし。** 線と結びつけた資料は無い。
- ComfyUI の 0.999 は時刻の札（`VISUAL_COND_TIMESTEP`）と同じ値で、札と中身をそろえるための値と読める。PR #15375 の 2026-08-15 のコミット `989e7a9b` は「align the mask blend with token timesteps in scale_latent_inpaint」で、意図は札と中身の整合である。
- WanGP の「<0.75」は別の仕組みである（§3.2）。0.001 のノイズの量は潜在の大きさに比べて小さい。線への効き目は小さいと見込まれる（推測）。

### (c) VAE の復号器が、固定の潜在と生成の潜在の境目で縁を作る
- **部分的な裏付け**:
  - 復号器のタイルの中は全結合の注意で、固定の領域の画素も生成の潜在の影響を受ける（§17.3）。
  - Blended Latent Diffusion と arXiv 2512.05198 は、潜在を混ぜると VAE を通して継ぎ目と色ずれが出ると書く。
  - TwoAbove の `decoded` の説明は「the seam is decoder-continuous」で、復号器は境目をまたいで連続な画素を出す、という見方である。
- ComfyUI #17429 の線は、復号器の**窓の格子**に出る線で、潜在の境目とは結びつかない。本製品で線が元の映像の境界に沿って出て、タイルの格子に沿っていないなら、#17429 の型ではない（推測）。
- (c) と (e) は区別しにくい。縁のトークンの中身が「鏡像を含む縁」で、その隣が生成の潜在なら、復号器はその食い違いを縁として描きうる（推測）。

### (d) 貼り戻し（元の画素と復号画素の差）
- T8 の報告（#3）では、硬い貼り戻しで輪郭が途切れた。TwoAbove はこの段差を、フレームごとの RGB の補正となじませで扱う。
- **本製品では、なじませの幅（0／30／90 画素）を変えても線が出るので、主因ではない見込み**（依頼文の観察。T8 でも、継ぎ目の修正版で線が残った）。

### (e) 固定トークンの潜在が「単独で符号化した元の映像」で、境界の外側が鏡写しだった
- **最も強い裏付けがある。**
  - TwoAbove（#1）: 「the VAE folds whatever lies past the source edge into that row」。単独で符号化すると、縁の外側は符号化器の詰め物（H3 では鏡写し。§17.3）になる。
  - MMH3Tools（#4）: 詰め物の画素が潜在に「何かがある」という構造を焼き付け、モデルはそれを残そうとする。
  - VideoX-Fun（§17.2）: 詰め物が VAE の受容野を通して残す領域を汚す。
  - arXiv 2602.14157（#9）: マスクの外の輪郭が、縮小を通して境界の潜在に漏れる。
- 本製品の符号化器のダウンサンプルは、右と下にだけ 1 画素を鏡写しで足す（非対称。§17.3）。上下左右で影響の大きさが違うかもしれない（推測。確かめていない）。
- 小さい pad（13／19／8／8 画素）で線がほぼ見えない観察とも合う。pad が 32 画素より小さいと、帯に面したトークンは生成の側になる（2×2 の最大値で寄せるため）。その結果、**縁のトークンはもともと固定されていない**（推測。本製品の寄せ方が ComfyUI と同じならそうなる）。

### (f) 角が丸くなる理由
- **資料は見つからなかった。**
- MMH3Tools の内側のなじませは、軸ごとの傾斜を `max()` で組み合わせるので、角は四角になる（コードを読んだ）。TwoAbove の貼り戻しの重みは、行と列の smoothstep の積なので、角はわずかに丸くなる（コードを読んだ）。ただし本製品はなじませの幅に依存せずに線が出るので、貼り戻しの形では説明できない。
- 推測（確かめていない）:
  - 角のトークンは 2 方向に鏡写しの詰め物を持ち、角では外側の文脈が斜めにも折り込まれる。
  - 復号器の注意は RoPE〔位置の回転埋め込み〕の相対位置で減衰するので、境目の影響はほぼ等方的に広がる。
  - これらが重なると、線は角で丸く見えうる。

---

## 3. 対策の候補と一次資料

### 3.1 ComfyUI の H3 の denoise mask が連続値を許した経緯（PR #15375）
- 初めのコミット `3c474997`（2026-08-05）: 説明は「Video masks snap to the 2x2 latent patch grid, audio masks to whole latent frames, both rounded to binary」。0.5 のしきい値で 0/1 に丸めていた。
- `28aa924e`（2026-08-08）「Allow fractional mask values for MiniMax-H3」: 連続値を許した。コミットの文に目的は書かれていない。MMH3Tools の記録によると、効果は「部分的な強さ（`overlap_strength`）が時刻の札にも効くようになった」ことである。
- `355e8d29`（2026-08-08）: 0.995 以上を 1、0.05 以下を 0 に寄せた。`a2980c29`（2026-08-09）: 1/256 に量子化し、札の種類の数を抑えた。MMH3Tools によると、0.995／0.05 の寄せは 2026-08-15 に外された。
- `989e7a9b`（2026-08-15）: 別のフックをやめ、`scale_latent_inpaint` の中で、マスクの混ぜ合わせをトークンの時刻にそろえた。comfyanonymous の「LTX と同じようにできないか」という指摘への対応である。
- **境界をぼかすための連続値という説明は、PR のどこにも無い。** 修復の例のワークフローはあるが、Outpainting と継ぎ目への言及は無い（§17.2）。

### 3.2 WanGP の masking strength の実装（仕様として読んだ。`5b63ae9`）
- `pipeline.py`: `mask_end_step = denoising_start_step + ceil(model_steps × masking_strength)`（ステップ数で上限を切る）。固定の行への元の潜在の入れ直しと、固定の時刻（Grouped Rows）は、`step + 1 < mask_end_step` の間だけ続く。その後のステップでは**元の領域の全体**が自由に生成される。
- つまり、masking strength は「固定を何割のステップで続けるか」であり、境界だけを変える仕組みではない。既定値はモデルの定義による（H3 の既定は確かめていない）。
- 説明文の「Use a lower masking strength (<0.75) to facilitate continuity with masked areas」は、Denoise Control Video（V2V の編集）の項目の助言である。Outpainting 専用ではない。計測の記録は無い。
- 4 ステップでの換算（算術）: 0.75 は固定を 3 ステップ続けて最後の 1 ステップで外す。0.5 は 2 ステップで外す。

### 3.3 DiffSynth の Retake の `return_to_timestep` との違い
- 設計正本 §3.1 のとおり、本製品の m=0 の更新は `return_to_timestep` と同じ式である。DiffSynth の映像の Retake は注入がノイズなし・時刻 1.0。本製品の Outpainting は注入がノイズなし・時刻 0.999。違うのは時刻の上限だけである。
- 空間の Outpainting の前例として DiffSynth を使える資料は無い（入口は時間方向だけ。§17.2）。

### 3.4 マスクを広げる・ぼかす（潜在で grow／blur）の H3 での推奨
- H3 の公式（MiniMax）には無い。
- H3 のコミュニティでは次のものがある:
  - TwoAbove: 縁の潜在行を固定から外す。マスクを 1 行広げるのと同じ効果になる。
  - dkackman: `mask_grow` の既定は 1 マス。
  - MMH3Tools: 内側へのなじませ（既定 64 画素）。作者は今の ComfyUI では疑わしいと注意し、代わりに「広げて切り戻す」を勧める。
- 他のモデルでは、LTX 公式の `spatial_radius` と、ComfyUI の画像の修復の `grow_mask_by`。

### 3.5 「固定を緩めても、最後に元の画素を貼り戻せば忠実さは保てる」という使い方の前例
- **TwoAbove が、まさにこの組み合わせである。** 縁の行を生成に回し、`exact` のモードで元の画素を貼り戻す。注釈は「Real source pixels replace the decode, except on the free rows facing a band, where they fade into the decode」。
- 他の例:
  - WanGP の Remake: マスクを 1.5% 広げてから 2 回ぼかし、元の画素を戻す（§17.2）。
  - LanPaint（GPL-3.0。README だけを読んだ）: H3 の `LanPaint_AVDecode` が、マスクでなじませた境目で元の映像へ合成する。
  - 一般の修復でも、貼り戻しは広く使われる（ComfyUI の VACE の雛形の ImageCompositeMasked など。§17.4）。

---

## 4. 探したが見つからなかった範囲

- ComfyUI 本体の issue と PR で「H3 outpaint」「border」「seam」を検索した。見つかったのは #15981（格子。修正済み）、#15978（v0.34 での劣化）、#16516、#17429（復号器の格子）、PR #16184 で、Outpainting の境界の線の報告は無かった。PR #15375 の人間の書き込みは 6 件で、継ぎ目や枠への言及は無い（PR の本体は読んだ。添付の動画は見ていない）。
- WanGP の issue で「outpaint」を検索した。H3 の境界の線の報告は無かった（LTX-2 の黒い帯の #2036 だけ）。README と H3 の 3 ファイルにも無い（§17.2 と同じ）。
- vLLM-Omni・LongMedia・drozbay: §17.2 で読んだ範囲に、線の報告は無い。今回は検索し直していない。
- Wan VACE／Wan 2.x・HunyuanVideo・CogVideoX: ウェブ検索で、Outpainting の境界の線の報告は見つけられなかった。
- 画像の拡散（SD／Flux）: 硬い潜在のマスクで角の丸い枠が出るという報告は見つけられなかった。一般の助言（マスクのぼかし・広げる・継ぎ目の 2 段目）だけがある。
- MiniMax 公式（論文・モデルカード・GitHub）: 空間の部分固定と Outpainting の記述は見つけられなかった（§17.2 と同じ）。閉じたモデルの H3 Max には Reframe（Outpainting）の API が fal.ai にあるが、作りは書かれていない。
- Reddit・Discord・CivitAI: ウェブ検索では、該当するスレッドに当たらなかった。Discord は調べられない。
- **確かめていないこと**: TwoAbove が縁の行を外す前と後の画像の比較（GIF の更新だけがあり、比較は公開されていない）。T8 の「6.6-second strict-mask experiment」の中身（統合されておらず、文書にも詳細が無い）。

---

## 5. 出典一覧（取得はすべて 2026-11-10）

| 出典 | 版 | ライセンス | 読んだところ |
|---|---|---|---|
| TwoAbove/ComfyUI-H3VideoOutpaint https://github.com/TwoAbove/ComfyUI-H3VideoOutpaint | `59b51ef`（2026-09-08）。履歴 `cdd4049`〜 | MIT（引用可） | README・`nodes.py`（`_pinned_span`・`_match_band_tone`・`_source_weight`・`_reencode_generated`・符号化）・`tests/test_nodes.py` |
| Comfy-Org/ComfyUI PR #16184 https://github.com/Comfy-Org/ComfyUI/pull/16184 | 2026-09-08・open | GPL-3.0（本文だけ） | 説明文 |
| Comfy-Org/ComfyUI PR #15375 https://github.com/Comfy-Org/ComfyUI/pull/15375 | 2026-08-18 マージ | GPL-3.0（仕様として読んだ。コードは写していない） | 説明文・コミットの一覧と文・レビューの書き込み |
| Comfy-Org/ComfyUI issue #17429 https://github.com/Comfy-Org/ComfyUI/issues/17429 | 2026-10-29・open | — | 本文・書き込み |
| ckinpdx/ComfyUI-MMH3Tools https://github.com/ckinpdx/ComfyUI-MMH3Tools | `82ea5fa`（2026-10-24） | MIT | README・CHANGELOG・`docs/core-changes.md`・`mmh3tools/nodes_trim.py` の `MMH3OutpaintLatent` |
| T8mars/comfyui-minimax-h3-audio-T8 https://github.com/T8mars/comfyui-minimax-h3-audio-T8 | `ee569da`（2026-11-09） | GPL-3.0-or-later（文書だけを読んだ。コードは読んでいない） | `docs/OUTPAINT_SOURCE_MODES.md`・`docs/H3_VIDEO_OUTPAINT_PLAN.md` |
| deepbeepmeep/WanGP https://github.com/deepbeepmeep/WanGP | `5b63ae9`（2026-11-10 00:55 +0100） | WanGP Community License 2.0（仕様として読んだ。写していない） | `models/minimax_h3/pipeline.py`（masking strength）・`minimax_h3_handler.py`・`docs/CHANGELOG.md`・issue #2036 |
| dkackman/diffusers-workflow issue #1077 https://github.com/dkackman/diffusers-workflow/issues/1077 | 2026-10-31 | Apache-2.0 | 本文・書き込み |
| vrgamegirl19/comfyui-vrgamedevgirl PR #216 https://github.com/vrgamegirl19/comfyui-vrgamedevgirl/pull/216 | 2026-09-26 マージ | NOASSERTION | 説明文 |
| arXiv 2602.14157 https://arxiv.org/html/2602.14157 | v2（2026-03-30） | 論文 | 付録 B・図 3 |
| arXiv 2512.05198 https://arxiv.org/abs/2512.05198 | v1（2025-12-04） | 論文 | 要旨 |
| arXiv 2506.12530 https://arxiv.org/abs/2506.12530 | v1（2025-06-14） | 論文 | 要旨（VAE の色の不均衡と学習の 2 段。本件との関係は薄い） |
| LTX 公式の in/outpainting https://docs.ltx.io/open-source-model/feature-guides/editing-effects/in-outpainting | 日付なし | — | 全文 |
| scraed/LanPaint https://github.com/scraed/LanPaint | — | GPL-3.0 | README |
| fal.ai minimax/h3-max/reframe-video https://fal.ai/models/minimax/h3-max/reframe-video/api ・ fuser.studio https://fuser.studio/models/minimax-h3-reframe | 日付なし | 商用 | 説明文（二次資料） |
| runcomfy の H3 Motion Transfer https://www.runcomfy.com/comfyui-workflows/minimax-h3-motion-transfer-in-comfyui-spatial-outpainting ・ comfy.icu の T8 のノードの頁 | 日付なし | — | 説明文（二次資料。境界の線への言及は無い） |
| 既存の調査 | — | — | 研究ノート `MINIMAX_H3_RESEARCH_NOTES.md` §17、保管庫の R3 報告、設計正本 `MINIMAX_H3_DESIGN.md` §3.1 |
