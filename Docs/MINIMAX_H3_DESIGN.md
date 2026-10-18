# MiniMax H3 設計 — 映像と音声を同時に作る第 3 のベースモデルの載せ方

## 本書の位置づけ

**生きた文書（MiniMax H3 系統の設計正本）。** ヘッダーのドロップダウンで選べる第 3 のベースモデル「MiniMax H3」（映像と音声を 1 つのモデルで同時に作る 33B の拡散モデル）を、本製品でどう動かすかを記述する。**本書には現在の姿だけを現在形で書く。** どういう順序でここへ辿り着いたか（調査の経緯・実装での訂正・敵対的レビューの指摘）は記録簿側が保持しているので、本書は繰り返さない。

- マルチエンジン機構（ベースモデルの切り替え、記述子、エンジン系統ごとのワーカーと仮想環境）の土台は [`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) である。本書はその上に載る H3 固有の取り決めだけを書く。
- 調査の記録は [`MINIMAX_H3_RESEARCH_NOTES.md`](MINIMAX_H3_RESEARCH_NOTES.md) である（**参考資料**。設計の正本ではない）。本書と記述が食い違った場合は本書を採る。
- α版の作業は完了している（記録は台帳 [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) 3-239）。次の作業の入口は台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §1 の MiniMax H3 の項目（§1-95・§1-97・§1-100・§1-102）である。
- **現在の実装範囲**: α版（第 1〜3 弾）は完了している。第 1 弾（ワーカーのセルフテスト）・第 2 弾（I2V〔先頭フレーム〕・常駐・アダプタ・系統登録・記述子・事前検査・API の系統別制約・操作パネル・Gradio・MCP）・第 3 弾（インストーラ `install-MiniMaxH3.bat`。仮想環境の構築と重みの取得を 1 回で行う。§10）で、記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §165〜§167。範囲の全体は §11。α版の後に、VRAM 16GB mode（台帳 §1-101 の第 2 段〔CLOSED 3-240〕。α版の後の最初の改修で、台帳では β版以降の項目。製品の H3 は α版のまま。§6.5）を実装した。実装と機械の門に合格し（記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169）、B-G5 のオーナーの目視にも合格した（台帳 CLOSED 3-240）。

本書のうち **【オーナー裁定】** と書いた規則は、オーナーの承認なしに変えない設計上の約束である。

### 関連文書

| 文書 | 関係 |
|------|------|
| [`PENDING_TASKS.md`](PENDING_TASKS.md) §1 の MiniMax H3 の項目 | 次の作業の入口（α版の記録は [`PENDING_TASKS_CLOSED.md`](PENDING_TASKS_CLOSED.md) 3-239） |
| [`MULTI_ENGINE_DESIGN.md`](MULTI_ENGINE_DESIGN.md) | 土台。記述子（§4）・エンジンアダプタ（§5.2）・仮想環境（§5.3）・推論モード（§5.8）・写像表（§9） |
| [`MINIMAX_H3_RESEARCH_NOTES.md`](MINIMAX_H3_RESEARCH_NOTES.md) | 調査の記録（参考資料）。**進め方の裁定**は同ノート §10。配布ファイルの実測は同 §13.2 |
| [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) | 実測と検証の正本（第 1 弾の門の結果は §165、第 2 弾は §166、第 3 弾は §167、VRAM 16GB mode の計測は §168・実装と門は §169） |
| `engine_h3/venv-engine-h3.freeze.txt` | 仮想環境の**版の正本**。本書は版の数字を書かない |
| `engine_h3/engine_h3-venv-pyproject.toml` | 仮想環境を手で作り直すときの依存の仕様（なぜその版かの説明つき） |

---

## 1. 結論（何をするのか）

1. MiniMax H3 を第 3 のエンジン系統（`h3`）として載せている。**専用の仮想環境 `.venv-engine-h3`・専用のワーカー `engine_h3/worker.py`・専用のアダプタ**を持つ（MULTI_ENGINE_DESIGN §3.3 の「同居できない系統」パターン）。
2. 重みは公式系 FL2VA の**枝刈り（AdaLN の枝を表に畳んだ版）int8 ConvRot** の DiT と、**int8 ConvRot のテキストエンコーダ**、Comfy-Org 配布の映像 VAE と音声 VAE である。int8 の行列積は comfy-kitchen の CUDA 拡張に任せ、CPU オフロードは本製品の block swap と先読みに載せる。VRAM 16GB で動かす。
3. 推論は CFG なし・1 ステップ 1 回の前向き計算で、映像と音声を同じ系列の中で同時にノイズ除去する。ステップ数と shift は要求の項目として API と操作パネルから渡し、ワーカーの引数で受け取る（§9）。
4. ノイズ除去の段には「VRAM 16GB mode」（既定 on）がある。活性化をヘッド群と行の塊に分けて VRAM の山を下げ、共有メモリへの退避を防ぐ（§6.5）。

---

## 2. オーナー裁定

番号は研究ノート [`MINIMAX_H3_RESEARCH_NOTES.md`](MINIMAX_H3_RESEARCH_NOTES.md) §10 の裁定番号をそのまま使う。**進め方の裁定（作業の順序・了承の取り方・インストーラの名前など）は同ノート §10 に残してあり、本書には移していない。**

- **裁定 3 【オーナー裁定】**: 最初のタスク範囲は、FL2VA 重み 1 本での T2V（文章から動画）と I2V（画像から動画）の両対応とする。Ref2VA（参照素材からの生成）は後回しである。α版の I2V は先頭フレームだけを受け付ける。
- **裁定 6 【オーナー裁定】**: 重みの形式は int8 ConvRot とする。利用者が CivitAI などから落としたファイルをフォルダに置くだけで動かせることを理想とする。理由は、ファインチューンが大量に出回っており、コミュニティの主流が int8 ConvRot だからである。
- **裁定 7 【オーナー裁定】**: 配布は公開リポジトリからの直接ダウンロードとし、再ホストはしない（ライセンスに地域条項があるため）。
- **裁定 8 【オーナー裁定】**: テキストエンコーダは int8 ConvRot を使う。nvfp4 や bf16 は将来の拡張とする。
- **裁定 9 【オーナー裁定】**: 動画の尺は「快適上限を測って決める」方針に揃える。外部ガイドの 1344×768・345 フレームは正式目標にしない。
- **裁定 10 【オーナー裁定】**: LoRA は後回しとする。Turbo LoRA（少ないステップで生成するための蒸留 LoRA）への対応は Style LoRA への対応とほぼ同じ仕組みになるので、Style LoRA に対応するときに一緒に進める。
- **裁定 11 【オーナー裁定】**: H3 の仮想環境は、torch の版を LTX に揃えなくてよい。comfy-kitchen の CUDA 拡張が効く版を選ぶ。利用者側の条件（ドライバ R580 以上・ディスクの増加）は README に書く。
- **裁定 16 【オーナー裁定】**: int8 の実行は comfy-kitchen の CUDA 拡張に任せる。H3 の仮想環境の torch は、MiniMax H3 を動かすことに最も適した版を選ぶ。本製品はモデルごとに仮想環境を持てる設計なので、既存の torch の版にこだわらない。
- **裁定 17 【オーナー裁定】**: 実装の土台は DiffSynth-Studio（Apache-2.0）、diffusers は数値の正解データ源、ComfyUI（GPL-3.0）は仕様書として読むだけとする。次の 4 点を設計に織り込む。(1) int8 の実行は comfy-kitchen に任せる（既存の読み込み〔呼び出しごとに bf16 へ戻す・groupsize 256 のみ〕はヘッダの判定には使えても、H3 の実行経路には向かない）。(2) attention は DiffSynth の区間ごとの Python ループを使わず、diffusers の「1 リクエストを 1 つの系列に詰めてマスクなしで渡す」形を手本にして書き、本製品に導入済みの SageAttention を効かせる。(3) CPU オフロードは DiffSynth の仕組みではなく、本製品の block swap と先読みを接続する。(4) ローダは「揺れを吸収する」作りで書く（DiffSynth の変換表を出発点に、ComfyUI の判定の振る舞いを仕様として自前で書く）。
- **裁定 18 【オーナー裁定】**: ステップ数と shift は定数の既定値を持ち、ワーカーは引数で受け取る（画面に出すのは裁定 22）。
- **裁定 19 【オーナー裁定】**: α版の I2V は先頭フレームのみとする。末尾キーフレームと中間キーフレームは台帳 §1-97 で扱う。
- **裁定 22 【オーナー裁定】**: ステップ数と shift（映像・音声）の 3 本のスライダーを、操作パネルと Gradio に出す。既存の CFG スライダーのモックを差し替え、項目名・スライダー本体・数値を 1 行に並べる。既定値と範囲は裁定 24 である。背景は、Turbo を融合した低ステップの配布物が多く、LoRA 未対応でも速く生成できる見込みがあることである。
- **裁定 24 【オーナー裁定】**: ステップ数と shift（映像・音声）の 3 本のスライダーの既定値は **20／12／3**（ステップ数／shift〔映像〕／shift〔音声〕）。範囲は、ステップ数が 1〜50 の整数、shift が映像・音声とも 1〜20（刻みは 0.1）である。
- **裁定 26 【オーナー裁定】**: 常駐（生成のあいだに読んだ重みを次の生成まで捨てずに持つこと）は α版から有りである。本製品の推奨スペックは VRAM 16GB・主メモリ 64GB であり、これに足りない機械への対応は後から考える。常駐の on／off は Settings 画面の「Acceleration／生成の高速化」の区画にある既存のトグル（モデル骨格の常駐）をそのまま使う。
- **裁定 28 【オーナー裁定】**: I2V の先頭画像の VAE 符号化は、分布の平均を使う（サンプルしない）。ComfyUI と DiffSynth も平均を取り、diffusers だけが固定の種でサンプルする。
- **裁定 29 【オーナー裁定】**: サンプラは Euler 固定とし、選択肢は出さない。
- **裁定 30 【オーナー裁定】**: 操作パネルの方針。LTX に固有の機能（IC-LoRA、Edit の Alpha Gen、Chained の Stage-2〔upscale pass〕の clip length など）は H3 では非表示にする。動画生成の基本機能（中間キーフレームの指定、Chained、Inpainting、Outpainting など）は H3 でも将来実装する予定なので、グレーアウトして操作できない状態で配置だけ決めておく。H3 には「Ref2V」（参照素材からの生成）という目玉機能があり、LTX とまったく違う UI が要る。その UI／UX は Ref2V の実装時に考える。
- **裁定 32 【オーナー裁定】**: 常駐の方式。テキストエンコーダは層ごとにファイルから GPU へ直接読む（§4）。理由は、DiT と VAE を常駐させたまま、テキストエンコーダを主メモリへ丸ごと読む方式を取ると、主メモリの山が 64GB 機でもページファイルに溢れ、常駐の目的（速くする）と逆になるからである。直接読みにすれば、テキストエンコーダは主メモリをほぼ使わない。
- **裁定 33 【オーナー裁定】**: 映像 VAE は符号化器と復号器に分けて組み、T2V では符号化器を読まない（§8）。
- **裁定 34 【オーナー裁定】**: 操作パネルの Edit タブは、H3 では 4 つのサブタブ（Retake・Outpainting・Inpainting・Alpha Gen）が全部閉じるため、既存の規則によりタブごと灰色になる。裁定 30 の「グレーアウトして配置だけ決めておく」はこの形で満たす。
- **裁定 35 【オーナー裁定】**: NAG のアコーディオンは H3 では非表示でなく灰色にし、要求に NAG の項目を載せない（非表示にすると、有効にしたまま解除できず全ジョブが 422 になるため）。IC-LoRA の参照動画のアコーディオンは非表示にする。Single の A2V カードとバッチ A2V は H3 では非表示にする（2026-10-18 の目視の所感。当初の「連結要求そのものなので `chain` で閉じる〔灰色〕」を置き換えた。いまのコードは `featureScope.ts` の `chain` の行で灰色にしており、非表示への変更は台帳 §1-100）。
- **裁定 36 【オーナー裁定】**: ライセンスは README に明記する（名称・適用除外地域・取得者の自己責任）。インストーラは取得前に注意文とライセンスの URL を表示するだけで、同意の入力は作らない（§10）。
- **裁定 37 【オーナー裁定】**: README で H3 を「α版」と明示し、できること・できないこと・快適上限が未較正で大きな画角では遅くなること・推奨（VRAM 16GB・主メモリ 64GB）を書く（README §7.6）。
  > **訂正（2026-10-19）**: 「大きな画角では遅くなる」の README の文は、VRAM 16GB mode の実装（台帳 CLOSED 3-240）で現行化した。README では快適上限が未較正であることを残している。記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169。
- **裁定 38 【オーナー裁定】**: GPU 世代の文言は正直に書く。「ドライバ R580 以上・RTX 20 以降（GTX 10 以前は動かない）。確かめたのは RTX 40 だけ。RTX 30・50 は未確認。RTX 20 は int8 が遅い経路になり SageAttention も使えず、動作も未確認」。
  > **訂正（2026-10-18）**: 文言は裁定 43 で更新した（RTX 30 でも確認・RTX 50 は未確認・主メモリ 32GB はページファイル前提で短い生成を確認・推奨 64GB）。現在の文言は §10 と README にある。
- **裁定 41 【オーナー裁定】**: インストーラは `MiniMaxAI/MiniMax-H3` の `LICENSE` も取得して `models/MiniMaxH3/LICENSE` に置く。本製品はモデルを再配布しないが、ライセンスは第三者へ渡すときに写しと NOTICE を求めているので、写しを手元に置く安全側の手当てである。NOTICE は作らない（§10）。
- **裁定 43 【オーナー裁定】**: README の GPU 世代と主メモリの文言を、サブマシンの門の結果を受けて更新する（RTX 30 で確認・主メモリ 32GB はページファイル前提で短い生成を確認・推奨 64GB）。裁定 38 の文言の更新である。
- **裁定 44 【オーナー裁定】**: 名称は「VRAM 16GB mode」。Settings の「Acceleration／生成の高速化」のトグルで、**H3 では既定 on**、off も選べる。Gradio と MCP にも出す。値はジョブごとに毎回送る（§6.5）。
- **裁定 45 【オーナー裁定】**: 切り替えは同じワーカーの中で行う。off は現行の経路に一切触れず（基準の指紋 `0da6111c…` の不変で担保する）、on が新しい経路である。ワーカーや仮想環境は複製しない。
- **裁定 46 【オーナー裁定】**: 数値の約束。目標はビット一致である。門で最終桁の差が出たときは、ブロックごとの最大差がゆらぎ級であることと、オーナーの目視で品質差が無いことの 2 つで採否を決める。
- **裁定 47 【オーナー裁定】**: OOM は、ジョブを失敗させて「解像度かフレーム数を下げるか、VRAM 16GB mode を切る」と案内する（「画角」と書かない）。黙って退避を許す経路は作らない。既知の限界: 割り当て器の上限はこのプロセスの割り当てだけを縛るので、ジョブ中に他のプロセスが VRAM を増やした場合の退避は防げない。
- **裁定 51 【オーナー裁定】**: 分割の単位（ヘッド群・行の塊）と常駐数は、形状だけから決める決定的な規則とする。割り当て器の上限だけは、実行時の値から導いてよい。
- **裁定 53 【オーナー裁定】**: VRAM 16GB mode の数値の差を採用する。sage のヘッド数依存の差（50 ブロック中 2 ブロック・相対差 6.6e-3 以下）は、B-G5 のオーナーの目視で合格した。sdpa はビット一致である（§6.5）。

### 裁定ではなく設計として決めたこと（裁定 17 の論点）

- (a) DiffSynth は**移植**する（パッケージとしては入れない）。§8。
- (b) 映像 VAE は Comfy-Org の fp16 版、音声 VAE は同 fp32 版を使う。§8。
- (c) CFG は使わない。§3。
- (d) comfy-kitchen の triton 経路は切り、CUDA 経路に固定する。§5。

---

## 3. モデルの正体と推論の形

配布ファイルの実測は研究ノート §13.2 にある。ここでは推論の形だけを書く。

- **規模と構成**: DiT は約 33B パラメータ。隠れ幅 5376・50 ブロック・56 ヘッド×128 次元。このうち約 13B は AdaLN（時刻に応じて各層の変調量を決める枝）で、枝刈り版ではこの枝が表に畳まれている。
- **映像と音声を同時に作る**: 映像と音声（32kHz ステレオ）のトークンを 1 本の系列に詰め、同じ DiT で同時にノイズ除去する。
- **CFG なし【設計】**: モデルはガイダンス蒸留済みで、negative prompt も guidance scale も無い。1 ステップにつき前向き計算は 1 回である。
- **Rectified Flow の Euler**: モデルの出力は「データ方向の速度」v で、`x0 = x + σ·v` と読む（一般の flow-match と符号が逆）。更新は `x_next = x + (σ − σ_next)·v`。N ステップは N 回の評価である。
- **shift は 2 本立て**: σ の列は `base = linspace(1, 0, N+1)[:-1]`、`σ = s·base / (1 + (s−1)·base)` で作る。s は映像用と音声用の 2 つで、同じ添字 i について映像は σ_v[i]、音声は σ_a[i] を使い、出力もそれぞれの列で別々に更新する。モデルに渡す時刻は `t = 1 − σ`（1 が無ノイズ）。
- **フレームと解像度の規約**: 24fps 固定。フレーム数は **17n+5**。潜在フレーム数は `T' = 5n + 2`、音声の潜在数は `A = round(F / 24 × 40)`（例: 124 フレームで T'=37、A=207）。縦横は **32 の倍数**（映像 VAE の 16 倍圧縮とパッチ 2 の積）。モデルの学習上の既定は短辺 768 である（本製品の既定の画角ではない）。
- **系列の並び**: `[ text(L) | cond | audio(2A) | video(T'×(H/32)×(W/32)) ]`。cond は条件付けのキーフレーム行で、T2V では空である。attention は系列全体で全結合（区間ごとの制限もマスクも無い）。token_refiner（テキストだけを整える 2 ブロック）はテキスト行の中に閉じ、時刻にも位置にも依存しないので、ループの前に 1 回だけ計算する。
- **AdaLN の添字**: 各行が使う変調の行は `時刻の番号 × 3 + タグ`（タグは 0＝映像、1＝テキスト、2＝音声）。時刻の番号は、映像・テキスト・音声（と条件行）の時刻を `unique` でまとめた番号である。最終層の添字は時刻の番号だけである。
- **枝刈り版の AdaLN（表の補間）**: 時刻はスカラーなので、「時刻の埋め込み→AdaLN 射影」の写像を格子点の表（`adaln_t_table`、格子数 G・ランク R）に先計算してある。表の引き方は次の式で、SiLU は挟まない（float32）。

  ```text
  pos = clamp(t, 0, 1) × (G − 1)
  i0  = min(floor(pos), G − 2)        … 上端 t=1.0 でも範囲外を読まない
  w   = pos − i0
  e   = lerp(table[i0], table[i0+1], w)
  変調 = adaln_proj.linear(e)
  ```

- **RoPE（位置符号化）**: 3 軸（時間・縦・横）。座標は float64 で作り、角度と cos／sin は float32 で計算する。128 次元のうち先頭 96 次元だけを回す。映像の時間は一様でなく、`5/3 × (1,4,4,4,4)` の周期で進む。
- **I2V（先頭フレーム）の条件行**: 系列の cond の位置（`grid = (H/32)×(W/32)` 行）に先頭画像を入れる。
  - **値**: 先頭画像を映像 VAE の符号化器（f32。§8）で符号化し（分布の平均。裁定 28）、パッチに並べた `clean`（`[grid, 96]`・f32）を作る。条件行は `cond = 0.999·clean + (1 − 0.999)·noise` で、すべて f32 で計算する。`0.999` は定数 `KEYFRAME_NOISE_AUG` である。
  - **乱数の引き順**: `noise`（`[grid, 96]`・f32）は、ジョブの種から作った CPU の生成器から、**映像の初期乱数 → 音声の初期乱数 → 条件行の乱数**の順に引く。条件行の乱数は 1 回だけ引き、全ステップで使い回す。T2V は 3 本目を引かないので、T2V の結果は I2V の対応で変わらない。
  - **毎ステップ入れ直す**: 条件行はノイズ除去の状態（映像・音声）に入れず、毎ステップ同じ値を `embed_rows` に渡す。DiT の出力は映像と音声の行だけを返すので、条件行は更新されない。
  - **時刻**: 条件行の時刻は `max(t_video, 0.999)`（Python の float で求めて f32 の行に入れる）。時刻の番号は `unique` でまとめるので、番号は最大 3 つになる。
  - **AdaLN のタグ**: 条件行は映像（0）として扱う。
  - **RoPE の座標**: 条件行の時間は `t = L`（`L` は視覚ブロックを含むテキストの全長）、縦横は映像の格子と同じである。
- **I2V の視覚塔とプロンプトの書式**: テキストエンコーダは `Qwen3VLModel`（文章＋視覚塔）を使い、T2V でも I2V でも同じ型で組む（T2V は視覚塔を読まない）。画像のトークン列は、次の 3 つを**別々にトークン化して ID を連結**する（つなげた文字列をまとめてトークン化すると、`": "` の空白が次の語とくっつくため）。チャットテンプレートは使わない。
  1. `"<Picture 1>: "`（AdaLN のタグは文章＝1）
  2. `<|vision_start|>` ＋ `<|image_pad|>` × N ＋ `<|vision_end|>`（タグは映像＝0）
  3. プロンプト（タグは文章＝1）

  `N` は画像プロセッサが返す格子の積を `merge_size²` で割った値である。キャンバスが 32 の倍数で、画素数がプロセッサの上下限の内なら `N = (H/32)×(W/32)` になる。プロセッサの設定は tokenizer のフォルダの `preprocessor_config.json` に従う。HF の `forward` はそのまま使う（DeepStack〔視覚塔の中間層の特徴を文章側の先頭の層へ足す処理〕と画像トークンの 3 軸の位置は、層のループの中で起きるため）。前向きには、画像トークンの位置を示す `mm_token_type_ids` が要る。
- **先頭画像のキャンバスへの合わせ方**: 縦横比を保って**覆うように拡縮し、中央で切り抜く**（補間は PIL の LANCZOS。EXIF の向きを直して RGB にする。H.264 の再圧縮はしない）。切り抜き後の画像を、映像 VAE の符号化器と視覚塔の**両方**に渡す。符号化は ImageNet 方式の正規化・空間タイル（256／重なり 64）を常に使い、1 フレームを複製しない。画像の存在と読めることは、テキストエンコーダを読む**前**に確かめる。

---

## 4. 推論の段と常駐

推論の本体は条件付け・ノイズ除去・デコードの 3 段で、I2V は前に画像の段が加わる。一度に GPU に置くのは 1 段ぶんだけである。段が終わるたびに解放する（常駐を求められたときの保持物を除く。下の「常駐」）。

| 段 | 内容 | GPU に置くもの | 解放するもの |
|---|---|---|---|
| 0. 画像の読み込み（I2V のみ） | 先頭画像を読み、キャンバスに合わせる | なし | — |
| 1. 条件付け（`encode`） | テキストエンコーダでプロンプトを埋め込みにする | いま計算している 1 層（層ごとにファイルから直接読む）。I2V では視覚塔も計算の前に読む | テキストエンコーダ全体 |
| 2. 画像の符号化（`image_encode`。I2V のみ） | 映像 VAE の符号化器で先頭画像を符号化する | 符号化器（f32） | 符号化器（保持しない） |
| 3. ノイズ除去（`stage1_denoise`） | DiT をステップ数だけ回す | ブロック以外（埋め込み・最終層など）と、block swap が選んだ一部のブロック（`blocks_on_gpu`） | DiT のブロック以外と全ブロック（常駐しなければ DiT 全体も） |
| 4. デコード（`decode`） | 映像 VAE と音声 VAE で復号し、mp4 に書く | 映像の復号器 → 音声 VAE の順に 1 つずつ | VAE（常駐しなければ） |

- **テキストエンコーダは直接読む（裁定 32）**: プロンプトの前向き計算は 1 ジョブで 1 回しかない（CFG が無い）ので、層を主メモリに置く意味がない。HF の `forward` はそのまま使い、各デコーダ層の前向きの直前にファイルから GPU へその層だけを読み、直後に捨てるフックを掛ける。埋め込み表と視覚塔は、計算の前に読んで計算の後に捨てる。主メモリに残るのは骨格（meta）だけである。層数は数えて、全部の層が走ったことを検査する。`te_layers_on_gpu` という設定は無い。
- **主メモリ**: 直接読みにしたので、テキストエンコーダの段で主メモリは増えない。作業セット（実メモリに置いている量）とコミット量（OS が確保を約束した量）の実測は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §166 を見よ。32GB 機の扱いは §13。
- **常駐（`keep_resident`。裁定 26）**: 要求の `keep_resident` が真のジョブだけが、終わりに次の物を **CPU に**保持する。
  - 保持する物: DiT（ブロックもブロック以外も CPU へ戻す）・映像の復号器・音声 VAE。映像の符号化器は保持しない（I2V のジョブごとに組んで捨てる）。テキストエンコーダは直接読みなので保持する物が無い。`BlockSwapService` は元からプロセスに 1 つで、ページ固定の領域と転送用の流れをジョブをまたいで持つ。
  - **GPU には何も残さない**（cuBLAS が持つ作業領域は除く）。同期経路は最後の窓のブロックを GPU に残すので、ジョブの終わりに明示的に CPU へ戻す。
  - block swap の包みは、共有部品の印（元の `forward` を覚えている属性）から元の `forward` を戻して剥がす。ジョブの終わりと DiT を GPU へ置く直前の 2 か所で剥がし、次のジョブで入れ直す。先読みの on／off が常駐中に変わっても包みが二重にならない。共有部品は変更しない。
  - **キーが無い（偽の）ジョブ**は、保持物があれば使い回し（読み込みを飛ばす）、ジョブの終わりに解放する。**失敗したジョブ**は、どの段で止まったか分からないので保持物を丸ごと解放し、次のジョブはファイルから読み直す。検査の失敗や画像が読めない失敗でも同じである。
  - 使い回しの有無は `done` の `h3.phases` の `dit_source`・`vae_source`（`"file"`／`"resident"`）に出る。
- 常駐でないジョブは、毎生成、段 1 から段 4 まで組み直す。
- **VRAM 16GB mode（`vram16_mode`。H3 では既定 on）のノイズ除去の段**は、上の表の段 3 と次の点だけが違う（規則の正本は §6.5）。block swap の常駐ブロック数を `min(blocks_on_gpu, 4)` にする（DiT 全体を GPU に置く設定ならそのまま）。段の頭で割り当て器に上限を課し、段の終わりに外す（上限は実行時の空きから導く）。token_refiner と `condition_proj` は `refine_text` の後に GPU から降ろす（常駐するジョブは CPU へ、しないジョブは meta へ）。
- **ジョブごとの設定**: `attention_backend` と `block_swap_prefetch` は `generate` でもジョブごとに受ける（`load` の値は既定値になる）。SageAttention が使えないときは SDPA へ落とし、使った値を `done` に返す。attention の既定は本製品の現状の既定に従い、H3 専用の既定は作らない（裁定 27）。

---

## 5. int8 ConvRot の実行方式

- **素の int8 `Parameter` ＋ f32 `[N,1]` のスケール**: 量子化された Linear は `H3Int8Linear` という自前のモジュールにする。重みは素の int8 の `nn.Parameter`（`requires_grad=False`）、スケールはファイルどおりの f32 `[出力次元, 1]` のバッファ、bias は任意である。`forward` は `comfy_kitchen.int8_linear` を呼ぶ。ConvRot（活性化を回転してから量子化する方式）の有無と groupsize は層ごとの記述（`comfy_quant`）から取る。
- **量子化テンソル型を使わない理由**: comfy-kitchen には重みをラッパー型のテンソルとして持つ量子化テンソル型もある。しかし、ラッパー型のテンソルは本製品の block swap と先読み（素のテンソルを前提に重みを出し入れする）に乗らない。関数 API の `int8_linear` は同じ CUDA カーネルに届くので、素のテンソルで足りる。
- **CUDA 経路に固定する**: 起動時に `comfy_kitchen.list_backends()["cuda"]` が使用可能で、能力に `int8_linear` を含むことを確かめ、満たさなければロードを拒否する（理由の文を返す）。そのうえで **triton と eager の両経路を切る**。
  - triton を切る理由: comfy-kitchen の triton 経路の int8 行列積は、トークン数×出力幅が 2³¹ 以上で出力の位置が int32 であふれ、プロセスごと落ちる既知の不具合がある（上流 issue #136）。CUDA 経路は出力の位置を int64 で計算しているので、この限界を持たない。
  - eager を切る理由: eager は制約から外れた呼び出しを黙って受けて遅くなるだけで、速度の低下から区別できない。切っておけば、CUDA 経路に乗らない呼び出しは `NoCapableBackendError` で止まる（**fail loud**）。
- **CUDA 経路に乗る条件**: 重みとスケールが GPU 上にあること。block swap が重みを GPU に載せた状態で呼ぶ限り満たされる。
- groupsize は 256 以外（例: 64）も、そのまま `int8_linear` に渡す。

---

## 6. 既存部品との接続

- **block swap と先読み**: 既存の `engine/transformer/block_swap_service.py` と `block_swap_prefetch.py` をそのまま使う。`BlockSwapService` はパイプラインに常駐し、ジョブごとには作らない。H3 の本体ブロック列の名前が `blocks` で、block swap が探す候補名に当たる。素の `nn.Module` の列なので、既存の形でそのまま乗る。共有部品は変更しない。
- **テキストエンコーダの読み方**: 層ごとにファイルから GPU へ直接読む（§4）。`GemmaLayerOffloadService` は使わない。
- **推論の囲み【設計】**: `engine_h3` は**全経路 `torch.no_grad()`** で書く（`inference_mode` は 1 箇所も書かない）。規約と理由は MULTI_ENGINE_DESIGN §5.8。CPU テストで `engine_h3` 配下に `inference_mode` が現れないことを固定する。
- **精度の島の dtype**: 読み込みの直後に `H3DiT.normalize_dtypes()` で、精度の島の内外の dtype を揃える（公式ファイルでは AdaLN 射影の F16→F32 だけが変わる）。
- **meta で組んで `assign=True`【設計】**: DiT もテキストエンコーダも `torch.device("meta")` の中で組み、量子化層を `H3Int8Linear`（meta）に差し替え、テンソルを 1 本ずつ読んで `load_state_dict(assign=True)` で入れる。読み込みの後に meta のまま残ったテンソルが無いことを、読んだ部分ごとに検査する（テキストエンコーダは、T2V では文章側だけ、I2V では視覚塔も含める。T2V は視覚塔を読まず meta のまま残すため）。普通に組むと DiT が f32 で約 77GB、テキストエンコーダが bf16 で約 49GB を先に確保して主メモリ（64GB）を超えるので、この作りは必須である。

---

## 6.5 VRAM 16GB mode（ノイズ除去の活性化を分割して VRAM 16GB に収める）

**結論**: ノイズ除去の段で、1 ブロックの計算を「ヘッド群」と「行の塊」に分けて回し、活性化（計算途中のテンソル）の山を下げる。割り当て器（torch の CUDA メモリの割り当ての仕組み）には上限を課して、返されない空きセグメントが共有メモリへの退避（VRAM から主メモリ側へ溢れること）を起こさないようにする。H3 では**既定で on** で、Settings の「Acceleration／生成の高速化」から切れる（裁定 44）。製品の条件（I2V 896×1152×124 フレーム）で、ノイズ除去の割り当て／予約が 12,807／16,828MiB から 4,411／7,262MiB に下がり、1 ステップが約 203 秒から約 22 秒になる（退避ゼロ。詳しい実測は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169.3）。精度を落とす手段は使わない（裁定 46）。

### 何をするか

on の経路で行うことは次の 8 つである。#1・#7・#8 は VRAM の山そのものではなく、予約と固定費を下げる。

| # | 単位 | 内容 |
|---|---|---|
| 1 | 割り当て器の上限 | ノイズ除去の頭で、このプロセスの割り当て器に上限を課し、終わりに外す。導き方は下の「上限と OOM」 |
| 2 | 早期解放 | q・k・v を重みの行で切って射影し（融合した qkv を作らない）、埋め込み済みのトークンは bf16 にした直後に捨てる。norm 後の入力は最後のヘッド群の後に捨てる |
| 3 | ヘッド分割 | 注意計算をヘッド群ごとに回す。呼ぶのは off と同じ `_attention` |
| 4 | FFN の行の塊化 | FFN を行の塊ごとに回し、結果を 1 本の受け皿へ書く |
| 5 | AdaLN の塊化 | 変調とゲートを行の塊ごとに回す |
| 6 | 最終層 | 正規化と変調だけを行の塊ごとに回し、f32 の連続バッファに集めて、出力の行列積は映像と音声それぞれで 1 回行う（行の方向には分けない） |
| 7 | 降ろす | `refine_text` の後で、token_refiner と `condition_proj` を GPU から降ろす（下の「降ろす規則」）。本書で「退避」と書くのは共有メモリへの退避（spill）だけである |
| 8 | 常駐数 | block swap の常駐ブロック数を `min(n_load, 4)` にする（下の「常駐数」） |

- **off の経路は変えない（裁定 45）**。on と off は同じワーカーの中の切り替えで、ワーカーや仮想環境は複製しない。off の保証は、基準の指紋 `0da6111c…` が変わらないことで担保する（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169.3 の A-G3）。
- 採らないもの: Q だけの流し込み・q と k の strided な渡し・2 分割の MLP・融合した SwiGLU・RMSNorm と RoPE の融合・`smooth_k` の無効化・VAE の fp16 累積・CUTLASS 設定の強制。数値に触れる手段は、精度を落とさない目標（裁定 46）と合わないので使わない。

### 置き方

- **`engine_h3/lowvram.py`**: 新しい計算を書かない。off の経路が呼ぶ既存の関数（`_modulate`・`_gate`・`block.mlp`・`_attention`・`apply_rope`・`_linear_f32`・各 norm）を、**行の範囲・ヘッドの群ごとに呼ぶだけ**で、結果は事前に確保した受け皿へ書く。含むのは `Vram16Plan`（分割の単位）・`row_chunks`・`blocks_on_gpu_for`・`derive_memory_cap`・`block_forward`・`final_rows_chunked`。`_modulate`・`_gate`・`_attention`・`apply_rope`・`_linear_f32` の 5 つは `engine_h3.dit` のモジュールを経由して呼ぶので、テストでの差し替えが効く（`block.mlp` と各 norm はブロックから直接呼ぶ）。
- **`engine_h3/dit.py`** は分岐と引数だけ。`H3DiTBlock.vram16`（クラス属性。None なら off）に計画が入っていれば、`forward` の先頭で `lowvram.block_forward` へ渡す。`H3DiT.forward` は `vram16` と `tokens_box`（埋め込み済みトークンを入れた容れ物。取り出した後に呼び出し側の参照が残らないようにするため）を受け、ブロックごとに `block.vram16` を**毎回**代入する（None も。常駐した DiT の後始末になる）。`final_rows` も `vram16` の有無で分岐する。off の経路の行は変えない。
- **`H3Int8Linear.forward_rows(x, n0, n1)`**: 出力の行 `[n0, n1)` だけを、毎回 `self.weight` から切った連続ビューで計算する。呼び出し側がヘッドの境目で切るので、`n0` は結果として `head_dim`（128）の倍数になる。
- 割り当て器の上限と、モジュールを降ろす処理と、`tokens_box` は `pipeline_h3.py` の `_denoise`（`_vram16_begin`・`_vram16_finish`・`_evict_text_modules`）にある。ワーカーとアダプタの鍵は §8。

### 定数と理由

正本は `engine_h3/constants.py`（コメントに根拠の実測がある）。分割の単位と常駐数は**形状だけから決める決定的な規則**で、実行時の値から導いてよいのは割り当て器の上限だけである（裁定 51）。

| 定数 | 値 | 理由 |
|---|---|---|
| `VRAM16_HEAD_GROUP` | 8 | ヘッド群（56 ヘッド → 7 群）。28／14／8 の 3 択はどれも最速から +10% 以内で、割り当てが最小なのが 8（事前に決めた判定規則） |
| `VRAM16_ROW_CHUNK` | 4096 | FFN・AdaLN・最終層で共通の行の塊。8,192／4,096／2,048 は 4,096 と 2,048 が同点で、同点なら大きい塊を採る。塊は `ceil(S / 4096)` 個に**等分**する（S ≤ 4,096 は分割しない）。端の塊が小さくならないので、`int8_linear` の m ≤ 2（Turing は m ≤ 128）の別経路に入らない |
| `VRAM16_BLOCKS_ON_GPU` | 4 | on のときの常駐ブロック数の上限。4 と 8 は時間に差がなく、割り当てだけが 1,488MiB 違う。画角の長さによるしきい値は設けない |
| `VRAM16_CAP_MARGIN_MIB` | 1024 | 割り当て器の上限から引く余白 |

### 上限と OOM

- **上限**: `set_per_process_memory_fraction` に、`(このプロセスの予約 + mem_get_info の空き − 余白) / 総量` を渡す（`derive_memory_cap`。0〜1 に収める）。ノイズ除去の頭（`_place_dit` の前）で課し、`finally` の先頭で 1.0 に戻す。CUDA のデバイスは番号つきで渡す（番号無しの `cuda` を渡すと上限が課せない）。製品の条件での値は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169.3 にある。
- **記録**: `done` の `h3.phases.denoise.vram16` に、上限（比と MiB）・常駐ブロック数・ヘッド群・行の塊・降ろした先（`text_modules`）・割り当て器の再試行と OOM の回数の差分・`inactive_split` のピークを入れる。
- **OOM は失敗させる（裁定 47）**。on のとき `torch.OutOfMemoryError` を `VRAM16_OOM_MESSAGE`（「VRAM 16GB mode でも GPU メモリが足りません (CUDA out of memory)。解像度かフレーム数を下げるか、VRAM 16GB mode を切ってください」）の `RuntimeError` に包む。アプリ側は「out of memory」を含む文言を `GPU_OOM` に分類するので、ジョブは失敗し、`get_job` の `error` に案内が出る。黙って退避を許す経路は作らない。案内に「画角」と書かない。
- **カーネルが投げる `RuntimeError("CUDA error: out of memory")` は包まれない**。包むのは `torch.OutOfMemoryError` だけである（記録だけの小さな漏れ）。

### 降ろす規則

token_refiner と `condition_proj` は、`refine_text` の後に走らない。常駐するジョブ（`keep_resident`）は CPU へ移し（`_park_dit` が戻すのと同じ行き先）、常駐しないジョブは meta へ落として捨てる（`drop_to_meta`）。**降ろした後に `empty_cache` は呼ばない**。外した約 1.5GB は割り当て器のキャッシュに残る。重みの区画は 1 つあたり約 231MB 以下で、活性化（U＝約 408MiB）はそこに収まらないので、上限に当たったときに「キャッシュを返して再試行」が起き、`num_alloc_retries` に数えられうる（推測。割り当て器の内部は読んでいない）。再試行の回数を VRAM の逼迫の指標として読むときの注意として残す（製品の条件の実測では 0 回）。

### 常駐数

block swap の常駐ブロック数 `blocks_on_gpu`（block swap が GPU に置いておくブロックの数。モデルの保持〔`keep_resident`〕とは別）は、on でも off でも毎回、パイプラインが `install` の前に代入する（on のジョブで減らした数が次の off のジョブに残らない）。on のとき、`n_load` が 0 以下か全ブロック以上（DiT 全体を GPU に置く）ならそのまま、そうでなければ `min(n_load, 4)`。先読みの受け皿は常駐数 + 2 本なので、常駐 8 → 4 で受け皿が 10 → 6 本になり、固定費（受け皿）が減る。実測の割り当ての差は 1,488MiB（[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169.3）。時間は変わらない。

### 数値の約束

- **目標はビット一致（裁定 46）**。行ごと・ヘッドごとの演算だけを分割しているので、同じ関数を範囲ごとに呼べば一致する。
- 確かめた範囲（結果は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169.3）: GPU テストで `forward_rows` の結合・行の塊の縦結合・`k.mean` のヘッド群比較が完全一致。端から端まで（640×384・124 フレーム・20 ステップ・sdpa）の指紋は off も on も `0da6111c…` に全桁一致。
- **製品の形（S＝39,764・実重み）の 50 ブロック比較では、sdpa は 50 ブロックともビット一致、sage は 50 ブロック中 2 ブロックで値が違う**（ブロック 22 とブロック 40。再現性あり）。ブロックごとの最大絶対差は 22 で 256・40 で 1,792、値が違う要素の割合は 22 で約 54%・40 で約 3.3%、出力全体の相対差は 22 で 6.6e-3・40 で 3.1e-3（正本は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169.3）。ブロック 22 で切り分けたところ、この要因だけで説明できた。SageAttention の計算核が 1 回の呼び出しのヘッド数（off は 56、on は 8）で値を変える。q・k・v は全群で一致し、off の入力を 8 ヘッドずつで sage に渡すと on と完全一致する。ブロック 40 は切り分けていない（同じ仕組みと見られる。推測）。**この差を受け入れるかは裁定 46 に従い、ブロックごとの最大差がゆらぎ級であることと、オーナーの目視で品質差が無いことで決める**。B-G5 の目視（I2V 896×1152×124 フレーム・20 ステップ）で合格し、採用した（裁定 53）。

### 山の内訳

製品の条件（I2V 896×1152×124 フレーム、S＝39,764、sage）での**概算**。U は bf16 の `[S, 5376]` のテンソル 1 本（約 408MiB）で、bf16 の `[S, 7168]` は 1.33U、f32 の `[S, 5376]` は 2U。行の塊（4,096 行）とヘッド群（8 ヘッド）の一時テンソルは「+」で数える。ブロック 1 以降で、入口の `x` はブロックの最後まで生きている。

| 区間 | 生きているもの | on の概算（U） | off の概算（U） |
|---|---|---|---|
| 入口 | `x`（トークンはもともと bf16。`tokens_box` で呼び出し側に参照を残さず、ブロック 0 を抜けると解放される） | 1 | 2（トークンも保持） |
| norm1 と変調 | `x`・norm1 の出力・変調後の `h` + 塊の f32 の一時 | 約 3.4 | 7 |
| 注意（山） | `x`・`h`・出力の受け皿 `o`（1.33）+ 群の q・k・v・RoPE の出力・sage の内部 | 約 5 | 18.33 |
| out_proj | `x`・`o`・int8 の写し・出力 | 約 4 | 約 12.7 |
| 注意の残差 | `x`・注意の出力・新しい `x2` + 塊の一時 | 約 3.4 | 約 6 |
| norm2 と変調 | `x`・`x2`・norm2 の出力・`h2` | 4 | 8 |
| FFN | `x`・`x2`・`h2`・`m` + 塊の fc1 出力（`[4096, 28672]`）と活性化 | 約 5.3 | 14.67 |
| FFN の残差 | `x2`・`m`・出力 + 塊の一時 | 約 3.5 | 約 7 |

- off の列は第 1 段の棚卸し（[`Outputs-archive/h3-vram-2026-10-18/reports/R1_vram_inventory.md`](Outputs-archive/h3-vram-2026-10-18/reports/R1_vram_inventory.md) §3.2）の値で、活性化の山は 18.33U ≈ 7,475MiB。on の山は約 5U ≈ 2.1GB。
- 実測との照合: on のノイズ除去の割り当てのピーク 4,411MiB から、先読みの受け皿 6 本（371.08 × 6 ≈ 2,226MiB）を引くと約 2,185MiB ≈ 5.4U で、概算の山（約 5U）と合う。
- 山が下がるので、割り当て器の上限（14,033MiB）の下に収まり、返されない空きセグメントが共有メモリへの退避に化けない。

### 範囲と既知の限界

- **範囲**: on の q・k・v は群ごとに連続で、最大のオフセットは S × 群のヘッド数 × 128。群 8 で S ≈ 209 万行まで int32 の範囲に収まる。防御用の関数は作らない。
- **上限はこのプロセスの割り当てだけを縛る（裁定 47）**。ジョブ中に他のプロセスが VRAM を増やした場合の退避は防げない。
- **VRAM 16GB mode の実測は開発機（RTX 4070 Ti SUPER・Ada）だけである。** RTX 30 系（Ampere）は未確認（サブマシンで確かめたのは α版の T2V だけ。[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §167.8）。
- **off の経路の限界**: SDPA の mem-efficient 経路は、系列が約 10 万行を超えて v が view（行の刻みが詰まっていない）のとき、不正なメモリアクセスになる。実測は 98,304 行で正常・100,352 行で不正（v を連続にすれば正常。[`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §168.3 の T6）。on は群ごとに連続な q・k・v を渡すので、この境界に近づかない。off の経路には防御を足していない。
- 上限のせいで先読みの受け皿が確保できないとき、同期の経路に黙って切り替わる（記録だけ。on の条件では起きていない）。
- qkv の量子化が 1 ブロックで 3 × 7 群の 21 回走る。時間の要因として記録するが、実測の 1 ステップ（約 22 秒）に収まっている。

---

## 7. ヘッダの判定と吸収する揺れ

- **判定はテンソル名の指紋**: H3 の DiT は `{接頭辞}video_patch_proj.weight` と `{接頭辞}audio_patch_proj.weight` の**同時存在**で判定する。GGUF の KV のようなメタデータには頼らない（コミュニティ配布物は KV が偽装・欠落しており、safetensors の `__metadata__` も枝刈り版では空である）。MULTI_ENGINE_DESIGN §2.2・§2.4 の判定とは別の経路で、H3 系統は safetensors にこの指紋を使う。
- **吸収する揺れ**:
  - 接頭辞 4 通り（無し／`model.diffusion_model.`／`model.model.`／`net.`）を剥がす。
  - 構成（層数・幅・ヘッド数・ffn・AdaLN 表の格子数 G とランク・枝刈りか否か）は、メタデータでなく**テンソルの形状**から導く。
  - `comfy_quant` は `format`・`convrot`・`convrot_groupsize` の 3 キーだけを読み、他のキーは無視する。
  - groupsize は 256 以外も通す。量子化されていない層は、読んだ後に島の外側を bf16、内側を f32 に揃える。
  - 知らないテンソルは警告ログを出して捨てる。
- **拒否するもの**: 非枝刈り版（`adaln_t_table` が無い）と、旧式の量子化記述（`_quantization_metadata`）、`int8_tensorwise` 以外の形式、VSA の追加テンソル、PDD ヘッドは、理由の文つきで拒否する（または警告して捨てる）。主に試すのは公式のファイルで、コミュニティ版への対応は後回しである。例外として、オーナーが PinkCherry Hybrid を置いて生成に成功した（実機の事実・2026-10-18）。構造が同じことの根拠は研究ノート §13.3（ComfyUI のコードの読解であり、実測ではない）。対応の保証は §12 のとおり後回し（§13）。
- **置き場**: 判定は torch に依存しない `h3_format.py`（リポジトリ直下。`sft_quant_format.py`・`chain_math.py` と同じ置き場）に書く。アプリ側（torch の無い仮想環境）の事前検査からも呼ぶためである。
- **アプリ側の事前検査**: `services/model_registry.py` の `precheck_model_file` が記述子の系統を見て、`h3` なら DiT を `inspect_h3_dit`、テキストエンコーダを `inspect_h3_te` にかける。VAE 2 本は汎用の safetensors 検査（長さと JSON）である。`h3_format.py` の `H3FormatError` は `MODEL_INCOMPATIBLE` に変換して返す。
- **FL2VA と Ref2VA はヘッダで区別できない**（テンソル名・形状が同じ）。記述子（`scripts/manifests/40-minimaxh3.json`）で配布ファイル名を固定して縛る。

---

## 8. ファイル構成と境界

### 実装の土台

DiffSynth-Studio の H3 部品を**移植**する（Apache-2.0。移植したファイルの先頭に由来と表示を書く）。パッケージとしては入れない（`import diffsynth` が core 全体と modelscope などを芋づるに引き込むため）。ComfyUI（GPL-3.0）のコードは写さない。映像 VAE は Comfy-Org の `minimax_h3_video_vae_fp16.safetensors`（復号の計算は bf16、符号化器は f32）、音声 VAE は同 `minimax_h3_audio_vae_fp32.safetensors`（計算は f32）である。`latents_mean`／`std` は `__metadata__` の JSON（f32）から取る。

### ファイル構成

| ファイル | 役割 |
|---|---|
| `h3_format.py`（リポジトリ直下・torch 非依存） | ヘッダの判定・接頭辞の剥がし・構成の導出・`comfy_quant` の読み・量子化層の一覧（§7） |
| `engine_h3/constants.py` | 定数と既定値（§9）。フレーム規約の算術 |
| `engine_h3/sft_load.py` | テンソルを 1 本ずつ読んで meta のモデルへ入れる（`assign=True`）。読み残しの検査 |
| `engine_h3/quant_linear.py` | `H3Int8Linear`（出力の行の範囲だけを計算する `forward_rows` を含む）・CUDA 経路の検査と固定・量子化層の差し替え（§5） |
| `engine_h3/dit.py` | DiT 本体・枝刈り版の AdaLN・attention（SDPA／SageAttention）・行の並べ替え。VRAM 16GB mode の分岐（`vram16`）と `tokens_box` を持つ（§6.5） |
| `engine_h3/lowvram.py` | VRAM 16GB mode の経路（§6.5）。`dit.py` の既存関数をヘッド群・行の塊ごとに呼ぶだけで、新しい計算は持たない |
| `engine_h3/rope.py` | 3 軸 RoPE |
| `engine_h3/text_encoder.py` | Qwen3-VL（文章側 50 層＋I2V の視覚塔。正規化前の隠れ状態を出す）・層ごとの直接読み・トークン列の書式（§3・§4） |
| `engine_h3/scheduler.py` | σ の列と Euler の 1 ステップ |
| `engine_h3/video_vae.py`・`engine_h3/audio_vae.py` | 映像 VAE（符号化器 `H3VideoEncoder`＝f32・I2V のときだけ組む／復号器 `H3VideoDecoder`＝bf16。下の「映像 VAE の構成」）・音声 VAE |
| `engine_h3/image_cond.py` | 先頭画像の読み込みとキャンバスへの合わせ方（§3）・符号化器へ渡すテンソルへの変換 |
| `engine_h3/media_out.py` | mp4 の書き出し（映像 libx264、音声 AAC、色空間は BT.709 を明示して変換） |
| `engine_h3/pipeline_h3.py` | 段の組み立て（§4）・I2V の条件行（§3）・常駐の保持と解放 |
| `scripts/run_h3_pytest.py` | エンジン用の仮想環境でテストを回す実行器 |
| `services/engines/h3/adapter.py` | アプリ側のアダプタ。`H3Runner`（`LTXRunner` を継ぐ）・4 分類表（動作・拒否・無視・従属）・系統の規則表 `REQUEST_RULES`（§9）・`UNSUPPORTED_FEATURES`・ワーカーへのメッセージの組み立て・`metadata.json` の `h3` 欄 |
| `services/engines/ltx/request_rules.py` | 系統の規則表の型 `RequestRules` と、LTX 2.3／2.5 共通の表。H3 の表も同じ型で `adapter.py` に置く |
| `scripts/manifests/40-minimaxh3.json` | 記述子。`engine_family: "h3"`・カテゴリは既存の 4 語・`assets.tokenizer_dir`・`comfort` は持たない・`downloads`（配布元の 4 本と tokenizer 一式） |
| `tests/test_h3_format.py`・`tests/test_h3_engine.py`・`tests/test_h3_media.py`・`tests/test_engine_h3_adapter.py` | ヘッダの判定（アプリ側の仮想環境）／DiT・テキストエンコーダ・ワーカーの約束事（GPU のテストは既定で飛ばす〔`H3_GPU_TESTS=1` で回す〕。件数は `-rs` の出力が正本）／映像 VAE・音声 VAE・mp4 書き出し・画像の切り抜き（CPU）／アダプタ・規則表・`GET /models`・422 の形・mock での T2V と I2V |
| `engine_h3/worker.py` | ワーカー本体。`engine25/worker.py` と同じ作法（`python -m engine_h3.worker`、stdin の JSON 行、stdout の `@@LTX@@`＋JSON、stderr のログ `[h3_worker]`）。torch はモジュール最上位で import せず、`load` の中で遅延 import する |

### 呼び出しの分担

`load` はパスの検査とヘッダの判定だけを行い、重みは読まない。`generate` が毎回、段を組み直す（常駐するジョブは保持物を使い回す。§4）。

### ワーカーのメッセージ

アダプタの `_build_load_payload` がそのまま写せるよう、キー名はここで固定する。

- **`load`**: `transformer_path`・`text_encoder_path`・`video_vae_path`・`audio_vae_path`・`tokenizer_dir`・`blocks_on_gpu`・`attention_backend`・`block_swap_prefetch`・`deterministic`（`attention_backend` と `block_swap_prefetch` は、ジョブごとの値が無いときの既定になる）
- **`generate`**: `prompt`・`seed`・`width`・`height`・`num_frames`・`frame_rate`・`num_inference_steps`・`shift_video`・`shift_audio`・`images`・`keep_resident`・`attention_backend`・`block_swap_prefetch`・`vram16_mode`・`output_path`（ステップ数と shift は省略可。省略時は既定値）
  - `images`: `[{path, frame_idx}]`。無ければ T2V。**0 個か 1 個で、`frame_idx` は 0 だけ**を受け、それ以外は日本語の理由つきの `ValueError` でジョブを止める（プロセスは生き続ける）。要素の知らないキー（強度など）は無視してログに 1 回出す。アダプタは強度を送らない（H3 に強度の概念が無い）。
  - `keep_resident`: 真のときだけ保持する。無ければ偽（保持物を解放する）。アダプタは真のときだけ送る。
  - `attention_backend`・`block_swap_prefetch`: 無ければ `load` の値。アダプタは毎回送る。
  - `vram16_mode`: 真のときだけ VRAM 16GB mode の経路で回す（§6.5）。無ければ偽（ワーカーの既定は off。H3 の既定 on はアプリ側の規則表が書き戻す）。アダプタは毎回送る。
- **イベント**: `ready`（`sage_available`）／`progress`（`stage`・`index`・`total`）／`done`／`error`（`detail`）
- **進捗の段階名**: LTX の語彙を借りる。`encode`／`stage1_denoise`（単段のノイズ除去。H3 は 1 段なのでこの名前だけを使う）／`decode`。アダプタは進捗の割合計算に `_read_worker_events(one_stage=True)` を使う。`encode` の総数は T2V で 3、I2V で 4（4 つ目が画像の符号化）である。
- **`done` のキー**: `seed_used`・`sampler`・`num_inference_steps`・`shift_video`・`shift_audio`・`peak_vram_mb`・`peak_vram_reserved_mb`・`rss_peak_gib`・`seconds`・`num_frames`・`frame_rate`・`width`・`height`・`size_bytes`・`attention_used`（ジョブごとの値）・`block_swap_prefetch_used`・`keep_resident_used`（`"on"`／`"off"`。このジョブの終わりに保持したか）・`vram16_mode_used`（`"on"`／`"off"`。VRAM 16GB mode で回したか）・`images_used`（0 か 1）・`commit_gib`・`commit_peak_gib`（コミット量）・`h3`（段ごとの秒数と RSS を `phases` に、使われた量子化経路を `quant_backends` に持つ。`phases.decode` には `audio_samples`〔切り揃え後〕と `audio_samples_decoded`〔復号した全量〕が入る〔`engine_h3/pipeline_h3.py` の `_decode`〕。I2V のときは `phases` に `image_encode` が入る。`phases` の `denoise` には `dit_source`、`decode` には `vae_source` がいつも入る〔`"file"`／`"resident"`〕。VRAM 16GB mode のジョブは `phases.denoise.vram16` に上限・常駐数・分割の単位・割り当て器の再試行と OOM の回数の差分・`inactive_split` のピーク・退避の行き先〔`text_modules`〕が入る〔§6.5〕）

### 映像 VAE の構成

符号化器と復号器に分け、別々にファイルから読む（裁定 33）。共通の部分（正規化の平均と標準偏差・空間タイルの分割・継ぎ目の混ぜ）は基底の型に置く。

- **`H3VideoEncoder`**（`encoder`＋`quant_conv`）: f32。精度の島は無い。I2V のジョブごとに組んで先頭画像 1 枚を符号化し、すぐ捨てる。T2V では組まない。戻り値は事後分布の平均を `latents_mean`／`std` で正規化した潜在である。
- **`H3VideoDecoder`**（`post_quant_conv`＋`decoder`）: bf16。精度の島（`post_quant_conv`・入力の埋め込み・出力の正規化・各ブロックの正規化とスケール）は f32。
- ファイルの名前の振り分けは `encoder_key_map`／`decoder_key_map` で行い、それぞれの型は自分のテンソルだけを持つので、読み残しは例外になる。

### 映像 VAE の特例

- 潜在フレームが 1 つ（T'=1）のときは、単独では復号が悪いので、5 つに複製して復号し、3 番目を取る。
- **復号は CPU 上の uint8 `[F, H, W, 3]` を返す**。窓ごとに確定した片（継ぎ目を混ぜた後）を、その場で正規化を戻し→clamp→×255→round→uint8 の順で変換して CPU へ移し、連結は CPU で行う（T'=1 の特例も同じ契約）。GPU に全フレームを置かないので、大きな画角でも復号の VRAM は窓の大きさで決まる。バッチは 1 だけを受ける。音声の復号は窓なしである。

### tokenizer の渡し方

テキストエンコーダの tokenizer は、記述子のカテゴリではなく**記述子の `assets.tokenizer_dir`** で渡す（LTX 2.3 の `gemma_root` と同じフォルダ方式）。アダプタが `load` の `tokenizer_dir` に写す。`<d>` などの追加特殊トークンは `tokenizer_config.json` にしか無いので、`tokenizer.json` だけでは足りず、ファイル一式を 1 つのフォルダとして扱う。I2V の画像プロセッサ（`preprocessor_config.json`）も同じフォルダから作る。追加特殊トークン 7 語が語彙の 151669〜151675 に入ることを、ロード時に検査する（入らなければ明示的に足す）。

### セルフテスト

`python -m engine_h3.worker --selftest-generate` が、`load`／`generate` を合成メッセージで直接呼んで全イベントを記録し、mp4 の SHA-256 と `done` を JSON で書く。`--image P`（先頭フレーム 1 枚）で I2V、`--keep-resident` で常駐を、`--vram16` で VRAM 16GB mode を試せる。回帰の基準の指紋 `0da6111c…` は、`--prompt` に第 1 弾の門 G5 のプロンプト（黄色いレインコートの女性の文。正確な文は `Docs/Outputs-archive/h3-gate3/g2_selftest_g5prompt.cmd`）を渡し、ほかはセルフテストの既定（640×384・124 フレーム・20 ステップ・seed 42・sdpa）で回したときのものである。既定のプロンプトのまま回すと別の指紋になり、一致しない（条件は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §167）。

---

## 9. 定数と既定値

- **ワーカーの既定値と定数の正本は `engine_h3/constants.py`**、**API の既定と範囲の正本は `services/engines/h3/adapter.py` の `REQUEST_RULES`** である。本節の数値は裁定 24 と規約の代表値として引くもので、食い違えば正本を採る。
- 置くもの: ステップ数・shift（映像・音声）の既定値、24fps、空間 32（1 トークン＝32px 四方）、VAE の圧縮率とクリップ長、音声の潜在レート・サンプルレート・ホップ長、モダリティのタグ（映像 0・テキスト 1・音声 2）、ログの接頭辞 `[h3_worker]`、`latent_frames(F)`・`audio_latents(F)`（17n+5 を外れた F は `ValueError`）。
- ステップ数と shift の既定は **20／12／3**、範囲はステップ数が 1〜50 の整数・shift が 1〜20（刻み 0.1）である（裁定 24）。ワーカーの `generate` は引数で受け取り、省略時に既定値を使う。API では `num_inference_steps`・`shift_video`・`shift_audio` の既定が `null`（＝系統の既定値）で、アプリ側が系統の既定値を要求へ書き戻してからワーカーへ渡す。
- VRAM 16GB mode の定数（`VRAM16_HEAD_GROUP`・`VRAM16_ROW_CHUNK`・`VRAM16_BLOCKS_ON_GPU`・`VRAM16_CAP_MARGIN_MIB`）も `constants.py` が正本で、値と理由は §6.5 の表にある。
- 解像度の規約: 縦横は 32 の倍数、フレーム数は 17n+5 かつ `MIN_NUM_FRAMES = 22` 以上（F ≥ 22。F=5 は拒否する）、フレームレートは 24 固定。違反は日本語の理由つきの `ValueError` で止める。
- **アプリ側（API）の制約の正本は、系統の規則表**（`services/engines/h3/adapter.py` の `REQUEST_RULES`）である。フレーム数の格子（17n+5・22 以上・481 以下）・縦横の倍数（32）・フレームレート（24 固定）・ステップ数と shift の既定と範囲を持ち、`GET /models` の `base_models[].constraints` に機械的に出る。操作パネル・Gradio・MCP はこれを読む。形は MULTI_ENGINE_DESIGN §5.6、検査の仕組みは同 §7 を見よ。

---

## 10. 仮想環境

- **名前**: `.venv-engine-h3`。MULTI_ENGINE_DESIGN §5.3 の一覧に並ぶ、3 つ目のエンジン用仮想環境である。
- **torch を LTX に揃えない理由【オーナー裁定：裁定 11・16】**: comfy-kitchen の CUDA 拡張は CUDA 13 系の torch で効く。拡張自体は torch の DLL にリンクしていないので、CUDA 13 系でさえあれば torch の版は問わない。そのうえで、SageAttention・triton-windows などの周辺の実績がある版を選ぶ。既存の 2 系統の torch（CUDA 12.8 系）のままでは CUDA 拡張が使えない。
- **利用者側の条件**: NVIDIA ドライバ **R580 以上**（CUDA 13 系の下限）。GPU は **Turing（RTX 20 系）以降**（CUDA 13 で Pascal 以前が削除された）。動作を確かめたのは Ada（RTX 4070 Ti SUPER）と Ampere（RTX 3080 Laptop 16GB）で、Blackwell（RTX 50 系）は未確認、Turing は int8 が遅い経路になり SageAttention も使えず、動作も未確認である（裁定 38・43）。主メモリの推奨は 64GB で、32GB 機はページファイル（「システム管理」）を前提に短い生成を確認している。コミット量の山が 32GB を超えるのでページファイルに頼り、遅くなる場合がある（裁定 26・43。実測は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §167）。利用者向けの書き方（確かめた世代と未確認の世代の区別、主メモリと容量）は README の「ハードウェア要件の詳細」「対応する GPU 世代」と §7.6 にある。
- **作るのは導入バッチ**: `setup.bat` はこの仮想環境を作らない。`install-MiniMaxH3.bat` が `install_ltx.ps1` の `Ensure-EngineVenv`（H3 用の呼び出し）で作り（`-SkipVenv` を越える。物体追尾の `.venv-utils` と同じ形）、そのあと重みを取る。導入バッチの位置づけは MULTI_ENGINE_DESIGN §6.2。`setup.bat` の再実行では当て直されないので、freeze が変わったときは `install-MiniMaxH3.bat` を再実行する。
- **ライセンスの注意【オーナー裁定：裁定 36・41】**: `install_model.ps1` は、記述子の `id` が `MiniMaxH3` のときだけ、取得の前にライセンスの注意文（名称・利用できる地域・取得者の自己責任）と本文の URL を表示する。表示するだけで、同意の入力は求めない。記述子は `MiniMaxAI/MiniMax-H3` の `LICENSE` も取得して `models/MiniMaxH3/LICENSE` に置く（検査の行の鍵は `h3_license`。サーバーは読まないが、写しを手元に置くための行なので消さない）。NOTICE は作らない。利用者向けの文面は README の「ライセンス」節である。
- **容量が増える**: torch の版が既存の仮想環境と違うため、uv のハードリンク共有（MULTI_ENGINE_DESIGN §5.3「容量は心配しなくてよい」）が効かず、この仮想環境ぶんの数 GB がそのまま増える。実測値は freeze と記録簿を見る。必要空き容量の正本は README のハードウェア要件表である。
- **入れないもの**: torchaudio（CUDA 13 の Windows 版が torch の新しい版と組めず、H3 の経路は使わない）、diffsynth（部品は移植した）。
- **版の正本は `engine_h3/venv-engine-h3.freeze.txt`** である。手で作り直すときの仕様と、各版を選んだ理由は `engine_h3/engine_h3-venv-pyproject.toml` に書く。

---

## 11. 弾の分け方と門

### 弾の分け方

| 弾 | 範囲 |
|---|---|
| 第 1 弾（完了。記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §165） | 仮想環境・重みの取得・`engine_h3/` と `h3_format.py`・T2V セルフテスト 1 本・本書 |
| 第 2 弾（完了。記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §166） | I2V（先頭フレーム）のセルフテスト → アダプタ・系統登録・記述子・ヘッダ判定（`h3_format.py` をアプリ側の事前検査から呼ぶ）・API の制約（17n+5・24fps 固定・32 の倍数）・API にステップ数と shift（映像・音声）の 3 項目・操作パネル（ドロップダウンは自動追随・フレーム数のスナップ・快適上限マーカーの非表示・`featureScope.ts`・ステップ数／Shift（映像）／Shift（音声）の 3 本のスライダー〔裁定 22〕）・Gradio の同じ 3 本・MCP の最小対応・テスト・常駐 |
| 第 3 弾（完了。サブマシンの門も合格。記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §167） | `install-MiniMaxH3.bat`（仮想環境の構築と取得を 1 回で）・`Ensure-EngineVenv` の H3 用の呼び出し・インストーラで作った環境と freeze の照合・LICENSE の写しの取得とライセンスの注意文・README（導入手順・要件〔ドライバ・GPU 世代・主メモリ・容量〕・α版の制限〔§7.6〕・ライセンス） |
| VRAM 16GB mode（台帳 §1-101 の第 2 段。α版の後の最初の改修〔台帳では β版以降の項目〕。完了（B-G5 のオーナーの目視まで合格。台帳 CLOSED 3-240）。記録は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169） | ＝エンジンの on の経路・API の `vram16_mode`・アダプタ・Gradio・MCP・操作パネルの Settings の行・OOM の案内（§6.5） |
| 以後 | 台帳 [`PENDING_TASKS.md`](PENDING_TASKS.md) §1 の MiniMax H3 の項目 |

- 快適上限は、較正が済むまで記述子に `comfort` の行を持たせず、操作パネルのマーカーを出さない。

### 門

各弾の門の内容と結果は記録簿にある（第 1 弾は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §165.5、第 2 弾は同 §166.3〜§166.5・§166.9、第 3 弾は同 §167.3・§167.4・§167.8、VRAM 16GB mode は同 §169.3〜§169.5）。GPU を使う門は同時に 1 体だけ回し、クローズはオーナーの判断である。

---

## 12. 書かないもの・後回し

次は書かない（後回し）。

- LoRA・Turbo（裁定 10）
- Ref2VA・Hybrid（FL2VA と Ref2VA を混ぜた配布物）。ただしオーナーが PinkCherry Hybrid を置いて生成に成功した（実機の事実・2026-10-18）。構造が同じことの根拠は研究ノート §13.3（ComfyUI のコードの読解であり、実測ではない）。対応の保証は §12 のとおり後回しである（§13）。
- 末尾キーフレームと中間キーフレーム（台帳 §1-97）
- 連結・画角拡張・NAG／VSF

量子化の旧式の記述（`_quantization_metadata`）・VSA の追加テンソル・PDD ヘッドは、警告して捨てるか拒否する（§7）。

---

## 13. 未決事項

- **性能**: 大きな画角でノイズ除去の段の VRAM が溢れる問題は、VRAM 16GB mode（§6.5）の on で消えた（開発機の機械の門。製品の条件で退避ゼロ・1 ステップ約 22 秒。実測は [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) §169.3）。配置後のオーナーの目視（B-G5）にも合格し、sage の 2 ブロックの値の差は採用した（裁定 53）。残るのは次の 2 点である。(1) 計測台の T1 で重い減速が出なかった理由は未解明のまま残している（オーナー裁定で追わない。同 §168.9）。(2) 快適上限の較正（台帳 §1-95）は、退避そのものの量で線を引く（同 §168.1）。溢れが消えたので、較正の遅さを理由にした LoRA（Turbo）対応の優先順位の引き上げ（研究ノート §10 の裁定 21）が要るかは、較正（§1-95）の計画のときに確かめる。
- **PinkCherry Hybrid**: オーナーが PinkCherry Hybrid を置いて生成に成功した（実機の事実・2026-10-18）。構造が同じことの根拠は研究ノート §13.3（ComfyUI のコードの読解であり、実測ではない）。対応の保証は §12 のとおり後回し（Ref2VA の参照経路も後回し）。
- **m×n ≥ 2³¹ の CUDA 経路**: ソースの水準では限界が無いが、実測は台帳 §1-95（快適上限の較正）で行う。VRAM 16GB mode の on では行の塊で分割されるので届かず、off でだけ届きうる（算術による推測。扱いは §1-95 の計画で決める）。
- **ヘッダの揺れのうち後回しにしたもの**: コミュニティ版の読み込み、非枝刈り版、groupsize の混在への対応範囲。
