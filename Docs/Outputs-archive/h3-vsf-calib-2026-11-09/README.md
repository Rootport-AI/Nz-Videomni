# MiniMax H3 の VSF の実用域の較正（18 本）

MiniMax H3 のネガティブプロンプトの強さ（scale）の既定値と、操作パネルのスライダーの最大値を決めるために、条件を変えた 18 本の動画を作ってオーナーが目で見るための、台本と記録の置き場です。ネガティブプロンプトの方式は VSF（Value Sign Flip。負のプロンプトの行の値の符号を反転し、1 回の注意の計算で負に挙げた要素を遠ざける方式）です。台帳 `PENDING_TASKS.md` §3-278、記録簿 `VERIFICATION_LOG.md` §203。計画の正本は `C:\Users\phenotype nezu\.claude\plans\agile-dazzling-catmull.md` の §3（18 本の表・見るもの・決め方）と §4（作り方）です。

18 本の条件の正本は `calib_plan.json` で、この説明書の表はその写しです。結果の数値はこの説明書には書かず、`summary.json`・`jobs\<名前>\metadata.json`・`VISUAL_TABLE.md` に残します。

## フォルダの中身

| ファイル・フォルダ | 役目 |
|---|---|
| `calib_plan.json` | 18 本の条件（名前・段・seed・寸法・フレーム・負の文・scale・目的）と共通の引数。生成しない対照 `A_off` も持つ |
| `h3_vsf_calib.py` | 18 本を、別のポートに起こした実際のバックエンドへ MCP サーバー経由で送る台本 |
| `make_calib_sheet.py` | 回した後に、表（`VISUAL_TABLE.md`）・段ごとの一覧の画像（`sheets\`）・Hugging Face に上げるフォルダを作る補助（GPU もサーバーも使わない） |
| `preconditions.py` | 回す前と後の点検（git の状態・GPU の使用量・python のプロセス・ワーカーの残り・AviUtl2 の有無）。`preconditions_before.txt`／`preconditions_after.txt` に書く |
| `jobs\<名前>\` | 1 本ごとの `request.json`（送った引数）・`metadata.json`・動画（ジョブの番号の名前）・`worker_log_excerpt.txt`（ワーカーのログの抜き書き）。`jobs\A_off\` は第 1 弾の T0 の写し |
| `summary.json` | 台本が回した全本の段・scale・seed・寸法・負の文・状態・秒・VRAM の山・VSF の記録・sha256・エラー。回すたびに、同じ名前の本は上書き、他は残る |
| `VISUAL_TABLE.md`・`sheets\` | `make_calib_sheet.py` が書く表と一覧の画像 |
| `logs\` | MCP の呼び出しの控え（全文）と MCP サーバーの標準エラー |
| `_rt\` | 一時サーバーの設定・状態・ログ・出力・アップロードの置き場。**保管しない**（終わったら消してよい。git に入れない） |

動画（mp4）はリポジトリに入れません。

## 共通の条件

- 正のプロンプト（第 1 弾と同じ）: `A Japanese woman in casual clothes walks toward the camera along a city street and smiles, reaching one hand toward the viewer; steady camera, natural daylight.`
- Turbo LoRA `minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16` 強度 0.8・4 ステップ・shift は映像 12／音声 3・注意の計算 sage・VRAM 16GB mode on・24fps・transformer は `default`（FL2VA）。
- 生成する本はすべて `nag_enabled=true`・`neg_method="vsf"`・負の文・`vsf_scale`（0 を含む）を送る。`embed_mp4_metadata` と `keep_resident` は送らない（既定のまま。第 1 弾の T0 と同じ）。
- 負の文 N1 = `blue clothes, cars`（T0 に写っている 2 つの要素）、N2 = `blue denim shirt, blue jacket, blue clothing, white car, vehicle, automobile`（N1 を同義語で厚くした言い換え）。
- scale 0 は対照: 負の文の行を 1.5 や 3 と同じ配置で置き、反転した値だけをゼロにする。負の文なしの対照は T0 を `A_off` として流用する（生成しない）。

## 18 本の表

| 段 | 名前 | seed | 寸法×フレーム | 負の文 | scale | 目的 |
|---|---|---|---|---|---|---|
| A | `A_s0`・`A_s1`・`A_s1.5`・`A_s2`・`A_s2.5`・`A_s3`・`A_s4`・`A_s5` | 42 | 1344×768×56 | N1 | 0／1／1.5／2／2.5／3／4／5 | 既定値の帯（消える最小の scale）と、場面が保たれる上限 |
| A′ | `R_s0`・`R_s1`・`R_s1.5`・`R_s2` | 43 | 1344×768×56 | N1 | 0／1／1.5／2 | 既定値の候補の帯を 2 つ目の seed で確かめる |
| B | `B_s6`・`B_s8` | 42 | 1344×768×56 | N1 | 6／8 | 最大値の上: 崩れ・逆転の位置 |
| C | `C_s1.5_N2` | 42 | 1344×768×56 | N2 | 1.5 | 言い換えと、数値を上げることを押す力（scale × m_neg）をそろえて比べる。相手は実測の m から、押す力の目安が近い `A_s2`・`A_s2.5` に決めた（計画の当初の見込みは N1 の 4〜5。値は `VISUAL_TABLE.md`） |
| D | `L_s0`・`L_s1.5`・`L_s3` | 42 | 1344×768×124 | N1 | 0／1.5／3 | 系列が長いと効きが弱まるか |
| （対照） | `A_off` | 42 | 1344×768×56 | なし | off | 第 1 弾の T0 の写し（生成しない） |

回す順は A → B → A′ → C → D（台本の `--stage` では A′ を `R`、長い尺を `D` と書く）。1 本 約 1.5 分、段 D は 1 本 約 5 分の見込み。

## 確かめること

この較正には合否の門はありません。判断はオーナーの目視です。機械で確かめるのは次のことだけです。

- `summary.json` で 18 本が completed。
- scale 0 を含む生成した全本の `metadata.json` に `h3.vsf`（`rows`・`scale`・`m_neg`・`m_text`）がある。
- seed・寸法・フレーム・負の文が同じ本どうしで、`rows` と m の値が scale によらず同じ（反転した値に掛ける倍率に依らない値なので）。
- 表と一覧の画像が揃う。

結果: 4 点とも満たしました（2026-11-09）。同じ組は 3 組（seed 42 の 56 フレームの N1・seed 43 の N1・124 フレームの N1）です。値は `VISUAL_TABLE.md` にあります。

## 台本の使い方

すべてリポジトリ直下（`S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`）から回します。GPU を使う前に、その都度オーナーへ知らせます。AviUtl2 は閉じておきます。

1. 乾いた確認（GPU もサーバーも使わない）:
   - `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-calib-2026-11-09\h3_vsf_calib.py --selfcheck`（`calib_plan.json` を計画書 §3 の表と突き合わせ、`A_off` の元を確かめる。すべて PASS で終了コード 0）
   - `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-calib-2026-11-09\h3_vsf_calib.py --dry-run --stage all`（送る 18 本の引数を表示）
2. 前の点検: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-calib-2026-11-09\preconditions.py before`
   終了コード 2 は AviUtl2 が動いている、3 はワーカーのプロセスが残っている、という意味です（台本は何も止めません）。
3. 段 A: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-calib-2026-11-09\h3_vsf_calib.py --stage A --keep-server`
   （最初の回で `A_off` を `jobs\A_off\` に写します。`--keep-server` を付けた回のサーバーと読み込んだ H3 を次の回がそのまま使います）
4. 段 A の表と画像と Hugging Face のフォルダ: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-calib-2026-11-09\make_calib_sheet.py --hf-dir outputs\h3-vsf-calib-2026-11-09\hf-round12`
   上げる: `.venv\Scripts\python.exe -m huggingface_hub.cli.hf upload Rootport/Nz-LTX23-eval-clips outputs\h3-vsf-calib-2026-11-09\hf-round12 round12-h3-vsf-calib --repo-type dataset`
5. 残り: `h3_vsf_calib.py --stage B --keep-server`、`--stage R --keep-server`、`--stage C --keep-server`、`--stage D`（最後の回は `--keep-server` を付けず、終わりに H3 を降ろしてサーバーを止める）。
   1 回で済ませるなら `h3_vsf_calib.py --stage all`（A → B → R → C → D）。
6. 表・画像・Hugging Face のフォルダを作り直して追加で上げる（手順 4 と同じ命令。既存のファイルは消さず、足りない本を足します）。Hugging Face のフォルダは mp4 を持つので、git が無視する `outputs\` の下に作ります（第 1 弾の動画と同じく、mp4 はリポジトリに入れない）。
7. 後の点検: `preconditions.py after`

`h3_vsf_calib.py` の引数:

| 引数 | 既定 | 意味 |
|---|---|---|
| `--stage` | `all` | `A`・`B`・`R`（計画書の A′）・`C`・`D`（長い尺）・`all`（A → B → R → C → D） |
| `--only` | なし | 名前をカンマ区切りで（例 `A_s0,A_s1.5`）。`--stage` と重ねて絞る |
| `--port` | 18699 | 一時サーバーのポート。普段のサーバー（18620）とは別 |
| `--keep-server` | 切 | 終わりに H3 を降ろさず、サーバーも止めない。次の回は同じポートのサーバーを使い回す（一時の設定ファイル `config.h3vsfcalib.yaml` で起こしたサーバーだけ。別のサーバーが応答していたら止まる） |
| `--load-timeout-sec` | 1200 | H3 の読み込みを待つ上限 |
| `--job-timeout-sec` | 3600 | 1 本を待つ上限。超えたら残りは送らない（待ちきれなかった本がバックエンドで走り続けるため） |
| `--dry-run` | 切 | 計画と送る引数を表示して終わる（サーバーも MCP サーバーも起こさず、何も写さない） |
| `--selfcheck` | 切 | 計画の突き合わせだけをして終わる（すべて PASS で 0、どれかが FAIL で 1） |

終了コード: 0＝回した本がすべて completed、3＝どれかが失敗（先へは進む）、1＝準備の段で止まった（サーバーが起きない・道具の引数名が違う・読み込みの失敗など）。

`make_calib_sheet.py` の引数: `--hf-dir <フォルダ>`（Hugging Face に上げるフォルダを作る。mp4 を `<名前>_<幅>x<高さ>_seed<seed>.mp4` の名前で写し、一覧の画像と `README.md` を置く）・`--no-sheets`（表だけ）。`--plan`・`--jobs-dir`・`--summary`・`--out-dir` は試しに別の入力や書き先へ向けるためのもの。動画がまだ無い本は、表では「未実行」、一覧の画像では行を省きます。

## 実行の記録

2026-11-09 に、dev `e3389f6` のコードで 2 回に分けて回しました（GPU の使用は 18 本を 1 つの段として了承を取った）。前の点検 `preconditions_before.txt` は問題なし（終了コード 0）です。後の点検（`preconditions_after.txt`）は作っていません。

| 回 | 時刻 | 中身 | 終了コード |
|---|---|---|---|
| 1 回目 `h3_vsf_calib.py --stage A --keep-server` | 11:02〜11:16 | `A_off` の写し・一時サーバーの起動（ポート 18699。H3 の読み込み 16.3 秒）・段 A の 8 本 | 0 |
| 2 回目（残り） | 11:18〜11:39 | 前の回のサーバーを使い回して残りの 10 本（B → A′ → C → D の順。台本の表示は「段 all・10 本」）。終わりに H3 を降ろしてサーバーを止めた（残っていた子プロセスなし） | 0 |

各回のコンソールのログは `_rt\stage_A.console.log`・`_rt\stage_rest.console.log` にあります（`_rt\` は保管しない）。

**結果の置き場**:

- 1 本ごとの数値（負の行・m_neg・m_text・押す力の目安・生成の秒・VRAM の山・sha256）: `VISUAL_TABLE.md`
- 段ごとの一覧の画像: `sheets\sheet_A.png`・`sheet_R.png`・`sheet_B.png`・`sheet_C.png`・`sheet_D.png`
- Hugging Face: 非公開データセット `Rootport/Nz-LTX23-eval-clips` の `round12-h3-vsf-calib/`（mp4 19 本・一覧の画像 5 枚・README）。段 A の後にコミット `ffc078ed22fb01ad73ede07b2569106677193dc5`（15 ファイル）、全本の後にコミット `24b585339b0dfc991f599bd627e4f2aa3b2d3983`（25 ファイル）で上げました。上げる元は `outputs\h3-vsf-calib-2026-11-09\hf-round12\`（リポジトリの外）です。
- 計測の要約・監督の予備観察・未決: 記録簿 `VERIFICATION_LOG.md` §203

## オーナーの目視の所見

裁定（オーナー。2026-11-09）の原文: 「VSFの規定値や最大値、入力可能な刻みなどは現状のままで良さそうだ。君の言うとおり、LTXとH3で分ける理由もない。H3側で必要な改修は、NAGを選択できないようにすることだけだと思う。」

- 既定値・スライダーの最大値・刻み・ヒント文は現状のままで、LTX と H3 で分けません。長い尺の再試験もしません。
- 詳細は記録簿 `Docs/VERIFICATION_LOG.md` §203.11、台帳は `Docs/PENDING_TASKS_CLOSED.md` 3-278。
