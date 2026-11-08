# MiniMax H3 のネガティブプロンプト（VSF）の門と目視の実験

MiniMax H3 にネガティブプロンプトを VSF（Value Sign Flip。負のプロンプトの行の値の符号を反転し、1 回の注意の計算で負の要素を遠ざける方式）で載せた最小実装について、機械で確かめる門 4 つと、オーナーが目で見て効き目を判断する 10 本の動画を作るための台本と記録の置き場です（台帳 `PENDING_TASKS.md` §1-127）。計画の正本は `C:\Users\phenotype nezu\.claude\plans\agile-dazzling-catmull.md` の §4（機械の門）と §5（目視の実験）です。結果の数値はこの説明書には書かず、各フォルダの `result.json`・`summary.json`・`VISUAL_TABLE.md` に残します。

## フォルダの中身

| ファイル・フォルダ | 役目 |
|---|---|
| `preconditions.py` | 回す前と後の点検（git の状態・GPU の使用量・python のプロセス・ワーカーの残り・AviUtl2 の有無）。`preconditions_before.txt`／`preconditions_after.txt` に書く |
| `g0_on.cmd`・`g0_off.cmd` | 門 G0（VSF を使わないときに絵が 1 ビットも変わらないこと）。VRAM 16GB mode の on と off の 2 本 |
| `g1_on.cmd`・`g1_off.cmd` | 門 G1（VSF を使ったとき、VRAM 16GB mode の on と off で同じ絵と同じ計測値になること） |
| `h3_vsf_gate.py` | 門 G2・G3 と目視の実験を、別のポートに起こした実際のバックエンドへ MCP サーバー経由で送る台本 |
| `make_visual_sheet.py` | 目視の実験の後に、10 本の表（`VISUAL_TABLE.md`）を記録から書き起こす補助 |
| `g0_on\`・`g0_off\`・`g1_on\`・`g1_off\` | 各門の動画・報告の JSON・画面の出力（cmd を回すとできる） |
| `g2\`・`g3\` | 門 G2・G3 の `result.json` と、ジョブごとの `request.json`・`metadata.json`・動画 |
| `visual\T0`〜`visual\B4` | 目視の実験の 1 本ごとの `request.json`（送った引数）・`metadata.json`・動画・ワーカーのログの抜き書き |
| `summary.json` | 台本が回した全ジョブの条件・秒・指紋（sha256）・VSF の記録・合否。回すたびに、同じ名前のジョブは上書き、他は残る |
| `logs\` | MCP の呼び出しの控え（全文）と MCP サーバーの標準エラー |
| `_rt\` | 一時サーバーの設定・状態・ログ・出力・アップロードの置き場。**保管しない**（終わったら消してよい。git に入れない） |

## 実行の記録

2026-11-09 に回しました。門 G0・G1 は 00:5x〜01:12（動画の完了は g0_on 01:07・g0_off 01:09・g1_on 01:11・g1_off 01:12）、G2 は 01:20〜01:21、G3 は 01:26 に完了、目視の 10 本は T 系列が 01:27〜、B 系列が 01:34〜で 01:51 までに完了しました（`logs\` の MCP の呼び出しの控えの時刻による）。

## 門と合否

### G0: VSF を使わないときの回帰

条件は過去の門 G-I5（`Docs\Outputs-archive\h3-te-vram-2026-10-29\g_i5_t2v_fingerprint.cmd`）と同じです。FL2VA の重み・黄色いレインコートの女性のプロンプト・640×384・124 フレーム・20 ステップ・seed 42・shift 12/3・sdpa。元の cmd は VRAM 16GB mode が on に固定され、報告の書き先が過去の記録に固定されているので、書き先だけをこのフォルダに向けた写しを 2 本作りました（`g0_off.cmd` は `--vram16` を `--no-vram16` に替えただけ）。

合格: 両方とも終了コード 0 で、動画の指紋が `0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9`。

| 本 | 指紋（先頭） | 合否 |
|---|---|---|
| g0_on（VRAM 16GB mode on） | `0da6111c…` | 合格（01:07 完了） |
| g0_off（VRAM 16GB mode off） | `0da6111c…` | 合格（01:09 完了） |

### G1: VSF を使ったときの on と off の一致

G0 の形を小さくして（512×288・56 フレーム）、負のプロンプト `blurry, low quality` と倍率 3 を足した 2 本です（sdpa）。`--negative-prompt` と `--vsf-scale` は今回の実装でワーカーの自己試験に足される引数です。

合格: 2 本の動画の指紋が同じで、報告の `rounds[0].done.h3.vsf`（`rows`・`scale`・`m_neg`・`m_text`）も同じ。

| 本 | 指紋（先頭） | m_neg | m_text | 合否 |
|---|---|---|---|---|
| g1_on | `8cd9702c…` | 0.004364 | 0.010767 | 合格（01:11 完了） |
| g1_off | `8cd9702c…` | 0.004364 | 0.010767 | 合格（01:12 完了。指紋・`rows`＝[41, 46]・`scale`＝3.0・m の値とも on と一致） |

比べ方（リポジトリ直下で）:

```
.venv\Scripts\python.exe -c "import json;a,b=[json.load(open(rf'Docs\Outputs-archive\h3-vsf-2026-11-09\{n}\{n}.json',encoding='utf-8'))['rounds'][0] for n in ('g1_on','g1_off')];print(a['sha256']==b['sha256'],a['done'].get('h3',{}).get('vsf'),b['done'].get('h3',{}).get('vsf'))"
```

### G2: 受理と拒否（`h3_vsf_gate.py --mode g2`）

| 確かめること | 期待 | 合否 |
|---|---|---|
| ① `neg_method="nag"`（既定値）で送る | 断られる。文に `FEATURE_UNSUPPORTED` と機能名 `nag` | 合格（「MiniMax H3 のネガティブプロンプトは VSF 方式だけです」と案内して断られた） |
| ② `neg_method="vsf"`・倍率 3・4 ステップで送る | 受理されて完了し、`metadata.json` の `h3.vsf` に `rows`・`scale`・`m_neg`・`m_text` の 4 つ | 合格（rows＝[41, 43]・scale＝3.0・m_neg＝0.001272・m_text＝0.003671） |
| ③ `list_models` の MiniMax H3 の `unsupported_features` | `nag` が残っている（操作パネルの灰色の表示は今回は変えないため） | 合格（`nag` が残っている） |
| ④ `submit_chain`（連結生成）に `neg_method="vsf"` で送る | 断られる。文に `FEATURE_UNSUPPORTED` と `nag`（連結生成は方式に関係なく断る） | 合格（「連結生成（Chained）は nag に対応していません」と断られた） |

台本は ③ → ① → ④ → ② の順に回します（断られるはずのものを先に済ませ、動画を作るのは最後の 1 本だけ）。断られるはずのものが万一受理されたときは、すぐ取り消して不合格にします。結果は `g2\result.json`（4 つとも合格、01:20〜01:21）。

### G3: VSF のジョブの直後に持ち越しが無いこと（`h3_vsf_gate.py --mode g3`）

同じサーバーで、① VSF のジョブ（G0 と同じ形・負のプロンプト `blurry, low quality`・倍率 3・`keep_resident=True`・`vram16_mode=True`・`embed_mp4_metadata=False`）を回し、② その直後に VSF なしの同じジョブを回します。

| 確かめること | 合否 |
|---|---|
| ① が完了し、`metadata.json` の `h3.vsf` に 4 つの鍵がある | 合格 |
| ① のワーカーのログに `neg=vsf` の行がある | 合格 |
| ② の指紋が G0 と同じ `0da6111c…`（前のジョブの反転が残っていない） | 合格 |
| ② の `metadata.json` に `vsf` の鍵が無い | 合格 |
| ② のワーカーのログに `neg=off` の行がある | 合格 |

結果は `g3\result.json`（5 つとも合格、01:26 完了）。VSF のジョブの指紋は `53f034f2…`・`h3.vsf` は rows＝[41, 46]・scale＝3.0・m_neg＝0.001403・m_text＝0.003653。VSF なしのジョブの指紋は G0 と同じ `0da6111c…`。

## 目視の実験の 10 本（`h3_vsf_gate.py --mode visual`）

共通の条件: 1344×768・56 フレーム・24fps・seed 42・注意の計算 sage（普段の設定）・VRAM 16GB mode on・transformer は `default`（FL2VA）・shift は映像 12／音声 3。

- 正のプロンプト: `A Japanese woman in casual clothes walks toward the camera along a city street and smiles, reaching one hand toward the viewer; steady camera, natural daylight.`
- 負のプロンプト: `black hair, long hair, eyeglasses, cars`

Turbo の系列の shift を 12/3 にしたのは、オーナーが 2026-11-08 に操作パネルで回したジョブの `metadata.json` で観測した普段の設定に合わせたためです。Turbo LoRA の配布元が 768p 版に勧めている値（6/3）とは違いますが、「計測の条件は利用者の既定に合わせる」という決まりを優先しました。Turbo の系列で効きが弱かったときに shift の影響を切り分けたくなったら、6/3 の 1 本を足して確かめます（そのときは改めて了承を取ります）。

`m_neg` は、負のプロンプトの行が注意の配分をどれだけ奪ったかの割合、`m_text` は正のプロンプトの行に乗った割合です（ステップ 0・最初のブロックで、映像の行から等間隔に 256 本を選んで平均）。この値は sdpa の計算の定義で求めているので、sage で回した実際のジョブの重みとはわずかに違うことがあります。

10 本の値（m_neg・m_text・負の行・生成の秒・VRAM の山・指紋）は `VISUAL_TABLE.md` を正本とし、ここには写しません（数値を 2 箇所に持たないため）。全 10 本が完了しました。生成は Turbo の 1 本が 78〜98 秒、素の 20 ステップの 1 本が 185〜193 秒、VRAM の予約の山は約 7,600MiB でした。

所要の見込みは Turbo が 1 本約 4 分、素の 20 ステップが 1 本約 11 分でした（実績は上の通り）。回した後に `make_visual_sheet.py` を実行すると、10 本の表 `VISUAL_TABLE.md` ができます（生成の秒と待ちの秒、VRAM の山は予約と割り当ての 2 つ）。

### 比較の対

1 つの対では、違う条件が 1 つだけになるように組んであります。

| 対 | Turbo の系列 | 素の系列 | 見ること | 感想 |
|---|---|---|---|---|
| なし 対 1.5 | T0・T1 | B0・B1 | 効くか | |
| 1.5 対 5 | T1・T3 | B1・B3 | 倍率のつまみが効くか | |
| 5 対 8 | T3・T4 | B3・B4 | 絵の崩れや、逆に負の要素が強まる逆転が出るか | |
| Turbo 対 素 | T0〜T4 | B0〜B4 | ステップ数で効き方が変わるか | |

### 見るもの

- 負に挙げた要素（黒い髪・長い髪・眼鏡・車）が消えるか、減るか。
- 正の文の要素（街路・こちらへ歩く・微笑む・手を伸ばす）が残っているか。
- 強い倍率で絵が崩れないか。
- 逆に、負に挙げた要素がかえって強まる「逆転」（先行事例の作者が報告した現象）が出ないか。

### 監督の予備観察

各本の中間フレームを 1 枚ずつ見た監督の所見（オーナーの目視の代わりではなく、推測を含む）は、記録簿 `Docs/VERIFICATION_LOG.md` §201.8 にだけ書いてある。ここには写さない。

## 台本の使い方

すべてリポジトリ直下（`S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`）から回します。GPU を使う前に、その都度オーナーへ知らせます。AviUtl2 は閉じておきます。

1. 前の点検: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-2026-11-09\preconditions.py before`
   終了コード 2 は AviUtl2 が動いている、3 はワーカーのプロセスが残っている、という意味です（台本は何も止めません）。
2. G0: `Docs\Outputs-archive\h3-vsf-2026-11-09\g0_on.cmd` と `g0_off.cmd`（1 本ずつ・約 2 分半ずつ）。
3. G1: `g1_on.cmd` と `g1_off.cmd`。
4. G2 と G3: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-2026-11-09\h3_vsf_gate.py --mode g2 --keep-server`、続けて `--mode g3`（`--keep-server` を付けた回のサーバーと読み込んだ H3 を次の回がそのまま使います。付けない回の最後にサーバーを止めます）。
5. 目視の実験: `h3_vsf_gate.py --mode visual --series T`（オーナーへ知らせてから）、続けて `--series B`。
6. 表: `.venv\Scripts\python.exe Docs\Outputs-archive\h3-vsf-2026-11-09\make_visual_sheet.py`
7. 後の点検: `preconditions.py after`

`h3_vsf_gate.py` の引数:

| 引数 | 既定 | 意味 |
|---|---|---|
| `--mode` | （必須） | `g2`・`g3`・`visual`・`all`（all は g2 → g3 → visual を 1 つのサーバーで続けて回す） |
| `--port` | 18699 | 一時サーバーのポート。普段のサーバー（18620）とは別 |
| `--keep-server` | 切 | 終わりに H3 を降ろさず、サーバーも止めない。次の回は同じポートのサーバーを使い回す（一時の設定ファイルの名前で起こしたサーバーだけ。別のサーバーが応答していたら止まる） |
| `--series` | `both` | 目視の実験の系列。`T`＝Turbo の 5 本、`B`＝素の 5 本、`both`＝T の後に B |
| `--load-timeout-sec` | 1200 | H3 の読み込みを待つ上限 |
| `--job-timeout-sec` | 3600 | 1 本のジョブを待つ上限。超えたら残りは送らない |
| `--dry-run` | 切 | 送る引数を表示して終わる（サーバーも MCP サーバーも起こさない） |

台本の動き:

- `config.yaml` は読むだけです。ポート・ログの置き場・出力の置き場・アップロードの置き場・状態ファイルの 5 つだけを `_rt\` に向けた一時の設定ファイル（`_rt\config.h3vsf.yaml`）と、MiniMax H3 を選んだ状態ファイル（`_rt\state.h3vsf.json`）を書き、`.venv\Scripts\python.exe main.py --config _rt\config.h3vsf.yaml` で起こします。オーナーの `state.json`・`logs\`・`outputs\` には書きません。
- 起動の環境は普段の起動（`run.ps1`）に合わせ、`tools\uv` と `tools\ffmpeg\bin` を PATH の先頭に足します。
- MCP サーバーには `LTX_MCP_BASE_URL`（一時サーバーの URL）に加えて `LTX_MCP_CONFIG`（一時の設定ファイル）と `LTX_MCP_OUTPUT_DIR`（`_rt\outputs`）を渡します。MCP サーバーは動画と `metadata.json` を自分の設定の出力先から探すので、これを渡さないと普段の `outputs\` を探してしまいます。
- 起動の後、`tools/list` の引数の定義で、台本が渡す引数名がすべてあるかを照合します（無ければ終了コード 1）。
- H3 の読み込みは `load_pipeline` を 1 回だけ呼び、あとは `backend_status` を 10 秒おきに見て待ちます（読み込み中に呼び直すと 409 で断られるため）。
- 1 本が失敗しても先へ進み、`summary.json` に `error` を残します。道具が返したエラーの文は全文を残します（`summary.json` と `logs\mcp_calls_<日時>.json`）。
- 終了コード: 0＝回した門がすべて合格（目視の実験はすべて完了）、3＝どれかが不合格か失敗、1＝準備の段で止まった。
- サーバーの止め方: H3 を降ろし（`unload_pipeline`）、サーバーを止め、止まらなければプロセスの木ごと強制終了します。最後にサーバーの子孫（ワーカー）が残っていれば、それも止めます。
