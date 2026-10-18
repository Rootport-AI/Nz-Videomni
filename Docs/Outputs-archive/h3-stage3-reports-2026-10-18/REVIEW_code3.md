# REVIEW_code3 — MiniMax H3 第 3 弾 インストーラ（C1）のコードの事実確認レビュー

対象: `Nz-Videomni`（dev）の未コミット差分のうち `install-MiniMaxH3.bat`（新規）・`scripts/install_ltx.ps1`・`scripts/install_model.ps1`・`scripts/manifests/40-minimaxh3.json`・`Docs/Outputs-archive/h3-gate3/`。読むだけ（リポジトリは編集していない）。実行したのは、バイト走査（Python）・PowerShell 5.1 の `ParseInput`（明示 UTF-8）・アプリ側のテスト 4 本（`.venv`）だけ。

重さ: Critical＝既存の導入経路を壊す／サブマシンの門に合格できない。Major＝直さないと設計の判断や利用者の行動を誤らせる。Minor＝文言・細部・記録。

## 0. 結論

- **Critical 0・Major 0・Minor 7。**
- 計画（§3.1・裁定 36・41・§7）との食い違いは見つからなかった。`setup.bat`・`install-LTX25.bat`・`install-UETrack.bat` の挙動を変える変更も見つからなかった（根拠は §B）。
- Minor は、古くなったコメント、記録（`h3-gate3/README.md`）の書き方、更新時の再実行の案内の細部である。

## A. 計画との一致（事実の照合。誤りなし）

| 項目 | 結果 | 根拠 |
|---|---|---|
| bat は UETrack の写し | 一致 | `diff install-UETrack.bat install-MiniMaxH3.bat` の差は冒頭コメント 2 か所・`title`・`-BaseModel MiniMaxH3` だけ |
| 呼び出しの条件 | `if ($foundIds -contains 'MiniMaxH3')` | `install_ltx.ps1:1549` |
| `-IgnoreSkipVenv` | あり | `:1556` |
| freeze のパス | `engine_h3\venv-engine-h3.freeze.txt` | `:1296` |
| stage (b) の index | `-IndexUrl https://download.pytorch.org/whl/cu130` | `:1554` |
| 直接ピン 3 本 | `torch==2.13.0+cu130`・`torchvision==0.28.0+cu130`・SageAttention の URL | `:1306-1310`。URL は `engine_h3/engine_h3-venv-pyproject.toml:72` と Python で文字列比較して完全一致。単一引用符。freeze 側の 3 行（`:74`・`:79`・`:80`）も同じ版 |
| `-DirectPinArgs` | `--index …/cu130`＋`--index-strategy unsafe-best-match` の 2 つ・`--no-sources` なし | `:1315-1318`。LTX 2.5 の `$ltx25UvArgs`（`:1262-1266`）は `--no-sources` を含む 3 つなので、「`--no-sources` を除いて同じ 2 つ」が正確 |
| stage (b) の除外 | `^(torch|torchvision|sageattention)==` が直接ピンから作られる | `:1385-1386`。freeze の 3 行がこれで除かれる |
| 検証の表の 2 行 | `h3_python`・`engine_h3 worker.py` | `:1854-1857`。UETrack の 2 行（`:1849-1852`）の後 |
| uv の環境変数 | 追加なし | `UV_HTTP*`・`UV_CONCURRENT*` は `scripts/*.ps1`・bat の中に 0 件 |
| `Ensure-EngineVenv`・`Get-EngineStateHash`・`Invoke-EngineFreezeApply` の本体 | 不変 | `git diff` のコメント以外の差分行は変数 4 つ・配列 2 つ・`if` のブロック 2 つだけ。関数の中の差はコメント行だけ |
| `install_model.ps1` の文面 3 テーマ | 「重みだけ」→`:349`・`:200`・`:22`・`:157`／「1〜2 時間」→`:351`／「例外は UETrack だけ」→`:157-158`・`:389-391` | 差分どおり |
| H3 だけの注意文 | `if ($plan.Id -eq 'MiniMaxH3')`（`:358`）。名称・地域・自己責任・URL・写しの保存先。同意の入力なし | `Read-Host`・`ReadKey` は 0 件。地域の文言は I2 ①（EU・英国・韓国・米国が除外）と一致 |
| 失敗時の 1 行 | 共通の案内に 1 行（`:271-272`） | 条件分岐は増えていない |
| 記述子の `_note` | リビジョン 2 つ（I2 の値と一致）・固定しない・tokenizer を 1 ファイルずつにする理由・LICENSE を独立項目にする理由 | `40-minimaxh3.json:6` |
| LICENSE の 3 番目の項目 | `repo` MiniMaxAI/MiniMax-H3・`include ["LICENSE"]`・`map LICENSE→MiniMaxH3/LICENSE`・`key`／`label` `h3_license`・`min` 16000（実物 17,604B の約 9% 下） | `:115-129`。fnmatch の `LICENSE` は `FL2VA/…` 配下に一致しないので、ほかのファイルは落ちてこない |
| tokenizer の `include` | 1 ファイルずつ 6 本のまま | 記述子を読み込んで確認 |

## B. 既存の導入経路を壊さないか（事実）

1. **`setup.bat` では H3 のブロックは走らない。** `setup.ps1` は `install_ltx.ps1` を `-BaseModel` なしで呼ぶ（`setup.ps1:103` のコメント・`:407` からの呼び出し）。既定値は `@('LTX23','Preprocessors')`（`install_ltx.ps1:141`）。`$foundIds` は `$TargetManifests`（＝`-BaseModel` に挙がった id の記述子）の id だけでできている（`:504-506`）。だから、`MiniMaxH3` が入るのは `-BaseModel` に `MiniMaxH3` を渡したときだけ。bat 経由ならそれは `install-MiniMaxH3.bat` だけである（`install_model.ps1:392` は `-BaseModel $plan.Id` を 1 つだけ渡す）。
2. **`install-LTX25.bat`・`install-UETrack.bat`**: `$foundIds` はそれぞれ `LTX25`・`UETrack` になる。H3 の venv のブロックも、検証の表の 2 行も追加されない。検証の表の行数と順序は変わらない（固定 3 行 → UETrack の 2 行〔UETrack のときだけ〕→ `$TargetManifests` の行。H3 の記述子は対象外）。
3. **すべての経路に効く変更は 2 つだけ。**
   - (a) 記述子の検証は、`-BaseModel` に関係なく全記述子に対して走る（`:483-487` のコメント）。新しい LICENSE の項目がもし検証に通らなければ、`setup.bat` も止まる。これは 3-G1 の 1 回目のログで「5 manifest(s) validated」と出ていて、通っている（事実）。
   - (b) `INSTALLED_PATHS.txt` は `-BaseModel` で絞らず、全記述子の `key` 付きの行のうち、実在するファイルだけを書く（`:1876-1893`）。H3 を導入していない機械では `models/MiniMaxH3/LICENSE` が無いので、出力は変わらない。H3 を導入済みの機械では、`setup.bat` を含むどの経路でも `h3_license` の 1 行が増える（計画 §3.1 の想定どおり）。
4. アプリ側のテスト（`test_model_registry`・`test_models_endpoint_compat`・`test_comfort_budgets`・`test_h3_format`）を `.venv` で実行して 80 passed（事実。このレビューで実行）。

## C. 文字と改行（事実。Python でバイト走査）

| ファイル | BOM | 改行 | 非 ASCII | 構文検査 |
|---|---|---|---|---|
| `install-MiniMaxH3.bat` | なし | CRLF 25／LF 25（素の LF・CR なし） | 0 | — |
| `scripts/install_ltx.ps1` | なし | CRLF 1985／LF 1985 | **0** | `ParseInput`＋明示 UTF-8 で 0 エラー（PS 5.1.26100） |
| `scripts/install_model.ps1` | **あり** | CRLF 447／LF 447 | あり（日本語） | 0 エラー |
| `scripts/manifests/40-minimaxh3.json` | なし（変更前から） | CRLF | あり（`_note`） | `install_ltx.ps1:294` は `ReadAllText(…, UTF8)`、`install_model.ps1:91` は `-Encoding UTF8` で読むので BOM なしでも問題ない |

- リポジトリ内の実体は LF（`git show HEAD:` で確認）で、作業ツリーの CRLF は `core.autocrlf=true` による。bat は `.gitattributes` で `eol=crlf`。

## D. ログの `NativeCommandError`

- **判断: 利用者がダブルクリックで実行したときは出ない（ログからの推論。根拠は事実）。**
- 根拠（事実）:
  1. エラーの記録の位置情報は `At line:1 char:382 + ... = Get-Date; & ".\install-MiniMaxH3.bat" *> "outputs\h3-gate3\g1_run1_ ...` である。これは C1 が外側の PowerShell で打った 1 行のコマンドで、インストーラの中のスクリプトの行ではない。外側の PowerShell が、bat（とその子プロセス）の標準エラー出力を `*>` で受け取ったときに、最初の 1 行（uv の `Using CPython 3.12.9`）を ErrorRecord に包んだ。
  2. 内側の `install_ltx.ps1` は `$ErrorActionPreference = "Stop"`（`:144`）である。もし内側で包まれていたら、その場で止まったはずである。実際は続行して終了コード 0 だった。内側の `uv venv`（`:1468`）は出力をリダイレクトしていないので、標準エラー出力は PowerShell を通らず、コンソールへ直接出る。
  3. 同じ呼び方（リダイレクトなしの `uv venv`）は、`setup.bat` の `.venv-engine` と `.venv-engine-ltx25`、`install-UETrack.bat` の `.venv-utils` で以前から使われている。サブマシンの門（§82）でも問題になっていない。
- 計画 §7 の「`2>&1` の罠」は、内側で `2>&1` を付けたときの話である。今回の差分は `2>&1` を足していない（差分のコメント以外の行を確認）。
- 未確認: ダブルクリックでの実行は、このレビューでは行っていない（インストーラの実行は禁止）。3-G3（サブマシン）で、画面に赤い `NativeCommandError` が出ないことを見れば確定する。

## E. 指紋の件の記録（`Docs/Outputs-archive/h3-gate3/README.md`）

- 事実の照合: 基準 `Docs/Outputs-archive/h3-gate1/g5_t2v_after_review.json` は `text_tokens` 41・`deterministic: false`・sha `0da6111c…`。`g2_t2v.json`（既定のプロンプト）は `text_tokens` 32・sha `9161e030…`。`g2_t2v_g5prompt.json` は `text_tokens` 41・sha `0da6111c…`。`_SELFTEST_PROMPT`（`engine_h3/worker.py:425-428`）は台所の女性の文で、`g2_selftest_g5prompt.cmd` の `--prompt` は `h3-gate2/g4_i2v_resident.py:43-46` の `PROMPT` と同じ文。**README の記述は正しい。**
- 取り違えを防ぐ書き方としての評価: 太字の注意と台本（`.cmd`）があり、この README を開いた人は取り違えない。ただし、次の担当者が最初に見るのは README ではなく、記録簿の基準の定義である。そこにプロンプトが無いことが、今回の取り違えの根本にある（m-5）。

## F. 指摘

### m-1【Minor】古くなったコメントが `install_ltx.ps1` に 5 か所残っている

- **ファイル:行**: `install_ltx.ps1:94-100`（SYNOPSIS の NOTES）・`:1350-1351`・`:1360-1362`・`:1366-1369`・`:1586-1595`。
- **根拠（事実）**:
  - `:96` と `:1588`: 「sageattention は **BOTH** engine venvs に」「**The SAME** sageattention wheel URL is pinned for BOTH engine venvs」。H3 の venv にも、別の wheel（cu130）の sageattention が入るようになった。`:99-100` は「Blackwell needs an R570+ driver」としか書いていないが、H3 は R580 以上が要る（`:1288-1289` には書いてある）。
  - `:1350`: 「`$DirectPins: $engineDirectPins or $ltx25DirectPins`」。
  - `:1360-1362`: 「`$DirectPinArgs / $DirectPinsLabel exist for the LTX 2.5 stack`」。
  - `:1366-1369`: 「`cu128 by default, the CPU index for .venv-utils`」「`("cu128" … and "cpu" …)`」。cu130 の記述がない。
  - レビュー m-8 は SYNOPSIS・段 5・`-SkipVenv`・`Ensure-EngineVenv`・`Get-DirectChildSize` を挙げていて、C1 はそれらを直した。上の 5 か所は m-8 の列挙から漏れていた箇所である。
- **直し方**: 各所に「H3 は cu130 の別の wheel／`$h3DirectPins`／`$h3UvArgs`」を 1 句ずつ足す。コメントだけなので挙動は変わらない（ASCII のみ）。

### m-2【Minor】`INSTALLED_PATHS.txt` の `key` の意味を説明するコメントと、`h3_license` が合わない

- **ファイル:行**: `install_ltx.ps1:1862-1866` と `40-minimaxh3.json:124-125`。
- **根拠（事実）**: コメントは「`key` を持つ `files[]` の行は、サーバーが名前で解決するパス（記述子の `categories[].default_file` か `assets`）」と書いている。`h3_license` はどちらでもない。既存の記述子で `key` を持たない行は IC-LoRA・DWPose・VDA・UETrack である（記述子を読み込んで確認）。`key` を付けたのは計画 §3.1 の決定（「`INSTALLED_PATHS.txt` に 1 行増える」）どおりである。
- **直し方**: どちらか 1 つ。(a) コメントに「ほかに、手元に置く写し（H3 の LICENSE）も載せる」と 1 句足す。(b) 引き算として `key` を外す。この場合、`INSTALLED_PATHS.txt` には出ない。検証の表と取得の判定には影響しない（`:1856` が `key` の無い行を飛ばす）。計画の決定を変えることになるので、監督の判断。

### m-3【Minor】LICENSE は実質的に「必須」。その扱いを記録に残す

- **ファイル:行**: `40-minimaxh3.json:116-129`、`install_ltx.ps1:1760-1762`・`:1790-1792`・`:1955-1965`。
- **根拠（事実）**: LICENSE の行は `min` 16000 で検証の表に入る。取得が 5 回失敗すれば `throw` で止まり、短ければ `throw`、表で MISSING なら `exit 1` になる。任意扱いにする仕組みは無い。取得の順は重み → tokenizer → LICENSE なので、LICENSE で止まっても重みは取れている。再実行では LICENSE だけを取りに行く。
- **【推測】の危険**: リビジョンを固定していない（計画の決定）。MiniMaxAI が LICENSE を改名・短縮（16,000B 未満）・ゲート化すると、H3 の導入全体が失敗扱いになる。今の取得元は公開・非ゲートで、17,604B（I2）。
- **直し方**: コードは今のままでよい（単純さを優先）。記録 §167 に「LICENSE は必須の行。上流が変わったら記述子の `min` を直す」と 1 行書く。

### m-4【Minor】更新時に再実行するとき、サーバーが H3 の venv を掴んでいると当て直しに失敗する。案内の 1 行が「何も消さずにもう一度」だけ

- **ファイル:行**: `install_model.ps1:271-272`、`install_ltx.ps1:1408-1418`（`Invoke-EngineFreezeApply` の `uv pip install`）。
- **根拠（事実）**: freeze かピンが変わると印のハッシュが変わり、`install-MiniMaxH3.bat` の再実行で当て直しになる（`:1441`・`:1475-1489`）。この点は、文言（`install_ltx.ps1:1541-1544` のコメント、README の該当節）と仕組みの両方で成り立っている。C1 の報告 §「止まった点」2 には、オーナーのサーバーの H3 ワーカーが常駐していたことが書かれている。
- **【推測】**: `run.bat`（サーバー）が `.venv-engine-h3` の Python で H3 ワーカーを動かしたまま当て直すと、Windows のファイルのロックで torch の DLL の置き換えに失敗する見込み。案内どおり「何も消さずにもう一度」を繰り返しても、同じ失敗になる。`setup.bat` と LTX の venv にも同じ性質があり、H3 に固有ではない（`setup.ps1` にも「サーバーを閉じて」の案内は無い）。未確認（実行していない）。
- **直し方**: `:272` の 1 行に「`run.bat` の画面を閉じてから」を足す。条件分岐は増えない。

### m-5【Minor】指紋の基準の定義（記録簿）にプロンプトが無い。h3-gate3 の README だけでは次の取り違えを防げない

- **ファイル:行**: `Docs/VERIFICATION_LOG.md:17109`（§166.2「回帰の基準は第 1 弾の新基準（sdpa・seed 42・640×384・124 フレーム・20 ステップ）の指紋」）・`:17084`。`Docs/Outputs-archive/h3-gate3/README.md:21`・`:26`。
- **根拠（事実）**: 基準の条件の列挙にプロンプトが無い。基準の JSON（`h3-gate1/g5_t2v_after_review.json`）にもプロンプトの文字列は記録されていない（`prompt` で grep して 0 件。`text_tokens` 41 だけ）。h3-gate3 の README は、プロンプトの出典に第 2 弾の台本（`h3-gate2/g4_i2v_resident.py`）を挙げている。基準を取った第 1 弾 G5 の台本ではない。今回の取り直しで sha が一致したので、同じ文であることは事実として確定している。
- **直し方**: (1) §167 で基準の定義を「＋プロンプト＝黄色いレインコートの文（`h3-gate3/g2_selftest_g5prompt.cmd`）。セルフテストの既定ではない」と書き、§166.2 には訂正行を足す（正本 1 か所の原則。文書係の担当）。(2) h3-gate3 の README の「G5 のプロンプト」に、文そのもの（1 行）を併記する。

### m-6【Minor】`stage1_vs_new_compare.json` は `identical: false` なのに、README は「一致」と書いている。その理由の説明がない

- **ファイル:行**: `Docs/Outputs-archive/h3-gate3/README.md:10`、`stage1_vs_new_compare.json`。
- **根拠（事実）**: JSON は `committed_count 43`・`only_in_committed: ["using python 3-12-9 environment at: …-venv-engine-h3-stage1==?"]`・`identical: false` である。第 1 弾の環境の freeze を取るとき、uv の標準エラー出力（`Using Python … environment at:`）が一緒に入った 1 行で、パッケージの差ではない（`version_changed` と `only_in_new` は空）。
- **直し方**: README の該当箇所に「JSON の false は uv の案内行 1 行が混入したためで、パッケージの差は 0」と 1 句足す。

### m-7【Minor】日本語の案内が「物体追尾と MiniMax H3 では」と個別の名前を列挙している

- **ファイル:行**: `install_model.ps1:200`・`:349`。
- **根拠（事実）**: どちらも全モデル共通の文で、LTX 2.5 を追加するときにも表示される。任意導入の venv が増えるたびに、列挙を書き足すことになる（好み「例外を増やさず単純に」・正本 1 か所の原則）。
- **直し方**: 「モデルによっては、専用の Python 環境も先に作ります」のように名前を挙げない言い方にする。今のままでも誤りではないので、採るかどうかは監督の判断。

## G. 抜けの観点への答え

- **更新時の当て直し**: 仕組みと文言の両方で成り立つ（事実）。仕組み: `Get-EngineStateHash` は freeze の有効行と `$h3DirectPins` からハッシュを計算する（`:1331-1345`）。`-IgnoreSkipVenv` で `install-MiniMaxH3.bat` の経路でも判定に入り、印が違えば当て直す（`:1475-1489`）。文言: `install_ltx.ps1:1541-1544` に「`git pull` してから `setup.bat`、では当て直されない。`install-MiniMaxH3.bat` を再実行する」とある。利用者向けは README（このレビューの対象外）。残る穴は m-4（サーバー起動中の当て直し）だけ。
- **LICENSE の取得失敗**: 必須扱い（m-3）。`min` で取得の判定と検証の表の両方に入る。失敗したときの日本語の案内は、既存の「MISSING の行のファイルを 1 個消して再実行」がそのまま当てはまる。LICENSE は最後に取るので、重みは失われない。
- **空き容量の見積り**（参考。番号は付けない）: 表示は「必要: 約 56 GB」（`min` の合計 48.1GiB × 1.15）。実物は重み 50.2GiB＋venv 3.4GiB（`du`）＝約 53.6GiB で、56 に収まる。uv のキャッシュが venv と別にどれだけ増えるかは未確認。計画は「venv の分は README に書く」と決めているので、変更は要らない。

## H. 過剰設計・引き算

- 新しい変数（`$h3Venv`・`$h3Py`・`$h3FreezeSrc`・`$h3StateFile`・`$h3DirectPins`・`$h3UvArgs`）は、LTX 2.5 と UETrack の前例と同じ形で、不要なものは無い。
- 新しい分岐は 3 つ（venv・検証の表・注意文）。どれも計画が決めたもの。
- 引き算の候補は、m-2(b) の `key` と m-7 の列挙だけ。

## I. 未確認の一覧

- ダブルクリックで実行したときに `NativeCommandError` が出ないこと（D。3-G3 で確定する）。
- サーバー起動中の当て直しが失敗すること（m-4。推論）。
- 実際の取得の経路（重み 51GB・tokenizer）。開発機ではどちらも SKIP で、LICENSE しか通っていない（3-G3 で確定する）。
- uv のキャッシュが使う容量（G）。
