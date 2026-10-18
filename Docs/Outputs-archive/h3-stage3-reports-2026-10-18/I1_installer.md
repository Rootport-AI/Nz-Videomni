# I1 インストーラ機構の棚卸し（MiniMax H3 α版 第 3 弾 `install-MiniMaxH3.bat`）

- 調査日: 2026-10-18。対象: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni`・ブランチ dev（先頭 `2c1cf72`）。
- 読むだけの調査。リポジトリの編集・インストール・ダウンロード・GPU の使用はしていない。行番号は dev の現物。
- 以下、`install_ltx.ps1` は `scripts/install_ltx.ps1`、`install_model.ps1` は `scripts/install_model.ps1` の略。

---

## ① 結論の要約

**既存の機構のまま、ほとんどの部品が使える。足りないのは「H3 の仮想環境を作る 3 回目の呼び出し」と、その周りの UETrack 型の条件分岐だけ。**

### 新しいバッチに必要な要素と、既存で足りるかどうか

| # | 要素 | 既存で足りるか | 根拠 |
|---|---|---|---|
| 1 | `install-MiniMaxH3.bat`（ASCII・CRLF。`install_model.ps1 -BaseModel MiniMaxH3` を呼ぶだけ） | **足りる**（`install-UETrack.bat` を写して id と title を替えるだけ） | `install-UETrack.bat:15,17`、`.gitattributes` の `*.bat text eol=crlf` |
| 2 | 記述子の受け付け（id `MiniMaxH3`） | **足りる**。`engine_family` があるので `Get-BaseModelPlan` が受け付ける | `install_model.ps1:106`、`40-minimaxh3.json:3,5` |
| 3 | 空き容量の見積り（`files[].min` の合計×1.15） | **動くが、仮想環境の分（約 3.4GB）が入らない**（⑤） | `install_model.ps1:119-148` |
| 4 | `setup.bat` 済みの確認（`.venv-engine` の hf が起動するか） | **足りる**（ダウンロード道具は `.venv-engine` のもの） | `install_model.ps1:159-209`、`install_ltx.ps1:1685` |
| 5 | `install_ltx.ps1 -BaseModel MiniMaxH3 -SkipVenv -SkipMigrate` の呼び出し | **足りる**（呼び出し側は変更不要） | `install_model.ps1:377` |
| 6 | 記述子の妥当性検査（`assets.tokenizer_dir` が `files[].path` と一致・`kind:"dir"`） | **足りる**。現行の 40 番はすでに検査を通る形 | `install_ltx.ps1:429-438,344`、`40-minimaxh3.json:34-36,105-113` |
| 7 | `.venv-engine-h3` の構築（cu130 の index・torch/torchvision の直接ピン・SageAttention の wheel URL・freeze） | **関数は足りる・呼び出しが無い**。`Ensure-EngineVenv` は index・直接ピン・追加の uv 引数・`-IgnoreSkipVenv` をすべて引数で受ける。足すのは変数 4〜5 個と、`if ($foundIds -contains 'MiniMaxH3') { Ensure-EngineVenv ... -IgnoreSkipVenv }` の 1 ブロック | `install_ltx.ps1:1396-1450,1485-1492` |
| 8 | 重み 4 本＋tokenizer フォルダの取得（再試行 5 回・`.dl` 経由・再開） | **足りる**（記述子駆動。コード変更不要）。ただし **Comfy-Org 側のリポジトリ内のフォルダ名が未確認**（⑧） | `install_ltx.ps1:1647-1739`、`40-minimaxh3.json:45-115` |
| 9 | 検証の表（PASS/MISSING） | **重みの行は自動で出る**。H3 の python と `engine_h3/worker.py` の行は無い（UETrack 型の追加行を足すかは分岐点） | `install_ltx.ps1:1775-1802,1783-1786` |
| 10 | `models/INSTALLED_PATHS.txt` | **足りる**（`key` のある 5 行が、実在すれば自動で載る）。python の行は `.venv-engine` だけ | `install_ltx.ps1:1851-1865`、`40-minimaxh3.json:65,72,79,86,111` |
| 11 | UI の案内（`install-<id>.bat`） | **足りる**。id から `install-MiniMaxH3.bat` を組み立てる | `webui/src/shell/useBaseModels.ts:107-109` |
| 12 | 完了案内・失敗案内の日本語 | **動くが文面が一部ずれる**（「重みファイルを追加します」「1〜2 時間」「物体追尾では専用の Python 環境も作ります」など。⑥） | `install_model.ps1:198,206,345-349,371-377` |
| 13 | ドライバ（R580 以上）・GPU 世代（Turing 以降）の確認 | **無い**。インストーラは GPU もドライバも一切見ない方針 | `install_ltx.ps1:511-521,88-103` |
| 14 | 第 1 弾の torch のタイムアウト対策 | **インストーラの順序だけで原因の大半は消える**（仮想環境→重みの順に 1 本ずつ走る）。uv 自体の再試行・待ち時間の設定は今は何もしていない（④） | `install_ltx.ps1:1454-1492` の後に `:1637` の取得段 |

### 要点 5 つ

1. **関数の追加は要らない。** `Ensure-EngineVenv`（`install_ltx.ps1:1396`）に `-IndexUrl https://download.pytorch.org/whl/cu130`・`-DirectPins`（torch／torchvision／sageattention）・`-DirectPinArgs`（`--index cu130`＋`--index-strategy unsafe-best-match`）・`-IgnoreSkipVenv` を渡せば、LTX 2.5 と UETrack の前例の組み合わせで作れる。freeze の冒頭も、この形で使われることをすでに予告している（`engine_h3/venv-engine-h3.freeze.txt:35-38`）。
2. **SageAttention（cu130 版）の wheel URL は、まだインストーラに無い。** 書かれているのは仕様書と freeze のコメントだけ（`engine_h3/engine_h3-venv-pyproject.toml:72`、`venv-engine-h3.freeze.txt:34`）。LTX の 2 系統は `install_ltx.ps1` の配列に直に書いている（`:1200`、`:1240`）。
3. **開発機の `.venv-engine-h3` には完了の印（`.nz-engine-state`）が無い。** 手で作った環境なので、初回の実行は「印が無い→freeze を当て直す」の分岐に入る（`install_ltx.ps1:1435-1448`）。さらにこの環境の torch は、別セッションの一時フォルダの wheel から入れたもの（`direct_url.json` が `file:///C:/Users/PHENOT~1/AppData/Local/Temp/claude/.../h3/wheels/torch-2.13.0+cu130-...whl`）。当て直しで torch が取り直しになるかは未確認（⑧）。
4. **第 1 弾の uv は `tools\uv` の 0.11.32 ではなく、システムの chocolatey の uv 0.6.4 だった**（`.venv-engine-h3/pyvenv.cfg` の `uv = 0.6.4`、`which uv` は `/c/ProgramData/chocolatey/bin/uv`）。インストーラは `tools\uv` を PATH の先頭に置く（`install_model.ps1:353`）ので、第 3 弾の構築は 0.11.32 で走る。freeze の取り直し（設計 §11 の第 3 弾の範囲）は、この差を吸収する意味がある。
5. **空き容量の見積り（`NeedGB`）は 56GB になるが、実物の合計（重み 約 50.2GiB＋仮想環境 約 3.4GB）に対して余裕は約 2GB しかない。** 見積りは `files[].min` だけから作られ、仮想環境の分は入らない（⑤）。

---

## ② 流れの図と引数

```
install-MiniMaxH3.bat                         （新規。ASCII・CRLF。install-UETrack.bat:1-24 の写し）
  └─ powershell -NoProfile -ExecutionPolicy Bypass
       -File "%~dp0scripts\install_model.ps1" -BaseModel MiniMaxH3          （install-UETrack.bat:17 の形）
       ※ -File は -BaseModel a,b を 1 文字列で渡すので、1 回に 1 つだけ（install_model.ps1:34-37）

scripts/install_model.ps1   （UTF-8 BOM 付き・日本語。install_model.ps1:25-26）
  1. 二重起動ガード logs/.setup.lock（setup.bat と共用）              :287-306
  2. Get-BaseModelPlan -Id MiniMaxH3                                  :79-149
       engine_family か opt_in が無い記述子は除外                      :106
       FileCount / MinBytes / NeedGB = ceil(min合計×1.15 / 1GiB)        :119-148
  3. Start-Transcript logs/install_MiniMaxH3_<日時>.log                :317-326
  4. Test-SetupDone（uv・.venv・.venv-engine・.venv-engine-ltx25 の python.exe と
       .venv-engine の `python -P -m huggingface_hub.cli.hf version`。2>&1 は付けない）:159-209
  5. Test-FreeSpaceFor（警告のみ）                                     :217-234
  6. 案内文（「重みファイルを追加します」「1〜2 時間」）              :344-350
  7. $env:PATH = tools\uv; tools\ffmpeg\bin; ...（プロセス内だけ）      :353
  8. & install_ltx.ps1 -BaseModel $plan.Id -SkipVenv -SkipMigrate       :377
       （ドットソースしない。throw と exit 1 の両方を拾う）             :364-386
  9. 成功: OptIn なら追尾の文面、そうでなければ「一覧から『MiniMax H3』を選ぶ」:394-411
     失敗: Show-ModelInstallFailureHelp（MISSING の 1 ファイルだけ消して再実行）:239-278
  10. exit $exitCode（明示必須）                                        :430

scripts/install_ltx.ps1    （ASCII のみ・BOM なし・英語表示）
  0. 記述子を全件読み・全件検査（-BaseModel に関係なく）             :279-478
     -BaseModel で $TargetManifests / $foundIds を絞る                 :493-509
  1. git・uv の存在確認（GPU は見ない）                                 :522-539
  2. 移行（-SkipMigrate で飛ばす）                                      :~917
  3. uv python install 3.12（-SkipVenv でも走る・既存ならSKIP）         :1090-1099
  4. .venv（-SkipVenv で SKIP）                                         :1105-1132
  5. Ensure-EngineVenv ×2（.venv-engine・.venv-engine-ltx25。-SkipVenv で SKIP）:1454-1464
     UETrack 選択時だけ .venv-utils（-IgnoreSkipVenv）                  :1485-1492
       └─ ここに MiniMaxH3 選択時の .venv-engine-h3 を足す想定
     Ensure-EngineVenv:
       $wantState = Get-EngineStateHash(freeze の有効行 + DirectPins)   :1419
       python.exe 無し → uv venv --python 3.12 → Invoke-EngineFreezeApply → 印を書く :1425-1432
       印が一致 → SKIP                                                  :1433-1434
       それ以外 → Invoke-EngineFreezeApply → 印を書く                    :1435-1449
     Invoke-EngineFreezeApply:
       (a) uv pip install --python <py> <DirectPinArgs> <DirectPins>    :1338-1351
       (b) freeze から直接ピンの名前の行を除いた一時コピーを
           uv pip install --python <py> --index <IndexUrl>
                          --index-strategy unsafe-best-match -r <tmp>  :1353-1370
  6. 取得（$TargetManifests の downloads ごと）                         :1637-1739
       各ファイルの min で判定 → 揃っていれば SKIP                      :1651-1657
       & $enginePy -P -m huggingface_hub.cli.hf download <repo>
           --include <全パターン> --local-dir models\.dl\MiniMaxH3-<n>  :1666-1668,1685
       5 回まで（待ち 5/10/20/40 秒）                                    :1680-1696
       map で models\ へ -Force で移動 → min で再判定 → .dl を消す       :1698-1728
  7. 検証の表 → models/INSTALLED_PATHS.txt を作り直す → MISSING なら exit 1 :1772-1896
```

### `install_model.ps1` → `install_ltx.ps1` の引数（実物）

- `-BaseModel $plan.Id`（記述子の id。大文字小文字は問わない。`install_ltx.ps1:491`）
- `-SkipVenv`（setup.bat が作った `.venv`・`.venv-engine`・`.venv-engine-ltx25` に触れない。`install_ltx.ps1:122-127`）
- `-SkipMigrate`（移行と config.yaml の書き換えをしない）

---

## ③ `Ensure-EngineVenv` に H3 の仕様をどう渡すか

### 関数の受け口（`install_ltx.ps1:1396-1409`）

| 引数 | 必須 | 既定 | H3 で渡すもの |
|---|---|---|---|
| `-VenvPath` | 必須 | — | `"$ProjectRoot\.venv-engine-h3"` |
| `-PythonPath` | 必須 | — | `"$ProjectRoot\.venv-engine-h3\Scripts\python.exe"`（`config.py:128` の既定と同じ場所） |
| `-StateFile` | 必須 | — | `"$ProjectRoot\.venv-engine-h3\.nz-engine-state"` |
| `-FreezeFile` | 必須 | — | `"$ProjectRoot\engine_h3\venv-engine-h3.freeze.txt"` |
| `-Label` | 必須 | — | `".venv-engine-h3"` |
| `-DirectPins` | 任意 | `@()` | 下の配列（分岐点 B1） |
| `-DirectPinArgs` | 任意 | `@()` | `--index cu130`・`--index-strategy unsafe-best-match`（＋`--no-sources` を付けるかは分岐点） |
| `-DirectPinsLabel` | 任意 | 2.3 の文言 | 表示用の文言（変えないと 2.3 の「diffusers / ltx-core…」が出る。`:1405`） |
| `-IndexUrl` | 任意 | cu128 | `https://download.pytorch.org/whl/cu130`（段 (b) の index。表示は末尾の `cu130` になる。`:1353`） |
| `-StackLabel` | 任意 | `torch cu128 stack` | 例 `torch cu130 stack`（見出しの表示だけ。`:1416`） |
| `-IgnoreSkipVenv` | 任意 | off | **必須で付ける**（install_model.ps1 は常に `-SkipVenv` を渡すため。UETrack の前例 `:1476-1491`） |

### LTX 2.5 の実例（引用）

```powershell
# install_ltx.ps1:1235-1241
$ltx25DirectPins = @(
    "torch==2.9.1+cu128"
    "torchaudio==2.9.1+cu128"
    "ltx-core @ git+https://github.com/Lightricks/LTX-2.git@d151147788a9284cca791edc6ce898007e727fe6#subdirectory=packages/ltx-core"
    "ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git@d151147788a9284cca791edc6ce898007e727fe6#subdirectory=packages/ltx-pipelines"
    'sageattention @ https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu128torch2.9.1.post6-cp310-abi3-win_amd64.whl'
)
# install_ltx.ps1:1256-1260
$ltx25UvArgs = @(
    "--no-sources"
    "--index", "https://download.pytorch.org/whl/cu128"
    "--index-strategy", "unsafe-best-match"
)
# install_ltx.ps1:1460-1464
Ensure-EngineVenv -VenvPath $ltx25Venv -PythonPath $ltx25Py `
    -StateFile $ltx25StateFile -FreezeFile $ltx25FreezeSrc `
    -DirectPins $ltx25DirectPins -Label ".venv-engine-ltx25" `
    -DirectPinArgs $ltx25UvArgs `
    -DirectPinsLabel "2 torch pins + 2 git pins + 1 wheel pin: torch / torchaudio / ltx-core / ltx-pipelines / sageattention"
```

- torch を段 (a) に入れる理由（`:1211-1220`）: 段 (a) は本物の解決なので、torch を入れないと uv が cu128 の index の最新（約 2.11）を取りに行き、段 (b) で 2.9.1 へ戻す二度手間になる。`+cu128` の局所版は torchaudio でも必須。
- `--no-sources` の理由（`:1245-1252`）: LTX-2 の公式パッケージが自前の uv の index 設定（cu132）を持ち、衝突するため。**H3 には git のピンが無いので、この理由は当てはまらない。** 付けても害は無い（未実測）。
- SageAttention の wheel は単一引用符で書く（`%2B` を PowerShell に解釈させない。`:1190,1233-1234`）。

### UETrack の実例（`-IgnoreSkipVenv` と条件付きの呼び出し。引用）

```powershell
# install_ltx.ps1:1485-1492
if ($foundIds -contains 'UETrack') {
    Ensure-EngineVenv -VenvPath $utilsVenv -PythonPath $utilsPy `
        -StateFile $utilsStateFile -FreezeFile $utilsFreezeSrc `
        -Label ".venv-utils" `
        -IndexUrl "https://download.pytorch.org/whl/cpu" `
        -StackLabel "torch CPU stack" `
        -IgnoreSkipVenv
}
```

- `$foundIds` の比較は `-contains`（大文字小文字を区別しない）。`$foundIds` は記述子の id の綴りのまま（`MiniMaxH3`）。`:502-503`

### H3 で渡すことになる素材（リポジトリにある事実）

- torch／torchvision の版: `torch==2.13.0+cu130`・`torchvision==0.28.0+cu130`（`engine_h3-venv-pyproject.toml:107-108`、freeze `:79-80`）。仕様書は「torch と torchvision の両方に `+cu130` を付けないと uv が別の組を選ぶ」と書く（`engine_h3-venv-pyproject.toml:90-93`）。
- SageAttention の URL（cu130・libtorch 安定 ABI 版）: `https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl`（`engine_h3-venv-pyproject.toml:72`、freeze `:34`）。freeze の行は `sageattention==2.2.0+cu130torch2.10.0andhigher.post6`（`:74`）。
- 開発機に入っている sageattention の `direct_url.json` は上の URL と一致（実物を確認）。METADATA に `Requires-Dist` は無い（＝sage を単独で入れても torch は引き込まれない）。
- freeze は 42 パッケージ。`hf-xet==1.7.0`（`:55`）・`huggingface-hub==1.33.0`（`:58`）は transformers 経由で入るもので、H3 の仮想環境はダウンロード道具には使わない（道具は `.venv-engine` の hub 0.36.2＋hf-xet 1.6.0。`install_ltx.ps1:1494-1502`、`engine/venv-engine.freeze.txt:37-45,58,61`）。
- 段 (b) の除外の正規表現は直接ピンの名前から作る（`:1349-1350`）。`torch==2.13.0+cu130` → `torch`、`sageattention @ https://...` → `sageattention`。`^(torch|torchvision|sageattention)==` は `torchgen` などに当たらない（`==` まで含めて照合するため）。
- freeze の冒頭の予告（`venv-engine-h3.freeze.txt:35-38`）: 「インストーラ（Ensure-EngineVenv の 3 回目の呼び出し・H3 α版の第 3 弾）がこの freeze を当てるとき、torch／torchvision／sageattention を直接ピンとして通常の `-r freeze` から外す。2.5 と同じ」。

### 状態ハッシュ `Get-EngineStateHash`（`install_ltx.ps1:1289-1304`）

- 入力は「freeze の有効行（空行と行頭 `#` の行を除く・改行を LF に正規化）」＋「直接ピンの配列」。**行頭のコメントや CRLF／LF の違いではハッシュは変わらない**（`:1278-1284,1294-1296`）。行末に付けたコメントは除かれないので、ピンの行に注記を書くと変わる。
  - 記憶の落とし穴「freeze のコメント変更でハッシュが変わり貼り直しが起きる」は、現行のコードでは**行頭のコメントについては当てはまらない**。当てはまるのは、ピンの行・直接ピンの文字列（URL の 1 文字でも）を変えたとき。
- 印が無いときは「当て直し」に入る（`:1436-1443`）。開発機の `.venv-engine-h3` は印が無い（実物を確認）ので、初回は当て直しになる。
- 当て直しは `uv pip install`（`sync` ではない）なので、freeze に無いパッケージを消さない。

### setup.ps1 との関係（依頼文との相違の訂正）

- `scripts/setup.ps1` は `Ensure-EngineVenv` を**直接は呼ばない**。`install_ltx.ps1` を引数なしで呼ぶだけ（`setup.ps1:418`）。`Ensure-EngineVenv` の 2 回の呼び出しは `install_ltx.ps1:1454` と `:1460`（どちらも無条件。`-SkipVenv` のときだけ SKIP）。
- 引数なしの既定は `-BaseModel @('LTX23','Preprocessors')`（`install_ltx.ps1:141`）なので、H3 の呼び出しを `$foundIds -contains 'MiniMaxH3'` で囲めば、`setup.bat` は H3 の仮想環境を作らない・当て直さない。

---

## ④ 第 1 弾の torch のタイムアウトを避ける手段の候補

### 事実

- 第 1 弾の経緯: uv が torch（約 1.9GB）の取得で 2 回タイムアウトした。**54GB の重みの取得と回線を分け合ったため**。curl で wheel を取ってローカルから入れた（`VERIFICATION_LOG.md:16999`、freeze `:16-22`）。記録は「回線が空いているときは問題にならない見込み・第 3 弾では取得の順序を考える」（同 `:16999`）。
- 第 1 弾の uv はシステムの 0.6.4。インストーラは `tools\uv` の 0.11.32（`setup.ps1:38`、`tools/uv/uv.exe --version` の実物）。
- インストーラの順序: 仮想環境（段 5）→重みの取得（段 6）を**同じプロセスで順番に**行う（`install_ltx.ps1:1454-1492` → `:1637-1739`）。並行には走らない。
- `Invoke-EngineFreezeApply` に再試行は無い。uv が非 0 で終われば `throw`（`:1342,1367`）。`hf download` には固定 5 回の再試行がある（`:1671-1696`。§82.7 のオーナー裁定「常識的な回数で打ち切り、明示的なエラーで止まる」`VERIFICATION_LOG.md:10078`）。
- uv の取得先の一時領域とキャッシュは `.uv_cache`（`install_ltx.ps1:154`。`UV_CACHE_DIR` が未設定のとき）。開発機の `.uv_cache\archive-v0\0c5h6MMSJf_IKyVP0JdjI\` に torch 2.13.0+cu130 の展開済みの実体がある（実物を確認）。

### 候補（設計の提案ではなく、選択肢の列挙）

| # | 手段 | 何が効くか | 確かさ |
|---|---|---|---|
| T1 | **何もしない（順序だけで足りる）** | 第 1 弾の原因（重みの取得との回線の取り合い）は、インストーラの順番実行では起きない | 事実（順番実行）＋推測（それで足りる） |
| T2 | uv の待ち時間を延ばす: `$env:UV_HTTP_TIMEOUT`（プロセス内だけ。既存の `UV_CACHE_DIR` の書き方 `:153-155` に倣える） | 遅い回線での読み取りタイムアウトを避ける | uv の文書による（既定は 30 秒）。**本調査では実機で未確認** |
| T3 | uv の再試行回数を増やす: `$env:UV_HTTP_RETRIES` | 一時的な接続の失敗を uv の中で吸収 | 同上（既定は 3 回とされる）。**未確認** |
| T4 | `Invoke-EngineFreezeApply` の uv の呼び出しを hf と同じ固定回数の再試行でくるむ | 接続の失敗で止まらない。打ち切りは有限 | 実装は単純。ただし uv は途中までの取得を再開しない見込み（**未確認**）なので、1.9GB は毎回最初から |
| T5 | 取得の順序を変える（例: H3 の仮想環境を先に作り、重みは後） | 現行の順序がすでにこれ（段 5 → 段 6） | 事実 |
| T6 | torch の wheel を先に別の道具（curl・`Invoke-WebRequest`）で取り、`--find-links` か直接のファイル指定で入れる（第 1 弾の手作業の再現） | 再開できる取得の道具を使える | 手順が増え、`direct_url.json` が `file:///` になる（開発機の現状と同じ。⑧ の問題を生む） |
| T7 | 同時の取得数を減らす: `$env:UV_CONCURRENT_DOWNLOADS` | 1 本の大きな wheel に帯域を集める | uv の文書による。**未確認** |

- 参考: hf の側は「`.incomplete` が残るので再試行は続きから」（`install_ltx.ps1:1675-1676`、§82.7）。uv にはこの性質が無い見込み（未確認）。

---

## ⑤ 空き容量の算出と README の見出しの名指し

### 算出（`install_model.ps1:119-148,217-234`）

- `MinBytes` = 記述子の全 `downloads[].files[].min` の合計（`kind:"dir"` の行の min も含む）。
- `NeedGB` = `ceil(MinBytes × 1.15 / 1GB)`（PowerShell の `1GB` は 1,073,741,824）。1.15 は「min と実物の差＋取得中の一時ファイル」の見込み（`:145-147`）。
- 判定は警告のみで止めない（`:225-227`）。ドライブは `$ProjectRoot` のあるドライブ（`:220`）。
- `.dl` から同じドライブ内で移動するので二重には使わない（`:214-215`）。

### H3 に当てはめた数字（計算）

| 項目 | バイト |
|---|---|
| min の合計（20,100,000,000＋26,000,000,000＋5,000,000,000＋580,000,000＋11,000,000） | 51,691,000,000 |
| ×1.15 | 59,444,650,000 |
| `NeedGB`（÷1,073,741,824 の切り上げ） | **56** |
| 実物の合計（`VERIFICATION_LOG.md:17008-17012` と記述子の `_min`: 20,970,379,616＋27,141,342,152＋5,207,808,496＋605,254,808＋11,493,942） | 53,936,279,014（約 50.2GiB） |
| `.venv-engine-h3` の実測（`du -sh` で 3.4G。freeze `:14` も約 3.4GB） | 約 3.4GB |

- **見積り 56GB に仮想環境の 3.4GB は入っていない。** 実物の合計（約 50.2GiB＋約 3.4GB≒約 53.6GiB）に対し、余裕は約 2GiB。uv のキャッシュは同じドライブならハードリンクで共有されるが、torch の版が他と違うので共有は効かない（`engine_h3-venv-pyproject.toml:16-18`、設計 §10）。
- `Test-FreeSpaceFor` の文面には仮想環境の分を足す仕組みが無い。足すなら、記述子に値を持たせるか・`install_model.ps1` に定数を持つかのどちらか（分岐点 B6）。

### README の見出しの名指し

- `scripts/setup.ps1:113` と `:115` が、README の見出し **「必要な空き容量の内訳」** を名前で指している（`setup.bat` の約 50GB の案内）。この見出し（`README.md:253`）は**改名すると案内が切れる**。
- `install_model.ps1` は README を名指ししない（数字は記述子から自前で作る。`:346`）。
- README の現状（H3 の行は無い）:
  - ハードウェア要件表の「ストレージ（ベースモデルを追加する場合）」は LTX 2.5 だけ（`README.md:238`）。
  - 「必要な空き容量の内訳」は「3 つの venv」「`.venv-engine-ltx25` は setup.bat の時点で作られる」と書く（`README.md:255-259`）。H3 の仮想環境は `install-MiniMaxH3.bat` で作られる点が、LTX 2.5 と逆になる。
  - venv の構成表（`README.md:170-175`）と後片付けの一覧（`README.md:163`）に `.venv-engine-h3` が無い。
  - ドライバの行は「R570 以上を推奨」（`README.md:235`）。H3 は R580 以上（`engine_h3-venv-pyproject.toml:81-82`、設計 §10）。
  - 追加の導線の節は「LTX 2.5 を追加する」（`:403-428`）と「物体追尾を追加する」（`:430-444`）。H3 の節は無い。
  - README の詳細な棚卸しは別の調査（I4）の範囲。

---

## ⑥ 検証の表・INSTALLED_PATHS・UI の案内

### 検証の表（`install_ltx.ps1:1772-1896`）

- 固定の 3 行（`engine_python`＝`.venv-engine`・`app_python`・`engine/worker.py`）は常に出る（`:1775-1779`）。
- UETrack のときだけ `utils_python`（`.venv-utils/Scripts/python.exe`）と `tracking/worker.py` の 2 行を足す（`:1783-1786`。「setup.bat が作らない環境を報告しない」理由）。**H3 には同等の行が無い**（足すかは分岐点 B4）。
- 重みの行は `$TargetManifests` の `files[]` から自動（`:1791-1802`）。H3 では 5 行（`h3_transformer`・`h3_text_encoder`・`h3_video_vae`・`h3_audio_vae`・`tokenizer_dir (Qwen3-VL tokenizer dir)`）。
- `kind:"dir"` の行は直下のファイルの合計で判定（`:1797,1811`、`Get-DirectChildSize :195-203`）。**`Get-DirectChildSize` のコメントは「唯一の利用者は tokenizer の `kind:"dir"` 行（10 ファイル）」と書いている**（`:191-194`）。H3 の tokenizer の行が 2 つ目の利用者になるので、このコメントは H3 の取得が動いた時点で事実と合わなくなる。コード（`:344`）は `kind:"dir"` を一般に受け付けている。
- MISSING があれば exit 1。`-SkipVenv` の注意は `.venv-engine` が無いときだけ出る（`:1883-1895`）。

### `models/INSTALLED_PATHS.txt`（`:1832-1881`）

- `-BaseModel` で絞らず、全記述子の `key` 付きの行を「実在すれば」載せる（オーナー裁定。`:1845-1850`）。H3 の 5 行はすでに `key` を持つ（`h3_transformer`・`h3_text_encoder`・`h3_video_vae`・`h3_audio`・`h3_tokenizer_dir`）。
- **開発機は H3 の重みが手で置いてあるので、今 setup.bat を走らせても H3 の 5 行は載る**（実在するため）。
- python の行は `engine_python`（`.venv-engine`）だけ固定で書く（`:1865`）。`.venv-engine-ltx25` も `.venv-utils` も書いていない（前例として「書かない」側）。
- ヘッダの文面（`:1870-1876`）は LTX の bf16 と QAT Gemma に触れる固定文で、H3 には関係しない。

### UI の案内

- 未導入のベースモデルを選ぶと `install-<id>.bat` を案内する。名前は id から作る（`webui/src/shell/useBaseModels.ts:107-109`、`api/types.ts:1100-1116`、`MULTI_ENGINE_DESIGN.md` §6.2 のガード (2)）。H3 の id は `MiniMaxH3` なので **`install-MiniMaxH3.bat`**。裁定 15 の名前（`MINIMAX_H3_RESEARCH_NOTES.md:263`）と一致する。
- 「導入済み」は重みの既定ファイルの有無で決まり（`present`／`installed`。`api/types.ts:1098-1110`）、**仮想環境の有無は見ない**。仮想環境が無いまま重みだけあるときは、選んだ時点でサーバーがアダプタの `engine_python_h3` の確認で失敗する（`services/engines/h3/adapter.py:320-324`。どの文面になるかは本調査では未確認）。
- 完了案内（`install_model.ps1:406-411`）は OptIn でない記述子の共通文「画面上部のベースモデルの一覧から『MiniMax H3』を選んでください」になる。そのまま使える。

### 文面がずれる箇所（`install_model.ps1`）

- `:22`・`:156-157`・`:198`・`:206`・`:372-374`: 「Python 環境は作らない。例外は UETrack の `.venv-utils`」という書き方。H3 で 2 つ目の例外になる。
- `:345`: 「重みファイルを追加します。ほかのベースモデルの環境やモデルには手を加えません。」（H3 では仮想環境も作る）。
- `:347`: 「回線速度により 1〜2 時間」（固定文。H3 は約 54GB＋torch 約 1.9GB）。
- `install_ltx.ps1` の英語のコメント（`:3-4`・`:30-32`・`:1135-1169` の「TWO of them」・`:1373-1395`）も 2 系統＋UETrack 前提の書き方。

---

## ⑦ 計画で決める分岐点（選択肢の列挙だけ）

- **B1 直接ピンの組み方**
  - (a) torch・torchvision・sageattention の 3 本（LTX 2.5 型。freeze の予告 `:35-38` と一致）
  - (b) sageattention だけ（LTX 2.3 型。sage は依存を持たないので torch を引き込まない。torch／torchvision は段 (b) で cu130 の index から入る）
- **B2 `-DirectPinArgs` に `--no-sources` を付けるか**（H3 には git のピンが無く、付ける理由は無い。付けても害は無い見込み）
- **B3 H3 の呼び出しの条件**
  - (a) `if ($foundIds -contains 'MiniMaxH3')` で囲み `-IgnoreSkipVenv`（UETrack 型。setup.bat は触らない）
  - (b) 記述子に「専用の仮想環境が要る」印を持たせ、それで判定（UETrack と H3 を一般化）
  - 帰結: (a)(b) とも、`git pull`＋`setup.bat` では H3 の仮想環境の当て直しは起きない。freeze が変わったら `install-MiniMaxH3.bat` の再実行が要る（UETrack と同じ）。
- **B4 検証の表に `h3_python`・`engine_h3/worker.py` の行を足すか**（UETrack 型の 2 行 `:1783-1786`）
- **B5 `install_model.ps1` の `Test-SetupDone` の対象**（現状のまま `.venv-engine-ltx25` まで要求するか。H3 の取得には `.venv-engine` だけが要る）
- **B6 空き容量の見積りに仮想環境の分を入れるか**
  - (a) 入れない（現状。余裕 約 2GiB）
  - (b) 記述子に追加の容量を持たせる（新しいキー）
  - (c) `install_model.ps1` に H3 用の定数
  - (d) 1.15 の係数を変える
- **B7 torch のタイムアウト対策**（④の T1〜T7 のどれを採るか・組み合わせ）
- **B8 ドライバ R580・GPU 世代の確認をインストーラに入れるか**（現方針は GPU を一切見ない `install_ltx.ps1:511-521`。入れるなら警告のみか止めるか）
- **B9 開発機の既存 `.venv-engine-h3` の扱い**
  - (a) そのまま実行して「印が無い→当て直し」を通す（torch の取り直しの有無は未確認）
  - (b) 実行前に退避・削除して新規構築を通す
  - (c) 印だけ手で書く（検証にならない）
- **B10 freeze の取り直し**（設計 §11 の第 3 弾の範囲）: インストーラで作った環境から `uv pip freeze` で取り直すか。取り直すとピンの行が変われば印のハッシュが変わる（第 1 弾の uv 0.6.4 と 0.11.32 で解決結果が違う可能性）。取り直しの手順と、METHOD の注記（`:16-22` の手での正規化 2 行）の扱い。
- **B11 文面の更新の範囲**（`install_model.ps1` の日本語・`install_ltx.ps1` の英語のコメント・`Get-DirectChildSize` のコメント・README）。`install_ltx.ps1` は ASCII のみ・BOM なし（非 ASCII を入れると PS 5.1 で化ける）、`install_model.ps1`・`setup.ps1` は UTF-8 BOM 付き。GNU `sed -i` は CRLF を LF にするので使わない（Edit か PowerShell）。
- **B12 実地の検証の場所**（開発機は重みが揃っているので取得段はすべて SKIP になる。取得を通すには、サブマシンかまっさらな環境か、ファイルの一時退避が要る）。§82 の LTX 2.5 はサブマシンで G0〜G7 を通した前例（`MULTI_ENGINE_DESIGN.md` §6.2）。

---

## ⑧ 未確認

1. **Comfy-Org/MiniMax-H3 のリポジトリ内のフォルダ名。** 記述子は `diffusion_models/`・`text_encoders/`・`vae/` を前提にしている（`40-minimaxh3.json:50-58`）。Comfy-Org の配布物は `split_files/diffusion_models/...` の形を取ることがあり、もしそうなら取得後の移動で「どの map も受け付けない」と止まる（`install_ltx.ps1:1710-1711`）。記述子の注記も「ゲートの有無とファイル名は未検証」（`40-minimaxh3.json:6`）。研究ノートは `resolve/main/` の下のヘッダを実測したと書くが、フォルダ名は書いていない（`MINIMAX_H3_RESEARCH_NOTES.md:344`）。HF の tree API（メタデータの読み取り）で確かめられる。I2（取得元の調査）の範囲と重なる。
2. **Comfy-Org/MiniMax-H3 が gated かどうか。** MiniMax 本家は gated を使っていない（`MINIMAX_H3_RESEARCH_NOTES.md:98`）。Comfy-Org 側は未記載。gated だとトークンなしの取得は 5 回失敗して止まる（インストーラはトークンを扱わない `install_ltx.ps1:1558-1561`）。
3. **既存の `.venv-engine-h3` への当て直しで torch が取り直しになるか。** torch の `direct_url.json` が `file:///` の一時フォルダを指している。uv がこれを「`torch==2.13.0+cu130`（index の指定）を満たす」とみなすかは未確認。みなさなければ 1.9GB の取得が走る（`.uv_cache` の展開済みの実体が再利用されるかも未確認）。
4. **uv 0.11.32 の `UV_HTTP_TIMEOUT`・`UV_HTTP_RETRIES`・`UV_CONCURRENT_DOWNLOADS` の既定値と効き方**、および uv が途中までの取得を再開しないこと。uv の文書の記憶に基づく記述で、実機では確かめていない。
5. **uv 0.11.32 での段 (a)(b) の解決結果が、第 1 弾（uv 0.6.4）の freeze と同じになるか。** 段 (a) は本物の解決なので、torch の依存（filelock・sympy・networkx など）は一度最新が入り、段 (b) で freeze の版へ戻る（LTX 2.5 と同じ挙動）。最終状態が freeze と一致するかは、実際に作るまで分からない。
6. **仮想環境が無く重みだけがあるとき、ベースモデルを選んだときの画面の文面。** `services/engines/h3/adapter.py:320-324` が `engine_python_h3` を見ることまでは確認。文面は未確認。
7. **サブマシンのドライバの版**（設計 §13「第 3 弾の README までに確かめる」）。
8. **`.uv_cache` の torch 2.13.0+cu130 が index 由来か file 由来か**（`wheels-v5/index/*/torch` が 3 つある。どれが cu130 かは見ていない）。
