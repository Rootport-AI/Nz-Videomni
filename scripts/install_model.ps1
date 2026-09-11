#Requires -Version 5.1
<#
.SYNOPSIS
    ベースモデルを1つ追加で導入するスクリプト（install-<ID>.bat から呼び出される本体）。

.DESCRIPTION
    setup.bat は「本体＋お試し用の LTX 2.3」までを入れる。それ以外のベースモデルは
    install-<ID>.bat をダブルクリックして足す、という2階建ての導線になっている
    （Docs/MULTI_ENGINE_DESIGN.md §6.2）。このファイルはその共通の入口で、
    <ID> は scripts/manifests/*.json の 'id'（＝ models/<ID>/ のフォルダ名）。

    ここでは次の順番で処理する。

      1. 二重起動ガード（logs/.setup.lock を setup.bat と共有する）
      2. 記録の開始（logs/install_<ID>_<日時>.log）
      3. 記述子の読み込み（表示名・ファイル数・おおよその容量）
      4. setup.bat が済んでいるかの確認（済んでいなければ、ここで止める）
      5. 空き容量の確認（警告のみ）
      6. これから何が起きるかの説明
      7. tools/ を PATH の先頭へ（このプロセスの中だけ）
      8. scripts/install_ltx.ps1 の実行（重みの取得のみ。Python 環境は作らない）
      9. 終了案内

    このファイルは UTF-8（BOM 付き）で保存すること。Windows PowerShell 5.1 は
    BOM の無い UTF-8 を ANSI として読むため、BOM を落とすと日本語が化ける。

.EXAMPLE
    ./scripts/install_model.ps1 -BaseModel LTX25
#>

[CmdletBinding()]
param(
    # 追加するベースモデルの記述子 id（scripts/manifests/*.json の 'id'）。
    # 大文字小文字は問わない。1回の実行で1つだけ（-File 経由の呼び出しは
    # コンマ区切りを1つの文字列として渡してしまうため、複数指定には対応しない）。
    [Parameter(Mandatory)] [string] $BaseModel
)

$ErrorActionPreference = 'Stop'
# 進捗バーの描画は Windows PowerShell 5.1 を極端に遅くする。切っておく。
$ProgressPreference = 'SilentlyContinue'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot

$ManifestDir = Join-Path $ProjectRoot 'scripts\manifests'
$ModelsDir   = Join-Path $ProjectRoot 'models'
$ToolsDir    = Join-Path $ProjectRoot 'tools'
$UvDir       = Join-Path $ToolsDir 'uv'
$FfmpegBin   = Join-Path $ToolsDir 'ffmpeg\bin'

$LogsDir     = Join-Path $ProjectRoot 'logs'
$LockFile    = Join-Path $LogsDir '.setup.lock'

# ---------------------------------------------------------------------------
# 画面表示の小道具
#
# scripts/setup.ps1 と同じ体裁をここへ写してある（意図的な複製）。共有できるのは
# 見た目だけで、事実は記述子 scripts/manifests/*.json から読む。共通ファイルへ
# 切り出すと setup.ps1 の改造を伴うため、そちらは触らない方針。
# ---------------------------------------------------------------------------
function Write-Head([string] $Message) {
    Write-Host ''
    Write-Host ('==== ' + $Message + ' ' + ('=' * [Math]::Max(4, 66 - $Message.Length))) -ForegroundColor Cyan
}
function Write-Info([string] $Message) { Write-Host ('  ' + $Message) }
function Write-Good([string] $Message) { Write-Host ('  [OK]   ' + $Message) -ForegroundColor Green }
function Write-Skip([string] $Message) { Write-Host ('  [済み] ' + $Message) -ForegroundColor DarkGray }
function Write-Warn([string] $Message) { Write-Host ('  [注意] ' + $Message) -ForegroundColor Yellow }
function Write-Bad([string] $Message)  { Write-Host ('  [失敗] ' + $Message) -ForegroundColor Red }

function Format-Size([long] $Bytes) {
    if ($Bytes -ge 1GB) { return ('{0:N1} GB' -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ('{0:N0} MB' -f ($Bytes / 1MB)) }
    return ("$Bytes B")
}

# ---------------------------------------------------------------------------
# 記述子から、画面に出すための情報だけを取り出す
#
# 取得も検証も scripts/install_ltx.ps1 が記述子を読み直して行う。ここで読むのは
# 「表示名・ファイル数・おおよその容量」を先に伝えるためだけで、判断はしない。
# ---------------------------------------------------------------------------
function Get-BaseModelPlan {
    param([Parameter(Mandatory)] [string] $Id)

    if (-not (Test-Path -LiteralPath $ManifestDir)) {
        throw ('記述子のフォルダが見つかりません: ' + $ManifestDir)
    }
    $known = @()
    $found = $null
    foreach ($f in (Get-ChildItem -LiteralPath $ManifestDir -Filter '*.json' -File | Sort-Object Name)) {
        $data = $null
        try {
            $data = (Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8) | ConvertFrom-Json
        } catch {
            throw ('記述子 ' + $f.Name + ' を読めませんでした: ' + $_.Exception.Message)
        }
        if (-not $data.id) { continue }
        # ベースモデルの記述子だけを相手にする。'engine_family' を持たないもの
        # （00-preprocessors.json のような共用の記述子）は setup.bat が本体と
        # 一緒に入れるもので、ヘッダーの一覧にも出ない。ここで受け付けると
        # 「一覧から Preprocessors を選んでください」という、実行できない案内を
        # 出してしまう（Docs/MULTI_ENGINE_DESIGN.md §4.2）。
        #
        # ただし 'opt_in' が真の記述子は例外として受け付ける。これは
        # 「ベースモデルではないが、setup.bat にも同梱せず、専用の
        # install-<ID>.bat で足すもの」の印で、30-uetrack.json（物体追尾）が
        # 最初の例（Docs/OBJECT_TRACKING_DESIGN.md）。この一行が無いと
        # install-UETrack.bat は「そのようなものはありません」で止まる。
        if ((-not $data.engine_family) -and (-not $data.opt_in)) { continue }
        $known += [string] $data.id
        if (([string] $data.id) -eq $Id) { $found = $data }
    }
    if (-not $found) {
        Write-Host ''
        Write-Bad ('「' + $Id + '」というベースモデルは、この配布物にはありません。')
        Write-Info ('選べるのは次のとおりです: ' + ($known -join '、'))
        Write-Info '（このスクリプトは install-<名前>.bat から呼ばれます。バッチの名前と、ここに出ている名前が食い違っていないか確かめてください。）'
        Write-Host ''
        exit 1
    }

    $count = 0
    $bytes = [long] 0
    foreach ($dl in @($found.downloads)) {
        foreach ($fl in @($dl.files)) {
            $count++
            $bytes += [long] $fl.min
        }
    }
    if ($count -eq 0) {
        Write-Host ''
        Write-Bad ('「' + $Id + '」には取得するファイルが登録されていません。')
        Write-Info ('記述子（' + $ManifestDir + '）の downloads が空です。配布物が壊れている可能性があります。')
        Write-Host ''
        exit 1
    }

    $displayName = if ($found.display_name) { [string] $found.display_name } else { [string] $found.id }
    return [pscustomobject]@{
        Id          = [string] $found.id
        DisplayName = $displayName
        # 'opt_in' が真の記述子は、ベースモデルではないもの（物体追尾など）。
        # ヘッダーのベースモデルの一覧には現れないため、完了案内の文面を
        # 切り替える必要がある（Docs/OBJECT_TRACKING_DESIGN.md §7）。
        OptIn       = [bool] $found.opt_in
        FileCount   = $count
        MinBytes    = $bytes
        # 実物は記述子の min（公式サイズの 4〜10%下）より必ず大きい。1.15 倍は
        # その差と、取得中の一時ファイルの分をまとめて見込んだ安全側の目安。
        NeedGB      = [Math]::Ceiling(($bytes * 1.15) / 1GB)
    }
}

# ---------------------------------------------------------------------------
# setup.bat が済んでいるか
#
# 済んでいなければ、ここで止めて setup.bat へ案内する。このスクリプトは
# Python 環境を作らない（作ると「重みを足したいだけ」の操作が、いま動いている
# LTX 2.3 の実行環境を書き換えてしまう）。
# ---------------------------------------------------------------------------
function Test-SetupDone {
    $needed = [ordered]@{
        'uv（道具）'             = Join-Path $ProjectRoot 'tools\uv\uv.exe'
        'アプリ用 Python 環境'   = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
        'エンジン用 Python 環境' = Join-Path $ProjectRoot '.venv-engine\Scripts\python.exe'
        'ダウンロード道具（hf）' = Join-Path $ProjectRoot '.venv-engine\Scripts\hf.exe'
    }
    $missing = @()
    foreach ($k in $needed.Keys) {
        if (-not (Test-Path -LiteralPath $needed[$k])) { $missing += ($k + '  … ' + $needed[$k]) }
    }

    # 存在するだけでは足りない。フォルダごと移動・改名すると、この hf.exe は
    # 自分の Python を見失って起動しなくなる（ファイルは残っているので、
    # 存在確認だけでは見抜けない）。実際に起動して終了コードを見るしかない。
    #
    # ここで 2>&1 を付けてはいけない。$ErrorActionPreference = 'Stop' のもとでは、
    # 外部プログラムが標準エラーへ書いた1行が NativeCommandError という例外に
    # 化けてしまい、「壊れている」ではなく「予期しない失敗」として扱われる。
    if ($missing.Count -eq 0) {
        $hfExe = $needed['ダウンロード道具（hf）']
        $hfOk = $false
        try {
            & $hfExe --help | Out-Null
            $hfOk = ($LASTEXITCODE -eq 0)
        } catch {
            $hfOk = $false
        }
        if (-not $hfOk) {
            $missing += ('ダウンロード道具（hf）が起動しません  … ' + $hfExe)
        }
    }

    if ($missing.Count -eq 0) { return $true }

    Write-Host ''
    Write-Bad '先に setup.bat を実行してください。'
    Write-Host ''
    Write-Info 'このバッチは、追加のモデルファイルを取ってくるだけのものです。'
    Write-Info 'Python 環境とダウンロード道具は setup.bat が用意します。まだ揃っていません。'
    Write-Host ''
    Write-Info '足りないもの、または動かないもの:'
    foreach ($m in $missing) { Write-Info ('  ・' + $m) }
    Write-Host ''
    Write-Info ('setup.bat の場所: ' + (Join-Path $ProjectRoot 'setup.bat'))
    Write-Info 'setup.bat をダブルクリックして、終わるのを待ってから、もう一度このバッチを実行してください。'
    Write-Info '（フォルダごと移動したり名前を変えたりした直後にも、この案内が出ます。その場合も setup.bat をもう一度実行すれば直ります。）'
    Write-Host ''
    return $false
}

# ---------------------------------------------------------------------------
# 空き容量（警告のみ。ここで処理を止めない）
#
# 取得したファイルは models\.dl\ にいったん置いてから、同じドライブの中で
# 移動するだけなので、二重に容量を使うことはない。
# ---------------------------------------------------------------------------
function Test-FreeSpaceFor {
    param([Parameter(Mandatory)] $Plan)
    try {
        $qualifier = Split-Path -Qualifier $ProjectRoot          # 例: "S:"
        $disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter ("DeviceID='" + $qualifier + "'")
        if (-not $disk) { Write-Warn '空き容量を確認できませんでした。'; return }
        $freeGB = [Math]::Round($disk.FreeSpace / 1GB, 1)
        Write-Info ($qualifier + ' ドライブの空き容量: ' + $freeGB + ' GB（必要: 約 ' + $Plan.NeedGB + ' GB）')
        if ($freeGB -lt $Plan.NeedGB) {
            Write-Warn '空き容量が不足気味です。足りないと途中で失敗します。'
            Write-Info '不要なファイルを整理してから実行してください。'
        } else {
            Write-Good '空き容量は足りています。'
        }
    } catch {
        Write-Warn '空き容量を確認できませんでした（処理は続けます）。'
    }
}

# ---------------------------------------------------------------------------
# install_ltx.ps1 が失敗したときの、日本語の手当て
# ---------------------------------------------------------------------------
function Show-ModelInstallFailureHelp {
    param([Parameter(Mandatory)] $Plan)

    $modelDir = Join-Path $ModelsDir $Plan.Id
    $batName  = 'install-' + $Plan.Id + '.bat'

    Write-Host ''
    Write-Host ('=' * 74) -ForegroundColor Red
    Write-Bad ($Plan.DisplayName + ' の追加に失敗しました。')
    Write-Host ('=' * 74) -ForegroundColor Red
    Write-Host ''
    Write-Info '理由は上の英語のメッセージにあります。'
    if ($script:transcriptPath) { Write-Info ('記録: ' + $script:transcriptPath) }
    Write-Host ''
    Write-Info '会社や学校の回線ではダウンロードが失敗することがあります。'
    Write-Info ('ウイルス対策ソフトが原因のときは models フォルダ（' + $ModelsDir + '）を除外設定に追加して再実行してください。')

    Write-Host ''
    Write-Host ('-' * 74) -ForegroundColor Yellow
    Write-Warn ('次のフォルダを、フォルダごと削除しないでください: ' + $modelDir)
    Write-Info 'ここには、配布物に含まれない＝二度と取り直せないあなたの資産が同居していることがあります'
    Write-Info '（あなたが集めた LoRA、自分で変換した GGUF など）。'
    Write-Host ('-' * 74) -ForegroundColor Yellow

    Write-Host ''
    Write-Info 'ファイルが一部だけ欠けているときの直しかたは、次の 1 手順だけです。'
    Write-Info '  1. 上の英語の一覧表で MISSING と書かれた行を探します。'
    Write-Info '  2. その行に書かれているファイルを 1 個だけ削除します（フォルダではありません）。'
    Write-Info ('  3. ' + $batName + ' をもう一度実行します。そのファイルだけ取り直します。')
    Write-Info 'すべて PASS になるまで、この 1 手順を繰り返してください。'

    # 取得の途中で止まったときだけ案内する。検証の表で MISSING が出て終わった
    # 場合は、取得そのものは終わっていて .dl は残っていない。
    $stagingDir = Join-Path $ModelsDir '.dl'
    if (Test-Path -LiteralPath $stagingDir) {
        Write-Host ''
        Write-Info ('途中まで取れたファイルは ' + $stagingDir + ' の中に残してあります。')
        Write-Info ('消さずにそのまま ' + $batName + ' を実行すると、続きから取得します。')
    }
}

# ===========================================================================
# ここから本編
# ===========================================================================
try { $Host.UI.RawUI.WindowTitle = 'Nz-Videomni ベースモデルの追加（' + $BaseModel + '）' } catch { }

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

# --- 二重起動ガード（setup.bat と同じ鍵を使う）---------------------------
if (Test-Path -LiteralPath $LockFile) {
    $otherPid = 0
    $raw = ''
    try { $raw = (Get-Content -LiteralPath $LockFile -Raw).Trim() } catch { }
    if ([int]::TryParse($raw, [ref] $otherPid) -and $otherPid -gt 0 -and $otherPid -ne $PID) {
        $proc = Get-Process -Id $otherPid -ErrorAction SilentlyContinue
        if ($proc -and $proc.ProcessName -match 'powershell|pwsh') {
            Write-Host ''
            Write-Warn 'すでに導入処理が動いています（setup.bat か、別の install-*.bat）。'
            Write-Info '先に開いた画面が終わるのを待ってから、もう一度実行してください。'
            Write-Info ('（プロセス番号 ' + $otherPid + '）')
            Write-Info ('止まったままになる場合は ' + $LockFile + ' を削除してから再実行してください。')
            Write-Host ''
            exit 1
        }
    }
    Remove-Item -LiteralPath $LockFile -Force -ErrorAction SilentlyContinue
}
Set-Content -LiteralPath $LockFile -Value ([string] $PID) -Encoding ascii

$exitCode = 0
$plan = $null
$transcribing = $false
$transcriptPath = ''
try {
    # 記述子を読むのはロックを取った後。読めなければ Get-BaseModelPlan が日本語で
    # 案内して終了するが、その場合も finally を通ってロックは外れる。
    $plan = Get-BaseModelPlan -Id $BaseModel

    $transcriptPath = Join-Path $LogsDir ('install_' + $plan.Id + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
    try {
        Start-Transcript -LiteralPath $transcriptPath | Out-Null
        $transcribing = $true
    } catch {
        # 記録が始まらなかったのにパスだけ案内すると、存在しないファイルを探させる
        # ことになる。空にして「記録:」の行を出さない。
        $transcriptPath = ''
        Write-Warn '記録を開始できませんでした（処理は続けます）。'
    }

    Write-Host ''
    Write-Host '========================================================================' -ForegroundColor Cyan
    Write-Host ('   Nz-Videomni  ' + $plan.DisplayName + ' の追加') -ForegroundColor Cyan
    Write-Host '========================================================================' -ForegroundColor Cyan
    Write-Host ''
    Write-Info ('作業フォルダ: ' + $ProjectRoot)

    Write-Head 'setup.bat が済んでいるか確認します'
    if (-not (Test-SetupDone)) {
        $exitCode = 1
    } else {
        Write-Good 'Python 環境とダウンロード道具は揃っています。'

        Write-Head 'パソコンの状態を確認します'
        Test-FreeSpaceFor -Plan $plan

        Write-Head 'これから行うこと'
        Write-Info ($plan.DisplayName + ' の重みファイルを追加します。ほかのベースモデルの環境やモデルには手を加えません。')
        Write-Info ('取得するファイル: ' + $plan.FileCount + ' 個（空き容量は ' + $plan.NeedGB + ' GB ほど見ておいてください）')
        Write-Info '回線速度により 1〜2 時間かかります。途中で閉じても、再実行で続きから再開します。'
        Write-Info 'スリープすると通信が止まるので、電源設定でスリープを「なし」にしてください。'
        Write-Info 'すでに揃っているファイルは取り直しません。'
        Write-Host ''

        # tools/ をこのプロセスの PATH の先頭へ（システム側の設定は変更しない）。
        $env:PATH = $UvDir + ';' + $FfmpegBin + ';' + $env:PATH

        Write-Head ($plan.DisplayName + ' の重みを取得します')
        Write-Info 'ここから先は英語表示になります。'
        Write-Host ''

        $installScript = Join-Path $PSScriptRoot 'install_ltx.ps1'
        if (-not (Test-Path -LiteralPath $installScript)) {
            throw ('導入スクリプトが見つかりません: ' + $installScript)
        }

        $installError = $null
        $global:LASTEXITCODE = 0
        try {
            # ドットソースではなく & で呼ぶ。install_ltx.ps1 は exit を使うので、
            # ドットソースするとこちらのスクリプトごと終了してしまう。
            # 出力には一切手を加えない（Tee-Object などを挟むと hf の進捗バーが消える）。
            #
            # -BaseModel: この実行で取得し、検証の表に出す記述子を1つに絞る。
            # -SkipVenv:  Python 環境は setup.bat が作ったものをそのまま使う。
            # -SkipMigrate: 古いフォルダ構成の移動と config.yaml の書き換えは
            #               setup.bat の仕事なので、ここでは触らない。
            & $installScript -BaseModel $plan.Id -SkipVenv -SkipMigrate
            if ($LASTEXITCODE -ne 0) {
                $installError = ('install_ltx.ps1 exited with code ' + $LASTEXITCODE)
            }
        } catch {
            # install_ltx.ps1 は throw と exit 1 の両方で失敗を表すため、どちらも拾う。
            $installError = $_.Exception.Message
            Write-Host ''
            Write-Host $installError -ForegroundColor Red
        }

        if ($installError) {
            # 英語のメッセージを先に出し切ってから日本語の案内を出す（順番が逆だと
            # 英語のスタックトレースに押し流されて画面の外へ消える）。
            Show-ModelInstallFailureHelp -Plan $plan
            $exitCode = 1
        } else {
            Write-Host ''
            Write-Host ('   ' + $plan.DisplayName + ' の追加が完了しました') -ForegroundColor Green
            Write-Host ''
            if ($plan.OptIn) {
                # ベースモデルではないものは、画面上部の一覧に現れない（追尾は
                # ベースモデルと無関係。Docs/OBJECT_TRACKING_DESIGN.md §7）。
                # 共通の文面のままだと、存在しない項目を一覧から探させてしまう。
                Write-Info ('次: ' + (Join-Path $ProjectRoot 'run.bat') + ' をダブルクリックしてアプリを起動し、AviUtl2 の操作パネルの')
                Write-Info '    Toolbox タブで「物体追尾」が使えることを確認してください。'
                Write-Info 'すでに画面を開いている場合は、いったん閉じて開き直してください。'
                Write-Info '（導入の有無は画面を開いたときにしか読み直しません。）'
                Write-Info '使い方: 部分フィルタを置いて枠を合わせ、タイムラインで右クリック →「物体追尾（部分フィルタを使用）」'
            } else {
                Write-Info ('次: ' + (Join-Path $ProjectRoot 'run.bat') + ' をダブルクリックし、画面上部のベースモデルの一覧から')
                Write-Info ('    「' + $plan.DisplayName + '」を選んでください。')
                Write-Info 'すでに画面を開いている場合は、いったん閉じて開き直してください。'
                Write-Info '（一覧は画面を開いたときにしか読み直さないため、開いたままでは「未導入」のままに見えます。）'
            }
        }
    }
} catch {
    if ($exitCode -eq 0) {
        Write-Host ''
        Write-Bad $_.Exception.Message
        if ($plan) { Show-ModelInstallFailureHelp -Plan $plan }
        $exitCode = 1
    }
} finally {
    Write-Host ''
    if ($transcriptPath) { Write-Info ('記録: ' + $transcriptPath) }
    Write-Host ''
    if ($transcribing) { try { Stop-Transcript | Out-Null } catch { } }
    Remove-Item -LiteralPath $LockFile -Force -ErrorAction SilentlyContinue
}

# -File で呼ばれた PowerShell は exit を書かないと常に 0 を返す。必ず明示する。
exit $exitCode
