#Requires -Version 5.1
<#
.SYNOPSIS
    Nz-LTX23 バックエンドの導入スクリプト（setup.bat から呼び出される本体）。

.DESCRIPTION
    PowerShell を自分で開けない利用者でも、setup.bat をダブルクリックするだけで
    導入が終わるようにするための入口。ここでは次の順番で処理する。

      1. 二重起動ガード（logs/.setup.lock）
      2. 記録の開始（logs/setup_<日時>.log）
      3. 事前チェック（空き容量／ページファイル／GPU）— 警告のみで続行する
      4. これから何が起きるかの説明
      5. tools/uv と tools/ffmpeg の用意（プロジェクトの中に閉じ込める）
      6. config.yaml が無ければ config.yaml.example から複製
      7. tools/ を PATH の先頭へ（このプロセスの中だけ）
      8. scripts/install_ltx.ps1 の実行（Python 環境とモデルの取得）
      9. 終了案内

    このファイルは UTF-8（BOM 付き）で保存すること。Windows PowerShell 5.1 は
    BOM の無い UTF-8 を ANSI として読むため、BOM を落とすと日本語が化ける。
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
# 進捗バーの描画は Windows PowerShell 5.1 の Invoke-WebRequest を極端に遅くする
# （公式 Issue の実測で 10〜70 倍）。必ず切っておき、代わりに自前で案内を出す。
$ProgressPreference = 'SilentlyContinue'

# TLS のバージョンは指定しない。Tls12 を固定すると TLS 1.3 への交渉を塞ぐ。
# 現行の Windows 10/11 は既定で TLS 1.2/1.3 を使う。

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot

$UvVersion   = '0.11.32'   # 固定。決定論方針（freeze 固定・git rev 固定）に合わせる。
$UvUrl       = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
$FfmpegUrl   = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'

$ToolsDir    = Join-Path $ProjectRoot 'tools'
$UvDir       = Join-Path $ToolsDir 'uv'
$UvExe       = Join-Path $UvDir 'uv.exe'
$FfmpegDir   = Join-Path $ToolsDir 'ffmpeg'
$FfmpegBin   = Join-Path $FfmpegDir 'bin'
$FfmpegExe   = Join-Path $FfmpegBin 'ffmpeg.exe'
$FfprobeExe  = Join-Path $FfmpegBin 'ffprobe.exe'

$LogsDir     = Join-Path $ProjectRoot 'logs'
$LockFile    = Join-Path $LogsDir '.setup.lock'

# ---------------------------------------------------------------------------
# 画面表示の小道具
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
# ダウンロード
# ---------------------------------------------------------------------------
function Invoke-FileDownload {
    param(
        [Parameter(Mandatory)] [string] $Uri,
        [Parameter(Mandatory)] [string] $OutFile,
        [Parameter(Mandatory)] [string] $Label
    )
    Write-Info ($Label + ' をダウンロード中…')
    Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing
    if (-not (Test-Path -LiteralPath $OutFile)) {
        throw ($Label + ' のダウンロードに失敗しました（ファイルができていません）。')
    }
    Write-Info ($Label + ' 取得完了（' + (Format-Size (Get-Item -LiteralPath $OutFile).Length) + '）')
}

# 一時展開先。tools\ の中に作るので、確定の Move-Item が同じドライブ内の
# 名前替えになる（別ドライブだと巨大なコピーになってしまう）。
function New-StagingDir {
    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    $dir = Join-Path $ToolsDir ('.staging-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    return $dir
}

# ---------------------------------------------------------------------------
# 事前チェック（すべて警告のみ。ここで処理を止めない）
# ---------------------------------------------------------------------------
function Test-FreeSpace {
    # 必要容量の表記は README §1「必要な空き容量の内訳」と揃えること（約 38〜40 GB）。
    # $needGB はその上端＝判定のしきい値。
    $needGB = 40
    try {
        $qualifier = Split-Path -Qualifier $ProjectRoot          # 例: "S:"
        $disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter ("DeviceID='" + $qualifier + "'")
        if (-not $disk) { Write-Warn '空き容量を確認できませんでした。'; return }
        $freeGB = [Math]::Round($disk.FreeSpace / 1GB, 1)
        Write-Info ($qualifier + ' ドライブの空き容量: ' + $freeGB + ' GB（必要: 約 38〜40 GB）')
        if ($freeGB -lt $needGB) {
            Write-Warn '空き容量が不足気味です（モデル 約 30 GB＋Python 環境 7〜8 GB＋道具類 約 0.4 GB）。'
            Write-Info '足りないと途中で失敗します。不要なファイルを整理してから実行してください。'
        } else {
            Write-Good '空き容量は足りています。'
        }
    } catch {
        Write-Warn '空き容量を確認できませんでした（処理は続けます）。'
    }
}

function Test-PageFile {
    try {
        $cs = Get-CimInstance -ClassName Win32_ComputerSystem
        if ($cs.AutomaticManagedPagefile) {
            Write-Good 'ページファイルは「システム管理サイズ」です。'
            return
        }
        $settings = @(Get-CimInstance -ClassName Win32_PageFileSetting -ErrorAction SilentlyContinue)
        if ($settings.Count -eq 0) {
            Write-Warn 'ページファイルが無効になっています。'
        } else {
            $desc = ($settings | ForEach-Object { $_.Name + '（初期 ' + $_.InitialSize + ' MB / 最大 ' + $_.MaximumSize + ' MB）' }) -join ', '
            Write-Warn ('ページファイルが手動設定です: ' + $desc)
        }
        Write-Info 'ページファイルが小さいと、生成の途中でログも残さずに落ちます。'
        Write-Info 'システムのプロパティ→詳細設定→パフォーマンス→詳細設定→仮想メモリ で「自動的に管理する」を有効にしてください。'
    } catch {
        Write-Warn 'ページファイルの設定を確認できませんでした（処理は続けます）。'
    }
}

function Test-Gpu {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        try {
            $line = & $smi.Source --query-gpu=name,memory.total --format=csv,noheader
            if ($LASTEXITCODE -eq 0 -and $line) {
                $first = @($line)[0]
                Write-Info ('GPU: ' + $first)
                $mib = 0
                if ($first -match '(\d+)\s*MiB') { $mib = [int] $Matches[1] }
                if ($mib -gt 0 -and $mib -lt 16000) {
                    Write-Warn ('VRAM ' + [Math]::Round($mib / 1024, 1) + ' GB。16 GB 未満では生成が失敗しやすくなります。')
                } else {
                    Write-Good 'NVIDIA GPU を確認しました。'
                }
                return
            }
        } catch {
            # 下の汎用チェックに落とす。
        }
    }
    try {
        $names = @(Get-CimInstance -ClassName Win32_VideoController | ForEach-Object { $_.Name })
        if ($names.Count -gt 0) { Write-Info ('画面表示のデバイス: ' + ($names -join ', ')) }
    } catch { }
    Write-Warn 'NVIDIA GPU を確認できませんでした。生成には NVIDIA 製 GPU（VRAM 16 GB 以上）が必要です。'
    Write-Info '導入自体は最後まで進みます（GPU が要るのは生成のとき）。'
}

# ---------------------------------------------------------------------------
# tools/uv と tools/ffmpeg の用意
#
# 判定は「フォルダがあるか」ではなく「実行ファイルがあるか」で行う。フォルダ判定に
# すると、途中で中断して空の bin/ が残ったときに永久にスキップされてしまう。
# ---------------------------------------------------------------------------
function Install-Uv {
    if (Test-Path -LiteralPath $UvExe) {
        Write-Skip ('uv は用意済みです（' + $UvExe + '）')
        return
    }
    if (Test-Path -LiteralPath $UvDir) {
        Write-Info '中途半端な tools\uv が残っていたので作り直します。'
        Remove-Item -LiteralPath $UvDir -Recurse -Force
    }
    $stage = New-StagingDir
    try {
        $zip = Join-Path $stage 'uv.zip'
        Invoke-FileDownload -Uri $UvUrl -OutFile $zip -Label ('uv ' + $UvVersion + '（約 20 MB）')
        $unpack = Join-Path $stage 'unpack'
        Write-Info '展開中…'
        Expand-Archive -LiteralPath $zip -DestinationPath $unpack -Force
        $found = Get-ChildItem -LiteralPath $unpack -Recurse -File -Filter 'uv.exe' | Select-Object -First 1
        if (-not $found) { throw 'ダウンロードした uv の書庫に uv.exe が入っていませんでした。' }
        Move-Item -LiteralPath $found.Directory.FullName -Destination $UvDir
        Write-Good ('uv を用意しました（' + $UvExe + '）')
    } finally {
        if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

function Install-Ffmpeg {
    if ((Test-Path -LiteralPath $FfmpegExe) -and (Test-Path -LiteralPath $FfprobeExe)) {
        Write-Skip ('ffmpeg と ffprobe は用意済みです（' + $FfmpegBin + '）')
        return
    }
    if (Test-Path -LiteralPath $FfmpegDir) {
        Write-Info '中途半端な tools\ffmpeg が残っていたので作り直します。'
        Remove-Item -LiteralPath $FfmpegDir -Recurse -Force
    }
    $stage = New-StagingDir
    try {
        $zip = Join-Path $stage 'ffmpeg.zip'
        Invoke-FileDownload -Uri $FfmpegUrl -OutFile $zip -Label 'ffmpeg（約 104 MB）'
        $unpack = Join-Path $stage 'unpack'
        Write-Info '展開中…（少し時間がかかります）'
        Expand-Archive -LiteralPath $zip -DestinationPath $unpack -Force
        # 展開後のフォルダ名にはバージョンが入る（ffmpeg-8.1.2-essentials_build など）。
        # 決め打ちにできないので、実行ファイルを探して場所を割り出す。
        $found = Get-ChildItem -LiteralPath $unpack -Recurse -File -Filter 'ffmpeg.exe' | Select-Object -First 1
        if (-not $found) { throw 'ダウンロードした ffmpeg の書庫に ffmpeg.exe が入っていませんでした。' }
        $binDir = $found.Directory.FullName
        if (-not (Test-Path -LiteralPath (Join-Path $binDir 'ffprobe.exe'))) {
            throw 'ダウンロードした ffmpeg の書庫に ffprobe.exe が入っていませんでした。'
        }
        $buildRoot = Split-Path -Parent $binDir
        Move-Item -LiteralPath $buildRoot -Destination $FfmpegDir
        Write-Good ('ffmpeg と ffprobe を用意しました（' + $FfmpegBin + '）')
    } finally {
        if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

function Test-Tools {
    # 実際に動かして確かめる。置いてあるだけでは動く保証にならない。
    # （Select-Object -First をパイプに挟むと途中で実行ファイルを打ち切ってしまい
    #   終了コードが当てにならなくなるので、いったん全部受け取ってから 1 行目を出す）
    $uvVer = @(& $UvExe --version)
    if ($LASTEXITCODE -ne 0) { throw 'uv.exe を実行できませんでした。' }
    Write-Info ('uv     : ' + ($uvVer -join ' '))

    $ffVer = @(& $FfmpegExe -hide_banner -version)
    if ($LASTEXITCODE -ne 0) { throw 'ffmpeg.exe を実行できませんでした。' }
    if ($ffVer.Count -gt 0) { Write-Info ('ffmpeg : ' + $ffVer[0]) }

    $fpVer = @(& $FfprobeExe -hide_banner -version)
    if ($LASTEXITCODE -ne 0) { throw 'ffprobe.exe を実行できませんでした。' }
    if ($fpVer.Count -gt 0) { Write-Info ('ffprobe: ' + $fpVer[0]) }
}

# ---------------------------------------------------------------------------
# .mcp.json の生成（AIエージェント連携・MCPサーバー登録）
#
# 絶対パスで書き出す。$CLAUDE_PROJECT_DIR のような相対解決に頼ると、開発機で
# 親フォルダをワークスペースとして開いた場合や、エンドユーザーがセットアップ前
# （.venv が無い状態）でクライアントを設定した場合に壊れる。.venv\Scripts\
# python.exe が実際に存在することを確認してから書き出す（無ければ、まだ
# セットアップの途中ということなので黙ってスキップする）。
# ---------------------------------------------------------------------------
function New-McpJson {
    $mcpJsonPath = Join-Path $ProjectRoot '.mcp.json'
    $pythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        Write-Warn 'MCPサーバー設定 (.mcp.json) を生成できませんでした（.venv がまだありません）。'
        return
    }
    $mcpConfig = [ordered]@{
        mcpServers = [ordered]@{
            'nz-ltx23' = [ordered]@{
                command = $pythonExe
                args    = @('-m', 'mcp_server')
                env     = [ordered]@{ PYTHONUTF8 = '1' }
            }
        }
    }
    ($mcpConfig | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $mcpJsonPath -Encoding utf8
    Write-Good ('MCPサーバー設定を作りました（' + $mcpJsonPath + '）')
    Write-Info 'Claude Code 等のMCPクライアントでこのフォルダを開くと、承認確認のうえで使えるようになります。'
}

# ---------------------------------------------------------------------------
# install_ltx.ps1 が失敗したときの、日本語の手当て
# ---------------------------------------------------------------------------
function Show-InstallFailureHelp {
    Write-Host ''
    Write-Host ('=' * 74) -ForegroundColor Red
    Write-Bad '導入に失敗しました。'
    Write-Host ('=' * 74) -ForegroundColor Red
    Write-Host ''
    Write-Info '理由は上の英語のメッセージにあります。'
    Write-Info ('記録: ' + $script:transcriptPath)
    Write-Host ''
    Write-Info '会社や学校の回線ではダウンロードが失敗することがあります。'
    Write-Info ('ウイルス対策ソフトが原因のときは tools フォルダ（' + $ToolsDir + '）を除外設定に追加して再実行してください。')

    Write-Host ''
    Write-Info 'モデルが一部だけ欠けている場合は、下のフォルダを削除して setup.bat を再実行すると取り直します。'
    Write-Info ('  ' + (Join-Path $ProjectRoot 'models\ltx-2.3'))
    Write-Info ('  ' + (Join-Path $ProjectRoot 'models\ltx-2.3-components'))
    Write-Info ('  ' + (Join-Path $ProjectRoot 'models\ltx-2.3-ic-lora'))
    Write-Info ('  ' + (Join-Path $ProjectRoot 'models\gemma-3-12b-it-gguf'))
    Write-Info ('  ' + (Join-Path $ProjectRoot 'models\gemma-3-12b-it-tokenizer'))
    Write-Info ('  ' + (Join-Path $ProjectRoot 'models\preprocessors'))

    # models\ltx-2.3-gguf だけは上のリストに入れてはいけない。ここには利用者が自分で
    # 用意した GGUF（当プロジェクトの配布物に含まれず、取り直せないもの）が同居しうる。
    # フォルダごと消させると、再取得できない資産を失わせることになる。
    Write-Host ''
    Write-Warn ('次のフォルダはフォルダごと削除しないでください: ' + (Join-Path $ProjectRoot 'models\ltx-2.3-gguf'))
    Write-Info '  自分で用意した GGUF が同居していることがあり、それは取り直せません。'
    Write-Info '  取り直せるのは次の 1 ファイルだけです。消すならこれだけにしてください。'
    Write-Info ('  ' + (Join-Path $ProjectRoot 'models\ltx-2.3-gguf\LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf'))
}

# ===========================================================================
# ここから本編
# ===========================================================================
try { $Host.UI.RawUI.WindowTitle = 'Nz-LTX23 セットアップ（setup.bat）' } catch { }

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

# --- 二重起動ガード -------------------------------------------------------
if (Test-Path -LiteralPath $LockFile) {
    $otherPid = 0
    $raw = ''
    try { $raw = (Get-Content -LiteralPath $LockFile -Raw).Trim() } catch { }
    if ([int]::TryParse($raw, [ref] $otherPid) -and $otherPid -gt 0 -and $otherPid -ne $PID) {
        $proc = Get-Process -Id $otherPid -ErrorAction SilentlyContinue
        if ($proc -and $proc.ProcessName -match 'powershell|pwsh') {
            Write-Host ''
            Write-Warn 'すでにセットアップが動いています。先に開いた画面が終わるのを待ってください。'
            Write-Info ('（プロセス番号 ' + $otherPid + '）')
            Write-Host ''
            exit 1
        }
    }
    Remove-Item -LiteralPath $LockFile -Force -ErrorAction SilentlyContinue
}
Set-Content -LiteralPath $LockFile -Value ([string] $PID) -Encoding ascii

$transcriptPath = Join-Path $LogsDir ('setup_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
$transcribing = $false
try {
    Start-Transcript -LiteralPath $transcriptPath | Out-Null
    $transcribing = $true
} catch {
    Write-Warn '記録を開始できませんでした（処理は続けます）。'
}

$exitCode = 0
try {
    Write-Host ''
    Write-Host '========================================================================' -ForegroundColor Cyan
    Write-Host '   Nz-LTX23 バックエンド セットアップ' -ForegroundColor Cyan
    Write-Host '========================================================================' -ForegroundColor Cyan
    Write-Host ''
    Write-Info ('作業フォルダ: ' + $ProjectRoot)

    Write-Head 'パソコンの状態を確認します'
    Test-FreeSpace
    Test-PageFile
    Test-Gpu

    Write-Head 'これから行うこと'
    Write-Info '1. 道具（uv / ffmpeg）を用意  2. 専用 Python 環境（約 7〜8 GB）  3. モデル取得（約 30 GB）'
    Write-Info '初回はモデル取得に時間がかかります（回線速度により 1〜3 時間）。'
    Write-Info '途中で閉じても、再実行で続きから再開します。'
    Write-Info 'スリープすると通信が止まるので、電源設定でスリープを「なし」にしてください。'
    Write-Host ''

    Write-Head '道具をそろえます（uv / ffmpeg）'
    Install-Uv
    Install-Ffmpeg
    Test-Tools

    Write-Head '設定ファイルを確認します'
    $configPath = Join-Path $ProjectRoot 'config.yaml'
    $examplePath = Join-Path $ProjectRoot 'config.yaml.example'
    if (Test-Path -LiteralPath $configPath) {
        Write-Skip ('config.yaml は既にあります（' + $configPath + '）')
    } elseif (Test-Path -LiteralPath $examplePath) {
        Copy-Item -LiteralPath $examplePath -Destination $configPath
        Write-Good 'config.yaml.example から config.yaml を作りました。'
    } else {
        Write-Warn 'config.yaml も config.yaml.example も見つかりません。'
        Write-Info '既定値のまま進みます（生成が「お試し表示」になることがあります）。'
    }

    # tools/ をこのプロセスの PATH の先頭へ。ここから先の uv / ffmpeg / ffprobe は
    # すべてプロジェクト内のものが使われる（システム側の設定は一切変更しない）。
    $env:PATH = $UvDir + ';' + $FfmpegBin + ';' + $env:PATH

    Write-Head 'Python 環境とモデルを用意します'
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
        & $installScript
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
        Show-InstallFailureHelp
        $exitCode = 1
    } else {
        Write-Host ''
        Write-Host '   セットアップ完了' -ForegroundColor Green
        Write-Host ''
        New-McpJson
        Write-Host ''
        Write-Info ('次: ' + (Join-Path $ProjectRoot 'run.bat') + ' をダブルクリック。')
        $pkg = Get-ChildItem -LiteralPath $ProjectRoot -Filter '*.au2pkg.zip' -File -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($pkg) {
            Write-Info ('AviUtl2 で使うには ' + $pkg.Name + ' を AviUtl2 のプレビュー画面へドラッグ。')
        } else {
            Write-Info 'AviUtl2 で使うには .au2pkg.zip を AviUtl2 のプレビュー画面へドラッグ。'
        }
        Write-Info 'AviUtl2 から使うときも run.bat の画面は開いたままにしてください。'
    }
} catch {
    if ($exitCode -eq 0) {
        Write-Host ''
        Write-Bad $_.Exception.Message
        Show-InstallFailureHelp
        $exitCode = 1
    }
} finally {
    Write-Host ''
    Write-Info ('記録: ' + $transcriptPath)
    Write-Host ''
    if ($transcribing) { try { Stop-Transcript | Out-Null } catch { } }
    Remove-Item -LiteralPath $LockFile -Force -ErrorAction SilentlyContinue
}

# -File で呼ばれた PowerShell は exit を書かないと常に 0 を返す。必ず明示する。
exit $exitCode
