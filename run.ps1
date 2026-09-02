# run.ps1 — Nz-Videomni バックエンドの起動（run.bat から呼び出される本体）
#
# このファイルは必ずリポジトリ直下に置くこと。$PSScriptRoot を基準に
# .python / .venv / main.py を探しているため、scripts\ などへ移すと
# すべて 1 階層ずれて動かなくなる。
#
# 環境の分離方針（仕様 2.5）:
#   - Python の実行ファイルを含め、すべてこのプロジェクトの中に置く。
#   - システムの Python には触れない。環境変数も永続化しない。
#   - 以下の環境変数はこのプロセスの中だけで有効。
#
# 使い方:
#   ./run.ps1                 # 自分のパソコンからだけ（ポート 18620）
#   ./run.ps1 --listen        # 家庭内 LAN に公開（0.0.0.0 で待ち受け）
#   ./run.ps1 --port 19000
#   ./run.ps1 --dit-cpu-load     # DiT を CPU 上で構築し、ブロックを GPU へ流す（既定は有効）
#   ./run.ps1 --no-dit-cpu-load  # DiT を GPU 上で構築してから退避（約 16.9GB の山が戻る）
#
# このファイルは UTF-8（BOM 付き）で保存すること。Windows PowerShell 5.1 は
# BOM の無い UTF-8 を ANSI として読むため、BOM を落とすと日本語が化ける。
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

try { $Host.UI.RawUI.WindowTitle = 'Nz-Videomni サーバー（この画面は閉じないでください）' } catch { }

# Keep uv-managed Python inside the project (process-scoped).
$env:UV_PYTHON_INSTALL_DIR = "$PSScriptRoot\.python"

# setup.bat が用意した tools\ をこのプロセスの PATH の先頭へ。
# services/video_io.py は shutil.which で ffmpeg / ffprobe を探すので、
# これだけで Python 側は何も変えずにプロジェクト内の ffmpeg を使う。
$toolsUvDir = "$PSScriptRoot\tools\uv"
$toolsFfBin = "$PSScriptRoot\tools\ffmpeg\bin"
$pathPrefix = ""
if (Test-Path $toolsUvDir) { $pathPrefix += "$toolsUvDir;" }
if (Test-Path $toolsFfBin) { $pathPrefix += "$toolsFfBin;" }
if ($pathPrefix) { $env:PATH = $pathPrefix + $env:PATH }

# 設定ファイルが無いと、既定値のまま起動して「お試し表示」に落ちることがある。
# Test-Path 1 回だけの軽い確認にとどめる。
$configPath = "$PSScriptRoot\config.yaml"
if (-not (Test-Path $configPath)) {
    $configExample = "$PSScriptRoot\config.yaml.example"
    if (Test-Path $configExample) {
        Copy-Item -LiteralPath $configExample -Destination $configPath
        Write-Host "config.yaml が無かったので config.yaml.example から作りました。" -ForegroundColor Yellow
    } else {
        Write-Host "config.yaml が見つかりません。設定が既定値のままになります。" -ForegroundColor Yellow
    }
}

$python = "$PSScriptRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    # ここで throw してはいけない。$ErrorActionPreference = "Stop" のもとでは
    # その場でスクリプトが終わり、下の案内にたどり着かない。
    Write-Host ""
    Write-Host "先に setup.bat を実行してください。" -ForegroundColor Yellow
    Write-Host "  $PSScriptRoot\setup.bat"
    Write-Host "（開発者向け: scripts\install_ltx.ps1 -SkipModels でモデル無しの環境を作れます）"
    Write-Host ""
    exit 1
}

# ---------------------------------------------------------------------------
# 待ち受けポートの割り出し。--port が渡されていればそれを、無ければ既定値。
# ここで得たポートは二重起動の判定にだけ使う。利用者に見せる URL は
# main.py の起動バナーが必ず表示する（そちらが表示の正本）。
# ---------------------------------------------------------------------------
$argList = @($Args)
$port = 0
for ($i = 0; $i -lt $argList.Count; $i++) {
    if ($argList[$i] -eq "--port" -and ($i + 1) -lt $argList.Count) {
        $parsed = $argList[$i + 1] -as [int]
        if ($parsed) { $port = $parsed }
    } elseif ($argList[$i] -match '^--port=(\d+)$') {
        $port = [int] $Matches[1]
    }
}
if ($port -le 0) { $port = 18620 }

# 既に同じポートで待ち受けているなら、それは二枚目の run.bat である可能性が高い。
# ここで案内しないと「setup.bat を実行してください」という誤った誘導になる。
$portInUse = $false
try {
    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    $portInUse = (@($listeners | Where-Object { $_.Port -eq $port }).Count -gt 0)
} catch {
    # 確認できなければ、そのまま起動を試みる。
}
if ($portInUse) {
    Write-Host ""
    Write-Host "すでに起動しています。先に開いた画面をそのまま使ってください。" -ForegroundColor Yellow
    Write-Host "見当たらないときは、黒い画面をすべて閉じてから run.bat を実行し直してください。"
    Write-Host ""
    exit 0
}

Write-Host ""
Write-Host "サーバー起動中…" -ForegroundColor Cyan
Write-Host ""

# 注意: & $python ... 2>&1 の形は使わないこと。Windows PowerShell 5.1 では
# $ErrorActionPreference = "Stop" と 2>&1 を併用すると、終了コードが 0 でも
# 標準エラー出力があるだけで停止する。Python も ffmpeg も進捗を標準エラーに出す。
& $python "$PSScriptRoot\main.py" @Args
$code = $LASTEXITCODE

Write-Host ""
if ($code -ne 0) {
    Write-Host "サーバーが異常終了しました。理由は上の英語のメッセージにあります。" -ForegroundColor Red
    Write-Host "git pull の直後なら setup.bat を再実行してください。"
    Write-Host "ログ: $PSScriptRoot\logs\server.log"
    Write-Host ""
} else {
    Write-Host "サーバーを終了しました。" -ForegroundColor Green
    Write-Host ""
}

# -File で呼ばれた PowerShell は exit を書かないと常に 0 を返す。必ず明示する。
exit $code
