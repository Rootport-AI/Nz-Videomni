# Shared by run_gates.ps1 (dot-sourced). Not meant to be run on its own.
# Copy of Docs/Outputs-archive/h3-chain-2026-10-29/run_common.ps1 (sec. 1-106 GPU gates G-C0..G-C2): $Tmp and the
# script names changed. Invoke-Cond returns the exit code (-1 on a timeout) so run_gates.ps1 can stop.
#
# One condition = one process: Invoke-Cond starts
#   cmd /c ""<python>" <script> <args> --report "<archive>\<Name>.json" > "<archive>\<Name>.console.log" 2>&1"
# from the repo root with PYTHONPATH=<repo root> and PYTHONUTF8=1, waits at most -TimeoutSec, and on a
# timeout kills the whole process tree (taskkill /T /F), appends "TIMEOUT" to the console log and writes
# {"timeout": true} as the report if the script did not write one. A fault that hangs or kills the
# process therefore stops that condition only.

param([switch]$SkipPreflight)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $Repo
$env:PYTHONPATH = $Repo
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'   # no __pycache__ in the tracked archive (c_chain imports analyze_chain)
$Py = Join-Path $Repo '.venv-engine-h3\Scripts\python.exe'   # httpx, numpy and PyAV are here (g_chain.py needs no product import)
$AppPy = Join-Path $Repo '.venv\Scripts\python.exe'          # the backend (main.py)
$Arch = $PSScriptRoot
$Rel = 'Docs\Outputs-archive\h3-chain-gates-2026-10-29'
$RelChain = 'Docs\Outputs-archive\h3-chain-2026-10-29'
$Tmp = Join-Path $Repo 'outputs\h3-chain-gates-2026-10-29'
New-Item -ItemType Directory -Force $Tmp | Out-Null
$G = Join-Path $Arch 'g_chain.py'
$C = Join-Path $Repo "$RelChain\c_chain.py"

function Invoke-Preflight {
    # Plan sec. 3: AviUtl2 not running, no other process on the GPU.
    Write-Host '--- nvidia-smi (compute processes) ---'
    try { & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv } catch { Write-Host "nvidia-smi: $_" }
    try { & nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv } catch { }
    $aviutl = Get-Process -Name 'aviutl2' -ErrorAction SilentlyContinue
    if ($aviutl -and -not $SkipPreflight) {
        throw 'AviUtl2 is running. Close it first (or pass -SkipPreflight).'
    }
}

function Invoke-Cond {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$Script,
        [Parameter(Mandatory)] [string[]]$CondArgs,
        [int]$TimeoutSec = 600
    )
    $report = Join-Path $Arch "$Name.json"
    $log = Join-Path $Arch "$Name.console.log"
    if (Test-Path $report) { Remove-Item $report -Force }
    $inner = "`"$Py`" `"$Script`" " + ($CondArgs -join ' ') + " --report `"$report`" > `"$log`" 2>&1"
    Write-Host ("[{0}] {1}: {2}" -f (Get-Date -Format 'HH:mm:ss'), $Name, ($CondArgs -join ' '))
    $started = Get-Date
    $p = Start-Process -FilePath $env:ComSpec -ArgumentList "/d /c `"$inner`"" -NoNewWindow -PassThru
    $null = $p.Handle   # keep the handle so ExitCode is readable after the exit (PowerShell 5.1)
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        & taskkill.exe /T /F /PID $p.Id | Out-Null
        Add-Content -Path $log -Value "TIMEOUT after $TimeoutSec s (process tree killed by run_common.ps1)" -Encoding utf8
        if (-not (Test-Path $report)) {
            Set-Content -Path $report -Value ("{`"timeout`": true, `"timeout_sec`": $TimeoutSec, `"name`": `"$Name`"}") -Encoding utf8
        }
        Write-Host "    TIMEOUT ($TimeoutSec s)"
        return -1
    }
    $p.WaitForExit()
    $sec = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    if (-not (Test-Path $report)) {
        Set-Content -Path $report -Value ("{`"no_report`": true, `"exit_code`": $($p.ExitCode), `"name`": `"$Name`"}") -Encoding utf8
    }
    Write-Host "    exit $($p.ExitCode) in $sec s -> $report"
    return $p.ExitCode
}
