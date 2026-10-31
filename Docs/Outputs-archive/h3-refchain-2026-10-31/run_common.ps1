# Shared by run_cr.ps1 (dot-sourced). Not meant to be run on its own.
# Copy of Docs/Outputs-archive/h3-chain-2026-10-29/run_common.ps1 (sec. 1-109 plan pure-knitting-shell.md sec. 5):
# the folder and script names changed; Invoke-Preflight also stops on another process on the GPU (more than
# 2,000 MiB used), on a commit limit (RAM + page file) under 45 GB and on a missing file (the checks copied from
# Docs/Outputs-archive/h3-chain-gates-2026-10-29/run_gates.ps1). No -SkipPreflight.
#
# One condition = one process: Invoke-Cond starts
#   cmd /c ""<python>" <script> <args> --report "<archive>\<Name>.json" > "<archive>\<Name>.console.log" 2>&1"
# from the repo root with PYTHONPATH=<repo root> and PYTHONUTF8=1, waits at most -TimeoutSec, and on a
# timeout kills the whole process tree (taskkill /T /F), appends "TIMEOUT" to the console log and writes
# {"timeout": true} as the report if the script did not write one. A fault that hangs or kills the
# process therefore stops that condition only.

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $Repo
$env:PYTHONPATH = $Repo
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'   # no __pycache__ in the tracked archive (cr_chain imports analyze_refchain)
$Py = Join-Path $Repo '.venv-engine-h3\Scripts\python.exe'
$Arch = $PSScriptRoot
$Rel = 'Docs\Outputs-archive\h3-refchain-2026-10-31'
$Tmp = Join-Path $Repo 'outputs\h3-refchain-2026-10-31'
New-Item -ItemType Directory -Force $Tmp | Out-Null
$C = Join-Path $Arch 'cr_chain.py'
$An = Join-Path $Arch 'analyze_refchain.py'
$Fl2va = 'models\MiniMaxH3\Weights\minimax_h3_fl2va_pruned_int8_convrot.safetensors'
$Ref2va = 'models\MiniMaxH3\Weights\minimax_h3_ref2va_pruned_int8_convrot.safetensors'

function Invoke-Preflight {
    # Plan sec. 7: nvidia-smi shown, AviUtl2 not running, no other process on the GPU, the commit limit,
    # the two weights, the two pictures, the venv and weight_identity.json present. Throws 'STOP: ...'.
    Write-Host '--- nvidia-smi (compute processes) ---'
    try { & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv } catch { Write-Host "nvidia-smi: $_" }
    try { & nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv } catch { }
    $aviutl = Get-Process -Name 'aviutl2' -ErrorAction SilentlyContinue
    if ($aviutl) { throw 'STOP: AviUtl2 is running. Close it first.' }
    $need = @(
        $Fl2va, $Ref2va,
        'models\MiniMaxH3\TextEncoder\qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
        'models\MiniMaxH3\TextEncoder\qwen3vl_tokenizer',
        'models\MiniMaxH3\VAE\minimax_h3_video_vae_fp16.safetensors',
        'models\MiniMaxH3\VAE\minimax_h3_audio_vae_fp32.safetensors',
        'Docs\Outputs-archive\h3-ref2va-2026-10-28\17fde0c1-image.png',
        'Docs\Outputs-archive\h3-ref2va-2026-10-28\d80320dd-image.png',
        '.venv-engine-h3\Scripts\python.exe', 'weight_identity.json',
        "$Rel\cr_chain.py", "$Rel\analyze_refchain.py", "$Rel\prompts_c0.json", "$Rel\prompts_c2.json",
        "$Rel\prompts_c3.json", "$Rel\refs_c2.json", "$Rel\refs_c3.json"
    )
    foreach ($n in $need) {
        if (-not (Test-Path (Join-Path $Repo $n))) { throw "STOP: missing: $n" }
    }
    Write-Host 'weights, pictures, venv, weight_identity.json, scripts: present'
    $gpu = & nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    $used = [int](($gpu -split ',')[0].Trim())
    if ($used -gt 2000) { throw "STOP: GPU already holds $used MiB: another process is on the GPU" }
    try {
        $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
        $pf = (Get-CimInstance Win32_PageFileUsage | Measure-Object -Property AllocatedBaseSize -Sum).Sum / 1024
        $limit = [math]::Round($ram + $pf, 1)
        Write-Host "RAM $ram GB + page file allocated $([math]::Round($pf,1)) GB = commit limit about $limit GB"
        if ($limit -lt 45) { throw 'STOP: the commit limit (RAM + page file) is under 45 GB' }
    } catch [System.Management.Automation.RuntimeException] {
        if ("$_" -like 'STOP:*') { throw }
        Write-Host "memory check skipped: $_"
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
