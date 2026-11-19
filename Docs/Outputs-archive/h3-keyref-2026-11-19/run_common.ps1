# Shared by run_stageA.ps1 (dot-sourced). Not meant to be run on its own.
# Copy of Docs/Outputs-archive/h3-refchain-2026-10-31/run_common.ps1 (sec. 1-115, plan glistening-splashing-hoare.md
# sec. 6): the folder, the script and the files checked changed; Invoke-Cond takes a -Suffix (the dry run's
# reports sit beside the real ones without replacing them). Windows PowerShell 5.1; saved as UTF-8 with BOM, CRLF.
#
# One condition = one process: Invoke-Cond starts
#   cmd /d /c ""<python>" <script> <args> --report "<archive>\<Name><Suffix>.json" > "<archive>\<Name><Suffix>.console.log" 2>&1"
# from the repo root and WAITS for it (at most -TimeoutSec). Before it starts, the GPU must be free (Wait-GpuFree)
# and an earlier report / console log of the same name is moved aside with a time stamp. On a timeout it kills
# the whole process tree (taskkill /T /F), waits for the exit and for the GPU to be free again, appends "TIMEOUT"
# to the console log and writes {"timeout": true} as the report if the script did not write one (each of the
# two writes only reported when it fails). Returns the exit code (-1 = timeout).

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $Repo
# The adapter's child environment (services/engines/h3/adapter.py:925-933) plus the archive's hygiene.
$env:PYTHONPATH = $Repo
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TORCH_COMPILE_DISABLE = '1'
$Py = Join-Path $Repo '.venv-engine-h3\Scripts\python.exe'
$Arch = $PSScriptRoot
$Rel = 'Docs\Outputs-archive\h3-keyref-2026-11-19'
$OutDir = Join-Path $Repo 'outputs\h3-keyref-2026-11-19'
New-Item -ItemType Directory -Force $OutDir | Out-Null
$KrSingle = Join-Path $Arch 'kr_single.py'
$Fl2va = 'models\MiniMaxH3\Weights\minimax_h3_fl2va_pruned_int8_convrot.safetensors'
$Ref2va = 'models\MiniMaxH3\Weights\minimax_h3_ref2va_pruned_int8_convrot.safetensors'
$KeyImage = 'uploads\key-ref-development\key-001.jpg'
$RefImage = 'uploads\key-ref-development\赤い日傘とゴシックロリータ三面図.png'
$PromptRef = "$Rel\prompt_ref.txt"
$PromptNoref = "$Rel\prompt_noref.txt"
$GpuFreeMiB = 2000

function Get-GpuUsedMiB {
    # The GPU's used memory in MiB, or $null when nvidia-smi gives no reading.
    try {
        $gpu = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
        if ($LASTEXITCODE -ne 0) { return $null }
        return [int]("$(@($gpu)[0])".Trim())
    } catch {
        return $null
    }
}

function Wait-GpuFree {
    # Every 2 s until the GPU holds $GpuFreeMiB MiB or less; past -TimeoutSec, throws 'STOP: ...'.
    param([int]$TimeoutSec = 180)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ($true) {
        $used = Get-GpuUsedMiB
        if (($null -ne $used) -and ($used -le $GpuFreeMiB)) { return }
        if ((Get-Date) -gt $deadline) {
            throw "STOP: the GPU still holds $used MiB after $TimeoutSec s (more than $GpuFreeMiB MiB)"
        }
        Start-Sleep -Seconds 2
    }
}

function Get-BackendProcess {
    # Python processes of the backend (main.py) or of an H3 worker (engine_h3.worker), never this runner's own.
    $pattern = '(engine_h3[\\/.]worker|(^|[\s\\/"])main\.py)'
    $procs = Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" -ErrorAction SilentlyContinue
    return @($procs | Where-Object { ($_.ProcessId -ne $PID) -and ($_.CommandLine -match $pattern) })
}

function Invoke-Preflight {
    # Plan sec. 6: AviUtl2 not running, no backend / H3 worker running, the GPU holding 2,000 MiB or less,
    # the commit limit, the files present. Throws 'STOP: ...'. -ShowOnly (the dry run): the GPU / AviUtl2 /
    # backend / memory lines are shown, never a stop.
    param([switch]$ShowOnly)
    $need = @(
        $Fl2va, $Ref2va,
        'models\MiniMaxH3\TextEncoder\qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
        'models\MiniMaxH3\TextEncoder\qwen3vl_tokenizer',
        'models\MiniMaxH3\VAE\minimax_h3_video_vae_fp16.safetensors',
        'models\MiniMaxH3\VAE\minimax_h3_audio_vae_fp32.safetensors',
        $KeyImage, $RefImage, $PromptRef, $PromptNoref,
        '.venv-engine-h3\Scripts\python.exe', 'weight_identity.json', 'tools\ffmpeg\bin\ffmpeg.exe',
        "$Rel\kr_single.py"
    )
    foreach ($n in $need) {
        if (-not (Test-Path -LiteralPath (Join-Path $Repo $n))) { throw "STOP: missing: $n" }
    }
    Write-Host 'weights, key frame, reference, prompts, venv, weight_identity.json, ffmpeg, script: present'
    Write-Host '--- nvidia-smi (compute processes) ---'
    try { & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv } catch { Write-Host "nvidia-smi: $_" }
    $aviutl = Get-Process -Name 'aviutl2' -ErrorAction SilentlyContinue
    if ($aviutl) {
        if ($ShowOnly) { Write-Host '(dry run) AviUtl2 is running' } else { throw 'STOP: AviUtl2 is running. Close it first.' }
    }
    $backend = Get-BackendProcess
    if ($backend.Count -gt 0) {
        foreach ($b in $backend) { Write-Host ("backend / H3 worker: pid {0}: {1}" -f $b.ProcessId, $b.CommandLine) }
        if ($ShowOnly) { Write-Host '(dry run) the backend or an H3 worker is running' } else { throw 'STOP: the backend or an H3 worker is running. Stop it first.' }
    } else {
        Write-Host 'backend / H3 worker: none'
    }
    $used = Get-GpuUsedMiB
    if ($null -eq $used) {
        if ($ShowOnly) { Write-Host '(dry run) nvidia-smi gave no reading' } else { throw 'STOP: nvidia-smi gave no GPU memory reading' }
    } else {
        Write-Host "GPU memory used: $used MiB"
        if ($used -gt $GpuFreeMiB) {
            if ($ShowOnly) { Write-Host "(dry run) GPU already holds $used MiB" } else { throw "STOP: GPU already holds $used MiB: another process is on the GPU" }
        }
    }
    try {
        $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
        $pf = (Get-CimInstance Win32_PageFileUsage | Measure-Object -Property AllocatedBaseSize -Sum).Sum / 1024
        $limit = [math]::Round($ram + $pf, 1)
        Write-Host "RAM $ram GB + page file allocated $([math]::Round($pf,1)) GB = commit limit about $limit GB"
        if (($limit -lt 45) -and (-not $ShowOnly)) { throw 'STOP: the commit limit (RAM + page file) is under 45 GB' }
    } catch [System.Management.Automation.RuntimeException] {
        if ("$_" -like 'STOP:*') { throw }
        Write-Host "memory check skipped: $_"
    }
}

function Test-ReportOk {
    # True when <archive>\<Name>.json exists and says "ok": true (the -Resume rule).
    param([Parameter(Mandatory)] [string]$Name)
    $report = Join-Path $Arch "$Name.json"
    if (-not (Test-Path -LiteralPath $report)) { return $false }
    try {
        $j = Get-Content -LiteralPath $report -Raw -Encoding UTF8 | ConvertFrom-Json
        return ($j.ok -eq $true)
    } catch {
        return $false
    }
}

function Move-Aside {
    # An existing file -> <base>.<yyyyMMdd-HHmmss><ext> beside it (a rerun never erases an earlier record).
    param([Parameter(Mandatory)] [string]$Path, [Parameter(Mandatory)] [string]$Stamp)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $dir = Split-Path -Parent $Path
    $leaf = Split-Path -Leaf $Path
    foreach ($ext in @('.console.log', '.json')) {
        if ($leaf.EndsWith($ext)) {
            $base = $leaf.Substring(0, $leaf.Length - $ext.Length)
            Move-Item -LiteralPath $Path -Destination (Join-Path $dir "$base.$Stamp$ext")
            return
        }
    }
    Move-Item -LiteralPath $Path -Destination (Join-Path $dir "$leaf.$Stamp")
}

function Invoke-Cond {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$Script,
        [Parameter(Mandatory)] [string[]]$CondArgs,
        [int]$TimeoutSec = 1500,
        [string]$Suffix = '',
        # Other reports the script writes beside its own (the gate's r01 / r02), moved aside the same way.
        [string[]]$ExtraReports = @(),
        # The dry run: no model, so the GPU is not waited for.
        [switch]$NoGpuWait
    )
    $report = Join-Path $Arch "$Name$Suffix.json"
    $log = Join-Path $Arch "$Name$Suffix.console.log"
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    Move-Aside -Path $report -Stamp $stamp
    Move-Aside -Path $log -Stamp $stamp
    foreach ($x in $ExtraReports) { Move-Aside -Path (Join-Path $Arch "$x$Suffix.json") -Stamp $stamp }
    if (-not $NoGpuWait) { Wait-GpuFree }
    # Every argument quoted (paths may hold spaces; the reference's name is Japanese).
    $quoted = ($CondArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $inner = "`"$Py`" `"$Script`" " + $quoted + " --report `"$report`" > `"$log`" 2>&1"
    Write-Host ("[{0}] {1}: {2}" -f (Get-Date -Format 'HH:mm:ss'), $Name, ($CondArgs -join ' '))
    $started = Get-Date
    $p = Start-Process -FilePath $env:ComSpec -ArgumentList "/d /c `"$inner`"" -NoNewWindow -PassThru
    $null = $p.Handle   # keep the handle so ExitCode is readable after the exit (PowerShell 5.1)
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        & taskkill.exe /T /F /PID $p.Id | Out-Null
        # taskkill does not wait: the log stays locked until the tree is gone.
        $null = $p.WaitForExit(120000)
        if (-not $NoGpuWait) { Wait-GpuFree }
        try {
            Add-Content -LiteralPath $log -Value "TIMEOUT after $TimeoutSec s (process tree killed by run_common.ps1)" -Encoding utf8
        } catch {
            Write-Host "    (the TIMEOUT line could not be added to $log : $_)"
        }
        try {
            if (-not (Test-Path -LiteralPath $report)) {
                Set-Content -LiteralPath $report -Value ("{`"timeout`": true, `"timeout_sec`": $TimeoutSec, `"name`": `"$Name`", `"ok`": false}") -Encoding utf8
            }
        } catch {
            Write-Host "    (the timeout report could not be written to $report : $_)"
        }
        Write-Host "    TIMEOUT ($TimeoutSec s)"
        return -1
    }
    $p.WaitForExit()
    $sec = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    if (-not (Test-Path -LiteralPath $report)) {
        Set-Content -LiteralPath $report -Value ("{`"no_report`": true, `"exit_code`": $($p.ExitCode), `"name`": `"$Name`", `"ok`": false}") -Encoding utf8
    }
    Write-Host "    exit $($p.ExitCode) in $sec s -> $report"
    return $p.ExitCode
}
