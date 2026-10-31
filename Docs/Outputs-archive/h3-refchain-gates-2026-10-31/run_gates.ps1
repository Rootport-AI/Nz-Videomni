# Sec. 1-109 MiniMax H3 reference chain: the GPU gates G-1 / G-1b / G-2 through the product's HTTP API (plan sec. 6 M6 / 7).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-refchain-gates-2026-10-31\run_gates.ps1 [-SkipPreflight] [-Port 18629] [-Gates G-1,G-1b,G-2]
#
# DO NOT RUN WITHOUT THE OWNER'S GPU APPROVAL. A chain job has no cooperative stop (it cannot be cancelled half way).
# Order: preflight -> selfcheck (CPU) -> start the backend (own port) -> G-1 (FL2VA) -> G-1b (Ref2VA, 640x384) ->
#   G-2 (Ref2VA, 1344x768) -> cleanup (unload) -> stop the backend, put state.json back.
# G-3 (LTX 2.3 answers 422 for clips[i].references) is NOT in this script: see README.md (covered by the app's unit tests).
# One gate = one process (run_common.ps1), each with a timeout: G-1 3,000 s, G-1b 2,000 s, G-2 4,200 s.
# Stops (every later gate needs the earlier ones to have run): the selfcheck not 0; a gate not 0
# (1 = the run failed, 2 = a criterion failed; -ContinueOnMismatch goes on after a 2, never after a 1 or a timeout).
# Reports: <archive>\<name>.json (g_1 / g_1b / g_2), console: <archive>\<name>.console.log, backend: <archive>\backend.console.log.
# Gate mp4 / metadata.json: outputs\<job_id>\ (not tracked; the G-2 mp4 is for the owner's eyes).

param(
    [switch]$SkipPreflight,
    [int]$Port = 18629,
    [switch]$ContinueOnMismatch,
    [switch]$UseRunningBackend,    # do not start / stop a backend; use the one on -Port (it must be real; state.json is not touched)
    [string[]]$Gates = @('G-1', 'G-1b', 'G-2')
)
. (Join-Path $PSScriptRoot 'run_common.ps1') -SkipPreflight:$SkipPreflight

$Gates = @($Gates | ForEach-Object { $_ -split ',' } | Where-Object { $_ })   # -File passes "G-1,G-1b" as one string
$Base = "http://127.0.0.1:$Port/api/v1"
$Timeouts = @{ 'G-1' = 3000; 'G-1b' = 2000; 'G-2' = 4200 }
$stateFile = Join-Path $Repo 'state.json'
$stateBackup = Join-Path $Tmp 'state.json.before'
$backend = $null
$runnerExit = 0

function Stop-Run([string]$why) {
    Write-Host "STOP: $why"
    $script:runnerExit = 1
    throw "STOP: $why"
}

function Test-BackendUp {
    try { $null = Invoke-RestMethod -Uri "$Base/status" -TimeoutSec 5; return $true } catch { return $false }
}

Start-Transcript -Path (Join-Path $Arch 'run_gates.runner.log') -Force | Out-Null
try {
    # ---- preflight ----------------------------------------------------------------------------------------
    foreach ($gateName in $Gates) {
        if (@('G-1', 'G-1b', 'G-2') -notcontains $gateName) { Stop-Run "unknown gate '$gateName' (G-1, G-1b, G-2)" }
    }
    Invoke-Preflight
    $need = @(
        'models\MiniMaxH3\Weights\minimax_h3_fl2va_pruned_int8_convrot.safetensors',
        'models\MiniMaxH3\Weights\minimax_h3_ref2va_pruned_int8_convrot.safetensors',
        'models\MiniMaxH3\TextEncoder\qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
        'models\MiniMaxH3\TextEncoder\qwen3vl_tokenizer',
        'models\MiniMaxH3\VAE\minimax_h3_video_vae_fp16.safetensors',
        'models\MiniMaxH3\VAE\minimax_h3_audio_vae_fp32.safetensors',
        '.venv\Scripts\python.exe', '.venv-engine-h3\Scripts\python.exe', 'main.py', 'config.yaml',
        "$RelChain\prompts_c.json", "$RelPrior\g_c0.json",
        "$RelRef\prompts_c2.json", "$RelRef\prompts_c3.json", "$RelRef\g_ref2va_gate.json",
        "$RelRef2\17fde0c1-image.png", "$RelRef2\d80320dd-image.png"
    )
    foreach ($n in $need) {
        if (-not (Test-Path (Join-Path $Repo $n))) { Stop-Run "missing: $n (the real backend needs the H3 weights, FL2VA and Ref2VA, under models\; the pictures and records under Docs\Outputs-archive\)" }
    }
    Write-Host 'weights (FL2VA, Ref2VA), venvs, scripts, pictures, records: present'
    $cfg = Select-String -Path (Join-Path $Repo 'config.yaml') -Pattern '^\s*backend:\s*"?(\w+)' | Select-Object -First 1
    $cfgName = '(not found)'
    if ($cfg) { $cfgName = $cfg.Matches[0].Groups[1].Value }
    Write-Host "config.yaml model.backend: $cfgName (must be auto or real; g_refchain.py also refuses a backend that looks mock)"
    if ($cfgName -eq 'mock') { Stop-Run 'config.yaml model.backend is mock' }
    $gpu = & nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    $used = [int](($gpu -split ',')[0].Trim())
    if ($used -gt 2000 -and -not $SkipPreflight) { Stop-Run "GPU already holds $used MiB: another process is on the GPU (or pass -SkipPreflight)" }
    # The chain's main memory (commit) is expected around 35 GB: the page file matters.
    try {
        $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
        $pf = (Get-CimInstance Win32_PageFileUsage | Measure-Object -Property AllocatedBaseSize -Sum).Sum / 1024
        $limit = [math]::Round($ram + $pf, 1)
        Write-Host "RAM $ram GB + page file allocated $([math]::Round($pf,1)) GB = commit limit about $limit GB (the gates expect about 35 GB)"
        if ($limit -lt 45 -and -not $SkipPreflight) { Stop-Run 'the commit limit (RAM + page file) is under 45 GB: a gate may fail (or pass -SkipPreflight)' }
    } catch [System.Management.Automation.RuntimeException] {
        if ("$_" -like 'STOP:*') { throw }
        Write-Host "memory check skipped: $_"
    }
    if ($UseRunningBackend) {
        if (-not (Test-BackendUp)) { Stop-Run "no backend answers on $Base" }
    } elseif (Test-BackendUp) {
        Stop-Run "something already answers on $Base (a backend is running). Stop it, pick another -Port, or pass -UseRunningBackend"
    }

    # ---- selfcheck (CPU only) -----------------------------------------------------------------------------
    $rc = Invoke-Cond 'g_selfcheck' $G @('--selfcheck') -TimeoutSec 300
    if ($rc -ne 0) { Stop-Run "g_refchain selfcheck exit $rc" }

    # ---- the backend (own port) ---------------------------------------------------------------------------
    if (-not $UseRunningBackend) {
        Copy-Item -LiteralPath $stateFile -Destination $stateBackup -Force
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $Repo '.python'
        $pre = ''
        foreach ($d in @((Join-Path $Repo 'tools\uv'), (Join-Path $Repo 'tools\ffmpeg\bin'))) { if (Test-Path $d) { $pre += "$d;" } }
        if ($pre) { $env:PATH = $pre + $env:PATH }
        $blog = Join-Path $Arch 'backend.console.log'
        $inner = "`"$AppPy`" `"$(Join-Path $Repo 'main.py')`" --port $Port > `"$blog`" 2>&1"
        Write-Host "starting the backend on port $Port"
        $backend = Start-Process -FilePath $env:ComSpec -ArgumentList "/d /c `"$inner`"" -NoNewWindow -PassThru
        $null = $backend.Handle
        $up = $false
        for ($i = 0; $i -lt 90; $i++) {
            if ($backend.HasExited) { Stop-Run "the backend exited at startup (exit $($backend.ExitCode)); see $blog" }
            if (Test-BackendUp) { $up = $true; break }
            Start-Sleep -Seconds 2
        }
        if (-not $up) { Stop-Run "the backend did not answer on $Base within 180 s; see $blog" }
    }
    $st = Invoke-RestMethod -Uri "$Base/status" -TimeoutSec 10
    Write-Host "backend: state=$($st.state) base_model=$($st.base_model) pipeline_loaded=$($st.pipeline_loaded)"
    if ($st.state -eq 'running') { Stop-Run 'the backend is busy with a job' }
    $bpid = 0
    if ($backend) { $bpid = $backend.Id }

    # ---- the gates ----------------------------------------------------------------------------------------
    foreach ($gate in $Gates) {
        $a = @($gate, '--base-url', $Base)
        if ($bpid) { $a += @('--backend-pid', "$bpid") }
        $name = $gate.Replace('-', '_').ToLower()   # g_1 / g_1b / g_2
        $rc = Invoke-Cond $name $G $a -TimeoutSec $Timeouts[$gate]
        if ($rc -eq 2 -and $ContinueOnMismatch) { Write-Host "${gate}: a criterion failed (exit 2), going on (-ContinueOnMismatch)"; continue }
        if ($rc -ne 0) { Stop-Run "$gate exit $rc (2 = a criterion failed, 1 = the run failed, -1 = timeout)" }
    }
    Write-Host 'all gates finished'
}
catch {
    Write-Host "runner: $_"
    if ($runnerExit -eq 0) { $runnerExit = 1 }
}
finally {
    # ---- the end (the G-R5 pattern): give the GPU back, stop what this runner started, put state.json back ----
    if ($backend -and -not $backend.HasExited -and (Test-BackendUp)) {
        $null = Invoke-Cond 'g_cleanup' $G @('cleanup', '--base-url', $Base) -TimeoutSec 400
    }
    if ($backend) {
        if (-not $backend.HasExited) { & taskkill.exe /T /F /PID $backend.Id | Out-Null }
        Write-Host 'backend stopped'
        if (Test-Path $stateBackup) { Copy-Item -LiteralPath $stateBackup -Destination $stateFile -Force; Write-Host 'state.json put back' }
    }
    try { & nvidia-smi --query-gpu=memory.used,memory.total --format=csv } catch { }
    Stop-Transcript | Out-Null
}
exit $runnerExit
