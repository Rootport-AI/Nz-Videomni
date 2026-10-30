# Sec. 1-106 MiniMax H3 Chained: the GPU gates G-C0 / G-C1 / G-C2 through the product's /generate/chain (plan sec. 7).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-chain-gates-2026-10-29\run_gates.ps1 [-SkipPreflight] [-Port 18629]
#
# DO NOT RUN WITHOUT THE OWNER'S GPU APPROVAL. G-C2 alone takes about 1.6-1.8 hours and cannot be cancelled
# half way (a chain job has no cooperative stop). Order:
#   preflight -> selfcheck (CPU) -> harness g_c0_harness (c_chain.py; H3 must NOT be loaded in any backend) ->
#   start the backend (own port) -> G-C0 -> G-C1 -> G-C2 -> cleanup (unload) -> stop the backend,
#   put state.json back.
# One gate = one process (run_common.ps1), each with a timeout: G-C0 1,500 s, G-C1 2,400 s, G-C2 11,000 s.
# Stops (every later gate needs the earlier ones to have run): the selfcheck or the harness not 0; a gate not 0
# (1 = the run failed, 2 = a criterion failed; -ContinueOnMismatch goes on after a 2, never after a 1 or a timeout).
# Reports: <archive>\<name>.json, console: <archive>\<name>.console.log, backend: <archive>\backend.console.log.
# Gate mp4 / metadata.json: outputs\<job_id>\ (not tracked); harness mp4 / .pt: outputs\h3-chain-2026-10-29\g_c0_harness\.

param(
    [switch]$SkipPreflight,
    [int]$Port = 18629,
    [switch]$ContinueOnMismatch,
    [switch]$SkipHarness,          # reuse an existing g_c0_harness.json (it must be ok; its mp4 / latents must still exist)
    [switch]$UseRunningBackend,    # do not start / stop a backend; use the one on -Port (it must be real; state.json is not touched)
    [string[]]$Gates = @('G-C0', 'G-C1', 'G-C2')
)
. (Join-Path $PSScriptRoot 'run_common.ps1') -SkipPreflight:$SkipPreflight

$Base = "http://127.0.0.1:$Port/api/v1"
$Timeouts = @{ 'G-C0' = 1500; 'G-C1' = 2400; 'G-C2' = 11000 }
$harnessName = 'g_c0_harness'
$harnessJson = "$Rel\$harnessName.json"
$prompts = "$RelChain\prompts_c.json"
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
    Invoke-Preflight
    $need = @(
        'models\MiniMaxH3\Weights\minimax_h3_fl2va_pruned_int8_convrot.safetensors',
        'models\MiniMaxH3\TextEncoder\qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
        'models\MiniMaxH3\TextEncoder\qwen3vl_tokenizer',
        'models\MiniMaxH3\VAE\minimax_h3_video_vae_fp16.safetensors',
        'models\MiniMaxH3\VAE\minimax_h3_audio_vae_fp32.safetensors',
        'models\MiniMaxH3\LoRA\minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors',
        '.venv\Scripts\python.exe', '.venv-engine-h3\Scripts\python.exe', 'main.py', 'config.yaml',
        "$RelChain\c_chain.py", "$RelChain\prompts_c.json"
    )
    foreach ($n in $need) {
        if (-not (Test-Path (Join-Path $Repo $n))) { Stop-Run "missing: $n (the real backend needs the H3 weights under models\)" }
    }
    Write-Host 'weights, LoRA, venvs, scripts: present'
    $cfg = Select-String -Path (Join-Path $Repo 'config.yaml') -Pattern '^\s*backend:\s*"?(\w+)' | Select-Object -First 1
    $cfgName = '(not found)'
    if ($cfg) { $cfgName = $cfg.Matches[0].Groups[1].Value }
    Write-Host "config.yaml model.backend: $cfgName (must be auto or real; g_chain.py also refuses a backend that looks mock)"
    if ($cfgName -eq 'mock') { Stop-Run 'config.yaml model.backend is mock' }
    $gpu = & nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    $used = [int](($gpu -split ',')[0].Trim())
    if ($used -gt 2000 -and -not $SkipPreflight) { Stop-Run "GPU already holds $used MiB: another process is on the GPU (or pass -SkipPreflight)" }
    # G-C2's commit is expected around 37 GB: the page file matters (plan sec. 7).
    try {
        $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
        $pf = (Get-CimInstance Win32_PageFileUsage | Measure-Object -Property AllocatedBaseSize -Sum).Sum / 1024
        $limit = [math]::Round($ram + $pf, 1)
        Write-Host "RAM $ram GB + page file allocated $([math]::Round($pf,1)) GB = commit limit about $limit GB (G-C2 expects about 37 GB)"
        if ($limit -lt 45 -and -not $SkipPreflight) { Stop-Run 'the commit limit (RAM + page file) is under 45 GB: G-C2 may fail (or pass -SkipPreflight)' }
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
    if ($rc -ne 0) { Stop-Run "g_chain selfcheck exit $rc" }

    # ---- harness: c_chain.py with G-C0's conditions. The H3 weights must not be loaded in any backend now. ----
    if ($SkipHarness) {
        $h = Get-Content (Join-Path $Repo $harnessJson) -Raw | ConvertFrom-Json
        if (-not $h.ok) { Stop-Run "-SkipHarness: $harnessJson is not ok" }
        Write-Host "harness reused: $harnessJson"
    } else {
        $rc = Invoke-Cond $harnessName $C @('--name', $harnessName, '--width', '640', '--height', '384', '--inject', 'clean',
            '--attention', 'sdpa', '--strength', '1.0', '--overlap', '22', '--decode', 'once', '--save-latents',
            '--prompts', $prompts, '--seed', '42') -TimeoutSec 1500
        if ($rc -ne 0) { Stop-Run "harness exit $rc (the gate cannot be judged without the script's sha256 values)" }
    }

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
        if ($gate -eq 'G-C0') { $a += @('--compare', $harnessJson) }
        $name = $gate.Replace('-', '_').ToLower()   # g_c0 ...
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
