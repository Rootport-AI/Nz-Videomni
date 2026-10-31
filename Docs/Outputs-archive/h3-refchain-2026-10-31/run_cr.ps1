# sec. 1-109 Ref2VA chain (fixed rows + per-clip references) measurement: the conditions in the plan's order
# (plan pure-knitting-shell.md sec. 3.2 and sec. 7).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-refchain-2026-10-31\run_cr.ps1
#
# DO NOT RUN WITHOUT THE OWNER'S GPU APPROVAL (about 1 h 50 min). Order:
#   preflight -> selfcheck -> g_fl2va_gate -> g_ref2va_gate -> c0_fl2va -> c1_ref2va_noref -> c2_ref2va_same ->
#   c3_ref2va_perclip (clip 0 reused from c2) -> analysis.
# One condition = one process (run_common.ps1), each with a timeout. Reports: <archive>\<name>.json, console:
# <archive>\<name>.console.log, this runner: <archive>\run_cr.runner.log. mp4 / .pt: outputs\h3-refchain-2026-10-31\<name>\
# (cr_chain.py's default --out-dir; not tracked).
# Stops (plan sec. 7): the preflight fails; selfcheck, g_fl2va_gate or g_ref2va_gate not 0 (2 = a gate's
# comparison failed, 1 = failed, -1 = timeout). c0 / c1 not 0: recorded, the run goes on. c2 not 0: c3 is
# skipped (it reuses c2's clip 0); the analysis still runs.
# Common to every GPU condition: 20 steps, shift 12/3, VRAM 16GB mode on, prefetch on, blocks_on_gpu 8,
# 124 frames x 3 clips, overlap 22 (k 7), strength 1.0 with the clean inject, one decode, seed 42 (+i per clip).

. (Join-Path $PSScriptRoot 'run_common.ps1')

$pc0 = "$Rel\prompts_c0.json"
$pc2 = "$Rel\prompts_c2.json"
$pc3 = "$Rel\prompts_c3.json"
$rc2 = "$Rel\refs_c2.json"
$rc3 = "$Rel\refs_c3.json"
$c2json = "$Rel\c2_ref2va_same.json"
$runnerExit = 0

function Stop-Run([string]$why) {
    Write-Host "STOP: $why"
    $script:runnerExit = 1
    throw "STOP: $why"
}

Start-Transcript -Path (Join-Path $Arch 'run_cr.runner.log') -Force | Out-Null
try {
    Invoke-Preflight

    $rc = Invoke-Cond 'selfcheck' $C @('--selfcheck') -TimeoutSec 300
    if ($rc -ne 0) { Stop-Run "selfcheck exit $($rc)" }

    $gate = @('--width', '640', '--height', '384', '--attention', 'sdpa', '--clips', '3', '--recheck-clip0')
    # G-F: FL2VA, no reference, the fingerprint prompt on all clips (the script's FP_PROMPT): clip 0 vs 0da6111c...
    $rc = Invoke-Cond 'g_fl2va_gate' $C (@('--name', 'g_fl2va_gate', '--fingerprint', '--transformer', $Fl2va) + $gate) -TimeoutSec 1200
    if ($rc -ne 0) { Stop-Run "g_fl2va_gate exit $($rc) (2 = fingerprint / recheck / refs_and_fix mismatch, 1 = failed, -1 = timeout)" }
    # G-R: Ref2VA, the gentleman on all clips: clip 0's _decode vs the product's generate(references=...).
    $rc = Invoke-Cond 'g_ref2va_gate' $C (@('--name', 'g_ref2va_gate', '--product-recheck', '--transformer', $Ref2va,
        '--prompts', $pc2, '--refs', $rc2) + $gate) -TimeoutSec 1500
    if ($rc -ne 0) { Stop-Run "g_ref2va_gate exit $($rc) (2 = product_recheck / recheck / refs_and_fix mismatch, 1 = failed, -1 = timeout)" }

    $main = @('--width', '1344', '--height', '768', '--attention', 'sage', '--clips', '3', '--save-latents')
    # C0: the control (FL2VA, no reference).
    $rc = Invoke-Cond 'c0_fl2va' $C (@('--name', 'c0_fl2va', '--transformer', $Fl2va, '--prompts', $pc0) + $main) -TimeoutSec 2400
    if ($rc -ne 0) { Write-Host "c0_fl2va exit $($rc): recorded, going on" }
    # C1: Ref2VA without a reference (for reference only).
    $rc = Invoke-Cond 'c1_ref2va_noref' $C (@('--name', 'c1_ref2va_noref', '--transformer', $Ref2va, '--prompts', $pc0) + $main) -TimeoutSec 2400
    if ($rc -ne 0) { Write-Host "c1_ref2va_noref exit $($rc): recorded, going on" }
    # C2: Ref2VA, the gentleman on all clips (the smallest form).
    $rc = Invoke-Cond 'c2_ref2va_same' $C (@('--name', 'c2_ref2va_same', '--transformer', $Ref2va, '--prompts', $pc2,
        '--refs', $rc2) + $main) -TimeoutSec 2700
    if ($rc -ne 0) {
        Write-Host "c2_ref2va_same exit $($rc): c3_ref2va_perclip skipped (it reuses c2's clip 0)"
    } else {
        # C3: Ref2VA, the gentleman / gentleman + girl / gentleman + girl (the product's form); clip 0 from C2.
        $rc = Invoke-Cond 'c3_ref2va_perclip' $C (@('--name', 'c3_ref2va_perclip', '--transformer', $Ref2va, '--prompts', $pc3,
            '--refs', $rc3, '--reuse-clip0', $c2json) + $main) -TimeoutSec 2400
        if ($rc -ne 0) { Write-Host "c3_ref2va_perclip exit $($rc): recorded" }
    }

    # Analysis (CPU): CSVs next to the reports.
    $rc = Invoke-Cond 'analysis' $An @('--out-dir', $Arch) -TimeoutSec 900
    if ($rc -ne 0) { Write-Host "analysis exit $($rc)" }
} catch {
    if ($runnerExit -eq 0) { $runnerExit = 1 }
    Write-Host "runner stopped: $_"
} finally {
    Stop-Transcript | Out-Null
}
exit $runnerExit
