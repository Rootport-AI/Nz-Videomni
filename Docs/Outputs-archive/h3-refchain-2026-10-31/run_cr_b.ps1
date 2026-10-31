# sec. 1-109 Ref2VA chain, supplement (plan pure-knitting-shell.md sec. 12): 2 isolation conditions (plan B).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-refchain-2026-10-31\run_cr_b.ps1
#
# DO NOT RUN WITHOUT THE OWNER'S GPU APPROVAL (about 45 min). Order: preflight -> c4_ref2va_same_stayprompt ->
# c5_ref2va_perclip_walkprompt -> analysis. Needs c2_ref2va_same.json of run_cr.ps1 (clip 0 is reused from it);
# without it the run stops at the preflight stage. No selfcheck and no gates: cr_chain.py is unchanged (see README).
# c4 / c5 not 0: recorded, the run goes on. Runner log: <archive>\run_cr_b.runner.log.
# Common conditions are the same as the 4 main conditions of run_cr.ps1 (1344x768, sage, 3 clips, --save-latents).

. (Join-Path $PSScriptRoot 'run_common.ps1')

$pc2 = "$Rel\prompts_c2.json"
$pc4 = "$Rel\prompts_c4.json"
$pc5 = "$Rel\prompts_c5.json"
$rc2 = "$Rel\refs_c2.json"
$rc3 = "$Rel\refs_c3.json"
$c2json = "$Rel\c2_ref2va_same.json"
$runnerExit = 0

function Stop-Run([string]$why) {
    Write-Host "STOP: $why"
    $script:runnerExit = 1
    throw "STOP: $why"
}

Start-Transcript -Path (Join-Path $Arch 'run_cr_b.runner.log') -Force | Out-Null
try {
    Invoke-Preflight
    foreach ($f in @($c2json, $pc4, $pc5, $rc2, $rc3)) {
        if (-not (Test-Path (Join-Path $Repo $f))) { Stop-Run "missing file: $f (run run_cr.ps1 first for c2_ref2va_same.json)" }
    }

    $main = @('--width', '1344', '--height', '768', '--attention', 'sage', '--clips', '3', '--save-latents')
    # C4: Ref2VA, the gentleman only (refs_c2) x3, clips 1-2 carry C3's text (no <Picture 2> tag); clip 0 from C2.
    $rc = Invoke-Cond 'c4_ref2va_same_stayprompt' $C (@('--name', 'c4_ref2va_same_stayprompt', '--transformer', $Ref2va, '--prompts', $pc4,
        '--refs', $rc2, '--reuse-clip0', $c2json) + $main) -TimeoutSec 2400
    if ($rc -ne 0) { Write-Host "c4_ref2va_same_stayprompt exit $($rc): recorded, going on" }
    # C5: Ref2VA, gentleman / gentleman + girl / gentleman + girl (refs_c3), clips 1-2 carry C2's text; clip 0 from C2.
    $rc = Invoke-Cond 'c5_ref2va_perclip_walkprompt' $C (@('--name', 'c5_ref2va_perclip_walkprompt', '--transformer', $Ref2va, '--prompts', $pc5,
        '--refs', $rc3, '--reuse-clip0', $c2json) + $main) -TimeoutSec 2400
    if ($rc -ne 0) { Write-Host "c5_ref2va_perclip_walkprompt exit $($rc): recorded, going on" }

    # Analysis (CPU): reads every c*_*.json, CSVs next to the reports are overwritten.
    $rc = Invoke-Cond 'analysis' $An @('--out-dir', $Arch) -TimeoutSec 900
    if ($rc -ne 0) { Write-Host "analysis exit $($rc)" }
} catch {
    if ($runnerExit -eq 0) { $runnerExit = 1 }
    Write-Host "runner stopped: $_"
} finally {
    Stop-Transcript | Out-Null
}
exit $runnerExit
