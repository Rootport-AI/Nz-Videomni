# sec. 1-115 Reference + key frame, stage A (the first frame): the plan's 11 runs in the plan's order
# (plan glistening-splashing-hoare.md sec. 4 and sec. 7).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-keyref-2026-11-19\run_stageA.ps1
#   ... -Resume    skip every condition whose <archive>\<name>.json already says "ok": true
#   ... -DryRun    every condition with kr_single.py --dry-run (no model, no GPU); reports <name>.dryrun.json
#
# DO NOT RUN WITHOUT THE OWNER'S GPU APPROVAL (about 1 h 45 min). Order:
#   preflight -> gate (r01_010_harness + r02_010_product, one process) -> r03_ctrlF -> r04_ctrlR -> r05_110 ->
#   r06_001 -> r07_011 -> r08_000 -> r09_100 -> r10_101 -> r11_111.
# One condition = one process (run_common.ps1), 1,500 s each (the gate 3,000 s: two runs).
# Reports: <archive>\<name>.json (the gate also writes r01_... / r02_....json), console: <name>.console.log,
# this runner: run_stageA.runner.log. mp4 / .pt: outputs\h3-keyref-2026-11-19\ (not tracked).
# Stops: the preflight fails, or the gate is not 0 (2 = the streams differ, 1 = failed, -1 = timeout); nothing
# after it runs. Any other condition not 0: recorded, the run goes on; the list of results comes at the end.
# Exit code: 0 = every condition ok, 1 = a stop or any condition not ok.
# Common (plan sec. 4): 1280x768, 124 frames, 20 steps, shift 12/3, seed 42, sage, deterministic on,
# blocks_on_gpu 8, prefetch on, VRAM 16GB mode on, keep_resident off (the values are kr_single.py's constants).

param(
    [switch]$Resume,
    [switch]$DryRun
)

. (Join-Path $PSScriptRoot 'run_common.ps1')

$runnerExit = 0
$results = New-Object System.Collections.ArrayList
$suffix = ''
if ($DryRun) { $suffix = '.dryrun' }

$key = @('--keyframe', "0=$KeyImage")
$ref = @('--ref-image', $RefImage)
$withRef = @('--prompt-file', $PromptRef) + $key + $ref

# name, timeout (s), arguments; in the plan's order (sec. 4). The gate is first and must pass.
$conds = @(
    @{ Name = 'gate'; Timeout = 3000; Args = @('--mode', 'gate', '--transformer', $Ref2va) + $withRef },
    @{ Name = 'r03_ctrlF'; Timeout = 1500; Args = @('--mode', 'product', '--transformer', $Fl2va, '--prompt-file', $PromptNoref) + $key },
    @{ Name = 'r04_ctrlR'; Timeout = 1500; Args = @('--mode', 'product', '--transformer', $Ref2va, '--prompt-file', $PromptNoref) + $key },
    @{ Name = 'r05_110'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $withRef + @('--T', 'on', '--C', 'on') },
    @{ Name = 'r06_001'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $withRef + @('--T', 'off', '--C', 'off', '--L', '0') },
    @{ Name = 'r07_011'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $withRef + @('--T', 'off', '--C', 'on', '--L', '0') },
    @{ Name = 'r08_000'; Timeout = 1500; Args = @('--mode', 'product', '--transformer', $Ref2va, '--prompt-file', $PromptRef) + $ref },
    @{ Name = 'r09_100'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $withRef + @('--T', 'on', '--C', 'off') },
    @{ Name = 'r10_101'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $withRef + @('--T', 'on', '--C', 'off', '--L', '0') },
    @{ Name = 'r11_111'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $withRef + @('--T', 'on', '--C', 'on', '--L', '0') }
)

$runnerLog = Join-Path $Arch ("run_stageA$suffix.runner.log")
# -Append: a rerun (-Resume) adds to the earlier runner log instead of erasing it.
Start-Transcript -Path $runnerLog -Append | Out-Null
try {
    if ($DryRun) { Invoke-Preflight -ShowOnly } else { Invoke-Preflight }
    foreach ($c in $conds) {
        $name = $c.Name
        if ($Resume -and (-not $DryRun) -and (Test-ReportOk $name)) {
            Write-Host "[skip] $name already ok"
            [void]$results.Add([pscustomobject]@{ Name = $name; Exit = 'skipped (ok before)' })
            continue
        }
        $condArgs = @('--name', $name) + $c.Args
        if ($DryRun) { $condArgs = $condArgs + @('--dry-run') }
        $extra = @()
        if ($name -eq 'gate') { $extra = @('r01_010_harness', 'r02_010_product') }
        $rc = Invoke-Cond -Name $name -Script $KrSingle -CondArgs $condArgs -TimeoutSec $c.Timeout -Suffix $suffix `
            -ExtraReports $extra -NoGpuWait:$DryRun
        [void]$results.Add([pscustomobject]@{ Name = $name; Exit = $rc })
        if ($rc -ne 0) {
            $runnerExit = 1
            if ($name -eq 'gate') {
                Write-Host "STOP: gate exit $rc (2 = the video / audio streams differ, 1 = failed, -1 = timeout). Nothing after it runs."
                break
            }
            Write-Host "$name exit $($rc): recorded, going on"
        }
    }
} catch {
    $runnerExit = 1
    Write-Host "runner stopped: $_"
} finally {
    Write-Host '--- results ---'
    foreach ($r in $results) { Write-Host ("{0,-12} {1}" -f $r.Name, $r.Exit) }
    Write-Host "runner exit $runnerExit"
    Stop-Transcript | Out-Null
}
exit $runnerExit
