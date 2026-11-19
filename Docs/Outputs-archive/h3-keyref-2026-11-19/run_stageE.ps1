# sec. 1-115 stage E (follow-up): the prompt that disagrees with the key frame, through the product as is,
# with five other seeds (a1_010_conflict is seed 42). 5 runs, no gate.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-keyref-2026-11-19\run_stageE.ps1
#   ... -Resume    skip every condition whose <archive>\<name>.json already says "ok": true
#   ... -DryRun    every condition with kr_single.py --dry-run (no model, no GPU); reports <name>.dryrun.json
#
# DO NOT RUN WITHOUT THE OWNER'S GPU APPROVAL. Order: preflight -> e_010_conflict_seed1 .. e_010_conflict_seed5.
# One condition = one process (run_common.ps1), 1,500 s each; a condition not 0 is recorded and the run goes on.
# Reports: <archive>\<name>.json, console: <name>.console.log, this runner: run_stageE.runner.log (appended).
# mp4: outputs\h3-keyref-2026-11-19\ (not tracked). Exit code: 0 = every condition ok, 1 = otherwise.
# Each run = run_stageA2.ps1's a1_010_conflict (--mode product, Ref2VA, prompt_a2_conflict.txt, 1344x768, the
# bridge at frame 0, the gentleman as <Picture 1>) + --seed <N>, N = 1..5.
# Common: 124 frames, 20 steps, shift 12/3, sage, deterministic on, blocks_on_gpu 8, prefetch on,
# VRAM 16GB mode on, keep_resident off (kr_single.py's constants).

param(
    [switch]$Resume,
    [switch]$DryRun
)

. (Join-Path $PSScriptRoot 'run_common.ps1')

$runnerExit = 0
$results = New-Object System.Collections.ArrayList
$suffix = ''
if ($DryRun) { $suffix = '.dryrun' }

$a2Key = 'Docs\Outputs-archive\h3-ref2va-2026-10-28\31d79f30-image.png'
$a2Ref = 'Docs\Outputs-archive\h3-ref2va-2026-10-28\17fde0c1-image.png'
$pConflict = "$Rel\prompt_a2_conflict.txt"
$base = @('--mode', 'product', '--transformer', $Ref2va, '--prompt-file', $pConflict, '--width', '1344', '--height', '768',
          '--keyframe', "0=$a2Key", '--ref-image', $a2Ref)

$conds = @()
foreach ($s in 1..5) {
    $conds += @{ Name = "e_010_conflict_seed$s"; Timeout = 1500; Args = $base + @('--seed', "$s") }
}

$runnerLog = Join-Path $Arch ("run_stageE$suffix.runner.log")
Start-Transcript -Path $runnerLog -Append | Out-Null
try {
    if ($DryRun) { Invoke-Preflight -ShowOnly } else { Invoke-Preflight }
    foreach ($n in @($a2Key, $a2Ref, $pConflict)) {
        if (-not (Test-Path -LiteralPath (Join-Path $Repo $n))) { throw "STOP: missing: $n" }
    }
    foreach ($c in $conds) {
        $name = $c.Name
        if ($Resume -and (-not $DryRun) -and (Test-ReportOk $name)) {
            Write-Host "[skip] $name already ok"
            [void]$results.Add([pscustomobject]@{ Name = $name; Exit = 'skipped (ok before)' })
            continue
        }
        $condArgs = @('--name', $name) + $c.Args
        if ($DryRun) { $condArgs = $condArgs + @('--dry-run') }
        $rc = Invoke-Cond -Name $name -Script $KrSingle -CondArgs $condArgs -TimeoutSec $c.Timeout -Suffix $suffix `
            -NoGpuWait:$DryRun
        [void]$results.Add([pscustomobject]@{ Name = $name; Exit = $rc })
        if ($rc -ne 0) {
            $runnerExit = 1
            Write-Host "$name exit $($rc): recorded, going on"
        }
    }
} catch {
    $runnerExit = 1
    Write-Host "runner stopped: $_"
} finally {
    Write-Host '--- results ---'
    foreach ($r in $results) { Write-Host ("{0,-24} {1}" -f $r.Name, $r.Exit) }
    Write-Host "runner exit $runnerExit"
    Stop-Transcript | Out-Null
}
exit $runnerExit
