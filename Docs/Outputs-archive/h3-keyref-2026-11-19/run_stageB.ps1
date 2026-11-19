# sec. 1-115 stage B: one key frame in the middle (key-001 at frame 17 only = latent 5), 4 runs (the gate's 2 + 2).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-keyref-2026-11-19\run_stageB.ps1
#   ... -Resume    skip every condition whose <archive>\<name>.json already says "ok": true
#   ... -DryRun    every condition with kr_single.py --dry-run (no model, no GPU); reports <name>.dryrun.json
#
# DO NOT RUN WITHOUT THE OWNER'S GPU APPROVAL. Order: preflight -> gate_B (b1_010_f17_harness then
# b1_010_f17_product in one process, kr_single.py --mode gate: the video / audio stream MD5s must match;
# 3,000 s) -> b2_001_f17 -> b3_011_f17. One condition = one process (run_common.ps1), 1,500 s each.
# The gate not 0 stops the stage (nothing after it runs); any other condition not 0 is recorded and the run goes on.
# Reports: <archive>\<name>.json (the gate: gate_B.json and the two runs' own), console: <name>.console.log,
# this runner: run_stageB.runner.log (appended). mp4 / .pt: outputs\h3-keyref-2026-11-19\ (not tracked).
# Exit code: 0 = every condition ok, 2 = the gate's streams differ, 1 = otherwise.
# Common: as stage A (1280x768, 124 frames, prompt_ref.txt, the picture reference as <Picture 1>, 20 steps,
# shift 12/3, seed 42, sage, deterministic on, blocks_on_gpu 8, prefetch on, VRAM 16GB mode on, keep_resident off).

param(
    [switch]$Resume,
    [switch]$DryRun
)

. (Join-Path $PSScriptRoot 'run_common.ps1')

$runnerExit = 0
$results = New-Object System.Collections.ArrayList
$suffix = ''
if ($DryRun) { $suffix = '.dryrun' }

$key = @('--keyframe', "17=$KeyImage")
$withRef = @('--prompt-file', $PromptRef) + $key + @('--ref-image', $RefImage)

$conds = @(
    @{ Name = 'gate_B'; Timeout = 3000; Extra = @('b1_010_f17_harness', 'b1_010_f17_product');
       Args = @('--mode', 'gate', '--gate-names', 'b1_010_f17_harness,b1_010_f17_product', '--transformer', $Ref2va) + $withRef },
    @{ Name = 'b2_001_f17'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $withRef + @('--T', 'off', '--C', 'off', '--L', '0') },
    @{ Name = 'b3_011_f17'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $withRef + @('--T', 'off', '--C', 'on', '--L', '0') }
)

$runnerLog = Join-Path $Arch ("run_stageB$suffix.runner.log")
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
        if ($c.Extra) { $extra = $c.Extra }
        $rc = Invoke-Cond -Name $name -Script $KrSingle -CondArgs $condArgs -TimeoutSec $c.Timeout -Suffix $suffix `
            -ExtraReports $extra -NoGpuWait:$DryRun
        [void]$results.Add([pscustomobject]@{ Name = $name; Exit = $rc })
        if ($rc -ne 0) {
            if ($name -like 'gate_*') {
                if ($rc -eq 2) { $runnerExit = 2 } else { $runnerExit = 1 }
                Write-Host "STOP: $name exit $($rc) (2 = the video / audio streams differ, 1 = failed, -1 = timeout). Nothing after it runs."
                break
            }
            $runnerExit = 1
            Write-Host "$name exit $($rc): recorded, going on"
        }
    }
} catch {
    $runnerExit = 1
    Write-Host "runner stopped: $_"
} finally {
    Write-Host '--- results ---'
    foreach ($r in $results) { Write-Host ("{0,-20} {1}" -f $r.Name, $r.Exit) }
    Write-Host "runner exit $runnerExit"
    Stop-Transcript | Out-Null
}
exit $runnerExit
