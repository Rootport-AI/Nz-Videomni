# sec. 1-115 stage D (follow-up): with the prompt that disagrees with the key frame, does a middle freeze
# strength or a freeze at two positions help? 4 conditions, 5 runs (the gate's 2 + 3).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-keyref-2026-11-19\run_stageD.ps1
#   ... -Resume    skip every condition whose <archive>\<name>.json already says "ok": true
#   ... -DryRun    every condition with kr_single.py --dry-run (no model, no GPU); reports <name>.dryrun.json
#
# DO NOT RUN WITHOUT THE OWNER'S GPU APPROVAL. Order: preflight -> gate_D (d2_010_k0k17_conflict_harness then
# d2_010_k0k17_conflict_product in one process, kr_single.py --mode gate: the video / audio stream MD5s must
# match; 3,000 s) -> d3_011_k0k17_conflict -> d1_011_s05_conflict -> d4_011_s05_dragon. One condition = one
# process (run_common.ps1), 1,500 s each. The gate not 0 stops the stage (nothing after it runs); any other
# condition not 0 is recorded and the run goes on. Reports: <archive>\<name>.json (the gate: gate_D.json and
# the two runs' own), console: <name>.console.log, this runner: run_stageD.runner.log (appended).
# mp4 / .pt: outputs\h3-keyref-2026-11-19\ (not tracked).
# Exit code: 0 = every condition ok, 2 = the gate's streams differ, 1 = otherwise.
#   d2 (gate): the bridge at frames 0 and 17 (condition rows only), the gentleman <Picture 1>, the conflict prompt.
#   d3: the same + both frozen (--L 0 = strength 1.0: latents 0 and 5).
#   d1: the bridge at frame 0 only, condition rows + strength 0.5 (--L 0.5).
#   d4: stage A's material (key-001 at frame 0, the picture reference, prompt_ref.txt), condition rows + --L 0.5.
# The key frames go to the condition rows, never to the text encoder (no --T). Bridge runs 1344x768, d4 1280x768.
# Common: Ref2VA weights, 124 frames, 20 steps, shift 12/3, seed 42, sage, deterministic on, blocks_on_gpu 8,
# prefetch on, VRAM 16GB mode on, keep_resident off (kr_single.py's constants).

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
$bridge = @('--prompt-file', $pConflict, '--width', '1344', '--height', '768', '--ref-image', $a2Ref)
$k0 = @('--keyframe', "0=$a2Key")
$k0k17 = @('--keyframe', "0=$a2Key", '--keyframe', "17=$a2Key")
$dragon = @('--prompt-file', $PromptRef, '--width', '1280', '--height', '768', '--keyframe', "0=$KeyImage", '--ref-image', $RefImage)

$conds = @(
    @{ Name = 'gate_D'; Timeout = 3000; Extra = @('d2_010_k0k17_conflict_harness', 'd2_010_k0k17_conflict_product');
       Args = @('--mode', 'gate', '--gate-names', 'd2_010_k0k17_conflict_harness,d2_010_k0k17_conflict_product', '--transformer', $Ref2va) + $bridge + $k0k17 },
    @{ Name = 'd3_011_k0k17_conflict'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $bridge + $k0k17 + @('--T', 'off', '--C', 'on', '--L', '0') },
    @{ Name = 'd1_011_s05_conflict'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $bridge + $k0 + @('--T', 'off', '--C', 'on', '--L', '0.5') },
    @{ Name = 'd4_011_s05_dragon'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $dragon + @('--T', 'off', '--C', 'on', '--L', '0.5') }
)

$runnerLog = Join-Path $Arch ("run_stageD$suffix.runner.log")
Start-Transcript -Path $runnerLog -Append | Out-Null
try {
    if ($DryRun) { Invoke-Preflight -ShowOnly } else { Invoke-Preflight }
    foreach ($n in @($a2Key, $a2Ref, $pConflict, $KeyImage, $RefImage, $PromptRef)) {
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
    foreach ($r in $results) { Write-Host ("{0,-24} {1}" -f $r.Name, $r.Exit) }
    Write-Host "runner exit $runnerExit"
    Stop-Transcript | Out-Null
}
exit $runnerExit
