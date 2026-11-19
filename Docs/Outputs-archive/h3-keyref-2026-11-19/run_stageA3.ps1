# sec. 1-115 stage A3: the key frame's role declared in the prompt ("<Picture 2> is the first frame of the
# video."), on the stage A2 material (the prompt that disagrees with the key frame), 3 runs.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-keyref-2026-11-19\run_stageA3.ps1
#   ... -Resume    skip every condition whose <archive>\<name>.json already says "ok": true
#   ... -DryRun    every condition with kr_single.py --dry-run (no model, no GPU); reports <name>.dryrun.json
#
# DO NOT RUN WITHOUT THE OWNER'S GPU APPROVAL. Order: preflight -> a8_110_decl -> a9_keyasref_decl ->
# a10_keyasref_nodecl. One condition = one process (run_common.ps1), 1,500 s each; a condition not 0 is
# recorded and the run goes on. Reports: <archive>\<name>.json, console: <name>.console.log, this runner:
# run_stageA3.runner.log (appended). mp4 / .pt: outputs\h3-keyref-2026-11-19\ (not tracked).
# Exit code: 0 = every condition ok, 1 = otherwise.
#   a8: the gentleman as <Picture 1>; the key frame (the bridge) at frame 0 as T (<Picture 2>, text encoder only) + C.
#   a9 / a10: the product as is, with two reference pictures (the gentleman <Picture 1>, the bridge <Picture 2>)
#   and the bridge also as the key frame at frame 0 (C); a9 with the declaration, a10 without (the A2 conflict prompt).
# Common: 1344x768, 124 frames, 20 steps, shift 12/3, seed 42, sage, deterministic on, blocks_on_gpu 8,
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
$pDecl = "$Rel\prompt_a3_decl.txt"
$pConflict = "$Rel\prompt_a2_conflict.txt"
$size = @('--width', '1344', '--height', '768')
$key = @('--keyframe', "0=$a2Key")
$refs2 = @('--ref-image', $a2Ref, '--ref-image', $a2Key)

$conds = @(
    @{ Name = 'a8_110_decl'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va, '--prompt-file', $pDecl) + $size + $key + @('--ref-image', $a2Ref, '--T', 'on', '--C', 'on') },
    @{ Name = 'a9_keyasref_decl'; Timeout = 1500; Args = @('--mode', 'product', '--transformer', $Ref2va, '--prompt-file', $pDecl) + $size + $key + $refs2 },
    @{ Name = 'a10_keyasref_nodecl'; Timeout = 1500; Args = @('--mode', 'product', '--transformer', $Ref2va, '--prompt-file', $pConflict) + $size + $key + $refs2 }
)

$runnerLog = Join-Path $Arch ("run_stageA3$suffix.runner.log")
Start-Transcript -Path $runnerLog -Append | Out-Null
try {
    if ($DryRun) { Invoke-Preflight -ShowOnly } else { Invoke-Preflight }
    foreach ($n in @($a2Key, $a2Ref, $pDecl, $pConflict)) {
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
    foreach ($r in $results) { Write-Host ("{0,-20} {1}" -f $r.Name, $r.Exit) }
    Write-Host "runner exit $runnerExit"
    Stop-Transcript | Out-Null
}
exit $runnerExit
