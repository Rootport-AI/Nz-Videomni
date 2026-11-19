# sec. 1-115 stage A2: G-R5 again (a prompt that disagrees with the key frame) and its splits, 7 runs.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-keyref-2026-11-19\run_stageA2.ps1
#   ... -Resume    skip every condition whose <archive>\<name>.json already says "ok": true
#   ... -DryRun    every condition with kr_single.py --dry-run (no model, no GPU); reports <name>.dryrun.json
#
# DO NOT RUN WITHOUT THE OWNER'S GPU APPROVAL. Order: preflight -> a1_010_conflict -> the reproduction gate
# (kr_metrics.py --first-psnr on a1, CPU only: frame 0 vs the key frame on the product's 1344x768 canvas;
# under 20 dB = reproduced) -> a2_010_consistent -> a3_ctrlF_conflict -> a4_ctrlR_conflict -> a5_110_conflict
# -> a6_001_conflict -> a7_011_conflict. One condition = one process (run_common.ps1), 1,500 s each.
# Reports: <archive>\<name>.json, console: <name>.console.log, the gate: stageA2_repro_gate.json, this runner:
# run_stageA2.runner.log (appended). mp4 / .pt: outputs\h3-keyref-2026-11-19\ (not tracked).
# Exit code: 0 = every condition ok; 1 = a stop (preflight, a1 failed, the gate could not be measured) or any
# condition not ok; 3 = a1 did not reproduce G-R5 (20 dB or more): recorded, nothing after it runs.
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
$ReproDb = 20.0
$Metrics = Join-Path $Arch 'kr_metrics.py'

$a2Key = 'Docs\Outputs-archive\h3-ref2va-2026-10-28\31d79f30-image.png'
$a2Ref = 'Docs\Outputs-archive\h3-ref2va-2026-10-28\17fde0c1-image.png'
$pConflict = "$Rel\prompt_a2_conflict.txt"
$pConsistent = "$Rel\prompt_a2_consistent.txt"
$pNoref = "$Rel\prompt_a2_conflict_noref.txt"
$size = @('--width', '1344', '--height', '768')
$key = @('--keyframe', "0=$a2Key")
$ref = @('--ref-image', $a2Ref)

# name, timeout (s), arguments; in the order of the stage's table.
$conds = @(
    @{ Name = 'a1_010_conflict'; Timeout = 1500; Args = @('--mode', 'product', '--transformer', $Ref2va, '--prompt-file', $pConflict) + $size + $key + $ref },
    @{ Name = 'a2_010_consistent'; Timeout = 1500; Args = @('--mode', 'product', '--transformer', $Ref2va, '--prompt-file', $pConsistent) + $size + $key + $ref },
    @{ Name = 'a3_ctrlF_conflict'; Timeout = 1500; Args = @('--mode', 'product', '--transformer', $Fl2va, '--prompt-file', $pNoref) + $size + $key },
    @{ Name = 'a4_ctrlR_conflict'; Timeout = 1500; Args = @('--mode', 'product', '--transformer', $Ref2va, '--prompt-file', $pNoref) + $size + $key },
    @{ Name = 'a5_110_conflict'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va, '--prompt-file', $pConflict) + $size + $key + $ref + @('--T', 'on', '--C', 'on') },
    @{ Name = 'a6_001_conflict'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va, '--prompt-file', $pConflict) + $size + $key + $ref + @('--T', 'off', '--C', 'off', '--L', '0') },
    @{ Name = 'a7_011_conflict'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va, '--prompt-file', $pConflict) + $size + $key + $ref + @('--T', 'off', '--C', 'on', '--L', '0') }
)

function Test-Reproduced {
    # The reproduction gate on a1 (CPU). Returns the PSNR (dB); throws 'STOP: ...' when it cannot be measured.
    $mp4 = Join-Path $OutDir 'a1_010_conflict.mp4'
    $rep = Join-Path $Arch 'a1_010_conflict.json'
    $out = & $Py $Metrics --first-psnr --mp4 $mp4 --report $rep
    if ($LASTEXITCODE -ne 0) { throw "STOP: the reproduction gate could not measure a1 (kr_metrics exit $LASTEXITCODE)" }
    $line = @($out | Where-Object { "$_" -like 'FIRST_PSNR=*' })
    if ($line.Count -lt 1) { throw "STOP: the reproduction gate got no FIRST_PSNR line ($out)" }
    return [double]::Parse(("$($line[0])" -split '=', 2)[1], [System.Globalization.CultureInfo]::InvariantCulture)
}

$runnerLog = Join-Path $Arch ("run_stageA2$suffix.runner.log")
Start-Transcript -Path $runnerLog -Append | Out-Null
try {
    if ($DryRun) { Invoke-Preflight -ShowOnly } else { Invoke-Preflight }
    foreach ($n in @($a2Key, $a2Ref, $pConflict, $pConsistent, $pNoref, "$Rel\kr_metrics.py")) {
        if (-not (Test-Path -LiteralPath (Join-Path $Repo $n))) { throw "STOP: missing: $n" }
    }
    foreach ($c in $conds) {
        $name = $c.Name
        if ($Resume -and (-not $DryRun) -and (Test-ReportOk $name)) {
            Write-Host "[skip] $name already ok"
            [void]$results.Add([pscustomobject]@{ Name = $name; Exit = 'skipped (ok before)' })
        } else {
            $condArgs = @('--name', $name) + $c.Args
            if ($DryRun) { $condArgs = $condArgs + @('--dry-run') }
            $rc = Invoke-Cond -Name $name -Script $KrSingle -CondArgs $condArgs -TimeoutSec $c.Timeout -Suffix $suffix `
                -NoGpuWait:$DryRun
            [void]$results.Add([pscustomobject]@{ Name = $name; Exit = $rc })
            if ($rc -ne 0) {
                $runnerExit = 1
                if ($name -eq 'a1_010_conflict') {
                    Write-Host "STOP: a1_010_conflict exit $($rc): the reproduction gate cannot be measured. Nothing after it runs."
                    break
                }
                Write-Host "$name exit $($rc): recorded, going on"
            }
        }
        if ($name -eq 'a1_010_conflict') {
            if ($DryRun) {
                Write-Host "(dry run) the reproduction gate is skipped (no a1 mp4); the threshold is $ReproDb dB"
                continue
            }
            $db = Test-Reproduced
            $reproduced = ($db -lt $ReproDb)
            $gate = [ordered]@{ run = 'a1_010_conflict'; first_psnr_db = $db; threshold_db = $ReproDb; reproduced = $reproduced;
                                measured = (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') }
            $gatePath = Join-Path $Arch 'stageA2_repro_gate.json'
            Move-Aside -Path $gatePath -Stamp (Get-Date -Format 'yyyyMMdd-HHmmss')
            $gate | ConvertTo-Json | Set-Content -LiteralPath $gatePath -Encoding utf8
            if ($reproduced) {
                Write-Host "reproduction gate: a1 frame 0 = $db dB (< $ReproDb): reproduced, going on"
            } else {
                Write-Host "reproduction gate: a1 frame 0 = $db dB (>= $ReproDb): NOT reproduced. Stage A2 stops here."
                $runnerExit = 3
                break
            }
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
