# sec. 1-115 stage C: five key frames (key-001..005 at frames 0, 34, 51, 85, 119 = latents 0, 10, 15, 25, 35)
# with a reference audio as <Audio 1> (no reference picture), 5 runs (the gate's 2 + 3).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-keyref-2026-11-19\run_stageC.ps1
#   ... -Resume    skip every condition whose <archive>\<name>.json already says "ok": true
#   ... -DryRun    every condition with kr_single.py --dry-run (no model, no GPU); reports <name>.dryrun.json
#
# DO NOT RUN WITHOUT THE OWNER'S GPU APPROVAL. Order: preflight -> gate_C (c1_010_audio_harness then
# c1_010_audio_product in one process, kr_single.py --mode gate: the video / audio stream MD5s must match;
# 3,000 s) -> c2_001_audio -> c3_011_audio -> c4_ctrl_noaudio. One condition = one process (run_common.ps1),
# 1,500 s each. The gate not 0 stops the stage (nothing after it runs); any other condition not 0 is recorded
# and the run goes on. Reports: <archive>\<name>.json (the gate: gate_C.json and the two runs' own), console:
# <name>.console.log, this runner: run_stageC.runner.log (appended). mp4 / .pt: outputs\h3-keyref-2026-11-19\.
# Exit code: 0 = every condition ok, 2 = the gate's streams differ, 1 = otherwise.
# c4 has no reference, so the product also shows the key frame at frame 0 to the text encoder (only frames 0 and
# F-1 go there; 119 is not F-1 = 123): a difference recorded, not removed.
# Common: 1280x768, 124 frames, 20 steps, shift 12/3, seed 42, sage, deterministic on, blocks_on_gpu 8,
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

$kdir = 'uploads\key-ref-development'
$refAudio = "$kdir\[OP用]OPED用ジングル６.mp3"
$pAudio = "$Rel\prompt_c_audio.txt"
$pNoaudio = "$Rel\prompt_c_noaudio.txt"
$keys = @('--keyframe', "0=$kdir\key-001.jpg", '--keyframe', "34=$kdir\key-002.jpg", '--keyframe', "51=$kdir\key-003.jpg",
          '--keyframe', "85=$kdir\key-004.jpg", '--keyframe', "119=$kdir\key-005.jpg")
$withAudio = @('--prompt-file', $pAudio) + $keys + @('--ref-audio', $refAudio)

$conds = @(
    @{ Name = 'gate_C'; Timeout = 3000; Extra = @('c1_010_audio_harness', 'c1_010_audio_product');
       Args = @('--mode', 'gate', '--gate-names', 'c1_010_audio_harness,c1_010_audio_product', '--transformer', $Ref2va) + $withAudio },
    @{ Name = 'c2_001_audio'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $withAudio + @('--T', 'off', '--C', 'off', '--L', '0') },
    @{ Name = 'c3_011_audio'; Timeout = 1500; Args = @('--mode', 'harness', '--transformer', $Ref2va) + $withAudio + @('--T', 'off', '--C', 'on', '--L', '0') },
    @{ Name = 'c4_ctrl_noaudio'; Timeout = 1500; Args = @('--mode', 'product', '--transformer', $Ref2va, '--prompt-file', $pNoaudio) + $keys }
)

$runnerLog = Join-Path $Arch ("run_stageC$suffix.runner.log")
Start-Transcript -Path $runnerLog -Append | Out-Null
try {
    if ($DryRun) { Invoke-Preflight -ShowOnly } else { Invoke-Preflight }
    foreach ($n in @("$kdir\key-002.jpg", "$kdir\key-003.jpg", "$kdir\key-004.jpg", "$kdir\key-005.jpg", $refAudio, $pAudio, $pNoaudio)) {
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
    foreach ($r in $results) { Write-Host ("{0,-20} {1}" -f $r.Name, $r.Exit) }
    Write-Host "runner exit $runnerExit"
    Stop-Transcript | Out-Null
}
exit $runnerExit
