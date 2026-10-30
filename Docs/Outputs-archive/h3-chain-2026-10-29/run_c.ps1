# sec. 1-106 Chained (latent carry-over) measurement: the conditions in the plan's order (plan sec. 5).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-chain-2026-10-29\run_c.ps1 [-SkipPreflight]
#
# selfcheck -> C0 -> C1 -> C3 -> C2 -> C4 -> C5 -> C7 -> analysis. One condition = one process (run_common.ps1),
# each with a timeout. Reports: <archive>\<name>.json, console: <archive>\<name>.console.log.
# mp4 / .pt: outputs\h3-chain-2026-10-29\<name>\ (not tracked).
# Stops (plan sec. 5): selfcheck not 0; C0 not 0 (2 = fingerprint or recheck mismatch, 1 = failed);
# C1 not 0 (the later conditions reuse its clip 0).
# Common to every GPU condition: FL2VA int8 ConvRot, 20 steps, shift 12/3, VRAM 16GB mode on, prefetch on,
# blocks_on_gpu 8 (4 in effect with VRAM 16GB mode), T2V, 124 frames x 3 clips, seed 42 (+i per clip).

param([switch]$SkipPreflight)
. (Join-Path $PSScriptRoot 'run_common.ps1') -SkipPreflight:$SkipPreflight
Invoke-Preflight

$rel = 'Docs\Outputs-archive\h3-chain-2026-10-29'
$pc = "$rel\prompts_c.json"
$pc0 = "$rel\prompts_c0.json"
$c1json = "$rel\c1_base.json"

function Stop-Run([string]$why) {
    Write-Host "STOP: $why"
    exit 1
}

$rc = Invoke-Cond 'selfcheck' $C @('--selfcheck') -TimeoutSec 300
if ($rc -ne 0) { Stop-Run "selfcheck exit $rc" }

# C0: the fingerprint gate (640x384, sdpa, the common prompt on all three clips; recheck of clip 0).
$rc = Invoke-Cond 'c0_gate' $C @('--name', 'c0_gate', '--width', '640', '--height', '384', '--overlap', '22',
    '--strength', '1.0', '--inject', 'noise', '--attention', 'sdpa', '--prompts', $pc0, '--decode', 'both',
    '--save-latents', '--fingerprint', '--recheck-clip0') -TimeoutSec 1500
if ($rc -ne 0) { Stop-Run "C0 exit $rc (2 = fingerprint or recheck mismatch, 1 = failed, -1 = timeout)" }

# C1: the base (1344x768, sage). Its clip 0 is reused by every later condition.
$rc = Invoke-Cond 'c1_base' $C @('--name', 'c1_base', '--overlap', '22', '--strength', '1.0', '--inject', 'noise',
    '--attention', 'sage', '--prompts', $pc, '--decode', 'both', '--save-latents') -TimeoutSec 3600
if ($rc -ne 0) { Stop-Run "C1 exit $rc (the later conditions need its clip 0)" }

$reuse = @('--reuse-clip0', $c1json, '--attention', 'sage', '--prompts', $pc, '--save-latents')
# C3: inject clean (H2).
$null = Invoke-Cond 'c3_inject_clean' $C (@('--name', 'c3_inject_clean', '--overlap', '22', '--strength', '1.0',
    '--inject', 'clean', '--decode', 'both') + $reuse) -TimeoutSec 3000
# C2: strength 0.5 (H1, H5: (a)(b) and the cross-fade (c)).
$null = Invoke-Cond 'c2_strength05' $C (@('--name', 'c2_strength05', '--overlap', '22', '--strength', '0.5',
    '--inject', 'noise', '--decode', 'both') + $reuse) -TimeoutSec 3000
# C4: overlap 5 frames (H3, H6).
$null = Invoke-Cond 'c4_overlap5' $C (@('--name', 'c4_overlap5', '--overlap', '5', '--strength', '1.0',
    '--inject', 'noise', '--decode', 'both') + $reuse) -TimeoutSec 3000
# C5: overlap 39 frames (H3, H6).
$null = Invoke-Cond 'c5_overlap39' $C (@('--name', 'c5_overlap39', '--overlap', '39', '--strength', '1.0',
    '--inject', 'noise', '--decode', 'both') + $reuse) -TimeoutSec 3000
# C7: the control, no fix (strength 0 = O 0): three independent clips side by side, decoded per clip.
$null = Invoke-Cond 'c7_control' $C (@('--name', 'c7_control', '--strength', '0', '--decode', 'perclip') + $reuse) -TimeoutSec 3000

# Analysis (CPU): CSVs next to the reports.
$null = Invoke-Cond 'analysis' $An @('--c1', 'c1_base', '--c3', 'c3_inject_clean', '--c7', 'c7_control') -TimeoutSec 1800
