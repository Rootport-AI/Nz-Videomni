# sec. 1-116 T2: the whole text encoder, 3 pictures + 1 audio (N = 3,123), A/B against today. Plan sec. 5.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-te-vram-2026-10-29\run_t2.ps1 [-SkipPreflight]
#
# One condition = one process (run_common.ps1), each with a timeout. Reports: <archive>\<name>.json,
# console: <archive>\<name>.console.log. Hidden states (.pt): outputs\h3-te-vram-2026-10-29\ (not tracked).

param([switch]$SkipPreflight)
. (Join-Path $PSScriptRoot 'run_common.ps1') -SkipPreflight:$SkipPreflight
Invoke-Preflight

$a = "`"$Tmp\today_a.pt`""
$b = "`"$Tmp\today_b.pt`""
$to = 900

# today twice: run-to-run equality is the premise of G's torch.equal (plan sec. 5, Major 4).
Invoke-Cond 't2_pictures3_today_a' $T2 @('--mode', 'today', '--refs', 'pictures3', '--save', $a) -TimeoutSec $to
Invoke-Cond 't2_pictures3_today_b' $T2 @('--mode', 'today', '--refs', 'pictures3', '--baseline', $a, '--save', $b) -TimeoutSec $to
# probe: the real call's arguments (mask None, is_causal, shapes, use_gqa_in_sdpa); pass-through.
Invoke-Cond 't2_pictures3_probe' $T2 @('--mode', 'probe', '--refs', 'pictures3', '--baseline', $a) -TimeoutSec $to
# F1 without / with J.
Invoke-Cond 't2_pictures3_f1' $T2 @('--mode', 'f1', '--refs', 'pictures3', '--baseline', $a) -TimeoutSec $to
Invoke-Cond 't2_pictures3_f1_dropearly' $T2 @('--mode', 'f1', '--drop-early', '--refs', 'pictures3', '--baseline', $a) -TimeoutSec $to
# G (Bq from T1; g512 by default -- switch to g256 if T1 says so) + J.
Invoke-Cond 't2_pictures3_g512_dropearly' $T2 @('--mode', 'g512', '--drop-early', '--refs', 'pictures3', '--baseline', $a) -TimeoutSec $to
# sage + J: the yardstick of level 2.
Invoke-Cond 't2_pictures3_sage_dropearly' $T2 @('--mode', 'sage', '--drop-early', '--refs', 'pictures3', '--baseline', $a) -TimeoutSec $to
# F0 + J: T1 found cuDNN eligible with GQA and faster than F1 (53 vs 79 ms at N=16,128), so it is measured (plan sec. 9).
Invoke-Cond 't2_pictures3_f0_dropearly' $T2 @('--mode', 'f0', '--drop-early', '--refs', 'pictures3', '--baseline', $a) -TimeoutSec $to
