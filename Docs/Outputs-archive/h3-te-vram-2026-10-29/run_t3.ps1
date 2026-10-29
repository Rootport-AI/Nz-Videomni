# sec. 1-116 T3: the two pass cases under the VRAM 16GB mode allocator cap. Plan sec. 6.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-te-vram-2026-10-29\run_t3.ps1 [-SkipPreflight]
#
# pictures9 = the 3 pictures x 3 (N = 9,258); video15 = 16 pairs of the 29.4 s 1792x1024 output (N = 16,314,
# t2_te_ab.py asserts N >= 16,000 before encode). "today" is not run (plan sec. 6). Pass = finishes under the cap
# (allocator num_ooms 0, decoder_peak_reserved_mb below cap_info.cap_mib).

param([switch]$SkipPreflight)
. (Join-Path $PSScriptRoot 'run_common.ps1') -SkipPreflight:$SkipPreflight
Invoke-Preflight

$to = 1800
foreach ($refs in @('pictures9', 'video15')) {
    Invoke-Cond "t3_${refs}_f1_dropearly_cap" $T2 @('--mode', 'f1', '--drop-early', '--cap', '--refs', $refs) -TimeoutSec $to
    # T2 showed g512 + J torch.equal with today (bit-identical over 50 layers):
    Invoke-Cond "t3_${refs}_g512_dropearly_cap" $T2 @('--mode', 'g512', '--drop-early', '--cap', '--refs', $refs) -TimeoutSec $to
    # T1 showed cuDNN eligible with GQA and faster than F1:
    Invoke-Cond "t3_${refs}_f0_dropearly_cap" $T2 @('--mode', 'f0', '--drop-early', '--cap', '--refs', $refs) -TimeoutSec $to
}
