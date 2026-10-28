# sec. 1-116 T1: one attention call per SDPA kernel (no weights). Plan sec. 4.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-te-vram-2026-10-29\run_t1.ps1 [-SkipPreflight]
#
# Run from anywhere (the script moves to the repo root). One condition = one process (run_common.ps1),
# each with a timeout. Reports: <archive>\<name>.json, console: <archive>\<name>.console.log.
# Saved outputs (.pt, baselines): outputs\h3-te-vram-2026-10-29\ (not tracked).

param([switch]$SkipPreflight)
. (Join-Path $PSScriptRoot 'run_common.ps1') -SkipPreflight:$SkipPreflight
Invoke-Preflight

$b1062 = "`"$Tmp\t1_today_n1062.pt`""
$b3130 = "`"$Tmp\t1_today_n3130.pt`""
$b3123 = "`"$Tmp\t1_today_n3123.pt`""

# today: the stock call (default backend choice); the outputs are the baselines.
Invoke-Cond 't1_today_n1062' $T1 @('--n', '1062', '--kernel', 'today', '--save', $b1062)
Invoke-Cond 't1_today_n3130' $T1 @('--n', '3130', '--kernel', 'today', '--save', $b3130)
# math forced: torch.equal with today = today runs the math kernel.
Invoke-Cond 't1_math_n1062' $T1 @('--n', '1062', '--kernel', 'math', '--baseline', $b1062)
Invoke-Cond 't1_math_n3130' $T1 @('--n', '3130', '--kernel', 'math', '--baseline', $b3130)
# probe: why efficient / cuDNN are (not) eligible.
Invoke-Cond 't1_probe_n3130' $T1 @('--n', '3130', '--kernel', 'probe')
Invoke-Cond 't1_probe_n16128' $T1 @('--n', '16128', '--kernel', 'probe')
# f0: cuDNN forced (may raise, or fault: the timeout guards the rest).
Invoke-Cond 't1_f0_n3130' $T1 @('--n', '3130', '--kernel', 'f0', '--baseline', $b3130) -TimeoutSec 300
Invoke-Cond 't1_f0_n9300' $T1 @('--n', '9300', '--kernel', 'f0') -TimeoutSec 300
Invoke-Cond 't1_f0_n16128' $T1 @('--n', '16128', '--kernel', 'f0') -TimeoutSec 300
# f1: k / v repeated to 64 heads, memory-efficient kernel forced.
Invoke-Cond 't1_f1_n3130' $T1 @('--n', '3130', '--kernel', 'f1', '--baseline', $b3130)
Invoke-Cond 't1_f1_n9300' $T1 @('--n', '9300', '--kernel', 'f1')
Invoke-Cond 't1_f1_n16128' $T1 @('--n', '16128', '--kernel', 'f1')
# g512 / g256: math, query blocks with bool mask slices.
Invoke-Cond 't1_g512_n3130' $T1 @('--n', '3130', '--kernel', 'g512', '--baseline', $b3130)
Invoke-Cond 't1_g512_n9300' $T1 @('--n', '9300', '--kernel', 'g512')
Invoke-Cond 't1_g512_n16128' $T1 @('--n', '16128', '--kernel', 'g512')
Invoke-Cond 't1_g256_n3130' $T1 @('--n', '3130', '--kernel', 'g256', '--baseline', $b3130)
# Added: N = 3,123 is T2's measured N (pictures3, t2_te_ab.py --selfcheck); G's torch.equal depends on the shape (plan sec. 4).
Invoke-Cond 't1_today_n3123' $T1 @('--n', '3123', '--kernel', 'today', '--save', $b3123)
Invoke-Cond 't1_g512_n3123' $T1 @('--n', '3123', '--kernel', 'g512', '--baseline', $b3123)
Invoke-Cond 't1_g256_n3123' $T1 @('--n', '3123', '--kernel', 'g256', '--baseline', $b3123)
