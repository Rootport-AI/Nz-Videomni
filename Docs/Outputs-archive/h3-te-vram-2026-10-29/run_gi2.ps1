# sec. 1-116 G-I1'..G-I3' (rerun after the short-tail fold, plan sec. 10.4): product text encoder gates.
# The stock baseline today_video6.pt from run_gi.ps1 (g_i2a_video6_today) is reused, not recomputed.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-te-vram-2026-10-29\run_gi2.ps1 [-SkipPreflight]

param([switch]$SkipPreflight)
. (Join-Path $PSScriptRoot 'run_common.ps1') -SkipPreflight:$SkipPreflight
Invoke-Preflight

$o = 'outputs\h3-te-vram-2026-10-29'
Invoke-Cond 'g_i1b_pictures3_product' $T2 @('--mode', 'product', '--refs', 'pictures3', '--baseline', "$o\today_a.pt") -TimeoutSec 900
Invoke-Cond 'g_i2c_video6_product' $T2 @('--mode', 'product', '--refs', 'video15', '--pairs', '6', '--baseline', "$o\today_video6.pt") -TimeoutSec 1800
Invoke-Cond 'g_i3b_pictures9_product_cap' $T2 @('--mode', 'product', '--refs', 'pictures9', '--cap') -TimeoutSec 1800
Invoke-Cond 'g_i3b_video15_product_cap' $T2 @('--mode', 'product', '--refs', 'video15', '--cap') -TimeoutSec 1800
