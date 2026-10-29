# sec. 1-116 G-I1..G-I3: product text encoder (G+J built in) gates. Plan sec. 5.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-te-vram-2026-10-29\run_gi.ps1 [-SkipPreflight]

param([switch]$SkipPreflight)
. (Join-Path $PSScriptRoot 'run_common.ps1') -SkipPreflight:$SkipPreflight
Invoke-Preflight

$o = 'outputs\h3-te-vram-2026-10-29'
Invoke-Cond 'g_i1_pictures3_product' $T2 @('--mode', 'product', '--refs', 'pictures3', '--baseline', "$o\today_a.pt") -TimeoutSec 900
# stock path at N~6,148: spills into shared memory (~25 GB), about 10 minutes (627 s measured)
Invoke-Cond 'g_i2a_video6_today' $T2 @('--mode', 'today', '--refs', 'video15', '--pairs', '6', '--save', "$o\today_video6.pt") -TimeoutSec 1800
Invoke-Cond 'g_i2b_video6_product' $T2 @('--mode', 'product', '--refs', 'video15', '--pairs', '6', '--baseline', "$o\today_video6.pt") -TimeoutSec 1800
Invoke-Cond 'g_i3_pictures9_product_cap' $T2 @('--mode', 'product', '--refs', 'pictures9', '--cap') -TimeoutSec 1800
Invoke-Cond 'g_i3_video15_product_cap' $T2 @('--mode', 'product', '--refs', 'video15', '--cap') -TimeoutSec 1800
