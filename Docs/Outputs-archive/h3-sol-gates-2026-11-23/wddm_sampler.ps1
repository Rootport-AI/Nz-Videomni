param(
    [Parameter(Mandatory = $true)][string]$Luid,
    [Parameter(Mandatory = $true)][string]$Csv,
    [Parameter(Mandatory = $true)][string]$Stop
)

# WDDM VRAM sampler (window-sweep experiment edition).
#
# Differences from outputs/vram_sweep/vram_sampler.ps1 (the 2.3-era original,
# left untouched):
#   * -SampleInterval is GONE. On this box it forced a >=1s wait per sample and
#     the observed effective period was ~5s. Without it Get-Counter returns the
#     current sample immediately, so the loop runs as fast as the counter API
#     allows and the real period is whatever we measure below.
#   * timestamps carry the DATE (yyyy-MM-dd HH:mm:ss.fff), so samples can be
#     joined against the worker-log arrival times and nvidia-smi dmon rows
#     across a run that crosses midnight.
#   * a third column, dt_ms, records the measured gap from the previous sample,
#     which is how the effective interval gets into the manifest instead of
#     being assumed.
# CSV columns: timestamp,dedicated_MB,shared_MB,dt_ms

"timestamp,dedicated_MB,shared_MB,dt_ms" | Out-File -FilePath $Csv -Encoding ascii

$dedPathPat = 'dedicated usage'
$shrPathPat = 'shared usage'
$prev = $null

while (-not (Test-Path -LiteralPath $Stop)) {
    try {
        $s = Get-Counter -Counter "\GPU Adapter Memory(*)\Dedicated Usage", "\GPU Adapter Memory(*)\Shared Usage" -MaxSamples 1 -ErrorAction Stop
        $ded = 0.0
        $shr = 0.0
        foreach ($smp in $s.CounterSamples) {
            if ($smp.Path -match [regex]::Escape($Luid)) {
                $mb = [math]::Round($smp.CookedValue / 1MB, 1)
                if ($smp.Path -match $dedPathPat) { $ded = $mb }
                elseif ($smp.Path -match $shrPathPat) { $shr = $mb }
            }
        }
        $now = Get-Date
        $dt = if ($null -eq $prev) { 0 } else { [math]::Round(($now - $prev).TotalMilliseconds, 0) }
        $prev = $now
        $ts = $now.ToString('yyyy-MM-dd HH:mm:ss.fff')
        "$ts,$ded,$shr,$dt" | Out-File -FilePath $Csv -Append -Encoding ascii
    }
    catch {
        Start-Sleep -Milliseconds 200
    }
}
