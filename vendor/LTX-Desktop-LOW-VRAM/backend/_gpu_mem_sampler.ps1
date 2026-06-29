param(
    [string]$OutFile = "S:\OriginalApps\12_Nz-LTX23-backend\outputs\gpu_mem_sampler.log",
    [double]$IntervalSec = 0.5,
    [int]$MaxSeconds = 3600
)
# Samples Windows GPU Adapter Memory perf counters (same source as Task Manager).
# Logs timestamp, max Dedicated Usage (MB), max Shared Usage (MB) across all adapters.
"timestamp,dedicated_MB,shared_MB" | Out-File -FilePath $OutFile -Encoding utf8
$deadline = (Get-Date).AddSeconds($MaxSeconds)
while ((Get-Date) -lt $deadline) {
    try {
        $ded = (Get-Counter "\GPU Adapter Memory(*)\Dedicated Usage" -ErrorAction Stop).CounterSamples |
               Measure-Object -Property CookedValue -Maximum
        $shr = (Get-Counter "\GPU Adapter Memory(*)\Shared Usage" -ErrorAction Stop).CounterSamples |
               Measure-Object -Property CookedValue -Maximum
        $ts = (Get-Date).ToString("HH:mm:ss.fff")
        $dMB = [math]::Round($ded.Maximum / 1MB, 1)
        $sMB = [math]::Round($shr.Maximum / 1MB, 1)
        "$ts,$dMB,$sMB" | Out-File -FilePath $OutFile -Append -Encoding utf8
    } catch {
        # transient counter failure: skip this sample, keep going
    }
    Start-Sleep -Seconds $IntervalSec
}
