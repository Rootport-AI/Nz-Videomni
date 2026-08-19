<#
.SYNOPSIS
    Launch AviUtl2 (for manually verifying the plugin loads).

.PARAMETER ExePath
    Full path to aviutl2.exe. Defaults to the local v2.0.54 portable install location.

.PARAMETER Force
    Launch even if an AviUtl2 instance is already running.
#>
[CmdletBinding()]
param(
    [string]$ExePath = "D:\For_Videos\AviUtl2\aviutl2_v2.0.54\aviutl2.exe",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) {
    Write-Error "aviutl2.exe not found at: $ExePath"
    exit 1
}

$running = Get-Process -Name "aviutl2" -ErrorAction SilentlyContinue
if ($running -and -not $Force) {
    Write-Warning ("AviUtl2 is already running (PID $($running.Id -join ', ')). " +
        "Use -Force to launch another instance.")
    exit 0
}

Write-Host "Launching: $ExePath"
$proc = Start-Process -FilePath $ExePath -PassThru
Write-Host "Started AviUtl2 (PID $($proc.Id))."
