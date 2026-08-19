<#
.SYNOPSIS
    Package the (embedded) NzLTX23.aux2 and its Language files into a
    distributable .au2pkg.zip (milestone M7c release packaging).

.DESCRIPTION
    Produces  dist\NzLTX23-<Version>.au2pkg.zip  with the multi-file plugin
    layout AviUtl2 expects (Docs\SDK_REFERENCE.md section 1). Only the folders
    Plugin\ Language\ (and the package.ini / package.txt manifest) are placed at
    the zip root; AviUtl2 extracts them relative to its application-data folder:

        NzLTX23-<Version>.au2pkg.zip
        |--- Plugin\NzLTX23\NzLTX23.aux2
        |--- Language\English.NzLTX23.aul2
        |--- Language\Japanese.NzLTX23.aul2
        |--- package.ini
        |--- package.txt

    The .aux2 is taken from the build output and MUST have been built with the
    single-file Web UI embedded (scripts\build.ps1's default, or explicitly
    -EmbedWebui), so no webui\ folder ships in the package. Embedded-only
    operation (2026-07-17): the script stops with an error (throw) if the
    embedded Web UI resource is not detected in the DLL, since packaging a
    non-embedded build would ship a broken release with no UI. See
    scripts\deploy.ps1's header comment for the identical embedded-only
    rationale.

.PARAMETER Config
    Which build output to package: Release (default) or Debug.

.PARAMETER Version
    Version string used in the zip name and package.ini. Defaults to the current
    plugin version (kPluginVersion), 1.0.0-rc1.

.PARAMETER OutDir
    Output directory for the .au2pkg.zip. Defaults to <repo>\dist.
#>
[CmdletBinding()]
param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",
    [string]$Version = "1.0.0-rc1",
    [string]$OutDir
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

$preset  = if ($Config -eq "Debug") { "ninja-debug" } else { "ninja-release" }
$aux2    = Join-Path $RepoRoot "build\$preset\NzLTX23.aux2"
if (-not $OutDir) { $OutDir = Join-Path $RepoRoot "dist" }

if (-not (Test-Path $aux2)) {
    throw "Artifact not found: $aux2`nBuild it first: scripts\build.ps1 -Config $Config -EmbedWebui"
}

# --- hard check: is the single-file Web UI actually embedded? ---------------
# NOTE: scripts\deploy.ps1 runs the identical dual-needle check as a hard
# gate before deploying. Keep this comment and deploy.ps1's in sync if either
# search, the resource name, or the HTML content marker changes.
#
# A single-needle search for the UTF-16LE resource name NZLTX_WEBUI_INDEX is
# NOT sufficient on its own: that string is the literal name argument passed
# to FindResourceW() in native/src/webview_host.cpp, so it is compiled into
# the DLL's code unconditionally regardless of -DNZLTX_EMBED_WEBUI, and its
# raw bytes are present in EVERY build, embedded or not (verified empirically
# 2026-07-17; this used to be exactly the soft Write-Warning check below,
# which as a result silently always reported "embedded" -- see
# Docs\DEVLOG.md §16.2/§16.5). The authoritative signal is the embedded HTML
# *content* itself: native/res/webui.rc.in embeds webui\dist-single\index.html
# verbatim as the RCDATA resource body, and that file always starts with the
# literal ASCII bytes "<!doctype html" (Vite's standard single-file output) --
# bytes that only end up in the .aux2 when the HTML was actually linked in.
# Both needles must match.
function Test-ByteNeedle([byte[]]$Haystack, [byte[]]$Needle) {
    for ($i = 0; $i -le ($Haystack.Length - $Needle.Length); $i++) {
        $match = $true
        for ($j = 0; $j -lt $Needle.Length; $j++) {
            if ($Haystack[$i + $j] -ne $Needle[$j]) { $match = $false; break }
        }
        if ($match) { return $true }
    }
    return $false
}

$bytes = [System.IO.File]::ReadAllBytes($aux2)
$resourceNameNeedle = [System.Text.Encoding]::Unicode.GetBytes("NZLTX_WEBUI_INDEX")
$htmlContentNeedle = [System.Text.Encoding]::ASCII.GetBytes("<!doctype html")
$embedded = (Test-ByteNeedle $bytes $resourceNameNeedle) -and (Test-ByteNeedle $bytes $htmlContentNeedle)
if ($embedded) {
    Write-Host "Embedded Web UI resource detected in NzLTX23.aux2."
} else {
    throw ("$aux2 does not contain the embedded Web UI resource. Packaging a " +
        "non-embedded build is not supported (embedded-only operation): the " +
        ".au2pkg.zip would ship a plugin with no UI. Rebuild with " +
        "scripts\build.ps1 (embeds the Web UI by default; do not pass " +
        "-NoEmbedWebui).")
}

# --- stage the package tree --------------------------------------------------
$stage = Join-Path $RepoRoot "build\package-stage"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
$pluginStage   = Join-Path $stage "Plugin\NzLTX23"
$languageStage = Join-Path $stage "Language"
New-Item -ItemType Directory -Force -Path $pluginStage   | Out-Null
New-Item -ItemType Directory -Force -Path $languageStage | Out-Null

Copy-Item -Path $aux2 -Destination (Join-Path $pluginStage "NzLTX23.aux2") -Force

$languageSrc = Join-Path $RepoRoot "Language"
$langFiles = Get-ChildItem -Path $languageSrc -Filter "*.NzLTX23.aul2" -ErrorAction SilentlyContinue
if (-not $langFiles) { Write-Warning "No Language\*.NzLTX23.aul2 files found; package will ship without them." }
foreach ($f in $langFiles) {
    Copy-Item -Path $f.FullName -Destination (Join-Path $languageStage $f.Name) -Force
}

# --- package.ini / package.txt manifest --------------------------------------
$packageIni = @"
[package]
id=NzLTX23
name=Nz-LTX23
information=LTX 2.3 video generation frontend for AviUtl2 ($Version)
uninstallSubFolderFile=Plugin\NzLTX23\NzLTX23.aux2
"@
$packageTxt = @"
Nz-LTX23 $Version
LTX 2.3 video generation frontend for AviUtl2.

A dockable window plugin (.aux2) that hosts the LTX 2.3 generation Web UI and
bridges it to the AviUtl2 timeline (frame capture, media insert, uploads).
The Web UI is embedded in the plugin DLL; no extra files are required beyond
the language files in this package.
"@
Set-Content -Path (Join-Path $stage "package.ini") -Value $packageIni -Encoding UTF8
Set-Content -Path (Join-Path $stage "package.txt") -Value $packageTxt -Encoding UTF8

# --- zip it ------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$zipPath = Join-Path $OutDir "NzLTX23-$Version.au2pkg.zip"
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -Force

Write-Host ""
Write-Host "Package created: $zipPath"
Write-Host "Contents:"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    foreach ($entry in $zip.Entries | Sort-Object FullName) {
        "  {0,10}  {1}" -f $entry.Length, $entry.FullName | Write-Host
    }
} finally {
    $zip.Dispose()
}
