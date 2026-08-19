<#
.SYNOPSIS
    Deploy NzVideomni.aux2 (embedded Web UI build only) into the AviUtl2 Plugin
    folder.

.DESCRIPTION
    Embedded-only deploy (2026-07-17 one-off to embedded-only operation): the
    non-embedded layout (a bare .aux2 plus an on-disk Plugin\NzVideomni\webui\
    folder copied from webui\dist) existed only because rebuilding the native
    side used to be required to pick up webui\ changes. That reason went away
    once CMake's OBJECT_DEPENDS was fixed to track webui\dist-single\index.html
    (see README.md's former "webui-only rebuild trap" note), and all
    development since has used embedded builds exclusively (scripts\build.ps1,
    embedding by default). This script therefore now REQUIRES an embedded
    .aux2 and no longer copies a webui\ folder:

        <PluginDir>\NzVideomni\NzVideomni.aux2   (embedded Web UI resource inside)

    If the build artifact does not contain the embedded Web UI resource, this
    script stops with an error and makes NO changes to the deployment target
    (checked before any copy/delete against the target). Rebuild with
    scripts\build.ps1 (embeds by default; pass -NoEmbedWebui only for
    native-only debug builds, which this script will then refuse to deploy).

    If a stale Plugin\NzVideomni\webui\ folder exists on the target from an
    earlier non-embedded deploy, it is removed as part of the (embedded)
    deploy, since the embedded .aux2 does not need it and its presence is
    leftover development-era clutter.

    The old single-file layout <PluginDir>\NzVideomni.aux2 is removed if present,
    otherwise the host would load the plugin twice.

    Milestone M7a also deploys the native-side Language files (see Language\)
    next to Plugin\, matching the .au2pkg.zip layout documented in
    Docs\SDK_REFERENCE.md section 1 (Language\ is a sibling of Plugin\, not a
    subfolder of it):

        <PluginDir's parent>\Language\English.NzVideomni.aul2
        <PluginDir's parent>\Language\Japanese.NzVideomni.aul2

    Refuses to deploy while AviUtl2 is running (the DLL would be locked and the
    host would not pick up the new build). Build first with scripts\build.ps1.

.PARAMETER Config
    Which build output to deploy: Release (default) or Debug.

.PARAMETER PluginDir
    Root plugin directory. Defaults to D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin.

.PARAMETER DistDir
    Distribution copy inside the backend repo (2026-07-31, owner decision):
    the .aux2 is committed to the backend repo so end users get it via
    git clone and install it by drag-and-drop, replacing the old zip-based
    distribution. Every deploy also refreshes this copy so the repo never
    ships a stale build. Pass an empty string to skip.
#>
[CmdletBinding()]
param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",
    [string]$PluginDir = "D:\For_Videos\AviUtl2\aviutl2_v2.0.54\data\Plugin",
    [string]$DistDir = "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\AviUtl2-Plugin"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

$preset = if ($Config -eq "Debug") { "ninja-debug" } else { "ninja-release" }
$aux2 = Join-Path $RepoRoot "build\$preset\NzVideomni.aux2"

if (-not (Test-Path $aux2)) {
    Write-Error "Artifact not found: $aux2`nRun scripts\build.ps1 -Config $Config first."
    exit 1
}

# Abort if AviUtl2 is running (the plugin file would be locked / not reloaded).
$running = Get-Process -Name "aviutl2" -ErrorAction SilentlyContinue
if ($running) {
    Write-Error ("AviUtl2 is currently running (PID $($running.Id -join ', ')). " +
        "Close it before deploying, otherwise the .aux2 is locked and the new build " +
        "will not be loaded.")
    exit 1
}

# --- embedded-only guard: refuse non-embedded builds, before touching the ----
# --- deploy target at all -----------------------------------------------------
# NOTE: scripts\package.ps1 (L57-89) runs the identical dual-needle check as
# its own hard gate (a `throw`, since packaging a non-embedded build into the
# .au2pkg.zip would ship a broken release). A single needle search for the
# UTF-16LE resource name NZVIDEOMNI_WEBUI_INDEX is NOT sufficient on its own: the
# L"NZVIDEOMNI_WEBUI_INDEX" string literal passed to FindResourceW() in
# native/src/webview_host.cpp is compiled into the DLL's code unconditionally,
# regardless of -DNZVIDEOMNI_EMBED_WEBUI, so its raw bytes are present in EVERY
# build (embedded or not) -- verified empirically (2026-07-17) that relying on
# it alone makes this gate a no-op (a -NoEmbedWebui build still matched; this
# was also a latent, harmless-as-a-warning flaw in package.ps1's original soft
# check, since fixed there too). The authoritative signal added below is the
# embedded HTML *content* itself: native/res/webui.rc.in embeds
# webui\dist-single\index.html verbatim as the RCDATA resource body, and that
# file always starts with the literal ASCII bytes "<!doctype html" (Vite's
# standard single-file output) -- bytes that only end up in the .aux2 when
# the HTML was actually linked in. Both needles must match; keep this comment
# and package.ps1's in sync if either search or the resource name changes.
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
$resourceNameNeedle = [System.Text.Encoding]::Unicode.GetBytes("NZVIDEOMNI_WEBUI_INDEX")
$htmlContentNeedle = [System.Text.Encoding]::ASCII.GetBytes("<!doctype html")
$embedded = (Test-ByteNeedle $bytes $resourceNameNeedle) -and (Test-ByteNeedle $bytes $htmlContentNeedle)
if (-not $embedded) {
    Write-Error ("$aux2 does not contain the embedded Web UI resource. " +
        "Deploying a non-embedded build is not supported (embedded-only " +
        "operation). Rebuild with scripts\build.ps1 (embeds the Web UI by " +
        "default; do not pass -NoEmbedWebui). The deploy target was not " +
        "touched.")
    exit 1
}

# --- clean up the old single-file layout (avoid a double load) --------------
$legacy = Join-Path $PluginDir "NzVideomni.aux2"
if (Test-Path $legacy) {
    Write-Host "Removing stale legacy plugin: $legacy"
    Remove-Item -Path $legacy -Force
}

# --- deploy into the NzVideomni subfolder --------------------------------------
$destDir = Join-Path $PluginDir "NzVideomni"
$webuiDir = Join-Path $destDir "webui"
if (-not (Test-Path $destDir)) {
    Write-Host "Creating plugin directory: $destDir"
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
}

$destAux2 = Join-Path $destDir "NzVideomni.aux2"
Copy-Item -Path $aux2 -Destination $destAux2 -Force
Write-Host "Deployed plugin:"
Write-Host "  from $aux2"
Write-Host "  to   $destAux2"

# --- clean up a stale non-embedded-era Web UI folder, if any ----------------
# The .aux2 deployed above is guaranteed embedded (checked earlier), so it
# does not read from an on-disk webui\ folder. Remove any leftover from an
# earlier non-embedded deploy so it doesn't linger as dead weight.
if (Test-Path $webuiDir) {
    Remove-Item -Path $webuiDir -Recurse -Force
    Write-Host "Removed stale on-disk Web UI folder: $webuiDir"
}

# --- distribution copy into the backend repo (git-clone-and-drag-drop) ------
# Probe guard (2026-07-31): an investigation build (NZVIDEOMNI_PROBE_VIDEO_ITEMS)
# must never reach the distribution copy the backend repo ships to end users.
# The probe's log prefix "[PROBE]" only exists in the binary when the probe
# code was compiled in, so its absence is the shipping signal. The primary
# (local machine) deploy above intentionally still runs - that is where the
# investigation build is used.
$probeNeedle = [System.Text.Encoding]::ASCII.GetBytes("[PROBE]")
$isProbeBuild = Test-ByteNeedle $bytes $probeNeedle
if ($isProbeBuild -and $DistDir) {
    Write-Warning ("Investigation build detected ([PROBE] marker present); " +
        "SKIPPING the distribution copy to $DistDir. Rebuild without " +
        "-ProbeVideoItems before releasing.")
    $DistDir = ""
}
if ($DistDir) {
    if (-not (Test-Path $DistDir)) {
        Write-Host "Creating distribution directory: $DistDir"
        New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
    }
    $distAux2 = Join-Path $DistDir "NzVideomni.aux2"
    Copy-Item -Path $aux2 -Destination $distAux2 -Force
    Write-Host "Deployed distribution copy:"
    Write-Host "  to   $distAux2"
}

# --- deploy the native Language files (Language\ is a sibling of Plugin\) ---
$languageSrc = Join-Path $RepoRoot "Language"
$languageDir = Join-Path (Split-Path -Parent $PluginDir) "Language"
if (Test-Path $languageSrc) {
    if (-not (Test-Path $languageDir)) {
        Write-Host "Creating language directory: $languageDir"
        New-Item -ItemType Directory -Force -Path $languageDir | Out-Null
    }
    Get-ChildItem -Path $languageSrc -Filter "*.NzVideomni.aul2" | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination (Join-Path $languageDir $_.Name) -Force
    }
    Write-Host "Deployed language files:"
    Write-Host "  from $languageSrc"
    Write-Host "  to   $languageDir"
} else {
    Write-Warning "Language folder not found ($languageSrc); skipping language file deploy."
}
