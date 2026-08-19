<#
.SYNOPSIS
    Configure and build the Nz-Videomni native plugin (.aux2) and its test exe.

.DESCRIPTION
    Uses the Ninja + MSVC (vcvars) toolchain, which is the working path on this
    machine: the installed Visual Studio is v18 (2026 preview) and does NOT ship
    the "C++ CMake tools" component, so there is no VS-bundled cmake/ninja, and
    stock CMake generators do not yet target the VS 18 generator. This script
    therefore resolves cmake + ninja from (in order):
      1. Explicit env vars  NZVIDEOMNI_CMAKE / NZVIDEOMNI_NINJA (full path to the .exe)
      2. A Visual Studio-bundled copy, if present
      3. Whatever is already on PATH
      4. A per-user cache under %LOCALAPPDATA%\NzVideomni\buildtools
         (downloaded on first use - NOT a global/system install)

    MSVC itself always comes from Visual Studio via vcvars64.bat; nothing is
    installed system-wide and PATH is only modified for this process.

.PARAMETER Config
    Release (default) or Debug.

.PARAMETER RunTests
    Run the test suite after a successful build, excluding the "integration"
    doctest suite (invokes NzVideomni_tests.exe --test-suite-exclude=integration
    directly, not ctest, precisely so this works without a mock backend
    running). To also run "integration", start the mock backend
    (Nz-Videomni's run.ps1 with config.yaml's model.backend: mock) and
    run build\<preset>\NzVideomni_tests.exe by hand with no filter (or ctest).

.PARAMETER Clean
    Delete the preset's build directory before configuring.

.PARAMETER NoEmbedWebui
    Opt out of the (now default) embedded Web UI build: skips the
    `npm run build:single` step entirely and configures with
    -DNZVIDEOMNI_EMBED_WEBUI=OFF, so no npm install is required. The resulting
    NzVideomni.aux2 has no UI resource of its own; it falls back to the native
    on-disk folder-mapping path in webview_host.cpp (see that file for the
    fallback's behavior) and is not accepted by scripts\deploy.ps1, which
    requires an embedded build (see deploy.ps1's header comment). Use this
    only for native-only debugging where npm/webui aren't available or
    needed. Combining -NoEmbedWebui with -EmbedWebui is an error.

.PARAMETER EmbedWebui
    Kept for backward compatibility: an explicit, redundant way to request
    the embedded build, which is the default with neither switch passed.
    When embedding (default), the single-file Web UI (webui\ `npm run
    build:single` -> webui\dist-single\index.html) is built and CMake is
    configured with -DNZVIDEOMNI_EMBED_WEBUI=ON so that HTML is compiled into the
    DLL as an RCDATA resource. The resulting NzVideomni.aux2 serves its UI from
    memory and needs no on-disk webui\ folder. Use scripts\package.ps1
    afterwards to build the .au2pkg.zip.

.PARAMETER SkipWebuiBuild
    When embedding, skip the `npm run build:single` step and embed whatever
    webui\dist-single\index.html already exists (or the file named by
    -EmbeddedHtml). Useful when the single-file UI has been built separately.

.PARAMETER EmbeddedHtml
    When embedding, embed this single-file HTML instead of
    webui\dist-single\index.html (passed through to NZVIDEOMNI_WEBUI_SINGLE_HTML).
#>
[CmdletBinding()]
param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",
    [switch]$RunTests,
    [switch]$Clean,
    [switch]$EmbedWebui,
    [switch]$NoEmbedWebui,
    [switch]$SkipWebuiBuild,
    [string]$EmbeddedHtml
)

if ($EmbedWebui -and $NoEmbedWebui) {
    throw "-EmbedWebui and -NoEmbedWebui are mutually exclusive."
}
# Embedding is the default; -EmbedWebui is a redundant (backward-compatible)
# way to say so, and -NoEmbedWebui is the only way to opt out.
$embed = -not $NoEmbedWebui

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

# Pinned portable tool versions (used only if a local download is required).
$CMakeVersion = "3.31.6"
$NinjaVersion = "1.12.1"
$CacheDir = Join-Path $env:LOCALAPPDATA "NzVideomni\buildtools"

function Find-VsBundled([string]$relative) {
    $roots = @(
        "$env:ProgramFiles\Microsoft Visual Studio",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio"
    )
    foreach ($root in $roots) {
        if (Test-Path $root) {
            $hit = Get-ChildItem -Path $root -Recurse -Filter (Split-Path $relative -Leaf) `
                -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -like "*$relative" } |
                Select-Object -First 1
            if ($hit) { return $hit.FullName }
        }
    }
    return $null
}

function Resolve-CMake {
    if ($env:NZVIDEOMNI_CMAKE -and (Test-Path $env:NZVIDEOMNI_CMAKE)) { return $env:NZVIDEOMNI_CMAKE }
    $vs = Find-VsBundled "CMake\CMake\bin\cmake.exe"
    if ($vs) { return $vs }
    $cmd = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cached = Join-Path $CacheDir "cmake-$CMakeVersion-windows-x86_64\bin\cmake.exe"
    if (Test-Path $cached) { return $cached }
    Write-Host "cmake not found; downloading portable CMake $CMakeVersion to $CacheDir ..."
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    $zip = Join-Path $CacheDir "cmake.zip"
    $url = "https://github.com/Kitware/CMake/releases/download/v$CMakeVersion/cmake-$CMakeVersion-windows-x86_64.zip"
    Invoke-WebRequest -Uri $url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $CacheDir -Force
    Remove-Item $zip -Force
    return $cached
}

function Resolve-Ninja {
    if ($env:NZVIDEOMNI_NINJA -and (Test-Path $env:NZVIDEOMNI_NINJA)) { return $env:NZVIDEOMNI_NINJA }
    $vs = Find-VsBundled "Ninja\ninja.exe"
    if ($vs) { return $vs }
    $cmd = Get-Command ninja -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cached = Join-Path $CacheDir "ninja\ninja.exe"
    if (Test-Path $cached) { return $cached }
    Write-Host "ninja not found; downloading portable Ninja $NinjaVersion to $CacheDir ..."
    New-Item -ItemType Directory -Force -Path (Join-Path $CacheDir "ninja") | Out-Null
    $zip = Join-Path $CacheDir "ninja.zip"
    $url = "https://github.com/ninja-build/ninja/releases/download/v$NinjaVersion/ninja-win.zip"
    Invoke-WebRequest -Uri $url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath (Join-Path $CacheDir "ninja") -Force
    Remove-Item $zip -Force
    return $cached
}

function Resolve-VcVars {
    if ($env:NZVIDEOMNI_VCVARS -and (Test-Path $env:NZVIDEOMNI_VCVARS)) { return $env:NZVIDEOMNI_VCVARS }
    $hit = Find-VsBundled "VC\Auxiliary\Build\vcvars64.bat"
    if ($hit) { return $hit }
    throw "vcvars64.bat not found. Set NZVIDEOMNI_VCVARS to its full path."
}

function Import-VcVars([string]$vcvars) {
    Write-Host "Importing MSVC environment from: $vcvars"
    cmd /c "`"$vcvars`" > nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
}

# --- resolve toolchain -------------------------------------------------------
$CMake  = Resolve-CMake
$Ninja  = Resolve-Ninja
$VcVars = Resolve-VcVars
Write-Host "cmake : $CMake"
Write-Host "ninja : $Ninja"

Import-VcVars $VcVars
$env:PATH = "$(Split-Path $CMake);$(Split-Path $Ninja);$env:PATH"

# --- configure + build -------------------------------------------------------
$preset = if ($Config -eq "Debug") { "ninja-debug" } else { "ninja-release" }
$buildDir = Join-Path $RepoRoot "build\$preset"

if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "Cleaning $buildDir"
    Remove-Item -Recurse -Force $buildDir
}

# --- single-file Web UI for the embedded build (default) --------------------
$configureArgs = @("--preset", $preset)
if ($embed) {
    $webuiDir = Join-Path $RepoRoot "webui"
    if (-not $SkipWebuiBuild) {
        Write-Host "=== Web UI single-file build (npm run build:single) ==="
        $npm = Get-Command npm -ErrorAction SilentlyContinue
        if (-not $npm) { throw "npm not found on PATH; needed for the embedded Web UI build (pass -NoEmbedWebui to opt out, or -SkipWebuiBuild)." }
        Push-Location $webuiDir
        try {
            & $npm.Source "run" "build:single"
            if ($LASTEXITCODE -ne 0) { throw "npm run build:single failed ($LASTEXITCODE)" }
        } finally {
            Pop-Location
        }
    }
    if ($EmbeddedHtml) {
        $htmlPath = (Resolve-Path $EmbeddedHtml).Path
    } else {
        $htmlPath = Join-Path $webuiDir "dist-single\index.html"
    }
    if (-not (Test-Path $htmlPath)) {
        throw "Single-file Web UI not found: $htmlPath (run webui\ `npm run build:single`, or pass -EmbeddedHtml)."
    }
    Write-Host "Embedding Web UI: $htmlPath"
    $configureArgs += "-DNZVIDEOMNI_EMBED_WEBUI=ON"
    $configureArgs += "-DNZVIDEOMNI_WEBUI_SINGLE_HTML=$htmlPath"
} else {
    # Ensure an earlier embedded configure in this build dir does not linger.
    $configureArgs += "-DNZVIDEOMNI_EMBED_WEBUI=OFF"
}

Set-Location $RepoRoot
Write-Host "=== Configure ($preset) ==="
& $CMake @configureArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

Write-Host "=== Build ($preset) ==="
& $CMake --build --preset $preset
if ($LASTEXITCODE -ne 0) { throw "cmake build failed ($LASTEXITCODE)" }

$aux2 = Join-Path $buildDir "NzVideomni.aux2"
if (-not (Test-Path $aux2)) { throw "Expected artifact not found: $aux2" }
Write-Host ""
Write-Host "Build succeeded: $aux2"

if ($RunTests) {
    # Runs the exe directly (not ctest) with the "integration" suite excluded,
    # so -RunTests works without a mock backend up. See .PARAMETER RunTests
    # for how to run the full suite including "integration".
    Write-Host "=== Tests (excluding the 'integration' suite) ==="
    $testExe = Join-Path $buildDir "NzVideomni_tests.exe"
    if (-not (Test-Path $testExe)) { throw "Test executable not found: $testExe" }
    & $testExe --test-suite-exclude=integration
    if ($LASTEXITCODE -ne 0) { throw "tests failed ($LASTEXITCODE)" }
}
