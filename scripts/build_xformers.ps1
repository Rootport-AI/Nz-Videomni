<#
.SYNOPSIS
    MANUAL TOOL, FOR FUTURE PERFORMANCE EXPERIMENTS ONLY. Not part of setup.
    Builds an xformers wheel from source for THIS project's pinned stack
    (the engine venv's cu128 torch pinned in engine/venv-engine.freeze.txt /
    CUDA 12.8 / Windows / Python 3.12, Ada Lovelace sm_89).

.DESCRIPTION
    NOTHING RUNS THIS FOR YOU. install_ltx.ps1 does not call it and does not look
    in wheels/; the attention backends the engine venvs carry are PyTorch SDPA
    (the default) and SageAttention (chosen per job via attention_backend). The
    project's own backend code does not import xformers (the vendored
    Video-Depth-Anything DINOv2 layers only try-import it, and the depth
    preprocessing path does not use it). Installing the wheel is NOT inert,
    though: the 2.3 engine's ltx-core in .venv-engine picks xformers for its
    default attention whenever xformers is importable, so after -Install the
    2.3 engine's sdpa jobs run on xformers. Keep this script for the day you
    want to measure whether xformers is worth that switch.

    The cu128 xformers wheel is Linux-only, so on Windows we compile it ourselves.

    Flow:
      1. Check that torch imports in the engine venv and print its version and
         CUDA build (the build target; the versions are shown, not compared).
      2. Set up the MSVC build environment (vcvars64) and CUDA 12.8 toolchain.
      3. Clone xformers at the commit LTX pins, init the CUTLASS submodule.
      4. `pip wheel --no-build-isolation` against the engine venv's torch.
      5. Output xformers-*.whl into wheels/ and (only with -Install) install it.

    Everything stays inside the project. MSVC and the CUDA Toolkit are system
    build tools (not Python) -- they do not violate the Python-isolation rule.

.NOTES
    Prerequisites (install manually first):
    - Visual Studio 2022 Build Tools with the C++ workload (MSVC v143 + Win SDK)
    - CUDA Toolkit 12.8  (nvcc; must match torch's cu128)
    - .venv-engine already created (cu128 torch from engine/venv-engine.freeze.txt)
      (run:  ./scripts/install_ltx.ps1 -SkipModels  first)

.EXAMPLE
    ./scripts/build_xformers.ps1
    ./scripts/build_xformers.ps1 -Install          # also install into .venv-engine
    ./scripts/build_xformers.ps1 -MaxJobs 2        # limit RAM use during compile
#>

[CmdletBinding()]
param(
    # Engine venv that holds the torch build target (the cu128 torch pinned in
    # engine/venv-engine.freeze.txt).
    [string] $EngineDir = ".venv-engine",
    # Commit LTX-2's lock pins: xformers 0.0.33+5d4b92a5.d20251029
    [string] $XformersRef = "5d4b92a5",
    # Compute capability to compile cubins for. The wheel is good ONLY for the
    # generation you name here (default 8.9 = Ada Lovelace / RTX 40). See the
    # closing notes at the bottom of this script before reusing a wheel elsewhere.
    [string] $Arch = "8.9",
    [string] $OutDir = "wheels",
    # Match the cu128 torch stack. CUDA Toolkit 12.8 (nvcc) must match torch's cu128.
    [string] $CudaVersion = "12.8",
    # MSVC toolset to select via vcvars. CUDA 12.8 (the -CudaVersion default)
    # does NOT support the VS 2026 default MSVC (v14.50+); it supports VS 2019/2022
    # (v14.2x-v14.4x). On VS 2026 add the "MSVC v143 - VS 2022 C++ build tools
    # (v14.44)" component and keep this at 14.44 so nvcc accepts the compiler.
    # Empty = use the VS default.
    [string] $VcvarsVer = "14.44",
    [int]    $MaxJobs = 0,                   # 0 = let ninja decide
    [switch] $Install                        # install the built wheel into the engine venv
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

$env:UV_PYTHON_INSTALL_DIR = "$ProjectRoot\.python"
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = "$ProjectRoot\.uv_cache" }

function Write-Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }

# --- 1) Check build target: torch imports in the engine venv (printed only) ---
Write-Step "Checking engine venv (torch build target)"
$venvPy = "$ProjectRoot\$EngineDir\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    throw "Engine venv not found: $venvPy`nRun ./scripts/install_ltx.ps1 -SkipModels first (it installs torch 2.9.1+cu128)."
}
$torchInfo = & $venvPy -c "import torch;print(torch.__version__, torch.version.cuda)" 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "torch not importable in the engine venv. Run ./scripts/install_ltx.ps1 -SkipModels first.`n$torchInfo"
}
Write-Host "Engine venv torch: $torchInfo"

# --- 2a) CUDA toolchain ------------------------------------------------------
# Prefer the driver/toolkit-provided CUDA_PATH env vars (set by the CUDA Toolkit
# installer); fall back to the conventional Program Files path. Must match the
# cu128 torch stack (CUDA 12.8).
Write-Step "Selecting CUDA Toolkit $CudaVersion"
$cudaRoot = $null
if ($env:CUDA_PATH_V12_8 -and (Test-Path "$env:CUDA_PATH_V12_8\bin\nvcc.exe")) {
    $cudaRoot = $env:CUDA_PATH_V12_8
} elseif ($env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) {
    $cudaRoot = $env:CUDA_PATH
} else {
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$CudaVersion"
}
if (-not (Test-Path "$cudaRoot\bin\nvcc.exe")) {
    throw "CUDA Toolkit $CudaVersion not found (checked `$env:CUDA_PATH_V12_8, `$env:CUDA_PATH, and $cudaRoot). Install CUDA Toolkit 12.8 -- must match torch's cu128."
}
$env:CUDA_PATH = $cudaRoot
$env:CUDA_HOME = $cudaRoot
$env:PATH = "$cudaRoot\bin;$env:PATH"
Write-Host (& "$cudaRoot\bin\nvcc.exe" --version | Select-String "release")

# --- 2b) MSVC build environment (vcvars64) -----------------------------------
Write-Step "Setting up MSVC (Visual Studio 2022 C++)"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere not found -- install VS 2022 Build Tools with the C++ workload." }
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw "MSVC C++ tools not found -- install the 'Desktop development with C++' workload." }
$vcvars = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }
# Select an nvcc-compatible toolset (v143 on VS 2026). Import its env into PS.
$verArg = if ($VcvarsVer) { "-vcvars_ver=$VcvarsVer" } else { "" }
Write-Host "vcvars64 $verArg"
cmd /c "`"$vcvars`" $verArg >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $cl) {
    throw "cl.exe not on PATH after vcvars. Is the C++ workload installed? If on VS 2026, add the 'MSVC v143 - VS 2022 C++ build tools (v14.44)' component."
}
$clVer = (& cl.exe 2>&1 | Select-Object -First 1)
Write-Host "MSVC: $($cl.Source)"
Write-Host "cl: $clVer"

# --- 3) Clone xformers at the pinned commit + submodules ---------------------
Write-Step "Fetching xformers @ $XformersRef"
$buildDir = "$ProjectRoot\vendor\xformers-build"
if (-not (Test-Path "$buildDir\.git")) {
    git clone https://github.com/facebookresearch/xformers.git $buildDir
}
Push-Location $buildDir
try {
    git config --local core.longpaths true
    git fetch --all --tags
    git checkout $XformersRef
    git submodule update --init --recursive   # CUTLASS etc.

    # --- 4) Build deps + compile the wheel -----------------------------------
    Write-Step "Installing build deps into the engine venv"
    uv pip install --python $venvPy ninja setuptools wheel

    Write-Step "Building xformers wheel (this can take 30-90 min)"
    $env:TORCH_CUDA_ARCH_LIST = $Arch          # build only Ada -> faster
    $env:DISTUTILS_USE_SDK = "1"               # required for MSVC builds
    $env:FORCE_CUDA = "1"
    $env:XFORMERS_BUILD_TYPE = "Release"
    if ($MaxJobs -gt 0) { $env:MAX_JOBS = "$MaxJobs" }

    $outAbs = "$ProjectRoot\$OutDir"
    New-Item -ItemType Directory -Force -Path $outAbs | Out-Null
    & $venvPy -m pip wheel . --no-build-isolation --no-deps -w $outAbs
    if ($LASTEXITCODE -ne 0) { throw "xformers build failed." }
}
finally {
    Pop-Location
}

# --- 5) Report (and optionally install) --------------------------------------
$wheel = Get-ChildItem "$ProjectRoot\$OutDir" -Filter "xformers-*.whl" |
    Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $wheel) { throw "No xformers wheel was produced in $OutDir." }

Write-Step "Built wheel"
Write-Host $wheel.FullName -ForegroundColor Green
Write-Host "Size: $([math]::Round($wheel.Length/1MB,1)) MB"

if ($Install) {
    Write-Step "Installing the wheel into the engine venv"
    uv pip install --python $venvPy $wheel.FullName
    & $venvPy -c "import xformers, xformers.ops; print('xformers', xformers.__version__, 'OK')"
}

Write-Host "`nNOTE: this wheel is NOT installed automatically." -ForegroundColor Yellow
Write-Host "  install_ltx.ps1 never looks in $OutDir\ -- the default attention backend"
Write-Host "  is PyTorch SDPA (sageattention can be selected per job). To use the wheel,"
Write-Host "  install it yourself into the engine venv:"
Write-Host "    uv pip install --python `"$venvPy`" `"$($wheel.FullName)`"" -ForegroundColor Cyan
Write-Host "  (or re-run this script with -Install, which does exactly that)."
Write-Host ""
Write-Host "  ARCH LOCK: this wheel only runs on the generation you built it for" -ForegroundColor Yellow
Write-Host "  (-Arch $Arch). CUDA cubin compatibility is one-way: a cubin built for a"
Write-Host "  LOWER minor within the SAME major runs on a HIGHER-minor device, never the"
Write-Host "  reverse. So the default -Arch 8.9 (Ada / RTX 40) will NOT run on Ampere"
Write-Host "  (sm_86) -- for Ampere rebuild with -Arch 8.6. The install itself succeeds"
Write-Host "  either way; a mismatch only shows up at run time as"
Write-Host "  'no kernel image is available for execution on the device'."
Write-Host ""
Write-Host "  Also: the project's own code never imports xformers, but the LTX 2.3 engine's"
Write-Host "  upstream ltx-core picks it up when it is importable and uses it for the"
Write-Host "  default attention path. The LTX 2.5 engine does not reference it."
