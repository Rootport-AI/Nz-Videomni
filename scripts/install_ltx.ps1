<#
.SYNOPSIS
    Install the official LTX-2 stack and LTX-2.3 weights for this backend,
    keeping EVERYTHING inside the project directory (env isolation, spec 2.5).

.DESCRIPTION
    Run once after cloning this repo, when ready to swap the mock runner for the
    real model (Step 7). It:

      1. Clones https://github.com/Lightricks/LTX-2 into  vendor/LTX-2  (gitignored)
      2. Runs `uv sync --frozen` there (its OWN venv under vendor/LTX-2/.venv) and
         installs the GPU-generation-appropriate ATTENTION BACKEND
      3. Downloads LTX-2.3 weights from HuggingFace into  models/ltx-2.3  (gitignored)
      4. (optional) Downloads the gated Gemma text encoder into  models/gemma-2-2b-it
      5. Prints the exact paths to paste into config.yaml (model.* section)

    Nothing is installed into the system Python. The uv-managed interpreter is
    pinned to the project's .python directory. To fully uninstall, delete
    vendor/ and models/ .

.GPU GENERATIONS (the only GPU-specific knob)
    The official LTX-2 stack pins  torch~=2.7  from the PyTorch  cu129  (CUDA 12.9)
    wheel index. The attention backend differs by GPU microarchitecture:

      -GpuArch ada       (DEFAULT) RTX 40-series / RTX Ada / L40 / L4  (sm_89) -> xformers
      -GpuArch ampere    RTX 30-series / A100 / A6000                 (sm_80/86) -> xformers
      -GpuArch hopper    H100 / H200                                  (sm_90)    -> xformers
      -GpuArch blackwell RTX 50-series / B200 / RTX PRO 6000          (sm_100/120) -> flash-attn-4==4.0.0b9

    This machine is Ada Lovelace, so the default is correct here. Users on other
    generations only need to pass a different -GpuArch (no code edits).

    Requirements: a CUDA 12.9-capable NVIDIA driver (Windows: ~R576+), git, uv,
    ffmpeg on PATH. Ada Lovelace has FP8 (E4M3/E5M2) tensor cores, so the runtime
    `--quantization fp8-cast` path (for bf16 checkpoints) is supported.

.NOTES
    The gated Gemma encoder requires accepting the license at
    huggingface.co/google/gemma-2-2b-it and passing -HfToken (or $env:HF_TOKEN).

.EXAMPLE
    ./scripts/install_ltx.ps1                          # Ada Lovelace (this machine)
    ./scripts/install_ltx.ps1 -GpuArch blackwell       # RTX 50-series
    ./scripts/install_ltx.ps1 -WithGemma -HfToken hf_xxx
    ./scripts/install_ltx.ps1 -SkipDownload            # only clone + uv sync
#>

[CmdletBinding()]
param(
    [ValidateSet("ada", "ampere", "hopper", "blackwell")]
    [string] $GpuArch = "ada",
    [string] $LtxDir = "vendor/LTX-2",
    [string] $ModelsDir = "models/ltx-2.3",
    [string] $GemmaDir = "models/gemma-2-2b-it",
    [string] $ModelRepo = "Lightricks/LTX-2.3",
    # EXACT files to fetch (verified against the repo's file list). We pin precise
    # names because broad globs like *distilled* would also pull the dev model,
    # the distilled-1.1 model and LoRAs — each a multi-GB 22B file.
    #   - main distilled checkpoint (VAE is bundled INSIDE this file)
    #   - one spatial upscaler (note: repo spells it "upscaler"; the CLI flag is
    #     --spatial-upsampler-path)
    # Other variants (dev, distilled-1.1, LoRAs, x1.5 / temporal upscalers) are
    # optional — add them to -Include if needed.
    [string[]] $Include = @(
        "ltx-2.3-22b-distilled.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    ),
    [switch] $WithGemma,
    [string] $HfToken = $env:HF_TOKEN,
    [switch] $SkipClone,
    [switch] $SkipSync,
    [switch] $SkipDownload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

# --- Environment isolation: keep uv-managed Python inside the project ---------
$env:UV_PYTHON_INSTALL_DIR = "$ProjectRoot\.python"

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Require-Cmd($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Required command '$name' not found on PATH. Please install it first."
    }
}

Write-Step "Checking prerequisites"
Require-Cmd git
Require-Cmd uv
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg not found on PATH. The backend needs it to encode MP4."
}
Write-Host "git : $((Get-Command git).Source)"
Write-Host "uv  : $((Get-Command uv).Source)"
Write-Host "GPU target architecture: $GpuArch"

# Informational GPU detection (does not block).
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($gpuName) {
        Write-Host "Detected GPU: $gpuName"
        if ($GpuArch -eq "ada" -and $gpuName -match "RTX 50|B200|RTX PRO 6000") {
            Write-Warning "GPU looks Blackwell but -GpuArch is 'ada'. Consider -GpuArch blackwell."
        }
    }
} else {
    Write-Warning "nvidia-smi not found. Cannot verify GPU; proceeding with -GpuArch $GpuArch."
}

# --- 1) Clone official LTX-2 --------------------------------------------------
if (-not $SkipClone) {
    Write-Step "Cloning LTX-2 into $LtxDir"
    if (Test-Path "$LtxDir/.git") {
        Write-Host "Already cloned. Pulling latest..."
        git -C $LtxDir pull --ff-only
    } else {
        $parent = Split-Path $LtxDir
        if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
        git clone https://github.com/Lightricks/LTX-2.git $LtxDir
    }
}

# --- 2) uv sync + attention backend (per GPU generation) ----------------------
if (-not $SkipSync) {
    Push-Location $LtxDir
    try {
        if ($GpuArch -eq "blackwell") {
            Write-Step "uv sync --frozen  (Blackwell: + flash-attn-4)"
            uv sync --frozen
            Write-Host "Installing flash-attn-4==4.0.0b9 (Blackwell attention backend)..."
            uv pip install 'flash-attn-4==4.0.0b9'
        } else {
            # ada / ampere / hopper -> xformers extra (from the pytorch cu129 index)
            Write-Step "uv sync --frozen --extra xformers  ($GpuArch attention backend)"
            uv sync --frozen --extra xformers
        }
    } finally {
        Pop-Location
    }
}

# --- 3) Download LTX-2.3 weights ---------------------------------------------
if (-not $SkipDownload) {
    Write-Step "Downloading $ModelRepo -> $ModelsDir"
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

    $includeArgs = @()
    foreach ($pat in $Include) { $includeArgs += @("--include", $pat) }

    # Use LTX-2's venv (has huggingface_hub) via `uv run --directory`.
    & uv run --directory $LtxDir hf download $ModelRepo `
        @includeArgs `
        --local-dir (Resolve-Path $ModelsDir)
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "hf download failed. Check the include patterns against the repo's file list:"
        Write-Warning "  https://huggingface.co/$ModelRepo/tree/main"
        throw "Model download failed."
    }

    # --- 4) Gemma text encoder (gated) ---------------------------------------
    if ($WithGemma) {
        Write-Step "Downloading gated google/gemma-2-2b-it -> $GemmaDir"
        if (-not $HfToken) {
            throw "Gemma is gated. Accept the license at https://huggingface.co/google/gemma-2-2b-it then pass -HfToken or set `$env:HF_TOKEN."
        }
        New-Item -ItemType Directory -Force -Path $GemmaDir | Out-Null
        & uv run --directory $LtxDir hf download google/gemma-2-2b-it `
            --local-dir (Resolve-Path $GemmaDir) `
            --token $HfToken
        if ($LASTEXITCODE -ne 0) { throw "Gemma download failed." }
    } else {
        Write-Host "`n(Skipping Gemma. Re-run with -WithGemma -HfToken <token> when ready.)" -ForegroundColor Yellow
    }
}

# --- 5) Resolve and report paths for config.yaml -----------------------------
Write-Step "Detected paths (paste into config.yaml -> model:)"

# Match the main distilled checkpoint but NOT the lora variants.
$ckpt = Get-ChildItem -Path $ModelsDir -Recurse -Filter "ltx-2.3-22b-distilled*.safetensors" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notmatch "lora" } | Select-Object -First 1
$ckpt = if ($ckpt) { $ckpt.FullName.Replace($ProjectRoot, ".").Replace("\", "/") } else { $null }
# Repo spells it "upscaler"; some forks use "upsampler" — match either.
$ups  = Get-ChildItem -Path $ModelsDir -Recurse -Filter "*spatial-up*.safetensors" -ErrorAction SilentlyContinue | Select-Object -First 1
$ups  = if ($ups) { $ups.FullName.Replace($ProjectRoot, ".").Replace("\", "/") } else { $null }
$gem  = if (Test-Path $GemmaDir) { "./" + ((Resolve-Path $GemmaDir).Path.Replace($ProjectRoot + "\", "").Replace("\", "/")) } else { $null }

$report = @"
  ltx_repo_dir: "./$($LtxDir -replace '\\','/')"
  checkpoint_path: $(if ($ckpt) { '"' + $ckpt + '"' } else { 'null   # not found - set manually' })
  spatial_upsampler_path: $(if ($ups) { '"' + $ups + '"' } else { 'null   # not found - set manually' })
  gemma_root: $(if ($gem) { '"' + $gem + '"' } else { 'null   # run with -WithGemma' })
  quantization: "fp8-cast"
"@

Write-Host $report -ForegroundColor Green
$report | Out-File -FilePath "models/INSTALLED_PATHS.txt" -Encoding utf8
Write-Host "`nSaved to models/INSTALLED_PATHS.txt"
Write-Host "Next: update config.yaml with the paths above, then implement the real" -ForegroundColor Cyan
Write-Host "pipeline calls in services/ltx_runner.py (see its docstring / README Step 7)." -ForegroundColor Cyan
