<#
.SYNOPSIS
    Full, idempotent clean installer for the LTX-2.3 video-gen backend.
    Brings a fresh Windows box (git + uv + ffmpeg + NVIDIA driver already present)
    to a runnable state, and is SAFE TO RE-RUN on an already-installed machine
    (every step guards on existing artifacts and prints SKIP).

    Everything stays inside the project directory (env isolation, spec 2.5):
    uv-managed Python, uv cache, and the HuggingFace cache all live under the
    project root; we never touch the system Python and never set persistent
    system env vars (all env vars below are process-scoped only).

.DESCRIPTION
    Steps (each skip-guarded / idempotent):
      1. Prereqs (git, uv; ffmpeg warn) + process-scoped env isolation.
      2. uv-managed Python 3.12.
      3. App venv  .venv        (torch-FREE; `uv sync` of root pyproject.toml).
      4. Engine venv .venv-engine (torch 2.9.1+cu128 stack). DEFAULT = deterministic
         FREEZE path (reproduces the VALIDATED stack all verification ran on).
         -ResolveLatest opts into a fresh resolve (UNVALIDATED newer torch).
      5. Attention backend: SDPA is the backend for ALL archs (xformers/flash-attn
         are NOT auto-installed, per the NEXT_SESSION_HANDOFF install-compat policy).
         Optional on ada/ampere/hopper: a prebuilt xformers wheel from wheels/ is
         used only if present (build via scripts/build_xformers.ps1). Blackwell = SDPA.
      6. Model downloads (~28GB, 2 guarded items) via .venv-engine's hf.exe,
         pulled from two PUBLIC, NON-GATED repos. No HuggingFace account,
         login or token is required at any point.
      7. Verification table (PASS/MISSING) + regenerate models/INSTALLED_PATHS.txt.

    The GGUF + component-file recipe is the ONLY supported real path. It never
    opens the old 46GB monolith (ltx-2.3-22b-distilled-1.1.safetensors) or the
    22.7GB QAT Gemma dir -- both were physically deleted; this installer never
    downloads them. The monolith path survives only in config.yaml as a
    reference-only payload field (see config.py ModelConfig.checkpoint_path).

.GPU ARCHITECTURES (the only GPU-specific knob)
    -GpuArch is AUTO-DETECTED from nvidia-smi unless passed explicitly. ALL archs
    use PyTorch SDPA; xformers/flash-attn are NOT auto-installed (see step 5):
      ada       RTX 40-series / RTX Ada / L40 / L4   (sm_89)      -> SDPA (opt. xformers wheel)
      ampere    RTX 30-series / A100 / A6000         (sm_80/86)   -> SDPA (opt. xformers wheel)
      hopper    H100 / H200                          (sm_90)      -> SDPA (opt. xformers wheel)
      blackwell RTX 50-series / B200 / RTX PRO 6000  (sm_100/120) -> SDPA (R570+ driver)
    An explicit -GpuArch always wins over detection. If detection fails, the
    script warns and requires an explicit -GpuArch.

.EXAMPLE
    ./scripts/install_ltx.ps1                       # auto-detect GPU, full install
    ./scripts/install_ltx.ps1 -GpuArch blackwell    # force Blackwell attention
    ./scripts/install_ltx.ps1 -ResolveLatest        # fresh (unvalidated) engine resolve
    ./scripts/install_ltx.ps1 -SkipModels           # venvs only, no downloads
    ./scripts/install_ltx.ps1 -RunSmoke             # + mock GPU-free smoke test
#>

[CmdletBinding()]
param(
    # Auto-detect from nvidia-smi when not passed. Explicit value wins.
    [ValidateSet("ada", "ampere", "hopper", "blackwell")]
    [string] $GpuArch,
    # Opt into a fresh resolve of the engine venv (UNVALIDATED newer torch ~2.11).
    # Default (unset) uses the deterministic freeze path.
    [switch] $ResolveLatest,
    # After install, run the mock, GPU-free smoke test.
    [switch] $RunSmoke,
    # Clone upstream Lightricks/LTX-2 into vendor/LTX-2 as REFERENCE ONLY (no venv
    # is ever built there). Off by default -- the old clone+sync-in-fork flow is gone.
    [switch] $CloneUpstreamReference,
    # Convenience skips.
    [switch] $SkipModels,
    [switch] $SkipVenv
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

# ----------------------------------------------------------------------------
# Env isolation (process-scoped only; mirrors run.ps1 L22-28). Keep the
# uv-managed interpreter, uv download cache and HuggingFace cache in-project.
# ----------------------------------------------------------------------------
$env:UV_PYTHON_INSTALL_DIR = "$ProjectRoot\.python"
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = "$ProjectRoot\.uv_cache" }
if (-not $env:HF_HOME) { $env:HF_HOME = "$ProjectRoot\hf_home" }
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True" }

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Skip($msg) { Write-Host "  SKIP  $msg" -ForegroundColor DarkGray }
function Write-Do($msg) { Write-Host "  ..    $msg" -ForegroundColor Yellow }
function Write-Ok($msg) { Write-Host "  OK    $msg" -ForegroundColor Green }

function Require-Cmd($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Required command '$name' not found on PATH. Please install it first."
    }
}

# Human-readable size for reporting.
function Format-Size([long] $bytes) {
    if ($bytes -ge 1GB) { return "{0:N2} GB" -f ($bytes / 1GB) }
    if ($bytes -ge 1MB) { return "{0:N1} MB" -f ($bytes / 1MB) }
    if ($bytes -ge 1KB) { return "{0:N0} KB" -f ($bytes / 1KB) }
    return "$bytes B"
}

# Total bytes of a file, or of every file under a directory (0 if missing).
function Get-PathSize([string] $absPath) {
    if (-not (Test-Path $absPath)) { return [long] 0 }
    $item = Get-Item $absPath
    if ($item.PSIsContainer) {
        $sum = (Get-ChildItem $absPath -Recurse -File -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
        if ($null -eq $sum) { return [long] 0 }
        return [long] $sum
    }
    return [long] $item.Length
}

# ----------------------------------------------------------------------------
# 1) Prerequisites + GPU-arch resolution
# ----------------------------------------------------------------------------
Write-Step "Checking prerequisites"
Require-Cmd git
Require-Cmd uv
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg not found on PATH. The backend needs it to encode MP4."
}
Write-Host "git    : $((Get-Command git).Source)"
Write-Host "uv     : $((Get-Command uv).Source)"
Write-Host "root   : $ProjectRoot"
Write-Host "HF_HOME: $env:HF_HOME"

# Auto-detect the GPU architecture unless the caller passed -GpuArch. An explicit
# param always wins. Detection failure => warn + require explicit -GpuArch.
if (-not $GpuArch) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    $gpuName = $null
    if ($smi) {
        # Capture fully BEFORE Select-Object: piping a native exe straight into
        # `Select-Object -First 1` stops the pipeline early and leaves
        # $LASTEXITCODE = -1 (broken pipe), which would otherwise leak out as the
        # script's own exit code on the success path.
        $smiOut = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null)
        $gpuName = ($smiOut | Select-Object -First 1)
    }
    if ($gpuName) {
        Write-Host "Detected GPU: $gpuName"
        switch -Regex ($gpuName) {
            "RTX 50|B200|RTX PRO 6000|Blackwell" { $GpuArch = "blackwell"; break }
            "H100|H200|Hopper"                   { $GpuArch = "hopper";    break }
            "RTX 30|A100|A6000|A40|Ampere"       { $GpuArch = "ampere";    break }
            "RTX 40|Ada|L40|L4"                  { $GpuArch = "ada";       break }
            default { $GpuArch = $null }
        }
    }
    if (-not $GpuArch) {
        Write-Warning "Could not auto-detect GPU architecture (nvidia-smi missing or name unmatched)."
        throw "Pass an explicit -GpuArch {ada|ampere|hopper|blackwell}."
    }
    Write-Host "Auto-selected GpuArch: $GpuArch" -ForegroundColor Green
} else {
    Write-Host "GpuArch (explicit): $GpuArch"
}

# ----------------------------------------------------------------------------
# 2) uv-managed Python 3.12  (skip if an in-project cpython-3.12 is already present)
# ----------------------------------------------------------------------------
Write-Step "Python 3.12 (uv-managed, in-project)"
$py312 = Get-ChildItem "$ProjectRoot\.python" -Filter "cpython-3.12*" -Directory -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($py312) {
    Write-Skip "uv Python 3.12 already installed ($($py312.Name))"
} else {
    Write-Do "uv python install 3.12"
    uv python install 3.12
    if ($LASTEXITCODE -ne 0) { throw "uv python install 3.12 failed." }
}

# ----------------------------------------------------------------------------
# 3) App venv .venv  (torch-FREE; plain `uv sync` of root pyproject.toml)
# ----------------------------------------------------------------------------
$appPy = "$ProjectRoot\.venv\Scripts\python.exe"
if ($SkipVenv) {
    Write-Step "App venv .venv"
    Write-Skip "-SkipVenv given"
} else {
    Write-Step "App venv .venv (torch-free)"
    if (Test-Path $appPy) {
        Write-Skip ".venv already exists ($appPy)"
    } else {
        Write-Do "uv venv --python 3.12 .venv"
        uv venv --python 3.12 .venv
        if ($LASTEXITCODE -ne 0) { throw "uv venv .venv failed." }
    }
    # `uv sync` is cheap+idempotent, so run it every time to reconcile the app deps
    # with the root pyproject.toml (this is the torch-free FastAPI/Gradio side).
    Write-Do "uv sync  (root pyproject.toml, app deps)"
    uv sync
    if ($LASTEXITCODE -ne 0) { throw "uv sync (app venv) failed." }
    Write-Ok ".venv ready"
}

# ----------------------------------------------------------------------------
# 4) Engine venv .venv-engine  (torch cu128 stack)
#
#    There is NO committed uv.lock -> we must NEVER use `uv sync --frozen`.
#    DEFAULT = deterministic FREEZE path: reproduces the VALIDATED torch
#    2.9.1+cu128 stack (engine/venv-engine.freeze.txt) that every verification
#    run in this project used. -ResolveLatest opts into a fresh resolve from
#    engine/engine-venv-pyproject.toml (yields an UNVALIDATED newer torch).
#
#    hf.exe lives in THIS venv, so it must be created BEFORE the model downloads.
# ----------------------------------------------------------------------------
$enginePy = "$ProjectRoot\.venv-engine\Scripts\python.exe"
$freezeSrc = "$ProjectRoot\engine\venv-engine.freeze.txt"
$enginePyprojectDir = "$ProjectRoot\engine"

if ($SkipVenv) {
    Write-Step "Engine venv .venv-engine"
    Write-Skip "-SkipVenv given"
} elseif (Test-Path $enginePy) {
    Write-Step "Engine venv .venv-engine"
    Write-Skip ".venv-engine already exists ($enginePy)"
} else {
    Write-Step "Engine venv .venv-engine  (torch cu128 stack)"
    Write-Do "uv venv --python 3.12 .venv-engine"
    uv venv --python 3.12 .venv-engine
    if ($LASTEXITCODE -ne 0) { throw "uv venv .venv-engine failed." }

    if ($ResolveLatest) {
        # --------------------------------------------------------------------
        # Opt-in fresh resolve. --index-strategy unsafe-best-match is REQUIRED:
        # the pyproject's torchaudio source lacks a platform marker, and without
        # unsafe-best-match uv pulls a CPU-only torchaudio (known bug). This path
        # is UNVALIDATED (newer torch ~2.11 vs the verified 2.9.1).
        # --------------------------------------------------------------------
        Write-Warning "-ResolveLatest: resolving the engine venv fresh from engine-venv-pyproject.toml."
        Write-Warning "This yields an UNVALIDATED newer torch (~2.11). All project verification ran on"
        Write-Warning "the frozen 2.9.1+cu128 stack (the default). Use only if you accept re-validating."
        Write-Do "uv pip install --index-strategy unsafe-best-match <engine pyproject dir>"
        uv pip install --python $enginePy --index-strategy unsafe-best-match $enginePyprojectDir
        if ($LASTEXITCODE -ne 0) { throw "engine venv resolve (--ResolveLatest) failed." }
    } else {
        # --------------------------------------------------------------------
        # DEFAULT deterministic freeze path.
        #  (a) install the 3 git packages at their EXACT pinned revs. The freeze
        #      file lists them only as bare `name==version` (their real source
        #      URLs live in the freeze header comments), so a plain `-r freeze`
        #      cannot fetch them -- we install them by URL here first.
        #  (b) install the remaining pinned wheels from a TEMP copy of the freeze
        #      that has the 3 git lines removed (they're already installed above),
        #      from the cu128 index.
        # --------------------------------------------------------------------
        if (-not (Test-Path $freezeSrc)) { throw "Engine freeze file not found: $freezeSrc" }

        Write-Do "install 3 git packages at pinned revs (diffusers / ltx-core / ltx-pipelines)"
        uv pip install --python $enginePy `
            "diffusers @ git+https://github.com/huggingface/diffusers.git@01de02e8b4f2cc91df4f3e91cb6535ebcbeb490c" `
            "ltx-core @ git+https://github.com/Lightricks/LTX-2.git@00dc53d3f81c405932f9f16d9c57557de411e702#subdirectory=packages/ltx-core" `
            "ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git@00dc53d3f81c405932f9f16d9c57557de411e702#subdirectory=packages/ltx-pipelines"
        if ($LASTEXITCODE -ne 0) { throw "engine git-package install failed." }

        # TEMP freeze without the 3 git lines (they're installed above; leaving
        # them as bare name==version would let uv fetch PyPI/other builds).
        $tmpFreeze = Join-Path ([System.IO.Path]::GetTempPath()) ("venv-engine.freeze.nogit.{0}.txt" -f ([guid]::NewGuid().ToString("N")))
        try {
            Get-Content $freezeSrc |
                Where-Object { $_ -notmatch '^(diffusers|ltx-core|ltx-pipelines)==' } |
                Set-Content -Path $tmpFreeze -Encoding utf8

            Write-Do "install pinned wheels from freeze (cu128 index)"
            uv pip install --python $enginePy `
                --index https://download.pytorch.org/whl/cu128 `
                --index-strategy unsafe-best-match `
                -r $tmpFreeze
            if ($LASTEXITCODE -ne 0) { throw "engine freeze install failed." }
        } finally {
            if (Test-Path $tmpFreeze) { Remove-Item $tmpFreeze -Force -ErrorAction SilentlyContinue }
        }
    }
    Write-Ok ".venv-engine ready"
}

# hf.exe (used by the model downloads below) must exist in the engine venv.
$hfExe = "$ProjectRoot\.venv-engine\Scripts\hf.exe"

# ----------------------------------------------------------------------------
# 5) Attention backend (per GPU arch)
# ----------------------------------------------------------------------------
Write-Step "Attention backend ($GpuArch)"
# POLICY (NEXT_SESSION_HANDOFF install-compat memo): SDPA is the attention backend
# for this stack. xformers/flash-attn/sageattention are intentionally NOT
# auto-installed -- SDPA is what's used, and on Blackwell adding flash-attn can jam
# it. xformers stays an OPTIONAL manual speedup on ada/ampere/hopper: the installer
# picks up a prebuilt wheel from wheels/ ONLY if you built one via
# scripts/build_xformers.ps1. Absent (the default) => SDPA.
if ($SkipVenv) {
    Write-Skip "-SkipVenv given"
} elseif ($GpuArch -eq "blackwell") {
    # Blackwell (sm_100/120): SDPA only -- do NOT add flash-attn (see policy above).
    # Requires an R570+ driver. Nothing to install.
    Write-Ok "SDPA backend (Blackwell: no xformers/flash-attn added, per install-compat policy)"
} else {
    # ada / ampere / hopper: SDPA by default. Install an OPTIONAL prebuilt xformers
    # wheel from wheels/ ONLY if present (built by build_xformers.ps1); never required.
    $wheel = Get-ChildItem "$ProjectRoot\wheels" -Filter "xformers-*.whl" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if ($wheel) {
        Write-Do "uv pip install $($wheel.Name)  (optional xformers speedup)"
        uv pip install --python $enginePy $wheel.FullName
        if ($LASTEXITCODE -ne 0) { throw "xformers wheel install failed." }
        Write-Ok "xformers installed ($($wheel.Name))"
    } else {
        Write-Ok "SDPA backend (default). Optional speedup: .\scripts\build_xformers.ps1 -Install"
    }
}

# ----------------------------------------------------------------------------
# Optional: clone upstream LTX-2 as REFERENCE ONLY (no venv built there).
# ----------------------------------------------------------------------------
if ($CloneUpstreamReference) {
    Write-Step "Cloning upstream Lightricks/LTX-2 -> vendor/LTX-2 (reference only)"
    $vendorDir = "$ProjectRoot\vendor\LTX-2"
    if (Test-Path "$vendorDir\.git") {
        Write-Skip "vendor/LTX-2 already cloned (not touching it)"
    } else {
        New-Item -ItemType Directory -Force -Path "$ProjectRoot\vendor" | Out-Null
        Write-Do "git clone https://github.com/Lightricks/LTX-2.git vendor/LTX-2"
        git clone https://github.com/Lightricks/LTX-2.git $vendorDir
        if ($LASTEXITCODE -ne 0) { throw "vendor/LTX-2 clone failed." }
    }
    Write-Host "  (reference only: config.model.ltx_repo_dir points here; no venv built)"
}

# ----------------------------------------------------------------------------
# 6) Model downloads (~28GB) via the engine venv's hf.exe.
#    Everything comes from TWO self-hosted repos that are PUBLIC and NON-GATED,
#    so no HuggingFace account, login or token is involved anywhere:
#      Rootport/Nz-LTX23-weights -> ltx-2.3/, ltx-2.3-components/, ltx-2.3-gguf/
#      Rootport/Nz-Gemma3-12B    -> gemma-3-12b-it-gguf/, gemma-3-12b-it-tokenizer/
#    Both repos mirror this project's models/ layout 1:1, so each one expands
#    straight into models/ with no post-processing (no flattening, no renames).
#    Each item guards on a minimum on-disk size -> SKIP; else download.
# ----------------------------------------------------------------------------

# One download item. `LocalDir` is relative-to-root and is where the repo expands
# (both repos below expand into models/, since their internal layout already
# matches ours). `Include` is one or more glob patterns passed under a SINGLE
# --include flag.
#
#   NOTE (argparse nargs gotcha, carried over from the old script): `hf download
#   --include` is nargs="*". Repeating the flag (--include A --include B) makes
#   argparse keep only the LAST group and silently drop earlier files. So we build
#   ONE "--include" followed by all patterns.
#
#   NOTE (glob semantics, verified live against both repos): --include matches with
#   Python fnmatch against the repo-relative path, and `*` DOES cross '/'. So
#   "ltx-2.3-components/*" reaches the nested vae/ and text_encoders/ files two
#   levels down. Just as importantly, the repo-root card files (LICENSE /
#   NOTICE.md / README.md / .gitattributes) match NO "<dir>/*" pattern -- which is
#   what stops the two repos' identically-named cards from landing in models/ and
#   overwriting each other.
#
#   NOTE (why CheckDir is separate from LocalDir): both calls below pass
#   LocalDir = "models", so sizing the guard on LocalDir would see the ~23GB LTX
#   download and then wrongly SKIP the Gemma one. CheckDir instead names the
#   subdirectories that THIS repo expands into, and MinBytes is compared against
#   their combined size. Caveat, deliberately accepted: models/ltx-2.3-gguf/ also
#   holds any extra self-converted transformer GGUFs the user dropped in (10Eros /
#   Sulphur, ~16.5GB each), which inflates the LTX total and makes that guard more
#   lenient than the numbers suggest. The guard only exists to catch a missing or
#   truncated download; the per-file step 7 verification table below is the real
#   correctness gate.
function Invoke-ModelDownload {
    param(
        [Parameter(Mandatory)] [string]   $Name,
        [Parameter(Mandatory)] [string]   $Repo,
        [Parameter(Mandatory)] [string[]] $Include,    # glob(s), one --include
        [Parameter(Mandatory)] [string]   $LocalDir,   # relative to root
        [Parameter(Mandatory)] [string[]] $CheckDir,   # relative to root; sizes summed
        [Parameter(Mandatory)] [long]     $MinBytes
    )
    $absLocal = Join-Path $ProjectRoot $LocalDir
    $have = [long] 0
    foreach ($d in $CheckDir) { $have += Get-PathSize (Join-Path $ProjectRoot $d) }
    if ($have -ge $MinBytes) {
        Write-Skip "$Name  ($(Format-Size $have) already present in $($CheckDir -join ', '))"
        return
    }
    if (-not (Test-Path $hfExe)) {
        throw "hf.exe not found at $hfExe. The engine venv must be created first (do not pass -SkipVenv)."
    }
    New-Item -ItemType Directory -Force -Path $absLocal | Out-Null

    # Single --include with ALL patterns (see nargs note above).
    $argv = @("download", $Repo)
    $argv += @("--include") + $Include
    $argv += @("--local-dir", $absLocal)

    Write-Do "$Name  download $Repo"
    & $hfExe @argv
    if ($LASTEXITCODE -ne 0) {
        throw "Download of '$Name' failed. The repo is public and needs no token, so check your network first, then the include globs against https://huggingface.co/$Repo/tree/main"
    }
    $now = [long] 0
    foreach ($d in $CheckDir) { $now += Get-PathSize (Join-Path $ProjectRoot $d) }
    if ($now -lt $MinBytes) {
        throw "'$Name' downloaded but is smaller than expected ($(Format-Size $now) < $(Format-Size $MinBytes)). Check the include globs."
    }
    Write-Ok "$Name  ($(Format-Size $now))"
}

if ($SkipModels) {
    Write-Step "Model downloads"
    Write-Skip "-SkipModels given"
} else {
    Write-Step "Model downloads (~28GB total, 2 public repos, no token needed)"

    # 1) LTX-2.3 weights, 5 files / 22,888,021,726 B:
    #      ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors            (0.93GB)
    #      ltx-2.3-components/vae/LTX23_video_vae_bf16.safetensors        (1.35GB)
    #      ltx-2.3-components/vae/LTX23_audio_vae_bf16.safetensors        (0.34GB)
    #      ltx-2.3-components/text_encoders/..._text_projection_bf16...   (2.15GB)
    #      ltx-2.3-gguf/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf            (16.54GB)
    #    Already laid out exactly as the project wants them, so this expands into
    #    models/ verbatim -- in particular the transformer GGUF arrives directly in
    #    models/ltx-2.3-gguf/ and the old "flatten one level up" fixup is gone.
    Invoke-ModelDownload -Name "LTX-2.3 weights (5 files)" `
        -Repo "Rootport/Nz-LTX23-weights" `
        -Include @("ltx-2.3/*", "ltx-2.3-components/*", "ltx-2.3-gguf/*") `
        -LocalDir "models" `
        -CheckDir @("models/ltx-2.3", "models/ltx-2.3-components", "models/ltx-2.3-gguf") `
        -MinBytes 22500000000   # repo total 22,888,021,726

    # 2) Gemma-3-12B, 11 files / 7,339,810,357 B:
    #      gemma-3-12b-it-gguf/gemma-3-12b-it-Q4_K_M.gguf                 (6.80GB)
    #      gemma-3-12b-it-tokenizer/  (10 small files, ~38MB)
    #    The tokenizer dir is tokenizer/preprocessor config ONLY -- the repo holds
    #    no multi-GB model-*.safetensors, so there is nothing here to exclude.
    Invoke-ModelDownload -Name "Gemma-3-12B GGUF + tokenizer set (11 files)" `
        -Repo "Rootport/Nz-Gemma3-12B" `
        -Include @("gemma-3-12b-it-gguf/*", "gemma-3-12b-it-tokenizer/*") `
        -LocalDir "models" `
        -CheckDir @("models/gemma-3-12b-it-gguf", "models/gemma-3-12b-it-tokenizer") `
        -MinBytes 7200000000    # repo total 7,339,810,357
}

# ----------------------------------------------------------------------------
# 7) Verification + regenerate models/INSTALLED_PATHS.txt
#
#    $required is derived DIRECTLY from services/ltx_runner.py:
#      * _real_available()  (L178-198): engine_python, gemma_root,
#        spatial_upsampler_path, gguf_transformer_path, gguf_gemma_path, and the
#        3 component_* files; plus engine_dir/worker.py.
#      * env/launch _require_path() (L489-507): the same load-bearing files.
#    The 46GB monolith checkpoint_path is DELIBERATELY NOT gated (reference-only
#    payload field; the GGUF+component path never opens it).
#    We ALSO check the app venv python (needed to run the server) and the smoke
#    test file when -RunSmoke.
# ----------------------------------------------------------------------------
Write-Step "Verification (required load-bearing artifacts)"

# Each row: label | project-relative path | isDir | minBytes (0 => existence-only)
$required = @(
    @{ Label = "engine_python";           Rel = ".venv-engine/Scripts/python.exe";                                                 IsDir = $false; Min = [long]0 }
    @{ Label = "app_python";              Rel = ".venv/Scripts/python.exe";                                                        IsDir = $false; Min = [long]0 }
    @{ Label = "engine worker.py";        Rel = "engine/worker.py";                                                                IsDir = $false; Min = [long]0 }
    @{ Label = "gguf_transformer";        Rel = "models/ltx-2.3-gguf/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf";                       IsDir = $false; Min = [long]17000000000 }
    @{ Label = "gguf_gemma";              Rel = "models/gemma-3-12b-it-gguf/gemma-3-12b-it-Q4_K_M.gguf";                           IsDir = $false; Min = [long]7000000000 }
    @{ Label = "component_video_vae";     Rel = "models/ltx-2.3-components/vae/LTX23_video_vae_bf16.safetensors";                  IsDir = $false; Min = [long]1000000000 }
    @{ Label = "component_audio_vae";     Rel = "models/ltx-2.3-components/vae/LTX23_audio_vae_bf16.safetensors";                  IsDir = $false; Min = [long]200000000 }
    @{ Label = "component_text_projection"; Rel = "models/ltx-2.3-components/text_encoders/ltx-2.3_text_projection_bf16.safetensors"; IsDir = $false; Min = [long]1500000000 }
    @{ Label = "spatial_upsampler";       Rel = "models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors";                      IsDir = $false; Min = [long]800000000 }
    @{ Label = "gemma_root (tokenizer dir)"; Rel = "models/gemma-3-12b-it-tokenizer";                                             IsDir = $true;  Min = [long]20000000 }
)

$rows = @()
$anyMissing = $false
foreach ($r in $required) {
    $abs = Join-Path $ProjectRoot $r.Rel
    $exists = Test-Path $abs
    $size = if ($exists) { Get-PathSize $abs } else { [long]0 }
    $ok = $exists -and ($size -ge $r.Min)
    if (-not $ok) { $anyMissing = $true }
    $rows += [pscustomobject]@{
        Status = if ($ok) { "PASS" } else { "MISSING" }
        Label  = $r.Label
        Size   = if ($exists) { Format-Size $size } else { "-" }
        Path   = $r.Rel
    }
}

# Aligned table.
$wLabel = ($rows | ForEach-Object { $_.Label.Length } | Measure-Object -Maximum).Maximum
$wSize = ($rows | ForEach-Object { $_.Size.Length } | Measure-Object -Maximum).Maximum
foreach ($row in $rows) {
    $color = if ($row.Status -eq "PASS") { "Green" } else { "Red" }
    $line = "  {0,-8}{1,-$($wLabel + 2)}{2,$wSize}  {3}" -f $row.Status, $row.Label, $row.Size, $row.Path
    Write-Host $line -ForegroundColor $color
}

# ----------------------------------------------------------------------------
# Regenerate models/INSTALLED_PATHS.txt to match the CURRENT config.yaml (GGUF +
# component recipe). NO monolith, NO QAT. This supersedes the stale file that
# still referenced the deleted 46GB monolith + old gemma-3-12b-it-qat dir.
# ----------------------------------------------------------------------------
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$installedPaths = @"
# Regenerated by scripts/install_ltx.ps1 at $stamp
# GGUF + component-file recipe (matches config.yaml -> model:). The 46GB monolith
# and 22.7GB QAT Gemma are intentionally absent (deleted; never re-downloaded).
# The monolith path below is a reference-only payload field (never opened).
  gguf_transformer:          "./models/ltx-2.3-gguf/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
  gguf_gemma:                "./models/gemma-3-12b-it-gguf/gemma-3-12b-it-Q4_K_M.gguf"
  component_video_vae:       "./models/ltx-2.3-components/vae/LTX23_video_vae_bf16.safetensors"
  component_audio_vae:       "./models/ltx-2.3-components/vae/LTX23_audio_vae_bf16.safetensors"
  component_text_projection: "./models/ltx-2.3-components/text_encoders/ltx-2.3_text_projection_bf16.safetensors"
  spatial_upsampler:         "./models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
  gemma_root:                "./models/gemma-3-12b-it-tokenizer"
  engine_python:             "./.venv-engine/Scripts/python.exe"
  checkpoint_path (ref-only): "./models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
"@
$installedPaths | Out-File -FilePath "$ProjectRoot\models\INSTALLED_PATHS.txt" -Encoding utf8
Write-Host "`nRegenerated models/INSTALLED_PATHS.txt" -ForegroundColor Cyan

if ($anyMissing) {
    Write-Host ""
    Write-Warning "One or more required artifacts are MISSING (see table above)."
    if ($SkipModels) { Write-Warning "You passed -SkipModels; re-run without it to fetch models." }
    if ($SkipVenv) { Write-Warning "You passed -SkipVenv; re-run without it to build the venvs." }
    exit 1
}
Write-Ok "All required artifacts present."

# ----------------------------------------------------------------------------
# Optional mock, GPU-free smoke test (uses the app venv).
# ----------------------------------------------------------------------------
if ($RunSmoke) {
    Write-Step "Smoke test (mock, GPU-free)"
    if (-not (Test-Path $appPy)) { throw "App venv python not found for smoke test: $appPy" }
    & $appPy -m pytest -q tests/test_smoke.py
    if ($LASTEXITCODE -ne 0) { throw "Smoke test failed." }
    Write-Ok "Smoke test passed."
}

Write-Host "`nDone. Backend is ready to run:  ./run.ps1" -ForegroundColor Green
# Explicit success code: the last native command's $LASTEXITCODE must not leak.
exit 0
