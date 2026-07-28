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
      1. Prereqs (git, uv; ffmpeg + ffprobe warn) + process-scoped env isolation.
      2. uv-managed Python 3.12.
      3. App venv  .venv        (torch-FREE; `uv sync` of root pyproject.toml).
      4. Engine venv .venv-engine (torch 2.9.1+cu128 stack). DEFAULT = deterministic
         FREEZE path (reproduces the VALIDATED stack all verification ran on).
         Re-applied automatically whenever the pinned dependency set changes, so
         "git pull, then re-run this" actually updates the venv.
         -ResolveLatest opts into a fresh resolve (UNVALIDATED newer torch).
      5. Model downloads (~30GB, 3 guarded items) via .venv-engine's hf.exe,
         pulled from three PUBLIC, NON-GATED repos. No HuggingFace account,
         login or token is required at any point.
      6. Verification table (PASS/MISSING) + regenerate models/INSTALLED_PATHS.txt.

    The GGUF + component-file recipe is the ONLY supported real path. It never
    opens the old 46GB monolith (ltx-2.3-22b-distilled-1.1.safetensors) or the
    22.7GB QAT Gemma dir -- both were physically deleted; this installer never
    downloads them. The monolith path is not configurable at all any more
    (config.yaml's checkpoint_path key was removed 2026-07-28, PENDING_TASKS.md
    3-26, once confirmed dead): the worker payload's checkpoint_path field is
    now a hardcoded "" in services/ltx_runner.py.

.NOTES
    ATTENTION BACKEND (no GPU-specific knob)
    ----------------------------------------
    There is none to choose: PyTorch SDPA is the attention backend on EVERY arch
    (Ada / Ampere / Hopper / Blackwell). xformers and flash-attn are never
    installed by this script -- SDPA is what the code actually uses, and on
    Blackwell adding flash-attn can jam it. sageattention was historically an
    exception (a declared-but-unused engine-venv dependency, pure dead weight
    since nothing imported it) -- removed in the 2026-07-28 dependency cleanup
    (PENDING_TASKS.md 3-25; see engine/venv-engine.freeze.txt). Blackwell needs
    an R570+ driver. Because nothing here is arch-dependent, this installer does
    not detect or take a GPU architecture at all.

.EXAMPLE
    ./scripts/install_ltx.ps1                       # full install
    ./scripts/install_ltx.ps1 -ResolveLatest        # fresh (unvalidated) engine resolve
    ./scripts/install_ltx.ps1 -SkipModels           # venvs only, no downloads
    ./scripts/install_ltx.ps1 -RunSmoke             # + mock GPU-free smoke test
#>

[CmdletBinding()]
param(
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
# Env isolation (process-scoped only; mirrors the $env: block in run.ps1 -- no
# line numbers, they go stale). Keep the
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
# 1) Prerequisites
#
#    NOTE (deliberate behaviour change, 2026-07): this step no longer probes
#    nvidia-smi and no longer aborts when no NVIDIA GPU can be seen. The old
#    probe existed only to pick an attention backend, and there is nothing left
#    to pick (SDPA on every arch). Neither the venvs nor the model downloads
#    care about the GPU -- it is first needed at generation time -- so install
#    now succeeds on a box with no driver / no nvidia-smi (e.g. a build agent),
#    and a wrong or missing GPU surfaces when you actually run the engine.
# ----------------------------------------------------------------------------
Write-Step "Checking prerequisites"
Require-Cmd git
Require-Cmd uv
# Both CLIs are needed, and BOTH are warn-only on purpose: the installer's caller
# puts an in-project tools/ffmpeg on PATH, so a miss here is usually recoverable.
# ffprobe is the easy one to forget -- services/video_io.py's has_audio_stream()
# shells out to it, and when it is absent that helper just answers False, so every
# input looks silent and its audio track is dropped WITHOUT any error.
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg not found on PATH. The backend needs it to encode MP4."
}
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Write-Warning "ffprobe not found on PATH. The backend needs it to detect audio streams; without it audio tracks are silently dropped."
}
Write-Host "git    : $((Get-Command git).Source)"
Write-Host "uv     : $((Get-Command uv).Source)"
Write-Host "root   : $ProjectRoot"
Write-Host "HF_HOME: $env:HF_HOME"

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
    #
    # --extra dev is ALWAYS passed (2026-07-28 owner decision, PENDING_TASKS.md
    # 3-26, option (a)): `uv sync` treats the venv as authoritative for exactly
    # the extras it is told to sync, so a plain `uv sync` doesn't just skip the
    # `dev` optional-dependency group (pytest / iniconfig / pluggy) -- it PRUNES
    # it back out if it was ever installed. That silently deletes pytest on every
    # re-run of this step, which is exactly the trap that bit the 2026-07-28
    # NAG work (recovered only by a manual `uv sync --extra dev`). Always
    # including it costs a few MB and keeps "git pull, re-run setup" from ever
    # breaking `pytest` again -- worth it over saving that space for end users.
    Write-Do "uv sync --extra dev  (root pyproject.toml, app deps + dev extra)"
    uv sync --extra dev
    if ($LASTEXITCODE -ne 0) { throw "uv sync --extra dev (app venv) failed." }
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
#
#    RE-SYNC (2026-07): this step no longer skips merely because .venv-engine
#    exists. A `git pull` can move the freeze file or the pinned revs below, and
#    the documented update flow is "git pull, then re-run the installer" -- which
#    only works if the freeze is re-applied when the pinned set changes. The
#    applied state is recorded as a hash in .venv-engine/.nz-engine-state, which
#    doubles as the COMPLETION MARKER: an interrupted install leaves a half-built
#    venv with no marker, and the next run re-applies instead of trusting it.
# ----------------------------------------------------------------------------
$enginePy = "$ProjectRoot\.venv-engine\Scripts\python.exe"
$freezeSrc = "$ProjectRoot\engine\venv-engine.freeze.txt"
$enginePyprojectDir = "$ProjectRoot\engine"
# Applied-state hash + completion marker. It lives INSIDE .venv-engine, which
# .gitignore already excludes, so it is never tracked and dies with the venv.
$engineStateFile = "$ProjectRoot\.venv-engine\.nz-engine-state"

# The 3 git packages as uv requirement strings, at their EXACT pinned revs. The
# freeze file lists them only as bare `name==version` (no such build exists on
# PyPI), so a plain `-r freeze` cannot fetch them -- they are installed from
# these URLs first, and their bare lines are filtered out of the freeze copy.
# Keeping them in ONE array is what lets the state hash below cover them: bump a
# rev here and the hash changes, which forces a re-apply on the next run.
$engineGitPins = @(
    "diffusers @ git+https://github.com/huggingface/diffusers.git@01de02e8b4f2cc91df4f3e91cb6535ebcbeb490c"
    "ltx-core @ git+https://github.com/Lightricks/LTX-2.git@00dc53d3f81c405932f9f16d9c57557de411e702#subdirectory=packages/ltx-core"
    "ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git@00dc53d3f81c405932f9f16d9c57557de411e702#subdirectory=packages/ltx-pipelines"
)

# SHA-256 over the freeze body PLUS the git pins. The freeze file alone is NOT a
# sufficient input: the 3 revs are hardcoded in this script, so a rev bump would
# otherwise leave the hash unchanged and never re-apply. Line endings are
# normalised first so a CRLF/LF checkout flip does not masquerade as a change.
function Get-EngineStateHash {
    param(
        [Parameter(Mandatory)] [string]   $FreezeFile,
        [Parameter(Mandatory)] [string[]] $GitPins
    )
    $body = [System.IO.File]::ReadAllText($FreezeFile).Replace("`r`n", "`n")
    $payload = $body + "`n" + ($GitPins -join "`n") + "`n"
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($payload))
    } finally {
        $sha.Dispose()
    }
    return (($digest | ForEach-Object { $_.ToString("x2") }) -join "")
}

# The 2-stage deterministic freeze apply, called by BOTH the create path and the
# re-sync path (never inline a second copy of this).
#  (a) install the 3 git packages at their pinned revs (see $engineGitPins).
#  (b) install the remaining pinned wheels from a TEMP copy of the freeze that has
#      those 3 bare lines removed (already installed in (a); left in place uv
#      would fetch some other PyPI build of the same version).
# The cu128 --index and --index-strategy in (b) are BOTH load-bearing: without
# them uv resolves CPU-only torch/torchaudio wheels (known uv bug for the
# platform-marker-less torchaudio source).
function Invoke-EngineFreezeApply {
    param(
        [Parameter(Mandatory)] [string]   $EnginePython,
        [Parameter(Mandatory)] [string]   $FreezeFile,
        [Parameter(Mandatory)] [string[]] $GitPins
    )
    Write-Do "install 3 git packages at pinned revs (diffusers / ltx-core / ltx-pipelines)"
    $gitArgs = @("pip", "install", "--python", $EnginePython) + $GitPins
    uv @gitArgs
    if ($LASTEXITCODE -ne 0) { throw "engine git-package install failed." }

    # Distribution names taken from the pins themselves, so this filter cannot
    # drift out of sync with the list above.
    $gitNames = $GitPins | ForEach-Object { [regex]::Escape((($_ -split ' ')[0])) }
    $gitLineRe = "^(" + ($gitNames -join "|") + ")=="

    $tmpFreeze = Join-Path ([System.IO.Path]::GetTempPath()) ("venv-engine.freeze.nogit.{0}.txt" -f ([guid]::NewGuid().ToString("N")))
    try {
        Get-Content $FreezeFile |
            Where-Object { $_ -notmatch $gitLineRe } |
            Set-Content -Path $tmpFreeze -Encoding utf8

        Write-Do "install pinned wheels from freeze (cu128 index)"
        uv pip install --python $EnginePython `
            --index https://download.pytorch.org/whl/cu128 `
            --index-strategy unsafe-best-match `
            -r $tmpFreeze
        if ($LASTEXITCODE -ne 0) { throw "engine freeze install failed." }
    } finally {
        if (Test-Path $tmpFreeze) { Remove-Item $tmpFreeze -Force -ErrorAction SilentlyContinue }
    }
}

if ($SkipVenv) {
    Write-Step "Engine venv .venv-engine"
    Write-Skip "-SkipVenv given"
} elseif ($ResolveLatest) {
    # ------------------------------------------------------------------------
    # Opt-in fresh resolve. --index-strategy unsafe-best-match is REQUIRED: the
    # pyproject's torchaudio source lacks a platform marker, and without
    # unsafe-best-match uv pulls a CPU-only torchaudio (known bug). This path is
    # UNVALIDATED (newer torch ~2.11 vs the verified 2.9.1). Unlike before, an
    # existing venv no longer makes the flag a no-op -- asking for a fresh
    # resolve now always performs one.
    # ------------------------------------------------------------------------
    Write-Step "Engine venv .venv-engine  (-ResolveLatest: fresh resolve)"
    if (-not (Test-Path $enginePy)) {
        Write-Do "uv venv --python 3.12 .venv-engine"
        uv venv --python 3.12 .venv-engine
        if ($LASTEXITCODE -ne 0) { throw "uv venv .venv-engine failed." }
    }
    Write-Warning "-ResolveLatest: resolving the engine venv fresh from engine-venv-pyproject.toml."
    Write-Warning "This yields an UNVALIDATED newer torch (~2.11). All project verification ran on"
    Write-Warning "the frozen 2.9.1+cu128 stack (the default). Use only if you accept re-validating."
    Write-Do "uv pip install --index-strategy unsafe-best-match <engine pyproject dir>"
    uv pip install --python $enginePy --index-strategy unsafe-best-match $enginePyprojectDir
    if ($LASTEXITCODE -ne 0) { throw "engine venv resolve (-ResolveLatest) failed." }
    # No state marker is written here, and a stale one is dropped: the resulting
    # venv is NOT the frozen stack, so claiming it matches would be a lie. The
    # absence is meaningful in both directions -- it records "this venv came in by
    # the other route", and it makes the next default run re-pin it to the freeze.
    if (Test-Path $engineStateFile) { Remove-Item $engineStateFile -Force }
    Write-Ok ".venv-engine ready (unvalidated resolve)"
} else {
    Write-Step "Engine venv .venv-engine  (torch cu128 stack, deterministic freeze)"
    if (-not (Test-Path $freezeSrc)) { throw "Engine freeze file not found: $freezeSrc" }

    $wantState = Get-EngineStateHash -FreezeFile $freezeSrc -GitPins $engineGitPins
    $haveState = ""
    if (Test-Path $engineStateFile) {
        $haveState = ((Get-Content $engineStateFile -Raw) -replace '\s', '')
    }

    if (-not (Test-Path $enginePy)) {
        Write-Do "uv venv --python 3.12 .venv-engine"
        uv venv --python 3.12 .venv-engine
        if ($LASTEXITCODE -ne 0) { throw "uv venv .venv-engine failed." }
        Invoke-EngineFreezeApply -EnginePython $enginePy -FreezeFile $freezeSrc -GitPins $engineGitPins
        Set-Content -Path $engineStateFile -Value $wantState -Encoding ascii
        Write-Ok ".venv-engine ready"
    } elseif ($haveState -eq $wantState) {
        Write-Skip ".venv-engine already matches the pinned dependency set"
    } else {
        # A MISSING marker deliberately means "re-apply", not "assume fine": that
        # is what rescues both an interrupted install and every venv built before
        # this marker existed. The reverse default would strand exactly the people
        # who just updated.
        if ($haveState) {
            Write-Do "the pinned dependency set changed since the last install -- re-applying the freeze"
        } else {
            Write-Do "no completion marker found (interrupted install, or built before re-sync existed) -- re-applying the freeze"
        }
        Invoke-EngineFreezeApply -EnginePython $enginePy -FreezeFile $freezeSrc -GitPins $engineGitPins
        Set-Content -Path $engineStateFile -Value $wantState -Encoding ascii
        Write-Ok ".venv-engine re-synced"
    }
}

# hf.exe (used by the model downloads below) must exist in the engine venv.
$hfExe = "$ProjectRoot\.venv-engine\Scripts\hf.exe"

# ----------------------------------------------------------------------------
# Attention backend: nothing to install, on any GPU.
#
# PyTorch SDPA is the backend the engine actually uses, on every architecture.
# This step installs nothing at all -- not on any arch, not optionally.
# (Historically this step could pick up a prebuilt xformers wheel out of wheels/;
# that path was removed because such a wheel is compiled for ONE compute
# capability and installs cleanly on machines it cannot run on. If you want to
# experiment with xformers, build and install it by hand -- see
# scripts/build_xformers.ps1 -- and note the engine code does not import it.)
# ----------------------------------------------------------------------------

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
# 5) Model downloads (~30GB) via the engine venv's hf.exe.
#    Everything comes from THREE self-hosted repos that are PUBLIC and NON-GATED,
#    so no HuggingFace account, login or token is involved anywhere:
#      Rootport/Nz-LTX23-weights -> ltx-2.3/, ltx-2.3-components/, ltx-2.3-gguf/,
#                                   ltx-2.3-ic-lora/
#      Rootport/Nz-Gemma3-12B    -> gemma-3-12b-it-gguf/, gemma-3-12b-it-tokenizer/
#      Rootport/Nz-DWPose        -> preprocessors/
#    All three repos mirror this project's models/ layout 1:1, so each one expands
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
#   WARNING (huggingface_hub version dependency, verified 2026-07-26): the single-
#   flag form above is correct ONLY for huggingface_hub 0.36.2, the engine venv's
#   current pin. On 1.20.1 it flips: one --include with multiple patterns warns
#   "Ignoring --include since filenames have been explicitly set." and silently
#   drops files (spatial upscaler went missing in testing). If that venv's
#   huggingface_hub is ever upgraded, switch this to repeated --include flags.
#   The per-directory Min guard below does catch the resulting short download, but
#   the thrown error gives no hint that a version bump is the cause -- look here
#   first.
#
#   NOTE (glob semantics, verified live against both repos): --include matches with
#   Python fnmatch against the repo-relative path, and `*` DOES cross '/'. So
#   "ltx-2.3-components/*" reaches the nested vae/ and text_encoders/ files two
#   levels down. Just as importantly, the repo-root card files (LICENSE /
#   NOTICE.md / README.md / .gitattributes) match NO "<dir>/*" pattern -- which is
#   what stops the two repos' identically-named cards from landing in models/ and
#   overwriting each other.
#
#   NOTE (why Check is separate from LocalDir): all three calls below pass
#   LocalDir = "models", so sizing the guard on LocalDir would see the ~24GB LTX
#   download and then wrongly SKIP the Gemma and DWPose ones. Check instead names
#   the subdirectories that THIS repo expands into.
#
#   NOTE (the guard is PER-DIRECTORY -- read before adding a repo): every Check
#   entry carries its OWN Min, and EACH one must clear it independently or the
#   download runs. This is deliberate, and it replaced a single combined MinBytes
#   compared against the SUM of the directories. The sum form is unsafe: one large
#   file can stand in for an entirely missing sibling directory. The Gemma repo
#   really had that bug -- its 7.3GB GGUF alone cleared the combined threshold, so
#   a wholly absent tokenizer dir still SKIPped here, while the step 6 table below
#   reported gemma_root MISSING and exited 1. Re-running changed nothing: a
#   permanent, unrecoverable deadlock. Per-directory Mins make that class of
#   mistake structurally impossible.
#
#   How to size one Min: the per-file step 6 table is the real correctness gate,
#   so a Min is correct when losing anything THAT TABLE would flag drops the
#   directory below it. In practice, set Min just under the directory's full
#   expected size, and always ABOVE (directory total - smallest file the table
#   checks inside it). Files the table never inspects individually do not have to
#   be catchable -- losing one cannot produce a MISSING row, so it cannot deadlock.
#   (Concretely: the tokenizer dir is gated only as a whole, at 20MB, so the
#   35-byte added_tokens.json is not separately catchable and does not need to be.
#   Chasing it would have forced a 35-byte-wide threshold window.) Each call below
#   shows its own arithmetic.
#
#   Caveat, deliberately accepted for models/ltx-2.3-gguf/ only: that directory
#   also holds any extra self-converted transformer GGUFs the user dropped in
#   (10Eros / Sulphur, ~16.5GB each), so its size can float over any threshold even
#   with the real file gone. No size guard can fix that one; the step 6 table can,
#   and does.
function Invoke-ModelDownload {
    param(
        [Parameter(Mandatory)] [string]      $Name,
        [Parameter(Mandatory)] [string]      $Repo,
        [Parameter(Mandatory)] [string[]]    $Include,   # glob(s), one --include
        [Parameter(Mandatory)] [string]      $LocalDir,  # relative to root
        # One entry per directory this repo expands into:
        #   @{ Dir = "<path relative to root>"; Min = <bytes> }
        # Checked INDEPENDENTLY -- sizes are never summed against a single floor.
        [Parameter(Mandatory)] [hashtable[]] $Check
    )
    $absLocal = Join-Path $ProjectRoot $LocalDir

    # Returns Short = the entries that are under their own Min (EMPTY when every
    # entry passes, which is the only "present" verdict), plus Total = the combined
    # size, used solely for the human-readable message.
    function Test-CheckSet {
        param([hashtable[]] $Set)
        $short = @()
        $total = [long] 0
        foreach ($c in $Set) {
            $sz = Get-PathSize (Join-Path $ProjectRoot $c.Dir)
            $total += $sz
            if ($sz -lt [long] $c.Min) {
                $short += "$($c.Dir) ($(Format-Size $sz) < $(Format-Size ([long] $c.Min)))"
            }
        }
        return [pscustomobject]@{ Short = @($short); Total = $total }
    }

    $state = Test-CheckSet -Set $Check
    if ($state.Short.Count -eq 0) {
        $dirs = ($Check | ForEach-Object { $_.Dir }) -join ', '
        Write-Skip "$Name  ($(Format-Size $state.Total) already present in $dirs)"
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
    $state = Test-CheckSet -Set $Check
    if ($state.Short.Count -gt 0) {
        throw "'$Name' downloaded but these directories are smaller than expected: $($state.Short -join '; '). Check the include globs."
    }
    Write-Ok "$Name  ($(Format-Size $state.Total))"
}

if ($SkipModels) {
    Write-Step "Model downloads"
    Write-Skip "-SkipModels given"
} else {
    Write-Step "Model downloads (~30GB total, 3 public repos, no token needed)"

    # 1) LTX-2.3 weights, 7 files / 24,196,952,364 B:
    #      ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors            (0.93GB)
    #      ltx-2.3-components/vae/LTX23_video_vae_bf16.safetensors        (1.35GB)
    #      ltx-2.3-components/vae/LTX23_audio_vae_bf16.safetensors        (0.34GB)
    #      ltx-2.3-components/text_encoders/..._text_projection_bf16...   (2.15GB)
    #      ltx-2.3-gguf/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf            (16.54GB)
    #      ltx-2.3-ic-lora/pixel-spatial-upscaler/...-x2-0.9.safetensors   (0.65GB)
    #      ltx-2.3-ic-lora/union-control/...-union-control-ref0.5...       (0.65GB)
    #    Already laid out exactly as the project wants them, so this expands into
    #    models/ verbatim -- in particular the transformer GGUF arrives directly in
    #    models/ltx-2.3-gguf/ and the old "flatten one level up" fixup is gone.
    #    The 2 IC-LoRA adapters back all three entries config.yaml registers under
    #    ic_loras: -- pixel-spatial-upscaler-x2 (resolution boost), plus the ONE
    #    union-control file published twice, as canny-control (edge-outline
    #    guidance) and pose-control (skeleton guidance). They are not optional in
    #    practice: gradio_ui/adapters.py lists those three names from a static
    #    fallback even when nothing is registered, so a fresh install without these
    #    files puts three adapters in the UI that 404 the moment they are picked.
    #    The x4 upscaler variant is deliberately not in the repo (unregistered).
    #    One pattern covers both nested files: `*` crosses '/' (glob note above).
    #    Per-directory Mins (see the sizing rule above); every file below is gated
    #    individually by the step 6 table, so each Min sits above
    #    (dir total - smallest file in that dir):
    #      ltx-2.3            995,743,560 , 1 file          -> Min   900,000,000
    #      ltx-2.3-components 4,129,262,838, smallest 364,855,188
    #                                        (4,129,262,838-364,855,188=3,764,407,650)
    #                                                        -> Min 4,000,000,000
    #      ltx-2.3-gguf       17,763,015,328, 1 file        -> Min 17,000,000,000
    #      ltx-2.3-ic-lora    1,308,930,638, smallest 654,465,286
    #                                        (leaves 654,465,352)
    #                                                        -> Min 1,000,000,000
    Invoke-ModelDownload -Name "LTX-2.3 weights + IC-LoRA (7 files)" `
        -Repo "Rootport/Nz-LTX23-weights" `
        -Include @("ltx-2.3/*", "ltx-2.3-components/*", "ltx-2.3-gguf/*", "ltx-2.3-ic-lora/*") `
        -LocalDir "models" `
        -Check @(
            @{ Dir = "models/ltx-2.3";            Min = [long]   900000000 }
            @{ Dir = "models/ltx-2.3-components"; Min = [long]  4000000000 }
            @{ Dir = "models/ltx-2.3-gguf";       Min = [long] 17000000000 }
            @{ Dir = "models/ltx-2.3-ic-lora";    Min = [long]  1000000000 }
        )

    # 2) Gemma-3-12B, 11 files / 7,339,810,357 B:
    #      gemma-3-12b-it-gguf/gemma-3-12b-it-Q4_K_M.gguf                 (6.80GB)
    #      gemma-3-12b-it-tokenizer/  (10 small files, ~38MB)
    #    The tokenizer dir is tokenizer/preprocessor config ONLY -- the repo holds
    #    no multi-GB model-*.safetensors, so there is nothing here to exclude.
    #    THIS is the repo the old single-sum guard broke on: the GGUF is
    #    7,300,574,976 B on its own, already over the former MinBytes 7,200,000,000,
    #    so a completely missing tokenizer dir SKIPped here and then failed step 6
    #    forever. Per-directory Mins:
    #      gemma-3-12b-it-gguf      7,300,574,976, 1 file -> Min 7,300,000,000
    #      gemma-3-12b-it-tokenizer    39,235,381, 10 files -> Min    39,000,000
    #    The tokenizer Min is set against what step 6 actually gates -- that dir as
    #    a WHOLE, at 20MB. Losing any of the three files big enough to matter
    #    (tokenizer.json 33,384,570 / tokenizer.model 4,689,074 /
    #    tokenizer_config.json 1,157,001) drops the dir under 39,000,000 and
    #    re-downloads. The seven tiny files (35 B .. 1,615 B) are not individually
    #    catchable, and do not need to be: step 6 never looks at them one by one,
    #    so their loss cannot produce a MISSING row. Sizing for them instead would
    #    demand a 35-byte-wide window (>39,235,346 and <=39,235,381).
    Invoke-ModelDownload -Name "Gemma-3-12B GGUF + tokenizer set (11 files)" `
        -Repo "Rootport/Nz-Gemma3-12B" `
        -Include @("gemma-3-12b-it-gguf/*", "gemma-3-12b-it-tokenizer/*") `
        -LocalDir "models" `
        -Check @(
            @{ Dir = "models/gemma-3-12b-it-gguf";      Min = [long] 7300000000 }
            @{ Dir = "models/gemma-3-12b-it-tokenizer"; Min = [long]   39000000 }
        )

    # 3) DWPose preprocessor models, 2 files / 352,756,773 B:
    #      preprocessors/yolox_l.torchscript.pt              (207.6MB)
    #      preprocessors/dw-ll_ucoco_384_bs5.torchscript.pt  (128.8MB)
    #    These two TorchScript modules are the pose estimator behind the
    #    pose-control IC-LoRA: yolox_l finds the people, dw-ll_ucoco turns each one
    #    into the skeleton image that is fed to the adapter. engine/preprocess/
    #    dwpose.py loads BOTH from models/preprocessors/ by an absolute path built
    #    from its own file location, so this layout is not negotiable.
    #    Single directory, so this one was never at risk of the cross-directory
    #    masking above; it keeps the same 340,000,000, which is above
    #    (352,756,773 - smaller file 135,059,124 = 217,697,649) and below the full
    #    352,756,773, so either file going missing re-triggers the download.
    Invoke-ModelDownload -Name "DWPose preprocessor models (2 files)" `
        -Repo "Rootport/Nz-DWPose" `
        -Include @("preprocessors/*") `
        -LocalDir "models" `
        -Check @(
            @{ Dir = "models/preprocessors"; Min = [long] 340000000 }
        )
}

# ----------------------------------------------------------------------------
# 6) Verification + regenerate models/INSTALLED_PATHS.txt
#
#    $required is derived DIRECTLY from services/ltx_runner.py (referenced by
#    FUNCTION NAME only -- line numbers here went stale once already):
#      * LTXRunner._real_available(): engine_python, gemma_root,
#        spatial_upsampler_path, gguf_transformer_path, gguf_gemma_path, and the
#        3 component_* files; plus engine_dir/worker.py.
#      * env/launch _RealBackend._require_path(): the same load-bearing files.
#    The 46GB monolith is not gated here: it is no longer a config option at all
#    (checkpoint_path was removed from config.yaml 2026-07-28, PENDING_TASKS.md
#    3-26) and the GGUF+component path never opened it even before that.
#    We ALSO check the app venv python (needed to run the server) and the smoke
#    test file when -RunSmoke.
#
#    The 4 IC-LoRA / DWPose rows at the end are a deliberate widening: _real_available()
#    does not look at them (their absence downgrades no backend to mock), but
#    config.yaml registers all three ic_loras: entries unconditionally and
#    gradio_ui/adapters.py falls back to the same three names even when nothing is
#    registered. A missing file there is therefore invisible until a user picks the
#    adapter and gets a 404, which is exactly the failure this table exists to
#    convert into an up-front, named MISSING. Both source repos guard tightly enough
#    (see the per-directory Min note in step 5) that a MISSING here is cleared by
#    re-running.
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
    @{ Label = "ic_lora pixel-spatial-upscaler-x2"; Rel = "models/ltx-2.3-ic-lora/pixel-spatial-upscaler/ltx-2.3-22b-ic-lora-pixel-spatial-upscaler-x2-0.9.safetensors"; IsDir = $false; Min = [long]600000000 }
    @{ Label = "ic_lora union-control (canny/pose)"; Rel = "models/ltx-2.3-ic-lora/union-control/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"; IsDir = $false; Min = [long]600000000 }
    @{ Label = "dwpose detector (yolox_l)"; Rel = "models/preprocessors/yolox_l.torchscript.pt";                                  IsDir = $false; Min = [long]200000000 }
    @{ Label = "dwpose estimator (dw-ll_ucoco)"; Rel = "models/preprocessors/dw-ll_ucoco_384_bs5.torchscript.pt";                 IsDir = $false; Min = [long]120000000 }
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
# The monolith path is no longer a config option at all (checkpoint_path was
# removed from config.yaml 2026-07-28, PENDING_TASKS.md 3-26, confirmed dead).
  gguf_transformer:          "./models/ltx-2.3-gguf/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
  gguf_gemma:                "./models/gemma-3-12b-it-gguf/gemma-3-12b-it-Q4_K_M.gguf"
  component_video_vae:       "./models/ltx-2.3-components/vae/LTX23_video_vae_bf16.safetensors"
  component_audio_vae:       "./models/ltx-2.3-components/vae/LTX23_audio_vae_bf16.safetensors"
  component_text_projection: "./models/ltx-2.3-components/text_encoders/ltx-2.3_text_projection_bf16.safetensors"
  spatial_upsampler:         "./models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
  gemma_root:                "./models/gemma-3-12b-it-tokenizer"
  engine_python:             "./.venv-engine/Scripts/python.exe"
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
    # No extra `uv sync --extra dev` needed here (2026-07-28): step 3 now always
    # syncs with --extra dev, so pytest / iniconfig / pluggy are already present
    # by the time this branch runs. The special-case re-sync that used to live
    # here was made redundant by that step-3 change and has been removed.
    & $appPy -m pytest -q tests/test_smoke.py
    if ($LASTEXITCODE -ne 0) { throw "Smoke test failed." }
    Write-Ok "Smoke test passed."
}

Write-Host "`nDone. Backend is ready to run:  ./run.ps1" -ForegroundColor Green
# Explicit success code: the last native command's $LASTEXITCODE must not leak.
exit 0
