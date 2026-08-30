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
      0. Load + validate scripts/manifests/*.json  (the single source of truth
         for downloads, the on-disk layout, migration, the verification table
         and models/INSTALLED_PATHS.txt).
      1. Prereqs (git, uv; ffmpeg + ffprobe warn) + process-scoped env isolation.
      2. Migrate an existing pre-2026-08 models/ tree to the base-model-first
         layout, rewrite config.yaml's model paths to match, and delete the
         retired model.* key lines config.py no longer honours (they would
         otherwise log a WARNING on every server start).
      3. uv-managed Python 3.12.
      4. App venv  .venv        (torch-FREE; `uv sync` of root pyproject.toml).
      5. Engine venvs .venv-engine (LTX 2.3) and .venv-engine-ltx25 (LTX 2.5),
         both torch 2.9.1+cu128 stacks, built by one shared Ensure-EngineVenv.
         Two venvs, not one: the 2.5 stack needs transformers 5.x (Gemma 4) and
         the 2.3 stack is pinned to 4.57.6, so they cannot share an interpreter.
         DEFAULT = deterministic FREEZE path (reproduces the VALIDATED stacks all
         verification ran on). Re-applied automatically whenever a pinned
         dependency set changes, so "git pull, then re-run this" actually
         updates the venvs.
         -ResolveLatest opts into a fresh resolve (UNVALIDATED newer torch).
      6. Model downloads (~31GB) via .venv-engine's hf.exe, driven entirely by
         the manifests: staged into models\.dl\ and then remapped into place.
      7. Verification table (PASS/MISSING) + regenerate models/INSTALLED_PATHS.txt,
         both generated from the manifests' `files` arrays.
      8. Optional -RunSmoke.

    The GGUF + component-file recipe is the ONLY supported real path. It never
    opens the old 46GB monolith (ltx-2.3-22b-distilled-1.1.safetensors) or the
    22.7GB QAT Gemma dir -- both were physically deleted; this installer never
    downloads them. The monolith path is not configurable at all any more
    (config.yaml's checkpoint_path key was removed 2026-07-28, PENDING_TASKS.md
    3-26, once confirmed dead): the worker payload's checkpoint_path field is
    now a hardcoded "" in services/ltx_runner.py.

.NOTES
    LAYOUT + GUARD DOCTRINE (2026-08-19 rewrite)
    --------------------------------------------
    models/ is BASE-MODEL-FIRST:  models/<BaseModel>/<Category>/...
    (models/LTX23/Weights, models/LTX23/TextEncoder, models/Preprocessors/DWPose,
    ...). The folder IS the declaration -- where a file sits says which base model
    it belongs to, so nothing has to fingerprint weights.

    Every guard in this script is now PER EXPECTED FILE (manifest `files[]`, each
    with its own `min`), never a per-directory recursive total. The old
    directory-total guards are gone, and with them two whole classes of bug:

      * A directory total cannot decide "present" when two different repos feed
        the SAME directory. models/LTX23/TextEncoder is filled by BOTH
        Rootport/Nz-LTX23-weights (text projection) and Rootport/Nz-Gemma3-12B
        (the GGUF) -- with a directory-total guard, whichever downloads first
        pushes the directory over the floor and the post-download re-check of the
        other one throws, making a fresh install fail 100% of the time.
      * A directory total is inflated by files the user brought themselves
        (self-converted GGUFs in Weights/, LoRAs in StyleLoRA/), which can hide a
        missing OFFICIAL file forever. The old script documented that as an
        accepted, unfixable limitation for models/ltx-2.3-gguf/. Per-file guards
        make it structurally impossible: an official file that is absent is
        always downloaded, and a user file never affects any verdict.

    Consequently there is NO LONGER any constraint on how directories are
    arranged relative to each other. The old "each repo needs its OWN sibling
    check directory, never a child" rule (models/preprocessors-vda deliberately
    kept a SIBLING of models/preprocessors, IC-LoRA deblur/in-outpainting kept
    out of models/ltx-2.3-ic-lora/) existed ONLY to keep recursive size sums from
    masking each other. It is void: the new layout nests all four IC-LoRA
    adapters under models/LTX23/IC-LoRA/ and both preprocessors under
    models/Preprocessors/, and nothing can mask anything.

    ATTENTION BACKEND (no GPU-specific knob)
    ----------------------------------------
    PyTorch SDPA remains the attention backend on EVERY arch (Ada / Ampere /
    Hopper / Blackwell) and is always installed and always usable. xformers and
    flash-attn are never installed by this script -- SDPA is the baseline the
    code always falls back to, and on Blackwell adding flash-attn can jam it.
    sageattention was historically an exception (a declared-but-unused
    engine-venv dependency, pure dead weight since nothing imported it) --
    removed in the 2026-07-28 dependency cleanup (PENDING_TASKS.md 3-25; see
    engine/venv-engine.freeze.txt). It is back as of 2026-07-31: the
    Acceleration feature's SageAttentionService (engine/transformer/
    sage_attention_service.py) is now a real consumer, so sageattention 2.2.0
    (prebuilt wheel, cu128/torch2.9.1) and triton-windows (its runtime JIT
    dependency) are installed by default -- see $engineDirectPins below and
    engine/venv-engine.freeze.txt. As of 2026-08-25 the SAME wheel is installed
    into the LTX 2.5 venv as well ($ltx25DirectPins +
    engine25/venv-engine-ltx25.freeze.txt): sage is a per-job backend on BOTH
    engines now, not a 2.3-only one. SDPA stays the default at generation time;
    sage is opt-in per job via `attention_backend`. Blackwell needs an R570+
    driver. Because nothing here is arch-dependent, this installer does not
    detect or take a GPU architecture at all.

.EXAMPLE
    ./scripts/install_ltx.ps1                       # full install
    ./scripts/install_ltx.ps1 -DryRun               # print the migration plan only
    ./scripts/install_ltx.ps1 -ResolveLatest        # fresh (unvalidated) engine resolve
    ./scripts/install_ltx.ps1 -SkipModels           # venvs only, no downloads
    ./scripts/install_ltx.ps1 -RunSmoke             # + mock GPU-free smoke test
    ./scripts/install_ltx.ps1 -BaseModel LTX25 -SkipVenv -SkipMigrate   # add ONE more base model (install-LTX25.bat)
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
    [switch] $SkipVenv,
    # Print the models/ migration plan (and the config.yaml lines that would be
    # rewritten) and exit 0 WITHOUT touching a single byte on disk.
    [switch] $DryRun,
    # Leave an existing models/ tree exactly as it is (no migration, no
    # config.yaml rewrite).
    [switch] $SkipMigrate,
    # Which manifests this RUN actually downloads and lists in the verification
    # table, selected by the 'id' field of scripts/manifests/*.json. The default
    # is exactly the set setup.bat ships -- the app plus the try-it-out LTX 2.3
    # plus the shared preprocessors (owner's ruling 2026-08-23,
    # Docs/MULTI_ENGINE_DESIGN.md §6.2). Any further base model is added by its
    # own install-<ID>.bat, which passes its id here. Loading, validation and the
    # migrate merge always cover EVERY manifest regardless of this switch.
    [string[]] $BaseModel = @('LTX23', 'Preprocessors')
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

$ModelsDir = "$ProjectRoot\models"
$ManifestDir = "$ProjectRoot\scripts\manifests"
$StagingRoot = "$ModelsDir\.dl"

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

# Size of ONE file (0 when absent / when the path is a directory).
function Get-FileSize([string] $absPath) {
    if (-not (Test-Path -LiteralPath $absPath)) { return [long] 0 }
    $item = Get-Item -LiteralPath $absPath -Force
    if ($item.PSIsContainer) { return [long] 0 }
    return [long] $item.Length
}

# Sum of the files DIRECTLY inside a directory (NOT recursive). The only
# consumer is the tokenizer `kind:"dir"` manifest row, which is gated as a whole
# because its ten files run from 35 B to 33 MB and the seven tiny ones are not
# individually catchable. Nothing else in this script measures a directory.
function Get-DirectChildSize([string] $absPath) {
    if (-not (Test-Path -LiteralPath $absPath)) { return [long] 0 }
    $item = Get-Item -LiteralPath $absPath -Force
    if (-not $item.PSIsContainer) { return [long] 0 }
    $sum = (Get-ChildItem -LiteralPath $absPath -File -Force -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) { return [long] 0 }
    return [long] $sum
}

# ----------------------------------------------------------------------------
# Path helpers. Every manifest path is stored with '/' separators and is
# relative to models/. We normalise to '/' before any comparison so that a
# Windows '\' path and a manifest path always compare the same way.
# ----------------------------------------------------------------------------
function ConvertTo-SlashPath([string] $p) { return $p.Replace('\', '/') }

# Reject anything that could escape models/ or hard-code a drive.
function Test-RelPathSafe([string] $p) {
    if ([string]::IsNullOrWhiteSpace($p)) { return $false }
    if ($p -match '\\') { return $false }
    if ($p -match '^[A-Za-z]:') { return $false }
    if ($p.StartsWith('/')) { return $false }
    foreach ($seg in ($p -split '/')) {
        if ($seg -eq '' -or $seg -eq '.' -or $seg -eq '..') { return $false }
    }
    return $true
}

# PATH-SEPARATOR-BOUNDARY prefix test: "ltx-2.3" matches "ltx-2.3/x" but NEVER
# "ltx-2.3-gguf/x". This boundary rule is what lets the migrate table carry both
# `ltx-2.3` and `ltx-2.3-gguf` as independent entries. Case-insensitive on
# purpose: NTFS is, so an on-disk "preprocessors" and a manifest "Preprocessors"
# name the same directory.
function Test-UnderPrefix([string] $rel, [string] $prefix) {
    if ($rel -eq $prefix) { return $true }
    return $rel.StartsWith($prefix + '/', [System.StringComparison]::OrdinalIgnoreCase)
}

# LONGEST-MATCH-WINS remap. Returns the new models-relative path, or $null when
# no entry claims this file. The remainder after the matched prefix is carried
# over verbatim, which is why sub-folders nobody configured (VAE/prunavaed/,
# IC-LoRA/pixel-spatial-upscaler/'s user-supplied x4, every LoRA thumbnail)
# survive with no per-file rules at all.
function Resolve-MapTarget {
    param(
        [Parameter(Mandatory)] [string] $Rel,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Pairs
    )
    $best = $null
    foreach ($m in $Pairs) {
        if (Test-UnderPrefix $Rel $m.from) {
            if (($null -eq $best) -or ($m.from.Length -gt $best.from.Length)) { $best = $m }
        }
    }
    if ($null -eq $best) { return $null }
    if ($Rel.Length -le $best.from.Length) { return $best.to }
    return ($best.to + '/' + $Rel.Substring($best.from.Length + 1))
}

# ----------------------------------------------------------------------------
# 0) Manifests: load + validate.
#
#    scripts/manifests/*.json is the ONE place that knows the layout. Each file
#    declares, for one base model (or the shared preprocessor set):
#      downloads[] : name / repo / include globs / map[] / files[]
#      migrate[]   : from -> to, for an existing pre-2026-08 tree
#    `files[]` is the single source of truth used THREE times over -- as the
#    pre-download guard, as the post-download re-check, and as the step 7
#    verification table + INSTALLED_PATHS.txt. Adding a base model later
#    (LTX 2.5, Wan 2.x, ...) is a new JSON file and no change here.
#
#    Files are processed in FILENAME order, so the numeric prefixes
#    (00-, 10-, ...) fix the order deterministically.
#
#    Validation is fail-loud and runs on EVERY invocation, before anything else
#    can act on a bad table: schema must be 1, every path must be relative and
#    inside models/, and every `files[].path` must live under one of that same
#    download's `map[].to` -- otherwise the remap would drop the file somewhere
#    the guard never looks and the install would loop forever.
# ----------------------------------------------------------------------------
function Import-ModelManifests {
    param([Parameter(Mandatory)] [string] $Dir)

    if (-not (Test-Path -LiteralPath $Dir)) { throw "Manifest directory not found: $Dir" }
    $files = @(Get-ChildItem -LiteralPath $Dir -Filter "*.json" -File | Sort-Object Name)
    if ($files.Count -eq 0) { throw "No manifests found in $Dir (expected at least one *.json)." }

    $loaded = @()
    foreach ($f in $files) {
        # ReadAllText with an explicit UTF8 encoding still honours a BOM, so the
        # manifests can be saved either way without breaking ConvertFrom-Json
        # (a leading U+FEFF is a parse error).
        $raw = [System.IO.File]::ReadAllText($f.FullName, [System.Text.Encoding]::UTF8)
        try {
            $obj = $raw | ConvertFrom-Json
        } catch {
            throw "Manifest $($f.Name) is not valid JSON: $($_.Exception.Message)"
        }
        Test-ManifestShape -Manifest $obj -Name $f.Name
        $loaded += [pscustomobject]@{ Name = $f.Name; Data = $obj }
    }
    return $loaded
}

function Test-ManifestShape {
    param(
        [Parameter(Mandatory)] $Manifest,
        [Parameter(Mandatory)] [string] $Name
    )
    if ($Manifest.schema -ne 1 -and $Manifest.schema -ne 2) {
        throw "Manifest ${Name}: unsupported schema '$($Manifest.schema)' (this installer understands schema 1 or 2)."
    }
    if ([string]::IsNullOrWhiteSpace($Manifest.id)) { throw "Manifest ${Name}: missing 'id'." }
    if ($Manifest.id -notmatch '^[A-Za-z0-9._-]+$') {
        throw "Manifest ${Name}: 'id' must be a plain filename-safe token (it names the staging directory)."
    }
    # schema 1 keeps the original strict rule (a schema-1 manifest with no
    # downloads is almost certainly a mistake). schema 2 base-model descriptors
    # are allowed an empty (or absent) 'downloads' -- a descriptor whose weights
    # are not distributed through this installer yet still needs to validate and
    # load so its 'categories'/'default_selection' are visible to the registry,
    # even though this run has nothing to fetch for it
    # (S-2 / F2, MULTI_ENGINE_DESIGN.md §4.3).
    if ($Manifest.schema -eq 1 -and -not $Manifest.downloads) {
        throw "Manifest ${Name}: missing 'downloads'."
    }

    $n = 0
    foreach ($dl in @($Manifest.downloads)) {
        $n++
        $where = "${Name} downloads[$n]"
        if ([string]::IsNullOrWhiteSpace($dl.name)) { throw "${where}: missing 'name'." }
        if ([string]::IsNullOrWhiteSpace($dl.repo)) { throw "${where}: missing 'repo'." }
        if (-not $dl.include) { throw "${where}: missing 'include'." }
        if (-not $dl.map) { throw "${where}: missing 'map'." }
        if (-not $dl.files) { throw "${where}: missing 'files'." }

        foreach ($m in @($dl.map)) {
            if (-not (Test-RelPathSafe $m.from)) { throw "${where}: map.from '$($m.from)' is not a safe relative path." }
            if (-not (Test-RelPathSafe $m.to)) { throw "${where}: map.to '$($m.to)' is not a safe relative path." }
        }
        foreach ($fl in @($dl.files)) {
            if (-not (Test-RelPathSafe $fl.path)) { throw "${where}: files.path '$($fl.path)' is not a safe relative path." }
            if ([string]::IsNullOrWhiteSpace($fl.label)) { throw "${where}: files.path '$($fl.path)' has no 'label'." }
            if ($null -eq $fl.min) { throw "${where}: files.path '$($fl.path)' has no 'min'." }
            if ($fl.kind -and ($fl.kind -ne 'dir')) { throw "${where}: files.path '$($fl.path)' has unknown kind '$($fl.kind)'." }
            $covered = $false
            foreach ($m in @($dl.map)) {
                if (Test-UnderPrefix $fl.path $m.to) { $covered = $true; break }
            }
            if (-not $covered) {
                throw "${where}: files.path '$($fl.path)' is not under any map.to of the same download. The remap would never place it there, so its guard could never be satisfied."
            }
        }
    }

    foreach ($mg in @($Manifest.migrate)) {
        if (-not (Test-RelPathSafe $mg.from)) { throw "${Name}: migrate.from '$($mg.from)' is not a safe relative path." }
        if (-not (Test-RelPathSafe $mg.to)) { throw "${Name}: migrate.to '$($mg.to)' is not a safe relative path." }
    }

    # A manifest that carries 'engine_family' is a BASE MODEL descriptor (as
    # opposed to a shared-asset manifest like 00-preprocessors.json, which has
    # no engine_family and is validated by the rules above only). See
    # MULTI_ENGINE_DESIGN.md §4.2/§4.3.
    if ($Manifest.engine_family) {
        Test-BaseModelShape -Manifest $Manifest -Name $Name
    }
}

# Validates the fields that ONLY a base-model descriptor (schema 2,
# 'engine_family' present) carries: display_name / engine_family / categories /
# assets / default_selection (MULTI_ENGINE_DESIGN.md §4.3). Called from
# Test-ManifestShape, never standalone, so $Manifest has already passed the
# schema/id/downloads/migrate checks above.
function Test-BaseModelShape {
    param(
        [Parameter(Mandatory)] $Manifest,
        [Parameter(Mandatory)] [string] $Name
    )
    if ([string]::IsNullOrWhiteSpace($Manifest.display_name)) {
        throw "Manifest ${Name}: missing 'display_name' (required when 'engine_family' is present)."
    }
    if ($Manifest.engine_family -notmatch '^[a-z0-9_]+$') {
        throw "Manifest ${Name}: 'engine_family' must match ^[a-z0-9_]+`$, got '$($Manifest.engine_family)'."
    }
    if (-not $Manifest.categories) {
        throw "Manifest ${Name}: missing 'categories' (required when 'engine_family' is present)."
    }

    # Every downloads[].files[].path in this SAME manifest, for the drift check
    # below (default_file / assets must name a file the downloads table actually
    # produces -- otherwise the descriptor and the download table have silently
    # diverged). Skipped entirely when downloads is empty: a descriptor with an
    # empty downloads[] has nothing to drift against.
    $hasDownloads = (@($Manifest.downloads)).Count -gt 0
    $filePaths = @{}
    foreach ($dl in @($Manifest.downloads)) {
        foreach ($fl in @($dl.files)) { $filePaths[[string] $fl.path] = $true }
    }

    $catNames = @($Manifest.categories.PSObject.Properties.Name)
    if ($catNames.Count -eq 0) {
        throw "Manifest ${Name}: 'categories' has no entries."
    }
    foreach ($catName in $catNames) {
        $cat = $Manifest.categories.$catName
        $where = "${Name} categories.$catName"
        if (-not $cat.scan -or @($cat.scan).Count -eq 0) {
            throw "${where}: 'scan' must be a non-empty array."
        }
        foreach ($s in @($cat.scan)) {
            if (-not (Test-RelPathSafe $s)) { throw "${where}: scan entry '$s' is not a safe relative path." }
        }
        if (-not $cat.extensions -or @($cat.extensions).Count -eq 0) {
            throw "${where}: 'extensions' must be a non-empty array."
        }
        foreach ($e in @($cat.extensions)) {
            if ($e -notmatch '^\.') { throw "${where}: extension '$e' must start with '.'." }
        }
        if ($cat.default_file) {
            if (-not (Test-RelPathSafe $cat.default_file)) {
                throw "${where}: 'default_file' is not a safe relative path."
            }
            if ($hasDownloads -and -not $filePaths.ContainsKey([string] $cat.default_file)) {
                throw "${where}: 'default_file' ('$($cat.default_file)') does not match any downloads[].files[].path in $Name -- descriptor drift."
            }
        }
    }

    if ($Manifest.assets) {
        foreach ($prop in @($Manifest.assets.PSObject.Properties)) {
            $val = [string] $prop.Value
            if (-not (Test-RelPathSafe $val)) {
                throw "${Name} assets.$($prop.Name): '$val' is not a safe relative path."
            }
            if ($hasDownloads -and -not $filePaths.ContainsKey($val)) {
                throw "${Name} assets.$($prop.Name): '$val' does not match any downloads[].files[].path in $Name -- descriptor drift."
            }
        }
    }

    if ($Manifest.default_selection) {
        foreach ($prop in @($Manifest.default_selection.PSObject.Properties)) {
            if ($catNames -notcontains $prop.Name) {
                throw "${Name} default_selection: key '$($prop.Name)' is not a declared category."
            }
        }
    }
}

Write-Step "Model manifests"
$Manifests = Import-ModelManifests -Dir $ManifestDir

# One line summarising the base models (schema 2, 'engine_family' present)
# this run knows about -- the ones that will populate the header dropdown
# (MULTI_ENGINE_DESIGN.md §1/§4.2). Shared-asset manifests (00-preprocessors,
# no engine_family) are counted in $Manifests.Count above but not listed here.
$BaseModelNames = @($Manifests | Where-Object { $_.Data.engine_family } | ForEach-Object { $_.Data.display_name })
Write-Ok "base models: $($BaseModelNames -join ', ')"

# Aggregate every manifest's migrate table into ONE list. A `from` may appear
# only once across ALL manifests: two base models both claiming the same old
# directory would make the destination depend on file order, which is exactly
# the kind of silent surprise a 74GB move must not have.
$MigrateAll = @()
$seenFrom = @{}
foreach ($mf in $Manifests) {
    foreach ($mg in @($mf.Data.migrate)) {
        $key = $mg.from.ToLowerInvariant()
        if ($seenFrom.ContainsKey($key)) {
            throw "Duplicate migrate.from '$($mg.from)' (in $($mf.Name) and $($seenFrom[$key])). Each legacy directory may have exactly one destination."
        }
        $seenFrom[$key] = $mf.Name
        $MigrateAll += [pscustomobject]@{ from = $mg.from; to = $mg.to }
    }
}
$manifestNames = ($Manifests | ForEach-Object { $_.Name }) -join ', '
Write-Ok "$($Manifests.Count) manifest(s) validated: $manifestNames  ($($MigrateAll.Count) migrate entries)"

# ----------------------------------------------------------------------------
# Which manifests THIS run downloads and verifies (-BaseModel, see param()).
# Everything above stays whole-set: loading, validation and the migrate merge
# always see every descriptor, because a partial view of those would let two
# manifests disagree without anyone noticing.
#
# The filter iterates $Manifests, NOT $BaseModel: the file-name order
# (00- -> 10- -> 20-) is what fixes the row order of the verification table, and
# setup.bat's table must not change shape just because the default value happens
# to list LTX23 before Preprocessors.
#
# -contains is case-insensitive by default, so `-BaseModel ltx25` works too.
# ----------------------------------------------------------------------------
$wanted = @($BaseModel | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
$knownIds = @($Manifests | ForEach-Object { [string] $_.Data.id })
# An empty selection must NOT be read as "nothing to check": with no manifests
# left, the table would shrink to its three fixed rows, every one of them would
# PASS, and the run would exit 0 having downloaded and verified no model at all.
# -BaseModel '' / @() / '  ' all land here.
if ($wanted.Count -eq 0) {
    throw "-BaseModel resolved to no ids. Known ids: $($knownIds -join ', ')"
}
$TargetManifests = @($Manifests | Where-Object { $wanted -contains [string] $_.Data.id })
$foundIds = @($TargetManifests | ForEach-Object { [string] $_.Data.id })
foreach ($w in $wanted) {
    if ($foundIds -notcontains $w) {
        throw "-BaseModel '$w' matches no manifest id in $ManifestDir. Known ids: $($knownIds -join ', ')"
    }
}
Write-Ok "-BaseModel: downloads + verification limited to $($foundIds -join ', ')"

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
# 2) Migrate an existing models/ tree to the base-model-first layout, then bring
#    config.yaml along with it.
#
#    Design (all four points are load-bearing):
#
#    (a) NOTHING MOVES UNTIL EVERY PRE-FLIGHT GUARD PASSES. The plan is built
#        with zero side effects, then checked (same volume, no reparse points,
#        no destination collisions), then printed, then executed. A 74GB tree
#        must never be left half-moved because the 40th file hit a surprise.
#
#    (b) MOVE, NEVER OVERWRITE. Move-Item runs WITHOUT -Force here. A
#        destination that already exists is a collision the user must resolve --
#        it is by definition a file we did not put there. (The download remap in
#        step 6 is the deliberate opposite; see the asymmetry note there.)
#
#    (c) IDEMPOTENT + CRASH-SAFE. The log is appended one line per file as the
#        move happens, so an interrupted run leaves an accurate record; a re-run
#        simply re-plans whatever is still in the old place and moves the rest.
#        Files that already sit under a destination prefix are recognised as
#        DONE, not re-moved into a doubled path.
#
#    (d) NTFS CASE COLLISION (found the hard way on the reference machine).
#        Windows resolves models\Preprocessors and models\preprocessors to the
#        SAME directory, so a repo that ships models/Preprocessors/... lands its
#        files inside the user's existing lowercase models/preprocessors/ and the
#        new tree quietly never appears. -eq cannot see this (PowerShell string
#        comparison is case-insensitive); -ceq can. So before ANY content moves
#        we compare each destination's top-level component against the real
#        on-disk directory names with -ceq and, on a case-only mismatch, fix the
#        NAME with a two-stage rename (X -> __case_fix_* -> X). Two stages are
#        required: a direct rename to a case-variant of the same name is a no-op
#        on NTFS. No file inside is touched -- this is a directory-entry rename,
#        instantaneous even for 74GB.
# ----------------------------------------------------------------------------

# The destination prefixes, used to recognise "already migrated" files.
$MigrateTos = @($MigrateAll | ForEach-Object { $_.to })
# Distinct top-level component of every destination (LTX23, Preprocessors, ...).
$MigrateTopDirs = @($MigrateTos | ForEach-Object { ($_ -split '/')[0] } | Select-Object -Unique)

# Case-only mismatches between what we want and what is on disk (see (d)).
function Get-CaseFixPlan {
    param([Parameter(Mandatory)] [string[]] $WantNames)
    $out = @()
    if (-not (Test-Path -LiteralPath $ModelsDir)) { return $out }
    $onDisk = @(Get-ChildItem -LiteralPath $ModelsDir -Directory -Force -ErrorAction SilentlyContinue)
    foreach ($want in $WantNames) {
        foreach ($d in $onDisk) {
            # -eq  : same name ignoring case  -> NTFS says these are one directory
            # -ceq : same name including case -> nothing to fix
            if (($d.Name -eq $want) -and -not ($d.Name -ceq $want)) {
                $out += [pscustomobject]@{ Old = $d.Name; New = $want }
            }
        }
    }
    return $out
}

function Invoke-CaseFix {
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Plan)
    foreach ($c in $Plan) {
        $tmp = "__case_fix_" + [guid]::NewGuid().ToString("N").Substring(0, 8)
        Rename-Item -LiteralPath (Join-Path $ModelsDir $c.Old) -NewName $tmp
        Rename-Item -LiteralPath (Join-Path $ModelsDir $tmp) -NewName $c.New
    }
}

# Build the move plan. Pure: reads the tree, writes nothing.
#   MOVE    - matched a migrate.from, destination differs
#   DONE    - already under a migrate.to prefix (or already at its target)
#   DROP    - inside a .cache/ segment (HuggingFace download bookkeeping); the
#             .cache directories themselves are dropped too (DropDirs)
#   KEEP    - matched nothing; left exactly where it is
function New-MigrationPlan {
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $CaseFix)

    $moves = @(); $drops = @(); $dropDirs = @(); $keeps = @(); $done = 0
    if (-not (Test-Path -LiteralPath $ModelsDir)) {
        return [pscustomobject]@{ Moves = $moves; Drops = $drops; DropDirs = $dropDirs; Keeps = $keeps; Done = $done }
    }
    # Map an on-disk top-level name to the name it will have after the case fix,
    # so a -DryRun (which renames nothing) still prints the real destination.
    $rename = @{}
    foreach ($c in $CaseFix) { $rename[$c.Old] = $c.New }

    $all = @(Get-ChildItem -LiteralPath $ModelsDir -Recurse -File -Force -ErrorAction SilentlyContinue)
    foreach ($f in $all) {
        $rel = ConvertTo-SlashPath $f.FullName.Substring($ModelsDir.Length + 1)
        $segs = $rel -split '/'
        if ($rename.ContainsKey($segs[0])) {
            $segs[0] = $rename[$segs[0]]
            $rel = $segs -join '/'
        }
        # The staging area is step 6's business, never migration's.
        if ($segs[0] -eq '.dl') { continue }
        # HuggingFace leaves .cache/huggingface/ next to every download: lock and
        # .metadata bookkeeping for a cache root we are about to invalidate by
        # moving the payload. It is regenerated on demand, so it is dropped
        # rather than carried into the new tree.
        if ($segs -contains '.cache') { $drops += $rel; continue }

        $atDest = $false
        foreach ($t in $MigrateTos) { if (Test-UnderPrefix $rel $t) { $atDest = $true; break } }
        if ($atDest) { $done++; continue }

        $target = Resolve-MapTarget -Rel $rel -Pairs $MigrateAll
        if ($null -eq $target) { $keeps += $rel; continue }
        if ($target -eq $rel) { $done++; continue }

        $moves += [pscustomobject]@{
            Rel    = $rel
            Target = $target
            Src    = $f.FullName
            Dst    = (Join-Path $ModelsDir ($target -replace '/', '\'))
            Size   = [long] $f.Length
        }
    }
    # The bookkeeping DIRECTORIES go as well, not just the files inside them.
    # Dropping only the files leaves an empty .cache/huggingface/download/ husk,
    # which keeps the legacy source root non-empty and therefore un-removable by
    # Remove-EmptyLegacyDirs -- observed in the field as an empty
    # models/ltx-2.3-components/ surviving a fully successful migration.
    # This is the ONLY -Recurse deletion in the migration, and it is aimed at a
    # directory whose entire contents are droppable by definition.
    foreach ($d in @(Get-ChildItem -LiteralPath $ModelsDir -Recurse -Directory -Force -ErrorAction SilentlyContinue)) {
        $rel = ConvertTo-SlashPath $d.FullName.Substring($ModelsDir.Length + 1)
        $segs = $rel -split '/'
        if ($rename.ContainsKey($segs[0])) {
            $segs[0] = $rename[$segs[0]]
            $rel = $segs -join '/'
        }
        if ($segs[0] -eq '.dl') { continue }
        # Only the topmost .cache of a chain: the recursive delete takes the rest.
        if ($segs[$segs.Count - 1] -ne '.cache') { continue }
        $nested = $false
        for ($i = 0; $i -lt ($segs.Count - 1); $i++) { if ($segs[$i] -eq '.cache') { $nested = $true; break } }
        if ($nested) { continue }
        $dropDirs += $rel
    }

    return [pscustomobject]@{ Moves = @($moves); Drops = @($drops); DropDirs = @($dropDirs); Keeps = @($keeps); Done = $done }
}

function Test-MigrationSafety {
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Moves)

    if (-not (Test-Path -LiteralPath $ModelsDir)) { return }

    # Same volume. Everything we move stays inside models/, so a cross-volume
    # move (= a silent 74GB COPY, hours instead of seconds, and a half-full disk)
    # can only sneak in through a junction/symlink pointing off-volume. Reject
    # every reparse point rather than trying to decide which ones are benign.
    $links = @(Get-ChildItem -LiteralPath $ModelsDir -Recurse -Directory -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.LinkType })
    $modelsItem = Get-Item -LiteralPath $ModelsDir -Force
    if ($modelsItem.LinkType) { $links = @($modelsItem) + $links }
    if ($links.Count -gt 0) {
        $names = ($links | ForEach-Object { "$($_.FullName) [$($_.LinkType)]" }) -join "; "
        throw "models/ contains junction(s)/symlink(s): $names. Migration refuses to run because a move across them can silently become a multi-hour copy onto another volume. Resolve them (or re-run with -SkipMigrate and move the files by hand) first."
    }

    # Destination collisions. NO -Force anywhere in migration, so an existing
    # destination is a hard stop: whatever is there is not ours to overwrite.
    $collide = @()
    foreach ($m in $Moves) {
        if (Test-Path -LiteralPath $m.Dst) { $collide += "$($m.Rel) -> $($m.Target)" }
    }
    # ...including collisions the plan would create with itself.
    $byTarget = @{}
    foreach ($m in $Moves) {
        $k = $m.Target.ToLowerInvariant()
        if ($byTarget.ContainsKey($k)) { $collide += "$($byTarget[$k]) and $($m.Rel) both -> $($m.Target)" }
        $byTarget[$k] = $m.Rel
    }
    if ($collide.Count -gt 0) {
        throw "Migration would overwrite existing files, which it never does. Resolve these by hand and re-run: $($collide -join '; ')"
    }

    # MAX_PATH. Not fatal (this project is normally installed near a drive root
    # and the new layout is SHORTER than the old one), but worth saying out loud.
    foreach ($m in $Moves) {
        if ($m.Dst.Length -gt 250) {
            Write-Warning "Destination path is $($m.Dst.Length) characters, close to the classic 260-character limit: $($m.Dst)"
        }
    }
}

# Bottom-up removal of directories the migration emptied. -Recurse is NEVER used
# here (it would delete a directory that still holds something we failed to
# move); only genuinely empty directories go, deepest first, and only inside the
# legacy source roots. The single -Recurse in this script is the .cache drop,
# which runs before this and is what makes those roots empty in the first place. Any directory that is a destination -- or an ancestor of one --
# is protected outright.
function Remove-EmptyLegacyDirs {
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Migrate)

    $protect = @{}
    foreach ($m in $Migrate) {
        $acc = $ModelsDir
        foreach ($seg in ($m.to -split '/')) {
            $acc = Join-Path $acc $seg
            $protect[$acc.ToLowerInvariant()] = $true
        }
    }
    $protect[$ModelsDir.ToLowerInvariant()] = $true

    $removed = @()
    foreach ($m in $Migrate) {
        $root = Join-Path $ModelsDir ($m.from -replace '/', '\')
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $cands = @(Get-ChildItem -LiteralPath $root -Recurse -Directory -Force -ErrorAction SilentlyContinue |
            Sort-Object { $_.FullName.Length } -Descending)
        $cands += @(Get-Item -LiteralPath $root -Force)
        # ...and then the ancestors of the source root, up to but never including
        # models/ itself. A migrate.from can be two levels deep
        # (ltx-2.3-components/vae), and without this the now-empty
        # models/ltx-2.3-components/ would be left behind for every user.
        $anc = Split-Path -Parent $root
        while ($anc -and ($anc.Length -gt $ModelsDir.Length) -and $anc.StartsWith($ModelsDir, [System.StringComparison]::OrdinalIgnoreCase)) {
            if (Test-Path -LiteralPath $anc) { $cands += @(Get-Item -LiteralPath $anc -Force) }
            $anc = Split-Path -Parent $anc
        }
        foreach ($d in $cands) {
            if ($protect.ContainsKey($d.FullName.ToLowerInvariant())) { continue }
            if (-not (Test-Path -LiteralPath $d.FullName)) { continue }
            $n = @(Get-ChildItem -LiteralPath $d.FullName -Force -ErrorAction SilentlyContinue).Count
            if ($n -eq 0) {
                Remove-Item -LiteralPath $d.FullName -Force
                $removed += (ConvertTo-SlashPath $d.FullName.Substring($ModelsDir.Length + 1))
            }
        }
    }
    return $removed
}

# ----------------------------------------------------------------------------
# config.yaml follow-up.
#
# WHY THIS IS NOT OPTIONAL: the six ic_loras entries (and lora_dir) exist ONLY
# in config.yaml. If they still point at the emptied legacy directories after a
# migration, picking an adapter 404s. The weight paths that used to be here too
# (spatial_upsampler_path, gemma_root, ...) moved into the base-model
# descriptors in §3-97 P3b and are migrated by the descriptor's own `migrate`
# table instead, but rewriting the surviving config paths is still done in the
# same breath as the move.
#
# Scope discipline: a config.yaml.bak is written first; only path tokens that
# start with models/ AND match the migrate table are touched; comments are left
# alone (a '#' outside quotes ends the live part of the line); every other
# value, key, blank line and byte is preserved, as is the file's BOM state.
#
# The one other config edit is the deprecated-key sweep below.
# ----------------------------------------------------------------------------

# The model.* keys config.py stopped honouring in the §3-97 P3b move of the
# weight paths into the base-model descriptors. MUST stay identical to
# config.py's DEPRECATED_MODEL_KEYS. A config.yaml that still carries one boots
# fine and generates fine -- the value is simply ignored -- but config.py logs
# one WARNING per leftover key on EVERY start, which is exactly the noise a
# person re-running setup.bat should stop seeing. The WARNING itself stays in
# config.py for the environments that never re-run the installer.
$DeprecatedModelKeys = @(
    'gguf_transformer_path',
    'gguf_gemma_path',
    'component_video_vae_path',
    'component_audio_vae_path',
    'component_text_projection_path',
    'component_video_vae_pruned_path',
    'spatial_upsampler_path',
    'gemma_root'
)

# Delete whole lines carrying a deprecated model.* key -- that line only,
# trailing inline comment included. Standalone comment lines around it (which
# may explain neighbouring keys too) are never touched, and neither is any key
# outside the top-level `model:` block.
function Remove-DeprecatedModelKeys {
    param(
        [Parameter(Mandatory)] [string] $Text,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Keys
    )
    if ($Keys.Count -eq 0) { return [pscustomobject]@{ Text = $Text; Removals = @() } }

    # Split so that each element KEEPS its own line terminator (zero-width
    # lookbehind). Rejoining the survivors with '' therefore reproduces the
    # file byte-for-byte apart from the deleted lines -- CRLF stays CRLF, LF
    # stays LF, and a missing final newline stays missing.
    # The CR half of a CRLF must NOT be a split point (that would cut the pair
    # in two and leave the orphaned LF behind as a blank line when the CR half
    # is deleted), hence: after any LF, or after a CR that no LF follows.
    $lines = [regex]::Split($Text, '(?<=\n)|(?<=\r)(?!\n)')
    $keyRx = [regex] ('^\s+(?:' + (($Keys | ForEach-Object { [regex]::Escape($_) }) -join '|') + ')\s*:')

    $kept = New-Object System.Collections.Generic.List[string]
    $removals = @()
    $inModel = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        $content = $line -replace '(\r\n|\n|\r)$', ''
        # Block tracking: `model:` at column 0 opens it, the next column-0 key
        # closes it. Column-0 comments and blank lines are neutral (a comment
        # between two model keys must not end the block).
        if ($content -match '^model\s*:') { $inModel = $true }
        elseif ($content -match '^[^\s#]') { $inModel = $false }

        if ($inModel -and $keyRx.IsMatch($content)) {
            $removals += [pscustomobject]@{ Line = ($i + 1); Text = $content.Trim() }
            continue
        }
        $kept.Add($line)
    }
    return [pscustomobject]@{ Text = ($kept -join ''); Removals = @($removals) }
}

function Convert-ConfigModelPaths {
    param(
        [Parameter(Mandatory)] [string] $Text,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Migrate
    )
    # A models/ path token: optional "./" or ".\", then models, then a separator,
    # then anything that is not a quote, whitespace or '#'. Because the token
    # cannot contain whitespace it can never span a line, so the replacement
    # leaves every CR/LF exactly where it was.
    $pattern = '(?:\.[\\/])?models[\\/][^"\x27\s#]+'

    $evaluator = {
        param($mt)
        $orig = $mt.Value

        # Is this token inside a comment? Walk the line prefix tracking double
        # quotes; a '#' seen outside quotes means everything after it is a
        # comment and must not be rewritten.
        $ls = 0
        if ($mt.Index -gt 0) {
            $ls = $Text.LastIndexOfAny([char[]]@("`n", "`r"), $mt.Index - 1) + 1
        }
        $prefix = $Text.Substring($ls, $mt.Index - $ls)
        $inQuote = $false
        for ($i = 0; $i -lt $prefix.Length; $i++) {
            $ch = $prefix[$i]
            if ($ch -eq '"') { $inQuote = -not $inQuote }
            elseif (($ch -eq '#') -and (-not $inQuote)) { return $orig }
        }

        $norm = ConvertTo-SlashPath $orig
        $lead = ''
        if ($norm.StartsWith('./')) { $lead = './'; $norm = $norm.Substring(2) }
        if (-not $norm.StartsWith('models/')) { return $orig }
        $rel = $norm.Substring('models/'.Length)
        $target = Resolve-MapTarget -Rel $rel -Pairs $Migrate
        if ($null -eq $target) { return $orig }
        return ($lead + 'models/' + $target)
    }

    $newText = [regex]::Replace($Text, $pattern, $evaluator)

    $oldLines = $Text -split "`r`n|`n|`r"
    $newLines = $newText -split "`r`n|`n|`r"
    $changes = @()
    for ($i = 0; $i -lt [Math]::Min($oldLines.Count, $newLines.Count); $i++) {
        if ($oldLines[$i] -cne $newLines[$i]) {
            $changes += [pscustomobject]@{ Line = ($i + 1); Old = $oldLines[$i].Trim(); New = $newLines[$i].Trim() }
        }
    }

    # The deprecated-key sweep runs AFTER the diff above so the reported line
    # numbers still refer to the file as it is on disk (the rewrite is
    # line-preserving; only this step changes the line count).
    $swept = Remove-DeprecatedModelKeys -Text $newText -Keys $DeprecatedModelKeys

    return [pscustomobject]@{ Text = $swept.Text; Changes = @($changes); Removals = @($swept.Removals) }
}

if ($SkipMigrate) {
    Write-Step "Migrate models/ to the base-model-first layout"
    Write-Skip "-SkipMigrate given"
    if ($DryRun) { Write-Host "`n-DryRun: nothing was changed." -ForegroundColor Cyan; exit 0 }
} else {
    Write-Step "Migrate models/ to the base-model-first layout"

    $caseFix = @(Get-CaseFixPlan -WantNames $MigrateTopDirs)
    $plan = New-MigrationPlan -CaseFix $caseFix

    $modelsVolume = "(models/ does not exist yet)"
    if (Test-Path -LiteralPath $ModelsDir) {
        $modelsVolume = [System.IO.Path]::GetPathRoot((Get-Item -LiteralPath $ModelsDir -Force).FullName)
    }
    $rootVolume = [System.IO.Path]::GetPathRoot($ProjectRoot)

    Write-Host ""
    Write-Host "  layout   : base-model-first (models/<BaseModel>/<Category>)"
    Write-Host "  volume   : project $rootVolume / models $modelsVolume  (same volume => rename, not copy)"
    foreach ($c in $caseFix) {
        Write-Host ("  CASEFIX  models\{0}  ->  models\{1}   (NTFS case-only rename, no file is touched)" -f $c.Old, $c.New) -ForegroundColor Yellow
    }

    $moveBytes = [long] 0
    foreach ($m in $plan.Moves) { $moveBytes += $m.Size }

    Write-Host ""
    Write-Host ("  MOVE {0} file(s), {1}" -f $plan.Moves.Count, (Format-Size $moveBytes))
    foreach ($m in ($plan.Moves | Sort-Object Rel)) {
        Write-Host ("    {0,10}  {1}  ->  {2}" -f (Format-Size $m.Size), $m.Rel, $m.Target)
    }
    if ($plan.Drops.Count -gt 0 -or $plan.DropDirs.Count -gt 0) {
        Write-Host ""
        Write-Host ("  DROP {0} HuggingFace .cache bookkeeping file(s) in {1} .cache directory/ies (regenerated on demand; the directories go too)" -f $plan.Drops.Count, $plan.DropDirs.Count) -ForegroundColor DarkGray
        foreach ($dd in ($plan.DropDirs | Sort-Object)) { Write-Host "    $dd" -ForegroundColor DarkGray }
    }
    if ($plan.Keeps.Count -gt 0) {
        Write-Host ""
        Write-Host ("  KEEP {0} file(s) that match no migrate entry (left exactly where they are):" -f $plan.Keeps.Count) -ForegroundColor DarkGray
        foreach ($k in ($plan.Keeps | Sort-Object)) { Write-Host "    $k" -ForegroundColor DarkGray }
    }
    if ($plan.Done -gt 0) {
        Write-Host ""
        Write-Host ("  DONE {0} file(s) already in the new layout" -f $plan.Done) -ForegroundColor DarkGray
    }

    # config.yaml preview / rewrite is prepared here so -DryRun can show it too.
    $configPath = "$ProjectRoot\config.yaml"
    $configResult = $null
    $configHasBom = $false
    if (Test-Path -LiteralPath $configPath) {
        $bytes = [System.IO.File]::ReadAllBytes($configPath)
        $configHasBom = ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)
        $cfgText = [System.Text.Encoding]::UTF8.GetString($bytes)
        if ($configHasBom) { $cfgText = $cfgText.Substring(1) }
        $configResult = Convert-ConfigModelPaths -Text $cfgText -Migrate $MigrateAll
    }

    Write-Host ""
    if ($null -eq $configResult) {
        Write-Host "  config.yaml : not present (nothing to rewrite)" -ForegroundColor DarkGray
    } elseif ($configResult.Changes.Count -eq 0 -and $configResult.Removals.Count -eq 0) {
        Write-Host "  config.yaml : no legacy model paths or retired keys found (nothing to rewrite)" -ForegroundColor DarkGray
    } else {
        Write-Host ("  config.yaml : {0} line(s) to rewrite, {1} retired key line(s) to delete (a config.yaml.bak is written first)" -f $configResult.Changes.Count, $configResult.Removals.Count)
        foreach ($c in $configResult.Changes) {
            Write-Host ("    line {0,4}  - {1}" -f $c.Line, $c.Old) -ForegroundColor DarkGray
            Write-Host ("    line {0,4}  + {1}" -f $c.Line, $c.New) -ForegroundColor Green
        }
        foreach ($r in $configResult.Removals) {
            Write-Host ("    line {0,4}  DEL {1}" -f $r.Line, $r.Text) -ForegroundColor Yellow
        }
    }

    if ($DryRun) {
        Write-Host ""
        Write-Host "-DryRun: plan only. Nothing on disk was read-modified, renamed, moved or deleted." -ForegroundColor Cyan
        exit 0
    }

    if (($caseFix.Count -eq 0) -and ($plan.Moves.Count -eq 0) -and ($plan.Drops.Count -eq 0) -and
        ($plan.DropDirs.Count -eq 0) -and
        ($null -eq $configResult -or ($configResult.Changes.Count -eq 0 -and $configResult.Removals.Count -eq 0))) {
        Write-Host ""
        Write-Skip "models/ is already in the base-model-first layout"
    } else {
        # Pre-flight: every guard passes before the first byte moves.
        Test-MigrationSafety -Moves $plan.Moves

        # The case fix has to happen before content moves, and it invalidates the
        # absolute source paths captured in the plan (models\preprocessors\x.pt
        # becomes models\Preprocessors\x.pt). NTFS resolves both spellings to the
        # same file, so the captured paths keep working -- but re-plan anyway so
        # the log records exactly what is on disk.
        if ($caseFix.Count -gt 0) {
            Invoke-CaseFix -Plan $caseFix
            $plan = New-MigrationPlan -CaseFix @()
            Test-MigrationSafety -Moves $plan.Moves
        }

        New-Item -ItemType Directory -Force -Path "$ProjectRoot\logs" | Out-Null
        $migLog = "$ProjectRoot\logs\model_migration_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
        # One Add-Content call per file: it opens, writes and closes, so the line
        # is on disk before the next move starts. A crash mid-migration therefore
        # leaves a log that is accurate to the last completed move -- which is
        # what a manual reverse-replay needs.
        Add-Content -LiteralPath $migLog -Value "# models/ migration to the base-model-first layout" -Encoding utf8
        Add-Content -LiteralPath $migLog -Value "# started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  root=$ProjectRoot" -Encoding utf8
        foreach ($c in $caseFix) {
            Add-Content -LiteralPath $migLog -Value ("CASEFIX`tmodels/{0}`tmodels/{1}" -f $c.Old, $c.New) -Encoding utf8
        }

        $moved = 0
        foreach ($m in $plan.Moves) {
            $parent = Split-Path -Parent $m.Dst
            if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
            # NO -Force. See design note (b): a pre-existing destination is a
            # collision Test-MigrationSafety already refused, and if one appears
            # between plan and execution we want the throw, not an overwrite.
            Move-Item -LiteralPath $m.Src -Destination $m.Dst
            Add-Content -LiteralPath $migLog -Value ("MOVE`t{0}`t{1}`t{2}" -f $m.Rel, $m.Target, $m.Size) -Encoding utf8
            $moved++
        }

        $dropped = 0
        foreach ($d in $plan.Drops) {
            $abs = Join-Path $ModelsDir ($d -replace '/', '\')
            if (Test-Path -LiteralPath $abs) {
                Remove-Item -LiteralPath $abs -Force
                Add-Content -LiteralPath $migLog -Value ("DROP`t{0}" -f $d) -Encoding utf8
                $dropped++
            }
        }

        # The .cache directories themselves, after their files are logged. This
        # runs BEFORE Remove-EmptyLegacyDirs so the legacy source roots are
        # genuinely empty by the time it looks at them.
        $droppedDirs = 0
        foreach ($dd in $plan.DropDirs) {
            $abs = Join-Path $ModelsDir ($dd -replace '/', '\')
            if (Test-Path -LiteralPath $abs) {
                Remove-Item -LiteralPath $abs -Recurse -Force
                Add-Content -LiteralPath $migLog -Value ("DROPDIR`t{0}" -f $dd) -Encoding utf8
                $droppedDirs++
            }
        }

        $removedDirs = Remove-EmptyLegacyDirs -Migrate $MigrateAll
        foreach ($rd in $removedDirs) {
            Add-Content -LiteralPath $migLog -Value ("RMDIR`t{0}" -f $rd) -Encoding utf8
        }

        if ($null -ne $configResult -and ($configResult.Changes.Count -gt 0 -or $configResult.Removals.Count -gt 0)) {
            Copy-Item -LiteralPath $configPath -Destination "$configPath.bak" -Force
            # WriteAllText with an explicit UTF8Encoding keeps the file's original
            # BOM state (Set-Content/Out-File would not) and writes nothing else.
            $enc = New-Object System.Text.UTF8Encoding($configHasBom)
            [System.IO.File]::WriteAllText($configPath, $configResult.Text, $enc)
            Add-Content -LiteralPath $migLog -Value ("CONFIG`tconfig.yaml`t{0} line(s) rewritten, {1} retired key line(s) deleted, backup at config.yaml.bak" -f $configResult.Changes.Count, $configResult.Removals.Count) -Encoding utf8
            foreach ($r in $configResult.Removals) {
                Add-Content -LiteralPath $migLog -Value ("CONFIGDEL`tconfig.yaml`tline {0}`t{1}" -f $r.Line, $r.Text) -Encoding utf8
            }
            Write-Ok "config.yaml rewritten ($($configResult.Changes.Count) path line(s), $($configResult.Removals.Count) retired key line(s) deleted); previous version saved as config.yaml.bak"
        }

        Add-Content -LiteralPath $migLog -Value "# finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding utf8
        Write-Ok "moved $moved file(s), dropped $dropped cache file(s) in $droppedDirs .cache dir(s), removed $($removedDirs.Count) empty legacy dir(s)"
        Write-Host "  log: $migLog"
    }
}

# ----------------------------------------------------------------------------
# 3) uv-managed Python 3.12  (skip if an in-project cpython-3.12 is already present)
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
# 4) App venv .venv  (torch-FREE; plain `uv sync` of root pyproject.toml)
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
# 5) Engine venvs  (torch cu128 stacks)
#
#    TWO of them, built by ONE shared Ensure-EngineVenv:
#      .venv-engine        LTX 2.3 -- transformers 4.57.6, sageattention, diffusers
#      .venv-engine-ltx25  LTX 2.5 -- transformers 5.x (Gemma 4), official LTX-2
#                                     v1.2.0, sageattention, no diffusers
#    They are siblings, not versions of each other: transformers 4.57 and 5.x
#    cannot coexist in one interpreter, so each engine gets its own. Neither
#    worker can import the other's stack, which is exactly the isolation the
#    multi-engine design relies on.
#
#    There is NO committed uv.lock -> we must NEVER use `uv sync --frozen`.
#    DEFAULT = deterministic FREEZE path: reproduces the VALIDATED torch
#    2.9.1+cu128 stacks (engine/venv-engine.freeze.txt and
#    engine25/venv-engine-ltx25.freeze.txt) that every verification run in this
#    project used. -ResolveLatest opts into a fresh resolve from the matching
#    *-venv-pyproject.toml (yields an UNVALIDATED newer torch).
#
#    hf.exe lives in the 2.3 venv, so that one must be created BEFORE the model
#    downloads. The 2.5 venv has no such ordering constraint -- it is built right
#    after, for symmetry and so one installer run leaves both engines runnable.
#
#    RE-SYNC (2026-07): this step no longer skips merely because a venv exists.
#    A `git pull` can move a freeze file or the pinned revs below, and the
#    documented update flow is "git pull, then re-run the installer" -- which
#    only works if the freeze is re-applied when the pinned set changes. The
#    applied state is recorded as a hash in <venv>/.nz-engine-state, which
#    doubles as the COMPLETION MARKER: an interrupted install leaves a half-built
#    venv with no marker, and the next run re-applies instead of trusting it.
#    Each venv carries its OWN marker, so a change to one never re-installs the
#    other.
# ----------------------------------------------------------------------------
$enginePy = "$ProjectRoot\.venv-engine\Scripts\python.exe"
$freezeSrc = "$ProjectRoot\engine\venv-engine.freeze.txt"
$enginePyprojectDir = "$ProjectRoot\engine"
# Applied-state hash + completion marker. It lives INSIDE .venv-engine, which
# .gitignore already excludes, so it is never tracked and dies with the venv.
$engineStateFile = "$ProjectRoot\.venv-engine\.nz-engine-state"

# The 3 git packages PLUS 1 direct-URL wheel, as uv requirement strings, all
# pinned exactly. The freeze file lists all 4 only as bare `name==version` (no
# such build exists on PyPI for any of them), so a plain `-r freeze` cannot
# fetch them -- they are installed from these direct references first, and
# their bare lines are filtered out of the freeze copy.
# Keeping them in ONE array is what lets the state hash below cover them: bump a
# rev/wheel URL here and the hash changes, which forces a re-apply on the next
# run.
#
# The sageattention entry is a prebuilt wheel (not a git pin): woct0rdho's
# Windows build for sageattention 2.2.0, matched to this project's exact
# torch 2.9.1+cu128 (a mismatched wheel fails the ABI-tagged import at load
# time, not at install time). `%2B` is the URL-encoded form of `+` inside the
# wheel filename's local version segment (`2.2.0+cu128torch2.9.1.post6`) --
# keep this string SINGLE-quoted so PowerShell does not try to interpolate it.
# Re-added 2026-07-31 (see .NOTES above and D4 in the Acceleration plan): this
# is the only supported source for sageattention -- engine-venv-pyproject.toml
# deliberately does NOT list it (see that file's header comment), because
# -ResolveLatest resolves an unvalidated newer torch this ABI-pinned wheel is
# not built against.
$engineDirectPins = @(
    "diffusers @ git+https://github.com/huggingface/diffusers.git@01de02e8b4f2cc91df4f3e91cb6535ebcbeb490c"
    "ltx-core @ git+https://github.com/Lightricks/LTX-2.git@00dc53d3f81c405932f9f16d9c57557de411e702#subdirectory=packages/ltx-core"
    "ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git@00dc53d3f81c405932f9f16d9c57557de411e702#subdirectory=packages/ltx-pipelines"
    'sageattention @ https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu128torch2.9.1.post6-cp310-abi3-win_amd64.whl'
)

# ---------------------------------------------------------------------------
# LTX 2.5 sibling venv (.venv-engine-ltx25). Same machinery, different pins.
# ---------------------------------------------------------------------------
$ltx25Venv = "$ProjectRoot\.venv-engine-ltx25"
$ltx25Py = "$ProjectRoot\.venv-engine-ltx25\Scripts\python.exe"
$ltx25FreezeSrc = "$ProjectRoot\engine25\venv-engine-ltx25.freeze.txt"
$ltx25PyprojectDir = "$ProjectRoot\engine25"
$ltx25StateFile = "$ProjectRoot\.venv-engine-ltx25\.nz-engine-state"

# The 2.5 direct-reference set. Unlike the 2.3 array above this one ALSO carries
# torch/torchaudio, and deliberately so: stage (a) is a real resolve, and if it
# ran without them uv would fetch whatever the cu128 index calls newest (~2.11,
# multiple GB) only for stage (b) to downgrade it back to 2.9.1. Pinning them
# here makes stage (a) land on the right build the first time; their bare lines
# are then filtered out of the freeze exactly like the git pins are.
#
# torchaudio's `+cu128` local version is NOT decoration. With torchaudio left
# unbounded, uv was measured resolving torchaudio 2.11, which then drags a
# matching torch in behind it and silently defeats the torch pin.
#
# The official LTX-2 packages are pinned by FULL SHA (v1.2.0 =
# d151147788a9284cca791edc6ce898007e727fe6) rather than by tag, so a moved tag
# cannot change what gets installed.
#
# sageattention is the SAME wheel URL as the 2.3 array above -- byte for byte
# the same string, deliberately, because it is the same wheel: cp310-abi3
# (one build serves every CPython >= 3.10, so the 2.5 venv's 3.12 is covered)
# against the same torch 2.9.1+cu128 both stacks pin. Added 2026-08-25 for the
# Acceleration third wave, which opens `attention_backend: "sage"` on the 2.5
# engine; before that the 2.5 venv had no sageattention at all. Its runtime JIT
# dependency, triton-windows, was already here (it arrived 2026-08-24 for the
# fused GGUF dequantisation kernels), so nothing else had to change.
# `%2B` is the URL-encoded `+` of the wheel's local version segment -- keep the
# line SINGLE-quoted here too, or PowerShell will try to interpolate it.
$ltx25DirectPins = @(
    "torch==2.9.1+cu128"
    "torchaudio==2.9.1+cu128"
    "ltx-core @ git+https://github.com/Lightricks/LTX-2.git@d151147788a9284cca791edc6ce898007e727fe6#subdirectory=packages/ltx-core"
    "ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git@d151147788a9284cca791edc6ce898007e727fe6#subdirectory=packages/ltx-pipelines"
    'sageattention @ https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu128torch2.9.1.post6-cp310-abi3-win_amd64.whl'
)

# Extra uv arguments the 2.5 stack needs and the 2.3 stack does not.
#
#   --no-sources  The 2.3 stack expresses its torch pin through a
#                 `[tool.uv.sources]` table. That pattern was tried for 2.5 and
#                 does NOT work here: the official LTX-2 packages carry their own
#                 uv index configuration (cu132) and the two collide during
#                 resolution. engine25-venv-pyproject.toml therefore has no
#                 sources table at all -- it uses PEP 508 direct URLs -- and
#                 --no-sources is what keeps the dependency-side tables out of
#                 the resolve as well.
#   --index / --index-strategy  supply the cu128 CUDA builds. Same
#                 unsafe-best-match as the 2.3 freeze stage, for the same known
#                 uv bug around the platform-marker-less torchaudio source.
$ltx25UvArgs = @(
    "--no-sources"
    "--index", "https://download.pytorch.org/whl/cu128"
    "--index-strategy", "unsafe-best-match"
)

# SHA-256 over the freeze body PLUS the direct pins. The freeze file alone is
# NOT a sufficient input: the pinned revs/URLs are hardcoded in this script, so
# a bump would otherwise leave the hash unchanged and never re-apply. Line
# endings are normalised first so a CRLF/LF checkout flip does not masquerade
# as a change.
function Get-EngineStateHash {
    param(
        [Parameter(Mandatory)] [string]   $FreezeFile,
        [Parameter(Mandatory)] [string[]] $DirectPins
    )
    $body = [System.IO.File]::ReadAllText($FreezeFile).Replace("`r`n", "`n")
    $payload = $body + "`n" + ($DirectPins -join "`n") + "`n"
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
#  (a) install the 3 git packages + 1 direct-URL wheel at their pinned refs (see
#      $engineDirectPins).
#  (b) install the remaining pinned wheels from a TEMP copy of the freeze that has
#      those 4 bare lines removed (already installed in (a); left in place uv
#      would fetch some other PyPI build of the same version -- and for
#      sageattention there IS no PyPI build at all).
# The cu128 --index and --index-strategy in (b) are BOTH load-bearing: without
# them uv resolves CPU-only torch/torchaudio wheels (known uv bug for the
# platform-marker-less torchaudio source).
#
# $DirectPinArgs / $DirectPinsLabel exist for the LTX 2.5 stack, whose stage (a)
# needs the cu128 index and --no-sources (see $ltx25UvArgs). Their DEFAULTS
# reproduce the 2.3 call byte for byte, so the 2.3 path is unchanged by their
# existence.
function Invoke-EngineFreezeApply {
    param(
        [Parameter(Mandatory)] [string]   $EnginePython,
        [Parameter(Mandatory)] [string]   $FreezeFile,
        [Parameter(Mandatory)] [string[]] $DirectPins,
        [string[]] $DirectPinArgs = @(),
        [string]   $DirectPinsLabel = "3 git pins + 1 wheel URL: diffusers / ltx-core / ltx-pipelines / sageattention"
    )
    Write-Do "install direct-reference packages ($DirectPinsLabel)"
    $directArgs = @("pip", "install", "--python", $EnginePython) + $DirectPinArgs + $DirectPins
    uv @directArgs
    if ($LASTEXITCODE -ne 0) { throw "engine direct-pin install failed." }

    # Distribution names taken from the pins themselves, so this filter cannot
    # drift out of sync with the list above. The split covers BOTH pin shapes:
    # "name @ <url>" (all four 2.3 pins) and a bare "name==version" (the 2.5
    # torch/torchaudio pins) -- taking only the leading distribution name in
    # either case. For the 2.3 pins the result is the same string it always was.
    $directNames = $DirectPins | ForEach-Object { [regex]::Escape((($_ -split '[\s=<>~!;\[]')[0])) }
    $gitLineRe = "^(" + ($directNames -join "|") + ")=="

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

# One engine venv, start to finish: the -SkipVenv / -ResolveLatest / freeze
# three-way, the create-vs-re-sync decision, and the state marker. Called once
# per engine (2.3, then 2.5) -- this used to be inline, and inlining it a second
# time for the 2.5 stack would have meant two copies of the marker logic that
# drift apart on the first fix.
#
# $Label is the venv's directory name relative to $ProjectRoot (the cwd this
# script pins at startup) and is the human name in every message; $VenvPath is
# the same directory absolute, and is what actually gets passed to `uv venv` so
# the call cannot be hurt by a future cwd change.
#
# $DirectPinArgs / $ResolveArgs carry the per-engine uv flag differences (see
# $ltx25UvArgs); their defaults reproduce the 2.3 behaviour exactly.
function Ensure-EngineVenv {
    param(
        [Parameter(Mandatory)] [string]   $VenvPath,
        [Parameter(Mandatory)] [string]   $PythonPath,
        [Parameter(Mandatory)] [string]   $StateFile,
        [Parameter(Mandatory)] [string]   $FreezeFile,
        [Parameter(Mandatory)] [string]   $PyprojectDir,
        [Parameter(Mandatory)] [string[]] $DirectPins,
        [Parameter(Mandatory)] [string]   $Label,
        [string[]] $DirectPinArgs = @(),
        [string]   $DirectPinsLabel = "3 git pins + 1 wheel URL: diffusers / ltx-core / ltx-pipelines / sageattention",
        [string[]] $ResolveArgs = @("--index-strategy", "unsafe-best-match")
    )
    if ($SkipVenv) {
        Write-Step "Engine venv $Label"
        Write-Skip "-SkipVenv given"
        return
    }

    if ($ResolveLatest) {
        # --------------------------------------------------------------------
        # Opt-in fresh resolve. --index-strategy unsafe-best-match is REQUIRED:
        # the pyproject's torchaudio source lacks a platform marker, and without
        # unsafe-best-match uv pulls a CPU-only torchaudio (known bug). This path
        # is UNVALIDATED (newer torch ~2.11 vs the verified 2.9.1). Unlike
        # before, an existing venv no longer makes the flag a no-op -- asking for
        # a fresh resolve now always performs one.
        # --------------------------------------------------------------------
        Write-Step "Engine venv $Label  (-ResolveLatest: fresh resolve)"
        if (-not (Test-Path $PythonPath)) {
            Write-Do "uv venv --python 3.12 $Label"
            uv venv --python 3.12 $VenvPath
            if ($LASTEXITCODE -ne 0) { throw "uv venv $Label failed." }
        }
        Write-Warning "-ResolveLatest: resolving the engine venv fresh from engine-venv-pyproject.toml."
        Write-Warning "This yields an UNVALIDATED newer torch (~2.11). All project verification ran on"
        Write-Warning "the frozen 2.9.1+cu128 stack (the default). Use only if you accept re-validating."
        Write-Do "uv pip install $($ResolveArgs -join ' ') <engine pyproject dir>"
        $resolveArgv = @("pip", "install", "--python", $PythonPath) + $ResolveArgs + @($PyprojectDir)
        uv @resolveArgv
        if ($LASTEXITCODE -ne 0) { throw "engine venv resolve (-ResolveLatest) failed." }
        # No state marker is written here, and a stale one is dropped: the
        # resulting venv is NOT the frozen stack, so claiming it matches would be
        # a lie. The absence is meaningful in both directions -- it records "this
        # venv came in by the other route", and it makes the next default run
        # re-pin it to the freeze.
        if (Test-Path $StateFile) { Remove-Item $StateFile -Force }
        Write-Ok "$Label ready (unvalidated resolve)"
        return
    }

    Write-Step "Engine venv $Label  (torch cu128 stack, deterministic freeze)"
    if (-not (Test-Path $FreezeFile)) { throw "Engine freeze file not found: $FreezeFile" }

    $wantState = Get-EngineStateHash -FreezeFile $FreezeFile -DirectPins $DirectPins
    $haveState = ""
    if (Test-Path $StateFile) {
        $haveState = ((Get-Content $StateFile -Raw) -replace '\s', '')
    }

    if (-not (Test-Path $PythonPath)) {
        Write-Do "uv venv --python 3.12 $Label"
        uv venv --python 3.12 $VenvPath
        if ($LASTEXITCODE -ne 0) { throw "uv venv $Label failed." }
        Invoke-EngineFreezeApply -EnginePython $PythonPath -FreezeFile $FreezeFile -DirectPins $DirectPins `
            -DirectPinArgs $DirectPinArgs -DirectPinsLabel $DirectPinsLabel
        Set-Content -Path $StateFile -Value $wantState -Encoding ascii
        Write-Ok "$Label ready"
    } elseif ($haveState -eq $wantState) {
        Write-Skip "$Label already matches the pinned dependency set"
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
        Invoke-EngineFreezeApply -EnginePython $PythonPath -FreezeFile $FreezeFile -DirectPins $DirectPins `
            -DirectPinArgs $DirectPinArgs -DirectPinsLabel $DirectPinsLabel
        Set-Content -Path $StateFile -Value $wantState -Encoding ascii
        Write-Ok "$Label re-synced"
    }
}

# LTX 2.3 -- FIRST, because hf.exe (used by the model downloads below) lives here.
Ensure-EngineVenv -VenvPath "$ProjectRoot\.venv-engine" -PythonPath $enginePy `
    -StateFile $engineStateFile -FreezeFile $freezeSrc -PyprojectDir $enginePyprojectDir `
    -DirectPins $engineDirectPins -Label ".venv-engine"

# LTX 2.5 -- the sibling stack. Same machinery, different pins and (per the
# M2 correction) the extra --no-sources / cu128-index flags on BOTH uv stages.
Ensure-EngineVenv -VenvPath $ltx25Venv -PythonPath $ltx25Py `
    -StateFile $ltx25StateFile -FreezeFile $ltx25FreezeSrc -PyprojectDir $ltx25PyprojectDir `
    -DirectPins $ltx25DirectPins -Label ".venv-engine-ltx25" `
    -DirectPinArgs $ltx25UvArgs `
    -DirectPinsLabel "2 torch pins + 2 git pins + 1 wheel pin: torch / torchaudio / ltx-core / ltx-pipelines / sageattention" `
    -ResolveArgs $ltx25UvArgs

# hf.exe (used by the model downloads below) must exist in the engine venv.
$hfExe = "$ProjectRoot\.venv-engine\Scripts\hf.exe"

# ----------------------------------------------------------------------------
# Attention backend: nothing EXTRA to install here, on any GPU.
#
# PyTorch SDPA is the always-available backend on every architecture, and this
# step (still) does not install anything for it -- not on any arch, not
# optionally. (Historically this step could pick up a prebuilt xformers wheel
# out of wheels/; that path was removed because such a wheel is compiled for
# ONE compute capability and installs cleanly on machines it cannot run on. If
# you want to experiment with xformers, build and install it by hand -- see
# scripts/build_xformers.ps1 -- and note the engine code does not import it.)
#
# sageattention, the optional second backend, is NOT installed here either --
# it is a pinned wheel in the $engineDirectPins / $ltx25DirectPins arrays above
# (step 5), alongside triton-windows (its runtime JIT dependency, in both engine
# freezes) which bundles its own TinyCC/ptxas and needs no Visual Studio on the
# end-user machine. It was removed as dead weight in the 2026-07-28 cleanup
# (PENDING_TASKS.md 3-25) and came back 2026-07-31 once the Acceleration
# feature's SageAttentionService gave it a real consumer. SDPA remains the
# default at generation time; sage is opt-in per job.
#
# sageattention is NOT a 2.3-only package any more either: as of 2026-08-25 the
# SAME wheel URL is pinned for BOTH venvs (Acceleration third wave, which opens
# `attention_backend: "sage"` on the 2.5 engine). One wheel serves both because
# it is cp310-abi3 -- a single build for every CPython >= 3.10, so .venv-engine's
# 3.12 and .venv-engine-ltx25's 3.12 take the identical file -- and because both
# stacks pin the identical torch 2.9.1+cu128 the wheel's ABI tag names. The two
# venvs still never import each other's packages; they simply install the same
# artefact from the same URL.
#
# triton-windows is NOT a 2.3-only package any more: as of 2026-08-24 it is
# pinned in BOTH engine freezes (engine/venv-engine.freeze.txt and
# engine25/venv-engine-ltx25.freeze.txt, same 3.5.1.post24). In the 2.5 venv it
# arrived for a reason unrelated to sage -- it is the runtime JIT for the fused
# GGUF K-quant dequantisation kernels (engine/gguf/dequant_triton_kernels.py),
# which the 2.5 transformer and text-encoder paths share with the 2.3 engine --
# and it now serves sage's JIT there as well, exactly as it does in .venv-engine.
# No installer CODE change was needed for that: the freeze is applied verbatim
# and the state-hash marker re-applies it on the next run.
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
# 6) Model downloads (~31GB) via the engine venv's hf.exe, driven by the
#    manifests. Everything comes from self-hosted repos that are PUBLIC and
#    NON-GATED, so no HuggingFace account, login or token is involved anywhere.
#
#    Per download entry:
#      guard  -- every `files[]` row must exist and be at least its own `min`.
#                All rows pass => SKIP. This is a PER-FILE test; see the guard
#                doctrine in .NOTES for why directory totals are gone.
#      stage  -- fetch into models\.dl\<manifest-id>-<n>\ (same volume as
#                models/, inside .gitignore's models/** block). Downloading
#                straight into the final tree is not possible any more: the
#                repos still ship the OLD directory names, so their contents
#                have to be remapped, and a staging area is what makes that a
#                pure rename instead of a merge into live user data.
#      remap  -- move each staged file to map[]'s destination.
#      verify -- re-test the same `files[]` rows; throw if anything is short.
#      clean  -- only after all of the above succeeds is the staging directory
#                removed. On failure it is LEFT IN PLACE so the next run resumes
#                the HuggingFace download instead of re-fetching 17GB.
#
#   FORCE ASYMMETRY -- READ BEFORE CHANGING EITHER SIDE.
#   The remap below moves with -Force; the step 2 migration moves WITHOUT it.
#   That is deliberate and the two must not be made to match:
#     * step 6 writes only files the manifest names as OFFICIAL, freshly
#       downloaded from the pinned repo. Overwriting is the POINT -- it is how a
#       truncated or corrupted official file gets replaced by re-running.
#     * step 2 moves the USER'S existing tree, including files nothing here has
#       ever produced (self-converted GGUFs, purchased LoRAs, an x4 upscaler that
#       does not exist in any repo). Overwriting there destroys unrecoverable
#       data, so a collision must stop the install instead.
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
#   The per-file guard below does catch the resulting short download, but the
#   thrown error gives no hint that a version bump is the cause -- look here
#   first.
#
#   NOTE (glob semantics, verified live against both repos): --include matches with
#   Python fnmatch against the repo-relative path, and `*` DOES cross '/'. So
#   "ltx-2.3-components/*" reaches the nested vae/ and text_encoders/ files two
#   levels down. Just as importantly, the repo-root card files (LICENSE /
#   NOTICE.md / README.md / .gitattributes) match NO "<dir>/*" pattern -- which is
#   what stops the repos' identically-named cards from landing in the staging
#   directory, where the remap would throw on them as unmapped.
# ----------------------------------------------------------------------------

# One `files[]` row -> its size on disk. `kind:"dir"` sums the DIRECT children;
# everything else is a plain file length.
function Get-ManifestEntrySize {
    param([Parameter(Mandatory)] $Entry)
    $abs = Join-Path $ModelsDir ($Entry.path -replace '/', '\')
    if ($Entry.kind -eq 'dir') { return (Get-DirectChildSize $abs) }
    return (Get-FileSize $abs)
}

# The rows that are absent or under their own min. EMPTY means "present".
function Get-ShortEntries {
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Files)
    $short = @()
    foreach ($fl in $Files) {
        $sz = Get-ManifestEntrySize -Entry $fl
        if ($sz -lt [long] $fl.min) {
            $short += "$($fl.path) ($(Format-Size $sz) < $(Format-Size ([long] $fl.min)))"
        }
    }
    return @($short)
}

if ($SkipModels) {
    Write-Step "Model downloads"
    Write-Skip "-SkipModels given"
} else {
    # No total size in this heading: what a run fetches now depends on
    # -BaseModel, and the manifests' `min` values are deliberately 4-10% under
    # the official sizes, so any number computed here would be wrong. The
    # per-batch estimate is printed by scripts/setup.ps1 / scripts/install_model.ps1.
    Write-Step "Model downloads (public repos, no token needed)"

    foreach ($mf in $TargetManifests) {
        $idx = 0
        foreach ($dl in @($mf.Data.downloads)) {
            $idx++
            $short = Get-ShortEntries -Files @($dl.files)
            if ($short.Count -eq 0) {
                $total = [long] 0
                foreach ($fl in @($dl.files)) { $total += (Get-ManifestEntrySize -Entry $fl) }
                Write-Skip "$($dl.name)  ($(Format-Size $total) already present)"
                continue
            }
            if (-not (Test-Path $hfExe)) {
                throw "hf.exe not found at $hfExe. The engine venv must be created first (do not pass -SkipVenv)."
            }

            $stage = Join-Path $StagingRoot ("{0}-{1}" -f $mf.Data.id, $idx)
            New-Item -ItemType Directory -Force -Path $stage | Out-Null

            # Single --include with ALL patterns (see nargs note above).
            $argv = @("download", $dl.repo)
            $argv += @("--include") + @($dl.include)
            $argv += @("--local-dir", $stage)

            Write-Do "$($dl.name)  download $($dl.repo)  (missing: $($short -join '; '))"
            & $hfExe @argv
            if ($LASTEXITCODE -ne 0) {
                throw "Download of '$($dl.name)' failed. The repo is public and needs no token, so check your network first, then the include globs against https://huggingface.co/$($dl.repo)/tree/main  (staging kept at $stage so the next run resumes)"
            }

            # Remap staging -> final layout.
            $staged = @(Get-ChildItem -LiteralPath $stage -Recurse -File -Force -ErrorAction SilentlyContinue)
            foreach ($sf in $staged) {
                $rel = ConvertTo-SlashPath $sf.FullName.Substring($stage.Length + 1)
                if (($rel -split '/') -contains '.cache') {
                    # HuggingFace bookkeeping for a cache root that dies with the
                    # staging directory. Dropped silently, on purpose.
                    Remove-Item -LiteralPath $sf.FullName -Force
                    continue
                }
                $target = Resolve-MapTarget -Rel $rel -Pairs @($dl.map)
                if ($null -eq $target) {
                    throw "'$($dl.name)' downloaded '$rel', which no map entry claims. Refusing to guess where it belongs -- widen the map (or narrow the include globs) in $($mf.Name). Staging kept at $stage."
                }
                $dst = Join-Path $ModelsDir ($target -replace '/', '\')
                $parent = Split-Path -Parent $dst
                if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
                # -Force HERE ONLY. See the force-asymmetry note above.
                Move-Item -LiteralPath $sf.FullName -Destination $dst -Force
            }

            $short = Get-ShortEntries -Files @($dl.files)
            if ($short.Count -gt 0) {
                throw "'$($dl.name)' downloaded but these expected files are missing or short: $($short -join '; '). Check the include globs and the map in $($mf.Name). Staging kept at $stage."
            }

            Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
            $total = [long] 0
            foreach ($fl in @($dl.files)) { $total += (Get-ManifestEntrySize -Entry $fl) }
            Write-Ok "$($dl.name)  ($(Format-Size $total))"
        }
    }

    # Drop the staging root when nothing is left in it (a kept staging directory
    # from a failed run must survive, so this is conditional).
    if (Test-Path -LiteralPath $StagingRoot) {
        if (@(Get-ChildItem -LiteralPath $StagingRoot -Force -ErrorAction SilentlyContinue).Count -eq 0) {
            Remove-Item -LiteralPath $StagingRoot -Force
        }
    }
}

# ----------------------------------------------------------------------------
# 7) Verification + regenerate models/INSTALLED_PATHS.txt
#
#    The model rows come STRAIGHT from the manifests' `files[]` -- the same
#    array that guards the downloads in step 6. There is no second list to keep
#    in sync any more: a file is guarded, verified and published in
#    INSTALLED_PATHS.txt from one declaration.
#
#    Three rows are the script's own, because they are not models:
#      engine_python / app_python  -- the two interpreters
#      engine worker.py            -- the engine entry point
#    They match what the engine adapter's LTXRunner._real_available() and
#    _RealBackend._require_path() demand (referenced by FUNCTION NAME only --
#    line numbers here went stale once already); every OTHER row comes from the
#    base-model descriptor, which is where those file paths now live (§3-97
#    P3b) -- _real_available() reads the same categories/assets this table is
#    generated from. The 46GB monolith is not gated:
#    it is no longer a config option at all (checkpoint_path was removed from
#    config.yaml 2026-07-28, PENDING_TASKS.md 3-26).
#
#    The IC-LoRA / preprocessor rows are a deliberate widening: _real_available()
#    does not look at them (their absence downgrades no backend to mock), but
#    config.yaml registers every ic_loras: entry unconditionally and
#    gradio_ui/adapters.py falls back to the same names even when nothing is
#    registered. A missing file there is invisible until a user picks the adapter
#    and gets a 404 -- which is exactly the failure this table converts into an
#    up-front, named MISSING. Because the guards are now per-file, a MISSING here
#    is ALWAYS cleared by re-running: the guard for that one file cannot be
#    satisfied by anything else, so the download runs.
# ----------------------------------------------------------------------------
Write-Step "Verification (required load-bearing artifacts)"

# Each row: label | project-relative path | isDir | minBytes (0 => existence-only)
$required = @(
    @{ Label = "engine_python";    Rel = ".venv-engine/Scripts/python.exe"; IsDir = $false; Min = [long]0 }
    @{ Label = "app_python";       Rel = ".venv/Scripts/python.exe";        IsDir = $false; Min = [long]0 }
    @{ Label = "engine worker.py"; Rel = "engine/worker.py";                IsDir = $false; Min = [long]0 }
)
# Model rows come from the manifests THIS run is responsible for (-BaseModel):
# install-LTX25.bat must not report LTX 2.3 as MISSING, and setup.bat must not
# report LTX 2.5 as MISSING. The three fixed rows above stay unconditional --
# they are this script's own prerequisites, not model files.
foreach ($mf in $TargetManifests) {
    foreach ($dl in @($mf.Data.downloads)) {
        foreach ($fl in @($dl.files)) {
            $required += @{
                Label = [string] $fl.label
                Rel   = "models/" + $fl.path
                IsDir = ($fl.kind -eq 'dir')
                Min   = [long] $fl.min
            }
        }
    }
}

$rows = @()
$anyMissing = $false
foreach ($r in $required) {
    $abs = Join-Path $ProjectRoot ($r.Rel -replace '/', '\')
    $exists = Test-Path -LiteralPath $abs
    $size = [long]0
    if ($exists) {
        $size = if ($r.IsDir) { Get-DirectChildSize $abs } else { Get-FileSize $abs }
    }
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
# Regenerate models/INSTALLED_PATHS.txt from the SAME manifest rows: every
# `files[]` entry that carries a `key` is a path the SERVER resolves by name --
# a base-model descriptor `categories[].default_file` or `assets` entry (the
# `key` is that asset key / category, not a config.yaml key any more: the fixed
# default paths left config.yaml in §3-97 P3b).
# ----------------------------------------------------------------------------
#
# This list and the table above no longer answer the same question, so do not
# expect them to match row for row: the table is narrowed by -BaseModel and
# judges each file against its `min`, while this list covers every manifest and
# only asks whether the file exists.
#
# This one list is NOT narrowed by -BaseModel (owner's ruling, 2026-08-30): the
# file describes what is on this disk, so it must not shrink just because the
# batch that happened to run last was only responsible for one base model.
# Instead each row is kept only when the file is really there, using the same
# existence test as the table above -- so after install-LTX25.bat the LTX 2.3
# rows survive, and a base model nobody has downloaded yet contributes nothing.
$pathRows = @()
foreach ($mf in $Manifests) {
    foreach ($dl in @($mf.Data.downloads)) {
        foreach ($fl in @($dl.files)) {
            if (-not $fl.key) { continue }
            $absPath = Join-Path $ModelsDir ($fl.path -replace '/', '\')
            if (-not (Test-Path -LiteralPath $absPath)) { continue }
            $pathRows += [pscustomobject]@{ Key = [string] $fl.key; Value = "./models/" + $fl.path }
        }
    }
}
# Unconditional: it is the interpreter, not a model file, and it is the row a
# reader looks for when nothing else is installed yet (also keeps $keyWidth,
# computed below over the filtered list, from seeing an empty collection).
$pathRows += [pscustomobject]@{ Key = "engine_python"; Value = "./.venv-engine/Scripts/python.exe" }

$keyWidth = ($pathRows | ForEach-Object { $_.Key.Length } | Measure-Object -Maximum).Maximum
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$sb = New-Object System.Text.StringBuilder
[void] $sb.AppendLine("# Regenerated by scripts/install_ltx.ps1 at $stamp")
[void] $sb.AppendLine("# layout: base-model-first (models/<BaseModel>/<Category>)")
[void] $sb.AppendLine("# Generated from scripts/manifests/*.json -- the same rows that guard the")
[void] $sb.AppendLine("# downloads and drive the verification table. GGUF + component-file recipe")
[void] $sb.AppendLine("# (matches the base-model descriptor: scripts/manifests/*.json -> categories/assets).")
[void] $sb.AppendLine("# The 46GB monolith and the 22.7GB QAT Gemma")
[void] $sb.AppendLine("# are intentionally absent (deleted; never re-downloaded).")
foreach ($pr in $pathRows) {
    [void] $sb.AppendLine(("  {0}: {1}""{2}""" -f $pr.Key, (' ' * ($keyWidth - $pr.Key.Length)), $pr.Value))
}
[System.IO.File]::WriteAllText("$ModelsDir\INSTALLED_PATHS.txt", $sb.ToString(), (New-Object System.Text.UTF8Encoding($false)))
Write-Host "`nRegenerated models/INSTALLED_PATHS.txt" -ForegroundColor Cyan

if ($anyMissing) {
    Write-Host ""
    Write-Warning "One or more required artifacts are MISSING (see table above)."
    Write-Warning "Each MISSING row names ONE file. Deleting that single file and re-running is"
    Write-Warning "always enough -- never delete a whole models/ folder, it holds your own LoRAs"
    Write-Warning "and self-converted GGUFs, which no repo can give back."
    if ($SkipModels) { Write-Warning "You passed -SkipModels; re-run without it to fetch models." }
    # Only worth saying when the venv is actually missing: install-<ID>.bat
    # passes -SkipVenv on purpose (it must not re-sync the working engine venv),
    # so an unconditional "re-run without it" would send that user the wrong way.
    if ($SkipVenv -and -not (Test-Path $hfExe)) { Write-Warning "You passed -SkipVenv; re-run without it to build the venvs." }
    exit 1
}
Write-Ok "All required artifacts present."

# ----------------------------------------------------------------------------
# 8) Optional mock, GPU-free smoke test (uses the app venv).
# ----------------------------------------------------------------------------
if ($RunSmoke) {
    Write-Step "Smoke test (mock, GPU-free)"
    if (-not (Test-Path $appPy)) { throw "App venv python not found for smoke test: $appPy" }
    # No extra `uv sync --extra dev` needed here (2026-07-28): step 4 now always
    # syncs with --extra dev, so pytest / iniconfig / pluggy are already present
    # by the time this branch runs. The special-case re-sync that used to live
    # here was made redundant by that step-4 change and has been removed.
    & $appPy -m pytest -q tests/test_smoke.py
    if ($LASTEXITCODE -ne 0) { throw "Smoke test failed." }
    Write-Ok "Smoke test passed."
}

Write-Host "`nDone. Backend is ready to run:  ./run.ps1" -ForegroundColor Green
# Explicit success code: the last native command's $LASTEXITCODE must not leak.
exit 0
