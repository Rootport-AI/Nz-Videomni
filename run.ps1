# run.ps1 — Phase 1 launcher for LTX-AviUtl2-Bridge
#
# Environment isolation policy (spec 2.5):
#   - Everything (incl. the Python interpreter) lives under this project directory.
#   - We never touch the system Python and never set persistent system env vars.
#   - All env vars below are process-scoped only.
#
# Usage:
#   ./run.ps1                 # localhost only, port 18620
#   ./run.ps1 --listen        # bind 0.0.0.0 (home LAN)
#   ./run.ps1 --port 19000
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Keep uv-managed Python inside the project (process-scoped).
$env:UV_PYTHON_INSTALL_DIR = "$PSScriptRoot\.python"

# Reduce CUDA fragmentation OOM (harmless when no GPU is present).
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}

$python = "$PSScriptRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Project venv not found. Creating it now..." -ForegroundColor Yellow
    uv venv --python 3.12 .venv
    uv sync
}

& $python "$PSScriptRoot\main.py" @Args
