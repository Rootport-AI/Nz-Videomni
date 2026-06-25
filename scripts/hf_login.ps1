<#
.SYNOPSIS
    One-time HuggingFace login for the gated Gemma 3 text encoder, kept inside
    the project (token stored under hf_home/, which is gitignored).

.DESCRIPTION
    Run this ONCE before scripts/install_ltx.ps1. It:
      1. Points HF_HOME at the project (so the saved token never lands in your
         user profile) — process-scoped, no persistent system changes.
      2. Runs `hf auth login` using THIS project's .venv (no system Python).
      3. Verifies with `hf auth whoami`.

    After this, install_ltx.ps1 -WithGemma works WITHOUT passing -HfToken: the
    download reuses the stored login automatically.

    PREREQUISITE (manual, one-time, in a browser):
      Accept the license for the gated model, otherwise the token cannot pull it:
        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
      Create a READ token at:
        https://huggingface.co/settings/tokens

.NOTES
    To log out / remove the stored token:  .venv\Scripts\hf.exe auth logout
    The token file lives at  hf_home\token  (gitignored).

.EXAMPLE
    ./scripts/hf_login.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

# Keep the saved token inside the project (process-scoped).
if (-not $env:HF_HOME) { $env:HF_HOME = "$ProjectRoot\hf_home" }
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null

$hf = "$ProjectRoot\.venv\Scripts\hf.exe"
if (-not (Test-Path $hf)) {
    throw "hf CLI not found at $hf. Run setup first (uv sync --extra dev)."
}

Write-Host "HF_HOME = $env:HF_HOME (token will be saved here, gitignored)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Before continuing, make sure you have (one-time, in a browser):" -ForegroundColor Yellow
Write-Host "  1. Accepted the license at" -ForegroundColor Yellow
Write-Host "     https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
Write-Host "  2. Created a READ token at https://huggingface.co/settings/tokens"
Write-Host ""

# Interactive: prompts for the token (input hidden), does NOT touch git creds.
# --force overwrites any existing (possibly stale/invalidated) stored token.
& $hf auth login --force

Write-Host "`nVerifying..." -ForegroundColor Cyan
& $hf auth whoami

Write-Host "`nDone. You can now run:" -ForegroundColor Green
Write-Host "  ./scripts/install_ltx.ps1 -WithGemma" -ForegroundColor Green
Write-Host "(no -HfToken needed; the stored login is reused)"
