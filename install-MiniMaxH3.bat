@echo off
rem ---------------------------------------------------------------------------
rem  Nz-Videomni: add the MiniMax H3 base model (alpha).
rem
rem  Run setup.bat FIRST. This launcher adds ONE optional base model: its own
rem  engine interpreter .venv-engine-h3 (torch cu130 stack) and the MiniMax H3
rem  weights. It never touches the app venv or the LTX engine venvs, so it
rem  cannot disturb a working install.
rem
rem  This file is intentionally pure ASCII with CRLF line endings: cmd.exe reads
rem  it with the OEM code page, so any non-ASCII text here would be mojibake on a
rem  machine whose code page differs. All Japanese wording lives in
rem  scripts\install_model.ps1.
rem ---------------------------------------------------------------------------
setlocal
title Nz-Videomni Install MiniMax H3

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_model.ps1" -BaseModel MiniMaxH3

rem Capture the exit code BEFORE pause: %ERRORLEVEL% is not reliable afterwards.
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
