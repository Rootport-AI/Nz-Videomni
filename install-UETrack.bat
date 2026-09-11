@echo off
rem ---------------------------------------------------------------------------
rem  Nz-Videomni: add the object-tracking utility AI (UETrack).
rem
rem  Run setup.bat FIRST. This launcher adds ONE optional extra: the CPU-only
rem  .venv-utils interpreter and the UETrack Base weights. It never touches the
rem  app venv or either engine venv, so it cannot disturb a working install.
rem
rem  This file is intentionally pure ASCII with CRLF line endings: cmd.exe reads
rem  it with the OEM code page, so any non-ASCII text here would be mojibake on a
rem  machine whose code page differs. All Japanese wording lives in
rem  scripts\install_model.ps1.
rem ---------------------------------------------------------------------------
setlocal
title Nz-Videomni Install UETrack

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_model.ps1" -BaseModel UETrack

rem Capture the exit code BEFORE pause: %ERRORLEVEL% is not reliable afterwards.
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
