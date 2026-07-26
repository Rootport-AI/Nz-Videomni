@echo off
rem ---------------------------------------------------------------------------
rem  Nz-LTX23 backend setup launcher.
rem
rem  This file is intentionally pure ASCII with CRLF line endings: cmd.exe reads
rem  it with the OEM code page, so any non-ASCII text here would be mojibake on a
rem  machine whose code page differs. All Japanese wording lives in setup.ps1.
rem ---------------------------------------------------------------------------
setlocal
title Nz-LTX23 Setup

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"

rem Capture the exit code BEFORE pause: %ERRORLEVEL% is not reliable afterwards.
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
