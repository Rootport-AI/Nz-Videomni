@echo off
rem ---------------------------------------------------------------------------
rem  Nz-Videomni: add the LTX 2.5 base model (weights only).
rem
rem  Run setup.bat FIRST. This launcher downloads model files only -- it never
rem  creates or re-syncs a Python environment, so it cannot disturb a working
rem  LTX 2.3 install.
rem
rem  This file is intentionally pure ASCII with CRLF line endings: cmd.exe reads
rem  it with the OEM code page, so any non-ASCII text here would be mojibake on a
rem  machine whose code page differs. All Japanese wording lives in
rem  scripts\install_model.ps1.
rem ---------------------------------------------------------------------------
setlocal
title Nz-Videomni Install LTX 2.5

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_model.ps1" -BaseModel LTX25

rem Capture the exit code BEFORE pause: %ERRORLEVEL% is not reliable afterwards.
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
