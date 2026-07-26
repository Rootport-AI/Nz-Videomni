@echo off
rem ---------------------------------------------------------------------------
rem  Nz-LTX23 backend launcher.
rem
rem  Pure ASCII + CRLF on purpose (see setup.bat). All Japanese wording lives in
rem  run.ps1, which MUST stay in the repository root: it resolves .python,
rem  .venv and main.py from its own directory.
rem ---------------------------------------------------------------------------
setlocal
title Nz-LTX23 Server

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*

rem Capture the exit code BEFORE pause: %ERRORLEVEL% is not reliable afterwards.
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
