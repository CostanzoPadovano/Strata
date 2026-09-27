@echo off
setlocal EnableExtensions
set "_STRATA_DEBUG_PATH=%PATH%"
set "Path="
set "PATH="
set "PATH=%_STRATA_DEBUG_PATH%"
set "_STRATA_DEBUG_PATH="
cd /d "%~dp0"
title ISTA Strata - Debug RAM GPU Paging - 98K
"%~dp0research\qwen-strata-20260926\.venv\Scripts\python.exe" "%~dp0research\qwen-strata-debug-20260927\debug_launcher.py" %*
set "STRATA_DEBUG_EXIT=%ERRORLEVEL%"
echo.
echo Debug terminato con codice %STRATA_DEBUG_EXIT%.
if not "%~1"=="--check" pause
exit /b %STRATA_DEBUG_EXIT%
