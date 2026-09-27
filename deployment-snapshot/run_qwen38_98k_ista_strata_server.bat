@echo off
setlocal EnableExtensions
set "_STRATA_LAUNCH_PATH=%PATH%"
set "Path="
set "PATH="
set "PATH=%_STRATA_LAUNCH_PATH%"
set "_STRATA_LAUNCH_PATH="
cd /d "%~dp0"
title ISTA Strata - Cache + Vision CPU - 98K - sperimentale
rem Desktop 1d is the final server entry point; Pi remains an independent client.
rem Output uses exact remaining context. Memory/thermal gates stay enabled.
if "%~1"=="--text-only" goto TEXT
if "%~1"=="--check" goto CHECK
echo ISTA Strata 98K - testo e vision CPU. Attendi PRONTO, poi apri pi in WSL.
echo Pi puo essere aperto anche a server spento per utilizzare altri modelli.
"%~dp0research\qwen-strata-20260926\.venv\Scripts\python.exe" "%~dp0research\qwen-strata-update-20260927\manual\serve_only_guard.py" manual %*
goto DONE
:CHECK
"%~dp0research\qwen-strata-20260926\.venv\Scripts\python.exe" "%~dp0research\qwen-strata-update-20260927\manual\serve_only_guard.py" check
goto DONE
:TEXT
"%~dp0research\qwen-strata-20260926\.venv\Scripts\python.exe" "%~dp0research\qwen-strata-agent-20260927\manual\manual_guard.py"
:DONE
set "STRATA_EXIT=%ERRORLEVEL%"
echo.
echo Server terminato con codice %STRATA_EXIT%.
if not "%~1"=="--check" pause
exit /b %STRATA_EXIT%
