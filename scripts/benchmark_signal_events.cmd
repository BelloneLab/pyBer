@echo off
rem Run the reproducible synthetic benchmark in the application's environment.
cd /d "%~dp0.."
call conda run --no-capture-output -n pyBer python "%~dp0benchmark_signal_events.py"
if errorlevel 1 pause
