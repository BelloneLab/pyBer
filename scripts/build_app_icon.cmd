@echo off
cd /d "%~dp0.."
call conda run -n pyBer python scripts/build_app_icon.py
pause
