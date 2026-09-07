@echo off
rem Double-click to validate the real postprocessing inputs in the pyBer environment.
rem Optional arguments, such as --data-dir, are forwarded to the Python script.
call conda run --no-capture-output -n pyBer python "%~dp0validate_postprocessing.py" %*
if errorlevel 1 echo Validation failed. Review the messages above.
pause
