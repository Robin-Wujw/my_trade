@echo off
setlocal
chcp 65001 >nul
echo Creating scheduled task for daily stock selection...
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0setup_scheduled_task.ps1" %*
set "SETUP_EXIT_CODE=%ERRORLEVEL%"
if not "%SETUP_EXIT_CODE%"=="0" echo Scheduled task setup failed with exit code %SETUP_EXIT_CODE%. 1>&2
exit /b %SETUP_EXIT_CODE%
