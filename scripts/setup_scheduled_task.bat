@echo off
chcp 65001 >nul
set "PROJECT_ROOT=%~dp0.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
set "MY_TRADE_PROJECT_ROOT=%PROJECT_ROOT%"
echo Creating scheduled task for daily stock selection...
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
"$ProjectRoot = $env:MY_TRADE_PROJECT_ROOT; ^
$TaskName = 'StockSelection-Daily'; ^
$ScriptPath = Join-Path $ProjectRoot 'scripts\run_daily_analysis.ps1'; ^
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) { ^
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false; ^
    Write-Host 'Removed existing task'; ^
}; ^
$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument \"-NoProfile -ExecutionPolicy Bypass -File '$ScriptPath'\" -WorkingDirectory $ProjectRoot; ^
$Trigger = New-ScheduledTaskTrigger -Daily -At '20:30'; ^
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 12); ^
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description 'Daily stock selection pipeline' -User $env:USERNAME -RunLevel Highest; ^
Write-Host ''; ^
Write-Host 'Task created successfully!'; ^
Write-Host 'Task name: StockSelection-Daily'; ^
Write-Host 'Schedule: Every day at 20:30'; ^
Write-Host ('Script: ' + $ScriptPath)"

echo.
echo Done! Press any key to exit...
pause >nul
