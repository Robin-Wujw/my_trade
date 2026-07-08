@echo off
chcp 65001 >nul
echo Creating scheduled task for daily stock selection...
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
"$TaskName = 'StockSelection-Daily'; ^
$ScriptPath = 'D:\MyCodes\my_trade\scripts\run_daily_analysis.ps1'; ^
$WorkingDir = 'D:\MyCodes\my_trade'; ^
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) { ^
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false; ^
    Write-Host 'Removed existing task'; ^
}; ^
$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument \"-NoProfile -ExecutionPolicy Bypass -File '$ScriptPath'\" -WorkingDirectory $WorkingDir; ^
$Trigger = New-ScheduledTaskTrigger -Daily -At '20:30'; ^
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 12); ^
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description 'Daily stock selection pipeline' -User $env:USERNAME -RunLevel Highest; ^
Write-Host ''; ^
Write-Host 'Task created successfully!'; ^
Write-Host 'Task name: StockSelection-Daily'; ^
Write-Host 'Schedule: Every day at 20:30'; ^
Write-Host 'Script: D:\MyCodes\my_trade\scripts\run_daily_analysis.ps1'"

echo.
echo Done! Press any key to exit...
pause >nul
