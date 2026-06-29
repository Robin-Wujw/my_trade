param(
    [string]$Time = "16:30",
    [string]$TaskName = "MyTradeDailyAnalysis"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $ScriptDir "run_daily_analysis.ps1"
if (-not (Test-Path $Runner)) { throw "Missing runner: $Runner" }

$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument (
    "-NoProfile -ExecutionPolicy Bypass -File `"{0}`"" -f $Runner
)
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 6)
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "A-share full-market incremental analysis" -Force
Write-Host "Installed scheduled task $TaskName at $Time"
