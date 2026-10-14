[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$TaskName = 'StockSelection-Daily'
)

$ErrorActionPreference = 'Stop'
$ExitCode = 1
$TranscriptStarted = $false

try {
    $ProjectRoot = (Get-Item -LiteralPath $ProjectRoot).FullName
    $LogDir = Join-Path $ProjectRoot 'var\logs'
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $LogPath = Join-Path $LogDir ('scheduled_task_setup_{0}_{1}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID)
    Start-Transcript -LiteralPath $LogPath -Force | Out-Null
    $TranscriptStarted = $true
    Write-Host ('Log file: ' + $LogPath)

    $ScriptPath = Join-Path $ProjectRoot 'scripts\run_daily_analysis.ps1'
    if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
        throw "Daily pipeline script does not exist: $ScriptPath"
    }

    $PowerShellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    # Task Scheduler passes a native command line: -File requires double quotes.
    $ActionArguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}"' -f $ScriptPath
    $Action = New-ScheduledTaskAction -Execute $PowerShellPath -Argument $ActionArguments -WorkingDirectory $ProjectRoot
    $Trigger = New-ScheduledTaskTrigger -Daily -At '20:30'
    $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 12)
    $TaskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

    # Replace in place. Never remove the previous task before registration succeeds.
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description 'Daily stock selection pipeline' -User $TaskUser -RunLevel Highest -Force -ErrorAction Stop | Out-Null
    Write-Host ('Task created or updated: ' + $TaskName)
    Write-Host 'Schedule: Every day at 20:30'
    Write-Host ('Script: ' + $ScriptPath)
    $ExitCode = 0
} catch {
    Write-Host ('Scheduled task setup failed: ' + $_.Exception.Message)
    Write-Host $_.InvocationInfo.PositionMessage
    Write-Host $_.ScriptStackTrace
    [Console]::Error.WriteLine('Scheduled task setup failed: ' + $_.Exception.Message)
} finally {
    if ($TranscriptStarted) {
        try {
            Stop-Transcript | Out-Null
        } catch {
            # Log cleanup must not replace the registration result.
            [Console]::Error.WriteLine('Unable to stop setup transcript: ' + $_.Exception.Message)
        }
    }
}

exit $ExitCode
