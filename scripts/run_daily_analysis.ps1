$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
try { chcp 65001 | Out-Null } catch {}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LogDir = Join-Path $ProjectRoot "var\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("daily_analysis_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$TranscriptStarted = $false
try {
    Start-Transcript -Path $LogPath -Force | Out-Null
    $TranscriptStarted = $true
    Write-Host "Log file: $LogPath"
} catch {
    Write-Host "Unable to start transcript log: $_"
}

$DefaultPython = "D:\ActionsRunner\my-trade\python\python.exe"
$PythonBin = if ($env:PYTHON_BIN) {
    $env:PYTHON_BIN
} elseif (Test-Path $DefaultPython) {
    $DefaultPython
} else {
    "python"
}

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

# Disable proxy by default (stock data APIs are accessible without proxy)
if (-not $env:DISABLE_DEFAULT_PROXY) {
    $env:DISABLE_DEFAULT_PROXY = "1"
}

# Python resolves the report period from the Formula33 observation date using
# the shared statutory-disclosure rule. REPORT_PERIOD remains an explicit override.

$DefaultProxy = "http://127.0.0.1:7897"
if ($env:DISABLE_DEFAULT_PROXY -ne "1") {
    if (-not $env:HTTP_PROXY) { $env:HTTP_PROXY = $DefaultProxy }
    if (-not $env:HTTPS_PROXY) { $env:HTTPS_PROXY = $DefaultProxy }
    if (-not $env:ALL_PROXY) { $env:ALL_PROXY = $DefaultProxy }
}

# Save current proxy settings to restore later
$SavedHTTP = $env:HTTP_PROXY
$SavedHTTPS = $env:HTTPS_PROXY
$SavedALL = $env:ALL_PROXY

# Temporarily clear proxy for Python subprocess
if ($env:DISABLE_DEFAULT_PROXY -eq "1") {
    $env:HTTP_PROXY = $null
    $env:HTTPS_PROXY = $null
    $env:ALL_PROXY = $null
}

$ExitCode = 1
try {
    & $PythonBin -u -m apps.daily_pipeline @args
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        Write-Host "Daily pipeline failed with exit code $ExitCode. Log: $LogPath"
    }
} catch {
    Write-Host "Daily pipeline launch failed: $_. Log: $LogPath"
} finally {
    # Restore original proxy settings for this PowerShell session
    if ($env:DISABLE_DEFAULT_PROXY -eq "1") {
        $env:HTTP_PROXY = $SavedHTTP
        $env:HTTPS_PROXY = $SavedHTTPS
        $env:ALL_PROXY = $SavedALL
    }
    if ($TranscriptStarted) {
        try { Stop-Transcript | Out-Null } catch {
            Write-Host "Unable to stop transcript log: $_"
        }
    }
}

exit $ExitCode
