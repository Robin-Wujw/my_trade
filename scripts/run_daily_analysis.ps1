$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

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

# Auto-select latest available financial report period
if (-not $env:REPORT_PERIOD) {
    $Today = Get-Date
    $Year = $Today.Year
    $Month = $Today.Month

    if ($Month -ge 1 -and $Month -le 4) {
        # Jan-Apr: Use previous year's annual report (Q4)
        $env:REPORT_PERIOD = "{0}-12-31" -f ($Year - 1)
    } elseif ($Month -ge 5 -and $Month -le 8) {
        # May-Aug: Use current year Q1 report
        $env:REPORT_PERIOD = "{0}-03-31" -f $Year
    } elseif ($Month -ge 9 -and $Month -le 10) {
        # Sep-Oct: Use current year Q2 report
        $env:REPORT_PERIOD = "{0}-06-30" -f $Year
    } else {
        # Nov-Dec: Use current year Q3 report
        $env:REPORT_PERIOD = "{0}-09-30" -f $Year
    }
    Write-Host "Auto-selected report period: $env:REPORT_PERIOD"
}

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

try {
    & $PythonBin -u -m apps.daily_pipeline @args
    $ExitCode = $LASTEXITCODE
} finally {
    # Restore original proxy settings for this PowerShell session
    if ($env:DISABLE_DEFAULT_PROXY -eq "1") {
        $env:HTTP_PROXY = $SavedHTTP
        $env:HTTPS_PROXY = $SavedHTTPS
        $env:ALL_PROXY = $SavedALL
    }
}

exit $ExitCode
