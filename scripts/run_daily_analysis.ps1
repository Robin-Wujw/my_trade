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

$DefaultProxy = "http://127.0.0.1:7897"
if ($env:DISABLE_DEFAULT_PROXY -ne "1") {
    if (-not $env:HTTP_PROXY) { $env:HTTP_PROXY = $DefaultProxy }
    if (-not $env:HTTPS_PROXY) { $env:HTTPS_PROXY = $DefaultProxy }
    if (-not $env:ALL_PROXY) { $env:ALL_PROXY = $DefaultProxy }
}

& $PythonBin -u -m apps.daily_pipeline @args
exit $LASTEXITCODE
