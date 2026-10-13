param(
    [int]$Port = 8501,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPython = Join-Path $ProjectRoot '.venv-dashboard\Scripts\python.exe'
Set-Location $ProjectRoot
if (-not (Test-Path $VenvPython)) { Write-Error "Dashboard environment not found at $VenvPython. Run .\scripts\install_dashboard.ps1 first."; exit 1 }
$versionText = (& $VenvPython -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null)
if ($LASTEXITCODE -ne 0) { Write-Error 'The dashboard Python environment is not usable. Run the installer again.'; exit 1 }
$versionParts = $versionText.Trim().Split('.')
if ([int]$versionParts[0] -lt 3 -or ([int]$versionParts[0] -eq 3 -and [int]$versionParts[1] -lt 10)) { Write-Error "Python 3.10 or newer is required; found $versionText."; exit 1 }
if ($Port -lt 1 -or $Port -gt 65535) { Write-Error "Port must be between 1 and 65535; found $Port."; exit 1 }
$url = "http://127.0.0.1:$Port"
if (-not $NoBrowser) { Start-Process $url }

& $VenvPython -m streamlit run apps/dashboard.py `
    --server.address 127.0.0.1 `
    --server.port $Port `
    --server.headless true `
    --browser.gatherUsageStats false
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) { Write-Error "Dashboard exited with code $exitCode." }
exit $exitCode
