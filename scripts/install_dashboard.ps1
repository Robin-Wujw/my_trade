param(
    [string]$PythonBin = $env:PYTHON_BIN,
    [bool]$CreateShortcut = $true
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPath = Join-Path $ProjectRoot '.venv-dashboard'
$VenvPython = Join-Path $VenvPath 'Scripts\python.exe'
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($PythonBin)) { $PythonBin = 'python' }

function Invoke-Native {
    param([string]$FilePath, [string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')" }
}

try {
    $versionText = (& $PythonBin -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null)
    if ($LASTEXITCODE -ne 0) { throw "Unable to run '$PythonBin'. Install Python 3.10+ and retry." }
    $versionParts = $versionText.Trim().Split('.')
    if ([int]$versionParts[0] -lt 3 -or ([int]$versionParts[0] -eq 3 -and [int]$versionParts[1] -lt 10)) { throw "Python 3.10 or newer is required; found $versionText." }
    if (-not (Test-Path $VenvPython)) {
        Write-Host "Creating isolated dashboard environment at $VenvPath ..."
        Invoke-Native $PythonBin @('-m', 'venv', $VenvPath)
    } else { Write-Host "Reusing dashboard environment at $VenvPath ..." }
    Invoke-Native $VenvPython @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-Native $VenvPython @('-m', 'pip', 'install', '-e', '.[dashboard]')
    if ($CreateShortcut) {
        try {
            $desktop = [Environment]::GetFolderPath('Desktop')
            $shortcutPath = Join-Path $desktop 'my_trade Dashboard.lnk'
            $shell = New-Object -ComObject WScript.Shell
            $shortcut = $shell.CreateShortcut($shortcutPath)
            $shortcut.TargetPath = Join-Path $ProjectRoot 'scripts\start_dashboard.bat'
            $shortcut.WorkingDirectory = $ProjectRoot
            $shortcut.Description = 'Start the my_trade local research dashboard'
            $shortcut.Save()
            Write-Host "Desktop shortcut created: $shortcutPath"
        } catch { Write-Warning "Could not create the desktop shortcut: $($_.Exception.Message)" }
    }
    Write-Host 'Dashboard installation completed.'
    Write-Host 'Start it with: .\scripts\start_dashboard.ps1'
    exit 0
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
