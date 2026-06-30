param(
    [string]$Repository = "Robin-Wujw/my_trade",
    [string]$InstallDir = "D:\ActionsRunner\my-trade",
    [string]$Proxy = "http://127.0.0.1:7897"
)

$ErrorActionPreference = "Stop"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Please run this script from an Administrator PowerShell window."
}

$gh = "C:\Program Files\GitHub CLI\gh.exe"
if (-not (Test-Path $gh)) { throw "GitHub CLI not found: $gh" }

$env:HTTP_PROXY = $Proxy
$env:HTTPS_PROXY = $Proxy

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
if (Test-Path (Join-Path $InstallDir ".runner")) {
    throw "A GitHub Actions runner is already configured in $InstallDir"
}

$release = & $gh api repos/actions/runner/releases/latest --jq .tag_name
if ($LASTEXITCODE -ne 0 -or -not $release) { throw "Unable to resolve the latest runner release." }
$version = $release.TrimStart("v")
$archive = Join-Path $env:TEMP "actions-runner-win-x64-$version.zip"
$url = "https://github.com/actions/runner/releases/download/$release/actions-runner-win-x64-$version.zip"

Write-Host "Downloading GitHub Actions runner $release..."
Invoke-WebRequest -Uri $url -Proxy $Proxy -OutFile $archive
Expand-Archive -LiteralPath $archive -DestinationPath $InstallDir -Force

# The runner service reads proxy settings from this file on startup.
@(
    "HTTP_PROXY=$Proxy"
    "HTTPS_PROXY=$Proxy"
) | Set-Content -Path (Join-Path $InstallDir ".env") -Encoding ascii

$token = & $gh api --method POST "repos/$Repository/actions/runners/registration-token" --jq .token
if ($LASTEXITCODE -ne 0 -or -not $token) { throw "Unable to create a runner registration token." }

Push-Location $InstallDir
try {
    & .\config.cmd --unattended `
        --url "https://github.com/$Repository" `
        --token $token `
        --name "$env:COMPUTERNAME-my-trade" `
        --labels "my-trade" `
        --work "_work" `
        --runasservice `
        --replace
    if ($LASTEXITCODE -ne 0) { throw "Runner configuration failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

Write-Host "GitHub Actions runner installed as a Windows service."
Write-Host "It will run queued jobs whenever this computer is powered on and online."
