$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$DefaultPython = "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } elseif (Test-Path $DefaultPython) { $DefaultPython } else { "python" }

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$DefaultProxy = "http://127.0.0.1:7897"
if ($env:DISABLE_DEFAULT_PROXY -ne "1") {
    if (-not $env:HTTP_PROXY) { $env:HTTP_PROXY = $DefaultProxy }
    if (-not $env:HTTPS_PROXY) { $env:HTTPS_PROXY = $DefaultProxy }
    if (-not $env:ALL_PROXY) { $env:ALL_PROXY = $DefaultProxy }
}

$LogDir = if ($env:LOG_DIR) { $env:LOG_DIR } else { Join-Path $ScriptDir "logs" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("daily_analysis_{0}.log" -f (Get-Date -Format "yyyyMMdd"))
$script:FailedSteps = @()

function Add-LogLine {
    param([string]$Text)
    Add-Content -Path $LogFile -Value $Text -Encoding utf8
}

function Run-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Add-LogLine ""
    Add-LogLine ("========== {0} START {1} ==========" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Name)
    Add-LogLine ("COMMAND: {0} {1}" -f $PythonBin, ($Arguments -join " "))

    & $PythonBin -u @Arguments 2>&1 | ForEach-Object {
        Add-LogLine ([string]$_)
    }
    $status = $LASTEXITCODE

    Add-LogLine ("========== {0} END {1} status={2} ==========" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Name, $status)
    if ($status -ne 0) { $script:FailedSteps += ("{0}:{1}" -f $Name, $status) }
    return $status
}

$FactorWorkers = if ($env:FACTOR_WORKERS) { $env:FACTOR_WORKERS } else { "1" }
$Formula33Workers = if ($env:FORMULA33_WORKERS) { $env:FORMULA33_WORKERS } else { "1" }
$Formula33Sleep = if ($env:FORMULA33_SLEEP) { $env:FORMULA33_SLEEP } else { "0.2" }
$Formula33Retries = if ($env:FORMULA33_RETRIES) { $env:FORMULA33_RETRIES } else { "5" }
$Formula33RetryDelay = if ($env:FORMULA33_RETRY_DELAY) { $env:FORMULA33_RETRY_DELAY } else { "5" }
$Formula33CapitalWorkers = if ($env:FORMULA33_CAPITAL_WORKERS) { $env:FORMULA33_CAPITAL_WORKERS } else { "1" }
$SectorSleep = if ($env:SECTOR_SLEEP) { $env:SECTOR_SLEEP } else { "0.3" }
$SectorRetries = if ($env:SECTOR_RETRIES) { $env:SECTOR_RETRIES } else { "5" }
$SectorRetryDelay = if ($env:SECTOR_RETRY_DELAY) { $env:SECTOR_RETRY_DELAY } else { "5" }
$FinancialUpdates = if ($env:FINANCIAL_UPDATES) { $env:FINANCIAL_UPDATES } else { "100" }

Run-Step "formula33 market structure" @(
    "formula33Stats.py",
    "--lookback", "21",
    "--history-days", "420",
    "--workers", $Formula33Workers,
    "--sleep", $Formula33Sleep,
    "--retries", $Formula33Retries,
    "--retry-delay", $Formula33RetryDelay,
    "--capital-workers", $Formula33CapitalWorkers,
    "--require-end-trade",
    "--price-source", "akshare",
    "--metadata-source", "akshare",
    "--missing-mktcap-policy", "pass",
    "--market-cap-source", "none"
) | Out-Null

Run-Step "sector horizontal statistics" @(
    "sectorStats.py",
    "--lookback", "10",
    "--history-days", "90",
    "--top-amount", "50",
    "--sleep", $SectorSleep,
    "--retries", $SectorRetries,
    "--retry-delay", $SectorRetryDelay,
    "--fallback-sample"
) | Out-Null

Run-Step "sector mainline watch" @(
    "sectorWatch.py",
    "--top", "30",
    "--workers", "4",
    "--days", "80",
    "--limit-up-days", "5",
    "--sleep", $SectorSleep,
    "--retries", $SectorRetries,
    "--retry-delay", $SectorRetryDelay,
    "--fallback-sample"
) | Out-Null

Run-Step "factorStock daily selection" @(
    "factorStock.py",
    "--top", "200",
    "--core-min-score", "80",
    "--low-min-score", "75",
    "--quality-min-score", "80",
    "--value-min-mktcap", "100",
    "--workers", $FactorWorkers,
    "--value-watch-ratio", "1.08",
    "--value-watch-top", "20",
    "--akshare-cache-only",
    "--allow-login-fail"
) | Out-Null

Run-Step "full market fundamental cache and snapshot" @(
    "fullMarketFundamentalUpdate.py",
    "--max-updates", $FinancialUpdates,
    "--workers", "2",
    "--min-price-coverage", "0.90",
    "--min-financial-coverage", "0.35",
    "--target-financial-coverage", "0.95",
    "--alert"
) | Out-Null

Run-Step "daily fundamental sections" @(
    "dailyFundamentalSelect.py",
    "--value-ratio", "1.08",
    "--normal-top", "30"
) | Out-Null

$ReportArguments = @(
    "dailyReportPush.py",
    "--top", "10",
    "--selection-top", "30",
    "--max-chars", "12000"
)
if ($env:NO_PUSH -eq "1") { $ReportArguments += "--no-push" }
Run-Step "daily consolidated PushPlus report" $ReportArguments | Out-Null

Add-LogLine ""
Add-LogLine ("{0} daily analysis finished" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Add-LogLine "Outputs:"
$SelectionDir = -join (0x9009, 0x80A1, 0x7ED3, 0x679C | ForEach-Object { [char]$_ })
$BoardDir = -join (0x677F, 0x5757, 0x89C2, 0x5BDF | ForEach-Object { [char]$_ })
foreach ($dir in @($SelectionDir, $BoardDir)) {
    $path = Join-Path $ScriptDir $dir
    if (Test-Path $path) {
        Get-ChildItem -Path $path -File | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-2) } | Sort-Object FullName | ForEach-Object {
            Add-LogLine $_.FullName
        }
    }
}

Write-Host "Daily analysis finished. Log: $LogFile"
if ($script:FailedSteps.Count -gt 0) {
    $summary = "FAILED STEPS: " + ($script:FailedSteps -join ", ")
    Add-LogLine $summary
    & $PythonBin -u "pipelineAlert.py" --title "Daily selection pipeline failed" --message $summary 2>&1 | ForEach-Object { Add-LogLine ([string]$_) }
    Write-Error $summary
    exit 1
}
