param(
    [string]$StartDate = "2024-09-24",
    [string]$EndDate = "2026-07-14",
    [string]$KlineDirectory = "var/cache/formula33_kline/akshare",
    [string]$OutputDirectory = "var/cache/tushare_amount_json",
    [string]$Codes = "",
    [int]$Limit = 0,
    [int]$Retries = 3,
    [int]$SleepMs = 1300,
    [string]$NodeExecutable = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe",
    [switch]$AllowInsecure,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$helper = Join-Path $projectRoot "scripts\tushare_fetch.js"
$klineRoot = Resolve-Path (Join-Path $projectRoot $KlineDirectory)
$outputRoot = Join-Path $projectRoot $OutputDirectory
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

if (-not $env:TUSHARE_TOKEN) {
    throw "TUSHARE_TOKEN is not configured in the current environment"
}

if ($Codes.Trim()) {
    $items = $Codes.Split(",") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ } |
        ForEach-Object {
            $pure = $_.Split(".")[-1].PadLeft(6, "0")
            [pscustomobject]@{ Code = $pure }
        }
} else {
    $items = Get-ChildItem -Path $klineRoot -Filter "*.csv" |
        ForEach-Object {
            [pscustomobject]@{ Code = $_.BaseName.Split("_")[-1].PadLeft(6, "0") }
        } |
        Sort-Object Code
}

if ($Limit -gt 0) {
    $items = $items | Select-Object -First $Limit
}

$beg = $StartDate.Replace("-", "")
$end = $EndDate.Replace("-", "")
$env:TUSHARE_ALLOW_INSECURE = if ($AllowInsecure) { "1" } else { "0" }

$done = 0
$failed = 0
$summaries = New-Object System.Collections.Generic.List[object]
foreach ($item in $items) {
    $code = $item.Code
    $market = if ($code.StartsWith("6") -or $code.StartsWith("9")) { "SH" } else { "SZ" }
    $tsCode = "$code.$market"
    $target = Join-Path $outputRoot "$code.json"
    $ok = $false
    $usedCache = $false
    $message = ""
    if ((-not $Force) -and (Test-Path $target)) {
        try {
            $existing = Get-Content -Raw -Path $target | ConvertFrom-Json
            $existingRows = 0
            if ($existing.data -and $existing.data.items) {
                $existingRows = @($existing.data.items).Count
            }
            if ($existing.code -eq 0 -and $existingRows -gt 0) {
                $ok = $true
                $usedCache = $true
                $message = "cached_rows=$existingRows"
            }
        } catch {
            $ok = $false
        }
    }
    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        if ($ok) { break }
        try {
            $raw = & $NodeExecutable $helper $tsCode $beg $end 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw ($raw -join "`n")
            }
            $raw | Set-Content -Path $target -Encoding UTF8
            $payload = Get-Content -Raw -Path $target | ConvertFrom-Json
            $rows = 0
            if ($payload.data -and $payload.data.items) {
                $rows = @($payload.data.items).Count
            }
            if ($payload.code -ne 0) {
                throw "tushare code=$($payload.code) msg=$($payload.msg)"
            }
            if ($rows -le 0) {
                throw "empty items"
            }
            $ok = $true
            $usedCache = $false
            $message = "rows=$rows"
            break
        } catch {
            $message = $_.Exception.Message
            if ($attempt -lt $Retries) {
                Start-Sleep -Milliseconds ([Math]::Max($SleepMs, 100) * $attempt)
            }
        }
    }
    if ($ok) { $done += 1 } else { $failed += 1 }
    $summaries.Add([pscustomobject]@{
        code = $code
        ok = $ok
        message = $message
        path = $target
    })
    if (($done + $failed) % 100 -eq 0 -or ($done + $failed) -eq @($items).Count) {
        Write-Host "tushare json progress $($done + $failed)/$(@($items).Count) ok=$done failed=$failed"
    }
    if ((-not $usedCache) -and $SleepMs -gt 0) {
        Start-Sleep -Milliseconds $SleepMs
    }
}

$summaryPath = Join-Path $outputRoot "_fetch_summary.csv"
$summaries | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8
Write-Host "tushare json done ok=$done failed=$failed output=$outputRoot summary=$summaryPath"
if ($failed -gt 0) { exit 2 }
