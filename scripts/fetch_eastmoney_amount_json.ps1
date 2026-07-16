param(
    [string]$StartDate = "2024-09-24",
    [string]$EndDate = "2026-07-14",
    [string]$KlineDirectory = "var/cache/formula33_kline/akshare",
    [string]$OutputDirectory = "var/cache/eastmoney_amount_json",
    [string]$Codes = "",
    [int]$Limit = 0,
    [int]$Retries = 3,
    [int]$SleepMs = 250,
    [string]$NodeExecutable = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe",
    [switch]$AllowInsecure,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$helper = Join-Path $projectRoot "scripts\eastmoney_fetch.js"
$klineRoot = Resolve-Path (Join-Path $projectRoot $KlineDirectory)
$outputRoot = Join-Path $projectRoot $OutputDirectory
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

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
$env:EASTMONEY_ALLOW_INSECURE = if ($AllowInsecure) { "1" } else { "0" }

$done = 0
$failed = 0
$summaries = New-Object System.Collections.Generic.List[object]
foreach ($item in $items) {
    $code = $item.Code
    $market = if ($code.StartsWith("6") -or $code.StartsWith("9")) { "1" } else { "0" }
    $url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=$market.$code&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=101&fqt=1&beg=$beg&end=$end&smplmt=10000&lmt=1000000"
    $target = Join-Path $outputRoot "$code.json"
    $ok = $false
    $usedCache = $false
    $message = ""
    if ((-not $Force) -and (Test-Path $target)) {
        try {
            $existing = Get-Content -Raw -Path $target | ConvertFrom-Json
            $existingRows = 0
            if ($existing.data -and $existing.data.klines) {
                $existingRows = @($existing.data.klines).Count
            }
            if ($existingRows -gt 0) {
                $ok = $true
                $usedCache = $true
                $message = "cached_rows=$existingRows"
            }
        } catch {
            $ok = $false
        }
    }
    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        if ($ok) {
            break
        }
        try {
            $raw = & $NodeExecutable $helper $url 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw ($raw -join "`n")
            }
            $raw | Set-Content -Path $target -Encoding UTF8
            $payload = Get-Content -Raw -Path $target | ConvertFrom-Json
            $rows = 0
            if ($payload.data -and $payload.data.klines) {
                $rows = @($payload.data.klines).Count
            }
            if ($rows -le 0) {
                throw "empty klines"
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
    if ($ok) {
        $done += 1
    } else {
        $failed += 1
    }
    $summaries.Add([pscustomobject]@{
        code = $code
        ok = $ok
        message = $message
        path = $target
    })
    if (($done + $failed) % 100 -eq 0 -or ($done + $failed) -eq @($items).Count) {
        Write-Host "eastmoney json progress $($done + $failed)/$(@($items).Count) ok=$done failed=$failed"
    }
    if ((-not $usedCache) -and $SleepMs -gt 0) {
        Start-Sleep -Milliseconds $SleepMs
    }
}

$summaryPath = Join-Path $outputRoot "_fetch_summary.csv"
$summaries | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8
Write-Host "eastmoney json done ok=$done failed=$failed output=$outputRoot summary=$summaryPath"
if ($failed -gt 0) {
    exit 2
}
