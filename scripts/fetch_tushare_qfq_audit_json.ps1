param(
    [Parameter(Mandatory=$true)]
    [string]$Codes,
    [string]$StartDate = "2024-01-01",
    [string]$EndDate = "2026-07-14",
    [string]$OutputDirectory = "var/cache/tushare_qfq_audit_json",
    [int]$Retries = 3,
    [int]$SleepMs = 1300,
    [string]$NodeExecutable = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe",
    [switch]$AllowInsecure,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$helper = Join-Path $projectRoot "scripts\tushare_query.js"
$outputRoot = Join-Path $projectRoot $OutputDirectory
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$beg = $StartDate.Replace("-", "")
$end = $EndDate.Replace("-", "")
$env:TUSHARE_ALLOW_INSECURE = if ($AllowInsecure) { "1" } else { "0" }

$items = $Codes.Split(",") |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ } |
    ForEach-Object {
        $pure = $_.Split(".")[-1].PadLeft(6, "0")
        $market = if ($pure.StartsWith("6") -or $pure.StartsWith("9")) { "SH" } else { "SZ" }
        [pscustomobject]@{ Code = $pure; TsCode = "$pure.$market" }
    }

$summaries = New-Object System.Collections.Generic.List[object]
foreach ($item in $items) {
    foreach ($api in @("daily", "adj_factor")) {
        $fields = if ($api -eq "daily") {
            "ts_code,trade_date,open,high,low,close,vol,amount"
        } else {
            "ts_code,trade_date,adj_factor"
        }
        $target = Join-Path $outputRoot "$($item.Code)_$api.json"
        $ok = $false
        $message = ""
        if ((-not $Force) -and (Test-Path $target)) {
            try {
                $existing = Get-Content -Raw -Path $target | ConvertFrom-Json
                $rows = 0
                if ($existing.data -and $existing.data.items) {
                    $rows = @($existing.data.items).Count
                }
                if ($existing.code -eq 0 -and $rows -gt 0) {
                    $ok = $true
                    $message = "cached_rows=$rows"
                }
            } catch {
                $ok = $false
            }
        }
        for ($attempt = 1; $attempt -le $Retries; $attempt++) {
            if ($ok) { break }
            try {
                $raw = & $NodeExecutable $helper $api $item.TsCode $beg $end $fields 2>&1
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
                    $messageText = "tushare code=$($payload.code) msg=$($payload.msg)"
                    if (($payload.code -eq 40203 -or "$($payload.msg)" -like "*频率超限*") -and $attempt -lt $Retries) {
                        $message = $messageText
                        Start-Sleep -Seconds 65
                        continue
                    }
                    throw $messageText
                }
                if ($rows -le 0) {
                    throw "empty items"
                }
                $ok = $true
                $message = "rows=$rows"
                break
            } catch {
                $message = $_.Exception.Message
                if ($attempt -lt $Retries) {
                    Start-Sleep -Milliseconds ([Math]::Max($SleepMs, 100) * $attempt)
                }
            }
        }
        $summaries.Add([pscustomobject]@{
            code = $item.Code
            api = $api
            ok = $ok
            message = $message
            path = $target
        })
        if (-not $ok) {
            Write-Warning "$($item.Code) $api failed: $message"
        }
        if ($SleepMs -gt 0) {
            Start-Sleep -Milliseconds $SleepMs
        }
    }
}

$summaryPath = Join-Path $outputRoot "_fetch_summary.csv"
$summaries | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8
$failed = @($summaries | Where-Object { -not $_.ok }).Count
Write-Host "tushare qfq audit json done failed=$failed output=$outputRoot summary=$summaryPath"
if ($failed -gt 0) { exit 2 }
