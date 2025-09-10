# scripts/run_backfill_and_check.ps1
param(
    [string]$Symbol = "EUR/USD",
    [string]$Timeframe = "1m",
    [int]$Days = 30
)

function Load-DotEnv {
    param([string]$Path = ".env")
    if (Test-Path $Path) {
        Get-Content $Path | ForEach-Object {
            $line = $_.Trim()
            if ($line -and -not $line.StartsWith("#")) {
                $idx = $line.IndexOf("=")
                if ($idx -gt 0) {
                    $k = $line.Substring(0, $idx).Trim()
                    $v = $line.Substring($idx + 1).Trim()
                    # strip surrounding single or double quotes if any
                    $v = $v.Trim("'`" + '"' )
                    if ($k) { ${env:$k} = $v }
                }
            }
        }
    }
}

# Ensure running from project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir
Set-Location ..

# Load .env into environment
Load-DotEnv

Write-Output "Running run_backfill_and_check for symbol='$Symbol' timeframe='$Timeframe' days=$Days..."
$py = "python"
& $py "tests/manual_tests/run_backfill_and_check.py" --symbol $Symbol --timeframe $Timeframe --days $Days
if ($LASTEXITCODE -ne 0) {
    Write-Error "run_backfill_and_check.py exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}
