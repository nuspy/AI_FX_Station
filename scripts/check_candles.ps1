# scripts/check_candles.ps1
param(
    [string]$Symbol = "EUR/USD",
    [string]$Timeframe = "1m"
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

Write-Output "Running check_candles for symbol='$Symbol' timeframe='$Timeframe'..."
$py = "python"
& $py "tests/manual_tests/check_candles.py" --symbol $Symbol --timeframe $Timeframe
if ($LASTEXITCODE -ne 0) {
    Write-Error "check_candles.py exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}
