# scripts/weighted_forecast.ps1
param(
    [string]$Symbol = "EUR/USD",
    [string]$Timeframe = "1m",
    [int]$Horizon = 5,
    [int]$Days = 7,
    [string]$Model = "ridge",
    [string]$Encoder = "none",
    [string]$ForecastMethod = "supervised"
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
                    if ($v.Length -ge 2) {
                        if ($v.StartsWith("'") -and $v.EndsWith("'")) { $v = $v.Substring(1, $v.Length-2) }
                        elseif ($v.StartsWith('"') -and $v.EndsWith('"')) { $v = $v.Substring(1, $v.Length-2) }
                    }
                    if ($k) { ${env:$k} = $v }
                }
            }
        }
    }
}

Load-DotEnv

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir
Set-Location ..

$py = "python"
& $py "tests/manual_tests/weighted_forecast.py" --symbol $Symbol --timeframe $Timeframe --horizon $Horizon --days $Days --model $Model --encoder $Encoder --forecast_method $ForecastMethod
if ($LASTEXITCODE -ne 0) {
    Write-Error "weighted_forecast.py exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}
