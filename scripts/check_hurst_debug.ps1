# scripts/check_hurst_debug.ps1
param(
    [string]$Symbol = "EUR/USD",
    [string]$Timeframe = "1m",
    [int]$TsUtc,
    [int]$WindowMax = 1024
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

Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Definition)
Set-Location ..

python tests/manual_tests/check_hurst_debug.py --symbol $Symbol --timeframe $Timeframe --ts_utc $TsUtc --window_max $WindowMax
