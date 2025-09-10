# scripts/ml_workflow_check.ps1
param(
    [string]$Symbol = "EUR/USD",
    [string]$Timeframe = "1m",
    [int]$DaysBackfill = 3,
    [int]$NClusters = 8,
    [int]$MaxFeatures = 2000
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
                    # strip surrounding single or double quotes if any in a safe way
                    if ($v.Length -ge 2) {
                        if ($v.StartsWith("'") -and $v.EndsWith("'")) {
                            $v = $v.Substring(1, $v.Length - 2)
                        } elseif ($v.StartsWith('"') -and $v.EndsWith('"')) {
                            $v = $v.Substring(1, $v.Length - 2)
                        }
                    }
                    if ($k) { ${env:$k} = $v }
                }
            }
        }
    }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir
Set-Location ..

# Load .env into environment
Load-DotEnv

Write-Output "Running ML workflow check for $Symbol $Timeframe (DaysBackfill=$DaysBackfill, NClusters=$NClusters)..."
$py = "python"
& $py "tests/manual_tests/ml_workflow_check.py" --symbol $Symbol --timeframe $Timeframe --days_backfill $DaysBackfill --n_clusters $NClusters --max_features $MaxFeatures
if ($LASTEXITCODE -ne 0) {
    Write-Error "ml_workflow_check.py exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}
