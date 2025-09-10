# scripts/reset_db.ps1
param(
    [switch]$Alembic
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

# Confirm destructive action if DB file exists
$cfg = python - <<'PY' 
from forex_diffusion.utils.config import get_config
cfg = get_config()
url = getattr(cfg.db, "database_url", None) or (cfg.db.get("database_url") if isinstance(cfg.db, dict) else None)
print(url)
PY
# Note: above prints DB URL; proceed
Write-Output "This will recreate the local sqlite DB (if configured)."
if (-not $PSCmdlet.IsInteractive) {
    Write-Output "Non-interactive shell: proceeding."
} else {
    $confirm = Read-Host "Proceed with DB reset? Type 'yes' to continue"
    if ($confirm -ne "yes") {
        Write-Output "Aborted by user."
        exit 0
    }
}

$py = "python"
if ($Alembic.IsPresent) {
    & $py "scripts/reset_db.py" --alembic
} else {
    & $py "scripts/reset_db.py"
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "reset_db.py exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}
