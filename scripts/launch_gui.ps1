# scripts/launch_gui.ps1
# Helper: activate .venv (PowerShell) and launch the GUI helper
# Usage: from project root in PowerShell: .\scripts\launch_gui.ps1

# clear console to remove previous logs
cls

# Ensure script runs from repo root
$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $PSScriptRoot
Set-Location ..

# Activate virtualenv (PowerShell)
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Output "Activating .venv..."
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Output ".venv not found. Create and install dependencies first: python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e ."
    exit 1
}

# Optional: show current env vars summary
Write-Output "TIINGO_API_KEY: $env:TIINGO_API_KEY"
Write-Output "DATABASE_URL: $env:DATABASE_URL"
Write-Output "ADMIN_TOKENS: $env:ADMIN_TOKENS"

# Launch GUI helper
Write-Output "Starting GUI (run_gui.py)..."
python .\scripts\run_gui.py
