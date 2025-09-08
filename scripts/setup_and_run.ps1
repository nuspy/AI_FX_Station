#!/usr/bin/env pwsh
# setup_and_run.ps1 - setup venv, update repo, install deps and run GUI (Windows PowerShell)
# Usage: .\scripts\setup_and_run.ps1

param(
    [string] $Branch = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
Set-Location $ROOT

Write-Host "[setup] Git pull (fast-forward) current branch"
try {
    if ($Branch) { git fetch origin $Branch; git checkout $Branch; git pull --ff-only origin $Branch }
    else { git pull --ff-only }
} catch {
    Write-Warning "[setup] git pull failed or offline: $_"
}

# ensure python exists
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Error "Python not found on PATH. Install Python 3.12 and re-run."
    exit 2
}

Write-Host "[setup] Using python: $($py.Name)"
# create venv if missing
if (-not (Test-Path ".venv")) {
    Write-Host "[setup] Creating virtualenv .venv"
    python -m venv .venv
}

# Bypass execution policy for this process to allow activation script
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Activate venv
$activate = Join-Path ".\.venv\Scripts" "Activate.ps1"
. $activate

Write-Host "[setup] Upgrading pip and installing package"
python -m pip install --upgrade pip setuptools wheel
pip install -e .

Write-Host "[setup] Running GUI"
python .\scripts\run_gui.py
