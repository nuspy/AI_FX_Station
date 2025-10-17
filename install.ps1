# ForexGPT Installation Script - Windows PowerShell
# Automated installation with venv creation, dependencies, and post-install steps

param(
    [switch]$SkipNvidia,
    [switch]$SkipCTrader,
    [switch]$SkipVectorBT
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ForexGPT Installation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR

# Step 1: Check Python version
Write-Host "[1/8] Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green

    # Extract version number
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "  ERROR: Python 3.10+ required, found $major.$minor" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "  ERROR: Python not found in PATH" -ForegroundColor Red
    Write-Host "  Please install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment
Write-Host "`n[2/8] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  Virtual environment already exists, skipping..." -ForegroundColor Gray
} else {
    python -m venv venv
    Write-Host "  Created virtual environment: venv/" -ForegroundColor Green
}

# Step 3: Activate virtual environment
Write-Host "`n[3/8] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "  Virtual environment activated" -ForegroundColor Green

# Step 4: Upgrade pip
Write-Host "`n[4/8] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
Write-Host "  pip upgraded successfully" -ForegroundColor Green

# Step 5: Install PyTorch with CUDA support (before main install)
Write-Host "`n[5/8] Installing PyTorch with CUDA 13.0 support..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor Gray
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
Write-Host "  PyTorch with CUDA installed" -ForegroundColor Green

# Step 6: Install main package
Write-Host "`n[6/8] Installing ForexGPT package and dependencies..." -ForegroundColor Yellow
Write-Host "  This will take 10-20 minutes depending on your internet connection..." -ForegroundColor Gray
python -m pip install -e .
Write-Host "  Main package installed successfully" -ForegroundColor Green

# Step 7: Post-installation - Optional packages
Write-Host "`n[7/8] Installing optional packages..." -ForegroundColor Yellow

# 7a. VectorBT Pro
if (-not $SkipVectorBT) {
    if (Test-Path "VectorBt_PRO\vectorbtpro-2025.7.27-py3-none-any.whl") {
        Write-Host "  Installing VectorBT Pro..." -ForegroundColor Cyan
        python -m pip install .\VectorBt_PRO\vectorbtpro-2025.7.27-py3-none-any.whl
        Write-Host "  VectorBT Pro installed" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: VectorBT Pro wheel not found in VectorBt_PRO/" -ForegroundColor Yellow
        Write-Host "  Skipping VectorBT Pro installation" -ForegroundColor Yellow
    }
}

# 7b. cTrader Open API
if (-not $SkipCTrader) {
    Write-Host "  Installing cTrader Open API (without deps to avoid conflicts)..." -ForegroundColor Cyan
    python -m pip install ctrader-open-api==0.9.2 --no-deps
    Write-Host "  cTrader Open API installed" -ForegroundColor Green
}

# 7c. NVIDIA Optimization Stack
if (-not $SkipNvidia) {
    Write-Host "  Installing NVIDIA optimization stack..." -ForegroundColor Cyan
    Write-Host "  This includes: xFormers, APEX (if available), Flash Attention 2 (if GPU supports it)" -ForegroundColor Gray
    python install_nvidia_stack.py --all
    Write-Host "  NVIDIA stack installation completed" -ForegroundColor Green
}

# Step 8: Verify installation
Write-Host "`n[8/8] Verifying installation..." -ForegroundColor Yellow
Write-Host "  Testing critical imports..." -ForegroundColor Gray

$verifyScript = @"
import sys
try:
    # Core packages
    import tensorflow as tf
    import torch
    import pandas as pd
    import numpy as np

    # GUI
    from PySide6 import QtWidgets

    # Trading packages
    from forex_diffusion import __version__ as fx_version

    print(f'  ✓ TensorFlow: {tf.__version__}')
    print(f'  ✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
    print(f'  ✓ Pandas: {pd.__version__}')
    print(f'  ✓ NumPy: {np.__version__}')
    print(f'  ✓ PySide6: OK')
    print(f'  ✓ ForexGPT: {fx_version}')

    # Optional: cTrader
    try:
        from ctrader_open_api import Client
        print('  ✓ cTrader Open API: OK')
    except ImportError:
        print('  ⚠ cTrader Open API: Not installed (optional)')

    sys.exit(0)
except Exception as e:
    print(f'  ✗ Error: {e}', file=sys.stderr)
    sys.exit(1)
"@

$verifyScript | python
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n  All critical packages verified successfully!" -ForegroundColor Green
} else {
    Write-Host "`n  WARNING: Some packages failed verification" -ForegroundColor Yellow
    Write-Host "  The installation may still work, but some features might be missing" -ForegroundColor Yellow
}

# Step 8: Download Hugging Face models for LDM4TS
Write-Host "`n[8/8] Downloading Hugging Face models for LDM4TS..." -ForegroundColor Yellow

& .\venv\Scripts\python.exe -c @"
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
import os

# Download to project models directory
vae_dir = os.path.join('models', 'vae')
clip_dir = os.path.join('models', 'clip')
os.makedirs(vae_dir, exist_ok=True)
os.makedirs(clip_dir, exist_ok=True)

print('  Downloading Stable Diffusion VAE...')
vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
vae.save_pretrained(vae_dir)
print(f'  VAE saved to: {vae_dir}')

print('  Downloading CLIP models...')
tok = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
tok.save_pretrained(clip_dir)
model.save_pretrained(clip_dir)
print(f'  CLIP saved to: {clip_dir}')
print('  All models ready!')
"@

if ($LASTEXITCODE -eq 0) {
    Write-Host "  Hugging Face models downloaded successfully!" -ForegroundColor Green
} else {
    Write-Host "  WARNING: Model download failed" -ForegroundColor Yellow
    Write-Host "  LDM4TS features may not work until models are downloaded" -ForegroundColor Yellow
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate the virtual environment:" -ForegroundColor White
Write-Host "     .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Set up your API keys (if using Tiingo or cTrader):" -ForegroundColor White
Write-Host "     `$env:TIINGO_APIKEY = 'your-api-key'" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Run the application:" -ForegroundColor White
Write-Host "     python -m forex_diffusion.main" -ForegroundColor Gray
Write-Host ""
Write-Host "For help, see README.md or run: python -m forex_diffusion.main --help" -ForegroundColor Yellow
Write-Host ""
