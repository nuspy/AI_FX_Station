@echo off
REM Full NVIDIA Stack Installer for ForexGPT
REM
REM This script installs the complete NVIDIA GPU acceleration stack:
REM 1. Base dependencies (nvidia-ml-py, xformers)
REM 2. Compiled components (APEX, Flash Attention)
REM 3. DALI via WSL (optional)
REM
REM Requirements:
REM - Python 3.10+ with pip
REM - CUDA 12.x installed
REM - Visual Studio Build Tools (for APEX compilation)
REM - Ampere+ GPU for Flash Attention (RTX 30xx/40xx, A100)
REM - WSL2 with Ubuntu for DALI (optional)

setlocal enabledelayedexpansion

echo ========================================
echo ForexGPT - Full NVIDIA Stack Installer
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Please install Python 3.10+ first
    pause
    exit /b 1
)

echo [1/5] Installing base NVIDIA dependencies...
echo Command: pip install -e ".[nvidia-full]"
echo.
pip install -e ".[nvidia-full]"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install base dependencies
    pause
    exit /b 1
)
echo ✓ Base dependencies installed
echo.

echo [2/5] Installing xFormers (auto-versioned)...
echo.
python install_nvidia_stack.py --xformers
if %errorlevel% neq 0 (
    echo WARNING: xFormers installation failed
    echo This is optional - continuing...
)
echo.

echo [3/5] Installing NVIDIA APEX (fused optimizers)...
echo NOTE: This requires C++ compiler and may take 10-30 minutes
echo.
set /p "INSTALL_APEX=Install APEX? (y/N): "
if /i "!INSTALL_APEX!"=="y" (
    python install_nvidia_stack.py --apex
    if %errorlevel% neq 0 (
        echo WARNING: APEX installation failed
        echo Check that Visual Studio Build Tools are installed
    )
) else (
    echo Skipping APEX installation
)
echo.

echo [4/5] Installing Flash Attention 2...
echo NOTE: Requires Ampere+ GPU (RTX 30xx/40xx, A100)
echo.
set /p "INSTALL_FLASH=Install Flash Attention 2? (y/N): "
if /i "!INSTALL_FLASH!"=="y" (
    python install_nvidia_stack.py --flash-attn
    if %errorlevel% neq 0 (
        echo WARNING: Flash Attention installation failed
        echo This requires Ampere+ GPU (compute capability >= 8.0)
    )
) else (
    echo Skipping Flash Attention installation
)
echo.

echo [5/5] NVIDIA DALI (Linux/WSL only)...
echo NOTE: DALI requires WSL2 with Ubuntu on Windows
echo.
set /p "INSTALL_DALI=Install DALI via WSL? (y/N): "
if /i "!INSTALL_DALI!"=="y" (
    wsl --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: WSL not found
        echo Install WSL2 with: wsl --install
        echo Then retry DALI installation
    ) else (
        echo Running: wsl python install_nvidia_stack.py --dali
        wsl python install_nvidia_stack.py --dali
        if %errorlevel% neq 0 (
            echo WARNING: DALI installation failed
            echo Make sure Python is installed in WSL
        )
    )
) else (
    echo Skipping DALI installation (Windows not supported)
)
echo.

echo ========================================
echo Installation Summary
echo ========================================
echo.

REM Check installed packages
echo Checking installed packages...
echo.

python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul
python -c "import xformers; print(f'xFormers: {xformers.__version__}')" 2>nul || echo xFormers: Not installed
python -c "import apex; print('APEX: Installed')" 2>nul || echo APEX: Not installed
python -c "import flash_attn; print('Flash Attention: Installed')" 2>nul || echo Flash Attention: Not installed
echo.

echo ========================================
echo Next Steps
echo ========================================
echo.
echo 1. Restart your IDE/terminal
echo 2. Test GPU: python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
echo 3. Run tests: pytest tests/test_e2e_complete.py -v
echo 4. Start training: fx-train-lightning --device cuda
echo.

pause
exit /b 0
