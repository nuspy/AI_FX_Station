@echo off
REM NVIDIA GPU Acceleration Stack Installer for Windows
REM
REM This script installs optional NVIDIA dependencies for GPU acceleration.
REM
REM Usage:
REM   install_nvidia_stack.bat [check|apex|flash-attn|xformers|dali|all]
REM
REM Options:
REM   check      - Check GPU compatibility only
REM   apex       - Install NVIDIA APEX (fused optimizers)
REM   flash-attn - Install Flash Attention 2 (requires Ampere+ GPU)
REM   xformers   - Install xFormers (memory efficient transformers)
REM   dali       - Install NVIDIA DALI (data loading)
REM   all        - Install all components (default)

setlocal

echo ========================================
echo NVIDIA GPU Acceleration Stack Installer
echo ForexGPT - Advanced ML Trading System
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.10+ first
    pause
    exit /b 1
)

REM Get the argument (default to "all" if none provided)
set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=all"

REM Map action to Python script arguments
if /i "%ACTION%"=="check" (
    set "ARGS=--check"
) else if /i "%ACTION%"=="apex" (
    set "ARGS=--apex"
) else if /i "%ACTION%"=="flash-attn" (
    set "ARGS=--flash-attn"
) else if /i "%ACTION%"=="xformers" (
    set "ARGS=--xformers"
) else if /i "%ACTION%"=="dali" (
    set "ARGS=--dali"
) else if /i "%ACTION%"=="all" (
    set "ARGS=--all"
) else (
    echo ERROR: Unknown action "%ACTION%"
    echo.
    echo Usage: install_nvidia_stack.bat [check^|apex^|flash-attn^|xformers^|dali^|all]
    pause
    exit /b 1
)

REM Run the Python installer script
echo Running: python install_nvidia_stack.py %ARGS%
echo.
python install_nvidia_stack.py %ARGS%

set "EXIT_CODE=%errorlevel%"

echo.
if %EXIT_CODE% equ 0 (
    echo ========================================
    echo Installation completed successfully!
    echo ========================================
) else (
    echo ========================================
    echo Installation completed with errors
    echo ========================================
)

pause
exit /b %EXIT_CODE%
