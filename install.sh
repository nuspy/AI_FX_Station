#!/usr/bin/env bash
# ForexGPT Installation Script - Linux/macOS Bash
# Automated installation with venv creation, dependencies, and post-install steps

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_NVIDIA=false
SKIP_CTRADER=false
SKIP_VECTORBT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-nvidia)
            SKIP_NVIDIA=true
            shift
            ;;
        --skip-ctrader)
            SKIP_CTRADER=true
            shift
            ;;
        --skip-vectorbt)
            SKIP_VECTORBT=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--skip-nvidia] [--skip-ctrader] [--skip-vectorbt]"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}========================================"
echo "  ForexGPT Installation Script"
echo "========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Check Python version
echo -e "${YELLOW}[1/8] Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}  ERROR: python3 not found in PATH${NC}"
    echo -e "${RED}  Please install Python 3.10+ from https://www.python.org/downloads/${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}  Found: Python $PYTHON_VERSION${NC}"

# Extract major.minor version
MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo -e "${RED}  ERROR: Python 3.10+ required, found $MAJOR.$MINOR${NC}"
    exit 1
fi

# Step 2: Create virtual environment
echo -e "\n${YELLOW}[2/8] Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${GREEN}  Virtual environment already exists, skipping...${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}  Created virtual environment: venv/${NC}"
fi

# Step 3: Activate virtual environment
echo -e "\n${YELLOW}[3/8] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}  Virtual environment activated${NC}"

# Step 4: Upgrade pip
echo -e "\n${YELLOW}[4/8] Upgrading pip...${NC}"
python -m pip install --upgrade pip setuptools wheel
echo -e "${GREEN}  pip upgraded successfully${NC}"

# Step 5: Install PyTorch with CUDA support (before main install)
echo -e "\n${YELLOW}[5/8] Installing PyTorch with CUDA 13.0 support...${NC}"
echo -e "  This may take several minutes..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
echo -e "${GREEN}  PyTorch with CUDA installed${NC}"

# Step 6: Install main package
echo -e "\n${YELLOW}[6/8] Installing ForexGPT package and dependencies...${NC}"
echo -e "  This will take 10-20 minutes depending on your internet connection..."
python -m pip install -e .
echo -e "${GREEN}  Main package installed successfully${NC}"

# Step 7: Post-installation - Optional packages
echo -e "\n${YELLOW}[7/8] Installing optional packages...${NC}"

# 7a. VectorBT Pro
if [ "$SKIP_VECTORBT" = false ]; then
    if [ -f "VectorBt_PRO/vectorbtpro-2025.7.27-py3-none-any.whl" ]; then
        echo -e "${CYAN}  Installing VectorBT Pro...${NC}"
        python -m pip install ./VectorBt_PRO/vectorbtpro-2025.7.27-py3-none-any.whl
        echo -e "${GREEN}  VectorBT Pro installed${NC}"
    else
        echo -e "${YELLOW}  WARNING: VectorBT Pro wheel not found in VectorBt_PRO/${NC}"
        echo -e "${YELLOW}  Skipping VectorBT Pro installation${NC}"
    fi
fi

# 7b. cTrader Open API
if [ "$SKIP_CTRADER" = false ]; then
    echo -e "${CYAN}  Installing cTrader Open API (without deps to avoid conflicts)...${NC}"
    python -m pip install ctrader-open-api==0.9.2 --no-deps
    echo -e "${GREEN}  cTrader Open API installed${NC}"
fi

# 7c. NVIDIA Optimization Stack
if [ "$SKIP_NVIDIA" = false ]; then
    echo -e "${CYAN}  Installing NVIDIA optimization stack...${NC}"
    echo -e "  This includes: xFormers, APEX (if available), Flash Attention 2 (if GPU supports it)"
    python install_nvidia_stack.py --all
    echo -e "${GREEN}  NVIDIA stack installation completed${NC}"
fi

# Step 8: Verify installation
echo -e "\n${YELLOW}[8/8] Verifying installation...${NC}"
echo -e "  Testing critical imports..."

python << 'EOF'
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
EOF

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}  All critical packages verified successfully!${NC}"
else
    echo -e "\n${YELLOW}  WARNING: Some packages failed verification${NC}"
    echo -e "${YELLOW}  The installation may still work, but some features might be missing${NC}"
fi

# Summary
echo -e "\n${CYAN}========================================"
echo -e "  Installation Complete!"
echo -e "========================================${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo -e "${NC}  1. Activate the virtual environment:${NC}"
echo -e "     ${GREEN}source venv/bin/activate${NC}"
echo ""
echo -e "  2. Set up your API keys (if using Tiingo or cTrader):"
echo -e "     ${GREEN}export TIINGO_APIKEY='your-api-key'${NC}"
echo ""
echo -e "  3. Run the application:"
echo -e "     ${GREEN}python -m forex_diffusion.main${NC}"
echo ""
echo -e "${YELLOW}For help, see README.md or run: python -m forex_diffusion.main --help${NC}"
echo ""
