# NVIDIA GPU Acceleration Stack Installation Guide

This guide explains how to install optional NVIDIA dependencies for GPU acceleration in ForexGPT.

## Quick Start

### Windows
```bash
# Install all NVIDIA components
install_nvidia_stack.bat all

# Or install specific components
install_nvidia_stack.bat xformers
```

### Linux/Mac
```bash
# Install all NVIDIA components
python install_nvidia_stack.py --all

# Or install specific components
python install_nvidia_stack.py --xformers
```

## Available Components

### 1. xFormers (Memory Efficient Transformers)
**Recommended for transformer-based models**

- Provides memory-efficient attention mechanisms
- Reduces VRAM usage significantly
- Auto-detects PyTorch version and installs compatible version

**Important**: xFormers version must match PyTorch version:
- PyTorch 2.5.x → xFormers 0.0.30
- PyTorch 2.7.x → xFormers 0.0.31.post1

```bash
# Windows
install_nvidia_stack.bat xformers

# Linux/Mac
python install_nvidia_stack.py --xformers
```

### 2. Flash Attention 2
**Requires Ampere or newer GPU (RTX 30xx/40xx, A100, etc.)**

- Ultra-fast attention mechanism
- Requires GPU with compute capability >= 8.0
- Dramatically speeds up transformer training

```bash
# Windows
install_nvidia_stack.bat flash-attn

# Linux/Mac
python install_nvidia_stack.py --flash-attn
```

### 3. NVIDIA APEX
**Fused optimizers for faster training**

- Provides fused Adam, SGD, and other optimizers
- Mixed precision training utilities
- Compatible with all CUDA-capable GPUs

```bash
# Windows
install_nvidia_stack.bat apex

# Linux/Mac
python install_nvidia_stack.py --apex
```

### 4. NVIDIA DALI
**Data loading acceleration (Linux only)**

- GPU-accelerated data preprocessing
- Not available on Windows

```bash
# Linux only
python install_nvidia_stack.py --dali
```

## Checking GPU Compatibility

Before installing, check your GPU compatibility:

```bash
# Windows
install_nvidia_stack.bat check

# Linux/Mac
python install_nvidia_stack.py --check
```

This will show:
- Detected GPUs
- Compute capability
- CUDA version
- Flash Attention compatibility

## Troubleshooting

### xFormers Version Mismatch

If you see: `xformers 0.0.31.post1 requires torch==2.7.1, but you have torch 2.5.1+cu121`

**Solution**: Run the installer to fix the version mismatch:
```bash
pip uninstall xformers
install_nvidia_stack.bat xformers  # Windows
python install_nvidia_stack.py --xformers  # Linux/Mac
```

The script will automatically detect your PyTorch version and install the correct xFormers version.

### Flash Attention Installation Fails

If Flash Attention installation fails, check:

1. **GPU Compatibility**: Requires Ampere or newer (RTX 30xx/40xx, A100)
   ```bash
   install_nvidia_stack.bat check
   ```

2. **CUDA Version**: Requires CUDA 12.x
   ```bash
   nvcc --version
   ```

3. **Compiler**: Requires C++ compiler (Visual Studio on Windows, GCC on Linux)

### APEX Build Errors

APEX requires compilation from source. Common issues:

1. **No C++ Compiler**: Install Visual Studio Build Tools (Windows) or GCC (Linux)
2. **CUDA Toolkit**: Must be installed system-wide
3. **Long Build Time**: APEX can take 10-30 minutes to compile

## Version Compatibility Matrix

| Component | PyTorch 2.5.x | PyTorch 2.7.x | GPU Requirement |
|-----------|---------------|---------------|-----------------|
| xFormers  | 0.0.30        | 0.0.31.post1  | CUDA-capable    |
| Flash Attention 2 | ✅ | ✅ | Compute >= 8.0 (Ampere+) |
| APEX      | ✅            | ✅            | CUDA-capable    |
| DALI      | ✅ (Linux)    | ✅ (Linux)    | CUDA-capable    |

## Current PyTorch Installation

Check your current PyTorch version:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Recommended Installation Order

For best results, install in this order:

1. **Check compatibility**: `install_nvidia_stack.bat check`
2. **Install xFormers**: `install_nvidia_stack.bat xformers`
3. **Install Flash Attention**: `install_nvidia_stack.bat flash-attn` (if compatible GPU)
4. **Install APEX**: `install_nvidia_stack.bat apex` (optional, slow to build)
5. **Install DALI**: `python install_nvidia_stack.py --dali` (Linux only, optional)

Or install all at once: `install_nvidia_stack.bat all`

## Performance Impact

Expected speedups with NVIDIA stack:

- **xFormers**: 20-40% VRAM reduction, 10-20% speed increase
- **Flash Attention 2**: 40-60% speed increase for attention layers
- **APEX**: 5-15% speed increase from fused optimizers
- **DALI**: 30-50% faster data loading (Linux only)

## Getting Help

If installation fails:

1. Check the error messages in the terminal
2. Verify GPU compatibility: `install_nvidia_stack.bat check`
3. Check CUDA installation: `nvcc --version`
4. Review the component's official documentation:
   - [xFormers](https://github.com/facebookresearch/xformers)
   - [Flash Attention](https://github.com/Dao-AILab/flash-attention)
   - [NVIDIA APEX](https://github.com/NVIDIA/apex)
   - [NVIDIA DALI](https://github.com/NVIDIA/DALI)

## Uninstalling

To uninstall components:

```bash
pip uninstall xformers flash-attn apex nvidia-dali-cuda120
```

Then reinstall with the correct versions using the installer scripts.
