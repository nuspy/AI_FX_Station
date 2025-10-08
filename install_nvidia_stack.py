#!/usr/bin/env python3
"""
NVIDIA GPU Acceleration Stack Installer

This script installs optional NVIDIA dependencies for GPU acceleration:
- NVIDIA APEX (fused optimizers)
- Flash Attention 2 (requires Ampere+ GPU)
- xFormers (memory efficient transformers with PyTorch version matching)
- NVIDIA DALI (optional data loading acceleration)

Usage:
    python install_nvidia_stack.py [--apex] [--flash-attn] [--xformers] [--dali] [--all]

Options:
    --apex         Install NVIDIA APEX (fused optimizers)
    --flash-attn   Install Flash Attention 2 (requires Ampere+ GPU: RTX 30xx/40xx, A100, etc.)
    --xformers     Install xFormers with version matching installed PyTorch (auto-detects 2.5.x vs 2.7.x)
    --dali         Install NVIDIA DALI (data loading acceleration)
    --all          Install all NVIDIA dependencies
    --check        Check GPU compatibility without installing

Requirements:
    - CUDA 12.x installed
    - For Flash Attention: Ampere or newer GPU (compute capability >= 8.0)
    - For xFormers: Compatible PyTorch version (2.5.x → xformers 0.0.30, 2.7.x → xformers 0.0.31.post1)

Note on xFormers compatibility:
    xformers 0.0.31.post1 requires torch==2.7.1 exactly
    xformers 0.0.30 is compatible with torch 2.5.x
    This script auto-detects your PyTorch version and installs the correct xformers version.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Installing: {description}")
    print(f"Command: {cmd}")
    print('='*60)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {description}")
        print(f"Error: {e}")
        return False


def check_gpu_compatibility():
    """Check NVIDIA GPU compatibility"""
    print("\n" + "="*60)
    print("Checking GPU Compatibility")
    print("="*60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ No CUDA-capable GPU detected")
            print("   PyTorch cannot find CUDA. Please install CUDA 12.x first.")
            return False

        gpu_count = torch.cuda.device_count()
        print(f"✅ Found {gpu_count} CUDA GPU(s)")

        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            compute = f"{capability[0]}.{capability[1]}"

            print(f"\n   GPU {i}: {name}")
            print(f"   Compute Capability: {compute}")

            # Check for Flash Attention compatibility (requires compute >= 8.0)
            if capability[0] >= 8:
                print(f"   ✅ Compatible with Flash Attention 2")
            else:
                print(f"   ⚠️  Not compatible with Flash Attention 2 (requires compute >= 8.0)")

        cuda_version = torch.version.cuda
        print(f"\n   CUDA Version: {cuda_version}")

        return True

    except ImportError:
        print("❌ PyTorch not installed")
        print("   Please install PyTorch first:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def install_apex():
    """Install NVIDIA APEX"""
    cmd = (
        "pip install -v --disable-pip-version-check --no-cache-dir "
        "--no-build-isolation "
        "--config-settings '--build-option=--cpp_ext' "
        "--config-settings '--build-option=--cuda_ext' "
        "git+https://github.com/NVIDIA/apex.git"
    )
    return run_command(cmd, "NVIDIA APEX (fused optimizers)")


def install_flash_attention():
    """Install Flash Attention 2"""
    # Check GPU compatibility first
    try:
        import torch
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            if capability[0] < 8:
                print("\n⚠️  WARNING: Flash Attention 2 requires Ampere or newer GPU")
                print(f"   Your GPU has compute capability {capability[0]}.{capability[1]}")
                print("   Flash Attention requires compute capability >= 8.0")
                response = input("   Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    print("   Skipping Flash Attention installation")
                    return False
    except:
        pass

    cmd = "pip install flash-attn --no-build-isolation"
    return run_command(cmd, "Flash Attention 2")


def install_xformers():
    """Install xFormers with compatible PyTorch version"""
    print("\n" + "="*60)
    print("Installing xFormers")
    print("="*60)

    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # Remove +cu121 suffix
        print(f"\nDetected PyTorch version: {torch_version}")

        # xformers 0.0.31.post1 requires torch==2.7.1
        # For older torch versions, we need compatible xformers
        if torch_version.startswith("2.5"):
            # Use xformers compatible with torch 2.5.x
            cmd = "pip install xformers==0.0.30"
            print(f"Installing xformers 0.0.30 (compatible with PyTorch {torch_version})")
        elif torch_version.startswith("2.7"):
            cmd = "pip install xformers==0.0.31.post1"
            print(f"Installing xformers 0.0.31.post1 (compatible with PyTorch {torch_version})")
        else:
            print(f"⚠️  PyTorch {torch_version} detected - installing latest compatible xformers")
            cmd = "pip install xformers"

        return run_command(cmd, f"xFormers (for PyTorch {torch_version})")

    except ImportError:
        print("❌ PyTorch not installed - cannot determine compatible xformers version")
        print("   Please install PyTorch first")
        return False
    except Exception as e:
        print(f"❌ Error determining xformers version: {e}")
        return False


def install_dali():
    """Install NVIDIA DALI"""
    cmd = "pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120"
    return run_command(cmd, "NVIDIA DALI (data loading acceleration)")


def main():
    parser = argparse.ArgumentParser(
        description="Install NVIDIA GPU acceleration stack for ForexGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--apex', action='store_true', help='Install NVIDIA APEX')
    parser.add_argument('--flash-attn', action='store_true', help='Install Flash Attention 2')
    parser.add_argument('--xformers', action='store_true', help='Install xFormers (memory efficient transformers)')
    parser.add_argument('--dali', action='store_true', help='Install NVIDIA DALI')
    parser.add_argument('--all', action='store_true', help='Install all NVIDIA dependencies')
    parser.add_argument('--check', action='store_true', help='Check GPU compatibility only')

    args = parser.parse_args()

    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    print("\n" + "="*60)
    print("NVIDIA GPU Acceleration Stack Installer")
    print("ForexGPT - Advanced ML Trading System")
    print("="*60)

    # Always check compatibility first
    if not check_gpu_compatibility():
        print("\n❌ GPU compatibility check failed")
        print("   Please fix GPU/CUDA issues before installing NVIDIA stack")
        return 1

    if args.check:
        print("\n✅ GPU compatibility check complete")
        return 0

    # Track installation results
    results = {}

    # Install requested components
    if args.all or args.apex:
        results['APEX'] = install_apex()

    if args.all or args.flash_attn:
        results['Flash Attention'] = install_flash_attention()

    if args.all or args.xformers:
        results['xFormers'] = install_xformers()

    if args.all or args.dali:
        results['DALI'] = install_dali()

    # Summary
    print("\n" + "="*60)
    print("Installation Summary")
    print("="*60)

    for component, success in results.items():
        status = "✅ Installed" if success else "❌ Failed"
        print(f"{component}: {status}")

    all_success = all(results.values())

    if all_success:
        print("\n✅ All requested components installed successfully!")
        print("\nNext steps:")
        print("1. Restart your Python kernel/IDE")
        print("2. Test GPU acceleration with: python -c 'import torch; print(torch.cuda.is_available())'")
        print("3. Run training with GPU: fx-train-lightning --device cuda")
        return 0
    else:
        print("\n⚠️  Some components failed to install")
        print("   Check the error messages above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
