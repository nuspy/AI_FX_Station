#!/usr/bin/env python3
"""
Install memory optimization packages for LDM4TS training.

This script installs SageAttention 2 and optionally FlashAttention 2
to reduce VRAM usage during training.

Usage:
    python install_memory_optimization.py              # Install SageAttention only (fast)
    python install_memory_optimization.py --flash      # Install SageAttention + FlashAttention
    python install_memory_optimization.py --check      # Check what's installed
"""
import subprocess
import sys
import argparse
from pathlib import Path


def check_gpu_capability():
    """Check GPU compute capability for FlashAttention 2 compatibility."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  No CUDA GPU detected")
            return None
        
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        capability_str = f"{capability[0]}.{capability[1]}"
        
        print(f"✓ GPU Detected: {gpu_name}")
        print(f"✓ Compute Capability: {capability_str}")
        
        # FlashAttention 2 requires compute capability >= 8.0 (Ampere+)
        if capability[0] >= 8:
            print("✓ GPU supports FlashAttention 2 (Ampere or newer)")
            return True
        else:
            print("⚠️  GPU does not support FlashAttention 2 (requires Ampere+)")
            print("   SageAttention 2 will work on this GPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, cannot detect GPU")
        return None


def check_cuda_version():
    """Check CUDA version."""
    try:
        import torch
        cuda_version = torch.version.cuda
        print(f"✓ CUDA Version: {cuda_version}")
        
        cuda_major = int(cuda_version.split('.')[0])
        if cuda_major >= 12:
            print("✓ CUDA 12+ detected, all packages compatible")
            return True
        elif cuda_major == 11:
            print("✓ CUDA 11.x detected")
            minor = int(cuda_version.split('.')[1])
            if minor >= 8:
                print("  - FlashAttention 2 compatible (CUDA 11.8+)")
                return True
            elif minor >= 6:
                print("  - SageAttention compatible (CUDA 11.6+)")
                print("  - FlashAttention requires CUDA 11.8+")
                return False
            else:
                print("  ⚠️  CUDA 11.6+ required for memory optimization")
                return False
        else:
            print("⚠️  CUDA version too old, upgrade to CUDA 11.6+")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed")
        return None


def install_sageattention():
    """Install SageAttention 2."""
    print("\n" + "="*60)
    print("Installing SageAttention 2...")
    print("="*60)
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "sageattention>=2.0.0"],
            check=True
        )
        print("✓ SageAttention 2 installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install SageAttention 2: {e}")
        return False


def install_flashattention():
    """Install FlashAttention 2."""
    print("\n" + "="*60)
    print("Installing FlashAttention 2...")
    print("="*60)
    
    import platform
    import torch
    
    # Get PyTorch and CUDA version
    torch_version = torch.__version__.split('+')[0]
    cuda_version = torch.version.cuda
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    
    print(f"Detected: Python {sys.version_info.major}.{sys.version_info.minor}, PyTorch {torch_version}, CUDA {cuda_version}")
    
    # Try pre-built wheel first (much faster and more reliable)
    print("\nAttempt 1: Trying pre-built wheel...")
    print("(Searching for compatible wheel on PyPI...)")
    
    try:
        # Try installing without building (will use pre-built wheel if available)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "flash-attn", "--prefer-binary"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ FlashAttention 2 installed from pre-built wheel!")
            return True
        else:
            print("✗ No pre-built wheel available for your configuration")
            print(f"   (Python {python_version}, PyTorch {torch_version}, CUDA {cuda_version})")
    except Exception as e:
        print(f"✗ Pre-built wheel installation failed: {e}")
    
    # Fallback: compile from source
    print("\nAttempt 2: Compiling from source...")
    print("This may take ~10-15 minutes...")
    print("Installing build dependencies...")
    
    try:
        # Install build tools
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "ninja", "packaging"],
            check=True,
            capture_output=True
        )
        
        # Compile FlashAttention
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"],
            check=True
        )
        print("✓ FlashAttention 2 compiled and installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install FlashAttention 2")
        print("\nCompilation failed. Common issues:")
        print("1. GPU not supported (requires Ampere+ / RTX 30/40 series)")
        print("2. CUDA version too old (requires CUDA 11.8+)")
        print("3. Missing Visual Studio Build Tools (Windows)")
        print("4. PyTorch/CUDA version mismatch")
        print("\nSOLUTION: Use SageAttention 2 instead")
        print("- Already installed and working ✓")
        print("- Provides ~35% VRAM reduction")
        print("- No compilation required")
        return False


def check_installed():
    """Check what's currently installed."""
    print("\n" + "="*60)
    print("Checking installed packages...")
    print("="*60)
    
    packages = {
        "sageattention": "SageAttention 2",
        "flash_attn": "FlashAttention 2"
    }
    
    for pkg_name, display_name in packages.items():
        try:
            __import__(pkg_name)
            print(f"✓ {display_name} is installed")
        except ImportError:
            print(f"✗ {display_name} is NOT installed")
    
    # Check memory_efficient_attention module
    try:
        from forex_diffusion.models.memory_efficient_attention import (
            SAGEATTENTION_AVAILABLE, 
            FLASHATTENTION_AVAILABLE
        )
        print("\n" + "="*60)
        print("ForexGPT Detection:")
        print("="*60)
        print(f"SageAttention available: {SAGEATTENTION_AVAILABLE}")
        print(f"FlashAttention available: {FLASHATTENTION_AVAILABLE}")
    except ImportError:
        print("\n⚠️  ForexGPT memory_efficient_attention module not found")
        print("   Make sure you're in the ForexGPT directory")


def main():
    parser = argparse.ArgumentParser(
        description="Install memory optimization packages for LDM4TS training"
    )
    parser.add_argument(
        "--flash",
        action="store_true",
        help="Also install FlashAttention 2 (requires Ampere+ GPU, ~10 min compilation)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check what's currently installed"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Install even if GPU compatibility check fails"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ForexGPT Memory Optimization Installer")
    print("="*60)
    
    # Check mode
    if args.check:
        check_installed()
        return
    
    # Check environment
    print("\nChecking environment...")
    cuda_ok = check_cuda_version()
    gpu_ok = check_gpu_capability()
    
    if not args.force:
        if cuda_ok is False:
            print("\n✗ CUDA version incompatible")
            print("  Upgrade to CUDA 11.6+ or use --force to install anyway")
            sys.exit(1)
        
        if args.flash and gpu_ok is False:
            print("\n✗ GPU not compatible with FlashAttention 2")
            print("  Use --force to install anyway (may not work)")
            response = input("Install SageAttention only instead? [y/N]: ")
            if response.lower() != 'y':
                sys.exit(1)
            args.flash = False
    
    # Install packages
    success = True
    
    # Always install SageAttention (fast, compatible with most GPUs)
    if not install_sageattention():
        success = False
    
    # Install FlashAttention if requested
    if args.flash:
        if not install_flashattention():
            success = False
            print("\n⚠️  FlashAttention 2 failed, but SageAttention is available")
    
    # Summary
    print("\n" + "="*60)
    print("Installation Summary")
    print("="*60)
    check_installed()
    
    if success:
        print("\n✓ Installation complete!")
        print("\nNext steps:")
        print("1. Open ForexGPT GUI")
        print("2. Go to: Generative Forecast → LDM4TS Training")
        print("3. Select Attention Backend:")
        if args.flash:
            print("   - Try 'FlashAttention 2' first (best VRAM savings)")
            print("   - Fallback to 'SageAttention 2' if FlashAttention fails")
        else:
            print("   - Select 'SageAttention 2' (~35% VRAM reduction)")
        print("4. Enable 'Gradient Checkpointing' for additional savings")
        print("\nSee docs/LDM4TS_MEMORY_OPTIMIZATION.md for recommended configs")
    else:
        print("\n⚠️  Some packages failed to install")
        print("   Check errors above and try manual installation")
        sys.exit(1)


if __name__ == "__main__":
    main()
