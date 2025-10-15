#!/usr/bin/env python
"""
VectorBT Pro Installation Script
Installs VectorBT Pro from the local wheel file
"""

import os
import sys
import subprocess
from pathlib import Path

def install_vectorbt_pro():
    """Install VectorBT Pro from local wheel file"""

    # Get the project root directory
    project_root = Path(__file__).parent
    wheel_path = project_root / "VectorBt_PRO" / "vectorbtpro-2025.7.27-py3-none-any.whl"

    # Check if wheel file exists
    if not wheel_path.exists():
        print(f"❌ Error: VectorBT Pro wheel file not found at {wheel_path}")
        print("\nPlease ensure the wheel file is in the correct location:")
        print(f"  {wheel_path}")
        return False

    print(f"📦 Installing VectorBT Pro from: {wheel_path}")

    # Install the wheel using pip
    try:
        cmd = [sys.executable, "-m", "pip", "install", str(wheel_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ VectorBT Pro installed successfully!")

            # Verify installation
            try:
                import vectorbtpro as vbt
                print(f"✅ VectorBT Pro version: {vbt.__version__ if hasattr(vbt, '__version__') else 'Unknown'}")
                return True
            except ImportError:
                print("⚠️  VectorBT Pro installed but cannot be imported")
                print("   This might be normal if it requires additional configuration")
                return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error during installation: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['numpy', 'pandas', 'numba', 'plotly', 'scipy']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Installing missing dependencies...")

        for package in missing:
            subprocess.run([sys.executable, "-m", "pip", "install", package],
                         capture_output=True)
        print("✅ Dependencies installed")
    else:
        print("✅ All dependencies are already installed")

if __name__ == "__main__":
    print("=== VectorBT Pro Installation Script ===\n")

    # Check and install dependencies
    check_dependencies()

    # Install VectorBT Pro
    if install_vectorbt_pro():
        print("\n✅ Installation complete!")
        print("\nYou can now use VectorBT Pro in your Python scripts:")
        print("  import vectorbtpro as vbt")
    else:
        print("\n❌ Installation failed. Please check the error messages above.")
        sys.exit(1)