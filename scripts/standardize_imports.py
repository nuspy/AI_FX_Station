"""
Import Standardization Script

Applies PEP 8 import ordering to all Python files.
Part of IMPORT-001 implementation.

Uses isort for automatic import sorting.
"""
import subprocess
import sys
from pathlib import Path

def standardize_imports(directory: str = "src/forex_diffusion"):
    """
    Standardize imports in all Python files.
    
    Order (PEP 8):
    1. Future imports
    2. Standard library
    3. Third-party
    4. Local/relative imports
    """
    root = Path(directory)
    
    if not root.exists():
        print(f"Error: Directory {directory} does not exist")
        return False
    
    # Find all Python files
    py_files = list(root.rglob("*.py"))
    
    print(f"Found {len(py_files)} Python files in {directory}")
    print("Standardizing imports with isort...")
    
    # Run isort
    try:
        result = subprocess.run(
            ["python", "-m", "isort", str(root), "--profile", "black"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print("✓ Import standardization complete")
            print(result.stdout)
            return True
        else:
            print("⚠ Some files may need manual review")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("Error: isort not installed")
        print("Install with: pip install isort")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def check_imports(directory: str = "src/forex_diffusion"):
    """Check if imports need standardization"""
    root = Path(directory)
    
    try:
        result = subprocess.run(
            ["python", "-m", "isort", str(root), "--check-only", "--profile", "black"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print("✓ All imports already standardized")
            return True
        else:
            print("⚠ Some imports need standardization:")
            print(result.stdout)
            return False
            
    except FileNotFoundError:
        print("Error: isort not installed")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize Python imports")
    parser.add_argument("--check", action="store_true", help="Check only, don't modify")
    parser.add_argument("--directory", default="src/forex_diffusion", help="Directory to process")
    
    args = parser.parse_args()
    
    if args.check:
        success = check_imports(args.directory)
    else:
        success = standardize_imports(args.directory)
    
    sys.exit(0 if success else 1)
