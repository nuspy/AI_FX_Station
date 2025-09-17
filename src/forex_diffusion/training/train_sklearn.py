"""
Compatibility wrapper for launching the sklearn trainer.

Allows running:
    python -m src.forex_diffusion.training.train_sklearn

Resolves the real entrypoint from:
 - dynamic: f"{root}.train.train_sklearn" where root = __package__.rsplit('.', 1)[0]
 - absolute with src prefix: src.forex_diffusion.train.train_sklearn
 - plain package: forex_diffusion.train.train_sklearn
 - top-level script: train_sklearn.py
 - file-based fallback: load ../train/train_sklearn.py by path
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

def _resolve_main():
    # 1) Dynamic relative-to-package resolution
    try:
        pkg = __package__ or ""
        root = pkg.rsplit(".", 1)[0] if "." in pkg else "src.forex_diffusion"
        mod_name = f"{root}.train.train_sklearn"
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "main"):
            return getattr(mod, "main")
    except Exception:
        pass
    # 2) Absolute with 'src.' prefix
    try:
        mod = importlib.import_module("src.forex_diffusion.train.train_sklearn")
        if hasattr(mod, "main"):
            return getattr(mod, "main")
    except Exception:
        pass
    # 3) Plain package (when PYTHONPATH includes 'src')
    try:
        mod = importlib.import_module("forex_diffusion.train.train_sklearn")
        if hasattr(mod, "main"):
            return getattr(mod, "main")
    except Exception:
        pass
    # 4) Top-level script in repo root (as a last resort via module import)
    try:
        mod = importlib.import_module("train_sklearn")
        if hasattr(mod, "main"):
            return getattr(mod, "main")
    except Exception:
        pass
    # 5) File-based fallback: load ../train/train_sklearn.py relative to this file
    try:
        here = Path(__file__).resolve()
        candidate = here.parent.parent / "train" / "train_sklearn.py"
        if candidate.exists():
            spec = spec_from_file_location("fd_train_sklearn_file", str(candidate))
            if spec and spec.loader:
                mod = module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                if hasattr(mod, "main"):
                    return getattr(mod, "main")
    except Exception:
        pass

    def _err():
        sys.stderr.write(
            "[error] Unable to locate the real 'train_sklearn' entrypoint.\n"
            "Tried dynamic relative, 'src.forex_diffusion.train.train_sklearn', "
            "'forex_diffusion.train.train_sklearn', top-level 'train_sklearn.py', "
            "and file-based '../train/train_sklearn.py'.\n"
        )
        sys.exit(1)
    return _err

def main():
    real_main = _resolve_main()
    return real_main()

if __name__ == "__main__":
    main()
