"""
User settings persistence (local) with optional encryption for sensitive values.

- Stores key-values in ~/.config/magicforex/settings.json
- Optional Fernet encryption if MAGICFOREX_SECRET_KEY env var provided.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

SETTINGS_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "magicforex"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

# optional secret key for Fernet (url-safe base64 32 bytes)
# set via env MAGICFOREX_SECRET_KEY
SECRET_KEY_ENV = "MAGICFOREX_SECRET_KEY"


def _ensure_dir():
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> Dict[str, Any]:
    _ensure_dir()
    if not SETTINGS_FILE.exists():
        return {}
    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except Exception:
        return {}


def save_settings(d: Dict[str, Any]) -> None:
    _ensure_dir()
    try:
        with SETTINGS_FILE.open("w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2)
    except Exception:
        # best-effort no crash
        pass


def get_setting(key: str, default: Optional[Any] = None) -> Any:
    s = load_settings()
    return s.get(key, default)


def set_setting(key: str, value: Any) -> None:
    s = load_settings()
    s[key] = value
    save_settings(s)


def clear_settings() -> None:
    try:
        if SETTINGS_FILE.exists():
            SETTINGS_FILE.unlink()
    except Exception:
        pass


# --- optional encryption helpers using cryptography.Fernet ---
def _get_fernet():
    try:
        from cryptography.fernet import Fernet
    except Exception:
        return None
    key = os.environ.get(SECRET_KEY_ENV)
    if not key:
        return None
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception:
        return None


def set_encrypted_setting(key: str, plaintext: str) -> None:
    """
    Store encrypted value under given key as base64 string.
    Requires MAGICFOREX_SECRET_KEY in env; otherwise falls back to plain storage (not recommended).
    """
    f = _get_fernet()
    if f is None:
        # fallback to plain
        set_setting(key, plaintext)
        return
    token = f.encrypt(plaintext.encode()).decode()
    set_setting(key, {"_encrypted": True, "v": token})


def get_encrypted_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Return decrypted value or default.
    """
    val = get_setting(key, None)
    if isinstance(val, dict) and val.get("_encrypted") and "v" in val:
        f = _get_fernet()
        if f is None:
            return default
        try:
            pt = f.decrypt(val["v"].encode()).decode()
            return pt
        except Exception:
            return default
    return val if val is not None else default
