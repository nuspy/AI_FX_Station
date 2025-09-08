"""
Central logging configuration for MagicForex.

- Uses loguru for rich logging when available.
- Adds file sink (rotation/retention) and console sink based on config.
- Installs an InterceptHandler to forward stdlib logging to loguru.
"""

from __future__ import annotations

from importlib import import_module as _import_module
import importlib
import sys
import os
from typing import Optional, Any

# NOTE: avoid importing stdlib 'logging' at module import time to prevent circular imports.
# We'll import it dynamically inside functions when needed.

# Try multiple import strategies for get_config so module works when executed
# as part of a package or as a standalone script (runpy/run_module).
try:
    # Preferred: relative import when package context is available
    from .config import get_config
except Exception:
    try:
        # Absolute package path used in some run contexts
        from src.forex_diffusion.utils.config import get_config
    except Exception:
        try:
            cfg_mod = _import_module("src.forex_diffusion.utils.config")
            get_config = getattr(cfg_mod, "get_config")
        except Exception:
            try:
                cfg_mod = _import_module("utils.config")
                get_config = getattr(cfg_mod, "get_config")
            except Exception:
                # Last-resort stub
                def get_config():
                    return {}

# Keep a flag so setup_logging is idempotent in the process
_LOGGING_INITIALIZED = False

# Placeholder for the loguru logger object; set when loguru is imported in setup_logging/get_logger
_logger = None  # type: ignore

def _ensure_loguru() -> None:
    """Lazily import loguru.logger into _logger if available."""
    global _logger
    if _logger is None:
        try:
            mod = _import_module("loguru")
            _logger = getattr(mod, "logger")
        except Exception:
            _logger = None


def _make_intercept_handler():
    """
    Dynamically create and return an InterceptHandler class instance that subclasses
    the real stdlib logging.Handler. This avoids referencing stdlib logging at module import time.
    """
    logging_mod = importlib.import_module("logging")
    # Define the handler class dynamically so it uses the real stdlib logging module.
    class InterceptHandler(logging_mod.Handler):
        def emit(self, record: logging_mod.LogRecord) -> None:
            try:
                global _logger
                # If loguru isn't ready, skip forwarding.
                if _logger is None:
                    return
                # Map level
                level_name = record.levelname
                try:
                    lvl = _logger.level(level_name).name
                except Exception:
                    lvl = record.levelno
                # Find caller depth
                frame = logging_mod.currentframe()
                depth = 2
                while frame and getattr(frame.f_code, "co_filename", None) == getattr(logging_mod, "__file__", None):
                    frame = frame.f_back
                    depth += 1
                _logger.opt(depth=depth, exception=record.exc_info).log(lvl, record.getMessage())
            except Exception:
                try:
                    if _logger is not None:
                        _logger.exception("Failed to emit log record from stdlib logging")
                except Exception:
                    pass
    return InterceptHandler()


def setup_logging(config: Optional[object] = None) -> None:
    """
    Initialize logging according to configuration from get_config().
    Idempotent.
    """
    global _LOGGING_INITIALIZED, _logger
    if _LOGGING_INITIALIZED:
        return

    if config is None:
        config = get_config()

    _ensure_loguru()

    # Prepare log directory and options
    log_cfg = getattr(config, "logging", None) or {}
    file_cfg = log_cfg.get("file", {}) if isinstance(log_cfg, dict) else getattr(log_cfg, "file", None)

    log_dir = None
    rotation = "10 MB"
    retention = "14 days"
    level = getattr(config, "app", {}).get("debug", False) and "DEBUG" or getattr(config, "logging", {}).get("level", "INFO")

    if file_cfg:
        log_dir = file_cfg.get("dir", "./logs")
        rotation = file_cfg.get("rotation", rotation)
        retention = file_cfg.get("retention", retention)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, "magicforex_{time:YYYY-MM-DD}.log")
    else:
        file_path = None

    # If loguru available, configure it and forward stdlib logs to it.
    if _logger is not None:
        _logger.remove()
        if file_path:
            _logger.add(
                file_path,
                rotation=rotation,
                retention=retention,
                level=level,
                enqueue=True,
                backtrace=True,
                diagnose=False,
            )
        _logger.add(
            sink=lambda msg: print(msg, end=""),
            level=level,
            colorize=True,
            enqueue=True,
        )
        # Attach intercept handler to stdlib logging (imported at runtime)
        logging_mod = importlib.import_module("logging")
        try:
            handler = _make_intercept_handler()
            logging_mod.root.handlers = [handler]
            logging_mod.root.setLevel(level)
            try:
                logging_mod.getLogger("asyncio").setLevel(logging_mod.WARNING)
            except Exception:
                pass
            try:
                logging_mod.getLogger("urllib3").setLevel(logging_mod.WARNING)
            except Exception:
                pass
        except Exception:
            # If any of the interception fails, continue (loguru still works)
            pass

        _logger.info("Logging initialized (level={}, file={})", level, file_path or "disabled")
    else:
        # Fallback to stdlib basicConfig
        try:
            logging_mod = importlib.import_module("logging")
            numeric_level = getattr(logging_mod, level, logging_mod.INFO)
            logging_mod.basicConfig(level=numeric_level)
            logging_mod.getLogger(__name__).info("Stdlib logging initialized (loguru not available).")
        except Exception:
            pass

    _LOGGING_INITIALIZED = True


def get_logger() -> Any:
    """
    Return loguru logger if available, otherwise return a simple stdlib-compatible wrapper.
    """
    global _logger
    if not _LOGGING_INITIALIZED:
        try:
            setup_logging()
        except Exception:
            pass

    _ensure_loguru()
    if _logger is not None:
        return _logger

    # Fallback wrapper using stdlib logging imported at runtime
    logging_mod = importlib.import_module("logging")
    std = logging_mod.getLogger("magicforex")

    class _SimpleWrapper:
        def info(self, *args, **kwargs):
            std.info(*args, **kwargs)
        def warning(self, *args, **kwargs):
            std.warning(*args, **kwargs)
        def exception(self, *args, **kwargs):
            std.exception(*args, **kwargs)
        def debug(self, *args, **kwargs):
            std.debug(*args, **kwargs)
        def add(self, *args, **kwargs):
            return None
        def opt(self, *args, **kwargs):
            return self
        def log(self, level, *args, **kwargs):
            std.log(level, *args, **kwargs)

    return _SimpleWrapper()


# Expose a factory for InterceptHandler so callers can access a handler instance if needed.
def InterceptHandler():
    return _make_intercept_handler()

__all__ = ["setup_logging", "get_logger", "InterceptHandler"]
