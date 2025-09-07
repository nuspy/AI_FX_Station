"""
Central logging configuration for MagicForex.

- Uses loguru for rich logging.
- Adds file sink (rotation/retention) and console sink based on config.
- Installs an InterceptHandler to forward stdlib logging to loguru.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from loguru import logger

from .config import get_config

# Keep a flag so setup_logging is idempotent in the process
_LOGGING_INITIALIZED = False


class InterceptHandler(logging.Handler):
    """
    Default handler to intercept standard library logs and forward them to loguru.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Determine Loguru level
            level_name = record.levelname
            try:
                level = logger.level(level_name).name
            except Exception:
                level = record.levelno
            # Find caller depth so that loguru displays the correct source location.
            frame = logging.currentframe()
            depth = 2
            # Walk back to the first caller outside logging module
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
        except Exception:
            # Avoid crashing the application due to logging errors
            logger.exception("Failed to emit log record from stdlib logging")


def setup_logging(config: Optional[object] = None) -> None:
    """
    Initialize logging according to configuration from get_config().

    This function is idempotent and safe to call multiple times.
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    if config is None:
        config = get_config()

    # Prepare log directory
    log_cfg = getattr(config, "logging", None) or {}
    file_cfg = log_cfg.get("file", {}) if isinstance(log_cfg, dict) else getattr(log_cfg, "file", None)
    console_cfg = log_cfg.get("console", {}) if isinstance(log_cfg, dict) else getattr(log_cfg, "console", None)

    log_dir = None
    rotation = "10 MB"
    retention = "14 days"
    level = getattr(config, "app", {}).get("debug", False) and "DEBUG" or getattr(config, "logging", {}).get("level", "INFO")

    if file_cfg:
        log_dir = file_cfg.get("dir", "./logs")
        rotation = file_cfg.get("rotation", rotation)
        retention = file_cfg.get("retention", retention)

    # Ensure directory exists
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, "magicforex_{time:YYYY-MM-DD}.log")
    else:
        file_path = None

    # Remove existing handlers/sinks
    logger.remove()

    # Add file sink if configured
    if file_path:
        logger.add(
            file_path,
            rotation=rotation,
            retention=retention,
            level=level,
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )

    # Add console sink
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        enqueue=True,
    )

    # Intercept stdlib logging
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(level)

    # Optionally set some noisy libraries to WARNING
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.info("Logging initialized (level={}, file={})", level, file_path or "disabled")

    _LOGGING_INITIALIZED = True


def get_logger() -> "loguru.Logger":
    """
    Ensure logging is set up and return the loguru logger.
    """
    if not _LOGGING_INITIALIZED:
        try:
            setup_logging()
        except Exception:
            # Best-effort fallback
            logger.add(lambda msg: print(msg, end=""), level="INFO")
    return logger


__all__ = ["setup_logging", "get_logger", "InterceptHandler"]
