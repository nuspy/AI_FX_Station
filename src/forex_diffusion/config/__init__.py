"""Configuration management for ForexGPT."""
from .sssd_config import SSSDConfig, load_sssd_config

__all__ = ["SSSDConfig", "load_sssd_config"]
