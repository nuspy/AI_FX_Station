"""Configuration management for ForexGPT."""
from .sssd_config import SSSDConfig, load_sssd_config
from ..utils.config import Settings, get_config, load_config


# Create a settings object for backward compatibility
class _SettingsProxy:
    """
    Proxy object for backward compatibility with old code that expects:
    from forex_diffusion.config import settings
    settings.database_path
    """
    
    def __init__(self):
        self._config = None
    
    def _get_config(self):
        if self._config is None:
            self._config = get_config()
        return self._config
    
    @property
    def database_path(self):
        """Extract database path from database_url."""
        config = self._get_config()
        db_url = config.db.database_url
        
        # Extract path from sqlite:///path/to/db.db
        if db_url.startswith("sqlite:///"):
            return db_url.replace("sqlite:///", "")
        elif db_url.startswith("sqlite://"):
            return db_url.replace("sqlite://", "")
        else:
            # For non-sqlite, return the full URL
            return db_url
    
    def __getattr__(self, name):
        """Forward all other attributes to the actual config."""
        return getattr(self._get_config(), name)


# Create singleton instance
settings = _SettingsProxy()

__all__ = [
    "SSSDConfig", 
    "load_sssd_config",
    "Settings",
    "get_config",
    "load_config",
    "settings"  # Backward compatibility
]
