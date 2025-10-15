"""
Configuration file watcher for automatic cache invalidation.

Monitors config files for changes and automatically invalidates
service caches to ensure services use latest configuration.

Usage:
    from forex_diffusion.services.config_watcher import ConfigWatcher
    
    watcher = ConfigWatcher(
        services=[aggregator_service, dom_service, sentiment_service],
        config_paths=["configs/default.yaml", "configs/user.yaml"]
    )
    watcher.start()
    
    # Watcher runs in background and auto-invalidates caches on config change
    # ...
    
    watcher.stop()
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional, Callable
from datetime import datetime

from loguru import logger

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("watchdog not installed, config file watching unavailable")


class ConfigFileHandler(FileSystemEventHandler):
    """
    File system event handler for config file changes.
    
    Watches specified config files and triggers callback on modification.
    """
    
    def __init__(
        self,
        config_paths: List[Path],
        on_change: Callable[[Path], None],
        debounce_seconds: float = 1.0
    ):
        """
        Initialize config file handler.
        
        Args:
            config_paths: List of config file paths to watch
            on_change: Callback function called on config change
            debounce_seconds: Minimum time between change notifications (avoid spam)
        """
        super().__init__()
        self.config_paths = [Path(p).resolve() for p in config_paths]
        self.on_change = on_change
        self.debounce_seconds = debounce_seconds
        self._last_change_time: dict[Path, float] = {}
    
    def on_modified(self, event):
        """Handle file modification event."""
        if event.is_directory:
            return
        
        modified_path = Path(event.src_path).resolve()
        
        # Check if modified file is one we're watching
        if modified_path not in self.config_paths:
            return
        
        # Debounce (avoid multiple notifications for same change)
        now = time.time()
        last_change = self._last_change_time.get(modified_path, 0)
        
        if now - last_change < self.debounce_seconds:
            logger.debug(f"Ignoring debounced change: {modified_path.name}")
            return
        
        self._last_change_time[modified_path] = now
        
        # Trigger callback
        logger.info(f"Config file changed: {modified_path.name}")
        try:
            self.on_change(modified_path)
        except Exception as e:
            logger.error(f"Error in config change callback: {e}")


class ConfigWatcher:
    """
    Configuration file watcher with automatic cache invalidation.
    
    Monitors config files and automatically invalidates service symbol caches
    when configuration changes are detected.
    
    Features:
    - Automatic cache invalidation on config change
    - Debouncing to avoid multiple notifications
    - Support for multiple config files
    - Optional callback for custom actions
    - Fallback polling mode if watchdog not available
    
    Example:
        watcher = ConfigWatcher(
            services=[aggregator_service, dom_service],
            config_paths=["configs/default.yaml"]
        )
        watcher.start()
        
        # Config change detected → all services' caches invalidated automatically
    """
    
    def __init__(
        self,
        services: List = None,
        config_paths: List[str] = None,
        debounce_seconds: float = 1.0,
        on_change: Optional[Callable[[Path], None]] = None,
        use_polling: bool = False,
        polling_interval: float = 5.0
    ):
        """
        Initialize config watcher.
        
        Args:
            services: List of services with invalidate_symbol_cache() method
            config_paths: List of config file paths to watch
            debounce_seconds: Minimum time between notifications (default: 1s)
            on_change: Optional custom callback on config change
            use_polling: Force polling mode (default: False, use watchdog if available)
            polling_interval: Seconds between polls if using polling (default: 5s)
        """
        self.services = services or []
        self.config_paths = [Path(p).resolve() for p in (config_paths or ["configs/default.yaml"])]
        self.debounce_seconds = debounce_seconds
        self.custom_on_change = on_change
        self.polling_interval = polling_interval
        
        # Determine watching mode
        self.use_watchdog = WATCHDOG_AVAILABLE and not use_polling
        
        # Watchdog mode
        self._observer: Optional[Observer] = None
        
        # Polling mode
        self._polling_thread: Optional[threading.Thread] = None
        self._stop_polling = threading.Event()
        self._file_mtimes: dict[Path, float] = {}
        
        # Validate config paths exist
        for path in self.config_paths:
            if not path.exists():
                logger.warning(f"Config file does not exist: {path}")
    
    def start(self):
        """Start watching config files."""
        if self.use_watchdog:
            self._start_watchdog()
        else:
            self._start_polling()
        
        logger.info(
            f"ConfigWatcher started "
            f"(mode={'watchdog' if self.use_watchdog else 'polling'}, "
            f"files={[p.name for p in self.config_paths]})"
        )
    
    def stop(self):
        """Stop watching config files."""
        if self.use_watchdog and self._observer:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None
        elif self._polling_thread:
            self._stop_polling.set()
            self._polling_thread.join(timeout=2.0)
            self._polling_thread = None
        
        logger.info("ConfigWatcher stopped")
    
    def _start_watchdog(self):
        """Start watchdog observer for file system events."""
        if not WATCHDOG_AVAILABLE:
            logger.error("Cannot start watchdog: library not available")
            return
        
        # Create event handler
        handler = ConfigFileHandler(
            config_paths=self.config_paths,
            on_change=self._on_config_change,
            debounce_seconds=self.debounce_seconds
        )
        
        # Create observer and watch config directories
        self._observer = Observer()
        
        # Watch all parent directories (watchdog watches directories, not files)
        watched_dirs = set(path.parent for path in self.config_paths)
        for directory in watched_dirs:
            if directory.exists():
                self._observer.schedule(handler, str(directory), recursive=False)
                logger.debug(f"Watching directory: {directory}")
        
        self._observer.start()
    
    def _start_polling(self):
        """Start polling thread for file modification times."""
        # Initialize modification times
        for path in self.config_paths:
            if path.exists():
                self._file_mtimes[path] = path.stat().st_mtime
        
        # Start polling thread
        self._stop_polling.clear()
        self._polling_thread = threading.Thread(
            target=self._polling_loop,
            daemon=True,
            name="ConfigWatcherPolling"
        )
        self._polling_thread.start()
    
    def _polling_loop(self):
        """Polling loop for checking file modification times."""
        logger.info(f"Config watcher polling started (interval={self.polling_interval}s)")
        
        while not self._stop_polling.is_set():
            try:
                for path in self.config_paths:
                    if not path.exists():
                        continue
                    
                    current_mtime = path.stat().st_mtime
                    last_mtime = self._file_mtimes.get(path, 0)
                    
                    if current_mtime > last_mtime:
                        logger.info(f"Config file changed (polling): {path.name}")
                        self._file_mtimes[path] = current_mtime
                        self._on_config_change(path)
                
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
            
            # Sleep with interruptible wait
            self._stop_polling.wait(timeout=self.polling_interval)
    
    def _on_config_change(self, changed_path: Path):
        """
        Handle config file change.
        
        Invalidates symbol caches for all registered services and
        calls custom callback if provided.
        """
        logger.info(f"Processing config change: {changed_path.name}")
        
        # Invalidate all service caches
        for service in self.services:
            if hasattr(service, 'invalidate_symbol_cache'):
                try:
                    service.invalidate_symbol_cache()
                    logger.debug(f"Invalidated cache for: {service.service_name}")
                except Exception as e:
                    logger.error(f"Failed to invalidate cache for {service}: {e}")
        
        # Call custom callback if provided
        if self.custom_on_change:
            try:
                self.custom_on_change(changed_path)
            except Exception as e:
                logger.error(f"Error in custom config change callback: {e}")
    
    def add_service(self, service):
        """
        Add service to watch list.
        
        Args:
            service: Service with invalidate_symbol_cache() method
        """
        if service not in self.services:
            self.services.append(service)
            logger.debug(f"Added service to watcher: {service.service_name}")
    
    def remove_service(self, service):
        """
        Remove service from watch list.
        
        Args:
            service: Service to remove
        """
        if service in self.services:
            self.services.remove(service)
            logger.debug(f"Removed service from watcher: {service.service_name}")
    
    def add_config_path(self, path: str):
        """
        Add config file path to watch list.
        
        Args:
            path: Config file path to watch
        """
        resolved_path = Path(path).resolve()
        if resolved_path not in self.config_paths:
            self.config_paths.append(resolved_path)
            logger.debug(f"Added config path to watcher: {resolved_path.name}")
            
            # If already running, need to restart to watch new path
            if self._observer or self._polling_thread:
                logger.info("Restarting watcher to include new config path")
                self.stop()
                self.start()
    
    def is_running(self) -> bool:
        """Check if watcher is currently running."""
        if self.use_watchdog:
            return self._observer is not None and self._observer.is_alive()
        else:
            return self._polling_thread is not None and self._polling_thread.is_alive()
    
    def get_stats(self) -> dict:
        """Get watcher statistics."""
        return {
            "is_running": self.is_running(),
            "mode": "watchdog" if self.use_watchdog else "polling",
            "watchdog_available": WATCHDOG_AVAILABLE,
            "config_paths": [str(p) for p in self.config_paths],
            "service_count": len(self.services),
            "polling_interval": self.polling_interval if not self.use_watchdog else None,
            "debounce_seconds": self.debounce_seconds
        }
    
    def __repr__(self) -> str:
        return (
            f"<ConfigWatcher "
            f"mode={'watchdog' if self.use_watchdog else 'polling'} "
            f"files={len(self.config_paths)} "
            f"services={len(self.services)} "
            f"running={self.is_running()}>"
        )


# Convenience function for quick setup
def create_config_watcher(
    services: List,
    config_paths: List[str] = None,
    auto_start: bool = True
) -> ConfigWatcher:
    """
    Create and optionally start a config watcher.
    
    Args:
        services: List of services to watch
        config_paths: Config files to watch (default: ["configs/default.yaml"])
        auto_start: Automatically start watcher (default: True)
    
    Returns:
        ConfigWatcher instance
        
    Example:
        watcher = create_config_watcher(
            services=[aggregator, dom_service],
            config_paths=["configs/default.yaml"],
            auto_start=True
        )
    """
    watcher = ConfigWatcher(
        services=services,
        config_paths=config_paths
    )
    
    if auto_start:
        watcher.start()
    
    return watcher
