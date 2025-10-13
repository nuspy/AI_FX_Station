"""
Base classes for background services with threading support.

Provides abstract base classes for common service patterns:
- ThreadedBackgroundService: Background processing with start/stop lifecycle
- CircuitBreaker: Error recovery and failure handling
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Callable, Dict, Any
from collections import deque
from pathlib import Path

from loguru import logger
from sqlalchemy.engine import Engine

from .db_service import DBService
from ..utils.symbol_utils import get_symbols_from_config


class CircuitBreakerState(Enum):
    """Circuit breaker states for error recovery."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker for error recovery with async notifications.
    
    Prevents repeated failures from cascading by temporarily blocking
    operations after a threshold of consecutive failures.
    
    States:
    - CLOSED: Normal operation, all requests allowed
    - OPEN: Too many failures, block all requests
    - HALF_OPEN: Testing recovery, allow limited requests
    
    Example:
        def on_open(cb):
            logger.critical(f"Circuit breaker opened: {cb.service_name}")
        
        cb = CircuitBreaker(
            failure_threshold=5, 
            timeout=60,
            on_open=on_open,
            on_close=lambda cb: logger.info(f"Circuit recovered: {cb.service_name}")
        )
        
        if cb.can_execute():
            try:
                result = risky_operation()
                cb.record_success()
            except Exception as e:
                cb.record_failure()
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        half_open_max_calls: int = 3,
        on_open: Optional[Callable[['CircuitBreaker'], None]] = None,
        on_close: Optional[Callable[['CircuitBreaker'], None]] = None,
        on_half_open: Optional[Callable[['CircuitBreaker'], None]] = None,
        service_name: str = "Unknown"
    ):
        """
        Initialize circuit breaker with async notification callbacks.
        
        Args:
            failure_threshold: Number of consecutive failures before opening
            timeout: Seconds to wait before trying half-open state
            half_open_max_calls: Max calls allowed in half-open state
            on_open: Callback when circuit opens (for alerts)
            on_close: Callback when circuit closes (recovery notification)
            on_half_open: Callback when entering half-open state
            service_name: Name of service for logging/notifications
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls
        self.service_name = service_name
        
        # Notification callbacks
        self.on_open = on_open
        self.on_close = on_close
        self.on_half_open = on_half_open
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True
            
            if self._state == CircuitBreakerState.OPEN:
                # Check if timeout elapsed
                if self._last_failure_time is None:
                    return False
                
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.timeout:
                    # Try half-open
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"{self.service_name} circuit breaker entering HALF_OPEN state after {elapsed:.1f}s")
                    
                    # Notify callback
                    if self.on_half_open:
                        try:
                            self.on_half_open(self)
                        except Exception as e:
                            logger.error(f"Circuit breaker on_half_open callback failed: {e}")
                    
                    return True
                return False
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                # Allow limited calls
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            
            return False
    
    def record_success(self):
        """Record successful operation."""
        with self._lock:
            self._success_count += 1
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                # Successful test, close circuit
                logger.info(f"{self.service_name} circuit breaker recovered, closing (successes={self._success_count})")
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                
                # Notify callback
                if self.on_close:
                    try:
                        self.on_close(self)
                    except Exception as e:
                        logger.error(f"Circuit breaker on_close callback failed: {e}")
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def record_failure(self):
        """Record failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                # Failed test, reopen circuit
                logger.warning(f"{self.service_name} circuit breaker test failed, reopening")
                self._state = CircuitBreakerState.OPEN
                self._half_open_calls = 0
            elif self._state == CircuitBreakerState.CLOSED:
                # Check threshold
                if self._failure_count >= self.failure_threshold:
                    logger.error(
                        f"{self.service_name} circuit breaker threshold reached "
                        f"({self._failure_count} failures), opening for {self.timeout}s"
                    )
                    self._state = CircuitBreakerState.OPEN
                    
                    # Notify callback
                    if self.on_open:
                        try:
                            self.on_open(self)
                        except Exception as e:
                            logger.error(f"Circuit breaker on_open callback failed: {e}")
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking)."""
        return self._state == CircuitBreakerState.OPEN
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            logger.info(f"{self.service_name} circuit breaker manually reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "timeout": self.timeout,
                "last_failure_time": self._last_failure_time,
                "time_until_half_open": (
                    max(0, self.timeout - (time.time() - self._last_failure_time))
                    if self._last_failure_time and self._state == CircuitBreakerState.OPEN
                    else 0
                )
            }


class ThreadedBackgroundService(ABC):
    """
    Abstract base class for background services with threading.
    
    Provides common infrastructure for services that run in background threads:
    - Lifecycle management (start/stop)
    - Thread safety primitives
    - Error handling with circuit breaker
    - Symbol configuration
    - Database access
    
    Subclasses must implement:
    - _process_iteration(): Main processing logic
    - service_name: Name for logging
    
    Example:
        class MyService(ThreadedBackgroundService):
            @property
            def service_name(self) -> str:
                return "MyService"
            
            def _process_iteration(self):
                symbols = self.get_symbols()
                for symbol in symbols:
                    self._process_symbol(symbol)
    """
    
    def __init__(
        self,
        engine: Engine,
        symbols: List[str] | None = None,
        interval_seconds: float = 5.0,
        enable_circuit_breaker: bool = True,
        symbol_cache_ttl: float = 300.0
    ):
        """
        Initialize background service.
        
        Args:
            engine: SQLAlchemy engine for database access
            symbols: List of symbols to process (None = load from config)
            interval_seconds: Sleep interval between iterations
            enable_circuit_breaker: Enable circuit breaker for error recovery
            symbol_cache_ttl: Symbol cache time-to-live in seconds (default: 5 minutes)
        """
        self.engine = engine
        self.db = DBService(engine=self.engine)
        self._configured_symbols = symbols
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Symbol caching with TTL
        self._cached_symbols: Optional[List[str]] = None
        self._symbol_cache_time: Optional[float] = None
        self._symbol_cache_ttl = symbol_cache_ttl
        self._symbol_cache_lock = threading.Lock()
        
        # Circuit breaker for error recovery
        self._circuit_breaker: Optional[CircuitBreaker] = None
        if enable_circuit_breaker:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                timeout=60.0,
                half_open_max_calls=3,
                service_name=self.service_name,  # Will be set by subclass
                on_open=self._on_circuit_open,
                on_close=self._on_circuit_close,
                on_half_open=self._on_circuit_half_open
            )
        
        # Metrics
        self._iteration_count = 0
        self._error_count = 0
        self._last_error_time: Optional[float] = None
        self._metrics_lock = threading.Lock()
        
        # Recent errors for debugging
        self._recent_errors: deque = deque(maxlen=10)
    
    @property
    @abstractmethod
    def service_name(self) -> str:
        """Service name for logging (must be implemented by subclass)."""
        pass
    
    @abstractmethod
    def _process_iteration(self):
        """
        Process one iteration (must be implemented by subclass).
        
        This method is called repeatedly in the background thread.
        Should process all symbols or perform one unit of work.
        
        Raises:
            Exception: Any exception will be caught and logged by the framework
        """
        pass
    
    def start(self):
        """Start background service thread."""
        if self._thread and self._thread.is_alive():
            logger.warning(f"{self.service_name} already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name=self.service_name)
        self._thread.start()
        logger.info(
            f"{self.service_name} started "
            f"(interval={self._interval}s, symbols={self._configured_symbols or '<config>'}, "
            f"circuit_breaker={self._circuit_breaker is not None})"
        )
    
    def stop(self, timeout: float = 5.0):
        """
        Stop background service thread.
        
        Args:
            timeout: Maximum time to wait for thread to stop (seconds)
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(f"{self.service_name} thread did not stop within {timeout}s timeout")
            else:
                logger.info(f"{self.service_name} stopped cleanly")
        self._thread = None
    
    def is_running(self) -> bool:
        """Check if service is currently running."""
        return self._thread is not None and self._thread.is_alive()
    
    def get_symbols(self, bypass_cache: bool = False) -> List[str]:
        """
        Get list of symbols to process with caching.
        
        Args:
            bypass_cache: If True, force reload from config (default: False)
        
        Returns:
            List of symbols to process
            
        Note:
            Symbols are cached for symbol_cache_ttl seconds (default: 5 minutes)
            to reduce config file reads. Cache is thread-safe.
        """
        # If configured explicitly, always use those (no caching needed)
        if self._configured_symbols:
            return self._configured_symbols
        
        now = time.time()
        
        # Check cache validity
        with self._symbol_cache_lock:
            if (not bypass_cache and
                self._cached_symbols is not None and
                self._symbol_cache_time is not None and
                now - self._symbol_cache_time < self._symbol_cache_ttl):
                return self._cached_symbols
            
            # Refresh cache
            logger.debug(f"{self.service_name} refreshing symbol cache")
            self._cached_symbols = get_symbols_from_config()
            self._symbol_cache_time = now
            
            return self._cached_symbols
    
    def invalidate_symbol_cache(self):
        """
        Manually invalidate symbol cache.
        
        Useful when config is known to have changed externally.
        Next call to get_symbols() will reload from config.
        """
        with self._symbol_cache_lock:
            self._cached_symbols = None
            self._symbol_cache_time = None
            logger.info(f"{self.service_name} symbol cache invalidated")
    
    def _run_loop(self):
        """Main service loop (runs in background thread)."""
        logger.info(f"{self.service_name} loop started")
        
        while not self._stop_event.is_set():
            try:
                # Check circuit breaker
                if self._circuit_breaker and not self._circuit_breaker.can_execute():
                    logger.debug(
                        f"{self.service_name} circuit breaker is {self._circuit_breaker.state.value}, "
                        "skipping iteration"
                    )
                    time.sleep(self._interval)
                    continue
                
                # Process one iteration
                self._process_iteration()
                
                # Record success
                if self._circuit_breaker:
                    self._circuit_breaker.record_success()
                
                # Update metrics
                with self._metrics_lock:
                    self._iteration_count += 1
            
            except Exception as e:
                # Record failure
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                
                # Update error metrics
                with self._metrics_lock:
                    self._error_count += 1
                    self._last_error_time = time.time()
                    self._recent_errors.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": str(e),
                        "type": type(e).__name__
                    })
                
                logger.exception(f"{self.service_name} iteration error: {e}")
            
            # Sleep until next iteration
            time.sleep(self._interval)
        
        logger.info(f"{self.service_name} loop ended")
    
    def get_metrics(self) -> dict:
        """
        Get service metrics.
        
        Returns:
            Dictionary with service metrics (iterations, errors, uptime, etc.)
        """
        with self._metrics_lock:
            metrics = {
                "service_name": self.service_name,
                "is_running": self.is_running(),
                "iteration_count": self._iteration_count,
                "error_count": self._error_count,
                "last_error_time": self._last_error_time,
                "circuit_breaker_state": self._circuit_breaker.state.value if self._circuit_breaker else None,
                "recent_errors": list(self._recent_errors),
                "interval_seconds": self._interval,
                "symbols": self.get_symbols(),
                "symbol_cache_enabled": self._configured_symbols is None,
                "symbol_cache_age_seconds": (
                    time.time() - self._symbol_cache_time
                    if self._symbol_cache_time is not None
                    else None
                )
            }
            
            # Add detailed circuit breaker stats if available
            if self._circuit_breaker:
                metrics["circuit_breaker_stats"] = self._circuit_breaker.get_stats()
            
            return metrics
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker (admin operation)."""
        if self._circuit_breaker:
            self._circuit_breaker.reset()
            logger.info(f"{self.service_name} circuit breaker manually reset")
    
    def _on_circuit_open(self, circuit_breaker: CircuitBreaker):
        """
        Called when circuit breaker opens (override for custom notifications).
        
        Default implementation logs critical message.
        Override in subclass to add Slack/email alerts.
        """
        logger.critical(
            f"🚨 {self.service_name} CIRCUIT BREAKER OPENED: "
            f"{circuit_breaker._failure_count} consecutive failures"
        )
    
    def _on_circuit_close(self, circuit_breaker: CircuitBreaker):
        """
        Called when circuit breaker closes (override for custom notifications).
        
        Default implementation logs info message.
        """
        logger.info(
            f"✅ {self.service_name} CIRCUIT BREAKER CLOSED: "
            f"Service recovered after {circuit_breaker._success_count} successful operations"
        )
    
    def _on_circuit_half_open(self, circuit_breaker: CircuitBreaker):
        """
        Called when circuit breaker enters half-open state.
        
        Default implementation logs warning message.
        """
        logger.warning(
            f"⚠️ {self.service_name} CIRCUIT BREAKER HALF-OPEN: "
            f"Testing service recovery (will try {circuit_breaker.half_open_max_calls} operations)"
        )
    
    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.service_name} "
            f"running={self.is_running()} "
            f"iterations={self._iteration_count} "
            f"errors={self._error_count}>"
        )
