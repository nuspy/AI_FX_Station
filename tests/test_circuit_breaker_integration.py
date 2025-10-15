"""
Integration tests for circuit breaker behavior in market data services.

These tests verify:
1. Circuit breaker opens after threshold failures
2. Circuit breaker auto-recovers after timeout
3. Symbol caching reduces config reads
4. Notifications are triggered correctly
"""
from __future__ import annotations

import time
import threading
from unittest.mock import Mock, patch, MagicMock
import pytest
from sqlalchemy import create_engine

from forex_diffusion.services.base_service import (
    ThreadedBackgroundService,
    CircuitBreaker,
    CircuitBreakerState
)


class MockService(ThreadedBackgroundService):
    """Mock service for testing."""
    
    def __init__(self, *args, fail_count=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_count = fail_count
        self.iterations_completed = 0
    
    @property
    def service_name(self) -> str:
        return "MockService"
    
    def _process_iteration(self):
        """Simulates processing that may fail."""
        if self.fail_count > 0:
            self.fail_count -= 1
            raise RuntimeError("Simulated failure")
        self.iterations_completed += 1


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create in-memory SQLite engine for testing."""
        return create_engine("sqlite:///:memory:", future=True)
    
    def test_circuit_breaker_opens_after_failures(self, engine):
        """
        Test that circuit breaker opens after threshold failures.
        
        Expected behavior:
        1. Service starts normally (CLOSED)
        2. After 5 consecutive failures, circuit opens (OPEN)
        3. Service stops processing while OPEN
        """
        # Create service with 10 failures configured
        service = MockService(
            engine=engine,
            symbols=["EUR/USD"],
            interval_seconds=0.1,  # Fast iteration for testing
            enable_circuit_breaker=True,
            fail_count=10
        )
        
        # Verify initial state
        assert service._circuit_breaker.state == CircuitBreakerState.CLOSED
        
        # Start service and let it fail
        service.start()
        time.sleep(1.0)  # Allow time for 5+ failures
        
        # Circuit breaker should now be OPEN
        assert service._circuit_breaker.state == CircuitBreakerState.OPEN
        assert service._circuit_breaker._failure_count >= 5
        
        # Stop service
        service.stop()
    
    def test_circuit_breaker_auto_recovery(self, engine):
        """
        Test that circuit breaker auto-recovers after timeout.
        
        Expected behavior:
        1. Circuit opens after failures
        2. After timeout, enters HALF_OPEN
        3. Successful operations close circuit
        """
        # Create service with short timeout for testing
        service = MockService(
            engine=engine,
            symbols=["EUR/USD"],
            interval_seconds=0.1,
            enable_circuit_breaker=True,
            fail_count=5  # Fail first 5, then succeed
        )
        
        # Override circuit breaker with short timeout
        service._circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=2.0,  # Short timeout for testing
            half_open_max_calls=2,
            service_name="MockService"
        )
        
        # Start service and let it fail
        service.start()
        time.sleep(0.5)  # Allow failures
        
        # Should be OPEN
        assert service._circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Wait for timeout
        time.sleep(2.5)  # Timeout + processing time
        
        # Should enter HALF_OPEN
        assert service._circuit_breaker.state in [CircuitBreakerState.HALF_OPEN, CircuitBreakerState.CLOSED]
        
        # Wait for successful operations
        time.sleep(1.0)
        
        # Should be CLOSED (recovered)
        assert service._circuit_breaker.state == CircuitBreakerState.CLOSED
        assert service.iterations_completed > 0
        
        service.stop()
    
    def test_symbol_caching_reduces_config_reads(self, engine):
        """
        Test that symbol caching reduces config file reads.
        
        Expected behavior:
        1. First call loads from config
        2. Subsequent calls use cache
        3. Cache expires after TTL
        """
        service = MockService(
            engine=engine,
            symbols=None,  # Will load from config
            interval_seconds=1.0,
            symbol_cache_ttl=2.0  # 2 second cache for testing
        )
        
        with patch('forex_diffusion.utils.symbol_utils.get_symbols_from_config') as mock_get:
            mock_get.return_value = ["EUR/USD", "GBP/USD"]
            
            # First call should load from config
            symbols1 = service.get_symbols()
            assert mock_get.call_count == 1
            assert symbols1 == ["EUR/USD", "GBP/USD"]
            
            # Second call should use cache (no additional config read)
            symbols2 = service.get_symbols()
            assert mock_get.call_count == 1  # Still 1, no new call
            assert symbols2 == symbols1
            
            # Wait for cache to expire
            time.sleep(2.5)
            
            # Third call should reload from config
            mock_get.return_value = ["EUR/USD", "GBP/USD", "USD/JPY"]  # Config changed
            symbols3 = service.get_symbols()
            assert mock_get.call_count == 2  # New call after expiry
            assert symbols3 == ["EUR/USD", "GBP/USD", "USD/JPY"]
            
            # Bypass cache explicitly
            mock_get.return_value = ["EUR/USD"]
            symbols4 = service.get_symbols(bypass_cache=True)
            assert mock_get.call_count == 3  # Forced reload
            assert symbols4 == ["EUR/USD"]
    
    def test_manual_cache_invalidation(self, engine):
        """Test that manual cache invalidation forces reload."""
        service = MockService(engine=engine, symbols=None)
        
        with patch('forex_diffusion.utils.symbol_utils.get_symbols_from_config') as mock_get:
            mock_get.return_value = ["EUR/USD"]
            
            # Load symbols (cache)
            symbols1 = service.get_symbols()
            assert mock_get.call_count == 1
            
            # Use cache
            service.get_symbols()
            assert mock_get.call_count == 1
            
            # Invalidate cache
            service.invalidate_symbol_cache()
            
            # Should reload from config
            mock_get.return_value = ["EUR/USD", "GBP/USD"]
            symbols2 = service.get_symbols()
            assert mock_get.call_count == 2
            assert len(symbols2) == 2
    
    def test_circuit_breaker_notifications(self, engine):
        """
        Test that circuit breaker notifications are triggered.
        
        Expected behavior:
        1. on_open called when circuit opens
        2. on_half_open called when entering half-open
        3. on_close called when circuit closes
        """
        on_open_called = threading.Event()
        on_half_open_called = threading.Event()
        on_close_called = threading.Event()
        
        def on_open(cb):
            on_open_called.set()
        
        def on_half_open(cb):
            on_half_open_called.set()
        
        def on_close(cb):
            on_close_called.set()
        
        # Create service with callbacks
        service = MockService(
            engine=engine,
            symbols=["EUR/USD"],
            interval_seconds=0.1,
            enable_circuit_breaker=True,
            fail_count=5
        )
        
        service._circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=1.0,
            service_name="MockService",
            on_open=on_open,
            on_half_open=on_half_open,
            on_close=on_close
        )
        
        # Start and let it fail
        service.start()
        time.sleep(0.5)
        
        # Should have called on_open
        assert on_open_called.is_set(), "on_open callback was not called"
        
        # Wait for half-open
        time.sleep(1.5)
        
        # Should have called on_half_open
        assert on_half_open_called.is_set(), "on_half_open callback was not called"
        
        # Wait for recovery (no more failures)
        time.sleep(1.0)
        
        # Should have called on_close
        assert on_close_called.is_set(), "on_close callback was not called"
        
        service.stop()
    
    def test_metrics_collection(self, engine):
        """
        Test that service metrics are collected correctly.
        
        Expected metrics:
        - iteration_count
        - error_count
        - circuit_breaker_state
        - symbol_cache_age
        """
        service = MockService(
            engine=engine,
            symbols=["EUR/USD"],
            interval_seconds=0.1,
            enable_circuit_breaker=True,
            fail_count=2  # Fail first 2, then succeed
        )
        
        # Start service
        service.start()
        time.sleep(0.5)  # Let it run a bit
        
        # Get metrics
        metrics = service.get_metrics()
        
        # Verify metrics structure
        assert "service_name" in metrics
        assert metrics["service_name"] == "MockService"
        
        assert "is_running" in metrics
        assert metrics["is_running"] is True
        
        assert "iteration_count" in metrics
        # Note: May not have completed iterations if still failing
        
        assert "error_count" in metrics
        assert metrics["error_count"] >= 0
        
        assert "circuit_breaker_state" in metrics
        assert metrics["circuit_breaker_state"] in ["closed", "open", "half_open"]
        
        assert "circuit_breaker_stats" in metrics
        cb_stats = metrics["circuit_breaker_stats"]
        assert "state" in cb_stats
        assert "failure_count" in cb_stats
        assert "success_count" in cb_stats
        
        assert "symbol_cache_enabled" in metrics
        assert metrics["symbol_cache_enabled"] is False  # Symbols provided explicitly
        
        service.stop()


class TestCircuitBreakerStandalone:
    """Unit tests for circuit breaker standalone functionality."""
    
    def test_circuit_breaker_get_stats(self):
        """Test circuit breaker statistics."""
        cb = CircuitBreaker(
            failure_threshold=5,
            timeout=60.0,
            service_name="TestService"
        )
        
        stats = cb.get_stats()
        
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0
        assert stats["success_count"] == 0
        assert stats["failure_threshold"] == 5
        assert stats["timeout"] == 60.0
        assert stats["last_failure_time"] is None
        assert stats["time_until_half_open"] == 0
    
    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state machine."""
        cb = CircuitBreaker(
            failure_threshold=3,
            timeout=1.0,
            service_name="TestService"
        )
        
        # Initial state: CLOSED
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.can_execute() is True
        
        # Record 3 failures
        for _ in range(3):
            cb.record_failure()
        
        # Should be OPEN now
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.can_execute() is False
        
        # Wait for timeout
        time.sleep(1.5)
        
        # Should allow half-open test
        assert cb.can_execute() is True
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Record success
        cb.record_success()
        
        # Should close
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_reset(self):
        """Test manual circuit breaker reset."""
        cb = CircuitBreaker(failure_threshold=3, service_name="TestService")
        
        # Open circuit
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # Manual reset
        cb.reset()
        
        # Should be closed
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb._failure_count == 0
        assert cb._success_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
