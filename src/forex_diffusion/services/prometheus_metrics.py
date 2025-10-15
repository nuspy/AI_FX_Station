"""
Prometheus metrics export for market data services.

Exposes service metrics in Prometheus format for scraping and visualization.

Features:
- Service-level metrics (iterations, errors, uptime)
- Circuit breaker state tracking
- Symbol cache hit/miss rates
- Custom metric registration
- Multi-process safe (using prometheus_client multiprocess mode)

Usage:
    from forex_diffusion.services.prometheus_metrics import (
        setup_prometheus_metrics,
        start_prometheus_server
    )
    
    # Setup metrics for services
    setup_prometheus_metrics([aggregator, dom_service, sentiment_service])
    
    # Start HTTP server for Prometheus to scrape
    start_prometheus_server(port=9090)
    
    # Metrics available at http://localhost:9090/metrics
"""
from __future__ import annotations

import time
from typing import List, Optional, Dict
from threading import Thread

from loguru import logger

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Info,
        CollectorRegistry, generate_latest,
        start_http_server, REGISTRY
    )
    from prometheus_client.exposition import MetricsHandler
    from http.server import HTTPServer
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, metrics export unavailable")


class ServiceMetricsCollector:
    """
    Collects and exposes service metrics in Prometheus format.
    
    Metrics exposed:
    - service_iterations_total: Total iterations completed
    - service_errors_total: Total errors encountered
    - service_uptime_seconds: Service uptime
    - circuit_breaker_state: Circuit breaker state (0=closed, 1=half_open, 2=open)
    - circuit_breaker_failures_total: Total circuit breaker failures
    - symbol_cache_hits_total: Symbol cache hits
    - symbol_cache_misses_total: Symbol cache misses
    - symbol_cache_age_seconds: Symbol cache age
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.
        
        Args:
            registry: Prometheus registry (default: use global REGISTRY)
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client not installed")
        
        self.registry = registry or REGISTRY
        
        # Service metrics
        self.iterations_total = Counter(
            'service_iterations_total',
            'Total number of service iterations',
            ['service'],
            registry=self.registry
        )
        
        self.errors_total = Counter(
            'service_errors_total',
            'Total number of service errors',
            ['service', 'error_type'],
            registry=self.registry
        )
        
        self.is_running = Gauge(
            'service_is_running',
            'Whether service is currently running (1=running, 0=stopped)',
            ['service'],
            registry=self.registry
        )
        
        self.iteration_duration_seconds = Histogram(
            'service_iteration_duration_seconds',
            'Service iteration duration in seconds',
            ['service'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half_open, 2=open)',
            ['service'],
            registry=self.registry
        )
        
        self.circuit_breaker_failures_total = Counter(
            'circuit_breaker_failures_total',
            'Total circuit breaker failures',
            ['service'],
            registry=self.registry
        )
        
        self.circuit_breaker_opens_total = Counter(
            'circuit_breaker_opens_total',
            'Total times circuit breaker opened',
            ['service'],
            registry=self.registry
        )
        
        self.circuit_breaker_time_until_half_open = Gauge(
            'circuit_breaker_time_until_half_open_seconds',
            'Seconds until circuit breaker enters half-open state',
            ['service'],
            registry=self.registry
        )
        
        # Symbol cache metrics
        self.symbol_cache_hits_total = Counter(
            'symbol_cache_hits_total',
            'Total symbol cache hits',
            ['service'],
            registry=self.registry
        )
        
        self.symbol_cache_misses_total = Counter(
            'symbol_cache_misses_total',
            'Total symbol cache misses',
            ['service'],
            registry=self.registry
        )
        
        self.symbol_cache_age_seconds = Gauge(
            'symbol_cache_age_seconds',
            'Age of symbol cache in seconds',
            ['service'],
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'service_info',
            'Service information',
            ['service'],
            registry=self.registry
        )
        
        # Track last metrics state for delta calculation
        self._last_metrics: Dict[str, dict] = {}
    
    def update_metrics(self, services: List):
        """
        Update Prometheus metrics from service metrics.
        
        Args:
            services: List of services with get_metrics() method
        """
        for service in services:
            try:
                metrics = service.get_metrics()
                service_name = metrics.get("service_name", "unknown")
                
                # Update running status
                self.is_running.labels(service=service_name).set(
                    1 if metrics.get("is_running", False) else 0
                )
                
                # Update iteration and error counts (use deltas to avoid duplicates)
                current_iterations = metrics.get("iteration_count", 0)
                current_errors = metrics.get("error_count", 0)
                
                last_state = self._last_metrics.get(service_name, {})
                last_iterations = last_state.get("iteration_count", 0)
                last_errors = last_state.get("error_count", 0)
                
                # Increment by delta
                if current_iterations > last_iterations:
                    self.iterations_total.labels(service=service_name).inc(
                        current_iterations - last_iterations
                    )
                
                if current_errors > last_errors:
                    # For errors, use generic error_type since we don't have details
                    self.errors_total.labels(service=service_name, error_type="general").inc(
                        current_errors - last_errors
                    )
                
                # Update circuit breaker metrics
                cb_state = metrics.get("circuit_breaker_state")
                if cb_state:
                    state_map = {"closed": 0, "half_open": 1, "open": 2}
                    self.circuit_breaker_state.labels(service=service_name).set(
                        state_map.get(cb_state, -1)
                    )
                
                cb_stats = metrics.get("circuit_breaker_stats", {})
                if cb_stats:
                    # Circuit breaker time until half-open
                    time_until_half_open = cb_stats.get("time_until_half_open", 0)
                    self.circuit_breaker_time_until_half_open.labels(service=service_name).set(
                        time_until_half_open
                    )
                    
                    # Track circuit breaker opens (delta)
                    if cb_state == "open" and last_state.get("circuit_breaker_state") != "open":
                        self.circuit_breaker_opens_total.labels(service=service_name).inc()
                
                # Update symbol cache metrics
                cache_age = metrics.get("symbol_cache_age_seconds")
                if cache_age is not None:
                    self.symbol_cache_age_seconds.labels(service=service_name).set(cache_age)
                    
                    # Track cache hits/misses (when cache refreshes, it's a miss)
                    last_cache_age = last_state.get("symbol_cache_age_seconds", 0)
                    if cache_age < last_cache_age:
                        # Cache was refreshed (miss)
                        self.symbol_cache_misses_total.labels(service=service_name).inc()
                    elif cache_age > 0:
                        # Cache is being used (hit)
                        # Only increment on first iteration to avoid over-counting
                        if last_cache_age == 0:
                            self.symbol_cache_hits_total.labels(service=service_name).inc()
                
                # Update service info
                self.service_info.labels(service=service_name).info({
                    "interval_seconds": str(metrics.get("interval_seconds", "unknown")),
                    "symbols": ",".join(metrics.get("symbols", [])),
                    "cache_enabled": str(metrics.get("symbol_cache_enabled", False))
                })
                
                # Store current metrics for next delta calculation
                self._last_metrics[service_name] = metrics
                
            except Exception as e:
                logger.error(f"Failed to update metrics for service: {e}")


class PrometheusMetricsServer:
    """
    HTTP server for Prometheus metrics endpoint.
    
    Starts an HTTP server that exposes /metrics endpoint for Prometheus scraping.
    Automatically updates metrics from services at regular intervals.
    """
    
    def __init__(
        self,
        services: List,
        port: int = 9090,
        addr: str = "0.0.0.0",
        update_interval: float = 5.0,
        registry: Optional[CollectorRegistry] = None
    ):
        """
        Initialize Prometheus metrics server.
        
        Args:
            services: List of services to collect metrics from
            port: HTTP server port (default: 9090)
            addr: Bind address (default: 0.0.0.0 for all interfaces)
            update_interval: Metrics update interval in seconds (default: 5s)
            registry: Custom registry (default: use global)
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client not installed")
        
        self.services = services
        self.port = port
        self.addr = addr
        self.update_interval = update_interval
        self.registry = registry or REGISTRY
        
        self.collector = ServiceMetricsCollector(registry=self.registry)
        self._server_thread: Optional[Thread] = None
        self._update_thread: Optional[Thread] = None
        self._stop_flag = False
    
    def start(self):
        """Start HTTP server and metrics update loop."""
        # Start metrics update loop
        self._stop_flag = False
        self._update_thread = Thread(
            target=self._update_loop,
            daemon=True,
            name="PrometheusMetricsUpdater"
        )
        self._update_thread.start()
        
        # Start HTTP server
        try:
            start_http_server(self.port, addr=self.addr, registry=self.registry)
            logger.info(f"Prometheus metrics server started on {self.addr}:{self.port}")
            logger.info(f"Metrics available at http://{self.addr}:{self.port}/metrics")
        except OSError as e:
            logger.error(f"Failed to start Prometheus server on port {self.port}: {e}")
            raise
    
    def stop(self):
        """Stop metrics update loop."""
        self._stop_flag = True
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        logger.info("Prometheus metrics server stopped")
    
    def _update_loop(self):
        """Background loop to update metrics periodically."""
        logger.info(f"Prometheus metrics update loop started (interval={self.update_interval}s)")
        
        while not self._stop_flag:
            try:
                self.collector.update_metrics(self.services)
            except Exception as e:
                logger.error(f"Error updating Prometheus metrics: {e}")
            
            # Sleep with early exit on stop
            for _ in range(int(self.update_interval * 10)):
                if self._stop_flag:
                    break
                time.sleep(0.1)
    
    def add_service(self, service):
        """Add service to metrics collection."""
        if service not in self.services:
            self.services.append(service)
            logger.debug(f"Added service to Prometheus metrics: {service.service_name}")
    
    def remove_service(self, service):
        """Remove service from metrics collection."""
        if service in self.services:
            self.services.remove(service)
            logger.debug(f"Removed service from Prometheus metrics: {service.service_name}")


# Convenience functions
def setup_prometheus_metrics(
    services: List,
    port: int = 9090,
    addr: str = "0.0.0.0",
    update_interval: float = 5.0,
    auto_start: bool = True
) -> Optional[PrometheusMetricsServer]:
    """
    Setup and start Prometheus metrics server.
    
    Args:
        services: List of services to monitor
        port: HTTP server port (default: 9090)
        addr: Bind address (default: 0.0.0.0)
        update_interval: Metrics update interval (default: 5s)
        auto_start: Automatically start server (default: True)
    
    Returns:
        PrometheusMetricsServer instance or None if prometheus_client not available
        
    Example:
        server = setup_prometheus_metrics(
            services=[aggregator, dom_service],
            port=9090,
            auto_start=True
        )
        
        # Metrics now available at http://localhost:9090/metrics
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Cannot setup Prometheus metrics: prometheus_client not installed")
        return None
    
    try:
        server = PrometheusMetricsServer(
            services=services,
            port=port,
            addr=addr,
            update_interval=update_interval
        )
        
        if auto_start:
            server.start()
        
        return server
        
    except Exception as e:
        logger.error(f"Failed to setup Prometheus metrics: {e}")
        return None


def start_prometheus_server(
    port: int = 9090,
    addr: str = "0.0.0.0",
    registry: Optional[CollectorRegistry] = None
):
    """
    Start basic Prometheus HTTP server (without automatic service metrics updates).
    
    Use this if you want to manually update metrics or use prometheus_client directly.
    
    Args:
        port: HTTP server port
        addr: Bind address
        registry: Custom registry (default: global REGISTRY)
        
    Example:
        start_prometheus_server(port=9090)
        
        # Manually update metrics
        my_counter.inc()
        my_gauge.set(42)
    """
    if not PROMETHEUS_AVAILABLE:
        logger.error("Cannot start Prometheus server: prometheus_client not installed")
        return
    
    try:
        start_http_server(port, addr=addr, registry=registry or REGISTRY)
        logger.info(f"Prometheus server started on {addr}:{port}")
    except OSError as e:
        logger.error(f"Failed to start Prometheus server: {e}")
        raise
