"""
Example: Production setup with Phase 4 features.

Demonstrates complete integration of:
- Config file watcher (auto cache invalidation)
- Prometheus metrics export
- Health check REST API
- All 3 aggregator services

This is a reference implementation for production deployment.
"""
from __future__ import annotations

import os
import time
from sqlalchemy import create_engine

from forex_diffusion.services.aggregator import AggregatorService
from forex_diffusion.services.dom_aggregator import DOMAggregatorService
from forex_diffusion.services.sentiment_aggregator import SentimentAggregatorService

# Phase 4 imports
from forex_diffusion.services.config_watcher import create_config_watcher
from forex_diffusion.services.prometheus_metrics import setup_prometheus_metrics
from forex_diffusion.services.health_endpoint import run_health_server

# Optional: Notification helpers
from forex_diffusion.services.notification_helpers import send_slack_alert

def setup_production_services():
    """
    Setup all services with Phase 4 features.
    
    Returns:
        Tuple of (services, config_watcher, prometheus_server)
    """
    # Database connection
    db_url = os.getenv("DATABASE_URL", "sqlite:///./forexgpt.db")
    engine = create_engine(db_url, future=True)
    
    # Initialize services with custom parameters
    aggregator = AggregatorService(
        engine=engine,
        symbols=None,  # Load from config
        interval_seconds=60.0,  # 1 minute
        symbol_cache_ttl=300.0  # 5 minutes
    )
    
    dom_service = DOMAggregatorService(
        engine=engine,
        symbols=None,
        interval_seconds=5.0,  # 5 seconds
        symbol_cache_ttl=300.0
    )
    
    sentiment_service = SentimentAggregatorService(
        engine=engine,
        symbols=None,
        interval_seconds=30.0,  # 30 seconds
        symbol_cache_ttl=300.0
    )
    
    services = [aggregator, dom_service, sentiment_service]
    
    # Override circuit breaker callbacks for Slack notifications (optional)
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook:
        for service in services:
            if hasattr(service, '_circuit_breaker') and service._circuit_breaker:
                # Original on_open callback
                original_on_open = service._on_circuit_open
                
                def create_slack_callback(svc, original):
                    def on_open(cb):
                        # Call original (logs)
                        original(cb)
                        # Send Slack alert
                        send_slack_alert(
                            f"🚨 CRITICAL: {svc.service_name} circuit breaker OPENED!\n"
                            f"Failures: {cb._failure_count}\n"
                            f"Timeout: {cb.timeout}s",
                            severity="critical"
                        )
                    return on_open
                
                service._circuit_breaker.on_open = create_slack_callback(service, original_on_open)
        
        print("✅ Slack notifications configured")
    
    # Start services
    for service in services:
        service.start()
        print(f"✅ Started: {service.service_name}")
    
    return services, engine


def setup_config_watcher(services):
    """Setup config file watcher with auto cache invalidation."""
    config_paths = [
        "configs/default.yaml",
        "configs/user.yaml",
        ".env"
    ]
    
    watcher = create_config_watcher(
        services=services,
        config_paths=config_paths,
        auto_start=True
    )
    
    stats = watcher.get_stats()
    print(f"✅ Config watcher started (mode={stats['mode']})")
    print(f"   Watching: {', '.join([p.split('/')[-1] for p in stats['config_paths']])}")
    
    return watcher


def setup_prometheus(services):
    """Setup Prometheus metrics export."""
    port = int(os.getenv("PROMETHEUS_PORT", "9090"))
    
    prometheus_server = setup_prometheus_metrics(
        services=services,
        port=port,
        update_interval=5.0,
        auto_start=True
    )
    
    if prometheus_server:
        print(f"✅ Prometheus metrics started on port {port}")
        print(f"   Metrics: http://localhost:{port}/metrics")
    else:
        print("⚠️  Prometheus metrics unavailable (prometheus_client not installed)")
    
    return prometheus_server


def run_health_api(services):
    """Run health check REST API (blocking)."""
    port = int(os.getenv("HEALTH_API_PORT", "8080"))
    
    print(f"✅ Starting health API on port {port}")
    print(f"   Health: http://localhost:{port}/health")
    print(f"   Metrics: http://localhost:{port}/metrics")
    print(f"   Circuit Breaker: http://localhost:{port}/circuit-breaker")
    
    # This is blocking - runs until interrupted
    run_health_server(
        services=services,
        port=port,
        host="0.0.0.0"
    )


def main():
    """
    Main production setup.
    
    Environment Variables:
        DATABASE_URL: Database connection string
        SLACK_WEBHOOK_URL: Slack webhook for alerts (optional)
        PROMETHEUS_PORT: Prometheus metrics port (default: 9090)
        HEALTH_API_PORT: Health API port (default: 8080)
    """
    print("=" * 60)
    print("ForexGPT Production Setup - Phase 4")
    print("=" * 60)
    print()
    
    # Setup services
    print("1. Initializing services...")
    services, engine = setup_production_services()
    print()
    
    # Setup config watcher
    print("2. Setting up config file watcher...")
    config_watcher = setup_config_watcher(services)
    print()
    
    # Setup Prometheus
    print("3. Setting up Prometheus metrics...")
    prometheus_server = setup_prometheus(services)
    print()
    
    # Run health API (blocking)
    print("4. Starting health API...")
    print()
    print("=" * 60)
    print("System Ready!")
    print("=" * 60)
    print()
    print("Available endpoints:")
    print("  - Health API: http://localhost:8080/health")
    print("  - Metrics API: http://localhost:8080/metrics")
    print("  - Prometheus: http://localhost:9090/metrics")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    try:
        run_health_api(services)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        
        # Stop services
        for service in services:
            service.stop()
            print(f"✅ Stopped: {service.service_name}")
        
        # Stop config watcher
        if config_watcher:
            config_watcher.stop()
            print("✅ Stopped: Config watcher")
        
        # Stop Prometheus (metrics server runs in background)
        if prometheus_server:
            prometheus_server.stop()
            print("✅ Stopped: Prometheus metrics")
        
        print("\nShutdown complete")


if __name__ == "__main__":
    main()
