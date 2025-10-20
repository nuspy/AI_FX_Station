"""
Health check REST API endpoint for market data services.

Provides HTTP endpoints for service health monitoring, metrics, and management.

Features:
- /health: Overall system health status
- /health/{service}: Individual service health
- /metrics: Service metrics in JSON format
- /circuit-breaker: Circuit breaker status and control
- /cache: Symbol cache status and control

Usage with FastAPI:
    from fastapi import FastAPI
    from forex_diffusion.services.health_endpoint import create_health_router
    
    app = FastAPI()
    health_router = create_health_router(
        services=[aggregator, dom_service, sentiment_service]
    )
    app.include_router(health_router, prefix="/api")
    
    # Available at:
    # GET /api/health
    # GET /api/health/AggregatorService
    # GET /api/metrics
    # GET /api/circuit-breaker
    # POST /api/circuit-breaker/reset
    # POST /api/cache/invalidate

Usage standalone:
    from forex_diffusion.services.health_endpoint import run_health_server
    
    run_health_server(
        services=[aggregator, dom_service],
        port=8080
    )
"""
from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime

from loguru import logger

try:
    from fastapi import FastAPI, HTTPException, status, APIRouter
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("fastapi not installed, health endpoint unavailable")


class HealthChecker:
    """
    Health checker for market data services.
    
    Evaluates service health based on:
    - Running status
    - Circuit breaker state
    - Error rate (errors / iterations)
    - Recent errors
    """
    
    def __init__(self, error_rate_threshold: float = 0.05):
        """
        Initialize health checker.
        
        Args:
            error_rate_threshold: Maximum acceptable error rate (default: 5%)
        """
        self.error_rate_threshold = error_rate_threshold
    
    def check_service(self, service) -> Dict[str, Any]:
        """
        Check health of a single service.
        
        Args:
            service: Service instance with get_metrics() method
            
        Returns:
            Health status dictionary
        """
        try:
            metrics = service.get_metrics()
            
            # Calculate health status
            is_running = metrics.get("is_running", False)
            cb_state = metrics.get("circuit_breaker_state", "closed")
            iteration_count = metrics.get("iteration_count", 0)
            error_count = metrics.get("error_count", 0)
            
            # Calculate error rate
            error_rate = (error_count / max(iteration_count, 1)) if iteration_count > 0 else 0
            
            # Determine health status
            if not is_running:
                status_value = "stopped"
                healthy = False
            elif cb_state == "open":
                status_value = "degraded"
                healthy = False
            elif error_rate > self.error_rate_threshold:
                status_value = "degraded"
                healthy = False
            else:
                status_value = "healthy"
                healthy = True
            
            return {
                "service": metrics.get("service_name", "unknown"),
                "status": status_value,
                "healthy": healthy,
                "running": is_running,
                "circuit_breaker": cb_state,
                "iterations": iteration_count,
                "errors": error_count,
                "error_rate": f"{error_rate * 100:.2f}%",
                "last_error_time": metrics.get("last_error_time"),
                "recent_errors": metrics.get("recent_errors", [])[-3:],  # Last 3 errors
                "cache_enabled": metrics.get("symbol_cache_enabled", False),
                "cache_age_seconds": metrics.get("symbol_cache_age_seconds")
            }
            
        except Exception as e:
            logger.error(f"Failed to check service health: {e}")
            return {
                "service": str(service),
                "status": "error",
                "healthy": False,
                "error": str(e)
            }
    
    def check_all_services(self, services: List) -> Dict[str, Any]:
        """
        Check health of all services.
        
        Args:
            services: List of service instances
            
        Returns:
            Overall health status dictionary
        """
        service_health = {}
        all_healthy = True
        
        for service in services:
            health = self.check_service(service)
            service_name = health.get("service", "unknown")
            service_health[service_name] = health
            
            if not health.get("healthy", False):
                all_healthy = False
        
        return {
            "overall_status": "healthy" if all_healthy else "degraded",
            "healthy": all_healthy,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "services": service_health
        }


def create_health_router(
    services: List,
    error_rate_threshold: float = 0.05
) -> 'APIRouter':
    """
    Create FastAPI router with health endpoints.
    
    Args:
        services: List of services to monitor
        error_rate_threshold: Maximum acceptable error rate
        
    Returns:
        FastAPI APIRouter instance
        
    Endpoints:
        GET /health - Overall health status
        GET /health/{service_name} - Individual service health
        GET /metrics - All service metrics
        GET /circuit-breaker - Circuit breaker status for all services
        POST /circuit-breaker/reset - Reset circuit breaker (requires service_name param)
        POST /cache/invalidate - Invalidate symbol cache (requires service_name param)
        
    Example:
        from fastapi import FastAPI
        
        app = FastAPI()
        router = create_health_router(services=[aggregator, dom_service])
        app.include_router(router, prefix="/api")
        
        # uvicorn app:app --host 0.0.0.0 --port 8080
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("fastapi not installed")
    
    router = APIRouter()
    checker = HealthChecker(error_rate_threshold=error_rate_threshold)
    
    # Map service names to instances for quick lookup
    service_map = {service.service_name: service for service in services}
    
    @router.get("/health")
    async def get_overall_health():
        """Get overall system health status."""
        return checker.check_all_services(services)
    
    @router.get("/health/{service_name}")
    async def get_service_health(service_name: str):
        """Get health status for specific service."""
        service = service_map.get(service_name)
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Service not found: {service_name}"
            )
        
        return checker.check_service(service)
    
    @router.get("/metrics")
    async def get_all_metrics():
        """Get detailed metrics for all services."""
        metrics = {}
        for service in services:
            try:
                metrics[service.service_name] = service.get_metrics()
            except Exception as e:
                metrics[service.service_name] = {"error": str(e)}
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "services": metrics
        }
    
    @router.get("/metrics/{service_name}")
    async def get_service_metrics(service_name: str):
        """Get detailed metrics for specific service."""
        service = service_map.get(service_name)
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Service not found: {service_name}"
            )
        
        return service.get_metrics()
    
    @router.get("/circuit-breaker")
    async def get_circuit_breaker_status():
        """Get circuit breaker status for all services."""
        cb_status = {}
        for service in services:
            if hasattr(service, '_circuit_breaker') and service._circuit_breaker:
                cb_status[service.service_name] = service._circuit_breaker.get_stats()
            else:
                cb_status[service.service_name] = {"enabled": False}
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "circuit_breakers": cb_status
        }
    
    @router.post("/circuit-breaker/reset")
    async def reset_circuit_breaker(service_name: str):
        """
        Reset circuit breaker for specific service.
        
        Args:
            service_name: Name of service to reset
            
        Returns:
            Success message
        """
        service = service_map.get(service_name)
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Service not found: {service_name}"
            )
        
        try:
            service.reset_circuit_breaker()
            return {
                "success": True,
                "message": f"Circuit breaker reset for {service_name}",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to reset circuit breaker: {str(e)}"
            )
    
    @router.post("/cache/invalidate")
    async def invalidate_cache(service_name: str):
        """
        Invalidate symbol cache for specific service.
        
        Args:
            service_name: Name of service to invalidate cache
            
        Returns:
            Success message
        """
        service = service_map.get(service_name)
        if not service:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Service not found: {service_name}"
            )
        
        try:
            service.invalidate_symbol_cache()
            return {
                "success": True,
                "message": f"Symbol cache invalidated for {service_name}",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to invalidate cache: {str(e)}"
            )
    
    @router.get("/")
    async def root():
        """API info endpoint."""
        return {
            "name": "ForexGPT Health API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics",
                "circuit_breaker": "/circuit-breaker",
                "cache": "/cache"
            },
            "services": list(service_map.keys())
        }
    
    return router


def run_health_server(
    services: List,
    port: int = 8080,
    host: str = "0.0.0.0",
    error_rate_threshold: float = 0.05
):
    """
    Run standalone health check server.
    
    Args:
        services: List of services to monitor
        port: HTTP server port (default: 8080)
        host: Bind address (default: 0.0.0.0)
        error_rate_threshold: Maximum acceptable error rate
        
    Example:
        run_health_server(
            services=[aggregator, dom_service],
            port=8080
        )
        
        # API available at http://localhost:8080/
    """
    if not FASTAPI_AVAILABLE:
        logger.error("Cannot run health server: fastapi not installed")
        return
    
    try:
        app = FastAPI(
            title="ForexGPT Health API",
            description="Health monitoring and management for market data services",
            version="1.0.0"
        )
        
        router = create_health_router(
            services=services,
            error_rate_threshold=error_rate_threshold
        )
        app.include_router(router)
        
        logger.info(f"Starting health server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")
        
    except Exception as e:
        logger.error(f"Failed to start health server: {e}")
        raise


# Simple health check function for non-FastAPI usage
def check_services_health(services: List, error_rate_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Simple health check without HTTP server.
    
    Args:
        services: List of services to check
        error_rate_threshold: Maximum acceptable error rate
        
    Returns:
        Health status dictionary
        
    Example:
        health = check_services_health([aggregator, dom_service])
        if not health["healthy"]:
            print(f"System degraded: {health['overall_status']}")
    """
    checker = HealthChecker(error_rate_threshold=error_rate_threshold)
    return checker.check_all_services(services)
