#!/usr/bin/env python3
"""
ForexGPT API Server Runner

Starts the FastAPI server for ForexGPT sentiment and calendar API.

Usage:
    python run_api.py                    # Development mode
    python run_api.py --production       # Production mode
    python run_api.py --port 8080        # Custom port
"""
import argparse
import uvicorn
from loguru import logger

from src.forex_diffusion.api.config import settings


def main():
    """Run the API server"""
    parser = argparse.ArgumentParser(description="ForexGPT API Server")
    parser.add_argument(
        "--host",
        type=str,
        default=settings.host,
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run in production mode (no reload, multiple workers)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.workers,
        help="Number of worker processes (production mode only)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=settings.log_level.lower(),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=args.log_level.upper(),
    )

    logger.info("Starting ForexGPT API server...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Mode: {'Production' if args.production else 'Development'}")

    if args.production:
        logger.info(f"Workers: {args.workers}")

        # Production mode: multiple workers, no reload
        uvicorn.run(
            "src.forex_diffusion.api.main:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level,
            access_log=True,
        )
    else:
        logger.info("Reload: enabled")

        # Development mode: single worker, auto-reload
        uvicorn.run(
            "src.forex_diffusion.api.main:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
            access_log=True,
        )


if __name__ == "__main__":
    main()
