"""
VIX Service - Background service for CBOE Volatility Index monitoring.

Fetches VIX from Yahoo Finance and provides volatility filtering for trading.
"""
from __future__ import annotations

from typing import Optional, Dict
from datetime import datetime, timezone
import asyncio

from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .base_service import ThreadedBackgroundService


class VIXService(ThreadedBackgroundService):
    """
    Background service that fetches and monitors VIX (CBOE Volatility Index).
    
    VIX Levels:
    - < 12: Complacency (low volatility)
    - 12-20: Normal volatility
    - 20-30: Elevated volatility (concern)
    - > 30: High volatility (fear/panic)
    
    Used in trading engine as volatility filter for position sizing.
    """

    def __init__(self, engine: Engine, interval_seconds: int = 300):
        """
        Initialize VIX service.
        
        Args:
            engine: SQLAlchemy engine for database access
            interval_seconds: Fetch interval (default: 300s = 5min)
        """
        super().__init__(
            engine=engine,
            symbols=["VIX"],  # Special symbol for VIX
            interval_seconds=interval_seconds,
            enable_circuit_breaker=True
        )
        
        # Cache latest VIX value
        self._latest_vix: Optional[float] = None
        self._latest_classification: Optional[str] = None
        self._latest_timestamp: Optional[int] = None
        
        # Create VIX provider
        from ..providers.sentiment_provider import VIXProvider
        self.vix_provider = VIXProvider()
    
    @property
    def service_name(self) -> str:
        """Service name for logging."""
        return "VIXService"
    
    def _process_iteration(self):
        """
        Process one VIX fetch iteration.
        
        Called by base class in background thread. Fetches VIX from Yahoo Finance
        and stores in database.
        """
        try:
            # Fetch VIX asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            vix_data = loop.run_until_complete(
                self.vix_provider._fetch_sentiment_impl("VIX", None, None)
            )
            
            loop.close()
            
            if vix_data and len(vix_data) > 0:
                data = vix_data[0]
                
                vix_value = data.get('value')
                classification = data.get('classification')
                timestamp = data.get('timestamp')
                
                # Update cache
                self._latest_vix = vix_value
                self._latest_classification = classification
                self._latest_timestamp = timestamp
                
                # Store in database
                self._store_vix(vix_value, classification, timestamp)
                
                logger.info(
                    f"VIX updated: {vix_value:.2f} ({classification})"
                )
            else:
                logger.warning("No VIX data received")
                
        except Exception as e:
            logger.error(f"Failed to fetch VIX: {e}")
    
    def _store_vix(self, value: float, classification: str, timestamp: int):
        """Store VIX data in database."""
        try:
            with self.engine.begin() as conn:
                # Create table if not exists
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS vix_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts_utc BIGINT NOT NULL,
                        value REAL NOT NULL,
                        classification TEXT NOT NULL,
                        ts_created_ms BIGINT NOT NULL
                    )
                """))
                
                # Create index if not exists
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_vix_ts ON vix_data(ts_utc)
                """))
                
                # Insert VIX data
                conn.execute(text("""
                    INSERT INTO vix_data (ts_utc, value, classification, ts_created_ms)
                    VALUES (:ts_utc, :value, :classification, :ts_created_ms)
                """), {
                    "ts_utc": timestamp,
                    "value": value,
                    "classification": classification,
                    "ts_created_ms": int(datetime.now(timezone.utc).timestamp() * 1000)
                })
                
                logger.debug(f"Stored VIX: {value:.2f} at {timestamp}")
                
        except Exception as e:
            logger.error(f"Failed to store VIX data: {e}")
    
    def get_latest_vix(self) -> Optional[float]:
        """
        Get latest VIX value.
        
        Returns:
            VIX value or None if not available
        """
        # Try cache first
        if self._latest_vix is not None:
            return self._latest_vix
        
        # Fallback to database
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT value FROM vix_data ORDER BY ts_utc DESC LIMIT 1"
                ))
                row = result.fetchone()
                if row:
                    self._latest_vix = row[0]
                    return self._latest_vix
        except Exception as e:
            logger.debug(f"Could not fetch VIX from database: {e}")
        
        return None
    
    def get_latest_classification(self) -> Optional[str]:
        """
        Get latest VIX classification.
        
        Returns:
            Classification string or None
        """
        if self._latest_classification is not None:
            return self._latest_classification
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT classification FROM vix_data ORDER BY ts_utc DESC LIMIT 1"
                ))
                row = result.fetchone()
                if row:
                    self._latest_classification = row[0]
                    return self._latest_classification
        except Exception as e:
            logger.debug(f"Could not fetch VIX classification: {e}")
        
        return None
    
    def get_vix_metrics(self) -> Optional[Dict]:
        """
        Get complete VIX metrics.
        
        Returns:
            Dictionary with VIX value, classification, and timestamp
        """
        if self._latest_vix is not None:
            return {
                'value': self._latest_vix,
                'classification': self._latest_classification,
                'timestamp': self._latest_timestamp
            }
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT value, classification, ts_utc FROM vix_data "
                    "ORDER BY ts_utc DESC LIMIT 1"
                ))
                row = result.fetchone()
                if row:
                    return {
                        'value': row[0],
                        'classification': row[1],
                        'timestamp': row[2]
                    }
        except Exception as e:
            logger.debug(f"Could not fetch VIX metrics: {e}")
        
        return None
    
    def get_volatility_adjustment(self, base_size: float) -> float:
        """
        Calculate position size adjustment based on VIX level.
        
        Args:
            base_size: Base position size
            
        Returns:
            Adjusted position size
        """
        vix_value = self.get_latest_vix()
        
        if vix_value is None:
            return base_size
        
        # VIX-based adjustments
        if vix_value > 30:
            # High volatility (fear) - reduce size significantly
            adjustment = 0.7
            logger.info(f"🔴 VIX > 30 ({vix_value:.1f}): Position reduced 0.7x (Fear)")
        elif vix_value > 20:
            # Elevated volatility - reduce size moderately
            adjustment = 0.85
            logger.info(f"🟠 VIX 20-30 ({vix_value:.1f}): Position reduced 0.85x (Concern)")
        elif vix_value < 12:
            # Low volatility (complacency) - slight caution on mean reversion
            adjustment = 0.95
            logger.info(f"🟡 VIX < 12 ({vix_value:.1f}): Slight caution 0.95x (Complacency)")
        else:
            # Normal volatility - no adjustment
            adjustment = 1.0
        
        return base_size * adjustment
