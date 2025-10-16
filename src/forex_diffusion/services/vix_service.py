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
        
        # Yahoo Finance VIX endpoint
        self.base_url = "https://query1.finance.yahoo.com/v7/finance/quote"
        
        # Rate limiting tracking
        self._last_429_time: Optional[float] = None
        self._backoff_until: Optional[float] = None
    
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
            # Check if we're in backoff period
            import time
            if self._backoff_until and time.time() < self._backoff_until:
                remaining = int(self._backoff_until - time.time())
                logger.debug(f"VIX fetch in backoff period, {remaining}s remaining")
                return
            
            # Fetch VIX synchronously using requests
            import requests
            from requests.exceptions import HTTPError
            
            params = {
                'symbols': '^VIX',
                'fields': 'regularMarketPrice,regularMarketTime'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'quoteResponse' not in data or 'result' not in data['quoteResponse']:
                logger.warning("No VIX data returned from Yahoo Finance")
                return
            
            result = data['quoteResponse']['result'][0]
            
            vix_value = result.get('regularMarketPrice')
            timestamp = result.get('regularMarketTime', int(datetime.now(timezone.utc).timestamp()))
            
            if vix_value is None:
                logger.warning("VIX value is None")
                return
            
            # Classify VIX level
            if vix_value < 12:
                classification = "Complacency"
            elif vix_value < 20:
                classification = "Normal"
            elif vix_value < 30:
                classification = "Concern"
            else:
                classification = "Fear"
            
            # Convert timestamp to milliseconds
            timestamp_ms = timestamp * 1000
            
            # Update cache
            self._latest_vix = vix_value
            self._latest_classification = classification
            self._latest_timestamp = timestamp_ms
            
            # Store in database
            self._store_vix(vix_value, classification, timestamp_ms)
            
            logger.info(f"VIX updated: {vix_value:.2f} ({classification})")
            
            # Clear backoff on successful fetch
            self._backoff_until = None
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited - implement exponential backoff
                import time
                now = time.time()
                
                # Calculate backoff duration (5 min, 10 min, 30 min, 1 hour)
                if self._last_429_time and (now - self._last_429_time) < 3600:
                    # Recent 429, increase backoff
                    backoff_minutes = min(60, (now - self._last_429_time) / 60 * 2)
                else:
                    # First 429 or long time since last, start with 5 min
                    backoff_minutes = 5
                
                self._last_429_time = now
                self._backoff_until = now + (backoff_minutes * 60)
                
                logger.warning(
                    f"VIX fetch rate limited (429). Backing off for {backoff_minutes:.0f} minutes. "
                    f"Yahoo Finance allows ~2000 requests/hour. Current interval: {self._interval}s"
                )
            else:
                logger.error(f"Failed to fetch VIX: HTTP {e.response.status_code} - {e}")
        
        except Exception as e:
            logger.error(f"Failed to fetch VIX: {e}")
    
    def _store_vix(self, value: float, classification: str, timestamp: int):
        """
        Store VIX data in database.
        
        Note: Requires vix_data table created by Alembic migration 0016.
        """
        try:
            with self.engine.begin() as conn:
                # Insert VIX data (table created by Alembic migration 0016)
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
            logger.error(f"Failed to store VIX data: {e} (Did you run 'alembic upgrade head'?)")
    
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
