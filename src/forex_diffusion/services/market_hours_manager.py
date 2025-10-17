"""
Market Hours Manager

Manages WebSocket connection lifecycle based on Forex market hours.
Disconnects during weekend closure (Friday 22:00 UTC - Sunday 22:00 UTC)
and reconnects automatically when market opens.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable
from loguru import logger


class MarketHoursManager:
    """
    Manages WebSocket connection based on Forex market hours.
    
    Forex Market Hours (UTC):
    - Open: Sunday 22:00 UTC
    - Close: Friday 22:00 UTC
    """
    
    def __init__(
        self,
        on_market_open: Optional[Callable[[], None]] = None,
        on_market_close: Optional[Callable[[], None]] = None,
        check_interval_seconds: int = 60
    ):
        """
        Initialize market hours manager.
        
        Args:
            on_market_open: Callback when market opens (reconnect WS)
            on_market_close: Callback when market closes (disconnect WS)
            check_interval_seconds: How often to check market status
        """
        self.on_market_open = on_market_open
        self.on_market_close = on_market_close
        self.check_interval = check_interval_seconds
        
        self._timer: Optional[threading.Timer] = None
        self._running = False
        self._last_market_state: Optional[bool] = None
        
        logger.info(
            f"MarketHoursManager initialized (check interval: {check_interval_seconds}s)"
        )
    
    def is_market_open(self, now: Optional[datetime] = None) -> bool:
        """
        Check if Forex market is currently open.
        
        Market hours: Sunday 22:00 UTC → Friday 22:00 UTC
        
        Args:
            now: Current time (UTC), defaults to datetime.now(timezone.utc)
            
        Returns:
            True if market is open
        """
        if now is None:
            now = datetime.now(timezone.utc)
        
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        hour = now.hour
        
        # Friday after 22:00 UTC → CLOSED
        if weekday == 4 and hour >= 22:
            return False
        
        # Saturday (all day) → CLOSED
        if weekday == 5:
            return False
        
        # Sunday before 22:00 UTC → CLOSED
        if weekday == 6 and hour < 22:
            return False
        
        # All other times → OPEN
        return True
    
    def get_next_transition(self, now: Optional[datetime] = None) -> tuple[str, datetime]:
        """
        Get next market open/close event.
        
        Args:
            now: Current time (UTC)
            
        Returns:
            Tuple of (event_type, timestamp) where event_type is "open" or "close"
        """
        if now is None:
            now = datetime.now(timezone.utc)
        
        weekday = now.weekday()
        hour = now.hour
        
        # Currently open → next close is Friday 22:00
        if self.is_market_open(now):
            days_until_friday = (4 - weekday) % 7
            if weekday == 4 and hour < 22:
                # It's Friday before 22:00
                next_close = now.replace(hour=22, minute=0, second=0, microsecond=0)
            else:
                # Some other day
                next_close = now + timedelta(days=days_until_friday)
                next_close = next_close.replace(hour=22, minute=0, second=0, microsecond=0)
            return ("close", next_close)
        
        # Currently closed → next open is Sunday 22:00
        days_until_sunday = (6 - weekday) % 7
        if weekday == 6 and hour >= 22:
            # It's Sunday after 22:00 (but we'd be open, shouldn't reach here)
            # This is edge case, market opens in <1 hour
            next_open = now.replace(hour=22, minute=0, second=0, microsecond=0)
        else:
            # Saturday or Friday night or Sunday morning
            if weekday == 6:
                # Sunday before 22:00
                next_open = now.replace(hour=22, minute=0, second=0, microsecond=0)
            else:
                # Friday night or Saturday
                next_open = now + timedelta(days=days_until_sunday)
                next_open = next_open.replace(hour=22, minute=0, second=0, microsecond=0)
        
        return ("open", next_open)
    
    def start(self):
        """Start monitoring market hours."""
        if self._running:
            logger.warning("MarketHoursManager already running")
            return
        
        self._running = True
        self._last_market_state = None
        
        logger.info("MarketHoursManager started")
        self._schedule_check()
    
    def stop(self):
        """Stop monitoring market hours."""
        self._running = False
        
        if self._timer:
            self._timer.cancel()
            self._timer = None
        
        logger.info("MarketHoursManager stopped")
    
    def _schedule_check(self):
        """Schedule next market status check."""
        if not self._running:
            return
        
        self._timer = threading.Timer(self.check_interval, self._check_market_status)
        self._timer.daemon = True
        self._timer.start()
    
    def _check_market_status(self):
        """Check market status and trigger callbacks on transitions."""
        try:
            now = datetime.now(timezone.utc)
            current_state = self.is_market_open(now)
            
            # Detect state change
            if self._last_market_state is not None and self._last_market_state != current_state:
                if current_state:
                    # Market just opened
                    logger.info("🔔 Forex market OPENED (Sunday 22:00 UTC)")
                    if self.on_market_open:
                        try:
                            self.on_market_open()
                        except Exception as e:
                            logger.error(f"Error in on_market_open callback: {e}")
                else:
                    # Market just closed
                    logger.info("🔔 Forex market CLOSED (Friday 22:00 UTC)")
                    if self.on_market_close:
                        try:
                            self.on_market_close()
                        except Exception as e:
                            logger.error(f"Error in on_market_close callback: {e}")
            
            # Log status on first check
            if self._last_market_state is None:
                status = "OPEN" if current_state else "CLOSED"
                event_type, next_time = self.get_next_transition(now)
                time_until = next_time - now
                hours = int(time_until.total_seconds() // 3600)
                minutes = int((time_until.total_seconds() % 3600) // 60)
                
                logger.info(
                    f"Market currently {status}, next {event_type} in {hours}h {minutes}m "
                    f"({next_time.strftime('%Y-%m-%d %H:%M UTC')})"
                )
            
            self._last_market_state = current_state
        
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
        
        finally:
            # Schedule next check
            self._schedule_check()
