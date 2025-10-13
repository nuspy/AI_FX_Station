"""
Error Recovery System for Live Trading

Provides comprehensive error handling and recovery mechanisms to prevent
silent failures and minimize losses during live trading.

Key Features:
- Broker connection recovery with exponential backoff
- Insufficient funds handling with position size reduction
- Emergency procedures for critical failures
- Error classification and routing
- Administrator alerting
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from loguru import logger


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class BrokerConnectionError(Exception):
    """Broker connection lost or failed"""
    pass


class InsufficientFundsError(Exception):
    """Insufficient funds for trade"""
    pass


class InvalidOrderError(Exception):
    """Order parameters invalid"""
    pass


class PositionMonitoringError(Exception):
    """Position monitoring failed"""
    pass


class CriticalSystemError(Exception):
    """Critical system failure requiring shutdown"""
    pass


@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    timestamp: datetime
    error_type: str
    severity: ErrorSeverity
    message: str
    context: dict
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class ErrorRecoveryManager:
    """
    Manages error recovery strategies for live trading.
    
    Features:
    - Exponential backoff for transient errors
    - Automatic reconnection for broker disconnects
    - Position size reduction for insufficient funds
    - Emergency shutdown for critical errors
    - Error tracking and statistics
    """
    
    def __init__(self, max_consecutive_errors: int = 5):
        self.max_consecutive_errors = max_consecutive_errors
        self.consecutive_errors = 0
        self.error_history: list[ErrorRecord] = []
        self.alert_callback: Optional[Callable] = None
    
    def set_alert_callback(self, callback: Callable[[str], None]):
        """Set callback for administrator alerts"""
        self.alert_callback = callback
    
    def alert_administrator(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR):
        """Send alert to administrator"""
        logger.log(severity.value.upper(), f"ADMIN ALERT: {message}")
        
        if self.alert_callback:
            try:
                self.alert_callback(message)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
    
    def record_error(
        self,
        error_type: str,
        severity: ErrorSeverity,
        message: str,
        context: dict = None
    ) -> ErrorRecord:
        """Record an error occurrence"""
        record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=error_type,
            severity=severity,
            message=message,
            context=context or {}
        )
        
        self.error_history.append(record)
        
        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        return record
    
    def resolve_error(self, record: ErrorRecord):
        """Mark error as resolved"""
        record.resolved = True
        record.resolution_time = datetime.now()
    
    def reset_consecutive_errors(self):
        """Reset consecutive error counter (call on success)"""
        if self.consecutive_errors > 0:
            logger.info(f"Recovered after {self.consecutive_errors} consecutive errors")
        self.consecutive_errors = 0
    
    def increment_consecutive_errors(self) -> int:
        """Increment and return consecutive error count"""
        self.consecutive_errors += 1
        return self.consecutive_errors
    
    def should_shutdown(self) -> bool:
        """Check if system should shutdown due to errors"""
        return self.consecutive_errors >= self.max_consecutive_errors
    
    def get_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        return min(2 ** attempt, 60)  # Max 60 seconds
    
    def retry_with_backoff(
        self,
        func: Callable,
        max_retries: int = 3,
        error_types: tuple = (Exception,),
        operation_name: str = "operation"
    ) -> Any:
        """
        Retry function with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            error_types: Tuple of exception types to catch
            operation_name: Name for logging
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = func()
                
                if attempt > 0:
                    logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                
                return result
                
            except error_types as e:
                last_exception = e
                
                if attempt < max_retries - 1:
                    delay = self.get_backoff_delay(attempt)
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"{operation_name} failed after {max_retries} attempts: {e}"
                    )
        
        raise last_exception
    
    def handle_broker_connection_error(
        self,
        broker_api: Any,
        signal: dict = None
    ) -> bool:
        """
        Handle broker connection error.
        
        Returns:
            True if reconnected, False otherwise
        """
        error = self.record_error(
            error_type="BrokerConnectionError",
            severity=ErrorSeverity.ERROR,
            message="Broker connection lost",
            context={"signal": signal}
        )
        
        logger.error("Broker connection error detected, attempting reconnection...")
        
        try:
            # Attempt reconnection with exponential backoff
            def reconnect():
                if hasattr(broker_api, 'disconnect'):
                    broker_api.disconnect()
                time.sleep(1)
                if hasattr(broker_api, 'connect'):
                    broker_api.connect()
                elif hasattr(broker_api, 'reconnect'):
                    broker_api.reconnect()
                return True
            
            self.retry_with_backoff(
                reconnect,
                max_retries=5,
                error_types=(Exception,),
                operation_name="Broker reconnection"
            )
            
            logger.info("Broker reconnected successfully")
            self.resolve_error(error)
            return True
            
        except Exception as e:
            logger.critical(f"Failed to reconnect to broker after multiple attempts: {e}")
            self.alert_administrator(
                f"CRITICAL: Broker reconnection failed: {e}",
                ErrorSeverity.CRITICAL
            )
            return False
    
    def handle_insufficient_funds(
        self,
        signal: dict,
        account_balance: float,
        reduction_factor: float = 0.5
    ) -> Optional[dict]:
        """
        Handle insufficient funds error by reducing position size.
        
        Args:
            signal: Original trading signal
            account_balance: Current account balance
            reduction_factor: Factor to reduce position size by
            
        Returns:
            Modified signal with reduced size, or None if still insufficient
        """
        error = self.record_error(
            error_type="InsufficientFundsError",
            severity=ErrorSeverity.WARNING,
            message=f"Insufficient funds for trade (balance: ${account_balance:.2f})",
            context={"signal": signal, "balance": account_balance}
        )
        
        logger.warning(f"Insufficient funds, reducing position size by {reduction_factor}")
        
        # Create modified signal
        reduced_signal = signal.copy()
        
        if 'size' in reduced_signal:
            original_size = reduced_signal['size']
            reduced_signal['size'] = original_size * reduction_factor
            
            logger.info(
                f"Reduced position size: {original_size:.2f} → {reduced_signal['size']:.2f}"
            )
            
            self.resolve_error(error)
            return reduced_signal
        
        logger.error("Cannot reduce position size (no 'size' field in signal)")
        return None
    
    def handle_invalid_order(self, signal: dict, error: Exception):
        """Handle invalid order error"""
        error_record = self.record_error(
            error_type="InvalidOrderError",
            severity=ErrorSeverity.ERROR,
            message=f"Invalid order parameters: {error}",
            context={"signal": signal}
        )
        
        logger.error(f"Invalid order error: {error}")
        self.alert_administrator(
            f"Invalid order detected: {signal}. Error: {error}",
            ErrorSeverity.ERROR
        )
    
    def emergency_close_all_positions(
        self,
        broker_api: Any,
        positions: dict
    ) -> list[str]:
        """
        Emergency procedure to close all open positions.
        
        Returns:
            List of symbols that failed to close
        """
        logger.critical("EMERGENCY: Attempting to close all positions")
        
        self.alert_administrator(
            "EMERGENCY SHUTDOWN: Closing all positions",
            ErrorSeverity.CRITICAL
        )
        
        failed_closes = []
        
        for symbol, position in positions.items():
            logger.info(f"Emergency closing position: {symbol}")
            
            # Try multiple times with different methods
            closed = False
            
            for attempt in range(3):
                try:
                    if hasattr(broker_api, 'close_position'):
                        broker_api.close_position(symbol, reason="emergency")
                        closed = True
                        break
                    elif hasattr(broker_api, 'market_close'):
                        broker_api.market_close(symbol)
                        closed = True
                        break
                    elif hasattr(broker_api, 'place_order'):
                        # Place opposite order to close
                        close_order = {
                            'symbol': symbol,
                            'direction': 'sell' if position.direction == 'long' else 'buy',
                            'size': position.size,
                            'type': 'market'
                        }
                        broker_api.place_order(close_order)
                        closed = True
                        break
                except Exception as e:
                    logger.error(f"Failed to close {symbol} (attempt {attempt + 1}): {e}")
                    time.sleep(1)
            
            if not closed:
                logger.critical(f"FAILED to close position: {symbol}")
                failed_closes.append(symbol)
        
        if failed_closes:
            self.alert_administrator(
                f"CRITICAL: Failed to close positions: {failed_closes}",
                ErrorSeverity.CRITICAL
            )
        else:
            logger.info("All positions closed successfully")
        
        return failed_closes
    
    def get_error_statistics(self) -> dict:
        """Get error statistics"""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {
                'total_errors': 0,
                'by_type': {},
                'by_severity': {},
                'resolved_count': 0,
                'unresolved_count': 0
            }
        
        # Count by type
        by_type = {}
        for error in self.error_history:
            by_type[error.error_type] = by_type.get(error.error_type, 0) + 1
        
        # Count by severity
        by_severity = {}
        for error in self.error_history:
            sev = error.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        # Count resolved
        resolved_count = sum(1 for e in self.error_history if e.resolved)
        unresolved_count = total_errors - resolved_count
        
        return {
            'total_errors': total_errors,
            'by_type': by_type,
            'by_severity': by_severity,
            'resolved_count': resolved_count,
            'unresolved_count': unresolved_count,
            'consecutive_errors': self.consecutive_errors
        }
