"""
Multi-Level Stop Loss System

Implements comprehensive risk management with multiple stop loss types.
Based on Two Sigma/Citadel risk management practices.

Stop Loss Types:
1. TECHNICAL: Pattern invalidation
2. VOLATILITY: ATR-based dynamic stops
3. TIME: Maximum holding period
4. CORRELATION: Systemic risk detection
5. DAILY_LOSS: Daily loss limit
6. TRAILING: Lock in profits
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import pandas as pd


class StopLossType(Enum):
    """Types of stop loss triggers."""
    TECHNICAL = "technical"  # Pattern invalidation
    VOLATILITY = "volatility"  # ATR-based
    TIME = "time"  # Max holding period
    CORRELATION = "correlation"  # Market correlation spike
    DAILY_LOSS = "daily_loss"  # Daily loss limit
    TRAILING = "trailing"  # Trailing stop


@dataclass
class StopLossLevel:
    """Individual stop loss level configuration."""
    type: StopLossType
    trigger_value: float
    priority: int  # Lower = higher priority
    enabled: bool = True


class MultiLevelStopLoss:
    """
    Multi-level stop loss system integrating multiple risk controls.

    Based on Citadel/Two Sigma risk management practices.

    Features:
    - Multiple stop loss types with priority ordering
    - ATR-based volatility stops
    - Time-based exits
    - Correlation-based systemic risk protection
    - Daily loss limits
    - Trailing stops to lock profits

    Example:
        >>> stop_manager = MultiLevelStopLoss(
        ...     atr_multiplier=2.0,
        ...     max_holding_hours=48,
        ...     daily_loss_limit_pct=3.0
        ... )
        >>> position = {
        ...     'entry_price': 1.1000,
        ...     'direction': 'long',
        ...     'entry_time': pd.Timestamp.now(),
        ...     'account_balance': 10000,
        ...     'size': 10000
        ... }
        >>> triggered, stop_type, reason = stop_manager.check_stop_triggered(
        ...     position, current_price=1.0970, atr=0.0015
        ... )
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        max_holding_hours: int = 48,
        correlation_threshold: float = 0.85,
        daily_loss_limit_pct: float = 3.0,
        trailing_stop_pct: float = 2.0
    ):
        """
        Initialize multi-level stop loss system.

        Args:
            atr_multiplier: Multiplier for ATR-based stops (default: 2.0)
            max_holding_hours: Maximum holding period in hours (default: 48)
            correlation_threshold: Market correlation threshold (default: 0.85)
            daily_loss_limit_pct: Daily loss limit as % of account (default: 3.0)
            trailing_stop_pct: Trailing stop percentage (default: 2.0)
        """
        self.atr_multiplier = atr_multiplier
        self.max_holding_hours = max_holding_hours
        self.correlation_threshold = correlation_threshold
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.trailing_stop_pct = trailing_stop_pct

        # Track daily P&L
        self.daily_pnl = 0.0
        self.daily_pnl_reset_time = pd.Timestamp.now().normalize()

    def calculate_stop_levels(
        self,
        position: Dict,
        current_price: float,
        atr: float,
        market_correlation: Optional[float] = None
    ) -> Dict[StopLossType, float]:
        """
        Calculate all stop loss levels for a position.

        Args:
            position: Dict with entry_price, direction, entry_time, pattern_type
            current_price: Current market price
            atr: Average True Range
            market_correlation: Current market correlation (optional)

        Returns:
            Dictionary mapping StopLossType to stop price/value
        """
        entry_price = position['entry_price']
        direction = position['direction']  # 'long' or 'short'

        stop_levels = {}

        # 1. TECHNICAL STOP: Pattern invalidation
        # For long: below pattern low
        # For short: above pattern high
        if 'pattern_invalidation_price' in position:
            stop_levels[StopLossType.TECHNICAL] = position['pattern_invalidation_price']

        # 2. VOLATILITY STOP: ATR-based
        # Stop at entry_price ± (atr_multiplier * ATR)
        if direction == 'long':
            stop_levels[StopLossType.VOLATILITY] = entry_price - (self.atr_multiplier * atr)
        else:  # short
            stop_levels[StopLossType.VOLATILITY] = entry_price + (self.atr_multiplier * atr)

        # 3. TIME STOP: Max holding period
        # Not a price level, but checked separately
        entry_time = pd.Timestamp(position['entry_time'])
        hours_held = (pd.Timestamp.now() - entry_time).total_seconds() / 3600
        stop_levels[StopLossType.TIME] = hours_held  # Special: not a price

        # 4. CORRELATION STOP: Market correlation spike
        if market_correlation is not None and market_correlation > self.correlation_threshold:
            # Exit if market becomes too correlated (systemic risk)
            # This is a flag, not a price level
            stop_levels[StopLossType.CORRELATION] = market_correlation

        # 5. DAILY LOSS LIMIT
        # Check if daily loss limit would be breached
        position_pnl = self._calculate_position_pnl(position, current_price)
        potential_total_pnl = self.daily_pnl + position_pnl

        # Reset daily P&L if new day
        current_date = pd.Timestamp.now().normalize()
        if current_date > self.daily_pnl_reset_time:
            self.daily_pnl = 0.0
            self.daily_pnl_reset_time = current_date

        # Calculate price that would trigger daily loss limit
        # This is approximate - assumes this is the only position
        daily_loss_limit = position.get('account_balance', 10000) * (self.daily_loss_limit_pct / 100)
        if abs(self.daily_pnl) >= daily_loss_limit * 0.8:  # 80% of limit
            stop_levels[StopLossType.DAILY_LOSS] = current_price  # Exit now

        # 6. TRAILING STOP
        # Move stop loss as price moves in favorable direction
        if 'highest_price' in position:  # For longs
            trailing_stop = position['highest_price'] * (1 - self.trailing_stop_pct / 100)
            stop_levels[StopLossType.TRAILING] = trailing_stop
        elif 'lowest_price' in position:  # For shorts
            trailing_stop = position['lowest_price'] * (1 + self.trailing_stop_pct / 100)
            stop_levels[StopLossType.TRAILING] = trailing_stop

        return stop_levels

    def check_stop_triggered(
        self,
        position: Dict,
        current_price: float,
        atr: float,
        market_correlation: Optional[float] = None
    ) -> Tuple[bool, Optional[StopLossType], Optional[str]]:
        """
        Check if any stop loss level has been triggered.

        Args:
            position: Position dictionary
            current_price: Current market price
            atr: Average True Range
            market_correlation: Market correlation (optional)

        Returns:
            Tuple of (triggered, stop_type, reason)
        """
        direction = position['direction']
        stop_levels = self.calculate_stop_levels(position, current_price, atr, market_correlation)

        # Check each stop level in priority order
        priority_order = [
            (StopLossType.DAILY_LOSS, 1),
            (StopLossType.CORRELATION, 2),
            (StopLossType.TIME, 3),
            (StopLossType.VOLATILITY, 4),
            (StopLossType.TECHNICAL, 5),
            (StopLossType.TRAILING, 6)
        ]

        for stop_type, priority in priority_order:
            if stop_type not in stop_levels:
                continue

            triggered = False
            reason = ""

            if stop_type == StopLossType.TECHNICAL:
                stop_price = stop_levels[stop_type]
                if direction == 'long' and current_price <= stop_price:
                    triggered = True
                    reason = f"Technical stop hit: price {current_price:.5f} <= {stop_price:.5f}"
                elif direction == 'short' and current_price >= stop_price:
                    triggered = True
                    reason = f"Technical stop hit: price {current_price:.5f} >= {stop_price:.5f}"

            elif stop_type == StopLossType.VOLATILITY:
                stop_price = stop_levels[stop_type]
                if direction == 'long' and current_price <= stop_price:
                    triggered = True
                    reason = f"Volatility stop hit: price {current_price:.5f} <= {stop_price:.5f} ({self.atr_multiplier}x ATR)"
                elif direction == 'short' and current_price >= stop_price:
                    triggered = True
                    reason = f"Volatility stop hit: price {current_price:.5f} >= {stop_price:.5f} ({self.atr_multiplier}x ATR)"

            elif stop_type == StopLossType.TIME:
                hours_held = stop_levels[stop_type]
                if hours_held >= self.max_holding_hours:
                    triggered = True
                    reason = f"Time stop hit: held for {hours_held:.1f} hours (max {self.max_holding_hours})"

            elif stop_type == StopLossType.CORRELATION:
                correlation = stop_levels[stop_type]
                if correlation > self.correlation_threshold:
                    triggered = True
                    reason = f"Correlation stop hit: market correlation {correlation:.2f} > {self.correlation_threshold}"

            elif stop_type == StopLossType.DAILY_LOSS:
                # Already triggered if in stop_levels with current_price
                triggered = True
                reason = f"Daily loss limit approaching: {self.daily_pnl:.2f} (limit {self.daily_loss_limit_pct}%)"

            elif stop_type == StopLossType.TRAILING:
                stop_price = stop_levels[stop_type]
                if direction == 'long' and current_price <= stop_price:
                    triggered = True
                    reason = f"Trailing stop hit: price {current_price:.5f} <= {stop_price:.5f}"
                elif direction == 'short' and current_price >= stop_price:
                    triggered = True
                    reason = f"Trailing stop hit: price {current_price:.5f} >= {stop_price:.5f}"

            if triggered:
                return True, stop_type, reason

        return False, None, None

    def _calculate_position_pnl(self, position: Dict, current_price: float) -> float:
        """
        Calculate current P&L for position.

        Args:
            position: Position dictionary
            current_price: Current market price

        Returns:
            Current P&L value
        """
        entry_price = position['entry_price']
        size = position.get('size', 1.0)
        direction = position['direction']

        if direction == 'long':
            pnl = (current_price - entry_price) * size
        else:  # short
            pnl = (entry_price - current_price) * size

        return pnl

    def update_trailing_stops(self, position: Dict, current_price: float) -> Dict:
        """
        Update trailing stop levels based on current price.

        Args:
            position: Position dictionary
            current_price: Current market price

        Returns:
            Updated position dictionary
        """
        direction = position['direction']

        if direction == 'long':
            # Update highest price seen
            if 'highest_price' not in position or current_price > position['highest_price']:
                position['highest_price'] = current_price
        else:  # short
            # Update lowest price seen
            if 'lowest_price' not in position or current_price < position['lowest_price']:
                position['lowest_price'] = current_price

        return position

    def update_daily_pnl(self, realized_pnl: float):
        """
        Update daily P&L tracker.

        Args:
            realized_pnl: Realized P&L from closed position
        """
        # Reset if new day
        current_date = pd.Timestamp.now().normalize()
        if current_date > self.daily_pnl_reset_time:
            self.daily_pnl = 0.0
            self.daily_pnl_reset_time = current_date

        self.daily_pnl += realized_pnl

    def get_risk_metrics(self, position: Dict, current_price: float, atr: float) -> Dict:
        """
        Get comprehensive risk metrics for a position.

        Args:
            position: Position dictionary
            current_price: Current market price
            atr: Average True Range

        Returns:
            Dictionary with risk metrics
        """
        stop_levels = self.calculate_stop_levels(position, current_price, atr)
        unrealized_pnl = self._calculate_position_pnl(position, current_price)

        # Calculate distance to each stop
        distances = {}
        for stop_type, stop_value in stop_levels.items():
            if stop_type in [StopLossType.TECHNICAL, StopLossType.VOLATILITY, StopLossType.TRAILING]:
                distance_pips = abs(current_price - stop_value) * 10000  # Convert to pips
                distances[stop_type.value] = distance_pips

        # Calculate risk/reward
        entry_price = position['entry_price']
        position_risk_pips = abs(entry_price - stop_levels.get(StopLossType.VOLATILITY, entry_price)) * 10000

        return {
            'unrealized_pnl': unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'position_risk_pips': position_risk_pips,
            'stop_distances': distances,
            'active_stops': [st.value for st in stop_levels.keys()],
            'nearest_stop': min(distances.values()) if distances else None
        }
