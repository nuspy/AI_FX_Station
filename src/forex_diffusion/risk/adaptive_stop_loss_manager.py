"""
Adaptive Stop Loss Manager

Dynamically adjusts stop loss and take profit based on multiple real-time factors:
- ATR (volatility)
- Spread (market conditions)
- News events (risk reduction before/after news)
- Market regime
- Multi-level stop loss (hard, volatility-based, time-based)
- Dynamic trailing stops

Integrates with AutomatedTradingEngine for intelligent risk management.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import numpy as np


@dataclass
class StopLossLevel:
    """Represents a stop loss level."""
    type: str  # 'hard', 'volatility', 'time', 'trailing'
    price: float
    active: bool
    triggered_at: Optional[datetime] = None


@dataclass
class AdaptationFactors:
    """Factors that influence SL/TP adaptation."""
    atr: float
    current_spread: float
    avg_spread: float
    spread_ratio: float  # current / avg
    news_risk_level: str  # 'none', 'low', 'medium', 'high'
    regime: Optional[str]
    time_in_position_hours: float
    unrealized_pnl_pct: float


class AdaptiveStopLossManager:
    """
    Manages adaptive stop loss and take profit levels.

    Features:
    - ATR-based volatility stops
    - Spread-aware adjustments
    - News event risk reduction
    - Multi-level stops (hard, volatility, time)
    - Dynamic trailing stops
    - Regime-specific adjustments
    """

    def __init__(
        self,
        base_sl_atr_multiplier: float = 2.0,
        base_tp_atr_multiplier: float = 3.0,
        trailing_enabled: bool = True,
        trailing_activation_pct: float = 50.0,
        max_spread_multiplier: float = 3.0,
        news_sl_tightening_pct: float = 20.0,
    ):
        """
        Initialize adaptive stop loss manager.

        Args:
            base_sl_atr_multiplier: Base SL as multiple of ATR
            base_tp_atr_multiplier: Base TP as multiple of ATR
            trailing_enabled: Enable trailing stop
            trailing_activation_pct: % of TP to activate trailing (0-100)
            max_spread_multiplier: Max spread vs avg before widening SL
            news_sl_tightening_pct: % to tighten SL before news events
        """
        self.base_sl_atr_multiplier = base_sl_atr_multiplier
        self.base_tp_atr_multiplier = base_tp_atr_multiplier
        self.trailing_enabled = trailing_enabled
        self.trailing_activation_pct = trailing_activation_pct
        self.max_spread_multiplier = max_spread_multiplier
        self.news_sl_tightening_pct = news_sl_tightening_pct

        # Tracking
        self.position_stops: Dict[str, List[StopLossLevel]] = {}
        self.adaptation_history: List[Dict] = []

        logger.info(
            f"AdaptiveStopLossManager initialized: "
            f"base_sl={base_sl_atr_multiplier:.1f}x ATR, "
            f"base_tp={base_tp_atr_multiplier:.1f}x ATR, "
            f"trailing={trailing_enabled}"
        )

    def calculate_initial_stops(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        atr: float,
        current_spread: float,
        avg_spread: float,
        regime: Optional[str] = None,
        sl_multiplier_override: Optional[float] = None,
        tp_multiplier_override: Optional[float] = None,
    ) -> Tuple[float, float, List[StopLossLevel]]:
        """
        Calculate initial stop loss and take profit with multi-level stops.

        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            entry_price: Entry price
            atr: Current ATR
            current_spread: Current spread
            avg_spread: Average spread
            regime: Market regime
            sl_multiplier_override: Override base SL multiplier
            tp_multiplier_override: Override base TP multiplier

        Returns:
            Tuple of (stop_loss, take_profit, stop_levels)
        """
        is_long = direction == 'long'

        # Use overrides if provided (from optimized parameters)
        sl_mult = sl_multiplier_override or self.base_sl_atr_multiplier
        tp_mult = tp_multiplier_override or self.base_tp_atr_multiplier

        # Adjust for spread conditions
        spread_ratio = current_spread / avg_spread if avg_spread > 0 else 1.0
        if spread_ratio > self.max_spread_multiplier:
            # Widen SL if spread is abnormally high
            spread_adjustment = 1.0 + (spread_ratio - self.max_spread_multiplier) * 0.1
            sl_mult *= spread_adjustment
            logger.warning(
                f"High spread detected ({spread_ratio:.2f}x avg), "
                f"widening SL to {sl_mult:.2f}x ATR"
            )

        # Adjust for regime (if provided)
        if regime in ['high_volatility', 'transition']:
            sl_mult *= 1.2  # Wider stops in volatile regimes
            logger.info(f"Regime {regime}: widening SL to {sl_mult:.2f}x ATR")
        elif regime in ['ranging', 'accumulation']:
            sl_mult *= 0.9  # Tighter stops in ranging markets
            logger.info(f"Regime {regime}: tightening SL to {sl_mult:.2f}x ATR")

        # Calculate base stops
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult

        stop_loss = entry_price - sl_distance if is_long else entry_price + sl_distance
        take_profit = entry_price + tp_distance if is_long else entry_price - tp_distance

        # Create multi-level stops
        stop_levels = []

        # 1. Hard stop (primary)
        stop_levels.append(StopLossLevel(
            type='hard',
            price=stop_loss,
            active=True
        ))

        # 2. Volatility-based stop (wider than hard stop)
        volatility_sl_distance = atr * (sl_mult * 1.5)
        volatility_stop = (
            entry_price - volatility_sl_distance if is_long
            else entry_price + volatility_sl_distance
        )
        stop_levels.append(StopLossLevel(
            type='volatility',
            price=volatility_stop,
            active=True
        ))

        # 3. Time-based stop (exits after X hours if no profit)
        # This is checked separately in update_stops

        # Store levels
        self.position_stops[symbol] = stop_levels

        logger.info(
            f"Calculated stops for {direction} {symbol}: "
            f"SL={stop_loss:.5f} ({sl_mult:.2f}x ATR), "
            f"TP={take_profit:.5f} ({tp_mult:.2f}x ATR), "
            f"{len(stop_levels)} levels"
        )

        return stop_loss, take_profit, stop_levels

    def update_stops(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        entry_time: datetime,
        current_price: float,
        current_stop_loss: float,
        current_take_profit: float,
        factors: AdaptationFactors,
    ) -> Tuple[float, Optional[float], bool, Optional[str]]:
        """
        Update stop loss based on real-time adaptation factors.

        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            entry_price: Entry price
            entry_time: Entry timestamp
            current_price: Current market price
            current_stop_loss: Current SL price
            current_take_profit: Current TP price
            factors: Adaptation factors

        Returns:
            Tuple of (new_sl, new_tp, stop_triggered, trigger_reason)
        """
        is_long = direction == 'long'
        new_sl = current_stop_loss
        new_tp = current_take_profit
        stop_triggered = False
        trigger_reason = None

        # Check hard stop
        if is_long and current_price <= current_stop_loss:
            stop_triggered = True
            trigger_reason = "hard_stop"
        elif not is_long and current_price >= current_stop_loss:
            stop_triggered = True
            trigger_reason = "hard_stop"

        if stop_triggered:
            logger.info(f"Hard stop triggered for {symbol} at {current_price:.5f}")
            return new_sl, new_tp, stop_triggered, trigger_reason

        # Adjust for news risk
        if factors.news_risk_level in ['medium', 'high']:
            # Tighten SL before news events
            tightening_factor = (
                self.news_sl_tightening_pct / 100.0
                if factors.news_risk_level == 'medium'
                else self.news_sl_tightening_pct * 1.5 / 100.0
            )

            if is_long:
                # Move SL closer to current price (up)
                tightened_sl = current_stop_loss + (current_price - current_stop_loss) * tightening_factor
                if tightened_sl > new_sl:
                    new_sl = tightened_sl
                    logger.info(
                        f"Tightening SL for {symbol} due to {factors.news_risk_level} news risk: "
                        f"{current_stop_loss:.5f} -> {new_sl:.5f}"
                    )
            else:
                # Move SL closer to current price (down)
                tightened_sl = current_stop_loss - (current_stop_loss - current_price) * tightening_factor
                if tightened_sl < new_sl:
                    new_sl = tightened_sl
                    logger.info(
                        f"Tightening SL for {symbol} due to {factors.news_risk_level} news risk: "
                        f"{current_stop_loss:.5f} -> {new_sl:.5f}"
                    )

        # Implement trailing stop
        if self.trailing_enabled and current_take_profit is not None:
            # Calculate profit progress
            if is_long:
                target_move = current_take_profit - entry_price
                current_move = current_price - entry_price
            else:
                target_move = entry_price - current_take_profit
                current_move = entry_price - current_price

            profit_progress_pct = (current_move / target_move * 100.0) if target_move > 0 else 0.0

            # Activate trailing if we've reached activation threshold
            if profit_progress_pct >= self.trailing_activation_pct:
                # Trail at 50% of current profit
                trail_distance = factors.atr * 1.0

                if is_long:
                    trailing_sl = current_price - trail_distance
                    if trailing_sl > new_sl:
                        new_sl = trailing_sl
                        logger.info(
                            f"Trailing stop activated for {symbol}: {new_sl:.5f} "
                            f"(profit progress: {profit_progress_pct:.1f}%)"
                        )
                else:
                    trailing_sl = current_price + trail_distance
                    if trailing_sl < new_sl:
                        new_sl = trailing_sl
                        logger.info(
                            f"Trailing stop activated for {symbol}: {new_sl:.5f} "
                            f"(profit progress: {profit_progress_pct:.1f}%)"
                        )

        # Time-based exit (if position is losing after X hours)
        if factors.time_in_position_hours > 4.0 and factors.unrealized_pnl_pct < -0.5:
            # Position losing after 4 hours - consider time stop
            logger.warning(
                f"Time-based stop consideration for {symbol}: "
                f"{factors.time_in_position_hours:.1f}h, {factors.unrealized_pnl_pct:.2f}% P&L"
            )
            # Could trigger here or tighten SL significantly

        # Check volatility-based stop
        if symbol in self.position_stops:
            for level in self.position_stops[symbol]:
                if level.type == 'volatility' and level.active:
                    if is_long and current_price <= level.price:
                        stop_triggered = True
                        trigger_reason = "volatility_stop"
                        level.triggered_at = datetime.now()
                    elif not is_long and current_price >= level.price:
                        stop_triggered = True
                        trigger_reason = "volatility_stop"
                        level.triggered_at = datetime.now()

        # Record adaptation
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'old_sl': current_stop_loss,
            'new_sl': new_sl,
            'current_price': current_price,
            'factors': factors.__dict__,
        })

        return new_sl, new_tp, stop_triggered, trigger_reason

    def assess_news_risk(
        self,
        symbol: str,
        current_time: datetime,
        upcoming_news_events: List[Dict[str, Any]]
    ) -> str:
        """
        Assess news risk level based on upcoming events.

        Args:
            symbol: Trading symbol
            current_time: Current timestamp
            upcoming_news_events: List of upcoming news events

        Returns:
            Risk level: 'none', 'low', 'medium', 'high'
        """
        if not upcoming_news_events:
            return 'none'

        # Find closest high-impact event
        high_impact_events = [
            e for e in upcoming_news_events
            if e.get('impact', '') in ['high', 'medium']
            and symbol[:3] in e.get('currency', '')  # Affects this currency
        ]

        if not high_impact_events:
            return 'none'

        # Check time to event
        closest_event = min(
            high_impact_events,
            key=lambda e: abs((e['time'] - current_time).total_seconds())
        )

        time_to_event = (closest_event['time'] - current_time).total_seconds() / 60  # minutes

        # Risk increases as event approaches
        if abs(time_to_event) < 15:  # Within 15 minutes
            return 'high'
        elif abs(time_to_event) < 60:  # Within 1 hour
            return 'medium'
        elif abs(time_to_event) < 240:  # Within 4 hours
            return 'low'
        else:
            return 'none'

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics on stop loss adaptations."""
        if not self.adaptation_history:
            return {
                'total_adaptations': 0,
                'avg_sl_change_pips': 0.0,
            }

        sl_changes = [
            abs(a['new_sl'] - a['old_sl']) * 10000  # Convert to pips
            for a in self.adaptation_history
        ]

        return {
            'total_adaptations': len(self.adaptation_history),
            'avg_sl_change_pips': np.mean(sl_changes) if sl_changes else 0.0,
            'max_sl_change_pips': max(sl_changes) if sl_changes else 0.0,
        }

    def clear_position(self, symbol: str):
        """Clear stops for closed position."""
        if symbol in self.position_stops:
            del self.position_stops[symbol]
