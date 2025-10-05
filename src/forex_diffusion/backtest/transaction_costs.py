"""
Realistic Transaction Cost Model

Models all trading costs to avoid overoptimistic backtest results:
- Spread (bid-ask, time-varying)
- Commission (fixed or percentage, tiered)
- Slippage (volume and volatility dependent)
- Market Impact (for large orders)

Reference: "Algorithmic Trading" by Ernest Chan
"""
from __future__ import annotations

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, time
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class TradeExecution:
    """Execution details with costs"""
    entry_price: float  # Intended price
    execution_price: float  # Actual execution after costs
    spread_cost: float  # Bid-ask spread cost
    commission_cost: float  # Broker commission
    slippage_cost: float  # Slippage due to volume/volatility
    market_impact_cost: float  # Price impact of order
    total_cost: float  # Sum of all costs
    cost_bps: float  # Total cost in basis points


class CostModel:
    """
    Realistic transaction cost model for backtesting.

    Ensures backtest results account for all real trading costs.
    """

    def __init__(
        self,
        # Spread parameters
        base_spread_bps: float = 1.0,  # Base spread in basis points
        offhours_spread_mult: float = 1.5,  # Spread multiplier outside trading hours
        news_spread_mult: float = 2.0,  # Spread multiplier during news

        # Commission parameters
        commission_type: str = "percentage",  # "fixed" or "percentage"
        commission_rate: float = 0.0002,  # 0.02% = 2 bps
        commission_fixed: float = 0.0,  # Fixed cost per trade
        commission_tiers: Optional[Dict[float, float]] = None,  # Volume tiers

        # Slippage parameters
        slippage_k: float = 0.1,  # Slippage coefficient
        min_slippage_bps: float = 0.1,  # Minimum slippage
        max_slippage_bps: float = 10.0,  # Maximum slippage cap

        # Market impact parameters
        impact_enabled: bool = True,
        impact_coefficient: float = 0.5,  # Impact coefficient for square-root law
        adv_window: int = 20,  # Average daily volume window
    ):
        """
        Initialize cost model.

        Args:
            base_spread_bps: Base bid-ask spread in basis points
            offhours_spread_mult: Spread multiplier for off-hours trading
            news_spread_mult: Spread multiplier during high-impact news
            commission_type: "fixed" or "percentage"
            commission_rate: Commission as % of trade value
            commission_fixed: Fixed commission per trade
            commission_tiers: Optional volume-based tiers {volume: rate}
            slippage_k: Slippage sensitivity coefficient
            min_slippage_bps: Minimum slippage in bps
            max_slippage_bps: Maximum slippage cap in bps
            impact_enabled: Enable market impact modeling
            impact_coefficient: Square-root law coefficient
            adv_window: Window for average daily volume calculation
        """
        self.base_spread_bps = base_spread_bps
        self.offhours_spread_mult = offhours_spread_mult
        self.news_spread_mult = news_spread_mult

        self.commission_type = commission_type
        self.commission_rate = commission_rate
        self.commission_fixed = commission_fixed
        self.commission_tiers = commission_tiers or {}

        self.slippage_k = slippage_k
        self.min_slippage_bps = min_slippage_bps
        self.max_slippage_bps = max_slippage_bps

        self.impact_enabled = impact_enabled
        self.impact_coefficient = impact_coefficient
        self.adv_window = adv_window

        # Cache for ADV calculation
        self._adv_cache: Dict[str, float] = {}

    def calculate_spread(
        self,
        price: float,
        timestamp: datetime,
        is_news_period: bool = False,
    ) -> float:
        """
        Calculate bid-ask spread cost.

        Args:
            price: Current market price
            timestamp: Execution timestamp
            is_news_period: True if during high-impact news

        Returns:
            Spread cost in currency units
        """
        # Base spread in bps
        spread_bps = self.base_spread_bps

        # Adjust for off-hours (outside 8:00-17:00 UTC)
        if timestamp.time() < time(8, 0) or timestamp.time() > time(17, 0):
            spread_bps *= self.offhours_spread_mult

        # Adjust for news events
        if is_news_period:
            spread_bps *= self.news_spread_mult

        # Convert bps to currency
        spread_cost = price * (spread_bps / 10000.0)

        return spread_cost

    def calculate_commission(
        self,
        trade_value: float,
        cumulative_volume: float = 0.0,
    ) -> float:
        """
        Calculate broker commission.

        Args:
            trade_value: Absolute value of trade (price * quantity)
            cumulative_volume: Cumulative trading volume (for tiered pricing)

        Returns:
            Commission cost in currency units
        """
        if self.commission_type == "fixed":
            return self.commission_fixed

        # Check for tiered pricing
        commission_rate = self.commission_rate
        if self.commission_tiers:
            for volume_threshold, tier_rate in sorted(self.commission_tiers.items()):
                if cumulative_volume >= volume_threshold:
                    commission_rate = tier_rate

        return trade_value * commission_rate

    def calculate_slippage(
        self,
        price: float,
        order_size: float,
        avg_volume: float,
        volatility: float,
    ) -> float:
        """
        Calculate slippage cost.

        Slippage formula: slippage_bps = k * sqrt(order_size / avg_volume) * volatility

        Args:
            price: Execution price
            order_size: Size of order (in currency units)
            avg_volume: Average recent volume
            volatility: Recent price volatility (e.g., ATR or stddev)

        Returns:
            Slippage cost in currency units
        """
        if avg_volume <= 0 or order_size <= 0:
            slippage_bps = self.min_slippage_bps
        else:
            # Square-root scaling with order size
            size_ratio = order_size / avg_volume
            vol_factor = max(1.0, volatility)  # Normalize volatility

            slippage_bps = self.slippage_k * np.sqrt(size_ratio) * vol_factor * 100

        # Clamp to min/max
        slippage_bps = np.clip(slippage_bps, self.min_slippage_bps, self.max_slippage_bps)

        # Convert to currency
        slippage_cost = price * (slippage_bps / 10000.0)

        return slippage_cost

    def calculate_market_impact(
        self,
        price: float,
        order_size: float,
        avg_daily_volume: float,
    ) -> float:
        """
        Calculate market impact cost (permanent price movement).

        Uses square-root law: impact ∝ sqrt(order_size / ADV)

        Args:
            price: Current market price
            order_size: Size of order (in shares/lots)
            avg_daily_volume: Average daily volume

        Returns:
            Market impact cost in currency units
        """
        if not self.impact_enabled or avg_daily_volume <= 0:
            return 0.0

        # Participation rate (what % of ADV is our order)
        participation_rate = order_size / avg_daily_volume

        # Square-root law
        impact_bps = self.impact_coefficient * np.sqrt(participation_rate) * 10000

        # Convert to currency
        impact_cost = price * (impact_bps / 10000.0)

        return impact_cost

    def calculate_total_cost(
        self,
        entry_price: float,
        quantity: float,
        timestamp: datetime,
        is_news_period: bool = False,
        avg_volume: float = 1000000.0,
        volatility: float = 1.0,
        avg_daily_volume: float = 10000000.0,
        cumulative_volume: float = 0.0,
    ) -> TradeExecution:
        """
        Calculate all costs for a trade execution.

        Args:
            entry_price: Intended entry price
            quantity: Trade quantity (positive for buy, negative for sell)
            timestamp: Execution timestamp
            is_news_period: True if during news event
            avg_volume: Recent average volume
            volatility: Recent price volatility
            avg_daily_volume: Average daily volume for impact calculation
            cumulative_volume: Cumulative trading volume for tiered commission

        Returns:
            TradeExecution with detailed cost breakdown
        """
        trade_value = abs(entry_price * quantity)
        order_size_value = trade_value

        # 1. Spread cost (always half-spread for taker)
        spread_cost = self.calculate_spread(entry_price, timestamp, is_news_period) / 2.0

        # 2. Commission cost
        commission_cost = self.calculate_commission(trade_value, cumulative_volume)

        # 3. Slippage cost
        slippage_cost = self.calculate_slippage(
            entry_price, order_size_value, avg_volume, volatility
        )

        # 4. Market impact cost
        impact_cost = self.calculate_market_impact(
            entry_price, abs(quantity), avg_daily_volume
        )

        # Total cost
        total_cost = spread_cost + commission_cost + slippage_cost + impact_cost

        # Execution price (worse price due to costs)
        # For buy: execution price is higher
        # For sell: execution price is lower
        direction = 1 if quantity > 0 else -1
        execution_price = entry_price + (total_cost * direction)

        # Cost in basis points (relative to entry price)
        cost_bps = (total_cost / entry_price) * 10000 if entry_price > 0 else 0

        return TradeExecution(
            entry_price=entry_price,
            execution_price=execution_price,
            spread_cost=spread_cost,
            commission_cost=commission_cost,
            slippage_cost=slippage_cost,
            market_impact_cost=impact_cost,
            total_cost=total_cost,
            cost_bps=cost_bps,
        )

    def get_cost_summary(
        self,
        executions: list[TradeExecution],
    ) -> Dict[str, float]:
        """
        Get summary statistics for a list of executions.

        Args:
            executions: List of TradeExecution objects

        Returns:
            Dictionary with cost statistics
        """
        if not executions:
            return {
                "total_spread": 0.0,
                "total_commission": 0.0,
                "total_slippage": 0.0,
                "total_impact": 0.0,
                "total_cost": 0.0,
                "avg_cost_bps": 0.0,
                "num_trades": 0,
            }

        return {
            "total_spread": sum(e.spread_cost for e in executions),
            "total_commission": sum(e.commission_cost for e in executions),
            "total_slippage": sum(e.slippage_cost for e in executions),
            "total_impact": sum(e.market_impact_cost for e in executions),
            "total_cost": sum(e.total_cost for e in executions),
            "avg_cost_bps": np.mean([e.cost_bps for e in executions]),
            "num_trades": len(executions),
        }


# Preset cost models for common broker types
COST_PRESETS = {
    "retail_ecn": CostModel(
        base_spread_bps=0.5,  # Tight ECN spreads
        commission_type="percentage",
        commission_rate=0.00005,  # $0.50 per $10k = 0.5 bps
        slippage_k=0.05,  # Low slippage
        impact_enabled=False,  # Retail orders don't move market
    ),

    "retail_market_maker": CostModel(
        base_spread_bps=1.5,  # Wider spreads
        commission_type="fixed",
        commission_fixed=0.0,  # No commission but wider spread
        slippage_k=0.1,
        impact_enabled=False,
    ),

    "institutional": CostModel(
        base_spread_bps=0.3,  # Best pricing
        commission_type="percentage",
        commission_rate=0.00002,  # $0.20 per $10k = 0.2 bps
        commission_tiers={
            1000000: 0.00001,  # $0.10 per $10k above $1M volume
        },
        slippage_k=0.02,  # Very low slippage
        impact_enabled=True,  # Large orders move market
        impact_coefficient=0.3,
    ),

    "high_cost_broker": CostModel(
        base_spread_bps=3.0,  # Wide spreads
        commission_type="percentage",
        commission_rate=0.0005,  # 5 bps = expensive
        slippage_k=0.2,  # High slippage
        impact_enabled=False,
    ),
}


def get_cost_model(preset: str = "retail_ecn") -> CostModel:
    """
    Get a preset cost model.

    Args:
        preset: One of "retail_ecn", "retail_market_maker", "institutional", "high_cost_broker"

    Returns:
        CostModel instance
    """
    if preset not in COST_PRESETS:
        logger.warning(f"Unknown preset '{preset}', using 'retail_ecn'")
        preset = "retail_ecn"

    return COST_PRESETS[preset]
