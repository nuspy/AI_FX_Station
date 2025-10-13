"""
Smart Execution Optimization

Optimizes order execution with advanced techniques:
- Smart timing (avoid spread widening)
- Slippage modeling
- Market impact estimation
- Order splitting
- TWAP/VWAP strategies

Based on Citadel/Two Sigma execution practices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings


class ExecutionStrategy(Enum):
    """Order execution strategies."""
    MARKET = "market"  # Immediate execution
    LIMIT = "limit"  # Limit order
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price
    ADAPTIVE = "adaptive"  # Adaptive strategy


@dataclass
class ExecutionCost:
    """Breakdown of execution costs."""
    spread_cost: float  # Bid-ask spread
    slippage_cost: float  # Price movement during execution
    market_impact: float  # Permanent price impact
    total_cost: float  # Total execution cost
    execution_price: float  # Final execution price


class SmartExecutionOptimizer:
    """
    Smart execution optimizer for minimizing transaction costs.

    Features:
    - Spread modeling (time-of-day, volatility)
    - Slippage estimation
    - Market impact (order size vs liquidity)
    - Optimal timing (avoid spread widening)
    - Order splitting strategies

    Example:
        >>> optimizer = SmartExecutionOptimizer()
        >>> cost = optimizer.estimate_execution_cost(
        ...     order_size=10000,
        ...     current_price=1.1000,
        ...     current_spread=0.0002,
        ...     volatility=0.001
        ... )
        >>> print(f"Total cost: ${cost.total_cost:.2f}")
    """

    def __init__(
        self,
        base_spread_bps: float = 1.0,  # Base spread in basis points
        volatility_multiplier: float = 0.5,  # Spread widens with volatility
        market_impact_coef: float = 0.1,  # Market impact coefficient
        slippage_factor: float = 0.3  # Slippage as fraction of spread
    ):
        """
        Initialize smart execution optimizer.

        Args:
            base_spread_bps: Base bid-ask spread in basis points (default: 1.0)
            volatility_multiplier: Spread widening multiplier (default: 0.5)
            market_impact_coef: Market impact coefficient (default: 0.1)
            slippage_factor: Slippage as fraction of spread (default: 0.3)
        """
        self.base_spread_bps = base_spread_bps
        self.volatility_multiplier = volatility_multiplier
        self.market_impact_coef = market_impact_coef
        self.slippage_factor = slippage_factor

        # Time-of-day spread multipliers (based on market sessions)
        self.tod_multipliers = self._initialize_tod_multipliers()

    def _initialize_tod_multipliers(self) -> Dict[int, float]:
        """
        Initialize time-of-day spread multipliers.

        Returns:
            Dict mapping hour to spread multiplier
        """
        multipliers = {}

        # Hour 0-6: Asian session (wider spreads)
        for hour in range(0, 7):
            multipliers[hour] = 1.3

        # Hour 7-9: Transition (medium spreads)
        for hour in range(7, 10):
            multipliers[hour] = 1.1

        # Hour 10-16: London/NY overlap (tighter spreads)
        for hour in range(10, 17):
            multipliers[hour] = 0.8

        # Hour 17-20: NY session (normal spreads)
        for hour in range(17, 21):
            multipliers[hour] = 1.0

        # Hour 21-23: Transition to Asian (wider spreads)
        for hour in range(21, 24):
            multipliers[hour] = 1.2

        return multipliers

    def estimate_execution_cost(
        self,
        order_size: float,
        current_price: float,
        direction: str,  # 'buy' or 'sell'
        current_spread: Optional[float] = None,
        volatility: Optional[float] = None,
        current_hour: Optional[int] = None,
        average_volume: Optional[float] = None,
        dom_snapshot: Optional[Dict] = None
    ) -> ExecutionCost:
        """
        Estimate total execution cost for an order.

        Args:
            order_size: Size of order (in base currency)
            current_price: Current market price
            direction: 'buy' or 'sell'
            current_spread: Current bid-ask spread (optional, overridden by DOM)
            volatility: Current volatility (optional)
            current_hour: Hour of day (0-23, optional)
            average_volume: Recent average volume (optional)
            dom_snapshot: Real-time order book snapshot (optional) with keys:
                - bids: [[price, volume], ...] (sorted descending)
                - asks: [[price, volume], ...] (sorted ascending)
                - spread: float
                - depth: Dict with bid_depth, ask_depth

        Returns:
            ExecutionCost breakdown
        """
        # Check if DOM data is available for enhanced estimation
        use_dom = dom_snapshot is not None and 'bids' in dom_snapshot and 'asks' in dom_snapshot

        # 1. Estimate spread cost (prefer real DOM spread)
        if use_dom and 'spread' in dom_snapshot:
            spread = dom_snapshot['spread']
        else:
            spread = self._estimate_spread(
                current_price,
                current_spread,
                volatility,
                current_hour
            )

        spread_cost = spread * order_size / 2  # Half-spread crossing

        # 2. Estimate slippage (use DOM-based calculation if available)
        if use_dom:
            slippage = self._calculate_dom_slippage(
                order_size,
                direction,
                dom_snapshot['bids'],
                dom_snapshot['asks']
            )
        else:
            slippage = self._estimate_slippage(
                order_size,
                current_price,
                volatility,
                average_volume
            )

        slippage_cost = slippage * order_size

        # 3. Estimate market impact (use liquidity-based if DOM available)
        if use_dom:
            impact = self._calculate_liquidity_based_impact(
                order_size,
                current_price,
                dom_snapshot
            )
        else:
            impact = self._estimate_market_impact(
                order_size,
                current_price,
                average_volume
            )

        market_impact_cost = impact * order_size

        # 4. Total cost
        total_cost = spread_cost + slippage_cost + market_impact_cost

        # 5. Execution price
        if direction == 'buy':
            execution_price = current_price + (total_cost / order_size)
        else:  # sell
            execution_price = current_price - (total_cost / order_size)

        return ExecutionCost(
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            market_impact=market_impact_cost,
            total_cost=total_cost,
            execution_price=execution_price
        )

    def _estimate_spread(
        self,
        price: float,
        current_spread: Optional[float],
        volatility: Optional[float],
        current_hour: Optional[int]
    ) -> float:
        """
        Estimate bid-ask spread.

        Args:
            price: Current price
            current_spread: Observed spread (if available)
            volatility: Current volatility
            current_hour: Hour of day

        Returns:
            Estimated spread
        """
        if current_spread is not None:
            return current_spread

        # Base spread in price units
        base_spread = price * (self.base_spread_bps / 10000)

        # Adjust for volatility
        if volatility is not None:
            vol_adjustment = 1.0 + (volatility * self.volatility_multiplier)
        else:
            vol_adjustment = 1.0

        # Adjust for time of day
        if current_hour is not None:
            tod_adjustment = self.tod_multipliers.get(current_hour, 1.0)
        else:
            tod_adjustment = 1.0

        estimated_spread = base_spread * vol_adjustment * tod_adjustment

        return estimated_spread

    def _estimate_slippage(
        self,
        order_size: float,
        price: float,
        volatility: Optional[float],
        average_volume: Optional[float]
    ) -> float:
        """
        Estimate slippage per unit.

        Args:
            order_size: Order size
            price: Current price
            volatility: Current volatility
            average_volume: Average volume

        Returns:
            Slippage per unit
        """
        # Base slippage (function of spread)
        base_spread = price * (self.base_spread_bps / 10000)
        base_slippage = base_spread * self.slippage_factor

        # Adjust for order size relative to volume
        if average_volume is not None and average_volume > 0:
            # Larger orders relative to volume = more slippage
            size_ratio = order_size / average_volume
            size_multiplier = 1.0 + (size_ratio * 0.5)
        else:
            size_multiplier = 1.0

        # Adjust for volatility
        if volatility is not None:
            vol_multiplier = 1.0 + volatility
        else:
            vol_multiplier = 1.0

        estimated_slippage = base_slippage * size_multiplier * vol_multiplier

        return estimated_slippage

    def _estimate_market_impact(
        self,
        order_size: float,
        price: float,
        average_volume: Optional[float]
    ) -> float:
        """
        Estimate permanent market impact.

        Args:
            order_size: Order size
            price: Current price
            average_volume: Average volume

        Returns:
            Market impact per unit
        """
        if average_volume is None or average_volume == 0:
            # Conservative estimate if no volume data
            return price * (self.market_impact_coef / 10000)

        # Square root impact model
        # Impact ∝ sqrt(order_size / average_volume)
        size_ratio = order_size / average_volume
        impact_multiplier = np.sqrt(size_ratio)

        # Base impact
        base_impact = price * (self.market_impact_coef / 10000)

        market_impact = base_impact * impact_multiplier

        return market_impact

    def _calculate_dom_slippage(
        self,
        order_size: float,
        direction: str,
        bids: List[List[float]],
        asks: List[List[float]]
    ) -> float:
        """
        Calculate slippage by walking through order book levels.

        Args:
            order_size: Size of order
            direction: 'buy' or 'sell'
            bids: List of [price, volume] for bids (descending)
            asks: List of [price, volume] for asks (ascending)

        Returns:
            Slippage per unit (difference from best price)
        """
        if not bids or not asks:
            # Fallback if no DOM data
            return 0.0

        # Get best bid/ask
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0

        if direction == 'buy':
            # Buying: walk up the ask side
            levels = asks
            reference_price = best_ask
        else:
            # Selling: walk down the bid side
            levels = bids
            reference_price = best_bid

        # Walk through levels and fill order
        remaining_size = order_size
        total_cost = 0.0

        for price, volume in levels:
            if remaining_size <= 0:
                break

            fill_size = min(remaining_size, volume)
            total_cost += fill_size * price
            remaining_size -= fill_size

        # If order couldn't be fully filled from available depth
        if remaining_size > 0:
            # Use last level price plus penalty
            last_price = levels[-1][0] if levels else reference_price
            penalty = abs(last_price - reference_price) * 0.5  # 50% penalty
            total_cost += remaining_size * (last_price + penalty)

        # Calculate average fill price
        avg_fill_price = total_cost / order_size

        # Slippage is difference from best price
        slippage_per_unit = abs(avg_fill_price - reference_price)

        return slippage_per_unit

    def _calculate_liquidity_based_impact(
        self,
        order_size: float,
        current_price: float,
        dom_snapshot: Dict
    ) -> float:
        """
        Calculate market impact based on available liquidity in order book.

        Args:
            order_size: Size of order
            current_price: Current market price
            dom_snapshot: Order book snapshot

        Returns:
            Market impact per unit
        """
        bids = dom_snapshot.get('bids', [])
        asks = dom_snapshot.get('asks', [])

        if not bids or not asks:
            # Fallback to statistical model
            return current_price * (self.market_impact_coef / 10000)

        # Calculate total depth (top 10 levels)
        max_levels = min(10, len(bids), len(asks))
        bid_depth = sum(vol for _, vol in bids[:max_levels])
        ask_depth = sum(vol for _, vol in asks[:max_levels])
        total_depth = bid_depth + ask_depth

        if total_depth == 0:
            # No liquidity, high impact
            return current_price * (self.market_impact_coef * 5 / 10000)

        # Calculate impact ratio: order size relative to available liquidity
        impact_ratio = order_size / total_depth

        # Impact increases non-linearly with ratio
        if impact_ratio < 0.1:
            # Small order, minimal impact
            impact_multiplier = impact_ratio
        elif impact_ratio < 0.5:
            # Medium order, moderate impact
            impact_multiplier = 0.1 + (impact_ratio - 0.1) * 2.0
        else:
            # Large order, high impact
            impact_multiplier = 0.9 + (impact_ratio - 0.5) * 3.0

        # Base impact scaled by liquidity ratio
        base_impact = current_price * (self.market_impact_coef / 10000)
        liquidity_impact = base_impact * impact_multiplier

        return liquidity_impact

    def check_high_impact_order(
        self,
        order_size: float,
        current_price: float,
        dom_snapshot: Optional[Dict] = None
    ) -> Dict:
        """
        Check if order would have high market impact.

        Args:
            order_size: Size of order
            current_price: Current price
            dom_snapshot: Order book snapshot (optional)

        Returns:
            Dict with:
                - high_impact: bool
                - impact_ratio: float (order size / depth)
                - recommendation: str
                - suggested_split: int (number of slices)
        """
        if not dom_snapshot or 'bids' not in dom_snapshot:
            # Can't determine without DOM data
            return {
                'high_impact': False,
                'impact_ratio': 0.0,
                'recommendation': 'Execute normally (no DOM data)',
                'suggested_split': 1
            }

        bids = dom_snapshot.get('bids', [])
        asks = dom_snapshot.get('asks', [])

        # Calculate top-5 depth
        top5_bid_depth = sum(vol for _, vol in bids[:5]) if len(bids) >= 5 else sum(vol for _, vol in bids)
        top5_ask_depth = sum(vol for _, vol in asks[:5]) if len(asks) >= 5 else sum(vol for _, vol in asks)
        top5_depth = top5_bid_depth + top5_ask_depth

        if top5_depth == 0:
            return {
                'high_impact': True,
                'impact_ratio': float('inf'),
                'recommendation': 'CRITICAL: No liquidity available',
                'suggested_split': 10
            }

        impact_ratio = order_size / top5_depth

        # Thresholds from specs
        if impact_ratio > 0.5:
            return {
                'high_impact': True,
                'impact_ratio': impact_ratio,
                'recommendation': 'SEVERE impact (>50% of depth). Reject or heavily reduce size.',
                'suggested_split': 10
            }
        elif impact_ratio > 0.2:
            return {
                'high_impact': True,
                'impact_ratio': impact_ratio,
                'recommendation': 'HIGH impact (20-50% of depth). Consider splitting order.',
                'suggested_split': 5
            }
        elif impact_ratio > 0.1:
            return {
                'high_impact': False,
                'impact_ratio': impact_ratio,
                'recommendation': 'MODERATE impact (10-20% of depth). Acceptable with caution.',
                'suggested_split': 3
            }
        else:
            return {
                'high_impact': False,
                'impact_ratio': impact_ratio,
                'recommendation': 'LOW impact (<10% of depth). Execute normally.',
                'suggested_split': 1
            }

    def optimize_execution_timing(
        self,
        order_size: float,
        current_price: float,
        direction: str,
        forecast_hours: int = 24
    ) -> Dict:
        """
        Find optimal execution time within forecast window.

        Args:
            order_size: Order size
            current_price: Current price
            direction: 'buy' or 'sell'
            forecast_hours: Hours to forecast (default: 24)

        Returns:
            Dict with optimal_hour, estimated_cost, reasoning
        """
        # Simulate costs for each hour
        hour_costs = []

        for hour_offset in range(forecast_hours):
            hour = (pd.Timestamp.now().hour + hour_offset) % 24

            # Estimate cost for this hour
            cost = self.estimate_execution_cost(
                order_size=order_size,
                current_price=current_price,
                direction=direction,
                current_hour=hour
            )

            hour_costs.append({
                'hour_offset': hour_offset,
                'hour': hour,
                'total_cost': cost.total_cost,
                'spread_cost': cost.spread_cost
            })

        # Find optimal (minimum cost)
        optimal = min(hour_costs, key=lambda x: x['total_cost'])

        # Reasoning
        tod_name = self._get_session_name(optimal['hour'])

        reasoning = (
            f"Optimal execution in {optimal['hour_offset']} hours "
            f"(hour {optimal['hour']}, {tod_name} session). "
            f"Estimated savings: ${hour_costs[0]['total_cost'] - optimal['total_cost']:.2f} "
            f"vs immediate execution."
        )

        return {
            'optimal_hour_offset': optimal['hour_offset'],
            'optimal_hour': optimal['hour'],
            'estimated_cost': optimal['total_cost'],
            'immediate_cost': hour_costs[0]['total_cost'],
            'savings': hour_costs[0]['total_cost'] - optimal['total_cost'],
            'reasoning': reasoning
        }

    def _get_session_name(self, hour: int) -> str:
        """Get market session name for hour."""
        if 0 <= hour < 7:
            return "Asian"
        elif 7 <= hour < 10:
            return "Transition (Asian-London)"
        elif 10 <= hour < 17:
            return "London-NY Overlap"
        elif 17 <= hour < 21:
            return "NY"
        else:
            return "Transition (NY-Asian)"

    def split_order_twap(
        self,
        total_size: float,
        duration_minutes: int,
        num_slices: int = 10
    ) -> List[Dict]:
        """
        Split order using Time-Weighted Average Price strategy.

        Args:
            total_size: Total order size
            duration_minutes: Execution duration in minutes
            num_slices: Number of slices (default: 10)

        Returns:
            List of order slices with timing
        """
        slice_size = total_size / num_slices
        interval_minutes = duration_minutes / num_slices

        slices = []
        for i in range(num_slices):
            slices.append({
                'slice_number': i + 1,
                'size': slice_size,
                'delay_minutes': i * interval_minutes,
                'strategy': 'TWAP'
            })

        return slices

    def get_execution_recommendation(
        self,
        order_size: float,
        current_price: float,
        direction: str,
        urgency: str = 'normal'  # 'immediate', 'normal', 'patient'
    ) -> Dict:
        """
        Get execution recommendation based on order characteristics.

        Args:
            order_size: Order size
            current_price: Current price
            direction: 'buy' or 'sell'
            urgency: Execution urgency level

        Returns:
            Execution recommendation
        """
        # Estimate immediate execution cost
        immediate_cost = self.estimate_execution_cost(
            order_size=order_size,
            current_price=current_price,
            direction=direction
        )

        # Determine strategy based on urgency and size
        if urgency == 'immediate':
            strategy = ExecutionStrategy.MARKET
            reasoning = "Immediate execution required"
            slices = 1

        elif urgency == 'patient':
            # Optimize timing
            timing = self.optimize_execution_timing(
                order_size=order_size,
                current_price=current_price,
                direction=direction
            )

            if timing['savings'] > immediate_cost.total_cost * 0.1:  # >10% savings
                strategy = ExecutionStrategy.ADAPTIVE
                reasoning = f"Wait for optimal timing. {timing['reasoning']}"
                slices = 5
            else:
                strategy = ExecutionStrategy.TWAP
                reasoning = "TWAP execution to minimize impact"
                slices = 10

        else:  # normal
            # Use TWAP for large orders
            if order_size > 10000:  # Configurable threshold
                strategy = ExecutionStrategy.TWAP
                reasoning = "Large order - use TWAP to minimize impact"
                slices = 10
            else:
                strategy = ExecutionStrategy.MARKET
                reasoning = "Normal size - market execution acceptable"
                slices = 1

        return {
            'strategy': strategy.value,
            'recommended_slices': slices,
            'estimated_cost': immediate_cost.total_cost,
            'estimated_price': immediate_cost.execution_price,
            'reasoning': reasoning
        }
