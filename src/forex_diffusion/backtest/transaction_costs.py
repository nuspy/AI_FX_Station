"""
Standardized Transaction Cost Model

Provides unified transaction cost calculation across all backtest engines.
Fixes BUG-003 - inconsistent costs between engines.

Cost Components:
- Spread (bid-ask difference)
- Slippage (execution vs expected price)
- Commission (broker fees)
- Market impact (for large orders)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from loguru import logger


@dataclass
class TransactionCostModel:
    """
    Standardized transaction costs for backtesting.
    
    All costs should be measured from real broker data.
    Default values are conservative estimates for major forex pairs.
    """
    # Spread (bid-ask, varies by volatility)
    spread_pips_base: float = 1.0  # Base spread in pips (normal conditions)
    spread_multiplier_volatile: float = 2.0  # Multiply in high volatility
    
    # Commission (fixed per broker)
    commission_per_lot: float = 0.0  # Most forex brokers: zero commission
    commission_pct: float = 0.0  # Alternative: % of trade value
    
    # Slippage (varies by order type and size)
    slippage_pips_market: float = 0.5  # Market order slippage
    slippage_pips_limit: float = 0.0  # Limit order (no slippage if filled)
    slippage_multiplier_large: float = 1.5  # Multiply for large orders
    
    # Market impact (for very large orders)
    market_impact_threshold_lots: float = 10.0  # Impact starts above this
    market_impact_pips_per_lot: float = 0.1  # Additional cost per lot
    
    # Volatility threshold for spread widening
    volatility_threshold_atr_multiple: float = 1.5


class TransactionCostCalculator:
    """
    Calculate realistic transaction costs for backtesting.
    
    Usage:
        calculator = TransactionCostCalculator(model)
        cost = calculator.calculate_total_cost(
            order_type='market',
            size_lots=2.0,
            price=1.1000,
            volatility=0.0015,
            avg_volatility=0.0010
        )
    """
    
    def __init__(self, model: Optional[TransactionCostModel] = None):
        self.model = model or TransactionCostModel()
        logger.info(
            f"TransactionCostCalculator initialized: "
            f"spread={self.model.spread_pips_base} pips, "
            f"slippage_market={self.model.slippage_pips_market} pips"
        )
    
    def calculate_total_cost(
        self,
        order_type: str,  # 'market' or 'limit'
        size_lots: float,
        price: float,
        volatility: Optional[float] = None,
        avg_volatility: Optional[float] = None,
        pip_size: float = 0.0001
    ) -> Dict[str, float]:
        """
        Calculate total transaction cost.
        
        Args:
            order_type: 'market' or 'limit'
            size_lots: Position size in lots
            price: Entry/exit price
            volatility: Current volatility (ATR or similar)
            avg_volatility: Average volatility (for comparison)
            pip_size: Pip size for the instrument
            
        Returns:
            Dict with breakdown:
                - spread_pips: Spread cost in pips
                - slippage_pips: Slippage cost in pips
                - commission: Commission cost in currency
                - market_impact_pips: Market impact in pips
                - total_pips: Total cost in pips
                - total_currency: Total cost in currency
        """
        # 1. Calculate spread
        spread_pips = self._calculate_spread(volatility, avg_volatility)
        
        # 2. Calculate slippage
        slippage_pips = self._calculate_slippage(order_type, size_lots)
        
        # 3. Calculate commission
        commission = self._calculate_commission(size_lots, price)
        
        # 4. Calculate market impact
        market_impact_pips = self._calculate_market_impact(size_lots)
        
        # 5. Total costs
        total_pips = spread_pips + slippage_pips + market_impact_pips
        
        # Convert pips to currency
        pip_value_per_lot = pip_size * 100000  # Standard lot
        total_currency_from_pips = total_pips * pip_value_per_lot * size_lots
        total_currency = total_currency_from_pips + commission
        
        return {
            'spread_pips': spread_pips,
            'slippage_pips': slippage_pips,
            'commission': commission,
            'market_impact_pips': market_impact_pips,
            'total_pips': total_pips,
            'total_currency': total_currency,
            'cost_pct': (total_currency / (price * 100000 * size_lots)) * 100
        }
    
    def _calculate_spread(
        self,
        volatility: Optional[float],
        avg_volatility: Optional[float]
    ) -> float:
        """Calculate spread cost in pips"""
        spread = self.model.spread_pips_base
        
        # Widen spread in high volatility
        if volatility and avg_volatility:
            volatility_ratio = volatility / avg_volatility
            if volatility_ratio > self.model.volatility_threshold_atr_multiple:
                spread *= self.model.spread_multiplier_volatile
                logger.debug(
                    f"Widened spread due to high volatility: "
                    f"{self.model.spread_pips_base} → {spread} pips"
                )
        
        return spread
    
    def _calculate_slippage(self, order_type: str, size_lots: float) -> float:
        """Calculate slippage cost in pips"""
        if order_type == 'limit':
            # Limit orders don't have slippage (if filled)
            return self.model.slippage_pips_limit
        
        # Market orders
        slippage = self.model.slippage_pips_market
        
        # Increase slippage for large orders
        if size_lots > 5.0:  # Arbitrary threshold
            slippage *= self.model.slippage_multiplier_large
            logger.debug(
                f"Increased slippage for large order ({size_lots} lots): "
                f"{self.model.slippage_pips_market} → {slippage} pips"
            )
        
        return slippage
    
    def _calculate_commission(self, size_lots: float, price: float) -> float:
        """Calculate commission cost in currency"""
        # Per-lot commission
        commission = size_lots * self.model.commission_per_lot
        
        # Or percentage-based
        if self.model.commission_pct > 0:
            trade_value = price * 100000 * size_lots  # Standard lot
            commission += trade_value * self.model.commission_pct
        
        return commission
    
    def _calculate_market_impact(self, size_lots: float) -> float:
        """Calculate market impact cost in pips"""
        if size_lots <= self.model.market_impact_threshold_lots:
            return 0.0
        
        # Impact above threshold
        excess_lots = size_lots - self.model.market_impact_threshold_lots
        impact = excess_lots * self.model.market_impact_pips_per_lot
        
        logger.debug(
            f"Market impact for large order ({size_lots} lots): {impact} pips"
        )
        
        return impact
    
    def get_total_cost_pips(
        self,
        order_type: str,
        size_lots: float,
        price: float,
        **kwargs
    ) -> float:
        """Convenience method to get total cost in pips only"""
        result = self.calculate_total_cost(
            order_type=order_type,
            size_lots=size_lots,
            price=price,
            **kwargs
        )
        return result['total_pips']
    
    def get_total_cost_currency(
        self,
        order_type: str,
        size_lots: float,
        price: float,
        **kwargs
    ) -> float:
        """Convenience method to get total cost in currency only"""
        result = self.calculate_total_cost(
            order_type=order_type,
            size_lots=size_lots,
            price=price,
            **kwargs
        )
        return result['total_currency']


# Default models for common asset classes
FOREX_MAJOR_MODEL = TransactionCostModel(
    spread_pips_base=1.0,
    slippage_pips_market=0.5,
    commission_per_lot=0.0
)

FOREX_MINOR_MODEL = TransactionCostModel(
    spread_pips_base=2.0,
    slippage_pips_market=1.0,
    commission_per_lot=0.0
)

FOREX_EXOTIC_MODEL = TransactionCostModel(
    spread_pips_base=5.0,
    slippage_pips_market=2.0,
    commission_per_lot=0.0
)

# Factory function
def get_cost_model(asset_class: str = 'forex_major') -> TransactionCostModel:
    """
    Get standard cost model for asset class.
    
    Args:
        asset_class: 'forex_major', 'forex_minor', 'forex_exotic'
        
    Returns:
        TransactionCostModel
    """
    models = {
        'forex_major': FOREX_MAJOR_MODEL,
        'forex_minor': FOREX_MINOR_MODEL,
        'forex_exotic': FOREX_EXOTIC_MODEL
    }
    
    return models.get(asset_class, FOREX_MAJOR_MODEL)
