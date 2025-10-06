"""
Regime-Aware Position Sizing

Adaptive position sizing based on market regime, volatility, and pattern confidence.
Implements Risk Parity and Kelly Criterion principles.

Based on AQR/Two Sigma risk management practices.
"""
from enum import Enum
from typing import Optional, Dict
import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT_PREPARATION = "breakout_preparation"


class RegimePositionSizer:
    """
    Regime-aware position sizing using risk parity principles.

    Size inversely proportional to volatility, adjusted for regime.
    Based on AQR/Two Sigma risk management.

    Features:
    - Regime-based size multipliers
    - Confidence-based adjustments
    - Risk Parity (inverse volatility weighting)
    - Kelly Criterion with fractional sizing
    - Maximum risk caps

    Example:
        >>> sizer = RegimePositionSizer(
        ...     base_risk_per_trade_pct=1.0,
        ...     use_kelly_criterion=True
        ... )
        >>> result = sizer.calculate_position_size(
        ...     account_balance=10000,
        ...     entry_price=1.1000,
        ...     stop_loss_price=1.0970,
        ...     current_regime=MarketRegime.TRENDING_UP,
        ...     pattern_confidence=0.75
        ... )
        >>> print(f"Position size: {result['position_size']:.2f}")
    """

    def __init__(
        self,
        base_risk_per_trade_pct: float = 1.0,
        max_risk_per_trade_pct: float = 2.0,
        volatility_lookback: int = 20,
        use_kelly_criterion: bool = True,
        kelly_fraction: float = 0.25  # Quarter-Kelly for safety
    ):
        """
        Initialize regime-aware position sizer.

        Args:
            base_risk_per_trade_pct: Base risk % of account per trade (default: 1.0)
            max_risk_per_trade_pct: Maximum risk % allowed (default: 2.0)
            volatility_lookback: Periods for volatility calculation (default: 20)
            use_kelly_criterion: Use Kelly for sizing (default: True)
            kelly_fraction: Fraction of Kelly to use (default: 0.25 - Quarter Kelly)
        """
        self.base_risk_pct = base_risk_per_trade_pct
        self.max_risk_pct = max_risk_per_trade_pct
        self.volatility_lookback = volatility_lookback
        self.use_kelly_criterion = use_kelly_criterion
        self.kelly_fraction = kelly_fraction

        # Regime multipliers (empirically derived from hedge fund research)
        self.regime_multipliers = {
            MarketRegime.TRENDING_UP: 1.2,  # Increase size in uptrends
            MarketRegime.TRENDING_DOWN: 1.0,  # Normal size in downtrends
            MarketRegime.RANGING: 0.7,  # Reduce size in ranges (choppy)
            MarketRegime.VOLATILE: 0.5,  # Significantly reduce in high vol
            MarketRegime.BREAKOUT_PREPARATION: 0.8  # Moderate size before breakouts
        }

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        current_regime: MarketRegime,
        pattern_confidence: float,
        recent_returns: Optional[pd.Series] = None,
        pattern_win_rate: Optional[float] = None
    ) -> Dict:
        """
        Calculate optimal position size based on regime and risk parameters.

        Args:
            account_balance: Current account balance
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            current_regime: Current market regime
            pattern_confidence: Pattern confidence score (0-1)
            recent_returns: Recent returns for volatility calculation
            pattern_win_rate: Historical win rate for this pattern (0-1)

        Returns:
            Dict with:
                - position_size: Calculated position size
                - risk_amount: Dollar risk amount
                - risk_pct: Risk as percentage of account
                - reasoning: Dict with all multipliers for transparency
        """
        # 1. Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)

        if risk_per_unit == 0:
            return {
                'position_size': 0,
                'risk_amount': 0,
                'risk_pct': 0,
                'reasoning': {'error': 'Zero risk per unit - invalid stop loss'}
            }

        # 2. Base risk amount (% of account)
        base_risk_amount = account_balance * (self.base_risk_pct / 100)

        # 3. Adjust for regime
        regime_multiplier = self.regime_multipliers.get(current_regime, 1.0)
        regime_adjusted_risk = base_risk_amount * regime_multiplier

        # 4. Adjust for confidence
        # Higher confidence = larger size (but not linear to avoid over-leverage)
        confidence_multiplier = 0.5 + (pattern_confidence * 0.5)  # Range: 0.5 to 1.0
        confidence_adjusted_risk = regime_adjusted_risk * confidence_multiplier

        # 5. Adjust for volatility (Risk Parity)
        # Inverse volatility weighting: reduce size when volatility is high
        volatility_multiplier = 1.0
        if recent_returns is not None and len(recent_returns) >= self.volatility_lookback:
            current_volatility = recent_returns.std()
            avg_volatility = recent_returns.rolling(window=self.volatility_lookback).std().mean()

            if current_volatility > 0 and avg_volatility > 0:
                # Inverse volatility weighting
                volatility_multiplier = avg_volatility / current_volatility
                # Cap multiplier to prevent extreme sizes
                volatility_multiplier = np.clip(volatility_multiplier, 0.5, 2.0)

        volatility_adjusted_risk = confidence_adjusted_risk * volatility_multiplier

        # 6. Kelly Criterion adjustment (if enabled)
        kelly_multiplier = 1.0
        if self.use_kelly_criterion and pattern_win_rate is not None:
            kelly_size = self._calculate_kelly_size(
                win_rate=pattern_win_rate,
                avg_win=2.0,  # Assume 2:1 reward:risk (conservative)
                avg_loss=1.0
            )
            # Apply Kelly fraction for safety (full Kelly is too aggressive)
            kelly_multiplier = kelly_size * self.kelly_fraction
            kelly_multiplier = np.clip(kelly_multiplier, 0.1, 1.5)

        final_risk_amount = volatility_adjusted_risk * kelly_multiplier

        # 7. Cap at maximum risk
        max_risk_amount = account_balance * (self.max_risk_pct / 100)
        final_risk_amount = min(final_risk_amount, max_risk_amount)

        # 8. Calculate position size
        position_size = final_risk_amount / risk_per_unit

        # 9. Calculate final risk percentage
        final_risk_pct = (final_risk_amount / account_balance) * 100

        # Reasoning for logging/debugging/transparency
        reasoning = {
            'base_risk_amount': base_risk_amount,
            'regime': current_regime.value,
            'regime_multiplier': regime_multiplier,
            'confidence': pattern_confidence,
            'confidence_multiplier': confidence_multiplier,
            'volatility_multiplier': volatility_multiplier,
            'kelly_multiplier': kelly_multiplier,
            'final_risk_amount': final_risk_amount,
            'risk_per_unit': risk_per_unit,
            'capped_at_max': final_risk_amount >= max_risk_amount
        }

        return {
            'position_size': position_size,
            'risk_amount': final_risk_amount,
            'risk_pct': final_risk_pct,
            'reasoning': reasoning
        }

    def _calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion position size.

        Kelly % = (p * b - q) / b

        where:
            p = win probability
            q = loss probability (1 - p)
            b = ratio of average win to average loss

        Args:
            win_rate: Win probability (0-1)
            avg_win: Average win size
            avg_loss: Average loss size

        Returns:
            Kelly percentage (0-1)
        """
        if avg_loss == 0:
            return 0.0

        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss

        kelly_pct = (p * b - q) / b

        # Kelly can be negative (don't trade) or >1 (unrealistic)
        kelly_pct = np.clip(kelly_pct, 0.0, 1.0)

        return kelly_pct

    def get_regime_recommendation(self, current_regime: MarketRegime) -> str:
        """
        Get human-readable recommendation for current regime.

        Args:
            current_regime: Current market regime

        Returns:
            Recommendation string
        """
        multiplier = self.regime_multipliers.get(current_regime, 1.0)

        if multiplier >= 1.2:
            return f"Favorable conditions ({current_regime.value}). Increase position size (+20%)."
        elif multiplier >= 1.0:
            return f"Normal conditions ({current_regime.value}). Standard position size."
        elif multiplier >= 0.7:
            return f"Cautious conditions ({current_regime.value}). Reduce position size (-30%)."
        else:
            return f"Unfavorable conditions ({current_regime.value}). Significantly reduce size (-50%)."

    def calculate_batch_sizes(
        self,
        account_balance: float,
        trades: list[Dict]
    ) -> list[Dict]:
        """
        Calculate position sizes for multiple trades.

        Useful for portfolio-level position sizing.

        Args:
            account_balance: Current account balance
            trades: List of trade dicts with required fields

        Returns:
            List of sizing results
        """
        results = []

        for trade in trades:
            sizing = self.calculate_position_size(
                account_balance=account_balance,
                entry_price=trade['entry_price'],
                stop_loss_price=trade['stop_loss_price'],
                current_regime=trade['regime'],
                pattern_confidence=trade['confidence'],
                recent_returns=trade.get('recent_returns'),
                pattern_win_rate=trade.get('win_rate')
            )

            results.append({
                **trade,
                **sizing
            })

        return results

    def optimize_regime_multipliers(
        self,
        historical_trades: pd.DataFrame,
        regime_column: str = 'regime',
        return_column: str = 'return'
    ) -> Dict[MarketRegime, float]:
        """
        Optimize regime multipliers based on historical performance.

        Analyzes which regimes had better Sharpe ratios and adjusts multipliers.

        Args:
            historical_trades: DataFrame with historical trades
            regime_column: Column name for regime
            return_column: Column name for returns

        Returns:
            Optimized regime multipliers
        """
        optimized = {}

        for regime in MarketRegime:
            regime_trades = historical_trades[
                historical_trades[regime_column] == regime.value
            ]

            if len(regime_trades) > 0:
                # Calculate Sharpe ratio for this regime
                returns = regime_trades[return_column]
                sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0

                # Base multiplier
                base_multiplier = self.regime_multipliers.get(regime, 1.0)

                # Adjust based on Sharpe
                # If Sharpe > 1.0 (good), increase multiplier
                # If Sharpe < 0.5 (poor), decrease multiplier
                if sharpe > 1.0:
                    optimized[regime] = min(base_multiplier * 1.2, 1.5)
                elif sharpe < 0.5:
                    optimized[regime] = max(base_multiplier * 0.8, 0.3)
                else:
                    optimized[regime] = base_multiplier
            else:
                optimized[regime] = self.regime_multipliers.get(regime, 1.0)

        return optimized
