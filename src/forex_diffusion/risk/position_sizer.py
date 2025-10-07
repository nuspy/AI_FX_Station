"""
Position Sizing Component

Calculates optimal position sizes using multiple methods:
- Fixed Fractional (% of capital at risk)
- Kelly Criterion (mathematically optimal from backtest data)
- Optimal f (Ralph Vince's method)
- Volatility-adjusted sizing
- Drawdown protection logic

Integrates with RiskProfile configuration and backtesting results.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class BacktestTradeHistory:
    """Historical trades from backtesting for Kelly/Optimal f calculation."""
    wins: List[float]  # Win amounts (as fraction of capital)
    losses: List[float]  # Loss amounts (as positive fractions)
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_consecutive_losses: int


class PositionSizer:
    """
    Calculates position sizes using multiple methods with safety constraints.

    Methods:
    - fixed_fractional: Risk fixed % per trade
    - kelly: Kelly Criterion (from backtest stats)
    - optimal_f: Ralph Vince's Optimal f
    - volatility_adjusted: Adjust for ATR/volatility

    Safety features:
    - Maximum position size cap
    - Minimum position size floor
    - Drawdown protection (reduce size in drawdown)
    - Correlation-based diversification
    - Pre-trade validation
    """

    def __init__(
        self,
        base_risk_pct: float = 1.0,
        kelly_fraction: float = 0.25,  # Quarter Kelly (safer)
        max_position_size_pct: float = 5.0,
        min_position_size_pct: float = 0.1,
        max_total_exposure_pct: float = 20.0,
        drawdown_reduction_enabled: bool = True,
        drawdown_threshold_pct: float = 10.0,
        drawdown_size_multiplier: float = 0.5,
    ):
        """
        Initialize position sizer.

        Args:
            base_risk_pct: Base risk per trade (% of capital)
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
            max_position_size_pct: Maximum position size (% of capital)
            min_position_size_pct: Minimum position size (% of capital)
            max_total_exposure_pct: Maximum total portfolio exposure
            drawdown_reduction_enabled: Enable drawdown protection
            drawdown_threshold_pct: Drawdown % to trigger size reduction
            drawdown_size_multiplier: Multiplier when in drawdown (0.5 = half size)
        """
        self.base_risk_pct = base_risk_pct
        self.kelly_fraction = kelly_fraction
        self.max_position_size_pct = max_position_size_pct
        self.min_position_size_pct = min_position_size_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.drawdown_reduction_enabled = drawdown_reduction_enabled
        self.drawdown_threshold_pct = drawdown_threshold_pct
        self.drawdown_size_multiplier = drawdown_size_multiplier

        logger.info(
            f"PositionSizer initialized: base_risk={base_risk_pct}%, "
            f"kelly_fraction={kelly_fraction}, max_size={max_position_size_pct}%"
        )

    def calculate_position_size(
        self,
        method: str,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        backtest_history: Optional[BacktestTradeHistory] = None,
        atr: Optional[float] = None,
        current_drawdown_pct: float = 0.0,
        existing_exposure_pct: float = 0.0,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate position size using specified method.

        Args:
            method: 'fixed_fractional', 'kelly', 'optimal_f', 'volatility_adjusted'
            account_balance: Current account balance
            entry_price: Entry price
            stop_loss_price: Stop loss price
            backtest_history: Historical trades (required for kelly/optimal_f)
            atr: Average True Range (required for volatility_adjusted)
            current_drawdown_pct: Current drawdown percentage
            existing_exposure_pct: Current total portfolio exposure

        Returns:
            Tuple of (position_size, metadata)
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            logger.error("Risk per share is zero, cannot calculate position size")
            return 0.0, {'error': 'zero_risk'}

        # Calculate base size based on method
        if method == 'fixed_fractional':
            size = self._calculate_fixed_fractional(
                account_balance, risk_per_share
            )
        elif method == 'kelly':
            if not backtest_history:
                logger.warning("Kelly requires backtest history, falling back to fixed fractional")
                size = self._calculate_fixed_fractional(account_balance, risk_per_share)
            else:
                size = self._calculate_kelly(
                    account_balance, risk_per_share, backtest_history
                )
        elif method == 'optimal_f':
            if not backtest_history:
                logger.warning("Optimal f requires backtest history, falling back to fixed fractional")
                size = self._calculate_fixed_fractional(account_balance, risk_per_share)
            else:
                size = self._calculate_optimal_f(
                    account_balance, risk_per_share, backtest_history
                )
        elif method == 'volatility_adjusted':
            if atr is None:
                logger.warning("Volatility adjusted requires ATR, falling back to fixed fractional")
                size = self._calculate_fixed_fractional(account_balance, risk_per_share)
            else:
                size = self._calculate_volatility_adjusted(
                    account_balance, risk_per_share, entry_price, atr
                )
        else:
            logger.error(f"Unknown method: {method}")
            size = self._calculate_fixed_fractional(account_balance, risk_per_share)

        # Apply drawdown protection
        if self.drawdown_reduction_enabled and current_drawdown_pct > self.drawdown_threshold_pct:
            original_size = size
            size *= self.drawdown_size_multiplier
            logger.warning(
                f"Drawdown protection: reducing size by {(1-self.drawdown_size_multiplier)*100:.0f}% "
                f"(drawdown={current_drawdown_pct:.1f}%): {original_size:.2f} -> {size:.2f}"
            )

        # Apply size constraints
        size_pct = (size * entry_price / account_balance) * 100
        constrained_size = size
        constraint_applied = None

        if size_pct > self.max_position_size_pct:
            constrained_size = (self.max_position_size_pct / 100) * account_balance / entry_price
            constraint_applied = f'max_size ({self.max_position_size_pct}%)'
        elif size_pct < self.min_position_size_pct:
            constrained_size = (self.min_position_size_pct / 100) * account_balance / entry_price
            constraint_applied = f'min_size ({self.min_position_size_pct}%)'

        # Check total exposure
        new_exposure_pct = existing_exposure_pct + (constrained_size * entry_price / account_balance * 100)
        if new_exposure_pct > self.max_total_exposure_pct:
            # Reduce size to fit within exposure limit
            available_exposure = self.max_total_exposure_pct - existing_exposure_pct
            if available_exposure > 0:
                constrained_size = (available_exposure / 100) * account_balance / entry_price
                constraint_applied = f'max_exposure ({self.max_total_exposure_pct}%)'
            else:
                constrained_size = 0.0
                constraint_applied = 'max_exposure_exceeded'

        # Metadata
        metadata = {
            'method': method,
            'calculated_size': size,
            'final_size': constrained_size,
            'size_pct': (constrained_size * entry_price / account_balance) * 100,
            'risk_per_share': risk_per_share,
            'constraint_applied': constraint_applied,
            'drawdown_protection_active': (
                self.drawdown_reduction_enabled
                and current_drawdown_pct > self.drawdown_threshold_pct
            ),
        }

        logger.info(
            f"Position size calculated: method={method}, "
            f"size={constrained_size:.2f} ({metadata['size_pct']:.2f}%), "
            f"constraint={constraint_applied}"
        )

        return constrained_size, metadata

    def _calculate_fixed_fractional(
        self,
        account_balance: float,
        risk_per_share: float
    ) -> float:
        """
        Calculate position size using fixed fractional method.

        Risk fixed percentage of capital per trade.
        """
        risk_amount = account_balance * (self.base_risk_pct / 100)
        size = risk_amount / risk_per_share
        return size

    def _calculate_kelly(
        self,
        account_balance: float,
        risk_per_share: float,
        history: BacktestTradeHistory
    ) -> float:
        """
        Calculate position size using Kelly Criterion.

        Kelly% = W - [(1 - W) / R]
        Where:
        - W = Win rate
        - R = Average win / Average loss ratio

        We use fractional Kelly (default 0.25 = quarter Kelly) for safety.
        """
        if history.avg_loss == 0:
            logger.warning("Avg loss is zero, cannot calculate Kelly")
            return self._calculate_fixed_fractional(account_balance, risk_per_share)

        win_rate = history.win_rate
        rr_ratio = history.avg_win / history.avg_loss

        # Kelly formula
        kelly_pct = win_rate - ((1 - win_rate) / rr_ratio)

        # Apply Kelly fraction (quarter Kelly is safer)
        kelly_pct *= self.kelly_fraction

        # Ensure non-negative
        kelly_pct = max(0.0, kelly_pct)

        # Convert to position size
        risk_amount = account_balance * kelly_pct
        size = risk_amount / risk_per_share

        logger.debug(
            f"Kelly calculation: win_rate={win_rate:.2f}, R={rr_ratio:.2f}, "
            f"kelly={kelly_pct*100:.2f}%"
        )

        return size

    def _calculate_optimal_f(
        self,
        account_balance: float,
        risk_per_share: float,
        history: BacktestTradeHistory
    ) -> float:
        """
        Calculate position size using Optimal f (Ralph Vince method).

        Optimal f maximizes geometric growth by optimizing position size
        based on largest historical loss.

        Note: This is a simplified version. Full optimal f requires
        iterative optimization over trade sequence.
        """
        # Combine wins and losses into single list
        all_trades = history.wins + [-abs(loss) for loss in history.losses]

        if not all_trades:
            logger.warning("No trade history for optimal f")
            return self._calculate_fixed_fractional(account_balance, risk_per_share)

        # Find largest loss
        largest_loss = abs(min(all_trades))

        if largest_loss == 0:
            logger.warning("Largest loss is zero, cannot calculate optimal f")
            return self._calculate_fixed_fractional(account_balance, risk_per_share)

        # Optimal f approximation: f = 1 / (largest_loss_pct * num_consecutive_losses_expected)
        # Conservative approach: use max consecutive losses from history
        max_consecutive = max(history.max_consecutive_losses, 2)
        optimal_f = 1.0 / (largest_loss * max_consecutive)

        # Cap at reasonable value (often very aggressive)
        optimal_f = min(optimal_f, 0.05)  # Max 5% per trade

        # Convert to position size
        risk_amount = account_balance * optimal_f
        size = risk_amount / risk_per_share

        logger.debug(
            f"Optimal f calculation: largest_loss={largest_loss:.4f}, "
            f"max_consec={max_consecutive}, f={optimal_f:.4f}"
        )

        return size

    def _calculate_volatility_adjusted(
        self,
        account_balance: float,
        risk_per_share: float,
        entry_price: float,
        atr: float
    ) -> float:
        """
        Calculate position size adjusted for current volatility (ATR).

        Higher volatility = smaller position size
        Lower volatility = larger position size
        """
        # Normalize ATR to percentage of price
        atr_pct = (atr / entry_price) * 100

        # Reference volatility (e.g., 1% ATR)
        reference_atr_pct = 1.0

        # Adjustment factor (inverse relationship)
        volatility_factor = reference_atr_pct / atr_pct if atr_pct > 0 else 1.0

        # Cap adjustment to reasonable range
        volatility_factor = max(0.5, min(2.0, volatility_factor))

        # Calculate base size and adjust
        base_size = self._calculate_fixed_fractional(account_balance, risk_per_share)
        adjusted_size = base_size * volatility_factor

        logger.debug(
            f"Volatility adjustment: ATR={atr_pct:.2f}%, "
            f"factor={volatility_factor:.2f}, "
            f"size={base_size:.2f} -> {adjusted_size:.2f}"
        )

        return adjusted_size

    def validate_trade(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        account_balance: float,
        existing_positions: Dict[str, Any],
        correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        max_correlated_positions: int = 2,
        correlation_threshold: float = 0.7,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate trade before execution.

        Checks:
        - Position size not zero
        - Sufficient capital
        - Correlation limits

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        # Check size
        if position_size <= 0:
            return False, "position_size_zero"

        # Check capital
        required_capital = position_size * entry_price
        if required_capital > account_balance:
            return False, "insufficient_capital"

        # Check correlation limits (if provided)
        if correlation_matrix and existing_positions:
            highly_correlated_count = 0

            for existing_symbol in existing_positions.keys():
                if existing_symbol == symbol:
                    continue

                # Check correlation
                corr_key = tuple(sorted([symbol, existing_symbol]))
                correlation = correlation_matrix.get(corr_key, 0.0)

                if abs(correlation) > correlation_threshold:
                    highly_correlated_count += 1

            if highly_correlated_count >= max_correlated_positions:
                return False, f"max_correlated_positions_exceeded ({highly_correlated_count})"

        return True, None
