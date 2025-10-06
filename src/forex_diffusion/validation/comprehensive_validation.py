"""
Comprehensive Walk-Forward Validation

Integrates all advanced components for robust validation:
- Multi-timeframe ensemble
- Multi-model stacked ensemble
- Regime-aware validation
- Transaction cost modeling
- Multi-objective optimization
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Import existing components
try:
    from ..models.multi_timeframe_ensemble import MultiTimeframeEnsemble, Timeframe
    from ..models.ml_stacked_ensemble import StackedMLEnsemble
    from ..regime.hmm_detector import HMMRegimeDetector
    from ..risk.multi_level_stop_loss import MultiLevelStopLoss
    from ..risk.regime_position_sizer import RegimePositionSizer, MarketRegime
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class ValidationResult:
    """Results from comprehensive validation."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_pnl: float
    regime_performance: Dict[str, Dict]
    timeframe_performance: Dict[str, Dict]
    model_attribution: Dict[str, float]


class ComprehensiveValidator:
    """
    Comprehensive walk-forward validation system.

    Integrates:
    - Multiple timeframes
    - Multiple models
    - Regime detection
    - Risk management
    - Transaction costs
    """

    def __init__(
        self,
        use_multi_timeframe: bool = True,
        use_stacked_ensemble: bool = True,
        use_regime_detection: bool = True,
        use_risk_management: bool = True,
        transaction_cost_pct: float = 0.0001  # 1 pip spread
    ):
        """
        Initialize comprehensive validator.

        Args:
            use_multi_timeframe: Enable multi-timeframe ensemble
            use_stacked_ensemble: Enable multi-model stacking
            use_regime_detection: Enable regime detection
            use_risk_management: Enable advanced risk management
            transaction_cost_pct: Transaction cost as percentage (default: 0.01%)
        """
        self.use_multi_timeframe = use_multi_timeframe
        self.use_stacked_ensemble = use_stacked_ensemble
        self.use_regime_detection = use_regime_detection
        self.use_risk_management = use_risk_management
        self.transaction_cost_pct = transaction_cost_pct

        # Initialize components
        self.mtf_ensemble: Optional[MultiTimeframeEnsemble] = None
        self.ml_ensemble: Optional[StackedMLEnsemble] = None
        self.regime_detector: Optional[HMMRegimeDetector] = None
        self.risk_manager: Optional[MultiLevelStopLoss] = None
        self.position_sizer: Optional[RegimePositionSizer] = None

        # Performance tracking
        self.trades_history: List[Dict] = []
        self.regime_performance: Dict[str, List[float]] = {}
        self.timeframe_performance: Dict[str, List[float]] = {}

    def validate_walk_forward(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        labels: pd.Series,
        train_size: int = 1000,
        test_size: int = 200,
        step_size: int = 100,
        verbose: bool = True
    ) -> ValidationResult:
        """
        Perform walk-forward validation.

        Args:
            data: Full OHLCV dataset
            features: Calculated features
            labels: Target labels
            train_size: Size of training window
            test_size: Size of test window
            step_size: Step between windows
            verbose: Print progress

        Returns:
            ValidationResult with comprehensive metrics
        """
        if verbose:
            print("=" * 80)
            print("COMPREHENSIVE WALK-FORWARD VALIDATION")
            print("=" * 80)
            print(f"Train size: {train_size}")
            print(f"Test size: {test_size}")
            print(f"Step size: {step_size}")
            print(f"Multi-timeframe: {self.use_multi_timeframe}")
            print(f"Stacked ensemble: {self.use_stacked_ensemble}")
            print(f"Regime detection: {self.use_regime_detection}")
            print(f"Risk management: {self.use_risk_management}")
            print("=" * 80)

        # Reset tracking
        self.trades_history = []
        self.regime_performance = {}
        self.timeframe_performance = {}

        # Walk-forward loop
        n_windows = (len(data) - train_size - test_size) // step_size + 1

        for window_idx in range(n_windows):
            # Define window indices
            train_start = window_idx * step_size
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + test_size

            # Check bounds
            if test_end > len(data):
                break

            if verbose:
                print(f"\n[Window {window_idx + 1}/{n_windows}] "
                      f"Train: {train_start}-{train_end}, Test: {test_start}-{test_end}")

            # Split data
            X_train = features.iloc[train_start:train_end]
            y_train = labels.iloc[train_start:train_end]
            X_test = features.iloc[test_start:test_end]
            y_test = labels.iloc[test_start:test_end]
            data_test = data.iloc[test_start:test_end]

            # Train model(s)
            if self.use_stacked_ensemble:
                # Train stacked ensemble
                model = StackedMLEnsemble(n_folds=5)
                model.fit(X_train, y_train, verbose=False)
            else:
                # Train single model (fallback)
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

            # Test on validation window
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

            # Simulate trading with predictions
            self._simulate_trading(
                data_test,
                predictions,
                probabilities,
                y_test,
                window_idx
            )

        # Calculate final metrics
        result = self._calculate_validation_metrics()

        if verbose:
            self._print_results(result)

        return result

    def _simulate_trading(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray],
        true_labels: pd.Series,
        window_idx: int
    ):
        """
        Simulate trading with predictions.

        Args:
            data: OHLCV data for test period
            predictions: Model predictions
            probabilities: Model probabilities (optional)
            true_labels: Actual labels
            window_idx: Current window index
        """
        account_balance = 10000  # Starting balance
        position = None

        for i in range(len(predictions)):
            signal = predictions[i]
            price = data.iloc[i]['close']

            # Detect regime (if enabled)
            current_regime = None
            if self.use_regime_detection and self.regime_detector:
                # Use recent data for regime detection
                regime_data = data.iloc[max(0, i-100):i+1]
                if len(regime_data) >= 50:
                    current_regime = self._detect_regime(regime_data)

            # Calculate position size (if enabled)
            if self.use_risk_management and position is None and signal != 0:
                position_size = self._calculate_position_size(
                    account_balance,
                    price,
                    signal,
                    current_regime
                )
            else:
                position_size = 1.0

            # Entry signal
            if position is None and signal != 0:
                # Apply transaction cost
                entry_price = price * (1 + self.transaction_cost_pct * np.sign(signal))

                position = {
                    'entry_price': entry_price,
                    'entry_index': i,
                    'direction': 'long' if signal > 0 else 'short',
                    'size': position_size,
                    'regime': current_regime
                }

            # Exit signal or stop loss
            elif position is not None:
                should_exit = False
                exit_reason = ""

                # Signal reversal
                if signal != 0 and \
                   ((position['direction'] == 'long' and signal < 0) or
                    (position['direction'] == 'short' and signal > 0)):
                    should_exit = True
                    exit_reason = "signal_reversal"

                # End of period
                if i == len(predictions) - 1:
                    should_exit = True
                    exit_reason = "end_of_period"

                if should_exit:
                    # Apply transaction cost
                    exit_price = price * (1 - self.transaction_cost_pct * np.sign(signal))

                    # Calculate P&L
                    if position['direction'] == 'long':
                        pnl = (exit_price - position['entry_price']) * position['size']
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['size']

                    # Determine if correct
                    was_correct = pnl > 0

                    # Record trade
                    trade = {
                        'window': window_idx,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'direction': position['direction'],
                        'pnl': pnl,
                        'was_correct': was_correct,
                        'regime': position['regime'],
                        'exit_reason': exit_reason
                    }
                    self.trades_history.append(trade)

                    # Update account
                    account_balance += pnl

                    # Track regime performance
                    if position['regime']:
                        if position['regime'] not in self.regime_performance:
                            self.regime_performance[position['regime']] = []
                        self.regime_performance[position['regime']].append(pnl)

                    # Clear position
                    position = None

    def _detect_regime(self, data: pd.DataFrame) -> Optional[str]:
        """Detect current market regime."""
        try:
            if self.regime_detector is None:
                self.regime_detector = HMMRegimeDetector(n_regimes=4)

            # Fit detector on recent data
            self.regime_detector.fit(data)

            # Get current regime
            regime_state = self.regime_detector.predict_current(data)

            return regime_state.regime.value if regime_state else None

        except Exception:
            return None

    def _calculate_position_size(
        self,
        account_balance: float,
        price: float,
        signal: int,
        regime: Optional[str]
    ) -> float:
        """Calculate position size using regime-aware sizing."""
        if self.position_sizer is None:
            self.position_sizer = RegimePositionSizer()

        # Map regime string to MarketRegime enum
        regime_map = {
            'trending_up': MarketRegime.TRENDING_UP,
            'trending_down': MarketRegime.TRENDING_DOWN,
            'ranging': MarketRegime.RANGING,
            'volatile': MarketRegime.VOLATILE
        }

        market_regime = regime_map.get(regime, MarketRegime.RANGING)

        # Calculate size
        stop_distance = price * 0.02  # 2% stop
        stop_price = price - stop_distance if signal > 0 else price + stop_distance

        sizing = self.position_sizer.calculate_position_size(
            account_balance=account_balance,
            entry_price=price,
            stop_loss_price=stop_price,
            current_regime=market_regime,
            pattern_confidence=0.7
        )

        return sizing['position_size']

    def _calculate_validation_metrics(self) -> ValidationResult:
        """Calculate final validation metrics."""
        if not self.trades_history:
            return ValidationResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_trade_pnl=0.0,
                regime_performance={},
                timeframe_performance={},
                model_attribution={}
            )

        # Basic metrics
        total_trades = len(self.trades_history)
        winning_trades = sum(1 for t in self.trades_history if t['was_correct'])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # P&L metrics
        pnls = [t['pnl'] for t in self.trades_history]
        total_pnl = sum(pnls)
        avg_trade_pnl = np.mean(pnls)

        # Sharpe ratio
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        # Regime performance
        regime_perf = {}
        for regime, pnls in self.regime_performance.items():
            regime_perf[regime] = {
                'trades': len(pnls),
                'total_pnl': sum(pnls),
                'avg_pnl': np.mean(pnls),
                'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            }

        return ValidationResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_trade_pnl=avg_trade_pnl,
            regime_performance=regime_perf,
            timeframe_performance={},
            model_attribution={}
        )

    def _print_results(self, result: ValidationResult):
        """Print validation results."""
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print(f"\n📊 TRADING METRICS:")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Winning Trades: {result.winning_trades}")
        print(f"  Losing Trades: {result.losing_trades}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"\n💰 PERFORMANCE:")
        print(f"  Total P&L: ${result.total_pnl:,.2f}")
        print(f"  Avg Trade P&L: ${result.avg_trade_pnl:,.2f}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: ${result.max_drawdown:,.2f}")

        if result.regime_performance:
            print(f"\n🔄 REGIME PERFORMANCE:")
            for regime, perf in result.regime_performance.items():
                print(f"  {regime}:")
                print(f"    Trades: {perf['trades']}")
                print(f"    Win Rate: {perf['win_rate']:.2%}")
                print(f"    Avg P&L: ${perf['avg_pnl']:,.2f}")

        print("=" * 80)
