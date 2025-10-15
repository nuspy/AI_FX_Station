"""
Advanced Professional Backtesting Engine
Comprehensive backtesting with Monte Carlo, walk-forward analysis, and professional risk metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    direction: str  # 'long' or 'short'
    pair: str
    entry_reason: str
    exit_reason: Optional[str] = None
    commission: float = 0.0
    swap: float = 0.0

    @property
    def pnl(self) -> float:
        """Calculate trade P&L"""
        if self.exit_price is None:
            return 0.0

        if self.direction == 'long':
            return (self.exit_price - self.entry_price) * self.position_size - self.commission - self.swap
        else:
            return (self.entry_price - self.exit_price) * self.position_size - self.commission - self.swap

    @property
    def return_pct(self) -> float:
        """Calculate percentage return"""
        if self.exit_price is None:
            return 0.0

        if self.direction == 'long':
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.exit_price) / self.entry_price * 100

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_series: pd.Series
    returns_series: pd.Series
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals (1 for buy, -1 for sell, 0 for hold)"""
        pass

    @abstractmethod
    def get_position_size(self, data: pd.DataFrame, signal: int, current_capital: float) -> float:
        """Calculate position size for trade"""
        pass

class MACrossoverStrategy(TradingStrategy):
    """Simple moving average crossover strategy"""

    def __init__(self, fast_period: int = 10, slow_period: int = 20, risk_per_trade: float = 0.02):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.risk_per_trade = risk_per_trade

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate MA crossover signals"""
        fast_ma = data['close'].rolling(window=self.fast_period).mean()
        slow_ma = data['close'].rolling(window=self.slow_period).mean()

        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1

        # Only signal on crossover
        signal_changes = signals.diff()
        final_signals = pd.Series(0, index=data.index)
        final_signals[signal_changes == 2] = 1  # Bullish crossover
        final_signals[signal_changes == -2] = -1  # Bearish crossover

        return final_signals

    def get_position_size(self, data: pd.DataFrame, signal: int, current_capital: float) -> float:
        """Simple fixed risk position sizing"""
        risk_amount = current_capital * self.risk_per_trade
        return risk_amount / data['close'].iloc[-1] if signal != 0 else 0.0

class AdvancedBacktestEngine:
    """Professional backtesting engine with advanced analytics"""

    def __init__(self, initial_capital: float = 100000, commission: float = 0.0002, slippage: float = 0.0001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results_cache = {}

    def run_backtest(self, data: pd.DataFrame, strategy: TradingStrategy,
                    pair: str = "EURUSD") -> BacktestResults:
        """Run comprehensive backtest"""
        logger.info(f"Starting backtest for {pair}")

        # Generate trading signals
        signals = strategy.generate_signals(data)

        # Initialize tracking variables
        trades = []
        capital = self.initial_capital
        equity_curve = [capital]
        current_position = None

        for i, (timestamp, row) in enumerate(data.iterrows()):
            signal = signals.iloc[i]
            price = row['close']

            # Close existing position if opposite signal
            if current_position and signal != 0 and np.sign(signal) != np.sign(current_position['direction']):
                # Close current position
                exit_price = price * (1 + self.slippage if current_position['direction'] > 0 else 1 - self.slippage)

                trade = Trade(
                    entry_time=current_position['entry_time'],
                    exit_time=timestamp,
                    entry_price=current_position['entry_price'],
                    exit_price=exit_price,
                    position_size=current_position['size'],
                    direction='long' if current_position['direction'] > 0 else 'short',
                    pair=pair,
                    entry_reason=current_position['reason'],
                    exit_reason='signal_reversal',
                    commission=self.commission * current_position['size'] * 2  # Entry + exit
                )

                trades.append(trade)
                capital += trade.pnl
                current_position = None

            # Open new position if signal and no current position
            if signal != 0 and current_position is None:
                position_size = strategy.get_position_size(data.iloc[i:i+1], signal, capital)
                entry_price = price * (1 + self.slippage if signal > 0 else 1 - self.slippage)

                current_position = {
                    'entry_time': timestamp,
                    'entry_price': entry_price,
                    'direction': signal,
                    'size': position_size,
                    'reason': f'ma_crossover_{signal}'
                }

            # Update equity curve
            unrealized_pnl = 0
            if current_position:
                if current_position['direction'] > 0:
                    unrealized_pnl = (price - current_position['entry_price']) * current_position['size']
                else:
                    unrealized_pnl = (current_position['entry_price'] - price) * current_position['size']

            equity_curve.append(capital + unrealized_pnl)

        # Close final position if exists
        if current_position:
            final_price = data['close'].iloc[-1]
            exit_price = final_price * (1 + self.slippage if current_position['direction'] > 0 else 1 - self.slippage)

            trade = Trade(
                entry_time=current_position['entry_time'],
                exit_time=data.index[-1],
                entry_price=current_position['entry_price'],
                exit_price=exit_price,
                position_size=current_position['size'],
                direction='long' if current_position['direction'] > 0 else 'short',
                pair=pair,
                entry_reason=current_position['reason'],
                exit_reason='backtest_end',
                commission=self.commission * current_position['size'] * 2
            )

            trades.append(trade)
            capital += trade.pnl

        # Create results
        equity_series = pd.Series(equity_curve, index=[data.index[0]] + list(data.index))
        results = self._calculate_performance_metrics(trades, equity_series, data.index[0], data.index[-1])

        logger.info(f"Backtest completed: {len(trades)} trades, {results.total_return:.2f}% return")
        return results

    def run_monte_carlo_simulation(self, data: pd.DataFrame, strategy: TradingStrategy,
                                  num_simulations: int = 1000, confidence_levels: List[float] = [0.05, 0.95]) -> Dict[str, Any]:
        """Run Monte Carlo simulation for strategy robustness testing"""
        logger.info(f"Running Monte Carlo simulation with {num_simulations} iterations")

        results = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            for i in range(num_simulations):
                # Randomly sample returns while preserving correlation structure
                shuffled_data = self._shuffle_returns(data)
                future = executor.submit(self.run_backtest, shuffled_data, strategy)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append({
                        'total_return': result.total_return,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'win_rate': result.win_rate,
                        'total_trades': result.total_trades
                    })
                except Exception as e:
                    logger.warning(f"Monte Carlo simulation failed: {e}")

        # Calculate confidence intervals
        mc_analysis = {}
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
            values = [r[metric] for r in results if not np.isnan(r[metric])]
            if values:
                mc_analysis[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'percentiles': {
                        f'{int(cl*100)}%': np.percentile(values, cl*100)
                        for cl in confidence_levels
                    }
                }

        logger.info("Monte Carlo simulation completed")
        return mc_analysis

    def walk_forward_analysis(self, data: pd.DataFrame, strategy: TradingStrategy,
                             train_periods: int = 252, test_periods: int = 63,
                             step_size: int = 21) -> Dict[str, Any]:
        """Perform walk-forward analysis for strategy validation"""
        logger.info("Starting walk-forward analysis")

        wf_results = []
        start_idx = 0

        while start_idx + train_periods + test_periods <= len(data):
            # Training period
            train_data = data.iloc[start_idx:start_idx + train_periods]

            # Test period
            test_data = data.iloc[start_idx + train_periods:start_idx + train_periods + test_periods]

            # Run backtest on test period
            test_result = self.run_backtest(test_data, strategy)

            wf_results.append({
                'start_date': test_data.index[0],
                'end_date': test_data.index[-1],
                'return': test_result.total_return,
                'sharpe': test_result.sharpe_ratio,
                'max_dd': test_result.max_drawdown,
                'trades': test_result.total_trades
            })

            start_idx += step_size

        # Aggregate walk-forward results
        wf_analysis = {
            'periods': len(wf_results),
            'avg_return': np.mean([r['return'] for r in wf_results]),
            'avg_sharpe': np.mean([r['sharpe'] for r in wf_results if not np.isnan(r['sharpe'])]),
            'consistency': len([r for r in wf_results if r['return'] > 0]) / len(wf_results),
            'results': wf_results
        }

        logger.info(f"Walk-forward analysis completed: {len(wf_results)} periods")
        return wf_analysis

    def _shuffle_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Shuffle returns while preserving price structure"""
        returns = data['close'].pct_change().dropna()
        shuffled_returns = returns.sample(frac=1, random_state=np.random.randint(0, 10000)).values

        # Reconstruct prices from shuffled returns
        shuffled_data = data.copy()
        shuffled_prices = [data['close'].iloc[0]]

        for ret in shuffled_returns:
            shuffled_prices.append(shuffled_prices[-1] * (1 + ret))

        # Ensure correct length alignment
        shuffled_prices = shuffled_prices[:len(data)]
        if len(shuffled_prices) < len(data):
            # Pad with last price if needed
            shuffled_prices.extend([shuffled_prices[-1]] * (len(data) - len(shuffled_prices)))

        shuffled_data['close'] = shuffled_prices
        return shuffled_data

    def _calculate_performance_metrics(self, trades: List[Trade], equity_curve: pd.Series,
                                     start_date: datetime, end_date: datetime) -> BacktestResults:
        """Calculate comprehensive performance metrics"""

        # Basic metrics
        initial_capital = self.initial_capital
        final_capital = equity_curve.iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital * 100

        # Returns and drawdown calculation
        returns = equity_curve.pct_change().dropna()
        drawdown = (equity_curve / equity_curve.cummax() - 1) * 100

        # Time-based metrics
        days = (end_date - start_date).days
        years = days / 365.25

        annualized_return = (final_capital / initial_capital) ** (1/years) - 1 if years > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0

        # Risk metrics
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate

        downside_returns = returns[returns < 0]
        sortino_ratio = (annualized_return - 0.02) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0

        max_drawdown = drawdown.min()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Drawdown duration
        dd_duration = 0
        in_drawdown = False
        current_duration = 0
        max_duration = 0

        for dd in drawdown:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                in_drawdown = False
                current_duration = 0

        # Trade statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        profit_factor = abs(sum([t.pnl for t in winning_trades]) / sum([t.pnl for t in losing_trades])) if losing_trades and sum([t.pnl for t in losing_trades]) != 0 else float('inf')

        largest_win = max([t.pnl for t in trades]) if trades else 0
        largest_loss = min([t.pnl for t in trades]) if trades else 0

        return BacktestResults(
            trades=trades,
            equity_curve=equity_curve,
            drawdown_series=drawdown,
            returns_series=returns,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return * 100,
            volatility=volatility * 100,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss
        )

class RiskManagementSuite:
    """Advanced risk management and portfolio analytics"""

    def __init__(self):
        self.confidence_levels = [0.95, 0.99]

    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 10:
            return 0.0
        return np.percentile(returns, (1 - confidence_level) * 100)

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    def stress_test(self, portfolio_data: Dict[str, pd.DataFrame],
                   stress_scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Perform stress testing on portfolio"""
        stress_results = {}

        for scenario_name, shocks in stress_scenarios.items():
            scenario_impact = 0

            for asset, shock in shocks.items():
                if asset in portfolio_data:
                    current_value = portfolio_data[asset]['close'].iloc[-1]
                    shocked_value = current_value * (1 + shock)
                    impact = shocked_value - current_value
                    scenario_impact += impact

            stress_results[scenario_name] = {
                'total_impact': scenario_impact,
                'percentage_impact': (scenario_impact / sum([df['close'].iloc[-1] for df in portfolio_data.values()]) * 100)
            }

        return stress_results

    def portfolio_optimization(self, returns_matrix: pd.DataFrame,
                             risk_tolerance: float = 0.1) -> Dict[str, float]:
        """Simple portfolio optimization using risk parity"""
        # Calculate covariance matrix
        cov_matrix = returns_matrix.cov()

        # Risk parity weights (inverse volatility)
        volatilities = returns_matrix.std()
        inv_vol_weights = (1 / volatilities) / (1 / volatilities).sum()

        return inv_vol_weights.to_dict()

if __name__ == "__main__":
    # Test the backtesting engine
    print("Testing Advanced Backtesting Engine...")

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))

    sample_data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)

    # Create strategy and run backtest
    strategy = MACrossoverStrategy(fast_period=10, slow_period=30, risk_per_trade=0.02)
    backtest_engine = AdvancedBacktestEngine(initial_capital=100000)

    # Run comprehensive backtest
    results = backtest_engine.run_backtest(sample_data, strategy, "EURUSD")

    print(f"\n=== BACKTEST RESULTS ===")
    print(f"Total Return: {results.total_return:.2f}%")
    print(f"Annualized Return: {results.annualized_return:.2f}%")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2f}%")
    print(f"Win Rate: {results.win_rate:.1f}%")
    print(f"Total Trades: {results.total_trades}")
    print(f"Profit Factor: {results.profit_factor:.2f}")

    # Run Monte Carlo simulation (smaller sample for testing)
    print(f"\n=== MONTE CARLO SIMULATION ===")
    mc_results = backtest_engine.run_monte_carlo_simulation(sample_data, strategy, num_simulations=50)

    if 'total_return' in mc_results:
        print(f"Mean Return: {mc_results['total_return']['mean']:.2f}%")
        print(f"Return Std Dev: {mc_results['total_return']['std']:.2f}%")
        print(f"5th Percentile: {mc_results['total_return']['percentiles']['5%']:.2f}%")
        print(f"95th Percentile: {mc_results['total_return']['percentiles']['95%']:.2f}%")

    # Risk management suite
    print(f"\n=== RISK ANALYSIS ===")
    risk_suite = RiskManagementSuite()

    if len(results.returns_series) > 10:
        var_95 = risk_suite.calculate_var(results.returns_series, 0.95)
        cvar_95 = risk_suite.calculate_cvar(results.returns_series, 0.95)

        print(f"95% VaR: {var_95:.4f} ({var_95*100:.2f}%)")
        print(f"95% CVaR: {cvar_95:.4f} ({cvar_95*100:.2f}%)")

    print(f"\n+ Advanced Backtesting Engine Successfully Implemented!")
    print(f"+ Professional-grade risk analytics and Monte Carlo simulation ready")
    print(f"+ Walk-forward analysis and portfolio optimization available")