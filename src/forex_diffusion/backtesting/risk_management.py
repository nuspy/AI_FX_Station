"""
Professional Risk Management Suite
Advanced portfolio risk analytics, position sizing, and risk monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from scipy import stats

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics container"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    downside_deviation: float
    upside_capture: float
    downside_capture: float

@dataclass
class PositionSizingResult:
    """Position sizing calculation result"""
    position_size: float
    risk_amount: float
    risk_percentage: float
    kelly_fraction: float
    recommended_size: float
    max_position_size: float
    reasoning: str

class PortfolioRiskAnalyzer:
    """Advanced portfolio risk analysis and monitoring"""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = [0.90, 0.95, 0.99]

    def calculate_comprehensive_metrics(self, returns: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""

        if len(returns) < 10:
            logger.warning("Insufficient data for risk calculations")
            return self._empty_risk_metrics()

        # Basic volatility metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        mean_return = returns.mean() * 252

        # VaR calculations
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # CVaR calculations
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99

        # Performance ratios
        sharpe_ratio = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Downside metrics
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Benchmark-relative metrics (if benchmark provided)
        beta = 0.0
        alpha = 0.0
        information_ratio = 0.0
        tracking_error = 0.0
        upside_capture = 0.0
        downside_capture = 0.0

        if benchmark_returns is not None and len(benchmark_returns) >= len(returns):
            try:
                # Align series
                aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')

                if len(aligned_returns) > 10:
                    # Beta calculation
                    covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                    benchmark_variance = np.var(aligned_benchmark)
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

                    # Alpha calculation
                    benchmark_mean = aligned_benchmark.mean() * 252
                    alpha = mean_return - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))

                    # Tracking error and information ratio
                    active_returns = aligned_returns - aligned_benchmark
                    tracking_error = active_returns.std() * np.sqrt(252)
                    information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0

                    # Capture ratios
                    up_periods = aligned_benchmark > 0
                    down_periods = aligned_benchmark < 0

                    if up_periods.sum() > 0:
                        upside_capture = aligned_returns[up_periods].mean() / aligned_benchmark[up_periods].mean()

                    if down_periods.sum() > 0:
                        downside_capture = aligned_returns[down_periods].mean() / aligned_benchmark[down_periods].mean()

            except Exception as e:
                logger.warning(f"Error calculating benchmark-relative metrics: {e}")

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            downside_deviation=downside_deviation,
            upside_capture=upside_capture,
            downside_capture=downside_capture
        )

    def stress_test_portfolio(self, portfolio_returns: pd.DataFrame,
                            custom_scenarios: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """Comprehensive portfolio stress testing"""

        # Default stress scenarios
        default_scenarios = {
            'market_crash': {'equity_shock': -0.30, 'bond_shock': -0.10, 'currency_shock': 0.15},
            'interest_rate_shock': {'equity_shock': -0.15, 'bond_shock': -0.20, 'currency_shock': 0.05},
            'currency_crisis': {'equity_shock': -0.10, 'bond_shock': -0.05, 'currency_shock': 0.25},
            'inflation_spike': {'equity_shock': -0.20, 'bond_shock': -0.25, 'currency_shock': 0.10},
            'liquidity_crisis': {'equity_shock': -0.35, 'bond_shock': -0.15, 'currency_shock': 0.20}
        }

        scenarios = custom_scenarios if custom_scenarios else default_scenarios
        stress_results = {}

        for scenario_name, shocks in scenarios.items():
            scenario_impact = 0
            detailed_impact = {}

            for asset in portfolio_returns.columns:
                # Map generic shock types to assets
                shock_value = 0
                if 'equity' in asset.lower() or 'stock' in asset.lower():
                    shock_value = shocks.get('equity_shock', 0)
                elif 'bond' in asset.lower() or 'treasury' in asset.lower():
                    shock_value = shocks.get('bond_shock', 0)
                elif 'currency' in asset.lower() or 'forex' in asset.lower() or 'usd' in asset.lower():
                    shock_value = shocks.get('currency_shock', 0)

                # Calculate impact
                current_value = portfolio_returns[asset].iloc[-1] if not portfolio_returns[asset].empty else 0
                impact = current_value * shock_value
                detailed_impact[asset] = impact
                scenario_impact += impact

            # Calculate portfolio-level impact
            total_portfolio_value = sum([portfolio_returns[col].iloc[-1] for col in portfolio_returns.columns])
            percentage_impact = (scenario_impact / total_portfolio_value * 100) if total_portfolio_value != 0 else 0

            stress_results[scenario_name] = {
                'total_impact': scenario_impact,
                'percentage_impact': percentage_impact,
                'asset_impacts': detailed_impact,
                'severity': self._classify_stress_severity(percentage_impact)
            }

        return stress_results

    def calculate_portfolio_var(self, portfolio_returns: pd.DataFrame,
                               weights: Optional[np.ndarray] = None,
                               confidence_level: float = 0.95,
                               holding_period: int = 1) -> Dict[str, float]:
        """Calculate portfolio Value at Risk"""

        if weights is None:
            weights = np.array([1.0 / len(portfolio_returns.columns)] * len(portfolio_returns.columns))

        # Portfolio returns
        portfolio_rets = (portfolio_returns * weights).sum(axis=1)

        # Parametric VaR
        portfolio_vol = portfolio_rets.std()
        portfolio_mean = portfolio_rets.mean()

        z_score = stats.norm.ppf(1 - confidence_level)
        parametric_var = -(portfolio_mean + z_score * portfolio_vol) * np.sqrt(holding_period)

        # Historical VaR
        historical_var = -np.percentile(portfolio_rets, (1 - confidence_level) * 100) * np.sqrt(holding_period)

        # Monte Carlo VaR (simplified)
        mc_simulations = np.random.multivariate_normal(
            mean=portfolio_returns.mean(),
            cov=portfolio_returns.cov(),
            size=10000
        )

        mc_portfolio_returns = (mc_simulations * weights).sum(axis=1)
        mc_var = -np.percentile(mc_portfolio_returns, (1 - confidence_level) * 100) * np.sqrt(holding_period)

        return {
            'parametric_var': parametric_var,
            'historical_var': historical_var,
            'monte_carlo_var': mc_var,
            'confidence_level': confidence_level,
            'holding_period_days': holding_period
        }

    def _classify_stress_severity(self, percentage_impact: float) -> str:
        """Classify stress test severity"""
        abs_impact = abs(percentage_impact)

        if abs_impact < 5:
            return 'LOW'
        elif abs_impact < 15:
            return 'MODERATE'
        elif abs_impact < 25:
            return 'HIGH'
        else:
            return 'SEVERE'

    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics for insufficient data"""
        return RiskMetrics(
            var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
            volatility=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
            max_drawdown=0.0, beta=0.0, alpha=0.0,
            information_ratio=0.0, tracking_error=0.0,
            downside_deviation=0.0, upside_capture=0.0, downside_capture=0.0
        )

class PositionSizingEngine:
    """Advanced position sizing with multiple methodologies"""

    def __init__(self, max_portfolio_risk: float = 0.02, max_position_size: float = 0.10):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size

    def calculate_position_size(self, account_balance: float, entry_price: float,
                              stop_loss_price: float, confidence: float = 0.6,
                              win_rate: float = 0.5, avg_win: float = 1.0,
                              avg_loss: float = -1.0,
                              dom_metrics: Optional[Dict[str, Any]] = None) -> PositionSizingResult:
        """
        Calculate optimal position size using multiple methodologies with DOM awareness.

        Args:
            account_balance: Available capital
            entry_price: Entry price for position
            stop_loss_price: Stop loss price
            confidence: Trading signal confidence (0-1)
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade size
            avg_loss: Average losing trade size
            dom_metrics: Optional DOM data with keys:
                - spread: Current spread
                - bid_depth: Available bid volume
                - ask_depth: Available ask volume
                - imbalance: Order book imbalance (-1 to +1)
                - bids: List of [price, volume] pairs
                - asks: List of [price, volume] pairs

        Returns:
            PositionSizingResult with sizing recommendation
        """

        # Ensure we have valid inputs
        if entry_price <= 0 or account_balance <= 0:
            return PositionSizingResult(
                position_size=0, risk_amount=0, risk_percentage=0,
                kelly_fraction=0, recommended_size=0, max_position_size=0,
                reasoning="Invalid input parameters"
            )

        # Risk-based position sizing
        risk_per_share = abs(entry_price - stop_loss_price) if stop_loss_price > 0 else entry_price * 0.02
        risk_amount = account_balance * self.max_portfolio_risk
        risk_based_size = risk_amount / risk_per_share if risk_per_share > 0 else 0

        # Kelly Criterion with proper calculation
        kelly_fraction = 0
        if win_rate > 0 and win_rate < 1:
            # Convert dollar amounts to ratios if needed
            if abs(avg_loss) > 1:  # Assume dollar amounts
                avg_win_ratio = avg_win / account_balance
                avg_loss_ratio = abs(avg_loss) / account_balance
            else:  # Already ratios
                avg_win_ratio = avg_win
                avg_loss_ratio = abs(avg_loss)

            # Kelly formula: f = (p*b - q) / b
            # where p = win_rate, q = 1-win_rate, b = avg_win/avg_loss ratio
            if avg_loss_ratio > 0:
                b = avg_win_ratio / avg_loss_ratio
                kelly_fraction = (win_rate * b - (1 - win_rate)) / b if b > 0 else 0
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

        kelly_size = (account_balance * kelly_fraction) / entry_price if kelly_fraction > 0 else 0

        # Fixed fractional
        fixed_fractional_size = (account_balance * self.max_position_size) / entry_price if entry_price > 0 else 0

        # Volatility-based sizing
        volatility_multiplier = max(0.5, min(2.0, 1 / (confidence if confidence > 0 else 0.5)))
        volatility_adjusted_size = risk_based_size / volatility_multiplier

        # DOM-BASED CONSTRAINTS
        liquidity_size = float('inf')  # No limit if DOM not available
        spread_penalty = 1.0
        dom_reasoning_parts = []

        if dom_metrics:
            # 1. Liquidity Size Constraint
            bid_depth = dom_metrics.get('bid_depth', 0)
            ask_depth = dom_metrics.get('ask_depth', 0)
            available_depth = bid_depth + ask_depth

            if available_depth > 0:
                # Max 50% of available depth
                liquidity_size = available_depth * 0.5
                dom_reasoning_parts.append(f"Liquidity constraint: {liquidity_size:.2f} (50% of depth {available_depth:.2f})")

            # 2. Spread Cost Adjustment
            spread = dom_metrics.get('spread', 0)
            if spread > 0:
                spread_bps = (spread / entry_price) * 10000  # Convert to basis points

                if spread_bps > 10:
                    spread_penalty = 0.6
                    dom_reasoning_parts.append(f"Spread penalty 0.6x (very wide: {spread_bps:.1f} bps)")
                elif spread_bps > 3:
                    spread_penalty = 0.8
                    dom_reasoning_parts.append(f"Spread penalty 0.8x (wide: {spread_bps:.1f} bps)")
                else:
                    dom_reasoning_parts.append(f"Spread normal ({spread_bps:.1f} bps)")

        # Select recommended size (most conservative including liquidity)
        recommended_size = min(
            risk_based_size,
            kelly_size,
            fixed_fractional_size,
            volatility_adjusted_size,
            liquidity_size
        )

        # Apply spread penalty
        recommended_size *= spread_penalty

        # Final validation
        max_allowed_size = (account_balance * self.max_position_size) / entry_price if entry_price > 0 else 0
        final_size = min(recommended_size, max_allowed_size)

        # Risk calculations
        actual_risk_amount = final_size * risk_per_share
        risk_percentage = (actual_risk_amount / account_balance) * 100

        # Reasoning
        sizing_method = "risk_based"
        if final_size == kelly_size:
            sizing_method = "kelly_optimal"
        elif final_size == fixed_fractional_size:
            sizing_method = "fixed_fractional"
        elif final_size == volatility_adjusted_size:
            sizing_method = "volatility_adjusted"
        elif final_size == liquidity_size * spread_penalty:
            sizing_method = "liquidity_constrained"

        reasoning = f"Using {sizing_method} methodology. Risk per share: {risk_per_share:.4f}, Kelly fraction: {kelly_fraction:.3f}"

        # Add DOM reasoning if available
        if dom_reasoning_parts:
            reasoning += ". DOM: " + "; ".join(dom_reasoning_parts)

        return PositionSizingResult(
            position_size=final_size,
            risk_amount=actual_risk_amount,
            risk_percentage=risk_percentage,
            kelly_fraction=kelly_fraction,
            recommended_size=recommended_size,
            max_position_size=max_allowed_size,
            reasoning=reasoning
        )

    def portfolio_heat_check(self, current_positions: List[Dict[str, Any]],
                           account_balance: float) -> Dict[str, Any]:
        """Analyze current portfolio heat and risk concentration"""

        total_risk = 0
        position_risks = []
        sector_exposure = {}
        currency_exposure = {}

        for position in current_positions:
            position_risk = position.get('risk_amount', 0)
            total_risk += position_risk

            # Sector analysis
            sector = position.get('sector', 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position.get('value', 0)

            # Currency analysis
            currency = position.get('currency', 'USD')
            currency_exposure[currency] = currency_exposure.get(currency, 0) + position.get('value', 0)

            position_risks.append(position_risk)

        # Risk concentration analysis
        total_portfolio_risk = (total_risk / account_balance) * 100
        max_individual_risk = max(position_risks) if position_risks else 0
        avg_position_risk = np.mean(position_risks) if position_risks else 0

        # Diversification metrics
        sector_concentration = max(sector_exposure.values()) / sum(sector_exposure.values()) * 100 if sector_exposure else 0
        currency_concentration = max(currency_exposure.values()) / sum(currency_exposure.values()) * 100 if currency_exposure else 0

        # Risk level classification
        risk_level = "LOW"
        if total_portfolio_risk > 10:
            risk_level = "HIGH"
        elif total_portfolio_risk > 5:
            risk_level = "MODERATE"

        return {
            'total_portfolio_risk': total_portfolio_risk,
            'risk_level': risk_level,
            'position_count': len(current_positions),
            'max_individual_risk': max_individual_risk,
            'avg_position_risk': avg_position_risk,
            'sector_concentration': sector_concentration,
            'currency_concentration': currency_concentration,
            'diversification_score': min(100, (100 - sector_concentration + 100 - currency_concentration) / 2),
            'recommendations': self._generate_risk_recommendations(total_portfolio_risk, sector_concentration, currency_concentration)
        }

    def _generate_risk_recommendations(self, total_risk: float, sector_conc: float, currency_conc: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []

        if total_risk > 10:
            recommendations.append("URGENT: Portfolio risk exceeds 10%. Consider reducing position sizes.")
        elif total_risk > 5:
            recommendations.append("WARNING: Portfolio risk above 5%. Monitor positions closely.")

        if sector_conc > 40:
            recommendations.append("High sector concentration detected. Consider diversifying across sectors.")

        if currency_conc > 60:
            recommendations.append("High currency concentration. Consider hedging or diversifying currencies.")

        if not recommendations:
            recommendations.append("Portfolio risk levels are within acceptable limits.")

        return recommendations

if __name__ == "__main__":
    # Test the risk management suite
    print("Testing Professional Risk Management Suite...")

    # Generate sample portfolio data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')

    # Multi-asset portfolio simulation
    portfolio_data = pd.DataFrame({
        'EURUSD': np.random.normal(0.0005, 0.015, len(dates)),
        'GBPUSD': np.random.normal(0.0003, 0.018, len(dates)),
        'USDJPY': np.random.normal(0.0002, 0.012, len(dates)),
        'GOLD': np.random.normal(0.0008, 0.020, len(dates))
    }, index=dates)

    # Create benchmark
    benchmark = pd.Series(np.random.normal(0.0004, 0.01, len(dates)), index=dates)

    # Test risk analyzer
    risk_analyzer = PortfolioRiskAnalyzer()

    print("\n=== PORTFOLIO RISK ANALYSIS ===")
    for asset in portfolio_data.columns:
        metrics = risk_analyzer.calculate_comprehensive_metrics(
            portfolio_data[asset], benchmark
        )

        print(f"\n{asset} Risk Metrics:")
        print(f"  Volatility: {metrics.volatility:.2f}%")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2f}%")
        print(f"  95% VaR: {metrics.var_95:.4f}")
        print(f"  Beta: {metrics.beta:.2f}")

    # Portfolio VaR
    print("\n=== PORTFOLIO VALUE AT RISK ===")
    portfolio_var = risk_analyzer.calculate_portfolio_var(portfolio_data)
    print(f"Parametric VaR (95%): {portfolio_var['parametric_var']:.4f}")
    print(f"Historical VaR (95%): {portfolio_var['historical_var']:.4f}")
    print(f"Monte Carlo VaR (95%): {portfolio_var['monte_carlo_var']:.4f}")

    # Stress testing
    print("\n=== STRESS TEST RESULTS ===")
    stress_results = risk_analyzer.stress_test_portfolio(portfolio_data)
    for scenario, result in stress_results.items():
        print(f"{scenario}: {result['percentage_impact']:.1f}% impact ({result['severity']})")

    # Position sizing
    print("\n=== POSITION SIZING ===")
    position_engine = PositionSizingEngine()

    position_result = position_engine.calculate_position_size(
        account_balance=100000,
        entry_price=1.1000,
        stop_loss_price=1.0950,
        confidence=0.65,
        win_rate=0.55,
        avg_win=150,
        avg_loss=-75
    )

    print(f"Recommended Position Size: {position_result.position_size:.0f} units")
    print(f"Risk Amount: ${position_result.risk_amount:.2f}")
    print(f"Risk Percentage: {position_result.risk_percentage:.2f}%")
    print(f"Kelly Fraction: {position_result.kelly_fraction:.3f}")
    print(f"Method: {position_result.reasoning}")

    # Portfolio heat check
    sample_positions = [
        {'risk_amount': 1000, 'value': 10000, 'sector': 'Currency', 'currency': 'EUR'},
        {'risk_amount': 800, 'value': 8000, 'sector': 'Currency', 'currency': 'GBP'},
        {'risk_amount': 1200, 'value': 12000, 'sector': 'Commodity', 'currency': 'USD'}
    ]

    heat_check = position_engine.portfolio_heat_check(sample_positions, 100000)
    print("\n=== PORTFOLIO HEAT CHECK ===")
    print(f"Total Portfolio Risk: {heat_check['total_portfolio_risk']:.2f}%")
    print(f"Risk Level: {heat_check['risk_level']}")
    print(f"Diversification Score: {heat_check['diversification_score']:.1f}/100")

    for rec in heat_check['recommendations']:
        print(f"- {rec}")

    print("\n+ Professional Risk Management Suite Successfully Implemented!")
    print("+ Comprehensive risk metrics, VaR, stress testing ready")
    print("+ Advanced position sizing and portfolio heat monitoring available")