"""
Cross-Asset Correlation Analyzer

Analyzes inter-asset relationships and generates correlation-based signals.
Monitors systemic risk and provides portfolio correlation safety scores.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
import json


class CorrelationSignalType(Enum):
    """Types of correlation signals"""
    BREAKDOWN = "breakdown"  # Correlation breaks down
    DIVERGENCE = "divergence"  # Assets diverge
    CONVERGENCE = "convergence"  # Assets converge
    BASKET_STRENGTH = "basket_strength"  # Currency basket strength
    BASKET_WEAKNESS = "basket_weakness"  # Currency basket weakness
    SYSTEMIC_RISK = "systemic_risk"  # High correlation across portfolio


@dataclass
class CorrelationMatrix:
    """Correlation matrix snapshot"""
    timestamp: int
    timeframe: str
    window_size: int

    # Correlation data
    matrix_data: Dict[Tuple[str, str], float]  # (asset1, asset2) -> correlation
    asset_list: List[str]

    # Statistics
    avg_correlation: float
    max_correlation: float
    min_correlation: float

    # Regime context
    regime: Optional[str] = None
    correlation_regime: Optional[str] = None  # high, low, mixed


@dataclass
class CorrelationSignal:
    """Correlation-based trading signal"""
    signal_type: CorrelationSignalType
    primary_asset: str
    related_assets: List[str]
    direction: str  # 'bull' or 'bear'
    strength: float  # 0-1
    confidence: float  # 0-1
    timestamp: int
    correlation_value: float
    expected_correlation: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PortfolioCorrelationRisk:
    """Portfolio-level correlation risk assessment"""
    timestamp: int
    positions: List[str]
    avg_correlation: float
    max_correlation: float
    correlation_score: float  # 0-1, higher = more diversified
    systemic_risk_level: str  # low, medium, high, critical
    recommended_action: Optional[str] = None  # close_correlated, reduce_exposure, etc.


class CrossAssetCorrelationAnalyzer:
    """
    Analyzes correlations between assets for signal generation and risk management.

    Monitors:
    - Currency pair correlations (EUR/USD ↔ GBP/USD)
    - Commodity-currency links (Gold ↔ AUD/USD)
    - Risk-on/risk-off indicators
    - Interest rate differentials
    """

    # Typical known correlations (baseline expectations)
    KNOWN_CORRELATIONS = {
        ('EURUSD', 'GBPUSD'): 0.75,
        ('EURUSD', 'USDCHF'): -0.85,
        ('AUDUSD', 'NZDUSD'): 0.85,
        ('AUDUSD', 'XAUUSD'): 0.65,  # Gold
        ('USDJPY', 'SPX500'): 0.70,  # Risk-on correlation
        ('EURUSD', 'USDJPY'): -0.60,
    }

    # Correlation-based asset groups
    ASSET_GROUPS = {
        'USD_majors': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY'],
        'EUR_crosses': ['EURUSD', 'EURGBP', 'EURJPY', 'EURAUD'],
        'Commodities': ['XAUUSD', 'XAGUSD', 'WTIUSD'],
        'Safe_havens': ['USDJPY', 'USDCHF', 'XAUUSD'],
        'Risk_on': ['AUDUSD', 'NZDUSD', 'SPX500']
    }

    def __init__(
        self,
        rolling_window: int = 50,
        min_samples: int = 30,
        breakdown_threshold: float = 0.3,
        divergence_threshold: float = 0.5,
        systemic_risk_threshold: float = 0.75,
        max_portfolio_correlation: float = 0.70
    ):
        """
        Initialize correlation analyzer.

        Args:
            rolling_window: Bars for rolling correlation calculation
            min_samples: Minimum samples needed for reliable correlation
            breakdown_threshold: Threshold for correlation breakdown detection
            divergence_threshold: Threshold for divergence opportunities
            systemic_risk_threshold: Correlation level indicating systemic risk
            max_portfolio_correlation: Maximum acceptable portfolio correlation
        """
        self.rolling_window = rolling_window
        self.min_samples = min_samples
        self.breakdown_threshold = breakdown_threshold
        self.divergence_threshold = divergence_threshold
        self.systemic_risk_threshold = systemic_risk_threshold
        self.max_portfolio_correlation = max_portfolio_correlation

        # State
        self.correlation_history: List[CorrelationMatrix] = []
        self.regime_correlations: Dict[str, CorrelationMatrix] = {}

    def compute_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
        timestamp: int,
        timeframe: str,
        window: Optional[int] = None,
        regime: Optional[str] = None
    ) -> CorrelationMatrix:
        """
        Compute correlation matrix from returns data.

        Args:
            returns_df: DataFrame with asset returns (columns = assets)
            timestamp: Current timestamp
            timeframe: Timeframe
            window: Rolling window (uses default if None)
            regime: Current regime

        Returns:
            Correlation matrix
        """
        window = window or self.rolling_window

        # Take last N rows
        recent_returns = returns_df.tail(window)

        if len(recent_returns) < self.min_samples:
            return self._empty_correlation_matrix(timestamp, timeframe, window, regime)

        # Compute correlation matrix
        corr_matrix = recent_returns.corr()
        asset_list = list(corr_matrix.columns)

        # Extract pairwise correlations
        matrix_data = {}
        correlations = []

        for i, asset1 in enumerate(asset_list):
            for j, asset2 in enumerate(asset_list):
                if i < j:  # Upper triangle only (avoid duplicates)
                    corr_value = corr_matrix.loc[asset1, asset2]
                    matrix_data[(asset1, asset2)] = float(corr_value)
                    correlations.append(abs(corr_value))

        # Statistics
        avg_corr = np.mean(correlations) if correlations else 0.0
        max_corr = np.max(correlations) if correlations else 0.0
        min_corr = np.min(correlations) if correlations else 0.0

        # Determine correlation regime
        if avg_corr > 0.7:
            corr_regime = 'high'
        elif avg_corr < 0.3:
            corr_regime = 'low'
        else:
            corr_regime = 'mixed'

        matrix = CorrelationMatrix(
            timestamp=timestamp,
            timeframe=timeframe,
            window_size=window,
            matrix_data=matrix_data,
            asset_list=asset_list,
            avg_correlation=float(avg_corr),
            max_correlation=float(max_corr),
            min_correlation=float(min_corr),
            regime=regime,
            correlation_regime=corr_regime
        )

        # Store in history
        self.correlation_history.append(matrix)
        if len(self.correlation_history) > 100:
            self.correlation_history.pop(0)

        # Store regime-specific
        if regime:
            self.regime_correlations[regime] = matrix

        return matrix

    def _empty_correlation_matrix(
        self,
        timestamp: int,
        timeframe: str,
        window: int,
        regime: Optional[str]
    ) -> CorrelationMatrix:
        """Create empty correlation matrix"""
        return CorrelationMatrix(
            timestamp=timestamp,
            timeframe=timeframe,
            window_size=window,
            matrix_data={},
            asset_list=[],
            avg_correlation=0.0,
            max_correlation=0.0,
            min_correlation=0.0,
            regime=regime,
            correlation_regime='unknown'
        )

    def detect_correlation_breakdown(
        self,
        asset1: str,
        asset2: str,
        current_corr: float,
        expected_corr: Optional[float] = None
    ) -> Optional[CorrelationSignal]:
        """
        Detect correlation breakdown between two assets.

        Args:
            asset1: First asset
            asset2: Second asset
            current_corr: Current correlation
            expected_corr: Expected correlation (uses known if None)

        Returns:
            Signal if breakdown detected
        """
        # Get expected correlation
        if expected_corr is None:
            pair = (asset1, asset2) if (asset1, asset2) in self.KNOWN_CORRELATIONS else (asset2, asset1)
            expected_corr = self.KNOWN_CORRELATIONS.get(pair, 0.0)

        # Check for breakdown
        breakdown = abs(current_corr - expected_corr) > self.breakdown_threshold

        if breakdown:
            # Determine trading direction
            # If correlation breaks down (becomes less correlated), trade the divergence
            if expected_corr > 0 and current_corr < expected_corr:
                # Positive correlation broke down -> divergence opportunity
                direction = 'bull'  # Buy the laggard
            elif expected_corr < 0 and current_corr > expected_corr:
                # Negative correlation broke down
                direction = 'bear'
            else:
                direction = 'neutral'

            strength = min(abs(current_corr - expected_corr), 1.0)
            confidence = 0.6 + (0.3 * strength)  # Higher breakdown = higher confidence

            return CorrelationSignal(
                signal_type=CorrelationSignalType.BREAKDOWN,
                primary_asset=asset1,
                related_assets=[asset2],
                direction=direction,
                strength=strength,
                confidence=min(confidence, 1.0),
                timestamp=int(datetime.now().timestamp() * 1000),
                correlation_value=current_corr,
                expected_correlation=expected_corr,
                metadata={'breakdown_magnitude': abs(current_corr - expected_corr)}
            )

        return None

    def detect_divergence_opportunities(
        self,
        price_changes: Dict[str, float],
        correlation_matrix: CorrelationMatrix
    ) -> List[CorrelationSignal]:
        """
        Detect divergence opportunities where correlated assets move differently.

        Args:
            price_changes: Recent price changes per asset
            correlation_matrix: Current correlation matrix

        Returns:
            List of divergence signals
        """
        signals: List[CorrelationSignal] = []

        # Look for high correlation pairs with divergent price moves
        for (asset1, asset2), corr_value in correlation_matrix.matrix_data.items():
            if abs(corr_value) < 0.6:  # Not highly correlated
                continue

            if asset1 not in price_changes or asset2 not in price_changes:
                continue

            change1 = price_changes[asset1]
            change2 = price_changes[asset2]

            # Check for divergence
            if corr_value > 0:  # Positive correlation
                # Should move together, but check if they diverge
                if (change1 > 0 and change2 < 0) or (change1 < 0 and change2 > 0):
                    # Divergence detected -> convergence trade
                    strength = min(abs(change1 - change2), 1.0)

                    # Trade direction: buy the laggard
                    primary = asset1 if abs(change1) < abs(change2) else asset2
                    direction = 'bull' if (primary == asset1 and change1 < 0) or (primary == asset2 and change2 < 0) else 'bear'

                    signals.append(CorrelationSignal(
                        signal_type=CorrelationSignalType.DIVERGENCE,
                        primary_asset=primary,
                        related_assets=[asset1 if primary == asset2 else asset2],
                        direction=direction,
                        strength=strength,
                        confidence=0.65,
                        timestamp=correlation_matrix.timestamp,
                        correlation_value=corr_value,
                        expected_correlation=corr_value,
                        metadata={
                            'change1': change1,
                            'change2': change2,
                            'divergence': abs(change1 - change2)
                        }
                    ))

        return signals

    def analyze_basket_strength(
        self,
        basket_name: str,
        price_changes: Dict[str, float],
        correlation_matrix: CorrelationMatrix
    ) -> Optional[CorrelationSignal]:
        """
        Analyze currency basket strength/weakness.

        Args:
            basket_name: Name of basket (e.g., 'USD_majors')
            price_changes: Recent price changes
            correlation_matrix: Current correlation matrix

        Returns:
            Basket signal if detected
        """
        if basket_name not in self.ASSET_GROUPS:
            return None

        basket_assets = self.ASSET_GROUPS[basket_name]

        # Get price changes for basket assets
        basket_changes = [price_changes.get(asset, 0.0) for asset in basket_assets if asset in price_changes]

        if len(basket_changes) < 2:
            return None

        # Calculate basket movement
        avg_change = np.mean(basket_changes)
        consistency = 1.0 - (np.std(basket_changes) / (abs(avg_change) + 0.001))  # High consistency = strong basket move

        # Require significant move and high consistency
        if abs(avg_change) > 0.003 and consistency > 0.7:  # 0.3% move, 70% consistency
            signal_type = CorrelationSignalType.BASKET_STRENGTH if avg_change > 0 else CorrelationSignalType.BASKET_WEAKNESS
            direction = 'bull' if avg_change > 0 else 'bear'
            strength = min(abs(avg_change) * 100, 1.0)
            confidence = consistency

            return CorrelationSignal(
                signal_type=signal_type,
                primary_asset=basket_name,
                related_assets=basket_assets,
                direction=direction,
                strength=strength,
                confidence=confidence,
                timestamp=correlation_matrix.timestamp,
                correlation_value=consistency,
                expected_correlation=0.7,
                metadata={
                    'avg_change': avg_change,
                    'consistency': consistency,
                    'basket_size': len(basket_changes)
                }
            )

        return None

    def assess_portfolio_correlation_risk(
        self,
        open_positions: List[str],
        correlation_matrix: CorrelationMatrix
    ) -> PortfolioCorrelationRisk:
        """
        Assess correlation risk for current portfolio.

        Args:
            open_positions: List of assets with open positions
            correlation_matrix: Current correlation matrix

        Returns:
            Portfolio correlation risk assessment
        """
        if len(open_positions) < 2:
            return PortfolioCorrelationRisk(
                timestamp=correlation_matrix.timestamp,
                positions=open_positions,
                avg_correlation=0.0,
                max_correlation=0.0,
                correlation_score=1.0,
                systemic_risk_level='low',
                recommended_action=None
            )

        # Extract correlations between open positions
        position_correlations = []
        max_corr = 0.0

        for i, asset1 in enumerate(open_positions):
            for j, asset2 in enumerate(open_positions):
                if i < j:
                    pair = (asset1, asset2)
                    reverse_pair = (asset2, asset1)

                    corr = correlation_matrix.matrix_data.get(pair) or correlation_matrix.matrix_data.get(reverse_pair)

                    if corr is not None:
                        position_correlations.append(abs(corr))
                        max_corr = max(max_corr, abs(corr))

        # Calculate metrics
        avg_corr = np.mean(position_correlations) if position_correlations else 0.0

        # Correlation score: 1.0 = fully diversified, 0.0 = fully correlated
        correlation_score = 1.0 - avg_corr

        # Determine risk level
        if max_corr > 0.9 or avg_corr > 0.8:
            risk_level = 'critical'
            action = 'close_correlated_positions'
        elif max_corr > 0.8 or avg_corr > self.systemic_risk_threshold:
            risk_level = 'high'
            action = 'reduce_exposure'
        elif max_corr > 0.7 or avg_corr > 0.6:
            risk_level = 'medium'
            action = 'monitor_closely'
        else:
            risk_level = 'low'
            action = None

        return PortfolioCorrelationRisk(
            timestamp=correlation_matrix.timestamp,
            positions=open_positions,
            avg_correlation=float(avg_corr),
            max_correlation=float(max_corr),
            correlation_score=float(correlation_score),
            systemic_risk_level=risk_level,
            recommended_action=action
        )

    def generate_correlation_safety_score(
        self,
        proposed_asset: str,
        open_positions: List[str],
        correlation_matrix: CorrelationMatrix
    ) -> float:
        """
        Generate correlation safety score for a proposed new position.

        Args:
            proposed_asset: Asset being considered
            open_positions: Current open positions
            correlation_matrix: Current correlation matrix

        Returns:
            Safety score (0-1, higher = safer/more diversified)
        """
        if not open_positions:
            return 1.0  # No existing positions, fully safe

        correlations = []

        for existing_asset in open_positions:
            pair = (proposed_asset, existing_asset)
            reverse_pair = (existing_asset, proposed_asset)

            corr = correlation_matrix.matrix_data.get(pair) or correlation_matrix.matrix_data.get(reverse_pair)

            if corr is not None:
                correlations.append(abs(corr))

        if not correlations:
            return 0.8  # No correlation data, assume moderately safe

        # Calculate safety: inverse of average correlation
        avg_corr = np.mean(correlations)
        max_corr = np.max(correlations)

        # Safety = 1.0 - weighted combination of avg and max
        safety = 1.0 - (0.6 * avg_corr + 0.4 * max_corr)

        return np.clip(safety, 0.0, 1.0)
