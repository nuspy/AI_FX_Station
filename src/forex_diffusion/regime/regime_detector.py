"""
Unsupervised Machine Learning Regime Detection System.

Implements clustering-based regime detection using logical groups of technical indicators
for each timeframe. Provides automated regime classification without manual labeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..cache import cache_decorator


@dataclass
class RegimeState:
    """Current market regime state"""
    regime_id: int
    regime_name: str
    confidence: float
    dominant_factors: List[str]
    timeframe: str
    timestamp: datetime
    characteristics: Dict[str, float]


@dataclass
class IndicatorGroup:
    """Group of related technical indicators"""
    name: str
    title: str
    indicators: List[str]
    weight: float
    values: Dict[str, float]


class TechnicalIndicatorCalculator:
    """
    Calculates technical indicators for regime detection.

    Organized into logical groups as specified:
    1. Trend & Momentum Regime
    2. Volatility Regime
    3. Market Structure Regime
    4. Market Sentiment Regime
    """

    def __init__(self):
        self.indicators_cache = {}

    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, IndicatorGroup]:
        """Calculate all indicator groups for regime detection"""
        if data.empty or len(data) < 50:
            logger.warning("Insufficient data for regime detection")
            return {}

        try:
            groups = {}

            # 1. Trend & Momentum Regime
            groups['trend_momentum'] = self._calculate_trend_momentum_group(data)

            # 2. Volatility Regime
            groups['volatility_regime'] = self._calculate_volatility_group(data)

            # 3. Market Structure Regime
            groups['market_structure'] = self._calculate_market_structure_group(data)

            # 4. Market Sentiment Regime
            groups['sentiment_flow'] = self._calculate_sentiment_group(data)

            return groups

        except Exception as e:
            logger.error(f"Error calculating regime indicators: {e}")
            return {}

    def _calculate_trend_momentum_group(self, data: pd.DataFrame) -> IndicatorGroup:
        """Calculate Trend & Momentum indicators"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']

            # SMA cross indicator (20/50)
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            sma_cross = (sma_20 / sma_50 - 1.0).iloc[-1] if len(sma_20) > 50 else 0.0

            # RSI (14-period)
            rsi_14 = self._calculate_rsi(close, 14).iloc[-1] if len(close) > 14 else 50.0

            # MACD Signal
            macd_line, macd_signal, _ = self._calculate_macd(close)
            macd_signal_val = (macd_line.iloc[-1] - macd_signal.iloc[-1]) if len(macd_line) > 26 else 0.0

            # ADX (14-period) - Trend Strength
            adx_14 = self._calculate_adx(high, low, close, 14).iloc[-1] if len(close) > 14 else 25.0

            return IndicatorGroup(
                name="trend_momentum",
                title="Trend & Momentum Regime",
                indicators=["sma_20_50_cross", "rsi_14", "macd_signal", "adx_14"],
                weight=0.3,
                values={
                    "sma_20_50_cross": sma_cross,
                    "rsi_14": (rsi_14 - 50.0) / 50.0,  # Normalize -1 to 1
                    "macd_signal": macd_signal_val,
                    "adx_14": (adx_14 - 25.0) / 75.0  # Normalize around neutral
                }
            )

        except Exception as e:
            logger.debug(f"Error calculating trend momentum indicators: {e}")
            return self._get_default_group("trend_momentum", "Trend & Momentum Regime", 0.3)

    def _calculate_volatility_group(self, data: pd.DataFrame) -> IndicatorGroup:
        """Calculate Volatility regime indicators"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']

            # ATR (14-period)
            atr_14 = self._calculate_atr(high, low, close, 14)
            atr_normalized = (atr_14.iloc[-1] / close.iloc[-1]) if len(atr_14) > 14 else 0.02

            # Bollinger Band Width
            bb_upper, bb_lower = self._calculate_bollinger_bands(close, 20, 2.0)
            bb_width = ((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / close.iloc[-1]) if len(bb_upper) > 20 else 0.04

            # Price Range Ratio (High-Low/Close)
            price_range_ratio = ((high - low) / close).rolling(10).mean().iloc[-1] if len(close) > 10 else 0.02

            # GARCH-style volatility estimate
            returns = close.pct_change().dropna()
            garch_vol = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else 0.01

            return IndicatorGroup(
                name="volatility_regime",
                title="Volatility Regime",
                indicators=["atr_14", "bollinger_bandwidth", "price_range_ratio", "garch_volatility"],
                weight=0.25,
                values={
                    "atr_14": min(atr_normalized * 50, 1.0),  # Scale to 0-1
                    "bollinger_bandwidth": min(bb_width * 25, 1.0),
                    "price_range_ratio": min(price_range_ratio * 50, 1.0),
                    "garch_volatility": min(garch_vol * 100, 1.0)
                }
            )

        except Exception as e:
            logger.debug(f"Error calculating volatility indicators: {e}")
            return self._get_default_group("volatility_regime", "Volatility Regime", 0.25)

    def _calculate_market_structure_group(self, data: pd.DataFrame) -> IndicatorGroup:
        """Calculate Market Structure indicators"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']

            # Support/Resistance Strength (pivot point analysis)
            sr_strength = self._calculate_support_resistance_strength(high, low, close)

            # Fractal Dimension (market efficiency measure)
            fractal_dim = self._calculate_fractal_dimension(close)

            # Hurst Exponent (trend persistence)
            hurst_exp = self._calculate_hurst_exponent(close)

            # Market Efficiency Ratio
            efficiency_ratio = self._calculate_market_efficiency_ratio(close)

            return IndicatorGroup(
                name="market_structure",
                title="Market Structure Regime",
                indicators=["support_resistance_strength", "fractal_dimension", "hurst_exponent", "market_efficiency_ratio"],
                weight=0.25,
                values={
                    "support_resistance_strength": sr_strength,
                    "fractal_dimension": fractal_dim,
                    "hurst_exponent": hurst_exp,
                    "market_efficiency_ratio": efficiency_ratio
                }
            )

        except Exception as e:
            logger.debug(f"Error calculating market structure indicators: {e}")
            return self._get_default_group("market_structure", "Market Structure Regime", 0.25)

    def _calculate_sentiment_group(self, data: pd.DataFrame) -> IndicatorGroup:
        """Calculate Market Sentiment indicators"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']

            # Price-Momentum Divergence
            momentum_div = self._calculate_momentum_divergence(close, high, low)

            # Breakout Failure Rate
            breakout_failure = self._calculate_breakout_failure_rate(high, low, close)

            # Gap Analysis
            gap_frequency = self._calculate_gap_analysis(close)

            # Reversal Pattern Frequency
            reversal_freq = self._calculate_reversal_pattern_frequency(close, high, low)

            return IndicatorGroup(
                name="sentiment_flow",
                title="Market Sentiment Regime",
                indicators=["price_momentum_divergence", "breakout_failure_rate", "gap_analysis", "reversal_pattern_frequency"],
                weight=0.2,
                values={
                    "price_momentum_divergence": momentum_div,
                    "breakout_failure_rate": breakout_failure,
                    "gap_analysis": gap_frequency,
                    "reversal_pattern_frequency": reversal_freq
                }
            )

        except Exception as e:
            logger.debug(f"Error calculating sentiment indicators: {e}")
            return self._get_default_group("sentiment_flow", "Market Sentiment Regime", 0.2)

    # Technical Indicator Calculation Methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ADX"""
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def _calculate_support_resistance_strength(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate support/resistance strength using pivot points"""
        try:
            # Simple pivot point analysis
            recent_high = high.rolling(20).max().iloc[-1]
            recent_low = low.rolling(20).min().iloc[-1]
            current_price = close.iloc[-1]

            # Measure how often price tests and holds key levels
            touches_high = (abs(high.rolling(5).max() - recent_high) / recent_high < 0.002).sum()
            touches_low = (abs(low.rolling(5).min() - recent_low) / recent_low < 0.002).sum()

            # Normalize strength indicator
            strength = (touches_high + touches_low) / 40.0  # Normalize to 0-1
            return min(strength, 1.0)

        except Exception:
            return 0.5

    def _calculate_fractal_dimension(self, prices: pd.Series) -> float:
        """Calculate fractal dimension of price series"""
        try:
            # Simplified fractal dimension using Higuchi method
            n = min(len(prices), 50)
            if n < 10:
                return 1.5  # Default

            price_array = prices.tail(n).values

            # Calculate lengths for different k values
            lengths = []
            for k in range(1, min(6, n//2)):
                length = 0
                for i in range(k):
                    indices = range(i, n, k)
                    if len(indices) > 1:
                        sub_series = price_array[indices]
                        length += np.sum(np.abs(np.diff(sub_series)))

                if length > 0:
                    lengths.append(length / ((n-1)//k))

            if len(lengths) < 2:
                return 1.5

            # Estimate fractal dimension
            k_values = range(1, len(lengths) + 1)
            if np.var(np.log(k_values)) > 0:
                slope = np.cov(np.log(k_values), np.log(lengths))[0,1] / np.var(np.log(k_values))
                fractal_dim = -slope
                return max(1.0, min(2.0, fractal_dim))

            return 1.5

        except Exception:
            return 1.5

    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """Calculate Hurst exponent for trend persistence"""
        try:
            n = min(len(prices), 100)
            if n < 20:
                return 0.5

            price_array = prices.tail(n).values

            # Calculate log returns
            returns = np.diff(np.log(price_array))

            # Calculate R/S statistic
            rs_values = []
            for i in range(10, n//2, 5):
                if i >= len(returns):
                    break

                sub_returns = returns[:i]
                mean_return = np.mean(sub_returns)
                cumulative_devs = np.cumsum(sub_returns - mean_return)

                r = np.max(cumulative_devs) - np.min(cumulative_devs)
                s = np.std(sub_returns)

                if s > 0:
                    rs_values.append(r / s)

            if len(rs_values) < 3:
                return 0.5

            # Estimate Hurst exponent
            periods = range(10, 10 + len(rs_values) * 5, 5)
            log_rs = np.log(rs_values)
            log_periods = np.log(periods[:len(rs_values)])

            if np.var(log_periods) > 0:
                hurst = np.cov(log_periods, log_rs)[0,1] / np.var(log_periods)
                return max(0.0, min(1.0, hurst))

            return 0.5

        except Exception:
            return 0.5

    def _calculate_market_efficiency_ratio(self, prices: pd.Series) -> float:
        """Calculate market efficiency ratio"""
        try:
            if len(prices) < 20:
                return 0.5

            # Direction (net change)
            direction = abs(prices.iloc[-1] - prices.iloc[-20])

            # Volatility (sum of absolute changes)
            volatility = abs(prices.diff()).tail(19).sum()

            # Efficiency ratio
            if volatility > 0:
                efficiency = direction / volatility
                return min(efficiency, 1.0)

            return 0.5

        except Exception:
            return 0.5

    def _calculate_momentum_divergence(self, close: pd.Series, high: pd.Series, low: pd.Series) -> float:
        """Calculate price-momentum divergence"""
        try:
            if len(close) < 20:
                return 0.0

            # Price trend (20-period)
            price_trend = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]

            # Momentum trend (RSI change)
            rsi = self._calculate_rsi(close, 14)
            if len(rsi) > 20:
                momentum_trend = (rsi.iloc[-1] - rsi.iloc[-20]) / 100.0

                # Divergence = difference in direction
                divergence = abs(np.sign(price_trend) - np.sign(momentum_trend)) / 2.0
                return min(divergence, 1.0)

            return 0.0

        except Exception:
            return 0.0

    def _calculate_breakout_failure_rate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate breakout failure rate"""
        try:
            if len(close) < 30:
                return 0.5

            # Identify potential breakouts (price exceeding 20-period high/low)
            breakouts = 0
            failures = 0

            for i in range(20, len(close) - 5):
                recent_high = high.iloc[i-20:i].max()
                recent_low = low.iloc[i-20:i].min()

                # Check for upward breakout
                if high.iloc[i] > recent_high:
                    breakouts += 1
                    # Check if price fell back within range in next 5 periods
                    if close.iloc[i+1:i+6].min() <= recent_high:
                        failures += 1

                # Check for downward breakout
                elif low.iloc[i] < recent_low:
                    breakouts += 1
                    # Check if price rose back within range in next 5 periods
                    if close.iloc[i+1:i+6].max() >= recent_low:
                        failures += 1

            if breakouts > 0:
                failure_rate = failures / breakouts
                return min(failure_rate, 1.0)

            return 0.5

        except Exception:
            return 0.5

    def _calculate_gap_analysis(self, close: pd.Series) -> float:
        """Calculate gap frequency and magnitude"""
        try:
            if len(close) < 10:
                return 0.0

            # Calculate gaps (significant overnight moves)
            price_changes = close.pct_change().abs()

            # Define gap as move > 2 standard deviations
            std_threshold = price_changes.rolling(20).std() * 2
            gaps = (price_changes > std_threshold).sum()

            # Normalize gap frequency
            gap_frequency = gaps / len(price_changes)
            return min(gap_frequency * 10, 1.0)  # Scale up for visibility

        except Exception:
            return 0.0

    def _calculate_reversal_pattern_frequency(self, close: pd.Series, high: pd.Series, low: pd.Series) -> float:
        """Calculate reversal vs continuation pattern frequency"""
        try:
            if len(close) < 30:
                return 0.5

            reversals = 0
            total_patterns = 0

            # Simple pattern detection: look for local highs/lows
            for i in range(10, len(close) - 10):
                # Check if current point is local high
                if (high.iloc[i] == high.iloc[i-5:i+6].max()):
                    total_patterns += 1
                    # Check if price reversed (fell significantly after)
                    if close.iloc[i+5] < close.iloc[i] * 0.98:
                        reversals += 1

                # Check if current point is local low
                elif (low.iloc[i] == low.iloc[i-5:i+6].min()):
                    total_patterns += 1
                    # Check if price reversed (rose significantly after)
                    if close.iloc[i+5] > close.iloc[i] * 1.02:
                        reversals += 1

            if total_patterns > 0:
                reversal_rate = reversals / total_patterns
                return reversal_rate

            return 0.5

        except Exception:
            return 0.5

    def _get_default_group(self, name: str, title: str, weight: float) -> IndicatorGroup:
        """Get default indicator group when calculation fails"""
        return IndicatorGroup(
            name=name,
            title=title,
            indicators=[],
            weight=weight,
            values={}
        )


class RegimeDetector:
    """
    Unsupervised Machine Learning Regime Detector.

    Uses clustering algorithms to identify market regimes based on
    technical indicator groups without manual labeling.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('regime_detection', {})
        self.enabled = self.config.get('enabled', True)

        # Clustering configuration
        clustering_config = self.config.get('clustering', {})
        self.n_clusters = clustering_config.get('n_clusters', 5)
        self.algorithm = clustering_config.get('algorithm', 'kmeans')
        self.lookback_periods = clustering_config.get('lookback_periods', 100)

        # Components
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.scalers: Dict[str, StandardScaler] = {}
        self.models: Dict[str, Any] = {}
        self.regime_labels: Dict[str, List[str]] = {}

    @cache_decorator('regime_detection', ttl=3600)
    def detect_regime(self, data: pd.DataFrame, asset: str, timeframe: str) -> Optional[RegimeState]:
        """
        Detect current market regime for given asset and timeframe.

        Args:
            data: OHLC price data
            asset: Asset symbol
            timeframe: Timeframe string

        Returns:
            RegimeState with current regime information
        """
        if not self.enabled or data.empty:
            return None

        try:
            # Calculate indicator groups
            indicator_groups = self.indicator_calculator.calculate_all_indicators(data)

            if not indicator_groups:
                return None

            # Prepare feature matrix
            features = self._prepare_features(indicator_groups)

            if features is None:
                return None

            # Get or create model for this timeframe
            model_key = f"{asset}_{timeframe}"

            if model_key not in self.models:
                # Train new model with historical data
                historical_features = self._prepare_historical_features(data, asset, timeframe)
                if historical_features is not None:
                    self._train_model(historical_features, model_key)

            # Predict regime
            if model_key in self.models:
                regime_id, confidence = self._predict_regime(features, model_key)

                # Generate regime characteristics
                regime_name = self._get_regime_name(regime_id, indicator_groups)
                dominant_factors = self._get_dominant_factors(features, indicator_groups)
                characteristics = self._get_regime_characteristics(indicator_groups)

                return RegimeState(
                    regime_id=regime_id,
                    regime_name=regime_name,
                    confidence=confidence,
                    dominant_factors=dominant_factors,
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    characteristics=characteristics
                )

            return None

        except Exception as e:
            logger.error(f"Error detecting regime for {asset}/{timeframe}: {e}")
            return None

    def _prepare_features(self, indicator_groups: Dict[str, IndicatorGroup]) -> Optional[np.ndarray]:
        """Prepare feature vector from indicator groups"""
        try:
            features = []

            for group_name, group in indicator_groups.items():
                # Apply group weight and add all indicator values
                for indicator_name, value in group.values.items():
                    weighted_value = value * group.weight
                    features.append(weighted_value)

            if features:
                return np.array(features).reshape(1, -1)

            return None

        except Exception as e:
            logger.debug(f"Error preparing features: {e}")
            return None

    def _prepare_historical_features(self, data: pd.DataFrame, asset: str, timeframe: str) -> Optional[np.ndarray]:
        """Prepare historical feature matrix for model training"""
        try:
            if len(data) < self.lookback_periods + 50:
                return None

            features_list = []

            # Generate features for sliding windows
            for i in range(50, len(data) - self.lookback_periods, 5):  # Every 5 periods
                window_data = data.iloc[i:i+self.lookback_periods]

                if len(window_data) >= 50:  # Minimum data for indicators
                    indicator_groups = self.indicator_calculator.calculate_all_indicators(window_data)

                    if indicator_groups:
                        features = self._prepare_features(indicator_groups)
                        if features is not None:
                            features_list.append(features.flatten())

            if len(features_list) >= self.n_clusters:
                return np.array(features_list)

            return None

        except Exception as e:
            logger.debug(f"Error preparing historical features: {e}")
            return None

    def _train_model(self, features: np.ndarray, model_key: str):
        """Train clustering model on historical features"""
        try:
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # Store scaler
            self.scalers[model_key] = scaler

            # Train clustering model
            if self.algorithm == 'kmeans':
                model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            elif self.algorithm == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
            elif self.algorithm == 'gaussian_mixture':
                model = GaussianMixture(n_components=self.n_clusters, random_state=42)
            else:
                model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)

            # Fit model
            model.fit(scaled_features)

            # Store model
            self.models[model_key] = model

            # Generate regime labels for interpretation
            labels = model.predict(scaled_features) if hasattr(model, 'predict') else model.labels_
            self.regime_labels[model_key] = self._generate_regime_labels(labels, features, model_key)

            logger.info(f"Regime model trained for {model_key}: {self.n_clusters} regimes identified")

        except Exception as e:
            logger.error(f"Error training regime model for {model_key}: {e}")

    def _predict_regime(self, features: np.ndarray, model_key: str) -> Tuple[int, float]:
        """Predict current regime and confidence"""
        try:
            model = self.models[model_key]
            scaler = self.scalers[model_key]

            # Scale features
            scaled_features = scaler.transform(features)

            # Predict regime
            if hasattr(model, 'predict'):
                regime_id = model.predict(scaled_features)[0]

                # Calculate confidence based on distance to cluster center
                if hasattr(model, 'cluster_centers_'):
                    distances = np.linalg.norm(scaled_features - model.cluster_centers_, axis=1)
                    min_distance = np.min(distances)
                    max_distance = np.max(distances)

                    # Convert distance to confidence (closer = higher confidence)
                    if max_distance > min_distance:
                        confidence = 1.0 - (min_distance / max_distance)
                    else:
                        confidence = 0.8
                else:
                    confidence = 0.7  # Default confidence for models without centers

            else:
                # For models without predict method (like DBSCAN)
                regime_id = 0
                confidence = 0.5

            return int(regime_id), float(confidence)

        except Exception as e:
            logger.debug(f"Error predicting regime: {e}")
            return 0, 0.5

    def _generate_regime_labels(self, labels: np.ndarray, features: np.ndarray, model_key: str) -> List[str]:
        """Generate interpretable regime labels based on cluster characteristics"""
        try:
            regime_names = []

            for regime_id in range(self.n_clusters):
                mask = labels == regime_id
                if not np.any(mask):
                    regime_names.append(f"Regime_{regime_id}")
                    continue

                # Analyze cluster characteristics
                cluster_features = features[mask]
                feature_means = np.mean(cluster_features, axis=0)

                # Interpret based on feature values (simplified)
                # This would be more sophisticated with proper feature mapping
                if len(feature_means) >= 4:
                    trend_strength = feature_means[0]  # First feature (trend momentum)
                    volatility = feature_means[1] if len(feature_means) > 1 else 0

                    if trend_strength > 0.3 and volatility < 0.3:
                        name = "Strong_Trend_Low_Vol"
                    elif trend_strength > 0.3 and volatility > 0.7:
                        name = "Strong_Trend_High_Vol"
                    elif abs(trend_strength) < 0.2 and volatility < 0.3:
                        name = "Consolidation_Low_Vol"
                    elif abs(trend_strength) < 0.2 and volatility > 0.7:
                        name = "Consolidation_High_Vol"
                    else:
                        name = f"Mixed_Regime_{regime_id}"
                else:
                    name = f"Regime_{regime_id}"

                regime_names.append(name)

            return regime_names

        except Exception as e:
            logger.debug(f"Error generating regime labels: {e}")
            return [f"Regime_{i}" for i in range(self.n_clusters)]

    def _get_regime_name(self, regime_id: int, indicator_groups: Dict[str, IndicatorGroup]) -> str:
        """Get human-readable regime name"""
        try:
            # Use cached labels if available
            for model_key, labels in self.regime_labels.items():
                if regime_id < len(labels):
                    return labels[regime_id]

            # Fallback: generate based on current indicators
            dominant_factor = self._get_dominant_factors(None, indicator_groups)[0] if indicator_groups else "Unknown"
            return f"{dominant_factor}_Regime_{regime_id}"

        except Exception:
            return f"Regime_{regime_id}"

    def _get_dominant_factors(self, features: Optional[np.ndarray], indicator_groups: Dict[str, IndicatorGroup]) -> List[str]:
        """Identify dominant factors in current regime"""
        try:
            factor_scores = {}

            for group_name, group in indicator_groups.items():
                # Calculate group strength based on indicator values
                group_strength = 0.0
                for indicator_name, value in group.values.items():
                    group_strength += abs(value) * group.weight

                factor_scores[group.title] = group_strength

            # Sort by strength and return top factors
            sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
            return [factor[0] for factor in sorted_factors[:3]]

        except Exception:
            return ["Market_Structure", "Volatility", "Trend"]

    def _get_regime_characteristics(self, indicator_groups: Dict[str, IndicatorGroup]) -> Dict[str, float]:
        """Get detailed regime characteristics"""
        characteristics = {}

        for group_name, group in indicator_groups.items():
            for indicator_name, value in group.values.items():
                characteristics[f"{group_name}_{indicator_name}"] = value

        return characteristics

    def get_regime_summary(self, regime_state: RegimeState) -> Dict[str, Any]:
        """Get formatted regime summary for display"""
        return {
            'regime_name': regime_state.regime_name,
            'confidence': f"{regime_state.confidence:.1%}",
            'timeframe': regime_state.timeframe,
            'dominant_factors': regime_state.dominant_factors,
            'characteristics_summary': {
                'Trend_Strength': regime_state.characteristics.get('trend_momentum_adx_14', 0.0),
                'Volatility_Level': regime_state.characteristics.get('volatility_regime_atr_14', 0.0),
                'Market_Efficiency': regime_state.characteristics.get('market_structure_market_efficiency_ratio', 0.0),
                'Sentiment_Bias': regime_state.characteristics.get('sentiment_flow_price_momentum_divergence', 0.0)
            }
        }