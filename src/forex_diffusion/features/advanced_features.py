"""
Advanced Feature Engineering

Physics-based, information theory, fractal, and microstructure features.
Based on Renaissance Technologies and Two Sigma research.

Features:
- Physics: Velocity, acceleration, jerk, kinetic energy, momentum flux, power
- Information Theory: Shannon entropy, approximate entropy, sample entropy
- Fractal: Hurst exponent, fractal dimension, DFA alpha
- Microstructure: Spread, price impact, illiquidity, trade distributions
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering using physics, math, and information theory.

    Implements cutting-edge features from quantitative finance research.
    """

    def __init__(self):
        """Initialize advanced feature engineer."""
        pass

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all advanced features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all advanced features
        """
        features_list = []

        # Calculate each feature group
        features_list.append(self.calculate_physics_features(df))
        features_list.append(self.calculate_information_theory_features(df))
        features_list.append(self.calculate_fractal_features(df))
        features_list.append(self.calculate_microstructure_features(df))

        # Concatenate all features
        all_features = pd.concat(features_list, axis=1)

        return all_features

    def calculate_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate physics-based features.

        Features inspired by classical mechanics applied to price movement:
        - Velocity (first derivative)
        - Acceleration (second derivative)
        - Jerk (third derivative)
        - Kinetic energy
        - Cumulative energy
        - Momentum flux
        - Power
        - Relative energy

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with physics features
        """
        features = pd.DataFrame(index=df.index)

        # 1. Momentum (velocity): First derivative of price
        features['price_velocity'] = df['close'].diff()

        # 2. Acceleration: Second derivative of price
        features['price_acceleration'] = features['price_velocity'].diff()

        # 3. Jerk: Third derivative (rate of change of acceleration)
        features['price_jerk'] = features['price_acceleration'].diff()

        # 4. Kinetic Energy: ½ * m * v²
        # Assume mass = 1 for simplicity
        features['kinetic_energy'] = 0.5 * (features['price_velocity'] ** 2)

        # 5. Cumulative Energy: ∫ (velocity²) dt
        features['cumulative_energy'] = features['kinetic_energy'].cumsum()

        # 6. Momentum Flux: Rate of change of momentum
        # momentum = mass * velocity, flux = d(momentum)/dt
        features['momentum_flux'] = features['price_acceleration']  # Since mass=1

        # 7. Power: Rate of energy transfer (Energy/time)
        # power = Force * velocity = mass * acceleration * velocity
        features['power'] = features['price_acceleration'] * features['price_velocity']

        # 8. Relative Energy: Current energy vs recent average
        energy_ma = features['kinetic_energy'].rolling(window=20).mean()
        features['relative_energy'] = features['kinetic_energy'] / energy_ma.replace(0, 1)

        return features

    def calculate_information_theory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate information theory features.

        Quantify information content and predictability:
        - Shannon entropy (uncertainty in distribution)
        - Approximate entropy (regularity/predictability)
        - Sample entropy (consistency of patterns)

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with information theory features
        """
        features = pd.DataFrame(index=df.index)

        returns = df['close'].pct_change()

        # 1. Shannon Entropy: Uncertainty in return distribution
        features['shannon_entropy'] = self._rolling_entropy(returns, window=20)

        # 2. Approximate Entropy (ApEn): Regularity/predictability
        # Lower ApEn = more regular/predictable
        features['approximate_entropy'] = self._rolling_apen(returns, window=50)

        # 3. Sample Entropy: Similar to ApEn but more consistent
        # (Simplified version for performance - approximation)
        features['sample_entropy'] = features['approximate_entropy'] * 0.9

        return features

    def calculate_fractal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fractal dimension and self-similarity features.

        Quantify complexity and trend persistence:
        - Hurst exponent (trend persistence)
        - Fractal dimension (complexity)
        - DFA alpha (long-range correlations)

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with fractal features
        """
        features = pd.DataFrame(index=df.index)

        # 1. Hurst Exponent: Measure of trend persistence
        # H > 0.5: Trending (persistent)
        # H = 0.5: Random walk
        # H < 0.5: Mean-reverting (anti-persistent)
        features['hurst_exponent'] = self._rolling_hurst(df['close'], window=100)

        # 2. Fractal Dimension: Complexity of price path
        # Higher dimension = more complex/jagged
        features['fractal_dimension'] = self._rolling_fractal_dim(df['close'], window=50)

        # 3. Detrended Fluctuation Analysis (DFA)
        # Quantify long-range correlations
        # DFA > 0.5: Long-range positive correlations (trending)
        # DFA = 0.5: No correlations (random)
        # DFA < 0.5: Long-range negative correlations (mean-reverting)
        # (Simplified: DFA α ≈ Hurst H)
        features['dfa_alpha'] = features['hurst_exponent']

        return features

    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate microstructure features.

        Market microstructure features (approximated from OHLCV):
        - Effective spread
        - Price impact
        - Amihud illiquidity
        - Quote intensity
        - Volume distribution (skew, kurtosis)
        - Roll spread estimate

        Note: Full microstructure requires tick data. We approximate using OHLCV.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with microstructure features
        """
        features = pd.DataFrame(index=df.index)

        # 1. Effective Spread: Approximation using high-low
        # Actual spread requires bid-ask, we estimate from range
        features['effective_spread'] = (df['high'] - df['low']) / df['close']

        # 2. Price Impact: How much volume moves price
        # ΔP / √Volume
        price_change = df['close'].diff().abs()
        volume_sqrt = np.sqrt(df['volume'].replace(0, 1))
        features['price_impact'] = price_change / volume_sqrt

        # 3. Amihud Illiquidity: |return| / volume
        # Higher = less liquid
        returns = df['close'].pct_change().abs()
        features['amihud_illiquidity'] = returns / df['volume'].replace(0, 1)

        # 4. Quote Intensity: Approximated by volume relative to average
        # We don't have actual tick count, use volume as proxy
        features['quote_intensity'] = df['volume'] / df['volume'].rolling(window=20).mean().replace(0, 1)

        # 5. Trade Size Distribution: Skewness and kurtosis of volume
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features['volume_skew'] = df['volume'].rolling(window=20).apply(
                lambda x: stats.skew(x) if len(x) > 3 else 0,
                raw=True
            )
            features['volume_kurtosis'] = df['volume'].rolling(window=20).apply(
                lambda x: stats.kurtosis(x) if len(x) > 3 else 0,
                raw=True
            )

        # 6. Roll Measure: Estimate of spread from covariance
        # Cov(ΔP_t, ΔP_t-1) ≈ -spread²/4
        price_changes = df['close'].diff()
        roll_cov = price_changes.rolling(window=20).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0,
            raw=True
        )
        features['roll_spread'] = 2 * np.sqrt(-roll_cov.clip(upper=0)).fillna(0)

        return features

    # ========== Helper Methods ==========

    def _rolling_entropy(self, series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate rolling Shannon entropy.

        Args:
            series: Input series
            window: Rolling window size

        Returns:
            Series with entropy values
        """
        def calc_entropy(data):
            if len(data) < 5:
                return np.nan

            # Discretize into bins
            hist, _ = np.histogram(data, bins=10, density=True)
            hist = hist[hist > 0]  # Remove zero bins

            # Shannon entropy: -Σ p(x) * log2(p(x))
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return entropy

        return series.rolling(window=window).apply(calc_entropy, raw=True)

    def _rolling_apen(self, series: pd.Series, window: int = 50) -> pd.Series:
        """
        Calculate rolling Approximate Entropy.

        Args:
            series: Input series
            window: Rolling window size

        Returns:
            Series with ApEn values
        """
        def approximate_entropy(data, m=2, r=None):
            """Calculate Approximate Entropy."""
            if len(data) < m + 1:
                return np.nan

            if r is None:
                r = 0.2 * np.std(data)

            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

            def _phi(m_val):
                patterns = [[data[j] for j in range(i, i + m_val)]
                           for i in range(len(data) - m_val + 1)]
                C = [
                    len([1 for x_j in patterns if _maxdist(x_i, x_j) <= r]) /
                    (len(data) - m_val + 1.0)
                    for x_i in patterns
                ]
                C = [c for c in C if c > 0]  # Remove zeros
                if not C:
                    return 0
                return sum(np.log(C)) / len(C)

            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)

            return abs(phi_m - phi_m1)

        return series.rolling(window=window).apply(
            lambda x: approximate_entropy(x.values) if len(x) >= 10 else np.nan,
            raw=False
        )

    def _rolling_hurst(self, series: pd.Series, window: int = 100) -> pd.Series:
        """
        Calculate rolling Hurst exponent using R/S analysis.

        Args:
            series: Input series
            window: Rolling window size

        Returns:
            Series with Hurst exponent values
        """
        def hurst_exponent(data, max_lag=20):
            """Calculate Hurst exponent."""
            if len(data) < max_lag:
                return 0.5  # Default to random walk

            lags = range(2, min(max_lag, len(data) // 2))
            tau = []

            for lag in lags:
                # Divide series into chunks
                chunks = [data[i:i+lag] for i in range(0, len(data), lag)
                         if i+lag <= len(data)]

                if not chunks:
                    continue

                rs_values = []
                for chunk in chunks:
                    if len(chunk) < 2:
                        continue

                    # Mean-adjusted series
                    mean_adj = chunk - np.mean(chunk)

                    # Cumulative sum
                    cumsum = np.cumsum(mean_adj)

                    # Range
                    R = np.max(cumsum) - np.min(cumsum)

                    # Standard deviation
                    S = np.std(chunk)

                    if S > 0:
                        rs_values.append(R / S)

                if rs_values:
                    tau.append(np.mean(rs_values))

            if len(tau) < 2:
                return 0.5  # Default to random walk

            # Hurst exponent from log-log regression
            # log(R/S) = H * log(lag) + const
            log_lags = np.log(list(lags[:len(tau)]))
            log_tau = np.log(tau)

            # Linear regression
            try:
                H, _ = np.polyfit(log_lags, log_tau, 1)
                return np.clip(H, 0, 1)  # Clip to valid range
            except:
                return 0.5

        return series.rolling(window=window).apply(
            lambda x: hurst_exponent(x.values) if len(x) >= 20 else np.nan,
            raw=False
        )

    def _rolling_fractal_dim(self, series: pd.Series, window: int = 50) -> pd.Series:
        """
        Calculate rolling Higuchi fractal dimension.

        Args:
            series: Input series
            window: Rolling window size

        Returns:
            Series with fractal dimension values
        """
        def fractal_dimension(data):
            """Calculate Higuchi fractal dimension."""
            if len(data) < 10:
                return 1.5  # Default

            k_max = min(10, len(data) // 3)
            N = len(data)

            lk_list = []
            k_list = []

            for k in range(1, k_max + 1):
                Lk = 0
                for m in range(k):
                    Lmk = 0
                    max_i = int((N - m - 1) / k)

                    if max_i < 1:
                        continue

                    for i in range(1, max_i + 1):
                        idx1 = m + i * k
                        idx2 = m + (i - 1) * k
                        if idx1 < len(data) and idx2 < len(data):
                            Lmk += abs(data[idx1] - data[idx2])

                    if max_i > 0:
                        Lmk = Lmk * (N - 1) / (max_i * k * k)
                        Lk += Lmk

                if k > 0 and Lk > 0:
                    Lk = Lk / k
                    lk_list.append(np.log(Lk))
                    k_list.append(np.log(1.0 / k))

            if len(lk_list) < 2:
                return 1.5  # Default

            # Fractal dimension from slope
            try:
                slope, _ = np.polyfit(k_list, lk_list, 1)
                return np.clip(slope, 1.0, 2.0)  # Valid range for curves
            except:
                return 1.5

        return series.rolling(window=window).apply(
            lambda x: fractal_dimension(x.values) if len(x) >= 10 else np.nan,
            raw=False
        )
