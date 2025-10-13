"""
Unified Feature Pipeline for Training and Inference consistency.

This module ensures that features computed during training are exactly
replicated during inference, including multi-timeframe indicators and
relative OHLC normalization.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path
import json
from loguru import logger

from .pipeline import (
    log_returns, rolling_std, atr, bollinger, macd, rsi_wilder,
    donchian, hurst_feature, time_cyclic_and_session, Standardizer
)
from .indicators import ema, sma


class FeatureConfig:
    """Configuration for feature engineering pipeline."""

    def __init__(self, config_dict: Optional[Dict] = None):
        self.config = config_dict or {}
        self._validate_config()

    def _validate_config(self):
        """Validate and set defaults for configuration."""
        defaults = {
            "base_features": {
                "relative_ohlc": True,
                "log_returns": True,
                "time_features": True,
                "session_features": True
            },
            "indicators": {
                "atr": {"enabled": True, "n": 14},
                "rsi": {"enabled": True, "n": 14},
                "bollinger": {"enabled": True, "n": 20, "k": 2.0},
                "macd": {"enabled": True, "fast": 12, "slow": 26, "signal": 9},
                "donchian": {"enabled": False, "n": 20},
                "keltner": {"enabled": False, "ema": 20, "atr": 10, "mult": 1.5},
                "hurst": {"enabled": False, "window": 64},
                "ema": {"enabled": False, "fast": 12, "slow": 26}
            },
            "multi_timeframe": {
                "enabled": False,
                "timeframes": ["1m", "5m", "15m"],
                "base_timeframe": "1m",
                "query_timeframe": "5m",  # Timeframe di interrogazione selezionabile
                "indicators": ["atr", "rsi", "bollinger"],
                "hierarchical_mode": True,  # Modalità gerarchica con candela madre
                "exclude_children": True,   # Esclude i children dal modeling
                "auto_group_selection": True  # Selezione automatica gruppo candele
            },
            "standardization": {
                "enabled": True,
                "window_bars": 1000,
                "method": "rolling"
            },
            "warmup_bars": 64,
            "rv_window": 60
        }

        # Deep merge defaults with provided config
        self.config = self._deep_merge(defaults, self.config)

    def _deep_merge(self, default: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_indicator_config(self, indicator: str) -> Dict[str, Any]:
        """Get configuration for a specific indicator."""
        return self.config.get("indicators", {}).get(indicator, {})

    def is_indicator_enabled(self, indicator: str) -> bool:
        """Check if an indicator is enabled."""
        return self.get_indicator_config(indicator).get("enabled", False)

    def get_multi_timeframe_config(self) -> Dict[str, Any]:
        """Get multi-timeframe configuration."""
        return self.config.get("multi_timeframe", {})

    def is_multi_timeframe_enabled(self) -> bool:
        """Check if multi-timeframe features are enabled."""
        return self.get_multi_timeframe_config().get("enabled", False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeatureConfig':
        """Create FeatureConfig from dictionary."""
        return cls(config_dict)

    @classmethod
    def from_training_args(cls, args: Any) -> 'FeatureConfig':
        """Create FeatureConfig from training arguments."""
        config = {
            "indicators": {
                "atr": {"enabled": True, "n": getattr(args, "atr_n", 14)},
                "rsi": {"enabled": True, "n": getattr(args, "rsi_n", 14)},
                "bollinger": {"enabled": True, "n": getattr(args, "bb_n", 20), "k": 2.0},
                "hurst": {"enabled": True, "window": getattr(args, "hurst_window", 64)}
            },
            "warmup_bars": getattr(args, "warmup_bars", 64),
            "rv_window": getattr(args, "rv_window", 60),
            "standardization": {
                "enabled": True,
                "window_bars": getattr(args, "rv_window", 60),
                "method": "rolling"
            }
        }

        # Add multi-timeframe if indicator_tfs present
        indicator_tfs = getattr(args, "indicator_tfs", None)
        if indicator_tfs:
            try:
                if isinstance(indicator_tfs, str):
                    import json
                    indicator_tfs = json.loads(indicator_tfs)
                if isinstance(indicator_tfs, dict) and indicator_tfs:
                    config["multi_timeframe"] = {
                        "enabled": True,
                        "timeframes": sorted(set(tf for tfs in indicator_tfs.values() for tf in tfs)),
                        "base_timeframe": getattr(args, "timeframe", "1m"),
                        "indicators": list(indicator_tfs.keys()),
                        "indicator_tfs": indicator_tfs
                    }
            except Exception as e:
                logger.warning(f"Failed to parse indicator_tfs: {e}")

        return cls(config)


def _timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    """Convert timeframe string to pandas Timedelta."""
    tf = str(tf).strip().lower()
    if tf.endswith("m"):
        return pd.Timedelta(minutes=int(tf[:-1]))
    elif tf.endswith("h"):
        return pd.Timedelta(hours=int(tf[:-1]))
    elif tf.endswith("d"):
        return pd.Timedelta(days=int(tf[:-1]))
    else:
        raise ValueError(f"Unsupported timeframe: {tf}")


def _resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLC data to specified timeframe."""
    if timeframe.endswith("m"):
        minutes = int(timeframe[:-1])
        if minutes <= 0:
            raise ValueError(f"Invalid timeframe: {timeframe} (must be positive)")
        rule = f"{minutes}T"
    elif timeframe.endswith("h"):
        hours = int(timeframe[:-1])
        if hours <= 0:
            raise ValueError(f"Invalid timeframe: {timeframe} (must be positive)")
        rule = f"{hours}H"
    elif timeframe.endswith("d"):
        days = int(timeframe[:-1])
        if days <= 0:
            raise ValueError(f"Invalid timeframe: {timeframe} (must be positive)")
        rule = f"{days}D"
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    df_indexed = df.copy()
    df_indexed.index = pd.to_datetime(df_indexed["ts_utc"], unit="ms", utc=True)

    ohlc = df_indexed[["open", "high", "low", "close"]].astype(float).resample(
        rule, label="right"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    if "volume" in df_indexed.columns:
        ohlc["volume"] = df_indexed["volume"].astype(float).resample(rule, label="right").sum()

    ohlc["ts_utc"] = (ohlc.index.view("int64") // 10**6)
    return ohlc.reset_index(drop=True)


def _relative_ohlc_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply relative OHLC normalization as used in training.
    This ensures inference uses the same normalization as training.
    
    NOTE: This is a wrapper around feature_engineering.relative_ohlc()
    to maintain API compatibility. Use the centralized version directly
    for new code to avoid duplication.
    """
    from .feature_engineering import relative_ohlc
    return relative_ohlc(df)


def _compute_multi_timeframe_indicators(
    df: pd.DataFrame,
    config: FeatureConfig,
    base_timeframe: str
) -> pd.DataFrame:
    """Compute indicators across multiple timeframes."""
    if not config.is_multi_timeframe_enabled():
        return df

    mtf_config = config.get_multi_timeframe_config()
    timeframes = mtf_config.get("timeframes", [])
    indicators = mtf_config.get("indicators", [])
    indicator_tfs = mtf_config.get("indicator_tfs", {})

    # Base dataframe with datetime index for merging
    base_df = df.copy()
    base_df.index = pd.to_datetime(base_df["ts_utc"], unit="ms", utc=True)

    result_df = base_df.copy()

    for indicator in indicators:
        if not config.is_indicator_enabled(indicator):
            continue

        ind_config = config.get_indicator_config(indicator)
        # Get specific timeframes for this indicator
        ind_timeframes = indicator_tfs.get(indicator, timeframes)

        for tf in ind_timeframes:
            if tf == base_timeframe:
                continue  # Already computed in base pipeline

            try:
                # Resample to target timeframe
                resampled_df = _resample_ohlc(df, tf)
                resampled_df.index = pd.to_datetime(resampled_df["ts_utc"], unit="ms", utc=True)

                # Compute indicator on resampled data
                if indicator == "atr":
                    ind_df = atr(resampled_df, n=ind_config.get("n", 14))
                    feature_col = f"atr_{tf}_{ind_config.get('n', 14)}"
                    ind_df = ind_df.rename(columns={"atr": feature_col})

                elif indicator == "rsi":
                    ind_df = rsi_wilder(resampled_df, n=ind_config.get("n", 14))
                    feature_col = f"rsi_{tf}_{ind_config.get('n', 14)}"
                    ind_df = ind_df.rename(columns={"rsi": feature_col})

                elif indicator == "bollinger":
                    n = ind_config.get("n", 20)
                    k = ind_config.get("k", 2.0)
                    ind_df = bollinger(resampled_df, n=n, k=k)
                    # Rename columns to include timeframe
                    old_cols = [f"bb_upper_{n}", f"bb_lower_{n}", f"bb_pctb_{n}"]
                    new_cols = [f"bb_upper_{tf}_{n}_{k}", f"bb_lower_{tf}_{n}_{k}", f"bb_pctb_{tf}_{n}_{k}"]
                    rename_map = {old: new for old, new in zip(old_cols, new_cols) if old in ind_df.columns}
                    ind_df = ind_df.rename(columns=rename_map)

                else:
                    continue

                # Merge back to base timeframe using forward-fill
                try:
                    feature_cols = [col for col in ind_df.columns if col.startswith(f"{indicator}_{tf}_")]
                    if feature_cols:
                        merge_df = ind_df[feature_cols + ["ts_utc"]].copy()
                        merge_df.index = pd.to_datetime(merge_df["ts_utc"], unit="ms", utc=True)

                        # Use merge_asof for time-series alignment
                        for col in feature_cols:
                            aligned_series = pd.merge_asof(
                                result_df[["ts_utc"]].reset_index(),
                                merge_df[[col]].reset_index(),
                                left_on="index",
                                right_on="index",
                                direction="backward"
                            )[col]
                            result_df[col] = aligned_series.values

                except Exception as e:
                    logger.warning(f"Failed to merge {indicator}_{tf}: {e}")

            except Exception as e:
                logger.warning(f"Failed to compute {indicator} for {tf}: {e}")

    return result_df.reset_index(drop=True)


def unified_feature_pipeline(
    df: pd.DataFrame,
    config: FeatureConfig,
    timeframe: str = "1m",
    standardizer: Optional[Standardizer] = None,
    fit_standardizer: bool = True,
    output_format: str = "flat"
) -> Union[Tuple[pd.DataFrame, Standardizer, List[str]], Tuple[Dict[str, pd.DataFrame], Standardizer, List[str]]]:
    """
    Unified feature engineering pipeline for both training and inference.

    Args:
        df: Input OHLC dataframe
        config: Feature configuration
        timeframe: Base timeframe for the data
        standardizer: Pre-fitted standardizer (for inference)
        fit_standardizer: Whether to fit standardizer (training) or use existing (inference)
        output_format: Output format - "flat" (single DataFrame), "sequence" (temporal ordering preserved),
                      "multi_timeframe" (dict of DataFrames per timeframe for SSSD)

    Returns:
        If output_format="multi_timeframe":
            Tuple of (features_dict, standardizer, feature_names)
            where features_dict = {"5m": df_5m, "15m": df_15m, ...}
        Otherwise:
            Tuple of (features_df, standardizer, feature_names)
    """
    tmp = df.copy().sort_values("ts_utc").reset_index(drop=True)

    # Step 1: Base features
    if config.config["base_features"]["relative_ohlc"]:
        tmp = _relative_ohlc_normalization(tmp)

    if config.config["base_features"]["log_returns"]:
        tmp = log_returns(tmp, col="close", out_col="r")

    # Step 2: Rolling features
    rv_window = config.config.get("rv_window", 60)
    std_window = config.config["standardization"]["window_bars"]
    tmp = rolling_std(tmp, col="r", window=std_window, out_col=f"r_std_{std_window}")

    # Realized volatility
    if "r" in tmp.columns:
        rv = tmp["r"].pow(2).rolling(window=rv_window, min_periods=1).sum().apply(np.sqrt)
        tmp[f"rv_{rv_window}"] = rv

    # Step 3: Single timeframe indicators
    indicators_config = config.config["indicators"]

    if config.is_indicator_enabled("atr"):
        atr_config = config.get_indicator_config("atr")
        tmp = atr(tmp, n=atr_config["n"])

    if config.is_indicator_enabled("rsi"):
        rsi_config = config.get_indicator_config("rsi")
        tmp = rsi_wilder(tmp, n=rsi_config["n"])

    if config.is_indicator_enabled("bollinger"):
        bb_config = config.get_indicator_config("bollinger")
        tmp = bollinger(tmp, n=bb_config["n"], k=bb_config["k"])

    if config.is_indicator_enabled("macd"):
        macd_config = config.get_indicator_config("macd")
        tmp = macd(tmp, fast=macd_config["fast"], slow=macd_config["slow"], signal=macd_config["signal"])

    if config.is_indicator_enabled("donchian"):
        don_config = config.get_indicator_config("donchian")
        tmp = donchian(tmp, n=don_config["n"])

    if config.is_indicator_enabled("hurst"):
        hurst_config = config.get_indicator_config("hurst")
        tmp = hurst_feature(tmp, window=hurst_config["window"], out_col="hurst")

    if config.is_indicator_enabled("ema"):
        ema_config = config.get_indicator_config("ema")
        tmp["ema_fast"] = ema(tmp["close"], span=ema_config["fast"])
        tmp["ema_slow"] = ema(tmp["close"], span=ema_config["slow"])
        tmp["ema_slope"] = tmp["ema_fast"].diff().fillna(0.0)

    # Step 4: Time and session features
    if config.config["base_features"]["time_features"]:
        tmp = time_cyclic_and_session(tmp)

    # Step 5: Multi-timeframe indicators
    tmp = _compute_multi_timeframe_indicators(tmp, config, timeframe)

    # Step 6: Define feature columns
    feature_names = _get_feature_names(tmp, config)

    # Keep only existing features
    available_features = [f for f in feature_names if f in tmp.columns]

    # Step 7: Drop warmup bars
    warmup = config.config["warmup_bars"]
    if warmup > 0 and len(tmp) > warmup:
        features_df = tmp.iloc[warmup:][available_features].copy()
    else:
        features_df = tmp[available_features].copy()

    # Step 8: Standardization
    if config.config["standardization"]["enabled"]:
        if standardizer is None and fit_standardizer:
            standardizer = Standardizer(cols=available_features)
            standardizer.fit(features_df)

        if standardizer is not None:
            features_df = standardizer.transform(features_df)
    else:
        if standardizer is None:
            standardizer = Standardizer(cols=[], mu={}, sigma={})

    features_df = features_df.reset_index(drop=True)

    # Step 9: Handle output format
    if output_format == "multi_timeframe":
        # For SSSD: split features by timeframe into separate DataFrames
        features_dict = _split_features_by_timeframe(tmp, features_df, config, available_features)
        return features_dict, standardizer, available_features
    elif output_format == "sequence":
        # Preserve temporal ordering (already sorted)
        return features_df, standardizer, available_features
    else:
        # Default flat format
        return features_df, standardizer, available_features


def _split_features_by_timeframe(
    tmp: pd.DataFrame,
    features_df: pd.DataFrame,
    config: FeatureConfig,
    available_features: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Split features into separate DataFrames per timeframe for SSSD.

    Args:
        tmp: Full DataFrame with timestamps
        features_df: Computed features DataFrame
        config: Feature configuration
        available_features: List of feature names

    Returns:
        Dict mapping timeframe to features DataFrame
        Format: {"5m": df_5m, "15m": df_15m, "1h": df_1h, "4h": df_4h}
        Each df has columns: [timestamp, feature_0, feature_1, ...]
    """
    # Get timeframes from config (default SSSD timeframes)
    if config.is_multi_timeframe_enabled():
        mtf_config = config.get_multi_timeframe_config()
        timeframes = mtf_config.get("timeframes", ["5m", "15m", "1h", "4h"])
    else:
        # Default SSSD timeframes
        timeframes = ["5m", "15m", "1h", "4h"]

    # Add timestamp column to features_df
    features_with_ts = features_df.copy()

    # Get timestamps from original dataframe (after warmup)
    warmup = config.config["warmup_bars"]
    if warmup > 0 and len(tmp) > warmup:
        timestamps = tmp.iloc[warmup:]["ts_utc"].values
    else:
        timestamps = tmp["ts_utc"].values

    features_with_ts.insert(0, "timestamp", pd.to_datetime(timestamps, unit="ms", utc=True))

    # Create dict of DataFrames per timeframe
    features_dict = {}

    for tf in timeframes:
        # Resample features to target timeframe
        tf_df = _resample_features_to_timeframe(features_with_ts, tf, available_features)

        # Rename timestamp column
        tf_df = tf_df.rename(columns={"timestamp": "timestamp"})

        features_dict[tf] = tf_df

    return features_dict


def _resample_features_to_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Resample features to target timeframe.

    Args:
        df: DataFrame with timestamp and features
        timeframe: Target timeframe (e.g., "5m", "15m", "1h", "4h")
        feature_columns: List of feature columns to resample

    Returns:
        Resampled DataFrame
    """
    # Parse timeframe
    if timeframe.endswith("m"):
        rule = f"{int(timeframe[:-1])}T"
    elif timeframe.endswith("h"):
        rule = f"{int(timeframe[:-1])}H"
    elif timeframe.endswith("d"):
        rule = f"{int(timeframe[:-1])}D"
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # Set index to timestamp
    df_indexed = df.set_index("timestamp")

    # Resample features (use last value for each period)
    resampled = df_indexed[feature_columns].resample(rule, label="right").last().dropna()

    # Reset index to get timestamp column
    resampled = resampled.reset_index()

    return resampled


def _get_feature_names(df: pd.DataFrame, config: FeatureConfig) -> List[str]:
    """Get ordered list of feature names based on configuration."""
    features = []

    # Base features
    if config.config["base_features"]["relative_ohlc"]:
        features.extend(["r_open", "r_high", "r_low", "r_close"])

    if config.config["base_features"]["log_returns"]:
        features.append("r")
        std_window = config.config["standardization"]["window_bars"]
        features.append(f"r_std_{std_window}")

    # Realized volatility
    rv_window = config.config.get("rv_window", 60)
    features.append(f"rv_{rv_window}")

    # Single timeframe indicators
    if config.is_indicator_enabled("atr"):
        features.append("atr")

    if config.is_indicator_enabled("rsi"):
        features.append("rsi")

    if config.is_indicator_enabled("bollinger"):
        bb_config = config.get_indicator_config("bollinger")
        n = bb_config["n"]
        features.extend([f"bb_upper_{n}", f"bb_lower_{n}", f"bb_pctb_{n}"])

    if config.is_indicator_enabled("macd"):
        features.extend(["macd", "macd_signal", "macd_hist"])

    if config.is_indicator_enabled("donchian"):
        don_config = config.get_indicator_config("donchian")
        n = don_config["n"]
        features.extend([f"don_upper_{n}", f"don_lower_{n}"])

    if config.is_indicator_enabled("hurst"):
        features.append("hurst")

    if config.is_indicator_enabled("ema"):
        features.extend(["ema_fast", "ema_slow", "ema_slope"])

    # Time features
    if config.config["base_features"]["time_features"]:
        features.extend([
            "hour_sin", "hour_cos", "hour_int_sin", "hour_int_cos",
            "dow_sin", "dow_cos"
        ])

    if config.config["base_features"]["session_features"]:
        features.extend(["session_tokyo", "session_london", "session_ny"])

    # Multi-timeframe features
    if config.is_multi_timeframe_enabled():
        mtf_config = config.get_multi_timeframe_config()
        indicator_tfs = mtf_config.get("indicator_tfs", {})

        for indicator, timeframes in indicator_tfs.items():
            if not config.is_indicator_enabled(indicator):
                continue

            ind_config = config.get_indicator_config(indicator)

            for tf in timeframes:
                if indicator == "atr":
                    features.append(f"atr_{tf}_{ind_config['n']}")
                elif indicator == "rsi":
                    features.append(f"rsi_{tf}_{ind_config['n']}")
                elif indicator == "bollinger":
                    n, k = ind_config["n"], ind_config["k"]
                    features.extend([
                        f"bb_upper_{tf}_{n}_{k}",
                        f"bb_lower_{tf}_{n}_{k}",
                        f"bb_pctb_{tf}_{n}_{k}"
                    ])

    # Only return features that exist in the dataframe
    return [f for f in features if f in df.columns]


def save_feature_config(config: FeatureConfig, filepath: Union[str, Path]) -> None:
    """Save feature configuration to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def load_feature_config(filepath: Union[str, Path]) -> FeatureConfig:
    """Load feature configuration from JSON file."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Feature config not found: {filepath}")

    with open(filepath, 'r') as f:
        config_dict = json.load(f)

    return FeatureConfig.from_dict(config_dict)


# ========== MULTI-TIMEFRAME HIERARCHICAL SYSTEM ==========

class CandleHierarchy:
    """
    Sistema gerarchico multi-timeframe dove ogni candela ha riferimento alla candela madre.
    Implementa la proposta di modellazione multi-timeframe con timeframe di interrogazione selezionabile.
    """

    def __init__(self, base_timeframe: str = "1m"):
        self.base_timeframe = base_timeframe
        self.hierarchy_map = {}
        self.parent_child_map = {}

    def build_hierarchy(self, df: pd.DataFrame, timeframes: List[str]) -> pd.DataFrame:
        """
        Costruisce la gerarchia di candele con riferimenti parent-child.

        Args:
            df: DataFrame con candele del timeframe base
            timeframes: Lista di timeframes da includere nella gerarchia

        Returns:
            DataFrame arricchito con colonne di gerarchia
        """
        # Assicurati che il DataFrame abbia un indice temporale
        result_df = df.copy()
        result_df['ts_dt'] = pd.to_datetime(result_df['ts_utc'], unit='ms', utc=True)
        result_df = result_df.set_index('ts_dt').sort_index()

        # Aggiungi colonne per la gerarchia
        result_df['base_timeframe'] = self.base_timeframe
        result_df['candle_id'] = range(len(result_df))

        # Per ogni timeframe superiore, crea il mapping parent-child
        for tf in sorted(timeframes):
            if tf == self.base_timeframe:
                result_df[f'parent_{tf}_id'] = result_df['candle_id']
                continue

            # Resample al timeframe superiore
            parent_df = self._resample_with_ids(result_df, tf)

            # Crea il mapping parent-child
            parent_mapping = self._create_parent_mapping(result_df, parent_df, tf)
            result_df[f'parent_{tf}_id'] = parent_mapping

        return result_df.reset_index()

    def _resample_with_ids(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample mantenendo gli ID delle candele."""
        if timeframe.endswith("m"):
            rule = f"{int(timeframe[:-1])}T"
        elif timeframe.endswith("h"):
            rule = f"{int(timeframe[:-1])}H"
        elif timeframe.endswith("d"):
            rule = f"{int(timeframe[:-1])}D"
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Resample OHLC
        ohlc = df[["open", "high", "low", "close"]].resample(rule, label="right").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        }).dropna()

        # Volume se presente
        if "volume" in df.columns:
            ohlc["volume"] = df["volume"].resample(rule, label="right").sum()

        # Crea ID per le candele del timeframe superiore
        ohlc['parent_candle_id'] = range(len(ohlc))
        ohlc['timeframe'] = timeframe

        return ohlc

    def _create_parent_mapping(self, child_df: pd.DataFrame, parent_df: pd.DataFrame, parent_tf: str) -> pd.Series:
        """Crea il mapping da candele figlie a candele madri."""
        mapping = pd.Series(index=child_df.index, dtype='Int64')

        for parent_idx, parent_row in parent_df.iterrows():
            # Trova tutte le candele figlie che appartengono a questa candela madre
            start_time = parent_idx

            # Calcola il timeframe precedente per il range
            if parent_tf.endswith("m"):
                minutes = int(parent_tf[:-1])
                end_time = start_time + pd.Timedelta(minutes=minutes)
            elif parent_tf.endswith("h"):
                hours = int(parent_tf[:-1])
                end_time = start_time + pd.Timedelta(hours=hours)
            elif parent_tf.endswith("d"):
                days = int(parent_tf[:-1])
                end_time = start_time + pd.Timedelta(days=days)

            # Trova le candele figlie in questo range
            mask = (child_df.index > start_time - pd.Timedelta(microseconds=1)) & (child_df.index <= end_time)
            mapping.loc[mask] = parent_row['parent_candle_id']

        return mapping

    def select_query_group(self, df: pd.DataFrame, query_timeframe: str, exclude_children: bool = True) -> pd.DataFrame:
        """
        Seleziona il gruppo di candele per il modeling basato sul timeframe di interrogazione.

        Args:
            df: DataFrame con gerarchia costruita
            query_timeframe: Timeframe di interrogazione per il modeling
            exclude_children: Se True, esclude le candele children dal gruppo

        Returns:
            DataFrame filtrato per il modeling
        """
        if not exclude_children:
            return df

        # Se il query_timeframe è il base timeframe, restituisci tutto
        if query_timeframe == self.base_timeframe:
            return df

        # Altrimenti, mantieni solo le candele che sono "rappresentative" del query_timeframe
        # Questo significa prendere una candela ogni N candele base
        tf_ratio = self._calculate_timeframe_ratio(self.base_timeframe, query_timeframe)

        # Seleziona ogni N-esima candela per ridurre ridondanza
        selected_indices = range(0, len(df), tf_ratio)
        return df.iloc[selected_indices].copy()

    def _calculate_timeframe_ratio(self, base_tf: str, target_tf: str) -> int:
        """Calcola il rapporto tra timeframes."""
        base_minutes = self._tf_to_minutes(base_tf)
        target_minutes = self._tf_to_minutes(target_tf)
        return max(1, target_minutes // base_minutes)

    def _tf_to_minutes(self, tf: str) -> int:
        """Converte timeframe in minuti."""
        if tf.endswith("m"):
            return int(tf[:-1])
        elif tf.endswith("h"):
            return int(tf[:-1]) * 60
        elif tf.endswith("d"):
            return int(tf[:-1]) * 24 * 60
        else:
            return 1

    def get_hierarchical_features(self, df: pd.DataFrame, feature_columns: List[str], query_timeframe: str) -> pd.DataFrame:
        """
        Estrae features gerarchiche basate sul timeframe di interrogazione.

        Args:
            df: DataFrame con gerarchia e features
            feature_columns: Colonne di features da estrarre
            query_timeframe: Timeframe di interrogazione

        Returns:
            DataFrame con features gerarchiche
        """
        result_df = df.copy()

        # Aggiungi features dalla candela madre del query_timeframe
        parent_col = f'parent_{query_timeframe}_id'
        if parent_col in result_df.columns:
            # Crea features aggregate dalla candela madre
            for feature in feature_columns:
                if feature in result_df.columns:
                    # Calcola media/max/min del feature per gruppo di candele madri
                    grouped = result_df.groupby(parent_col)[feature]
                    result_df[f'{feature}_parent_mean'] = grouped.transform('mean')
                    result_df[f'{feature}_parent_max'] = grouped.transform('max')
                    result_df[f'{feature}_parent_min'] = grouped.transform('min')
                    result_df[f'{feature}_parent_std'] = grouped.transform('std').fillna(0)

        return result_df


def hierarchical_multi_timeframe_pipeline(
    df: pd.DataFrame,
    config: FeatureConfig,
    timeframe: str = "1m",
    standardizer: Optional[Standardizer] = None,
    fit_standardizer: bool = True
) -> Tuple[pd.DataFrame, Standardizer, List[str], CandleHierarchy]:
    """
    Pipeline unificata con sistema gerarchico multi-timeframe.

    Implementa la proposta di modellazione multi-timeframe dove:
    - Ogni candela ha riferimento alla candela madre
    - Timeframe di interrogazione selezionabile
    - Esclusione automatica dei children
    - Selezione automatica del gruppo di candele

    Args:
        df: Input OHLC dataframe
        config: Feature configuration con multi_timeframe abilitato
        timeframe: Base timeframe per i dati
        standardizer: Pre-fitted standardizer (per inference)
        fit_standardizer: Whether to fit standardizer (training) or use existing (inference)

    Returns:
        Tuple of (features_df, standardizer, feature_names, hierarchy)
    """
    if not config.is_multi_timeframe_enabled():
        # Fallback al pipeline standard
        features_df, standardizer, feature_names = unified_feature_pipeline(
            df, config, timeframe, standardizer, fit_standardizer
        )
        return features_df, standardizer, feature_names, None

    mtf_config = config.get_multi_timeframe_config()
    query_timeframe = mtf_config.get("query_timeframe", "5m")
    timeframes = mtf_config.get("timeframes", ["1m", "5m", "15m"])
    exclude_children = mtf_config.get("exclude_children", True)

    # Step 1: Costruisci la gerarchia di candele
    hierarchy = CandleHierarchy(base_timeframe=timeframe)
    df_with_hierarchy = hierarchy.build_hierarchy(df, timeframes)

    # Step 2: Applica il pipeline standard
    features_df, standardizer, base_feature_names = unified_feature_pipeline(
        df_with_hierarchy, config, timeframe, standardizer, fit_standardizer
    )

    # Step 3: Aggiungi features gerarchiche
    features_df = hierarchy.get_hierarchical_features(
        features_df, base_feature_names, query_timeframe
    )

    # Step 4: Seleziona il gruppo di candele per il modeling
    if exclude_children:
        features_df = hierarchy.select_query_group(
            features_df, query_timeframe, exclude_children
        )

    # Step 5: Aggiorna lista features con quelle gerarchiche
    hierarchical_features = [col for col in features_df.columns
                           if col.endswith(('_parent_mean', '_parent_max', '_parent_min', '_parent_std'))]

    all_feature_names = base_feature_names + hierarchical_features
    available_features = [f for f in all_feature_names if f in features_df.columns]

    # Step 6: Ri-standardizza se necessario includendo le nuove features
    if hierarchical_features and config.config["standardization"]["enabled"]:
        if fit_standardizer:
            standardizer = Standardizer(cols=available_features)
            standardizer.fit(features_df[available_features])

        if standardizer is not None:
            features_df[available_features] = standardizer.transform(features_df[available_features])

    logger.info(f"Hierarchical multi-timeframe pipeline completed: "
               f"query_timeframe={query_timeframe}, "
               f"features={len(available_features)}, "
               f"samples={len(features_df)}")

    return features_df[available_features], standardizer, available_features, hierarchy