"""
Feature Name Standardization Utilities

HIGH-002: Ensures consistent lowercase_underscore naming across all features.

Problem: Features created with different naming conventions:
- 'SMA_20', 'RSI_14', 'MACD' (uppercase)
- 'sma_20', 'rsi_14', 'macd' (lowercase)

Solution: Standardize all to lowercase_underscore format.
"""
from __future__ import annotations

from typing import List, Dict, Set
import re
import pandas as pd
from loguru import logger


# Mapping of common uppercase patterns to lowercase
FEATURE_NAME_MAPPINGS = {
    # Technical indicators
    'SMA': 'sma',
    'EMA': 'ema',
    'RSI': 'rsi',
    'MACD': 'macd',
    'ATR': 'atr',
    'BB': 'bb',
    'ADX': 'adx',
    'CCI': 'cci',
    'MFI': 'mfi',
    'OBV': 'obv',
    'VWAP': 'vwap',
    'STOCH': 'stoch',
    
    # Candlestick patterns
    'DOJI': 'doji',
    'HAMMER': 'hammer',
    'ENGULFING': 'engulfing',
    'HARAMI': 'harami',
    'MORNING_STAR': 'morning_star',
    'EVENING_STAR': 'evening_star',
    
    # Chart patterns
    'HEAD_AND_SHOULDERS': 'head_and_shoulders',
    'TRIANGLE': 'triangle',
    'WEDGE': 'wedge',
    'FLAG': 'flag',
    'PENNANT': 'pennant',
    
    # Price-based
    'OPEN': 'open',
    'HIGH': 'high',
    'LOW': 'low',
    'CLOSE': 'close',
    'VOLUME': 'volume',
    
    # Derived features
    'RETURN': 'return',
    'LOG_RETURN': 'log_return',
    'VOLATILITY': 'volatility',
    'RANGE': 'range'
}


def standardize_feature_name(name: str) -> str:
    """
    Standardize a single feature name to lowercase_underscore format.
    
    Args:
        name: Feature name (e.g., 'SMA_20', 'RSI_14', 'MACD')
        
    Returns:
        Standardized name (e.g., 'sma_20', 'rsi_14', 'macd')
        
    Examples:
        >>> standardize_feature_name('SMA_20')
        'sma_20'
        >>> standardize_feature_name('RSI_14')
        'rsi_14'
        >>> standardize_feature_name('MACD')
        'macd'
        >>> standardize_feature_name('BB_UPPER')
        'bb_upper'
    """
    if not name:
        return name
    
    # Already lowercase?
    if name.islower() or name.replace('_', '').islower():
        return name
    
    # Try direct mapping first
    if name in FEATURE_NAME_MAPPINGS:
        return FEATURE_NAME_MAPPINGS[name]
    
    # Split by underscore and process each part
    parts = name.split('_')
    standardized_parts = []
    
    for part in parts:
        # Try mapping
        if part in FEATURE_NAME_MAPPINGS:
            standardized_parts.append(FEATURE_NAME_MAPPINGS[part])
        else:
            # Convert to lowercase
            standardized_parts.append(part.lower())
    
    return '_'.join(standardized_parts)


def standardize_feature_names(names: List[str]) -> List[str]:
    """
    Standardize a list of feature names.
    
    Args:
        names: List of feature names
        
    Returns:
        List of standardized feature names
        
    Example:
        >>> standardize_feature_names(['SMA_20', 'RSI_14', 'MACD'])
        ['sma_20', 'rsi_14', 'macd']
    """
    return [standardize_feature_name(name) for name in names]


def standardize_dataframe_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Standardize all column names in a DataFrame to lowercase_underscore format.
    
    Args:
        df: DataFrame with feature columns
        inplace: Whether to modify the DataFrame in place
        
    Returns:
        DataFrame with standardized column names
        
    Example:
        >>> df = pd.DataFrame({'SMA_20': [1, 2], 'RSI_14': [3, 4]})
        >>> standardized = standardize_dataframe_columns(df)
        >>> print(standardized.columns.tolist())
        ['sma_20', 'rsi_14']
    """
    if not inplace:
        df = df.copy()
    
    # Create mapping of old names to new names
    column_mapping = {}
    for col in df.columns:
        if isinstance(col, str):
            new_name = standardize_feature_name(col)
            if new_name != col:
                column_mapping[col] = new_name
    
    if column_mapping:
        df.rename(columns=column_mapping, inplace=True)
        logger.debug(f"Standardized {len(column_mapping)} column names")
    
    return df


def create_feature_mapping(old_names: List[str], new_names: List[str]) -> Dict[str, str]:
    """
    Create a mapping from old feature names to new standardized names.
    
    Args:
        old_names: List of old feature names
        new_names: List of new feature names
        
    Returns:
        Dictionary mapping old names to new names
    """
    if len(old_names) != len(new_names):
        raise ValueError("old_names and new_names must have same length")
    
    return {old: new for old, new in zip(old_names, new_names)}


def validate_feature_names(names: List[str]) -> tuple[bool, List[str]]:
    """
    Validate that feature names follow lowercase_underscore convention.
    
    Args:
        names: List of feature names to validate
        
    Returns:
        Tuple of (all_valid, invalid_names)
        
    Example:
        >>> validate_feature_names(['sma_20', 'rsi_14', 'MACD'])
        (False, ['MACD'])
    """
    invalid_names = []
    
    for name in names:
        if not isinstance(name, str):
            continue
            
        # Check if name is already lowercase_underscore
        if not re.match(r'^[a-z0-9_]+$', name):
            invalid_names.append(name)
    
    return len(invalid_names) == 0, invalid_names


def get_feature_conflicts(names: List[str]) -> Dict[str, List[str]]:
    """
    Find feature names that would standardize to the same name (conflicts).
    
    Args:
        names: List of feature names
        
    Returns:
        Dictionary mapping standardized names to lists of original names
        
    Example:
        >>> get_feature_conflicts(['SMA_20', 'sma_20', 'RSI_14'])
        {'sma_20': ['SMA_20', 'sma_20']}
    """
    standardized_to_originals = {}
    
    for name in names:
        std_name = standardize_feature_name(name)
        if std_name not in standardized_to_originals:
            standardized_to_originals[std_name] = []
        standardized_to_originals[std_name].append(name)
    
    # Filter to only conflicts (multiple originals for one standardized name)
    conflicts = {
        std: originals
        for std, originals in standardized_to_originals.items()
        if len(originals) > 1
    }
    
    return conflicts


def auto_fix_feature_names(df: pd.DataFrame, report: bool = True) -> pd.DataFrame:
    """
    Automatically fix feature names in a DataFrame with conflict detection.
    
    Args:
        df: DataFrame with features
        report: Whether to log a report of changes
        
    Returns:
        DataFrame with standardized feature names
        
    Raises:
        ValueError: If conflicts are detected
    """
    # Check for conflicts
    conflicts = get_feature_conflicts(df.columns.tolist())
    if conflicts:
        conflict_report = "\n".join(
            f"  {std}: {originals}" for std, originals in conflicts.items()
        )
        raise ValueError(f"Feature name conflicts detected:\n{conflict_report}")
    
    # Standardize
    result = standardize_dataframe_columns(df, inplace=False)
    
    if report:
        changed = sum(1 for old, new in zip(df.columns, result.columns) if old != new)
        if changed > 0:
            logger.info(f"✅ Standardized {changed} feature names to lowercase_underscore")
    
    return result


def get_standardization_report(df: pd.DataFrame) -> str:
    """
    Generate a report of feature name standardization changes.
    
    Args:
        df: DataFrame with features
        
    Returns:
        Human-readable report string
    """
    lines = ["Feature Name Standardization Report", "=" * 50]
    
    # Validate current names
    valid, invalid = validate_feature_names(df.columns.tolist())
    
    if valid:
        lines.append("✅ All feature names are already standardized")
    else:
        lines.append(f"⚠️  {len(invalid)} feature names need standardization:")
        lines.append("")
        
        for name in invalid:
            std_name = standardize_feature_name(name)
            lines.append(f"  {name} → {std_name}")
    
    # Check for conflicts
    conflicts = get_feature_conflicts(df.columns.tolist())
    if conflicts:
        lines.append("")
        lines.append(f"❌ {len(conflicts)} conflicts detected:")
        for std, originals in conflicts.items():
            lines.append(f"  {std}: {originals}")
    
    return "\n".join(lines)


# Convenience function for backward compatibility
def normalize_feature_names(names: List[str]) -> List[str]:
    """Alias for standardize_feature_names for backward compatibility."""
    return standardize_feature_names(names)
