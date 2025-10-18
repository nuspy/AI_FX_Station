"""
Dynamic Horizon Suggestions

Provides smart horizon suggestions based on timeframe and trading style.
"""

from typing import List, Dict, Tuple
from loguru import logger


# Preset horizon configurations for different timeframes and styles
HORIZON_PRESETS = {
    # Scalping (very short term)
    'scalping': {
        '1m': [5, 10, 15],
        '5m': [3, 6, 12],
        '15m': [2, 4, 8],
        '30m': [1, 2, 4],
        '1h': [1, 2, 3],
    },
    
    # Day trading (short-medium term)
    'daytrading': {
        '1m': [15, 60, 240],
        '5m': [12, 48, 96],
        '15m': [4, 16, 32],
        '30m': [2, 8, 16],
        '1h': [1, 4, 8],
        '4h': [1, 2, 4],
    },
    
    # Swing trading (medium term)
    'swing': {
        '1m': [240, 480, 1440],
        '5m': [96, 192, 288],
        '15m': [32, 64, 96],
        '30m': [16, 32, 48],
        '1h': [8, 16, 24],
        '4h': [4, 8, 12],
        '1d': [1, 3, 5],
    },
    
    # Position trading (long term)
    'position': {
        '1h': [24, 72, 168],
        '4h': [6, 18, 42],
        '1d': [5, 10, 20],
        '1w': [1, 2, 4],
    },
    
    # Balanced (default - good for general use)
    'balanced': {
        '1m': [15, 60, 240],
        '5m': [12, 48, 96],
        '15m': [8, 32, 64],
        '30m': [4, 16, 32],
        '1h': [4, 12, 24],
        '4h': [2, 6, 12],
        '1d': [1, 5, 10],
    }
}


def suggest_horizons(
    timeframe: str,
    style: str = 'balanced',
    custom_multipliers: List[int] = None
) -> List[int]:
    """
    Suggest optimal horizons based on timeframe and trading style.
    
    Args:
        timeframe: Trading timeframe (e.g., '1m', '5m', '1h')
        style: Trading style ('scalping', 'daytrading', 'swing', 'position', 'balanced')
        custom_multipliers: Custom multipliers to apply (e.g., [1, 4, 8])
    
    Returns:
        List of suggested horizons in bars
    
    Examples:
        >>> suggest_horizons('1m', 'daytrading')
        [15, 60, 240]
        
        >>> suggest_horizons('5m', 'scalping')
        [3, 6, 12]
        
        >>> suggest_horizons('1h', custom_multipliers=[1, 2, 4, 8])
        [1, 2, 4, 8]
    """
    # Normalize timeframe
    tf = timeframe.strip().lower()
    
    # Custom multipliers override presets
    if custom_multipliers:
        logger.info(f"Using custom multipliers: {custom_multipliers}")
        return sorted(custom_multipliers)
    
    # Get preset for style
    if style not in HORIZON_PRESETS:
        logger.warning(f"Unknown style '{style}', using 'balanced'")
        style = 'balanced'
    
    preset = HORIZON_PRESETS[style]
    
    # Get horizons for timeframe
    if tf in preset:
        horizons = preset[tf]
        logger.info(f"Suggested horizons for {tf} ({style}): {horizons}")
        return horizons
    else:
        # Fallback: use balanced for closest timeframe
        logger.warning(f"No preset for {tf} in {style}, using balanced")
        return suggest_horizons(tf, style='balanced')


def get_time_labels(horizons: List[int], timeframe: str) -> List[str]:
    """
    Convert horizon bars to human-readable time labels.
    
    Args:
        horizons: List of horizons in bars
        timeframe: Base timeframe
    
    Returns:
        List of time labels
    
    Examples:
        >>> get_time_labels([15, 60, 240], '1m')
        ['15min', '1h', '4h']
    """
    from .horizon_format_adapter import bars_to_time_labels
    
    labels_str = bars_to_time_labels(horizons, timeframe)
    return labels_str.split(',')


def describe_horizons(horizons: List[int], timeframe: str) -> str:
    """
    Generate human-readable description of horizons.
    
    Args:
        horizons: List of horizons in bars
        timeframe: Base timeframe
    
    Returns:
        Description string
    
    Examples:
        >>> describe_horizons([15, 60, 240], '1m')
        "Short (15min), Medium (1h), Long (4h)"
    """
    if not horizons:
        return "No horizons"
    
    time_labels = get_time_labels(horizons, timeframe)
    
    # Categorize
    descriptions = []
    for i, (horizon, label) in enumerate(zip(horizons, time_labels)):
        if i == 0:
            category = "Short"
        elif i == len(horizons) - 1:
            category = "Long"
        else:
            category = "Medium"
        
        descriptions.append(f"{category} ({label})")
    
    return ", ".join(descriptions)


def validate_horizons_for_style(
    horizons: List[int],
    timeframe: str,
    style: str = 'balanced'
) -> Tuple[bool, str]:
    """
    Validate if horizons are appropriate for the given style.
    
    Args:
        horizons: List of horizons in bars
        timeframe: Base timeframe
        style: Trading style
    
    Returns:
        Tuple of (is_valid, message)
    
    Examples:
        >>> validate_horizons_for_style([1, 2, 3], '1m', 'scalping')
        (True, "Horizons are appropriate for scalping")
        
        >>> validate_horizons_for_style([1440, 2880], '1m', 'scalping')
        (False, "Horizons too long for scalping (use < 20 bars)")
    """
    if not horizons:
        return False, "No horizons specified"
    
    # Get suggested horizons
    suggested = suggest_horizons(timeframe, style)
    
    # Define acceptable ranges per style
    ranges = {
        'scalping': (1, 20),
        'daytrading': (5, 300),
        'swing': (100, 2000),
        'position': (500, 10000),
        'balanced': (5, 500)
    }
    
    min_bars, max_bars = ranges.get(style, (1, 1000))
    
    # Check if all horizons are within range
    out_of_range = [h for h in horizons if h < min_bars or h > max_bars]
    
    if out_of_range:
        return False, (
            f"Horizons {out_of_range} are outside the recommended range "
            f"for {style} ({min_bars}-{max_bars} bars)"
        )
    
    # Check if horizons are close to suggested
    if set(horizons) == set(suggested):
        return True, f"Perfect match with {style} preset"
    
    # Check if horizons are reasonable multiples
    max_horizon = max(horizons)
    min_horizon = min(horizons)
    
    if max_horizon / min_horizon > 100:
        return False, (
            f"Horizon range too wide ({min_horizon} to {max_horizon}). "
            f"Consider narrower range for better accuracy."
        )
    
    return True, f"Horizons are appropriate for {style}"


def get_all_presets() -> Dict[str, Dict[str, List[int]]]:
    """Get all available horizon presets."""
    return HORIZON_PRESETS.copy()


def get_styles() -> List[str]:
    """Get all available trading styles."""
    return list(HORIZON_PRESETS.keys())
