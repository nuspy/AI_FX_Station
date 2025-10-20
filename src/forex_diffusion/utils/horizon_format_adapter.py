"""
Horizon Format Adapter

Converts between different horizon formats:
- Bars format: "15,60,240" (used in training with --horizon)
- Time format: "15m,1h,4h" (used in GUI inference settings)

Ensures compatibility between training and inference systems.
"""

from __future__ import annotations
from typing import List, Union
from loguru import logger

from .horizon_converter import (
    timeframe_to_minutes,
    minutes_to_timeframe,
    time_labels_to_horizon_bars,
    convert_horizons_for_inference
)
from .horizon_parser import parse_horizon_spec, format_horizon_spec


def bars_to_time_labels(
    horizon_bars: Union[List[int], str],
    base_timeframe: str
) -> str:
    """
    Convert horizon bars to time labels.
    
    Args:
        horizon_bars: List of bars or string "15,60,240" or "1-7/2"
        base_timeframe: Base timeframe (e.g., "1m", "5m")
    
    Returns:
        Time labels string (e.g., "15m,1h,4h")
    
    Examples:
        >>> bars_to_time_labels([15, 60, 240], "1m")
        "15m,1h,4h"
        
        >>> bars_to_time_labels("15,60,240", "1m")
        "15m,1h,4h"
        
        >>> bars_to_time_labels("1-7/2", "5m")
        "5m,15m,25m,35m"
    """
    # Parse if string
    if isinstance(horizon_bars, str):
        horizon_bars = parse_horizon_spec(horizon_bars)
    
    # Convert to minutes
    base_minutes = timeframe_to_minutes(base_timeframe)
    
    time_labels = []
    for bars in horizon_bars:
        total_minutes = bars * base_minutes
        label = minutes_to_timeframe(total_minutes)
        time_labels.append(label)
    
    result = ','.join(time_labels)
    logger.debug(f"Converted bars {horizon_bars} @ {base_timeframe} → {result}")
    
    return result


def time_labels_to_bars(
    time_labels: Union[List[str], str],
    base_timeframe: str
) -> str:
    """
    Convert time labels to horizon bars.
    
    Args:
        time_labels: List of time labels or string "15m,1h,4h"
        base_timeframe: Base timeframe (e.g., "1m", "5m")
    
    Returns:
        Bars string (e.g., "15,60,240")
    
    Examples:
        >>> time_labels_to_bars("15m,1h,4h", "1m")
        "15,60,240"
        
        >>> time_labels_to_bars(["15m", "1h", "4h"], "1m")
        "15,60,240"
    """
    # Parse if string
    if isinstance(time_labels, str):
        # Use convert_horizons_for_inference to handle all formats
        _, horizon_bars = convert_horizons_for_inference(time_labels, base_timeframe)
    else:
        # List of labels
        horizon_bars = time_labels_to_horizon_bars(time_labels, base_timeframe)
    
    # Format as compact string
    result = format_horizon_spec(horizon_bars)
    logger.debug(f"Converted time labels {time_labels} @ {base_timeframe} → {result}")
    
    return result


def normalize_horizon_input(
    horizon_input: str,
    base_timeframe: str,
    output_format: str = 'bars'
) -> str:
    """
    Normalize horizon input to consistent format.
    
    Args:
        horizon_input: User input (can be bars or time labels)
        base_timeframe: Base timeframe
        output_format: 'bars' or 'time'
    
    Returns:
        Normalized string
    
    Examples:
        >>> normalize_horizon_input("15,60,240", "1m", "time")
        "15m,1h,4h"
        
        >>> normalize_horizon_input("15m,1h,4h", "1m", "bars")
        "15,60,240"
    """
    # Detect input format
    input_format = detect_horizon_format(horizon_input)
    
    if input_format == output_format:
        # Already in correct format, just clean up
        if output_format == 'bars':
            bars = parse_horizon_spec(horizon_input)
            return format_horizon_spec(bars)
        else:
            # Clean up time labels
            labels, _ = convert_horizons_for_inference(horizon_input, base_timeframe)
            return ','.join(labels)
    
    # Convert
    if input_format == 'bars' and output_format == 'time':
        return bars_to_time_labels(horizon_input, base_timeframe)
    elif input_format == 'time' and output_format == 'bars':
        return time_labels_to_bars(horizon_input, base_timeframe)
    else:
        raise ValueError(f"Unknown conversion: {input_format} → {output_format}")


def detect_horizon_format(horizon_input: str) -> str:
    """
    Detect whether input is in 'bars' or 'time' format.
    
    Args:
        horizon_input: Input string
    
    Returns:
        'bars' or 'time'
    
    Examples:
        >>> detect_horizon_format("15,60,240")
        'bars'
        
        >>> detect_horizon_format("15m,1h,4h")
        'time'
        
        >>> detect_horizon_format("1-7/2")
        'bars'
    """
    horizon_input = horizon_input.strip()
    
    # Check for time format indicators
    has_time_units = any(c in horizon_input for c in ['m', 'h', 'd', 'w'])
    
    # Check for pure numbers (bars format)
    # Remove range operators and delimiters
    test_str = horizon_input.replace(',', ' ').replace('-', ' ').replace('/', ' ')
    parts = test_str.split()
    
    # If all parts are pure integers, it's bars format
    all_integers = all(p.isdigit() for p in parts)
    
    if has_time_units:
        return 'time'
    elif all_integers:
        return 'bars'
    else:
        # Ambiguous - default to bars
        logger.warning(f"Ambiguous horizon format: {horizon_input}, defaulting to 'bars'")
        return 'bars'


def adapt_horizons_for_training(
    gui_horizons: str,
    base_timeframe: str
) -> str:
    """
    Adapt GUI horizons (time format) to training format (bars).
    
    Args:
        gui_horizons: Horizons from GUI (e.g., "15m,1h,4h")
        base_timeframe: Base timeframe
    
    Returns:
        Bars format for training (e.g., "15,60,240")
    """
    return normalize_horizon_input(gui_horizons, base_timeframe, output_format='bars')


def adapt_horizons_for_inference(
    training_horizons: str,
    base_timeframe: str
) -> str:
    """
    Adapt training horizons (bars format) to inference format (time).
    
    Args:
        training_horizons: Horizons from training (e.g., "15,60,240")
        base_timeframe: Base timeframe
    
    Returns:
        Time format for inference GUI (e.g., "15m,1h,4h")
    """
    return normalize_horizon_input(training_horizons, base_timeframe, output_format='time')


def get_inference_horizons_as_bars(
    gui_horizons: str,
    base_timeframe: str
) -> List[int]:
    """
    Get inference horizons as list of bars (for model prediction).
    
    Args:
        gui_horizons: Horizons from GUI (can be bars or time)
        base_timeframe: Base timeframe
    
    Returns:
        List of horizon bars
    """
    # Normalize to bars format
    bars_str = normalize_horizon_input(gui_horizons, base_timeframe, output_format='bars')
    
    # Parse to list
    return parse_horizon_spec(bars_str)
