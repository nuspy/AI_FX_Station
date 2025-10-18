"""
Horizon Parser Utility

Parses horizon specifications with support for:
- Single values: '5'
- Lists: '15,60,240'
- Ranges with step: '1-7/2' → [1, 3, 5, 7]
- Mixed: '1-7/2,60,100-200/50' → [1, 3, 5, 7, 60, 100, 150, 200]
"""

from typing import List


def parse_horizon_spec(horizon_str: str) -> List[int]:
    """
    Parse horizon specification string into list of integers.
    
    Args:
        horizon_str: Horizon specification string
        
    Formats supported:
        - Single: '5' → [5]
        - List: '15,60,240' → [15, 60, 240]
        - Range: '1-7/2' → [1, 3, 5, 7]
        - Range (step=1): '5-10' → [5, 6, 7, 8, 9, 10]
        - Mixed: '1-7/2,60,100-200/50' → [1, 3, 5, 7, 60, 100, 150, 200]
        
    Returns:
        List of unique horizons, sorted ascending
        
    Examples:
        >>> parse_horizon_spec('5')
        [5]
        >>> parse_horizon_spec('15,60,240')
        [15, 60, 240]
        >>> parse_horizon_spec('1-7/2')
        [1, 3, 5, 7]
        >>> parse_horizon_spec('5-10')
        [5, 6, 7, 8, 9, 10]
        >>> parse_horizon_spec('1-7/2,60,100-200/50')
        [1, 3, 5, 7, 60, 100, 150, 200]
    """
    horizon_str = horizon_str.strip()
    if not horizon_str:
        raise ValueError("Horizon specification cannot be empty")
    
    horizons = []
    
    # Split by comma to get individual tokens
    tokens = [t.strip() for t in horizon_str.split(',')]
    
    for token in tokens:
        if not token:
            continue
            
        # Check if it's a range: x-y or x-y/z
        if '-' in token:
            # Parse range
            parts = token.split('/')
            if len(parts) == 1:
                # Format: x-y (step=1)
                range_part = parts[0]
                step = 1
            elif len(parts) == 2:
                # Format: x-y/z
                range_part = parts[0]
                try:
                    step = int(parts[1].strip())
                    if step <= 0:
                        raise ValueError(f"Step must be positive, got {step}")
                except ValueError as e:
                    raise ValueError(f"Invalid step in '{token}': {e}")
            else:
                raise ValueError(f"Invalid range format '{token}'. Expected 'x-y' or 'x-y/z'")
            
            # Parse x-y
            range_bounds = range_part.split('-')
            if len(range_bounds) != 2:
                raise ValueError(f"Invalid range format '{token}'. Expected 'x-y'")
            
            try:
                start = int(range_bounds[0].strip())
                end = int(range_bounds[1].strip())
            except ValueError as e:
                raise ValueError(f"Invalid range bounds in '{token}': {e}")
            
            if start > end:
                raise ValueError(f"Range start must be <= end in '{token}': {start} > {end}")
            
            # Generate range
            range_values = list(range(start, end + 1, step))
            horizons.extend(range_values)
        else:
            # Single value
            try:
                value = int(token.strip())
                horizons.append(value)
            except ValueError as e:
                raise ValueError(f"Invalid horizon value '{token}': {e}")
    
    # Validate all horizons are positive
    for h in horizons:
        if h <= 0:
            raise ValueError(f"All horizons must be positive, got {h}")
    
    # Remove duplicates and sort
    unique_horizons = sorted(set(horizons))
    
    return unique_horizons


def format_horizon_spec(horizons: List[int]) -> str:
    """
    Format list of horizons as compact string.
    
    Args:
        horizons: List of horizon values
        
    Returns:
        Compact string representation
        
    Examples:
        >>> format_horizon_spec([5])
        '5'
        >>> format_horizon_spec([1, 3, 5, 7])
        '1-7/2'
        >>> format_horizon_spec([15, 60, 240])
        '15,60,240'
        >>> format_horizon_spec([1, 3, 5, 7, 60, 100, 150, 200])
        '1-7/2,60,100-200/50'
    """
    if not horizons:
        return ""
    
    if len(horizons) == 1:
        return str(horizons[0])
    
    # Try to detect ranges
    sorted_h = sorted(set(horizons))
    result_parts = []
    i = 0
    
    while i < len(sorted_h):
        start = sorted_h[i]
        
        # Check if we can form a range
        if i + 2 < len(sorted_h):
            # Need at least 3 values for a range
            step = sorted_h[i + 1] - sorted_h[i]
            
            # Check if subsequent values follow the same step
            j = i + 1
            while j < len(sorted_h) and sorted_h[j] - sorted_h[j - 1] == step:
                j += 1
            
            if j - i >= 3:
                # We have a range of at least 3 values
                end = sorted_h[j - 1]
                if step == 1:
                    result_parts.append(f"{start}-{end}")
                else:
                    result_parts.append(f"{start}-{end}/{step}")
                i = j
                continue
        
        # Single value
        result_parts.append(str(start))
        i += 1
    
    return ','.join(result_parts)


# Aliases for backward compatibility
parse_horizons = parse_horizon_spec


def validate_inference_horizons(model_horizons: List[int], requested_horizons: List[int]) -> tuple[bool, str]:
    """
    Validate that requested inference horizons match model training horizons.
    
    Args:
        model_horizons: Horizons the model was trained on
        requested_horizons: Horizons requested for inference
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Examples:
        >>> validate_inference_horizons([15, 60, 240], [15, 60, 240])
        (True, '')
        >>> validate_inference_horizons([15, 60], [15, 60, 240])
        (False, 'Model trained on [15, 60] but inference requested [15, 60, 240]...')
    """
    if sorted(model_horizons) == sorted(requested_horizons):
        return True, ""
    
    error_msg = (
        f"Horizon mismatch!\n"
        f"Model trained on: {model_horizons}\n"
        f"Inference requested: {requested_horizons}\n\n"
        f"Options:\n"
        f"1. Use same horizons as training: {model_horizons}\n"
        f"2. Re-train model with desired horizons: {requested_horizons}\n"
        f"3. Train separate models for different horizon scales"
    )
    
    return False, error_msg


def get_model_horizons_from_metadata(metadata: dict) -> List[int]:
    """
    Extract horizons from model metadata (handles legacy formats).
    
    Args:
        metadata: Model metadata dictionary
        
    Returns:
        List of horizons
        
    Examples:
        >>> get_model_horizons_from_metadata({'horizons': [15, 60, 240]})
        [15, 60, 240]
        >>> get_model_horizons_from_metadata({'horizon_bars': 60})
        [60]
    """
    # New format: 'horizons' key
    if 'horizons' in metadata:
        return metadata['horizons']
    
    # Legacy format: 'horizon_bars' (single or list)
    if 'horizon_bars' in metadata:
        horizon_bars = metadata['horizon_bars']
        if isinstance(horizon_bars, list):
            return horizon_bars
        else:
            return [horizon_bars]
    
    # Very old format: 'horizon' key
    if 'horizon' in metadata:
        horizon = metadata['horizon']
        if isinstance(horizon, list):
            return horizon
        else:
            return [horizon]
    
    # Fallback
    raise ValueError("No horizon information found in model metadata")
