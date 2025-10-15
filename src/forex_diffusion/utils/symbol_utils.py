"""
Symbol utilities for market data services.

Provides shared utilities for symbol management, validation, and configuration loading.
"""
from __future__ import annotations

from typing import List
from loguru import logger


def get_symbols_from_config() -> List[str]:
    """
    Get configured trading symbols from application config.
    
    Returns:
        List of symbol strings (e.g., ["EUR/USD", "GBP/USD"])
        Returns empty list if config not available or symbols not configured.
    """
    try:
        from .config import get_config
        cfg = get_config()
        symbols = getattr(cfg.data, "symbols", [])
        if not symbols:
            logger.warning("No symbols configured in config.data.symbols")
        return symbols
    except Exception as e:
        logger.error(f"Failed to load symbols from config: {e}")
        return []


def validate_symbol(symbol: str) -> bool:
    """
    Validate symbol format.
    
    Args:
        symbol: Symbol string to validate
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> validate_symbol("EUR/USD")
        True
        >>> validate_symbol("EURUSD")
        False  # Missing slash
        >>> validate_symbol("EUR")
        False  # Incomplete
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation: must contain slash and have 2 parts
    parts = symbol.split("/")
    if len(parts) != 2:
        return False
    
    # Each part should be 3-4 characters (most forex pairs)
    base, quote = parts
    if not (2 <= len(base) <= 4 and 2 <= len(quote) <= 4):
        return False
    
    # Should be uppercase letters
    if not (base.isalpha() and base.isupper() and quote.isalpha() and quote.isupper()):
        return False
    
    return True


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol format to standard slash-separated uppercase.
    
    Args:
        symbol: Symbol in various formats
        
    Returns:
        Normalized symbol string
        
    Examples:
        >>> normalize_symbol("eurusd")
        "EUR/USD"
        >>> normalize_symbol("EUR-USD")
        "EUR/USD"
        >>> normalize_symbol("eur usd")
        "EUR/USD"
    """
    if not symbol:
        return symbol
    
    # Remove common separators
    clean = symbol.replace("-", "").replace("_", "").replace(" ", "").strip().upper()
    
    # If already has slash, validate and return
    if "/" in clean:
        return clean
    
    # Assume 3+3 or 3+4 format for forex pairs
    if len(clean) == 6:
        return f"{clean[:3]}/{clean[3:]}"
    elif len(clean) == 7:
        # Could be 3+4 or 4+3
        # Default to 3+4 (e.g., EURUSDX -> EUR/USDX)
        return f"{clean[:3]}/{clean[3:]}"
    
    # Return as-is if can't normalize
    return symbol


def get_base_and_quote(symbol: str) -> tuple[str, str]:
    """
    Extract base and quote currencies from symbol.
    
    Args:
        symbol: Symbol like "EUR/USD"
        
    Returns:
        Tuple of (base, quote) currencies
        
    Raises:
        ValueError: If symbol format is invalid
        
    Examples:
        >>> get_base_and_quote("EUR/USD")
        ("EUR", "USD")
    """
    if "/" not in symbol:
        raise ValueError(f"Invalid symbol format: {symbol}. Expected format: 'BASE/QUOTE'")
    
    parts = symbol.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid symbol format: {symbol}. Expected exactly one slash")
    
    base, quote = parts
    return base.strip(), quote.strip()
