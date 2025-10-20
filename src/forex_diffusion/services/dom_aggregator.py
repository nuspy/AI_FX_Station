"""
DOM (Depth of Market) aggregator service.

Processes market depth data and calculates derived metrics like:
- Mid price (weighted average)
- Spread
- Order book imbalance
- Liquidity metrics

Refactored to use ThreadedBackgroundService base class.
"""
from __future__ import annotations

from typing import List, Dict, Optional

from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .base_service import ThreadedBackgroundService


class DOMAggregatorService(ThreadedBackgroundService):
    """
    Background service that processes DOM snapshots and calculates metrics.
    
    Inherits from ThreadedBackgroundService for lifecycle management and error recovery.
    """

    def __init__(self, engine: Engine, symbols: List[str] | None = None, interval_seconds: int = 5, provider=None):
        """
        Initialize DOM aggregator service.
        
        Args:
            engine: SQLAlchemy engine for database access
            symbols: List of symbols to process (None = load from config)
            interval_seconds: Interval between DOM processing runs (default: 5s)
            provider: Optional provider instance to read DOM from RAM buffer instead of database
        """
        # Initialize base class with circuit breaker enabled
        super().__init__(
            engine=engine,
            symbols=symbols,
            interval_seconds=interval_seconds,
            enable_circuit_breaker=True
        )
        self.provider = provider  # Store provider reference for RAM buffer access
        self._symbol_format_cache: Dict[str, str] = {}  # Cache: requested_symbol -> buffer_symbol
    
    @property
    def service_name(self) -> str:
        """Service name for logging."""
        return "DOMAggregatorService"
    
    def _process_iteration(self):
        """
        Process one DOM aggregation iteration.
        
        Called by base class in background thread. Processes DOM snapshots
        and calculates metrics for all configured symbols.
        """
        symbols = self.get_symbols()  # Use base class method
        for sym in symbols:
            self._process_dom_for_symbol(sym)

    def _process_dom_for_symbol(self, symbol: str):
        """Process latest DOM snapshot for a symbol."""
        try:
            # Get latest DOM snapshot
            with self.engine.connect() as conn:
                query = text(
                    "SELECT id, ts_utc, bids, asks, mid_price, spread, imbalance "
                    "FROM market_depth "
                    "WHERE symbol = :symbol "
                    "ORDER BY ts_utc DESC LIMIT 1"
                )
                row = conn.execute(query, {"symbol": symbol}).fetchone()

            if not row:
                return

            dom_id, ts_utc, bids, asks, mid_price, spread, imbalance = row

            # If metrics already calculated, skip
            if mid_price is not None:
                return

            # Parse JSON bids/asks
            import json
            bids_list = json.loads(bids) if isinstance(bids, str) else bids
            asks_list = json.loads(asks) if isinstance(asks, str) else asks

            if not bids_list or not asks_list:
                return

            # Calculate derived metrics
            metrics = self._calculate_dom_metrics(bids_list, asks_list)

            # Update database
            with self.engine.begin() as conn:
                update = text(
                    "UPDATE market_depth SET "
                    "mid_price = :mid_price, "
                    "spread = :spread, "
                    "imbalance = :imbalance "
                    "WHERE id = :dom_id"
                )
                conn.execute(update, {
                    "mid_price": metrics["mid_price"],
                    "spread": metrics["spread"],
                    "imbalance": metrics["imbalance"],
                    "dom_id": dom_id
                })

            logger.debug(
                f"DOM metrics calculated for {symbol}: "
                f"mid={metrics['mid_price']:.5f}, spread={metrics['spread']:.5f}, "
                f"imbalance={metrics['imbalance']:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to process DOM for {symbol}: {e}")

    def _calculate_dom_metrics(self, bids: List, asks: List) -> Dict[str, float]:
        """
        Calculate DOM metrics from bids/asks.

        Args:
            bids: List of [price, volume] sorted descending by price
            asks: List of [price, volume] sorted ascending by price

        Returns:
            Dictionary with mid_price, spread, imbalance
        """
        # Best bid/ask
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0

        # Mid price (simple average of best bid/ask)
        mid_price = (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0

        # Spread
        spread = best_ask - best_bid if best_bid and best_ask else 0.0

        # Order book imbalance (bid volume / ask volume)
        bid_volume = sum(vol for _, vol in bids)
        ask_volume = sum(vol for _, vol in asks)
        imbalance = bid_volume / ask_volume if ask_volume > 0 else 0.0

        # Note: Weighted mid price calculation removed (was commented out, not used)
        # If needed in future, implement as separate method with clear documentation

        return {
            "mid_price": mid_price,
            "spread": spread,
            "imbalance": imbalance,
        }



    def get_latest_dom_metrics(self, symbol: str) -> Optional[Dict]:
        """Get latest DOM metrics for a symbol."""
        try:
            with self.engine.connect() as conn:
                query = text(
                    "SELECT ts_utc, mid_price, spread, imbalance "
                    "FROM market_depth "
                    "WHERE symbol = :symbol AND mid_price IS NOT NULL "
                    "ORDER BY ts_utc DESC LIMIT 1"
                )
                row = conn.execute(query, {"symbol": symbol}).fetchone()

            if not row:
                return None

            ts_utc, mid_price, spread, imbalance = row
            return {
                "timestamp": ts_utc,
                "mid_price": mid_price,
                "spread": spread,
                "imbalance": imbalance,
            }

        except Exception as e:
            logger.error(f"Failed to get DOM metrics for {symbol}: {e}")
            return None

    def get_latest_dom_snapshot(self, symbol: str) -> Optional[Dict]:
        """
        Get complete DOM snapshot with full order book data.
        
        Reads from provider RAM buffer if available (faster),
        otherwise falls back to database.

        Returns:
            Dictionary with bids, asks arrays and computed metrics,
            or None if no data available.
        """
        try:
            # TRY RAM BUFFER FIRST (faster, real-time)
            if self.provider and hasattr(self.provider, '_dom_buffer'):
                # Check cache first
                if symbol in self._symbol_format_cache:
                    cached_format = self._symbol_format_cache[symbol]
                    if cached_format in self.provider._dom_buffer:
                        return self.provider._dom_buffer[cached_format]
                
                # Cache miss - try variants
                buffer_keys = list(self.provider._dom_buffer.keys())
                
                # Try multiple symbol formats (EURUSD, EUR/USD, EUR-USD)
                symbol_variants = [
                    symbol,  # Original (e.g., EURUSD)
                    symbol[:3] + '/' + symbol[3:] if len(symbol) == 6 and '/' not in symbol else symbol,  # Add slash (EURUSD → EUR/USD)
                    symbol[:3] + '-' + symbol[3:] if len(symbol) == 6 and '-' not in symbol else symbol,  # Add dash (EURUSD → EUR-USD)
                    symbol.replace('/', ''),  # Remove slash (EUR/USD → EURUSD)
                    symbol.replace('-', ''),  # Remove dash (EUR-USD → EURUSD)
                ]
                
                for sym_variant in symbol_variants:
                    if sym_variant in self.provider._dom_buffer:
                        # Cache the successful format
                        self._symbol_format_cache[symbol] = sym_variant
                        logger.info(f"✓ DOM snapshot for {symbol} (found as '{sym_variant}', cached) retrieved from RAM buffer")
                        return self.provider._dom_buffer[sym_variant]
                
                logger.warning(f"⚠️ DOM for {symbol} not found in buffer. Buffer has: {buffer_keys}")
            
            # FALLBACK TO DATABASE (for historical or if provider not available)
            with self.engine.connect() as conn:
                query = text(
                    "SELECT ts_utc, bids, asks, mid_price, spread, imbalance "
                    "FROM market_depth "
                    "WHERE symbol = :symbol "
                    "ORDER BY ts_utc DESC LIMIT 1"
                )
                row = conn.execute(query, {"symbol": symbol}).fetchone()

            if not row:
                return None

            ts_utc, bids, asks, mid_price, spread, imbalance = row

            # Parse JSON bids/asks
            import json
            bids_list = json.loads(bids) if isinstance(bids, str) else bids
            asks_list = json.loads(asks) if isinstance(asks, str) else asks

            if not bids_list or not asks_list:
                return None

            # Calculate additional metrics
            best_bid = bids_list[0][0] if bids_list else 0.0
            best_ask = asks_list[0][0] if asks_list else 0.0

            # Calculate depth (top 20 levels)
            max_levels = min(20, len(bids_list), len(asks_list))
            bid_depth = sum(vol for _, vol in bids_list[:max_levels])
            ask_depth = sum(vol for _, vol in asks_list[:max_levels])

            # Calculate order flow imbalance (-1 to +1)
            total_depth = bid_depth + ask_depth
            depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

            return {
                "symbol": symbol,
                "timestamp": ts_utc,
                "bids": bids_list,
                "asks": asks_list,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": mid_price or (best_bid + best_ask) / 2.0,
                "spread": spread or (best_ask - best_bid),
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "depth_imbalance": depth_imbalance,
                "imbalance": imbalance or 0.0,
            }

        except Exception as e:
            logger.error(f"Failed to get DOM snapshot for {symbol}: {e}")
            return None
