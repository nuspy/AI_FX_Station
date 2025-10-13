"""
DOM (Depth of Market) aggregator service.

Processes market depth data and calculates derived metrics like:
- Mid price (weighted average)
- Spread
- Order book imbalance
- Liquidity metrics
"""
from __future__ import annotations

import threading
import time
from typing import List, Dict, Optional
from datetime import datetime, timezone
from collections import deque

import pandas as pd
from loguru import logger
from sqlalchemy import text

from .db_service import DBService


class DOMAggregatorService:
    """
    Background service that processes DOM snapshots and calculates metrics.
    """

    def __init__(self, engine, symbols: List[str] | None = None, interval_seconds: int = 5):
        self.engine = engine
        self.db = DBService(engine=self.engine)
        self._symbols = symbols or []
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread = None

        # Cache for recent DOM data (avoid recalculation)
        self._dom_cache: Dict[str, deque] = {}
        self._cache_size = 100  # Keep last 100 snapshots per symbol

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"DOMAggreg atorService started (interval={self._interval}s, symbols={self._symbols or '<all>'})")

    def stop(self, timeout: float = 2.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("DOMAggreg atorService stopped")

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                symbols = self._symbols or self._get_symbols_from_config()
                for sym in symbols:
                    self._process_dom_for_symbol(sym)
            except Exception as e:
                logger.exception(f"DOMAggreg atorService loop error: {e}")

            time.sleep(self._interval)

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

        # Weighted mid price (volume-weighted average)
        # weighted_bid = sum(price * vol for price, vol in bids[:5]) / sum(vol for _, vol in bids[:5]) if bids else 0.0
        # weighted_ask = sum(price * vol for price, vol in asks[:5]) / sum(vol for _, vol in asks[:5]) if asks else 0.0
        # weighted_mid = (weighted_bid + weighted_ask) / 2.0

        return {
            "mid_price": mid_price,
            "spread": spread,
            "imbalance": imbalance,
        }

    def _get_symbols_from_config(self) -> List[str]:
        from ..utils.config import get_config
        cfg = get_config()
        return getattr(cfg.data, "symbols", [])

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

        Returns:
            Dictionary with bids, asks arrays and computed metrics,
            or None if no data available.
        """
        try:
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
