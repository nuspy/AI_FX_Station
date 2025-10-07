# WebSocket Integration - Quick Fix Implementation Guide

**Target:** Make order book data flow from cTrader WebSocket → GUI
**Estimated Time:** 4-6 hours
**Difficulty:** Medium

---

## Quick Fix Strategy

Instead of implementing the full Twisted→asyncio bridge, we'll create a **simplified polling-based order book fetcher** that works immediately, then upgrade to WebSocket later.

---

## Phase 1: Immediate Fix (Polling-Based) - 2 hours

### Step 1: Create OrderBook Polling Service

**File:** `src/forex_diffusion/services/orderbook_poller.py`

```python
"""
Order book polling service - temporary solution until WebSocket is complete.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger
from sqlalchemy import text

class OrderBookPoller:
    """Poll cTrader for order book snapshots and write to database."""

    def __init__(self, engine, provider, symbols: List[str], interval: float = 5.0):
        """
        Args:
            engine: SQLAlchemy engine
            provider: BaseProvider instance (must have get_market_depth capability)
            symbols: List of symbols to poll
            interval: Polling interval in seconds
        """
        self.engine = engine
        self.provider = provider
        self.symbols = symbols
        self.interval = interval
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        """Start polling in background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("OrderBookPoller already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"OrderBookPoller started: symbols={self.symbols}, interval={self.interval}s")

    def stop(self):
        """Stop polling."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("OrderBookPoller stopped")

    def _run_loop(self):
        """Main polling loop."""
        while not self._stop_event.is_set():
            for symbol in self.symbols:
                if self._stop_event.is_set():
                    break
                try:
                    self._poll_and_store(symbol)
                except Exception as e:
                    logger.error(f"OrderBookPoller error for {symbol}: {e}")

            time.sleep(self.interval)

    def _poll_and_store(self, symbol: str):
        """Poll order book and store in database."""
        try:
            # Get market depth from provider
            depth = self.provider.get_market_depth(symbol, levels=10)

            if not depth or 'bids' not in depth or 'asks' not in depth:
                logger.debug(f"No depth data for {symbol}")
                return

            # Store in database
            import json
            timestamp = int(time.time() * 1000)

            with self.engine.begin() as conn:
                query = text(
                    "INSERT INTO market_depth (symbol, ts_utc, bids, asks, provider) "
                    "VALUES (:symbol, :ts_utc, :bids, :asks, :provider)"
                )
                conn.execute(query, {
                    'symbol': symbol,
                    'ts_utc': timestamp,
                    'bids': json.dumps(depth['bids']),
                    'asks': json.dumps(depth['asks']),
                    'provider': self.provider.name
                })

            logger.debug(f"Stored order book for {symbol}: {len(depth['bids'])} bids, {len(depth['asks'])} asks")

        except Exception as e:
            logger.exception(f"Failed to poll order book for {symbol}: {e}")
```

---

### Step 2: Update cTrader Provider - Implement get_market_depth()

**File:** `src/forex_diffusion/providers/ctrader_provider.py`

Replace the placeholder `_get_market_depth_impl()` (lines 373-394):

```python
async def _get_market_depth_impl(self, symbol: str, levels: int) -> Optional[Dict[str, Any]]:
    """Get market depth (DOM) snapshot from cTrader."""
    try:
        await self._rate_limit_wait()

        if not self.client or not self.health.is_connected:
            await self.connect()

        symbol_id = await self._get_symbol_id(symbol)

        # Request depth snapshot
        request = Messages.ProtoOASubscribeSpotsReq()
        request.ctidTraderAccountId = self._account_id
        request.symbolId.append(symbol_id)

        response = await self._send_and_wait(request, Messages.ProtoOASubscribeSpotsRes)

        if not response:
            logger.warning(f"No DOM response for {symbol}")
            return None

        # Parse depth from response
        # NOTE: cTrader doesn't have a direct DOM endpoint - we fake it from spot prices
        # In production, you'd subscribe to depth updates via WebSocket
        # For now, return synthetic depth based on bid/ask

        if hasattr(response, 'spot') and response.spot:
            spot = response.spot[0]
            bid = spot.bid / 100000
            ask = spot.ask / 100000
            spread = ask - bid

            # Synthetic depth (replace with real DOM when WebSocket is ready)
            bids = [[bid - (i * spread), 10000.0 * (1.0 - i * 0.1)] for i in range(levels)]
            asks = [[ask + (i * spread), 10000.0 * (1.0 - i * 0.1)] for i in range(levels)]

            return {
                'symbol': symbol,
                'timestamp': int(time.time() * 1000),
                'bids': bids,  # [[price, volume], ...]
                'asks': asks,
                'levels': levels
            }

        return None

    except Exception as e:
        logger.error(f"[{self.name}] Failed to get market depth for {symbol}: {e}")
        self.health.errors.append(str(e))
        return None
```

**Note:** This returns **synthetic depth** based on spot prices. It's a temporary solution until true DOM WebSocket is implemented.

---

### Step 3: Connect to GUI

**File:** `src/forex_diffusion/ui/app.py` (or wherever main GUI is initialized)

Add after database and provider initialization:

```python
from forex_diffusion.services.orderbook_poller import OrderBookPoller
from forex_diffusion.services.dom_aggregator import DOMAggregatorService
from forex_diffusion.analysis.order_flow_analyzer import OrderFlowAnalyzer

# Initialize services
orderbook_poller = OrderBookPoller(
    engine=db_engine,
    provider=market_service.provider,
    symbols=['EURUSD', 'GBPUSD'],
    interval=5.0  # Poll every 5 seconds
)

dom_aggregator = DOMAggregatorService(
    engine=db_engine,
    symbols=['EURUSD', 'GBPUSD'],
    interval_seconds=3  # Process every 3 seconds
)

order_flow_analyzer = OrderFlowAnalyzer()

# Start background services
orderbook_poller.start()
dom_aggregator.start()

# Connect to OrderFlowPanel (assuming you have instance)
def update_order_flow():
    """Update order flow panel with latest data."""
    try:
        symbol = order_flow_panel.symbol_combo.currentText()

        # Get latest DOM metrics
        metrics = dom_aggregator.get_latest_dom_metrics(symbol)

        if metrics:
            # Convert to OrderFlowPanel format
            panel_data = {
                'spread': metrics['spread'],
                'spread_zscore': 0.0,  # Would need historical stats
                'bid_depth': 0.0,  # Not in DOMAggregator - need to add
                'ask_depth': 0.0,
                'buy_volume': 0.0,  # Would come from bar data
                'sell_volume': 0.0,
                'depth_imbalance': metrics['imbalance'],
                'volume_imbalance': 0.0,  # Need to calculate
                'large_order_detected': False,
                'absorption_detected': False,
                'exhaustion_detected': False
            }

            order_flow_panel.update_metrics(panel_data)

    except Exception as e:
        logger.error(f"Error updating order flow panel: {e}")

# Connect to timer
order_flow_timer = QTimer()
order_flow_timer.timeout.connect(update_order_flow)
order_flow_timer.start(2000)  # Update every 2 seconds
```

---

## Phase 2: Enhanced Integration - 2-3 hours

### Step 4: Enhance DOMAggregatorService

**File:** `src/forex_diffusion/services/dom_aggregator.py`

Add method to return full metrics for OrderFlowAnalyzer:

```python
def get_full_dom_data(self, symbol: str) -> Optional[Dict[str, Any]]:
    """Get full DOM data including bids/asks for analyzer."""
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

        # Parse JSON
        import json
        bids_list = json.loads(bids) if isinstance(bids, str) else bids
        asks_list = json.loads(asks) if isinstance(asks, str) else asks

        return {
            'timestamp': ts_utc,
            'bids': bids_list,
            'asks': asks_list,
            'mid_price': mid_price,
            'spread': spread,
            'imbalance': imbalance,
            'bid_depth': sum(vol for _, vol in bids_list),
            'ask_depth': sum(vol for _, vol in asks_list)
        }

    except Exception as e:
        logger.error(f"Failed to get full DOM data for {symbol}: {e}")
        return None
```

---

### Step 5: Full OrderFlowAnalyzer Integration

**File:** Update GUI connection code:

```python
def update_order_flow_full():
    """Full order flow analysis with signals."""
    try:
        symbol = order_flow_panel.symbol_combo.currentText()

        # Get full DOM data
        dom_data = dom_aggregator.get_full_dom_data(symbol)

        if not dom_data:
            return

        # Get current price and volume (from bar data)
        # This would come from your market data service
        current_price = dom_data['mid_price']
        buy_volume = 1000.0  # Placeholder - get from bar data
        sell_volume = 1000.0

        # Compute order flow metrics
        of_metrics = order_flow_analyzer.compute_metrics(
            timestamp=dom_data['timestamp'],
            symbol=symbol,
            timeframe='1m',
            bid_price=dom_data['bids'][0][0] if dom_data['bids'] else current_price,
            ask_price=dom_data['asks'][0][0] if dom_data['asks'] else current_price,
            bid_size=dom_data['bid_depth'],
            ask_size=dom_data['ask_depth'],
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            current_price=current_price
        )

        # Update GUI
        order_flow_panel.update_metrics({
            'spread': of_metrics.bid_ask_spread,
            'spread_zscore': of_metrics.spread_zscore,
            'bid_depth': of_metrics.bid_depth,
            'ask_depth': of_metrics.ask_depth,
            'buy_volume': of_metrics.buy_volume,
            'sell_volume': of_metrics.sell_volume,
            'depth_imbalance': of_metrics.depth_imbalance,
            'volume_imbalance': of_metrics.volume_imbalance,
            'large_order_detected': of_metrics.large_order_count > 0,
            'large_order_direction': 'buy' if of_metrics.volume_imbalance > 0 else 'sell',
            'absorption_detected': of_metrics.absorption_detected,
            'exhaustion_detected': of_metrics.exhaustion_detected
        })

        # Generate and display signals
        atr = current_price * 0.01  # 1% ATR estimate
        signals = order_flow_analyzer.generate_signals(of_metrics, current_price, atr)

        # Convert to GUI format
        signal_dicts = []
        for sig in signals:
            signal_dicts.append({
                'timestamp': sig.timestamp,
                'signal_type': sig.signal_type.value,
                'direction': sig.direction,
                'strength': sig.strength,
                'confidence': sig.confidence,
                'status': 'active'
            })

        order_flow_panel.update_signals(signal_dicts)

    except Exception as e:
        logger.exception(f"Error in full order flow update: {e}")
```

---

## Phase 3: Real-time Volume - 1-2 hours

### Step 6: Volume from Bar Data

**File:** Modify update function to get volume from aggregated bars:

```python
def get_realtime_volume(symbol: str, timeframe: str = '1m') -> Tuple[float, float]:
    """Get buy/sell volume from latest bar."""
    try:
        with db_engine.connect() as conn:
            query = text(
                "SELECT volume, close, open "
                "FROM candle_bars "
                "WHERE symbol = :symbol AND timeframe = :timeframe "
                "ORDER BY ts_utc DESC LIMIT 1"
            )
            row = conn.execute(query, {'symbol': symbol, 'timeframe': timeframe}).fetchone()

        if not row:
            return 0.0, 0.0

        volume, close, open_price = row

        # Estimate buy/sell split based on bar direction
        if close > open_price:
            # Bullish bar - more buying
            buy_volume = volume * 0.6
            sell_volume = volume * 0.4
        elif close < open_price:
            # Bearish bar - more selling
            buy_volume = volume * 0.4
            sell_volume = volume * 0.6
        else:
            # Neutral
            buy_volume = volume * 0.5
            sell_volume = volume * 0.5

        return buy_volume, sell_volume

    except Exception as e:
        logger.error(f"Failed to get volume for {symbol}: {e}")
        return 0.0, 0.0
```

Then use in update function:

```python
buy_volume, sell_volume = get_realtime_volume(symbol, '1m')
```

---

## Testing Checklist

### Immediate Fix Testing

- [ ] Run `orderbook_poller.start()` and check database writes
- [ ] Verify `market_depth` table has new rows every 5 seconds
- [ ] Check `DOMAggregatorService` calculates metrics
- [ ] Confirm `OrderFlowPanel` displays spread and imbalance
- [ ] Monitor logs for errors

### Commands to Run

```bash
# Check database writes
sqlite3 forexgpt.db "SELECT COUNT(*) FROM market_depth WHERE ts_utc > $(date -d '5 minutes ago' +%s)000"

# View latest order book
sqlite3 forexgpt.db "SELECT symbol, datetime(ts_utc/1000, 'unixepoch'), bids, asks FROM market_depth ORDER BY ts_utc DESC LIMIT 5"

# Check metrics calculation
sqlite3 forexgpt.db "SELECT symbol, datetime(ts_utc/1000, 'unixepoch'), spread, imbalance FROM market_depth WHERE mid_price IS NOT NULL ORDER BY ts_utc DESC LIMIT 10"
```

---

## Performance Expectations

### Polling-Based (Phase 1)
- **Latency:** 5 seconds (polling interval)
- **CPU:** Negligible (1 query per symbol every 5s)
- **Network:** Minimal (1 API call per symbol every 5s)

### With Full Integration (Phase 2)
- **GUI Update:** 2 seconds
- **Signal Generation:** Real-time (on every update)
- **Database Load:** ~20 writes/minute per symbol

---

## Migration to WebSocket (Future)

Once this is working, migrate to true WebSocket:

1. **Replace OrderBookPoller** with WebSocket subscriber
2. **Keep DOMAggregatorService** unchanged (same data flow)
3. **Keep GUI connection** unchanged
4. **Benefits:**
   - Sub-second latency (vs 5s polling)
   - Real-time order book updates
   - True tick volume data

---

## Troubleshooting

### Issue: No data in market_depth table
**Fix:** Check provider connection, verify `get_market_depth()` returns data

### Issue: GUI not updating
**Fix:** Check timer is running, verify `update_order_flow()` is called

### Issue: Wrong metrics displayed
**Fix:** Check data format in `update_metrics()`, ensure all fields present

### Issue: High CPU usage
**Fix:** Increase polling interval, reduce DOM levels

---

## Conclusion

This quick fix provides:
- ✅ Order book data in GUI **immediately**
- ✅ All analysis components working
- ✅ Real-time alerts (with 2-5s delay)
- ✅ Foundation for WebSocket upgrade

**Total Implementation Time:** 4-6 hours

**Next Steps:**
1. Implement Phase 1 (polling)
2. Test with demo account
3. Verify GUI updates
4. Proceed to Phase 2 (full integration)
5. Plan WebSocket migration

---

**Generated by:** Claude Code
**Last Updated:** 2025-10-07
