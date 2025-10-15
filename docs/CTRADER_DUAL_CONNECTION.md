# cTrader Dual Connection Architecture

## Overview

ForexGPT uses **TWO separate cTrader connections** for different purposes. This is **by design** and necessary for proper functionality.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ForexGPT Application                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────┐  ┌─────────────────────────┐ │
│  │   CTraderProvider        │  │ CTraderWebSocketService │ │
│  │   (via CTraderClient)    │  │                         │ │
│  ├──────────────────────────┤  ├─────────────────────────┤ │
│  │ Purpose:                 │  │ Purpose:                │ │
│  │ - Historical data        │  │ - Real-time ticks       │ │
│  │ - REST API calls         │  │ - DOM streaming         │ │
│  │ - Backfill candles       │  │ - Volume data           │ │
│  │ - Symbol info            │  │ - Sentiment tracking    │ │
│  ├──────────────────────────┤  ├─────────────────────────┤ │
│  │ Connection:              │  │ Connection:             │ │
│  │ - Twisted WebSocket      │  │ - Twisted WebSocket     │ │
│  │ - Shared reactor         │  │ - Shared reactor        │ │
│  │ - Auth: App + Account    │  │ - Auth: App + Account   │ │
│  └──────────────────────────┘  └─────────────────────────┘ │
│           │                              │                   │
│           └──────────────┬───────────────┘                   │
│                          │                                   │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  cTrader Open API      │
              │  (demo.ctraderapi.com) │
              └────────────────────────┘
```

---

## Why Two Connections?

### **Connection 1: CTraderProvider (Historical Data)**
- **Initialized by**: `MarketDataService` at app startup
- **Used for**:
  - Fetching historical candles (1m, 5m, 15m, etc.)
  - Backfilling missing data
  - Getting symbol information
  - REST-like synchronous operations
- **Authentication**: 
  1. `ProtoOAApplicationAuthReq` (app credentials)
  2. `ProtoOAAccountAuthReq` (trading account)

### **Connection 2: CTraderWebSocketService (Real-Time Streams)**
- **Initialized by**: `app.py` when cTrader is primary provider
- **Used for**:
  - Real-time tick data (ProtoOASpotEvent)
  - DOM/Depth updates (ProtoOADepthEvent) 
  - Volume tracking
  - Sentiment calculation from order flow
  - Writing to `market_depth` table
- **Authentication**:
  1. `ProtoOAApplicationAuthReq` (same app credentials)
  2. `ProtoOAAccountAuthReq` (same trading account)

---

## Reactor Sharing Strategy

Both connections use **Twisted reactor** but avoid conflicts:

```python
# CTraderProvider starts reactor first
def run_reactor():
    reactor.run(installSignalHandlers=False)
    
reactor_thread = threading.Thread(target=run_reactor, daemon=True)
reactor_thread.start()

# CTraderWebSocketService detects and reuses reactor
if reactor.running:
    logger.info("✓ Reusing existing reactor")
    # No new thread needed
else:
    # Start new reactor thread
    reactor_thread.start()
```

### **Key Points:**
1. ✅ Only ONE Twisted reactor runs (in a single thread)
2. ✅ Both services create separate `Client` instances
3. ✅ Each client has its own WebSocket connection
4. ✅ Both authenticate independently (normal cTrader API behavior)

---

## Authentication Flow

### **Expected Startup Sequence:**

```
Time    Component              Action
──────  ────────────────────  ────────────────────────────────────
T+0s    MarketDataService     Initialize CTraderClient
T+0.5s  CTraderProvider       Connect → Auth Application
T+1s    CTraderProvider       Auth Account (ID: 44838933)
T+1.5s  CTraderProvider       ✓ Connected successfully
T+2s    App.py                Initialize CTraderWebSocketService
T+2.5s  WebSocketService      Reuse reactor → Connect
T+3s    WebSocketService      Auth Application (SAME credentials)
T+3.5s  WebSocketService      Auth Account (SAME ID: 44838933)
T+4s    WebSocketService      ✓ Connected successfully
T+4.5s  WebSocketService      Subscribe to symbols for streaming
```

### **Why 2 Authentications?**

Each WebSocket connection to cTrader API must:
1. Authenticate the **application** (client_id + client_secret)
2. Authenticate the **trading account** (access_token + account_id)

This is **required by cTrader API** - even if using the same credentials, each connection authenticates independently.

---

## Benefits of Dual Connection

| Feature | CTraderProvider | WebSocketService | Why Separate? |
|---------|-----------------|------------------|---------------|
| **Historical Data** | ✅ Primary | ❌ No | Synchronous batch requests |
| **Real-Time Ticks** | ❌ No | ✅ Primary | Asynchronous streaming |
| **DOM Streaming** | ❌ No | ✅ Primary | Real-time order book updates |
| **Backfill** | ✅ Primary | ❌ No | Large date ranges, REST-like |
| **Trading** | ✅ Can be used | ❌ No | Direct access via provider |

**Separation of Concerns:**
- Historical data fetching doesn't block real-time streams
- Real-time streams don't interfere with bulk backfill operations
- Each connection can be restarted independently if issues occur

---

## Alternative Considered (Single Connection)

### ❌ Why NOT Use Single Connection?

**Option A: Only CTraderProvider**
- ❌ Would need to poll for real-time data (inefficient)
- ❌ No true streaming (DOM updates delayed)
- ❌ Higher latency for live trading

**Option B: Only WebSocketService**
- ❌ Complex to handle both streaming and REST requests
- ❌ Historical data fetching blocks real-time stream
- ❌ Single point of failure

**Option C: Shared Client Instance**
- ❌ Thread-safety issues (one client, multiple consumers)
- ❌ Message routing complexity (which handler gets which message?)
- ❌ Twisted reactor limitations (callback conflicts)

---

## Troubleshooting

### ⚠️ "Multiple authentications detected"

**This is NORMAL behavior.** You should see:
```
[12:50:41] INFO | Authenticating application...      ← Connection 1
[12:50:42] INFO | Application authenticated ✓
[12:50:42] INFO | Account authenticated ✓             ← Connection 1 complete
[12:50:43] INFO | Authenticating application...      ← Connection 2
[12:50:44] INFO | Application authenticated ✓
[12:50:44] INFO | Account authenticated ✓             ← Connection 2 complete
```

**Both connections are needed** - don't disable either one.

### ❌ "Reactor already running" error

**Fixed in code.** WebSocketService now detects and reuses reactor:
```python
if reactor.running:
    logger.info("✓ Reusing existing reactor")
```

### ❌ "No DOM data available"

Check:
1. Is `CTraderWebSocketService` started? (look for log: "✓ cTrader WebSocket service started")
2. Is account type `demo` or `live`? (demo may not have DOM access)
3. Check `market_depth` table: `SELECT COUNT(*) FROM market_depth;`

---

## Configuration

### Enable/Disable WebSocket Service

```python
# In user_settings or config
ctrader_enabled = True  # Enable WebSocket streaming (recommended)
```

### When to Disable WebSocket Service?

Only disable if:
- ❌ You're ONLY backtesting (no live trading)
- ❌ You're ONLY training AI models (no real-time data needed)
- ❌ You want to reduce connection overhead

**DO NOT disable if**:
- ✅ You're doing live trading
- ✅ You want real-time market monitoring
- ✅ You need DOM/order flow analysis

---

## Performance Impact

### Resource Usage (Both Connections):
- **Memory**: ~50-80 MB total (both services)
- **CPU**: <5% idle, <15% during market hours
- **Network**: ~10-50 KB/s streaming (varies by # of symbols)
- **Database**: ~100-500 inserts/min to `market_depth` table

### Single Connection (Hypothetical):
- **Memory**: ~30-40 MB (only CTraderProvider)
- **CPU**: <3% idle, <10% during polling
- **Network**: ~5-20 KB/s (REST polling only)
- **Database**: ~10-20 inserts/min (polled data)

**Trade-off**: Dual connection uses more resources but provides **10x faster data updates** for live trading.

---

## Summary

✅ **Two cTrader connections are INTENTIONAL and NECESSARY**

✅ **Two authentications are NORMAL cTrader API behavior**

✅ **Reactor is SHARED (no conflict)**

✅ **Each connection serves a DIFFERENT purpose**

❌ **DO NOT disable WebSocketService for live trading**

---

**Last Updated**: 2025-01-14  
**Author**: ForexGPT Development Team
