#!/usr/bin/env python3
"""
Manual test helper: insert a tick/candle row into market_data_candles and publish an internal 'tick' event.
Run this while the GUI is running to simulate realtime tick arrival without websockets.
#!/usr/bin/env python3

Manual test helper: try to send a tick via local WebSocket server (preferred).
If WS not available, try to insert a row into DB and (best-effort) publish internal event.
This helps debugging when LocalWebsocketServer is running (GUI receives tick)
or when DB is available for direct insertion.

Usage:
  python tests/manual_tests/insert_tick_db.py --symbol "EUR/USD" --timeframe "1m" --price 1.23456 --bid 1.23450 --ask 1.23462
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import socket

def build_payload(symbol: str, timeframe: str, price: float, bid: float | None, ask: float | None):
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "ts_utc": int(time.time() * 1000),
        "price": float(price),
        "bid": float(bid) if bid is not None else None,
        "ask": float(ask) if ask is not None else None,
    }

async def try_ws_send(uri: str, payload: dict):
    try:
        import websockets
    except Exception as e:
        raise RuntimeError("Missing dependency 'websockets'. Install with: pip install websockets") from e

    # short timeout connect
    try:
        async with websockets.connect(uri, open_timeout=2) as ws:
            await ws.send(json.dumps(payload))
            return True, "sent via ws"
    except Exception as e:
        return False, e

def try_socket_connect(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def try_db_insert(payload: dict):
    # attempt to import DBService and insert; return tuple(success,msg)
    try:
        # ensure package is importable
        import sys
        ROOT = Path(__file__).resolve().parents[2]
        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))
        from src.forex_diffusion.services.db_service import DBService
    except Exception as e:
        return False, f"Failed to import DBService: {e}"

    try:
        from sqlalchemy import MetaData, insert
        db = DBService()
    except Exception as e:
        return False, f"Failed to instantiate DBService: {e}"

    try:
        meta = MetaData()
        meta.reflect(bind=db.engine, only=["market_data_candles"])
        tbl = meta.tables.get("market_data_candles")
        if tbl is None:
            return False, "Table 'market_data_candles' not present in DB."
        with db.engine.connect() as conn:
            stmt = insert(tbl).values(
                symbol=payload["symbol"],
                timeframe=payload["timeframe"],
                ts_utc=int(payload["ts_utc"]),
                open=float(payload["price"]),
                high=float(payload["price"]),
                low=float(payload["price"]),
                close=float(payload["price"]),
                volume=None,
                resampled=False,
            )
            conn.execute(stmt)
            conn.commit()
        return True, "inserted into DB"
    except Exception as e:
        # if sqlite cannot open file, provide actionable hint
        msg = str(e)
        if "unable to open database file" in msg.lower() or "sqlite3.OperationalError" in msg:
            return False, f"SQLite OperationalError: {e}. Check configs/default.yaml for DB path and ensure directory exists and is writable."
        return False, f"DB insert failed: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EUR/USD")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--price", type=float, default=1.23456)
    parser.add_argument("--bid", type=float, default=1.23450)
    parser.add_argument("--ask", type=float, default=1.23462)
    parser.add_argument("--ws-uri", default="ws://127.0.0.1:8765")
    args = parser.parse_args()

    payload = build_payload(args.symbol, args.timeframe, args.price, args.bid, args.ask)
    print("Payload:", payload)

    # 1) Try WS quickly if server reachable (fast path)
    host = "127.0.0.1"
    port = 8765
    ws_reachable = try_socket_connect(host, port, timeout=0.6)
    if ws_reachable:
        print(f"WebSocket port {host}:{port} reachable -> attempting WS send")
        import asyncio
        ok, info = asyncio.run(try_ws_send(args.ws_uri, payload))
        if ok:
            print("OK (WS):", info)
            return
        else:
            print("WS send failed:", info)

    # 2) Fallback: try DB insert
    print("Attempting DB insert fallback...")
    ok_db, msg_db = try_db_insert(payload)
    if ok_db:
        print("OK (DB):", msg_db)
        # Try to publish via event_bus for in-process GUI (best-effort)
        try:
            from src.forex_diffusion.utils.event_bus import publish
            try:
                publish("tick", payload)
                print("Published 'tick' via event_bus")
            except Exception as ee:
                print("event_bus.publish failed:", ee)
        except Exception:
            pass
        return
    else:
        print("DB fallback failed:", msg_db)
        print()
        print("Suggested actions to fix DB error:")
        print(" - Open configs/default.yaml and check 'database' connection string or sqlite path.")
        print(" - If path is sqlite file, ensure the directory exists and is writable:")
        print("     python -c \"from pathlib import Path; p=Path('path/to/your.db'); p.parent.mkdir(parents=True,exist_ok=True); p.touch()\"")
        print(" - Start the GUI (it will attempt to start LocalWebsocketServer on port 8765); re-run this script afterwards.")
        return

if __name__ == "__main__":
    main()
