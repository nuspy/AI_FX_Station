#!/usr/bin/env python3
"""
tests/manual_tests/tiingo_ws_test.py

Tiingo WebSocket tester.

Reads .env for:
  TIINGO_WS_URI (default wss://api.tiingo.com/fx)
  TIINGO_APIKEY

Sends a subscribe payload with 'authorization' and 'eventData.thresholdLevel' (string) and prints incoming messages.

Usage:
  pip install websocket-client python-dotenv
  # create .env in project root:
  TIINGO_APIKEY=d867b4314010495a5fa40593610eb3deae5e2dcd

  python tests/manual_tests/tiingo_ws_test.py --duration 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
import os

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing dependency 'python-dotenv'. Install with: pip install python-dotenv")
    sys.exit(1)

# try to use simplejson for nicer dumps if present
try:
    import simplejson as sj
    _json_dumps = lambda o: sj.dumps(o, indent=2)
except Exception:
    _json_dumps = lambda o: json.dumps(o, indent=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uri", help="WebSocket URI (overrides .env)")
    p.add_argument("--apikey", help="Tiingo API key (overrides .env)")
    p.add_argument("--threshold", help="thresholdLevel string (1-7)", default="5")
    p.add_argument("--tickers", help="comma-separated tickers string (e.g. 'audusd' or 'eurusd')", default="audusd")
    p.add_argument("--duration", type=float, help="Seconds to run (optional)")
    args = p.parse_args()

    # Load .env file from the project root
    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded environment variables from {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}")

    uri = args.uri or os.environ.get("TIINGO_WS_URI") or "wss://api.tiingo.com/fx"
    api_key = args.apikey or os.environ.get("TIINGO_APIKEY") or os.environ.get("TIINGO_API_KEY")

    if not api_key:
        print("Warning: TIINGO API key not set (pass --apikey or put TIINGO_APIKEY in .env). Attempting connect anyway.")

    try:
        from websocket import create_connection, WebSocketException  # websocket-client
    except Exception:
        print("Missing dependency 'websocket-client'. Install with: pip install websocket-client")
        raise SystemExit(1)

    threshold_str = str(args.threshold)
    tickers_arg = str(args.tickers).strip()

    try:
        if tickers_arg.startswith("[") and tickers_arg.endswith("]"):
            tickers_value = json.loads(tickers_arg)
        elif "," in tickers_arg:
            tickers_value = [t.strip() for t in tickers_arg.split(",") if t.strip()]
        else:
            tickers_value = [tickers_arg]
    except Exception:
        tickers_value = [tickers_arg]

    try:
        tickers_value = [str(t).lower() for t in tickers_value]
    except Exception:
        pass

    sub_payload = {
        "eventName": "subscribe",
        "authorization": api_key or "",
        "eventData": {
            "thresholdLevel": threshold_str,
            "tickers": tickers_value
        }
    }

    print("Connecting to", uri)
    try:
        ws = create_connection(uri, timeout=10)
    except Exception as e:
        print("Connection failed:", e)
        raise SystemExit(1)

    try:
        try:
            ws.send(_json_dumps(sub_payload))
            print("Sent subscribe payload:", sub_payload)
        except Exception as e:
            print("Failed to send subscribe payload:", e)

        start = time.time()
        while True:
            if args.duration and (time.time() - start) > args.duration:
                print("Duration elapsed, closing.")
                break
            try:
                msg = ws.recv()
                if not msg:
                    continue
                try:
                    obj = json.loads(msg)
                    try:
                        mtype = obj.get("messageType")
                        if mtype == "A" and isinstance(obj.get("data"), list):
                            data = obj["data"]
                            tag = data[0] if len(data) > 0 else ""
                            pair = data[1] if len(data) > 1 else ""
                            iso_ts = data[2] if len(data) > 2 else ""
                            bid = data[4] if len(data) > 4 else None
                            ask = data[5] if len(data) > 5 else None
                            print(f"QUOTE {pair} time={iso_ts} bid={bid} ask={ask}")
                        else:
                            print("MSG:", _json_dumps(obj))
                    except Exception:
                        print("MSG:", _json_dumps(obj))
                except Exception:
                    print("MSG:", msg)
            except WebSocketException as e:
                print("WebSocket error recv:", e)
                break
            except Exception as e:
                print("Recv exception:", e)
                break
    finally:
        try:
            ws.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
