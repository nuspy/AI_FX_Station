# src/forex_diffusion/services/tiingo_ws_connector.py
from __future__ import annotations

import threading
import time
import json
import os
from typing import Optional, Iterable
from loguru import logger

# try websocket-client
try:
    from websocket import create_connection, WebSocketException
    _HAS_WS_CLIENT = True
except Exception:
    create_connection = None
    WebSocketException = Exception
    _HAS_WS_CLIENT = False

class TiingoWSConnector:
    """
    Background connector to Tiingo WebSocket. Runs in a daemon thread.
    Publishes 'tick' payloads via event_bus.publish(payload) when quotes arrive.
    """
    def __init__(self, uri: str = "wss://api.tiingo.com/fx", api_key: Optional[str] = None, tickers: Optional[Iterable[str]] = None, threshold: str = "5", db_engine=None):
        self.uri = uri
        self.api_key = api_key or os.environ.get("TIINGO_APIKEY") or os.environ.get("TIINGO_API_KEY")
        self.tickers = [str(t).lower() for t in (list(tickers) if tickers else ["eurusd"])]
        self.threshold = str(threshold)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._db_engine = db_engine

    def start(self):
        if not _HAS_WS_CLIENT:
            logger.warning("TiingoWSConnector not started: missing 'websocket-client' package")
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.debug("TiingoWSConnector thread launched")
        return True

    def stop(self, timeout: float = 1.0):
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=timeout)
        except Exception:
            pass

    def _run(self):
        try:
            from ..utils.event_bus import publish
        except Exception:
            publish = lambda *a, **k: None

        while not self._stop.is_set():
            try:
                logger.info("TiingoWSConnector connecting to %s", self.uri)
                try:
                    ws = create_connection(self.uri, timeout=10)
                except Exception as e:
                    logger.warning("TiingoWSConnector connection failed: %s", e)
                    time.sleep(3.0)
                    continue

                try:
                    # build subscribe payload
                    sub = {
                        "eventName": "subscribe",
                        "authorization": self.api_key or "",
                        "eventData": {
                            "thresholdLevel": self.threshold,
                            "tickers": self.tickers
                        }
                    }
                    try:
                        ws.send(json.dumps(sub))
                        logger.info("TiingoWSConnector subscribe sent: tickers=%s threshold=%s", self.tickers, self.threshold)
                    except Exception as e:
                        logger.warning("TiingoWSConnector failed to send subscribe: %s", e)

                    # receive loop
                    while not self._stop.is_set():
                        try:
                            raw = ws.recv()
                            if not raw:
                                continue
                            # debug-log raw message
                            try:
                                logger.debug(f"TiingoWSConnector raw msg: {raw}")
                            except Exception:
                                pass
                            try:
                                msg = json.loads(raw)
                            except Exception:
                                continue
                            # handle messages: look for quote arrays ("messageType": "A")
                            try:
                                mtype = msg.get("messageType")
                                if mtype == "A":
                                    data = msg.get("data", [])
                                    # expected layout: ["Q", "pair", "iso_ts", size, bid, ask, ...]
                                    if isinstance(data, list) and len(data) >= 6:
                                        pair = data[1]
                                        iso_ts = data[2]
                                        bid = data[4]
                                        ask = data[5]
                                        # convert iso_ts to ms
                                        ts_ms = None
                                        try:
                                            import pandas as _pd
                                            ts_ms = int(_pd.to_datetime(iso_ts).value // 1_000_000)
                                        except Exception:
                                            ts_ms = int(time.time() * 1000)
                                        price = bid if bid is not None else ask

                                        # Normalize pair format 'eurusd' -> 'EUR/USD' to match ChartTab/DB expectations
                                        try:
                                            if isinstance(pair, str):
                                                p = pair.strip()
                                                if "/" in p:
                                                    # already contains slash
                                                    norm_symbol = p.upper()
                                                elif len(p) == 6 and p.isalpha():
                                                    norm_symbol = f"{p[0:3].upper()}/{p[3:6].upper()}"
                                                else:
                                                    # fallback: uppercase
                                                    norm_symbol = p.upper()
                                            else:
                                                norm_symbol = pair
                                        except Exception:
                                            norm_symbol = (pair.upper() if isinstance(pair, str) else pair)

                                        payload = {"symbol": norm_symbol, "timeframe": "1m", "ts_utc": int(ts_ms), "price": float(price) if price is not None else None, "bid": bid, "ask": ask}
                                        try:
                                            publish("tick", payload)
                                            logger.info(f"TiingoWSConnector published tick: {payload}")
                                        except Exception:
                                            pass

                                        # optional DB upsert if engine provided, using normalized symbol
                                        if self._db_engine is not None:
                                            try:
                                                import pandas as _pd
                                                from ..data import io as data_io
                                                df = _pd.DataFrame([{"ts_utc": int(ts_ms), "open": float(price), "high": float(price), "low": float(price), "close": float(price), "volume": None}])
                                                data_io.upsert_candles(self._db_engine, df, norm_symbol, "1m", resampled=False)
                                            except Exception:
                                                pass

                                        # Diagnostic: check event_bus status and call subscribers directly as fallback
                                        try:
                                            from ..utils.event_bus import debug_status, get_subscribers
                                            st = debug_status()
                                            logger.info(f"TiingoWSConnector event_bus status after publish: {st}")
                                            subs = get_subscribers("tick")
                                            if subs:
                                                # deduplicate by identity to avoid multiple calls to same callable
                                                uniq = []
                                                seen = set()
                                                for cb in subs:
                                                    iid = id(cb)
                                                    if iid in seen:
                                                        continue
                                                    seen.add(iid)
                                                    uniq.append(cb)
                                                if len(uniq) != len(subs):
                                                    logger.info(f"TiingoWSConnector: removed {len(subs)-len(uniq)} duplicate subscribers before invoking")
                                                logger.info(f"TiingoWSConnector: invoking {len(uniq)} unique subscribers directly as fallback")
                                                for cb in uniq:
                                                    try:
                                                        # call subscriber in current thread; many subscribers (e.g. bridge._on_event) will enqueue to UI
                                                        cb(payload)
                                                    except Exception as e:
                                                        logger.debug("TiingoWSConnector: subscriber direct call failed: {}", e)
                                            else:
                                                logger.debug("TiingoWSConnector: no subscribers found in registry")
                                        except Exception as e:
                                            logger.debug("TiingoWSConnector: debug/fallback subscriber invocation failed: {}", e)
                                        # optional DB upsert if engine provided
                                        if self._db_engine is not None:
                                            try:
                                                import pandas as _pd
                                                from ..data import io as data_io
                                                df = _pd.DataFrame([{"ts_utc": int(ts_ms), "open": float(price), "high": float(price), "low": float(price), "close": float(price), "volume": None}])
                                                data_io.upsert_candles(self._db_engine, df, payload["symbol"], "1m", resampled=False)
                                            except Exception:
                                                pass
                                else:
                                    # log heartbeat/info at debug
                                    if msg.get("messageType") in ("I", "H"):
                                        logger.debug("TiingoWSConnector info/heartbeat: %s", msg.get("response", msg))
                            except Exception:
                                pass
                        except WebSocketException:
                            break
                        except Exception as e:
                            logger.debug("TiingoWSConnector recv exception: %s", e)
                            break
                finally:
                    try:
                        ws.close()
                    except Exception:
                        pass
            except Exception as e:
                logger.exception("TiingoWSConnector worker exception: {}", e)
                time.sleep(3.0)
        logger.info("TiingoWSConnector stopped")
