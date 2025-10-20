from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np
import time
from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QTableWidgetItem, QListWidgetItem

from .base import ChartServiceBase


class DataService(ChartServiceBase):
    """Auto-generated service extracted from ChartTab."""
    
    def _persist_realtime_candle(self, symbol: str, timeframe: str, ts_utc: int, price: float, bid: float = None, ask: float = None):
        """
        Persist real-time candle to database (UPSERT pattern).
        
        Critical for data continuity - without this, real-time data is lost on restart
        and creates gaps that backfill cannot fill.
        """
        if timeframe == 'tick':
            # Don't persist individual ticks (too many)
            return
        
        try:
            controller = getattr(self._main_window, "controller", None)
            eng = getattr(getattr(controller, "market_service", None), "engine", None) if controller else None
            if eng is None:
                return
            
            # Get OHLC from last candle in buffer
            if self._last_df is not None and not self._last_df.empty:
                last_row = self._last_df.iloc[-1]
                if int(last_row["ts_utc"]) == int(ts_utc):
                    # UPSERT: insert or update the candle
                    from sqlalchemy import text
                    with eng.begin() as conn:
                        # Try insert first, on conflict update
                        query = text("""
                            INSERT INTO market_data_candles (symbol, timeframe, ts_utc, open, high, low, close, volume)
                            VALUES (:symbol, :timeframe, :ts_utc, :open, :high, :low, :close, :volume)
                            ON CONFLICT(symbol, timeframe, ts_utc) 
                            DO UPDATE SET high=:high, low=:low, close=:close, volume=:volume
                        """)
                        conn.execute(query, {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "ts_utc": int(ts_utc),
                            "open": float(last_row["open"]),
                            "high": float(last_row["high"]),
                            "low": float(last_row["low"]),
                            "close": float(last_row["close"]),
                            "volume": float(last_row.get("volume", 0))
                        })
                        # logger.debug(f"Persisted {timeframe} candle for {symbol} at {ts_utc}")
        except Exception as e:
            logger.error(f"Failed to persist candle: {e}")
    
    def _load_1m_fallback_for_ticks(self, symbol: str):
        """Load 1m candles when timeframe='tick' but no ticks available.
        
        This runs ONCE on startup if buffer is empty. It fills the chart with
        historical 1m candles. Real-time ticks will append/update on top.
        """
        try:
            logger.info(f"Loading 1m fallback data for tick chart: {symbol}")
            db_service = getattr(self.view, 'db_service', None)
            if not db_service:
                logger.warning("No db_service available for 1m fallback")
                return
            
            # Query last 3000 1m candles
            query = f"""
                SELECT ts_utc, open, high, low, close, volume
                FROM candles
                WHERE symbol = '{symbol}' AND timeframe = '1m'
                ORDER BY ts_utc DESC
                LIMIT 3000
            """
            
            df = pd.read_sql(query, db_service.engine)
            if not df.empty:
                df = df.sort_values('ts_utc').reset_index(drop=True)
                self._last_df = df
                
                # Mark as dirty to trigger redraw
                self._rt_dirty = True
                
                logger.info(f"Loaded {len(df)} 1m candles as fallback for tick chart")
            else:
                logger.warning(f"No 1m candles found for {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to load 1m fallback: {e}", exc_info=True)

    def _handle_tick(self, payload: dict):
        """Thread-safe entrypoint: enqueue tick to GUI thread."""
        try:
            self.tickArrived.emit(payload)
        except Exception as e:
            logger.exception("Failed to emit tick: {}", e)

    def _on_tick_main(self, payload: dict):
        """GUI-thread handler: aggiorna Market Watch e buffer, delega il redraw al throttler."""
        try:
            if not isinstance(payload, dict):
                return
            sym = payload.get("symbol") or getattr(self, "symbol", None)
            
            # Fallback: if timeframe='tick' and no tick data in buffer, load 1m candles
            current_tf = getattr(self, 'timeframe', '1m')
            if current_tf == 'tick' and (self._last_df is None or self._last_df.empty):
                self._load_1m_fallback_for_ticks(sym)

            # Update active provider label (get from settings)
            # DISABLED: causes widget deletion errors
            # self._update_provider_label()

            # Market watch update
            try:
                if sym:
                    bid_val = payload.get('bid')
                    ask_val = payload.get('ask', payload.get('offer'))
                    self._update_market_quote(sym, bid_val, ask_val, payload.get('ts_utc'))
                    
                    # Update order books widget if available (from parent ChartTab)
                    if hasattr(self.view, 'order_books_widget') and self.view.order_books_widget:
                        # Get actual DOM data from dom_service (passed to ChartTab)
                        dom_service = getattr(self.view, 'dom_service', None)
                        if dom_service:
                            dom_snapshot = dom_service.get_latest_dom_snapshot(sym)
                            if dom_snapshot:
                                bids = dom_snapshot.get('bids', [])
                                asks = dom_snapshot.get('asks', [])
                                self.view.order_books_widget.update_dom(bids, asks)
                                
                                # Update order flow bar
                                if hasattr(self.view, 'order_flow_bar') and bids and asks:
                                    total_bid = sum(b.get('size', 0) for b in bids)
                                    total_ask = sum(a.get('size', 0) for a in asks)
                                    if total_bid + total_ask > 0:
                                        imbalance = int(((total_bid - total_ask) / (total_bid + total_ask)) * 100)
                                        self.view.order_flow_bar.setValue(imbalance)
                                        self.view.order_flow_label.setText(f"Bid: {total_bid:.2f} | Ask: {total_ask:.2f}")
            except Exception as e:
                logger.error(f"Failed to update market watch: {e}")

            # Update chart buffer only for current symbol (O(1) append)
            current_symbol = getattr(self, "symbol", None)
            if sym and sym == current_symbol:
                try:
                    # Get timestamp (try ts_utc, then timestamp, then now)
                    ts = payload.get("ts_utc") or payload.get("timestamp") or int(time.time() * 1000)
                    if not isinstance(ts, int):
                        ts = int(ts)
                    
                    # Get price (try close, price, or mid of bid/ask)
                    y = payload.get("close") or payload.get("price")
                    bid_val = payload.get('bid')
                    ask_val = payload.get('ask', payload.get('offer'))
                    
                    # Calculate mid price if y not available
                    if y is None and bid_val is not None and ask_val is not None:
                        y = (bid_val + ask_val) / 2
                    # Get current timeframe for aggregation
                    current_tf = getattr(self, 'timeframe', '1m')
                    
                    # For tick timeframe, use actual tick timestamp (no aggregation)
                    if current_tf == 'tick':
                        candle_ts = ts
                        candle_interval_ms = 0  # No aggregation for ticks
                    else:
                        # Calculate candle interval in milliseconds
                        tf_map = {
                            '1m': 60_000, '5m': 300_000, '15m': 900_000, '30m': 1_800_000,
                            '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000, '1w': 604_800_000
                        }
                        candle_interval_ms = tf_map.get(current_tf, 60_000)  # Default 1m
                        
                        # Normalize timestamp to candle boundary
                        candle_ts = (ts // candle_interval_ms) * candle_interval_ms
                    
                    # se buffer vuoto: crea
                    if self._last_df is None or self._last_df.empty:
                        # Create with OHLC columns for candle charts
                        row = {"ts_utc": candle_ts, "open": y, "high": y, "low": y, "close": y, "volume": 0, "bid": bid_val, "ask": ask_val}
                        self._last_df = pd.DataFrame([row])
                    else:
                        last_candle_ts = int(self._last_df["ts_utc"].iloc[-1])
                        
                        # For tick timeframe, always append (no aggregation)
                        if current_tf == 'tick':
                            new_row = {"ts_utc": candle_ts, "open": y, "high": y, "low": y, "close": y, "volume": 0}
                            if bid_val is not None:
                                new_row["bid"] = bid_val
                            if ask_val is not None:
                                new_row["ask"] = ask_val
                            self._last_df.loc[len(self._last_df)] = new_row
                            # trim buffer to last 5000 ticks
                            if len(self._last_df) > 5000:
                                self._last_df = self._last_df.iloc[-5000:].reset_index(drop=True)
                        elif candle_ts > last_candle_ts:
                            # New candle - append new row
                            new_row = {"ts_utc": candle_ts, "open": y, "high": y, "low": y, "close": y, "volume": 0}
                            if bid_val is not None:
                                new_row["bid"] = bid_val
                            if ask_val is not None:
                                new_row["ask"] = ask_val
                            self._last_df.loc[len(self._last_df)] = new_row
                            # trim buffer
                            if len(self._last_df) > 10000:
                                self._last_df = self._last_df.iloc[-10000:].reset_index(drop=True)
                        else:
                            # Same candle - update OHLC
                            idx = len(self._last_df) - 1
                            self._last_df.at[idx, "high"] = max(self._last_df.at[idx, "high"], y)
                            self._last_df.at[idx, "low"] = min(self._last_df.at[idx, "low"], y)
                            self._last_df.at[idx, "close"] = y
                            if bid_val is not None:
                                self._last_df.at[len(self._last_df)-1, 'bid'] = bid_val
                            if ask_val is not None:
                                self._last_df.at[len(self._last_df)-1, 'ask'] = ask_val
                        # se ts < last_ts, ignora (fuori ordine)
                    
                    # mark dirty for throttled redraw
                    self._rt_dirty = True
                    
                    # CRITICAL: Persist real-time candles to DB
                    # Save the last candle to database (upsert pattern)
                    try:
                        self._persist_realtime_candle(sym, current_tf, candle_ts, y, bid_val, ask_val)
                    except Exception as persist_err:
                        logger.error(f"Failed to persist real-time candle: {persist_err}")

                except Exception as e:
                    logger.error(f"Failed to update chart buffer: {e}")

                # update bid/ask label
                try:
                    if payload.get('bid') is not None and payload.get('ask') is not None:
                        self.bidask_label.setText(f"Bid: {float(payload['bid']):.5f}    Ask: {float(payload['ask']):.5f}")
                except Exception:
                    try:
                        self.bidask_label.setText(f"Bid: {payload.get('bid')}    Ask: {payload.get('ask')}")
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.exception("Failed to handle tick on GUI: {}", e)

    def _rt_flush(self):
        """Throttled redraw preserving zoom/pan."""
        try:
            if not getattr(self, "_rt_dirty", False):
                return
            self._rt_dirty = False

            # preserve current view
            try:
                # PyQtGraph uses viewRange() instead of get_xlim/get_ylim
                if hasattr(self.ax, 'viewRange'):
                    view_range = self.ax.viewRange()
                    prev_xlim = view_range[0]
                    prev_ylim = view_range[1]
                else:
                    prev_xlim = self.ax.get_xlim()
                    prev_ylim = self.ax.get_ylim()
            except Exception:
                prev_xlim = prev_ylim = None
            # redraw base chart (quantiles overlay mantenuti da _forecasts)
            if self._last_df is not None and not self._last_df.empty:

                self.update_plot(self._last_df, restore_xlim=prev_xlim, restore_ylim=prev_ylim)
            else:
                logger.warning(f"⚠️ Cannot update plot: _last_df is {'None' if self._last_df is None else 'empty'}")
                # Trigger follow mode if enabled and timeout expired
                try:
                    if hasattr(self, '_follow_center_if_needed'):
                        self._follow_center_if_needed()
                except Exception:
                    pass
                    
        except Exception as e:
            logger.exception("Realtime flush failed: {}", e)

    def _schedule_view_reload(self):
        """Throttle view-window reload after user interaction."""
        try:
            self._reload_timer.stop()
            self._reload_timer.start()
        except Exception:
            pass

    def _resolution_for_span(self, ms_span: int) -> str:
        """Pick best timeframe by visible span (ms)."""
        try:
            mins = max(1, int(ms_span / 60000))
            # mapping: <=30m -> 1m; <=5h -> 5m; <=24h -> 15m; <=7d -> 1h; >7d -> 4h
            if mins <= 30:
                return "1m"   # ticks storici non disponibili in DB: usiamo 1m
            if mins <= 5 * 60:
                return "5m"
            if mins <= 24 * 60:
                return "15m"
            if mins <= 7 * 24 * 60:
                return "1h"
            return "4h"
        except Exception:
            return "15m"

    def _reload_view_window(self):
        """Reload only data covering [view_left .. view_right] plus one span of history."""
        # TODO: Reimplement for PyQtGraph without matplotlib dependencies
        # For now, disable dynamic reloading
        return
        try:
            # get current view in data coordinates
            if hasattr(self.ax, 'viewRange'):
                # PyQtGraph
                xlim = self.ax.viewRange()[0]
            else:
                # matplotlib (fallback)
                xlim = self.ax.get_xlim()

            if not xlim or xlim[0] >= xlim[1]:
                return
            # ensure UTC ms
            from datetime import timezone
            if left_dt.tzinfo is None:
                left_dt = left_dt.replace(tzinfo=timezone.utc)
            else:
                left_dt = left_dt.astimezone(timezone.utc)
            if right_dt.tzinfo is None:
                right_dt = right_dt.replace(tzinfo=timezone.utc)
            else:
                right_dt = right_dt.astimezone(timezone.utc)
            left_ms = int(left_dt.timestamp() * 1000)
            right_ms = int(right_dt.timestamp() * 1000)
            span = max(1, right_ms - left_ms)
            start_ms = max(0, left_ms - span)  # una finestra “cache” a sinistra
            end_ms = right_ms

            # pick timeframe: combo or auto
            tf_sel = getattr(self, "tf_combo", None).currentText() if hasattr(self, "tf_combo") else "auto"
            tf_req = self._resolution_for_span(span) if (not tf_sel or tf_sel == "auto") else tf_sel
            sym = getattr(self, "symbol", None)
            if not sym or not tf_req:
                return

            # se cache già copre bene, skip
            if self._current_cache_tf == tf_req and self._current_cache_range:
                c0, c1 = self._current_cache_range
                if start_ms >= c0 and end_ms <= c1:
                    return  # nulla da fare

            # OPTIMIZED: Load only viewport + 2x buffer on each side (dynamic loading)
            # This prevents loading 50k candles when only ~500 are needed
            # Buffer ensures smooth panning without reloading
            viewport_size_ms = end_ms - start_ms
            buffer_ms = viewport_size_ms * 2  # 2x buffer on each side
            
            buffered_start = max(0, start_ms - buffer_ms)
            buffered_end = end_ms + buffer_ms
            
            # Calculate reasonable limit based on timeframe
            # Viewport + 2x buffer before + 2x buffer after = 5x viewport
            tf_to_ms = {
                "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
                "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000,
            }
            candle_duration = tf_to_ms.get(tf_req, 60_000)
            estimated_candles = int(viewport_size_ms / candle_duration) * 5
            dynamic_limit = max(500, min(10000, int(estimated_candles)))
            
            logger.debug(f"Dynamic loading: viewport={start_ms}-{end_ms}, buffered={buffered_start}-{buffered_end}, limit={dynamic_limit}")
            
            # carica da DB solo la finestra necessaria con buffer
            df = self._load_candles_from_db(sym, tf_req, limit=dynamic_limit, start_ms=buffered_start, end_ms=buffered_end)
            if df is None or df.empty:
                return

            # aggiorna cache state
            self._current_cache_tf = tf_req
            self._current_cache_range = (int(df["ts_utc"].iat[0]), int(df["ts_utc"].iat[-1]))

            # ridisegna preservando i limiti attuali
            try:
                prev_xlim = self.ax.get_xlim()
                prev_ylim = self.ax.get_ylim()
            except Exception:
                prev_xlim = prev_ylim = None
            self.update_plot(df, restore_xlim=prev_xlim, restore_ylim=prev_ylim)
            try:
                self.tf_used_label.setText(f"TF used: {tf_req}")
            except Exception:
                pass
        except Exception as e:
            logger.exception("View-window reload failed: {}", e)

    def _tf_to_timedelta(self, tf: str):
        """Convert timeframe string like '1m','5m','1h' into pandas Timedelta."""
        try:
            tf = str(tf).strip().lower()
            if tf.endswith("m"):
                return pd.to_timedelta(int(tf[:-1]), unit="m")
            if tf.endswith("h"):
                return pd.to_timedelta(int(tf[:-1]), unit="h")
            if tf.endswith("d"):
                return pd.to_timedelta(int(tf[:-1]), unit="d")
            # default 1 minute
            return pd.to_timedelta(1, unit="m")
        except Exception:
            return pd.to_timedelta(1, unit="m")

    # ---- compressed X helpers ----
    def _expand_compressed_x(self, x_c: float) -> float:
        """Map compressed x (weekend removed) back to real matplotlib date float."""
        try:
            segs = getattr(self, "_x_segments_compressed", None)
            if not segs:
                return float(x_c)
            for s_c, e_c, closed_before in segs:
                if float(s_c) <= float(x_c) <= float(e_c):
                    return float(x_c) + float(closed_before)
            if float(x_c) < float(segs[0][0]):
                return float(x_c) + float(segs[0][2])
            return float(x_c) + float(segs[-1][2])
        except Exception:
            return float(x_c)

    # ---- backfill 1m completo (saltando weekend) ----
    def _compute_missing_1m_gaps(self, start_ms: int, end_ms: int) -> list[tuple[int, int]]:
        """Return list of (gap_start_ms, gap_end_ms) where 1m candles are missing between start..end, excluding weekend minutes."""
        try:
            sym = getattr(self, "symbol", None)
            if not sym or start_ms is None or end_ms is None or start_ms >= end_ms:
                return []
            # load actual 1m candles in range
            df1 = self._load_candles_from_db(sym, "1m", limit=10000, start_ms=int(start_ms), end_ms=int(end_ms))
            actual = np.array([], dtype=np.int64)
            if df1 is not None and not df1.empty and "ts_utc" in df1.columns:
                actual = df1["ts_utc"].astype("int64").dropna().values
            # expected timestamps for full range (no weekend split - provider data already excludes weekends)
            import pandas as _pd
            sdt = _pd.to_datetime(int(start_ms), unit="ms", utc=True)
            edt = _pd.to_datetime(int(end_ms), unit="ms", utc=True)
            # 1-min grid; include end boundary
            idx = _pd.date_range(start=sdt, end=edt, freq="1min", tz="UTC")
            expected = (idx.astype("int64") // 10**6).to_numpy(dtype=np.int64) if len(idx) else np.array([], dtype=np.int64)
            if expected.size == 0:
                return []
            # missing := expected \ actual
            if actual.size:
                missing = np.setdiff1d(expected, actual, assume_unique=False)
            else:
                missing = expected
            if missing.size == 0:
                return []
            # group contiguous minutes into ranges
            diffs = np.diff(missing)
            splits = np.where(diffs != 60000)[0] + 1
            groups = np.split(missing, splits)
            gaps: list[tuple[int, int]] = []
            for g in groups:
                if g.size == 0:
                    continue
                gaps.append((int(g[0]), int(g[-1])))
            return gaps
        except Exception:
            return []

    def _backfill_all_missing_1m(self) -> None:
        """Scan all 1m gaps (excluding weekend) and backfill them sequentially up to now via REST."""
        try:
            sym = getattr(self, "symbol", None)
            if not sym:
                return
            # pick scan range: from earliest loaded candle if available, else last 30 days
            if getattr(self, "_last_df", None) is not None and not self._last_df.empty:
                start_ms = int(self._last_df["ts_utc"].iloc[0])
            else:
                from datetime import datetime, timezone, timedelta
                start_ms = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)
            from datetime import datetime, timezone
            end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            gaps = self._compute_missing_1m_gaps(start_ms, end_ms)
            if not gaps:
                return

            controller = getattr(self._main_window, "controller", None)
            ms = getattr(controller, "market_service", None) if controller else None
            if ms is None:
                return

            from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

            class _Signals(QObject):
                finished = Signal(bool)

            class _Job(QRunnable):
                def __init__(self, svc, symbol, timeframe, start_ms, signals):
                    super().__init__()
                    self.svc = svc; self.symbol = symbol; self.timeframe = timeframe
                    self.start_ms = int(start_ms); self.signals = signals
                def run(self):
                    ok = True
                    try:
                        # backfill da earliest gap fino ad oggi
                        self.svc.backfill_symbol_timeframe(self.symbol, self.timeframe, force_full=False, progress_cb=None, start_ms_override=self.start_ms)
                    except Exception:
                        ok = False
                    finally:
                        try:
                            self.signals.finished.emit(ok)
                        except Exception:
                            pass

            # kick first job from earliest gap; upon finish, re-check and repeat if needed
            def _launch_next():
                cur_gaps = self._compute_missing_1m_gaps(start_ms, end_ms)
                if not cur_gaps:
                    # all good -> reload view to reflect new data
                    try:
                        self._schedule_view_reload()
                    except Exception:
                        pass
                    return
                sig = _Signals(self.view)
                sig.finished.connect(lambda _ok: _launch_next())
                QThreadPool.globalInstance().start(_Job(ms, sym, "1m", int(cur_gaps[0][0]), sig))

            _launch_next()
        except Exception:
            pass

    # ---- compressed X helpers & backfill-on-open ----
    def _expand_compressed_x(self, x_c: float) -> float:
        """Map compressed x (weekend removed) back to real matplotlib date float."""
        try:
            segs = getattr(self, "_x_segments_compressed", None)
            if not segs:
                return float(x_c)
            # segs: list of (s_comp, e_comp, closed_before_days)
            for s_c, e_c, closed_before in segs:
                if float(s_c) <= float(x_c) <= float(e_c):
                    return float(x_c) + float(closed_before)
            # outside known segments: best-effort
            if float(x_c) < float(segs[0][0]):
                return float(x_c) + float(segs[0][2])
            return float(x_c) + float(segs[-1][2])
        except Exception:
            return float(x_c)

    def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
        self.db_service = db_service
        self.symbol = symbol
        self.timeframe = timeframe
        # reset view-window cache on context change
        self._current_cache_tf = None
        self._current_cache_range = None
        # reset view-window cache on context change
        self._current_cache_tf = None
        self._current_cache_range = None
        # sync combo if present
        try:
            if hasattr(self, "symbol_combo") and symbol:
                idx = self.symbol_combo.findText(symbol)
                if idx >= 0:
                    self.symbol_combo.setCurrentIndex(idx)
        except Exception:
            pass
        # compute view range from UI (Years/Months) and load initial candles
        try:
            from datetime import datetime, timezone, timedelta
            yrs = int(self.years_combo.currentText() or "0") if hasattr(self, "years_combo") else 0
            mos = int(self.months_combo.currentText() or "0") if hasattr(self, "months_combo") else 0
            days = yrs * 365 + mos * 30
            start_ms_view = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000) if days > 0 else None
        except Exception:
            start_ms_view = None
        try:
            # Load reasonable amount of candles on startup
            limit_candles = 3000

            df = self._load_candles_from_db(self.symbol, self.timeframe, limit=limit_candles, start_ms=start_ms_view)
            if df is not None and not df.empty:
                self.update_plot(df)
                # backfill-on-open: fill data holes across the visible history (skip weekend closures)
                try:
                    self._backfill_on_open(df)
                except Exception:
                    pass
            else:
                logger.info("No candles found in DB for {} {}", self.symbol, self.timeframe)
        except Exception as e:
            logger.exception("Initial load failed: {}", e)

    def _on_symbol_changed(self, new_symbol: str):
        """Handle symbol change from combo: update context and reload candles from DB."""
        try:
            if not new_symbol:
                logger.warning("Symbol changed to empty string, ignoring")
                return

            logger.info(f"Symbol changed from {getattr(self, 'symbol', None)} to {new_symbol}")
            self.symbol = new_symbol

            # Update settings
            from ...utils.user_settings import set_setting
            set_setting('chart.symbol', new_symbol)

            # reset cache so next reload uses the new context
            self._current_cache_tf = None
            self._current_cache_range = None

            # reload last candles for this symbol/timeframe (respect view range)
            from datetime import datetime, timezone, timedelta
            try:
                yrs = int(self.years_combo.currentText() or "0")
                mos = int(self.months_combo.currentText() or "0")
                days = yrs * 365 + mos * 30
                start_ms_view = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000) if days > 0 else None
                logger.info(f"Loading candles for {new_symbol}: years={yrs}, months={mos}, days={days}, start_ms_view={start_ms_view}")
            except Exception as e:
                logger.warning(f"Failed to calculate date range: {e}")
                start_ms_view = None

            # Load candles in background to avoid UI freeze
            from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject
            
            class LoadSymbolSignals(QObject):
                finished = Signal(object, str, str)  # (df, symbol, timeframe)
                
            class LoadSymbolTask(QRunnable):
                def __init__(self, service, symbol, timeframe, start_ms, signals):
                    super().__init__()
                    self.service = service
                    self.symbol = symbol
                    self.timeframe = timeframe
                    self.start_ms = start_ms
                    self.signals = signals
                
                def run(self):
                    try:
                        logger.info(f"[BG] Loading {self.symbol} {self.timeframe} from DB...")
                        # Calculate end_ms for bounded query
                        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                        df = self.service._load_candles_from_db(
                            self.symbol, self.timeframe, 
                            limit=None, start_ms=self.start_ms, end_ms=end_ms
                        )
                        
                        # If no data, trigger auto-download
                        if df is None or df.empty:
                            logger.warning(f"[BG] No data for {self.symbol}, triggering auto-download...")
                            self.service._trigger_auto_download(self.symbol)
                            # Retry after download
                            df = self.service._load_candles_from_db(
                                self.symbol, self.timeframe,
                                limit=None, start_ms=self.start_ms, end_ms=end_ms
                            )
                        
                        if df is not None and not df.empty:
                            logger.info(f"[BG] Loaded {len(df)} candles for {self.symbol}")
                        
                        self.signals.finished.emit(df, self.symbol, self.timeframe)
                        
                    except Exception as e:
                        logger.error(f"[BG] Failed to load symbol data: {e}")
                        self.signals.finished.emit(None, self.symbol, self.timeframe)
            
            current_timeframe = getattr(self, "timeframe", "1m")
            signals = LoadSymbolSignals()
            signals.finished.connect(self._on_symbol_data_loaded)
            task = LoadSymbolTask(self, new_symbol, current_timeframe, start_ms_view, signals)
            QThreadPool.globalInstance().start(task)
        except Exception as e:
            logger.exception("Failed to switch symbol: {}", e)

    def _trigger_auto_download(self, symbol: str):
        """Trigger automatic download of candles when no data is available."""
        try:
            logger.info(f"Auto-downloading candles for {symbol}...")
            from datetime import datetime, timezone, timedelta

            # Download last 30 days of 1m candles
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=30)

            # Use existing download mechanism
            if hasattr(self, 'data_manager') and self.data_manager:
                self.data_manager.download_candles(
                    symbol=symbol,
                    timeframe='1m',
                    start_date=start_date,
                    end_date=end_date
                )
                logger.info(f"Auto-download completed for {symbol}")
            else:
                logger.warning("Data manager not available for auto-download")
        except Exception as e:
            logger.error(f"Auto-download failed for {symbol}: {e}")

    def _on_timeframe_changed(self, new_timeframe: str):
        """Handle timeframe change: reload candles from DB with new timeframe (async to avoid UI freeze)."""
        try:
            logger.info(f"DataService: Handling timeframe change to {new_timeframe}")
            if not new_timeframe:
                logger.warning("Empty timeframe provided")
                return
            
            self.timeframe = new_timeframe
            # reset cache so next reload uses the new context
            self._current_cache_tf = None
            self._current_cache_range = None
            
            # Calculate parameters
            from datetime import datetime, timezone, timedelta
            try:
                yrs = int(self.years_combo.currentText() or "0")
                mos = int(self.months_combo.currentText() or "0")
                days = yrs * 365 + mos * 30
                start_ms_view = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000) if days > 0 else None
            except Exception:
                start_ms_view = None
            
            symbol = getattr(self, "symbol", "EURUSD")
            
            # Smart cache: NO fixed limit, load based on date range + buffer
            BUFFER_CANDLES = 500
            
            # Calculate how many candles to show initially (viewport)
            tf_initial_display = {
                "tick": 1000, "1m": 500, "5m": 300, "15m": 200,
                "30m": 150, "1h": 168, "4h": 168, "1d": 90, "1w": 52
            }
            initial_display = tf_initial_display.get(new_timeframe, 500)
            
            # Calculate start_ms for initial display + buffer
            if start_ms_view is None:
                tf_ms = {
                    "tick": 1000, "1m": 60_000, "5m": 300_000, "15m": 900_000,
                    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
                    "1d": 86_400_000, "1w": 604_800_000
                }
                candle_duration_ms = tf_ms.get(new_timeframe, 60_000)
                lookback_ms = candle_duration_ms * (initial_display + BUFFER_CANDLES)
                start_ms_view = int((datetime.now(timezone.utc).timestamp() * 1000) - lookback_ms)
            
            end_ms_view = int(datetime.now(timezone.utc).timestamp() * 1000)
            
            # Load data in background thread to avoid UI freeze
            from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject
            
            class LoadSignals(QObject):
                finished = Signal(object, str)  # (df, timeframe)
            
            class LoadDataTask(QRunnable):
                def __init__(self, service, symbol, timeframe, start_ms, end_ms, signals):
                    super().__init__()
                    self.service = service
                    self.symbol = symbol
                    self.timeframe = timeframe
                    self.start_ms = start_ms
                    self.end_ms = end_ms
                    self.signals = signals
                
                def run(self):
                    try:
                        logger.info(f"[BG] Loading {self.symbol} {self.timeframe} from DB...")
                        df = self.service._load_candles_from_db(
                            self.symbol, self.timeframe, 
                            limit=None, start_ms=self.start_ms, end_ms=self.end_ms
                        )
                        
                        if df is not None and not df.empty:
                            logger.info(f"[BG] Loaded {len(df)} candles, emitting signal...")
                            self.signals.finished.emit(df, self.timeframe)
                        else:
                            logger.warning(f"[BG] No data found for {self.symbol} {self.timeframe}")
                            self.signals.finished.emit(None, self.timeframe)
                    except Exception as e:
                        logger.error(f"[BG] Failed to load data: {e}")
                        self.signals.finished.emit(None, self.timeframe)
            
            signals = LoadSignals()
            signals.finished.connect(self._on_data_loaded)
            task = LoadDataTask(self, symbol, new_timeframe, start_ms_view, end_ms_view, signals)
            QThreadPool.globalInstance().start(task)
            
        except Exception as e:
            logger.exception("Failed to switch timeframe: {}", e)
    
    def _on_data_loaded(self, df, timeframe):
        """Slot called on main thread when background data loading completes (timeframe change)."""
        try:
            if df is None or df.empty:
                logger.warning(f"No data loaded for {timeframe}")
                return
            
            logger.info(f"Updating plot with {len(df)} candles for {timeframe}")
            self.update_plot(df)
            logger.info("Plot updated successfully")
            
            # Start backfill in background
            try:
                self._backfill_on_open(df)
            except Exception as e:
                logger.error(f"Backfill failed: {e}")
        except Exception as e:
            logger.error(f"Failed to update plot: {e}")
    
    def _on_symbol_data_loaded(self, df, symbol, timeframe):
        """Slot called on main thread when background symbol data loading completes."""
        try:
            if df is None or df.empty:
                logger.error(f"No candles available for {symbol} {timeframe} after auto-download attempt")
                return
            
            logger.info(f"Updating plot with {len(df)} candles for {symbol}")
            self.update_plot(df)
            logger.info(f"Plot updated successfully for {symbol}")
            
            # Start backfill in background
            try:
                self._backfill_on_open(df)
            except Exception as e:
                logger.error(f"Backfill failed: {e}")
        except Exception as e:
            logger.error(f"Failed to update plot: {e}")

    # --- Backfill-on-open helpers ---
    def _period_includes_weekend(self, start_ms: int, end_ms: int) -> bool:
        """Return True if [start..end] overlaps Saturday/Sunday."""
        try:
            import pandas as _pd
            s = _pd.to_datetime(int(start_ms), unit="ms", utc=True).tz_convert(None)
            e = _pd.to_datetime(int(end_ms), unit="ms", utc=True).tz_convert(None)
            days = _pd.date_range(s.normalize(), e.normalize(), freq="D")
            return any(d.weekday() >= 5 for d in days)
        except Exception:
            return False

    def _find_earliest_data_gap(self, df: pd.DataFrame, timeframe: str) -> int | None:
        """Find earliest missing-data gap (weekday hours), return suggested start_ms for backfill."""
        try:
            if df is None or df.empty or "ts_utc" not in df.columns:
                return None
            ts = df["ts_utc"].astype("int64").to_numpy()
            if len(ts) < 2:
                return None
            # expected spacing from timeframe; fallback to median
            try:
                tf = str(timeframe or "auto").lower()
                if tf != "auto":
                    td = self._tf_to_timedelta(tf)
                    exp_ms = int(td.total_seconds() * 1000.0)
                else:
                    diffs = (ts[1:] - ts[:-1]).astype("int64")
                    exp_ms = int(max(60000, float(pd.Series(diffs).median())))
            except Exception:
                exp_ms = 60000
            for i in range(1, len(ts)):
                gap = int(ts[i] - ts[i-1])
                if gap > exp_ms * 2:
                    # skip if the gap overlaps weekend/market-close
                    if self._period_includes_weekend(ts[i-1], ts[i]):
                        continue
                    # earliest backfill start within the gap
                    return int(ts[i-1] + exp_ms)
            return None
        except Exception:
            return None

    def _backfill_on_open(self, df: pd.DataFrame) -> None:
        """Launch background backfill from earliest hole to present (skip weekend closures)."""
        try:
            start_override = self._find_earliest_data_gap(df, getattr(self, "timeframe", "auto"))
            if start_override is None:
                logger.debug(f"No data gaps found for backfill (symbol={getattr(self, 'symbol', 'N/A')}, tf={getattr(self, 'timeframe', 'N/A')})")
                return
            
            logger.info(f"Auto-backfill triggered: filling gap from {start_override} for {getattr(self, 'symbol', 'N/A')} {getattr(self, 'timeframe', 'N/A')}")
            controller = getattr(self._main_window, "controller", None)
            ms = getattr(controller, "market_service", None) if controller else None
            if ms is None:
                logger.warning(f"Auto-backfill skipped: MarketDataService not available (controller={controller})")
                return

            from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

            class _Signals(QObject):
                finished = Signal(bool)

            class _Job(QRunnable):
                def __init__(self, svc, symbol, timeframe, start_ms, signals):
                    super().__init__()
                    self.svc = svc; self.symbol = symbol; self.timeframe = timeframe
                    self.start_ms = start_ms; self.signals = signals
                def run(self):
                    ok = True
                    try:
                        self.svc.backfill_symbol_timeframe(self.symbol, self.timeframe, force_full=False, progress_cb=None, start_ms_override=int(self.start_ms))
                    except Exception:
                        ok = False
                    finally:
                        try:
                            self.signals.finished.emit(ok)
                        except Exception:
                            pass

            sig = _Signals(self.view)
            def _on_done(_ok: bool):
                # reload current window after backfill to fill the holes
                try:
                    logger.info(f"Backfill completed (ok={_ok}), reloading data from DB")
                    # Reload data from DB for current symbol/timeframe
                    current_symbol = getattr(self, "symbol", None)
                    current_tf = getattr(self, "timeframe", None)
                    if current_symbol and current_tf and _ok:
                        # Calculate smart range to avoid loading millions of candles
                        from datetime import datetime, timezone, timedelta
                        
                        # Get current view range or use default
                        try:
                            yrs = int(self.years_combo.currentText() or "0")
                            mos = int(self.months_combo.currentText() or "0")
                            days = yrs * 365 + mos * 30
                            start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000) if days > 0 else None
                        except Exception:
                            start_ms = None
                        
                        # If no range specified, use smart default based on timeframe
                        if start_ms is None:
                            tf_lookback_days = {
                                "tick": 1, "1m": 7, "5m": 14, "15m": 30,
                                "30m": 60, "1h": 90, "4h": 180, "1d": 365, "1w": 730
                            }
                            days_lookback = tf_lookback_days.get(current_tf, 30)
                            start_ms = int((datetime.now(timezone.utc) - timedelta(days=days_lookback)).timestamp() * 1000)
                        
                        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                        
                        # Reload with bounded range
                        df = self._load_candles_from_db(current_symbol, current_tf, limit=None, start_ms=start_ms, end_ms=end_ms)
                        if df is not None and not df.empty:
                            self.update_plot(df)
                            logger.info(f"Chart refreshed with {len(df)} candles after backfill (range: {days_lookback if start_ms else 'unlimited'} days)")
                except Exception as e:
                    logger.error(f"Failed to reload after backfill: {e}")
            sig.finished.connect(_on_done)
            QThreadPool.globalInstance().start(_Job(ms, getattr(self, "symbol", ""), getattr(self, "timeframe", ""), int(start_override), sig))
        except Exception:
            pass

    def _on_backfill_missing_clicked(self):
        """Trigger backfill for current symbol/timeframe asynchronously with determinate progress."""
        controller = getattr(self._main_window, "controller", None)
        ms = getattr(controller, "market_service", None) if controller else None
        if ms is None:
            QMessageBox.warning(self.view, "Backfill", "MarketDataService non disponibile.")
            return
        sym = getattr(self, "symbol", None)
        tf = getattr(self, "timeframe", None)
        if not sym or not tf:
            QMessageBox.information(self.view, "Backfill", "Imposta prima symbol e timeframe.")
            return

        # compute start override from UI years/months (if >0)
        try:
            yrs = int(self.years_combo.currentText() or "0")
            mos = int(self.months_combo.currentText() or "0")
        except Exception:
            yrs = 0; mos = 0
        start_override = None
        if yrs > 0 or mos > 0:
            from datetime import datetime, timezone, timedelta
            days = yrs * 365 + mos * 30
            start_dt = datetime.now(timezone.utc) - timedelta(days=days)
            start_override = int(start_dt.timestamp() * 1000)
            try:
                logger.info("Backfill requested via UI: Years={}, Months={} -> start={} (UTC)", yrs, mos, start_dt.isoformat())
            except Exception:
                pass
        else:
            try:
                logger.info("Backfill requested via UI: Years=0, Months=0 -> no override (service decides from last candle)")
            except Exception:
                pass

        from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

        class BackfillSignals(QObject):
            progress = Signal(int)
            finished = Signal(bool)

        class BackfillJob(QRunnable):
            def __init__(self, svc, symbol, timeframes, start_override, signals):
                super().__init__()
                self.svc = svc
                self.symbol = symbol
                self.timeframes = timeframes  # Now accepts list of all timeframes
                self.start_override = start_override
                self.signals = signals

            def run(self):
                ok = True
                try:
                    # enable REST backfill for this job scope
                    prev = getattr(self.svc, "rest_enabled", False)
                    try:
                        self.svc.rest_enabled = True
                    except Exception:
                        pass

                    # Backfill ALL timeframes, not just the displayed one
                    total_tfs = len(self.timeframes)
                    for i, tf in enumerate(self.timeframes):
                        def _cb(pct: int):
                            try:
                                # Calculate overall progress across all timeframes
                                overall_pct = int((i * 100 + pct) / total_tfs)
                                self.signals.progress.emit(overall_pct)
                            except Exception:
                                pass
                        self.svc.backfill_symbol_timeframe(self.symbol, tf, force_full=False, progress_cb=_cb, start_ms_override=self.start_override)
                except Exception:
                    ok = False
                finally:
                    # restore REST flag and emit completion
                    try:
                        self.svc.rest_enabled = prev
                    except Exception:
                        pass
                    try:
                        self.signals.finished.emit(ok)
                    except Exception:
                        pass

        self.setDisabled(True)
        self.backfill_progress.setRange(0, 100)
        self.backfill_progress.setValue(0)
        self.backfill_progress.setVisible(True)  # Show progress bar when starting

        self._bf_signals = BackfillSignals(self.view)
        self._bf_signals.progress.connect(self.backfill_progress.setValue)

        def _on_done(ok: bool):
            try:
                # reload last candles for the current view range
                from datetime import datetime, timezone, timedelta
                try:
                    yrs_v = int(self.years_combo.currentText() or "0")
                    mos_v = int(self.months_combo.currentText() or "0")
                    days_v = yrs_v * 365 + mos_v * 30
                    start_ms_view = int((datetime.now(timezone.utc) - timedelta(days=days_v)).timestamp() * 1000) if days_v > 0 else None
                except Exception:
                    start_ms_view = None
                df = self._load_candles_from_db(sym, tf, limit=3000, start_ms=start_ms_view)
                if df is not None and not df.empty:
                    self.update_plot(df)
                if ok:
                    QMessageBox.information(self.view, "Backfill", f"Backfill completato per {sym}.")
                else:
                    QMessageBox.warning(self.view, "Backfill", "Backfill fallito (vedi log).")
            finally:
                self.setDisabled(False)
                self.backfill_progress.setValue(100)
                self.backfill_progress.setVisible(False)  # Hide progress bar when done

        self._bf_signals.finished.connect(_on_done)
        # Backfill ALL timeframes, not just the currently displayed one
        all_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        job = BackfillJob(ms, sym, all_timeframes, start_override, self._bf_signals)
        QThreadPool.globalInstance().start(job)

    def _load_candles_from_db(self, symbol: str, timeframe: str, limit: int = 5000, start_ms: Optional[int] = None, end_ms: Optional[int] = None):
        """Load data from DB for symbol/timeframe, optionally constraining [start_ms, end_ms].
        - If timeframe == 'tick': read from market_data_ticks and map to (ts_utc, price).
        - Else: read candles from market_data_candles as before.
        """
        try:
            controller = getattr(self._main_window, "controller", None)
            eng = getattr(getattr(controller, "market_service", None), "engine", None) if controller else None
            if eng is None:
                logger.warning(f"Cannot load candles: engine is None (controller={controller})")
                return pd.DataFrame()

            logger.debug(f"Loading candles from DB: symbol={symbol}, tf={timeframe}, limit={limit}, start_ms={start_ms}, end_ms={end_ms}")
            from sqlalchemy import MetaData, select, and_
            meta = MetaData()
            if str(timeframe).lower() == "tick":
                # --- load ticks ---
                meta.reflect(bind=eng, only=["market_data_ticks"])
                tkt = meta.tables.get("market_data_ticks")
                if tkt is None:
                    return pd.DataFrame()
                with eng.connect() as conn:
                    conds = [tkt.c.symbol == symbol]
                    if start_ms is not None:
                        conds.append(tkt.c.ts_utc >= int(start_ms))
                    if end_ms is not None:
                        conds.append(tkt.c.ts_utc <= int(end_ms))
                    cond = and_(*conds)
                    stmt = select(tkt.c.ts_utc, tkt.c.price, getattr(tkt.c, "bid", None), getattr(tkt.c, "ask", None))\
                        .where(cond).order_by(tkt.c.ts_utc.desc()).limit(limit)
                    rows = conn.execute(stmt).fetchall()
                    if not rows:
                        # No tick data at all, fallback to 1m candles for entire range
                        logger.info(f"No tick data found for {symbol}, falling back to 1m candles")
                        return self._load_candles_from_db(symbol, "1m", limit=limit, start_ms=start_ms, end_ms=end_ms)
                    # rows may be tuples with (ts_utc, price, bid, ask) depending on schema
                    # Build price column: prefer 'price', else (bid+ask)/2
                    try:
                        # convert to DataFrame with robust column naming
                        cols = ["ts_utc", "price", "bid", "ask"]
                        df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
                    except Exception:
                        # fallback: coerce manually
                        recs = []
                        for r in rows:
                            try:
                                tsu = int(r[0])
                                pr = float(r[1]) if len(r) > 1 and r[1] is not None else None
                                bd = float(r[2]) if len(r) > 2 and r[2] is not None else None
                                ak = float(r[3]) if len(r) > 3 and r[3] is not None else None
                                if pr is None and (bd is not None and ak is not None):
                                    pr = (bd + ak) / 2.0
                                recs.append({"ts_utc": tsu, "price": pr, "bid": bd, "ask": ak})
                            except Exception:
                                continue
                        df = pd.DataFrame(recs)
                    # typing + clean + ASC order + dedup on ts_utc
                    try:
                        df["ts_utc"] = pd.to_numeric(df["ts_utc"], errors="coerce").astype("Int64")
                        if "price" in df.columns:
                            df["price"] = pd.to_numeric(df["price"], errors="coerce")
                        if "bid" in df.columns:
                            df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
                        if "ask" in df.columns:
                            df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
                        df = df.dropna(subset=["ts_utc", "price"]).reset_index(drop=True)
                        df["ts_utc"] = df["ts_utc"].astype("int64")
                        df = df.sort_values("ts_utc").reset_index(drop=True)
                        df = df.drop_duplicates(subset=["ts_utc"], keep="last").reset_index(drop=True)
                    except Exception:
                        pass
                    
                    # If tick data is completely empty, fallback to 1m candles
                    if df.empty:
                        logger.info(f"No tick data found for {symbol}, falling back to 1m candles")
                        return self._load_candles_from_db(symbol, "1m", limit=limit, start_ms=start_ms, end_ms=end_ms)
                    
                    # SMART FALLBACK: Fill gaps in tick data with 1m candles
                    # Detect gaps > 60 seconds (no ticks) and fill with 1m candles
                    if start_ms is not None and end_ms is not None and len(df) > 1:
                        df_sorted = df.sort_values("ts_utc").reset_index(drop=True)
                        gaps = []
                        
                        # Check for gap at start
                        if df_sorted["ts_utc"].iloc[0] - start_ms > 60_000:  # >1 min gap at start
                            gaps.append((start_ms, df_sorted["ts_utc"].iloc[0]))
                        
                        # Check gaps between ticks
                        for i in range(len(df_sorted) - 1):
                            gap_ms = df_sorted["ts_utc"].iloc[i+1] - df_sorted["ts_utc"].iloc[i]
                            if gap_ms > 60_000:  # Gap > 1 minute
                                gaps.append((df_sorted["ts_utc"].iloc[i], df_sorted["ts_utc"].iloc[i+1]))
                        
                        # Check for gap at end
                        if end_ms - df_sorted["ts_utc"].iloc[-1] > 60_000:  # >1 min gap at end
                            gaps.append((df_sorted["ts_utc"].iloc[-1], end_ms))
                        
                        # Fill gaps with 1m candles
                        if gaps:
                            logger.info(f"Found {len(gaps)} tick gaps for {symbol}, filling with 1m candles")
                            df_1m_parts = []
                            for gap_start, gap_end in gaps:
                                df_1m = self._load_candles_from_db(symbol, "1m", limit=None, start_ms=int(gap_start), end_ms=int(gap_end))
                                if not df_1m.empty:
                                    # Convert 1m candles to tick format (use close price)
                                    df_1m_ticks = pd.DataFrame({
                                        "ts_utc": df_1m["ts_utc"],
                                        "price": df_1m["close"]
                                    })
                                    df_1m_parts.append(df_1m_ticks)
                            
                            # Merge ticks and 1m data
                            if df_1m_parts:
                                df_combined = pd.concat([df[["ts_utc", "price"]]] + df_1m_parts, ignore_index=True)
                                df_combined = df_combined.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"]).reset_index(drop=True)
                                logger.info(f"Merged {len(df)} ticks + {sum(len(p) for p in df_1m_parts)} 1m candles = {len(df_combined)} total")
                                return df_combined
                    
                    return df[["ts_utc", "price"] + ([c for c in ["bid","ask"] if c in df.columns])]
            else:
                # --- load candles ---
                meta.reflect(bind=eng, only=["market_data_candles"])
                tbl = meta.tables.get("market_data_candles")
                if tbl is None:
                    return pd.DataFrame()
                with eng.connect() as conn:
                    conds = [tbl.c.symbol == symbol, tbl.c.timeframe == timeframe]
                    if start_ms is not None:
                        conds.append(tbl.c.ts_utc >= int(start_ms))
                    if end_ms is not None:
                        conds.append(tbl.c.ts_utc <= int(end_ms))
                    cond = and_(*conds)
                    # prendi le barre più recenti nel range e poi ordinale ASC per il plot
                    stmt = select(tbl.c.ts_utc, tbl.c.open, tbl.c.high, tbl.c.low, tbl.c.close, tbl.c.volume)\
                        .where(cond).order_by(tbl.c.ts_utc.desc()).limit(limit)
                    rows = conn.execute(stmt).fetchall()
                    if not rows:
                        logger.warning(f"No candles found in DB for {symbol} {timeframe} (start_ms={start_ms}, end_ms={end_ms}, limit={limit})")
                        return pd.DataFrame()

                    logger.debug(f"Loaded {len(rows)} candles from DB for {symbol} {timeframe}")
                    df = pd.DataFrame(rows, columns=["ts_utc","open","high","low","close","volume"])
                    # typing e ordinamento ASC
                    try:
                        df["ts_utc"] = pd.to_numeric(df["ts_utc"], errors="coerce").astype("Int64")
                        for c in ["open","high","low","close","volume"]:
                            if c in df.columns:
                                df[c] = pd.to_numeric(df[c], errors="coerce")
                        df = df.dropna(subset=["ts_utc"]).reset_index(drop=True)
                        df["ts_utc"] = df["ts_utc"].astype("int64")
                        df = df.sort_values("ts_utc").reset_index(drop=True)
                        # drop duplicates on timestamp to avoid multi-insert artifacts
                        before = len(df)
                        df = df.drop_duplicates(subset=["ts_utc"], keep="last").reset_index(drop=True)
                        trimmed = before - len(df)
                        if trimmed > 0:
                            try:
                                logger.info("Trimmed {} duplicate bars for {} {}", trimmed, symbol, timeframe)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    return df
        except Exception as e:
            logger.exception("Load candles failed: {}", e)
            return pd.DataFrame()

    def _update_market_quote(self, symbol: str, bid: float, ask: float, ts_ms: int):
        """
        Update market watch with bid/ask prices and spread color coding.

        TASK 1: Market Watch Quote Updates
        Implements spread tracking with color coding:
        - Green: Spread widening (compared to 10-tick history)
        - Red: Spread narrowing
        - Black: Stable (no significant change in 10+ ticks)
        """
        try:
            if not hasattr(self, 'market_watch') or self.market_watch is None:
                logger.warning("⚠️ Market watch widget not available")
                return

            if bid is None or ask is None:
                logger.warning(f"⚠️ Bid or ask is None for {symbol}")
                return

            # Calculate spread
            spread = ask - bid
            spread_pips = spread * 10000  # Assuming EUR/USD-like pair
            
            # Calculate mid price for price change detection
            mid_price = (bid + ask) / 2

            # Initialize price history and last change time if needed
            if not hasattr(self, '_price_history'):
                self._price_history = {}
            if not hasattr(self, '_last_price_change'):
                self._last_price_change = {}
            if not hasattr(self, '_last_color'):
                self._last_color = {}

            # Get or create price history for this symbol
            if symbol not in self._price_history:
                self._price_history[symbol] = mid_price
                self._last_price_change[symbol] = time.time()
                self._last_color[symbol] = "white"

            last_price = self._price_history[symbol]
            last_change_time = self._last_price_change[symbol]
            current_time = time.time()

            # Determine color based on price movement
            price_color = self._last_color.get(symbol, "white")
            
            if mid_price > last_price:
                # Price increased
                price_color = "green"
                self._last_price_change[symbol] = current_time
                self._price_history[symbol] = mid_price
                self._last_color[symbol] = "green"
            elif mid_price < last_price:
                # Price decreased
                price_color = "red"
                self._last_price_change[symbol] = current_time
                self._price_history[symbol] = mid_price
                self._last_color[symbol] = "red"
            else:
                # Price unchanged - check if >10 seconds since last change
                if current_time - last_change_time > 10:
                    price_color = "white"
                    self._last_color[symbol] = "white"
                # else: keep previous color

            # Update market watch list widget
            # Format: "SYMBOL | Bid: X.XXXXX | Ask: X.XXXXX | Spread: X.X pips"
            display_text = f"{symbol} | Bid: {bid:.5f} | Ask: {ask:.5f} | Spread: {spread_pips:.1f}"

            # Find existing item for this symbol or add new one
            found = False
            search_prefix = f"{symbol} |"
            for i in range(self.market_watch.count()):
                item = self.market_watch.item(i)
                item_text = item.text() if item else ""
                if item and item_text.startswith(search_prefix):
                    item.setText(display_text)
                    # Apply color based on price movement
                    if price_color == "green":
                        item.setForeground(Qt.green)
                    elif price_color == "red":
                        item.setForeground(Qt.red)
                    else:
                        item.setForeground(Qt.white)
                    found = True
                    break

            if not found:
                # Add new item
                item = QListWidgetItem(display_text)
                if price_color == "green":
                    item.setForeground(Qt.green)
                elif price_color == "red":
                    item.setForeground(Qt.red)
                else:
                    item.setForeground(Qt.white)
                self.market_watch.addItem(item)
                # Widget auto-updates on addItem

            # logger.debug(f"Market watch updated: {symbol} bid={bid:.5f} ask={ask:.5f} spread={spread_pips:.1f} color={spread_color}")

        except Exception as e:
            logger.exception(f"Failed to update market quote for {symbol}: {e}")

    def _refresh_orders(self):
        """
        Pull open orders from broker and refresh the table.

        TASK 3: Orders Table Integration
        Enhanced to draw order lines on chart as horizontal price levels.
        """
        try:
            # Check if broker is available
            if not hasattr(self, 'broker') or self.broker is None:
                logger.debug("Broker not available for orders refresh")
                return

            orders = self.broker.get_open_orders()
            self.orders_table.setRowCount(len(orders))

            # Clear existing order lines
            if not hasattr(self, '_order_lines'):
                self._order_lines = []

            # Remove old order lines from chart
            for line in self._order_lines:
                try:
                    if hasattr(self, 'ax') and self.ax and hasattr(line, 'remove'):
                        line.remove()
                except Exception:
                    pass
            self._order_lines.clear()

            # Populate table and draw order lines
            for r, o in enumerate(orders):
                vals = [
                    str(o.get("id","")),
                    o.get("time",""),
                    o.get("symbol",""),
                    o.get("side","") + " " + o.get("type",""),
                    str(o.get("volume","")),
                    f"{o.get('price','')}",
                    f"{o.get('sl','')}",
                    f"{o.get('tp','')}",
                    o.get("status","")
                ]
                for c, v in enumerate(vals):
                    self.orders_table.setItem(r, c, QTableWidgetItem(str(v)))

                # Draw order line on chart if symbol matches current chart symbol
                try:
                    order_symbol = o.get("symbol", "")
                    order_price = o.get("price")
                    order_side = o.get("side", "")

                    if order_symbol == getattr(self, 'symbol', None) and order_price:
                        self._draw_order_line(float(order_price), order_side, o.get("id", ""))
                except Exception as e:
                    logger.debug(f"Could not draw order line: {e}")

            # PlutoTouch logger.debug(f"Refreshed {len(orders)} orders with chart overlays")

        except Exception as e:
            logger.debug(f"Orders refresh skipped: {e}")

    def _draw_order_line(self, price: float, side: str, order_id: str):
        """
        Draw a horizontal line on the chart for an open order.

        TASK 3: Orders Table Integration
        """
        try:
            if not hasattr(self, 'ax') or self.ax is None:
                return

            # Determine color based on order side
            color = 'blue' if side.upper() == 'BUY' else 'red'

            # Try finplot/PyQtGraph approach first
            try:
                from pyqtgraph import InfiniteLine
                line = InfiniteLine(pos=price, angle=0, pen={'color': color, 'style': 2}, movable=False)  # style 2 = dashed
                line.setZValue(5)  # Above candles but below overlays
                self.ax.addItem(line)
                self._order_lines.append(line)
                logger.debug(f"Drew order line at {price} ({side})")
            except Exception:
                # Fallback to matplotlib if finplot not available
                try:
                    line = self.ax.axhline(y=price, color=color, linestyle='--', linewidth=1, alpha=0.7)
                    self._order_lines.append(line)
                except Exception as e:
                    logger.debug(f"Could not draw order line with matplotlib: {e}")

        except Exception as e:
            logger.exception(f"Failed to draw order line: {e}")

    def _toggle_orders(self, visible: bool):
        """
        Toggle visibility of order lines on chart.

        TASK 3: Orders Table Integration
        """
        try:
            if not hasattr(self, '_order_lines'):
                return

            for line in self._order_lines:
                try:
                    if hasattr(line, 'setVisible'):
                        line.setVisible(visible)
                    elif hasattr(line, 'set_visible'):
                        line.set_visible(visible)
                except Exception:
                    pass

            logger.debug(f"Order lines visibility set to {visible}")

        except Exception as e:
            logger.exception(f"Failed to toggle order lines: {e}")

    def _update_provider_label(self):
        """Update the active provider label on the chart showing RT and historical providers."""
        try:
            if hasattr(self, 'provider_label') and self.provider_label is not None:
                # Get settings
                try:
                    from forex_diffusion.utils.user_settings import get_setting
                except ImportError:
                    # Fallback to default values if import fails
                    primary = "tiingo"
                    use_ws = False
                    show_label = True
                else:
                    # Get active provider from settings
                    primary = get_setting("primary_data_provider", "tiingo")
                    use_ws = get_setting("use_websocket_streaming", False)
                    show_label = get_setting("show_provider_label", True)

                # Check if label should be hidden
                if not show_label:
                    self.provider_label.setText("")
                    return

                # Verify if WebSocket is ACTUALLY active by checking connector
                ws_actually_active = False
                if hasattr(self, '_main_window') and self._main_window:
                    # Check if Tiingo WebSocket connector exists and is running
                    if hasattr(self._main_window, 'tiingo_ws_connector'):
                        connector = self._main_window.tiingo_ws_connector
                        if connector and hasattr(connector, '_thread'):
                            if connector._thread and connector._thread.is_alive():
                                ws_actually_active = True

                # RT provider (WebSocket if ACTUALLY active, otherwise REST)
                rt_provider = primary.upper()
                rt_connection = "WS" if ws_actually_active else "REST"

                # Historical provider (always REST)
                historical_provider = primary.upper()
                historical_connection = "REST"

                # Check if main_window has active adapter info (more accurate)
                if hasattr(self, '_main_window') and self._main_window:
                    controller = getattr(self._main_window, "controller", None)
                    if controller and hasattr(controller, 'active_adapter'):
                        adapter = controller.active_adapter
                        if adapter:
                            # Get provider name from adapter
                            adapter_name = getattr(adapter, 'name', primary).upper()

                            # For RT connection, still check if WS is actually running
                            # (adapter might support WS but not be using it)
                            rt_provider = adapter_name
                            # Keep the previously determined ws_actually_active value
                            # Don't override it based on adapter alone

                            historical_provider = adapter_name

                # Build label text with RT and Historical on separate lines
                label_text = f"RT data: {rt_provider} ({rt_connection})\nHistorical: {historical_provider} (REST)"

                # Update text (check if widget still valid)
                if hasattr(self, 'provider_label') and self.provider_label is not None:
                    try:
                        # Test if C++ object still exists
                        _ = self.provider_label.toPlainText()
                        self.provider_label.setText(label_text)
                        
                        # Update position to top-right corner
                        if hasattr(self, 'main_plot') and self.main_plot is not None:
                            view_range = self.main_plot.viewRange()
                            if view_range and len(view_range) == 2:
                                x_range, y_range = view_range
                                # Position at top-right: max_x, max_y
                                self.provider_label.setPos(x_range[1], y_range[1])
                    except (RuntimeError, AttributeError):
                        # Widget deleted, clear reference
                        self.provider_label = None

        except Exception as e:
            logger.debug(f"Failed to update provider label: {e}")
    
    # ============================================================================
    # SMART BUFFER - Automatic data loading on chart scroll
    # ============================================================================
    
    def setup_smart_buffer(self):
        """
        Connect to chart view range changes to enable automatic data loading on scroll.
        
        This should be called after the plot is created (in plot_service).
        """
        try:
            # Get the main plot's ViewBox
            plot_service = self.controller.plot_service
            if not hasattr(plot_service, 'ax') or plot_service.ax is None:
                logger.debug("Smart buffer: plot not initialized yet")
                return
            
            viewbox = plot_service.ax.getViewBox()
            
            # Disconnect if already connected (avoid duplicates)
            try:
                viewbox.sigRangeChanged.disconnect(self._on_view_range_changed)
            except:
                pass  # Not connected yet
            
            # Connect to range change signal
            viewbox.sigRangeChanged.connect(self._on_view_range_changed)
            
            # Initialize buffer state (only first time)
            if not hasattr(self, '_smart_buffer_state'):
                self._smart_buffer_state = {
                    'loading': False,
                    'last_load_time': 0,
                    'min_load_interval': 2.0,  # Minimum 2 seconds between loads
                    'buffer_threshold': 0.2,  # Load when within 20% of edge
                    'buffer_size': 200  # Load 200 candles at a time
                }
                #logger.info("✅ Smart buffer initialized and connected to view range changes")
            # else:
            #    logger.debug("Smart buffer reconnected (state preserved)")
            
        except Exception as e:
            logger.error(f"Failed to setup smart buffer: {e}")
    
    def _on_view_range_changed(self, viewbox, ranges):
        """
        Called when user scrolls or zooms the chart.
        
        Args:
            viewbox: PyQtGraph ViewBox
            ranges: [[x_min, x_max], [y_min, y_max]]
        """
        try:
            # Check if buffer state exists
            if not hasattr(self, '_smart_buffer_state'):
                logger.warning("Smart buffer: state not initialized, skipping")
                return
            
            # Check if we have data
            if not hasattr(self, '_last_df') or self._last_df is None or self._last_df.empty:
                logger.debug("Smart buffer: no data available yet")
                return
            
            # Throttle: don't load too frequently
            import time
            current_time = time.time()
            if self._smart_buffer_state['loading']:
                # PlutoTouch logger.debug("Smart buffer: already loading, skipping")
                return  # Already loading
            
            last_load = self._smart_buffer_state['last_load_time']
            min_interval = self._smart_buffer_state['min_load_interval']
            if current_time - last_load < min_interval:
                return  # Too soon
            
            # Get visible X range (timestamps)
            x_range = ranges[0]
            x_min_visible, x_max_visible = x_range
            
            # Get data bounds
            if 'ts_utc' not in self._last_df.columns:
                logger.debug("Smart buffer: no ts_utc column")
                return
            
            data_min = self._last_df['ts_utc'].min()
            data_max = self._last_df['ts_utc'].max()
            
            # Calculate data range width
            data_width = data_max - data_min
            if data_width == 0:
                return
            
            # Calculate visible width
            visible_width = x_max_visible - x_min_visible
            
            # Check if scrolled near edges (need to load more data)
            threshold = self._smart_buffer_state['buffer_threshold']
            
            # Calculate distance from visible range to data boundaries as fraction of visible width
            # Left edge: how far is the left visible edge from the left data edge
            left_distance = (x_min_visible - data_min) / visible_width
            
            # Right edge: how far is the right visible edge from the right data edge  
            right_distance = (data_max - x_max_visible) / visible_width
            
            # Debug: log distances
            logger.debug(f"Smart buffer check: left_dist={left_distance:.2f}, right_dist={right_distance:.2f}, threshold={threshold}, visible=[{x_min_visible:.0f}, {x_max_visible:.0f}], data=[{data_min:.0f}, {data_max:.0f}]")
            
            # If scrolled close to left edge (or beyond it), load older data
            # left_distance < threshold means we're within threshold of the data start
            # left_distance < 0 means we've scrolled PAST the data start (need older data urgently!)
            if left_distance < threshold:
                logger.info(f"📜 Smart buffer: near/past left edge (distance={left_distance:.2f}×visible_width), loading older data...")
                self._load_more_data(direction='older', before_ts=data_min)
                return
            
            # If scrolled close to right edge (or beyond it), load newer data
            # right_distance < threshold means we're within threshold of the data end
            # right_distance < 0 means we've scrolled PAST the data end (need newer data urgently!)
            if right_distance < threshold:
                logger.info(f"📜 Smart buffer: near/past right edge (distance={right_distance:.2f}×visible_width), loading newer data...")
                self._load_more_data(direction='newer', after_ts=data_max)
                return
                
        except Exception as e:
            logger.error(f"Smart buffer range change error: {e}")
    
    def _load_more_data(self, direction: str, before_ts: int = None, after_ts: int = None):
        """
        Load more data in the specified direction.
        
        Args:
            direction: 'older' or 'newer'
            before_ts: Load data before this timestamp (for 'older')
            after_ts: Load data after this timestamp (for 'newer')
        """
        try:
            self._smart_buffer_state['loading'] = True
            
            symbol = getattr(self, 'symbol', 'EURUSD')
            timeframe = getattr(self, 'timeframe', '1m')
            buffer_size = self._smart_buffer_state['buffer_size']
            
            # Calculate timestamp range based on direction
            if direction == 'older':
                # Load older data: end_ms = before_ts, start_ms = before_ts - buffer
                end_ms = int(before_ts)
                
                # Calculate buffer duration in milliseconds
                tf_ms = {
                    "tick": 1000, "1m": 60_000, "5m": 300_000, "15m": 900_000,
                    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
                    "1d": 86_400_000, "1w": 604_800_000
                }
                candle_duration_ms = tf_ms.get(timeframe, 60_000)
                buffer_duration_ms = candle_duration_ms * buffer_size
                
                start_ms = end_ms - buffer_duration_ms
                
            else:  # newer
                # Load newer data: start_ms = after_ts, end_ms = after_ts + buffer
                start_ms = int(after_ts)
                
                tf_ms = {
                    "tick": 1000, "1m": 60_000, "5m": 300_000, "15m": 900_000,
                    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
                    "1d": 86_400_000, "1w": 604_800_000
                }
                candle_duration_ms = tf_ms.get(timeframe, 60_000)
                buffer_duration_ms = candle_duration_ms * buffer_size
                
                from datetime import datetime, timezone
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                end_ms = min(start_ms + buffer_duration_ms, now_ms)
            
            logger.info(f"Smart buffer: loading {direction} data for {symbol} {timeframe} from {start_ms} to {end_ms}")
            
            # Load data in background
            from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject
            
            class LoadMoreSignals(QObject):
                finished = Signal(object, str)  # (df, direction)
            
            class LoadMoreTask(QRunnable):
                def __init__(self, service, symbol, timeframe, start_ms, end_ms, direction, signals):
                    super().__init__()
                    self.service = service
                    self.symbol = symbol
                    self.timeframe = timeframe
                    self.start_ms = start_ms
                    self.end_ms = end_ms
                    self.direction = direction
                    self.signals = signals
                
                def run(self):
                    try:
                        logger.debug(f"[BG] Loading {self.direction} data...")
                        df = self.service._load_candles_from_db(
                            self.symbol, self.timeframe,
                            limit=None, start_ms=self.start_ms, end_ms=self.end_ms
                        )
                        
                        if df is not None and not df.empty:
                            logger.info(f"[BG] Loaded {len(df)} {self.direction} candles")
                            self.signals.finished.emit(df, self.direction)
                        else:
                            logger.debug(f"[BG] No {self.direction} data found")
                            self.signals.finished.emit(None, self.direction)
                    except Exception as e:
                        logger.error(f"[BG] Failed to load {self.direction} data: {e}")
                        self.signals.finished.emit(None, self.direction)
            
            signals = LoadMoreSignals()
            signals.finished.connect(self._on_more_data_loaded)
            task = LoadMoreTask(self, symbol, timeframe, start_ms, end_ms, direction, signals)
            QThreadPool.globalInstance().start(task)
            
        except Exception as e:
            logger.error(f"Failed to load more data: {e}")
            self._smart_buffer_state['loading'] = False
    
    def _on_more_data_loaded(self, new_df, direction):
        """Called when background data loading completes."""
        try:
            import time
            self._smart_buffer_state['loading'] = False
            self._smart_buffer_state['last_load_time'] = time.time()
            
            if new_df is None or new_df.empty:
                logger.debug(f"Smart buffer: no {direction} data to append")
                return
            
            # Merge with existing data
            if self._last_df is None or self._last_df.empty:
                logger.warning("Smart buffer: no existing data to merge with")
                return
            
            # Save current view range
            plot_service = self.controller.plot_service
            if hasattr(plot_service, 'ax') and plot_service.ax is not None:
                try:
                    current_range = plot_service.ax.viewRange()
                    plot_service._saved_view_range = current_range
                    logger.debug(f"Saved view range before merge: {current_range}")
                except:
                    pass
            
            # Combine dataframes
            if direction == 'older':
                # Prepend older data
                combined_df = pd.concat([new_df, self._last_df], ignore_index=True)
            else:
                # Append newer data
                combined_df = pd.concat([self._last_df, new_df], ignore_index=True)
            
            # Remove duplicates (keep first occurrence)
            if 'ts_utc' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['ts_utc'], keep='first')
                combined_df = combined_df.sort_values('ts_utc').reset_index(drop=True)
            
            logger.info(f"✅ Smart buffer: merged {len(new_df)} {direction} candles (total: {len(self._last_df)} → {len(combined_df)})")
            
            # Update plot with merged data
            self.controller.plot_service.update_plot(combined_df)
            
        except Exception as e:
            logger.error(f"Failed to merge loaded data: {e}")
