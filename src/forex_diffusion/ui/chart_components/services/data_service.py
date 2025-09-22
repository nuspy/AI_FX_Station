from __future__ import annotations

from typing import Optional
import pandas as pd
from loguru import logger
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox, QTableWidgetItem

from .base import ChartServiceBase


class DataService(ChartServiceBase):
    """Auto-generated service extracted from ChartTab."""

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
            # Market watch update
            try:
                if sym:
                    bid_val = payload.get('bid')
                    ask_val = payload.get('ask', payload.get('offer'))
                    self._update_market_quote(sym, bid_val, ask_val, payload.get('ts_utc'))
            except Exception:
                pass

            # Update chart buffer only for current symbol (O(1) append)
            if sym and sym == getattr(self, "symbol", None):
                try:
                    ts = int(payload.get("ts_utc"))
                    y = payload.get("close", payload.get("price"))
                    bid_val = payload.get('bid')
                    ask_val = payload.get('ask', payload.get('offer'))
                    # se buffer vuoto: crea
                    if self._last_df is None or self._last_df.empty:
                        row = {"ts_utc": ts, "price": y, "bid": bid_val, "ask": ask_val}
                        self._last_df = pd.DataFrame([row])
                    else:
                        last_ts = int(self._last_df["ts_utc"].iloc[-1])
                        if ts > last_ts:
                            # append
                            self._last_df.loc[len(self._last_df)] = {"ts_utc": ts, "price": y, "bid": bid_val, "ask": ask_val}
                            # trim buffer
                            if len(self._last_df) > 10000:
                                self._last_df = self._last_df.iloc[-10000:].reset_index(drop=True)
                        elif ts == last_ts:
                            # update last row
                            col = "close" if "close" in self._last_df.columns else "price"
                            self._last_df.at[len(self._last_df)-1, col] = y
                            if bid_val is not None:
                                self._last_df.at[len(self._last_df)-1, 'bid'] = bid_val
                            if ask_val is not None:
                                self._last_df.at[len(self._last_df)-1, 'ask'] = ask_val
                        # se ts < last_ts, ignora (fuori ordine)
                except Exception:
                    pass

                # update bid/ask label
                try:
                    if payload.get('bid') is not None and payload.get('ask') is not None:
                        self.bidask_label.setText(f"Bid: {float(payload['bid']):.5f}    Ask: {float(payload['ask']):.5f}")
                except Exception:
                    try:
                        self.bidask_label.setText(f"Bid: {payload.get('bid')}    Ask: {payload.get('ask')}")
                    except Exception:
                        pass

                # mark dirty for throttled redraw
                self._rt_dirty = True
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
                prev_xlim = self.ax.get_xlim()
                prev_ylim = self.ax.get_ylim()
            except Exception:
                prev_xlim = prev_ylim = None
            # redraw base chart (quantiles overlay mantenuti da _forecasts)
            if self._last_df is not None and not self._last_df.empty:
                self.update_plot(self._last_df, restore_xlim=prev_xlim, restore_ylim=prev_ylim)
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
        try:
            # get current view in data coordinates
            xlim = self.ax.get_xlim()
            if not xlim or xlim[0] >= xlim[1]:
                return
            # matplotlib date floats -> UTC ms
            import matplotlib.dates as mdates
            left_dt = mdates.num2date(xlim[0])
            right_dt = mdates.num2date(xlim[1])
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

            # carica da DB solo la finestra necessaria
            df = self._load_candles_from_db(sym, tf_req, limit=50000, start_ms=start_ms, end_ms=end_ms)
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

    def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
        self.db_service = db_service
        self.symbol = symbol
        self.timeframe = timeframe
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
            df = self._load_candles_from_db(self.symbol, self.timeframe, limit=3000, start_ms=start_ms_view)
            if df is not None and not df.empty:
                self.update_plot(df)
                # logger.info("Plotted {} points for {} {}", len(df), self.symbol, self.timeframe)
            else:
                logger.info("No candles found in DB for {} {}", self.symbol, self.timeframe)
        except Exception as e:
            logger.exception("Initial load failed: {}", e)

    def _on_symbol_changed(self, new_symbol: str):
        """Handle symbol change from combo: update context and reload candles from DB."""
        try:
            if not new_symbol:
                return
            self.symbol = new_symbol
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
            except Exception:
                start_ms_view = None
            df = self._load_candles_from_db(new_symbol, getattr(self, "timeframe", "1m"), limit=3000, start_ms=start_ms_view)
            if df is not None and not df.empty:
                self.update_plot(df)
        except Exception as e:
            logger.exception("Failed to switch symbol: {}", e)

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
            def __init__(self, svc, symbol, timeframe, start_override, signals):
                super().__init__()
                self.svc = svc
                self.symbol = symbol
                self.timeframe = timeframe
                self.start_override = start_override
                self.signals = signals

            def run(self):
                ok = True
                try:
                    def _cb(pct: int):
                        try:
                            self.signals.progress.emit(int(pct))
                        except Exception:
                            pass
                    self.svc.backfill_symbol_timeframe(self.symbol, self.timeframe, force_full=False, progress_cb=_cb, start_ms_override=self.start_override)
                except Exception as e:
                    ok = False
                finally:
                    # keep REST setting unchanged
                    try:
                        self.signals.finished.emit(ok)
                    except Exception:
                        pass

        self.setDisabled(True)
        self.backfill_progress.setRange(0, 100)
        self.backfill_progress.setValue(0)

        self._bf_signals = BackfillSignals(self)
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
                    QMessageBox.information(self.view, "Backfill", f"Backfill completato per {sym} {tf}.")
                else:
                    QMessageBox.warning(self.view, "Backfill", "Backfill fallito (vedi log).")
            finally:
                self.setDisabled(False)
                self.backfill_progress.setValue(100)

        self._bf_signals.finished.connect(_on_done)
        job = BackfillJob(ms, sym, tf, start_override, self._bf_signals)
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
                return pd.DataFrame()
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
                        return pd.DataFrame()
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
                        return pd.DataFrame()
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

    def _refresh_orders(self):
        """Pull open orders from broker and refresh the table."""
        try:
            orders = self.broker.get_open_orders()
            self.orders_table.setRowCount(len(orders))
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
        except Exception:
            pass
