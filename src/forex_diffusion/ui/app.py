"""
UI application helpers for MagicForex GUI.
"""
from __future__ import annotations

import os
from typing import Any

from loguru import logger
from PySide6.QtWidgets import QWidget, QTabWidget

from ..services.db_service import DBService
from ..services.db_writer import DBWriter
from ..services.marketdata import MarketDataService
from ..services.tiingo_ws_connector import TiingoWSConnector
from ..services.local_ws import LocalWebsocketServer
from ..services.aggregator import AggregatorService
from .controllers import UIController
from .training_tab import TrainingTab
from .signals_tab import SignalsTab
from .chart_tab import ChartTab
from .backtesting_tab import BacktestingTab

def setup_ui(
    main_window: QWidget, 
    layout, 
    menu_bar, 
    viewer, 
    status_label, 
    engine_url: str = "http://127.0.0.1:8000", 
    use_test_server: bool = False
) -> dict:
    """
    Initializes all UI components, services, and their connections.
    """
    logger.critical("--- EXECUTING LATEST APP.PY VERSION ---")
    result: dict[str, Any] = {}

    # --- Core Services ---
    db_service = DBService()
    market_service = MarketDataService(database_url=db_service.engine.url)
    # UI: abilita REST per backfill alla apertura
    try:
        setattr(market_service, "rest_enabled", True)
    except Exception:
        pass
    db_writer = DBWriter(db_service=db_service)
    db_writer.start()

    result["db_service"] = db_service
    result["market_service"] = market_service
    result["db_writer"] = db_writer

    # --- Start Aggregator Service ---
    symbols_to_aggregate = ["EUR/USD"]  # Or load from config
    aggregator = AggregatorService(engine=db_service.engine, symbols=symbols_to_aggregate)
    aggregator.start()
    result["aggregator"] = aggregator
    logger.info("AggregatorService started.")

    # --- UI Tabs and Controller ---
    controller = UIController(main_window=main_window, market_service=market_service, db_writer=db_writer)
    controller.bind_menu_signals(menu_bar.signals)
    result["controller"] = controller
    try:
        # expose controller on main window so tabs can reach market_service
        setattr(main_window, "controller", controller)
    except Exception:
        pass

    tab_widget = QTabWidget()
    chart_tab = ChartTab(main_window)
    training_tab = TrainingTab(main_window)
    signals_tab = SignalsTab(main_window, db_service=db_service)
    backtesting_tab = BacktestingTab(main_window)

    tab_widget.addTab(chart_tab, "Chart")
    tab_widget.addTab(training_tab, "Training")
    tab_widget.addTab(signals_tab, "Signals")
    tab_widget.addTab(backtesting_tab, "Backtesting")
    layout.addWidget(tab_widget)
    result["chart_tab"] = chart_tab
    result["training_tab"] = training_tab
    result["backtesting_tab"] = backtesting_tab
    result["tab_widget"] = tab_widget
    try:
        chart_tab.controller = controller
        controller.chart_tab = chart_tab
        setattr(chart_tab, "controller", controller)

    except Exception:
        pass
    # expose chart_tab on controller for symbol/timeframe discovery
    try:
        controller.chart_tab = chart_tab
        setattr(chart_tab, "controller", controller)
    except Exception:
        pass
    # connect ChartTab forecast requests to controller handler, and results back to the chart
    try:
        chart_tab.forecastRequested.connect(controller.handle_forecast_payload)
    except Exception:
        logger.warning("Failed to connect chart_tab.forecastRequested")
    # Wrap forecastReady -> compute adherence if possible, then forward to chart
    try:
        def _on_forecast_ready_with_adherence(df, quantiles):
            try:
                # Lazy import to avoid hard deps at import time
                from forex_diffusion.postproc.adherence import adherence_metrics, atr_sigma_from_df
                import pandas as _pd
                import numpy as _np
                # Anchor-based sigma from pre-anchor candles
                anchor_ts = None
                try:
                    anchor_ts = int(_pd.to_numeric(df["ts_utc"].iloc[-1]))
                except Exception:
                    pass
                atr_n = 14
                try:
                    # try take from controller indicators settings if available
                    atr_n = int(getattr(controller, "indicators_settings", {}).get("atr_n", 14))
                except Exception:
                    pass
                sigma_vol = float(atr_sigma_from_df(df, n=atr_n, pre_anchor_only=True, anchor_ts=anchor_ts, robust=True))
                # Prepare alignment data
                fut_ts = list(quantiles.get("future_ts") or [])
                m = list(quantiles.get("q50") or [])
                q05 = list(quantiles.get("q05") or [])
                q95 = list(quantiles.get("q95") or [])
                actual_ts, actual_y = [], []
                if fut_ts:
                    # Fetch realized closes at those timestamps from DB; fallback to in-memory merge_asof if needed
                    try:
                        from sqlalchemy import text
                        engine = getattr(controller.market_service, "engine", None)
                        if engine is None:
                            engine = getattr(getattr(controller, "db_service", None), "engine", None)
                        if engine is not None:
                            with engine.connect() as conn:
                                vals = ",".join(f"({int(t)})" for t in fut_ts)
                                q = text(
                                    f"WITH fut(ts) AS (VALUES {vals}) "
                                    "SELECT c.ts_utc AS ts, c.close AS y "
                                    "FROM fut JOIN market_data_candles c ON c.ts_utc = fut.ts "
                                    "WHERE c.symbol = :symbol AND c.timeframe = :timeframe "
                                    "ORDER BY c.ts_utc ASC"
                                )
                                sym = getattr(chart_tab, 'symbol', 'EUR/USD')
                                tf = getattr(chart_tab, 'timeframe', '1m')
                                rows = conn.execute(q, {'symbol': sym, 'timeframe': tf}).fetchall()
                                if rows:
                                    tmp = _pd.DataFrame(rows, columns=['ts', 'y'])
                                    actual_ts = tmp['ts'].astype('int64').tolist()
                                    actual_y = tmp['y'].astype(float).tolist()
                    except Exception:
                        pass
                    # Fallback: align to nearest in-memory bar within timeframe tolerance (datetime-based asof)
                    try:
                        if (not actual_ts) and hasattr(chart_tab, "_last_df") and chart_tab._last_df is not None and not chart_tab._last_df.empty:
                            dfa = chart_tab._last_df.copy()
                            ycol = "close" if "close" in dfa.columns else "price"
                            dfa = dfa.dropna(subset=["ts_utc", ycol]).reset_index(drop=True)
                            dfa["ts"] = _pd.to_numeric(dfa["ts_utc"], errors="coerce").astype("int64")
                            dfa["y"] = _pd.to_numeric(dfa[ycol], errors="coerce").astype(float)
                            # Build datetime keys
                            dfa["ts_dt"] = _pd.to_datetime(dfa["ts"], unit="ms", utc=True).dt.tz_convert(None)
                            df_fut = _pd.DataFrame({"ts": _pd.to_numeric(fut_ts, errors="coerce").astype("int64")}).sort_values("ts")
                            df_fut["ts_dt"] = _pd.to_datetime(df_fut["ts"], unit="ms", utc=True).dt.tz_convert(None)
                            # timeframe tolerance in ms
                            def _tf_ms(tf: str) -> int:
                                try:
                                    s = str(tf).strip().lower()
                                    if s.endswith("m"): return int(s[:-1]) * 60_000
                                    if s.endswith("h"): return int(s[:-1]) * 3_600_000
                                    if s.endswith("d"): return int(s[:-1]) * 86_400_000
                                except Exception:
                                    pass
                                return 60_000
                            tol_ms = max(1_000, int(0.51 * _tf_ms(getattr(chart_tab, 'timeframe', '1m') or '1m')))
                            merged = _pd.merge_asof(
                                df_fut.sort_values("ts_dt"),
                                dfa.sort_values("ts_dt")[["ts_dt", "ts", "y"]],
                                left_on="ts_dt",
                                right_on="ts_dt",
                                direction="nearest",
                                tolerance=_pd.Timedelta(milliseconds=tol_ms),
                                suffixes=("", "_real"),
                            ).dropna().reset_index(drop=True)
                            if not merged.empty:
                                actual_ts = merged["ts"].astype("int64").tolist()
                                actual_y = merged["y"].astype(float).tolist()
                    except Exception:
                        pass
                # Compute metrics if we have realized points (inner alignment inside the function will prune mismatches)
                metrics = {}
                try:
                    if fut_ts and (actual_ts and actual_y):
                        metrics = adherence_metrics(
                            fut_ts=fut_ts, m=m, q05=q05, q95=q95,
                            actual_ts=actual_ts, actual_y=actual_y,
                            sigma_vol=sigma_vol, band_target=0.90
                        )
                except Exception:
                    metrics = {}
                # Attach metrics to quantiles for downstream consumers
                try:
                    quantiles = dict(quantiles or {})
                    if metrics:
                        quantiles["adherence_metrics"] = metrics
                except Exception:
                    pass
            except Exception:
                # never block UI due to metrics
                pass
            try:
                chart_tab.on_forecast_ready(df, quantiles)
            except Exception:
                # fallback: ignore if chart handler fails
                pass

            # --- Draw adherence badge at the end of upper tolerance (q95) line ---
            try:
                q = quantiles or {}
                metrics = (q.get("adherence_metrics") or {})
                if metrics and hasattr(chart_tab, "ax") and hasattr(chart_tab, "canvas"):
                    import numpy as _np

                    adh = metrics.get("adherence", None)
                    fut_ts = list(q.get("future_ts") or [])
                    q95 = list(q.get("q95") or [])
                    m_vals = list(q.get("q50") or [])
                    if isinstance(adh, (int, float)) and fut_ts and q95 and m_vals and len(fut_ts) == len(q95) == len(m_vals):
                        ax = chart_tab.ax
                        x_last = float(fut_ts[-1])
                        y_last = float(q95[-1])

                        # Detect median line color (fallback to default)
                        line_color = "#555555"
                        try:
                            lines = list(reversed(ax.get_lines()))
                            x_ref = _np.asarray(fut_ts, dtype=float)
                            y_ref = _np.asarray(m_vals, dtype=float)
                            for ln in lines:
                                xd = _np.asarray(ln.get_xdata(), dtype=float)
                                yd = _np.asarray(ln.get_ydata(), dtype=float)
                                # quick shape check, then allclose
                                if xd.size == x_ref.size and yd.size == y_ref.size:
                                    if _np.allclose(xd, x_ref, rtol=1e-6, atol=1e-6) and _np.allclose(yd, y_ref, rtol=1e-6, atol=1e-6):
                                        line_color = ln.get_color()
                                        break
                        except Exception:
                            pass

                        # Inside-axes placement: clamp to current x/y limits with small margin
                        try:
                            xmin, xmax = ax.get_xlim()
                            ymin, ymax = ax.get_ylim()
                            dx = max(1.0, 0.015 * (xmax - xmin))
                            dy = 0.015 * (ymax - ymin)
                            x_text = min(x_last, xmax - dx)
                            x_text = max(x_text, xmin + dx)
                            y_text = min(max(y_last, ymin + dy), ymax - dy)
                        except Exception:
                            x_text, y_text = x_last, y_last

                        # Manage per-forecast badge registry
                        try:
                            label_key = str(q.get("label") or q.get("source") or "forecast")
                        except Exception:
                            label_key = "forecast"
                        if not hasattr(chart_tab, "_adh_badges"):
                            setattr(chart_tab, "_adh_badges", {})
                        # Remove previous badge for this label if any
                        old = chart_tab._adh_badges.get(label_key)
                        if old is not None:
                            try:
                                old.remove()
                            except Exception:
                                pass

                        text_str = f"{float(adh):.2f}"
                        badge = ax.text(
                            x_text, y_text, text_str,
                            transform=ax.transData,
                            ha="left", va="center",
                            fontsize=9, fontweight="bold",
                            bbox=dict(
                                facecolor="white",
                                edgecolor=line_color,
                                boxstyle="round,pad=0.38",  # rounded -> oval-like and width adapts to text
                                linewidth=1.2,
                                alpha=0.98
                            ),
                            zorder=100,
                        )
                        chart_tab._adh_badges[label_key] = badge
                        try:
                            chart_tab.canvas.draw_idle()
                        except Exception:
                            pass
            except Exception:
                # never block UI for badge drawing
                pass

        controller.signals.forecastReady.connect(_on_forecast_ready_with_adherence)
    except Exception:
        logger.warning("Failed to connect controller.forecastReady to chart")

    # bring Training tab to front on menu->Train
    try:
        menu_bar.signals.trainRequested.connect(lambda: tab_widget.setCurrentWidget(training_tab))
    except Exception:
        pass

    # --- WebSocket and Direct Data Flow ---
    ws_uri = "ws://127.0.0.1:8766" if use_test_server else "wss://api.tiingo.com/fx"
    if use_test_server:
        logger.info(f"Redirecting Tiingo WebSocket to test server: {ws_uri}")

    # track backtest state to suppress misleading WS logs during runs
    backtest_active = {"flag": False}

    def _ws_status(msg: str):
        try:
            if msg == "ws_down":
                if backtest_active["flag"]:
                    logger.info("Realtime WS stopped (backtest active) - suppressing REST fallback.")
                    controller.signals.status.emit("Realtime: WS stopped (backtest active)")
                else:
                    logger.warning("Realtime WS down detected. REST fallback is DISABLED.")
                    controller.signals.status.emit("Realtime: WS down (no REST fallback)")
                #controller.signals.status.emit("Realtime: WS down (fallback REST attivo)")

            elif msg == "ws_restored":
                logger.info("Realtime WS restored.")
                controller.signals.status.emit("Realtime: WS restored")
        except Exception:
            pass

    connector = None
    if os.environ.get("FOREX_ENABLE_WS", "1") == "1":
        connector = TiingoWSConnector(
            uri=ws_uri,
            api_key=os.environ.get("TIINGO_APIKEY"),
            tickers=["eurusd"],
            chart_handler=chart_tab._handle_tick,
            db_handler=db_writer.write_tick_async
            , status_handler=_ws_status
        )
        connector.start()
        result["tiingo_ws_connector"] = connector
        logger.info("TiingoWSConnector started with direct handlers for ChartTab and DBWriter.")
    else:
        logger.info("WS connector disabled (FOREX_ENABLE_WS!=1)")

    # --- Final UI Setup ---
    default_symbol = "EUR/USD"
    default_tf = "1m"
    chart_tab.set_symbol_timeframe(db_service, default_symbol, default_tf)

    # Auto backfill on startup (abilitato per default)
    try:
        if os.environ.get("FOREX_DISABLE_AUTOBACKFILL", "0") == "1":
            logger.info("Auto backfill disabled by FOREX_DISABLE_AUTOBACKFILL=1")
            raise RuntimeError("skip_autobackfill")
        from PySide6.QtCore import QRunnable, QThreadPool, QObject, Signal
        class _BFSignals(QObject):
            progress = Signal(int)
            status = Signal(str)
            done = Signal()

        class _BFJob(QRunnable):
            def __init__(self, market_service, symbols, years, months, signals):
                super().__init__()
                self.ms = market_service
                self.symbols = symbols
                self.years = int(years)
                self.months = int(months)
                self.signals = signals

            def run(self):
                import math, time
                total = 0
                # compute total subranges estimate by summing across symbols later; we update per-symbol progressively
                for sym in self.symbols:
                    try:
                        first_ts = self.ms._get_first_candle_ts(sym)
                        if first_ts is None:
                            continue
                        total += 1
                    except Exception:
                        continue
                done = 0
                for sym in self.symbols:
                    try:
                        first_ts = self.ms._get_first_candle_ts(sym)
                        if first_ts is None:
                            # skip symbols with no candles at all
                            self.signals.status.emit(f"[Backfill] Skip {sym}: no candles in DB")
                            continue
                        # compute start override from UI (0/0 -> full from first), else from now - (years, months)
                        import datetime
                        now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
                        # compute LAST candle ts from DB across ALL timeframes for incremental logic
                        last_ts = None
                        try:
                            from sqlalchemy import text
                            eng = getattr(self.ms, "engine", None)
                            if eng is not None:
                                with eng.connect() as conn:
                                    row = conn.execute(
                                        text("SELECT MAX(ts_utc) FROM market_data_candles WHERE symbol = :sym"),
                                        {"sym": sym},
                                    ).fetchone()
                                    if row and row[0] is not None:
                                        last_ts = int(row[0])
                        except Exception:
                            last_ts = None

                        # If already fresh (updated in last 5 minutes), skip backfill for this symbol
                        try:
                            if last_ts is not None and (now_ms - int(last_ts)) < 5 * 60 * 1000:
                                self.signals.status.emit(f"[Backfill] {sym}: up-to-date (skip)")
                                continue
                        except Exception:
                            pass

                        if (self.years == 0 and self.months == 0):
                            # default: incremental from last known candle (if any), else from the very beginning
                            if last_ts is not None and last_ts > 0:
                                start_ms = int(last_ts) + 1  # avoid re-fetching last included bar
                                self.signals.status.emit(f"[Backfill] {sym}: incremental from {start_ms}")
                            else:
                                start_ms = first_ts
                                self.signals.status.emit(f"[Backfill] {sym}: initial full from {start_ms}")
                        else:
                            # explicit historical range requested from UI
                            days = self.years * 365 + self.months * 30
                            start_ms = max(first_ts, now_ms - days * 24 * 3600 * 1000)
                            self.signals.status.emit(f"[Backfill] {sym}: range start {start_ms}")

                        # nested progress bridge
                        def _cb(p):
                            try:
                                self.signals.status.emit(f"[Backfill] {sym}: {p}%")
                            except Exception:
                                pass
                        # use '1d' to cover all timeframes up to daily
                        self.ms.backfill_symbol_timeframe(sym, "1d", force_full=False, progress_cb=_cb, start_ms_override=start_ms)
                    except Exception as e:
                        try:
                            self.signals.status.emit(f"[Backfill] {sym} failed: {e}")
                        except Exception:
                            pass
                    finally:
                        done += 1
                        pct = int(min(100, max(0, done / max(1,total) * 100)))
                        try:
                            self.signals.progress.emit(pct)
                        except Exception:
                            pass
                try:
                    self.signals.done.emit()
                except Exception:
                    pass

        bf_signals = _BFSignals()
        bf_signals.status.connect(status_label.setText)
        bf_signals.progress.connect(lambda p: status_label.setText(f"Backfill: {p}%"))
        QThreadPool.globalInstance().start(_BFJob(market_service, chart_tab._symbols_supported, chart_tab.years_combo.currentText(), chart_tab.months_combo.currentText(), bf_signals))
    except Exception as e:
        if str(e) != "skip_autobackfill":
            logger.warning("Auto backfill job not started: {}", e)

    controller.signals.status.connect(status_label.setText)
    controller.signals.error.connect(status_label.setText)

    # --- Backtesting Tab Handlers ---
    try:
        import httpx
        def _on_bt_start(payload: dict):
            try:
                base = engine_url.rstrip("/")
                # small retry/backoff for async path
                def _get_with_retry(client, url, max_wait_s=6.0):
                    import time as _t
                    t0 = _t.time()
                    delay = 0.3
                    while True:
                        rr = client.get(url, params={"top_k": 20})
                        if rr.status_code == 200 and rr.json().get("results"):
                            return rr
                        if _t.time() - t0 > max_wait_s:
                            return rr
                        _t.sleep(delay)
                        delay = min(1.2, delay * 1.5)

                # Temporarily disable REST backfill and WS during backtest
                prev_rest_flag = getattr(market_service, "rest_enabled", False)
                try:
                    setattr(market_service, "rest_enabled", False)
                except Exception:
                    pass
                try:
                    if connector is not None:
                        try:
                            connector.stop()
                        except Exception:
                            pass
                except Exception:
                    pass
                # mark backtest active to adjust WS status reporting
                backtest_active["flag"] = True

                with httpx.Client(timeout=60.0) as client:
                    r = client.post(f"{base}/backtests", json=payload)
                    r.raise_for_status()
                    job = r.json()
                    job_id = int(job.get("job_id", 0))
                    # fetch results with retry (async queue) and update status label with progress polling
                    if job_id:
                        # quick status poll loop while waiting
                        try:
                            import time as _t
                            t0 = _t.time()
                            while _t.time() - t0 < 6.0:
                                st = client.get(f"{base}/backtests/{job_id}/status")
                                if st.status_code == 200:
                                    js = st.json()
                                    p = js.get("progress")
                                    if p is not None:
                                        status_label.setText(f"Backtesting: in corso ({int(float(p)*100)}%)")
                                    if js.get("status") == "done":
                                        break
                                _t.sleep(0.5)
                        except Exception:
                            pass
                        rr = _get_with_retry(client, f"{base}/backtests/{job_id}/results")
                        if rr.status_code == 200:
                            data = rr.json()
                            for row in data.get("results", []):
                                p = row.get("payload") or {}
                                model = (p.get("model") or p.get("model_name") or "?")
                                ptype = p.get("ptype") or p.get("prediction_type") or "?"
                                tf = p.get("timeframe") or "?"
                                backtesting_tab.add_result_row(
                                    model, ptype, tf,
                                    float(row.get("adherence_mean") or 0.0),
                                    float(row.get("p50") or 0.0),
                                    float(row.get("win_rate_delta") or 0.0),
                                    row.get("coverage_observed"), row.get("band_efficiency"), row.get("composite_score"),
                                    int(row.get("config_id") or 0)
                                )
                            # attach job_id to tab for apply-config calls
                            try:
                                setattr(backtesting_tab, "last_job_id", job_id)
                            except Exception:
                                pass
                status_label.setText("Backtesting: job completed")
            finally:
                # restore flags/services
                try:
                    setattr(market_service, "rest_enabled", prev_rest_flag)
                except Exception:
                    pass
                try:
                    if connector is not None and os.environ.get("FOREX_ENABLE_WS", "1") == "1":
                        connector.start()
                except Exception:
                    pass
                backtest_active["flag"] = False
            #except Exception as e:
            #    status_label.setText(f"Backtesting error: {e}")
        backtesting_tab.startRequested.connect(_on_bt_start)
    except Exception:
        pass

    # --- Graceful shutdown on app exit ---
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        def _graceful_shutdown():
            try:
                logger.info("Shutting down services...")
                try:
                    connector.stop()
                except Exception:
                    pass
                try:
                    aggregator.stop()
                except Exception:
                    pass
                try:
                    db_writer.stop()
                except Exception:
                    pass
            except Exception:
                pass
        if app is not None:
            app.aboutToQuit.connect(_graceful_shutdown)
    except Exception:
        pass

    return result
