# src/forex_diffusion/ui/chart_tab.py
from __future__ import annotations

from typing import Optional
import pandas as pd

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox
from PySide6.QtCore import QTimer, Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sqlalchemy import MetaData, select
from loguru import logger

from ..features.indicators import sma, ema, bollinger, rsi, macd


class ChartTab(QWidget):
    """
    Simple matplotlib-based chart tab with pan/zoom toolbar.
    Exposes update_plot(df) where df is a DataFrame with ts_utc and close/open/high/low.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # keep reference to main window (statusBar) for visible notifications
        self._main_window = parent
        self.layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure(figsize=(6,4)))
        self.toolbar = NavigationToolbar(self.canvas, self)
        # indicators button - always create the button (visible), enable only if dialog available
        self.indicators_btn = QPushButton("Indicatori")
        self.indicators_btn.setToolTip("Apri la finestra degli indicatori")
        # place toolbar and button in a horizontal layout; always add topbar so button is visible
        topbar = QWidget()
        top_layout = QHBoxLayout(topbar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.toolbar)
        top_layout.addWidget(self.indicators_btn)
        # real-time bid/ask label (initial placeholder)
        try:
            self.bidask_label = QLabel("Bid: -    Ask: -")
            self.bidask_label.setMinimumWidth(220)
            self.bidask_label.setStyleSheet("font-weight: bold;")
            top_layout.addWidget(self.bidask_label)
        except Exception:
            # fallback: ignore if QLabel not available
            self.bidask_label = None
        self.layout.addWidget(topbar)

        # Always connect the indicators button; if packaged dialog is missing, we'll show a dynamic fallback
        try:
            self.indicators_btn.clicked.connect(self._open_indicators_dialog)
            self.indicators_btn.setEnabled(True)
        except Exception:
            # best-effort: keep button visible but it will be inert (very unlikely)
            try:
                self.indicators_btn.setEnabled(False)
                self.indicators_btn.setToolTip("Cannot connect indicators handler")
            except Exception:
                pass
        self.layout.addWidget(self.canvas)
        # ensure canvas receives wheel/mouse events and can get focus for interactions
        try:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setFocus()
            self.canvas.setMouseTracking(True)
        except Exception:
            pass
        self.ax = self.canvas.figure.subplots()
        self.ax.set_title("Historical chart")
        self._last_df = None
        # track plotted artists per indicator name so we can remove/replace them instead of duplicating
        self._indicator_artists: dict[str, list] = {}
        # auxiliary axes for indicators that require their own y-axis (rsi, macd)
        self._aux_axes: dict[str, any] = {}

        # load persisted indicators config if present (normalize so "enabled" flags are honored)
        try:
            from pathlib import Path
            import json
            cfg_path = Path(__file__).resolve().parents[3] / "configs" / "indicators.json"
            if cfg_path.exists():
                try:
                    with cfg_path.open("r", encoding="utf-8") as fh:
                        raw = json.load(fh)
                        try:
                            # normalize into canonical structure {indicator: {enabled:bool,...}}
                            self._indicator_cfg = self._normalize_indicator_cfg(raw)
                            try:
                                logger.info(f"Loaded normalized indicator config from {cfg_path}")
                            except Exception:
                                logger.info("Loaded indicator config")
                        except Exception:
                            # fallback: keep raw but ensure subsequent code handles missing 'enabled'
                            self._indicator_cfg = raw
                except Exception:
                    self._indicator_cfg = None
            else:
                self._indicator_cfg = None
        except Exception:
            self._indicator_cfg = None

        # load persisted indicators config if present (normalize so enabled flags are honored)
        try:
            from pathlib import Path
            import json
            cfg_path = Path(__file__).resolve().parents[3] / "configs" / "indicators.json"
            if cfg_path.exists():
                try:
                    with cfg_path.open("r", encoding="utf-8") as fh:
                        raw = json.load(fh)
                        try:
                            self._indicator_cfg = self._normalize_indicator_cfg(raw)
                            try:
                                logger.info(f"Loaded normalized indicator config from {cfg_path}")
                            except Exception:
                                logger.info("Loaded indicator config")
                        except Exception:
                            self._indicator_cfg = raw
                except Exception:
                    self._indicator_cfg = None
            else:
                self._indicator_cfg = None
        except Exception:
            self._indicator_cfg = None

        # Interaction state for panning
        self._is_panning = False
        self._pan_start = None  # (xpress, ypress)
        # connect mouse events via helper (idempotent)
        try:
            self._connect_mpl_events()
        except Exception:
            pass

        # Note: ChartTab no longer subscribes directly to event_bus here to avoid
        # cross-thread Qt calls and duplicate subscriptions when an EventBridge
        # is registered in setup_ui. EventBridge.tickReceived will be connected
        # to chart_tab._on_tick_event in the application setup.
        try:
            logger.debug("ChartTab will receive ticks via EventBridge (if present); not subscribing directly to event_bus")
        except Exception:
            pass

        # start a DB poll timer to detect external inserts (fallback when websocket not available)
        try:
            self._last_polled_ts = None
            self._poll_timer = QTimer(self)
            self._poll_timer.setInterval(1000)  # ms
            self._poll_timer.timeout.connect(self._poll_latest_tick)
            self._poll_timer.start()
            try:
                logger.debug("ChartTab poll timer started interval=%sms timer_id=%s", self._poll_timer.interval(), id(self._poll_timer))
            except Exception:
                logger.debug("ChartTab poll timer started")
        except Exception as e:
            logger.debug("ChartTab failed to start poll timer: {}", e)

    def set_symbol_timeframe(self, db_service, symbol: str, timeframe: str):
        self.db_service = db_service
        self.symbol = symbol
        self.timeframe = timeframe

    def _poll_latest_tick(self):
        """
        Poll DB for latest market_data_candles row for current symbol/timeframe.
        If a newer ts_utc found compared to self._last_polled_ts, append to _last_df and redraw.
        This enables updates when external processes insert into the DB.
        """
        try:
            # diagnostic trace: show invocation and key state
            try:
                logger.debug(f"ChartTab._poll_latest_tick invoked last_polled_ts={getattr(self, '_last_polled_ts', None)} db_service_present={getattr(self, 'db_service', None) is not None} symbol={getattr(self, 'symbol', None)} timeframe={getattr(self, 'timeframe', None)}")
            except Exception:
                logger.debug("ChartTab._poll_latest_tick invoked")

            if getattr(self, "db_service", None) is None or getattr(self, "symbol", None) is None:
                try:
                    logger.debug("ChartTab._poll_latest_tick early exit: db_service or symbol not configured")
                except Exception:
                    pass
                return
            meta = MetaData()
            try:
                meta.reflect(bind=self.db_service.engine, only=["market_data_candles"])
            except Exception:
                try:
                    logger.debug("ChartTab._poll_latest_tick: metadata reflect failed")
                except Exception:
                    pass
                return
            tbl = meta.tables.get("market_data_candles")
            if tbl is None:
                return
            with self.db_service.engine.connect() as conn:
                stmt = select(tbl).where(tbl.c.symbol == self.symbol).where(tbl.c.timeframe == self.timeframe).order_by(tbl.c.ts_utc.desc()).limit(1)
                row = conn.execute(stmt).fetchone()
                if not row:
                    return
                # normalize row to dict robustly (support Row._mapping and Row.keys())
                try:
                    if hasattr(row, "_mapping"):
                        rec = dict(row._mapping)
                    elif hasattr(row, "keys"):
                        rec = {k: row[k] for k in row.keys()}
                    else:
                        rec = dict(row)
                except Exception:
                    # last resort: try to coerce via tuple->dict if possible
                    try:
                        rec = dict(row._asdict()) if hasattr(row, "_asdict") else dict(row)
                    except Exception:
                        # give up and skip
                        logger.debug("ChartTab _poll_latest_tick: cannot convert row to dict, skipping")
                        return
                ts = int(rec.get("ts_utc", 0))
                if self._last_polled_ts is None or ts > int(self._last_polled_ts):
                    self._last_polled_ts = ts
                    # append into _last_df
                    try:
                        import pandas as pd
                        df_row = pd.DataFrame([rec])
                        if getattr(self, "_last_df", None) is None or self._last_df.empty:
                            self._last_df = df_row
                        else:
                            df_combined = pd.concat([self._last_df, df_row], ignore_index=True)
                            df_combined = df_combined.drop_duplicates(subset=["ts_utc"], keep="last").sort_values("ts_utc").reset_index(drop=True)
                            self._last_df = df_combined
                    except Exception:
                        pass
                    # update bid/ask label if present in record
                    try:
                        bid = rec.get("bid", None)
                        ask = rec.get("ask", None)
                        price = rec.get("close", rec.get("price", None))
                        if getattr(self, "bidask_label", None) is not None:
                            try:
                                bv = bid if bid is not None else price
                                av = ask if ask is not None else price
                                if bv is not None and av is not None:
                                    self.bidask_label.setText(f"Bid: {float(bv):.5f}    Ask: {float(av):.5f}")
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # redraw chart with new buffer
                    try:
                        self.update_plot(self._last_df, timeframe=self.timeframe)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("ChartTab _poll_latest_tick exception: {}", e)

    # --- Mouse interaction handlers (wheel zoom on x axis, left-drag pan)
    def _on_scroll(self, event):
        try:
            if event.inaxes != self.ax:
                return
            # event.xdata is in matplotlib date floating point (if plotted with dates)
            import matplotlib.dates as mdates
            xdata = event.xdata
            if xdata is None:
                return
            cur_xlim = self.ax.get_xlim()
            # scale factor: zoom in/out by 20% per scroll step
            base_scale = 1.2
            if event.button == "up":
                scale_factor = 1 / base_scale
            elif event.button == "down":
                scale_factor = base_scale
            else:
                scale_factor = 1.0
            left = xdata - (xdata - cur_xlim[0]) * scale_factor
            right = xdata + (cur_xlim[1] - xdata) * scale_factor
            self.ax.set_xlim(left, right)
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_button_press(self, event):
        try:
            if event.inaxes != self.ax:
                return
            # left button = pan
            if event.button == 1:
                self._is_panning = True
                self._pan_start = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())
        except Exception:
            pass

    def _on_motion(self, event):
        try:
            # pan while left button pressed
            if not self._is_panning or self._pan_start is None:
                return
            if event.inaxes != self.ax:
                return
            xpress, ypress, (x0, x1), (y0, y1) = self._pan_start
            dx = event.x - xpress
            # compute shift in data coordinates using axis transform
            inv = self.ax.transData.inverted()
            p1 = inv.transform((xpress, 0))
            p2 = inv.transform((event.x, 0))
            dx_data = p2[0] - p1[0]
            self.ax.set_xlim(x0 - dx_data, x1 - dx_data)
            self.canvas.draw_idle()
        except Exception as e:
            logger.debug("ChartTab _on_motion exception: {}", e)

    def _on_button_release(self, event):
        try:
            if event.button == 1:
                self._is_panning = False
                self._pan_start = None
        except Exception:
            pass

    def _tick_noop(self, payload):
        """Fallback noop tick handler (UI-thread); logs payload for diagnostics."""
        try:
            logger.debug("Tick received but handler not ready: {}", payload)
            # attempt to update bid/ask label if payload contains simple fields
            try:
                bid = payload.get("bid") if isinstance(payload, dict) else None
                ask = payload.get("ask") if isinstance(payload, dict) else None
                if getattr(self, "bidask_label", None) is not None and bid is not None and ask is not None:
                    try:
                        self.bidask_label.setText(f"Bid: {float(bid):.5f}    Ask: {float(ask):.5f}")
                    except Exception:
                        pass
            except Exception:
                pass

        except Exception:
            pass

    # Public convenience methods (aliases/wrappers) to avoid AttributeError when other modules call them
    def _normalize_indicator_cfg(self, raw: dict | None) -> dict:
        """
        Normalize raw config (from dialog or old format) into canonical structure:
        { "sma": {"enabled":bool,"window":int,"color":str}, ... }
        """
        # defaults
        defaults = {
            "sma": {"enabled": False, "window": 20, "color": "#1f77b4"},
            "ema": {"enabled": False, "span": 20, "color": "#ff7f0e"},
            "bollinger": {"enabled": False, "window": 20, "n_std": 2.0, "color_up": "#2ca02c", "color_low": "#d62728"},
            "rsi": {"enabled": False, "period": 14, "color": "#9467bd"},
            "macd": {"enabled": False, "fast": 12, "slow": 26, "signal": 9, "color": "#ff7f0e"},
        }
        out = {}
        try:
            if not isinstance(raw, dict):
                return defaults
            for k, v in defaults.items():
                rv = raw.get(k)
                if isinstance(rv, dict):
                    # if 'enabled' present use it, else presence = enabled True
                    enabled = bool(rv.get("enabled", True)) if rv else False
                    if k == "sma":
                        out[k] = {"enabled": enabled, "window": int(rv.get("window", v["window"])), "color": rv.get("color", v["color"])}
                    elif k == "ema":
                        out[k] = {"enabled": enabled, "span": int(rv.get("span", v["span"])), "color": rv.get("color", v["color"])}
                    elif k == "bollinger":
                        nstd = rv.get("n_std", rv.get("k", v["n_std"]))
                        try:
                            nstd = float(nstd)
                        except Exception:
                            nstd = v["n_std"]
                        out[k] = {"enabled": enabled, "window": int(rv.get("window", v["window"])), "n_std": nstd, "color_up": rv.get("color_up", v["color_up"]), "color_low": rv.get("color_low", v["color_low"])}
                    elif k == "rsi":
                        out[k] = {"enabled": enabled, "period": int(rv.get("period", v["period"])), "color": rv.get("color", v["color"])}
                    elif k == "macd":
                        out[k] = {"enabled": enabled, "fast": int(rv.get("fast", v["fast"])), "slow": int(rv.get("slow", v["slow"])), "signal": int(rv.get("signal", v["signal"])), "color": rv.get("color", v["color"])}
                else:
                    # missing -> use defaults
                    out[k] = v
        except Exception:
            return defaults
        return out

    def apply_indicators(self, cfg: dict):
        """
        Public wrapper to apply indicators (safe entry point). Normalizes cfg and calls internal implementation.
        """
        try:
            norm = self._normalize_indicator_cfg(cfg)
            # store canonical config
            self._indicator_cfg = norm
            # call internal implementation
            try:
                self._apply_indicators(norm)
            except Exception as e:
                logger.exception("apply_indicators internal failure: {}", e)
        except Exception as e:
            logger.exception("apply_indicators failed: {}", e)

    def _connect_mpl_events(self):
        """
        Disconnect previous canvas cids (if any) and connect canvas mouse events for zoom & pan.
        Stores connection ids in self._mpl_cids so we can clean them up on reconnect.
        """
        try:
            # disconnect existing connections first
            try:
                if getattr(self, "_mpl_cids", None):
                    for name, cid in list(self._mpl_cids.items()):
                        try:
                            if cid is not None:
                                self.canvas.mpl_disconnect(cid)
                        except Exception:
                            pass
            except Exception:
                pass
            # reset registry
            self._mpl_cids = {}
            # connect events
            try:
                self._mpl_cids["scroll"] = self.canvas.mpl_connect("scroll_event", self._on_scroll)
            except Exception:
                self._mpl_cids["scroll"] = None
            try:
                self._mpl_cids["button_press"] = self.canvas.mpl_connect("button_press_event", self._on_button_press)
            except Exception:
                self._mpl_cids["button_press"] = None
            try:
                self._mpl_cids["motion"] = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
            except Exception:
                self._mpl_cids["motion"] = None
            try:
                self._mpl_cids["button_release"] = self.canvas.mpl_connect("button_release_event", self._on_button_release)
            except Exception:
                self._mpl_cids["button_release"] = None
            # xlim_changed handler: ensure connected once (we keep flag)
            try:
                if not getattr(self, "_xlim_connected", False):
                    cid = self.canvas.mpl_connect("xlim_changed", self._on_xlim_changed)
                    self._mpl_cids["xlim_changed"] = cid
                    self._xlim_connected = True
            except Exception:
                pass
            # diagnostic: log connected cids
            try:
                logger.debug(f"ChartTab _connect_mpl_events: cids={self._mpl_cids}")
            except Exception:
                pass
        except Exception:
            pass

    def redraw(self):
        """
        Public redraw helper: replot last_df if available.
        """
        if getattr(self, "_last_df", None) is not None:
            self.update_plot(self._last_df, timeframe=getattr(self, "timeframe", None))

    def showEvent(self, event):
        """
        When the Chart tab is shown, ensure the chart is populated automatically
        by loading recent historical rows from DB (if not already loaded).
        """
        try:
            # only act on first show when we have db_service and symbol/timeframe
            if getattr(self, "_last_df", None) is None and getattr(self, "db_service", None) is not None and getattr(self, "symbol", None):
                try:
                    from sqlalchemy import select
                    meta = MetaData()
                    meta.reflect(bind=self.db_service.engine, only=["market_data_candles"])
                    tbl = meta.tables.get("market_data_candles")
                    if tbl is not None:
                        with self.db_service.engine.connect() as conn:
                            stmt = select(tbl).where(tbl.c.symbol == self.symbol).where(tbl.c.timeframe == self.timeframe).order_by(tbl.c.ts_utc.desc()).limit(500)
                            rows = conn.execute(stmt).fetchall()
                            if rows:
                                import pandas as pd
                                df = pd.DataFrame([dict(r) for r in rows[::-1]])  # ascending
                                try:
                                    self._last_df = df
                                    self.update_plot(df, timeframe=self.timeframe)
                                except Exception:
                                    pass
                except Exception as e:
                    logger.debug("showEvent failed to load initial data: {}", e)
        except Exception:
            pass
        # call base
        try:
            super().showEvent(event)
        except Exception:
            pass

    def _apply_indicators(self, cfg: dict):
        """
        Internal: remove previous indicator artists and overlay configured indicators.
        Does NOT redraw base 'close' line (update_plot is responsible for that).
        Reuses auxiliary axes for indicators requiring separate y-axis (rsi, macd).
        Only indicators with 'enabled': True are drawn; disabled ones are removed.
        """
        try:
            if self._last_df is None or self._last_df.empty:
                return
            import pandas as pd
            df = self._last_df.copy()
            if "ts_utc" not in df.columns or "close" not in df.columns:
                return
            df = df.sort_values("ts_utc").reset_index(drop=True)
            x_dt = pd.to_datetime(df["ts_utc"].astype("int64"), unit="ms", utc=True).dt.tz_convert(None)
            try:
                x_num = mdates.date2num(x_dt.to_pydatetime() if hasattr(x_dt, "to_pydatetime") else pd.to_datetime(x_dt).to_pydatetime())
            except Exception:
                try:
                    x_num = [mdates.date2num(d) for d in list(x_dt)]
                except Exception:
                    x_num = list(range(len(df)))

            # normalize cfg if needed (ensure canonical structure)
            try:
                norm = self._normalize_indicator_cfg(cfg)
            except Exception:
                norm = cfg

            # remove existing indicator artists and clear aux axes if present
            try:
                for name, artists in list(self._indicator_artists.items()):
                    for art in artists:
                        try:
                            art.remove()
                        except Exception:
                            pass
                self._indicator_artists = {}
                # clear aux axes lines
                for k, ax_aux in list(self._aux_axes.items()):
                    try:
                        ax_aux.cla()
                    except Exception:
                        pass
            except Exception:
                pass

            def _get_color(entry, default):
                try:
                    if isinstance(entry, dict):
                        return entry.get("color") or entry.get("color_up") or entry.get("color_low") or default
                    return default
                except Exception:
                    return default

            def _register(name: str, arts):
                try:
                    if name not in self._indicator_artists:
                        self._indicator_artists[name] = []
                    self._indicator_artists[name].extend(arts if isinstance(arts, list) else [arts])
                except Exception:
                    pass

            # SMA
            if norm.get("sma", {}).get("enabled"):
                try:
                    w = int(norm["sma"]["window"])
                    col = _get_color(norm["sma"], "#1f77b4")
                    s = sma(df["close"], w)
                    ln, = self.ax.plot(x_num, s, label=f"SMA({w})", color=col, alpha=0.9)
                    _register(f"sma_{w}_{col}", ln)
                except Exception:
                    pass

            # EMA
            if norm.get("ema", {}).get("enabled"):
                try:
                    sp = int(norm["ema"]["span"])
                    col = _get_color(norm["ema"], "#ff7f0e")
                    e = ema(df["close"], sp)
                    ln, = self.ax.plot(x_num, e, label=f"EMA({sp})", color=col, alpha=0.9)
                    _register(f"ema_{sp}_{col}", ln)
                except Exception:
                    pass

            # Bollinger
            if norm.get("bollinger", {}).get("enabled"):
                try:
                    w = int(norm["bollinger"]["window"])
                    nstd = float(norm["bollinger"]["n_std"])
                    col_up = norm["bollinger"].get("color_up") or _get_color(norm.get("bollinger"), "#2ca02c")
                    col_low = norm["bollinger"].get("color_low") or _get_color(norm.get("bollinger"), "#d62728")
                    up, low = bollinger(df["close"], window=w, n_std=nstd)
                    l1, = self.ax.plot(x_num, up, label=f"BB_up({w},{nstd})", color=col_up, alpha=0.6)
                    l2, = self.ax.plot(x_num, low, label=f"BB_low({w},{nstd})", color=col_low, alpha=0.6)
                    _register(f"bollinger_{w}_{nstd}", [l1, l2])
                except Exception:
                    pass

            # RSI (reuse/create aux axis)
            if norm.get("rsi", {}).get("enabled"):
                try:
                    p = int(norm["rsi"]["period"])
                    col = _get_color(norm["rsi"], "#9467bd")
                    r = rsi(df["close"], period=p)
                    ax_rsi = self._aux_axes.get("rsi")
                    if ax_rsi is None:
                        ax_rsi = self.ax.twinx()
                        try:
                            ax_rsi.spines["right"].set_position(("axes", 1.05))
                        except Exception:
                            pass
                        self._aux_axes["rsi"] = ax_rsi
                    ln, = ax_rsi.plot(x_num, r, label=f"RSI({p})", color=col, alpha=0.8)
                    ax_rsi.set_ylabel("RSI")
                    _register(f"rsi_{p}_{col}", ln)
                except Exception:
                    pass

            # MACD (reuse/create aux axis)
            if norm.get("macd", {}).get("enabled"):
                try:
                    f = int(norm["macd"]["fast"])
                    s = int(norm["macd"]["slow"])
                    sg = int(norm["macd"]["signal"])
                    col = _get_color(norm["macd"], "#ff7f0e")
                    m = macd(df["close"], fast=f, slow=s, signal=sg)
                    ax_macd = self._aux_axes.get("macd")
                    if ax_macd is None:
                        ax_macd = self.ax.twinx()
                        try:
                            ax_macd.spines["right"].set_position(("axes", 1.10))
                        except Exception:
                            pass
                        self._aux_axes["macd"] = ax_macd
                    ln1, = ax_macd.plot(x_num, m["macd"], label=f"MACD({f},{s})", color=col, alpha=0.9)
                    ln2, = ax_macd.plot(x_num, m["signal"], label=f"Signal({sg})", color="#1f77b4", alpha=0.6)
                    _register(f"macd_{f}_{s}_{sg}", [ln1, ln2])
                except Exception:
                    pass

            # refresh legends: combine base and aux handles
            try:
                handles, labels = self.ax.get_legend_handles_labels()
                # include aux axes legends
                for name, ax_aux in self._aux_axes.items():
                    try:
                        h, l = ax_aux.get_legend_handles_labels()
                        handles += h
                        labels += l
                    except Exception:
                        pass
                if handles:
                    self.ax.legend(handles, labels, loc="upper left", fontsize="small")
            except Exception:
                pass

            self.canvas.draw()
        except Exception as e:
            logger.exception("Internal _apply_indicators failed: {}", e)

    def update_plot(self, df: pd.DataFrame, timeframe: Optional[str] = None):
        """
        Public plotting entrypoint: draw base 'close' line and reapply indicators.
        Uses matplotlib numeric dates for X to ensure live updates work reliably.
        """
        try:
            if df is None or getattr(df, "empty", True):
                try:
                    self.ax.set_title("No data")
                    self.canvas.draw_idle()
                except Exception:
                    pass
                return

            # ensure ts_utc column exists and convert to datetimes (local tz)
            try:
                x_dt = pd.to_datetime(df["ts_utc"].astype("int64"), unit="ms", utc=True)
                x_dt = x_dt.dt.tz_convert(None)
            except Exception:
                x_dt = pd.to_datetime(df.get("ts_utc", df.index))
            try:
                y = df["close"].astype(float)
            except Exception:
                try:
                    y = df.iloc[:, -1].astype(float)
                except Exception:
                    y = pd.Series([0.0] * len(x_dt))

            # convert to matplotlib numeric date values
            try:
                x_num = mdates.date2num(x_dt.to_pydatetime() if hasattr(x_dt, "to_pydatetime") else pd.to_datetime(x_dt).to_pydatetime())
            except Exception:
                # fallback: try per-value conversion
                try:
                    x_num = [mdates.date2num(d) for d in list(x_dt)]
                except Exception:
                    x_num = list(range(len(y)))

            # If base line exists, update its data instead of clearing axes (preserve toolbar/events)
            try:
                if getattr(self, "_base_line", None) is not None:
                    try:
                        self._base_line.set_xdata(x_num)
                        self._base_line.set_ydata(y)
                    except Exception:
                        try:
                            self._base_line.remove()
                        except Exception:
                            pass
                        self._base_line, = self.ax.plot(x_num, y, "-", color="black", linewidth=1.0, label="close")
                else:
                    self._base_line, = self.ax.plot(x_num, y, "-", color="black", linewidth=1.0, label="close")
            except Exception:
                try:
                    self._base_line, = self.ax.plot(x_num, y, "-", color="black", linewidth=1.0, label="close")
                except Exception:
                    pass

            self.ax.set_title("Historical close")
            # set date formatter and locator (works with numeric dates)
            try:
                self.ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
                self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            except Exception:
                pass
            try:
                self.ax.relim()
                self.ax.autoscale_view()
            except Exception:
                pass

            # non-blocking draw
            try:
                self.canvas.draw_idle()
            except Exception:
                try:
                    self.canvas.draw()
                except Exception:
                    pass

            # update buffer
            self._last_df = df

            # update last polled ts so DB poller can detect new inserts (if running)
            try:
                if getattr(self, "_last_polled_ts", None) is None:
                    try:
                        self._last_polled_ts = int(self._last_df["ts_utc"].max())
                    except Exception:
                        self._last_polled_ts = None
                else:
                    try:
                        max_ts = int(self._last_df["ts_utc"].max())
                        if max_ts and (self._last_polled_ts is None or max_ts > self._last_polled_ts):
                            self._last_polled_ts = max_ts
                    except Exception:
                        pass
            except Exception:
                pass

            # diagnostic log
            try:
                tail_preview = df.tail(3).to_dict(orient="records") if not getattr(df, "empty", True) else []
                logger.debug("update_plot: plotted df shape=%s tail=%s last_polled_ts=%s", getattr(df, "shape", None), tail_preview, getattr(self, "_last_polled_ts", None))
            except Exception:
                logger.debug("update_plot: plotted df (could not stringify tail)")

            # connect xlim_changed handler once
            try:
                if not getattr(self, "_xlim_connected", False):
                    self.canvas.mpl_connect("xlim_changed", self._on_xlim_changed)
                    self._xlim_connected = True
            except Exception:
                pass
            # ensure mouse events are connected (idempotent)
            try:
                self._connect_mpl_events()
            except Exception:
                pass

            # apply indicators if configured
            try:
                if getattr(self, "_indicator_cfg", None):
                    try:
                        # use public wrapper which normalizes and calls internal impl
                        self.apply_indicators(self._indicator_cfg)
                    except Exception:
                        # best-effort: try internal
                        try:
                            self._apply_indicators(self._indicator_cfg)
                        except Exception:
                            pass
            except Exception:
                pass

        except Exception as e:
            try:
                self.ax.clear()
                self.ax.set_title(f"Plot error: {e}")
                self.canvas.draw()
            except Exception:
                pass

    def _on_tick_event(self, payload):
        """Schedule handling of incoming tick on UI thread."""
        try:
            QTimer.singleShot(0, lambda p=payload: self._handle_tick(p))
        except Exception as e:
            logger.debug("Failed to schedule tick handling: {}", e)

    def _handle_tick(self, payload):
        """
        UI-thread: accept payload (dict-like) and append to last_df, update bid/ask label and redraw.
        Accept ticks even when payload lacks symbol/timeframe; ignore only if payload explicitly targets different symbol/timeframe.
        """
        try:
            logger.info("ChartTab._handle_tick invoked with payload incoming")
            # Normalize payload to dict if possible
            if not isinstance(payload, dict):
                try:
                    payload = dict(payload)
                except Exception:
                    payload = {"price": payload}
            sym = payload.get("symbol", None)
            tf = payload.get("timeframe", None)
            try:
                logger.info(f"ChartTab._handle_tick normalized payload: {payload}")
            except Exception:
                pass

            # If this ChartTab has a target symbol/timeframe and the payload explicitly targets another, ignore.
            if getattr(self, "symbol", None) is not None and sym is not None and sym != self.symbol:
                return
            if getattr(self, "timeframe", None) is not None and tf is not None and tf != self.timeframe:
                return

            import pandas as pd
            ts = int(payload.get("ts_utc", int(pd.Timestamp.utcnow().value // 1_000_000)))
            price = float(payload.get("price", 0.0))
            bid = payload.get("bid", None)
            ask = payload.get("ask", None)

            # Append/update in-memory candle/tick buffer
            row = {"ts_utc": int(ts), "open": price, "high": price, "low": price, "close": price, "volume": None}
            if getattr(self, "_last_df", None) is None or self._last_df.empty:
                self._last_df = pd.DataFrame([row])
            else:
                df_append = pd.DataFrame([row])
                df_combined = pd.concat([self._last_df, df_append], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=["ts_utc"], keep="last").sort_values("ts_utc").reset_index(drop=True)
                self._last_df = df_combined

            # update bid/ask label (prefer explicit bid/ask, fallback to price)
            try:
                if getattr(self, "bidask_label", None) is not None:
                    bv = bid if bid is not None else price
                    av = ask if ask is not None else price
                    try:
                        self.bidask_label.setText(f"Bid: {float(bv):.5f}    Ask: {float(av):.5f}")
                    except Exception:
                        # fallback to simple repr
                        self.bidask_label.setText(f"Bid: {bv} Ask: {av}")
            except Exception:
                pass

            # redraw efficiently: update_plot will draw base and reapply indicators
            try:
                try:
                    prev_len = len(self._last_df) if getattr(self, "_last_df", None) is not None else 0
                except Exception:
                    prev_len = None
                logger.debug("handle_tick: updating plot, prev_len=%s", prev_len)
                self.update_plot(self._last_df, timeframe=getattr(self, "timeframe", None))
                try:
                    # request idle redraw (non-blocking)
                    self.canvas.draw_idle()
                except Exception:
                    try:
                        self.canvas.draw()
                    except Exception:
                        pass
                try:
                    logger.debug("handle_tick: plot update requested, new_len=%s", len(self._last_df) if getattr(self, "_last_df", None) is not None else 0)
                except Exception:
                    pass
            except Exception as e:
                logger.debug("Failed to update plot after tick: {}", e)

            # UI-visible notification in main window status bar (short-lived)
            try:
                mw = getattr(self, "_main_window", None)
                if mw is not None and hasattr(mw, "statusBar"):
                    try:
                        sb = mw.statusBar()
                        if sb is not None:
                            label = sym if (sym is not None) else getattr(self, "symbol", "unknown")
                            try:
                                sb.showMessage(f"Tick {label} {float(price):.5f}", 3000)
                            except Exception:
                                sb.showMessage(f"Tick {label}", 3000)
                    except Exception:
                        pass
            except Exception:
                pass

            logger.debug("Handled tick payload: sym=%s tf=%s price=%s bid=%s ask=%s", sym, tf, price, bid, ask)
        except Exception as e:
            logger.exception("Error handling tick payload: {}", e)

    def _open_indicators_dialog(self):
        """
        Open indicators dialog (packaged if available) or fallback expanded dialog.
        Fallback includes SMA, EMA, Bollinger, RSI, MACD with color selectors.
        """
        cfg = None
        try:
            from .indicators_dialog import IndicatorsDialog  # type: ignore
            dlg = IndicatorsDialog(parent=self, initial=self._indicator_cfg if getattr(self, "_indicator_cfg", None) else None)
            if not dlg.exec():
                return
            cfg = dlg.result()
        except Exception as e:
            logger.debug("Packaged IndicatorsDialog not available or failed: {}", e)
            # build richer fallback dialog inline (SMA, EMA, Bollinger, RSI, MACD)
            try:
                from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QDialogButtonBox, QCheckBox, QLabel, QSpinBox, QLineEdit, QComboBox
                class FallbackDialog(QDialog):
                    def __init__(self, parent=None, initial=None):
                        super().__init__(parent)
                        self.setWindowTitle("Indicatori (fallback)")
                        self.layout = QVBoxLayout(self)
                        palette = [("blue","#1f77b4"),("red","#d62728"),("green","#2ca02c"),("orange","#ff7f0e"),("purple","#9467bd"),("black","#000000")]

                        # SMA
                        row = QHBoxLayout()
                        self.sma_cb = QCheckBox("SMA")
                        self.sma_w = QSpinBox(); self.sma_w.setRange(1,500); self.sma_w.setValue(20)
                        self.sma_col = QComboBox()
                        for n,h in palette: self.sma_col.addItem(f"{n} ({h})", h)
                        row.addWidget(self.sma_cb); row.addWidget(QLabel("w:")); row.addWidget(self.sma_w); row.addWidget(QLabel("color:")); row.addWidget(self.sma_col)
                        self.layout.addLayout(row)

                        # EMA
                        row = QHBoxLayout()
                        self.ema_cb = QCheckBox("EMA")
                        self.ema_span = QSpinBox(); self.ema_span.setRange(1,500); self.ema_span.setValue(20)
                        self.ema_col = QComboBox()
                        for n,h in palette: self.ema_col.addItem(f"{n} ({h})", h)
                        row.addWidget(self.ema_cb); row.addWidget(QLabel("span:")); row.addWidget(self.ema_span); row.addWidget(QLabel("color:")); row.addWidget(self.ema_col)
                        self.layout.addLayout(row)

                        # Bollinger
                        row = QHBoxLayout()
                        self.bb_cb = QCheckBox("Bollinger")
                        self.bb_n = QSpinBox(); self.bb_n.setRange(1,500); self.bb_n.setValue(20)
                        self.bb_k = QLineEdit("2.0")
                        self.bb_col_u = QComboBox(); self.bb_col_l = QComboBox()
                        for n,h in palette:
                            self.bb_col_u.addItem(f"{n} ({h})", h)
                            self.bb_col_l.addItem(f"{n} ({h})", h)
                        row.addWidget(self.bb_cb); row.addWidget(QLabel("n:")); row.addWidget(self.bb_n); row.addWidget(QLabel("k:")); row.addWidget(self.bb_k)
                        row.addWidget(QLabel("up col:")); row.addWidget(self.bb_col_u); row.addWidget(QLabel("low col:")); row.addWidget(self.bb_col_l)
                        self.layout.addLayout(row)

                        # RSI
                        row = QHBoxLayout()
                        self.rsi_cb = QCheckBox("RSI")
                        self.rsi_p = QSpinBox(); self.rsi_p.setRange(1,500); self.rsi_p.setValue(14)
                        self.rsi_col = QComboBox()
                        for n,h in palette: self.rsi_col.addItem(f"{n} ({h})", h)
                        row.addWidget(self.rsi_cb); row.addWidget(QLabel("period:")); row.addWidget(self.rsi_p); row.addWidget(QLabel("color:")); row.addWidget(self.rsi_col)
                        self.layout.addLayout(row)

                        # MACD
                        row = QHBoxLayout()
                        self.macd_cb = QCheckBox("MACD")
                        self.macd_fast = QSpinBox(); self.macd_fast.setRange(1,200); self.macd_fast.setValue(12)
                        self.macd_slow = QSpinBox(); self.macd_slow.setRange(1,500); self.macd_slow.setValue(26)
                        self.macd_sig = QSpinBox(); self.macd_sig.setRange(1,200); self.macd_sig.setValue(9)
                        self.macd_col = QComboBox()
                        for n,h in palette: self.macd_col.addItem(f"{n} ({h})", h)
                        row.addWidget(self.macd_cb); row.addWidget(QLabel("fast:")); row.addWidget(self.macd_fast)
                        row.addWidget(QLabel("slow:")); row.addWidget(self.macd_slow); row.addWidget(QLabel("sig:")); row.addWidget(self.macd_sig)
                        row.addWidget(QLabel("color:")); row.addWidget(self.macd_col)
                        self.layout.addLayout(row)

                        # Buttons
                        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
                        self.layout.addWidget(bb)

                        # load initial if provided (best-effort) - populate all controls including Bollinger, RSI, MACD
                        if initial:
                            try:
                                # SMA
                                if "sma" in initial and isinstance(initial["sma"], dict):
                                    try:
                                        ena = bool(initial["sma"].get("enabled", False))
                                        self.sma_cb.setChecked(ena)
                                        if "window" in initial["sma"]:
                                            self.sma_w.setValue(int(initial["sma"]["window"]))
                                        if "color" in initial["sma"]:
                                            col = initial["sma"]["color"]
                                            for idx in range(self.sma_col.count()):
                                                if self.sma_col.itemData(idx) == col:
                                                    self.sma_col.setCurrentIndex(idx)
                                                    break
                                    except Exception:
                                        pass
                                # EMA
                                if "ema" in initial and isinstance(initial["ema"], dict):
                                    try:
                                        ena = bool(initial["ema"].get("enabled", False))
                                        self.ema_cb.setChecked(ena)
                                        if "span" in initial["ema"]:
                                            self.ema_span.setValue(int(initial["ema"]["span"]))
                                        if "color" in initial["ema"]:
                                            col = initial["ema"]["color"]
                                            for idx in range(self.ema_col.count()):
                                                if self.ema_col.itemData(idx) == col:
                                                    self.ema_col.setCurrentIndex(idx)
                                                    break
                                    except Exception:
                                        pass
                                # Bollinger
                                if "bollinger" in initial and isinstance(initial["bollinger"], dict):
                                    try:
                                        ena = bool(initial["bollinger"].get("enabled", False))
                                        self.bb_cb.setChecked(ena)
                                        if "window" in initial["bollinger"]:
                                            self.bb_n.setValue(int(initial["bollinger"]["window"]))
                                        if "n_std" in initial["bollinger"]:
                                            self.bb_k.setText(str(initial["bollinger"]["n_std"]))
                                        if "color_up" in initial["bollinger"]:
                                            col = initial["bollinger"]["color_up"]
                                            for idx in range(self.bb_col_u.count()):
                                                if self.bb_col_u.itemData(idx) == col:
                                                    self.bb_col_u.setCurrentIndex(idx); break
                                        if "color_low" in initial["bollinger"]:
                                            col = initial["bollinger"]["color_low"]
                                            for idx in range(self.bb_col_l.count()):
                                                if self.bb_col_l.itemData(idx) == col:
                                                    self.bb_col_l.setCurrentIndex(idx); break
                                    except Exception:
                                        pass
                                # RSI
                                if "rsi" in initial and isinstance(initial["rsi"], dict):
                                    try:
                                        ena = bool(initial["rsi"].get("enabled", False))
                                        self.rsi_cb.setChecked(ena)
                                        if "period" in initial["rsi"]:
                                            self.rsi_p.setValue(int(initial["rsi"]["period"]))
                                        if "color" in initial["rsi"]:
                                            col = initial["rsi"]["color"]
                                            for idx in range(self.rsi_col.count()):
                                                if self.rsi_col.itemData(idx) == col:
                                                    self.rsi_col.setCurrentIndex(idx); break
                                    except Exception:
                                        pass
                                # MACD
                                if "macd" in initial and isinstance(initial["macd"], dict):
                                    try:
                                        ena = bool(initial["macd"].get("enabled", False))
                                        self.macd_cb.setChecked(ena)
                                        if "fast" in initial["macd"]:
                                            self.macd_fast.setValue(int(initial["macd"]["fast"]))
                                        if "slow" in initial["macd"]:
                                            self.macd_slow.setValue(int(initial["macd"]["slow"]))
                                        if "signal" in initial["macd"]:
                                            self.macd_sig.setValue(int(initial["macd"]["signal"]))
                                        if "color" in initial["macd"]:
                                            col = initial["macd"]["color"]
                                            for idx in range(self.macd_col.count()):
                                                if self.macd_col.itemData(idx) == col:
                                                    self.macd_col.setCurrentIndex(idx); break
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                    def result(self):
                        """
                        Return a complete config dict with enabled flags and parameters for all indicators.
                        Ensures persistence is stable and the format is consistent across sessions.
                        """
                        out = {}
                        # SMA
                        out["sma"] = {
                            "enabled": bool(self.sma_cb.isChecked()),
                            "window": int(self.sma_w.value()),
                            "color": self.sma_col.currentData()
                        }
                        # EMA
                        out["ema"] = {
                            "enabled": bool(self.ema_cb.isChecked()),
                            "span": int(self.ema_span.value()),
                            "color": self.ema_col.currentData()
                        }
                        # Bollinger
                        try:
                            nstd = float(self.bb_k.text())
                        except Exception:
                            nstd = 2.0
                        out["bollinger"] = {
                            "enabled": bool(self.bb_cb.isChecked()),
                            "window": int(self.bb_n.value()),
                            "n_std": nstd,
                            "color_up": self.bb_col_u.currentData(),
                            "color_low": self.bb_col_l.currentData()
                        }
                        # RSI
                        out["rsi"] = {
                            "enabled": bool(self.rsi_cb.isChecked()),
                            "period": int(self.rsi_p.value()),
                            "color": self.rsi_col.currentData()
                        }
                        # MACD
                        out["macd"] = {
                            "enabled": bool(self.macd_cb.isChecked()),
                            "fast": int(self.macd_fast.value()),
                            "slow": int(self.macd_slow.value()),
                            "signal": int(self.macd_sig.value()),
                            "color": self.macd_col.currentData()
                        }
                        return out

                dlg = FallbackDialog(parent=self, initial=self._indicator_cfg if getattr(self, "_indicator_cfg", None) else None)
                if not dlg.exec():
                    return
                cfg = dlg.result()
            except Exception as ee:
                logger.exception("Failed to build fallback Indicators dialog: {}", ee)
                try:
                    QMessageBox.warning(self, "Indicators error", f"Cannot open indicators dialog: {ee}")
                except Exception:
                    pass
                return

        # persist (normalized) config and apply
        if not cfg:
            return
        try:
            try:
                norm_cfg = self._normalize_indicator_cfg(cfg)
            except Exception:
                norm_cfg = cfg
            self._indicator_cfg = norm_cfg
            from pathlib import Path
            import json
            cfg_path = Path(__file__).resolve().parents[3] / "configs" / "indicators.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with cfg_path.open("w", encoding="utf-8") as fh:
                json.dump(norm_cfg, fh, indent=2)
            try:
                logger.info(f"Saved indicator config to {cfg_path}")
            except Exception:
                logger.info("Saved indicator config")
        except Exception as e:
            logger.exception("Failed to persist indicators config: {}", e)

        try:
            try:
                # Apply indicators (internal) and force a full redraw from last_df to ensure removals are reflected
                self.apply_indicators(self._indicator_cfg)
            except Exception as e:
                logger.exception("apply_indicators failed: {}", e)
            # Force a redraw using current buffer so changes (including disabling) are reflected
            try:
                if getattr(self, "_last_df", None) is not None:
                    self.update_plot(self._last_df, timeframe=getattr(self, "timeframe", None))
            except Exception as e:
                logger.exception("Failed to force redraw after applying indicators: {}", e)
        except Exception as e:
            logger.exception("Failed to apply indicators after dialog: {}", e)
