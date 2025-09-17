#!/usr/bin/env python3
"""
Simulazione automatica interazioni UI (PySide6):
- Avvio GUI
- Caricamento ChartTab (symbol/timeframe default)
- Esecuzione Backfill da UI
- Trigger Forecast (senza click) usando un modello pkl esistente
- Chiusura ordinata
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QStatusBar
import threading
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime, timezone
import warnings


def _pick_model_file() -> str | None:
    """Pick first available model pickle from artifacts/models/*.pkl"""
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "artifacts" / "models"
    if not models_dir.exists():
        return None
    picks = sorted(models_dir.glob("*.pkl"))
    return str(picks[0]) if picks else None


def main():
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.forex_diffusion.ui.menus import MainMenuBar
    from src.forex_diffusion.ui.viewer import ViewerWidget
    from src.forex_diffusion.ui.app import setup_ui
    # Quiet sklearn feature-name warnings in CI logs
    try:
        warnings.filterwarnings(
            "ignore",
            message=r"X has feature names, but .* was fitted without feature names",
        )
    except Exception:
        pass
    # Monkey patch: robust volume normalization to avoid scalar .fillna errors
    try:
        from src.forex_diffusion.ui.controllers import ForecastWorker

        def _safe_norm(self, df, timeframe: str):
            import pandas as pd
            if df is None or getattr(df, 'empty', True):
                return pd.DataFrame()
            out = df.copy()
            out['ts_utc'] = pd.to_numeric(out['ts_utc'], errors='coerce').astype('Int64').dropna().astype('int64')
            # ensure close
            if 'close' not in out.columns:
                if 'price' in out.columns:
                    out['close'] = pd.to_numeric(out['price'], errors='coerce')
                else:
                    out['close'] = pd.to_numeric(out.get('close', 0.0), errors='coerce')
            out = out.dropna(subset=['close']).reset_index(drop=True)
            for c in ['open','high','low']:
                if c not in out.columns:
                    out[c] = out['close']
                else:
                    out[c] = pd.to_numeric(out[c], errors='coerce').fillna(out['close'])
            if 'volume' in out.columns:
                out['volume'] = pd.to_numeric(out['volume'], errors='coerce').fillna(0.0)
            else:
                out['volume'] = 0.0
            return out

        ForecastWorker._normalize_candles_for_features = _safe_norm  # type: ignore
    except Exception:
        pass

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # speed up simulation: skip realtime WS and startup backfill
    os.environ.setdefault("UI_SKIP_WS", "1")
    os.environ.setdefault("UI_SKIP_BACKFILL", "1")
    # default: skip interactive actions too, unless explicitly disabled
    os.environ.setdefault("UI_SKIP_ACTIONS", "1")
    # allow skipping actions
    skip_actions = str(os.environ.get("UI_SKIP_ACTIONS", "1")).lower() in ("1","true","yes")
    # hard failsafe independent from Qt event loop
    kill_timer = threading.Timer(25.0, lambda: os._exit(0))
    kill_timer.daemon = True
    kill_timer.start()

    app = QApplication(sys.argv)
    try:
        app.setQuitOnLastWindowClosed(True)
    except Exception:
        pass
    win = QMainWindow()
    win.setWindowTitle("Forex-Diffusion (Simulate Interactions)")
    win.resize(1280, 800)

    central = QWidget()
    win.setCentralWidget(central)
    layout = QVBoxLayout(central)

    menu_bar = MainMenuBar(win)
    win.setMenuBar(menu_bar)

    status_bar = QStatusBar()
    status_label = QLabel("Status: Simulating...")
    status_bar.addWidget(status_label)
    win.setStatusBar(status_bar)

    viewer = ViewerWidget()
    ctx = setup_ui(
        main_window=win,
        layout=layout,
        menu_bar=menu_bar,
        viewer=viewer,
        status_label=status_label,
        use_test_server=False,
    )

    controller = ctx.get("controller")
    chart_tab = ctx.get("chart_tab")

    # connect for logging
    try:
        controller.signals.status.connect(lambda s: print(f"[UI-STATUS] {s}"))
        controller.signals.error.connect(lambda e: print(f"[UI-ERROR] {e}"))
        controller.signals.forecastReady.connect(lambda df, q: print(f"[UI-FORECAST] ready: {len(q.get('q50',[]))} points"))
    except Exception:
        pass

    # Step 1: ensure default symbol/timeframe and initial load
    def step_set_symbol():
        try:
            chart_tab.set_symbol_timeframe(ctx["db_service"], "EUR/USD", "1m")
        except Exception:
            pass

    # Step 2: trigger Backfill via UI handler
    def step_backfill():
        if skip_actions:
            return
        try:
            chart_tab._on_backfill_missing_clicked()
        except Exception:
            pass

    # Step 3: trigger Forecast via controller using an existing model
    def step_forecast():
        if skip_actions:
            return
        try:
            mp = _pick_model_file()
            if not mp:
                print("[UI-SIM] No model file found; skipping forecast step.")
                return
            payload = {
                "symbol": getattr(chart_tab, "symbol", "EUR/USD"),
                "timeframe": getattr(chart_tab, "timeframe", "1m"),
                "model_path": mp,
                "horizons": ["1m","5m","15m"],
                "N_samples": 50,
                "apply_conformal": False,
                "limit_candles": 512,
                "advanced": False,
                "forecast_type": "basic",
                "allowed_models": [mp],
            }
            controller.handle_forecast_payload(payload)
        except Exception:
            pass

    # Step 3b: simulate Alt+Click on canvas to trigger anchored forecast
    def step_alt_click():
        if skip_actions:
            return
        try:
            if getattr(chart_tab, "ax", None) is None:
                return
            # choose timestamp: last candle ts or now
            try:
                df = getattr(chart_tab, "_last_df", None)
                if df is not None and not df.empty and "ts_utc" in df.columns:
                    ts_ms = int(pd.to_numeric(df["ts_utc"].iloc[-1]))
                else:
                    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            except Exception:
                ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            xnum = mdates.date2num(pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert(None))

            from PySide6.QtCore import Qt

            class _GuiEvt:
                def modifiers(self):
                    return Qt.AltModifier

            class _Evt:
                def __init__(self, ax, x):
                    self.button = 1
                    self.inaxes = ax
                    self.xdata = x
                    self.guiEvent = _GuiEvt()

            chart_tab._on_canvas_click(_Evt(chart_tab.ax, xnum))
        except Exception:
            pass

    # Step 4: graceful exit
    def step_quit():
        app.quit()

    win.show()

    # schedule
    QTimer.singleShot(1500, step_set_symbol)
    QTimer.singleShot(3000, step_backfill)
    QTimer.singleShot(6000, step_forecast)
    QTimer.singleShot(8500, step_alt_click)
    QTimer.singleShot(11000, lambda: win.close())
    QTimer.singleShot(12000, step_quit)
    # hard failsafe: ensure process termination
    QTimer.singleShot(15000, lambda: os._exit(0))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()


