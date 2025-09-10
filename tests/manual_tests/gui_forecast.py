#!/usr/bin/env python3
"""
tests/manual_tests/gui_forecast.py

Extended Tkinter GUI:
 - loads latest candles from DB
 - computes features via pipeline_process
 - loads saved model (pickle / torch)
 - generates forecast on button click and overlays forecast on the chart (gray 70%)
 - settings dialog allows editing pipeline/model/query params and selecting model file

Usage:
  python tests/manual_tests/gui_forecast.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pickle
import json

ROOT = Path(__file__).resolve().parents[2]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# imports project and plotting libs
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    from sqlalchemy import text
    from forex_diffusion.services.db_service import DBService
    from forex_diffusion.features.pipeline import pipeline_process
except Exception as e:
    print("Missing dependencies or project imports:", e)
    raise

# optional torch
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# default settings
DEFAULT_SETTINGS = {
    "symbol": "EUR/USD",
    "timeframe": "1m",
    "limit_candles": 256,   # default per spec
    "warmup_bars": 16,
    "atr_n": 14,
    "rsi_n": 14,
    "bb_n": 20,
    "hurst_window": 64,
    "rv_window": 60,
    "horizon": 5,
    "model_path": "",  # user selects
    "output_type": "returns",  # or "prices"
}

def timeframe_to_timedelta(tf: str):
    """Convert timeframe like '1m','5m','1h','1d' to pandas Timedelta"""
    tf = str(tf).strip().lower()
    try:
        if tf.endswith("m"):
            return pd.to_timedelta(int(tf[:-1]), unit="m")
        if tf.endswith("h"):
            return pd.to_timedelta(int(tf[:-1]), unit="h")
        if tf.endswith("d"):
            return pd.to_timedelta(int(tf[:-1]), unit="d")
        # fallback minutes
        return pd.to_timedelta(int(tf), unit="m")
    except Exception:
        return pd.to_timedelta(1, unit="m")

class ForecastGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sistema Previsionale - GUI")
        self.settings = DEFAULT_SETTINGS.copy()
        self.db = DBService()
        self.fig = Figure(figsize=(10,5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()
        toolbar.pack()

        # control frame
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill=tk.X, padx=4, pady=4)
        self.btn_refresh = ttk.Button(ctrl, text="Refresh candles", command=self._on_refresh)
        self.btn_refresh.pack(side=tk.LEFT, padx=2)
        self.btn_predict = ttk.Button(ctrl, text="Genera previsione", command=self._on_predict)
        self.btn_predict.pack(side=tk.LEFT, padx=2)
        self.btn_settings = ttk.Button(ctrl, text="Settings previsione", command=self._open_settings)
        self.btn_settings.pack(side=tk.LEFT, padx=2)
        self.lbl_model = ttk.Label(ctrl, text="Model: (none)")
        self.lbl_model.pack(side=tk.LEFT, padx=8)

        # status
        self.status = ttk.Label(self.root, text="Pronto")
        self.status.pack(fill=tk.X, padx=4, pady=2)

        # internal state
        self.candles = pd.DataFrame()
        self.features = pd.DataFrame()
        self.model_payload = None

        # initial load
        self._on_refresh()

    def _set_status(self, text: str):
        try:
            self.status.config(text=text)
            self.root.update_idletasks()
        except Exception:
            pass

    def _on_refresh(self):
        self._set_status("Caricamento candles...")
        t = threading.Thread(target=self._load_and_plot, daemon=True)
        t.start()

    def _load_and_plot(self):
        try:
            limit = int(self.settings.get("limit_candles", 256))
            symbol = self.settings.get("symbol", "EUR/USD")
            tf = self.settings.get("timeframe", "1m")
            with self.db.engine.connect() as conn:
                q = "SELECT ts_utc, open, high, low, close FROM market_data_candles WHERE symbol=:s AND timeframe=:tf ORDER BY ts_utc DESC LIMIT :lim"
                rows = conn.execute(text(q), {"s": symbol, "tf": tf, "lim": limit}).fetchall()
                if not rows:
                    self._set_status("No candles found")
                    return
                rows = rows[::-1]
                df = pd.DataFrame([dict(r._mapping) if hasattr(r, "_mapping") else {"ts_utc": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4]} for r in rows])
            self.candles = df.sort_values("ts_utc").reset_index(drop=True)
            # plot close series
            self.ax.clear()
            dt = pd.to_datetime(self.candles["ts_utc"].astype("int64"), unit="ms", utc=True)
            self.ax.plot(dt, self.candles["close"], color="blue", label="close")
            self.ax.set_title(f"{symbol} {tf} - Ultimi {len(self.candles)} candles")
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            self.canvas.draw()
            self._set_status("Candles caricati")
        except Exception as e:
            self._set_status(f"Errore caricamento candles: {e}")

    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings previsione")
        frm = ttk.Frame(win, padding=8)
        frm.pack(fill=tk.BOTH, expand=1)

        # symbol combobox populated from DB
        ttk.Label(frm, text="symbol").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        symbols = ["EUR/USD"]
        try:
            with self.db.engine.connect() as conn:
                rows = conn.execute(text("SELECT DISTINCT symbol FROM market_data_candles ORDER BY symbol")).fetchall()
                symbols = [r._mapping["symbol"] if hasattr(r, "_mapping") else r[0] for r in rows] or symbols
        except Exception:
            pass
        sym_var = tk.StringVar(value=self.settings.get("symbol"))
        sym_cb = ttk.Combobox(frm, values=symbols, textvariable=sym_var, width=30)
        sym_cb.grid(row=0, column=1, sticky=tk.W, padx=4, pady=2)

        # timeframe combobox
        ttk.Label(frm, text="timeframe").grid(row=1, column=0, sticky=tk.W, padx=4, pady=2)
        tf_values = ["1m","5m","15m","30m","1h","4h","1d"]
        tf_var = tk.StringVar(value=self.settings.get("timeframe"))
        tf_cb = ttk.Combobox(frm, values=tf_values, textvariable=tf_var, width=15)
        tf_cb.grid(row=1, column=1, sticky=tk.W, padx=4, pady=2)

        # limit_candles spinbox
        ttk.Label(frm, text="limit_candles (N)").grid(row=2, column=0, sticky=tk.W, padx=4, pady=2)
        lim_var = tk.IntVar(value=int(self.settings.get("limit_candles", 256)))
        lim_sp = ttk.Spinbox(frm, from_=32, to=5000, textvariable=lim_var, width=10)
        lim_sp.grid(row=2, column=1, sticky=tk.W, padx=4, pady=2)

        # horizon spinbox
        ttk.Label(frm, text="horizon (bars ahead)").grid(row=3, column=0, sticky=tk.W, padx=4, pady=2)
        hor_var = tk.IntVar(value=int(self.settings.get("horizon", 5)))
        hor_sp = ttk.Spinbox(frm, from_=1, to=500, textvariable=hor_var, width=10)
        hor_sp.grid(row=3, column=1, sticky=tk.W, padx=4, pady=2)

        # output type radio
        ttk.Label(frm, text="output type").grid(row=4, column=0, sticky=tk.W, padx=4, pady=2)
        out_var = tk.StringVar(value=self.settings.get("output_type", "returns"))
        rb1 = ttk.Radiobutton(frm, text="returns", value="returns", variable=out_var)
        rb2 = ttk.Radiobutton(frm, text="prices", value="prices", variable=out_var)
        rb1.grid(row=4, column=1, sticky=tk.W)
        rb2.grid(row=4, column=1, sticky=tk.E)

        # model path entry + browse
        ttk.Label(frm, text="model_path").grid(row=5, column=0, sticky=tk.W, padx=4, pady=2)
        mp_ent = ttk.Entry(frm, width=60)
        mp_ent.insert(0, str(self.settings.get("model_path","")))
        mp_ent.grid(row=5, column=1, sticky=tk.W, padx=4, pady=2)
        def browse_model():
            fn = filedialog.askopenfilename(title="Select model", filetypes=[("Pickle/Torch files","*.pkl;*.pt;*.pth"),("All files","*.*")])
            if fn:
                mp_ent.delete(0, tk.END); mp_ent.insert(0, fn)
        btn = ttk.Button(frm, text="Browse", command=browse_model)
        btn.grid(row=5, column=2, sticky=tk.W, padx=4, pady=2)

        # save
        def save_and_close():
            self.settings["symbol"] = sym_var.get().strip()
            self.settings["timeframe"] = tf_var.get().strip()
            try:
                self.settings["limit_candles"] = int(lim_var.get())
            except Exception:
                self.settings["limit_candles"] = DEFAULT_SETTINGS["limit_candles"]
            try:
                self.settings["horizon"] = int(hor_var.get())
            except Exception:
                self.settings["horizon"] = DEFAULT_SETTINGS["horizon"]
            self.settings["output_type"] = out_var.get()
            self.settings["model_path"] = mp_ent.get().strip()
            # update model label
            self.lbl_model.config(text=f"Model: {Path(self.settings.get('model_path','')).name or '(none)'}")
            # TODO: persist settings in DB table 'settings' if exists
            win.destroy()

        btn_save = ttk.Button(frm, text="Save", command=save_and_close)
        btn_save.grid(row=6, column=0, columnspan=3, pady=8)

    def _on_predict(self):
        t = threading.Thread(target=self._run_prediction, daemon=True)
        t.start()

    def _load_model(self):
        mp = self.settings.get("model_path", "") or ""
        if not mp:
            messagebox.showwarning("Model missing", "Seleziona prima un file modello nelle settings")
            return None
        p = Path(mp)
        if not p.exists():
            messagebox.showerror("Model missing", f"File non trovato: {mp}")
            return None
        # try pickle first
        try:
            payload = pickle.loads(p.read_bytes())
            # if payload is a bare model, wrap it
            if not isinstance(payload, dict):
                payload = {"model": payload}
            # normalize keys
            self.model_payload = payload
            return payload
        except Exception:
            # try torch
            if TORCH_AVAILABLE:
                try:
                    raw = torch.load(str(p))
                    if isinstance(raw, dict):
                        payload = raw
                    else:
                        payload = {"model": raw}
                    self.model_payload = payload
                    return payload
                except Exception as e:
                    messagebox.showerror("Model load error", f"Error loading with torch: {e}")
                    return None
            else:
                messagebox.showerror("Model load error", "Failed to load model (pickle failed and torch not available)")
                return None

    def _run_prediction(self):
        self._set_status("Generazione previsione...")
        try:
            payload = self._load_model()
            if not payload:
                self._set_status("Modello non caricato")
                return
            model = payload.get("model")
            features_list = payload.get("features") or []
            mu = payload.get("std_mu", {}) or {}
            sigma = payload.get("std_sigma", {}) or {}
            encoder = payload.get("encoder", None)

            # fetch last N candles per spec
            N = int(self.settings.get("limit_candles", 256))
            symbol = self.settings.get("symbol", "EUR/USD")
            tf = self.settings.get("timeframe", "1m")
            with self.db.engine.connect() as conn:
                q = "SELECT * FROM market_data_candles WHERE symbol=:s AND timeframe=:tf ORDER BY ts_utc DESC LIMIT :N"
                rows = conn.execute(text(q), {"s": symbol, "tf": tf, "N": N}).fetchall()
                if not rows:
                    self._set_status("No candles found for prediction")
                    return
                rows = rows[::-1]
                # build DataFrame trying to preserve columns used by pipeline
                sample = rows[0]
                if hasattr(sample, "_mapping"):
                    df = pd.DataFrame([dict(r._mapping) for r in rows])
                else:
                    # fallback assume ts_utc,open,high,low,close,volume...
                    df = pd.DataFrame(rows, columns=["ts_utc","open","high","low","close"] + ([ "volume" ] * (len(rows[0]) - 5)))
            candles = df.sort_values("ts_utc").reset_index(drop=True)

            # compute features via pipeline
            features_config = {
                "warmup_bars": int(self.settings.get("warmup_bars", 16)),
                "indicators": {
                    "atr": {"n": int(self.settings.get("atr_n", 14))},
                    "rsi": {"n": int(self.settings.get("rsi_n", 14))},
                    "bollinger": {"n": int(self.settings.get("bb_n", 20))},
                    "hurst": {"window": int(self.settings.get("hurst_window", 64))},
                },
                "standardization": {"window_bars": int(self.settings.get("rv_window", 60))}
            }
            feats, std_local = pipeline_process(candles.copy(), timeframe=tf, features_config=features_config)
            if "ts_utc" not in feats.columns:
                feats = feats.reset_index(drop=True)
            if feats.empty:
                self._set_status("Nessuna feature calcolata")
                return

            # Build X matrix for all available rows (apply order = features_list)
            if encoder and encoder.get("type") == "latents":
                # latents handling: get latest latent (not implemented fully here)
                messagebox.showinfo("Encoder latents", "Latents encoder not fully supported in this test GUI")
                self._set_status("Encoder latents - abort")
                return

            if not features_list:
                messagebox.showerror("Model error", "Model payload missing 'features' list")
                self._set_status("Model senza features")
                return

            missing = [f for f in features_list if f not in feats.columns]
            if missing:
                messagebox.showerror("Feature mismatch", f"Missing features in pipeline: {missing}")
                self._set_status(f"Feature missing: {missing}")
                return

            X = feats[features_list].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # standardize using mu/sigma from payload where available
            sigma_missing = False
            for col in features_list:
                mu_c = mu.get(col, None)
                sig_c = sigma.get(col, None)
                if mu_c is None or sig_c is None:
                    sigma_missing = True
                    continue
                try:
                    X[col] = (X[col].astype(float) - float(mu_c)) / (float(sig_c) if float(sig_c) != 0.0 else 1.0)
                except Exception:
                    X[col] = X[col].astype(float).fillna(0.0)

            if sigma_missing:
                messagebox.showwarning("Standardization warning", "Some sigma/mu values missing in model payload; local standardization partial")

            X_arr = X.to_numpy(dtype=float)

            # predict using sklearn-like predict or torch
            preds = None
            try:
                if TORCH_AVAILABLE and (hasattr(model, "forward") or isinstance(model, torch.nn.Module)):
                    model.eval()
                    with torch.no_grad():
                        t_in = torch.tensor(X_arr, dtype=torch.float32)
                        out = model(t_in)
                        preds = out.cpu().numpy()
                else:
                    # sklearn-like
                    preds = model.predict(X_arr)
                preds = np.asarray(preds).squeeze()
            except Exception as e:
                messagebox.showerror("Prediction error", f"Error during model.predict: {e}")
                self._set_status(f"Errore in predict: {e}")
                return

            # Determine forecast sequence for horizon:
            horizon = int(self.settings.get("horizon", 5))
            output_type = self.settings.get("output_type", "returns")

            # If preds length equals number of rows, take last 'horizon' predictions
            if preds.ndim == 0:
                seq = np.repeat(float(preds), horizon)
            elif preds.ndim == 1:
                if len(preds) >= horizon:
                    seq = preds[-horizon:]
                else:
                    # pad/extend
                    seq = np.concatenate([preds, np.repeat(preds[-1], horizon - len(preds))])
            else:
                # multi-dim output, try to use last row and first dim
                try:
                    seq = preds[-1].reshape(-1)[:horizon]
                    if seq.size < horizon:
                        seq = np.pad(seq, (0, horizon - seq.size), mode='edge')
                except Exception:
                    seq = np.repeat(float(preds.flatten()[-1]), horizon)

            last_close = float(candles["close"].iat[-1])

            if output_type == "returns":
                # seq are returns per bar; build cumulative prices
                prices = [last_close]
                for r in seq:
                    prices.append(prices[-1] * (1.0 + float(r)))
                # drop the initial last_close, keep only future points
                forecast_prices = np.array(prices[1:])
            else:
                # prices directly
                forecast_prices = np.array(seq, dtype=float)

            # build future timestamps
            last_ts = pd.to_datetime(candles["ts_utc"].astype("int64"), unit="ms", utc=True).iat[-1]
            delta = timeframe_to_timedelta(tf)
            future_ts = [last_ts + delta * (i + 1) for i in range(len(forecast_prices))]

            # plot base candles close if not already
            self.ax.clear()
            dt = pd.to_datetime(candles["ts_utc"].astype("int64"), unit="ms", utc=True)
            self.ax.plot(dt, candles["close"], color="blue", label="close")
            # plot forecast as broken gray line (RGB 180,180,180) alpha 0.7
            gray_rgb = (180/255.0, 180/255.0, 180/255.0)
            self.ax.plot(future_ts, forecast_prices, color=gray_rgb, alpha=0.7, linewidth=2, marker="o", label=f"Forecast ({Path(self.settings.get('model_path','')).name or 'model'})")
            self.ax.set_title(f"{symbol} {tf} - Ultimi {len(candles)} candles + forecast")
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            self.canvas.draw()

            self._set_status(f"Forecast generata ({len(forecast_prices)} points). Last price {forecast_prices[-1]:.6f}")
        except Exception as e:
            self._set_status(f"Errore previsione: {e}")
            messagebox.showerror("Errore previsione", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ForecastGUI(root)
    root.mainloop()
