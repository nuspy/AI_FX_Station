#!/usr/bin/env python3
"""
tests/manual_tests/gui_forecast.py

GUI Tkinter che:
 - carica gli ultimi candles dal DB
 - carica un modello serializzato (pickle/torch) scelto dall'utente
 - calcola features via pipeline_process
 - genera la previsione e la sovrappone al grafico come linea grigia (opacità 66.6%)
 - dialog di impostazioni per parametri di query/forecast/modello con persistenza tra sessioni

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

SETTINGS_FILE = ROOT / ".forecast_gui_settings.json"

DEFAULT_SETTINGS = {
    "symbol": "EUR/USD",
    "timeframe": "1m",
    "limit_candles": 256,
    "warmup_bars": 16,
    "atr_n": 14,
    "rsi_n": 14,
    "bb_n": 20,
    "hurst_window": 64,
    "rv_window": 60,
    "horizon": 5,
    "model_path": "",  # selezionato dall'utente
    "output_type": "returns",  # "returns" o "prices"
}

def timeframe_to_timedelta(tf: str):
    """Convert timeframe like '1m','5m','1h','1d' to pandas Timedelta."""
    tf = str(tf).strip().lower()
    try:
        if tf.endswith("m"):
            return pd.to_timedelta(int(tf[:-1]), unit="m")
        if tf.endswith("h"):
            return pd.to_timedelta(int(tf[:-1]), unit="h")
        if tf.endswith("d"):
            return pd.to_timedelta(int(tf[:-1]), unit="d")
        return pd.to_timedelta(int(tf), unit="m")
    except Exception:
        return pd.to_timedelta(1, unit="m")

class SettingsStore:
    @staticmethod
    def load() -> dict:
        if SETTINGS_FILE.exists():
            try:
                return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            except Exception:
                return DEFAULT_SETTINGS.copy()
        return DEFAULT_SETTINGS.copy()

    @staticmethod
       def save(data: dict) -> None:
        try:
            SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print("Impossibile salvare le impostazioni:", e)

class ForecastGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sistema Previsionale - GUI")
        # load persisted settings
        self.settings = DEFAULT_SETTINGS.copy()
        self.settings.update(SettingsStore.load() or {})
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
        self.btn_predict = ttk.Button(ctrl, text="Fai previsione", command=self._on_predict)
        self.btn_predict.pack(side=tk.LEFT, padx=2)
        self.btn_settings = ttk.Button(ctrl, text="Setting previsione", command=self._open_settings)
        self.btn_settings.pack(side=tk.LEFT, padx=2)
        self.lbl_model = ttk.Label(ctrl, text=f"Model: {Path(self.settings.get('model_path','')).name or '(none)'}")
        self.lbl_model.pack(side=tk.LEFT, padx=8)

        # status
        self.status = ttk.Label(self.root, text="Pronto")
        self.status.pack(fill=tk.X, padx=4, pady=2)

        # internal state
        self.candles = pd.DataFrame()
        self.features = pd.DataFrame()
        self.model_payload = None
        self.forecast_line = None  # handle to last forecast overlay

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
            self.forecast_line = None  # reset overlay on refresh
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
        win.title("Setting previsione")
        frm = ttk.Frame(win, padding=8)
        frm.pack(fill=tk.BOTH, expand=1)

        # symbol combobox populated from DB
        ttk.Label(frm, text="Simbolo").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        symbols = ["EUR/USD"]
        try:
            with self.db.engine.connect() as conn:
                rows = conn.execute(text("SELECT DISTINCT symbol FROM market_data_candles ORDER BY symbol")).fetchall()
                symbols = [r._mapping["symbol"] if hasattr(r, "_mapping") else r[0] for r in rows] or symbols
        except Exception:
            pass
        sym_var = tk.StringVar(value=self.settings.get("symbol"))
        ttk.Combobox(frm, values=symbols, textvariable=sym_var, width=30).grid(row=0, column=1, sticky=tk.W, padx=4, pady=2)

        # timeframe combobox
        ttk.Label(frm, text="Timeframe").grid(row=1, column=0, sticky=tk.W, padx=4, pady=2)
        tf_values = ["1m","5m","15m","30m","1h","4h","1d"]
        tf_var = tk.StringVar(value=self.settings.get("timeframe"))
        ttk.Combobox(frm, values=tf_values, textvariable=tf_var, width=15).grid(row=1, column=1, sticky=tk.W, padx=4, pady=2)

        # limit_candles spinbox
        ttk.Label(frm, text="N candles").grid(row=2, column=0, sticky=tk.W, padx=4, pady=2)
        lim_var = tk.IntVar(value=int(self.settings.get("limit_candles", 256)))
        ttk.Spinbox(frm, from_=32, to=10000, textvariable=lim_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=4, pady=2)

        # horizon spinbox
        ttk.Label(frm, text="Orizzonte forecast (barre)").grid(row=3, column=0, sticky=tk.W, padx=4, pady=2)
        hor_var = tk.IntVar(value=int(self.settings.get("horizon", 5)))
        ttk.Spinbox(frm, from_=1, to=1000, textvariable=hor_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=4, pady=2)

        # output type radio
        ttk.Label(frm, text="Tipo output").grid(row=4, column=0, sticky=tk.W, padx=4, pady=2)
        out_var = tk.StringVar(value=self.settings.get("output_type", "returns"))
        ttk.Radiobutton(frm, text="ritorni", value="returns", variable=out_var).grid(row=4, column=1, sticky=tk.W)
        ttk.Radiobutton(frm, text="prezzi", value="prices", variable=out_var).grid(row=4, column=1, sticky=tk.E)

        # model path entry + browse
        ttk.Label(frm, text="File modello").grid(row=5, column=0, sticky=tk.W, padx=4, pady=2)
        mp_ent = ttk.Entry(frm, width=60)
        mp_ent.insert(0, str(self.settings.get("model_path","")))
        mp_ent.grid(row=5, column=1, sticky=tk.W, padx=4, pady=2)
        def browse_model():
            fn = filedialog.askopenfilename(title="Seleziona modello", filetypes=[("Modelli (pickle/torch)","*.pkl;*.pt;*.pth"),("Tutti i file","*.*")])
            if fn:
                mp_ent.delete(0, tk.END); mp_ent.insert(0, fn)
        ttk.Button(frm, text="Sfoglia", command=browse_model).grid(row=5, column=2, sticky=tk.W, padx=4, pady=2)

        # pipeline tweakables
        ttk.Label(frm, text="warmup_bars").grid(row=6, column=0, sticky=tk.W, padx=4, pady=2)
        warm_var = tk.IntVar(value=int(self.settings.get("warmup_bars", 16)))
        ttk.Spinbox(frm, from_=0, to=500, textvariable=warm_var, width=10).grid(row=6, column=1, sticky=tk.W, padx=4, pady=2)

        ttk.Label(frm, text="atr_n").grid(row=7, column=0, sticky=tk.W, padx=4, pady=2)
        atr_var = tk.IntVar(value=int(self.settings.get("atr_n", 14)))
        ttk.Spinbox(frm, from_=1, to=500, textvariable=atr_var, width=10).grid(row=7, column=1, sticky=tk.W, padx=4, pady=2)

        ttk.Label(frm, text="rsi_n").grid(row=8, column=0, sticky=tk.W, padx=4, pady=2)
        rsi_var = tk.IntVar(value=int(self.settings.get("rsi_n", 14)))
        ttk.Spinbox(frm, from_=1, to=500, textvariable=rsi_var, width=10).grid(row=8, column=1, sticky=tk.W, padx=4, pady=2)

        ttk.Label(frm, text="bb_n").grid(row=9, column=0, sticky=tk.W, padx=4, pady=2)
        bb_var = tk.IntVar(value=int(self.settings.get("bb_n", 20)))
        ttk.Spinbox(frm, from_=1, to=500, textvariable=bb_var, width=10).grid(row=9, column=1, sticky=tk.W, padx=4, pady=2)

        ttk.Label(frm, text="hurst_window").grid(row=10, column=0, sticky=tk.W, padx=4, pady=2)
        hurst_var = tk.IntVar(value=int(self.settings.get("hurst_window", 64)))
        ttk.Spinbox(frm, from_=2, to=5000, textvariable=hurst_var, width=10).grid(row=10, column=1, sticky=tk.W, padx=4, pady=2)

        ttk.Label(frm, text="rv_window").grid(row=11, column=0, sticky=tk.W, padx=4, pady=2)
        rv_var = tk.IntVar(value=int(self.settings.get("rv_window", 60)))
        ttk.Spinbox(frm, from_=2, to=5000, textvariable=rv_var, width=10).grid(row=11, column=1, sticky=tk.W, padx=4, pady=2)

        def save_and_close():
            self.settings["symbol"] = sym_var.get().strip()
            self.settings["timeframe"] = tf_var.get().strip()
            self.settings["limit_candles"] = int(lim_var.get())
            self.settings["horizon"] = int(hor_var.get())
            self.settings["output_type"] = out_var.get()
            self.settings["model_path"] = mp_ent.get().strip()
            self.settings["warmup_bars"] = int(warm_var.get())
            self.settings["atr_n"] = int(atr_var.get())
            self.settings["rsi_n"] = int(rsi_var.get())
            self.settings["bb_n"] = int(bb_var.get())
            self.settings["hurst_window"] = int(hurst_var.get())
            self.settings["rv_window"] = int(rv_var.get())
            # persist to disk
            SettingsStore.save(self.settings)
            # update model label
            self.lbl_model.config(text=f"Model: {Path(self.settings.get('model_path','')).name or '(none)'}")
            win.destroy()

        ttk.Button(frm, text="Salva", command=save_and_close).grid(row=12, column=0, columnspan=3, pady=8)

    def _on_predict(self):
        t = threading.Thread(target=self._run_prediction, daemon=True)
        t.start()

    def _load_model(self):
        mp = self.settings.get("model_path", "") or ""
        if not mp:
            messagebox.showwarning("Modello mancante", "Seleziona prima un file modello nelle impostazioni")
            return None
        p = Path(mp)
        if not p.exists():
            messagebox.showerror("Modello mancante", f"File non trovato: {mp}")
            return None
        # try pickle first
        try:
            payload = pickle.loads(p.read_bytes())
            if not isinstance(payload, dict):
                payload = {"model": payload}
            self.model_payload = payload
            return payload
        except Exception:
            # try torch
            if TORCH_AVAILABLE:
                try:
                    raw = torch.load(str(p))
                    payload = raw if isinstance(raw, dict) else {"model": raw}
                    self.model_payload = payload
                    return payload
                except Exception as e:
                    messagebox.showerror("Errore caricamento modello", f"Errore torch.load: {e}")
                    return None
            messagebox.showerror("Errore caricamento modello", "Impossibile caricare il modello (pickle fallito e torch non disponibile)")
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

            # fetch last N candles
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
            if hasattr(rows[0], "_mapping"):
                candles = pd.DataFrame([dict(r._mapping) for r in rows]).sort_values("ts_utc").reset_index(drop=True)
            else:
                candles = pd.DataFrame(rows, columns=["ts_utc","open","high","low","close"]).sort_values("ts_utc").reset_index(drop=True)

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
            feats, _ = pipeline_process(candles.copy(), timeframe=tf, features_config=features_config)
            if feats.empty:
                self._set_status("Nessuna feature calcolata")
                return

            # build X
            if not features_list:
                messagebox.showerror("Errore modello", "Il payload del modello non contiene la lista 'features'")
                self._set_status("Model senza features")
                return
            missing = [f for f in features_list if f not in feats.columns]
            if missing:
                messagebox.showerror("Mismatch features", f"Feature mancanti nella pipeline: {missing}")
                self._set_status(f"Feature missing: {missing}")
                return

            X = feats[features_list].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # standardize using mu/sigma from payload where available
            sigma_missing = False
            for col in features_list:
                mu_c = mu.get(col)
                sig_c = sigma.get(col)
                if mu_c is None or sig_c is None:
                    sigma_missing = True
                    continue
                try:
                    denom = float(sig_c) if float(sig_c) != 0.0 else 1.0
                    X[col] = (X[col].astype(float) - float(mu_c)) / denom
                except Exception:
                    X[col] = X[col].astype(float).fillna(0.0)
            if sigma_missing:
                messagebox.showwarning("Standardizzazione parziale", "Alcuni valori di mu/sigma non sono presenti nel modello.")

            X_arr = X.to_numpy(dtype=float)

            # predict (sklearn-like or torch)
            try:
                if TORCH_AVAILABLE and (hasattr(model, "forward") or isinstance(model, torch.nn.Module)):
                    model.eval()
                    with torch.no_grad():
                        t_in = torch.tensor(X_arr, dtype=torch.float32)
                        out = model(t_in)
                        preds = out.cpu().numpy().squeeze()
                else:
                    preds = np.asarray(model.predict(X_arr)).squeeze()
            except Exception as e:
                messagebox.showerror("Errore predict", f"Errore durante la predizione: {e}")
                self._set_status(f"Errore in predict: {e}")
                return

            horizon = int(self.settings.get("horizon", 5))
            output_type = self.settings.get("output_type", "returns")

            # obtain sequence of length horizon
            if preds.ndim == 0:
                seq = np.repeat(float(preds), horizon)
            elif preds.ndim == 1:
                seq = preds[-horizon:] if len(preds) >= horizon else np.concatenate([preds, np.repeat(preds[-1], horizon - len(preds))])
            else:
                try:
                    seq = preds[-1].reshape(-1)[:horizon]
                    if seq.size < horizon:
                        seq = np.pad(seq, (0, horizon - seq.size), mode='edge')
                except Exception:
                    seq = np.repeat(float(np.ravel(preds)[-1]), horizon)

            last_close = float(candles["close"].iat[-1])

            if output_type == "returns":
                prices = [last_close]
                for r in seq:
                    prices.append(prices[-1] * (1.0 + float(r)))
                forecast_prices = np.array(prices[1:], dtype=float)
            else:
                forecast_prices = np.array(seq, dtype=float)

            last_ts = pd.to_datetime(candles["ts_utc"].astype("int64"), unit="ms", utc=True).iat[-1]
            delta = timeframe_to_timedelta(tf)
            future_ts = [last_ts + delta * (i + 1) for i in range(len(forecast_prices))]

            # overlay on existing chart without clearing
            if self.forecast_line is not None:
                try:
                    self.forecast_line.remove()
                except Exception:
                    pass
                self.forecast_line = None

            # draw base line if empty (safety)
            if not self.ax.lines:
                dt = pd.to_datetime(candles["ts_utc"].astype("int64"), unit="ms", utc=True)
                self.ax.plot(dt, candles["close"], color="blue", label="close")

            self.forecast_line, = self.ax.plot(
                future_ts,
                forecast_prices,
                color="gray",
                alpha=0.666,  # 66.6% opacity
                linewidth=2.0,
                marker="o",
                label=f"Forecast ({Path(self.settings.get('model_path','')).name or 'model'})",
            )
            self.ax.legend()
            self.canvas.draw()
            self._set_status(f"Forecast generata: {len(forecast_prices)} punti")
        except Exception as e:
            self._set_status(f"Errore previsione: {e}")
            messagebox.showerror("Errore previsione", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ForecastGUI(root)
    root.mainloop()
