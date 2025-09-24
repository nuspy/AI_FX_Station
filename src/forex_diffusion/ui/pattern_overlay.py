# pattern_overlay.py
from __future__ import annotations
import math
from typing import Iterable, List, Optional
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from loguru import logger
import matplotlib.axes as mpla
from loguru import logger


MAX_OVERLAYS = 80          # limite “sanity” per non saturare la UI
MIN_SEP_BARS = 8           # distanza minima (in barre) tra annotazioni del medesimo pattern

class PatternOverlayRenderer:
    def __init__(self, controller, info_provider):
        self.controller = controller
        self.info = info_provider
        self.ax = getattr(controller, "axes_price", None) or getattr(controller.view, "ax_price", None)
        self._badges = []
        self.use_badges = True
        self._last_mode_log = None  # evita log ripetuti su auto X-mode

    # ---------- helpers ----------

    def set_axes(self, ax: mpla.Axes) -> None:
        """Permette a plot_service di passare esplicitamente l'asse prezzo."""
        self.ax = ax

    def _resolve_axes(self) -> mpla.Axes | None:
        """
        Trova un asse adatto su cui disegnare:
          1) se self.ax è già valido -> usalo;
          2) prova su controller / plot_service / view con nomi comuni (ax_price, axes_price, price_ax);
          3) fallback: primo axes della figura.
        """
        # 1) asse già impostato
        if getattr(self, "ax", None) is not None and getattr(self.ax, "figure", None) is not None:
            return self.ax

        candidates: list[mpla.Axes] = []

        def _collect(obj):
            if not obj:
                return
            # nomi più comuni nell'app
            for name in ("ax_price", "axes_price", "price_ax", "ax", "axes"):
                a = getattr(obj, name, None)
                if isinstance(a, mpla.Axes):
                    candidates.append(a)
                elif isinstance(a, dict):
                    for v in a.values():
                        if isinstance(v, mpla.Axes):
                            candidates.append(v)
                elif isinstance(a, (list, tuple)):
                    for v in a:
                        if isinstance(v, mpla.Axes):
                            candidates.append(v)
            # prova a risalire alla figura
            fig = getattr(getattr(obj, "canvas", None), "figure", None) or getattr(obj, "figure", None)
            if fig and getattr(fig, "axes", None):
                for a in fig.axes:
                    if isinstance(a, mpla.Axes):
                        candidates.append(a)

        # 2) cerca in controller/plot_service/view
        ctrl = getattr(self, "controller", None)
        _collect(ctrl)
        _collect(getattr(ctrl, "plot_service", None))
        _collect(getattr(ctrl, "view", None))

        # 3) scegli l’asse con più elementi grafici (euristica)
        if candidates:
            def _score(ax: mpla.Axes) -> int:
                return len(ax.lines) + len(ax.collections) + len(ax.patches) + len(ax.texts)

            self.ax = max(candidates, key=_score)
            return self.ax

        logger.debug("PatternOverlayRenderer: nessun axes trovato")
        return None

    def _to_y(self, price) -> float:
        try:
            return float(price)
        except Exception:
            return np.nan

    def _clear_badges(self):
        try:
            for a in self._badges:
                try: a.remove()
                except Exception: pass
        finally:
            self._badges = []

    # ---------- API ----------
    def clear(self):
        self._clear_badges()
        if self.ax:
            self.ax.figure.canvas.draw_idle()

    def _axis_uses_dates(self) -> str:
        """
        Rileva il "time mode" dell'asse X:
          - "matplotlib_date" (mdates date2num ~ 2e4..1e6)
          - "ms_epoch" (1.6e12..)
          - "index" (0..N)
        """
        if not self.ax:
            return "index"
        # prova a leggere i dati x della prima linea/collection
        xs = None
        try:
            if self.ax.lines:
                xd = self.ax.lines[0].get_xdata()
                xs = np.asarray(xd, dtype=float) if len(xd) else None
            if xs is None and self.ax.collections:
                try:
                    xs = np.asarray(self.ax.collections[0].get_offsets())[:, 0]
                except Exception:
                    xs = None
        except Exception:
            xs = None

        if xs is not None and len(xs) >= 2:
            x0 = float(xs[-1])
            if x0 > 1e10:
                return "ms_epoch"
            if 1e3 < x0 < 1e7:
                return "matplotlib_date"
            return "index"
        # fallback: guarda i limiti asse
        xmin, xmax = self.ax.get_xlim()
        if xmax > 1e10:
            return "ms_epoch"
        if 1e3 < xmax < 1e7:
            return "matplotlib_date"
        return "index"

    def _to_axis_x(self, ts, mode: str) -> float:
        """Converte ts in coordinate X secondo 'mode'."""
        import pandas as pd
        try:
            if isinstance(ts, (pd.Timestamp, np.datetime64)):
                t = pd.Timestamp(ts)
            elif isinstance(ts, (int, np.integer, float)) and ts > 1e11:
                t = pd.to_datetime(int(ts), unit="ms", utc=True)
            elif isinstance(ts, (int, np.integer, float)) and ts > 1e9:
                t = pd.to_datetime(int(ts), unit="s", utc=True)
            else:
                t = pd.to_datetime(ts, utc=True)
        except Exception:
            # se è già un numero (es. indice)
            try:
                return float(ts)
            except Exception:
                return np.nan

        if mode == "matplotlib_date":
            return mdates.date2num(t.tz_convert(None).to_pydatetime())
        elif mode == "ms_epoch":
            return float(t.value // 1_000_000)  # ms
        else:
            # su asse indice, usiamo la posizione relativa (ultimo punto)
            # (metodo semplice: mappa il tempo su indice supponendo ordinamento)
            return np.nan  # verrà filtrato

    def draw(self, events: Iterable[object]):
        if not self._resolve_axes():
            logger.debug("PatternOverlay: no axes to draw on")
            return

        self._clear_badges()
        evs = list(events) if events else []
        if not evs:
            self.ax.figure.canvas.draw_idle()
            return

        mode = self._axis_uses_dates()

        # Normalizza → (key, x, y, kind, direction, obj)
        norm = []
        for e in evs:
            key = getattr(e, "key", getattr(e, "name", type(e).__name__))
            ts = getattr(e, "confirm_ts", getattr(e, "ts", None))
            px = getattr(e, "confirm_price", getattr(e, "price", getattr(e, "target_price", None)))
            kind = getattr(e, "kind", "unknown")
            direction = getattr(e, "direction", "neutral")

            x = self._x_from_ts_safest(ts)

            try:
                y = float(px) if px is not None else np.nan
            except Exception:
                y = np.nan

            if not np.isnan(x) and not np.isnan(y):
                norm.append((key, x, y, kind, direction, e))

        # ordina per x decrescente (più recenti)
        try:
            norm.sort(key=lambda t: float(t[1]), reverse=True)
        except Exception:
            pass

        kept = []
        picked_any = set()
        last_x_by_key = {}
        for key, x, y, kind, direction, e in norm:
            # Garantisci almeno 1 badge per pattern (anche se densissimo)
            if key not in picked_any:
                kept.append((x, y, key, kind, direction, e))
                picked_any.add(key)
                last_x_by_key[key] = x
                if len(kept) >= MAX_OVERLAYS:
                    break
                continue

            # de-dup per vicinanza
            try:
                xd = self.ax.lines[0].get_xdata()
                step = float(xd[-1]) - float(xd[-2]) if len(xd) > 2 else 1.0
            except Exception:
                step = 1.0
            if abs(x - last_x_by_key.get(key, -1e18)) < max(step * MIN_SEP_BARS, 1.0):
                continue
            last_x_by_key[key] = x
            kept.append((x, y, key, kind, direction, e))
            if len(kept) >= MAX_OVERLAYS:
                break

        logger.info(f"PatternOverlay: drawing {len(kept)}/{len(evs)} events on ax=Axes (mode={mode})")

        for x, y, key, kind, direction, e in kept:
            try:
                self._draw_badge(x, y, key, direction, e)
            except Exception as ex:
                logger.debug(f"overlay draw_event failed: {ex}")

        try:
            self.ax.figure.canvas.draw_idle()
        except Exception:
            pass

    def _draw_badge(self, x: float, y: float, key: str, direction: str, event_obj: object):
        """
        Badge semplice: un marcatore e un testo corto sopra il punto.
        (L’icona 'i' / info box la lasciamo al pick handler già esistente).
        """
        # stile base
        color = "#2ecc71" if str(direction).lower().startswith("up") else "#e74c3c" if str(
            direction).lower().startswith("down") else "#3498db"
        ms = 10.0  # prima 6.0
        self.ax.plot([x], [y], marker="o", markersize=ms, markerfacecolor=color,
                     markeredgecolor="black", markeredgewidth=0.8, zorder=100, picker=6)
        txt = self.ax.text(x, y, f" {key} ",
                           color="white",
                           bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="black", lw=0.6, alpha=0.95),
                           fontsize=9, va="bottom", ha="left", zorder=101)
        self._badges.append(txt)

    # opzionale: gestore pick per mostrare info estese
    def on_pick(self, event):
        # qui puoi collegare la tua info-box esistente (PatternInfoProvider)
        pass

    def _to_ts(self, ts):
        """Normalizza ts a pandas.Timestamp (UTC)."""
        try:
            if isinstance(ts, (pd.Timestamp, np.datetime64)):
                t = pd.Timestamp(ts)
            elif isinstance(ts, (int, np.integer, float)) and ts > 1e11:   # ms epoch
                t = pd.to_datetime(int(ts), unit="ms", utc=True)
            elif isinstance(ts, (int, np.integer, float)) and ts > 1e9:    # s epoch
                t = pd.to_datetime(int(ts), unit="s", utc=True)
            else:
                t = pd.to_datetime(ts, utc=True)
            return t.tz_convert("UTC")
        except Exception:
            return None

    def _x_from_ts_candidates(self, ts):
        """Restituisce (x_date2num, x_ms_epoch) per lo stesso timestamp."""
        t = self._to_ts(ts)
        if t is None:
            return (np.nan, np.nan)
        x_date = mdates.date2num(t.tz_convert(None).to_pydatetime())    # ~ 20_000 .. 1_000_000
        x_ms   = float(t.value // 1_000_000)                            # ~ 1.7e12
        return (x_date, x_ms)

    def _x_from_ts_safest(self, ts) -> float:
        """
        Sceglie automaticamente la coordinata X che cade dentro l'intervallo dell'asse:
        - calcola sia date2num che ms_epoch;
        - se uno dei due sta in [xmin, xmax], usa quello;
        - se entrambi fuori, prende quello più vicino al centro dell'asse.
        """
        if not self.ax:
            return np.nan
        xmin, xmax = self.ax.get_xlim()
        x_date, x_ms = self._x_from_ts_candidates(ts)
        mid = 0.5 * (xmin + xmax)

        def score(x):
            if np.isnan(x):
                return (False, np.inf)
            inside = (xmin <= x <= xmax)
            dist = abs(x - mid)
            return (inside, dist)

        in_date, d_date = score(x_date)
        in_ms,   d_ms   = score(x_ms)

        if in_date and not in_ms:
            mode = "date2num"
            x = x_date
        elif in_ms and not in_date:
            mode = "ms_epoch"
            x = x_ms
        else:
            # entrambi dentro o entrambi fuori -> scegli più vicino al centro
            if d_date <= d_ms:
                mode, x = "date2num", x_date
            else:
                mode, x = "ms_epoch", x_ms

        # log una sola volta per capire che mode stiamo usando
        if not hasattr(self, "_last_mode_log") or self._last_mode_log != mode:
            logger.info(f"PatternOverlay: auto X-mode → {mode} (xlim=({xmin:.2f},{xmax:.2f}))")
            self._last_mode_log = mode

        return x