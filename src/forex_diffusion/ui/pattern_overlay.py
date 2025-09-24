# pattern_overlay.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Dict
import math
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.axes as mpla
from loguru import logger

# Limiti UI
MAX_OVERLAYS = 80
MIN_SEP_BARS = 8
HIT_RADIUS_PX = 12  # raggio hit-test per hover/click

class PatternOverlayRenderer:
    """
    Renderizza badge pattern + tooltip + dialog info + freccia target.
    Funziona con asse X in ms-epoch o matplotlib date2num; autodetect.
    """
    def __init__(self, controller, info_provider):
        self.controller = controller
        self.info = info_provider
        self.ax: Optional[mpla.Axes] = getattr(controller, "axes_price", None) or getattr(getattr(controller, "view", None), "ax_price", None)
        self.use_badges = True

        # Stato grafico
        self._badges: List[mpla.Artist] = []
        self._arrows: List[mpla.Artist] = []
        self._artist_map: Dict[mpla.Artist, object] = {}   # artist -> event obj
        self._last_mode_log: Optional[str] = None

        # Interazione
        self._cid_move = None
        self._cid_click = None
        self._tooltip = None
        self._last_hover_artist: Optional[mpla.Artist] = None

    # ---------- Public API ----------
    def set_axes(self, ax: mpla.Axes) -> None:
        """Permette a plot_service di passare esplicitamente l'asse prezzo."""
        self.ax = ax
        self._bind_canvas_events()

    def clear(self) -> None:
        self._clear_all()
        if self.ax and self.ax.figure:
            try:
                self.ax.figure.canvas.draw_idle()
            except Exception:
                pass

    def draw(self, events: Iterable[object]) -> None:
        ax = self._resolve_axes()
        if not ax:
            logger.debug("PatternOverlay: no axes to draw on")
            return

        self._clear_all()
        evs = list(events) if events else []
        if not evs:
            ax.figure.canvas.draw_idle()
            return

        # Normalizza e filtra densità
        norm = self._normalize_events(ax, evs)
        kept = self._density_filter(ax, norm)
        mode = self._axis_mode(ax)
        logger.info(f"PatternOverlay: drawing {len(kept)}/{len(evs)} events on ax=Axes (mode={mode})")

        # Disegno badge + testo + freccia target
        for x, y, key, kind, direction, e in kept:
            try:
                self._draw_badge(x, y, key, direction, e)
                self._draw_target_arrow(x, y, direction, e)
            except Exception as ex:
                logger.debug(f"overlay draw_event failed: {ex}")

        self._bind_canvas_events()  # assicura eventi attivi
        try:
            ax.figure.canvas.draw_idle()
        except Exception:
            pass

    # ---------- Internals: axes & time ----------
    def _resolve_axes(self) -> Optional[mpla.Axes]:
        if getattr(self, "ax", None) is not None and getattr(self.ax, "figure", None) is not None:
            return self.ax

        candidates: List[mpla.Axes] = []

        def _collect(obj):
            if not obj:
                return
            for name in ("ax_price", "axes_price", "price_ax", "ax", "axes"):
                a = getattr(obj, name, None)
                if isinstance(a, mpla.Axes):
                    candidates.append(a)
                elif isinstance(a, dict):
                    candidates.extend([v for v in a.values() if isinstance(v, mpla.Axes)])
                elif isinstance(a, (list, tuple)):
                    candidates.extend([v for v in a if isinstance(v, mpla.Axes)])
            fig = getattr(getattr(obj, "canvas", None), "figure", None) or getattr(obj, "figure", None)
            if fig and getattr(fig, "axes", None):
                candidates.extend([a for a in fig.axes if isinstance(a, mpla.Axes)])

        _collect(self.controller)
        _collect(getattr(self.controller, "plot_service", None))
        _collect(getattr(self.controller, "view", None))

        if candidates:
            def _score(ax: mpla.Axes) -> int:
                return len(ax.lines) + len(ax.collections) + len(ax.patches) + len(ax.texts)
            self.ax = max(candidates, key=_score)
            return self.ax

        logger.debug("PatternOverlayRenderer: nessun axes trovato")
        return None

    def _axis_mode(self, ax: mpla.Axes) -> str:
        try:
            if ax.lines:
                xd = ax.lines[0].get_xdata()
                if len(xd):
                    x0 = float(np.asarray(xd)[-1])
                    if x0 > 1e10: return "ms_epoch"
                    if 1e3 < x0 < 1e7: return "matplotlib_date"
                    return "index"
        except Exception:
            pass
        xmin, xmax = ax.get_xlim()
        if xmax > 1e10: return "ms_epoch"
        if 1e3 < xmax < 1e7: return "matplotlib_date"
        return "index"

    def _to_ts(self, ts) -> Optional[pd.Timestamp]:
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

    def _x_from_ts_bimode(self, ts) -> Tuple[float, float]:
        t = self._to_ts(ts)
        if t is None:
            return (np.nan, np.nan)
        x_date = mdates.date2num(t.tz_convert(None).to_pydatetime())
        x_ms = float(t.value // 1_000_000)  # ms
        return (x_date, x_ms)

    def _x_from_ts_safest(self, ax: mpla.Axes, ts) -> float:
        xmin, xmax = ax.get_xlim()
        x_date, x_ms = self._x_from_ts_bimode(ts)
        mid = 0.5 * (xmin + xmax)

        def score(x):
            if np.isnan(x): return (False, np.inf)
            inside = (xmin <= x <= xmax)
            dist = abs(x - mid)
            return (inside, dist)

        in_date, d_date = score(x_date)
        in_ms, d_ms = score(x_ms)
        if in_date and not in_ms:
            mode, x = "date2num", x_date
        elif in_ms and not in_date:
            mode, x = "ms_epoch", x_ms
        else:
            mode, x = ("date2num", x_date) if d_date <= d_ms else ("ms_epoch", x_ms)

        if self._last_mode_log != mode:
            logger.info(f"PatternOverlay: auto X-mode → {mode} (xlim=({xmin:.2f},{xmax:.2f}))")
            self._last_mode_log = mode
        return x

    # ---------- Internals: normalize & density ----------
    def _normalize_events(self, ax: mpla.Axes, evs: List[object]) -> List[Tuple[float,float,str,str,str,object]]:
        norm = []
        for e in evs:
            key = getattr(e, "key", getattr(e, "name", type(e).__name__))
            ts = getattr(e, "confirm_ts", getattr(e, "ts", None))
            px = getattr(e, "confirm_price", getattr(e, "price", getattr(e, "target_price", None)))
            kind = getattr(e, "kind", "unknown")
            direction = getattr(e, "direction", "neutral")

            x = self._x_from_ts_safest(ax, ts)
            try:
                y = float(px) if px is not None else np.nan
            except Exception:
                y = np.nan

            if not np.isnan(x) and not np.isnan(y):
                norm.append((x, y, key, kind, direction, e))

        try:
            norm.sort(key=lambda t: float(t[0]), reverse=True)  # più recenti
        except Exception:
            pass
        return norm

    def _density_filter(self, ax: mpla.Axes, norm: List[Tuple[float,float,str,str,str,object]]):
        kept = []
        picked_any = set()
        last_x_by_key: Dict[str, float] = {}
        try:
            xd = ax.lines[0].get_xdata()
            step = float(xd[-1]) - float(xd[-2]) if len(xd) > 2 else 1.0
        except Exception:
            step = 1.0

        for x, y, key, kind, direction, e in norm:
            if key not in picked_any:
                kept.append((x, y, key, kind, direction, e))
                picked_any.add(key); last_x_by_key[key] = x
            else:
                if abs(x - last_x_by_key.get(key, -1e18)) < max(step * MIN_SEP_BARS, 1.0):
                    continue
                last_x_by_key[key] = x
                kept.append((x, y, key, kind, direction, e))
            if len(kept) >= MAX_OVERLAYS:
                break
        return kept

    # ---------- Drawing primitives ----------
    def _clear_all(self) -> None:
        for art in self._badges + self._arrows:
            try: art.remove()
            except Exception: pass
        self._badges.clear()
        self._arrows.clear()
        self._artist_map.clear()
        # tooltip
        if self._tooltip is not None:
            try:
                self._tooltip.remove()
            except Exception:
                pass
        self._tooltip = None
        self._last_hover_artist = None

    def _badge_color(self, direction: str) -> str:
        d = str(direction).lower()
        if d.startswith("up"): return "#2ecc71"
        if d.startswith("down"): return "#e74c3c"
        return "#3498db"

    def _draw_badge(self, x: float, y: float, key: str, direction: str, event_obj: object) -> None:
        ax = self.ax
        color = self._badge_color(direction)
        ms = 9.0
        # marcatore
        ln = ax.plot([x], [y], marker="o", markersize=ms, markerfacecolor=color,
                     markeredgecolor="black", markeredgewidth=0.8, zorder=120, picker=HIT_RADIUS_PX)[0]
        # etichetta compatta
        txt = ax.text(x, y, f" {key} ", color="white",
                      bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="black", lw=0.6, alpha=0.95),
                      fontsize=9, va="bottom", ha="left", zorder=121)
        self._badges.extend([ln, txt])
        self._artist_map[ln] = event_obj
        self._artist_map[txt] = event_obj

    def _draw_target_arrow(self, x: float, y: float, direction: str, e: object) -> None:
        target = getattr(e, "target_price", None)
        if target is None:
            return
        try:
            ty = float(target)
        except Exception:
            return
        ax = self.ax
        color = self._badge_color(direction)
        dy = ty - y
        if abs(dy) < 1e-12:
            dy = 0.0001  # evita zero-length
        # una freccia orizzontalmente corta, verticale verso target
        arr = ax.annotate(
            "", xy=(x, ty), xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color=color, shrinkA=0, shrinkB=0),
            zorder=119
        )
        # piccolo label del target
        lab = ax.text(x, ty, f"{ty:.5f}", fontsize=8, color=color,
                      va="bottom" if dy>0 else "top", ha="left",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.6, alpha=0.8),
                      zorder=119)
        self._arrows.extend([arr, lab])

    # ---------- Interaction (hover/click) ----------
    def _bind_canvas_events(self) -> None:
        ax = self._resolve_axes()
        if not ax or not ax.figure:
            return
        canvas = ax.figure.canvas
        if self._cid_move is None:
            self._cid_move = canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        if self._cid_click is None:
            self._cid_click = canvas.mpl_connect("button_press_event", self._on_mouse_click)

    def _hit_test(self, event) -> Optional[mpla.Artist]:
        """Ritorna l’artist più vicino entro HIT_RADIUS_PX, in coordinate pixel."""
        if event.inaxes is not self.ax:
            return None
        if not self._badges:
            return None
        ax = self.ax
        ex, ey = event.x, event.y
        best = None
        best_d2 = HIT_RADIUS_PX**2 + 1
        # Consideriamo solo i marker Line2D (più precisi per hit-test)
        for art in self._badges:
            if not hasattr(art, "get_position") and not hasattr(art, "get_xdata"):
                continue
            try:
                if hasattr(art, "get_xdata"):
                    xdata, ydata = art.get_xdata(), art.get_ydata()
                    if len(xdata) != 1:
                        continue
                    x, y = float(xdata[0]), float(ydata[0])
                else:
                    x, y = art.get_position()
                px, py = ax.transData.transform((x, y))
                d2 = (px - ex)**2 + (py - ey)**2
                if d2 < best_d2:
                    best_d2 = d2; best = art
            except Exception:
                continue
        if best is not None and best_d2 <= HIT_RADIUS_PX**2:
            return best
        return None

    def _ensure_tooltip(self):
        if self._tooltip is None and self.ax:
            self._tooltip = self.ax.annotate(
                "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", lw=0.6, alpha=0.85),
                color="white", fontsize=9, zorder=200
            )
            self._tooltip.set_visible(False)

    def _on_mouse_move(self, event) -> None:
        ax = self.ax
        if not ax or event.inaxes is not ax:
            if self._tooltip and self._tooltip.get_visible():
                self._tooltip.set_visible(False)
                try: ax.figure.canvas.draw_idle()
                except Exception: pass
            return

        art = self._hit_test(event)
        self._ensure_tooltip()
        if art is None:
            if self._tooltip.get_visible():
                self._tooltip.set_visible(False)
                ax.figure.canvas.draw_idle()
            self._last_hover_artist = None
            return

        if art is self._last_hover_artist and self._tooltip.get_visible():
            # niente da aggiornare
            return

        ev = self._artist_map.get(art)
        key = getattr(ev, "key", getattr(ev, "name", type(ev).__name__)) if ev else "pattern"
        self._tooltip.xy = (event.xdata, event.ydata)
        self._tooltip.set_text(str(key))
        self._tooltip.set_visible(True)
        self._last_hover_artist = art
        try:
            ax.figure.canvas.draw_idle()
        except Exception:
            pass

    def _on_mouse_click(self, event) -> None:
        ax = self.ax
        if not ax or event.inaxes is not ax or event.button != 1:
            return
        art = self._hit_test(event)
        if art is None:
            return
        ev = self._artist_map.get(art)
        if ev is None:
            return
        # prova la dialog custom del controller
        opener = getattr(self.controller, "open_pattern_info", None)
        if callable(opener):
            try:
                opener(ev)
                return
            except Exception as ex:
                logger.debug(f"open_pattern_info failed: {ex}")
        # fallback: QMessageBox con riassunto
        try:
            from PySide6 import QtWidgets
            ts = getattr(ev, "confirm_ts", getattr(ev, "ts", None))
            price = getattr(ev, "confirm_price", getattr(ev, "price", None))
            tprice = getattr(ev, "target_price", None)
            direction = getattr(ev, "direction", "neutral")
            kind = getattr(ev, "kind", "unknown")
            key = getattr(ev, "key", getattr(ev, "name", type(ev).__name__))
            # format timestamp
            t = self._to_ts(ts)
            t_str = str(t.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S")) if t is not None else str(ts)
            msg = (f"<b>{key}</b><br>"
                   f"Kind: {kind}<br>"
                   f"Direction: {direction}<br>"
                   f"Confirm: {t_str} @ {price}<br>"
                   f"Target: {'' if tprice is None else tprice}")
            QtWidgets.QMessageBox.information(None, f"Pattern: {key}", msg)
        except Exception as ex:
            logger.debug(f"fallback info dialog failed: {ex}")
