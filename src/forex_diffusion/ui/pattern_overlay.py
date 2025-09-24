
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.text import Text

# Qt is optional here; only used for the info dialog
try:
    from PySide6 import QtCore, QtWidgets
    _HAVE_QT = True
except Exception:
    _HAVE_QT = False


@dataclass
class PatternEvent:
    key: str
    name: str
    kind: str            # "chart" | "candle"
    direction: str       # "bull" | "bear" | "neutral"
    start_ts: int        # ms epoch
    confirm_ts: int      # ms epoch
    end_ts: Optional[int] = None
    confirm_price: Optional[float] = None
    target_price: Optional[float] = None
    info_html: Optional[str] = None
    # Optional trail for "formation line": list of (ts_ms, price)
    trail: Optional[List[Tuple[int, float]]] = None
    # Storage for UI (not persisted)
    _badge: Optional[FancyBboxPatch] = field(default=None, repr=False, compare=False)
    _label: Optional[Text] = field(default=None, repr=False, compare=False)


class PatternOverlayRenderer:
    """
    Draws pattern badges and formation lines onto a Matplotlib Axes.
    - set_axes(ax): provide axes to draw on
    - set_events(events, *, mode): provide events and redraw
    mode: "ms_epoch" (x are epoch ms) or "datetime"
    """
    def __init__(self):
        self._ax: Optional[Axes] = None
        self._canvas = None
        self._artists: List[Any] = []
        self._events: List[PatternEvent] = []
        self._mode: str = "ms_epoch"
        self._last_n_drawn: int = 0
        self._hover_annot: Optional[Text] = None
        self._mpl_cids: List[int] = []
        self._last_mode_log: Optional[str] = None  # for diagnostics

    # ------------- Axes / wiring -------------

    def set_axes(self, ax: Axes):
        """Attach axes and wire basic mpl events."""
        self._ax = ax
        self._canvas = ax.figure.canvas
        self._wire_mpl_events()
        # A redraw may be pending if events were set first
        self.draw(self._events or [])

    def _wire_mpl_events(self):
        if not self._canvas:
            return
        # disconnect previous
        for cid in self._mpl_cids:
            try:
                self._canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._mpl_cids.clear()

        def _on_motion(event):
            if event.inaxes is not self._ax or not self._events:
                return
            # simple nearest-badge hover
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            hit = None
            min_dx = float("inf")
            for ev in self._events:
                cx = ev.confirm_ts if self._mode == "ms_epoch" else ev.confirm_ts  # value used as-is
                cy = ev.confirm_price if ev.confirm_price is not None else y
                dx = abs(x - cx) + (abs(y - cy) if cy is not None else 0.0)
                if dx < min_dx and dx < 0.02 * (self._ax.get_ylim()[1] - self._ax.get_ylim()[0]):
                    hit = ev
                    min_dx = dx
            if hit:
                self._show_hover(event, hit)
            else:
                self._hide_hover()

        def _on_click(event):
            if event.button != 1 or event.inaxes is not self._ax or not self._events:
                return
            ev = self._pick_event_near(event.xdata, event.ydata)
            if ev:
                self._open_info_dialog(ev)

        self._mpl_cids.append(self._canvas.mpl_connect("motion_notify_event", _on_motion))
        self._mpl_cids.append(self._canvas.mpl_connect("button_press_event", _on_click))

    def _pick_event_near(self, x: float, y: float) -> Optional[PatternEvent]:
        best = None
        best_d = float("inf")
        for ev in self._events:
            cx = ev.confirm_ts
            cy = ev.confirm_price if ev.confirm_price is not None else y
            d = abs(x - cx) + (abs(y - cy) if cy is not None else 0.0)
            if d < best_d and d < 0.02 * (self._ax.get_ylim()[1] - self._ax.get_ylim()[0]):
                best = ev
                best_d = d
        return best

    # ------------- Public API -------------

    def set_events(self, events: List[PatternEvent], *, mode: str = "ms_epoch"):
        self._mode = mode or "ms_epoch"
        # normalize to PatternEvent
        norm: List[PatternEvent] = []
        for e in events:
            if isinstance(e, PatternEvent):
                norm.append(e)
            elif isinstance(e, dict):
                norm.append(PatternEvent(**e))
        self._events = norm
        self.draw(self._events)

    def draw(self, events: List[PatternEvent]):
        if self._ax is None:
            # axes not yet set; nothing to draw
            return
        self._clear_artists()

        n = 0
        for ev in events:
            try:
                self._draw_formation(ev)
                self._draw_badge(ev)
                n += 1
            except Exception:
                # don't crash drawing loop
                continue

        self._last_n_drawn = n
        if self._canvas:
            try:
                self._canvas.draw_idle()
            except Exception:
                pass

    # ------------- Drawing helpers -------------

    def _clear_artists(self):
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists.clear()

    def _draw_formation(self, ev: PatternEvent):
        if ev.trail and len(ev.trail) >= 2:
            xs = [t for (t, _p) in ev.trail]
            ys = [p for (_t, p) in ev.trail]
        else:
            # fallback: simple segment start->confirm if prices provided
            if ev.confirm_price is None:
                return
            xs = [ev.start_ts, ev.confirm_ts]
            ys = [ev.confirm_price, ev.confirm_price]

        # light blue, semi-transparent, thicker, behind price
        ln = Line2D(xs, ys, linewidth=3.0, alpha=0.35, color="#4FC3F7", zorder=1)
        self._ax.add_line(ln)
        self._artists.append(ln)

    def _draw_badge(self, ev: PatternEvent):
        # badge rectangle near confirm point
        if ev.confirm_price is None:
            return
        x = ev.confirm_ts
        y = ev.confirm_price
        label = ev.name or "Pattern"

        pad_x = 40_000  # ms to shift badge right
        bx = x + pad_x
        by = y

        color = "#2ECC71" if (ev.direction or "").lower().startswith("bull") else "#E74C3C"
        facecolor = color
        edgecolor = "black"

        width = max(60_000, 7_000 * max(1, len(label)))  # crude width in ms
        height = (self._ax.get_ylim()[1] - self._ax.get_ylim()[0]) * 0.03

        box = FancyBboxPatch(
            (bx, by),
            width, height,
            boxstyle="round,pad=0.02,rounding_size=0.015",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=0.6, alpha=0.75, zorder=4
        )
        self._ax.add_patch(box)
        self._artists.append(box)

        txt = self._ax.text(
            bx + width * 0.5, by + height * 0.5, label,
            ha="center", va="center", fontsize=8, color="white", zorder=5
        )
        self._artists.append(txt)
        ev._badge = box
        ev._label = txt

        # target arrow (always visible if present)
        if ev.target_price is not None:
            self._draw_target_arrow(ev)

    def _draw_target_arrow(self, ev: PatternEvent):
        x = ev.confirm_ts
        y0 = ev.confirm_price if ev.confirm_price is not None else None
        y1 = ev.target_price
        if y0 is None or y1 is None:
            return
        col = "#27AE60" if y1 >= y0 else "#C0392B"
        ln = Line2D([x, x], [y0, y1], linewidth=2.0, alpha=0.9, color=col, zorder=4)
        self._ax.add_line(ln)
        self._artists.append(ln)

    # ------------- Hover + Dialog -------------

    def _show_hover(self, mpl_event, ev: PatternEvent):
        label = ev.name or "Pattern"
        if self._hover_annot is None:
            self._hover_annot = self._ax.annotate(
                label, (ev.confirm_ts, ev.confirm_price or self._ax.get_ylim()[0]),
                xytext=(12, 12), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w", ec="0.4", alpha=0.9),
                fontsize=8, zorder=6
            )
            self._artists.append(self._hover_annot)
        else:
            self._hover_annot.set_text(label)
            self._hover_annot.set_position((ev.confirm_ts, ev.confirm_price or self._ax.get_ylim()[0]))
        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    def _hide_hover(self):
        if self._hover_annot is not None:
            try:
                self._hover_annot.remove()
            except Exception:
                pass
            self._hover_annot = None
            if self._canvas:
                try:
                    self._canvas.draw_idle()
                except Exception:
                    pass

    def _open_info_dialog(self, ev: PatternEvent):
        if not _HAVE_QT:
            return
        dlg = QtWidgets.QMessageBox(parent=None)
        dlg.setWindowTitle(ev.name or "Pattern")
        html = [
            f"<b>{ev.name}</b>",
            f"<div>Kind: {ev.kind} &nbsp; Direction: <b>{ev.direction}</b></div>",
        ]
        if ev.target_price is not None:
            html.append(f"<div>Target: <b>{ev.target_price:.5f}</b></div>")
        if ev.info_html:
            html.append("<hr/>")
            html.append(ev.info_html)
        dlg.setTextFormat(QtCore.Qt.TextFormat.RichText)
        dlg.setText("<br/>".join(html))
        dlg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        dlg.exec()
