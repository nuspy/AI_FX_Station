
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import math
import numpy as np

from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt, QObject, Slot

import matplotlib
import matplotlib.dates as mdates
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.text import Annotation


Number = Union[int, float, np.number]
TsType = Union[int, float, np.datetime64]

@dataclass
class PatternEvent:
    name: str
    kind: str                 # "chart" | "candle"
    direction: Optional[str]  # "bull" | "bear" | None
    ts_start: TsType
    ts_end: TsType
    confirm_ts: Optional[TsType] = None
    target_price: Optional[float] = None
    info_html: Optional[str] = None


class PatternOverlayRenderer(QObject):
    """
    Matplotlib overlay that draws:
      - a thick semi-transparent light-blue line under price during formation window [ts_start, ts_end]
      - a circular badge at confirm_ts with pattern name (green bull, red bear)
      - optional target as dashed horizontal line and arrow from confirm to target
    It is tolerant to x being either epoch-ms integers or matplotlib date floats.
    """
    def __init__(self, parent_widget: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent_widget)
        self._parent_widget = parent_widget
        self._ax: Optional[Axes] = None
        self._canvas = None
        self._events: List[PatternEvent] = []
        self._artists: List[Any] = []
        self._hover_annot: Optional[Annotation] = None
        self._is_panning: bool = False
        self._x_mode: str = "auto"   # "ms_epoch" | "mdates" | "auto"
        self._mpl_cids: List[int] = []
        self._last_drawn_count: int = 0

    # ---------- Public API ----------
    def set_axes(self, ax: Axes, canvas=None) -> None:
        """Attach to an Axes and its canvas; safe to call multiple times."""
        self._disconnect_mpl_events()
        self._ax = ax
        self._canvas = canvas or ax.figure.canvas
        self._connect_mpl_events()

    def set_events(self, events: List[Dict[str, Any]], x_mode: str = "auto") -> None:
        """Accept list of dict or PatternEvent and schedule a redraw."""
        parsed: List[PatternEvent] = []
        for e in events or []:
            if isinstance(e, PatternEvent):
                parsed.append(e)
            else:
                parsed.append(PatternEvent(**e))
        self._events = parsed
        self._x_mode = x_mode or "auto"
        self.draw()

    # ---------- Internals ----------
    # Utilities
    def _disconnect_mpl_events(self) -> None:
        if self._canvas and self._mpl_cids:
            for cid in self._mpl_cids:
                try:
                    self._canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._mpl_cids = []

    def _connect_mpl_events(self) -> None:
        if not self._canvas:
            return
        self._mpl_cids.append(self._canvas.mpl_connect("draw_event", self._on_draw))
        self._mpl_cids.append(self._canvas.mpl_connect("motion_notify_event", self._on_motion))
        self._mpl_cids.append(self._canvas.mpl_connect("button_press_event", self._on_press))
        self._mpl_cids.append(self._canvas.mpl_connect("button_release_event", self._on_release))
        self._mpl_cids.append(self._canvas.mpl_connect("pick_event", self._on_pick))

    def _clear_artists(self) -> None:
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists.clear()
        if self._hover_annot is not None:
            try:
                self._hover_annot.remove()
            except Exception:
                pass
            self._hover_annot = None

    def _price_line(self) -> Optional[Line2D]:
        if not self._ax:
            return None
        # Heuristic: pick the longest visible Line2D (often the price line)
        lines = [l for l in self._ax.lines if l.get_visible()]
        if not lines:
            return None
        return max(lines, key=lambda l: len(l.get_xdata(orig=False)))

    def _axis_mode(self) -> str:
        if self._x_mode != "auto":
            return self._x_mode
        line = self._price_line()
        if line is None:
            return "ms_epoch"
        x = np.asarray(line.get_xdata(orig=False))
        if x.size == 0:
            return "ms_epoch"
        # mdates are floats roughly ~ 20000
        return "mdates" if (np.issubdtype(x.dtype, np.floating_) and x.mean() > 1000) else "ms_epoch"

    def _to_axis_x(self, ts: TsType) -> Optional[float]:
        if ts is None:
            return None
        mode = self._axis_mode()
        # ts can be pandas Timestamp -> convert to ms then to mdates if needed.
        if hasattr(ts, "value"):  # pandas Timestamp
            # value is ns since epoch
            ts_ms = int(int(ts.value) // 1_000_000)
        elif isinstance(ts, (np.integer, int)):
            ts_ms = int(ts)
        elif isinstance(ts, (float, np.floating)):
            # already an axis value in mdates perhaps
            return float(ts)
        else:
            try:
                ts_ms = int(ts)  # hope implicit cast
            except Exception:
                return None
        if mode == "mdates":
            return mdates.epoch2num(ts_ms / 1000.0)
        return float(ts_ms)

    def _y_at(self, x_query: float) -> Optional[float]:
        """Interpolate approximate price y on price line at x_query (axis units)."""
        line = self._price_line()
        if line is None:
            return None
        x = np.asarray(line.get_xdata(orig=False), dtype=float)
        y = np.asarray(line.get_ydata(orig=False), dtype=float)
        if x.size < 2:
            return None
        idx = np.searchsorted(x, x_query)
        idx = np.clip(idx, 1, x.size - 1)
        x0, x1 = x[idx - 1], x[idx]
        y0, y1 = y[idx - 1], y[idx]
        if x1 == x0:
            return float(y0)
        t = (x_query - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    def _in_view_x(self, x: float) -> bool:
        if not self._ax:
            return False
        x0, x1 = self._ax.get_xlim()
        return (x >= min(x0, x1)) and (x <= max(x0, x1))

    def _badge_face(self, direction: Optional[str]) -> str:
        if direction == "bull":
            return "#2ecc71"  # green
        if direction == "bear":
            return "#e74c3c"  # red
        return "#95a5a6"      # gray

    def _draw_event(self, ev: PatternEvent) -> None:
        if not self._ax:
            return
        x_start = self._to_axis_x(ev.ts_start)
        x_end = self._to_axis_x(ev.ts_end)
        x_conf = self._to_axis_x(ev.confirm_ts) if ev.confirm_ts is not None else None

        if x_start is None or x_end is None:
            return
        if not (self._in_view_x(x_start) or self._in_view_x(x_end) or (x_conf is not None and self._in_view_x(x_conf))):
            # Skip drawing completely off-screen
            return

        # 1) Formation ribbon under price
        # draw by sampling price line between x_start..x_end
        line = self._price_line()
        if line is not None:
            x = np.asarray(line.get_xdata(orig=False), dtype=float)
            y = np.asarray(line.get_ydata(orig=False), dtype=float)
            mask = (x >= min(x_start, x_end)) & (x <= max(x_start, x_end))
            if mask.any():
                hl = Line2D(x[mask], y[mask], linewidth=3.5, color="#5dade2", alpha=0.35, zorder=1.0)
                self._ax.add_line(hl)
                self._artists.append(hl)

        # 2) Badge at confirmation with name
        if x_conf is not None:
            y_conf = self._y_at(x_conf) or (self._ax.get_ylim()[0] + self._ax.get_ylim()[1]) * 0.5
            circ = Circle((x_conf, y_conf), radius=0.0, transform=self._ax.transData)
            # Use annotate to get a filled rounded box
            color = self._badge_face(ev.direction)
            annot = self._ax.annotate(
                ev.name,
                xy=(x_conf, y_conf),
                xytext=(6, 12),
                textcoords="offset points",
                ha="left", va="bottom",
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.95),
                picker=True,
                zorder=5.0,
            )
            self._artists.extend([circ, annot])

        # 3) Target line + arrow
        if ev.target_price is not None and x_conf is not None:
            y_tgt = float(ev.target_price)
            tgt = self._ax.axhline(y_tgt, linestyle="--", linewidth=1.0, color="#7f8c8d", alpha=0.8, zorder=1.5)
            # arrow from confirmation to target
            y_conf = self._y_at(x_conf) or y_tgt
            col = "#27ae60" if (ev.direction == "bull") else "#c0392b"
            arr = FancyArrowPatch(
                (x_conf, y_conf), (x_conf, y_tgt),
                arrowstyle="-|>",
                mutation_scale=10,
                color=col,
                alpha=0.85,
                linewidth=1.2,
                zorder=2.0,
            )
            self._ax.add_artist(arr)
            self._artists.extend([tgt, arr])

    def draw(self, events: Optional[List[Dict[str, Any]]] = None) -> None:
        """Redraw overlay for current events; if events provided, replaces current first."""
        if events is not None:
            self.set_events(events)
            return
        if self._ax is None or self._canvas is None:
            return
        if self._is_panning:
            return  # don't redraw during pan, wait for release
        self._clear_artists()
        count = 0
        for ev in self._events:
            try:
                self._draw_event(ev)
                count += 1
                if count >= 80:
                    # Draw at most 80 badges for performance; prioritize recent (assumed last)
                    break
            except Exception:
                continue
        self._last_drawn_count = count
        try:
            self._canvas.draw_idle()
        except Exception:
            pass

    # ---------- Event handlers ----------
    def _on_draw(self, _evt=None) -> None:
        # ensure hover annotation is on top
        if self._hover_annot and self._hover_annot.get_visible():
            try:
                self._hover_annot.set_zorder(10.0)
            except Exception:
                pass

    def _on_press(self, evt) -> None:
        if evt.button == MouseButton.LEFT:
            self._is_panning = True

    def _on_release(self, evt) -> None:
        if evt.button == MouseButton.LEFT:
            self._is_panning = False
            # redraw after pan
            self.draw()

    def _on_pick(self, evt) -> None:
        # Clicking any badge annotation opens info dialog
        # We approximate by showing info for the closest event to x under cursor
        if not self._parent_widget or not self._events or evt.mouseevent is None:
            return
        x = evt.mouseevent.xdata
        if x is None:
            return
        chosen = None
        best = 1e18
        for ev in self._events:
            xc = self._to_axis_x(ev.confirm_ts) if ev.confirm_ts is not None else None
            if xc is None:
                continue
            d = abs(float(xc) - float(x))
            if d < best:
                best = d
                chosen = ev
        if chosen is None:
            return
        text = chosen.info_html or f"<b>{chosen.name}</b><br/>Pattern: {chosen.kind}<br/>Bias: {chosen.direction or '-'}"
        dlg = QtWidgets.QMessageBox(self._parent_widget)
        dlg.setIcon(QtWidgets.QMessageBox.Information)
        dlg.setWindowTitle(f"Pattern: {chosen.name}")
        dlg.setTextFormat(Qt.TextFormat.RichText)
        dlg.setText(text)
        dlg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        dlg.exec()

    def _on_motion(self, evt) -> None:
        if self._ax is None or evt.inaxes is None or evt.inaxes != self._ax:
            if self._hover_annot is not None:
                self._hover_annot.set_visible(False)
                try:
                    self._canvas.draw_idle()
                except Exception:
                    pass
            return
        x = evt.xdata
        if x is None or not self._events:
            return
        # show nearest event name within tolerance in pixels (~30px)
        tol_px = 30.0
        chosen = None
        best_px = 1e9
        for ev in self._events:
            xc = self._to_axis_x(ev.confirm_ts) if ev.confirm_ts is not None else None
            if xc is None:
                continue
            # data->pixel distance
            try:
                x_px, y_px = self._ax.transData.transform((xc, self._y_at(xc) or 0.0))
                d = abs(x_px - evt.x)
                if d < best_px and d <= tol_px:
                    best_px = d
                    chosen = ev
            except Exception:
                continue
        if chosen is None:
            if self._hover_annot is not None:
                self._hover_annot.set_visible(False)
                try:
                    self._canvas.draw_idle()
                except Exception:
                    pass
            return
        label = chosen.name
        if self._hover_annot is None:
            self._hover_annot = self._ax.annotate(
                label, xy=(evt.xdata, evt.ydata), xytext=(10, 10), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc="#34495e", ec="none", alpha=0.85),
                color="white", fontsize=8, zorder=10.0, visible=True)
        else:
            self._hover_annot.set_text(label)
            self._hover_annot.xy = (evt.xdata, evt.ydata)
            self._hover_annot.set_visible(True)
        try:
            self._canvas.draw_idle()
        except Exception:
            pass
