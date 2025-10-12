from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
# matplotlib removed - pattern overlay will be reimplemented with finplot
# import matplotlib.dates as mdates
# import matplotlib.axes as mpla
from PySide6.QtCore import Qt
# from matplotlib.backend_bases import MouseEvent
from loguru import logger
from PySide6 import QtWidgets

# Stub types for matplotlib removal
class mpla:
    class Axes: pass
    class Artist: pass

class MouseEvent: pass


class PyQtGraphAxesWrapper:
    """
    Wrapper that makes PyQtGraph PlotItem compatible with matplotlib Axes API
    used by PatternOverlayRenderer.
    """
    def __init__(self, plot_item):
        self.plot_item = plot_item
        self.lines = []  # Track added lines for matplotlib compatibility
        self.collections = []
        self.patches = []
        self.texts = []
        self._pattern_items = []  # Track pattern overlay items

    def get_xlim(self):
        """Get x-axis limits from PyQtGraph viewbox"""
        try:
            viewbox = self.plot_item.getViewBox()
            if viewbox:
                [[xmin, xmax], [ymin, ymax]] = viewbox.viewRange()
                return (xmin, xmax)
        except Exception:
            pass
        return (0, 100)

    def get_ylim(self):
        """Get y-axis limits from PyQtGraph viewbox"""
        try:
            viewbox = self.plot_item.getViewBox()
            if viewbox:
                [[xmin, xmax], [ymin, ymax]] = viewbox.viewRange()
                return (ymin, ymax)
        except Exception:
            pass
        return (0, 100)

    def plot(self, *args, **kwargs):
        """Add a line plot - matplotlib compatible signature"""
        import pyqtgraph as pg
        from PySide6.QtGui import QPen, QColor, QBrush
        from PySide6.QtCore import Qt

        # Parse matplotlib-style arguments
        if len(args) >= 2:
            x, y = args[0], args[1]
        else:
            x, y = [], []

        color = kwargs.get('color', '#FFFFFF')
        linewidth = kwargs.get('linewidth', 1)
        linestyle = kwargs.get('linestyle', '-')
        alpha = kwargs.get('alpha', 1.0)
        zorder = kwargs.get('zorder', 0)
        marker = kwargs.get('marker', None)
        markersize = kwargs.get('markersize', 5)
        markerfacecolor = kwargs.get('markerfacecolor', color)
        markeredgecolor = kwargs.get('markeredgecolor', 'black')
        markeredgewidth = kwargs.get('markeredgewidth', 1)
        picker = kwargs.get('picker', None)

        items = []

        # Handle marker (scatter plot)
        if marker and len(x) > 0:
            # Convert colors
            face_color = QColor(markerfacecolor)
            face_color.setAlphaF(alpha)
            edge_color = QColor(markeredgecolor)

            brush = QBrush(face_color)
            pen = QPen(edge_color)
            pen.setWidthF(markeredgewidth)

            # Create scatter plot
            scatter = pg.ScatterPlotItem(
                x=x, y=y,
                size=markersize * 2,  # PyQtGraph uses diameter, matplotlib uses radius-ish
                brush=brush,
                pen=pen,
                symbol='o'  # Circle marker
            )
            self.plot_item.addItem(scatter)
            self._pattern_items.append(scatter)
            items.append(scatter)

        # Handle line (if no marker or linestyle is specified)
        elif not marker or linestyle != 'None':
            # Convert matplotlib color to QColor
            qcolor = QColor(color)
            qcolor.setAlphaF(alpha)

            pen = QPen(qcolor)
            pen.setWidthF(linewidth)

            # Convert linestyle
            if linestyle == '--':
                pen.setStyle(Qt.PenStyle.DashLine)
            elif linestyle == ':':
                pen.setStyle(Qt.PenStyle.DotLine)
            elif linestyle == 'None':
                pen.setStyle(Qt.PenStyle.NoPen)

            # Create PyQtGraph curve
            if len(x) > 0:
                curve = pg.PlotCurveItem(x=x, y=y, pen=pen)
                self.plot_item.addItem(curve)
                self._pattern_items.append(curve)
                items.append(curve)

        # Fake line object for matplotlib compatibility
        class FakeLine:
            def __init__(self, xdata, ydata):
                self._xdata = xdata
                self._ydata = ydata
            def get_xdata(self):
                return self._xdata
            def get_ydata(self):
                return self._ydata
            def remove(self):
                pass  # Handled by clear_pattern_overlays

        fake_line = FakeLine(x, y)
        self.lines.append(fake_line)
        return [fake_line]

    def axhline(self, y, **kwargs):
        """Add horizontal line at y"""
        xmin, xmax = self.get_xlim()
        return self.plot([xmin, xmax], [y, y], **kwargs)

    def axvline(self, x, **kwargs):
        """Add vertical line at x"""
        ymin, ymax = self.get_ylim()
        return self.plot([x, x], [ymin, ymax], **kwargs)

    def annotate(self, text, xy, xytext=None, **kwargs):
        """Add arrow annotation - matplotlib compatible"""
        import pyqtgraph as pg
        from PySide6.QtGui import QColor, QPen
        from PySide6.QtCore import Qt

        arrowprops = kwargs.get('arrowprops', {})
        if not arrowprops:
            # Just text annotation without arrow
            return self.text(xy[0], xy[1], text, **kwargs)

        # Draw arrow from xytext to xy
        if xytext is None:
            xytext = xy

        x0, y0 = xytext
        x1, y1 = xy

        # Extract arrow properties
        color = arrowprops.get('color', '#FFFFFF')
        linewidth = arrowprops.get('lw', 1.2)
        alpha = kwargs.get('alpha', 1.0)

        # Convert color
        qcolor = QColor(color)
        qcolor.setAlphaF(alpha)
        pen = QPen(qcolor)
        pen.setWidthF(linewidth)

        # Create arrow
        arrow = pg.ArrowItem(
            angle=0,  # Will be rotated based on direction
            tipAngle=30,
            headLen=10,
            tailLen=None,
            pen=pen,
            brush=qcolor
        )

        # Calculate angle and position
        import math
        dx = x1 - x0
        dy = y1 - y0
        angle = math.degrees(math.atan2(dy, dx))

        arrow.setPos(x1, y1)
        arrow.setStyle(angle=angle)

        # Add line from start to arrow (plot returns list with fake line)
        line_result = self.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth, alpha=alpha)

        # Add arrow to plot
        self.plot_item.addItem(arrow)
        self._pattern_items.append(arrow)

        # Return fake annotation for compatibility
        class FakeAnnotation:
            def __init__(self, arrow_item):
                self._arrow = arrow_item
            def remove(self):
                # Cleanup handled by clear_pattern_overlays
                pass
            def set_visible(self, visible):
                # Support visibility toggle
                try:
                    self._arrow.setVisible(visible)
                except Exception:
                    pass

        return FakeAnnotation(arrow)

    def text(self, x, y, text, **kwargs):
        """Add text annotation - matplotlib compatible"""
        import pyqtgraph as pg
        from PySide6.QtGui import QColor

        fontsize = kwargs.get('fontsize', 10)
        color = kwargs.get('color', '#FFFFFF')
        zorder = kwargs.get('zorder', 0)
        va = kwargs.get('va', 'center')  # vertical alignment
        ha = kwargs.get('ha', 'center')  # horizontal alignment
        bbox = kwargs.get('bbox', None)  # Background box styling

        # Map matplotlib alignment to PyQtGraph anchor
        anchor_map = {
            ('center', 'center'): (0.5, 0.5),
            ('center', 'top'): (0.5, 0.0),
            ('center', 'bottom'): (0.5, 1.0),
            ('left', 'center'): (0.0, 0.5),
            ('right', 'center'): (1.0, 0.5),
            ('left', 'bottom'): (0.0, 1.0),
            ('right', 'bottom'): (1.0, 1.0),
        }
        anchor = anchor_map.get((ha, va), (0.5, 0.5))

        # Create text with HTML formatting if bbox is specified
        if bbox:
            # Extract background color from bbox
            bg_color = bbox.get('fc', bbox.get('facecolor', '#2196F3'))
            border_color = bbox.get('ec', bbox.get('edgecolor', 'black'))
            alpha_val = bbox.get('alpha', 0.95)

            # Create HTML-styled text with background
            html_text = f'<div style="background-color: {bg_color}; color: {color}; padding: 2px 5px; border: 1px solid {border_color}; border-radius: 3px; opacity: {alpha_val};">{text}</div>'
            text_item = pg.TextItem(html=html_text, anchor=anchor)
        else:
            text_item = pg.TextItem(text=str(text), anchor=anchor, color=QColor(color))

        text_item.setPos(x, y)
        self.plot_item.addItem(text_item)
        self._pattern_items.append(text_item)
        self.texts.append(text_item)

        # Return fake text object for matplotlib compatibility
        class FakeText:
            def __init__(self):
                pass
            def remove(self):
                pass

        return FakeText()

    def clear_pattern_overlays(self):
        """Remove all pattern overlay items"""
        for item in self._pattern_items:
            try:
                self.plot_item.removeItem(item)
            except Exception:
                pass
        self._pattern_items.clear()
        self.lines.clear()
        self.texts.clear()
        self.collections.clear()
        self.patches.clear()

    # Fake figure/canvas for compatibility
    @property
    def figure(self):
        class FakeFigure:
            class FakeCanvas:
                def draw_idle(self):
                    pass
            canvas = FakeCanvas()
        return FakeFigure()

    def transData(self):
        """Fake transform for compatibility"""
        class FakeTransform:
            def transform(self, points):
                return points
        return FakeTransform()

# ---------------- UI CONSTANTS ----------------
MAX_OVERLAYS = 80
MIN_SEP_BARS = 8
HIT_RADIUS_PX = 12  # hit-test raggio in pixel
FORMATION_LINE_ALPHA = 0.35
FORMATION_LINE_WIDTH = 3.0
FORMATION_LINE_COLOR = "#33A1FD"  # light blue

# --------------- RENDERER ---------------------

class PatternDetailsDialog(QtWidgets.QDialog):
    def __init__(self, parent, event, info_provider=None):
        super().__init__(parent)
        pattern_key = getattr(event, 'pattern_key', 'unknown')
        direction = getattr(event, 'direction', 'neutral').lower()

        self.setWindowTitle(f"Pattern: {pattern_key}")
        self.setMinimumSize(500, 400)

        lay = QtWidgets.QVBoxLayout(self)

        # Load pattern info from JSON
        pattern_info = self._load_pattern_info(pattern_key, info_provider)

        def lab(s, style=""):
            l = QtWidgets.QLabel(s)
            l.setTextFormat(Qt.TextFormat.RichText)
            l.setWordWrap(True)
            if style:
                l.setStyleSheet(style)
            return l

        # Pattern name and effect
        name = pattern_info.get('name', pattern_key)
        effect = pattern_info.get('effect', 'Unknown')
        effect_color = "#2ecc71" if effect == "Reversal" else "#3498db" if effect == "Continuation" else "#95a5a6"

        title_html = f'<h2 style="color: {effect_color};">{name}</h2>'
        title_html += f'<p style="color: {effect_color}; font-weight: bold; font-size: 14px;">{effect} Pattern</p>'
        lay.addWidget(lab(title_html))

        # Description
        description = pattern_info.get('description', 'No description available.')
        lay.addWidget(lab(f'<p style="font-size: 12px; color: #34495e;"><b>Description:</b> {description}</p>'))

        # Direction-specific notes
        direction_info = pattern_info.get(direction, {})
        if direction_info:
            notes = direction_info.get('Notes', '')
            if notes:
                lay.addWidget(lab(f'<p style="font-size: 12px; color: #8e44ad;"><b>{direction.title()} Direction:</b> {notes}</p>'))

        # Performance benchmarks
        benchmarks = pattern_info.get('benchmarks', {})
        if benchmarks:
            bench_html = '<h3 style="color: #2c3e50;">Performance Statistics:</h3><table style="width:100%; border-collapse: collapse;">'
            for key, value in benchmarks.items():
                bench_html += f'<tr><td style="padding: 2px; border-bottom: 1px solid #bdc3c7;"><b>{key}:</b></td><td style="padding: 2px; border-bottom: 1px solid #bdc3c7;">{value}</td></tr>'
            bench_html += '</table>'
            lay.addWidget(lab(bench_html))

        # Event details
        event_html = '<h3 style="color: #2c3e50;">Event Details:</h3>'
        fields = [
            ("Kind", getattr(event,'kind','')),
            ("Direction", getattr(event,'direction','')),
            ("Confirm Time", str(getattr(event,'confirm_ts',''))),
            ("Target Price", str(getattr(event,'target_price', 'N/A'))),
            ("Failure Price", str(getattr(event,'failure_price', 'N/A')))
        ]

        event_html += '<table style="width:100%; border-collapse: collapse;">'
        for k, v in fields:
            event_html += f'<tr><td style="padding: 2px; border-bottom: 1px solid #bdc3c7;"><b>{k}:</b></td><td style="padding: 2px; border-bottom: 1px solid #bdc3c7;">{v}</td></tr>'
        event_html += '</table>'
        lay.addWidget(lab(event_html))

        # Close button
        btn = QtWidgets.QPushButton("Chiudi")
        btn.clicked.connect(self.accept)
        btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; padding: 8px 16px; border: none; border-radius: 4px; } QPushButton:hover { background-color: #2980b9; }")
        lay.addWidget(btn)

    def _load_pattern_info(self, pattern_key, info_provider):
        """Load pattern information from JSON configuration"""
        try:
            import json
            import os

            # Try to use info_provider first
            if info_provider and hasattr(info_provider, 'get_pattern_info'):
                info = info_provider.get_pattern_info(pattern_key)
                if info:
                    return info

            # Fallback: load directly from config file
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'configs', 'pattern_info.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    pattern_data = json.load(f)
                    return pattern_data.get(pattern_key, {})
        except Exception as e:
            print(f"Error loading pattern info: {e}")

        # Default fallback
        return {
            'name': pattern_key,
            'effect': 'Unknown',
            'description': 'Pattern information not available.',
            'benchmarks': {},
            'bull': {},
            'bear': {}
        }

class PatternOverlayRenderer:
    """
    Renderizza:
      - badge con nome pattern (colore: verde bull, rosso bear),
      - tooltip on-hover col nome pattern,
      - click → dialog dettagli (usa controller.open_pattern_info se disponibile),
      - freccia target (se target_price),
      - linea di formazione del pattern: segmento della price-line dietro al prezzo
        (azzurro semi-trasparente), o fallback ad axvspan se nessuna price-line.
    Supporta asse X in ms-epoch o in date2num; autodetect.
    """
    def __init__(self, controller, info_provider=None):
        self.controller = controller
        self.info = info_provider
        # prova a trovare un asse prezzo noto
        self.ax: Optional[mpla.Axes] = getattr(controller, "axes_price", None) \
                                       or getattr(getattr(controller, "view", None), "ax_price", None)
        # Stato grafico
        self._badges: List[mpla.Artist] = []
        self._arrows: List[mpla.Artist] = []
        self._formations: List[mpla.Arist] = []  # typo fixed below
        self._formations: List[mpla.Artist] = []
        self._artist_map: Dict[mpla.Axes, object] = {}  # types corrected below
        self._artist_map: Dict[mpla.Artist, object] = {}
        self._last_mode_log: Optional[str] = None

        # Interazione
        self._cid_move = None
        self._cid_click = None
        self._tooltip = None
        self._last_hover_artist: Optional[mpla.Axes] = None  # types corrected below
        self._last_hover_artist: Optional[mpla.Artist] = None

    # ---------- Public API ----------
    def set_axes(self, ax: mpla.Axes) -> None:
        self.ax = ax
        self._bind_canvas_events()

    def clear(self) -> None:
        self._clear_all()
        if self.ax and self.ax.figure:
            try: self.ax.figure.canvas.draw_idle()
            except Exception: pass

    def _draw_target_failure(self, ax, e, x_pos):
        try:
            t=getattr(e,'target_price',None); f=getattr(e,'failure_price',None)
            if t is not None: ax.axhline(t, linestyle='--', linewidth=1, zorder=2); ax.text(x_pos,t,f"T: {t:.5f}",fontsize=8,va='bottom',zorder=3)
            if f is not None: ax.axhline(f, linestyle=':', linewidth=1, zorder=2); ax.text(x_pos,f,f"F: {f:.5f}",fontsize=8,va='top',zorder=3); ax.plot([x_pos,x_pos],[getattr(e,'price',f),f], color='black', linewidth=1, zorder=2)
        except Exception: pass
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

        # Disegna in ordine: formazione (dietro), poi badge/testo + target (davanti)
        for x, y, label, kind, direction, e in kept:
            try:
                self._draw_formation_segment(x, e)  # dietro
            except Exception as ex:
                logger.debug(f"formation draw failed: {ex}")

        for x, y, label, kind, direction, e in kept:
            try:
                self._draw_badge(x, y, label, direction, e)  # davanti
                self._draw_target_arrow(x, y, direction, e)  # davanti
                self._draw_invalidation_arrow(x, y, direction, e)  # freccia invalidazione
                self._draw_invalidation_timeline(x, y, direction, e)  # linea grigia 50% tempo max invalidazione
            except Exception as ex:
                logger.debug(f"overlay draw_event failed: {ex}")

        self._bind_canvas_events()
        try: ax.figure.canvas.draw_idle()
        except Exception: pass

    # ---------- Axes & Time Helpers ----------
    def _resolve_axes(self) -> Optional[mpla.Axes]:
        # PYQTGRAPH SUPPORT: Check for main_plot first (PyQtGraph)
        def _find_pyqtgraph_plot(obj):
            if not obj:
                return None
            # Check for main_plot attribute (PyQtGraph PlotItem)
            for name in ("main_plot", "plot", "plot_widget"):
                plot = getattr(obj, name, None)
                if plot and hasattr(plot, 'addItem'):  # PyQtGraph PlotItem has addItem
                    return plot
            return None

        # Try to find PyQtGraph plot
        pyqt_plot = (_find_pyqtgraph_plot(self.controller) or
                     _find_pyqtgraph_plot(getattr(self.controller, "plot_service", None)) or
                     _find_pyqtgraph_plot(getattr(self.controller, "view", None)))

        if pyqt_plot:
            # Store the PyQtGraph plot wrapped in a compatible object
            if not hasattr(self, '_pyqtgraph_wrapper'):
                self._pyqtgraph_wrapper = PyQtGraphAxesWrapper(pyqt_plot)
            self.ax = self._pyqtgraph_wrapper
            return self.ax

        # MATPLOTLIB SUPPORT (legacy fallback - currently not used)
        if getattr(self, "ax", None) is not None and getattr(self.ax, "figure", None) is not None:
            return self.ax

        candidates: List[mpla.Axes] = []

        def _collect(obj):
            if not obj: return
            for name in ("ax_price", "axes_price", "price_ax", "ax", "axes"):
                a = getattr(obj, name, None)
                if isinstance(a, mpla.Axes): candidates.append(a)
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
        except Exception: pass
        xmin, xmax = ax.get_xlim()
        if xmax > 1e10: return "ms_epoch"
        if 1e3 < xmax < 1e7: return "matplotlib_date"
        return "index"

    def _to_ts(self, ts):
        try:
            if isinstance(ts, (pd.Timestamp, np.datetime64)):
                t = pd.Timestamp(ts)
            elif isinstance(ts, (int, np.integer, float)) and ts > 1e11:
                t = pd.to_datetime(int(ts), unit="ms", utc=True)
            elif isinstance(ts, (int, np.integer, float)) and ts > 1e9:
                t = pd.to_datetime(int(ts), unit="s", utc=True)
            else:
                t = pd.to_datetime(ts, utc=True)
            return t.tz_convert("UTC")
        except Exception:
            return None

    def _x_from_ts_bimode(self, ts):
        t = self._to_ts(ts)
        if t is None:
            return (np.nan, np.nan)
        # mdates removed - using timestamp conversion
        x_date = t.timestamp()  # Unix timestamp in seconds
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
            x = x_date
        elif in_ms and not in_date:
            x = x_ms
        else:
            x = x_date if d_date <= d_ms else x_ms
        return x

    # ---------- Event normalization & density ----------
    def _pattern_label(self, e: object) -> str:
        # Prefer stable id, then human title from info provider
        cand_keys = ("name", "key", "pattern_name", "pattern", "label", "kind")
        raw = None
        for k in cand_keys:
            if isinstance(e, dict) and k in e and e[k]:
                raw = str(e[k]); break
            v = getattr(e, k, None)
            if v:
                raw = str(v); break
        if not raw:
            raw = type(e).__name__
        # Try to map to human-friendly title via info provider
        try:
            if getattr(self, "info", None):
                meta = None
                if hasattr(self.info, "get"):
                    meta = self.info.get(raw)
                if meta is None and hasattr(self.info, "get_info"):
                    meta = self.info.get_info(raw)
                if isinstance(meta, dict):
                    title = meta.get("title") or meta.get("name") or meta.get("label")
                    if title:
                        return str(title)
        except Exception:
            pass
        return raw

    def _direction_color(self, direction: str, effect: str = None) -> str:
        """
        Return color based on direction and pattern effect:
        - Green for bullish/up movement (reversal or continuation)
        - Red for bearish/down movement (reversal or continuation)
        - Blue for neutral/continuation patterns
        """
        d = str(direction or "").lower()
        e = str(effect or "").lower()

        if d in ("up", "bull", "bullish", "long", "buy"):
            return "#2ecc71"  # green for bullish
        elif d in ("down", "bear", "bearish", "short", "sell"):
            return "#e74c3c"  # red for bearish
        elif e in ("continuation", "continue"):
            return "#3498db"  # blue for continuation
        else:
            return "#3498db"  # default blue for neutral

    def _normalize_events(self, ax: mpla.Axes, evs):
        norm = []
        for e in evs:
            label = self._pattern_label(e)
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
                norm.append((x, y, label, kind, direction, e))

        try:
            norm.sort(key=lambda t: float(t[0]), reverse=True)
        except Exception:
            pass
        return norm

    def _density_filter(self, ax: mpla.Axes, norm):
        kept = []
        picked_lbl = set()
        last_x_by_lbl: Dict[str, float] = {}
        try:
            xd = ax.lines[0].get_xdata()
            step = float(xd[-1]) - float(xd[-2]) if len(xd) > 2 else 1.0
        except Exception:
            step = 1.0

        for x, y, label, kind, direction, e in norm:
            if label not in picked_lbl:
                kept.append((x, y, label, kind, direction, e))
                picked_lbl.add(label); last_x_by_lbl[label] = x
            else:
                if abs(x - last_x_by_lbl.get(label, -1e18)) < max(step * MIN_SEP_BARS, 1.0):
                    continue
                last_x_by_lbl[label] = x
                kept.append((x, y, label, kind, direction, e))
            if len(kept) >= MAX_OVERLAYS:
                break
        return kept

    # ---------- Drawing ----------
    def _clear_all(self) -> None:
        # PyQtGraph support: use wrapper's clear method if available
        if hasattr(self.ax, 'clear_pattern_overlays'):
            self.ax.clear_pattern_overlays()

        # Matplotlib support (legacy)
        for art in self._badges + self._arrows + self._formations:
            try: art.remove()
            except Exception: pass
        self._badges.clear(); self._arrows.clear(); self._formations.clear()
        self._artist_map.clear()
        if self._tooltip is not None:
            try: self._tooltip.remove()
            except Exception: pass
        self._tooltip = None
        self._last_hover_artist = None

    def _draw_badge(self, x: float, y: float, label: str, direction: str, event_obj: object) -> None:
        ax = self.ax
        color = self._direction_color(direction)
        ms = 9.0
        ln = ax.plot([x], [y], marker="o", markersize=ms, markerfacecolor=color,
                     markeredgecolor="black", markeredgewidth=0.8, zorder=120, picker=HIT_RADIUS_PX)[0]
        txt = ax.text(x, y, f" {label} ", color="white",
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
        color = self._direction_color(direction)
        dy = ty - y
        if abs(dy) < 1e-12:
            dy = 0.0001
        arr = ax.annotate(
            "", xy=(x, ty), xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color=color, shrinkA=0, shrinkB=0),
            zorder=119
        )
        lab = ax.text(x, ty, f"{ty:.5f}", fontsize=8, color=color,
                      va="bottom" if dy>0 else "top", ha="left",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.6, alpha=0.8),
                      zorder=119)
        self._arrows.extend([arr, lab])

    def _draw_invalidation_arrow(self, x: float, y: float, direction: str, e: object) -> None:
        """Draw black vertical arrow indicating pattern invalidation point"""
        failure_price = getattr(e, "failure_price", None)
        if failure_price is None:
            return
        try:
            fp = float(failure_price)
        except Exception:
            return

        ax = self.ax
        # Draw black vertical arrow pointing down to invalidation level
        arrow = ax.annotate(
            "", xy=(x, fp), xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color='black', shrinkA=0, shrinkB=0),
            zorder=120
        )

        # Add small label for invalidation point
        label = ax.text(x, fp, f"INV", fontsize=7, color='black',
                       va="bottom" if fp < y else "top", ha="center",
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.5, alpha=0.9),
                       zorder=120)

        self._arrows.extend([arrow, label])

    def _draw_invalidation_timeline(self, x: float, y: float, direction: str, e: object) -> None:
        """Draw gray 50% opacity line indicating maximum invalidation time"""
        horizon_bars = getattr(e, "horizon_bars", None)
        if horizon_bars is None:
            return

        try:
            ax = self.ax
            if not ax:
                return

            # Calculate the end time for invalidation (50% of horizon_bars from confirmation)
            confirm_ts = getattr(e, "confirm_ts", None)
            if confirm_ts is None:
                return

            # Get the time array to calculate the invalidation end time
            xlim = ax.get_xlim()

            # Convert horizon_bars to time units (approximate)
            # Assuming each bar represents a time unit based on the current timeframe
            horizon_value = float(horizon_bars) if not isinstance(horizon_bars, dict) else 40  # Default fallback
            invalidation_time_offset = horizon_value * 0.5  # 50% of the horizon

            # Calculate end x position for the timeline
            start_x = x
            end_x = start_x + invalidation_time_offset

            # Make sure end_x is within plot bounds
            if end_x > xlim[1]:
                end_x = xlim[1]

            # Draw horizontal gray line at current price level
            failure_price = getattr(e, "failure_price", None)
            if failure_price:
                try:
                    fp = float(failure_price)

                    # Draw horizontal line from confirmation time to 50% invalidation time
                    line = ax.plot([start_x, end_x], [fp, fp],
                                 color='gray', linewidth=1.5, alpha=0.5,
                                 linestyle='--', zorder=110)

                    # Add small label at the end
                    label = ax.text(end_x, fp, "50%", fontsize=6, color='gray',
                                  va="bottom", ha="left", alpha=0.7,
                                  bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                          ec="gray", lw=0.5, alpha=0.5),
                                  zorder=110)

                    self._arrows.extend(line + [label])

                except (ValueError, TypeError):
                    pass

        except Exception as e:
            logger.debug(f"Error drawing invalidation timeline: {e}")

    def _event_window(self, e: object):
        ax = self.ax
        if not ax: return None
        start = getattr(e, "start_ts", None) or getattr(e, "begin_ts", None) \
                or getattr(e, "left_ts", None) or getattr(e, "formation_start_ts", None)
        end   = getattr(e, "end_ts", None)   or getattr(e, "finish_ts", None) \
                or getattr(e, "right_ts", None) or getattr(e, "formation_end_ts", None)

        if start is None and end is None:
            confirm = getattr(e, "confirm_ts", None)
            lookback = getattr(e, "lookback", None) or getattr(e, "window", None)
            if confirm is None or lookback is None:
                return None
            x_c = self._x_from_ts_safest(ax, confirm)
            try:
                xd = ax.lines[0].get_xdata()
                if len(xd) >= 3:
                    step = float(xd[-1]) - float(xd[-2])
                else:
                    step = 1.0
            except Exception:
                step = 1.0
            x0 = x_c - step * float(lookback)
            x1 = x_c
            return (x0, x1)

        if start is None:
            start = getattr(e, "ts", None)
        if end is None:
            end = getattr(e, "confirm_ts", None) or getattr(e, "ts", None)

        if start is None or end is None:
            return None

        x0 = self._x_from_ts_safest(ax, start)
        x1 = self._x_from_ts_safest(ax, end)
        if not np.isfinite(x0) or not np.isfinite(x1):
            return None
        if x1 < x0: x0, x1 = x1, x0
        return (x0, x1)

    def _get_main_price_line(self):
        # Prefer the underlying price line from PlotService even if hidden
        try:
            ps = getattr(self.controller, "plot_service", None)
            pl = getattr(ps, "_price_line", None)
            if pl is not None and hasattr(pl, "get_xdata") and len(pl.get_xdata()) >= 2:
                return pl
        except Exception:
            pass
        ax = self.ax
        if not ax:
            return None
        try:
            if ax.lines:
                # pick the line with most points
                lines = [ln for ln in ax.lines if len(getattr(ln, "get_xdata", lambda:[])()) >= 2]
                if lines:
                    return max(lines, key=lambda ln: len(ln.get_xdata()))
        except Exception:
            pass
        return None

    def _draw_formation_segment(self, x_confirm: float, e: object) -> None:
        ax = self.ax
        if not ax:
            return
        win = self._event_window(e)
        if not win:
            return
        x0, x1 = win
        if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
            return

        price_line = self._get_main_price_line()
        if price_line is None:
            # No vertical bands fallback: if price-line is unavailable, skip drawing
            return

        xd = np.asarray(price_line.get_xdata(), dtype=float)
        yd = np.asarray(price_line.get_ydata(), dtype=float)
        mask = (xd >= x0) & (xd <= x1)
        if mask.sum() < 2:
            return

        # Thicker than the price line and color by direction and effect
        try:
            base_lw = float(getattr(price_line, "get_linewidth", lambda: 1.2)())
        except Exception:
            base_lw = 1.2
        linewidth = max(base_lw * 2.0, base_lw + 2.0, 4.0)  # Make formation line more prominent

        direction = getattr(e, "direction", None)
        effect = getattr(e, "effect", None)
        color = self._direction_color(direction, effect)
        z = max(price_line.get_zorder() - 1, 1)

        ln, = ax.plot(xd[mask], yd[mask],
                      color=color, linewidth=linewidth,
                      alpha=max(FORMATION_LINE_ALPHA, 0.5), zorder=z)
        self._formations.append(ln)

    # ---------- Interaction ----------
    def _bind_canvas_events(self) -> None:
        ax = self._resolve_axes()
        if not ax or not ax.figure:
            return
        canvas = ax.figure.canvas
        if self._cid_move is None:
            self._cid_move = canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        if self._cid_click is None:
            self._cid_click = canvas.mpl_connect("button_press_event", self._on_mouse_click)

    def _hit_test(self, event: MouseEvent) -> Optional[mpla.Artist]:
        if event.inaxes is not self.ax:
            return None
        if not self._badges:
            return None
        ax = self.ax
        ex, ey = event.x, event.y
        best = None
        best_d2 = HIT_RADIUS_PX**2 + 1
        for art in self._badges:
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

    def _on_mouse_move(self, event: MouseEvent) -> None:
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
            return

        ev = self._artist_map.get(art)
        label = self._pattern_label(ev) if ev is not None else "pattern"
        self._tooltip.xy = (event.xdata, event.ydata)
        self._tooltip.set_text(str(label))
        self._tooltip.set_visible(True)
        self._last_hover_artist = art
        try: ax.figure.canvas.draw_idle()
        except Exception: pass

    def _on_mouse_click(self, event: MouseEvent) -> None:
        ax = self.ax
        if not ax or event.inaxes is not ax or event.button != 1:
            return
        art = self._hit_test(event)
        if art is None:
            return
        ev = self._artist_map.get(art)
        if ev is None:
            return
        opener = getattr(self.controller, "open_pattern_info", None)
        if callable(opener):
            try:
                opener(ev); return
            except Exception:
                pass
        self._fallback_info_dialog(ev)

    def _fallback_info_dialog(self, ev: object) -> None:
        try:
            from PySide6 import QtWidgets
            from PySide6.QtCore import Qt
        except Exception:
            return

        # Use new enhanced dialog with pattern info
        try:
            dialog = PatternDetailsDialog(self.ax.figure.canvas.parent(), ev, self.info)
            dialog.exec()
            return
        except Exception as e:
            print(f"Error showing enhanced dialog: {e}")

        # Fallback to old simple dialog
        def _esc(x):
            try: return str(x).replace("<","&lt;").replace(">","&gt;")
            except Exception: return str(x)

        key = self._pattern_label(ev)
        kind = getattr(ev, "kind", "unknown")
        direction = getattr(ev, "direction", "neutral")
        ts = getattr(ev, "confirm_ts", getattr(ev, "ts", None))
        price = getattr(ev, "confirm_price", getattr(ev, "price", None))
        tprice = getattr(ev, "target_price", None)

        t = self._to_ts(ts)
        t_str = str(t.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S")) if t is not None else str(ts)

        html = (
            f"<h3 style='margin:0'>{_esc(key)}</h3>"
            f"<p><b>Kind:</b> {_esc(kind)} &nbsp; <b>Direction:</b> {_esc(direction)}</p>"
            f"<p><b>Confirm:</b> {_esc(t_str)} @ {_esc(price)}</p>"
            f"<p><b>Target:</b> {_esc('' if tprice is None else tprice)}</p>"
        )

        dlg = PatternDetailsDialog(getattr(self.controller, "window", None) or None)
        dlg.setWindowTitle(f"Pattern: {key}")
        dlg.setTextFormat(Qt.TextFormat.RichText)
        dlg.setText(html)
        dlg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        dlg.exec()

    def on_pick(self, event):
        """Handle pick events from matplotlib"""
        try:
            if hasattr(event, 'artist') and hasattr(event.artist, '_pattern_info'):
                pattern_info = event.artist._pattern_info
                # Open pattern details dialog
                dialog = PatternDetailsDialog(
                    parent=getattr(self.controller, 'view', None),
                    event=pattern_info,
                    info_provider=self.info
                )
                dialog.exec()
        except Exception as e:
            logger.debug(f"Error handling pick event: {e}")
