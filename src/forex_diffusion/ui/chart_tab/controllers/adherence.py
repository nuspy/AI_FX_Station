from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import QObject


class AdherenceBadgeController(QObject):
    def __init__(self, chart_tab):
        super().__init__(chart_tab)
        self.view = chart_tab
        if not hasattr(self.view, "_adh_badges"):
            setattr(self.view, "_adh_badges", {})

    def draw_badge(self, future_ts: Sequence[float], median: Sequence[float], upper: Sequence[float], adherence: Optional[float]):
        try:
            if getattr(self.view, "plot", None) is None:
                return
            import numpy as _np
            if adherence is None:
                return
            fut_ts = _np.asarray(future_ts, dtype=float)
            m_vals = _np.asarray(median, dtype=float)
            q95 = _np.asarray(upper, dtype=float)
            if not (fut_ts.size and m_vals.size and q95.size and fut_ts.size == q95.size == m_vals.size):
                return
            ax = self.view.plot.getPlotItem()
            x_last = float(fut_ts[-1]); y_last = float(q95[-1])
            # position with margins
            try:
                vb = ax.vb
                xmin, xmax = vb.viewRange()[0]
                ymin, ymax = vb.viewRange()[1]
                dx = max(1.0, 0.015 * (xmax - xmin))
                dy = 0.015 * (ymax - ymin)
                x_text = min(x_last, xmax - dx); x_text = max(x_text, xmin + dx)
                y_text = min(max(y_last, ymin + dy), ymax - dy)
            except Exception:
                x_text, y_text = x_last, y_last
            # make key stable per-source if available
            try:
                label_key = str(getattr(self.view, 'current_source_label', 'forecast'))
            except Exception:
                label_key = "forecast"
            old = self.view._adh_badges.get(label_key)
            if old is not None:
                try:
                    self.view.plot.removeItem(old)
                except Exception:
                    pass
            try:
                from pyqtgraph import TextItem
                badge = TextItem(text=f"{float(adherence):.2f}", anchor=(0,0.5), color=(0,0,0))
                badge.setPos(x_text, y_text)
                self.view.plot.addItem(badge)
                self.view._adh_badges[label_key] = badge
            except Exception:
                pass
        except Exception:
            pass


