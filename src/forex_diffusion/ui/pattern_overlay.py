from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, Circle
from PySide6.QtWidgets import QMessageBox
from ..patterns.engine import PatternEvent
from ..patterns.info_provider import PatternInfoProvider

if TYPE_CHECKING:
    from .chart_components.controllers.chart_controller import ChartTabController

class PatternOverlayRenderer:
    def __init__(self, controller: "ChartTabController", info_provider: Optional[PatternInfoProvider]=None) -> None:
        self.controller = controller
        self.view = controller.view
        self.ax = getattr(self.view, "ax", None)
        self.info_provider = info_provider
        self._badge_artists = []

    def clear(self):
        for art in list(self._badge_artists):
            try:
                art.remove()
            except Exception:
                pass
        self._badge_artists.clear()

    def draw(self, events: List[PatternEvent]):
        ax = self.ax
        if ax is None or not events:
            return
        self.clear()
        df = getattr(self.controller.plot_service, "_last_df", None)
        if df is None:
            return
        for ev in events:
            # lines for chart patterns
            if ev.kind == "chart":
                up = ev.overlay.get("upper_line", None)
                lo = ev.overlay.get("lower_line", None)
                if up and lo:
                    i0, i1, m_hi, b_hi = up
                    j0, j1, m_lo, b_lo = lo
                    xs = list(range(i0, i1+1))
                    ys_hi = [m_hi*x + b_hi for x in xs]
                    ys_lo = [m_lo*x + b_lo for x in xs]
                    try:
                        ts = mdates.date2num(df.loc[i0:i1,"ts_dt"])  # precomputed in plot_service
                    except Exception:
                        ts = mdates.date2num(df.loc[i0:i1,"ts_utc"].pipe(lambda s: s.astype('int64')).pipe(lambda s: s))
                    ax.plot(ts, ys_hi, linewidth=1.0, alpha=0.7, color="#8888ff")
                    ax.plot(ts, ys_lo, linewidth=1.0, alpha=0.7, color="#8888ff")
            # target arrow
            if ev.target_price is not None:
                # arrow from confirm_ts to target
                try:
                    tnum = mdates.date2num(ev.confirm_ts)
                    dy = float(ev.target_price) - float(df.iloc[-1]["close"])
                    color = "#2ecc71" if dy>0 else "#e74c3c"
                    arr = FancyArrowPatch((tnum, df.iloc[-1]["close"]), (tnum, ev.target_price),
                                          mutation_scale=12, arrowstyle="-|>", color=color, alpha=0.8, linewidth=1.4)
                    ax.add_artist(arr)
                except Exception:
                    pass
            # info badge
            try:
                tnum = mdates.date2num(ev.confirm_ts)
                y = float(df.iloc[-1]["close"])
                circ = Circle((tnum, y), radius=0.0008, color="#3498db", alpha=0.9, picker=True)
                ax.add_artist(circ)
                self._badge_artists.append(circ)
                circ._pattern_key = ev.pattern_key  # type: ignore
            except Exception:
                pass
        try:
            self.view.canvas.draw_idle()
        except Exception:
            self.view.canvas.draw()

    def on_pick(self, event):
        art = getattr(event, "artist", None)
        key = getattr(art, "_pattern_key", None)
        if not key:
            return
        info = self.info_provider.describe(key) if self.info_provider else None
        text = f"Pattern: {key}"
        if info:
            text = f"{info.name} ({info.effect})\n\n{info.description}\n\nBenchmarks:\n"
            for k,v in info.benchmarks.items():
                text += f" - {k}: {v}\n"
            text += "\nBull: " + ", ".join(f"{k}:{v}" for k,v in info.bull.items())
            text += "\nBear: " + ", ".join(f"{k}:{v}" for k,v in info.bear.items())
        QMessageBox.information(self.view, "Pattern info", text)
