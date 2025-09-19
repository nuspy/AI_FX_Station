from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QEvent


def _get_viewbox(plot):
    try:
        return plot.getViewBox()
    except Exception:
        return None


class ZoomController(QObject):
    def __init__(self, chart_tab):
        super().__init__(chart_tab)
        self.view = chart_tab
        self._install()

    def _install(self):
        try:
            if getattr(self.view, "plot", None) is None:
                return
            vb = _get_viewbox(self.view.plot)
            if vb is not None:
                # enable default interactions (pan/zoom)
                vb.setMouseEnabled(x=True, y=True)
            # install wheel axis-zoom filter (CTRL -> X only, SHIFT -> Y only)
            self.view.plot.scene().installEventFilter(self)
        except Exception:
            pass

    def zoom_x(self, center: float, factor: float):
        try:
            if getattr(self.view, "plot", None) is None:
                return
            vb = _get_viewbox(self.view.plot)
            if vb is not None:
                vb.scaleBy((factor, 1.0), center=(center, 0))
        except Exception:
            pass

    def zoom_y(self, center: float, factor: float):
        try:
            if getattr(self.view, "plot", None) is None:
                return
            vb = _get_viewbox(self.view.plot)
            if vb is not None:
                vb.scaleBy((1.0, factor), center=(0, center))
        except Exception:
            pass

    def eventFilter(self, obj, event):
        # CTRL+Wheel -> X-only; SHIFT+Wheel -> Y-only
        try:
            if event.type() == QEvent.GraphicsSceneWheel:
                vb = _get_viewbox(self.view.plot)
                if vb is None:
                    return False
                mods = int(getattr(event, 'modifiers', lambda: 0)())
                # Fallback if modifiers not available on graphics scene wheel
                ctrl = bool(mods & 0x04000000)  # Qt.ControlModifier
                shift = bool(mods & 0x02000000)  # Qt.ShiftModifier
                if ctrl or shift:
                    delta = float(getattr(event, 'delta', lambda: 0)()) if hasattr(event, 'delta') else float(getattr(event, 'deltaY', lambda: 0)())
                    if delta == 0:
                        return False
                    factor = 0.9 if delta > 0 else 1.1
                    if ctrl:
                        vb.scaleBy((factor, 1.0))
                        return True
                    if shift:
                        vb.scaleBy((1.0, factor))
                        return True
            # RMB drag -> vertical-only pan, LMB drag -> horizontal-only pan
            if event.type() == QEvent.GraphicsSceneMouseMove:
                vb = _get_viewbox(self.view.plot)
                if vb is None:
                    return False
                btns = int(getattr(event, 'buttons', lambda: 0)())
                rmb = bool(btns & 0x00000004)  # Qt.RightButton
                lmb = bool(btns & 0x00000001)  # Qt.LeftButton
                if rmb or lmb:
                    # pixel-perfect pan: map pixel delta to data delta via view range and pixel size
                    delta = getattr(event, 'scenePos')() - getattr(event, 'lastScenePos')()
                    dx = float(delta.x()); dy = float(delta.y())
                    if abs(dx) + abs(dy) < 0.1:
                        return False
                    try:
                        (xmin, xmax), (ymin, ymax) = vb.viewRange()
                        w_px = float(vb.width()) if hasattr(vb, 'width') else 1.0
                        h_px = float(vb.height()) if hasattr(vb, 'height') else 1.0
                        w_px = max(1.0, w_px); h_px = max(1.0, h_px)
                        data_per_px_x = (xmax - xmin) / w_px
                        data_per_px_y = (ymax - ymin) / h_px
                    except Exception:
                        data_per_px_x = data_per_px_y = 1.0
                    if lmb and not rmb:
                        # horizontal pan only (invert dx to follow cursor)
                        vb.translateBy(x=-dx * data_per_px_x, y=0)
                        return True
                    if rmb and not lmb:
                        # vertical pan only (positive dy moves down -> increase y)
                        vb.translateBy(x=0, y=dy * data_per_px_y)
                        return True
        except Exception:
            return False
        return False


