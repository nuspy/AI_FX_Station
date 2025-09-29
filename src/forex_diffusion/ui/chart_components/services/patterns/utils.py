# ui/chart_components/services/patterns/utils.py
# Utility functions for pattern services
from __future__ import annotations

from PySide6.QtCore import QThread
from .historical_scan_worker import HistoricalScanWorker


def start_historical_scan(self, df_snapshot):
    """Start a historical pattern scan across multiple timeframes"""
    try:
        self._hist_thread = QThread(self.view)
        self._hist_worker = HistoricalScanWorker(self)
        self._hist_worker.moveToThread(self._hist_thread)
        self._hist_thread.started.connect(self._hist_worker.run)
        try:
            self._hist_worker.finished.connect(self._hist_thread.quit)
        except Exception:
            pass
        try:
            self._hist_worker.set_snapshot(df_snapshot)
        except Exception:
            pass
        self._hist_thread.start()
    except Exception:
        try:
            self.on_update_plot(df_snapshot)
        except Exception:
            pass


def _min_required_bars(self, det) -> int:
    """Calculate minimum required bars for a detector"""
    for attr in ('window', 'max_span'):
        if hasattr(det, attr):
            try:
                v = int(getattr(det, attr))
                if v and v > 0:
                    return max(60, v)
            except Exception:
                pass
    key = getattr(det, 'key', '') or ''
    long_families = {
        'head_and_shoulders': 140,
        'inverse_head_and_shoulders': 140,
        'diamond_': 160,
        'triple_': 140,
        'double_': 120,
        'triangle': 120,
        'wedge_': 120,
        'channel': 120,
        'broadening': 120,
        'cup_and_handle': 160,
        'rounding_': 160,
        'barr_': 200,
        'harmonic_': 160
    }
    for frag, v in long_families.items():
        if frag in key:
            return v
    return 80