# ui/chart_components/services/patterns/historical_scan_worker.py
# Worker for running historical pattern scans across multiple timeframes
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot


class HistoricalScanWorker(QObject):
    finished = Signal()

    def __init__(self, parent) -> None:
        super().__init__()
        self._parent = parent
        self._df_snapshot = None
        self._tfs = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

    @Slot(object)
    def set_snapshot(self, df):
        self._df_snapshot = df

    @Slot()
    def run(self):
        try:
            ps = self._parent
            if self._df_snapshot is None or getattr(self._df_snapshot, 'empty', True):
                self.finished.emit()
                return
            for tf in self._tfs:
                try:
                    setattr(ps.view, '_patterns_scan_tf_hint', tf)
                except Exception:
                    pass
                try:
                    ps.on_update_plot(self._df_snapshot)
                except Exception:
                    continue
        finally:
            self.finished.emit()