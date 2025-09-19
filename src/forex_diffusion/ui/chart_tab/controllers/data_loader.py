from __future__ import annotations

from typing import Optional, List

from PySide6.QtCore import QObject, Slot


class DataLoaderController(QObject):
    def __init__(self, chart_tab, db_service):
        super().__init__(chart_tab)
        self.view = chart_tab
        self.db_service = db_service
        # persist settings across sessions
        try:
            from ...utils.user_settings import get_setting
            sel = get_setting("chart_tab.settings", {}) or {}
            if isinstance(sel, dict):
                sym = sel.get("symbol"); tf = sel.get("timeframe")
                if sym: setattr(self.view, "symbol", str(sym))
                if tf: setattr(self.view, "timeframe", str(tf))
        except Exception:
            pass

    def populate_symbols_timeframes(self):
        symbols = self._fetch_symbols()
        tfs = self._fetch_timeframes()
        try:
            cb = getattr(self.view, "symbolCombo", None)
            if cb is not None:
                cb.clear()
                cb.addItems(symbols or ["EUR/USD"])
                # restore selection
                try:
                    idx = cb.findText(self.view.symbol)
                    if idx >= 0: cb.setCurrentIndex(idx)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            cb = getattr(self.view, "timeframeCombo", None)
            if cb is not None:
                cb.clear()
                cb.addItems(tfs or ["1m","5m","15m","30m","1h","4h","1d"])
                try:
                    idx = cb.findText(self.view.timeframe)
                    if idx >= 0: cb.setCurrentIndex(idx)
                except Exception:
                    pass
        except Exception:
            pass

    def _fetch_symbols(self) -> List[str]:
        try:
            from sqlalchemy import text
            engine = getattr(self.db_service, "engine", None)
            if engine is None:
                return ["EUR/USD"]
            with engine.connect() as conn:
                rows = conn.execute(text("SELECT DISTINCT symbol FROM market_data_candles ORDER BY symbol ASC"))
                syms = [str(r[0]) for r in rows if r and r[0]]
                return syms or ["EUR/USD"]
        except Exception:
            return ["EUR/USD"]

    def _fetch_timeframes(self) -> List[str]:
        try:
            from sqlalchemy import text
            engine = getattr(self.db_service, "engine", None)
            if engine is None:
                return ["1m","5m","15m","30m","1h","4h","1d"]
            with engine.connect() as conn:
                rows = conn.execute(text("SELECT DISTINCT timeframe FROM market_data_candles ORDER BY timeframe ASC"))
                tfs = [str(r[0]) for r in rows if r and r[0]]
                return tfs or ["1m","5m","15m","30m","1h","4h","1d"]
        except Exception:
            return ["1m","5m","15m","30m","1h","4h","1d"]


