MagicForex — guida rapida (locale)

1) Crea e attiva l'ambiente virtuale:
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install -e .
   pip install websocket-client

2) Provider realtime:
   - Default: tiingo (REST). Se provider espone WebSocket, RealTimeIngestService userà lo streaming.
   - Puoi cambiare provider dalla UI (SignalsTab) o in configs/default.yaml -> providers.default

3) Avvia GUI:
   python .\scripts\run_gui.py

4) Test e Backfill:
   - Per testare la scrittura dei tick: `python .\tests\manual_tests\write_3_ticks.py`
   - Corretta la logica di ingestione realtime: ora salva i tick grezzi e li aggrega in candele periodicamente, invece di salvare una candela per ogni tick.

GUI tabs:
 - Signals: recent signals + admin controls
 - History: historical candles table per symbol/timeframe (Refresh, Backfill)
 - Chart: matplotlib chart with pan/zoom (update via HistoryTab refresh or programmatically)

5) Avvia realtime helper (foreground):
   python .\scripts\start_realtime.py

Per operazioni avanzate vedere la cartella scripts/ e configs/.
