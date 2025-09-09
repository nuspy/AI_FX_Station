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

4) Test rapido DB/signals:
   python .\scripts\send_and_check_signals.py --count 5 --interval 0.2 --show 10

5) Avvia realtime helper (foreground):
   python .\scripts\start_realtime.py

Per operazioni avanzate vedere la cartella scripts/ e configs/.
