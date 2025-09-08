MagicForex — guida rapida (locale)

1) Crea e attiva l'ambiente virtuale:
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install -e .

2) Imposta variabili (PowerShell):
   $env:DATABASE_URL = "sqlite:///./data/local.db"
   $env:ADMIN_TOKENS = "localtoken:admin"
   $env:ARTIFACTS_DIR = "./artifacts"

3) Avvia GUI:
   python .\scripts\run_gui.py

4) Test rapido DB/signals:
   python .\scripts\send_and_check_signals.py --count 5 --interval 0.2 --show 10

5) Logs & debug: controlla la console per i log (DBWriter avviato dal GUI).

Per operazioni avanzate vedere la cartella scripts/ e configs/.
