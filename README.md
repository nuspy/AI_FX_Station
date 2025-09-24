# ForexMagic Patch v1

Contenuti principali:
- **train_sklearn.py**: OHLC in **variazione relativa** (log-returns), feature **ora del giorno** e **giorno settimana**, indicatori con **parametri UI** e **timeframe** selezionati, modelli **ridge/lasso/elasticnet + rf**, salvataggio **joblib**.
- **train.py** (Lightning stub) per permettere il **ritorno al ramo Lightning** dalla UI.
- **ui/training_launcher.py**: invocazione robusta (senza prefisso `src.`) che passa *tutti* i parametri.
- **db_adapter.py**: collega qui il fetch delle candele dal DB (ritorna ts_utc/open/high/low/close/volume).
