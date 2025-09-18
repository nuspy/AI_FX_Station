# src/forex_diffusion/db_adapter.py
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Optional
import pandas as pd

# === Config via env (con default sensati per il tuo repo) ===
SQLITE_PATH  = os.environ.get("FOREX_DB_SQLITE",  r"D:\Projects\ForexGPT\data\market.db")
DUCKDB_PATH  = os.environ.get("FOREX_DB_DUCKDB",  r"D:\Projects\ForexGPT\data\market.duckdb")
DATA_DIR     = os.environ.get("FOREX_DATA_DIR",   r"D:\Projects\ForexGPT\data")

REQUIRED_COLS = ["ts_utc", "open", "high", "low", "close"]

def _now_ms() -> int:
    return int(time.time() * 1000)

def _since_ms(days_history: int) -> int:
    return _now_ms() - int(days_history) * 86_400_000

def _sym_key(symbol: str) -> str:
    # EUR/USD -> EURUSD
    return symbol.replace("/", "").replace(" ", "").upper()

def _cast_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # tipizza, ordina, riempi volume
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Colonna mancante nel dataset: {c}")
    df = df[["ts_utc", "open", "high", "low", "close"] + ([c for c in df.columns if c == "volume"])]
    df["ts_utc"] = pd.to_numeric(df["ts_utc"], errors="coerce").astype("Int64")
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["ts_utc","open","high","low","close"]).astype({"ts_utc":"int64"})
    df = df.sort_values("ts_utc").reset_index(drop=True)
    return df

def _try_sqlite(path: Optional[str], symbol: str, timeframe: str, since_ms: int) -> Optional[pd.DataFrame]:
    if not path or not Path(path).exists():
        return None
    import sqlite3
    q = """
    SELECT ts_utc, open, high, low, close, COALESCE(volume,0) AS volume
    FROM candles
    WHERE symbol = ? AND timeframe = ? AND ts_utc >= ?
    ORDER BY ts_utc
    """
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(q, con, params=[symbol, timeframe, since_ms])
    finally:
        con.close()
    return _cast_and_clean(df) if len(df) else pd.DataFrame(columns=REQUIRED_COLS)

def _try_duckdb(path: Optional[str], symbol: str, timeframe: str, since_ms: int) -> Optional[pd.DataFrame]:
    if not path or not Path(path).exists():
        return None
    import duckdb
    con = duckdb.connect(path)
    try:
        df = con.execute(
            """
            SELECT ts_utc, open, high, low, close,
                   COALESCE(volume, 0) AS volume
            FROM candles
            WHERE symbol = ? AND timeframe = ? AND ts_utc >= ?
            ORDER BY ts_utc
            """,
            [symbol, timeframe, since_ms],
        ).df()
    finally:
        con.close()
    return _cast_and_clean(df) if len(df) else pd.DataFrame(columns=REQUIRED_COLS)

def _try_parquet_csv(data_dir: Optional[str], symbol: str, timeframe: str, since_ms: int) -> Optional[pd.DataFrame]:
    if not data_dir:
        return None
    base = Path(data_dir)
    key  = f"{_sym_key(symbol)}_{timeframe}"
    # Prova vari pattern comuni
    candidates = [
        base / f"{key}.parquet",
        base / f"{key}.pq",
        base / f"{key}.feather",
        base / f"{key}.csv",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix.lower() in [".parquet",".pq"]:
                df = pd.read_parquet(p)
            elif p.suffix.lower() == ".feather":
                df = pd.read_feather(p)
            else:
                df = pd.read_csv(p)
            if "ts" in df.columns and "ts_utc" not in df.columns:
                df = df.rename(columns={"ts":"ts_utc"})
            df = _cast_and_clean(df)
            # filtro since_ms (se file contiene più storico)
            if "ts_utc" in df.columns:
                df = df[df["ts_utc"] >= since_ms].reset_index(drop=True)
            return df
    return None

def fetch_candles_from_db(symbol: str, timeframe: str, days_history: int) -> pd.DataFrame:
    """
    Ritorna DataFrame con colonne: ts_utc (ms UTC), open, high, low, close, volume.
    Origine: SQLite -> DuckDB -> Parquet/CSV (prima disponibile).
    """
    since = _since_ms(int(days_history))
    # 1) SQLite
    df = _try_sqlite(SQLITE_PATH, symbol, timeframe, since)
    if df is not None and len(df):
        return df
    # 2) DuckDB
    df = _try_duckdb(DUCKDB_PATH, symbol, timeframe, since)
    if df is not None and len(df):
        return df
    # 3) Parquet / CSV
    df = _try_parquet_csv(DATA_DIR, symbol, timeframe, since)
    if df is not None and len(df):
        return df

    # Se nessuna sorgente disponibile:
    tried = [
        f"SQLite: {SQLITE_PATH}",
        f"DuckDB: {DUCKDB_PATH}",
        f"Files:  {DATA_DIR}\\{_sym_key(symbol)}_{timeframe}.(parquet|pq|feather|csv)"
    ]
    raise FileNotFoundError(
        "Nessuna sorgente dati trovata per le candele.\n"
        + "\n".join(f"- {t}" for t in tried)
        + "\nImposta le variabili d'ambiente FOREX_DB_SQLITE / FOREX_DB_DUCKDB / FOREX_DATA_DIR "
          "oppure popola i file Parquet/CSV con lo schema richiesto."
    )
