# tmp/nn_forecast.py
from pathlib import Path
import sys, json, numpy as np
sys.path.insert(0, str(Path('.').resolve()/'src'))
from forex_diffusion.services.db_service import DBService
from forex_diffusion.services.regime_service import RegimeService
from sqlalchemy import text
import numpy as np

SYM = "EUR/USD"
TF = "1m"
K = 10
HORIZON_BARS = [5,10,20]  # forecast horizons in bars

db = DBService()
eng = db.engine
svc = RegimeService(engine=eng)

# load last latent for symbol/timeframe
with eng.connect() as conn:
    r = conn.execute(text("SELECT id, ts_utc, latent_json FROM latents WHERE symbol=:s AND timeframe=:tf ORDER BY ts_utc DESC LIMIT 1"), {"s":SYM, "tf":TF}).fetchone()
    if not r:
        print("No latent found"); sys.exit(1)
    lj = r._mapping["latent_json"] if hasattr(r, "_mapping") else r[2]
    qvec = np.asarray(json.loads(lj), dtype=np.float32)

# query regime index
res = svc.query_regime(qvec.tolist(), k=K)
print("Query regime result:", res)

# gather neighbor ids and compute successor returns
neighbor_ids = res.get("neighbor_ids", [])
if not neighbor_ids:
    print("No neighbor ids")
    sys.exit(0)

with eng.connect() as conn:
    # fetch ts_utc for neighbors and compute future returns
    rows = conn.execute(text("SELECT id, ts_utc FROM latents WHERE id IN :ids"), {"ids": tuple(neighbor_ids)}).fetchall()
    id2ts = {r[0]: r[1] for r in rows}
    # for each neighbor, fetch candle at ts and future close at +h bars
    forecasts = {h: [] for h in HORIZON_BARS}
    for nid in neighbor_ids:
        ts = id2ts.get(nid)
        if ts is None: continue
        # get close at ts and close at ts + h*period (approx: use ORDER BY ts_utc and LIMIT)
        base = conn.execute(text("SELECT close FROM market_data_candles WHERE symbol=:s AND timeframe=:tf AND ts_utc=:t LIMIT 1"), {"s":SYM,"tf":TF,"t":ts}).fetchone()
        if not base: continue
        base_close = float(base[0])
        for h in HORIZON_BARS:
            # get h-th future bar
            fut = conn.execute(text("SELECT close FROM market_data_candles WHERE symbol=:s AND timeframe=:tf AND ts_utc > :t ORDER BY ts_utc ASC LIMIT :lim"), {"s":SYM,"tf":TF,"t":ts,"lim":h}).fetchall()
            if len(fut)>=h:
                fut_close = float(fut[-1][0])
                ret = (fut_close - base_close)/base_close
                forecasts[h].append(ret)
    # aggregate
    for h in HORIZON_BARS:
        arr = np.array(forecasts[h]) if forecasts[h] else np.array([])
        print(f"H={h} bars: n_samples={len(arr)} mean_ret={float(arr.mean()) if arr.size else None} median_ret={float(np.median(arr)) if arr.size else None}")