import os
import tempfile
import json
import numpy as np
from sqlalchemy import create_engine
from src.forex_diffusion.services.db_service import DBService
from src.forex_diffusion.services.regime_service import RegimeService

def generate_cluster(center, n, dim, scale=0.01, seed=None):
    rng = np.random.RandomState(seed)
    return (rng.randn(n, dim) * scale + np.asarray(center)).tolist()

def test_build_regime_index_and_query():
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tf.close()
    db_url = f"sqlite:///{tf.name}"
    engine = create_engine(db_url, future=True)
    dbs = DBService(engine=engine)

    # create two clusters
    dim = 8
    n = 300
    center1 = [0.2]*dim
    center2 = [-0.2]*dim
    lat1 = generate_cluster(center1, n, dim, seed=1)
    lat2 = generate_cluster(center2, n, dim, seed=2)
    rows = []
    ts_base = 1630000000000
    for i, v in enumerate(lat1+lat2):
        rows.append({"symbol": "EUR/USD", "timeframe": "1m", "ts_utc": ts_base + i*60000, "latent": v, "model_version": "v1"})
    dbs.write_latents_bulk(rows)

    rs = RegimeService(engine=engine)
    rs.fit_clusters_and_index(n_clusters=2, limit=None)
    # query near center1
    q = [0.2 + 0.001]*dim
    res = rs.query_regime(q, k=10)
    assert res.get("found", False) is True
    assert "regime" in res and res["regime"].startswith("r")
    # cleanup
    os.unlink(tf.name)
