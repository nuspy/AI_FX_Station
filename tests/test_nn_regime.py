import os
import tempfile
import json
import numpy as np
from sqlalchemy import create_engine, MetaData, select
from src.forex_diffusion.services.db_service import DBService
from src.forex_diffusion.services.db_writer import DBWriter
from src.forex_diffusion.services.nn_service import NearestNeighborService

def generate_cluster(center, n, dim, scale=0.01, seed=None):
    rng = np.random.RandomState(seed)
    return (rng.randn(n, dim) * scale + np.asarray(center)).tolist()

def test_nn_regime_with_synthetic_latents_and_ticks():
    # create temporary sqlite db
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tf.close()
    db_url = f"sqlite:///{tf.name}"
    engine = create_engine(db_url, future=True)

    # init DBService to create tables
    dbs = DBService(engine=engine)

    # create synthetic latents: two clusters
    dim = 8
    n_per_cluster = 200
    center_a = [0.1] * dim
    center_b = [-0.1] * dim
    latents_a = generate_cluster(center_a, n_per_cluster, dim, scale=0.01, seed=1)
    latents_b = generate_cluster(center_b, n_per_cluster, dim, scale=0.01, seed=2)

    rows = []
    ts_base = 1630000000000
    for i, v in enumerate(latents_a):
        rows.append({"symbol": "EUR/USD", "timeframe": "1m", "ts_utc": ts_base + i * 60000, "latent": v, "model_version": "v1"})
    for i, v in enumerate(latents_b):
        rows.append({"symbol": "EUR/USD", "timeframe": "1m", "ts_utc": ts_base + (n_per_cluster + i) * 60000, "latent": v, "model_version": "v1"})

    # bulk insert latents synchronously
    dbs.write_latents_bulk(rows)

    # create tick aggregates (simulate 1-min buckets)
    tick_rows = []
    for i in range(0, 20):
        tick_rows.append({"symbol": "EUR/USD", "timeframe": "1m", "ts_utc": ts_base + i * 60000, "tick_count": int(10 + i % 5)})
    dbs.write_ticks_bulk(tick_rows)

    # run NN query near center_a
    query = [0.1 + 0.001] * dim
    nn = NearestNeighborService(engine=engine)
    res = nn.get_regime(query_latent=query, symbol="EUR/USD", timeframe="1m", k=10)
    assert res.get("found", False) is True
    assert "regime_score" in res
    assert res["regime_score"] > 0.0

    # verify ticks persisted
    meta = MetaData()
    meta.reflect(bind=engine, only=["ticks_aggregate"])
    tbl = meta.tables.get("ticks_aggregate")
    assert tbl is not None
    with engine.connect() as conn:
        r = conn.execute(select(tbl.c.tick_count).where(tbl.c.symbol == "EUR/USD")).fetchall()
        assert len(r) >= 1

    # cleanup
    os.unlink(tf.name)
