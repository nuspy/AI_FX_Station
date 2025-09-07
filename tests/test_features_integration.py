import os
import tempfile
import time
import json
from sqlalchemy import create_engine, MetaData, select
from src.forex_diffusion.services.db_service import DBService
from src.forex_diffusion.services.db_writer import DBWriter

def test_features_async_persistence():
    # temporary sqlite file
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tf.close()
    db_url = f"sqlite:///{tf.name}"
    engine = create_engine(db_url, future=True)
    dbs = DBService(engine=engine)
    dbw = DBWriter(db_service=dbs)
    dbw.start()

    try:
        # enqueue a fake features row
        features = {"r": 0.001, "atr": 0.0002, "rsi": 55.3}
        ok = dbw.write_features_async(symbol="EUR/USD", timeframe="1m", ts_utc=1630000000000, features=features, pipeline_version="v1")
        assert ok is True
        # wait briefly for worker to flush
        time.sleep(0.5)
    finally:
        dbw.stop(flush=True, timeout=2.0)

    # verify in DB
    meta = MetaData()
    meta.reflect(bind=engine, only=["features"])
    tbl = meta.tables.get("features")
    assert tbl is not None
    with engine.connect() as conn:
        rows = conn.execute(select(tbl.c.symbol, tbl.c.timeframe, tbl.c.ts_utc, tbl.c.features_json)).fetchall()
        assert len(rows) >= 1
        found = False
        for r in rows:
            if r[0] == "EUR/USD" and r[1] == "1m" and int(r[2]) == 1630000000000:
                payload = json.loads(r[3])
                assert payload.get("r") == 0.001
                found = True
        assert found

    # cleanup
    os.unlink(tf.name)
