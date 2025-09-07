import time
from fastapi.testclient import TestClient
import importlib

def test_inference_lifecycle_starts_and_stops_db_writer():
    # import the inference service module
    svc = importlib.import_module("src.forex_diffusion.inference.service")
    client = TestClient(svc.app)
    # entering the context triggers startup (lifespan)
    with client:
        # give some time for DBWriter thread to start
        time.sleep(0.2)
        dbw = getattr(svc, "db_writer", None)
        assert dbw is not None, "db_writer should be instantiated"
        assert getattr(dbw, "_started", False) is True, "DBWriter should be started during lifespan"
        # also check /health endpoint responds
        r = client.get("/health")
        assert r.status_code == 200
        assert "model_loaded" in r.json()
    # after exiting the context, lifespan shutdown should have stopped the db_writer
    time.sleep(0.2)
    dbw = getattr(svc, "db_writer", None)
    if dbw is not None:
        assert getattr(dbw, "_started", False) is False, "DBWriter should be stopped after lifespan shutdown"
