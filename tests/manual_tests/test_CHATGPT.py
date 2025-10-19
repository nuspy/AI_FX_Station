import pickle, sys, hashlib, pathlib
import pytest

p = pathlib.Path(r"D:\Projects\ForexGPT\artifacts\models\weighted_forecast_EURUSD_1m_h5_ridge_none.pkl")

if not p.exists():
    print("missing artifact, skipping")
    pytest.skip("required artifact missing", allow_module_level=True)

obj = pickle.load(open(p, "rb"))
print("keys:", sorted(obj.keys()))
print("name:", obj.get("name"))
print("features(len):", len(obj.get("features", [])))
print("first_features:", obj.get("features", [])[:10])
mu = obj.get("std_mu", {}); sg = obj.get("std_sigma", {})
print("mu/sigma count:", len(mu), len(sg))
print("model_type:", type(obj.get("model")).__name__)
print("sha16:", hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])