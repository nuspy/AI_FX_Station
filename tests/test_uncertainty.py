import numpy as np
from src.forex_diffusion.postproc.uncertainty import weighted_icp_calibrate


def test_weighted_icp_basic():
    # Generate synthetic history: 200 samples
    rng = np.random.RandomState(0)
    M = 200
    # Simulate true values y ~ N(0,1)
    y = rng.randn(M)
    # Simulate quantile predictions with some noise: q05 = y - u, q95 = y + v where u,v ~ Uniform(0,0.5)
    u = rng.uniform(0.0, 0.5, size=M)
    v = rng.uniform(0.0, 0.5, size=M)
    q05 = y - u
    q95 = y + v
    # timestamps evenly spaced in ms
    now = int(1_700_000_000_000)  # arbitrary epoch ms
    ts = np.arange(M).astype(float)
    ts_ms = now - (M - ts) * 60_000  # 1-min spacing

    result = weighted_icp_calibrate(q05_hist=q05, q95_hist=q95, y_hist=y, ts_hist_ms=ts_ms, t_now_ms=now, half_life_days=30.0, alpha=0.10)

    assert "delta_global" in result
    assert result["delta_global"] >= 0.0
    assert 0.0 <= result["cov_hat"] <= 1.0
    assert "weights" in result
    ws = result["weights"]
    assert abs(ws.sum() - 1.0) < 1e-6
