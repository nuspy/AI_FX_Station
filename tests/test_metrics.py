import numpy as np
from src.forex_diffusion.utils import metrics


def test_annualized_sharpe_and_mdd():
    # synthetic returns: small positive mean
    returns = np.repeat(0.0001, 252 * 10)  # constant small returns
    sh = metrics.annualized_sharpe(returns, bars_per_day=1440 / 60, annual_days=252)  # assuming hourly approx
    assert sh == sh  # not NaN

    equity = np.array([1.0, 1.1, 1.05, 1.2, 0.9, 1.0])
    mdd = metrics.max_drawdown(equity)
    assert 0.0 <= mdd <= 1.0


def test_crps_and_pit():
    # samples from N(0,1) for 1000 draws, single observation y=0.1
    N = 500
    samples = np.random.randn(N)
    y = 0.1
    crps = metrics.crps_sample_np(samples, y)
    assert crps >= 0.0

    pit, ks_stat, ks_p = metrics.pit_ks_np(samples, y)
    assert pit.shape[0] == 1
    assert 0.0 <= ks_p <= 1.0
