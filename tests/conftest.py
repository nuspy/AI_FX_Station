
import pandas as pd
import numpy as np
import pytest

def make_trending_series(n=300, slope=0.01, noise=0.05, seed=123):
    rng = np.random.default_rng(seed)
    base = np.arange(n)*slope + rng.normal(0, noise, size=n)
    high = base + rng.uniform(0.05,0.1,size=n)
    low  = base - rng.uniform(0.05,0.1,size=n)
    open_ = base + rng.normal(0, noise/2, size=n)
    close = base + rng.normal(0, noise/2, size=n)
    time = pd.date_range("2020-01-01", periods=n, freq="H")
    return pd.DataFrame({"time":time,"open":open_,"high":high,"low":low,"close":close,"volume":1.0})
