
from forex_diffusion.patterns.candles import make_candle_detectors
from .conftest import make_trending_series

def test_candles_factory():
    df = make_trending_series()
    dets = make_candle_detectors()
    assert any(d.key.startswith("harami_cross") for d in dets)
