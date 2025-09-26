
import pandas as pd
import numpy as np
from src.forex_diffusion.patterns.registry import PatternRegistry

def _synth_series(n=300):
    idx = pd.date_range('2024-01-01', periods=n, freq='T')
    base = np.sin(np.linspace(0, 12*np.pi, n)) * 0.001 + 1.10
    noise = np.random.normal(scale=0.0005, size=n)
    price = base + noise
    df = pd.DataFrame({'open':price, 'high':price+0.0005, 'low':price-0.0005, 'close':price}, index=idx)
    return df

def test_registry_smoke():
    reg = PatternRegistry()
    dets_chart = [d for d in reg.detectors(['chart'])]
    dets_candle = [d for d in reg.detectors(['candle'])]
    assert len(dets_chart) >= 20  # expanded set
    assert len(dets_candle) >= 6

def test_detectors_run():
    df = _synth_series(600)
    reg = PatternRegistry()
    for d in reg.detectors(['chart']):
        evs = d.detect(df)
        assert isinstance(evs, list)
