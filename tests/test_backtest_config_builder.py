import pytest

from forex_diffusion.backtest.config_builder import build_param_grid, expand_indicator_timeframes, numeric_range


def test_numeric_range_inclusive():
    assert numeric_range(0, 4, 2) == [0, 2, 4]
    assert numeric_range(5, 5, 10) == [5]


def test_build_param_grid_cartesian():
    numeric = {"warmup_bars": (10, 12, 1), "atr_n": (14, 14, 5)}
    booleans = {"apply_conformal": [True], "auto_predict": [True, False]}
    grid = build_param_grid(numeric, booleans)
    assert len(grid) == 6  # 3 warmup values * 1 atr * 2 bool combinations
    assert {entry["warmup_bars"] for entry in grid} == {10, 11, 12}
    assert all(entry["atr_n"] == 14 for entry in grid)
    assert any(entry["auto_predict"] for entry in grid)
    assert any(not entry["auto_predict"] for entry in grid)


def test_expand_indicator_timeframes_generates_variants():
    selection = {"ATR": ["1m", "5m"], "RSI": ["15m"]}
    variants = expand_indicator_timeframes(selection)
    # includes original selection and single-timeframe combinations
    assert any(set(v.get("ATR", [])) == {"1m", "5m"} for v in variants)
    single_variants = [v for v in variants if set(v.keys()) == {"ATR", "RSI"} and len(v["ATR"]) == 1]
    assert {tuple(v["ATR"]) for v in single_variants} == {("1m",), ("5m",)}
    assert all(v["RSI"] == ["15m"] for v in variants if "RSI" in v)
