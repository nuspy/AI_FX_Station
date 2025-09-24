from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


class WalkForwardCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    train: str = Field("90d", description="Train window, e.g., '90d'")
    test: str = Field("7d", description="Test window, e.g., '7d'")
    step: str = Field("7d", description="Step between slices, e.g., '7d'")
    gap: str = Field("0d", description="Gap to avoid leakage, e.g., '0d'")


class TimeFilters(BaseModel):
    model_config = ConfigDict(extra="ignore")
    trading_hours: Optional[List[Tuple[str, str]]] = Field(default=None, description="List of (HH:MM,HH:MM)")
    exclude_weekends: bool = True
    exclude_holidays: bool = False
    high_vol_events: bool = False


class BacktestInterval(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = Field("preset", description="'preset'|'absolute'|'relative'")
    preset: Optional[str] = Field(default="30d", description="e.g., '7d','30d','90d','1Y','YTD'")
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None
    relative_n: Optional[int] = Field(default=None, description="for relative type: number of bars/days")
    relative_unit: Optional[str] = Field(default=None, description="'bars'|'days'")
    timezone: str = Field("UTC")
    walkforward: WalkForwardCfg = Field(default_factory=WalkForwardCfg)
    filters: TimeFilters = Field(default_factory=TimeFilters)


class IndicatorsCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # Each indicator maps to {enabled:bool, timeframe_params:{tf:{...}}}
    ATR: Optional[Dict[str, Dict]] = None
    RSI: Optional[Dict[str, Dict]] = None
    Bollinger: Optional[Dict[str, Dict]] = None
    MACD: Optional[Dict[str, Dict]] = None
    Donchian: Optional[Dict[str, Dict]] = None
    Keltner: Optional[Dict[str, Dict]] = None
    Hurst: Optional[Dict[str, Dict]] = None


class BacktestConfigPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: Optional[int] = None
    symbol: str = Field("EUR/USD", description="Symbol, e.g., 'EUR/USD'")
    prediction_types: List[str] = Field(..., description="['Basic','Advanced','Baseline']")
    models: List[str] = Field(default_factory=list, description="Model identifiers or paths")
    timeframe: str = Field(..., description="e.g., '1m','5m','1h'")
    horizons_raw: str = Field(..., description="e.g., '30s, 1m, (5-15)m'")
    samples_range: Tuple[int, int, int] = Field((200, 1500, 200), description="(start, stop, step)")
    indicators: IndicatorsCfg = Field(default_factory=IndicatorsCfg)
    interval: BacktestInterval = Field(default_factory=BacktestInterval)
    data_version: Optional[str] = Field(default=None)
    use_cache: bool = True


class BacktestResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    adherence_mean: Optional[float] = None
    adherence_std: Optional[float] = None
    p10: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None
    win_rate_delta: Optional[float] = None
    delta_used: Optional[float] = None
    skill_rw: Optional[float] = None
    coverage_observed: Optional[float] = None
    coverage_target: Optional[float] = None
    coverage_abs_error: Optional[float] = None
    band_efficiency: Optional[float] = None
    n_points: Optional[int] = None
    cv_horizons: Optional[float] = None
    cv_time: Optional[float] = None
    robustness_index: Optional[float] = None
    complexity_penalty: Optional[float] = None
    composite_score: Optional[float] = None
    dropped: Optional[bool] = None
    drop_reason: Optional[str] = None
    horizon_profile_json: Optional[dict] = None
    time_profile_json: Optional[dict] = None
    coverage_ratio: Optional[float] = None


__all__ = [
    "WalkForwardCfg",
    "TimeFilters",
    "BacktestInterval",
    "IndicatorsCfg",
    "BacktestConfigPayload",
    "BacktestResult",
]


