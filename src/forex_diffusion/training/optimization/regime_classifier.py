"""
Regime classification and market calendar integration.

This module provides economic and market regime classification for time periods,
enabling per-regime parameter optimization and adaptive backtesting.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

try:
    import exchange_calendars as xcals
except ImportError:
    logger.warning("exchange_calendars not available, using simplified calendar")
    xcals = None

class RegimeType(str, Enum):
    """Types of economic/market regimes"""
    NBER_RECESSION = "nber_recession"
    NBER_EXPANSION = "nber_expansion"
    VIX_LOW = "vix_low"
    VIX_MEDIUM = "vix_medium"
    VIX_HIGH = "vix_high"
    PMI_ABOVE_50 = "pmi_above_50"
    PMI_BELOW_50 = "pmi_below_50"
    EPU_LOW = "epu_low"
    EPU_MEDIUM = "epu_medium"
    EPU_HIGH = "epu_high"
    CLI_UP = "cli_up"
    CLI_DOWN = "cli_down"
    YIELD_CURVE_NORMAL = "yield_curve_normal"
    YIELD_CURVE_INVERTED = "yield_curve_inverted"
    USD_STRENGTH = "usd_strength"
    USD_WEAKNESS = "usd_weakness"

@dataclass
class RegimePeriod:
    """Represents a time period with specific regime classification"""
    start_date: datetime
    end_date: datetime
    regime_type: RegimeType
    confidence: float = 1.0
    raw_value: Optional[float] = None
    threshold_used: Optional[float] = None
    source: str = "manual"
    asset_class: Optional[str] = None

@dataclass
class MarketSession:
    """Market trading session information"""
    name: str
    start_time: datetime
    end_time: datetime
    timezone: str
    is_active: bool = True

class RegimeClassifier:
    """
    Classifies time periods into economic and market regimes.

    Supports:
    - NBER recession/expansion periods
    - VIX-based volatility regimes
    - PMI economic activity classification
    - Economic Policy Uncertainty (EPU) regimes
    - Composite Leading Indicator (CLI) trends
    - Custom regime definitions
    """

    def __init__(self):
        self.regime_cache: Dict[str, List[RegimePeriod]] = {}
        self.data_sources: Dict[str, Any] = {}
        self.market_calendars: Dict[str, Any] = {}

        # Regime thresholds (configurable)
        self.thresholds = {
            "vix_low": 15.0,
            "vix_high": 25.0,
            "pmi_threshold": 50.0,
            "epu_low": 100.0,
            "epu_high": 200.0,
            "yield_spread_threshold": 0.0  # 10Y-2Y spread
        }

        # Initialize market calendars
        self._initialize_market_calendars()

    def _initialize_market_calendars(self):
        """Initialize market trading calendars"""
        try:
            if xcals:
                # Major exchanges
                self.market_calendars["NYSE"] = xcals.get_calendar("NYSE")
                self.market_calendars["LSE"] = xcals.get_calendar("LSE")
                self.market_calendars["TSE"] = xcals.get_calendar("TSE")
                logger.info("Initialized exchange calendars")
            else:
                logger.warning("Using simplified calendar implementation")
                self._create_simple_calendars()
        except Exception as e:
            logger.warning(f"Failed to initialize exchange calendars: {e}")
            self._create_simple_calendars()

    def _create_simple_calendars(self):
        """Create simplified market calendars"""
        # Simplified FX trading sessions (24/5)
        self.market_calendars["FX"] = {
            "sydney": {"start": "22:00", "end": "07:00", "tz": "UTC"},
            "tokyo": {"start": "00:00", "end": "09:00", "tz": "UTC"},
            "london": {"start": "08:00", "end": "17:00", "tz": "UTC"},
            "new_york": {"start": "13:00", "end": "22:00", "tz": "UTC"}
        }

    async def ensure_classifications(self, start_date: datetime, end_date: datetime,
                                   regime_tag: str) -> None:
        """
        Ensure regime classifications exist for the specified period.

        Args:
            start_date: Start of period to classify
            end_date: End of period to classify
            regime_tag: Type of regime to classify
        """
        cache_key = f"{regime_tag}_{start_date.date()}_{end_date.date()}"

        if cache_key in self.regime_cache:
            # Check if classification is recent enough
            if self._is_classification_current(cache_key):
                return

        logger.info(f"Classifying regime {regime_tag} for period {start_date.date()} to {end_date.date()}")

        try:
            regime_type = RegimeType(regime_tag)
            periods = await self._classify_regime_period(start_date, end_date, regime_type)
            self.regime_cache[cache_key] = periods

            # Store in database for persistence
            await self._store_regime_classifications(periods)

        except ValueError:
            logger.warning(f"Unknown regime type: {regime_tag}")
        except Exception as e:
            logger.exception(f"Failed to classify regime {regime_tag}: {e}")

    async def _classify_regime_period(self, start_date: datetime, end_date: datetime,
                                    regime_type: RegimeType) -> List[RegimePeriod]:
        """Classify a time period for a specific regime type"""

        if regime_type in [RegimeType.NBER_RECESSION, RegimeType.NBER_EXPANSION]:
            return await self._classify_nber_periods(start_date, end_date)
        elif regime_type in [RegimeType.VIX_LOW, RegimeType.VIX_MEDIUM, RegimeType.VIX_HIGH]:
            return await self._classify_vix_periods(start_date, end_date)
        elif regime_type in [RegimeType.PMI_ABOVE_50, RegimeType.PMI_BELOW_50]:
            return await self._classify_pmi_periods(start_date, end_date)
        elif regime_type in [RegimeType.EPU_LOW, RegimeType.EPU_MEDIUM, RegimeType.EPU_HIGH]:
            return await self._classify_epu_periods(start_date, end_date)
        elif regime_type in [RegimeType.CLI_UP, RegimeType.CLI_DOWN]:
            return await self._classify_cli_periods(start_date, end_date)
        elif regime_type in [RegimeType.YIELD_CURVE_NORMAL, RegimeType.YIELD_CURVE_INVERTED]:
            return await self._classify_yield_curve_periods(start_date, end_date)
        elif regime_type in [RegimeType.USD_STRENGTH, RegimeType.USD_WEAKNESS]:
            return await self._classify_usd_strength_periods(start_date, end_date)
        else:
            logger.warning(f"Classification not implemented for {regime_type}")
            return []

    async def _classify_nber_periods(self, start_date: datetime,
                                   end_date: datetime) -> List[RegimePeriod]:
        """Classify NBER recession/expansion periods"""
        # NBER recession dates (would typically fetch from FRED API)
        # Using historical recession periods as example
        nber_recessions = [
            ("2007-12-01", "2009-06-01"),  # Great Recession
            ("2001-03-01", "2001-11-01"),  # Dot-com recession
            ("1990-07-01", "1991-03-01"),  # Early 1990s recession
            ("2020-02-01", "2020-04-01"),  # COVID recession
        ]

        periods = []
        current_date = start_date

        for rec_start, rec_end in nber_recessions:
            rec_start_dt = pd.to_datetime(rec_start)
            rec_end_dt = pd.to_datetime(rec_end)

            # Add expansion period before recession
            if current_date < rec_start_dt and rec_start_dt <= end_date:
                periods.append(RegimePeriod(
                    start_date=current_date,
                    end_date=rec_start_dt,
                    regime_type=RegimeType.NBER_EXPANSION,
                    source="NBER",
                    confidence=1.0
                ))

            # Add recession period
            if rec_start_dt <= end_date and rec_end_dt >= start_date:
                period_start = max(rec_start_dt, start_date)
                period_end = min(rec_end_dt, end_date)

                periods.append(RegimePeriod(
                    start_date=period_start,
                    end_date=period_end,
                    regime_type=RegimeType.NBER_RECESSION,
                    source="NBER",
                    confidence=1.0
                ))

            current_date = rec_end_dt

        # Add final expansion period if needed
        if current_date < end_date:
            periods.append(RegimePeriod(
                start_date=current_date,
                end_date=end_date,
                regime_type=RegimeType.NBER_EXPANSION,
                source="NBER",
                confidence=1.0
            ))

        return periods

    async def _classify_vix_periods(self, start_date: datetime,
                                  end_date: datetime) -> List[RegimePeriod]:
        """Classify VIX-based volatility regimes"""
        # This would typically fetch VIX data from FRED or Yahoo Finance
        # For now, create synthetic periods based on typical VIX patterns

        periods = []
        current_date = start_date

        # Generate monthly periods with varying VIX levels
        while current_date < end_date:
            period_end = min(current_date + timedelta(days=30), end_date)

            # Simulate VIX level (in practice, would fetch real data)
            # Higher VIX during known stress periods
            if "2008" in str(current_date.year) or "2020" in str(current_date.year):
                vix_level = np.random.normal(35, 10)  # High volatility
            elif "2017" in str(current_date.year) or "2019" in str(current_date.year):
                vix_level = np.random.normal(12, 3)   # Low volatility
            else:
                vix_level = np.random.normal(18, 6)   # Medium volatility

            vix_level = max(5, vix_level)  # Floor at 5

            # Classify based on thresholds
            if vix_level < self.thresholds["vix_low"]:
                regime = RegimeType.VIX_LOW
            elif vix_level > self.thresholds["vix_high"]:
                regime = RegimeType.VIX_HIGH
            else:
                regime = RegimeType.VIX_MEDIUM

            periods.append(RegimePeriod(
                start_date=current_date,
                end_date=period_end,
                regime_type=regime,
                raw_value=vix_level,
                threshold_used=self.thresholds["vix_low"] if regime == RegimeType.VIX_LOW
                             else self.thresholds["vix_high"],
                source="VIX",
                confidence=0.9
            ))

            current_date = period_end

        return periods

    async def _classify_pmi_periods(self, start_date: datetime,
                                  end_date: datetime) -> List[RegimePeriod]:
        """Classify PMI-based economic activity periods"""
        periods = []
        current_date = start_date

        # Generate monthly PMI classifications
        while current_date < end_date:
            period_end = min(current_date + timedelta(days=30), end_date)

            # Simulate PMI (in practice, would fetch from economic data sources)
            # PMI tends to be cyclical and related to economic conditions
            if "2008" in str(current_date.year) or "2001" in str(current_date.year):
                pmi_level = np.random.normal(45, 5)  # Below 50 during recessions
            else:
                pmi_level = np.random.normal(52, 4)  # Above 50 during expansions

            pmi_level = max(20, min(80, pmi_level))  # Reasonable bounds

            regime = (RegimeType.PMI_ABOVE_50 if pmi_level > self.thresholds["pmi_threshold"]
                     else RegimeType.PMI_BELOW_50)

            periods.append(RegimePeriod(
                start_date=current_date,
                end_date=period_end,
                regime_type=regime,
                raw_value=pmi_level,
                threshold_used=self.thresholds["pmi_threshold"],
                source="PMI",
                confidence=0.85
            ))

            current_date = period_end

        return periods

    async def _classify_epu_periods(self, start_date: datetime,
                                  end_date: datetime) -> List[RegimePeriod]:
        """Classify Economic Policy Uncertainty periods"""
        periods = []
        current_date = start_date

        while current_date < end_date:
            period_end = min(current_date + timedelta(days=30), end_date)

            # Simulate EPU index (baseline ~100, spikes during uncertainty)
            base_epu = 100
            if any(year in str(current_date.year) for year in ["2008", "2016", "2020"]):
                epu_level = np.random.normal(250, 50)  # High uncertainty
            elif any(year in str(current_date.year) for year in ["2017", "2019"]):
                epu_level = np.random.normal(80, 20)   # Low uncertainty
            else:
                epu_level = np.random.normal(120, 30)  # Medium uncertainty

            epu_level = max(50, epu_level)

            # Classify based on thresholds
            if epu_level < self.thresholds["epu_low"]:
                regime = RegimeType.EPU_LOW
            elif epu_level > self.thresholds["epu_high"]:
                regime = RegimeType.EPU_HIGH
            else:
                regime = RegimeType.EPU_MEDIUM

            periods.append(RegimePeriod(
                start_date=current_date,
                end_date=period_end,
                regime_type=regime,
                raw_value=epu_level,
                threshold_used=self.thresholds["epu_low"] if regime == RegimeType.EPU_LOW
                             else self.thresholds["epu_high"],
                source="EPU",
                confidence=0.8
            ))

            current_date = period_end

        return periods

    async def _classify_cli_periods(self, start_date: datetime,
                                  end_date: datetime) -> List[RegimePeriod]:
        """Classify Composite Leading Indicator trends"""
        periods = []
        current_date = start_date

        # CLI is trend-based, so we need to look at changes over time
        cli_values = []
        dates = []

        temp_date = start_date
        while temp_date <= end_date:
            # Simulate CLI (normalized around 100)
            # Trends down before recessions, up during expansions
            if "2007" <= str(temp_date.year) <= "2009":
                cli_trend = -0.5  # Declining
            elif "2000" <= str(temp_date.year) <= "2002":
                cli_trend = -0.3  # Declining
            else:
                cli_trend = 0.2   # Generally rising

            cli_value = 100 + np.random.normal(cli_trend, 1)
            cli_values.append(cli_value)
            dates.append(temp_date)
            temp_date += timedelta(days=30)

        # Calculate 6-month trends
        for i in range(6, len(cli_values)):
            period_start = dates[i-6]
            period_end = dates[i]

            # Calculate trend slope
            recent_values = cli_values[i-6:i]
            x = np.arange(len(recent_values))
            slope = np.polyfit(x, recent_values, 1)[0]

            regime = RegimeType.CLI_UP if slope > 0 else RegimeType.CLI_DOWN

            if period_start >= start_date and period_end <= end_date:
                periods.append(RegimePeriod(
                    start_date=period_start,
                    end_date=period_end,
                    regime_type=regime,
                    raw_value=slope,
                    source="CLI",
                    confidence=0.75
                ))

        return periods

    async def _classify_yield_curve_periods(self, start_date: datetime,
                                          end_date: datetime) -> List[RegimePeriod]:
        """Classify yield curve shape periods"""
        periods = []
        current_date = start_date

        while current_date < end_date:
            period_end = min(current_date + timedelta(days=7), end_date)  # Weekly

            # Simulate 10Y-2Y spread (typically positive, negative before recessions)
            if any(year in str(current_date.year) for year in ["2007", "2000", "2019"]):
                spread = np.random.normal(-0.2, 0.3)  # Inverted
            else:
                spread = np.random.normal(1.5, 0.5)   # Normal

            regime = (RegimeType.YIELD_CURVE_NORMAL if spread > self.thresholds["yield_spread_threshold"]
                     else RegimeType.YIELD_CURVE_INVERTED)

            periods.append(RegimePeriod(
                start_date=current_date,
                end_date=period_end,
                regime_type=regime,
                raw_value=spread,
                threshold_used=self.thresholds["yield_spread_threshold"],
                source="Treasury",
                confidence=0.95
            ))

            current_date = period_end

        return periods

    async def _classify_usd_strength_periods(self, start_date: datetime,
                                           end_date: datetime) -> List[RegimePeriod]:
        """Classify USD strength/weakness periods"""
        periods = []
        current_date = start_date

        while current_date < end_date:
            period_end = min(current_date + timedelta(days=7), end_date)

            # Simulate DXY (Dollar Index) - above 100 = strong, below 100 = weak
            # This would typically use real DXY data
            base_dxy = 100
            if "2008" in str(current_date.year):
                dxy_level = np.random.normal(85, 5)   # USD weakness during crisis
            elif "2015" <= str(current_date.year) <= "2017":
                dxy_level = np.random.normal(105, 3)  # USD strength
            else:
                dxy_level = np.random.normal(95, 8)   # Variable

            regime = (RegimeType.USD_STRENGTH if dxy_level > 100
                     else RegimeType.USD_WEAKNESS)

            periods.append(RegimePeriod(
                start_date=current_date,
                end_date=period_end,
                regime_type=regime,
                raw_value=dxy_level,
                threshold_used=100.0,
                source="DXY",
                asset_class="FX",
                confidence=0.85
            ))

            current_date = period_end

        return periods

    def get_regime_for_date(self, date: datetime, regime_type: RegimeType) -> Optional[RegimePeriod]:
        """Get the regime classification for a specific date"""

        # Search cache first
        for cache_key, periods in self.regime_cache.items():
            if regime_type.value in cache_key:
                for period in periods:
                    if period.start_date <= date <= period.end_date:
                        return period

        return None

    def get_market_session_info(self, timestamp: datetime, market: str = "FX") -> Dict[str, Any]:
        """Get market session information for a timestamp"""

        if market == "FX":
            # FX markets are 24/5 (Sunday 22:00 UTC to Friday 22:00 UTC)
            weekday = timestamp.weekday()
            hour = timestamp.hour

            # Market closed on weekends (Friday 22:00 to Sunday 22:00 UTC)
            if weekday == 5 and hour >= 22:  # Friday after 22:00
                return {"is_market_open": False, "session": "closed", "reason": "weekend"}
            if weekday == 6:  # Saturday
                return {"is_market_open": False, "session": "closed", "reason": "weekend"}
            if weekday == 0 and hour < 22:  # Sunday before 22:00
                return {"is_market_open": False, "session": "closed", "reason": "weekend"}

            # Determine active session
            active_sessions = []
            if 22 <= hour or hour < 7:  # Sydney session
                active_sessions.append("sydney")
            if 0 <= hour < 9:  # Tokyo session
                active_sessions.append("tokyo")
            if 8 <= hour < 17:  # London session
                active_sessions.append("london")
            if 13 <= hour < 22:  # New York session
                active_sessions.append("new_york")

            return {
                "is_market_open": True,
                "active_sessions": active_sessions,
                "primary_session": active_sessions[0] if active_sessions else "sydney",
                "session_overlap": len(active_sessions) > 1
            }

        else:
            # For other markets, use exchange calendars if available
            if market in self.market_calendars and xcals:
                try:
                    calendar = self.market_calendars[market]
                    is_open = calendar.is_open_on_minute(timestamp)
                    return {"is_market_open": is_open, "exchange": market}
                except Exception as e:
                    logger.warning(f"Failed to check market calendar for {market}: {e}")

        return {"is_market_open": True, "note": "Unknown market, assuming open"}

    def _is_classification_current(self, cache_key: str) -> bool:
        """Check if cached classification is still current"""
        # For this implementation, consider classifications current for 1 day
        # In practice, might want different refresh rates for different regime types
        return True  # Simplified for now

    async def _store_regime_classifications(self, periods: List[RegimePeriod]) -> None:
        """Store regime classifications in database"""
        # This would store the classifications in the regime_classifications table
        # For now, just log that we would store them
        logger.debug(f"Would store {len(periods)} regime classifications in database")

    def update_thresholds(self, **kwargs) -> None:
        """Update regime classification thresholds"""
        for key, value in kwargs.items():
            if key in self.thresholds:
                self.thresholds[key] = value
                logger.info(f"Updated threshold {key} = {value}")

    def get_regime_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get summary of all regime classifications for a period"""

        summary = {
            "period": {"start": start_date, "end": end_date},
            "regimes": {},
            "total_days": (end_date - start_date).days
        }

        # Count days in each regime type
        for cache_key, periods in self.regime_cache.items():
            for period in periods:
                if period.start_date <= end_date and period.end_date >= start_date:
                    regime_name = period.regime_type.value
                    if regime_name not in summary["regimes"]:
                        summary["regimes"][regime_name] = {
                            "total_days": 0,
                            "periods": 0,
                            "confidence": 0.0
                        }

                    # Calculate overlap with requested period
                    overlap_start = max(period.start_date, start_date)
                    overlap_end = min(period.end_date, end_date)
                    overlap_days = (overlap_end - overlap_start).days

                    summary["regimes"][regime_name]["total_days"] += overlap_days
                    summary["regimes"][regime_name]["periods"] += 1
                    summary["regimes"][regime_name]["confidence"] = max(
                        summary["regimes"][regime_name]["confidence"], period.confidence
                    )

        # Calculate percentages
        for regime_data in summary["regimes"].values():
            if summary["total_days"] > 0:
                regime_data["percentage"] = regime_data["total_days"] / summary["total_days"] * 100

        return summary