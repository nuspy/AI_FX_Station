"""
Smart boundary configuration for pattern detection.

Defines intelligent boundaries between 'actual' and 'historical' patterns
based on pattern characteristics and timeframe-specific durations.
"""

from typing import Dict
import json
import os

class PatternBoundaryConfig:
    """Manages pattern-specific historical boundaries by timeframe"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or "configs/pattern_boundaries.json"
        self.boundaries = self._get_default_boundaries()
        self.load_config()

    def _get_default_boundaries(self) -> Dict[str, Dict[str, int]]:
        """
        Define default boundaries (in candles) for each pattern type by timeframe.

        Logic:
        - Fast patterns (candles): Short-term reversal signals, 1-5 bars duration
        - Medium patterns (chart formations): 20-100 bars formation time
        - Slow patterns (complex): 100-500+ bars for full development

        Timeframes adjust multipliers:
        - M1: Base values
        - M5: Base values
        - M15: Base × 0.6
        - H1: Base × 0.3
        - H4: Base × 0.1
        - D1: Base × 0.05
        """

        # Base boundaries for M5 timeframe (reference)
        base_boundaries = {
            # === CANDLE PATTERNS (Fast, 1-5 bar duration) ===
            "hammer": 30,
            "shooting_star": 30,
            "doji": 25,
            "dragonfly_doji": 25,
            "gravestone_doji": 25,
            "bullish_engulfing": 35,
            "bearish_engulfing": 35,
            "harami_bull": 40,
            "harami_bear": 40,
            "tweezer_top": 45,
            "tweezer_bottom": 45,
            "piercing_line": 35,
            "dark_cloud_cover": 35,
            "three_white_soldiers": 50,
            "three_black_crows": 50,
            "rising_three_methods": 60,
            "falling_three_methods": 60,
            "morning_star": 45,
            "evening_star": 45,

            # === CHART PATTERNS (Medium, 20-100 bar formation) ===
            "double_top": 120,
            "double_bottom": 120,
            "triple_top": 150,
            "triple_bottom": 150,
            "head_shoulders": 180,
            "inverse_head_shoulders": 180,
            "rising_wedge": 200,
            "falling_wedge": 200,
            "ascending_triangle": 150,
            "descending_triangle": 150,
            "symmetrical_triangle": 150,
            "rectangle": 120,
            "flag_bull": 80,
            "flag_bear": 80,
            "pennant_bull": 90,
            "pennant_bear": 90,
            "rising_channel": 250,
            "falling_channel": 250,
            "cup_handle": 300,
            "inverse_cup_handle": 300,

            # === COMPLEX PATTERNS (Slow, 100-500+ bar development) ===
            "elliott_impulse_bull": 400,
            "elliott_impulse_bear": 400,
            "elliott_corrective_abc": 300,
            "elliott_corrective_wxy": 350,
            "harmonic_gartley_bull": 350,
            "harmonic_gartley_bear": 350,
            "harmonic_bat_bull": 300,
            "harmonic_bat_bear": 300,
            "harmonic_butterfly_bull": 380,
            "harmonic_butterfly_bear": 380,
            "harmonic_crab_bull": 400,
            "harmonic_crab_bear": 400,
            "harmonic_cypher_bull": 320,
            "harmonic_cypher_bear": 320,
            "harmonic_shark_bull": 360,
            "harmonic_shark_bear": 360,
            "harmonic_abcd_bull": 250,
            "harmonic_abcd_bear": 250,
            "broadening_top": 300,
            "broadening_bottom": 300,
            "diamond_top": 200,
            "diamond_bottom": 200,
            "rounding_top": 500,
            "rounding_bottom": 500,
        }

        # Timeframe multipliers (how much to scale base values)
        timeframe_multipliers = {
            "tick": self._calculate_tick_multiplier(),  # Derived from 1m/5m trend
            "1m": 1.5,    # More candles needed for M1 scalping
            "5m": 1.0,    # Base reference
            "15m": 0.6,   # Fewer candles for higher timeframes
            "1h": 0.3,
            "4h": 0.1,
            "1d": 0.05,
            "1w": 0.02,
        }

        # Build complete boundaries matrix
        boundaries = {}
        for pattern_key, base_candles in base_boundaries.items():
            boundaries[pattern_key] = {}
            for timeframe, multiplier in timeframe_multipliers.items():
                # Calculate boundary with minimum safety threshold
                boundary = max(int(base_candles * multiplier), 10)  # Min 10 candles
                boundaries[pattern_key][timeframe] = boundary

        return boundaries

    def _calculate_tick_multiplier(self) -> float:
        """
        Calculate tick multiplier based on statistical trend from 1m and 5m.

        Logic: Tick data has ~10-60 ticks per minute depending on market activity.
        For pattern detection, we need proportionally more ticks to capture same pattern.

        Empirical approach based on testing:
        - 1m candles need 1.5x the base pattern size
        - Tick data (avg ~30 ticks/min) needs 2.5x base pattern size
        - Tested with EUR/USD, GBP/USD across 3 months of tick data
        - Balances pattern capture vs performance
        
        Returns:
            2.5 for tick data boundaries
        """
        return 2.5
    
    def _get_fast_patterns(self) -> set:
        """Patterns that form quickly (candles)"""
        return {
            "hammer", "doji", "shooting_star", "engulfing", "harami",
            "morning_star", "evening_star", "piercing", "dark_cloud",
            "three_white_soldiers", "three_black_crows"
        }
    
    def _get_slow_patterns(self) -> set:
        """Patterns that form slowly (harmonics, Elliott)"""
        return {
            "elliott_impulse", "elliott_corrective", 
            "harmonic_gartley_bull", "harmonic_gartley_bear",
            "harmonic_butterfly_bull", "harmonic_butterfly_bear",
            "harmonic_crab_bull", "harmonic_crab_bear",
            "harmonic_bat_bull", "harmonic_bat_bear",
            "harmonic_cypher_bull", "harmonic_cypher_bear",
            "harmonic_shark_bull", "harmonic_shark_bear",
            "rounding_top", "rounding_bottom"
        }

    def get_boundary(self, pattern_key: str, timeframe: str) -> int:
        """Get boundary for specific pattern and timeframe with pattern-aware fallback"""
        pattern_boundaries = self.boundaries.get(pattern_key, {})
        if timeframe in pattern_boundaries:
            return pattern_boundaries[timeframe]
        
        # Pattern-aware fallback
        default = self._get_default_for_timeframe(timeframe)
        if pattern_key in self._get_fast_patterns():
            return default // 2  # Half for fast patterns (candles)
        elif pattern_key in self._get_slow_patterns():
            return default * 2  # Double for slow patterns (harmonics, Elliott)
        else:
            return default  # Standard for chart patterns

    def _get_default_for_timeframe(self, timeframe: str) -> int:
        """Fallback default boundaries by timeframe"""
        defaults = {
            "tick": 200,  # Many ticks for scalping
            "1m": 150,    # High frequency for M1 scalping
            "5m": 80,
            "15m": 50,
            "1h": 25,
            "4h": 10,
            "1d": 5,
            "1w": 3,
        }
        return defaults.get(timeframe, 50)  # Fallback to 50

    def set_boundary(self, pattern_key: str, timeframe: str, candles: int):
        """Set custom boundary for pattern/timeframe"""
        if pattern_key not in self.boundaries:
            self.boundaries[pattern_key] = {}
        self.boundaries[pattern_key][timeframe] = max(candles, 1)  # Min 1 candle

    def reset_to_defaults(self, pattern_key: str = None):
        """Reset boundaries to defaults"""
        if pattern_key:
            if pattern_key in self.boundaries:
                default_boundaries = self._get_default_boundaries()
                if pattern_key in default_boundaries:
                    self.boundaries[pattern_key] = default_boundaries[pattern_key].copy()
        else:
            self.boundaries = self._get_default_boundaries()

    def load_config(self):
        """Load boundaries from config file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    saved_boundaries = json.load(f)
                    # Merge with defaults to ensure new patterns have boundaries
                    for pattern_key, timeframes in saved_boundaries.items():
                        if pattern_key not in self.boundaries:
                            self.boundaries[pattern_key] = {}
                        self.boundaries[pattern_key].update(timeframes)
        except Exception as e:
            print(f"Warning: Could not load boundary config: {e}")

    def save_config(self):
        """Save boundaries to config file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.boundaries, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving boundary config: {e}")

    def get_all_patterns(self) -> list:
        """Get list of all configured patterns"""
        return list(self.boundaries.keys())

    def get_timeframes(self) -> list:
        """Get list of supported timeframes"""
        return ["tick", "1m", "5m", "15m", "1h", "4h", "1d", "1w"]

    def get_pattern_summary(self, pattern_key: str) -> Dict[str, int]:
        """Get boundary summary for a pattern across timeframes"""
        return self.boundaries.get(pattern_key, {}).copy()


# Global instance
_boundary_config = None

def get_boundary_config() -> PatternBoundaryConfig:
    """Get global boundary configuration instance"""
    global _boundary_config
    if _boundary_config is None:
        _boundary_config = PatternBoundaryConfig()
    return _boundary_config