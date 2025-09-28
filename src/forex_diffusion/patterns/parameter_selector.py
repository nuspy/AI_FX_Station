"""
Database-driven Parameter Selection System.

Selects optimal parameters for pattern detection based on historical performance
for specific combinations of asset, timeframe, and regime. Uses intelligent
fallback strategy when specific parameters are not available.
"""

from __future__ import annotations

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from ..cache import cache_decorator, get_pattern_cache


@dataclass
class ParameterSet:
    """Set of parameters for pattern detection"""
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    source: str  # Where these parameters came from
    confidence: float  # How confident we are in these parameters
    last_updated: datetime
    trial_count: int = 0
    success_rate: float = 0.0


@dataclass
class ParameterContext:
    """Context for parameter selection"""
    asset: str
    timeframe: str
    regime: str
    pattern_type: str
    current_market_conditions: Dict[str, Any] = None


class DatabaseParameterSelector:
    """
    Intelligent parameter selection system that chooses optimal parameters
    based on historical performance data stored in the database.

    Priority order (as specified):
    1. asset_timeframe_regime (most specific)
    2. asset_timeframe
    3. timeframe_regime
    4. asset_regime
    5. timeframe (general timeframe defaults)
    6. default (global defaults)

    When parameters are missing: skip pattern for that timeframe and log warning.
    """

    def __init__(self, config: Dict[str, Any], db_session=None):
        self.config = config.get('database', {}).get('parameters', {})
        self.db_session = db_session

        # Configuration
        self.selection_strategy = self.config.get('selection_strategy', 'historical_performance')
        self.fallback_strategy = self.config.get('fallback_strategy', 'skip_pattern')

        # Priority order for parameter selection
        self.priority_order = self.config.get('priority_order', [
            'asset_timeframe_regime',
            'asset_timeframe',
            'timeframe_regime',
            'asset_regime',
            'timeframe',
            'default'
        ])

        # Optimization configuration
        opt_config = self.config.get('optimization', {})
        self.min_trials = opt_config.get('min_trials', 100)
        self.min_success_rate = opt_config.get('min_success_rate', 0.4)
        self.performance_window_days = opt_config.get('performance_window', 90)

        # Cache for parameter sets
        self._parameter_cache: Dict[str, ParameterSet] = {}
        self._missing_parameters_log: List[Dict[str, Any]] = []

    @cache_decorator('parameter_selection', ttl=7200)  # 2 hours cache
    def get_optimal_parameters(self, context: ParameterContext) -> Optional[ParameterSet]:
        """
        Get optimal parameters for given context using priority-based selection.

        Returns None if no suitable parameters found and fallback strategy is 'skip_pattern'.
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(context)

            # Check cache first
            if cache_key in self._parameter_cache:
                cached_params = self._parameter_cache[cache_key]
                if self._is_cache_valid(cached_params):
                    return cached_params

            # Try each priority level
            for priority in self.priority_order:
                parameter_set = self._get_parameters_for_priority(context, priority)

                if parameter_set and self._validate_parameter_set(parameter_set):
                    # Cache and return
                    self._parameter_cache[cache_key] = parameter_set
                    return parameter_set

            # No parameters found - handle according to fallback strategy
            return self._handle_missing_parameters(context)

        except Exception as e:
            logger.error(f"Error getting optimal parameters: {e}")
            return self._handle_missing_parameters(context)

    def _get_parameters_for_priority(self, context: ParameterContext, priority: str) -> Optional[ParameterSet]:
        """Get parameters for specific priority level"""
        try:
            if priority == 'asset_timeframe_regime':
                return self._query_specific_parameters(
                    context.asset, context.timeframe, context.regime, context.pattern_type
                )
            elif priority == 'asset_timeframe':
                return self._query_specific_parameters(
                    context.asset, context.timeframe, None, context.pattern_type
                )
            elif priority == 'timeframe_regime':
                return self._query_specific_parameters(
                    None, context.timeframe, context.regime, context.pattern_type
                )
            elif priority == 'asset_regime':
                return self._query_specific_parameters(
                    context.asset, None, context.regime, context.pattern_type
                )
            elif priority == 'timeframe':
                return self._query_timeframe_defaults(context.timeframe, context.pattern_type)
            elif priority == 'default':
                return self._get_global_defaults(context.pattern_type)

            return None

        except Exception as e:
            logger.debug(f"Error getting parameters for priority {priority}: {e}")
            return None

    def _query_specific_parameters(self,
                                 asset: Optional[str],
                                 timeframe: Optional[str],
                                 regime: Optional[str],
                                 pattern_type: str) -> Optional[ParameterSet]:
        """Query database for specific parameter combination"""
        try:
            if not self.db_session:
                # Mock database query for development
                return self._mock_database_query(asset, timeframe, regime, pattern_type)

            # Build query based on available filters
            from sqlalchemy import and_

            # Import optimization tables
            from ...training.optimization.task_manager import OptimizationStudy, OptimizationTrial

            query = self.db_session.query(OptimizationStudy).join(OptimizationTrial)

            # Apply filters
            filters = [OptimizationStudy.pattern_key == pattern_type]

            if asset:
                filters.append(OptimizationStudy.asset == asset)
            if timeframe:
                filters.append(OptimizationStudy.timeframe == timeframe)
            if regime:
                filters.append(OptimizationStudy.regime_tag == regime)

            # Only completed studies with sufficient trials
            filters.append(OptimizationStudy.status == 'completed')

            query = query.filter(and_(*filters))

            # Get best performing study
            studies = query.all()

            if studies:
                best_study = self._select_best_study(studies)
                return self._extract_parameter_set_from_study(best_study)

            return None

        except Exception as e:
            logger.debug(f"Error querying specific parameters: {e}")
            return None

    def _mock_database_query(self,
                           asset: Optional[str],
                           timeframe: Optional[str],
                           regime: Optional[str],
                           pattern_type: str) -> Optional[ParameterSet]:
        """Mock database query for development/testing"""
        try:
            # Simulate database lookup with realistic parameters
            mock_parameters = self._get_mock_parameters(asset, timeframe, regime, pattern_type)

            if mock_parameters:
                return ParameterSet(
                    parameters=mock_parameters,
                    performance_metrics={
                        'total_return': 0.15,
                        'sharpe_ratio': 1.2,
                        'success_rate': 0.65,
                        'max_drawdown': 0.08
                    },
                    source=f"mock_{asset}_{timeframe}_{regime}",
                    confidence=0.8,
                    last_updated=datetime.now() - timedelta(days=1),
                    trial_count=150,
                    success_rate=0.65
                )

            return None

        except Exception:
            return None

    def _get_mock_parameters(self,
                           asset: Optional[str],
                           timeframe: Optional[str],
                           regime: Optional[str],
                           pattern_type: str) -> Optional[Dict[str, Any]]:
        """Get mock parameters for testing"""

        # Base parameters for different pattern types
        base_params = {
            'head_shoulders': {
                'min_shoulder_ratio': 0.8,
                'max_shoulder_ratio': 1.2,
                'neckline_slope_tolerance': 0.05,
                'min_formation_periods': 20,
                'max_formation_periods': 100
            },
            'triangle': {
                'min_touches': 4,
                'convergence_ratio': 0.7,
                'slope_tolerance': 0.1,
                'min_formation_periods': 15,
                'breakout_volume_ratio': 1.5
            },
            'double_top': {
                'peak_similarity_ratio': 0.95,
                'valley_depth_ratio': 0.3,
                'min_formation_periods': 10,
                'max_formation_periods': 50
            },
            'flag': {
                'consolidation_ratio': 0.3,
                'pole_min_size': 0.05,
                'flag_max_duration': 20,
                'breakout_confirmation': 2
            }
        }

        # Get base parameters
        params = base_params.get(pattern_type.lower())
        if not params:
            params = base_params.get('triangle')  # Default fallback

        # Adjust parameters based on timeframe
        if timeframe:
            params = self._adjust_parameters_for_timeframe(params.copy(), timeframe)

        # Adjust parameters based on regime
        if regime:
            params = self._adjust_parameters_for_regime(params.copy(), regime)

        # Adjust parameters based on asset
        if asset:
            params = self._adjust_parameters_for_asset(params.copy(), asset)

        return params

    def _adjust_parameters_for_timeframe(self, params: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """Adjust parameters based on timeframe characteristics"""
        tf_multipliers = {
            '1m': {'formation_periods': 0.5, 'tolerance': 1.5},
            '5m': {'formation_periods': 0.7, 'tolerance': 1.3},
            '15m': {'formation_periods': 1.0, 'tolerance': 1.0},
            '1h': {'formation_periods': 1.5, 'tolerance': 0.8},
            '4h': {'formation_periods': 2.0, 'tolerance': 0.6},
            '1d': {'formation_periods': 3.0, 'tolerance': 0.5}
        }

        multiplier = tf_multipliers.get(timeframe, {'formation_periods': 1.0, 'tolerance': 1.0})

        # Adjust formation periods
        for key in params:
            if 'period' in key.lower() and isinstance(params[key], (int, float)):
                params[key] = int(params[key] * multiplier['formation_periods'])

        # Adjust tolerance parameters
        for key in params:
            if 'tolerance' in key.lower() or 'ratio' in key.lower():
                if isinstance(params[key], (int, float)) and key != 'consolidation_ratio':
                    params[key] = params[key] * multiplier['tolerance']

        return params

    def _adjust_parameters_for_regime(self, params: Dict[str, Any], regime: str) -> Dict[str, Any]:
        """Adjust parameters based on market regime"""
        regime_adjustments = {
            'high_volatility': {'tolerance': 1.5, 'periods': 0.8},
            'low_volatility': {'tolerance': 0.7, 'periods': 1.3},
            'trending': {'slope_tolerance': 1.2, 'periods': 1.1},
            'ranging': {'slope_tolerance': 0.8, 'periods': 0.9},
            'bullish': {'breakout_confirmation': 1.2},
            'bearish': {'breakout_confirmation': 1.3}
        }

        # Apply regime-specific adjustments
        for regime_type, adjustments in regime_adjustments.items():
            if regime_type.lower() in regime.lower():
                for adj_key, adj_value in adjustments.items():
                    for param_key in params:
                        if adj_key in param_key.lower() and isinstance(params[param_key], (int, float)):
                            params[param_key] = params[param_key] * adj_value

        return params

    def _adjust_parameters_for_asset(self, params: Dict[str, Any], asset: str) -> Dict[str, Any]:
        """Adjust parameters based on asset characteristics"""
        # Asset-specific adjustments based on typical volatility and behavior
        asset_adjustments = {
            'EUR/USD': {'tolerance': 0.9, 'formation_periods': 1.1},  # Stable major pair
            'GBP/USD': {'tolerance': 1.2, 'formation_periods': 0.9},  # More volatile
            'USD/JPY': {'tolerance': 0.8, 'formation_periods': 1.2},  # Trending tendencies
            'AUD/USD': {'tolerance': 1.1, 'formation_periods': 1.0},  # Commodity-linked
            'USD/CHF': {'tolerance': 0.7, 'formation_periods': 1.3}   # Safe haven
        }

        adjustment = asset_adjustments.get(asset, {'tolerance': 1.0, 'formation_periods': 1.0})

        # Apply adjustments
        for key in params:
            if 'tolerance' in key.lower() and isinstance(params[key], (int, float)):
                params[key] = params[key] * adjustment['tolerance']
            elif 'period' in key.lower() and isinstance(params[key], (int, float)):
                params[key] = int(params[key] * adjustment['formation_periods'])

        return params

    def _query_timeframe_defaults(self, timeframe: str, pattern_type: str) -> Optional[ParameterSet]:
        """Get timeframe-specific default parameters"""
        try:
            # Load timeframe defaults from configuration or database
            timeframe_defaults = self._load_timeframe_defaults()

            params = timeframe_defaults.get(timeframe, {}).get(pattern_type)
            if params:
                return ParameterSet(
                    parameters=params,
                    performance_metrics={'success_rate': 0.5},
                    source=f"timeframe_default_{timeframe}",
                    confidence=0.6,
                    last_updated=datetime.now(),
                    trial_count=50,
                    success_rate=0.5
                )

            return None

        except Exception:
            return None

    def _get_global_defaults(self, pattern_type: str) -> Optional[ParameterSet]:
        """Get global default parameters as last resort"""
        try:
            global_defaults = {
                'head_shoulders': {
                    'min_shoulder_ratio': 0.85,
                    'max_shoulder_ratio': 1.15,
                    'neckline_slope_tolerance': 0.03,
                    'min_formation_periods': 25,
                    'max_formation_periods': 80
                },
                'triangle': {
                    'min_touches': 4,
                    'convergence_ratio': 0.75,
                    'slope_tolerance': 0.08,
                    'min_formation_periods': 20,
                    'breakout_volume_ratio': 1.3
                },
                'double_top': {
                    'peak_similarity_ratio': 0.97,
                    'valley_depth_ratio': 0.25,
                    'min_formation_periods': 15,
                    'max_formation_periods': 40
                },
                'flag': {
                    'consolidation_ratio': 0.25,
                    'pole_min_size': 0.04,
                    'flag_max_duration': 15,
                    'breakout_confirmation': 1.5
                }
            }

            params = global_defaults.get(pattern_type.lower())
            if params:
                return ParameterSet(
                    parameters=params,
                    performance_metrics={'success_rate': 0.45},
                    source="global_default",
                    confidence=0.4,
                    last_updated=datetime.now(),
                    trial_count=25,
                    success_rate=0.45
                )

            return None

        except Exception:
            return None

    def _select_best_study(self, studies: List[Any]) -> Any:
        """Select best performing study from available options"""
        try:
            if self.selection_strategy == 'historical_performance':
                # Select based on performance metrics
                best_study = None
                best_score = -float('inf')

                for study in studies:
                    # Calculate composite performance score
                    score = self._calculate_performance_score(study)
                    if score > best_score:
                        best_score = score
                        best_study = study

                return best_study

            elif self.selection_strategy == 'most_recent':
                # Select most recently updated
                return max(studies, key=lambda s: s.updated_at)

            elif self.selection_strategy == 'highest_confidence':
                # Select with highest trial count
                return max(studies, key=lambda s: len(s.trials))

            else:
                # Default: first available
                return studies[0] if studies else None

        except Exception:
            return studies[0] if studies else None

    def _calculate_performance_score(self, study: Any) -> float:
        """Calculate composite performance score for study"""
        try:
            # Get study metrics (would be calculated from trials)
            total_return = getattr(study, 'total_return', 0.0)
            sharpe_ratio = getattr(study, 'sharpe_ratio', 0.0)
            success_rate = getattr(study, 'success_rate', 0.0)
            max_drawdown = getattr(study, 'max_drawdown', 0.1)

            # Composite score (customize weights as needed)
            score = (
                total_return * 0.3 +
                sharpe_ratio * 0.25 +
                success_rate * 0.3 +
                (1 - max_drawdown) * 0.15  # Lower drawdown is better
            )

            return score

        except Exception:
            return 0.0

    def _extract_parameter_set_from_study(self, study: Any) -> ParameterSet:
        """Extract parameter set from optimization study"""
        try:
            # Get best trial parameters
            best_trial = max(study.trials, key=lambda t: getattr(t, 'value', 0.0))

            return ParameterSet(
                parameters=best_trial.parameters,
                performance_metrics={
                    'total_return': getattr(study, 'total_return', 0.0),
                    'sharpe_ratio': getattr(study, 'sharpe_ratio', 0.0),
                    'success_rate': getattr(study, 'success_rate', 0.0),
                    'max_drawdown': getattr(study, 'max_drawdown', 0.0)
                },
                source=f"study_{study.id}",
                confidence=min(len(study.trials) / self.min_trials, 1.0),
                last_updated=study.updated_at,
                trial_count=len(study.trials),
                success_rate=getattr(study, 'success_rate', 0.0)
            )

        except Exception as e:
            logger.error(f"Error extracting parameter set from study: {e}")
            return None

    def _validate_parameter_set(self, parameter_set: ParameterSet) -> bool:
        """Validate if parameter set meets quality requirements"""
        try:
            # Check minimum trial count
            if parameter_set.trial_count < self.min_trials:
                return False

            # Check minimum success rate
            if parameter_set.success_rate < self.min_success_rate:
                return False

            # Check if parameters are recent enough
            cutoff_date = datetime.now() - timedelta(days=self.performance_window_days)
            if parameter_set.last_updated < cutoff_date:
                return False

            # Check parameter completeness
            if not parameter_set.parameters:
                return False

            return True

        except Exception:
            return False

    def _handle_missing_parameters(self, context: ParameterContext) -> Optional[ParameterSet]:
        """Handle case when no suitable parameters are found"""
        try:
            # Log missing parameters for monitoring
            missing_entry = {
                'timestamp': datetime.now(),
                'asset': context.asset,
                'timeframe': context.timeframe,
                'regime': context.regime,
                'pattern_type': context.pattern_type,
                'reason': 'no_suitable_parameters_found'
            }

            self._missing_parameters_log.append(missing_entry)

            # Log warning
            logger.warning(
                f"Missing parameters for {context.pattern_type} on {context.asset}/{context.timeframe} "
                f"in {context.regime} regime. Pattern will be skipped."
            )

            # Return None to skip pattern according to fallback strategy
            if self.fallback_strategy == 'skip_pattern':
                return None
            elif self.fallback_strategy == 'use_defaults':
                # Force use of global defaults even if they're low quality
                return self._get_global_defaults(context.pattern_type)
            else:
                return None

        except Exception as e:
            logger.error(f"Error handling missing parameters: {e}")
            return None

    def _generate_cache_key(self, context: ParameterContext) -> str:
        """Generate cache key for parameter context"""
        return f"{context.asset}_{context.timeframe}_{context.regime}_{context.pattern_type}"

    def _is_cache_valid(self, parameter_set: ParameterSet) -> bool:
        """Check if cached parameter set is still valid"""
        try:
            # Cache is valid for 2 hours
            cache_expiry = parameter_set.last_updated + timedelta(hours=2)
            return datetime.now() < cache_expiry

        except Exception:
            return False

    def _load_timeframe_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Load timeframe-specific default parameters"""
        # This would typically load from a configuration file or database
        return {
            '1m': {
                'triangle': {'min_touches': 3, 'convergence_ratio': 0.8},
                'flag': {'flag_max_duration': 8}
            },
            '5m': {
                'triangle': {'min_touches': 4, 'convergence_ratio': 0.75},
                'flag': {'flag_max_duration': 12}
            },
            '15m': {
                'triangle': {'min_touches': 4, 'convergence_ratio': 0.7},
                'flag': {'flag_max_duration': 15}
            },
            '1h': {
                'triangle': {'min_touches': 5, 'convergence_ratio': 0.65},
                'flag': {'flag_max_duration': 20}
            },
            '4h': {
                'triangle': {'min_touches': 5, 'convergence_ratio': 0.6},
                'flag': {'flag_max_duration': 25}
            },
            '1d': {
                'triangle': {'min_touches': 6, 'convergence_ratio': 0.55},
                'flag': {'flag_max_duration': 30}
            }
        }

    def get_missing_parameters_log(self) -> List[Dict[str, Any]]:
        """Get log of missing parameters for monitoring"""
        return self._missing_parameters_log.copy()

    def clear_missing_parameters_log(self):
        """Clear the missing parameters log"""
        self._missing_parameters_log.clear()

    def get_parameter_coverage_stats(self) -> Dict[str, Any]:
        """Get statistics about parameter coverage"""
        try:
            total_requests = len(self._parameter_cache) + len(self._missing_parameters_log)
            missing_count = len(self._missing_parameters_log)

            coverage_rate = (total_requests - missing_count) / total_requests if total_requests > 0 else 0.0

            # Group missing by timeframe and pattern type
            missing_by_tf = {}
            missing_by_pattern = {}

            for entry in self._missing_parameters_log:
                tf = entry['timeframe']
                pattern = entry['pattern_type']

                missing_by_tf[tf] = missing_by_tf.get(tf, 0) + 1
                missing_by_pattern[pattern] = missing_by_pattern.get(pattern, 0) + 1

            return {
                'total_requests': total_requests,
                'successful_requests': total_requests - missing_count,
                'missing_requests': missing_count,
                'coverage_rate': coverage_rate,
                'missing_by_timeframe': missing_by_tf,
                'missing_by_pattern': missing_by_pattern
            }

        except Exception:
            return {
                'total_requests': 0,
                'successful_requests': 0,
                'missing_requests': 0,
                'coverage_rate': 0.0,
                'missing_by_timeframe': {},
                'missing_by_pattern': {}
            }