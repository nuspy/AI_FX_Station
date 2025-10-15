"""
Parameter Loader Service

Loads optimized parameters from database with caching and fallback logic.
Integrates with automated trading engine to use backtesting-optimized parameters
instead of hard-coded defaults.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import json
from dataclasses import dataclass
from loguru import logger
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session

from forex_diffusion.training_pipeline.database_models import OptimizedParameters


@dataclass
class ParameterSet:
    """Container for optimized parameters."""

    pattern_type: str
    symbol: str
    timeframe: str
    market_regime: Optional[str]
    form_params: Dict[str, Any]
    action_params: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    optimization_timestamp: datetime
    validation_status: str
    source: str  # 'database', 'cache', 'default'


class ParameterLoaderService:
    """
    Loads optimized parameters from database with intelligent fallback.

    Priority order:
    1. Regime-specific optimized parameters (if regime provided)
    2. Generic optimized parameters for pattern/symbol/timeframe
    3. Pattern-specific defaults
    4. Global defaults

    Features:
    - In-memory caching with TTL
    - Validation status filtering
    - Performance-based selection
    - Comprehensive logging
    """

    def __init__(
        self,
        db_path: str,
        cache_ttl_seconds: int = 3600,
        require_validation: bool = True
    ):
        """
        Initialize parameter loader.

        Args:
            db_path: Path to SQLite database
            cache_ttl_seconds: Cache time-to-live in seconds (default 1 hour)
            require_validation: Only load validated parameters (default True)
        """
        self.db_path = db_path
        self.cache_ttl_seconds = cache_ttl_seconds
        self.require_validation = require_validation

        # Database session
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Cache: key = (pattern, symbol, tf, regime) -> (params, timestamp)
        self._cache: Dict[tuple, tuple[ParameterSet, datetime]] = {}

        # Default parameters by pattern type
        self._default_params = self._initialize_defaults()

        logger.info(
            f"ParameterLoaderService initialized: "
            f"cache_ttl={cache_ttl_seconds}s, "
            f"require_validation={require_validation}"
        )

    def _initialize_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default parameters for each pattern type."""
        return {
            'harmonic': {
                'form_params': {
                    'tolerance_pct': 2.0,
                    'min_pattern_size': 50,
                    'require_confluence': True,
                },
                'action_params': {
                    'sl_atr_multiplier': 2.0,
                    'tp_atr_multiplier': 3.0,
                    'min_rr_ratio': 2.0,
                    'partial_close_enabled': True,
                    'partial_close_pct': 50.0,
                },
            },
            'orderflow': {
                'form_params': {
                    'imbalance_threshold': 0.3,
                    'volume_spike_multiplier': 2.0,
                    'min_depth_levels': 5,
                },
                'action_params': {
                    'sl_atr_multiplier': 1.5,
                    'tp_atr_multiplier': 2.5,
                    'min_rr_ratio': 1.5,
                },
            },
            'correlation': {
                'form_params': {
                    'correlation_threshold': 0.7,
                    'divergence_threshold': 0.3,
                    'lookback_periods': 20,
                },
                'action_params': {
                    'sl_atr_multiplier': 2.0,
                    'tp_atr_multiplier': 2.5,
                    'min_rr_ratio': 1.5,
                },
            },
            'pattern': {
                'form_params': {
                    'pattern_confidence_threshold': 0.7,
                    'require_volume_confirmation': True,
                },
                'action_params': {
                    'sl_atr_multiplier': 2.0,
                    'tp_atr_multiplier': 3.0,
                    'min_rr_ratio': 2.0,
                },
            },
        }

    def load_parameters(
        self,
        pattern_type: str,
        symbol: str,
        timeframe: str,
        market_regime: Optional[str] = None
    ) -> ParameterSet:
        """
        Load best parameters for given pattern/symbol/timeframe/regime.

        Priority:
        1. Cache (if valid)
        2. Database regime-specific
        3. Database generic (regime=None)
        4. Defaults

        Args:
            pattern_type: Pattern type (harmonic, orderflow, etc.)
            symbol: Trading symbol
            timeframe: Timeframe string
            market_regime: Optional regime filter

        Returns:
            ParameterSet with optimized or default parameters
        """
        cache_key = (pattern_type, symbol, timeframe, market_regime)

        # Check cache
        if cache_key in self._cache:
            params, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < timedelta(seconds=self.cache_ttl_seconds):
                logger.debug(
                    f"Cache hit: {pattern_type}/{symbol}/{timeframe}/{market_regime}"
                )
                return params
            else:
                # Cache expired
                del self._cache[cache_key]
                logger.debug(f"Cache expired for {cache_key}")

        # Load from database
        db_params = self._load_from_database(
            pattern_type, symbol, timeframe, market_regime
        )

        if db_params:
            # Cache and return
            self._cache[cache_key] = (db_params, datetime.now())
            logger.info(
                f"Loaded optimized params from DB: {pattern_type}/{symbol}/{timeframe}/{market_regime} "
                f"(optimized {db_params.optimization_timestamp.strftime('%Y-%m-%d')})"
            )
            return db_params

        # Fallback to defaults
        logger.warning(
            f"No optimized params found, using defaults: "
            f"{pattern_type}/{symbol}/{timeframe}/{market_regime}"
        )
        return self._create_default_params(pattern_type, symbol, timeframe, market_regime)

    def _load_from_database(
        self,
        pattern_type: str,
        symbol: str,
        timeframe: str,
        market_regime: Optional[str]
    ) -> Optional[ParameterSet]:
        """Load parameters from database with regime fallback."""

        session: Session = self.SessionLocal()
        try:
            # Try regime-specific first (if regime provided)
            if market_regime:
                params = self._query_parameters(
                    session, pattern_type, symbol, timeframe, market_regime
                )
                if params:
                    return self._convert_to_parameter_set(params, 'database')

            # Try generic (regime=None)
            params = self._query_parameters(
                session, pattern_type, symbol, timeframe, None
            )
            if params:
                return self._convert_to_parameter_set(params, 'database')

            return None

        finally:
            session.close()

    def _query_parameters(
        self,
        session: Session,
        pattern_type: str,
        symbol: str,
        timeframe: str,
        market_regime: Optional[str]
    ) -> Optional[OptimizedParameters]:
        """
        Query database for best parameters.

        Selection logic:
        - Filter by pattern/symbol/timeframe/regime
        - Filter by validation status (if required)
        - Order by optimization_timestamp (most recent first)
        - Return first (best) result
        """
        query = session.query(OptimizedParameters).filter(
            OptimizedParameters.pattern_type == pattern_type,
            OptimizedParameters.symbol == symbol,
            OptimizedParameters.timeframe == timeframe,
        )

        # Regime filter
        if market_regime is not None:
            query = query.filter(OptimizedParameters.market_regime == market_regime)
        else:
            query = query.filter(OptimizedParameters.market_regime.is_(None))

        # Validation filter
        if self.require_validation:
            query = query.filter(
                OptimizedParameters.validation_status.in_(['validated', 'deployed'])
            )

        # Order by most recent
        query = query.order_by(desc(OptimizedParameters.optimization_timestamp))

        return query.first()

    def _convert_to_parameter_set(
        self,
        db_params: OptimizedParameters,
        source: str
    ) -> ParameterSet:
        """Convert database model to ParameterSet."""
        return ParameterSet(
            pattern_type=db_params.pattern_type,
            symbol=db_params.symbol,
            timeframe=db_params.timeframe,
            market_regime=db_params.market_regime,
            form_params=json.loads(db_params.form_params),
            action_params=json.loads(db_params.action_params),
            performance_metrics=json.loads(db_params.performance_metrics),
            optimization_timestamp=db_params.optimization_timestamp,
            validation_status=db_params.validation_status,
            source=source,
        )

    def _create_default_params(
        self,
        pattern_type: str,
        symbol: str,
        timeframe: str,
        market_regime: Optional[str]
    ) -> ParameterSet:
        """Create default parameter set."""

        # Get defaults for pattern type
        defaults = self._default_params.get(
            pattern_type,
            self._default_params['pattern']  # Generic fallback
        )

        return ParameterSet(
            pattern_type=pattern_type,
            symbol=symbol,
            timeframe=timeframe,
            market_regime=market_regime,
            form_params=defaults['form_params'].copy(),
            action_params=defaults['action_params'].copy(),
            performance_metrics={},
            optimization_timestamp=datetime.now(),
            validation_status='default',
            source='default',
        )

    def clear_cache(self):
        """Clear parameter cache."""
        self._cache.clear()
        logger.info("Parameter cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        valid_entries = sum(
            1 for _, (_, cached_at) in self._cache.items()
            if now - cached_at < timedelta(seconds=self.cache_ttl_seconds)
        )

        return {
            'total_entries': len(self._cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self._cache) - valid_entries,
            'cache_ttl_seconds': self.cache_ttl_seconds,
        }

    def preload_parameters(
        self,
        pattern_types: list[str],
        symbols: list[str],
        timeframes: list[str],
        regimes: Optional[list[str]] = None
    ):
        """
        Preload parameters into cache for frequently used combinations.

        Useful for warming up cache at startup.
        """
        regimes = regimes or [None]
        count = 0

        for pattern in pattern_types:
            for symbol in symbols:
                for tf in timeframes:
                    for regime in regimes:
                        self.load_parameters(pattern, symbol, tf, regime)
                        count += 1

        logger.info(f"Preloaded {count} parameter sets into cache")
