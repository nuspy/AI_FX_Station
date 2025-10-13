"""
Parameter Refresh Manager

Automatically checks and refreshes optimized parameters based on:
- Age (e.g., > 90 days old)
- Performance degradation (e.g., >15% win rate drop)
- Market regime changes

Triggers re-optimization when refresh is needed.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger


@dataclass
class RefreshPolicy:
    """Policy for parameter refresh"""
    max_age_days: int = 90  # Refresh after 90 days
    min_performance_degradation: float = 0.15  # Refresh if performance drops >15%
    check_interval_hours: int = 24  # Check every 24 hours
    min_samples_for_comparison: int = 20  # Min trades to assess degradation


@dataclass
class RefreshDecision:
    """Decision about parameter refresh"""
    should_refresh: bool
    reason: str
    study_info: Dict[str, Any]
    priority: str  # "high", "medium", "low"


class ParameterRefreshManager:
    """
    Manages automatic refresh of optimized parameters.
    
    Features:
    - Age-based refresh (parameters too old)
    - Performance-based refresh (win rate degradation)
    - Regime-based refresh (market conditions changed)
    """
    
    def __init__(self, db_session, refresh_policy: Optional[RefreshPolicy] = None):
        """
        Initialize refresh manager.
        
        Args:
            db_session: Database session for querying studies
            refresh_policy: Refresh policy configuration
        """
        self.db_session = db_session
        self.policy = refresh_policy or RefreshPolicy()
        self._last_check_time: Optional[datetime] = None
    
    def check_all_studies(self) -> List[RefreshDecision]:
        """
        Check all studies and return list of refresh decisions.
        
        Returns:
            List of RefreshDecision for studies that need refresh
        """
        try:
            # Skip if checked recently
            if self._last_check_time:
                time_since_check = (datetime.now() - self._last_check_time).total_seconds() / 3600
                if time_since_check < self.policy.check_interval_hours:
                    logger.debug(f"Skipping refresh check (last check {time_since_check:.1f}h ago)")
                    return []
            
            self._last_check_time = datetime.now()
            
            # Get all active studies from database
            studies = self._get_all_studies()
            
            refresh_decisions = []
            for study in studies:
                decision = self._check_study_refresh(study)
                if decision.should_refresh:
                    refresh_decisions.append(decision)
            
            if refresh_decisions:
                logger.info(f"Found {len(refresh_decisions)} studies needing refresh")
            else:
                logger.debug("No studies need refresh")
            
            return refresh_decisions
            
        except Exception as e:
            logger.error(f"Error checking studies for refresh: {e}")
            return []
    
    def _check_study_refresh(self, study: Dict[str, Any]) -> RefreshDecision:
        """
        Check if a single study needs refresh.
        
        Args:
            study: Study info from database
            
        Returns:
            RefreshDecision with refresh recommendation
        """
        study_id = study.get('id')
        pattern_key = study.get('pattern_key')
        last_updated = study.get('last_updated')
        original_performance = study.get('original_performance', {})
        
        # Check 1: Age-based refresh
        age_days = self._get_age_days(last_updated)
        if age_days > self.policy.max_age_days:
            return RefreshDecision(
                should_refresh=True,
                reason=f"Parameters are {age_days} days old (> {self.policy.max_age_days} day threshold)",
                study_info=study,
                priority="high" if age_days > self.policy.max_age_days * 1.5 else "medium"
            )
        
        # Check 2: Performance degradation
        recent_performance = self._get_recent_performance(study)
        if recent_performance is not None and original_performance:
            original_win_rate = original_performance.get('win_rate', 0.5)
            degradation = original_win_rate - recent_performance
            
            if degradation > self.policy.min_performance_degradation:
                return RefreshDecision(
                    should_refresh=True,
                    reason=f"Performance degraded by {degradation:.1%} "
                           f"(original: {original_win_rate:.1%}, recent: {recent_performance:.1%})",
                    study_info=study,
                    priority="high"
                )
        
        # Check 3: Regime change (if tracked)
        if self._has_regime_changed(study):
            return RefreshDecision(
                should_refresh=True,
                reason="Market regime changed since optimization",
                study_info=study,
                priority="medium"
            )
        
        # No refresh needed
        return RefreshDecision(
            should_refresh=False,
            reason="Parameters still valid",
            study_info=study,
            priority="low"
        )
    
    def _get_age_days(self, last_updated: Optional[datetime]) -> int:
        """Calculate age in days"""
        if not last_updated:
            return 999  # Very old if no timestamp
        return (datetime.now() - last_updated).days
    
    def _get_recent_performance(self, study: Dict[str, Any]) -> Optional[float]:
        """
        Get recent performance (last N days) for this study's parameters.
        
        Returns:
            Recent win rate or None if insufficient data
        """
        try:
            # Query recent pattern outcomes for this study
            # This would need integration with confidence_calibrator or outcomes table
            # For now, return None (not implemented)
            
            # TODO: Implement by querying PatternOutcome table filtered by:
            # - pattern_key = study['pattern_key']
            # - detection_date > last N days (e.g., 30 days)
            # - Calculate win_rate from outcomes
            
            return None
            
        except Exception as e:
            logger.debug(f"Error getting recent performance: {e}")
            return None
    
    def _has_regime_changed(self, study: Dict[str, Any]) -> bool:
        """
        Check if market regime has changed since optimization.
        
        Returns:
            True if regime changed
        """
        try:
            # Would need integration with RegimeClassifier
            # For now, return False
            
            # TODO: Implement by:
            # 1. Get original regime from study
            # 2. Get current regime from RegimeClassifier
            # 3. Compare
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking regime change: {e}")
            return False
    
    def _get_all_studies(self) -> List[Dict[str, Any]]:
        """
        Get all optimization studies from database.
        
        Returns:
            List of study dicts
        """
        try:
            if not self.db_session:
                logger.warning("No database session, cannot get studies")
                return []
            
            # TODO: Query OptimizationStudy table
            # For now, return empty list
            
            # Example query (when models are available):
            # from ...data.models import OptimizationStudy
            # studies = self.db_session.query(OptimizationStudy).filter(
            #     OptimizationStudy.status == 'completed'
            # ).all()
            # return [s.to_dict() for s in studies]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting studies from database: {e}")
            return []
    
    def queue_reoptimization(self, study_info: Dict[str, Any], priority: str = "medium") -> bool:
        """
        Queue a study for re-optimization.
        
        Args:
            study_info: Study to re-optimize
            priority: Priority level ("high", "medium", "low")
            
        Returns:
            True if successfully queued
        """
        try:
            # TODO: Integrate with OptimizationEngine to trigger new study
            # For now, just log
            
            pattern_key = study_info.get('pattern_key')
            asset = study_info.get('asset')
            timeframe = study_info.get('timeframe')
            
            logger.info(
                f"Queued re-optimization for {pattern_key} "
                f"({asset}/{timeframe}) with priority={priority}"
            )
            
            # Example integration:
            # from ...training.optimization.engine import OptimizationEngine
            # engine = OptimizationEngine(config)
            # config = OptimizationConfig(
            #     pattern_key=pattern_key,
            #     asset=asset,
            #     timeframe=timeframe,
            #     ...
            # )
            # engine.run_optimization(config)
            
            return True
            
        except Exception as e:
            logger.error(f"Error queuing re-optimization: {e}")
            return False


def auto_refresh_parameters(db_session, policy: Optional[RefreshPolicy] = None) -> List[RefreshDecision]:
    """
    Convenience function to check and queue parameter refreshes.
    
    Args:
        db_session: Database session
        policy: Optional custom refresh policy
        
    Returns:
        List of refresh decisions that were queued
    """
    manager = ParameterRefreshManager(db_session, policy)
    
    # Check all studies
    decisions = manager.check_all_studies()
    
    # Queue high-priority refreshes
    queued = []
    for decision in decisions:
        if decision.priority == "high":
            success = manager.queue_reoptimization(
                decision.study_info,
                decision.priority
            )
            if success:
                queued.append(decision)
    
    return queued
