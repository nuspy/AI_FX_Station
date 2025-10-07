"""
Risk Profile Loader

Loads active risk profile from database and applies settings to trading engine.
Supports predefined profiles (Conservative, Moderate, Aggressive) and custom profiles.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from forex_diffusion.training_pipeline.database_models import RiskProfile


@dataclass
class RiskProfileSettings:
    """Risk profile settings container."""

    profile_name: str
    profile_type: str

    # Position sizing
    max_risk_per_trade_pct: float
    max_portfolio_risk_pct: float
    position_sizing_method: str
    kelly_fraction: float

    # Stop loss/take profit
    base_sl_atr_multiplier: float
    base_tp_atr_multiplier: float
    use_trailing_stop: bool
    trailing_activation_pct: Optional[float]

    # Adaptive adjustments
    regime_adjustment_enabled: bool
    volatility_adjustment_enabled: bool
    news_awareness_enabled: bool

    # Diversification
    max_correlated_positions: int
    correlation_threshold: float
    max_positions_per_symbol: int
    max_total_positions: int

    # Drawdown protection
    max_daily_loss_pct: float
    max_drawdown_pct: float
    recovery_mode_threshold_pct: float
    recovery_risk_multiplier: float


class RiskProfileLoader:
    """
    Loads and manages risk profile configuration.

    Features:
    - Load active profile from database
    - Load specific profile by name
    - Activate/deactivate profiles
    - Validate profile settings
    """

    def __init__(self, db_path: str):
        """
        Initialize risk profile loader.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        logger.info("RiskProfileLoader initialized")

    def load_active_profile(self) -> Optional[RiskProfileSettings]:
        """
        Load currently active risk profile.

        Returns:
            RiskProfileSettings if active profile found, None otherwise
        """
        session: Session = self.SessionLocal()
        try:
            profile = session.query(RiskProfile).filter(
                RiskProfile.is_active == True
            ).first()

            if not profile:
                logger.warning("No active risk profile found")
                return None

            settings = self._convert_to_settings(profile)
            logger.info(f"Loaded active profile: {settings.profile_name} ({settings.profile_type})")
            return settings

        finally:
            session.close()

    def load_profile_by_name(self, profile_name: str) -> Optional[RiskProfileSettings]:
        """
        Load specific risk profile by name.

        Args:
            profile_name: Name of profile to load

        Returns:
            RiskProfileSettings if found, None otherwise
        """
        session: Session = self.SessionLocal()
        try:
            profile = session.query(RiskProfile).filter(
                RiskProfile.profile_name == profile_name
            ).first()

            if not profile:
                logger.warning(f"Profile '{profile_name}' not found")
                return None

            settings = self._convert_to_settings(profile)
            logger.info(f"Loaded profile: {settings.profile_name}")
            return settings

        finally:
            session.close()

    def activate_profile(self, profile_name: str) -> bool:
        """
        Activate a risk profile (and deactivate others).

        Args:
            profile_name: Name of profile to activate

        Returns:
            True if successful, False otherwise
        """
        session: Session = self.SessionLocal()
        try:
            # Deactivate all profiles
            session.query(RiskProfile).update({'is_active': False})

            # Activate selected profile
            profile = session.query(RiskProfile).filter(
                RiskProfile.profile_name == profile_name
            ).first()

            if not profile:
                logger.error(f"Profile '{profile_name}' not found")
                return False

            profile.is_active = True
            session.commit()

            logger.info(f"✅ Activated risk profile: {profile_name}")
            return True

        except Exception as e:
            logger.error(f"Error activating profile: {e}")
            session.rollback()
            return False

        finally:
            session.close()

    def list_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available risk profiles.

        Returns:
            Dict mapping profile name to summary info
        """
        session: Session = self.SessionLocal()
        try:
            profiles = session.query(RiskProfile).all()

            result = {}
            for profile in profiles:
                result[profile.profile_name] = {
                    'type': profile.profile_type,
                    'is_active': profile.is_active,
                    'max_risk_per_trade_pct': profile.max_risk_per_trade_pct,
                    'max_positions': profile.max_total_positions,
                    'position_sizing_method': profile.position_sizing_method,
                    'description': profile.description,
                }

            return result

        finally:
            session.close()

    def _convert_to_settings(self, profile: RiskProfile) -> RiskProfileSettings:
        """Convert ORM model to settings dataclass."""
        return RiskProfileSettings(
            profile_name=profile.profile_name,
            profile_type=profile.profile_type,
            max_risk_per_trade_pct=profile.max_risk_per_trade_pct,
            max_portfolio_risk_pct=profile.max_portfolio_risk_pct,
            position_sizing_method=profile.position_sizing_method,
            kelly_fraction=profile.kelly_fraction,
            base_sl_atr_multiplier=profile.base_sl_atr_multiplier,
            base_tp_atr_multiplier=profile.base_tp_atr_multiplier,
            use_trailing_stop=profile.use_trailing_stop,
            trailing_activation_pct=profile.trailing_activation_pct,
            regime_adjustment_enabled=profile.regime_adjustment_enabled,
            volatility_adjustment_enabled=profile.volatility_adjustment_enabled,
            news_awareness_enabled=profile.news_awareness_enabled,
            max_correlated_positions=profile.max_correlated_positions,
            correlation_threshold=profile.correlation_threshold,
            max_positions_per_symbol=profile.max_positions_per_symbol,
            max_total_positions=profile.max_total_positions,
            max_daily_loss_pct=profile.max_daily_loss_pct,
            max_drawdown_pct=profile.max_drawdown_pct,
            recovery_mode_threshold_pct=profile.recovery_mode_threshold_pct,
            recovery_risk_multiplier=profile.recovery_risk_multiplier,
        )

    def get_default_settings(self) -> RiskProfileSettings:
        """
        Get default risk profile settings (Moderate profile).

        Used as fallback if no profile is active.
        """
        return RiskProfileSettings(
            profile_name='Default',
            profile_type='default',
            max_risk_per_trade_pct=1.0,
            max_portfolio_risk_pct=5.0,
            position_sizing_method='kelly',
            kelly_fraction=0.25,
            base_sl_atr_multiplier=2.0,
            base_tp_atr_multiplier=3.0,
            use_trailing_stop=True,
            trailing_activation_pct=50.0,
            regime_adjustment_enabled=True,
            volatility_adjustment_enabled=True,
            news_awareness_enabled=True,
            max_correlated_positions=2,
            correlation_threshold=0.7,
            max_positions_per_symbol=2,
            max_total_positions=5,
            max_daily_loss_pct=2.0,
            max_drawdown_pct=10.0,
            recovery_mode_threshold_pct=5.0,
            recovery_risk_multiplier=0.5,
        )
