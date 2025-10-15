"""
Initialize Predefined Risk Profiles

Creates Conservative, Moderate, and Aggressive risk profiles in the database.
These can be used as templates or activated for trading.

Run this script after running migrations to populate the risk_profiles table.
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from forex_diffusion.training_pipeline.database_models import RiskProfile
from loguru import logger


def create_predefined_profiles(db_path: str = "forex_data.db"):
    """Create predefined risk profiles in database."""

    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Check if profiles already exist
        existing = session.query(RiskProfile).filter(
            RiskProfile.profile_type == 'predefined'
        ).count()

        if existing > 0:
            logger.warning(f"Found {existing} existing predefined profiles. Skipping creation.")
            return

        # Conservative Profile
        conservative = RiskProfile(
            profile_name='Conservative',
            profile_type='predefined',
            is_active=False,
            description='Low risk, capital preservation focused. Suitable for beginners or risk-averse traders.',

            # Position sizing
            max_risk_per_trade_pct=0.5,  # 0.5% per trade
            max_portfolio_risk_pct=2.0,  # 2% total
            position_sizing_method='fixed_fractional',
            kelly_fraction=0.1,  # Very conservative Kelly

            # Stop loss/take profit
            base_sl_atr_multiplier=2.5,  # Wider stops
            base_tp_atr_multiplier=4.0,  # Higher R:R
            use_trailing_stop=True,
            trailing_activation_pct=60.0,  # Trail after 60% to target

            # Adaptive adjustments
            regime_adjustment_enabled=True,
            volatility_adjustment_enabled=True,
            news_awareness_enabled=True,

            # Diversification
            max_correlated_positions=1,  # Only 1 correlated position
            correlation_threshold=0.6,
            max_positions_per_symbol=1,
            max_total_positions=3,  # Max 3 positions total

            # Drawdown protection
            max_daily_loss_pct=1.0,  # Stop after 1% daily loss
            max_drawdown_pct=5.0,  # Stop after 5% drawdown
            recovery_mode_threshold_pct=3.0,
            recovery_risk_multiplier=0.5,  # Half risk in recovery
        )

        # Moderate Profile
        moderate = RiskProfile(
            profile_name='Moderate',
            profile_type='predefined',
            is_active=False,
            description='Balanced risk/reward. Suitable for experienced traders with moderate risk tolerance.',

            # Position sizing
            max_risk_per_trade_pct=1.0,  # 1% per trade
            max_portfolio_risk_pct=5.0,  # 5% total
            position_sizing_method='kelly',
            kelly_fraction=0.25,  # Quarter Kelly

            # Stop loss/take profit
            base_sl_atr_multiplier=2.0,
            base_tp_atr_multiplier=3.0,
            use_trailing_stop=True,
            trailing_activation_pct=50.0,

            # Adaptive adjustments
            regime_adjustment_enabled=True,
            volatility_adjustment_enabled=True,
            news_awareness_enabled=True,

            # Diversification
            max_correlated_positions=2,
            correlation_threshold=0.7,
            max_positions_per_symbol=2,
            max_total_positions=5,

            # Drawdown protection
            max_daily_loss_pct=2.0,
            max_drawdown_pct=10.0,
            recovery_mode_threshold_pct=5.0,
            recovery_risk_multiplier=0.5,
        )

        # Aggressive Profile
        aggressive = RiskProfile(
            profile_name='Aggressive',
            profile_type='predefined',
            is_active=False,
            description='High risk/high reward. For experienced traders with high risk tolerance and sufficient capital.',

            # Position sizing
            max_risk_per_trade_pct=2.0,  # 2% per trade
            max_portfolio_risk_pct=10.0,  # 10% total
            position_sizing_method='optimal_f',
            kelly_fraction=0.5,  # Half Kelly (still safer than full)

            # Stop loss/take profit
            base_sl_atr_multiplier=1.5,  # Tighter stops
            base_tp_atr_multiplier=2.5,
            use_trailing_stop=True,
            trailing_activation_pct=40.0,  # Trail earlier

            # Adaptive adjustments
            regime_adjustment_enabled=True,
            volatility_adjustment_enabled=True,
            news_awareness_enabled=False,  # Trade through news

            # Diversification
            max_correlated_positions=3,
            correlation_threshold=0.8,  # Less strict
            max_positions_per_symbol=3,
            max_total_positions=10,

            # Drawdown protection
            max_daily_loss_pct=5.0,
            max_drawdown_pct=20.0,
            recovery_mode_threshold_pct=10.0,
            recovery_risk_multiplier=0.7,  # Less reduction
        )

        # Add to session
        session.add(conservative)
        session.add(moderate)
        session.add(aggressive)

        # Commit
        session.commit()

        logger.info("✅ Created 3 predefined risk profiles:")
        logger.info("  - Conservative: 0.5% risk/trade, max 3 positions")
        logger.info("  - Moderate: 1.0% risk/trade, max 5 positions")
        logger.info("  - Aggressive: 2.0% risk/trade, max 10 positions")

    except Exception as e:
        logger.error(f"Error creating profiles: {e}")
        session.rollback()
        raise

    finally:
        session.close()


if __name__ == "__main__":
    logger.info("Initializing predefined risk profiles...")
    create_predefined_profiles()
    logger.info("Done!")
