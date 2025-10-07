"""
Database ORM models and access layer for Two-Phase Training System.

Provides SQLAlchemy models for all training pipeline tables and utility functions
for database operations.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from contextlib import contextmanager
import uuid

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, JSON, Text,
    ForeignKey, create_engine, func, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool

from forex_diffusion.config import settings

Base = declarative_base()


# ==================== ORM Models ====================

class TrainingRun(Base):
    """Tracks every model training attempt with full configuration."""

    __tablename__ = 'training_runs'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_uuid = Column(String(36), nullable=False, unique=True, default=lambda: str(uuid.uuid4()))

    # Status
    status = Column(String(20), nullable=False, default='pending')  # pending, running, completed, failed, cancelled

    # Model Configuration
    model_type = Column(String(50), nullable=False)  # diffusion, random_forest, gradient_boosting, etc.
    encoder = Column(String(50), nullable=False)  # none, vae, autoencoder
    symbol = Column(String(20), nullable=False)
    base_timeframe = Column(String(10), nullable=False)
    days_history = Column(Integer, nullable=False)
    horizon = Column(Integer, nullable=False)

    # Feature Configuration (JSON)
    indicator_tfs = Column(JSON, nullable=True)  # List of indicator timeframes
    additional_features = Column(JSON, nullable=True)  # List of feature names
    preprocessing_params = Column(JSON, nullable=True)  # Preprocessing config
    model_hyperparams = Column(JSON, nullable=True)  # Model-specific hyperparameters

    # Training Results
    training_metrics = Column(JSON, nullable=True)  # Training performance metrics
    feature_count = Column(Integer, nullable=True)
    training_duration_seconds = Column(Float, nullable=True)

    # File Management
    model_file_path = Column(String(500), nullable=True)
    model_file_size_bytes = Column(Integer, nullable=True)
    is_model_kept = Column(Boolean, default=False)

    # Regime Performance
    best_regimes = Column(JSON, nullable=True)  # List of regime names where this is best

    # Timestamps
    created_at = Column(DateTime, server_default=func.current_timestamp())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Provenance
    created_by = Column(String(50), default='system')
    config_hash = Column(String(64), nullable=False)  # SHA256 hash of configuration

    # Relationships
    inference_backtests = relationship("InferenceBacktest", back_populates="training_run", cascade="all, delete-orphan")
    regime_best_models = relationship("RegimeBestModel", back_populates="training_run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TrainingRun(id={self.id}, uuid={self.run_uuid[:8]}..., status={self.status}, model={self.model_type})>"


class InferenceBacktest(Base):
    """Tracks inference backtests on trained models."""

    __tablename__ = 'inference_backtests'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_uuid = Column(String(36), nullable=False, unique=True, default=lambda: str(uuid.uuid4()))

    # Foreign Key
    training_run_id = Column(Integer, ForeignKey('training_runs.id', ondelete='CASCADE'), nullable=False)

    # Inference Configuration
    prediction_method = Column(String(50), nullable=False)  # direct, recursive, direct_multi
    ensemble_method = Column(String(50), nullable=True)  # mean, weighted, stacking
    confidence_threshold = Column(Float, nullable=True)  # 0.0 - 1.0
    lookback_window = Column(Integer, nullable=True)  # Bars for prediction
    inference_params = Column(JSON, nullable=True)  # Additional inference parameters

    # Backtest Results
    backtest_metrics = Column(JSON, nullable=True)  # Overall performance metrics
    backtest_duration_seconds = Column(Float, nullable=True)
    regime_metrics = Column(JSON, nullable=True)  # Performance by regime

    # Timestamps
    created_at = Column(DateTime, server_default=func.current_timestamp())
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    training_run = relationship("TrainingRun", back_populates="inference_backtests")
    regime_best_models = relationship("RegimeBestModel", back_populates="inference_backtest", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<InferenceBacktest(id={self.id}, run_id={self.training_run_id}, method={self.prediction_method})>"


class RegimeDefinition(Base):
    """Defines market regimes with detection rules."""

    __tablename__ = 'regime_definitions'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Regime Info
    regime_name = Column(String(50), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    detection_rules = Column(JSON, nullable=True)  # Rules for regime classification

    # Timestamps
    created_at = Column(DateTime, server_default=func.current_timestamp())
    is_active = Column(Boolean, default=True)

    # Relationships
    regime_best_models = relationship("RegimeBestModel", back_populates="regime_definition", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<RegimeDefinition(id={self.id}, name={self.regime_name}, active={self.is_active})>"


class RegimeBestModel(Base):
    """Tracks best performing model per regime."""

    __tablename__ = 'regime_best_models'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign Keys
    regime_name = Column(String(50), ForeignKey('regime_definitions.regime_name', ondelete='CASCADE'), nullable=False)
    training_run_id = Column(Integer, ForeignKey('training_runs.id', ondelete='CASCADE'), nullable=False)
    inference_backtest_id = Column(Integer, ForeignKey('inference_backtests.id', ondelete='CASCADE'), nullable=False)

    # Performance Metrics
    performance_score = Column(Float, nullable=False)  # Primary metric (e.g., Sharpe ratio)
    secondary_metrics = Column(JSON, nullable=True)  # Other metrics

    # Timestamp
    achieved_at = Column(DateTime, server_default=func.current_timestamp())

    # Unique constraint: one best model per regime
    __table_args__ = (
        UniqueConstraint('regime_name', name='uq_regime_best_model'),
    )

    # Relationships
    regime_definition = relationship("RegimeDefinition", back_populates="regime_best_models")
    training_run = relationship("TrainingRun", back_populates="regime_best_models")
    inference_backtest = relationship("InferenceBacktest", back_populates="regime_best_models")

    def __repr__(self):
        return f"<RegimeBestModel(id={self.id}, regime={self.regime_name}, score={self.performance_score:.4f})>"


class TrainingQueue(Base):
    """Manages queued training jobs for interruption/resume."""

    __tablename__ = 'training_queue'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    queue_uuid = Column(String(36), nullable=False, unique=True, default=lambda: str(uuid.uuid4()))

    # Configuration
    config_grid = Column(JSON, nullable=False)  # List of all configurations to train
    current_index = Column(Integer, default=0)  # Current position in grid
    total_configs = Column(Integer, nullable=False)

    # Status
    status = Column(String(20), nullable=False, default='pending')  # pending, running, paused, completed, cancelled

    # Progress Tracking
    completed_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    skipped_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, server_default=func.current_timestamp())
    started_at = Column(DateTime, nullable=True)
    paused_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Settings
    priority = Column(Integer, default=0)  # Higher priority queues run first
    max_parallel = Column(Integer, default=1)  # Max parallel training runs

    def __repr__(self):
        return f"<TrainingQueue(id={self.id}, status={self.status}, progress={self.completed_count}/{self.total_configs})>"


# ==================== Database Session Management ====================

# Global session factory
_SessionFactory: Optional[sessionmaker] = None


def init_db(database_url: Optional[str] = None) -> None:
    """
    Initialize database connection and session factory.

    Args:
        database_url: SQLAlchemy database URL. If None, uses settings.database_path
    """
    global _SessionFactory

    if database_url is None:
        database_url = f"sqlite:///{settings.database_path}"

    # Create engine
    if database_url.startswith("sqlite"):
        # SQLite-specific settings
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
    else:
        # PostgreSQL/MySQL settings
        engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False
        )

    # Create session factory
    _SessionFactory = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_session() -> Session:
    """
    Get a new database session.

    Returns:
        SQLAlchemy Session object

    Raises:
        RuntimeError: If database not initialized
    """
    if _SessionFactory is None:
        init_db()

    return _SessionFactory()


@contextmanager
def session_scope():
    """
    Context manager for database sessions with automatic commit/rollback.

    Usage:
        with session_scope() as session:
            session.add(obj)
            # Automatic commit on success, rollback on exception
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ==================== CRUD Operations ====================

# --- TrainingRun Operations ---

def create_training_run(
    session: Session,
    model_type: str,
    encoder: str,
    symbol: str,
    base_timeframe: str,
    days_history: int,
    horizon: int,
    config_hash: str,
    indicator_tfs: Optional[List[str]] = None,
    additional_features: Optional[List[str]] = None,
    preprocessing_params: Optional[Dict[str, Any]] = None,
    model_hyperparams: Optional[Dict[str, Any]] = None,
    created_by: str = 'system'
) -> TrainingRun:
    """Create a new training run record."""
    run = TrainingRun(
        model_type=model_type,
        encoder=encoder,
        symbol=symbol,
        base_timeframe=base_timeframe,
        days_history=days_history,
        horizon=horizon,
        config_hash=config_hash,
        indicator_tfs=indicator_tfs,
        additional_features=additional_features,
        preprocessing_params=preprocessing_params,
        model_hyperparams=model_hyperparams,
        created_by=created_by,
        status='pending'
    )
    session.add(run)
    session.flush()
    return run


def get_training_run_by_id(session: Session, run_id: int) -> Optional[TrainingRun]:
    """Get training run by ID."""
    return session.query(TrainingRun).filter(TrainingRun.id == run_id).first()


def get_training_run_by_uuid(session: Session, run_uuid: str) -> Optional[TrainingRun]:
    """Get training run by UUID."""
    return session.query(TrainingRun).filter(TrainingRun.run_uuid == run_uuid).first()


def get_training_run_by_config_hash(session: Session, config_hash: str) -> Optional[TrainingRun]:
    """Check if configuration has already been trained."""
    return session.query(TrainingRun).filter(
        TrainingRun.config_hash == config_hash,
        TrainingRun.status == 'completed'
    ).first()


def update_training_run_status(
    session: Session,
    run_id: int,
    status: str,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None
) -> None:
    """Update training run status and timestamps."""
    run = session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if run:
        run.status = status
        if started_at:
            run.started_at = started_at
        if completed_at:
            run.completed_at = completed_at
        session.flush()


def update_training_run_results(
    session: Session,
    run_id: int,
    training_metrics: Dict[str, Any],
    feature_count: int,
    training_duration_seconds: float,
    model_file_path: str,
    model_file_size_bytes: int
) -> None:
    """Update training run with results."""
    run = session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if run:
        run.training_metrics = training_metrics
        run.feature_count = feature_count
        run.training_duration_seconds = training_duration_seconds
        run.model_file_path = model_file_path
        run.model_file_size_bytes = model_file_size_bytes
        session.flush()


def mark_model_as_kept(session: Session, run_id: int, best_regimes: List[str]) -> None:
    """Mark model as kept with list of regimes it's best for."""
    run = session.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if run:
        run.is_model_kept = True
        run.best_regimes = best_regimes
        session.flush()


def delete_training_run(session: Session, training_run_id: int) -> bool:
    """
    Delete a training run and all associated records.

    Cascades to:
    - InferenceBacktest records
    - Model file (if exists)

    Args:
        session: SQLAlchemy session
        training_run_id: Training run ID to delete

    Returns:
        True if deleted, False if not found
    """
    run = session.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()

    if run:
        # Delete model file if it exists
        if run.model_file_path:
            from pathlib import Path
            model_path = Path(run.model_file_path)
            if model_path.exists():
                try:
                    model_path.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete model file {model_path}: {e}")

        # Delete the run (cascade will handle related records)
        session.delete(run)
        session.flush()
        return True

    return False


# --- InferenceBacktest Operations ---

def create_inference_backtest(
    session: Session,
    training_run_id: int,
    prediction_method: str,
    ensemble_method: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    lookback_window: Optional[int] = None,
    inference_params: Optional[Dict[str, Any]] = None
) -> InferenceBacktest:
    """Create a new inference backtest record."""
    backtest = InferenceBacktest(
        training_run_id=training_run_id,
        prediction_method=prediction_method,
        ensemble_method=ensemble_method,
        confidence_threshold=confidence_threshold,
        lookback_window=lookback_window,
        inference_params=inference_params
    )
    session.add(backtest)
    session.flush()
    return backtest


def update_inference_backtest_results(
    session: Session,
    backtest_id: int,
    backtest_metrics: Dict[str, Any],
    regime_metrics: Dict[str, Dict[str, Any]],
    backtest_duration_seconds: float
) -> None:
    """Update inference backtest with results."""
    backtest = session.query(InferenceBacktest).filter(InferenceBacktest.id == backtest_id).first()
    if backtest:
        backtest.backtest_metrics = backtest_metrics
        backtest.regime_metrics = regime_metrics
        backtest.backtest_duration_seconds = backtest_duration_seconds
        backtest.completed_at = datetime.utcnow()
        session.flush()


def get_inference_backtests_by_run(session: Session, training_run_id: int) -> List[InferenceBacktest]:
    """Get all inference backtests for a training run."""
    return session.query(InferenceBacktest).filter(
        InferenceBacktest.training_run_id == training_run_id
    ).all()


# --- RegimeDefinition Operations ---

def get_all_active_regimes(session: Session) -> List[RegimeDefinition]:
    """Get all active regime definitions."""
    return session.query(RegimeDefinition).filter(RegimeDefinition.is_active == True).all()


def get_regime_by_name(session: Session, regime_name: str) -> Optional[RegimeDefinition]:
    """Get regime definition by name."""
    return session.query(RegimeDefinition).filter(RegimeDefinition.regime_name == regime_name).first()


def get_all_regimes(session: Session) -> List[RegimeDefinition]:
    """Get all regime definitions (active and inactive)."""
    return session.query(RegimeDefinition).order_by(RegimeDefinition.created_at.desc()).all()


def get_regime_by_id(session: Session, regime_id: int) -> Optional[RegimeDefinition]:
    """Get regime definition by ID."""
    return session.query(RegimeDefinition).filter(RegimeDefinition.id == regime_id).first()


def create_regime_definition(
    session: Session,
    regime_name: str,
    description: str,
    detection_rules: Dict[str, Any],
    is_active: bool = True
) -> RegimeDefinition:
    """Create a new regime definition."""
    regime = RegimeDefinition(
        regime_name=regime_name,
        description=description,
        detection_rules=detection_rules,
        is_active=is_active
    )
    session.add(regime)
    session.flush()
    return regime


def update_regime_definition(
    session: Session,
    regime_id: int,
    description: Optional[str] = None,
    detection_rules: Optional[Dict[str, Any]] = None,
    is_active: Optional[bool] = None
) -> Optional[RegimeDefinition]:
    """Update an existing regime definition."""
    regime = session.query(RegimeDefinition).filter(RegimeDefinition.id == regime_id).first()

    if regime:
        if description is not None:
            regime.description = description
        if detection_rules is not None:
            regime.detection_rules = detection_rules
        if is_active is not None:
            regime.is_active = is_active

        session.flush()

    return regime


def delete_regime_definition(session: Session, regime_id: int) -> bool:
    """
    Delete a regime definition.

    Note: This will cascade delete related RegimeBestModel records.

    Returns:
        True if deleted, False if not found
    """
    regime = session.query(RegimeDefinition).filter(RegimeDefinition.id == regime_id).first()

    if regime:
        session.delete(regime)
        session.flush()
        return True

    return False


# --- RegimeBestModel Operations ---

def get_best_model_for_regime(session: Session, regime_name: str) -> Optional[RegimeBestModel]:
    """Get current best model for a regime."""
    return session.query(RegimeBestModel).filter(RegimeBestModel.regime_name == regime_name).first()


def update_best_model_for_regime(
    session: Session,
    regime_name: str,
    training_run_id: int,
    inference_backtest_id: int,
    performance_score: float,
    secondary_metrics: Optional[Dict[str, Any]] = None
) -> RegimeBestModel:
    """Update or create best model for a regime."""
    best_model = session.query(RegimeBestModel).filter(RegimeBestModel.regime_name == regime_name).first()

    if best_model:
        # Update existing
        best_model.training_run_id = training_run_id
        best_model.inference_backtest_id = inference_backtest_id
        best_model.performance_score = performance_score
        best_model.secondary_metrics = secondary_metrics
        best_model.achieved_at = datetime.utcnow()
    else:
        # Create new
        best_model = RegimeBestModel(
            regime_name=regime_name,
            training_run_id=training_run_id,
            inference_backtest_id=inference_backtest_id,
            performance_score=performance_score,
            secondary_metrics=secondary_metrics
        )
        session.add(best_model)

    session.flush()
    return best_model


def get_all_regime_best_models(session: Session) -> List[RegimeBestModel]:
    """Get all current best models across all regimes."""
    return session.query(RegimeBestModel).all()


# --- TrainingQueue Operations ---

def create_training_queue(
    session: Session,
    config_grid: List[Dict[str, Any]],
    priority: int = 0,
    max_parallel: int = 1
) -> TrainingQueue:
    """Create a new training queue."""
    queue = TrainingQueue(
        config_grid=config_grid,
        total_configs=len(config_grid),
        priority=priority,
        max_parallel=max_parallel,
        status='pending'
    )
    session.add(queue)
    session.flush()
    return queue


def get_training_queue_by_id(session: Session, queue_id: int) -> Optional[TrainingQueue]:
    """Get training queue by ID."""
    return session.query(TrainingQueue).filter(TrainingQueue.id == queue_id).first()


def get_training_queue_by_uuid(session: Session, queue_uuid: str) -> Optional[TrainingQueue]:
    """Get training queue by UUID."""
    return session.query(TrainingQueue).filter(TrainingQueue.queue_uuid == queue_uuid).first()


def update_queue_status(
    session: Session,
    queue_id: int,
    status: str,
    started_at: Optional[datetime] = None,
    paused_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None
) -> None:
    """Update queue status and timestamps."""
    queue = session.query(TrainingQueue).filter(TrainingQueue.id == queue_id).first()
    if queue:
        queue.status = status
        if started_at:
            queue.started_at = started_at
        if paused_at:
            queue.paused_at = paused_at
        if completed_at:
            queue.completed_at = completed_at
        session.flush()


def update_queue_progress(
    session: Session,
    queue_id: int,
    current_index: int,
    completed_count: int,
    failed_count: int,
    skipped_count: int
) -> None:
    """Update queue progress counters."""
    queue = session.query(TrainingQueue).filter(TrainingQueue.id == queue_id).first()
    if queue:
        queue.current_index = current_index
        queue.completed_count = completed_count
        queue.failed_count = failed_count
        queue.skipped_count = skipped_count
        session.flush()


def get_pending_queues(session: Session) -> List[TrainingQueue]:
    """Get all pending queues ordered by priority."""
    return session.query(TrainingQueue).filter(
        TrainingQueue.status.in_(['pending', 'paused'])
    ).order_by(TrainingQueue.priority.desc(), TrainingQueue.created_at.asc()).all()


# ==================== Query Helpers ====================

def get_training_runs_summary(
    session: Session,
    symbol: Optional[str] = None,
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
) -> List[TrainingRun]:
    """Get training runs with optional filters."""
    query = session.query(TrainingRun)

    if symbol:
        query = query.filter(TrainingRun.symbol == symbol)
    if model_type:
        query = query.filter(TrainingRun.model_type == model_type)
    if status:
        query = query.filter(TrainingRun.status == status)

    return query.order_by(TrainingRun.created_at.desc()).limit(limit).all()


def get_storage_stats(session: Session) -> Dict[str, Any]:
    """Calculate storage statistics for model files."""
    kept_models = session.query(TrainingRun).filter(TrainingRun.is_model_kept == True).all()
    all_completed = session.query(TrainingRun).filter(TrainingRun.status == 'completed').all()

    kept_size = sum(m.model_file_size_bytes or 0 for m in kept_models)
    total_size = sum(m.model_file_size_bytes or 0 for m in all_completed)

    return {
        'kept_models_count': len(kept_models),
        'total_models_count': len(all_completed),
        'kept_size_bytes': kept_size,
        'total_size_bytes': total_size,
        'kept_size_mb': kept_size / (1024 * 1024),
        'total_size_mb': total_size / (1024 * 1024),
        'deletable_count': len(all_completed) - len(kept_models),
        'deletable_size_mb': (total_size - kept_size) / (1024 * 1024)
    }


def get_models_to_delete(session: Session) -> List[TrainingRun]:
    """Get all completed models that are not marked as kept."""
    return session.query(TrainingRun).filter(
        TrainingRun.status == 'completed',
        TrainingRun.is_model_kept == False,
        TrainingRun.model_file_path.isnot(None)
    ).all()
