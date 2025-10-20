"""
Database integration for optimization results.

Stores optimization runs, parameter configurations, and results for analysis.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from loguru import logger

Base = declarative_base()


class OptimizationRun(Base):
    """Table for optimization run metadata."""
    __tablename__ = 'optimization_runs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    optimizer_type = Column(String(50), nullable=False)  # 'grid', 'genetic', 'hybrid'
    config = Column(JSON)  # Optimizer configuration
    status = Column(String(20), default='running')  # 'running', 'completed', 'failed'
    n_evaluations = Column(Integer)
    duration_seconds = Column(Float)


class OptimizationResult(Base):
    """Table for individual optimization results."""
    __tablename__ = 'optimization_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, nullable=False, index=True)
    evaluation_number = Column(Integer)
    parameters = Column(JSON, nullable=False)  # Parameter configuration
    return_pct = Column(Float)
    sharpe_ratio = Column(Float, index=True)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    num_trades = Column(Integer)
    profit_factor = Column(Float)
    expectancy = Column(Float)
    avg_trade_duration = Column(Float)
    equity_final = Column(Float)
    is_pareto_optimal = Column(Integer, default=0)  # Boolean: 1 if on Pareto front


class ParetoFront(Base):
    """Table for Pareto-optimal solutions."""
    __tablename__ = 'pareto_fronts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, nullable=False, index=True)
    result_id = Column(Integer, nullable=False)
    dominance_rank = Column(Integer)  # Rank in Pareto front (1 = best)
    created_at = Column(DateTime, default=datetime.utcnow)


class OptimizationDB:
    """Database manager for optimization results."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize optimization database.

        Args:
            db_path: Path to SQLite database (defaults to data/optimization.db)
        """
        if db_path is None:
            db_path = Path("data/optimization.db")

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.Session = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)

        logger.info(f"Initialized optimization database: {db_path}")

    def start_run(
        self,
        symbol: str,
        timeframe: str,
        strategy_name: str,
        optimizer_type: str,
        config: Dict[str, Any]
    ) -> int:
        """
        Start a new optimization run.

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            strategy_name: Strategy class name
            optimizer_type: Optimizer type ('grid', 'genetic', 'hybrid')
            config: Optimizer configuration

        Returns:
            Run ID
        """
        session = self.Session()
        try:
            run = OptimizationRun(
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy_name,
                optimizer_type=optimizer_type,
                config=config,
                status='running'
            )
            session.add(run)
            session.commit()

            run_id = run.id
            logger.info(f"Started optimization run {run_id}: {symbol} {timeframe} ({optimizer_type})")
            return run_id

        finally:
            session.close()

    def add_result(
        self,
        run_id: int,
        parameters: Dict[str, float],
        metrics: Dict[str, float],
        evaluation_number: Optional[int] = None
    ) -> int:
        """
        Add optimization result.

        Args:
            run_id: Optimization run ID
            parameters: Parameter configuration
            metrics: Performance metrics
            evaluation_number: Sequential evaluation number

        Returns:
            Result ID
        """
        session = self.Session()
        try:
            result = OptimizationResult(
                run_id=run_id,
                evaluation_number=evaluation_number,
                parameters=parameters,
                return_pct=metrics.get('return', 0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                max_drawdown=metrics.get('max_drawdown', 100),
                win_rate=metrics.get('win_rate', 0),
                num_trades=metrics.get('num_trades', 0),
                profit_factor=metrics.get('profit_factor', 0),
                expectancy=metrics.get('expectancy', 0),
                avg_trade_duration=metrics.get('avg_trade_duration', 0),
                equity_final=metrics.get('equity_final', 10000)
            )
            session.add(result)
            session.commit()

            return result.id

        finally:
            session.close()

    def complete_run(
        self,
        run_id: int,
        n_evaluations: int,
        duration_seconds: float,
        status: str = 'completed'
    ):
        """
        Mark run as complete.

        Args:
            run_id: Run ID
            n_evaluations: Total number of evaluations
            duration_seconds: Total duration
            status: Final status ('completed' or 'failed')
        """
        session = self.Session()
        try:
            run = session.query(OptimizationRun).filter_by(id=run_id).first()
            if run:
                run.status = status
                run.n_evaluations = n_evaluations
                run.duration_seconds = duration_seconds
                session.commit()

                logger.info(f"Completed run {run_id}: {n_evaluations} evals in {duration_seconds:.1f}s")

        finally:
            session.close()

    def get_run_results(self, run_id: int) -> pd.DataFrame:
        """
        Get all results for a run.

        Args:
            run_id: Run ID

        Returns:
            DataFrame with results
        """
        session = self.Session()
        try:
            results = session.query(OptimizationResult).filter_by(run_id=run_id).all()

            data = []
            for result in results:
                row = result.parameters.copy()
                row.update({
                    'result_id': result.id,
                    'return': result.return_pct,
                    'sharpe': result.sharpe_ratio,
                    'drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'num_trades': result.num_trades,
                    'is_pareto': result.is_pareto_optimal
                })
                data.append(row)

            return pd.DataFrame(data)

        finally:
            session.close()

    def get_best_result(self, run_id: int, criterion: str = 'sharpe') -> Dict[str, Any]:
        """
        Get best result from a run.

        Args:
            run_id: Run ID
            criterion: Criterion ('sharpe', 'return', 'drawdown')

        Returns:
            Best result dictionary
        """
        session = self.Session()
        try:
            if criterion == 'sharpe':
                result = session.query(OptimizationResult).filter_by(run_id=run_id).order_by(
                    OptimizationResult.sharpe_ratio.desc()
                ).first()
            elif criterion == 'return':
                result = session.query(OptimizationResult).filter_by(run_id=run_id).order_by(
                    OptimizationResult.return_pct.desc()
                ).first()
            elif criterion == 'drawdown':
                result = session.query(OptimizationResult).filter_by(run_id=run_id).order_by(
                    OptimizationResult.max_drawdown.asc()
                ).first()
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

            if result:
                return {
                    'result_id': result.id,
                    'params': result.parameters,
                    'return': result.return_pct,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'num_trades': result.num_trades
                }
            else:
                return {}

        finally:
            session.close()

    def store_pareto_front(self, run_id: int, pareto_result_ids: List[int]):
        """
        Store Pareto front for a run.

        Args:
            run_id: Run ID
            pareto_result_ids: List of result IDs on Pareto front (ranked)
        """
        session = self.Session()
        try:
            # Mark results as Pareto-optimal
            for result_id in pareto_result_ids:
                result = session.query(OptimizationResult).filter_by(id=result_id).first()
                if result:
                    result.is_pareto_optimal = 1

            # Store Pareto front entries
            for rank, result_id in enumerate(pareto_result_ids, 1):
                pf = ParetoFront(
                    run_id=run_id,
                    result_id=result_id,
                    dominance_rank=rank
                )
                session.add(pf)

            session.commit()
            logger.info(f"Stored Pareto front with {len(pareto_result_ids)} solutions for run {run_id}")

        finally:
            session.close()

    def get_pareto_front(self, run_id: int) -> pd.DataFrame:
        """
        Get Pareto front for a run.

        Args:
            run_id: Run ID

        Returns:
            DataFrame with Pareto-optimal solutions
        """
        session = self.Session()
        try:
            pareto_entries = session.query(ParetoFront).filter_by(run_id=run_id).order_by(
                ParetoFront.dominance_rank
            ).all()

            data = []
            for entry in pareto_entries:
                result = session.query(OptimizationResult).filter_by(id=entry.result_id).first()
                if result:
                    row = result.parameters.copy()
                    row.update({
                        'result_id': result.id,
                        'rank': entry.dominance_rank,
                        'return': result.return_pct,
                        'sharpe': result.sharpe_ratio,
                        'drawdown': result.max_drawdown,
                        'win_rate': result.win_rate,
                        'num_trades': result.num_trades
                    })
                    data.append(row)

            return pd.DataFrame(data)

        finally:
            session.close()

    def list_runs(
        self,
        symbol: Optional[str] = None,
        optimizer_type: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        List optimization runs.

        Args:
            symbol: Optional symbol filter
            optimizer_type: Optional optimizer type filter
            limit: Maximum number of runs to return

        Returns:
            DataFrame with run metadata
        """
        session = self.Session()
        try:
            query = session.query(OptimizationRun)

            if symbol:
                query = query.filter_by(symbol=symbol)
            if optimizer_type:
                query = query.filter_by(optimizer_type=optimizer_type)

            runs = query.order_by(OptimizationRun.created_at.desc()).limit(limit).all()

            data = []
            for run in runs:
                data.append({
                    'run_id': run.id,
                    'created_at': run.created_at,
                    'symbol': run.symbol,
                    'timeframe': run.timeframe,
                    'strategy': run.strategy_name,
                    'optimizer': run.optimizer_type,
                    'status': run.status,
                    'n_evals': run.n_evaluations,
                    'duration_s': run.duration_seconds
                })

            return pd.DataFrame(data)

        finally:
            session.close()
