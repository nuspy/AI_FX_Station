"""E2E Backend Bridge - UI ↔ Backend Connector"""
from __future__ import annotations
from typing import Dict, Optional
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QThread
from loguru import logger

from ...optimization import E2EOptimizer, E2EOptimizerConfig
from ...backtest.e2e_backtest_wrapper import E2EBacktestWrapper

class OptimizationWorker(QThread):
    """Background worker for optimization"""
    
    progress = Signal(int, int, float)  # trial, total, best_sharpe
    finished = Signal(int, dict)  # run_id, results
    error = Signal(str)  # error_message
    
    def __init__(self, config: Dict, db_session=None):
        super().__init__()
        self.config = config
        self.db_session = db_session
        self.should_stop = False
    
    def run(self):
        """Main optimization loop"""
        try:
            logger.info("Optimization worker started")
            
            # Create E2E optimizer config
            optimizer_config = E2EOptimizerConfig(
                symbol=self.config['symbol'],
                timeframe=self.config['timeframe'],
                start_date=self.config['start_date'],
                end_date=self.config['end_date'],
                optimization_method=self.config['method'],
                n_trials=self.config['n_trials'],
                enable_sssd=self.config.get('enable_sssd', False),
                enable_riskfolio=self.config.get('enable_riskfolio', True),
                enable_patterns=self.config.get('enable_patterns', True),
                enable_rl=self.config.get('enable_rl', False),
                enable_vix_filter=self.config.get('enable_vix_filter', True),
                enable_sentiment_filter=self.config.get('enable_sentiment_filter', True),
                enable_volume_filter=self.config.get('enable_volume_filter', True)
            )
            
            # Create optimizer
            optimizer = E2EOptimizer(optimizer_config, db_session=self.db_session)
            
            # Create backtest wrapper
            backtest_wrapper = E2EBacktestWrapper(db_session=self.db_session)
            
            # Mock data for now (in production, load from database)
            import pandas as pd
            import numpy as np
            dates = pd.date_range(self.config['start_date'], self.config['end_date'], freq='5min')
            mock_data = pd.DataFrame({
                'open': np.random.randn(len(dates)).cumsum() + 1.1,
                'high': np.random.randn(len(dates)).cumsum() + 1.11,
                'low': np.random.randn(len(dates)).cumsum() + 1.09,
                'close': np.random.randn(len(dates)).cumsum() + 1.1,
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            
            # Define backtest function
            def backtest_func(params):
                if self.should_stop:
                    raise StopIteration("Optimization stopped by user")
                return backtest_wrapper.run_backtest(mock_data, params)
            
            # Run optimization
            result = optimizer.run_optimization(backtest_func)
            
            # Emit finished signal
            self.finished.emit(result['run_id'], result)
            
            logger.info("Optimization worker completed")
            
        except Exception as e:
            logger.error(f"Optimization worker error: {e}")
            self.error.emit(str(e))
    
    def stop(self):
        """Stop optimization gracefully"""
        self.should_stop = True
        logger.info("Optimization stop requested")


class E2EBackendBridge(QObject):
    """Bridge between UI and E2E Optimizer backend"""
    
    optimization_progress = Signal(int, int, float)  # trial, total, best_sharpe
    optimization_completed = Signal(int, dict)  # run_id, results
    optimization_failed = Signal(int, str)  # run_id, error
    deployment_activated = Signal(int)  # deployment_id
    
    def __init__(self, db_session=None):
        super().__init__()
        self.db_session = db_session
        self.worker: Optional[OptimizationWorker] = None
    
    def start_optimization(self, config: Dict):
        """Start optimization in background thread"""
        logger.info(f"Starting optimization: {config['symbol']} {config['timeframe']}")
        
        # Create worker
        self.worker = OptimizationWorker(config, db_session=self.db_session)
        
        # Connect signals
        self.worker.progress.connect(self.optimization_progress.emit)
        self.worker.finished.connect(self.optimization_completed.emit)
        self.worker.error.connect(lambda err: self.optimization_failed.emit(0, err))
        
        # Start worker
        self.worker.start()
    
    def stop_optimization(self, run_id: int):
        """Stop running optimization"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            logger.info(f"Optimization {run_id} stopped")
    
    def get_optimization_status(self, run_id: int) -> Dict:
        """Get status of optimization run"""
        if not self.db_session:
            return {'status': 'unknown'}
        
        from ...database.e2e_optimization_models import E2EOptimizationRun
        
        run = self.db_session.query(E2EOptimizationRun).filter_by(id=run_id).first()
        if run:
            return run.to_dict()
        else:
            return {'status': 'not_found'}
    
    def get_run_history(self, limit: int = 50) -> list:
        """Get optimization run history"""
        if not self.db_session:
            return []
        
        from ...database.e2e_optimization_models import E2EOptimizationRun
        
        runs = self.db_session.query(E2EOptimizationRun).order_by(
            E2EOptimizationRun.created_at.desc()
        ).limit(limit).all()
        
        return [run.to_dict() for run in runs]
    
    def deploy_parameters(self, run_id: int, deployment_config: Dict):
        """Deploy optimized parameters to live trading"""
        logger.info(f"Deploying parameters from run {run_id}")
        
        if not self.db_session:
            logger.error("No database session")
            return
        
        from ...database.e2e_optimization_models import E2EDeploymentConfig
        
        # Create deployment record
        deployment = E2EDeploymentConfig(
            symbol=deployment_config['symbol'],
            timeframe=deployment_config['timeframe'],
            deployment_mode='global',
            deployed_by='user',
            is_active=True
        )
        
        self.db_session.add(deployment)
        self.db_session.commit()
        
        logger.info(f"Deployment created: ID={deployment.id}")
        self.deployment_activated.emit(deployment.id)
    
    def get_active_deployments(self) -> list:
        """Get active deployments"""
        if not self.db_session:
            return []
        
        from ...database.e2e_optimization_models import E2EDeploymentConfig
        
        deployments = self.db_session.query(E2EDeploymentConfig).filter_by(
            is_active=True
        ).all()
        
        return [d.to_dict() for d in deployments]
