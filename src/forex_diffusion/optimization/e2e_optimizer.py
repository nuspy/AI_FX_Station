"""E2E Optimizer Orchestrator - Main Entry Point"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable
import uuid
import json
import time
import numpy as np
from loguru import logger

from .parameter_space import ParameterSpace
from .bayesian_optimizer import BayesianOptimizer, BayesianConfig
from .objective_functions import ObjectiveCalculator
from .convergence_detector import ConvergenceDetector

@dataclass
class E2EOptimizerConfig:
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    optimization_method: str = 'bayesian'  # or 'genetic'
    regime_mode: str = 'global'  # or 'per_regime'
    n_trials: int = 100
    objectives_weights: Dict[str, float] = field(default_factory=lambda: {
        'sharpe': 0.5, 'max_dd': 0.3, 'pf': 0.15, 'cost': 0.05
    })
    constraints: Dict[str, float] = field(default_factory=lambda: {
        'max_drawdown_pct': 15.0, 'min_sharpe': 1.0
    })
    enable_sssd: bool = False
    enable_riskfolio: bool = True
    enable_patterns: bool = True
    enable_rl: bool = False
    enable_vix_filter: bool = True
    enable_sentiment_filter: bool = True
    enable_volume_filter: bool = True

class E2EOptimizer:
    def __init__(self, config: E2EOptimizerConfig, db_session=None):
        self.config = config
        self.db_session = db_session
        self.parameter_space = ParameterSpace()
        self.objective_calc = ObjectiveCalculator(weights=config.objectives_weights)
        self.convergence_detector = ConvergenceDetector(patience=20, min_delta=0.01)
        self.run_id: Optional[int] = None
        self.run_uuid: str = str(uuid.uuid4())
        self.best_trial: Optional[Dict] = None
        self.best_sharpe: float = -np.inf
    
    def run_optimization(self, backtest_func: Callable) -> Dict:
        """Main optimization loop"""
        logger.info(f"Starting E2E optimization: {self.config.symbol} {self.config.timeframe}")
        logger.info(f"Method: {self.config.optimization_method}, Trials: {self.config.n_trials}")
        
        start_time = time.time()
        
        # Create optimization run record in DB
        self.run_id = self._create_optimization_run()
        
        # Run optimization based on method
        if self.config.optimization_method == 'bayesian':
            result = self._run_bayesian(backtest_func)
        else:
            result = self._run_genetic(backtest_func)
        
        # Calculate duration
        total_duration = int(time.time() - start_time)
        
        # Update run record
        self._complete_optimization_run(total_duration, result)
        
        logger.info(f"Optimization complete! Best Sharpe: {self.best_sharpe:.3f}")
        
        return {
            'run_id': self.run_id,
            'run_uuid': self.run_uuid,
            'best_sharpe': self.best_sharpe,
            'best_trial': self.best_trial,
            'total_duration_sec': total_duration,
            'n_trials_completed': result.get('n_trials', 0)
        }
    
    def _run_bayesian(self, backtest_func: Callable) -> Dict:
        """Run Bayesian optimization"""
        config = BayesianConfig(
            sampler='tpe',
            n_startup_trials=min(10, self.config.n_trials // 10),
            direction=['maximize']  # Maximize combined score
        )
        
        optimizer = BayesianOptimizer(self.parameter_space, config)
        
        def objective(trial_num: int, params: Dict) -> float:
            return self._evaluate_trial(trial_num, params, backtest_func)
        
        result = optimizer.optimize(objective, self.config.n_trials, study_name=f"e2e_{self.run_uuid}")
        
        return result
    
    def _run_genetic(self, backtest_func: Callable) -> Dict:
        """Run Genetic Algorithm (NSGA-II) - placeholder"""
        logger.warning("Genetic Algorithm not yet implemented, using Bayesian instead")
        return self._run_bayesian(backtest_func)
    
    def _evaluate_trial(self, trial_num: int, params: Dict, backtest_func: Callable) -> float:
        """Evaluate single trial"""
        logger.info(f"Trial {trial_num + 1}/{self.config.n_trials} starting...")
        
        trial_start = time.time()
        
        try:
            # Run backtest with parameters
            backtest_result = backtest_func(params)
            
            # Calculate objectives
            objectives = self.objective_calc.calculate(backtest_result)
            
            # Store trial parameters to DB
            self._store_trial_parameters(trial_num, params)
            
            # Store trial results to DB
            self._store_trial_results(trial_num, backtest_result, objectives)
            
            # Update best
            if objectives.sharpe_ratio > self.best_sharpe:
                self.best_sharpe = objectives.sharpe_ratio
                self.best_trial = {
                    'trial_number': trial_num,
                    'params': params,
                    'sharpe': objectives.sharpe_ratio,
                    'max_dd': objectives.max_drawdown_pct,
                    'pf': objectives.profit_factor
                }
                logger.info(f"✨ New best! Sharpe: {self.best_sharpe:.3f}")
            
            # Check convergence
            if self.convergence_detector.update(objectives.combined_score):
                logger.info(f"Convergence detected at trial {trial_num}")
            
            trial_time = time.time() - trial_start
            logger.info(f"Trial {trial_num + 1} complete in {trial_time:.1f}s | Sharpe: {objectives.sharpe_ratio:.3f}")
            
            return objectives.combined_score
            
        except Exception as e:
            logger.error(f"Trial {trial_num} failed: {e}")
            return -1000.0  # Penalty for failed trials
    
    def _create_optimization_run(self) -> int:
        """Create optimization run record in DB"""
        if not self.db_session:
            return 0
        
        from ..database.e2e_optimization_models import E2EOptimizationRun
        
        run = E2EOptimizationRun(
            run_uuid=self.run_uuid,
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            optimization_method=self.config.optimization_method,
            regime_mode=self.config.regime_mode,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            n_trials=self.config.n_trials,
            status='running',
            optimize_sssd=self.config.enable_sssd,
            optimize_riskfolio=self.config.enable_riskfolio,
            optimize_patterns=self.config.enable_patterns,
            optimize_rl=self.config.enable_rl,
            optimize_vix_filter=self.config.enable_vix_filter,
            optimize_sentiment_filter=self.config.enable_sentiment_filter,
            optimize_volume_filter=self.config.enable_volume_filter,
            objectives_config=json.dumps(self.config.objectives_weights),
            constraints_config=json.dumps(self.config.constraints),
            started_at=datetime.utcnow()
        )
        
        self.db_session.add(run)
        self.db_session.commit()
        
        logger.info(f"Created optimization run: ID={run.id}, UUID={run.run_uuid}")
        
        return run.id
    
    def _store_trial_parameters(self, trial_num: int, params: Dict):
        """Store trial parameters to DB"""
        if not self.db_session or not self.run_id:
            return
        
        from ..database.e2e_optimization_models import E2EOptimizationParameter
        
        for param_name, param_value in params.items():
            param_def = self.parameter_space.parameters.get(param_name)
            if not param_def:
                continue
            
            param_record = E2EOptimizationParameter(
                run_id=self.run_id,
                trial_number=trial_num,
                regime='global',
                parameter_group=param_def.group,
                parameter_name=param_name,
                parameter_value=json.dumps(param_value) if isinstance(param_value, (list, dict)) else str(param_value),
                parameter_type=param_def.type
            )
            
            self.db_session.add(param_record)
        
        self.db_session.commit()
    
    def _store_trial_results(self, trial_num: int, backtest_result: Dict, objectives):
        """Store trial results to DB"""
        if not self.db_session or not self.run_id:
            return
        
        from ..database.e2e_optimization_models import E2EOptimizationResult
        
        result = E2EOptimizationResult(
            run_id=self.run_id,
            trial_number=trial_num,
            regime='global',
            total_return=backtest_result.get('total_return', 0.0),
            total_return_pct=backtest_result.get('total_return_pct', 0.0),
            sharpe_ratio=objectives.sharpe_ratio,
            sortino_ratio=backtest_result.get('sortino_ratio', 0.0),
            calmar_ratio=backtest_result.get('calmar_ratio', 0.0),
            max_drawdown=backtest_result.get('max_drawdown', 0.0),
            max_drawdown_pct=objectives.max_drawdown_pct,
            win_rate=backtest_result.get('win_rate', 0.0),
            profit_factor=objectives.profit_factor,
            total_trades=backtest_result.get('total_trades', 0),
            winning_trades=backtest_result.get('winning_trades', 0),
            losing_trades=backtest_result.get('losing_trades', 0),
            avg_win=backtest_result.get('avg_win', 0.0),
            avg_loss=backtest_result.get('avg_loss', 0.0),
            expectancy=backtest_result.get('expectancy', 0.0),
            total_costs=backtest_result.get('total_costs', 0.0),
            avg_cost_per_trade=backtest_result.get('avg_cost_per_trade', 0.0),
            avg_holding_time_hrs=backtest_result.get('avg_holding_time_hrs', 0.0),
            objective_value=objectives.combined_score
        )
        
        self.db_session.add(result)
        self.db_session.commit()
    
    def _complete_optimization_run(self, total_duration: int, result: Dict):
        """Update optimization run with final results"""
        if not self.db_session or not self.run_id:
            return
        
        from ..database.e2e_optimization_models import E2EOptimizationRun
        
        run = self.db_session.query(E2EOptimizationRun).filter_by(id=self.run_id).first()
        if run:
            run.status = 'completed'
            run.best_sharpe = self.best_sharpe
            run.best_trial_number = self.best_trial['trial_number'] if self.best_trial else None
            run.total_duration_sec = total_duration
            run.avg_trial_duration_sec = total_duration / result.get('n_trials', 1)
            run.completed_at = datetime.utcnow()
            
            self.db_session.commit()
            
            logger.info(f"Optimization run completed: ID={self.run_id}")
