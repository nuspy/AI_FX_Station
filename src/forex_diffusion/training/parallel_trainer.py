"""
Parallel Model Training Module

Trains multiple models in parallel using multiprocessing.
Implements OPT-001 - reduces training time from 6h to 45min (8x speedup).
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .train_sklearn import train_single_model  # Existing function


@dataclass
class TrainingJob:
    """Single model training job specification"""
    symbol: str
    timeframe: str
    horizon: int
    algorithm: str
    job_id: str


@dataclass
class TrainingResult:
    """Result from training job"""
    job: TrainingJob
    success: bool
    model_path: Optional[str] = None
    metrics: Optional[Dict] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


class ParallelTrainer:
    """
    Train multiple models in parallel for significant speedup.
    
    Features:
    - Parallel training across CPU cores
    - Progress tracking
    - Error handling per job
    - Result aggregation
    - Automatic retry on failure
    
    Example speedup:
        Sequential: 36 models × 10 min = 360 min (6 hours)
        Parallel (8 cores): 36 models / 8 = 45 minutes
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        retry_failed: bool = True,
        artifacts_dir: str = "artifacts"
    ):
        """
        Initialize parallel trainer.
        
        Args:
            max_workers: Number of parallel processes (default: CPU count - 1)
            retry_failed: Retry failed jobs once
            artifacts_dir: Directory for model artifacts
        """
        if max_workers is None:
            max_workers = max(1, mp.cpu_count() - 1)
        
        self.max_workers = max_workers
        self.retry_failed = retry_failed
        self.artifacts_dir = Path(artifacts_dir)
        
        logger.info(
            f"ParallelTrainer initialized with {max_workers} workers "
            f"(CPU count: {mp.cpu_count()})"
        )
    
    def train_all_models(
        self,
        symbols: List[str],
        timeframes: List[str],
        horizons: List[int],
        algorithms: List[str]
    ) -> List[TrainingResult]:
        """
        Train all model combinations in parallel.
        
        Args:
            symbols: List of symbols (e.g., ['EUR/USD', 'GBP/USD'])
            timeframes: List of timeframes (e.g., ['15m', '1h', '4h'])
            horizons: List of horizons (e.g., [1, 4, 8, 24])
            algorithms: List of algorithms (e.g., ['ridge', 'lasso', 'rf'])
            
        Returns:
            List of TrainingResult for each job
        """
        # Generate all combinations
        jobs = self._generate_jobs(symbols, timeframes, horizons, algorithms)
        
        total_jobs = len(jobs)
        logger.info(f"Training {total_jobs} models in parallel...")
        
        start_time = datetime.now()
        
        # Execute jobs in parallel
        results = self._execute_parallel(jobs)
        
        # Calculate statistics
        duration = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in results if r.success)
        failed = total_jobs - successful
        
        logger.info(
            f"Parallel training complete: {successful}/{total_jobs} successful, "
            f"{failed} failed, duration: {duration:.1f}s "
            f"({duration/60:.1f} minutes)"
        )
        
        # Retry failed jobs if enabled
        if self.retry_failed and failed > 0:
            logger.warning(f"Retrying {failed} failed jobs...")
            results = self._retry_failed_jobs(results)
        
        return results
    
    def _generate_jobs(
        self,
        symbols: List[str],
        timeframes: List[str],
        horizons: List[int],
        algorithms: List[str]
    ) -> List[TrainingJob]:
        """Generate all training job combinations"""
        jobs = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                for horizon in horizons:
                    for algorithm in algorithms:
                        job_id = f"{symbol}_{timeframe}_h{horizon}_{algorithm}"
                        job_id = job_id.replace('/', '_')  # Clean job ID
                        
                        jobs.append(TrainingJob(
                            symbol=symbol,
                            timeframe=timeframe,
                            horizon=horizon,
                            algorithm=algorithm,
                            job_id=job_id
                        ))
        
        return jobs
    
    def _execute_parallel(self, jobs: List[TrainingJob]) -> List[TrainingResult]:
        """Execute jobs in parallel"""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._train_single, job): job
                for job in jobs
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    completed += 1
                    status = "✓" if result.success else "✗"
                    logger.info(
                        f"[{completed}/{len(jobs)}] {status} {result.job.job_id} "
                        f"({result.duration_seconds:.1f}s)"
                    )
                    
                except Exception as e:
                    logger.error(f"Job {job.job_id} raised exception: {e}")
                    results.append(TrainingResult(
                        job=job,
                        success=False,
                        error=str(e)
                    ))
                    completed += 1
        
        return results
    
    def _train_single(self, job: TrainingJob) -> TrainingResult:
        """
        Train a single model (executed in subprocess).
        
        This method runs in a separate process.
        """
        start_time = datetime.now()
        
        try:
            # Call existing train_sklearn function
            from .train_sklearn import main as train_main
            
            # Prepare arguments
            args = type('Args', (), {
                'symbol': job.symbol,
                'timeframe': job.timeframe,
                'horizon': job.horizon,
                'algorithm': job.algorithm,
                'artifacts_dir': str(self.artifacts_dir),
                'use_gpu': False,  # Each process can't share GPU
                'verbose': False
            })()
            
            # Train model
            model_path, metrics = train_main(args)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TrainingResult(
                job=job,
                success=True,
                model_path=str(model_path),
                metrics=metrics,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Training failed for {job.job_id}: {e}")
            
            return TrainingResult(
                job=job,
                success=False,
                error=str(e),
                duration_seconds=duration
            )
    
    def _retry_failed_jobs(
        self,
        results: List[TrainingResult]
    ) -> List[TrainingResult]:
        """Retry failed jobs once"""
        failed_jobs = [r.job for r in results if not r.success]
        
        if not failed_jobs:
            return results
        
        logger.info(f"Retrying {len(failed_jobs)} failed jobs...")
        
        # Retry failed jobs
        retry_results = self._execute_parallel(failed_jobs)
        
        # Replace failed results with retry results
        result_dict = {r.job.job_id: r for r in results}
        for retry_result in retry_results:
            result_dict[retry_result.job.job_id] = retry_result
        
        final_results = list(result_dict.values())
        
        # Report retry statistics
        retry_successful = sum(1 for r in retry_results if r.success)
        logger.info(f"Retry complete: {retry_successful}/{len(failed_jobs)} recovered")
        
        return final_results
    
    def generate_summary_report(self, results: List[TrainingResult]) -> Dict:
        """Generate summary report from training results"""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        
        # Calculate statistics
        if successful > 0:
            successful_results = [r for r in results if r.success]
            avg_duration = np.mean([r.duration_seconds for r in successful_results])
            total_duration = sum(r.duration_seconds for r in results)
        else:
            avg_duration = 0
            total_duration = 0
        
        # Group by algorithm
        by_algorithm = {}
        for result in results:
            algo = result.job.algorithm
            if algo not in by_algorithm:
                by_algorithm[algo] = {'total': 0, 'successful': 0}
            by_algorithm[algo]['total'] += 1
            if result.success:
                by_algorithm[algo]['successful'] += 1
        
        return {
            'total_jobs': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'avg_duration_seconds': avg_duration,
            'total_duration_seconds': total_duration,
            'by_algorithm': by_algorithm,
            'failed_jobs': [
                {
                    'job_id': r.job.job_id,
                    'error': r.error
                }
                for r in results if not r.success
            ]
        }


def train_all_parallel(
    symbols: List[str] = ['EUR/USD', 'GBP/USD', 'USD/JPY'],
    timeframes: List[str] = ['15m', '1h', '4h'],
    horizons: List[int] = [1, 4, 8, 24],
    algorithms: List[str] = ['ridge', 'lasso', 'rf'],
    max_workers: Optional[int] = None,
    artifacts_dir: str = "artifacts"
) -> List[TrainingResult]:
    """
    Convenience function to train all models in parallel.
    
    Example:
        results = train_all_parallel(
            symbols=['EUR/USD', 'GBP/USD'],
            timeframes=['15m', '1h'],
            horizons=[4, 8],
            algorithms=['ridge']
        )
        
        # 2 × 2 × 2 × 1 = 8 models
        # Sequential: 8 × 10min = 80 minutes
        # Parallel (4 cores): 8 / 4 = 20 minutes (4x speedup)
    """
    trainer = ParallelTrainer(
        max_workers=max_workers,
        artifacts_dir=artifacts_dir
    )
    
    results = trainer.train_all_models(
        symbols=symbols,
        timeframes=timeframes,
        horizons=horizons,
        algorithms=algorithms
    )
    
    # Print summary
    summary = trainer.generate_summary_report(results)
    
    logger.info("=" * 60)
    logger.info("PARALLEL TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total jobs: {summary['total_jobs']}")
    logger.info(f"Successful: {summary['successful']} ({summary['success_rate']:.1%})")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Avg duration: {summary['avg_duration_seconds']:.1f}s per model")
    logger.info(f"Total duration: {summary['total_duration_seconds']/60:.1f} minutes")
    logger.info("=" * 60)
    
    return results
