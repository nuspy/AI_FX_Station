"""
Comprehensive logging and reporting system for optimization and backtesting.

Provides structured logging, performance metrics, progress tracking, and
detailed reports for optimization studies and trial execution.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from loguru import logger
import pandas as pd


@dataclass
class StudyProgress:
    """Progress tracking for optimization studies"""
    study_id: int
    pattern_key: str
    asset: str
    timeframe: str
    total_trials: int
    completed_trials: int
    pruned_trials: int
    failed_trials: int
    best_score: Optional[float]
    start_time: datetime
    estimated_completion: Optional[datetime]
    current_trial_id: Optional[int]


@dataclass
class TrialSummary:
    """Summary of trial execution"""
    trial_id: int
    task_id: str
    status: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    execution_time: float
    pruned_reason: Optional[str]
    error_message: Optional[str]


@dataclass
class StudyReport:
    """Comprehensive study report"""
    study_id: int
    study_name: str
    pattern_key: str
    direction: str
    asset: str
    timeframe: str
    regime_tag: Optional[str]

    # Execution summary
    status: str
    total_trials: int
    completed_trials: int
    pruned_trials: int
    failed_trials: int
    total_execution_time: float

    # Best results
    best_parameters: Dict[str, Any]
    best_metrics: Dict[str, float]
    pareto_frontier: List[Dict[str, Any]]

    # Performance analysis
    parameter_importance: Dict[str, float]
    convergence_analysis: Dict[str, Any]
    regime_performance: Dict[str, Dict[str, float]]

    # Resource usage
    avg_trial_time: float
    cpu_usage_stats: Dict[str, float]
    memory_usage_stats: Dict[str, float]


class LoggingReporter:
    """
    Comprehensive logging and reporting system for optimization studies.

    Provides structured logging, progress tracking, performance analysis,
    and detailed reports for optimization execution.
    """

    def __init__(self, base_log_dir: str = "logs/optimization"):
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)

        # Active studies tracking
        self.active_studies: Dict[int, StudyProgress] = {}
        self.study_logs: Dict[int, List[Dict]] = {}
        self.trial_metrics: Dict[int, List[TrialSummary]] = {}

        # Performance tracking
        self.performance_history: List[Dict] = []
        self.resource_snapshots: List[Dict] = []

        # Configure structured logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure structured logging for optimization"""

        # Study-level logs
        study_log_path = self.base_log_dir / "studies.log"
        logger.add(
            study_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | STUDY | {message}",
            filter=lambda record: "STUDY" in record["message"],
            rotation="10 MB",
            retention="30 days"
        )

        # Trial-level logs
        trial_log_path = self.base_log_dir / "trials.log"
        logger.add(
            trial_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | TRIAL | {message}",
            filter=lambda record: "TRIAL" in record["message"],
            rotation="50 MB",
            retention="30 days"
        )

        # Performance logs
        perf_log_path = self.base_log_dir / "performance.log"
        logger.add(
            perf_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | PERF | {message}",
            filter=lambda record: "PERF" in record["message"],
            rotation="10 MB",
            retention="7 days"
        )

    def start_study_tracking(self, study_id: int, study_config: Dict[str, Any]):
        """Start tracking a new optimization study"""

        progress = StudyProgress(
            study_id=study_id,
            pattern_key=study_config["pattern_key"],
            asset=study_config["asset"],
            timeframe=study_config["timeframe"],
            total_trials=study_config.get("max_trials", 1000),
            completed_trials=0,
            pruned_trials=0,
            failed_trials=0,
            best_score=None,
            start_time=datetime.now(),
            estimated_completion=None,
            current_trial_id=None
        )

        self.active_studies[study_id] = progress
        self.study_logs[study_id] = []
        self.trial_metrics[study_id] = []

        logger.info(f"STUDY | Started study {study_id} for {study_config['pattern_key']} "
                   f"on {study_config['asset']} {study_config['timeframe']}")

        # Log study configuration
        config_log = {
            "timestamp": datetime.now().isoformat(),
            "event": "study_started",
            "study_id": study_id,
            "config": study_config
        }
        self.study_logs[study_id].append(config_log)

    def log_trial_start(self, study_id: int, trial_id: int, task_id: str,
                       parameters: Dict[str, Any]):
        """Log trial start"""

        if study_id in self.active_studies:
            self.active_studies[study_id].current_trial_id = trial_id

        logger.info(f"TRIAL | Started trial {trial_id} (task: {task_id}) for study {study_id}")

        trial_log = {
            "timestamp": datetime.now().isoformat(),
            "event": "trial_started",
            "study_id": study_id,
            "trial_id": trial_id,
            "task_id": task_id,
            "parameters": parameters
        }

        if study_id in self.study_logs:
            self.study_logs[study_id].append(trial_log)

    def log_trial_completion(self, study_id: int, trial_id: int, task_id: str,
                           status: str, metrics: Optional[Dict[str, float]] = None,
                           execution_time: float = 0.0, error_message: Optional[str] = None,
                           pruned_reason: Optional[str] = None):
        """Log trial completion"""

        # Update study progress
        if study_id in self.active_studies:
            progress = self.active_studies[study_id]

            if status == "completed":
                progress.completed_trials += 1
                if metrics and "combined_score" in metrics:
                    if progress.best_score is None or metrics["combined_score"] > progress.best_score:
                        progress.best_score = metrics["combined_score"]
            elif status == "pruned":
                progress.pruned_trials += 1
            elif status == "failed":
                progress.failed_trials += 1

            # Update estimated completion
            total_completed = progress.completed_trials + progress.pruned_trials + progress.failed_trials
            if total_completed > 0:
                elapsed = datetime.now() - progress.start_time
                avg_time_per_trial = elapsed.total_seconds() / total_completed
                remaining_trials = progress.total_trials - total_completed
                estimated_remaining = timedelta(seconds=avg_time_per_trial * remaining_trials)
                progress.estimated_completion = datetime.now() + estimated_remaining

        # Create trial summary
        trial_summary = TrialSummary(
            trial_id=trial_id,
            task_id=task_id,
            status=status,
            parameters={},  # Will be filled from database if needed
            metrics=metrics or {},
            execution_time=execution_time,
            pruned_reason=pruned_reason,
            error_message=error_message
        )

        if study_id in self.trial_metrics:
            self.trial_metrics[study_id].append(trial_summary)

        # Log trial completion
        logger.info(f"TRIAL | Completed trial {trial_id} (task: {task_id}) for study {study_id} "
                   f"with status {status} in {execution_time:.1f}s")

        if error_message:
            logger.error(f"TRIAL | Trial {trial_id} failed: {error_message}")

        if pruned_reason:
            logger.info(f"TRIAL | Trial {trial_id} pruned: {pruned_reason}")

        trial_log = {
            "timestamp": datetime.now().isoformat(),
            "event": "trial_completed",
            "study_id": study_id,
            "trial_id": trial_id,
            "task_id": task_id,
            "status": status,
            "metrics": metrics,
            "execution_time": execution_time,
            "error_message": error_message,
            "pruned_reason": pruned_reason
        }

        if study_id in self.study_logs:
            self.study_logs[study_id].append(trial_log)

    def log_study_completion(self, study_id: int, status: str, final_metrics: Dict[str, Any]):
        """Log study completion"""

        if study_id in self.active_studies:
            progress = self.active_studies[study_id]
            total_time = (datetime.now() - progress.start_time).total_seconds()

            logger.info(f"STUDY | Completed study {study_id} with status {status} "
                       f"after {total_time:.1f}s ({progress.completed_trials} trials)")

            # Move to completed studies
            del self.active_studies[study_id]

        completion_log = {
            "timestamp": datetime.now().isoformat(),
            "event": "study_completed",
            "study_id": study_id,
            "status": status,
            "final_metrics": final_metrics
        }

        if study_id in self.study_logs:
            self.study_logs[study_id].append(completion_log)

    def log_resource_usage(self, cpu_percent: float, memory_percent: float,
                          active_workers: int, queue_size: int):
        """Log current resource usage"""

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "active_workers": active_workers,
            "queue_size": queue_size
        }

        self.resource_snapshots.append(snapshot)

        # Keep only last 1000 snapshots
        if len(self.resource_snapshots) > 1000:
            self.resource_snapshots = self.resource_snapshots[-1000:]

        logger.info(f"PERF | CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, "
                   f"Workers: {active_workers}, Queue: {queue_size}")

    def get_study_progress(self, study_id: int) -> Optional[StudyProgress]:
        """Get current progress for a study"""
        return self.active_studies.get(study_id)

    def get_all_active_studies(self) -> List[StudyProgress]:
        """Get progress for all active studies"""
        return list(self.active_studies.values())

    def generate_study_report(self, study_id: int, task_manager) -> StudyReport:
        """Generate comprehensive report for a completed study"""

        # Get study details from database
        study = task_manager.get_study(study_id)
        if not study:
            raise ValueError(f"Study {study_id} not found")

        # Get all trials for this study
        trials = task_manager.get_study_trials(study_id)
        completed_trials = [t for t in trials if t.status == "completed"]
        pruned_trials = [t for t in trials if t.status == "pruned"]
        failed_trials = [t for t in trials if t.status == "failed"]

        # Calculate execution time
        start_time = min(t.started_at for t in trials if t.started_at) if trials else datetime.now()
        end_time = max(t.completed_at for t in trials if t.completed_at) if trials else datetime.now()
        total_execution_time = (end_time - start_time).total_seconds()

        # Get best results
        best_trial = None
        best_metrics = {}
        if completed_trials:
            # Find trial with highest combined score
            best_trial = max(completed_trials,
                           key=lambda t: task_manager.get_trial_metrics(t.id).get("combined_score", 0))
            best_metrics = task_manager.get_trial_metrics(best_trial.id)

        # Analyze parameter importance (simplified)
        parameter_importance = self._analyze_parameter_importance(completed_trials, task_manager)

        # Generate convergence analysis
        convergence_analysis = self._analyze_convergence(completed_trials, task_manager)

        # Resource usage statistics
        trial_times = [t.execution_time_seconds for t in completed_trials if t.execution_time_seconds]
        avg_trial_time = sum(trial_times) / len(trial_times) if trial_times else 0.0

        return StudyReport(
            study_id=study_id,
            study_name=study.study_name,
            pattern_key=study.pattern_key,
            direction=study.direction,
            asset=study.asset,
            timeframe=study.timeframe,
            regime_tag=study.regime_tag,
            status=study.status,
            total_trials=len(trials),
            completed_trials=len(completed_trials),
            pruned_trials=len(pruned_trials),
            failed_trials=len(failed_trials),
            total_execution_time=total_execution_time,
            best_parameters=json.loads(best_trial.form_parameters) if best_trial else {},
            best_metrics=best_metrics,
            pareto_frontier=[],  # Would need to implement Pareto analysis
            parameter_importance=parameter_importance,
            convergence_analysis=convergence_analysis,
            regime_performance={},  # Would need regime-specific analysis
            avg_trial_time=avg_trial_time,
            cpu_usage_stats={},
            memory_usage_stats={}
        )

    def _analyze_parameter_importance(self, trials: List, task_manager) -> Dict[str, float]:
        """Analyze parameter importance using correlation with performance"""

        if len(trials) < 10:
            return {}

        # Collect parameter values and scores
        param_data = []
        scores = []

        for trial in trials:
            metrics = task_manager.get_trial_metrics(trial.id)
            if "combined_score" in metrics:
                params = json.loads(trial.form_parameters)
                param_data.append(params)
                scores.append(metrics["combined_score"])

        if len(param_data) < 10:
            return {}

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(param_data)
        df["score"] = scores

        # Calculate correlations
        importance = {}
        for col in df.columns:
            if col != "score" and df[col].dtype in ["float64", "int64"]:
                correlation = abs(df[col].corr(df["score"]))
                if not pd.isna(correlation):
                    importance[col] = correlation

        return importance

    def _analyze_convergence(self, trials: List, task_manager) -> Dict[str, Any]:
        """Analyze optimization convergence"""

        if len(trials) < 10:
            return {"converged": False, "reason": "insufficient_trials"}

        # Get scores over time
        scores = []
        for trial in sorted(trials, key=lambda t: t.trial_number):
            metrics = task_manager.get_trial_metrics(trial.id)
            if "combined_score" in metrics:
                scores.append(metrics["combined_score"])

        if len(scores) < 10:
            return {"converged": False, "reason": "insufficient_scores"}

        # Analyze improvement rate
        recent_window = min(50, len(scores) // 4)
        recent_scores = scores[-recent_window:]
        early_scores = scores[:recent_window]

        recent_avg = sum(recent_scores) / len(recent_scores)
        early_avg = sum(early_scores) / len(early_scores)
        improvement = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0

        # Check for plateau
        last_20_percent = scores[-max(10, len(scores) // 5):]
        score_variance = pd.Series(last_20_percent).var() if len(last_20_percent) > 1 else float('inf')

        converged = improvement < 0.05 and score_variance < 0.01

        return {
            "converged": converged,
            "improvement_rate": improvement,
            "recent_variance": float(score_variance),
            "best_score": max(scores),
            "best_trial_number": scores.index(max(scores)) + 1,
            "plateau_detected": score_variance < 0.01
        }

    def export_study_logs(self, study_id: int, output_path: str):
        """Export study logs to file"""

        if study_id not in self.study_logs:
            raise ValueError(f"No logs found for study {study_id}")

        logs = self.study_logs[study_id]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(logs, f, indent=2, default=str)

        logger.info(f"Exported logs for study {study_id} to {output_path}")

    def export_performance_report(self, output_path: str):
        """Export comprehensive performance report"""

        report = {
            "generated_at": datetime.now().isoformat(),
            "active_studies": len(self.active_studies),
            "total_studies_tracked": len(self.study_logs),
            "resource_snapshots": len(self.resource_snapshots),
            "active_study_details": [asdict(progress) for progress in self.active_studies.values()],
            "recent_resource_usage": self.resource_snapshots[-100:] if self.resource_snapshots else []
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Exported performance report to {output_path}")