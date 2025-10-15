"""
E2E Test Utilities for ForexGPT Testing Framework.

This module provides utility functions for:
- Database schema validation and management
- Data quality validation
- Performance monitoring
- Resource tracking
- Report generation
"""

from __future__ import annotations

import os
import sys
import time
import json
import shutil
import psutil
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forex_diffusion.training_pipeline.database_models import (
    Base, TrainingRun, InferenceBacktest, RegimeDefinition,
    RegimeBestModel, TrainingQueue, OptimizedParameters,
    RiskProfile, AdvancedMetrics
)


@dataclass
class E2ETestConfig:
    """Configuration for E2E test execution."""
    # Resource limits
    max_data_months: int = 3
    max_models: int = 3
    max_backtests: int = 10
    max_pattern_optimization_cycles: int = 5
    max_trading_system_optimization_cycles: int = 10
    target_positions: int = 5
    trading_timeout_minutes: int = 15

    # Database paths
    db_path: str = r"D:\Projects\ForexGPT\data\market.db"
    data_dir: str = r"D:\Projects\ForexGPT\data"

    # Test output paths
    log_file: Optional[str] = None
    report_html: Optional[str] = None
    metrics_json: Optional[str] = None
    error_file: Optional[str] = None

    # Test execution
    enable_concurrency: bool = False
    test_start_time: Optional[datetime] = None

    def __post_init__(self):
        """Initialize paths with timestamp."""
        if self.test_start_time is None:
            self.test_start_time = datetime.now()

        timestamp = self.test_start_time.strftime("%Y%m%d_%H%M%S")
        base_dir = Path(self.data_dir) / f"e2e_test_results_{timestamp}"
        base_dir.mkdir(parents=True, exist_ok=True)

        logs_dir = base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        reports_dir = base_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        self.log_file = str(logs_dir / "main_log.log")
        self.error_file = str(logs_dir / "error_log.log")
        self.report_html = str(reports_dir / "test_summary.html")
        self.metrics_json = str(reports_dir / "metrics.json")


class DatabaseValidator:
    """Validates database schema and manages database operations."""

    # Required tables from spec
    REQUIRED_TABLES = [
        'candles',
        'market_data_ticks',
        'training_runs',
        'inference_backtests',
        'regime_definitions',
        'regime_best_models',
        'training_queue',
        'optimized_parameters',
        'risk_profiles',
        'advanced_metrics',
        'pattern_defs',
        'pattern_benchmarks',
        'pattern_events',
        'bt_job',
        'bt_config',
        'bt_result',
        'bt_trace',
    ]

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Connect to database."""
        self.conn = sqlite3.connect(self.db_path)

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def validate_schema(self) -> Tuple[bool, List[str]]:
        """
        Validate that all required tables exist.

        Returns:
            Tuple of (success, missing_tables)
        """
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}

        missing_tables = [
            table for table in self.REQUIRED_TABLES
            if table not in existing_tables
        ]

        return len(missing_tables) == 0, missing_tables

    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get column information for a table."""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        return [
            {
                'cid': col[0],
                'name': col[1],
                'type': col[2],
                'notnull': bool(col[3]),
                'default': col[4],
                'pk': bool(col[5])
            }
            for col in columns
        ]

    def clear_table(self, table_name: str) -> int:
        """
        Clear all rows from a table.

        Returns:
            Number of rows deleted
        """
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

        cursor.execute(f"DELETE FROM {table_name}")
        self.conn.commit()

        return row_count

    def clear_all_tables(self) -> Dict[str, int]:
        """
        Clear all tables in FK-safe order.

        Returns:
            Dict mapping table_name to rows_deleted
        """
        # Order respects FK constraints
        clear_order = [
            'inference_backtests',
            'regime_best_models',
            'training_runs',
            'training_queue',
            'optimized_parameters',
            'advanced_metrics',
            'market_data_ticks',
            'candles',
            'pattern_events',
            'pattern_benchmarks',
            'pattern_defs',
            'bt_trace',
            'bt_result',
            'bt_config',
            'bt_job',
        ]

        results = {}
        for table in clear_order:
            try:
                count = self.clear_table(table)
                results[table] = count
                logger.info(f"Cleared {count} rows from {table}")
            except Exception as e:
                logger.warning(f"Could not clear {table}: {e}")
                results[table] = -1

        return results

    def get_row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]


class DataQualityValidator:
    """Validates data quality for OHLC data."""

    @staticmethod
    def validate_ohlc_consistency(df: pd.DataFrame, tolerance: float = 0.0001) -> Dict[str, Any]:
        """
        Validate OHLC price consistency.

        Returns:
            Dict with validation results
        """
        results = {
            'total_candles': len(df),
            'high_vs_open': 0,
            'high_vs_close': 0,
            'low_vs_open': 0,
            'low_vs_close': 0,
            'high_vs_low': 0,
            'invalid_count': 0,
            'consistency_rate': 0.0
        }

        if len(df) == 0:
            return results

        # Check: High >= Open, High >= Close
        results['high_vs_open'] = ((df['high'] + tolerance) >= df['open']).sum()
        results['high_vs_close'] = ((df['high'] + tolerance) >= df['close']).sum()

        # Check: Low <= Open, Low <= Close
        results['low_vs_open'] = ((df['low'] - tolerance) <= df['open']).sum()
        results['low_vs_close'] = ((df['low'] - tolerance) <= df['close']).sum()

        # Check: High >= Low
        results['high_vs_low'] = ((df['high'] + tolerance) >= df['low']).sum()

        # Calculate total valid candles
        valid = (
            ((df['high'] + tolerance) >= df['open']) &
            ((df['high'] + tolerance) >= df['close']) &
            ((df['low'] - tolerance) <= df['open']) &
            ((df['low'] - tolerance) <= df['close']) &
            ((df['high'] + tolerance) >= df['low'])
        ).sum()

        results['invalid_count'] = len(df) - valid
        results['consistency_rate'] = valid / len(df) if len(df) > 0 else 0.0

        return results

    @staticmethod
    def validate_timestamps(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate timestamp consistency."""
        results = {
            'total_candles': len(df),
            'duplicates': 0,
            'out_of_order': 0,
            'future_timestamps': 0,
            'gaps': []
        }

        if len(df) == 0:
            return results

        # Check duplicates
        results['duplicates'] = df['ts_utc'].duplicated().sum()

        # Check ordering
        results['out_of_order'] = (df['ts_utc'].diff() < 0).sum()

        # Check future timestamps
        now_ms = int(time.time() * 1000)
        results['future_timestamps'] = (df['ts_utc'] > now_ms).sum()

        # Detect gaps (>5 consecutive missing candles)
        if len(df) > 1:
            diffs = df['ts_utc'].diff()
            median_diff = diffs.median()
            gaps = diffs[diffs > median_diff * 5]
            results['gaps'] = gaps.tolist()

        return results

    @staticmethod
    def validate_volume(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate volume data."""
        results = {
            'total_candles': len(df),
            'negative_volume': 0,
            'zero_volume': 0,
            'zero_volume_pct': 0.0,
            'volume_spikes': 0
        }

        if len(df) == 0 or 'volume' not in df.columns:
            return results

        # Check negative volumes
        results['negative_volume'] = (df['volume'] < 0).sum()

        # Check zero volumes
        results['zero_volume'] = (df['volume'] == 0).sum()
        results['zero_volume_pct'] = results['zero_volume'] / len(df) * 100

        # Detect volume spikes (>10x median)
        if len(df) > 0:
            median_vol = df['volume'].median()
            if median_vol > 0:
                results['volume_spikes'] = (df['volume'] > median_vol * 10).sum()

        return results


class PerformanceMonitor:
    """Monitors system performance and resources."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
        self.measurements: List[Dict[str, Any]] = []

    def record_measurement(self, phase: str) -> Dict[str, Any]:
        """Record a performance measurement."""
        measurement = {
            'phase': phase,
            'timestamp': time.time(),
            'elapsed_seconds': time.time() - self.start_time,
            'memory_mb': self.process.memory_info().rss / (1024 * 1024),
            'memory_delta_mb': (self.process.memory_info().rss - self.start_memory) / (1024 * 1024),
            'cpu_percent': self.process.cpu_percent(),
        }

        # Add disk usage if data dir exists
        try:
            disk = psutil.disk_usage(r'D:\Projects\ForexGPT\data')
            measurement['disk_free_gb'] = disk.free / (1024 ** 3)
            measurement['disk_used_gb'] = disk.used / (1024 ** 3)
        except:
            pass

        self.measurements.append(measurement)
        return measurement

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.measurements:
            return {}

        memory_values = [m['memory_mb'] for m in self.measurements]
        cpu_values = [m['cpu_percent'] for m in self.measurements if m['cpu_percent'] > 0]

        return {
            'total_duration_seconds': time.time() - self.start_time,
            'peak_memory_mb': max(memory_values) if memory_values else 0,
            'avg_memory_mb': np.mean(memory_values) if memory_values else 0,
            'memory_growth_mb': memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0,
            'avg_cpu_percent': np.mean(cpu_values) if cpu_values else 0,
            'measurements_count': len(self.measurements)
        }


class ReportGenerator:
    """Generates HTML and JSON reports."""

    @staticmethod
    def generate_html_report(
        config: E2ETestConfig,
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """Generate HTML test report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>E2E Test Report - {config.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-weight: bold; color: #7f8c8d; }}
        .metric-value {{ font-size: 1.2em; color: #2c3e50; }}
        .pass {{ color: #27ae60; }}
        .fail {{ color: #e74c3c; }}
        .warn {{ color: #f39c12; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
        th {{ background: #34495e; color: white; }}
        tr:nth-child(even) {{ background: #ecf0f1; }}
    </style>
</head>
<body>
    <h1>ForexGPT E2E Test Report</h1>
    <div class="summary">
        <div class="metric">
            <span class="metric-label">Test Start:</span>
            <span class="metric-value">{config.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Status:</span>
            <span class="metric-value {results.get('status', 'unknown').lower()}">{results.get('status', 'UNKNOWN')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Duration:</span>
            <span class="metric-value">{results.get('duration_minutes', 0):.1f} min</span>
        </div>
    </div>

    <h2>Test Configuration</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Max Data Period</td><td>{config.max_data_months} months</td></tr>
        <tr><td>Max Models</td><td>{config.max_models}</td></tr>
        <tr><td>Max Backtests</td><td>{config.max_backtests}</td></tr>
        <tr><td>Max Pattern Optimization</td><td>{config.max_pattern_optimization_cycles} cycles</td></tr>
        <tr><td>Target Positions</td><td>{config.target_positions}</td></tr>
        <tr><td>Concurrency Enabled</td><td>{config.enable_concurrency}</td></tr>
    </table>

    <h2>Phase Completion</h2>
    <table>
        <tr><th>Phase</th><th>Status</th><th>Duration (s)</th></tr>
        {ReportGenerator._generate_phase_rows(results.get('phases', {}))}
    </table>

    <h2>Resource Usage</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        {ReportGenerator._generate_resource_rows(results.get('performance', {}))}
    </table>

    <h2>Data Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        {ReportGenerator._generate_data_rows(results.get('data_stats', {}))}
    </table>
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f"HTML report generated: {output_path}")

    @staticmethod
    def _generate_phase_rows(phases: Dict[str, Any]) -> str:
        rows = []
        for phase_name, phase_data in phases.items():
            status = phase_data.get('status', 'unknown')
            status_class = 'pass' if status == 'completed' else 'fail' if status == 'failed' else 'warn'
            duration = phase_data.get('duration_seconds', 0)
            rows.append(
                f'<tr><td>{phase_name}</td>'
                f'<td class="{status_class}">{status.upper()}</td>'
                f'<td>{duration:.1f}</td></tr>'
            )
        return '\n'.join(rows)

    @staticmethod
    def _generate_resource_rows(performance: Dict[str, Any]) -> str:
        rows = []
        for key, value in performance.items():
            if isinstance(value, (int, float)):
                rows.append(f'<tr><td>{key.replace("_", " ").title()}</td><td>{value:.2f}</td></tr>')
        return '\n'.join(rows)

    @staticmethod
    def _generate_data_rows(data_stats: Dict[str, Any]) -> str:
        rows = []
        for key, value in data_stats.items():
            rows.append(f'<tr><td>{key.replace("_", " ").title()}</td><td>{value}</td></tr>')
        return '\n'.join(rows)

    @staticmethod
    def generate_json_metrics(results: Dict[str, Any], output_path: str) -> None:
        """Generate JSON metrics file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"JSON metrics generated: {output_path}")


def setup_test_logger(config: E2ETestConfig) -> None:
    """Setup loguru logger for test execution."""
    # Remove default handler
    logger.remove()

    # Add console handler with INFO level
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level="INFO"
    )

    # Add file handler for main log
    logger.add(
        config.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} | {message}",
        level="DEBUG",
        rotation="100 MB"
    )

    # Add file handler for errors only
    logger.add(
        config.error_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} | {message}",
        level="ERROR",
        rotation="50 MB"
    )

    logger.info(f"Test logger initialized. Logs: {config.log_file}")


def check_disk_space(data_dir: str, min_gb: float = 5.0) -> Tuple[bool, float]:
    """
    Check if sufficient disk space is available.

    Returns:
        Tuple of (sufficient, available_gb)
    """
    try:
        disk = psutil.disk_usage(data_dir)
        available_gb = disk.free / (1024 ** 3)
        return available_gb >= min_gb, available_gb
    except Exception as e:
        logger.error(f"Could not check disk space: {e}")
        return False, 0.0


def check_gpu_available() -> Tuple[bool, str]:
    """
    Check if GPU is available.

    Returns:
        Tuple of (available, device_name)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, device_name
        return False, "CPU only"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"Error: {str(e)}"
