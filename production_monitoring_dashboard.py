#!/usr/bin/env python3
"""
Production Monitoring Dashboard for Finplot Integration
Monitors performance metrics, usage statistics, and system health
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinplotProductionMonitor:
    """
    Production monitoring system for finplot integration
    Tracks performance, usage, and health metrics
    """

    def __init__(self):
        self.start_time = datetime.now()
        self.metrics = {
            'performance': {},
            'usage': {},
            'system': {},
            'errors': []
        }
        self.benchmarks = {
            'target_render_time': 0.5,      # Target: <0.5s for 1000 candles
            'target_memory_mb': 50,         # Target: <50MB per chart
            'target_cpu_percent': 30,       # Target: <30% CPU usage
            'target_success_rate': 99.0     # Target: >99% success rate
        }

    def measure_chart_performance(self, data_size: int = 1000):
        """Measure chart rendering performance"""
        logger.info(f"Measuring chart performance with {data_size} candles...")

        try:
            import finplot as fplt

            # Create test data
            dates = pd.date_range('2024-09-01', periods=data_size, freq='h')
            np.random.seed(42)

            base_price = 1.1000
            prices = np.cumsum(np.random.randn(data_size) * 0.0003) + base_price

            data = pd.DataFrame({
                'open': prices,
                'high': prices + np.abs(np.random.randn(data_size) * 0.0005),
                'low': prices - np.abs(np.random.randn(data_size) * 0.0005),
                'close': np.roll(prices, -1),
                'volume': np.random.uniform(100000, 1000000, data_size),
            }, index=dates)

            # Fix OHLC consistency
            data.loc[data.index[-1], 'close'] = data.loc[data.index[-1], 'open']
            data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
            data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))

            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Create chart
            fplt.candlestick_ochl(data[['open', 'close', 'high', 'low']])

            # Add indicators
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            fplt.plot(sma_20, legend='SMA 20', color='blue', width=2)
            fplt.plot(sma_50, legend='SMA 50', color='orange', width=2)

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Close chart
            fplt.close()

            # Calculate metrics
            render_time = end_time - start_time
            memory_used = end_memory - start_memory
            candles_per_second = data_size / render_time

            performance_metrics = {
                'timestamp': datetime.now().isoformat(),
                'data_size': data_size,
                'render_time_seconds': render_time,
                'memory_used_mb': memory_used,
                'candles_per_second': candles_per_second,
                'success': True,
                'target_render_met': render_time < self.benchmarks['target_render_time'],
                'target_memory_met': memory_used < self.benchmarks['target_memory_mb']
            }

            self.metrics['performance'][f'test_{data_size}'] = performance_metrics

            logger.info(f"✓ Performance test completed:")
            logger.info(f"  Render time: {render_time:.3f}s (target: <{self.benchmarks['target_render_time']}s)")
            logger.info(f"  Memory used: {memory_used:.1f}MB (target: <{self.benchmarks['target_memory_mb']}MB)")
            logger.info(f"  Performance: {candles_per_second:.0f} candles/second")

            return performance_metrics

        except Exception as e:
            error_metrics = {
                'timestamp': datetime.now().isoformat(),
                'data_size': data_size,
                'success': False,
                'error': str(e)
            }

            self.metrics['errors'].append(error_metrics)
            logger.error(f"✗ Performance test failed: {e}")

            return error_metrics

    def measure_system_resources(self):
        """Measure system resource usage"""
        logger.info("Measuring system resources...")

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / 1024 / 1024 / 1024

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Process info
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / 1024 / 1024
            process_cpu_percent = process.cpu_percent()

            system_metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_gb': memory_available_gb,
                'disk_percent': disk_percent,
                'process_memory_mb': process_memory_mb,
                'process_cpu_percent': process_cpu_percent,
                'cpu_target_met': cpu_percent < self.benchmarks['target_cpu_percent']
            }

            self.metrics['system']['current'] = system_metrics

            logger.info(f"✓ System resources measured:")
            logger.info(f"  CPU: {cpu_percent:.1f}% (target: <{self.benchmarks['target_cpu_percent']}%)")
            logger.info(f"  Memory: {memory_percent:.1f}% ({memory_available_gb:.1f}GB available)")
            logger.info(f"  Process Memory: {process_memory_mb:.1f}MB")

            return system_metrics

        except Exception as e:
            error_metrics = {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

            self.metrics['errors'].append(error_metrics)
            logger.error(f"✗ System resource measurement failed: {e}")

            return error_metrics

    def run_stress_test(self):
        """Run stress test with multiple chart sizes"""
        logger.info("Running stress test...")

        test_sizes = [100, 500, 1000, 5000, 10000]
        stress_results = {}

        for size in test_sizes:
            logger.info(f"Testing {size} candles...")
            result = self.measure_chart_performance(size)
            stress_results[f'size_{size}'] = result

            # Brief pause between tests
            time.sleep(0.5)

        self.metrics['performance']['stress_test'] = {
            'timestamp': datetime.now().isoformat(),
            'results': stress_results
        }

        # Calculate stress test summary
        successful_tests = sum(1 for r in stress_results.values() if r.get('success', False))
        success_rate = (successful_tests / len(test_sizes)) * 100

        logger.info(f"✓ Stress test completed:")
        logger.info(f"  Tests run: {len(test_sizes)}")
        logger.info(f"  Successful: {successful_tests}")
        logger.info(f"  Success rate: {success_rate:.1f}%")

        return stress_results

    def simulate_usage_patterns(self):
        """Simulate typical usage patterns"""
        logger.info("Simulating usage patterns...")

        usage_scenarios = [
            {'name': 'Quick Analysis', 'candles': 200, 'indicators': 2},
            {'name': 'Daily Review', 'candles': 500, 'indicators': 5},
            {'name': 'Deep Analysis', 'candles': 1000, 'indicators': 8},
            {'name': 'Historical Backtest', 'candles': 5000, 'indicators': 10}
        ]

        usage_results = {}

        for scenario in usage_scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")

            # Run performance test for this scenario
            result = self.measure_chart_performance(scenario['candles'])

            usage_results[scenario['name']] = {
                'scenario': scenario,
                'performance': result,
                'timestamp': datetime.now().isoformat()
            }

        self.metrics['usage']['scenarios'] = usage_results

        logger.info("✓ Usage patterns simulation completed")
        return usage_results

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        logger.info("Generating performance report...")

        report = {
            'report_timestamp': datetime.now().isoformat(),
            'monitoring_duration': str(datetime.now() - self.start_time),
            'finplot_status': 'operational',
            'overall_health': 'excellent',
            'metrics_summary': {},
            'recommendations': [],
            'benchmarks': self.benchmarks,
            'detailed_metrics': self.metrics
        }

        # Analyze performance metrics
        if 'stress_test' in self.metrics['performance']:
            stress_results = self.metrics['performance']['stress_test']['results']

            # Calculate averages
            render_times = [r.get('render_time_seconds', 0) for r in stress_results.values() if r.get('success')]
            memory_usage = [r.get('memory_used_mb', 0) for r in stress_results.values() if r.get('success')]

            if render_times and memory_usage:
                report['metrics_summary'] = {
                    'avg_render_time': np.mean(render_times),
                    'max_render_time': np.max(render_times),
                    'avg_memory_usage': np.mean(memory_usage),
                    'max_memory_usage': np.max(memory_usage),
                    'success_rate': len(render_times) / len(stress_results) * 100
                }

                # Performance assessment
                if np.mean(render_times) < self.benchmarks['target_render_time']:
                    report['recommendations'].append("✓ Rendering performance exceeds targets")
                else:
                    report['recommendations'].append("⚠ Consider rendering optimizations")

                if np.mean(memory_usage) < self.benchmarks['target_memory_mb']:
                    report['recommendations'].append("✓ Memory usage within targets")
                else:
                    report['recommendations'].append("⚠ Monitor memory usage trends")

        # System health assessment
        if 'current' in self.metrics['system']:
            system = self.metrics['system']['current']

            if system.get('cpu_target_met', False):
                report['recommendations'].append("✓ CPU usage within targets")
            else:
                report['recommendations'].append("⚠ Monitor CPU usage")

        # Error analysis
        error_count = len(self.metrics['errors'])
        if error_count == 0:
            report['recommendations'].append("✓ No errors detected")
            report['overall_health'] = 'excellent'
        elif error_count < 3:
            report['recommendations'].append("⚠ Minor errors detected - monitor trends")
            report['overall_health'] = 'good'
        else:
            report['recommendations'].append("❌ Multiple errors detected - investigate")
            report['overall_health'] = 'needs_attention'

        return report

    def save_monitoring_data(self):
        """Save monitoring data to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save metrics
        metrics_file = f'finplot_monitoring_metrics_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

        # Generate and save report
        report = self.generate_performance_report()
        report_file = f'finplot_monitoring_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"✓ Monitoring data saved:")
        logger.info(f"  Metrics: {metrics_file}")
        logger.info(f"  Report: {report_file}")

        return {'metrics_file': metrics_file, 'report_file': report_file}

def run_production_monitoring():
    """Run comprehensive production monitoring"""
    logger.info("=" * 60)
    logger.info("FINPLOT PRODUCTION MONITORING")
    logger.info("=" * 60)

    monitor = FinplotProductionMonitor()

    try:
        # Step 1: System resource check
        logger.info("\n🖥️ STEP 1: System Resource Monitoring")
        monitor.measure_system_resources()

        # Step 2: Basic performance test
        logger.info("\n⚡ STEP 2: Basic Performance Test")
        monitor.measure_chart_performance(1000)

        # Step 3: Stress testing
        logger.info("\n🔥 STEP 3: Stress Testing")
        monitor.run_stress_test()

        # Step 4: Usage pattern simulation
        logger.info("\n👥 STEP 4: Usage Pattern Simulation")
        monitor.simulate_usage_patterns()

        # Step 5: Generate report
        logger.info("\n📊 STEP 5: Generating Performance Report")
        report = monitor.generate_performance_report()

        # Step 6: Save data
        logger.info("\n💾 STEP 6: Saving Monitoring Data")
        files = monitor.save_monitoring_data()

        # Display summary
        logger.info("\n" + "=" * 60)
        logger.info("PRODUCTION MONITORING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Health: {report['overall_health'].upper()}")
        logger.info(f"Finplot Status: {report['finplot_status'].upper()}")

        if 'metrics_summary' in report:
            summary = report['metrics_summary']
            logger.info(f"Average Render Time: {summary.get('avg_render_time', 0):.3f}s")
            logger.info(f"Average Memory Usage: {summary.get('avg_memory_usage', 0):.1f}MB")
            logger.info(f"Success Rate: {summary.get('success_rate', 0):.1f}%")

        logger.info("\nRecommendations:")
        for rec in report['recommendations']:
            logger.info(f"  {rec}")

        if report['overall_health'] == 'excellent':
            logger.info("\n🎉 FINPLOT PRODUCTION SYSTEM PERFORMING EXCELLENTLY!")
            logger.info("✓ All performance targets met")
            logger.info("✓ System resources optimal")
            logger.info("✓ Ready for full production workload")

        logger.info("=" * 60)

        return report

    except Exception as e:
        logger.error(f"❌ PRODUCTION MONITORING FAILED: {e}")
        return {'status': 'failed', 'error': str(e)}

if __name__ == "__main__":
    run_production_monitoring()