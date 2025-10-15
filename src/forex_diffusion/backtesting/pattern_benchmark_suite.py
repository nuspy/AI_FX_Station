"""
Pattern Benchmark Suite
Comprehensive benchmarking system for pattern detection accuracy and performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import logging
from enum import Enum
import json
import pickle
from collections import defaultdict
from abc import ABC, abstractmethod

# Statistical imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from scipy import stats
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Pattern types for benchmarking"""
    # Reversal Patterns
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"

    # Continuation Patterns
    FLAG = "flag"
    PENNANT = "pennant"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    RECTANGLE = "rectangle"

    # Candlestick Patterns
    HAMMER = "hammer"
    DOJI = "doji"
    ENGULFING_BULL = "engulfing_bull"
    ENGULFING_BEAR = "engulfing_bear"
    SHOOTING_STAR = "shooting_star"

    # Complex Patterns
    CUP_HANDLE = "cup_handle"
    DIAMOND = "diamond"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"

class DetectionMethod(Enum):
    """Pattern detection methods"""
    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"

@dataclass
class PatternDetectionResult:
    """Result of pattern detection"""
    pattern_type: PatternType
    detection_time: datetime
    confidence: float
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    key_points: Optional[List[Tuple[datetime, float]]] = None

    # Prediction info
    predicted_direction: Optional[str] = None  # 'up', 'down', 'neutral'
    predicted_magnitude: Optional[float] = None
    predicted_timeframe: Optional[int] = None  # hours

    # Detection method metadata
    detection_method: DetectionMethod = DetectionMethod.RULE_BASED
    model_name: Optional[str] = None

@dataclass
class PatternGroundTruth:
    """Ground truth for pattern validation"""
    pattern_type: PatternType
    start_time: datetime
    end_time: datetime
    actual_direction: str  # 'up', 'down', 'neutral'
    actual_magnitude: float
    actual_timeframe: int

    # Pattern quality rating (expert annotation)
    quality_score: float = 1.0  # 0-1 scale
    annotator: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class BenchmarkMetrics:
    """Comprehensive pattern benchmarking metrics"""

    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    specificity: float = 0.0

    # Pattern-specific metrics
    detection_rate: float = 0.0  # How many patterns were detected
    false_positive_rate: float = 0.0
    missed_pattern_rate: float = 0.0

    # Quality metrics
    confidence_correlation: float = 0.0  # Correlation between confidence and accuracy
    temporal_accuracy: float = 0.0  # How well pattern timing is predicted
    magnitude_mae: float = 0.0  # MAE for magnitude predictions
    direction_accuracy: float = 0.0  # Accuracy for direction predictions

    # Performance metrics
    avg_detection_time: float = 0.0  # seconds
    patterns_per_second: float = 0.0

    # Statistical significance
    mcnemar_p_value: Optional[float] = None  # For comparing detection methods

    # Meta information
    total_patterns: int = 0
    detected_patterns: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

@dataclass
class BenchmarkResults:
    """Complete benchmark results"""

    # Results by pattern type
    pattern_metrics: Dict[str, BenchmarkMetrics] = field(default_factory=dict)

    # Overall metrics
    overall_metrics: Optional[BenchmarkMetrics] = None

    # Method comparison
    method_comparison: Dict[str, BenchmarkMetrics] = field(default_factory=dict)

    # Statistical tests
    statistical_tests: Dict[str, Any] = field(default_factory=dict)

    # Meta information
    evaluation_period: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now(), datetime.now()))
    data_pairs: List[str] = field(default_factory=list)
    methods_tested: List[DetectionMethod] = field(default_factory=list)

    # Performance benchmarks
    speed_benchmarks: Dict[str, float] = field(default_factory=dict)

class PatternDetector(ABC):
    """Abstract base class for pattern detectors"""

    @abstractmethod
    def detect_patterns(self,
                       data: pd.DataFrame,
                       pattern_types: Optional[List[PatternType]] = None) -> List[PatternDetectionResult]:
        """Detect patterns in market data"""
        pass

    @abstractmethod
    def get_detection_method(self) -> DetectionMethod:
        """Return the detection method type"""
        pass

class PatternBenchmarkSuite:
    """
    Comprehensive pattern detection benchmarking suite.

    Features:
    - Multi-method comparison (rule-based vs ML vs hybrid)
    - Pattern-specific accuracy metrics
    - Statistical significance testing
    - Performance benchmarking
    - Temporal accuracy assessment
    - Direction and magnitude prediction evaluation
    """

    def __init__(self,
                 results_dir: str = "benchmark_results/patterns/",
                 ground_truth_dir: str = "ground_truth/patterns/"):

        self.results_dir = Path(results_dir)
        self.ground_truth_dir = Path(ground_truth_dir)

        # Create directories
        for dir_path in [self.results_dir, self.ground_truth_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Ground truth data
        self.ground_truth: List[PatternGroundTruth] = []

        # Detection results by method
        self.detection_results: Dict[str, List[PatternDetectionResult]] = defaultdict(list)

        # Registered detectors
        self.detectors: Dict[str, PatternDetector] = {}

        # Benchmark configuration
        self.pattern_types = list(PatternType)
        self.overlap_threshold = 0.5  # Minimum overlap for pattern matching

        logger.info("PatternBenchmarkSuite initialized")

    def register_detector(self, name: str, detector: PatternDetector):
        """Register a pattern detector for benchmarking"""
        self.detectors[name] = detector
        logger.info(f"Registered detector: {name}")

    def load_ground_truth(self, ground_truth_file: str):
        """Load ground truth pattern data"""
        file_path = self.ground_truth_dir / ground_truth_file

        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    ground_truth_data = json.load(f)

                self.ground_truth = []
                for gt_dict in ground_truth_data:
                    gt = PatternGroundTruth(
                        pattern_type=PatternType(gt_dict['pattern_type']),
                        start_time=datetime.fromisoformat(gt_dict['start_time']),
                        end_time=datetime.fromisoformat(gt_dict['end_time']),
                        actual_direction=gt_dict['actual_direction'],
                        actual_magnitude=gt_dict['actual_magnitude'],
                        actual_timeframe=gt_dict['actual_timeframe'],
                        quality_score=gt_dict.get('quality_score', 1.0),
                        annotator=gt_dict.get('annotator'),
                        notes=gt_dict.get('notes')
                    )
                    self.ground_truth.append(gt)

            elif file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    self.ground_truth = pickle.load(f)

            logger.info(f"Loaded {len(self.ground_truth)} ground truth patterns from {file_path}")

        except Exception as e:
            logger.error(f"Failed to load ground truth from {file_path}: {e}")

    def create_synthetic_ground_truth(self,
                                    data: pd.DataFrame,
                                    n_patterns_per_type: int = 10) -> List[PatternGroundTruth]:
        """Create synthetic ground truth for testing (simplified patterns)"""

        synthetic_gt = []
        np.random.seed(42)

        data_start = data.index[0]
        data_end = data.index[-1]
        data_duration = data_end - data_start

        for pattern_type in [PatternType.DOUBLE_TOP, PatternType.DOUBLE_BOTTOM,
                           PatternType.HEAD_SHOULDERS, PatternType.FLAG]:

            for i in range(n_patterns_per_type):
                # Random start time
                random_offset = np.random.random() * 0.8  # Use first 80% of data
                start_time = data_start + timedelta(seconds=data_duration.total_seconds() * random_offset)

                # Pattern duration (6-48 hours)
                duration_hours = np.random.uniform(6, 48)
                end_time = start_time + timedelta(hours=duration_hours)

                if end_time > data_end:
                    continue

                # Get price data for this period
                pattern_data = data[(data.index >= start_time) & (data.index <= end_time)]

                if len(pattern_data) < 10:
                    continue

                # Calculate actual direction and magnitude
                start_price = pattern_data['close'].iloc[0]
                end_price = pattern_data['close'].iloc[-1]

                price_change = (end_price - start_price) / start_price
                actual_magnitude = abs(price_change)

                if price_change > 0.001:  # 0.1% threshold
                    actual_direction = 'up'
                elif price_change < -0.001:
                    actual_direction = 'down'
                else:
                    actual_direction = 'neutral'

                # Timeframe (time to realize the movement)
                actual_timeframe = int(duration_hours)

                gt = PatternGroundTruth(
                    pattern_type=pattern_type,
                    start_time=start_time,
                    end_time=end_time,
                    actual_direction=actual_direction,
                    actual_magnitude=actual_magnitude,
                    actual_timeframe=actual_timeframe,
                    quality_score=np.random.uniform(0.7, 1.0),
                    annotator="synthetic"
                )

                synthetic_gt.append(gt)

        self.ground_truth = synthetic_gt
        logger.info(f"Created {len(synthetic_gt)} synthetic ground truth patterns")

        return synthetic_gt

    def run_detection_benchmark(self,
                              data: pd.DataFrame,
                              detector_names: Optional[List[str]] = None) -> Dict[str, List[PatternDetectionResult]]:
        """Run pattern detection benchmark on all registered detectors"""

        if detector_names is None:
            detector_names = list(self.detectors.keys())

        logger.info(f"Running detection benchmark for {len(detector_names)} detectors")

        all_results = {}

        for detector_name in detector_names:
            if detector_name not in self.detectors:
                logger.warning(f"Detector {detector_name} not registered, skipping")
                continue

            detector = self.detectors[detector_name]

            # Measure detection time
            start_time = datetime.now()

            try:
                detection_results = detector.detect_patterns(data, self.pattern_types)

                detection_time = (datetime.now() - start_time).total_seconds()

                # Add performance metadata
                for result in detection_results:
                    result.model_name = detector_name

                all_results[detector_name] = detection_results
                self.detection_results[detector_name] = detection_results

                logger.info(f"Detector {detector_name}: {len(detection_results)} patterns detected in {detection_time:.2f}s")

            except Exception as e:
                logger.error(f"Detection failed for {detector_name}: {e}")
                all_results[detector_name] = []

        return all_results

    def calculate_pattern_overlap(self,
                                detected: PatternDetectionResult,
                                ground_truth: PatternGroundTruth) -> float:
        """Calculate temporal overlap between detected and ground truth patterns"""

        if detected.start_time is None or detected.end_time is None:
            # Use detection time as point estimate
            det_start = det_end = detected.detection_time
        else:
            det_start = detected.start_time
            det_end = detected.end_time

        gt_start = ground_truth.start_time
        gt_end = ground_truth.end_time

        # Calculate overlap
        overlap_start = max(det_start, gt_start)
        overlap_end = min(det_end, gt_end)

        if overlap_end <= overlap_start:
            return 0.0  # No overlap

        overlap_duration = (overlap_end - overlap_start).total_seconds()
        gt_duration = (gt_end - gt_start).total_seconds()
        det_duration = (det_end - det_start).total_seconds()

        # Use IoU-like metric: overlap / union
        union_duration = gt_duration + det_duration - overlap_duration

        return overlap_duration / union_duration if union_duration > 0 else 0.0

    def match_detections_to_ground_truth(self,
                                       detections: List[PatternDetectionResult],
                                       method_name: str) -> Tuple[List[Tuple], List[PatternDetectionResult], List[PatternGroundTruth]]:
        """Match detections to ground truth patterns"""

        matched_pairs = []
        unmatched_detections = detections.copy()
        unmatched_ground_truth = self.ground_truth.copy()

        # For each ground truth pattern, find best matching detection
        for gt in self.ground_truth[:]:  # Copy to avoid modification during iteration
            best_match = None
            best_overlap = 0.0
            best_detection = None

            for detection in unmatched_detections[:]:
                # Only match same pattern types
                if detection.pattern_type != gt.pattern_type:
                    continue

                overlap = self.calculate_pattern_overlap(detection, gt)

                if overlap > best_overlap and overlap >= self.overlap_threshold:
                    best_overlap = overlap
                    best_match = detection
                    best_detection = detection

            if best_match:
                matched_pairs.append((gt, best_match, best_overlap))
                unmatched_detections.remove(best_match)
                unmatched_ground_truth.remove(gt)

        return matched_pairs, unmatched_detections, unmatched_ground_truth

    def calculate_benchmark_metrics(self,
                                  method_name: str,
                                  detections: List[PatternDetectionResult]) -> BenchmarkMetrics:
        """Calculate comprehensive benchmark metrics for a detection method"""

        # Match detections to ground truth
        matched_pairs, false_positives, false_negatives = self.match_detections_to_ground_truth(detections, method_name)

        # Basic counts
        true_positives = len(matched_pairs)
        false_positive_count = len(false_positives)
        false_negative_count = len(false_negatives)
        total_ground_truth = len(self.ground_truth)
        total_detections = len(detections)

        # Classification metrics
        precision = true_positives / total_detections if total_detections > 0 else 0.0
        recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        detection_rate = true_positives / total_ground_truth if total_ground_truth > 0 else 0.0
        false_positive_rate = false_positive_count / total_detections if total_detections > 0 else 0.0
        missed_pattern_rate = false_negative_count / total_ground_truth if total_ground_truth > 0 else 0.0

        # Quality metrics for matched pairs
        confidence_scores = []
        temporal_accuracy_scores = []
        magnitude_errors = []
        direction_correct = []

        for gt, detection, overlap in matched_pairs:
            confidence_scores.append(detection.confidence)
            temporal_accuracy_scores.append(overlap)

            # Magnitude error
            if detection.predicted_magnitude and gt.actual_magnitude:
                magnitude_error = abs(detection.predicted_magnitude - gt.actual_magnitude)
                magnitude_errors.append(magnitude_error)

            # Direction accuracy
            if detection.predicted_direction and gt.actual_direction:
                direction_correct.append(detection.predicted_direction == gt.actual_direction)

        # Calculate quality metrics
        confidence_correlation = 0.0
        if len(confidence_scores) > 1 and len(temporal_accuracy_scores) > 1:
            try:
                correlation, _ = stats.pearsonr(confidence_scores, temporal_accuracy_scores)
                confidence_correlation = correlation if not np.isnan(correlation) else 0.0
            except Exception:
                confidence_correlation = 0.0

        temporal_accuracy = np.mean(temporal_accuracy_scores) if temporal_accuracy_scores else 0.0
        magnitude_mae = np.mean(magnitude_errors) if magnitude_errors else 0.0
        direction_accuracy = np.mean(direction_correct) if direction_correct else 0.0

        return BenchmarkMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            missed_pattern_rate=missed_pattern_rate,
            confidence_correlation=confidence_correlation,
            temporal_accuracy=temporal_accuracy,
            magnitude_mae=magnitude_mae,
            direction_accuracy=direction_accuracy,
            total_patterns=total_ground_truth,
            detected_patterns=total_detections,
            true_positives=true_positives,
            false_positives=false_positive_count,
            false_negatives=false_negative_count
        )

    def run_comprehensive_benchmark(self,
                                  data: pd.DataFrame,
                                  create_synthetic_gt: bool = True) -> BenchmarkResults:
        """Run comprehensive pattern detection benchmark"""

        logger.info("Starting comprehensive pattern benchmark")

        # Create or load ground truth
        if create_synthetic_gt or not self.ground_truth:
            self.create_synthetic_ground_truth(data)

        if not self.ground_truth:
            raise ValueError("No ground truth patterns available for benchmarking")

        # Run detection benchmark
        detection_results = self.run_detection_benchmark(data)

        if not detection_results:
            raise ValueError("No detection results available")

        # Initialize results
        results = BenchmarkResults(
            evaluation_period=(data.index[0], data.index[-1]),
            methods_tested=[self.detectors[name].get_detection_method() for name in detection_results.keys()]
        )

        # Calculate metrics for each method
        method_metrics = {}

        for method_name, detections in detection_results.items():
            method_metrics[method_name] = self.calculate_benchmark_metrics(method_name, detections)

        results.method_comparison = method_metrics

        # Calculate pattern-specific metrics
        pattern_specific_metrics = {}

        for pattern_type in self.pattern_types:
            # Get ground truth for this pattern type
            gt_for_pattern = [gt for gt in self.ground_truth if gt.pattern_type == pattern_type]

            if not gt_for_pattern:
                continue

            # Combine all detections for this pattern type
            all_detections_for_pattern = []
            for method_name, detections in detection_results.items():
                pattern_detections = [d for d in detections if d.pattern_type == pattern_type]
                all_detections_for_pattern.extend(pattern_detections)

            if all_detections_for_pattern:
                # Temporarily set ground truth to just this pattern type
                original_gt = self.ground_truth
                self.ground_truth = gt_for_pattern

                pattern_metrics = self.calculate_benchmark_metrics(f"combined_{pattern_type.value}", all_detections_for_pattern)
                pattern_specific_metrics[pattern_type.value] = pattern_metrics

                # Restore original ground truth
                self.ground_truth = original_gt

        results.pattern_metrics = pattern_specific_metrics

        # Calculate overall metrics (aggregate across all methods)
        all_detections = []
        for detections in detection_results.values():
            all_detections.extend(detections)

        if all_detections:
            results.overall_metrics = self.calculate_benchmark_metrics("overall", all_detections)

        # Statistical significance tests
        results.statistical_tests = self._run_statistical_tests(method_metrics)

        # Speed benchmarks
        results.speed_benchmarks = self._calculate_speed_benchmarks(detection_results, data)

        logger.info("Comprehensive benchmark completed")

        return results

    def _run_statistical_tests(self, method_metrics: Dict[str, BenchmarkMetrics]) -> Dict[str, Any]:
        """Run statistical significance tests between methods"""

        statistical_tests = {}

        method_names = list(method_metrics.keys())

        if len(method_names) < 2:
            return statistical_tests

        # McNemar's test for paired comparison of methods
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:

                metrics1 = method_metrics[method1]
                metrics2 = method_metrics[method2]

                # Create contingency table for McNemar's test
                # Need to track individual pattern results for this
                # Simplified approach using aggregate metrics

                try:
                    # Use F1 scores for comparison
                    f1_diff = metrics1.f1_score - metrics2.f1_score

                    # Simple significance test based on confidence intervals
                    # This is a simplified approach - in practice you'd need individual predictions

                    statistical_tests[f"{method1}_vs_{method2}"] = {
                        'f1_difference': f1_diff,
                        'method1_f1': metrics1.f1_score,
                        'method2_f1': metrics2.f1_score,
                        'significant': abs(f1_diff) > 0.05  # Simplified significance threshold
                    }

                except Exception as e:
                    logger.warning(f"Statistical test failed for {method1} vs {method2}: {e}")

        return statistical_tests

    def _calculate_speed_benchmarks(self,
                                  detection_results: Dict[str, List[PatternDetectionResult]],
                                  data: pd.DataFrame) -> Dict[str, float]:
        """Calculate speed benchmarks for different methods"""

        speed_benchmarks = {}

        data_hours = len(data) / (24 if 'h' in str(data.index.freq) else 1)  # Approximate

        for method_name, detections in detection_results.items():

            # Estimated processing time (would need actual timing in production)
            # This is a placeholder calculation
            patterns_detected = len(detections)
            estimated_time = patterns_detected * 0.01  # 10ms per pattern (placeholder)

            patterns_per_hour = patterns_detected / data_hours if data_hours > 0 else 0
            patterns_per_second = patterns_per_hour / 3600 if patterns_per_hour > 0 else 0

            speed_benchmarks[method_name] = {
                'patterns_detected': patterns_detected,
                'estimated_time_seconds': estimated_time,
                'patterns_per_second': patterns_per_second,
                'data_hours_processed': data_hours
            }

        return speed_benchmarks

    def generate_benchmark_report(self, results: BenchmarkResults) -> str:
        """Generate comprehensive benchmark report"""

        report = []
        report.append("=" * 80)
        report.append("PATTERN DETECTION BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Evaluation Period: {results.evaluation_period[0]} to {results.evaluation_period[1]}")
        report.append(f"Methods Tested: {', '.join([m.value for m in results.methods_tested])}")
        report.append("")

        # Overall metrics
        if results.overall_metrics:
            overall = results.overall_metrics
            report.append("OVERALL PERFORMANCE:")
            report.append(f"  Total Ground Truth Patterns: {overall.total_patterns}")
            report.append(f"  Total Detected Patterns: {overall.detected_patterns}")
            report.append(f"  Precision: {overall.precision:.4f}")
            report.append(f"  Recall: {overall.recall:.4f}")
            report.append(f"  F1-Score: {overall.f1_score:.4f}")
            report.append(f"  Detection Rate: {overall.detection_rate:.4f}")
            report.append("")

        # Method comparison
        if results.method_comparison:
            report.append("METHOD COMPARISON:")
            report.append("-" * 40)

            for method_name, metrics in results.method_comparison.items():
                report.append(f"\n{method_name.upper()}:")
                report.append(f"  Precision: {metrics.precision:.4f}")
                report.append(f"  Recall: {metrics.recall:.4f}")
                report.append(f"  F1-Score: {metrics.f1_score:.4f}")
                report.append(f"  Detection Rate: {metrics.detection_rate:.4f}")
                report.append(f"  False Positive Rate: {metrics.false_positive_rate:.4f}")
                report.append(f"  Direction Accuracy: {metrics.direction_accuracy:.4f}")
                report.append(f"  Temporal Accuracy: {metrics.temporal_accuracy:.4f}")

        # Pattern-specific results
        if results.pattern_metrics:
            report.append("\nPATTERN-SPECIFIC PERFORMANCE:")
            report.append("-" * 40)

            for pattern_type, metrics in results.pattern_metrics.items():
                report.append(f"\n{pattern_type.upper()}:")
                report.append(f"  F1-Score: {metrics.f1_score:.4f}")
                report.append(f"  Detection Rate: {metrics.detection_rate:.4f}")
                report.append(f"  Patterns: {metrics.total_patterns} GT, {metrics.detected_patterns} detected")

        # Statistical tests
        if results.statistical_tests:
            report.append("\nSTATISTICAL SIGNIFICANCE TESTS:")
            report.append("-" * 40)

            for test_name, test_result in results.statistical_tests.items():
                significance = "SIGNIFICANT" if test_result['significant'] else "NOT SIGNIFICANT"
                report.append(f"{test_name}: {significance} (ΔF1: {test_result['f1_difference']:.4f})")

        # Speed benchmarks
        if results.speed_benchmarks:
            report.append("\nPERFORMANCE BENCHMARKS:")
            report.append("-" * 40)

            for method_name, speed_data in results.speed_benchmarks.items():
                report.append(f"\n{method_name}:")
                report.append(f"  Patterns/Second: {speed_data['patterns_per_second']:.2f}")
                report.append(f"  Processing Time: {speed_data['estimated_time_seconds']:.2f}s")

        return "\n".join(report)

    def save_benchmark_results(self, results: BenchmarkResults, filename: Optional[str] = None):
        """Save benchmark results to file"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pattern_benchmark_{timestamp}.pkl"

        results_path = self.results_dir / filename

        try:
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)

            logger.info(f"Benchmark results saved to {results_path}")

        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")


# Example pattern detectors for testing

class DummyRuleBasedDetector(PatternDetector):
    """Dummy rule-based detector for testing"""

    def detect_patterns(self, data: pd.DataFrame, pattern_types: Optional[List[PatternType]] = None) -> List[PatternDetectionResult]:
        results = []

        # Simple logic: detect patterns based on price movements
        for i in range(10, len(data) - 10):
            window = data.iloc[i-10:i+10]

            # Simple double top detection (placeholder)
            if len(window) > 0:
                price_std = window['close'].std()

                if np.random.random() > 0.95:  # Random detection for testing
                    result = PatternDetectionResult(
                        pattern_type=PatternType.DOUBLE_TOP,
                        detection_time=data.index[i],
                        confidence=np.random.uniform(0.6, 0.9),
                        start_time=data.index[i-5],
                        end_time=data.index[i+5],
                        predicted_direction='down',
                        predicted_magnitude=price_std,
                        predicted_timeframe=24,
                        detection_method=DetectionMethod.RULE_BASED
                    )
                    results.append(result)

        return results

    def get_detection_method(self) -> DetectionMethod:
        return DetectionMethod.RULE_BASED

class DummyMLDetector(PatternDetector):
    """Dummy ML-based detector for testing"""

    def detect_patterns(self, data: pd.DataFrame, pattern_types: Optional[List[PatternType]] = None) -> List[PatternDetectionResult]:
        results = []

        # ML-based logic (placeholder)
        for i in range(20, len(data) - 20):
            if np.random.random() > 0.92:  # Different detection rate
                pattern_types_list = [PatternType.DOUBLE_TOP, PatternType.HEAD_SHOULDERS, PatternType.FLAG]
                selected_pattern = np.random.choice(pattern_types_list)

                result = PatternDetectionResult(
                    pattern_type=selected_pattern,
                    detection_time=data.index[i],
                    confidence=np.random.uniform(0.7, 0.95),  # Generally higher confidence
                    start_time=data.index[i-10],
                    end_time=data.index[i+10],
                    predicted_direction=np.random.choice(['up', 'down']),
                    predicted_magnitude=np.random.uniform(0.001, 0.05),
                    predicted_timeframe=np.random.choice([6, 12, 24, 48]),
                    detection_method=DetectionMethod.ML_BASED
                )
                results.append(result)

        return results

    def get_detection_method(self) -> DetectionMethod:
        return DetectionMethod.ML_BASED


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Pattern Benchmark Suite...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-03-01', freq='h')
    prices = 1.1 + np.cumsum(np.random.randn(len(dates)) * 0.001)

    data = pd.DataFrame({
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)

    # Initialize benchmark suite
    benchmark = PatternBenchmarkSuite()

    # Register detectors
    benchmark.register_detector("rule_based", DummyRuleBasedDetector())
    benchmark.register_detector("ml_based", DummyMLDetector())

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(data)

    # Generate report
    report = benchmark.generate_benchmark_report(results)
    print(report)

    print("\nPattern Benchmark Suite test completed!")