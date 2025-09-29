# ui/chart_components/services/patterns_service.py
# Compatibility module - imports from refactored patterns modules for backward compatibility
from __future__ import annotations

# Import all components from their new locations
from .patterns.scan_worker import ScanWorker
from .patterns.detection_worker import DetectionWorker
from .patterns.patterns_service import PatternsService, OHLC_SYNONYMS, TS_SYNONYMS
from .patterns.historical_scan_worker import HistoricalScanWorker
from .patterns.utils import start_historical_scan, _min_required_bars

# Maintain backward compatibility with old private class names
_ScanWorker = ScanWorker
_DetectionWorker = DetectionWorker
_HistoricalScanWorker = HistoricalScanWorker

# Maintain backward compatibility
__all__ = [
    "PatternsService",
    "ScanWorker",
    "DetectionWorker",
    "HistoricalScanWorker",
    "_ScanWorker",
    "_DetectionWorker",
    "_HistoricalScanWorker",
    "start_historical_scan",
    "_min_required_bars",
    "OHLC_SYNONYMS",
    "TS_SYNONYMS"
]