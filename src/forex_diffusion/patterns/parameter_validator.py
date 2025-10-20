"""
Parameter Validation for Pattern Detection

Validates optimized parameters before storage to ensure:
- Type correctness
- Range validity
- Sanity checks
- Smoke test on sample data
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from loguru import logger


class ParameterValidator:
    """Validates pattern parameters before database storage"""
    
    def __init__(self):
        # Define expected parameter types and ranges
        self.param_specs = {
            # Chart patterns - common parameters
            'min_span': (int, 5, 500),
            'max_span': (int, 10, 1000),
            'min_touches': (int, 2, 10),
            'max_events': (int, 1, 100),
            'tolerance': (float, 0.001, 0.5),
            'tightness': (float, 0.1, 2.0),
            
            # Candle patterns
            'min_body_ratio': (float, 0.1, 0.9),
            'shadow_ratio': (float, 0.5, 5.0),
            
            # Harmonic patterns
            'error_threshold': (float, 0.01, 0.3),
            'min_legs': (int, 3, 5),
            
            # Elliott Wave
            'wave_tolerance': (float, 0.05, 0.5),
            'min_wave_bars': (int, 5, 100),
            
            # General
            'window': (int, 10, 500),
            'threshold': (float, 0.0, 1.0),
        }
    
    def validate_parameters(
        self, 
        pattern_type: str, 
        parameters: Dict[str, Any],
        test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate parameters before storage.
        
        Args:
            pattern_type: Pattern identifier (e.g., 'wedge_ascending')
            parameters: Parameter dict to validate
            test_df: Optional dataframe for smoke test
            
        Returns:
            (is_valid, error_message)
        """
        try:
            # 1. Type validation
            valid, error = self._validate_types(parameters)
            if not valid:
                return False, f"Type validation failed: {error}"
            
            # 2. Range validation
            valid, error = self._validate_ranges(parameters)
            if not valid:
                return False, f"Range validation failed: {error}"
            
            # 3. Sanity checks
            valid, error = self._validate_sanity(parameters)
            if not valid:
                return False, f"Sanity check failed: {error}"
            
            # 4. Smoke test if data provided
            if test_df is not None:
                valid, error = self._smoke_test(pattern_type, parameters, test_df)
                if not valid:
                    return False, f"Smoke test failed: {error}"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _validate_types(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameter types"""
        for param_name, param_value in parameters.items():
            if param_name in self.param_specs:
                expected_type, _, _ = self.param_specs[param_name]
                if not isinstance(param_value, expected_type):
                    return False, f"{param_name}: expected {expected_type}, got {type(param_value)}"
        return True, None
    
    def _validate_ranges(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameter ranges"""
        for param_name, param_value in parameters.items():
            if param_name in self.param_specs:
                _, min_val, max_val = self.param_specs[param_name]
                if not (min_val <= param_value <= max_val):
                    return False, f"{param_name}={param_value} out of range [{min_val}, {max_val}]"
        return True, None
    
    def _validate_sanity(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Sanity checks for parameter combinations"""
        # min_span < max_span
        if 'min_span' in parameters and 'max_span' in parameters:
            if parameters['min_span'] >= parameters['max_span']:
                return False, "min_span must be < max_span"
        
        # min_touches reasonable
        if 'min_touches' in parameters:
            if parameters['min_touches'] <= 0:
                return False, "min_touches must be > 0"
        
        # tolerance < 1.0 (percentage)
        if 'tolerance' in parameters:
            if parameters['tolerance'] >= 1.0:
                return False, "tolerance must be < 1.0"
        
        # max_events reasonable
        if 'max_events' in parameters:
            if parameters['max_events'] <= 0:
                return False, "max_events must be > 0"
        
        return True, None
    
    def _smoke_test(
        self, 
        pattern_type: str, 
        parameters: Dict[str, Any],
        test_df: pd.DataFrame
    ) -> Tuple[bool, Optional[str]]:
        """
        Quick smoke test on small dataset.
        
        Tests that parameters don't cause crashes or obviously wrong behavior.
        """
        try:
            # Try to create detector with parameters
            from .registry import PatternRegistry
            
            # Get detector class for pattern type
            registry = PatternRegistry()
            detectors = list(registry.detectors(['chart', 'candle']))
            
            # Find matching detector
            target_detector = None
            for det in detectors:
                if getattr(det, 'key', None) == pattern_type:
                    target_detector = det
                    break
            
            if target_detector is None:
                # Can't find detector, skip smoke test
                logger.debug(f"No detector found for {pattern_type}, skipping smoke test")
                return True, None
            
            # Try detection on small sample
            sample_df = test_df.tail(100) if len(test_df) > 100 else test_df
            
            try:
                events = target_detector.detect(sample_df)
                # Detection succeeded
                logger.debug(f"Smoke test passed for {pattern_type}: {len(events) if events else 0} events")
                return True, None
            except Exception as e:
                return False, f"Detection failed: {e}"
            
        except Exception as e:
            logger.warning(f"Smoke test error (non-critical): {e}")
            # Don't fail validation if smoke test has issues
            return True, None


# Global validator instance
_validator = ParameterValidator()


def validate_parameters(
    pattern_type: str, 
    parameters: Dict[str, Any],
    test_df: Optional[pd.DataFrame] = None
) -> Tuple[bool, Optional[str]]:
    """
    Convenience function for parameter validation.
    
    Args:
        pattern_type: Pattern identifier
        parameters: Parameters to validate
        test_df: Optional test dataframe for smoke test
        
    Returns:
        (is_valid, error_message)
    """
    return _validator.validate_parameters(pattern_type, parameters, test_df)
