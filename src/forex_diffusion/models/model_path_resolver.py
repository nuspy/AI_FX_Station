"""
Unified Model Path Resolver for standardizing model selection across the application.

Consolidates the 3 different methods of specifying models into a single, clear system.
"""
from __future__ import annotations

import os
import sys
import re
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger


class ModelPathResolver:
    """
    Unified resolver for model paths from various sources.

    Handles:
    1. Single model path (legacy compatibility)
    2. Multiple model paths (UI multi-selection)
    3. Model paths from text area (one per line)

    Priority order:
    1. Multi-selection paths (highest priority)
    2. Text area paths
    3. Single model path (lowest priority, legacy)
    """

    def __init__(self):
        self.supported_extensions = {'.pt', '.pth', '.ckpt', '.pkl', '.pickle', '.joblib'}

    def resolve_model_paths(self, settings: Dict[str, Any]) -> List[str]:
        """
        Resolve model paths from settings with clear priority order.

        Args:
            settings: Settings dictionary from PredictionSettingsDialog

        Returns:
            List of validated, canonical model paths
        """
        raw_paths = self._collect_paths_by_priority(settings)
        normalized_paths = self._normalize_paths(raw_paths)
        validated_paths = self._validate_paths(normalized_paths)

        if validated_paths:
            logger.info(f"Resolved {len(validated_paths)} model paths")
            for i, path in enumerate(validated_paths):
                logger.debug(f"  {i+1}. {path}")
        else:
            logger.warning("No valid model paths found")

        return validated_paths

    def _collect_paths_by_priority(self, settings: Dict[str, Any]) -> List[str]:
        """Collect paths from all sources in priority order.

        Uses first non-empty source only (priority order):
        1. Multi-selection paths
        2. Text area paths
        3. Single model path
        """
        # Priority 1: Multi-selection paths (from dialog cache)
        try:
            from ..ui.prediction_settings_dialog import PredictionSettingsDialog
            multi_paths = PredictionSettingsDialog.get_model_paths()
            if multi_paths:
                logger.debug(f"Using {len(multi_paths)} paths from multi-selection (highest priority)")
                return multi_paths
        except Exception as e:
            logger.debug(f"Could not get multi-selection paths: {e}")

        # Priority 2: Text area paths (models_edit field)
        text_paths = self._parse_text_paths(settings.get("model_paths", []))
        if text_paths:
            logger.debug(f"Using {len(text_paths)} paths from text area")
            return text_paths

        # Priority 3: Single model path (legacy)
        single_path = settings.get("model_path")
        if single_path and single_path.strip():
            logger.debug("Using single model path (legacy)")
            return [single_path.strip()]

        return []

    def _parse_text_paths(self, text_input: Any) -> List[str]:
        """Parse paths from text input (supports various formats)."""
        if not text_input:
            return []

        paths = []

        if isinstance(text_input, (list, tuple)):
            # Already a list
            paths.extend([str(p).strip() for p in text_input if str(p).strip()])
        elif isinstance(text_input, str):
            # String with multiple paths
            text = text_input.strip()
            if not text:
                return []

            # Split by common separators
            if any(sep in text for sep in ['\n', ';', ',']):
                # Multi-line or delimited format
                for line in re.split(r'[\n;,]+', text):
                    line = line.strip()
                    if line:
                        paths.append(line)
            else:
                # Single path
                paths.append(text)

        return paths

    def _normalize_paths(self, raw_paths: List[str]) -> List[str]:
        """Normalize and canonicalize paths."""
        normalized = []
        seen = set()

        for raw_path in raw_paths:
            try:
                # Clean the path
                path = str(raw_path).strip().strip('"').strip("'")
                if not path:
                    continue

                # Expand variables and user home
                path = os.path.expandvars(os.path.expanduser(path))

                # Make absolute
                if not os.path.isabs(path):
                    path = os.path.abspath(path)

                # Canonicalize
                path = os.path.realpath(path)

                # Deduplication (case-insensitive on Windows)
                key = path.lower() if sys.platform.startswith("win") else path
                if key not in seen:
                    seen.add(key)
                    normalized.append(path)

            except Exception as e:
                logger.warning(f"Failed to normalize path '{raw_path}': {e}")

        return normalized

    def _validate_paths(self, paths: List[str]) -> List[str]:
        """Validate that paths exist and are model files."""
        validated = []

        for path in paths:
            try:
                path_obj = Path(path)

                # Check existence
                if not path_obj.exists():
                    logger.warning(f"Model file does not exist: {path}")
                    continue

                # Check if it's a file
                if not path_obj.is_file():
                    logger.warning(f"Path is not a file: {path}")
                    continue

                # Check extension
                if path_obj.suffix.lower() not in self.supported_extensions:
                    logger.warning(f"Unsupported model file extension: {path_obj.suffix} (path: {path})")
                    continue

                # Check file size (basic sanity check)
                if path_obj.stat().st_size == 0:
                    logger.warning(f"Model file is empty: {path}")
                    continue

                validated.append(path)
                logger.debug(f"Validated model path: {path}")

            except Exception as e:
                logger.warning(f"Failed to validate path '{path}': {e}")

        return validated

    def create_model_labels(self, model_paths: List[str]) -> Dict[str, str]:
        """
        Create unique labels for model paths for UI display.

        Args:
            model_paths: List of validated model paths

        Returns:
            Dictionary mapping path -> display label
        """
        if not model_paths:
            return {}

        labels = {}
        basename_counts = {}

        # Count occurrences of each basename
        for path in model_paths:
            basename = Path(path).stem  # Filename without extension
            basename_counts[basename] = basename_counts.get(basename, 0) + 1

        # Create unique labels
        basename_indices = {}
        for path in model_paths:
            basename = Path(path).stem

            if basename_counts[basename] == 1:
                # Unique basename
                labels[path] = basename
            else:
                # Multiple files with same basename - add index
                index = basename_indices.get(basename, 0) + 1
                basename_indices[basename] = index
                labels[path] = f"{basename}#{index}"

        return labels

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Get basic information about a model file.

        Args:
            model_path: Path to model file

        Returns:
            Dictionary with model information
        """
        try:
            path_obj = Path(model_path)
            info = {
                'path': model_path,
                'filename': path_obj.name,
                'basename': path_obj.stem,
                'extension': path_obj.suffix.lower(),
                'size_bytes': path_obj.stat().st_size,
                'size_mb': round(path_obj.stat().st_size / (1024 * 1024), 2),
                'modified': path_obj.stat().st_mtime,
                'exists': True,
                'readable': os.access(model_path, os.R_OK)
            }

            # Try to determine model type from extension
            if info['extension'] in {'.pt', '.pth'}:
                info['model_type'] = 'pytorch'
            elif info['extension'] in {'.pkl', '.pickle'}:
                info['model_type'] = 'sklearn'
            elif info['extension'] == '.joblib':
                info['model_type'] = 'sklearn_joblib'
            else:
                info['model_type'] = 'unknown'

            return info

        except Exception as e:
            return {
                'path': model_path,
                'exists': False,
                'error': str(e)
            }

    def validate_settings_consistency(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate consistency of model path settings.

        Returns:
            Validation results with warnings and recommendations
        """
        results = {
            'valid': True,
            'warnings': [],
            'recommendations': []
        }

        # Count non-empty sources
        sources = []

        try:
            from ..ui.prediction_settings_dialog import PredictionSettingsDialog
            multi_paths = PredictionSettingsDialog.get_model_paths()
            if multi_paths:
                sources.append('multi_selection')
        except Exception:
            pass

        text_paths = self._parse_text_paths(settings.get("model_paths", []))
        if text_paths:
            sources.append('text_area')

        single_path = settings.get("model_path")
        if single_path and single_path.strip():
            sources.append('single_path')

        # Check for conflicts
        if len(sources) > 1:
            results['warnings'].append(
                f"Multiple model path sources configured: {', '.join(sources)}. "
                f"Using priority order: multi_selection > text_area > single_path"
            )

        if not sources:
            results['valid'] = False
            results['recommendations'].append(
                "No model paths configured. Please select at least one model file."
            )

        return results