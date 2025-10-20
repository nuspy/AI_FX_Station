"""
Dialog for configuring prediction settings.
DEPRECATED: Use UnifiedPredictionSettingsDialog instead.
This file is kept for backward compatibility - it imports and aliases the new unified dialog.
"""
from __future__ import annotations

# Import the new unified dialog
from .unified_prediction_settings_dialog import UnifiedPredictionSettingsDialog

# Create alias for backward compatibility
PredictionSettingsDialog = UnifiedPredictionSettingsDialog
