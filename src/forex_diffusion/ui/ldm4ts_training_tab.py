"""
LDM4TS Training Tab

Standalone tab for LDM4TS model training (moved from Forecast Settings).
"""

from PySide6.QtWidgets import QWidget
from loguru import logger

from .unified_prediction_settings_dialog import UnifiedPredictionSettingsDialog


class LDM4TSTrainingTab(QWidget):
    """
    LDM4TS Training Tab - Standalone training interface.
    
    This tab was moved from:
      Forecast Settings → Generative Forecast → LDM4TS Training
    To:
      Training → LDM4TS
      
    It reuses the implementation from UnifiedPredictionSettingsDialog._create_ldm4ts_training_tab()
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create a temporary dialog instance to reuse the tab creation method
        # This is a bit hacky but avoids code duplication
        temp_dialog = UnifiedPredictionSettingsDialog(parent)
        
        # Get the training tab widget
        training_widget = temp_dialog._create_ldm4ts_training_tab()
        
        # Copy the widget's layout to this tab
        from PySide6.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(training_widget)
        
        # Copy references to UI elements so they work
        # This allows the training methods to work correctly
        self._copy_ldm4ts_references(temp_dialog)
        
        logger.info("LDM4TS Training Tab initialized")
    
    def _copy_ldm4ts_references(self, source_dialog):
        """
        Copy LDM4TS training UI element references from source dialog.
        
        This ensures that training methods (like _start_ldm4ts_training) work correctly.
        Note: Only copies attributes that exist on source_dialog to avoid AttributeError.
        """
        # Helper to safely copy attribute
        def safe_copy(attr_name):
            if hasattr(source_dialog, attr_name):
                setattr(self, attr_name, getattr(source_dialog, attr_name))
                return True
            else:
                logger.warning(f"Attribute {attr_name} not found on source_dialog, skipping")
                return False
        
        # Data settings
        safe_copy('ldm4ts_train_symbol_combo')
        safe_copy('ldm4ts_train_timeframe_combo')
        safe_copy('ldm4ts_train_window_spinbox')
        safe_copy('ldm4ts_train_horizons_edit')
        
        # Model settings
        safe_copy('ldm4ts_train_diffusion_steps_spinbox')
        safe_copy('ldm4ts_train_image_size_combo')
        
        # Training settings
        safe_copy('ldm4ts_train_epochs_spinbox')
        safe_copy('ldm4ts_train_batch_size_spinbox')
        safe_copy('ldm4ts_train_lr_spinbox')
        safe_copy('ldm4ts_train_use_gpu_cb')
        
        # Memory optimization
        safe_copy('ldm4ts_train_attention_combo')
        safe_copy('ldm4ts_train_grad_checkpoint_cb')
        safe_copy('ldm4ts_vram_estimate_label')
        
        # Output
        safe_copy('ldm4ts_train_output_edit')
        # Note: browse button is local variable, not stored as attribute
        
        # Progress
        safe_copy('ldm4ts_train_progress_bar')
        safe_copy('ldm4ts_train_status_label')
        
        # Buttons
        safe_copy('ldm4ts_start_training_btn')
        safe_copy('ldm4ts_stop_training_btn')
        
        # Copy methods (bind to this instance) - only if they exist
        method_names = [
            '_start_ldm4ts_training',
            '_stop_ldm4ts_training',
            '_browse_ldm4ts_train_output',
            '_update_vram_estimate',
            '_on_training_progress',
            '_on_training_status',
            '_on_epoch_complete',
            '_on_training_complete',
            '_on_training_error'
        ]
        
        for method_name in method_names:
            if hasattr(source_dialog, method_name):
                method = getattr(source_dialog, method_name)
                setattr(self, method_name, method.__get__(self, type(self)))
            else:
                logger.warning(f"Method {method_name} not found on source_dialog, skipping")
        
        # Copy worker reference placeholder
        self._training_worker = None
        
        # Copy checkpoint edit reference if exists (for setting path after training)
        safe_copy('ldm4ts_checkpoint_edit')
