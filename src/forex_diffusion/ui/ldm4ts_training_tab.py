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
        """
        # Data settings
        self.ldm4ts_train_symbol_combo = source_dialog.ldm4ts_train_symbol_combo
        self.ldm4ts_train_timeframe_combo = source_dialog.ldm4ts_train_timeframe_combo
        self.ldm4ts_train_window_spinbox = source_dialog.ldm4ts_train_window_spinbox
        self.ldm4ts_train_horizons_edit = source_dialog.ldm4ts_train_horizons_edit
        
        # Model settings
        self.ldm4ts_train_diffusion_steps_spinbox = source_dialog.ldm4ts_train_diffusion_steps_spinbox
        self.ldm4ts_train_image_size_combo = source_dialog.ldm4ts_train_image_size_combo
        
        # Training settings
        self.ldm4ts_train_epochs_spinbox = source_dialog.ldm4ts_train_epochs_spinbox
        self.ldm4ts_train_batch_size_spinbox = source_dialog.ldm4ts_train_batch_size_spinbox
        self.ldm4ts_train_lr_spinbox = source_dialog.ldm4ts_train_lr_spinbox
        self.ldm4ts_train_use_gpu_cb = source_dialog.ldm4ts_train_use_gpu_cb
        
        # Memory optimization
        self.ldm4ts_train_attention_combo = source_dialog.ldm4ts_train_attention_combo
        self.ldm4ts_train_grad_checkpoint_cb = source_dialog.ldm4ts_train_grad_checkpoint_cb
        self.ldm4ts_vram_estimate_label = source_dialog.ldm4ts_vram_estimate_label
        
        # Output
        self.ldm4ts_train_output_edit = source_dialog.ldm4ts_train_output_edit
        self.ldm4ts_train_browse_output_btn = source_dialog.ldm4ts_train_browse_output_btn
        
        # Progress
        self.ldm4ts_train_progress_bar = source_dialog.ldm4ts_train_progress_bar
        self.ldm4ts_train_status_label = source_dialog.ldm4ts_train_status_label
        
        # Buttons
        self.ldm4ts_start_training_btn = source_dialog.ldm4ts_start_training_btn
        self.ldm4ts_stop_training_btn = source_dialog.ldm4ts_stop_training_btn
        
        # Copy methods (bind to this instance)
        self._start_ldm4ts_training = source_dialog._start_ldm4ts_training.__get__(self, type(self))
        self._stop_ldm4ts_training = source_dialog._stop_ldm4ts_training.__get__(self, type(self))
        self._browse_ldm4ts_train_output = source_dialog._browse_ldm4ts_train_output.__get__(self, type(self))
        self._update_vram_estimate = source_dialog._update_vram_estimate.__get__(self, type(self))
        self._on_training_progress = source_dialog._on_training_progress.__get__(self, type(self))
        self._on_training_status = source_dialog._on_training_status.__get__(self, type(self))
        self._on_epoch_complete = source_dialog._on_epoch_complete.__get__(self, type(self))
        self._on_training_complete = source_dialog._on_training_complete.__get__(self, type(self))
        self._on_training_error = source_dialog._on_training_error.__get__(self, type(self))
        
        # Copy worker reference placeholder
        self._training_worker = None
        
        # Copy checkpoint edit reference if exists (for setting path after training)
        if hasattr(source_dialog, 'ldm4ts_checkpoint_edit'):
            self.ldm4ts_checkpoint_edit = source_dialog.ldm4ts_checkpoint_edit
