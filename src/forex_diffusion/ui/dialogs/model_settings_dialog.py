"""
Model Settings Auto-Configuration Dialog

Shows model metadata and prompts user to auto-apply inference settings.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QGroupBox
)
from typing import Dict, Any


class ModelSettingsDialog(QDialog):
    """
    Dialog to show model metadata and ask if user wants to auto-configure.
    """
    
    def __init__(self, metadata: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.metadata = metadata
        self.accepted_settings = False
        self.setup_ui()
        
    def setup_ui(self):
        """Setup UI components."""
        self.setWindowTitle("Model Metadata Detected")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("📊 Model Training Settings Detected")
        title_font = title.font()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Info text
        info = QLabel(
            "This model was trained with specific settings.\n"
            "Would you like to automatically apply these settings for optimal inference?"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Metadata display
        meta_group = QGroupBox("Model Training Configuration")
        meta_layout = QVBoxLayout()
        
        meta_text = self._format_metadata()
        meta_display = QTextEdit()
        meta_display.setPlainText(meta_text)
        meta_display.setReadOnly(True)
        meta_display.setMaximumHeight(250)
        meta_layout.addWidget(meta_display)
        
        meta_group.setLayout(meta_layout)
        layout.addWidget(meta_group)
        
        # Recommendations
        rec_group = QGroupBox("Recommended Inference Settings")
        rec_layout = QVBoxLayout()
        
        rec_text = self._format_recommendations()
        rec_display = QTextEdit()
        rec_display.setPlainText(rec_text)
        rec_display.setReadOnly(True)
        rec_display.setMaximumHeight(150)
        rec_layout.addWidget(rec_display)
        
        rec_group.setLayout(rec_layout)
        layout.addWidget(rec_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        apply_btn = QPushButton("✓ Auto-Apply Settings")
        apply_btn.clicked.connect(self.on_accept)
        apply_btn.setDefault(True)
        
        manual_btn = QPushButton("✗ Configure Manually")
        manual_btn.clicked.connect(self.on_reject)
        
        button_layout.addWidget(manual_btn)
        button_layout.addStretch()
        button_layout.addWidget(apply_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _format_metadata(self) -> str:
        """Format metadata for display."""
        lines = []
        lines.append(f"Model Type: {self.metadata.get('model_type', 'unknown').upper()}")
        
        if self.metadata.get('model_type') == 'sklearn':
            lines.append(f"Algorithm: {self.metadata.get('algorithm', 'unknown')}")
            lines.append(f"Encoder: {self.metadata.get('encoder_type', 'none')}")
            lines.append(f"Features: {self.metadata.get('num_features', 0)} features")
            if self.metadata.get('val_mae'):
                lines.append(f"Validation MAE: {self.metadata['val_mae']:.6f}")
        
        lines.append(f"\nSymbol: {self.metadata.get('symbol', 'EUR/USD')}")
        lines.append(f"Timeframe: {self.metadata.get('timeframe', '1m')}")
        
        horizons = self.metadata.get('horizons', [60])
        lines.append(f"\nHorizons: {horizons}")
        lines.append(f"  - Number of horizons: {len(horizons)}")
        lines.append(f"  - Multi-horizon: {'Yes' if len(horizons) > 1 else 'No'}")
        
        if len(horizons) > 1:
            lines.append("\n  Horizon Details:")
            tf = self.metadata.get('timeframe', '1m')
            for h in horizons:
                time_str = self._bars_to_time(h, tf)
                lines.append(f"    • {h} bars = {time_str}")
        
        return '\n'.join(lines)
    
    def _format_recommendations(self) -> str:
        """Format recommendations for display."""
        lines = []
        lines.append("For optimal accuracy, use the following settings:")
        lines.append("")
        
        lines.append(f"Symbol: {self.metadata.get('symbol', 'EUR/USD')}")
        lines.append(f"Timeframe: {self.metadata.get('timeframe', '1m')}")
        
        horizons = self.metadata.get('horizons', [60])
        horizon_str = ','.join(map(str, horizons))
        lines.append(f"Horizons: {horizon_str}")
        
        if self.metadata.get('model_type') == 'lightning':
            lines.append("Samples: 50 (recommended for diffusion models)")
        
        lines.append("")
        lines.append("⚠️ Using different settings may reduce accuracy or fail.")
        
        return '\n'.join(lines)
    
    def _bars_to_time(self, bars: int, timeframe: str) -> str:
        """Convert bars to human-readable time."""
        tf_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        minutes = bars * tf_minutes.get(timeframe, 1)
        
        if minutes < 60:
            return f"{minutes} minutes"
        elif minutes < 1440:
            hours = minutes / 60
            return f"{hours:.1f} hours"
        else:
            days = minutes / 1440
            return f"{days:.1f} days"
    
    def on_accept(self):
        """User accepted auto-apply."""
        self.accepted_settings = True
        self.accept()
    
    def on_reject(self):
        """User chose manual configuration."""
        self.accepted_settings = False
        self.reject()


class InferenceCompatibilityDialog(QDialog):
    """
    Dialog to show compatibility warnings and interpolation info.
    """
    
    def __init__(
        self,
        warnings: list,
        errors: list,
        interpolation_plan: dict = None,
        parent=None
    ):
        super().__init__(parent)
        self.warnings = warnings
        self.errors = errors
        self.interpolation_plan = interpolation_plan
        self.setup_ui()
        
    def setup_ui(self):
        """Setup UI components."""
        has_errors = len(self.errors) > 0
        
        self.setWindowTitle("Inference Compatibility Check")
        self.setMinimumWidth(650)
        
        layout = QVBoxLayout()
        
        # Icon and title
        if has_errors:
            title = QLabel("❌ Incompatible Settings Detected")
            icon_color = "#d32f2f"
        elif self.warnings:
            title = QLabel("⚠️ Compatibility Warnings")
            icon_color = "#f57c00"
        else:
            title = QLabel("✓ Settings Compatible")
            icon_color = "#388e3c"
        
        title_font = title.font()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {icon_color};")
        layout.addWidget(title)
        
        # Errors (if any)
        if self.errors:
            error_group = QGroupBox("❌ Critical Incompatibilities")
            error_layout = QVBoxLayout()
            
            error_text = '\n\n'.join(self.errors)
            error_display = QTextEdit()
            error_display.setPlainText(error_text)
            error_display.setReadOnly(True)
            error_display.setStyleSheet("background-color: #ffebee; color: #c62828;")
            error_layout.addWidget(error_display)
            
            error_group.setLayout(error_layout)
            layout.addWidget(error_group)
        
        # Warnings (if any)
        if self.warnings:
            warning_group = QGroupBox("⚠️ Warnings")
            warning_layout = QVBoxLayout()
            
            warning_text = '\n\n'.join(self.warnings)
            warning_display = QTextEdit()
            warning_display.setPlainText(warning_text)
            warning_display.setReadOnly(True)
            warning_display.setStyleSheet("background-color: #fff3e0; color: #e65100;")
            warning_display.setMaximumHeight(200)
            warning_layout.addWidget(warning_display)
            
            warning_group.setLayout(warning_layout)
            layout.addWidget(warning_group)
        
        # Interpolation plan (if applicable)
        if self.interpolation_plan:
            interp_group = QGroupBox("🔧 Interpolation Plan")
            interp_layout = QVBoxLayout()
            
            interp_text = self._format_interpolation_plan()
            interp_display = QTextEdit()
            interp_display.setPlainText(interp_text)
            interp_display.setReadOnly(True)
            interp_display.setMaximumHeight(150)
            interp_display.setStyleSheet("background-color: #e3f2fd; font-family: monospace;")
            interp_layout.addWidget(interp_display)
            
            interp_group.setLayout(interp_layout)
            layout.addWidget(interp_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        if has_errors:
            # Only allow close if errors
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.reject)
            button_layout.addStretch()
            button_layout.addWidget(close_btn)
        else:
            # Allow proceed if only warnings
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(self.reject)
            
            proceed_btn = QPushButton("Proceed with Interpolation" if self.interpolation_plan else "Proceed")
            proceed_btn.clicked.connect(self.accept)
            proceed_btn.setDefault(True)
            
            button_layout.addWidget(cancel_btn)
            button_layout.addStretch()
            button_layout.addWidget(proceed_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _format_interpolation_plan(self) -> str:
        """Format interpolation plan for display."""
        if not self.interpolation_plan:
            return "No interpolation needed."
        
        lines = []
        lines.append("The following horizons will be INTERPOLATED:")
        lines.append("")
        
        for horizon, (lower, upper, weight) in sorted(self.interpolation_plan.items()):
            lines.append(f"Horizon {horizon}:")
            lines.append(f"  └─ Interpolate between {lower} and {upper}")
            lines.append(f"     Weight: {weight:.3f} (closer to {lower if weight < 0.5 else upper})")
            lines.append(f"     Formula: pred_{horizon} = pred_{lower} × {1-weight:.3f} + pred_{upper} × {weight:.3f}")
            lines.append("")
        
        lines.append("⚠️ Note: Interpolation is less accurate than trained horizons.")
        lines.append("   Recommended: Re-train model with desired horizons for best results.")
        
        return '\n'.join(lines)
