"""
i18n Widget Helper

Automatically applies i18n labels and tooltips to Qt widgets.
"""

from typing import Optional
from PySide6.QtWidgets import QWidget, QLabel, QCheckBox, QPushButton, QGroupBox
from . import tr


def apply_i18n(widget: QWidget, key: str, category: str = "training") -> None:
    """
    Apply i18n label and tooltip to a widget.
    
    Args:
        widget: Qt widget to apply i18n to
        key: Translation key (e.g., "model_name", "atr")
        category: Category in translations (e.g., "training", "backtesting")
    """
    full_key = f"{category}.{key}"
    
    # Get label and tooltip from translations
    label_text = tr(f"{full_key}.label", default=None)
    tooltip_text = tr(f"{full_key}.tooltip", default=None)
    
    # Apply label if widget supports it
    if label_text and isinstance(widget, (QLabel, QCheckBox, QPushButton, QGroupBox)):
        if hasattr(widget, 'setText'):
            widget.setText(label_text)
    
    # Apply tooltip
    if tooltip_text:
        widget.setToolTip(tooltip_text)


def create_label(key: str, category: str = "training") -> QLabel:
    """
    Create a QLabel with i18n label and tooltip.
    
    Args:
        key: Translation key
        category: Category in translations
        
    Returns:
        QLabel with label and tooltip from translations
    """
    full_key = f"{category}.{key}"
    label_text = tr(f"{full_key}.label", default=key)
    tooltip_text = tr(f"{full_key}.tooltip", default="")
    
    label = QLabel(label_text)
    if tooltip_text:
        label.setToolTip(tooltip_text)
    
    return label


def apply_tooltip(widget: QWidget, key: str, category: str = "training") -> None:
    """
    Apply only tooltip to a widget (useful when label is already set).
    
    Args:
        widget: Qt widget
        key: Translation key
        category: Category in translations
    """
    full_key = f"{category}.{key}"
    tooltip_text = tr(f"{full_key}.tooltip", default=None)
    
    if tooltip_text:
        widget.setToolTip(tooltip_text)
