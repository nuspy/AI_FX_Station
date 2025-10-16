"""
i18n Widget Helper

Automatically applies i18n labels and tooltips to Qt widgets.
"""

from typing import Optional, Dict
from PySide6.QtWidgets import QWidget, QLabel, QCheckBox, QPushButton, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit
from PySide6.QtCore import QObject
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


def apply_tooltips_to_tab(tab_widget: QWidget, mappings: Dict[str, tuple]) -> None:
    """
    Apply tooltips to all widgets in a tab using object name mappings.
    
    Args:
        tab_widget: Tab widget to process
        mappings: Dict mapping object names to (category, key) tuples
        
    Example:
        mappings = {
            "symbol_combo": ("training", "symbol"),
            "days_spin": ("training", "days"),
        }
        apply_tooltips_to_tab(training_tab, mappings)
    """
    def apply_recursive(widget: QWidget):
        # Check object name
        obj_name = widget.objectName()
        if obj_name and obj_name in mappings:
            category, key = mappings[obj_name]
            apply_tooltip(widget, key, category)
        
        # Process children
        for child in widget.findChildren(QObject):
            if isinstance(child, QWidget):
                apply_recursive(child)
    
    apply_recursive(tab_widget)


def auto_apply_tooltips(parent_widget: QWidget, category: str = "training") -> None:
    """
    Automatically apply tooltips to all widgets with objectName set.
    Assumes objectName matches i18n key.
    
    Args:
        parent_widget: Parent widget to scan
        category: Category for all tooltips
        
    Example:
        # Widget created with objectName
        combo = QComboBox()
        combo.setObjectName("symbol")  # Will use training.symbol.tooltip
        
        # Apply tooltips
        auto_apply_tooltips(training_tab, "training")
    """
    def apply_recursive(widget: QWidget):
        obj_name = widget.objectName()
        if obj_name and not obj_name.startswith("qt_"):  # Skip Qt internal widgets
            # Try to apply tooltip
            tooltip_key = f"{category}.{obj_name}.tooltip"
            tooltip_text = tr(tooltip_key, default=None)
            if tooltip_text and tooltip_text != tooltip_key:  # Check if translation exists
                widget.setToolTip(tooltip_text)
        
        # Process children
        for child in widget.findChildren(QWidget):
            apply_recursive(child)
    
    apply_recursive(parent_widget)
