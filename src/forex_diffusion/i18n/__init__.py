"""
Internationalization (i18n) module for ForexGPT.

Provides translation support for all UI elements including labels, buttons,
tooltips, and error messages across multiple languages.

Supported languages:
- en_US: English (United States) - Default
- it_IT: Italian (Italy)
- es_ES: Spanish (Spain)
- fr_FR: French (France)
- de_DE: German (Germany)
- ja_JP: Japanese (Japan)
- zh_CN: Chinese (Simplified)

Usage:
    from forex_diffusion.i18n import tr, set_language, get_available_languages
    
    # Translate a string
    text = tr("training.symbol.label")
    
    # Get tooltip with all 6 sections
    tooltip = tr("training.symbol.tooltip")
    
    # Change language
    set_language("it_IT")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

# Current language (default English)
_current_language = "en_US"

# Translation cache: {language: {key: value}}
_translations: Dict[str, Dict[str, str]] = {}

# Path to translation files
_TRANSLATIONS_DIR = Path(__file__).parent / "translations"


def load_translations(language: str) -> Dict[str, str]:
    """
    Load translations for a specific language from JSON file.
    
    Args:
        language: Language code (e.g., "en_US", "it_IT")
    
    Returns:
        Dictionary of translations {key: translated_text}
    """
    translation_file = _TRANSLATIONS_DIR / f"{language}.json"
    
    if not translation_file.exists():
        logger.warning(f"Translation file not found: {translation_file}, falling back to en_US")
        translation_file = _TRANSLATIONS_DIR / "en_US.json"
    
    try:
        with open(translation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load translations for {language}: {e}")
        return {}


def get_available_languages() -> list[str]:
    """
    Get list of available language codes.
    
    Returns:
        List of language codes (e.g., ["en_US", "it_IT", "es_ES"])
    """
    if not _TRANSLATIONS_DIR.exists():
        return ["en_US"]
    
    languages = []
    for file in _TRANSLATIONS_DIR.glob("*.json"):
        languages.append(file.stem)
    
    return sorted(languages)


def set_language(language: str) -> bool:
    """
    Set the current UI language.
    
    Args:
        language: Language code (e.g., "en_US", "it_IT")
    
    Returns:
        True if language was set successfully, False otherwise
    """
    global _current_language, _translations
    
    if language not in _translations:
        translations = load_translations(language)
        if not translations:
            logger.error(f"Failed to set language to {language}")
            return False
        _translations[language] = translations
    
    _current_language = language
    logger.info(f"Language set to: {language}")
    return True


def get_current_language() -> str:
    """Get the currently active language code."""
    return _current_language


def tr(key: str, **kwargs) -> str:
    """
    Translate a key to the current language.
    
    Args:
        key: Translation key in dot notation (e.g., "training.symbol.label")
        **kwargs: Optional format arguments for string interpolation
    
    Returns:
        Translated string, or the key itself if translation not found
    
    Examples:
        >>> tr("training.symbol.label")
        "Symbol:"
        
        >>> tr("training.days.tooltip")
        "1) COSA È:\\nNumero di giorni storici..."
        
        >>> tr("error.file_not_found", filename="data.csv")
        "File not found: data.csv"
    """
    global _current_language, _translations
    
    # Load translations if not cached
    if _current_language not in _translations:
        _translations[_current_language] = load_translations(_current_language)
    
    # Get translation
    translation = _translations[_current_language].get(key)
    
    if translation is None:
        # Fallback to English if current language doesn't have the key
        if _current_language != "en_US":
            if "en_US" not in _translations:
                _translations["en_US"] = load_translations("en_US")
            translation = _translations["en_US"].get(key)
        
        if translation is None:
            logger.warning(f"Translation not found for key: {key}")
            return key
    
    # Apply string formatting if kwargs provided
    if kwargs:
        try:
            translation = translation.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing format argument for key {key}: {e}")
    
    return translation


def tr_tooltip(base_key: str) -> str:
    """
    Get a complete tooltip with all 6 sections.
    
    This is a convenience function that constructs the full tooltip key
    from a base key (e.g., "training.symbol" -> "training.symbol.tooltip")
    
    Args:
        base_key: Base translation key without ".tooltip" suffix
    
    Returns:
        Complete tooltip text with all 6 sections formatted
    
    Example:
        >>> tr_tooltip("training.symbol")
        "Symbol Selection\\n\\n1) COSA È:\\n..."
    """
    return tr(f"{base_key}.tooltip")


# Initialize with English on module import
set_language("en_US")
