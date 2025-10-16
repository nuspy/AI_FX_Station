"""
Resetta lo stato salvato degli splitter per permettere al SentimentPanel di essere visibile.

Se il pannello non appare, esegui questo script prima di avviare ForexGPT.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger
from forex_diffusion.utils.user_settings import set_setting, get_setting

def reset_splitter_states():
    """Resetta gli stati salvati degli splitter."""
    logger.info("=" * 80)
    logger.info("RESET STATI SPLITTER")
    logger.info("=" * 80)
    
    # Settings to reset
    settings_to_reset = [
        'chart.right_splitter_state',
        'chart.left_splitter_state',
        'chart.main_splitter_state',
    ]
    
    for setting_key in settings_to_reset:
        old_value = get_setting(setting_key)
        if old_value:
            logger.info(f"  Resetting {setting_key}: {len(old_value)} bytes")
            set_setting(setting_key, None)
            logger.success(f"  ✓ {setting_key} resettato")
        else:
            logger.info(f"  {setting_key}: già None")
    
    logger.info("\n" + "=" * 80)
    logger.success("✓ Stati splitter resettati!")
    logger.info("=" * 80)
    logger.info("\nAlla prossima apertura di ForexGPT, gli splitter useranno le dimensioni di default.")
    logger.info("Il SentimentPanel sarà visibile con altezza 250px.")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")
    reset_splitter_states()
