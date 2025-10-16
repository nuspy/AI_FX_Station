"""
Remove hardcoded .setToolTip() calls from UI files.
This allows i18n tooltips to take effect.
"""

import re
from pathlib import Path

def remove_hardcoded_tooltips(file_path: Path) -> bool:
    """Remove .setToolTip() calls from a file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content
        
        # Pattern to match .setToolTip("..." or '''...''') including multiline
        # We'll do multiple passes to handle different cases
        
        # Pattern 1: Simple single-line tooltips
        # widget.setToolTip("text")
        content = re.sub(
            r'\.setToolTip\(\s*["\']([^"\']*?)["\']\s*\)',
            '',  # Remove completely
            content
        )
        
        # Pattern 2: Multi-line tooltips with triple quotes
        # widget.setToolTip("""...""") or widget.setToolTip('''...''')
        content = re.sub(
            r'\.setToolTip\(\s*["\']["\']["\'].*?["\']["\']["\'][^\)]*\)',
            '',
            content,
            flags=re.DOTALL
        )
        
        # Pattern 3: Multi-line tooltips with parentheses and newlines
        # widget.setToolTip(
        #     "line1\n"
        #     "line2"
        # )
        content = re.sub(
            r'\.setToolTip\(\s*\n\s*["\'].*?["\'].*?\)',
            '',
            content,
            flags=re.DOTALL
        )
        
        # Pattern 4: f-strings
        content = re.sub(
            r'\.setToolTip\(\s*f["\'].*?["\']\s*\)',
            '',
            content,
            flags=re.DOTALL
        )
        
        if content != original:
            file_path.write_text(content, encoding='utf-8')
            print(f"[OK] Cleaned: {file_path.name}")
            return True
        else:
            print(f"  No tooltips found: {file_path.name}")
            return False
            
    except Exception as e:
        print(f"[ERROR] {file_path.name}: {e}")
        return False

def main():
    ui_dir = Path(__file__).parent / "src" / "forex_diffusion" / "ui"
    
    if not ui_dir.exists():
        print(f"Error: UI directory not found: {ui_dir}")
        return
    
    print("Removing hardcoded tooltips from UI files...\n")
    
    # Process only tab files we integrated
    tab_files = [
        "training_tab.py",
        "backtesting_tab.py",
        "live_trading_tab.py",
        "portfolio_tab.py",
        "signals_tab.py",
        "pattern_training_tab.py",
        "news_calendar_tab.py",
        "regime_analysis_tab.py",
        "training_queue_tab.py",
        "training_history_tab.py",
        "signal_quality_tab.py",
        "reports_3d_tab.py",
        "parameter_adaptation_tab.py",
        "logs_tab.py",
        "sentiment_panel.py",
        "data_sources_tab.py",
        "correlation_matrix_widget.py",
        "settings_dialog.py",
        "admin_login_dialog.py",
        "pretrade_calc_dialog.py",
        "trade_dialog.py",
        "indicators_dialog.py",
        "color_settings_dialog.py",
        "checkpoint_selector_dialog.py",
        "fxpro_credentials_dialog.py",
    ]
    
    modified_count = 0
    for filename in tab_files:
        file_path = ui_dir / filename
        if file_path.exists():
            if remove_hardcoded_tooltips(file_path):
                modified_count += 1
    
    print(f"\n[COMPLETE] {modified_count} files cleaned")
    print("\nNote: This removes ALL .setToolTip() calls.")
    print("The i18n system in _apply_i18n_tooltips() will now provide tooltips.")

if __name__ == "__main__":
    main()
