"""
Remove all hardcoded .setToolTip() calls from UI files
while preserving i18n apply_tooltip() calls
"""

from pathlib import Path
import re

ui_dir = Path("D:/Projects/ForexGPT/src/forex_diffusion/ui")

# Files with i18n integration
integrated_files = [
    "training_tab.py",
    "backtesting_tab.py",
    "pattern_training_tab.py",
    "live_trading_tab.py",
    "portfolio_tab.py",
    "settings_dialog.py",
    "signals_tab.py",
    "logs_tab.py",
    "sentiment_panel.py",
    "news_calendar_tab.py",
    "regime_analysis_tab.py",
    "correlation_matrix_widget.py",
    "training_queue_tab.py",
    "training_history_tab.py",
    "signal_quality_tab.py",
    "reports_3d_tab.py",
    "parameter_adaptation_tab.py",
    "admin_login_dialog.py",
    "pretrade_calc_dialog.py",
    "trade_dialog.py",
    "indicators_dialog.py",
    "color_settings_dialog.py",
    "checkpoint_selector_dialog.py",
    "fxpro_credentials_dialog.py",
]

def remove_hardcoded_tooltips(file_path):
    """Remove hardcoded .setToolTip() while keeping apply_tooltip"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content
        
        lines = content.split('\n')
        result = []
        skip_lines = 0
        
        for line in lines:
            # If we're in skip mode (multi-line tooltip)
            if skip_lines > 0:
                skip_lines -= 1
                continue
            
            # Check if line contains .setToolTip but NOT apply_tooltip
            if '.setToolTip(' in line and 'apply_tooltip' not in line:
                # Count parentheses to see if tooltip spans multiple lines
                open_count = line.count('(')
                close_count = line.count(')')
                
                if open_count > close_count:
                    # Multi-line tooltip - skip until balanced
                    skip_lines = 0
                    temp_line = line
                    while open_count > close_count:
                        skip_lines += 1
                        if skip_lines >= 20:  # Safety limit
                            break
                        next_idx = len(result) + skip_lines
                        if next_idx < len(lines):
                            temp_line = lines[next_idx]
                            open_count += temp_line.count('(')
                            close_count += temp_line.count(')')
                
                # Don't add this line (remove tooltip)
                continue
            
            result.append(line)
        
        new_content = '\n'.join(result)
        
        if new_content != original:
            file_path.write_text(new_content, encoding='utf-8')
            removed = original.count('.setToolTip(') - new_content.count('.setToolTip(')
            if removed > 0:
                print(f"[OK] {file_path.name:40s} - removed {removed:3d} hardcoded tooltips")
                return removed
            else:
                print(f"[SKIP] {file_path.name:40s} - no changes needed")
                return 0
        else:
            print(f"[SKIP] {file_path.name:40s} - already clean")
            return 0
            
    except Exception as e:
        print(f"[ERROR] {file_path.name:40s} - {e}")
        return 0

# Process all integrated files
print("Removing hardcoded tooltips from i18n-integrated files...")
print("=" * 80)

total_removed = 0
for filename in integrated_files:
    file_path = ui_dir / filename
    if file_path.exists():
        removed = remove_hardcoded_tooltips(file_path)
        total_removed += removed
    else:
        print(f"[MISSING] {filename:40s} - file not found")

print("=" * 80)
print(f"Total hardcoded tooltips removed: {total_removed}")
print("\nNow all i18n tooltips should be visible in the app!")
