"""
Comment out hardcoded .setToolTip() calls in training_tab.py
This allows i18n tooltips to be visible.
"""

import re
from pathlib import Path

file_path = Path("D:/Projects/ForexGPT/src/forex_diffusion/ui/training_tab.py")

print(f"Processing {file_path}...")

content = file_path.read_text(encoding='utf-8')
original = content

# Pattern to find .setToolTip(...) calls
# We'll comment them out by adding # before the line

lines = content.split('\n')
modified_lines = []
in_tooltip = False
tooltip_indent = 0

for i, line in enumerate(lines):
    # Check if this line contains .setToolTip(
    if '.setToolTip(' in line and 'apply_tooltip' not in line:
        # Start of tooltip
        in_tooltip = True
        tooltip_indent = len(line) - len(line.lstrip())
        modified_lines.append('        # ' + line.lstrip() + '  # i18n: tooltip set in _apply_i18n_tooltips()')
        
        # Check if tooltip closes on same line
        if line.count('(') == line.count(')'):
            in_tooltip = False
    elif in_tooltip:
        # Inside multi-line tooltip
        current_indent = len(line) - len(line.lstrip())
        
        # Check if we're still in the tooltip
        if line.strip().endswith(')') and current_indent <= tooltip_indent + 4:
            # End of tooltip
            modified_lines.append('        # ' + line.lstrip())
            in_tooltip = False
        else:
            # Continue tooltip
            modified_lines.append('        # ' + line.lstrip())
    else:
        # Normal line
        modified_lines.append(line)

content = '\n'.join(modified_lines)

if content != original:
    file_path.write_text(content, encoding='utf-8')
    print(f"[OK] Commented out hardcoded tooltips in {file_path.name}")
    print(f"Modified {content.count('# i18n: tooltip')} tooltip locations")
else:
    print(f"No changes needed")
