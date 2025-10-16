"""
Fix training_tab.py syntax errors by removing orphan string literals
left after tooltip removal
"""

from pathlib import Path

file_path = Path("D:/Projects/ForexGPT/src/forex_diffusion/ui/training_tab.py")

print(f"Reading {file_path}...")
content = file_path.read_text(encoding='utf-8')
lines = content.split('\n')

# Remove orphan string literals (lines that are ONLY strings, no assignments/calls)
output = []
in_orphan_string = False
orphan_indent = 0

for i, line in enumerate(lines):
    stripped = line.strip()
    
    # Skip empty lines or comment-only lines
    if not stripped or stripped.startswith('#'):
        output.append(line)
        continue
    
    # Check if this is an orphan string literal
    # (starts with quote, contains only strings, no = or ( except in quotes)
    if stripped.startswith('"') or stripped.startswith("'"):
        # Check if it's part of an assignment or function call
        # Look at previous non-empty line
        prev_line = ''
        for j in range(i-1, max(0, i-20), -1):
            if lines[j].strip():
                prev_line = lines[j].strip()
                break
        
        # If previous line doesn't end with = or (, and current line is just a string
        # then it's likely an orphan tooltip string
        is_orphan = (
            not prev_line.endswith('=') and
            not prev_line.endswith('(') and
            not '=' in line and
            not 'def ' in line and
            not 'class ' in line and
            not 'import ' in line and
            not 'return ' in line
        )
        
        if is_orphan:
            # Skip this line (orphan string)
            continue
    
    output.append(line)

# Write fixed content
file_path.write_text('\n'.join(output), encoding='utf-8')
print(f"[OK] Fixed {len(lines) - len(output)} orphan string lines")
print(f"Original: {len(lines)} lines → Fixed: {len(output)} lines")
