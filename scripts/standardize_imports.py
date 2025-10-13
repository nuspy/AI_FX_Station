"""
Standardize import patterns across pattern detection modules.

PEP 8 ordering:
1. Future imports
2. Standard library (alphabetical)
3. Third-party (alphabetical) 
4. Local application/library (alphabetical, prefer relative)
"""
import os
import re
from pathlib import Path
from typing import List, Tuple


def parse_imports(content: str) -> Tuple[List[str], List[str], List[str], List[str], str]:
    """
    Parse imports from file content.
    
    Returns:
        (future_imports, stdlib_imports, third_party_imports, local_imports, rest_of_code)
    """
    lines = content.split('\n')
    
    future = []
    stdlib = []
    third_party = []
    local = []
    rest_start = 0
    
    # Standard library modules (common ones)
    stdlib_modules = {
        'os', 'sys', 'time', 'json', 'datetime', 'collections', 'typing', 
        'dataclasses', 'enum', 'pathlib', 'asyncio', 'hashlib', 're',
        'concurrent', 'functools', 'itertools', 'copy'
    }
    
    # Third-party modules
    third_party_modules = {
        'numpy', 'pandas', 'loguru', 'sqlalchemy', 'pydantic', 'fastapi',
        'pytest', 'matplotlib', 'scipy', 'sklearn', 'optuna', 'psutil'
    }
    
    in_imports = True
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines and docstrings at the top
        if not stripped or stripped.startswith('"""') or stripped.startswith("'''"):
            if in_imports and i < 20:  # Allow docstrings in first 20 lines
                continue
        
        # Future imports
        if stripped.startswith('from __future__'):
            future.append(line)
            continue
        
        # Import statements
        if stripped.startswith('import ') or stripped.startswith('from '):
            # Determine import type
            if stripped.startswith('from .') or stripped.startswith('from ..'):
                local.append(line)
            else:
                # Extract module name
                if stripped.startswith('import '):
                    module = stripped.split()[1].split('.')[0]
                else:  # from X import Y
                    module = stripped.split()[1].split('.')[0]
                
                if module in stdlib_modules:
                    stdlib.append(line)
                elif module in third_party_modules:
                    third_party.append(line)
                else:
                    # Guess: if starts with capital letter, likely third-party
                    if module[0].isupper():
                        third_party.append(line)
                    else:
                        # Default to local
                        local.append(line)
            continue
        
        # Non-import line found
        if stripped and not stripped.startswith('#'):
            in_imports = False
            rest_start = i
            break
    
    rest_of_code = '\n'.join(lines[rest_start:])
    
    return future, stdlib, third_party, local, rest_of_code


def standardize_file_imports(filepath: Path) -> bool:
    """
    Standardize imports in a single file.
    
    Returns:
        True if file was modified
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if already has "# Imports standardized" comment
        if '# Imports standardized' in content:
            return False
        
        # Parse imports
        future, stdlib, third_party, local, rest = parse_imports(content)
        
        # Rebuild with standardized order
        new_content_parts = []
        
        # Extract docstring if present
        docstring_match = re.match(r'^(""".*?"""|\'\'\'.*?\'\'\')\s*', content, re.DOTALL)
        if docstring_match:
            new_content_parts.append(docstring_match.group(1))
            new_content_parts.append('')
        
        # Add imports in order
        if future:
            new_content_parts.extend(future)
            new_content_parts.append('')
        
        if stdlib:
            new_content_parts.extend(sorted(stdlib))
            new_content_parts.append('')
        
        if third_party:
            new_content_parts.extend(sorted(third_party))
            new_content_parts.append('')
        
        if local:
            new_content_parts.extend(sorted(local))
            new_content_parts.append('')
        
        # Add rest of code
        new_content_parts.append(rest)
        
        new_content = '\n'.join(new_content_parts)
        
        # Only write if changed
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Standardize imports in pattern detection modules"""
    # Target directories
    pattern_dir = Path(__file__).parent.parent / 'src' / 'forex_diffusion' / 'patterns'
    
    if not pattern_dir.exists():
        print(f"Pattern directory not found: {pattern_dir}")
        return
    
    # Get all Python files
    py_files = list(pattern_dir.glob('*.py'))
    
    print(f"Found {len(py_files)} Python files in {pattern_dir}")
    
    modified = 0
    for filepath in py_files:
        if filepath.name.startswith('_') and filepath.name != '__init__.py':
            continue  # Skip private files except __init__
        
        if standardize_file_imports(filepath):
            print(f"✓ Standardized: {filepath.name}")
            modified += 1
        else:
            print(f"  Skipped: {filepath.name}")
    
    print(f"\nModified {modified}/{len(py_files)} files")


if __name__ == '__main__':
    main()
