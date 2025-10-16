"""
Bulk apply i18n tooltips to remaining tabs.
Adds _apply_i18n_tooltips() method to each tab file.
"""

from pathlib import Path

# Tab files to process with their tooltip mappings
TAB_CONFIGS = {
    "data_sources_tab.py": {
        "category": "data_sources",
        "widgets": [
            ("primary_source_combo", "primary_source"),
            ("backup_source_combo", "backup_source"),
            ("api_credentials_edit", "api_credentials"),
            ("rate_limit_spin", "rate_limit"),
            ("download_on_startup_check", "download_on_startup"),
            ("data_storage_edit", "data_storage"),
        ]
    },
    "logs_tab.py": {
        "category": "logs",
        "widgets": [
            ("log_filter_combo", "log_filter"),
            ("auto_scroll_check", "auto_scroll"),
            ("save_logs_check", "save_logs"),
            ("max_log_lines_spin", "max_log_lines"),
        ]
    },
    "sentiment_panel.py": {
        "category": "trading_intelligence.sentiment",
        "widgets": [
            ("enable_sentiment_check", "enable_sentiment"),
            ("sentiment_sources_combo", "sentiment_sources"),
            ("sentiment_weight_spin", "sentiment_weight"),
            ("sentiment_lookback_spin", "sentiment_lookback"),
        ]
    },
}


def generate_tooltip_method(category: str, widgets: list) -> str:
    """Generate _apply_i18n_tooltips() method code"""
    lines = [
        "    def _apply_i18n_tooltips(self):",
        '        """Apply i18n tooltips to all widgets"""',
        "        from ..i18n.widget_helper import apply_tooltip",
        "        ",
    ]
    
    for widget_name, key in widgets:
        lines.extend([
            f"        if hasattr(self, '{widget_name}'):",
            f'            apply_tooltip(self.{widget_name}, "{key}", "{category}")',
        ])
    
    return "\n".join(lines)


def apply_to_file(file_path: Path, category: str, widgets: list):
    """Add i18n tooltips to a tab file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Check if already applied
        if "_apply_i18n_tooltips" in content:
            print(f"  Skipped (already has method): {file_path.name}")
            return False
        
        # Find __init__ method end
        init_end = content.find("def __init__")
        if init_end == -1:
            print(f"  Skipped (no __init__): {file_path.name}")
            return False
        
        # Find first method after __init__
        next_def = content.find("\n    def ", init_end + 20)
        if next_def == -1:
            # Add at end of class
            next_def = len(content)
        
        # Insert method before next def
        method_code = "\n" + generate_tooltip_method(category, widgets) + "\n    \n"
        new_content = content[:next_def] + method_code + content[next_def:]
        
        # Add call in __init__
        # Find end of __init__ (before first "def " after it)
        init_start = content.find("def __init__", init_end)
        init_body_end = content.find("\n    def ", init_start + 20)
        if init_body_end == -1:
            init_body_end = len(content)
        
        # Find last line of __init__
        init_section = new_content[init_start:init_body_end]
        last_indent_line = init_section.rfind("\n        ")
        if last_indent_line != -1:
            insert_pos = init_start + last_indent_line + len("\n        ")
            # Find end of that line
            line_end = new_content.find("\n", insert_pos)
            if line_end != -1:
                # Insert after this line
                new_content = (
                    new_content[:line_end] +
                    "\n        \n        # Apply i18n tooltips\n        self._apply_i18n_tooltips()" +
                    new_content[line_end:]
                )
        
        file_path.write_text(new_content, encoding='utf-8')
        print(f"[OK] Modified: {file_path.name}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error processing {file_path.name}: {e}")
        return False


def main():
    ui_dir = Path(__file__).parent / "src" / "forex_diffusion" / "ui"
    
    if not ui_dir.exists():
        print(f"Error: UI directory not found: {ui_dir}")
        return
    
    print("Applying i18n tooltips to remaining tabs...\n")
    
    modified_count = 0
    for filename, config in TAB_CONFIGS.items():
        file_path = ui_dir / filename
        if not file_path.exists():
            print(f"  Skipped (not found): {filename}")
            continue
        
        if apply_to_file(file_path, config["category"], config["widgets"]):
            modified_count += 1
    
    print(f"\n[COMPLETE] {modified_count} files modified")


if __name__ == "__main__":
    main()
