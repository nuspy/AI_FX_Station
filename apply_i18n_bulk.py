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
    "pattern_training_tab.py": {
        "category": "pattern_training",
        "widgets": [
            ("pattern_type_combo", "pattern_type"),
            ("min_pattern_bars_spin", "min_pattern_bars"),
            ("pattern_tolerance_spin", "pattern_tolerance"),
            ("min_pattern_samples_spin", "min_pattern_samples"),
            ("pattern_features_edit", "pattern_features"),
        ]
    },
    "news_calendar_tab.py": {
        "category": "news_calendar",
        "widgets": [
            ("impact_filter_combo", "impact_filter"),
            ("currency_filter_edit", "currency_filter"),
            ("auto_disable_trading_check", "auto_disable_trading"),
            ("news_lookback_hours_spin", "news_lookback_hours"),
            ("news_lookahead_hours_spin", "news_lookahead_hours"),
        ]
    },
    "regime_analysis_tab.py": {
        "category": "regime_analysis",
        "widgets": [
            ("regime_window_spin", "regime_window"),
            ("trend_threshold_spin", "trend_threshold"),
            ("volatility_threshold_spin", "volatility_threshold"),
            ("regime_stability_spin", "regime_stability"),
            ("adaptive_strategy_check", "adaptive_strategy"),
        ]
    },
    "correlation_matrix_widget.py": {
        "category": "correlation_matrix",
        "widgets": [
            ("correlation_window_spin", "correlation_window"),
            ("correlation_method_combo", "correlation_method"),
            ("highlight_threshold_spin", "highlight_threshold"),
            ("auto_diversify_check", "auto_diversify"),
        ]
    },
    "training_queue_tab.py": {
        "category": "training_queue",
        "widgets": [
            ("max_concurrent_jobs_spin", "max_concurrent_jobs"),
            ("auto_retry_failed_check", "auto_retry_failed"),
            ("priority_queue_check", "priority_queue"),
            ("save_failed_configs_check", "save_failed_configs"),
        ]
    },
    "training_history_tab.py": {
        "category": "training_history",
        "widgets": [
            ("history_retention_days_spin", "history_retention_days"),
            ("compare_runs_btn", "compare_runs"),
            ("export_tensorboard_btn", "export_tensorboard"),
        ]
    },
    "signal_quality_tab.py": {
        "category": "signal_quality",
        "widgets": [
            ("min_win_rate_spin", "min_win_rate"),
            ("min_profit_factor_spin", "min_profit_factor"),
            ("min_sharpe_ratio_spin", "min_sharpe_ratio"),
            ("track_signal_performance_check", "track_signal_performance"),
            ("auto_disable_poor_signals_check", "auto_disable_poor_signals"),
        ]
    },
    "reports_3d_tab.py": {
        "category": "reports_3d",
        "widgets": [
            ("plot_type_combo", "plot_type"),
            ("x_parameter_combo", "x_parameter"),
            ("y_parameter_combo", "y_parameter"),
            ("z_metric_combo", "z_metric"),
            ("interpolation_check", "interpolation"),
        ]
    },
    "parameter_adaptation_tab.py": {
        "category": "parameter_adaptation",
        "widgets": [
            ("enable_adaptation_check", "enable_adaptation"),
            ("adaptation_frequency_combo", "adaptation_frequency"),
            ("adaptation_metric_combo", "adaptation_metric"),
            ("adaptation_method_combo", "adaptation_method"),
        ]
    },
    "settings_dialog.py": {
        "category": "settings",
        "widgets": [
            ("log_level_combo", "log_level"),
            ("save_artifacts_check", "save_artifacts"),
            ("cache_data_check", "cache_data"),
            ("parallel_workers_spin", "parallel_workers"),
            ("random_seed_spin", "random_seed"),
        ]
    },
    "admin_login_dialog.py": {
        "category": "admin",
        "widgets": [
            ("reset_database_btn", "reset_database"),
            ("export_config_btn", "export_config"),
            ("import_config_btn", "import_config"),
            ("system_diagnostics_btn", "system_diagnostics"),
            ("clear_cache_btn", "clear_cache"),
        ]
    },
    "pretrade_calc_dialog.py": {
        "category": "pretrade_validation",
        "widgets": [
            ("enable_validation_check", "enable_validation"),
            ("max_trade_size_spin", "max_trade_size"),
            ("require_confirmation_check", "require_confirmation"),
            ("check_spread_check", "check_spread"),
            ("check_margin_check", "check_margin"),
        ]
    },
    "trade_dialog.py": {
        "category": "trade_execution",
        "widgets": [
            ("execution_speed_combo", "execution_speed"),
            ("slippage_tolerance_spin", "slippage_tolerance"),
            ("partial_fills_check", "partial_fills"),
            ("smart_routing_check", "smart_routing"),
            ("iceberg_orders_check", "iceberg_orders"),
        ]
    },
    "indicators_dialog.py": {
        "category": "indicators_config",
        "widgets": [
            ("save_preset_btn", "save_indicator_preset"),
            ("load_preset_btn", "load_indicator_preset"),
            ("multi_timeframe_check", "multi_timeframe_indicators"),
            ("indicator_alerts_check", "indicator_alerts"),
        ]
    },
    "color_settings_dialog.py": {
        "category": "color_themes",
        "widgets": [
            ("theme_combo", "theme_selection"),
            ("bullish_color_btn", "bullish_color"),
            ("bearish_color_btn", "bearish_color"),
            ("grid_opacity_spin", "grid_opacity"),
            ("custom_theme_import_btn", "custom_theme_import"),
        ]
    },
    "checkpoint_selector_dialog.py": {
        "category": "checkpoint_management",
        "widgets": [
            ("auto_checkpoint_check", "auto_checkpoint"),
            ("checkpoint_frequency_spin", "checkpoint_frequency"),
            ("keep_best_only_check", "keep_best_only"),
            ("checkpoint_compression_check", "checkpoint_compression"),
        ]
    },
    "fxpro_credentials_dialog.py": {
        "category": "fxpro_integration",
        "widgets": [
            ("fxpro_account_edit", "fxpro_account"),
            ("fxpro_server_combo", "fxpro_server"),
            ("auto_sync_check", "auto_sync"),
            ("webhook_url_edit", "webhook_url"),
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
