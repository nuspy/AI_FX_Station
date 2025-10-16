"""
Script to automatically migrate hardcoded tooltips to i18n system.
Replaces .setToolTip() calls with tr() function calls.
"""

import re
from pathlib import Path

# Mapping from label text to i18n key
LABEL_TO_KEY = {
    # Training Tab
    "Model Name:": ("training", "model_name"),
    "Symbol:": ("training", "symbol"),
    "Base TF:": ("training", "timeframe"),
    "Days:": ("training", "days"),
    "Horizon:": ("training", "horizon"),
    "Model:": ("training", "model"),
    "Encoder:": ("training", "encoder"),
    "Usa GPU": ("training", "use_gpu_training"),
    "Opt:": ("training", "optimization"),
    "Gen:": ("training", "generations"),
    "Pop:": ("training", "population"),
    
    # Indicators
    "ATR": ("training.indicators", "atr"),
    "RSI": ("training.indicators", "rsi"),
    "MACD": ("training.indicators", "macd"),
    "Bollinger": ("training.indicators", "bollinger"),
    "Stochastic": ("training.indicators", "stochastic"),
    "CCI": ("training.indicators", "cci"),
    "Williams%R": ("training.indicators", "williams_r"),
    "ADX": ("training.indicators", "adx"),
    "MFI": ("training.indicators", "mfi"),
    "OBV": ("training.indicators", "obv"),
    "TRIX": ("training.indicators", "trix"),
    "Ultimate": ("training.indicators", "ultimate"),
    "Donchian": ("training.indicators", "donchian"),
    "Keltner": ("training.indicators", "keltner"),
    "EMA": ("training.indicators", "ema"),
    "SMA": ("training.indicators", "sma"),
    "Hurst": ("training.indicators", "hurst"),
    "VWAP": ("training.indicators", "vwap"),
    
    # Feature Engineering
    "Returns & Volatility": ("training.features", "returns_volatility"),
    "Trading Sessions": ("training.features", "trading_sessions"),
    "Candlestick Patterns": ("training.features", "candlestick_patterns"),
    "Volume Profile": ("training.features", "volume_profile"),
    "VSA (Volume Spread Analysis)": ("training.features", "vsa"),
    
    # Advanced Parameters
    "Warmup bars:": ("training.advanced", "warmup_bars"),
    "RV window:": ("training.advanced", "rv_window"),
    "Min coverage:": ("training.advanced", "min_coverage"),
    "ATR period:": ("training.advanced", "atr_n"),
    "RSI period:": ("training.advanced", "rsi_n"),
    "Bollinger period:": ("training.advanced", "bb_n"),
    "Hurst window:": ("training.advanced", "hurst_window"),
    
    # LightGBM
    "Lightning epochs:": ("training.lightgbm", "epochs"),
    "Lightning batch:": ("training.lightgbm", "batch"),
    "Lightning val_frac:": ("training.lightgbm", "validation_fraction"),
    
    # Encoder
    "Encoder latent dim:": ("training.encoder", "latent_dim"),
    "Encoder epochs:": ("training.encoder", "epochs"),
    "Lightning patch:": ("training.encoder", "patch_len"),
    
    # Diffusion
    "Diffusion timesteps:": ("training.diffusion", "timesteps"),
    "Learning rate:": ("training.diffusion", "learning_rate"),
    "Batch size DL:": ("training.diffusion", "batch_size"),
    "Model channels:": ("training.diffusion", "model_channels"),
    "Dropout:": ("training.diffusion", "dropout"),
    "Num heads:": ("training.diffusion", "num_heads"),
    
    # NVIDIA GPU
    "Enable NVIDIA Stack": ("training.nvidia", "enable"),
    "Use AMP": ("training.nvidia", "use_amp"),
    "Precision:": ("training.nvidia", "precision"),
    "Compile Model": ("training.nvidia", "compile_model"),
    "Fused Optimizer": ("training.nvidia", "fused_optimizer"),
    "Flash Attention": ("training.nvidia", "flash_attention"),
    "Grad Accumulation:": ("training.nvidia", "grad_accumulation_steps"),
}


def migrate_file(file_path: Path) -> bool:
    """
    Migrate a single Python file to use i18n tooltips.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        True if file was modified
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Add i18n import if not present
        if 'from ..i18n import tr' not in content and 'from ...i18n import tr' not in content:
            # Find imports section
            import_match = re.search(r'(from .+? import .+?\n)+', content)
            if import_match:
                imports_end = import_match.end()
                # Add i18n import
                if 'from ..utils' in content:
                    import_line = "from ..i18n import tr\n"
                else:
                    import_line = "from ...i18n import tr\n"
                content = content[:imports_end] + import_line + content[imports_end:]
        
        # Replace .setToolTip() with tr() calls
        # Pattern: widget.setToolTip("long hardcoded text...")
        pattern = r'\.setToolTip\(\s*["\'](.+?)["\']\s*\)'
        
        def replace_tooltip(match):
            tooltip_text = match.group(1)
            # Try to find corresponding i18n key
            # This is a simple heuristic - may need manual review
            return '.setToolTip(tr("training.REPLACE_ME.tooltip"))'
        
        # For now, just add comment markers for manual review
        content = re.sub(
            r'(\w+)\.setToolTip\(\s*["\']((?:[^"\'\\]|\\.)+)["\']\s*\)',
            lambda m: f'{m.group(1)}.setToolTip(tr("FIXME.tooltip"))  # Original: {m.group(2)[:50]}...',
            content,
            flags=re.MULTILINE | re.DOTALL
        )
        
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"✓ Modified: {file_path}")
            return True
        else:
            print(f"  Skipped: {file_path} (no changes)")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False


def main():
    """Main migration function"""
    ui_dir = Path(__file__).parent.parent / "src" / "forex_diffusion" / "ui"
    
    if not ui_dir.exists():
        print(f"Error: UI directory not found: {ui_dir}")
        return
    
    print(f"Scanning {ui_dir}...\n")
    
    modified_count = 0
    for py_file in ui_dir.glob("*.py"):
        if migrate_file(py_file):
            modified_count += 1
    
    print(f"\n✓ Migration complete: {modified_count} files modified")
    print("\nNOTE: Files contain 'FIXME.tooltip' markers that need manual review")
    print("Replace FIXME with actual i18n key from translations/en_US.json")


if __name__ == "__main__":
    main()
