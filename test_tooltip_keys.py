"""Test that tooltip keys match between code and JSON"""
from src.forex_diffusion.i18n import tr

print("Testing tooltip key resolution...\n")

# Test keys used in training_tab.py _apply_i18n_tooltips()
test_keys = [
    ("training.model_name.tooltip", "Model Name"),
    ("training.symbol.tooltip", "Symbol"),
    ("training.timeframe.tooltip", "Timeframe"),
    ("training.days.tooltip", "Days"),
    ("training.horizon.tooltip", "Horizon"),
    ("training.model.tooltip", "Model"),
    ("training.encoder.tooltip", "Encoder"),
    ("training.use_gpu_training.tooltip", "GPU Training"),
    ("training.optimization.tooltip", "Optimization"),
    ("training.genetic.generations.tooltip", "Generations"),
    ("training.genetic.population.tooltip", "Population"),
    ("training.indicators.atr.tooltip", "ATR Indicator"),
    ("training.features.returns_volatility.tooltip", "Returns & Volatility"),
    ("training.advanced.warmup_bars.tooltip", "Warmup Bars"),
    ("training.lightgbm.epochs.tooltip", "LightGBM Epochs"),
    ("training.encoder.patch_len.tooltip", "Encoder Patch Length"),
    ("training.diffusion.timesteps.tooltip", "Diffusion Timesteps"),
    ("training.nvidia.enable.tooltip", "NVIDIA GPU Enable"),
]

print("Testing key resolution:")
print("-" * 80)

found = 0
not_found = 0

for key, description in test_keys:
    result = tr(key, default="NOT_FOUND")
    if result == "NOT_FOUND" or result == key:
        print(f"[MISSING] {description:30s} -> {key}")
        not_found += 1
    else:
        # Show first 60 chars
        preview = result[:60] + "..." if len(result) > 60 else result
        print(f"[OK]      {description:30s} -> {preview}")
        found += 1

print("-" * 80)
print(f"\nResults: {found} found, {not_found} missing")

if not_found > 0:
    print("\nDEBUG: Checking JSON structure...")
    from src.forex_diffusion.i18n import _translations
    import json
    
    # Load translations
    tr("training.symbol.label")  # Force load
    
    if "en_US" in _translations:
        data = _translations["en_US"]
        print(f"\nTop-level keys in JSON: {list(data.keys())[:10]}")
        
        if "training" in data:
            print(f"Keys under 'training': {list(data['training'].keys())[:10]}")
            
            if "symbol" in data["training"]:
                print(f"Keys under 'training.symbol': {list(data['training']['symbol'].keys())}")
