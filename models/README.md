# Pre-trained Models Directory

This directory contains pre-trained models used by ForexGPT LDM4TS.

## Required Models

### 1. Stable Diffusion VAE
**Location:** `models/vae/`

**Model:** `stabilityai/sd-vae-ft-mse`

**Files required:**
```
models/vae/
├── config.json
├── diffusion_pytorch_model.bin (or .safetensors)
└── ... (other model files)
```

**Download from Hugging Face:**
```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download VAE model
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='stabilityai/sd-vae-ft-mse',
    local_dir='models/vae',
    local_dir_use_symlinks=False
)
"
```

**Or manually:**
1. Visit: https://huggingface.co/stabilityai/sd-vae-ft-mse
2. Download all files to `models/vae/`

---

### 2. CLIP Text Encoder
**Location:** `models/clip/`

**Model:** `openai/clip-vit-base-patch32`

**Files required:**
```
models/clip/
├── config.json
├── pytorch_model.bin (or model.safetensors)
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── ... (other model files)
```

**Download from Hugging Face:**
```bash
# Download CLIP model
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='openai/clip-vit-base-patch32',
    local_dir='models/clip',
    local_dir_use_symlinks=False
)
"
```

**Or manually:**
1. Visit: https://huggingface.co/openai/clip-vit-base-patch32
2. Download all files to `models/clip/`

---

## Quick Setup Script

Create a file `download_models.py` in the project root:

```python
#!/usr/bin/env python3
"""Download pre-trained models for LDM4TS"""
from huggingface_hub import snapshot_download
from pathlib import Path

models_dir = Path(__file__).parent / "models"
models_dir.mkdir(exist_ok=True)

print("Downloading Stable Diffusion VAE...")
snapshot_download(
    repo_id='stabilityai/sd-vae-ft-mse',
    local_dir=str(models_dir / 'vae'),
    local_dir_use_symlinks=False
)
print("✓ VAE downloaded")

print("\nDownloading CLIP model...")
snapshot_download(
    repo_id='openai/clip-vit-base-patch32',
    local_dir=str(models_dir / 'clip'),
    local_dir_use_symlinks=False
)
print("✓ CLIP downloaded")

print("\n✓ All models downloaded successfully!")
```

Then run:
```bash
python download_models.py
```

---

## Verification

Check that models are correctly installed:

```python
from pathlib import Path

project_root = Path(__file__).parent
vae_path = project_root / "models" / "vae" / "config.json"
clip_path = project_root / "models" / "clip" / "config.json"

print(f"VAE config exists: {vae_path.exists()}")
print(f"CLIP config exists: {clip_path.exists()}")

if vae_path.exists() and clip_path.exists():
    print("✓ All models are correctly installed!")
else:
    print("✗ Some models are missing. Please download them.")
```

---

## Distribution Package

When distributing ForexGPT, include these models in the package:

```
ForexGPT/
├── models/
│   ├── vae/
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model.safetensors
│   │   └── ...
│   └── clip/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── vocab.json
│       ├── merges.txt
│       └── ...
├── src/
└── ...
```

**Model Sizes:**
- VAE: ~330 MB
- CLIP: ~470 MB
- **Total:** ~800 MB

---

## Troubleshooting

### "VAE model not found" Error

```
FileNotFoundError: VAE model not found at D:\Projects\ForexGPT\models\vae
```

**Solution:**
1. Run `python download_models.py` to download models
2. Or manually download from Hugging Face
3. Ensure `models/vae/config.json` exists

### "CLIP model not found" Error

```
FileNotFoundError: CLIP model not found at D:\Projects\ForexGPT\models\clip
```

**Solution:**
1. Run `python download_models.py` to download models
2. Or manually download from Hugging Face
3. Ensure `models/clip/config.json` exists

### Models in wrong location

The code expects models at:
- `<project_root>/models/vae/`
- `<project_root>/models/clip/`

Where `<project_root>` is 4 levels up from the model files:
```
<project_root>/src/forex_diffusion/models/ldm4ts_vae.py
```

---

## License Notes

**Stable Diffusion VAE:**
- License: CreativeML Open RAIL-M
- Commercial use allowed with restrictions
- See: https://huggingface.co/stabilityai/sd-vae-ft-mse

**CLIP:**
- License: MIT
- Commercial use allowed
- See: https://huggingface.co/openai/clip-vit-base-patch32

**Always verify license compliance for your use case.**
