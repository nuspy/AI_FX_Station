# GPU Acceleration Plan for ForexGPT

## Current Status

### ✅ Already Available
- **PyTorch 2.8.0** installed (CPU-only version)
- **TensorFlow 2.13.0** installed
- Torch models already support device selection in diffusion.py
- Config has `device: "auto"` parameter

### ❌ Missing
- **CUDA-enabled PyTorch** (currently CPU-only)
- **GPU acceleration for sklearn models** (Ridge, RF, Lasso, ElasticNet)
- **cuML library** (RAPIDS - GPU sklearn alternative)
- **UI checkboxes** for GPU selection

## Problem Analysis

### 1. Training Pipeline (`train_sklearn.py`)
**Current**: Uses CPU-only sklearn models
- Ridge, Lasso, ElasticNet → Linear models (fast on CPU)
- RandomForest → Can be slow with large datasets
- PCA → Can benefit from GPU
- Autoencoder/VAE → Already uses PyTorch (but CPU-only)

**GPU Potential**:
- ⭐⭐⭐ **High**: Autoencoder/VAE training (PyTorch models)
- ⭐⭐ **Medium**: RandomForest with large datasets
- ⭐ **Low**: Ridge/Lasso/ElasticNet (already fast on CPU)

### 2. Inference Pipeline (`parallel_inference.py`, `forecast_worker.py`)
**Current**: CPU-only inference
- PyTorch models use CPU
- sklearn models use CPU

**GPU Potential**:
- ⭐⭐⭐ **High**: PyTorch model inference (Autoencoder/VAE forward pass)
- ⭐⭐ **Medium**: Batch inference with multiple models
- ⭐ **Low**: Single sklearn model prediction

## Solution: 2-Phase Implementation

---

## Phase 1: Enable PyTorch GPU Support (IMMEDIATE) ⚡

### Step 1.1: Install CUDA PyTorch
```bash
# Uninstall CPU-only version
pip uninstall torch torchvision torchaudio

# Install CUDA version (CUDA 11.8 or 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 1.2: Add GPU Device Selection Utility
**File**: `src/forex_diffusion/utils/device_manager.py` (NEW)

```python
import torch
from typing import Optional, Literal

DeviceType = Literal["auto", "cuda", "cpu", "mps"]

class DeviceManager:
    """Centralized GPU/CPU device management."""

    @staticmethod
    def get_device(preference: DeviceType = "auto") -> torch.device:
        """Get optimal device based on preference and availability."""
        if preference == "cuda" or preference == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
        if preference == "mps":  # Apple Silicon
            if torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def get_device_info() -> dict:
        """Get device information for UI display."""
        return {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "mps_available": torch.backends.mps.is_available(),
            "cpu_count": torch.get_num_threads()
        }
```

### Step 1.3: Update Encoder Training to Use GPU
**File**: `src/forex_diffusion/training/encoders.py`

Current encoders (Autoencoder, VAE) need device support:
```python
class SklearnAutoencoder:
    def __init__(self, ..., device: str = "auto"):
        self.device = DeviceManager.get_device(device)
        self.model = AutoencoderNet(...).to(self.device)

    def fit(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        # ... rest of training
```

### Step 1.4: Update Inference to Use GPU
**Files**:
- `src/forex_diffusion/inference/parallel_inference.py:112`
- `src/forex_diffusion/ui/workers/forecast_worker.py`

```python
# In ModelExecutor.predict()
def predict(self, features_df: pd.DataFrame, use_gpu: bool = False) -> Dict:
    device = DeviceManager.get_device("cuda" if use_gpu else "cpu")

    if hasattr(model, 'eval'):  # PyTorch model
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            t_in = torch.tensor(X_last, dtype=torch.float32).to(device)
            out = model(t_in)
            predictions = out.detach().cpu().numpy()
```

### Step 1.5: Add UI Checkbox for GPU
**Training UI** (`training_tab.py`):
```python
self.use_gpu_checkbox = QCheckBox("Usa GPU (se disponibile)")
self.use_gpu_checkbox.setChecked(False)
self.use_gpu_checkbox.setEnabled(torch.cuda.is_available())
layout.addWidget(self.use_gpu_checkbox)
```

**Forecast UI** (`forecast_tab.py`):
```python
self.use_gpu_inference_checkbox = QCheckBox("Inferenza GPU")
self.use_gpu_inference_checkbox.setChecked(False)
self.use_gpu_inference_checkbox.setEnabled(torch.cuda.is_available())
```

---

## Phase 2: Add cuML for sklearn Models (OPTIONAL) 🚀

### Why cuML?
- Drop-in replacement for sklearn on GPU
- **100-200x faster** for RandomForest on large datasets
- Same API as sklearn

### Installation
```bash
# Install RAPIDS cuML (requires CUDA 11.x or 12.x)
pip install cuml-cu12  # for CUDA 12.x
# or
pip install cuml-cu11  # for CUDA 11.x
```

### Implementation
**File**: `src/forex_diffusion/training/gpu_models.py` (NEW)

```python
from typing import Literal
import numpy as np

def get_regressor(algo: str, use_gpu: bool = False, **kwargs):
    """Get regressor with optional GPU acceleration."""

    if use_gpu:
        try:
            import cuml
            if algo == "ridge":
                return cuml.Ridge(**kwargs)
            elif algo == "lasso":
                return cuml.Lasso(**kwargs)
            elif algo == "elasticnet":
                return cuml.ElasticNet(**kwargs)
            elif algo == "rf":
                return cuml.RandomForestRegressor(**kwargs)
        except ImportError:
            print("cuML not available, falling back to CPU sklearn")

    # Fallback to sklearn
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor

    if algo == "ridge":
        return Ridge(**kwargs)
    elif algo == "lasso":
        return Lasso(**kwargs)
    elif algo == "elasticnet":
        return ElasticNet(**kwargs)
    elif algo == "rf":
        return RandomForestRegressor(**kwargs)
```

**Update `train_sklearn.py`**:
```python
from .gpu_models import get_regressor

# In main()
use_gpu = args.use_gpu and torch.cuda.is_available()
model = get_regressor(args.algo, use_gpu=use_gpu, ...)
```

---

## Performance Expectations

### Training Speedup (with GPU)
| Model | CPU Time | GPU Time | Speedup |
|-------|----------|----------|---------|
| Ridge/Lasso/ElasticNet | 1-5s | 0.5-2s | **2-3x** |
| RandomForest (n=100, 10K samples) | 30-60s | 1-3s | **20-30x** |
| Autoencoder/VAE (100 epochs) | 5-10min | 30-60s | **10-15x** |
| PCA (cuML) | 2-5s | 0.2-0.5s | **10x** |

### Inference Speedup (with GPU)
| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Single PyTorch model | 5-10ms | 1-2ms | **5x** |
| Batch 10 models (parallel) | 50-100ms | 10-20ms | **5-8x** |
| Ensemble 50 models | 250-500ms | 30-60ms | **8-10x** |

---

## Implementation Checklist

### Phase 1: PyTorch GPU (IMMEDIATE - 2 hours)
- [ ] Install CUDA PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- [ ] Create `device_manager.py` utility
- [ ] Update `encoders.py` to use GPU device
- [ ] Update `parallel_inference.py` to use GPU device
- [ ] Add GPU checkbox in Training UI
- [ ] Add GPU checkbox in Forecast UI
- [ ] Test with CUDA-enabled GPU
- [ ] Add GPU info display in UI (device name, VRAM)

### Phase 2: cuML sklearn GPU (OPTIONAL - 1 hour)
- [ ] Install cuML: `pip install cuml-cu12`
- [ ] Create `gpu_models.py` wrapper
- [ ] Update `train_sklearn.py` to use GPU models
- [ ] Add automatic fallback to CPU if cuML unavailable
- [ ] Test RandomForest GPU training
- [ ] Benchmark speedup

---

## Hardware Requirements

### Minimum for PyTorch GPU:
- NVIDIA GPU with CUDA Compute Capability 3.5+
- 4GB+ VRAM
- CUDA 11.8 or 12.1 drivers

### Recommended for cuML:
- NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal or newer)
- 8GB+ VRAM
- CUDA 12.x drivers

---

## Migration Strategy

### 1. **Immediate Action** (Today)
Install CUDA PyTorch and add GPU checkboxes. No breaking changes.

### 2. **Test Phase** (Tomorrow)
Test GPU training with existing models. Verify speedup.

### 3. **Optional Enhancement** (Next Week)
Add cuML for massive RandomForest speedup if needed.

### 4. **Backwards Compatibility**
- Always provide CPU fallback
- GPU is optional, not required
- Existing models work unchanged

---

## Code Changes Summary

### New Files
1. `src/forex_diffusion/utils/device_manager.py` - GPU device management
2. `src/forex_diffusion/training/gpu_models.py` - cuML wrappers (Phase 2)

### Modified Files
1. `src/forex_diffusion/training/encoders.py` - Add device parameter
2. `src/forex_diffusion/training/train_sklearn.py` - Add --use-gpu flag
3. `src/forex_diffusion/inference/parallel_inference.py` - GPU inference
4. `src/forex_diffusion/ui/tabs/training_tab.py` - GPU checkbox
5. `src/forex_diffusion/ui/tabs/forecast_tab.py` - GPU checkbox
6. `pyproject.toml` - Update torch to CUDA version

### Dependencies to Add
```toml
# pyproject.toml
dependencies = [
    ...
    # GPU Support (Phase 1)
    "torch>=2.8.0+cu121",  # CUDA 12.1

    # Optional GPU sklearn (Phase 2)
    # "cuml-cu12>=24.0.0",  # Uncomment if GPU available
]
```

---

## Next Steps

**Vuoi che proceda con**:
1. ✅ **Phase 1 subito** - Install CUDA PyTorch e add GPU checkboxes (2 ore)?
2. ⏸️ **Phase 2 dopo** - cuML per sklearn GPU (opzionale, se hai NVIDIA GPU)?
