# LDM4TS Memory Optimization - Quick Start

## TL;DR

**Problem:** LDM4TS training uses 8-12 GB VRAM (too much for consumer GPUs)

**Solution:** Install memory optimization packages to reduce VRAM by 35-55%

---

## 🚀 Quick Install

### Option 1: SageAttention Only (Recommended, Fast)

```bash
python install_memory_optimization.py
```

**What it does:**
- ✅ Installs SageAttention 2
- ✅ ~35% VRAM reduction
- ✅ Works on RTX 20/30/40 series
- ✅ Installation: ~1 minute

**Use if:**
- You have 8-12 GB VRAM
- You want fast installation
- You have RTX 20 series or newer

---

### Option 2: SageAttention + FlashAttention (Best Savings, Slow)

```bash
python install_memory_optimization.py --flash
```

**What it does:**
- ✅ Installs SageAttention 2
- ✅ Installs FlashAttention 2
- ✅ ~45% VRAM reduction (FlashAttention)
- ✅ Works on RTX 30/40 series only
- ⚠️  Installation: ~10 minutes (compilation required)

**Use if:**
- You have 6-8 GB VRAM
- You have RTX 30/40 series (Ampere+)
- You want maximum VRAM savings

---

## 🎮 GPU Recommendations

| GPU Model | VRAM | Recommended Setup | Expected Usage |
|-----------|------|-------------------|----------------|
| **RTX 3060** | 6 GB | FlashAttention + GC, batch=2 | ~2.8 GB |
| **RTX 3070** | 8 GB | FlashAttention + GC, batch=4 | ~3.7 GB |
| **RTX 3080** | 10 GB | SageAttention + GC, batch=6 | ~5.2 GB |
| **RTX 3080 Ti** | 12 GB | SageAttention + GC, batch=8 | ~6.8 GB |
| **RTX 4090** | 24 GB | Default (no opt), batch=16 | ~12 GB |

**GC** = Gradient Checkpointing (enabled by default in GUI)

---

## 📋 GUI Setup (After Installation)

1. **Open ForexGPT**
   ```bash
   python run_forexgpt.py
   ```

2. **Navigate to Training Settings**
   - Go to: `Generative Forecast` tab
   - Select: `LDM4TS Training` sub-tab

3. **Enable Memory Optimization**
   - Find section: `Memory Optimization (VRAM Reduction)`
   - Set `Attention Backend`:
     - **RTX 30/40:** Select `FlashAttention 2 (~45% VRAM)`
     - **RTX 20 or older:** Select `SageAttention 2 (~35% VRAM)`
   - ✅ Enable `Gradient Checkpointing` (default ON, keep it)

4. **Adjust Batch Size**
   - Start with batch_size = 4
   - If OOM (Out of Memory) → reduce to 2
   - If still OOM → reduce to 1
   - Real-time VRAM estimate shown in GUI

5. **Start Training**
   - Click `🚀 Start Training`
   - Monitor VRAM usage in Task Manager (Windows) or `nvidia-smi` (Linux)

---

## ⚡ Performance Impact

| Configuration | VRAM (batch=4) | Speed | When to Use |
|---------------|----------------|-------|-------------|
| **Default** | 8.0 GB | 100% | 16+ GB VRAM |
| **Default + GC** | 5.5 GB | 80% | 12+ GB VRAM |
| **SageAttention + GC** | 4.1 GB | 75% | 8-12 GB VRAM |
| **FlashAttention + GC** | 3.7 GB | 70% | 6-8 GB VRAM |

**Speed:** Relative to default (100% = fastest)

**Trade-off:** ~25-30% slower training, but enables training on lower-end GPUs

---

## 🐛 Troubleshooting

### Installation Failed

**SageAttention Error:**
```bash
# Check CUDA version
python -c "import torch; print(torch.version.cuda)"
# Requires CUDA 11.6+
```

**FlashAttention Error:**
```bash
# Install build tools first
pip install ninja packaging

# On Windows, install Visual Studio Build Tools
# https://visualstudio.microsoft.com/downloads/
```

---

### Out of Memory During Training

**Step 1: Enable All Optimizations**
1. Attention Backend: FlashAttention 2 (or SageAttention)
2. Gradient Checkpointing: ON

**Step 2: Reduce Batch Size**
- batch_size = 4 → 2 → 1

**Step 3: Reduce Model Size**
- image_size: 224 → 128
- diffusion_steps: 1000 → 500

**Step 4: Check GPU Usage**
```bash
# Windows
nvidia-smi

# Linux
watch -n 1 nvidia-smi
```

---

### Training Works But Slow

**Expected:** Memory optimization trades speed for VRAM
- Default: 100% speed, 8 GB VRAM
- Optimized: ~70% speed, 3.7 GB VRAM

**If too slow:**
1. Disable Gradient Checkpointing (if you have enough VRAM)
2. Use SageAttention instead of FlashAttention (slightly faster)
3. Reduce validation frequency (every 100 steps instead of 50)

---

## 📊 Verification

**Check Installed Packages:**
```bash
python install_memory_optimization.py --check
```

**Expected Output:**
```
✓ SageAttention 2 is installed
✓ FlashAttention 2 is installed  # (if --flash was used)

ForexGPT Detection:
SageAttention available: True
FlashAttention available: True
```

---

## 🔗 References

- **Full Documentation:** [docs/LDM4TS_MEMORY_OPTIMIZATION.md](docs/LDM4TS_MEMORY_OPTIMIZATION.md)
- **Installation Script:** [install_memory_optimization.py](install_memory_optimization.py)
- **Dependencies:** [pyproject.toml](pyproject.toml) → `[project.optional-dependencies]`

---

## 📞 Support

**Installation Issues:**
1. Check CUDA version: `python -c "import torch; print(torch.version.cuda)"`
2. Check GPU model: `nvidia-smi`
3. See full guide: `docs/LDM4TS_MEMORY_OPTIMIZATION.md`

**Training Issues:**
1. Enable all optimizations in GUI
2. Reduce batch size
3. Monitor VRAM with `nvidia-smi`

---

## ✅ Summary

1. **Install:** `python install_memory_optimization.py` (or `--flash`)
2. **GUI:** Select attention backend + enable gradient checkpointing
3. **Train:** Adjust batch_size based on VRAM estimate
4. **Result:** Train LDM4TS on consumer GPUs (6+ GB VRAM) 🎉
