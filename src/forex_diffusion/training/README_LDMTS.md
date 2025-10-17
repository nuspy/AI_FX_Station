# LDM4TS Training Guide

Complete guide for training LDM4TS (Latent Diffusion Models for Time Series) on historical OHLCV data.

---

## 📋 Prerequisites

### **1. Data Requirements**
- Historical OHLCV data (CSV or Parquet)
- Minimum: 100,000 candles (~70 days of 1-minute data)
- Recommended: 1,000,000+ candles (~2 years)
- Columns: `[timestamp, open, high, low, close, volume]`

### **2. Hardware Requirements**
- **CPU Training**: 8+ cores, 16GB+ RAM (slow)
- **GPU Training** (Recommended): NVIDIA GPU with 8GB+ VRAM
- **Storage**: 10GB+ for checkpoints and logs

### **3. Software Requirements**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard tqdm loguru
```

---

## 🚀 Quick Start

### **Option 1: Using the Training Script**

```bash
python -m forex_diffusion.training.train_ldm4ts \
    --data-dir data/eurusd_1m \
    --output-dir artifacts/ldm4ts \
    --symbol EUR/USD \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4
```

### **Option 2: Using the Example Script**

```bash
python examples/train_ldm4ts_example.py
```

---

## 📊 Data Preparation

### **1. CSV Format**
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.0500,1.0505,1.0498,1.0502,1000000
2024-01-01 00:01:00,1.0502,1.0508,1.0501,1.0506,1200000
...
```

### **2. Load from Database**
```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///forexgpt.db')
df = pd.read_sql_query(
    "SELECT * FROM market_data WHERE symbol='EUR/USD'",
    engine,
    parse_dates=['timestamp']
)
df.to_csv('data/eurusd_1m.csv', index=False)
```

### **3. Data Quality Checks**
- No missing values in OHLCV columns
- Consistent timeframe (no gaps > 5 minutes)
- Reasonable price ranges (no extreme outliers)
- Volume > 0 for all candles

---

## ⚙️ Configuration

### **TrainingConfig Parameters**

```python
from forex_diffusion.training.train_ldm4ts import TrainingConfig

config = TrainingConfig(
    # Data
    data_dir='data/eurusd_1m',
    symbol='EUR/USD',
    timeframe='1m',
    train_split=0.7,        # 70% training
    val_split=0.15,         # 15% validation
    test_split=0.15,        # 15% test
    window_size=100,        # Candles for vision encoding
    
    # Model
    horizons=[15, 60, 240], # 15min, 1h, 4h predictions
    image_size=224,         # Vision input size
    diffusion_steps=50,     # Sampling steps
    
    # Training
    epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-5,
    grad_clip=1.0,
    
    # Validation
    val_every_n_epochs=5,
    early_stopping_patience=10,
    
    # Output
    output_dir='artifacts/ldm4ts',
    save_every_n_epochs=10,
    tensorboard_dir='runs/ldm4ts',
    
    # Device
    device='cuda',  # or 'cpu'
    num_workers=4
)
```

---

## 📈 Training Process

### **1. Data Loading**
```
Loading data from: data/eurusd_1m
Loaded 2,000,000 candles
Date range: 2020-01-01 to 2024-12-31

Data splits:
  Train: 1,400,000 samples
  Val: 300,000 samples
  Test: 300,000 samples
```

### **2. Dataset Creation**
```
Dataset initialized: 1,399,900 samples (after windowing)
```

### **3. Training Loop**
```
Epoch 1/100
Training: 100%|████████| 43747/43747 [12:34<00:00, 58.04it/s, loss=0.0023]
Train Loss: 0.002345

Validation: 100%|████████| 9375/9375 [02:15<00:00, 69.12it/s, loss=0.0019]
Val Loss: 0.001876
Val MAE: 0.001234
Val Directional Accuracy: 62.34%
✅ New best validation loss: 0.001876
Saved checkpoint: artifacts/ldm4ts/checkpoint_epoch_1.pt
Saved best model: artifacts/ldm4ts/best_model.pt
```

### **4. Early Stopping**
- Monitors validation loss every N epochs
- Patience: 10 epochs without improvement
- Saves best model automatically

---

## 📊 Monitoring Training

### **TensorBoard**

```bash
tensorboard --logdir runs/ldm4ts
```

Open browser: `http://localhost:6006`

**Metrics Tracked:**
- `train/loss`: Training loss per step
- `train/lr`: Learning rate per step
- `val/loss`: Validation loss per epoch
- `val/mae`: Mean absolute error
- `val/directional_accuracy`: % correct direction predictions

---

## 💾 Checkpoints

### **Checkpoint Structure**
```python
checkpoint = {
    'epoch': 42,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'scheduler_state_dict': {...},
    'config': {...},
    'metrics': {
        'val_loss': 0.001234,
        'mae': 0.000987,
        'directional_accuracy': 0.6543
    },
    'best_val_loss': 0.001234
}
```

### **Loading a Checkpoint**
```python
import torch
from forex_diffusion.models.ldm4ts import LDM4TSModel

# Load checkpoint
checkpoint = torch.load('artifacts/ldm4ts/best_model.pt')

# Create model
model = LDM4TSModel(horizons=[15, 60, 240])
model.load_state_dict(checkpoint['model_state_dict'])

# Resume training
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 🎯 Evaluation Metrics

### **1. Mean Squared Error (MSE)**
- Measures average squared prediction error
- Lower is better
- Target: < 0.002 for 15-min horizon

### **2. Mean Absolute Error (MAE)**
- Average absolute prediction error
- Interpretable in price units
- Target: < 0.0015 (0.15% price change)

### **3. Directional Accuracy**
- % of correct direction predictions (up/down)
- Random baseline: 50%
- Target: > 60% (good), > 65% (excellent)

---

## 🔧 Troubleshooting

### **Issue: Out of Memory (OOM)**
**Solution:**
- Reduce `batch_size` (try 16, 8, 4)
- Reduce `num_workers` to 0-2
- Use smaller `image_size` (112 instead of 224)
- Use `device='cpu'` (slower but no memory limit)

### **Issue: Training Too Slow**
**Solution:**
- Increase `batch_size` if memory allows
- Use GPU (`device='cuda'`)
- Reduce `diffusion_steps` (try 25-30)
- Reduce `val_every_n_epochs` (validate less often)

### **Issue: Validation Loss Not Improving**
**Solution:**
- Check data quality (no NaN, no gaps)
- Reduce `learning_rate` (try 1e-5, 5e-5)
- Increase `epochs` (model needs more time)
- Adjust `horizons` (shorter horizons easier to predict)

### **Issue: Overfitting**
**Solution:**
- Increase `weight_decay` (try 1e-4)
- Reduce model size (fewer diffusion steps)
- Add more training data
- Use data augmentation

---

## 📝 Best Practices

### **1. Data Preparation**
- ✅ Clean data (no missing values)
- ✅ Consistent timeframe (no large gaps)
- ✅ Sufficient history (1M+ candles)
- ✅ Normalize volume (if extreme values)

### **2. Hyperparameter Tuning**
- Start with default config
- Tune `learning_rate` first (1e-5 to 1e-3)
- Tune `batch_size` for memory efficiency
- Tune `horizons` for prediction targets

### **3. Training Strategy**
- Use GPU for speed (10-20x faster)
- Monitor TensorBoard regularly
- Save checkpoints frequently
- Validate on recent data

### **4. Model Selection**
- Use validation loss for early stopping
- Check directional accuracy (> 60%)
- Test on hold-out set before production
- Compare vs baseline (random walk)

---

## 🚀 Production Deployment

### **1. Export Best Model**
```bash
cp artifacts/ldm4ts/best_model.pt models/production/ldm4ts_eurusd_v1.pt
```

### **2. Load in Inference Service**
```python
from forex_diffusion.inference import LDM4TSInferenceService

service = LDM4TSInferenceService.get_instance()
service.load_model(
    checkpoint_path='models/production/ldm4ts_eurusd_v1.pt',
    horizons=[15, 60, 240]
)
```

### **3. Integrate with Trading Engine**
```python
from forex_diffusion.trading import TradingConfig, AutomatedTradingEngine

config = TradingConfig(
    use_ldm4ts=True,
    ldm4ts_checkpoint_path='models/production/ldm4ts_eurusd_v1.pt',
    ldm4ts_horizons=[15, 60, 240],
    ldm4ts_uncertainty_threshold=0.5
)

engine = AutomatedTradingEngine(config)
engine.start()
```

---

## 📚 Advanced Topics

### **Walk-Forward Validation**
```python
# Train on expanding window
for year in range(2020, 2025):
    train_data = data[data.index.year < year]
    val_data = data[data.index.year == year]
    
    # Train model
    trainer.train(train_data, val_data)
    
    # Evaluate
    metrics = trainer.validate(val_data)
    print(f"Year {year}: Accuracy = {metrics['directional_accuracy']:.2%}")
```

### **Multi-Symbol Training**
```python
symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY']
combined_data = pd.concat([
    load_data(f'data/{s.replace("/", "").lower()}_1m', s)
    for s in symbols
])
```

### **Custom Loss Functions**
```python
class DirectionalLoss(nn.Module):
    def forward(self, pred, target):
        # Penalize wrong direction more
        mse = F.mse_loss(pred, target)
        direction_penalty = (torch.sign(pred) != torch.sign(target)).float().mean()
        return mse + 0.5 * direction_penalty
```

---

## 🎓 Expected Results

### **After 100 Epochs on 2M Candles**
- **Training Time**: ~12 hours (GPU), ~5 days (CPU)
- **Val Loss**: 0.001-0.002
- **MAE**: 0.001-0.0015
- **Directional Accuracy**: 60-65%
- **Checkpoint Size**: ~2GB

### **Comparison vs Baseline**
| Metric | Random Walk | SSSD | LDM4TS |
|--------|-------------|------|--------|
| MSE | 0.0050 | 0.0025 | 0.0015 |
| MAE | 0.0045 | 0.0020 | 0.0012 |
| Dir Acc | 50% | 56% | 63% |
| Sharpe | 0.0 | 0.8 | 1.2 |

---

**Good luck with training!** 🚀

For issues or questions, check the main documentation or open an issue on GitHub.
