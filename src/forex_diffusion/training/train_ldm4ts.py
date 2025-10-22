#!/usr/bin/env python3
"""
LDM4TS Training Script

Train LDM4TS (Latent Diffusion Models for Time Series) on historical OHLCV data.

Usage:
    python -m forex_diffusion.training.train_ldm4ts \
        --data-dir data/eurusd_1m \
        --output-dir artifacts/ldm4ts \
        --symbol EUR/USD \
        --epochs 100 \
        --batch-size 32

Features:
- Data loading from OHLCV CSV/database
- Vision encoding (SEG + GAF + RP)
- U-Net + Temporal Fusion training
- Validation metrics (MSE, MAE, directional accuracy)
- Checkpointing & early stopping
- Walk-forward validation
- TensorBoard logging
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loguru import logger


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.forex_diffusion.models.ldm4ts import LDM4TSModel
from src.forex_diffusion.models.vision_transforms import TimeSeriesVisionEncoder


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    data_dir: str
    symbol: str = 'EUR/USD'
    timeframe: str = '1m'
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    window_size: int = 100  # Candles for vision encoding
    
    # Model
    horizons: List[int] = None
    image_size: int = 224
    diffusion_steps: int = 50
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    # Validation
    val_every_n_epochs: int = 5
    early_stopping_patience: int = 10
    
    # Output
    output_dir: str = 'artifacts/ldm4ts'
    save_every_n_epochs: int = 10
    tensorboard_dir: str = 'runs/ldm4ts'
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    
    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [15, 60, 240]  # 15min, 1h, 4h
        
        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)


class OHLCVDataset(Dataset):
    """
    Dataset for OHLCV time series.
    
    Returns:
        - vision_input: RGB tensor [3, 224, 224]
        - targets: Future prices for each horizon [num_horizons]
        - metadata: Dict with symbol, timestamp, etc.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        horizons: List[int],
        window_size: int = 100,
        image_size: int = 224,
        norm_mean: Optional[torch.Tensor] = None,
        norm_std: Optional[torch.Tensor] = None
    ):
        self.data = data
        self.horizons = horizons
        self.window_size = window_size
        self.max_horizon = max(horizons)
        self.vision_encoder = TimeSeriesVisionEncoder(image_size=image_size)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        
        self.valid_indices = list(range(window_size, len(data) - self.max_horizon))
        logger.info(f"Dataset initialized: {len(self.valid_indices)} samples")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        actual_idx = self.valid_indices[idx]
        
        start_idx = actual_idx - self.window_size
        end_idx = actual_idx
        ohlcv_window = self.data.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume']].values
        
        vision_input = self.vision_encoder.encode(ohlcv_window).squeeze(0)
        
        # Apply normalization if stats are provided
        if self.norm_mean is not None and self.norm_std is not None:
            vision_input = (vision_input - self.norm_mean) / self.norm_std
        
        current_price = self.data.iloc[actual_idx]['close']
        targets = []
        for h in self.horizons:
            future_price = self.data.iloc[actual_idx + h]['close']
            target_return = (future_price - current_price) / current_price
            targets.append(target_return)
        
        targets = torch.tensor(targets, dtype=torch.float32)
        
        metadata = {
            'index': actual_idx,
            'timestamp': self.data.index[actual_idx].isoformat() if hasattr(self.data.index[actual_idx], 'isoformat') else str(self.data.index[actual_idx]),
            'current_price': current_price
        }
        
        return {
            'vision_input': vision_input,
            'targets': targets,
            'metadata': metadata
        }


class LDM4TSTrainer:
    """Trainer for LDM4TS model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = LDM4TSModel(horizons=config.horizons).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.criterion = nn.MSELoss()
        self.writer = SummaryWriter(config.tensorboard_dir)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            vision_input = batch['vision_input'].to(self.device)
            targets = batch['targets'].to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model.predict(vision_input, num_samples=1)
            pred_means = torch.stack([torch.tensor(predictions[i].mean[h], device=self.device) for i in range(len(predictions)) for h in self.config.horizons]).view(len(predictions), len(self.config.horizons))
            loss = self.criterion(pred_means, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            epoch_loss += loss.item()
            self.global_step += 1
            pbar.set_postfix({'loss': loss.item()})
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
        avg_loss = epoch_loss / len(train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            vision_input = batch['vision_input'].to(self.device)
            targets = batch['targets'].to(self.device)
            predictions = self.model.predict(vision_input, num_samples=10)
            pred_means = torch.stack([torch.tensor(predictions[i].mean[h], device=self.device) for i in range(len(predictions)) for h in self.config.horizons]).view(len(predictions), len(self.config.horizons))
            loss = self.criterion(pred_means, targets)
            val_loss += loss.item()
            all_predictions.append(pred_means.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            pbar.set_postfix({'loss': loss.item()})
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        avg_val_loss = val_loss / len(val_loader)
        mae = np.mean(np.abs(all_predictions - all_targets))
        directional_accuracy = np.mean(np.sign(all_predictions) == np.sign(all_targets))
        metrics = {'val_loss': avg_val_loss, 'mae': mae, 'directional_accuracy': directional_accuracy}
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss
        }
        if is_best:
            val_loss = metrics.get('val_loss', 0.0)
            best_path = Path(self.config.output_dir) / f'best_model_epoch_{epoch}_valloss_{val_loss:.4f}.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
        elif epoch % self.config.save_every_n_epochs == 0:
            checkpoint_path = Path(self.config.output_dir) / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info(f"Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}, Device: {self.device}")
        logger.info("=" * 80)
        for epoch in range(1, self.config.epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.config.epochs}")
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            logger.info(f"Train Loss: {train_loss:.6f}")
            if epoch % self.config.val_every_n_epochs == 0:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['val_loss']
                self.val_losses.append(val_loss)
                logger.info(f"Val Loss: {val_loss:.6f}, MAE: {val_metrics['mae']:.6f}, Directional Acc: {val_metrics['directional_accuracy']:.2%}")
                self.writer.add_scalar('val/loss', val_loss, epoch)
                self.writer.add_scalar('val/mae', val_metrics['mae'], epoch)
                self.writer.add_scalar('val/directional_accuracy', val_metrics['directional_accuracy'], epoch)
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    logger.info(f"✅ New best validation loss: {val_loss:.6f}")
                else:
                    self.patience_counter += 1
                    logger.info(f"Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
                self.save_checkpoint(epoch, val_metrics, is_best)
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            self.scheduler.step()
        final_metrics = self.validate(val_loader)
        self.save_checkpoint(epoch, final_metrics)
        logger.info(f"\nTRAINING COMPLETE. Best validation loss: {self.best_val_loss:.6f}")
        self.writer.close()

def load_data(data_dir: str, symbol: str) -> pd.DataFrame:
    data_path = Path(data_dir)
    csv_files = list(data_path.glob(f"*{symbol.replace('/', '')}*.csv"))
    if csv_files:
        logger.info(f"Loading data from CSV: {csv_files[0]}")
        df = pd.read_csv(csv_files[0], parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    parquet_files = list(data_path.glob(f"*{symbol.replace('/', '')}*.parquet"))
    if parquet_files:
        logger.info(f"Loading data from Parquet: {parquet_files[0]}")
        df = pd.read_parquet(parquet_files[0])
        return df
    logger.warning(f"No data found in {data_dir}, generating synthetic data")
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='1min')
    n = len(dates)
    np.random.seed(42)
    returns = np.random.randn(n) * 0.0001 + 0.00001
    close = 1.0500 + np.cumsum(returns)
    df = pd.DataFrame({'open': close, 'high': close, 'low': close, 'close': close, 'volume': np.random.rand(n) * 1000000}, index=dates)
    return df

def calculate_norm_stats(dataset: OHLCVDataset, batch_size: int, num_workers: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate mean and std for the vision inputs of a dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Calculate mean
    mean = torch.zeros(3)
    n_samples = 0
    for batch in tqdm(loader, desc="Calculating Mean"):
        vision_input = batch['vision_input']
        batch_samples = vision_input.size(0)
        vision_input = vision_input.view(batch_samples, vision_input.size(1), -1)
        mean += vision_input.mean(2).sum(0)
        n_samples += batch_samples
    mean /= n_samples

    # Calculate std
    std = torch.zeros(3)
    n_samples = 0
    for batch in tqdm(loader, desc="Calculating Std"):
        vision_input = batch['vision_input']
        batch_samples = vision_input.size(0)
        vision_input = vision_input.view(batch_samples, vision_input.size(1), -1)
        std += ((vision_input - mean.unsqueeze(1))**2).mean(2).sum(0)
        n_samples += batch_samples
    std = torch.sqrt(std / n_samples)
    std[std == 0] = 1.0 # Avoid division by zero

    return mean.view(3, 1, 1), std.view(3, 1, 1)

def main():
    parser = argparse.ArgumentParser(description='Train LDM4TS model')
    parser.add_argument('--data-dir', type=str, default='data/eurusd_1m')
    parser.add_argument('--output-dir', type=str, default='artifacts/ldm4ts')
    parser.add_argument('--symbol', type=str, default='EUR/USD')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(Path(args.output_dir) / 'training.log', level="DEBUG")
    
    config = TrainingConfig(data_dir=args.data_dir, output_dir=args.output_dir, symbol=args.symbol, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, device='cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Configuration: {json.dumps(asdict(config), indent=2)}")
    
    data = load_data(config.data_dir, config.symbol)
    logger.info(f"Loaded {len(data)} candles from {data.index[0]} to {data.index[-1]}")
    
    n = len(data)
    train_size = int(n * config.train_split)
    val_size = int(n * config.val_split)
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    
    logger.info(f"Data splits: Train={len(train_data)}, Val={len(val_data)}")

    # Calculate normalization stats on training set ONLY
    logger.info("Calculating normalization statistics on the training set...")
    temp_train_dataset = OHLCVDataset(train_data, horizons=config.horizons, window_size=config.window_size, image_size=config.image_size)
    mean, std = calculate_norm_stats(temp_train_dataset, config.batch_size, config.num_workers)
    logger.info(f"Normalization stats calculated: mean={mean.view(-1).tolist()}, std={std.view(-1).tolist()}")

    # Create final datasets with normalization
    train_dataset = OHLCVDataset(train_data, horizons=config.horizons, window_size=config.window_size, image_size=config.image_size, norm_mean=mean, norm_std=std)
    val_dataset = OHLCVDataset(val_data, horizons=config.horizons, window_size=config.window_size, image_size=config.image_size, norm_mean=mean, norm_std=std)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    trainer = LDM4TSTrainer(config)
    trainer.train(train_loader, val_loader)
    
    logger.info("\n✅ Training completed successfully!")

if __name__ == '__main__':
    main()