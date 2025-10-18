"""
LDM4TS Training Worker - Background worker for LDM4TS model training.

Handles:
- Data loading and preparation
- Vision encoding (OHLCV → RGB images)
- LDM4TS training with progress tracking
- Checkpoint saving
- Metrics reporting
"""
from __future__ import annotations

import time
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from PySide6.QtCore import QRunnable, QObject, Signal
from loguru import logger


class LDM4TSTrainingSignals(QObject):
    """Signals for LDM4TS training worker"""
    progress = Signal(int)  # Progress percentage (0-100)
    status = Signal(str)  # Status message
    epoch_complete = Signal(int, dict)  # Epoch number, metrics dict
    training_complete = Signal(str)  # Checkpoint path
    error = Signal(str)  # Error message


class LDM4TSTrainingWorker(QRunnable):
    """
    Background worker for LDM4TS training.
    Runs training in separate thread and emits progress signals.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params
        self.signals = LDM4TSTrainingSignals()
        self._stop_requested = False

    def stop(self):
        """Request worker to stop"""
        self._stop_requested = True
        logger.info("LDM4TS training stop requested")

    def run(self):
        """Execute training"""
        try:
            self.signals.status.emit("Initializing training...")
            logger.info(f"Starting LDM4TS training with params: {self.params}")
            
            # Extract parameters
            symbol = self.params['symbol']
            timeframe = self.params['timeframe']
            window_size = self.params['window_size']
            horizons_str = self.params['horizons']
            diffusion_steps = self.params['diffusion_steps']
            image_size = self.params['image_size']
            epochs = self.params['epochs']
            batch_size = self.params['batch_size']
            learning_rate = self.params['learning_rate']
            use_gpu = self.params['use_gpu']
            output_dir = Path(self.params['output_dir'])
            
            # Parse horizons
            horizons = [int(h.strip()) for h in horizons_str.split(',')]
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_dir / f"{symbol.replace('/', '')}_{timeframe}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Output directory: {run_dir}")
            self.signals.status.emit(f"Output: {run_dir}")
            
            # Step 1: Load data
            self.signals.status.emit("Loading market data...")
            self.signals.progress.emit(5)
            
            df_train, df_val = self._load_data(symbol, timeframe, window_size)
            
            if df_train is None or len(df_train) < window_size:
                raise RuntimeError(f"Insufficient training data: {len(df_train) if df_train is not None else 0} candles")
            
            logger.info(f"Loaded {len(df_train)} training samples, {len(df_val) if df_val is not None else 0} validation samples")
            self.signals.status.emit(f"Data loaded: {len(df_train)} train, {len(df_val) if df_val is not None else 0} val")
            self.signals.progress.emit(10)
            
            # Step 2: Initialize LDM4TS model
            self.signals.status.emit("Initializing LDM4TS model...")
            self.signals.progress.emit(15)
            
            import torch
            import torch.optim as optim
            from ...models.ldm4ts import LDM4TSModel
            
            # Create model
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            
            model = LDM4TSModel(
                image_size=(image_size, image_size),
                horizons=horizons,
                diffusion_steps=diffusion_steps,
                device=device
            )
            
            # Optimizer
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-5
            )
            
            # Loss function
            criterion = torch.nn.MSELoss()
            
            logger.info(f"Model initialized on {device}")
            self.signals.status.emit(f"Model initialized on {device}")
            self.signals.progress.emit(20)
            
            # Step 3: Training loop
            self.signals.status.emit("Starting training...")
            
            # Prepare data for batching
            from torch.utils.data import TensorDataset, DataLoader
            
            # Create simple dataset (sliding windows)
            train_windows = []
            train_targets = []
            
            for i in range(len(df_train) - window_size - max(horizons)):
                window = df_train.iloc[i:i+window_size][['open', 'high', 'low', 'close', 'volume']].values
                
                # Targets: future close prices at each horizon
                targets = []
                for h in horizons:
                    future_idx = i + window_size + h - 1
                    if future_idx < len(df_train):
                        targets.append(df_train.iloc[future_idx]['close'])
                
                if len(targets) == len(horizons):
                    train_windows.append(window)
                    train_targets.append(targets)
            
            train_windows = torch.FloatTensor(np.array(train_windows))
            train_targets = torch.FloatTensor(np.array(train_targets))
            
            train_dataset = TensorDataset(train_windows, train_targets)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            logger.info(f"Created {len(train_dataset)} training samples")
            
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                if self._stop_requested:
                    logger.info("Training stopped by user")
                    self.signals.status.emit("Training stopped by user")
                    return
                
                epoch_start = time.time()
                model.train()
                
                epoch_loss = 0.0
                num_batches = len(train_loader)
                
                self.signals.status.emit(f"Epoch {epoch + 1}/{epochs} - Training...")
                
                for batch_idx, (batch_windows, batch_targets) in enumerate(train_loader):
                    if self._stop_requested:
                        break
                    
                    batch_windows = batch_windows.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    
                    # Get predictions
                    current_prices = batch_windows[:, -1, 3]  # Last close price
                    
                    try:
                        outputs = model(
                            batch_windows,
                            current_price=current_prices,
                            num_samples=1,  # Single sample for training
                            return_all=False
                        )
                        
                        predictions = outputs['mean']  # [B, num_horizons]
                        
                        # Compute loss
                        loss = criterion(predictions, batch_targets)
                        
                        # Backward pass
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        
                    except Exception as e:
                        logger.warning(f"Batch {batch_idx} failed: {e}")
                        continue
                    
                    # Update progress within epoch
                    epoch_progress = 20 + int(((epoch + (batch_idx / num_batches)) / epochs) * 70)
                    self.signals.progress.emit(epoch_progress)
                
                avg_train_loss = epoch_loss / max(num_batches, 1)
                
                # Validation
                val_loss = self._validate(model, df_val, window_size, horizons, criterion, device)
                
                epoch_time = time.time() - epoch_start
                
                metrics = {
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'epoch_time': epoch_time,
                    'learning_rate': learning_rate,
                }
                
                logger.info(f"Epoch {epoch + 1}/{epochs} - train_loss: {metrics['train_loss']:.4f}, val_loss: {metrics['val_loss']:.4f}")
                
                # Emit epoch complete
                self.signals.epoch_complete.emit(epoch + 1, metrics)
                self.signals.status.emit(f"Epoch {epoch + 1}/{epochs} - Loss: {metrics['train_loss']:.4f}")
                
                # Save checkpoint every epoch
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                
                checkpoint_path = run_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                self._save_checkpoint(model, optimizer, checkpoint_path, epoch + 1, metrics, is_best)
            
            # Training complete
            self.signals.progress.emit(100)
            
            # Save final checkpoint
            final_checkpoint = run_dir / "model_final.pt"
            self._save_checkpoint(model, optimizer, final_checkpoint, epochs, metrics, is_best=False)
            
            # Also save best checkpoint reference
            best_checkpoint = run_dir / "model_best.pt"
            if best_checkpoint.exists():
                logger.info(f"Best checkpoint: {best_checkpoint} (val_loss={best_val_loss:.4f})")
            
            logger.info(f"Training completed. Final checkpoint: {final_checkpoint}")
            self.signals.status.emit("Training completed successfully!")
            self.signals.training_complete.emit(str(final_checkpoint))
            
        except Exception as e:
            logger.exception(f"LDM4TS training failed: {e}")
            self.signals.error.emit(str(e))
            self.signals.status.emit(f"Training failed: {e}")

    def _load_data(self, symbol: str, timeframe: str, window_size: int) -> tuple:
        """Load training and validation data"""
        try:
            from ...services.marketdata import MarketDataService
            from sqlalchemy import text
            
            ms = MarketDataService()
            
            # Fetch recent data (last 10000 candles)
            with ms.engine.connect() as conn:
                query = text("""
                    SELECT ts_utc, open, high, low, close, volume
                    FROM market_data_candles
                    WHERE symbol = :symbol AND timeframe = :timeframe
                    ORDER BY ts_utc DESC
                    LIMIT 10000
                """)
                rows = conn.execute(query, {
                    "symbol": symbol,
                    "timeframe": timeframe
                }).fetchall()
            
            if not rows:
                return None, None
            
            # Convert to DataFrame (reverse to chronological order)
            df = pd.DataFrame([
                {
                    'timestamp': pd.to_datetime(r[0], unit='ms', utc=True),
                    'open': float(r[1]),
                    'high': float(r[2]),
                    'low': float(r[3]),
                    'close': float(r[4]),
                    'volume': float(r[5]) if r[5] is not None else 0.0
                }
                for r in reversed(rows)
            ])
            df = df.set_index('timestamp')
            
            # Split train/val (80/20)
            split_idx = int(len(df) * 0.8)
            df_train = df.iloc[:split_idx]
            df_val = df.iloc[split_idx:]
            
            return df_train, df_val
            
        except Exception as e:
            logger.exception(f"Failed to load data: {e}")
            return None, None

    def _validate(self, model, df_val, window_size, horizons, criterion, device):
        """Run validation"""
        if df_val is None or len(df_val) < window_size:
            return 0.0
        
        model.eval()
        val_loss = 0.0
        num_samples = 0
        
        import torch
        
        with torch.no_grad():
            for i in range(0, len(df_val) - window_size - max(horizons), 50):  # Sample every 50 steps
                window = df_val.iloc[i:i+window_size][['open', 'high', 'low', 'close', 'volume']].values
                
                targets = []
                for h in horizons:
                    future_idx = i + window_size + h - 1
                    if future_idx < len(df_val):
                        targets.append(df_val.iloc[future_idx]['close'])
                
                if len(targets) == len(horizons):
                    window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
                    targets_tensor = torch.FloatTensor(targets).unsqueeze(0).to(device)
                    
                    current_price = window[-1, 3]
                    
                    try:
                        outputs = model(window_tensor, current_price=current_price, num_samples=1, return_all=False)
                        predictions = outputs['mean']
                        
                        loss = criterion(predictions, targets_tensor)
                        val_loss += loss.item()
                        num_samples += 1
                    except Exception:
                        continue
        
        return val_loss / max(num_samples, 1)
    
    def _save_checkpoint(self, model, optimizer, checkpoint_path: Path, epoch: int, metrics: dict, is_best: bool = False):
        """Save training checkpoint"""
        try:
            import torch
            
            # Create checkpoint data
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'params': self.params,
            }
            
            # Save PyTorch checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # If best, also save as model_best.pt
            if is_best:
                best_path = checkpoint_path.parent / "model_best.pt"
                torch.save(checkpoint_data, best_path)
                logger.info(f"Best model saved: {best_path}")
            
        except Exception as e:
            logger.exception(f"Failed to save checkpoint: {e}")
