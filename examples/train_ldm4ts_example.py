#!/usr/bin/env python3
"""
LDM4TS Training Example

Quick start example for training LDM4TS on historical data.

Usage:
    python examples/train_ldm4ts_example.py
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.forex_diffusion.training.train_ldm4ts import (
    TrainingConfig,
    LDM4TSTrainer,
    OHLCVDataset,
    load_data
)
from torch.utils.data import DataLoader
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")


def main():
    """Run quick training example"""
    
    logger.info("=" * 80)
    logger.info("LDM4TS TRAINING EXAMPLE")
    logger.info("=" * 80)
    
    # 1. Configure training
    config = TrainingConfig(
        data_dir='data/eurusd_1m',
        output_dir='artifacts/ldm4ts_example',
        symbol='EUR/USD',
        timeframe='1m',
        
        # Model settings
        horizons=[15, 60, 240],  # 15min, 1h, 4h
        image_size=224,
        diffusion_steps=50,
        
        # Training settings (small for demo)
        epochs=10,
        batch_size=16,
        learning_rate=1e-4,
        
        # Validation
        val_every_n_epochs=2,
        early_stopping_patience=5,
        
        # Output
        save_every_n_epochs=5
    )
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Symbol: {config.symbol}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Horizons: {config.horizons}")
    logger.info(f"  Device: {config.device}")
    
    # 2. Load data
    logger.info(f"\nLoading data from: {config.data_dir}")
    data = load_data(config.data_dir, config.symbol)
    logger.info(f"Loaded {len(data)} candles")
    
    # 3. Split data
    n = len(data)
    train_size = int(n * config.train_split)
    val_size = int(n * config.val_split)
    
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    
    logger.info(f"\nData splits:")
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Val: {len(val_data)} samples")
    
    # 4. Create datasets
    train_dataset = OHLCVDataset(
        train_data,
        horizons=config.horizons,
        window_size=config.window_size,
        image_size=config.image_size
    )
    
    val_dataset = OHLCVDataset(
        val_data,
        horizons=config.horizons,
        window_size=config.window_size,
        image_size=config.image_size
    )
    
    logger.info(f"\nDatasets created:")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")
    
    # 5. Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 6. Initialize trainer
    logger.info("\nInitializing trainer...")
    trainer = LDM4TSTrainer(config)
    
    # 7. Train
    logger.info("\nStarting training...\n")
    try:
        trainer.train(train_loader, val_loader)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"\nBest model saved to: {config.output_dir}/best_model.pt")
        logger.info(f"TensorBoard logs: {config.tensorboard_dir}")
        logger.info("\nTo view training progress:")
        logger.info(f"  tensorboard --logdir {config.tensorboard_dir}")
        logger.info("\nTo use the trained model:")
        logger.info(f"  from forex_diffusion.inference import LDM4TSInferenceService")
        logger.info(f"  service = LDM4TSInferenceService.get_instance()")
        logger.info(f"  service.load_model('{config.output_dir}/best_model.pt')")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)


if __name__ == '__main__':
    main()
