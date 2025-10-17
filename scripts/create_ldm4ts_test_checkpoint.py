#!/usr/bin/env python3
"""
Create LDM4TS Test Checkpoint

Generates a LDM4TS checkpoint with initialized weights for testing purposes.
This allows testing the inference pipeline without training a full model.

Usage:
    python scripts/create_ldm4ts_test_checkpoint.py
"""

import sys
from pathlib import Path
from datetime import datetime
import torch
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.forex_diffusion.models.ldm4ts import LDM4TSModel


def create_test_checkpoint(
    output_path: str = "artifacts/ldm4ts/test_checkpoint.pt",
    horizons: list[int] = [15, 60, 240],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Create a test checkpoint with initialized weights.
    
    Args:
        output_path: Where to save checkpoint
        horizons: Forecast horizons (minutes)
        device: Device to use for initialization
    """
    logger.info("Creating LDM4TS test checkpoint...")
    logger.info(f"  Horizons: {horizons}")
    logger.info(f"  Device: {device}")
    
    # Create model
    model = LDM4TSModel(
        horizons=horizons,
        image_size=(224, 224),
        device=device
    )
    
    # Initialize weights (already done in __init__, but ensure proper initialization)
    def init_weights(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    logger.info("Model initialized with Kaiming normal weights")
    
    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'horizons': horizons,
            'image_size': 224,
            'diffusion_steps': 50,
            'vision_types': ['seg', 'gaf', 'rp'],
        },
        'metadata': {
            'checkpoint_type': 'test',
            'created_at': datetime.now().isoformat(),
            'note': 'Test checkpoint with random initialized weights for inference testing',
        }
    }
    
    # Save checkpoint
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, output_path)
    logger.info(f"✅ Test checkpoint saved to: {output_path}")
    
    # Verify checkpoint
    loaded = torch.load(output_path, map_location='cpu')
    logger.info(f"✅ Checkpoint verified:")
    logger.info(f"   - Model parameters: {len(loaded['model_state_dict'])} tensors")
    logger.info(f"   - Config: {loaded['config']}")
    logger.info(f"   - Metadata: {loaded['metadata']}")
    
    # Test loading into model
    test_model = LDM4TSModel(
        horizons=horizons,
        image_size=(224, 224),
        device='cpu'
    )
    test_model.load_state_dict(loaded['model_state_dict'])
    logger.info("✅ Checkpoint successfully loaded into model")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create LDM4TS test checkpoint")
    parser.add_argument(
        "--output",
        default="artifacts/ldm4ts/test_checkpoint.pt",
        help="Output checkpoint path"
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[15, 60, 240],
        help="Forecast horizons in minutes"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    try:
        checkpoint_path = create_test_checkpoint(
            output_path=args.output,
            horizons=args.horizons,
            device=args.device
        )
        logger.success(f"🎉 Test checkpoint created successfully: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to create checkpoint: {e}")
        sys.exit(1)
