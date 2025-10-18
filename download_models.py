#!/usr/bin/env python3
"""
Download pre-trained models for LDM4TS.

This script downloads the required models from Hugging Face:
- Stable Diffusion VAE (stabilityai/sd-vae-ft-mse) ~330 MB
- CLIP Text Encoder (openai/clip-vit-base-patch32) ~470 MB

Usage:
    python download_models.py              # Download both models
    python download_models.py --vae        # Download VAE only
    python download_models.py --clip       # Download CLIP only
    python download_models.py --check      # Check if models exist
"""
import argparse
import sys
from pathlib import Path


def check_models():
    """Check if models are already downloaded."""
    project_root = Path(__file__).parent
    vae_path = project_root / "models" / "vae" / "config.json"
    clip_path = project_root / "models" / "clip" / "config.json"
    
    print("="*60)
    print("Checking installed models...")
    print("="*60)
    
    vae_exists = vae_path.exists()
    clip_exists = clip_path.exists()
    
    print(f"VAE (Stable Diffusion):  {'✓ Installed' if vae_exists else '✗ Missing'}")
    print(f"  Location: {project_root / 'models' / 'vae'}")
    
    print(f"CLIP (Text Encoder):     {'✓ Installed' if clip_exists else '✗ Missing'}")
    print(f"  Location: {project_root / 'models' / 'clip'}")
    
    print("="*60)
    
    if vae_exists and clip_exists:
        print("✓ All models are installed!")
        return True
    else:
        print("✗ Some models are missing")
        print("  Run: python download_models.py")
        return False


def download_vae():
    """Download Stable Diffusion VAE."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("✗ huggingface_hub not installed")
        print("  Install with: pip install huggingface_hub")
        return False
    
    print("\n" + "="*60)
    print("Downloading Stable Diffusion VAE...")
    print("Model: stabilityai/sd-vae-ft-mse (~330 MB)")
    print("="*60)
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    try:
        snapshot_download(
            repo_id='stabilityai/sd-vae-ft-mse',
            local_dir=str(models_dir / 'vae'),
            local_dir_use_symlinks=False
        )
        print("✓ VAE downloaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed to download VAE: {e}")
        return False


def download_clip():
    """Download CLIP model."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("✗ huggingface_hub not installed")
        print("  Install with: pip install huggingface_hub")
        return False
    
    print("\n" + "="*60)
    print("Downloading CLIP Text Encoder...")
    print("Model: openai/clip-vit-base-patch32 (~470 MB)")
    print("="*60)
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    try:
        snapshot_download(
            repo_id='openai/clip-vit-base-patch32',
            local_dir=str(models_dir / 'clip'),
            local_dir_use_symlinks=False
        )
        print("✓ CLIP downloaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed to download CLIP: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained models for LDM4TS"
    )
    parser.add_argument(
        "--vae",
        action="store_true",
        help="Download VAE only"
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="Download CLIP only"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if models are installed"
    )
    
    args = parser.parse_args()
    
    # Check mode
    if args.check:
        check_models()
        return
    
    # Determine what to download
    download_both = not (args.vae or args.clip)
    
    success = True
    
    if download_both or args.vae:
        if not download_vae():
            success = False
    
    if download_both or args.clip:
        if not download_clip():
            success = False
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✓ Download complete!")
        print("\nYou can now use LDM4TS training and inference.")
        print("The models are stored in: ./models/")
    else:
        print("✗ Some downloads failed")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Install huggingface_hub: pip install huggingface_hub")
        print("3. Check disk space (~800 MB required)")
        sys.exit(1)


if __name__ == "__main__":
    main()
