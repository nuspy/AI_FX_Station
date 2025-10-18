"""
Fix Lightning checkpoint paths that have subdirectories due to val/loss in filename.

This script moves checkpoints from incorrect subdirectory structure:
  EURUSD-1m-epoch=00-val/loss=1.7588.ckpt
To correct flat structure:
  EURUSD-1m-epoch=00-val_loss=1.7588.ckpt
"""

import os
import shutil
from pathlib import Path
from loguru import logger


def fix_checkpoint_paths(artifacts_dir: Path, dry_run: bool = True):
    """
    Fix checkpoint paths by moving files from subdirectories to flat structure.
    
    Args:
        artifacts_dir: Path to artifacts directory
        dry_run: If True, only show what would be done without actually moving files
    """
    lightning_dir = artifacts_dir / "lightning"
    
    if not lightning_dir.exists():
        logger.error(f"Lightning directory not found: {lightning_dir}")
        return
    
    fixes_needed = []
    
    # Find all .ckpt files
    for ckpt_file in lightning_dir.rglob("*.ckpt"):
        # Check if file is in a subdirectory (not directly in lightning/)
        relative_path = ckpt_file.relative_to(lightning_dir)
        
        if len(relative_path.parts) > 1:
            # File is in a subdirectory
            parent_dir = ckpt_file.parent
            parent_name = parent_dir.name
            file_name = ckpt_file.name
            
            # Check if this looks like the val/loss issue
            if "loss=" in file_name and "val" in parent_name:
                # Expected: parent=EURUSD-1m-epoch=00-val, file=loss=1.7588.ckpt
                # Want: EURUSD-1m-epoch=00-val_loss=1.7588.ckpt
                
                new_filename = f"{parent_name}_{file_name}"
                new_path = lightning_dir / new_filename
                
                fixes_needed.append({
                    'old': ckpt_file,
                    'new': new_path,
                    'parent_dir': parent_dir
                })
    
    if not fixes_needed:
        logger.info("✓ No checkpoint paths need fixing")
        return
    
    logger.info(f"Found {len(fixes_needed)} checkpoint(s) that need fixing")
    
    for fix in fixes_needed:
        old_path = fix['old']
        new_path = fix['new']
        parent_dir = fix['parent_dir']
        
        logger.info(f"\n  Old: {old_path.relative_to(artifacts_dir)}")
        logger.info(f"  New: {new_path.relative_to(artifacts_dir)}")
        
        if not dry_run:
            try:
                # Move file
                shutil.move(str(old_path), str(new_path))
                logger.info(f"  ✓ Moved")
                
                # Try to remove empty parent directory
                try:
                    if parent_dir != lightning_dir:
                        parent_dir.rmdir()
                        logger.info(f"  ✓ Removed empty directory: {parent_dir.name}")
                except OSError:
                    logger.debug(f"  Directory not empty or already removed: {parent_dir.name}")
                    
            except Exception as e:
                logger.error(f"  ✗ Failed to move: {e}")
    
    if dry_run:
        logger.warning("\n⚠️  DRY RUN - No files were actually moved")
        logger.info("Run with --apply to actually fix the paths")
    else:
        logger.info(f"\n✓ Fixed {len(fixes_needed)} checkpoint path(s)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix Lightning checkpoint paths")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(__file__).parent.parent / "artifacts",
        help="Path to artifacts directory"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files (default is dry-run)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Scanning artifacts directory: {args.artifacts_dir}")
    logger.info(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    
    fix_checkpoint_paths(args.artifacts_dir, dry_run=not args.apply)
    
    if not args.apply:
        logger.info("\nTo actually fix the paths, run:")
        logger.info("  python scripts/fix_checkpoint_paths.py --apply")
