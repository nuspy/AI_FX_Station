"""
CLI for artifact management and cleanup.

PROC-003: Implements artifact cleanup commands to manage disk space.

Usage:
    python -m forex_diffusion.cli.artifacts clean --keep-best 10
    python -m forex_diffusion.cli.artifacts status
    python -m forex_diffusion.cli.artifacts list
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from loguru import logger

from ..models.artifact_manager import ArtifactManager
from ..utils.config import get_config


def cmd_clean(args):
    """Clean up old artifacts."""
    config = get_config()
    artifacts_dir = Path(args.artifacts_dir or config.model.artifacts_dir)

    if not artifacts_dir.exists():
        logger.warning(f"Artifacts directory does not exist: {artifacts_dir}")
        return

    manager = ArtifactManager(artifacts_dir)

    # Get current status before cleanup
    before_usage = manager.get_artifacts_disk_usage()
    before_count = before_usage['artifact_count']

    logger.info(f"Current artifacts: {before_count} ({before_usage['total_mb']:.2f} MB)")

    # Perform cleanup
    max_saved = args.keep_best if args.keep_best is not None else config.model.max_saved
    deleted_count = manager.cleanup_old_artifacts(max_saved=max_saved, keep_best=not args.no_protect)

    # Get status after cleanup
    after_usage = manager.get_artifacts_disk_usage()
    after_count = after_usage['artifact_count']

    freed_mb = before_usage['total_mb'] - after_usage['total_mb']

    if deleted_count > 0:
        logger.info(
            f"Cleanup complete: Deleted {deleted_count} artifacts, "
            f"freed {freed_mb:.2f} MB "
            f"({before_count} → {after_count} artifacts remaining)"
        )
    else:
        logger.info("No cleanup needed")


def cmd_status(args):
    """Show artifacts disk usage status."""
    config = get_config()
    artifacts_dir = Path(args.artifacts_dir or config.model.artifacts_dir)

    if not artifacts_dir.exists():
        print(f"Artifacts directory does not exist: {artifacts_dir}")
        return

    manager = ArtifactManager(artifacts_dir)
    usage = manager.get_artifacts_disk_usage()

    print("=" * 60)
    print("ARTIFACT STORAGE STATUS")
    print("=" * 60)
    print(f"Directory: {usage['artifacts_dir']}")
    print(f"Artifacts: {usage['artifact_count']}")
    print(f"Files:     {usage['file_count']}")
    print(f"Total Size: {usage['total_gb']:.2f} GB ({usage['total_mb']:.2f} MB)")
    print("=" * 60)

    # Warning thresholds
    if usage['total_gb'] > 5.0:
        print("⚠️  WARNING: Artifacts directory is large (> 5 GB)")
        print("   Consider running: python -m forex_diffusion.cli.artifacts clean")
    elif usage['total_gb'] > 10.0:
        print("❌ ERROR: Artifacts directory is very large (> 10 GB)")
        print("   Run cleanup immediately: python -m forex_diffusion.cli.artifacts clean --keep-best 5")


def cmd_list(args):
    """List all artifacts."""
    config = get_config()
    artifacts_dir = Path(args.artifacts_dir or config.model.artifacts_dir)

    if not artifacts_dir.exists():
        print(f"Artifacts directory does not exist: {artifacts_dir}")
        return

    manager = ArtifactManager(artifacts_dir)
    artifacts = manager.list_artifacts(limit=args.limit)

    if not artifacts:
        print("No artifacts found")
        return

    print(f"Found {len(artifacts)} artifacts:")
    print("-" * 80)

    for i, artifact in enumerate(artifacts, 1):
        tags_str = ", ".join(artifact.get('tags', [])) if artifact.get('tags') else "none"
        print(f"{i}. {artifact['id']}")
        print(f"   Version: {artifact['version']}")
        print(f"   Created: {artifact['created_at']}")
        print(f"   Tags: {tags_str}")
        print(f"   Path: {artifact['checkpoint_path']}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Artifact management CLI (PROC-003)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--artifacts-dir',
        type=str,
        help='Artifacts directory (default: from config)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Clean command
    parser_clean = subparsers.add_parser('clean', help='Clean up old artifacts')
    parser_clean.add_argument(
        '--keep-best',
        type=int,
        help='Maximum number of artifacts to keep (default: from config)'
    )
    parser_clean.add_argument(
        '--no-protect',
        action='store_true',
        help='Do not protect artifacts tagged as "best" or "production"'
    )
    parser_clean.set_defaults(func=cmd_clean)

    # Status command
    parser_status = subparsers.add_parser('status', help='Show disk usage status')
    parser_status.set_defaults(func=cmd_status)

    # List command
    parser_list = subparsers.add_parser('list', help='List all artifacts')
    parser_list.add_argument(
        '--limit',
        type=int,
        default=50,
        help='Maximum number of artifacts to list (default: 50)'
    )
    parser_list.set_defaults(func=cmd_list)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
