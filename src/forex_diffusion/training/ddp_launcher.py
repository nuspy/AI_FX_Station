"""
DDP (Distributed Data Parallel) launcher for multi-GPU training.

Handles process spawning, distributed setup, and checkpoint synchronization
for flexible 1-N GPU configuration.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger

from .optimization_config import OptimizationConfig, DistributedBackend


def setup_distributed(rank: int, world_size: int, backend: str, master_port: int = 29500) -> None:
    """
    Initialize distributed process group.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        backend: Backend to use ("nccl" or "gloo")
        master_port: Port for master process communication
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )

    # Set device for this process
    if backend == "nccl":
        torch.cuda.set_device(rank)

    logger.info(f"Initialized DDP: rank={rank}, world_size={world_size}, backend={backend}")


def cleanup_distributed() -> None:
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def ddp_worker(
    rank: int,
    world_size: int,
    opt_config: OptimizationConfig,
    train_fn: Callable,
    train_fn_kwargs: Dict[str, Any]
) -> None:
    """
    Worker function for DDP training.

    Each GPU spawns this function in a separate process.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        opt_config: Optimization configuration
        train_fn: Training function to execute
        train_fn_kwargs: Keyword arguments for training function
    """
    try:
        # Setup distributed
        backend = opt_config.distributed_backend.value
        setup_distributed(rank, world_size, backend)

        # Only log from rank 0
        if rank != 0:
            logger.remove()  # Remove default logger
            logger.add(
                sys.stderr,
                level="WARNING",
                filter=lambda record: record["extra"].get("rank") == 0
            )

        logger.info(f"[Rank {rank}] Starting training worker")

        # Add rank to kwargs
        train_fn_kwargs['rank'] = rank
        train_fn_kwargs['world_size'] = world_size
        train_fn_kwargs['opt_config'] = opt_config

        # Execute training function
        train_fn(**train_fn_kwargs)

        logger.info(f"[Rank {rank}] Training worker completed")

    except Exception as e:
        logger.error(f"[Rank {rank}] Training worker failed: {e}")
        raise

    finally:
        cleanup_distributed()


def launch_ddp_training(
    train_fn: Callable,
    opt_config: OptimizationConfig,
    **train_fn_kwargs
) -> None:
    """
    Launch DDP training across multiple GPUs.

    Spawns one process per GPU and coordinates training.

    Args:
        train_fn: Training function with signature:
                  train_fn(rank: int, world_size: int, opt_config: OptimizationConfig, **kwargs)
        opt_config: Optimization configuration
        **train_fn_kwargs: Additional keyword arguments for training function

    Example:
        >>> def train_worker(rank, world_size, opt_config, model, dataset, ...):
        >>>     # Training code here
        >>>     pass
        >>>
        >>> opt_config = OptimizationConfig(num_gpus=4, use_ddp=True)
        >>> launch_ddp_training(train_worker, opt_config, model=my_model, dataset=my_dataset)
    """
    if not opt_config.use_ddp:
        logger.warning("DDP not enabled in config, running single-process training")
        train_fn_kwargs['rank'] = 0
        train_fn_kwargs['world_size'] = 1
        train_fn_kwargs['opt_config'] = opt_config
        train_fn(**train_fn_kwargs)
        return

    world_size = opt_config.num_gpus

    if world_size < 2:
        logger.warning("DDP requires at least 2 GPUs, running single-process training")
        train_fn_kwargs['rank'] = 0
        train_fn_kwargs['world_size'] = 1
        train_fn_kwargs['opt_config'] = opt_config
        train_fn(**train_fn_kwargs)
        return

    logger.info(f"Launching DDP training with {world_size} GPUs")
    logger.info(f"Backend: {opt_config.distributed_backend.value}")

    # Spawn processes
    mp.spawn(
        ddp_worker,
        args=(world_size, opt_config, train_fn, train_fn_kwargs),
        nprocs=world_size,
        join=True
    )

    logger.info("DDP training completed")


class DDPCheckpointManager:
    """
    Manages checkpoints in DDP training.

    Only rank 0 saves checkpoints to avoid conflicts.
    All ranks can load checkpoints.
    """

    def __init__(self, rank: int, checkpoint_dir: Path):
        self.rank = rank
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Save checkpoint (only on rank 0).

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Training metrics
            filename: Optional checkpoint filename

        Returns:
            Path to saved checkpoint (None if not rank 0)
        """
        if self.rank != 0:
            return None

        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        # Get model state dict (unwrap DDP if needed)
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"[Rank 0] Saved checkpoint: {checkpoint_path}")

        return checkpoint_path

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint (all ranks).

        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            checkpoint_path: Path to checkpoint (if None, loads latest)

        Returns:
            Checkpoint data dictionary
        """
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
            checkpoint_path = checkpoints[-1]

        logger.info(f"[Rank {self.rank}] Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state (handle DDP wrapper)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"[Rank {self.rank}] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

        return checkpoint

    def barrier(self) -> None:
        """Synchronization barrier (waits for all processes)."""
        if dist.is_initialized():
            dist.barrier()


def get_ddp_sampler(
    dataset: torch.utils.data.Dataset,
    rank: int,
    world_size: int,
    shuffle: bool = True,
    seed: int = 0
) -> torch.utils.data.distributed.DistributedSampler:
    """
    Create a DistributedSampler for DDP training.

    Ensures each GPU sees different data without overlap.

    Args:
        dataset: PyTorch dataset
        rank: Process rank
        world_size: Total number of processes
        shuffle: Whether to shuffle data
        seed: Random seed for reproducibility

    Returns:
        DistributedSampler
    """
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed
    )

    return sampler


def reduce_tensor(tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """
    Reduce tensor across all processes in DDP.

    Useful for aggregating metrics (loss, accuracy, etc.) across GPUs.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation ("mean" or "sum")

    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor

    # Clone to avoid modifying original
    reduced = tensor.clone()

    # All-reduce
    if op == "mean":
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= dist.get_world_size()
    elif op == "sum":
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    else:
        raise ValueError(f"Unknown reduction op: {op}")

    return reduced


def print_rank_0(message: str) -> None:
    """Print message only from rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message)


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
