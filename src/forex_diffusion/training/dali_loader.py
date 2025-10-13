"""
NVIDIA DALI DataLoader integration for GPU-accelerated data preprocessing.

DALI (Data Loading Library) offloads data preprocessing to GPU, reducing
CPU bottleneck and improving training throughput.
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Iterator
from pathlib import Path
import numpy as np
import torch
from loguru import logger


class DALIWrapper:
    """
    Wrapper for NVIDIA DALI pipeline.

    Provides GPU-accelerated data loading and preprocessing for time series data.
    Falls back to standard PyTorch DataLoader if DALI is not available.
    """

    def __init__(
        self,
        data_path: Path,
        batch_size: int,
        num_threads: int = 4,
        device_id: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.shuffle = shuffle
        self.seed = seed

        # Check if DALI is available
        self.use_dali = self._check_dali()

        if not self.use_dali:
            logger.warning("DALI not available, will use standard DataLoader")

    def _check_dali(self) -> bool:
        """Check if DALI can be used."""
        if not torch.cuda.is_available():
            return False

        try:
            import nvidia.dali

            return True
        except ImportError:
            logger.warning("nvidia-dali library not installed")
            return False

    def create_pipeline(self) -> Optional[object]:
        """
        Create DALI pipeline for time series data.

        Returns:
            DALI pipeline or None if DALI not available
        """
        if not self.use_dali:
            return None

        try:
            from nvidia.dali import pipeline_def
            from nvidia.dali import fn
            from nvidia.dali import types

            @pipeline_def
            def time_series_pipeline():
                """DALI pipeline for time series data."""
                # Read data from numpy files
                # Note: This is a simplified example - adapt to your data format
                data = fn.readers.numpy(
                    device="cpu",
                    file_root=str(self.data_path),
                    random_shuffle=self.shuffle,
                    seed=self.seed,
                    shard_id=0,
                    num_shards=1,
                )

                # Move to GPU
                data = data.gpu()

                # Normalize (example - customize as needed)
                # data = fn.normalize(data, device="gpu")

                return data

            # Build pipeline
            pipe = time_series_pipeline(
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                device_id=self.device_id,
            )
            pipe.build()

            logger.info(
                f"Created DALI pipeline: batch_size={self.batch_size}, "
                f"threads={self.num_threads}, device_id={self.device_id}"
            )

            return pipe

        except Exception as e:
            logger.warning(f"Failed to create DALI pipeline: {e}")
            return None


class DALIGenericIterator:
    """
    Generic DALI iterator that mimics PyTorch DataLoader interface.

    This allows drop-in replacement of DataLoader with DALI pipeline.
    """

    def __init__(
        self,
        pipeline,
        output_map: List[str] = None,
        size: int = -1,
        auto_reset: bool = True,
        fill_last_batch: bool = True,
        last_batch_padded: bool = False,
    ):
        """
        Initialize DALI iterator.

        Args:
            pipeline: DALI pipeline
            output_map: List of output names (e.g., ["data", "label"])
            size: Number of samples (use -1 for auto-detection)
            auto_reset: Auto-reset pipeline after each epoch
            fill_last_batch: Fill last batch with dummy data if incomplete
            last_batch_padded: Whether last batch is padded
        """
        self.pipeline = pipeline
        self.output_map = output_map or ["data"]
        self.size = size
        self.auto_reset = auto_reset
        self.fill_last_batch = fill_last_batch
        self.last_batch_padded = last_batch_padded

        # Import DALI iterator
        try:
            from nvidia.dali.plugin.pytorch import DALIGenericIterator as _DALIIterator

            self.iterator_class = _DALIIterator
        except ImportError:
            raise ImportError("nvidia-dali library required for DALIGenericIterator")

        # Create internal iterator
        self._create_iterator()

    def _create_iterator(self):
        """Create internal DALI iterator."""
        self._iterator = self.iterator_class(
            self.pipeline,
            self.output_map,
            size=self.size,
            auto_reset=self.auto_reset,
            fill_last_batch=self.fill_last_batch,
            last_batch_padded=self.last_batch_padded,
        )

    def __iter__(self) -> Iterator:
        """Return iterator."""
        return self

    def __next__(self) -> dict:
        """Get next batch."""
        try:
            batch = next(self._iterator)
            # Convert DALI batch to PyTorch tensors
            return {k: v[0] for k, v in batch[0].items()}
        except StopIteration:
            if self.auto_reset:
                self._create_iterator()
            raise

    def __len__(self) -> int:
        """Return number of batches."""
        if self.size > 0:
            batch_size = self.pipeline.max_batch_size
            return (self.size + batch_size - 1) // batch_size
        return 0

    def reset(self):
        """Reset iterator."""
        self._create_iterator()


def create_dali_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 4,
    device_id: int = 0,
    shuffle: bool = True,
    seed: int = 0,
) -> Optional[DALIGenericIterator]:
    """
    Create a DALI DataLoader for PyTorch dataset.

    Note: This requires the dataset to be compatible with DALI.
    For most use cases, standard DataLoader is sufficient.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        num_workers: Number of worker threads
        device_id: GPU device ID
        shuffle: Whether to shuffle data
        seed: Random seed

    Returns:
        DALI iterator or None if DALI not available
    """
    # Check DALI availability
    try:
        import nvidia.dali
    except ImportError:
        logger.warning("DALI not available")
        return None

    # For now, return None and use standard DataLoader
    # DALI integration requires specific data format (e.g., TFRecord, LMDB)
    # For time series data, standard DataLoader is usually sufficient
    logger.info(
        "DALI integration requires custom pipeline for your data format. "
        "Using standard PyTorch DataLoader instead."
    )
    return None


def benchmark_dataloader(
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 100,
    device: str = "cuda",
) -> dict:
    """
    Benchmark DataLoader performance.

    Useful for comparing DALI vs standard DataLoader.

    Args:
        dataloader: DataLoader to benchmark
        num_batches: Number of batches to time
        device: Device to transfer data to

    Returns:
        Dictionary with benchmark results
    """
    import time

    device = torch.device(device)
    times = []

    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break
        if isinstance(batch, dict) and "x" in batch:
            _ = batch["x"].to(device)

    # Benchmark
    start = time.time()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        batch_start = time.time()
        if isinstance(batch, dict) and "x" in batch:
            _ = batch["x"].to(device)
        batch_end = time.time()

        times.append(batch_end - batch_start)

    total_time = time.time() - start

    results = {
        "total_time": total_time,
        "avg_batch_time": np.mean(times),
        "std_batch_time": np.std(times),
        "min_batch_time": np.min(times),
        "max_batch_time": np.max(times),
        "throughput_batches_per_sec": num_batches / total_time,
    }

    logger.info("DataLoader Benchmark Results:")
    logger.info(f"  Total time: {total_time:.3f}s")
    logger.info(f"  Avg batch time: {results['avg_batch_time']*1000:.2f}ms")
    logger.info(f"  Throughput: {results['throughput_batches_per_sec']:.2f} batches/s")

    return results


# Example DALI pipeline for financial time series (template)
def create_financial_dali_pipeline(
    data_dir: Path,
    batch_size: int,
    num_threads: int = 4,
    device_id: int = 0,
    sequence_length: int = 64,
    num_features: int = 7,
):
    """
    Template for creating DALI pipeline for financial time series.

    This is a template - customize based on your data format.

    Args:
        data_dir: Directory containing data files
        batch_size: Batch size
        num_threads: Number of CPU threads
        device_id: GPU device ID
        sequence_length: Length of sequences
        num_features: Number of features per timestep

    Returns:
        DALI pipeline
    """
    try:
        from nvidia.dali import pipeline_def
        from nvidia.dali import fn
        from nvidia.dali import types

        @pipeline_def
        def financial_pipeline():
            """
            DALI pipeline for financial time series.

            Expected data format:
            - Files: numpy arrays of shape (N, sequence_length, num_features)
            - Stored as .npy files in data_dir
            """
            # Read numpy files
            # Note: Adapt this to your actual data format
            data = fn.readers.numpy(
                device="cpu",
                file_root=str(data_dir),
                file_filter="*.npy",
                random_shuffle=True,
                pad_last_batch=True,
            )

            # Move to GPU
            data = data.gpu()

            # Normalize (per-feature standardization)
            # Note: You may want to use precomputed statistics
            # mean = fn.reductions.mean(data, axes=[0, 1])
            # std = fn.reductions.std(data, axes=[0, 1])
            # data = fn.normalize(data, mean=mean, stddev=std)

            return data

        # Build pipeline
        pipe = financial_pipeline(
            batch_size=batch_size, num_threads=num_threads, device_id=device_id
        )
        pipe.build()

        logger.info("Created financial DALI pipeline")
        return pipe

    except ImportError:
        logger.error("DALI not available")
        return None
    except Exception as e:
        logger.error(f"Failed to create DALI pipeline: {e}")
        return None
