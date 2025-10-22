
from __future__ import annotations

from typing import Tuple, Optional
from loguru import logger

# Global handle to NVML library to avoid re-initializing
_nvml_handle = None

def _initialize_nvml():
    """Initialize the NVML library."""
    global _nvml_handle
    if _nvml_handle is not None:
        return _nvml_handle

    try:
        from pynvml.smi import NVSMI
        _nvml_handle = NVSMI()
        logger.info("Successfully initialized NVML for GPU monitoring.")
        return _nvml_handle
    except Exception as e:
        logger.warning(f"Could not initialize NVML for GPU monitoring: {e}. VRAM display will be disabled.")
        logger.warning("Please ensure 'pynvml' is installed (`pip install pynvml`) and NVIDIA drivers are up to date.")
        _nvml_handle = None
        return None

def get_vram_usage(gpu_index: int = 0) -> Optional[Tuple[float, float]]:
    """
    Get the used and total VRAM for a specific GPU in gigabytes.

    Args:
        gpu_index: The index of the GPU to query.

    Returns:
        A tuple of (used_vram_gb, total_vram_gb), or None if NVML is not available.
    """
    nvml = _initialize_nvml()
    if nvml is None:
        return None

    try:
        # The NVSMI object from pynvml provides a list of GPUs
        # Note: This might be slow if called very frequently, but NVSMI is a singleton.
        gpu = nvml.DeviceQuery()['gpu'][gpu_index]
        
        # Values are in MiB, convert to GiB
        total_vram_gb = gpu['fb_memory_total'] / 1024.0
        used_vram_gb = gpu['fb_memory_used'] / 1024.0
        
        return round(used_vram_gb, 2), round(total_vram_gb, 2)
    except IndexError:
        logger.warning(f"GPU with index {gpu_index} not found.")
        return None
    except Exception as e:
        logger.error(f"Failed to get VRAM usage: {e}")
        return None
