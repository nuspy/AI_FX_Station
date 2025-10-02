"""
GPU/CPU Device Management for ForexGPT.

Provides centralized device selection and information for training and inference.
"""
from __future__ import annotations

import torch
from typing import Optional, Literal, Dict, Any
from loguru import logger

DeviceType = Literal["auto", "cuda", "cpu", "mps"]


class DeviceManager:
    """Centralized GPU/CPU device management."""

    @staticmethod
    def get_device(preference: DeviceType = "auto") -> torch.device:
        """
        Get optimal device based on preference and availability.

        Args:
            preference: Device preference ("auto", "cuda", "cpu", "mps")
                - "auto": Use CUDA if available, otherwise CPU
                - "cuda": Force CUDA (raises error if unavailable)
                - "cpu": Force CPU
                - "mps": Use Apple Metal Performance Shaders (Mac only)

        Returns:
            torch.device object

        Raises:
            RuntimeError: If CUDA/MPS requested but not available
        """
        if preference == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available. Install CUDA-enabled PyTorch.")
            return torch.device("cuda")

        if preference == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                return device

        if preference == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Using Apple MPS (Metal Performance Shaders)")
                return torch.device("mps")
            else:
                raise RuntimeError("MPS requested but not available (requires Apple Silicon Mac)")

        logger.info("Using CPU")
        return torch.device("cpu")

    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """
        Get comprehensive device information for UI display.

        Returns:
            Dictionary with device capabilities:
            {
                "cuda_available": bool,
                "cuda_device_count": int,
                "cuda_device_name": str or None,
                "cuda_version": str or None,
                "cuda_memory_total_gb": float or None,
                "cuda_memory_free_gb": float or None,
                "mps_available": bool,
                "cpu_count": int,
                "recommended_device": str
            }
        """
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "cuda_version": None,
            "cuda_memory_total_gb": None,
            "cuda_memory_free_gb": None,
            "mps_available": False,
            "cpu_count": torch.get_num_threads(),
            "recommended_device": "cpu"
        }

        # CUDA info
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda

            # GPU memory info
            try:
                props = torch.cuda.get_device_properties(0)
                total_memory = props.total_memory / (1024**3)  # Convert to GB
                info["cuda_memory_total_gb"] = round(total_memory, 2)

                # Free memory
                torch.cuda.empty_cache()
                free_memory = torch.cuda.mem_get_info()[0] / (1024**3)
                info["cuda_memory_free_gb"] = round(free_memory, 2)
            except Exception as e:
                logger.warning(f"Could not get CUDA memory info: {e}")

            info["recommended_device"] = "cuda"

        # MPS info (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["mps_available"] = True
            if not info["cuda_available"]:
                info["recommended_device"] = "mps"

        return info

    @staticmethod
    def log_device_info():
        """Log device information for debugging."""
        info = DeviceManager.get_device_info()

        logger.info("=" * 50)
        logger.info("GPU/CPU Device Information")
        logger.info("=" * 50)

        if info["cuda_available"]:
            logger.info(f"✅ CUDA Available: YES")
            logger.info(f"   GPU: {info['cuda_device_name']}")
            logger.info(f"   CUDA Version: {info['cuda_version']}")
            logger.info(f"   Device Count: {info['cuda_device_count']}")
            if info["cuda_memory_total_gb"]:
                logger.info(f"   Total Memory: {info['cuda_memory_total_gb']} GB")
                logger.info(f"   Free Memory: {info['cuda_memory_free_gb']} GB")
        else:
            logger.info(f"❌ CUDA Available: NO")

        if info["mps_available"]:
            logger.info(f"✅ Apple MPS Available: YES")
        else:
            logger.info(f"❌ Apple MPS Available: NO")

        logger.info(f"💻 CPU Threads: {info['cpu_count']}")
        logger.info(f"🎯 Recommended Device: {info['recommended_device'].upper()}")
        logger.info("=" * 50)

    @staticmethod
    def get_optimal_batch_size(device: torch.device, model_size_mb: float = 100) -> int:
        """
        Estimate optimal batch size based on available GPU memory.

        Args:
            device: torch.device object
            model_size_mb: Estimated model size in MB

        Returns:
            Recommended batch size
        """
        if device.type == "cuda":
            try:
                free_memory_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                # Use 80% of free memory
                usable_memory_gb = free_memory_gb * 0.8
                # Estimate: each sample uses ~model_size/10 MB
                sample_memory_gb = (model_size_mb / 10) / 1024
                batch_size = int(usable_memory_gb / sample_memory_gb)
                return max(1, min(batch_size, 512))  # Cap at 512
            except Exception:
                return 32  # Default fallback

        return 32  # CPU default

    @staticmethod
    def move_to_device(obj, device: torch.device):
        """
        Move tensor or model to device (handles both tensors and nn.Module).

        Args:
            obj: Tensor, model, or dict/list of tensors
            device: Target device

        Returns:
            Object moved to device
        """
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, torch.nn.Module):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: DeviceManager.move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(DeviceManager.move_to_device(item, device) for item in obj)
        else:
            return obj


# Convenience function for backward compatibility
def get_device(preference: DeviceType = "auto") -> torch.device:
    """Shortcut to DeviceManager.get_device()."""
    return DeviceManager.get_device(preference)


# Auto-log device info on import (only in debug mode)
if __name__ != "__main__":
    try:
        DeviceManager.log_device_info()
    except Exception:
        pass  # Silent fail if no GPU
