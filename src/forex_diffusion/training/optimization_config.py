"""
Training optimization configuration.

Manages all NVIDIA optimization settings with hardware auto-detection.
Supports 1-N GPUs with graceful degradation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import torch
from loguru import logger


class PrecisionMode(Enum):
    """Training precision modes."""
    FP32 = "fp32"  # Full precision
    FP16 = "fp16"  # Mixed precision (AMP)
    BF16 = "bf16"  # BFloat16 (Ampere+)


class CompileMode(Enum):
    """torch.compile modes."""
    DISABLED = "disabled"
    DEFAULT = "default"  # Balanced
    REDUCE_OVERHEAD = "reduce-overhead"  # Fast compilation
    MAX_AUTOTUNE = "max-autotune"  # Slow compilation, max speed


class DistributedBackend(Enum):
    """Distributed training backends."""
    NONE = "none"  # Single GPU
    NCCL = "nccl"  # NVIDIA GPUs (fastest)
    GLOO = "gloo"  # CPU or mixed (more stable)


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""

    # GPU detection
    num_gpus: int = 0
    gpu_names: List[str] = field(default_factory=list)
    gpu_memory_gb: List[float] = field(default_factory=list)
    gpu_compute_capabilities: List[tuple] = field(default_factory=list)
    has_nvlink: bool = False

    # CPU detection
    num_cpu_cores: int = 0
    total_ram_gb: float = 0.0

    # Library availability
    has_cuda: bool = False
    has_apex: bool = False
    has_flash_attn: bool = False
    has_dali: bool = False
    has_nccl: bool = False

    # CUDA info
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None

    def __post_init__(self):
        """Detect hardware on initialization."""
        self.detect_hardware()

    def detect_hardware(self):
        """Detect all available hardware and libraries."""
        # Detect GPUs
        self.has_cuda = torch.cuda.is_available()

        if self.has_cuda:
            self.num_gpus = torch.cuda.device_count()

            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                self.gpu_names.append(props.name)
                self.gpu_memory_gb.append(props.total_memory / (1024**3))
                self.gpu_compute_capabilities.append((props.major, props.minor))

            # Check CUDA/cuDNN versions
            self.cuda_version = torch.version.cuda
            self.cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None

            # Check NVLink (simplified check)
            try:
                import pynvml
                pynvml.nvmlInit()
                # Check if GPUs have NVLink connections
                if self.num_gpus > 1:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    # Try to get NVLink state (if supported)
                    try:
                        for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
                            state = pynvml.nvmlDeviceGetNvLinkState(handle, link)
                            if state == pynvml.NVML_FEATURE_ENABLED:
                                self.has_nvlink = True
                                break
                    except:
                        pass
                pynvml.nvmlShutdown()
            except:
                pass

        # Detect CPU
        try:
            import psutil
            self.num_cpu_cores = psutil.cpu_count(logical=False)
            self.total_ram_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            import os
            self.num_cpu_cores = os.cpu_count() or 1

        # Detect APEX
        try:
            import apex
            self.has_apex = True
        except ImportError:
            self.has_apex = False

        # Detect Flash Attention
        try:
            import flash_attn
            self.has_flash_attn = True
        except ImportError:
            self.has_flash_attn = False

        # Detect DALI
        try:
            import nvidia.dali
            self.has_dali = True
        except ImportError:
            self.has_dali = False

        # Detect NCCL
        self.has_nccl = torch.distributed.is_nccl_available() if self.has_cuda else False

    def supports_mixed_precision(self, device_id: int = 0) -> bool:
        """Check if device supports mixed precision (FP16)."""
        if not self.has_cuda or device_id >= self.num_gpus:
            return False

        # Requires compute capability >= 7.0 (Volta+)
        major, minor = self.gpu_compute_capabilities[device_id]
        return major >= 7

    def supports_bfloat16(self, device_id: int = 0) -> bool:
        """Check if device supports BFloat16."""
        if not self.has_cuda or device_id >= self.num_gpus:
            return False

        # Requires compute capability >= 8.0 (Ampere+)
        major, minor = self.gpu_compute_capabilities[device_id]
        return major >= 8

    def supports_flash_attention(self, device_id: int = 0) -> bool:
        """Check if device supports Flash Attention 2."""
        if not self.has_cuda or device_id >= self.num_gpus or not self.has_flash_attn:
            return False

        # Requires Ampere+ (compute capability >= 8.0)
        major, minor = self.gpu_compute_capabilities[device_id]
        return major >= 8

    def get_recommended_batch_size(self, device_id: int = 0, model_size_mb: float = 500) -> int:
        """Recommend batch size based on GPU memory."""
        if not self.has_cuda or device_id >= self.num_gpus:
            return 32  # Conservative for CPU

        available_memory_gb = self.gpu_memory_gb[device_id]

        # Heuristic: reserve 20% for overhead, allocate rest to batch
        usable_memory_gb = available_memory_gb * 0.8
        model_memory_gb = model_size_mb / 1024

        # Estimate: each batch item uses roughly same memory as model
        estimated_batch_size = int((usable_memory_gb - model_memory_gb) / (model_size_mb / 1024))

        # Clamp to reasonable range
        batch_size = max(8, min(512, estimated_batch_size))

        # Round to power of 2 for efficiency
        import math
        batch_size = 2 ** int(math.log2(batch_size))

        return batch_size

    def log_hardware_info(self):
        """Log detected hardware information."""
        logger.info("=== Hardware Detection ===")
        logger.info(f"CUDA Available: {self.has_cuda}")

        if self.has_cuda:
            logger.info(f"CUDA Version: {self.cuda_version}")
            logger.info(f"cuDNN Version: {self.cudnn_version}")
            logger.info(f"Number of GPUs: {self.num_gpus}")

            for i in range(self.num_gpus):
                logger.info(f"GPU {i}: {self.gpu_names[i]}")
                logger.info(f"  Memory: {self.gpu_memory_gb[i]:.1f} GB")
                logger.info(f"  Compute Capability: {self.gpu_compute_capabilities[i][0]}.{self.gpu_compute_capabilities[i][1]}")
                logger.info(f"  FP16 Support: {self.supports_mixed_precision(i)}")
                logger.info(f"  BF16 Support: {self.supports_bfloat16(i)}")
                logger.info(f"  Flash Attention Support: {self.supports_flash_attention(i)}")

            logger.info(f"NVLink Detected: {self.has_nvlink}")

        logger.info(f"CPU Cores: {self.num_cpu_cores}")
        logger.info(f"RAM: {self.total_ram_gb:.1f} GB")
        logger.info(f"APEX Available: {self.has_apex}")
        logger.info(f"Flash Attention Available: {self.has_flash_attn}")
        logger.info(f"DALI Available: {self.has_dali}")
        logger.info(f"NCCL Available: {self.has_nccl}")
        logger.info("=" * 50)


@dataclass
class OptimizationConfig:
    """Complete optimization configuration."""

    # Precision settings
    precision: PrecisionMode = PrecisionMode.FP16
    use_amp: bool = True  # Automatic Mixed Precision

    # Compilation settings
    compile_mode: CompileMode = CompileMode.DEFAULT
    compile_model: bool = True

    # Optimizer settings
    use_fused_optimizer: bool = True  # APEX fused optimizers

    # Attention settings
    use_flash_attention: bool = True

    # Memory settings
    use_channels_last: bool = True
    use_gradient_checkpointing: bool = False

    # Batch settings
    batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Distributed settings
    num_gpus: int = 1  # 1-N GPUs
    distributed_backend: DistributedBackend = DistributedBackend.NCCL
    use_ddp: bool = False

    # Data loading settings
    use_dali: bool = False
    num_workers: int = 4
    pin_memory: bool = True

    # cuDNN settings
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    # Auto-configuration
    auto_configure: bool = True
    hardware_info: Optional[HardwareInfo] = None

    def __post_init__(self):
        """Auto-configure based on hardware if requested."""
        if self.hardware_info is None:
            self.hardware_info = HardwareInfo()

        if self.auto_configure:
            self.auto_configure_optimizations()

    def auto_configure_optimizations(self):
        """Automatically configure optimizations based on hardware."""
        hw = self.hardware_info

        logger.info("Auto-configuring optimizations based on hardware...")

        # Precision configuration
        if hw.has_cuda and hw.supports_bfloat16(0):
            self.precision = PrecisionMode.BF16
            self.use_amp = True
            logger.info("Using BFloat16 precision (Ampere+ GPU detected)")
        elif hw.has_cuda and hw.supports_mixed_precision(0):
            self.precision = PrecisionMode.FP16
            self.use_amp = True
            logger.info("Using FP16 mixed precision")
        else:
            self.precision = PrecisionMode.FP32
            self.use_amp = False
            logger.info("Using FP32 precision (GPU doesn't support FP16)")

        # Compilation
        if torch.__version__ >= "2.0":
            self.compile_model = True
            self.compile_mode = CompileMode.REDUCE_OVERHEAD  # Fast compilation
            logger.info("Enabling torch.compile (PyTorch 2.0+ detected)")
        else:
            self.compile_model = False
            logger.warning("torch.compile disabled (requires PyTorch 2.0+)")

        # Fused optimizers
        if hw.has_cuda and hw.has_apex:
            self.use_fused_optimizer = True
            logger.info("Enabling APEX fused optimizers")
        else:
            self.use_fused_optimizer = False
            if hw.has_cuda:
                logger.warning("APEX not available, using standard optimizers")

        # Flash Attention
        if hw.has_cuda and hw.supports_flash_attention(0):
            self.use_flash_attention = True
            logger.info("Enabling Flash Attention 2")
        else:
            self.use_flash_attention = False
            if hw.has_cuda:
                logger.warning("Flash Attention not available or GPU not supported")

        # Distributed training
        if self.num_gpus > 1 and hw.num_gpus > 1:
            self.use_ddp = True

            # Choose backend
            if hw.has_nccl:
                self.distributed_backend = DistributedBackend.NCCL
                logger.info(f"Enabling DDP with NCCL backend ({self.num_gpus} GPUs)")
            else:
                self.distributed_backend = DistributedBackend.GLOO
                logger.info(f"Enabling DDP with GLOO backend ({self.num_gpus} GPUs)")
        else:
            self.use_ddp = False
            self.num_gpus = 1
            logger.info("Single GPU training")

        # DALI DataLoader
        if hw.has_cuda and hw.has_dali:
            self.use_dali = True
            logger.info("DALI DataLoader available (can be enabled)")
        else:
            self.use_dali = False

        # cuDNN
        if hw.has_cuda:
            self.cudnn_benchmark = True  # Enable auto-tuner
            logger.info("Enabling cuDNN benchmark auto-tuner")

        # Channels last (for Conv models only)
        if hw.has_cuda and hw.supports_bfloat16(0):
            self.use_channels_last = True
            logger.info("Enabling channels_last memory format")
        else:
            self.use_channels_last = False

        # Batch size recommendation
        if hw.has_cuda:
            recommended_bs = hw.get_recommended_batch_size(0)
            if self.batch_size == 64:  # Default value
                self.batch_size = recommended_bs
                logger.info(f"Auto-configured batch size: {self.batch_size}")

        # Gradient checkpointing (off by default, enable if OOM)
        self.use_gradient_checkpointing = False

        logger.info("Auto-configuration complete")

    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation and DDP."""
        effective = self.batch_size * self.gradient_accumulation_steps

        if self.use_ddp:
            effective *= self.num_gpus

        return effective

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "precision": self.precision.value,
            "use_amp": self.use_amp,
            "compile_mode": self.compile_mode.value,
            "compile_model": self.compile_model,
            "use_fused_optimizer": self.use_fused_optimizer,
            "use_flash_attention": self.use_flash_attention,
            "use_channels_last": self.use_channels_last,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_gpus": self.num_gpus,
            "distributed_backend": self.distributed_backend.value,
            "use_ddp": self.use_ddp,
            "use_dali": self.use_dali,
            "num_workers": self.num_workers,
            "cudnn_benchmark": self.cudnn_benchmark,
            "effective_batch_size": self.get_effective_batch_size(),
        }

    def log_config(self):
        """Log current optimization configuration."""
        logger.info("=== Optimization Configuration ===")
        config_dict = self.to_dict()

        for key, value in config_dict.items():
            logger.info(f"{key}: {value}")

        logger.info("=" * 50)


def get_optimization_config(num_gpus: int = 1, auto_configure: bool = True) -> OptimizationConfig:
    """Factory function to create optimization configuration."""
    config = OptimizationConfig(
        num_gpus=num_gpus,
        auto_configure=auto_configure
    )

    return config
