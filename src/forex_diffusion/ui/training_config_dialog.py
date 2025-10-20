"""
Training Configuration Dialog - Hardware/GPU Selection and Advanced Settings

Provides a comprehensive dialog for configuring training hardware settings,
GPU optimizations, and distributed training options.
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QCheckBox, QSpinBox, QTabWidget, QWidget, QFormLayout, QTextEdit, QMessageBox
)
from PySide6.QtCore import Signal

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TrainingConfigDialog(QDialog):
    """
    Advanced training configuration dialog for hardware/GPU selection.

    Features:
    - GPU detection and selection
    - NVIDIA optimization stack configuration
    - Distributed training setup
    - Performance estimation
    - Hardware capability checking
    """

    config_accepted = Signal(dict)  # Emits configuration when accepted

    def __init__(self, parent=None, current_config: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.current_config = current_config or {}
        self.detected_hardware = self._detect_hardware()

        self.setWindowTitle("Training Hardware Configuration")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)

        self._build_ui()
        self._load_current_config()

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware and capabilities"""
        hw_info = {
            'torch_available': TORCH_AVAILABLE,
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_devices': [],
            'cuda_version': None,
            'cudnn_version': None,
            'compute_capabilities': [],
            'total_vram_gb': 0,
        }

        if not TORCH_AVAILABLE:
            return hw_info

        hw_info['cuda_available'] = torch.cuda.is_available()

        if hw_info['cuda_available']:
            hw_info['gpu_count'] = torch.cuda.device_count()
            hw_info['cuda_version'] = torch.version.cuda
            hw_info['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None

            for i in range(hw_info['gpu_count']):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    'index': i,
                    'name': props.name,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'total_memory_gb': props.total_memory / (1024**3),
                    'multi_processor_count': props.multi_processor_count,
                }
                hw_info['gpu_devices'].append(device_info)
                hw_info['compute_capabilities'].append((props.major, props.minor))
                hw_info['total_vram_gb'] += device_info['total_memory_gb']

        return hw_info

    def _build_ui(self):
        """Build the dialog UI"""
        layout = QVBoxLayout(self)

        # Tab widget for different configuration sections
        tabs = QTabWidget()

        # Tab 1: Hardware Selection
        hardware_tab = self._build_hardware_tab()
        tabs.addTab(hardware_tab, "Hardware Selection")

        # Tab 2: GPU Optimizations
        gpu_opts_tab = self._build_gpu_optimizations_tab()
        tabs.addTab(gpu_opts_tab, "GPU Optimizations")

        # Tab 3: Distributed Training
        distributed_tab = self._build_distributed_tab()
        tabs.addTab(distributed_tab, "Distributed Training")

        # Tab 4: Performance Estimates
        performance_tab = self._build_performance_tab()
        tabs.addTab(performance_tab, "Performance Info")

        layout.addWidget(tabs)

        # Action buttons
        buttons = QHBoxLayout()

        self.test_btn = QPushButton("Test Configuration")
        self.test_btn.clicked.connect(self._test_configuration)

        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_to_defaults)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self._apply_configuration)

        buttons.addWidget(self.test_btn)
        buttons.addWidget(self.reset_btn)
        buttons.addStretch()
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.apply_btn)

        layout.addLayout(buttons)

    def _build_hardware_tab(self) -> QWidget:
        """Build hardware selection tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # GPU Detection Info
        detection_group = QGroupBox("Detected Hardware")
        detection_layout = QFormLayout(detection_group)

        if not self.detected_hardware['torch_available']:
            detection_layout.addRow(QLabel("⚠️ PyTorch not available"))
        elif not self.detected_hardware['cuda_available']:
            detection_layout.addRow(QLabel("⚠️ CUDA not available - CPU training only"))
        else:
            detection_layout.addRow("CUDA Version:", QLabel(str(self.detected_hardware['cuda_version'])))
            detection_layout.addRow("cuDNN Version:", QLabel(str(self.detected_hardware['cudnn_version'])))
            detection_layout.addRow("GPU Count:", QLabel(str(self.detected_hardware['gpu_count'])))
            detection_layout.addRow("Total VRAM:", QLabel(f"{self.detected_hardware['total_vram_gb']:.1f} GB"))

        layout.addWidget(detection_group)

        # GPU Selection
        if self.detected_hardware['cuda_available'] and self.detected_hardware['gpu_count'] > 0:
            gpu_group = QGroupBox("GPU Selection")
            gpu_layout = QVBoxLayout(gpu_group)

            # Device selection
            device_row = QHBoxLayout()
            device_row.addWidget(QLabel("Training Device:"))

            self.device_combo = QComboBox()
            self.device_combo.addItem("CPU", "cpu")

            for gpu in self.detected_hardware['gpu_devices']:
                label = f"GPU {gpu['index']}: {gpu['name']} ({gpu['total_memory_gb']:.1f} GB)"
                self.device_combo.addItem(label, f"cuda:{gpu['index']}")

            if self.detected_hardware['gpu_count'] > 1:
                self.device_combo.addItem("All GPUs (Multi-GPU)", "cuda:all")

            self.device_combo.setCurrentIndex(1)  # Default to first GPU
            device_row.addWidget(self.device_combo)
            device_row.addStretch()

            gpu_layout.addLayout(device_row)

            # GPU info display
            self.gpu_info_text = QTextEdit()
            self.gpu_info_text.setReadOnly(True)
            self.gpu_info_text.setMaximumHeight(150)
            self._update_gpu_info()

            self.device_combo.currentIndexChanged.connect(self._update_gpu_info)

            gpu_layout.addWidget(QLabel("Selected GPU Details:"))
            gpu_layout.addWidget(self.gpu_info_text)

            layout.addWidget(gpu_group)
        else:
            self.device_combo = QComboBox()
            self.device_combo.addItem("CPU", "cpu")

        # Memory Management
        memory_group = QGroupBox("Memory Management")
        memory_layout = QFormLayout(memory_group)

        self.pin_memory = QCheckBox()
        self.pin_memory.setChecked(self.detected_hardware['cuda_available'])
        memory_layout.addRow("Pin Memory:", self.pin_memory)

        self.num_workers = QSpinBox()
        self.num_workers.setRange(0, 16)
        self.num_workers.setValue(4 if self.detected_hardware['cuda_available'] else 0)
        memory_layout.addRow("DataLoader Workers:", self.num_workers)

        self.prefetch_factor = QSpinBox()
        self.prefetch_factor.setRange(2, 10)
        self.prefetch_factor.setValue(2)
        memory_layout.addRow("Prefetch Factor:", self.prefetch_factor)

        layout.addWidget(memory_group)
        layout.addStretch()

        return widget

    def _build_gpu_optimizations_tab(self) -> QWidget:
        """Build GPU optimizations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # NVIDIA Optimization Stack
        nvidia_group = QGroupBox("NVIDIA Optimization Stack")
        nvidia_layout = QFormLayout(nvidia_group)

        self.use_amp = QCheckBox()
        self.use_amp.setChecked(True)
        self.use_amp.setToolTip("Automatic Mixed Precision - 2-3x speedup")
        nvidia_layout.addRow("AMP (Mixed Precision):", self.use_amp)

        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["fp16", "bf16", "fp32"])
        self.precision_combo.setCurrentText("fp16")
        self.precision_combo.setToolTip(
            "fp16: Fast, works on most GPUs (RTX 20xx+)\n"
            "bf16: More stable, requires Ampere+ (RTX 30xx+)\n"
            "fp32: Full precision, slowest"
        )
        nvidia_layout.addRow("Precision Mode:", self.precision_combo)

        self.compile_model = QCheckBox()
        self.compile_model.setChecked(True)
        self.compile_model.setToolTip("torch.compile - 1.5-2x speedup (PyTorch 2.0+)")
        nvidia_layout.addRow("torch.compile:", self.compile_model)

        self.use_fused_optimizer = QCheckBox()
        self.use_fused_optimizer.setChecked(False)
        self.use_fused_optimizer.setToolTip("NVIDIA APEX Fused Optimizers - requires APEX installation")
        nvidia_layout.addRow("Fused Optimizer:", self.use_fused_optimizer)

        self.use_flash_attention = QCheckBox()
        self.use_flash_attention.setChecked(False)
        self.use_flash_attention.setToolTip("Flash Attention 2 - requires Ampere+ GPU")
        nvidia_layout.addRow("Flash Attention:", self.use_flash_attention)

        layout.addWidget(nvidia_group)

        # Performance Tuning
        perf_group = QGroupBox("Performance Tuning")
        perf_layout = QFormLayout(perf_group)

        self.grad_accumulation = QSpinBox()
        self.grad_accumulation.setRange(1, 32)
        self.grad_accumulation.setValue(1)
        self.grad_accumulation.setToolTip("Accumulate gradients over N steps to simulate larger batch size")
        perf_layout.addRow("Gradient Accumulation:", self.grad_accumulation)

        self.channels_last = QCheckBox()
        self.channels_last.setChecked(True)
        self.channels_last.setToolTip("Use channels-last memory format (faster for CNNs)")
        perf_layout.addRow("Channels Last:", self.channels_last)

        self.gradient_checkpointing = QCheckBox()
        self.gradient_checkpointing.setChecked(False)
        self.gradient_checkpointing.setToolTip("Trade compute for memory (enables larger models)")
        perf_layout.addRow("Gradient Checkpointing:", self.gradient_checkpointing)

        layout.addWidget(perf_group)
        layout.addStretch()

        return widget

    def _build_distributed_tab(self) -> QWidget:
        """Build distributed training tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        distributed_group = QGroupBox("Distributed Training (Multi-GPU)")
        distributed_layout = QFormLayout(distributed_group)

        self.use_distributed = QCheckBox()
        self.use_distributed.setChecked(False)
        self.use_distributed.setEnabled(self.detected_hardware['gpu_count'] > 1)
        distributed_layout.addRow("Enable Distributed:", self.use_distributed)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["nccl", "gloo", "mpi"])
        self.backend_combo.setCurrentText("nccl")
        distributed_layout.addRow("Backend:", self.backend_combo)

        self.find_unused_parameters = QCheckBox()
        self.find_unused_parameters.setChecked(False)
        distributed_layout.addRow("Find Unused Parameters:", self.find_unused_parameters)

        layout.addWidget(distributed_group)

        info_label = QLabel(
            "ℹ️ Distributed training requires multiple GPUs.\n"
            f"Detected: {self.detected_hardware['gpu_count']} GPU(s)\n\n"
            "NCCL: Fastest for NVIDIA GPUs (recommended)\n"
            "Gloo: CPU and GPU support\n"
            "MPI: Requires MPI installation"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()

        return widget

    def _build_performance_tab(self) -> QWidget:
        """Build performance estimation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_group = QGroupBox("Performance Estimates")
        info_layout = QVBoxLayout(info_group)

        self.perf_info = QTextEdit()
        self.perf_info.setReadOnly(True)
        self._update_performance_estimates()

        info_layout.addWidget(self.perf_info)

        layout.addWidget(info_group)

        # Recommendations
        rec_group = QGroupBox("Recommendations")
        rec_layout = QVBoxLayout(rec_group)

        self.recommendations = QTextEdit()
        self.recommendations.setReadOnly(True)
        self._update_recommendations()

        rec_layout.addWidget(self.recommendations)

        layout.addWidget(rec_group)

        return widget

    def _update_gpu_info(self):
        """Update GPU information display"""
        if not hasattr(self, 'gpu_info_text'):
            return

        device_data = self.device_combo.currentData()

        if device_data == "cpu":
            self.gpu_info_text.setText("CPU training selected - no GPU acceleration")
            return

        if device_data == "cuda:all":
            info = "Multi-GPU training configuration:\n\n"
            for gpu in self.detected_hardware['gpu_devices']:
                info += f"GPU {gpu['index']}: {gpu['name']}\n"
                info += f"  Compute: {gpu['compute_capability']}\n"
                info += f"  VRAM: {gpu['total_memory_gb']:.1f} GB\n"
                info += f"  SMs: {gpu['multi_processor_count']}\n\n"
            self.gpu_info_text.setText(info)
            return

        # Single GPU
        gpu_idx = int(device_data.split(':')[1])
        gpu = self.detected_hardware['gpu_devices'][gpu_idx]

        info = f"{gpu['name']}\n\n"
        info += f"Compute Capability: {gpu['compute_capability']}\n"
        info += f"Total VRAM: {gpu['total_memory_gb']:.1f} GB\n"
        info += f"Streaming Multiprocessors: {gpu['multi_processor_count']}\n\n"

        # Capability checks
        major, minor = map(int, gpu['compute_capability'].split('.'))

        if major >= 7:
            info += "✅ Supports Tensor Cores (FP16/BF16)\n"
        if major >= 8:
            info += "✅ Supports Flash Attention 2\n"
            info += "✅ Ampere architecture or newer\n"

        self.gpu_info_text.setText(info)

    def _update_performance_estimates(self):
        """Update performance estimates"""
        if not hasattr(self, 'perf_info'):
            return

        estimates = "Performance Estimates (approximate):\n\n"

        if self.detected_hardware['cuda_available']:
            base_speed = "1.0x (baseline)"

            if hasattr(self, 'use_amp') and self.use_amp.isChecked():
                base_speed = "2.0-3.0x (with AMP)"

            if hasattr(self, 'compile_model') and self.compile_model.isChecked():
                base_speed += " × 1.5x (with compile)"

            estimates += f"Expected Speedup: {base_speed}\n\n"

            # Memory estimates
            gpu_count = len([d for d in self.detected_hardware['gpu_devices']])
            if gpu_count > 0:
                avg_vram = self.detected_hardware['total_vram_gb'] / gpu_count
                estimates += f"Available VRAM per GPU: {avg_vram:.1f} GB\n"
                estimates += f"Recommended batch size: {int(avg_vram * 4)}-{int(avg_vram * 8)}\n"
        else:
            estimates += "CPU training - significantly slower than GPU\n"

        self.perf_info.setText(estimates)

    def _update_recommendations(self):
        """Update recommendations based on hardware"""
        if not hasattr(self, 'recommendations'):
            return

        recs = []

        if not self.detected_hardware['cuda_available']:
            recs.append("⚠️ No CUDA GPU detected - training will be very slow")
            recs.append("💡 Consider using cloud GPU (Google Colab, AWS, etc.)")
        else:
            for gpu in self.detected_hardware['gpu_devices']:
                major, minor = map(int, gpu['compute_capability'].split('.'))

                if major >= 8:
                    recs.append(f"✅ {gpu['name']}: Excellent for ML training")
                    recs.append("💡 Enable Flash Attention for faster training")
                elif major >= 7:
                    recs.append(f"✅ {gpu['name']}: Good for ML training")
                    recs.append("💡 Use AMP (fp16) for 2-3x speedup")
                else:
                    recs.append(f"⚠️ {gpu['name']}: Older GPU, limited optimizations")

            if self.detected_hardware['gpu_count'] > 1:
                recs.append(f"\n💡 {self.detected_hardware['gpu_count']} GPUs detected - enable distributed training")

        self.recommendations.setText("\n".join(recs))

    def _load_current_config(self):
        """Load current configuration into UI"""
        if not self.current_config:
            return

        # Load device selection
        if 'device' in self.current_config and hasattr(self, 'device_combo'):
            device = self.current_config['device']
            idx = self.device_combo.findData(device)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)

        # Load GPU optimizations
        if hasattr(self, 'use_amp'):
            self.use_amp.setChecked(self.current_config.get('use_amp', True))
        if hasattr(self, 'precision_combo'):
            self.precision_combo.setCurrentText(self.current_config.get('precision', 'fp16'))
        if hasattr(self, 'compile_model'):
            self.compile_model.setChecked(self.current_config.get('compile_model', True))
        if hasattr(self, 'use_fused_optimizer'):
            self.use_fused_optimizer.setChecked(self.current_config.get('use_fused_optimizer', False))
        if hasattr(self, 'use_flash_attention'):
            self.use_flash_attention.setChecked(self.current_config.get('use_flash_attention', False))

        # Load memory settings
        if hasattr(self, 'num_workers'):
            self.num_workers.setValue(self.current_config.get('num_workers', 4))
        if hasattr(self, 'pin_memory'):
            self.pin_memory.setChecked(self.current_config.get('pin_memory', True))

    def _test_configuration(self):
        """Test the current configuration"""
        config = self._get_configuration()

        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "PyTorch Not Available", "Cannot test - PyTorch not installed")
            return

        try:
            device = config['device']
            if device.startswith('cuda'):
                if not torch.cuda.is_available():
                    QMessageBox.warning(self, "CUDA Not Available", "CUDA is not available on this system")
                    return

                # Test GPU allocation
                if device == 'cuda:all':
                    for i in range(torch.cuda.device_count()):
                        test_tensor = torch.randn(100, 100, device=f'cuda:{i}')
                        del test_tensor
                else:
                    test_tensor = torch.randn(100, 100, device=device)
                    del test_tensor

                torch.cuda.empty_cache()

            QMessageBox.information(
                self,
                "Configuration Test Passed",
                f"✅ Configuration is valid and ready to use\n\nDevice: {device}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Configuration Test Failed", f"Error testing configuration:\n{str(e)}")

    def _reset_to_defaults(self):
        """Reset to default configuration"""
        if hasattr(self, 'device_combo') and self.device_combo.count() > 1:
            self.device_combo.setCurrentIndex(1)  # First GPU

        if hasattr(self, 'use_amp'):
            self.use_amp.setChecked(True)
        if hasattr(self, 'precision_combo'):
            self.precision_combo.setCurrentText('fp16')
        if hasattr(self, 'compile_model'):
            self.compile_model.setChecked(True)
        if hasattr(self, 'use_fused_optimizer'):
            self.use_fused_optimizer.setChecked(False)
        if hasattr(self, 'use_flash_attention'):
            self.use_flash_attention.setChecked(False)
        if hasattr(self, 'grad_accumulation'):
            self.grad_accumulation.setValue(1)
        if hasattr(self, 'num_workers'):
            self.num_workers.setValue(4)
        if hasattr(self, 'pin_memory'):
            self.pin_memory.setChecked(True)

    def _apply_configuration(self):
        """Apply configuration and close dialog"""
        config = self._get_configuration()
        self.config_accepted.emit(config)
        self.accept()

    def _get_configuration(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        config = {
            'device': self.device_combo.currentData() if hasattr(self, 'device_combo') else 'cpu',
            'num_workers': self.num_workers.value() if hasattr(self, 'num_workers') else 0,
            'pin_memory': self.pin_memory.isChecked() if hasattr(self, 'pin_memory') else False,
            'prefetch_factor': self.prefetch_factor.value() if hasattr(self, 'prefetch_factor') else 2,
        }

        # GPU optimizations
        if hasattr(self, 'use_amp'):
            config['use_amp'] = self.use_amp.isChecked()
            config['precision'] = self.precision_combo.currentText()
            config['compile_model'] = self.compile_model.isChecked()
            config['use_fused_optimizer'] = self.use_fused_optimizer.isChecked()
            config['use_flash_attention'] = self.use_flash_attention.isChecked()
            config['gradient_accumulation_steps'] = self.grad_accumulation.value()
            config['channels_last'] = self.channels_last.isChecked()
            config['gradient_checkpointing'] = self.gradient_checkpointing.isChecked()

        # Distributed training
        if hasattr(self, 'use_distributed'):
            config['use_distributed'] = self.use_distributed.isChecked()
            config['distributed_backend'] = self.backend_combo.currentText()
            config['find_unused_parameters'] = self.find_unused_parameters.isChecked()

        return config

    def get_configuration(self) -> Dict[str, Any]:
        """Public method to get configuration"""
        return self._get_configuration()
