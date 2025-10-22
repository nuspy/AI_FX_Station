# ui/controllers/training_controller.py
# Training Controller for ForexGPT - handles external training process execution
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool


class TrainingControllerSignals(QObject):
    log = Signal(str)
    progress = Signal(int)   # 0..100; -1 means indeterminate
    finished = Signal(bool)  # True if rc==0


class TrainingController(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = TrainingControllerSignals()
        self.pool = QThreadPool.globalInstance()

    def start_training(self, args: list[str], cwd: str | None = None):
        """Spawn the external trainer (e.g., train_sklearn or lightning) and stream logs to the UI."""
        class _Runner(QRunnable):
            def __init__(self, outer, args, cwd):
                super().__init__()
                self.outer = outer
                self.args = args
                self.cwd = cwd

            def run(self):
                import subprocess
                import os
                ok = False
                try:
                    # indeterminate progress while running
                    self.outer.signals.progress.emit(-1)

                    # Set PYTHONIOENCODING to handle PyTorch progress bar characters on Windows
                    env = os.environ.copy()
                    env['PYTHONIOENCODING'] = 'utf-8'

                    p = subprocess.Popen(
                        self.args,
                        cwd=self.cwd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",  # Replace problematic bytes with  to avoid crash
                        bufsize=1,
                        env=env,
                    )
                    for line in iter(p.stdout.readline, ""):
                        if not line:
                            break
                        self.outer.signals.log.emit(line.rstrip("\n"))
                    rc = p.wait()
                    ok = (rc == 0)
                except Exception as e:
                    self.outer.signals.log.emit(f"[error] {e}")
                    ok = False
                finally:
                    self.outer.signals.progress.emit(100 if ok else 0)
                    self.outer.signals.finished.emit(ok)

        self.pool.start(_Runner(self, args, cwd))