# ui/controllers_training_inproc.py
from __future__ import annotations
from PySide6.QtCore import QObject, QThread, Signal

from forex_diffusion.training.inproc import train_sklearn_inproc

class InprocSignals(QObject):
    log = Signal(str)
    progress = Signal(int)
    finished = Signal(bool)

class InprocWorker(QObject):
    def __init__(self, cfg: dict, fetch_fn):
        super().__init__()
        self.cfg = cfg
        self.fetch_fn = fetch_fn
        self.signals = InprocSignals()

    def run(self):
        try:
            def _log(s: str): self.signals.log.emit(s)
            def _prog(p: int): self.signals.progress.emit(int(p))
            res = train_sklearn_inproc(fetch_candles=self.fetch_fn, log=_log, progress=_prog, **self.cfg)
            self.signals.log.emit(f"[res] {res}")
            self.signals.finished.emit(True)
        except Exception as e:
            self.signals.log.emit(f"[err] {e}")
            self.signals.finished.emit(False)

class TrainingControllerInproc:
    def __init__(self, parent_widget):
        self.parent = parent_widget
        self.thread: QThread | None = None
        self.worker: InprocWorker | None = None

    def start(self, cfg: dict, fetch_fn, log_cb, progress_cb, finished_cb):
        self.thread = QThread(self.parent)
        self.worker = InprocWorker(cfg, fetch_fn)
        self.worker.signals.log.connect(log_cb)
        self.worker.signals.progress.connect(progress_cb)
        self.worker.signals.finished.connect(finished_cb)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
