import traceback

from PyQt5.QtCore import QThread, pyqtSignal


class TaskThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, target, *args, **kwargs):
        super().__init__()
        self._target = target
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._target(*self._args, **self._kwargs)
        except Exception as exc:  # pragma: no cover
            message = f"{exc}\n\n{traceback.format_exc()}"
            self.failed.emit(message)
            return
        self.succeeded.emit(result)
