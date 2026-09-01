"""Generic Qt background execution with cooperative cancellation signals."""

from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class CancellationToken:
    """Thread-safe cancellation request checked at safe backend boundaries."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


class WorkerSignals(QObject):
    stage = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str, str)
    finished = Signal()


class FunctionWorker(QRunnable):
    """Run an injected operation outside the GUI thread."""

    def __init__(self, operation: Callable[[Callable[[str], None], Callable[[], bool]], Any]) -> None:
        super().__init__()
        self.operation = operation
        self.token = CancellationToken()
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.operation(self.signals.stage.emit, self.token.is_cancelled)
        except Exception as error:  # Qt boundary converts technical errors to safe UI data.
            self.signals.failed.emit(type(error).__name__, str(error))
        else:
            self.signals.succeeded.emit(result)
        finally:
            self.signals.finished.emit()

    def cancel(self) -> None:
        self.token.cancel()
