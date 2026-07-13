from __future__ import annotations

import traceback
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from leaps.models import StageEvent
from leaps.platform_support import SleepInhibitor
from leaps.science import CancellationToken


class WorkerSignals(QObject):
    event = Signal(object)
    result = Signal(object)
    error = Signal(object)
    finished = Signal()


class StageWorker(QRunnable):
    def __init__(
        self,
        function: Callable[..., Any],
        *args: Any,
        inhibit_sleep: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.token = CancellationToken()
        self.inhibit_sleep = inhibit_sleep
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.kwargs.setdefault("token", self.token)
            self.kwargs.setdefault("emit", self.signals.event.emit)
            context = (
                SleepInhibitor("LEAPS is processing an observing run")
                if self.inhibit_sleep
                else nullcontext()
            )
            with context:
                result = self.function(*self.args, **self.kwargs)
            self._emit(self.signals.result, result)
        except BaseException as exc:
            if not getattr(exc, "technical_details", None):
                try:
                    exc.technical_details = "".join(traceback.format_exception(exc))
                except Exception:
                    pass
            self._emit(self.signals.error, exc)
        finally:
            self._emit(self.signals.finished)

    @staticmethod
    def _emit(signal: Signal, *args: Any) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:
            # The app or owning window may have closed while a safe background
            # operation was finishing. Its result no longer has a UI receiver.
            pass


class TaskRunner(QObject):
    busyChanged = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self.current: StageWorker | None = None
        self.current_operation = ""

    def start(
        self,
        function: Callable[..., Any],
        *args: Any,
        event: Callable[[StageEvent], None] | None = None,
        result: Callable[[Any], None] | None = None,
        error: Callable[[BaseException], None] | None = None,
        finished: Callable[[], None] | None = None,
        inhibit_sleep: bool = True,
        operation: str = "Background operation",
        **kwargs: Any,
    ) -> StageWorker:
        if self.current is not None:
            raise RuntimeError("A task is already running")
        worker = StageWorker(function, *args, inhibit_sleep=inhibit_sleep, **kwargs)
        self.current = worker
        self.current_operation = operation
        if event:
            worker.signals.event.connect(event)
        if result:
            worker.signals.result.connect(result)
        if error:
            worker.signals.error.connect(error)
        if finished:
            worker.signals.finished.connect(finished)
        worker.signals.finished.connect(self._finish)
        self.busyChanged.emit(True)
        self.pool.start(worker)
        return worker

    @Slot()
    def cancel(self) -> None:
        if self.current is not None:
            self.current.token.cancel()

    @Slot()
    def _finish(self) -> None:
        self.current = None
        self.current_operation = ""
        self.busyChanged.emit(False)
