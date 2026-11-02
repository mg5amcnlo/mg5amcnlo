# SPDX-License-Identifier: 0BSD
"""Typed progress phases for the generation service."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock

from pyamplicol.reporting import (
    NullProgressSink,
    ProgressEnd,
    ProgressSink,
    ProgressStart,
    ProgressUpdate,
)
from pyamplicol.reporting.progress import ProgressDetailValue


@dataclass(slots=True)
class PhaseHandle:
    task_id: str
    sink: ProgressSink | None
    total: int | None
    completed: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def update(
        self,
        completed: int,
        *,
        message: str | None = None,
        total: int | None = None,
        details: Mapping[str, ProgressDetailValue] | None = None,
    ) -> None:
        with self._lock:
            self._update_locked(
                completed,
                message=message,
                total=total,
                details=details,
            )

    def advance(
        self,
        *,
        message: str | None = None,
        details: Mapping[str, ProgressDetailValue] | None = None,
    ) -> None:
        with self._lock:
            self._update_locked(
                self.completed + 1,
                message=message,
                total=None,
                details=details,
            )

    def _update_locked(
        self,
        completed: int,
        *,
        message: str | None,
        total: int | None,
        details: Mapping[str, ProgressDetailValue] | None,
    ) -> None:
        active_total = self.total if total is None else total
        if total is not None:
            self.total = total
        bounded = completed
        if active_total is not None:
            bounded = min(completed, active_total)
        self.completed = max(self.completed, bounded)
        if self.sink is not None:
            self.sink.emit(
                ProgressUpdate(
                    self.task_id,
                    completed=self.completed,
                    total=active_total,
                    message=message,
                    details={} if details is None else details,
                )
            )

    @contextmanager
    def child(
        self,
        key: str,
        description: str,
        *,
        total: int | None = None,
        unit: str = "items",
        details: Mapping[str, ProgressDetailValue] | None = None,
    ) -> Iterator[PhaseHandle]:
        task_id = f"{self.task_id}:{key}"
        if self.sink is not None:
            self.sink.emit(
                ProgressStart(
                    task_id,
                    description,
                    total=total,
                    parent_task_id=self.task_id,
                    unit=unit,
                    details={} if details is None else details,
                )
            )
        handle = PhaseHandle(task_id, self.sink, total)
        started = time.perf_counter()
        try:
            yield handle
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            if self.sink is not None:
                self.sink.emit(
                    ProgressEnd(
                        task_id,
                        success=False,
                        message=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=elapsed,
                    )
                )
            raise
        elapsed = time.perf_counter() - started
        if total is not None and handle.completed < total:
            handle.update(total)
        if self.sink is not None:
            self.sink.emit(ProgressEnd(task_id, elapsed_seconds=elapsed))


@dataclass(slots=True)
class GenerationPhaseReporter:
    sink: ProgressSink | None
    timings: dict[str, float] = field(default_factory=dict)
    root: PhaseHandle | None = None
    _root_started_at: float | None = None

    def __post_init__(self) -> None:
        # Progress-off must not install callbacks into generation machinery.
        if isinstance(self.sink, NullProgressSink):
            self.sink = None

    def start_generation(
        self,
        *,
        total: int,
        details: Mapping[str, ProgressDetailValue],
    ) -> None:
        task_id = "generation"
        self._root_started_at = time.perf_counter()
        if self.sink is not None:
            self.sink.emit(
                ProgressStart(
                    task_id,
                    "Generating processes",
                    total=total,
                    unit="phases",
                    details=details,
                )
            )
        self.root = PhaseHandle(task_id, self.sink, total)

    def finish_generation(
        self,
        *,
        success: bool = True,
        message: str | None = None,
    ) -> None:
        if self.root is None or self._root_started_at is None:
            return
        elapsed = time.perf_counter() - self._root_started_at
        if (
            success
            and self.root.total is not None
            and self.root.completed < self.root.total
        ):
            self.root.update(self.root.total)
        if self.sink is not None:
            self.sink.emit(
                ProgressEnd(
                    self.root.task_id,
                    success=success,
                    message=message,
                    elapsed_seconds=elapsed,
                )
            )
        self._root_started_at = None

    @contextmanager
    def phase(
        self,
        name: str,
        description: str,
        *,
        total: int | None = None,
    ) -> Iterator[PhaseHandle]:
        task_id = f"generation:{name}"
        if self.sink is not None:
            self.sink.emit(
                ProgressStart(
                    task_id,
                    description,
                    total=total,
                    parent_task_id=None if self.root is None else self.root.task_id,
                )
            )
        handle = PhaseHandle(task_id, self.sink, total)
        started = time.perf_counter()
        try:
            yield handle
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            self.timings[name] = elapsed
            if self.sink is not None:
                self.sink.emit(
                    ProgressEnd(
                        task_id,
                        success=False,
                        message=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=elapsed,
                    )
                )
            raise
        elapsed = time.perf_counter() - started
        self.timings[name] = elapsed
        if total is not None and handle.completed < total:
            handle.update(total)
        if self.sink is not None:
            self.sink.emit(ProgressEnd(task_id, elapsed_seconds=elapsed))
        if self.root is not None:
            self.root.advance(
                message=description,
                details={"step": description},
            )


__all__ = ["GenerationPhaseReporter", "PhaseHandle"]
