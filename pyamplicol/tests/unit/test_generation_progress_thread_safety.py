# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from pyamplicol.generation.progress import GenerationPhaseReporter, PhaseHandle
from pyamplicol.reporting import (
    CallbackProgressSink,
    NullProgressSink,
    ProgressEnd,
    ProgressUpdate,
)


def test_phase_advances_are_thread_safe_monotonic_and_exact() -> None:
    events: list[object] = []
    handle = PhaseHandle(
        "generation:test",
        CallbackProgressSink(events.append),
        total=200,
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(handle.advance, message=str(index)) for index in range(200)
        ]
        for future in futures:
            future.result()

    updates = [event for event in events if isinstance(event, ProgressUpdate)]
    assert handle.completed == 200
    assert [event.completed for event in updates] == list(range(1, 201))
    assert {event.total for event in updates} == {200}


def test_progress_off_normalizes_to_no_generation_callbacks() -> None:
    reporter = GenerationPhaseReporter(NullProgressSink())

    assert reporter.sink is None


def test_child_progress_records_interrupt_without_timing_message() -> None:
    events: list[object] = []
    reporter = GenerationPhaseReporter(CallbackProgressSink(events.append))
    reporter.start_generation(total=1, details={"execution_mode": "compiled"})

    with (
        pytest.raises(KeyboardInterrupt),
        reporter.phase("dag", "DAG construction", total=1) as phase,
        phase.child("process", "process DAG"),
    ):
        raise KeyboardInterrupt

    endings = [event for event in events if isinstance(event, ProgressEnd)]
    assert endings
    assert all(event.elapsed_seconds is not None for event in endings)
    assert all(
        event.message is None or not event.message.endswith("s") for event in endings
    )
    assert endings[0].success is False
