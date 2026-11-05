# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import re
import threading
import time
from dataclasses import FrozenInstanceError

import pytest

from pyamplicol.reporting import (
    CallbackProgressSink,
    LoggingProgressSink,
    ProgressEnd,
    ProgressStart,
    ProgressUpdate,
    StreamProgressSink,
    TtyProgressSink,
    progress_sink,
)
from pyamplicol.reporting import progress as progress_module
from pyamplicol.reporting.progress import _dashboard_lines, _TaskState
from pyamplicol.reporting.resources import ResourceUsage


def test_progress_events_are_frozen_and_validated() -> None:
    event = ProgressStart(
        "build",
        "Building",
        total=3,
        parent_task_id="root",
        unit="chunks",
        details={"chunk_index": 1},
    )
    assert event.parent_task_id == "root"
    assert event.unit == "chunks"
    assert event.details == {"chunk_index": 1}
    with pytest.raises(FrozenInstanceError):
        event.total = 4  # type: ignore[misc]
    with pytest.raises(ValueError, match="exceed"):
        ProgressUpdate("build", completed=4, total=3)
    with pytest.raises(ValueError, match="finite"):
        ProgressUpdate("build", 1, details={"waiting_seconds": float("inf")})


def test_callback_and_stream_sinks_receive_typed_events() -> None:
    received: list[object] = []
    CallbackProgressSink(received.append).emit(ProgressEnd("build"))
    assert received == [ProgressEnd("build")]

    stream = io.StringIO()
    StreamProgressSink(stream).emit(ProgressUpdate("build", 2, 3, "currents"))
    assert stream.getvalue() == "build 2/3: currents\n"


def test_log_progress_includes_recurrence_stage_counters() -> None:
    stream = io.StringIO()
    StreamProgressSink(stream).emit(
        ProgressUpdate(
            "generation:recurrence:process:rust-builder",
            4,
            8,
            "recurrence stage",
            details={
                "process": "uubar_Z_4g",
                "stage_index": 3,
                "stage_total": 5,
                "subset_size": 4,
                "candidate_parent_pair_count": 65_536,
                "candidate_parent_pair_total": 120_000,
                "current_count": 18_304,
                "dynamic_color_state_count": 812,
                "color_target_prune_count": 12_345,
                "contribution_count": 96_112,
            },
        )
    )

    line = stream.getvalue()
    assert "process=uubar_Z_4g" in line
    assert "stage=3/5" in line
    assert "candidate_parent_pair_count=65536" in line
    assert "color_target_prune_count=12345" in line
    assert "current_count=18304" in line
    assert "contribution_count=96112" in line


def test_log_progress_renders_native_self_rss_samples() -> None:
    stream = io.StringIO()
    StreamProgressSink(stream).emit(
        ProgressUpdate(
            "runtime-warm-up:query_family",
            2,
            3,
            details={
                "native_current_rss_bytes": 256 * 1024**2,
                "native_peak_rss_bytes": 384 * 1024**2,
            },
        )
    )

    assert "self_rss=0.25/0.38 GiB current/peak" in stream.getvalue()


def test_auto_progress_uses_logging_for_non_tty_streams() -> None:
    assert isinstance(progress_sink("auto", stream=io.StringIO()), LoggingProgressSink)
    assert isinstance(progress_sink("tty", stream=io.StringIO()), TtyProgressSink)


def test_logging_progress_rate_limits_updates_but_always_emits_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sink = LoggingProgressSink(minimum_interval=60.0)
    with caplog.at_level("INFO", logger="pyamplicol.progress"):
        sink.emit(ProgressStart("build", "Building", 3))
        sink.emit(ProgressUpdate("build", 1, 3))
        sink.emit(ProgressUpdate("build", 2, 3))
        sink.emit(ProgressUpdate("build", 3, 3))
        sink.emit(ProgressEnd("build"))
    messages = [record.getMessage() for record in caplog.records]
    assert "build 2/3" not in messages
    assert "build 3/3" in messages
    assert messages[-1] == "build done"


def test_logging_progress_colors_typed_benchmark_statistics_without_changing_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    event = ProgressUpdate(
        "runtime-benchmark",
        3,
        5,
        (
            "0.555/1s; wall 1.7773 us/point +/- 0.0072 us "
            "(relative standard error 0.41%); 813 calls x 128 points"
        ),
        details={
            "progress_kind": "benchmark-statistics",
            "elapsed_seconds": 0.555,
            "target_seconds": 1.0,
            "wall_seconds_per_point": 1.7773e-6,
            "wall_standard_error_seconds_per_point": 0.0072e-6,
            "wall_relative_standard_error": 0.0041,
            "repetitions": 813,
            "batch_size": 128,
            "sample_count": 3,
        },
    )
    plain = progress_module._format_event(event)
    sink = LoggingProgressSink(minimum_interval=0.0, color=True)

    with caplog.at_level("INFO", logger="pyamplicol.progress"):
        sink.emit(event)

    colored = caplog.records[-1].getMessage()
    assert "\x1b[" in colored
    assert re.sub(r"\x1b\[[0-9;]*m", "", colored) == plain
    assert "runtime-benchmark 3/5: 0.555/1s" in plain
    assert "wall 1.7773 us/point" in plain
    assert "relative standard error 0.41%" in plain


def test_log_progress_factory_honors_resolved_color_setting() -> None:
    sink = progress_sink("log", stream=io.StringIO(), color=True)

    assert isinstance(sink, LoggingProgressSink)
    assert sink.color is True


def test_tty_progress_accepts_concurrent_phase_shape() -> None:
    stream = io.StringIO()
    sink = TtyProgressSink(stream)
    sink.emit(ProgressStart("dag", "Building DAG", 2))
    sink.emit(ProgressUpdate("dag", 1, 2, "currents"))
    sink.emit(ProgressUpdate("dag", 2, 2))
    sink.emit(ProgressEnd("dag"))
    assert "Building DAG" in stream.getvalue()


def test_tty_message_stream_preserves_write_only_text_stream_semantics() -> None:
    class FlushRecordingStream(io.StringIO):
        flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1
            super().flush()

    stream = FlushRecordingStream()
    message_stream = TtyProgressSink(stream).message_stream

    assert message_stream.write("diagnostic") == len("diagnostic")
    message_stream.flush()
    assert stream.getvalue() == "diagnostic"
    assert stream.flush_count == 1
    assert message_stream.writable()
    assert not message_stream.readable()
    assert not message_stream.seekable()
    with pytest.raises(io.UnsupportedOperation):
        message_stream.read()
    with pytest.raises(io.UnsupportedOperation):
        message_stream.tell()


def test_tty_progress_renders_without_holding_the_task_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TtyStream(io.StringIO):
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            raise OSError

    class SnapshotDashboard:
        def __init__(self, *, snapshot: object, **_: object) -> None:
            self.snapshot = snapshot

        def start(self) -> None:
            return None

        def render_now(self) -> None:
            completed = threading.Event()

            def take_snapshot() -> None:
                self.snapshot()  # type: ignore[operator]
                completed.set()

            thread = threading.Thread(target=take_snapshot, daemon=True)
            thread.start()
            thread.join(timeout=0.5)
            assert completed.is_set(), "dashboard snapshot blocked on the sink lock"

        def close(self) -> None:
            return None

    monkeypatch.setattr(progress_module, "_Dashboard", SnapshotDashboard)
    sink = TtyProgressSink(TtyStream())
    sink.emit(ProgressStart("build", "Building", 2))
    sink.emit(ProgressUpdate("build", 1, 2))
    sink.close()


def test_forced_tty_on_non_tty_is_deterministic_and_ansi_free() -> None:
    stream = io.StringIO()
    sink = TtyProgressSink(stream, color=True)
    sink.emit(ProgressStart("profile", "Profiling runtime", 1))
    sink.emit(ProgressUpdate("profile", 1, 1, "measured"))
    sink.emit(ProgressEnd("profile"))

    assert "Profiling runtime" in stream.getvalue()
    assert "measured" in stream.getvalue()
    assert "\x1b[" not in stream.getvalue()


def test_dashboard_renders_elapsed_peak_rss_and_granular_status() -> None:
    started = time.monotonic() - 68.0
    root = _TaskState(
        "generation",
        "Generating processes",
        8,
        None,
        "phases",
        {"execution_mode": "compiled", "backend": "JIT", "optimization": "O2"},
        started,
        started + 60.0,
        completed=3,
    )
    phase = _TaskState(
        "generation:dag",
        "DAG construction",
        4,
        "generation",
        "processes",
        {},
        started + 40.0,
        started + 65.0,
        completed=3,
    )
    detail = _TaskState(
        "generation:dag:uubar_Z_5g",
        "uubar_Z_5g DAG",
        62,
        "generation:dag",
        "masks",
        {
            "process": "uubar_Z_5g",
            "step": "recursion",
            "stage_index": 5,
            "stage_total": 6,
            "subset_size": 6,
            "mask_index": 47,
            "mask_total": 62,
            "current_count": 18_304,
            "interaction_count": 96_112,
        },
        started + 45.0,
        started + 67.0,
        completed=47,
    )

    lines = _dashboard_lines(
        (root, phase, detail),
        now=started + 68.0,
        usage=ResourceUsage(
            current_rss_bytes=3 * 1024**3,
            peak_rss_bytes=4 * 1024**3,
            process_count=3,
        ),
        width=180,
        color=True,
    )

    assert "elapsed 1m08s" in lines[0]
    assert "RSS 3.00/4.00 GiB current/peak" in lines[0]
    assert "children 2" in lines[0]
    assert "DAG construction" in lines[1]
    assert "ETA" in lines[1]
    assert "uubar_Z_5g" in lines[2]
    assert "\x1b[35m  stage \x1b[0m" in lines[2]
    assert "\x1b[32m5/6\x1b[0m" in lines[2]
    assert "\x1b[35m  subset \x1b[0m" in lines[2]
    assert "\x1b[32m6\x1b[0m" in lines[2]
    assert "masks 47/62" in lines[2]
    assert "currents 18,304" in lines[2]
    assert "interactions 96,112" in lines[2]
    assert any("\x1b[" in line for line in lines)

    narrow = _dashboard_lines(
        (root, phase, detail),
        now=started + 68.0,
        usage=ResourceUsage(),
        width=48,
        color=False,
    )
    assert all(len(line) <= 48 for line in narrow)
    assert "RSS N/A" in narrow[0]


def test_dashboard_enriches_legacy_evaluator_callback_messages() -> None:
    now = time.monotonic()
    task = _TaskState(
        "generation:evaluators:process",
        "Compile process evaluators",
        1,
        None,
        "stages",
        {"process": "uubar_Z_5g", "backend": "jit", "stage": "jit compile"},
        now - 20.0,
        now,
        message="stage_4 chunk 12/43 p=1525/8000 waiting 18s",
    )

    detail = _dashboard_lines(
        (task,),
        now=now,
        usage=ResourceUsage(),
        width=180,
        color=False,
    )[2]

    assert "chunk compilation" in detail
    assert "chunks 12/43" in detail
    assert "inputs 1,525" in detail
    assert "waiting 18s" in detail
