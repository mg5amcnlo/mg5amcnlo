# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import time

from pyamplicol.reporting.resources import (
    ProcessInfo,
    ProcessTreeSampler,
    ResourceUsageMonitor,
    _parse_proc_stat_ppid,
    _parse_proc_status_rss,
    _parse_ps_output,
)


def test_resource_parsers_accept_proc_and_ps_shapes() -> None:
    assert _parse_proc_stat_ppid("12 (python worker) S 7 12 12 0") == 7
    assert _parse_proc_status_rss("Name:\tpython\nVmRSS:\t1536 kB\n") == 1536 * 1024
    assert _parse_ps_output(" 12 7 1536\n") == {
        12: ProcessInfo(pid=12, ppid=7, rss_bytes=1536 * 1024)
    }


def test_process_tree_sampler_aggregates_descendants_only() -> None:
    records = {
        10: ProcessInfo(10, 1, 100),
        11: ProcessInfo(11, 10, 200),
        12: ProcessInfo(12, 11, 300),
        20: ProcessInfo(20, 1, 500),
    }
    sampler = ProcessTreeSampler(10)
    assert sampler.sample(records) == (600, 3)

    # A known descendant remains associated if its direct parent exits.
    reparanted = {
        10: ProcessInfo(10, 1, 110),
        12: ProcessInfo(12, 1, 310),
        20: ProcessInfo(20, 1, 500),
    }
    assert sampler.sample(reparanted) == (420, 2)


def test_resource_monitor_tracks_current_and_peak_rss() -> None:
    snapshots = iter(
        (
            {10: ProcessInfo(10, 1, 100)},
            {
                10: ProcessInfo(10, 1, 120),
                11: ProcessInfo(11, 10, 80),
            },
            {10: ProcessInfo(10, 1, 90)},
        )
    )
    monitor = ResourceUsageMonitor(root_pid=10, snapshotter=lambda: next(snapshots))
    monitor._sample_once()
    assert monitor.usage.current_rss_bytes == 100
    assert monitor.usage.peak_rss_bytes == 100
    monitor._sample_once()
    assert monitor.usage.current_rss_bytes == 200
    assert monitor.usage.peak_rss_bytes == 200
    assert monitor.usage.process_count == 2
    monitor._sample_once()
    assert monitor.usage.current_rss_bytes == 90
    assert monitor.usage.peak_rss_bytes == 200


def test_resource_monitor_fails_open() -> None:
    def unavailable() -> dict[int, ProcessInfo]:
        raise OSError("not available")

    monitor = ResourceUsageMonitor(root_pid=10, snapshotter=unavailable)
    monitor._sample_once()
    assert monitor.usage.current_rss_bytes is None
    assert monitor.usage.peak_rss_bytes is None
    assert monitor.usage.process_count is None


def test_resource_monitor_thread_stops_cleanly() -> None:
    monitor = ResourceUsageMonitor(
        root_pid=10,
        interval_seconds=0.01,
        snapshotter=lambda: {10: ProcessInfo(10, 1, 123)},
    )
    monitor.start()
    time.sleep(0.03)
    monitor.close()

    assert monitor._thread is None
    assert monitor.usage.current_rss_bytes == 123
    assert monitor.usage.peak_rss_bytes == 123
