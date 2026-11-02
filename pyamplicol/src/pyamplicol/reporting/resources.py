# SPDX-License-Identifier: 0BSD
"""Portable, read-only process-tree resource sampling for live progress."""

from __future__ import annotations

import os
import platform
import subprocess
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class ResourceProbeError(RuntimeError):
    """Raised when resident-memory information cannot be collected."""


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    pid: int
    ppid: int
    rss_bytes: int


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    current_rss_bytes: int | None = None
    peak_rss_bytes: int | None = None
    process_count: int | None = None


ProcessSnapshot = Callable[[], dict[int, ProcessInfo]]


def _parse_proc_stat_ppid(text: str) -> int:
    closing_parenthesis = text.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("missing process-name terminator")
    fields = text[closing_parenthesis + 1 :].split()
    if len(fields) < 2:
        raise ValueError("incomplete /proc stat record")
    return int(fields[1])


def _parse_proc_status_rss(text: str) -> int:
    for line in text.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2].lower() != "kb":
            raise ValueError(f"invalid VmRSS record: {line!r}")
        return int(fields[1]) * 1024
    return 0


def _parse_ps_output(text: str) -> dict[int, ProcessInfo]:
    records: dict[int, ProcessInfo] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"invalid ps row {line_number}: {line!r}")
        pid, ppid, rss_kib = map(int, fields)
        if min(pid, ppid, rss_kib) < 0:
            raise ValueError(f"negative ps value on row {line_number}")
        records[pid] = ProcessInfo(pid, ppid, rss_kib * 1024)
    return records


def _linux_proc_snapshot(proc_root: Path = Path("/proc")) -> dict[int, ProcessInfo]:
    records: dict[int, ProcessInfo] = {}
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise ResourceProbeError(f"cannot enumerate {proc_root}: {exc}") from exc
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            status = (entry / "status").read_text(
                encoding="utf-8", errors="replace"
            )
            pid = int(entry.name)
            records[pid] = ProcessInfo(
                pid=pid,
                ppid=_parse_proc_stat_ppid(stat),
                rss_bytes=_parse_proc_status_rss(status),
            )
        except (OSError, ValueError):
            # Processes may disappear between enumeration and reading.
            continue
    if not records:
        raise ResourceProbeError(f"{proc_root} yielded no readable processes")
    return records


def _ps_snapshot() -> dict[int, ProcessInfo]:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        probe = subprocess.Popen(
            ["ps", "-axo", "pid=,ppid=,rss="],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        stdout, stderr = probe.communicate(timeout=10)
    except subprocess.TimeoutExpired as exc:
        probe.kill()
        probe.communicate()
        raise ResourceProbeError("ps process probe timed out") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourceProbeError(f"cannot execute ps: {exc}") from exc
    if probe.returncode != 0:
        detail = stderr.strip() or f"exit code {probe.returncode}"
        raise ResourceProbeError(f"ps process probe failed: {detail}")
    try:
        records = _parse_ps_output(stdout)
    except ValueError as exc:
        raise ResourceProbeError(str(exc)) from exc
    # The sampling command is itself a short-lived descendant of pyAmpliCol.
    records.pop(probe.pid, None)
    if not records:
        raise ResourceProbeError("ps process probe yielded no records")
    return records


def process_snapshot(system: str | None = None) -> dict[int, ProcessInfo]:
    host = system or platform.system()
    if host == "Linux":
        try:
            return _linux_proc_snapshot()
        except ResourceProbeError:
            return _ps_snapshot()
    if host == "Darwin":
        return _ps_snapshot()
    raise ResourceProbeError(f"unsupported host for RSS monitoring: {host!r}")


class ProcessTreeSampler:
    """Sample one root and its descendants, retaining live re-parented children."""

    def __init__(self, root_pid: int) -> None:
        if root_pid < 1:
            raise ValueError("resource sampler root PID must be positive")
        self.root_pid = root_pid
        self._known_pids = {root_pid}

    def sample(self, records: dict[int, ProcessInfo]) -> tuple[int, int]:
        children: dict[int, list[int]] = defaultdict(list)
        for record in records.values():
            children[record.ppid].append(record.pid)

        selected = {pid for pid in self._known_pids if pid in records}
        if self.root_pid in records:
            selected.add(self.root_pid)
        pending = list(selected)
        while pending:
            parent = pending.pop()
            for child in children.get(parent, ()):
                if child not in selected:
                    selected.add(child)
                    pending.append(child)

        self._known_pids = selected
        return sum(records[pid].rss_bytes for pid in selected), len(selected)


class ResourceUsageMonitor:
    """Sample aggregate RSS in a daemon thread without affecting generation."""

    def __init__(
        self,
        *,
        root_pid: int | None = None,
        interval_seconds: float = 1.0,
        snapshotter: ProcessSnapshot = process_snapshot,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource monitor interval must be positive")
        self._sampler = ProcessTreeSampler(root_pid or os.getpid())
        self._interval_seconds = float(interval_seconds)
        self._snapshotter = snapshotter
        self._usage = ResourceUsage()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def usage(self) -> ResourceUsage:
        with self._lock:
            return self._usage

    def start(self) -> None:
        if self._thread is not None:
            return
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="pyamplicol-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0 * self._interval_seconds, 1.0))
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        try:
            rss_bytes, process_count = self._sampler.sample(self._snapshotter())
        except (OSError, ResourceProbeError, subprocess.SubprocessError, ValueError):
            return
        with self._lock:
            peak = max(self._usage.peak_rss_bytes or 0, rss_bytes)
            self._usage = ResourceUsage(
                current_rss_bytes=rss_bytes,
                peak_rss_bytes=peak,
                process_count=process_count,
            )


__all__ = [
    "ProcessInfo",
    "ProcessTreeSampler",
    "ResourceProbeError",
    "ResourceUsage",
    "ResourceUsageMonitor",
    "process_snapshot",
]
