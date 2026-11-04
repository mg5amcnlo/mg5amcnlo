#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run a command under a conservative aggregate process-tree memory limit.

The implementation deliberately uses only the Python standard library. Linux
RSS data comes from ``/proc``; macOS uses the platform ``ps`` command and
``proc_pid_rusage`` physical-footprint data. The child starts in a new process
session so the watchdog can terminate both its process group and descendants
that create another process group.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import datetime as dt
import errno
import hashlib
import json
import math
import os
import platform
import shlex
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

GIB = 1024**3
MIB = 1024**2
DEFAULT_LIMIT_GIB = 30.0
DEFAULT_POLL_INTERVAL = 0.25
DEFAULT_TERMINATE_GRACE = 5.0
MEMORY_LIMIT_EXIT_CODE = 137
WATCHDOG_ERROR_EXIT_CODE = 125
RSS_LIMIT_REASON = "process-tree-rss-limit"
MEMORY_PROBE_REASON = "process-tree-memory-probe-unavailable"
DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON = (
    "darwin-process-tree-physical-footprint-limit"
)
DARWIN_PHYSICAL_FOOTPRINT_PROBE_REASON = (
    "darwin-process-tree-physical-footprint-probe-unavailable"
)
WATCHDOG_REPORT_KIND = "pyamplicol-memory-watchdog-execution-report"
WATCHDOG_REPORT_SCHEMA = 2
WATCHDOG_SCOPE = "complete-orchestrator-process-tree-v1"
_RUSAGE_INFO_V0 = 0
_RUSAGE_INFO_V0_BYTES = 96
_RUSAGE_INFO_PHYS_FOOTPRINT_OFFSET = 72


class ProbeError(RuntimeError):
    """Raised when resident-memory information cannot be collected."""


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    """One process record from a platform RSS probe."""

    pid: int
    ppid: int
    pgid: int
    rss_bytes: int
    cpu_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class MemorySample:
    """Aggregate memory and members belonging to the guarded command."""

    rss_bytes: int
    members: tuple[ProcessInfo, ...]
    physical_footprint_bytes: int | None = None


Snapshotter = Callable[[], dict[int, ProcessInfo]]
PhysicalFootprintProbe = Callable[[Iterable[int]], dict[int, int]]
PidExistsProbe = Callable[[int], bool]


def _pid_exists(pid: int) -> bool:
    """Return whether ``pid`` still exists without requiring signal authority."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _result_identity(path: Path) -> dict[str, object]:
    lexical = Path(os.path.abspath(path.expanduser()))
    resolved = lexical.resolve(strict=True)
    if not resolved.is_file():
        raise OSError(f"bound result is not a regular file: {path}")
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = resolved.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise OSError(f"bound result changed while it was hashed: {path}")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": after.st_size,
        "sha256": digest.hexdigest(),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(os.path.abspath(path.expanduser()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise OSError(f"watchdog report already exists: {destination}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _positive_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _nonnegative_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def _parse_proc_stat(text: str) -> tuple[int, int]:
    """Return ``(ppid, pgid)`` from Linux ``/proc/PID/stat`` text."""

    closing_parenthesis = text.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("missing process-name terminator")
    fields = text[closing_parenthesis + 1 :].split()
    # fields starts at the kernel's field 3 (state).
    if len(fields) < 3:
        raise ValueError("incomplete /proc stat record")
    return int(fields[1]), int(fields[2])


def _parse_proc_status_rss(text: str) -> int:
    """Return RSS bytes from Linux ``/proc/PID/status`` text."""

    for line in text.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2].lower() != "kb":
            raise ValueError(f"invalid VmRSS record: {line!r}")
        return int(fields[1]) * 1024
    return 0


def _parse_ps_output(text: str) -> dict[int, ProcessInfo]:
    """Parse ``ps -axo pid=,ppid=,pgid=,rss=`` output."""

    records: dict[int, ProcessInfo] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid ps row {line_number}: {line!r}")
        pid, ppid, pgid, rss_kib = map(int, fields)
        if min(pid, ppid, pgid, rss_kib) < 0:
            raise ValueError(f"negative ps value on row {line_number}")
        records[pid] = ProcessInfo(pid, ppid, pgid, rss_kib * 1024)
    return records


def _linux_proc_snapshot(proc_root: Path = Path("/proc")) -> dict[int, ProcessInfo]:
    """Collect a Linux process snapshot directly from procfs."""

    records: dict[int, ProcessInfo] = {}
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as error:
        raise ProbeError(f"cannot enumerate {proc_root}: {error}") from error
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            status_text = (entry / "status").read_text(
                encoding="utf-8", errors="replace"
            )
            ppid, pgid = _parse_proc_stat(stat_text)
            rss_bytes = _parse_proc_status_rss(status_text)
        except (OSError, ValueError):
            # Processes can disappear or become inaccessible between directory
            # enumeration and reading their records.
            continue
        records[pid] = ProcessInfo(pid, ppid, pgid, rss_bytes)
    if not records:
        raise ProbeError(f"{proc_root} yielded no readable process records")
    return records


def _ps_snapshot() -> dict[int, ProcessInfo]:
    """Collect a portable BSD/POSIX ``ps`` process snapshot."""

    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,rss="],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ProbeError(f"cannot execute ps: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise ProbeError(f"ps process probe failed: {detail}")
    try:
        records = _parse_ps_output(completed.stdout)
    except ValueError as error:
        raise ProbeError(str(error)) from error
    if not records:
        raise ProbeError("ps process probe yielded no records")
    return records


def _parse_darwin_bsdinfo(text: bytes) -> tuple[int, int, int]:
    """Return ``(pid, ppid, pgid)`` from Darwin ``proc_bsdinfo`` bytes."""

    # These stable fields precede variable-size names in libproc.h.  pbi_pgid
    # follows pbi_nfiles after the two fixed 16/32-byte name buffers.
    if len(text) < 104:
        raise ValueError("incomplete Darwin proc_bsdinfo record")
    pid = struct.unpack_from("=I", text, 12)[0]
    ppid = struct.unpack_from("=I", text, 16)[0]
    pgid = struct.unpack_from("=I", text, 100)[0]
    return pid, ppid, pgid


def _parse_darwin_taskinfo_rss(text: bytes) -> int:
    """Return resident bytes from Darwin ``proc_taskinfo`` bytes."""

    if len(text) < 16:
        raise ValueError("incomplete Darwin proc_taskinfo record")
    return struct.unpack_from("=Q", text, 8)[0]


def _parse_darwin_taskinfo_cpu_seconds(
    text: bytes,
    *,
    timebase_numer: int,
    timebase_denom: int,
) -> float:
    """Return cumulative user and system CPU seconds from ``proc_taskinfo``."""

    if len(text) < 32:
        raise ValueError("incomplete Darwin proc_taskinfo CPU record")
    if timebase_numer <= 0 or timebase_denom <= 0:
        raise ValueError("invalid Darwin Mach timebase")
    user_ticks = struct.unpack_from("=Q", text, 16)[0]
    system_ticks = struct.unpack_from("=Q", text, 24)[0]
    return (user_ticks + system_ticks) * timebase_numer / timebase_denom / 1e9


class _MachTimebaseInfo(ctypes.Structure):
    _fields_ = (("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32))


def _darwin_mach_timebase(library: object | None = None) -> tuple[int, int]:
    """Return the Mach absolute-time nanosecond conversion ratio."""

    if library is None:
        try:
            library = ctypes.CDLL(None)
        except OSError as error:
            raise ProbeError(f"cannot load Darwin Mach runtime: {error}") from error
    try:
        mach_timebase_info = library.mach_timebase_info  # type: ignore[attr-defined]
    except AttributeError as error:
        raise ProbeError(f"cannot load mach_timebase_info: {error}") from error
    mach_timebase_info.argtypes = (ctypes.POINTER(_MachTimebaseInfo),)
    mach_timebase_info.restype = ctypes.c_int
    timebase = _MachTimebaseInfo()
    result = mach_timebase_info(ctypes.byref(timebase))
    if result != 0 or timebase.numer <= 0 or timebase.denom <= 0:
        raise ProbeError(
            "Darwin mach_timebase_info failed "
            f"(result={result}, numer={timebase.numer}, denom={timebase.denom})"
        )
    return int(timebase.numer), int(timebase.denom)


def _parse_darwin_rusage_phys_footprint(text: bytes) -> int:
    """Return ``ri_phys_footprint`` from Darwin ``rusage_info_v0`` bytes."""

    minimum_size = _RUSAGE_INFO_PHYS_FOOTPRINT_OFFSET + 8
    if len(text) < minimum_size:
        raise ValueError("incomplete Darwin rusage_info_v0 record")
    return struct.unpack_from(
        "=Q", text, _RUSAGE_INFO_PHYS_FOOTPRINT_OFFSET
    )[0]


class DarwinPhysicalFootprintProbe:
    """Read per-process physical footprints through Darwin ``libproc``."""

    def __init__(
        self,
        library: object | None = None,
        *,
        pid_exists: PidExistsProbe = _pid_exists,
    ) -> None:
        if library is None:
            library_name = (
                ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
            )
            try:
                library = ctypes.CDLL(library_name, use_errno=True)
            except OSError as error:
                raise ProbeError(
                    f"cannot load Darwin proc_pid_rusage: {error}"
                ) from error
        try:
            proc_pid_rusage = library.proc_pid_rusage  # type: ignore[attr-defined]
        except AttributeError as error:
            raise ProbeError(
                f"cannot load Darwin proc_pid_rusage: {error}"
            ) from error
        proc_pid_rusage.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        proc_pid_rusage.restype = ctypes.c_int
        self._proc_pid_rusage = proc_pid_rusage
        self._pid_exists = pid_exists

    def __call__(self, pids: Iterable[int]) -> dict[int, int]:
        footprints: dict[int, int] = {}
        for pid in sorted(set(pids)):
            error_number = 0
            buffer = ctypes.create_string_buffer(_RUSAGE_INFO_V0_BYTES)
            for attempt in range(3):
                ctypes.set_errno(0)
                result = self._proc_pid_rusage(
                    pid,
                    _RUSAGE_INFO_V0,
                    buffer,
                )
                if result == 0:
                    footprints[pid] = _parse_darwin_rusage_phys_footprint(
                        buffer.raw
                    )
                    break
                error_number = ctypes.get_errno()
                if error_number in {errno.ENOENT, errno.ESRCH}:
                    # The process exited between the process-tree and
                    # footprint snapshots. Its prior RSS remains in this
                    # sample.
                    break
                if error_number not in {errno.EACCES, errno.EPERM} or attempt == 2:
                    break
                # Darwin can transiently report EPERM while a short-lived
                # descendant crosses exec/exit. Retry the same PID in-place;
                # persistent access failures still fail closed below.
            else:  # pragma: no cover - the bounded loop always breaks
                result = -1
            if result == 0 or error_number in {errno.ENOENT, errno.ESRCH}:
                continue
            if error_number in {errno.EACCES, errno.EPERM} and not self._pid_exists(
                pid
            ):
                # A short-lived descendant can remain in the preceding ``ps``
                # snapshot while Darwin reports EPERM as it exits.  Omitting it
                # lets the sampler conservatively retain that member's last RSS
                # instead of discarding the aggregate sample.
                continue
            detail = (
                os.strerror(error_number) if error_number else "unknown error"
            )
            raise ProbeError(
                f"proc_pid_rusage({pid}) failed: {detail}"
            )
        return footprints


def _darwin_libproc_snapshot() -> dict[int, ProcessInfo]:
    """Collect a Darwin snapshot through libproc when ``ps`` is unavailable."""

    library_name = ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
    try:
        library = ctypes.CDLL(library_name)
    except OSError as error:
        raise ProbeError(f"cannot load Darwin libproc: {error}") from error

    library.proc_listallpids.argtypes = (ctypes.c_void_p, ctypes.c_int)
    library.proc_listallpids.restype = ctypes.c_int
    library.proc_pidinfo.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    )
    library.proc_pidinfo.restype = ctypes.c_int

    try:
        timebase = _darwin_mach_timebase()
    except ProbeError:
        # CPU time is optional telemetry. Process inventory and RSS remain
        # authoritative when the host cannot expose the Mach timebase.
        timebase = None

    estimated_count = library.proc_listallpids(None, 0)
    if estimated_count <= 0:
        raise ProbeError("Darwin libproc process enumeration failed")
    capacity = estimated_count + 1024
    pid_buffer = (ctypes.c_int * capacity)()
    listed_count = library.proc_listallpids(pid_buffer, ctypes.sizeof(pid_buffer))
    if listed_count <= 0:
        raise ProbeError("Darwin libproc process enumeration yielded no records")

    # PROC_PIDTBSDINFO and PROC_PIDTASKINFO are stable public libproc flavours.
    proc_pidtbsdinfo = 3
    proc_pidtaskinfo = 4
    bsd_buffer = ctypes.create_string_buffer(256)
    task_buffer = ctypes.create_string_buffer(256)
    records: dict[int, ProcessInfo] = {}
    for raw_pid in pid_buffer[: min(listed_count, capacity)]:
        pid = int(raw_pid)
        if pid <= 0:
            continue
        bsd_size = library.proc_pidinfo(
            pid,
            proc_pidtbsdinfo,
            0,
            bsd_buffer,
            len(bsd_buffer),
        )
        task_size = library.proc_pidinfo(
            pid,
            proc_pidtaskinfo,
            0,
            task_buffer,
            len(task_buffer),
        )
        if bsd_size < 104 or task_size < 16:
            continue
        try:
            record_pid, ppid, pgid = _parse_darwin_bsdinfo(
                bsd_buffer.raw[:bsd_size]
            )
            rss_bytes = _parse_darwin_taskinfo_rss(task_buffer.raw[:task_size])
        except ValueError:
            continue
        if record_pid != pid:
            continue
        cpu_seconds = None
        if timebase is not None:
            with suppress(ValueError):
                cpu_seconds = _parse_darwin_taskinfo_cpu_seconds(
                    task_buffer.raw[:task_size],
                    timebase_numer=timebase[0],
                    timebase_denom=timebase[1],
                )
        records[pid] = ProcessInfo(pid, ppid, pgid, rss_bytes, cpu_seconds)
    if not records:
        raise ProbeError("Darwin libproc process probe yielded no records")
    return records


def process_snapshot(system: str | None = None) -> dict[int, ProcessInfo]:
    """Collect one process snapshot on supported macOS and Linux hosts."""

    host = system or platform.system()
    if host == "Linux":
        try:
            return _linux_proc_snapshot()
        except ProbeError:
            return _ps_snapshot()
    if host == "Darwin":
        try:
            return _ps_snapshot()
        except ProbeError:
            return _darwin_libproc_snapshot()
    raise ProbeError(f"unsupported host for RSS monitoring: {host or '<unknown>'}")


class ProcessTreeSampler:
    """Track one process tree while retaining descendants that re-parent."""

    def __init__(self, root_pid: int, root_pgid: int) -> None:
        self.root_pid = root_pid
        self.root_pgid = root_pgid
        self._known_pids = {root_pid}

    def sample(
        self,
        records: dict[int, ProcessInfo],
        physical_footprint_probe: PhysicalFootprintProbe | None = None,
    ) -> MemorySample:
        children: dict[int, list[int]] = defaultdict(list)
        for record in records.values():
            children[record.ppid].append(record.pid)

        selected = {pid for pid in self._known_pids if pid in records} | {
            record.pid for record in records.values() if record.pgid == self.root_pgid
        }
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
        members = tuple(records[pid] for pid in sorted(selected))
        physical_footprint: int | None = None
        if physical_footprint_probe is not None:
            footprints = physical_footprint_probe(
                member.pid for member in members
            )
            # A member can exit between probes. Falling back to the member's
            # RSS for that one sample is conservative and avoids dropping its
            # last observed memory from the aggregate.
            physical_footprint = sum(
                footprints.get(member.pid, member.rss_bytes)
                for member in members
            )
        return MemorySample(
            rss_bytes=sum(record.rss_bytes for record in members),
            members=members,
            physical_footprint_bytes=physical_footprint,
        )


def _format_bytes(value: int) -> str:
    return f"{value / GIB:.3f} GiB"


def _guard_observation(sample: MemorySample) -> tuple[int, str]:
    """Return the conservative enforced byte count and stable reason code."""

    physical = sample.physical_footprint_bytes
    if physical is not None and physical >= sample.rss_bytes:
        return physical, DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON
    return sample.rss_bytes, RSS_LIMIT_REASON


def _normalized_exit_code(returncode: int) -> int:
    if returncode >= 0:
        return returncode
    return 128 + min(-returncode, 127)


def _signal_members(
    members: Iterable[ProcessInfo], root_pgid: int, selected_signal: int
) -> None:
    try:
        os.killpg(root_pgid, selected_signal)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass
    for member in members:
        if member.pgid == root_pgid:
            continue
        try:
            os.kill(member.pid, selected_signal)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass


def _terminate_tree(
    process: subprocess.Popen[bytes],
    sampler: ProcessTreeSampler,
    snapshotter: Snapshotter,
    *,
    grace_period: float,
    poll_interval: float,
) -> None:
    try:
        sample = sampler.sample(snapshotter())
    except ProbeError:
        sample = MemorySample(0, ())
    _signal_members(sample.members, sampler.root_pgid, signal.SIGTERM)

    deadline = time.monotonic() + grace_period
    while time.monotonic() < deadline:
        try:
            sample = sampler.sample(snapshotter())
        except ProbeError:
            sample = MemorySample(0, ())
        if process.poll() is not None and not sample.members:
            return
        time.sleep(min(poll_interval, max(deadline - time.monotonic(), 0.01)))

    try:
        sample = sampler.sample(snapshotter())
    except ProbeError:
        sample = MemorySample(0, ())
    _signal_members(sample.members, sampler.root_pgid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=max(grace_period, 1.0))


def _emit_final_report(
    *,
    report_path: Path | None,
    bound_result_path: Path | None,
    command: Sequence[str],
    working_directory: str,
    started_at_utc: str,
    started_monotonic: float,
    child_pid: int | None,
    child_exit_code: int | None,
    watchdog_exit_code: int,
    outcome: str,
    reason: str | None,
    limit_bytes: int,
    poll_interval: float,
    terminate_grace: float,
    metric: str,
    probe_sample_count: int,
    probe_failure_count: int,
    maximum_consecutive_probe_failures: int,
    peak_rss_bytes: int,
    peak_physical_footprint_bytes: int | None,
    peak_guard_bytes: int,
    peak_processes: int,
    stderr: TextIO,
) -> int:
    """Write one terminal content-addressed report and return the final status."""

    if report_path is None:
        return watchdog_exit_code

    binding_error: str | None = None
    bound_result: dict[str, object] | None = None
    if bound_result_path is not None:
        try:
            bound_result = _result_identity(bound_result_path)
        except OSError as error:
            binding_error = str(error)
            if watchdog_exit_code == 0:
                watchdog_exit_code = WATCHDOG_ERROR_EXIT_CODE
                outcome = "result-binding-failed"
                reason = "bound-result-unavailable"

    enforcement_completed = (
        watchdog_exit_code == 0
        and child_exit_code == 0
        and outcome == "command-finished"
    )
    payload: dict[str, object] = {
        "kind": WATCHDOG_REPORT_KIND,
        "schema_version": WATCHDOG_REPORT_SCHEMA,
        "complete": True,
        "passes": (
            watchdog_exit_code == 0
            and child_exit_code == 0
            and enforcement_completed
            and (bound_result_path is None or bound_result is not None)
        ),
        "watchdog": _result_identity(Path(__file__).resolve()),
        "working_directory": working_directory,
        "execution": {
            "command": list(command),
            "command_sha256": _sha256(list(command)),
            "started_at_utc": started_at_utc,
            "finished_at_utc": _utc_now(),
            "elapsed_wall_seconds": max(time.monotonic() - started_monotonic, 0.0),
            "child_pid": child_pid,
            "child_exit_code": child_exit_code,
            "watchdog_exit_code": watchdog_exit_code,
            "outcome": outcome,
            "reason": reason,
        },
        "enforcement": {
            "scope": WATCHDOG_SCOPE,
            "limit_bytes": limit_bytes,
            "poll_interval_seconds": poll_interval,
            "terminate_grace_seconds": terminate_grace,
            "metric": metric,
            "probe_sample_count": probe_sample_count,
            "probe_failure_count": probe_failure_count,
            "maximum_consecutive_probe_failures": (
                maximum_consecutive_probe_failures
            ),
            "completed_under_retry_policy": enforcement_completed,
            "peak_rss_bytes": peak_rss_bytes,
            "peak_physical_footprint_bytes": peak_physical_footprint_bytes,
            "peak_guard_bytes": peak_guard_bytes,
            "peak_processes": peak_processes,
        },
        "result_binding": {
            "requested": bound_result_path is not None,
            "requested_path": (
                None if bound_result_path is None else str(bound_result_path)
            ),
            "identity": bound_result,
            "error": binding_error,
        },
    }
    payload["content_sha256"] = _sha256(payload)
    try:
        _write_json_atomic(report_path, payload)
    except OSError as error:
        print(
            f"memory-watchdog: cannot write final report: {error}",
            file=stderr,
            flush=True,
        )
        return WATCHDOG_ERROR_EXIT_CODE
    return watchdog_exit_code


def run_guarded(
    command: Sequence[str],
    *,
    limit_bytes: int,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    terminate_grace: float = DEFAULT_TERMINATE_GRACE,
    snapshotter: Snapshotter = process_snapshot,
    physical_footprint_probe: PhysicalFootprintProbe | None = None,
    stderr: TextIO = sys.stderr,
    report_path: Path | None = None,
    bound_result_path: Path | None = None,
) -> int:
    """Run ``command`` and return its shell-compatible exit status."""

    if not command:
        raise ValueError("command must not be empty")
    if limit_bytes <= 0:
        raise ValueError("limit_bytes must be positive")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    if terminate_grace < 0:
        raise ValueError("terminate_grace must be non-negative")

    started_at_utc = _utc_now()
    started_monotonic = time.monotonic()
    working_directory = str(Path.cwd())
    metric = (
        "max(process-tree-rss,darwin-process-tree-physical-footprint)"
        if physical_footprint_probe is not None
        else "process-tree-rss"
    )
    child_pid: int | None = None
    peak_rss = 0
    peak_physical_footprint: int | None = (
        0 if physical_footprint_probe is not None else None
    )
    peak_guard = 0
    peak_processes = 0
    probe_sample_count = 0
    probe_failure_count = 0
    maximum_consecutive_probe_failures = 0
    report_emitted = False

    def finish(
        exit_code: int,
        *,
        child_exit_code: int | None,
        outcome: str,
        reason: str | None = None,
    ) -> int:
        nonlocal report_emitted
        report_emitted = True
        return _emit_final_report(
            report_path=report_path,
            bound_result_path=bound_result_path,
            command=command,
            working_directory=working_directory,
            started_at_utc=started_at_utc,
            started_monotonic=started_monotonic,
            child_pid=child_pid,
            child_exit_code=child_exit_code,
            watchdog_exit_code=exit_code,
            outcome=outcome,
            reason=reason,
            limit_bytes=limit_bytes,
            poll_interval=poll_interval,
            terminate_grace=terminate_grace,
            metric=metric,
            probe_sample_count=probe_sample_count,
            probe_failure_count=probe_failure_count,
            maximum_consecutive_probe_failures=(
                maximum_consecutive_probe_failures
            ),
            peak_rss_bytes=peak_rss,
            peak_physical_footprint_bytes=peak_physical_footprint,
            peak_guard_bytes=peak_guard,
            peak_processes=peak_processes,
            stderr=stderr,
        )

    try:
        process = subprocess.Popen(tuple(command), start_new_session=True)
    except FileNotFoundError:
        print(f"memory-watchdog: command not found: {command[0]}", file=stderr)
        return finish(
            127,
            child_exit_code=None,
            outcome="command-not-found",
            reason="command-not-found",
        )
    except OSError as error:
        print(f"memory-watchdog: cannot start command: {error}", file=stderr)
        return finish(
            126,
            child_exit_code=None,
            outcome="command-start-failed",
            reason="command-start-failed",
        )

    child_pid = process.pid
    sampler = ProcessTreeSampler(process.pid, process.pid)
    print(
        "memory-watchdog: guarding"
        f" pid={process.pid} limit={_format_bytes(limit_bytes)}"
        f" metric={metric}"
        f" command={shlex.join(command)}",
        file=stderr,
        flush=True,
    )

    received_signal: int | None = None
    previous_handlers: dict[int, signal.Handlers] = {}

    def remember_signal(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum

    if threading.current_thread() is threading.main_thread():
        for selected_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[selected_signal] = signal.getsignal(selected_signal)
            signal.signal(selected_signal, remember_signal)

    consecutive_probe_failures = 0
    pending_probe_reason: str | None = None
    try:
        while True:
            if received_signal is not None:
                _terminate_tree(
                    process,
                    sampler,
                    snapshotter,
                    grace_period=terminate_grace,
                    poll_interval=poll_interval,
                )
                return finish(
                    128 + received_signal,
                    child_exit_code=(
                        None
                        if process.poll() is None
                        else _normalized_exit_code(process.returncode)
                    ),
                    outcome="watchdog-signalled",
                    reason=f"signal-{received_signal}",
                )

            if pending_probe_reason is not None:
                returncode = process.poll()
                if returncode is not None:
                    print(
                        "memory-watchdog: command exited while memory"
                        " enforcement was unavailable"
                        f" reason={pending_probe_reason}"
                        f" child_exit={_normalized_exit_code(returncode)}",
                        file=stderr,
                        flush=True,
                    )
                    return finish(
                        WATCHDOG_ERROR_EXIT_CODE,
                        child_exit_code=_normalized_exit_code(returncode),
                        outcome="memory-enforcement-unavailable",
                        reason=pending_probe_reason,
                    )

            probe_reason = MEMORY_PROBE_REASON
            try:
                records = snapshotter()
                if physical_footprint_probe is not None:
                    probe_reason = DARWIN_PHYSICAL_FOOTPRINT_PROBE_REASON
                sample = sampler.sample(
                    records,
                    physical_footprint_probe=physical_footprint_probe,
                )
            except ProbeError as error:
                consecutive_probe_failures += 1
                probe_failure_count += 1
                maximum_consecutive_probe_failures = max(
                    maximum_consecutive_probe_failures,
                    consecutive_probe_failures,
                )
                pending_probe_reason = probe_reason
                print(
                    "memory-watchdog: memory probe failed"
                    f" ({consecutive_probe_failures}/3)"
                    f" reason={probe_reason}: {error}",
                    file=stderr,
                    flush=True,
                )
                returncode = process.poll()
                if returncode is not None:
                    print(
                        "memory-watchdog: command exited while memory"
                        " enforcement was unavailable"
                        f" reason={probe_reason}"
                        f" child_exit={_normalized_exit_code(returncode)}",
                        file=stderr,
                        flush=True,
                    )
                    return finish(
                        WATCHDOG_ERROR_EXIT_CODE,
                        child_exit_code=_normalized_exit_code(returncode),
                        outcome="memory-enforcement-unavailable",
                        reason=probe_reason,
                    )
                if consecutive_probe_failures >= 3:
                    _terminate_tree(
                        process,
                        sampler,
                        snapshotter,
                        grace_period=terminate_grace,
                        poll_interval=poll_interval,
                    )
                    print(
                        "memory-watchdog: terminating after repeated memory"
                        f" probe failures reason={probe_reason}",
                        file=stderr,
                        flush=True,
                    )
                    return finish(
                        WATCHDOG_ERROR_EXIT_CODE,
                        child_exit_code=(
                            None
                            if process.poll() is None
                            else _normalized_exit_code(process.returncode)
                        ),
                        outcome="memory-probe-failed",
                        reason=probe_reason,
                    )
            else:
                probe_sample_count += 1
                if pending_probe_reason is not None:
                    returncode = process.poll()
                    if returncode is not None:
                        print(
                            "memory-watchdog: command exited while memory"
                            " enforcement was unavailable"
                            f" reason={pending_probe_reason}"
                            f" child_exit={_normalized_exit_code(returncode)}",
                            file=stderr,
                            flush=True,
                        )
                        return finish(
                            WATCHDOG_ERROR_EXIT_CODE,
                            child_exit_code=_normalized_exit_code(returncode),
                            outcome="memory-enforcement-unavailable",
                            reason=pending_probe_reason,
                        )
                consecutive_probe_failures = 0
                pending_probe_reason = None
                peak_rss = max(peak_rss, sample.rss_bytes)
                if sample.physical_footprint_bytes is not None:
                    peak_physical_footprint = max(
                        peak_physical_footprint or 0,
                        sample.physical_footprint_bytes,
                    )
                guard_bytes, limit_reason = _guard_observation(sample)
                peak_guard = max(peak_guard, guard_bytes)
                peak_processes = max(peak_processes, len(sample.members))
                if guard_bytes > limit_bytes:
                    physical_detail = (
                        _format_bytes(sample.physical_footprint_bytes)
                        if sample.physical_footprint_bytes is not None
                        else "unavailable"
                    )
                    print(
                        "memory-watchdog: memory limit exceeded"
                        f" reason={limit_reason}"
                        f" observed={_format_bytes(guard_bytes)}"
                        f" limit={_format_bytes(limit_bytes)}"
                        f" rss={_format_bytes(sample.rss_bytes)}"
                        f" physical_footprint={physical_detail}"
                        f" processes={len(sample.members)};"
                        " terminating tree",
                        file=stderr,
                        flush=True,
                    )
                    _terminate_tree(
                        process,
                        sampler,
                        snapshotter,
                        grace_period=terminate_grace,
                        poll_interval=poll_interval,
                    )
                    return finish(
                        MEMORY_LIMIT_EXIT_CODE,
                        child_exit_code=(
                            None
                            if process.poll() is None
                            else _normalized_exit_code(process.returncode)
                        ),
                        outcome="memory-limit-exceeded",
                        reason=limit_reason,
                    )

            returncode = process.poll()
            if returncode is not None:
                if consecutive_probe_failures:
                    assert pending_probe_reason is not None
                    print(
                        "memory-watchdog: command exited while memory"
                        " enforcement was unavailable"
                        f" reason={pending_probe_reason}"
                        f" child_exit={_normalized_exit_code(returncode)}",
                        file=stderr,
                        flush=True,
                    )
                    return finish(
                        WATCHDOG_ERROR_EXIT_CODE,
                        child_exit_code=_normalized_exit_code(returncode),
                        outcome="memory-enforcement-unavailable",
                        reason=pending_probe_reason,
                    )
                normalized = _normalized_exit_code(returncode)
                footprint_detail = (
                    _format_bytes(peak_physical_footprint)
                    if peak_physical_footprint is not None
                    else "unavailable"
                )
                print(
                    "memory-watchdog: command finished"
                    f" exit={normalized} peak_rss={_format_bytes(peak_rss)}"
                    f" peak_physical_footprint={footprint_detail}"
                    f" peak_guard={_format_bytes(peak_guard)}"
                    f" peak_processes={peak_processes}",
                    file=stderr,
                    flush=True,
                )
                return finish(
                    normalized,
                    child_exit_code=normalized,
                    outcome="command-finished",
                    reason=None,
                )
            time.sleep(poll_interval)
    finally:
        for selected_signal, previous in previous_handlers.items():
            signal.signal(selected_signal, previous)
        if process.poll() is None:
            _terminate_tree(
                process,
                sampler,
                snapshotter,
                grace_period=terminate_grace,
                poll_interval=poll_interval,
            )
        if not report_emitted:
            finish(
                WATCHDOG_ERROR_EXIT_CODE,
                child_exit_code=(
                    None
                    if process.poll() is None
                    else _normalized_exit_code(process.returncode)
                ),
                outcome="watchdog-exception",
                reason="watchdog-exception",
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "run a command and terminate its process tree when conservative "
            "aggregate memory exceeds a limit"
        )
    )
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument(
        "--limit-gib",
        type=_positive_finite,
        help=f"memory limit in GiB (default: {DEFAULT_LIMIT_GIB:g})",
    )
    limits.add_argument(
        "--limit-mib",
        type=_positive_finite,
        help="memory limit in MiB; useful for focused watchdog tests",
    )
    parser.add_argument(
        "--poll-interval",
        type=_positive_finite,
        default=DEFAULT_POLL_INTERVAL,
        help=f"seconds between probes (default: {DEFAULT_POLL_INTERVAL:g})",
    )
    parser.add_argument(
        "--terminate-grace",
        type=_nonnegative_finite,
        default=DEFAULT_TERMINATE_GRACE,
        help=(
            "seconds between TERM and KILL when stopping a tree "
            f"(default: {DEFAULT_TERMINATE_GRACE:g})"
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help=(
            "write one terminal content-addressed execution report on success "
            "or failure"
        ),
    )
    parser.add_argument(
        "--bind-result-json",
        type=Path,
        help=(
            "content-address this command result in --report-json; the guarded "
            "command must create it"
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run, conventionally preceded by --",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    command = list(arguments.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")
    if arguments.bind_result_json is not None and arguments.report_json is None:
        parser.error("--bind-result-json requires --report-json")
    if (
        arguments.report_json is not None
        and arguments.bind_result_json is not None
        and Path(os.path.abspath(arguments.report_json.expanduser()))
        == Path(os.path.abspath(arguments.bind_result_json.expanduser()))
    ):
        parser.error("--report-json and --bind-result-json must be different files")

    if arguments.limit_mib is not None:
        limit_bytes = int(arguments.limit_mib * MIB)
    else:
        limit_gib = arguments.limit_gib or DEFAULT_LIMIT_GIB
        limit_bytes = int(limit_gib * GIB)
    physical_footprint_probe: PhysicalFootprintProbe | None = None
    if platform.system() == "Darwin":
        try:
            physical_footprint_probe = DarwinPhysicalFootprintProbe()
        except ProbeError as error:
            started_at_utc = _utc_now()
            started_monotonic = time.monotonic()
            print(
                "memory-watchdog: Darwin physical-footprint probe unavailable"
                f" reason={DARWIN_PHYSICAL_FOOTPRINT_PROBE_REASON}: {error}",
                file=sys.stderr,
                flush=True,
            )
            return _emit_final_report(
                report_path=arguments.report_json,
                bound_result_path=arguments.bind_result_json,
                command=command,
                working_directory=str(Path.cwd()),
                started_at_utc=started_at_utc,
                started_monotonic=started_monotonic,
                child_pid=None,
                child_exit_code=None,
                watchdog_exit_code=WATCHDOG_ERROR_EXIT_CODE,
                outcome="memory-probe-initialization-failed",
                reason=DARWIN_PHYSICAL_FOOTPRINT_PROBE_REASON,
                limit_bytes=limit_bytes,
                poll_interval=arguments.poll_interval,
                terminate_grace=arguments.terminate_grace,
                metric=(
                    "max(process-tree-rss,"
                    "darwin-process-tree-physical-footprint)"
                ),
                probe_sample_count=0,
                probe_failure_count=1,
                maximum_consecutive_probe_failures=1,
                peak_rss_bytes=0,
                peak_physical_footprint_bytes=0,
                peak_guard_bytes=0,
                peak_processes=0,
                stderr=sys.stderr,
            )
    return run_guarded(
        command,
        limit_bytes=limit_bytes,
        poll_interval=arguments.poll_interval,
        terminate_grace=arguments.terminate_grace,
        physical_footprint_probe=physical_footprint_probe,
        report_path=arguments.report_json,
        bound_result_path=arguments.bind_result_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
