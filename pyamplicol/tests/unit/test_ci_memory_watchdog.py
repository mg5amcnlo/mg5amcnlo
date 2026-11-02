# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WATCHDOG = ROOT / "tools/ci/memory_watchdog.py"
if WATCHDOG.is_file():
    from tools.ci import memory_watchdog as watchdog
else:  # CI-only tools are intentionally absent from unpacked source distributions.
    watchdog = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    watchdog is None,
    reason="the CI-only memory watchdog is intentionally absent from the sdist",
)


def test_argument_validation_and_default_limit() -> None:
    parser = watchdog._parser()
    parsed = parser.parse_args(("--", sys.executable, "-c", "pass"))

    assert parsed.limit_gib is None
    assert parsed.limit_mib is None
    assert parsed.report_json is None
    assert parsed.bind_result_json is None
    assert watchdog.DEFAULT_LIMIT_GIB == 30.0
    with pytest.raises(SystemExit) as missing:
        watchdog.main(())
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as invalid:
        watchdog.main(("--limit-gib", "0", "--", "true"))
    assert invalid.value.code == 2
    with pytest.raises(SystemExit) as unreported_binding:
        watchdog.main(("--bind-result-json", "result.json", "--", "true"))
    assert unreported_binding.value.code == 2
    with pytest.raises(SystemExit) as aliased_outputs:
        watchdog.main(
            (
                "--report-json",
                "evidence.json",
                "--bind-result-json",
                "evidence.json",
                "--",
                "true",
            )
        )
    assert aliased_outputs.value.code == 2


def test_final_report_is_content_addressed_and_binds_result(tmp_path: Path) -> None:
    result_path = tmp_path / "campaign.json"
    result_path.write_text('{"complete":true}\n', encoding="ascii")
    report_path = tmp_path / "watchdog.json"

    returncode = watchdog._emit_final_report(
        report_path=report_path,
        bound_result_path=result_path,
        command=("python", "paired.py", "--result-json", str(result_path)),
        working_directory=str(tmp_path),
        started_at_utc="2026-07-31T00:00:00+00:00",
        started_monotonic=time.monotonic() - 1.0,
        child_pid=123,
        child_exit_code=0,
        watchdog_exit_code=0,
        outcome="command-finished",
        reason=None,
        limit_bytes=30 * watchdog.GIB,
        poll_interval=0.25,
        terminate_grace=5.0,
        metric="process-tree-rss",
        probe_sample_count=4,
        probe_failure_count=2,
        maximum_consecutive_probe_failures=1,
        peak_rss_bytes=1024,
        peak_physical_footprint_bytes=None,
        peak_guard_bytes=1024,
        peak_processes=2,
        stderr=io.StringIO(),
    )

    payload = json.loads(report_path.read_text(encoding="ascii"))
    unsigned = dict(payload)
    digest = unsigned.pop("content_sha256")
    assert returncode == 0
    assert digest == watchdog._sha256(unsigned)
    assert payload["passes"] is True
    assert payload["enforcement"]["completed_under_retry_policy"] is True
    assert payload["enforcement"]["probe_failure_count"] == 2
    assert payload["enforcement"]["maximum_consecutive_probe_failures"] == 1
    identity = payload["result_binding"]["identity"]
    assert identity["resolved_path"] == str(result_path.resolve())
    assert identity["sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()


def test_requested_report_is_emitted_when_command_cannot_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "watchdog-failure.json"

    def missing(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr(watchdog.subprocess, "Popen", missing)
    returncode = watchdog.run_guarded(
        ("missing-command",),
        limit_bytes=watchdog.MIB,
        report_path=report_path,
        stderr=io.StringIO(),
    )

    payload = json.loads(report_path.read_text(encoding="ascii"))
    assert returncode == 127
    assert payload["complete"] is True
    assert payload["passes"] is False
    assert payload["execution"]["outcome"] == "command-not-found"
    assert payload["execution"]["watchdog_exit_code"] == 127


def test_success_without_requested_bound_result_fails_closed(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "watchdog.json"
    missing_result = tmp_path / "missing-campaign.json"

    returncode = watchdog._emit_final_report(
        report_path=report_path,
        bound_result_path=missing_result,
        command=("python", "paired.py"),
        working_directory=str(tmp_path),
        started_at_utc="2026-07-31T00:00:00+00:00",
        started_monotonic=time.monotonic() - 1.0,
        child_pid=123,
        child_exit_code=0,
        watchdog_exit_code=0,
        outcome="command-finished",
        reason=None,
        limit_bytes=30 * watchdog.GIB,
        poll_interval=0.25,
        terminate_grace=5.0,
        metric="process-tree-rss",
        probe_sample_count=4,
        probe_failure_count=0,
        maximum_consecutive_probe_failures=0,
        peak_rss_bytes=1024,
        peak_physical_footprint_bytes=None,
        peak_guard_bytes=1024,
        peak_processes=2,
        stderr=io.StringIO(),
    )

    payload = json.loads(report_path.read_text(encoding="ascii"))
    assert returncode == watchdog.WATCHDOG_ERROR_EXIT_CODE
    assert payload["passes"] is False
    assert payload["execution"]["outcome"] == "result-binding-failed"
    assert payload["result_binding"]["identity"] is None
    assert payload["result_binding"]["error"]


def test_linux_proc_parsers_handle_parentheses_and_rss() -> None:
    stat = "412 (worker (phase 2)) S 99 401 401 0 -1 0"
    status = "Name:\tworker\nVmPeak:\t9000 kB\nVmRSS:\t1234 kB\n"

    assert watchdog._parse_proc_stat(stat) == (99, 401)
    assert watchdog._parse_proc_status_rss(status) == 1234 * 1024
    assert watchdog._parse_proc_status_rss("Name:\tworker\n") == 0


def test_ps_parser_and_tree_sampler_include_group_and_escaped_descendants() -> None:
    records = watchdog._parse_ps_output(
        """
        100 1 100 1024
        101 100 100 2048
        102 101 102 4096
        103 1 100 512
        900 1 900 9999
        """
    )
    sampler = watchdog.ProcessTreeSampler(root_pid=100, root_pgid=100)

    footprints = {
        100: 8 * watchdog.MIB,
        101: 7 * watchdog.MIB,
        103: 6 * watchdog.MIB,
    }
    sample = sampler.sample(
        records,
        physical_footprint_probe=lambda pids: {
            pid: footprints[pid] for pid in pids if pid in footprints
        },
    )

    assert tuple(member.pid for member in sample.members) == (100, 101, 102, 103)
    assert sample.rss_bytes == (1024 + 2048 + 4096 + 512) * 1024
    assert sample.physical_footprint_bytes == (
        8 * watchdog.MIB
        + 7 * watchdog.MIB
        + 4096 * 1024  # Conservative RSS fallback for the raced-out PID.
        + 6 * watchdog.MIB
    )
    assert watchdog._guard_observation(sample) == (
        sample.physical_footprint_bytes,
        watchdog.DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON,
    )


def test_raced_out_footprint_member_falls_back_to_its_last_rss() -> None:
    records = {
        100: watchdog.ProcessInfo(100, 1, 100, 3 * watchdog.MIB),
        101: watchdog.ProcessInfo(101, 100, 100, 5 * watchdog.MIB),
    }
    sampler = watchdog.ProcessTreeSampler(root_pid=100, root_pgid=100)

    sample = sampler.sample(
        records,
        physical_footprint_probe=lambda _pids: {100: 7 * watchdog.MIB},
    )

    assert sample.rss_bytes == 8 * watchdog.MIB
    assert sample.physical_footprint_bytes == 12 * watchdog.MIB


def test_guard_observation_falls_back_to_rss_without_darwin_metric() -> None:
    sample = watchdog.MemorySample(
        rss_bytes=11 * watchdog.MIB,
        members=(),
    )
    lower_footprint = watchdog.MemorySample(
        rss_bytes=11 * watchdog.MIB,
        members=(),
        physical_footprint_bytes=9 * watchdog.MIB,
    )

    assert watchdog._guard_observation(sample) == (
        11 * watchdog.MIB,
        watchdog.RSS_LIMIT_REASON,
    )
    assert watchdog._guard_observation(lower_footprint) == (
        11 * watchdog.MIB,
        watchdog.RSS_LIMIT_REASON,
    )


def test_footprint_over_rss_terminates_with_stable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 43210

        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            del timeout
            self.returncode = self.returncode or -9
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(
        watchdog.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    def record_signal(_pgid: int, selected_signal: int) -> None:
        process.returncode = -selected_signal

    monkeypatch.setattr(watchdog.os, "killpg", record_signal)
    monkeypatch.setattr(watchdog.os, "kill", lambda *_args: None)

    def snapshotter() -> dict[int, watchdog.ProcessInfo]:
        return {
            process.pid: watchdog.ProcessInfo(
                process.pid,
                1,
                process.pid,
                8 * watchdog.MIB,
            )
        }

    stderr = io.StringIO()

    returncode = watchdog.run_guarded(
        ("synthetic-child",),
        limit_bytes=16 * watchdog.MIB,
        poll_interval=0.001,
        terminate_grace=0,
        snapshotter=snapshotter,
        physical_footprint_probe=lambda pids: {
            pid: 20 * watchdog.MIB for pid in pids
        },
        stderr=stderr,
    )

    assert returncode == watchdog.MEMORY_LIMIT_EXIT_CODE
    assert (
        f"reason={watchdog.DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON}" in stderr.getvalue()
    )
    assert "rss=0.008 GiB" in stderr.getvalue()
    assert "physical_footprint=0.020 GiB" in stderr.getvalue()


def test_short_child_cannot_succeed_during_required_footprint_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FinishedProcess:
        pid = 43211
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float) -> int:
            del timeout
            return self.returncode

    process = FinishedProcess()
    monkeypatch.setattr(
        watchdog.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    def snapshotter() -> dict[int, watchdog.ProcessInfo]:
        return {
            process.pid: watchdog.ProcessInfo(
                process.pid,
                1,
                process.pid,
                8 * watchdog.MIB,
            )
        }

    def unavailable_footprint(_pids: object) -> dict[int, int]:
        raise watchdog.ProbeError("synthetic footprint probe failure")

    stderr = io.StringIO()
    returncode = watchdog.run_guarded(
        ("short-child",),
        limit_bytes=16 * watchdog.MIB,
        poll_interval=0.001,
        snapshotter=snapshotter,
        physical_footprint_probe=unavailable_footprint,
        stderr=stderr,
    )

    assert returncode == watchdog.WATCHDOG_ERROR_EXIT_CODE
    assert (
        "command exited while memory enforcement was unavailable"
        in stderr.getvalue()
    )
    assert (
        f"reason={watchdog.DARWIN_PHYSICAL_FOOTPRINT_PROBE_REASON}"
        in stderr.getvalue()
    )
    assert "child_exit=0" in stderr.getvalue()


def test_child_exit_before_retry_cannot_be_cleared_by_zombie_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitingProcess:
        pid = 43213

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            return self.returncode

        def wait(self, timeout: float) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

    process = ExitingProcess()
    monkeypatch.setattr(
        watchdog.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    snapshot_count = 0

    def snapshotter() -> dict[int, watchdog.ProcessInfo]:
        nonlocal snapshot_count
        snapshot_count += 1
        return {
            process.pid: watchdog.ProcessInfo(
                process.pid,
                1,
                process.pid,
                8 * watchdog.MIB,
            )
        }

    footprint_count = 0

    def footprint(_pids: object) -> dict[int, int]:
        nonlocal footprint_count
        footprint_count += 1
        if footprint_count == 1:
            raise watchdog.ProbeError("synthetic footprint probe failure")
        return {process.pid: 0}

    def exit_during_retry_sleep(_duration: float) -> None:
        process.returncode = 0

    monkeypatch.setattr(watchdog.time, "sleep", exit_during_retry_sleep)
    stderr = io.StringIO()
    returncode = watchdog.run_guarded(
        ("zombie-child",),
        limit_bytes=16 * watchdog.MIB,
        poll_interval=0.001,
        snapshotter=snapshotter,
        physical_footprint_probe=footprint,
        stderr=stderr,
    )

    assert returncode == watchdog.WATCHDOG_ERROR_EXIT_CODE
    assert process.poll_count >= 3
    assert snapshot_count == 1
    assert footprint_count == 1
    assert (
        f"reason={watchdog.DARWIN_PHYSICAL_FOOTPRINT_PROBE_REASON}"
        in stderr.getvalue()
    )
    assert "child_exit=0" in stderr.getvalue()


def test_repeated_required_footprint_probe_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RunningProcess:
        pid = 43212

        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            del timeout
            self.returncode = self.returncode or -9
            return self.returncode

    process = RunningProcess()
    monkeypatch.setattr(
        watchdog.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    def record_signal(_pgid: int, selected_signal: int) -> None:
        process.returncode = -selected_signal

    monkeypatch.setattr(watchdog.os, "killpg", record_signal)
    monkeypatch.setattr(watchdog.os, "kill", lambda *_args: None)
    probe_count = 0

    def snapshotter() -> dict[int, watchdog.ProcessInfo]:
        return {
            process.pid: watchdog.ProcessInfo(
                process.pid,
                1,
                process.pid,
                8 * watchdog.MIB,
            )
        }

    def unavailable_footprint(_pids: object) -> dict[int, int]:
        nonlocal probe_count
        probe_count += 1
        raise watchdog.ProbeError("synthetic footprint probe failure")

    stderr = io.StringIO()
    returncode = watchdog.run_guarded(
        ("long-child",),
        limit_bytes=16 * watchdog.MIB,
        poll_interval=0.001,
        terminate_grace=0,
        snapshotter=snapshotter,
        physical_footprint_probe=unavailable_footprint,
        stderr=stderr,
    )

    assert returncode == watchdog.WATCHDOG_ERROR_EXIT_CODE
    assert probe_count >= 3
    assert "terminating after repeated memory probe failures" in stderr.getvalue()
    assert (
        f"reason={watchdog.DARWIN_PHYSICAL_FOOTPRINT_PROBE_REASON}"
        in stderr.getvalue()
    )


def test_main_injects_darwin_footprint_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def probe(_pids: object) -> dict[int, int]:
        return {}

    def fake_run_guarded(
        command: tuple[str, ...] | list[str],
        **kwargs: object,
    ) -> int:
        captured["command"] = tuple(command)
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(watchdog.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        watchdog,
        "DarwinPhysicalFootprintProbe",
        lambda: probe,
    )
    monkeypatch.setattr(watchdog, "run_guarded", fake_run_guarded)

    assert watchdog.main(("--limit-mib", "32", "--", "synthetic-child")) == 0
    assert captured["physical_footprint_probe"] is probe


def test_main_fails_closed_when_darwin_probe_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_probe() -> object:
        raise watchdog.ProbeError("synthetic constructor failure")

    monkeypatch.setattr(watchdog.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(watchdog, "DarwinPhysicalFootprintProbe", fail_probe)

    returncode = watchdog.main(
        ("--limit-mib", "32", "--", "synthetic-child")
    )

    assert returncode == watchdog.WATCHDOG_ERROR_EXIT_CODE
    assert (
        f"reason={watchdog.DARWIN_PHYSICAL_FOOTPRINT_PROBE_REASON}"
        in capsys.readouterr().err
    )


def test_main_uses_rss_only_on_non_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_guarded(
        command: tuple[str, ...] | list[str],
        **kwargs: object,
    ) -> int:
        captured["command"] = tuple(command)
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(watchdog.platform, "system", lambda: "Linux")
    monkeypatch.setattr(watchdog, "run_guarded", fake_run_guarded)

    assert watchdog.main(("--limit-mib", "32", "--", "synthetic-child")) == 0
    assert captured["physical_footprint_probe"] is None
    assert captured["limit_bytes"] == 32 * watchdog.MIB


def test_darwin_libproc_parsers_extract_identity_rss_and_cpu() -> None:
    bsd = bytearray(136)
    struct.pack_into("=I", bsd, 12, 123)
    struct.pack_into("=I", bsd, 16, 45)
    struct.pack_into("=I", bsd, 100, 67)
    task = bytearray(96)
    struct.pack_into("=Q", task, 8, 987_654_321)
    struct.pack_into("=Q", task, 16, 4_000_000_000)
    struct.pack_into("=Q", task, 24, 2_000_000_000)
    rusage = bytearray(96)
    struct.pack_into("=Q", rusage, 72, 1_234_567_890)

    assert watchdog._parse_darwin_bsdinfo(bytes(bsd)) == (123, 45, 67)
    assert watchdog._parse_darwin_taskinfo_rss(bytes(task)) == 987_654_321
    assert watchdog._parse_darwin_taskinfo_cpu_seconds(
        bytes(task),
        timebase_numer=1,
        timebase_denom=2,
    ) == 3.0
    assert watchdog._parse_darwin_rusage_phys_footprint(bytes(rusage)) == 1_234_567_890
    with pytest.raises(ValueError, match="incomplete Darwin proc_bsdinfo"):
        watchdog._parse_darwin_bsdinfo(b"short")
    with pytest.raises(ValueError, match="incomplete Darwin proc_taskinfo"):
        watchdog._parse_darwin_taskinfo_rss(b"short")
    with pytest.raises(ValueError, match="incomplete Darwin proc_taskinfo CPU"):
        watchdog._parse_darwin_taskinfo_cpu_seconds(
            b"short", timebase_numer=1, timebase_denom=1
        )
    with pytest.raises(ValueError, match="invalid Darwin Mach timebase"):
        watchdog._parse_darwin_taskinfo_cpu_seconds(
            bytes(task), timebase_numer=1, timebase_denom=0
        )
    with pytest.raises(ValueError, match="incomplete Darwin rusage_info_v0"):
        watchdog._parse_darwin_rusage_phys_footprint(b"short")


def test_darwin_mach_timebase_is_validated() -> None:
    class FakeMachTimebaseInfo:
        argtypes: object = None
        restype: object = None

        def __init__(self, *, result: int = 0, numer: int = 125, denom: int = 3):
            self.result = result
            self.numer = numer
            self.denom = denom

        def __call__(self, pointer: object) -> int:
            info = ctypes.cast(
                pointer,
                ctypes.POINTER(watchdog._MachTimebaseInfo),
            ).contents
            info.numer = self.numer
            info.denom = self.denom
            return self.result

    class FakeLibrary:
        def __init__(self, function: FakeMachTimebaseInfo) -> None:
            self.mach_timebase_info = function

    assert watchdog._darwin_mach_timebase(
        FakeLibrary(FakeMachTimebaseInfo())
    ) == (125, 3)
    with pytest.raises(watchdog.ProbeError, match="mach_timebase_info failed"):
        watchdog._darwin_mach_timebase(
            FakeLibrary(FakeMachTimebaseInfo(result=5))
        )
    with pytest.raises(watchdog.ProbeError, match="mach_timebase_info failed"):
        watchdog._darwin_mach_timebase(
            FakeLibrary(FakeMachTimebaseInfo(denom=0))
        )


def test_fake_darwin_libproc_handles_success_race_and_unexpected_error() -> None:
    class FakeProcPidRusage:
        argtypes: object = None
        restype: object = None

        def __call__(
            self,
            pid: int,
            _flavor: int,
            buffer: object,
        ) -> int:
            if pid == 1:
                payload = struct.pack("=Q", 987_654_321)
                ctypes.memmove(
                    ctypes.addressof(buffer) + 72,
                    payload,
                    len(payload),
                )
                return 0
            ctypes.set_errno(errno.ESRCH if pid == 2 else errno.EACCES)
            return -1

    class FakeLibrary:
        proc_pid_rusage = FakeProcPidRusage()

    probe = watchdog.DarwinPhysicalFootprintProbe(
        FakeLibrary(),
        pid_exists=lambda _pid: True,
    )

    assert probe((1, 2)) == {1: 987_654_321}
    with pytest.raises(watchdog.ProbeError, match="Permission denied"):
        probe((3,))


def test_darwin_footprint_probe_retries_transient_permission_race() -> None:
    class FakeProcPidRusage:
        argtypes: object = None
        restype: object = None

        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            _pid: int,
            _flavor: int,
            buffer: object,
        ) -> int:
            self.calls += 1
            if self.calls == 1:
                ctypes.set_errno(errno.EPERM)
                return -1
            payload = struct.pack("=Q", 123_456_789)
            ctypes.memmove(
                ctypes.addressof(buffer) + 72,
                payload,
                len(payload),
            )
            return 0

    class FakeLibrary:
        proc_pid_rusage = FakeProcPidRusage()

    library = FakeLibrary()
    probe = watchdog.DarwinPhysicalFootprintProbe(library)

    assert probe((7,)) == {7: 123_456_789}
    assert library.proc_pid_rusage.calls == 2


def test_darwin_footprint_probe_accepts_exit_after_permission_race() -> None:
    class FakeProcPidRusage:
        argtypes: object = None
        restype: object = None

        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            _pid: int,
            _flavor: int,
            _buffer: object,
        ) -> int:
            self.calls += 1
            ctypes.set_errno(errno.EPERM if self.calls == 1 else errno.ESRCH)
            return -1

    class FakeLibrary:
        proc_pid_rusage = FakeProcPidRusage()

    library = FakeLibrary()
    probe = watchdog.DarwinPhysicalFootprintProbe(library)

    assert probe((7,)) == {}
    assert library.proc_pid_rusage.calls == 2


def test_darwin_footprint_probe_omits_exited_pid_after_permission_retries() -> None:
    class FakeProcPidRusage:
        argtypes: object = None
        restype: object = None

        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            _pid: int,
            _flavor: int,
            _buffer: object,
        ) -> int:
            self.calls += 1
            ctypes.set_errno(errno.EPERM)
            return -1

    class FakeLibrary:
        proc_pid_rusage = FakeProcPidRusage()

    library = FakeLibrary()
    probe = watchdog.DarwinPhysicalFootprintProbe(
        library,
        pid_exists=lambda pid: pid != 7,
    )

    assert probe((7,)) == {}
    assert library.proc_pid_rusage.calls == 3


def test_darwin_probe_constructor_rejects_missing_symbol() -> None:
    with pytest.raises(watchdog.ProbeError, match="proc_pid_rusage"):
        watchdog.DarwinPhysicalFootprintProbe(object())


def test_platform_probe_rejects_unsupported_hosts() -> None:
    with pytest.raises(watchdog.ProbeError, match="unsupported host"):
        watchdog.process_snapshot("Plan9")


@pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"},
    reason="the CI memory probe supports macOS and Linux",
)
def test_host_probe_observes_current_process() -> None:
    records = watchdog.process_snapshot()

    assert os.getpid() in records
    assert records[os.getpid()].rss_bytes > 0


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="physical-footprint probing uses Darwin libproc",
)
def test_darwin_physical_footprint_probe_observes_current_process() -> None:
    footprints = watchdog.DarwinPhysicalFootprintProbe()((os.getpid(),))

    assert footprints[os.getpid()] > 0


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_exit_code_is_propagated() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(WATCHDOG),
            "--limit-mib",
            "256",
            "--poll-interval",
            "0.02",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(23)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 23
    assert "command finished exit=23" in completed.stderr


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_low_limit_terminates_child_process_tree(tmp_path: Path) -> None:
    marker = tmp_path / "child-terminated"
    pid_path = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(
        """
import os
import signal
import sys
import time
from pathlib import Path

marker = Path(sys.argv[1])

def terminate(_signum, _frame):
    marker.write_text("terminated", encoding="ascii")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, terminate)
payload = bytearray(96 * 1024 * 1024)
marker.with_suffix(".ready").write_text(str(len(payload)), encoding="ascii")
time.sleep(30)
""".lstrip(),
        encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        """
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
Path(sys.argv[3]).write_text(str(child.pid), encoding="ascii")
time.sleep(30)
""".lstrip(),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(WATCHDOG),
            "--limit-mib",
            "64",
            "--poll-interval",
            "0.02",
            "--terminate-grace",
            "1",
            "--",
            sys.executable,
            str(parent),
            str(child),
            str(marker),
            str(pid_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == watchdog.MEMORY_LIMIT_EXIT_CODE
    assert "memory limit exceeded" in completed.stderr
    assert f"reason={watchdog.RSS_LIMIT_REASON}" in completed.stderr or (
        f"reason={watchdog.DARWIN_PHYSICAL_FOOTPRINT_LIMIT_REASON}" in completed.stderr
    )
    assert marker.read_text(encoding="ascii") == "terminated"
    child_pid = int(pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 3
    while _pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _pid_exists(child_pid)


def test_ci_helper_is_sdist_only_for_packaged_profiling_runtime() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = pyproject["tool"]["maturin"]["include"]
    ci_entries = {
        entry["path"]: entry.get("format")
        for entry in includes
        if isinstance(entry, dict)
        and (
            entry.get("path") == "tools/ci"
            or str(entry.get("path", "")).startswith("tools/ci/")
        )
    }

    assert ci_entries == {
        "tools/ci/__init__.py": ["sdist"],
        "tools/ci/memory_watchdog.py": ["sdist"],
    }
    assert not any(
        isinstance(entry, str)
        and (entry == "tools/ci" or entry.startswith("tools/ci/"))
        for entry in includes
    )


# The CI workflow lives in the MadGraph7 repository that contains pyamplicol/.
_MADGRAPH_WORKFLOW = ROOT.parent / ".github/workflows/pyamplicol.yml"


@pytest.mark.skipif(
    not _MADGRAPH_WORKFLOW.is_file(), reason="outside the MadGraph7 repository"
)
def test_tests_workflow_guards_every_heavy_validation_phase() -> None:
    text = _MADGRAPH_WORKFLOW.read_text(encoding="utf-8")
    workflow = text.split("\n  release-runtime:\n", 1)[1].split("\n  # ----", 1)[0]

    assert text.startswith("name: pyAmpliCol\n")
    assert 'python-version: ["3.11", "3.12", "3.13", "3.14"]' in text
    assert "ulimit -v" not in workflow
    assert "psutil" not in workflow
    guarded = "tools/ci/memory_watchdog.py --limit-gib 30 --"
    # The installer stages the native runtime within its guarded invocation.
    # Restricted-mode and ordinary integration checks use separate invocations.
    assert workflow.count(guarded) == 8
    for heavy_command in (
        "dependencies/install_dependencies.py",
        "tests/unit/test_generation_execution_schema.py",
        "tests/integration/test_schema_v3_generation_runtime.py",
        "tests/integration/test_multilanguage_api.py",
        "just rust-check",
        "just rust-test",
    ):
        assert heavy_command in workflow
