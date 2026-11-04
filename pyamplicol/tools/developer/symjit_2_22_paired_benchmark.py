#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run the SymJIT migration benchmark as alternating baseline/candidate pairs.

The command itself must be run behind ``tools/ci/memory_watchdog.py`` with a
30-GiB limit, ``--report-json``, and ``--bind-result-json`` naming this
campaign's ``--result-json``.  It starts one authenticated capture driver at a
time for generation, then admits exactly one profile worker at a time.
Baseline and candidate workers for the same layout/mode/batch/round are
adjacent, and the first role alternates by round.

The candidate revision, both native build-input digests, and both prepared
model digests are externally pinned CLI inputs. They are checked before useful
measurement work and rechecked by the comparison gate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO

BASELINE_REVISION = "172e58fd33a3c65563866c50cfbb5e1ddcd7b302"
LAYOUTS = ("topology-replay", "all-flow-union")
AUTHORITATIVE_MODES = ("compiled", "recurrence")
DIAGNOSTIC_MODES = ("eager",)
BATCH_SIZES = (1, 128, 1024)
ROLES = ("baseline", "candidate")
MINIMUM_SUBPROCESS_SAMPLES = 7
MINIMUM_WARMUPS = 2
MINIMUM_TARGET_RUNTIME_SECONDS = 5.0
MINIMUM_GENERATION_TIMEOUT_SECONDS = 10_800.0
MINIMUM_COORDINATION_TIMEOUT_SECONDS = 43_200.0
CAMPAIGN_KIND = "pyamplicol-symjit-2.22-paired-benchmark-campaign"
CAMPAIGN_SCHEMA = 2
HARNESS_IDENTITY_KIND = "pyamplicol-paired-benchmark-harness-identity"
HARNESS_IDENTITY_SCHEMA = 1
PAIRED_DRIVER_RELATIVE_PATH = "tools/developer/symjit_2_22_paired_benchmark.py"
WATCHDOG_REPORT_KIND = "pyamplicol-memory-watchdog-execution-report"
WATCHDOG_REPORT_SCHEMA = 2
WATCHDOG_LIMIT_BYTES = 30 * 1024**3
WATCHDOG_SCOPE = "complete-orchestrator-process-tree-v1"
WATCHDOG_BINDING = "outer-command-session-result-v1"
READY_KIND = "pyamplicol-paired-profile-ready"
TOKEN_KIND = "pyamplicol-paired-profile-token"
COMPLETION_KIND = "pyamplicol-paired-profile-completion"
COORDINATION_SCHEMA = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


class CampaignError(RuntimeError):
    """Raised when a paired campaign cannot remain authenticated."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise CampaignError("campaign evidence is not canonical JSON") from error


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CampaignError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _validate_addressed_mapping(
    payload: Mapping[str, object],
    *,
    label: str,
) -> None:
    unsigned = dict(payload)
    digest = unsigned.pop("content_sha256", None)
    if (
        not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or digest != _sha256(unsigned)
    ):
        raise CampaignError(f"{label} is not content-addressed")
    _validate_nested_content_addresses(unsigned, label=label)


def _validate_nested_content_addresses(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        if "content_sha256" in value:
            _validate_addressed_mapping(value, label=label)
            return
        for field, child in value.items():
            _validate_nested_content_addresses(
                child,
                label=f"{label}.{field}",
            )
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, child in enumerate(value):
            _validate_nested_content_addresses(
                child,
                label=f"{label}[{index}]",
            )


def _address(payload: Mapping[str, object]) -> dict[str, object]:
    _validate_nested_content_addresses(payload, label="campaign evidence")
    result = dict(payload)
    result["content_sha256"] = _sha256(result)
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(_canonical_json_bytes(payload) + b"\n")
        os.replace(temporary, path)
    except OSError as error:
        raise CampaignError(f"cannot write campaign evidence: {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_addressed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise CampaignError(f"{label} is not a JSON object: {path}")
    try:
        _validate_addressed_mapping(payload, label=label)
    except CampaignError as error:
        raise CampaignError(f"{error}: {path}") from error
    return payload


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _utc(value: object, *, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise CampaignError(f"{label} has no UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as error:
        raise CampaignError(f"{label} has an invalid UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise CampaignError(f"{label} timestamp is not UTC")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


def _revision_argument(value: str) -> str:
    if _REVISION.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "must be a lowercase 40-digit Git revision"
        )
    return value


def _sha256_argument(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "must be a lowercase 64-digit SHA-256 value"
        )
    return value


def _source_revision(path: Path, *, role: str) -> str:
    try:
        root = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"{role} source root does not exist") from error
    if not root.is_dir():
        raise CampaignError(f"{role} source root is not a directory")
    revision = subprocess.run(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    top_level = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = revision.stdout.strip()
    if revision.returncode != 0 or _REVISION.fullmatch(head) is None:
        raise CampaignError(f"{role} source root has no full Git revision")
    repository_root_text = top_level.stdout.strip()
    if top_level.returncode != 0 or not repository_root_text:
        raise CampaignError(f"{role} source root has no Git top-level")
    try:
        repository_root = Path(repository_root_text).resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"{role} source root has no Git top-level") from error
    if repository_root != root:
        raise CampaignError(f"{role} source root is not the exact Git repository root")
    if status.returncode != 0 or status.stdout.strip():
        raise CampaignError(f"{role} source root is not clean")
    if role == "baseline" and head != BASELINE_REVISION:
        raise CampaignError(
            f"baseline source is not immutable revision {BASELINE_REVISION}"
        )
    if role == "candidate" and head == BASELINE_REVISION:
        raise CampaignError("candidate source still identifies the baseline revision")
    return head


def _regular_file(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"{label} does not exist: {path}") from error
    if not resolved.is_file():
        raise CampaignError(f"{label} is not a regular file: {path}")
    return resolved


def _executable_file(path: Path, *, label: str) -> tuple[Path, Path]:
    """Return the lexical executable path and its separately verified target."""

    expanded = path.expanduser()
    lexical = Path(os.path.abspath(expanded))
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"{label} does not exist: {path}") from error
    if not resolved.is_file():
        raise CampaignError(f"{label} is not a regular file: {path}")
    if not os.access(lexical, os.X_OK):
        raise CampaignError(f"{label} is not executable: {path}")
    return lexical, resolved


def _head_bound_file_identity(
    path: Path,
    *,
    source_root: Path,
    label: str,
) -> tuple[Path, dict[str, object]]:
    """Return a tracked, HEAD-identical file and its content address."""

    resolved = _regular_file(path, label=label)
    try:
        relative = resolved.relative_to(source_root)
    except ValueError as error:
        raise CampaignError(
            f"{label} must come from the candidate source snapshot"
        ) from error
    relative_text = relative.as_posix()
    tracked = subprocess.run(
        ("git", "ls-files", "--error-unmatch", "--", relative_text),
        cwd=source_root,
        check=False,
        capture_output=True,
    )
    if tracked.returncode != 0:
        raise CampaignError(f"{label} is not tracked by the candidate repository")
    head_blob = subprocess.run(
        ("git", "cat-file", "blob", f"HEAD:{relative_text}"),
        cwd=source_root,
        check=False,
        capture_output=True,
    )
    if head_blob.returncode != 0:
        raise CampaignError(f"{label} is absent from candidate HEAD")
    try:
        working_bytes = resolved.read_bytes()
    except OSError as error:
        raise CampaignError(f"cannot read {label}: {resolved}") from error
    matches_head = working_bytes == head_blob.stdout
    if not matches_head:
        raise CampaignError(f"{label} is not byte-identical to candidate HEAD")
    identity = _address(
        {
            "kind": HARNESS_IDENTITY_KIND,
            "schema_version": HARNESS_IDENTITY_SCHEMA,
            "candidate_relative_path": relative_text,
            "head_blob_sha256": hashlib.sha256(head_blob.stdout).hexdigest(),
            "working_file_sha256": hashlib.sha256(working_bytes).hexdigest(),
            "head_blob_equals_working_file": matches_head,
        }
    )
    return resolved, identity


def _head_bound_file(path: Path, *, source_root: Path, label: str) -> Path:
    """Require a regular file to be tracked and byte-identical to ``HEAD``."""

    resolved, _identity = _head_bound_file_identity(
        path,
        source_root=source_root,
        label=label,
    )
    return resolved


def _verify_campaign_inputs_unchanged(
    *,
    source_roots: Mapping[str, Path],
    revisions: Mapping[str, str],
    pythons: Mapping[str, Path],
    python_targets: Mapping[str, Path],
    python_hashes: Mapping[str, str],
    prepared_models: Mapping[str, Path],
    prepared_model_hashes: Mapping[str, str],
    harness: Path,
    harness_identity: Mapping[str, object],
    orchestrator: Path,
    orchestrator_identity: Mapping[str, object],
) -> None:
    for role in ROLES:
        if _source_revision(source_roots[role], role=role) != revisions[role]:
            raise CampaignError(f"{role} source revision changed during campaign")
        current_python, current_target = _executable_file(
            pythons[role],
            label=f"{role} Python",
        )
        if (
            current_python != pythons[role]
            or current_target != python_targets[role]
            or _sha256_file(current_target) != python_hashes[role]
        ):
            raise CampaignError(f"{role} Python changed during campaign")
        current_prepared_model = _regular_file(
            prepared_models[role],
            label=f"{role} prepared model",
        )
        if (
            current_prepared_model != prepared_models[role]
            or _sha256_file(current_prepared_model) != prepared_model_hashes[role]
        ):
            raise CampaignError(f"{role} prepared model changed during campaign")
    current_harness, current_harness_identity = _head_bound_file_identity(
        harness,
        source_root=source_roots["candidate"],
        label="benchmark harness",
    )
    if current_harness != harness or current_harness_identity != dict(harness_identity):
        raise CampaignError("benchmark harness changed during campaign")
    current_orchestrator, current_orchestrator_identity = _head_bound_file_identity(
        orchestrator,
        source_root=source_roots["candidate"],
        label="paired benchmark orchestrator",
    )
    if (
        current_orchestrator != orchestrator
        or current_orchestrator_identity != dict(orchestrator_identity)
        or _sha256_file(Path(__file__).resolve())
        != current_orchestrator_identity["working_file_sha256"]
    ):
        raise CampaignError("paired benchmark orchestrator changed during campaign")


def _wait_for_record(
    path: Path,
    *,
    label: str,
    timeout: float,
    process: subprocess.Popen[bytes] | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if process is not None and process.poll() is not None:
            raise CampaignError(
                f"{label} producer exited with status {process.returncode}"
            )
        if time.monotonic() >= deadline:
            raise CampaignError(f"timed out waiting for {label}: {path}")
        time.sleep(0.1)
    return _read_addressed(path, label=label)


def _role_directory(
    coordination_root: Path,
    *,
    session_id: str,
    layout: str,
    role: str,
) -> Path:
    return coordination_root / session_id / layout / role


def _capture_command(
    arguments: argparse.Namespace,
    *,
    role: str,
    layout: str,
    source_root: Path,
    python: Path,
    prepared_model: Path,
    result_json: Path,
    output_root: Path,
    coordination_root: Path,
) -> tuple[str, ...]:
    modes = (
        (*AUTHORITATIVE_MODES, *DIAGNOSTIC_MODES)
        if arguments.include_eager
        else AUTHORITATIVE_MODES
    )
    selector_arguments = (
        (
            "--color-flow",
            arguments.topology_color_flow,
            "--helicity",
            arguments.topology_helicity,
        )
        if layout == "topology-replay"
        else (
            "--color-flow",
            arguments.union_color_flow,
            "--helicity",
            arguments.union_helicity,
        )
    )
    expected_runtime_arguments = (
        "--expected-source-revision",
        (
            BASELINE_REVISION
            if role == "baseline"
            else arguments.expected_candidate_source_revision
        ),
        "--expected-native-build-inputs-sha256",
        getattr(
            arguments,
            f"expected_{role}_native_build_inputs_sha256",
        ),
    )
    return (
        str(python),
        str(arguments.harness),
        "--output-root",
        str(output_root),
        "--result-json",
        str(result_json),
        "--prepared-model",
        str(prepared_model),
        *expected_runtime_arguments,
        "--process-expression",
        arguments.process_expression,
        "--jit-optimization-level",
        "3",
        "--lc-flow-layout",
        layout,
        *sum((("--mode", mode) for mode in modes), ()),
        "--batch-size",
        "1",
        "--batch-size",
        "128",
        "--batch-size",
        "1024",
        "--target-runtime",
        str(arguments.target_runtime),
        "--minimum-samples",
        str(arguments.minimum_samples),
        "--subprocess-samples",
        str(arguments.subprocess_samples),
        "--warmup-runs",
        str(arguments.warmup_runs),
        "--validation-samples",
        str(arguments.validation_samples),
        "--point-tile-size",
        str(arguments.point_tile_size),
        "--generation-timeout",
        str(arguments.generation_timeout),
        "--profile-timeout",
        str(arguments.profile_timeout),
        "--paired-profile-coordination-dir",
        str(coordination_root),
        "--paired-profile-role",
        role,
        "--paired-profile-session-id",
        arguments.session_id,
        "--paired-profile-wait-timeout",
        str(arguments.coordination_timeout),
        *selector_arguments,
    )


def _role_environment(
    output_root: Path,
    *,
    source_root: Path,
    role: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    cache_root = output_root / "workspace-local-environment" / role
    paths = {
        "TMPDIR": cache_root / "tmp",
        "PIP_CACHE_DIR": cache_root / "pip-cache",
        "XDG_CACHE_HOME": cache_root / "xdg-cache",
        "PYTHONPYCACHEPREFIX": cache_root / "pycache",
        "CARGO_HOME": cache_root / "cargo-home",
        "CARGO_TARGET_DIR": cache_root / "cargo-target",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    environment.update({key: str(path) for key, path in paths.items()})
    environment["PYAMPLICOL_BENCHMARK_SOURCE_ROOT"] = str(source_root)
    return environment


def _start_capture(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    log: TextIO,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        tuple(command),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(environment),
        start_new_session=True,
    )


def _signal_capture_group(
    process: subprocess.Popen[bytes],
    signal_number: int,
) -> None:
    """Signal the isolated capture group, falling back to its leader."""

    killpg = getattr(os, "killpg", None)
    if killpg is not None:
        try:
            killpg(process.pid, signal_number)
            return
        except (OSError, ValueError):
            pass
    if process.poll() is None:
        with suppress(OSError, ValueError):
            process.send_signal(signal_number)


def _capture_group_exists(process: subprocess.Popen[bytes]) -> bool:
    killpg = getattr(os, "killpg", None)
    if killpg is not None:
        try:
            killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except (OSError, ValueError):
            return process.poll() is None
        return True
    return process.poll() is None


def _terminate(
    processes: Mapping[str, subprocess.Popen[bytes]],
    *,
    grace_seconds: float = 10.0,
) -> None:
    captures = tuple(processes.values())
    for process in captures:
        _signal_capture_group(process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while (
        any(_capture_group_exists(process) for process in captures)
        and time.monotonic() < deadline
    ):
        for process in captures:
            process.poll()
        time.sleep(0.05)
    for process in captures:
        if _capture_group_exists(process):
            _signal_capture_group(process, signal.SIGKILL)
    for process in captures:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            with suppress(OSError):
                process.kill()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1.0)


def _validate_ready_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    layout: str,
    session_id: str,
    baseline_revision: str,
    candidate_revision: str,
    modes: Sequence[str],
    subprocess_samples: int,
) -> list[dict[str, object]]:
    for role, ready, revision in (
        ("baseline", baseline, baseline_revision),
        ("candidate", candidate, candidate_revision),
    ):
        expected = {
            "kind": READY_KIND,
            "schema_version": COORDINATION_SCHEMA,
            "session_id": session_id,
            "role": role,
            "layout": layout,
            "source_revision": revision,
        }
        if any(ready.get(field) != value for field, value in expected.items()):
            raise CampaignError(f"{layout} {role} ready record has drifted")
        _utc(ready.get("ready_at_utc"), label=f"{layout} {role} ready record")
    baseline_plan = baseline.get("profile_schedule_plan")
    candidate_plan = candidate.get("profile_schedule_plan")
    if (
        not isinstance(baseline_plan, list)
        or baseline_plan != candidate_plan
        or baseline.get("profile_schedule_plan_sha256") != _sha256(baseline_plan)
        or candidate.get("profile_schedule_plan_sha256") != _sha256(candidate_plan)
    ):
        raise CampaignError(f"{layout} role profile plans are not identical")
    expected_cells = {
        (mode, batch_size, round_index)
        for mode in modes
        for batch_size in BATCH_SIZES
        for round_index in range(subprocess_samples)
    }
    observed_cells = {
        (entry.get("mode"), entry.get("batch_size"), entry.get("round"))
        for entry in baseline_plan
        if isinstance(entry, Mapping)
    }
    if len(baseline_plan) != len(expected_cells) or observed_cells != expected_cells:
        raise CampaignError(f"{layout} profile plan is incomplete")
    if any(
        entry.get("schedule_index") != index
        for index, entry in enumerate(baseline_plan)
        if isinstance(entry, Mapping)
    ):
        raise CampaignError(f"{layout} profile plan indices are not canonical")
    return [dict(entry) for entry in baseline_plan if isinstance(entry, Mapping)]


def _issue_token(
    coordination_root: Path,
    *,
    session_id: str,
    layout: str,
    role: str,
    entry: Mapping[str, object],
    pair_index: int,
    order_in_pair: int,
) -> dict[str, object]:
    token = _address(
        {
            "kind": TOKEN_KIND,
            "schema_version": COORDINATION_SCHEMA,
            "session_id": session_id,
            "role": role,
            "layout": layout,
            **{
                field: entry.get(field)
                for field in ("schedule_index", "round", "mode", "batch_size")
            },
            "pair_index": pair_index,
            "order_in_pair": order_in_pair,
            "issued_at_utc": _utc_now(),
        }
    )
    schedule_index = entry["schedule_index"]
    assert isinstance(schedule_index, int)
    path = (
        _role_directory(
            coordination_root,
            session_id=session_id,
            layout=layout,
            role=role,
        )
        / f"token-{schedule_index:06d}.json"
    )
    if path.exists():
        raise CampaignError(f"paired scheduler token already exists: {path}")
    _write_json_atomic(path, token)
    return token


def _validate_completion(
    completion: Mapping[str, Any],
    *,
    token: Mapping[str, object],
    role: str,
    layout: str,
    entry: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "kind": COMPLETION_KIND,
        "schema_version": COORDINATION_SCHEMA,
        "session_id": token.get("session_id"),
        "role": role,
        "layout": layout,
        "token_sha256": token.get("content_sha256"),
        **{
            field: entry.get(field)
            for field in ("schedule_index", "round", "mode", "batch_size")
        },
    }
    if any(completion.get(field) != value for field, value in expected.items()):
        raise CampaignError(f"{layout} {role} completion does not match its token")
    started = _utc(
        completion.get("worker_started_at_utc"),
        label=f"{layout} {role} worker start",
    )
    finished = _utc(
        completion.get("worker_finished_at_utc"),
        label=f"{layout} {role} worker finish",
    )
    recorded = _utc(
        completion.get("recorded_at_utc"),
        label=f"{layout} {role} completion",
    )
    issued = _utc(token.get("issued_at_utc"), label=f"{layout} {role} token")
    if not issued <= started <= finished <= recorded:
        raise CampaignError(f"{layout} {role} paired timestamps are inverted")
    for field in ("worker_invocation_sha256", "worker_result_record_sha256"):
        value = completion.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise CampaignError(f"{layout} {role} completion lacks {field}")
    return dict(completion)


def _capture_identity(path: Path) -> dict[str, object]:
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError as error:
        raise CampaignError(f"capture result does not exist: {path}") from error
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(resolved),
    }


def _run_layout(
    arguments: argparse.Namespace,
    *,
    layout: str,
    revisions: Mapping[str, str],
    source_roots: Mapping[str, Path],
    pythons: Mapping[str, Path],
    prepared_models: Mapping[str, Path],
    output_root: Path,
    coordination_root: Path,
) -> dict[str, object]:
    processes: dict[str, subprocess.Popen[bytes]] = {}
    logs: dict[str, TextIO] = {}
    result_paths = {role: output_root / role / f"{layout}.json" for role in ROLES}
    capture_roots = {role: output_root / role / layout for role in ROLES}
    try:
        for role in ROLES:
            log_path = output_root / "logs" / f"{layout}-{role}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            logs[role] = log
            command = _capture_command(
                arguments,
                role=role,
                layout=layout,
                source_root=source_roots[role],
                python=pythons[role],
                prepared_model=prepared_models[role],
                result_json=result_paths[role],
                output_root=capture_roots[role],
                coordination_root=coordination_root,
            )
            processes[role] = _start_capture(
                command,
                environment=_role_environment(
                    output_root,
                    source_root=source_roots[role],
                    role=role,
                ),
                log=log,
            )
            ready_path = (
                _role_directory(
                    coordination_root,
                    session_id=arguments.session_id,
                    layout=layout,
                    role=role,
                )
                / "ready.json"
            )
            _wait_for_record(
                ready_path,
                label=f"{layout} {role} ready record",
                timeout=arguments.coordination_timeout,
                process=processes[role],
            )

        ready_records = {
            role: _read_addressed(
                _role_directory(
                    coordination_root,
                    session_id=arguments.session_id,
                    layout=layout,
                    role=role,
                )
                / "ready.json",
                label=f"{layout} {role} ready record",
            )
            for role in ROLES
        }
        modes = (
            (*AUTHORITATIVE_MODES, *DIAGNOSTIC_MODES)
            if arguments.include_eager
            else AUTHORITATIVE_MODES
        )
        plan = _validate_ready_pair(
            ready_records["baseline"],
            ready_records["candidate"],
            layout=layout,
            session_id=arguments.session_id,
            baseline_revision=revisions["baseline"],
            candidate_revision=revisions["candidate"],
            modes=modes,
            subprocess_samples=arguments.subprocess_samples,
        )
        pair_records: list[dict[str, object]] = []
        previous_pair_finished: dt.datetime | None = None
        for pair_index, entry in enumerate(plan):
            round_index = entry.get("round")
            assert isinstance(round_index, int)
            role_order = (
                ("baseline", "candidate")
                if round_index % 2 == 0
                else ("candidate", "baseline")
            )
            completions: dict[str, dict[str, object]] = {}
            for order_in_pair, role in enumerate(role_order):
                token = _issue_token(
                    coordination_root,
                    session_id=arguments.session_id,
                    layout=layout,
                    role=role,
                    entry=entry,
                    pair_index=pair_index,
                    order_in_pair=order_in_pair,
                )
                schedule_index = entry["schedule_index"]
                assert isinstance(schedule_index, int)
                completion_path = (
                    _role_directory(
                        coordination_root,
                        session_id=arguments.session_id,
                        layout=layout,
                        role=role,
                    )
                    / f"completion-{schedule_index:06d}.json"
                )
                completion = _wait_for_record(
                    completion_path,
                    label=f"{layout} {role} completion",
                    timeout=arguments.coordination_timeout,
                    process=processes[role],
                )
                completions[role] = _validate_completion(
                    completion,
                    token=token,
                    role=role,
                    layout=layout,
                    entry=entry,
                )
            first = completions[role_order[0]]
            second = completions[role_order[1]]
            first_finished = _utc(
                first["worker_finished_at_utc"],
                label=f"{layout} first paired completion",
            )
            second_started = _utc(
                second["worker_started_at_utc"],
                label=f"{layout} second paired completion",
            )
            if first_finished > second_started:
                raise CampaignError(f"{layout} paired profile workers overlapped")
            pair_started = _utc(
                first["worker_started_at_utc"],
                label=f"{layout} pair start",
            )
            pair_finished = _utc(
                second["worker_finished_at_utc"],
                label=f"{layout} pair finish",
            )
            if (
                previous_pair_finished is not None
                and previous_pair_finished > pair_started
            ):
                raise CampaignError(f"{layout} profile pairs overlap or reorder")
            previous_pair_finished = pair_finished
            pair_records.append(
                _address(
                    {
                        "kind": "pyamplicol-paired-profile-pair",
                        "schema_version": COORDINATION_SCHEMA,
                        "pair_index": pair_index,
                        **{
                            field: entry.get(field)
                            for field in (
                                "schedule_index",
                                "round",
                                "mode",
                                "batch_size",
                            )
                        },
                        "role_order": list(role_order),
                        "completions": completions,
                    }
                )
            )

        for role, process in processes.items():
            try:
                return_code = process.wait(timeout=arguments.coordination_timeout)
            except subprocess.TimeoutExpired as error:
                raise CampaignError(f"{layout} {role} driver did not finish") from error
            if return_code != 0:
                raise CampaignError(
                    f"{layout} {role} driver exited with status {return_code}"
                )
        return {
            "layout": layout,
            "ready_records": ready_records,
            "paired_schedule_algorithm": (
                "slot-adjacent-round-alternating-role-order-v1"
            ),
            "pairs": pair_records,
            "captures": {role: _capture_identity(result_paths[role]) for role in ROLES},
        }
    except BaseException:
        _terminate(processes)
        raise
    finally:
        for log in logs.values():
            log.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline-python", type=Path, required=True)
    result.add_argument("--baseline-source-root", type=Path, required=True)
    result.add_argument("--baseline-prepared-model", type=Path, required=True)
    result.add_argument("--candidate-python", type=Path, required=True)
    result.add_argument("--candidate-source-root", type=Path, required=True)
    result.add_argument("--candidate-prepared-model", type=Path, required=True)
    result.add_argument(
        "--expected-candidate-source-revision",
        type=_revision_argument,
        required=True,
    )
    result.add_argument(
        "--expected-baseline-native-build-inputs-sha256",
        type=_sha256_argument,
        required=True,
    )
    result.add_argument(
        "--expected-candidate-native-build-inputs-sha256",
        type=_sha256_argument,
        required=True,
    )
    result.add_argument(
        "--expected-baseline-prepared-model-sha256",
        type=_sha256_argument,
        required=True,
    )
    result.add_argument(
        "--expected-candidate-prepared-model-sha256",
        type=_sha256_argument,
        required=True,
    )
    result.add_argument(
        "--harness",
        type=Path,
        default=Path(__file__).with_name("recurrence_z6g_benchmark.py"),
    )
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--result-json", type=Path, required=True)
    result.add_argument("--session-id", required=True)
    eager = result.add_mutually_exclusive_group()
    eager.add_argument(
        "--include-eager",
        dest="include_eager",
        action="store_true",
        default=True,
        help="include the gate-required eager lane (default)",
    )
    eager.add_argument(
        "--no-eager",
        dest="include_eager",
        action="store_false",
        help="omit eager only for focused harness tests or diagnostics",
    )
    result.add_argument("--process-expression", default="d d~ > z + 6*g")
    result.add_argument("--target-runtime", type=_positive_float, default=5.0)
    result.add_argument(
        "--minimum-samples",
        type=_positive_int,
        default=MINIMUM_SUBPROCESS_SAMPLES,
    )
    result.add_argument(
        "--subprocess-samples",
        type=_positive_int,
        default=MINIMUM_SUBPROCESS_SAMPLES,
    )
    result.add_argument("--warmup-runs", type=int, default=MINIMUM_WARMUPS)
    result.add_argument("--validation-samples", type=_positive_int, default=3)
    result.add_argument("--point-tile-size", type=_positive_int, default=1024)
    result.add_argument(
        "--generation-timeout",
        type=_positive_float,
        default=MINIMUM_GENERATION_TIMEOUT_SECONDS,
    )
    result.add_argument("--profile-timeout", type=_positive_float, default=900.0)
    result.add_argument(
        "--coordination-timeout",
        type=_positive_float,
        default=MINIMUM_COORDINATION_TIMEOUT_SECONDS,
    )
    result.add_argument(
        "--topology-color-flow",
        default="flow:2,4,5,6,7,8,9,1",
    )
    result.add_argument("--topology-helicity", default="1")
    result.add_argument("--union-color-flow", default="1")
    result.add_argument(
        "--union-helicity",
        default="h:-1,+1,-1,+1,-1,+1,-1,+1,-1",
    )
    return result


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if (
        arguments.subprocess_samples != MINIMUM_SUBPROCESS_SAMPLES
        or arguments.minimum_samples < MINIMUM_SUBPROCESS_SAMPLES
        or arguments.warmup_runs != MINIMUM_WARMUPS
        or arguments.target_runtime < MINIMUM_TARGET_RUNTIME_SECONDS
    ):
        raise CampaignError(
            "paired campaign requires exactly seven independent subprocess "
            "pairs, at least seven internal 5-second samples, and exactly two "
            "warmups per subprocess"
        )
    if (
        arguments.generation_timeout < MINIMUM_GENERATION_TIMEOUT_SECONDS
        or arguments.coordination_timeout < MINIMUM_COORDINATION_TIMEOUT_SECONDS
    ):
        raise CampaignError(
            "paired Z+6g campaign requires at least "
            f"{MINIMUM_GENERATION_TIMEOUT_SECONDS:g} seconds for generation "
            f"and {MINIMUM_COORDINATION_TIMEOUT_SECONDS:g} seconds for "
            "coordination"
        )
    expected_candidate_revision = getattr(
        arguments,
        "expected_candidate_source_revision",
        None,
    )
    if (
        not isinstance(expected_candidate_revision, str)
        or _REVISION.fullmatch(expected_candidate_revision) is None
    ):
        raise CampaignError(
            "paired campaign requires an externally pinned candidate revision"
        )
    expected_sha256 = {
        (role, identity): getattr(
            arguments,
            f"expected_{role}_{identity}_sha256",
            None,
        )
        for role in ROLES
        for identity in (
            "native_build_inputs",
            "prepared_model",
        )
    }
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in expected_sha256.values()
    ):
        raise CampaignError(
            "paired campaign requires externally pinned native-build and "
            "prepared-model SHA-256 values for both roles"
        )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", arguments.session_id) is None:
        raise CampaignError("session ID is invalid")
    normalized_process = " ".join(arguments.process_expression.split()).casefold()
    if normalized_process not in {
        "d d~ > z + 6*g",
        "d d~ > z g g g g g g",
    }:
        raise CampaignError("paired migration campaign requires exact d d~ > Z+6g")
    output_root = arguments.output_root.expanduser().resolve()
    if output_root.exists():
        raise CampaignError("paired campaign output root already exists")
    result_json = arguments.result_json.expanduser().resolve()
    if result_json.exists():
        raise CampaignError("paired campaign result JSON already exists")
    source_roots = {
        role: getattr(arguments, f"{role}_source_root")
        .expanduser()
        .resolve(strict=True)
        for role in ROLES
    }
    if source_roots["baseline"] == source_roots["candidate"]:
        raise CampaignError("baseline and candidate source roots must be distinct")
    revisions = {
        role: _source_revision(source_roots[role], role=role) for role in ROLES
    }
    if revisions["candidate"] != expected_candidate_revision:
        raise CampaignError(
            "candidate source does not match the externally pinned revision"
        )
    orchestrator, orchestrator_identity = _head_bound_file_identity(
        source_roots["candidate"] / PAIRED_DRIVER_RELATIVE_PATH,
        source_root=source_roots["candidate"],
        label="paired benchmark orchestrator",
    )
    if (
        _sha256_file(Path(__file__).resolve())
        != orchestrator_identity["working_file_sha256"]
    ):
        raise CampaignError(
            "running paired benchmark orchestrator is not candidate HEAD"
        )
    arguments.harness, harness_identity = _head_bound_file_identity(
        arguments.harness,
        source_root=source_roots["candidate"],
        label="benchmark harness",
    )
    python_identities = {
        role: _executable_file(
            getattr(arguments, f"{role}_python"),
            label=f"{role} Python",
        )
        for role in ROLES
    }
    pythons = {role: identity[0] for role, identity in python_identities.items()}
    python_targets = {role: identity[1] for role, identity in python_identities.items()}
    python_hashes = {role: _sha256_file(python_targets[role]) for role in ROLES}
    if python_hashes["baseline"] != python_hashes["candidate"]:
        raise CampaignError(
            "baseline and candidate must use the same Python interpreter binary"
        )
    prepared_models = {
        role: _regular_file(
            getattr(arguments, f"{role}_prepared_model"),
            label=f"{role} prepared model",
        )
        for role in ROLES
    }
    prepared_model_hashes = {
        role: _sha256_file(prepared_models[role]) for role in ROLES
    }
    for role in ROLES:
        expected = expected_sha256[(role, "prepared_model")]
        if prepared_model_hashes[role] != expected:
            raise CampaignError(
                f"{role} prepared model does not match the externally pinned "
                "SHA-256 value"
            )
    output_root.mkdir(parents=True)
    coordination_root = output_root / "coordination"
    started_at = _utc_now()
    layouts: dict[str, object] = {}
    for layout in LAYOUTS:
        layouts[layout] = _run_layout(
            arguments,
            layout=layout,
            revisions=revisions,
            source_roots=source_roots,
            pythons=pythons,
            prepared_models=prepared_models,
            output_root=output_root,
            coordination_root=coordination_root,
        )
    _verify_campaign_inputs_unchanged(
        source_roots=source_roots,
        revisions=revisions,
        pythons=pythons,
        python_targets=python_targets,
        python_hashes=python_hashes,
        prepared_models=prepared_models,
        prepared_model_hashes=prepared_model_hashes,
        harness=arguments.harness,
        harness_identity=harness_identity,
        orchestrator=orchestrator,
        orchestrator_identity=orchestrator_identity,
    )
    payload = {
        "kind": CAMPAIGN_KIND,
        "schema_version": CAMPAIGN_SCHEMA,
        "complete": True,
        "session_id": arguments.session_id,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "process": "d d~ > Z g g g g g g",
        "orchestrator": orchestrator_identity,
        "harness": harness_identity,
        "roles": {
            role: {
                "source_root": str(source_roots[role]),
                "source_revision": revisions[role],
                "python": str(pythons[role]),
                "python_resolved_target": str(python_targets[role]),
                "python_sha256": python_hashes[role],
                "prepared_model": str(prepared_models[role]),
                "prepared_model_sha256": prepared_model_hashes[role],
            }
            for role in ROLES
        },
        "configuration": {
            "authoritative_modes": list(AUTHORITATIVE_MODES),
            "diagnostic_modes": (
                list(DIAGNOSTIC_MODES) if arguments.include_eager else []
            ),
            "batch_sizes": list(BATCH_SIZES),
            "target_runtime_seconds": arguments.target_runtime,
            "minimum_samples": arguments.minimum_samples,
            "subprocess_samples": arguments.subprocess_samples,
            "warmup_runs": arguments.warmup_runs,
            "watchdog": {
                "required": True,
                "report_kind": WATCHDOG_REPORT_KIND,
                "report_schema_version": WATCHDOG_REPORT_SCHEMA,
                "limit_bytes": WATCHDOG_LIMIT_BYTES,
                "scope": WATCHDOG_SCOPE,
                "binding": WATCHDOG_BINDING,
            },
        },
        "layouts": layouts,
    }
    addressed = _address(payload)
    _write_json_atomic(result_json, addressed)
    return addressed


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parser().parse_args(argv))
    except (CampaignError, OSError, ValueError) as error:
        print(f"paired benchmark campaign: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
