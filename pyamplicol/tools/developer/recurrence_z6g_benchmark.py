#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Benchmark compiled, eager, and recurrence execution for qq_Zng.

This is a developer harness, independent of the publication-report tooling.
Run the whole command behind
``tools/ci/memory_watchdog.py`` when generating the large artifacts.
Generation and profiling run in isolated worker processes so each phase has a
meaningful process-level ``resource.getrusage`` peak-RSS record.

One invocation captures one LC layout. Its result embeds a fail-closed
milestone-0 evidence manifest, but cannot certify milestone 0 by itself: a
separate orchestrator must combine content-hashed topology-replay and
all-flow-union captures with pinned legacy AmpliCol evidence.
Authoritative lane timing is the median and raw MAD of seven independently
warmed, identity-verified subprocess measurements per mode/batch cell.

For a developer-only comparison, set
``PYAMPLICOL_RECURRENCE_Z6G_SOURCE_CHECKOUT`` to an absolute clean worktree
inside this driver's workspace. The frozen driver remains the executable
script while source and installed-runtime provenance bind to that worktree.

For an immutable baseline recapture, the harness itself may live outside the
source tree under test. Set ``PYAMPLICOL_BENCHMARK_SOURCE_ROOT`` to the
absolute path of that clean checkout; command provenance still authenticates
the external harness while source/build provenance is read from the selected
checkout.

Example::

    .venv/bin/python tools/ci/memory_watchdog.py --limit-gib 30 -- \
      .venv/bin/python tools/developer/recurrence_z6g_benchmark.py
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast


class HarnessError(RuntimeError):
    """Raised when the benchmark contract cannot be completed."""


SOURCE_CHECKOUT_OVERRIDE_ENV = "PYAMPLICOL_RECURRENCE_Z6G_SOURCE_CHECKOUT"
BENCHMARK_SOURCE_ROOT_ENV = "PYAMPLICOL_BENCHMARK_SOURCE_ROOT"
DRIVER_PATH = Path(__file__).resolve()
DRIVER_ROOT = DRIVER_PATH.parents[2]
_SOURCE_CHECKOUT_FILE_MARKERS = (
    Path("Cargo.toml"),
    Path("pyproject.toml"),
    Path("src/pyamplicol/__init__.py"),
    Path("tools/developer/recurrence_z6g_benchmark.py"),
)


def _resolve_source_checkout_root(
    *,
    driver_root: Path = DRIVER_ROOT,
    environment: Mapping[str, str] = os.environ,
) -> Path:
    """Resolve the optional measured-checkout override, failing closed."""

    workspace_root = driver_root.resolve(strict=True)
    raw_override = environment.get(SOURCE_CHECKOUT_OVERRIDE_ENV)
    if raw_override is None:
        return workspace_root
    if not raw_override:
        raise HarnessError(
            f"{SOURCE_CHECKOUT_OVERRIDE_ENV} must name an absolute source checkout"
        )
    override = Path(raw_override)
    if not override.is_absolute():
        raise HarnessError(f"{SOURCE_CHECKOUT_OVERRIDE_ENV} must be an absolute path")
    try:
        checkout = override.resolve(strict=True)
    except OSError as error:
        raise HarnessError(
            f"{SOURCE_CHECKOUT_OVERRIDE_ENV} source checkout is unavailable"
        ) from error
    if not checkout.is_dir():
        raise HarnessError(f"{SOURCE_CHECKOUT_OVERRIDE_ENV} does not name a directory")
    if checkout != workspace_root and workspace_root not in checkout.parents:
        raise HarnessError(
            f"{SOURCE_CHECKOUT_OVERRIDE_ENV} must stay inside the driver workspace"
        )
    git_marker = checkout / ".git"
    missing = [
        marker.as_posix()
        for marker in _SOURCE_CHECKOUT_FILE_MARKERS
        if not (checkout / marker).is_file()
    ]
    if not git_marker.exists() or missing:
        if not git_marker.exists():
            missing.insert(0, ".git")
        raise HarnessError(
            f"{SOURCE_CHECKOUT_OVERRIDE_ENV} is not a pyAmpliCol source checkout; "
            f"missing {', '.join(missing)}"
        )
    return checkout


def _resolve_benchmark_source_root(
    requested: str | None,
    *,
    script_path: Path = Path(__file__),
) -> Path:
    default = script_path.resolve().parents[2]
    if requested is None or not requested.strip():
        return default
    candidate = Path(requested)
    if not candidate.is_absolute():
        raise RuntimeError("PYAMPLICOL_BENCHMARK_SOURCE_ROOT must be an absolute path")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            "PYAMPLICOL_BENCHMARK_SOURCE_ROOT does not resolve to a source checkout"
        ) from error
    required = ("Cargo.toml", "pyproject.toml")
    if not resolved.is_dir() or any(
        not (resolved / relative).is_file() for relative in required
    ):
        raise RuntimeError(
            "PYAMPLICOL_BENCHMARK_SOURCE_ROOT is not a pyAmpliCol source checkout"
        )
    return resolved


ROOT = _resolve_source_checkout_root(
    driver_root=_resolve_benchmark_source_root(
        os.environ.get(BENCHMARK_SOURCE_ROOT_ENV)
    )
)
PREPARED_MODEL_ID = "built-in-sm-jit-o2"
PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL = 2
DEFAULT_BATCH_SIZES = (1, 128, 1024)
MIN_AUTHORITATIVE_SAMPLES = 7
AUTHORITATIVE_EXECUTION_MODES = ("compiled", "recurrence")
DIAGNOSTIC_EXECUTION_MODES = ("eager",)
EXECUTION_MODES = ("compiled", "eager", "recurrence")
LC_FLOW_LAYOUTS = ("topology-replay", "all-flow-union")
VALIDATION_SEED = 12345
RTOL = 1.0e-12
ATOL = 1.0e-15
RESULT_KIND = "pyamplicol-recurrence-z6g-benchmark"
RESULT_SCHEMA = 7
RAW_NATIVE_WALL_BLOCK_KIND = "pyamplicol-raw-native-wall-blocks"
RAW_NATIVE_WALL_BLOCK_SCHEMA = 2
RAW_NATIVE_WALL_MEASUREMENT_CONTRACT = (
    "warmed-native-wall-minimum-duration-v1"
)
ARTIFACT_SEMANTIC_IDENTITY_SCHEMA = 3
LOGICAL_REDUCTION_ORDER_ABI = "helicity-major-color-minor-v1"
REUSE_SIGNATURE_KIND = "pyamplicol-benchmark-artifact-reuse-signature"
REUSE_SIGNATURE_SCHEMA = 3
PROFILE_SCHEDULE_KIND = "pyamplicol-interleaved-subprocess-profile-schedule"
PROFILE_SCHEDULE_SCHEMA = 2
RECURRENCE_RUNTIME_ACCEPTANCE_KIND = "pyamplicol-recurrence-runtime-acceptance"
RECURRENCE_RUNTIME_ACCEPTANCE_SCHEMA = 1
PAIRED_PROFILE_READY_KIND = "pyamplicol-paired-profile-ready"
PAIRED_PROFILE_TOKEN_KIND = "pyamplicol-paired-profile-token"
PAIRED_PROFILE_COMPLETION_KIND = "pyamplicol-paired-profile-completion"
PAIRED_PROFILE_COORDINATION_KIND = "pyamplicol-paired-profile-coordination"
PAIRED_PROFILE_COORDINATION_SCHEMA = 1
WORKER_VERIFICATION_KIND = "pyamplicol-profile-worker-pre-timing-verification"
WORKER_VERIFICATION_SCHEMA = 1
RETAINED_WORKER_RESULT_KIND = "pyamplicol-retained-profile-worker-result"
RETAINED_WORKER_RESULT_SCHEMA = 1
PRESERVED_WORKER_RESULT_KIND = "pyamplicol-preserved-worker-result-evidence"
PRESERVED_WORKER_RESULT_SCHEMA = 1
CAPTURE_ACCEPTANCE_SCHEMA = 6
M0_ACCEPTANCE_KIND = "pyamplicol-milestone-0-evidence-manifest"
M0_ACCEPTANCE_SCHEMA = 4
_WORKER_MARKER = "PYAMPLICOL_RECURRENCE_Z6G_WORKER_RESULT="
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_RESOURCE_PEAK_SEMANTICS = (
    "self high-water mark and maximum completed-child high-water mark; "
    "not an aggregate process-tree sample"
)
_RESOURCE_PEAK_KEYS = frozenset(
    {
        "source",
        "self_peak_bytes",
        "maximum_child_peak_bytes",
        "observed_lower_bound_bytes",
        "semantics",
    }
)
MIGRATION_PROCESS_REQUEST = "d d~ > z + 6*g"
MIGRATION_PROCESS_CANONICAL = "d d~ > Z g g g g g g"
MIGRATION_PROCESS_NAME = "ddbar_Z_6g"
MIGRATION_UNION_SELECTED_HELICITY_ID = "h:-1,+1,-1,+1,-1,+1,-1,+1,-1"
MIGRATION_UNION_SELECTED_HELICITY_VALUES = (-1, 1, -1, 1, -1, 1, -1, 1, -1)
MIGRATION_UNION_PHYSICAL_COLOR_COUNT = 720


def _within_numerical_tolerance(
    actual: complex | float,
    expected: complex | float,
) -> bool:
    return abs(actual - expected) <= ATOL + RTOL * abs(expected)


def _process(gluon_count: int) -> str:
    return "u u~ > Z" + " g" * gluon_count


def _process_name(gluon_count: int) -> str:
    return f"uubar_Z_{gluon_count}g"


def _canonical_process_expression(expression: str) -> str:
    normalized = " ".join(expression.split())
    if normalized.casefold() == MIGRATION_PROCESS_REQUEST.casefold():
        return MIGRATION_PROCESS_CANONICAL
    return normalized


def _selected_process(arguments: argparse.Namespace) -> str:
    return (
        _process(arguments.gluon_count)
        if arguments.process_expression is None
        else _canonical_process_expression(arguments.process_expression)
    )


def _selected_process_name(arguments: argparse.Namespace) -> str:
    if arguments.process_expression is None:
        return _process_name(arguments.gluon_count)
    if _selected_process(arguments) == MIGRATION_PROCESS_CANONICAL:
        return MIGRATION_PROCESS_NAME
    return "custom_process"


def _authoritative_process_family(arguments: argparse.Namespace) -> bool:
    """Return whether the selected process belongs to an acceptance campaign."""

    return (
        arguments.process_expression is None
        or _selected_process(arguments) == MIGRATION_PROCESS_CANONICAL
    )


def _selected_union_semantic_exception_enabled(
    arguments: argparse.Namespace | None,
) -> bool:
    """Return whether the one audited union-workload projection is applicable."""

    return bool(
        arguments is not None
        and arguments.lc_flow_layout == "all-flow-union"
        and _selected_process(arguments) == MIGRATION_PROCESS_CANONICAL
        and arguments.helicity == MIGRATION_UNION_SELECTED_HELICITY_ID
        and arguments.specialize_flow_at_generation is False
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _rss_bytes(usage: resource.struct_rusage) -> int:
    """Normalize ru_maxrss (bytes on macOS, KiB on Linux) to bytes."""

    raw = float(usage.ru_maxrss)
    if platform.system() == "Darwin":
        return int(raw)
    return int(raw * 1024.0)


def _resource_peak() -> dict[str, object]:
    self_bytes = _rss_bytes(resource.getrusage(resource.RUSAGE_SELF))
    children_bytes = _rss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN))
    return {
        "source": "resource.getrusage",
        "self_peak_bytes": self_bytes,
        "maximum_child_peak_bytes": children_bytes,
        "observed_lower_bound_bytes": max(self_bytes, children_bytes),
        "semantics": _RESOURCE_PEAK_SEMANTICS,
    }


def _valid_resource_peak(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    self_bytes = value.get("self_peak_bytes")
    child_bytes = value.get("maximum_child_peak_bytes")
    observed_bytes = value.get("observed_lower_bound_bytes")
    return (
        value.get("source") == "resource.getrusage"
        and not isinstance(self_bytes, bool)
        and isinstance(self_bytes, int)
        and self_bytes >= 0
        and not isinstance(child_bytes, bool)
        and isinstance(child_bytes, int)
        and child_bytes >= 0
        and not isinstance(observed_bytes, bool)
        and isinstance(observed_bytes, int)
        and observed_bytes == max(self_bytes, child_bytes)
        and isinstance(value.get("semantics"), str)
        and bool(value["semantics"])
    )


def _artifact_stats(path: Path) -> dict[str, int]:
    file_count = 0
    size_bytes = 0
    for root, _directories, files in os.walk(path):
        directory = Path(root)
        for name in files:
            candidate = directory / name
            if candidate.is_file():
                file_count += 1
                size_bytes += candidate.stat().st_size
    return {"file_count": file_count, "size_bytes": size_bytes}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_identity(path: Path) -> dict[str, object]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        stat = resolved.stat()
    except OSError as error:
        raise HarnessError(f"cannot inspect file identity: {path}") from error
    if not resolved.is_file():
        raise HarnessError(f"identity path is not a regular file: {path}")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(resolved),
    }


def _path_state_identity(path: Path) -> dict[str, object]:
    """Record both present and legitimately absent provenance inputs."""

    if path.exists():
        if not path.is_file():
            raise HarnessError(f"provenance path is not a regular file: {path}")
        return {
            "present": True,
            **_path_identity(path),
        }
    return {
        "present": False,
        "path": str(path),
        "resolved_path": str(path.expanduser().resolve(strict=False)),
    }


def _tree_identity(path: Path) -> dict[str, object]:
    """Hash relative names and contents for a location-independent tree ID."""

    try:
        root = path.resolve(strict=True)
    except OSError as error:
        raise HarnessError(f"cannot resolve identity tree: {path}") from error
    if not root.is_dir():
        raise HarnessError(f"identity tree is not a directory: {path}")
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    try:
        candidates = list(root.rglob("*"))
        for candidate in candidates:
            if candidate.is_symlink():
                raise HarnessError(
                    f"identity tree contains an unsupported symlink: {candidate}"
                )
        members = sorted(
            (candidate for candidate in candidates if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        )
        for member in members:
            relative_text = member.relative_to(root).as_posix()
            relative = relative_text.encode("utf-8")
            size = member.stat().st_size
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(size.to_bytes(8, "big"))
            with member.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            file_count += 1
            size_bytes += size
    except OSError as error:
        raise HarnessError(f"cannot hash identity tree: {path}") from error
    return {
        "algorithm": "sha256-relative-path-size-content-v1",
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _command_identity(command: Sequence[object]) -> dict[str, object]:
    argv = [str(value) for value in command]
    return {
        "argv": argv,
        "argv_sha256": _canonical_sha256(argv),
    }


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise HarnessError(f"{label} must be a JSON object: {path}")
    return payload


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _parse_utc_timestamp(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        return None
    return parsed


def _is_utc_timestamp(value: object) -> bool:
    return _parse_utc_timestamp(value) is not None


def _utc_timestamps_nondecreasing(*values: object) -> bool:
    parsed: list[dt.datetime] = []
    for value in values:
        timestamp = _parse_utc_timestamp(value)
        if timestamp is None:
            return False
        parsed.append(timestamp)
    return all(left <= right for left, right in pairwise(parsed))


def _is_exact_int(value: object, expected: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value == expected


def _host_identity() -> dict[str, object]:
    cpu_model = platform.processor().strip()
    if not cpu_model and platform.system() == "Darwin":
        completed = subprocess.run(
            ("sysctl", "-n", "machdep.cpu.brand_string"),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            cpu_model = completed.stdout.strip()
    if not cpu_model and platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.casefold().startswith(("model name", "hardware")):
                    cpu_model = line.partition(":")[2].strip()
                    break
        except OSError:
            pass
    uname = platform.uname()
    return {
        "platform": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_model": cpu_model or None,
        "logical_cpu_count": os.cpu_count(),
    }


def _git_source_identity() -> dict[str, object]:
    revision = subprocess.run(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    head = revision.stdout.strip()
    if revision.returncode != 0 or _REVISION_PATTERN.fullmatch(head) is None:
        raise HarnessError("benchmark source has no Git revision")
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise HarnessError("could not determine benchmark source cleanliness")
    if status.stdout.strip():
        raise HarnessError(
            "benchmark source is dirty; commit or discard tracked and "
            "untracked changes before measuring"
        )
    return {
        "checkout": str(ROOT.resolve()),
        "revision": head,
        "dirty": False,
        "untracked_files_checked": True,
    }


def _installed_distribution_identity() -> dict[str, object]:
    try:
        distribution = importlib.metadata.distribution("pyamplicol")
    except importlib.metadata.PackageNotFoundError as error:
        raise HarnessError(
            "benchmark interpreter has no installed pyamplicol distribution"
        ) from error
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    native_modules: list[dict[str, object]] = []
    build_info_files: list[dict[str, object]] = []
    for entry in sorted(distribution.files or (), key=str):
        relative_text = str(entry)
        path = Path(str(distribution.locate_file(entry)))
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        relative = relative_text.encode("utf-8")
        size = resolved.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        file_count += 1
        size_bytes += size
        identity = {
            "relative_path": relative_text,
            **_path_identity(resolved),
        }
        if resolved.name.startswith("_rusticol.") and resolved.suffix in {
            ".so",
            ".pyd",
            ".dylib",
        }:
            native_modules.append(identity)
        if relative_text.endswith("pyamplicol/_build_info.json"):
            build_info_files.append(identity)
    return {
        "package_version": distribution.version,
        "distribution_content": {
            "algorithm": "sha256-relative-path-size-content-v1",
            "sha256": digest.hexdigest(),
            "file_count": file_count,
            "size_bytes": size_bytes,
        },
        "native_modules": native_modules,
        "build_info_files": build_info_files,
    }


def _active_build_info_identity() -> dict[str, object]:
    try:
        versions = importlib.import_module("pyamplicol._internal.versions")
        payload = versions._active_build_info()
    except (AttributeError, ImportError, RuntimeError) as error:
        raise HarnessError(
            "cannot resolve active pyamplicol build provenance"
        ) from error
    if not isinstance(payload, dict):
        raise HarnessError(
            "active pyamplicol runtime has no strict build-provenance record"
        )
    candidate_paths = (
        getattr(versions, "_SOURCE_BUILD_INFO_PATH", None),
        getattr(versions, "_PACKAGE_BUILD_INFO_PATH", None),
    )
    matching: list[Path] = []
    for raw_path in candidate_paths:
        if not isinstance(raw_path, Path) or not raw_path.is_file():
            continue
        try:
            candidate = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if candidate == payload:
            matching.append(raw_path)
    if not matching:
        raise HarnessError("active pyamplicol build provenance has no file identity")
    path = matching[0]
    return {
        **_path_identity(path),
        "payload": payload,
    }


def _validate_runtime_binding(
    source_identity: Mapping[str, object],
    build_info: Mapping[str, object],
    *,
    native_build_inputs_sha256: str,
) -> None:
    revision = source_identity.get("revision")
    checkout = source_identity.get("checkout")
    build_revision = build_info.get("source_revision")
    build_checkout = build_info.get("source_checkout")
    build_digest = build_info.get("native_build_inputs_sha256")
    if (
        not isinstance(revision, str)
        or _REVISION_PATTERN.fullmatch(revision) is None
        or not isinstance(checkout, str)
    ):
        raise HarnessError("benchmark source identity is incomplete")
    if (
        not isinstance(build_revision, str)
        or _REVISION_PATTERN.fullmatch(build_revision) is None
    ):
        raise HarnessError(
            "installed runtime has no clean source revision; rebuild from this checkout"
        )
    if build_revision != revision:
        raise HarnessError(
            "installed runtime source revision does not match benchmark HEAD"
        )
    if not isinstance(build_checkout, str) or not build_checkout:
        raise HarnessError("installed runtime has no source-checkout identity")
    try:
        bound_checkout = Path(build_checkout).expanduser().resolve(strict=True)
        source_checkout = Path(checkout).expanduser().resolve(strict=True)
    except OSError as error:
        raise HarnessError(
            "installed runtime source checkout is unavailable"
        ) from error
    if bound_checkout != source_checkout:
        raise HarnessError(
            "installed runtime was built from a different source checkout"
        )
    if (
        not isinstance(build_digest, str)
        or _SHA256_PATTERN.fullmatch(build_digest) is None
        or build_digest != native_build_inputs_sha256
    ):
        raise HarnessError(
            "installed runtime native build inputs do not match its build provenance"
        )


def _runtime_provenance(
    source_identity: Mapping[str, object],
) -> dict[str, object]:
    try:
        native = importlib.import_module("pyamplicol._rusticol")
    except ImportError as error:
        raise HarnessError(
            "benchmark provenance requires the pyamplicol._rusticol extension"
        ) from error
    native_path_raw = getattr(native, "__file__", None)
    if not isinstance(native_path_raw, str):
        raise HarnessError("native runtime extension has no filesystem identity")
    native_path = Path(native_path_raw).resolve()
    native_digest = getattr(native, "native_build_inputs_sha256", None)
    if not callable(native_digest):
        raise HarnessError("native runtime exposes no build-input digest")
    build_inputs_sha256 = native_digest()
    if (
        not isinstance(build_inputs_sha256, str)
        or _SHA256_PATTERN.fullmatch(build_inputs_sha256) is None
    ):
        raise HarnessError("native runtime build-input digest is invalid")
    build_info_identity = _active_build_info_identity()
    raw_build_info = build_info_identity.get("payload")
    if not isinstance(raw_build_info, Mapping):
        raise HarnessError("active pyamplicol build provenance is invalid")
    _validate_runtime_binding(
        source_identity,
        raw_build_info,
        native_build_inputs_sha256=build_inputs_sha256,
    )
    dependency_paths = (
        ROOT / "Cargo.lock",
        ROOT / "Cargo.toml",
        ROOT / "pyproject.toml",
        ROOT / "rust-toolchain.toml",
        ROOT / "dependencies" / "candidate-Cargo.lock",
        ROOT / "dependencies" / "candidate-cargo-config.toml",
        ROOT / "dependencies" / "contributor-lock.toml",
        ROOT / "dependencies" / "install-state.json",
        ROOT / "dependencies" / "python-runtime-lock.toml",
        ROOT / "dependencies" / "release-lock.toml",
    )
    native_identity = _path_identity(native_path)
    native_version = getattr(native, "package_version", None)
    native_package_version = native_version() if callable(native_version) else None
    distribution_identity = _installed_distribution_identity()
    if (
        isinstance(native_package_version, str)
        and native_package_version.replace("-dev.", ".dev")
        != distribution_identity["package_version"]
    ):
        raise HarnessError(
            "native runtime package version does not match installed distribution"
        )
    return {
        "interpreter": {
            **_path_identity(Path(sys.executable)),
            "python_version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "installed_distribution": distribution_identity,
        "active_build_info": build_info_identity,
        "native_extension": {
            **native_identity,
            "package_version": native_package_version,
            "build_inputs_sha256": build_inputs_sha256,
        },
        "dependencies": {
            str(path.relative_to(ROOT)): _path_state_identity(path)
            for path in dependency_paths
        },
    }


def _validate_expected_runtime_identity(
    source_identity: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
    *,
    expected_source_revision: str | None,
    expected_native_build_inputs_sha256: str | None,
) -> None:
    if (expected_source_revision is None) != (
        expected_native_build_inputs_sha256 is None
    ):
        raise HarnessError(
            "expected source revision and native-build-input digest must be "
            "provided together"
        )
    if expected_source_revision is None:
        return
    if _REVISION_PATTERN.fullmatch(expected_source_revision) is None:
        raise HarnessError("expected source revision is not a full Git revision")
    if (
        expected_native_build_inputs_sha256 is None
        or _SHA256_PATTERN.fullmatch(expected_native_build_inputs_sha256) is None
    ):
        raise HarnessError(
            "expected native-build-input digest is not a SHA-256 value"
        )
    native_extension = runtime_provenance.get("native_extension")
    if not isinstance(native_extension, Mapping):
        raise HarnessError("active runtime provenance has no native extension")
    if source_identity.get("revision") != expected_source_revision:
        raise HarnessError(
            "benchmark source revision does not match the externally pinned "
            "revision"
        )
    if (
        native_extension.get("build_inputs_sha256")
        != expected_native_build_inputs_sha256
    ):
        raise HarnessError(
            "benchmark native build does not match the externally pinned "
            "build-input digest"
        )


def _selected_model_identity(
    arguments: argparse.Namespace,
    *,
    mode: str,
    source_identity: Mapping[str, object],
) -> dict[str, object]:
    if arguments.prepared_model is not None:
        path = arguments.prepared_model.expanduser().resolve(strict=True)
        if not path.is_file():
            raise HarnessError(f"prepared model does not exist: {path}")
        return {
            "kind": "explicit-prepared-model",
            "resource_id": None,
            "file": _path_identity(path),
            "compile_excluded_from_generation": True,
        }
    if mode == "compiled":
        return {
            "kind": "built-in-sm-source",
            "resource_id": None,
            "source_revision": source_identity["revision"],
            "compile_excluded_from_generation": False,
        }
    try:
        from pyamplicol.assets.prepared_models import (
            BUILTIN_SM_JIT_O2,
            packaged_prepared_model_path,
        )

        with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as path:
            identity = _path_identity(path)
    except (ImportError, OSError, RuntimeError) as error:
        raise HarnessError(
            "cannot resolve the packaged built-in prepared model"
        ) from error
    return {
        "kind": "packaged-prepared-model",
        "resource_id": PREPARED_MODEL_ID,
        "size_bytes": identity["size_bytes"],
        "sha256": identity["sha256"],
        "compile_excluded_from_generation": True,
    }


def _validate_worker_model_identity(
    expected: Mapping[str, object],
    observed: object,
) -> None:
    if not isinstance(observed, Mapping):
        raise HarnessError("generation worker returned no model identity")
    for key in ("kind", "resource_id", "compile_excluded_from_generation"):
        if observed.get(key) != expected.get(key):
            raise HarnessError(f"generation worker model identity disagrees on {key}")
    expected_file = expected.get("file")
    observed_file = observed.get("file")
    if isinstance(expected_file, Mapping):
        if not isinstance(observed_file, Mapping):
            raise HarnessError(
                "generation worker returned no explicit model file identity"
            )
        for key in ("resolved_path", "size_bytes", "sha256"):
            if observed_file.get(key) != expected_file.get(key):
                raise HarnessError(
                    f"generation worker explicit model disagrees on {key}"
                )
    elif expected.get("kind") == "packaged-prepared-model":
        for key in ("size_bytes", "sha256"):
            if observed.get(key) != expected.get(key):
                raise HarnessError(
                    f"generation worker packaged model disagrees on {key}"
                )


def _semantic_generation_signature(
    arguments: argparse.Namespace,
    *,
    mode: str,
    source_identity: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
    model_identity: Mapping[str, object],
) -> dict[str, object]:
    selected_flow_word = _generation_selected_flow_word(arguments)
    return {
        "kind": "pyamplicol-benchmark-generation-signature",
        "schema_version": 1,
        "source_revision": source_identity["revision"],
        "runtime_provenance_sha256": _canonical_sha256(runtime_provenance),
        "mode": mode,
        "process": " ".join(_selected_process(arguments).split()).casefold(),
        "model": dict(model_identity),
        "color_accuracy": "lc",
        "lc_flow_layout": arguments.lc_flow_layout,
        "jit_optimization_level": arguments.jit_optimization_level,
        "point_tile_size": arguments.point_tile_size,
        "validation": {
            "enabled": True,
            "samples": arguments.validation_samples,
            "seed": VALIDATION_SEED,
            "relative_tolerance": RTOL,
            "absolute_tolerance": ATOL,
            "post_build_validation": True,
        },
        "generation": {
            "workers": 1,
            "emit_api_bundle": False,
            "specialize_flow_at_generation": (arguments.specialize_flow_at_generation),
            "selected_flow_word": (
                None
                if selected_flow_word is None
                else [int(label) for label in selected_flow_word]
            ),
        },
    }


def _artifact_payload_digests(
    manifest: Mapping[str, object],
    *,
    artifact: Path,
) -> list[dict[str, object]]:
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise HarnessError(f"artifact has no payload digest inventory: {artifact}")
    root = artifact.resolve(strict=True)
    result: list[dict[str, object]] = []
    for entry in payloads:
        if not isinstance(entry, Mapping):
            raise HarnessError(f"artifact payload entry is invalid: {artifact}")
        relative = entry.get("path")
        expected_sha256 = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(expected_sha256, str)
            or _SHA256_PATTERN.fullmatch(expected_sha256) is None
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise HarnessError(f"artifact payload identity is invalid: {artifact}")
        try:
            path = (artifact / relative).resolve(strict=True)
            path.relative_to(root)
        except (OSError, ValueError) as error:
            raise HarnessError(
                f"artifact payload escapes or is missing: {relative!r}"
            ) from error
        observed = _path_identity(path)
        if (
            observed["size_bytes"] != expected_size
            or observed["sha256"] != expected_sha256
        ):
            raise HarnessError(
                f"artifact payload does not match its manifest: {relative!r}"
            )
        result.append(
            {
                "path": relative,
                "size_bytes": expected_size,
                "sha256": expected_sha256,
                "role": entry.get("role"),
                "process_id": entry.get("process_id"),
            }
        )
    return sorted(result, key=lambda item: str(item["path"]))


def _artifact_member(
    artifact: Path,
    relative: object,
    *,
    label: str,
) -> Path:
    if not isinstance(relative, str) or not relative:
        raise HarnessError(f"artifact has no {label} path: {artifact}")
    try:
        root = artifact.resolve(strict=True)
        member = (artifact / relative).resolve(strict=True)
        member.relative_to(root)
    except (OSError, ValueError) as error:
        raise HarnessError(
            f"artifact {label} escapes or is missing: {relative!r}"
        ) from error
    if not member.is_file():
        raise HarnessError(f"artifact {label} is not a regular file: {member}")
    return member


def _ordered_physical_axis(
    raw_entries: object,
    *,
    label: str,
    require_structural_zero: bool,
) -> dict[str, object]:
    if not isinstance(raw_entries, list) or not raw_entries:
        raise HarnessError(f"artifact physical {label} axis is missing")
    identifiers: list[str] = []
    entries: list[dict[str, object]] = []
    for expected_index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise HarnessError(f"artifact physical {label} entry is invalid")
        index = raw_entry.get("index")
        identifier = raw_entry.get("id")
        representative_id = raw_entry.get("representative_id")
        computed = raw_entry.get("computed")
        structural_zero = raw_entry.get("structural_zero")
        coefficient = raw_entry.get("coefficient")
        color_kind = raw_entry.get("kind")
        color_word = raw_entry.get("word")
        helicity_values = raw_entry.get("values")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index != expected_index
            or not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or not isinstance(representative_id, str)
            or not representative_id
            or not isinstance(computed, bool)
            or (require_structural_zero and not isinstance(structural_zero, bool))
            or isinstance(coefficient, bool)
            or not isinstance(coefficient, (float, int))
            or not math.isfinite(float(coefficient))
            or (
                require_structural_zero
                and (
                    not isinstance(helicity_values, list)
                    or not helicity_values
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in helicity_values
                    )
                )
            )
            or (
                not require_structural_zero
                and (
                    color_kind != "lc-flow"
                    or not isinstance(color_word, list)
                    or not color_word
                    or structural_zero is not None
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in color_word
                    )
                )
            )
        ):
            raise HarnessError(
                f"artifact physical {label} axis is not complete and ordered"
            )
        if require_structural_zero:
            assert isinstance(helicity_values, list)
            axis_details: dict[str, object] = {"values": list(helicity_values)}
        else:
            assert isinstance(color_word, list)
            axis_details = {
                "kind": color_kind,
                "word": list(color_word),
            }
        identifiers.append(identifier)
        entries.append(
            {
                "index": index,
                "id": identifier,
                "representative_id": representative_id,
                "computed": computed,
                "coefficient": float(coefficient),
                "structural_zero": (
                    structural_zero if require_structural_zero else None
                ),
                **axis_details,
            }
        )
    by_identifier = {str(entry["id"]): entry for entry in entries}
    for entry in entries:
        representative_id = str(entry["representative_id"])
        representative = by_identifier.get(representative_id)
        computed = entry["computed"] is True
        structural_zero = entry["structural_zero"] is True
        if (
            representative is None
            or (computed and representative_id != entry["id"])
            or (structural_zero and (computed or representative_id != entry["id"]))
            or (structural_zero and entry["coefficient"] != 0.0)
            or (
                require_structural_zero
                and not structural_zero
                and entry["coefficient"] == 0.0
            )
            or (not structural_zero and representative.get("computed") is not True)
        ):
            raise HarnessError(
                f"artifact physical {label} representative mapping is not closed"
            )
    return {
        "count": len(identifiers),
        "ordered_ids": identifiers,
        "ordered_ids_sha256": _canonical_sha256(identifiers),
        "ordered_entries": entries,
        "ordered_entries_sha256": _canonical_sha256(entries),
    }


def _logical_physical_axis(
    validated_axis: Mapping[str, object],
    *,
    require_structural_zero: bool,
) -> dict[str, object]:
    """Project a validated axis onto lane-independent physical fields."""

    raw_entries = validated_axis.get("ordered_entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise HarnessError("validated physical axis has no ordered entries")
    entries: list[dict[str, object]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise HarnessError("validated physical axis entry is invalid")
        if require_structural_zero:
            entry = {
                "index": raw_entry["index"],
                "id": raw_entry["id"],
                "values": list(cast(list[int], raw_entry["values"])),
                "coefficient": raw_entry["coefficient"],
                "structural_zero": raw_entry["structural_zero"],
            }
        else:
            entry = {
                "index": raw_entry["index"],
                "id": raw_entry["id"],
                "kind": raw_entry["kind"],
                "word": list(cast(list[int], raw_entry["word"])),
                "coefficient": raw_entry["coefficient"],
            }
        entries.append(entry)
    identifiers = [str(entry["id"]) for entry in entries]
    return {
        "count": len(identifiers),
        "ordered_ids": identifiers,
        "ordered_ids_sha256": _canonical_sha256(identifiers),
        "ordered_entries": entries,
        "ordered_entries_sha256": _canonical_sha256(entries),
    }


def _validated_logical_physical_axis(
    axis: Mapping[str, object],
    *,
    label: str,
    require_structural_zero: bool,
) -> dict[str, object]:
    """Revalidate a stored lane-independent physical-axis projection."""

    raw_entries = axis.get("ordered_entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise HarnessError(f"logical physical {label} axis is missing")
    expanded: list[dict[str, object]] = []
    expected_keys = (
        {"index", "id", "values", "coefficient", "structural_zero"}
        if require_structural_zero
        else {"index", "id", "kind", "word", "coefficient"}
    )
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != expected_keys:
            raise HarnessError(f"logical physical {label} axis is invalid")
        identifier = raw_entry.get("id")
        structural_zero = (
            raw_entry.get("structural_zero") if require_structural_zero else False
        )
        expanded.append(
            {
                **dict(raw_entry),
                "representative_id": identifier,
                "computed": not bool(structural_zero),
            }
        )
    validated = _logical_physical_axis(
        _ordered_physical_axis(
            expanded,
            label=label,
            require_structural_zero=require_structural_zero,
        ),
        require_structural_zero=require_structural_zero,
    )
    if set(axis) != {
        "count",
        "ordered_ids",
        "ordered_ids_sha256",
        "ordered_entries",
        "ordered_entries_sha256",
    } or any(
        (
            not _is_exact_int(axis.get(field), cast(int, validated[field]))
            if field == "count"
            else _canonical_sha256(axis.get(field))
            != _canonical_sha256(validated[field])
        )
        for field in (
            "count",
            "ordered_ids",
            "ordered_ids_sha256",
            "ordered_entries",
            "ordered_entries_sha256",
        )
    ):
        raise HarnessError(f"logical physical {label} axis is invalid")
    return validated


def _manifest_model_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise HarnessError("artifact manifest has no model identity")
    name = model.get("name")
    content_sha256 = model.get("content_sha256")
    compiled_schema_version = model.get("compiled_schema_version")
    restriction = model.get("restriction")
    source_kind = model.get("source_kind")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(content_sha256, str)
        or _SHA256_PATTERN.fullmatch(content_sha256) is None
        or isinstance(compiled_schema_version, bool)
        or not isinstance(compiled_schema_version, int)
        or compiled_schema_version <= 0
        or (restriction is not None and not isinstance(restriction, str))
        or not isinstance(source_kind, str)
        or not source_kind
    ):
        raise HarnessError("artifact manifest model identity is incomplete")
    common_identity = {
        "name": name,
        "content_sha256": content_sha256,
        "compiled_schema_version": compiled_schema_version,
        "restriction": restriction,
    }
    full_identity = {
        **common_identity,
        "source_kind": source_kind,
    }
    return {
        "manifest": full_identity,
        "manifest_sha256": _canonical_sha256(full_identity),
        "common_physics_identity": common_identity,
        "common_physics_identity_sha256": _canonical_sha256(common_identity),
    }


def _runtime_selector_semantic_identity(
    runtime_selectors: Mapping[str, object],
    *,
    color_coverage: str,
    helicity_coverage: str,
    artifact: Path,
) -> dict[str, object]:
    axes = runtime_selectors.get("axes")
    specialized_axes = runtime_selectors.get("generation_specialized_axes")
    if (
        runtime_selectors.get("kind") != "pyamplicol-runtime-selectors"
        or not _is_exact_int(runtime_selectors.get("contract_version"), 1)
        or not isinstance(axes, Mapping)
        or set(axes) != {"color_flow", "helicity"}
        or not isinstance(specialized_axes, list)
        or any(axis not in {"helicity", "color_flow"} for axis in specialized_axes)
        or len(set(specialized_axes)) != len(specialized_axes)
    ):
        raise HarnessError(
            f"artifact runtime-selector semantics are incomplete: {artifact}"
        )
    color_axis = axes.get("color_flow")
    helicity_axis = axes.get("helicity")
    if not isinstance(color_axis, Mapping) or not isinstance(
        helicity_axis,
        Mapping,
    ):
        raise HarnessError(f"artifact runtime-selector axes are incomplete: {artifact}")
    if set(color_axis) != {
        "generation_coverage",
        "generation_selection",
        "runtime_contract",
    } or set(helicity_axis) != {
        "generation_coverage",
        "generation_selection",
        "runtime_contract",
    }:
        raise HarnessError(
            f"artifact runtime-selector axis fields are incomplete: {artifact}"
        )
    color_selection = color_axis.get("generation_selection")
    helicity_selection = helicity_axis.get("generation_selection")
    if (
        not isinstance(color_selection, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in color_selection
        )
        or len(set(color_selection)) != len(color_selection)
        or not isinstance(helicity_selection, Mapping)
        or any(
            not isinstance(label, str)
            or not label
            or isinstance(value, bool)
            or not isinstance(value, int)
            for label, value in helicity_selection.items()
        )
    ):
        raise HarnessError(
            f"artifact runtime-selector generation selection is invalid: {artifact}"
        )
    color_complete = color_coverage == "complete" and not color_selection
    helicity_complete = helicity_coverage == "complete" and not helicity_selection
    expected_specialized_axes = [
        *(["helicity"] if not helicity_complete else []),
        *(["color_flow"] if not color_complete else []),
    ]
    if (
        color_axis.get("generation_coverage") != color_coverage
        or helicity_axis.get("generation_coverage") != helicity_coverage
        or color_axis.get("runtime_contract")
        != ("complete-reusable" if color_complete else "generation-specialized")
        or helicity_axis.get("runtime_contract")
        != ("complete-reusable" if helicity_complete else "generation-specialized")
        or (
            not color_complete and (color_coverage != "selected" or not color_selection)
        )
        or (
            not helicity_complete
            and (helicity_coverage != "selected" or not helicity_selection)
        )
        or specialized_axes != expected_specialized_axes
    ):
        raise HarnessError(
            "artifact runtime-selector axes disagree with physical coverage: "
            f"{artifact}"
        )
    return {
        "kind": "pyamplicol-runtime-selectors",
        "contract_version": 1,
        "axes": {
            "color_flow": dict(color_axis),
            "helicity": dict(helicity_axis),
        },
        "generation_specialized_axes": list(specialized_axes),
    }


def _reduction_ordering_identity(
    reduction: Mapping[str, object],
    *,
    color_axis: Mapping[str, object],
    helicity_axis: Mapping[str, object],
    artifact: Path,
) -> dict[str, object]:
    reduction_kind = reduction.get("kind")
    raw_groups = reduction.get("groups")
    if (
        not isinstance(reduction_kind, str)
        or not reduction_kind
        or not isinstance(raw_groups, list)
    ):
        raise HarnessError(f"artifact ordered reduction groups are missing: {artifact}")
    raw_color_entries = color_axis.get("ordered_entries")
    raw_helicity_entries = helicity_axis.get("ordered_entries")
    if not isinstance(raw_color_entries, list) or not isinstance(
        raw_helicity_entries,
        list,
    ):
        raise HarnessError(f"artifact physical axes cannot bind reduction: {artifact}")
    color_entries = {
        str(entry["id"]): entry
        for entry in raw_color_entries
        if isinstance(entry, Mapping)
    }
    helicity_entries = {
        str(entry["id"]): entry
        for entry in raw_helicity_entries
        if isinstance(entry, Mapping)
    }
    color_ids = set(color_entries)
    computed_color_ids = {
        identifier
        for identifier, entry in color_entries.items()
        if entry.get("computed") is True
    }
    live_helicity_ids = {
        identifier
        for identifier, entry in helicity_entries.items()
        if entry.get("structural_zero") is False
    }
    reduction_groups: list[dict[str, object]] = []
    group_ids: list[str] = []
    observed_pairs: set[tuple[str, str]] = set()
    coverage_errors: list[str] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, Mapping):
            raise HarnessError(
                f"artifact ordered reduction group is invalid: {artifact}"
            )
        group_id = raw_group.get("id")
        group_color_ids = raw_group.get("physical_color_ids")
        group_helicity_ids = raw_group.get("physical_helicity_ids")
        representative_color_id = raw_group.get("representative_color_id")
        representative_helicity_id = raw_group.get("representative_helicity_id")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in group_ids
            or not isinstance(group_color_ids, list)
            or not group_color_ids
            or not isinstance(group_helicity_ids, list)
            or not group_helicity_ids
            or any(not isinstance(value, str) or not value for value in group_color_ids)
            or any(
                not isinstance(value, str) or not value for value in group_helicity_ids
            )
            or len(set(group_color_ids)) != len(group_color_ids)
            or len(set(group_helicity_ids)) != len(group_helicity_ids)
            or not isinstance(representative_color_id, str)
            or not isinstance(representative_helicity_id, str)
        ):
            raise HarnessError(
                f"artifact ordered reduction group is incomplete: {artifact}"
            )
        group_ids.append(group_id)
        if (
            not set(group_color_ids) <= color_ids
            or not set(group_helicity_ids) <= live_helicity_ids
            or representative_color_id not in group_color_ids
            or representative_helicity_id not in group_helicity_ids
            or color_entries[representative_color_id].get("computed") is not True
            or helicity_entries[representative_helicity_id].get("computed") is not True
            or any(
                color_entries[identifier].get("representative_id")
                != representative_color_id
                for identifier in group_color_ids
            )
            or any(
                helicity_entries[identifier].get("representative_id")
                != representative_helicity_id
                for identifier in group_helicity_ids
            )
        ):
            raise HarnessError(
                f"artifact reduction group is not closed over physical axes: {artifact}"
            )
        for color_id in group_color_ids:
            for helicity_id in group_helicity_ids:
                pair = (color_id, helicity_id)
                if pair in observed_pairs:
                    coverage_errors.append(f"duplicate pair {color_id}/{helicity_id}")
                observed_pairs.add(pair)
        reduction_groups.append(
            {
                "id": group_id,
                "physical_color_ids": list(group_color_ids),
                "physical_helicity_ids": list(group_helicity_ids),
                "representative_color_id": representative_color_id,
                "representative_helicity_id": representative_helicity_id,
            }
        )
    required_pairs = {
        (color_id, helicity_id)
        for color_id in computed_color_ids
        for helicity_id in live_helicity_ids
    }
    missing_pairs = required_pairs.difference(observed_pairs)
    expected_pair_count = len(required_pairs)
    if not computed_color_ids:
        coverage_errors.append("physical color axis has no computed representatives")
    if not live_helicity_ids:
        coverage_errors.append("physical helicity axis has no live entries")
    if missing_pairs:
        coverage_errors.append(
            f"covered {expected_pair_count - len(missing_pairs)} of "
            f"{expected_pair_count} required physical pairs"
        )
    ordering = {
        "kind": reduction_kind,
        "ordered_groups": reduction_groups,
    }
    return {
        "ordering": ordering,
        "ordering_sha256": _canonical_sha256(ordering),
        "coverage": {
            "complete": not coverage_errors,
            "expected_physical_pair_count": expected_pair_count,
            "observed_physical_pair_count": len(observed_pairs),
            "errors": coverage_errors,
        },
    }


def _logical_reduction_ordering_identity(
    reduction_kind: object,
    *,
    color_axis: Mapping[str, object],
    helicity_axis: Mapping[str, object],
    artifact: Path,
) -> dict[str, object]:
    """Describe public reduction order without encoding lane materialization."""

    if not isinstance(reduction_kind, str) or not reduction_kind:
        raise HarnessError(f"artifact logical reduction kind is invalid: {artifact}")
    raw_color_ids = color_axis.get("ordered_ids")
    raw_helicity_entries = helicity_axis.get("ordered_entries")
    if not isinstance(raw_color_ids, list) or not isinstance(
        raw_helicity_entries, list
    ):
        raise HarnessError(
            f"artifact logical reduction has incomplete physical axes: {artifact}"
        )
    color_ids = [
        str(identifier)
        for identifier in raw_color_ids
        if isinstance(identifier, str) and identifier
    ]
    live_helicity_ids = [
        str(entry["id"])
        for entry in raw_helicity_entries
        if isinstance(entry, Mapping) and entry.get("structural_zero") is False
    ]
    coverage_errors: list[str] = []
    if len(color_ids) != len(raw_color_ids) or len(set(color_ids)) != len(color_ids):
        coverage_errors.append("physical color IDs are invalid")
    if not live_helicity_ids or len(set(live_helicity_ids)) != len(live_helicity_ids):
        coverage_errors.append("live physical helicity IDs are invalid")
    pair_count = len(color_ids) * len(live_helicity_ids)
    ordering = {
        "kind": reduction_kind,
        "pair_order_abi": LOGICAL_REDUCTION_ORDER_ABI,
        "physical_color_count": len(color_ids),
        "physical_color_ids_sha256": _canonical_sha256(color_ids),
        "live_physical_helicity_count": len(live_helicity_ids),
        "live_physical_helicity_ids_sha256": _canonical_sha256(live_helicity_ids),
        "physical_pair_count": pair_count,
    }
    return {
        "ordering": ordering,
        "ordering_sha256": _canonical_sha256(ordering),
        "coverage": {
            "complete": not coverage_errors and pair_count > 0,
            "expected_physical_pair_count": pair_count,
            "observed_physical_pair_count": (pair_count if not coverage_errors else 0),
            "errors": coverage_errors,
        },
    }


def _load_compact_eager_reduction_groups(
    artifact: Path,
    process_id: str,
) -> list[dict[str, object]]:
    try:
        from pyamplicol.runtime.eager_exact._plan_v3 import (
            _load_eager_reduction_groups_v1,
        )

        groups = _load_eager_reduction_groups_v1(artifact, process_id)
    except Exception as error:
        raise HarnessError(
            f"could not authenticate compact eager reduction: {artifact}"
        ) from error
    return [dict(group) for group in groups]


def _load_compact_recurrence_reduction(
    artifact: Path,
    process_id: str,
) -> Any:
    try:
        from pyamplicol.runtime.recurrence_exact._plan_v2 import (
            _load_recurrence_exact_sections_v1,
        )

        return _load_recurrence_exact_sections_v1(artifact, process_id)
    except Exception as error:
        raise HarnessError(
            f"could not authenticate compact recurrence reduction: {artifact}"
        ) from error


def _execution_reduction_identity(
    reduction: Mapping[str, object],
    *,
    extensions: Mapping[str, object],
    color_axis: Mapping[str, object],
    helicity_axis: Mapping[str, object],
    logical_color_axis: Mapping[str, object],
    logical_helicity_axis: Mapping[str, object],
    artifact: Path,
    process_id: str,
) -> dict[str, object]:
    """Authenticate expanded or native lane-local reduction materialization."""

    reduction_kind = reduction.get("kind")
    raw_groups = reduction.get("groups")
    if (
        not isinstance(reduction_kind, str)
        or not reduction_kind
        or not isinstance(raw_groups, list)
    ):
        raise HarnessError(f"artifact execution reduction is invalid: {artifact}")
    color_entries = color_axis.get("ordered_entries")
    helicity_entries = helicity_axis.get("ordered_entries")
    if not isinstance(color_entries, list) or not isinstance(
        helicity_entries,
        list,
    ):
        raise HarnessError(f"artifact execution reduction axes are invalid: {artifact}")
    computed_color_count = sum(
        isinstance(entry, Mapping) and entry.get("computed") is True
        for entry in color_entries
    )
    live_helicity_count = sum(
        isinstance(entry, Mapping) and entry.get("structural_zero") is False
        for entry in helicity_entries
    )
    if computed_color_count <= 0 or live_helicity_count <= 0:
        raise HarnessError(
            f"artifact execution reduction axes are incomplete: {artifact}"
        )
    has_eager_descriptor = "native_reduction_groups" in extensions
    has_recurrence_descriptor = "recurrence_runtime_reduction" in extensions
    eager_descriptor = extensions.get("native_reduction_groups")
    recurrence_descriptor = extensions.get("recurrence_runtime_reduction")
    if raw_groups:
        if has_eager_descriptor or has_recurrence_descriptor:
            raise HarnessError(
                f"expanded reduction duplicates a native descriptor: {artifact}"
            )
        expanded = _reduction_ordering_identity(
            reduction,
            color_axis=color_axis,
            helicity_axis=helicity_axis,
            artifact=artifact,
        )
        return {
            "identity": {
                "kind": "expanded-json-reduction-v1",
                "reduction_kind": reduction_kind,
                "group_count": len(raw_groups),
                "computed_physical_color_count": computed_color_count,
                "live_physical_helicity_count": live_helicity_count,
                "materialized_ordering_sha256": expanded["ordering_sha256"],
            },
            "coverage": expanded["coverage"],
        }

    if has_eager_descriptor:
        expected_keys = {
            "kind",
            "schema_version",
            "storage_abi",
            "runtime_layout_abi",
            "container_path",
            "group_member",
            "entry_member",
            "group_count",
        }
        group_count = (
            eager_descriptor.get("group_count")
            if isinstance(eager_descriptor, Mapping)
            else None
        )
        if (
            has_recurrence_descriptor
            or not isinstance(eager_descriptor, Mapping)
            or set(eager_descriptor) != expected_keys
            or eager_descriptor.get("kind")
            != "pyamplicol-eager-plan-v3-reduction-groups"
            or not _is_exact_int(eager_descriptor.get("schema_version"), 1)
            or eager_descriptor.get("storage_abi") != "pacbin-v1"
            or eager_descriptor.get("runtime_layout_abi")
            != "pyamplicol-eager-runtime-layout-v1"
            or eager_descriptor.get("container_path") != "eager-runtime.pacbin"
            or eager_descriptor.get("group_member") != "reductions/groups.bin"
            or eager_descriptor.get("entry_member") != "reductions/entries.bin"
            or isinstance(group_count, bool)
            or not isinstance(group_count, int)
            or group_count <= 0
        ):
            raise HarnessError(
                f"compact eager reduction descriptor is invalid: {artifact}"
            )
        groups = _load_compact_eager_reduction_groups(artifact, process_id)
        if len(groups) != group_count:
            raise HarnessError(
                f"compact eager reduction count is inconsistent: {artifact}"
            )
        hydrated = _reduction_ordering_identity(
            {"kind": reduction_kind, "groups": groups},
            color_axis=color_axis,
            helicity_axis=helicity_axis,
            artifact=artifact,
        )
        return {
            "identity": {
                "kind": "eager-plan-v3-pacbin-reduction-v1",
                "reduction_kind": reduction_kind,
                "descriptor": dict(eager_descriptor),
                "group_count": len(groups),
                "computed_physical_color_count": computed_color_count,
                "live_physical_helicity_count": live_helicity_count,
                "materialized_ordering_sha256": hydrated["ordering_sha256"],
            },
            "coverage": hydrated["coverage"],
        }

    if has_recurrence_descriptor:
        expected_keys = {
            "kind",
            "runtime_layout_abi",
            "container_path",
            "plan_member_path",
        }
        if (
            has_eager_descriptor
            or not isinstance(recurrence_descriptor, Mapping)
            or set(recurrence_descriptor) != expected_keys
            or recurrence_descriptor.get("kind")
            != "pyamplicol-recurrence-native-reduction-v2"
            or recurrence_descriptor.get("runtime_layout_abi")
            != "pyamplicol-recurrence-runtime-layout-v2"
            or recurrence_descriptor.get("container_path")
            != "recurrence-runtime.pacbin"
            or recurrence_descriptor.get("plan_member_path")
            != "schedule/recurrence-direct-schedule-v2.bin"
        ):
            raise HarnessError(
                f"compact recurrence reduction descriptor is invalid: {artifact}"
            )
        sections = _load_compact_recurrence_reduction(artifact, process_id)
        color_ids = cast(list[str], logical_color_axis["ordered_ids"])
        live_entries = [
            entry
            for entry in cast(
                list[Mapping[str, object]],
                logical_helicity_axis["ordered_entries"],
            )
            if entry.get("structural_zero") is False
        ]
        live_values = [
            tuple(cast(list[int], entry["values"])) for entry in live_entries
        ]
        resolved_helicities = tuple(sections.resolved_helicities)
        public_helicities = tuple(sections.public_helicities)
        section_values = [
            tuple(
                public_helicities[
                    row.public_helicity_start : row.public_helicity_start
                    + row.public_helicity_count
                ]
            )
            for row in resolved_helicities
        ]
        public_flow_ids = tuple(sections.public_flow_ids)
        strategy = sections.strategy
        pair_count = len(color_ids) * len(live_values)
        complete = (
            len(public_flow_ids) == len(color_ids) and section_values == live_values
        )
        if strategy == "topology-replay":
            replay_targets = tuple(sections.replay_targets)
            complete = (
                complete
                and public_flow_ids == tuple(range(len(color_ids)))
                and tuple(row.public_flow_id for row in replay_targets)
                == public_flow_ids
                and sections.amplitude_destination_count == len(live_values)
                and len(sections.replay_helicity_map) == pair_count
            )
        elif strategy == "all-flow-union":
            destination_sectors = {
                row.target_sector_id for row in sections.amplitude_destinations
            }
            complete = (
                complete
                and destination_sectors == set(public_flow_ids)
                and len(sections.amplitude_destinations) == len(destination_sectors)
                and sections.amplitude_destination_count
                == len(sections.amplitude_destinations)
            )
        else:
            complete = False
        if not complete or pair_count <= 0:
            raise HarnessError(
                f"compact recurrence reduction coverage is incomplete: {artifact}"
            )
        destination_rows = [
            [row.target_sector_id, row.target_helicity_id]
            for row in sections.amplitude_destinations
        ]
        replay_map = list(sections.replay_helicity_map)
        return {
            "identity": {
                "kind": "recurrence-plan-v2-pacbin-reduction-v1",
                "reduction_kind": reduction_kind,
                "descriptor": dict(recurrence_descriptor),
                "strategy": strategy,
                "semantic_digest": sections.semantic_digest,
                "runtime_layout_digest": sections.runtime_layout_digest,
                "physical_color_count": len(color_ids),
                "live_physical_helicity_count": len(live_values),
                "public_flow_binding_count": len(public_flow_ids),
                "public_flow_bindings": list(public_flow_ids),
                "public_flow_bindings_sha256": _canonical_sha256(list(public_flow_ids)),
                "construction_sector_count": len(set(public_flow_ids)),
                "amplitude_destination_count": (sections.amplitude_destination_count),
                "destination_row_count": len(destination_rows),
                "destination_rows": destination_rows,
                "destination_rows_sha256": _canonical_sha256(destination_rows),
                "replay_helicity_map_count": len(replay_map),
                "replay_helicity_map_sha256": _canonical_sha256(replay_map),
            },
            "coverage": {
                "complete": True,
                "expected_physical_pair_count": pair_count,
                "observed_physical_pair_count": pair_count,
                "errors": [],
            },
        }

    raise HarnessError(
        f"empty reduction groups have no authenticated native descriptor: {artifact}"
    )


def _validate_execution_reduction_summary(
    identity: Mapping[str, object],
    coverage: Mapping[str, object],
    *,
    logical_reduction: Mapping[str, object],
    artifact: Path,
) -> None:
    """Revalidate the stored lane-local summary without reopening its artifact."""

    logical_kind = logical_reduction.get("kind")
    logical_pair_count = logical_reduction.get("physical_pair_count")
    physical_color_count = logical_reduction.get("physical_color_count")
    live_helicity_count = logical_reduction.get("live_physical_helicity_count")
    expected_pair_count = coverage.get("expected_physical_pair_count")
    observed_pair_count = coverage.get("observed_physical_pair_count")
    if (
        not isinstance(logical_kind, str)
        or not logical_kind
        or isinstance(logical_pair_count, bool)
        or not isinstance(logical_pair_count, int)
        or logical_pair_count <= 0
        or isinstance(physical_color_count, bool)
        or not isinstance(physical_color_count, int)
        or physical_color_count <= 0
        or isinstance(live_helicity_count, bool)
        or not isinstance(live_helicity_count, int)
        or live_helicity_count <= 0
        or set(coverage)
        != {
            "complete",
            "expected_physical_pair_count",
            "observed_physical_pair_count",
            "errors",
        }
        or coverage.get("complete") is not True
        or isinstance(expected_pair_count, bool)
        or not isinstance(expected_pair_count, int)
        or expected_pair_count <= 0
        or expected_pair_count > logical_pair_count
        or not _is_exact_int(observed_pair_count, expected_pair_count)
        or coverage.get("errors") != []
    ):
        raise HarnessError(
            f"artifact execution reduction coverage is invalid: {artifact}"
        )

    kind = identity.get("kind")
    if kind == "expanded-json-reduction-v1":
        group_count = identity.get("group_count")
        computed_color_count = identity.get("computed_physical_color_count")
        materialized_sha256 = identity.get("materialized_ordering_sha256")
        if (
            set(identity)
            != {
                "kind",
                "reduction_kind",
                "group_count",
                "computed_physical_color_count",
                "live_physical_helicity_count",
                "materialized_ordering_sha256",
            }
            or identity.get("reduction_kind") != logical_kind
            or isinstance(group_count, bool)
            or not isinstance(group_count, int)
            or group_count <= 0
            or group_count > observed_pair_count
            or isinstance(computed_color_count, bool)
            or not isinstance(computed_color_count, int)
            or computed_color_count <= 0
            or computed_color_count > physical_color_count
            or not _is_exact_int(
                identity.get("live_physical_helicity_count"),
                live_helicity_count,
            )
            or expected_pair_count != computed_color_count * live_helicity_count
            or not isinstance(materialized_sha256, str)
            or _SHA256_PATTERN.fullmatch(materialized_sha256) is None
        ):
            raise HarnessError(
                f"expanded execution reduction summary is invalid: {artifact}"
            )
        return

    if kind == "eager-plan-v3-pacbin-reduction-v1":
        descriptor = identity.get("descriptor")
        group_count = identity.get("group_count")
        computed_color_count = identity.get("computed_physical_color_count")
        materialized_sha256 = identity.get("materialized_ordering_sha256")
        if (
            set(identity)
            != {
                "kind",
                "reduction_kind",
                "descriptor",
                "group_count",
                "computed_physical_color_count",
                "live_physical_helicity_count",
                "materialized_ordering_sha256",
            }
            or identity.get("reduction_kind") != logical_kind
            or not isinstance(descriptor, Mapping)
            or set(descriptor)
            != {
                "kind",
                "schema_version",
                "storage_abi",
                "runtime_layout_abi",
                "container_path",
                "group_member",
                "entry_member",
                "group_count",
            }
            or descriptor.get("kind") != "pyamplicol-eager-plan-v3-reduction-groups"
            or not _is_exact_int(descriptor.get("schema_version"), 1)
            or descriptor.get("storage_abi") != "pacbin-v1"
            or descriptor.get("runtime_layout_abi")
            != "pyamplicol-eager-runtime-layout-v1"
            or descriptor.get("container_path") != "eager-runtime.pacbin"
            or descriptor.get("group_member") != "reductions/groups.bin"
            or descriptor.get("entry_member") != "reductions/entries.bin"
            or isinstance(group_count, bool)
            or not isinstance(group_count, int)
            or group_count <= 0
            or group_count > observed_pair_count
            or not _is_exact_int(descriptor.get("group_count"), group_count)
            or isinstance(computed_color_count, bool)
            or not isinstance(computed_color_count, int)
            or computed_color_count <= 0
            or computed_color_count > physical_color_count
            or not _is_exact_int(
                identity.get("live_physical_helicity_count"),
                live_helicity_count,
            )
            or expected_pair_count != computed_color_count * live_helicity_count
            or not isinstance(materialized_sha256, str)
            or _SHA256_PATTERN.fullmatch(materialized_sha256) is None
        ):
            raise HarnessError(
                f"compact eager execution reduction summary is invalid: {artifact}"
            )
        return

    if kind == "recurrence-plan-v2-pacbin-reduction-v1":
        descriptor = identity.get("descriptor")
        strategy = identity.get("strategy")
        semantic_digest = identity.get("semantic_digest")
        runtime_layout_digest = identity.get("runtime_layout_digest")
        public_flow_binding_count = identity.get("public_flow_binding_count")
        public_flow_bindings = identity.get("public_flow_bindings")
        public_flow_bindings_sha256 = identity.get("public_flow_bindings_sha256")
        construction_sector_count = identity.get("construction_sector_count")
        amplitude_destination_count = identity.get("amplitude_destination_count")
        destination_row_count = identity.get("destination_row_count")
        destination_rows = identity.get("destination_rows")
        destination_rows_sha256 = identity.get("destination_rows_sha256")
        replay_map_count = identity.get("replay_helicity_map_count")
        replay_map_sha256 = identity.get("replay_helicity_map_sha256")
        public_flow_bindings_are_valid = isinstance(
            public_flow_bindings,
            list,
        ) and all(
            not isinstance(value, bool) and isinstance(value, int) and value >= 0
            for value in public_flow_bindings
        )
        destination_rows_are_valid = isinstance(destination_rows, list) and all(
            isinstance(row, list)
            and len(row) == 2
            and all(
                not isinstance(value, bool)
                and isinstance(value, int)
                and 0 <= value < 2**32
                for value in row
            )
            for row in destination_rows
        )
        construction_sector_ids = (
            set(public_flow_bindings) if public_flow_bindings_are_valid else set()
        )
        destination_sector_ids = (
            [row[0] for row in destination_rows] if destination_rows_are_valid else []
        )
        destination_helicity_ids = (
            [row[1] for row in destination_rows] if destination_rows_are_valid else []
        )
        if (
            set(identity)
            != {
                "kind",
                "reduction_kind",
                "descriptor",
                "strategy",
                "semantic_digest",
                "runtime_layout_digest",
                "physical_color_count",
                "live_physical_helicity_count",
                "public_flow_binding_count",
                "public_flow_bindings",
                "public_flow_bindings_sha256",
                "construction_sector_count",
                "amplitude_destination_count",
                "destination_row_count",
                "destination_rows",
                "destination_rows_sha256",
                "replay_helicity_map_count",
                "replay_helicity_map_sha256",
            }
            or identity.get("reduction_kind") != logical_kind
            or not isinstance(descriptor, Mapping)
            or set(descriptor)
            != {
                "kind",
                "runtime_layout_abi",
                "container_path",
                "plan_member_path",
            }
            or descriptor.get("kind") != "pyamplicol-recurrence-native-reduction-v2"
            or descriptor.get("runtime_layout_abi")
            != "pyamplicol-recurrence-runtime-layout-v2"
            or descriptor.get("container_path") != "recurrence-runtime.pacbin"
            or descriptor.get("plan_member_path")
            != "schedule/recurrence-direct-schedule-v2.bin"
            or strategy not in {"topology-replay", "all-flow-union"}
            or not isinstance(semantic_digest, str)
            or _SHA256_PATTERN.fullmatch(semantic_digest) is None
            or not isinstance(runtime_layout_digest, str)
            or _SHA256_PATTERN.fullmatch(runtime_layout_digest) is None
            or not _is_exact_int(
                identity.get("physical_color_count"),
                physical_color_count,
            )
            or not _is_exact_int(
                identity.get("live_physical_helicity_count"),
                live_helicity_count,
            )
            or not _is_exact_int(public_flow_binding_count, physical_color_count)
            or not public_flow_bindings_are_valid
            or len(public_flow_bindings) != physical_color_count
            or not isinstance(public_flow_bindings_sha256, str)
            or _SHA256_PATTERN.fullmatch(public_flow_bindings_sha256) is None
            or public_flow_bindings_sha256 != _canonical_sha256(public_flow_bindings)
            or isinstance(construction_sector_count, bool)
            or not isinstance(construction_sector_count, int)
            or construction_sector_count <= 0
            or construction_sector_count > physical_color_count
            or len(construction_sector_ids) != construction_sector_count
            or construction_sector_ids != set(range(construction_sector_count))
            or isinstance(amplitude_destination_count, bool)
            or not isinstance(amplitude_destination_count, int)
            or amplitude_destination_count <= 0
            or not _is_exact_int(destination_row_count, amplitude_destination_count)
            or not destination_rows_are_valid
            or len(destination_rows) != amplitude_destination_count
            or len({tuple(row) for row in destination_rows})
            != amplitude_destination_count
            or not isinstance(destination_rows_sha256, str)
            or _SHA256_PATTERN.fullmatch(destination_rows_sha256) is None
            or destination_rows_sha256 != _canonical_sha256(destination_rows)
            or isinstance(replay_map_count, bool)
            or not isinstance(replay_map_count, int)
            or replay_map_count < 0
            or not isinstance(replay_map_sha256, str)
            or _SHA256_PATTERN.fullmatch(replay_map_sha256) is None
            or expected_pair_count != logical_pair_count
            or (
                strategy == "topology-replay"
                and (
                    construction_sector_count != physical_color_count
                    or public_flow_bindings != list(range(physical_color_count))
                    or amplitude_destination_count != live_helicity_count
                    or destination_helicity_ids != list(range(live_helicity_count))
                    or not set(destination_sector_ids) <= construction_sector_ids
                    or replay_map_count != logical_pair_count
                )
            )
            or (
                strategy == "all-flow-union"
                and (
                    amplitude_destination_count != construction_sector_count
                    or set(destination_sector_ids) != construction_sector_ids
                    or any(value != 2**32 - 1 for value in destination_helicity_ids)
                    or replay_map_count != 0
                    or replay_map_sha256 != _canonical_sha256([])
                )
            )
        ):
            raise HarnessError(
                f"compact recurrence execution reduction summary is invalid: {artifact}"
            )
        return

    raise HarnessError(f"artifact execution reduction summary is invalid: {artifact}")


def _artifact_semantic_identity(
    artifact: Path,
    manifest: Mapping[str, object],
    process: Mapping[str, object],
) -> dict[str, object]:
    """Bind physical axes, normalization, and ordered execution reductions."""

    physics_path = _artifact_member(
        artifact,
        process.get("physics_path"),
        label="resolved physics",
    )
    physics = _json_object(physics_path, label="artifact resolved physics")
    coverage = physics.get("coverage")
    extensions = physics.get("extensions")
    reduction = physics.get("reduction")
    if (
        not isinstance(coverage, Mapping)
        or not isinstance(extensions, Mapping)
        or not isinstance(reduction, Mapping)
    ):
        raise HarnessError(
            f"artifact semantic physics contract is incomplete: {artifact}"
        )
    color_coverage = coverage.get("color")
    helicity_coverage = coverage.get("helicities")
    if not isinstance(color_coverage, str) or not isinstance(
        helicity_coverage,
        str,
    ):
        raise HarnessError(f"artifact physical-axis coverage is invalid: {artifact}")
    execution_color_axis = _ordered_physical_axis(
        physics.get("color_components"),
        label="color-flow",
        require_structural_zero=False,
    )
    execution_helicity_axis = _ordered_physical_axis(
        physics.get("helicities"),
        label="helicity",
        require_structural_zero=True,
    )
    color_axis = _logical_physical_axis(
        execution_color_axis,
        require_structural_zero=False,
    )
    helicity_axis = _logical_physical_axis(
        execution_helicity_axis,
        require_structural_zero=True,
    )
    normalization = extensions.get("normalization")
    runtime_selectors = extensions.get("runtime_selectors")
    if not isinstance(normalization, Mapping) or not normalization:
        raise HarnessError(f"artifact normalization contract is missing: {artifact}")
    if not isinstance(runtime_selectors, Mapping):
        raise HarnessError(f"artifact runtime-selector contract is missing: {artifact}")
    runtime_selector_semantics = _runtime_selector_semantic_identity(
        runtime_selectors,
        color_coverage=color_coverage,
        helicity_coverage=helicity_coverage,
        artifact=artifact,
    )
    specialized_axes = runtime_selector_semantics["generation_specialized_axes"]
    assert isinstance(specialized_axes, list)
    process_id = process.get("id")
    if not isinstance(process_id, str) or not process_id:
        raise HarnessError(f"artifact process ID is invalid: {artifact}")
    reduction_identity = _logical_reduction_ordering_identity(
        reduction.get("kind"),
        color_axis=color_axis,
        helicity_axis=helicity_axis,
        artifact=artifact,
    )
    execution_reduction = _execution_reduction_identity(
        reduction,
        extensions=extensions,
        color_axis=execution_color_axis,
        helicity_axis=execution_helicity_axis,
        logical_color_axis=color_axis,
        logical_helicity_axis=helicity_axis,
        artifact=artifact,
        process_id=process_id,
    )
    model_identity = _manifest_model_identity(manifest)
    manifest_extensions = manifest.get("extensions")
    generation = (
        manifest_extensions.get("generation")
        if isinstance(manifest_extensions, Mapping)
        else None
    )
    concrete_processes = (
        generation.get("concrete_processes")
        if isinstance(generation, Mapping)
        else None
    )
    matching_processes = (
        [
            dict(entry)
            for entry in concrete_processes
            if isinstance(entry, Mapping) and entry.get("id") == process_id
        ]
        if isinstance(concrete_processes, list)
        else []
    )
    if len(matching_processes) != 1:
        raise HarnessError(
            f"artifact runtime schedule identity is missing or ambiguous: {artifact}"
        )
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise HarnessError(f"artifact payload ordering is missing: {artifact}")
    payload_order: list[dict[str, object]] = []
    for entry in payloads:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise HarnessError(f"artifact payload ordering is invalid: {artifact}")
        payload_order.append(
            {
                "path": entry["path"],
                "role": entry.get("role"),
                "process_id": entry.get("process_id"),
            }
        )
    normalization_record = dict(normalization)
    execution_schedule_ordering = {
        "runtime_process_contract": matching_processes[0],
        "manifest_payload_order": payload_order,
    }
    complete_physical_axes = (
        color_coverage == "complete" and helicity_coverage == "complete"
    )
    physics_file_identity = _path_identity(physics_path)
    return {
        "kind": "pyamplicol-benchmark-artifact-semantic-identity",
        "schema_version": ARTIFACT_SEMANTIC_IDENTITY_SCHEMA,
        "physics_file": {
            "relative_path": process["physics_path"],
            "size_bytes": physics_file_identity["size_bytes"],
            "sha256": physics_file_identity["sha256"],
        },
        "coverage": {
            "color": color_coverage,
            "helicities": helicity_coverage,
            "complete_physical_axes": complete_physical_axes,
        },
        "physical_color_flows": color_axis,
        "physical_helicities": helicity_axis,
        "normalization": normalization_record,
        "normalization_sha256": _canonical_sha256(normalization_record),
        "manifest_model_identity": model_identity,
        "reduction_ordering": reduction_identity["ordering"],
        "reduction_ordering_sha256": reduction_identity["ordering_sha256"],
        "reduction_coverage": reduction_identity["coverage"],
        "execution_reduction_identity": execution_reduction["identity"],
        "execution_reduction_identity_sha256": _canonical_sha256(
            execution_reduction["identity"]
        ),
        "execution_reduction_coverage": execution_reduction["coverage"],
        "execution_schedule_ordering": execution_schedule_ordering,
        "execution_schedule_ordering_sha256": _canonical_sha256(
            execution_schedule_ordering
        ),
        "generation_specialized_axes": list(specialized_axes),
        "runtime_selector_semantics": runtime_selector_semantics,
        "runtime_selector_semantics_sha256": _canonical_sha256(
            runtime_selector_semantics
        ),
        "runtime_selectors": dict(runtime_selectors),
        "runtime_selectors_sha256": _canonical_sha256(runtime_selectors),
    }


def _artifact_identity(path: Path) -> dict[str, object]:
    manifest_path = path / "artifact.json"
    manifest = _json_object(manifest_path, label="artifact manifest")
    processes = manifest.get("processes")
    if not isinstance(processes, list) or len(processes) != 1:
        raise HarnessError(f"benchmark artifact must contain one process: {path}")
    process = processes[0]
    if not isinstance(process, Mapping):
        raise HarnessError(f"artifact process record is invalid: {path}")
    expression = process.get("expression")
    process_id = process.get("id")
    color_accuracy = process.get("color_accuracy")
    if not isinstance(expression, str) or not expression:
        raise HarnessError(f"artifact process expression is invalid: {path}")
    if not isinstance(process_id, str) or not process_id:
        raise HarnessError(f"artifact process ID is invalid: {path}")
    semantic_identity = _artifact_semantic_identity(path, manifest, process)
    model_identity = _manifest_model_identity(manifest)
    return {
        "path": str(path.resolve()),
        "artifact_id": manifest.get("artifact_id"),
        "manifest": _path_identity(manifest_path),
        "tree": _tree_identity(path),
        "payloads": _artifact_payload_digests(manifest, artifact=path),
        "process_id": process_id,
        "process_expression": expression,
        "color_accuracy": color_accuracy,
        "producer": manifest.get("producer"),
        "model_identity": model_identity,
        "semantic_identity": semantic_identity,
        "semantic_identity_sha256": _canonical_sha256(semantic_identity),
    }


def _validate_artifact_contract(
    artifact: Path,
    identity: Mapping[str, object],
    *,
    arguments: argparse.Namespace,
    mode: str,
) -> dict[str, object]:
    expected_process = " ".join(_selected_process(arguments).split()).casefold()
    observed_process = identity.get("process_expression")
    if (
        not isinstance(observed_process, str)
        or " ".join(observed_process.split()).casefold() != expected_process
    ):
        raise HarnessError(f"reused artifact process does not match: {artifact}")
    if identity.get("color_accuracy") != "lc":
        raise HarnessError(f"benchmark artifact is not LC: {artifact}")
    effective = _effective_contract(artifact)
    expected = {
        "execution_mode": mode,
        "backend": "jit",
        "jit_optimization_level": _expected_effective_jit_optimization_level(
            arguments,
            mode=mode,
        ),
        "color_accuracy": "lc",
        "lc_flow_layout": arguments.lc_flow_layout,
    }
    for key, value in expected.items():
        if effective.get(key) != value:
            raise HarnessError(
                f"artifact effective {key} does not match {value!r}: {artifact}"
            )
    return effective


def _expected_effective_jit_optimization_level(
    arguments: argparse.Namespace,
    *,
    mode: str,
) -> int:
    """Return the executable optimization level, not merely the request.

    Process-local compiled DAGs honor the requested JIT level, including when
    their model source is an explicit prepared-model bundle.  Eager and
    recurrence lanes execute that bundle's immutable portable applications,
    which are deliberately stored at O2 even when process generation requests
    O3.
    """

    if mode != "compiled":
        return PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL
    return int(arguments.jit_optimization_level)


def _reuse_signature_path(artifact: Path) -> Path:
    return artifact.with_name(f"{artifact.name}.benchmark-reuse.json")


def _validated_command_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise HarnessError("command identity is missing")
    argv = value.get("argv")
    digest = value.get("argv_sha256")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(entry, str) for entry in argv)
        or not isinstance(digest, str)
        or digest != _canonical_sha256(argv)
    ):
        raise HarnessError("command identity is invalid")
    return {
        "argv": list(argv),
        "argv_sha256": digest,
    }


def _write_reuse_signature(
    artifact: Path,
    *,
    signature: Mapping[str, object],
    artifact_identity: Mapping[str, object],
    generation_command: object,
) -> None:
    artifact_semantic_identity = artifact_identity.get("semantic_identity")
    if not isinstance(artifact_semantic_identity, Mapping):
        raise HarnessError("artifact has no semantic identity for its reuse signature")
    combined_signature = {
        "generation_request": dict(signature),
        "artifact_semantic_identity": dict(artifact_semantic_identity),
    }
    payload = {
        "kind": REUSE_SIGNATURE_KIND,
        "schema_version": REUSE_SIGNATURE_SCHEMA,
        "created_at_utc": _utc_now(),
        "generation_command": _validated_command_identity(generation_command),
        "semantic_signature": combined_signature,
        "semantic_signature_sha256": _canonical_sha256(combined_signature),
        "generation_request_sha256": _canonical_sha256(signature),
        "artifact_semantic_identity_sha256": _canonical_sha256(
            artifact_semantic_identity
        ),
        "artifact_tree": artifact_identity["tree"],
    }
    _write_json_atomic(_reuse_signature_path(artifact), payload)


def _validated_reuse_signature(
    artifact: Path,
    *,
    artifact_identity: Mapping[str, object],
    expected_signature: Mapping[str, object] | None,
) -> dict[str, object]:
    sidecar_path = _reuse_signature_path(artifact)
    if not sidecar_path.is_file():
        raise HarnessError(
            f"artifact reuse signature is missing; rerun with --force: {artifact}"
        )
    sidecar = _json_object(sidecar_path, label="artifact reuse signature")
    if sidecar.get("kind") != REUSE_SIGNATURE_KIND or not _is_exact_int(
        sidecar.get("schema_version"),
        REUSE_SIGNATURE_SCHEMA,
    ):
        raise HarnessError(
            f"artifact reuse signature has an unsupported schema: {sidecar_path}"
        )
    _validated_command_identity(sidecar.get("generation_command"))
    created_at = sidecar.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at:
        raise HarnessError(f"artifact reuse signature has no timestamp: {sidecar_path}")
    try:
        parsed_created_at = dt.datetime.fromisoformat(created_at)
    except ValueError as error:
        raise HarnessError(
            f"artifact reuse signature has an invalid timestamp: {sidecar_path}"
        ) from error
    if parsed_created_at.tzinfo is None:
        raise HarnessError(
            f"artifact reuse signature timestamp is not UTC-bound: {sidecar_path}"
        )
    observed_signature = sidecar.get("semantic_signature")
    if not isinstance(observed_signature, Mapping):
        raise HarnessError(f"artifact reuse signature is invalid: {sidecar_path}")
    generation_request = observed_signature.get("generation_request")
    artifact_semantic_identity = observed_signature.get("artifact_semantic_identity")
    observed_artifact_semantics = artifact_identity.get("semantic_identity")
    if not isinstance(generation_request, Mapping) or not isinstance(
        artifact_semantic_identity,
        Mapping,
    ):
        raise HarnessError(f"artifact reuse signature is incomplete: {sidecar_path}")
    observed_digest = sidecar.get("semantic_signature_sha256")
    if (
        not isinstance(observed_digest, str)
        or observed_digest != _canonical_sha256(observed_signature)
        or sidecar.get("generation_request_sha256")
        != _canonical_sha256(generation_request)
        or sidecar.get("artifact_semantic_identity_sha256")
        != _canonical_sha256(artifact_semantic_identity)
        or not isinstance(observed_artifact_semantics, Mapping)
        or dict(artifact_semantic_identity) != dict(observed_artifact_semantics)
        or sidecar.get("artifact_tree") != artifact_identity.get("tree")
    ):
        raise HarnessError(
            "artifact semantic identity or tree changed after generation; rerun "
            f"with --force: {artifact}"
        )
    if expected_signature is not None and (
        dict(generation_request) != dict(expected_signature)
        or sidecar.get("generation_request_sha256")
        != _canonical_sha256(expected_signature)
    ):
        raise HarnessError(
            "artifact semantic generation request changed; rerun with --force: "
            f"{artifact}"
        )
    return sidecar


def _require_reusable_artifact(
    artifact: Path,
    *,
    expected_signature: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    identity = _artifact_identity(artifact)
    sidecar = _validated_reuse_signature(
        artifact,
        artifact_identity=identity,
        expected_signature=expected_signature,
    )
    return identity, sidecar


def _artifact_identity_from_bound_semantics(
    artifact: Path,
    *,
    tree: Mapping[str, object],
    semantic_identity: Mapping[str, object],
) -> dict[str, object]:
    """Recover cheap identity fields after exact tree/sidecar authentication.

    The initial artifact bind validates every manifest payload and reconstructs
    the lane-local semantic identity.  Its sibling reuse sidecar content
    addresses that semantic identity and the complete artifact tree.  Once both
    byte identities have been pinned, later subprocesses can recover the
    immutable semantic body without hydrating a multi-gigabyte native plan.
    """

    manifest_path = artifact / "artifact.json"
    manifest = _json_object(manifest_path, label="artifact manifest")
    processes = manifest.get("processes")
    if not isinstance(processes, list) or len(processes) != 1:
        raise HarnessError(f"benchmark artifact must contain one process: {artifact}")
    process = processes[0]
    if not isinstance(process, Mapping):
        raise HarnessError(f"artifact process record is invalid: {artifact}")
    expression = process.get("expression")
    process_id = process.get("id")
    color_accuracy = process.get("color_accuracy")
    if not isinstance(expression, str) or not expression:
        raise HarnessError(f"artifact process expression is invalid: {artifact}")
    if not isinstance(process_id, str) or not process_id:
        raise HarnessError(f"artifact process ID is invalid: {artifact}")
    model_identity = _manifest_model_identity(manifest)
    bound_model_identity = semantic_identity.get("manifest_model_identity")
    if (
        not isinstance(bound_model_identity, Mapping)
        or dict(bound_model_identity) != model_identity
    ):
        raise HarnessError(
            f"artifact semantic model identity disagrees with its manifest: {artifact}"
        )
    semantic_identity_sha256 = _canonical_sha256(semantic_identity)
    return {
        "path": str(artifact.resolve()),
        "artifact_id": manifest.get("artifact_id"),
        "manifest": _path_identity(manifest_path),
        "tree": dict(tree),
        "process_id": process_id,
        "process_expression": expression,
        "color_accuracy": color_accuracy,
        "producer": manifest.get("producer"),
        "model_identity": model_identity,
        "semantic_identity": dict(semantic_identity),
        "semantic_identity_sha256": semantic_identity_sha256,
    }


def _require_bound_reusable_artifact(
    artifact: Path,
    *,
    expected_tree_sha256: str,
    expected_semantic_identity_sha256: str,
    expected_reuse_semantic_signature_sha256: str,
    expected_sidecar_sha256: str,
    expected_signature: Mapping[str, object] | None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Rebind an already authenticated artifact through exact byte identities.

    This is deliberately not the initial reuse path.  The caller must supply
    digests retained from a prior full semantic bind.  We hash the complete
    artifact tree and the sibling reuse sidecar again, validate the sidecar's
    self-addressed semantic/generation payload, and only then recover the
    semantic body stored in that sidecar.
    """

    expected_tree_sha256 = _required_sha256(
        expected_tree_sha256,
        label="expected bound artifact tree",
    )
    expected_semantic_identity_sha256 = _required_sha256(
        expected_semantic_identity_sha256,
        label="expected bound artifact semantic identity",
    )
    expected_reuse_semantic_signature_sha256 = _required_sha256(
        expected_reuse_semantic_signature_sha256,
        label="expected bound reuse semantic signature",
    )
    expected_sidecar_sha256 = _required_sha256(
        expected_sidecar_sha256,
        label="expected bound reuse sidecar",
    )
    sidecar_path = _reuse_signature_path(artifact)
    sidecar_identity_before = _path_identity(sidecar_path)
    if sidecar_identity_before.get("sha256") != expected_sidecar_sha256:
        raise HarnessError(
            f"artifact reuse sidecar changed after semantic binding: {artifact}"
        )
    sidecar_payload = _json_object(
        sidecar_path,
        label="artifact reuse signature",
    )
    raw_signature = sidecar_payload.get("semantic_signature")
    semantic_identity = (
        raw_signature.get("artifact_semantic_identity")
        if isinstance(raw_signature, Mapping)
        else None
    )
    if not isinstance(semantic_identity, Mapping):
        raise HarnessError(
            f"artifact reuse signature has no bound semantic identity: {sidecar_path}"
        )
    tree = _tree_identity(artifact)
    if tree.get("sha256") != expected_tree_sha256:
        raise HarnessError(f"artifact tree changed after semantic binding: {artifact}")
    identity = _artifact_identity_from_bound_semantics(
        artifact,
        tree=tree,
        semantic_identity=semantic_identity,
    )
    sidecar = _validated_reuse_signature(
        artifact,
        artifact_identity=identity,
        expected_signature=expected_signature,
    )
    if identity.get("semantic_identity_sha256") != expected_semantic_identity_sha256:
        raise HarnessError(
            f"artifact semantic identity changed after semantic binding: {artifact}"
        )
    if (
        sidecar.get("semantic_signature_sha256")
        != expected_reuse_semantic_signature_sha256
    ):
        raise HarnessError(
            f"artifact reuse semantics changed after semantic binding: {artifact}"
        )
    sidecar_identity_after = _path_identity(sidecar_path)
    if (
        sidecar_identity_after != sidecar_identity_before
        or sidecar_identity_after.get("sha256") != expected_sidecar_sha256
    ):
        raise HarnessError(
            f"artifact reuse sidecar drifted during semantic binding: {artifact}"
        )
    return identity, sidecar, sidecar_identity_after


_PROFILE_EXPECTATION_FIELDS = (
    "source_identity_sha256",
    "runtime_provenance_sha256",
    "interpreter_sha256",
    "native_extension_sha256",
    "artifact_id",
    "artifact_tree_sha256",
    "artifact_semantic_identity_sha256",
    "reuse_semantic_signature_sha256",
    "reuse_sidecar_sha256",
)


def _required_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise HarnessError(f"{label} is not a SHA-256 identity")
    return value


def _profile_identity_expectations(
    source_identity: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
    artifact_identity: Mapping[str, object],
    reuse_signature: Mapping[str, object],
    *,
    reuse_sidecar: Path,
) -> dict[str, str]:
    interpreter = runtime_provenance.get("interpreter")
    native = runtime_provenance.get("native_extension")
    tree = artifact_identity.get("tree")
    if (
        not isinstance(interpreter, Mapping)
        or not isinstance(native, Mapping)
        or not isinstance(tree, Mapping)
    ):
        raise HarnessError("driver cannot construct complete worker expectations")
    return {
        "source_identity_sha256": _canonical_sha256(source_identity),
        "runtime_provenance_sha256": _canonical_sha256(runtime_provenance),
        "interpreter_sha256": _required_sha256(
            interpreter.get("sha256"),
            label="interpreter identity",
        ),
        "native_extension_sha256": _required_sha256(
            native.get("sha256"),
            label="native extension identity",
        ),
        "artifact_id": _required_sha256(
            artifact_identity.get("artifact_id"),
            label="artifact manifest identity",
        ),
        "artifact_tree_sha256": _required_sha256(
            tree.get("sha256"),
            label="artifact tree identity",
        ),
        "artifact_semantic_identity_sha256": _required_sha256(
            artifact_identity.get("semantic_identity_sha256"),
            label="artifact semantic identity",
        ),
        "reuse_semantic_signature_sha256": _required_sha256(
            reuse_signature.get("semantic_signature_sha256"),
            label="artifact reuse semantic signature",
        ),
        "reuse_sidecar_sha256": _required_sha256(
            _path_identity(reuse_sidecar).get("sha256"),
            label="artifact reuse sidecar",
        ),
    }


def _validate_profile_worker_expectations(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
) -> None:
    for field in _PROFILE_EXPECTATION_FIELDS:
        expected_value = _required_sha256(
            expected.get(field),
            label=f"expected {field}",
        )
        observed_value = _required_sha256(
            observed.get(field),
            label=f"observed {field}",
        )
        if observed_value != expected_value:
            raise HarnessError(
                f"profile worker {field.replace('_', ' ')} drifted before timing"
            )


def _profile_expectation_arguments(
    expectations: Mapping[str, object],
) -> tuple[str, ...]:
    result: list[str] = []
    for field in _PROFILE_EXPECTATION_FIELDS:
        value = _required_sha256(
            expectations.get(field),
            label=f"expected {field}",
        )
        result.extend((f"--expected-{field.replace('_', '-')}", value))
    return tuple(result)


def _profile_worker_expectations_from_arguments(
    arguments: argparse.Namespace,
) -> dict[str, str]:
    return {
        field: _required_sha256(
            getattr(arguments, f"expected_{field}", None),
            label=f"profile worker expected {field}",
        )
        for field in _PROFILE_EXPECTATION_FIELDS
    }


def _verify_profile_worker_environment(
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """Recompute every byte identity inside the worker before loading.

    Native artifact semantics were fully authenticated by the driver before it
    scheduled any timing worker.  The worker rebinds that semantic body through
    the exact artifact-tree and reuse-sidecar digests supplied by the driver,
    avoiding a redundant native-plan hydration before every timed load.
    """

    expected = _profile_worker_expectations_from_arguments(arguments)
    source_identity = _git_source_identity()
    runtime_provenance = _runtime_provenance(source_identity)
    artifact = arguments.artifact.resolve(strict=True)
    artifact_identity, reuse_signature, reuse_sidecar_identity = (
        _require_bound_reusable_artifact(
            artifact,
            expected_tree_sha256=expected["artifact_tree_sha256"],
            expected_semantic_identity_sha256=expected[
                "artifact_semantic_identity_sha256"
            ],
            expected_reuse_semantic_signature_sha256=expected[
                "reuse_semantic_signature_sha256"
            ],
            expected_sidecar_sha256=expected["reuse_sidecar_sha256"],
            expected_signature=None,
        )
    )
    interpreter = runtime_provenance.get("interpreter")
    native = runtime_provenance.get("native_extension")
    tree = artifact_identity.get("tree")
    if (
        not isinstance(interpreter, Mapping)
        or not isinstance(native, Mapping)
        or not isinstance(tree, Mapping)
    ):
        raise HarnessError("profile worker observed incomplete runtime identities")
    observed = {
        "source_identity_sha256": _canonical_sha256(source_identity),
        "runtime_provenance_sha256": _canonical_sha256(runtime_provenance),
        "interpreter_sha256": interpreter.get("sha256"),
        "native_extension_sha256": native.get("sha256"),
        "artifact_id": artifact_identity.get("artifact_id"),
        "artifact_tree_sha256": tree.get("sha256"),
        "artifact_semantic_identity_sha256": artifact_identity.get(
            "semantic_identity_sha256"
        ),
        "reuse_semantic_signature_sha256": reuse_signature.get(
            "semantic_signature_sha256"
        ),
        "reuse_sidecar_sha256": reuse_sidecar_identity.get("sha256"),
    }
    _validate_profile_worker_expectations(expected, observed)
    effective_contract = _validate_artifact_contract(
        artifact,
        artifact_identity,
        arguments=arguments,
        mode=arguments.mode,
    )
    return {
        "kind": WORKER_VERIFICATION_KIND,
        "schema_version": WORKER_VERIFICATION_SCHEMA,
        "verified_at_utc": _utc_now(),
        "expected": expected,
        "observed": observed,
        "artifact_semantic_identity": artifact_identity["semantic_identity"],
        "artifact_semantic_identity_sha256": artifact_identity[
            "semantic_identity_sha256"
        ],
        "effective_contract": effective_contract,
    }


def _loaded_runtime_artifact_verification(
    runtime: object,
    *,
    expected_artifact_id: object,
    phase: str,
) -> dict[str, object]:
    """Bind an in-memory runtime to the manifest identity it authenticated.

    The native loader validates the manifest identity, the exact declared tree,
    and every payload before it publishes the runtime.  Comparing that retained
    identity closes the path replacement window between the worker's fast tree
    rebind and ``Runtime.load`` without rehydrating the eager semantic plan.
    """

    expected = _required_sha256(
        expected_artifact_id,
        label="expected loaded artifact identity",
    )
    observed = _required_sha256(
        getattr(runtime, "artifact_id", None),
        label="native loaded artifact identity",
    )
    if observed != expected:
        raise HarnessError(
            "profile worker loaded a different artifact after its pre-load "
            "identity check"
        )
    return {
        "kind": "pyamplicol-loaded-runtime-artifact-verification",
        "schema_version": 1,
        "phase": phase,
        "checked_at_utc": _utc_now(),
        "expected_artifact_id": expected,
        "loaded_artifact_id": observed,
        "passes": True,
    }


def _validate_loaded_runtime_artifact_verification(
    value: object,
    *,
    expected_artifact_id: object,
    phase: str,
) -> None:
    expected = _required_sha256(
        expected_artifact_id,
        label="expected retained loaded artifact identity",
    )
    if not isinstance(value, Mapping):
        raise HarnessError("loaded runtime artifact verification is missing")
    if (
        value.get("kind") != "pyamplicol-loaded-runtime-artifact-verification"
        or not _is_exact_int(value.get("schema_version"), 1)
        or value.get("phase") != phase
        or not _is_utc_timestamp(value.get("checked_at_utc"))
        or value.get("expected_artifact_id") != expected
        or value.get("loaded_artifact_id") != expected
        or value.get("passes") is not True
    ):
        raise HarnessError("loaded runtime artifact verification is invalid")


def _artifact_phases(path: Path) -> dict[str, float]:
    try:
        artifact = json.loads((path / "artifact.json").read_text(encoding="utf-8"))
        raw = artifact["extensions"]["generation"]["phase_timings_seconds"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise HarnessError(
            f"artifact has no valid generation timings: {path}"
        ) from error
    if not isinstance(raw, Mapping):
        raise HarnessError(f"artifact generation timings are not an object: {path}")
    result: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or isinstance(value, bool):
            raise HarnessError(f"artifact has an invalid generation phase: {path}")
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0:
            raise HarnessError(f"artifact has an invalid phase duration: {path}")
        result[name] = seconds
    return result


def _effective_contract(path: Path) -> dict[str, object]:
    try:
        payload = tomllib.loads(
            (path / "config" / "effective.toml").read_text(encoding="utf-8")
        )
        color = payload["color"]
        evaluator = payload["evaluator"]
        jit = evaluator["jit"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise HarnessError(
            f"artifact has no valid effective configuration: {path}"
        ) from error
    return {
        "execution_mode": str(evaluator["execution_mode"]),
        "backend": str(evaluator["backend"]),
        "jit_optimization_level": int(jit["optimization_level"]),
        "color_accuracy": str(color["accuracy"]),
        "lc_flow_layout": str(color["lc_flow_layout"]),
    }


def _validation_fixture(
    artifact: Path,
    process_id: str,
) -> tuple[
    tuple[tuple[tuple[float, ...], ...], ...],
    dict[str, object],
]:
    path = artifact / "processes" / process_id / "validation-momenta.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        points = payload["points"]
        converted = tuple(
            tuple(
                tuple(float(component) for component in particle["momentum"])
                for particle in point
            )
            for point in points
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise HarnessError(f"invalid validation momenta at {path}") from error
    return (
        converted,
        {
            "file": _path_identity(path),
            "point_count": len(converted),
            "points_sha256": _canonical_sha256(converted),
        },
    )


def _complex_payload(value: complex) -> list[float]:
    return [value.real, value.imag]


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(entry) for key, entry in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(entry) for entry in value]
    return value


def _statistics_payload(value: Any) -> dict[str, float]:
    return {
        "standard_deviation_seconds_per_point": float(value.standard_deviation),
        "standard_error_seconds_per_point": float(value.standard_error),
        "relative_standard_error": float(value.relative_standard_error),
    }


def _benchmark_payload(result: Any) -> dict[str, object]:
    evaluator_uncertainty = result.evaluator_uncertainty
    timing_breakdown = getattr(result, "timing_breakdown", None)
    return {
        "batch_size": int(result.effective_config.batch_size),
        "sample_count": int(result.sample_count),
        "repetitions_per_sample": int(result.repetitions_per_sample),
        "evaluation_count": int(result.evaluation_count),
        "evaluated_point_count": int(result.evaluated_point_count),
        "wall_seconds_per_point": float(result.wall_time_per_point),
        "evaluator_seconds_per_point": (
            None
            if result.evaluator_time_per_point is None
            else float(result.evaluator_time_per_point)
        ),
        "wall_uncertainty": _statistics_payload(result.uncertainty),
        "evaluator_uncertainty": (
            None
            if evaluator_uncertainty is None
            else _statistics_payload(evaluator_uncertainty)
        ),
        "timing_breakdown": (
            None
            if timing_breakdown is None
            else _plain(dataclasses.asdict(timing_breakdown))
        ),
        "timing_sources": {
            "wall": result.environment.get("wall_time_source"),
            "evaluator": result.environment.get("evaluator_time_source"),
        },
        "environment": _plain(result.environment),
        "interrupted": bool(result.interrupted),
    }


def _native_wall_block_payload(
    runtime: object,
    validation_points: Sequence[object],
    *,
    batch_size: int,
    repetitions_per_block: int,
    block_count: int,
    minimum_native_wall_seconds: float,
    calibration: Mapping[str, object],
    selectors: Mapping[str, object],
    fixture_points_sha256: str,
) -> dict[str, object]:
    if (
        isinstance(repetitions_per_block, bool)
        or not isinstance(repetitions_per_block, int)
        or repetitions_per_block <= 0
        or isinstance(block_count, bool)
        or not isinstance(block_count, int)
        or block_count < MIN_AUTHORITATIVE_SAMPLES
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or not validation_points
        or _SHA256_PATTERN.fullmatch(fixture_points_sha256) is None
        or isinstance(minimum_native_wall_seconds, bool)
        or not isinstance(minimum_native_wall_seconds, (float, int))
        or not math.isfinite(float(minimum_native_wall_seconds))
        or float(minimum_native_wall_seconds) <= 0.0
    ):
        raise HarnessError("raw native-wall block configuration is invalid")
    backend = getattr(runtime, "_backend", None)
    timer = getattr(backend, "_benchmark_f64_wall_time", None)
    if not callable(timer):
        raise HarnessError("runtime exposes no raw native-wall block timer")
    batch = tuple(
        validation_points[index % len(validation_points)] for index in range(batch_size)
    )
    blocks: list[dict[str, object]] = []
    seconds_per_point: list[float] = []
    observed_native_wall_seconds = 0.0
    observed_caller_elapsed_seconds = 0.0
    maximum_block_count = max(block_count * 4, block_count + 16)
    while (
        len(blocks) < block_count
        or observed_native_wall_seconds < float(minimum_native_wall_seconds)
    ):
        if len(blocks) >= maximum_block_count:
            raise HarnessError(
                "raw native-wall calibration could not satisfy the configured "
                "minimum duration"
            )
        block_index = len(blocks)
        started_at = _utc_now()
        started = time.perf_counter()
        duration = timer(
            batch,
            repetitions_per_block,
            precision=16,
            **dict(selectors),
        )
        finished_at = _utc_now()
        elapsed = time.perf_counter() - started
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (float, int))
            or not math.isfinite(float(duration))
            or float(duration) <= 0.0
        ):
            raise HarnessError("native-wall timer returned an invalid raw block")
        per_point = float(duration) / (repetitions_per_block * batch_size)
        record = {
            "block_index": block_index,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "caller_elapsed_seconds": elapsed,
            "native_wall_seconds": float(duration),
            "wall_seconds_per_point": per_point,
            "repetitions": repetitions_per_block,
            "batch_size": batch_size,
            "evaluation_count": repetitions_per_block,
            "evaluated_point_count": repetitions_per_block * batch_size,
        }
        record["content_sha256"] = _canonical_sha256(record)
        blocks.append(record)
        seconds_per_point.append(per_point)
        observed_native_wall_seconds += float(duration)
        observed_caller_elapsed_seconds += elapsed
    median = statistics.median(seconds_per_point)
    mad = statistics.median(abs(value - median) for value in seconds_per_point)
    return {
        "kind": RAW_NATIVE_WALL_BLOCK_KIND,
        "schema_version": RAW_NATIVE_WALL_BLOCK_SCHEMA,
        "measurement_contract": RAW_NATIVE_WALL_MEASUREMENT_CONTRACT,
        "source": "runtime._benchmark_f64_wall_time",
        "fixture_points_sha256": fixture_points_sha256,
        "minimum_native_wall_seconds": float(minimum_native_wall_seconds),
        "observed_native_wall_seconds": observed_native_wall_seconds,
        "observed_caller_elapsed_seconds": observed_caller_elapsed_seconds,
        "minimum_duration_satisfied": (
            observed_native_wall_seconds >= float(minimum_native_wall_seconds)
        ),
        "calibration": dict(calibration),
        "block_count": len(blocks),
        "repetitions_per_block": repetitions_per_block,
        "evaluation_count": len(blocks) * repetitions_per_block,
        "evaluated_point_count": (len(blocks) * repetitions_per_block * batch_size),
        "wall_seconds_per_point_median": median,
        "wall_seconds_per_point_mad": mad,
        "blocks": blocks,
        "blocks_sha256": _canonical_sha256(blocks),
    }


def _reference_color_order(gluon_count: int) -> tuple[int, ...]:
    return (2, *range(4, 4 + gluon_count), 1, 3)


def _parse_flow_word(requested: str) -> tuple[int, ...] | None:
    if not requested.startswith("flow:"):
        return None
    raw = requested.removeprefix("flow:")
    if not raw:
        raise HarnessError("flow ID must contain at least one label")
    try:
        word = tuple(int(label) for label in raw.split(","))
    except ValueError as error:
        raise HarnessError(f"invalid flow ID {requested!r}") from error
    if not word or any(label <= 0 for label in word):
        raise HarnessError(f"invalid flow ID {requested!r}")
    return word


def _generation_selected_flow_word(
    arguments: argparse.Namespace,
) -> tuple[int, ...] | None:
    if not arguments.specialize_flow_at_generation:
        return None
    if arguments.lc_flow_layout != "topology-replay":
        raise HarnessError(
            "generation-time flow specialization is available only for topology-replay"
        )
    parsed = _parse_flow_word(arguments.color_flow)
    if parsed is not None:
        return parsed
    try:
        ordinal = int(arguments.color_flow, 10)
    except ValueError as error:
        raise HarnessError(
            "generation-time flow specialization requires a stable flow ID or "
            "the first flow ordinal"
        ) from error
    if ordinal != 1:
        raise HarnessError(
            "generation-time flow specialization currently supports ordinal 1 "
            "or an explicit stable flow ID"
        )
    if arguments.process_expression is not None:
        raise HarnessError(
            "generation-time flow specialization for custom processes requires "
            "an explicit stable flow ID"
        )
    return _reference_color_order(arguments.gluon_count)


def _generation_config(
    execution_mode: str,
    *,
    validation_samples: int,
    lc_flow_layout: str,
    point_tile_size: int,
    jit_optimization_level: int,
    gluon_count: int | None = None,
) -> Any:
    from pyamplicol.config import (
        Action,
        ColorAccuracy,
        ColorConfig,
        EvaluatorBackend,
        EvaluatorConfig,
        EvaluatorExecutionMode,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        GenerationValidationConfig,
        JITConfig,
        LCFlowLayout,
        ProcessConfig,
        RecurrenceEvaluatorConfig,
        RunConfig,
    )

    if execution_mode not in {"compiled", "eager", "recurrence"}:
        raise HarnessError(f"unsupported generation mode {execution_mode!r}")
    if lc_flow_layout not in {"topology-replay", "all-flow-union"}:
        raise HarnessError(f"unsupported LC flow layout {lc_flow_layout!r}")
    return RunConfig(
        action=Action.GENERATE,
        color=ColorConfig(
            accuracy=ColorAccuracy.LC,
            lc_flow_layout=LCFlowLayout(lc_flow_layout),
        ),
        process=ProcessConfig(
            reference_color_order=(
                () if gluon_count is None else _reference_color_order(gluon_count)
            ),
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=True,
                samples=validation_samples,
                seed=VALIDATION_SEED,
                relative_tolerance=RTOL,
                absolute_tolerance=ATOL,
                post_build_validation=True,
            ),
        ),
        evaluator=EvaluatorConfig(
            backend=EvaluatorBackend.JIT,
            execution_mode=EvaluatorExecutionMode(execution_mode),
            batch_size=64,
            output_chunk_size=512,
            optimization=EvaluatorOptimizationConfig(
                horner_iterations=10,
                cores=1,
                max_horner_variables=1000,
                max_common_pair_cache_entries=5_000_000,
                max_common_pair_distance=1000,
                collect_factors="auto",
            ),
            # Recurrence process generation consumes its prepared pack without
            # compiling. Compiled mode performs process-specific JIT work in
            # the measured generation interval.
            jit=JITConfig(
                optimization_level=cast(
                    Literal[0, 1, 2, 3],
                    jit_optimization_level,
                )
            ),
            recurrence=RecurrenceEvaluatorConfig(
                point_tile_size=point_tile_size,
            ),
        ),
    )


def _generate_worker(arguments: argparse.Namespace) -> dict[str, object]:
    from pyamplicol import Generator, ModelSource
    from pyamplicol.assets.prepared_models import (
        BUILTIN_SM_JIT_O2,
        packaged_prepared_model_path,
    )
    from pyamplicol.reporting import StreamProgressSink

    artifact = arguments.artifact.resolve()
    prepared_access_started = time.perf_counter()
    prepared_context = (
        contextlib.nullcontext(arguments.prepared_model.resolve())
        if arguments.prepared_model is not None
        else (
            contextlib.nullcontext(None)
            if arguments.mode == "compiled"
            else packaged_prepared_model_path(BUILTIN_SM_JIT_O2)
        )
    )
    try:
        with prepared_context as prepared_model:
            if prepared_model is None:
                model_source = ModelSource.built_in_sm()
                model_record: dict[str, object] = {
                    "kind": "built-in-sm-source",
                    "resource_id": None,
                    "compile_excluded_from_generation": False,
                }
            else:
                if not prepared_model.is_file():
                    raise HarnessError(
                        f"prepared model does not exist: {prepared_model}"
                    )
                model_source = ModelSource.from_path(prepared_model)
                prepared_identity = _path_identity(prepared_model)
                model_record = {
                    "kind": (
                        "packaged-prepared-model"
                        if arguments.prepared_model is None
                        else "explicit-prepared-model"
                    ),
                    "resource_id": (
                        PREPARED_MODEL_ID if arguments.prepared_model is None else None
                    ),
                    "file": (
                        None if arguments.prepared_model is None else prepared_identity
                    ),
                    "size_bytes": prepared_identity["size_bytes"],
                    "sha256": prepared_identity["sha256"],
                    "compile_excluded_from_generation": True,
                }
            prepared_access_seconds = time.perf_counter() - prepared_access_started
            generation_started = time.perf_counter()
            config = _generation_config(
                arguments.mode,
                validation_samples=arguments.validation_samples,
                lc_flow_layout=arguments.lc_flow_layout,
                point_tile_size=arguments.point_tile_size,
                jit_optimization_level=arguments.jit_optimization_level,
                gluon_count=(
                    arguments.gluon_count
                    if arguments.process_expression is None
                    else None
                ),
            )
            selected_flow_word = _generation_selected_flow_word(arguments)
            if selected_flow_word is not None:
                config = dataclasses.replace(
                    config,
                    process=dataclasses.replace(
                        config.process,
                        reference_color_order=selected_flow_word,
                        selected_color_sector_ids=(0,),
                    ),
                )
            Generator(
                config,
                progress=StreamProgressSink(sys.stderr),
            ).generate(
                _selected_process(arguments),
                artifact,
                model=model_source,
                mode=arguments.write_mode,
            )
    except Exception as error:
        if arguments.mode == "recurrence":
            raise HarnessError(
                "recurrence generation is unavailable or failed; install a native "
                "build with recurrence support and a current built-in prepared "
                f"model pack: {error}"
            ) from error
        raise
    return {
        "mode": arguments.mode,
        "generation_wall_seconds": time.perf_counter() - generation_started,
        "generation_reused": False,
        "peak_rss": _resource_peak(),
        "specialized_flow_word": (
            None
            if selected_flow_word is None
            else list(int(label) for label in selected_flow_word)
        ),
        "model_source": {
            **model_record,
            "access_seconds": prepared_access_seconds,
        },
    }


def _profile_worker(arguments: argparse.Namespace) -> dict[str, object]:
    from pyamplicol import BenchmarkRunner, Runtime
    from pyamplicol.config import BenchmarkConfig

    timing_profile = getattr(arguments, "operation", "profile") == "profile"
    if timing_profile and len(arguments.batch_size) != 1:
        raise HarnessError(
            "profile timing workers require exactly one batch size per subprocess"
        )
    if not timing_profile and arguments.batch_size:
        raise HarnessError("preflight workers do not accept timing batch sizes")
    pre_timing_verification = _verify_profile_worker_environment(arguments)
    artifact = arguments.artifact.resolve()
    load_started = time.perf_counter()
    try:
        process = _selected_process(arguments)
        runtime = Runtime.load(artifact, process=process)
    except Exception as error:
        if arguments.mode == "recurrence":
            raise HarnessError(
                "the installed runtime cannot load the recurrence artifact; "
                f"rebuild pyAmpliCol with recurrence support: {error}"
            ) from error
        raise
    loaded_artifact_before_timing = _loaded_runtime_artifact_verification(
        runtime,
        expected_artifact_id=pre_timing_verification["expected"]["artifact_id"],
        phase="after-native-load-before-timing",
    )
    pre_timing_verification["loaded_runtime_artifact"] = loaded_artifact_before_timing
    cold_load_seconds = time.perf_counter() - load_started
    peak_after_load = _resource_peak()
    physics = runtime.physics
    if physics.process.casefold() != process.casefold():
        raise HarnessError(
            f"artifact resolved process {physics.process!r}, expected {process!r}"
        )
    if physics.color_accuracy != "lc" or not physics.color_flows:
        raise HarnessError("benchmark artifact does not expose physical LC flows")

    def resolve_axis(
        requested: str,
        available_ids: tuple[str, ...],
        *,
        label: str,
    ) -> str:
        try:
            ordinal = int(requested, 10)
        except ValueError:
            ordinal = 0
        if ordinal:
            if ordinal < 1 or ordinal > len(available_ids):
                raise HarnessError(
                    f"{label} ordinal is outside 1..{len(available_ids)}"
                )
            return available_ids[ordinal - 1]
        if requested not in set(available_ids):
            raise HarnessError(
                f"unknown {label} {requested!r}; artifact exposes "
                f"{len(available_ids)} values"
            )
        return requested

    color_flow_id = resolve_axis(
        arguments.color_flow,
        tuple(flow.id for flow in physics.color_flows),
        label="color flow",
    )
    helicity_id = resolve_axis(
        arguments.helicity,
        tuple(helicity.id for helicity in physics.helicities),
        label="helicity",
    )
    union_workload = arguments.lc_flow_layout == "all-flow-union"
    selectors: dict[str, Any] = (
        {"helicities": (helicity_id,)}
        if union_workload
        else {"color_flows": (color_flow_id,)}
    )

    validation_point_artifact = arguments.validation_point_artifact.resolve()
    validation_points, validation_fixture = _validation_fixture(
        validation_point_artifact,
        physics.process_id,
    )
    if not validation_points:
        raise HarnessError("artifact contains no deterministic validation point")
    selected_totals = tuple(
        complex(value) for value in runtime.evaluate(validation_points, **selectors)
    )
    selected_resolved = runtime.evaluate_resolved(validation_points, **selectors)
    resolved_totals = tuple(complex(value) for value in selected_resolved.total())
    resolved_components = [
        [
            _complex_payload(complex(value))
            for helicity_row in point_values
            for value in helicity_row
        ]
        for point_values in selected_resolved.values
    ]
    if len(selected_totals) != len(validation_points) or len(resolved_totals) != len(
        validation_points
    ):
        raise HarnessError("runtime validation result count does not match fixture")
    point_comparisons: list[dict[str, object]] = []
    for index, (selected_total, resolved_total) in enumerate(
        zip(selected_totals, resolved_totals, strict=True)
    ):
        absolute = abs(selected_total - resolved_total)
        relative = absolute / max(abs(selected_total), abs(resolved_total), 1.0e-300)
        point_comparisons.append(
            {
                "point_index": index,
                "selected_total": _complex_payload(selected_total),
                "resolved_sum": _complex_payload(resolved_total),
                "absolute_difference": absolute,
                "relative_difference": relative,
                "passes": _within_numerical_tolerance(
                    selected_total,
                    resolved_total,
                ),
            }
        )

    profiles: list[dict[str, object]] = []
    for batch_size in arguments.batch_size if timing_profile else ():
        try:
            result = BenchmarkRunner(
                BenchmarkConfig(
                    target_runtime=arguments.target_runtime,
                    batch_size=batch_size,
                    precision=16,
                    warmup_runs=arguments.warmup_runs,
                    minimum_samples=arguments.minimum_samples,
                    color_flow_ids=(() if union_workload else (color_flow_id,)),
                    helicity_ids=((helicity_id,) if union_workload else ()),
                )
            ).run(runtime, points=validation_points)
        except Exception as error:
            if arguments.mode == "recurrence":
                raise HarnessError(
                    "the installed runtime cannot profile recurrence execution at "
                    f"batch {batch_size}: {error}"
                ) from error
            raise
        if (
            isinstance(result.effective_config.batch_size, bool)
            or not isinstance(result.effective_config.batch_size, int)
            or result.effective_config.batch_size != batch_size
            or isinstance(result.sample_count, bool)
            or not isinstance(result.sample_count, int)
            or result.sample_count <= 0
            or isinstance(result.repetitions_per_sample, bool)
            or not isinstance(result.repetitions_per_sample, int)
            or result.repetitions_per_sample <= 0
            or isinstance(result.evaluation_count, bool)
            or not isinstance(result.evaluation_count, int)
            or result.evaluation_count <= 0
            or isinstance(result.evaluated_point_count, bool)
            or not isinstance(result.evaluated_point_count, int)
            or result.evaluated_point_count <= 0
            or result.evaluation_count
            != result.sample_count * result.repetitions_per_sample
            or result.evaluated_point_count != result.evaluation_count * batch_size
        ):
            raise HarnessError("benchmark runner returned invalid timing counts")
        minimum_block_count = max(
            MIN_AUTHORITATIVE_SAMPLES,
            arguments.minimum_samples,
        )
        estimated_seconds_per_repetition = (
            float(result.wall_time_per_point) * batch_size
        )
        if (
            not math.isfinite(estimated_seconds_per_repetition)
            or estimated_seconds_per_repetition <= 0.0
        ):
            raise HarnessError(
                "benchmark runner returned an invalid raw-wall calibration"
            )
        repetitions_per_raw_block = max(
            result.repetitions_per_sample,
            math.ceil(
                (arguments.target_runtime * 1.02)
                / (minimum_block_count * estimated_seconds_per_repetition)
            ),
        )
        calibration = {
            "kind": "benchmark-runner-wall-rate-calibration",
            "schema_version": 1,
            "benchmark_runner_sample_count": result.sample_count,
            "benchmark_runner_repetitions_per_sample": (
                result.repetitions_per_sample
            ),
            "benchmark_runner_total_repetitions": (
                result.sample_count * result.repetitions_per_sample
            ),
            "benchmark_runner_wall_seconds_per_point": (
                float(result.wall_time_per_point)
            ),
            "requested_minimum_block_count": minimum_block_count,
            "preceded_by_benchmark_runner_warmup_runs": (
                arguments.warmup_runs
            ),
            "duration_headroom_factor": 1.02,
            "scaled_repetitions_per_block": repetitions_per_raw_block,
        }
        raw_blocks = _native_wall_block_payload(
            runtime,
            validation_points,
            batch_size=batch_size,
            repetitions_per_block=repetitions_per_raw_block,
            block_count=minimum_block_count,
            minimum_native_wall_seconds=arguments.target_runtime,
            calibration=calibration,
            selectors=selectors,
            fixture_points_sha256=str(validation_fixture["points_sha256"]),
        )
        measurement = _benchmark_payload(result)
        for field in (
            "wall_seconds_per_point",
            "wall_uncertainty",
            "sample_count",
            "repetitions_per_sample",
            "evaluation_count",
            "evaluated_point_count",
        ):
            measurement[f"benchmark_runner_{field}"] = measurement[field]
        measurement["wall_seconds_per_point"] = raw_blocks[
            "wall_seconds_per_point_median"
        ]
        measurement["wall_uncertainty"] = {
            "statistics_contract": "median-and-raw-mad-v1",
            "raw_mad_seconds_per_point": raw_blocks["wall_seconds_per_point_mad"],
        }
        measurement["sample_count"] = raw_blocks["block_count"]
        measurement["repetitions_per_sample"] = raw_blocks[
            "repetitions_per_block"
        ]
        measurement["evaluation_count"] = raw_blocks["evaluation_count"]
        measurement["evaluated_point_count"] = raw_blocks["evaluated_point_count"]
        measurement["inner_native_wall_blocks"] = raw_blocks
        profiles.append(measurement)

    loaded_artifact_after_timing = _loaded_runtime_artifact_verification(
        runtime,
        expected_artifact_id=pre_timing_verification["expected"]["artifact_id"],
        phase=("after-timing" if timing_profile else "after-validation-preflight"),
    )

    return {
        "mode": arguments.mode,
        "schedule_index": arguments.schedule_index,
        "schedule_round": arguments.schedule_round,
        "pre_timing_verification": pre_timing_verification,
        "post_timing_loaded_runtime_artifact": loaded_artifact_after_timing,
        "timing_configuration": {
            "minimum_internal_samples": arguments.minimum_samples,
            "warmup_runs": arguments.warmup_runs,
            "target_runtime_seconds": arguments.target_runtime,
        },
        "cold_load_seconds": cold_load_seconds,
        "peak_rss_after_cold_load": peak_after_load,
        "peak_rss_after_profile": _resource_peak(),
        "process_id": physics.process_id,
        "process_expression": physics.process,
        "selector_contract": {
            "color_flow_request": arguments.color_flow,
            "resolved_color_flow_id": (None if union_workload else color_flow_id),
            "helicity_request": arguments.helicity,
            "resolved_helicity_id": helicity_id if union_workload else None,
            "color_flow_count": len(physics.color_flows),
            "helicity_count": len(physics.helicities),
            "structural_zero_helicity_count": (physics.structural_zero_helicity_count),
            "workload": (
                "all-flows/runtime-selected-single-helicity"
                if union_workload
                else "single-runtime-selected-flow/helicity-sum"
            ),
        },
        "validation": {
            "point_source_artifact": str(validation_point_artifact),
            "fixture": validation_fixture,
            "selected_totals": [_complex_payload(value) for value in selected_totals],
            "resolved_sums": [_complex_payload(value) for value in resolved_totals],
            "resolved_helicity_ids": list(selected_resolved.helicity_ids),
            "resolved_color_ids": list(selected_resolved.color_ids),
            "resolved_components": resolved_components,
            "point_comparisons": point_comparisons,
            "maximum_absolute_difference": max(
                abs(selected - resolved)
                for selected, resolved in zip(
                    selected_totals,
                    resolved_totals,
                    strict=True,
                )
            ),
            "maximum_relative_difference": max(
                abs(selected - resolved) / max(abs(selected), abs(resolved), 1.0e-300)
                for selected, resolved in zip(
                    selected_totals,
                    resolved_totals,
                    strict=True,
                )
            ),
            "passes": all(bool(item["passes"]) for item in point_comparisons),
        },
        "profiles": profiles,
    }


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("operation", choices=("generate", "preflight", "profile"))
    parser.add_argument(
        "--mode",
        choices=("compiled", "eager", "recurrence"),
        required=True,
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--validation-point-artifact", type=Path)
    parser.add_argument("--write-mode", choices=("error", "replace"), default="error")
    parser.add_argument("--batch-size", type=_positive_int, action="append", default=[])
    parser.add_argument("--schedule-index", type=int)
    parser.add_argument("--schedule-round", type=int)
    for field in _PROFILE_EXPECTATION_FIELDS:
        parser.add_argument(f"--expected-{field.replace('_', '-')}")
    parser.add_argument("--target-runtime", type=_positive_float, default=5.0)
    parser.add_argument("--minimum-samples", type=_positive_int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--color-flow", default="1")
    parser.add_argument("--helicity", default="1")
    parser.add_argument(
        "--lc-flow-layout",
        choices=("topology-replay", "all-flow-union"),
        default="topology-replay",
    )
    parser.add_argument("--gluon-count", type=_positive_int, default=6)
    parser.add_argument("--process-expression")
    parser.add_argument("--validation-samples", type=_positive_int, default=10)
    parser.add_argument("--point-tile-size", type=_positive_int, default=1024)
    parser.add_argument(
        "--jit-optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=2,
    )
    parser.add_argument("--prepared-model", type=Path)
    parser.add_argument(
        "--specialize-flow-at-generation",
        action="store_true",
    )
    return parser


def _worker_main(argv: Sequence[str]) -> int:
    started_at = _utc_now()
    started = time.perf_counter()
    arguments = _worker_parser().parse_args(argv)
    if arguments.warmup_runs < 0:
        raise HarnessError("warmup runs must be non-negative")
    if (
        arguments.operation in {"preflight", "profile"}
        and arguments.validation_point_artifact is None
    ):
        raise HarnessError(
            "validation workers require an explicit --validation-point-artifact"
        )
    if arguments.operation == "profile" and (
        arguments.schedule_index is None
        or arguments.schedule_index < 0
        or arguments.schedule_round is None
        or arguments.schedule_round < 0
    ):
        raise HarnessError("profile workers require non-negative schedule coordinates")
    operation = (
        _generate_worker if arguments.operation == "generate" else _profile_worker
    )
    payload = operation(arguments)
    process_record = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "process_id": os.getpid(),
        "operation": arguments.operation,
        "mode": arguments.mode,
        "payload_sha256": _canonical_sha256(payload),
    }
    process_record["content_sha256"] = _canonical_sha256(process_record)
    payload["worker_process_record"] = process_record
    print(_WORKER_MARKER + json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


def _run_worker(
    arguments: Sequence[str],
    *,
    mode: str,
    phase: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    command = (sys.executable, str(DRIVER_PATH), "_worker", *arguments)
    expected_operation = arguments[0] if arguments else None
    environment = os.environ.copy()
    environment.setdefault("SYMBOLICA_HIDE_BANNER", "1")
    environment.setdefault("PYTHONFAULTHANDLER", "1")
    started_at = _utc_now()
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise HarnessError(
            f"{mode} {phase} worker exceeded {timeout_seconds:g} seconds"
        ) from error
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        stderr_text = completed.stderr or ""
        detail = (
            "\n".join(
                section
                for section in (
                    completed.stdout.strip()[-4000:],
                    stderr_text.strip()[-4000:],
                )
                if section
            )
            or "no worker output"
        )
        raise HarnessError(
            f"{mode} {phase} worker failed with exit {completed.returncode}: {detail}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_WORKER_MARKER):
            payload = json.loads(line.removeprefix(_WORKER_MARKER))
            if isinstance(payload, dict):
                process_record = payload.get("worker_process_record")
                payload_without_process_record = dict(payload)
                payload_without_process_record.pop("worker_process_record", None)
                if not isinstance(process_record, Mapping):
                    raise HarnessError(
                        f"{mode} {phase} worker returned no content-addressed "
                        "process record"
                    )
                process_record_without_digest = dict(process_record)
                process_record_digest = process_record_without_digest.pop(
                    "content_sha256",
                    None,
                )
                process_wall_seconds = process_record.get("wall_seconds")
                process_id = process_record.get("process_id")
                if (
                    process_record.get("payload_sha256")
                    != _canonical_sha256(payload_without_process_record)
                    or process_record_digest
                    != _canonical_sha256(process_record_without_digest)
                    or not _is_utc_timestamp(process_record.get("started_at_utc"))
                    or not _is_utc_timestamp(process_record.get("finished_at_utc"))
                    or not _utc_timestamps_nondecreasing(
                        process_record.get("started_at_utc"),
                        process_record.get("finished_at_utc"),
                    )
                    or isinstance(process_wall_seconds, bool)
                    or not isinstance(process_wall_seconds, (float, int))
                    or not math.isfinite(float(process_wall_seconds))
                    or float(process_wall_seconds) <= 0.0
                    or isinstance(process_id, bool)
                    or not isinstance(process_id, int)
                    or process_id <= 0
                    or process_record.get("operation") != expected_operation
                    or process_record.get("mode") != mode
                ):
                    raise HarnessError(
                        f"{mode} {phase} worker process record failed its "
                        "content-address check"
                    )
                command_identity = _command_identity(command)
                payload["worker_command"] = command_identity
                invocation = {
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc_now(),
                    "wall_seconds": time.perf_counter() - started,
                    "command": command_identity,
                }
                invocation["content_sha256"] = _canonical_sha256(invocation)
                payload["worker_invocation"] = invocation
                result_record = {
                    "recorded_at_utc": _utc_now(),
                    "addressed_payload_sha256": _canonical_sha256(payload),
                    "worker_process_record_sha256": process_record_digest,
                    "worker_invocation_sha256": invocation["content_sha256"],
                }
                result_record["content_sha256"] = _canonical_sha256(result_record)
                payload["worker_result_record"] = result_record
                return payload
            break
    raise HarnessError(f"{mode} {phase} worker did not emit a JSON result")


def _assert_identity_unchanged(
    label: str,
    expected: object,
    observed: object,
) -> None:
    if _canonical_sha256(expected) != _canonical_sha256(observed):
        raise HarnessError(f"{label} drifted during a worker or long-running phase")


def _artifact_drift_baseline(
    artifact: Path,
    *,
    generation_signature: Mapping[str, object],
    artifact_identity: Mapping[str, object],
    reuse_signature: Mapping[str, object],
) -> dict[str, object]:
    return {
        "artifact": str(artifact.resolve()),
        "generation_signature": dict(generation_signature),
        "artifact_identity": dict(artifact_identity),
        "reuse_signature": dict(reuse_signature),
        "reuse_sidecar_sha256": _path_identity(_reuse_signature_path(artifact))[
            "sha256"
        ],
    }


def _recheck_driver_state(
    source_identity: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
    artifact_baselines: Mapping[str, Mapping[str, object]],
    *,
    phase: str,
) -> dict[str, object]:
    """Fail immediately if source, runtime, or any completed artifact drifted."""

    observed_source = _git_source_identity()
    _assert_identity_unchanged(
        "benchmark source identity",
        source_identity,
        observed_source,
    )
    observed_runtime = _runtime_provenance(observed_source)
    _assert_identity_unchanged(
        "benchmark runtime provenance",
        runtime_provenance,
        observed_runtime,
    )
    observed_artifacts: dict[str, dict[str, str]] = {}
    for mode, baseline in artifact_baselines.items():
        raw_artifact = baseline.get("artifact")
        expected_generation = baseline.get("generation_signature")
        expected_identity = baseline.get("artifact_identity")
        expected_reuse = baseline.get("reuse_signature")
        if (
            not isinstance(raw_artifact, str)
            or not isinstance(expected_generation, Mapping)
            or not isinstance(expected_identity, Mapping)
            or not isinstance(expected_reuse, Mapping)
        ):
            raise HarnessError(f"driver artifact drift baseline is invalid for {mode}")
        artifact = Path(raw_artifact)
        expected_tree = expected_identity.get("tree")
        if not isinstance(expected_tree, Mapping):
            raise HarnessError(f"driver artifact tree baseline is invalid for {mode}")
        expected_sidecar_sha = _required_sha256(
            baseline.get("reuse_sidecar_sha256"),
            label=f"{mode} reuse sidecar baseline",
        )
        observed_identity, observed_reuse, observed_sidecar_identity = (
            _require_bound_reusable_artifact(
                artifact,
                expected_tree_sha256=_required_sha256(
                    expected_tree.get("sha256"),
                    label=f"{mode} artifact tree baseline",
                ),
                expected_semantic_identity_sha256=_required_sha256(
                    expected_identity.get("semantic_identity_sha256"),
                    label=f"{mode} artifact semantic baseline",
                ),
                expected_reuse_semantic_signature_sha256=_required_sha256(
                    expected_reuse.get("semantic_signature_sha256"),
                    label=f"{mode} reuse semantic baseline",
                ),
                expected_sidecar_sha256=expected_sidecar_sha,
                expected_signature=expected_generation,
            )
        )
        observed_tree = observed_identity.get("tree")
        if not isinstance(observed_tree, Mapping):
            raise HarnessError(f"{mode} bound artifact tree is incomplete")
        _assert_identity_unchanged(
            f"{mode} artifact tree",
            expected_tree,
            observed_tree,
        )
        expected_semantic = expected_identity.get("semantic_identity")
        observed_semantic = observed_identity.get("semantic_identity")
        if not isinstance(expected_semantic, Mapping) or not isinstance(
            observed_semantic,
            Mapping,
        ):
            raise HarnessError(f"{mode} artifact semantic baseline is incomplete")
        _assert_identity_unchanged(
            f"{mode} artifact semantic identity",
            expected_semantic,
            observed_semantic,
        )
        expected_bound_identity = {
            str(key): value
            for key, value in expected_identity.items()
            if key != "payloads"
        }
        _assert_identity_unchanged(
            f"{mode} bound artifact identity",
            expected_bound_identity,
            observed_identity,
        )
        expected_payloads = expected_identity.get("payloads")
        if not isinstance(expected_payloads, list):
            raise HarnessError(
                f"driver artifact payload baseline is invalid for {mode}"
            )
        observed_full_identity = {
            **observed_identity,
            "payloads": list(expected_payloads),
        }
        _assert_identity_unchanged(
            f"{mode} reconstructed full artifact identity",
            expected_identity,
            observed_full_identity,
        )
        _assert_identity_unchanged(
            f"{mode} artifact reuse signature",
            expected_reuse,
            observed_reuse,
        )
        observed_sidecar_sha = _required_sha256(
            observed_sidecar_identity.get("sha256"),
            label=f"{mode} observed reuse sidecar",
        )
        if observed_sidecar_sha != expected_sidecar_sha:
            raise HarnessError(
                f"{mode} artifact reuse sidecar drifted during a worker or long run"
            )
        observed_artifacts[mode] = {
            "artifact_identity_sha256": _canonical_sha256(observed_full_identity),
            "reuse_signature_sha256": _canonical_sha256(observed_reuse),
            "reuse_sidecar_sha256": observed_sidecar_sha,
        }
    return {
        "phase": phase,
        "checked_at_utc": _utc_now(),
        "source_identity_sha256": _canonical_sha256(observed_source),
        "runtime_provenance_sha256": _canonical_sha256(observed_runtime),
        "artifacts": observed_artifacts,
    }


def _complex_values(value: object, *, label: str) -> tuple[complex, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise HarnessError(f"{label} must be a sequence")
    result: list[complex] = []
    for entry in value:
        if (
            not isinstance(entry, Sequence)
            or isinstance(entry, (str, bytes, bytearray))
            or len(entry) != 2
        ):
            raise HarnessError(f"{label} contains an invalid complex value")
        real = entry[0]
        imaginary = entry[1]
        if (
            isinstance(real, bool)
            or not isinstance(real, (float, int))
            or not math.isfinite(float(real))
            or isinstance(imaginary, bool)
            or not isinstance(imaginary, (float, int))
            or not math.isfinite(float(imaginary))
        ):
            raise HarnessError(f"{label} contains an invalid complex value")
        result.append(complex(float(real), float(imaginary)))
    if not result:
        raise HarnessError(f"{label} contains no validation values")
    return tuple(result)


def _comparison(
    left_mode: str,
    left: object,
    right_mode: str,
    right: object,
) -> dict[str, object]:
    left_values = _complex_values(left, label=f"{left_mode} validation values")
    right_values = _complex_values(right, label=f"{right_mode} validation values")
    point_comparisons: list[dict[str, object]] = []
    for index, (left_value, right_value) in enumerate(
        zip(left_values, right_values, strict=False)
    ):
        absolute = abs(left_value - right_value)
        relative = absolute / max(abs(left_value), abs(right_value), 1.0e-300)
        point_comparisons.append(
            {
                "point_index": index,
                "left": _complex_payload(left_value),
                "right": _complex_payload(right_value),
                "absolute_difference": absolute,
                "relative_difference": relative,
                "passes": _within_numerical_tolerance(
                    left_value,
                    right_value,
                ),
            }
        )
    counts_match = len(left_values) == len(right_values)
    return {
        "left_mode": left_mode,
        "right_mode": right_mode,
        "left_point_count": len(left_values),
        "right_point_count": len(right_values),
        "point_counts_match": counts_match,
        "point_comparisons": point_comparisons,
        "maximum_absolute_difference": max(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left_values,
                right_values,
                strict=False,
            )
        ),
        "maximum_relative_difference": max(
            abs(left_value - right_value)
            / max(abs(left_value), abs(right_value), 1.0e-300)
            for left_value, right_value in zip(
                left_values,
                right_values,
                strict=False,
            )
        ),
        "passes": counts_match
        and all(bool(item["passes"]) for item in point_comparisons),
    }


def _validation_fixture_contract(
    validation: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, object]:
    fixture = validation.get("fixture")
    if not isinstance(fixture, Mapping):
        raise HarnessError(f"{mode} profile has no validation fixture identity")
    raw_file = fixture.get("file")
    if not isinstance(raw_file, Mapping):
        raise HarnessError(f"{mode} profile has no validation fixture file identity")
    point_count = fixture.get("point_count")
    points_sha256 = fixture.get("points_sha256")
    file_sha256 = raw_file.get("sha256")
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count <= 0
        or not isinstance(points_sha256, str)
        or _SHA256_PATTERN.fullmatch(points_sha256) is None
        or not isinstance(file_sha256, str)
        or _SHA256_PATTERN.fullmatch(file_sha256) is None
    ):
        raise HarnessError(f"{mode} profile validation fixture identity is invalid")
    return {
        "point_count": point_count,
        "points_sha256": points_sha256,
        "file_sha256": file_sha256,
    }


def _profile_selector_contract_matches(
    arguments: argparse.Namespace,
    selector_contract: object,
    semantic_identity: object,
) -> bool:
    if not isinstance(selector_contract, Mapping) or set(selector_contract) != {
        "color_flow_request",
        "resolved_color_flow_id",
        "helicity_request",
        "resolved_helicity_id",
        "color_flow_count",
        "helicity_count",
        "structural_zero_helicity_count",
        "workload",
    }:
        return False
    if not isinstance(semantic_identity, Mapping):
        return False
    color_axis = semantic_identity.get("physical_color_flows")
    helicity_axis = semantic_identity.get("physical_helicities")
    if not isinstance(color_axis, Mapping) or not isinstance(
        helicity_axis,
        Mapping,
    ):
        return False
    color_ids = color_axis.get("ordered_ids")
    helicity_ids = helicity_axis.get("ordered_ids")
    helicity_entries = helicity_axis.get("ordered_entries")
    if (
        not isinstance(color_ids, list)
        or not isinstance(helicity_ids, list)
        or not isinstance(helicity_entries, list)
    ):
        return False
    structural_zero_ids = {
        entry.get("id")
        for entry in helicity_entries
        if isinstance(entry, Mapping) and entry.get("structural_zero") is True
    }
    color_count = selector_contract.get("color_flow_count")
    helicity_count = selector_contract.get("helicity_count")
    structural_zero_count = selector_contract.get("structural_zero_helicity_count")
    if (
        selector_contract.get("color_flow_request") != arguments.color_flow
        or selector_contract.get("helicity_request") != arguments.helicity
        or isinstance(color_count, bool)
        or not isinstance(color_count, int)
        or color_count <= 0
        or isinstance(helicity_count, bool)
        or not isinstance(helicity_count, int)
        or helicity_count <= 0
        or isinstance(structural_zero_count, bool)
        or not isinstance(structural_zero_count, int)
        or structural_zero_count < 0
        or structural_zero_count > helicity_count
        or not _is_exact_int(color_axis.get("count"), color_count)
        or not _is_exact_int(helicity_axis.get("count"), helicity_count)
        or structural_zero_count != len(structural_zero_ids)
    ):
        return False
    color_id = selector_contract.get("resolved_color_flow_id")
    helicity_id = selector_contract.get("resolved_helicity_id")
    if arguments.lc_flow_layout == "all-flow-union":
        return (
            color_id is None
            and isinstance(helicity_id, str)
            and bool(helicity_id)
            and helicity_id in helicity_ids
            and helicity_id not in structural_zero_ids
            and selector_contract.get("workload")
            == "all-flows/runtime-selected-single-helicity"
        )
    return (
        isinstance(color_id, str)
        and bool(color_id)
        and color_id in color_ids
        and helicity_id is None
        and selector_contract.get("workload")
        == "single-runtime-selected-flow/helicity-sum"
    )


def _selected_union_selector_projection(
    arguments: argparse.Namespace,
    selector_contract: object,
    semantic_identity: object,
) -> dict[str, object]:
    """Project away only the audited nonselected structural-zero census."""

    if not _selected_union_semantic_exception_enabled(arguments):
        raise HarnessError(
            "selected-union selector projection is outside its audited workload"
        )
    if not _profile_selector_contract_matches(
        arguments,
        selector_contract,
        semantic_identity,
    ):
        raise HarnessError("selected-union selector contract is invalid")
    assert isinstance(selector_contract, Mapping)
    if (
        selector_contract.get("resolved_helicity_id")
        != MIGRATION_UNION_SELECTED_HELICITY_ID
        or selector_contract.get("workload")
        != "all-flows/runtime-selected-single-helicity"
    ):
        raise HarnessError("selected-union selector does not resolve the audited ID")
    return {
        key: selector_contract[key]
        for key in (
            "color_flow_request",
            "resolved_color_flow_id",
            "helicity_request",
            "resolved_helicity_id",
            "color_flow_count",
            "helicity_count",
            "workload",
        )
    }


def _validated_lane_validation_values(
    validation: Mapping[str, Any],
    fixture_contract: Mapping[str, object],
    *,
    mode: str,
) -> tuple[complex, ...]:
    point_count = fixture_contract.get("point_count")
    if isinstance(point_count, bool) or not isinstance(point_count, int):
        raise HarnessError(f"{mode} validation fixture point count is invalid")
    selected = _complex_values(
        validation.get("selected_totals"),
        label=f"{mode} selected validation totals",
    )
    resolved = _complex_values(
        validation.get("resolved_sums"),
        label=f"{mode} resolved validation sums",
    )
    comparisons = validation.get("point_comparisons")
    if (
        len(selected) != point_count
        or len(resolved) != point_count
        or not isinstance(comparisons, list)
        or len(comparisons) != point_count
    ):
        raise HarnessError(f"{mode} validation inventory disagrees with fixture")
    absolute_values: list[float] = []
    relative_values: list[float] = []
    expected_passes: list[bool] = []
    for index, (selected_value, resolved_value, comparison) in enumerate(
        zip(selected, resolved, comparisons, strict=True)
    ):
        absolute = abs(selected_value - resolved_value)
        relative = absolute / max(
            abs(selected_value),
            abs(resolved_value),
            1.0e-300,
        )
        expected_pass = _within_numerical_tolerance(
            selected_value,
            resolved_value,
        )
        if (
            not math.isfinite(absolute)
            or not math.isfinite(relative)
            or not isinstance(comparison, Mapping)
            or set(comparison)
            != {
                "point_index",
                "selected_total",
                "resolved_sum",
                "absolute_difference",
                "relative_difference",
                "passes",
            }
            or not _is_exact_int(comparison.get("point_index"), index)
            or _canonical_sha256(comparison.get("selected_total"))
            != _canonical_sha256(_complex_payload(selected_value))
            or _canonical_sha256(comparison.get("resolved_sum"))
            != _canonical_sha256(_complex_payload(resolved_value))
            or isinstance(comparison.get("absolute_difference"), bool)
            or not isinstance(
                comparison.get("absolute_difference"),
                (float, int),
            )
            or float(comparison["absolute_difference"]) != absolute
            or isinstance(comparison.get("relative_difference"), bool)
            or not isinstance(
                comparison.get("relative_difference"),
                (float, int),
            )
            or float(comparison["relative_difference"]) != relative
            or comparison.get("passes") is not expected_pass
        ):
            raise HarnessError(f"{mode} validation point evidence is inconsistent")
        absolute_values.append(absolute)
        relative_values.append(relative)
        expected_passes.append(expected_pass)
    maximum_absolute = validation.get("maximum_absolute_difference")
    maximum_relative = validation.get("maximum_relative_difference")
    lane_passes = all(expected_passes)
    if (
        isinstance(maximum_absolute, bool)
        or not isinstance(maximum_absolute, (float, int))
        or float(maximum_absolute) != max(absolute_values)
        or isinstance(maximum_relative, bool)
        or not isinstance(maximum_relative, (float, int))
        or float(maximum_relative) != max(relative_values)
        or validation.get("passes") is not lane_passes
    ):
        raise HarnessError(f"{mode} validation summary is inconsistent")
    resolved_components = _validated_resolved_components(
        validation,
        point_count=point_count,
        mode=mode,
    )
    for resolved_value, components in zip(
        resolved,
        resolved_components,
        strict=True,
    ):
        component_sum = sum(components, 0.0j)
        absolute = abs(resolved_value - component_sum)
        relative = absolute / max(
            abs(resolved_value),
            abs(component_sum),
            1.0e-300,
        )
        if not _within_numerical_tolerance(component_sum, resolved_value):
            raise HarnessError(
                f"{mode} resolved sum disagrees with its physical components"
            )
    return selected


def _validated_resolved_components(
    validation: Mapping[str, Any],
    *,
    point_count: int,
    mode: str,
) -> tuple[tuple[complex, ...], ...]:
    helicity_ids = validation.get("resolved_helicity_ids")
    color_ids = validation.get("resolved_color_ids")
    raw_points = validation.get("resolved_components")
    if (
        not isinstance(helicity_ids, list)
        or not helicity_ids
        or any(not isinstance(value, str) or not value for value in helicity_ids)
        or len(set(helicity_ids)) != len(helicity_ids)
        or not isinstance(color_ids, list)
        or not color_ids
        or any(not isinstance(value, str) or not value for value in color_ids)
        or len(set(color_ids)) != len(color_ids)
        or not isinstance(raw_points, list)
        or len(raw_points) != point_count
    ):
        raise HarnessError(f"{mode} resolved-component inventory is invalid")
    expected_component_count = len(helicity_ids) * len(color_ids)
    points: list[tuple[complex, ...]] = []
    for raw_point in raw_points:
        point = _complex_values(
            raw_point,
            label=f"{mode} resolved component values",
        )
        if len(point) != expected_component_count:
            raise HarnessError(
                f"{mode} resolved-component inventory disagrees with its axes"
            )
        points.append(point)
    return tuple(points)


def _resolved_component_comparison(
    left_mode: str,
    left: Mapping[str, Any],
    right_mode: str,
    right: Mapping[str, Any],
) -> dict[str, object]:
    left_raw_points = left.get("resolved_components")
    right_raw_points = right.get("resolved_components")
    left_points = _validated_resolved_components(
        left,
        point_count=len(left_raw_points) if isinstance(left_raw_points, list) else -1,
        mode=left_mode,
    )
    right_points = _validated_resolved_components(
        right,
        point_count=(
            len(right_raw_points) if isinstance(right_raw_points, list) else -1
        ),
        mode=right_mode,
    )
    axes_match = left.get("resolved_helicity_ids") == right.get(
        "resolved_helicity_ids"
    ) and left.get("resolved_color_ids") == right.get("resolved_color_ids")
    point_counts_match = len(left_points) == len(right_points)
    component_counts_match = point_counts_match and all(
        len(left_point) == len(right_point)
        for left_point, right_point in zip(left_points, right_points, strict=True)
    )
    maximum_absolute = 0.0
    maximum_relative = 0.0
    values_pass = True
    compared_component_count = 0
    for left_point, right_point in zip(left_points, right_points, strict=False):
        for left_value, right_value in zip(
            left_point,
            right_point,
            strict=False,
        ):
            absolute = abs(left_value - right_value)
            relative = absolute / max(abs(left_value), abs(right_value), 1.0e-300)
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            values_pass &= _within_numerical_tolerance(left_value, right_value)
            compared_component_count += 1
    return {
        "left_mode": left_mode,
        "right_mode": right_mode,
        "axes_match": axes_match,
        "left_point_count": len(left_points),
        "right_point_count": len(right_points),
        "point_counts_match": point_counts_match,
        "component_counts_match": component_counts_match,
        "compared_component_count": compared_component_count,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "passes": (
            axes_match and point_counts_match and component_counts_match and values_pass
        ),
    }


def _pairwise_profile_validation(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    arguments: argparse.Namespace | None = None,
) -> dict[str, object]:
    unknown = set(profiles).difference(EXECUTION_MODES)
    if unknown:
        raise HarnessError(
            "profile validation received unknown modes: " + ", ".join(sorted(unknown))
        )
    comparisons: dict[str, dict[str, object]] = {}
    resolved_component_comparisons: dict[str, dict[str, object]] = {}
    ordered_modes = [mode for mode in EXECUTION_MODES if mode in profiles]
    if not ordered_modes:
        raise HarnessError("profile validation requires at least one lane")
    lane_validation_passes = True
    selector_contracts: list[object] = []
    selector_projections: list[dict[str, object]] = []
    fixture_contracts: list[dict[str, object]] = []
    selected_union_projection = _selected_union_semantic_exception_enabled(arguments)
    for mode in ordered_modes:
        validation = profiles[mode].get("validation")
        if not isinstance(validation, Mapping):
            raise HarnessError(f"{mode} profile returned invalid validation metadata")
        selector_contract = profiles[mode].get("selector_contract")
        selector_contracts.append(selector_contract)
        if selected_union_projection:
            assert arguments is not None
            selector_projections.append(
                _selected_union_selector_projection(
                    arguments,
                    selector_contract,
                    profiles[mode].get("artifact_semantic_identity"),
                )
            )
        fixture_contract = _validation_fixture_contract(validation, mode=mode)
        fixture_contracts.append(fixture_contract)
        _validated_lane_validation_values(
            validation,
            fixture_contract,
            mode=mode,
        )
        lane_validation_passes = (
            lane_validation_passes and validation.get("passes") is True
        )
    for left_index, left_mode in enumerate(ordered_modes):
        left_validation = profiles[left_mode]["validation"]
        for right_mode in ordered_modes[left_index + 1 :]:
            right_validation = profiles[right_mode]["validation"]
            key = f"{left_mode}__{right_mode}"
            comparisons[key] = _comparison(
                left_mode,
                left_validation["selected_totals"],
                right_mode,
                right_validation["selected_totals"],
            )
            resolved_component_comparisons[key] = _resolved_component_comparison(
                left_mode,
                left_validation,
                right_mode,
                right_validation,
            )
    selectors_match = (
        all(
            projection == selector_projections[0]
            for projection in selector_projections[1:]
        )
        if selected_union_projection
        else all(
            contract == selector_contracts[0] for contract in selector_contracts[1:]
        )
    )
    fixtures_match = all(
        contract == fixture_contracts[0] for contract in fixture_contracts[1:]
    )
    pairwise_passes = (
        None
        if not comparisons
        else all(bool(item["passes"]) for item in comparisons.values())
        and all(
            bool(item["passes"]) for item in resolved_component_comparisons.values()
        )
    )
    validation_passes = (
        lane_validation_passes
        and selectors_match
        and fixtures_match
        and pairwise_passes is not False
    )
    return {
        "comparisons": comparisons,
        "resolved_component_comparisons": resolved_component_comparisons,
        "selectors_match": selectors_match,
        "selector_comparison_policy": (
            "ddbar-z6g-all-flow-union-selected-helicity-v1"
            if selected_union_projection
            else "strict-exact-v1"
        ),
        "fixtures_match": fixtures_match,
        "fixture_contract": fixture_contracts[0] if fixtures_match else None,
        "lane_validation_passes": lane_validation_passes,
        "pairwise_validation_passes": pairwise_passes,
        "passes": validation_passes,
    }


def _preflight_profile_from_worker(
    worker: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the immutable semantic/numerical body from a preflight worker."""

    mode = worker.get("mode")
    verification = worker.get("pre_timing_verification")
    measurements = worker.get("profiles")
    if (
        not isinstance(mode, str)
        or mode not in EXECUTION_MODES
        or not isinstance(verification, Mapping)
        or measurements != []
    ):
        raise HarnessError("validation preflight worker returned an invalid payload")
    semantic_identity = verification.get("artifact_semantic_identity")
    semantic_identity_sha256 = verification.get("artifact_semantic_identity_sha256")
    if not isinstance(
        semantic_identity, Mapping
    ) or semantic_identity_sha256 != _canonical_sha256(semantic_identity):
        raise HarnessError(
            "validation preflight worker returned unaddressed artifact semantics"
        )
    return {
        "mode": mode,
        "process_id": worker.get("process_id"),
        "process_expression": worker.get("process_expression"),
        "selector_contract": worker.get("selector_contract"),
        "validation": worker.get("validation"),
        "artifact_semantic_identity": dict(semantic_identity),
        "artifact_semantic_identity_sha256": semantic_identity_sha256,
        "profiles": [],
    }


def _validation_preflight_contract(
    arguments: argparse.Namespace,
    profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Validate lane semantics and deterministic physics before any timing."""

    semantic = _profile_artifact_semantic_contract(
        profiles,
        arguments=arguments,
    )
    numerical = _pairwise_profile_validation(
        profiles,
        arguments=arguments,
    )
    passes = (
        bool(profiles)
        and semantic.get("passes") is True
        and semantic.get("lanes_match") is True
        and numerical.get("passes") is True
    )
    return {
        "kind": "pyamplicol-pre-profile-validation-contract",
        "schema_version": 1,
        "passes": passes,
        "observed_modes": [mode for mode in EXECUTION_MODES if mode in profiles],
        "semantic_contract": semantic,
        "numerical_contract": numerical,
    }


def _require_validation_preflight(
    arguments: argparse.Namespace,
    profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    try:
        contract = _validation_preflight_contract(arguments, profiles)
    except HarnessError as error:
        raise HarnessError(
            "pre-profile lane validation failed before timing: " + str(error)
        ) from error
    if contract["passes"] is not True:
        semantic = contract["semantic_contract"]
        assert isinstance(semantic, Mapping)
        raw_errors = semantic.get("errors")
        errors = (
            [str(error) for error in raw_errors] if isinstance(raw_errors, list) else []
        )
        detail = "; ".join(errors) or "deterministic numerical comparison failed"
        raise HarnessError(
            "pre-profile lane validation failed before timing: " + detail
        )
    return contract


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _content_addressed_record(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["content_sha256"] = _canonical_sha256(result)
    return result


def _read_content_addressed_record(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"{label} is not readable canonical JSON: {path}") from error
    if not isinstance(payload, dict):
        raise HarnessError(f"{label} is not a JSON object: {path}")
    unsigned = dict(payload)
    digest = unsigned.pop("content_sha256", None)
    if (
        not isinstance(digest, str)
        or _SHA256_PATTERN.fullmatch(digest) is None
        or digest != _canonical_sha256(unsigned)
    ):
        raise HarnessError(f"{label} is not content-addressed: {path}")
    return payload


def _paired_coordination_role_directory(arguments: argparse.Namespace) -> Path | None:
    root = arguments.paired_profile_coordination_dir
    if root is None:
        return None
    return (
        root.resolve()
        / arguments.paired_profile_session_id
        / arguments.lc_flow_layout
        / arguments.paired_profile_role
    )


def _paired_schedule_plan(
    profile_schedule: Mapping[str, object],
) -> list[dict[str, object]]:
    raw_entries = profile_schedule.get("entries")
    if not isinstance(raw_entries, list):
        raise HarnessError("paired profile coordination requires a planned schedule")
    result: list[dict[str, object]] = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping):
            raise HarnessError(f"paired profile schedule entry {index} is invalid")
        result.append(
            {
                field: raw.get(field)
                for field in ("schedule_index", "round", "mode", "batch_size")
            }
        )
    return result


def _initialize_paired_profile_coordination(
    arguments: argparse.Namespace,
    *,
    source_identity: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
    profile_schedule: Mapping[str, object],
    result_json: Path,
) -> dict[str, object] | None:
    role_directory = _paired_coordination_role_directory(arguments)
    if role_directory is None:
        return None
    role_directory.mkdir(parents=True, exist_ok=True)
    plan = _paired_schedule_plan(profile_schedule)
    ready = _content_addressed_record(
        {
            "kind": PAIRED_PROFILE_READY_KIND,
            "schema_version": PAIRED_PROFILE_COORDINATION_SCHEMA,
            "session_id": arguments.paired_profile_session_id,
            "role": arguments.paired_profile_role,
            "layout": arguments.lc_flow_layout,
            "source_revision": source_identity.get("revision"),
            "source_identity_sha256": _canonical_sha256(source_identity),
            "runtime_provenance_sha256": _canonical_sha256(runtime_provenance),
            "profile_schedule_plan": plan,
            "profile_schedule_plan_sha256": _canonical_sha256(plan),
            "result_json": str(result_json),
            "ready_at_utc": _utc_now(),
        }
    )
    ready_path = role_directory / "ready.json"
    if ready_path.exists():
        raise HarnessError(
            "paired profile coordination ready record already exists; "
            "use a fresh session directory"
        )
    _write_json_atomic(ready_path, ready)
    return {
        "kind": PAIRED_PROFILE_COORDINATION_KIND,
        "schema_version": PAIRED_PROFILE_COORDINATION_SCHEMA,
        "session_id": arguments.paired_profile_session_id,
        "role": arguments.paired_profile_role,
        "layout": arguments.lc_flow_layout,
        "ready_record": ready,
        "completion_records": [],
    }


def _wait_for_paired_profile_token(
    arguments: argparse.Namespace,
    schedule_entry: Mapping[str, object],
) -> dict[str, object] | None:
    role_directory = _paired_coordination_role_directory(arguments)
    if role_directory is None:
        return None
    schedule_index = schedule_entry.get("schedule_index")
    if isinstance(schedule_index, bool) or not isinstance(schedule_index, int):
        raise HarnessError("paired profile schedule index is invalid")
    path = role_directory / f"token-{schedule_index:06d}.json"
    deadline = time.monotonic() + arguments.paired_profile_wait_timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise HarnessError(f"timed out waiting for paired profile token: {path}")
        time.sleep(0.1)
    token = _read_content_addressed_record(path, label="paired profile token")
    expected = {
        "kind": PAIRED_PROFILE_TOKEN_KIND,
        "schema_version": PAIRED_PROFILE_COORDINATION_SCHEMA,
        "session_id": arguments.paired_profile_session_id,
        "role": arguments.paired_profile_role,
        "layout": arguments.lc_flow_layout,
        **{
            field: schedule_entry.get(field)
            for field in ("schedule_index", "round", "mode", "batch_size")
        },
    }
    if any(token.get(field) != value for field, value in expected.items()):
        raise HarnessError(f"paired profile token does not match its slot: {path}")
    if not _is_utc_timestamp(token.get("issued_at_utc")):
        raise HarnessError(f"paired profile token has no valid timestamp: {path}")
    return token


def _record_paired_profile_completion(
    arguments: argparse.Namespace,
    *,
    schedule_entry: Mapping[str, object],
    token: Mapping[str, object] | None,
    worker_result: Mapping[str, object],
) -> dict[str, object] | None:
    role_directory = _paired_coordination_role_directory(arguments)
    if role_directory is None:
        return None
    if token is None:
        raise HarnessError("paired profile completion has no scheduler token")
    invocation = worker_result.get("worker_invocation")
    result_record = worker_result.get("worker_result_record")
    if not isinstance(invocation, Mapping) or not isinstance(result_record, Mapping):
        raise HarnessError("paired profile completion lacks worker provenance")
    schedule_index = schedule_entry.get("schedule_index")
    assert isinstance(schedule_index, int)
    completion = _content_addressed_record(
        {
            "kind": PAIRED_PROFILE_COMPLETION_KIND,
            "schema_version": PAIRED_PROFILE_COORDINATION_SCHEMA,
            "session_id": arguments.paired_profile_session_id,
            "role": arguments.paired_profile_role,
            "layout": arguments.lc_flow_layout,
            **{
                field: schedule_entry.get(field)
                for field in ("schedule_index", "round", "mode", "batch_size")
            },
            "token_sha256": token.get("content_sha256"),
            "worker_invocation_sha256": invocation.get("content_sha256"),
            "worker_result_record_sha256": result_record.get("content_sha256"),
            "worker_started_at_utc": invocation.get("started_at_utc"),
            "worker_finished_at_utc": invocation.get("finished_at_utc"),
            "recorded_at_utc": _utc_now(),
        }
    )
    path = role_directory / f"completion-{schedule_index:06d}.json"
    if path.exists():
        raise HarnessError(f"paired profile completion already exists: {path}")
    _write_json_atomic(path, completion)
    return completion


def _build_profile_schedule(
    modes: Sequence[str],
    batch_sizes: Sequence[int],
    *,
    subprocess_samples: int,
) -> dict[str, object]:
    if subprocess_samples <= 0 or not modes or not batch_sizes:
        raise HarnessError("profile schedule dimensions must be non-empty")
    entries: list[dict[str, object]] = []
    schedule_index = 0
    for round_index in range(subprocess_samples):
        rotated_modes = tuple(modes[round_index % len(modes) :]) + tuple(
            modes[: round_index % len(modes)]
        )
        rotated_batches = tuple(batch_sizes[round_index % len(batch_sizes) :]) + tuple(
            batch_sizes[: round_index % len(batch_sizes)]
        )
        for batch_size in rotated_batches:
            for mode in rotated_modes:
                entries.append(
                    {
                        "schedule_index": schedule_index,
                        "round": round_index,
                        "mode": mode,
                        "batch_size": batch_size,
                    }
                )
                schedule_index += 1
    return {
        "kind": PROFILE_SCHEDULE_KIND,
        "schema_version": PROFILE_SCHEDULE_SCHEMA,
        "algorithm": "round-major-cyclic-mode-and-batch-interleave-v1",
        "sample_unit": "independent-profile-worker-subprocess",
        "subprocess_samples_per_cell": subprocess_samples,
        "modes": list(modes),
        "batch_sizes": list(batch_sizes),
        "entries": entries,
    }


def _profile_schedule_contract(
    schedule: Mapping[str, object] | None,
    *,
    modes: Sequence[str],
    batch_sizes: Sequence[int],
    subprocess_samples: int,
) -> dict[str, object]:
    errors: list[str] = []
    if not isinstance(schedule, Mapping):
        return {
            "passes": False,
            "errors": ["interleaved subprocess schedule is missing"],
        }
    expected = _build_profile_schedule(
        modes,
        batch_sizes,
        subprocess_samples=subprocess_samples,
    )
    for key in (
        "kind",
        "algorithm",
        "sample_unit",
    ):
        if schedule.get(key) != expected[key]:
            errors.append(f"profile schedule {key} does not match its contract")
    for key in ("modes", "batch_sizes"):
        if _canonical_sha256(schedule.get(key)) != _canonical_sha256(expected[key]):
            errors.append(f"profile schedule {key} does not match its contract")
    if not _is_exact_int(
        schedule.get("schema_version"),
        PROFILE_SCHEDULE_SCHEMA,
    ):
        errors.append("profile schedule schema_version does not match its contract")
    if not _is_exact_int(
        schedule.get("subprocess_samples_per_cell"),
        subprocess_samples,
    ):
        errors.append(
            "profile schedule subprocess_samples_per_cell does not match its contract"
        )
    raw_entries = schedule.get("entries")
    expected_entries = expected["entries"]
    assert isinstance(expected_entries, list)
    if not isinstance(raw_entries, list):
        raw_entries = []
        errors.append("profile schedule entries are missing")
    if len(raw_entries) != len(expected_entries):
        errors.append("profile schedule entry count is incomplete")
    command_digests: list[str] = []
    cells: dict[tuple[str, int], int] = {}
    for index, (raw, planned) in enumerate(
        zip(raw_entries, expected_entries, strict=False)
    ):
        if not isinstance(raw, Mapping):
            errors.append(f"profile schedule entry {index} is invalid")
            continue
        schedule_index = raw.get("schedule_index")
        round_index = raw.get("round")
        mode = raw.get("mode")
        batch_size = raw.get("batch_size")
        if (
            isinstance(schedule_index, bool)
            or not isinstance(schedule_index, int)
            or schedule_index < 0
            or isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index < 0
            or not isinstance(mode, str)
            or not mode
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            errors.append(
                f"profile schedule entry {index} has invalid slot coordinates"
            )
        for field in ("schedule_index", "round", "mode", "batch_size"):
            if raw.get(field) != planned[field]:
                errors.append(f"profile schedule entry {index} changed planned {field}")
        if (
            isinstance(mode, str)
            and mode
            and not isinstance(batch_size, bool)
            and isinstance(batch_size, int)
            and batch_size > 0
        ):
            cell = (mode, batch_size)
            cells[cell] = cells.get(cell, 0) + 1
        invocation = raw.get("worker_invocation")
        if not isinstance(invocation, Mapping):
            errors.append(f"profile schedule entry {index} has no worker provenance")
            continue
        try:
            command = _validated_command_identity(invocation.get("command"))
        except HarnessError:
            errors.append(f"profile schedule entry {index} command is invalid")
            continue
        command_digests.append(str(command["argv_sha256"]))
        for timestamp_field in ("started_at_utc", "finished_at_utc"):
            timestamp = invocation.get(timestamp_field)
            if not _is_utc_timestamp(timestamp):
                errors.append(
                    f"profile schedule entry {index} has invalid {timestamp_field}"
                )
        if not _utc_timestamps_nondecreasing(
            invocation.get("started_at_utc"),
            invocation.get("finished_at_utc"),
        ):
            errors.append(
                f"profile schedule entry {index} has inverted invocation timestamps"
            )
        wall_seconds = invocation.get("wall_seconds")
        if (
            isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, (float, int))
            or not math.isfinite(float(wall_seconds))
            or float(wall_seconds) <= 0.0
        ):
            errors.append(
                f"profile schedule entry {index} has invalid worker wall time"
            )
        invocation_without_digest = dict(invocation)
        invocation_digest = invocation_without_digest.pop("content_sha256", None)
        if invocation_digest != _canonical_sha256(invocation_without_digest):
            errors.append(
                f"profile schedule entry {index} invocation is not content-addressed"
            )
        verification = raw.get("pre_timing_verification")
        if (
            not isinstance(verification, Mapping)
            or verification.get("kind") != WORKER_VERIFICATION_KIND
            or not _is_exact_int(
                verification.get("schema_version"),
                WORKER_VERIFICATION_SCHEMA,
            )
            or not _is_utc_timestamp(verification.get("verified_at_utc"))
        ):
            errors.append(
                f"profile schedule entry {index} lacks pre-timing verification"
            )
        else:
            expected_identities = verification.get("expected")
            observed_identities = verification.get("observed")
            effective_contract = verification.get("effective_contract")
            if not isinstance(expected_identities, Mapping) or not isinstance(
                observed_identities,
                Mapping,
            ):
                errors.append(
                    f"profile schedule entry {index} identity verification is invalid"
                )
            else:
                try:
                    _validate_profile_worker_expectations(
                        expected_identities,
                        observed_identities,
                    )
                    _validate_loaded_runtime_artifact_verification(
                        verification.get("loaded_runtime_artifact"),
                        expected_artifact_id=expected_identities.get("artifact_id"),
                        phase="after-native-load-before-timing",
                    )
                except HarnessError:
                    errors.append(
                        f"profile schedule entry {index} contains identity drift"
                    )
                semantic_identity_sha256 = verification.get(
                    "artifact_semantic_identity_sha256"
                )
                if (
                    not isinstance(semantic_identity_sha256, str)
                    or _SHA256_PATTERN.fullmatch(semantic_identity_sha256) is None
                    or expected_identities.get("artifact_semantic_identity_sha256")
                    != semantic_identity_sha256
                    or observed_identities.get("artifact_semantic_identity_sha256")
                    != semantic_identity_sha256
                ):
                    errors.append(
                        f"profile schedule entry {index} artifact semantic "
                        "identity is not bound to verification"
                    )
            if (
                not isinstance(effective_contract, Mapping)
                or effective_contract.get("execution_mode") != mode
                or effective_contract.get("backend") != "jit"
                or effective_contract.get("color_accuracy") != "lc"
            ):
                errors.append(
                    f"profile schedule entry {index} effective contract is invalid"
                )
        result_digest = raw.get("worker_result_sha256")
        result_record = raw.get("worker_result_record")
        if (
            not isinstance(result_digest, str)
            or _SHA256_PATTERN.fullmatch(result_digest) is None
        ):
            errors.append(
                f"profile schedule entry {index} lacks a worker-result digest"
            )
        if not isinstance(result_record, Mapping):
            errors.append(
                f"profile schedule entry {index} lacks a worker-result record"
            )
        else:
            record_without_digest = dict(result_record)
            record_digest = record_without_digest.pop("content_sha256", None)
            recorded_at = result_record.get("recorded_at_utc")
            if (
                record_digest != _canonical_sha256(record_without_digest)
                or result_record.get("kind") != RETAINED_WORKER_RESULT_KIND
                or not _is_exact_int(
                    result_record.get("schema_version"),
                    RETAINED_WORKER_RESULT_SCHEMA,
                )
                or not _is_utc_timestamp(recorded_at)
                or not _utc_timestamps_nondecreasing(
                    invocation.get("finished_at_utc"),
                    recorded_at,
                )
                or any(
                    _SHA256_PATTERN.fullmatch(value) is None
                    for value in (
                        result_record.get("addressed_payload_sha256"),
                        result_record.get("upstream_worker_result_record_sha256"),
                        result_record.get("worker_process_record_sha256"),
                        result_record.get("worker_invocation_sha256"),
                    )
                    if isinstance(value, str)
                )
                or not all(
                    isinstance(result_record.get(field), str)
                    for field in (
                        "addressed_payload_sha256",
                        "upstream_worker_result_record_sha256",
                        "worker_process_record_sha256",
                        "worker_invocation_sha256",
                    )
                )
                or result_record.get("worker_invocation_sha256") != invocation_digest
                or result_record.get("addressed_payload_sha256") != result_digest
            ):
                errors.append(
                    f"profile schedule entry {index} result record is not "
                    "content-addressed"
                )
    if len(command_digests) != len(set(command_digests)):
        errors.append("profile schedule reused a subprocess command identity")
    for mode in modes:
        for batch_size in batch_sizes:
            if cells.get((mode, batch_size), 0) != subprocess_samples:
                errors.append(
                    f"profile schedule cell {mode}/{batch_size} has the wrong "
                    "subprocess sample count"
                )
    if len(modes) > 1:
        observed_modes = [
            entry.get("mode") for entry in raw_entries if isinstance(entry, Mapping)
        ]
        if any(left == right for left, right in pairwise(observed_modes)):
            errors.append("profile subprocess lanes are not interleaved")
    return {
        "passes": not errors,
        "errors": errors,
        "entry_count": len(raw_entries),
        "unique_worker_command_count": len(set(command_digests)),
        "subprocess_samples_per_cell": subprocess_samples,
    }


def _compact_profile_verification(
    verification: Mapping[str, object],
) -> dict[str, object]:
    return {
        "kind": verification.get("kind"),
        "schema_version": verification.get("schema_version"),
        "verified_at_utc": verification.get("verified_at_utc"),
        "expected": verification.get("expected"),
        "observed": verification.get("observed"),
        "artifact_semantic_identity_sha256": verification.get(
            "artifact_semantic_identity_sha256"
        ),
        "effective_contract": verification.get("effective_contract"),
        "loaded_runtime_artifact": verification.get("loaded_runtime_artifact"),
    }


def _retained_profile_sample_evidence(
    sample: Mapping[str, object],
) -> dict[str, object]:
    return {
        str(key): value
        for key, value in sample.items()
        if key != "worker_result_record"
    }


def _preserved_worker_result_evidence(
    result: Mapping[str, object],
) -> dict[str, object]:
    result_record = result.get("worker_result_record")
    process_record = result.get("worker_process_record")
    invocation = result.get("worker_invocation")
    payload = {
        str(key): value
        for key, value in result.items()
        if key != "worker_result_record"
    }
    operation_payload = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "worker_process_record",
            "worker_command",
            "worker_invocation",
        }
    }
    if (
        not isinstance(result_record, Mapping)
        or not isinstance(process_record, Mapping)
        or not isinstance(invocation, Mapping)
    ):
        raise HarnessError("worker result cannot be preserved without provenance")
    result_record_without_digest = dict(result_record)
    result_record_sha256 = result_record_without_digest.pop("content_sha256", None)
    process_record_without_digest = dict(process_record)
    process_record_sha256 = process_record_without_digest.pop("content_sha256", None)
    invocation_without_digest = dict(invocation)
    invocation_sha256 = invocation_without_digest.pop("content_sha256", None)
    process_wall_seconds = process_record.get("wall_seconds")
    process_id = process_record.get("process_id")
    invocation_wall_seconds = invocation.get("wall_seconds")
    try:
        worker_command = _validated_command_identity(payload.get("worker_command"))
        invocation_command = _validated_command_identity(invocation.get("command"))
    except HarnessError as error:
        raise HarnessError(
            "worker result evidence has invalid command provenance"
        ) from error
    if (
        set(process_record)
        != {
            "started_at_utc",
            "finished_at_utc",
            "wall_seconds",
            "process_id",
            "operation",
            "mode",
            "payload_sha256",
            "content_sha256",
        }
        or set(invocation)
        != {
            "started_at_utc",
            "finished_at_utc",
            "wall_seconds",
            "command",
            "content_sha256",
        }
        or set(result_record)
        != {
            "recorded_at_utc",
            "addressed_payload_sha256",
            "worker_process_record_sha256",
            "worker_invocation_sha256",
            "content_sha256",
        }
        or result_record_sha256 != _canonical_sha256(result_record_without_digest)
        or process_record_sha256 != _canonical_sha256(process_record_without_digest)
        or invocation_sha256 != _canonical_sha256(invocation_without_digest)
        or process_record.get("payload_sha256") != _canonical_sha256(operation_payload)
        or process_record.get("operation") not in {"generate", "profile"}
        or process_record.get("mode") not in EXECUTION_MODES
        or process_record.get("mode") != operation_payload.get("mode")
        or isinstance(process_wall_seconds, bool)
        or not isinstance(process_wall_seconds, (float, int))
        or not math.isfinite(float(process_wall_seconds))
        or float(process_wall_seconds) <= 0.0
        or isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
        or isinstance(invocation_wall_seconds, bool)
        or not isinstance(invocation_wall_seconds, (float, int))
        or not math.isfinite(float(invocation_wall_seconds))
        or float(invocation_wall_seconds) <= 0.0
        or worker_command != invocation_command
        or result_record.get("addressed_payload_sha256") != _canonical_sha256(payload)
        or result_record.get("worker_process_record_sha256") != process_record_sha256
        or result_record.get("worker_invocation_sha256") != invocation_sha256
        or not _utc_timestamps_nondecreasing(
            invocation.get("started_at_utc"),
            process_record.get("started_at_utc"),
            process_record.get("finished_at_utc"),
            invocation.get("finished_at_utc"),
            result_record.get("recorded_at_utc"),
        )
    ):
        raise HarnessError("worker result evidence failed its content-address contract")
    evidence = {
        "kind": PRESERVED_WORKER_RESULT_KIND,
        "schema_version": PRESERVED_WORKER_RESULT_SCHEMA,
        "payload": payload,
        "worker_result_record": dict(result_record),
    }
    evidence["content_sha256"] = _canonical_sha256(evidence)
    return evidence


def _retained_profile_worker_result_record(
    sample: Mapping[str, object],
    *,
    upstream_result_record: Mapping[str, object],
) -> dict[str, object]:
    invocation = sample.get("worker_invocation")
    process_record = sample.get("worker_process_record")
    if not isinstance(invocation, Mapping) or not isinstance(
        process_record,
        Mapping,
    ):
        raise HarnessError("retained profile sample has incomplete worker provenance")
    invocation_sha256 = _required_sha256(
        invocation.get("content_sha256"),
        label="retained worker invocation",
    )
    process_record_sha256 = _required_sha256(
        process_record.get("content_sha256"),
        label="retained worker process record",
    )
    upstream_sha256 = _required_sha256(
        upstream_result_record.get("content_sha256"),
        label="upstream worker result record",
    )
    recorded_at = upstream_result_record.get("recorded_at_utc")
    if not _is_utc_timestamp(recorded_at):
        raise HarnessError("upstream worker result record has an invalid timestamp")
    if not _utc_timestamps_nondecreasing(
        invocation.get("started_at_utc"),
        process_record.get("started_at_utc"),
        process_record.get("finished_at_utc"),
        invocation.get("finished_at_utc"),
        recorded_at,
    ):
        raise HarnessError("retained profile worker timestamps are not chronological")
    record = {
        "kind": RETAINED_WORKER_RESULT_KIND,
        "schema_version": RETAINED_WORKER_RESULT_SCHEMA,
        "recorded_at_utc": recorded_at,
        "addressed_payload_sha256": _canonical_sha256(
            _retained_profile_sample_evidence(sample)
        ),
        "upstream_worker_result_record_sha256": upstream_sha256,
        "worker_process_record_sha256": process_record_sha256,
        "worker_invocation_sha256": invocation_sha256,
    }
    record["content_sha256"] = _canonical_sha256(record)
    return record


def _aggregate_profile_workers(
    schedule: Mapping[str, object],
    worker_results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    raw_entries = schedule.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != len(worker_results):
        raise HarnessError("cannot aggregate an incomplete profile worker schedule")
    lane_contracts: dict[str, dict[str, object]] = {}
    lane_samples: dict[str, dict[int, list[dict[str, object]]]] = {}
    for raw_entry, worker in zip(raw_entries, worker_results, strict=True):
        if not isinstance(raw_entry, Mapping):
            raise HarnessError("profile schedule contains an invalid entry")
        mode = raw_entry.get("mode")
        batch_size = raw_entry.get("batch_size")
        schedule_index = raw_entry.get("schedule_index")
        round_index = raw_entry.get("round")
        if (
            not isinstance(mode, str)
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or isinstance(schedule_index, bool)
            or not isinstance(schedule_index, int)
            or isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or worker.get("mode") != mode
            or worker.get("schedule_index") != schedule_index
            or worker.get("schedule_round") != round_index
        ):
            raise HarnessError("profile worker result does not match its schedule slot")
        measurements = worker.get("profiles")
        if not isinstance(measurements, list) or len(measurements) != 1:
            raise HarnessError(
                "profile worker must return exactly one batch measurement"
            )
        measurement = measurements[0]
        if (
            not isinstance(measurement, Mapping)
            or measurement.get("batch_size") != batch_size
        ):
            raise HarnessError("profile worker returned the wrong batch measurement")
        verification = worker.get("pre_timing_verification")
        if not isinstance(verification, Mapping):
            raise HarnessError("profile worker returned no pre-timing verification")
        semantic_identity = verification.get("artifact_semantic_identity")
        if not isinstance(semantic_identity, Mapping):
            raise HarnessError("profile worker returned no artifact semantic identity")
        semantic_identity_sha256 = verification.get("artifact_semantic_identity_sha256")
        if semantic_identity_sha256 != _canonical_sha256(semantic_identity):
            raise HarnessError(
                "profile worker artifact semantic identity is not content-addressed"
            )
        contract = {
            "process_id": worker.get("process_id"),
            "process_expression": worker.get("process_expression"),
            "selector_contract": worker.get("selector_contract"),
            "validation": worker.get("validation"),
            "artifact_semantic_identity": dict(semantic_identity),
            "artifact_semantic_identity_sha256": semantic_identity_sha256,
        }
        existing_contract = lane_contracts.setdefault(mode, contract)
        _assert_identity_unchanged(
            f"{mode} repeated profile semantic contract",
            existing_contract,
            contract,
        )
        worker_process_record = worker.get("worker_process_record")
        upstream_result_record = worker.get("worker_result_record")
        if not isinstance(worker_process_record, Mapping) or not isinstance(
            upstream_result_record,
            Mapping,
        ):
            raise HarnessError("profile worker returned incomplete result provenance")
        sample = {
            "schedule_index": schedule_index,
            "round": round_index,
            "worker_command": worker.get("worker_command"),
            "worker_invocation": worker.get("worker_invocation"),
            "worker_process_record": dict(worker_process_record),
            "peak_rss_after_cold_load": worker.get("peak_rss_after_cold_load"),
            "peak_rss_after_profile": worker.get("peak_rss_after_profile"),
            "pre_timing_verification": _compact_profile_verification(verification),
            "post_timing_loaded_runtime_artifact": worker.get(
                "post_timing_loaded_runtime_artifact"
            ),
            "cold_load_seconds": worker.get("cold_load_seconds"),
            "lane_contract_sha256": _canonical_sha256(contract),
            "timing_configuration": worker.get("timing_configuration"),
            "worker_measurement": dict(measurement),
            "internal_sample_count": measurement.get("sample_count"),
            "repetitions_per_sample": measurement.get("repetitions_per_sample"),
            "evaluation_count": measurement.get("evaluation_count"),
            "evaluated_point_count": measurement.get("evaluated_point_count"),
            "wall_seconds_per_point": measurement.get("wall_seconds_per_point"),
            "inner_native_wall_blocks": measurement.get("inner_native_wall_blocks"),
            "timing_sources": measurement.get("timing_sources"),
            "environment": measurement.get("environment"),
            "interrupted": measurement.get("interrupted"),
        }
        retained_result_record = _retained_profile_worker_result_record(
            sample,
            upstream_result_record=upstream_result_record,
        )
        sample["worker_result_record"] = retained_result_record
        if not isinstance(raw_entry, dict):
            raise HarnessError("profile schedule entry cannot retain worker provenance")
        raw_entry["worker_result_record"] = dict(retained_result_record)
        raw_entry["worker_result_sha256"] = retained_result_record[
            "addressed_payload_sha256"
        ]
        lane_samples.setdefault(mode, {}).setdefault(batch_size, []).append(sample)
    profiles: dict[str, dict[str, Any]] = {}
    for mode, contract in lane_contracts.items():
        aggregated_measurements: list[dict[str, object]] = []
        for batch_size, samples in sorted(lane_samples[mode].items()):
            values: list[float] = []
            cold_load_values: list[float] = []
            cold_load_rss_values: list[int] = []
            profiled_rss_values: list[int] = []
            for sample in samples:
                value = sample.get("wall_seconds_per_point")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise HarnessError(
                        f"{mode}/{batch_size} worker returned invalid wall timing"
                    )
                values.append(float(value))
                cold_load = sample.get("cold_load_seconds")
                peak_after_load = sample.get("peak_rss_after_cold_load")
                peak_after_profile = sample.get("peak_rss_after_profile")
                if (
                    isinstance(cold_load, bool)
                    or not isinstance(cold_load, (float, int))
                    or not math.isfinite(float(cold_load))
                    or float(cold_load) <= 0.0
                    or not _valid_resource_peak(peak_after_load)
                    or not _valid_resource_peak(peak_after_profile)
                ):
                    raise HarnessError(
                        f"{mode}/{batch_size} worker returned invalid cold-load "
                        "resource evidence"
                    )
                assert isinstance(peak_after_load, Mapping)
                assert isinstance(peak_after_profile, Mapping)
                cold_load_values.append(float(cold_load))
                cold_load_rss_values.append(
                    int(peak_after_load["observed_lower_bound_bytes"])
                )
                profiled_rss_values.append(
                    int(peak_after_profile["observed_lower_bound_bytes"])
                )
            median = statistics.median(values)
            mad = statistics.median(abs(value - median) for value in values)
            cold_load_median = statistics.median(cold_load_values)
            cold_load_mad = statistics.median(
                abs(value - cold_load_median) for value in cold_load_values
            )
            aggregated_measurements.append(
                {
                    "batch_size": batch_size,
                    "sample_count": len(samples),
                    "subprocess_sample_count": len(samples),
                    "wall_seconds_per_point": median,
                    "wall_seconds_per_point_median": median,
                    "wall_seconds_per_point_mad": mad,
                    "statistics_contract": "subprocess-median-and-raw-mad-v1",
                    "cold_load_seconds_median": cold_load_median,
                    "cold_load_seconds_mad": cold_load_mad,
                    "cold_load_peak_rss_bytes_median": statistics.median(
                        cold_load_rss_values
                    ),
                    "profiled_peak_rss_bytes_median": statistics.median(
                        profiled_rss_values
                    ),
                    "resource_statistics_contract": (
                        "subprocess-median-and-raw-mad-v1"
                    ),
                    "subprocess_samples": samples,
                    "interrupted": any(
                        sample.get("interrupted") is not False for sample in samples
                    ),
                }
            )
        profiles[mode] = {
            "mode": mode,
            **contract,
            "profiles": aggregated_measurements,
        }
    return profiles


def _profile_measurement_contract(
    arguments: argparse.Namespace,
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    profile_schedule: Mapping[str, object] | None,
    required_modes: Sequence[str] | None = None,
) -> dict[str, object]:
    selected_modes = tuple(
        arguments.modes if required_modes is None else required_modes
    )
    lanes: dict[str, dict[str, object]] = {}
    root_process_contracts = [
        (
            profile.get("process_id"),
            (
                " ".join(process_expression.split()).casefold()
                if isinstance(
                    (process_expression := profile.get("process_expression")),
                    str,
                )
                else None
            ),
        )
        for mode in selected_modes
        if (profile := profiles.get(mode)) is not None
    ]
    root_processes_match = bool(root_process_contracts) and all(
        contract == root_process_contracts[0] for contract in root_process_contracts[1:]
    )
    schedule_contract = _profile_schedule_contract(
        profile_schedule,
        modes=arguments.modes,
        batch_sizes=arguments.batch_size,
        subprocess_samples=arguments.subprocess_samples,
    )
    raw_schedule_entries = (
        profile_schedule.get("entries")
        if isinstance(profile_schedule, Mapping)
        else None
    )
    scheduled_entries = (
        {
            entry["schedule_index"]: entry
            for entry in raw_schedule_entries
            if isinstance(entry, Mapping)
            and isinstance(entry.get("schedule_index"), int)
        }
        if isinstance(raw_schedule_entries, list)
        else {}
    )
    passes = (
        schedule_contract["passes"] is True
        and arguments.subprocess_samples >= MIN_AUTHORITATIVE_SAMPLES
        and arguments.minimum_samples >= MIN_AUTHORITATIVE_SAMPLES
        and arguments.warmup_runs >= 2
        and arguments.target_runtime >= 5.0
    )
    for mode in selected_modes:
        profile = profiles.get(mode)
        if profile is None:
            lanes[mode] = {
                "passes": False,
                "observed_batch_sizes": [],
                "missing_batch_sizes": list(DEFAULT_BATCH_SIZES),
                "errors": ["profile lane is missing"],
            }
            passes = False
            continue
        raw_measurements = profile.get("profiles")
        errors: list[str] = []
        observed: list[int] = []
        if profile.get("mode") != mode:
            errors.append("profile lane mode does not match its inventory key")
        process_id = profile.get("process_id")
        process_expression = profile.get("process_expression")
        semantic_identity = profile.get("artifact_semantic_identity")
        execution_ordering = (
            semantic_identity.get("execution_schedule_ordering")
            if isinstance(semantic_identity, Mapping)
            else None
        )
        runtime_process_contract = (
            execution_ordering.get("runtime_process_contract")
            if isinstance(execution_ordering, Mapping)
            else None
        )
        expected_process = " ".join(_selected_process(arguments).split()).casefold()
        if (
            not root_processes_match
            or not isinstance(process_id, str)
            or not process_id
            or not isinstance(process_expression, str)
            or " ".join(process_expression.split()).casefold() != expected_process
            or not isinstance(runtime_process_contract, Mapping)
            or runtime_process_contract.get("id") != process_id
        ):
            errors.append(
                "profile lane process identity is inconsistent with requested process"
            )
        if not _profile_selector_contract_matches(
            arguments,
            profile.get("selector_contract"),
            semantic_identity,
        ):
            errors.append("profile lane selector workload contract is invalid")
        validation = profile.get("validation")
        fixture_contract = (
            _validation_fixture_contract(validation, mode=mode)
            if isinstance(validation, Mapping)
            else None
        )
        expected_fixture_sha256 = (
            fixture_contract.get("points_sha256")
            if isinstance(fixture_contract, Mapping)
            else None
        )
        profile_semantic_sha256 = profile.get("artifact_semantic_identity_sha256")
        lane_contract_payload = {
            field: profile.get(field)
            for field in (
                "process_id",
                "process_expression",
                "selector_contract",
                "validation",
                "artifact_semantic_identity",
                "artifact_semantic_identity_sha256",
            )
        }
        expected_lane_contract_sha256 = _canonical_sha256(lane_contract_payload)
        if not isinstance(expected_fixture_sha256, str):
            errors.append("profile has no shared validation fixture identity")
        if (
            not isinstance(profile_semantic_sha256, str)
            or _SHA256_PATTERN.fullmatch(profile_semantic_sha256) is None
        ):
            errors.append("profile has no artifact semantic identity digest")
        if not isinstance(raw_measurements, list):
            errors.append("profile measurements are missing")
            raw_measurements = []
        for raw in raw_measurements:
            if not isinstance(raw, Mapping):
                errors.append("profile measurement is not an object")
                continue
            batch_size = raw.get("batch_size")
            sample_count = raw.get("subprocess_sample_count")
            headline_sample_count = raw.get("sample_count")
            wall_seconds = raw.get("wall_seconds_per_point")
            median = raw.get("wall_seconds_per_point_median")
            mad = raw.get("wall_seconds_per_point_mad")
            statistics_contract = raw.get("statistics_contract")
            subprocess_samples = raw.get("subprocess_samples")
            interrupted = raw.get("interrupted")
            if (
                isinstance(batch_size, bool)
                or not isinstance(batch_size, int)
                or batch_size <= 0
            ):
                errors.append("profile measurement has an invalid batch size")
                continue
            if batch_size in observed:
                errors.append(f"profile batch {batch_size} is duplicated")
                continue
            observed.append(batch_size)
            if (
                isinstance(sample_count, bool)
                or not isinstance(sample_count, int)
                or sample_count < MIN_AUTHORITATIVE_SAMPLES
                or sample_count != arguments.subprocess_samples
                or not _is_exact_int(headline_sample_count, sample_count)
            ):
                errors.append(
                    f"profile batch {batch_size} has too few independent subprocess "
                    "samples"
                )
            if (
                isinstance(wall_seconds, bool)
                or not isinstance(wall_seconds, (float, int))
                or not math.isfinite(float(wall_seconds))
                or float(wall_seconds) <= 0.0
            ):
                errors.append(f"profile batch {batch_size} has an invalid wall time")
            if (
                isinstance(median, bool)
                or not isinstance(median, (float, int))
                or not math.isfinite(float(median))
                or float(median) <= 0.0
                or wall_seconds != median
                or isinstance(mad, bool)
                or not isinstance(mad, (float, int))
                or not math.isfinite(float(mad))
                or float(mad) < 0.0
                or statistics_contract != "subprocess-median-and-raw-mad-v1"
            ):
                errors.append(
                    f"profile batch {batch_size} lacks an explicit median/MAD"
                )
            if not isinstance(subprocess_samples, list):
                errors.append(
                    f"profile batch {batch_size} has no subprocess sample inventory"
                )
                subprocess_samples = []
            if len(subprocess_samples) != sample_count:
                errors.append(
                    f"profile batch {batch_size} subprocess inventory is incomplete"
                )
            sample_values: list[float] = []
            schedule_indices: list[int] = []
            for sample_index, sample in enumerate(subprocess_samples):
                if not isinstance(sample, Mapping):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} is invalid"
                    )
                    continue
                internal_count = sample.get("internal_sample_count")
                repetitions = sample.get("repetitions_per_sample")
                evaluation_count = sample.get("evaluation_count")
                evaluated_point_count = sample.get("evaluated_point_count")
                sample_wall = sample.get("wall_seconds_per_point")
                raw_blocks = sample.get("inner_native_wall_blocks")
                worker_measurement = sample.get("worker_measurement")
                worker_process_record = sample.get("worker_process_record")
                worker_result_record = sample.get("worker_result_record")
                sources = sample.get("timing_sources")
                environment = sample.get("environment")
                timing_configuration = sample.get("timing_configuration")
                cold_load_seconds = sample.get("cold_load_seconds")
                peak_rss_after_cold_load = sample.get("peak_rss_after_cold_load")
                peak_rss_after_profile = sample.get("peak_rss_after_profile")
                schedule_index = sample.get("schedule_index")
                verification = sample.get("pre_timing_verification")
                post_timing_loaded_artifact = sample.get(
                    "post_timing_loaded_runtime_artifact"
                )
                sample_round = sample.get("round")
                invocation = sample.get("worker_invocation")
                worker_command = sample.get("worker_command")
                if (
                    isinstance(cold_load_seconds, bool)
                    or not isinstance(cold_load_seconds, (float, int))
                    or not math.isfinite(float(cold_load_seconds))
                    or float(cold_load_seconds) <= 0.0
                    or not _valid_resource_peak(peak_rss_after_cold_load)
                    or not _valid_resource_peak(peak_rss_after_profile)
                ):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} lacks cold-load/resource evidence"
                    )
                if (
                    isinstance(sample_round, bool)
                    or not isinstance(sample_round, int)
                    or sample_round < 0
                ):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} has an invalid schedule round"
                    )
                try:
                    retained_worker_command = _validated_command_identity(
                        worker_command
                    )
                    invocation_command = (
                        _validated_command_identity(invocation.get("command"))
                        if isinstance(invocation, Mapping)
                        else None
                    )
                except HarnessError:
                    retained_worker_command = None
                    invocation_command = None
                if (
                    retained_worker_command is None
                    or invocation_command is None
                    or retained_worker_command != invocation_command
                ):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} has inconsistent worker command provenance"
                    )
                if sample.get("lane_contract_sha256") != (
                    expected_lane_contract_sha256
                ):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} is not bound to its lane contract"
                    )
                if not isinstance(verification, Mapping):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} has no artifact verification"
                    )
                else:
                    expected_identities = verification.get("expected")
                    observed_identities = verification.get("observed")
                    effective_contract = verification.get("effective_contract")
                    expected_effective_contract = {
                        "execution_mode": mode,
                        "backend": "jit",
                        "jit_optimization_level": (
                            _expected_effective_jit_optimization_level(
                                arguments,
                                mode=mode,
                            )
                        ),
                        "color_accuracy": "lc",
                        "lc_flow_layout": arguments.lc_flow_layout,
                    }
                    if (
                        verification.get("kind") != WORKER_VERIFICATION_KIND
                        or not _is_exact_int(
                            verification.get("schema_version"),
                            WORKER_VERIFICATION_SCHEMA,
                        )
                        or not _is_utc_timestamp(verification.get("verified_at_utc"))
                        or verification.get("artifact_semantic_identity_sha256")
                        != profile_semantic_sha256
                        or not isinstance(expected_identities, Mapping)
                        or not isinstance(observed_identities, Mapping)
                        or expected_identities.get("artifact_semantic_identity_sha256")
                        != profile_semantic_sha256
                        or observed_identities.get("artifact_semantic_identity_sha256")
                        != profile_semantic_sha256
                        or not isinstance(effective_contract, Mapping)
                        or _canonical_sha256(effective_contract)
                        != _canonical_sha256(expected_effective_contract)
                    ):
                        errors.append(
                            f"profile batch {batch_size} subprocess sample "
                            f"{sample_index} artifact identity is not bound"
                        )
                    else:
                        try:
                            _validate_loaded_runtime_artifact_verification(
                                verification.get("loaded_runtime_artifact"),
                                expected_artifact_id=expected_identities.get(
                                    "artifact_id"
                                ),
                                phase="after-native-load-before-timing",
                            )
                            _validate_loaded_runtime_artifact_verification(
                                post_timing_loaded_artifact,
                                expected_artifact_id=expected_identities.get(
                                    "artifact_id"
                                ),
                                phase="after-timing",
                            )
                        except HarnessError:
                            errors.append(
                                f"profile batch {batch_size} subprocess sample "
                                f"{sample_index} loaded artifact identity is not bound"
                            )
                if (
                    isinstance(internal_count, bool)
                    or not isinstance(internal_count, int)
                    or internal_count < MIN_AUTHORITATIVE_SAMPLES
                    or internal_count < arguments.minimum_samples
                ):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} has too few native timing blocks"
                    )
                if (
                    isinstance(repetitions, bool)
                    or not isinstance(repetitions, int)
                    or repetitions <= 0
                    or isinstance(evaluation_count, bool)
                    or not isinstance(evaluation_count, int)
                    or evaluation_count <= 0
                    or isinstance(evaluated_point_count, bool)
                    or not isinstance(evaluated_point_count, int)
                    or evaluated_point_count <= 0
                    or (
                        isinstance(internal_count, int)
                        and evaluation_count != internal_count * repetitions
                    )
                    or evaluated_point_count != evaluation_count * batch_size
                ):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} has invalid repetition/evaluation counts"
                    )
                if (
                    not isinstance(worker_measurement, Mapping)
                    or not _is_exact_int(
                        worker_measurement.get("batch_size"),
                        batch_size,
                    )
                    or not isinstance(internal_count, int)
                    or not _is_exact_int(
                        worker_measurement.get("sample_count"),
                        internal_count,
                    )
                    or not isinstance(repetitions, int)
                    or not _is_exact_int(
                        worker_measurement.get("repetitions_per_sample"),
                        repetitions,
                    )
                    or not isinstance(evaluation_count, int)
                    or not _is_exact_int(
                        worker_measurement.get("evaluation_count"),
                        evaluation_count,
                    )
                    or not isinstance(evaluated_point_count, int)
                    or not _is_exact_int(
                        worker_measurement.get("evaluated_point_count"),
                        evaluated_point_count,
                    )
                    or worker_measurement.get("wall_seconds_per_point") != sample_wall
                    or worker_measurement.get("inner_native_wall_blocks") != raw_blocks
                    or worker_measurement.get("timing_sources") != sources
                    or worker_measurement.get("environment") != environment
                    or worker_measurement.get("interrupted")
                    != sample.get("interrupted")
                ):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} disagrees with its retained worker measurement"
                    )
                if not isinstance(worker_process_record, Mapping):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} has no worker process record"
                    )
                else:
                    process_record_without_digest = dict(worker_process_record)
                    process_record_digest = process_record_without_digest.pop(
                        "content_sha256",
                        None,
                    )
                    process_wall_seconds = worker_process_record.get("wall_seconds")
                    process_id = worker_process_record.get("process_id")
                    process_payload_sha256 = worker_process_record.get("payload_sha256")
                    if (
                        process_record_digest
                        != _canonical_sha256(process_record_without_digest)
                        or not _is_utc_timestamp(
                            worker_process_record.get("started_at_utc")
                        )
                        or not _is_utc_timestamp(
                            worker_process_record.get("finished_at_utc")
                        )
                        or not _utc_timestamps_nondecreasing(
                            worker_process_record.get("started_at_utc"),
                            (
                                verification.get("verified_at_utc")
                                if isinstance(verification, Mapping)
                                else None
                            ),
                            worker_process_record.get("finished_at_utc"),
                        )
                        or isinstance(process_wall_seconds, bool)
                        or not isinstance(process_wall_seconds, (float, int))
                        or not math.isfinite(float(process_wall_seconds))
                        or float(process_wall_seconds) <= 0.0
                        or isinstance(process_id, bool)
                        or not isinstance(process_id, int)
                        or process_id <= 0
                        or worker_process_record.get("operation") != "profile"
                        or worker_process_record.get("mode") != mode
                        or not isinstance(process_payload_sha256, str)
                        or _SHA256_PATTERN.fullmatch(process_payload_sha256) is None
                    ):
                        errors.append(
                            f"profile batch {batch_size} subprocess sample "
                            f"{sample_index} worker process record is invalid"
                        )
                if not isinstance(worker_result_record, Mapping):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} has no worker result record"
                    )
                else:
                    result_record_without_digest = dict(worker_result_record)
                    result_record_digest = result_record_without_digest.pop(
                        "content_sha256",
                        None,
                    )
                    invocation = sample.get("worker_invocation")
                    process_record_sha256 = (
                        worker_process_record.get("content_sha256")
                        if isinstance(worker_process_record, Mapping)
                        else None
                    )
                    invocation_sha256 = (
                        invocation.get("content_sha256")
                        if isinstance(invocation, Mapping)
                        else None
                    )
                    if (
                        worker_result_record.get("kind") != RETAINED_WORKER_RESULT_KIND
                        or not _is_exact_int(
                            worker_result_record.get("schema_version"),
                            RETAINED_WORKER_RESULT_SCHEMA,
                        )
                        or result_record_digest
                        != _canonical_sha256(result_record_without_digest)
                        or not _is_utc_timestamp(
                            worker_result_record.get("recorded_at_utc")
                        )
                        or not isinstance(invocation, Mapping)
                        or not _utc_timestamps_nondecreasing(
                            invocation.get("started_at_utc"),
                            (
                                worker_process_record.get("started_at_utc")
                                if isinstance(worker_process_record, Mapping)
                                else None
                            ),
                            (
                                worker_process_record.get("finished_at_utc")
                                if isinstance(worker_process_record, Mapping)
                                else None
                            ),
                            invocation.get("finished_at_utc"),
                            worker_result_record.get("recorded_at_utc"),
                        )
                        or worker_result_record.get("addressed_payload_sha256")
                        != _canonical_sha256(_retained_profile_sample_evidence(sample))
                        or worker_result_record.get("worker_process_record_sha256")
                        != process_record_sha256
                        or worker_result_record.get("worker_invocation_sha256")
                        != invocation_sha256
                        or not isinstance(
                            worker_result_record.get(
                                "upstream_worker_result_record_sha256"
                            ),
                            str,
                        )
                        or _SHA256_PATTERN.fullmatch(
                            worker_result_record["upstream_worker_result_record_sha256"]
                        )
                        is None
                    ):
                        errors.append(
                            f"profile batch {batch_size} subprocess sample "
                            f"{sample_index} worker result address is invalid"
                        )
                if (
                    isinstance(sample_wall, bool)
                    or not isinstance(sample_wall, (float, int))
                    or not math.isfinite(float(sample_wall))
                    or float(sample_wall) <= 0.0
                ):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} has invalid wall timing"
                    )
                else:
                    sample_values.append(float(sample_wall))
                if (
                    not isinstance(sources, Mapping)
                    or sources.get("wall") != "runtime_core_repeated_wall_time"
                    or not isinstance(environment, Mapping)
                    or environment.get("wall_time_source")
                    != "runtime_core_repeated_wall_time"
                    or environment.get("wall_time_sample_pass")
                    != "runtime._benchmark_f64_wall_time"
                ):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} is not an unprofiled native-wall sample"
                    )
                if (
                    not isinstance(timing_configuration, Mapping)
                    or not _is_exact_int(
                        timing_configuration.get("minimum_internal_samples"),
                        arguments.minimum_samples,
                    )
                    or not _is_exact_int(
                        timing_configuration.get("warmup_runs"),
                        arguments.warmup_runs,
                    )
                    or isinstance(
                        timing_configuration.get("target_runtime_seconds"),
                        bool,
                    )
                    or not isinstance(
                        timing_configuration.get("target_runtime_seconds"),
                        (float, int),
                    )
                    or not math.isfinite(
                        float(timing_configuration["target_runtime_seconds"])
                    )
                    or float(timing_configuration["target_runtime_seconds"]) <= 0.0
                    or timing_configuration.get("target_runtime_seconds")
                    != arguments.target_runtime
                    or arguments.warmup_runs < 1
                ):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} is not warmed"
                    )
                if sample.get("interrupted") is not False:
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} was interrupted"
                    )
                if not isinstance(raw_blocks, Mapping):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} has no raw native-wall blocks"
                    )
                else:
                    raw_inventory = raw_blocks.get("blocks")
                    raw_block_count = raw_blocks.get("block_count")
                    raw_repetitions = raw_blocks.get("repetitions_per_block")
                    raw_evaluation_count = raw_blocks.get("evaluation_count")
                    raw_evaluated_point_count = raw_blocks.get("evaluated_point_count")
                    raw_wall_median = raw_blocks.get("wall_seconds_per_point_median")
                    raw_wall_mad = raw_blocks.get("wall_seconds_per_point_mad")
                    raw_minimum_duration = raw_blocks.get(
                        "minimum_native_wall_seconds"
                    )
                    raw_observed_duration = raw_blocks.get(
                        "observed_native_wall_seconds"
                    )
                    raw_caller_duration = raw_blocks.get(
                        "observed_caller_elapsed_seconds"
                    )
                    raw_calibration = raw_blocks.get("calibration")
                    calibration_runner_samples = (
                        raw_calibration.get("benchmark_runner_sample_count")
                        if isinstance(raw_calibration, Mapping)
                        else None
                    )
                    calibration_runner_repetitions = (
                        raw_calibration.get(
                            "benchmark_runner_repetitions_per_sample"
                        )
                        if isinstance(raw_calibration, Mapping)
                        else None
                    )
                    calibration_runner_total = (
                        raw_calibration.get("benchmark_runner_total_repetitions")
                        if isinstance(raw_calibration, Mapping)
                        else None
                    )
                    calibration_runner_wall = (
                        raw_calibration.get(
                            "benchmark_runner_wall_seconds_per_point"
                        )
                        if isinstance(raw_calibration, Mapping)
                        else None
                    )
                    calibration_headroom = (
                        raw_calibration.get("duration_headroom_factor")
                        if isinstance(raw_calibration, Mapping)
                        else None
                    )
                    if (
                        raw_blocks.get("kind") != RAW_NATIVE_WALL_BLOCK_KIND
                        or not _is_exact_int(
                            raw_blocks.get("schema_version"),
                            RAW_NATIVE_WALL_BLOCK_SCHEMA,
                        )
                        or raw_blocks.get("measurement_contract")
                        != RAW_NATIVE_WALL_MEASUREMENT_CONTRACT
                        or raw_blocks.get("source")
                        != "runtime._benchmark_f64_wall_time"
                        or raw_blocks.get("fixture_points_sha256")
                        != expected_fixture_sha256
                        or isinstance(raw_repetitions, bool)
                        or not isinstance(raw_repetitions, int)
                        or raw_repetitions <= 0
                        or raw_repetitions != repetitions
                        or isinstance(raw_evaluation_count, bool)
                        or not isinstance(raw_evaluation_count, int)
                        or raw_evaluation_count <= 0
                        or raw_evaluation_count != evaluation_count
                        or isinstance(raw_evaluated_point_count, bool)
                        or not isinstance(raw_evaluated_point_count, int)
                        or raw_evaluated_point_count <= 0
                        or raw_evaluated_point_count != evaluated_point_count
                        or isinstance(raw_block_count, bool)
                        or not isinstance(raw_block_count, int)
                        or raw_block_count < MIN_AUTHORITATIVE_SAMPLES
                        or raw_block_count != internal_count
                        or raw_evaluation_count != raw_block_count * raw_repetitions
                        or raw_evaluated_point_count
                        != raw_evaluation_count * batch_size
                        or isinstance(raw_wall_median, bool)
                        or not isinstance(raw_wall_median, (float, int))
                        or not math.isfinite(float(raw_wall_median))
                        or float(raw_wall_median) <= 0.0
                        or isinstance(raw_wall_mad, bool)
                        or not isinstance(raw_wall_mad, (float, int))
                        or not math.isfinite(float(raw_wall_mad))
                        or float(raw_wall_mad) < 0.0
                        or not isinstance(raw_inventory, list)
                        or not raw_inventory
                        or len(raw_inventory) != raw_block_count
                        or raw_blocks.get("blocks_sha256")
                        != _canonical_sha256(raw_inventory)
                        or raw_wall_median != sample_wall
                        or isinstance(raw_minimum_duration, bool)
                        or not isinstance(raw_minimum_duration, (float, int))
                        or not math.isfinite(float(raw_minimum_duration))
                        or raw_minimum_duration != arguments.target_runtime
                        or isinstance(raw_observed_duration, bool)
                        or not isinstance(raw_observed_duration, (float, int))
                        or not math.isfinite(float(raw_observed_duration))
                        or float(raw_observed_duration)
                        < float(raw_minimum_duration)
                        or isinstance(raw_caller_duration, bool)
                        or not isinstance(raw_caller_duration, (float, int))
                        or not math.isfinite(float(raw_caller_duration))
                        or float(raw_caller_duration) <= 0.0
                        or float(raw_caller_duration)
                        < float(raw_observed_duration)
                        or raw_blocks.get("minimum_duration_satisfied") is not True
                        or not isinstance(raw_calibration, Mapping)
                        or raw_calibration.get("kind")
                        != "benchmark-runner-wall-rate-calibration"
                        or not _is_exact_int(
                            raw_calibration.get("schema_version"),
                            1,
                        )
                        or isinstance(calibration_runner_samples, bool)
                        or not isinstance(calibration_runner_samples, int)
                        or calibration_runner_samples <= 0
                        or isinstance(calibration_runner_repetitions, bool)
                        or not isinstance(calibration_runner_repetitions, int)
                        or calibration_runner_repetitions <= 0
                        or not _is_exact_int(
                            calibration_runner_total,
                            calibration_runner_samples
                            * calibration_runner_repetitions,
                        )
                        or isinstance(calibration_runner_wall, bool)
                        or not isinstance(calibration_runner_wall, (float, int))
                        or not math.isfinite(float(calibration_runner_wall))
                        or float(calibration_runner_wall) <= 0.0
                        or isinstance(calibration_headroom, bool)
                        or not isinstance(calibration_headroom, (float, int))
                        or not math.isfinite(float(calibration_headroom))
                        or float(calibration_headroom) != 1.02
                        or not _is_exact_int(
                            raw_calibration.get(
                                "requested_minimum_block_count"
                            ),
                            max(
                                MIN_AUTHORITATIVE_SAMPLES,
                                arguments.minimum_samples,
                            ),
                        )
                        or not _is_exact_int(
                            raw_calibration.get(
                                "scaled_repetitions_per_block"
                            ),
                            raw_repetitions,
                        )
                        or not _is_exact_int(
                            raw_calibration.get(
                                "preceded_by_benchmark_runner_warmup_runs"
                            ),
                            arguments.warmup_runs,
                        )
                    ):
                        errors.append(
                            f"profile batch {batch_size} subprocess sample "
                            f"{sample_index} raw native-wall inventory is invalid"
                        )
                    elif any(
                        not isinstance(block, Mapping)
                        or isinstance(block.get("block_index"), bool)
                        or not isinstance(block.get("block_index"), int)
                        or block.get("block_index") != block_index
                        or not _is_utc_timestamp(block.get("started_at_utc"))
                        or not _is_utc_timestamp(block.get("finished_at_utc"))
                        or not _utc_timestamps_nondecreasing(
                            (
                                worker_process_record.get("started_at_utc")
                                if isinstance(worker_process_record, Mapping)
                                else None
                            ),
                            block.get("started_at_utc"),
                            block.get("finished_at_utc"),
                            (
                                worker_process_record.get("finished_at_utc")
                                if isinstance(worker_process_record, Mapping)
                                else None
                            ),
                        )
                        or (
                            block_index == 0
                            and not _utc_timestamps_nondecreasing(
                                (
                                    verification.get("verified_at_utc")
                                    if isinstance(verification, Mapping)
                                    else None
                                ),
                                block.get("started_at_utc"),
                            )
                        )
                        or (
                            block_index > 0
                            and (
                                not isinstance(
                                    raw_inventory[block_index - 1],
                                    Mapping,
                                )
                                or not _utc_timestamps_nondecreasing(
                                    raw_inventory[block_index - 1].get(
                                        "finished_at_utc"
                                    ),
                                    block.get("started_at_utc"),
                                )
                            )
                        )
                        or isinstance(block.get("caller_elapsed_seconds"), bool)
                        or not isinstance(
                            block.get("caller_elapsed_seconds"),
                            (float, int),
                        )
                        or not math.isfinite(float(block["caller_elapsed_seconds"]))
                        or float(block["caller_elapsed_seconds"]) <= 0.0
                        or isinstance(block.get("native_wall_seconds"), bool)
                        or not isinstance(
                            block.get("native_wall_seconds"),
                            (float, int),
                        )
                        or not math.isfinite(float(block["native_wall_seconds"]))
                        or float(block["native_wall_seconds"]) <= 0.0
                        or float(block["caller_elapsed_seconds"])
                        < float(block["native_wall_seconds"])
                        or isinstance(
                            block.get("wall_seconds_per_point"),
                            bool,
                        )
                        or not isinstance(
                            block.get("wall_seconds_per_point"),
                            (float, int),
                        )
                        or not math.isfinite(float(block["wall_seconds_per_point"]))
                        or float(block["wall_seconds_per_point"]) <= 0.0
                        or not math.isclose(
                            float(block["wall_seconds_per_point"]),
                            float(block["native_wall_seconds"])
                            / (repetitions * batch_size),
                            rel_tol=1.0e-15,
                            abs_tol=0.0,
                        )
                        or any(
                            isinstance(block.get(field), bool)
                            or not isinstance(block.get(field), int)
                            or block[field] <= 0
                            for field in (
                                "repetitions",
                                "batch_size",
                                "evaluation_count",
                                "evaluated_point_count",
                            )
                        )
                        or block.get("repetitions") != repetitions
                        or block.get("batch_size") != batch_size
                        or block.get("evaluation_count") != repetitions
                        or block.get("evaluated_point_count")
                        != repetitions * batch_size
                        or block.get("content_sha256")
                        != _canonical_sha256(
                            {
                                key: value
                                for key, value in block.items()
                                if key != "content_sha256"
                            }
                        )
                        for block_index, block in enumerate(raw_inventory)
                    ):
                        errors.append(
                            f"profile batch {batch_size} subprocess sample "
                            f"{sample_index} raw block content address is invalid"
                        )
                    else:
                        raw_values = [
                            float(block["wall_seconds_per_point"])
                            for block in raw_inventory
                        ]
                        raw_median = statistics.median(raw_values)
                        raw_mad = statistics.median(
                            abs(value - raw_median) for value in raw_values
                        )
                        observed_native_duration = sum(
                            float(block["native_wall_seconds"])
                            for block in raw_inventory
                        )
                        observed_caller_duration = sum(
                            float(block["caller_elapsed_seconds"])
                            for block in raw_inventory
                        )
                        requested_block_count = int(
                            raw_calibration["requested_minimum_block_count"]
                        )
                        runner_repetition_count = int(
                            calibration_runner_repetitions
                        )
                        runner_wall_seconds = float(calibration_runner_wall)
                        headroom_factor = float(calibration_headroom)
                        runner_total_repetitions = int(calibration_runner_total)
                        expected_scaled_repetitions = max(
                            runner_repetition_count,
                            math.ceil(
                                (
                                    float(arguments.target_runtime)
                                    * headroom_factor
                                )
                                / (
                                    requested_block_count
                                    * runner_wall_seconds
                                    * batch_size
                                )
                            ),
                        )
                        invocation_wall_duration = (
                            invocation.get("wall_seconds")
                            if isinstance(invocation, Mapping)
                            else None
                        )
                        process_wall_duration = (
                            worker_process_record.get("wall_seconds")
                            if isinstance(worker_process_record, Mapping)
                            else None
                        )
                        if (
                            not raw_values
                            or any(
                                not math.isfinite(value) or value <= 0.0
                                for value in raw_values
                            )
                            or raw_wall_median != raw_median
                            or raw_wall_mad != raw_mad
                            or raw_observed_duration != observed_native_duration
                            or raw_caller_duration != observed_caller_duration
                            or observed_native_duration
                            < float(arguments.target_runtime)
                            or observed_caller_duration
                            < float(arguments.target_runtime)
                            or raw_repetitions != expected_scaled_repetitions
                            or not isinstance(worker_measurement, Mapping)
                            or worker_measurement.get(
                                "benchmark_runner_sample_count"
                            )
                            != calibration_runner_samples
                            or worker_measurement.get(
                                "benchmark_runner_repetitions_per_sample"
                            )
                            != runner_repetition_count
                            or worker_measurement.get(
                                "benchmark_runner_evaluation_count"
                            )
                            != runner_total_repetitions
                            or worker_measurement.get(
                                "benchmark_runner_evaluated_point_count"
                            )
                            != runner_total_repetitions * batch_size
                            or worker_measurement.get(
                                "benchmark_runner_wall_seconds_per_point"
                            )
                            != runner_wall_seconds
                            or isinstance(invocation_wall_duration, bool)
                            or not isinstance(
                                invocation_wall_duration,
                                (float, int),
                            )
                            or not math.isfinite(
                                float(invocation_wall_duration)
                            )
                            or float(invocation_wall_duration)
                            < observed_native_duration
                            or float(invocation_wall_duration)
                            < observed_caller_duration
                            or isinstance(process_wall_duration, bool)
                            or not isinstance(
                                process_wall_duration,
                                (float, int),
                            )
                            or not math.isfinite(
                                float(process_wall_duration)
                            )
                            or float(process_wall_duration)
                            < observed_native_duration
                            or float(process_wall_duration)
                            < observed_caller_duration
                        ):
                            errors.append(
                                f"profile batch {batch_size} subprocess sample "
                                f"{sample_index} raw block statistics are invalid"
                            )
                if isinstance(schedule_index, bool) or not isinstance(
                    schedule_index, int
                ):
                    errors.append(
                        f"profile batch {batch_size} subprocess sample "
                        f"{sample_index} has no schedule identity"
                    )
                else:
                    schedule_indices.append(schedule_index)
                    scheduled = scheduled_entries.get(schedule_index)
                    if (
                        not isinstance(scheduled, Mapping)
                        or scheduled.get("mode") != mode
                        or scheduled.get("batch_size") != batch_size
                        or scheduled.get("round") != sample.get("round")
                        or scheduled.get("worker_invocation")
                        != sample.get("worker_invocation")
                        or scheduled.get("pre_timing_verification")
                        != sample.get("pre_timing_verification")
                        or scheduled.get("worker_result_record")
                        != sample.get("worker_result_record")
                        or not isinstance(worker_result_record, Mapping)
                        or scheduled.get("worker_result_sha256")
                        != worker_result_record.get("addressed_payload_sha256")
                    ):
                        errors.append(
                            f"profile batch {batch_size} subprocess sample "
                            f"{sample_index} is not bound to the interleaved schedule"
                        )
            if len(schedule_indices) != len(set(schedule_indices)):
                errors.append(f"profile batch {batch_size} reused a worker subprocess")
            if len(sample_values) == len(subprocess_samples) and sample_values:
                expected_median = statistics.median(sample_values)
                expected_mad = statistics.median(
                    abs(value - expected_median) for value in sample_values
                )
                if median != expected_median or mad != expected_mad:
                    errors.append(
                        f"profile batch {batch_size} median/MAD is not reproducible"
                    )
            if interrupted is not False:
                errors.append(f"profile batch {batch_size} was interrupted")
        missing = [
            batch_size
            for batch_size in DEFAULT_BATCH_SIZES
            if batch_size not in observed
        ]
        lane_passes = not errors and not missing
        lanes[mode] = {
            "passes": lane_passes,
            "observed_batch_sizes": observed,
            "missing_batch_sizes": missing,
            "errors": errors,
        }
        passes = passes and lane_passes
    return {
        "passes": passes,
        "lanes": lanes,
        "schedule": schedule_contract,
        "minimum_authoritative_samples": MIN_AUTHORITATIVE_SAMPLES,
        "configured_subprocess_samples": arguments.subprocess_samples,
        "configured_internal_minimum_samples": arguments.minimum_samples,
        "configured_warmup_runs": arguments.warmup_runs,
        "root_processes_match": root_processes_match,
    }


def _selected_union_semantic_projection(
    arguments: argparse.Namespace,
    profile: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, object]:
    """Build the sole relaxed cross-lane semantic projection.

    Every lane is still fully authenticated and internally validated.  Across
    lanes, this projection ignores only nonselected helicity coefficients and
    structural-zero flags plus logical/execution reduction identities.  Those
    fields may legitimately differ because recurrence carries a conservative
    live-helicity census while compiled artifacts can certify more zeros.
    """

    if not _selected_union_semantic_exception_enabled(arguments):
        raise HarnessError(
            "selected-union semantic projection is outside its audited workload"
        )
    coverage = identity.get("coverage")
    color_axis = identity.get("physical_color_flows")
    helicity_axis = identity.get("physical_helicities")
    specialized_axes = identity.get("generation_specialized_axes")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("color") != "complete"
        or coverage.get("helicities") != "complete"
        or coverage.get("complete_physical_axes") is not True
        or specialized_axes != []
        or not isinstance(color_axis, Mapping)
        or not isinstance(helicity_axis, Mapping)
    ):
        raise HarnessError(
            f"{mode} selected-union artifact does not expose complete physical axes"
        )
    validated_color = _validated_logical_physical_axis(
        color_axis,
        label="physical_color_flows",
        require_structural_zero=False,
    )
    validated_helicity = _validated_logical_physical_axis(
        helicity_axis,
        label="physical_helicities",
        require_structural_zero=True,
    )
    color_ids = validated_color["ordered_ids"]
    color_entries = validated_color["ordered_entries"]
    helicity_ids = validated_helicity["ordered_ids"]
    helicity_entries = validated_helicity["ordered_entries"]
    assert isinstance(color_ids, list)
    assert isinstance(color_entries, list)
    assert isinstance(helicity_ids, list)
    assert isinstance(helicity_entries, list)
    if (
        validated_color.get("count") != MIGRATION_UNION_PHYSICAL_COLOR_COUNT
        or len(color_ids) != MIGRATION_UNION_PHYSICAL_COLOR_COUNT
        or MIGRATION_UNION_SELECTED_HELICITY_ID not in helicity_ids
    ):
        raise HarnessError(
            f"{mode} selected-union artifact has the wrong physical-axis inventory"
        )
    selected_index = helicity_ids.index(MIGRATION_UNION_SELECTED_HELICITY_ID)
    selected_entry = helicity_entries[selected_index]
    assert isinstance(selected_entry, Mapping)
    selected_coefficient = selected_entry.get("coefficient")
    if (
        selected_entry.get("index") != selected_index
        or selected_entry.get("id") != MIGRATION_UNION_SELECTED_HELICITY_ID
        or tuple(selected_entry.get("values", ()))
        != MIGRATION_UNION_SELECTED_HELICITY_VALUES
        or isinstance(selected_coefficient, bool)
        or not isinstance(selected_coefficient, (float, int))
        or not math.isfinite(float(selected_coefficient))
        or float(selected_coefficient) <= 0.0
        or selected_entry.get("structural_zero") is not False
    ):
        raise HarnessError(
            f"{mode} selected-union helicity entry is not live and canonical"
        )
    selector_projection = _selected_union_selector_projection(
        arguments,
        profile.get("selector_contract"),
        identity,
    )
    validation = profile.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("resolved_helicity_ids")
        != [MIGRATION_UNION_SELECTED_HELICITY_ID]
        or validation.get("resolved_color_ids") != color_ids
    ):
        raise HarnessError(
            f"{mode} selected-union resolved axes do not match artifact physics"
        )
    process_id = profile.get("process_id")
    process_expression = profile.get("process_expression")
    if (
        not isinstance(process_id, str)
        or not process_id
        or not isinstance(process_expression, str)
        or " ".join(process_expression.split()).casefold()
        != MIGRATION_PROCESS_CANONICAL.casefold()
    ):
        raise HarnessError(
            f"{mode} selected-union profile has the wrong process identity"
        )
    helicity_kinematics = [
        {
            "index": entry["index"],
            "id": entry["id"],
            "values": entry["values"],
            **(
                {
                    "coefficient": entry["coefficient"],
                    "structural_zero": entry["structural_zero"],
                }
                if entry["id"] == MIGRATION_UNION_SELECTED_HELICITY_ID
                else {}
            ),
        }
        for entry in helicity_entries
        if isinstance(entry, Mapping)
    ]
    model_identity = identity.get("manifest_model_identity")
    if not isinstance(model_identity, Mapping):
        raise HarnessError(f"{mode} selected-union artifact has no model identity")
    return {
        "policy": "ddbar-z6g-all-flow-union-selected-helicity-v1",
        "process_id": process_id,
        "process_expression": MIGRATION_PROCESS_CANONICAL,
        "physical_color_flows": {
            "count": validated_color["count"],
            "ordered_ids_sha256": validated_color["ordered_ids_sha256"],
            "ordered_entries_sha256": validated_color["ordered_entries_sha256"],
        },
        "physical_helicities": {
            "count": validated_helicity["count"],
            "ordered_ids_sha256": validated_helicity["ordered_ids_sha256"],
            "kinematic_entries_sha256": _canonical_sha256(helicity_kinematics),
            "selected_index": selected_index,
            "selected_entry": dict(selected_entry),
        },
        "normalization_sha256": identity.get("normalization_sha256"),
        "model_common_physics_identity_sha256": model_identity.get(
            "common_physics_identity_sha256"
        ),
        "runtime_selector_semantics_sha256": identity.get(
            "runtime_selector_semantics_sha256"
        ),
        "profile_selector": selector_projection,
        "resolved_color_ids_sha256": _canonical_sha256(color_ids),
    }


def _profile_artifact_semantic_contract(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    arguments: argparse.Namespace | None = None,
) -> dict[str, object]:
    lane_contracts: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    selected_union_projection = _selected_union_semantic_exception_enabled(arguments)
    for mode, profile in profiles.items():
        identity = profile.get("artifact_semantic_identity")
        identity_sha256 = profile.get("artifact_semantic_identity_sha256")
        if not isinstance(identity, Mapping):
            errors.append(f"{mode} profile has no artifact semantic identity")
            continue
        if identity_sha256 != _canonical_sha256(identity):
            errors.append(
                f"{mode} profile artifact semantic identity is not content-addressed"
            )
            continue
        coverage = identity.get("coverage")
        color_axis = identity.get("physical_color_flows")
        helicity_axis = identity.get("physical_helicities")
        normalization = identity.get("normalization")
        model_identity = identity.get("manifest_model_identity")
        selector_semantics = identity.get("runtime_selector_semantics")
        runtime_selectors = identity.get("runtime_selectors")
        reduction_ordering = identity.get("reduction_ordering")
        reduction_coverage = identity.get("reduction_coverage")
        execution_reduction = identity.get("execution_reduction_identity")
        execution_reduction_coverage = identity.get("execution_reduction_coverage")
        execution_ordering = identity.get("execution_schedule_ordering")
        if (
            identity.get("kind") != "pyamplicol-benchmark-artifact-semantic-identity"
            or not _is_exact_int(
                identity.get("schema_version"),
                ARTIFACT_SEMANTIC_IDENTITY_SCHEMA,
            )
            or not isinstance(coverage, Mapping)
            or not isinstance(color_axis, Mapping)
            or not isinstance(helicity_axis, Mapping)
            or not isinstance(normalization, Mapping)
            or not isinstance(model_identity, Mapping)
            or not isinstance(selector_semantics, Mapping)
            or not isinstance(runtime_selectors, Mapping)
            or not isinstance(reduction_ordering, Mapping)
            or not isinstance(reduction_coverage, Mapping)
            or not isinstance(execution_reduction, Mapping)
            or not isinstance(execution_reduction_coverage, Mapping)
            or not isinstance(execution_ordering, Mapping)
        ):
            errors.append(f"{mode} artifact semantic identity is incomplete")
            continue
        axis_contracts: dict[str, dict[str, object]] = {}
        axes_are_valid = True
        for label, axis, require_structural_zero in (
            ("physical_color_flows", color_axis, False),
            ("physical_helicities", helicity_axis, True),
        ):
            try:
                validated_axis = _validated_logical_physical_axis(
                    axis,
                    label=label,
                    require_structural_zero=require_structural_zero,
                )
            except HarnessError:
                errors.append(f"{mode} {label} semantic identity is invalid")
                axes_are_valid = False
                continue
            axis_contracts[label] = {
                "count": validated_axis["count"],
                "ordered_ids_sha256": validated_axis["ordered_ids_sha256"],
                "ordered_entries_sha256": validated_axis["ordered_entries_sha256"],
            }
        if not axes_are_valid or len(axis_contracts) != 2:
            continue
        normalization_sha256 = identity.get("normalization_sha256")
        manifest_model = model_identity.get("manifest")
        manifest_model_sha256 = model_identity.get("manifest_sha256")
        model_common = model_identity.get("common_physics_identity")
        model_common_sha256 = model_identity.get("common_physics_identity_sha256")
        selector_semantics_sha256 = identity.get("runtime_selector_semantics_sha256")
        runtime_selectors_sha256 = identity.get("runtime_selectors_sha256")
        reduction_ordering_sha256 = identity.get("reduction_ordering_sha256")
        execution_reduction_sha256 = identity.get("execution_reduction_identity_sha256")
        execution_ordering_sha256 = identity.get("execution_schedule_ordering_sha256")
        try:
            validated_model_identity = _manifest_model_identity(
                {"model": manifest_model}
            )
        except HarnessError:
            validated_model_identity = None
        color_coverage = coverage.get("color")
        helicity_coverage = coverage.get("helicities")
        try:
            validated_selector_semantics = (
                _runtime_selector_semantic_identity(
                    runtime_selectors,
                    color_coverage=color_coverage,
                    helicity_coverage=helicity_coverage,
                    artifact=Path(f"<{mode}-profile>"),
                )
                if isinstance(color_coverage, str)
                and isinstance(helicity_coverage, str)
                else None
            )
        except HarnessError:
            validated_selector_semantics = None
        runtime_process_contract = execution_ordering.get("runtime_process_contract")
        manifest_payload_order = execution_ordering.get("manifest_payload_order")
        execution_ordering_is_valid = (
            isinstance(runtime_process_contract, Mapping)
            and bool(runtime_process_contract)
            and isinstance(manifest_payload_order, list)
            and bool(manifest_payload_order)
            and all(
                isinstance(entry, Mapping)
                and isinstance(entry.get("path"), str)
                and bool(entry.get("path"))
                for entry in manifest_payload_order
            )
        )
        try:
            validated_reduction = _logical_reduction_ordering_identity(
                reduction_ordering.get("kind"),
                color_axis=color_axis,
                helicity_axis=helicity_axis,
                artifact=Path(f"<{mode}-profile>"),
            )
        except HarnessError:
            validated_reduction = None
        execution_reduction_is_valid = False
        if validated_reduction is not None:
            try:
                _validate_execution_reduction_summary(
                    execution_reduction,
                    execution_reduction_coverage,
                    logical_reduction=validated_reduction["ordering"],
                    artifact=Path(f"<{mode}-profile>"),
                )
            except HarnessError:
                pass
            else:
                execution_reduction_is_valid = True
        if (
            normalization_sha256 != _canonical_sha256(normalization)
            or coverage.get("complete_physical_axes")
            is not (color_coverage == "complete" and helicity_coverage == "complete")
            or validated_model_identity is None
            or manifest_model_sha256 != validated_model_identity["manifest_sha256"]
            or _canonical_sha256(manifest_model)
            != _canonical_sha256(validated_model_identity["manifest"])
            or not isinstance(model_common, Mapping)
            or _canonical_sha256(model_common)
            != _canonical_sha256(validated_model_identity["common_physics_identity"])
            or model_common_sha256
            != validated_model_identity["common_physics_identity_sha256"]
            or validated_selector_semantics is None
            or _canonical_sha256(selector_semantics)
            != _canonical_sha256(validated_selector_semantics)
            or selector_semantics.get("generation_specialized_axes")
            != identity.get("generation_specialized_axes")
            or selector_semantics_sha256 != _canonical_sha256(selector_semantics)
            or runtime_selectors_sha256 != _canonical_sha256(runtime_selectors)
            or validated_reduction is None
            or _canonical_sha256(reduction_ordering)
            != _canonical_sha256(validated_reduction["ordering"])
            or reduction_ordering_sha256 != validated_reduction["ordering_sha256"]
            or _canonical_sha256(reduction_coverage)
            != _canonical_sha256(validated_reduction["coverage"])
            or execution_reduction_sha256 != _canonical_sha256(execution_reduction)
            or not execution_reduction_is_valid
            or not execution_ordering_is_valid
            or execution_ordering_sha256 != _canonical_sha256(execution_ordering)
            or reduction_coverage.get("complete") is not True
        ):
            errors.append(
                f"{mode} model/selector/reduction semantic identity is invalid"
            )
            continue
        lane_contract: dict[str, object] = {
            **axis_contracts,
            "normalization_sha256": normalization_sha256,
            "model_common_physics_identity_sha256": model_common_sha256,
            "runtime_selector_semantics_sha256": selector_semantics_sha256,
            "runtime_selectors_sha256": runtime_selectors_sha256,
            "reduction_ordering_sha256": reduction_ordering_sha256,
            "execution_reduction_identity_sha256": (execution_reduction_sha256),
            "execution_schedule_ordering_sha256": execution_ordering_sha256,
        }
        if selected_union_projection:
            assert arguments is not None
            try:
                lane_contract["selected_union_workload_projection"] = (
                    _selected_union_semantic_projection(
                        arguments,
                        profile,
                        identity,
                        mode=mode,
                    )
                )
            except HarnessError as error:
                errors.append(str(error))
                continue
        lane_contracts[mode] = lane_contract
    common_physics = (
        {
            mode: contract["selected_union_workload_projection"]
            for mode, contract in lane_contracts.items()
        }
        if selected_union_projection
        else {
            mode: {
                "physical_color_flows": contract["physical_color_flows"],
                "physical_helicities": contract["physical_helicities"],
                "normalization_sha256": contract["normalization_sha256"],
                "model_common_physics_identity_sha256": contract[
                    "model_common_physics_identity_sha256"
                ],
                "runtime_selector_semantics_sha256": contract[
                    "runtime_selector_semantics_sha256"
                ],
                "reduction_ordering_sha256": contract["reduction_ordering_sha256"],
            }
            for mode, contract in lane_contracts.items()
        }
    )
    common_values = list(common_physics.values())
    lanes_match = (
        bool(common_values)
        and set(lane_contracts) == set(profiles)
        and all(value == common_values[0] for value in common_values[1:])
    )
    if common_values and set(lane_contracts) == set(profiles) and not lanes_match:
        errors.append(
            "profile lanes differ outside the audited selected-union "
            "helicity/reduction projection"
            if selected_union_projection
            else (
                "profile lanes have different model, ordered physical-axis "
                "entries, runtime-selector semantics, normalization, or "
                "reduction ordering"
            )
        )
    return {
        "passes": bool(profiles)
        and not errors
        and len(lane_contracts) == len(profiles),
        "errors": errors,
        "lanes_match": lanes_match,
        "comparison_policy": (
            "ddbar-z6g-all-flow-union-selected-helicity-v1"
            if selected_union_projection
            else "strict-exact-v1"
        ),
        "lane_contracts": lane_contracts,
        "common_physics_contract": (
            common_values[0] if common_values and lanes_match else None
        ),
    }


def _resource_peak_mapping_errors(
    value: object,
    *,
    label: str,
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{label} is missing"]
    errors: list[str] = []
    if set(value) != _RESOURCE_PEAK_KEYS:
        errors.append(f"{label} has unexpected keys")
    if value.get("source") != "resource.getrusage":
        errors.append(f"{label} has an invalid source")
    if value.get("semantics") != _RESOURCE_PEAK_SEMANTICS:
        errors.append(f"{label} has invalid semantics")
    integer_fields = (
        "self_peak_bytes",
        "maximum_child_peak_bytes",
        "observed_lower_bound_bytes",
    )
    integers: dict[str, int] = {}
    for field in integer_fields:
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            errors.append(f"{label} {field} is not a non-negative integer")
        else:
            integers[field] = raw
    if len(integers) == len(integer_fields) and integers[
        "observed_lower_bound_bytes"
    ] != max(
        integers["self_peak_bytes"],
        integers["maximum_child_peak_bytes"],
    ):
        errors.append(f"{label} observed lower bound is inconsistent")
    return errors


def _recurrence_profile_rss_contract(
    profile: Mapping[str, Any] | None,
) -> dict[str, object]:
    errors: list[str] = []
    worker_count = 0
    maximum_cold_load_bytes = 0
    maximum_post_profile_bytes = 0
    raw_measurements = profile.get("profiles") if isinstance(profile, Mapping) else None
    if not isinstance(raw_measurements, list):
        return {
            "passes": False,
            "errors": ["recurrence profile measurements are missing for RSS"],
            "worker_count": 0,
            "maximum_cold_load_observed_lower_bound_bytes": None,
            "maximum_post_profile_observed_lower_bound_bytes": None,
        }
    for measurement_index, measurement in enumerate(raw_measurements):
        samples = (
            measurement.get("subprocess_samples")
            if isinstance(measurement, Mapping)
            else None
        )
        if not isinstance(samples, list):
            errors.append(
                f"recurrence RSS measurement {measurement_index} has no workers"
            )
            continue
        for sample_index, sample in enumerate(samples):
            worker_count += 1
            label = f"recurrence RSS worker {measurement_index}/{sample_index}"
            if not isinstance(sample, Mapping):
                errors.append(f"{label} is invalid")
                continue
            cold = sample.get("peak_rss_after_cold_load")
            post = sample.get("peak_rss_after_profile")
            cold_errors = _resource_peak_mapping_errors(
                cold,
                label=f"{label} cold-load mapping",
            )
            post_errors = _resource_peak_mapping_errors(
                post,
                label=f"{label} post-profile mapping",
            )
            errors.extend(cold_errors)
            errors.extend(post_errors)
            if cold_errors or post_errors:
                continue
            assert isinstance(cold, Mapping)
            assert isinstance(post, Mapping)
            cold_values = {
                field: cast(int, cold[field])
                for field in (
                    "self_peak_bytes",
                    "maximum_child_peak_bytes",
                    "observed_lower_bound_bytes",
                )
            }
            post_values = {
                field: cast(int, post[field])
                for field in (
                    "self_peak_bytes",
                    "maximum_child_peak_bytes",
                    "observed_lower_bound_bytes",
                )
            }
            for field in cold_values:
                if post_values[field] < cold_values[field]:
                    errors.append(f"{label} post-profile {field} is below cold-load")
            maximum_cold_load_bytes = max(
                maximum_cold_load_bytes,
                cold_values["observed_lower_bound_bytes"],
            )
            maximum_post_profile_bytes = max(
                maximum_post_profile_bytes,
                post_values["observed_lower_bound_bytes"],
            )
    if worker_count == 0:
        errors.append("recurrence RSS profile contains no workers")
    return {
        "passes": not errors,
        "errors": errors,
        "worker_count": worker_count,
        "maximum_cold_load_observed_lower_bound_bytes": (
            maximum_cold_load_bytes if worker_count else None
        ),
        "maximum_post_profile_observed_lower_bound_bytes": (
            maximum_post_profile_bytes if worker_count else None
        ),
    }


def _recurrence_runtime_acceptance(
    arguments: argparse.Namespace,
    *,
    source_identity: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
    generation: Mapping[str, Mapping[str, object]],
    profiles: Mapping[str, Mapping[str, Any]],
    validation_summary: Mapping[str, object] | None,
    profile_schedule: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build a developer-only, fail-closed recurrence runtime evidence record."""

    errors: list[str] = []
    required_configuration = {
        "modes": ["recurrence"],
        "generation_only": False,
        "batch_sizes": list(DEFAULT_BATCH_SIZES),
        "target_runtime_seconds": 5.0,
        "warmup_runs": 2,
        "minimum_internal_samples": MIN_AUTHORITATIVE_SAMPLES,
        "subprocess_samples_minimum": MIN_AUTHORITATIVE_SAMPLES,
        "subprocess_samples_maximum": 21,
        "process_family": "d d~ > Z + (n-1)*g",
        "accepted_n_values": [6, 7],
        "generation_specialized_axes_allowed": False,
        "complete_physical_axes_required": True,
    }
    observed_configuration = {
        "modes": getattr(arguments, "modes", None),
        "generation_only": getattr(arguments, "generation_only", None),
        "batch_sizes": getattr(arguments, "batch_size", None),
        "target_runtime_seconds": getattr(arguments, "target_runtime", None),
        "warmup_runs": getattr(arguments, "warmup_runs", None),
        "minimum_internal_samples": getattr(arguments, "minimum_samples", None),
        "subprocess_samples": getattr(arguments, "subprocess_samples", None),
        "gluon_count": getattr(arguments, "gluon_count", None),
        "process_expression_override": getattr(
            arguments,
            "process_expression",
            None,
        ),
        "lc_flow_layout": getattr(arguments, "lc_flow_layout", None),
        "color_flow_request": getattr(arguments, "color_flow", None),
        "helicity_request": getattr(arguments, "helicity", None),
        "validation_samples": getattr(arguments, "validation_samples", None),
        "point_tile_size": getattr(arguments, "point_tile_size", None),
        "jit_optimization_level": getattr(
            arguments,
            "jit_optimization_level",
            None,
        ),
        "specialize_flow_at_generation": getattr(
            arguments,
            "specialize_flow_at_generation",
            None,
        ),
        "prepared_model_path": (
            None
            if getattr(arguments, "prepared_model", None) is None
            else str(arguments.prepared_model)
        ),
    }
    if getattr(arguments, "modes", None) != ["recurrence"]:
        errors.append("runtime acceptance requires only the recurrence mode")
    if getattr(arguments, "generation_only", None) is not False:
        errors.append("runtime acceptance requires completed profiling")
    if getattr(arguments, "batch_size", None) != list(DEFAULT_BATCH_SIZES):
        errors.append("runtime acceptance requires exact batches [1, 128, 1024]")
    if not _is_exact_int(getattr(arguments, "warmup_runs", None), 2):
        errors.append("runtime acceptance requires exactly two warm-up runs")
    target_runtime = getattr(arguments, "target_runtime", None)
    if (
        isinstance(target_runtime, bool)
        or not isinstance(target_runtime, (float, int))
        or not math.isfinite(float(target_runtime))
        or float(target_runtime) != 5.0
    ):
        errors.append("runtime acceptance requires a five-second target")
    minimum_samples = getattr(arguments, "minimum_samples", None)
    if (
        isinstance(minimum_samples, bool)
        or not isinstance(minimum_samples, int)
        or minimum_samples < MIN_AUTHORITATIVE_SAMPLES
    ):
        errors.append("runtime acceptance requires at least seven native samples")
    subprocess_samples = getattr(arguments, "subprocess_samples", None)
    if (
        isinstance(subprocess_samples, bool)
        or not isinstance(subprocess_samples, int)
        or not MIN_AUTHORITATIVE_SAMPLES <= subprocess_samples <= 21
    ):
        errors.append(
            "runtime acceptance requires seven through twenty-one worker samples"
        )
    gluon_count = getattr(arguments, "gluon_count", None)
    selected_process = _selected_process(arguments)
    normalized_selected_process = " ".join(selected_process.split()).casefold()
    expected_ladder_process = (
        "d d~ > z" + " g" * gluon_count
        if (
            not isinstance(gluon_count, bool)
            and isinstance(gluon_count, int)
            and gluon_count in (5, 6)
        )
        else None
    )
    if (
        expected_ladder_process is None
        or normalized_selected_process != expected_ladder_process
    ):
        errors.append(
            "runtime acceptance requires the exact d d~ > Z+(n-1)g "
            "ladder process for n=6 or n=7"
        )
    if getattr(arguments, "lc_flow_layout", None) not in LC_FLOW_LAYOUTS:
        errors.append("runtime acceptance has an invalid LC flow layout")
    if getattr(arguments, "specialize_flow_at_generation", None) is not False:
        errors.append("runtime acceptance forbids generation specialization")

    recurrence_profile = profiles.get("recurrence")
    recurrence_profiles = (
        {"recurrence": recurrence_profile}
        if isinstance(recurrence_profile, Mapping)
        else {}
    )
    if set(profiles) != {"recurrence"}:
        errors.append("runtime acceptance profile inventory is not recurrence-only")

    try:
        measurement_contract = _profile_measurement_contract(
            arguments,
            recurrence_profiles,
            profile_schedule=profile_schedule,
            required_modes=("recurrence",),
        )
    except (HarnessError, TypeError, ValueError) as error:
        measurement_contract = {
            "passes": False,
            "errors": [f"measurement contract failed closed: {error}"],
        }
    if measurement_contract.get("passes") is not True:
        errors.append("authenticated recurrence measurement contract did not pass")

    try:
        artifact_semantic_contract = _profile_artifact_semantic_contract(
            recurrence_profiles
        )
    except (HarnessError, TypeError, ValueError) as error:
        artifact_semantic_contract = {
            "passes": False,
            "errors": [f"artifact semantic contract failed closed: {error}"],
        }
    if artifact_semantic_contract.get("passes") is not True:
        errors.append("recurrence artifact semantic contract did not pass")

    try:
        recomputed_validation = _pairwise_profile_validation(recurrence_profiles)
    except (HarnessError, TypeError, ValueError) as error:
        recomputed_validation = {
            "passes": False,
            "errors": [f"lane validation failed closed: {error}"],
        }
    if (
        recomputed_validation.get("passes") is not True
        or recomputed_validation.get("lane_validation_passes") is not True
        or recomputed_validation.get("selectors_match") is not True
        or recomputed_validation.get("fixtures_match") is not True
        or recomputed_validation.get("pairwise_validation_passes") is not None
    ):
        errors.append("recurrence lane self-validation did not pass")
    if not isinstance(validation_summary, Mapping) or _canonical_sha256(
        validation_summary
    ) != _canonical_sha256(recomputed_validation):
        errors.append("retained validation summary does not match recomputation")

    rss_contract = _recurrence_profile_rss_contract(
        recurrence_profile if isinstance(recurrence_profile, Mapping) else None
    )
    if rss_contract.get("passes") is not True:
        errors.append("recurrence worker RSS contract did not pass")

    semantic_identity = (
        recurrence_profile.get("artifact_semantic_identity")
        if isinstance(recurrence_profile, Mapping)
        else None
    )
    semantic_identity_sha256 = (
        recurrence_profile.get("artifact_semantic_identity_sha256")
        if isinstance(recurrence_profile, Mapping)
        else None
    )
    coverage = (
        semantic_identity.get("coverage")
        if isinstance(semantic_identity, Mapping)
        else None
    )
    generation_specialized_axes = (
        semantic_identity.get("generation_specialized_axes")
        if isinstance(semantic_identity, Mapping)
        else None
    )
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("complete_physical_axes") is not True
        or generation_specialized_axes != []
    ):
        errors.append("recurrence artifact lacks complete unspecialized physical axes")

    generation_record = generation.get("recurrence")
    if (
        not isinstance(generation_record, Mapping)
        or generation_record.get("mode") != "recurrence"
        or generation_record.get("generation_reused") is not False
    ):
        errors.append("recurrence generation evidence is not a fresh recurrence lane")
    artifact_identity = (
        generation_record.get("artifact_identity")
        if isinstance(generation_record, Mapping)
        else None
    )
    generation_semantic_identity = (
        artifact_identity.get("semantic_identity")
        if isinstance(artifact_identity, Mapping)
        else None
    )
    artifact_id: str | None = None
    try:
        artifact_id = _required_sha256(
            (
                artifact_identity.get("artifact_id")
                if isinstance(artifact_identity, Mapping)
                else None
            ),
            label="recurrence generation artifact",
        )
    except HarnessError:
        errors.append("recurrence generation artifact ID is invalid")
    generation_semantic_sha256 = (
        artifact_identity.get("semantic_identity_sha256")
        if isinstance(artifact_identity, Mapping)
        else None
    )
    generation_top_level_semantic_sha256 = (
        generation_record.get("artifact_semantic_identity_sha256")
        if isinstance(generation_record, Mapping)
        else None
    )
    if (
        not isinstance(semantic_identity_sha256, str)
        or _SHA256_PATTERN.fullmatch(semantic_identity_sha256) is None
        or not isinstance(generation_semantic_identity, Mapping)
        or _canonical_sha256(generation_semantic_identity) != semantic_identity_sha256
        or generation_semantic_sha256 != semantic_identity_sha256
        or generation_top_level_semantic_sha256 != semantic_identity_sha256
    ):
        errors.append(
            "recurrence generation artifact semantic identity is not profile-bound"
        )
    expected_effective_contract = {
        "execution_mode": "recurrence",
        "backend": "jit",
        "jit_optimization_level": (
            _expected_effective_jit_optimization_level(
                arguments,
                mode="recurrence",
            )
        ),
        "color_accuracy": "lc",
        "lc_flow_layout": getattr(arguments, "lc_flow_layout", None),
    }
    effective_contract = (
        generation_record.get("effective_contract")
        if isinstance(generation_record, Mapping)
        else None
    )
    if not isinstance(effective_contract, Mapping) or any(
        effective_contract.get(key) != value
        for key, value in expected_effective_contract.items()
    ):
        errors.append("recurrence generation artifact contract is invalid")

    selector_contract = (
        recurrence_profile.get("selector_contract")
        if isinstance(recurrence_profile, Mapping)
        else None
    )
    validation = (
        recurrence_profile.get("validation")
        if isinstance(recurrence_profile, Mapping)
        else None
    )
    try:
        fixture_contract = (
            _validation_fixture_contract(validation, mode="recurrence")
            if isinstance(validation, Mapping)
            else None
        )
    except HarnessError:
        fixture_contract = None
    if not isinstance(selector_contract, Mapping):
        errors.append("recurrence selector identity is missing")
    if not isinstance(fixture_contract, Mapping):
        errors.append("recurrence validation fixture identity is missing")

    source_identity_sha256 = (
        _canonical_sha256(source_identity)
        if isinstance(source_identity, Mapping) and source_identity
        else None
    )
    runtime_provenance_sha256 = (
        _canonical_sha256(runtime_provenance)
        if isinstance(runtime_provenance, Mapping) and runtime_provenance
        else None
    )
    if source_identity_sha256 is None:
        errors.append("source identity is missing")
    if runtime_provenance_sha256 is None:
        errors.append("runtime provenance is missing")
    expected_worker_bindings = {
        "source_identity_sha256": source_identity_sha256,
        "runtime_provenance_sha256": runtime_provenance_sha256,
        "artifact_id": artifact_id,
        "artifact_semantic_identity_sha256": semantic_identity_sha256,
    }
    raw_schedule_entries = (
        profile_schedule.get("entries")
        if isinstance(profile_schedule, Mapping)
        else None
    )
    workers_are_generation_bound = isinstance(raw_schedule_entries, list) and bool(
        raw_schedule_entries
    )
    if isinstance(raw_schedule_entries, list):
        for entry in raw_schedule_entries:
            verification = (
                entry.get("pre_timing_verification")
                if isinstance(entry, Mapping)
                else None
            )
            expected = (
                verification.get("expected")
                if isinstance(verification, Mapping)
                else None
            )
            observed = (
                verification.get("observed")
                if isinstance(verification, Mapping)
                else None
            )
            if (
                not isinstance(expected, Mapping)
                or not isinstance(observed, Mapping)
                or any(
                    expected.get(field) != value or observed.get(field) != value
                    for field, value in expected_worker_bindings.items()
                )
            ):
                workers_are_generation_bound = False
                break
    if not workers_are_generation_bound:
        errors.append(
            "recurrence profile workers are not bound to source, runtime, and "
            "generation identities"
        )

    process_expression = (
        " ".join(
            cast(str, recurrence_profile.get("process_expression")).split()
        ).casefold()
        if isinstance(recurrence_profile, Mapping)
        and isinstance(recurrence_profile.get("process_expression"), str)
        else None
    )
    expected_process_expression = " ".join(
        _selected_process(arguments).split()
    ).casefold()
    process_id = (
        recurrence_profile.get("process_id")
        if isinstance(recurrence_profile, Mapping)
        else None
    )
    if (
        process_expression != expected_process_expression
        or not isinstance(process_id, str)
        or not process_id
    ):
        errors.append("recurrence process identity is invalid")

    bindings = {
        "source_identity_sha256": source_identity_sha256,
        "runtime_provenance_sha256": runtime_provenance_sha256,
        "generation_artifact_id": artifact_id,
        "generation_artifact_semantic_identity_sha256": (generation_semantic_sha256),
        "profile_schedule_sha256": (
            _canonical_sha256(profile_schedule)
            if isinstance(profile_schedule, Mapping)
            else None
        ),
        "recurrence_profile_sha256": (
            _canonical_sha256(recurrence_profile)
            if isinstance(recurrence_profile, Mapping)
            else None
        ),
        "selector_contract": (
            dict(selector_contract) if isinstance(selector_contract, Mapping) else None
        ),
        "selector_contract_sha256": (
            _canonical_sha256(selector_contract)
            if isinstance(selector_contract, Mapping)
            else None
        ),
        "validation_fixture": (
            dict(fixture_contract) if isinstance(fixture_contract, Mapping) else None
        ),
        "validation_fixture_sha256": (
            _canonical_sha256(fixture_contract)
            if isinstance(fixture_contract, Mapping)
            else None
        ),
        "process_id": process_id,
        "process_expression": process_expression,
        "lc_flow_layout": getattr(arguments, "lc_flow_layout", None),
        "configuration": observed_configuration,
        "configuration_sha256": _canonical_sha256(observed_configuration),
    }
    accepted = not errors
    record: dict[str, object] = {
        "kind": RECURRENCE_RUNTIME_ACCEPTANCE_KIND,
        "schema_version": RECURRENCE_RUNTIME_ACCEPTANCE_SCHEMA,
        "accepted": accepted,
        "status": "accepted" if accepted else "rejected",
        "errors": errors,
        "required": required_configuration,
        "contracts": {
            "measurement": measurement_contract,
            "artifact_semantics": artifact_semantic_contract,
            "lane_validation": {
                "passes": recomputed_validation.get("passes"),
                "lane_validation_passes": recomputed_validation.get(
                    "lane_validation_passes"
                ),
                "selectors_match": recomputed_validation.get("selectors_match"),
                "fixtures_match": recomputed_validation.get("fixtures_match"),
                "pairwise_validation_passes": recomputed_validation.get(
                    "pairwise_validation_passes"
                ),
                "summary_sha256": _canonical_sha256(recomputed_validation),
            },
            "worker_rss": rss_contract,
        },
        "bindings": bindings,
    }
    record["content_sha256"] = _canonical_sha256(record)
    return record


def _capture_acceptance(
    arguments: argparse.Namespace,
    profiles: Mapping[str, Mapping[str, Any]],
    validation_summary: Mapping[str, object] | None,
    *,
    profile_schedule: Mapping[str, object] | None,
) -> dict[str, object]:
    authoritative_profiles = {
        mode: profiles[mode]
        for mode in AUTHORITATIVE_EXECUTION_MODES
        if mode in profiles
    }
    observed_modes = [
        mode for mode in AUTHORITATIVE_EXECUTION_MODES if mode in profiles
    ]
    missing_modes = [
        mode for mode in AUTHORITATIVE_EXECUTION_MODES if mode not in profiles
    ]
    missing_batch_sizes = [
        batch_size
        for batch_size in DEFAULT_BATCH_SIZES
        if batch_size not in set(arguments.batch_size)
    ]
    measurement_contract = _profile_measurement_contract(
        arguments,
        profiles,
        profile_schedule=profile_schedule,
        required_modes=AUTHORITATIVE_EXECUTION_MODES,
    )
    artifact_semantic_contract = _profile_artifact_semantic_contract(
        authoritative_profiles,
        arguments=arguments,
    )
    authoritative_validation = (
        _pairwise_profile_validation(
            authoritative_profiles,
            arguments=arguments,
        )
        if not missing_modes
        else None
    )
    specialized_axes_by_mode: dict[str, list[str]] = {}
    incomplete_physical_axes: list[str] = []
    for mode, profile in authoritative_profiles.items():
        semantic_identity = profile.get("artifact_semantic_identity")
        coverage = (
            semantic_identity.get("coverage")
            if isinstance(semantic_identity, Mapping)
            else None
        )
        specialized_axes = (
            semantic_identity.get("generation_specialized_axes")
            if isinstance(semantic_identity, Mapping)
            else None
        )
        if not isinstance(specialized_axes, list) or any(
            not isinstance(axis, str) for axis in specialized_axes
        ):
            specialized_axes_by_mode[mode] = ["unknown"]
        elif specialized_axes:
            specialized_axes_by_mode[mode] = list(specialized_axes)
        if (
            not isinstance(coverage, Mapping)
            or coverage.get("complete_physical_axes") is not True
        ):
            incomplete_physical_axes.append(mode)
    ineligibility_reasons: list[str] = []
    if not _authoritative_process_family(arguments):
        ineligibility_reasons.append(
            "the custom process expression is outside the authoritative "
            "u u~ or d d~ Z+6g campaigns"
        )
    if arguments.gluon_count != 6:
        ineligibility_reasons.append(
            "milestone 0 requires the default Z+6g process family"
        )
    if arguments.specialize_flow_at_generation:
        ineligibility_reasons.append(
            "generation-time flow specialization was requested"
        )
    if specialized_axes_by_mode:
        ineligibility_reasons.append(
            "one or more artifacts have generation-specialized physical axes"
        )
    if incomplete_physical_axes:
        ineligibility_reasons.append(
            "one or more artifacts lack complete physical flow/helicity coverage"
        )
    if artifact_semantic_contract["passes"] is not True:
        ineligibility_reasons.append(
            "artifact physical-axis, normalization, or reduction-order semantics "
            "are incomplete or inconsistent"
        )
    authoritative_eligible = not ineligibility_reasons
    evidence_complete = (
        not arguments.generation_only
        and not missing_modes
        and not missing_batch_sizes
        and measurement_contract["passes"] is True
    )
    complete = evidence_complete and authoritative_eligible
    passes = (
        authoritative_validation.get("passes") is True
        and measurement_contract["passes"] is True
        if complete and authoritative_validation is not None
        else None
    )
    eager_requested = "eager" in arguments.modes
    eager_observed = "eager" in profiles
    eager_profiles = {
        mode: profiles[mode]
        for mode in (*AUTHORITATIVE_EXECUTION_MODES, "eager")
        if mode in profiles
    }
    eager_measurement = (
        _profile_measurement_contract(
            arguments,
            profiles,
            profile_schedule=profile_schedule,
            required_modes=("eager",),
        )
        if eager_requested
        else None
    )
    eager_semantics = (
        _profile_artifact_semantic_contract(
            eager_profiles,
            arguments=arguments,
        )
        if eager_observed and not missing_modes
        else None
    )
    eager_validation = (
        _pairwise_profile_validation(
            eager_profiles,
            arguments=arguments,
        )
        if eager_observed and not missing_modes
        else None
    )
    eager_complete = bool(
        eager_requested
        and eager_observed
        and not missing_batch_sizes
        and eager_measurement is not None
        and eager_measurement.get("passes") is True
        and eager_semantics is not None
        and eager_semantics.get("passes") is True
        and eager_semantics.get("lanes_match") is True
    )
    eager_passes = (
        eager_validation.get("passes") is True
        if eager_complete and eager_validation is not None
        else None
    )
    eager_diagnostic = {
        "requested": eager_requested,
        "observed": eager_observed,
        "complete": eager_complete,
        "passes": eager_passes,
        "measurement_contract": eager_measurement,
        "artifact_semantic_contract": eager_semantics,
        "validation_summary": eager_validation,
        "ineligibility_reasons": [
            reason
            for condition, reason in (
                (
                    eager_requested and not eager_observed,
                    "the requested eager profile lane is missing",
                ),
                (
                    eager_observed
                    and eager_measurement is not None
                    and eager_measurement.get("passes") is not True,
                    "eager measurement evidence is incomplete",
                ),
                (
                    eager_observed
                    and eager_semantics is not None
                    and eager_semantics.get("passes") is not True,
                    "eager selectors or physical semantics are not comparable",
                ),
                (
                    eager_observed
                    and eager_validation is not None
                    and eager_validation.get("passes") is not True,
                    "eager numerical validation is not comparable or passing",
                ),
            )
            if condition
        ],
    }
    return {
        "kind": "pyamplicol-three-lane-layout-capture",
        "schema_version": CAPTURE_ACCEPTANCE_SCHEMA,
        "complete": complete,
        "evidence_complete": evidence_complete,
        "passes": passes,
        "authoritative_eligible": authoritative_eligible,
        "authoritative_ineligibility_reasons": ineligibility_reasons,
        "generation_specialized_axes_by_mode": specialized_axes_by_mode,
        "incomplete_physical_axes": incomplete_physical_axes,
        "required_modes": list(AUTHORITATIVE_EXECUTION_MODES),
        "observed_modes": observed_modes,
        "missing_modes": missing_modes,
        "diagnostic_modes": list(DIAGNOSTIC_EXECUTION_MODES),
        "eager_diagnostic": eager_diagnostic,
        "required_batch_sizes": list(DEFAULT_BATCH_SIZES),
        "observed_batch_sizes": list(arguments.batch_size),
        "missing_batch_sizes": missing_batch_sizes,
        "measurement_contract": measurement_contract,
        "artifact_semantic_contract": artifact_semantic_contract,
        "layout": arguments.lc_flow_layout,
        "generation_only": arguments.generation_only,
        "lane_self_validation_passes": (
            None
            if authoritative_validation is None
            else authoritative_validation.get("lane_validation_passes")
        ),
        "pairwise_validation_passes": (
            None
            if authoritative_validation is None
            else authoritative_validation.get("pairwise_validation_passes")
        ),
    }


def _milestone0_acceptance_manifest(
    arguments: argparse.Namespace,
    capture: Mapping[str, object],
) -> dict[str, object]:
    """Describe the external evidence still required; never self-certify M0."""

    missing_layouts = [
        layout for layout in LC_FLOW_LAYOUTS if layout != arguments.lc_flow_layout
    ]
    missing_evidence: list[dict[str, object]] = [
        {"kind": "layout_capture", "value": layout} for layout in missing_layouts
    ]
    missing_evidence.append({"kind": "external_lane", "value": "amplicol"})
    if capture.get("complete") is not True:
        missing_evidence.append(
            {
                "kind": "complete_current_layout_capture",
                "value": arguments.lc_flow_layout,
            }
        )
    elif capture.get("passes") is not True:
        missing_evidence.append(
            {
                "kind": "passing_current_layout_capture",
                "value": arguments.lc_flow_layout,
            }
        )
    return {
        "kind": M0_ACCEPTANCE_KIND,
        "schema_version": M0_ACCEPTANCE_SCHEMA,
        "accepted": False,
        "status": "incomplete",
        "required": {
            "authoritative_execution_modes": list(AUTHORITATIVE_EXECUTION_MODES),
            "diagnostic_execution_modes": list(DIAGNOSTIC_EXECUTION_MODES),
            "lc_flow_layouts": list(LC_FLOW_LAYOUTS),
            "batch_sizes": list(DEFAULT_BATCH_SIZES),
            "external_lanes": ["amplicol"],
            "generation_specialized_axes_allowed": False,
            "timing": {
                "minimum_native_wall_blocks_per_worker": (MIN_AUTHORITATIVE_SAMPLES),
                "minimum_interleaved_worker_subprocesses_per_cell": (
                    MIN_AUTHORITATIVE_SAMPLES
                ),
                "headline_source": "runtime_core_repeated_wall_time",
                "statistics": "subprocess-median-and-raw-mad-v1",
            },
        },
        "observed_current_layout_capture": dict(capture),
        "missing_evidence": missing_evidence,
        "integration_step": (
            "A separate fail-closed orchestrator must combine content-hashed "
            "topology-replay and all-flow-union captures with pinned AmpliCol "
            "selected-flow and all-flow raw evidence containing at least seven "
            "paired interleaved subprocess samples before "
            "milestone 0 can be accepted."
        ),
    }


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if (
        getattr(arguments, "paired_profile_coordination_dir", None) is not None
        and arguments.warmup_runs != 2
    ):
        raise HarnessError(
            "paired migration captures require exactly two warmup runs"
        )
    started_at = _utc_now()
    started = time.perf_counter()
    source_identity = _git_source_identity()
    runtime_provenance = _runtime_provenance(source_identity)
    _validate_expected_runtime_identity(
        source_identity,
        runtime_provenance,
        expected_source_revision=getattr(
            arguments,
            "expected_source_revision",
            None,
        ),
        expected_native_build_inputs_sha256=getattr(
            arguments,
            "expected_native_build_inputs_sha256",
            None,
        ),
    )
    if arguments.prepared_model is not None:
        try:
            arguments.prepared_model = arguments.prepared_model.expanduser().resolve(
                strict=True
            )
        except OSError as error:
            raise HarnessError(
                f"prepared model does not exist: {arguments.prepared_model}"
            ) from error
        if not arguments.prepared_model.is_file():
            raise HarnessError(
                f"prepared model is not a regular file: {arguments.prepared_model}"
            )
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_json = (
        arguments.result_json.resolve()
        if arguments.result_json is not None
        else output_root / "result.json"
    )
    layout_suffix = arguments.lc_flow_layout
    all_artifacts = {
        "recurrence": output_root / f"recurrence-{layout_suffix}",
        "compiled": (
            output_root
            / f"compiled-jit-o{arguments.jit_optimization_level}-{layout_suffix}"
        ),
        "eager": (
            output_root
            / f"eager-jit-o{arguments.jit_optimization_level}-{layout_suffix}"
        ),
    }
    artifacts = {mode: all_artifacts[mode] for mode in arguments.modes}
    model_identities = {
        mode: _selected_model_identity(
            arguments,
            mode=mode,
            source_identity=source_identity,
        )
        for mode in arguments.modes
    }
    generation_signatures = {
        mode: _semantic_generation_signature(
            arguments,
            mode=mode,
            source_identity=source_identity,
            runtime_provenance=runtime_provenance,
            model_identity=model_identities[mode],
        )
        for mode in arguments.modes
    }
    generation: dict[str, dict[str, object]] = {}
    artifact_baselines: dict[str, dict[str, object]] = {}
    drift_rechecks: list[dict[str, object]] = []
    for mode, artifact in artifacts.items():
        signature = generation_signatures[mode]
        if artifact.exists() and not arguments.force:
            if not artifact.is_dir():
                raise HarnessError(f"artifact path is not a directory: {artifact}")
            artifact_identity, reuse_provenance = _require_reusable_artifact(
                artifact,
                expected_signature=signature,
            )
            record: dict[str, object] = {
                "mode": mode,
                "generation_wall_seconds": None,
                "generation_reused": True,
                "peak_rss": None,
                "worker_command": None,
                "original_generation_command": reuse_provenance["generation_command"],
                "model_source": model_identities[mode],
            }
        else:
            if arguments.reuse_only:
                raise HarnessError(f"required artifact does not exist: {artifact}")
            print(f"Generating {mode} artifact at {artifact}", file=sys.stderr)
            record = _run_worker(
                (
                    "generate",
                    "--mode",
                    mode,
                    "--artifact",
                    str(artifact),
                    "--write-mode",
                    "replace" if arguments.force else "error",
                    "--gluon-count",
                    str(arguments.gluon_count),
                    *(
                        ()
                        if arguments.process_expression is None
                        else ("--process-expression", arguments.process_expression)
                    ),
                    "--validation-samples",
                    str(arguments.validation_samples),
                    "--point-tile-size",
                    str(arguments.point_tile_size),
                    "--jit-optimization-level",
                    str(arguments.jit_optimization_level),
                    "--lc-flow-layout",
                    arguments.lc_flow_layout,
                    *(
                        ("--specialize-flow-at-generation",)
                        if arguments.specialize_flow_at_generation
                        else ()
                    ),
                    *(
                        ()
                        if arguments.prepared_model is None
                        else ("--prepared-model", str(arguments.prepared_model))
                    ),
                ),
                mode=mode,
                phase="generation",
                timeout_seconds=arguments.generation_timeout,
            )
            drift_rechecks.append(
                _recheck_driver_state(
                    source_identity,
                    runtime_provenance,
                    artifact_baselines,
                    phase=f"after-{mode}-generation-worker",
                )
            )
            generation_worker_result_evidence = _preserved_worker_result_evidence(
                record
            )
            record.pop("worker_result_record", None)
            worker_model_source = record.pop("model_source", None)
            _validate_worker_model_identity(
                model_identities[mode],
                worker_model_source,
            )
            record["worker_reported_model_source"] = worker_model_source
            record["model_source"] = model_identities[mode]
            record["generation_worker_result_evidence"] = (
                generation_worker_result_evidence
            )
            artifact_identity = _artifact_identity(artifact)
            _write_reuse_signature(
                artifact,
                signature=signature,
                artifact_identity=artifact_identity,
                generation_command=record.get("worker_command"),
            )
            reuse_provenance = _json_object(
                _reuse_signature_path(artifact),
                label="artifact reuse signature",
            )
        effective_contract = _validate_artifact_contract(
            artifact,
            artifact_identity,
            arguments=arguments,
            mode=mode,
        )
        phases = _artifact_phases(artifact)
        record.update(
            {
                "artifact": str(artifact),
                "artifact_stats": _artifact_stats(artifact),
                "artifact_identity": artifact_identity,
                "artifact_semantic_identity": artifact_identity["semantic_identity"],
                "artifact_semantic_identity_sha256": artifact_identity[
                    "semantic_identity_sha256"
                ],
                "artifact_reuse_provenance": reuse_provenance,
                "phase_timings_seconds": phases,
                "phase_total_seconds": sum(phases.values()),
                "effective_contract": effective_contract,
                "semantic_generation_signature": signature,
                "semantic_generation_signature_sha256": (_canonical_sha256(signature)),
                "combined_reuse_semantic_signature_sha256": reuse_provenance[
                    "semantic_signature_sha256"
                ],
                "reuse_signature": _path_identity(_reuse_signature_path(artifact)),
            }
        )
        generation[mode] = record
        artifact_baselines[mode] = _artifact_drift_baseline(
            artifact,
            generation_signature=signature,
            artifact_identity=artifact_identity,
            reuse_signature=reuse_provenance,
        )
        drift_rechecks.append(
            _recheck_driver_state(
                source_identity,
                runtime_provenance,
                artifact_baselines,
                phase=f"after-{mode}-artifact-binding",
            )
        )

    validation_point_artifact = artifacts.get(
        "recurrence",
        next(iter(artifacts.values())),
    )
    validation_worker_arguments: list[str] = [
        "--validation-point-artifact",
        str(validation_point_artifact),
        "--target-runtime",
        str(arguments.target_runtime),
        "--minimum-samples",
        str(arguments.minimum_samples),
        "--warmup-runs",
        str(arguments.warmup_runs),
        "--color-flow",
        arguments.color_flow,
        "--helicity",
        arguments.helicity,
        "--lc-flow-layout",
        arguments.lc_flow_layout,
        "--gluon-count",
        str(arguments.gluon_count),
        *(
            ()
            if arguments.process_expression is None
            else ("--process-expression", arguments.process_expression)
        ),
        "--validation-samples",
        str(arguments.validation_samples),
        "--jit-optimization-level",
        str(arguments.jit_optimization_level),
    ]
    validation_preflight: dict[str, object] | None = None
    preflight_profiles: dict[str, dict[str, Any]] = {}
    preflight_worker_evidence: dict[str, dict[str, object]] = {}
    if not arguments.generation_only:
        for mode, artifact in artifacts.items():
            generation_record = generation[mode]
            scheduled_artifact_identity = generation_record["artifact_identity"]
            scheduled_reuse_provenance = generation_record["artifact_reuse_provenance"]
            assert isinstance(scheduled_artifact_identity, Mapping)
            assert isinstance(scheduled_reuse_provenance, Mapping)
            expectations = _profile_identity_expectations(
                source_identity,
                runtime_provenance,
                scheduled_artifact_identity,
                scheduled_reuse_provenance,
                reuse_sidecar=_reuse_signature_path(artifact),
            )
            print(f"Validating {mode} lane before profiling", file=sys.stderr)
            worker = _run_worker(
                (
                    "preflight",
                    *validation_worker_arguments,
                    "--mode",
                    mode,
                    "--artifact",
                    str(artifact),
                    *_profile_expectation_arguments(expectations),
                ),
                mode=mode,
                phase="validation-preflight",
                timeout_seconds=arguments.profile_timeout,
            )
            drift_rechecks.append(
                _recheck_driver_state(
                    source_identity,
                    runtime_provenance,
                    artifact_baselines,
                    phase=f"after-{mode}-validation-preflight",
                )
            )
            preflight_profiles[mode] = _preflight_profile_from_worker(worker)
            verification = worker.get("pre_timing_verification")
            assert isinstance(verification, Mapping)
            fixture = preflight_profiles[mode]["validation"]
            fixture_contract = (
                _validation_fixture_contract(fixture, mode=mode)
                if isinstance(fixture, Mapping)
                else None
            )
            preflight_worker_evidence[mode] = {
                "worker_invocation": worker.get("worker_invocation"),
                "worker_process_record": worker.get("worker_process_record"),
                "worker_result_record": worker.get("worker_result_record"),
                "cold_load_seconds": worker.get("cold_load_seconds"),
                "peak_rss_after_cold_load": worker.get("peak_rss_after_cold_load"),
                "peak_rss_after_validation": worker.get("peak_rss_after_profile"),
                "artifact_semantic_identity_sha256": verification.get(
                    "artifact_semantic_identity_sha256"
                ),
                "validation_fixture": fixture_contract,
            }
        authoritative_preflight_profiles = {
            mode: preflight_profiles[mode]
            for mode in AUTHORITATIVE_EXECUTION_MODES
            if mode in preflight_profiles
        }
        if not authoritative_preflight_profiles:
            authoritative_preflight_profiles = dict(preflight_profiles)
        validation_preflight = _require_validation_preflight(
            arguments,
            authoritative_preflight_profiles,
        )
        validation_preflight["worker_evidence"] = preflight_worker_evidence

    worker_profile_arguments: list[str] = [
        "profile",
        *validation_worker_arguments,
    ]
    profile_schedule: dict[str, object] | None = None
    paired_profile_coordination: dict[str, object] | None = None
    profile_worker_results: list[dict[str, Any]] = []
    if not arguments.generation_only:
        profile_schedule = _build_profile_schedule(
            arguments.modes,
            arguments.batch_size,
            subprocess_samples=arguments.subprocess_samples,
        )
        paired_profile_coordination = _initialize_paired_profile_coordination(
            arguments,
            source_identity=source_identity,
            runtime_provenance=runtime_provenance,
            profile_schedule=profile_schedule,
            result_json=result_json,
        )
        raw_schedule_entries = profile_schedule["entries"]
        assert isinstance(raw_schedule_entries, list)
        for raw_entry in raw_schedule_entries:
            assert isinstance(raw_entry, dict)
            mode = str(raw_entry["mode"])
            batch_size = int(raw_entry["batch_size"])
            schedule_index = int(raw_entry["schedule_index"])
            schedule_round = int(raw_entry["round"])
            artifact = artifacts[mode]
            generation_record = generation[mode]
            scheduled_artifact_identity = generation_record["artifact_identity"]
            scheduled_reuse_provenance = generation_record["artifact_reuse_provenance"]
            assert isinstance(scheduled_artifact_identity, Mapping)
            assert isinstance(scheduled_reuse_provenance, Mapping)
            expectations = _profile_identity_expectations(
                source_identity,
                runtime_provenance,
                scheduled_artifact_identity,
                scheduled_reuse_provenance,
                reuse_sidecar=_reuse_signature_path(artifact),
            )
            paired_token = _wait_for_paired_profile_token(arguments, raw_entry)
            print(
                f"Profiling schedule {schedule_index}: round {schedule_round}, "
                f"{mode}, batch {batch_size}",
                file=sys.stderr,
            )
            worker_result = _run_worker(
                (
                    *worker_profile_arguments,
                    "--mode",
                    mode,
                    "--artifact",
                    str(artifact),
                    "--batch-size",
                    str(batch_size),
                    "--schedule-index",
                    str(schedule_index),
                    "--schedule-round",
                    str(schedule_round),
                    *_profile_expectation_arguments(expectations),
                ),
                mode=mode,
                phase="profile",
                timeout_seconds=arguments.profile_timeout,
            )
            drift_rechecks.append(
                _recheck_driver_state(
                    source_identity,
                    runtime_provenance,
                    artifact_baselines,
                    phase=f"after-profile-worker-{schedule_index}",
                )
            )
            verification = worker_result.get("pre_timing_verification")
            invocation = worker_result.get("worker_invocation")
            if not isinstance(verification, Mapping) or not isinstance(
                invocation,
                Mapping,
            ):
                raise HarnessError(
                    f"profile worker {schedule_index} returned incomplete provenance"
                )
            raw_entry["worker_invocation"] = dict(invocation)
            raw_entry["pre_timing_verification"] = _compact_profile_verification(
                verification
            )
            result_record = worker_result.get("worker_result_record")
            if not isinstance(result_record, Mapping):
                raise HarnessError(
                    f"profile worker {schedule_index} returned no result record"
                )
            raw_entry["worker_result_record"] = dict(result_record)
            raw_entry["worker_result_sha256"] = result_record[
                "addressed_payload_sha256"
            ]
            paired_completion = _record_paired_profile_completion(
                arguments,
                schedule_entry=raw_entry,
                token=paired_token,
                worker_result=worker_result,
            )
            if paired_completion is not None:
                raw_entry["paired_profile_token"] = dict(paired_token or {})
                raw_entry["paired_profile_completion"] = paired_completion
                assert paired_profile_coordination is not None
                completion_records = paired_profile_coordination["completion_records"]
                assert isinstance(completion_records, list)
                completion_records.append(paired_completion)
            profile_worker_results.append(worker_result)

    profiles = (
        {}
        if profile_schedule is None
        else _aggregate_profile_workers(profile_schedule, profile_worker_results)
    )

    validation_summary = (
        _pairwise_profile_validation(
            profiles,
            arguments=arguments,
        )
        if profiles
        else None
    )
    capture = _capture_acceptance(
        arguments,
        profiles,
        validation_summary,
        profile_schedule=profile_schedule,
    )
    milestone0 = _milestone0_acceptance_manifest(arguments, capture)
    recurrence_runtime_acceptance = _recurrence_runtime_acceptance(
        arguments,
        source_identity=source_identity,
        runtime_provenance=runtime_provenance,
        generation=generation,
        profiles=profiles,
        validation_summary=validation_summary,
        profile_schedule=profile_schedule,
    )
    finished_at = _utc_now()
    driver_command = getattr(arguments, "driver_command", None)
    if not isinstance(driver_command, Mapping):
        driver_command = _command_identity((sys.executable, str(DRIVER_PATH)))
    payload: dict[str, object] = {
        "kind": RESULT_KIND,
        "schema_version": RESULT_SCHEMA,
        "complete": capture["complete"],
        "passes": capture["passes"],
        "capture_acceptance": capture,
        "milestone0_acceptance": milestone0,
        "recurrence_runtime_acceptance": recurrence_runtime_acceptance,
        "source": source_identity,
        "runtime_provenance": runtime_provenance,
        "provenance": {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "wall_seconds": time.perf_counter() - started,
            "host": _host_identity(),
            "working_directory": str(Path.cwd()),
            "driver_command": dict(driver_command),
            "post_worker_identity_rechecks": drift_rechecks,
            "external_watchdog_required_for_long_runs": True,
        },
        "process": _selected_process(arguments),
        "process_name": _selected_process_name(arguments),
        "workload": (
            "all-flows/runtime-selected-single-helicity"
            if arguments.lc_flow_layout == "all-flow-union"
            else "single-runtime-selected-flow/helicity-sum"
        ),
        "configuration": {
            "batch_sizes": list(arguments.batch_size),
            "target_runtime_seconds": arguments.target_runtime,
            "minimum_samples": arguments.minimum_samples,
            "subprocess_samples": arguments.subprocess_samples,
            "warmup_runs": arguments.warmup_runs,
            "generation_timeout_seconds": arguments.generation_timeout,
            "profile_timeout_seconds": arguments.profile_timeout,
            "color_flow_request": arguments.color_flow,
            "helicity_request": arguments.helicity,
            "lc_flow_layout": arguments.lc_flow_layout,
            "gluon_count": arguments.gluon_count,
            "validation_samples": arguments.validation_samples,
            "point_tile_size": arguments.point_tile_size,
            "jit_optimization_level": arguments.jit_optimization_level,
            "validation_point_artifact": str(validation_point_artifact),
            "generation_only": arguments.generation_only,
            "allow_diagnostic_incomplete_success": (
                arguments.allow_diagnostic_incomplete_success
            ),
            "modes": list(arguments.modes),
            "prepared_model_path": (
                None
                if arguments.prepared_model is None
                else str(arguments.prepared_model)
            ),
            "model_identities": model_identities,
            "validation_seed": VALIDATION_SEED,
            "specialize_flow_at_generation": (arguments.specialize_flow_at_generation),
            "external_watchdog_required_for_long_runs": True,
        },
        "generation": generation,
        "profile_schedule": profile_schedule,
        "validation_preflight": validation_preflight,
        "paired_profile_coordination": paired_profile_coordination,
        "profiles": profiles,
        "validation_summary": validation_summary,
        "selector_contracts_match": (
            None
            if validation_summary is None
            else validation_summary["selectors_match"]
        ),
        "validation_fixtures_match": (
            None if validation_summary is None else validation_summary["fixtures_match"]
        ),
        "lane_comparisons": (
            None if validation_summary is None else validation_summary["comparisons"]
        ),
        "result_json": str(result_json),
    }
    _write_json_atomic(result_json, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".artifacts" / "developer" / "recurrence-z6g",
        help="artifact/result directory",
    )
    result.add_argument("--result-json", type=Path)
    result.add_argument("--gluon-count", type=_positive_int, default=6)
    result.add_argument(
        "--process-expression",
        help="explicit diagnostic process expression instead of the qq_Zng family",
    )
    result.add_argument("--validation-samples", type=_positive_int, default=10)
    result.add_argument(
        "--point-tile-size",
        type=_positive_int,
        default=1024,
        help="recurrence workspace point stride used for cache-tiling experiments",
    )
    result.add_argument(
        "--jit-optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=2,
        help="JIT optimization level used for generated artifacts (default: 2)",
    )
    result.add_argument(
        "--prepared-model",
        type=Path,
        help="explicit prepared-model bundle used by every selected execution mode",
    )
    result.add_argument(
        "--expected-source-revision",
        help=(
            "externally pinned clean source revision; must be supplied together "
            "with --expected-native-build-inputs-sha256"
        ),
    )
    result.add_argument(
        "--expected-native-build-inputs-sha256",
        help=(
            "externally pinned native build-input digest; must be supplied "
            "together with --expected-source-revision"
        ),
    )
    result.add_argument(
        "--generation-only",
        action="store_true",
        help="generate and report artifact statistics without runtime profiling",
    )
    result.add_argument(
        "--allow-diagnostic-incomplete-success",
        action="store_true",
        help=(
            "return success for an explicitly diagnostic incomplete capture; "
            "authoritative incomplete captures fail by default"
        ),
    )
    result.add_argument(
        "--mode",
        dest="modes",
        choices=EXECUTION_MODES,
        action="append",
        help=(
            "execution mode to generate/profile; repeat to select multiple "
            "(default: compiled, eager, recurrence)"
        ),
    )
    result.add_argument(
        "--only-mode",
        choices=EXECUTION_MODES,
        help=argparse.SUPPRESS,
    )
    result.add_argument(
        "--batch-size",
        type=_positive_int,
        action="append",
        default=None,
        help="native profiling batch size; repeat (default: 1, 128, and 1024)",
    )
    result.add_argument("--target-runtime", type=_positive_float, default=5.0)
    result.add_argument(
        "--generation-timeout",
        type=_positive_float,
        default=900.0,
        help="maximum seconds for either generation worker (default: 900)",
    )
    result.add_argument(
        "--profile-timeout",
        type=_positive_float,
        default=300.0,
        help="maximum seconds for either profiling worker (default: 300)",
    )
    result.add_argument(
        "--minimum-samples",
        type=_positive_int,
        default=MIN_AUTHORITATIVE_SAMPLES,
        help="minimum native-wall timing blocks per subprocess (default: 7)",
    )
    result.add_argument(
        "--subprocess-samples",
        type=_positive_int,
        default=MIN_AUTHORITATIVE_SAMPLES,
        help=(
            "independent interleaved worker subprocesses per mode/batch cell "
            "(default: 7)"
        ),
    )
    result.add_argument("--warmup-runs", type=int, default=2)
    result.add_argument(
        "--color-flow",
        default="1",
        help="one-based flow ordinal or stable flow ID (default: 1)",
    )
    result.add_argument(
        "--helicity",
        default="1",
        help="one-based helicity ordinal or stable helicity ID (default: 1)",
    )
    result.add_argument(
        "--lc-flow-layout",
        choices=("topology-replay", "all-flow-union"),
        default="topology-replay",
        help="recurrence/compiled LC layout and matching benchmark workload",
    )
    result.add_argument(
        "--force",
        action="store_true",
        help="replace every selected artifact before profiling",
    )
    result.add_argument(
        "--specialize-flow-at-generation",
        action="store_true",
        help=(
            "for topology-replay, generate a true one-flow artifact using the "
            "requested stable flow ID or ordinal 1"
        ),
    )
    result.add_argument(
        "--reuse-only",
        action="store_true",
        help="fail rather than generate when any selected artifact is missing",
    )
    result.add_argument(
        "--paired-profile-coordination-dir",
        type=Path,
        help=(
            "workspace-local coordination directory used by the paired "
            "baseline/candidate scheduler"
        ),
    )
    result.add_argument(
        "--paired-profile-role",
        choices=("baseline", "candidate"),
        help="role owned by this driver in a paired profile campaign",
    )
    result.add_argument(
        "--paired-profile-session-id",
        help="opaque scheduler session identifier for paired profiling",
    )
    result.add_argument(
        "--paired-profile-wait-timeout",
        type=_positive_float,
        default=7200.0,
        help="maximum seconds to wait for each paired scheduler token",
    )
    return result


def _normalize_modes(arguments: argparse.Namespace) -> list[str]:
    if arguments.only_mode is not None and arguments.modes is not None:
        raise HarnessError("--only-mode cannot be combined with --mode")
    if arguments.only_mode is not None:
        return [arguments.only_mode]
    if arguments.modes is None:
        return list(EXECUTION_MODES)
    if len(set(arguments.modes)) != len(arguments.modes):
        raise HarnessError("execution modes must be unique")
    return list(arguments.modes)


def _normalize_batch_sizes(arguments: argparse.Namespace) -> list[int]:
    if arguments.batch_size is None:
        return list(DEFAULT_BATCH_SIZES)
    if len(set(arguments.batch_size)) != len(arguments.batch_size):
        raise HarnessError("batch sizes must be unique")
    return list(arguments.batch_size)


def _result_exit_code(
    result: Mapping[str, object],
    *,
    allow_diagnostic_incomplete_success: bool,
) -> int:
    complete = result.get("complete")
    passes = result.get("passes")
    if complete is True and passes is True:
        return 0
    if complete is not True and passes is None and allow_diagnostic_incomplete_success:
        return 0
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        if values[:1] == ["_worker"]:
            return _worker_main(values[1:])
        arguments = parser().parse_args(values)
        if arguments.force and arguments.reuse_only:
            raise HarnessError("--force and --reuse-only are mutually exclusive")
        paired_fields = (
            arguments.paired_profile_coordination_dir,
            arguments.paired_profile_role,
            arguments.paired_profile_session_id,
        )
        if any(value is not None for value in paired_fields) and not all(
            value is not None for value in paired_fields
        ):
            raise HarnessError(
                "paired profile coordination directory, role, and session ID "
                "must be provided together"
            )
        if arguments.paired_profile_coordination_dir is not None:
            if arguments.generation_only:
                raise HarnessError(
                    "paired profile coordination cannot be generation-only"
                )
            if (
                not isinstance(arguments.paired_profile_session_id, str)
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
                    arguments.paired_profile_session_id,
                )
                is None
            ):
                raise HarnessError("paired profile session ID is invalid")
        arguments.modes = _normalize_modes(arguments)
        arguments.batch_size = _normalize_batch_sizes(arguments)
        arguments.driver_command = _command_identity(
            (sys.executable, str(DRIVER_PATH), *values)
        )
        if arguments.warmup_runs < 0:
            raise HarnessError("warmup runs must be non-negative")
        result = run(arguments)
    except (
        HarnessError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"recurrence-z6g-benchmark: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return _result_exit_code(
        result,
        allow_diagnostic_incomplete_success=(
            arguments.allow_diagnostic_incomplete_success
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
