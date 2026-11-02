#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Compare one eager or compiled runtime against a frozen baseline interpreter.

The driver generates one compiled JIT O3 artifact per interpreter and reuses
it while its cache signature remains valid.  Each reported outer sample comes
from a separate repository-owned sampling helper run by that interpreter. The
accepted headline timing comes directly from the warmed, unprofiled
``_benchmark_f64_wall_time`` pass, independent of the installed benchmark
coordinator version. Native profile attribution is collected in a paired pass
over the byte-identical batch and repetition count, and never substitutes for
the headline. A performance gate is authoritative only for one shared artifact
or byte-identical performance-relevant payloads, and warmed native values are
an independent numerical-correctness gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WATCHDOG = ROOT / "tools" / "ci" / "memory_watchdog.py"
NATIVE_SAMPLE_HELPER = ROOT / "tools" / "developer" / "compiled_mode_sample.py"
DEPENDENCY_ENTRY = ROOT / "tools" / "developer" / "python_dependency_entry.py"
MEMORY_LIMIT_GIB = 30.0
NATIVE_WALL_TIME_SOURCE = "runtime_core_repeated_wall_time"
NATIVE_WALL_TIME_SAMPLE_PASS = "runtime._benchmark_f64_wall_time"
PAIRED_TIMING_SAMPLE_CONTRACT = "paired_unprofiled_headline_profiled_attribution_v1"
PROFILE_ATTRIBUTION_SAMPLE_PASS = "runtime._profile_arena_repeated"
FROZEN_BASELINE_PROFILE_ATTRIBUTION_SAMPLE_PASS = "runtime.profile_repeated"
ARENA_PROFILE_BOUNDARY = "warmed-direct-arena-borrowed-input-preallocated-output-v1"
FROZEN_BASELINE_PROFILE_BOUNDARY = "materialized-native-profile-v1"
ARENA_PHASE_TIMING_SCOPE = "coarse-arena-boundary-only-v1"
FROZEN_BASELINE_PHASE_TIMING_SCOPE = "profiled-evaluator-phases-v1"
ARENA_PROFILE_PROTOCOL = "arena"
FROZEN_BASELINE_PROFILE_PROTOCOL = "frozen-pre-arena"
NATIVE_SAMPLE_RESULT_KIND = "pyamplicol-compiled-mode-native-sample"
NATIVE_SAMPLE_SCHEMA_VERSION = 5
INSTALLATION_IDENTITY_KIND = "pyamplicol-installed-distribution-identity"
INSTALLATION_IDENTITY_SCHEMA_VERSION = 2
RESULT_KIND = "pyamplicol-compiled-mode-regression"
CACHE_KIND = "pyamplicol-compiled-mode-regression-artifact-cache"
CACHE_SCHEMA_VERSION = 4
SCHEMA_VERSION = 6
DEFAULT_GENERATION_TIMEOUT = 300.0
DEFAULT_PROFILE_TIMEOUT = 120.0
DEFAULT_TARGET_RUNTIME = 5.0
DEFAULT_SAMPLE_COUNT = 7
DEFAULT_BATCH_SIZE = 1024
DEFAULT_WARMUP_RUNS = 2
RELATIVE_TOLERANCE = 0.03
MAD_MULTIPLIER = 3.0
GAIN_RELATIVE_THRESHOLD = 0.10
MATERIAL_RESOURCE_GROWTH_THRESHOLD = 0.03
VALIDATION_SEED = 20260719
VALIDATION_SAMPLE_COUNT = 8
CORRECTNESS_POINT_DERIVATION_KIND = (
    "pyamplicol-compiled-mode-correctness-point-derivation"
)
CORRECTNESS_POINT_DERIVATION_SCHEMA_VERSION = 1
CORRECTNESS_POINT_DERIVATION_CONTRACT = (
    "authenticated-first-validation-point-massive-rambo-v1"
)
CORRECTNESS_SEED_START = 20260726
MAX_CORRECTNESS_SEED_ATTEMPTS = 256
PRECISION32_CORRECTNESS_POLICY = "first-current-sample-exact-oracle-direct-all-f64-v1"
DEFAULT_LC_FLOW_LAYOUT = "topology-replay"
CORRECTNESS_RELATIVE_TOLERANCE = 1.0e-12
CORRECTNESS_ABSOLUTE_TOLERANCE = 1.0e-15
PERFORMANCE_RELEVANT_PAYLOAD_ROLES = frozenset(
    {
        "compiled-model",
        "evaluator-manifest",
        "evaluator-state",
        "model-parameters",
        "runtime-physics",
    }
)
REQUIRED_PERFORMANCE_PAYLOAD_ROLES = frozenset({"evaluator-state"})
_HASH_CHUNK_BYTES = 1024 * 1024
EXECUTION_MODES = ("eager", "compiled")
DEPENDENCY_DISTRIBUTIONS = (
    ("numpy", "numpy"),
    ("symbolica", "symbolica"),
    ("ufo-model-loader", "ufo_model_loader"),
)
EAGER_DIRECT_ARENA_CAPABILITY = "eager-direct-arena-v1"
COMPILED_DIRECT_ARENA_CAPABILITY = "compiled-plane-arena-v1"


class RegressionError(RuntimeError):
    """Raised when the regression measurement cannot be trusted."""


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _at_least_five(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 5:
        raise argparse.ArgumentTypeError("must be at least five")
    return parsed


def _absolute_path(path: Path) -> Path:
    """Return an absolute invocation path without resolving venv symlinks."""

    return Path(os.path.abspath(path.expanduser()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as error:
        raise RegressionError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _tree_identity(path: Path) -> dict[str, object]:
    """Hash relative names and bytes for an exact, location-independent tree ID."""

    try:
        root = path.resolve(strict=True)
    except OSError as error:
        raise RegressionError(f"cannot resolve identity tree: {path}") from error
    if not root.is_dir():
        raise RegressionError(f"identity tree is not a directory: {path}")
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    try:
        members = sorted(
            (candidate for candidate in root.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        )
        for member in members:
            relative = member.relative_to(root).as_posix().encode("utf-8")
            size = member.stat().st_size
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(size.to_bytes(8, "big"))
            with member.open("rb") as stream:
                while chunk := stream.read(_HASH_CHUNK_BYTES):
                    digest.update(chunk)
            file_count += 1
            size_bytes += size
    except OSError as error:
        raise RegressionError(f"cannot hash identity tree: {path}") from error
    return {
        "algorithm": "sha256-relative-path-size-content-v1",
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _dependency_site_identity(path: Path) -> dict[str, object]:
    """Hash every external distribution used by measured artifact generation."""

    try:
        root = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise RegressionError(f"cannot resolve dependency site: {path}") from error
    if not root.is_dir():
        raise RegressionError(f"dependency site is not a directory: {root}")
    available: dict[str, list[importlib.metadata.Distribution]] = {}
    for distribution in importlib.metadata.distributions(path=[str(root)]):
        name = distribution.metadata.get("Name")
        if isinstance(name, str):
            available.setdefault(_normalized_distribution_name(name), []).append(
                distribution
            )
    identities: dict[str, dict[str, object]] = {}
    for distribution_name, import_name in DEPENDENCY_DISTRIBUTIONS:
        normalized_name = _normalized_distribution_name(distribution_name)
        matches = available.get(normalized_name, [])
        if len(matches) != 1:
            raise RegressionError(
                "dependency site must contain exactly one "
                f"{distribution_name} distribution, "
                f"found {len(matches)}: {root}"
            )
        distribution = matches[0]
        digest = hashlib.sha256()
        file_count = 0
        size_bytes = 0
        package_origin: Path | None = None
        entries = sorted(distribution.files or (), key=str)
        for entry in entries:
            relative = Path(str(entry))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "__pycache__" in relative.parts
                or relative.suffix == ".pyc"
            ):
                continue
            candidate = root / relative
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as error:
                raise RegressionError(
                    f"{distribution_name} dependency file escapes its "
                    f"authenticated site: {entry}"
                ) from error
            if not resolved.is_file():
                continue
            encoded = relative.as_posix().encode("utf-8")
            size = resolved.stat().st_size
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(size.to_bytes(8, "big"))
            with resolved.open("rb") as stream:
                while chunk := stream.read(_HASH_CHUNK_BYTES):
                    digest.update(chunk)
            file_count += 1
            size_bytes += size
            if relative.as_posix() == f"{import_name}/__init__.py":
                package_origin = resolved
        if file_count == 0 or package_origin is None:
            raise RegressionError(
                f"dependency distribution {distribution_name!r} has no "
                f"authenticated package content in {root}"
            )
        identities[distribution_name] = {
            "name": str(distribution.metadata["Name"]),
            "version": distribution.version,
            "package_origin": str(package_origin),
            "algorithm": "sha256-relative-path-size-content-v1",
            "sha256": digest.hexdigest(),
            "file_count": file_count,
            "size_bytes": size_bytes,
        }
    digest_basis = {
        package: {
            key: value
            for key, value in identity.items()
            if key not in {"package_origin"}
        }
        for package, identity in identities.items()
    }
    return {
        "path": str(path),
        "resolved_path": str(root),
        "algorithm": ("sha256-numpy-symbolica-ufo-model-loader-distributions-v1"),
        "sha256": _canonical_sha256(digest_basis),
        "distributions": identities,
    }


def _command_identity(command: Sequence[str]) -> dict[str, object]:
    argv = [str(value) for value in command]
    canonical = json.dumps(
        argv,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "argv": argv,
        "argv_sha256": hashlib.sha256(canonical).hexdigest(),
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


def _model_argument(value: str) -> str:
    candidate = Path(value).expanduser()
    return str(_absolute_path(candidate)) if candidate.exists() else value


def _require_interpreter(path: Path, *, lane: str) -> Path:
    path = _absolute_path(path)
    if not path.is_file():
        raise RegressionError(f"{lane} Python does not exist: {path}")
    if not os.access(path, os.X_OK):
        raise RegressionError(f"{lane} Python is not executable: {path}")
    return path


def _guarded_command(command: Sequence[str]) -> tuple[str, ...]:
    if not WATCHDOG.is_file():
        raise RegressionError(f"memory watchdog does not exist: {WATCHDOG}")
    if not NATIVE_SAMPLE_HELPER.is_file():
        raise RegressionError(
            f"native sampling helper does not exist: {NATIVE_SAMPLE_HELPER}"
        )
    if not DEPENDENCY_ENTRY.is_file():
        raise RegressionError(
            f"dependency entry helper does not exist: {DEPENDENCY_ENTRY}"
        )
    return (
        sys.executable,
        str(WATCHDOG),
        "--limit-gib",
        f"{MEMORY_LIMIT_GIB:g}",
        "--",
        *command,
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_json(
    command: Sequence[str],
    *,
    timeout: float,
    environment: Mapping[str, str],
) -> tuple[dict[str, Any], float, str]:
    """Run a JSON-producing generation/sample command under the watchdog."""

    guarded = _guarded_command(command)
    started = time.monotonic()
    process = subprocess.Popen(
        guarded,
        cwd=ROOT,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _terminate_process_group(process)
        raise RegressionError(
            f"command exceeded {timeout:.1f}s: {' '.join(command)}"
        ) from error
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"exit code {process.returncode}"
        raise RegressionError(f"command failed: {' '.join(command)}\n{detail}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RegressionError(
            f"command did not emit JSON: {' '.join(command)}\n{stdout[-2000:]}"
        ) from error
    if not isinstance(payload, dict):
        raise RegressionError("command JSON result must be an object")
    return payload, elapsed, stderr


def _generation_command(
    python: Path,
    *,
    process: str,
    artifact: Path,
    model: str,
    color: str,
    execution_mode: str = "compiled",
    jit_optimization_level: int = 3,
    lc_flow_layout: str = DEFAULT_LC_FLOW_LAYOUT,
    dependency_site: Path | None = None,
) -> tuple[str, ...]:
    if execution_mode not in EXECUTION_MODES:
        raise RegressionError(f"unsupported execution mode: {execution_mode!r}")
    arguments = (
        "generate",
        process,
        str(artifact),
        "--model",
        model,
        "--backend",
        "jit",
        "--execution-mode",
        execution_mode,
        "--jit-optimization-level",
        str(jit_optimization_level),
        "--color-accuracy",
        color,
        "--lc-flow-layout",
        lc_flow_layout,
        "--mode",
        "replace",
        "--workers",
        "1",
        "--validation",
        "--validation-samples",
        str(VALIDATION_SAMPLE_COUNT),
        "--validation-seed",
        str(VALIDATION_SEED),
        "--no-post-build-validation",
        "--no-emit-api-bundle",
        "--progress",
        "off",
        "--json",
    )
    if dependency_site is None:
        return (str(python), "-m", "pyamplicol", *arguments)
    return (
        str(python),
        str(DEPENDENCY_ENTRY),
        "--dependency-site",
        str(dependency_site),
        "--module",
        "pyamplicol",
        "--",
        *arguments,
    )


def _profile_command(
    python: Path,
    *,
    artifact: Path,
    process_id: str,
    batch_size: int,
    target_runtime: float,
    minimum_samples: int,
    warmup_runs: int,
    helicities: Sequence[str],
    color_flows: Sequence[str],
    execution_mode: str = "compiled",
    profile_protocol: str = ARENA_PROFILE_PROTOCOL,
    dependency_site: Path | None = None,
    include_precision32: bool = False,
) -> tuple[str, ...]:
    command = [
        str(python),
        str(NATIVE_SAMPLE_HELPER),
        str(artifact),
        "--process",
        process_id,
        "--execution-mode",
        execution_mode,
        "--profile-protocol",
        profile_protocol,
        "--target-runtime",
        str(target_runtime),
        "--batch-size",
        str(batch_size),
        "--warmup-runs",
        str(warmup_runs),
        "--minimum-samples",
        str(minimum_samples),
    ]
    for helicity in helicities:
        command.extend(("--helicity", helicity))
    for color_flow in color_flows:
        command.extend(("--color-flow", color_flow))
    if dependency_site is not None:
        command.extend(("--dependency-site", str(dependency_site)))
    if include_precision32:
        command.append("--include-precision32")
    return tuple(command)


def _watchdog_peak_gib(stderr: str) -> float | None:
    matches = re.findall(r"peak_rss=([0-9]+(?:\.[0-9]+)?) GiB", stderr)
    return float(matches[-1]) if matches else None


def _watchdog_marker(stderr: str) -> str | None:
    markers = [
        line.strip()
        for line in stderr.splitlines()
        if line.startswith("memory-watchdog: command finished")
    ]
    return markers[-1] if markers else None


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegressionError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise RegressionError(f"{label} must be a JSON object: {path}")
    return payload


def _artifact_lc_flow_layout(
    manifest: Mapping[str, Any],
    *,
    process_id: str,
) -> str | None:
    extensions = manifest.get("extensions")
    if not isinstance(extensions, Mapping):
        return None
    generation = extensions.get("generation")
    if not isinstance(generation, Mapping):
        return None
    processes = generation.get("concrete_processes")
    if not isinstance(processes, list):
        return None
    for process in processes:
        if not isinstance(process, Mapping) or process.get("id") != process_id:
            continue
        filters = process.get("filters")
        if not isinstance(filters, Mapping):
            return None
        layout = filters.get("lc_flow_layout")
        return layout if isinstance(layout, str) and layout else None
    return None


def _artifact_payload_digests(
    manifest: Mapping[str, Any],
    *,
    artifact: Path,
) -> list[dict[str, object]]:
    payloads = manifest.get("payloads")
    if payloads is None:
        return []
    if not isinstance(payloads, list):
        raise RegressionError("artifact payload inventory must be a list")
    try:
        artifact_root = artifact.resolve(strict=True)
    except OSError as error:
        raise RegressionError(f"cannot resolve artifact root: {artifact}") from error
    result: list[dict[str, object]] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise RegressionError("artifact payload inventory entry is invalid")
        path = payload.get("path")
        sha256 = payload.get("sha256")
        size_bytes = payload.get("size_bytes")
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise RegressionError("artifact payload digest entry is invalid")
        try:
            payload_path = (artifact / path).resolve(strict=True)
            payload_path.relative_to(artifact_root)
        except (OSError, ValueError) as error:
            raise RegressionError(
                f"artifact payload path is invalid: {path!r}"
            ) from error
        observed_size = payload_path.stat().st_size
        observed_sha256 = _sha256_file(payload_path)
        if observed_size != size_bytes or observed_sha256 != sha256:
            raise RegressionError(
                f"artifact payload does not match its declared digest: {path!r}"
            )
        identity: dict[str, object] = {
            "path": path,
            "sha256": observed_sha256,
            "size_bytes": observed_size,
            "role": payload.get("role"),
        }
        process_id = payload.get("process_id")
        if process_id is not None:
            identity["process_id"] = process_id
        result.append(identity)
    return sorted(result, key=lambda entry: str(entry["path"]))


def _artifact_execution_mode(record: Mapping[str, Any]) -> str | None:
    capabilities = record.get("required_runtime_capabilities")
    if not isinstance(capabilities, list):
        return None
    normalized = tuple(str(capability) for capability in capabilities)
    if any(
        capability
        in {
            "eager-direct-arena-v1",
            "rusticol.eager-runtime-layout.complex-f64.v1",
        }
        or "eager-dag" in capability
        for capability in normalized
    ):
        return "eager"
    if any(
        capability == "compiled-plane-arena-v1"
        or capability == "symjit.application.complex-f64.v1"
        for capability in normalized
    ):
        return "compiled"
    return None


def _artifact_effective_contract(artifact: Path) -> dict[str, object] | None:
    path = artifact / "config" / "effective.toml"
    if not path.is_file():
        return None
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        evaluator = payload["evaluator"]
        color = payload["color"]
        jit = evaluator["jit"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise RegressionError(
            f"artifact has an invalid effective configuration: {path}"
        ) from error
    if not isinstance(evaluator, Mapping) or not isinstance(color, Mapping):
        raise RegressionError(f"artifact effective configuration is invalid: {path}")
    return {
        "backend": evaluator.get("backend"),
        "execution_mode": evaluator.get("execution_mode"),
        "jit_optimization_level": (
            jit.get("optimization_level") if isinstance(jit, Mapping) else None
        ),
        "lc_flow_layout": color.get("lc_flow_layout"),
        "color_accuracy": color.get("accuracy"),
    }


def _artifact_semantic_identity(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    artifact: Path,
    process_id: str,
    execution_mode: str,
    lc_flow_layout: str | None,
) -> dict[str, object] | None:
    """Build an ABI-neutral identity for the timed public physics workload."""

    physics_relative = record.get("physics_path")
    payloads = manifest.get("payloads")
    if not isinstance(physics_relative, str) or not isinstance(payloads, list):
        return None
    validation_records = [
        payload
        for payload in payloads
        if isinstance(payload, Mapping)
        and payload.get("role") == "validation-momenta"
        and payload.get("process_id") == process_id
        and isinstance(payload.get("path"), str)
    ]
    if len(validation_records) != 1:
        return None
    physics = _json_object(
        artifact / physics_relative,
        label="runtime physics",
    )
    validation = _json_object(
        artifact / str(validation_records[0]["path"]),
        label="validation momenta",
    )
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        return None
    common_model = {
        key: model.get(key)
        for key in (
            "name",
            "restriction",
            "compiled_schema_version",
            "content_sha256",
        )
    }
    effective_contract = _artifact_effective_contract(artifact)
    if effective_contract is None:
        return None
    identity = {
        "kind": "pyamplicol-abi-neutral-performance-workload",
        "schema_version": 1,
        "process": {
            "id": process_id,
            "expression": (
                " ".join(str(record.get("expression", "")).split()).casefold()
            ),
            "external_pdgs": record.get("external_pdgs"),
        },
        "model_common_physics": common_model,
        "color_accuracy": record.get("color_accuracy"),
        "lc_flow_layout": lc_flow_layout,
        "execution_mode": execution_mode,
        "effective_contract": effective_contract,
        "runtime_physics": physics,
        "runtime_physics_sha256": _canonical_sha256(physics),
        "validation_momenta": validation,
        "validation_momenta_sha256": _canonical_sha256(validation),
    }
    return {
        **identity,
        "sha256": _canonical_sha256(identity),
    }


def _artifact_metadata(
    artifact: Path,
    *,
    expected_process: str,
    expected_color: str,
    expected_execution_mode: str = "compiled",
    expected_lc_flow_layout: str | None = None,
) -> dict[str, Any]:
    manifest_path = artifact / "artifact.json"
    outer = _json_object(manifest_path, label="artifact manifest")
    processes = outer.get("processes")
    if not isinstance(processes, list) or len(processes) != 1:
        raise RegressionError(
            f"regression artifact must contain exactly one process: {artifact}"
        )
    record = processes[0]
    if not isinstance(record, dict):
        raise RegressionError(f"artifact process metadata is invalid: {artifact}")
    process_id = record.get("id")
    expression = record.get("expression")
    color = record.get("color_accuracy")
    if not isinstance(process_id, str) or not process_id:
        raise RegressionError(f"artifact process ID is invalid: {artifact}")
    if not isinstance(expression, str) or (
        " ".join(expression.split()).casefold()
        != " ".join(expected_process.split()).casefold()
    ):
        raise RegressionError(
            f"artifact process does not match {expected_process!r}: {artifact}"
        )
    if color != expected_color:
        raise RegressionError(
            f"artifact color accuracy does not match {expected_color!r}: {artifact}"
        )
    execution_mode = _artifact_execution_mode(record)
    if execution_mode != expected_execution_mode:
        raise RegressionError(
            f"artifact is not {expected_execution_mode} mode: {artifact}"
        )
    raw_capabilities = record.get("required_runtime_capabilities")
    if not isinstance(raw_capabilities, list) or not all(
        isinstance(value, str) for value in raw_capabilities
    ):
        raise RegressionError(
            f"artifact runtime capability inventory is invalid: {artifact}"
        )
    artifact_id = outer.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise RegressionError(f"artifact ID is invalid: {artifact}")
    lc_flow_layout = _artifact_lc_flow_layout(outer, process_id=process_id)
    if (
        expected_lc_flow_layout is not None
        and lc_flow_layout != expected_lc_flow_layout
    ):
        raise RegressionError(
            "artifact LC flow layout does not match "
            f"{expected_lc_flow_layout!r}: got {lc_flow_layout!r} in {artifact}"
        )
    tree_identity = _tree_identity(artifact)
    payload_digests = _artifact_payload_digests(outer, artifact=artifact)
    semantic_identity = _artifact_semantic_identity(
        outer,
        record,
        artifact=artifact,
        process_id=process_id,
        execution_mode=execution_mode,
        lc_flow_layout=lc_flow_layout,
    )
    extensions = outer.get("extensions")
    generation = (
        extensions.get("generation") if isinstance(extensions, Mapping) else None
    )
    raw_phases = (
        generation.get("phase_timings_seconds")
        if isinstance(generation, Mapping)
        else None
    )
    phase_timings: dict[str, float] | None = None
    core_generation_seconds: float | None = None
    if isinstance(raw_phases, Mapping):
        phase_timings = {}
        for name, raw_value in raw_phases.items():
            if (
                not isinstance(name, str)
                or isinstance(raw_value, bool)
                or not isinstance(raw_value, (float, int))
                or not math.isfinite(float(raw_value))
                or float(raw_value) < 0.0
            ):
                raise RegressionError(
                    f"artifact has an invalid generation phase: {artifact}"
                )
            phase_timings[name] = float(raw_value)
        core_generation_seconds = sum(
            value
            for name, value in phase_timings.items()
            if name not in {"model-loading", "process-expansion"}
        )
        if core_generation_seconds <= 0.0:
            core_generation_seconds = None
    material_payload_size_bytes = sum(
        int(payload["size_bytes"])
        for payload in payload_digests
        if str(payload.get("role"))
        not in {
            "configuration-effective",
            "configuration-requested",
            "validation-momenta",
        }
    )
    return {
        "artifact_id": artifact_id,
        "path": str(artifact),
        "process_id": process_id,
        "process_expression": expression,
        "color_accuracy": color,
        "execution_mode": execution_mode,
        "required_runtime_capabilities": list(raw_capabilities),
        "lc_flow_layout": lc_flow_layout,
        "manifest_sha256": _sha256_file(manifest_path),
        "tree_identity": tree_identity,
        "payload_digests": payload_digests,
        "payload_digest_count": len(payload_digests),
        "material_payload_size_bytes": material_payload_size_bytes,
        "size_bytes": tree_identity["size_bytes"],
        "producer": outer.get("producer"),
        "semantic_workload_identity": semantic_identity,
        "semantic_workload_sha256": (
            None if semantic_identity is None else semantic_identity["sha256"]
        ),
        "generation_phase_timings_seconds": phase_timings,
        "core_generation_seconds": core_generation_seconds,
    }


def _path_identity(path: Path) -> dict[str, object]:
    try:
        stat = path.stat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RegressionError(f"cannot inspect path identity: {path}") from error
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(resolved),
    }


def _model_identity(model: str) -> dict[str, object]:
    path = Path(model).expanduser()
    if not path.exists():
        return {"argument": model, "path_backed": False}
    try:
        absolute = _absolute_path(path)
        resolved = absolute.resolve(strict=True)
        stat = resolved.stat()
    except OSError as error:
        raise RegressionError(f"cannot inspect model identity: {model}") from error
    result: dict[str, object] = {
        "argument": model,
        "path_backed": True,
        "path": str(absolute),
        "resolved_path": str(resolved),
        "is_directory": resolved.is_dir(),
        "mtime_ns": stat.st_mtime_ns,
        "size_bytes": stat.st_size,
    }
    if resolved.is_dir():
        tree = _tree_identity(resolved)
        result["tree_identity"] = tree
        result["size_bytes"] = tree["size_bytes"]
    elif resolved.is_file():
        result["sha256"] = _sha256_file(resolved)
    else:
        raise RegressionError(f"model path is not a file or directory: {model}")
    return result


def _generation_signature(
    python: Path,
    *,
    installation_identity: Mapping[str, Any],
    process: str,
    model: str,
    color: str,
    execution_mode: str = "compiled",
    jit_optimization_level: int = 3,
    lc_flow_layout: str = DEFAULT_LC_FLOW_LAYOUT,
    dependency_site: Path | None = None,
    dependency_site_identity: Mapping[str, Any] | None = None,
    artifact: Path | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "generation_driver": _path_identity(Path(__file__)),
        "dependency_entry": _path_identity(DEPENDENCY_ENTRY),
        "python": _path_identity(python),
        "installed_pyamplicol": dict(installation_identity),
        "process": process,
        "model": _model_identity(model),
        "color_accuracy": color,
        "lc_flow_layout": lc_flow_layout,
        "execution_mode": execution_mode,
        "backend": "jit",
        "jit_optimization_level": jit_optimization_level,
        "dependency_site": (
            None
            if dependency_site is None
            else dict(
                dependency_site_identity
                if dependency_site_identity is not None
                else _dependency_site_identity(dependency_site)
            )
        ),
        "workers": 1,
        "validation_samples": VALIDATION_SAMPLE_COUNT,
        "validation_seed": VALIDATION_SEED,
        "post_build_validation": False,
        "emit_api_bundle": False,
    }
    if artifact is not None:
        result["generation_command"] = _command_identity(
            _generation_command(
                python,
                process=process,
                artifact=artifact,
                model=model,
                color=color,
                execution_mode=execution_mode,
                jit_optimization_level=jit_optimization_level,
                lc_flow_layout=lc_flow_layout,
                dependency_site=dependency_site,
            )
        )
    return result


def _installed_pyamplicol_identity(
    python: Path,
    *,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    command = (str(python), str(NATIVE_SAMPLE_HELPER), "--installation-identity")
    payload, _elapsed, _stderr = _run_json(
        command,
        timeout=DEFAULT_PROFILE_TIMEOUT,
        environment=environment,
    )
    _require_equal(
        payload,
        "kind",
        INSTALLATION_IDENTITY_KIND,
        label="installed pyamplicol identity kind",
    )
    _require_equal(
        payload,
        "schema_version",
        INSTALLATION_IDENTITY_SCHEMA_VERSION,
        label="installed pyamplicol identity schema_version",
    )
    distribution_content = payload.get("distribution_content")
    if not isinstance(distribution_content, Mapping):
        raise RegressionError(
            "installed pyamplicol identity has no distribution content identity"
        )
    digest = distribution_content.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RegressionError(
            "installed pyamplicol identity has an invalid distribution SHA-256"
        )
    native_modules = payload.get("native_modules")
    if (
        not isinstance(native_modules, Sequence)
        or isinstance(native_modules, (str, bytes))
        or len(native_modules) != 1
    ):
        raise RegressionError(
            "installed pyamplicol identity must contain one native module"
        )
    return dict(payload)


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ensure_artifact(
    lane: str,
    python: Path,
    *,
    output_root: Path,
    process: str,
    model: str,
    color: str,
    execution_mode: str = "compiled",
    jit_optimization_level: int = 3,
    generation_timeout: float,
    regenerate: bool,
    environment: Mapping[str, str],
    lc_flow_layout: str = DEFAULT_LC_FLOW_LAYOUT,
    dependency_site: Path | None = None,
    dependency_site_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    lane_root = output_root / lane
    artifact = lane_root / "artifact"
    cache_path = lane_root / "artifact-cache.json"
    expected_artifact_lc_flow_layout = lc_flow_layout if color == "lc" else None
    generation_command = _generation_command(
        python,
        process=process,
        artifact=artifact,
        model=model,
        color=color,
        execution_mode=execution_mode,
        jit_optimization_level=jit_optimization_level,
        lc_flow_layout=lc_flow_layout,
        dependency_site=dependency_site,
    )
    command_identity = _command_identity(generation_command)
    installation_identity = _installed_pyamplicol_identity(
        python,
        environment=environment,
    )
    signature = _generation_signature(
        python,
        installation_identity=installation_identity,
        process=process,
        model=model,
        color=color,
        execution_mode=execution_mode,
        jit_optimization_level=jit_optimization_level,
        lc_flow_layout=lc_flow_layout,
        dependency_site=dependency_site,
        dependency_site_identity=dependency_site_identity,
        artifact=artifact,
    )
    cache = None if regenerate else _read_cache(cache_path)
    if (
        cache is not None
        and cache.get("kind") == CACHE_KIND
        and cache.get("schema_version") == CACHE_SCHEMA_VERSION
        and cache.get("signature") == signature
    ):
        try:
            metadata = _artifact_metadata(
                artifact,
                expected_process=process,
                expected_color=color,
                expected_execution_mode=execution_mode,
                expected_lc_flow_layout=expected_artifact_lc_flow_layout,
            )
        except RegressionError:
            pass
        else:
            tree_identity = metadata["tree_identity"]
            assert isinstance(tree_identity, Mapping)
            if cache.get("artifact_id") == metadata["artifact_id"] and cache.get(
                "artifact_tree_sha256"
            ) == tree_identity.get("sha256"):
                return {
                    **metadata,
                    "installation_identity": installation_identity,
                    "reused": True,
                    "generation": None,
                    "generation_command": command_identity,
                    "cache_path": str(cache_path),
                }

    lane_root.mkdir(parents=True, exist_ok=True)
    payload, elapsed, stderr = _run_json(
        generation_command,
        timeout=generation_timeout,
        environment=environment,
    )
    metadata = _artifact_metadata(
        artifact,
        expected_process=process,
        expected_color=color,
        expected_execution_mode=execution_mode,
        expected_lc_flow_layout=expected_artifact_lc_flow_layout,
    )
    tree_identity = metadata["tree_identity"]
    assert isinstance(tree_identity, Mapping)
    _write_json_atomic(
        cache_path,
        {
            "kind": CACHE_KIND,
            "schema_version": CACHE_SCHEMA_VERSION,
            "signature": signature,
            "artifact_id": metadata["artifact_id"],
            "artifact_tree_sha256": tree_identity["sha256"],
        },
    )
    return {
        **metadata,
        "installation_identity": installation_identity,
        "reused": False,
        "generation": {
            "command": command_identity,
            "guarded_command": _command_identity(_guarded_command(generation_command)),
            "command_elapsed_seconds": elapsed,
            "peak_rss_gib": _watchdog_peak_gib(stderr),
            "watchdog": _watchdog_marker(stderr),
            "output": payload.get("output"),
            "schema_version": payload.get("schema_version"),
        },
        "generation_command": command_identity,
        "cache_path": str(cache_path),
    }


def _finite_positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise RegressionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RegressionError(f"{label} must be positive and finite")
    return result


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise RegressionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RegressionError(f"{label} must be finite and non-negative")
    return result


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise RegressionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RegressionError(f"{label} must be finite")
    return result


def _positive_mapping_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegressionError(f"{label} must be a positive integer")
    return value


def _require_equal(
    mapping: Mapping[str, Any],
    key: str,
    expected: object,
    *,
    label: str,
) -> None:
    value = mapping.get(key)
    if value != expected or type(value) is not type(expected):
        raise RegressionError(f"{label} must be {expected!r}, got {value!r}")


def _validated_complex_tree(value: object, *, label: str) -> object:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2 and all(
            not isinstance(component, (bool, Sequence))
            and isinstance(component, (float, int))
            for component in value
        ):
            return [
                _finite_number(value[0], label=f"{label} real"),
                _finite_number(value[1], label=f"{label} imaginary"),
            ]
        return [
            _validated_complex_tree(entry, label=f"{label}[{index}]")
            for index, entry in enumerate(value)
        ]
    raise RegressionError(f"{label} must be a nested complex-pair sequence")


def _flatten_complex_tree(value: object) -> list[tuple[float, float]]:
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(isinstance(component, (float, int)) for component in value)
    ):
        return [(float(value[0]), float(value[1]))]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [pair for entry in value for pair in _flatten_complex_tree(entry)]
    raise RegressionError("resolved complex value tree is invalid")


def _validated_complex_array(
    value: object,
    *,
    shape: Sequence[int],
    label: str,
) -> object:
    if not shape:
        return _validated_complex_tree(value, label=label)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RegressionError(f"{label} must be an exact nested complex array")
    expected = shape[0]
    if len(value) != expected:
        raise RegressionError(
            f"{label} axis length differs: {len(value)} != {expected}"
        )
    return [
        _validated_complex_array(
            entry,
            shape=shape[1:],
            label=f"{label}[{index}]",
        )
        for index, entry in enumerate(value)
    ]


def _validated_numerical_result(
    value: object,
    *,
    precision: int,
    helicities: Sequence[str],
    color_flows: Sequence[str],
    required: bool,
) -> dict[str, Any] | None:
    label = f"precision-{precision} numerical result"
    if value is None and not required:
        return None
    if not isinstance(value, Mapping):
        raise RegressionError(f"native sample has no {label}")
    _require_equal(value, "precision", precision, label=f"{label} precision")
    point_count = _positive_mapping_int(
        value,
        "point_count",
        label=f"{label} point_count",
    )
    distinct_point_count = _positive_mapping_int(
        value,
        "distinct_point_count",
        label=f"{label} distinct_point_count",
    )
    point_sha256 = value.get("point_sha256")
    if (
        point_count != VALIDATION_SAMPLE_COUNT
        or distinct_point_count != point_count
        or not isinstance(point_sha256, Sequence)
        or isinstance(point_sha256, (str, bytes))
        or len(point_sha256) != point_count
        or any(
            not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in point_sha256
        )
        or len(set(point_sha256)) != point_count
    ):
        raise RegressionError(
            f"{label} must contain {VALIDATION_SAMPLE_COUNT} distinct points"
        )
    batch_sha256 = value.get("batch_sha256")
    if (
        not isinstance(batch_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", batch_sha256) is None
    ):
        raise RegressionError(f"{label} has an invalid batch SHA-256")
    numerical_values = value.get("values_f64")
    numerical_hex = value.get("values_f64_hex")
    if (
        not isinstance(numerical_values, Sequence)
        or isinstance(numerical_values, (str, bytes))
        or not isinstance(numerical_hex, Sequence)
        or isinstance(numerical_hex, (str, bytes))
        or len(numerical_values) != point_count
        or len(numerical_hex) != point_count
    ):
        raise RegressionError(f"{label} has invalid value vectors")
    validated_values: list[float] = []
    validated_hex: list[str] = []
    for raw_value, raw_hex in zip(numerical_values, numerical_hex, strict=True):
        converted = _finite_number(raw_value, label=f"{label} value")
        if not isinstance(raw_hex, str) or raw_hex != converted.hex():
            raise RegressionError(f"{label} has an invalid raw f64 representation")
        validated_values.append(converted)
        validated_hex.append(raw_hex)
    for key, selector_expected in (
        ("helicities", list(helicities)),
        ("color_flows", list(color_flows)),
    ):
        if value.get(key) != selector_expected:
            raise RegressionError(f"{label} {key} do not match timed selectors")
    resolved = value.get("resolved")
    if not isinstance(resolved, Mapping):
        raise RegressionError(f"{label} has no resolved evidence")
    resolved_shape = resolved.get("shape")
    resolved_helicity_ids = resolved.get("helicity_ids")
    resolved_color_ids = resolved.get("color_ids")
    if (
        not isinstance(resolved_shape, Sequence)
        or isinstance(resolved_shape, (str, bytes))
        or len(resolved_shape) != 3
        or not all(
            isinstance(axis, int) and not isinstance(axis, bool) and axis >= 0
            for axis in resolved_shape
        )
        or not isinstance(resolved_helicity_ids, Sequence)
        or isinstance(resolved_helicity_ids, (str, bytes))
        or not all(isinstance(identifier, str) for identifier in resolved_helicity_ids)
        or not isinstance(resolved_color_ids, Sequence)
        or isinstance(resolved_color_ids, (str, bytes))
        or not all(isinstance(identifier, str) for identifier in resolved_color_ids)
    ):
        raise RegressionError(f"{label} resolved metadata is invalid")
    expected_shape = [
        point_count,
        len(resolved_helicity_ids),
        len(resolved_color_ids),
    ]
    if list(resolved_shape) != expected_shape:
        raise RegressionError(
            f"{label} resolved shape differs: {list(resolved_shape)} "
            f"!= {expected_shape}"
        )
    resolved_totals = _validated_complex_array(
        resolved.get("totals_complex"),
        shape=(point_count,),
        label=f"{label} resolved totals",
    )
    resolved_values = _validated_complex_array(
        resolved.get("values_complex"),
        shape=tuple(expected_shape),
        label=f"{label} resolved values",
    )
    assert isinstance(resolved_totals, list)
    for scalar, pair in zip(validated_values, resolved_totals, strict=True):
        if (
            not isinstance(pair, Sequence)
            or len(pair) != 2
            or not math.isclose(
                scalar,
                float(pair[0]),
                rel_tol=CORRECTNESS_RELATIVE_TOLERANCE,
                abs_tol=CORRECTNESS_ABSOLUTE_TOLERANCE,
            )
            or not math.isclose(
                float(pair[1]),
                0.0,
                rel_tol=0.0,
                abs_tol=CORRECTNESS_ABSOLUTE_TOLERANCE,
            )
        ):
            raise RegressionError(f"{label} evaluate() disagrees with resolved totals")
    return {
        "precision": precision,
        "point_count": point_count,
        "distinct_point_count": distinct_point_count,
        "point_sha256": list(point_sha256),
        "batch_sha256": batch_sha256,
        "values_f64": validated_values,
        "values_f64_hex": validated_hex,
        "helicities": list(helicities),
        "color_flows": list(color_flows),
        "resolved": {
            "shape": list(resolved_shape),
            "helicity_ids": list(resolved_helicity_ids),
            "color_ids": list(resolved_color_ids),
            "totals_complex": resolved_totals,
            "values_complex": resolved_values,
        },
    }


def _validated_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RegressionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validated_canonical_float_hex(
    value: object,
    *,
    label: str,
    positive: bool,
) -> str:
    if not isinstance(value, str):
        raise RegressionError(f"{label} must be a canonical f64 hexadecimal value")
    try:
        converted = float.fromhex(value)
    except ValueError as error:
        raise RegressionError(
            f"{label} must be a canonical f64 hexadecimal value"
        ) from error
    if (
        not math.isfinite(converted)
        or converted.hex() != value
        or (converted <= 0.0 if positive else converted < 0.0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise RegressionError(f"{label} must be canonical, finite, and {qualifier}")
    return value


def _require_exact_keys(
    mapping: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    observed = frozenset(str(key) for key in mapping)
    if observed != expected or any(not isinstance(key, str) for key in mapping):
        raise RegressionError(
            f"{label} fields differ: "
            f"missing={sorted(expected - observed)!r}, "
            f"unexpected={sorted(observed - expected)!r}"
        )


def _validated_correctness_point_derivation(
    value: object,
    *,
    precision16: Mapping[str, Any],
    precision32: Mapping[str, Any] | None,
) -> dict[str, Any]:
    label = "correctness point derivation"
    if not isinstance(value, Mapping):
        raise RegressionError(f"native sample has no {label}")
    _require_exact_keys(
        value,
        frozenset(
            {
                "kind",
                "schema_version",
                "contract",
                "source_validation",
                "numpy",
                "sqrt_s_f64_hex",
                "external_mass_f64_hex",
                "final_state_mass_f64_hex",
                "seed_start",
                "seed_attempt_limit",
                "attempt_count",
                "selected_seeds",
                "point_sha256",
                "invariant_sha256",
                "batch_sha256",
                "sha256",
            }
        ),
        label=label,
    )
    _require_equal(
        value,
        "kind",
        CORRECTNESS_POINT_DERIVATION_KIND,
        label=f"{label} kind",
    )
    _require_equal(
        value,
        "schema_version",
        CORRECTNESS_POINT_DERIVATION_SCHEMA_VERSION,
        label=f"{label} schema_version",
    )
    _require_equal(
        value,
        "contract",
        CORRECTNESS_POINT_DERIVATION_CONTRACT,
        label=f"{label} contract",
    )

    source = value.get("source_validation")
    if not isinstance(source, Mapping):
        raise RegressionError(f"{label} has no authenticated source identity")
    _require_exact_keys(
        source,
        frozenset(
            {
                "file_sha256",
                "canonical_sha256",
                "size_bytes",
                "process_id",
                "process_expression",
                "external_pdgs",
                "point_count",
                "selected_point_index",
                "selected_point_sha256",
            }
        ),
        label=f"{label} source_validation",
    )
    for key in ("file_sha256", "canonical_sha256", "selected_point_sha256"):
        _validated_sha256(
            source.get(key),
            label=f"{label} source_validation {key}",
        )
    source_size = source.get("size_bytes")
    if (
        isinstance(source_size, bool)
        or not isinstance(source_size, int)
        or source_size <= 0
    ):
        raise RegressionError(f"{label} source_validation size_bytes must be positive")
    for key in ("process_id", "process_expression"):
        source_text = source.get(key)
        if not isinstance(source_text, str) or not source_text.strip():
            raise RegressionError(f"{label} source_validation {key} must be non-empty")
    source_pdgs = source.get("external_pdgs")
    if (
        not isinstance(source_pdgs, Sequence)
        or isinstance(source_pdgs, (str, bytes))
        or len(source_pdgs) < 4
        or any(isinstance(pdg, bool) or not isinstance(pdg, int) for pdg in source_pdgs)
    ):
        raise RegressionError(f"{label} source_validation external_pdgs are invalid")
    source_point_count = source.get("point_count")
    if (
        isinstance(source_point_count, bool)
        or not isinstance(source_point_count, int)
        or source_point_count <= 0
    ):
        raise RegressionError(f"{label} source_validation point_count must be positive")
    _require_equal(
        source,
        "selected_point_index",
        0,
        label=f"{label} source_validation selected_point_index",
    )

    numpy_identity = value.get("numpy")
    if not isinstance(numpy_identity, Mapping):
        raise RegressionError(f"{label} has no NumPy identity")
    _require_exact_keys(
        numpy_identity,
        frozenset(
            {
                "module",
                "version",
                "origin_file_name",
                "origin_size_bytes",
                "origin_sha256",
                "random_generator",
                "bit_generator",
            }
        ),
        label=f"{label} NumPy identity",
    )
    _require_equal(
        numpy_identity,
        "module",
        "numpy",
        label=f"{label} NumPy module",
    )
    _require_equal(
        numpy_identity,
        "random_generator",
        "default_rng",
        label=f"{label} NumPy random_generator",
    )
    _require_equal(
        numpy_identity,
        "bit_generator",
        "PCG64",
        label=f"{label} NumPy bit_generator",
    )
    numpy_version = numpy_identity.get("version")
    origin_file_name = numpy_identity.get("origin_file_name")
    if not isinstance(numpy_version, str) or not numpy_version:
        raise RegressionError(f"{label} NumPy version is invalid")
    if (
        not isinstance(origin_file_name, str)
        or not origin_file_name
        or "/" in origin_file_name
        or "\\" in origin_file_name
    ):
        raise RegressionError(f"{label} NumPy origin must be path-free")
    origin_size = numpy_identity.get("origin_size_bytes")
    if (
        isinstance(origin_size, bool)
        or not isinstance(origin_size, int)
        or origin_size <= 0
    ):
        raise RegressionError(f"{label} NumPy origin size is invalid")
    _validated_sha256(
        numpy_identity.get("origin_sha256"),
        label=f"{label} NumPy origin_sha256",
    )

    sqrt_s_hex = _validated_canonical_float_hex(
        value.get("sqrt_s_f64_hex"),
        label=f"{label} sqrt_s_f64_hex",
        positive=True,
    )
    sqrt_s = float.fromhex(sqrt_s_hex)
    external_mass_hex = value.get("external_mass_f64_hex")
    final_mass_hex = value.get("final_state_mass_f64_hex")
    if (
        not isinstance(external_mass_hex, Sequence)
        or isinstance(external_mass_hex, (str, bytes))
        or len(external_mass_hex) != len(source_pdgs)
        or not isinstance(final_mass_hex, Sequence)
        or isinstance(final_mass_hex, (str, bytes))
        or len(final_mass_hex) != len(source_pdgs) - 2
    ):
        raise RegressionError(f"{label} mass vectors have invalid lengths")
    validated_external_masses = [
        _validated_canonical_float_hex(
            mass,
            label=f"{label} external mass {index}",
            positive=False,
        )
        for index, mass in enumerate(external_mass_hex)
    ]
    validated_final_masses = [
        _validated_canonical_float_hex(
            mass,
            label=f"{label} final-state mass {index}",
            positive=False,
        )
        for index, mass in enumerate(final_mass_hex)
    ]
    if validated_final_masses != validated_external_masses[2:]:
        raise RegressionError(f"{label} final-state masses are inconsistent")
    if math.fsum(float.fromhex(mass) for mass in validated_final_masses) >= sqrt_s:
        raise RegressionError(f"{label} final-state masses are not below threshold")

    _require_equal(
        value,
        "seed_start",
        CORRECTNESS_SEED_START,
        label=f"{label} seed_start",
    )
    _require_equal(
        value,
        "seed_attempt_limit",
        MAX_CORRECTNESS_SEED_ATTEMPTS,
        label=f"{label} seed_attempt_limit",
    )
    attempts = value.get("attempt_count")
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or not VALIDATION_SAMPLE_COUNT <= attempts <= MAX_CORRECTNESS_SEED_ATTEMPTS
    ):
        raise RegressionError(f"{label} attempt_count is invalid")
    selected_seeds = value.get("selected_seeds")
    if (
        not isinstance(selected_seeds, Sequence)
        or isinstance(selected_seeds, (str, bytes))
        or len(selected_seeds) != VALIDATION_SAMPLE_COUNT
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in selected_seeds
        )
        or list(selected_seeds) != sorted(set(selected_seeds))
        or selected_seeds[0] < CORRECTNESS_SEED_START
        or selected_seeds[-1] >= CORRECTNESS_SEED_START + MAX_CORRECTNESS_SEED_ATTEMPTS
        or selected_seeds[-1] != CORRECTNESS_SEED_START + attempts - 1
    ):
        raise RegressionError(f"{label} selected_seeds are invalid")

    hashes: dict[str, list[str]] = {}
    for key in ("point_sha256", "invariant_sha256"):
        raw_hashes = value.get(key)
        if (
            not isinstance(raw_hashes, Sequence)
            or isinstance(raw_hashes, (str, bytes))
            or len(raw_hashes) != VALIDATION_SAMPLE_COUNT
        ):
            raise RegressionError(f"{label} {key} has the wrong length")
        validated_hashes = [
            _validated_sha256(digest, label=f"{label} {key}[{index}]")
            for index, digest in enumerate(raw_hashes)
        ]
        if len(set(validated_hashes)) != VALIDATION_SAMPLE_COUNT:
            raise RegressionError(f"{label} {key} values are not distinct")
        hashes[key] = validated_hashes
    batch_sha256 = _validated_sha256(
        value.get("batch_sha256"),
        label=f"{label} batch_sha256",
    )
    for numerical_label, numerical in (
        ("precision-16", precision16),
        ("precision-32", precision32),
    ):
        if numerical is None:
            continue
        if numerical.get("point_sha256") != hashes["point_sha256"]:
            raise RegressionError(
                f"{label} point hashes differ from {numerical_label} evidence"
            )
        if numerical.get("batch_sha256") != batch_sha256:
            raise RegressionError(
                f"{label} batch digest differs from {numerical_label} evidence"
            )
    derivation_sha256 = _validated_sha256(
        value.get("sha256"),
        label=f"{label} sha256",
    )
    digest_basis = {key: entry for key, entry in value.items() if key != "sha256"}
    if _canonical_sha256(digest_basis) != derivation_sha256:
        raise RegressionError(f"{label} digest does not authenticate its metadata")
    return {
        **digest_basis,
        "sha256": derivation_sha256,
    }


def _native_profile_sample(
    payload: Mapping[str, Any],
    *,
    minimum_samples: int,
    batch_size: int | None = None,
    expected_execution_mode: str = "compiled",
    require_precision32: bool = False,
    require_arena_profile: bool = False,
) -> dict[str, Any]:
    _require_equal(
        payload,
        "kind",
        NATIVE_SAMPLE_RESULT_KIND,
        label="native sample kind",
    )
    _require_equal(
        payload,
        "schema_version",
        NATIVE_SAMPLE_SCHEMA_VERSION,
        label="native sample schema_version",
    )
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise RegressionError("profile result has no environment metadata")
    source = environment.get("wall_time_source")
    if source != NATIVE_WALL_TIME_SOURCE:
        raise RegressionError(
            f"profile did not use the native Rusticol repeated wall timer: {source!r}"
        )
    _require_equal(
        environment,
        "wall_time_sample_pass",
        NATIVE_WALL_TIME_SAMPLE_PASS,
        label="profile wall_time_sample_pass",
    )
    _require_equal(
        environment,
        "timing_sample_contract",
        PAIRED_TIMING_SAMPLE_CONTRACT,
        label="profile timing_sample_contract",
    )
    evaluator_sample_pass = environment.get("evaluator_time_sample_pass")
    timing_breakdown_sample_pass = environment.get("timing_breakdown_sample_pass")
    if evaluator_sample_pass != timing_breakdown_sample_pass:
        raise RegressionError(
            "profile evaluator and timing-breakdown attribution passes differ"
        )
    allowed_profile_passes = (
        {PROFILE_ATTRIBUTION_SAMPLE_PASS}
        if require_arena_profile
        else {
            PROFILE_ATTRIBUTION_SAMPLE_PASS,
            FROZEN_BASELINE_PROFILE_ATTRIBUTION_SAMPLE_PASS,
        }
    )
    if evaluator_sample_pass not in allowed_profile_passes:
        raise RegressionError(
            "profile attribution did not use the required warmed Arena boundary: "
            f"{evaluator_sample_pass!r}"
        )
    profile_attribution_boundary = environment.get("profile_attribution_boundary")
    profile_attribution_borrowed_flat_input = environment.get(
        "profile_attribution_borrowed_flat_input"
    )
    profile_attribution_preallocated_output = environment.get(
        "profile_attribution_preallocated_output"
    )
    profile_attribution_phase_timing_scope = environment.get(
        "profile_attribution_phase_timing_scope"
    )
    profile_attribution_evaluator_timing_available = environment.get(
        "profile_attribution_evaluator_timing_available"
    )
    profile_protocol = environment.get("profile_protocol")
    if evaluator_sample_pass == PROFILE_ATTRIBUTION_SAMPLE_PASS:
        expected_profile_boundary = ARENA_PROFILE_BOUNDARY
        expected_borrowed_input = True
        expected_preallocated_output = True
        expected_phase_timing_scope = ARENA_PHASE_TIMING_SCOPE
        expected_evaluator_timing_available = False
        expected_profile_protocol = ARENA_PROFILE_PROTOCOL
    else:
        expected_profile_boundary = FROZEN_BASELINE_PROFILE_BOUNDARY
        expected_borrowed_input = False
        expected_preallocated_output = False
        expected_phase_timing_scope = FROZEN_BASELINE_PHASE_TIMING_SCOPE
        expected_evaluator_timing_available = True
        expected_profile_protocol = FROZEN_BASELINE_PROFILE_PROTOCOL
    if (
        profile_attribution_boundary != expected_profile_boundary
        or profile_attribution_borrowed_flat_input is not expected_borrowed_input
        or profile_attribution_preallocated_output is not expected_preallocated_output
        or profile_attribution_phase_timing_scope != expected_phase_timing_scope
        or profile_attribution_evaluator_timing_available
        is not expected_evaluator_timing_available
        or profile_protocol != expected_profile_protocol
    ):
        raise RegressionError(
            "profile attribution boundary metadata does not match its sample pass"
        )
    for key in (
        "profile_attribution_paired_with_headline",
        "profile_attribution_identical_batch",
        "profile_attribution_identical_repetitions",
    ):
        _require_equal(
            environment,
            key,
            True,
            label=f"profile {key}",
        )
    _require_equal(
        environment,
        "execution_mode",
        expected_execution_mode,
        label="profile execution_mode",
    )
    sample_count = payload.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < minimum_samples
    ):
        raise RegressionError(
            "profile result contains fewer complete timing blocks than requested"
        )
    headline_samples = payload.get("wall_time_samples_seconds_per_point")
    if not isinstance(headline_samples, Sequence) or isinstance(
        headline_samples,
        (str, bytes),
    ):
        raise RegressionError("native sample has no headline wall sample vector")
    if len(headline_samples) != sample_count:
        raise RegressionError(
            "native headline wall sample vector does not match sample_count"
        )
    headline_values = [
        _finite_positive(value, label="native headline wall sample")
        for value in headline_samples
    ]
    _require_equal(
        payload,
        "interrupted",
        False,
        label="profile interrupted",
    )
    _require_equal(
        environment,
        "interrupted",
        False,
        label="profile environment interrupted",
    )
    repetitions = _positive_mapping_int(
        payload,
        "repetitions_per_sample",
        label="profile repetitions_per_sample",
    )
    reported_batch_size = _positive_mapping_int(
        environment,
        "batch_size",
        label="profile environment batch_size",
    )
    if batch_size is not None and reported_batch_size != batch_size:
        raise RegressionError(
            "profile environment batch_size does not match the requested batch: "
            f"{reported_batch_size} != {batch_size}"
        )
    expected_evaluations = sample_count * repetitions
    expected_points = expected_evaluations * reported_batch_size
    expected_points_per_sample = repetitions * reported_batch_size
    exact_counts = {
        "completed_sample_count": sample_count,
        "planned_sample_count": sample_count,
        "measured_evaluation_count": expected_evaluations,
        "measured_point_count": expected_points,
        "native_profile_sample_count": sample_count,
        "native_profile_sample_limit": sample_count,
        "native_profile_repetitions_per_sample": repetitions,
        "native_profile_points_per_sample": expected_points_per_sample,
        "profile_attribution_evaluation_count": expected_evaluations,
        "profile_attribution_point_count": expected_points,
    }
    for key, expected in exact_counts.items():
        _require_equal(
            environment,
            key,
            expected,
            label=f"profile environment {key}",
        )
    calls_per_block = _finite_positive(
        environment.get("native_profile_calls_per_block"),
        label="profile environment native_profile_calls_per_block",
    )
    if calls_per_block != 1.0:
        raise RegressionError(
            "profile native attribution must contain exactly one paired call "
            f"per headline block, got {calls_per_block!r}"
        )
    timing_breakdown = payload.get("timing_breakdown")
    if not isinstance(timing_breakdown, Mapping):
        raise RegressionError("profile result has no paired native timing breakdown")
    _require_equal(
        timing_breakdown,
        "sample_count",
        sample_count,
        label="profile timing breakdown sample_count",
    )
    _require_equal(
        timing_breakdown,
        "execution_mode",
        expected_execution_mode,
        label="profile timing breakdown execution_mode",
    )
    profile_wall = timing_breakdown.get("wall_time")
    if not isinstance(profile_wall, Mapping):
        raise RegressionError("profile timing breakdown has no profiled wall component")
    profile_wall_seconds_per_point = _finite_positive(
        profile_wall.get("mean_seconds_per_point"),
        label="paired profiled wall mean_seconds_per_point",
    )
    profile_evaluator_breakdown = timing_breakdown.get("evaluator_call_time")
    if expected_evaluator_timing_available:
        profile_evaluator_seconds_per_point: float | None = _finite_nonnegative(
            payload.get("evaluator_time_per_point"),
            label="paired profile evaluator_time_per_point",
        )
        if not isinstance(profile_evaluator_breakdown, Mapping):
            raise RegressionError(
                "legacy paired profile has no evaluator timing component"
            )
        if not isinstance(payload.get("evaluator_uncertainty"), Mapping):
            raise RegressionError(
                "legacy paired profile has no evaluator timing uncertainty"
            )
    else:
        if (
            payload.get("evaluator_time_per_point") is not None
            or payload.get("evaluator_uncertainty") is not None
            or profile_evaluator_breakdown is not None
        ):
            raise RegressionError(
                "coarse Arena boundary must record evaluator phase timing "
                "as unavailable"
            )
        profile_evaluator_seconds_per_point = None
    cold_load_seconds = _finite_nonnegative(
        payload.get("cold_load_seconds"),
        label="native cold_load_seconds",
    )
    headline_mean = _finite_positive(
        payload.get("wall_time_per_point"),
        label="profile wall_time_per_point",
    )
    computed_headline_mean = float(statistics.fmean(headline_values))
    if not math.isclose(
        headline_mean,
        computed_headline_mean,
        rel_tol=1.0e-15,
        abs_tol=0.0,
    ):
        raise RegressionError(
            "native headline mean does not match its unprofiled sample vector"
        )
    raw_profiles = timing_breakdown.get("raw_profile_samples")
    if not isinstance(raw_profiles, Sequence) or isinstance(raw_profiles, (str, bytes)):
        raise RegressionError("paired timing breakdown has no raw profile samples")
    if len(raw_profiles) != sample_count:
        raise RegressionError(
            "paired raw profile sample vector does not match sample_count"
        )
    if evaluator_sample_pass == PROFILE_ATTRIBUTION_SAMPLE_PASS:
        for profile_index, raw_profile in enumerate(raw_profiles):
            if not isinstance(raw_profile, Mapping):
                raise RegressionError(
                    f"paired raw Arena profile {profile_index} is invalid"
                )
            if (
                raw_profile.get("profile_boundary") != ARENA_PROFILE_BOUNDARY
                or raw_profile.get("borrowed_flat_input") is not True
                or raw_profile.get("preallocated_output") is not True
                or raw_profile.get("phase_timing_scope") != ARENA_PHASE_TIMING_SCOPE
                or raw_profile.get("evaluator_timing_available") is not False
            ):
                raise RegressionError(
                    "paired raw Arena profile does not authenticate its "
                    "borrowed-input, preallocated-output, coarse-timing boundary"
                )
    runtime_identity = environment.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping):
        raise RegressionError("native sample has no runtime identity")
    native_module_identity = runtime_identity.get("native_module")
    if not isinstance(native_module_identity, Mapping):
        raise RegressionError("native sample has no native module identity")
    native_module_sha256 = native_module_identity.get("sha256")
    if (
        not isinstance(native_module_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", native_module_sha256) is None
    ):
        raise RegressionError("native sample has an invalid native module SHA-256")
    batch_sha256 = environment.get("batch_sha256")
    if (
        not isinstance(batch_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", batch_sha256) is None
    ):
        raise RegressionError("native sample has an invalid batch SHA-256")
    helicities = environment.get("helicities")
    color_flows = environment.get("color_flows")
    if (
        not isinstance(helicities, Sequence)
        or isinstance(helicities, (str, bytes))
        or not all(isinstance(value, str) for value in helicities)
    ):
        raise RegressionError("native sample has invalid helicity selectors")
    if (
        not isinstance(color_flows, Sequence)
        or isinstance(color_flows, (str, bytes))
        or not all(isinstance(value, str) for value in color_flows)
    ):
        raise RegressionError("native sample has invalid color-flow selectors")
    warmed_numerical_result = _validated_numerical_result(
        payload.get("warmed_numerical_result"),
        precision=16,
        helicities=helicities,
        color_flows=color_flows,
        required=True,
    )
    assert warmed_numerical_result is not None
    precision32_numerical_result = _validated_numerical_result(
        payload.get("precision32_numerical_result"),
        precision=32,
        helicities=helicities,
        color_flows=color_flows,
        required=require_precision32,
    )
    correctness_point_derivation = _validated_correctness_point_derivation(
        payload.get("correctness_point_derivation"),
        precision16=warmed_numerical_result,
        precision32=precision32_numerical_result,
    )
    return {
        "wall_seconds_per_point": headline_mean,
        "wall_samples_seconds_per_point": headline_values,
        "native_wall_time_source": source,
        "native_wall_time_sample_pass": NATIVE_WALL_TIME_SAMPLE_PASS,
        "profile_attribution_sample_pass": evaluator_sample_pass,
        "profile_attribution_boundary": profile_attribution_boundary,
        "profile_attribution_borrowed_flat_input": (
            profile_attribution_borrowed_flat_input
        ),
        "profile_attribution_preallocated_output": (
            profile_attribution_preallocated_output
        ),
        "profile_attribution_phase_timing_scope": (
            profile_attribution_phase_timing_scope
        ),
        "profile_attribution_evaluator_timing_available": (
            profile_attribution_evaluator_timing_available
        ),
        "profile_protocol": profile_protocol,
        "timing_sample_contract": PAIRED_TIMING_SAMPLE_CONTRACT,
        "profile_timed_block_count": sample_count,
        "repetitions_per_timed_block": repetitions,
        "batch_size": reported_batch_size,
        "measured_evaluation_count": expected_evaluations,
        "measured_point_count": expected_points,
        "native_elapsed_seconds": environment.get("elapsed_seconds"),
        "cold_load_seconds": cold_load_seconds,
        "profile_uncertainty": payload.get("uncertainty"),
        "paired_profile_wall_seconds_per_point": (profile_wall_seconds_per_point),
        "paired_profile_evaluator_seconds_per_point": (
            profile_evaluator_seconds_per_point
        ),
        "paired_profile_evaluator_uncertainty": payload.get("evaluator_uncertainty"),
        "paired_profile_timing_breakdown": timing_breakdown,
        "runtime_identity": runtime_identity,
        "native_module_sha256": native_module_sha256,
        "batch_sha256": batch_sha256,
        "helicities": list(helicities),
        "color_flows": list(color_flows),
        "correctness_point_derivation": correctness_point_derivation,
        "warmed_numerical_result": warmed_numerical_result,
        "precision32_numerical_result": precision32_numerical_result,
    }


def _distribution(samples: Sequence[float]) -> dict[str, Any]:
    if not samples:
        raise RegressionError("timing distribution must contain samples")
    values = tuple(_finite_positive(value, label="timing sample") for value in samples)
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    return {
        "sample_count": len(values),
        "samples_seconds_per_point": list(values),
        "median_seconds_per_point": median,
        "mad_seconds_per_point": mad,
        "minimum_seconds_per_point": min(values),
        "maximum_seconds_per_point": max(values),
    }


def _regression_gate(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_median = _finite_positive(
        baseline.get("median_seconds_per_point"),
        label="baseline median",
    )
    current_median = _finite_positive(
        current.get("median_seconds_per_point"),
        label="current median",
    )
    raw_mad = baseline.get("mad_seconds_per_point")
    if isinstance(raw_mad, bool) or not isinstance(raw_mad, (float, int)):
        raise RegressionError("baseline MAD must be numeric")
    baseline_mad = float(raw_mad)
    if not math.isfinite(baseline_mad) or baseline_mad < 0.0:
        raise RegressionError("baseline MAD must be finite and non-negative")
    relative_limit = baseline_median * (1.0 + RELATIVE_TOLERANCE)
    mad_limit = baseline_median + MAD_MULTIPLIER * baseline_mad
    within_relative = current_median <= relative_limit
    within_mad = current_median <= mad_limit
    return {
        "relative_tolerance": RELATIVE_TOLERANCE,
        "mad_multiplier": MAD_MULTIPLIER,
        "baseline_median_seconds_per_point": baseline_median,
        "baseline_mad_seconds_per_point": baseline_mad,
        "current_median_seconds_per_point": current_median,
        "current_over_baseline": current_median / baseline_median,
        "relative_change": (current_median - baseline_median) / baseline_median,
        "three_percent_upper_bound_seconds_per_point": relative_limit,
        "three_baseline_mad_upper_bound_seconds_per_point": mad_limit,
        "within_three_percent": within_relative,
        "within_three_baseline_mad": within_mad,
        "passes": within_relative and within_mad,
    }


def _paired_distribution(
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pairs: dict[int, dict[str, float]] = {}
    for measurement in measurements:
        pair_index = measurement.get("pair_index")
        lane = measurement.get("lane")
        if (
            isinstance(pair_index, bool)
            or not isinstance(pair_index, int)
            or lane not in {"baseline", "current"}
        ):
            raise RegressionError("timing measurement has an invalid pair identity")
        pairs.setdefault(pair_index, {})[str(lane)] = _finite_positive(
            measurement.get("wall_seconds_per_point"),
            label="paired wall time",
        )
    if not pairs or any(
        set(pair) != {"baseline", "current"} for pair in pairs.values()
    ):
        raise RegressionError("timing measurements do not form complete lane pairs")
    differences = [
        pair["baseline"] - pair["current"] for _, pair in sorted(pairs.items())
    ]
    ratios = [pair["current"] / pair["baseline"] for _, pair in sorted(pairs.items())]
    difference_median = float(statistics.median(differences))
    difference_mad = float(
        statistics.median(abs(value - difference_median) for value in differences)
    )
    return {
        "pair_count": len(pairs),
        "differences_seconds_per_point": differences,
        "difference_median_seconds_per_point": difference_median,
        "difference_mad_seconds_per_point": difference_mad,
        "current_over_baseline_ratios": ratios,
        "median_current_over_baseline": float(statistics.median(ratios)),
    }


def _gain_gate(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    paired: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_median = _finite_positive(
        baseline.get("median_seconds_per_point"),
        label="baseline median",
    )
    current_median = _finite_positive(
        current.get("median_seconds_per_point"),
        label="current median",
    )
    baseline_mad = _finite_nonnegative(
        baseline.get("mad_seconds_per_point"),
        label="baseline MAD",
    )
    paired_difference_median = _finite_number(
        paired.get("difference_median_seconds_per_point"),
        label="paired difference median",
    )
    paired_difference_mad = _finite_nonnegative(
        paired.get("difference_mad_seconds_per_point"),
        label="paired difference MAD",
    )
    relative_gain = 1.0 - current_median / baseline_median
    noise_floor = MAD_MULTIPLIER * max(baseline_mad, paired_difference_mad)
    beyond_noise = (
        baseline_median - current_median >= noise_floor
        and paired_difference_median >= MAD_MULTIPLIER * paired_difference_mad
    )
    return {
        "required_relative_gain": GAIN_RELATIVE_THRESHOLD,
        "relative_gain": relative_gain,
        "baseline_minus_current_seconds_per_point": baseline_median - current_median,
        "noise_floor_seconds_per_point": noise_floor,
        "paired_difference_median_seconds_per_point": paired_difference_median,
        "paired_difference_mad_seconds_per_point": paired_difference_mad,
        "at_least_ten_percent": relative_gain >= GAIN_RELATIVE_THRESHOLD,
        "beyond_measurement_noise": beyond_noise,
        "passes": relative_gain >= GAIN_RELATIVE_THRESHOLD and beyond_noise,
    }


def _performance_payload_identity(
    artifact: Mapping[str, Any],
) -> dict[str, Any] | None:
    payloads = artifact.get("payload_digests")
    if not isinstance(payloads, Sequence) or isinstance(payloads, (str, bytes)):
        return None
    relevant: list[dict[str, object]] = []
    observed_roles: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, Mapping):
            return None
        role = payload.get("role")
        if role not in PERFORMANCE_RELEVANT_PAYLOAD_ROLES:
            continue
        path = payload.get("path")
        sha256 = payload.get("sha256")
        size_bytes = payload.get("size_bytes")
        if (
            not isinstance(role, str)
            or not isinstance(path, str)
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            return None
        observed_roles.add(role)
        relevant.append(
            {
                "path": path,
                "role": role,
                "process_id": payload.get("process_id"),
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    if not REQUIRED_PERFORMANCE_PAYLOAD_ROLES.issubset(observed_roles):
        return None
    relevant.sort(key=lambda value: (str(value["role"]), str(value["path"])))
    identity_basis = {
        "process_id": artifact.get("process_id"),
        "color_accuracy": artifact.get("color_accuracy"),
        "lc_flow_layout": artifact.get("lc_flow_layout"),
        "payloads": relevant,
    }
    return {
        "algorithm": "sha256-performance-relevant-artifact-payloads-v1",
        "sha256": _canonical_sha256(identity_basis),
        "roles": sorted(observed_roles),
        "payload_count": len(relevant),
        "identity_basis": identity_basis,
    }


def _performance_authority(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    shared_artifact: bool,
) -> dict[str, Any]:
    if shared_artifact:
        return {
            "authoritative": True,
            "basis": "single-read-only-shared-artifact",
            "comparison_mode": "shared-artifact",
            "payload_identities": None,
            "semantic_identities": None,
        }
    semantic_identities = {
        lane: artifact.get("semantic_workload_identity")
        for lane, artifact in artifacts.items()
    }
    if all(
        isinstance(identity, Mapping) and isinstance(identity.get("sha256"), str)
        for identity in semantic_identities.values()
    ):
        semantic_digests = {
            str(identity["sha256"])
            for identity in semantic_identities.values()
            if isinstance(identity, Mapping)
        }
        if len(semantic_digests) == 1:
            return {
                "authoritative": True,
                "basis": "matching-abi-neutral-semantic-workload-identities",
                "comparison_mode": "independently-generated-artifacts",
                "payload_identities": None,
                "semantic_identities": semantic_identities,
            }
    identities = {
        lane: _performance_payload_identity(artifact)
        for lane, artifact in artifacts.items()
    }
    if any(identity is None for identity in identities.values()):
        return {
            "authoritative": False,
            "basis": "per-lane-artifact-performance-payload-identity-unproven",
            "comparison_mode": "independently-generated-artifacts",
            "payload_identities": identities,
            "semantic_identities": semantic_identities,
        }
    digests = {
        str(identity["sha256"])
        for identity in identities.values()
        if identity is not None
    }
    matches = len(digests) == 1
    return {
        "authoritative": matches,
        "basis": (
            "matching-performance-relevant-payload-identities"
            if matches
            else "per-lane-artifact-performance-payload-identities-differ"
        ),
        "comparison_mode": "independently-generated-artifacts",
        "payload_identities": identities,
        "semantic_identities": semantic_identities,
    }


def _numerical_comparison(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    kind: str,
    lane: object,
    pair_index: object,
) -> dict[str, Any]:
    for key in (
        "point_count",
        "distinct_point_count",
        "point_sha256",
        "batch_sha256",
        "helicities",
        "color_flows",
    ):
        if observed.get(key) != expected.get(key):
            raise RegressionError(
                f"{kind} numerical results do not share identical inputs: {key} differs"
            )
    expected_resolved = expected.get("resolved")
    observed_resolved = observed.get("resolved")
    if not isinstance(expected_resolved, Mapping) or not isinstance(
        observed_resolved,
        Mapping,
    ):
        raise RegressionError(f"{kind} numerical result has no resolved evidence")
    for key in ("shape", "helicity_ids", "color_ids"):
        if observed_resolved.get(key) != expected_resolved.get(key):
            raise RegressionError(f"{kind} resolved evidence differs in {key}")
    expected_values = expected.get("values_f64")
    observed_values = observed.get("values_f64")
    if (
        not isinstance(expected_values, Sequence)
        or isinstance(expected_values, (str, bytes))
        or not isinstance(observed_values, Sequence)
        or isinstance(observed_values, (str, bytes))
        or len(expected_values) != len(observed_values)
    ):
        raise RegressionError(f"{kind} numerical scalar shape differs")
    vector_groups = (
        (
            [(float(value), 0.0) for value in expected_values],
            [(float(value), 0.0) for value in observed_values],
        ),
        (
            _flatten_complex_tree(expected_resolved.get("totals_complex")),
            _flatten_complex_tree(observed_resolved.get("totals_complex")),
        ),
        (
            _flatten_complex_tree(expected_resolved.get("values_complex")),
            _flatten_complex_tree(observed_resolved.get("values_complex")),
        ),
    )
    passes = True
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for expected_vector, observed_vector in vector_groups:
        if len(expected_vector) != len(observed_vector):
            raise RegressionError(f"{kind} numerical evidence shape differs")
        for expected_pair, observed_pair in zip(
            expected_vector,
            observed_vector,
            strict=True,
        ):
            for expected_component, observed_component in zip(
                expected_pair,
                observed_pair,
                strict=True,
            ):
                expected_number = _finite_number(
                    expected_component,
                    label=f"{kind} expected numerical component",
                )
                observed_number = _finite_number(
                    observed_component,
                    label=f"{kind} observed numerical component",
                )
                absolute_error = abs(observed_number - expected_number)
                relative_error = absolute_error / max(
                    abs(expected_number),
                    CORRECTNESS_ABSOLUTE_TOLERANCE,
                )
                maximum_absolute = max(maximum_absolute, absolute_error)
                maximum_relative = max(maximum_relative, relative_error)
                passes &= math.isclose(
                    observed_number,
                    expected_number,
                    rel_tol=CORRECTNESS_RELATIVE_TOLERANCE,
                    abs_tol=CORRECTNESS_ABSOLUTE_TOLERANCE,
                )
    return {
        "kind": kind,
        "lane": lane,
        "pair_index": pair_index,
        "passes": passes,
        "maximum_absolute_error": maximum_absolute,
        "maximum_relative_error": maximum_relative,
    }


def _correctness_gate(
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_reference = next(
        (
            measurement
            for measurement in measurements
            if measurement.get("lane") == "baseline"
        ),
        None,
    )
    if baseline_reference is None:
        raise RegressionError("correctness comparison has no baseline measurement")
    reference16 = baseline_reference.get("warmed_numerical_result")
    if not isinstance(reference16, Mapping):
        raise RegressionError("baseline measurement has no warmed numerical result")
    reference_derivation = baseline_reference.get("correctness_point_derivation")
    if not isinstance(reference_derivation, Mapping):
        raise RegressionError(
            "baseline measurement has no correctness point derivation"
        )
    comparisons: list[dict[str, Any]] = []
    for measurement in measurements:
        if measurement.get("correctness_point_derivation") != reference_derivation:
            raise RegressionError(
                "correctness measurements do not share a byte-identical "
                "point derivation"
            )
        candidate16 = measurement.get("warmed_numerical_result")
        if not isinstance(candidate16, Mapping):
            raise RegressionError("measurement has no warmed numerical result")
        comparisons.append(
            _numerical_comparison(
                reference16,
                candidate16,
                kind="precision16-cross-lane",
                lane=measurement.get("lane"),
                pair_index=measurement.get("pair_index"),
            )
        )
    precision32_measurements = [
        measurement
        for measurement in measurements
        if isinstance(measurement.get("precision32_numerical_result"), Mapping)
    ]
    if (
        len(precision32_measurements) != 1
        or precision32_measurements[0].get("lane") != "current"
        or precision32_measurements[0].get("pair_index") != 1
    ):
        raise RegressionError(
            "correctness comparison requires exactly one precision-32 "
            "exact-oracle result from current pair 1 and none from baseline"
        )
    current32_measurement = precision32_measurements[0]
    reference32 = current32_measurement["precision32_numerical_result"]
    assert isinstance(reference32, Mapping)
    direct_oracle_comparison_count = 0
    for measurement in measurements:
        candidate16 = measurement.get("warmed_numerical_result")
        if not isinstance(candidate16, Mapping):
            raise RegressionError("measurement has no warmed numerical result")
        comparisons.append(
            _numerical_comparison(
                reference32,
                candidate16,
                kind="precision16-vs-current-exact-precision32",
                lane=measurement.get("lane"),
                pair_index=measurement.get("pair_index"),
            )
        )
        direct_oracle_comparison_count += 1
    return {
        "relative_tolerance": CORRECTNESS_RELATIVE_TOLERANCE,
        "absolute_tolerance": CORRECTNESS_ABSOLUTE_TOLERANCE,
        "reference_lane": "baseline",
        "reference_pair_index": baseline_reference.get("pair_index"),
        "point_derivation_sha256": reference_derivation.get("sha256"),
        "correctness_batch_sha256": reference_derivation.get("batch_sha256"),
        "precision32_lane_count": 1,
        "precision32_measurement_count": 1,
        "precision32_policy": PRECISION32_CORRECTNESS_POLICY,
        "precision32_authority_basis": (
            "final-candidate-current-exact-oracle-directly-validates-every-"
            "f64-measurement"
        ),
        "precision32_reference_lane": "current",
        "precision32_reference_pair_index": current32_measurement.get("pair_index"),
        "precision32_direct_oracle_scope": "every-f64-measurement",
        "precision32_direct_oracle_comparison_count": (direct_oracle_comparison_count),
        "comparison_count": len(comparisons),
        "maximum_absolute_error": max(
            (float(comparison["maximum_absolute_error"]) for comparison in comparisons),
            default=0.0,
        ),
        "maximum_relative_error": max(
            (float(comparison["maximum_relative_error"]) for comparison in comparisons),
            default=0.0,
        ),
        "comparisons": comparisons,
        "passes": all(bool(comparison["passes"]) for comparison in comparisons),
    }


_ZERO_ARENA_PROFILE_COUNTERS = (
    "native_input_container_allocation_count",
    "stage_input_copy_component_count",
    "stage_leaf_input_copy_component_count",
    "stage_evaluator_output_gather_component_count",
    "stage_output_assign_component_count",
    "amplitude_input_copy_component_count",
    "amplitude_leaf_input_copy_component_count",
    "amplitude_evaluator_output_gather_component_count",
    "amplitude_output_remap_component_count",
    "selector_gather_point_count",
    "selector_gather_bytes",
    "selector_scatter_value_count",
    "observed_scratch_reallocation_count",
    "native_output_allocation_count",
)
_ZERO_COMPILED_BOUNDARY_COUNTERS = (
    "compiled_direct_arena_boundary_input_bytes",
    "compiled_direct_arena_boundary_current_output_bytes",
    "compiled_direct_arena_boundary_amplitude_output_bytes",
)
_ZERO_ARENA_PROFILE_TIMES = (
    "native_input_pack_time_s",
    "native_input_crossing_time_s",
    "state_prepare_time_s",
    "state_clear_time_s",
    "source_fill_time_s",
    "momentum_input_setup_time_s",
    "momentum_setup_time_s",
    "model_parameter_setup_time_s",
    "stage_input_pack_time_s",
    "stage_leaf_input_pack_time_s",
    "stage_evaluator_call_time_s",
    "stage_evaluator_time_s",
    "stage_backend_call_time_s",
    "stage_evaluator_output_gather_time_s",
    "output_assign_time_s",
    "amplitude_input_pack_time_s",
    "amplitude_leaf_input_pack_time_s",
    "amplitude_evaluator_call_time_s",
    "amplitude_backend_call_time_s",
    "amplitude_evaluator_output_gather_time_s",
    "amplitude_output_remap_time_s",
    "amplitude_evaluator_time_s",
    "reduction_time_s",
    "resolved_reduction_materialization_inclusive_time_s",
    "total_materialization_time_s",
    "final_output_copy_time_s",
    "eager_initialize_time_s",
    "eager_gather_time_s",
    "eager_kernel_call_time_s",
    "eager_invocation_scatter_time_s",
    "eager_finalization_time_s",
    "eager_scatter_finalization_time_s",
    "eager_closure_time_s",
    "eager_reduction_time_s",
    "eager_copy_out_time_s",
    "recurrence_momentum_fill_time_s",
    "recurrence_union_source_fill_time_s",
    "recurrence_schedule_time_s",
    "recurrence_source_kernel_time_s",
    "recurrence_contribution_kernel_time_s",
    "recurrence_finalization_time_s",
    "recurrence_closure_time_s",
    "recurrence_replay_output_mapping_time_s",
    "selector_planner_time_s",
    "selector_gather_time_s",
    "selector_scatter_time_s",
)
_EMPTY_ARENA_PROFILE_PHASE_VECTORS = (
    "stage_input_pack_by_stage_time_s",
    "stage_leaf_input_pack_by_stage_time_s",
    "stage_evaluator_call_by_stage_time_s",
    "stage_backend_call_by_stage_time_s",
    "stage_evaluator_output_gather_by_stage_time_s",
    "stage_output_assign_by_stage_time_s",
)


def _arena_profile_gate(
    measurements: Sequence[Mapping[str, Any]],
    *,
    execution_mode: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    minimum_profile_timed_blocks: int = 1,
) -> dict[str, Any]:
    if minimum_profile_timed_blocks <= 0:
        raise RegressionError("minimum Arena profile timed blocks must be positive")
    failures: list[dict[str, object]] = []
    expected_capability = (
        EAGER_DIRECT_ARENA_CAPABILITY
        if execution_mode == "eager"
        else COMPILED_DIRECT_ARENA_CAPABILITY
    )
    current_artifact = artifacts.get("current")
    capabilities = (
        current_artifact.get("required_runtime_capabilities")
        if isinstance(current_artifact, Mapping)
        else None
    )
    if (
        not isinstance(capabilities, Sequence)
        or isinstance(capabilities, (str, bytes))
        or expected_capability not in capabilities
    ):
        failures.append(
            {
                "artifact": "current",
                "counter": "required_runtime_capabilities",
                "observed": capabilities,
                "expected": expected_capability,
            }
        )
    profile_count = 0
    for measurement in measurements:
        if measurement.get("lane") != "current":
            continue
        for key, expected in (
            ("profile_attribution_sample_pass", PROFILE_ATTRIBUTION_SAMPLE_PASS),
            ("profile_attribution_boundary", ARENA_PROFILE_BOUNDARY),
            ("profile_attribution_borrowed_flat_input", True),
            ("profile_attribution_preallocated_output", True),
            ("profile_attribution_phase_timing_scope", ARENA_PHASE_TIMING_SCOPE),
            ("profile_attribution_evaluator_timing_available", False),
            ("profile_protocol", ARENA_PROFILE_PROTOCOL),
            ("paired_profile_evaluator_seconds_per_point", None),
            ("paired_profile_evaluator_uncertainty", None),
        ):
            if measurement.get(key) != expected:
                failures.append(
                    {
                        "pair_index": measurement.get("pair_index"),
                        "counter": key,
                        "observed": measurement.get(key),
                        "expected": expected,
                    }
                )
        breakdown = measurement.get("paired_profile_timing_breakdown")
        profile_timed_block_count = measurement.get("profile_timed_block_count")
        if (
            isinstance(profile_timed_block_count, bool)
            or not isinstance(profile_timed_block_count, int)
            or profile_timed_block_count < minimum_profile_timed_blocks
        ):
            failures.append(
                {
                    "pair_index": measurement.get("pair_index"),
                    "counter": "profile_timed_block_count",
                    "observed": profile_timed_block_count,
                    "expected": f"integer >= {minimum_profile_timed_blocks}",
                }
            )
        if (
            not isinstance(breakdown, Mapping)
            or "evaluator_call_time" not in breakdown
            or breakdown.get("evaluator_call_time") is not None
        ):
            failures.append(
                {
                    "pair_index": measurement.get("pair_index"),
                    "counter": ("paired_profile_timing_breakdown.evaluator_call_time"),
                    "observed": (
                        breakdown.get("evaluator_call_time")
                        if isinstance(breakdown, Mapping)
                        else breakdown
                    ),
                    "expected": None,
                }
            )
        raw_profiles = (
            breakdown.get("raw_profile_samples")
            if isinstance(breakdown, Mapping)
            else None
        )
        if not isinstance(raw_profiles, Sequence) or isinstance(
            raw_profiles,
            (str, bytes),
        ):
            raise RegressionError("current measurement has no raw arena profiles")
        if (
            isinstance(profile_timed_block_count, int)
            and not isinstance(profile_timed_block_count, bool)
            and (
                breakdown.get("sample_count") != profile_timed_block_count
                or len(raw_profiles) != profile_timed_block_count
            )
        ):
            failures.append(
                {
                    "pair_index": measurement.get("pair_index"),
                    "counter": "paired_profile_timed_block_coverage",
                    "observed": {
                        "measurement": profile_timed_block_count,
                        "breakdown": breakdown.get("sample_count"),
                        "raw_profiles": len(raw_profiles),
                    },
                    "expected": "equal positive counts",
                }
            )
        for profile_index, profile in enumerate(raw_profiles):
            if not isinstance(profile, Mapping):
                raise RegressionError("current raw arena profile is invalid")
            profile_count += 1
            if profile.get("execution_mode") != execution_mode:
                failures.append(
                    {
                        "pair_index": measurement.get("pair_index"),
                        "profile_index": profile_index,
                        "counter": "execution_mode",
                        "observed": profile.get("execution_mode"),
                        "expected": execution_mode,
                    }
                )
            for key, expected in (
                ("profile_boundary", ARENA_PROFILE_BOUNDARY),
                ("borrowed_flat_input", True),
                ("preallocated_output", True),
                ("phase_timing_scope", ARENA_PHASE_TIMING_SCOPE),
                ("evaluator_timing_available", False),
            ):
                if profile.get(key) != expected:
                    failures.append(
                        {
                            "pair_index": measurement.get("pair_index"),
                            "profile_index": profile_index,
                            "counter": key,
                            "observed": profile.get(key),
                            "expected": expected,
                        }
                    )
            required_zero = list(_ZERO_ARENA_PROFILE_COUNTERS)
            if execution_mode == "compiled":
                required_zero.extend(_ZERO_COMPILED_BOUNDARY_COUNTERS)
            for key in required_zero:
                value = profile.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                    failures.append(
                        {
                            "pair_index": measurement.get("pair_index"),
                            "profile_index": profile_index,
                            "counter": key,
                            "observed": value,
                            "expected": 0,
                        }
                    )
            for key in _ZERO_ARENA_PROFILE_TIMES:
                value = profile.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(float(value))
                    or float(value) != 0.0
                ):
                    failures.append(
                        {
                            "pair_index": measurement.get("pair_index"),
                            "profile_index": profile_index,
                            "counter": key,
                            "observed": value,
                            "expected": 0.0,
                        }
                    )
            for key in _EMPTY_ARENA_PROFILE_PHASE_VECTORS:
                value = profile.get(key)
                if value != []:
                    failures.append(
                        {
                            "pair_index": measurement.get("pair_index"),
                            "profile_index": profile_index,
                            "counter": key,
                            "observed": value,
                            "expected": [],
                        }
                    )
            wall_time = profile.get("wall_time_s")
            orchestration_time = profile.get("orchestration_time_s")
            if (
                isinstance(wall_time, bool)
                or not isinstance(wall_time, (float, int))
                or not math.isfinite(float(wall_time))
                or float(wall_time) <= 0.0
                or orchestration_time != wall_time
            ):
                failures.append(
                    {
                        "pair_index": measurement.get("pair_index"),
                        "profile_index": profile_index,
                        "counter": "coarse_arena_boundary_wall_accounting",
                        "observed": {
                            "wall_time_s": wall_time,
                            "orchestration_time_s": orchestration_time,
                        },
                        "expected": "equal finite positive values",
                    }
                )
            if execution_mode == "compiled":
                engine_count = profile.get("compiled_direct_arena_engine_count")
                call_count = profile.get("compiled_direct_arena_call_count")
                backend_call_count = profile.get("evaluator_backend_call_count")
                for key, value in (
                    ("compiled_direct_arena_engine_count", engine_count),
                    ("compiled_direct_arena_call_count", call_count),
                    ("evaluator_backend_call_count", backend_call_count),
                ):
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                    ):
                        failures.append(
                            {
                                "pair_index": measurement.get("pair_index"),
                                "profile_index": profile_index,
                                "counter": key,
                                "observed": value,
                                "expected": "positive integer",
                            }
                        )
                if call_count != backend_call_count:
                    failures.append(
                        {
                            "pair_index": measurement.get("pair_index"),
                            "profile_index": profile_index,
                            "counter": "compiled_direct_arena_call_coverage",
                            "observed": {
                                "compiled": call_count,
                                "backend": backend_call_count,
                            },
                            "expected": "equal",
                        }
                    )
    return {
        "execution_mode": execution_mode,
        "required_current_artifact_capability": expected_capability,
        "current_artifact_capabilities": capabilities,
        "profile_count": profile_count,
        "minimum_profile_timed_blocks": minimum_profile_timed_blocks,
        "zero_counters": list(_ZERO_ARENA_PROFILE_COUNTERS),
        "zero_compiled_boundary_counters": (
            list(_ZERO_COMPILED_BOUNDARY_COUNTERS)
            if execution_mode == "compiled"
            else []
        ),
        "zero_non_orchestration_phase_times": list(_ZERO_ARENA_PROFILE_TIMES),
        "failures": failures,
        "passes": profile_count > 0 and not failures,
    }


def _resource_summary(
    artifacts: Mapping[str, Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    peaks: list[float] = []
    for artifact in artifacts.values():
        generation = artifact.get("generation")
        if isinstance(generation, Mapping):
            peak = generation.get("peak_rss_gib")
            if isinstance(peak, (float, int)) and not isinstance(peak, bool):
                peaks.append(float(peak))
    for measurement in measurements:
        peak = measurement.get("peak_rss_gib")
        if isinstance(peak, (float, int)) and not isinstance(peak, bool):
            peaks.append(float(peak))
    return {
        "memory_limit_gib": MEMORY_LIMIT_GIB,
        "maximum_observed_peak_rss_gib": max(peaks, default=None),
        "generation_subprocess_count": sum(
            not bool(artifact.get("reused")) for artifact in artifacts.values()
        ),
        "sample_subprocess_count": len(measurements),
        "profile_subprocess_count": len(measurements),
        "elapsed_seconds": elapsed_seconds,
    }


def _median_measurement_metric(
    measurements: Sequence[Mapping[str, Any]],
    *,
    lane: str,
    key: str,
) -> float | None:
    values = [
        float(value)
        for measurement in measurements
        if measurement.get("lane") == lane
        for value in [measurement.get(key)]
        if isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    ]
    return float(statistics.median(values)) if values else None


def _cell_resource_gate(
    artifacts: Mapping[str, Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    *,
    gain_gate: Mapping[str, Any],
    shared_artifact: bool,
) -> dict[str, Any]:
    comparisons: dict[str, dict[str, object]] = {}

    def compare(name: str, baseline: float | None, current: float | None) -> None:
        if baseline is None or current is None or baseline <= 0.0:
            comparisons[name] = {
                "baseline": baseline,
                "current": current,
                "ratio": None,
                "material_growth": None,
                "offset_by_runtime_gain": False,
                "passes": shared_artifact,
            }
            return
        ratio = current / baseline
        material = ratio > 1.0 + MATERIAL_RESOURCE_GROWTH_THRESHOLD
        offset = not material or bool(gain_gate.get("passes"))
        comparisons[name] = {
            "baseline": baseline,
            "current": current,
            "ratio": ratio,
            "material_growth": material,
            "offset_by_runtime_gain": offset,
            "passes": offset,
        }

    compare(
        "material_payload_size_bytes",
        float(artifacts["baseline"].get("material_payload_size_bytes", 0.0)),
        float(artifacts["current"].get("material_payload_size_bytes", 0.0)),
    )
    compare(
        "cold_load_seconds",
        _median_measurement_metric(
            measurements,
            lane="baseline",
            key="cold_load_seconds",
        ),
        _median_measurement_metric(
            measurements,
            lane="current",
            key="cold_load_seconds",
        ),
    )
    compare(
        "peak_rss_gib",
        _median_measurement_metric(
            measurements,
            lane="baseline",
            key="peak_rss_gib",
        ),
        _median_measurement_metric(
            measurements,
            lane="current",
            key="peak_rss_gib",
        ),
    )
    baseline_generation = artifacts["baseline"].get("core_generation_seconds")
    current_generation = artifacts["current"].get("core_generation_seconds")
    generation_ratio = (
        float(current_generation) / float(baseline_generation)
        if isinstance(baseline_generation, (float, int))
        and not isinstance(baseline_generation, bool)
        and float(baseline_generation) > 0.0
        and isinstance(current_generation, (float, int))
        and not isinstance(current_generation, bool)
        else None
    )
    generation_passes = shared_artifact or (
        generation_ratio is not None and generation_ratio <= 1.10
    )
    return {
        "material_growth_threshold": MATERIAL_RESOURCE_GROWTH_THRESHOLD,
        "material_growth_requires_runtime_gain": True,
        "comparisons": comparisons,
        "generation": {
            "baseline_core_seconds": baseline_generation,
            "current_core_seconds": current_generation,
            "current_over_baseline": generation_ratio,
            "maximum_ratio": 1.10,
            "passes": generation_passes,
        },
        "passes": generation_passes
        and all(bool(value["passes"]) for value in comparisons.values()),
    }


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.setdefault("SYMBOLICA_HIDE_BANNER", "1")
    return environment


def _resolved_workload(arguments: argparse.Namespace, *, lc_flow_layout: str) -> str:
    requested = getattr(arguments, "workload", "auto")
    if arguments.color != "lc":
        derived = "summed"
        if arguments.helicity or arguments.color_flow:
            raise RegressionError("NLC/full summed workloads cannot use LC selectors")
    elif lc_flow_layout == "topology-replay":
        if not arguments.color_flow and not arguments.helicity and requested == "auto":
            return "summed"
        if not arguments.color_flow or arguments.helicity:
            raise RegressionError(
                "topology-replay acceptance requires one runtime color-flow selector "
                "and no helicity selector"
            )
        derived = "single-flow-helicity-sum"
    else:
        if not arguments.helicity and not arguments.color_flow and requested == "auto":
            return "summed"
        if not arguments.helicity or arguments.color_flow:
            raise RegressionError(
                "all-flow-union acceptance requires one runtime helicity selector "
                "and no color-flow selector"
            )
        derived = "all-flow-single-helicity"
    if requested not in {"auto", derived}:
        raise RegressionError(
            f"declared workload {requested!r} does not match "
            f"selectors/layout {derived!r}"
        )
    return derived


def run_regression(arguments: argparse.Namespace) -> dict[str, Any]:
    baseline_python = _require_interpreter(
        arguments.baseline_python,
        lane="baseline",
    )
    current_python = _require_interpreter(arguments.current_python, lane="current")
    output_root = _absolute_path(arguments.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model = _model_argument(arguments.model)
    execution_mode = getattr(arguments, "execution_mode", "compiled")
    if execution_mode not in EXECUTION_MODES:
        raise RegressionError(f"unsupported execution mode: {execution_mode!r}")
    jit_optimization_level = getattr(arguments, "jit_optimization_level", 3)
    dependency_sites = {
        "baseline": getattr(arguments, "baseline_dependency_site", None),
        "current": getattr(arguments, "current_dependency_site", None),
    }
    dependency_sites = {
        lane: (None if path is None else _absolute_path(path).resolve(strict=True))
        for lane, path in dependency_sites.items()
    }
    dependency_identity_cache: dict[str, dict[str, object]] = {}
    dependency_site_identities: dict[str, dict[str, object] | None] = {}
    for lane, path in dependency_sites.items():
        if path is None:
            dependency_site_identities[lane] = None
            continue
        key = str(path)
        identity = dependency_identity_cache.get(key)
        if identity is None:
            identity = _dependency_site_identity(path)
            dependency_identity_cache[key] = identity
        dependency_site_identities[lane] = identity
    lc_flow_layout = getattr(
        arguments,
        "lc_flow_layout",
        DEFAULT_LC_FLOW_LAYOUT,
    )
    expected_artifact_lc_flow_layout = (
        lc_flow_layout if arguments.color == "lc" else None
    )
    workload = _resolved_workload(arguments, lc_flow_layout=lc_flow_layout)
    shared_artifact_argument = getattr(arguments, "shared_artifact", None)
    if shared_artifact_argument is not None and arguments.regenerate_artifacts:
        raise RegressionError(
            "--shared-artifact and --regenerate-artifacts are mutually exclusive"
        )
    shared_artifact = (
        None
        if shared_artifact_argument is None
        else _absolute_path(shared_artifact_argument)
    )
    environment = _environment()
    started = time.monotonic()

    interpreters = {
        "baseline": baseline_python,
        "current": current_python,
    }
    provenance = {
        "driver": _path_identity(Path(__file__)),
        "watchdog": _path_identity(WATCHDOG),
        "native_sample_helper": _path_identity(NATIVE_SAMPLE_HELPER),
        "dependency_entry": _path_identity(DEPENDENCY_ENTRY),
        "interpreters": {
            lane: _path_identity(python) for lane, python in interpreters.items()
        },
        "model": _model_identity(model),
        "dependency_sites": dependency_site_identities,
    }
    if shared_artifact is None:
        artifacts = {
            lane: _ensure_artifact(
                lane,
                python,
                output_root=output_root,
                process=arguments.process,
                model=model,
                color=arguments.color,
                execution_mode=execution_mode,
                jit_optimization_level=jit_optimization_level,
                generation_timeout=arguments.generation_timeout,
                regenerate=arguments.regenerate_artifacts,
                environment=environment,
                lc_flow_layout=lc_flow_layout,
                dependency_site=dependency_sites[lane],
                dependency_site_identity=dependency_site_identities[lane],
            )
            for lane, python in interpreters.items()
        }
        for artifact in artifacts.values():
            artifact["comparison_role"] = "independently-generated-lane-artifact"
        shared_artifact_identity: dict[str, Any] | None = None
    else:
        shared_artifact_identity = _artifact_metadata(
            shared_artifact,
            expected_process=arguments.process,
            expected_color=arguments.color,
            expected_execution_mode=execution_mode,
            expected_lc_flow_layout=expected_artifact_lc_flow_layout,
        )
        artifacts = {
            lane: {
                **shared_artifact_identity,
                "shared": True,
                "reused": True,
                "generation": None,
                "generation_command": None,
                "cache_path": None,
                "comparison_role": "shared-read-only-artifact",
            }
            for lane in interpreters
        }
    partial: dict[str, Any] = {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "complete": False,
        "provenance": provenance,
        "artifacts": artifacts,
        "measurements": [],
    }
    result_path_argument = getattr(arguments, "result_path", None)
    result_path = (
        output_root / "result.json"
        if result_path_argument is None
        else _absolute_path(result_path_argument)
    )
    _write_json_atomic(result_path, partial)

    measurements: list[dict[str, Any]] = []
    values: dict[str, list[float]] = {"baseline": [], "current": []}
    native_module_sha256_by_lane: dict[str, str] = {}
    pair_orders: list[list[str]] = []
    for pair_index in range(arguments.samples):
        order = (
            ("baseline", "current") if pair_index % 2 == 0 else ("current", "baseline")
        )
        pair_orders.append(list(order))
        for order_index, lane in enumerate(order):
            artifact = artifacts[lane]
            request_precision32 = pair_index == 0 and lane == "current"
            profile_command = _profile_command(
                interpreters[lane],
                artifact=Path(str(artifact["path"])),
                process_id=str(artifact["process_id"]),
                batch_size=arguments.batch_size,
                target_runtime=arguments.target_runtime,
                minimum_samples=arguments.minimum_samples,
                warmup_runs=arguments.warmup_runs,
                helicities=arguments.helicity,
                color_flows=arguments.color_flow,
                execution_mode=execution_mode,
                profile_protocol=(
                    ARENA_PROFILE_PROTOCOL
                    if lane == "current"
                    else FROZEN_BASELINE_PROFILE_PROTOCOL
                ),
                dependency_site=dependency_sites[lane],
                include_precision32=request_precision32,
            )
            payload, command_elapsed, stderr = _run_json(
                profile_command,
                timeout=arguments.profile_timeout,
                environment=environment,
            )
            sample = _native_profile_sample(
                payload,
                minimum_samples=arguments.minimum_samples,
                batch_size=arguments.batch_size,
                expected_execution_mode=execution_mode,
                require_precision32=request_precision32,
                require_arena_profile=lane == "current",
            )
            observed_precision32 = isinstance(
                sample.get("precision32_numerical_result"),
                Mapping,
            )
            if observed_precision32 is not request_precision32:
                raise RegressionError(
                    "native sample precision-32 evidence does not match the "
                    f"{PRECISION32_CORRECTNESS_POLICY} request policy for "
                    f"{lane} pair {pair_index + 1}"
                )
            native_module_sha256 = str(sample["native_module_sha256"])
            expected_native_module_sha256 = native_module_sha256_by_lane.setdefault(
                lane,
                native_module_sha256,
            )
            if native_module_sha256 != expected_native_module_sha256:
                raise RegressionError(
                    f"{lane} native module changed during sampling: "
                    f"{native_module_sha256} != {expected_native_module_sha256}"
                )
            if measurements:
                reference = measurements[0]
                for key in (
                    "batch_sha256",
                    "helicities",
                    "color_flows",
                    "correctness_point_derivation",
                ):
                    if sample[key] != reference[key]:
                        raise RegressionError(
                            "native samples do not share byte-identical inputs: "
                            f"{key} differs in {lane} pair {pair_index + 1}"
                        )
            wall = float(sample["wall_seconds_per_point"])
            values[lane].append(wall)
            measurements.append(
                {
                    "pair_index": pair_index + 1,
                    "measurement_order": order_index + 1,
                    "lane": lane,
                    "command": _command_identity(profile_command),
                    "guarded_command": _command_identity(
                        _guarded_command(profile_command)
                    ),
                    **sample,
                    "command_elapsed_seconds": command_elapsed,
                    "peak_rss_gib": _watchdog_peak_gib(stderr),
                    "watchdog": _watchdog_marker(stderr),
                }
            )
        partial["measurements"] = measurements
        partial["pair_orders"] = pair_orders
        _write_json_atomic(result_path, partial)

    if shared_artifact_identity is not None:
        assert shared_artifact is not None
        observed_identity = _artifact_metadata(
            shared_artifact,
            expected_process=arguments.process,
            expected_color=arguments.color,
            expected_execution_mode=execution_mode,
            expected_lc_flow_layout=expected_artifact_lc_flow_layout,
        )
        for key in (
            "artifact_id",
            "manifest_sha256",
            "tree_identity",
            "payload_digests",
        ):
            if observed_identity[key] != shared_artifact_identity[key]:
                raise RegressionError(
                    f"shared artifact changed during sampling: {key} differs"
                )
    else:
        for lane, initial_identity in artifacts.items():
            observed_identity = _artifact_metadata(
                Path(str(initial_identity["path"])),
                expected_process=arguments.process,
                expected_color=arguments.color,
                expected_execution_mode=execution_mode,
                expected_lc_flow_layout=expected_artifact_lc_flow_layout,
            )
            for key in (
                "artifact_id",
                "manifest_sha256",
                "tree_identity",
                "payload_digests",
            ):
                if observed_identity[key] != initial_identity[key]:
                    raise RegressionError(
                        f"{lane} artifact changed during sampling: {key} differs"
                    )
    for lane, path in dependency_sites.items():
        if (
            path is not None
            and _dependency_site_identity(path) != (dependency_site_identities[lane])
        ):
            raise RegressionError(
                f"{lane} dependency site changed during generation/sampling"
            )
    if _model_identity(model) != provenance["model"]:
        raise RegressionError("model input changed during generation/sampling")
    distributions = {
        lane: _distribution(lane_values) for lane, lane_values in values.items()
    }
    authority = _performance_authority(
        artifacts,
        shared_artifact=shared_artifact_identity is not None,
    )
    measured_gate = _regression_gate(
        distributions["baseline"],
        distributions["current"],
    )
    measured_thresholds_pass = bool(measured_gate.pop("passes"))
    paired_distribution = _paired_distribution(measurements)
    gain_gate = _gain_gate(
        distributions["baseline"],
        distributions["current"],
        paired_distribution,
    )
    gate = {
        **measured_gate,
        "measured_thresholds_pass": measured_thresholds_pass,
        "authoritative": bool(authority["authoritative"]),
        "authority_basis": authority["basis"],
        "passes": (bool(authority["authoritative"]) and measured_thresholds_pass),
    }
    correctness_gate = _correctness_gate(measurements)
    arena_profile_gate = _arena_profile_gate(
        measurements,
        execution_mode=execution_mode,
        artifacts=artifacts,
        minimum_profile_timed_blocks=arguments.minimum_samples,
    )
    resource_gate = _cell_resource_gate(
        artifacts,
        measurements,
        gain_gate=gain_gate,
        shared_artifact=shared_artifact_identity is not None,
    )
    elapsed = time.monotonic() - started
    result = {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "performance_result_authoritative": bool(authority["authoritative"]),
        "passes": bool(
            gate["passes"]
            and correctness_gate["passes"]
            and arena_profile_gate["passes"]
            and resource_gate["passes"]
        ),
        "platform": platform.platform(),
        "provenance": provenance,
        "configuration": {
            "baseline_python": str(baseline_python),
            "current_python": str(current_python),
            "output_root": str(output_root),
            "process": arguments.process,
            "model": model,
            "model_label": getattr(arguments, "model_label", "custom"),
            "execution_mode": execution_mode,
            "workload": workload,
            "jit_optimization_level": jit_optimization_level,
            "color_accuracy": arguments.color,
            "lc_flow_layout": lc_flow_layout,
            "shared_artifact": (
                None if shared_artifact is None else str(shared_artifact)
            ),
            "batch_size": arguments.batch_size,
            "independent_samples_per_lane": arguments.samples,
            "target_runtime_per_profile_seconds": arguments.target_runtime,
            "target_runtime_per_native_sample_seconds": arguments.target_runtime,
            "minimum_native_timed_blocks_per_profile": arguments.minimum_samples,
            "warmup_runs_per_profile": arguments.warmup_runs,
            "generation_timeout_seconds": arguments.generation_timeout,
            "profile_timeout_seconds": arguments.profile_timeout,
            "native_sample_timeout_seconds": arguments.profile_timeout,
            "helicities": list(arguments.helicity),
            "color_flows": list(arguments.color_flow),
            "native_wall_time_source": NATIVE_WALL_TIME_SOURCE,
            "native_wall_time_sample_pass": NATIVE_WALL_TIME_SAMPLE_PASS,
            "required_current_profile_attribution_sample_pass": (
                PROFILE_ATTRIBUTION_SAMPLE_PASS
            ),
            "required_current_profile_attribution_boundary": (ARENA_PROFILE_BOUNDARY),
            "required_current_profile_attribution_phase_timing_scope": (
                ARENA_PHASE_TIMING_SCOPE
            ),
            "required_current_profile_attribution_evaluator_timing_available": False,
            "required_current_profile_protocol": ARENA_PROFILE_PROTOCOL,
            "timing_sample_contract": PAIRED_TIMING_SAMPLE_CONTRACT,
            "precision32_correctness_policy": PRECISION32_CORRECTNESS_POLICY,
            "dependency_sites": {
                lane: dependency_site_identities[lane] for lane in dependency_sites
            },
        },
        "artifacts": artifacts,
        "pair_orders": pair_orders,
        "measurements": measurements,
        "native_module_sha256_by_lane": native_module_sha256_by_lane,
        "distributions": distributions,
        "paired_distribution": paired_distribution,
        "performance_authority": authority,
        "gate": gate,
        "gain_gate": gain_gate,
        "correctness_gate": correctness_gate,
        "arena_profile_gate": arena_profile_gate,
        "resource_gate": resource_gate,
        "resources": _resource_summary(
            artifacts,
            measurements,
            elapsed_seconds=elapsed,
        ),
    }
    _write_json_atomic(result_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The gate passes only when the current median is no more than 3% "
            "above the baseline median and no more than three baseline MAD above it."
        ),
    )
    result.add_argument("--baseline-python", type=Path, required=True)
    result.add_argument("--current-python", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument(
        "--result-path",
        type=Path,
        help=(
            "write the result outside the artifact-cache root; this permits "
            "multiple timing cells to reuse one exact generated artifact"
        ),
    )
    result.add_argument(
        "--shared-artifact",
        type=Path,
        help=(
            "sample one existing read-only artifact in both lanes instead of "
            "generating per-lane artifacts"
        ),
    )
    result.add_argument("--process", required=True)
    result.add_argument("--model", default="built-in-sm")
    result.add_argument(
        "--model-label",
        choices=("built-in", "ufo-sm", "custom"),
        default="custom",
    )
    result.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODES,
        default="compiled",
    )
    result.add_argument(
        "--jit-optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=3,
    )
    result.add_argument(
        "--workload",
        choices=(
            "auto",
            "single-flow-helicity-sum",
            "all-flow-single-helicity",
            "summed",
        ),
        default="auto",
    )
    result.add_argument("--baseline-dependency-site", type=Path)
    result.add_argument("--current-dependency-site", type=Path)
    result.add_argument(
        "--color",
        "--color-accuracy",
        dest="color",
        choices=("lc", "nlc", "full"),
        default="lc",
    )
    result.add_argument(
        "--lc-flow-layout",
        choices=("topology-replay", "all-flow-union"),
        default=DEFAULT_LC_FLOW_LAYOUT,
        help="LC artifact layout generated identically in both lanes",
    )
    result.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    result.add_argument(
        "--samples",
        type=_at_least_five,
        default=DEFAULT_SAMPLE_COUNT,
        help=(
            "independent native sampling subprocesses per interpreter "
            "(acceptance default: 7; diagnostic minimum: 5)"
        ),
    )
    result.add_argument(
        "--target-runtime",
        type=_positive_float,
        default=DEFAULT_TARGET_RUNTIME,
        help="native headline timing target for each sampling subprocess",
    )
    result.add_argument(
        "--minimum-samples",
        type=_at_least_five,
        default=DEFAULT_SAMPLE_COUNT,
        help="minimum native timed blocks inside each sampling subprocess",
    )
    result.add_argument(
        "--warmup-runs",
        type=_positive_int,
        default=DEFAULT_WARMUP_RUNS,
    )
    result.add_argument("--helicity", action="append", default=[])
    result.add_argument("--color-flow", action="append", default=[])
    result.add_argument(
        "--generation-timeout",
        type=_positive_float,
        default=DEFAULT_GENERATION_TIMEOUT,
    )
    result.add_argument(
        "--profile-timeout",
        type=_positive_float,
        default=DEFAULT_PROFILE_TIMEOUT,
    )
    result.add_argument(
        "--regenerate-artifacts",
        action="store_true",
        help="replace both cached compiled artifacts before sampling",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        result = run_regression(arguments)
    except (RegressionError, OSError, ValueError) as error:
        print(f"compiled-mode-regression: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
