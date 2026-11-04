#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Fail-closed Milestone-0 evidence combiner for the arena migration.

This program does not run benchmarks and cannot turn a diagnostic capture into
authoritative evidence.  It revalidates four schema-7 pyAmpliCol captures and
two independently captured AmpliCol raw-evidence manifests, then emits one
content-addressed accepted or rejected record.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NoReturn

REQUEST_KIND = "pyamplicol-eager-compiled-arena-m0-request"
REQUEST_SCHEMA = 1
AMPLICOL_KIND = "pyamplicol-amplicol-m0-raw-evidence"
AMPLICOL_SCHEMA = 1
OUTPUT_KIND = "pyamplicol-eager-compiled-arena-m0-acceptance"
OUTPUT_SCHEMA = 1

CAPTURE_KIND = "pyamplicol-recurrence-z6g-benchmark"
CAPTURE_SCHEMA = 7
CAPTURE_ACCEPTANCE_KIND = "pyamplicol-three-lane-layout-capture"
CAPTURE_ACCEPTANCE_SCHEMA = 6
PER_LAYOUT_M0_KIND = "pyamplicol-milestone-0-evidence-manifest"
PER_LAYOUT_M0_SCHEMA = 4

MODES = ("compiled", "eager", "recurrence")
BATCHES = (1, 128, 1024)
COMPILED_RECURRENCE_BATCHES = (128, 1024)
MAX_COMPILED_OVER_RECURRENCE = 1.15
COMPILED_RECURRENCE_STATISTIC = "ratio-of-recomputed-subprocess-medians-v1"
LAYOUTS = ("topology-replay", "all-flow-union")
MODELS = ("built-in-sm", "ufo-sm")
MIN_SAMPLES = 7
RTOL = 1.0e-12
ATOL = 1.0e-15
PROCESS = "u u~ > z g g g g g g"
EXTERNAL_LEG_COUNT = 9
LC_COLOR_WORD_LENGTH = 8
BUILTIN_PACKAGED_MODEL_RESOURCE_ID = "built-in-sm-jit-o2"
PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL = 2

SELECTED_WORKLOAD = "single-runtime-selected-flow/helicity-sum"
UNION_WORKLOAD = "all-flows/runtime-selected-single-helicity"
AMPLICOL_SELECTED_ROLE = "selected-flow-helicity-sum"
AMPLICOL_UNION_ROLE = "all-flow-single-helicity"

_SHA256_LENGTH = 64
_CAPTURE_ROOT_KEYS = {
    "kind",
    "schema_version",
    "complete",
    "passes",
    "capture_acceptance",
    "milestone0_acceptance",
    "recurrence_runtime_acceptance",
    "source",
    "runtime_provenance",
    "provenance",
    "process",
    "process_name",
    "workload",
    "configuration",
    "generation",
    "profile_schedule",
    "validation_preflight",
    "paired_profile_coordination",
    "profiles",
    "validation_summary",
    "selector_contracts_match",
    "validation_fixtures_match",
    "lane_comparisons",
    "result_json",
}
_REQUEST_KEYS = {
    "kind",
    "schema_version",
    "captures",
    "amplicol_evidence",
    "expected",
}
_EXPECTED_KEYS = {
    "pyamplicol_source_revision",
    "amplicol_source_revision",
    "process",
    "runtime_provenance_sha256",
    "host_sha256",
    "momenta_points_sha256",
    "normalization_sha256",
    "model_common_physics_identity_sha256",
    "generation_model_identities_sha256",
    "color_flow",
    "helicity",
    "external_leg_permutation",
}
_FILE_REF_KEYS = {"path", "size_bytes", "sha256"}
_AMPLICOL_ROOT_KEYS = {
    "kind",
    "schema_version",
    "complete",
    "evidence_scope",
    "workload",
    "source",
    "host",
    "process",
    "physical_axes",
    "selector",
    "momenta",
    "normalization_sha256",
    "timing",
    "validation",
    "binary_evidence",
    "content_sha256",
}
_HOST_KEYS = {
    "platform",
    "system",
    "release",
    "version",
    "machine",
    "processor",
    "cpu_model",
    "logical_cpu_count",
}
_RUNTIME_KEYS = {
    "interpreter",
    "installed_distribution",
    "active_build_info",
    "native_extension",
    "dependencies",
}
_FILE_IDENTITY_KEYS = {"path", "resolved_path", "size_bytes", "sha256"}
_RUNTIME_DEPENDENCIES = {
    "Cargo.lock",
    "Cargo.toml",
    "pyproject.toml",
    "rust-toolchain.toml",
    "dependencies/candidate-Cargo.lock",
    "dependencies/candidate-cargo-config.toml",
    "dependencies/contributor-lock.toml",
    "dependencies/install-state.json",
    "dependencies/python-runtime-lock.toml",
    "dependencies/release-lock.toml",
}


class EvidenceError(RuntimeError):
    """Evidence is absent, malformed, inconsistent, or stale."""


@dataclass(frozen=True)
class FileRef:
    """An expected raw file address from the request manifest."""

    path: Path
    stated_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class LoadedJson:
    """Strictly parsed JSON together with both raw and canonical addresses."""

    ref: FileRef
    payload: dict[str, Any]
    canonical_sha256: str


@dataclass(frozen=True)
class Capture:
    """Validated schema-7 capture and extracted cross-capture contracts."""

    model: str
    layout: str
    loaded: LoadedJson
    source_identity: dict[str, Any]
    runtime_identity: dict[str, Any]
    host: dict[str, Any]
    fixture: dict[str, Any]
    normalization_sha256: str
    model_common_sha256: str
    generation_models_sha256: str
    color_axis: dict[str, Any]
    helicity_axis: dict[str, Any]
    runtime_selector_semantics_sha256: str
    reduction_ordering_sha256: str
    execution_schedule_ordering_sha256_by_mode: dict[str, str]
    selector: dict[str, Any]
    timings: dict[str, Any]
    validation_values: tuple[complex, ...]


@dataclass(frozen=True)
class AmpliColEvidence:
    """Validated raw AmpliCol evidence for one selector workload."""

    role: str
    loaded: LoadedJson
    source_identity: dict[str, Any]
    host: dict[str, Any]
    momenta_file_sha256: str
    color_axis: dict[str, Any]
    helicity_axis: dict[str, Any]
    selector: dict[str, Any]
    values: tuple[complex, ...]
    timing: dict[str, Any]
    interleave_group_sha256: str
    interleave_records: tuple[dict[str, Any], ...]


def _die(message: str) -> NoReturn:
    raise EvidenceError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if not _is_int(value):
        _die(f"{label} must be an integer")
    assert isinstance(value, int)
    if minimum is not None and value < minimum:
        _die(f"{label} must be at least {minimum}")
    return value


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _die(f"{label} must be a JSON object")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        _die(f"{label} must be a JSON array")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        _die(f"{label} keys differ: missing={missing}, unknown={unknown}")


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _die(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise EvidenceError("evidence is not canonical finite JSON") from error
    return text.encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise EvidenceError(f"{label} is not strict JSON: {error}") from error
    return _require_mapping(value, label)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_ref(value: object, *, base: Path, label: str) -> FileRef:
    raw = _require_mapping(value, label)
    _require_exact_keys(raw, _FILE_REF_KEYS, label)
    stated_path = raw["path"]
    size_bytes = raw["size_bytes"]
    if not isinstance(stated_path, str) or not stated_path:
        _die(f"{label}.path must be a non-empty string")
    if not _is_int(size_bytes) or size_bytes < 0:
        _die(f"{label}.size_bytes must be a non-negative integer")
    digest = _require_sha256(raw["sha256"], f"{label}.sha256")
    candidate = Path(stated_path)
    path = candidate if candidate.is_absolute() else base / candidate
    return FileRef(
        path=path.resolve(strict=False),
        stated_path=stated_path,
        size_bytes=size_bytes,
        sha256=digest,
    )


def _verify_file(ref: FileRef, label: str) -> bytes:
    try:
        if not ref.path.is_file():
            _die(f"{label} does not exist as a regular file: {ref.path}")
        observed_size = ref.path.stat().st_size
        if observed_size != ref.size_bytes:
            _die(
                f"{label} size drifted: expected {ref.size_bytes}, "
                f"observed {observed_size}"
            )
        raw = ref.path.read_bytes()
    except OSError as error:
        raise EvidenceError(f"cannot read {label}: {error}") from error
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != ref.sha256:
        _die(
            f"{label} content hash drifted: expected {ref.sha256}, "
            f"observed {observed_sha256}"
        )
    return raw


def _load_json_ref(ref: FileRef, label: str) -> LoadedJson:
    raw = _verify_file(ref, label)
    payload = _strict_json_bytes(raw, label)
    return LoadedJson(
        ref=ref,
        payload=payload,
        canonical_sha256=_canonical_sha256(payload),
    )


def _load_benchmark_module() -> ModuleType:
    path = Path(__file__).with_name("recurrence_z6g_benchmark.py")
    spec = importlib.util.spec_from_file_location(
        "_arena_m0_recurrence_z6g_benchmark",
        path,
    )
    if spec is None or spec.loader is None:
        _die(f"cannot load schema-7 validator from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise EvidenceError(f"cannot load schema-7 validator: {error}") from error
    return module


def _normalized_process(value: object) -> str:
    if not isinstance(value, str):
        _die("process expression must be a string")
    return " ".join(value.split()).casefold()


def _utc(value: object, label: str) -> dt.datetime:
    if not isinstance(value, str):
        _die(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError(f"{label} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        _die(f"{label} must carry UTC timezone information")
    return parsed


def _complex_values(value: object, label: str) -> tuple[complex, ...]:
    rows = _require_list(value, label)
    result: list[complex] = []
    for index, row in enumerate(rows):
        pair = _require_list(row, f"{label}[{index}]")
        if len(pair) != 2 or not all(_is_number(component) for component in pair):
            _die(f"{label}[{index}] must be a finite [real, imag] pair")
        result.append(complex(float(pair[0]), float(pair[1])))
    if not result:
        _die(f"{label} must not be empty")
    return tuple(result)


def _close(left: complex, right: complex) -> bool:
    return abs(left - right) <= ATOL + RTOL * abs(right)


def _assert_values_close(
    left: Sequence[complex],
    right: Sequence[complex],
    label: str,
) -> None:
    if len(left) != len(right):
        _die(f"{label} point counts differ: {len(left)} != {len(right)}")
    for index, (lhs, rhs) in enumerate(zip(left, right, strict=True)):
        if not _close(lhs, rhs):
            _die(f"{label} differs at point {index}: {lhs!r} versus {rhs!r}")


def _amplicol_interleave_group_sha256(
    records: Sequence[Mapping[str, Any]],
) -> str:
    return _canonical_sha256(
        {
            "kind": "pyamplicol-m0-paired-interleave",
            "schema_version": 1,
            "records": [dict(record) for record in records],
        }
    )


def _amplicol_source_tree_sha256(
    identities: Sequence[Mapping[str, Any]],
) -> str:
    members = sorted(
        (
            {
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
            for identity in identities
        ),
        key=lambda identity: (identity["sha256"], identity["size_bytes"]),
    )
    return _canonical_sha256(
        {
            "kind": "amplicol-source-content-set",
            "schema_version": 1,
            "members": members,
        }
    )


def _path_stripped(value: object) -> object:
    """Remove location-only fields while retaining all content identities."""

    if isinstance(value, dict):
        return {
            key: _path_stripped(item)
            for key, item in value.items()
            if key
            not in {
                "path",
                "resolved_path",
                "checkout",
                "working_directory",
            }
        }
    if isinstance(value, list):
        return [_path_stripped(item) for item in value]
    return value


def _validate_file_identity_shape(value: object, label: str) -> dict[str, Any]:
    identity = _require_mapping(value, label)
    if not _FILE_IDENTITY_KEYS.issubset(identity):
        _die(f"{label} lacks path/size/content identity")
    for key in ("path", "resolved_path"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            _die(f"{label}.{key} must be a non-empty path")
    _require_int(identity.get("size_bytes"), f"{label}.size_bytes", minimum=0)
    _require_sha256(identity.get("sha256"), f"{label}.sha256")
    return identity


def _validate_host(value: object, label: str) -> dict[str, Any]:
    host = _require_mapping(value, label)
    _require_exact_keys(host, _HOST_KEYS, label)
    for key in ("platform", "system", "release", "version", "machine", "processor"):
        if not isinstance(host.get(key), str):
            _die(f"{label}.{key} must be a string")
    cpu_model = host.get("cpu_model")
    if cpu_model is not None and (not isinstance(cpu_model, str) or not cpu_model):
        _die(f"{label}.cpu_model must be null or a non-empty string")
    _require_int(
        host.get("logical_cpu_count"),
        f"{label}.logical_cpu_count",
        minimum=1,
    )
    return host


def _runtime_semantics(
    runtime: object,
    *,
    source_revision: str,
) -> dict[str, Any]:
    raw = _require_mapping(runtime, "capture.runtime_provenance")
    _require_exact_keys(raw, _RUNTIME_KEYS, "capture.runtime_provenance")
    interpreter = _validate_file_identity_shape(
        raw.get("interpreter"),
        "capture.runtime_provenance.interpreter",
    )
    for key in ("python_version", "implementation"):
        if not isinstance(interpreter.get(key), str) or not interpreter[key]:
            _die(f"capture.runtime_provenance.interpreter.{key} is missing")
    native = _validate_file_identity_shape(
        raw.get("native_extension"),
        "capture.runtime_provenance.native_extension",
    )
    if (
        not isinstance(native.get("package_version"), str)
        or not native["package_version"]
    ):
        _die("capture runtime native extension has no package version")
    native_build_inputs = _require_sha256(
        native.get("build_inputs_sha256"),
        "capture.runtime_provenance.native_extension.build_inputs_sha256",
    )

    distribution = _require_mapping(
        raw.get("installed_distribution"),
        "capture.runtime_provenance.installed_distribution",
    )
    _require_exact_keys(
        distribution,
        {
            "package_version",
            "distribution_content",
            "native_modules",
            "build_info_files",
        },
        "capture.runtime_provenance.installed_distribution",
    )
    distribution_version = distribution.get("package_version")
    native_version = native["package_version"]
    if (
        not isinstance(distribution_version, str)
        or not distribution_version
        or not isinstance(native_version, str)
        or native_version.replace("-dev.", ".dev") != distribution_version
    ):
        _die("capture runtime package versions are incomplete or inconsistent")
    distribution_content = _require_mapping(
        distribution.get("distribution_content"),
        "capture runtime distribution content",
    )
    _require_exact_keys(
        distribution_content,
        {"algorithm", "sha256", "file_count", "size_bytes"},
        "capture runtime distribution content",
    )
    if distribution_content.get("algorithm") != "sha256-relative-path-size-content-v1":
        _die("capture runtime distribution content algorithm is unsupported")
    _require_sha256(
        distribution_content.get("sha256"),
        "capture runtime distribution content SHA-256",
    )
    _require_int(
        distribution_content.get("file_count"),
        "capture runtime distribution file count",
        minimum=1,
    )
    _require_int(
        distribution_content.get("size_bytes"),
        "capture runtime distribution size",
        minimum=1,
    )
    for collection in ("native_modules", "build_info_files"):
        identities = _require_list(
            distribution.get(collection),
            f"capture runtime distribution {collection}",
        )
        for index, identity in enumerate(identities):
            _validate_file_identity_shape(
                identity,
                f"capture runtime distribution {collection}[{index}]",
            )

    build_info = _validate_file_identity_shape(
        raw.get("active_build_info"),
        "capture.runtime_provenance.active_build_info",
    )
    build_payload = _require_mapping(
        build_info.get("payload"),
        "capture.runtime_provenance.active_build_info.payload",
    )
    if (
        build_payload.get("source_revision") != source_revision
        or build_payload.get("native_build_inputs_sha256") != native_build_inputs
    ):
        _die("capture active build info is not bound to source/native inputs")

    dependencies = _require_mapping(
        raw.get("dependencies"),
        "capture.runtime_provenance.dependencies",
    )
    if set(dependencies) != _RUNTIME_DEPENDENCIES:
        _die("capture runtime dependency inventory is incomplete or has unknown keys")
    for name, identity_value in dependencies.items():
        identity = _require_mapping(
            identity_value,
            f"capture.runtime_provenance.dependencies.{name}",
        )
        present = identity.get("present")
        if present is True:
            _validate_file_identity_shape(
                identity,
                f"capture.runtime_provenance.dependencies.{name}",
            )
        elif present is False:
            if not isinstance(identity.get("path"), str) or not isinstance(
                identity.get("resolved_path"), str
            ):
                _die(f"capture absent dependency {name} has no path identity")
        else:
            _die(f"capture dependency {name} has no exact presence flag")
    stripped = _path_stripped(raw)
    return _require_mapping(stripped, "path-stripped runtime provenance")


def _source_semantics(source: object) -> dict[str, Any]:
    raw = _require_mapping(source, "capture.source")
    revision = raw.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        _die("capture source revision is not a full Git SHA")
    if raw.get("dirty") is not False or raw.get("untracked_files_checked") is not True:
        _die("capture source was not a clean, untracked-files-checked checkout")
    return {
        "revision": revision,
        "dirty": False,
        "untracked_files_checked": True,
    }


def _revalidation_arguments(configuration: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        modes=configuration.get("modes"),
        batch_size=configuration.get("batch_sizes"),
        subprocess_samples=configuration.get("subprocess_samples"),
        minimum_samples=configuration.get("minimum_samples"),
        warmup_runs=configuration.get("warmup_runs"),
        target_runtime=configuration.get("target_runtime_seconds"),
        color_flow=configuration.get("color_flow_request"),
        helicity=configuration.get("helicity_request"),
        lc_flow_layout=configuration.get("lc_flow_layout"),
        jit_optimization_level=configuration.get("jit_optimization_level"),
        prepared_model_path=configuration.get("prepared_model_path"),
        process_expression=None,
        gluon_count=configuration.get("gluon_count"),
        specialize_flow_at_generation=configuration.get(
            "specialize_flow_at_generation"
        ),
        generation_only=configuration.get("generation_only"),
    )


def _revalidate_capture(
    loaded: LoadedJson,
    *,
    model: str,
    layout: str,
    benchmark: ModuleType,
) -> None:
    payload = loaded.payload
    _require_exact_keys(payload, _CAPTURE_ROOT_KEYS, f"{model}/{layout} capture")
    if (
        payload.get("kind") != CAPTURE_KIND
        or payload.get("schema_version") != CAPTURE_SCHEMA
    ):
        _die(f"{model}/{layout} is not an exact schema-7 capture")
    if payload.get("complete") is not True or payload.get("passes") is not True:
        _die(f"{model}/{layout} root is incomplete or non-passing")
    configuration = _require_mapping(
        payload.get("configuration"),
        f"{model}/{layout}.configuration",
    )
    arguments = _revalidation_arguments(configuration)
    profiles = _require_mapping(payload.get("profiles"), f"{model}/{layout}.profiles")
    schedule = _require_mapping(
        payload.get("profile_schedule"),
        f"{model}/{layout}.profile_schedule",
    )
    try:
        recomputed_summary = benchmark._pairwise_profile_validation(profiles)
        recomputed_capture = benchmark._capture_acceptance(
            arguments,
            profiles,
            recomputed_summary,
            profile_schedule=schedule,
        )
        recomputed_m0 = benchmark._milestone0_acceptance_manifest(
            arguments,
            recomputed_capture,
        )
        recomputed_preflight = benchmark._validation_preflight_contract(
            arguments,
            {
                mode: profiles[mode]
                for mode in benchmark.AUTHORITATIVE_EXECUTION_MODES
                if mode in profiles
            },
        )
    except Exception as error:
        raise EvidenceError(
            f"{model}/{layout} raw schema-7 evidence failed revalidation: {error}"
        ) from error
    preflight = _require_mapping(
        payload.get("validation_preflight"),
        f"{model}/{layout}.validation_preflight",
    )
    stored_preflight_contract = {
        key: value
        for key, value in preflight.items()
        if key != "worker_evidence"
    }
    if stored_preflight_contract != recomputed_preflight:
        _die(f"{model}/{layout} stored validation preflight is stale or forged")
    if payload.get("paired_profile_coordination") is not None:
        _die(
            f"{model}/{layout} portability capture unexpectedly uses paired "
            "profile coordination"
        )
    if payload.get("validation_summary") != recomputed_summary:
        _die(f"{model}/{layout} stored validation summary is stale or forged")
    if payload.get("capture_acceptance") != recomputed_capture:
        _die(f"{model}/{layout} stored capture acceptance is stale or forged")
    if payload.get("milestone0_acceptance") != recomputed_m0:
        _die(f"{model}/{layout} per-layout M0 record is stale or forged")
    if payload.get("selector_contracts_match") is not True:
        _die(f"{model}/{layout} selectors do not match across execution lanes")
    if payload.get("validation_fixtures_match") is not True:
        _die(f"{model}/{layout} validation fixtures do not match")
    if payload.get("lane_comparisons") != recomputed_summary.get("comparisons"):
        _die(f"{model}/{layout} lane comparisons are stale or forged")


def _validate_root_worker_bindings(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    source_sha256 = _canonical_sha256(payload.get("source"))
    runtime = _require_mapping(
        payload.get("runtime_provenance"),
        f"{label}.runtime_provenance",
    )
    runtime_sha256 = _canonical_sha256(runtime)
    interpreter = _require_mapping(
        runtime.get("interpreter"),
        f"{label}.runtime_provenance.interpreter",
    )
    native = _require_mapping(
        runtime.get("native_extension"),
        f"{label}.runtime_provenance.native_extension",
    )
    expected_bindings = {
        "source_identity_sha256": source_sha256,
        "runtime_provenance_sha256": runtime_sha256,
        "interpreter_sha256": _require_sha256(
            interpreter.get("sha256"),
            f"{label}.runtime interpreter SHA-256",
        ),
        "native_extension_sha256": _require_sha256(
            native.get("sha256"),
            f"{label}.runtime native extension SHA-256",
        ),
    }
    schedule = _require_mapping(
        payload.get("profile_schedule"),
        f"{label}.profile_schedule",
    )
    entries = _require_list(
        schedule.get("entries"),
        f"{label}.profile_schedule.entries",
    )
    if not entries:
        _die(f"{label} profile schedule is empty")
    for index, entry_value in enumerate(entries):
        entry = _require_mapping(
            entry_value,
            f"{label}.profile_schedule.entries[{index}]",
        )
        verification = _require_mapping(
            entry.get("pre_timing_verification"),
            f"{label}.profile_schedule.entries[{index}].pre_timing_verification",
        )
        for side in ("expected", "observed"):
            identities = _require_mapping(
                verification.get(side),
                f"{label}.profile_schedule.entries[{index}].{side}",
            )
            for key, expected_digest in expected_bindings.items():
                if identities.get(key) != expected_digest:
                    _die(
                        f"{label} worker {index} {side} {key} "
                        "is not bound to the root capture"
                    )


def _axis_contract(
    semantic: Mapping[str, Any],
    key: str,
    label: str,
) -> dict[str, Any]:
    axis = _require_mapping(semantic.get(key), f"{label}.{key}")
    count = _require_int(axis.get("count"), f"{label}.{key}.count", minimum=1)
    ids = _require_list(axis.get("ordered_ids"), f"{label}.{key}.ordered_ids")
    entries = _require_list(
        axis.get("ordered_entries"),
        f"{label}.{key}.ordered_entries",
    )
    if count != len(ids) or count != len(entries):
        _die(f"{label}.{key} count is incomplete")
    if len(set(item for item in ids if isinstance(item, str))) != len(ids):
        _die(f"{label}.{key} ordered IDs are invalid or duplicated")
    if axis.get("ordered_ids_sha256") != _canonical_sha256(ids):
        _die(f"{label}.{key} ordered ID digest is stale")
    if axis.get("ordered_entries_sha256") != _canonical_sha256(entries):
        _die(f"{label}.{key} ordered entry digest is stale")
    return axis


def _model_common_sha256(
    semantic: Mapping[str, Any],
    label: str,
) -> str:
    identity = _require_mapping(
        semantic.get("manifest_model_identity"),
        f"{label}.manifest_model_identity",
    )
    common = _require_mapping(
        identity.get("common_physics_identity"),
        f"{label}.model_common_physics_identity",
    )
    digest = _require_sha256(
        identity.get("common_physics_identity_sha256"),
        f"{label}.model_common_physics_identity_sha256",
    )
    if digest != _canonical_sha256(common):
        _die(f"{label} model common-physics digest is stale")
    return digest


def _semantic_subcontract_sha256(
    semantic: Mapping[str, Any],
    *,
    value_key: str,
    digest_key: str,
    label: str,
) -> str:
    value = semantic.get(value_key)
    digest = _require_sha256(
        semantic.get(digest_key),
        f"{label}.{digest_key}",
    )
    if digest != _canonical_sha256(value):
        _die(f"{label}.{value_key} digest is stale")
    return digest


def _validate_generation_model_identities(
    value: object,
    *,
    model_family: str,
    source_revision: str,
    label: str,
) -> dict[str, Any]:
    identities = _require_mapping(value, label)
    if set(identities) != set(MODES):
        _die(f"{label} must contain exactly compiled/eager/recurrence")
    for mode in MODES:
        identity = _require_mapping(identities[mode], f"{label}.{mode}")
        kind = identity.get("kind")
        expected_kind = (
            "built-in-sm-source"
            if model_family == "built-in-sm" and mode == "compiled"
            else (
                "packaged-prepared-model"
                if model_family == "built-in-sm"
                else "explicit-prepared-model"
            )
        )
        if kind != expected_kind:
            _die(
                f"{label}.{mode} must use {expected_kind} for "
                f"the {model_family} capture role"
            )
        if not isinstance(identity.get("compile_excluded_from_generation"), bool):
            _die(f"{label}.{mode} has no exact compile-exclusion flag")
        if kind == "built-in-sm-source":
            if (
                identity.get("source_revision") != source_revision
                or identity.get("compile_excluded_from_generation") is not False
            ):
                _die(f"{label}.{mode} built-in source identity is inconsistent")
        elif kind == "packaged-prepared-model":
            if (
                identity.get("resource_id") != BUILTIN_PACKAGED_MODEL_RESOURCE_ID
                or identity.get("compile_excluded_from_generation") is not True
            ):
                _die(f"{label}.{mode} packaged model identity is incomplete")
            _require_int(
                identity.get("size_bytes"),
                f"{label}.{mode}.size_bytes",
                minimum=1,
            )
            _require_sha256(identity.get("sha256"), f"{label}.{mode}.sha256")
        else:
            _validate_file_identity_shape(
                identity.get("file"),
                f"{label}.{mode}.file",
            )
            if identity.get("compile_excluded_from_generation") is not True:
                _die(f"{label}.{mode} explicit model must be precompiled")
            if identity.get("resource_id") is not None:
                _die(f"{label}.{mode} explicit model resource ID must be null")
    if model_family == "built-in-sm":
        if identities["eager"] != identities["recurrence"]:
            _die(f"{label} eager/recurrence packaged model identities differ")
    elif model_family == "ufo-sm":
        if any(identities[mode] != identities["compiled"] for mode in MODES[1:]):
            _die(f"{label} UFO prepared-model identities differ across lanes")
    else:
        _die(f"{label} has an unsupported model-family role")
    return identities


def _expected_effective_jit_optimization_level(
    model_identity: Mapping[str, Any],
    *,
    mode: str,
) -> int:
    kind = model_identity.get("kind")
    if kind not in {
        "built-in-sm-source",
        "packaged-prepared-model",
        "explicit-prepared-model",
    }:
        _die("generation model identity has an unsupported source kind")
    if mode not in MODES:
        _die(f"generation execution mode is unsupported: {mode!r}")
    if mode == "compiled":
        return 3
    return PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL


def _selector_entry(
    axis: Mapping[str, Any],
    *,
    stable_id: str,
    field: str,
    expected_value: object,
    label: str,
) -> dict[str, Any]:
    entries = _require_list(axis.get("ordered_entries"), f"{label}.ordered_entries")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("id") == stable_id
    ]
    if len(matches) != 1:
        _die(f"{label} does not contain stable selector {stable_id!r} exactly once")
    entry = matches[0]
    if entry.get(field) != expected_value:
        _die(f"{label} stable selector {stable_id!r} has the wrong {field}")
    if entry.get("structural_zero") is True:
        _die(f"{label} stable selector {stable_id!r} is structurally zero")
    return entry


def _capture_timings(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    profiles = _require_mapping(payload.get("profiles"), f"{label}.profiles")
    result: dict[str, Any] = {}
    for mode in MODES:
        profile = _require_mapping(profiles.get(mode), f"{label}.profiles.{mode}")
        measurements = _require_list(
            profile.get("profiles"),
            f"{label}.profiles.{mode}.profiles",
        )
        by_batch: dict[str, Any] = {}
        for measurement in measurements:
            row = _require_mapping(measurement, f"{label} {mode} timing row")
            batch = row.get("batch_size")
            if batch not in BATCHES:
                _die(f"{label} {mode} has an unexpected timing batch {batch!r}")
            samples = _require_list(
                row.get("subprocess_samples"),
                f"{label} {mode} batch {batch} samples",
            )
            values = [
                float(
                    _require_number(
                        sample,
                        "wall_seconds_per_point",
                        f"{label} {mode} batch {batch} sample",
                        positive=True,
                    )
                )
                for sample in samples
                if isinstance(sample, dict)
            ]
            if len(values) != len(samples) or len(values) < MIN_SAMPLES:
                _die(f"{label} {mode} batch {batch} lacks seven raw samples")
            median = statistics.median(values)
            mad = statistics.median(abs(value - median) for value in values)
            if row.get("wall_seconds_per_point_median") != median:
                _die(f"{label} {mode} batch {batch} median is stale")
            if row.get("wall_seconds_per_point_mad") != mad:
                _die(f"{label} {mode} batch {batch} MAD is stale")
            by_batch[str(batch)] = {
                "sample_count": len(values),
                "median_seconds_per_point": median,
                "mad_seconds_per_point": mad,
                "raw_seconds_per_point": values,
                "timing_boundary": "runtime_core_repeated_wall_time",
            }
        if set(by_batch) != {str(batch) for batch in BATCHES}:
            _die(f"{label} {mode} does not cover batches 1/128/1024")
        result[mode] = by_batch
    return result


def _require_number(
    value: Mapping[str, Any],
    key: str,
    label: str,
    *,
    positive: bool = False,
) -> float:
    raw = value.get(key)
    if not _is_number(raw):
        _die(f"{label}.{key} must be a finite{' positive' if positive else ''} number")
    assert isinstance(raw, (int, float)) and not isinstance(raw, bool)
    if positive and float(raw) <= 0.0:
        _die(f"{label}.{key} must be a finite positive number")
    return float(raw)


def _capture_fixture_and_values(
    payload: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Any], tuple[complex, ...]]:
    profiles = _require_mapping(payload.get("profiles"), f"{label}.profiles")
    fixtures: list[dict[str, Any]] = []
    values: list[tuple[complex, ...]] = []
    for mode in MODES:
        profile = _require_mapping(profiles.get(mode), f"{label}.profiles.{mode}")
        validation = _require_mapping(
            profile.get("validation"),
            f"{label}.profiles.{mode}.validation",
        )
        fixture = _require_mapping(
            validation.get("fixture"),
            f"{label}.profiles.{mode}.validation.fixture",
        )
        point_count = _require_int(
            fixture.get("point_count"),
            f"{label}.{mode}.fixture.point_count",
            minimum=1,
        )
        points_sha = _require_sha256(
            fixture.get("points_sha256"),
            f"{label}.{mode}.fixture.points_sha256",
        )
        selected = _complex_values(
            validation.get("selected_totals"),
            f"{label}.{mode}.selected_totals",
        )
        resolved = _complex_values(
            validation.get("resolved_sums"),
            f"{label}.{mode}.resolved_sums",
        )
        if len(selected) != point_count:
            _die(f"{label}.{mode} validation point inventory is incomplete")
        _assert_values_close(selected, resolved, f"{label}.{mode} resolved-sum closure")
        file_identity = _require_mapping(
            fixture.get("file"),
            f"{label}.{mode}.fixture.file",
        )
        fixture_path = file_identity.get("resolved_path", file_identity.get("path"))
        fixture_size = _require_int(
            file_identity.get("size_bytes"),
            f"{label}.{mode}.fixture.file.size_bytes",
            minimum=0,
        )
        fixture_sha = _require_sha256(
            file_identity.get("sha256"),
            f"{label}.{mode}.fixture.file.sha256",
        )
        if not isinstance(fixture_path, str) or not Path(fixture_path).is_absolute():
            _die(f"{label}.{mode} fixture file identity is invalid")
        fixture_file = Path(fixture_path).resolve(strict=False)
        if (
            not fixture_file.is_file()
            or fixture_file.stat().st_size != fixture_size
            or _file_sha256(fixture_file) != fixture_sha
        ):
            _die(f"{label}.{mode} validation fixture file drifted")
        fixtures.append(
            {
                "point_count": point_count,
                "points_sha256": points_sha,
                "file_sha256": fixture_sha,
            }
        )
        values.append(selected)
    if any(item != fixtures[0] for item in fixtures[1:]):
        _die(f"{label} execution lanes use different validation fixtures")
    for mode, mode_values in zip(MODES[1:], values[1:], strict=True):
        _assert_values_close(values[0], mode_values, f"{label} compiled versus {mode}")
    return fixtures[0], values[0]


def _validate_capture(
    loaded: LoadedJson,
    *,
    model: str,
    layout: str,
    expected: Mapping[str, Any],
    benchmark: ModuleType,
) -> Capture:
    label = f"{model}/{layout}"
    _revalidate_capture(loaded, model=model, layout=layout, benchmark=benchmark)
    payload = loaded.payload
    configuration = _require_mapping(payload["configuration"], f"{label}.configuration")
    expected_workload = (
        UNION_WORKLOAD if layout == "all-flow-union" else SELECTED_WORKLOAD
    )
    if payload.get("workload") != expected_workload:
        _die(f"{label} has the wrong workload")
    if _normalized_process(payload.get("process")) != PROCESS:
        _die(f"{label} is not the uubar Z+6g standard candle")
    if configuration.get("lc_flow_layout") != layout:
        _die(f"{label} configuration layout does not match its request role")
    if configuration.get("modes") != list(MODES):
        _die(f"{label} must contain exactly compiled/eager/recurrence")
    if configuration.get("batch_sizes") != list(BATCHES):
        _die(f"{label} must contain exactly batches 1/128/1024")
    _require_int(
        configuration.get("minimum_samples"),
        f"{label}.configuration.minimum_samples",
        minimum=MIN_SAMPLES,
    )
    _require_int(
        configuration.get("subprocess_samples"),
        f"{label}.configuration.subprocess_samples",
        minimum=MIN_SAMPLES,
    )
    if configuration.get("jit_optimization_level") != 3:
        _die(f"{label} is not a JIT O3 capture")
    if configuration.get("generation_only") is not False:
        _die(f"{label} is generation-only")
    if configuration.get("specialize_flow_at_generation") is not False:
        _die(f"{label} is generation-fixed and cannot be a headline capture")
    if configuration.get("external_watchdog_required_for_long_runs") is not True:
        _die(f"{label} does not require the 30 GiB external watchdog")
    capture_acceptance = _require_mapping(
        payload["capture_acceptance"],
        f"{label}.capture_acceptance",
    )
    if (
        capture_acceptance.get("kind") != CAPTURE_ACCEPTANCE_KIND
        or capture_acceptance.get("schema_version") != CAPTURE_ACCEPTANCE_SCHEMA
        or capture_acceptance.get("complete") is not True
        or capture_acceptance.get("passes") is not True
        or capture_acceptance.get("authoritative_eligible") is not True
    ):
        _die(f"{label} is not an authoritative passing layout capture")
    per_layout_m0 = _require_mapping(
        payload["milestone0_acceptance"],
        f"{label}.milestone0_acceptance",
    )
    if (
        per_layout_m0.get("kind") != PER_LAYOUT_M0_KIND
        or per_layout_m0.get("schema_version") != PER_LAYOUT_M0_SCHEMA
        or per_layout_m0.get("accepted") is not False
        or per_layout_m0.get("status") != "incomplete"
    ):
        _die(f"{label} improperly self-certifies its per-layout M0 evidence")

    source_identity = _source_semantics(payload["source"])
    if source_identity["revision"] != expected["pyamplicol_source_revision"]:
        _die(f"{label} source revision differs from the request pin")
    runtime_identity = _runtime_semantics(
        payload["runtime_provenance"],
        source_revision=source_identity["revision"],
    )
    if _canonical_sha256(runtime_identity) != expected["runtime_provenance_sha256"]:
        _die(f"{label} runtime provenance differs from the request pin")
    provenance = _require_mapping(payload["provenance"], f"{label}.provenance")
    if provenance.get("external_watchdog_required_for_long_runs") is not True:
        _die(f"{label} provenance does not record the required external watchdog")
    host = _validate_host(provenance.get("host"), f"{label}.provenance.host")
    if _canonical_sha256(host) != expected["host_sha256"]:
        _die(f"{label} host identity differs from the request pin")
    _validate_root_worker_bindings(payload, label=label)

    profiles = _require_mapping(payload["profiles"], f"{label}.profiles")
    compiled = _require_mapping(profiles.get("compiled"), f"{label}.profiles.compiled")
    semantic = _require_mapping(
        compiled.get("artifact_semantic_identity"),
        f"{label}.compiled.artifact_semantic_identity",
    )
    coverage = _require_mapping(
        semantic.get("coverage"),
        f"{label}.compiled.artifact_semantic_identity.coverage",
    )
    if coverage.get("complete_physical_axes") is not True:
        _die(f"{label} does not retain complete physical axes")
    if semantic.get("generation_specialized_axes") != []:
        _die(f"{label} has generation-specialized axes")
    color_axis = _axis_contract(semantic, "physical_color_flows", label)
    helicity_axis = _axis_contract(semantic, "physical_helicities", label)
    runtime_selector_semantics_sha = _semantic_subcontract_sha256(
        semantic,
        value_key="runtime_selector_semantics",
        digest_key="runtime_selector_semantics_sha256",
        label=label,
    )
    reduction_ordering_sha = _semantic_subcontract_sha256(
        semantic,
        value_key="reduction_ordering",
        digest_key="reduction_ordering_sha256",
        label=label,
    )
    execution_schedule_ordering_by_mode: dict[str, str] = {}
    for mode in MODES:
        mode_profile = _require_mapping(
            profiles.get(mode),
            f"{label}.profiles.{mode}",
        )
        mode_semantic = _require_mapping(
            mode_profile.get("artifact_semantic_identity"),
            f"{label}.profiles.{mode}.artifact_semantic_identity",
        )
        execution_schedule_ordering_by_mode[mode] = _semantic_subcontract_sha256(
            mode_semantic,
            value_key="execution_schedule_ordering",
            digest_key="execution_schedule_ordering_sha256",
            label=f"{label}.{mode}",
        )
    normalization = _require_sha256(
        semantic.get("normalization_sha256"),
        f"{label}.normalization_sha256",
    )
    if normalization != expected["normalization_sha256"]:
        _die(f"{label} normalization differs from the request pin")
    model_common_sha = _model_common_sha256(semantic, label)
    model_identity = _require_mapping(
        semantic.get("manifest_model_identity"),
        f"{label}.manifest_model_identity",
    )
    model_common = _require_mapping(
        model_identity.get("common_physics_identity"),
        f"{label}.model_common_physics_identity",
    )
    expected_model_name = "built-in-sm" if model == "built-in-sm" else "sm"
    if model_common.get("name") != expected_model_name:
        _die(f"{label} has the wrong model-family identity")
    expected_model_sha = expected["model_common_physics_identity_sha256"][model]
    if model_common_sha != expected_model_sha:
        _die(f"{label} model identity differs from the request pin")

    generation_model_identities = _validate_generation_model_identities(
        configuration.get("model_identities"),
        model_family=model,
        source_revision=source_identity["revision"],
        label=f"{label}.configuration.model_identities",
    )
    generation_models_sha = _canonical_sha256(
        _path_stripped(generation_model_identities)
    )
    if generation_models_sha != expected["generation_model_identities_sha256"][model]:
        _die(f"{label} generation model identities differ from the request pin")
    prepared_model_path = configuration.get("prepared_model_path")
    if model == "built-in-sm":
        if prepared_model_path is not None:
            _die(f"{label} built-in capture must not use an explicit prepared model")
    else:
        ufo_file = _require_mapping(
            generation_model_identities["compiled"].get("file"),
            f"{label}.configuration.model_identities.compiled.file",
        )
        if not isinstance(
            prepared_model_path, str
        ) or prepared_model_path != ufo_file.get("resolved_path"):
            _die(f"{label} UFO prepared-model path is not identity-bound")

    generation = _require_mapping(payload["generation"], f"{label}.generation")
    if set(generation) != set(MODES):
        _die(f"{label} generation inventory is incomplete")
    for mode in MODES:
        record = _require_mapping(generation[mode], f"{label}.generation.{mode}")
        expected_mode_model = generation_model_identities[mode]
        if record.get("model_source") != expected_mode_model:
            _die(f"{label} generation lane {mode} model source is not bound")
        contract = _require_mapping(
            record.get("effective_contract"),
            f"{label}.generation.{mode}.effective_contract",
        )
        expected_effective_jit_level = _expected_effective_jit_optimization_level(
            expected_mode_model,
            mode=mode,
        )
        if (
            contract.get("execution_mode") != mode
            or contract.get("backend") != "jit"
            or contract.get("jit_optimization_level") != expected_effective_jit_level
            or contract.get("color_accuracy") != "lc"
            or contract.get("lc_flow_layout") != layout
        ):
            _die(
                f"{label} generation lane {mode} has the wrong effective "
                "JIT optimization level"
            )
        signature = _require_mapping(
            record.get("semantic_generation_signature"),
            f"{label}.generation.{mode}.semantic_generation_signature",
        )
        signature_digest = _require_sha256(
            record.get("semantic_generation_signature_sha256"),
            f"{label}.generation.{mode}.semantic_generation_signature_sha256",
        )
        if signature_digest != _canonical_sha256(signature):
            _die(f"{label} generation lane {mode} signature digest is stale")
        if (
            signature.get("source_revision") != source_identity["revision"]
            or signature.get("runtime_provenance_sha256")
            != _canonical_sha256(payload["runtime_provenance"])
            or signature.get("mode") != mode
            or _normalized_process(signature.get("process")) != PROCESS
            or signature.get("lc_flow_layout") != layout
            or signature.get("jit_optimization_level") != 3
            or signature.get("model") != expected_mode_model
        ):
            _die(f"{label} generation lane {mode} signature is not root-bound")
        mode_profile = _require_mapping(
            profiles.get(mode),
            f"{label}.profiles.{mode}",
        )
        if record.get("artifact_semantic_identity") != mode_profile.get(
            "artifact_semantic_identity"
        ) or record.get("artifact_semantic_identity_sha256") != mode_profile.get(
            "artifact_semantic_identity_sha256"
        ):
            _die(f"{label} generation lane {mode} artifact semantics are not bound")

    expected_color = _require_mapping(expected["color_flow"], "expected.color_flow")
    expected_helicity = _require_mapping(expected["helicity"], "expected.helicity")
    color_id = expected_color["id"]
    helicity_id = expected_helicity["id"]
    _selector_entry(
        color_axis,
        stable_id=color_id,
        field="word",
        expected_value=expected_color["word"],
        label=f"{label}.physical_color_flows",
    )
    _selector_entry(
        helicity_axis,
        stable_id=helicity_id,
        field="values",
        expected_value=expected_helicity["values"],
        label=f"{label}.physical_helicities",
    )
    selector = _require_mapping(
        compiled.get("selector_contract"),
        f"{label}.compiled.selector_contract",
    )
    if layout == "topology-replay":
        if (
            selector.get("resolved_color_flow_id") != color_id
            or selector.get("resolved_helicity_id") is not None
        ):
            _die(f"{label} does not runtime-select the pinned physical flow")
    elif (
        selector.get("resolved_color_flow_id") is not None
        or selector.get("resolved_helicity_id") != helicity_id
    ):
        _die(f"{label} does not runtime-select the pinned source helicity")
    for mode in MODES[1:]:
        other = _require_mapping(profiles[mode], f"{label}.profiles.{mode}")
        if other.get("selector_contract") != selector:
            _die(f"{label} lane selectors differ")

    fixture, values = _capture_fixture_and_values(payload, label)
    if fixture["points_sha256"] != expected["momenta_points_sha256"]:
        _die(f"{label} momenta fixture differs from the request pin")
    return Capture(
        model=model,
        layout=layout,
        loaded=loaded,
        source_identity=source_identity,
        runtime_identity=runtime_identity,
        host=host,
        fixture=fixture,
        normalization_sha256=normalization,
        model_common_sha256=model_common_sha,
        generation_models_sha256=generation_models_sha,
        color_axis=color_axis,
        helicity_axis=helicity_axis,
        runtime_selector_semantics_sha256=runtime_selector_semantics_sha,
        reduction_ordering_sha256=reduction_ordering_sha,
        execution_schedule_ordering_sha256_by_mode=(
            execution_schedule_ordering_by_mode
        ),
        selector=selector,
        timings=_capture_timings(payload, label),
        validation_values=values,
    )


def _validate_embedded_file_ref(value: object, label: str) -> dict[str, Any]:
    raw = _require_mapping(value, label)
    _require_exact_keys(raw, _FILE_REF_KEYS, label)
    path_value = raw["path"]
    size = raw["size_bytes"]
    digest = _require_sha256(raw["sha256"], f"{label}.sha256")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        _die(f"{label}.path must be an absolute path")
    if not _is_int(size) or size < 0:
        _die(f"{label}.size_bytes must be a non-negative integer")
    path = Path(path_value).resolve(strict=False)
    if not path.is_file():
        _die(f"{label} file is missing: {path}")
    if path.stat().st_size != size or _file_sha256(path) != digest:
        _die(f"{label} file content drifted")
    return {"path": str(path), "size_bytes": size, "sha256": digest}


def _validate_amplicol_raw_sample(
    identity: Mapping[str, Any],
    *,
    role: str,
    sample_index: int,
    command_sha256: str,
    evaluated_point_count: int,
    elapsed_seconds: float,
    seconds_per_point: float,
    selected_values: Sequence[complex],
    resolved_values: Sequence[complex],
    label: str,
) -> None:
    path = Path(identity["path"])
    try:
        payload = _strict_json_bytes(path.read_bytes(), label)
    except OSError as error:
        raise EvidenceError(f"cannot read {label}: {error}") from error
    _require_exact_keys(
        payload,
        {
            "kind",
            "schema_version",
            "role",
            "sample_index",
            "command_sha256",
            "evaluated_point_count",
            "elapsed_seconds",
            "seconds_per_point",
            "stdout",
            "stdout_sha256",
            "content_sha256",
        },
        label,
    )
    without_digest = dict(payload)
    content_sha256 = without_digest.pop("content_sha256")
    if (
        payload.get("kind") != "pyamplicol-amplicol-m0-raw-sample"
        or payload.get("schema_version") != 1
        or _require_sha256(content_sha256, f"{label}.content_sha256")
        != _canonical_sha256(without_digest)
        or payload.get("role") != role
        or payload.get("sample_index") != sample_index
        or payload.get("command_sha256") != command_sha256
        or payload.get("evaluated_point_count") != evaluated_point_count
        or payload.get("elapsed_seconds") != elapsed_seconds
        or payload.get("seconds_per_point") != seconds_per_point
    ):
        _die(f"{label} is not bound to its timing sample")
    stdout = payload.get("stdout")
    if not isinstance(stdout, str) or not stdout:
        _die(f"{label}.stdout must retain non-empty raw probe output")
    stdout_sha256 = _require_sha256(
        payload.get("stdout_sha256"),
        f"{label}.stdout_sha256",
    )
    if stdout_sha256 != hashlib.sha256(stdout.encode("utf-8")).hexdigest():
        _die(f"{label}.stdout digest is stale")
    stdout_payload = _strict_json_bytes(
        stdout.encode("utf-8"),
        f"{label}.stdout",
    )
    _require_exact_keys(
        stdout_payload,
        {
            "kind",
            "schema_version",
            "role",
            "sample_index",
            "evaluated_point_count",
            "elapsed_seconds",
            "seconds_per_point",
            "selected_totals",
            "resolved_sums",
        },
        f"{label}.stdout",
    )
    if (
        stdout_payload.get("kind") != "amplicol-m0-probe-result"
        or stdout_payload.get("schema_version") != 1
        or stdout_payload.get("role") != role
        or stdout_payload.get("sample_index") != sample_index
        or stdout_payload.get("evaluated_point_count") != evaluated_point_count
        or stdout_payload.get("elapsed_seconds") != elapsed_seconds
        or stdout_payload.get("seconds_per_point") != seconds_per_point
    ):
        _die(f"{label}.stdout probe fields are not bound to the sample")
    stdout_selected = _complex_values(
        stdout_payload.get("selected_totals"),
        f"{label}.stdout.selected_totals",
    )
    stdout_resolved = _complex_values(
        stdout_payload.get("resolved_sums"),
        f"{label}.stdout.resolved_sums",
    )
    _assert_values_close(
        selected_values,
        stdout_selected,
        f"{label}.stdout selected totals",
    )
    _assert_values_close(
        resolved_values,
        stdout_resolved,
        f"{label}.stdout resolved sums",
    )


def _validate_amplicol(
    loaded: LoadedJson,
    *,
    role: str,
    expected: Mapping[str, Any],
) -> AmpliColEvidence:
    payload = loaded.payload
    label = f"amplicol/{role}"
    _require_exact_keys(payload, _AMPLICOL_ROOT_KEYS, label)
    if payload.get("kind") != AMPLICOL_KIND or payload.get("schema_version") != 1:
        _die(f"{label} has the wrong evidence schema")
    without_digest = dict(payload)
    content_sha = without_digest.pop("content_sha256")
    if _require_sha256(content_sha, f"{label}.content_sha256") != _canonical_sha256(
        without_digest
    ):
        _die(f"{label} canonical content digest is stale")
    if payload.get("complete") is not True:
        _die(f"{label} is incomplete")
    if payload.get("evidence_scope") != "authoritative-host-capture-v1":
        _die(
            f"{label} is fixture, synthetic, diagnostic, or otherwise non-authoritative"
        )
    expected_workload = (
        SELECTED_WORKLOAD if role == AMPLICOL_SELECTED_ROLE else UNION_WORKLOAD
    )
    if payload.get("workload") != expected_workload:
        _die(f"{label} has the wrong workload")

    source = _require_mapping(payload.get("source"), f"{label}.source")
    _require_exact_keys(
        source,
        {"revision", "dirty", "compiler", "source_tree_sha256"},
        f"{label}.source",
    )
    if (
        source.get("revision") != expected["amplicol_source_revision"]
        or source.get("dirty") is not False
    ):
        _die(f"{label} source revision/cleanliness differs from the request pin")
    _require_sha256(source.get("source_tree_sha256"), f"{label}.source_tree_sha256")
    compiler = _require_mapping(source.get("compiler"), f"{label}.source.compiler")
    _require_exact_keys(
        compiler,
        {"id", "version", "target", "flags_sha256"},
        f"{label}.source.compiler",
    )
    if any(
        not isinstance(compiler.get(key), str) or not compiler[key]
        for key in ("id", "version", "target")
    ):
        _die(f"{label} compiler identity is incomplete")
    _require_sha256(
        compiler.get("flags_sha256"),
        f"{label}.source.compiler.flags_sha256",
    )

    host = _validate_host(payload.get("host"), f"{label}.host")
    if _canonical_sha256(host) != expected["host_sha256"]:
        _die(f"{label} host identity differs from the request pin")
    process = _require_mapping(payload.get("process"), f"{label}.process")
    _require_exact_keys(
        process,
        {"expression", "normalized_expression"},
        f"{label}.process",
    )
    if (
        _normalized_process(process.get("expression")) != PROCESS
        or process.get("normalized_expression") != PROCESS
    ):
        _die(f"{label} is not the uubar Z+6g standard candle")

    physical_axes = _require_mapping(
        payload.get("physical_axes"),
        f"{label}.physical_axes",
    )
    _require_exact_keys(
        physical_axes,
        {"color_flow", "helicity"},
        f"{label}.physical_axes",
    )
    selector = _require_mapping(payload.get("selector"), f"{label}.selector")
    _require_exact_keys(
        selector,
        {
            "color_flow_request",
            "resolved_color_flow_id",
            "color_flow_word",
            "helicity_request",
            "resolved_helicity_id",
            "helicity_values",
            "sum_axis",
            "source_to_generated_permutation",
            "complete_physical_axes",
            "generation_specialized_axes",
        },
        f"{label}.selector",
    )
    color_expected = _require_mapping(expected["color_flow"], "expected.color_flow")
    helicity_expected = _require_mapping(expected["helicity"], "expected.helicity")
    if selector.get("complete_physical_axes") is not True:
        _die(f"{label} does not retain complete physical axes")
    if selector.get("generation_specialized_axes") != []:
        _die(f"{label} is a generation-fixed diagnostic")
    permutation = _require_list(
        selector.get("source_to_generated_permutation"),
        f"{label}.selector.source_to_generated_permutation",
    )
    if (
        len(permutation) != EXTERNAL_LEG_COUNT
        or any(not _is_int(item) or item < 0 for item in permutation)
        or sorted(permutation) != list(range(EXTERNAL_LEG_COUNT))
    ):
        _die(f"{label} source-to-generated permutation is invalid")
    if permutation != expected["external_leg_permutation"]:
        _die(f"{label} source-to-generated permutation differs from the request pin")
    if any(
        not isinstance(selector.get(key), str) or not selector[key]
        for key in ("color_flow_request", "helicity_request")
    ):
        _die(f"{label} runtime selector requests are invalid")
    if role == AMPLICOL_SELECTED_ROLE:
        selector_ok = (
            selector.get("resolved_color_flow_id") == color_expected["id"]
            and selector.get("color_flow_word") == color_expected["word"]
            and selector.get("resolved_helicity_id") is None
            and selector.get("sum_axis") == "helicity"
        )
    else:
        selector_ok = (
            selector.get("resolved_color_flow_id") is None
            and selector.get("resolved_helicity_id") == helicity_expected["id"]
            and selector.get("helicity_values") == helicity_expected["values"]
            and selector.get("sum_axis") == "color_flow"
        )
    if not selector_ok:
        _die(f"{label} resolved selector does not match the request pin")

    validated_axes: dict[str, dict[str, Any]] = {}
    for axis_name in ("color_flow", "helicity"):
        axis = _require_mapping(
            physical_axes.get(axis_name),
            f"{label}.physical_axes.{axis_name}",
        )
        _require_exact_keys(
            axis,
            {"count", "ordered_ids_sha256"},
            f"{label}.physical_axes.{axis_name}",
        )
        if not _is_int(axis.get("count")) or axis["count"] <= 0:
            _die(f"{label}.{axis_name} physical-axis count is invalid")
        _require_sha256(
            axis.get("ordered_ids_sha256"),
            f"{label}.{axis_name}.ordered_ids_sha256",
        )
        validated_axes[axis_name] = axis

    momenta = _require_mapping(payload.get("momenta"), f"{label}.momenta")
    _require_exact_keys(
        momenta,
        {"point_count", "points_sha256", "raw_file"},
        f"{label}.momenta",
    )
    point_count = _require_int(
        momenta.get("point_count"),
        f"{label}.momenta.point_count",
        minimum=1,
    )
    if momenta.get("points_sha256") != expected["momenta_points_sha256"]:
        _die(f"{label} momenta differ from the request pin")
    momenta_file = _validate_embedded_file_ref(
        momenta.get("raw_file"),
        f"{label}.momenta.raw_file",
    )
    if payload.get("normalization_sha256") != expected["normalization_sha256"]:
        _die(f"{label} normalization differs from the request pin")

    binary = _require_mapping(
        payload.get("binary_evidence"),
        f"{label}.binary_evidence",
    )
    _require_exact_keys(
        binary,
        {"executable", "linked_libraries", "source_files"},
        f"{label}.binary_evidence",
    )
    evidence_paths: list[str] = []
    executable = _validate_embedded_file_ref(
        binary.get("executable"),
        f"{label}.binary_evidence.executable",
    )
    if not os.access(executable["path"], os.X_OK):
        _die(f"{label} content-addressed executable is not executable")
    evidence_paths.append(executable["path"])
    source_file_identities: list[dict[str, Any]] = []
    for collection in ("linked_libraries", "source_files"):
        rows = _require_list(
            binary.get(collection),
            f"{label}.binary_evidence.{collection}",
        )
        if not rows:
            _die(f"{label}.binary_evidence.{collection} must not be empty")
        for index, row in enumerate(rows):
            identity = _validate_embedded_file_ref(
                row,
                f"{label}.binary_evidence.{collection}[{index}]",
            )
            evidence_paths.append(identity["path"])
            if collection == "source_files":
                source_file_identities.append(identity)
    if len(set(evidence_paths)) != len(evidence_paths):
        _die(f"{label} binary/source evidence files are duplicated")
    if source.get("source_tree_sha256") != _amplicol_source_tree_sha256(
        source_file_identities
    ):
        _die(f"{label} source-tree digest is not bound to source files")

    validation = _require_mapping(payload.get("validation"), f"{label}.validation")
    _require_exact_keys(
        validation,
        {
            "selected_totals",
            "resolved_sums",
            "point_comparisons",
            "maximum_absolute_difference",
            "maximum_relative_difference",
            "passes",
        },
        f"{label}.validation",
    )
    selected = _complex_values(
        validation.get("selected_totals"),
        f"{label}.validation.selected_totals",
    )
    resolved = _complex_values(
        validation.get("resolved_sums"),
        f"{label}.validation.resolved_sums",
    )
    if len(selected) != point_count:
        _die(f"{label} validation point inventory is incomplete")
    _assert_values_close(selected, resolved, f"{label} resolved-sum closure")
    comparisons = _require_list(
        validation.get("point_comparisons"),
        f"{label}.validation.point_comparisons",
    )
    if len(comparisons) != point_count:
        _die(f"{label} point-comparison inventory is incomplete")
    max_abs = 0.0
    max_rel = 0.0
    for index, (lhs, rhs, comparison) in enumerate(
        zip(selected, resolved, comparisons, strict=True)
    ):
        row = _require_mapping(comparison, f"{label}.point_comparisons[{index}]")
        if row.get("point_index") != index:
            _die(f"{label} point-comparison indices are not source ordered")
        difference = abs(lhs - rhs)
        relative = difference / max(abs(rhs), ATOL)
        max_abs = max(max_abs, difference)
        max_rel = max(max_rel, relative)
        if row.get("passes") is not _close(lhs, rhs):
            _die(f"{label} point-comparison pass flag is stale")
    if (
        validation.get("passes") is not True
        or validation.get("maximum_absolute_difference") != max_abs
        or validation.get("maximum_relative_difference") != max_rel
    ):
        _die(f"{label} validation summary is stale or non-passing")

    timing = _require_mapping(payload.get("timing"), f"{label}.timing")
    _require_exact_keys(
        timing,
        {
            "boundary",
            "batch_semantics",
            "statistics_contract",
            "interleave_group_sha256",
            "sample_count",
            "median_seconds_per_point",
            "mad_seconds_per_point",
            "samples",
            "samples_sha256",
        },
        f"{label}.timing",
    )
    expected_boundary = (
        "amplitude-evaluation"
        if role == AMPLICOL_SELECTED_ROLE
        else "direct-library-total"
    )
    if (
        timing.get("boundary") != expected_boundary
        or timing.get("batch_semantics") != "scalar-normalized-per-point"
        or timing.get("statistics_contract") != "subprocess-median-and-raw-mad-v1"
    ):
        _die(f"{label} timing boundary/statistics are not authoritative")
    interleave_group_sha256 = _require_sha256(
        timing.get("interleave_group_sha256"),
        f"{label}.timing.interleave_group_sha256",
    )
    samples = _require_list(timing.get("samples"), f"{label}.timing.samples")
    if len(samples) < MIN_SAMPLES or timing.get("sample_count") != len(samples):
        _die(f"{label} has fewer than seven subprocess timing samples")
    if timing.get("samples_sha256") != _canonical_sha256(samples):
        _die(f"{label} raw timing sample digest is stale")
    values: list[float] = []
    raw_paths: list[str] = []
    previous_finish: dt.datetime | None = None
    rounds: list[int] = []
    command_hashes: set[str] = set()
    interleave_records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        row = _require_mapping(sample, f"{label}.timing.samples[{index}]")
        _require_exact_keys(
            row,
            {
                "sample_index",
                "interleave_round",
                "interleave_position",
                "started_at_utc",
                "finished_at_utc",
                "subprocess",
                "command",
                "command_sha256",
                "evaluated_point_count",
                "elapsed_seconds",
                "seconds_per_point",
                "interrupted",
                "raw_output_file",
            },
            f"{label}.timing.samples[{index}]",
        )
        if row.get("sample_index") != index or row.get("subprocess") is not True:
            _die(f"{label} timing samples are not independent subprocess records")
        round_index = _require_int(
            row.get("interleave_round"),
            f"{label} sample {index} interleave round",
            minimum=0,
        )
        interleave_position = _require_int(
            row.get("interleave_position"),
            f"{label} sample {index} interleave position",
            minimum=0,
        )
        rounds.append(round_index)
        started = _utc(row.get("started_at_utc"), f"{label} sample {index} start")
        finished = _utc(row.get("finished_at_utc"), f"{label} sample {index} finish")
        if started >= finished or (
            previous_finish is not None and started < previous_finish
        ):
            _die(f"{label} timing samples are overlapping or not chronological")
        previous_finish = finished
        command = _require_list(
            row.get("command"),
            f"{label} sample {index} command",
        )
        if not command or any(
            not isinstance(item, str) or not item for item in command
        ):
            _die(f"{label} sample {index} command is invalid")
        selector_argument = (
            f"--color-flow-id={color_expected['id']}"
            if role == AMPLICOL_SELECTED_ROLE
            else f"--helicity-id={helicity_expected['id']}"
        )
        required_arguments = {
            f"--workload={role}",
            f"--round={index}",
            f"--momenta={momenta_file['path']}",
            f"--source-revision={source['revision']}",
            selector_argument,
        }
        if command[0] != executable["path"] or not required_arguments.issubset(
            command[1:]
        ):
            _die(f"{label} sample {index} command is not bound to its evidence")
        command_sha = _require_sha256(
            row.get("command_sha256"),
            f"{label} sample {index} command SHA",
        )
        if command_sha != _canonical_sha256(command):
            _die(f"{label} sample {index} command digest is stale")
        if command_sha in command_hashes:
            _die(f"{label} timing subprocess commands are not independently addressed")
        command_hashes.add(command_sha)
        evaluated = _require_int(
            row.get("evaluated_point_count"),
            f"{label} sample {index} evaluated-point count",
            minimum=1,
        )
        elapsed = _require_number(
            row,
            "elapsed_seconds",
            f"{label} sample {index}",
            positive=True,
        )
        seconds_per_point = _require_number(
            row,
            "seconds_per_point",
            f"{label} sample {index}",
            positive=True,
        )
        process_envelope_seconds = (finished - started).total_seconds()
        if elapsed > process_envelope_seconds + max(
            1.0e-9,
            process_envelope_seconds * 1.0e-9,
        ):
            _die(f"{label} sample {index} timing exceeds its subprocess envelope")
        if not math.isclose(
            seconds_per_point,
            elapsed / evaluated,
            rel_tol=0.0,
            abs_tol=max(1.0e-18, abs(seconds_per_point) * 1.0e-15),
        ):
            _die(f"{label} sample {index} seconds/point is stale")
        if row.get("interrupted") is not False:
            _die(f"{label} sample {index} was interrupted")
        raw_file = _validate_embedded_file_ref(
            row.get("raw_output_file"),
            f"{label}.timing.samples[{index}].raw_output_file",
        )
        _validate_amplicol_raw_sample(
            raw_file,
            role=role,
            sample_index=index,
            command_sha256=command_sha,
            evaluated_point_count=evaluated,
            elapsed_seconds=elapsed,
            seconds_per_point=seconds_per_point,
            selected_values=selected,
            resolved_values=resolved,
            label=f"{label}.timing.samples[{index}].raw_output",
        )
        raw_paths.append(raw_file["path"])
        values.append(seconds_per_point)
        interleave_records.append(
            {
                "role": role,
                "round": round_index,
                "position": interleave_position,
                "started_at_utc": row["started_at_utc"],
                "finished_at_utc": row["finished_at_utc"],
                "command_sha256": command_sha,
            }
        )
    if sorted(rounds) != list(range(len(samples))):
        _die(f"{label} timing samples do not cover unique interleave rounds")
    if len(set(raw_paths)) != len(raw_paths):
        _die(f"{label} timing samples reuse raw-output files")
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    if (
        timing.get("median_seconds_per_point") != median
        or timing.get("mad_seconds_per_point") != mad
    ):
        _die(f"{label} median/MAD is stale")
    return AmpliColEvidence(
        role=role,
        loaded=loaded,
        source_identity=source,
        host=host,
        momenta_file_sha256=momenta_file["sha256"],
        color_axis=validated_axes["color_flow"],
        helicity_axis=validated_axes["helicity"],
        selector=selector,
        values=selected,
        timing={
            "sample_count": len(values),
            "median_seconds_per_point": median,
            "mad_seconds_per_point": mad,
            "raw_seconds_per_point": values,
            "timing_boundary": expected_boundary,
            "batch_semantics": "scalar-normalized-per-point",
        },
        interleave_group_sha256=interleave_group_sha256,
        interleave_records=tuple(interleave_records),
    )


def _validate_expected(value: object) -> dict[str, Any]:
    expected = _require_mapping(value, "request.expected")
    _require_exact_keys(expected, _EXPECTED_KEYS, "request.expected")
    for key in ("pyamplicol_source_revision", "amplicol_source_revision"):
        revision = expected.get(key)
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            _die(f"request.expected.{key} must be a full lowercase Git SHA")
    if _normalized_process(expected.get("process")) != PROCESS:
        _die("request.expected.process must be the uubar Z+6g standard candle")
    for key in (
        "runtime_provenance_sha256",
        "host_sha256",
        "momenta_points_sha256",
        "normalization_sha256",
    ):
        _require_sha256(expected.get(key), f"request.expected.{key}")
    for key in (
        "model_common_physics_identity_sha256",
        "generation_model_identities_sha256",
    ):
        mapping = _require_mapping(expected.get(key), f"request.expected.{key}")
        if set(mapping) != set(MODELS):
            _die(f"request.expected.{key} must pin built-in-sm and ufo-sm")
        for model in MODELS:
            _require_sha256(mapping[model], f"request.expected.{key}.{model}")
    color = _require_mapping(expected.get("color_flow"), "request.expected.color_flow")
    _require_exact_keys(color, {"id", "word"}, "request.expected.color_flow")
    if not isinstance(color.get("id"), str) or not color["id"].startswith("flow:"):
        _die("request expected color flow must use a stable flow ID")
    word = _require_list(color.get("word"), "request.expected.color_flow.word")
    if len(word) != LC_COLOR_WORD_LENGTH or any(not _is_int(item) for item in word):
        _die("request expected color-flow word is invalid")
    helicity = _require_mapping(expected.get("helicity"), "request.expected.helicity")
    _require_exact_keys(helicity, {"id", "values"}, "request.expected.helicity")
    if not isinstance(helicity.get("id"), str) or not helicity["id"].startswith("h:"):
        _die("request expected helicity must use a stable helicity ID")
    values = _require_list(helicity.get("values"), "request.expected.helicity.values")
    if len(values) != EXTERNAL_LEG_COUNT or any(
        item not in (-1, 1) or isinstance(item, bool) for item in values
    ):
        _die("request expected helicity values are invalid")
    permutation = _require_list(
        expected.get("external_leg_permutation"),
        "request.expected.external_leg_permutation",
    )
    if (
        len(permutation) != EXTERNAL_LEG_COUNT
        or any(not _is_int(item) for item in permutation)
        or sorted(permutation) != list(range(EXTERNAL_LEG_COUNT))
    ):
        _die("request expected external-leg permutation is invalid")
    return expected


def _request_refs(
    request: Mapping[str, Any],
    request_dir: Path,
) -> tuple[dict[tuple[str, str], FileRef], dict[str, FileRef]]:
    captures = _require_mapping(request.get("captures"), "request.captures")
    if set(captures) != set(MODELS):
        _die("request.captures must contain exactly built-in-sm and ufo-sm")
    capture_refs: dict[tuple[str, str], FileRef] = {}
    for model in MODELS:
        layouts = _require_mapping(captures[model], f"request.captures.{model}")
        if set(layouts) != set(LAYOUTS):
            _die(f"request.captures.{model} must contain exactly both LC layouts")
        for layout in LAYOUTS:
            capture_refs[(model, layout)] = _file_ref(
                layouts[layout],
                base=request_dir,
                label=f"request.captures.{model}.{layout}",
            )
    amplicol = _require_mapping(
        request.get("amplicol_evidence"),
        "request.amplicol_evidence",
    )
    roles = {AMPLICOL_SELECTED_ROLE, AMPLICOL_UNION_ROLE}
    if set(amplicol) != roles:
        _die("request.amplicol_evidence must contain exactly both selector workloads")
    amplicol_refs = {
        role: _file_ref(
            amplicol[role],
            base=request_dir,
            label=f"request.amplicol_evidence.{role}",
        )
        for role in sorted(roles)
    }
    resolved_paths = [
        ref.path for ref in (*capture_refs.values(), *amplicol_refs.values())
    ]
    if len(set(resolved_paths)) != len(resolved_paths):
        _die("request evidence paths must be six distinct files")
    return capture_refs, amplicol_refs


def _cross_validate(
    captures: Mapping[tuple[str, str], Capture],
    amplicol: Mapping[str, AmpliColEvidence],
) -> dict[str, Any]:
    capture_values = list(captures.values())
    first = capture_values[0]
    for current in capture_values[1:]:
        if current.source_identity != first.source_identity:
            _die("pyAmpliCol source identities differ across captures")
        if current.runtime_identity != first.runtime_identity:
            _die("pyAmpliCol runtime identities differ across captures")
        if current.host != first.host:
            _die("host identities differ across pyAmpliCol captures")
        if current.fixture != first.fixture:
            _die("validation sample/momenta identities differ across captures")
        if current.normalization_sha256 != first.normalization_sha256:
            _die("normalization identities differ across captures")
        if (
            current.runtime_selector_semantics_sha256
            != first.runtime_selector_semantics_sha256
        ):
            _die("runtime-selector semantics differ across captures")
        if current.reduction_ordering_sha256 != first.reduction_ordering_sha256:
            _die("logical reduction ordering differs across captures")
        for axis_name in ("color_axis", "helicity_axis"):
            if getattr(current, axis_name) != getattr(first, axis_name):
                _die(f"ordered physical {axis_name} contracts differ across captures")
    for layout in LAYOUTS:
        if (
            captures[("built-in-sm", layout)].execution_schedule_ordering_sha256_by_mode
            != captures[("ufo-sm", layout)].execution_schedule_ordering_sha256_by_mode
        ):
            _die(f"execution schedule ordering differs across models for {layout}")
        _assert_values_close(
            captures[("built-in-sm", layout)].validation_values,
            captures[("ufo-sm", layout)].validation_values,
            f"built-in versus UFO pointwise values for {layout}",
        )
    for model in MODELS:
        topology = captures[(model, "topology-replay")]
        layout_union_capture = captures[(model, "all-flow-union")]
        if topology.model_common_sha256 != layout_union_capture.model_common_sha256:
            _die(f"{model} model identity differs between LC layouts")
        if (
            topology.generation_models_sha256
            != layout_union_capture.generation_models_sha256
        ):
            _die(f"{model} generation model identities differ between LC layouts")
    if (
        captures[("built-in-sm", "topology-replay")].model_common_sha256
        == captures[("ufo-sm", "topology-replay")].model_common_sha256
    ):
        _die("built-in and UFO captures unexpectedly share one model identity")
    compiled_recurrence = _compiled_recurrence_acceptance(captures)

    selected = amplicol[AMPLICOL_SELECTED_ROLE]
    union_evidence = amplicol[AMPLICOL_UNION_ROLE]
    if selected.source_identity != union_evidence.source_identity:
        _die("AmpliCol workloads use different source/compiler identities")
    if selected.host != first.host or union_evidence.host != first.host:
        _die("AmpliCol and pyAmpliCol captures do not share one host identity")
    if (
        selected.momenta_file_sha256 != first.fixture["file_sha256"]
        or union_evidence.momenta_file_sha256 != first.fixture["file_sha256"]
    ):
        _die("AmpliCol raw momenta file differs from the pyAmpliCol fixture")
    expected_color_axis = {
        "count": first.color_axis["count"],
        "ordered_ids_sha256": first.color_axis["ordered_ids_sha256"],
    }
    expected_helicity_axis = {
        "count": first.helicity_axis["count"],
        "ordered_ids_sha256": first.helicity_axis["ordered_ids_sha256"],
    }
    for evidence in (selected, union_evidence):
        if evidence.color_axis != expected_color_axis:
            _die(f"AmpliCol {evidence.role} color-flow axis differs from pyAmpliCol")
        if evidence.helicity_axis != expected_helicity_axis:
            _die(f"AmpliCol {evidence.role} helicity axis differs from pyAmpliCol")
    topology_selector = captures[("built-in-sm", "topology-replay")].selector
    union_selector = captures[("built-in-sm", "all-flow-union")].selector
    if (
        selected.selector["color_flow_request"]
        != topology_selector["color_flow_request"]
        or selected.selector["helicity_request"]
        != topology_selector["helicity_request"]
        or union_evidence.selector["color_flow_request"]
        != union_selector["color_flow_request"]
        or union_evidence.selector["helicity_request"]
        != union_selector["helicity_request"]
    ):
        _die("AmpliCol runtime selector requests differ from pyAmpliCol")
    if selected.interleave_group_sha256 != union_evidence.interleave_group_sha256:
        _die("AmpliCol workloads do not share one interleave schedule")
    combined_interleave = sorted(
        (*selected.interleave_records, *union_evidence.interleave_records),
        key=lambda record: record["position"],
    )
    if [record["position"] for record in combined_interleave] != list(
        range(len(combined_interleave))
    ):
        _die("AmpliCol interleave positions are incomplete or duplicated")
    previous_finish: dt.datetime | None = None
    command_hashes: set[str] = set()
    roles_by_round: dict[int, list[str]] = {}
    for expected_position, record in enumerate(combined_interleave):
        expected_role = (
            AMPLICOL_SELECTED_ROLE
            if expected_position % 2 == 0
            else AMPLICOL_UNION_ROLE
        )
        expected_round = expected_position // 2
        if (
            record["position"] != expected_position
            or record["role"] != expected_role
            or record["round"] != expected_round
        ):
            _die(
                "AmpliCol combined schedule is not paired "
                "selected/all-flow interleaving"
            )
        round_index = record["round"]
        assert isinstance(round_index, int)
        roles_by_round.setdefault(round_index, []).append(record["role"])
        started = _utc(record["started_at_utc"], "AmpliCol interleave start")
        finished = _utc(record["finished_at_utc"], "AmpliCol interleave finish")
        if previous_finish is not None and started < previous_finish:
            _die("AmpliCol workload subprocesses were not interleaved chronologically")
        previous_finish = finished
        command_sha = record["command_sha256"]
        if command_sha in command_hashes:
            _die("AmpliCol interleaved subprocess commands are duplicated")
        command_hashes.add(command_sha)
    if (
        len(roles_by_round) < MIN_SAMPLES
        or set(roles_by_round) != set(range(len(roles_by_round)))
        or any(
            roles
            != [
                AMPLICOL_SELECTED_ROLE,
                AMPLICOL_UNION_ROLE,
            ]
            for roles in roles_by_round.values()
        )
    ):
        _die("AmpliCol workloads lack seven paired interleaved rounds")
    expected_interleave_sha256 = _amplicol_interleave_group_sha256(combined_interleave)
    if selected.interleave_group_sha256 != expected_interleave_sha256:
        _die("AmpliCol interleave-group content digest is stale")
    _assert_values_close(
        captures[("built-in-sm", "topology-replay")].validation_values,
        selected.values,
        "pyAmpliCol versus AmpliCol selected-flow values",
    )
    _assert_values_close(
        captures[("built-in-sm", "all-flow-union")].validation_values,
        union_evidence.values,
        "pyAmpliCol versus AmpliCol all-flow values",
    )
    return {
        "point_count": first.fixture["point_count"],
        "pyamplicol_cross_model_and_layout_parity": True,
        "amplicol_selected_flow_parity": True,
        "amplicol_all_flow_parity": True,
        "relative_tolerance": RTOL,
        "absolute_tolerance": ATOL,
        "compiled_recurrence_runtime_ceiling": compiled_recurrence,
    }


def _input_identity(loaded: LoadedJson) -> dict[str, Any]:
    return {
        "path": str(loaded.ref.path),
        "size_bytes": loaded.ref.size_bytes,
        "raw_sha256": loaded.ref.sha256,
        "canonical_payload_sha256": loaded.canonical_sha256,
    }


def _compiled_recurrence_measurement(
    timings: Mapping[str, Any],
    *,
    mode: str,
    batch: int,
    label: str,
) -> dict[str, Any]:
    """Recompute one ceiling-gate measurement from its retained raw samples."""

    mode_timings = _require_mapping(timings.get(mode), f"{label}.{mode}")
    measurement = _require_mapping(
        mode_timings.get(str(batch)),
        f"{label}.{mode}.batch-{batch}",
    )
    samples = _require_list(
        measurement.get("raw_seconds_per_point"),
        f"{label}.{mode}.batch-{batch}.raw_seconds_per_point",
    )
    values: list[float] = []
    for index, sample in enumerate(samples):
        if not _is_number(sample) or float(sample) <= 0.0:
            _die(
                f"{label}.{mode}.batch-{batch} raw sample {index} "
                "must be a finite positive number"
            )
        values.append(float(sample))
    sample_count = _require_int(
        measurement.get("sample_count"),
        f"{label}.{mode}.batch-{batch}.sample_count",
        minimum=MIN_SAMPLES,
    )
    if sample_count != len(values):
        _die(f"{label}.{mode}.batch-{batch} raw sample count is inconsistent")
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    if measurement.get("median_seconds_per_point") != median:
        _die(f"{label}.{mode}.batch-{batch} retained median is stale")
    if measurement.get("mad_seconds_per_point") != mad:
        _die(f"{label}.{mode}.batch-{batch} retained MAD is stale")
    if measurement.get("timing_boundary") != "runtime_core_repeated_wall_time":
        _die(f"{label}.{mode}.batch-{batch} timing boundary is invalid")
    return {
        "sample_count": sample_count,
        "median_seconds_per_point": median,
        "mad_seconds_per_point": mad,
    }


def _compiled_recurrence_acceptance(
    captures: Mapping[tuple[str, str], Capture],
) -> dict[str, Any]:
    """Require compiled qq_Z6g to stay within 15% of recurrence."""

    cells: dict[str, Any] = {}
    for model in MODELS:
        model_cells: dict[str, Any] = {}
        for layout in LAYOUTS:
            label = f"{model}/{layout}"
            capture = captures.get((model, layout))
            if capture is None:
                _die(f"{label} capture is absent from the recurrence ceiling gate")
            batch_cells: dict[str, Any] = {}
            for batch in COMPILED_RECURRENCE_BATCHES:
                compiled = _compiled_recurrence_measurement(
                    capture.timings,
                    mode="compiled",
                    batch=batch,
                    label=label,
                )
                recurrence = _compiled_recurrence_measurement(
                    capture.timings,
                    mode="recurrence",
                    batch=batch,
                    label=label,
                )
                compiled_median = compiled["median_seconds_per_point"]
                recurrence_median = recurrence["median_seconds_per_point"]
                assert isinstance(compiled_median, float)
                assert isinstance(recurrence_median, float)
                ratio = compiled_median / recurrence_median
                passes = (
                    compiled_median <= recurrence_median * MAX_COMPILED_OVER_RECURRENCE
                )
                if not passes:
                    _die(
                        f"{label} compiled qq_Z6g exceeds the recurrence ceiling "
                        f"at batch {batch}: {ratio:.17g} > "
                        f"{MAX_COMPILED_OVER_RECURRENCE:.17g} "
                        f"(compiled median={compiled_median:.17g}, "
                        f"compiled raw MAD="
                        f"{compiled['mad_seconds_per_point']:.17g}, "
                        f"recurrence median={recurrence_median:.17g}, "
                        f"recurrence raw MAD="
                        f"{recurrence['mad_seconds_per_point']:.17g})"
                    )
                batch_cells[str(batch)] = {
                    "statistic": COMPILED_RECURRENCE_STATISTIC,
                    "compiled": compiled,
                    "recurrence": recurrence,
                    "compiled_over_recurrence": ratio,
                    "maximum_compiled_over_recurrence": (MAX_COMPILED_OVER_RECURRENCE),
                    "passes": True,
                }
            model_cells[layout] = batch_cells
        cells[model] = model_cells
    return {
        "statistic": COMPILED_RECURRENCE_STATISTIC,
        "required_batches": list(COMPILED_RECURRENCE_BATCHES),
        "maximum_compiled_over_recurrence": MAX_COMPILED_OVER_RECURRENCE,
        "cells": cells,
        "passes": True,
    }


def _accepted_manifest(
    *,
    request_loaded: LoadedJson,
    captures: Mapping[tuple[str, str], Capture],
    amplicol: Mapping[str, AmpliColEvidence],
    validation: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    capture_manifest: dict[str, Any] = {}
    timings: dict[str, Any] = {
        SELECTED_WORKLOAD: {},
        UNION_WORKLOAD: {},
    }
    for model in MODELS:
        capture_manifest[model] = {}
        for layout in LAYOUTS:
            capture = captures[(model, layout)]
            capture_manifest[model][layout] = {
                "input": _input_identity(capture.loaded),
                "recomputed_complete": True,
                "recomputed_passes": True,
                "resolved_selector": capture.selector,
                "model_common_physics_identity_sha256": capture.model_common_sha256,
                "generation_model_identities_sha256": (
                    capture.generation_models_sha256
                ),
            }
            workload = (
                UNION_WORKLOAD if layout == "all-flow-union" else SELECTED_WORKLOAD
            )
            timings[workload][model] = capture.timings
    external: dict[str, Any] = {}
    for role, evidence in amplicol.items():
        external[role] = {
            "input": _input_identity(evidence.loaded),
            "timing": evidence.timing,
        }
        workload = (
            SELECTED_WORKLOAD if role == AMPLICOL_SELECTED_ROLE else UNION_WORKLOAD
        )
        timings[workload]["amplicol"] = evidence.timing
    comparisons: dict[str, Any] = {}
    for workload in (SELECTED_WORKLOAD, UNION_WORKLOAD):
        amplicol_timing = timings[workload]["amplicol"]
        amplicol_median = amplicol_timing["median_seconds_per_point"]
        model_comparisons: dict[str, Any] = {}
        for model in MODELS:
            lane_comparisons: dict[str, Any] = {}
            for mode in MODES:
                batch_comparisons: dict[str, Any] = {}
                for batch in BATCHES:
                    py_timing = timings[workload][model][mode][str(batch)]
                    batch_comparisons[str(batch)] = {
                        "pyamplicol_seconds_per_point": (
                            py_timing["median_seconds_per_point"]
                        ),
                        "amplicol_seconds_per_point": amplicol_median,
                        "pyamplicol_over_amplicol": (
                            py_timing["median_seconds_per_point"] / amplicol_median
                        ),
                        "pyamplicol_boundary": (py_timing["timing_boundary"]),
                        "amplicol_boundary": (amplicol_timing["timing_boundary"]),
                        "amplicol_batch_semantics": (
                            amplicol_timing["batch_semantics"]
                        ),
                    }
                lane_comparisons[mode] = batch_comparisons
            model_comparisons[model] = lane_comparisons
        comparisons[workload] = model_comparisons
    first = captures[("built-in-sm", "topology-replay")]
    return {
        "kind": OUTPUT_KIND,
        "schema_version": OUTPUT_SCHEMA,
        "accepted": True,
        "status": "accepted",
        "errors": [],
        "policy": {
            "required_models": list(MODELS),
            "required_layouts": list(LAYOUTS),
            "required_modes": list(MODES),
            "required_batches": list(BATCHES),
            "minimum_interleaved_subprocess_samples": MIN_SAMPLES,
            "jit_optimization_level": 3,
            "generation_specialized_axes_allowed": False,
            "timing_statistics": "subprocess-median-and-raw-mad-v1",
            "relative_tolerance": RTOL,
            "absolute_tolerance": ATOL,
            "compiled_recurrence_statistic": COMPILED_RECURRENCE_STATISTIC,
            "compiled_recurrence_required_batches": list(COMPILED_RECURRENCE_BATCHES),
            "maximum_compiled_over_recurrence": MAX_COMPILED_OVER_RECURRENCE,
        },
        "request_identity": _input_identity(request_loaded),
        "common_contract": {
            "process": PROCESS,
            "source": first.source_identity,
            "runtime_provenance_sha256": _canonical_sha256(first.runtime_identity),
            "host_sha256": _canonical_sha256(first.host),
            "fixture": first.fixture,
            "normalization_sha256": first.normalization_sha256,
            "physical_color_flows_sha256": _canonical_sha256(first.color_axis),
            "physical_helicities_sha256": _canonical_sha256(first.helicity_axis),
            "runtime_selector_semantics_sha256": (
                first.runtime_selector_semantics_sha256
            ),
            "reduction_ordering_sha256": first.reduction_ordering_sha256,
            "execution_schedule_ordering_sha256_by_layout": {
                layout: captures[
                    (
                        "built-in-sm",
                        layout,
                    )
                ].execution_schedule_ordering_sha256_by_mode
                for layout in LAYOUTS
            },
            "color_flow": expected["color_flow"],
            "helicity": expected["helicity"],
        },
        "layout_captures": capture_manifest,
        "external_lanes": external,
        "validation": dict(validation),
        "timings": timings,
        "comparisons": comparisons,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


def _rejected_manifest(
    *,
    request_path: Path,
    request_sha256: str,
    error: str,
) -> dict[str, Any]:
    return {
        "kind": OUTPUT_KIND,
        "schema_version": OUTPUT_SCHEMA,
        "accepted": False,
        "status": "rejected",
        "errors": [error],
        "policy": {
            "required_models": list(MODELS),
            "required_layouts": list(LAYOUTS),
            "required_modes": list(MODES),
            "required_batches": list(BATCHES),
            "minimum_interleaved_subprocess_samples": MIN_SAMPLES,
            "jit_optimization_level": 3,
            "generation_specialized_axes_allowed": False,
            "compiled_recurrence_statistic": COMPILED_RECURRENCE_STATISTIC,
            "compiled_recurrence_required_batches": list(COMPILED_RECURRENCE_BATCHES),
            "maximum_compiled_over_recurrence": MAX_COMPILED_OVER_RECURRENCE,
        },
        "request_identity": {
            "path": str(request_path.resolve(strict=False)),
            "expected_raw_sha256": request_sha256,
        },
        "common_contract": None,
        "layout_captures": None,
        "external_lanes": None,
        "validation": None,
        "timings": None,
        "comparisons": None,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    addressed = dict(payload)
    addressed["content_sha256"] = _canonical_sha256(addressed)
    encoded = json.dumps(
        addressed,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(f"{encoded}\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return addressed


def combine(
    *,
    request_path: Path,
    request_sha256: str,
    output_path: Path,
) -> tuple[dict[str, Any], int]:
    """Validate all evidence, always emitting a content-addressed decision."""

    try:
        _require_sha256(request_sha256, "--request-sha256")
        request_ref = FileRef(
            path=request_path.resolve(strict=False),
            stated_path=str(request_path),
            size_bytes=request_path.stat().st_size,
            sha256=request_sha256,
        )
        request_loaded = _load_json_ref(request_ref, "request manifest")
        request = request_loaded.payload
        _require_exact_keys(request, _REQUEST_KEYS, "request manifest")
        if (
            request.get("kind") != REQUEST_KIND
            or request.get("schema_version") != REQUEST_SCHEMA
        ):
            _die("request manifest kind/schema is unsupported")
        expected = _validate_expected(request.get("expected"))
        capture_refs, amplicol_refs = _request_refs(
            request,
            request_path.resolve(strict=False).parent,
        )
        benchmark = _load_benchmark_module()
        captures = {
            key: _validate_capture(
                _load_json_ref(
                    ref,
                    f"{key[0]}/{key[1]} capture",
                ),
                model=key[0],
                layout=key[1],
                expected=expected,
                benchmark=benchmark,
            )
            for key, ref in capture_refs.items()
        }
        amplicol = {
            role: _validate_amplicol(
                _load_json_ref(ref, f"amplicol/{role} evidence"),
                role=role,
                expected=expected,
            )
            for role, ref in amplicol_refs.items()
        }
        validation = _cross_validate(captures, amplicol)
        payload = _accepted_manifest(
            request_loaded=request_loaded,
            captures=captures,
            amplicol=amplicol,
            validation=validation,
            expected=expected,
        )
        return _write_manifest(output_path, payload), 0
    except Exception as error:
        message = (
            str(error)
            if isinstance(error, (EvidenceError, OSError))
            else (
                "internal fail-closed validation error "
                f"({type(error).__name__}): {error}"
            )
        )
        payload = _rejected_manifest(
            request_path=request_path,
            request_sha256=request_sha256,
            error=message,
        )
        return _write_manifest(output_path, payload), 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--request",
        type=Path,
        required=True,
        help="strict content-addressed M0 orchestrator request JSON",
    )
    result.add_argument(
        "--request-sha256",
        required=True,
        help="expected raw SHA-256 of --request",
    )
    result.add_argument(
        "--output",
        type=Path,
        required=True,
        help="accepted/rejected content-addressed decision JSON",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    manifest, exit_code = combine(
        request_path=arguments.request,
        request_sha256=arguments.request_sha256,
        output_path=arguments.output,
    )
    print(
        json.dumps(
            {
                "accepted": manifest["accepted"],
                "status": manifest["status"],
                "content_sha256": manifest["content_sha256"],
                "output": str(arguments.output),
                "errors": manifest["errors"],
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
