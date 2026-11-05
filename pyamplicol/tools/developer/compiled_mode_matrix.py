#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run and audit the frozen 168-cell Direct-Arena acceptance matrix.

Each cell is measured by ``compiled_mode_regression.py``.  This outer driver
adds the acceptance rules that cannot be established by one cell in isolation:
exact matrix completeness, stable runtime/provenance identity, a measured
10% gain beyond noise in each execution mode, and a generation-time geometric
mean no more than 5% above the frozen baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.developer import compiled_mode_regression as regression  # noqa: E402

CELL_DRIVER = ROOT / "tools" / "developer" / "compiled_mode_regression.py"
RESULT_KIND = "pyamplicol-compiled-mode-acceptance-matrix"
SCHEMA_VERSION = 2
MATRIX_CONTRACT = "eager-and-compiled-arena-plan-168-v2"
BATCH_SIZES = (1, 128, 1024)
EXECUTION_MODES = ("eager", "compiled")
GENERATION_GEOMETRIC_MEAN_MAXIMUM = 1.05
DEFAULT_MATRIX_GENERATION_TIMEOUT = 2400.0
DEFAULT_MATRIX_PROFILE_TIMEOUT = 1200.0
ACCEPTANCE_SAMPLE_COUNT = regression.DEFAULT_SAMPLE_COUNT
ACCEPTANCE_TARGET_RUNTIME_SECONDS = regression.DEFAULT_TARGET_RUNTIME
ACCEPTANCE_MINIMUM_TIMED_BLOCKS = regression.DEFAULT_SAMPLE_COUNT
ACCEPTANCE_WARMUP_RUNS = regression.DEFAULT_WARMUP_RUNS
FROZEN_BASELINE_SOURCE_REVISION = "443f354a467cdda187996bef1a41fbd5a00ae28d"
FROZEN_BASELINE_NATIVE_INPUTS_SHA256 = (
    "f91ebcc3eb431e3e1e72ac8a4e02dea194c17c2118f57a2742d0c8c5a73b3088"
)
FROZEN_BASELINE_DISTRIBUTION_SHA256 = (
    "c615b3753eeedc427a03ff063804926be4d77994d4ab14579c8256dbd390d003"
)
FROZEN_BASELINE_NATIVE_MODULE_SHA256 = (
    "055d28af8101603c3d0431fa6bfd288d9fb62fe8434201404a8c709f261c7349"
)
FROZEN_BASELINE_BINARY_IDENTITIES = {
    ("Darwin", "arm64"): {
        "distribution_content_sha256": FROZEN_BASELINE_DISTRIBUTION_SHA256,
        "native_module_sha256": FROZEN_BASELINE_NATIVE_MODULE_SHA256,
    }
}
_HEX_SHA256_LENGTH = 64
_HEX_GIT_SHA_LENGTH = 40


class MatrixError(RuntimeError):
    """Raised when the matrix cannot be run or audited."""


@dataclass(frozen=True, slots=True)
class MatrixCell:
    category: str
    process_key: str
    process: str
    model_kind: str
    execution_mode: str
    workload_key: str
    workload: str
    color_accuracy: str
    lc_flow_layout: str
    jit_optimization_level: int
    batch_size: int
    helicities: tuple[str, ...] = ()
    color_flows: tuple[str, ...] = ()

    @property
    def cell_id(self) -> str:
        return "__".join(
            (
                self.category,
                self.process_key,
                self.model_kind,
                self.execution_mode,
                self.workload_key,
                f"b{self.batch_size}",
            )
        )

    @property
    def artifact_group_id(self) -> str:
        """Identify the generation inputs shared across timing-only batches."""

        return "__".join(
            (
                self.category,
                self.process_key,
                self.model_kind,
                self.execution_mode,
                f"o{self.jit_optimization_level}",
                self.color_accuracy,
                self.lc_flow_layout,
            )
        )


LC_WORKLOADS = (
    {
        "workload_key": "lc-topology",
        "workload": "single-flow-helicity-sum",
        "color_accuracy": "lc",
        "lc_flow_layout": "topology-replay",
        "helicities": (),
        "color_flows": ("1",),
    },
    {
        "workload_key": "lc-union",
        "workload": "all-flow-single-helicity",
        "color_accuracy": "lc",
        "lc_flow_layout": "all-flow-union",
        "helicities": ("1",),
        "color_flows": (),
    },
)
PRIMARY_LC_WORKLOADS = (
    {
        "workload_key": "lc-topology",
        "workload": "single-flow-helicity-sum",
        "color_accuracy": "lc",
        "lc_flow_layout": "topology-replay",
        "helicities": (),
        "color_flows": ("flow:2,4,5,6,7,8,9,1",),
    },
    {
        "workload_key": "lc-union",
        "workload": "all-flow-single-helicity",
        "color_accuracy": "lc",
        "lc_flow_layout": "all-flow-union",
        "helicities": ("h:-1,+1,-1,+1,-1,+1,-1,+1,-1",),
        "color_flows": (),
    },
)
SUMMED_WORKLOADS = (
    {
        "workload_key": "nlc-summed",
        "workload": "summed",
        "color_accuracy": "nlc",
        "lc_flow_layout": "topology-replay",
        "helicities": (),
        "color_flows": (),
    },
    {
        "workload_key": "full-summed",
        "workload": "summed",
        "color_accuracy": "full",
        "lc_flow_layout": "topology-replay",
        "helicities": (),
        "color_flows": (),
    },
)
MEDIUM_PROCESSES = (
    ("dd_z_3g", "d d~ > z g g g"),
    ("gg_tt_2g", "g g > t t~ g g"),
    ("gg_4g", "g g > g g g g"),
    ("dd_3q_1g", "d d~ > u u~ s s~ g"),
    ("dd_tt_3g", "d d~ > t t~ g g g"),
)
COLOR_HEAVY_PROCESSES = (
    ("gg_tt_3g", "g g > t t~ g g g"),
    ("gg_tt_4g", "g g > t t~ g g g g"),
)


def _cells_for(
    *,
    category: str,
    processes: Sequence[tuple[str, str]],
    model_kinds: Sequence[str],
    workloads: Sequence[Mapping[str, object]],
) -> list[MatrixCell]:
    return [
        MatrixCell(
            category=category,
            process_key=process_key,
            process=process,
            model_kind=model_kind,
            execution_mode=execution_mode,
            jit_optimization_level=2 if execution_mode == "eager" else 3,
            batch_size=batch_size,
            **workload,
        )
        for process_key, process in processes
        for model_kind in model_kinds
        for execution_mode in EXECUTION_MODES
        for workload in workloads
        for batch_size in BATCH_SIZES
    ]


def canonical_cells() -> tuple[MatrixCell, ...]:
    """Return the exact matrix documented in the Arena implementation plan."""

    cells = [
        *_cells_for(
            category="primary",
            processes=(("uu_z_6g", "u u~ > z g g g g g g"),),
            model_kinds=("built-in", "ufo-sm"),
            workloads=PRIMARY_LC_WORKLOADS,
        ),
        *_cells_for(
            category="medium",
            processes=MEDIUM_PROCESSES,
            model_kinds=("built-in",),
            workloads=(*LC_WORKLOADS, *SUMMED_WORKLOADS),
        ),
        *_cells_for(
            category="color-heavy",
            processes=COLOR_HEAVY_PROCESSES,
            model_kinds=("built-in",),
            workloads=SUMMED_WORKLOADS,
        ),
    ]
    if len(cells) != 168 or len({cell.cell_id for cell in cells}) != 168:
        raise AssertionError("the frozen Direct-Arena matrix must contain 168 cells")
    return tuple(cells)


CANONICAL_CELLS = canonical_cells()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MatrixError(f"cannot read matrix cell result: {path}") from error
    if not isinstance(payload, dict):
        raise MatrixError(f"matrix cell result is not an object: {path}")
    return payload


def _model(cell: MatrixCell, ufo_sm_model: Path) -> tuple[str, str]:
    if cell.model_kind == "built-in":
        return "built-in-sm", "built-in"
    return str(_absolute(ufo_sm_model)), "ufo-sm"


def cell_command(
    cell: MatrixCell,
    *,
    baseline_python: Path,
    current_python: Path,
    output_root: Path,
    ufo_sm_model: Path,
    baseline_dependency_site: Path | None = None,
    current_dependency_site: Path | None = None,
    samples: int = regression.DEFAULT_SAMPLE_COUNT,
    target_runtime: float = regression.DEFAULT_TARGET_RUNTIME,
    minimum_samples: int = regression.DEFAULT_SAMPLE_COUNT,
    warmup_runs: int = regression.DEFAULT_WARMUP_RUNS,
    generation_timeout: float = DEFAULT_MATRIX_GENERATION_TIMEOUT,
    profile_timeout: float = DEFAULT_MATRIX_PROFILE_TIMEOUT,
    regenerate_artifacts: bool = False,
) -> tuple[str, ...]:
    """Build the one-cell command without weakening the cell driver's gates."""

    model, model_label = _model(cell, ufo_sm_model)
    command = [
        str(_absolute(current_python)),
        str(CELL_DRIVER),
        "--baseline-python",
        str(_absolute(baseline_python)),
        "--current-python",
        str(_absolute(current_python)),
        "--output-root",
        str(_absolute(output_root) / "artifact-groups" / cell.artifact_group_id),
        "--result-path",
        str(_absolute(output_root) / "cells" / cell.cell_id / "result.json"),
        "--process",
        cell.process,
        "--model",
        model,
        "--model-label",
        model_label,
        "--execution-mode",
        cell.execution_mode,
        "--jit-optimization-level",
        str(cell.jit_optimization_level),
        "--workload",
        cell.workload,
        "--color",
        cell.color_accuracy,
        "--lc-flow-layout",
        cell.lc_flow_layout,
        "--batch-size",
        str(cell.batch_size),
        "--samples",
        str(samples),
        "--target-runtime",
        str(target_runtime),
        "--minimum-samples",
        str(minimum_samples),
        "--warmup-runs",
        str(warmup_runs),
        "--generation-timeout",
        str(generation_timeout),
        "--profile-timeout",
        str(profile_timeout),
    ]
    for helicity in cell.helicities:
        command.extend(("--helicity", helicity))
    for color_flow in cell.color_flows:
        command.extend(("--color-flow", color_flow))
    for option, site in (
        ("--baseline-dependency-site", baseline_dependency_site),
        ("--current-dependency-site", current_dependency_site),
    ):
        if site is not None:
            command.extend((option, str(_absolute(site))))
    if regenerate_artifacts:
        command.append("--regenerate-artifacts")
    return tuple(command)


def _finite_positive(value: object) -> float | None:
    if (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    ):
        return float(value)
    return None


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _HEX_SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _HEX_GIT_SHA_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _stable_file_identity_value(value: object) -> object:
    """Remove observational mtimes but retain content and path identities.

    Git checkout and artifact-extraction mtimes vary between independent x86
    shard runners.  Resolved paths, sizes, and SHA-256 digests remain part of
    the identity.  This projection is deliberately limited to ``mtime_ns``;
    it does not rewrite paths or any digest-covered runtime object.
    """

    if isinstance(value, Mapping):
        return {
            key: _stable_file_identity_value(item)
            for key, item in value.items()
            if key != "mtime_ns"
        }
    if isinstance(value, list):
        return [_stable_file_identity_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_stable_file_identity_value(item) for item in value)
    return value


def _stable_runtime_identity_value(
    value: object,
    *,
    _root: bool = True,
) -> object:
    """Project a runtime identity onto cross-runner immutable observations.

    The top-level kernel/platform string and filesystem mtimes are host
    observations.  Fixed paths, Python/package versions, sizes, every digest,
    build provenance, native inputs, and dependency identities remain exact.
    """

    if isinstance(value, Mapping):
        return {
            key: _stable_runtime_identity_value(item, _root=False)
            for key, item in value.items()
            if key != "mtime_ns" and not (_root and key == "platform")
        }
    if isinstance(value, list):
        return [_stable_runtime_identity_value(item, _root=False) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _stable_runtime_identity_value(item, _root=False) for item in value
        )
    return value


def _installation_identity_errors(
    identity: object,
    *,
    expected: Mapping[str, str] | None,
    label: str,
) -> list[str]:
    errors: list[str] = []
    record = _mapping(identity)
    if record is None:
        return [f"{label} installation identity is absent"]
    if record.get("kind") != regression.INSTALLATION_IDENTITY_KIND:
        errors.append(f"{label} installation identity kind is invalid")
    if record.get("schema_version") != regression.INSTALLATION_IDENTITY_SCHEMA_VERSION:
        errors.append(f"{label} installation identity schema is invalid")
    distribution = _mapping(record.get("distribution_content"))
    distribution_sha = distribution.get("sha256") if distribution is not None else None
    native_modules = record.get("native_modules")
    native_module = (
        _mapping(native_modules[0])
        if isinstance(native_modules, Sequence)
        and not isinstance(native_modules, (str, bytes))
        and len(native_modules) == 1
        else None
    )
    native_sha = native_module.get("sha256") if native_module is not None else None
    build_info_files = record.get("build_info_files")
    build_info = (
        _mapping(build_info_files[0])
        if isinstance(build_info_files, Sequence)
        and not isinstance(build_info_files, (str, bytes))
        and len(build_info_files) == 1
        else None
    )
    payload = _mapping(build_info.get("payload")) if build_info is not None else None
    if expected is None:
        return [*errors, f"{label} expected build identity is absent"]
    expected_distribution = expected.get("distribution_content_sha256")
    expected_native = expected.get("native_module_sha256")
    expected_revision = expected.get("source_revision")
    expected_native_inputs = expected.get("native_build_inputs_sha256")
    if (
        not _is_sha256(expected_distribution)
        or distribution_sha != expected_distribution
    ):
        errors.append(f"{label} distribution content does not match its pin")
    if not _is_sha256(expected_native) or native_sha != expected_native:
        errors.append(f"{label} native module does not match its pin")
    if payload is None:
        errors.append(f"{label} installed build provenance is absent")
    else:
        if (
            not _is_git_sha(expected_revision)
            or payload.get("source_revision") != expected_revision
        ):
            errors.append(f"{label} source revision does not match its pin")
        if (
            not _is_sha256(expected_native_inputs)
            or payload.get("native_build_inputs_sha256") != expected_native_inputs
        ):
            errors.append(f"{label} native build inputs do not match their pin")
        if payload.get("publishable") is not False:
            errors.append(f"{label} build is not a non-publishable candidate")
        bootstrap = payload.get("selftest_fixture_bootstrap")
        if (label == "current" and bootstrap is not False) or (
            label != "current" and bootstrap is not None and bootstrap is not False
        ):
            errors.append(f"{label} build is a self-test bootstrap wheel")
    return errors


def _cell_evidence(
    cell: MatrixCell,
    result: Mapping[str, Any],
    *,
    baseline_python: Path,
    current_python: Path,
    ufo_sm_model: Path,
    output_root: Path,
    expected_platform: str,
    expected_builds: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    errors: list[str] = []
    model, model_label = _model(cell, ufo_sm_model)
    expected_configuration = {
        "baseline_python": str(_absolute(baseline_python)),
        "current_python": str(_absolute(current_python)),
        "output_root": str(
            _absolute(output_root) / "artifact-groups" / cell.artifact_group_id
        ),
        "process": cell.process,
        "model": model,
        "model_label": model_label,
        "execution_mode": cell.execution_mode,
        "workload": cell.workload,
        "jit_optimization_level": cell.jit_optimization_level,
        "color_accuracy": cell.color_accuracy,
        "lc_flow_layout": cell.lc_flow_layout,
        "shared_artifact": None,
        "batch_size": cell.batch_size,
        "helicities": list(cell.helicities),
        "color_flows": list(cell.color_flows),
    }
    if result.get("kind") != regression.RESULT_KIND:
        errors.append("unexpected result kind")
    if result.get("schema_version") != regression.SCHEMA_VERSION:
        errors.append("unexpected result schema")
    if result.get("platform") != expected_platform:
        errors.append("result platform does not match the acceptance host")
    for field in ("complete", "performance_result_authoritative", "passes"):
        if result.get(field) is not True:
            errors.append(f"{field} is not true")
    configuration = _mapping(result.get("configuration"))
    if configuration is None:
        errors.append("configuration is absent")
    else:
        for key, expected in expected_configuration.items():
            if configuration.get(key) != expected:
                errors.append(f"configuration.{key} does not match the matrix")
        methodology = {
            "independent_samples_per_lane": ACCEPTANCE_SAMPLE_COUNT,
            "target_runtime_per_native_sample_seconds": (
                ACCEPTANCE_TARGET_RUNTIME_SECONDS
            ),
            "minimum_native_timed_blocks_per_profile": (
                ACCEPTANCE_MINIMUM_TIMED_BLOCKS
            ),
            "warmup_runs_per_profile": ACCEPTANCE_WARMUP_RUNS,
            "native_wall_time_source": regression.NATIVE_WALL_TIME_SOURCE,
            "native_wall_time_sample_pass": regression.NATIVE_WALL_TIME_SAMPLE_PASS,
            "required_current_profile_attribution_sample_pass": (
                regression.PROFILE_ATTRIBUTION_SAMPLE_PASS
            ),
            "required_current_profile_attribution_boundary": (
                regression.ARENA_PROFILE_BOUNDARY
            ),
            "required_current_profile_attribution_phase_timing_scope": (
                regression.ARENA_PHASE_TIMING_SCOPE
            ),
            "required_current_profile_attribution_evaluator_timing_available": False,
            "required_current_profile_protocol": regression.ARENA_PROFILE_PROTOCOL,
            "timing_sample_contract": regression.PAIRED_TIMING_SAMPLE_CONTRACT,
        }
        numeric_minimums = {
            "independent_samples_per_lane",
            "minimum_native_timed_blocks_per_profile",
            "warmup_runs_per_profile",
            "target_runtime_per_native_sample_seconds",
        }
        for key, minimum_or_expected in methodology.items():
            observed = configuration.get(key)
            if key in numeric_minimums:
                if (
                    isinstance(observed, bool)
                    or not isinstance(observed, (int, float))
                    or float(observed) < float(minimum_or_expected)
                ):
                    errors.append(
                        f"configuration.{key} is below the acceptance minimum"
                    )
            elif observed != minimum_or_expected:
                errors.append(f"configuration.{key} violates the timing contract")
    for gate_name in (
        "gate",
        "correctness_gate",
        "arena_profile_gate",
        "resource_gate",
    ):
        gate = _mapping(result.get(gate_name))
        if gate is None or gate.get("passes") is not True:
            errors.append(f"{gate_name}.passes is not true")

    gain = _mapping(result.get("gain_gate"))
    gain_passes = False
    relative_gain: float | None = None
    if gain is None:
        errors.append("gain_gate is absent")
    else:
        raw_relative_gain = gain.get("relative_gain")
        if (
            isinstance(raw_relative_gain, (float, int))
            and not isinstance(raw_relative_gain, bool)
            and math.isfinite(float(raw_relative_gain))
        ):
            relative_gain = float(raw_relative_gain)
        else:
            errors.append("gain_gate.relative_gain is invalid")
        gain_passes = (
            gain.get("passes") is True
            and gain.get("at_least_ten_percent") is True
            and gain.get("beyond_measurement_noise") is True
            and relative_gain is not None
            and relative_gain >= regression.GAIN_RELATIVE_THRESHOLD
        )

    resource_gate = _mapping(result.get("resource_gate"))
    generation = (
        _mapping(resource_gate.get("generation")) if resource_gate is not None else None
    )
    generation_ratio = (
        _finite_positive(generation.get("current_over_baseline"))
        if generation is not None
        else None
    )
    if generation_ratio is None:
        errors.append("generation ratio is absent or invalid")
    elif generation_ratio > 1.10 or generation.get("passes") is not True:
        errors.append("per-cell generation ratio gate failed")

    artifacts = _mapping(result.get("artifacts"))
    artifact_digests: dict[str, str] = {}
    if artifacts is None:
        errors.append("artifacts are absent")
    else:
        for lane in ("baseline", "current"):
            artifact = _mapping(artifacts.get(lane))
            expected_path = (
                _absolute(output_root)
                / "artifact-groups"
                / cell.artifact_group_id
                / lane
                / "artifact"
            )
            if artifact is None:
                errors.append(f"{lane} artifact identity is absent")
                continue
            if artifact.get("path") != str(expected_path):
                errors.append(f"{lane} artifact path does not match the matrix")
            errors.extend(
                _installation_identity_errors(
                    artifact.get("installation_identity"),
                    expected=expected_builds.get(lane),
                    label=lane,
                )
            )
            artifact_digests[lane] = _canonical_sha256(
                {
                    key: artifact.get(key)
                    for key in (
                        "path",
                        "artifact_id",
                        "manifest_sha256",
                        "tree_identity",
                        "payload_digests",
                        "installation_identity",
                    )
                }
            )

    native_hashes = _mapping(result.get("native_module_sha256_by_lane"))
    runtime_digests: dict[str, str] = {}
    measurement_counts: Counter[str] = Counter()
    measurement_runtime_digests: defaultdict[str, set[str]] = defaultdict(set)
    selected_nonzero_by_lane = {"baseline": False, "current": False}
    measurements = result.get("measurements")
    if not isinstance(measurements, list):
        errors.append("measurements are absent")
        measurements = []
    for measurement in measurements:
        if not isinstance(measurement, Mapping):
            errors.append("measurement is not an object")
            continue
        lane = measurement.get("lane")
        runtime_identity = measurement.get("runtime_identity")
        if lane not in {"baseline", "current"} or not isinstance(
            runtime_identity, Mapping
        ):
            errors.append("measurement lane/runtime identity is invalid")
            continue
        measurement_counts[str(lane)] += 1
        measurement_runtime_digests[str(lane)].add(
            _canonical_sha256(_stable_runtime_identity_value(runtime_identity))
        )
        numerical = _mapping(measurement.get("warmed_numerical_result"))
        numerical_values = (
            numerical.get("values_f64") if numerical is not None else None
        )
        if (
            isinstance(numerical_values, Sequence)
            and not isinstance(numerical_values, (str, bytes))
            and any(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) != 0.0
                for value in numerical_values
            )
        ):
            selected_nonzero_by_lane[str(lane)] = True
        native = _mapping(runtime_identity.get("native_module"))
        if (
            native_hashes is None
            or native is None
            or native.get("sha256") != native_hashes.get(lane)
        ):
            errors.append(f"{lane} runtime/native-module identity mismatch")
        expected_build = expected_builds.get(str(lane))
        build_info = _mapping(runtime_identity.get("build_info"))
        build_payload = (
            _mapping(build_info.get("payload")) if build_info is not None else None
        )
        if expected_build is None or build_payload is None:
            errors.append(f"{lane} runtime has no pinned build identity")
        else:
            expected_revision = expected_build.get("source_revision")
            expected_native_inputs = expected_build.get("native_build_inputs_sha256")
            expected_native_module = expected_build.get("native_module_sha256")
            if (
                not _is_git_sha(expected_revision)
                or build_payload.get("source_revision") != expected_revision
            ):
                errors.append(f"{lane} source revision does not match its pin")
            if (
                not _is_sha256(expected_native_inputs)
                or build_payload.get("native_build_inputs_sha256")
                != expected_native_inputs
                or runtime_identity.get("native_build_inputs_sha256")
                != expected_native_inputs
            ):
                errors.append(f"{lane} native build inputs do not match their pin")
            if build_payload.get("publishable") is not False:
                errors.append(f"{lane} build is not a non-publishable candidate")
            bootstrap = build_payload.get("selftest_fixture_bootstrap")
            if (lane == "current" and bootstrap is not False) or (
                lane != "current" and bootstrap is not None and bootstrap is not False
            ):
                errors.append(f"{lane} build is a self-test bootstrap wheel")
            if native is None or native.get("sha256") != expected_native_module:
                errors.append(f"{lane} native module does not match its pin")
    expected_samples = (
        configuration.get("independent_samples_per_lane")
        if configuration is not None
        else None
    )
    expected_pair_orders = (
        [
            ["baseline", "current"] if pair_index % 2 == 0 else ["current", "baseline"]
            for pair_index in range(expected_samples)
        ]
        if isinstance(expected_samples, int)
        and not isinstance(expected_samples, bool)
        and expected_samples >= ACCEPTANCE_SAMPLE_COUNT
        else None
    )
    if result.get("pair_orders") != expected_pair_orders:
        errors.append("paired subprocess order is incomplete or not interleaved")
    for lane in ("baseline", "current"):
        native_sha = native_hashes.get(lane) if native_hashes is not None else None
        if not _is_sha256(native_sha):
            errors.append(f"{lane} native-module SHA-256 is invalid")
        expected_native_sha = expected_builds.get(lane, {}).get("native_module_sha256")
        if native_sha != expected_native_sha:
            errors.append(f"{lane} native-module SHA-256 does not match its pin")
        if (
            not isinstance(expected_samples, int)
            or isinstance(expected_samples, bool)
            or expected_samples < 5
            or measurement_counts[lane] != expected_samples
        ):
            errors.append(f"{lane} measurement coverage is incomplete")
        digests = measurement_runtime_digests[lane]
        if len(digests) != 1:
            errors.append(f"{lane} runtime identity is not stable within the cell")
        else:
            runtime_digests[lane] = next(iter(digests))
        if cell.workload != "summed" and not selected_nonzero_by_lane[lane]:
            errors.append(f"{lane} selected workload is identically zero")

    if artifacts is not None:
        try:
            recomputed_arena_profile_gate = regression._arena_profile_gate(
                measurements,
                execution_mode=cell.execution_mode,
                artifacts=artifacts,
                minimum_profile_timed_blocks=ACCEPTANCE_MINIMUM_TIMED_BLOCKS,
            )
        except regression.RegressionError as error:
            errors.append(f"arena profile evidence cannot be recomputed: {error}")
        else:
            if recomputed_arena_profile_gate.get("passes") is not True:
                errors.append(
                    "arena profile evidence fails independent matrix recomputation"
                )

    provenance = _mapping(result.get("provenance"))
    provenance_digests: dict[str, str] = {}
    if provenance is None:
        errors.append("provenance is absent")
    else:
        current_tool_paths = {
            "driver": CELL_DRIVER,
            "watchdog": regression.WATCHDOG,
            "native_sample_helper": regression.NATIVE_SAMPLE_HELPER,
            "dependency_entry": regression.DEPENDENCY_ENTRY,
        }
        for key in (
            "driver",
            "watchdog",
            "native_sample_helper",
            "dependency_entry",
            "interpreters",
            "dependency_sites",
            "model",
        ):
            value = provenance.get(key)
            if value is None:
                errors.append(f"provenance.{key} is absent")
            else:
                provenance_digests[key] = _canonical_sha256(
                    _stable_file_identity_value(value)
                )
                if key in current_tool_paths:
                    record = _mapping(value)
                    expected_sha = _sha256_file(current_tool_paths[key])
                    if record is None or record.get("sha256") != expected_sha:
                        errors.append(
                            f"provenance.{key} does not match the current source"
                        )
        if configuration is not None and configuration.get(
            "dependency_sites"
        ) != provenance.get("dependency_sites"):
            errors.append("configuration/provenance dependency identities differ")
    return {
        "cell_id": cell.cell_id,
        "configuration": asdict(cell),
        "result_content_sha256": _canonical_sha256(result),
        "errors": errors,
        "passes": not errors,
        "gain_gate_passes": gain_passes,
        "relative_gain": relative_gain,
        "generation_current_over_baseline": generation_ratio,
        "runtime_identity_sha256_by_lane": runtime_digests,
        "native_module_sha256_by_lane": (
            dict(native_hashes) if native_hashes is not None else {}
        ),
        "artifact_identity_sha256_by_lane": artifact_digests,
        "provenance_sha256": provenance_digests,
    }


def audit_results(
    results: Mapping[str, Mapping[str, Any]],
    *,
    baseline_python: Path,
    current_python: Path,
    ufo_sm_model: Path,
    output_root: Path,
    expected_platform: str,
    expected_builds: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Audit exactly the frozen cell set and return a machine-readable verdict."""

    expected = {cell.cell_id: cell for cell in CANONICAL_CELLS}
    observed_ids = set(results)
    missing = sorted(set(expected) - observed_ids)
    unexpected = sorted(observed_ids - set(expected))
    evidence = [
        _cell_evidence(
            expected[cell_id],
            results[cell_id],
            baseline_python=baseline_python,
            current_python=current_python,
            ufo_sm_model=ufo_sm_model,
            output_root=output_root,
            expected_platform=expected_platform,
            expected_builds=expected_builds,
        )
        for cell_id in sorted(set(expected) & observed_ids)
    ]
    return audit_cell_evidence(
        evidence,
        missing=missing,
        unexpected=unexpected,
        expected_builds=expected_builds,
    )


def audit_cell_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    missing: Sequence[str],
    unexpected: Sequence[str],
    expected_builds: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Aggregate already authenticated cell evidence for the frozen matrix.

    The regular single-host driver builds this evidence directly above.  The
    x86 shard aggregator uses the same function after each shard has validated
    its own host-local paths and artifact trees.  Keeping the global gates in
    one function prevents the sharded path from weakening the 168-cell
    acceptance contract.
    """

    identity_sets: defaultdict[str, set[str]] = defaultdict(set)
    for record in evidence:
        for lane, digest in record["runtime_identity_sha256_by_lane"].items():
            identity_sets[f"runtime:{lane}"].add(digest)
        for lane, digest in record["native_module_sha256_by_lane"].items():
            if _is_sha256(digest):
                identity_sets[f"native:{lane}"].add(digest)
        artifact_group = MatrixCell(**record["configuration"]).artifact_group_id
        for lane, digest in record["artifact_identity_sha256_by_lane"].items():
            identity_sets[f"artifact:{artifact_group}:{lane}"].add(digest)
        provenance = record["provenance_sha256"]
        for key in (
            "driver",
            "watchdog",
            "native_sample_helper",
            "dependency_entry",
            "interpreters",
            "dependency_sites",
        ):
            if key in provenance:
                identity_sets[f"provenance:{key}"].add(provenance[key])
        if "model" in provenance:
            model_kind = record["configuration"]["model_kind"]
            identity_sets[f"model:{model_kind}"].add(provenance["model"])
    required_identity_keys = (
        *(f"runtime:{lane}" for lane in ("baseline", "current")),
        *(f"native:{lane}" for lane in ("baseline", "current")),
        *(
            f"provenance:{key}"
            for key in (
                "driver",
                "watchdog",
                "native_sample_helper",
                "dependency_entry",
                "interpreters",
                "dependency_sites",
            )
        ),
        "model:built-in",
        "model:ufo-sm",
    )
    identity_failures = {
        key: sorted(identity_sets[key])
        for key in required_identity_keys
        if len(identity_sets[key]) != 1
    }
    for key, values in identity_sets.items():
        if key.startswith("artifact:") and len(values) != 1:
            identity_failures[key] = sorted(values)

    gain_by_mode = {
        mode: [
            record["cell_id"]
            for record in evidence
            if record["configuration"]["execution_mode"] == mode
            and record["gain_gate_passes"]
        ]
        for mode in EXECUTION_MODES
    }
    primary_gain_by_mode = {
        mode: [
            record["cell_id"]
            for record in evidence
            if record["configuration"]["category"] == "primary"
            and record["configuration"]["execution_mode"] == mode
            and record["gain_gate_passes"]
        ]
        for mode in EXECUTION_MODES
    }
    ratios = [
        record["generation_current_over_baseline"]
        for record in evidence
        if record["generation_current_over_baseline"] is not None
    ]
    geometric_mean = (
        math.exp(math.fsum(math.log(ratio) for ratio in ratios) / len(ratios))
        if ratios
        else None
    )
    cell_failures = {
        record["cell_id"]: record["errors"] for record in evidence if record["errors"]
    }
    completeness_passes = (
        not missing and not unexpected and len(evidence) == len(CANONICAL_CELLS)
    )
    gain_passes = all(primary_gain_by_mode[mode] for mode in EXECUTION_MODES)
    generation_passes = (
        len(ratios) == len(CANONICAL_CELLS)
        and geometric_mean is not None
        and geometric_mean <= GENERATION_GEOMETRIC_MEAN_MAXIMUM
    )
    passes = (
        completeness_passes
        and not cell_failures
        and not identity_failures
        and gain_passes
        and generation_passes
    )
    return {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "matrix_contract": MATRIX_CONTRACT,
        "matrix_definition": {
            "sha256": _canonical_sha256(
                [asdict(cell) | {"cell_id": cell.cell_id} for cell in CANONICAL_CELLS]
            ),
            "category_counts": dict(
                sorted(Counter(cell.category for cell in CANONICAL_CELLS).items())
            ),
            "execution_mode_counts": dict(
                sorted(Counter(cell.execution_mode for cell in CANONICAL_CELLS).items())
            ),
            "expected_cell_ids": [cell.cell_id for cell in CANONICAL_CELLS],
        },
        "expected_builds": {
            lane: dict(identity) for lane, identity in sorted(expected_builds.items())
        },
        "complete": completeness_passes,
        "passes": passes,
        "coverage": {
            "expected": len(CANONICAL_CELLS),
            "observed": len(evidence),
            "missing": missing,
            "unexpected": unexpected,
            "passes": completeness_passes,
        },
        "cell_gate": {
            "failure_count": len(cell_failures),
            "failures": cell_failures,
            "passes": not cell_failures and completeness_passes,
        },
        "identity_gate": {
            "distinct_sha256": {
                key: sorted(value) for key, value in sorted(identity_sets.items())
            },
            "failures": identity_failures,
            "passes": not identity_failures and completeness_passes,
        },
        "gain_gate": {
            "required_relative_gain": regression.GAIN_RELATIVE_THRESHOLD,
            "requires_beyond_measurement_noise": True,
            "requires_primary_workload": True,
            "passing_cells_by_execution_mode": gain_by_mode,
            "passing_primary_cells_by_execution_mode": primary_gain_by_mode,
            "passes": gain_passes and completeness_passes,
        },
        "generation_gate": {
            "cell_ratio_count": len(ratios),
            "geometric_mean_current_over_baseline": geometric_mean,
            "maximum_geometric_mean": GENERATION_GEOMETRIC_MEAN_MAXIMUM,
            "passes": generation_passes and completeness_passes,
        },
        "cells": evidence,
    }


def _discover_results(output_root: Path) -> dict[str, dict[str, Any]]:
    cells_root = output_root / "cells"
    if not cells_root.is_dir():
        return {}
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(cells_root.glob("*/result.json")):
        cell_id = path.parent.name
        if cell_id in results:
            raise MatrixError(f"duplicate matrix cell result: {cell_id}")
        results[cell_id] = _json_object(path)
    return results


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise MatrixError(
            f"cannot inspect acceptance source checkout with git: {detail}"
        )
    return completed.stdout


def _repository_identity() -> dict[str, object]:
    revision = _git_output("rev-parse", "--verify", "HEAD").strip()
    if not _is_git_sha(revision):
        raise MatrixError("acceptance source checkout has an invalid HEAD revision")
    status = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    dirty_entries = [line for line in status.splitlines() if line]
    return {
        "root": str(ROOT),
        "head_revision": revision,
        "clean": not dirty_entries,
        "dirty_entries": dirty_entries,
    }


def _acceptance_state(
    *,
    baseline_python: Path,
    current_python: Path,
    baseline_dependency_site: Path,
    current_dependency_site: Path,
    ufo_sm_model: Path,
) -> dict[str, object]:
    environment = regression._environment()
    interpreters = {
        "baseline": _absolute(baseline_python),
        "current": _absolute(current_python),
    }
    dependency_sites = {
        "baseline": _absolute(baseline_dependency_site).resolve(strict=True),
        "current": _absolute(current_dependency_site).resolve(strict=True),
    }
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "repository": _repository_identity(),
        "tools": {
            "matrix_driver": regression._path_identity(Path(__file__)),
            "cell_driver": regression._path_identity(CELL_DRIVER),
            "watchdog": regression._path_identity(regression.WATCHDOG),
            "native_sample_helper": regression._path_identity(
                regression.NATIVE_SAMPLE_HELPER
            ),
            "dependency_entry": regression._path_identity(regression.DEPENDENCY_ENTRY),
        },
        "interpreters": {
            lane: regression._path_identity(interpreter)
            for lane, interpreter in interpreters.items()
        },
        "installed_pyamplicol": {
            lane: regression._installed_pyamplicol_identity(
                interpreter,
                environment=environment,
            )
            for lane, interpreter in interpreters.items()
        },
        "dependency_sites": {
            lane: regression._dependency_site_identity(path)
            for lane, path in dependency_sites.items()
        },
        "model": regression._model_identity(str(ufo_sm_model)),
    }


def _preflight(
    arguments: argparse.Namespace,
    *,
    ufo_sm_model: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    if (
        arguments.expected_baseline_source_revision != FROZEN_BASELINE_SOURCE_REVISION
        or arguments.expected_baseline_native_inputs_sha256
        != FROZEN_BASELINE_NATIVE_INPUTS_SHA256
    ):
        raise MatrixError(
            "baseline pins do not identify the frozen pre-Arena candidate"
        )
    expected_builds = {
        "baseline": {
            "source_revision": arguments.expected_baseline_source_revision,
            "native_build_inputs_sha256": (
                arguments.expected_baseline_native_inputs_sha256
            ),
            "distribution_content_sha256": (
                arguments.expected_baseline_distribution_sha256
            ),
            "native_module_sha256": (arguments.expected_baseline_native_module_sha256),
        },
        "current": {
            "source_revision": arguments.expected_current_source_revision,
            "native_build_inputs_sha256": (
                arguments.expected_current_native_inputs_sha256
            ),
            "distribution_content_sha256": (
                arguments.expected_current_distribution_sha256
            ),
            "native_module_sha256": arguments.expected_current_native_module_sha256,
        },
    }
    state = _acceptance_state(
        baseline_python=arguments.baseline_python,
        current_python=arguments.current_python,
        baseline_dependency_site=arguments.baseline_dependency_site,
        current_dependency_site=arguments.current_dependency_site,
        ufo_sm_model=ufo_sm_model,
    )
    frozen_binary = FROZEN_BASELINE_BINARY_IDENTITIES.get(
        (str(state["system"]), str(state["machine"]))
    )
    if frozen_binary is not None and (
        arguments.expected_baseline_distribution_sha256
        != frozen_binary["distribution_content_sha256"]
        or arguments.expected_baseline_native_module_sha256
        != frozen_binary["native_module_sha256"]
    ):
        raise MatrixError(
            "baseline binary pins do not identify the frozen candidate on "
            f"{state['system']}/{state['machine']}"
        )
    repository = _mapping(state.get("repository"))
    if repository is None or repository.get("clean") is not True:
        entries = repository.get("dirty_entries") if repository is not None else None
        raise MatrixError(
            "current acceptance source checkout is not clean: "
            f"{entries if entries is not None else 'identity absent'}"
        )
    if repository.get("head_revision") != arguments.expected_current_source_revision:
        raise MatrixError(
            "expected current source revision does not match clean checkout HEAD"
        )
    installations = _mapping(state.get("installed_pyamplicol"))
    errors: list[str] = []
    for lane in ("baseline", "current"):
        errors.extend(
            _installation_identity_errors(
                installations.get(lane) if installations is not None else None,
                expected=expected_builds[lane],
                label=lane,
            )
        )
    if errors:
        raise MatrixError("installation preflight failed: " + "; ".join(errors))
    return expected_builds, state


def _artifact_postflight_errors(
    results: Mapping[str, Mapping[str, Any]],
    *,
    output_root: Path,
) -> list[str]:
    errors: list[str] = []
    expected = {cell.cell_id: cell for cell in CANONICAL_CELLS}
    checked: set[tuple[str, str]] = set()
    for cell_id, result in sorted(results.items()):
        cell = expected.get(cell_id)
        if cell is None:
            continue
        artifacts = _mapping(result.get("artifacts"))
        if artifacts is None:
            continue
        for lane in ("baseline", "current"):
            key = (cell.artifact_group_id, lane)
            if key in checked:
                continue
            checked.add(key)
            record = _mapping(artifacts.get(lane))
            expected_path = (
                _absolute(output_root)
                / "artifact-groups"
                / cell.artifact_group_id
                / lane
                / "artifact"
            )
            if record is None or record.get("path") != str(expected_path):
                errors.append(
                    f"{cell.artifact_group_id}/{lane}: artifact path is not bound "
                    "to the matrix output"
                )
                continue
            try:
                observed = regression._artifact_metadata(
                    expected_path,
                    expected_process=cell.process,
                    expected_color=cell.color_accuracy,
                    expected_execution_mode=cell.execution_mode,
                    expected_lc_flow_layout=(
                        cell.lc_flow_layout if cell.color_accuracy == "lc" else None
                    ),
                )
            except (OSError, regression.RegressionError) as error:
                errors.append(
                    f"{cell.artifact_group_id}/{lane}: cannot reopen artifact: {error}"
                )
                continue
            for field in (
                "artifact_id",
                "manifest_sha256",
                "tree_identity",
                "payload_digests",
            ):
                if observed.get(field) != record.get(field):
                    errors.append(
                        f"{cell.artifact_group_id}/{lane}: reopened artifact "
                        f"{field} differs"
                    )
    return errors


def _mark_in_progress(
    audited: dict[str, Any],
    *,
    preflight_state: Mapping[str, object],
    output_root: Path,
) -> dict[str, Any]:
    audited["run_complete"] = False
    audited["passes"] = False
    audited["provenance"] = {
        "preflight": dict(preflight_state),
        "postflight": None,
        "output_root": str(_absolute(output_root)),
    }
    audited["outer_provenance_gate"] = {
        "errors": ["matrix run/postflight is incomplete"],
        "passes": False,
    }
    return audited


def run_matrix(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = _absolute(arguments.output_root)
    ufo_sm_model = _absolute(arguments.ufo_sm_model)
    if not ufo_sm_model.exists():
        raise MatrixError(f"UFO-SM model does not exist: {ufo_sm_model}")
    for label, interpreter in (
        ("baseline", _absolute(arguments.baseline_python)),
        ("current", _absolute(arguments.current_python)),
    ):
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise MatrixError(f"{label} Python is not executable: {interpreter}")
    expected_builds, preflight_state = _preflight(
        arguments,
        ufo_sm_model=ufo_sm_model,
    )
    expected_platform = str(preflight_state["platform"])
    output_root.mkdir(parents=True, exist_ok=True)

    if not arguments.audit_only:
        initial_results = _discover_results(output_root)
        initial = audit_results(
            initial_results,
            baseline_python=arguments.baseline_python,
            current_python=arguments.current_python,
            ufo_sm_model=ufo_sm_model,
            output_root=output_root,
            expected_platform=expected_platform,
            expected_builds=expected_builds,
        )
        _mark_in_progress(
            initial,
            preflight_state=preflight_state,
            output_root=output_root,
        )
        _write_json_atomic(output_root / "matrix-result.json", initial)
        regenerated_groups: set[str] = set()
        for index, cell in enumerate(CANONICAL_CELLS, start=1):
            result_path = output_root / "cells" / cell.cell_id / "result.json"
            if result_path.is_file() and not arguments.rerun_results:
                existing = _cell_evidence(
                    cell,
                    _json_object(result_path),
                    baseline_python=arguments.baseline_python,
                    current_python=arguments.current_python,
                    ufo_sm_model=ufo_sm_model,
                    output_root=output_root,
                    expected_platform=expected_platform,
                    expected_builds=expected_builds,
                )
                if existing["passes"]:
                    continue
            command = cell_command(
                cell,
                baseline_python=arguments.baseline_python,
                current_python=arguments.current_python,
                output_root=output_root,
                ufo_sm_model=ufo_sm_model,
                baseline_dependency_site=arguments.baseline_dependency_site,
                current_dependency_site=arguments.current_dependency_site,
                samples=arguments.samples,
                target_runtime=arguments.target_runtime,
                minimum_samples=arguments.minimum_samples,
                warmup_runs=arguments.warmup_runs,
                generation_timeout=arguments.generation_timeout,
                profile_timeout=arguments.profile_timeout,
                regenerate_artifacts=(
                    arguments.regenerate_artifacts
                    and cell.artifact_group_id not in regenerated_groups
                ),
            )
            print(
                f"[{index:03d}/{len(CANONICAL_CELLS)}] {cell.cell_id}",
                file=sys.stderr,
                flush=True,
            )
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if completed.returncode not in {0, 1} or not result_path.is_file():
                detail = completed.stderr.strip()
                raise MatrixError(
                    f"cell driver failed for {cell.cell_id} "
                    f"(exit {completed.returncode}): {detail}"
                )
            if arguments.regenerate_artifacts:
                regenerated_groups.add(cell.artifact_group_id)
            partial_results = _discover_results(output_root)
            partial = audit_results(
                partial_results,
                baseline_python=arguments.baseline_python,
                current_python=arguments.current_python,
                ufo_sm_model=ufo_sm_model,
                output_root=output_root,
                expected_platform=expected_platform,
                expected_builds=expected_builds,
            )
            _mark_in_progress(
                partial,
                preflight_state=preflight_state,
                output_root=output_root,
            )
            _write_json_atomic(output_root / "matrix-result.json", partial)

    results = _discover_results(output_root)
    # Unexpected result directories remain in ``results`` so stale output fails
    # completeness instead of being silently ignored.
    audited = audit_results(
        results,
        baseline_python=arguments.baseline_python,
        current_python=arguments.current_python,
        ufo_sm_model=ufo_sm_model,
        output_root=output_root,
        expected_platform=expected_platform,
        expected_builds=expected_builds,
    )
    postflight_state = _acceptance_state(
        baseline_python=arguments.baseline_python,
        current_python=arguments.current_python,
        baseline_dependency_site=arguments.baseline_dependency_site,
        current_dependency_site=arguments.current_dependency_site,
        ufo_sm_model=ufo_sm_model,
    )
    outer_errors = _artifact_postflight_errors(results, output_root=output_root)
    if postflight_state != preflight_state:
        outer_errors.append(
            "acceptance host/source/interpreter/dependency/model state changed "
            "between preflight and postflight"
        )
    outer_passes = not outer_errors and bool(audited["complete"])
    audited["run_complete"] = bool(audited["complete"])
    audited["provenance"] = {
        "preflight": preflight_state,
        "postflight": postflight_state,
        "output_root": str(output_root),
    }
    audited["outer_provenance_gate"] = {
        "errors": outer_errors,
        "passes": outer_passes,
    }
    audited["passes"] = bool(audited["passes"] and outer_passes)
    _write_json_atomic(output_root / "matrix-result.json", audited)
    return audited


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


def _git_sha(value: str) -> str:
    if not _is_git_sha(value):
        raise argparse.ArgumentTypeError("must be a lowercase 40-character Git SHA")
    return value


def _sha256(value: str) -> str:
    if not _is_sha256(value):
        raise argparse.ArgumentTypeError("must be a lowercase 64-character SHA-256")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline-python", type=Path, required=True)
    result.add_argument("--current-python", type=Path, required=True)
    result.add_argument("--ufo-sm-model", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument(
        "--expected-baseline-source-revision",
        type=_git_sha,
        required=True,
    )
    result.add_argument(
        "--expected-current-source-revision",
        type=_git_sha,
        required=True,
    )
    result.add_argument(
        "--expected-baseline-native-inputs-sha256",
        type=_sha256,
        required=True,
    )
    result.add_argument(
        "--expected-current-native-inputs-sha256",
        type=_sha256,
        required=True,
    )
    result.add_argument(
        "--expected-baseline-distribution-sha256",
        type=_sha256,
        required=True,
    )
    result.add_argument(
        "--expected-current-distribution-sha256",
        type=_sha256,
        required=True,
    )
    result.add_argument(
        "--expected-baseline-native-module-sha256",
        type=_sha256,
        required=True,
    )
    result.add_argument(
        "--expected-current-native-module-sha256",
        type=_sha256,
        required=True,
    )
    result.add_argument("--baseline-dependency-site", type=Path, required=True)
    result.add_argument("--current-dependency-site", type=Path, required=True)
    result.add_argument("--samples", type=_at_least_five, default=7)
    result.add_argument("--target-runtime", type=_positive_float, default=5.0)
    result.add_argument("--minimum-samples", type=_at_least_five, default=7)
    result.add_argument("--warmup-runs", type=_positive_int, default=2)
    result.add_argument(
        "--generation-timeout",
        type=_positive_float,
        default=DEFAULT_MATRIX_GENERATION_TIMEOUT,
    )
    result.add_argument(
        "--profile-timeout",
        type=_positive_float,
        default=DEFAULT_MATRIX_PROFILE_TIMEOUT,
    )
    result.add_argument(
        "--audit-only",
        action="store_true",
        help="audit existing cell results without launching subprocesses",
    )
    result.add_argument(
        "--rerun-results",
        action="store_true",
        help="rerun every cell even when a result.json exists",
    )
    result.add_argument(
        "--regenerate-artifacts",
        action="store_true",
        help="pass --regenerate-artifacts to every cell driver",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.audit_only and (
        arguments.rerun_results or arguments.regenerate_artifacts
    ):
        print(
            "compiled-mode-matrix: --audit-only cannot rerun results or artifacts",
            file=sys.stderr,
        )
        return 2
    if arguments.regenerate_artifacts and not arguments.rerun_results:
        print(
            "compiled-mode-matrix: --regenerate-artifacts requires "
            "--rerun-results so every shared group is refreshed coherently",
            file=sys.stderr,
        )
        return 2
    try:
        result = run_matrix(arguments)
    except (MatrixError, OSError, ValueError) as error:
        print(f"compiled-mode-matrix: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
