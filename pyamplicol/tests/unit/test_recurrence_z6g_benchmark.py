# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "developer" / "recurrence_z6g_benchmark.py"
SPEC = importlib.util.spec_from_file_location("recurrence_z6g_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def _arguments(*values: str) -> object:
    arguments = benchmark.parser().parse_args(list(values))
    arguments.modes = benchmark._normalize_modes(arguments)
    arguments.batch_size = benchmark._normalize_batch_sizes(arguments)
    return arguments


def test_modes_default_to_all_lanes_and_preserve_requested_order() -> None:
    default = benchmark.parser().parse_args([])
    assert benchmark._normalize_modes(default) == [
        "compiled",
        "eager",
        "recurrence",
    ]

    selected = benchmark.parser().parse_args(
        ["--mode", "recurrence", "--mode", "eager"]
    )
    assert benchmark._normalize_modes(selected) == ["recurrence", "eager"]


def test_modes_reject_duplicates_and_old_new_flag_mix() -> None:
    duplicate = benchmark.parser().parse_args(
        ["--mode", "compiled", "--mode", "compiled"]
    )
    with pytest.raises(benchmark.HarnessError, match="must be unique"):
        benchmark._normalize_modes(duplicate)

    mixed = benchmark.parser().parse_args(
        ["--mode", "compiled", "--only-mode", "recurrence"]
    )
    with pytest.raises(benchmark.HarnessError, match="cannot be combined"):
        benchmark._normalize_modes(mixed)


def test_default_batches_cover_milestone_zero_and_explicit_values_replace() -> None:
    assert benchmark.DEFAULT_BATCH_SIZES == (1, 128, 1024)
    defaults = benchmark.parser().parse_args([])
    assert defaults.minimum_samples == 7
    assert defaults.subprocess_samples == 7
    assert benchmark._normalize_batch_sizes(defaults) == [
        1,
        128,
        1024,
    ]
    assert benchmark._normalize_batch_sizes(
        benchmark.parser().parse_args(["--batch-size", "8", "--batch-size", "32"])
    ) == [8, 32]

    duplicate = benchmark.parser().parse_args(
        ["--batch-size", "8", "--batch-size", "8"]
    )
    with pytest.raises(benchmark.HarnessError, match="must be unique"):
        benchmark._normalize_batch_sizes(duplicate)


def test_paired_migration_capture_requires_exactly_two_warmups() -> None:
    arguments = _arguments(
        "--paired-profile-coordination-dir",
        "coordination",
        "--paired-profile-role",
        "candidate",
        "--paired-profile-session-id",
        "session",
        "--warmup-runs",
        "3",
    )

    with pytest.raises(benchmark.HarnessError, match="exactly two warmup"):
        benchmark.run(arguments)


def test_migration_process_request_expands_without_changing_default_route() -> None:
    default = benchmark.parser().parse_args([])
    assert benchmark._selected_process(default) == "u u~ > Z g g g g g g"
    assert benchmark._selected_process_name(default) == "uubar_Z_6g"
    assert benchmark._authoritative_process_family(default)

    migration = benchmark.parser().parse_args(
        ["--process-expression", "d d~ > z + 6*g"]
    )
    assert benchmark._selected_process(migration) == "d d~ > Z g g g g g g"
    assert benchmark._selected_process_name(migration) == "ddbar_Z_6g"
    assert benchmark._authoritative_process_family(migration)

    custom = benchmark.parser().parse_args(["--process-expression", "g g > t t~"])
    assert benchmark._selected_process_name(custom) == "custom_process"
    assert not benchmark._authoritative_process_family(custom)


def test_benchmark_source_root_can_select_an_immutable_external_checkout(
    tmp_path: Path,
) -> None:
    assert (
        benchmark._resolve_benchmark_source_root(str(ROOT), script_path=SCRIPT) == ROOT
    )
    with pytest.raises(RuntimeError, match="must be an absolute path"):
        benchmark._resolve_benchmark_source_root(
            "relative/checkout",
            script_path=SCRIPT,
        )
    with pytest.raises(RuntimeError, match="not a pyAmpliCol source checkout"):
        benchmark._resolve_benchmark_source_root(str(tmp_path), script_path=SCRIPT)


def test_incomplete_capture_requires_explicit_diagnostic_success() -> None:
    assert (
        benchmark._result_exit_code(
            {"complete": True, "passes": True},
            allow_diagnostic_incomplete_success=False,
        )
        == 0
    )
    assert (
        benchmark._result_exit_code(
            {"complete": True, "passes": False},
            allow_diagnostic_incomplete_success=True,
        )
        == 1
    )
    assert (
        benchmark._result_exit_code(
            {"complete": False, "passes": None},
            allow_diagnostic_incomplete_success=False,
        )
        == 1
    )
    assert (
        benchmark._result_exit_code(
            {"complete": False, "passes": None},
            allow_diagnostic_incomplete_success=True,
        )
        == 0
    )
    assert (
        benchmark._result_exit_code(
            {"complete": False, "passes": True},
            allow_diagnostic_incomplete_success=True,
        )
        == 1
    )


def _fixture(
    *,
    point_count: int = 2,
    points_sha256: str = "b" * 64,
    file_sha256: str = "c" * 64,
) -> dict[str, object]:
    return {
        "point_count": point_count,
        "points_sha256": points_sha256,
        "file": {
            "path": "/fixture.json",
            "resolved_path": "/fixture.json",
            "size_bytes": 123,
            "sha256": file_sha256,
        },
    }


def _resource_peak(
    *,
    self_peak_bytes: int = 1024,
    maximum_child_peak_bytes: int = 0,
) -> dict[str, object]:
    return {
        "source": "resource.getrusage",
        "self_peak_bytes": self_peak_bytes,
        "maximum_child_peak_bytes": maximum_child_peak_bytes,
        "observed_lower_bound_bytes": max(
            self_peak_bytes,
            maximum_child_peak_bytes,
        ),
        "semantics": benchmark._RESOURCE_PEAK_SEMANTICS,
    }


def _artifact_semantic_identity(
    *,
    specialized_axes: tuple[str, ...] = (),
    complete: bool = True,
    model_name: str = "built-in-sm",
    model_content_sha256: str = "c" * 64,
    color_id: str = "flow:2,1",
    color_word: tuple[int, ...] = (2, 1),
    color_count: int = 1,
    helicity_id: str = "h:-1,+1",
    helicity_values: tuple[int, ...] = (-1, 1),
    secondary_helicity_structural_zero: bool = False,
) -> dict[str, object]:
    color_ids = [
        color_id,
        *(f"flow:{index + 3},2,1" for index in range(color_count - 1)),
    ]
    opposite_helicity_values = tuple(-value for value in helicity_values)
    opposite_helicity_id = "h:" + ",".join(
        f"{value:+d}" for value in opposite_helicity_values
    )
    helicity_ids = [helicity_id, opposite_helicity_id]
    color_entries = [
        {
            "index": index,
            "id": identifier,
            "kind": "lc-flow",
            "word": (list(color_word) if index == 0 else [index + 3, 2, 1]),
            "representative_id": identifier,
            "computed": True,
            "coefficient": 1.0,
            "structural_zero": None,
        }
        for index, identifier in enumerate(color_ids)
    ]
    helicity_entries = [
        {
            "index": 0,
            "id": helicity_id,
            "values": list(helicity_values),
            "representative_id": helicity_id,
            "computed": True,
            "coefficient": 1.0,
            "structural_zero": False,
        },
        {
            "index": 1,
            "id": opposite_helicity_id,
            "values": list(opposite_helicity_values),
            "representative_id": (
                opposite_helicity_id
                if secondary_helicity_structural_zero
                else helicity_id
            ),
            "computed": False,
            "coefficient": (0.0 if secondary_helicity_structural_zero else 1.0),
            "structural_zero": secondary_helicity_structural_zero,
        },
    ]
    normalization = {"average_factor": 36, "color_factor": 3}
    model_common = {
        "name": model_name,
        "content_sha256": model_content_sha256,
        "compiled_schema_version": 9,
        "restriction": None,
    }
    model_manifest = {**model_common, "source_kind": "compiled-model"}
    model_identity = {
        "manifest": model_manifest,
        "manifest_sha256": benchmark._canonical_sha256(model_manifest),
        "common_physics_identity": model_common,
        "common_physics_identity_sha256": benchmark._canonical_sha256(model_common),
    }
    color_specialized = "color_flow" in specialized_axes
    helicity_specialized = "helicity" in specialized_axes
    selector_semantics = {
        "kind": "pyamplicol-runtime-selectors",
        "contract_version": 1,
        "axes": {
            "color_flow": {
                "generation_coverage": (
                    "selected" if color_specialized else "complete"
                ),
                "generation_selection": [0] if color_specialized else [],
                "runtime_contract": (
                    "generation-specialized"
                    if color_specialized
                    else "complete-reusable"
                ),
            },
            "helicity": {
                "generation_coverage": (
                    "selected" if helicity_specialized else "complete"
                ),
                "generation_selection": ({"1": -1} if helicity_specialized else {}),
                "runtime_contract": (
                    "generation-specialized"
                    if helicity_specialized
                    else "complete-reusable"
                ),
            },
        },
        "generation_specialized_axes": list(specialized_axes),
    }
    runtime_selectors = {
        **selector_semantics,
        "provenance": "test-runtime-selectors-v1",
    }
    reduction_ordering = {
        "kind": "lc-diagonal",
        "ordered_groups": [
            {
                "id": f"reduction:{index}",
                "physical_color_ids": [identifier],
                "physical_helicity_ids": (
                    [helicity_ids[0]]
                    if secondary_helicity_structural_zero
                    else helicity_ids
                ),
                "representative_color_id": identifier,
                "representative_helicity_id": helicity_ids[0],
            }
            for index, identifier in enumerate(color_ids)
        ],
    }
    execution_color_axis = benchmark._ordered_physical_axis(
        color_entries,
        label="color-flow",
        require_structural_zero=False,
    )
    execution_helicity_axis = benchmark._ordered_physical_axis(
        helicity_entries,
        label="helicity",
        require_structural_zero=True,
    )
    color_axis = benchmark._logical_physical_axis(
        execution_color_axis,
        require_structural_zero=False,
    )
    helicity_axis = benchmark._logical_physical_axis(
        execution_helicity_axis,
        require_structural_zero=True,
    )
    logical_reduction = benchmark._logical_reduction_ordering_identity(
        reduction_ordering["kind"],
        color_axis=color_axis,
        helicity_axis=helicity_axis,
        artifact=Path("<test-profile>"),
    )
    execution_reduction = benchmark._reduction_ordering_identity(
        {
            "kind": reduction_ordering["kind"],
            "groups": reduction_ordering["ordered_groups"],
        },
        color_axis=execution_color_axis,
        helicity_axis=execution_helicity_axis,
        artifact=Path("<test-profile>"),
    )
    execution_reduction_identity = {
        "kind": "expanded-json-reduction-v1",
        "reduction_kind": reduction_ordering["kind"],
        "group_count": len(reduction_ordering["ordered_groups"]),
        "computed_physical_color_count": sum(
            entry["computed"] is True for entry in color_entries
        ),
        "live_physical_helicity_count": sum(
            entry["structural_zero"] is False for entry in helicity_entries
        ),
        "materialized_ordering_sha256": execution_reduction["ordering_sha256"],
    }
    execution_ordering = {
        "runtime_process_contract": {"id": "process-1"},
        "manifest_payload_order": [{"path": "plan.bin", "role": "runtime"}],
    }
    return {
        "kind": "pyamplicol-benchmark-artifact-semantic-identity",
        "schema_version": benchmark.ARTIFACT_SEMANTIC_IDENTITY_SCHEMA,
        "coverage": {
            "color": ("complete" if complete and not color_specialized else "selected"),
            "helicities": (
                "complete" if complete and not helicity_specialized else "selected"
            ),
            "complete_physical_axes": (
                complete and not color_specialized and not helicity_specialized
            ),
        },
        "physical_color_flows": color_axis,
        "physical_helicities": helicity_axis,
        "normalization": normalization,
        "normalization_sha256": benchmark._canonical_sha256(normalization),
        "manifest_model_identity": model_identity,
        "runtime_selector_semantics": selector_semantics,
        "runtime_selector_semantics_sha256": benchmark._canonical_sha256(
            selector_semantics
        ),
        "runtime_selectors": runtime_selectors,
        "runtime_selectors_sha256": benchmark._canonical_sha256(runtime_selectors),
        "reduction_ordering": logical_reduction["ordering"],
        "reduction_ordering_sha256": logical_reduction["ordering_sha256"],
        "reduction_coverage": logical_reduction["coverage"],
        "execution_reduction_identity": execution_reduction_identity,
        "execution_reduction_identity_sha256": benchmark._canonical_sha256(
            execution_reduction_identity
        ),
        "execution_reduction_coverage": execution_reduction["coverage"],
        "execution_schedule_ordering": execution_ordering,
        "execution_schedule_ordering_sha256": benchmark._canonical_sha256(
            execution_ordering
        ),
        "generation_specialized_axes": list(specialized_axes),
    }


def _passing_schedule(
    *,
    modes: tuple[str, ...] = benchmark.EXECUTION_MODES,
    batch_sizes: tuple[int, ...] = benchmark.DEFAULT_BATCH_SIZES,
    subprocess_samples: int = benchmark.MIN_AUTHORITATIVE_SAMPLES,
    jit_optimization_level: int = 2,
    lc_flow_layout: str = "topology-replay",
) -> dict[str, object]:
    schedule: dict[str, object] = benchmark._build_profile_schedule(
        modes,
        batch_sizes,
        subprocess_samples=subprocess_samples,
    )
    entries = schedule["entries"]
    assert isinstance(entries, list)
    semantic_identity_sha256 = benchmark._canonical_sha256(
        _artifact_semantic_identity()
    )
    for entry in entries:
        assert isinstance(entry, dict)
        index = entry["schedule_index"]
        command = benchmark._command_identity(
            ("python", "profile-worker", "--schedule-index", str(index))
        )
        verification = {
            "kind": benchmark.WORKER_VERIFICATION_KIND,
            "schema_version": benchmark.WORKER_VERIFICATION_SCHEMA,
            "verified_at_utc": "2026-07-24T00:00:00+00:00",
            "expected": {
                field: (
                    semantic_identity_sha256
                    if field == "artifact_semantic_identity_sha256"
                    else "a" * 64
                )
                for field in benchmark._PROFILE_EXPECTATION_FIELDS
            },
            "observed": {
                field: (
                    semantic_identity_sha256
                    if field == "artifact_semantic_identity_sha256"
                    else "a" * 64
                )
                for field in benchmark._PROFILE_EXPECTATION_FIELDS
            },
            "artifact_semantic_identity_sha256": semantic_identity_sha256,
            "effective_contract": {
                "execution_mode": entry["mode"],
                "backend": "jit",
                "jit_optimization_level": jit_optimization_level,
                "color_accuracy": "lc",
                "lc_flow_layout": lc_flow_layout,
            },
            "loaded_runtime_artifact": {
                "kind": "pyamplicol-loaded-runtime-artifact-verification",
                "schema_version": 1,
                "phase": "after-native-load-before-timing",
                "checked_at_utc": "2026-07-24T00:00:00+00:00",
                "expected_artifact_id": "a" * 64,
                "loaded_artifact_id": "a" * 64,
                "passes": True,
            },
        }
        invocation = {
            "started_at_utc": "2026-07-24T00:00:00+00:00",
            "finished_at_utc": "2026-07-24T00:00:01+00:00",
            "wall_seconds": 12.0,
            "command": command,
        }
        invocation["content_sha256"] = benchmark._canonical_sha256(invocation)
        addressed_payload_sha256 = hashlib.sha256(
            f"worker-payload-{index}".encode()
        ).hexdigest()
        result_record = {
            "kind": benchmark.RETAINED_WORKER_RESULT_KIND,
            "schema_version": benchmark.RETAINED_WORKER_RESULT_SCHEMA,
            "recorded_at_utc": "2026-07-24T00:00:01+00:00",
            "addressed_payload_sha256": addressed_payload_sha256,
            "upstream_worker_result_record_sha256": hashlib.sha256(
                f"upstream-result-{index}".encode()
            ).hexdigest(),
            "worker_process_record_sha256": hashlib.sha256(
                f"worker-process-{index}".encode()
            ).hexdigest(),
            "worker_invocation_sha256": invocation["content_sha256"],
        }
        result_record["content_sha256"] = benchmark._canonical_sha256(result_record)
        entry["worker_invocation"] = invocation
        entry["pre_timing_verification"] = verification
        entry["worker_result_record"] = result_record
        entry["worker_result_sha256"] = addressed_payload_sha256
    return schedule


def _raw_native_wall_blocks(
    *,
    wall_seconds_per_point: float,
    batch_size: int,
    fixture_points_sha256: str,
    repetitions_per_block: int | None = None,
    block_count: int = benchmark.MIN_AUTHORITATIVE_SAMPLES,
    minimum_native_wall_seconds: float = 5.0,
) -> dict[str, object]:
    if repetitions_per_block is None:
        repetitions_per_block = math.ceil(
            (minimum_native_wall_seconds * 1.02)
            / (block_count * wall_seconds_per_point * batch_size)
        )
    blocks: list[dict[str, object]] = []
    for block_index in range(block_count):
        block = {
            "block_index": block_index,
            "started_at_utc": (f"2026-07-24T00:00:00.{block_index:06d}+00:00"),
            "finished_at_utc": (f"2026-07-24T00:00:00.{block_index + 1:06d}+00:00"),
            "caller_elapsed_seconds": (
                wall_seconds_per_point * repetitions_per_block * batch_size
            ),
            "native_wall_seconds": (
                wall_seconds_per_point * repetitions_per_block * batch_size
            ),
            "wall_seconds_per_point": wall_seconds_per_point,
            "repetitions": repetitions_per_block,
            "batch_size": batch_size,
            "evaluation_count": repetitions_per_block,
            "evaluated_point_count": repetitions_per_block * batch_size,
        }
        block["content_sha256"] = benchmark._canonical_sha256(block)
        blocks.append(block)
    observed_native_wall_seconds = sum(
        float(block["native_wall_seconds"]) for block in blocks
    )
    observed_caller_elapsed_seconds = sum(
        float(block["caller_elapsed_seconds"]) for block in blocks
    )
    return {
        "kind": benchmark.RAW_NATIVE_WALL_BLOCK_KIND,
        "schema_version": benchmark.RAW_NATIVE_WALL_BLOCK_SCHEMA,
        "measurement_contract": benchmark.RAW_NATIVE_WALL_MEASUREMENT_CONTRACT,
        "source": "runtime._benchmark_f64_wall_time",
        "fixture_points_sha256": fixture_points_sha256,
        "minimum_native_wall_seconds": minimum_native_wall_seconds,
        "observed_native_wall_seconds": observed_native_wall_seconds,
        "observed_caller_elapsed_seconds": observed_caller_elapsed_seconds,
        "minimum_duration_satisfied": (
            observed_native_wall_seconds >= minimum_native_wall_seconds
        ),
        "calibration": {
            "kind": "benchmark-runner-wall-rate-calibration",
            "schema_version": 1,
            "benchmark_runner_sample_count": 20,
            "benchmark_runner_repetitions_per_sample": 1,
            "benchmark_runner_total_repetitions": 20,
            "benchmark_runner_wall_seconds_per_point": wall_seconds_per_point,
            "requested_minimum_block_count": block_count,
            "preceded_by_benchmark_runner_warmup_runs": 2,
            "duration_headroom_factor": 1.02,
            "scaled_repetitions_per_block": repetitions_per_block,
        },
        "block_count": block_count,
        "repetitions_per_block": repetitions_per_block,
        "evaluation_count": block_count * repetitions_per_block,
        "evaluated_point_count": (block_count * repetitions_per_block * batch_size),
        "wall_seconds_per_point_median": wall_seconds_per_point,
        "wall_seconds_per_point_mad": 0.0,
        "blocks": blocks,
        "blocks_sha256": benchmark._canonical_sha256(blocks),
    }


def _profile(
    mode: str,
    values: tuple[complex, ...],
    *,
    fixture: dict[str, object] | None = None,
    passes: bool = True,
    schedule: dict[str, object] | None = None,
    specialized_axes: tuple[str, ...] = (),
    semantic_identity: dict[str, object] | None = None,
    selector_contract: dict[str, object] | None = None,
    process_expression: str = "u u~ > Z g g g g g g",
) -> dict[str, object]:
    schedule = _passing_schedule() if schedule is None else schedule
    validation_fixture = _fixture() if fixture is None else fixture
    selector_contract = (
        {
            "color_flow_request": "1",
            "resolved_color_flow_id": "flow:2,1",
            "helicity_request": "1",
            "resolved_helicity_id": None,
            "color_flow_count": 1,
            "helicity_count": 2,
            "structural_zero_helicity_count": 0,
            "workload": "single-runtime-selected-flow/helicity-sum",
        }
        if selector_contract is None
        else copy.deepcopy(selector_contract)
    )
    semantic_identity = (
        _artifact_semantic_identity(specialized_axes=specialized_axes)
        if semantic_identity is None
        else copy.deepcopy(semantic_identity)
    )
    semantic_identity_sha256 = benchmark._canonical_sha256(semantic_identity)
    complex_payloads = [[value.real, value.imag] for value in values]
    if selector_contract["workload"] == "all-flows/runtime-selected-single-helicity":
        resolved_helicity_ids = [selector_contract["resolved_helicity_id"]]
        resolved_color_ids = semantic_identity["physical_color_flows"]["ordered_ids"]
    else:
        resolved_helicity_ids = semantic_identity["physical_helicities"]["ordered_ids"]
        resolved_color_ids = [selector_contract["resolved_color_flow_id"]]
    component_count = len(resolved_helicity_ids) * len(resolved_color_ids)
    resolved_components = [
        [list(value), *([[0.0, 0.0]] * (component_count - 1))]
        for value in complex_payloads
    ]
    validation_record = {
        "passes": passes,
        "fixture": validation_fixture,
        "selected_totals": complex_payloads,
        "resolved_sums": copy.deepcopy(complex_payloads),
        "resolved_helicity_ids": resolved_helicity_ids,
        "resolved_color_ids": resolved_color_ids,
        "resolved_components": resolved_components,
        "point_comparisons": [
            {
                "point_index": index,
                "selected_total": value,
                "resolved_sum": list(value),
                "absolute_difference": 0.0,
                "relative_difference": 0.0,
                "passes": True,
            }
            for index, value in enumerate(complex_payloads)
        ],
        "maximum_absolute_difference": 0.0,
        "maximum_relative_difference": 0.0,
    }
    lane_contract = {
        "process_id": "process-1",
        "process_expression": process_expression,
        "selector_contract": selector_contract,
        "validation": validation_record,
        "artifact_semantic_identity": semantic_identity,
        "artifact_semantic_identity_sha256": semantic_identity_sha256,
    }
    lane_contract_sha256 = benchmark._canonical_sha256(lane_contract)
    fixture_points_sha256 = validation_fixture["points_sha256"]
    assert isinstance(fixture_points_sha256, str)
    entries = schedule["entries"]
    assert isinstance(entries, list)
    measurements: list[dict[str, object]] = []
    for batch_size in benchmark.DEFAULT_BATCH_SIZES:
        cell_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry["mode"] == mode
            and entry["batch_size"] == batch_size
        ]
        samples = []
        for entry in cell_entries:
            wall = 1.0e-6 * (1.0 + 0.01 * int(entry["round"]))
            raw_blocks = _raw_native_wall_blocks(
                wall_seconds_per_point=wall,
                batch_size=batch_size,
                fixture_points_sha256=fixture_points_sha256,
            )
            verification = copy.deepcopy(entry["pre_timing_verification"])
            verification["artifact_semantic_identity_sha256"] = semantic_identity_sha256
            verification["expected"]["artifact_semantic_identity_sha256"] = (
                semantic_identity_sha256
            )
            verification["observed"]["artifact_semantic_identity_sha256"] = (
                semantic_identity_sha256
            )
            entry["pre_timing_verification"] = verification
            process_record = {
                "started_at_utc": "2026-07-24T00:00:00+00:00",
                "finished_at_utc": "2026-07-24T00:00:01+00:00",
                "wall_seconds": 12.0,
                "process_id": 1000 + int(entry["schedule_index"]),
                "operation": "profile",
                "mode": mode,
                "payload_sha256": hashlib.sha256(
                    f"profile-payload-{entry['schedule_index']}".encode()
                ).hexdigest(),
            }
            process_record["content_sha256"] = benchmark._canonical_sha256(
                process_record
            )
            timing_sources = {
                "wall": "runtime_core_repeated_wall_time",
                "evaluator": "diagnostic",
            }
            environment = {
                "wall_time_source": "runtime_core_repeated_wall_time",
                "wall_time_sample_pass": "runtime._benchmark_f64_wall_time",
            }
            worker_measurement = {
                "batch_size": batch_size,
                "sample_count": raw_blocks["block_count"],
                "repetitions_per_sample": raw_blocks["repetitions_per_block"],
                "evaluation_count": raw_blocks["evaluation_count"],
                "evaluated_point_count": raw_blocks["evaluated_point_count"],
                "wall_seconds_per_point": wall,
                "inner_native_wall_blocks": raw_blocks,
                "benchmark_runner_sample_count": 20,
                "benchmark_runner_repetitions_per_sample": 1,
                "benchmark_runner_evaluation_count": 20,
                "benchmark_runner_evaluated_point_count": 20 * batch_size,
                "benchmark_runner_wall_seconds_per_point": wall,
                "timing_sources": timing_sources,
                "environment": environment,
                "interrupted": False,
            }
            sample = {
                "schedule_index": entry["schedule_index"],
                "round": entry["round"],
                "worker_command": entry["worker_invocation"]["command"],
                "worker_invocation": entry["worker_invocation"],
                "worker_process_record": process_record,
                "peak_rss_after_cold_load": _resource_peak(),
                "peak_rss_after_profile": _resource_peak(self_peak_bytes=2048),
                "pre_timing_verification": verification,
                "post_timing_loaded_runtime_artifact": {
                    "kind": "pyamplicol-loaded-runtime-artifact-verification",
                    "schema_version": 1,
                    "phase": "after-timing",
                    "checked_at_utc": "2026-07-24T00:00:01+00:00",
                    "expected_artifact_id": "a" * 64,
                    "loaded_artifact_id": "a" * 64,
                    "passes": True,
                },
                "cold_load_seconds": 0.01 + 0.001 * int(entry["round"]),
                "lane_contract_sha256": lane_contract_sha256,
                "timing_configuration": {
                    "minimum_internal_samples": (benchmark.MIN_AUTHORITATIVE_SAMPLES),
                    "warmup_runs": 2,
                    "target_runtime_seconds": 5.0,
                },
                "worker_measurement": worker_measurement,
                "internal_sample_count": raw_blocks["block_count"],
                "repetitions_per_sample": raw_blocks["repetitions_per_block"],
                "evaluation_count": raw_blocks["evaluation_count"],
                "evaluated_point_count": raw_blocks["evaluated_point_count"],
                "wall_seconds_per_point": wall,
                "inner_native_wall_blocks": raw_blocks,
                "timing_sources": timing_sources,
                "environment": environment,
                "interrupted": False,
            }
            retained_result_record = benchmark._retained_profile_worker_result_record(
                sample,
                upstream_result_record=entry["worker_result_record"],
            )
            sample["worker_result_record"] = retained_result_record
            entry["worker_result_record"] = retained_result_record
            entry["worker_result_sha256"] = retained_result_record[
                "addressed_payload_sha256"
            ]
            samples.append(sample)
        wall_values = [float(sample["wall_seconds_per_point"]) for sample in samples]
        median = statistics.median(wall_values)
        mad = statistics.median(abs(value - median) for value in wall_values)
        measurements.append(
            {
                "batch_size": batch_size,
                "sample_count": len(samples),
                "subprocess_sample_count": len(samples),
                "wall_seconds_per_point": median,
                "wall_seconds_per_point_median": median,
                "wall_seconds_per_point_mad": mad,
                "statistics_contract": "subprocess-median-and-raw-mad-v1",
                "subprocess_samples": samples,
                "interrupted": False,
            }
        )
    return {
        "process_id": "process-1",
        "process_expression": process_expression,
        "selector_contract": selector_contract,
        "validation": validation_record,
        "mode": mode,
        "artifact_semantic_identity": semantic_identity,
        "artifact_semantic_identity_sha256": semantic_identity_sha256,
        "profiles": measurements,
    }


def _passing_profiles(
    schedule: dict[str, object] | None = None,
    *,
    semantic_identity: dict[str, object] | None = None,
    selector_contract: dict[str, object] | None = None,
    fixture: dict[str, object] | None = None,
) -> dict[str, dict[str, Any]]:
    schedule = _passing_schedule() if schedule is None else schedule
    values = (1.0 + 2.0j, 3.0 + 4.0j)
    return {
        "compiled": _profile(
            "compiled",
            values,
            schedule=schedule,
            semantic_identity=semantic_identity,
            selector_contract=selector_contract,
            fixture=fixture,
        ),
        "eager": _profile(
            "eager",
            values,
            schedule=schedule,
            semantic_identity=semantic_identity,
            selector_contract=selector_contract,
            fixture=fixture,
        ),
        "recurrence": _profile(
            "recurrence",
            values,
            schedule=schedule,
            semantic_identity=semantic_identity,
            selector_contract=selector_contract,
            fixture=fixture,
        ),
    }


def _selected_union_profiles() -> dict[str, dict[str, Any]]:
    schedule = _passing_schedule(modes=("compiled", "recurrence"))
    selector_common = {
        "color_flow_request": "1",
        "resolved_color_flow_id": None,
        "helicity_request": benchmark.MIGRATION_UNION_SELECTED_HELICITY_ID,
        "resolved_helicity_id": benchmark.MIGRATION_UNION_SELECTED_HELICITY_ID,
        "color_flow_count": benchmark.MIGRATION_UNION_PHYSICAL_COLOR_COUNT,
        "helicity_count": 2,
        "workload": "all-flows/runtime-selected-single-helicity",
    }
    profiles: dict[str, dict[str, Any]] = {}
    for mode, secondary_zero in (("compiled", True), ("recurrence", False)):
        semantic_identity = _artifact_semantic_identity(
            color_count=benchmark.MIGRATION_UNION_PHYSICAL_COLOR_COUNT,
            helicity_id=benchmark.MIGRATION_UNION_SELECTED_HELICITY_ID,
            helicity_values=benchmark.MIGRATION_UNION_SELECTED_HELICITY_VALUES,
            secondary_helicity_structural_zero=secondary_zero,
        )
        profiles[mode] = _profile(
            mode,
            (1.0e-20 + 2.0e-21j, 3.0e-20 + 4.0e-21j),
            schedule=schedule,
            semantic_identity=semantic_identity,
            selector_contract={
                **selector_common,
                "structural_zero_helicity_count": int(secondary_zero),
            },
            process_expression=benchmark.MIGRATION_PROCESS_CANONICAL,
        )
    return profiles


def _rebind_sample_result(
    schedule: dict[str, object],
    sample: dict[str, Any],
) -> None:
    previous_record = sample["worker_result_record"]
    assert isinstance(previous_record, dict)
    retained_record = benchmark._retained_profile_worker_result_record(
        sample,
        upstream_result_record=previous_record,
    )
    sample["worker_result_record"] = retained_record
    entries = schedule["entries"]
    assert isinstance(entries, list)
    entry = next(
        item
        for item in entries
        if isinstance(item, dict) and item["schedule_index"] == sample["schedule_index"]
    )
    entry["worker_result_record"] = retained_record
    entry["worker_result_sha256"] = retained_record["addressed_payload_sha256"]


def test_preserved_worker_result_survives_driver_enrichment() -> None:
    command = benchmark._command_identity(("python", "worker"))
    invocation = {
        "started_at_utc": "2026-07-24T00:00:00+00:00",
        "finished_at_utc": "2026-07-24T00:00:03+00:00",
        "wall_seconds": 3.0,
        "command": command,
    }
    invocation["content_sha256"] = benchmark._canonical_sha256(invocation)
    operation_payload = {
        "mode": "compiled",
        "model_source": {"kind": "test"},
    }
    process_record = {
        "started_at_utc": "2026-07-24T00:00:01+00:00",
        "finished_at_utc": "2026-07-24T00:00:02+00:00",
        "wall_seconds": 1.0,
        "process_id": 123,
        "operation": "generate",
        "mode": "compiled",
        "payload_sha256": benchmark._canonical_sha256(operation_payload),
    }
    process_record["content_sha256"] = benchmark._canonical_sha256(process_record)
    payload = {
        **operation_payload,
        "worker_command": command,
        "worker_invocation": invocation,
        "worker_process_record": process_record,
    }
    result_record = {
        "recorded_at_utc": "2026-07-24T00:00:04+00:00",
        "addressed_payload_sha256": benchmark._canonical_sha256(payload),
        "worker_process_record_sha256": process_record["content_sha256"],
        "worker_invocation_sha256": invocation["content_sha256"],
    }
    result_record["content_sha256"] = benchmark._canonical_sha256(result_record)
    returned = {**payload, "worker_result_record": result_record}
    evidence = benchmark._preserved_worker_result_evidence(returned)
    forged = copy.deepcopy(returned)
    forged_process = forged["worker_process_record"]
    forged_process["payload_sha256"] = "b" * 64
    forged_process["content_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in forged_process.items() if key != "content_sha256"}
    )
    forged_record = forged["worker_result_record"]
    forged_record["worker_process_record_sha256"] = forged_process["content_sha256"]
    forged_payload = {
        key: value for key, value in forged.items() if key != "worker_result_record"
    }
    forged_record["addressed_payload_sha256"] = benchmark._canonical_sha256(
        forged_payload
    )
    forged_record["content_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in forged_record.items() if key != "content_sha256"}
    )
    with pytest.raises(
        benchmark.HarnessError,
        match="content-address contract",
    ):
        benchmark._preserved_worker_result_evidence(forged)
    returned["artifact_identity"] = {"later": "driver enrichment"}
    assert evidence["payload"] == payload
    assert evidence["worker_result_record"]["addressed_payload_sha256"] == (
        benchmark._canonical_sha256(evidence["payload"])
    )
    assert evidence["content_sha256"] == benchmark._canonical_sha256(
        {key: value for key, value in evidence.items() if key != "content_sha256"}
    )


def test_pairwise_validation_includes_eager_and_every_point() -> None:
    summary = benchmark._pairwise_profile_validation(_passing_profiles())
    comparisons = summary["comparisons"]
    assert set(comparisons) == {
        "compiled__eager",
        "compiled__recurrence",
        "eager__recurrence",
    }
    assert len(comparisons["compiled__eager"]["point_comparisons"]) == 2
    assert summary["selectors_match"]
    assert summary["fixtures_match"]
    assert summary["lane_validation_passes"]
    assert summary["pairwise_validation_passes"]
    assert summary["passes"]


def test_numerical_tolerance_is_additive_not_absolute_or_relative() -> None:
    expected = 1.0e-3
    actual = expected + 1.5e-15
    absolute = abs(actual - expected)
    old_relative = absolute / max(abs(actual), abs(expected), 1.0e-300)

    assert absolute > benchmark.ATOL
    assert old_relative > benchmark.RTOL
    assert benchmark._within_numerical_tolerance(actual, expected)
    assert not benchmark._within_numerical_tolerance(2.0 * benchmark.ATOL, 0.0)

    comparison = benchmark._comparison(
        "actual",
        [[actual, 0.0]],
        "expected",
        [[expected, 0.0]],
    )
    assert comparison["point_comparisons"][0]["relative_difference"] == old_relative
    assert comparison["passes"]


def test_lane_evidence_recomputation_uses_additive_tolerance() -> None:
    expected = 1.0e-3
    actual = expected + 1.5e-15
    profiles = {
        mode: _profile(mode, (complex(expected), 2.0e-3 + 0.0j))
        for mode in ("compiled", "eager", "recurrence")
    }
    absolute = abs(actual - expected)
    relative = absolute / max(abs(actual), abs(expected), 1.0e-300)
    for profile in profiles.values():
        validation = profile["validation"]
        validation["selected_totals"][0] = [actual, 0.0]
        comparison = validation["point_comparisons"][0]
        comparison["selected_total"] = [actual, 0.0]
        comparison["absolute_difference"] = absolute
        comparison["relative_difference"] = relative
        comparison["passes"] = True
        validation["maximum_absolute_difference"] = absolute
        validation["maximum_relative_difference"] = relative

    summary = benchmark._pairwise_profile_validation(profiles)
    assert summary["lane_validation_passes"]
    assert summary["pairwise_validation_passes"]
    assert summary["passes"]


def test_component_sum_and_pairwise_components_use_additive_tolerance() -> None:
    expected = 1.0e-3
    actual = expected + 1.5e-15
    profiles = {
        mode: _profile(mode, (complex(expected), 2.0e-3 + 0.0j))
        for mode in ("compiled", "eager", "recurrence")
    }
    profiles["eager"]["validation"]["resolved_components"][0][0] = [actual, 0.0]

    summary = benchmark._pairwise_profile_validation(profiles)
    assert summary["lane_validation_passes"]
    assert summary["resolved_component_comparisons"]["compiled__eager"]["passes"]
    assert summary["resolved_component_comparisons"]["eager__recurrence"]["passes"]
    assert summary["pairwise_validation_passes"]
    assert summary["passes"]


def test_selected_union_projection_permits_only_nonselected_zero_metadata() -> None:
    arguments = _arguments(
        "--process-expression",
        benchmark.MIGRATION_PROCESS_REQUEST,
        "--lc-flow-layout",
        "all-flow-union",
        "--helicity",
        benchmark.MIGRATION_UNION_SELECTED_HELICITY_ID,
    )
    profiles = _selected_union_profiles()

    strict_semantics = benchmark._profile_artifact_semantic_contract(profiles)
    strict_numerical = benchmark._pairwise_profile_validation(profiles)
    assert strict_semantics["lanes_match"] is False
    assert strict_numerical["selectors_match"] is False

    semantics = benchmark._profile_artifact_semantic_contract(
        profiles,
        arguments=arguments,
    )
    numerical = benchmark._pairwise_profile_validation(
        profiles,
        arguments=arguments,
    )
    preflight = benchmark._validation_preflight_contract(arguments, profiles)

    assert semantics["comparison_policy"].endswith("selected-helicity-v1")
    assert semantics["passes"]
    assert semantics["lanes_match"]
    assert numerical["selector_comparison_policy"].endswith("selected-helicity-v1")
    assert numerical["selectors_match"]
    assert numerical["pairwise_validation_passes"]
    assert preflight["passes"]


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("--process-expression", "u u~ > Z g g g g g g"),
        ("--lc-flow-layout", "topology-replay"),
        ("--helicity", "1"),
    ),
)
def test_selected_union_projection_is_not_used_outside_exact_route(
    argument: str,
    value: str,
) -> None:
    values = [
        "--process-expression",
        benchmark.MIGRATION_PROCESS_REQUEST,
        "--lc-flow-layout",
        "all-flow-union",
        "--helicity",
        benchmark.MIGRATION_UNION_SELECTED_HELICITY_ID,
    ]
    option_index = values.index(argument)
    values[option_index + 1] = value
    arguments = _arguments(*values)
    profiles = _selected_union_profiles()

    semantics = benchmark._profile_artifact_semantic_contract(
        profiles,
        arguments=arguments,
    )
    numerical = benchmark._pairwise_profile_validation(
        profiles,
        arguments=arguments,
    )

    assert semantics["comparison_policy"] == "strict-exact-v1"
    assert semantics["lanes_match"] is False
    assert numerical["selector_comparison_policy"] == "strict-exact-v1"
    assert numerical["selectors_match"] is False


def test_selected_union_projection_rejects_nonselected_helicity_values() -> None:
    arguments = _arguments(
        "--process-expression",
        benchmark.MIGRATION_PROCESS_REQUEST,
        "--lc-flow-layout",
        "all-flow-union",
        "--helicity",
        benchmark.MIGRATION_UNION_SELECTED_HELICITY_ID,
    )
    profiles = _selected_union_profiles()
    recurrence = profiles["recurrence"]
    identity = recurrence["artifact_semantic_identity"]
    helicities = identity["physical_helicities"]
    entries = copy.deepcopy(helicities["ordered_entries"])
    entries[1]["values"][0] = 7
    helicities["ordered_entries"] = entries
    helicities["ordered_entries_sha256"] = benchmark._canonical_sha256(entries)
    recurrence["artifact_semantic_identity_sha256"] = benchmark._canonical_sha256(
        identity
    )

    semantics = benchmark._profile_artifact_semantic_contract(
        profiles,
        arguments=arguments,
    )

    assert semantics["passes"] is False
    assert semantics["lanes_match"] is False
    assert semantics["errors"]


def test_selected_union_projection_rejects_selected_coefficient_mismatch() -> None:
    arguments = _arguments(
        "--process-expression",
        benchmark.MIGRATION_PROCESS_REQUEST,
        "--lc-flow-layout",
        "all-flow-union",
        "--helicity",
        benchmark.MIGRATION_UNION_SELECTED_HELICITY_ID,
    )
    profiles = _selected_union_profiles()
    recurrence = profiles["recurrence"]
    identity = recurrence["artifact_semantic_identity"]
    helicities = identity["physical_helicities"]
    entries = copy.deepcopy(helicities["ordered_entries"])
    selected = next(
        entry
        for entry in entries
        if entry["id"] == benchmark.MIGRATION_UNION_SELECTED_HELICITY_ID
    )
    selected["coefficient"] = 2.0
    helicities["ordered_entries"] = entries
    helicities["ordered_entries_sha256"] = benchmark._canonical_sha256(entries)
    recurrence["artifact_semantic_identity_sha256"] = benchmark._canonical_sha256(
        identity
    )

    semantics = benchmark._profile_artifact_semantic_contract(
        profiles,
        arguments=arguments,
    )

    assert semantics["passes"] is False
    assert semantics["lanes_match"] is False


def test_selected_union_preflight_rejects_resolved_component_drift() -> None:
    arguments = _arguments(
        "--process-expression",
        benchmark.MIGRATION_PROCESS_REQUEST,
        "--lc-flow-layout",
        "all-flow-union",
        "--helicity",
        benchmark.MIGRATION_UNION_SELECTED_HELICITY_ID,
    )
    profiles = _selected_union_profiles()
    components = profiles["recurrence"]["validation"]["resolved_components"]
    assert isinstance(components, list)
    components[0][0] = [9.0, 0.0]

    with pytest.raises(
        benchmark.HarnessError,
        match="failed before timing",
    ):
        benchmark._require_validation_preflight(arguments, profiles)


def test_pairwise_validation_fails_when_only_a_later_eager_point_differs() -> None:
    profiles = _passing_profiles()
    profiles["eager"] = _profile("eager", (1.0 + 2.0j, 3.1 + 4.0j))
    summary = benchmark._pairwise_profile_validation(profiles)
    comparison = summary["comparisons"]["compiled__eager"]
    assert comparison["point_comparisons"][0]["passes"]
    assert not comparison["point_comparisons"][1]["passes"]
    assert not summary["pairwise_validation_passes"]
    assert not summary["passes"]


def test_pairwise_validation_fails_on_resolved_component_difference() -> None:
    profiles = _passing_profiles()
    eager_components = profiles["eager"]["validation"]["resolved_components"]
    assert isinstance(eager_components, list)
    eager_components[0][0] = [0.5, 2.0]
    eager_components[0][1] = [0.5, 0.0]
    summary = benchmark._pairwise_profile_validation(profiles)
    comparison = summary["resolved_component_comparisons"]["compiled__eager"]
    assert not comparison["passes"]
    assert not summary["pairwise_validation_passes"]
    assert not summary["passes"]


def test_pairwise_validation_fails_on_resolved_axis_difference() -> None:
    profiles = _passing_profiles()
    profiles["eager"]["validation"]["resolved_helicity_ids"][0] = "h:+1,+1"
    summary = benchmark._pairwise_profile_validation(profiles)
    comparison = summary["resolved_component_comparisons"]["compiled__eager"]
    assert not comparison["axes_match"]
    assert not comparison["passes"]
    assert not summary["passes"]


def test_pairwise_validation_fails_on_fixture_mismatch() -> None:
    profiles = _passing_profiles()
    profiles["eager"] = _profile(
        "eager",
        (1.0 + 2.0j, 3.0 + 4.0j),
        fixture=_fixture(points_sha256="d" * 64),
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    assert not summary["fixtures_match"]
    assert not summary["passes"]


@pytest.mark.parametrize("value", ("false", 1))
def test_pairwise_validation_requires_boolean_lane_pass(value: object) -> None:
    profiles = _passing_profiles()
    profiles["compiled"]["validation"]["passes"] = value
    with pytest.raises(benchmark.HarnessError, match="summary is inconsistent"):
        benchmark._pairwise_profile_validation(profiles)


def test_pairwise_validation_requires_fixture_point_count() -> None:
    profiles = _passing_profiles()
    for profile in profiles.values():
        profile["validation"]["selected_totals"] = [[1.0, 2.0]]
    with pytest.raises(
        benchmark.HarnessError,
        match="inventory disagrees with fixture",
    ):
        benchmark._pairwise_profile_validation(profiles)


def test_pairwise_validation_rejects_boolean_complex_components() -> None:
    profiles = _passing_profiles()
    for profile in profiles.values():
        profile["validation"]["selected_totals"] = [
            [True, False],
            [True, False],
        ]
    with pytest.raises(
        benchmark.HarnessError,
        match="invalid complex value",
    ):
        benchmark._pairwise_profile_validation(profiles)


def test_pairwise_validation_recomputes_lane_point_evidence() -> None:
    profiles = _passing_profiles()
    profiles["compiled"]["validation"]["resolved_sums"][0] = [9.0, 9.0]
    with pytest.raises(
        benchmark.HarnessError,
        match="point evidence is inconsistent",
    ):
        benchmark._pairwise_profile_validation(profiles)


@pytest.mark.parametrize(
    ("mode", "prepared", "expected"),
    [
        ("compiled", False, 3),
        ("eager", False, benchmark.PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL),
        ("recurrence", False, benchmark.PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL),
        ("compiled", True, 3),
        ("eager", True, benchmark.PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL),
        ("recurrence", True, benchmark.PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL),
    ],
)
def test_effective_jit_level_distinguishes_source_and_prepared_execution(
    tmp_path: Path,
    mode: str,
    prepared: bool,
    expected: int,
) -> None:
    values = ["--jit-optimization-level", "3"]
    if prepared:
        values.extend(("--prepared-model", str(tmp_path / "model.pyamplicol-model")))
    arguments = _arguments(*values)
    assert (
        benchmark._expected_effective_jit_optimization_level(arguments, mode=mode)
        == expected
    )


def test_partial_lane_capture_never_vacuously_passes() -> None:
    arguments = _arguments("--mode", "recurrence")
    schedule = _passing_schedule(modes=("recurrence",))
    profiles = {
        "recurrence": _profile(
            "recurrence",
            (1.0 + 2.0j, 3.0 + 4.0j),
            schedule=schedule,
        )
    }
    summary = benchmark._pairwise_profile_validation(profiles)
    assert summary["lane_validation_passes"]
    assert summary["pairwise_validation_passes"] is None

    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert not capture["complete"]
    assert capture["passes"] is None
    assert capture["missing_modes"] == ["compiled"]
    assert capture["eager_diagnostic"]["requested"] is False


def _recurrence_runtime_acceptance_fixture(
    *,
    semantic_identity: dict[str, object] | None = None,
    gluon_count: int = 6,
    lc_flow_layout: str = "topology-replay",
) -> dict[str, Any]:
    process_expression = "d d~ > Z" + " g" * gluon_count
    arguments = _arguments(
        "--mode",
        "recurrence",
        "--gluon-count",
        str(gluon_count),
        "--process-expression",
        process_expression,
        "--lc-flow-layout",
        lc_flow_layout,
    )
    source_identity = {"revision": "1" * 40, "dirty": False}
    runtime_provenance = {"native_extension": {"sha256": "2" * 64}}
    schedule = _passing_schedule(
        modes=("recurrence",),
        lc_flow_layout=lc_flow_layout,
    )
    schedule_entries = schedule["entries"]
    assert isinstance(schedule_entries, list)
    for entry in schedule_entries:
        verification = entry["pre_timing_verification"]
        for identities in (verification["expected"], verification["observed"]):
            identities["source_identity_sha256"] = benchmark._canonical_sha256(
                source_identity
            )
            identities["runtime_provenance_sha256"] = benchmark._canonical_sha256(
                runtime_provenance
            )
    selector_contract = (
        None
        if lc_flow_layout == "topology-replay"
        else {
            "color_flow_request": "1",
            "resolved_color_flow_id": None,
            "helicity_request": "1",
            "resolved_helicity_id": "h:-1,+1",
            "color_flow_count": 1,
            "helicity_count": 2,
            "structural_zero_helicity_count": 0,
            "workload": "all-flows/runtime-selected-single-helicity",
        }
    )
    profile = _profile(
        "recurrence",
        (1.0 + 2.0j, 3.0 + 4.0j),
        schedule=schedule,
        semantic_identity=semantic_identity,
        selector_contract=selector_contract,
        process_expression=process_expression,
    )
    profiles = {"recurrence": profile}
    validation_summary = benchmark._pairwise_profile_validation(profiles)
    semantic_identity = profile["artifact_semantic_identity"]
    semantic_sha256 = profile["artifact_semantic_identity_sha256"]
    generation = {
        "recurrence": {
            "mode": "recurrence",
            "generation_reused": False,
            "artifact_identity": {
                "artifact_id": "a" * 64,
                "semantic_identity": copy.deepcopy(semantic_identity),
                "semantic_identity_sha256": semantic_sha256,
            },
            "artifact_semantic_identity_sha256": semantic_sha256,
            "effective_contract": {
                "execution_mode": "recurrence",
                "backend": "jit",
                "jit_optimization_level": (
                    benchmark.PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL
                ),
                "color_accuracy": "lc",
                "lc_flow_layout": lc_flow_layout,
            },
        }
    }
    return {
        "arguments": arguments,
        "source_identity": source_identity,
        "runtime_provenance": runtime_provenance,
        "generation": generation,
        "profiles": profiles,
        "validation_summary": validation_summary,
        "profile_schedule": schedule,
    }


def _recurrence_runtime_acceptance_record(
    fixture: dict[str, Any],
) -> dict[str, object]:
    return benchmark._recurrence_runtime_acceptance(**fixture)


@pytest.mark.parametrize("gluon_count", (5, 6))
@pytest.mark.parametrize(
    "lc_flow_layout",
    ("topology-replay", "all-flow-union"),
)
def test_recurrence_runtime_acceptance_passes_and_is_content_addressed(
    gluon_count: int,
    lc_flow_layout: str,
) -> None:
    fixture = _recurrence_runtime_acceptance_fixture(
        gluon_count=gluon_count,
        lc_flow_layout=lc_flow_layout,
    )
    record = _recurrence_runtime_acceptance_record(fixture)

    assert record["kind"] == benchmark.RECURRENCE_RUNTIME_ACCEPTANCE_KIND
    assert record["schema_version"] == benchmark.RECURRENCE_RUNTIME_ACCEPTANCE_SCHEMA
    assert record["accepted"] is True
    assert record["status"] == "accepted"
    assert record["errors"] == []
    assert record["contracts"]["measurement"]["passes"] is True
    assert record["contracts"]["artifact_semantics"]["passes"] is True
    assert record["contracts"]["lane_validation"] == {
        "passes": True,
        "lane_validation_passes": True,
        "selectors_match": True,
        "fixtures_match": True,
        "pairwise_validation_passes": None,
        "summary_sha256": benchmark._canonical_sha256(fixture["validation_summary"]),
    }
    assert record["contracts"]["worker_rss"]["passes"] is True
    assert record["contracts"]["worker_rss"]["worker_count"] == 21
    bindings = record["bindings"]
    assert bindings["generation_artifact_id"] == "a" * 64
    assert (
        bindings["generation_artifact_semantic_identity_sha256"]
        == fixture["profiles"]["recurrence"]["artifact_semantic_identity_sha256"]
    )
    assert bindings["profile_schedule_sha256"] == benchmark._canonical_sha256(
        fixture["profile_schedule"]
    )
    assert bindings["recurrence_profile_sha256"] == benchmark._canonical_sha256(
        fixture["profiles"]["recurrence"]
    )
    assert record["content_sha256"] == benchmark._canonical_sha256(
        {key: value for key, value in record.items() if key != "content_sha256"}
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("modes", ["compiled", "recurrence"], "only the recurrence mode"),
        ("generation_only", True, "completed profiling"),
        ("batch_size", [1, 1024, 128], "exact batches"),
        ("warmup_runs", 1, "two warm-up"),
        ("target_runtime", 4.999, "five-second"),
        ("minimum_samples", 6, "at least seven"),
        ("subprocess_samples", 6, "seven through twenty-one"),
        ("subprocess_samples", 22, "seven through twenty-one"),
        ("gluon_count", 4, "ladder process"),
        ("process_expression", "u u~ > Z g", "ladder process"),
        ("specialize_flow_at_generation", True, "forbids generation"),
    ),
)
def test_recurrence_runtime_acceptance_rejects_noncanonical_configuration(
    field: str,
    value: object,
    error: str,
) -> None:
    fixture = _recurrence_runtime_acceptance_fixture()
    setattr(fixture["arguments"], field, value)
    record = _recurrence_runtime_acceptance_record(fixture)
    assert record["accepted"] is False
    assert record["status"] == "rejected"
    assert any(error in item for item in record["errors"])


@pytest.mark.parametrize(
    "failure",
    (
        "schedule",
        "measurement",
        "lane_validation",
        "selector",
        "fixture",
        "profile_inventory",
    ),
)
def test_recurrence_runtime_acceptance_reuses_authenticated_contracts(
    failure: str,
) -> None:
    fixture = _recurrence_runtime_acceptance_fixture()
    profile = fixture["profiles"]["recurrence"]
    if failure == "schedule":
        del fixture["profile_schedule"]["entries"][0]["worker_invocation"]
    elif failure == "measurement":
        profile["profiles"][0]["subprocess_samples"][0]["interrupted"] = True
    elif failure == "lane_validation":
        profile["validation"]["passes"] = False
    elif failure == "selector":
        profile["selector_contract"]["color_flow_request"] = "forged"
    elif failure == "fixture":
        profile["validation"]["fixture"]["points_sha256"] = "invalid"
    else:
        fixture["profiles"]["compiled"] = copy.deepcopy(profile)
    record = _recurrence_runtime_acceptance_record(fixture)
    assert record["accepted"] is False
    if failure == "schedule":
        assert record["contracts"]["measurement"]["schedule"]["passes"] is False
    elif failure == "profile_inventory":
        assert any(
            "profile inventory is not recurrence-only" in item
            for item in record["errors"]
        )
    else:
        assert (
            record["contracts"]["measurement"]["passes"] is False
            or record["contracts"]["lane_validation"]["passes"] is False
        )


@pytest.mark.parametrize(
    "failure",
    (
        "incomplete_axes",
        "specialized_axes",
        "artifact_id",
        "semantic_digest",
        "effective_contract",
        "generation_reused",
        "worker_binding",
        "source",
        "runtime",
    ),
)
def test_recurrence_runtime_acceptance_rejects_unbound_semantics_or_provenance(
    failure: str,
) -> None:
    semantic_identity = (
        _artifact_semantic_identity(complete=False)
        if failure == "incomplete_axes"
        else (
            _artifact_semantic_identity(specialized_axes=("color_flow",))
            if failure == "specialized_axes"
            else None
        )
    )
    fixture = _recurrence_runtime_acceptance_fixture(
        semantic_identity=semantic_identity
    )
    generation = fixture["generation"]["recurrence"]
    if failure == "artifact_id":
        generation["artifact_identity"]["artifact_id"] = "invalid"
    elif failure == "semantic_digest":
        generation["artifact_identity"]["semantic_identity_sha256"] = "3" * 64
    elif failure == "effective_contract":
        generation["effective_contract"]["backend"] = "forged"
    elif failure == "generation_reused":
        generation["generation_reused"] = True
    elif failure == "worker_binding":
        fixture["profile_schedule"]["entries"][0]["pre_timing_verification"][
            "expected"
        ]["artifact_id"] = "4" * 64
    elif failure == "source":
        fixture["source_identity"] = {}
    elif failure == "runtime":
        fixture["runtime_provenance"] = {}
    record = _recurrence_runtime_acceptance_record(fixture)
    assert record["accepted"] is False
    if failure in {"incomplete_axes", "specialized_axes"}:
        assert any(
            "complete unspecialized physical axes" in item for item in record["errors"]
        )
    if failure == "worker_binding":
        assert any("profile workers are not bound" in item for item in record["errors"])


@pytest.mark.parametrize(
    "failure",
    (
        "missing",
        "extra_key",
        "source",
        "boolean",
        "negative",
        "maximum",
        "post_below_cold",
    ),
)
def test_recurrence_runtime_acceptance_validates_every_worker_rss_mapping(
    failure: str,
) -> None:
    fixture = _recurrence_runtime_acceptance_fixture()
    sample = fixture["profiles"]["recurrence"]["profiles"][0]["subprocess_samples"][0]
    if failure == "missing":
        del sample["peak_rss_after_cold_load"]
    elif failure == "extra_key":
        sample["peak_rss_after_cold_load"]["unexpected"] = 1
    elif failure == "source":
        sample["peak_rss_after_cold_load"]["source"] = "forged"
    elif failure == "boolean":
        sample["peak_rss_after_cold_load"]["self_peak_bytes"] = True
    elif failure == "negative":
        sample["peak_rss_after_cold_load"]["self_peak_bytes"] = -1
    elif failure == "maximum":
        sample["peak_rss_after_cold_load"]["observed_lower_bound_bytes"] = 1
    else:
        sample["peak_rss_after_profile"] = _resource_peak(self_peak_bytes=512)
    _rebind_sample_result(fixture["profile_schedule"], sample)
    record = _recurrence_runtime_acceptance_record(fixture)
    assert record["accepted"] is False
    rss_contract = record["contracts"]["worker_rss"]
    assert rss_contract["passes"] is False
    if failure == "post_below_cold":
        assert any("below cold-load" in item for item in rss_contract["errors"])


def test_complete_three_lane_capture_can_pass_but_m0_remains_fail_closed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"]
    assert capture["passes"]
    assert capture["schema_version"] == benchmark.CAPTURE_ACCEPTANCE_SCHEMA
    assert schedule["schema_version"] == 2

    manifest = benchmark._milestone0_acceptance_manifest(arguments, capture)
    assert manifest["kind"] == benchmark.M0_ACCEPTANCE_KIND
    assert manifest["schema_version"] == benchmark.M0_ACCEPTANCE_SCHEMA
    assert not manifest["accepted"]
    assert manifest["status"] == "incomplete"
    assert {"kind": "layout_capture", "value": "all-flow-union"} in manifest[
        "missing_evidence"
    ]
    assert {"kind": "external_lane", "value": "amplicol"} in manifest[
        "missing_evidence"
    ]
    assert "separate fail-closed orchestrator" in manifest["integration_step"]


def test_generation_specialized_axes_are_not_authoritative() -> None:
    arguments = _arguments("--specialize-flow-at-generation")
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    profiles["recurrence"] = _profile(
        "recurrence",
        (1.0 + 2.0j, 3.0 + 4.0j),
        schedule=schedule,
        specialized_axes=("color_flow",),
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["evidence_complete"]
    assert not capture["authoritative_eligible"]
    assert not capture["complete"]
    assert capture["passes"] is None
    assert capture["generation_specialized_axes_by_mode"] == {
        "recurrence": ["color_flow"]
    }


def test_eager_normalization_mismatch_is_diagnostic_only() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    eager_identity = profiles["eager"]["artifact_semantic_identity"]
    eager_identity["normalization"] = {"average_factor": 72, "color_factor": 3}
    eager_identity["normalization_sha256"] = benchmark._canonical_sha256(
        eager_identity["normalization"]
    )
    profiles["eager"]["artifact_semantic_identity_sha256"] = (
        benchmark._canonical_sha256(eager_identity)
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["authoritative_eligible"]
    assert capture["artifact_semantic_contract"]["lanes_match"]
    assert capture["complete"]
    diagnostic = capture["eager_diagnostic"]
    assert diagnostic["complete"] is False
    assert diagnostic["passes"] is None
    assert diagnostic["artifact_semantic_contract"]["lanes_match"] is False


def test_profile_semantic_identity_digest_is_required() -> None:
    profiles = _passing_profiles()
    eager_identity = profiles["eager"]["artifact_semantic_identity"]
    eager_identity["normalization"]["average_factor"] = 72
    eager_identity["normalization_sha256"] = benchmark._canonical_sha256(
        eager_identity["normalization"]
    )
    contract = benchmark._profile_artifact_semantic_contract(profiles)
    assert contract["passes"] is False
    assert contract["lanes_match"] is False
    assert any("not content-addressed" in error for error in contract["errors"])


def test_physical_axis_value_is_cross_compared() -> None:
    profiles = _passing_profiles()
    eager_profile = profiles["eager"]
    eager_identity = eager_profile["artifact_semantic_identity"]
    eager_axis = eager_identity["physical_helicities"]
    eager_entries = copy.deepcopy(eager_axis["ordered_entries"])
    eager_entries[1]["values"][0] = -7
    eager_axis["ordered_entries"] = eager_entries
    eager_axis["ordered_entries_sha256"] = benchmark._canonical_sha256(eager_entries)
    eager_profile["artifact_semantic_identity_sha256"] = benchmark._canonical_sha256(
        eager_identity
    )
    contract = benchmark._profile_artifact_semantic_contract(profiles)
    assert contract["passes"] is False
    assert contract["lanes_match"] is False


def test_physical_axis_coefficient_is_cross_compared() -> None:
    profiles = _passing_profiles()
    eager_profile = profiles["eager"]
    eager_identity = eager_profile["artifact_semantic_identity"]
    eager_axis = eager_identity["physical_helicities"]
    eager_entries = copy.deepcopy(eager_axis["ordered_entries"])
    eager_entries[1]["coefficient"] = 2.0
    eager_axis["ordered_entries"] = eager_entries
    eager_axis["ordered_entries_sha256"] = benchmark._canonical_sha256(eager_entries)
    eager_profile["artifact_semantic_identity_sha256"] = benchmark._canonical_sha256(
        eager_identity
    )

    contract = benchmark._profile_artifact_semantic_contract(profiles)

    assert contract["passes"] is False
    assert contract["lanes_match"] is False


def test_empty_runtime_selector_axes_are_ineligible() -> None:
    profiles = _passing_profiles()
    eager_profile = profiles["eager"]
    eager_identity = eager_profile["artifact_semantic_identity"]
    eager_selector_semantics = copy.deepcopy(
        eager_identity["runtime_selector_semantics"]
    )
    eager_runtime_selectors = copy.deepcopy(eager_identity["runtime_selectors"])
    eager_selector_semantics["axes"] = {}
    eager_runtime_selectors["axes"] = {}
    eager_identity["runtime_selector_semantics"] = eager_selector_semantics
    eager_identity["runtime_selector_semantics_sha256"] = benchmark._canonical_sha256(
        eager_selector_semantics
    )
    eager_identity["runtime_selectors"] = eager_runtime_selectors
    eager_identity["runtime_selectors_sha256"] = benchmark._canonical_sha256(
        eager_runtime_selectors
    )
    eager_profile["artifact_semantic_identity_sha256"] = benchmark._canonical_sha256(
        eager_identity
    )
    contract = benchmark._profile_artifact_semantic_contract(profiles)
    assert contract["passes"] is False
    assert any(
        "model/selector/reduction semantic identity is invalid" in error
        for error in contract["errors"]
    )


def test_empty_profile_selector_workload_contract_is_ineligible() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    for profile in profiles.values():
        profile["selector_contract"] = {}
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert capture["measurement_contract"]["passes"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("color_flow_count", 99),
        ("resolved_color_flow_id", "flow:forged"),
        ("structural_zero_helicity_count", 1),
    ),
)
def test_profile_selector_reconciles_with_semantic_axes(
    field: str,
    value: object,
) -> None:
    arguments = _arguments()
    profile = _profile("compiled", (1.0 + 2.0j, 3.0 + 4.0j))
    selector_contract = copy.deepcopy(profile["selector_contract"])
    assert isinstance(selector_contract, dict)
    selector_contract[field] = value
    assert (
        benchmark._profile_selector_contract_matches(
            arguments,
            selector_contract,
            profile["artifact_semantic_identity"],
        )
        is False
    )


def test_eager_root_process_mismatch_is_diagnostic_only() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    profiles["eager"]["process_id"] = "wrong"
    profiles["eager"]["process_expression"] = "wrong > process"
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is True
    assert capture["measurement_contract"]["root_processes_match"] is True
    assert capture["eager_diagnostic"]["complete"] is False
    assert capture["eager_diagnostic"]["passes"] is None


def test_runtime_selector_contract_version_rejects_boolean() -> None:
    identity = _artifact_semantic_identity()
    runtime_selectors = copy.deepcopy(identity["runtime_selectors"])
    assert isinstance(runtime_selectors, dict)
    runtime_selectors["contract_version"] = True
    with pytest.raises(
        benchmark.HarnessError,
        match="runtime-selector semantics are incomplete",
    ):
        benchmark._runtime_selector_semantic_identity(
            runtime_selectors,
            color_coverage="complete",
            helicity_coverage="complete",
            artifact=Path("<test-profile>"),
        )


def test_color_axis_rejects_empty_physics_word() -> None:
    identity = _artifact_semantic_identity()
    color_axis = identity["physical_color_flows"]
    assert isinstance(color_axis, dict)
    entries = copy.deepcopy(color_axis["ordered_entries"])
    entries[0]["word"] = []
    with pytest.raises(
        benchmark.HarnessError,
        match="not complete and ordered",
    ):
        benchmark._ordered_physical_axis(
            entries,
            label="color-flow",
            require_structural_zero=False,
        )


def test_missing_reduction_order_identity_is_ineligible() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    del profiles["recurrence"]["artifact_semantic_identity"][
        "reduction_ordering_sha256"
    ]
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert not capture["artifact_semantic_contract"]["passes"]
    assert not capture["authoritative_eligible"]
    assert not capture["complete"]


def test_eager_manifest_model_identity_mismatch_is_diagnostic_only() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    eager_identity = profiles["eager"]["artifact_semantic_identity"]
    eager_model_identity = eager_identity["manifest_model_identity"]
    eager_manifest_model = copy.deepcopy(eager_model_identity["manifest"])
    eager_manifest_model["content_sha256"] = "d" * 64
    eager_common_model = {
        key: value
        for key, value in eager_manifest_model.items()
        if key != "source_kind"
    }
    eager_model_identity.update(
        {
            "manifest": eager_manifest_model,
            "manifest_sha256": benchmark._canonical_sha256(eager_manifest_model),
            "common_physics_identity": eager_common_model,
            "common_physics_identity_sha256": benchmark._canonical_sha256(
                eager_common_model
            ),
        }
    )
    profiles["eager"]["artifact_semantic_identity_sha256"] = (
        benchmark._canonical_sha256(eager_identity)
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["artifact_semantic_contract"]["passes"] is True
    assert capture["authoritative_eligible"] is True
    eager_contract = capture["eager_diagnostic"]["artifact_semantic_contract"]
    assert eager_contract["passes"] is False
    assert eager_contract["lanes_match"] is False


def test_eager_structural_zero_entry_mismatch_is_diagnostic_only() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    eager_identity = profiles["eager"]["artifact_semantic_identity"]
    eager_helicities = eager_identity["physical_helicities"]
    eager_entries = copy.deepcopy(eager_helicities["ordered_entries"])
    eager_entries[1]["structural_zero"] = True
    eager_helicities["ordered_entries"] = eager_entries
    eager_helicities["ordered_entries_sha256"] = benchmark._canonical_sha256(
        eager_entries
    )
    profiles["eager"]["artifact_semantic_identity_sha256"] = (
        benchmark._canonical_sha256(eager_identity)
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["artifact_semantic_contract"]["passes"] is True
    assert capture["authoritative_eligible"] is True
    assert capture["eager_diagnostic"]["artifact_semantic_contract"]["passes"] is False


def test_eager_reduction_order_mismatch_is_diagnostic_only() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    eager_identity = profiles["eager"]["artifact_semantic_identity"]
    eager_reduction = copy.deepcopy(eager_identity["reduction_ordering"])
    eager_reduction["pair_order_abi"] = "color-major-helicity-minor-v1"
    eager_identity["reduction_ordering"] = eager_reduction
    eager_identity["reduction_ordering_sha256"] = benchmark._canonical_sha256(
        eager_reduction
    )
    profiles["eager"]["artifact_semantic_identity_sha256"] = (
        benchmark._canonical_sha256(eager_identity)
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["artifact_semantic_contract"]["passes"] is True
    assert capture["artifact_semantic_contract"]["lanes_match"] is True
    assert capture["authoritative_eligible"] is True
    eager_contract = capture["eager_diagnostic"]["artifact_semantic_contract"]
    assert eager_contract["passes"] is False
    assert eager_contract["lanes_match"] is False


def test_reduction_members_must_map_to_the_group_representative() -> None:
    identity = _artifact_semantic_identity()
    logical_helicities = identity["physical_helicities"]["ordered_entries"]
    logical_colors = identity["physical_color_flows"]["ordered_entries"]
    helicity_entries = [
        {
            **entry,
            "representative_id": logical_helicities[0]["id"],
            "computed": index == 0,
            "coefficient": 1.0,
        }
        for index, entry in enumerate(logical_helicities)
    ]
    helicity_entries[1]["computed"] = True
    helicity_entries[1]["representative_id"] = "h:+1,-1"
    helicity_axis = benchmark._ordered_physical_axis(
        helicity_entries,
        label="helicity",
        require_structural_zero=True,
    )
    color_axis = benchmark._ordered_physical_axis(
        [
            {
                **entry,
                "representative_id": entry["id"],
                "computed": True,
                "coefficient": 1.0,
            }
            for entry in logical_colors
        ],
        label="color-flow",
        require_structural_zero=False,
    )
    with pytest.raises(
        benchmark.HarnessError,
        match="not closed over physical axes",
    ):
        benchmark._reduction_ordering_identity(
            {
                "kind": "lc-diagonal",
                "groups": [
                    {
                        "id": "reduction:0",
                        "physical_color_ids": ["flow:2,1"],
                        "physical_helicity_ids": ["h:-1,+1", "h:+1,-1"],
                        "representative_color_id": "flow:2,1",
                        "representative_helicity_id": "h:-1,+1",
                    }
                ],
            },
            color_axis=color_axis,
            helicity_axis=helicity_axis,
            artifact=Path("<test-profile>"),
        )


@pytest.mark.parametrize(
    ("computed", "representative_id", "structural_zero", "coefficient"),
    (
        (True, "h:-1,+1", False, 1.0),
        (True, "h:+1,-1", True, 0.0),
        (False, "h:-1,+1", False, 0.0),
    ),
)
def test_helicity_axis_rejects_invalid_mapping_states(
    computed: bool,
    representative_id: str,
    structural_zero: bool,
    coefficient: float,
) -> None:
    helicity_axis = _artifact_semantic_identity()["physical_helicities"]
    assert isinstance(helicity_axis, dict)
    logical_entries = copy.deepcopy(helicity_axis["ordered_entries"])
    entries = [
        {
            **entry,
            "representative_id": logical_entries[0]["id"],
            "computed": index == 0,
            "coefficient": 1.0,
        }
        for index, entry in enumerate(logical_entries)
    ]
    assert isinstance(entries, list)
    assert isinstance(entries[1], dict)
    entries[1].update(
        {
            "computed": computed,
            "representative_id": representative_id,
            "structural_zero": structural_zero,
            "coefficient": coefficient,
        }
    )
    with pytest.raises(
        benchmark.HarnessError,
        match="representative mapping is not closed",
    ):
        benchmark._ordered_physical_axis(
            entries,
            label="helicity",
            require_structural_zero=True,
        )


def test_eager_execution_reduction_coverage_is_diagnostic_only() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    eager_identity = profiles["eager"]["artifact_semantic_identity"]
    eager_identity["execution_reduction_coverage"] = {
        "complete": False,
        "expected_physical_pair_count": 2,
        "observed_physical_pair_count": 1,
        "errors": ["missing one execution pair"],
    }
    profiles["eager"]["artifact_semantic_identity_sha256"] = (
        benchmark._canonical_sha256(eager_identity)
    )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["artifact_semantic_contract"]["passes"] is True
    eager_contract = capture["eager_diagnostic"]["artifact_semantic_contract"]
    assert eager_contract["passes"] is False
    assert any(
        "model/selector/reduction semantic identity is invalid" in error
        for error in eager_contract["errors"]
    )
    assert capture["authoritative_eligible"] is True


def test_execution_reduction_summary_is_revalidated() -> None:
    profiles = _passing_profiles()
    eager_profile = profiles["eager"]
    eager_identity = eager_profile["artifact_semantic_identity"]
    eager_identity["execution_reduction_identity"] = {
        "kind": "forged-reduction",
    }
    eager_identity["execution_reduction_identity_sha256"] = benchmark._canonical_sha256(
        eager_identity["execution_reduction_identity"]
    )
    eager_profile["artifact_semantic_identity_sha256"] = benchmark._canonical_sha256(
        eager_identity
    )

    contract = benchmark._profile_artifact_semantic_contract(profiles)

    assert contract["passes"] is False
    assert any(
        "model/selector/reduction semantic identity is invalid" in error
        for error in contract["errors"]
    )


def test_complete_execution_reduction_coverage_rejects_impossible_counts() -> None:
    profiles = _passing_profiles()
    eager_profile = profiles["eager"]
    eager_identity = eager_profile["artifact_semantic_identity"]
    eager_identity["execution_reduction_coverage"] = {
        "complete": True,
        "expected_physical_pair_count": 1,
        "observed_physical_pair_count": 1,
        "errors": [],
    }
    eager_profile["artifact_semantic_identity_sha256"] = benchmark._canonical_sha256(
        eager_identity
    )

    contract = benchmark._profile_artifact_semantic_contract(profiles)

    assert contract["passes"] is False


def test_capture_rejects_missing_or_interrupted_timing_measurements() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    profiles["eager"]["profiles"] = profiles["eager"]["profiles"][1:]
    profiles["recurrence"]["profiles"][0]["interrupted"] = True
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert not capture["complete"]
    assert capture["passes"] is None
    contract = capture["measurement_contract"]["lanes"]
    assert not contract["recurrence"]["passes"]
    eager_contract = capture["eager_diagnostic"]["measurement_contract"]["lanes"]
    assert eager_contract["eager"]["missing_batch_sizes"] == [1]


def test_single_worker_five_sample_timing_cannot_pass_authoritative_capture() -> None:
    arguments = _arguments(
        "--subprocess-samples",
        "1",
        "--minimum-samples",
        "5",
    )
    schedule = _passing_schedule(subprocess_samples=1)
    profiles = _passing_profiles(schedule)
    for profile in profiles.values():
        for measurement in profile["profiles"]:
            measurement["subprocess_samples"][0]["internal_sample_count"] = 5
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert not capture["complete"]
    contract = capture["measurement_contract"]
    assert not contract["passes"]
    assert contract["configured_subprocess_samples"] == 1
    assert contract["configured_internal_minimum_samples"] == 5


def test_aggregate_headline_sample_count_must_match_inventory() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    profiles["compiled"]["profiles"][0]["sample_count"] = False
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert capture["measurement_contract"]["passes"] is False


@pytest.mark.parametrize(
    "failure",
    (
        "native_wall",
        "warmup",
        "internal_samples",
        "repetitions",
        "evaluation_count",
        "raw_blocks",
        "worker_result_binding",
        "median_mad",
        "schedule",
        "result_record",
    ),
)
def test_authoritative_timing_contract_fails_closed(
    failure: str,
) -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    measurement = profiles["compiled"]["profiles"][0]
    first_sample = measurement["subprocess_samples"][0]
    if failure == "native_wall":
        first_sample["timing_sources"]["wall"] = "runtime_evaluate_wall_time"
    elif failure == "warmup":
        first_sample["timing_configuration"]["warmup_runs"] = 0
    elif failure == "internal_samples":
        first_sample["internal_sample_count"] = 6
    elif failure == "repetitions":
        first_sample["repetitions_per_sample"] = 0
    elif failure == "evaluation_count":
        first_sample["evaluation_count"] = 0
    elif failure == "raw_blocks":
        first_sample["inner_native_wall_blocks"]["blocks"][0][
            "native_wall_seconds"
        ] *= 2.0
    elif failure == "worker_result_binding":
        raw_blocks = first_sample["inner_native_wall_blocks"]
        block = raw_blocks["blocks"][0]
        block["started_at_utc"] = "2026-07-24T00:00:00.5+00:00"
        block["content_sha256"] = benchmark._canonical_sha256(
            {key: value for key, value in block.items() if key != "content_sha256"}
        )
        raw_blocks["blocks_sha256"] = benchmark._canonical_sha256(raw_blocks["blocks"])
    elif failure == "median_mad":
        measurement["wall_seconds_per_point_mad"] = None
    elif failure == "schedule":
        entries = schedule["entries"]
        assert isinstance(entries, list)
        del entries[0]["worker_invocation"]
    else:
        entries = schedule["entries"]
        assert isinstance(entries, list)
        entries[0]["worker_result_record"]["recorded_at_utc"] = (
            "2026-07-24T00:00:02+00:00"
        )
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert not capture["complete"]
    assert not capture["measurement_contract"]["passes"]


def test_raw_block_boolean_counts_fail_even_when_readdressed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    block = raw_blocks["blocks"][0]
    for field in (
        "repetitions",
        "batch_size",
        "evaluation_count",
        "evaluated_point_count",
    ):
        block[field] = True
    block["content_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in block.items() if key != "content_sha256"}
    )
    raw_blocks["blocks_sha256"] = benchmark._canonical_sha256(raw_blocks["blocks"])
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "raw block content address is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("repetitions_per_block", True),
        ("evaluation_count", 7.0),
        ("evaluated_point_count", 7.0),
    ),
)
def test_raw_inventory_counts_require_positive_integers_when_readdressed(
    field: str,
    value: object,
) -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    raw_blocks[field] = value
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "raw native-wall inventory is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


def test_raw_inventory_mad_rejects_boolean_when_readdressed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    raw_blocks["wall_seconds_per_point_mad"] = False
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "raw native-wall inventory is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


def test_retained_native_wall_duration_below_target_fails_when_readdressed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    raw_blocks["observed_native_wall_seconds"] = 4.999
    raw_blocks["minimum_duration_satisfied"] = True
    _rebind_sample_result(schedule, sample)

    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )

    assert capture["complete"] is False
    assert any(
        "raw native-wall inventory is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


def test_retained_raw_calibration_tampering_fails_when_readdressed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    raw_blocks["calibration"]["duration_headroom_factor"] = 1.01
    _rebind_sample_result(schedule, sample)

    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )

    assert capture["complete"] is False
    assert any(
        "raw native-wall inventory is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


def test_raw_inventory_schema_rejects_boolean_when_readdressed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    raw_blocks["schema_version"] = True
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "raw native-wall inventory is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


def test_raw_block_timestamps_must_be_chronological_when_readdressed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    block = raw_blocks["blocks"][0]
    block["started_at_utc"] = "2026-07-24T00:00:02+00:00"
    block["content_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in block.items() if key != "content_sha256"}
    )
    raw_blocks["blocks_sha256"] = benchmark._canonical_sha256(raw_blocks["blocks"])
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "raw block content address is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


def test_raw_block_inventory_must_be_chronological_when_readdressed() -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    raw_blocks = sample["inner_native_wall_blocks"]
    first, second = raw_blocks["blocks"][:2]
    first["started_at_utc"] = "2026-07-24T00:00:00.800000+00:00"
    first["finished_at_utc"] = "2026-07-24T00:00:00.900000+00:00"
    second["started_at_utc"] = "2026-07-24T00:00:00.100000+00:00"
    second["finished_at_utc"] = "2026-07-24T00:00:00.200000+00:00"
    for block in (first, second):
        block["content_sha256"] = benchmark._canonical_sha256(
            {key: value for key, value in block.items() if key != "content_sha256"}
        )
    raw_blocks["blocks_sha256"] = benchmark._canonical_sha256(raw_blocks["blocks"])
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "raw block content address is invalid" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("minimum_internal_samples", 0),
        ("warmup_runs", True),
        ("target_runtime_seconds", -1.0),
    ),
)
def test_retained_timing_configuration_must_match_cli(
    field: str,
    value: object,
) -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    sample["timing_configuration"][field] = value
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert any(
        "is not warmed" in error
        for error in capture["measurement_contract"]["lanes"]["compiled"]["errors"]
    )


@pytest.mark.parametrize(
    "contradiction",
    (
        "round",
        "measurement_batch",
        "worker_command",
        "verified_at",
        "effective_contract",
    ),
)
def test_retained_sample_contradictions_fail_after_readdressing(
    contradiction: str,
) -> None:
    arguments = _arguments()
    schedule = _passing_schedule()
    profiles = _passing_profiles(schedule)
    sample = profiles["compiled"]["profiles"][0]["subprocess_samples"][0]
    if contradiction == "round":
        sample["round"] = False
    elif contradiction == "measurement_batch":
        sample["worker_measurement"]["batch_size"] = True
    elif contradiction == "worker_command":
        sample["worker_command"] = benchmark._command_identity(("forged-worker",))
    elif contradiction == "verified_at":
        sample["pre_timing_verification"]["verified_at_utc"] = "not-a-time"
    else:
        sample["pre_timing_verification"]["effective_contract"] = {
            "execution_mode": "eager",
            "backend": "not-jit",
        }
    _rebind_sample_result(schedule, sample)
    summary = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        summary,
        profile_schedule=schedule,
    )
    assert capture["complete"] is False
    assert capture["measurement_contract"]["passes"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schedule_index", False),
        ("round", False),
        ("batch_size", True),
    ),
)
def test_profile_schedule_rejects_boolean_slot_coordinates(
    field: str,
    value: object,
) -> None:
    schedule = _passing_schedule()
    entries = schedule["entries"]
    assert isinstance(entries, list)
    entries[0][field] = value
    contract = benchmark._profile_schedule_contract(
        schedule,
        modes=benchmark.EXECUTION_MODES,
        batch_sizes=benchmark.DEFAULT_BATCH_SIZES,
        subprocess_samples=benchmark.MIN_AUTHORITATIVE_SAMPLES,
    )
    assert contract["passes"] is False
    assert any("invalid slot coordinates" in error for error in contract["errors"])


@pytest.mark.parametrize("record_kind", ("verification", "result"))
def test_profile_schedule_rejects_boolean_nested_schema_versions(
    record_kind: str,
) -> None:
    schedule = _passing_schedule()
    entries = schedule["entries"]
    assert isinstance(entries, list)
    entry = entries[0]
    if record_kind == "verification":
        entry["pre_timing_verification"]["schema_version"] = True
    else:
        result_record = entry["worker_result_record"]
        result_record["schema_version"] = True
        result_record["content_sha256"] = benchmark._canonical_sha256(
            {
                key: value
                for key, value in result_record.items()
                if key != "content_sha256"
            }
        )
    contract = benchmark._profile_schedule_contract(
        schedule,
        modes=benchmark.EXECUTION_MODES,
        batch_sizes=benchmark.DEFAULT_BATCH_SIZES,
        subprocess_samples=benchmark.MIN_AUTHORITATIVE_SAMPLES,
    )
    assert contract["passes"] is False


def test_profile_schedule_timestamps_must_be_chronological() -> None:
    schedule = _passing_schedule()
    entries = schedule["entries"]
    assert isinstance(entries, list)
    entry = entries[0]
    invocation = entry["worker_invocation"]
    invocation["started_at_utc"] = "2026-07-24T00:00:02+00:00"
    invocation["content_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in invocation.items() if key != "content_sha256"}
    )
    result_record = entry["worker_result_record"]
    result_record["worker_invocation_sha256"] = invocation["content_sha256"]
    result_record["content_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in result_record.items() if key != "content_sha256"}
    )
    contract = benchmark._profile_schedule_contract(
        schedule,
        modes=benchmark.EXECUTION_MODES,
        batch_sizes=benchmark.DEFAULT_BATCH_SIZES,
        subprocess_samples=benchmark.MIN_AUTHORITATIVE_SAMPLES,
    )
    assert contract["passes"] is False
    assert any(
        "inverted invocation timestamps" in error for error in contract["errors"]
    )


def test_profile_schedule_rejects_reordered_noninterleaved_lanes() -> None:
    schedule = _passing_schedule()
    entries = schedule["entries"]
    assert isinstance(entries, list)
    entries[1]["mode"] = entries[0]["mode"]
    contract = benchmark._profile_schedule_contract(
        schedule,
        modes=benchmark.EXECUTION_MODES,
        batch_sizes=benchmark.DEFAULT_BATCH_SIZES,
        subprocess_samples=benchmark.MIN_AUTHORITATIVE_SAMPLES,
    )
    assert not contract["passes"]
    assert any("interleaved" in error for error in contract["errors"])


def _write_source_checkout_markers(checkout: Path) -> None:
    (checkout / ".git").mkdir(parents=True)
    for marker in benchmark._SOURCE_CHECKOUT_FILE_MARKERS:
        path = checkout / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {marker.as_posix()}\n", encoding="utf-8")


def test_source_checkout_override_defaults_to_frozen_driver_root(
    tmp_path: Path,
) -> None:
    driver_root = tmp_path / "workspace"
    driver_root.mkdir()
    assert (
        benchmark._resolve_source_checkout_root(
            driver_root=driver_root,
            environment={},
        )
        == driver_root.resolve()
    )


def test_source_checkout_override_accepts_a_nested_source_worktree(
    tmp_path: Path,
) -> None:
    driver_root = tmp_path / "workspace"
    checkout = driver_root / ".artifacts" / "baseline" / "source-test"
    driver_root.mkdir()
    _write_source_checkout_markers(checkout)
    assert (
        benchmark._resolve_source_checkout_root(
            driver_root=driver_root,
            environment={
                benchmark.SOURCE_CHECKOUT_OVERRIDE_ENV: str(checkout.resolve()),
            },
        )
        == checkout.resolve()
    )


def test_source_checkout_override_fails_closed(
    tmp_path: Path,
) -> None:
    driver_root = tmp_path / "workspace"
    driver_root.mkdir()

    with pytest.raises(benchmark.HarnessError, match="absolute source checkout"):
        benchmark._resolve_source_checkout_root(
            driver_root=driver_root,
            environment={benchmark.SOURCE_CHECKOUT_OVERRIDE_ENV: ""},
        )
    with pytest.raises(benchmark.HarnessError, match="absolute path"):
        benchmark._resolve_source_checkout_root(
            driver_root=driver_root,
            environment={benchmark.SOURCE_CHECKOUT_OVERRIDE_ENV: "baseline"},
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(benchmark.HarnessError, match="inside the driver workspace"):
        benchmark._resolve_source_checkout_root(
            driver_root=driver_root,
            environment={
                benchmark.SOURCE_CHECKOUT_OVERRIDE_ENV: str(outside.resolve()),
            },
        )

    incomplete = driver_root / "incomplete"
    incomplete.mkdir()
    with pytest.raises(benchmark.HarnessError, match="not a pyAmpliCol source"):
        benchmark._resolve_source_checkout_root(
            driver_root=driver_root,
            environment={
                benchmark.SOURCE_CHECKOUT_OVERRIDE_ENV: str(incomplete.resolve()),
            },
        )


@pytest.mark.parametrize("status", (" M tracked.py\n", "?? scratch.py\n"))
def test_source_identity_fails_closed_on_tracked_and_untracked_changes(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    commands: list[tuple[str, ...]] = []
    results = iter(
        (
            SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"),
            SimpleNamespace(returncode=0, stdout=status),
        )
    )

    def run(command: tuple[str, ...], **_kwargs: object) -> object:
        commands.append(command)
        return next(results)

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    with pytest.raises(benchmark.HarnessError, match="source is dirty"):
        benchmark._git_source_identity()
    assert commands[1] == (
        "git",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )


def test_source_identity_accepts_only_a_full_clean_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        (
            SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"),
            SimpleNamespace(returncode=0, stdout=""),
        )
    )
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: next(results),
    )
    identity = benchmark._git_source_identity()
    assert identity["revision"] == "a" * 40
    assert identity["dirty"] is False
    assert identity["untracked_files_checked"] is True


def test_source_identity_uses_the_selected_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "selected"
    checkout.mkdir()
    calls: list[tuple[tuple[str, ...], Path]] = []
    results = iter(
        (
            SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"),
            SimpleNamespace(returncode=0, stdout=""),
        )
    )

    def run(command: tuple[str, ...], **kwargs: object) -> object:
        cwd = kwargs.get("cwd")
        assert isinstance(cwd, Path)
        calls.append((command, cwd))
        return next(results)

    monkeypatch.setattr(benchmark, "ROOT", checkout)
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    identity = benchmark._git_source_identity()

    assert identity["checkout"] == str(checkout.resolve())
    assert [cwd for _command, cwd in calls] == [checkout, checkout]


def test_worker_keeps_frozen_driver_and_inherits_checkout_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "selected"
    checkout.mkdir()
    override = str(checkout.resolve())
    captured: dict[str, object] = {}

    def run(command: tuple[str, ...], **kwargs: object) -> object:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr=None)

    monkeypatch.setattr(benchmark, "ROOT", checkout)
    monkeypatch.setenv(benchmark.SOURCE_CHECKOUT_OVERRIDE_ENV, override)
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    with pytest.raises(benchmark.HarnessError, match="did not emit a JSON result"):
        benchmark._run_worker(
            ("generate",),
            mode="recurrence",
            phase="generation",
            timeout_seconds=1.0,
        )

    command = captured["command"]
    environment = captured["env"]
    assert isinstance(command, tuple)
    assert isinstance(environment, dict)
    assert command[:3] == (
        sys.executable,
        str(benchmark.DRIVER_PATH),
        "_worker",
    )
    assert command[1] != str(
        checkout / "tools" / "developer" / "recurrence_z6g_benchmark.py"
    )
    assert captured["cwd"] == checkout
    assert environment[benchmark.SOURCE_CHECKOUT_OVERRIDE_ENV] == override


def test_path_state_identity_records_optional_absence(tmp_path: Path) -> None:
    missing = tmp_path / "candidate-Cargo.lock"
    absent = benchmark._path_state_identity(missing)
    assert absent == {
        "present": False,
        "path": str(missing),
        "resolved_path": str(missing.resolve()),
    }

    missing.write_bytes(b"candidate lock")
    present = benchmark._path_state_identity(missing)
    assert present["present"] is True
    assert present["size_bytes"] == len(b"candidate lock")
    assert present["sha256"] == hashlib.sha256(b"candidate lock").hexdigest()


def _build_info(
    checkout: Path,
    *,
    revision: object = "a" * 40,
    digest: object = "b" * 64,
) -> dict[str, object]:
    return {
        "source_checkout": str(checkout),
        "source_revision": revision,
        "native_build_inputs_sha256": digest,
    }


def test_runtime_binding_requires_clean_matching_source_revision(
    tmp_path: Path,
) -> None:
    source = {
        "checkout": str(tmp_path),
        "revision": "a" * 40,
    }
    benchmark._validate_runtime_binding(
        source,
        _build_info(tmp_path),
        native_build_inputs_sha256="b" * 64,
    )

    with pytest.raises(benchmark.HarnessError, match="no clean source revision"):
        benchmark._validate_runtime_binding(
            source,
            _build_info(tmp_path, revision=None),
            native_build_inputs_sha256="b" * 64,
        )
    with pytest.raises(benchmark.HarnessError, match="does not match benchmark HEAD"):
        benchmark._validate_runtime_binding(
            source,
            _build_info(tmp_path, revision="c" * 40),
            native_build_inputs_sha256="b" * 64,
        )
    with pytest.raises(benchmark.HarnessError, match="build inputs"):
        benchmark._validate_runtime_binding(
            source,
            _build_info(tmp_path, digest="c" * 64),
            native_build_inputs_sha256="b" * 64,
        )


def test_runtime_binding_rejects_a_different_checkout(
    tmp_path: Path,
) -> None:
    source_checkout = tmp_path / "source"
    other_checkout = tmp_path / "other"
    source_checkout.mkdir()
    other_checkout.mkdir()
    source = {
        "checkout": str(source_checkout),
        "revision": "a" * 40,
    }
    with pytest.raises(benchmark.HarnessError, match="different source checkout"):
        benchmark._validate_runtime_binding(
            source,
            _build_info(other_checkout),
            native_build_inputs_sha256="b" * 64,
        )


def test_expected_runtime_identity_is_externally_pinned() -> None:
    source = {"revision": "a" * 40}
    runtime = {
        "native_extension": {
            "build_inputs_sha256": "b" * 64,
        }
    }
    benchmark._validate_expected_runtime_identity(
        source,
        runtime,
        expected_source_revision="a" * 40,
        expected_native_build_inputs_sha256="b" * 64,
    )

    with pytest.raises(benchmark.HarnessError, match="provided together"):
        benchmark._validate_expected_runtime_identity(
            source,
            runtime,
            expected_source_revision="a" * 40,
            expected_native_build_inputs_sha256=None,
        )
    with pytest.raises(benchmark.HarnessError, match="externally pinned revision"):
        benchmark._validate_expected_runtime_identity(
            source,
            runtime,
            expected_source_revision="c" * 40,
            expected_native_build_inputs_sha256="b" * 64,
        )
    with pytest.raises(benchmark.HarnessError, match="externally pinned build-input"):
        benchmark._validate_expected_runtime_identity(
            source,
            runtime,
            expected_source_revision="a" * 40,
            expected_native_build_inputs_sha256="d" * 64,
        )


@pytest.mark.parametrize("field", benchmark._PROFILE_EXPECTATION_FIELDS)
def test_profile_worker_rejects_each_identity_mismatch(field: str) -> None:
    expected = {name: "a" * 64 for name in benchmark._PROFILE_EXPECTATION_FIELDS}
    observed = dict(expected)
    observed[field] = "b" * 64
    with pytest.raises(benchmark.HarnessError, match="drifted before timing"):
        benchmark._validate_profile_worker_expectations(expected, observed)


def test_profile_worker_rejects_artifact_retarget_after_tree_rebind() -> None:
    pinned_artifact_id = "a" * 64
    replacement_runtime = SimpleNamespace(artifact_id="b" * 64)

    with pytest.raises(
        benchmark.HarnessError,
        match="loaded a different artifact",
    ):
        benchmark._loaded_runtime_artifact_verification(
            replacement_runtime,
            expected_artifact_id=pinned_artifact_id,
            phase="after-native-load-before-timing",
        )

    accepted = benchmark._loaded_runtime_artifact_verification(
        SimpleNamespace(artifact_id=pinned_artifact_id),
        expected_artifact_id=pinned_artifact_id,
        phase="after-native-load-before-timing",
    )
    benchmark._validate_loaded_runtime_artifact_verification(
        accepted,
        expected_artifact_id=pinned_artifact_id,
        phase="after-native-load-before-timing",
    )


@pytest.mark.parametrize(
    "drift",
    ("source", "runtime", "artifact", "artifact-metadata"),
)
def test_driver_post_worker_recheck_rejects_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    source = {"revision": "a" * 40, "checkout": str(tmp_path)}
    runtime = {"native_extension": {"sha256": "b" * 64}}
    semantic_identity = {"kind": "test-semantic-identity"}
    artifact_identity = {
        "path": str((tmp_path / "artifact").resolve()),
        "artifact_id": "1" * 64,
        "manifest": {
            "resolved_path": str((tmp_path / "artifact" / "artifact.json").resolve()),
            "size_bytes": 1,
            "sha256": "2" * 64,
        },
        "tree": {
            "algorithm": "sha256-relative-path-size-content-v1",
            "sha256": "c" * 64,
            "file_count": 1,
            "size_bytes": 1,
        },
        "payloads": [],
        "process_id": "process-1",
        "process_expression": "d d~ > z g",
        "color_accuracy": "lc",
        "producer": {"name": "test"},
        "model_identity": {"name": "test-model"},
        "semantic_identity": semantic_identity,
        "semantic_identity_sha256": benchmark._canonical_sha256(semantic_identity),
    }
    reuse_signature = {"semantic_signature_sha256": "d" * 64}
    observed_source = dict(source)
    observed_runtime = dict(runtime)
    observed_artifact = dict(artifact_identity)
    if drift == "source":
        observed_source["revision"] = "e" * 40
    elif drift == "runtime":
        observed_runtime = {"native_extension": {"sha256": "e" * 64}}
    else:
        observed_artifact = copy.deepcopy(artifact_identity)
        if drift == "artifact":
            observed_artifact["tree"]["sha256"] = "e" * 64
        else:
            observed_artifact["path"] = str(
                (tmp_path / "retargeted-artifact").resolve()
            )
    observed_artifact.pop("payloads", None)

    monkeypatch.setattr(
        benchmark,
        "_git_source_identity",
        lambda: observed_source,
    )
    monkeypatch.setattr(
        benchmark,
        "_runtime_provenance",
        lambda _source: observed_runtime,
    )
    monkeypatch.setattr(
        benchmark,
        "_require_bound_reusable_artifact",
        lambda *_args, **_kwargs: (
            observed_artifact,
            reuse_signature,
            {"sha256": "f" * 64},
        ),
    )
    baselines = {
        "compiled": {
            "artifact": str(tmp_path / "artifact"),
            "generation_signature": {"kind": "generation"},
            "artifact_identity": artifact_identity,
            "reuse_signature": reuse_signature,
            "reuse_sidecar_sha256": "f" * 64,
        }
    }
    with pytest.raises(benchmark.HarnessError, match="drifted"):
        benchmark._recheck_driver_state(
            source,
            runtime,
            baselines,
            phase="after-test-worker",
        )


def test_explicit_model_identity_is_accurate_and_not_builtin(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "ufo-sm.pyamplicol-model"
    prepared.write_bytes(b"explicit model")
    arguments = _arguments("--prepared-model", str(prepared))
    identity = benchmark._selected_model_identity(
        arguments,
        mode="compiled",
        source_identity={"revision": "a" * 40},
    )
    assert identity["kind"] == "explicit-prepared-model"
    assert identity["resource_id"] is None
    assert identity["file"]["resolved_path"] == str(prepared.resolve())
    assert identity["file"]["sha256"] == hashlib.sha256(b"explicit model").hexdigest()
    benchmark._validate_worker_model_identity(
        identity,
        {
            "kind": "explicit-prepared-model",
            "resource_id": None,
            "compile_excluded_from_generation": True,
            "file": dict(identity["file"]),
        },
    )
    changed_file = dict(identity["file"])
    changed_file["sha256"] = "0" * 64
    with pytest.raises(benchmark.HarnessError, match="disagrees on sha256"):
        benchmark._validate_worker_model_identity(
            identity,
            {
                "kind": "explicit-prepared-model",
                "resource_id": None,
                "compile_excluded_from_generation": True,
                "file": changed_file,
            },
        )


def _write_fake_artifact(path: Path) -> None:
    path.mkdir()
    payload = path / "payload.bin"
    payload.write_bytes(b"payload")
    payload_sha256 = hashlib.sha256(b"payload").hexdigest()
    physics_path = path / "processes" / "process-1" / "physics.json"
    physics_path.parent.mkdir(parents=True)
    physics = {
        "coverage": {
            "color": "complete",
            "helicities": "complete",
        },
        "color_components": [
            {
                "index": 0,
                "id": "flow:2,1",
                "kind": "lc-flow",
                "word": [2, 1],
                "representative_id": "flow:2,1",
                "computed": True,
                "coefficient": 1.0,
            }
        ],
        "helicities": [
            {
                "index": 0,
                "id": "h:-1,+1",
                "values": [-1, 1],
                "representative_id": "h:-1,+1",
                "computed": True,
                "coefficient": 1.0,
                "structural_zero": False,
            },
            {
                "index": 1,
                "id": "h:+1,-1",
                "values": [1, -1],
                "representative_id": "h:-1,+1",
                "computed": False,
                "coefficient": 1.0,
                "structural_zero": False,
            },
        ],
        "extensions": {
            "normalization": {
                "average_factor": 36,
                "color_factor": 3,
                "coupling_policy": "stage-local",
            },
            "runtime_selectors": {
                "kind": "pyamplicol-runtime-selectors",
                "contract_version": 1,
                "generation_specialized_axes": [],
                "axes": {
                    "color_flow": {
                        "generation_coverage": "complete",
                        "generation_selection": [],
                        "runtime_contract": "complete-reusable",
                    },
                    "helicity": {
                        "generation_coverage": "complete",
                        "generation_selection": {},
                        "runtime_contract": "complete-reusable",
                    },
                },
            },
        },
        "reduction": {
            "kind": "lc-diagonal",
            "groups": [
                {
                    "id": "reduction:0",
                    "physical_color_ids": ["flow:2,1"],
                    "physical_helicity_ids": ["h:-1,+1", "h:+1,-1"],
                    "representative_color_id": "flow:2,1",
                    "representative_helicity_id": "h:-1,+1",
                }
            ],
        },
    }
    physics_path.write_text(json.dumps(physics, sort_keys=True), encoding="utf-8")
    manifest = {
        "artifact_id": "artifact-1",
        "producer": {"name": "test"},
        "model": {
            "name": "built-in-sm",
            "content_sha256": "c" * 64,
            "compiled_schema_version": 9,
            "restriction": None,
            "source_kind": "compiled-model",
        },
        "extensions": {
            "generation": {
                "concrete_processes": [
                    {
                        "id": "process-1",
                        "runtime_schema_sha256": "a" * 64,
                        "reduction_schedule": "manifest-order",
                    }
                ]
            }
        },
        "processes": [
            {
                "id": "process-1",
                "expression": "d d~ > z g",
                "color_accuracy": "lc",
                "physics_path": "processes/process-1/physics.json",
            }
        ],
        "payloads": [
            {
                "path": "payload.bin",
                "size_bytes": len(b"payload"),
                "sha256": payload_sha256,
                "role": "runtime",
                "process_id": "process-1",
            },
            {
                "path": "processes/process-1/physics.json",
                "size_bytes": physics_path.stat().st_size,
                "sha256": hashlib.sha256(physics_path.read_bytes()).hexdigest(),
                "role": "physics",
                "process_id": "process-1",
            },
        ],
    }
    (path / "artifact.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )


def test_logical_axis_projection_ignores_execution_representatives() -> None:
    physical = [
        {
            "index": 0,
            "id": "flow:2,1",
            "kind": "lc-flow",
            "word": [2, 1],
            "coefficient": 1.0,
        },
        {
            "index": 1,
            "id": "flow:1,2",
            "kind": "lc-flow",
            "word": [1, 2],
            "coefficient": 1.0,
        },
    ]
    quotient = [
        {
            **entry,
            "computed": index == 0,
            "representative_id": physical[0]["id"],
        }
        for index, entry in enumerate(physical)
    ]
    materialized = [
        {
            **entry,
            "computed": True,
            "representative_id": entry["id"],
        }
        for entry in physical
    ]

    quotient_axis = benchmark._logical_physical_axis(
        benchmark._ordered_physical_axis(
            quotient,
            label="color-flow",
            require_structural_zero=False,
        ),
        require_structural_zero=False,
    )
    materialized_axis = benchmark._logical_physical_axis(
        benchmark._ordered_physical_axis(
            materialized,
            label="color-flow",
            require_structural_zero=False,
        ),
        require_structural_zero=False,
    )

    assert quotient_axis == materialized_axis
    assert all(
        "computed" not in entry and "representative_id" not in entry
        for entry in quotient_axis["ordered_entries"]
    )


@pytest.mark.parametrize(
    "descriptor_key",
    ("native_reduction_groups", "recurrence_runtime_reduction"),
)
def test_expanded_reduction_rejects_reserved_native_descriptor_presence(
    tmp_path: Path,
    descriptor_key: str,
) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process = manifest["processes"][0]
    physics_path = artifact / process["physics_path"]
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    physics["extensions"][descriptor_key] = None
    physics_path.write_text(json.dumps(physics, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        benchmark.HarnessError,
        match="expanded reduction duplicates a native descriptor",
    ):
        benchmark._artifact_semantic_identity(artifact, manifest, process)


@pytest.mark.parametrize(
    ("descriptor_key", "descriptor", "error"),
    [
        (
            "native_reduction_groups",
            {
                "kind": "pyamplicol-eager-plan-v3-reduction-groups",
                "schema_version": 1,
                "storage_abi": "pacbin-v1",
                "runtime_layout_abi": "pyamplicol-eager-runtime-layout-v1",
                "container_path": "eager-runtime.pacbin",
                "group_member": "reductions/groups.bin",
                "entry_member": "reductions/entries.bin",
                "group_count": 1,
            },
            "could not authenticate compact eager reduction",
        ),
        (
            "recurrence_runtime_reduction",
            {
                "kind": "pyamplicol-recurrence-native-reduction-v2",
                "runtime_layout_abi": "pyamplicol-recurrence-runtime-layout-v2",
                "container_path": "recurrence-runtime.pacbin",
                "plan_member_path": "schedule/recurrence-direct-schedule-v2.bin",
            },
            "could not authenticate compact recurrence reduction",
        ),
    ],
)
def test_compact_reduction_missing_native_payload_fails_closed(
    tmp_path: Path,
    descriptor_key: str,
    descriptor: dict[str, object],
    error: str,
) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process = manifest["processes"][0]
    physics_path = artifact / process["physics_path"]
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    physics["reduction"]["groups"] = []
    physics["extensions"][descriptor_key] = descriptor
    physics_path.write_text(json.dumps(physics, sort_keys=True), encoding="utf-8")

    with pytest.raises(benchmark.HarnessError, match=error):
        benchmark._artifact_semantic_identity(artifact, manifest, process)


def test_compact_eager_reduction_is_authenticated_and_logically_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process = manifest["processes"][0]
    physics_path = artifact / process["physics_path"]
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    groups = physics["reduction"]["groups"]
    physics["reduction"]["groups"] = []
    physics["extensions"]["native_reduction_groups"] = {
        "kind": "pyamplicol-eager-plan-v3-reduction-groups",
        "schema_version": 1,
        "storage_abi": "pacbin-v1",
        "runtime_layout_abi": "pyamplicol-eager-runtime-layout-v1",
        "container_path": "eager-runtime.pacbin",
        "group_member": "reductions/groups.bin",
        "entry_member": "reductions/entries.bin",
        "group_count": 1,
    }
    physics_path.write_text(json.dumps(physics, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        benchmark,
        "_load_compact_eager_reduction_groups",
        lambda _artifact, _process_id: groups,
    )

    identity = benchmark._artifact_semantic_identity(artifact, manifest, process)

    assert identity["reduction_coverage"]["complete"] is True
    assert identity["execution_reduction_coverage"]["complete"] is True
    assert identity["execution_reduction_identity"]["kind"] == (
        "eager-plan-v3-pacbin-reduction-v1"
    )

    physics["extensions"]["native_reduction_groups"]["group_count"] = 0
    physics_path.write_text(json.dumps(physics, sort_keys=True), encoding="utf-8")
    with pytest.raises(
        benchmark.HarnessError,
        match="compact eager reduction descriptor is invalid",
    ):
        benchmark._artifact_semantic_identity(artifact, manifest, process)


def test_compact_recurrence_reduction_binds_complete_public_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process = manifest["processes"][0]
    physics_path = artifact / process["physics_path"]
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    physics["reduction"]["groups"] = []
    physics["extensions"]["recurrence_runtime_reduction"] = {
        "kind": "pyamplicol-recurrence-native-reduction-v2",
        "runtime_layout_abi": "pyamplicol-recurrence-runtime-layout-v2",
        "container_path": "recurrence-runtime.pacbin",
        "plan_member_path": "schedule/recurrence-direct-schedule-v2.bin",
    }
    physics_path.write_text(json.dumps(physics, sort_keys=True), encoding="utf-8")
    sections = SimpleNamespace(
        strategy="topology-replay",
        semantic_digest="a" * 64,
        runtime_layout_digest="b" * 64,
        public_flow_ids=(0,),
        resolved_helicities=(
            SimpleNamespace(public_helicity_start=0, public_helicity_count=2),
            SimpleNamespace(public_helicity_start=2, public_helicity_count=2),
        ),
        public_helicities=(-1, 1, 1, -1),
        replay_targets=(SimpleNamespace(public_flow_id=0),),
        amplitude_destination_count=2,
        replay_helicity_map=(0, 1),
        amplitude_destinations=(
            SimpleNamespace(target_sector_id=0, target_helicity_id=0),
            SimpleNamespace(target_sector_id=0, target_helicity_id=1),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_load_compact_recurrence_reduction",
        lambda _artifact, _process_id: sections,
    )

    identity = benchmark._artifact_semantic_identity(artifact, manifest, process)

    assert identity["reduction_coverage"] == {
        "complete": True,
        "expected_physical_pair_count": 2,
        "observed_physical_pair_count": 2,
        "errors": [],
    }
    assert identity["execution_reduction_identity"]["kind"] == (
        "recurrence-plan-v2-pacbin-reduction-v1"
    )

    sections.replay_helicity_map = (0,)
    with pytest.raises(
        benchmark.HarnessError,
        match="compact recurrence reduction coverage is incomplete",
    ):
        benchmark._artifact_semantic_identity(artifact, manifest, process)


def test_compact_recurrence_union_accepts_folded_public_flow_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process = manifest["processes"][0]
    physics_path = artifact / process["physics_path"]
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    physics["color_components"].append(
        {
            "index": 1,
            "id": "flow:1,2",
            "kind": "lc-flow",
            "word": [1, 2],
            "representative_id": "flow:1,2",
            "computed": True,
            "coefficient": 1.0,
        }
    )
    physics["reduction"]["groups"] = []
    physics["extensions"]["recurrence_runtime_reduction"] = {
        "kind": "pyamplicol-recurrence-native-reduction-v2",
        "runtime_layout_abi": "pyamplicol-recurrence-runtime-layout-v2",
        "container_path": "recurrence-runtime.pacbin",
        "plan_member_path": "schedule/recurrence-direct-schedule-v2.bin",
    }
    physics_path.write_text(json.dumps(physics, sort_keys=True), encoding="utf-8")
    sections = SimpleNamespace(
        strategy="all-flow-union",
        semantic_digest="a" * 64,
        runtime_layout_digest="b" * 64,
        public_flow_ids=(0, 0),
        resolved_helicities=(
            SimpleNamespace(public_helicity_start=0, public_helicity_count=2),
            SimpleNamespace(public_helicity_start=2, public_helicity_count=2),
        ),
        public_helicities=(-1, 1, 1, -1),
        replay_targets=(),
        amplitude_destination_count=1,
        replay_helicity_map=(),
        amplitude_destinations=(
            SimpleNamespace(target_sector_id=0, target_helicity_id=(2**32 - 1)),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_load_compact_recurrence_reduction",
        lambda _artifact, _process_id: sections,
    )

    identity = benchmark._artifact_semantic_identity(artifact, manifest, process)
    execution = identity["execution_reduction_identity"]

    assert identity["execution_reduction_coverage"] == {
        "complete": True,
        "expected_physical_pair_count": 4,
        "observed_physical_pair_count": 4,
        "errors": [],
    }
    assert execution["public_flow_binding_count"] == 2
    assert execution["construction_sector_count"] == 1
    assert execution["amplitude_destination_count"] == 1
    benchmark._validate_execution_reduction_summary(
        execution,
        identity["execution_reduction_coverage"],
        logical_reduction=identity["reduction_ordering"],
        artifact=artifact,
    )

    forged_execution = copy.deepcopy(execution)
    forged_execution["replay_helicity_map_sha256"] = "f" * 64
    with pytest.raises(
        benchmark.HarnessError,
        match="compact recurrence execution reduction summary is invalid",
    ):
        benchmark._validate_execution_reduction_summary(
            forged_execution,
            identity["execution_reduction_coverage"],
            logical_reduction=identity["reduction_ordering"],
            artifact=artifact,
        )

    sections.public_flow_ids = (0,)
    with pytest.raises(
        benchmark.HarnessError,
        match="compact recurrence reduction coverage is incomplete",
    ):
        benchmark._artifact_semantic_identity(artifact, manifest, process)


def test_artifact_reuse_requires_exact_signature_and_tree(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    signature = {"kind": "test-signature", "process": "d d~ > z g"}
    identity = benchmark._artifact_identity(artifact)
    benchmark._write_reuse_signature(
        artifact,
        signature=signature,
        artifact_identity=identity,
        generation_command=benchmark._command_identity(("python", "generate-artifact")),
    )
    reused, provenance = benchmark._require_reusable_artifact(
        artifact,
        expected_signature=signature,
    )
    assert reused["tree"] == identity["tree"]
    assert provenance["generation_command"]["argv"] == [
        "python",
        "generate-artifact",
    ]
    assert (
        provenance["semantic_signature"]["artifact_semantic_identity"]
        == identity["semantic_identity"]
    )

    with pytest.raises(benchmark.HarnessError, match="generation request changed"):
        benchmark._require_reusable_artifact(
            artifact,
            expected_signature={**signature, "process": "g g > g g"},
        )

    (artifact / "extra.bin").write_bytes(b"ambient mutation")
    with pytest.raises(benchmark.HarnessError, match="identity or tree changed"):
        benchmark._require_reusable_artifact(
            artifact,
            expected_signature=signature,
        )


def _bound_artifact_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    signature = {"kind": "test-signature", "process": "d d~ > z g"}
    identity = benchmark._artifact_identity(artifact)
    benchmark._write_reuse_signature(
        artifact,
        signature=signature,
        artifact_identity=identity,
        generation_command=benchmark._command_identity(("python", "generate-artifact")),
    )
    sidecar_path = benchmark._reuse_signature_path(artifact)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar_identity = benchmark._path_identity(sidecar_path)
    return (
        artifact,
        signature,
        identity,
        {
            "payload": sidecar,
            "identity": sidecar_identity,
        },
    )


def test_bound_artifact_recheck_skips_semantic_hydration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, signature, identity, sidecar = _bound_artifact_fixture(tmp_path)

    def reject_hydration(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bound recheck must not hydrate artifact semantics")

    monkeypatch.setattr(
        benchmark,
        "_artifact_semantic_identity",
        reject_hydration,
    )
    monkeypatch.setattr(
        benchmark,
        "_artifact_identity",
        reject_hydration,
    )
    for _ in range(3):
        rebound, provenance, observed_sidecar = (
            benchmark._require_bound_reusable_artifact(
                artifact,
                expected_tree_sha256=identity["tree"]["sha256"],
                expected_semantic_identity_sha256=identity["semantic_identity_sha256"],
                expected_reuse_semantic_signature_sha256=sidecar["payload"][
                    "semantic_signature_sha256"
                ],
                expected_sidecar_sha256=sidecar["identity"]["sha256"],
                expected_signature=signature,
            )
        )
        assert rebound["tree"] == identity["tree"]
        assert rebound["semantic_identity"] == identity["semantic_identity"]
        assert provenance == sidecar["payload"]
        assert observed_sidecar == sidecar["identity"]


def test_bound_artifact_recheck_rejects_same_size_tree_mutation(
    tmp_path: Path,
) -> None:
    artifact, signature, identity, sidecar = _bound_artifact_fixture(tmp_path)
    payload = artifact / "payload.bin"
    payload.write_bytes(b"payloae")
    assert payload.stat().st_size == len(b"payload")

    with pytest.raises(benchmark.HarnessError, match="tree changed"):
        benchmark._require_bound_reusable_artifact(
            artifact,
            expected_tree_sha256=identity["tree"]["sha256"],
            expected_semantic_identity_sha256=identity["semantic_identity_sha256"],
            expected_reuse_semantic_signature_sha256=sidecar["payload"][
                "semantic_signature_sha256"
            ],
            expected_sidecar_sha256=sidecar["identity"]["sha256"],
            expected_signature=signature,
        )


def test_bound_artifact_recheck_rejects_readdressed_sidecar_mutation(
    tmp_path: Path,
) -> None:
    artifact, signature, identity, sidecar = _bound_artifact_fixture(tmp_path)
    sidecar_path = benchmark._reuse_signature_path(artifact)
    forged = copy.deepcopy(sidecar["payload"])
    semantic_signature = forged["semantic_signature"]
    semantic_identity = semantic_signature["artifact_semantic_identity"]
    semantic_identity["normalization"]["average_factor"] = 999
    semantic_identity["normalization_sha256"] = benchmark._canonical_sha256(
        semantic_identity["normalization"]
    )
    forged["artifact_semantic_identity_sha256"] = benchmark._canonical_sha256(
        semantic_identity
    )
    forged["semantic_signature_sha256"] = benchmark._canonical_sha256(
        semantic_signature
    )
    sidecar_path.write_text(json.dumps(forged, sort_keys=True), encoding="utf-8")

    with pytest.raises(benchmark.HarnessError, match="sidecar changed"):
        benchmark._require_bound_reusable_artifact(
            artifact,
            expected_tree_sha256=identity["tree"]["sha256"],
            expected_semantic_identity_sha256=identity["semantic_identity_sha256"],
            expected_reuse_semantic_signature_sha256=sidecar["payload"][
                "semantic_signature_sha256"
            ],
            expected_sidecar_sha256=sidecar["identity"]["sha256"],
            expected_signature=signature,
        )


def test_bound_artifact_recheck_rejects_generation_request_drift(
    tmp_path: Path,
) -> None:
    artifact, _signature, identity, sidecar = _bound_artifact_fixture(tmp_path)

    with pytest.raises(benchmark.HarnessError, match="generation request changed"):
        benchmark._require_bound_reusable_artifact(
            artifact,
            expected_tree_sha256=identity["tree"]["sha256"],
            expected_semantic_identity_sha256=identity["semantic_identity_sha256"],
            expected_reuse_semantic_signature_sha256=sidecar["payload"][
                "semantic_signature_sha256"
            ],
            expected_sidecar_sha256=sidecar["identity"]["sha256"],
            expected_signature={"kind": "different-generation-request"},
        )


def test_artifact_reuse_rejects_missing_sidecar(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    with pytest.raises(benchmark.HarnessError, match="reuse signature is missing"):
        benchmark._require_reusable_artifact(
            artifact,
            expected_signature={"kind": "test-signature"},
        )


def test_artifact_tree_identity_rejects_symlinks(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload = artifact / "payload.bin"
    payload.write_bytes(b"payload")
    (artifact / "payload-link.bin").symlink_to(payload)
    with pytest.raises(benchmark.HarnessError, match="unsupported symlink"):
        benchmark._tree_identity(artifact)


def test_artifact_semantics_bind_axes_normalization_and_reduction_order(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process = manifest["processes"][0]
    original = benchmark._artifact_semantic_identity(artifact, manifest, process)
    assert original["physical_color_flows"]["ordered_ids"] == ["flow:2,1"]
    assert original["physical_helicities"]["ordered_ids"] == [
        "h:-1,+1",
        "h:+1,-1",
    ]

    physics_path = artifact / "processes" / "process-1" / "physics.json"
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    physics["extensions"]["normalization"]["average_factor"] = 72
    physics["reduction"]["groups"][0]["physical_helicity_ids"].reverse()
    physics_path.write_text(json.dumps(physics, sort_keys=True), encoding="utf-8")
    changed = benchmark._artifact_semantic_identity(artifact, manifest, process)
    assert changed["normalization_sha256"] != original["normalization_sha256"]
    assert (
        changed["reduction_ordering_sha256"] == (original["reduction_ordering_sha256"])
    )
    assert (
        changed["execution_reduction_identity_sha256"]
        != original["execution_reduction_identity_sha256"]
    )

    physics["helicities"].reverse()
    physics_path.write_text(json.dumps(physics, sort_keys=True), encoding="utf-8")
    with pytest.raises(benchmark.HarnessError, match="complete and ordered"):
        benchmark._artifact_semantic_identity(artifact, manifest, process)


def test_artifact_semantics_reject_model_free_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_fake_artifact(artifact)
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process = manifest["processes"][0]
    del manifest["model"]
    with pytest.raises(benchmark.HarnessError, match="no model identity"):
        benchmark._artifact_semantic_identity(artifact, manifest, process)


def test_semantic_signature_covers_generation_configuration() -> None:
    source = {"revision": "a" * 40}
    runtime = {"native": {"sha256": "b" * 64}}
    model = {"kind": "built-in-sm-source", "source_revision": "a" * 40}
    default = _arguments()
    changed = _arguments("--point-tile-size", "2048")
    default_signature = benchmark._semantic_generation_signature(
        default,
        mode="recurrence",
        source_identity=source,
        runtime_provenance=runtime,
        model_identity=model,
    )
    changed_signature = benchmark._semantic_generation_signature(
        changed,
        mode="recurrence",
        source_identity=source,
        runtime_provenance=runtime,
        model_identity=model,
    )
    assert default_signature["point_tile_size"] == 1024
    assert changed_signature["point_tile_size"] == 2048
    assert default_signature["validation"]["relative_tolerance"] == benchmark.RTOL
    assert default_signature["validation"]["absolute_tolerance"] == benchmark.ATOL
    assert benchmark.RTOL == 1.0e-12
    assert benchmark.ATOL == 1.0e-15
    assert benchmark._canonical_sha256(default_signature) != (
        benchmark._canonical_sha256(changed_signature)
    )


def test_generation_config_uses_additive_validation_tolerances() -> None:
    config = benchmark._generation_config(
        "recurrence",
        validation_samples=3,
        lc_flow_layout="topology-replay",
        point_tile_size=1024,
        jit_optimization_level=2,
    )

    validation = config.generation.validation
    assert validation.relative_tolerance == benchmark.RTOL
    assert validation.absolute_tolerance == benchmark.ATOL
    assert validation.absolute_tolerance == 1.0e-15


def test_validation_fixture_records_raw_and_semantic_digests(
    tmp_path: Path,
) -> None:
    process_root = tmp_path / "processes" / "process-1"
    process_root.mkdir(parents=True)
    path = process_root / "validation-momenta.json"
    payload = {
        "points": [
            [
                {"momentum": [1.0, 0.0, 0.0, 1.0]},
                {"momentum": [1.0, 0.0, 0.0, -1.0]},
            ],
            [
                {"momentum": [2.0, 0.0, 0.0, 2.0]},
                {"momentum": [2.0, 0.0, 0.0, -2.0]},
            ],
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    points, identity = benchmark._validation_fixture(tmp_path, "process-1")
    assert len(points) == 2
    assert identity["point_count"] == 2
    assert identity["file"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["points_sha256"] == benchmark._canonical_sha256(points)


def test_profile_worker_evaluates_every_validation_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    process_root = artifact / "processes" / "process-1"
    process_root.mkdir(parents=True)
    points = [
        [
            {"momentum": [float(index + 1), 0.0, 0.0, 1.0]},
            {"momentum": [float(index + 1), 0.0, 0.0, -1.0]},
        ]
        for index in range(3)
    ]
    (process_root / "validation-momenta.json").write_text(
        json.dumps({"points": points}),
        encoding="utf-8",
    )
    calls: list[tuple[str, int]] = []
    benchmark_point_sets: list[object] = []
    events: list[str] = []

    def verify(_arguments: object) -> dict[str, object]:
        events.append("verify")
        return {
            "kind": benchmark.WORKER_VERIFICATION_KIND,
            "schema_version": benchmark.WORKER_VERIFICATION_SCHEMA,
            "artifact_semantic_identity": _artifact_semantic_identity(),
            "artifact_semantic_identity_sha256": "a" * 64,
            "expected": {"artifact_id": "a" * 64},
        }

    monkeypatch.setattr(
        benchmark,
        "_verify_profile_worker_environment",
        verify,
    )

    class FakeResolved:
        def __init__(self, values: tuple[complex, ...]) -> None:
            self._values = values
            self.values = tuple(((value,),) for value in values)
            self.helicity_ids = ("h:-1,+1,-1,+1",)
            self.color_ids = ("flow:2,1",)

        def total(self) -> tuple[complex, ...]:
            return self._values

    class FakeRuntime:
        artifact_id = "a" * 64
        physics = SimpleNamespace(
            process="d d~ > z g",
            process_id="process-1",
            color_accuracy="lc",
            color_flows=(SimpleNamespace(id="flow:2,1"),),
            helicities=(SimpleNamespace(id="h:-1,+1,-1,+1"),),
            structural_zero_helicity_count=0,
        )

        def __init__(self) -> None:
            self._backend = SimpleNamespace(
                _benchmark_f64_wall_time=lambda batch, repetitions, **_selectors: (
                    1.0e-6 * len(batch) * repetitions
                )
            )

        def evaluate(
            self,
            batch: tuple[object, ...],
            **_selectors: object,
        ) -> tuple[complex, ...]:
            calls.append(("evaluate", len(batch)))
            return tuple(complex(index + 1) for index in range(len(batch)))

        def evaluate_resolved(
            self,
            batch: tuple[object, ...],
            **_selectors: object,
        ) -> FakeResolved:
            calls.append(("evaluate_resolved", len(batch)))
            return FakeResolved(
                tuple(complex(index + 1) for index in range(len(batch)))
            )

    class FakeRuntimeFactory:
        @staticmethod
        def load(_artifact: Path, *, process: str) -> FakeRuntime:
            events.append("load")
            assert process == "d d~ > z g"
            return FakeRuntime()

    class FakeBenchmarkConfig:
        batch_size: int

        def __init__(self, **values: object) -> None:
            self.__dict__.update(values)
            batch_size = values["batch_size"]
            assert isinstance(batch_size, int)
            self.batch_size = batch_size

    class FakeBenchmarkRunner:
        def __init__(self, config: FakeBenchmarkConfig) -> None:
            self.config = config

        def run(self, _runtime: FakeRuntime, *, points: object) -> object:
            benchmark_point_sets.append(points)
            uncertainty = SimpleNamespace(
                standard_deviation=0.0,
                standard_error=0.0,
                relative_standard_error=0.0,
            )
            return SimpleNamespace(
                effective_config=SimpleNamespace(batch_size=self.config.batch_size),
                sample_count=7,
                repetitions_per_sample=1,
                evaluation_count=7,
                evaluated_point_count=7 * self.config.batch_size,
                wall_time_per_point=1.0e-6,
                evaluator_time_per_point=1.0e-6,
                uncertainty=uncertainty,
                evaluator_uncertainty=uncertainty,
                environment={
                    "wall_time_source": "runtime_core_repeated_wall_time",
                    "wall_time_sample_pass": "runtime._benchmark_f64_wall_time",
                },
                interrupted=False,
            )

    pyamplicol = ModuleType("pyamplicol")
    pyamplicol.BenchmarkRunner = FakeBenchmarkRunner  # type: ignore[attr-defined]
    pyamplicol.Runtime = FakeRuntimeFactory  # type: ignore[attr-defined]
    config = ModuleType("pyamplicol.config")
    config.BenchmarkConfig = FakeBenchmarkConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyamplicol", pyamplicol)
    monkeypatch.setitem(sys.modules, "pyamplicol.config", config)

    arguments = SimpleNamespace(
        artifact=artifact,
        process_expression="d d~ > z g",
        gluon_count=1,
        mode="compiled",
        color_flow="1",
        helicity="1",
        lc_flow_layout="topology-replay",
        validation_point_artifact=artifact,
        batch_size=[1],
        schedule_index=0,
        schedule_round=0,
        target_runtime=0.1,
        minimum_samples=7,
        warmup_runs=2,
        validation_samples=3,
    )
    result = benchmark._profile_worker(arguments)
    assert events[:2] == ["verify", "load"]
    assert calls[:2] == [("evaluate", 3), ("evaluate_resolved", 3)]
    assert len(benchmark_point_sets) == 1
    assert (
        benchmark._canonical_sha256(benchmark_point_sets[0])
        == (result["validation"]["fixture"]["points_sha256"])
    )
    assert result["validation"]["fixture"]["point_count"] == 3
    assert len(result["validation"]["selected_totals"]) == 3
    assert len(result["validation"]["resolved_components"]) == 3
    assert len(result["validation"]["point_comparisons"]) == 3
    raw_blocks = result["profiles"][0]["inner_native_wall_blocks"]
    assert (
        raw_blocks["fixture_points_sha256"]
        == (result["validation"]["fixture"]["points_sha256"])
    )
    assert len(raw_blocks["blocks"]) == benchmark.MIN_AUTHORITATIVE_SAMPLES
    assert raw_blocks["observed_native_wall_seconds"] >= arguments.target_runtime
    assert raw_blocks["minimum_native_wall_seconds"] == arguments.target_runtime
    assert raw_blocks["minimum_duration_satisfied"] is True
    assert raw_blocks["repetitions_per_block"] > 1
    assert (
        result["profiles"][0]["repetitions_per_sample"]
        == raw_blocks["repetitions_per_block"]
    )
