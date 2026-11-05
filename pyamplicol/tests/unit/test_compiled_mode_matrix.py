# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.developer import (
    compiled_mode_matrix as matrix,
)
from tools.developer import (
    compiled_mode_regression as regression,
)
from tools.developer import compiled_mode_sample

BASELINE = Path("/matrix/baseline/bin/python")
CURRENT = Path("/matrix/current/bin/python")
UFO_SM = Path("/matrix/models/sm")
OUTPUT_ROOT = Path("/matrix/output")
BASELINE_SOURCE_REVISION = "1" * 40
CURRENT_SOURCE_REVISION = "2" * 40
BASELINE_NATIVE_INPUTS = "c" * 64
CURRENT_NATIVE_INPUTS = "d" * 64
BASELINE_DISTRIBUTION = "e" * 64
CURRENT_DISTRIBUTION = "f" * 64
BASELINE_NATIVE_MODULE = "a" * 64
CURRENT_NATIVE_MODULE = "b" * 64


def _synthetic_installation(lane: str) -> dict[str, object]:
    baseline = lane == "baseline"
    return {
        "kind": regression.INSTALLATION_IDENTITY_KIND,
        "schema_version": regression.INSTALLATION_IDENTITY_SCHEMA_VERSION,
        "distribution_content": {
            "sha256": BASELINE_DISTRIBUTION if baseline else CURRENT_DISTRIBUTION,
        },
        "native_modules": [
            {"sha256": (BASELINE_NATIVE_MODULE if baseline else CURRENT_NATIVE_MODULE)}
        ],
        "build_info_files": [
            {
                "payload": {
                    "source_revision": (
                        BASELINE_SOURCE_REVISION
                        if baseline
                        else CURRENT_SOURCE_REVISION
                    ),
                    "native_build_inputs_sha256": (
                        BASELINE_NATIVE_INPUTS if baseline else CURRENT_NATIVE_INPUTS
                    ),
                    "publishable": False,
                    "selftest_fixture_bootstrap": False,
                }
            }
        ],
    }


def _synthetic_result(
    cell: matrix.MatrixCell,
    *,
    generation_ratio: float = 1.04,
    gain: bool = False,
) -> dict[str, object]:
    model = "built-in-sm" if cell.model_kind == "built-in" else str(UFO_SM)
    model_label = "built-in" if cell.model_kind == "built-in" else "ufo-sm"
    native_hashes = {
        "baseline": BASELINE_NATIVE_MODULE,
        "current": CURRENT_NATIVE_MODULE,
    }
    measurements = [
        {
            "lane": lane,
            "runtime_identity": {
                "lane": lane,
                "native_module": {"sha256": native_hashes[lane]},
                "native_build_inputs_sha256": (
                    BASELINE_NATIVE_INPUTS
                    if lane == "baseline"
                    else CURRENT_NATIVE_INPUTS
                ),
                "build_info": {
                    "payload": {
                        "source_revision": (
                            BASELINE_SOURCE_REVISION
                            if lane == "baseline"
                            else CURRENT_SOURCE_REVISION
                        ),
                        "native_build_inputs_sha256": (
                            BASELINE_NATIVE_INPUTS
                            if lane == "baseline"
                            else CURRENT_NATIVE_INPUTS
                        ),
                        "publishable": False,
                        "selftest_fixture_bootstrap": False,
                    }
                },
            },
            "warmed_numerical_result": {
                "values_f64": [1.0] * regression.VALIDATION_SAMPLE_COUNT,
            },
            **(
                {
                    "profile_attribution_sample_pass": (
                        regression.PROFILE_ATTRIBUTION_SAMPLE_PASS
                    ),
                    "profile_attribution_boundary": (regression.ARENA_PROFILE_BOUNDARY),
                    "profile_attribution_borrowed_flat_input": True,
                    "profile_attribution_preallocated_output": True,
                    "profile_attribution_phase_timing_scope": (
                        regression.ARENA_PHASE_TIMING_SCOPE
                    ),
                    "profile_attribution_evaluator_timing_available": False,
                    "profile_protocol": regression.ARENA_PROFILE_PROTOCOL,
                    "profile_timed_block_count": 7,
                    "paired_profile_evaluator_seconds_per_point": None,
                    "paired_profile_evaluator_uncertainty": None,
                    "paired_profile_timing_breakdown": {
                        "sample_count": 7,
                        "evaluator_call_time": None,
                        "raw_profile_samples": [
                            {
                                "execution_mode": cell.execution_mode,
                                "wall_time_s": 1.0,
                                "orchestration_time_s": 1.0,
                                "profile_boundary": (regression.ARENA_PROFILE_BOUNDARY),
                                "borrowed_flat_input": True,
                                "preallocated_output": True,
                                "phase_timing_scope": (
                                    regression.ARENA_PHASE_TIMING_SCOPE
                                ),
                                "evaluator_timing_available": False,
                                **{
                                    key: 0
                                    for key in (
                                        *regression._ZERO_ARENA_PROFILE_COUNTERS,
                                        *regression._ZERO_COMPILED_BOUNDARY_COUNTERS,
                                    )
                                },
                                **{
                                    key: 0.0
                                    for key in regression._ZERO_ARENA_PROFILE_TIMES
                                },
                                **{
                                    key: []
                                    for key in (
                                        regression._EMPTY_ARENA_PROFILE_PHASE_VECTORS
                                    )
                                },
                                "compiled_direct_arena_engine_count": (
                                    1 if cell.execution_mode == "compiled" else 0
                                ),
                                "compiled_direct_arena_call_count": (
                                    1 if cell.execution_mode == "compiled" else 0
                                ),
                                "evaluator_backend_call_count": (
                                    1 if cell.execution_mode == "compiled" else 0
                                ),
                            }
                            for _ in range(7)
                        ],
                    },
                }
                if lane == "current"
                else {}
            ),
        }
        for _pair in range(7)
        for lane in ("baseline", "current")
    ]
    return {
        "kind": regression.RESULT_KIND,
        "schema_version": regression.SCHEMA_VERSION,
        "complete": True,
        "performance_result_authoritative": True,
        "passes": True,
        "platform": platform.platform(),
        "configuration": {
            "baseline_python": str(BASELINE),
            "current_python": str(CURRENT),
            "output_root": str(
                OUTPUT_ROOT / "artifact-groups" / cell.artifact_group_id
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
            "independent_samples_per_lane": 7,
            "target_runtime_per_native_sample_seconds": 5.0,
            "minimum_native_timed_blocks_per_profile": 7,
            "warmup_runs_per_profile": 2,
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
            "helicities": list(cell.helicities),
            "color_flows": list(cell.color_flows),
            "dependency_sites": {"baseline": None, "current": None},
        },
        "gate": {"passes": True},
        "correctness_gate": {"passes": True},
        "arena_profile_gate": {"passes": True},
        "resource_gate": {
            "passes": True,
            "generation": {
                "current_over_baseline": generation_ratio,
                "passes": True,
            },
        },
        "gain_gate": {
            "relative_gain": 0.12 if gain else 0.0,
            "at_least_ten_percent": gain,
            "beyond_measurement_noise": gain,
            "passes": gain,
        },
        "native_module_sha256_by_lane": native_hashes,
        "artifacts": {
            lane: {
                "path": str(
                    OUTPUT_ROOT
                    / "artifact-groups"
                    / cell.artifact_group_id
                    / lane
                    / "artifact"
                ),
                "artifact_id": f"{cell.artifact_group_id}-{lane}",
                "manifest_sha256": ("7" if lane == "baseline" else "8") * 64,
                "tree_identity": {"sha256": ("9" if lane == "baseline" else "0") * 64},
                "payload_digests": {
                    "evaluator-state": ("3" if lane == "baseline" else "4") * 64
                },
                "installation_identity": _synthetic_installation(lane),
                "required_runtime_capabilities": [
                    (
                        regression.EAGER_DIRECT_ARENA_CAPABILITY
                        if cell.execution_mode == "eager"
                        else regression.COMPILED_DIRECT_ARENA_CAPABILITY
                    )
                ],
            }
            for lane in ("baseline", "current")
        },
        "measurements": measurements,
        "pair_orders": [
            ["baseline", "current"] if pair_index % 2 == 0 else ["current", "baseline"]
            for pair_index in range(7)
        ],
        "provenance": {
            "driver": {"sha256": matrix._sha256_file(matrix.CELL_DRIVER)},
            "watchdog": {"sha256": matrix._sha256_file(regression.WATCHDOG)},
            "native_sample_helper": {
                "sha256": matrix._sha256_file(regression.NATIVE_SAMPLE_HELPER)
            },
            "dependency_entry": {
                "sha256": matrix._sha256_file(regression.DEPENDENCY_ENTRY)
            },
            "interpreters": {"baseline": "baseline", "current": "current"},
            "dependency_sites": {"baseline": None, "current": None},
            "model": {"kind": cell.model_kind},
        },
    }


def _passing_results() -> dict[str, dict[str, object]]:
    results = {cell.cell_id: _synthetic_result(cell) for cell in matrix.CANONICAL_CELLS}
    for mode in matrix.EXECUTION_MODES:
        cell = next(
            candidate
            for candidate in matrix.CANONICAL_CELLS
            if candidate.execution_mode == mode
        )
        results[cell.cell_id] = _synthetic_result(cell, gain=True)
    return results


def _audit(results: dict[str, dict[str, object]]) -> dict[str, object]:
    return matrix.audit_results(
        results,
        baseline_python=BASELINE,
        current_python=CURRENT,
        ufo_sm_model=UFO_SM,
        output_root=OUTPUT_ROOT,
        expected_platform=platform.platform(),
        expected_builds={
            "baseline": {
                "source_revision": BASELINE_SOURCE_REVISION,
                "native_build_inputs_sha256": BASELINE_NATIVE_INPUTS,
                "distribution_content_sha256": BASELINE_DISTRIBUTION,
                "native_module_sha256": BASELINE_NATIVE_MODULE,
            },
            "current": {
                "source_revision": CURRENT_SOURCE_REVISION,
                "native_build_inputs_sha256": CURRENT_NATIVE_INPUTS,
                "distribution_content_sha256": CURRENT_DISTRIBUTION,
                "native_module_sha256": CURRENT_NATIVE_MODULE,
            },
        },
    )


def test_canonical_matrix_is_the_documented_168_cells() -> None:
    cells = matrix.CANONICAL_CELLS

    assert len(cells) == len({cell.cell_id for cell in cells}) == 168
    assert Counter(cell.category for cell in cells) == {
        "primary": 24,
        "medium": 120,
        "color-heavy": 24,
    }
    assert Counter(cell.execution_mode for cell in cells) == {
        "eager": 84,
        "compiled": 84,
    }
    assert Counter(cell.jit_optimization_level for cell in cells) == {2: 84, 3: 84}
    assert {cell.batch_size for cell in cells} == {1, 128, 1024}
    assert len({cell.artifact_group_id for cell in cells}) == 56


def test_stable_file_identity_projection_only_removes_mtimes() -> None:
    value = {
        "path": "/fixed/tool.py",
        "resolved_path": "/fixed/tool.py",
        "size_bytes": 123,
        "sha256": "a" * 64,
        "mtime_ns": 100,
        "nested": [
            {
                "path": "/fixed/model",
                "mtime_ns": 200,
                "digest": {"sha256": "b" * 64},
            }
        ],
    }
    projected = matrix._stable_file_identity_value(value)
    assert projected == {
        "path": "/fixed/tool.py",
        "resolved_path": "/fixed/tool.py",
        "size_bytes": 123,
        "sha256": "a" * 64,
        "nested": [
            {
                "path": "/fixed/model",
                "digest": {"sha256": "b" * 64},
            }
        ],
    }
    assert matrix._stable_file_identity_value(projected) == projected


def test_stable_runtime_projection_drops_only_host_observations() -> None:
    runtime = {
        "platform": "Linux-6.11.0-runner-a-x86_64",
        "python": "3.11.9",
        "executable": {
            "path": "/tmp/runtime/current/bin/python",
            "size_bytes": 123,
            "sha256": "a" * 64,
            "mtime_ns": 100,
        },
        "native_module": {
            "path": "/tmp/runtime/current/pyamplicol/_rusticol.abi3.so",
            "sha256": "b" * 64,
        },
        "build_info": {
            "payload": {
                "source_revision": CURRENT_SOURCE_REVISION,
                "native_build_inputs_sha256": CURRENT_NATIVE_INPUTS,
            }
        },
    }
    equivalent = copy.deepcopy(runtime)
    equivalent["platform"] = "Linux-6.12.0-runner-b-x86_64"
    equivalent["executable"]["mtime_ns"] = 200
    assert matrix._stable_runtime_identity_value(runtime) == (
        matrix._stable_runtime_identity_value(equivalent)
    )

    changed_path = copy.deepcopy(equivalent)
    changed_path["executable"]["path"] = "/tmp/another/bin/python"
    assert matrix._stable_runtime_identity_value(runtime) != (
        matrix._stable_runtime_identity_value(changed_path)
    )
    changed_sha = copy.deepcopy(equivalent)
    changed_sha["native_module"]["sha256"] = "c" * 64
    assert matrix._stable_runtime_identity_value(runtime) != (
        matrix._stable_runtime_identity_value(changed_sha)
    )
    changed_build = copy.deepcopy(equivalent)
    changed_build["build_info"]["payload"]["source_revision"] = "f" * 40
    assert matrix._stable_runtime_identity_value(runtime) != (
        matrix._stable_runtime_identity_value(changed_build)
    )


def test_cell_command_wires_physical_ordinal_selectors_and_ufo_model() -> None:
    topology = next(
        cell
        for cell in matrix.CANONICAL_CELLS
        if cell.model_kind == "ufo-sm"
        and cell.execution_mode == "compiled"
        and cell.workload_key == "lc-topology"
    )
    union = next(
        cell
        for cell in matrix.CANONICAL_CELLS
        if cell.model_kind == "ufo-sm"
        and cell.execution_mode == "eager"
        and cell.workload_key == "lc-union"
    )

    topology_command = matrix.cell_command(
        topology,
        baseline_python=BASELINE,
        current_python=CURRENT,
        output_root=Path("/matrix/output"),
        ufo_sm_model=UFO_SM,
    )
    union_command = matrix.cell_command(
        union,
        baseline_python=BASELINE,
        current_python=CURRENT,
        output_root=Path("/matrix/output"),
        ufo_sm_model=UFO_SM,
    )

    assert regression.parser().parse_args(topology_command[2:]).process == (
        topology.process
    )
    assert regression.parser().parse_args(union_command[2:]).process == union.process
    topology_arguments = regression.parser().parse_args(topology_command[2:])
    assert (
        topology_arguments.output_root
        == Path("/matrix/output/artifact-groups") / topology.artifact_group_id
    )
    assert (
        topology_arguments.result_path
        == Path("/matrix/output/cells") / topology.cell_id / "result.json"
    )
    assert topology_command[topology_command.index("--model") + 1] == str(UFO_SM)
    assert topology_command[topology_command.index("--color-flow") + 1] == (
        "flow:2,4,5,6,7,8,9,1"
    )
    assert "--helicity" not in topology_command
    assert union_command[union_command.index("--helicity") + 1] == (
        "h:-1,+1,-1,+1,-1,+1,-1,+1,-1"
    )
    assert "--color-flow" not in union_command
    assert (
        topology_command[topology_command.index("--jit-optimization-level") + 1] == "3"
    )
    assert union_command[union_command.index("--jit-optimization-level") + 1] == "2"


def test_complete_matrix_passes_all_aggregate_gates() -> None:
    audited = _audit(_passing_results())

    assert audited["passes"] is True
    assert audited["coverage"]["expected"] == 168
    assert audited["identity_gate"]["passes"] is True
    assert audited["gain_gate"]["passes"] is True
    assert audited["generation_gate"][
        "geometric_mean_current_over_baseline"
    ] == pytest.approx(1.04)


def test_in_progress_matrix_never_retains_a_passing_verdict() -> None:
    audited = _audit(_passing_results())
    assert audited["passes"] is True

    partial = matrix._mark_in_progress(
        audited,
        preflight_state={"sealed": True},
        output_root=OUTPUT_ROOT,
    )

    assert partial["complete"] is True
    assert partial["run_complete"] is False
    assert partial["passes"] is False
    assert partial["outer_provenance_gate"]["passes"] is False
    assert partial["provenance"]["postflight"] is None


def test_matrix_writes_in_progress_sentinel_before_first_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_python = tmp_path / "baseline-python"
    current_python = tmp_path / "current-python"
    for interpreter in (baseline_python, current_python):
        interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
        interpreter.chmod(0o755)
    ufo_sm = tmp_path / "sm"
    ufo_sm.mkdir()
    output_root = tmp_path / "matrix"
    expected_builds = {
        "baseline": {
            "source_revision": BASELINE_SOURCE_REVISION,
            "native_build_inputs_sha256": BASELINE_NATIVE_INPUTS,
            "distribution_content_sha256": BASELINE_DISTRIBUTION,
            "native_module_sha256": BASELINE_NATIVE_MODULE,
        },
        "current": {
            "source_revision": CURRENT_SOURCE_REVISION,
            "native_build_inputs_sha256": CURRENT_NATIVE_INPUTS,
            "distribution_content_sha256": CURRENT_DISTRIBUTION,
            "native_module_sha256": CURRENT_NATIVE_MODULE,
        },
    }
    preflight_state = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    monkeypatch.setattr(
        matrix,
        "_preflight",
        lambda *_args, **_kwargs: (expected_builds, preflight_state),
    )
    monkeypatch.setattr(matrix, "CANONICAL_CELLS", matrix.CANONICAL_CELLS[:1])
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(
        matrix,
        "_write_json_atomic",
        lambda _path, payload: writes.append(copy.deepcopy(dict(payload))),
    )

    def fail_first_cell(
        command: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert writes
        assert writes[-1]["run_complete"] is False
        assert writes[-1]["passes"] is False
        assert writes[-1]["outer_provenance_gate"]["passes"] is False
        return subprocess.CompletedProcess(command, 2, "", "deliberate stop")

    monkeypatch.setattr(matrix.subprocess, "run", fail_first_cell)
    arguments = SimpleNamespace(
        output_root=output_root,
        ufo_sm_model=ufo_sm,
        baseline_python=baseline_python,
        current_python=current_python,
        baseline_dependency_site=tmp_path / "baseline-site",
        current_dependency_site=tmp_path / "current-site",
        audit_only=False,
        rerun_results=True,
        regenerate_artifacts=True,
        samples=7,
        target_runtime=5.0,
        minimum_samples=7,
        warmup_runs=2,
        generation_timeout=2400.0,
        profile_timeout=1200.0,
    )

    with pytest.raises(matrix.MatrixError, match="cell driver failed"):
        matrix.run_matrix(arguments)


def test_matrix_fails_closed_for_missing_or_unexpected_cells() -> None:
    results = _passing_results()
    missing_id = next(iter(results))
    del results[missing_id]
    results["stale-cell"] = {}

    audited = _audit(results)

    assert audited["passes"] is False
    assert audited["coverage"]["passes"] is False
    assert audited["coverage"]["missing"] == [missing_id]
    assert audited["coverage"]["unexpected"] == ["stale-cell"]


def test_matrix_requires_gain_in_each_execution_mode() -> None:
    results = _passing_results()
    for cell in matrix.CANONICAL_CELLS:
        if cell.execution_mode == "eager":
            result = results[cell.cell_id]
            result["gain_gate"] = {
                "relative_gain": 0.2,
                "at_least_ten_percent": True,
                "beyond_measurement_noise": False,
                "passes": False,
            }

    audited = _audit(results)

    assert audited["passes"] is False
    assert audited["gain_gate"]["passing_cells_by_execution_mode"]["eager"] == []
    assert audited["gain_gate"]["passes"] is False


def test_matrix_requires_gain_from_a_primary_workload_in_each_mode() -> None:
    results = {cell.cell_id: _synthetic_result(cell) for cell in matrix.CANONICAL_CELLS}
    for mode in matrix.EXECUTION_MODES:
        cell = next(
            candidate
            for candidate in matrix.CANONICAL_CELLS
            if candidate.category == "medium" and candidate.execution_mode == mode
        )
        results[cell.cell_id] = _synthetic_result(cell, gain=True)

    audited = _audit(results)

    assert audited["passes"] is False
    assert all(
        audited["gain_gate"]["passing_cells_by_execution_mode"][mode]
        for mode in matrix.EXECUTION_MODES
    )
    assert all(
        audited["gain_gate"]["passing_primary_cells_by_execution_mode"][mode] == []
        for mode in matrix.EXECUTION_MODES
    )


def test_matrix_rejects_diagnostic_timing_methodology() -> None:
    results = _passing_results()
    changed_cell = next(iter(results))
    result = results[changed_cell]
    result["configuration"]["independent_samples_per_lane"] = 5
    result["configuration"]["target_runtime_per_native_sample_seconds"] = 0.1
    result["configuration"]["minimum_native_timed_blocks_per_profile"] = 5
    result["measurements"] = result["measurements"][:10]
    result["pair_orders"] = result["pair_orders"][:5]

    audited = _audit(results)

    assert audited["passes"] is False
    assert changed_cell in audited["cell_gate"]["failures"]


def test_matrix_recomputes_arena_profile_evidence_instead_of_trusting_gate() -> None:
    results = _passing_results()
    changed_cell = next(iter(results))
    result = results[changed_cell]
    assert result["arena_profile_gate"]["passes"] is True
    for measurement in result["measurements"]:
        if measurement["lane"] != "current":
            continue
        measurement.pop("profile_attribution_phase_timing_scope")
        breakdown = measurement["paired_profile_timing_breakdown"]
        breakdown["raw_profile_samples"][0].pop("phase_timing_scope")

    audited = _audit(results)

    assert audited["passes"] is False
    failures = audited["cell_gate"]["failures"][changed_cell]
    assert "arena profile evidence fails independent matrix recomputation" in failures


def test_matrix_rejects_synthetic_nested_evaluator_zero() -> None:
    results = _passing_results()
    changed_cell = next(iter(results))
    result = results[changed_cell]
    assert result["arena_profile_gate"]["passes"] is True
    for measurement in result["measurements"]:
        if measurement["lane"] == "current":
            measurement["paired_profile_timing_breakdown"]["evaluator_call_time"] = {
                "mean_seconds_per_point": 0.0
            }

    audited = _audit(results)

    assert audited["passes"] is False
    failures = audited["cell_gate"]["failures"][changed_cell]
    assert "arena profile evidence fails independent matrix recomputation" in failures


def test_matrix_rejects_truncated_raw_arena_profile_vector() -> None:
    results = _passing_results()
    changed_cell = next(iter(results))
    result = results[changed_cell]
    assert result["arena_profile_gate"]["passes"] is True
    for measurement in result["measurements"]:
        if measurement["lane"] == "current":
            measurement["paired_profile_timing_breakdown"]["raw_profile_samples"] = [
                measurement["paired_profile_timing_breakdown"]["raw_profile_samples"][0]
            ]

    audited = _audit(results)

    assert audited["passes"] is False
    failures = audited["cell_gate"]["failures"][changed_cell]
    assert "arena profile evidence fails independent matrix recomputation" in failures


def test_matrix_rejects_phase_clocks_under_coarse_arena_scope() -> None:
    results = _passing_results()
    changed_cell = next(iter(results))
    result = results[changed_cell]
    assert result["arena_profile_gate"]["passes"] is True
    for measurement in result["measurements"]:
        if measurement["lane"] != "current":
            continue
        for profile in measurement["paired_profile_timing_breakdown"][
            "raw_profile_samples"
        ]:
            profile["stage_evaluator_call_time_s"] = 1.0e-9

    audited = _audit(results)

    assert audited["passes"] is False
    failures = audited["cell_gate"]["failures"][changed_cell]
    assert "arena profile evidence fails independent matrix recomputation" in failures


def test_matrix_rejects_an_identically_zero_lc_selector() -> None:
    results = _passing_results()
    changed_cell = next(
        cell for cell in matrix.CANONICAL_CELLS if cell.workload != "summed"
    )
    for measurement in results[changed_cell.cell_id]["measurements"]:
        measurement["warmed_numerical_result"]["values_f64"] = [
            0.0
        ] * regression.VALIDATION_SAMPLE_COUNT

    audited = _audit(results)

    assert audited["passes"] is False
    assert changed_cell.cell_id in audited["cell_gate"]["failures"]


def test_matrix_rejects_generation_geometric_mean_above_five_percent() -> None:
    results = _passing_results()
    for result in results.values():
        result["resource_gate"]["generation"]["current_over_baseline"] = 1.051

    audited = _audit(results)

    assert audited["passes"] is False
    assert audited["generation_gate"][
        "geometric_mean_current_over_baseline"
    ] == pytest.approx(1.051)
    assert audited["generation_gate"]["passes"] is False


def test_matrix_rejects_runtime_identity_drift() -> None:
    results = _passing_results()
    changed = copy.deepcopy(next(iter(results.values())))
    for measurement in changed["measurements"]:
        if measurement["lane"] == "current":
            measurement["runtime_identity"]["extra_build_marker"] = "changed"
    changed_cell = next(iter(results))
    results[changed_cell] = changed

    audited = _audit(results)

    assert audited["passes"] is False
    assert "runtime:current" in audited["identity_gate"]["failures"]


def test_matrix_rejects_a_stable_but_unpinned_runtime() -> None:
    results = _passing_results()
    for result in results.values():
        for measurement in result["measurements"]:
            if measurement["lane"] == "current":
                measurement["runtime_identity"]["build_info"]["payload"][
                    "source_revision"
                ] = "3" * 40

    audited = _audit(results)

    assert audited["passes"] is False
    assert all(
        "current source revision does not match its pin" in failures
        for failures in audited["cell_gate"]["failures"].values()
    )


def test_matrix_rejects_a_stable_but_unpinned_distribution() -> None:
    results = _passing_results()
    for result in results.values():
        result["artifacts"]["current"]["installation_identity"]["distribution_content"][
            "sha256"
        ] = "6" * 64

    audited = _audit(results)

    assert audited["passes"] is False
    assert all(
        "current distribution content does not match its pin" in failures
        for failures in audited["cell_gate"]["failures"].values()
    )


def test_matrix_binds_results_to_output_root_and_platform() -> None:
    results = _passing_results()
    changed_cell = next(iter(results))
    results[changed_cell]["configuration"]["output_root"] = "/copied/result"
    results[changed_cell]["platform"] = "different-host"

    audited = _audit(results)

    assert audited["passes"] is False
    failures = audited["cell_gate"]["failures"][changed_cell]
    assert "configuration.output_root does not match the matrix" in failures
    assert "result platform does not match the acceptance host" in failures


def test_preflight_rejects_a_dirty_or_wrong_current_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = SimpleNamespace(
        baseline_python=BASELINE,
        current_python=CURRENT,
        baseline_dependency_site=Path("/baseline/site"),
        current_dependency_site=Path("/current/site"),
        expected_baseline_source_revision=matrix.FROZEN_BASELINE_SOURCE_REVISION,
        expected_baseline_native_inputs_sha256=(
            matrix.FROZEN_BASELINE_NATIVE_INPUTS_SHA256
        ),
        expected_baseline_distribution_sha256=(
            matrix.FROZEN_BASELINE_DISTRIBUTION_SHA256
        ),
        expected_baseline_native_module_sha256=(
            matrix.FROZEN_BASELINE_NATIVE_MODULE_SHA256
        ),
        expected_current_source_revision=CURRENT_SOURCE_REVISION,
        expected_current_native_inputs_sha256=CURRENT_NATIVE_INPUTS,
        expected_current_distribution_sha256=CURRENT_DISTRIBUTION,
        expected_current_native_module_sha256=CURRENT_NATIVE_MODULE,
    )
    state = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "repository": {
            "head_revision": CURRENT_SOURCE_REVISION,
            "clean": False,
            "dirty_entries": [" M harness.py"],
        },
    }
    monkeypatch.setattr(matrix, "_acceptance_state", lambda **_kwargs: state)
    with pytest.raises(matrix.MatrixError, match="not clean"):
        matrix._preflight(arguments, ufo_sm_model=UFO_SM)

    state["repository"] = {
        "head_revision": "3" * 40,
        "clean": True,
        "dirty_entries": [],
    }
    with pytest.raises(matrix.MatrixError, match="does not match clean checkout HEAD"):
        matrix._preflight(arguments, ufo_sm_model=UFO_SM)


def test_preflight_accepts_exact_caller_pinned_x86_baseline_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x86_distribution = "5" * 64
    x86_native_module = "6" * 64
    arguments = SimpleNamespace(
        baseline_python=BASELINE,
        current_python=CURRENT,
        baseline_dependency_site=Path("/baseline/site"),
        current_dependency_site=Path("/current/site"),
        expected_baseline_source_revision=matrix.FROZEN_BASELINE_SOURCE_REVISION,
        expected_baseline_native_inputs_sha256=(
            matrix.FROZEN_BASELINE_NATIVE_INPUTS_SHA256
        ),
        expected_baseline_distribution_sha256=x86_distribution,
        expected_baseline_native_module_sha256=x86_native_module,
        expected_current_source_revision=CURRENT_SOURCE_REVISION,
        expected_current_native_inputs_sha256=CURRENT_NATIVE_INPUTS,
        expected_current_distribution_sha256=CURRENT_DISTRIBUTION,
        expected_current_native_module_sha256=CURRENT_NATIVE_MODULE,
    )
    baseline_installation = _synthetic_installation("baseline")
    baseline_installation["distribution_content"]["sha256"] = x86_distribution
    baseline_installation["native_modules"][0]["sha256"] = x86_native_module
    baseline_payload = baseline_installation["build_info_files"][0]["payload"]
    baseline_payload["source_revision"] = matrix.FROZEN_BASELINE_SOURCE_REVISION
    baseline_payload["native_build_inputs_sha256"] = (
        matrix.FROZEN_BASELINE_NATIVE_INPUTS_SHA256
    )
    state = {
        "platform": "Linux-x86_64",
        "system": "Linux",
        "machine": "x86_64",
        "repository": {
            "head_revision": CURRENT_SOURCE_REVISION,
            "clean": True,
            "dirty_entries": [],
        },
        "installed_pyamplicol": {
            "baseline": baseline_installation,
            "current": _synthetic_installation("current"),
        },
    }
    monkeypatch.setattr(matrix, "_acceptance_state", lambda **_kwargs: state)

    expected, observed = matrix._preflight(arguments, ufo_sm_model=UFO_SM)

    assert observed is state
    assert expected["baseline"]["distribution_content_sha256"] == x86_distribution
    assert expected["baseline"]["native_module_sha256"] == x86_native_module


def test_matrix_cli_requires_dependency_sites_and_has_long_cell_timeouts() -> None:
    arguments = matrix.parser().parse_args(
        [
            "--baseline-python",
            str(BASELINE),
            "--current-python",
            str(CURRENT),
            "--ufo-sm-model",
            str(UFO_SM),
            "--output-root",
            str(OUTPUT_ROOT),
            "--expected-baseline-source-revision",
            matrix.FROZEN_BASELINE_SOURCE_REVISION,
            "--expected-current-source-revision",
            CURRENT_SOURCE_REVISION,
            "--expected-baseline-native-inputs-sha256",
            matrix.FROZEN_BASELINE_NATIVE_INPUTS_SHA256,
            "--expected-current-native-inputs-sha256",
            CURRENT_NATIVE_INPUTS,
            "--expected-baseline-distribution-sha256",
            matrix.FROZEN_BASELINE_DISTRIBUTION_SHA256,
            "--expected-current-distribution-sha256",
            CURRENT_DISTRIBUTION,
            "--expected-baseline-native-module-sha256",
            matrix.FROZEN_BASELINE_NATIVE_MODULE_SHA256,
            "--expected-current-native-module-sha256",
            CURRENT_NATIVE_MODULE,
            "--baseline-dependency-site",
            "/baseline/site",
            "--current-dependency-site",
            "/current/site",
        ]
    )

    assert arguments.baseline_dependency_site == Path("/baseline/site")
    assert arguments.current_dependency_site == Path("/current/site")
    assert arguments.generation_timeout >= 1800.0
    assert arguments.profile_timeout >= 900.0


def test_helicity_ordinal_resolves_against_runtime_inventory(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "artifact.json").write_text(
        json.dumps(
            {
                "processes": [
                    {
                        "id": "process",
                        "expression": "d d~ > z",
                        "physics_path": "runtime-physics.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (artifact / "runtime-physics.json").write_text(
        json.dumps(
            {
                "helicities": [
                    {"id": "h:-1,-1", "structural_zero": True},
                    {"id": "h:-1,+1", "structural_zero": False},
                    {"id": "h:+1,-1"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert compiled_mode_sample._resolve_helicities(
        artifact,
        process="process",
        requested=("1", "h:+1,-1"),
    ) == ("h:-1,+1", "h:+1,-1")
    with pytest.raises(compiled_mode_sample.SampleError, match="out of range"):
        compiled_mode_sample._resolve_helicities(
            artifact,
            process="process",
            requested=("4",),
        )
