# SPDX-License-Identifier: 0BSD
"""Public generation/load canary for compact LC recurrence execution."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
from collections.abc import Iterator
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import pyamplicol.generation.service as generation_service
from pyamplicol import CompiledModel, Generator, ModelSource, Runtime
from pyamplicol.api.errors import ArtifactError, EvaluationError
from pyamplicol.artifacts import inspect_artifact, load_manifest
from pyamplicol.artifacts.process_groups import (
    read_process_groups,
    write_process_groups,
)
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)
from pyamplicol.models.builtin.validation import generic_validation_point
from pyamplicol.reporting import CallbackProgressSink, ProgressUpdate
from pyamplicol.runtime.recurrence_exact._plan import _validate_execution

_PROCESS = "d d~ > z g g"
_THREE_LINE_PROCESS = "d d~ > u u~ s s~"
_PURE_GLUON_PROCESS = "g g > g g"
_N5_PURE_GLUON_PROCESS = "g g > g g g g g"
_SAME_FLAVOUR_PROCESS = "d d~ > d d~"
_NEUTRAL_CURRENT_PROCESS = "d d~ > e+ e-"
_FOUR_LEPTON_PROCESS = "d d~ > e+ e- e+ e-"
_FOUR_LEPTON_MADGRAPH_SEED_101 = Decimal("2.66945931894745456e-16")
_DILEPTON_ZH_PROCESS = "d d~ > e+ e- z h"
_CHARGED_CURRENT_PROCESS = "u d~ > e+ ve"
_TWO_QUARK_LINE_PROCESS = "d d~ > t t~"
_RELATION_REUSE_PROCESS = "d d~ > t t~ g g"
# The immutable physics-v2 oracle was captured with the former rounded default.
_HISTORICAL_REFERENCE_ALPHA_EW = 0.007546771114
_CONTRACTED_COLOR_PROCESSES = (
    _PROCESS,
    _THREE_LINE_PROCESS,
    _PURE_GLUON_PROCESS,
    _SAME_FLAVOUR_PROCESS,
    _TWO_QUARK_LINE_PROCESS,
)
_TOPOLOGY_REPLAY_PROCESSES = (
    "d d~ > z g",
    _PROCESS,
)
_TOPOLOGY_REPLAY_STRUCTURE = {
    "d d~ > z g": (31, 34, 12),
    _PROCESS: (69, 126, 24),
}
_RECURRENCE_KIND = "pyamplicol-runtime-recurrence-execution"
_RECURRENCE_CAPABILITIES = {
    "rusticol.recurrence-color.lc.v1",
    "rusticol.recurrence-direct-arena.complex-f64.v1",
}
_CONTRACTED_RECURRENCE_CAPABILITIES = {
    "rusticol.recurrence-color.contracted.v1",
    "rusticol.recurrence-direct-arena.complex-f64.v1",
    "rusticol.recurrence-helicity-selector-companion.v2",
}
_UFO_SM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)
_REFERENCE_PAYLOAD = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "reference"
        / "physics-v2.json"
    ).read_text(encoding="utf-8")
)

_Point = tuple[tuple[float, ...], ...]
_Points = tuple[_Point, ...]


def _unavailable(reason: str) -> None:
    if os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _require_native_recurrence() -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        _unavailable("Symbolica is unavailable")
    rusticol = importlib.import_module("pyamplicol._rusticol")
    if not hasattr(rusticol, "_lower_recurrence_direct_v2"):
        _unavailable("the installed Rusticol extension lacks recurrence lowering")


def _generation_config(
    execution_mode: str,
    *,
    color_accuracy: str = "lc",
    lc_flow_layout: str = "topology-replay",
    relation_discovery_mode: str | None = "off",
) -> RunConfig:
    relation_discovery = GenerationRelationDiscoveryConfig(
        precision_digits=80,
        probe_count=2,
        verification_probe_count=2,
        seed=17,
    )
    if relation_discovery_mode is not None:
        relation_discovery = replace(
            relation_discovery,
            mode=relation_discovery_mode,
        )
    return RunConfig(
        action="generate",
        color=ColorConfig(
            accuracy=color_accuracy,
            # This suite asserts the direct-contraction artifact structure.
            # Automatic FFT selection is covered by test_adjoint_fft_runtime.
            contraction="direct",
            lc_flow_layout=lc_flow_layout,
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=relation_discovery,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(
                optimization_level=2 if execution_mode == "recurrence" else 1
            ),
        ),
    )


def _flatten(
    values: tuple[
        tuple[tuple[complex | Decimal, ...], ...],
        ...,
    ],
) -> tuple[complex, ...]:
    return tuple(
        complex(value) for point in values for helicity in point for value in helicity
    )


def _validation_points(process_expression: str) -> _Points:
    return (
        tuple(
            tuple(float(component) for component in particle.momentum)
            for particle in generic_validation_point(process_expression)
        ),
    )


def _seeded_validation_points(
    process_expression: str,
    *seeds: int,
) -> _Points:
    return tuple(
        tuple(
            tuple(float(component) for component in particle.momentum)
            for particle in generic_validation_point(
                process_expression,
                seed=seed,
            )
        )
        for seed in seeds
    )


def _reference_case(case_id: str) -> dict[str, Any]:
    return next(case for case in _REFERENCE_PAYLOAD["cases"] if case["id"] == case_id)


def _reference_point(point_id: str) -> _Point:
    point = next(
        item for item in _REFERENCE_PAYLOAD["points"] if item["id"] == point_id
    )
    return tuple(
        tuple(float(component) for component in momentum)
        for momentum in point["momenta"]
    )


def _assert_recurrence_per_point_selector_patterns(
    artifact: Path,
    point: _Point,
    *,
    expected_layout: str,
) -> None:
    """Exercise recurrence-native selector grouping without a compiled fallback."""

    execution_files = tuple((artifact / "processes").glob("*/execution.json"))
    assert len(execution_files) == 1
    execution = json.loads(execution_files[0].read_text(encoding="utf-8"))
    assert execution["kind"] == _RECURRENCE_KIND
    assert execution["recurrence_plan_abi"] == "pyamplicol-recurrence-plan-v2"
    assert execution["runtime_layout_abi"] == "pyamplicol-recurrence-runtime-layout-v2"
    assert execution["recurrence_summary"]["lc_flow_layout"] == expected_layout
    assert set(execution["required_runtime_capabilities"]) == _RECURRENCE_CAPABILITIES
    assert inspect_artifact(artifact).processes[0].execution_mode == "recurrence"

    runtime = Runtime.load(artifact)
    resolved = runtime.evaluate_resolved((point,))
    helicity_ids = tuple(
        helicity.id
        for helicity in runtime.physics.helicities
        if not helicity.structural_zero
    )
    color_ids = resolved.color_ids
    assert len(helicity_ids) >= 2
    assert len(color_ids) >= 2
    selector_pairs = (
        (helicity_ids[0], color_ids[0]),
        (helicity_ids[-1], color_ids[-1]),
    )
    patterns = {
        "homogeneous": (selector_pairs[0],) * 8,
        "alternating": selector_pairs * 4,
        "random": tuple(selector_pairs[index] for index in (1, 0, 1, 1, 0, 0, 1, 0)),
        "pre-grouped": (selector_pairs[0],) * 4 + (selector_pairs[1],) * 4,
    }

    for name, selectors in patterns.items():
        points = (point,) * len(selectors)
        actual = runtime.evaluate(
            points,
            helicity_by_point=tuple(selector[0] for selector in selectors),
            color_flow_by_point=tuple(selector[1] for selector in selectors),
        )
        expected = tuple(
            runtime.evaluate(
                (point,),
                helicities=(helicity_id,),
                color_flows=(color_id,),
            )[0]
            for helicity_id, color_id in selectors
        )
        assert actual == pytest.approx(
            expected,
            rel=1.0e-12,
            abs=1.0e-15,
        ), name


def _assert_topology_replay_structure(
    artifact: Path,
    process_expression: str,
) -> None:
    expected_currents, expected_contributions, expected_closures = (
        _TOPOLOGY_REPLAY_STRUCTURE[process_expression]
    )
    manifest = load_manifest(artifact)
    assert len(manifest.processes) == 1
    process_id = str(manifest.processes[0]["id"])
    execution = json.loads(
        (artifact / "processes" / process_id / "execution.json").read_text(
            encoding="utf-8"
        )
    )
    summary = execution["recurrence_summary"]
    assert (
        summary["current_count"],
        summary["contribution_count"],
        summary["closure_term_count"],
    ) == (
        expected_currents,
        expected_contributions,
        expected_closures,
    )

    inspection = inspect_artifact(artifact).processes[0]
    assert inspection.invocation_count == expected_contributions
    assert inspection.direct_contribution_row_count == expected_contributions
    assert inspection.closure_count == expected_closures
    assert inspection.direct_closure_row_count == expected_closures


def _assert_topology_replay_artifacts_match(
    recurrence_artifact: Path,
    compiled_artifact: Path,
    process_expression: str,
) -> tuple[Runtime, Runtime, _Points]:
    points = _validation_points(process_expression)
    recurrence = Runtime.load(recurrence_artifact)
    compiled = Runtime.load(compiled_artifact)

    recurrence_total = recurrence.evaluate(points)
    recurrence_resolved = recurrence.evaluate_resolved(points)
    compiled_total = compiled.evaluate(points)
    compiled_resolved = compiled.evaluate_resolved(points)

    assert recurrence_resolved.total() == pytest.approx(
        recurrence_total,
        rel=1.0e-13,
        abs=1.0e-15,
    )
    assert compiled_resolved.total() == pytest.approx(
        compiled_total,
        rel=1.0e-13,
        abs=1.0e-15,
    )
    assert recurrence_resolved.helicity_ids == compiled_resolved.helicity_ids
    assert recurrence_resolved.color_ids == compiled_resolved.color_ids
    assert recurrence_resolved.shape == compiled_resolved.shape
    assert _flatten(recurrence_resolved.values) == pytest.approx(
        _flatten(compiled_resolved.values),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert recurrence_total == pytest.approx(
        compiled_total,
        rel=1.0e-12,
        abs=1.0e-15,
    )

    # Public flow IDs are not recurrence construction-sector IDs. Exercise
    # every public selector through both resolved and optimized runtime paths.
    for color_id in recurrence_resolved.color_ids:
        recurrence_selected = recurrence.evaluate(points, color_flows=(color_id,))
        compiled_selected = compiled.evaluate(points, color_flows=(color_id,))
        recurrence_selected_resolved = recurrence.evaluate_resolved(
            points,
            color_flows=(color_id,),
        )
        compiled_selected_resolved = compiled.evaluate_resolved(
            points,
            color_flows=(color_id,),
        )
        assert recurrence_selected_resolved.helicity_ids == (
            compiled_selected_resolved.helicity_ids
        )
        assert recurrence_selected_resolved.color_ids == (
            compiled_selected_resolved.color_ids
        )
        assert recurrence_selected_resolved.shape == compiled_selected_resolved.shape
        assert _flatten(recurrence_selected_resolved.values) == pytest.approx(
            _flatten(compiled_selected_resolved.values),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        assert recurrence_selected == pytest.approx(
            compiled_selected,
            rel=1.0e-12,
            abs=1.0e-15,
        )

    return recurrence, compiled, points


def _assert_decimal_values_match(
    actual: object,
    expected: object,
    precision: int,
) -> None:
    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    assert len(actual) == len(expected)
    relative_tolerance = Decimal("1e-12")
    absolute_tolerance = Decimal("1e-15")
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if isinstance(actual_value, tuple):
            _assert_decimal_values_match(actual_value, expected_value, precision)
            continue
        assert isinstance(actual_value, Decimal)
        assert isinstance(expected_value, Decimal)
        assert abs(actual_value - expected_value) <= (
            absolute_tolerance + relative_tolerance * abs(expected_value)
        )


def _assert_scale_sensitive_values_match(
    actual: object,
    expected: object,
    *,
    relative_tolerance: Decimal = Decimal("1e-12"),
) -> None:
    """Compare tiny exact values without the report's absolute tolerance floor."""

    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if isinstance(actual_value, tuple):
            _assert_scale_sensitive_values_match(
                actual_value,
                expected_value,
                relative_tolerance=relative_tolerance,
            )
            continue
        assert isinstance(actual_value, Decimal)
        assert isinstance(expected_value, Decimal)
        scale = max(abs(actual_value), abs(expected_value))
        if scale == 0:
            assert actual_value == expected_value
        else:
            assert abs(actual_value - expected_value) <= relative_tolerance * scale


def _single_recurrence_execution(artifact: Path) -> dict[str, Any]:
    execution_paths = tuple((artifact / "processes").glob("*/execution.json"))
    assert len(execution_paths) == 1
    execution = json.loads(execution_paths[0].read_text(encoding="utf-8"))
    assert execution["kind"] == _RECURRENCE_KIND
    return execution


def _manifest_relation_discovery(artifact: Path) -> object:
    manifest = load_manifest(artifact)
    generation = manifest.extensions["generation"]
    assert isinstance(generation, dict)
    concrete_processes = generation["concrete_processes"]
    assert isinstance(concrete_processes, list)
    assert len(concrete_processes) == 1
    process = concrete_processes[0]
    assert isinstance(process, dict)
    filters = process["filters"]
    assert isinstance(filters, dict)
    return filters.get("relation_discovery")


def _assert_runtime_values_match(
    actual: Runtime,
    expected: Runtime,
    points: _Points,
) -> None:
    actual_resolved = actual.evaluate_resolved(points)
    expected_resolved = expected.evaluate_resolved(points)
    assert actual_resolved.helicity_ids == expected_resolved.helicity_ids
    assert actual_resolved.color_ids == expected_resolved.color_ids
    assert actual_resolved.shape == expected_resolved.shape
    assert _flatten(actual_resolved.values) == pytest.approx(
        _flatten(expected_resolved.values),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert actual_resolved.total() == pytest.approx(
        actual.evaluate(points),
        rel=1.0e-13,
        abs=1.0e-15,
    )
    assert actual.evaluate(points) == pytest.approx(
        expected.evaluate(points),
        rel=1.0e-12,
        abs=1.0e-15,
    )

    actual_exact = actual.evaluate_resolved(points, precision=50)
    expected_exact = expected.evaluate_resolved(points, precision=50)
    assert actual_exact.helicity_ids == expected_exact.helicity_ids
    assert actual_exact.color_ids == expected_exact.color_ids
    assert actual_exact.shape == expected_exact.shape
    _assert_decimal_values_match(
        actual_exact.values,
        expected_exact.values,
        50,
    )
    _assert_decimal_values_match(
        actual_exact.total(),
        actual.evaluate(points, precision=50),
        50,
    )
    _assert_decimal_values_match(
        actual.evaluate(points, precision=50),
        expected.evaluate(points, precision=50),
        50,
    )


def _assert_topology_replay_exact_matches_compiled(
    recurrence: Runtime,
    compiled: Runtime,
    points: _Points,
) -> None:
    for precision in (32, 50):
        recurrence_resolved = recurrence.evaluate_resolved(
            points,
            precision=precision,
        )
        compiled_resolved = compiled.evaluate_resolved(
            points,
            precision=precision,
        )
        assert recurrence_resolved.helicity_ids == compiled_resolved.helicity_ids
        assert recurrence_resolved.color_ids == compiled_resolved.color_ids
        assert recurrence_resolved.shape == compiled_resolved.shape
        _assert_decimal_values_match(
            recurrence_resolved.values,
            compiled_resolved.values,
            precision,
        )
        _assert_decimal_values_match(
            recurrence_resolved.total(),
            recurrence.evaluate(points, precision=precision),
            precision,
        )

        # Requesting every public ID explicitly must preserve the complete
        # resolved result, and each physical flow remains independently usable.
        explicit = recurrence.evaluate_resolved(
            points,
            helicities=recurrence_resolved.helicity_ids,
            color_flows=recurrence_resolved.color_ids,
            precision=precision,
        )
        _assert_decimal_values_match(
            explicit.values,
            recurrence_resolved.values,
            precision,
        )
        for color_id in recurrence_resolved.color_ids:
            actual = recurrence.evaluate_resolved(
                points,
                color_flows=(color_id,),
                precision=precision,
            )
            expected = compiled.evaluate_resolved(
                points,
                color_flows=(color_id,),
                precision=precision,
            )
            _assert_decimal_values_match(actual.values, expected.values, precision)

        sampled_helicities = tuple(
            recurrence_resolved.helicity_ids[index]
            for index in sorted(
                {
                    0,
                    len(recurrence_resolved.helicity_ids) // 2,
                    len(recurrence_resolved.helicity_ids) - 1,
                }
            )
        )
        actual = recurrence.evaluate_resolved(
            points,
            helicities=sampled_helicities,
            precision=precision,
        )
        expected = compiled.evaluate_resolved(
            points,
            helicities=sampled_helicities,
            precision=precision,
        )
        _assert_decimal_values_match(actual.values, expected.values, precision)

        doubled_points = points + points
        helicity_by_point = (
            recurrence_resolved.helicity_ids[0],
            recurrence_resolved.helicity_ids[-1],
        )
        color_by_point = (
            recurrence_resolved.color_ids[0],
            recurrence_resolved.color_ids[-1],
        )
        _assert_decimal_values_match(
            recurrence.evaluate(
                doubled_points,
                helicity_by_point=helicity_by_point,
                color_flow_by_point=color_by_point,
                precision=precision,
            ),
            compiled.evaluate(
                doubled_points,
                helicity_by_point=helicity_by_point,
                color_flow_by_point=color_by_point,
                precision=precision,
            ),
            precision,
        )


def _assert_contracted_color_artifacts_match(
    recurrence_artifact: Path,
    compiled_artifact: Path,
    process_expression: str,
    color_accuracy: str,
    *,
    parameter_update: tuple[str, float] | None = None,
) -> None:
    points = _validation_points(process_expression)
    recurrence = Runtime.load(recurrence_artifact)
    compiled = Runtime.load(compiled_artifact)
    assert recurrence.physics.color_accuracy == color_accuracy
    assert recurrence.physics.color_ids == ("color:contracted",)
    assert compiled.physics.color_ids == recurrence.physics.color_ids
    assert compiled.physics.helicity_ids == recurrence.physics.helicity_ids

    recurrence_total = recurrence.evaluate(points)
    compiled_total = compiled.evaluate(points)
    recurrence_resolved = recurrence.evaluate_resolved(points)
    compiled_resolved = compiled.evaluate_resolved(points)
    assert recurrence_resolved.total() == pytest.approx(
        recurrence_total,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert recurrence_total == pytest.approx(
        compiled_total,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert recurrence_resolved.shape == compiled_resolved.shape
    assert _flatten(recurrence_resolved.values) == pytest.approx(
        _flatten(compiled_resolved.values),
        rel=1.0e-12,
        abs=1.0e-15,
    )

    nonzero_helicities = tuple(
        helicity.id
        for helicity in recurrence.physics.helicities
        if not helicity.structural_zero
    )
    assert nonzero_helicities
    sampled_helicities = tuple(
        dict.fromkeys(
            (
                nonzero_helicities[0],
                nonzero_helicities[len(nonzero_helicities) // 2],
                nonzero_helicities[-1],
            )
        )
    )
    recurrence_selected = recurrence.evaluate_resolved(
        points,
        helicities=sampled_helicities,
    )
    compiled_selected = compiled.evaluate_resolved(
        points,
        helicities=sampled_helicities,
    )
    assert _flatten(recurrence_selected.values) == pytest.approx(
        _flatten(compiled_selected.values),
        rel=1.0e-12,
        abs=1.0e-15,
    )

    repeated_points = points * len(sampled_helicities)
    assert recurrence.evaluate(
        repeated_points,
        helicity_by_point=sampled_helicities,
    ) == pytest.approx(
        compiled.evaluate(
            repeated_points,
            helicity_by_point=sampled_helicities,
        ),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    with pytest.raises(EvaluationError, match="color-flow selection"):
        recurrence.evaluate(points, color_flows=("color:contracted",))

    for precision in (32, 50):
        recurrence_exact = recurrence.evaluate_resolved(
            points,
            helicities=sampled_helicities,
            precision=precision,
        )
        recurrence_f64 = recurrence.evaluate_resolved(
            points,
            helicities=sampled_helicities,
        )
        assert _flatten(recurrence_exact.values) == pytest.approx(
            _flatten(recurrence_f64.values),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        _assert_decimal_values_match(
            recurrence_exact.total(),
            recurrence.evaluate(
                points,
                helicities=sampled_helicities,
                precision=precision,
            ),
            precision,
        )

    manifest = load_manifest(recurrence_artifact)
    assert len(manifest.processes) == 1
    process_id = str(manifest.processes[0]["id"])
    execution = json.loads(
        (recurrence_artifact / "processes" / process_id / "execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["recurrence_summary"]["lc_flow_layout"] == "contracted-color-union"
    assert (
        set(execution["required_runtime_capabilities"])
        == _CONTRACTED_RECURRENCE_CAPABILITIES
    )
    color_reference = execution["runtime_metadata"]["color_contraction"]
    inspection = inspect_artifact(recurrence_artifact).processes[0]
    assert inspection.execution_mode == "recurrence"
    assert inspection.recurrence_color_accuracy == color_accuracy
    assert inspection.recurrence_color_storage == color_reference["storage"]
    assert inspection.recurrence_color_sector_count == color_reference["sector_count"]
    assert (
        inspection.recurrence_color_active_sector_count
        == color_reference["active_sector_count"]
    )
    assert (
        inspection.recurrence_color_component_count
        == color_reference["component_count"]
    )
    assert inspection.recurrence_color_group_count == color_reference["group_count"]
    assert inspection.recurrence_color_entry_count == color_reference["entry_count"]
    assert (
        inspection.recurrence_color_logical_entry_count
        == color_reference["logical_entry_count"]
    )
    if process_expression == _THREE_LINE_PROCESS:
        # Six endpoint pairings are physical. The six orderings of each
        # disconnected open-string forest are coherent aliases and must have
        # one deterministic owner, not six duplicated amplitude destinations.
        assert color_reference["sector_count"] == 36
        assert color_reference["active_sector_count"] == 6
        assert color_reference["group_count"] == (
            6 * color_reference["component_count"]
        )
        assert color_reference["storage"] == "repeated"
    color_path = (
        recurrence_artifact / "processes" / process_id / color_reference["path"]
    )
    assert color_path.is_file()
    assert color_path.stat().st_size == color_reference["size_bytes"]
    payloads = {
        record.path: record
        for record in manifest.payloads
        if record.process_id == process_id
    }
    relative_color_path = color_path.relative_to(recurrence_artifact).as_posix()
    assert payloads[relative_color_path].sha256 == color_reference["sha256"]

    if parameter_update is not None:
        parameter_name, parameter_value = parameter_update
        exact_before_update = recurrence.evaluate(points, precision=50)
        recurrence.set_model_parameters({parameter_name: parameter_value})
        compiled.set_model_parameters({parameter_name: parameter_value})
        assert recurrence.evaluate(points, precision=50) != exact_before_update
        assert recurrence.evaluate(points) == pytest.approx(
            compiled.evaluate(points),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        recurrence_exact = recurrence.evaluate_resolved(points, precision=50)
        recurrence_f64 = recurrence.evaluate_resolved(points)
        assert _flatten(recurrence_exact.values) == pytest.approx(
            _flatten(recurrence_f64.values),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        _assert_decimal_values_match(
            recurrence_exact.total(),
            recurrence.evaluate(points, precision=50),
            50,
        )


def _contracted_structure_signature(artifact: Path) -> dict[str, object]:
    manifest = load_manifest(artifact)
    assert len(manifest.processes) == 1
    process_id = str(manifest.processes[0]["id"])
    execution = json.loads(
        (artifact / "processes" / process_id / "execution.json").read_text(
            encoding="utf-8"
        )
    )
    summary = execution["recurrence_summary"]
    inspection = execution["plan"]["inspection_summary"]
    schedule = inspection["schedule"]
    construction = inspection["construction"]
    color = execution["runtime_metadata"]["color_contraction"]
    return {
        "layout": summary["lc_flow_layout"],
        "currents": summary["current_count"],
        "contributions": summary["contribution_count"],
        "closures": summary["closure_term_count"],
        "sources": schedule["source_row_count"],
        "finalizations": schedule["finalization_count"],
        "destinations": schedule["amplitude_destination_count"],
        "resolved_helicities": schedule["resolved_helicity_count"],
        "peak_contribution_count_semantics": construction[
            "peak_contribution_count_semantics"
        ],
        "dynamic_color_states": construction["peak_dynamic_color_state_count"],
        "color_storage": color["storage"],
        "color_sectors": color["sector_count"],
        "active_color_sectors": color["active_sector_count"],
        "color_components": color["component_count"],
        "color_groups": color["group_count"],
        "color_entries": color["entry_count"],
        "logical_color_entries": color["logical_entry_count"],
        "factorization": color["factorization"],
    }


@pytest.fixture(scope="module")
def builtin_sm_recurrence_jit_o2_model() -> Iterator[ModelSource]:
    """Use a source-current prepared pack without weakening packaged checks."""

    override = os.environ.get("PYAMPLICOL_RECURRENCE_TEST_PREPARED_MODEL")
    if override:
        yield ModelSource.from_path(Path(override))
        return
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        yield ModelSource.from_path(prepared_model)


@pytest.fixture(scope="module")
def builtin_sm_current_recurrence_jit_o2_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> CompiledModel:
    """Prepare the live built-in lowering contract for alias regressions."""

    _require_native_recurrence()
    root = tmp_path_factory.mktemp("builtin-sm-current-recurrence-jit-o2")
    return ModelSource.built_in_sm().compile(
        cache_dir=root / "model-cache",
        use_cache=True,
        prepared_output=root / "built-in-sm-jit-o2.pyamplicol-model",
        evaluator=_generation_config("recurrence").evaluator,
    )


@pytest.fixture(scope="module")
def ufo_sm_recurrence_jit_o2_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> CompiledModel:
    """Prepare one reusable UFO-SM recurrence pack for both public canaries."""

    _require_native_recurrence()
    root = tmp_path_factory.mktemp("ufo-sm-recurrence-jit-o2")
    model = ModelSource.from_path(
        _UFO_SM_ROOT / "sm.json",
        restriction=_UFO_SM_ROOT / "restrict_default.json",
    ).compile(
        cache_dir=root / "model-cache",
        use_cache=True,
        prepared_output=root / "ufo-sm-jit-o2.pyamplicol-model",
        evaluator=_generation_config("recurrence").evaluator,
    )
    assert model.is_prepared
    assert model.prepared_backend == "jit"
    return model


@pytest.mark.parametrize("model_source", ("builtin", "ufo"))
@pytest.mark.parametrize("color_accuracy", ("lc", "nlc", "full"))
def test_relation_discovery_modes_preserve_recurrence_artifacts_and_values(
    tmp_path: Path,
    model_source: str,
    color_accuracy: str,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Relation probes may optimize exact reuse but never alter amplitudes."""

    _require_native_recurrence()
    model = (
        builtin_sm_recurrence_jit_o2_model
        if model_source == "builtin"
        else ufo_sm_recurrence_jit_o2_model
    )
    compiled_artifact = tmp_path / f"{model_source}-{color_accuracy}-compiled"
    Generator(
        _generation_config(
            "compiled",
            color_accuracy=color_accuracy,
        )
    ).generate(
        _PROCESS,
        compiled_artifact,
        model=model,
    )
    compiled = Runtime.load(compiled_artifact)
    points = _seeded_validation_points(_PROCESS, 101, 211)

    executions: dict[str, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    lanes: dict[str, dict[str, Any]] = {}
    runtimes: dict[str, Runtime] = {}
    for mode in ("off", "diagnostic", "certified-reuse"):
        artifact = tmp_path / f"{model_source}-{color_accuracy}-{mode}"
        Generator(
            _generation_config(
                "recurrence",
                color_accuracy=color_accuracy,
                relation_discovery_mode=mode,
            )
        ).generate(
            _PROCESS,
            artifact,
            model=model,
        )
        execution = _single_recurrence_execution(artifact)
        executions[mode] = execution
        runtime = Runtime.load(artifact)
        runtimes[mode] = runtime
        _assert_runtime_values_match(runtime, compiled, points)

        report = execution["plan"]["inspection_summary"].get("relation_discovery")
        manifest_report = _manifest_relation_discovery(artifact)
        assert report is None
        assert isinstance(manifest_report, dict)
        reports[mode] = manifest_report
        assert manifest_report["abi"] == (
            "pyamplicol-artifact-numerical-current-reuse-v1"
        )
        assert manifest_report["requested_mode"] == mode
        assert manifest_report["execution_mode"] == "recurrence"
        assert manifest_report["lane_count"] == 1
        lane = manifest_report["lanes"]["primary"]
        assert isinstance(lane, dict)
        lanes[mode] = lane
        assert lane["requested_mode"] == mode
        assert lane["scope"] == {
            "execution_mode": "recurrence",
            "color_accuracy": color_accuracy,
            "representation": "recurrence-direct-plan-v2",
        }
        assert (
            manifest_report["certified_relation_count"]
            == lane["certified_relation_count"]
        )
        assert (
            manifest_report["applied_relation_count"] == lane["applied_relation_count"]
        )
        if mode == "off":
            assert lane["state"] == "disabled-by-user"
            assert lane["candidate_capture"] is None
            assert lane["verification_capture"] is None
            assert lane["application_capture"] is None
            assert lane["warning"]["required"] is False
            assert manifest_report["warning"]["required"] is False
            assert manifest_report["native_relation_application"] is None
            continue

        candidate = lane["candidate_capture"]
        verification = lane["verification_capture"]
        assert isinstance(candidate, dict)
        assert candidate["precision_digits"] == 80
        assert candidate["point_count"] == 2
        discovery = lane["discovery"]
        assert isinstance(discovery, dict)
        probe_contract = discovery["probe_contract"]
        assert isinstance(probe_contract, dict)
        assert probe_contract["algorithm"] == (
            "authenticated-independent-recursive-decimal-raw-probes-v2"
        )
        assert probe_contract["current_dimension_bound"] is True
        assert (
            probe_contract["runtime_parameter_schema_sha256"]
            == candidate["runtime_parameter_schema_sha256"]
        )
        assert candidate["parameter_contexts"]
        if discovery["numerical_candidate_count"] == 0:
            assert verification is None
            assert probe_contract["independent_verification"] is False
            assert probe_contract["verification_status"] == (
                "not-required-no-candidate-hypothesis"
            )
            assert probe_contract["verification_point_sha256s"] == []
            assert probe_contract["verification_capture_sha256"] is None
        else:
            assert isinstance(verification, dict)
            assert verification["point_count"] == 2
            assert set(candidate["kinematic_sha256s"]).isdisjoint(
                verification["kinematic_sha256s"]
            )
            assert probe_contract["independent_verification"] is True
            assert verification["parameter_contexts"]
            assert set(candidate["parameter_context_sha256s"]).isdisjoint(
                verification["parameter_context_sha256s"]
            )
        persisted = lane["persisted_numerical_evidence"]
        assert persisted["raw_evidence_retained"] is False
        if verification is None:
            assert persisted["verification_capture"] is None
            assert persisted["generation_raw_evidence_bytes"] == 0
            assert persisted["generation_evidence_transport_bytes"] == 0
            assert persisted["generation_evidence_encoding"] is None
        else:
            assert persisted["generation_raw_evidence_bytes"] > 0
        assert persisted["measured_payload_bytes"] < 64 << 20
        assert (
            persisted["full_census"]["decision_sha256"] == discovery["decision_sha256"]
        )
        assert (
            discovery["certified_numerical_relation_count"]
            == lane["certified_relation_count"]
        )
        application = lane["application"]
        assert isinstance(application, dict)
        assert (
            application["certified_relation_count"] == lane["certified_relation_count"]
        )
        native = manifest_report["native_relation_application"]
        certified = lane["certified_relation_count"]
        applied = lane["applied_relation_count"]
        if certified == 0:
            assert applied == 0
            assert native is None
        else:
            assert isinstance(native, dict)
            assert native["requested_mode"] == lane["effective_mode"]
            assert native["exact_certified_relation_count"] == certified
        effective_mode = lane["effective_mode"]
        if effective_mode == "diagnostic":
            assert applied == 0
            assert lane["warning"]["required"] is False
            assert manifest_report["warning"]["required"] is False
            if certified:
                assert isinstance(native, dict)
                assert native["applied_relation_count"] == 0
                assert native["scale_copy_row_count"] == 0
            if mode == "certified-reuse":
                assert lane["effective_reuse_state"] == "disabled"
                assert lane["effective_mode_reason"] in {
                    "multi-helicity-all-flow-member-scope-unproven",
                    "contracted-opposite-binary64-disabled",
                }
                assert (
                    lane["application_scope"]["effective_mode_reason"]
                    == (lane["effective_mode_reason"])
                )
        else:
            assert effective_mode == mode == "certified-reuse"
            # Discovery retains the complete verified census, while contracted
            # colour applies only the supported equal/zero subset.
            certificates = application["certificates"]
            expected_applied = sum(
                certificate["relation_kind"] != "opposite" or color_accuracy == "lc"
                for certificate in certificates
            )
            assert applied == expected_applied
            if applied:
                assert isinstance(native, dict)
                assert native["applied_relation_count"] == applied
                assert native["scale_copy_row_count"] == applied
            assert lane["warning"]["required"] is (applied > 0)
            assert manifest_report["warning"]["required"] is (applied > 0)
            assert lane["application_validation"]["status"] == (
                "verified" if applied else "not-required-no-applied-relations"
            )

    _assert_runtime_values_match(runtimes["diagnostic"], runtimes["off"], points)
    _assert_runtime_values_match(
        runtimes["certified-reuse"],
        runtimes["off"],
        points,
    )

    off_inspection = executions["off"]["plan"]["inspection_summary"]
    diagnostic_inspection = executions["diagnostic"]["plan"]["inspection_summary"]
    assert (
        executions["off"]["recurrence_summary"]
        == executions["diagnostic"]["recurrence_summary"]
    )
    for key in (
        "schedule",
        "construction",
        "selector_work_certificate",
        "direct_arena",
    ):
        assert off_inspection.get(key) == diagnostic_inspection.get(key)
    for capture_name in ("candidate_capture", "verification_capture"):
        diagnostic_capture = lanes["diagnostic"][capture_name]
        certified_capture = lanes["certified-reuse"][capture_name]
        if diagnostic_capture is None or certified_capture is None:
            assert diagnostic_capture is certified_capture is None
            continue
        assert isinstance(diagnostic_capture, dict)
        assert isinstance(certified_capture, dict)
        for key in (
            "precision_digits",
            "point_count",
            "point_sha256s",
            "kinematic_sha256s",
            "context_sha256s",
            "current_count",
            "current_dimensions_sha256",
            "context_policy",
        ):
            assert diagnostic_capture[key] == certified_capture[key]
    diagnostic_discovery = lanes["diagnostic"]["discovery"]
    certified_discovery = lanes["certified-reuse"]["discovery"]
    assert isinstance(diagnostic_discovery, dict)
    assert isinstance(certified_discovery, dict)
    for key in (
        "tested_hypothesis_count",
        "numerical_candidate_count",
        "verification_rejected_count",
        "certified_numerical_relation_count",
    ):
        assert diagnostic_discovery[key] == certified_discovery[key]

    def relation_identity(certificate: dict[str, Any]) -> tuple[object, ...]:
        return (
            certificate["current_id"],
            certificate["representative_id"],
            certificate["execution_representative_id"],
            certificate["relation_kind"],
            certificate["factor_integer"],
            certificate["current_dimension"],
        )

    assert [
        relation_identity(certificate)
        for certificate in lanes["diagnostic"]["application"]["certificates"]
    ] == [
        relation_identity(certificate)
        for certificate in lanes["certified-reuse"]["application"]["certificates"]
    ]


def test_charged_current_alias_uses_native_identity_for_numerical_probes(
    tmp_path: Path,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep display ownership separate from the canonical native schedule ID."""

    _require_native_recurrence()
    observed_process_ids: list[str] = []
    original = generation_service.build_recurrence_numerical_current_probe_points

    def recording_probe_builder(*args: Any, **kwargs: Any) -> Any:
        process_id = str(kwargs["process_id"])
        result = original(*args, **kwargs)
        observed_process_ids.append(process_id)
        assert all(
            point.process_id == process_id for capture in result for point in capture
        )
        assert all(
            point.process == _CHARGED_CURRENT_PROCESS
            for capture in result
            for point in capture
        )
        return result

    monkeypatch.setattr(
        generation_service,
        "build_recurrence_numerical_current_probe_points",
        recording_probe_builder,
    )
    artifact = tmp_path / "charged-current-alias"
    Generator(
        _generation_config(
            "recurrence",
            relation_discovery_mode="certified-reuse",
        )
    ).generate(
        _CHARGED_CURRENT_PROCESS,
        artifact,
        model=builtin_sm_recurrence_jit_o2_model,
    )

    assert len(observed_process_ids) == 1
    canonical_process_id = observed_process_ids[0]
    execution = _single_recurrence_execution(artifact)
    display_process_id = str(load_manifest(artifact).processes[0]["id"])
    assert canonical_process_id != display_process_id
    assert execution["plan"]["process_binding"]["process_id"] == display_process_id
    lane = _manifest_relation_discovery(artifact)["lanes"]["primary"]
    assert lane["requested_mode"] == "certified-reuse"


def test_no_relation_reuse_and_opt_out_emit_identical_recurrence_plan(
    tmp_path: Path,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
) -> None:
    """Certified-reuse discovery is byte-neutral when its certified set is empty."""

    _require_native_recurrence()
    artifacts: dict[str, Path] = {}
    executions: dict[str, dict[str, Any]] = {}
    for label, mode in (("certified", "certified-reuse"), ("off", "off")):
        artifact = tmp_path / label
        Generator(
            _generation_config(
                "recurrence",
                relation_discovery_mode=mode,
            )
        ).generate(
            _NEUTRAL_CURRENT_PROCESS,
            artifact,
            model=builtin_sm_recurrence_jit_o2_model,
        )
        artifacts[label] = artifact
        executions[label] = _single_recurrence_execution(artifact)

    certified_lane = _manifest_relation_discovery(artifacts["certified"])["lanes"][
        "primary"
    ]
    assert certified_lane["requested_mode"] == "certified-reuse"
    assert certified_lane["certified_relation_count"] == 0
    assert certified_lane["applied_relation_count"] == 0
    assert certified_lane["state"] == "no_certified_numerical_relation"
    certified_schedule = next(artifacts["certified"].rglob("recurrence-runtime.pacbin"))
    off_schedule = next(artifacts["off"].rglob("recurrence-runtime.pacbin"))
    assert certified_schedule.read_bytes() == off_schedule.read_bytes()
    assert (
        executions["certified"]["plan"]["inspection_summary"]["schedule_digest"]
        != executions["off"]["plan"]["inspection_summary"]["schedule_digest"]
    )
    assert (
        executions["certified"]["plan"]["process_binding"][
            "native_schedule_semantic_digest"
        ]
        == executions["off"]["plan"]["process_binding"][
            "native_schedule_semantic_digest"
        ]
    )
    assert (
        executions["certified"]["recurrence_summary"]
        == executions["off"]["recurrence_summary"]
    )
    for key in (
        "schedule",
        "construction",
        "selector_work_certificate",
        "direct_arena",
    ):
        assert executions["certified"]["plan"]["inspection_summary"].get(
            key
        ) == executions["off"]["plan"]["inspection_summary"].get(key)


@pytest.mark.parametrize("model_source", ("builtin", "ufo"))
def test_recurrence_audit_suppresses_unsafe_all_flow_selector_domain(
    tmp_path: Path,
    model_source: str,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip unsafe all-flow hypotheses and preserve both SM frontends."""

    _require_native_recurrence()
    model = (
        builtin_sm_recurrence_jit_o2_model
        if model_source == "builtin"
        else ufo_sm_recurrence_jit_o2_model
    )
    compiled_artifact = tmp_path / f"{model_source}-compiled"
    Generator(
        _generation_config(
            "compiled",
            lc_flow_layout="all-flow-union",
        )
    ).generate(
        _RELATION_REUSE_PROCESS,
        compiled_artifact,
        model=model,
    )
    compiled = Runtime.load(compiled_artifact)
    points = _seeded_validation_points(_RELATION_REUSE_PROCESS, 101, 211, 307)

    executions: dict[str, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    lanes: dict[str, dict[str, Any]] = {}
    runtimes: dict[str, Runtime] = {}
    active_artifact: Path | None = None
    active_mode: str | None = None
    warning_publications: list[tuple[str, bool, str]] = []
    warning_counts = {"off": 0, "diagnostic": 0, "certified-reuse": 0}
    original_warning = generation_service._LOGGER.warning

    def record_warning(message: object, *args: object, **kwargs: object) -> None:
        rendered = str(message) % args if args else str(message)
        if "proofless-numerical-current-relations-applied-v1" in rendered:
            assert active_mode is not None
            warning_counts[active_mode] += 1
            warning_publications.append(
                (
                    active_mode,
                    active_artifact is not None
                    and (active_artifact / "artifact.json").is_file(),
                    rendered,
                )
            )
        original_warning(message, *args, **kwargs)

    monkeypatch.setattr(generation_service._LOGGER, "warning", record_warning)
    for mode in ("off", "diagnostic", "certified-reuse"):
        artifact = tmp_path / f"{model_source}-{mode}"
        active_artifact = artifact
        active_mode = mode
        Generator(
            _generation_config(
                "recurrence",
                lc_flow_layout="all-flow-union",
                relation_discovery_mode=mode,
            )
        ).generate(
            _RELATION_REUSE_PROCESS,
            artifact,
            model=model,
        )
        execution = _single_recurrence_execution(artifact)
        executions[mode] = execution
        assert execution["plan"]["inspection_summary"].get("relation_discovery") is None
        report = _manifest_relation_discovery(artifact)
        assert isinstance(report, dict)
        reports[mode] = report
        lane = report["lanes"]["primary"]
        assert isinstance(lane, dict)
        lanes[mode] = lane
        native = report["native_relation_application"]
        if mode == "off" or lane["certified_relation_count"] == 0:
            assert lane["applied_relation_count"] == 0
            assert native is None
        else:
            assert isinstance(native, dict)

        runtime = Runtime.load(artifact)
        runtimes[mode] = runtime
        assert runtime.evaluate(points) == pytest.approx(
            compiled.evaluate(points),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        _assert_decimal_values_match(
            runtime.evaluate(points, precision=50),
            compiled.evaluate(points, precision=50),
            50,
        )

    diagnostic = lanes["diagnostic"]
    certified = lanes["certified-reuse"]
    assert diagnostic["state"] == "no_certified_numerical_relation"
    assert diagnostic["certified_relation_count"] == 0
    assert diagnostic["applied_relation_count"] == 0
    assert diagnostic["warning"]["required"] is False
    assert certified["requested_mode"] == "certified-reuse"
    assert certified["effective_mode"] == "diagnostic"
    assert certified["effective_reuse_state"] == "disabled"
    assert certified["effective_mode_reason"] == (
        "multi-helicity-all-flow-member-scope-unproven"
    )
    assert certified["application_scope"]["effective_mode_reason"] == (
        "multi-helicity-all-flow-member-scope-unproven"
    )
    assert certified["state"] == "no_certified_numerical_relation"
    assert certified["certified_relation_count"] == 0
    assert certified["applied_relation_count"] == 0
    assert certified["warning"]["required"] is False
    assert warning_counts == {
        "off": 0,
        "diagnostic": 0,
        "certified-reuse": 0,
    }
    assert warning_publications == []
    assert certified["application_validation"]["status"] == (
        "not-required-no-applied-relations"
    )
    for lane in (diagnostic, certified):
        application_scope = lane["application_scope"]
        assert application_scope["multi_helicity_all_flow_domain_ids"] == [0]
        assert application_scope["suppressed_selector_domain_ids"] == [0]
        assert application_scope["suppressed_current_count"] == 62
        assert lane["discovery"]["tested_hypothesis_count"] == 0
        assert lane["discovery"]["numerical_candidate_count"] == 0
        assert lane["discovery"]["verification_rejected_count"] == 0
        assert lane["discovery"]["rejected_hypothesis_count"] == 0
        assert lane["discovery"]["certificates"] == []
        assert lane["discovery"]["nearest_rejected_hypothesis"] is None
        assert lane["application"]["certificates"] == []
        assert lane["application"]["relation_kind_counts"] == {
            "equal": 0,
            "opposite": 0,
            "zero": 0,
        }
        assert lane["application"]["certificate_replay"]["status"] == (
            "no_certified_numerical_relation"
        )
    candidate_index = certified["discovery"]["candidate_index"]
    assert candidate_index == {
        "algorithm": "complete-contract-anchor-tolerance-window-v1",
        "completeness": "complete-within-configured-tolerance",
        "contract_count": 31,
        "exhaustive_fallback_contract_count": 0,
        "theoretical_pair_hypothesis_count": 0,
        "screened_pair_hypothesis_count": 0,
        "zero_hypothesis_count": 0,
        "nearest_rejected_scope": ("zero-and-tolerance-window-screened-hypotheses"),
    }
    persisted = certified["persisted_numerical_evidence"]
    assert persisted["raw_evidence_retained"] is False
    assert persisted["generation_raw_evidence_bytes"] == 0
    assert persisted["generation_evidence_transport_bytes"] == 0
    assert persisted["generation_evidence_encoding"] is None
    assert certified["verification_capture"] is None
    assert persisted["verification_capture"] is None
    assert persisted["full_census"] == {
        "inspected_current_count": 68,
        "tested_hypothesis_count": 0,
        "numerical_candidate_count": 0,
        "verification_rejected_count": 0,
        "uncertified_candidate_count": 0,
        "certified_relation_count": 0,
        "rejected_hypothesis_count": 0,
        "candidate_index": candidate_index,
        "decision_sha256": certified["discovery"]["decision_sha256"],
        "rejection_decision_sha256": certified["discovery"][
            "rejection_decision_sha256"
        ],
        "certificate_set_sha256": certified["application"]["certificate_replay"][
            "certificate_set_sha256"
        ],
    }
    replay_capture = persisted["candidate_capture"]
    assert replay_capture["certificate_current_ids"] == []
    assert replay_capture["observations"] == []
    assert replay_capture["full_batch_commitment"]["current_count"] == 68

    _assert_runtime_values_match(
        runtimes["certified-reuse"],
        runtimes["diagnostic"],
        points,
    )
    _assert_runtime_values_match(runtimes["diagnostic"], runtimes["off"], points)
    # Binary64 is covered by the helper above.  Reuse must also preserve every
    # resolved member and its total at the two higher precisions used by the
    # focused numerical-current diagnosis; these are evaluations of the
    # already-generated artifacts, not additional generation work.
    for precision in (32, 200):
        reference = runtimes["off"].evaluate_resolved(
            points,
            precision=precision,
        )
        candidate = runtimes["certified-reuse"].evaluate_resolved(
            points,
            precision=precision,
        )
        assert candidate.helicity_ids == reference.helicity_ids
        assert candidate.color_ids == reference.color_ids
        assert candidate.shape == reference.shape
        _assert_decimal_values_match(candidate.values, reference.values, precision)
        _assert_decimal_values_match(candidate.total(), reference.total(), precision)
        _assert_decimal_values_match(
            runtimes["certified-reuse"].evaluate(points, precision=precision),
            runtimes["off"].evaluate(points, precision=precision),
            precision,
        )
    off_inspection = executions["off"]["plan"]["inspection_summary"]
    diagnostic_inspection = executions["diagnostic"]["plan"]["inspection_summary"]
    assert (
        executions["off"]["recurrence_summary"]
        == executions["diagnostic"]["recurrence_summary"]
    )
    assert (
        executions["certified-reuse"]["plan"]["inspection_summary"]["direct_arena"]
        == off_inspection["direct_arena"]
    )
    for key in (
        "schedule",
        "construction",
        "selector_work_certificate",
        "direct_arena",
    ):
        assert off_inspection.get(key) == diagnostic_inspection.get(key)


@pytest.mark.parametrize("process_expression", _TOPOLOGY_REPLAY_PROCESSES)
def test_builtin_lc_recurrence_artifact_loads_and_matches_compiled(
    tmp_path: Path,
    process_expression: str,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
) -> None:
    """Exercise the first public topology-replay artifact end to end."""

    _require_native_recurrence()
    recurrence_artifact = tmp_path / "recurrence"
    compiled_artifact = tmp_path / "compiled"
    progress_events: list[object] = []

    Generator(
        _generation_config("recurrence"),
        progress=CallbackProgressSink(progress_events.append),
    ).generate(
        process_expression,
        recurrence_artifact,
        model=builtin_sm_recurrence_jit_o2_model,
    )
    Generator(_generation_config("compiled")).generate(
        process_expression,
        compiled_artifact,
    )
    _assert_topology_replay_structure(recurrence_artifact, process_expression)
    native_progress = [
        event
        for event in progress_events
        if isinstance(event, ProgressUpdate) and event.task_id.endswith(":rust-builder")
    ]
    assert native_progress
    assert any(
        event.details.get("step") == "recurrence stage" for event in native_progress
    )
    assert any(
        int(event.details.get("current_count", 0)) > 0 for event in native_progress
    )

    manifest = load_manifest(recurrence_artifact)
    assert len(manifest.processes) == 1
    process = manifest.processes[0]
    process_id = str(process["id"])
    assert set(process["required_runtime_capabilities"]) == _RECURRENCE_CAPABILITIES
    assert (
        set(manifest.runtime["required_runtime_capabilities"])
        == _RECURRENCE_CAPABILITIES
    )
    inspection = inspect_artifact(recurrence_artifact).processes[0]
    assert inspection.execution_mode == "recurrence"
    assert inspection.prepared_backend == "jit"
    assert inspection.prepared_kernel_count
    assert inspection.invocation_count
    assert inspection.finalization_count
    assert inspection.closure_count
    assert inspection.native_profile_phases == (
        "selector-plan",
        "source-fill",
        "momentum-form-fill",
        "recurrence-direct-contribution",
        "recurrence-direct-finalization",
        "recurrence-direct-closure",
        "reduction",
    )

    process_root = recurrence_artifact / "processes" / process_id
    bootstrap_path = process_root / "recurrence-bootstrap.bin"
    assert bootstrap_path.is_file()
    bootstrap_relative_path = bootstrap_path.relative_to(recurrence_artifact).as_posix()
    bootstrap_payload = next(
        record for record in manifest.payloads if record.path == bootstrap_relative_path
    )
    assert bootstrap_payload.role == "evaluator-state"
    assert bootstrap_payload.process_id == process_id
    execution_path = process_root / "execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    assert execution["kind"] == _RECURRENCE_KIND
    assert execution["plan"]["kind"] == _RECURRENCE_KIND
    assert execution["recurrence_summary"]["lc_flow_layout"] == "topology-replay"
    runtime_schedule = execution["plan"]["runtime_schedule"]
    assert runtime_schedule["storage_abi"] == "pacbin-v1"
    assert runtime_schedule["member_count"] >= 1
    assert execution["plan"]["process_binding"]["path"] == "recurrence-binding.bin"
    z_masses = {
        row["outgoing_pdg"]: row["mass"]
        for row in execution["runtime_metadata"]["particle_masses"]
    }
    assert z_masses[23] == pytest.approx(91.188)
    z_sources = [
        row
        for row in execution["runtime_metadata"]["source_templates"]
        if row["source_ir"]["identity"]["pdg_label"] == 23
    ]
    assert z_sources
    assert {row["source_ir"]["mass_parameter"] for row in z_sources} == {
        "particle.23.mass"
    }

    runtime_path = recurrence_artifact / runtime_schedule["path"]
    assert runtime_path.parent.parent.parent == recurrence_artifact / "recurrence"
    assert runtime_path.is_file()
    assert runtime_path.stat().st_size == runtime_schedule["size_bytes"]
    payloads = {record.path: record for record in manifest.payloads}
    runtime_payload = payloads[runtime_path.relative_to(recurrence_artifact).as_posix()]
    assert runtime_payload.size_bytes == runtime_path.stat().st_size
    assert runtime_payload.sha256 == runtime_schedule["sha256"]
    binding_path = process_root / execution["plan"]["process_binding"]["path"]
    assert binding_path.is_file()
    assert (recurrence_artifact / "evaluators.pacbin").is_file()

    recurrence, compiled, points = _assert_topology_replay_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        process_expression,
    )
    _assert_topology_replay_exact_matches_compiled(
        recurrence,
        compiled,
        points,
    )

    if process_expression == _PROCESS:
        _assert_recurrence_per_point_selector_patterns(
            recurrence_artifact,
            points[0],
            expected_layout="topology-replay",
        )

    # Prepared packs own the model-parameter derivation kernel. Recurrence must
    # refresh derived parameters after an independent runtime update exactly as
    # compiled mode does.
    exact_before_update = recurrence.evaluate(points, precision=50)
    recurrence.set_model_parameters({"particle.23.mass": 100.0})
    compiled.set_model_parameters({"particle.23.mass": 100.0})
    assert recurrence.evaluate(points, precision=50) != exact_before_update
    assert recurrence.evaluate(points) == pytest.approx(
        compiled.evaluate(points),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    _assert_topology_replay_exact_matches_compiled(
        recurrence,
        compiled,
        points,
    )

    # The image's payload inventory excludes process-groups.json, which is
    # rewritten in place without changing the artifact identity.
    if process_expression == _PROCESS:
        group_map = read_process_groups(recurrence_artifact)
        assert group_map is not None
        group_map.analyses.append({"variant": "in-place-update"})
        write_process_groups(recurrence_artifact, group_map)
        reloaded = Runtime.load(recurrence_artifact)
        assert reloaded.artifact_id == manifest.artifact_id
        assert reloaded.evaluate(points, precision=50) == exact_before_update

    # The process-ready image is authenticated before decoding and must never
    # fall back to the looser JSON path after corruption.
    if process_expression == _PROCESS:
        image = bytearray(bootstrap_path.read_bytes())
        image[len(image) // 2] ^= 1
        bootstrap_path.write_bytes(image)
        with pytest.raises(ArtifactError):
            Runtime.load(recurrence_artifact)


@pytest.mark.parametrize(
    "process_expression",
    (
        _FOUR_LEPTON_PROCESS,
        _DILEPTON_ZH_PROCESS,
        _NEUTRAL_CURRENT_PROCESS,
        "d d~ > z g",
    ),
)
def test_builtin_fermion_vector_aliases_match_compiled_at_small_scale(
    tmp_path: Path,
    process_expression: str,
    builtin_sm_current_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Reject mirrored EW-current double counting even below 1e-15."""

    _require_native_recurrence()
    recurrence_artifact = tmp_path / "recurrence"
    compiled_artifact = tmp_path / "compiled"
    color_accuracy = "full" if process_expression == _FOUR_LEPTON_PROCESS else "lc"
    Generator(_generation_config("recurrence", color_accuracy=color_accuracy)).generate(
        process_expression,
        recurrence_artifact,
        model=builtin_sm_current_recurrence_jit_o2_model,
    )
    Generator(_generation_config("compiled", color_accuracy=color_accuracy)).generate(
        process_expression,
        compiled_artifact,
    )

    points = _validation_points(process_expression)
    recurrence = Runtime.load(recurrence_artifact)
    compiled = Runtime.load(compiled_artifact)
    recurrence_resolved = recurrence.evaluate_resolved(points, precision=200)
    compiled_resolved = compiled.evaluate_resolved(points, precision=200)

    assert recurrence_resolved.helicity_ids == compiled_resolved.helicity_ids
    assert recurrence_resolved.color_ids == compiled_resolved.color_ids
    assert recurrence_resolved.shape == compiled_resolved.shape
    _assert_scale_sensitive_values_match(
        recurrence_resolved.values,
        compiled_resolved.values,
    )
    _assert_scale_sensitive_values_match(
        recurrence.evaluate(points, precision=200),
        compiled.evaluate(points, precision=200),
    )
    if process_expression == _FOUR_LEPTON_PROCESS:
        madgraph = (_FOUR_LEPTON_MADGRAPH_SEED_101,)
        _assert_scale_sensitive_values_match(
            recurrence.evaluate(points, precision=200),
            madgraph,
            relative_tolerance=Decimal("1e-10"),
        )
        _assert_scale_sensitive_values_match(
            compiled.evaluate(points, precision=200),
            madgraph,
            relative_tolerance=Decimal("1e-10"),
        )


@pytest.mark.parametrize(
    ("color_accuracy", "lc_flow_layout", "case_id"),
    (
        ("lc", "topology-replay", "case:sm_ddbar_ee:lc"),
        ("lc", "all-flow-union", "case:sm_ddbar_ee:lc"),
        ("nlc", "topology-replay", "case:sm_ddbar_ee:nlc"),
        ("full", "topology-replay", "case:sm_ddbar_ee:full"),
    ),
)
def test_builtin_neutral_current_recurrence_matches_legacy_oracle(
    tmp_path: Path,
    color_accuracy: str,
    lc_flow_layout: str,
    case_id: str,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
) -> None:
    """Guard the incoming-spin average and mirrored fermion-pair orientation."""

    _require_native_recurrence()
    artifact = tmp_path / f"{color_accuracy}-{lc_flow_layout}"
    Generator(
        _generation_config(
            "recurrence",
            color_accuracy=color_accuracy,
            lc_flow_layout=lc_flow_layout,
        )
    ).generate(
        _NEUTRAL_CURRENT_PROCESS,
        artifact,
        model=builtin_sm_recurrence_jit_o2_model,
    )

    reference = _reference_case(case_id)
    observation = reference["observations"][0]
    reference_point = _reference_point(observation["point_id"])
    point = (
        reference_point[0],
        reference_point[1],
        reference_point[3],
        reference_point[2],
    )
    runtime = Runtime.load(artifact)
    runtime.set_model_parameters(
        {"normalization.alpha_ew": _HISTORICAL_REFERENCE_ALPHA_EW}
    )
    resolved = runtime.evaluate_resolved((point,))
    helicity_values = {
        helicity.id: helicity.values for helicity in runtime.physics.helicities
    }
    actual = {
        (helicity_values[helicity_id], color_id): complex(
            resolved.values[0][helicity_index][color_index]
        )
        for helicity_index, helicity_id in enumerate(resolved.helicity_ids)
        for color_index, color_id in enumerate(resolved.color_ids)
    }
    expected = {
        (
            (
                helicity["values"][0],
                helicity["values"][1],
                helicity["values"][3],
                helicity["values"][2],
            ),
            color["id"],
        ): float(observation["values"][helicity_index][color_index])
        for helicity_index, helicity in enumerate(reference["axes"]["helicities"])
        for color_index, color in enumerate(reference["axes"]["colors"])
    }

    assert set(actual) == set(expected)
    assert {key: value.real for key, value in actual.items()} == pytest.approx(
        expected,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert {key: value.imag for key, value in actual.items()} == pytest.approx(
        dict.fromkeys(actual, 0.0),
        abs=1.0e-15,
    )
    expected_total = float(observation["total"])
    assert runtime.evaluate((point,))[0] == pytest.approx(
        complex(expected_total),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert resolved.total()[0] == pytest.approx(
        complex(expected_total),
        rel=1.0e-12,
        abs=1.0e-15,
    )


@pytest.mark.parametrize("process_expression", _TOPOLOGY_REPLAY_PROCESSES)
def test_ufo_sm_lc_recurrence_artifact_loads_and_matches_compiled(
    tmp_path: Path,
    process_expression: str,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Exercise public topology-replay artifacts with the prepared UFO-SM."""

    recurrence_artifact = tmp_path / "recurrence-ufo-sm"
    compiled_artifact = tmp_path / "compiled-ufo-sm"
    Generator(_generation_config("recurrence")).generate(
        process_expression,
        recurrence_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    Generator(_generation_config("compiled")).generate(
        process_expression,
        compiled_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    _assert_topology_replay_structure(recurrence_artifact, process_expression)
    recurrence, compiled, points = _assert_topology_replay_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        process_expression,
    )
    _assert_topology_replay_exact_matches_compiled(
        recurrence,
        compiled,
        points,
    )
    exact_before_update = recurrence.evaluate(points, precision=50)
    recurrence.set_model_parameters({"MZ": 100.0})
    compiled.set_model_parameters({"MZ": 100.0})
    assert recurrence.evaluate(points, precision=50) != exact_before_update
    _assert_topology_replay_exact_matches_compiled(
        recurrence,
        compiled,
        points,
    )


def test_ufo_sm_recurrence_refreshes_real_and_complex_derived_couplings(
    tmp_path: Path,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """An aS update must refresh every -G alias as well as the complex i*G²."""

    output = tmp_path / "recurrence-ufo-coupling-update"
    Generator(_generation_config("recurrence", color_accuracy="full")).generate(
        _PURE_GLUON_PROCESS,
        output,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    runtime = Runtime.load(output)
    points = _validation_points(_PURE_GLUON_PROCESS)
    original = runtime.evaluate(points)
    alpha_s = next(
        parameter.default_real
        for parameter in runtime.physics.model_parameters
        if parameter.name == "aS"
    )
    execution_path = next((output / "processes").glob("*/execution.json"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    negative_g_slots = [
        parameter["parameter_index"]
        for parameter in execution["runtime_metadata"]["runtime_parameters"]
        if parameter["kind"] == "derived_parameter_component"
        and parameter["runtime_name"] is None
        and parameter["default"] == pytest.approx(-math.sqrt(4 * math.pi * alpha_s))
    ]
    # The SM contains distinct derived coupling names for the same real -G.
    assert len(negative_g_slots) >= 2
    changed_alpha_s = 0.125
    expected = tuple(value * (changed_alpha_s / alpha_s) ** 2 for value in original)
    loaded_override = Runtime.load(output, model_parameters={"aS": changed_alpha_s})
    runtime.set_model_parameter("aS", changed_alpha_s)
    for changed_runtime in (runtime, loaded_override):
        state = json.loads(
            changed_runtime._backend._runtime._exact_runtime_state_json()
        )
        for index in negative_g_slots:
            assert state["model_parameter_values"][index] == pytest.approx(
                -math.sqrt(4 * math.pi * changed_alpha_s), rel=2e-14
            )
        assert changed_runtime.evaluate(points) == pytest.approx(expected, rel=3e-13)
        assert tuple(
            float(value) for value in changed_runtime.evaluate(points, precision=50)
        ) == pytest.approx(tuple(value.real for value in expected), rel=3e-13)
    runtime.set_model_parameter("aS", alpha_s)
    assert runtime.evaluate(points) == pytest.approx(original, rel=3e-13)


@pytest.mark.parametrize("process_expression", _CONTRACTED_COLOR_PROCESSES)
@pytest.mark.parametrize("color_accuracy", ("nlc", "full"))
def test_builtin_contracted_color_recurrence_matches_compiled(
    tmp_path: Path,
    color_accuracy: str,
    process_expression: str,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
) -> None:
    """Exercise built-in NLC/full generation and Direct-Arena contraction."""

    _require_native_recurrence()
    recurrence_artifact = tmp_path / f"recurrence-{color_accuracy}"
    compiled_artifact = tmp_path / f"compiled-{color_accuracy}"
    Generator(
        _generation_config(
            "recurrence",
            color_accuracy=color_accuracy,
        )
    ).generate(
        process_expression,
        recurrence_artifact,
        model=builtin_sm_recurrence_jit_o2_model,
    )
    Generator(
        _generation_config(
            "compiled",
            color_accuracy=color_accuracy,
        )
    ).generate(
        process_expression,
        compiled_artifact,
        model=builtin_sm_recurrence_jit_o2_model,
    )
    _assert_contracted_color_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        process_expression,
        color_accuracy,
        parameter_update=(
            ("particle.23.mass", 100.0) if process_expression == _PROCESS else None
        ),
    )


@pytest.mark.parametrize("process_expression", _CONTRACTED_COLOR_PROCESSES)
@pytest.mark.parametrize("color_accuracy", ("nlc", "full"))
def test_ufo_sm_contracted_color_recurrence_matches_compiled(
    tmp_path: Path,
    color_accuracy: str,
    process_expression: str,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Exercise model-generic NLC/full recurrence with the UFO-SM pack."""

    recurrence_artifact = tmp_path / f"recurrence-ufo-{color_accuracy}"
    compiled_artifact = tmp_path / f"compiled-ufo-{color_accuracy}"
    Generator(
        _generation_config(
            "recurrence",
            color_accuracy=color_accuracy,
        )
    ).generate(
        process_expression,
        recurrence_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    Generator(
        _generation_config(
            "compiled",
            color_accuracy=color_accuracy,
        )
    ).generate(
        process_expression,
        compiled_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    _assert_contracted_color_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        process_expression,
        color_accuracy,
        parameter_update=(("MZ", 100.0) if process_expression == _PROCESS else None),
    )


def test_builtin_and_ufo_contracted_recurrence_have_matching_structure(
    tmp_path: Path,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Equivalent SM models must construct the same generic color schedule."""

    artifacts = []
    for label, model in (
        ("builtin", builtin_sm_recurrence_jit_o2_model),
        ("ufo", ufo_sm_recurrence_jit_o2_model),
    ):
        artifact = tmp_path / label
        Generator(
            _generation_config(
                "recurrence",
                color_accuracy="full",
            )
        ).generate(
            _THREE_LINE_PROCESS,
            artifact,
            model=model,
        )
        artifacts.append(artifact)

    builtin_signature = _contracted_structure_signature(artifacts[0])
    ufo_signature = _contracted_structure_signature(artifacts[1])
    assert (
        builtin_signature["peak_contribution_count_semantics"]
        == "resident-pending-contributions-v1"
    )
    assert builtin_signature == ufo_signature


def test_builtin_full_n5_pure_gluon_recurrence_fits_static_template_budget(
    tmp_path: Path,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
) -> None:
    """Keep generated N5 pure-gluon preprojection within AmpliCol's budget."""

    _require_native_recurrence()
    artifact = tmp_path / "builtin-full-n5-pure-gluon"
    Generator(
        _generation_config(
            "recurrence",
            color_accuracy="full",
        )
    ).generate(
        _N5_PURE_GLUON_PROCESS,
        artifact,
        model=builtin_sm_recurrence_jit_o2_model,
    )

    manifest = load_manifest(artifact)
    assert len(manifest.processes) == 1
    process_id = str(manifest.processes[0]["id"])
    execution = json.loads(
        (artifact / "processes" / process_id / "execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["kind"] == _RECURRENCE_KIND
    summary = execution["recurrence_summary"]
    inspection = execution["plan"]["inspection_summary"]
    schedule = inspection["schedule"]
    construction = inspection["construction"]
    assert summary["lc_flow_layout"] == "contracted-color-union"
    assert (
        (
            summary["current_count"],
            summary["contribution_count"],
        )
        == (
            schedule["current_count"],
            schedule["contribution_count"],
        )
        == (1_795, 9_990)
    )
    assert (
        construction["peak_current_count"],
        construction["peak_contribution_count"],
    ) == (2_430, 11_597)
    assert (
        construction["peak_contribution_count_semantics"]
        == "resident-pending-contributions-v1"
    )

    # Integral ceilings for 1.05x the authenticated AmpliCol static modules.
    assert construction["peak_current_count"] <= 2_557
    assert construction["peak_contribution_count"] <= 15_372
    certificate = inspection["selector_work_certificate"]
    assert (
        sum(
            representative["public_flow_count"]
            for representative in certificate["representatives"]
        )
        == schedule["replay_target_count"]
        == 720
    )
    assert Runtime.load(artifact).physics.color_accuracy == "full"


def test_builtin_and_ufo_full_neutral_current_defaults_agree(
    tmp_path: Path,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Keep the two SM frontends on one exact electroweak normalization."""

    process = "d d~ > z"
    points = _validation_points(process)
    runtimes = []
    for label, model in (
        ("builtin", builtin_sm_recurrence_jit_o2_model),
        ("ufo", ufo_sm_recurrence_jit_o2_model),
    ):
        artifact = tmp_path / label
        Generator(
            _generation_config(
                "recurrence",
                color_accuracy="full",
            )
        ).generate(process, artifact, model=model)
        runtimes.append(Runtime.load(artifact))

    builtin_total = runtimes[0].evaluate(points)
    ufo_total = runtimes[1].evaluate(points)
    assert builtin_total == pytest.approx(ufo_total, rel=1.0e-12, abs=1.0e-15)
    for runtime, total in zip(runtimes, (builtin_total, ufo_total), strict=True):
        assert runtime.evaluate_resolved(points).total() == pytest.approx(
            total,
            rel=1.0e-12,
            abs=1.0e-15,
        )


@pytest.mark.parametrize(
    ("process_expression", "required_color_id"),
    (
        (_PROCESS, None),
        (_CHARGED_CURRENT_PROCESS, None),
        (_TWO_QUARK_LINE_PROCESS, "flow:2,4,3,1"),
    ),
)
def test_builtin_lc_all_flow_union_recurrence_matches_compiled(
    tmp_path: Path,
    process_expression: str,
    required_color_id: str | None,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
) -> None:
    """Exercise all-flow union report canaries through numerical execution."""

    _require_native_recurrence()
    recurrence_artifact = tmp_path / "recurrence-union"
    compiled_artifact = tmp_path / "compiled-union"
    Generator(
        _generation_config(
            "recurrence",
            lc_flow_layout="all-flow-union",
        )
    ).generate(
        process_expression,
        recurrence_artifact,
        model=builtin_sm_recurrence_jit_o2_model,
    )
    Generator(
        _generation_config(
            "compiled",
            lc_flow_layout="all-flow-union",
        )
    ).generate(
        process_expression,
        compiled_artifact,
    )
    _assert_all_flow_union_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        process_expression,
        parameter_update=(
            ("particle.23.mass", 100.0) if process_expression == _PROCESS else None
        ),
        required_color_id=required_color_id,
    )
    if process_expression == _PROCESS:
        _assert_recurrence_per_point_selector_patterns(
            recurrence_artifact,
            _validation_points(_PROCESS)[0],
            expected_layout="all-flow-union",
        )


def test_ufo_sm_lc_all_flow_union_recurrence_matches_compiled(
    tmp_path: Path,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Exercise the model-generic union lane with the prepared UFO-SM."""

    recurrence_artifact = tmp_path / "recurrence-ufo-union"
    compiled_artifact = tmp_path / "compiled-ufo-union"
    Generator(
        _generation_config(
            "recurrence",
            lc_flow_layout="all-flow-union",
        )
    ).generate(
        _PROCESS,
        recurrence_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    Generator(
        _generation_config(
            "compiled",
            lc_flow_layout="all-flow-union",
        )
    ).generate(
        _PROCESS,
        compiled_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    _assert_all_flow_union_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        _PROCESS,
        parameter_update=("MZ", 100.0),
    )


def test_ufo_sm_full_neutral_current_recurrence_executes_prepared_mass_slot(
    tmp_path: Path,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Exercise the report's UFO full-colour source-only ``Me`` slot."""

    recurrence_artifact = tmp_path / "recurrence-ufo-full-neutral-current"
    compiled_artifact = tmp_path / "compiled-ufo-full-neutral-current"
    Generator(
        _generation_config(
            "recurrence",
            color_accuracy="full",
        )
    ).generate(
        _NEUTRAL_CURRENT_PROCESS,
        recurrence_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    Generator(
        _generation_config(
            "compiled",
            color_accuracy="full",
        )
    ).generate(
        _NEUTRAL_CURRENT_PROCESS,
        compiled_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    runtimes = (
        Runtime.load(recurrence_artifact),
        Runtime.load(compiled_artifact),
    )
    for runtime in runtimes:
        electron_mass = next(
            parameter
            for parameter in runtime.physics.model_parameters
            if parameter.name == "Me"
        )
        assert electron_mass.kind == "derived"
        assert electron_mass.mutable is False
        assert electron_mass.default_real == 0.0
        assert electron_mass.default_imaginary == 0.0

    execution_path = next((recurrence_artifact / "processes").glob("*/execution.json"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    electron_mass_projection = next(
        row
        for row in execution["runtime_metadata"]["parameter_projection"]
        if row["runtime_name"] == "Me"
    )
    prepared_parameter_id = electron_mass_projection["prepared_parameter_id"]
    assert prepared_parameter_id is not None
    assert execution["runtime_metadata"]["prepared_parameter_defaults"][
        prepared_parameter_id
    ] == [0.0, 0.0]

    _assert_contracted_color_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        _NEUTRAL_CURRENT_PROCESS,
        "full",
        parameter_update=("aEWM1", 128.0),
    )


def _assert_all_flow_union_artifacts_match(
    recurrence_artifact: Path,
    compiled_artifact: Path,
    process_expression: str,
    *,
    parameter_update: tuple[str, float] | None = None,
    required_color_id: str | None = None,
) -> None:
    point = tuple(
        tuple(float(component) for component in particle.momentum)
        for particle in generic_validation_point(process_expression)
    )
    points = (point,)
    recurrence = Runtime.load(recurrence_artifact)
    compiled = Runtime.load(compiled_artifact)
    assert recurrence.physics.color_ids == compiled.physics.color_ids
    assert recurrence.physics.helicity_ids == compiled.physics.helicity_ids

    recurrence_resolved = recurrence.evaluate_resolved(points)
    compiled_resolved = compiled.evaluate_resolved(points)
    assert recurrence_resolved.total() == pytest.approx(
        recurrence.evaluate(points),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert compiled_resolved.total() == pytest.approx(
        compiled.evaluate(points),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert recurrence_resolved.shape == compiled_resolved.shape
    assert _flatten(recurrence_resolved.values) == pytest.approx(
        _flatten(compiled_resolved.values),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert recurrence.evaluate(points) == pytest.approx(
        compiled.evaluate(points),
        rel=1.0e-12,
        abs=1.0e-15,
    )

    helicity_ids = tuple(
        helicity.id
        for helicity in recurrence.physics.helicities
        if not helicity.structural_zero
    )
    assert helicity_ids
    selected_ids = tuple(
        dict.fromkeys(
            (
                helicity_ids[0],
                helicity_ids[len(helicity_ids) // 2],
                helicity_ids[-1],
                *(("h:-1,+1,-1,+1,-1",) if process_expression == _PROCESS else ()),
            )
        )
    )
    assert set(selected_ids) <= set(helicity_ids)
    for helicity_id in selected_ids:
        recurrence_resolved = recurrence.evaluate_resolved(
            points,
            helicities=(helicity_id,),
        )
        compiled_resolved = compiled.evaluate_resolved(
            points,
            helicities=(helicity_id,),
        )
        assert recurrence_resolved.shape == compiled_resolved.shape
        assert _flatten(recurrence_resolved.values) == pytest.approx(
            _flatten(compiled_resolved.values),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        assert recurrence.evaluate(
            points,
            helicities=(helicity_id,),
        ) == pytest.approx(
            compiled.evaluate(points, helicities=(helicity_id,)),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        recurrence_exact = recurrence.evaluate_resolved(
            points,
            helicities=(helicity_id,),
            precision=32,
        )
        compiled_exact = compiled.evaluate_resolved(
            points,
            helicities=(helicity_id,),
            precision=32,
        )
        assert recurrence_exact.color_ids == compiled_exact.color_ids
        assert recurrence_exact.helicity_ids == compiled_exact.helicity_ids
        _assert_decimal_values_match(
            recurrence_exact.values,
            compiled_exact.values,
            32,
        )
        _assert_decimal_values_match(
            recurrence_exact.total(),
            recurrence.evaluate(
                points,
                helicities=(helicity_id,),
                precision=32,
            ),
            32,
        )
    if required_color_id is not None:
        assert required_color_id in recurrence.physics.color_ids
        assert recurrence.evaluate(
            points,
            color_flows=(required_color_id,),
        ) == pytest.approx(
            compiled.evaluate(
                points,
                color_flows=(required_color_id,),
            ),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        recurrence_selected = recurrence.evaluate_resolved(
            points,
            color_flows=(required_color_id,),
        )
        compiled_selected = compiled.evaluate_resolved(
            points,
            color_flows=(required_color_id,),
        )
        assert _flatten(recurrence_selected.values) == pytest.approx(
            _flatten(compiled_selected.values),
            rel=1.0e-12,
            abs=1.0e-15,
        )
    if parameter_update is not None:
        parameter_name, parameter_value = parameter_update
        before = recurrence.evaluate(points, precision=50)
        recurrence.set_model_parameters({parameter_name: parameter_value})
        compiled.set_model_parameters({parameter_name: parameter_value})
        assert recurrence.evaluate(points, precision=50) != before
        for helicity_id in selected_ids:
            assert recurrence.evaluate(
                points,
                helicities=(helicity_id,),
            ) == pytest.approx(
                compiled.evaluate(points, helicities=(helicity_id,)),
                rel=1.0e-12,
                abs=1.0e-15,
            )


def test_recurrence_exact_accepts_all_flow_union_layout() -> None:
    """Accept both complete LC recurrence strategies."""

    _validate_execution(
        {
            "schema_version": 3,
            "kind": _RECURRENCE_KIND,
            "key": "d_dbar_to_z_g_g",
            "recurrence_plan_abi": "pyamplicol-recurrence-plan-v2",
            "runtime_layout_abi": "pyamplicol-recurrence-runtime-layout-v2",
            "recurrence_summary": {"lc_flow_layout": "all-flow-union"},
            "required_runtime_capabilities": sorted(_RECURRENCE_CAPABILITIES),
        },
        "d_dbar_to_z_g_g",
    )
