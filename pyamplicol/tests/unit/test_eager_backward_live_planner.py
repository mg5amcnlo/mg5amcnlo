# SPDX-License-Identifier: 0BSD
"""Correctness contracts for eager backward/live-state DAG construction."""

from __future__ import annotations

import inspect
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import pytest

from pyamplicol.config import EvaluatorConfig, RunConfig
from pyamplicol.generation import service as service_module
from pyamplicol.generation.dag_algorithms import prune_dag_to_amplitude_roots
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_types import (
    AmplitudeRoot,
    GenericDAG,
    InteractionNode,
)
from pyamplicol.generation.runtime_schema import build_runtime_schema
from pyamplicol.generation.service import GenerationBackend
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.external import CompiledUFOModel
from pyamplicol.models.loading import compile_model_source
from pyamplicol.processes.ir import CanonicalProcessIR
from pyamplicol.processes.model import build_model_process_ir

_BACKWARD_PLANNER_PARAMETER = "backward_live_planning"
_EXTERNAL_SM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)
_PARITY_CASES = (
    ("d d~ > z g g g", "nlc", {"QCD": 3, "QED": 1}),
    ("d d~ > z g g g", "full", {"QCD": 3, "QED": 1}),
    ("d d~ > u u~ s s~ g", "nlc", {"QCD": 5, "QED": 0}),
    ("d d~ > u u~ s s~ g", "full", {"QCD": 5, "QED": 0}),
)


def _compile_reference_and_eager(
    process: str,
    color_accuracy: str,
    coupling_limits: Mapping[str, int],
    *,
    selected_source_helicities: Mapping[int, int] | None = None,
) -> tuple[Model, GenericDAG, GenericDAG, GenericDAG]:
    model = BuiltinSMModel()
    process_ir = build_process_ir(process, color_accuracy=color_accuracy)
    options = {
        "model": model,
        "max_coupling_orders": coupling_limits,
        "selected_source_helicities": selected_source_helicities,
        "online_evaluation_reuse": True,
    }
    forward = compile_generic_dag(process_ir, **options)
    reference = prune_dag_to_amplitude_roots(forward)
    eager = compile_generic_dag(
        process_ir,
        **options,
        backward_live_planning=True,
    )
    return model, forward, reference, eager


def _interaction_signature(
    dag: GenericDAG,
    interaction: InteractionNode,
) -> tuple[object, ...]:
    return (
        interaction.vertex_kind,
        interaction.vertex_particles,
        dag.currents[interaction.left_id].index,
        dag.currents[interaction.right_id].index,
        dag.currents[interaction.result_id].index,
        interaction.coupling,
        interaction.color_weight,
        interaction.lowering_backend,
        interaction.full_tensor_network_ready,
        interaction.evaluation_factor,
    )


def _root_signature(
    dag: GenericDAG,
    root: AmplitudeRoot,
) -> tuple[object, ...]:
    return (
        root.kind,
        dag.currents[root.left_id].index,
        dag.currents[root.right_id].index,
        root.color_weight,
        root.contraction_ir,
        root.color_sector_id,
        root.vertex_kind,
        root.vertex_particles,
        root.coupling,
        root.helicity_weight,
    )


def _evaluation_group_partition(
    dag: GenericDAG,
) -> set[frozenset[tuple[object, ...]]]:
    by_group: dict[int | None, set[tuple[object, ...]]] = defaultdict(set)
    for interaction in dag.interactions:
        by_group[interaction.evaluation_group_id].add(
            _interaction_signature(dag, interaction)
        )
    return {frozenset(group) for group in by_group.values()}


def _assert_semantic_topology_parity(
    reference: GenericDAG,
    eager: GenericDAG,
) -> None:
    """Compare topology independently of traversal-assigned integer IDs."""

    reference_currents = {
        (
            current.index,
            current.dimension,
            current.is_source,
            current.source_leg_label,
            current.source_helicity,
        )
        for current in reference.currents
    }
    eager_currents = {
        (
            current.index,
            current.dimension,
            current.is_source,
            current.source_leg_label,
            current.source_helicity,
        )
        for current in eager.currents
    }
    reference_interactions = {
        _interaction_signature(reference, interaction)
        for interaction in reference.interactions
    }
    eager_interactions = {
        _interaction_signature(eager, interaction)
        for interaction in eager.interactions
    }
    reference_roots = {
        _root_signature(reference, root) for root in reference.amplitude_roots
    }
    eager_roots = {_root_signature(eager, root) for root in eager.amplitude_roots}

    assert eager.process == reference.process
    assert eager.color_plan == reference.color_plan
    assert eager_currents == reference_currents
    assert eager_interactions == reference_interactions
    assert eager_roots == reference_roots
    assert _evaluation_group_partition(eager) == _evaluation_group_partition(
        reference
    )
    assert (
        eager.truncated,
        eager.helicity_coverage,
        eager.color_coverage,
        eager.selected_source_helicities,
    ) == (
        reference.truncated,
        reference.helicity_coverage,
        reference.color_coverage,
        reference.selected_source_helicities,
    )


def _assert_runtime_physics_equivalent(
    reference: Mapping[str, object],
    eager: Mapping[str, object],
) -> None:
    reference_without_reduction = dict(reference)
    eager_without_reduction = dict(eager)
    reference_reduction = reference_without_reduction.pop("reduction")
    eager_reduction = eager_without_reduction.pop("reduction")
    assert eager_without_reduction == reference_without_reduction
    assert isinstance(reference_reduction, Mapping)
    assert isinstance(eager_reduction, Mapping)
    assert eager_reduction.get("kind") == reference_reduction.get("kind")

    def group_records(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
        groups = payload.get("groups")
        assert isinstance(groups, list)
        records = [group for group in groups if isinstance(group, Mapping)]
        assert len(records) == len(groups)
        return records

    def semantic_groups(
        groups: list[Mapping[str, object]],
    ) -> set[tuple[object, ...]]:
        return {
            (
                group["representative_helicity_id"],
                group["representative_color_id"],
                tuple(group["physical_helicity_ids"]),
                tuple(group["physical_color_ids"]),
            )
            for group in groups
        }

    eager_groups = group_records(eager_reduction)
    reference_groups = group_records(reference_reduction)
    assert len(eager_groups) == len(reference_groups)
    assert semantic_groups(eager_groups) == semantic_groups(reference_groups)


def _forbidden_backend_call(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("backward DAG planning invoked a backend compiler")


def test_backward_live_planning_is_keyword_only_and_opt_in() -> None:
    parameter = inspect.signature(compile_generic_dag).parameters.get(
        _BACKWARD_PLANNER_PARAMETER
    )

    assert parameter is not None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


@pytest.mark.parametrize("backward_live_planning", (False, True))
def test_dag_progress_reports_stage_boundaries_for_both_planners(
    backward_live_planning: bool,
) -> None:
    events: list[Mapping[str, str | int]] = []
    compile_generic_dag(
        build_process_ir("d d~ > z g", color_accuracy="nlc"),
        model=BuiltinSMModel(),
        online_evaluation_reuse=True,
        backward_live_planning=backward_live_planning,
        progress_callback=events.append,
    )

    steps = [event["step"] for event in events]
    assert "colour-plan" in steps
    assert "reachability" in steps
    assert "source-currents" in steps
    assert "recursion" in steps
    assert "amplitude-closure" in steps
    assert steps[-1] == "symmetry-reuse"
    recursion = [event for event in events if event["step"] == "recursion"]
    assert recursion
    assert all("mask_index" in event and "mask_total" in event for event in recursion)
    assert all("current_count" in event for event in recursion)
    assert all("interaction_count" in event for event in recursion)


@pytest.mark.parametrize(
    ("process", "color_accuracy", "coupling_limits"),
    _PARITY_CASES,
)
def test_backward_live_planner_matches_post_pruned_topology_and_physics(
    process: str,
    color_accuracy: str,
    coupling_limits: Mapping[str, int],
) -> None:
    model, forward, reference, eager = _compile_reference_and_eager(
        process,
        color_accuracy,
        coupling_limits,
    )

    _assert_semantic_topology_parity(reference, eager)
    assert prune_dag_to_amplitude_roots(eager) == eager
    assert build_runtime_schema(eager, model, process_id="parity")["physics"] == (
        build_runtime_schema(reference, model, process_id="parity")["physics"]
    )
    if "u u~ s s~" in process:
        assert len(forward.currents) > len(eager.currents)
        assert len(forward.interactions) > len(eager.interactions)


@pytest.mark.parametrize("color_accuracy", ("nlc", "full"))
def test_backward_live_planner_preserves_selected_helicity_topology(
    color_accuracy: str,
) -> None:
    model, forward, reference, eager = _compile_reference_and_eager(
        "d d~ > u u~ s s~ g",
        color_accuracy,
        {"QCD": 5, "QED": 0},
        selected_source_helicities={1: -1},
    )

    _assert_semantic_topology_parity(reference, eager)
    assert eager.helicity_coverage == "selected"
    reference_physics = build_runtime_schema(
        reference,
        model,
        process_id="selected",
    )["physics"]
    _assert_runtime_physics_equivalent(
        reference_physics,
        build_runtime_schema(forward, model, process_id="selected")["physics"],
    )
    _assert_runtime_physics_equivalent(
        reference_physics,
        build_runtime_schema(eager, model, process_id="selected")["physics"],
    )


@pytest.mark.parametrize("color_accuracy", ("nlc", "full"))
def test_backward_live_planner_matches_pruned_ufo_sm(color_accuracy: str) -> None:
    compiled = compile_model_source(
        _EXTERNAL_SM_ROOT / "sm.json",
        restriction=str((_EXTERNAL_SM_ROOT / "restrict_default.json").resolve()),
        use_cache=True,
    )
    model = CompiledUFOModel(compiled)
    process = build_model_process_ir(
        "d d~ > u u~ s s~ g",
        compiled.ir,
        color_accuracy=color_accuracy,
    )
    options = {
        "model": model,
        "max_coupling_orders": {"QCD": 5, "QED": 0},
        "online_evaluation_reuse": True,
    }
    reference = prune_dag_to_amplitude_roots(compile_generic_dag(process, **options))
    eager = compile_generic_dag(
        process,
        **options,
        backward_live_planning=True,
    )

    _assert_semantic_topology_parity(reference, eager)
    _assert_runtime_physics_equivalent(
        build_runtime_schema(reference, model, process_id="ufo-sm")["physics"],
        build_runtime_schema(eager, model, process_id="ufo-sm")["physics"],
    )


def test_backward_live_planner_is_deterministic() -> None:
    model = BuiltinSMModel()
    process = build_process_ir(
        "d d~ > u u~ s s~ g",
        color_accuracy="nlc",
    )
    options = {
        "model": model,
        "max_coupling_orders": {"QCD": 5, "QED": 0},
        "online_evaluation_reuse": True,
        "backward_live_planning": True,
    }

    assert compile_generic_dag(process, **options) == compile_generic_dag(
        process,
        **options,
    )


def test_backward_live_planner_never_compiles_backend_evaluators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.evaluators import symbolica_compile
    from pyamplicol.generation import stage_artifacts

    monkeypatch.setattr(
        symbolica_compile,
        "_compile_symbolica_outputs",
        _forbidden_backend_call,
    )
    monkeypatch.setattr(
        stage_artifacts,
        "_compile_stage_evaluator_artifact",
        _forbidden_backend_call,
    )
    monkeypatch.setattr(
        stage_artifacts,
        "build_and_write_generic_stage_evaluator_artifacts",
        _forbidden_backend_call,
    )
    monkeypatch.setattr(subprocess, "run", _forbidden_backend_call)
    monkeypatch.setattr(subprocess, "Popen", _forbidden_backend_call)
    try:
        from symbolica import Expression
    except ImportError:  # pragma: no cover - minimal source-only environment
        Expression = None
    if Expression is not None:
        for name in dir(Expression):
            if name == "evaluator" or name.startswith("evaluator_"):
                monkeypatch.setattr(Expression, name, _forbidden_backend_call)

    _model, _forward, reference, eager = _compile_reference_and_eager(
        "d d~ > u u~ s s~ g",
        "nlc",
        {"QCD": 5, "QED": 0},
    )

    _assert_semantic_topology_parity(reference, eager)


@pytest.mark.parametrize(
    ("execution_mode", "expected_backward_planning"),
    (("compiled", False), ("eager", True)),
)
def test_generation_service_enables_backward_planner_only_for_eager_mode(
    monkeypatch: pytest.MonkeyPatch,
    execution_mode: str,
    expected_backward_planning: bool,
) -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > z g g g")
    reference = compile_generic_dag(process, model=model)
    observed: list[bool] = []

    def capture_compile(
        process_ir: CanonicalProcessIR,
        *,
        model: Model,
        **options: object,
    ) -> GenericDAG:
        assert process_ir is process
        observed.append(bool(options.get(_BACKWARD_PLANNER_PARAMETER, False)))
        return reference

    monkeypatch.setattr(service_module, "compile_generic_dag", capture_compile)
    backend = GenerationBackend(
        RunConfig(
            action="generate",
            evaluator=EvaluatorConfig(execution_mode=execution_mode),
        ),
        None,
    )

    compiled, _coverage = backend._compile_concrete_process(process, model)

    assert compiled.currents == reference.currents
    assert compiled.interactions == reference.interactions
    assert compiled.amplitude_roots == reference.amplitude_roots
    assert compiled.lc_topology_replay is not None
    assert observed == [expected_backward_planning]
