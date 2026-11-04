# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import pyamplicol.generation.service as service_module
from pyamplicol.api import ProcessRequest
from pyamplicol.api.errors import GenerationError
from pyamplicol.color import plan_replay as replay_module
from pyamplicol.color.plan import (
    GenericColorPlan,
    build_color_plan,
    build_lc_topology_replay_plan,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation import artifact_writer as artifact_writer_module
from pyamplicol.generation.artifact_writer import CompiledProcessArtifact
from pyamplicol.generation.dag_algorithms import (
    prune_global_helicity_flip_equivalent_roots,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_types import GenericDAG, InteractionNode
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.generation.recurrence_schedule_sharing import (
    RecurrenceScheduleLoweringCache,
)
from pyamplicol.generation.runtime_schema import build_runtime_expression_schema
from pyamplicol.generation.service import (
    GenerationBackend,
    _ProcessSelection,
)
from pyamplicol.generation.stage_planning import (
    _compiled_representative_dependencies,
    _lc_materialized_sector_memberships,
)
from pyamplicol.generation.validation import ValidationPointRecord
from pyamplicol.models import BuiltinSMModel, CompiledUFOModel, compile_model_source
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.processes.model import build_model_process_ir

_UFO_SM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)


@dataclass(frozen=True)
class _ProjectedLogicalStub:
    external_legs: tuple[object, ...]
    physical_sectors: tuple[object, ...]
    public_flows: tuple[object, ...]
    replay_partitions: tuple[object, ...]
    selected_public_flow_ids: tuple[int, ...] | None


def test_complete_lc_generation_materializes_replay_representative() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > z g g")

    dag, coverage = GenerationBackend(None, None)._compile_concrete_process(
        process,
        model,
    )

    replay = dag.lc_topology_replay
    assert replay is not None
    assert replay.physical_sector_ids == (0, 1)
    assert replay.materialized_sector_ids == (0,)
    assert replay.residual_sector_ids == ()
    assert {int(root.color_sector_id or 0) for root in dag.amplitude_roots} == {0}
    assert tuple(sector.id for sector in dag.color_plan.sectors) == (0, 1)
    assert coverage["color_sector_count"] == 2
    assert coverage["materialized_color_sector_count"] == 1

    schema = build_runtime_expression_schema(dag, model).to_mapping()
    colors = schema["physics"]["color_components"]
    assert [(record["word"], record["computed"]) for record in colors] == [
        ([2, 4, 5, 1], True),
        ([2, 5, 4, 1], False),
    ]
    manifest = schema["lc_topology_replay"]
    assert manifest["physical_sector_count"] == 2
    assert manifest["materialized_sector_ids"] == [0]
    assert manifest["residual_sector_ids"] == []
    assert manifest["groups"][0]["active_sector_ids"] == [0, 1]
    assert all(
        item["weight"] > 0.0
        and item["sign"] in {-1, 1}
        and item["factor"] == [item["weight"] * item["sign"], 0.0]
        for item in manifest["groups"][0]["sector_permutations"]
    )


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_all_flow_union_materializes_complete_color_plan(
    execution_mode: str,
) -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > z g g")
    backend = GenerationBackend(
        RunConfig(
            action="generate",
            color=ColorConfig(lc_flow_layout="all-flow-union"),
            evaluator=EvaluatorConfig(execution_mode=execution_mode),
        ),
        None,
    )

    dag, coverage = backend._compile_concrete_process(process, model)

    assert dag.lc_topology_replay is None
    assert dag.color_coverage == "complete"
    assert dag.helicity_coverage == "complete"
    assert coverage["lc_flow_layout"] == "all-flow-union"
    assert {
        int(root.color_sector_id)
        for root in dag.amplitude_roots
        if root.color_sector_id is not None
    } == {int(sector.id) for sector in dag.color_plan.sectors}


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_contracted_color_coverage_has_no_lc_layout_provenance(
    accuracy: str,
) -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > z g g", color_accuracy=accuracy)
    backend = GenerationBackend(
        RunConfig(
            action="generate",
            color=ColorConfig(accuracy=accuracy),
            evaluator=EvaluatorConfig(execution_mode="compiled"),
        ),
        None,
    )

    _dag, coverage = backend._compile_concrete_process(process, model)

    assert "lc_flow_layout" not in coverage


def test_compiled_all_flow_union_is_the_only_materialized_execution_lane() -> None:
    model = BuiltinSMModel()
    expression = "d d~ > z g g"
    process = build_process_ir(expression)
    backend = GenerationBackend(
        RunConfig(
            action="generate",
            color=ColorConfig(lc_flow_layout="all-flow-union"),
            evaluator=EvaluatorConfig(execution_mode="compiled"),
        ),
        None,
    )
    dag, coverage = backend._compile_concrete_process(process, model)
    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(
            expanded=service_module._ExpandedProcess(
                request=ProcessRequest.parse(expression, name="all_flow_union"),
                process_ir=process,
            ),
            dag=dag,
            coverage=coverage,
        ),
        model,
        index=0,
        phase=PhaseHandle("test", None, 1),
    )
    evaluator = backend._construct_evaluator(
        prepared,
        model,
        PhaseHandle("test", None, 1),
    )

    assert prepared.helicity_sum_dag is None
    assert prepared.helicity_selector_union_dag is None
    assert prepared.dag.helicity_materialization is not None
    assert evaluator.helicity_sum_runtime_schema is None
    assert evaluator.helicity_selector_lanes
    assert all(
        lane.schedule_mode == "parent-closure"
        for lane in evaluator.helicity_selector_lanes
    )
    assert evaluator.color_selector_lanes == ()


@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
@pytest.mark.parametrize(
    "process",
    (
        ProcessConfig(max_color_sectors=1),
        ProcessConfig(selected_color_sector_ids=(0,)),
        ProcessConfig(selected_source_helicities={"1": -1}),
    ),
)
def test_all_flow_union_rejects_generation_selected_coverage(
    process: ProcessConfig,
    execution_mode: str,
) -> None:
    backend = GenerationBackend(
        RunConfig(
            action="generate",
            process=process,
            color=ColorConfig(lc_flow_layout="all-flow-union"),
            evaluator=EvaluatorConfig(execution_mode=execution_mode),
        ),
        None,
    )

    with pytest.raises(
        GenerationError,
        match="requires complete runtime flow and helicity coverage",
    ):
        backend._compile_concrete_process(
            build_process_ir("d d~ > z g g"),
            BuiltinSMModel(),
        )


@pytest.mark.parametrize(
    ("process_config", "expected_flows", "expected_helicities"),
    (
        (ProcessConfig(), None, None),
        (ProcessConfig(selected_color_sector_ids=(1,)), (0,), None),
        (
            ProcessConfig(
                selected_source_helicities={
                    "1": -1,
                    "2": 1,
                    "3": -1,
                    "4": 1,
                    "5": -1,
                }
            ),
            None,
            ((1, -1), (2, 1), (3, -1), (4, 1), (5, -1)),
        ),
        (
            ProcessConfig(
                selected_color_sector_ids=(1,),
                selected_source_helicities={
                    "1": -1,
                    "2": 1,
                    "3": -1,
                    "4": 1,
                    "5": -1,
                },
            ),
            (0,),
            ((1, -1), (2, 1), (3, -1), (4, 1), (5, -1)),
        ),
    ),
)
def test_recurrence_all_flow_union_forwards_generation_selected_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    process_config: ProcessConfig,
    expected_flows: tuple[int, ...] | None,
    expected_helicities: tuple[tuple[int, int], ...] | None,
) -> None:
    class _StopBeforeNativeLowering(Exception):
        pass

    expression = "d d~ > z g g"
    process = build_process_ir(expression)
    slices: list[object] = []

    def fake_project(
        _process: object,
        color_plan: GenericColorPlan,
        *_args: object,
        generation_slice: object,
        **_kwargs: object,
    ):
        slices.append(generation_slice)
        sectors = tuple(color_plan.sectors)
        return _ProjectedLogicalStub(
            external_legs=(1, 2, 3, 4, 5),
            physical_sectors=tuple(range(len(sectors))),
            public_flows=tuple(
                SimpleNamespace(flow_id=index, construction_sector_id=index)
                for index, _sector in enumerate(sectors)
            ),
            replay_partitions=(),
            selected_public_flow_ids=generation_slice.selected_public_flow_ids,
        )

    projected_logicals: list[_ProjectedLogicalStub] = []

    def stop_before_native_lowering(logical: _ProjectedLogicalStub) -> None:
        projected_logicals.append(logical)
        raise _StopBeforeNativeLowering

    monkeypatch.setattr(
        service_module,
        "project_recurrence_process_v1",
        fake_project,
    )
    monkeypatch.setattr(
        service_module,
        "build_recurrence_builder_input_v1",
        stop_before_native_lowering,
    )

    backend = GenerationBackend(
        RunConfig(
            action="generate",
            process=process_config,
            color=ColorConfig(lc_flow_layout="all-flow-union"),
            evaluator=EvaluatorConfig(execution_mode="recurrence"),
        ),
        None,
    )

    with pytest.raises(_StopBeforeNativeLowering):
        backend._construct_recurrence_artifact(
            service_module._ExpandedProcess(
                request=ProcessRequest.parse(expression, name="recurrence_union"),
                process_ir=process,
            ),
            BuiltinSMModel(),
            SimpleNamespace(catalog=object()),
            tmp_path,
            index=0,
            phase=PhaseHandle("test", None, 1),
            lowering_cache=RecurrenceScheduleLoweringCache(),
        )

    assert len(slices) == 1
    generation_slice = slices[0]
    assert generation_slice.selected_public_flow_ids is None
    assert generation_slice.selected_source_helicities == expected_helicities
    assert projected_logicals[-1].selected_public_flow_ids == expected_flows


def test_recurrence_selected_n6_flow_projects_one_dense_sector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _StopBeforeNativeLowering(Exception):
        pass

    expression = "d d~ > u u~ s s~ g g"
    process = build_process_ir(expression)
    expected_sector = build_color_plan(process).sectors[5]
    projected_plans: list[GenericColorPlan] = []
    slices: list[object] = []

    def fake_project(
        _process: object,
        color_plan: GenericColorPlan,
        *_args: object,
        generation_slice: object,
        **_kwargs: object,
    ):
        projected_plans.append(color_plan)
        slices.append(generation_slice)
        sectors = tuple(color_plan.sectors)
        return _ProjectedLogicalStub(
            external_legs=tuple(range(len(process.legs))),
            physical_sectors=tuple(range(len(sectors))),
            public_flows=tuple(
                SimpleNamespace(flow_id=index, construction_sector_id=index)
                for index, _sector in enumerate(sectors)
            ),
            replay_partitions=(),
            selected_public_flow_ids=generation_slice.selected_public_flow_ids,
        )

    projected_logicals: list[_ProjectedLogicalStub] = []

    def stop_before_native_lowering(logical: _ProjectedLogicalStub) -> None:
        projected_logicals.append(logical)
        raise _StopBeforeNativeLowering

    monkeypatch.setattr(
        service_module,
        "project_recurrence_process_v1",
        fake_project,
    )
    monkeypatch.setattr(
        service_module,
        "build_recurrence_builder_input_v1",
        stop_before_native_lowering,
    )
    backend = GenerationBackend(
        RunConfig(
            action="generate",
            process=ProcessConfig(selected_color_sector_ids=(5,)),
            evaluator=EvaluatorConfig(execution_mode="recurrence"),
        ),
        None,
    )

    with pytest.raises(_StopBeforeNativeLowering):
        backend._construct_recurrence_artifact(
            service_module._ExpandedProcess(
                request=ProcessRequest.parse(expression, name="recurrence_n6"),
                process_ir=process,
            ),
            BuiltinSMModel(),
            SimpleNamespace(catalog=object()),
            tmp_path,
            index=0,
            phase=PhaseHandle("test", None, 1),
            lowering_cache=RecurrenceScheduleLoweringCache(),
        )

    assert len(projected_plans) == 1
    assert tuple(sector.id for sector in projected_plans[0].sectors) == (0,)
    assert projected_plans[0].sectors[0].word_labels == expected_sector.word_labels
    assert slices[0].selected_public_flow_ids is None
    assert projected_logicals[-1].selected_public_flow_ids == (0,)


def test_recurrence_all_flow_union_rejects_truncated_color_coverage() -> None:
    backend = GenerationBackend(
        RunConfig(
            action="generate",
            process=ProcessConfig(max_color_sectors=1),
            color=ColorConfig(lc_flow_layout="all-flow-union"),
            evaluator=EvaluatorConfig(execution_mode="recurrence"),
        ),
        None,
    )

    with pytest.raises(
        GenerationError,
        match="requires complete runtime flow and helicity coverage",
    ):
        backend._prepare_process_construction(
            build_process_ir("d d~ > z g g"),
            BuiltinSMModel(),
        )


def test_lc_replay_derives_exact_shared_and_private_current_closures() -> None:
    model = BuiltinSMModel()
    dag, _coverage = GenerationBackend(None, None)._compile_concrete_process(
        build_process_ir("g g > g g"),
        model,
    )

    replay = dag.lc_topology_replay
    assert replay is not None
    assert replay.materialized_sector_ids == (0, 2)
    current_domains, root_domains = _lc_materialized_sector_memberships(dag)

    assert set(root_domains) == {root.id for root in dag.amplitude_roots}
    assert set(current_domains) == set(range(len(dag.currents)))
    assert {domain for domains in root_domains.values() for domain in domains} == {
        0,
        2,
    }
    assert any(domains == (0, 2) for domains in current_domains.values())
    assert any(domains == (0,) for domains in current_domains.values())
    assert any(domains == (2,) for domains in current_domains.values())
    compiled_dependencies = _compiled_representative_dependencies(dag)
    for current_id, parent_ids in compiled_dependencies.items():
        for parent_id in parent_ids:
            assert set(current_domains[current_id]).issubset(
                current_domains[parent_id]
            )


def test_lc_materialized_residuals_get_independent_selector_closures() -> None:
    model = BuiltinSMModel()
    dag, _coverage = GenerationBackend(None, None)._compile_concrete_process(
        build_process_ir("d d~ > u u~ s s~ g"),
        model,
    )

    assert dag.lc_topology_replay is None
    expected_sectors = {int(sector.id) for sector in dag.color_plan.sectors}
    assert len(expected_sectors) > 1

    current_domains, root_domains = _lc_materialized_sector_memberships(dag)

    assert {domain for domains in root_domains.values() for domain in domains} == (
        expected_sectors
    )
    assert set(root_domains) == {root.id for root in dag.amplitude_roots}
    assert set(dag.sources).issubset(current_domains)
    assert all(
        root.left_id in current_domains and root.right_id in current_domains
        for root in dag.amplitude_roots
    )
    assert any(len(domains) == 1 for domains in current_domains.values())
    assert any(len(domains) > 1 for domains in current_domains.values())


def test_eager_complete_lc_generation_materializes_replay_representative() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > z g g")
    backend = GenerationBackend(
        RunConfig(
            action="generate",
            evaluator=EvaluatorConfig(execution_mode="eager"),
        ),
        None,
    )

    dag, coverage = backend._compile_concrete_process(process, model)

    replay = dag.lc_topology_replay
    assert replay is not None
    assert replay.physical_sector_ids == (0, 1)
    assert replay.materialized_sector_ids == (0,)
    assert replay.residual_sector_ids == ()
    assert {int(root.color_sector_id or 0) for root in dag.amplitude_roots} == {0}
    assert tuple(sector.id for sector in dag.color_plan.sectors) == (0, 1)
    assert dag.color_coverage == "complete"
    assert dag.helicity_coverage == "complete"
    assert coverage["materialized_color_sector_count"] == 1


@pytest.mark.parametrize(
    "selection",
    (
        _ProcessSelection(selected_color_sector_ids=frozenset({1})),
        _ProcessSelection(
            selected_source_helicities={1: -1, 2: 1, 3: -1, 4: 1, 5: -1},
        ),
    ),
    ids=("selected-color", "selected-helicity"),
)
def test_eager_selected_generation_axis_does_not_enable_replay(
    selection: _ProcessSelection,
) -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > z g g")
    backend = GenerationBackend(
        RunConfig(
            action="generate",
            evaluator=EvaluatorConfig(execution_mode="eager"),
        ),
        None,
        process_selection=selection,
    )

    dag, coverage = backend._compile_concrete_process(process, model)
    schema = build_runtime_expression_schema(dag, model).to_mapping()

    assert dag.lc_topology_replay is None
    assert "lc_topology_replay" not in dag.to_json_dict()
    assert "lc_topology_replay" not in schema
    assert "materialized_color_sector_count" not in coverage


def test_specialized_lc_generation_retains_existing_schema() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > z g g")
    backend = GenerationBackend(
        None,
        None,
        process_selection=_ProcessSelection(
            selected_color_sector_ids=frozenset({1})
        ),
    )

    dag, coverage = backend._compile_concrete_process(process, model)
    schema = build_runtime_expression_schema(dag, model).to_mapping()

    assert tuple(sector.id for sector in dag.color_plan.sectors) == (1,)
    assert dag.color_coverage == "selected"
    assert dag.lc_topology_replay is None
    assert _lc_materialized_sector_memberships(dag) == ({}, {})
    assert "lc_topology_replay" not in dag.to_json_dict()
    assert "lc_topology_replay" not in schema
    assert "lc_topology_replay" not in schema["physics"]["extensions"]
    assert "materialized_color_sector_count" not in coverage
    assert "replayed_color_sector_count" not in coverage
    assert "residual_color_sector_count" not in coverage


@pytest.mark.parametrize("model_source", ("builtin", "ufo-sm"))
def test_reversed_selected_flow_retains_exchange_signed_three_gluon_currents(
    model_source: str,
) -> None:
    if model_source == "builtin":
        model = BuiltinSMModel()
        process = build_process_ir("d d~ > z g g")
    else:
        compiled = compile_model_source(
            _UFO_SM_ROOT / "sm.json",
            restriction=str((_UFO_SM_ROOT / "restrict_default.json").resolve()),
            use_cache=True,
        )
        model = CompiledUFOModel(compiled)
        process = build_model_process_ir("d d~ > z g g", compiled.ir)

    color_plan = build_color_plan(process)
    direct = compile_generic_dag(
        process,
        model=model,
        color_plan=color_plan,
        selected_color_sector_ids=(0,),
    )
    reversed_flow = compile_generic_dag(
        process,
        model=model,
        color_plan=color_plan,
        selected_color_sector_ids=(1,),
    )

    def three_gluon_terms(dag: GenericDAG) -> tuple[InteractionNode, ...]:
        return tuple(
            interaction
            for interaction in dag.interactions
            if tuple(abs(pdg) for pdg in interaction.vertex_particles)
            == (21, 21, 21)
            and model.vertex_evaluation_equivalence(
                interaction.vertex_kind
            ).input_exchange_factor
            == (-1.0, 0.0)
        )

    direct_terms = three_gluon_terms(direct)
    reversed_terms = three_gluon_terms(reversed_flow)
    assert len(direct_terms) == len(reversed_terms) == 4
    direct_weights = {term.color_weight for term in direct_terms}
    assert {term.color_weight for term in reversed_terms} == {
        (-real, -imaginary) for real, imaginary in direct_weights
    }
    assert {
        reversed_flow.currents[term.result_id].index.ordered_external_labels
        for term in reversed_terms
    } == {(5, 4)}
    assert len(direct.currents) == len(reversed_flow.currents)
    assert len(direct.interactions) == len(reversed_flow.interactions)
    assert len(direct.amplitude_roots) == len(reversed_flow.amplitude_roots)


def test_fixed_helicity_complete_color_generation_retains_existing_schema() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > z g g")
    backend = GenerationBackend(
        None,
        None,
        process_selection=_ProcessSelection(
            selected_source_helicities={1: -1, 2: 1, 3: -1, 4: 1, 5: -1},
        ),
    )

    dag, coverage = backend._compile_concrete_process(process, model)
    schema = build_runtime_expression_schema(dag, model).to_mapping()

    assert dag.color_coverage == "complete"
    assert dag.helicity_coverage == "selected"
    assert dag.lc_topology_replay is None
    assert "lc_topology_replay" not in dag.to_json_dict()
    assert "lc_topology_replay" not in schema
    assert "materialized_color_sector_count" not in coverage


def test_ufo_sm_replay_proof_uses_compiled_canonical_contracts() -> None:
    compiled = compile_model_source(
        _UFO_SM_ROOT / "sm.json",
        restriction=str((_UFO_SM_ROOT / "restrict_default.json").resolve()),
        use_cache=True,
    )
    model = CompiledUFOModel(compiled)
    process = build_model_process_ir("d d~ > z g g", compiled.ir)

    replay = build_lc_topology_replay_plan(build_color_plan(process), model)

    assert replay is not None
    assert replay.materialized_sector_ids == (0,)
    assert replay.residual_sector_ids == ()
    proof = replay.partitions[0]
    assert proof.proof_algorithm == (
        "canonical-model-contract-label-equivariance-v1"
    )
    assert proof.proof_digest is not None


def test_failed_replay_partition_does_not_disable_other_proven_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BuiltinSMModel()
    process = build_process_ir("g g > g g g")
    color_plan = build_color_plan(
        process,
        fold_trace_reflections=model.lc_trace_reflection_equivalence_is_proven(
            process
        ),
    )
    candidates = tuple(
        partition
        for partition in replay_module.lc_topology_replay_partitions(color_plan)
        if len(partition.active_sector_ids) > 1
    )
    assert len(candidates) > 1
    failed_partition = candidates[1]
    original = replay_module._prove_lc_topology_replay_partition

    def fail_one_partition(
        plan: object,
        partition: object,
        proof_model: object,
        *,
        model_contract_digest: str,
    ) -> object:
        if (
            partition.representative_sector_id  # type: ignore[attr-defined]
            == failed_partition.representative_sector_id
        ):
            return None
        return original(
            plan,  # type: ignore[arg-type]
            partition,  # type: ignore[arg-type]
            proof_model,  # type: ignore[arg-type]
            model_contract_digest=model_contract_digest,
        )

    monkeypatch.setattr(
        replay_module,
        "_prove_lc_topology_replay_partition",
        fail_one_partition,
    )

    dag, coverage = GenerationBackend(None, None)._compile_concrete_process(
        process,
        model,
    )
    replay = dag.lc_topology_replay

    assert replay is not None
    assert replay.partitions
    assert all(
        part.representative_sector_id
        != failed_partition.representative_sector_id
        for part in replay.partitions
    )
    assert set(failed_partition.active_sector_ids).issubset(
        replay.residual_sector_ids
    )
    assert set(replay.physical_sector_ids) == (
        {
            sector_id
            for partition in replay.partitions
            for sector_id in partition.active_sector_ids
        }
        | set(replay.residual_sector_ids)
    )
    assert {
        int(root.color_sector_id)
        for root in dag.amplitude_roots
        if root.color_sector_id is not None
    } == set(replay.materialized_sector_ids)
    assert coverage["materialized_color_sector_count"] == len(
        replay.materialized_sector_ids
    )
    assert coverage["residual_color_sector_count"] == len(
        replay.residual_sector_ids
    )


def test_execution_manifest_carries_additive_replay_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BuiltinSMModel()
    process_ir = build_process_ir("d d~ > z g g")
    dag, _coverage = GenerationBackend(None, None)._compile_concrete_process(
        process_ir,
        model,
    )
    schema = build_runtime_expression_schema(dag, model)
    validation = ValidationPointRecord(
        process_id=process_ir.key,
        process=process_ir.process,
        seed=1,
        particles=tuple(
            (int(leg.pdg or 0), (1.0, 0.0, 0.0, 0.0))
            for leg in process_ir.legs
        ),
    )
    artifact = CompiledProcessArtifact(
        process_id=process_ir.key,
        expression=process_ir.process,
        color_accuracy="lc",
        external_pdgs=(*process_ir.initial_pdgs, *process_ir.final_pdgs),
        aliases=(),
        runtime_schema=schema,
        stage_manifest={},
        model_parameter_evaluator=None,
        dag_summary={
            "current_count": len(dag.currents),
            "source_count": len(dag.sources),
            "interaction_count": len(dag.interactions),
            "interaction_evaluation_count": dag.interaction_evaluation_count,
            "amplitude_root_count": len(dag.amplitude_roots),
            "truncated": dag.truncated,
        },
        evaluator_root=Path("."),
        validation_point=validation,
        generation_filters={},
    )
    monkeypatch.setattr(
        artifact_writer_module,
        "_stage_evaluator_set",
        lambda _manifest: {"required_runtime_capabilities": []},
    )

    manifest = artifact_writer_module._execution_manifest(
        artifact,
        schema.to_mapping(),
    )

    replay = manifest["compiled"]["lc_topology_replay"]
    assert replay["enabled"] is True
    assert replay["contract_version"] == 2
    assert replay["materialized_sector_ids"] == [0]
    assert replay["residual_sector_ids"] == []


def test_structural_helicity_reduction_preserves_lc_replay_contract() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("d d~ > z g g")
    dag, _coverage = GenerationBackend(None, None)._compile_concrete_process(
        process,
        model,
    )
    replay = dag.lc_topology_replay
    assert replay is not None

    reduced = prune_global_helicity_flip_equivalent_roots(dag, model)

    assert reduced.lc_topology_replay == replay
    schema = build_runtime_expression_schema(reduced, model).to_mapping()
    assert schema["lc_topology_replay"]["contract_version"] == 2
