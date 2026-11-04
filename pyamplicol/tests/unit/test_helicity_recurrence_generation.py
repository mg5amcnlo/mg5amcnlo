# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MethodType

import pytest

import pyamplicol.generation.service as service_module
from pyamplicol.api import ProcessRequest
from pyamplicol.config import GenerationConfig
from pyamplicol.generation.dag_algorithms import (
    prune_global_helicity_flip_equivalent_roots,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.helicity_replay import (
    HELICITY_RECURRENCE_CONTRACT_VERSION,
    RUNTIME_SELECTOR_PROVENANCE,
    HelicityRecurrencePlan,
    build_helicity_recurrence_plan,
)
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.generation.runtime_schema import build_runtime_schema
from pyamplicol.models import BuiltinSMModel, CompiledUFOModel, compile_model_source
from pyamplicol.models.base import Model, VertexEvaluationEquivalence
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.processes.model import build_model_process_ir

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "src" / "pyamplicol" / "assets" / "models" / "json" / "sm"


@pytest.fixture(scope="module")
def ufo_sm() -> CompiledUFOModel:
    compiled = compile_model_source(
        MODEL_ROOT / "sm.json",
        restriction=str((MODEL_ROOT / "restrict_default.json").resolve()),
        use_cache=False,
    )
    return CompiledUFOModel(compiled)


def _built_in_plan(expression: str) -> tuple[object, HelicityRecurrencePlan]:
    model = BuiltinSMModel()
    dag = compile_generic_dag(build_process_ir(expression), model=model)
    reduced = prune_global_helicity_flip_equivalent_roots(dag, model)
    plan = build_helicity_recurrence_plan(reduced, model)
    assert plan is not None
    return reduced, plan


def _class_topology(plan: HelicityRecurrencePlan) -> tuple[object, ...]:
    return (
        sorted(
            (
                recurrence.source_class,
                recurrence.external_labels,
                len(recurrence.members),
            )
            for recurrence in plan.recurrence_classes
        ),
        sorted(len(recurrence.members) for recurrence in plan.amplitude_classes),
        plan.proof_counts(),
    )


def test_builtin_and_ufo_zjet_derive_the_same_recurrence_topology(
    ufo_sm: CompiledUFOModel,
) -> None:
    built_in_model = BuiltinSMModel()
    built_in_dag = compile_generic_dag(
        build_process_ir("d d~ > z g"),
        model=built_in_model,
    )
    built_in_dag = prune_global_helicity_flip_equivalent_roots(
        built_in_dag,
        built_in_model,
    )
    built_in_plan = build_helicity_recurrence_plan(
        built_in_dag,
        built_in_model,
    )

    ufo_dag = compile_generic_dag(
        build_model_process_ir("d d~ > z g", ufo_sm.compiled.ir),
        model=ufo_sm,
    )
    ufo_dag = prune_global_helicity_flip_equivalent_roots(ufo_dag, ufo_sm)
    ufo_plan = build_helicity_recurrence_plan(ufo_dag, ufo_sm)

    assert built_in_plan is not None
    assert ufo_plan is not None
    assert _class_topology(built_in_plan) == _class_topology(ufo_plan)
    assert not built_in_plan.residual_current_ids
    assert not ufo_plan.residual_current_ids
    assert any(
        contract.startswith("symbolica-sha256:")
        for recurrence in ufo_plan.recurrence_classes
        for contract in recurrence.transition_contract_ids
    )


def test_massive_qcd_source_helicities_share_recurrence_classes() -> None:
    _dag, plan = _built_in_plan("g g > t t~")

    for label in (3, 4):
        mappings = tuple(
            mapping
            for mapping in plan.source_state_mappings
            if mapping.external_label == label
        )
        assert {mapping.helicity for mapping in mappings} == {-1, 1}
        assert {mapping.chirality for mapping in mappings} == {0}
        assert len({mapping.recurrence_class_id for mapping in mappings}) == 1


def test_massless_chiral_source_states_remain_separate() -> None:
    _dag, plan = _built_in_plan("d d~ > z g")

    for label in (1, 2):
        mappings = tuple(
            mapping
            for mapping in plan.source_state_mappings
            if mapping.external_label == label
        )
        assert {mapping.helicity for mapping in mappings} == {-1, 1}
        assert {mapping.chirality for mapping in mappings} == {-1, 1}
        assert len({mapping.recurrence_class_id for mapping in mappings}) == 2


def test_unproven_transition_becomes_a_local_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(build_process_ir("d d~ > z g"), model=model)
    dag = prune_global_helicity_flip_equivalent_roots(dag, model)
    failed_kind = min(interaction.vertex_kind for interaction in dag.interactions)
    original = model.vertex_evaluation_equivalence

    def with_one_failed_contract(
        self: Model,
        kind: int,
    ) -> VertexEvaluationEquivalence:
        del self
        if kind == failed_kind:
            return VertexEvaluationEquivalence(
                class_id=f"deliberately-unproven:{kind}",
                verified=False,
            )
        return original(kind)

    monkeypatch.setattr(
        model,
        "vertex_evaluation_equivalence",
        MethodType(with_one_failed_contract, model),
    )
    plan = build_helicity_recurrence_plan(dag, model)

    assert plan is not None
    assert plan.residual_current_ids
    assert len(plan.residual_current_ids) < len(dag.currents)
    assert plan.optimized_class_count > 0
    assert any(f"kind {failed_kind}" in diagnostic for diagnostic in plan.diagnostics)


def test_post_reduction_manifest_ids_and_weight_two_domains_are_complete() -> None:
    model = BuiltinSMModel()
    raw = compile_generic_dag(build_process_ir("g g > g g"), model=model)
    reduced = prune_global_helicity_flip_equivalent_roots(raw, model)
    plan = build_helicity_recurrence_plan(reduced, model)

    assert plan is not None
    assert len(reduced.currents) < len(raw.currents)
    assert {root.helicity_weight for root in reduced.amplitude_roots} == {2.0}
    assert {
        member.current_id
        for recurrence in plan.recurrence_classes
        for member in recurrence.members
    } | set(plan.residual_current_ids) == set(range(len(reduced.currents)))
    assert {
        member.root_id
        for recurrence in plan.amplitude_classes
        for member in recurrence.members
    } | set(plan.residual_root_ids) == set(range(len(reduced.amplitude_roots)))
    assert all(
        len(member.selector_domain_ids) == 2
        for recurrence in plan.amplitude_classes
        for member in recurrence.members
    )
    assert plan.physical_helicity_count == 16


def test_generation_service_attaches_plan_after_final_dense_remap() -> None:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(GenerationConfig(), None)
    process_ir = build_process_ir("g g > g g")
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(
            expanded=service_module._ExpandedProcess(
                request=ProcessRequest.parse("g g > g g", name="gg_gg"),
                process_ir=process_ir,
            ),
            dag=dag,
            coverage=coverage,
        ),
        model,
        index=0,
        phase=PhaseHandle("test", None, 1),
    )

    plan = prepared.dag.helicity_recurrence
    materialization = prepared.dag.helicity_materialization
    assert plan is not None
    assert materialization is not None
    assert plan.current_count == materialization.proof_current_count
    assert plan.amplitude_root_count == materialization.proof_root_count
    assert materialization.materialized_current_count == len(prepared.dag.currents)
    assert materialization.materialized_root_count == len(prepared.dag.amplitude_roots)
    assert (
        max(
            member.current_id
            for recurrence in plan.recurrence_classes
            for member in recurrence.members
        )
        < materialization.proof_current_count
    )
    assert (
        max(
            member.root_id
            for recurrence in plan.amplitude_classes
            for member in recurrence.members
        )
        < materialization.proof_root_count
    )
    assert all(
        0 <= route.materialized_current_id < len(prepared.dag.currents)
        for route in materialization.source_routes
    )
    assert all(
        0 <= route.materialized_root_id < len(prepared.dag.amplitude_roots)
        for route in materialization.amplitude_routes
    )
    assert materialization.strategy == "quotient"
    assert {root.helicity_weight for root in prepared.dag.amplitude_roots} == {1.0}
    assert all(
        route.factor == pytest.approx((1.0, 0.0))
        for route in materialization.amplitude_routes
    )
    assert plan.physical_helicity_count == 16


def test_complete_artifact_emits_selector_provenance_and_contract() -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(build_process_ir("d d~ > z"), model=model)
    dag = prune_global_helicity_flip_equivalent_roots(dag, model)
    plan = build_helicity_recurrence_plan(dag, model)
    assert plan is not None
    schema = build_runtime_schema(
        replace(dag, helicity_recurrence=plan),
        model,
    )

    selector_contract = schema["physics"]["extensions"]["runtime_selectors"]
    assert selector_contract["provenance"] == RUNTIME_SELECTOR_PROVENANCE
    assert selector_contract["generation_specialized_axes"] == []
    assert selector_contract["helicity_recurrence"]["contract_version"] == (
        HELICITY_RECURRENCE_CONTRACT_VERSION
    )
    assert schema["helicity_recurrence"]["current_count"] == len(dag.currents)
    assert schema["helicity_recurrence"]["amplitude_root_count"] == len(
        dag.amplitude_roots
    )


def test_selected_helicity_keeps_complete_color_axis_reusable() -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("d d~ > z"),
        model=model,
        selected_source_helicities={1: -1},
    )
    before = dag.to_json_dict()

    assert build_helicity_recurrence_plan(dag, model) is None
    assert dag.to_json_dict() == before
    assert "helicity_recurrence" not in before
    schema = build_runtime_schema(dag, model)
    assert "helicity_recurrence" not in schema
    selectors = schema["physics"]["extensions"]["runtime_selectors"]
    assert selectors["generation_specialized_axes"] == ["helicity"]
    assert selectors["axes"]["helicity"]["runtime_contract"] == (
        "generation-specialized"
    )
    assert selectors["axes"]["color_flow"]["runtime_contract"] == ("complete-reusable")
    assert "helicity_recurrence" not in selectors


def test_selected_color_keeps_complete_helicity_axis_reusable() -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("d d~ > z g"),
        model=model,
        selected_color_sector_ids={0},
    )
    dag = prune_global_helicity_flip_equivalent_roots(dag, model)
    plan = build_helicity_recurrence_plan(dag, model)
    assert plan is not None
    schema = build_runtime_schema(
        replace(dag, helicity_recurrence=plan),
        model,
    )

    selectors = schema["physics"]["extensions"]["runtime_selectors"]
    assert selectors["generation_specialized_axes"] == ["color_flow"]
    assert selectors["axes"]["helicity"]["runtime_contract"] == ("complete-reusable")
    assert selectors["axes"]["color_flow"]["runtime_contract"] == (
        "generation-specialized"
    )
    assert selectors["helicity_recurrence"]["status"] == "available"
    assert "helicity_recurrence" in schema


def test_color_and_helicity_specialization_are_reported_independently() -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("d d~ > z g"),
        model=model,
        selected_color_sector_ids={0},
        selected_source_helicities={1: -1},
    )
    schema = build_runtime_schema(dag, model)

    selectors = schema["physics"]["extensions"]["runtime_selectors"]
    assert selectors["generation_specialized_axes"] == [
        "helicity",
        "color_flow",
    ]
    assert selectors["axes"]["helicity"]["generation_selection"] == {"1": -1}
    assert selectors["axes"]["color_flow"]["generation_selection"] == [0]
    assert "helicity_recurrence" not in selectors
