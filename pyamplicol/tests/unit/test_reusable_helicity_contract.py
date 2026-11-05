# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

import pyamplicol.generation.service as service_module
from pyamplicol.api import ProcessRequest
from pyamplicol.config import GenerationConfig
from pyamplicol.generation.dag_algorithms import (
    prune_global_helicity_flip_equivalent_roots,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.helicity_materialization import (
    HelicityMaterialization,
    materialize_helicity_recurrence,
)
from pyamplicol.generation.helicity_replay import build_helicity_recurrence_plan
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.generation.runtime_schema import build_runtime_schema
from pyamplicol.models import BuiltinSMModel, CompiledUFOModel, compile_model_source
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


def _prepare_builtin(
    expression: str,
    *,
    selected_source_helicities: dict[int, int] | None = None,
) -> tuple[object, object]:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(
        GenerationConfig(),
        None,
        process_selection=service_module._ProcessSelection(
            selected_source_helicities=selected_source_helicities,
        ),
    )
    process_ir = build_process_ir(expression)
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(
            expanded=service_module._ExpandedProcess(
                request=ProcessRequest.parse(expression, name="contract_case"),
                process_ir=process_ir,
            ),
            dag=dag,
            coverage=coverage,
        ),
        model,
        index=0,
        phase=PhaseHandle("test", None, 1),
    )
    return model, prepared


def _active_current_closure(
    materialization: HelicityMaterialization,
    root_ids: tuple[int, ...],
) -> tuple[int, ...]:
    interactions_by_result: dict[int, list[object]] = defaultdict(list)
    for interaction in materialization.dag.interactions:
        interactions_by_result[interaction.result_id].append(interaction)

    active: set[int] = set()
    pending: list[int] = []
    for root_id in root_ids:
        root = materialization.dag.amplitude_roots[root_id]
        pending.extend((root.left_id, root.right_id))
    while pending:
        current_id = pending.pop()
        if current_id in active:
            continue
        active.add(current_id)
        for interaction in interactions_by_result.get(current_id, ()):
            pending.extend((interaction.left_id, interaction.right_id))
    return tuple(sorted(active))


def _materialization_topology(materialization: HelicityMaterialization) -> object:
    return {
        "proof_counts": (
            materialization.proof_current_count,
            materialization.proof_root_count,
        ),
        "runtime_counts": (
            materialization.materialized_current_count,
            len(materialization.dag.interactions),
            materialization.materialized_root_count,
        ),
        "source_routes": sorted(
            (
                route.external_label,
                route.helicity,
                route.chirality,
                route.selector_domain_id,
            )
            for route in materialization.source_routes
        ),
        "amplitude_routes": sorted(
            (
                route.selector_domain_ids,
                route.residual,
            )
            for route in materialization.amplitude_routes
        ),
        "selector_schedules": sorted(
            (
                schedule.selector_domain_id,
                len(schedule.active_current_ids),
                len(schedule.active_root_ids),
                schedule.structural_zero,
            )
            for schedule in materialization.selector_schedules
        ),
    }


def test_complete_helicity_service_builds_exact_quotient_schedules() -> None:
    _model, prepared = _prepare_builtin("g g > g g")
    recurrence = prepared.dag.helicity_recurrence
    materialization = prepared.dag.helicity_materialization

    assert recurrence is not None
    assert materialization is not None
    complete_domains = {
        domain.id for domain in recurrence.selector_domains if domain.complete
    }
    assert {
        schedule.selector_domain_id
        for schedule in materialization.selector_schedules
    } == complete_domains

    roots_by_domain: dict[int, set[int]] = defaultdict(set)
    for route in materialization.amplitude_routes:
        for domain_id in route.selector_domain_ids:
            roots_by_domain[domain_id].add(route.materialized_root_id)
    for schedule in materialization.selector_schedules:
        expected_roots = tuple(
            sorted(roots_by_domain.get(schedule.selector_domain_id, ()))
        )
        assert schedule.active_root_ids == expected_roots
        assert schedule.active_current_ids == _active_current_closure(
            materialization,
            expected_roots,
        )


def test_generation_specialized_helicity_does_not_materialize_a_quotient() -> None:
    model, prepared = _prepare_builtin(
        "d d~ > z",
        selected_source_helicities={1: -1},
    )

    assert dict(prepared.dag.selected_source_helicities) == {1: -1}
    assert prepared.dag.helicity_recurrence is None
    assert prepared.dag.helicity_materialization is None
    assert "helicity_recurrence" not in prepared.filters
    schema = build_runtime_schema(prepared.dag, model)
    assert "helicity_recurrence" not in schema
    selectors = schema["physics"]["extensions"]["runtime_selectors"]
    assert selectors["generation_specialized_axes"] == ["helicity"]
    assert selectors["axes"]["helicity"]["runtime_contract"] == (
        "generation-specialized"
    )


def test_structural_zero_domains_have_empty_exact_schedules() -> None:
    _model, prepared = _prepare_builtin("g g > g g")
    recurrence = prepared.dag.helicity_recurrence
    materialization = prepared.dag.helicity_materialization
    assert recurrence is not None
    assert materialization is not None

    schedules = {
        schedule.selector_domain_id: schedule
        for schedule in materialization.selector_schedules
    }
    zero_domains = set(recurrence.structural_zero_selector_domain_ids)
    assert zero_domains
    assert {
        domain_id
        for domain_id, schedule in schedules.items()
        if schedule.structural_zero
    } == zero_domains
    assert all(
        not schedules[domain_id].active_current_ids
        and not schedules[domain_id].active_root_ids
        for domain_id in zero_domains
    )
    assert all(
        schedules[domain_id].active_current_ids
        and schedules[domain_id].active_root_ids
        for domain_id in schedules.keys() - zero_domains
    )


def test_builtin_and_ufo_sm_derive_the_same_materialized_topology(
    ufo_sm: CompiledUFOModel,
) -> None:
    expression = "d d~ > z g"
    builtin = BuiltinSMModel()
    builtin_dag = compile_generic_dag(build_process_ir(expression), model=builtin)
    builtin_dag = prune_global_helicity_flip_equivalent_roots(
        builtin_dag,
        builtin,
    )
    builtin_plan = build_helicity_recurrence_plan(builtin_dag, builtin)

    ufo_dag = compile_generic_dag(
        build_model_process_ir(expression, ufo_sm.compiled.ir),
        model=ufo_sm,
    )
    ufo_dag = prune_global_helicity_flip_equivalent_roots(ufo_dag, ufo_sm)
    ufo_plan = build_helicity_recurrence_plan(ufo_dag, ufo_sm)

    assert builtin_plan is not None
    assert ufo_plan is not None
    builtin_materialization = materialize_helicity_recurrence(
        builtin_dag,
        builtin_plan,
    )
    ufo_materialization = materialize_helicity_recurrence(ufo_dag, ufo_plan)
    assert _materialization_topology(builtin_materialization) == (
        _materialization_topology(ufo_materialization)
    )


def test_metadata_records_quotient_strategy_and_counts() -> None:
    model, prepared = _prepare_builtin("d d~ > z g")
    recurrence = prepared.dag.helicity_recurrence
    materialization = prepared.dag.helicity_materialization
    assert recurrence is not None
    assert materialization is not None
    assert materialization.strategy == "quotient"
    assert (
        materialization.materialized_current_count
        <= materialization.proof_current_count
    )
    assert materialization.materialized_root_count <= materialization.proof_root_count

    schema = build_runtime_schema(prepared.dag, model)
    manifest = schema["helicity_recurrence"]["materialization"]
    assert manifest["strategy"] == "quotient"
    assert manifest["proof_current_count"] == materialization.proof_current_count
    assert manifest["proof_root_count"] == materialization.proof_root_count
    assert manifest["materialized_current_count"] == len(prepared.dag.currents)
    assert manifest["materialized_root_count"] == len(prepared.dag.amplitude_roots)

    filter_counts = prepared.filters["helicity_recurrence"]
    assert filter_counts["materialization_strategy"] == "quotient"
    assert filter_counts["materialized_current_count"] == len(prepared.dag.currents)
    assert filter_counts["materialized_amplitude_count"] == len(
        prepared.dag.amplitude_roots
    )
    assert prepared.coverage["proof_current_count"] == (
        materialization.proof_current_count
    )
    assert prepared.coverage["helicity_materialization_strategy"] == (
        "quotient"
    )
    assert (
        prepared.coverage["current_count"]
        <= prepared.coverage["proof_current_count"]
    )
    assert prepared.coverage["amplitude_root_count"] <= (
        prepared.coverage["proof_amplitude_count"]
    )

    selector_counts = schema["physics"]["extensions"]["runtime_selectors"][
        "helicity_recurrence"
    ]
    assert selector_counts["execution"] == (
        "materialized-recurrence-quotient"
    )
    assert selector_counts["materialization_strategy"] == "quotient"
    assert selector_counts["proof_current_count"] == (
        materialization.proof_current_count
    )
    assert selector_counts["materialized_current_count"] == len(
        prepared.dag.currents
    )


def test_materialized_reduction_uses_the_quotient_helicity_representative() -> None:
    model, prepared = _prepare_builtin("g g > t t~")
    physics = build_runtime_schema(prepared.dag, model)["physics"]
    helicities = {record["id"]: record for record in physics["helicities"]}
    groups = physics["reduction"]["groups"]

    assert any(
        group["representative_helicity_id"] != group["physical_helicity_ids"][0]
        for group in groups
    )
    for group in groups:
        representatives = {
            helicities[identifier]["representative_id"]
            for identifier in group["physical_helicity_ids"]
            if not helicities[identifier]["structural_zero"]
        }
        assert representatives == {group["representative_helicity_id"]}
