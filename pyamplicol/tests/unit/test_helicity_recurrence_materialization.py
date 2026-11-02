# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from types import MethodType

import pytest

from pyamplicol.generation.dag_algorithms import (
    prune_global_helicity_flip_equivalent_roots,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.helicity_materialization import (
    materialize_helicity_recurrence,
    retain_helicity_recurrence_graph,
)
from pyamplicol.generation.helicity_replay import (
    build_helicity_recurrence_plan,
)
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model, VertexEvaluationEquivalence
from pyamplicol.models.builtin.process_ir import build_process_ir


def _materialize(expression: str):
    model = BuiltinSMModel()
    full = compile_generic_dag(build_process_ir(expression), model=model)
    reduced = prune_global_helicity_flip_equivalent_roots(full, model)
    plan = build_helicity_recurrence_plan(reduced, model)
    assert plan is not None
    return reduced, plan, materialize_helicity_recurrence(reduced, plan)


def test_zjet_materializes_one_current_and_root_per_proven_class() -> None:
    full, plan, materialization = _materialize("d d~ > z g")

    assert materialization.strategy == "quotient"
    assert not plan.residual_current_ids
    assert not plan.residual_root_ids
    assert materialization.materialized_current_count == len(plan.recurrence_classes)
    assert materialization.materialized_root_count == len(plan.amplitude_classes)
    assert materialization.materialized_current_count < len(full.currents)
    assert materialization.materialized_root_count < len(full.amplitude_roots)
    assert len(materialization.proof_to_materialized_current) == len(full.currents)
    assert set(materialization.proof_to_materialized_current) == set(
        range(materialization.materialized_current_count)
    )
    assert len(materialization.dag.interactions) < len(full.interactions)


def test_weight_two_roots_expand_to_two_physical_selector_routes() -> None:
    _full, plan, materialization = _materialize("g g > g g")

    expected_domains = {
        domain.id for domain in plan.selector_domains if domain.complete
    } - set(plan.structural_zero_selector_domain_ids)
    routed_domains = {
        domain_id
        for route in materialization.amplitude_routes
        for domain_id in route.selector_domain_ids
    }
    assert routed_domains == expected_domains
    assert all(
        root.helicity_weight == 1.0 for root in materialization.dag.amplitude_roots
    )
    assert all(route.factor == (1.0, 0.0) for route in materialization.amplitude_routes)
    assert {
        schedule.selector_domain_id
        for schedule in materialization.selector_schedules
        if schedule.structural_zero
    } == set(plan.structural_zero_selector_domain_ids)
    assert all(
        not schedule.active_current_ids and not schedule.active_root_ids
        for schedule in materialization.selector_schedules
        if schedule.structural_zero
    )
    assert all(
        schedule.active_current_ids and schedule.active_root_ids
        for schedule in materialization.selector_schedules
        if not schedule.structural_zero
    )


def test_runtime_source_routes_cover_every_declared_physical_state() -> None:
    _full, plan, materialization = _materialize("g g > t t~")

    expected = {
        (mapping.external_label, mapping.helicity, mapping.chirality)
        for mapping in plan.source_state_mappings
    }
    actual = {
        (route.external_label, route.helicity, route.chirality)
        for route in materialization.source_routes
    }
    assert actual == expected
    assert all(
        0 <= route.materialized_current_id < len(materialization.dag.currents)
        for route in materialization.source_routes
    )


def test_retained_graph_keeps_fused_sum_and_exact_selector_closures() -> None:
    model = BuiltinSMModel()
    full = compile_generic_dag(build_process_ir("d d~ > z g g"), model=model)
    reduced = prune_global_helicity_flip_equivalent_roots(full, model)
    plan = build_helicity_recurrence_plan(reduced, model)
    assert plan is not None

    retained = retain_helicity_recurrence_graph(reduced, plan)

    assert retained.strategy == "retained-proof-graph"
    assert retained.dag is reduced
    assert retained.proof_to_materialized_current == tuple(
        range(len(reduced.currents))
    )
    assert retained.materialized_current_count == len(reduced.currents)
    assert retained.materialized_root_count == len(reduced.amplitude_roots)
    assert {
        route.materialized_root_id for route in retained.amplitude_routes
    } == set(range(len(reduced.amplitude_roots)))
    for route in retained.amplitude_routes:
        expected = 1.0 / math.sqrt(
            retained.dag.amplitude_roots[
                route.materialized_root_id
            ].helicity_weight
        )
        assert route.factor == pytest.approx((expected, 0.0))
    assert all(route.factor == (1.0, 0.0) for route in retained.source_routes)
    assert all(
        bool(schedule.active_root_ids) is not schedule.structural_zero
        for schedule in retained.selector_schedules
    )
    assert any(
        len(schedule.active_current_ids) < len(reduced.currents)
        for schedule in retained.selector_schedules
        if not schedule.structural_zero
    )


def test_local_residuals_remain_materialized_without_disabling_other_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BuiltinSMModel()
    full = compile_generic_dag(build_process_ir("d d~ > z g"), model=model)
    full = prune_global_helicity_flip_equivalent_roots(full, model)
    failed_kind = min(interaction.vertex_kind for interaction in full.interactions)
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
    plan = build_helicity_recurrence_plan(full, model)
    assert plan is not None
    materialization = materialize_helicity_recurrence(full, plan)

    assert plan.residual_current_ids
    assert plan.residual_root_ids
    assert plan.optimized_class_count > 0
    assert len(materialization.dag.currents) <= len(full.currents)
    assert any(route.residual for route in materialization.amplitude_routes)

    complete_domains = {
        domain.id for domain in plan.selector_domains if domain.complete
    }
    schedules = {
        schedule.selector_domain_id: schedule
        for schedule in materialization.selector_schedules
    }
    routed_domains = {
        domain_id
        for route in materialization.amplitude_routes
        for domain_id in route.selector_domain_ids
    }
    assert set(schedules) == complete_domains
    assert routed_domains == {
        domain_id
        for domain_id in complete_domains
        if not schedules[domain_id].structural_zero
    }
    assert all(
        bool(schedule.active_root_ids) is not schedule.structural_zero
        for schedule in schedules.values()
    )
    selector_states_by_domain = {
        domain.id: set(domain.source_states) for domain in plan.selector_domains
    }
    for schedule in schedules.values():
        required_sources = {
            current_id
            for current_id in schedule.active_current_ids
            if materialization.dag.currents[current_id].is_source
        }
        complete_state = selector_states_by_domain[schedule.selector_domain_id]
        routed_sources = {
            route.materialized_current_id
            for route in materialization.source_routes
            if selector_states_by_domain[route.selector_domain_id] <= complete_state
        }
        assert required_sources <= routed_sources
    assert {
        (route.external_label, route.helicity)
        for route in materialization.source_routes
    } == {
        state
        for domain in plan.selector_domains
        for state in domain.source_states
    }
