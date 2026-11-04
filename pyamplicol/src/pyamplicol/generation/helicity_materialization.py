# SPDX-License-Identifier: 0BSD
"""Materialize one compiled recurrence representative per helicity proof class."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace

from .dag_types import AmplitudeRoot, GenericDAG, InteractionNode
from .helicity_replay import HelicityRecurrencePlan

_Weight = tuple[float, float]


@dataclass(frozen=True, slots=True)
class MaterializedSourceRoute:
    materialized_current_id: int
    external_label: int
    helicity: int
    chirality: int
    spin_state: int | tuple[int, ...]
    declared_state_index: int
    selector_domain_id: int
    factor: _Weight

    def to_runtime_manifest(self) -> dict[str, object]:
        spin_state: object = self.spin_state
        if isinstance(spin_state, tuple):
            spin_state = list(spin_state)
        return {
            "materialized_current_id": self.materialized_current_id,
            "external_label": self.external_label,
            "helicity": self.helicity,
            "chirality": self.chirality,
            "spin_state": spin_state,
            "declared_state_index": self.declared_state_index,
            "selector_domain_id": self.selector_domain_id,
            "factor": list(self.factor),
        }


@dataclass(frozen=True, slots=True)
class MaterializedAmplitudeRoute:
    materialized_root_id: int
    selector_domain_ids: tuple[int, ...]
    factor: _Weight
    residual: bool = False

    def to_runtime_manifest(self) -> dict[str, object]:
        return {
            "materialized_root_id": self.materialized_root_id,
            "selector_domain_ids": list(self.selector_domain_ids),
            "factor": list(self.factor),
            "residual": self.residual,
        }


@dataclass(frozen=True, slots=True)
class MaterializedSelectorSchedule:
    selector_domain_id: int
    active_current_ids: tuple[int, ...]
    active_root_ids: tuple[int, ...]
    structural_zero: bool

    def to_runtime_manifest(self) -> dict[str, object]:
        return {
            "selector_domain_id": self.selector_domain_id,
            "active_current_ids": list(self.active_current_ids),
            "active_root_ids": list(self.active_root_ids),
            "structural_zero": self.structural_zero,
        }


@dataclass(frozen=True, slots=True)
class HelicityMaterialization:
    strategy: str
    dag: GenericDAG
    proof_current_count: int
    proof_root_count: int
    proof_to_materialized_current: tuple[int, ...]
    source_routes: tuple[MaterializedSourceRoute, ...]
    amplitude_routes: tuple[MaterializedAmplitudeRoute, ...]
    selector_schedules: tuple[MaterializedSelectorSchedule, ...]

    @property
    def materialized_current_count(self) -> int:
        return len(self.dag.currents)

    @property
    def materialized_root_count(self) -> int:
        return len(self.dag.amplitude_roots)

    def to_runtime_manifest(self) -> dict[str, object]:
        return {
            "kind": "pyamplicol-helicity-recurrence-materialization",
            "contract_version": 1,
            "strategy": self.strategy,
            "proof_current_count": self.proof_current_count,
            "proof_root_count": self.proof_root_count,
            "materialized_current_count": self.materialized_current_count,
            "materialized_root_count": self.materialized_root_count,
            "proof_to_materialized_current": list(self.proof_to_materialized_current),
            "source_routes": [
                route.to_runtime_manifest() for route in self.source_routes
            ],
            "amplitude_routes": [
                route.to_runtime_manifest() for route in self.amplitude_routes
            ],
            "selector_schedules": [
                schedule.to_runtime_manifest() for schedule in self.selector_schedules
            ],
        }


def materialize_helicity_recurrence(
    dag: GenericDAG,
    plan: HelicityRecurrencePlan,
) -> HelicityMaterialization:
    """Build the quotient DAG certified by ``plan``.

    Every proven recurrence class contributes exactly one current. Residual
    currents remain one-to-one. Parent recurrence factors are folded into the
    remapped interaction coefficient, so compiled kernels see ordinary current
    slots and require no selector-dependent branches.
    """

    if plan.current_count != len(dag.currents):
        raise ValueError("helicity proof current count does not match the DAG")
    if plan.amplitude_root_count != len(dag.amplitude_roots):
        raise ValueError("helicity proof root count does not match the DAG")

    class_by_current: dict[int, tuple[object, _Weight]] = {}
    representative_factor_by_class: dict[str, _Weight] = {}
    target_old_ids: set[int] = set(plan.residual_current_ids)
    for recurrence in plan.recurrence_classes:
        representative = next(
            member
            for member in recurrence.members
            if member.current_id == recurrence.representative_current_id
        )
        representative_factor_by_class[recurrence.class_id] = representative.factor
        target_old_ids.add(recurrence.representative_current_id)
        for member in recurrence.members:
            class_by_current[member.current_id] = (recurrence, member.factor)

    target_order = tuple(
        sorted(
            target_old_ids,
            key=lambda current_id: (
                len(dag.currents[current_id].index.external_labels),
                current_id,
            ),
        )
    )
    target_new_id = {old_id: new_id for new_id, old_id in enumerate(target_order)}

    proof_to_materialized: list[int] = [-1] * len(dag.currents)
    factor_by_proof_current: list[_Weight] = [(1.0, 0.0)] * len(dag.currents)
    materialized_factor_by_proof_current: list[_Weight] = [(1.0, 0.0)] * len(
        dag.currents
    )
    residual_ids = set(plan.residual_current_ids)
    for current in dag.currents:
        if current.id in residual_ids:
            proof_to_materialized[current.id] = target_new_id[current.id]
            continue
        recurrence, factor = class_by_current[current.id]
        representative_id = recurrence.representative_current_id
        proof_to_materialized[current.id] = target_new_id[representative_id]
        factor_by_proof_current[current.id] = factor
        materialized_factor_by_proof_current[current.id] = (
            representative_factor_by_class[recurrence.class_id]
        )
    if any(current_id < 0 for current_id in proof_to_materialized):
        raise ValueError("helicity quotient left a proof current unmapped")

    currents = tuple(
        replace(dag.currents[old_id], id=new_id)
        for new_id, old_id in enumerate(target_order)
    )
    interactions_by_result: dict[int, list[InteractionNode]] = defaultdict(list)
    for interaction in dag.interactions:
        interactions_by_result[interaction.result_id].append(interaction)
    interactions: list[InteractionNode] = []
    for old_result_id in target_order:
        new_result_id = target_new_id[old_result_id]
        for interaction in interactions_by_result[old_result_id]:
            left_ratio = _divide_weight(
                factor_by_proof_current[interaction.left_id],
                materialized_factor_by_proof_current[interaction.left_id],
            )
            right_ratio = _divide_weight(
                factor_by_proof_current[interaction.right_id],
                materialized_factor_by_proof_current[interaction.right_id],
            )
            interactions.append(
                replace(
                    interaction,
                    id=len(interactions),
                    left_id=proof_to_materialized[interaction.left_id],
                    right_id=proof_to_materialized[interaction.right_id],
                    result_id=new_result_id,
                    color_weight=_multiply_weight(
                        interaction.color_weight,
                        _multiply_weight(left_ratio, right_ratio),
                    ),
                )
            )

    roots: list[AmplitudeRoot] = []
    amplitude_routes: list[MaterializedAmplitudeRoute] = []
    retained_roots: list[tuple[AmplitudeRoot, object | None]] = []
    for recurrence in plan.amplitude_classes:
        representative = next(
            member
            for member in recurrence.members
            if member.root_id == recurrence.representative_root_id
        )
        retained_roots.append(
            (dag.amplitude_roots[recurrence.representative_root_id], recurrence)
        )
        materialized_root_id = len(retained_roots) - 1
        for member in recurrence.members:
            amplitude_routes.append(
                MaterializedAmplitudeRoute(
                    materialized_root_id=materialized_root_id,
                    selector_domain_ids=member.selector_domain_ids,
                    factor=_divide_weight(member.factor, representative.factor),
                )
            )
    for root_id in plan.residual_root_ids:
        retained_roots.append((dag.amplitude_roots[root_id], None))

    domain_by_state = {
        domain.source_states: domain.id for domain in plan.selector_domains
    }
    source_state_by_bit = _source_state_by_ancestry_bit(dag)
    for root, recurrence in retained_roots:
        left_ratio = _divide_weight(
            factor_by_proof_current[root.left_id],
            materialized_factor_by_proof_current[root.left_id],
        )
        right_ratio = _divide_weight(
            factor_by_proof_current[root.right_id],
            materialized_factor_by_proof_current[root.right_id],
        )
        materialized_root_id = len(roots)
        roots.append(
            replace(
                root,
                id=materialized_root_id,
                left_id=proof_to_materialized[root.left_id],
                right_id=proof_to_materialized[root.right_id],
                color_weight=_multiply_weight(
                    root.color_weight,
                    _multiply_weight(left_ratio, right_ratio),
                ),
                helicity_weight=1.0,
            )
        )
        if recurrence is None:
            selector_states = _residual_root_selector_states(
                dag,
                root,
                source_state_by_bit,
            )
            selector_domain_ids = tuple(
                domain_by_state[state]
                for state in selector_states
                if state in domain_by_state
            )
            if len(selector_domain_ids) != len(selector_states):
                raise ValueError(
                    "residual amplitude root has no runtime selector domain"
                )
            amplitude_routes.append(
                MaterializedAmplitudeRoute(
                    materialized_root_id=materialized_root_id,
                    selector_domain_ids=selector_domain_ids,
                    factor=(1.0, 0.0),
                    residual=True,
                )
            )

    source_routes: list[MaterializedSourceRoute] = []
    for mapping in plan.source_state_mappings:
        recurrence, _factor = class_by_current[mapping.current_id]
        representative_factor = representative_factor_by_class[recurrence.class_id]
        source_routes.append(
            MaterializedSourceRoute(
                materialized_current_id=proof_to_materialized[mapping.current_id],
                external_label=mapping.external_label,
                helicity=mapping.helicity,
                chirality=mapping.chirality,
                spin_state=mapping.spin_state,
                declared_state_index=mapping.declared_state_index,
                selector_domain_id=mapping.selector_domain_id,
                factor=_divide_weight(mapping.factor, representative_factor),
            )
        )

    materialized = GenericDAG(
        process=dag.process,
        color_plan=dag.color_plan,
        currents=currents,
        sources=tuple(
            target_new_id[old_id]
            for old_id in target_order
            if dag.currents[old_id].is_source
        ),
        interactions=tuple(interactions),
        amplitude_roots=tuple(roots),
        truncated=dag.truncated,
        helicity_coverage=dag.helicity_coverage,
        color_coverage=dag.color_coverage,
        selected_source_helicities=dag.selected_source_helicities,
        selected_color_sector_ids=dag.selected_color_sector_ids,
        lc_topology_replay=dag.lc_topology_replay,
        color_topology_replay=dag.color_topology_replay,
    )
    routes_by_domain: dict[int, set[int]] = defaultdict(set)
    for route in amplitude_routes:
        for domain_id in route.selector_domain_ids:
            routes_by_domain[domain_id].add(route.materialized_root_id)
    interactions_by_materialized_result: dict[int, list[InteractionNode]] = defaultdict(
        list
    )
    for interaction in interactions:
        interactions_by_materialized_result[interaction.result_id].append(interaction)
    compiled_dependencies = _compiled_representative_dependencies(materialized)
    structural_zeros = set(plan.structural_zero_selector_domain_ids)
    selector_schedules: list[MaterializedSelectorSchedule] = []
    for domain in plan.selector_domains:
        if not domain.complete:
            continue
        active_roots = tuple(sorted(routes_by_domain.get(domain.id, set())))
        active_currents = _active_current_closure(
            materialized,
            active_roots,
            interactions_by_materialized_result,
            compiled_dependencies=compiled_dependencies,
        )
        structural_zero = domain.id in structural_zeros
        if structural_zero != (not active_roots):
            raise ValueError(
                "helicity selector schedule disagrees with structural-zero proof"
            )
        selector_schedules.append(
            MaterializedSelectorSchedule(
                selector_domain_id=domain.id,
                active_current_ids=active_currents,
                active_root_ids=active_roots,
                structural_zero=structural_zero,
            )
        )
    return HelicityMaterialization(
        strategy="quotient",
        dag=materialized,
        proof_current_count=len(dag.currents),
        proof_root_count=len(dag.amplitude_roots),
        proof_to_materialized_current=tuple(proof_to_materialized),
        source_routes=tuple(source_routes),
        amplitude_routes=tuple(amplitude_routes),
        selector_schedules=tuple(selector_schedules),
    )


def retain_helicity_recurrence_graph(
    dag: GenericDAG,
    plan: HelicityRecurrencePlan,
) -> HelicityMaterialization:
    """Attach selector schedules while retaining the shared proof DAG.

    Compiled summed-helicity execution benefits from common subexpressions
    between physical helicities.  Quotienting the graph to one current per
    recurrence class removes that sharing and requires repeated evaluator
    calls.  This form keeps every proof current for the fused summed-axis hot
    path, while selector schedules identify the exact chunk closure needed by
    a runtime-selected helicity.
    """

    if plan.current_count != len(dag.currents):
        raise ValueError("helicity proof current count does not match the DAG")
    if plan.amplitude_root_count != len(dag.amplitude_roots):
        raise ValueError("helicity proof root count does not match the DAG")

    amplitude_routes: list[MaterializedAmplitudeRoute] = []
    routed_root_ids: set[int] = set()
    for recurrence in plan.amplitude_classes:
        for member in recurrence.members:
            if member.root_id in routed_root_ids:
                raise ValueError("helicity proof amplitude root is routed twice")
            routed_root_ids.add(member.root_id)
            root_weight = float(dag.amplitude_roots[member.root_id].helicity_weight)
            if not math.isfinite(root_weight) or root_weight <= 0.0:
                raise ValueError(
                    "retained amplitude root has an invalid helicity weight"
                )
            amplitude_routes.append(
                MaterializedAmplitudeRoute(
                    materialized_root_id=member.root_id,
                    selector_domain_ids=member.selector_domain_ids,
                    # The fused sum keeps the root's physical-helicity weight.
                    # Selecting one member must remove that multiplicity before
                    # the amplitude is squared.
                    factor=(1.0 / math.sqrt(root_weight), 0.0),
                )
            )

    domain_by_state = {
        domain.source_states: domain.id for domain in plan.selector_domains
    }
    source_state_by_bit = _source_state_by_ancestry_bit(dag)
    for root_id in plan.residual_root_ids:
        if root_id in routed_root_ids:
            raise ValueError("residual amplitude root is also proof-routed")
        root = dag.amplitude_roots[root_id]
        selector_states = _residual_root_selector_states(
            dag,
            root,
            source_state_by_bit,
        )
        selector_domain_ids = tuple(
            domain_by_state[state]
            for state in selector_states
            if state in domain_by_state
        )
        if len(selector_domain_ids) != len(selector_states):
            raise ValueError("residual amplitude root has no runtime selector domain")
        routed_root_ids.add(root_id)
        root_weight = float(root.helicity_weight)
        if not math.isfinite(root_weight) or root_weight <= 0.0:
            raise ValueError("retained residual root has an invalid helicity weight")
        amplitude_routes.append(
            MaterializedAmplitudeRoute(
                materialized_root_id=root_id,
                selector_domain_ids=selector_domain_ids,
                factor=(1.0 / math.sqrt(root_weight), 0.0),
                residual=True,
            )
        )
    if routed_root_ids != set(range(len(dag.amplitude_roots))):
        raise ValueError("helicity proof does not route every retained amplitude root")

    source_routes = tuple(
        MaterializedSourceRoute(
            materialized_current_id=mapping.current_id,
            external_label=mapping.external_label,
            helicity=mapping.helicity,
            chirality=mapping.chirality,
            spin_state=mapping.spin_state,
            declared_state_index=mapping.declared_state_index,
            selector_domain_id=mapping.selector_domain_id,
            factor=(1.0, 0.0),
        )
        for mapping in plan.source_state_mappings
    )

    routes_by_domain: dict[int, set[int]] = defaultdict(set)
    for route in amplitude_routes:
        for domain_id in route.selector_domain_ids:
            routes_by_domain[domain_id].add(route.materialized_root_id)
    interactions_by_result: dict[int, list[InteractionNode]] = defaultdict(list)
    for interaction in dag.interactions:
        interactions_by_result[interaction.result_id].append(interaction)
    compiled_dependencies = _compiled_representative_dependencies(dag)
    structural_zeros = set(plan.structural_zero_selector_domain_ids)
    selector_schedules: list[MaterializedSelectorSchedule] = []
    for domain in plan.selector_domains:
        if not domain.complete:
            continue
        active_roots = tuple(sorted(routes_by_domain.get(domain.id, set())))
        active_currents = _active_current_closure(
            dag,
            active_roots,
            interactions_by_result,
            compiled_dependencies=compiled_dependencies,
        )
        structural_zero = domain.id in structural_zeros
        if structural_zero != (not active_roots):
            raise ValueError(
                "helicity selector schedule disagrees with structural-zero proof"
            )
        selector_schedules.append(
            MaterializedSelectorSchedule(
                selector_domain_id=domain.id,
                active_current_ids=active_currents,
                active_root_ids=active_roots,
                structural_zero=structural_zero,
            )
        )

    return HelicityMaterialization(
        strategy="retained-proof-graph",
        dag=dag,
        proof_current_count=len(dag.currents),
        proof_root_count=len(dag.amplitude_roots),
        proof_to_materialized_current=tuple(range(len(dag.currents))),
        source_routes=source_routes,
        amplitude_routes=tuple(
            sorted(amplitude_routes, key=lambda route: route.materialized_root_id)
        ),
        selector_schedules=tuple(selector_schedules),
    )


def _active_current_closure(
    dag: GenericDAG,
    root_ids: tuple[int, ...],
    interactions_by_result: dict[int, list[InteractionNode]],
    *,
    compiled_dependencies: Mapping[int, tuple[int, ...]] | None = None,
) -> tuple[int, ...]:
    active: set[int] = set()
    pending: list[int] = []
    for root_id in root_ids:
        root = dag.amplitude_roots[root_id]
        pending.extend((root.left_id, root.right_id))
    while pending:
        current_id = pending.pop()
        if current_id in active:
            continue
        active.add(current_id)
        for interaction in interactions_by_result.get(current_id, ()):
            pending.extend((interaction.left_id, interaction.right_id))
        if compiled_dependencies is not None:
            pending.extend(compiled_dependencies.get(current_id, ()))
    return tuple(sorted(active))


def _compiled_representative_dependencies(
    dag: GenericDAG,
) -> dict[int, tuple[int, ...]]:
    """Return the concrete parents read after evaluation-group reuse.

    A compiled stage caches the first interaction in each evaluation group and
    reuses that expression for later group members. Selector schedules must
    therefore retain the representative interaction's parents in addition to
    each current's physical DAG parents.
    """

    interactions_by_stage: dict[int, dict[int, list[InteractionNode]]] = {}
    for interaction in dag.interactions:
        result_id = int(interaction.result_id)
        subset_size = len(dag.currents[result_id].index.external_labels)
        interactions_by_stage.setdefault(subset_size, {}).setdefault(
            result_id, []
        ).append(interaction)

    dependencies: dict[int, set[int]] = {}
    for results in interactions_by_stage.values():
        representative_by_group: dict[int, InteractionNode] = {}
        for result_id in sorted(results):
            for interaction in results[result_id]:
                group_id = interaction.evaluation_group_id
                representative = interaction
                if group_id is not None:
                    representative = representative_by_group.setdefault(
                        int(group_id), interaction
                    )
                dependencies.setdefault(result_id, set()).update(
                    (int(representative.left_id), int(representative.right_id))
                )
    return {
        current_id: tuple(sorted(parent_ids))
        for current_id, parent_ids in dependencies.items()
    }


def _source_state_by_ancestry_bit(
    dag: GenericDAG,
) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for current in dag.currents:
        if not current.is_source:
            continue
        ancestry = int(current.index.helicity_ancestry)
        if ancestry <= 0 or ancestry & (ancestry - 1):
            continue
        if current.source_leg_label is None or current.source_helicity is None:
            continue
        result[ancestry.bit_length() - 1] = (
            int(current.source_leg_label),
            int(current.source_helicity),
        )
    return result


def _root_selector_state(
    dag: GenericDAG,
    root: AmplitudeRoot,
    source_state_by_bit: dict[int, tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    ancestry = int(
        dag.currents[root.left_id].index.helicity_ancestry
        | dag.currents[root.right_id].index.helicity_ancestry
    )
    state: dict[int, int] = {}
    while ancestry:
        bit = ancestry & -ancestry
        selector = source_state_by_bit.get(bit.bit_length() - 1)
        if selector is None:
            raise ValueError("residual root has an unknown source ancestry bit")
        label, helicity = selector
        previous = state.setdefault(label, helicity)
        if previous != helicity:
            raise ValueError("residual root selects two helicities for one leg")
        ancestry ^= bit
    return tuple(sorted(state.items()))


def _residual_root_selector_states(
    dag: GenericDAG,
    root: AmplitudeRoot,
    source_state_by_bit: dict[int, tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    selector_state = _root_selector_state(dag, root, source_state_by_bit)
    selector_states = [selector_state]
    if math.isclose(root.helicity_weight, 2.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
        flipped = tuple((label, -helicity) for label, helicity in selector_state)
        if flipped != selector_state:
            selector_states.append(flipped)
    elif not math.isclose(
        root.helicity_weight,
        1.0,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "residual amplitude root has unsupported helicity multiplicity "
            f"{root.helicity_weight}"
        )
    return tuple(selector_states)


def _multiply_weight(left: _Weight, right: _Weight) -> _Weight:
    value = complex(*left) * complex(*right)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("helicity replay produced a non-finite weight")
    return (float(value.real), float(value.imag))


def _divide_weight(numerator: _Weight, denominator: _Weight) -> _Weight:
    denominator_value = complex(*denominator)
    if denominator_value == 0j:
        raise ValueError("helicity replay representative factor is zero")
    value = complex(*numerator) / denominator_value
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("helicity replay produced a non-finite factor ratio")
    return (float(value.real), float(value.imag))
