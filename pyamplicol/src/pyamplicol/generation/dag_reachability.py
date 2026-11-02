# SPDX-License-Identifier: 0BSD
"""Particle-species and coupling-order reachability for generic DAGs."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from ..models.base import CouplingOrders, Model, Vertex
from ..processes.ir import CanonicalProcessIR
from .dag_ordering import _labels_mask
from .dag_types import ColorState

if TYPE_CHECKING:
    from .dag_color import ColorEngine


def _normalize_coupling_order_limits(
    limits: Mapping[str, int] | None,
) -> dict[str, int]:
    if limits is None:
        return {}
    return {
        str(name).upper(): int(value)
        for name, value in limits.items()
        if int(value) >= 0
    }


def _coupling_orders_within_limits(
    orders: tuple[tuple[str, int], ...],
    limits: Mapping[str, int],
) -> bool:
    if not limits:
        return True
    order_map = {str(name).upper(): int(value) for name, value in orders}
    return all(order_map.get(name, 0) <= int(limit) for name, limit in limits.items())


def _combine_coupling_order_tuples(
    *orders: CouplingOrders,
) -> CouplingOrders:
    totals: dict[str, int] = {}
    for order_tuple in orders:
        for name, value in order_tuple:
            totals[str(name).upper()] = totals.get(str(name).upper(), 0) + int(value)
    return tuple(sorted((name, value) for name, value in totals.items() if value))


def _lc_line_groups_within_limit(
    color_state: ColorState,
    limit: int | None,
) -> bool:
    if limit is None or color_state.accuracy != "lc":
        return True
    return len(color_state.line_groups) <= limit


def _right_particles_by_left(
    model: Model,
    *,
    color_accuracy: str,
) -> dict[int, tuple[int, ...]]:
    rights: dict[int, set[int]] = {}
    for vertex in model.iter_vertices(color_accuracy=color_accuracy):
        rights.setdefault(vertex.particles[0], set()).add(vertex.particles[1])
    return {
        left: tuple(sorted(right_particles)) for left, right_particles in rights.items()
    }


def _closure_right_particles_by_left(
    model: Model,
    *,
    color_accuracy: str,
) -> dict[int, tuple[int, ...]]:
    """Return sparse vertex and direct-contraction partners by left species."""

    rights = {
        left: set(right_particles)
        for left, right_particles in _right_particles_by_left(
            model,
            color_accuracy=color_accuracy,
        ).items()
    }
    for particle in model.particles.values():
        rights.setdefault(particle.pdg, set()).add(particle.anti_pdg)
        rights.setdefault(particle.anti_pdg, set()).add(particle.pdg)
    return {
        left: tuple(sorted(right_particles)) for left, right_particles in rights.items()
    }


UsefulStateMap = dict[int, dict[int, frozenset[CouplingOrders]]]
_ReachabilityState = tuple[int, int, CouplingOrders]


def _useful_states_by_mask(
    process_ir: CanonicalProcessIR,
    model: Model,
    color_engine: ColorEngine,
    closure_candidate_splits: Iterable[tuple[int, int]],
    closure_reachable_masks: frozenset[int] | None,
    color_order_reachable_masks: frozenset[int] | None,
    *,
    max_coupling_orders: Mapping[str, int],
    ignored_particle_ids: frozenset[int],
    ignored_vertex_kinds: frozenset[int],
) -> UsefulStateMap:
    """Return current states that can feed at least one amplitude closure.

    The full current table is keyed by helicity, chirality, flavour flow and
    colour state.  This prepass intentionally ignores those expensive labels,
    but it keeps particle id and model coupling-order totals.  The order
    tracking lets user-supplied generic constraints such as ``QED=1`` prune
    impossible branches before the expensive helicity/current sweep, without
    recognizing any process family.  It is allowed to overestimate, but it must
    never depend on a process-family name.
    """

    full_mask = _labels_mask(leg.label for leg in process_ir.legs)
    possible: dict[int, dict[int, set[CouplingOrders]]] = {}
    for leg in process_ir.legs:
        if leg.outgoing_pdg is None:
            continue
        particle_id = int(leg.outgoing_pdg)
        if particle_id in ignored_particle_ids:
            continue
        possible.setdefault(1 << (leg.label - 1), {}).setdefault(
            particle_id,
            set(),
        ).add(())

    reverse_transitions: dict[
        _ReachabilityState,
        set[tuple[_ReachabilityState, _ReachabilityState]],
    ] = {}
    vertices_by_input: dict[tuple[int, int], tuple[Vertex, ...]] = {}
    right_particles_by_left = _right_particles_by_left(
        model,
        color_accuracy=process_ir.color_accuracy,
    )
    for mask in _masks_by_size(full_mask):
        if mask & (mask - 1) == 0 or mask == full_mask:
            continue
        if not _mask_allowed_by_reachability(
            mask,
            closure_reachable_masks,
            color_order_reachable_masks,
        ):
            continue
        for left_mask, right_mask in _ordered_splits(mask):
            if not (
                _mask_allowed_by_reachability(
                    left_mask,
                    closure_reachable_masks,
                    color_order_reachable_masks,
                )
                and _mask_allowed_by_reachability(
                    right_mask,
                    closure_reachable_masks,
                    color_order_reachable_masks,
                )
            ):
                continue
            left_species = possible.get(left_mask)
            right_species = possible.get(right_mask)
            if not left_species or not right_species:
                continue
            for left_particle, left_orders_set in tuple(left_species.items()):
                for right_particle in right_particles_by_left.get(left_particle, ()):
                    right_orders_set = right_species.get(right_particle)
                    if not right_orders_set:
                        continue
                    vertex_key = (left_particle, right_particle)
                    vertices = vertices_by_input.get(vertex_key)
                    if vertices is None:
                        vertices = model.vertices_accepting(
                            left_particle,
                            right_particle,
                            color_accuracy=process_ir.color_accuracy,
                        )
                        vertices_by_input[vertex_key] = vertices
                    for vertex in vertices:
                        if (
                            vertex.kind in ignored_vertex_kinds
                            or vertex.particles[2] in ignored_particle_ids
                            or not color_engine.vertex_allowed(vertex)
                            or model.skip_duplicate_vertex_orientation(vertex)
                        ):
                            continue
                        result_particle = vertex.particles[2]
                        for left_orders in tuple(left_orders_set):
                            for right_orders in tuple(right_orders_set):
                                coupling_orders = _combine_coupling_order_tuples(
                                    left_orders,
                                    right_orders,
                                    model.vertex_coupling_orders(vertex),
                                )
                                if not _coupling_orders_within_limits(
                                    coupling_orders,
                                    max_coupling_orders,
                                ):
                                    continue
                                possible.setdefault(mask, {}).setdefault(
                                    result_particle,
                                    set(),
                                ).add(coupling_orders)
                                reverse_transitions.setdefault(
                                    (mask, result_particle, coupling_orders),
                                    set(),
                                ).add(
                                    (
                                        (left_mask, left_particle, left_orders),
                                        (right_mask, right_particle, right_orders),
                                    )
                                )

    closure_right_particles_by_left = _closure_right_particles_by_left(
        model,
        color_accuracy=process_ir.color_accuracy,
    )
    useful: dict[int, dict[int, set[CouplingOrders]]] = {}
    for left_mask, right_mask in closure_candidate_splits:
        left_species = possible.get(left_mask)
        right_species = possible.get(right_mask)
        if not left_species or not right_species:
            continue
        for left_particle, left_orders_set in left_species.items():
            for right_particle in closure_right_particles_by_left.get(
                left_particle,
                (),
            ):
                right_orders_set = right_species.get(right_particle)
                if not right_orders_set:
                    continue
                if model.direct_contraction_possible(left_particle, right_particle):
                    for left_orders in left_orders_set:
                        for right_orders in right_orders_set:
                            total_orders = _combine_coupling_order_tuples(
                                left_orders,
                                right_orders,
                                (),
                            )
                            if not _coupling_orders_within_limits(
                                total_orders,
                                max_coupling_orders,
                            ):
                                continue
                            useful.setdefault(left_mask, {}).setdefault(
                                left_particle,
                                set(),
                            ).add(left_orders)
                            useful.setdefault(right_mask, {}).setdefault(
                                right_particle,
                                set(),
                            ).add(right_orders)
                for vertex in model.vertices_accepting(
                    left_particle,
                    right_particle,
                    color_accuracy=process_ir.color_accuracy,
                ):
                    if (
                        vertex.kind in ignored_vertex_kinds
                        or vertex.particles[2] in ignored_particle_ids
                        or not color_engine.vertex_allowed(vertex)
                        or model.skip_duplicate_vertex_orientation(vertex)
                    ):
                        continue
                    if model.closure_contraction_ir(vertex.particles[2]) is None:
                        continue
                    vertex_orders = model.vertex_coupling_orders(vertex)
                    for left_orders in left_orders_set:
                        for right_orders in right_orders_set:
                            total_orders = _combine_coupling_order_tuples(
                                left_orders,
                                right_orders,
                                vertex_orders,
                            )
                            if not _coupling_orders_within_limits(
                                total_orders,
                                max_coupling_orders,
                            ):
                                continue
                            useful.setdefault(left_mask, {}).setdefault(
                                left_particle,
                                set(),
                            ).add(left_orders)
                            useful.setdefault(right_mask, {}).setdefault(
                                right_particle,
                                set(),
                            ).add(right_orders)

    pending = deque(
        (mask, particle, orders)
        for mask, species_orders in useful.items()
        for particle, order_set in species_orders.items()
        for orders in order_set
    )
    while pending:
        result_state = pending.popleft()
        for left_state, right_state in reverse_transitions.get(result_state, ()):
            for parent_mask, parent_particle, parent_orders in (
                left_state,
                right_state,
            ):
                parent_useful = useful.setdefault(parent_mask, {}).setdefault(
                    parent_particle,
                    set(),
                )
                if parent_orders in parent_useful:
                    continue
                parent_useful.add(parent_orders)
                pending.append((parent_mask, parent_particle, parent_orders))

    return {
        mask: {
            particle: frozenset(orders)
            for particle, orders in species_orders.items()
            if orders
        }
        for mask, species_orders in useful.items()
        if species_orders
    }


def _closure_total_coupling_orders(
    process_ir: CanonicalProcessIR,
    model: Model,
    color_engine: ColorEngine,
    closure_candidate_splits: Iterable[tuple[int, int]],
    closure_reachable_masks: frozenset[int] | None,
    color_order_reachable_masks: frozenset[int] | None,
    *,
    max_coupling_orders: Mapping[str, int],
    ignored_particle_ids: frozenset[int],
    ignored_vertex_kinds: frozenset[int],
) -> frozenset[CouplingOrders]:
    """Return model-reachable total coupling orders for amplitude closures."""

    full_mask = _labels_mask(leg.label for leg in process_ir.legs)
    possible: dict[int, dict[int, set[CouplingOrders]]] = {}
    for leg in process_ir.legs:
        if leg.outgoing_pdg is None:
            continue
        particle_id = int(leg.outgoing_pdg)
        if particle_id in ignored_particle_ids:
            continue
        possible.setdefault(1 << (leg.label - 1), {}).setdefault(
            particle_id,
            set(),
        ).add(())

    vertices_by_input: dict[tuple[int, int], tuple[Vertex, ...]] = {}
    right_particles_by_left = _right_particles_by_left(
        model,
        color_accuracy=process_ir.color_accuracy,
    )
    for mask in _masks_by_size(full_mask):
        if mask & (mask - 1) == 0 or mask == full_mask:
            continue
        if not _mask_allowed_by_reachability(
            mask,
            closure_reachable_masks,
            color_order_reachable_masks,
        ):
            continue
        for left_mask, right_mask in _ordered_splits(mask):
            if not (
                _mask_allowed_by_reachability(
                    left_mask,
                    closure_reachable_masks,
                    color_order_reachable_masks,
                )
                and _mask_allowed_by_reachability(
                    right_mask,
                    closure_reachable_masks,
                    color_order_reachable_masks,
                )
            ):
                continue
            left_species = possible.get(left_mask)
            right_species = possible.get(right_mask)
            if not left_species or not right_species:
                continue
            for left_particle, left_orders_set in tuple(left_species.items()):
                for right_particle in right_particles_by_left.get(left_particle, ()):
                    right_orders_set = right_species.get(right_particle)
                    if not right_orders_set:
                        continue
                    vertex_key = (left_particle, right_particle)
                    vertices = vertices_by_input.get(vertex_key)
                    if vertices is None:
                        vertices = model.vertices_accepting(
                            left_particle,
                            right_particle,
                            color_accuracy=process_ir.color_accuracy,
                        )
                        vertices_by_input[vertex_key] = vertices
                    for vertex in vertices:
                        if (
                            vertex.kind in ignored_vertex_kinds
                            or vertex.particles[2] in ignored_particle_ids
                            or not color_engine.vertex_allowed(vertex)
                            or model.skip_duplicate_vertex_orientation(vertex)
                        ):
                            continue
                        result_particle = vertex.particles[2]
                        order_bucket = possible.setdefault(mask, {}).setdefault(
                            result_particle,
                            set(),
                        )
                        vertex_orders = model.vertex_coupling_orders(vertex)
                        for left_orders in tuple(left_orders_set):
                            for right_orders in tuple(right_orders_set):
                                coupling_orders = _combine_coupling_order_tuples(
                                    left_orders,
                                    right_orders,
                                    vertex_orders,
                                )
                                if not _coupling_orders_within_limits(
                                    coupling_orders,
                                    max_coupling_orders,
                                ):
                                    continue
                                order_bucket.add(coupling_orders)
                        possible[mask][result_particle] = set(
                            _pareto_minimal_coupling_orders(order_bucket)
                        )

    closure_right_particles_by_left = _closure_right_particles_by_left(
        model,
        color_accuracy=process_ir.color_accuracy,
    )
    totals: set[CouplingOrders] = set()
    for left_mask, right_mask in closure_candidate_splits:
        left_species = possible.get(left_mask)
        right_species = possible.get(right_mask)
        if not left_species or not right_species:
            continue
        for left_particle, left_orders_set in left_species.items():
            for right_particle in closure_right_particles_by_left.get(
                left_particle,
                (),
            ):
                right_orders_set = right_species.get(right_particle)
                if not right_orders_set:
                    continue
                if model.direct_contraction_possible(left_particle, right_particle):
                    for left_orders in left_orders_set:
                        for right_orders in right_orders_set:
                            total_orders = _combine_coupling_order_tuples(
                                left_orders,
                                right_orders,
                                (),
                            )
                            if _coupling_orders_within_limits(
                                total_orders,
                                max_coupling_orders,
                            ):
                                totals.add(total_orders)
                for vertex in model.vertices_accepting(
                    left_particle,
                    right_particle,
                    color_accuracy=process_ir.color_accuracy,
                ):
                    if (
                        vertex.kind in ignored_vertex_kinds
                        or vertex.particles[2] in ignored_particle_ids
                        or not color_engine.vertex_allowed(vertex)
                        or model.skip_duplicate_vertex_orientation(vertex)
                    ):
                        continue
                    if model.closure_contraction_ir(vertex.particles[2]) is None:
                        continue
                    vertex_orders = model.vertex_coupling_orders(vertex)
                    for left_orders in left_orders_set:
                        for right_orders in right_orders_set:
                            total_orders = _combine_coupling_order_tuples(
                                left_orders,
                                right_orders,
                                vertex_orders,
                            )
                            if _coupling_orders_within_limits(
                                total_orders,
                                max_coupling_orders,
                            ):
                                totals.add(total_orders)
    return frozenset(_pareto_minimal_coupling_orders(totals))


def _coupling_order_degree(
    orders: CouplingOrders,
    *,
    hierarchies: Mapping[str, int] | None = None,
) -> int:
    priorities = hierarchies or {}
    return sum(
        int(value) * max(1, int(priorities.get(str(name).upper(), 1)))
        for name, value in orders
    )


def _coupling_order_envelope(orders: Iterable[CouplingOrders]) -> dict[str, int]:
    envelope: dict[str, int] = {}
    for order_tuple in orders:
        for name, value in order_tuple:
            normalized = str(name).upper()
            envelope[normalized] = max(envelope.get(normalized, 0), int(value))
    return envelope


def _pareto_minimal_coupling_orders(
    orders: Iterable[CouplingOrders],
) -> tuple[CouplingOrders, ...]:
    normalized = tuple(sorted(set(orders)))
    minimal: list[CouplingOrders] = []
    for candidate in normalized:
        if any(
            _coupling_orders_dominate(other, candidate)
            for other in normalized
            if other != candidate
        ):
            continue
        minimal.append(candidate)
    return tuple(minimal)


def _coupling_orders_dominate(left: CouplingOrders, right: CouplingOrders) -> bool:
    left_map = dict(left)
    right_map = dict(right)
    keys = set(left_map) | set(right_map)
    return all(left_map.get(key, 0) <= right_map.get(key, 0) for key in keys) and any(
        left_map.get(key, 0) < right_map.get(key, 0) for key in keys
    )


def _mask_allowed_by_reachability(
    mask: int,
    closure_reachable_masks: frozenset[int] | None,
    color_order_reachable_masks: frozenset[int] | None,
) -> bool:
    if closure_reachable_masks is not None and mask not in closure_reachable_masks:
        return False
    return not (
        color_order_reachable_masks is not None
        and mask not in color_order_reachable_masks
    )


def _state_allowed_by_reachability(
    useful_states_by_mask: UsefulStateMap,
    mask: int,
    particle_id: int,
    coupling_orders: CouplingOrders,
) -> bool:
    return coupling_orders in useful_states_by_mask.get(mask, {}).get(
        particle_id,
        (),
    )


def _nonzero_submasks(mask: int) -> tuple[int, ...]:
    submasks: list[int] = []
    submask = mask
    while submask:
        submasks.append(submask)
        submask = (submask - 1) & mask
    return tuple(submasks)


def _masks_by_size(full_mask: int) -> tuple[int, ...]:
    masks = []
    submask = full_mask
    while submask:
        masks.append(submask)
        submask = (submask - 1) & full_mask
    return tuple(sorted(masks, key=lambda value: (value.bit_count(), value)))


def _ordered_splits(mask: int) -> tuple[tuple[int, int], ...]:
    splits: list[tuple[int, int]] = []
    left = (mask - 1) & mask
    while left:
        right = mask ^ left
        if right:
            splits.append((left, right))
        left = (left - 1) & mask
    return tuple(splits)
