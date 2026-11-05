# SPDX-License-Identifier: 0BSD
"""Post-processing and selection transforms for generic DAGs."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import cast

from ..color.plan import GenericColorPlan, build_color_plan
from ..color.plan_types import _canonical_open_string_product_key
from ..models._physics_ir import ContractionIR
from ..models.base import (
    CouplingOrders,
    Model,
    QuantumFlow,
    QuantumNumberFlow,
    Vertex,
)
from ..processes.ir import CanonicalProcessIR
from .dag_color import ColorEngine
from .dag_ordering import (
    _closure_candidate_splits,
    _closure_side_reachable_masks,
    _complex_weight_mul,
    _direct_contraction_ir,
    _labels_mask,
    _lc_color_order_reachable_masks,
    _mask_labels,
)
from .dag_reachability import (
    UsefulStateMap,
    _closure_total_coupling_orders,
    _combine_coupling_order_tuples,
    _coupling_order_degree,
    _coupling_order_envelope,
    _coupling_orders_within_limits,
    _lc_line_groups_within_limit,
    _mask_allowed_by_reachability,
    _masks_by_size,
    _normalize_coupling_order_limits,
    _ordered_splits,
    _right_particles_by_left,
    _state_allowed_by_reachability,
)
from .dag_types import (
    AmplitudeRoot,
    ColorFlow,
    ColorState,
    CurrentIndex,
    CurrentNode,
    GenericDAG,
    InteractionNode,
)
from .helicity_replay import _semantic_amplitude_root_alias_signatures


def _normalize_generation_cap(value: int | None) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    return None if normalized < 0 else normalized


@dataclass(frozen=True, slots=True)
class _LiveCurrentShape:
    external_mask: int
    external_labels: tuple[int, ...]
    particle_id: int
    ordered_external_labels: tuple[int, ...]
    color_state: ColorState
    coupling_orders: CouplingOrders
    chirality: int
    spin_state: int | tuple[int, ...]
    flavour_flow: tuple[int, ...]
    quantum_number_flow: QuantumNumberFlow
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_hash",
            hash(
                (
                    self.external_mask,
                    self.external_labels,
                    self.particle_id,
                    self.ordered_external_labels,
                    self.color_state,
                    self.coupling_orders,
                    self.chirality,
                    self.spin_state,
                    self.flavour_flow,
                    self.quantum_number_flow,
                )
            ),
        )

    def __hash__(self) -> int:
        return self._hash


@dataclass(frozen=True, slots=True)
class BackwardLiveTransition:
    """One model-certified abstract current invocation retained by liveness."""

    left: _LiveCurrentShape
    right: _LiveCurrentShape
    result: _LiveCurrentShape
    vertex_kind: int
    vertex_particles: tuple[int, int, int]
    coupling: tuple[float, float]
    color_weight: tuple[float, float]


@dataclass(frozen=True, slots=True)
class BackwardLiveClosure:
    """One model-certified abstract amplitude closure."""

    left: _LiveCurrentShape
    right: _LiveCurrentShape
    kind: str
    color_weight: tuple[float, float]
    contraction_ir: ContractionIR
    color_sector_id: int | None
    vertex_kind: int | None = None
    vertex_particles: tuple[int, int, int] | None = None
    coupling: tuple[float, float] = (1.0, 0.0)


@dataclass(frozen=True, slots=True)
class BackwardLiveSource:
    """One retained source with its original ancestry-bit allocation."""

    shape: _LiveCurrentShape
    helicity_ancestry: int
    leg_label: int
    source_helicity: int


@dataclass(frozen=True, slots=True)
class BackwardLiveStatePlan:
    """Conservative exact-colour states that can feed a physical closure.

    The planner tracks model quantum flow but intentionally omits helicity
    ancestry and tensor payloads. A retained shape can therefore still produce
    more than one exact current, while a valid exact current cannot be removed
    solely because of those omitted fields. This makes the plan suitable as an
    eager-only construction filter without introducing a second physics
    implementation.
    """

    shapes: frozenset[_LiveCurrentShape]
    active_sector_ids: frozenset[int]
    transitions: tuple[BackwardLiveTransition, ...]
    closures: tuple[BackwardLiveClosure, ...]
    sources: tuple[BackwardLiveSource, ...]

    def allows(self, index: CurrentIndex) -> bool:
        return _live_current_shape(index) in self.shapes


def _live_current_shape(index: CurrentIndex) -> _LiveCurrentShape:
    return _LiveCurrentShape(
        external_mask=int(index.external_mask),
        external_labels=tuple(index.external_labels),
        particle_id=int(index.particle_id),
        ordered_external_labels=tuple(index.ordered_external_labels),
        color_state=index.color_state,
        coupling_orders=tuple(index.coupling_orders),
        chirality=int(index.chirality),
        spin_state=index.spin_state,
        flavour_flow=tuple(index.flavour_flow),
        quantum_number_flow=index.quantum_number_flow,
    )


def build_backward_live_state_plan(
    process: CanonicalProcessIR,
    *,
    model: Model,
    color_engine: ColorEngine,
    closure_candidate_splits: Sequence[tuple[int, int]],
    closure_reachable_masks: frozenset[int] | None,
    color_order_reachable_masks: frozenset[int] | None,
    useful_states_by_mask: UsefulStateMap | None,
    max_coupling_orders: Mapping[str, int],
    max_lc_current_line_groups: int | None,
    ignored_particle_ids: frozenset[int],
    ignored_vertex_kinds: frozenset[int],
    selected_source_helicities: Mapping[int, int] | None,
    global_flip_anchor: tuple[int, int] | None,
) -> BackwardLiveStatePlan | None:
    """Build an eager-only demand plan from physical amplitude sinks.

    Multi-line colour plans may enumerate many traversal sectors although the
    recursion uses one physical sink.  The forward compiler previously built
    every helicity and quantum-flow current in all of them before discovering
    at closure that most sectors were dead.  This prepass first keeps only
    physical words ending at a configured sink, then performs a lightweight
    model-driven recursion over particle species, coupling orders, ordered
    labels, and exact colour states.  A final backward walk retains only shapes
    feeding a model-supported direct or vertex closure.

    Shared-ordering and single-trace plans already collapse sectors before the
    current sweep.  They deliberately stay on the existing path because this
    planner would not remove work there.
    """

    color_plan = color_engine.color_plan
    if (
        color_plan.color_accuracy == "lc"
        or color_engine.shared_lc_orderings
        or color_engine.shared_single_trace
        or len(color_plan.sectors) <= 1
    ):
        return None

    active_sector_ids = _physical_sink_sector_ids(
        color_plan,
        closure_candidate_splits,
    )
    if active_sector_ids is None:
        return None

    full_mask = _labels_mask(leg.label for leg in process.legs)
    possible_by_mask: dict[int, list[_LiveCurrentShape]] = {}
    possible_seen: set[_LiveCurrentShape] = set()
    reverse_transitions: dict[
        _LiveCurrentShape,
        set[tuple[_LiveCurrentShape, _LiveCurrentShape]],
    ] = {}
    transitions: list[BackwardLiveTransition] = []

    def state_allowed(shape: _LiveCurrentShape) -> bool:
        if useful_states_by_mask is None:
            return True
        return _state_allowed_by_reachability(
            useful_states_by_mask,
            shape.external_mask,
            shape.particle_id,
            shape.coupling_orders,
        )

    def add_shape(shape: _LiveCurrentShape) -> bool:
        if shape in possible_seen or not state_allowed(shape):
            return False
        possible_seen.add(shape)
        possible_by_mask.setdefault(shape.external_mask, []).append(shape)
        return True

    source_candidates: list[BackwardLiveSource] = []
    next_source_bit = 0
    for leg in process.legs:
        if leg.outgoing_pdg is None:
            continue
        particle_id = int(leg.outgoing_pdg)
        if particle_id in ignored_particle_ids:
            continue
        mask = 1 << (leg.label - 1)
        source_ir = model._source_ir(particle_id)
        source_states = tuple(
            source_ir.crossing.apply(state) if leg.is_initial else state
            for state in source_ir.states
        )
        source_quantum_flow = model.quantum_number_flow(particle_id)
        for color_state in color_engine.source_states_for_leg(leg):
            if not _lc_line_groups_within_limit(
                color_state,
                max_lc_current_line_groups,
            ):
                continue
            color_state_is_live = color_state.sector_id in active_sector_ids
            for source_state in source_states:
                source_helicity = int(source_state.helicity)
                if (
                    global_flip_anchor is not None
                    and leg.label == global_flip_anchor[0]
                    and source_helicity != global_flip_anchor[1]
                ):
                    next_source_bit += 1
                    continue
                requested_helicity = (selected_source_helicities or {}).get(
                    int(leg.label)
                )
                if requested_helicity is not None and source_helicity != int(
                    requested_helicity
                ):
                    continue
                helicity_ancestry = 1 << next_source_bit
                next_source_bit += 1
                if not color_state_is_live:
                    continue
                source_shape = _LiveCurrentShape(
                    external_mask=mask,
                    external_labels=(int(leg.label),),
                    particle_id=particle_id,
                    ordered_external_labels=(int(leg.label),),
                    color_state=color_state,
                    coupling_orders=(),
                    chirality=int(source_state.chirality),
                    spin_state=source_state.spin_state,
                    flavour_flow=(particle_id,),
                    quantum_number_flow=source_quantum_flow,
                )
                add_shape(source_shape)
                if source_shape in possible_seen:
                    source_candidates.append(
                        BackwardLiveSource(
                            shape=source_shape,
                            helicity_ancestry=helicity_ancestry,
                            leg_label=int(leg.label),
                            source_helicity=source_helicity,
                        )
                    )

    right_particles_by_left = _right_particles_by_left(
        model,
        color_accuracy=process.color_accuracy,
    )
    vertices_by_input: dict[tuple[int, int], tuple[Vertex, ...]] = {}
    vertex_allowed_cache: dict[Vertex, bool] = {}
    duplicate_orientation_cache: dict[Vertex, bool] = {}
    coupling_orders_cache: dict[Vertex, CouplingOrders] = {}
    vertex_color_weight_cache: dict[Vertex, tuple[float, float]] = {}
    combined_orders_cache: dict[
        tuple[CouplingOrders, CouplingOrders, Vertex],
        CouplingOrders,
    ] = {}
    orders_allowed_cache: dict[CouplingOrders, bool] = {}
    quantum_flow_cache: dict[tuple[object, ...], tuple[QuantumFlow, ...]] = {}
    color_flow_cache: dict[tuple[object, ...], tuple[ColorFlow, ...]] = {}
    for mask in _masks_by_size(full_mask):
        if mask & (mask - 1) == 0 or mask == full_mask:
            continue
        if not _mask_allowed_by_reachability(
            mask,
            closure_reachable_masks,
            color_order_reachable_masks,
        ):
            continue
        if useful_states_by_mask is not None and mask not in useful_states_by_mask:
            continue
        labels = _mask_labels(mask)
        for left_mask, right_mask in _ordered_splits(mask):
            left_shapes = possible_by_mask.get(left_mask)
            right_shapes = possible_by_mask.get(right_mask)
            if not left_shapes or not right_shapes:
                continue
            right_by_sector_particle: dict[
                tuple[int, int], list[_LiveCurrentShape]
            ] = {}
            for right_shape in right_shapes:
                right_by_sector_particle.setdefault(
                    (
                        right_shape.color_state.sector_id,
                        right_shape.particle_id,
                    ),
                    [],
                ).append(right_shape)
            for left_shape in tuple(left_shapes):
                left_index = cast(CurrentIndex, left_shape)
                for right_particle in right_particles_by_left.get(
                    left_shape.particle_id,
                    (),
                ):
                    for right_shape in right_by_sector_particle.get(
                        (left_shape.color_state.sector_id, right_particle),
                        (),
                    ):
                        right_index = cast(CurrentIndex, right_shape)
                        input_key = (
                            left_shape.particle_id,
                            right_shape.particle_id,
                        )
                        vertices = vertices_by_input.get(input_key)
                        if vertices is None:
                            vertices = model.vertices_accepting(
                                *input_key,
                                color_accuracy=process.color_accuracy,
                            )
                            vertices_by_input[input_key] = vertices
                        for vertex in vertices:
                            if (
                                vertex.kind in ignored_vertex_kinds
                                or vertex.particles[2] in ignored_particle_ids
                            ):
                                continue
                            vertex_allowed = vertex_allowed_cache.get(vertex)
                            if vertex_allowed is None:
                                vertex_allowed = color_engine.vertex_allowed(vertex)
                                vertex_allowed_cache[vertex] = vertex_allowed
                            if not vertex_allowed:
                                continue
                            duplicate = duplicate_orientation_cache.get(vertex)
                            if duplicate is None:
                                duplicate = model.skip_duplicate_vertex_orientation(
                                    vertex
                                )
                                duplicate_orientation_cache[vertex] = duplicate
                            if duplicate:
                                continue
                            vertex_orders = coupling_orders_cache.get(vertex)
                            if vertex_orders is None:
                                vertex_orders = model.vertex_coupling_orders(vertex)
                                coupling_orders_cache[vertex] = vertex_orders
                            orders_key = (
                                left_shape.coupling_orders,
                                right_shape.coupling_orders,
                                vertex,
                            )
                            orders = combined_orders_cache.get(orders_key)
                            if orders is None:
                                orders = _combine_coupling_order_tuples(
                                    left_shape.coupling_orders,
                                    right_shape.coupling_orders,
                                    vertex_orders,
                                )
                                combined_orders_cache[orders_key] = orders
                            orders_allowed = orders_allowed_cache.get(orders)
                            if orders_allowed is None:
                                orders_allowed = _coupling_orders_within_limits(
                                    orders,
                                    max_coupling_orders,
                                )
                                orders_allowed_cache[orders] = orders_allowed
                            if not orders_allowed:
                                continue
                            ordered_labels = color_engine.ordered_combination_labels(
                                left_index,
                                right_index,
                                vertex,
                            )
                            if ordered_labels is None:
                                continue
                            quantum_key = (
                                vertex.kind,
                                vertex.particles,
                                left_shape.particle_id,
                                left_shape.chirality,
                                left_shape.spin_state,
                                left_shape.flavour_flow,
                                left_shape.quantum_number_flow,
                                right_shape.particle_id,
                                right_shape.chirality,
                                right_shape.spin_state,
                                right_shape.flavour_flow,
                                right_shape.quantum_number_flow,
                            )
                            quantum_flows = quantum_flow_cache.get(quantum_key)
                            if quantum_flows is None:
                                quantum_flows = tuple(
                                    model.allowed_quantum_flows(
                                        vertex,
                                        left_index,
                                        right_index,
                                    )
                                )
                                quantum_flow_cache[quantum_key] = quantum_flows
                            if not quantum_flows:
                                continue
                            color_key = (
                                left_shape.color_state,
                                right_shape.color_state,
                                vertex,
                                ordered_labels,
                            )
                            color_flows = color_flow_cache.get(color_key)
                            if color_flows is None:
                                color_flows = tuple(
                                    color_engine.combine(
                                        left_shape.color_state,
                                        right_shape.color_state,
                                        vertex,
                                        ordered_external_labels=ordered_labels,
                                    )
                                )
                                color_flow_cache[color_key] = color_flows
                            for color_flow in color_flows:
                                if not _lc_line_groups_within_limit(
                                    color_flow.state,
                                    max_lc_current_line_groups,
                                ):
                                    continue
                                for quantum_flow in quantum_flows:
                                    result_shape = _LiveCurrentShape(
                                        external_mask=mask,
                                        external_labels=labels,
                                        particle_id=int(vertex.particles[2]),
                                        ordered_external_labels=tuple(ordered_labels),
                                        color_state=color_flow.state,
                                        coupling_orders=orders,
                                        chirality=int(quantum_flow.chirality),
                                        spin_state=quantum_flow.spin_state,
                                        flavour_flow=tuple(quantum_flow.flavour_flow),
                                        quantum_number_flow=(
                                            quantum_flow.quantum_number_flow
                                        ),
                                    )
                                    if not model.current_allowed(result_shape):
                                        continue
                                    add_shape(result_shape)
                                    reverse_transitions.setdefault(
                                        result_shape,
                                        set(),
                                    ).add((left_shape, right_shape))
                                    vertex_color_weight = vertex_color_weight_cache.get(
                                        vertex
                                    )
                                    if vertex_color_weight is None:
                                        vertex_color_weight = model.vertex_color_weight(
                                            vertex,
                                            color_accuracy=process.color_accuracy,
                                        )
                                        vertex_color_weight_cache[vertex] = (
                                            vertex_color_weight
                                        )
                                    transitions.append(
                                        BackwardLiveTransition(
                                            left=left_shape,
                                            right=right_shape,
                                            result=result_shape,
                                            vertex_kind=int(vertex.kind),
                                            vertex_particles=tuple(vertex.particles),
                                            coupling=tuple(quantum_flow.coupling),
                                            color_weight=_complex_weight_mul(
                                                color_flow.weight,
                                                vertex_color_weight,
                                            ),
                                        )
                                    )

    closures: list[BackwardLiveClosure] = []
    for left_mask, right_mask in closure_candidate_splits:
        left_shapes = possible_by_mask.get(left_mask, ())
        right_shapes = possible_by_mask.get(right_mask, ())
        if not left_shapes or not right_shapes:
            continue
        right_by_sector: dict[int, list[_LiveCurrentShape]] = {}
        for right_shape in right_shapes:
            right_by_sector.setdefault(
                right_shape.color_state.sector_id,
                [],
            ).append(right_shape)
        for left_shape in left_shapes:
            left_index = cast(CurrentIndex, left_shape)
            for right_shape in right_by_sector.get(
                left_shape.color_state.sector_id,
                (),
            ):
                right_index = cast(CurrentIndex, right_shape)
                if not color_engine.ordered_closure_allowed(left_index, right_index):
                    continue
                color_flows = color_engine.closure_compatible(
                    left_shape.color_state,
                    right_shape.color_state,
                    full_mask=full_mask,
                )
                if not color_flows:
                    continue
                direct_contraction_ir = _direct_contraction_ir(
                    model,
                    left_index,
                    right_index,
                )
                pair_closures: list[BackwardLiveClosure] = []
                for color_flow in color_flows:
                    if direct_contraction_ir is not None:
                        pair_closures.append(
                            BackwardLiveClosure(
                                left=left_shape,
                                right=right_shape,
                                kind="direct-contraction",
                                color_weight=color_flow.weight,
                                contraction_ir=direct_contraction_ir,
                                color_sector_id=color_flow.state.sector_id,
                            )
                        )
                    for vertex in model.vertices_accepting(
                        left_shape.particle_id,
                        right_shape.particle_id,
                        color_accuracy=process.color_accuracy,
                    ):
                        if (
                            vertex.kind in ignored_vertex_kinds
                            or vertex.particles[2] in ignored_particle_ids
                            or not color_engine.vertex_allowed(vertex)
                            or not model.vertex_closure_allowed(vertex)
                        ):
                            continue
                        closure_contraction_ir = model.closure_contraction_ir(
                            vertex.particles[2]
                        )
                        if closure_contraction_ir is None:
                            continue
                        total_orders = model.combine_coupling_orders(
                            left_index,
                            right_index,
                            vertex,
                        )
                        if not _coupling_orders_within_limits(
                            total_orders,
                            max_coupling_orders,
                        ) or not model.allowed_quantum_flows(
                            vertex,
                            left_index,
                            right_index,
                        ):
                            continue
                        pair_closures.append(
                            BackwardLiveClosure(
                                left=left_shape,
                                right=right_shape,
                                kind="vertex-closure",
                                color_weight=_complex_weight_mul(
                                    color_flow.weight,
                                    model.vertex_color_weight(
                                        vertex,
                                        color_accuracy=process.color_accuracy,
                                    ),
                                ),
                                contraction_ir=closure_contraction_ir,
                                color_sector_id=color_flow.state.sector_id,
                                vertex_kind=vertex.kind,
                                vertex_particles=vertex.particles,
                                coupling=vertex.coupling,
                            )
                        )
                if pair_closures:
                    closures.extend(pair_closures)

    # Keep the union of every model-valid closure traversal.  Whole-sector
    # ownership is not sound for three or more open fermion lines: different
    # block traversals can expose distinct closure terms (and identical-line
    # exchange signs).  Exact semantic alias removal happens only after the
    # concrete DAG exists and can be certified by the shared replay proof.
    useful: set[_LiveCurrentShape] = {
        shape for closure in closures for shape in (closure.left, closure.right)
    }

    pending = deque(useful)
    while pending:
        result_shape = pending.popleft()
        for left_shape, right_shape in reverse_transitions.get(result_shape, ()):
            for parent in (left_shape, right_shape):
                if parent in useful:
                    continue
                useful.add(parent)
                pending.append(parent)

    if not useful:
        return None
    return BackwardLiveStatePlan(
        shapes=frozenset(useful),
        active_sector_ids=frozenset(shape.color_state.sector_id for shape in useful),
        transitions=tuple(
            transition
            for transition in transitions
            if transition.result in useful
            and transition.left in useful
            and transition.right in useful
        ),
        closures=tuple(closures),
        sources=tuple(source for source in source_candidates if source.shape in useful),
    )


def _physical_sink_sector_ids(
    color_plan: GenericColorPlan,
    closure_candidate_splits: Sequence[tuple[int, int]],
) -> frozenset[int] | None:
    """Return sectors whose canonical traversal ends at a closure sink.

    Compatibility traversals exist only to build intermediate currents.  The
    sector's authenticated canonical traversal owns the private recursion sink.
    Fail closed when a sector has no canonical sink or when a closure side is
    not a singleton.
    """

    sink_labels: set[int] = set()
    for _left_mask, right_mask in closure_candidate_splits:
        labels = _mask_labels(right_mask)
        if len(labels) != 1:
            return None
        sink_labels.add(labels[0])
    if not sink_labels:
        return None

    active: set[int] = set()
    for sector in color_plan.sectors:
        sink_label = sector.canonical_closure_sink_label(color_plan.process)
        if sink_label is None:
            return None
        if int(sink_label) in sink_labels:
            active.add(int(sector.id))
    if not active:
        return None
    return frozenset(active)


def _canonical_open_line_alias_owner_sector_ids(
    color_plan: GenericColorPlan,
    active_sector_ids: Iterable[int],
) -> frozenset[int]:
    """Choose one root-bearing owner per complete-open-line tensor.

    NLC and full-colour plans enumerate permutations of complete open-line
    blocks as public traversal sectors.  Those permutations are aliases of
    the same product of colour tensors and must not each contribute a coherent
    amplitude.  Choose the smallest *active* sector in each alias class: for
    three or more open lines the smallest public sector can be rootless, so an
    owner chosen before physical closure would incorrectly erase an amplitude.
    """

    active = frozenset(int(sector_id) for sector_id in active_sector_ids)
    sector_by_id = {int(sector.id): sector for sector in color_plan.sectors}
    unknown = active.difference(sector_by_id)
    if unknown:
        raise ValueError(
            "active colour sectors are absent from the color plan: "
            + ", ".join(str(sector_id) for sector_id in sorted(unknown))
        )
    if color_plan.color_accuracy not in {"nlc", "full"}:
        return active

    owner_by_key: dict[tuple[object, ...], int] = {}
    owners: set[int] = set()
    for sector_id in sorted(active):
        sector = sector_by_id[sector_id]
        if sector.kind == "open-lines":
            open_strings = _canonical_open_string_product_key(
                (
                    line.fundamental_label,
                    line.adjoint_labels,
                    line.antifundamental_label,
                    line.singlet_labels,
                )
                for line in sector.open_color_lines
            )
            key: tuple[object, ...] = ("open-lines", open_strings)
        else:
            key = (sector.kind, sector_id)
        owners.add(owner_by_key.setdefault(key, sector_id))
    return frozenset(owners)


def _canonical_open_line_alias_owner_sector_map(
    color_plan: GenericColorPlan,
) -> dict[int, int]:
    """Map traversal sectors onto canonical complete-open-line tensors."""

    if color_plan.color_accuracy not in {"nlc", "full"}:
        return {int(sector.id): int(sector.id) for sector in color_plan.sectors}
    sectors_by_key: dict[tuple[object, ...], list[int]] = {}
    for sector in color_plan.sectors:
        sector_id = int(sector.id)
        if sector.kind == "open-lines":
            open_strings = _canonical_open_string_product_key(
                (
                    line.fundamental_label,
                    line.adjoint_labels,
                    line.antifundamental_label,
                    line.singlet_labels,
                )
                for line in sector.open_color_lines
            )
            key: tuple[object, ...] = ("open-lines", open_strings)
        else:
            key = (sector.kind, sector_id)
        sectors_by_key.setdefault(key, []).append(sector_id)
    return {
        sector_id: min(sector_ids)
        for sector_ids in sectors_by_key.values()
        for sector_id in sector_ids
    }


def _retain_canonical_open_line_alias_roots(
    dag: GenericDAG,
    model: Model,
) -> GenericDAG:
    """Collapse only model-certified traversal aliases of amplitude roots.

    Public NLC/full plans retain every colour sector.  Complete open-line block
    permutations are private construction aliases, but a sector is not an
    indivisible ownership unit: with three or more open lines, separate
    traversals carry separate physical closure terms.  The shared helicity
    replay signature proves equality of the full recursive closure while a
    sector projection removes only that traversal identifier.  Distinct source
    labels, ordered traversals, fermion exchange factors and kernel contracts
    therefore remain distinct.
    """

    roots = dag.amplitude_roots
    color_plan = dag.color_plan
    if not roots or color_plan.color_accuracy not in {"nlc", "full"}:
        return dag

    owner_by_sector = _canonical_open_line_alias_owner_sector_map(color_plan)
    aliases_by_owner: dict[int, set[int]] = {}
    for sector_id, owner_id in owner_by_sector.items():
        aliases_by_owner.setdefault(owner_id, set()).add(sector_id)
    if all(len(sector_ids) == 1 for sector_ids in aliases_by_owner.values()):
        return dag

    signatures = _semantic_amplitude_root_alias_signatures(
        dag,
        model,
        color_sector_aliases=owner_by_sector,
    )
    grouped: dict[
        tuple[
            int,
            str,
            tuple[tuple[tuple[int, int], ...], ...],
            tuple[float, float],
        ],
        dict[int, list[AmplitudeRoot]],
    ] = {}
    untouched: list[AmplitudeRoot] = []
    for root, signature in zip(roots, signatures, strict=True):
        if root.color_sector_id is None:
            untouched.append(root)
            continue
        sector_id = int(root.color_sector_id)
        owner_id = owner_by_sector.get(sector_id, sector_id)
        if len(aliases_by_owner.get(owner_id, {sector_id})) == 1:
            untouched.append(root)
            continue
        if signature is None:
            raise ValueError(
                "open-line traversal alias has no model-certified recursive "
                f"closure signature: root {root.id}, sector {sector_id}"
            )
        key = (
            owner_id,
            signature.proof_digest,
            signature.selector_states,
            signature.factor,
        )
        grouped.setdefault(key, {}).setdefault(sector_id, []).append(root)

    retained = list(untouched)
    for key in sorted(grouped):
        roots_by_sector = grouped[key]
        multiplicities = {
            sector_id: len(sector_roots)
            for sector_id, sector_roots in roots_by_sector.items()
        }
        if len(set(multiplicities.values())) != 1:
            detail = ", ".join(
                f"sector {sector_id}: {count}"
                for sector_id, count in sorted(multiplicities.items())
            )
            raise ValueError(
                "proof-identical open-line traversal aliases have unequal "
                f"root multiplicities for owner {key[0]} ({detail})"
            )
        representative_sector = min(roots_by_sector)
        owner_id = key[0]
        retained.extend(
            replace(root, color_sector_id=owner_id)
            for root in roots_by_sector[representative_sector]
        )
    retained.sort(key=lambda root: root.id)
    rebound = tuple(
        replace(root, id=index) if root.id != index else root
        for index, root in enumerate(retained)
    )
    return dag if rebound == roots else replace(dag, amplitude_roots=rebound)


def infer_minimal_coupling_order_limits(
    process: CanonicalProcessIR,
    *,
    model: Model,
    max_color_sectors: int | None = None,
    selected_color_sector_ids: Iterable[int] | None = None,
    max_coupling_orders: Mapping[str, int] | None = None,
    closure_side_mask_pruning: bool = True,
    color_order_mask_pruning: bool = True,
    ignored_particle_ids: Iterable[int] | None = None,
    ignored_vertex_kinds: Iterable[int] | None = None,
) -> dict[str, int]:
    """Infer a generic lowest-order coupling envelope for a process.

    This is an opt-in generation accelerator.  It never recognizes a whole
    process family; it asks the model which local vertices can connect the
    external states, tracks UFO-style coupling orders, and returns the
    component-wise maximum over all closure paths with the lowest
    model-declared hierarchy-weighted order. The returned dictionary can be
    used as ordinary ``max_coupling_orders``. Model-declared orders absent from
    the minimal envelope are returned with a zero limit because an omitted
    limit means unrestricted to the DAG compiler.
    """

    if not isinstance(process, CanonicalProcessIR):
        raise TypeError(
            "coupling-order inference requires a model-resolved CanonicalProcessIR"
        )
    process_ir = process
    color_plan = build_color_plan(
        process_ir,
        color_accuracy=process_ir.color_accuracy,
        max_sectors=max_color_sectors,
        fold_trace_reflections=(
            model.lc_trace_reflection_equivalence_is_proven(process_ir)
        ),
    )
    return _infer_minimal_coupling_order_limits_from_color_plan(
        process_ir,
        model=model,
        color_plan=color_plan,
        selected_color_sector_ids=selected_color_sector_ids,
        max_coupling_orders=max_coupling_orders,
        closure_side_mask_pruning=closure_side_mask_pruning,
        color_order_mask_pruning=color_order_mask_pruning,
        ignored_particle_ids=ignored_particle_ids,
        ignored_vertex_kinds=ignored_vertex_kinds,
    )


def _infer_minimal_coupling_order_limits_from_color_plan(
    process: CanonicalProcessIR,
    *,
    model: Model,
    color_plan: GenericColorPlan,
    selected_color_sector_ids: Iterable[int] | None = None,
    max_coupling_orders: Mapping[str, int] | None = None,
    closure_side_mask_pruning: bool = True,
    color_order_mask_pruning: bool = True,
    ignored_particle_ids: Iterable[int] | None = None,
    ignored_vertex_kinds: Iterable[int] | None = None,
) -> dict[str, int]:
    """Infer the minimal coupling envelope from an already complete color plan."""

    if not isinstance(process, CanonicalProcessIR):
        raise TypeError(
            "coupling-order inference requires a model-resolved CanonicalProcessIR"
        )
    process_ir = process
    if color_plan.process is not process_ir and color_plan.process != process_ir:
        raise ValueError(
            "coupling-order inference color plan belongs to another process"
        )
    explicit_sector_ids = (
        None
        if selected_color_sector_ids is None
        else frozenset(int(sector_id) for sector_id in selected_color_sector_ids)
    )
    if explicit_sector_ids is not None and color_plan.color_accuracy == "lc":
        color_plan = GenericColorPlan(
            process=color_plan.process,
            color_accuracy=color_plan.color_accuracy,
            sectors=tuple(
                sector
                for sector in color_plan.sectors
                if sector.id in explicit_sector_ids
            ),
            diagnostics=color_plan.diagnostics,
            truncated=color_plan.truncated,
            trace_reflections_folded=color_plan.trace_reflections_folded,
            basis=color_plan.basis,
        )
    totals = _closure_totals_from_color_plan(
        process_ir,
        model=model,
        color_plan=color_plan,
        max_coupling_orders=max_coupling_orders,
        closure_side_mask_pruning=closure_side_mask_pruning,
        color_order_mask_pruning=color_order_mask_pruning,
        ignored_particle_ids=ignored_particle_ids,
        ignored_vertex_kinds=ignored_vertex_kinds,
    )
    if not totals:
        return {}
    hierarchies = {
        str(name).upper(): max(1, int(value))
        for name, value in model.coupling_order_hierarchies().items()
    }
    minimum_total_order = min(
        _coupling_order_degree(total, hierarchies=hierarchies) for total in totals
    )
    minimal_totals = tuple(
        total
        for total in totals
        if _coupling_order_degree(total, hierarchies=hierarchies) == minimum_total_order
    )
    envelope = _coupling_order_envelope(minimal_totals)
    order_names = set(hierarchies) | set(envelope)
    for total in totals:
        order_names.update(name for name, _value in total)
    return {name: envelope.get(name, 0) for name in sorted(order_names)}


def _closure_totals_from_color_plan(
    process_ir: CanonicalProcessIR,
    *,
    model: Model,
    color_plan: GenericColorPlan,
    max_coupling_orders: Mapping[str, int] | None = None,
    closure_side_mask_pruning: bool = True,
    color_order_mask_pruning: bool = True,
    ignored_particle_ids: Iterable[int] | None = None,
    ignored_vertex_kinds: Iterable[int] | None = None,
) -> tuple[CouplingOrders, ...]:
    """Coupling-order totals of the closures reachable within the limits."""

    color_engine = ColorEngine(color_plan, model)
    full_mask = _labels_mask(leg.label for leg in process_ir.legs)
    closure_candidate_splits = _closure_candidate_splits(
        process_ir,
        model,
        color_engine,
    )
    closure_reachable_masks = (
        _closure_side_reachable_masks(full_mask, closure_candidate_splits)
        if closure_side_mask_pruning
        else None
    )
    color_order_reachable_masks = (
        _lc_color_order_reachable_masks(process_ir, color_plan, model)
        if color_order_mask_pruning
        else None
    )
    return tuple(
        _closure_total_coupling_orders(
            process_ir,
            model,
            color_engine,
            closure_candidate_splits,
            closure_reachable_masks,
            color_order_reachable_masks,
            max_coupling_orders=_normalize_coupling_order_limits(max_coupling_orders),
            ignored_particle_ids=frozenset(
                int(particle_id) for particle_id in (ignored_particle_ids or ())
            ),
            ignored_vertex_kinds=frozenset(
                int(kind) for kind in (ignored_vertex_kinds or ())
            ),
        )
    )


def has_model_supported_amplitudes(
    process: CanonicalProcessIR,
    *,
    model: Model,
    color_plan: GenericColorPlan,
    max_coupling_orders: Mapping[str, int] | None = None,
) -> bool:
    """Whether any tree-level closure of the process exists in the model.

    Species, colour and coupling-order reachability only; it may overestimate,
    like the DAG pruning it shares, but never rejects a process with amplitudes.
    """

    return bool(
        _closure_totals_from_color_plan(
            process,
            model=model,
            color_plan=color_plan,
            max_coupling_orders=max_coupling_orders,
        )
    )


def prune_dag_to_amplitude_roots(dag: GenericDAG) -> GenericDAG:
    """Drop currents and interactions that cannot feed any amplitude root.

    The forward generic sweep intentionally over-generates valid local currents:
    it does not know which of them will survive closure until the full table is
    built.  Production artifacts should mirror AmpliCol's optimized library
    structure and keep only the backward-reachable sub-DAG feeding retained
    amplitude roots.
    """

    if not dag.amplitude_roots:
        return dag

    required_currents = bytearray(len(dag.currents))
    for root in dag.amplitude_roots:
        for current_id in (root.left_id, root.right_id):
            required_currents[current_id] = 1

    required_interactions = bytearray(len(dag.interactions))
    # Interactions are emitted in increasing result-subset order. Walking that
    # table backwards visits every consumer before the interactions producing
    # its parents and computes the complete live closure in one pass.
    for interaction in reversed(dag.interactions):
        if required_currents[interaction.result_id]:
            required_interactions[interaction.id] = 1
            required_currents[interaction.left_id] = 1
            required_currents[interaction.right_id] = 1

    if all(required_currents) and all(required_interactions):
        return dag

    current_id_map = [-1] * len(dag.currents)
    next_current_id = 0
    for old_id, required in enumerate(required_currents):
        if required:
            current_id_map[old_id] = next_current_id
            next_current_id += 1
    pruned_currents = tuple(
        CurrentNode(
            id=current_id_map[current.id],
            index=current.index,
            dimension=current.dimension,
            is_source=current.is_source,
            source_leg_label=current.source_leg_label,
            source_helicity=current.source_helicity,
        )
        for current in dag.currents
        if required_currents[current.id]
    )
    interaction_id_map = [-1] * len(dag.interactions)
    next_interaction_id = 0
    for old_id, required in enumerate(required_interactions):
        if required:
            interaction_id_map[old_id] = next_interaction_id
            next_interaction_id += 1
    pruned_interactions = tuple(
        InteractionNode(
            id=interaction_id_map[interaction.id],
            vertex_kind=interaction.vertex_kind,
            vertex_particles=interaction.vertex_particles,
            left_id=current_id_map[interaction.left_id],
            right_id=current_id_map[interaction.right_id],
            result_id=current_id_map[interaction.result_id],
            coupling=interaction.coupling,
            color_weight=interaction.color_weight,
            lowering_backend=interaction.lowering_backend,
            full_tensor_network_ready=interaction.full_tensor_network_ready,
            evaluation_group_id=interaction.evaluation_group_id,
            evaluation_factor=interaction.evaluation_factor,
        )
        for interaction in dag.interactions
        if required_interactions[interaction.id]
    )
    pruned_roots = tuple(
        AmplitudeRoot(
            id=new_id,
            kind=root.kind,
            left_id=current_id_map[root.left_id],
            right_id=current_id_map[root.right_id],
            color_weight=root.color_weight,
            contraction_ir=root.contraction_ir,
            color_sector_id=root.color_sector_id,
            vertex_kind=root.vertex_kind,
            vertex_particles=root.vertex_particles,
            coupling=root.coupling,
            helicity_weight=root.helicity_weight,
        )
        for new_id, root in enumerate(dag.amplitude_roots)
    )
    pruned_sources = tuple(
        current_id_map[source_id]
        for source_id in dag.sources
        if required_currents[source_id]
    )
    return GenericDAG(
        process=dag.process,
        color_plan=dag.color_plan,
        currents=pruned_currents,
        sources=pruned_sources,
        interactions=pruned_interactions,
        amplitude_roots=pruned_roots,
        truncated=dag.truncated,
        helicity_coverage=dag.helicity_coverage,
        color_coverage=dag.color_coverage,
        selected_source_helicities=dag.selected_source_helicities,
        selected_color_sector_ids=dag.selected_color_sector_ids,
        lc_topology_replay=dag.lc_topology_replay,
        color_topology_replay=dag.color_topology_replay,
    )


def prune_global_helicity_flip_equivalent_roots(
    dag: GenericDAG,
    model: Model,
) -> GenericDAG:
    """Group roots when the model proves a global-helicity-flip identity.

    This is the safe, structural subset of AmpliCol's numerical helicity
    filtering. Keeping one proven representative with doubled helicity weight
    reduces amplitude roots, after which dead-tree pruning removes currents
    that fed only the discarded partner roots.
    """

    parity_safe, vertex_inventory = _global_helicity_flip_equivalence_proof(
        dag,
        model,
    )
    if not parity_safe:
        return dag
    if not dag.amplitude_roots:
        return dag

    pure_massless_adjoint = _pure_massless_adjoint_helicity_pruning_safe(
        dag,
        model,
        vertex_inventory,
    )
    initial_leg_labels = {
        int(leg.label) for leg in dag.process.legs if leg.side == "initial"
    }
    compact_result = _compact_helicity_flip_representatives(
        dag,
        initial_leg_labels=initial_leg_labels,
        pure_massless_adjoint=pure_massless_adjoint,
    )
    if compact_result is None:
        retained, changed, zero_pruned = _generic_helicity_flip_representatives(
            dag,
            initial_leg_labels=initial_leg_labels,
            pure_massless_adjoint=pure_massless_adjoint,
        )
    else:
        retained, changed, zero_pruned = compact_result

    if not changed and not zero_pruned:
        return dag
    return prune_dag_to_amplitude_roots(
        GenericDAG(
            process=dag.process,
            color_plan=dag.color_plan,
            currents=dag.currents,
            sources=dag.sources,
            interactions=dag.interactions,
            amplitude_roots=tuple(retained),
            truncated=dag.truncated,
            helicity_coverage=dag.helicity_coverage,
            color_coverage=dag.color_coverage,
            selected_source_helicities=dag.selected_source_helicities,
            selected_color_sector_ids=dag.selected_color_sector_ids,
            lc_topology_replay=dag.lc_topology_replay,
            color_topology_replay=dag.color_topology_replay,
        )
    )


def _compact_helicity_flip_representatives(
    dag: GenericDAG,
    *,
    initial_leg_labels: set[int],
    pure_massless_adjoint: bool,
) -> tuple[list[AmplitudeRoot], bool, bool] | None:
    """Select flip representatives using dense physical-helicity bit codes.

    Color-decorated DAGs may contain thousands of source currents even though
    each root depends on one source per external leg. Walking the set ancestry
    bits avoids scanning every decorated source for every amplitude root. The
    conservative checks keep unfamiliar source layouts on the generic path.
    """

    encoding = _compact_source_helicity_encoding(dag, initial_leg_labels)
    if encoding is None:
        return None
    helicity_by_source_bit, leg_by_source_bit, all_legs_mask, initial_mask = encoding
    leg_count = all_legs_mask.bit_count()
    stride = all_legs_mask + 1
    roots_by_key: dict[int, list[AmplitudeRoot]] = {}
    zero_pruned = False

    for root in dag.amplitude_roots:
        ancestry = int(
            dag.currents[root.left_id].index.helicity_ancestry
            | dag.currents[root.right_id].index.helicity_ancestry
        )
        helicity_code = 0
        present_legs = 0
        remaining = ancestry
        while remaining:
            source_bit = remaining & -remaining
            bit_index = source_bit.bit_length() - 1
            if bit_index >= len(helicity_by_source_bit):
                return None
            helicity_value = helicity_by_source_bit[bit_index]
            leg_bit = leg_by_source_bit[bit_index]
            if helicity_value is None or leg_bit is None or present_legs & leg_bit:
                return None
            helicity_code |= helicity_value
            present_legs |= leg_bit
            remaining ^= source_bit
        if present_legs != all_legs_mask:
            return None

        if pure_massless_adjoint:
            positive_count = (helicity_code ^ initial_mask).bit_count()
            if positive_count < 2 or leg_count - positive_count < 2:
                zero_pruned = True
                continue

        sector_id = _amplitude_root_color_sector_id(dag, root)
        key = sector_id * stride + helicity_code
        roots_by_key.setdefault(key, []).append(root)

    if not roots_by_key:
        return [], False, False

    retained: list[AmplitudeRoot] = []
    changed = False
    for key in sorted(roots_by_key):
        sector_id, helicity_code = divmod(key, stride)
        flipped_key = sector_id * stride + (helicity_code ^ all_legs_mask)
        partner = roots_by_key.get(flipped_key)
        if partner is not None and flipped_key < key:
            continue
        weight = 1.0
        if partner is not None and flipped_key != key:
            weight = 2.0
            changed = True
        for root in roots_by_key[key]:
            retained.append(_weighted_amplitude_root(root, len(retained), weight))
    return retained, changed, zero_pruned


def _compact_source_helicity_encoding(
    dag: GenericDAG,
    initial_leg_labels: set[int],
) -> (
    tuple[
        list[int | None],
        list[int | None],
        int,
        int,
    ]
    | None
):
    labels = tuple(sorted(int(leg.label) for leg in dag.process.legs))
    if not labels or len(set(labels)) != len(labels) or not dag.sources:
        return None
    leg_bits = {
        label: 1 << (len(labels) - index - 1) for index, label in enumerate(labels)
    }

    source_records: list[tuple[int, int, int]] = []
    for source_id in dag.sources:
        current = dag.currents[source_id]
        ancestry = int(current.index.helicity_ancestry)
        label = int(current.source_leg_label or 0)
        helicity = int(current.source_helicity or 0)
        if (
            ancestry <= 0
            or ancestry & (ancestry - 1)
            or label not in leg_bits
            or helicity not in {-1, 1}
        ):
            return None
        source_records.append((ancestry, label, helicity))
    source_records.sort()
    if any(left[1] > right[1] for left, right in pairwise(source_records)):
        return None

    table_size = source_records[-1][0].bit_length()
    helicity_by_source_bit: list[int | None] = [None] * table_size
    leg_by_source_bit: list[int | None] = [None] * table_size
    for ancestry, label, helicity in source_records:
        bit_index = ancestry.bit_length() - 1
        leg_bit = leg_bits[label]
        helicity_value = leg_bit if helicity == 1 else 0
        existing_helicity = helicity_by_source_bit[bit_index]
        existing_leg = leg_by_source_bit[bit_index]
        if existing_helicity is not None and (
            existing_helicity != helicity_value or existing_leg != leg_bit
        ):
            return None
        helicity_by_source_bit[bit_index] = helicity_value
        leg_by_source_bit[bit_index] = leg_bit

    all_legs_mask = (1 << len(labels)) - 1
    initial_mask = 0
    for label in initial_leg_labels:
        leg_bit = leg_bits.get(label)
        if leg_bit is None:
            return None
        initial_mask |= leg_bit
    return (
        helicity_by_source_bit,
        leg_by_source_bit,
        all_legs_mask,
        initial_mask,
    )


def _generic_helicity_flip_representatives(
    dag: GenericDAG,
    *,
    initial_leg_labels: set[int],
    pure_massless_adjoint: bool,
) -> tuple[list[AmplitudeRoot], bool, bool]:
    """Retain the fully generic tuple-signature implementation as fallback."""

    source_by_bit = _source_helicity_signature_by_bit(dag)
    roots_by_signature: dict[tuple[object, ...], list[AmplitudeRoot]] = {}
    source_helicities_by_ancestry: dict[int, tuple[tuple[int, int], ...]] = {}
    zero_pruned = False
    for root in dag.amplitude_roots:
        signature = _root_physical_helicity_signature(
            dag,
            root,
            source_by_bit,
            source_helicities_by_ancestry,
        )
        if pure_massless_adjoint and _pure_massless_adjoint_helicity_signature_is_zero(
            signature,
            initial_leg_labels,
        ):
            zero_pruned = True
            continue
        roots_by_signature.setdefault(
            signature,
            [],
        ).append(root)
    if not roots_by_signature:
        return [], False, False

    handled: set[tuple[object, ...]] = set()
    retained: list[AmplitudeRoot] = []
    changed = False
    for signature in sorted(roots_by_signature):
        if signature in handled:
            continue
        flipped = _flip_root_physical_helicity_signature(signature)
        partner = roots_by_signature.get(flipped)
        handled.add(signature)
        if partner is not None:
            handled.add(flipped)
        weight = 1.0
        if partner is not None and flipped != signature:
            weight = 2.0
            changed = True
        for root in roots_by_signature[signature]:
            retained.append(_weighted_amplitude_root(root, len(retained), weight))
    return retained, changed, zero_pruned


def _weighted_amplitude_root(
    root: AmplitudeRoot,
    root_id: int,
    weight: float,
) -> AmplitudeRoot:
    return AmplitudeRoot(
        id=root_id,
        kind=root.kind,
        left_id=root.left_id,
        right_id=root.right_id,
        color_weight=root.color_weight,
        contraction_ir=root.contraction_ir,
        color_sector_id=root.color_sector_id,
        vertex_kind=root.vertex_kind,
        vertex_particles=root.vertex_particles,
        coupling=root.coupling,
        helicity_weight=root.helicity_weight * weight,
    )


def _global_helicity_flip_equivalence_proof(
    dag: GenericDAG,
    model: Model,
) -> tuple[bool, tuple[Vertex, ...]]:
    if dag.process.color_accuracy not in {"lc", "nlc", "full"}:
        return False, ()
    for leg in dag.process.legs:
        if leg.outgoing_pdg is None:
            return False, ()
        pdg = int(leg.outgoing_pdg)
        if not (
            model.is_massless_adjoint_vector(pdg)
            or model.is_fundamental_colored_fermion(pdg)
        ):
            return False, ()
        if model.mass(pdg) != 0.0:
            return False, ()
    vertices = _dag_vertex_inventory(dag)
    return (
        model.global_helicity_flip_equivalence_is_proven(vertices),
        vertices,
    )


def _dag_vertex_inventory(dag: GenericDAG) -> tuple[Vertex, ...]:
    """Return each local vertex contract once in deterministic order."""

    identities = {
        (int(interaction.vertex_kind), interaction.vertex_particles)
        for interaction in dag.interactions
    }
    identities.update(
        (int(root.vertex_kind), root.vertex_particles)
        for root in dag.amplitude_roots
        if root.vertex_kind is not None and root.vertex_particles is not None
    )
    return tuple(Vertex(kind, particles) for kind, particles in sorted(identities))


def _pure_massless_adjoint_helicity_pruning_safe(
    dag: GenericDAG,
    model: Model,
    vertices: Sequence[Vertex] | None = None,
) -> bool:
    return model.pure_massless_adjoint_helicity_zero_rule_is_proven(
        dag.process,
        _dag_vertex_inventory(dag) if vertices is None else vertices,
    )


def _pure_massless_adjoint_helicity_signature_is_zero(
    signature: tuple[object, ...],
    initial_leg_labels: set[int],
) -> bool:
    _sector_id, source_helicities = signature
    helicities = []
    for label, helicity in cast(Sequence[tuple[int, int]], source_helicities):
        value = int(helicity)
        if int(label) in initial_leg_labels:
            value = -value
        helicities.append(value)
    return helicities.count(1) < 2 or helicities.count(-1) < 2


def _root_physical_helicity_signature(
    dag: GenericDAG,
    root: AmplitudeRoot,
    source_by_bit: Mapping[int, tuple[int, int]],
    source_helicities_by_ancestry: dict[
        int,
        tuple[tuple[int, int], ...],
    ]
    | None = None,
) -> tuple[object, ...]:
    left = dag.currents[root.left_id].index
    right = dag.currents[root.right_id].index
    ancestry = int(left.helicity_ancestry | right.helicity_ancestry)
    source_helicities = (
        None
        if source_helicities_by_ancestry is None
        else source_helicities_by_ancestry.get(ancestry)
    )
    if source_helicities is None:
        source_helicities = _source_helicities_for_ancestry(
            ancestry,
            source_by_bit,
        )
        if source_helicities_by_ancestry is not None:
            source_helicities_by_ancestry[ancestry] = source_helicities
    sector_id = _amplitude_root_color_sector_id(dag, root)
    return (sector_id, source_helicities)


def _root_source_helicity_mapping(
    dag: GenericDAG,
    root: AmplitudeRoot,
    source_by_bit: Mapping[int, tuple[int, int]],
) -> dict[int, int]:
    left = dag.currents[root.left_id].index
    right = dag.currents[root.right_id].index
    ancestry = int(left.helicity_ancestry | right.helicity_ancestry)
    return {
        int(label): int(helicity)
        for label, helicity in _source_helicities_for_ancestry(
            ancestry,
            source_by_bit,
        )
    }


def _source_helicities_for_ancestry(
    ancestry: int,
    source_by_bit: Mapping[int, tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Resolve only set ancestry bits while preserving ascending-bit order."""

    records: list[tuple[int, int]] = []
    remaining = int(ancestry)
    while remaining:
        bit = remaining & -remaining
        source = source_by_bit.get(bit)
        if source is not None:
            records.append(source)
        remaining ^= bit
    return tuple(records)


def _source_helicity_signature_by_bit(
    dag: GenericDAG,
) -> dict[int, tuple[int, int]]:
    records: list[tuple[int, tuple[int, int]]] = []
    for source_id in dag.sources:
        current = dag.currents[source_id]
        records.append(
            (
                int(current.index.helicity_ancestry),
                (
                    int(current.source_leg_label or 0),
                    int(current.source_helicity or 0),
                ),
            )
        )
    return dict(sorted(records))


def _canonicalize_amplitude_root_order(dag: GenericDAG) -> GenericDAG:
    """Order roots by their physical colour/helicity representative.

    Eager preselection omits one member of every proven global-helicity-flip
    pair before source-bit allocation.  Its ancestry integers therefore differ
    from a complete DAG even though the retained physical states are identical.
    Canonicalizing by physical source labels keeps public reduction identifiers
    independent of that implementation detail.
    """

    source_by_bit = _source_helicity_signature_by_bit(dag)
    source_helicities_by_ancestry: dict[int, tuple[tuple[int, int], ...]] = {}
    ordered = tuple(
        sorted(
            dag.amplitude_roots,
            key=lambda root: (
                _root_physical_helicity_signature(
                    dag,
                    root,
                    source_by_bit,
                    source_helicities_by_ancestry,
                ),
                root.id,
            ),
        )
    )
    if all(root.id == index for index, root in enumerate(ordered)):
        return dag
    return replace(
        dag,
        amplitude_roots=tuple(
            replace(root, id=index) for index, root in enumerate(ordered)
        ),
    )


def _flip_root_physical_helicity_signature(
    signature: tuple[object, ...],
) -> tuple[object, ...]:
    sector_id, source_helicities = signature
    return (
        sector_id,
        tuple(
            (int(label), -int(helicity))
            for label, helicity in cast(
                Sequence[tuple[int, int]],
                source_helicities,
            )
        ),
    )


def contributing_color_sector_ids(dag: GenericDAG) -> tuple[int, ...]:
    """Return LC colour sectors that actually contribute amplitude roots."""

    return tuple(
        sorted(
            {_amplitude_root_color_sector_id(dag, root) for root in dag.amplitude_roots}
        )
    )


def _amplitude_root_color_sector_id(dag: GenericDAG, root: AmplitudeRoot) -> int:
    if root.color_sector_id is not None:
        return int(root.color_sector_id)
    left = dag.currents[root.left_id].index
    if left.color_state.accuracy in {"lc", "nlc", "full"}:
        return int(left.color_state.sector_id)
    return 0


def filter_dag_to_color_sectors(
    dag: GenericDAG,
    sector_ids: Iterable[int],
) -> GenericDAG:
    """Return a dense-current DAG restricted to the requested colour sectors.

    Full DAG construction remains useful for diagnostics, but schema-v3 runtime
    records expect dense current ids. This helper derives the runtime DAG by selecting
    roots whose LC colour sector is in ``sector_ids``, walking backward through
    the current DAG, and remapping the required currents/interactions densely.
    Root-based filtering is required for shared LC all-ordering DAGs where
    internal currents are sector-neutral but amplitude roots still carry the
    physical sector identity.
    """

    selected = set(sector_ids)
    if not selected:
        return GenericDAG(
            process=dag.process,
            color_plan=dag.color_plan,
            currents=(),
            sources=(),
            interactions=(),
            amplitude_roots=(),
            truncated=dag.truncated,
            helicity_coverage=dag.helicity_coverage,
            color_coverage="selected",
            selected_source_helicities=dag.selected_source_helicities,
            selected_color_sector_ids=tuple(sorted(selected)),
            lc_topology_replay=None,
            color_topology_replay=None,
        )

    selected_roots = tuple(
        root
        for root in dag.amplitude_roots
        if _amplitude_root_color_sector_id(dag, root) in selected
    )
    if not selected_roots:
        return GenericDAG(
            process=dag.process,
            color_plan=dag.color_plan,
            currents=(),
            sources=(),
            interactions=(),
            amplitude_roots=(),
            truncated=dag.truncated,
            helicity_coverage=dag.helicity_coverage,
            color_coverage="selected",
            selected_source_helicities=dag.selected_source_helicities,
            selected_color_sector_ids=tuple(sorted(selected)),
            lc_topology_replay=None,
            color_topology_replay=None,
        )

    interactions_by_result: dict[int, list[InteractionNode]] = {}
    for interaction in dag.interactions:
        interactions_by_result.setdefault(interaction.result_id, []).append(interaction)

    required_current_ids: set[int] = set()
    required_interaction_ids: set[int] = set()
    stack: list[int] = []
    for root in selected_roots:
        for current_id in (root.left_id, root.right_id):
            if current_id not in required_current_ids:
                required_current_ids.add(current_id)
                stack.append(current_id)

    while stack:
        current_id = stack.pop()
        for interaction in interactions_by_result.get(current_id, ()):
            if interaction.id in required_interaction_ids:
                continue
            required_interaction_ids.add(interaction.id)
            for parent_id in (interaction.left_id, interaction.right_id):
                if parent_id not in required_current_ids:
                    required_current_ids.add(parent_id)
                    stack.append(parent_id)

    current_id_map: dict[int, int] = {}
    currents: list[CurrentNode] = []
    for current in dag.currents:
        if current.id not in required_current_ids:
            continue
        new_id = len(currents)
        current_id_map[current.id] = new_id
        currents.append(
            CurrentNode(
                id=new_id,
                index=current.index,
                dimension=current.dimension,
                is_source=current.is_source,
                source_leg_label=current.source_leg_label,
                source_helicity=current.source_helicity,
            )
        )

    sources = tuple(
        current_id_map[source_id]
        for source_id in dag.sources
        if source_id in current_id_map
    )

    interactions: list[InteractionNode] = []
    for interaction in dag.interactions:
        if interaction.id not in required_interaction_ids:
            continue
        if (
            interaction.left_id not in current_id_map
            or interaction.right_id not in current_id_map
            or interaction.result_id not in current_id_map
        ):
            continue
        interactions.append(
            InteractionNode(
                id=len(interactions),
                vertex_kind=interaction.vertex_kind,
                vertex_particles=interaction.vertex_particles,
                left_id=current_id_map[interaction.left_id],
                right_id=current_id_map[interaction.right_id],
                result_id=current_id_map[interaction.result_id],
                coupling=interaction.coupling,
                color_weight=interaction.color_weight,
                lowering_backend=interaction.lowering_backend,
                full_tensor_network_ready=interaction.full_tensor_network_ready,
                evaluation_group_id=interaction.evaluation_group_id,
                evaluation_factor=interaction.evaluation_factor,
            )
        )

    amplitude_roots: list[AmplitudeRoot] = []
    for root in selected_roots:
        if root.left_id not in current_id_map or root.right_id not in current_id_map:
            continue
        amplitude_roots.append(
            AmplitudeRoot(
                id=len(amplitude_roots),
                kind=root.kind,
                left_id=current_id_map[root.left_id],
                right_id=current_id_map[root.right_id],
                color_weight=root.color_weight,
                contraction_ir=root.contraction_ir,
                color_sector_id=root.color_sector_id,
                vertex_kind=root.vertex_kind,
                vertex_particles=root.vertex_particles,
                coupling=root.coupling,
                helicity_weight=root.helicity_weight,
            )
        )

    return GenericDAG(
        process=dag.process,
        color_plan=dag.color_plan,
        currents=tuple(currents),
        sources=sources,
        interactions=tuple(interactions),
        amplitude_roots=tuple(amplitude_roots),
        truncated=dag.truncated,
        helicity_coverage=dag.helicity_coverage,
        color_coverage="selected",
        selected_source_helicities=dag.selected_source_helicities,
        selected_color_sector_ids=tuple(sorted(selected)),
        lc_topology_replay=None,
        color_topology_replay=None,
    )


def filter_dag_to_source_helicities(
    dag: GenericDAG,
    source_helicities: Mapping[int, int],
) -> GenericDAG:
    """Return a DAG restricted to roots matching fixed source helicities."""

    requested = {
        int(label): int(helicity) for label, helicity in source_helicities.items()
    }
    if not requested:
        return dag

    source_by_bit = _source_helicity_signature_by_bit(dag)
    selected_roots = tuple(
        root
        for root in dag.amplitude_roots
        if _root_source_helicity_mapping(dag, root, source_by_bit).items()
        >= requested.items()
    )
    if len(selected_roots) == len(dag.amplitude_roots):
        return dag
    return prune_dag_to_amplitude_roots(
        GenericDAG(
            process=dag.process,
            color_plan=dag.color_plan,
            currents=dag.currents,
            sources=dag.sources,
            interactions=dag.interactions,
            amplitude_roots=selected_roots,
            truncated=dag.truncated,
            helicity_coverage="selected",
            color_coverage=dag.color_coverage,
            selected_source_helicities=tuple(
                sorted({**dict(dag.selected_source_helicities), **requested}.items())
            ),
            selected_color_sector_ids=dag.selected_color_sector_ids,
            lc_topology_replay=dag.lc_topology_replay,
            color_topology_replay=dag.color_topology_replay,
        )
    )
