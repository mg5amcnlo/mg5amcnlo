# SPDX-License-Identifier: 0BSD
"""Current-table construction for generic process DAGs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace

from ..color.plan import GenericColorPlan, build_color_plan
from ..models._physics_ir import ContractionIR
from ..models.base import (
    CouplingOrders,
    Model,
    QuantumFlow,
    Vertex,
    VertexEvaluationEquivalence,
    VertexLoweringRule,
)
from ..processes.ir import CanonicalProcessIR
from .dag_algorithms import (
    BackwardLiveStatePlan,
    BackwardLiveTransition,
    _canonicalize_amplitude_root_order,
    _LiveCurrentShape,
    _normalize_generation_cap,
    _retain_canonical_open_line_alias_roots,
    build_backward_live_state_plan,
    prune_dag_to_amplitude_roots,
)
from .dag_color import ColorEngine
from .dag_equivalence import (
    RecursiveEvaluationReuseTracker,
    _canonical_kernel_evaluation,
    assign_recursive_current_evaluation_reuse,
)
from .dag_ordering import (
    _closure_candidate_splits,
    _closure_combination_matches_word,
    _closure_side_reachable_masks,
    _complex_weight_mul,
    _direct_contraction_ir,
    _labels_mask,
    _labels_projected_to_word,
    _lc_all_adjoint_symmetry_order_variants,
    _lc_color_order_reachable_masks,
    _mask_labels,
)
from .dag_reachability import (
    _coupling_orders_within_limits,
    _lc_line_groups_within_limit,
    _mask_allowed_by_reachability,
    _masks_by_size,
    _normalize_coupling_order_limits,
    _ordered_splits,
    _right_particles_by_left,
    _state_allowed_by_reachability,
    _useful_states_by_mask,
)
from .dag_table import _CurrentTable
from .dag_types import (
    AmplitudeRoot,
    ColorFlow,
    ColorState,
    CurrentIndex,
    CurrentNode,
    GenericDAG,
    InteractionNode,
)
from .fermion_ordering import apply_fermion_ordering_signs

DAGProgressCallback = Callable[[Mapping[str, str | int]], None]


def _restrict_color_plan(
    color_plan: GenericColorPlan,
    selected_color_sector_ids: frozenset[int] | None,
) -> tuple[GenericColorPlan, tuple[int, ...]]:
    """Apply an explicit sector selection without hiding missing ids."""

    if selected_color_sector_ids is None:
        return color_plan, ()
    selected_sectors = tuple(
        sector
        for sector in color_plan.sectors
        if sector.id in selected_color_sector_ids
    )
    selected_ids = {sector.id for sector in selected_sectors}
    missing_sector_ids = tuple(sorted(selected_color_sector_ids - selected_ids))
    diagnostics = color_plan.diagnostics
    if missing_sector_ids:
        diagnostics = (
            *diagnostics,
            "selected colour sector ids were not materialized: "
            + ", ".join(str(sector_id) for sector_id in missing_sector_ids),
        )
    return (
        GenericColorPlan(
            process=color_plan.process,
            color_accuracy=color_plan.color_accuracy,
            sectors=selected_sectors,
            diagnostics=diagnostics,
            truncated=color_plan.truncated or bool(missing_sector_ids),
            trace_reflections_folded=color_plan.trace_reflections_folded,
            basis=color_plan.basis,
        ),
        missing_sector_ids,
    )


class GenericDAGCompiler:
    """Compile a concrete process into a model-driven current DAG.

    The compiler never classifies the whole process as a family.  It sweeps
    external subsets, asks the model which local vertices are valid for two
    current particle ids, asks the colour engine whether their colour states
    combine, and deduplicates solely by ``CurrentIndex`` equality.
    """

    def __init__(
        self,
        *,
        model: Model,
        color_plan: GenericColorPlan | None = None,
        max_currents: int | None = None,
        max_color_sectors: int | None = None,
        reference_color_order: tuple[int, ...] | None = None,
        selected_color_sector_ids: Iterable[int] | None = None,
        max_coupling_orders: Mapping[str, int] | None = None,
        max_lc_current_line_groups: int | None = None,
        max_quark_pairs: int | None = None,
        closure_side_mask_pruning: bool = True,
        color_order_mask_pruning: bool = True,
        species_reachability_pruning: bool = True,
        ignored_particle_ids: Iterable[int] | None = None,
        ignored_vertex_kinds: Iterable[int] | None = None,
        selected_source_helicities: Mapping[int, int] | None = None,
        lc_all_ordering_symmetry: bool = True,
        online_evaluation_reuse: bool = False,
        backward_live_planning: bool = False,
        progress_callback: DAGProgressCallback | None = None,
    ) -> None:
        self.model = model
        self.color_plan = color_plan
        self.max_currents = _normalize_generation_cap(max_currents)
        self.max_color_sectors = _normalize_generation_cap(max_color_sectors)
        self.reference_color_order = reference_color_order
        self.selected_color_sector_ids = (
            None
            if selected_color_sector_ids is None
            else frozenset(int(sector_id) for sector_id in selected_color_sector_ids)
        )
        self.max_coupling_orders = _normalize_coupling_order_limits(
            max_coupling_orders,
        )
        self.max_lc_current_line_groups = (
            None
            if max_lc_current_line_groups is None
            else max(0, int(max_lc_current_line_groups))
        )
        self.max_quark_pairs = (
            None if max_quark_pairs is None else max(0, int(max_quark_pairs))
        )
        self.closure_side_mask_pruning = bool(closure_side_mask_pruning)
        self.color_order_mask_pruning = bool(color_order_mask_pruning)
        self.species_reachability_pruning = bool(species_reachability_pruning)
        self.ignored_particle_ids = frozenset(
            int(particle_id) for particle_id in (ignored_particle_ids or ())
        )
        self.ignored_vertex_kinds = frozenset(
            int(kind) for kind in (ignored_vertex_kinds or ())
        )
        self.selected_source_helicities = (
            None
            if selected_source_helicities is None
            else {
                int(label): int(helicity)
                for label, helicity in selected_source_helicities.items()
            }
        )
        self.lc_all_ordering_symmetry = bool(lc_all_ordering_symmetry)
        self.online_evaluation_reuse = bool(online_evaluation_reuse)
        self.backward_live_planning = bool(backward_live_planning)
        self.progress_callback = progress_callback

    def _report_progress(self, step: str, **details: str | int) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback({"step": step, **details})

    def _apply_fermion_ordering(self, dag: GenericDAG) -> GenericDAG:
        return apply_fermion_ordering_signs(dag, self.model)

    def compile(self, process: CanonicalProcessIR) -> GenericDAG:
        if not isinstance(process, CanonicalProcessIR):
            raise TypeError(
                "generic DAG compilation requires a model-resolved CanonicalProcessIR"
            )
        process_ir = process
        self._report_progress("colour planning")
        lc_trace_reflection_proven = bool(
            self.lc_all_ordering_symmetry
            and self.model.lc_trace_reflection_equivalence_is_proven(process_ir)
        )
        color_plan = self.color_plan
        if color_plan is None:
            color_plan = build_color_plan(
                process_ir,
                color_accuracy=process_ir.color_accuracy,
                max_sectors=self.max_color_sectors,
                reference_color_order=self.reference_color_order,
                fold_trace_reflections=lc_trace_reflection_proven,
            )
        elif color_plan.process != process_ir:
            raise ValueError("prebuilt color plan does not match the compiled process")
        elif color_plan.color_accuracy != process_ir.color_accuracy:
            raise ValueError(
                "prebuilt color plan accuracy does not match the compiled process"
            )
        self._report_progress(
            "colour-plan",
            color_sector_count=len(color_plan.sectors),
        )
        helicity_coverage = (
            "selected" if self.selected_source_helicities else "complete"
        )
        selected_source_helicities = tuple(
            sorted((self.selected_source_helicities or {}).items())
        )
        selected_color_sector_ids = tuple(sorted(self.selected_color_sector_ids or ()))
        color_coverage = (
            "selected"
            if color_plan.truncated or self.selected_color_sector_ids is not None
            else "complete"
        )
        if (
            self.max_quark_pairs is not None
            and process_ir.color_endpoints.pair_count > self.max_quark_pairs
        ):
            return self._apply_fermion_ordering(
                GenericDAG(
                    process=process_ir,
                    color_plan=color_plan,
                    currents=(),
                    sources=(),
                    interactions=(),
                    amplitude_roots=(),
                    truncated=False,
                    helicity_coverage=helicity_coverage,
                    color_coverage=color_coverage,
                    selected_source_helicities=selected_source_helicities,
                    selected_color_sector_ids=selected_color_sector_ids,
                )
            )
        complete_sector_ids = {sector.id for sector in color_plan.sectors}
        color_plan, _missing_sector_ids = _restrict_color_plan(
            color_plan,
            self.selected_color_sector_ids,
        )
        if (
            self.selected_color_sector_ids is not None
            or {sector.id for sector in color_plan.sectors} != complete_sector_ids
        ):
            color_coverage = "selected"
        color_engine = ColorEngine(
            color_plan,
            self.model,
            shared_lc_all_ordering_symmetry=(
                lc_trace_reflection_proven and self.selected_color_sector_ids is None
            ),
            cache_shared_lc_orderings=self.online_evaluation_reuse,
        )
        global_flip_anchor = self._eager_global_helicity_flip_anchor(process_ir)
        interactions: list[InteractionNode] = []
        interaction_keys: set[tuple[object, ...]] = set()
        evaluation_group_by_key: dict[tuple[object, ...], int] = {}
        evaluation_equivalence_by_kind: dict[int, VertexEvaluationEquivalence] = {}
        evaluation_equivalence_for = self.model.vertex_evaluation_equivalence
        equivalence_record = VertexEvaluationEquivalence
        vertices_by_input: dict[tuple[int, int], tuple[Vertex, ...]] = {}
        right_particles_by_left = _right_particles_by_left(
            self.model,
            color_accuracy=process_ir.color_accuracy,
        )
        vertex_allowed_cache: dict[tuple[int, int, int, int], bool] = {}
        quantum_flow_cache: dict[tuple[object, ...], tuple[QuantumFlow, ...]] = {}
        coupling_order_cache: dict[
            tuple[CouplingOrders, CouplingOrders, int, tuple[int, int, int]],
            CouplingOrders,
        ] = {}
        coupling_order_limit_cache: dict[CouplingOrders, bool] = {}
        state_allowed_cache: dict[
            tuple[int, int, CouplingOrders],
            bool,
        ] = {}
        allowed_current_ids_by_mask: dict[int, tuple[int, ...]] = {}
        candidate_particles_by_mask: dict[
            int,
            dict[int | None, dict[int, list[int]]],
        ] = {}
        candidate_right_ids_by_mask_sector_particle: dict[
            tuple[int, int | None, int],
            tuple[int, ...],
        ] = {}
        reflection_reuse_cache: dict[int, bool] = {}
        lowering_rule_cache: dict[int, VertexLoweringRule] = {}
        vertex_color_weight_cache: dict[Vertex, tuple[float, float]] = {}
        auxiliary_kind_cache: dict[int, str | None] = {}
        duplicate_orientation_cache: dict[Vertex, bool] = {}
        color_flow_cache: dict[
            tuple[
                ColorState,
                ColorState,
                Vertex,
                tuple[int, ...],
                tuple[tuple[int, ...], ...],
            ],
            tuple[ColorFlow, ...],
        ] = {}
        full_mask = _labels_mask(leg.label for leg in process_ir.legs)
        eager_shared_lc_orderings = (
            color_engine.shared_lc_orderings if self.online_evaluation_reuse else False
        )
        shared_lc_all_ordering_symmetry = color_engine.shared_lc_all_ordering_symmetry
        adjoint_vector_labels = frozenset(
            leg.label
            for leg in process_ir.legs
            if leg.outgoing_pdg is not None
            and self.model.is_massless_adjoint_vector(int(leg.outgoing_pdg))
        )
        locally_reflectable_current_ids: set[int] = set()
        closure_candidate_splits = _closure_candidate_splits(
            process_ir,
            self.model,
            color_engine,
            reference_color_order=self.reference_color_order,
        )
        self._report_progress(
            "reachability",
            closure_splits=len(closure_candidate_splits),
        )
        closure_reachable_masks = (
            _closure_side_reachable_masks(
                full_mask,
                closure_candidate_splits,
            )
            if self.closure_side_mask_pruning
            else None
        )
        backward_planner_candidate = bool(
            self.backward_live_planning
            and process_ir.color_accuracy in {"nlc", "full"}
            and not color_engine.shared_lc_orderings
            and not color_engine.shared_single_trace
            and len(color_plan.sectors) > 1
        )
        color_order_reachable_masks = (
            _lc_color_order_reachable_masks(
                process_ir,
                color_plan,
                self.model,
            )
            if self.color_order_mask_pruning and not backward_planner_candidate
            else None
        )
        useful_states_by_mask = (
            _useful_states_by_mask(
                process_ir,
                self.model,
                color_engine,
                closure_candidate_splits,
                closure_reachable_masks,
                color_order_reachable_masks,
                max_coupling_orders=self.max_coupling_orders,
                ignored_particle_ids=self.ignored_particle_ids,
                ignored_vertex_kinds=self.ignored_vertex_kinds,
            )
            if self.species_reachability_pruning
            else None
        )
        live_state_plan = (
            build_backward_live_state_plan(
                process_ir,
                model=self.model,
                color_engine=color_engine,
                closure_candidate_splits=closure_candidate_splits,
                closure_reachable_masks=closure_reachable_masks,
                color_order_reachable_masks=color_order_reachable_masks,
                useful_states_by_mask=useful_states_by_mask,
                max_coupling_orders=self.max_coupling_orders,
                max_lc_current_line_groups=self.max_lc_current_line_groups,
                ignored_particle_ids=self.ignored_particle_ids,
                ignored_vertex_kinds=self.ignored_vertex_kinds,
                selected_source_helicities=self.selected_source_helicities,
                global_flip_anchor=global_flip_anchor,
            )
            if self.backward_live_planning
            else None
        )
        if (
            live_state_plan is None
            and backward_planner_candidate
            and self.color_order_mask_pruning
        ):
            color_order_reachable_masks = _lc_color_order_reachable_masks(
                process_ir,
                color_plan,
                self.model,
            )
            useful_states_by_mask = (
                _useful_states_by_mask(
                    process_ir,
                    self.model,
                    color_engine,
                    closure_candidate_splits,
                    closure_reachable_masks,
                    color_order_reachable_masks,
                    max_coupling_orders=self.max_coupling_orders,
                    ignored_particle_ids=self.ignored_particle_ids,
                    ignored_vertex_kinds=self.ignored_vertex_kinds,
                )
                if self.species_reachability_pruning
                else None
            )
        table = _CurrentTable(self.model)
        current_index_factory = (
            CurrentIndex._from_trusted_values
            if self.online_evaluation_reuse
            else CurrentIndex
        )
        sources = self._build_sources(
            process_ir,
            color_engine,
            table,
            global_flip_anchor=global_flip_anchor,
            live_state_plan=live_state_plan,
        )
        self._report_progress(
            "source-currents",
            current_count=len(table.currents),
            source_count=len(sources),
        )
        reuse_tracker = (
            RecursiveEvaluationReuseTracker(self.model)
            if self.online_evaluation_reuse
            else None
        )
        if reuse_tracker is not None:
            for current in table.currents:
                reuse_tracker.register_source(current)
        truncated = False
        if any(
            int(leg.outgoing_pdg or 0) in self.ignored_particle_ids
            for leg in process_ir.legs
        ):
            return self._apply_fermion_ordering(
                GenericDAG(
                    process=process_ir,
                    color_plan=color_plan,
                    currents=tuple(table.currents),
                    sources=tuple(sources),
                    interactions=(),
                    amplitude_roots=(),
                    truncated=False,
                    helicity_coverage=helicity_coverage,
                    color_coverage=color_coverage,
                    selected_source_helicities=selected_source_helicities,
                    selected_color_sector_ids=selected_color_sector_ids,
                )
            )

        if live_state_plan is not None:
            return self._apply_fermion_ordering(
                self._compile_backward_live_plan(
                    process_ir=process_ir,
                    color_plan=color_plan,
                    table=table,
                    sources=sources,
                    reuse_tracker=reuse_tracker,
                    plan=live_state_plan,
                    global_flip_anchor=global_flip_anchor,
                    helicity_coverage=helicity_coverage,
                    color_coverage=color_coverage,
                    selected_source_helicities=selected_source_helicities,
                )
            )

        def state_allowed(
            mask: int,
            particle_id: int,
            coupling_orders: CouplingOrders,
        ) -> bool:
            if useful_states_by_mask is None:
                return True
            key = (mask, particle_id, coupling_orders)
            cached = state_allowed_cache.get(key)
            if cached is not None:
                return cached
            allowed = _state_allowed_by_reachability(
                useful_states_by_mask,
                mask,
                particle_id,
                coupling_orders,
            )
            state_allowed_cache[key] = allowed
            return allowed

        def combined_coupling_orders(
            left_index: CurrentIndex,
            right_index: CurrentIndex,
            vertex: Vertex,
        ) -> CouplingOrders:
            key = (
                left_index.coupling_orders,
                right_index.coupling_orders,
                vertex.kind,
                vertex.particles,
            )
            cached = coupling_order_cache.get(key)
            if cached is not None:
                return cached
            orders = self.model.combine_coupling_orders(
                left_index,
                right_index,
                vertex,
            )
            coupling_order_cache[key] = orders
            return orders

        def coupling_orders_allowed(orders: CouplingOrders) -> bool:
            if not self.online_evaluation_reuse:
                return _coupling_orders_within_limits(
                    orders,
                    self.max_coupling_orders,
                )
            cached = coupling_order_limit_cache.get(orders)
            if cached is None:
                cached = _coupling_orders_within_limits(
                    orders,
                    self.max_coupling_orders,
                )
                coupling_order_limit_cache[orders] = cached
            return cached

        def allowed_current_ids(mask: int) -> tuple[int, ...]:
            cached = allowed_current_ids_by_mask.get(mask)
            if cached is None:
                cached = tuple(
                    current_id
                    for current_id in table.ids_by_mask(mask)
                    if state_allowed(
                        mask,
                        table.current(current_id).index.particle_id,
                        table.current(current_id).index.coupling_orders,
                    )
                )
                allowed_current_ids_by_mask[mask] = cached
            return cached

        def all_adjoint_vector_current(index: CurrentIndex) -> bool:
            labels = index.external_labels
            return bool(labels) and all(
                label in adjoint_vector_labels for label in labels
            )

        def reflection_reusable_current(current_id: int) -> bool:
            if self.online_evaluation_reuse:
                if not eager_shared_lc_orderings:
                    return False
                cached = reflection_reuse_cache.get(current_id)
                if cached is not None:
                    return cached
            elif not color_engine.shared_lc_orderings:
                return False
            current = table.current(current_id)
            if not all_adjoint_vector_current(current.index):
                return False
            reusable = (
                shared_lc_all_ordering_symmetry
                or current.is_source
                or current_id in locally_reflectable_current_ids
            )
            if self.online_evaluation_reuse:
                reflection_reuse_cache[current_id] = reusable
            return reusable

        def candidate_right_ids_for_mask(
            right_mask: int,
            color_sector_id: int | None,
            left_particle_id: int,
        ) -> tuple[int, ...]:
            if not self.online_evaluation_reuse:
                raise AssertionError("eager candidate lookup requires online reuse")
            key = (right_mask, color_sector_id, left_particle_id)
            cached = candidate_right_ids_by_mask_sector_particle.get(key)
            if cached is not None:
                return cached
            particles_by_sector = candidate_particles_by_mask.get(right_mask)
            if particles_by_sector is None:
                particles_by_sector = {}
                for current_id in allowed_current_ids(right_mask):
                    index = table.currents[current_id].index
                    sector_id = (
                        None
                        if eager_shared_lc_orderings
                        else index.color_state.sector_id
                    )
                    ids_by_particle = particles_by_sector.get(sector_id)
                    if ids_by_particle is None:
                        ids_by_particle = {}
                        particles_by_sector[sector_id] = ids_by_particle
                    ids = ids_by_particle.get(index.particle_id)
                    if ids is None:
                        ids = []
                        ids_by_particle[index.particle_id] = ids
                    ids.append(current_id)
                candidate_particles_by_mask[right_mask] = particles_by_sector
            ids_by_particle = particles_by_sector.get(color_sector_id, {})
            candidates: list[int] = []
            for particle_id in right_particles_by_left.get(left_particle_id, ()):
                candidates.extend(ids_by_particle.get(particle_id, ()))
            cached = tuple(candidates)
            candidate_right_ids_by_mask_sector_particle[key] = cached
            return cached

        def cached_auxiliary_kind(particle_id: int) -> str | None:
            if particle_id in auxiliary_kind_cache:
                return auxiliary_kind_cache[particle_id]
            kind = self.model.auxiliary_kind(particle_id)
            auxiliary_kind_cache[particle_id] = kind
            return kind

        def cached_vertex_color_weight(
            vertex: Vertex,
        ) -> tuple[float, float]:
            cached = vertex_color_weight_cache.get(vertex)
            if cached is not None:
                return cached
            weight = self.model.vertex_color_weight(
                vertex,
                color_accuracy=process_ir.color_accuracy,
            )
            vertex_color_weight_cache[vertex] = weight
            return weight

        def combined_color_flows(
            left: CurrentIndex,
            right: CurrentIndex,
            vertex: Vertex,
            ordered_external_labels: tuple[int, ...],
            sector_support_label_variants: tuple[tuple[int, ...], ...],
        ) -> tuple[ColorFlow, ...]:
            if not self.online_evaluation_reuse:
                return color_engine.combine(
                    left.color_state,
                    right.color_state,
                    vertex,
                    ordered_external_labels=ordered_external_labels,
                    sector_support_label_variants=sector_support_label_variants,
                )
            key = (
                left.color_state,
                right.color_state,
                vertex,
                ordered_external_labels,
                sector_support_label_variants,
            )
            cached = color_flow_cache.get(key)
            if cached is None:
                cached = color_engine.combine(
                    left.color_state,
                    right.color_state,
                    vertex,
                    ordered_external_labels=ordered_external_labels,
                    sector_support_label_variants=sector_support_label_variants,
                )
                color_flow_cache[key] = cached
            return cached

        recursive_masks = tuple(
            mask
            for mask in _masks_by_size(full_mask)
            if mask & (mask - 1) and mask != full_mask
        )
        recursion_stage_total = max(len(process_ir.legs) - 2, 1)
        for mask_index, mask in enumerate(recursive_masks, start=1):
            labels = _mask_labels(mask)
            self._report_progress(
                "recursion",
                stage_index=max(len(labels) - 1, 1),
                stage_total=recursion_stage_total,
                subset_size=len(labels),
                mask_index=mask_index,
                mask_total=len(recursive_masks),
                current_count=len(table.currents),
                interaction_count=len(interactions),
            )
            if not _mask_allowed_by_reachability(
                mask,
                closure_reachable_masks,
                color_order_reachable_masks,
            ):
                continue
            if useful_states_by_mask is not None and mask not in useful_states_by_mask:
                continue
            mask_current_start = len(table.currents)
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
                if useful_states_by_mask is not None and (
                    left_mask not in useful_states_by_mask
                    or right_mask not in useful_states_by_mask
                ):
                    continue
                left_ids = (
                    allowed_current_ids(left_mask)
                    if self.online_evaluation_reuse
                    else table.ids_by_mask(left_mask)
                )
                if not left_ids or not table.has_mask(right_mask):
                    continue
                split_candidates: dict[
                    tuple[int | None, int],
                    tuple[int, ...],
                ] = {}
                for left_id in left_ids:
                    left = table.current(left_id)
                    if not self.online_evaluation_reuse and not state_allowed(
                        left_mask,
                        left.index.particle_id,
                        left.index.coupling_orders,
                    ):
                        continue
                    color_sector_id = (
                        None
                        if (
                            eager_shared_lc_orderings
                            if self.online_evaluation_reuse
                            else color_engine.shared_lc_orderings
                        )
                        else left.index.color_state.sector_id
                    )
                    if self.online_evaluation_reuse:
                        candidate_key = (
                            color_sector_id,
                            left.index.particle_id,
                        )
                        candidate_right_ids_for_left = split_candidates.get(
                            candidate_key
                        )
                        if candidate_right_ids_for_left is None:
                            candidate_right_ids_for_left = candidate_right_ids_for_mask(
                                right_mask,
                                color_sector_id,
                                left.index.particle_id,
                            )
                            split_candidates[candidate_key] = (
                                candidate_right_ids_for_left
                            )
                    else:
                        possible_right_particles = right_particles_by_left.get(
                            left.index.particle_id
                        )
                        if not possible_right_particles:
                            continue
                        candidate_right_ids_for_left = table.ids_by_mask_and_particles(
                            right_mask,
                            possible_right_particles,
                            color_sector_id=color_sector_id,
                        )
                    if not candidate_right_ids_for_left:
                        continue
                    left_reflection_reusable = reflection_reusable_current(left_id)
                    for right_id in candidate_right_ids_for_left:
                        right = table.current(right_id)
                        right_reflection_reusable = reflection_reusable_current(
                            right_id
                        )
                        if not self.online_evaluation_reuse and not state_allowed(
                            right_mask,
                            right.index.particle_id,
                            right.index.coupling_orders,
                        ):
                            continue
                        vertex_lookup_key = (
                            left.index.particle_id,
                            right.index.particle_id,
                        )
                        if vertex_lookup_key in vertices_by_input:
                            vertices = vertices_by_input[vertex_lookup_key]
                        else:
                            vertices = self.model.vertices_accepting(
                                left.index.particle_id,
                                right.index.particle_id,
                                color_accuracy=process_ir.color_accuracy,
                            )
                            vertices_by_input[vertex_lookup_key] = vertices
                        for vertex in vertices:
                            if (
                                vertex.kind in self.ignored_vertex_kinds
                                or vertex.particles[2] in self.ignored_particle_ids
                            ):
                                continue
                            if useful_states_by_mask is not None and vertex.particles[
                                2
                            ] not in useful_states_by_mask.get(mask, {}):
                                continue
                            vertex_key = (vertex.kind, *vertex.particles)
                            if vertex_key in vertex_allowed_cache:
                                vertex_allowed = vertex_allowed_cache[vertex_key]
                            else:
                                vertex_allowed = color_engine.vertex_allowed(vertex)
                                vertex_allowed_cache[vertex_key] = vertex_allowed
                            if not vertex_allowed:
                                continue
                            if self.online_evaluation_reuse:
                                duplicate_orientation = duplicate_orientation_cache.get(
                                    vertex
                                )
                                if duplicate_orientation is None:
                                    duplicate_orientation = (
                                        self.model.skip_duplicate_vertex_orientation(
                                            vertex
                                        )
                                    )
                                    duplicate_orientation_cache[vertex] = (
                                        duplicate_orientation
                                    )
                            else:
                                duplicate_orientation = (
                                    self.model.skip_duplicate_vertex_orientation(vertex)
                                )
                            if duplicate_orientation:
                                continue
                            coupling_orders = combined_coupling_orders(
                                left.index,
                                right.index,
                                vertex,
                            )
                            if not coupling_orders_allowed(coupling_orders):
                                continue
                            if not state_allowed(
                                mask,
                                vertex.particles[2],
                                coupling_orders,
                            ):
                                continue
                            local_two_source_reflection = (
                                (
                                    eager_shared_lc_orderings
                                    if self.online_evaluation_reuse
                                    else color_engine.shared_lc_orderings
                                )
                                and left.is_source
                                and right.is_source
                                and len(left.index.external_labels) == 1
                                and len(right.index.external_labels) == 1
                                and left_reflection_reusable
                                and right_reflection_reusable
                                and self.model.adjoint_current_reflection_phase(vertex)
                                == (-1.0, 0.0)
                            )
                            result_reflection_proven = (
                                shared_lc_all_ordering_symmetry
                                and all_adjoint_vector_current(left.index)
                                and all_adjoint_vector_current(right.index)
                            ) or local_two_source_reflection
                            if result_reflection_proven and max(
                                left.index.external_labels
                            ) >= max(right.index.external_labels):
                                continue
                            ordered_external_labels = (
                                color_engine.ordered_combination_labels(
                                    left.index,
                                    right.index,
                                    vertex,
                                )
                            )
                            if ordered_external_labels is None and not (
                                left_reflection_reusable or right_reflection_reusable
                            ):
                                continue
                            order_variants: tuple[
                                tuple[
                                    tuple[int, ...],
                                    tuple[tuple[int, ...], ...],
                                    tuple[float, float],
                                ],
                                ...,
                            ]
                            if left_reflection_reusable or right_reflection_reusable:
                                variants: list[
                                    tuple[
                                        tuple[int, ...],
                                        tuple[tuple[int, ...], ...],
                                        tuple[float, float],
                                    ]
                                ] = []
                                for (
                                    proposed_labels,
                                    symmetry_weight,
                                ) in _lc_all_adjoint_symmetry_order_variants(
                                    left.index.ordered_external_labels,
                                    right.index.ordered_external_labels,
                                    left_all_adjoint=left_reflection_reusable,
                                    right_all_adjoint=right_reflection_reusable,
                                    result_reflection_proven=(result_reflection_proven),
                                    fold_result_reflections=(
                                        color_coverage == "complete"
                                    ),
                                ):
                                    projected = (
                                        color_engine.shared_lc_ordered_proposed_labels(
                                            proposed_labels,
                                            allow_reversed=result_reflection_proven,
                                        )
                                    )
                                    if projected is None:
                                        continue
                                    support_label_variants = (proposed_labels,)
                                    if (
                                        result_reflection_proven
                                        and color_coverage == "complete"
                                    ):
                                        reflected_labels = tuple(
                                            reversed(proposed_labels)
                                        )
                                        if reflected_labels != proposed_labels:
                                            support_label_variants = (
                                                proposed_labels,
                                                reflected_labels,
                                            )
                                    variants.append(
                                        (
                                            projected,
                                            support_label_variants,
                                            symmetry_weight,
                                        )
                                    )
                                order_variants = tuple(variants)
                                if not order_variants:
                                    continue
                            else:
                                if ordered_external_labels is None:
                                    continue
                                order_variants = (
                                    (
                                        ordered_external_labels,
                                        (ordered_external_labels,),
                                        (1.0, 0.0),
                                    ),
                                )
                            quantum_flow_key = (
                                vertex.kind,
                                vertex.particles,
                                left.index.particle_id,
                                left.index.chirality,
                                left.index.flavour_flow,
                                right.index.particle_id,
                                right.index.chirality,
                                right.index.flavour_flow,
                            )
                            if quantum_flow_key in quantum_flow_cache:
                                quantum_flows = quantum_flow_cache[quantum_flow_key]
                            else:
                                quantum_flows = self.model.allowed_quantum_flows(
                                    vertex,
                                    left.index,
                                    right.index,
                                )
                                quantum_flow_cache[quantum_flow_key] = quantum_flows
                            for quantum_flow in quantum_flows:
                                for variant_index, (
                                    variant_ordered_labels,
                                    sector_support_label_variants,
                                    symmetry_weight,
                                ) in enumerate(order_variants):
                                    for color_flow in combined_color_flows(
                                        left.index,
                                        right.index,
                                        vertex,
                                        variant_ordered_labels,
                                        sector_support_label_variants,
                                    ):
                                        if not _lc_line_groups_within_limit(
                                            color_flow.state,
                                            self.max_lc_current_line_groups,
                                        ):
                                            continue
                                        out_index = current_index_factory(
                                            particle_id=vertex.particles[2],
                                            external_mask=mask,
                                            external_labels=labels,
                                            ordered_external_labels=variant_ordered_labels,
                                            helicity_ancestry=(
                                                left.index.helicity_ancestry
                                                | right.index.helicity_ancestry
                                            ),
                                            chirality=quantum_flow.chirality,
                                            spin_state=quantum_flow.spin_state,
                                            flavour_flow=quantum_flow.flavour_flow,
                                            quantum_number_flow=(
                                                quantum_flow.quantum_number_flow
                                            ),
                                            color_state=color_flow.state,
                                            momentum_mask=(
                                                left.index.momentum_mask
                                                | right.index.momentum_mask
                                            ),
                                            coupling_orders=coupling_orders,
                                            auxiliary_kind=(
                                                cached_auxiliary_kind(
                                                    vertex.particles[2]
                                                )
                                                if self.online_evaluation_reuse
                                                else self.model.auxiliary_kind(
                                                    vertex.particles[2]
                                                )
                                            ),
                                        )
                                        if (
                                            live_state_plan is not None
                                            and not live_state_plan.allows(out_index)
                                        ):
                                            continue
                                        if not self.model.current_allowed(out_index):
                                            continue
                                        result = table.add_or_get(
                                            out_index,
                                            is_source=False,
                                        )
                                        if local_two_source_reflection:
                                            locally_reflectable_current_ids.add(
                                                result.id
                                            )
                                        signed_color_weight = _complex_weight_mul(
                                            _complex_weight_mul(
                                                color_flow.weight,
                                                cached_vertex_color_weight(vertex)
                                                if self.online_evaluation_reuse
                                                else self.model.vertex_color_weight(
                                                    vertex,
                                                    color_accuracy=(
                                                        process_ir.color_accuracy
                                                    ),
                                                ),
                                            ),
                                            symmetry_weight,
                                        )
                                        key = (
                                            vertex.kind,
                                            left_id,
                                            right_id,
                                            result.id,
                                            variant_index,
                                            signed_color_weight,
                                        )
                                        if key in interaction_keys:
                                            continue
                                        interaction_keys.add(key)
                                        if self.online_evaluation_reuse:
                                            rule = lowering_rule_cache.get(vertex.kind)
                                            if rule is None:
                                                rule = self.model.vertex_lowering_rule(
                                                    vertex.kind
                                                )
                                                lowering_rule_cache[vertex.kind] = rule
                                        else:
                                            rule = self.model.vertex_lowering_rule(
                                                vertex.kind
                                            )
                                        if reuse_tracker is not None:
                                            (
                                                evaluation_group_id,
                                                evaluation_factor,
                                            ) = reuse_tracker.interaction_evaluation(
                                                vertex_kind=vertex.kind,
                                                vertex_particles=vertex.particles,
                                                left_id=left_id,
                                                right_id=right_id,
                                                result=result,
                                                coupling=quantum_flow.coupling,
                                                color_weight=signed_color_weight,
                                            )
                                        else:
                                            equivalence = (
                                                evaluation_equivalence_by_kind.get(
                                                    vertex.kind
                                                )
                                            )
                                            if equivalence is None:
                                                equivalence = (
                                                    evaluation_equivalence_for(
                                                        vertex.kind
                                                    )
                                                )
                                                if not equivalence.verified:
                                                    model_type = (
                                                        f"{type(self.model).__module__}."
                                                        f"{type(self.model).__qualname__}"
                                                    )
                                                    equivalence = equivalence_record(
                                                        class_id=(
                                                            f"{model_type}:"
                                                            f"{int(vertex.kind)}"
                                                        )
                                                    )
                                                evaluation_equivalence_by_kind[
                                                    vertex.kind
                                                ] = equivalence
                                            (
                                                canonical_inputs,
                                                evaluation_factor,
                                            ) = _canonical_kernel_evaluation(
                                                equivalence,
                                                left_id,
                                                right_id,
                                            )
                                            evaluation_key = (
                                                equivalence.class_id,
                                                canonical_inputs,
                                                int(result.index.particle_id),
                                                int(result.index.chirality),
                                                quantum_flow.coupling,
                                            )
                                            existing_group_id = (
                                                evaluation_group_by_key.get(
                                                    evaluation_key
                                                )
                                            )
                                            if existing_group_id is None:
                                                existing_group_id = len(
                                                    evaluation_group_by_key
                                                )
                                                evaluation_group_by_key[
                                                    evaluation_key
                                                ] = existing_group_id
                                            evaluation_group_id = existing_group_id
                                        interactions.append(
                                            InteractionNode(
                                                id=len(interactions),
                                                vertex_kind=vertex.kind,
                                                vertex_particles=vertex.particles,
                                                left_id=left_id,
                                                right_id=right_id,
                                                result_id=result.id,
                                                coupling=quantum_flow.coupling,
                                                color_weight=signed_color_weight,
                                                lowering_backend=rule.backend,
                                                full_tensor_network_ready=(
                                                    rule.full_tensor_network_ready
                                                ),
                                                evaluation_group_id=(
                                                    evaluation_group_id
                                                ),
                                                evaluation_factor=evaluation_factor,
                                            )
                                        )
                                        if (
                                            self.max_currents is not None
                                            and len(table.currents) > self.max_currents
                                        ):
                                            truncated = True
                                            return self._apply_fermion_ordering(
                                                GenericDAG(
                                                    process=process_ir,
                                                    color_plan=color_plan,
                                                    currents=tuple(table.currents),
                                                    sources=tuple(sources),
                                                    interactions=tuple(interactions),
                                                    amplitude_roots=tuple(
                                                        self._build_amplitude_roots(
                                                            process_ir,
                                                            table,
                                                            color_engine,
                                                            candidate_splits=closure_candidate_splits,
                                                        )
                                                    ),
                                                    truncated=truncated,
                                                    helicity_coverage=helicity_coverage,
                                                    color_coverage=color_coverage,
                                                    selected_source_helicities=(
                                                        selected_source_helicities
                                                    ),
                                                    selected_color_sector_ids=(
                                                        selected_color_sector_ids
                                                    ),
                                                )
                                            )

            if reuse_tracker is not None:
                reuse_tracker.finalize_currents(
                    table.currents[mask_current_start:],
                )

        self._report_progress(
            "amplitude-closure",
            current_count=len(table.currents),
            interaction_count=len(interactions),
        )
        amplitude_roots = tuple(
            self._build_amplitude_roots(
                process_ir,
                table,
                color_engine,
                candidate_splits=closure_candidate_splits,
            )
        )
        if global_flip_anchor is not None:
            amplitude_roots = tuple(
                replace(
                    root,
                    helicity_weight=2.0 * float(root.helicity_weight),
                )
                for root in amplitude_roots
            )
        dag = GenericDAG(
            process=process_ir,
            color_plan=color_plan,
            currents=tuple(table.currents),
            sources=tuple(sources),
            interactions=tuple(interactions),
            amplitude_roots=amplitude_roots,
            truncated=truncated,
            helicity_coverage=helicity_coverage,
            color_coverage=color_coverage,
            selected_source_helicities=selected_source_helicities,
            selected_color_sector_ids=selected_color_sector_ids,
        )
        dag = _retain_canonical_open_line_alias_roots(dag, self.model)
        if global_flip_anchor is not None:
            dag = _canonicalize_amplitude_root_order(dag)
        if reuse_tracker is not None:
            self._report_progress(
                "symmetry-reuse",
                current_count=len(dag.currents),
                interaction_count=len(dag.interactions),
                amplitude_count=len(dag.amplitude_roots),
            )
            return self._apply_fermion_ordering(dag)
        dag = assign_recursive_current_evaluation_reuse(dag, self.model)
        self._report_progress(
            "symmetry-reuse",
            current_count=len(dag.currents),
            interaction_count=len(dag.interactions),
            amplitude_count=len(dag.amplitude_roots),
        )
        return self._apply_fermion_ordering(dag)

    def _compile_backward_live_plan(
        self,
        *,
        process_ir: CanonicalProcessIR,
        color_plan: GenericColorPlan,
        table: _CurrentTable,
        sources: Sequence[int],
        reuse_tracker: RecursiveEvaluationReuseTracker | None,
        plan: BackwardLiveStatePlan,
        global_flip_anchor: tuple[int, int] | None,
        helicity_coverage: str,
        color_coverage: str,
        selected_source_helicities: tuple[tuple[int, int], ...],
    ) -> GenericDAG:
        """Replay certified live transition templates over exact helicities.

        The backward planner has already performed model, colour, coupling,
        and quantum-flow resolution.  Replay only expands independent source
        helicity ancestries and records the ordinary current/interactions DTOs.
        No backend evaluator or prepared-kernel operation occurs here.
        """

        if len(plan.sources) != len(sources):
            raise ValueError("backward-live source plan does not match source table")
        current_index_factory = (
            CurrentIndex._from_trusted_values
            if self.online_evaluation_reuse
            else CurrentIndex
        )
        ids_by_shape: dict[_LiveCurrentShape, dict[int, int]] = {}
        for source, current_id in zip(plan.sources, sources, strict=True):
            ids_by_shape.setdefault(source.shape, {})[source.helicity_ancestry] = (
                current_id
            )
        transitions_by_mask: dict[int, list[BackwardLiveTransition]] = {}
        for transition in plan.transitions:
            transitions_by_mask.setdefault(
                transition.result.external_mask,
                [],
            ).append(transition)

        interactions: list[InteractionNode] = []
        interaction_keys: set[tuple[object, ...]] = set()
        lowering_rules: dict[int, VertexLoweringRule] = {}
        auxiliary_kinds: dict[int, str | None] = {}
        full_mask = _labels_mask(leg.label for leg in process_ir.legs)
        truncated = False

        def replay_transition(
            transition: BackwardLiveTransition,
            left_id: int,
            right_id: int,
        ) -> None:
            nonlocal truncated
            left = table.current(left_id)
            right = table.current(right_id)
            rule = lowering_rules.get(transition.vertex_kind)
            if rule is None:
                rule = self.model.vertex_lowering_rule(transition.vertex_kind)
                lowering_rules[transition.vertex_kind] = rule
            result_ancestry = int(
                left.index.helicity_ancestry | right.index.helicity_ancestry
            )
            result_by_ancestry = ids_by_shape.setdefault(transition.result, {})
            result_id = result_by_ancestry.get(result_ancestry)
            if result_id is None:
                particle_id = transition.result.particle_id
                if particle_id not in auxiliary_kinds:
                    auxiliary_kinds[particle_id] = self.model.auxiliary_kind(
                        particle_id
                    )
                result = table.add_or_get(
                    current_index_factory(
                        particle_id=particle_id,
                        external_mask=transition.result.external_mask,
                        external_labels=transition.result.external_labels,
                        ordered_external_labels=(
                            transition.result.ordered_external_labels
                        ),
                        helicity_ancestry=result_ancestry,
                        chirality=transition.result.chirality,
                        spin_state=transition.result.spin_state,
                        flavour_flow=transition.result.flavour_flow,
                        quantum_number_flow=(transition.result.quantum_number_flow),
                        color_state=transition.result.color_state,
                        momentum_mask=transition.result.external_mask,
                        coupling_orders=transition.result.coupling_orders,
                        auxiliary_kind=auxiliary_kinds[particle_id],
                    ),
                    is_source=False,
                )
                result_by_ancestry[result_ancestry] = result.id
            else:
                result = table.current(result_id)
            key = (
                transition.vertex_kind,
                left_id,
                right_id,
                result.id,
                0,
                transition.color_weight,
            )
            if key in interaction_keys:
                return
            interaction_keys.add(key)
            if reuse_tracker is None:
                evaluation_group_id = None
                evaluation_factor = (1.0, 0.0)
            else:
                (
                    evaluation_group_id,
                    evaluation_factor,
                ) = reuse_tracker.interaction_evaluation(
                    vertex_kind=transition.vertex_kind,
                    vertex_particles=transition.vertex_particles,
                    left_id=left_id,
                    right_id=right_id,
                    result=result,
                    coupling=transition.coupling,
                    color_weight=transition.color_weight,
                )
            interactions.append(
                InteractionNode(
                    id=len(interactions),
                    vertex_kind=transition.vertex_kind,
                    vertex_particles=transition.vertex_particles,
                    left_id=left_id,
                    right_id=right_id,
                    result_id=result.id,
                    coupling=transition.coupling,
                    color_weight=transition.color_weight,
                    lowering_backend=rule.backend,
                    full_tensor_network_ready=rule.full_tensor_network_ready,
                    evaluation_group_id=evaluation_group_id,
                    evaluation_factor=evaluation_factor,
                )
            )
            if (
                self.max_currents is not None
                and len(table.currents) > self.max_currents
            ):
                truncated = True

        recursive_masks = tuple(
            mask for mask in _masks_by_size(full_mask) if mask in transitions_by_mask
        )
        recursion_stage_total = max(len(process_ir.legs) - 2, 1)
        for mask_index, mask in enumerate(recursive_masks, start=1):
            stage_transitions = transitions_by_mask[mask]
            labels = _mask_labels(mask)
            self._report_progress(
                "recursion",
                stage_index=max(len(labels) - 1, 1),
                stage_total=recursion_stage_total,
                subset_size=len(labels),
                mask_index=mask_index,
                mask_total=len(recursive_masks),
                current_count=len(table.currents),
                interaction_count=len(interactions),
            )
            mask_current_start = len(table.currents)
            transitions_by_split: dict[
                tuple[int, int],
                list[BackwardLiveTransition],
            ] = {}
            for transition in stage_transitions:
                transitions_by_split.setdefault(
                    (
                        transition.left.external_mask,
                        transition.right.external_mask,
                    ),
                    [],
                ).append(transition)
            for split_transitions in transitions_by_split.values():
                transitions_by_left: dict[
                    _LiveCurrentShape,
                    list[BackwardLiveTransition],
                ] = {}
                for transition in split_transitions:
                    transitions_by_left.setdefault(transition.left, []).append(
                        transition
                    )
                exact_left_currents = sorted(
                    (
                        (left_id, left_shape)
                        for left_shape in transitions_by_left
                        for left_id in ids_by_shape.get(left_shape, {}).values()
                    ),
                    key=lambda entry: entry[0],
                )
                for left_id, left_shape in exact_left_currents:
                    for transition in transitions_by_left[left_shape]:
                        right_by_ancestry = ids_by_shape.get(transition.right)
                        if not right_by_ancestry:
                            continue
                        for right_id in sorted(right_by_ancestry.values()):
                            replay_transition(transition, left_id, right_id)
                            if truncated:
                                break
                        if truncated:
                            break
                    if truncated:
                        break
                if truncated:
                    break
            if reuse_tracker is not None:
                reuse_tracker.finalize_currents(table.currents[mask_current_start:])
            if truncated:
                break

        self._report_progress(
            "amplitude-closure",
            current_count=len(table.currents),
            interaction_count=len(interactions),
        )
        amplitude_roots_list: list[AmplitudeRoot] = []
        seen_roots: set[tuple[object, ...]] = set()
        for closure in plan.closures:
            left_by_ancestry = ids_by_shape.get(closure.left)
            right_by_ancestry = ids_by_shape.get(closure.right)
            if not left_by_ancestry or not right_by_ancestry:
                continue
            for left_id in left_by_ancestry.values():
                for right_id in right_by_ancestry.values():
                    root_key = (
                        closure.kind,
                        closure.contraction_ir,
                        left_id,
                        right_id,
                        closure.color_sector_id,
                        closure.vertex_kind,
                        closure.vertex_particles,
                        closure.color_weight,
                    )
                    if root_key in seen_roots:
                        continue
                    seen_roots.add(root_key)
                    amplitude_roots_list.append(
                        AmplitudeRoot(
                            id=len(amplitude_roots_list),
                            kind=closure.kind,
                            left_id=left_id,
                            right_id=right_id,
                            color_weight=closure.color_weight,
                            contraction_ir=closure.contraction_ir,
                            color_sector_id=closure.color_sector_id,
                            vertex_kind=closure.vertex_kind,
                            vertex_particles=closure.vertex_particles,
                            coupling=closure.coupling,
                        )
                    )
        amplitude_roots = tuple(amplitude_roots_list)
        if global_flip_anchor is not None:
            amplitude_roots = tuple(
                replace(
                    root,
                    helicity_weight=2.0 * float(root.helicity_weight),
                )
                for root in amplitude_roots
            )
        dag = GenericDAG(
            process=process_ir,
            color_plan=color_plan,
            currents=tuple(table.currents),
            sources=tuple(sources),
            interactions=tuple(interactions),
            amplitude_roots=amplitude_roots,
            truncated=truncated,
            helicity_coverage=helicity_coverage,
            color_coverage=color_coverage,
            selected_source_helicities=selected_source_helicities,
            selected_color_sector_ids=tuple(
                sorted(self.selected_color_sector_ids or ())
            ),
        )
        dag = _retain_canonical_open_line_alias_roots(dag, self.model)
        dag = prune_dag_to_amplitude_roots(dag)
        if global_flip_anchor is not None:
            dag = _canonicalize_amplitude_root_order(dag)
        if reuse_tracker is not None:
            self._report_progress(
                "symmetry-reuse",
                current_count=len(dag.currents),
                interaction_count=len(dag.interactions),
                amplitude_count=len(dag.amplitude_roots),
            )
            return dag
        dag = assign_recursive_current_evaluation_reuse(dag, self.model)
        self._report_progress(
            "symmetry-reuse",
            current_count=len(dag.currents),
            interaction_count=len(dag.interactions),
            amplitude_count=len(dag.amplitude_roots),
        )
        return dag

    def _build_sources(
        self,
        process_ir: CanonicalProcessIR,
        color_engine: ColorEngine,
        table: _CurrentTable,
        *,
        global_flip_anchor: tuple[int, int] | None = None,
        live_state_plan: BackwardLiveStatePlan | None = None,
    ) -> list[int]:
        current_index_factory = (
            CurrentIndex._from_trusted_values
            if self.online_evaluation_reuse
            else CurrentIndex
        )
        if live_state_plan is not None:
            sources: list[int] = []
            auxiliary_kinds: dict[int, str | None] = {}
            for source in live_state_plan.sources:
                shape = source.shape
                particle_id = shape.particle_id
                if particle_id not in auxiliary_kinds:
                    auxiliary_kinds[particle_id] = self.model.auxiliary_kind(
                        particle_id
                    )
                current = table.add_or_get(
                    current_index_factory(
                        particle_id=particle_id,
                        external_mask=shape.external_mask,
                        external_labels=shape.external_labels,
                        ordered_external_labels=shape.ordered_external_labels,
                        helicity_ancestry=source.helicity_ancestry,
                        chirality=shape.chirality,
                        spin_state=shape.spin_state,
                        flavour_flow=shape.flavour_flow,
                        quantum_number_flow=shape.quantum_number_flow,
                        color_state=shape.color_state,
                        momentum_mask=shape.external_mask,
                        coupling_orders=shape.coupling_orders,
                        auxiliary_kind=auxiliary_kinds[particle_id],
                    ),
                    is_source=True,
                    source_leg_label=source.leg_label,
                    source_helicity=source.source_helicity,
                )
                sources.append(current.id)
            return sources

        sources: list[int] = []
        next_source_bit = 0
        for leg in process_ir.legs:
            if leg.outgoing_pdg is None:
                continue
            particle_id = int(leg.outgoing_pdg)
            source_ir = self.model._source_ir(particle_id)
            source_quantum_flow = (
                self.model.quantum_number_flow(particle_id)
                if live_state_plan is not None
                else None
            )
            source_auxiliary_kind = (
                self.model.auxiliary_kind(particle_id)
                if live_state_plan is not None
                else None
            )
            for color_state in color_engine.source_states_for_leg(leg):
                if not _lc_line_groups_within_limit(
                    color_state,
                    self.max_lc_current_line_groups,
                ):
                    continue
                color_state_is_live = (
                    live_state_plan is None
                    or color_state.sector_id in live_state_plan.active_sector_ids
                )
                for declared_state in source_ir.states:
                    source_state = (
                        source_ir.crossing.apply(declared_state)
                        if leg.is_initial
                        else declared_state
                    )
                    chirality = source_state.chirality
                    source_helicity = source_state.helicity
                    spin_state = source_state.spin_state
                    if (
                        global_flip_anchor is not None
                        and leg.label == global_flip_anchor[0]
                        and int(source_helicity) != global_flip_anchor[1]
                    ):
                        # Preserve the ancestry bit positions of the complete
                        # source basis. This keeps the retained half directly
                        # comparable with the existing post-DAG parity prune.
                        next_source_bit += 1
                        continue
                    if (
                        self.selected_source_helicities is not None
                        and (
                            requested_helicity := self.selected_source_helicities.get(
                                leg.label
                            )
                        )
                        is not None
                        and int(source_helicity) != requested_helicity
                    ):
                        continue
                    helicity_ancestry = 1 << next_source_bit
                    next_source_bit += 1
                    if not color_state_is_live:
                        continue
                    if live_state_plan is not None:
                        assert source_quantum_flow is not None
                        source_shape = _LiveCurrentShape(
                            external_mask=1 << (leg.label - 1),
                            external_labels=(leg.label,),
                            particle_id=particle_id,
                            ordered_external_labels=(leg.label,),
                            color_state=color_state,
                            coupling_orders=(),
                            chirality=chirality,
                            spin_state=spin_state,
                            flavour_flow=(particle_id,),
                            quantum_number_flow=source_quantum_flow,
                        )
                        if source_shape not in live_state_plan.shapes:
                            continue
                    index = current_index_factory(
                        particle_id=particle_id,
                        external_mask=1 << (leg.label - 1),
                        external_labels=(leg.label,),
                        ordered_external_labels=(leg.label,),
                        helicity_ancestry=helicity_ancestry,
                        chirality=chirality,
                        spin_state=spin_state,
                        flavour_flow=(particle_id,),
                        quantum_number_flow=(
                            source_quantum_flow
                            if source_quantum_flow is not None
                            else self.model.quantum_number_flow(particle_id)
                        ),
                        color_state=color_state,
                        momentum_mask=1 << (leg.label - 1),
                        coupling_orders=(),
                        auxiliary_kind=(
                            source_auxiliary_kind
                            if live_state_plan is not None
                            else self.model.auxiliary_kind(particle_id)
                        ),
                    )
                    current = table.add_or_get(
                        index,
                        is_source=True,
                        source_leg_label=leg.label,
                        source_helicity=source_helicity,
                    )
                    sources.append(current.id)
        return sources

    def _eager_global_helicity_flip_anchor(
        self,
        process_ir: CanonicalProcessIR,
    ) -> tuple[int, int] | None:
        """Return one safely fixed source helicity for eager parity pairing.

        Eager mode can avoid constructing both members of a global-helicity-
        flip pair when the same model proof used by the structural post-pass is
        available before recursion. The proof inventory is conservative: it
        includes every non-ignored model vertex admitted by the requested
        coupling-order limits, not only vertices expected for this process.
        """

        if (
            not self.online_evaluation_reuse
            or self.selected_source_helicities
            or process_ir.color_accuracy not in {"lc", "nlc", "full"}
        ):
            return None

        has_fundamental = False
        for leg in process_ir.legs:
            if leg.outgoing_pdg is None:
                return None
            particle_id = int(leg.outgoing_pdg)
            is_fundamental = self.model.is_fundamental_colored_fermion(particle_id)
            if not (
                is_fundamental or self.model.is_massless_adjoint_vector(particle_id)
            ):
                return None
            if self.model.mass(particle_id) != 0.0:
                return None
            has_fundamental = has_fundamental or is_fundamental
        # Pure-adjoint trees also use the stronger all-equal/one-opposite zero
        # theorem. Leave them to the existing combined proof pass for now.
        if not has_fundamental:
            return None

        proof_vertices = tuple(
            vertex
            for vertex in self.model.vertices
            if vertex.kind not in self.ignored_vertex_kinds
            and not any(
                particle_id in self.ignored_particle_ids
                for particle_id in vertex.particles
            )
            and _coupling_orders_within_limits(
                self.model.vertex_coupling_orders(vertex),
                self.max_coupling_orders,
            )
        )
        if (
            not proof_vertices
            or not self.model.global_helicity_flip_equivalence_is_proven(proof_vertices)
        ):
            return None

        for leg in process_ir.legs:
            assert leg.outgoing_pdg is not None
            source_ir = self.model._source_ir(int(leg.outgoing_pdg))
            helicities = sorted(
                {
                    int(
                        (
                            source_ir.crossing.apply(state) if leg.is_initial else state
                        ).helicity
                    )
                    for state in source_ir.states
                }
            )
            if (
                len(helicities) == 2
                and helicities[0] != 0
                and helicities[0] == -helicities[1]
            ):
                return int(leg.label), helicities[0]
        return None

    def _build_amplitude_roots(
        self,
        process_ir: CanonicalProcessIR,
        table: _CurrentTable,
        color_engine: ColorEngine,
        *,
        candidate_splits: tuple[tuple[int, int], ...] | None = None,
    ) -> list[AmplitudeRoot]:
        full_mask = _labels_mask(leg.label for leg in process_ir.legs)
        if candidate_splits is None:
            candidate_splits = _closure_candidate_splits(
                process_ir,
                self.model,
                color_engine,
            )
        roots: list[AmplitudeRoot] = []
        seen: set[tuple[object, ...]] = set()
        right_ids_by_mask_sector: dict[tuple[int, int], tuple[int, ...]] = {}
        eager_color_flows: dict[
            tuple[
                tuple[int, ...],
                tuple[int, ...],
                ColorState,
                ColorState,
            ],
            tuple[ColorFlow, ...],
        ] = {}
        eager_direct_contractions: dict[
            tuple[int, int, int, int],
            ContractionIR | None,
        ] = {}
        eager_vertices: dict[tuple[int, int], tuple[Vertex, ...]] = {}
        eager_vertex_allowed: dict[Vertex, bool] = {}
        eager_vertex_closure_allowed: dict[Vertex, bool] = {}
        eager_vertex_orders: dict[
            tuple[CouplingOrders, CouplingOrders, Vertex],
            CouplingOrders,
        ] = {}
        eager_vertex_quantum_flows: dict[
            tuple[object, ...],
            tuple[QuantumFlow, ...],
        ] = {}
        eager_closure_contractions: dict[int, ContractionIR | None] = {}
        eager_vertex_color_weights: dict[Vertex, tuple[float, float]] = {}

        def closure_color_flows(
            left: CurrentNode,
            right: CurrentNode,
        ) -> tuple[ColorFlow, ...]:
            if not self.online_evaluation_reuse:
                if not color_engine.ordered_closure_allowed(left.index, right.index):
                    return ()
                if color_engine.shared_single_trace:
                    return color_engine.shared_single_trace_closure_flows(
                        left.index,
                        right.index,
                    )
                if color_engine.shared_lc_orderings:
                    return color_engine.shared_lc_closure_flows(
                        left.index,
                        right.index,
                    )
                return color_engine.closure_compatible(
                    left.index.color_state,
                    right.index.color_state,
                    full_mask=full_mask,
                )
            key = (
                left.index.ordered_external_labels,
                right.index.ordered_external_labels,
                left.index.color_state,
                right.index.color_state,
            )
            cached = eager_color_flows.get(key)
            if cached is not None:
                return cached
            if color_engine.shared_single_trace:
                resolved = color_engine.shared_single_trace_closure_flows(
                    left.index,
                    right.index,
                )
            elif color_engine.shared_lc_orderings:
                resolved = color_engine.shared_lc_closure_flows(
                    left.index,
                    right.index,
                )
            elif color_engine.ordered_closure_allowed(left.index, right.index):
                resolved = color_engine.closure_compatible(
                    left.index.color_state,
                    right.index.color_state,
                    full_mask=full_mask,
                )
            else:
                resolved = ()
            eager_color_flows[key] = resolved
            return resolved

        def direct_contraction(
            left: CurrentNode,
            right: CurrentNode,
        ) -> ContractionIR | None:
            if not self.online_evaluation_reuse:
                return _direct_contraction_ir(self.model, left.index, right.index)
            key = (
                left.index.particle_id,
                right.index.particle_id,
                left.index.chirality,
                right.index.chirality,
            )
            if key not in eager_direct_contractions:
                eager_direct_contractions[key] = _direct_contraction_ir(
                    self.model,
                    left.index,
                    right.index,
                )
            return eager_direct_contractions[key]

        def closure_right_ids(
            right_mask: int,
            left: CurrentNode,
        ) -> Sequence[int]:
            if (
                not self.online_evaluation_reuse
                or color_engine.shared_single_trace
                or color_engine.shared_lc_orderings
            ):
                return table.ids_by_mask(right_mask)
            key = (right_mask, left.index.color_state.sector_id)
            cached = right_ids_by_mask_sector.get(key)
            if cached is None:
                sector_id = key[1]
                cached = tuple(
                    current_id
                    for current_id in table.ids_by_mask(right_mask)
                    if table.current(current_id).index.color_state.sector_id
                    == sector_id
                )
                right_ids_by_mask_sector[key] = cached
            return cached

        def closure_vertices(
            left: CurrentNode,
            right: CurrentNode,
        ) -> tuple[Vertex, ...]:
            key = (left.index.particle_id, right.index.particle_id)
            if not self.online_evaluation_reuse:
                return self.model.vertices_accepting(
                    *key,
                    color_accuracy=process_ir.color_accuracy,
                )
            cached = eager_vertices.get(key)
            if cached is None:
                cached = self.model.vertices_accepting(
                    *key,
                    color_accuracy=process_ir.color_accuracy,
                )
                eager_vertices[key] = cached
            return cached

        def closure_vertex_allowed(vertex: Vertex) -> bool:
            if not self.online_evaluation_reuse:
                return color_engine.vertex_allowed(vertex)
            cached = eager_vertex_allowed.get(vertex)
            if cached is None:
                cached = color_engine.vertex_allowed(vertex)
                eager_vertex_allowed[vertex] = cached
            return cached

        def model_vertex_closure_allowed(vertex: Vertex) -> bool:
            if not self.online_evaluation_reuse:
                return self.model.vertex_closure_allowed(vertex)
            cached = eager_vertex_closure_allowed.get(vertex)
            if cached is None:
                cached = self.model.vertex_closure_allowed(vertex)
                eager_vertex_closure_allowed[vertex] = cached
            return cached

        def closure_coupling_orders(
            left: CurrentNode,
            right: CurrentNode,
            vertex: Vertex,
        ) -> CouplingOrders:
            if not self.online_evaluation_reuse:
                return self.model.combine_coupling_orders(
                    left.index,
                    right.index,
                    vertex,
                )
            key = (
                left.index.coupling_orders,
                right.index.coupling_orders,
                vertex,
            )
            cached = eager_vertex_orders.get(key)
            if cached is None:
                cached = self.model.combine_coupling_orders(
                    left.index,
                    right.index,
                    vertex,
                )
                eager_vertex_orders[key] = cached
            return cached

        def closure_quantum_flows(
            left: CurrentNode,
            right: CurrentNode,
            vertex: Vertex,
        ) -> tuple[QuantumFlow, ...]:
            if not self.online_evaluation_reuse:
                return self.model.allowed_quantum_flows(
                    vertex,
                    left.index,
                    right.index,
                )
            key = (
                vertex,
                left.index.particle_id,
                left.index.chirality,
                left.index.spin_state,
                left.index.flavour_flow,
                left.index.quantum_number_flow,
                right.index.particle_id,
                right.index.chirality,
                right.index.spin_state,
                right.index.flavour_flow,
                right.index.quantum_number_flow,
            )
            cached = eager_vertex_quantum_flows.get(key)
            if cached is None:
                cached = self.model.allowed_quantum_flows(
                    vertex,
                    left.index,
                    right.index,
                )
                eager_vertex_quantum_flows[key] = cached
            return cached

        def closure_contraction(vertex: Vertex) -> ContractionIR | None:
            particle_id = vertex.particles[2]
            if not self.online_evaluation_reuse:
                return self.model.closure_contraction_ir(particle_id)
            if particle_id not in eager_closure_contractions:
                eager_closure_contractions[particle_id] = (
                    self.model.closure_contraction_ir(particle_id)
                )
            return eager_closure_contractions[particle_id]

        def closure_vertex_color_weight(vertex: Vertex) -> tuple[float, float]:
            if not self.online_evaluation_reuse:
                return self.model.vertex_color_weight(
                    vertex,
                    color_accuracy=process_ir.color_accuracy,
                )
            cached = eager_vertex_color_weights.get(vertex)
            if cached is None:
                cached = self.model.vertex_color_weight(
                    vertex,
                    color_accuracy=process_ir.color_accuracy,
                )
                eager_vertex_color_weights[vertex] = cached
            return cached

        for left_mask, right_mask in candidate_splits:
            if left_mask == 0 or right_mask == 0:
                continue
            for left_id in table.ids_by_mask(left_mask):
                left = table.current(left_id)
                for right_id in closure_right_ids(right_mask, left):
                    right = table.current(right_id)
                    if left.index.overlaps(right.index):
                        continue
                    color_flows = closure_color_flows(left, right)
                    if not color_flows:
                        continue
                    if (
                        self.reference_color_order is not None
                        and process_ir.color_accuracy == "lc"
                    ):
                        sector = color_engine.color_plan.sector(
                            left.index.color_state.sector_id
                        )
                        if sector is not None and (
                            self.reference_color_order
                            in sector.admissible_traversal_words
                        ):
                            closure_word = sector.canonical_closure_traversal_word(
                                process_ir
                            )
                            if not _closure_combination_matches_word(
                                _labels_projected_to_word(
                                    left.index.ordered_external_labels,
                                    closure_word,
                                ),
                                _labels_projected_to_word(
                                    right.index.ordered_external_labels,
                                    closure_word,
                                ),
                                closure_word,
                            ):
                                continue
                    direct_contraction_ir = direct_contraction(left, right)
                    for color_flow in color_flows:
                        if direct_contraction_ir is not None:
                            direct_key: tuple[object, ...] = (
                                "direct",
                                direct_contraction_ir,
                                left_id,
                                right_id,
                                color_flow.state,
                            )
                            if direct_key not in seen:
                                seen.add(direct_key)
                                roots.append(
                                    AmplitudeRoot(
                                        id=len(roots),
                                        kind="direct-contraction",
                                        left_id=left_id,
                                        right_id=right_id,
                                        color_weight=color_flow.weight,
                                        contraction_ir=direct_contraction_ir,
                                        color_sector_id=color_flow.state.sector_id,
                                    )
                                )
                        for vertex in closure_vertices(left, right):
                            if (
                                vertex.kind in self.ignored_vertex_kinds
                                or vertex.particles[2] in self.ignored_particle_ids
                            ):
                                continue
                            if not closure_vertex_allowed(vertex):
                                continue
                            if not model_vertex_closure_allowed(vertex):
                                continue
                            coupling_orders = closure_coupling_orders(
                                left,
                                right,
                                vertex,
                            )
                            if not _coupling_orders_within_limits(
                                coupling_orders,
                                self.max_coupling_orders,
                            ):
                                continue
                            closure_contraction_ir = closure_contraction(vertex)
                            if closure_contraction_ir is None:
                                continue
                            quantum_flows = closure_quantum_flows(
                                left,
                                right,
                                vertex,
                            )
                            if not quantum_flows:
                                continue
                            vertex_key: tuple[object, ...] = (
                                "vertex",
                                vertex.kind,
                                vertex.particles,
                                left_id,
                                right_id,
                                color_flow.state,
                            )
                            if vertex_key in seen:
                                continue
                            seen.add(vertex_key)
                            roots.append(
                                AmplitudeRoot(
                                    id=len(roots),
                                    kind="vertex-closure",
                                    left_id=left_id,
                                    right_id=right_id,
                                    color_weight=_complex_weight_mul(
                                        color_flow.weight,
                                        closure_vertex_color_weight(vertex),
                                    ),
                                    contraction_ir=closure_contraction_ir,
                                    color_sector_id=color_flow.state.sector_id,
                                    vertex_kind=vertex.kind,
                                    vertex_particles=vertex.particles,
                                    coupling=vertex.coupling,
                                )
                            )
        return roots


def compile_generic_dag(
    process: CanonicalProcessIR,
    *,
    model: Model,
    color_plan: GenericColorPlan | None = None,
    max_currents: int | None = None,
    max_color_sectors: int | None = None,
    reference_color_order: tuple[int, ...] | None = None,
    selected_color_sector_ids: Iterable[int] | None = None,
    max_coupling_orders: Mapping[str, int] | None = None,
    max_lc_current_line_groups: int | None = None,
    max_quark_pairs: int | None = None,
    closure_side_mask_pruning: bool = True,
    color_order_mask_pruning: bool = True,
    species_reachability_pruning: bool = True,
    ignored_particle_ids: Iterable[int] | None = None,
    ignored_vertex_kinds: Iterable[int] | None = None,
    selected_source_helicities: Mapping[int, int] | None = None,
    lc_all_ordering_symmetry: bool = True,
    online_evaluation_reuse: bool = False,
    backward_live_planning: bool = False,
    progress_callback: DAGProgressCallback | None = None,
) -> GenericDAG:
    return GenericDAGCompiler(
        model=model,
        color_plan=color_plan,
        max_currents=max_currents,
        max_color_sectors=max_color_sectors,
        reference_color_order=reference_color_order,
        selected_color_sector_ids=selected_color_sector_ids,
        max_coupling_orders=max_coupling_orders,
        max_lc_current_line_groups=max_lc_current_line_groups,
        max_quark_pairs=max_quark_pairs,
        closure_side_mask_pruning=closure_side_mask_pruning,
        color_order_mask_pruning=color_order_mask_pruning,
        species_reachability_pruning=species_reachability_pruning,
        ignored_particle_ids=ignored_particle_ids,
        ignored_vertex_kinds=ignored_vertex_kinds,
        selected_source_helicities=selected_source_helicities,
        lc_all_ordering_symmetry=lc_all_ordering_symmetry,
        online_evaluation_reuse=online_evaluation_reuse,
        backward_live_planning=backward_live_planning,
        progress_callback=progress_callback,
    ).compile(process)
