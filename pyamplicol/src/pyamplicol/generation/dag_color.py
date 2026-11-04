# SPDX-License-Identifier: 0BSD
"""Local color-flow state transitions for generic DAG generation."""

from __future__ import annotations

from collections.abc import Iterable

from ..color.plan import GenericColorPlan
from ..models.base import Model, Vertex
from ..processes.ir import ProcessLegIR
from .dag_ordering import (
    _closure_combination_matches_word,
    _known_color_representation,
    _known_fermion_statistics,
    _labels_projected_to_word,
    _lc_word_with_sink_last,
    _line_local_singlet_extras_allowed,
    _mask_labels,
    _ordered_combination_matches_word,
    _ordered_combination_segment,
    _sector_group_indices_for_label,
    _sector_intermediate_order_words,
    _word_contains_ordered_segment,
)
from .dag_types import (
    _LC_FIERZ_SINGLET_BASIS,
    ColorFlow,
    ColorState,
    CurrentIndex,
    _lc_color_identity_closure_key,
    _lc_color_identity_closures,
    _lc_color_identity_closures_compatible,
)

_LC_SECTOR_SUPPORT_PREFIX = "lc-sector-support:"


def _lc_sector_support_key(sector_ids: Iterable[int]) -> str:
    normalized = sorted(set(int(value) for value in sector_ids))
    return _LC_SECTOR_SUPPORT_PREFIX + ",".join(str(value) for value in normalized)


def _lc_sector_support(basis_keys: Iterable[str]) -> frozenset[int] | None:
    supports: list[frozenset[int]] = []
    for key in basis_keys:
        if not key.startswith(_LC_SECTOR_SUPPORT_PREFIX):
            continue
        payload = key.removeprefix(_LC_SECTOR_SUPPORT_PREFIX)
        try:
            support = frozenset(int(value) for value in payload.split(",") if value)
        except ValueError:
            return frozenset()
        if not support:
            return frozenset()
        supports.append(support)
    if not supports:
        return None
    result = set(supports[0])
    for support in supports[1:]:
        result.intersection_update(support)
    return frozenset(result)


def _without_lc_sector_support(basis_keys: Iterable[str]) -> set[str]:
    return {key for key in basis_keys if not key.startswith(_LC_SECTOR_SUPPORT_PREFIX)}


class ColorEngine:
    """Local colour-flow engine used by the process-generic recursion."""

    def __init__(
        self,
        color_plan: GenericColorPlan,
        model: Model,
        *,
        shared_lc_all_ordering_symmetry: bool = False,
        cache_shared_lc_orderings: bool = False,
    ) -> None:
        self.color_plan = color_plan
        self.model = model
        self._sector_by_id = {sector.id: sector for sector in color_plan.sectors}
        self._leg_by_label = {leg.label: leg for leg in color_plan.process.legs}
        self._color_rep_by_label: dict[int, int | None] = {}
        for label, leg in self._leg_by_label.items():
            if leg.outgoing_pdg is None:
                self._color_rep_by_label[label] = None
                continue
            self._color_rep_by_label[label] = _known_color_representation(
                self.model,
                leg.outgoing_pdg,
            )
        self._label_is_color_singlet = {
            label: representation == 1
            for label, representation in self._color_rep_by_label.items()
        }
        self._shared_lc_coloured_labels = {
            label
            for label, representation in self._color_rep_by_label.items()
            if representation is not None and representation != 1
        }
        self._shared_lc_singlet_labels = {
            label
            for label, representation in self._color_rep_by_label.items()
            if representation == 1
        }
        self._fundamental_line_auxiliary_ids = (
            self._find_fundamental_line_auxiliary_ids()
        )
        self._fundamental_fermion_pair_count = (
            self._external_fundamental_fermion_pair_count()
        )
        self._vertex_has_colour_cache: dict[tuple[int, int, int, int], bool] = {}
        self._ordered_combination_labels_cache: dict[
            tuple[int, tuple[int, ...], tuple[int, ...], int, int, int, int],
            tuple[int, ...] | None,
        ] = {}
        self._shared_lc_proposed_labels_cache: (
            dict[tuple[tuple[int, ...], bool], tuple[int, ...] | None] | None
        ) = {} if cache_shared_lc_orderings else None
        color_plan_matches_model = self._color_plan_matches_model_roles()
        all_external_massless_adjoint_vectors = (
            self._all_external_massless_adjoint_vectors()
        )
        try:
            shared_single_trace_color_basis = (
                self.model.shared_single_trace_color_basis_is_proven(color_plan.process)
            )
        except (KeyError, NotImplementedError, TypeError, ValueError):
            shared_single_trace_color_basis = False
        self._shared_single_trace = bool(
            color_plan.color_accuracy in {"nlc", "full"}
            and bool(color_plan.sectors)
            and color_plan_matches_model
            and all_external_massless_adjoint_vectors
            and shared_single_trace_color_basis
            and all(sector.kind == "single-trace" for sector in color_plan.sectors)
        )
        self._shared_single_trace_words = tuple(
            word for sector in color_plan.sectors for word in (sector.color_words or ())
        )
        self._shared_single_trace_word_set = frozenset(self._shared_single_trace_words)
        self._shared_single_trace_segments = frozenset(
            tuple(word[start:stop])
            for word in self._shared_single_trace_words
            for start in range(len(word))
            for stop in range(start + 1, len(word) + 1)
        )
        closure_sector_ids_by_word: dict[tuple[int, ...], set[int]] = {}
        for sector in color_plan.sectors:
            for word in sector.admissible_traversal_words:
                closure_sector_ids_by_word.setdefault(tuple(word), set()).add(
                    int(sector.id)
                )
        self._shared_single_trace_closure_sector_ids_by_word = {
            word: tuple(sorted(sector_ids))
            for word, sector_ids in closure_sector_ids_by_word.items()
        }
        self._shared_lc_orderings = (
            (
                color_plan.color_accuracy == "lc"
                or (
                    color_plan.color_accuracy in {"nlc", "full"}
                    and self._fundamental_fermion_pair_count == 1
                )
            )
            and bool(color_plan.sectors)
            and bool(self._shared_lc_coloured_labels)
            and color_plan_matches_model
        )
        self._shared_lc_all_ordering_symmetry = bool(
            shared_lc_all_ordering_symmetry
            and self._shared_lc_orderings
            and not color_plan.truncated
        )
        all_adjoint_process_symmetry = bool(
            self._shared_lc_all_ordering_symmetry
            and all_external_massless_adjoint_vectors
        )
        self._shared_lc_fixed_sink_label = (
            max(self._shared_lc_coloured_labels)
            if all_adjoint_process_symmetry and self._shared_lc_coloured_labels
            else None
        )
        self._shared_lc_words_by_sector = tuple(
            (
                sector,
                tuple(
                    _lc_word_with_sink_last(
                        word,
                        self._shared_lc_fixed_sink_label,
                    )
                    for word in _sector_intermediate_order_words(
                        sector,
                        include_compatible_traversals=self._shared_lc_orderings,
                        process=color_plan.process,
                    )
                ),
            )
            for sector in color_plan.sectors
        )
        self._shared_lc_sector_ids_by_word: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._shared_lc_sector_ids_by_segment: dict[
            tuple[int, ...], tuple[int, ...]
        ] = {}
        if self._shared_lc_orderings:
            sector_ids_by_word: dict[tuple[int, ...], list[int]] = {}
            sector_ids_by_segment: dict[tuple[int, ...], set[int]] = {}
            segments: set[tuple[int, ...]] = set()
            for sector in color_plan.sectors:
                for word in _sector_intermediate_order_words(
                    sector,
                    process=color_plan.process,
                ):
                    normalized_word = _lc_word_with_sink_last(
                        word,
                        self._shared_lc_fixed_sink_label,
                    )
                    sector_ids_by_word.setdefault(normalized_word, []).append(
                        int(sector.id)
                    )
                for word in _sector_intermediate_order_words(
                    sector,
                    process=color_plan.process,
                ):
                    normalized_word = _lc_word_with_sink_last(
                        word,
                        self._shared_lc_fixed_sink_label,
                    )
                    for start in range(len(normalized_word)):
                        for stop in range(start + 1, len(normalized_word) + 1):
                            segment = tuple(normalized_word[start:stop])
                            segments.add(segment)
                            sector_ids_by_segment.setdefault(segment, set()).add(
                                int(sector.id)
                            )
            self._shared_lc_sector_ids_by_word = {
                word: tuple(ids) for word, ids in sector_ids_by_word.items()
            }
            self._shared_lc_sector_ids_by_segment = {
                segment: tuple(sorted(ids))
                for segment, ids in sector_ids_by_segment.items()
            }
            self._shared_lc_segments = frozenset(segments)
        else:
            self._shared_lc_segments = frozenset()

    def _color_plan_matches_model_roles(self) -> bool:
        if any(
            representation is None
            for representation in self._color_rep_by_label.values()
        ):
            return False
        for sector in self.color_plan.sectors:
            sector_coloured_labels = {
                label for group in sector.coloured_label_groups for label in group
            }
            if sector_coloured_labels != self._shared_lc_coloured_labels:
                return False
            if set(sector.singlet_labels) != self._shared_lc_singlet_labels:
                return False
        return True

    def _all_external_massless_adjoint_vectors(self) -> bool:
        if not self._leg_by_label:
            return False
        for leg in self._leg_by_label.values():
            if leg.outgoing_pdg is None:
                return False
            try:
                if not self.model.is_massless_adjoint_vector(leg.outgoing_pdg):
                    return False
            except (KeyError, NotImplementedError, TypeError, ValueError):
                return False
        return True

    def _is_fundamental_colored_fermion(self, particle_id: int) -> bool:
        representation = _known_color_representation(self.model, particle_id)
        return (
            representation is not None
            and abs(representation) == 3
            and _known_fermion_statistics(self.model, particle_id) is True
        )

    def _is_fundamental_line_auxiliary(self, particle_id: int) -> bool:
        try:
            auxiliary_kind = self.model.auxiliary_kind(particle_id)
        except (KeyError, NotImplementedError, TypeError, ValueError):
            return False
        return auxiliary_kind == "u1-subtraction-color-flow-vector"

    def _find_fundamental_line_auxiliary_ids(self) -> frozenset[int]:
        """Identify proven unphysical singlet currents attached to fundamental lines."""

        auxiliary_ids: set[int] = set()
        for vertex in self.model.vertices:
            for index, particle_id in enumerate(vertex.particles):
                if not self._is_fundamental_line_auxiliary(particle_id):
                    continue
                other_particles = (
                    vertex.particles[:index] + vertex.particles[index + 1 :]
                )
                if all(
                    self._is_fundamental_colored_fermion(other_particle)
                    for other_particle in other_particles
                ):
                    auxiliary_ids.add(particle_id)
        return frozenset(auxiliary_ids)

    def _external_fundamental_fermion_pair_count(self) -> int:
        fundamental_count = 0
        antifundamental_count = 0
        for leg in self._leg_by_label.values():
            if leg.outgoing_pdg is None or not self._is_fundamental_colored_fermion(
                leg.outgoing_pdg
            ):
                continue
            representation = _known_color_representation(
                self.model,
                leg.outgoing_pdg,
            )
            if representation == 3:
                fundamental_count += 1
            elif representation == -3:
                antifundamental_count += 1
        return min(fundamental_count, antifundamental_count)

    def source_states_for_leg(self, leg: ProcessLegIR) -> tuple[ColorState, ...]:
        if self._shared_lc_orderings:
            leg_is_singlet = self._label_is_color_singlet.get(leg.label, False)
            return (
                ColorState(
                    accuracy=self.color_plan.color_accuracy,
                    sector_id=0,
                    line_groups=() if leg_is_singlet else (0,),
                ),
            )
        if self._shared_single_trace:
            return (
                ColorState(
                    accuracy=self.color_plan.color_accuracy,
                    sector_id=0,
                    line_groups=(0,),
                ),
            )
        if not self.color_plan.sectors:
            return (ColorState(accuracy=self.color_plan.color_accuracy),)
        states = []
        leg_is_singlet = self._label_is_color_singlet.get(leg.label, False)
        for sector in self.color_plan.sectors:
            groups = (
                ()
                if leg_is_singlet
                else _sector_group_indices_for_label(sector, leg.label)
            )
            if groups or leg_is_singlet or not sector.line_label_groups:
                states.append(
                    ColorState(
                        accuracy=self.color_plan.color_accuracy,
                        sector_id=sector.id,
                        line_groups=groups,
                    )
                )
        return tuple(states)

    def combine(
        self,
        left: ColorState,
        right: ColorState,
        vertex: Vertex,
        *,
        ordered_external_labels: tuple[int, ...] = (),
        sector_support_label_variants: tuple[tuple[int, ...], ...] | None = None,
    ) -> tuple[ColorFlow, ...]:
        if left.accuracy != right.accuracy:
            return ()
        if not self.color_plan.sectors:
            return (
                ColorFlow(
                    state=ColorState(
                        accuracy=left.accuracy,
                        basis_key=tuple(
                            sorted(set(left.basis_key) | set(right.basis_key))
                        ),
                    )
                ),
            )
        if self._shared_single_trace:
            groups = tuple(sorted(set(left.line_groups) | set(right.line_groups)))
            return (
                ColorFlow(
                    state=ColorState(
                        accuracy=left.accuracy,
                        sector_id=0,
                        line_groups=groups or (0,),
                    )
                ),
            )
        if self._shared_lc_orderings:
            groups = self._lc_combined_line_groups(left, right, vertex)
            if groups is None:
                return ()
            projections = self._lc_projected_basis_and_weights(
                left,
                right,
                vertex,
                ordered_external_labels,
                sector_support_label_variants=sector_support_label_variants,
            )
            if not projections:
                return ()
            return tuple(
                ColorFlow(
                    state=ColorState(
                        accuracy=left.accuracy,
                        sector_id=0,
                        line_groups=groups,
                        basis_key=basis_key,
                    ),
                    weight=weight,
                )
                for basis_key, weight in projections
            )
        if left.sector_id != right.sector_id:
            return ()
        groups = self._lc_combined_line_groups(left, right, vertex)
        if groups is None:
            return ()
        projections = self._lc_projected_basis_and_weights(
            left,
            right,
            vertex,
            ordered_external_labels,
            sector_support_label_variants=sector_support_label_variants,
        )
        if not projections:
            return ()
        return tuple(
            ColorFlow(
                state=ColorState(
                    accuracy=left.accuracy,
                    sector_id=left.sector_id,
                    line_groups=groups,
                    basis_key=basis_key,
                ),
                weight=weight,
            )
            for basis_key, weight in projections
        )

    def _lc_projected_basis_and_weights(
        self,
        left: ColorState,
        right: ColorState,
        vertex: Vertex,
        ordered_external_labels: tuple[int, ...],
        *,
        sector_support_label_variants: tuple[tuple[int, ...], ...] | None = None,
    ) -> tuple[tuple[tuple[str, ...], tuple[float, float]], ...]:
        basis_with_support = set(left.basis_key) | set(right.basis_key)
        inherited_support = _lc_sector_support(basis_with_support)
        basis = _without_lc_sector_support(basis_with_support)
        structure = self.model.vertex_color_structure(vertex)
        reps = tuple(abs(self.model.color_rep(pdg)) for pdg in vertex.particles)
        has_fierz_singlet = _LC_FIERZ_SINGLET_BASIS in basis
        if has_fierz_singlet:
            consumes_singlet_exchange = (
                structure == "fundamental-generator"
                and reps[2] == 3
                and sorted(reps[:2]) == [3, 8]
            )
            if not consumes_singlet_exchange:
                return ()
            basis.remove(_LC_FIERZ_SINGLET_BASIS)

        creates_fierz_projection = (
            structure == "fundamental-generator"
            and reps[:2] == (3, 3)
            and reps[2] == 8
            and not self._fundamental_line_auxiliary_ids
        )

        if (
            self._shared_lc_orderings
            and structure == "color-identity"
            and sorted(reps[:2]) == [3, 3]
            and reps[2] == 1
            and vertex.particles[2] not in self._fundamental_line_auxiliary_ids
        ):
            colored_labels = tuple(
                sorted(
                    label
                    for label in ordered_external_labels
                    if label in self._shared_lc_coloured_labels
                )
            )
            if colored_labels:
                basis.add(_lc_color_identity_closure_key(colored_labels))

        if self._shared_lc_orderings:
            support_label_variants = (
                (ordered_external_labels,)
                if sector_support_label_variants is None
                else sector_support_label_variants
            )
            compatible_sector_ids: set[int] = set()
            for support_labels in support_label_variants:
                compatible_sector_ids.update(
                    self._shared_lc_compatible_sector_ids(support_labels)
                )
            if inherited_support is not None:
                compatible_sector_ids.intersection_update(inherited_support)
            if not compatible_sector_ids:
                return ()
            all_sector_ids = set(self._sector_by_id)
            track_sector_support = (
                inherited_support is not None
                or creates_fierz_projection
                or (
                    self.color_plan.color_accuracy == "lc"
                    and compatible_sector_ids != all_sector_ids
                )
            )

            candidates: list[tuple[set[str], tuple[float, float], set[int]]] = []
            if creates_fierz_projection:
                fierz_sector_ids = (
                    self._lc_fierz_sector_ids(ordered_external_labels)
                    & compatible_sector_ids
                )
                ordinary_sector_ids = compatible_sector_ids - fierz_sector_ids
                if ordinary_sector_ids:
                    candidates.append((set(basis), (1.0, 0.0), ordinary_sector_ids))
                if fierz_sector_ids:
                    fierz_basis = set(basis)
                    fierz_basis.add(_LC_FIERZ_SINGLET_BASIS)
                    candidates.append(
                        (
                            fierz_basis,
                            (1.0 / 3.0, 0.0),
                            fierz_sector_ids,
                        )
                    )
            else:
                candidates.append((set(basis), (1.0, 0.0), compatible_sector_ids))

            projections: list[tuple[tuple[str, ...], tuple[float, float]]] = []
            for candidate_basis, weight, support in candidates:
                if track_sector_support and support != all_sector_ids:
                    candidate_basis.add(_lc_sector_support_key(support))
                if not self._shared_lc_basis_has_compatible_sector(
                    candidate_basis,
                    ordered_external_labels=ordered_external_labels,
                ):
                    continue
                projections.append((tuple(sorted(candidate_basis)), weight))
            return tuple(projections)

        if creates_fierz_projection and self._lc_labels_close_one_open_line(
            ordered_external_labels
        ):
            basis.add(_LC_FIERZ_SINGLET_BASIS)
            return ((tuple(sorted(basis)), (1.0 / 3.0, 0.0)),)
        return ((tuple(sorted(basis)), (1.0, 0.0)),)

    def _shared_lc_compatible_sector_ids(
        self,
        ordered_external_labels: tuple[int, ...],
    ) -> set[int]:
        colored = tuple(
            label
            for label in ordered_external_labels
            if label in self._shared_lc_coloured_labels
        )
        if not colored:
            return set(self._sector_by_id)
        return set(self._shared_lc_sector_ids_by_segment.get(colored, ()))

    def _lc_fierz_sector_ids(
        self,
        ordered_external_labels: tuple[int, ...],
    ) -> set[int]:
        colored = tuple(
            label
            for label in ordered_external_labels
            if label in self._shared_lc_coloured_labels
        )
        labels = set(colored)
        sector_ids: set[int] = set()
        if not labels:
            return sector_ids
        for sector, words in self._shared_lc_words_by_sector:
            if not any(_word_contains_ordered_segment(word, colored) for word in words):
                continue
            complete_groups = tuple(
                set(group)
                for group in sector.line_label_groups
                if set(group).issubset(labels)
            )
            if complete_groups and set().union(*complete_groups) == labels:
                sector_ids.add(int(sector.id))
        return sector_ids

    def _shared_lc_basis_has_compatible_sector(
        self,
        basis_keys: Iterable[str],
        *,
        ordered_external_labels: Iterable[int] = (),
    ) -> bool:
        basis_tuple = tuple(basis_keys)
        closures = _lc_color_identity_closures(basis_tuple)
        support = _lc_sector_support(basis_tuple)
        colored_segment = tuple(
            label
            for label in ordered_external_labels
            if label in self._shared_lc_coloured_labels
        )
        sector_ids = (
            self._shared_lc_sector_ids_by_segment.get(colored_segment, ())
            if colored_segment
            else tuple(self._sector_by_id)
        )
        return any(
            sector is not None
            and (support is None or sector_id in support)
            and _lc_color_identity_closures_compatible(closures, sector)
            for sector_id in sector_ids
            for sector in (self._sector_by_id.get(sector_id),)
        )

    def _lc_labels_close_one_open_line(
        self,
        ordered_external_labels: tuple[int, ...],
    ) -> bool:
        colored = tuple(
            label
            for label in ordered_external_labels
            if label in self._shared_lc_coloured_labels
        )
        if not colored:
            return False
        return bool(self._lc_fierz_sector_ids(colored))

    def _lc_combined_line_groups(
        self,
        left: ColorState,
        right: ColorState,
        vertex: Vertex,
    ) -> tuple[int, ...] | None:
        left_colored = self._particle_has_colour(vertex.particles[0])
        right_colored = self._particle_has_colour(vertex.particles[1])
        result_colored = self._particle_has_colour(vertex.particles[2])
        left_groups = set(left.line_groups)
        right_groups = set(right.line_groups)

        if not left_colored and not right_colored:
            if left_groups and right_groups and left_groups != right_groups:
                return None
            return tuple(sorted(left_groups | right_groups))

        if left_colored != right_colored:
            colored_groups = left_groups if left_colored else right_groups
            singlet_groups = right_groups if left_colored else left_groups
            if (
                singlet_groups
                and colored_groups
                and not singlet_groups.issubset(colored_groups)
            ):
                return None
            return tuple(sorted(colored_groups or singlet_groups))

        if (
            not result_colored
            and left_groups
            and right_groups
            and left_groups != right_groups
        ):
            return None
        return tuple(sorted(left_groups | right_groups))

    def _particle_has_colour(self, pdg: int) -> bool:
        representation = _known_color_representation(self.model, pdg)
        return (
            representation is not None and representation != 1
        ) or pdg in self._fundamental_line_auxiliary_ids

    def closure_compatible(
        self,
        left: ColorState,
        right: ColorState,
        *,
        full_mask: int,
    ) -> tuple[ColorFlow, ...]:
        if left.accuracy != right.accuracy:
            return ()
        if not self.color_plan.sectors:
            return (
                ColorFlow(
                    state=ColorState(
                        accuracy=left.accuracy,
                        basis_key=tuple(
                            sorted(set(left.basis_key) | set(right.basis_key))
                        ),
                    )
                ),
            )
        if self._shared_single_trace:
            # Shared single-trace closure needs the endpoint order labels, not
            # just the state.  _build_amplitude_roots therefore calls
            # shared_single_trace_closure_flows() with full CurrentIndex data.
            return ()
        if left.sector_id != right.sector_id:
            return ()
        groups = tuple(sorted(set(left.line_groups) | set(right.line_groups)))
        sector = self._sector_by_id.get(left.sector_id)
        if sector is None:
            return (ColorFlow(state=left),)
        basis = tuple(sorted(set(left.basis_key) | set(right.basis_key)))
        if not _lc_color_identity_closures_compatible(
            _lc_color_identity_closures(basis),
            sector,
        ):
            return ()
        full_labels = set(_mask_labels(full_mask))
        sector_labels = set(sector.singlet_labels)
        for group in sector.line_label_groups:
            sector_labels.update(group)
        if full_labels and not full_labels.issubset(sector_labels):
            return ()
        return (
            ColorFlow(
                state=ColorState(
                    accuracy=left.accuracy,
                    sector_id=left.sector_id,
                    line_groups=groups,
                )
            ),
        )

    def _vertex_has_colour(self, vertex: Vertex) -> bool:
        key = (vertex.kind, *vertex.particles)
        cached = self._vertex_has_colour_cache.get(key)
        if cached is not None:
            return cached
        for pdg in vertex.particles:
            if self._particle_has_colour(pdg):
                self._vertex_has_colour_cache[key] = True
                return True
        self._vertex_has_colour_cache[key] = False
        return False

    def vertex_allowed(self, vertex: Vertex) -> bool:
        contains_fundamental_auxiliary = any(
            particle_id in self._fundamental_line_auxiliary_ids
            for particle_id in vertex.particles
        )
        if contains_fundamental_auxiliary:
            return self._fundamental_fermion_pair_count >= 2
        if (
            self._fundamental_line_auxiliary_ids
            and self._fundamental_fermion_pair_count >= 2
            and self.model.vertex_color_structure(vertex) == "fundamental-generator"
        ):
            representations = tuple(
                self.model.color_rep(particle_id) for particle_id in vertex.particles
            )
            if representations.count(8) == 1 and sorted(
                abs(value) for value in representations if value != 8
            ) == [3, 3]:
                return representations in {
                    (3, 8, 3),
                    (8, -3, -3),
                    (-3, 3, 8),
                }
        return True

    def ordered_combination_allowed(
        self,
        left_index: CurrentIndex,
        right_index: CurrentIndex,
        vertex: Vertex,
    ) -> bool:
        if not self._vertex_has_colour(vertex):
            return True
        if self._shared_lc_orderings:
            return (
                self._shared_lc_ordered_combination_labels(
                    left_index,
                    right_index,
                )
                is not None
            )
        sector = self._sector_by_id.get(left_index.color_state.sector_id)
        if sector is None:
            return True
        if sector.kind == "singlet" and all(
            not word for word in sector.admissible_traversal_words
        ):
            return True
        return any(
            _ordered_combination_matches_word(
                left_index.ordered_external_labels,
                right_index.ordered_external_labels,
                word,
            )
            for word in _sector_intermediate_order_words(
                sector,
                include_compatible_traversals=False,
                process=self.color_plan.process,
            )
        )

    def ordered_combination_labels(
        self,
        left_index: CurrentIndex,
        right_index: CurrentIndex,
        vertex: Vertex,
    ) -> tuple[int, ...] | None:
        cache_key = (
            left_index.color_state.sector_id,
            left_index.ordered_external_labels,
            right_index.ordered_external_labels,
            vertex.kind,
            *vertex.particles,
        )
        if cache_key in self._ordered_combination_labels_cache:
            return self._ordered_combination_labels_cache[cache_key]
        proposed = (
            *left_index.ordered_external_labels,
            *right_index.ordered_external_labels,
        )
        singlet_order = self._singlet_order_allowed(
            left_index,
            right_index,
            vertex,
        )
        if not singlet_order:
            self._ordered_combination_labels_cache[cache_key] = None
            return None
        if self._shared_single_trace:
            segment = (
                *left_index.ordered_external_labels,
                *right_index.ordered_external_labels,
            )
            labels = segment if segment in self._shared_single_trace_segments else None
            self._ordered_combination_labels_cache[cache_key] = labels
            return labels
        if self._shared_lc_orderings:
            labels = self._shared_lc_ordered_combination_labels(
                left_index,
                right_index,
            )
            self._ordered_combination_labels_cache[cache_key] = labels
            return labels
        sector = self._sector_by_id.get(left_index.color_state.sector_id)
        if sector is None:
            self._ordered_combination_labels_cache[cache_key] = proposed
            return proposed
        for word in _sector_intermediate_order_words(
            sector,
            include_compatible_traversals=False,
            process=self.color_plan.process,
        ):
            segment = _ordered_combination_segment(
                left_index.ordered_external_labels,
                right_index.ordered_external_labels,
                word,
            )
            if segment is None:
                continue
            word_labels = set(word)
            extras = tuple(
                sorted(label for label in proposed if label not in word_labels)
            )
            if not _line_local_singlet_extras_allowed(segment, extras, sector):
                continue
            labels = (*segment, *extras)
            self._ordered_combination_labels_cache[cache_key] = labels
            return labels
        self._ordered_combination_labels_cache[cache_key] = None
        return None

    def _singlet_order_allowed(
        self,
        left_index: CurrentIndex,
        right_index: CurrentIndex,
        vertex: Vertex,
    ) -> bool:
        left_singlet = self._all_external_labels_color_singlet(
            left_index.external_labels
        )
        right_singlet = self._all_external_labels_color_singlet(
            right_index.external_labels
        )
        if left_singlet and not right_singlet:
            return False
        if left_singlet and right_singlet:
            return max(left_index.external_labels) < max(right_index.external_labels)
        return True

    def _all_external_labels_color_singlet(self, labels: Iterable[int]) -> bool:
        for label in labels:
            is_singlet = self._label_is_color_singlet.get(label)
            if is_singlet is None:
                return False
            if not is_singlet:
                return False
        return True

    def ordered_closure_allowed(
        self,
        left_index: CurrentIndex,
        right_index: CurrentIndex,
    ) -> bool:
        if self._shared_single_trace:
            return (
                *left_index.ordered_external_labels,
                *right_index.ordered_external_labels,
            ) in self._shared_single_trace_word_set
        if self._shared_lc_orderings:
            return self._shared_lc_closure_word(left_index, right_index) is not None
        if left_index.color_state.sector_id != right_index.color_state.sector_id:
            return False
        sector = self._sector_by_id.get(left_index.color_state.sector_id)
        if sector is None:
            return True
        if sector.kind == "singlet" and all(
            not word for word in sector.admissible_traversal_words
        ):
            return True
        if sector.kind == "open-lines":
            word = sector.canonical_closure_traversal_word(self.color_plan.process)
            return _closure_combination_matches_word(
                _labels_projected_to_word(
                    left_index.ordered_external_labels,
                    word,
                ),
                _labels_projected_to_word(
                    right_index.ordered_external_labels,
                    word,
                ),
                word,
            )
        return any(
            _closure_combination_matches_word(
                _labels_projected_to_word(left_index.ordered_external_labels, word),
                _labels_projected_to_word(right_index.ordered_external_labels, word),
                word,
            )
            for word in sector.admissible_traversal_words
        )

    def shared_single_trace_closure_flows(
        self,
        left_index: CurrentIndex,
        right_index: CurrentIndex,
    ) -> tuple[ColorFlow, ...]:
        if not self._shared_single_trace:
            return ()
        closure_word = (
            *left_index.ordered_external_labels,
            *right_index.ordered_external_labels,
        )
        return tuple(
            ColorFlow(
                state=ColorState(
                    accuracy=self.color_plan.color_accuracy,
                    sector_id=sector_id,
                    line_groups=(0,),
                )
            )
            for sector_id in self._shared_single_trace_closure_sector_ids_by_word.get(
                closure_word,
                (),
            )
        )

    @property
    def shared_single_trace(self) -> bool:
        return self._shared_single_trace

    def shared_lc_closure_flows(
        self,
        left_index: CurrentIndex,
        right_index: CurrentIndex,
    ) -> tuple[ColorFlow, ...]:
        if not self._shared_lc_orderings:
            return ()
        flows: list[ColorFlow] = []
        closure = self._shared_lc_closure(left_index, right_index)
        if closure is None:
            return ()
        _word, sector_ids = closure
        for sector_id in sector_ids:
            flows.append(
                ColorFlow(
                    state=ColorState(
                        accuracy=self.color_plan.color_accuracy,
                        sector_id=sector_id,
                        line_groups=(0,),
                    )
                )
            )
        return tuple(flows)

    @property
    def shared_lc_orderings(self) -> bool:
        return self._shared_lc_orderings

    @property
    def shared_lc_all_ordering_symmetry(self) -> bool:
        return self._shared_lc_all_ordering_symmetry

    @property
    def shared_lc_fixed_sink_label(self) -> int | None:
        return self._shared_lc_fixed_sink_label

    def _shared_lc_ordered_combination_labels(
        self,
        left_index: CurrentIndex,
        right_index: CurrentIndex,
    ) -> tuple[int, ...] | None:
        proposed = (
            *left_index.ordered_external_labels,
            *right_index.ordered_external_labels,
        )
        return self._shared_lc_ordered_proposed_labels(proposed)

    def shared_lc_ordered_proposed_labels(
        self,
        proposed: Iterable[int],
        *,
        allow_reversed: bool = False,
    ) -> tuple[int, ...] | None:
        if not self._shared_lc_orderings:
            return None
        return self._shared_lc_ordered_proposed_labels(
            tuple(proposed),
            allow_reversed=allow_reversed,
        )

    def _shared_lc_ordered_proposed_labels(
        self,
        proposed: tuple[int, ...],
        *,
        allow_reversed: bool = False,
    ) -> tuple[int, ...] | None:
        cache = self._shared_lc_proposed_labels_cache
        cache_key = (proposed, allow_reversed)
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        coloured_segment = tuple(
            label for label in proposed if label in self._shared_lc_coloured_labels
        )
        extras = tuple(
            sorted(
                label
                for label in proposed
                if label not in self._shared_lc_coloured_labels
            )
        )
        result: tuple[int, ...] | None
        if extras and not set(extras).issubset(self._shared_lc_singlet_labels):
            result = None
        elif not coloured_segment:
            result = extras
        elif coloured_segment not in self._shared_lc_segments:
            reversed_segment = tuple(reversed(coloured_segment))
            if not (allow_reversed and reversed_segment in self._shared_lc_segments):
                result = None
            else:
                result = (*coloured_segment, *extras)
        else:
            result = (*coloured_segment, *extras)
        if cache is not None:
            cache[cache_key] = result
        return result

    def _shared_lc_closure_word(
        self,
        left_index: CurrentIndex,
        right_index: CurrentIndex,
    ) -> tuple[int, ...] | None:
        closure = self._shared_lc_closure(left_index, right_index)
        return None if closure is None else closure[0]

    def _shared_lc_closure(
        self,
        left_index: CurrentIndex,
        right_index: CurrentIndex,
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        left_word = tuple(
            label
            for label in left_index.ordered_external_labels
            if label in self._shared_lc_coloured_labels
        )
        right_word = tuple(
            label
            for label in right_index.ordered_external_labels
            if label in self._shared_lc_coloured_labels
        )
        if not left_word or not right_word:
            return None
        candidates = [(*left_word, *right_word)]
        if self.color_plan.color_accuracy != "lc":
            candidates.append((*right_word, *left_word))
        word = next(
            (
                candidate
                for candidate in candidates
                if candidate in self._shared_lc_sector_ids_by_word
            ),
            None,
        )
        if word is None:
            return None
        sector_ids = self._shared_lc_sector_ids_by_word[word]
        basis_keys = (
            *left_index.color_state.basis_key,
            *right_index.color_state.basis_key,
        )
        closures = _lc_color_identity_closures(basis_keys)
        support = _lc_sector_support(basis_keys)
        compatible_sector_ids = tuple(
            sector_id
            for sector_id in sector_ids
            if (sector := self._sector_by_id.get(sector_id)) is not None
            and (support is None or sector_id in support)
            and _lc_color_identity_closures_compatible(closures, sector)
        )
        if not compatible_sector_ids:
            return None
        return word, compatible_sector_ids
