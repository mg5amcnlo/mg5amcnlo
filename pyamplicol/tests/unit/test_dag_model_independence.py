# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import permutations

import pytest

import pyamplicol.generation.dag_compiler as dag_compiler_module
from pyamplicol.color.plan import GenericColorPlan, LCColorSector, build_color_plan
from pyamplicol.generation.dag_algorithms import (
    prune_global_helicity_flip_equivalent_roots,
)
from pyamplicol.generation.dag_color import ColorEngine
from pyamplicol.generation.dag_ordering import (
    _canonical_sink_mask,
    _lc_color_order_reachable_masks,
    _ordered_combination_segment,
    _shared_single_trace_closure_matches_word,
)
from pyamplicol.generation.dag_types import ColorState, CurrentIndex
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model, Particle, Vertex
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.processes.ir import (
    CanonicalProcessIR,
    ColorEndpointSummary,
    ProcessLegIR,
)


class _StructuralModel(Model):
    def __init__(
        self,
        *,
        particles: Iterable[Particle],
        representations: Mapping[int, int],
        fermions: Iterable[int] = (),
        massless_adjoint_vectors: Iterable[int] | None = (),
        vertices: Iterable[Vertex] = (),
        auxiliary_kinds: Mapping[int, str] | None = None,
        trace_reflection_proven: bool = False,
        single_trace_basis_proven: bool = False,
    ) -> None:
        particle_tuple = tuple(particles)
        super().__init__(
            name="relabeled-structural-model",
            particles={particle.pdg: particle for particle in particle_tuple},
            vertices=tuple(vertices),
        )
        self._representations = dict(representations)
        self._fermions = frozenset(fermions)
        self._massless_adjoint_vectors = (
            None
            if massless_adjoint_vectors is None
            else frozenset(massless_adjoint_vectors)
        )
        self._auxiliary_kinds = dict(auxiliary_kinds or {})
        self._trace_reflection_proven = bool(trace_reflection_proven)
        self._single_trace_basis_proven = bool(single_trace_basis_proven)

    def color_rep(self, pdg: int) -> int:
        return self._representations[int(pdg)]

    def is_fermion(self, pdg: int) -> bool:
        return int(pdg) in self._fermions

    def is_massless_adjoint_vector(self, pdg: int) -> bool:
        if self._massless_adjoint_vectors is None:
            raise NotImplementedError("adjoint-vector role is not proven")
        return int(pdg) in self._massless_adjoint_vectors

    def auxiliary_kind(self, particle_id: int) -> str | None:
        return self._auxiliary_kinds.get(int(particle_id))

    def lc_trace_reflection_equivalence_is_proven(self, process: object) -> bool:
        del process
        return self._trace_reflection_proven

    def shared_single_trace_color_basis_is_proven(self, process: object) -> bool:
        del process
        return self._single_trace_basis_proven


def _process(
    pdgs: tuple[int, ...],
    *,
    statistics: tuple[str, ...] | None = None,
    wavefunction_families: tuple[str, ...] | None = None,
    color_roles: tuple[str, ...] | None = None,
    source_orientations: tuple[str, ...] | None = None,
    color_accuracy: str = "lc",
) -> CanonicalProcessIR:
    names = tuple(f"state_{index}" for index in range(1, len(pdgs) + 1))
    statistics = statistics or ("boson",) * len(pdgs)
    wavefunction_families = wavefunction_families or ("scalar",) * len(pdgs)
    color_roles = color_roles or ("singlet",) * len(pdgs)
    source_orientations = source_orientations or ("self-conjugate",) * len(pdgs)
    legs = tuple(
        ProcessLegIR(
            label=index,
            side="initial" if index <= 2 else "final",
            particle=names[index - 1],
            outgoing_particle=names[index - 1],
            pdg=pdg,
            outgoing_pdg=pdg,
            statistics=statistics[index - 1],
            wavefunction_family=wavefunction_families[index - 1],
            color_role=color_roles[index - 1],
            source_orientation=source_orientations[index - 1],
        )
        for index, pdg in enumerate(pdgs, start=1)
    )
    return CanonicalProcessIR(
        process=f"{' '.join(names[:2])} > {' '.join(names[2:])}",
        key="relabeled_process",
        color_accuracy=color_accuracy,
        legs=legs,
        color_endpoints=ColorEndpointSummary(
            fundamental_count=color_roles.count("fundamental"),
            antifundamental_count=color_roles.count("antifundamental"),
            pair_count=min(
                color_roles.count("fundamental"),
                color_roles.count("antifundamental"),
            ),
        ),
    )


def _single_trace_plan(process: CanonicalProcessIR) -> GenericColorPlan:
    return GenericColorPlan(
        process=process,
        color_accuracy=process.color_accuracy,
        sectors=(
            LCColorSector(
                id=0,
                kind="single-trace",
                trace_labels=tuple(leg.label for leg in process.legs),
            ),
        ),
    )


def _adjoint_index(
    labels: tuple[int, ...],
    *,
    ordered: tuple[int, ...],
    accuracy: str = "full",
) -> CurrentIndex:
    mask = sum(1 << (label - 1) for label in labels)
    return CurrentIndex(
        particle_id=21,
        external_mask=mask,
        external_labels=tuple(sorted(labels)),
        ordered_external_labels=ordered,
        helicity_ancestry=mask,
        chirality=0,
        spin_state=0,
        flavour_flow=(),
        quantum_number_flow=(),
        color_state=ColorState(
            accuracy=accuracy,
            sector_id=0,
            line_groups=(0,),
        ),
        momentum_mask=mask,
    )


def test_sink_selection_uses_model_fermion_statistics_not_process_metadata() -> None:
    scalar = 710_001
    fermion = 810_001
    model = _StructuralModel(
        particles=(
            Particle(scalar, scalar, spin=1, dimension=1, color_rep=1),
            Particle(fermion, -fermion, spin=2, dimension=4, color_rep=1),
        ),
        representations={scalar: 1, fermion: 1, -fermion: 1},
        fermions=(fermion, -fermion),
    )
    process = _process(
        (scalar, fermion, scalar),
        statistics=("fermion", "boson", "boson"),
    )

    assert _canonical_sink_mask(process, model) == 1 << 1


def test_relabelled_adjoint_symmetry_uses_model_role_and_fails_closed() -> None:
    adjoint = 910_101
    particle = Particle(adjoint, adjoint, spin=3, dimension=4, color_rep=8)
    lc_process = _process(
        (adjoint,) * 4,
    )
    full_process = _process(
        (adjoint,) * 4,
        color_accuracy="full",
    )
    proven_model = _StructuralModel(
        particles=(particle,),
        representations={adjoint: 8},
        massless_adjoint_vectors=(adjoint,),
        single_trace_basis_proven=True,
    )

    proven_engine = ColorEngine(
        _single_trace_plan(lc_process),
        proven_model,
        shared_lc_all_ordering_symmetry=True,
    )
    proven_full_engine = ColorEngine(
        _single_trace_plan(full_process),
        proven_model,
    )

    assert proven_engine.shared_lc_orderings is True
    assert proven_engine.shared_lc_fixed_sink_label == 4
    assert proven_full_engine.shared_single_trace is True
    assert (
        _lc_color_order_reachable_masks(
            lc_process,
            _single_trace_plan(lc_process),
            proven_model,
        )
        is not None
    )

    unproven_model = _StructuralModel(
        particles=(particle,),
        representations={adjoint: 8},
        massless_adjoint_vectors=None,
    )
    unproven_engine = ColorEngine(
        _single_trace_plan(lc_process),
        unproven_model,
        shared_lc_all_ordering_symmetry=True,
    )
    unproven_full_engine = ColorEngine(
        _single_trace_plan(full_process),
        unproven_model,
    )

    assert unproven_engine.shared_lc_orderings is True
    assert unproven_engine.shared_lc_fixed_sink_label is None
    assert unproven_full_engine.shared_single_trace is False
    assert unproven_full_engine.shared_lc_orderings is False

    role_only_model = _StructuralModel(
        particles=(particle,),
        representations={adjoint: 8},
        massless_adjoint_vectors=(adjoint,),
    )
    role_only_full_engine = ColorEngine(
        _single_trace_plan(full_process),
        role_only_model,
    )
    assert role_only_full_engine.shared_single_trace is False
    assert role_only_full_engine.shared_lc_orderings is False


def test_shared_single_trace_indexes_match_word_scan_exactly() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("g g > g g", color_accuracy="full")
    plan = build_color_plan(process, color_accuracy="full")
    engine = ColorEngine(plan, model)
    vertex = next(
        vertex
        for vertex in model.vertices
        if vertex.particles == (21, 21, 21)
    )
    primary_words = tuple(
        word for sector in plan.sectors for word in (sector.color_words or ())
    )

    for ordering in permutations((1, 2, 3, 4)):
        for split in range(1, len(ordering)):
            left_order = tuple(ordering[:split])
            right_order = tuple(ordering[split:])
            left = _adjoint_index(left_order, ordered=left_order)
            right = _adjoint_index(right_order, ordered=right_order)
            expected_segment = next(
                (
                    segment
                    for word in primary_words
                    for segment in (
                        _ordered_combination_segment(
                            left_order,
                            right_order,
                            word,
                        ),
                    )
                    if segment is not None
                ),
                None,
            )
            assert (
                engine.ordered_combination_labels(left, right, vertex)
                == expected_segment
            )

            expected_allowed = any(
                _shared_single_trace_closure_matches_word(
                    left_order,
                    right_order,
                    word,
                )
                for word in primary_words
            )
            assert engine.ordered_closure_allowed(left, right) is expected_allowed
            expected_sector_ids = tuple(
                sector.id
                for sector in plan.sectors
                if any(
                    _shared_single_trace_closure_matches_word(
                        left_order,
                        right_order,
                        word,
                    )
                    for word in sector.admissible_traversal_words
                )
            )
            actual_sector_ids = tuple(
                flow.state.sector_id
                for flow in engine.shared_single_trace_closure_flows(left, right)
            )
            assert actual_sector_ids == expected_sector_ids


def test_color_order_mask_pruning_rejects_plan_roles_not_proven_by_model() -> None:
    adjoint = 910_101
    process = _process(
        (adjoint,) * 4,
    )
    model = _StructuralModel(
        particles=(Particle(adjoint, adjoint, spin=3, dimension=4, color_rep=8),),
        representations={adjoint: 8},
        massless_adjoint_vectors=(adjoint,),
    )
    mismatched_plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="single-trace",
                trace_labels=(1, 2, 3),
                singlet_labels=(4,),
            ),
        ),
    )

    assert _lc_color_order_reachable_masks(process, mismatched_plan, model) is None


def test_relabelled_fundamental_line_auxiliary_requires_model_declaration() -> None:
    fermion = 810_001
    adjoint = 910_101
    auxiliary = 770_001
    auxiliary_vertex = Vertex(1, (fermion, -fermion, auxiliary))
    model = _StructuralModel(
        particles=(
            Particle(fermion, -fermion, spin=2, dimension=4, color_rep=3),
            Particle(adjoint, adjoint, spin=3, dimension=4, color_rep=8),
            Particle(auxiliary, auxiliary, spin=-1, dimension=4, color_rep=1),
        ),
        representations={
            fermion: 3,
            -fermion: -3,
            adjoint: 8,
            auxiliary: 1,
        },
        fermions=(fermion, -fermion),
        massless_adjoint_vectors=(adjoint,),
        vertices=(auxiliary_vertex,),
        auxiliary_kinds={auxiliary: "u1-subtraction-color-flow-vector"},
    )
    one_line_process = _process(
        (-fermion, fermion, adjoint, adjoint),
    )
    two_line_process = _process(
        (-fermion, fermion, fermion, -fermion),
    )

    one_line_engine = ColorEngine(
        GenericColorPlan(one_line_process, "lc", ()),
        model,
    )
    two_line_engine = ColorEngine(
        GenericColorPlan(two_line_process, "lc", ()),
        model,
    )

    assert one_line_engine._particle_has_colour(auxiliary) is True
    assert one_line_engine.vertex_allowed(auxiliary_vertex) is False
    assert two_line_engine.vertex_allowed(auxiliary_vertex) is True

    unrelated_model = _StructuralModel(
        particles=model.particles.values(),
        representations=model._representations,
        fermions=(fermion, -fermion),
        massless_adjoint_vectors=(adjoint,),
        vertices=(auxiliary_vertex,),
        auxiliary_kinds={auxiliary: "ufo-contact:unrelated-vector"},
    )
    unrelated_engine = ColorEngine(
        GenericColorPlan(one_line_process, "lc", ()),
        unrelated_model,
    )
    assert unrelated_engine._particle_has_colour(auxiliary) is False


def test_trace_reflection_folding_requires_model_proof() -> None:
    adjoint = 910_101
    process = _process(
        (adjoint,) * 5,
        wavefunction_families=("vector",) * 5,
        color_roles=("adjoint",) * 5,
    )
    unfolded = build_color_plan(process)
    folded = build_color_plan(process, fold_trace_reflections=True)

    assert unfolded.trace_reflections_folded is False
    assert folded.trace_reflections_folded is True
    assert unfolded.sector_count == 24
    assert folded.sector_count == 12


def test_compiler_requires_trace_proof_for_all_adjoint_fixed_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[bool, int | None]] = []
    color_engine_type = dag_compiler_module.ColorEngine

    class RecordingColorEngine(color_engine_type):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            observed.append(
                (
                    self.shared_lc_all_ordering_symmetry,
                    self.shared_lc_fixed_sink_label,
                )
            )

    class UnprovenBuiltinModel(BuiltinSMModel):
        def lc_trace_reflection_equivalence_is_proven(
            self,
            process: object,
        ) -> bool:
            del process
            return False

    monkeypatch.setattr(
        dag_compiler_module,
        "ColorEngine",
        RecordingColorEngine,
    )
    process = build_process_ir("g g > g g")

    dag_compiler_module.compile_generic_dag(
        process,
        model=UnprovenBuiltinModel(),
    )
    dag_compiler_module.compile_generic_dag(
        process,
        model=BuiltinSMModel(),
    )

    assert observed == [(False, None), (True, 4)]


def test_pure_adjoint_helicity_zero_pruning_requires_its_own_model_proof() -> None:
    class ParityOnlyBuiltinModel(BuiltinSMModel):
        def pure_massless_adjoint_helicity_zero_rule_is_proven(
            self,
            process: object,
            vertices: Iterable[Vertex],
        ) -> bool:
            del process, vertices
            return False

    process = build_process_ir("g g > g g")
    dag = dag_compiler_module.compile_generic_dag(
        process,
        model=BuiltinSMModel(),
    )

    parity_only = prune_global_helicity_flip_equivalent_roots(
        dag,
        ParityOnlyBuiltinModel(),
    )
    proven = prune_global_helicity_flip_equivalent_roots(
        dag,
        BuiltinSMModel(),
    )

    assert len(parity_only.amplitude_roots) == 24
    assert len(proven.amplitude_roots) == 9
    assert sum(root.helicity_weight for root in parity_only.amplitude_roots) == 48.0
    assert sum(root.helicity_weight for root in proven.amplitude_roots) == 18.0


def test_builtin_sink_adjoint_and_color_auxiliary_behavior_is_preserved() -> None:
    model = BuiltinSMModel()
    leptonic_process = build_process_ir("e- e+ > mu- mu+")
    gluon_process = build_process_ir("g g > g g")
    gluon_engine = ColorEngine(
        build_color_plan(gluon_process),
        model,
        shared_lc_all_ordering_symmetry=True,
    )
    full_gluon_process = build_process_ir("g g > g g", color_accuracy="full")
    full_gluon_engine = ColorEngine(
        build_color_plan(full_gluon_process, color_accuracy="full"),
        model,
    )
    auxiliary_vertex = next(
        vertex for vertex in model.vertices if vertex.particles == (1, -1, 99)
    )
    one_line_process = build_process_ir("d d~ > g g")
    two_line_process = build_process_ir("d d~ > d d~")
    one_line_engine = ColorEngine(build_color_plan(one_line_process), model)
    two_line_engine = ColorEngine(build_color_plan(two_line_process), model)

    assert _canonical_sink_mask(leptonic_process, model) == 1
    assert gluon_engine.shared_lc_fixed_sink_label == 4
    assert full_gluon_engine.shared_single_trace is True
    assert one_line_engine._particle_has_colour(99) is True
    assert one_line_engine.vertex_allowed(auxiliary_vertex) is False
    assert two_line_engine.vertex_allowed(auxiliary_vertex) is True
