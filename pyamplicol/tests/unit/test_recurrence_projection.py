# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

import pytest

from pyamplicol.color.plan import (
    GenericColorPlan,
    LCColorSector,
    LCColorSectorReplayPartition,
    LCColorTopologyReplayPlan,
    LCOpenColorLine,
    build_color_plan,
    build_color_topology_replay_certificate,
)
from pyamplicol.generation.recurrence_columnar import (
    ExactComplexRationalV1 as BuilderExact,
)
from pyamplicol.generation.recurrence_columnar import (
    RecurrenceNormalizationV1,
    build_recurrence_builder_input_v1,
)
from pyamplicol.generation.recurrence_physics import (
    recurrence_color_contraction_destinations,
    recurrence_color_sector_owner_map,
)
from pyamplicol.generation.recurrence_projection import (
    RecurrenceGenerationSliceV1,
    RecurrenceProjectionError,
    project_recurrence_process_v1,
)
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.prepared_catalog import build_prepared_kernel_catalog
from pyamplicol.models.recurrence_catalog_builder import (
    build_recurrence_template_catalog,
)
from pyamplicol.models.recurrence_template import (
    CurrentStateTemplateV1,
    EvaluatorBindingV1,
    ExactComplexRationalV1,
    LCColorSourceSeedV1,
    ParameterTemplateV1,
    RecurrenceTemplateCatalog,
    SourceTemplateV1,
    SymmetryProofV1,
)
from pyamplicol.processes.ir import (
    CanonicalProcessIR,
    ColorEndpointSummary,
    ParticleStatistics,
    ProcessLegIR,
    WavefunctionFamily,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _crossing(
    *,
    momentum_transform: str = "negate-four-momentum",
    phase: tuple[int, int] = (1, 0),
) -> str:
    return json.dumps(
        {
            "chirality_factor": -1,
            "helicity_factor": -1,
            "momentum_transform": momentum_transform,
            "phase": {
                "imag_denominator": "1",
                "imag_numerator": str(phase[1]),
                "real_denominator": "1",
                "real_numerator": str(phase[0]),
            },
            "spin_state_factor": -1,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _state(
    template_id: str,
    particle_id: int,
    *,
    family: str,
    dimension: int,
) -> CurrentStateTemplateV1:
    color_representation = 8 if particle_id == 21 else 3 if particle_id > 0 else -3
    color_shape = (
        "adjoint-segment"
        if particle_id == 21
        else "fundamental-open-string"
        if particle_id > 0
        else "antifundamental-open-string"
    )
    return CurrentStateTemplateV1(
        template_id=template_id,
        particle_id=particle_id,
        anti_particle_id=-particle_id if abs(particle_id) == 1 else particle_id,
        species_id=f"species:{abs(particle_id)}",
        orientation=(
            "self-conjugate"
            if particle_id == 21
            else "particle"
            if particle_id > 0
            else "antiparticle"
        ),
        statistics="fermion" if family == "fermion" else "boson",
        color_representation=color_representation,
        basis=family,
        tensor_ordering=tuple(f"{family}:c{index}" for index in range(dimension)),
        dimension=dimension,
        chirality=0,
        lc_color_shape_kind=color_shape,
        auxiliary_kind=None,
        mass_parameter_id=None,
        width_parameter_id=None,
    )


def _source(
    state: CurrentStateTemplateV1,
    *,
    helicity: int,
    family: str,
) -> tuple[SourceTemplateV1, EvaluatorBindingV1]:
    token = f"source:{state.particle_id}:{helicity}"
    resolver = f"resolver:{state.particle_id}:{helicity}"
    signature = _sha(f"callable:{token}")
    source = SourceTemplateV1(
        template_id=token,
        state_template_id=state.template_id,
        crossing=_crossing(),
        wavefunction_family=family,
        helicity=helicity,
        spin_state=helicity,
        flavour_flow=(state.particle_id,),
        quantum_number_flow=(),
        lc_color_seed=LCColorSourceSeedV1(
            operation="singleton",
            output_shape_kind=state.lc_color_shape_kind,
            component_kind=(
                "adjoint-segment"
                if state.lc_color_shape_kind == "adjoint-segment"
                else "open-string"
            ),
            component_role="active",
            proof_digest=_sha(f"source-seed:{state.template_id}"),
            provenance=(("test-fixture", "model-generic-source-contract"),),
        ),
        wavefunction_expression_digest=signature,
        evaluator_resolver_key=resolver,
    )
    output_layout = tuple(
        f"source-component:{index}" for index in range(state.dimension)
    )
    binding = EvaluatorBindingV1(
        resolver_key=resolver,
        prepared_kernel_id=None,
        callable_kind="rusticol-template",
        runtime_template=(f"rusticol.source-fill.{family}.v1:{signature[:24]}"),
        contract_kind="source",
        callable_signature=signature,
        input_state_template_ids=(),
        output_state_template_id=state.template_id,
        input_layout=("momentum:energy",),
        output_layout=output_layout,
        exact_expression_digests=tuple(
            _sha(f"expression:{token}:{index}") for index in range(state.dimension)
        ),
        semantic_template_ids=(source.template_id,),
    )
    return source, binding


def _catalog(*, omit_particle: int | None = None) -> RecurrenceTemplateCatalog:
    states = (
        _state("state:anti-u", -1, family="fermion", dimension=2),
        _state("state:u", 1, family="fermion", dimension=2),
        _state("state:g", 21, family="vector", dimension=4),
    )
    source_pairs = tuple(
        _source(
            state,
            helicity=helicity,
            family="vector" if state.particle_id == 21 else "fermion",
        )
        for state in states
        if state.particle_id != omit_particle
        for helicity in (-1, 1)
    )
    parameter = ParameterTemplateV1(
        template_id="parameter:alpha_s",
        name="alpha_s",
        parameter_kind="external",
        value_type="real",
        mutable=True,
        default_value=ExactComplexRationalV1(1, 10, 0, 1),
        exact_expression_digest=None,
        dependency_parameter_ids=(),
    )
    return RecurrenceTemplateCatalog.create(
        compiled_model_digest=_sha("compiled-model"),
        prepared_kernel_pack_digest=_sha("prepared-kernel-pack"),
        parameters=(parameter,),
        current_states=states,
        sources=tuple(pair[0] for pair in source_pairs),
        evaluator_bindings=tuple(pair[1] for pair in source_pairs),
    )


def _catalog_with_reflection_phases(
    *phases: int,
) -> RecurrenceTemplateCatalog:
    base = _catalog()
    proofs = tuple(
        SymmetryProofV1(
            template_id=f"proof:current-reflection:{index}",
            proof_algorithm="canonical-current-word-reversal-v1",
            subject_template_ids=("state:g",),
            input_permutation=(1, 0),
            exact_phase=ExactComplexRationalV1.from_fractions(phase),
            expression_digests=(_sha(f"reflection-expression:{index}"),),
            witness_digest=_sha(f"reflection-witness:{index}"),
        )
        for index, phase in enumerate(phases)
    )
    return RecurrenceTemplateCatalog.create(
        compiled_model_digest=base.header.compiled_model_digest,
        prepared_kernel_pack_digest=base.header.prepared_kernel_pack_digest,
        parameters=base.parameters,
        current_states=base.current_states,
        sources=base.sources,
        quantum_flows=base.quantum_flows,
        transitions=base.transitions,
        propagators=base.propagators,
        closures=base.closures,
        color_contractions=base.color_contractions,
        symmetry_proofs=proofs,
        runtime_helicity_contracts=base.runtime_helicity_contracts,
        evaluator_bindings=base.evaluator_bindings,
    )


def _process(*, color_accuracy: str = "lc") -> CanonicalProcessIR:
    return CanonicalProcessIR(
        process="u u~ > g g",
        key="u_ubar_to_g_g",
        color_accuracy=color_accuracy,
        legs=(
            ProcessLegIR(
                1,
                "initial",
                "u",
                "u~",
                1,
                -1,
                "fermion",
                "fermion",
                "antifundamental",
                "antiparticle",
            ),
            ProcessLegIR(
                2,
                "initial",
                "u~",
                "u",
                -1,
                1,
                "fermion",
                "fermion",
                "fundamental",
                "particle",
            ),
            ProcessLegIR(
                3,
                "final",
                "g",
                "g",
                21,
                21,
                "boson",
                "vector",
                "adjoint",
                "self-conjugate",
            ),
            ProcessLegIR(
                4,
                "final",
                "g",
                "g",
                21,
                21,
                "boson",
                "vector",
                "adjoint",
                "self-conjugate",
            ),
        ),
        color_endpoints=ColorEndpointSummary(1, 1, 1),
    )


def _pure_gluon_process() -> CanonicalProcessIR:
    legs = tuple(
        ProcessLegIR(
            label,
            "initial" if label <= 2 else "final",
            "g",
            "g",
            21,
            21,
            "boson",
            "vector",
            "adjoint",
            "self-conjugate",
        )
        for label in range(1, 5)
    )
    return CanonicalProcessIR(
        process="g g > g g",
        key="g_g_to_g_g",
        color_accuracy="lc",
        legs=legs,
        color_endpoints=ColorEndpointSummary(0, 0, 0),
    )


def _all_singlet_process() -> CanonicalProcessIR:
    process = _process()
    return replace(
        process,
        key="all_singlet_process",
        legs=tuple(replace(leg, color_role="singlet") for leg in process.legs),
        color_endpoints=ColorEndpointSummary(0, 0, 0),
    )


def _all_singlet_boson_process() -> CanonicalProcessIR:
    process = _pure_gluon_process()
    return replace(
        process,
        key="all_singlet_boson_process",
        legs=tuple(replace(leg, color_role="singlet") for leg in process.legs),
        color_endpoints=ColorEndpointSummary(0, 0, 0),
    )


def _color_plan(
    process: CanonicalProcessIR,
    *,
    truncated: bool = False,
    bad_label: bool = False,
) -> GenericColorPlan:
    final_label = 99 if bad_label else 4
    return GenericColorPlan(
        process=process,
        color_accuracy=process.color_accuracy,
        sectors=(
            LCColorSector(
                id=0,
                kind="open-lines",
                open_color_lines=(LCOpenColorLine(2, 1, (3, final_label)),),
                word_labels=(2, 3, final_label, 1),
            ),
            LCColorSector(
                id=1,
                kind="open-lines",
                open_color_lines=(LCOpenColorLine(2, 1, (final_label, 3)),),
                word_labels=(2, final_label, 3, 1),
            ),
        ),
        truncated=truncated,
    )


def _replay() -> LCColorTopologyReplayPlan:
    return LCColorTopologyReplayPlan(
        physical_sector_ids=(0, 1),
        partitions=(
            LCColorSectorReplayPartition(
                representative_sector_id=0,
                materialized_sector_id=0,
                active_sector_ids=(0, 1),
                label_permutations=((), ((3, 4), (4, 3))),
                replay_weights=(1.0, 1.0),
                replay_signs=(1, -1),
                proof_algorithm="canonical-recurrence-replay-witness-v1",
                proof_digest=_sha("replay-proof"),
            ),
        ),
        residual_sector_ids=(),
    )


def _normalization() -> RecurrenceNormalizationV1:
    return RecurrenceNormalizationV1(
        factor=BuilderExact(1, 4),
        convention="spin-colour-average-v1",
        semantic_digest=_sha("normalization"),
    )


def test_topology_replay_projects_exact_axes_and_generation_coverage() -> None:
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="topology-replay",
        topology_replay=_replay(),
        normalization=_normalization(),
        generation_slice=RecurrenceGenerationSliceV1(
            selected_public_flow_ids=(1,),
            # Initial-state crossing maps the catalog's -1 source to +1.
            selected_source_helicities=((1, 1),),
        ),
        coupling_order_limits={"qcd": 4},
    )

    assert logical.process_id == "u_ubar_to_g_g"
    assert tuple(leg.public_label for leg in logical.external_legs) == (1, 2, 3, 4)
    assert tuple(len(leg.source_states) for leg in logical.external_legs) == (
        2,
        2,
        2,
        2,
    )
    assert tuple(leg.is_initial for leg in logical.external_legs) == (
        True,
        True,
        False,
        False,
    )
    assert tuple(
        state.public_helicity for state in logical.external_legs[0].source_states
    ) == (-1, 1)
    assert all(
        state.crossing_phase == BuilderExact(1)
        for state in logical.external_legs[0].source_states
    )
    assert all(
        state.momentum_sign == -1 for state in logical.external_legs[0].source_states
    )
    assert logical.selected_public_flow_ids == (1,)
    assert logical.selected_source_coverage is not None
    assert logical.selected_source_coverage[0].source_slot == 0
    assert logical.selected_source_coverage[0].source_state_indices == (1,)
    assert tuple(sector.public_id for sector in logical.physical_sectors) == (
        "flow:2,3,4,1",
        "flow:2,4,3,1",
    )
    assert tuple(sector.closure_source_slot for sector in logical.physical_sectors) == (
        0,
        0,
    )
    assert {sector.closure_proof_algorithm for sector in logical.physical_sectors} == {
        "canonical-lc-closure-anchor-v3"
    }
    assert all(
        len(sector.closure_proof_digest) == 64 for sector in logical.physical_sectors
    )
    assert (
        len({sector.closure_proof_digest for sector in logical.physical_sectors}) == 2
    )
    assert len(logical.replay_partitions) == 1
    replayed = sorted(
        logical.replay_partitions[0].targets,
        key=lambda target: target.sector_id,
    )
    assert replayed[0].external_permutation == (0, 1, 2, 3)
    assert replayed[1].external_permutation == (0, 1, 3, 2)
    assert replayed[1].source_slot_permutation == (0, 1, 3, 2)
    assert replayed[1].fermion_sign == -1
    assert logical.coupling_limits[0].name == "QCD"
    assert logical.parameter_projection[0].runtime_name == "alpha_s"
    assert logical.parameter_projection[0].prepared_parameter_id is None

    # The result is accepted unchanged by the existing deterministic encoder.
    assert len(build_recurrence_builder_input_v1(logical).canonical_digest) == 64


def test_projection_binds_global_helicity_flip_proof_only_for_complete_axes() -> None:
    process = _process()
    common = {
        "process": process,
        "color_plan": _color_plan(process),
        "template_catalog": _catalog(),
        "layout": "topology-replay",
        "topology_replay": _replay(),
        "normalization": _normalization(),
        "coupling_order_limits": {"qcd": 4},
        "model": BuiltinSMModel(),
    }

    complete = project_recurrence_process_v1(**common)
    specialized = project_recurrence_process_v1(
        **common,
        generation_slice=RecurrenceGenerationSliceV1(
            selected_source_helicities=((1, 1),),
        ),
    )

    role = "helicity-equivalence:global-flip-v1"
    complete_proofs = tuple(
        item for item in complete.semantic_digests if item.role == role
    )
    assert len(complete_proofs) == 1
    assert len(complete_proofs[0].digest) == 64
    assert all(item.role != role for item in specialized.semantic_digests)


def test_projection_global_flip_proof_rejects_massive_fundamental_fermions() -> None:
    """Massive helicity partners are not equal at a fixed phase-space point.

    This is an adversarial boundary regression: parity relates a massive
    fermion amplitude to parity-transformed kinematics, so the massless
    fixed-point global-flip reduction must remain disabled even when every
    reachable interaction is QCD and parity preserving.
    """

    class MassiveFundamentalModel(BuiltinSMModel):
        def mass(self, pdg: int) -> float:
            if abs(int(pdg)) == 1:
                return 173.0
            return super().mass(pdg)

    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="topology-replay",
        topology_replay=_replay(),
        normalization=_normalization(),
        coupling_order_limits={"qcd": 4},
        model=MassiveFundamentalModel(),
    )

    assert all(
        item.role != "helicity-equivalence:global-flip-v1"
        for item in logical.semantic_digests
    )


def test_projection_attaches_exact_fixed_width_fermion_pairing_tables() -> None:
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="topology-replay",
        topology_replay=_replay(),
        normalization=_normalization(),
    )

    pairing = logical.fermion_pairing_catalog
    assert pairing is not None
    assert pairing.process_key == process.key
    assert pairing.source_count == len(process.legs)
    assert tuple(endpoint.source_slot for endpoint in pairing.endpoints) == (0, 1)
    assert tuple(endpoint.color_orientation for endpoint in pairing.endpoints) == (
        "antifundamental",
        "fundamental",
    )
    assert pairing.rules[0].endpoint_pairings == ((1, 0),)
    digest_roles = {item.role: item.digest for item in logical.semantic_digests}
    assert digest_roles["fermion-pairing-semantic"] == pairing.semantic_digest
    assert digest_roles["fermion-pairing-topology"] == pairing.topology_digest

    encoded = build_recurrence_builder_input_v1(logical)
    assert encoded.fermion_pairing_digest is not None
    assert len(encoded.fermion_pairing_digest) == 64
    header = encoded.fermion_pairing_table("header")
    assert int(header.column("source_count")[0]) == 4
    assert int(header.column("endpoint_count")[0]) == 2
    assert int(header.column("pairing_class_count")[0]) == 1
    assert int(header.column("rule_count")[0]) == 1
    endpoints = encoded.fermion_pairing_table("endpoints")
    assert tuple(int(value) for value in endpoints.column("source_slot")) == (0, 1)
    for table in encoded.fermion_pairing_tables:
        for column in table.columns:
            assert column.values.dtype.kind != "O"
            assert column.values.flags.c_contiguous
            assert column.values.flags.owndata
            assert not column.values.flags.writeable


def test_all_flow_union_retains_all_sectors_without_replay_partitions() -> None:
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="all-flow-union",
        topology_replay=_replay(),
        normalization=_normalization(),
    )

    assert logical.selected_public_flow_ids is None
    assert logical.selected_source_coverage is None
    assert tuple(sector.sector_id for sector in logical.physical_sectors) == (0, 1)
    assert logical.replay_partitions == ()
    encoded = build_recurrence_builder_input_v1(logical)
    assert encoded.table("selected_public_flow_coverage").row_count == 0
    assert encoded.table("selected_source_coverage").row_count == 0
    assert len(encoded.canonical_digest) == 64


def test_contracted_pure_gluon_replay_projects_six_n5_template_orbits() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("g g > g g g g g", color_accuracy="full")
    color_plan = build_color_plan(process, color_accuracy="full")
    replay = build_color_topology_replay_certificate(color_plan, model)

    assert replay is not None
    logical = project_recurrence_process_v1(
        process,
        color_plan,
        _catalog(),
        layout="contracted-color-union",
        topology_replay=replay,
        normalization=_normalization(),
        model=model,
    )

    assert len(logical.physical_sectors) == 720
    assert len(logical.replay_partitions) == 6
    assert tuple(
        partition.materialized_sector_id for partition in logical.replay_partitions
    ) == (0, 120, 144, 150, 152, 153)
    assert {len(partition.targets) for partition in logical.replay_partitions} == {120}
    assert {
        target.sector_id
        for partition in logical.replay_partitions
        for target in partition.targets
    } == set(range(720))
    assert all(
        target.fermion_sign == 1
        for partition in logical.replay_partitions
        for target in partition.targets
    )
    encoded = build_recurrence_builder_input_v1(logical)
    assert encoded.table("replay_partitions").row_count == 6
    assert encoded.table("replay_targets").row_count == 720
    contraction_destinations = recurrence_color_contraction_destinations(
        logical,
        ((-1, -1), (-1, 1), (1, -1), (1, 1)),
        tuple(
            (sector_id, helicity_id)
            for sector_id in replay.materialized_sector_ids
            for helicity_id in range(4)
        ),
    )
    assert len(contraction_destinations) == 720 * 4
    assert contraction_destinations[:4] == ((0, 0), (0, 1), (0, 2), (0, 3))
    assert contraction_destinations[-1] == (719, 3)


def test_symmetric_group_anchor_falls_back_for_complete_contracted_replay() -> None:
    model = BuiltinSMModel()
    process = build_process_ir("g g > g g", color_accuracy="full")
    color_plan = build_color_plan(process, color_accuracy="full")
    replay = build_color_topology_replay_certificate(color_plan, model)
    catalog = build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest=_sha("closure-anchor-compiled-model"),
        prepared_kernel_pack_digest=_sha("closure-anchor-prepared-pack"),
    )
    common = {
        "process": process,
        "color_plan": color_plan,
        "template_catalog": catalog,
        "layout": "contracted-color-union",
        "topology_replay": replay,
        "normalization": _normalization(),
        "model": model,
    }
    direct = project_recurrence_process_v1(**common)
    anchored = project_recurrence_process_v1(
        **common,
        prefer_symmetric_group_closure_anchor=True,
    )

    assert {sector.closure_proof_algorithm for sector in direct.physical_sectors} == {
        "canonical-lc-closure-anchor-v2"
    }
    assert anchored.physical_sectors == direct.physical_sectors
    assert anchored.public_flows == direct.public_flows
    assert anchored.replay_partitions == direct.replay_partitions
    assert anchored.replay_partitions
    build_recurrence_builder_input_v1(anchored)


def test_symmetric_group_anchor_is_independent_of_particle_21_identity() -> None:
    synthetic_particle_id = 9_000_001
    model = BuiltinSMModel()
    base_catalog = build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest=_sha("synthetic-adjoint-compiled-model"),
        prepared_kernel_pack_digest=_sha("synthetic-adjoint-prepared-pack"),
    )
    base_state = next(
        state
        for state in base_catalog.current_states
        if state.particle_id == 21
        and state.orientation == "self-conjugate"
        and state.color_representation == 8
    )
    state = replace(
        base_state,
        particle_id=synthetic_particle_id,
        anti_particle_id=synthetic_particle_id,
        species_id="synthetic-adjoint",
        semantic_digest="",
    )
    sources = tuple(
        replace(
            source,
            flavour_flow=(synthetic_particle_id,),
            semantic_digest="",
        )
        for source in base_catalog.sources
        if source.state_template_id == state.template_id
    )
    closure = next(
        item
        for item in base_catalog.closures
        if item.input_state_template_ids == (state.template_id, state.template_id)
        and item.result_state_template_id is None
        and not item.coupling_parameter_ids
        and not item.coupling_orders
        and not item.eligible_quantum_flow_template_ids
        and item.output_factor_source == "none"
        and item.equivalence_class == "direct-contraction"
    )
    resolver_keys = {
        closure.evaluator_resolver_key,
        *(source.evaluator_resolver_key for source in sources),
    }
    catalog = RecurrenceTemplateCatalog.create(
        compiled_model_digest=base_catalog.header.compiled_model_digest,
        prepared_kernel_pack_digest=(base_catalog.header.prepared_kernel_pack_digest),
        current_states=(state,),
        sources=sources,
        closures=(closure,),
        color_contractions=tuple(
            item
            for item in base_catalog.color_contractions
            if item.template_id == closure.color_contraction_template_id
        ),
        evaluator_bindings=tuple(
            item
            for item in base_catalog.evaluator_bindings
            if item.resolver_key in resolver_keys
        ),
    )
    process = replace(
        _pure_gluon_process(),
        process="x x > x x",
        key="synthetic_adjoint_scattering",
        legs=tuple(
            replace(
                leg,
                particle="x",
                outgoing_particle="x",
                pdg=synthetic_particle_id,
                outgoing_pdg=synthetic_particle_id,
            )
            for leg in _pure_gluon_process().legs
        ),
    )
    plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="single-trace",
                trace_labels=(1, 2, 3, 4),
                word_labels=(1, 2, 3, 4),
            ),
        ),
    )
    logical = project_recurrence_process_v1(
        process,
        plan,
        catalog,
        layout="all-flow-union",
        normalization=_normalization(),
        prefer_symmetric_group_closure_anchor=True,
    )

    assert {leg.physical_pdg for leg in logical.external_legs} == {
        synthetic_particle_id
    }
    assert logical.physical_sectors[0].closure_proof_algorithm == (
        "canonical-lc-closure-anchor-v4"
    )
    assert logical.physical_sectors[0].closure_source_slot == 0


def test_symmetric_group_anchor_falls_back_without_exact_template_contract() -> None:
    process = _pure_gluon_process()
    plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="single-trace",
                trace_labels=(1, 2, 3, 4),
                word_labels=(1, 2, 3, 4),
            ),
        ),
    )
    logical = project_recurrence_process_v1(
        process,
        plan,
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
        prefer_symmetric_group_closure_anchor=True,
    )

    assert logical.physical_sectors[0].closure_proof_algorithm == (
        "canonical-lc-closure-anchor-v2"
    )
    assert logical.physical_sectors[0].closure_source_slot == 3


def test_symmetric_group_anchor_preserves_high_n_factorial_trace_domains() -> None:
    model = BuiltinSMModel()
    catalog = build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest=_sha("closure-anchor-count-compiled-model"),
        prepared_kernel_pack_digest=_sha("closure-anchor-count-prepared-pack"),
    )
    for total_gluons in (7, 8):
        process = build_process_ir(
            "g g > " + " ".join("g" for _ in range(total_gluons - 2)),
            color_accuracy="full",
        )
        color_plan = build_color_plan(process, color_accuracy="full")
        logical = project_recurrence_process_v1(
            process,
            color_plan,
            catalog,
            layout="contracted-color-union",
            normalization=_normalization(),
            model=model,
            prefer_symmetric_group_closure_anchor=True,
        )

        assert len(logical.physical_sectors) == math.factorial(total_gluons - 1)
        assert all(
            sector.closure_proof_algorithm == "canonical-lc-closure-anchor-v4"
            and sector.closure_source_slot == 0
            for sector in logical.physical_sectors
        )
        assert logical.replay_partitions == ()


def test_contracted_color_owner_map_rejects_an_unproved_missing_sector() -> None:
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
    )

    with pytest.raises(ValueError, match="no complete structural-zero certificate"):
        recurrence_color_sector_owner_map(logical, {0}, set())


def test_contracted_color_owner_map_preserves_open_string_singlets() -> None:
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
    )
    first, second = logical.physical_sectors
    second_string = replace(
        second.open_strings[0],
        adjoint_source_slots=first.open_strings[0].adjoint_source_slots,
        singlet_source_slots=(3,),
    )
    logical = replace(
        logical,
        physical_sectors=(
            first,
            replace(second, open_strings=(second_string,)),
        ),
    )

    with pytest.raises(ValueError, match="no complete structural-zero certificate"):
        recurrence_color_sector_owner_map(logical, {0}, set())


def test_contracted_color_owner_map_accepts_a_complete_zero_class() -> None:
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
    )

    assert recurrence_color_sector_owner_map(logical, {0}, {1}) == (
        0,
        0xFFFF_FFFF,
    )


def test_contracted_color_owner_map_rejects_partial_zero_owner_class() -> None:
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
    )
    first, second = logical.physical_sectors
    logical = replace(
        logical,
        physical_sectors=(first, replace(second, open_strings=first.open_strings)),
    )

    with pytest.raises(ValueError, match="uncertified sectors=\\[1\\]"):
        recurrence_color_sector_owner_map(logical, set(), {0})
    assert recurrence_color_sector_owner_map(logical, set(), {0, 1}) == (
        0xFFFF_FFFF,
        0xFFFF_FFFF,
    )


def test_contracted_color_owner_map_rejects_active_zero_overlap() -> None:
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
    )

    with pytest.raises(ValueError, match="overlaps active sectors"):
        recurrence_color_sector_owner_map(logical, {0}, {0, 1})


def test_contracted_color_owner_map_rejects_out_of_range_certificate() -> None:
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
    )

    with pytest.raises(ValueError, match="references unknown sectors"):
        recurrence_color_sector_owner_map(logical, {0}, {2})


def test_all_singlet_sector_uses_smallest_fermionic_source_as_closure_anchor() -> None:
    process = _all_singlet_process()
    plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="singlet",
                singlet_labels=tuple(leg.label for leg in process.legs),
            ),
        ),
    )
    logical = project_recurrence_process_v1(
        process,
        plan,
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
    )

    sector = logical.physical_sectors[0]
    assert sector.word_source_slots == ()
    assert sector.closure_source_slot == 0
    assert sector.closure_proof_algorithm == "canonical-lc-closure-anchor-v2"
    assert (
        sector.closure_proof_digest
        == "ce2ac679bcb9dba2688fe07d4a0c6382b28a76e4a91ac24026041782623b07ef"
    )


def test_all_singlet_boson_sector_uses_smallest_source_as_closure_anchor() -> None:
    process = _all_singlet_boson_process()
    plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="singlet",
                singlet_labels=tuple(leg.label for leg in process.legs),
            ),
        ),
    )
    logical = project_recurrence_process_v1(
        process,
        plan,
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
    )

    sector = logical.physical_sectors[0]
    assert sector.closure_source_slot == 0
    assert sector.closure_proof_algorithm == "canonical-lc-closure-anchor-v2"
    assert (
        sector.closure_proof_digest
        == "225a0e4e145b1c0e28719d4670b4379bb602fbbe59fe6c6804ae5694f7b754f2"
    )


@pytest.mark.parametrize(
    ("statistics", "wavefunction_family", "message"),
    [
        ("inclusive", "inclusive", "requires concrete"),
        ("boson", "fermion", "inconsistent"),
    ],
)
def test_closure_anchor_fails_closed_for_ambiguous_leg_spin_metadata(
    statistics: ParticleStatistics,
    wavefunction_family: WavefunctionFamily,
    message: str,
) -> None:
    process = _all_singlet_process()
    process = replace(
        process,
        legs=(
            replace(
                process.legs[0],
                statistics=statistics,
                wavefunction_family=wavefunction_family,
            ),
            *process.legs[1:],
        ),
    )
    plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="singlet",
                singlet_labels=tuple(leg.label for leg in process.legs),
            ),
        ),
    )

    with pytest.raises(RecurrenceProjectionError, match=message):
        project_recurrence_process_v1(
            process,
            plan,
            _catalog(),
            layout="all-flow-union",
            normalization=_normalization(),
        )


def test_projection_does_not_materialize_source_incompatible_anchor_replay() -> None:
    process = _all_singlet_process()
    labels = tuple(leg.label for leg in process.legs)
    plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=tuple(
            LCColorSector(id=sector_id, kind="singlet", singlet_labels=labels)
            for sector_id in range(2)
        ),
    )
    replay = LCColorTopologyReplayPlan(
        physical_sector_ids=(0, 1),
        partitions=(
            LCColorSectorReplayPartition(
                representative_sector_id=0,
                materialized_sector_id=0,
                active_sector_ids=(0, 1),
                label_permutations=((), ((1, 2), (2, 1))),
                replay_weights=(1.0, 1.0),
                replay_signs=(1, 1),
                proof_algorithm="canonical-recurrence-replay-witness-v1",
                proof_digest=_sha("singlet-replay-proof"),
            ),
        ),
        residual_sector_ids=(),
    )

    with pytest.raises(RecurrenceProjectionError, match="crossing contract"):
        project_recurrence_process_v1(
            process,
            plan,
            _catalog(),
            layout="topology-replay",
            topology_replay=replay,
            normalization=_normalization(),
        )


def test_folded_trace_reflection_remains_a_public_runtime_flow() -> None:
    process = _pure_gluon_process()
    plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="single-trace",
                trace_labels=(1, 2, 3, 4),
                word_labels=(1, 2, 3, 4),
            ),
        ),
        trace_reflections_folded=True,
    )
    replay = LCColorTopologyReplayPlan(
        physical_sector_ids=(0,),
        partitions=(),
        residual_sector_ids=(0,),
    )
    logical = project_recurrence_process_v1(
        process,
        plan,
        _catalog_with_reflection_phases(-1),
        layout="topology-replay",
        topology_replay=replay,
        normalization=_normalization(),
    )

    assert tuple(flow.public_id for flow in logical.public_flows) == (
        "flow:1,2,3,4",
        "flow:1,4,3,2",
    )
    assert tuple(flow.construction_sector_id for flow in logical.public_flows) == (
        0,
        0,
    )
    assert logical.public_flows[1].source_slot_permutation == (0, 1, 2, 3)
    assert len(logical.replay_partitions) == 1
    residual = logical.replay_partitions[0]
    assert residual.proof_algorithm == (
        "canonical-recurrence-identity-materialization-v1"
    )
    assert residual.representative_sector_id == residual.materialized_sector_id == 0
    assert residual.targets[0].source_slot_permutation == (0, 1, 2, 3)
    assert len(build_recurrence_builder_input_v1(logical).canonical_digest) == 64


def test_folded_trace_reflection_rejects_missing_current_proof() -> None:
    process = _pure_gluon_process()
    plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="single-trace",
                trace_labels=(1, 2, 3, 4),
                word_labels=(1, 2, 3, 4),
            ),
        ),
        trace_reflections_folded=True,
    )

    with pytest.raises(
        RecurrenceProjectionError,
        match="folded LC trace reflections require an exact model",
    ):
        project_recurrence_process_v1(
            process,
            plan,
            _catalog(),
            layout="all-flow-union",
            normalization=_normalization(),
        )


def test_folded_trace_reflection_rejects_inconsistent_exact_phases() -> None:
    process = _pure_gluon_process()
    plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="single-trace",
                trace_labels=(1, 2, 3, 4),
                word_labels=(1, 2, 3, 4),
            ),
        ),
        trace_reflections_folded=True,
    )

    with pytest.raises(
        RecurrenceProjectionError,
        match="inconsistent exact phases",
    ):
        project_recurrence_process_v1(
            process,
            plan,
            _catalog_with_reflection_phases(-1, 1),
            layout="topology-replay",
            normalization=_normalization(),
        )


@pytest.mark.parametrize("flow_id", [0, 1])
def test_folded_trace_flows_can_be_specialized_independently(flow_id: int) -> None:
    process = _pure_gluon_process()
    plan = GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="single-trace",
                trace_labels=(1, 2, 3, 4),
                word_labels=(1, 2, 3, 4),
            ),
        ),
        trace_reflections_folded=True,
    )
    logical = project_recurrence_process_v1(
        process,
        plan,
        _catalog_with_reflection_phases(-1),
        layout="topology-replay",
        normalization=_normalization(),
        generation_slice=RecurrenceGenerationSliceV1(
            selected_public_flow_ids=(flow_id,)
        ),
    )
    columns = build_recurrence_builder_input_v1(logical)

    assert logical.selected_public_flow_ids == (flow_id,)
    assert tuple(flow.flow_id for flow in logical.public_flows) == (0, 1)
    assert tuple(
        int(value)
        for value in columns.table("selected_public_flow_coverage").column("flow_id")
    ) == (flow_id,)


def test_crossed_source_carries_effective_momentum_transform_and_phase() -> None:
    base = _catalog()
    source = next(
        item
        for item in base.sources
        if item.state_template_id == "state:anti-u" and item.helicity == -1
    )
    changed = replace(
        source,
        crossing=_crossing(momentum_transform="identity", phase=(0, 1)),
        semantic_digest="",
    )
    catalog = RecurrenceTemplateCatalog.create(
        compiled_model_digest=base.header.compiled_model_digest,
        prepared_kernel_pack_digest=base.header.prepared_kernel_pack_digest,
        parameters=base.parameters,
        current_states=base.current_states,
        sources=tuple(changed if item == source else item for item in base.sources),
        evaluator_bindings=base.evaluator_bindings,
    )
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        catalog,
        layout="all-flow-union",
        normalization=_normalization(),
    )

    crossed = next(
        item
        for item in logical.external_legs[0].source_states
        if item.source_template_id
        == sorted(template.template_id for template in catalog.sources).index(
            changed.template_id
        )
    )
    assert crossed.momentum_sign == 1
    assert crossed.crossing_phase == BuilderExact(0, 1, 1, 1)
    assert all(
        state.momentum_sign == 1 and state.crossing_phase == BuilderExact(1)
        for state in logical.external_legs[2].source_states
    )


def test_projection_and_columnar_digest_are_deterministic() -> None:
    process = _process()
    arguments = dict(
        process=process,
        color_plan=_color_plan(process),
        template_catalog=_catalog(),
        layout="topology-replay",
        topology_replay=_replay(),
        normalization=_normalization(),
        coupling_order_limits={"QED": 2, "QCD": 4},
    )
    first = project_recurrence_process_v1(**arguments)
    second = project_recurrence_process_v1(**arguments)

    assert first == second
    first_columns = build_recurrence_builder_input_v1(first)
    second_columns = build_recurrence_builder_input_v1(second)
    assert first_columns.canonical_digest == second_columns.canonical_digest


@pytest.mark.parametrize("accuracy", ["nlc", "full"])
def test_non_lc_projection_requires_contracted_color_union(accuracy: str) -> None:
    process = _process(color_accuracy=accuracy)
    with pytest.raises(
        RecurrenceProjectionError,
        match="requires the contracted-color-union strategy",
    ):
        project_recurrence_process_v1(
            process,
            _color_plan(process),
            _catalog(),
            layout="all-flow-union",
            normalization=_normalization(),
        )

    physical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
        contracted_physical_color_union=True,
    )
    assert physical.layout == "all-flow-union"
    assert physical.replay_partitions == ()
    assert physical.selected_public_flow_ids is None
    assert physical.selected_source_coverage is None


def test_contracted_physical_union_densely_projects_open_line_owners() -> None:
    process = build_process_ir("d d~ > d d~", color_accuracy="full")
    fundamentals = process.fundamental_labels
    antifundamentals = process.antifundamental_labels
    assert len(fundamentals) == len(antifundamentals) == 2
    first = LCOpenColorLine(fundamentals[0], antifundamentals[0], ())
    second = LCOpenColorLine(fundamentals[1], antifundamentals[1], ())
    first_word = (*first.coloured_labels, *second.coloured_labels)
    second_word = (*second.coloured_labels, *first.coloured_labels)
    plan = GenericColorPlan(
        process=process,
        color_accuracy="full",
        sectors=(
            LCColorSector(
                id=0,
                kind="open-lines",
                open_color_lines=(first, second),
                word_labels=first_word,
            ),
            LCColorSector(
                id=1,
                kind="open-lines",
                open_color_lines=(second, first),
                word_labels=second_word,
            ),
        ),
    )

    logical = project_recurrence_process_v1(
        process,
        plan,
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
        contracted_physical_color_union=True,
    )

    assert tuple(sector.sector_id for sector in logical.physical_sectors) == (0,)
    assert tuple(flow.construction_sector_id for flow in logical.public_flows) == (0,)
    assert logical.replay_partitions == ()


def test_projection_rejects_truncated_plan_and_allows_residual_only_replay() -> None:
    process = _process()
    with pytest.raises(RecurrenceProjectionError, match="truncated"):
        project_recurrence_process_v1(
            process,
            _color_plan(process, truncated=True),
            _catalog(),
            layout="all-flow-union",
            normalization=_normalization(),
        )
    residual = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="topology-replay",
        normalization=_normalization(),
    )
    assert residual.replay_partitions == ()


def test_projection_rejects_missing_source_semantics_and_malformed_references() -> None:
    process = _process()
    with pytest.raises(RecurrenceProjectionError, match="no supported source"):
        project_recurrence_process_v1(
            process,
            _color_plan(process),
            _catalog(omit_particle=-1),
            layout="all-flow-union",
            normalization=_normalization(),
        )
    with pytest.raises(RecurrenceProjectionError, match="unknown external label"):
        project_recurrence_process_v1(
            process,
            _color_plan(process, bad_label=True),
            _catalog(),
            layout="all-flow-union",
            normalization=_normalization(),
        )


def test_union_projects_generation_selected_flow_and_partial_helicity() -> None:
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        _catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
        generation_slice=RecurrenceGenerationSliceV1(
            selected_public_flow_ids=(1,),
            # Initial-state crossing maps the catalog's -1 source to +1.
            selected_source_helicities=((1, 1),),
        ),
    )
    encoded = build_recurrence_builder_input_v1(logical)

    assert logical.selected_public_flow_ids == (1,)
    assert logical.selected_source_coverage is not None
    assert tuple(row.source_slot for row in logical.selected_source_coverage) == (
        0,
        1,
        2,
        3,
    )
    assert logical.selected_source_coverage[0].source_state_indices == (1,)
    assert all(
        row.source_state_indices == (0, 1)
        for row in logical.selected_source_coverage[1:]
    )
    assert tuple(
        int(value)
        for value in encoded.table("selected_public_flow_coverage").column("flow_id")
    ) == (1,)
    assert tuple(
        (
            int(source_slot),
            int(state_index),
        )
        for source_slot, state_index in zip(
            encoded.table("selected_source_coverage").column("source_slot"),
            encoded.table("selected_source_coverage").column("source_state_index"),
            strict=True,
        )
    ) == (
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    )
    assert int(encoded.table("header").column("selected_flow_mode")[0]) == 1
    assert int(encoded.table("header").column("selected_source_mode")[0]) == 1


@pytest.mark.parametrize(
    ("generation_slice", "message"),
    [
        (
            RecurrenceGenerationSliceV1(selected_public_flow_ids=(99,)),
            "unknown public LC flows",
        ),
        (
            RecurrenceGenerationSliceV1(selected_source_helicities=((99, 1),)),
            "unknown source labels",
        ),
        (
            RecurrenceGenerationSliceV1(selected_source_helicities=((1, 0),)),
            "selected helicity 0 is unavailable",
        ),
    ],
)
def test_union_generation_specialization_fails_closed(
    generation_slice: RecurrenceGenerationSliceV1,
    message: str,
) -> None:
    process = _process()
    with pytest.raises(RecurrenceProjectionError, match=message):
        project_recurrence_process_v1(
            process,
            _color_plan(process),
            _catalog(),
            layout="all-flow-union",
            normalization=_normalization(),
            generation_slice=generation_slice,
        )


def test_projection_does_not_depend_on_generic_dag_types() -> None:
    import pyamplicol.generation.recurrence_projection as projection

    names = set(projection.__dict__)
    assert "GenericDAG" not in names
    assert "compile_generic_dag" not in names


def test_mismatched_color_plan_process_is_rejected() -> None:
    process = _process()
    other = replace(process, key="different_process")
    with pytest.raises(RecurrenceProjectionError, match="does not belong"):
        project_recurrence_process_v1(
            process,
            _color_plan(other),
            _catalog(),
            layout="all-flow-union",
            normalization=_normalization(),
        )


def test_projection_rejects_ambiguous_builder_axes() -> None:
    process = _process()
    common = dict(
        process=process,
        color_plan=_color_plan(process),
        template_catalog=_catalog(),
        layout="all-flow-union",
        normalization=_normalization(),
    )
    with pytest.raises(RecurrenceProjectionError, match="support mask"):
        project_recurrence_process_v1(**common, process_support_mask=1.0)
    with pytest.raises(RecurrenceProjectionError, match="unique"):
        project_recurrence_process_v1(
            **common,
            coupling_order_limits={"QCD": 4, "qcd": 4},
        )


def test_complex_parameters_use_authoritative_prepared_slots() -> None:
    base = _catalog()
    complex_parameter = ParameterTemplateV1(
        template_id="parameter:complex_coupling",
        name="complex_coupling",
        parameter_kind="external",
        value_type="complex",
        mutable=True,
        default_value=ExactComplexRationalV1(1, 10, 1, 20),
        exact_expression_digest=None,
        dependency_parameter_ids=(),
        prepared_parameter_id=17,
    )
    catalog = RecurrenceTemplateCatalog.create(
        compiled_model_digest=base.header.compiled_model_digest,
        prepared_kernel_pack_digest=base.header.prepared_kernel_pack_digest,
        parameters=(*base.parameters, complex_parameter),
        current_states=base.current_states,
        sources=base.sources,
        evaluator_bindings=base.evaluator_bindings,
    )
    process = _process()
    logical = project_recurrence_process_v1(
        process,
        _color_plan(process),
        catalog,
        layout="all-flow-union",
        normalization=_normalization(),
    )

    complex_rows = tuple(
        item
        for item in logical.parameter_projection
        if item.runtime_name == "complex_coupling"
    )
    assert tuple(item.component for item in complex_rows) == (0, 1)
    assert tuple(item.prepared_parameter_id for item in complex_rows) == (17, 17)
    assert tuple(item.runtime_slot for item in logical.parameter_projection) == tuple(
        range(len(logical.parameter_projection))
    )
