# SPDX-License-Identifier: 0BSD
"""Exact, bounded external-fermion pairing catalog tests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import pytest

from pyamplicol.generation.recurrence_fermion_pairing import (
    NO_FERMION_LINE,
    RecurrenceFermionPairingError,
    build_recurrence_fermion_pairing_catalog_v1,
)
from pyamplicol.models.recurrence_template import (
    CurrentStateTemplateV1,
    ExactComplexRationalV1,
    QuantumFlowTemplateV1,
)
from pyamplicol.processes.ir import (
    CanonicalProcessIR,
    ColorEndpointSummary,
    ProcessLegIR,
)


def _state_pair(
    token: str,
    particle_id: int,
    *,
    species_id: str | None = None,
) -> tuple[CurrentStateTemplateV1, CurrentStateTemplateV1]:
    species = species_id or f"contract:{token}:species"
    particle = CurrentStateTemplateV1(
        template_id=f"{token}:particle",
        particle_id=particle_id,
        anti_particle_id=-particle_id,
        species_id=species,
        orientation="particle",
        statistics="fermion",
        color_representation=3,
        basis="generic-fermion",
        tensor_ordering=("s0", "s1"),
        dimension=2,
        chirality=1,
        lc_color_shape_kind="fundamental-open-string",
        auxiliary_kind=None,
        mass_parameter_id=None,
        width_parameter_id=None,
    )
    antiparticle = CurrentStateTemplateV1(
        template_id=f"{token}:antiparticle",
        particle_id=-particle_id,
        anti_particle_id=particle_id,
        species_id=species,
        orientation="antiparticle",
        statistics="fermion",
        color_representation=-3,
        basis="generic-fermion",
        tensor_ordering=("s0", "s1"),
        dimension=2,
        chirality=-1,
        lc_color_shape_kind="antifundamental-open-string",
        auxiliary_kind=None,
        mass_parameter_id=None,
        width_parameter_id=None,
    )
    return particle, antiparticle


def _leg(label: int, state: CurrentStateTemplateV1) -> ProcessLegIR:
    role = (
        "fundamental"
        if state.lc_color_shape_kind == "fundamental-open-string"
        else "antifundamental"
    )
    return ProcessLegIR(
        label=label,
        side="final",
        particle=state.template_id,
        outgoing_particle=state.template_id,
        pdg=state.particle_id,
        outgoing_pdg=state.particle_id,
        statistics="fermion",
        wavefunction_family="fermion",
        color_role=role,
        source_orientation=state.orientation,
    )


def _charged_current_flow(
    down: CurrentStateTemplateV1,
    up: CurrentStateTemplateV1,
) -> tuple[CurrentStateTemplateV1, QuantumFlowTemplateV1]:
    boson = CurrentStateTemplateV1(
        template_id="charged-current:w",
        particle_id=990024,
        anti_particle_id=-990024,
        species_id="contract:charged-current:w",
        orientation="particle",
        statistics="boson",
        color_representation=1,
        basis="vector",
        tensor_ordering=("v0", "v1", "v2", "v3"),
        dimension=4,
        chirality=0,
        lc_color_shape_kind="singlet-forest",
        auxiliary_kind=None,
        mass_parameter_id=None,
        width_parameter_id=None,
    )
    return boson, QuantumFlowTemplateV1(
        template_id="charged-current:d-w-to-u",
        input_state_template_ids=(down.template_id, boson.template_id),
        input_spin_states=(1, 0),
        input_flavour_flows=((down.particle_id,), (boson.particle_id,)),
        input_quantum_number_flows=((), ()),
        flavour_flow_operation="append-left-result",
        quantum_number_flow_operation="particle-static-result",
        coupling_orders=(("QED", 1),),
        result_state_template_id=up.template_id,
        result_spin_state=1,
        result_flavour_flow=(down.particle_id, up.particle_id),
        result_quantum_number_flow=(),
        exact_coupling=ExactComplexRationalV1.one(),
        predicate_digest="a" * 64,
    )


def _process(
    key: str,
    states: Iterable[CurrentStateTemplateV1],
    *,
    color_accuracy: str = "lc",
) -> CanonicalProcessIR:
    materialized = tuple(states)
    fundamental_count = sum(
        state.lc_color_shape_kind == "fundamental-open-string" for state in materialized
    )
    antifundamental_count = len(materialized) - fundamental_count
    return CanonicalProcessIR(
        process="canonical contract fixture",
        key=key,
        color_accuracy=color_accuracy,
        legs=tuple(_leg(index + 1, state) for index, state in enumerate(materialized)),
        color_endpoints=ColorEndpointSummary(
            fundamental_count=fundamental_count,
            antifundamental_count=antifundamental_count,
            pair_count=min(fundamental_count, antifundamental_count),
        ),
    )


def test_distinct_two_line_pairing_has_one_even_rule() -> None:
    first, first_anti = _state_pair("first", 910001)
    second, second_anti = _state_pair("second", 920001)
    external = (first_anti, first, second, second_anti)

    catalog = build_recurrence_fermion_pairing_catalog_v1(
        _process("distinct-two-line", external),
        (second_anti, first, first_anti, second),
    )

    assert len(catalog.pairing_classes) == 2
    assert tuple(row.pairing_count for row in catalog.pairing_classes) == (1, 1)
    assert len(catalog.rules) == 1
    assert catalog.rules[0].fermion_parity == 1
    assert catalog.rules[0].endpoint_pairings == ((1, 0), (2, 3))
    assert catalog.rules[0].lineage_by_source_slot == (0, 0, 1, 1)


def test_identical_two_line_pairings_are_direct_plus_exchange_minus() -> None:
    particle, antiparticle = _state_pair("identical", 930001)
    external = (antiparticle, particle, particle, antiparticle)

    catalog = build_recurrence_fermion_pairing_catalog_v1(
        _process("identical-two-line", external),
        (particle, antiparticle),
    )

    assert len(catalog.pairing_classes) == 1
    assert catalog.pairing_classes[0].reference_pairings == ((1, 0), (2, 3))
    assert tuple(rule.fermion_parity for rule in catalog.rules) == (1, -1)
    assert tuple(rule.exact_factor.real_numerator for rule in catalog.rules) == (
        1,
        -1,
    )
    assert catalog.rules[0].endpoint_pairings == ((1, 0), (2, 3))
    assert catalog.rules[0].source_slot_permutation == (0, 1, 2, 3)
    assert catalog.rules[1].endpoint_pairings == ((1, 3), (2, 0))
    assert catalog.rules[1].source_slot_permutation == (3, 1, 2, 0)


def test_four_identical_lines_have_accuracy_independent_canonical_pairings() -> None:
    particle, antiparticle = _state_pair("four-identical", 930011)
    external = (antiparticle, particle) * 4
    catalogs = tuple(
        build_recurrence_fermion_pairing_catalog_v1(
            _process(
                "four-identical-lines",
                external,
                color_accuracy=accuracy,
            ),
            (particle, antiparticle),
        )
        for accuracy in ("lc", "nlc", "full")
    )

    reference = catalogs[0]
    canonical_pairings = tuple(rule.endpoint_pairings for rule in reference.rules)
    assert len(canonical_pairings) == len(set(canonical_pairings)) == 24
    assert all(pairings == tuple(sorted(pairings)) for pairings in canonical_pairings)
    assert Counter(
        (rule.fermion_parity, rule.exact_factor) for rule in reference.rules
    ) == {
        (1, ExactComplexRationalV1(1, 1, 0, 1)): 12,
        (-1, ExactComplexRationalV1(-1, 1, 0, 1)): 12,
    }
    assert all(
        (catalog.topology_digest, catalog.rules)
        == (reference.topology_digest, reference.rules)
        for catalog in catalogs[1:]
    )


def test_three_distinct_lines_have_one_rule_and_three_lineages() -> None:
    one, one_anti = _state_pair("one", 940001)
    two, two_anti = _state_pair("two", 940002)
    three, three_anti = _state_pair("three", 940003)
    external = (one, two_anti, three, one_anti, two, three_anti)

    catalog = build_recurrence_fermion_pairing_catalog_v1(
        _process("three-distinct-lines", external),
        (three_anti, two, one_anti, three, one, two_anti),
    )

    assert len(catalog.pairing_classes) == 3
    assert len(catalog.rules) == 1
    assert catalog.rules[0].fermion_parity == 1
    assert catalog.rules[0].endpoint_pairings == ((0, 3), (2, 5), (4, 1))
    assert catalog.rules[0].lineage_by_source_slot == (0, 2, 1, 0, 2, 1)
    assert NO_FERMION_LINE not in catalog.rules[0].lineage_by_source_slot


def test_incompatible_species_are_rejected_before_rule_enumeration() -> None:
    first, _first_anti = _state_pair("first-only", 950001)
    _second, second_anti = _state_pair("second-only", 950002)
    process = _process("incompatible-species", (first, second_anti))
    states = (*_state_pair("first-only", 950001), *_state_pair("second-only", 950002))

    with pytest.raises(
        RecurrenceFermionPairingError,
        match="incompatible fermion species endpoints",
    ):
        build_recurrence_fermion_pairing_catalog_v1(process, states)


def test_model_proven_charged_current_species_share_one_pairing_class() -> None:
    down, down_anti = _state_pair("charged-down", 950011)
    up, up_anti = _state_pair("charged-up", 950012)
    boson, flow = _charged_current_flow(down, up)

    catalog = build_recurrence_fermion_pairing_catalog_v1(
        _process("charged-current", (down, up_anti)),
        (down, down_anti, up, up_anti, boson),
        quantum_flows=(flow,),
    )

    assert len(catalog.pairing_classes) == 1
    assert catalog.pairing_classes[0].fundamental_source_slots == (0,)
    assert catalog.pairing_classes[0].antifundamental_source_slots == (1,)
    assert catalog.rules[0].endpoint_pairings == ((0, 1),)


def test_equivalent_model_contracts_have_identical_pairing_topology() -> None:
    builtin_particle, builtin_anti = _state_pair(
        "builtin:matter",
        960001,
        species_id="builtin:model-owned:matter",
    )
    ufo_particle, ufo_anti = _state_pair(
        "ufo:matter",
        960001,
        species_id="ufo:model-owned:matter",
    )
    builtin_external = (
        builtin_anti,
        builtin_particle,
        builtin_particle,
        builtin_anti,
    )
    ufo_external = (ufo_anti, ufo_particle, ufo_particle, ufo_anti)

    builtin = build_recurrence_fermion_pairing_catalog_v1(
        _process("equivalent-model-contract", builtin_external),
        (builtin_particle, builtin_anti),
    )
    ufo = build_recurrence_fermion_pairing_catalog_v1(
        _process("equivalent-model-contract", ufo_external),
        (ufo_anti, ufo_particle),
    )

    assert builtin.topology_digest == ufo.topology_digest
    assert builtin.rules == ufo.rules
    assert tuple(row.species_class_id for row in builtin.endpoints) == tuple(
        row.species_class_id for row in ufo.endpoints
    )
    assert builtin.semantic_digest != ufo.semantic_digest
