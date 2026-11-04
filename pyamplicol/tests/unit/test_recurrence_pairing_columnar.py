# SPDX-License-Identifier: 0BSD
"""Private fixed-width fermion-pairing columnar ABI tests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np
import pytest

import pyamplicol.generation.recurrence_pairing_columnar as pairing_columnar
from pyamplicol.generation.recurrence_columnar import RecurrenceColumnarInputError
from pyamplicol.generation.recurrence_fermion_pairing import (
    NO_FERMION_LINE,
    FermionPairingCatalogV1,
    build_recurrence_fermion_pairing_catalog_v1,
)
from pyamplicol.models.recurrence_template import CurrentStateTemplateV1
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


def _fermion_leg(label: int, state: CurrentStateTemplateV1) -> ProcessLegIR:
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


def _fermion_process(
    key: str, states: Iterable[CurrentStateTemplateV1]
) -> CanonicalProcessIR:
    materialized = tuple(states)
    fundamentals = sum(
        state.lc_color_shape_kind == "fundamental-open-string" for state in materialized
    )
    antifundamentals = len(materialized) - fundamentals
    return CanonicalProcessIR(
        process="canonical pairing-columnar fixture",
        key=key,
        color_accuracy="lc",
        legs=tuple(
            _fermion_leg(index + 1, state) for index, state in enumerate(materialized)
        ),
        color_endpoints=ColorEndpointSummary(
            fundamental_count=fundamentals,
            antifundamental_count=antifundamentals,
            pair_count=min(fundamentals, antifundamentals),
        ),
    )


def _boson_process() -> CanonicalProcessIR:
    legs = tuple(
        ProcessLegIR(
            label=index + 1,
            side="final",
            particle="g",
            outgoing_particle="g",
            pdg=21,
            outgoing_pdg=21,
            statistics="boson",
            wavefunction_family="vector",
            color_role="adjoint",
            source_orientation="inclusive",
        )
        for index in range(2)
    )
    return CanonicalProcessIR(
        process="g g",
        key="boson-only",
        color_accuracy="lc",
        legs=legs,
        color_endpoints=ColorEndpointSummary(
            fundamental_count=0,
            antifundamental_count=0,
            pair_count=0,
        ),
    )


def _encode(catalog: FermionPairingCatalogV1):  # type: ignore[no-untyped-def]
    return pairing_columnar._encode_fermion_pairing_catalog_v1(catalog)


def _table_rows(payload, name: str, *columns: str):  # type: ignore[no-untyped-def]
    table = payload.table(name)
    return tuple(
        tuple(int(table.column(column)[row]) for column in columns)
        for row in range(table.row_count)
    )


def _header_digest(payload, name: str) -> str:  # type: ignore[no-untyped-def]
    return bytes(payload.table("header").column(name)[0]).hex()


def _identical_catalog(
    *, token: str = "identical", species_id: str | None = None
) -> FermionPairingCatalogV1:
    particle, antiparticle = _state_pair(token, 930001, species_id=species_id)
    external = (antiparticle, particle, particle, antiparticle)
    return build_recurrence_fermion_pairing_catalog_v1(
        _fermion_process("identical-two-line", external),
        (particle, antiparticle),
    )


def test_no_fermion_catalog_encodes_one_trivial_rule() -> None:
    catalog = build_recurrence_fermion_pairing_catalog_v1(_boson_process(), ())
    payload = _encode(catalog)

    assert payload.table("endpoints").row_count == 0
    assert payload.table("pairing_classes").row_count == 0
    assert payload.table("rules").row_count == 1
    assert _table_rows(payload, "rule_source_slot_permutations", "source_slot") == (
        (0,),
        (1,),
    )
    assert _table_rows(payload, "rule_lineages", "line_id") == (
        (NO_FERMION_LINE,),
        (NO_FERMION_LINE,),
    )
    assert int(payload.table("rules").column("fermion_parity")[0]) == 1


def test_distinct_lines_encode_endpoint_classes_and_flat_ranges() -> None:
    first, first_anti = _state_pair("first", 910001)
    second, second_anti = _state_pair("second", 920001)
    external = (first_anti, first, second, second_anti)
    catalog = build_recurrence_fermion_pairing_catalog_v1(
        _fermion_process("distinct-two-line", external),
        (second_anti, first, first_anti, second),
    )

    payload = _encode(catalog)

    assert _table_rows(
        payload,
        "rule_endpoint_pairings",
        "fundamental_source_slot",
        "antifundamental_source_slot",
    ) == ((1, 0), (2, 3))
    assert _table_rows(payload, "rule_lineages", "line_id") == (
        (0,),
        (0,),
        (1,),
        (1,),
    )
    classes = payload.table("pairing_classes")
    assert tuple(int(value) for value in classes.column("pairing_count")) == (1, 1)
    assert tuple(int(value) for value in classes.column("reference_pairing_start")) == (
        0,
        1,
    )


def test_identical_lines_encode_direct_exchange_and_exact_integer_limbs() -> None:
    payload = _encode(_identical_catalog())

    rules = payload.table("rules")
    assert tuple(int(value) for value in rules.column("fermion_parity")) == (1, -1)
    assert _table_rows(
        payload,
        "rule_endpoint_pairings",
        "fundamental_source_slot",
        "antifundamental_source_slot",
    ) == ((1, 0), (2, 3), (1, 3), (2, 0))
    assert _table_rows(payload, "rule_source_slot_permutations", "source_slot") == (
        (0,),
        (1,),
        (2,),
        (3,),
        (3,),
        (1,),
        (2,),
        (0,),
    )

    integers = payload.table("exact_integers")
    signs = tuple(int(value) for value in integers.column("sign"))
    assert signs == (-1, 0, 1)
    assert _table_rows(payload, "exact_integer_limbs", "value") == ((1,), (1,))
    assert tuple(int(value) for value in rules.column("real_numerator_integer_id")) == (
        2,
        0,
    )


def test_encoding_is_deterministic_and_model_equivalent_topology_is_stable() -> None:
    builtin = _identical_catalog(
        token="builtin:matter", species_id="builtin:model-owned:matter"
    )
    ufo = _identical_catalog(token="ufo:matter", species_id="ufo:model-owned:matter")
    first = _encode(builtin)
    repeated = _encode(builtin)
    equivalent = _encode(ufo)

    for left, right in zip(first.tables, repeated.tables, strict=True):
        for left_column, right_column in zip(left.columns, right.columns, strict=True):
            assert np.array_equal(left_column.values, right_column.values)
            assert left_column.values.dtype.str in {"|u1", "<u4", "<u8", "<i4"}
            assert left_column.values.flags.c_contiguous
            assert left_column.values.flags.owndata
            assert not left_column.values.flags.writeable

    assert _header_digest(first, "topology_digest") == _header_digest(
        equivalent, "topology_digest"
    )
    assert _header_digest(first, "semantic_digest") != _header_digest(
        equivalent, "semantic_digest"
    )
    for table_name in (
        "rules",
        "rule_class_pairing_indices",
        "rule_endpoint_pairings",
        "rule_source_slot_permutations",
        "rule_lineages",
    ):
        left = first.table(table_name)
        right = equivalent.table(table_name)
        for left_column, right_column in zip(left.columns, right.columns, strict=True):
            if left_column.name == "proof_algorithm_string_id":
                continue
            assert np.array_equal(left_column.values, right_column.values)


def test_private_freeze_adopts_schema_exact_owned_columns() -> None:
    columns = pairing_columnar._allocate("rule_endpoint_pairings", 2)
    columns["fundamental_source_slot"][:] = (2, 3)
    columns["antifundamental_source_slot"][:] = (4, 5)
    original = dict(columns)

    table = pairing_columnar._freeze_table("rule_endpoint_pairings", columns)

    for column in table.columns:
        assert column.values is original[column.name]
        assert column.values.flags.owndata
        assert column.values.flags.c_contiguous
        assert not column.values.flags.writeable


def test_private_freeze_does_not_coerce_a_schema_mismatch() -> None:
    payload = _encode(_identical_catalog())
    columns = pairing_columnar._allocate("rule_endpoint_pairings", 4)
    wrong_dtype = columns["fundamental_source_slot"].astype("<u8")
    columns["fundamental_source_slot"] = wrong_dtype

    malformed = pairing_columnar._freeze_table("rule_endpoint_pairings", columns)

    assert malformed.column("fundamental_source_slot") is not wrong_dtype
    assert malformed.column("fundamental_source_slot").dtype == wrong_dtype.dtype
    tables = tuple(
        malformed if table.name == "rule_endpoint_pairings" else table
        for table in payload.tables
    )
    with pytest.raises(
        RecurrenceColumnarInputError,
        match="wrong little-endian fixed-width representation",
    ):
        pairing_columnar._FermionPairingColumnarV1(payload.abi, tables)


def test_malformed_catalog_and_integer_bounds_fail_closed() -> None:
    catalog = _identical_catalog()
    with pytest.raises(RecurrenceColumnarInputError, match="topology digest"):
        _encode(replace(catalog, topology_digest="0" * 64))

    stale_endpoint = replace(catalog.endpoints[0], endpoint_id=1)
    with pytest.raises(RecurrenceColumnarInputError, match="endpoint IDs"):
        _encode(replace(catalog, endpoints=(stale_endpoint, *catalog.endpoints[1:])))

    stale_rule = replace(
        catalog.rules[0],
        source_slot_permutation=(0, 0, 2, 3),
    )
    with pytest.raises(RecurrenceColumnarInputError, match="source-slot permutation"):
        _encode(replace(catalog, rules=(stale_rule, *catalog.rules[1:])))

    with pytest.raises(RecurrenceColumnarInputError, match="does not fit u32"):
        _encode(replace(catalog, source_count=1 << 32))
