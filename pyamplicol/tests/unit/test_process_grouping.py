# SPDX-License-Identifier: 0BSD
"""Flavour-profile canonicalization and process-group maps."""

from __future__ import annotations

from fractions import Fraction

import pytest

from pyamplicol.generation.process_grouping import (
    GroupingBasis,
    GroupingParticle,
    GroupingProcess,
    GroupMember,
    ParticleType,
    ProcessGroupMap,
    candidate_mappings,
    coarse_key,
    flavor_blind_classes,
    flavor_blind_mapping,
    flavor_class_key,
    profile_key,
    profile_mappings,
    rational_factor,
    resolved_parameter_name,
)

_QUARK = ParticleType(2, 3, "ZERO", "ZERO", False)
_GLUON = ParticleType(3, 8, "ZERO", "ZERO", True)
_LEPTON = ParticleType(2, 1, "ZERO", "ZERO", False)
_NAMES = {1: "d", 2: "u", 3: "s", 4: "c", 21: "g", 11: "e-", 12: "ve"}


def _table() -> dict[int, GroupingParticle]:
    table = {}
    for pdg, name in _NAMES.items():
        kind = _GLUON if pdg == 21 else _QUARK if pdg < 10 else _LEPTON
        table[pdg] = GroupingParticle(pdg, name, kind)
        if not kind.self_conjugate:
            table[-pdg] = GroupingParticle(-pdg, name + "~", kind)
    return table


TABLE = _table()
_PDGS = {
    **{name: pdg for pdg, name in _NAMES.items()},
    **{name + "~": -pdg for pdg, name in _NAMES.items() if pdg != 21},
}


def process(expression: str, process_id: str | None = None) -> GroupingProcess:
    initial, final = expression.split(">")
    pdgs = tuple(_PDGS[name] for name in (*initial.split(), *final.split()))
    return GroupingProcess(
        process_id=process_id or expression,
        expression=expression,
        pdgs=pdgs,
        initial_count=len(initial.split()),
    )


def _momentum_labels(member: GroupingProcess, permutation) -> list[str]:
    """Particle names the representative legs receive from the member."""

    names = {pdg: name for name, pdg in _PDGS.items()}
    return [names[member.pdgs[index]] for index in permutation]


def test_profile_order_follows_flavour_relabelling_where_pdg_order_fails() -> None:
    first = process("u u~ > d u d~ u~")
    second = process("d d~ > d u d~ u~")

    assert profile_key(first, TABLE) == profile_key(second, TABLE)
    (mapping,) = profile_mappings(first, second, TABLE, limit=1)
    # The representative's "d d~" beams receive the member's "u u~" beams and
    # its single "u u~" final pair receives the member's "d d~" pair.
    received = _momentum_labels(first, mapping.permutation)
    assert received[:2] == ["u", "u~"]
    representative_names = [
        {pdg: name for name, pdg in _PDGS.items()}[pdg] for pdg in second.pdgs
    ]
    swap = {"u": "d", "d": "u", "u~": "d~", "d~": "u~"}
    assert [swap[name] for name in received] == representative_names


def test_flavour_patterns_are_global_across_initial_and_final_states() -> None:
    assert profile_key(process("d d~ > u u~"), TABLE) != profile_key(
        process("d d~ > d d~"), TABLE
    )
    assert profile_key(process("g g > d u d~ u~"), TABLE) != profile_key(
        process("g g > d d d~ d~"), TABLE
    )
    assert profile_key(process("g g > d d~"), TABLE) == profile_key(
        process("g g > u u~"), TABLE
    )
    assert profile_key(process("u u~ > d d~"), TABLE) == profile_key(
        process("d d~ > u u~"), TABLE
    )
    assert coarse_key(process("d d~ > u u~"), TABLE) == coarse_key(
        process("d d~ > d d~"), TABLE
    )


def test_charge_conjugation_is_a_separate_relation_with_reversed_flows() -> None:
    quark = process("u g > u g")
    antiquark = process("u~ g > u~ g")

    assert profile_key(quark, TABLE) != profile_key(antiquark, TABLE)
    assert flavor_class_key(
        quark, TABLE, consider_charge_conjugation=True
    ) == flavor_class_key(antiquark, TABLE, consider_charge_conjugation=True)
    assert flavor_class_key(
        quark, TABLE, consider_charge_conjugation=False
    ) != flavor_class_key(antiquark, TABLE, consider_charge_conjugation=False)
    mapping = flavor_blind_mapping(
        antiquark, quark, TABLE, consider_charge_conjugation=True
    )
    assert mapping.conjugated
    assert mapping.permutation == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="flavour relabelling"):
        flavor_blind_mapping(antiquark, quark, TABLE, consider_charge_conjugation=False)


def test_candidate_mappings_cover_ties_conjugation_and_signed_pdg_fallback() -> None:
    member = process("u c > u c")
    representative = process("d s > d s")
    mappings = candidate_mappings(
        member, representative, TABLE, consider_charge_conjugation=True
    )
    permutations = {mapping.permutation for mapping in mappings}
    # Both pairings of the two tied flavours are offered.
    assert (0, 1, 2, 3) in permutations
    assert (1, 0, 3, 2) in permutations
    fallback = candidate_mappings(
        process("u d > u d"),
        process("u d~ > u d~"),
        TABLE,
        consider_charge_conjugation=False,
    )
    assert [mapping.method for mapping in fallback] == ["signed-pdg"]


def test_flavour_blind_classes_choose_the_canonical_representative() -> None:
    processes = [
        process(expression, f"p{index}")
        for index, expression in enumerate(
            (
                "u u~ > g g",
                "d d~ > g g",
                "u~ u > g g",
                "u g > u g",
                "u~ g > u~ g",
                "d d~ > u u~",
                "d d~ > d d~",
            )
        )
    ]
    classes = flavor_blind_classes(processes, TABLE, consider_charge_conjugation=True)
    by_representative = {
        group[0].expression: sorted(p.expression for p in group[1:])
        for group in classes
    }
    assert by_representative["d d~ > g g"] == ["u u~ > g g", "u~ u > g g"]
    assert by_representative["u g > u g"] == ["u~ g > u~ g"]
    assert by_representative["d d~ > d d~"] == []
    assert by_representative["d d~ > u u~"] == []


def test_group_map_merges_compose_permutations_factors_and_conjugation() -> None:
    a = process("d d~ > g g", "a")
    b = process("u u~ > g g", "b")
    c = process("u~ u > g g", "c")
    group_map = ProcessGroupMap.trivial([a, b, c])
    mapping_b = profile_mappings(b, a, TABLE, limit=1)[0]
    group_map.merge(
        "a",
        GroupMember.identity(b).related(
            mapping_b, factor=Fraction(1), basis=GroupingBasis.NUMERICAL
        ),
        TABLE,
    )
    mapping_c = profile_mappings(c, b, TABLE, limit=1)[0]
    assert mapping_c.permutation == (1, 0, 2, 3)
    group_map.merge(
        group_map.group_of("a").representative.process_id,  # type: ignore[union-attr]
        GroupMember.identity(c).related(
            mapping_c, factor=Fraction(-3, 2), basis=GroupingBasis.NUMERICAL
        ),
        TABLE,
    )
    (group,) = group_map.groups.values()
    assert group.representative.process_id == "a"
    members = {member.process_id: member for member in group.members}
    assert members["c"].factor == Fraction(-3, 2)
    assert members["c"].permutation == (1, 0, 2, 3)
    assert members["b"].basis is GroupingBasis.NUMERICAL

    restored = ProcessGroupMap.from_json(group_map.to_json())
    assert restored.as_document() == group_map.as_document()


def test_merge_keeps_the_smaller_representative_and_inverts_the_link() -> None:
    small = process("d d~ > g g", "small")
    large = process("u u~ > g g", "large")
    group_map = ProcessGroupMap.trivial([small, large])
    mapping = profile_mappings(small, large, TABLE, limit=1)[0]
    kept = group_map.merge(
        "large",
        GroupMember.identity(small).related(
            mapping, factor=Fraction(2), basis=GroupingBasis.NUMERICAL
        ),
        TABLE,
    )
    assert kept == "small"
    members = {m.process_id: m for m in group_map.groups["small"].members}
    assert members["large"].factor == Fraction(1, 2)


def test_rational_factor_and_parameter_resolution() -> None:
    assert rational_factor(4.0 + 1e-13, tolerance=1e-9) == Fraction(4)
    assert rational_factor(-0.5, tolerance=1e-9) == Fraction(-1, 2)
    assert rational_factor(3.14159, tolerance=1e-9) is None
    assert rational_factor(150.0, tolerance=1e-9) is None

    class Record:
        def __init__(self, resolved: str) -> None:
            self.resolved_expression = resolved

    parameters = {"MC": Record("0"), "MB": Record("model::MB")}
    assert resolved_parameter_name("MC", parameters) == "ZERO"
    assert resolved_parameter_name("ZERO", parameters) == "ZERO"
    assert resolved_parameter_name("MB", parameters) == "MB"
