# SPDX-License-Identifier: 0BSD
"""Independent small structural counts and ordering checks for soft catalogues."""

import json
from collections import Counter
from itertools import product

import pytest

from pyamplicol.color.connection_catalogue import (
    _labelled_histories,
    build_color_correlator_catalogue,
)
from pyamplicol.color.connections import (
    ColorConnection,
    ColorLeg,
    EmitGluon,
    SplitGluon,
)


def _histories(catalogue, order):
    result = {}
    for request in catalogue:
        if request.order != order:
            continue
        _, left, right = request.id.split("/")
        if left == right:
            assert request.bra == request.ket
            result[int(left[1:])] = request.bra
    assert tuple(result) == tuple(range(len(result)))
    return result


@pytest.fixture(scope="module")
def pair_catalogue():
    legs = (ColorLeg(1, 3), ColorLeg(2, -3))
    return legs, build_color_correlator_catalogue(legs, through_order=3)


@pytest.mark.parametrize("count", (1, 2, 3))
def test_nlo_and_nnlo_exhaustive_hand_counts(count):
    # A: distinct Born emitters, B: same Born line, C: emitted gluon,
    # D: emitted gluon splits. These count labelled maps, not multiplicities.
    legs = tuple(ColorLeg(index, 8) for index in range(1, count + 1))
    catalogue = build_color_correlator_catalogue(legs, through_order=2)
    assert len(_histories(catalogue, 1)) == count
    assert sum(request.order == 1 for request in catalogue) == count**2
    histories = _histories(catalogue, 2)
    categories = Counter()
    for operations in histories.values():
        first, second = operations
        if isinstance(second, SplitGluon):
            categories["D"] += 1
            assert second.parent_label == first.emitted_label < -2
        elif second.emitter_label < 0:
            categories["C"] += 1
            assert second.emitter_label == first.emitted_label
        elif second.emitter_label == first.emitter_label:
            categories["B"] += 1
        else:
            categories["A"] += 1
    assert {category: categories[category] for category in "ABCD"} == {
        "A": count * (count - 1),
        "B": 2 * count,
        "C": 2 * count,
        "D": 2 * count,
    }
    assert len(histories) == count**2 + 5 * count
    assert (
        sum(request.order == 2 for request in catalogue)
        == (count**2 + 3 * count) ** 2 + 2 * count**2
    )


def test_n3lo_exhaustive_small_forest_counts_and_all_directed_pairs(pair_catalogue):
    legs, catalogue = pair_catalogue
    histories = _histories(catalogue, 3)
    # Labelled ordered forests give k(k+4)(k+5) gluonic histories; each
    # of the six q/qbar/g assignments has k(k+5) pair histories.
    count = 2
    gluonic = count * (count + 4) * (count + 5)
    pair_block = count * (count + 5)
    outputs = {
        index: ColorConnection(legs, operations).output_legs
        for index, operations in histories.items()
    }
    assert sorted(Counter(outputs.values()).values()) == [pair_block] * 6 + [gluonic]
    assert len(histories) == gluonic + 6 * pair_block
    observed = {
        tuple(int(index[1:]) for index in request.id.split("/")[1:])
        for request in catalogue
        if request.order == 3
    }
    expected = {
        (left, right)
        for left, right in product(histories, repeat=2)
        if outputs[left] == outputs[right]
    }
    assert observed == expected
    assert len(observed) == gluonic**2 + 6 * pair_block**2 == 8232


def test_independent_interleavings_collapse_but_shared_line_order_survives(
    pair_catalogue,
):
    _, catalogue = pair_catalogue
    histories = tuple(_histories(catalogue, 2).values())
    independent = {EmitGluon(1, -1), EmitGluon(2, -2)}
    assert sum(set(history) == independent for history in histories) == 1
    assert (EmitGluon(1, -1), EmitGluon(1, -2)) in histories
    assert (EmitGluon(1, -2), EmitGluon(1, -1)) in histories
    # No canonical sorting is permitted to move the consumer before creation.
    assert (EmitGluon(1, -1), EmitGluon(-1, -2)) in histories
    assert (EmitGluon(-1, -2), EmitGluon(1, -1)) not in histories


def test_n3lo_pair_first_and_gluon_first_cross_order_overlap_is_present(
    pair_catalogue,
):
    legs, catalogue = pair_catalogue
    histories = _histories(catalogue, 3)
    pair_first = (
        EmitGluon(1, -4),
        SplitGluon(-4, -1, -2),
        EmitGluon(-1, -3),
    )
    # Its raw creation history can emit the final gluon before making the
    # pair. Final-role relabelling must still match q=-1,qbar=-2,g=-3.
    independent_steps = {
        EmitGluon(2, -3),
        EmitGluon(1, -4),
        SplitGluon(-4, -1, -2),
    }
    pair_index = next(i for i, steps in histories.items() if steps == pair_first)
    other_indices = [
        i for i, steps in histories.items() if set(steps) == independent_steps
    ]
    assert len(other_indices) == 1
    other_index = other_indices[0]
    assert (
        ColorConnection(legs, pair_first).output_legs
        == ColorConnection(legs, histories[other_index]).output_legs
    )
    requests = {request.id: request for request in catalogue}
    assert requests[f"N3LO/c{pair_index}/c{other_index}"].bra == pair_first
    assert requests[f"N3LO/c{other_index}/c{pair_index}"].ket == pair_first


def test_soft_scope_labels_and_deterministic_json(pair_catalogue):
    legs, catalogue = pair_catalogue
    again = build_color_correlator_catalogue(tuple(reversed(legs)), through_order=3)
    assert json.dumps([request.to_json_dict() for request in catalogue]) == json.dumps(
        [request.to_json_dict() for request in again]
    )
    assert len({request.id for request in catalogue}) == len(catalogue)
    assert all(request.id != "born" for request in catalogue)
    for order in range(1, 4):
        for operations in _histories(catalogue, order).values():
            connection = ColorConnection(legs, operations)
            assert {leg.label for leg in connection.output_legs if leg.label < 0} == {
                -index for index in range(1, order + 1)
            }
            for operation in operations:
                if isinstance(operation, SplitGluon):
                    assert operation.parent_label < -order
    # A singlet neither emits nor changes catalogue IDs or operations.
    assert build_color_correlator_catalogue(
        (*legs, ColorLeg(9, 1)), through_order=2
    ) == tuple(request for request in catalogue if request.order <= 2)


def test_born_adjoint_does_not_split_and_singlets_have_empty_catalogues():
    catalogue = build_color_correlator_catalogue((ColorLeg(1, 8),), through_order=2)
    assert not any(
        isinstance(step, SplitGluon) and step.parent_label > 0
        for request in catalogue
        for step in (*request.bra, *request.ket)
    )
    assert build_color_correlator_catalogue((ColorLeg(1, 1),), through_order=3) == ()
    assert build_color_correlator_catalogue((ColorLeg(1, 1),), through_order=5) == ()


def test_n4lo_two_pairs_canonicalize_retired_labels_and_commuting_splits():
    legs = (ColorLeg(1, 3),)
    first = ColorConnection(
        legs,
        (
            EmitGluon(1, -10),
            SplitGluon(-10, -11, -12),
            EmitGluon(1, -20),
            SplitGluon(-20, -21, -22),
        ),
    )
    interleaved = ColorConnection(
        legs,
        (
            EmitGluon(1, -80),
            EmitGluon(1, -70),
            SplitGluon(-70, -21, -22),
            SplitGluon(-80, -11, -12),
        ),
    )
    # The two Born charge applications retain their relative order. Splits
    # commute across independent active legs; retired names are pure dummies.
    expected = dict(_labelled_histories(first))
    assert dict(_labelled_histories(interleaved)) == expected
    assert len(expected) == 24
    output_counts = Counter()
    for operations in expected.values():
        connection = ColorConnection(legs, operations)
        output_counts[connection.output_legs] += 1
        assert {
            step.parent_label for step in operations if isinstance(step, SplitGluon)
        } == {
            -5,
            -6,
        }
    # Six q/qbar label orientations, each retaining both quark-pair
    # assignments and both noncommuting hard-line charge orders.
    assert sorted(output_counts.values()) == [4] * 6


@pytest.mark.parametrize("order", (0, -1, True, 1.0, "2"))
def test_invalid_orders_are_rejected(order):
    with pytest.raises(ValueError, match="through_order"):
        build_color_correlator_catalogue((ColorLeg(1, 8),), through_order=order)


@pytest.mark.parametrize(
    "legs",
    ((ColorLeg(0, 8),), (ColorLeg(-1, 8),), (ColorLeg(1, 8), ColorLeg(1, 3))),
)
def test_invalid_born_labels_are_rejected(legs):
    with pytest.raises(ValueError):
        build_color_correlator_catalogue(legs, through_order=1)
