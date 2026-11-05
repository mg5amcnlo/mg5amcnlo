# SPDX-License-Identifier: 0BSD
"""Small independent generator checks for ordered exact colour connections."""

from fractions import Fraction
from itertools import permutations, product
from math import sqrt

import pytest

from pyamplicol.color.connections import (
    ColorConnection,
    ColorLeg,
    ColorTensor,
    ConnectedColorTensor,
    EmitGluon,
    ExactColorCoefficient,
    OpenColorString,
    SplitGluon,
    TensorTerm,
    all_outgoing_color_leg,
    apply_color_connection,
    color_connection_matrix_element,
    contract_connected_tensor_nc_terms,
    contract_connected_tensors,
    evaluate_color_nc_terms,
)

_QQ_LEGS = (ColorLeg(1, 3), ColorLeg(2, -3))
_QQ = ColorTensor((OpenColorString(1, (), 2),))

# Literal Gell-Mann matrices, independently of the symbolic Fierz machinery.
_LAMBDA = (
    ((0, 1, 0), (1, 0, 0), (0, 0, 0)),
    ((0, -1j, 0), (1j, 0, 0), (0, 0, 0)),
    ((1, 0, 0), (0, -1, 0), (0, 0, 0)),
    ((0, 0, 1), (0, 0, 0), (1, 0, 0)),
    ((0, 0, -1j), (0, 0, 0), (1j, 0, 0)),
    ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 0, 0), (0, 0, -1j), (0, 1j, 0)),
    ((1 / sqrt(3), 0, 0), (0, 1 / sqrt(3), 0), (0, 0, -2 / sqrt(3))),
)
_T = tuple(
    tuple(tuple(value / 2 for value in row) for row in matrix) for matrix in _LAMBDA
)
_TAU = tuple(
    tuple(tuple(value / sqrt(2) for value in row) for row in matrix)
    for matrix in _LAMBDA
)
_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_F = {}
for _indices, _value in (
    ((1, 2, 3), 1),
    ((1, 4, 7), 0.5),
    ((1, 5, 6), -0.5),
    ((2, 4, 6), 0.5),
    ((2, 5, 7), 0.5),
    ((3, 4, 5), 0.5),
    ((3, 6, 7), -0.5),
    ((4, 5, 8), sqrt(3) / 2),
    ((6, 7, 8), sqrt(3) / 2),
):
    for _permutation in permutations(range(3)):
        _sign = (-1) ** sum(
            _permutation[i] > _permutation[j] for i in range(3) for j in range(i + 1, 3)
        )
        _F[tuple(_indices[i] - 1 for i in _permutation)] = _sign * _value


def _matrix_product(left, right):
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _word_matrix(word, values):
    result = _IDENTITY
    for label in word:
        result = _matrix_product(result, _TAU[values[label]])
    return result


def _tensor_component(tensor, values):
    result = 1 + 0j
    for string in tensor.open_strings:
        result *= _word_matrix(string.adjoint_labels, values)[
            values[string.fundamental_label]
        ][values[string.antifundamental_label]]
    for trace in tensor.traces:
        matrix = _word_matrix(trace, values)
        result *= sum(matrix[index][index] for index in range(3))
    return result


def _assignments(legs):
    labels = tuple(leg.label for leg in legs)
    sizes = tuple(
        8 if leg.representation == 8 else 1 if leg.representation == 1 else 3
        for leg in legs
    )
    for indices in product(*(range(size) for size in sizes)):
        yield indices, dict(zip(labels, indices, strict=True))


def _explicit_state(tensor, connection):
    """Apply explicit physical 3x3 or 8x8 generators to colour components."""
    legs = connection.initial_legs
    state = {
        key: _tensor_component(tensor, values) for key, values in _assignments(legs)
    }
    for step, emission in enumerate(connection.emissions):
        out_legs = ColorConnection(
            connection.initial_legs, connection.emissions[: step + 1]
        ).output_legs
        out_state = {}
        if isinstance(emission, EmitGluon):
            representation = next(
                leg.representation
                for leg in legs
                if leg.label == emission.emitter_label
            )
            dimension = 8 if representation == 8 else 3
            for key, values in _assignments(out_legs):
                b, a = values[emission.emitted_label], values[emission.emitter_label]
                result = 0j
                for old in range(dimension):
                    charge = (
                        _T[b][a][old]
                        if representation == 3
                        else -_T[b][old][a]
                        if representation == -3
                        else -1j * _F.get((b, a, old), 0)
                    )
                    source = tuple(
                        old
                        if leg.label == emission.emitter_label
                        else values[leg.label]
                        for leg in legs
                    )
                    result += charge * state[source]
                out_state[key] = result
        else:
            for key, values in _assignments(out_legs):
                out_state[key] = sum(
                    _T[a][values[emission.quark_label]][
                        values[emission.antiquark_label]
                    ]
                    * state[
                        tuple(
                            a
                            if leg.label == emission.parent_label
                            else values[leg.label]
                            for leg in legs
                        )
                    ]
                    for a in range(8)
                )
        legs, state = out_legs, out_state
    return state


def _explicit_overlap(left_tensor, left_graph, right_tensor, right_graph):
    left = _explicit_state(left_tensor, left_graph)
    right = _explicit_state(right_tensor, right_graph)
    return sum(value.conjugate() * right[key] for key, value in left.items())


def test_physical_dipole_normalization_and_signed_offdiagonal():
    identity = ColorConnection(_QQ_LEGS)
    quark = ColorConnection(_QQ_LEGS, (EmitGluon(1, 3),))
    anti = ColorConnection(_QQ_LEGS, (EmitGluon(2, 3),))
    born = color_connection_matrix_element(_QQ, identity, _QQ, identity)
    assert born == ExactColorCoefficient(3)
    assert color_connection_matrix_element(_QQ, quark, _QQ, quark) == born.scaled(
        Fraction(4, 3)
    )
    assert color_connection_matrix_element(
        _QQ, anti, _QQ, anti
    ) == ExactColorCoefficient(4)
    assert color_connection_matrix_element(
        _QQ, quark, _QQ, anti
    ) == ExactColorCoefficient(-4)


@pytest.mark.parametrize(
    ("emissions", "expected"),
    (
        ((EmitGluon(1, 3),), {2: Fraction(1, 2), 0: Fraction(-1, 2)}),
        (
            (EmitGluon(1, 3), EmitGluon(1, 4)),
            {3: Fraction(1, 4), 1: Fraction(-1, 2), -1: Fraction(1, 4)},
        ),
        (
            (EmitGluon(1, 3), SplitGluon(3, 4, 5)),
            {2: Fraction(1, 4), 0: Fraction(-1, 4)},
        ),
        (
            (EmitGluon(1, 3), EmitGluon(1, 4), EmitGluon(1, 5)),
            {
                4: Fraction(1, 8),
                2: Fraction(-3, 8),
                0: Fraction(3, 8),
                -2: Fraction(-1, 8),
            },
        ),
        (
            (EmitGluon(1, 3), SplitGluon(3, 4, 5), EmitGluon(4, 6)),
            {3: Fraction(1, 8), 1: Fraction(-1, 4), -1: Fraction(1, 8)},
        ),
    ),
)
def test_laurent_coefficients_preserve_casimirs_and_splitting_through_n3lo(
    emissions, expected
):
    image = apply_color_connection(_QQ, ColorConnection(_QQ_LEGS, emissions))
    terms = contract_connected_tensor_nc_terms(image, image)
    assert terms == {
        power: ExactColorCoefficient(value) for power, value in expected.items()
    }
    assert evaluate_color_nc_terms(terms) == contract_connected_tensors(image, image)


def test_laurent_terms_are_not_cancelled_only_because_nc_is_three():
    # (Nc - 3) delta_ij is zero at Nc=3, but not the zero Laurent tensor.
    image = ConnectedColorTensor(
        _QQ_LEGS,
        (
            TensorTerm(ExactColorCoefficient(1), _QQ, 1),
            TensorTerm(ExactColorCoefficient(-3), _QQ),
        ),
    )
    assert len(image.terms) == 2
    terms = contract_connected_tensor_nc_terms(image, image)
    assert terms == {
        3: ExactColorCoefficient(1),
        2: ExactColorCoefficient(-6),
        1: ExactColorCoefficient(9),
    }
    assert evaluate_color_nc_terms(terms) == ExactColorCoefficient()


def test_empty_trace_retains_its_symbolic_nc_power():
    tensor = ColorTensor(_QQ.open_strings, ((),))
    image = apply_color_connection(tensor, ColorConnection(_QQ_LEGS))
    assert image.terms == (TensorTerm(ExactColorCoefficient(1), _QQ, 1),)
    assert contract_connected_tensor_nc_terms(image, image) == {
        3: ExactColorCoefficient(1)
    }


def test_all_outgoing_crossing_is_separate_and_dualizes_only_fundamentals():
    assert all_outgoing_color_leg(1, 3, incoming=True) == ColorLeg(1, -3)
    assert all_outgoing_color_leg(1, -3, incoming=True) == ColorLeg(1, 3)
    assert all_outgoing_color_leg(1, 8, incoming=True) == ColorLeg(1, 8)
    assert all_outgoing_color_leg(1, 1, incoming=True) == ColorLeg(1, 1)
    assert all_outgoing_color_leg(1, 3) == ColorLeg(1, 3)
    legs = (all_outgoing_color_leg(1, 3, incoming=True), ColorLeg(2, 3))
    tensor = ColorTensor((OpenColorString(2, (), 1),))
    crossed = ColorConnection(legs, (EmitGluon(1, 3),))
    outgoing = ColorConnection(legs, (EmitGluon(2, 3),))
    assert color_connection_matrix_element(
        tensor, crossed, tensor, outgoing
    ) == ExactColorCoefficient(-4)


def test_quark_antiquark_and_adjoint_charges_obey_exact_colour_conservation():
    legs = (*_QQ_LEGS, ColorLeg(3, 8))
    tensor = ColorTensor((OpenColorString(1, (3,), 2),))
    graphs = tuple(ColorConnection(legs, (EmitGluon(label, 4),)) for label in (1, 2, 3))
    images = tuple(apply_color_connection(tensor, graph) for graph in graphs)
    summed = ConnectedColorTensor(
        images[0].legs, tuple(term for image in images for term in image.terms), 1
    )
    assert summed.terms == ()
    for i, graph in enumerate(graphs):
        row = [
            color_connection_matrix_element(tensor, graph, tensor, other)
            for other in graphs
        ]
        assert sum(value.real for value in row) == 0
        assert all(value.imag == 0 for value in row)
        assert row[i] == ExactColorCoefficient(
            Fraction(8) * (3 if i == 2 else Fraction(4, 3))
        )
        for j, other in enumerate(graphs):
            assert complex(row[j]) == pytest.approx(
                _explicit_overlap(tensor, graph, tensor, other), abs=1e-12
            )


def test_ordered_emissions_do_not_commute():
    first = ColorConnection(_QQ_LEGS, (EmitGluon(1, 3), EmitGluon(1, 4)))
    reversed_order = ColorConnection(_QQ_LEGS, (EmitGluon(1, 4), EmitGluon(1, 3)))
    assert (
        apply_color_connection(_QQ, first).terms
        != apply_color_connection(_QQ, reversed_order).terms
    )
    assert color_connection_matrix_element(
        _QQ, first, _QQ, first
    ) == ExactColorCoefficient(Fraction(16, 3))
    cross = color_connection_matrix_element(_QQ, first, _QQ, reversed_order)
    assert cross == ExactColorCoefficient(Fraction(-2, 3))
    assert complex(cross) == pytest.approx(
        _explicit_overlap(_QQ, first, _QQ, reversed_order), abs=1e-12
    )


def test_shared_leg_dipole_products_and_documented_anticommutator():
    legs = (ColorLeg(1, 3), ColorLeg(2, -3), ColorLeg(3, 3), ColorLeg(4, -3))
    # Literal tensors: no generated-basis endpoint-pairing phase is attached.
    basis = (
        ColorTensor((OpenColorString(1, (), 2), OpenColorString(3, (), 4))),
        ColorTensor((OpenColorString(1, (), 4), OpenColorString(3, (), 2))),
    )
    bra = ColorConnection(legs, (EmitGluon(3, -1), EmitGluon(1, -2)))
    ket_ab = ColorConnection(legs, (EmitGluon(2, -2), EmitGluon(2, -1)))
    ket_ba = ColorConnection(legs, (EmitGluon(2, -1), EmitGluon(2, -2)))
    states = tuple(_explicit_state(tensor, ColorConnection(legs)) for tensor in basis)

    def charge(state, label, adjoint):
        position = label - 1
        result = {}
        for indices in state:
            value = 0j
            for old in range(3):
                source = list(indices)
                source[position] = old
                generator = (
                    _T[adjoint][indices[position]][old]
                    if legs[position].representation == 3
                    else -_T[adjoint][old][indices[position]]
                )
                value += generator * state[tuple(source)]
            result[indices] = value
        return result

    def dipole(state, first, second):
        result = dict.fromkeys(state, 0j)
        for adjoint in range(8):
            branch = charge(charge(state, second, adjoint), first, adjoint)
            for indices, value in branch.items():
                result[indices] += value
        return result

    # A = T2.T3 and B = T1.T2 share leg 2: their order cannot be exchanged.
    expected_ab = ((Fraction(0), Fraction(-2, 3)), (Fraction(16, 3), Fraction(0)))
    expected_ba = ((Fraction(0), Fraction(16, 3)), (Fraction(-2, 3), Fraction(0)))
    expected_anticommutator = (
        (Fraction(0), Fraction(14, 3)),
        (Fraction(14, 3), Fraction(0)),
    )
    for i, (left, left_state) in enumerate(zip(basis, states, strict=True)):
        for j, (right, right_state) in enumerate(zip(basis, states, strict=True)):
            ab = color_connection_matrix_element(left, bra, right, ket_ab)
            ba = color_connection_matrix_element(left, bra, right, ket_ba)
            assert ab == ExactColorCoefficient(expected_ab[i][j])
            assert ba == ExactColorCoefficient(expected_ba[i][j])
            assert ab + ba == ExactColorCoefficient(expected_anticommutator[i][j])
            assert ab == ExactColorCoefficient(expected_ba[j][i]).conjugate()
            # Independent physical generators act directly on Born colour
            # components; no connected-overlap matrix is multiplied here.
            for actual, explicit in (
                (ab, dipole(dipole(right_state, 1, 2), 2, 3)),
                (ba, dipole(dipole(right_state, 2, 3), 1, 2)),
            ):
                expected = sum(
                    value.conjugate() * explicit[indices]
                    for indices, value in left_state.items()
                )
                assert complex(actual) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("emissions", "expected"),
    (
        ((EmitGluon(1, 3), EmitGluon(1, 4), EmitGluon(1, 5)), Fraction(64, 9)),
        ((EmitGluon(1, 3), EmitGluon(3, 4), EmitGluon(4, 5)), Fraction(36)),
        ((EmitGluon(1, 3), SplitGluon(3, 4, 5), EmitGluon(4, 6)), Fraction(8, 3)),
        ((EmitGluon(1, 3), EmitGluon(3, 4), SplitGluon(4, 5, 6)), Fraction(6)),
    ),
)
def test_three_emissions_include_radiating_emitted_gluons_and_quark_pairs(
    emissions, expected
):
    graph = ColorConnection(_QQ_LEGS, emissions)
    image = apply_color_connection(_QQ, graph)
    exact = contract_connected_tensors(image, image)
    assert image.normalization_power == 3
    assert exact == ExactColorCoefficient(expected)
    assert complex(exact) == pytest.approx(
        _explicit_overlap(_QQ, graph, _QQ, graph), abs=2e-11
    )


@pytest.mark.parametrize("order", (4, 5))
def test_generic_order_same_line_casimir_powers(order):
    emissions = tuple(EmitGluon(1, -index) for index in range(1, order + 1))
    image = apply_color_connection(_QQ, ColorConnection(_QQ_LEGS, emissions))
    assert contract_connected_tensors(image, image) == ExactColorCoefficient(
        3 * Fraction(4, 3) ** order
    )


def test_four_operations_with_two_quark_pairs():
    graph = ColorConnection(_QQ_LEGS, (
        EmitGluon(1, -5), SplitGluon(-5, -1, -2),
        EmitGluon(1, -6), SplitGluon(-6, -3, -4),
    ))
    image = apply_color_connection(_QQ, graph)
    assert graph.order == 4
    assert contract_connected_tensors(image, image) == ExactColorCoefficient(
        3 * (Fraction(4, 3) * Fraction(1, 2)) ** 2
    )


def test_n3lo_quark_pair_cross_connections_match_explicit_generators():
    left = ColorConnection(
        _QQ_LEGS, (EmitGluon(1, 3), SplitGluon(3, 4, 5), EmitGluon(4, 6))
    )
    right = ColorConnection(
        _QQ_LEGS, (EmitGluon(2, 6), EmitGluon(1, 3), SplitGluon(3, 4, 5))
    )
    actual = color_connection_matrix_element(_QQ, left, _QQ, right)
    assert complex(actual) == pytest.approx(
        _explicit_overlap(_QQ, left, _QQ, right), abs=1e-12
    )
    assert color_connection_matrix_element(_QQ, right, _QQ, left) == actual.conjugate()


@pytest.mark.parametrize(
    ("prefix", "next_gluon"),
    (
        ((EmitGluon(1, 3),), 4),
        ((EmitGluon(1, 3), EmitGluon(3, 4)), 5),
        ((EmitGluon(1, 3), SplitGluon(3, 4, 5)), 6),
    ),
)
def test_nnlo_n3lo_colour_coherence_includes_every_active_emitted_parton(
    prefix, next_gluon
):
    intermediate = ColorConnection(_QQ_LEGS, prefix)
    images = tuple(
        apply_color_connection(
            _QQ,
            ColorConnection(_QQ_LEGS, (*prefix, EmitGluon(leg.label, next_gluon))),
        )
        for leg in intermediate.output_legs
    )
    # Charge conservation acts on the intermediate state, including auxiliary
    # gluons and both endpoints of an emitted quark pair. The new operation is
    # applied last on every branch, with unit sum weights and physical signs
    # supplied by the quark/antiquark/adjoint charge maps themselves.
    total = ConnectedColorTensor(
        images[0].legs,
        tuple(term for image in images for term in image.terms),
        len(prefix) + 1,
    )
    assert total.terms == ()
    for bra in images:
        row_sum = ExactColorCoefficient()
        for ket in images:
            row_sum = row_sum + contract_connected_tensors(bra, ket)
        assert row_sum == ExactColorCoefficient()

    # These examples would fail if a higher-order sum only included Born legs.
    born_only = ConnectedColorTensor(
        images[0].legs,
        tuple(
            term
            for leg, image in zip(intermediate.output_legs, images, strict=True)
            if leg.label in (1, 2)
            for term in image.terms
        ),
        len(prefix) + 1,
    )
    assert contract_connected_tensors(born_only, born_only).real > 0


def test_splitting_a_trace_keeps_open_plus_closed_trace_intermediates():
    legs = tuple(ColorLeg(label, 8) for label in (1, 2, 3))
    tensor = ColorTensor(traces=((1, 2, 3),))
    graph = ColorConnection(legs, (SplitGluon(1, 4, 5),))
    image = apply_color_connection(tensor, graph)
    assert len(image.terms) == 2
    assert any(term.tensor.open_strings and term.tensor.traces for term in image.terms)
    actual = contract_connected_tensors(image, image)
    assert complex(actual) == pytest.approx(
        _explicit_overlap(tensor, graph, tensor, graph), abs=1e-12
    )
    assert actual == ExactColorCoefficient(Fraction(28, 3))


def test_multiple_traces_and_conjugation_match_literal_generators():
    legs = tuple(ColorLeg(label, 8) for label in (1, 2, 3, 4))
    graph = ColorConnection(legs)
    left = ColorTensor(traces=((1, 2), (3, 4)))
    right = ColorTensor(traces=((1, 3, 2, 4),))
    actual = color_connection_matrix_element(left, graph, right, graph)
    assert complex(actual) == pytest.approx(
        _explicit_overlap(left, graph, right, graph), abs=1e-12
    )
    assert ColorTensor(traces=((2, 3, 1),)) == ColorTensor(traces=((1, 2, 3),))
    assert ColorTensor(traces=((1, 3, 2),)) != ColorTensor(traces=((1, 2, 3),))


def test_complex_trace_components_are_conjugated_by_reversing_the_bra_word():
    legs = tuple(ColorLeg(label, 8) for label in (1, 2, 3))
    identity = ColorConnection(legs)
    left = ColorTensor(traces=((1, 2, 3),))
    right = ColorTensor(traces=((1, 3, 2),))
    assert color_connection_matrix_element(
        left, identity, left, identity
    ) == ExactColorCoefficient(Fraction(56, 3))
    actual = color_connection_matrix_element(left, identity, right, identity)
    assert actual == ExactColorCoefficient(Fraction(-16, 3))
    assert complex(actual) == pytest.approx(
        _explicit_overlap(left, identity, right, identity), abs=1e-12
    )
    images = tuple(
        apply_color_connection(left, ColorConnection(legs, (EmitGluon(label, 4),)))
        for label in (1, 2, 3)
    )
    assert (
        ConnectedColorTensor(
            images[0].legs, tuple(term for image in images for term in image.terms), 1
        ).terms
        == ()
    )


def test_literal_endpoint_pairings_do_not_inject_a_fermion_permutation_sign():
    legs = (*_QQ_LEGS, ColorLeg(3, 3), ColorLeg(4, -3))
    graph = ColorConnection(legs)
    direct = ColorTensor((OpenColorString(1, (), 2), OpenColorString(3, (), 4)))
    crossed = ColorTensor((OpenColorString(1, (), 4), OpenColorString(3, (), 2)))
    assert color_connection_matrix_element(
        direct, graph, crossed, graph
    ) == ExactColorCoefficient(3)


def test_overlap_retains_complex_coefficients_and_bra_conjugation():
    graph = ColorConnection(_QQ_LEGS)
    actual = color_connection_matrix_element(
        _QQ,
        graph,
        _QQ,
        graph,
        left_coefficient=ExactColorCoefficient(1, 2),
        right_coefficient=ExactColorCoefficient(3, -1),
    )
    assert actual == ExactColorCoefficient(3, -21)


@pytest.mark.parametrize(
    "emissions",
    (
        (EmitGluon(9, 3),),
        (EmitGluon(1, 1),),
        (SplitGluon(1, 3, 4),),
        (EmitGluon(1, 3), SplitGluon(3, 4, 4)),
        (EmitGluon(1, 3), SplitGluon(3, 4, 5), EmitGluon(3, 6)),
        (EmitGluon(1, 3), SplitGluon(3, 4, 5), EmitGluon(1, 3)),
    ),
)
def test_invalid_ordered_graphs_are_rejected(emissions):
    with pytest.raises(ValueError):
        ColorConnection(_QQ_LEGS, emissions)


def test_mismatched_tensor_domains_and_graph_outputs_are_rejected():
    with pytest.raises(ValueError, match="exactly once"):
        apply_color_connection(ColorTensor(), ColorConnection(_QQ_LEGS))
    with pytest.raises(ValueError, match="typed output"):
        color_connection_matrix_element(
            _QQ,
            ColorConnection(_QQ_LEGS, (EmitGluon(1, 3),)),
            _QQ,
            ColorConnection(_QQ_LEGS, (EmitGluon(1, 4),)),
        )
    with pytest.raises(ValueError, match="odd"):
        contract_connected_tensors(
            ConnectedColorTensor(
                _QQ_LEGS, (TensorTerm(ExactColorCoefficient(1), _QQ),), 0
            ),
            ConnectedColorTensor(
                _QQ_LEGS, (TensorTerm(ExactColorCoefficient(1), _QQ),), 1
            ),
        )
