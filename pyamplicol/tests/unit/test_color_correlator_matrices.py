# SPDX-License-Identifier: 0BSD
"""Generated-basis compatibility and directed exact correlator matrix tests."""

import json
from dataclasses import replace
from fractions import Fraction

import pytest

from pyamplicol.color import correlator_matrices
from pyamplicol.color.connections import (
    ColorConnection,
    ColorLeg,
    EmitGluon,
    ExactColorCoefficient,
    SplitGluon,
    apply_color_connection,
    contract_connected_tensor_nc_terms,
    evaluate_color_nc_terms,
)
from pyamplicol.color.contraction_factors import exact_color_contraction_factor
from pyamplicol.color.correlator_matrices import (
    ColorCorrelator,
    ColorCorrelatorMatrixEntry,
    build_color_correlator_matrices,
    build_color_correlator_matrix,
    process_color_legs,
    sector_color_tensor,
)
from pyamplicol.color.plan import build_color_plan
from pyamplicol.models.builtin.process_ir import build_process_ir


def _plan(expression, *, accuracy="full"):
    return build_color_plan(
        build_process_ir(expression, color_accuracy=accuracy), color_accuracy=accuracy
    )


def _entries(matrix):
    return {
        (entry.left_sector_id, entry.right_sector_id): entry.weight
        for entry in matrix.entries
    }


@pytest.mark.parametrize("accuracy", ("lc", "nlc", "full"))
@pytest.mark.parametrize(
    "expression",
    (
        "e- e+ > a a",
        "g g > g g",
        "d d~ > z g g",
        "d d~ > u u~",
        "d d~ > u u~ g g",
        "d d~ > d d~ g",
        "d d~ > u u~ s s~ g",
    ),
)
def test_identity_matches_every_existing_metric_entry(expression, accuracy):
    plan = _plan(expression)
    matrix = build_color_correlator_matrix(
        plan, ColorCorrelator("born"), color_accuracy=accuracy
    )
    entries = _entries(matrix)
    assert matrix.order == 0
    assert matrix.sector_ids == tuple(sector.id for sector in plan.sectors)
    assert matrix.to_json_dict()["color_accuracy"] == accuracy
    for left in plan.sectors:
        for right in plan.sectors:
            expected = exact_color_contraction_factor(
                plan, left, right, accuracy=accuracy
            )
            assert entries.get(
                (left.id, right.id), ExactColorCoefficient()
            ) == ExactColorCoefficient(expected)


def test_sector_adapter_adds_pairing_phase_and_ignores_traversal_replicas():
    plan = _plan("d d~ > u u~")
    terms = tuple(sector_color_tensor(sector, plan.process) for sector in plan.sectors)
    assert {term.coefficient.real for term in terms} == {Fraction(-1), Fraction(1)}
    assert len(set(terms)) == 2
    matrix = build_color_correlator_matrix(plan, ColorCorrelator("born"))
    assert any(entry.weight.real == -3 for entry in matrix.entries)


def test_literal_identity_matches_crossed_multiquark_metric():
    plan = _plan("d d~ > u u~ g")
    identity = ColorConnection(process_color_legs(plan.process))
    images = []
    for sector in plan.sectors:
        term = sector_color_tensor(sector, plan.process)
        images.append(
            apply_color_connection(term.tensor, identity, coefficient=term.coefficient)
        )
    # The Born matrix bypasses the literal adapter and uses the inherited
    # metric directly. Contract its tensor images independently to protect
    # crossed endpoints, pairing phases and gluon placement in the adapter.
    for left, bra in zip(plan.sectors, images, strict=True):
        for right, ket in zip(plan.sectors, images, strict=True):
            actual = evaluate_color_nc_terms(
                contract_connected_tensor_nc_terms(bra, ket)
            )
            expected = exact_color_contraction_factor(
                plan, left, right, accuracy="full"
            )
            assert actual == ExactColorCoefficient(expected)


def test_process_roles_are_already_crossed_and_include_singlets():
    process = _plan("d d~ > z").process
    assert process_color_legs(process) == (
        ColorLeg(1, -3),
        ColorLeg(2, 3),
        ColorLeg(3, 1),
    )
    invalid = replace(
        process,
        legs=(replace(process.legs[0], color_role="inclusive"), *process.legs[1:]),
    )
    with pytest.raises(ValueError, match="concrete"):
        process_color_legs(invalid)


def test_dipole_convenience_and_n3lo_emitted_pair_normalization():
    plan = _plan("d d~ > z")
    pair = ColorCorrelator(
        "pair",
        (
            EmitGluon(2, -1),
            SplitGluon(-1, -2, -3),
            EmitGluon(-2, -4),
        ),
        (
            EmitGluon(2, -1),
            SplitGluon(-1, -2, -3),
            EmitGluon(-2, -4),
        ),
    )
    born, dipole, triple = build_color_correlator_matrices(
        plan,
        (
            ColorCorrelator("born"),
            ColorCorrelator.dipole("B12", 1, 2),
            pair,
        ),
    )
    assert born.entries[0].weight == ExactColorCoefficient(3)
    assert dipole.entries[0].weight == ExactColorCoefficient(-4)
    assert triple.order == 3
    assert triple.entries[0].weight == ExactColorCoefficient(Fraction(8, 3))
    assert ColorLeg(-2, 3) in triple.output_legs
    assert ColorLeg(-3, -3) in triple.output_legs
    assert all(leg.label != -1 for leg in triple.output_legs)
    assert ColorCorrelator.dipole("B12", 1, 2).to_json_dict() == {
        "id": "B12",
        "bra": [{"kind": "emit-gluon", "emitter": 1, "emitted": -1}],
        "ket": [{"kind": "emit-gluon", "emitter": 2, "emitted": -1}],
    }
    assert pair.to_json_dict()["bra"][1] == {
        "kind": "split-gluon",
        "parent": -1,
        "quark": -2,
        "antiquark": -3,
    }


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_ordered_connections_store_both_triangles_without_hermiticity_assumptions(
    accuracy,
):
    plan = _plan("d d~ > g g")
    request = ColorCorrelator(
        "ordered",
        (EmitGluon(2, -1), EmitGluon(1, -2)),
        (EmitGluon(3, -2), EmitGluon(1, -1)),
    )
    matrix, adjoint = build_color_correlator_matrices(
        plan,
        (
            request,
            ColorCorrelator("adjoint", request.ket, request.bra),
        ),
        color_accuracy=accuracy,
    )
    entries = _entries(matrix)
    expected = {
        (0, 0): ExactColorCoefficient(Fraction(-16, 3)),
        (0, 1): ExactColorCoefficient(Fraction(-16, 3)),
        (1, 0): ExactColorCoefficient(Fraction(20, 3)),
    }
    if accuracy == "full":
        expected[1, 1] = ExactColorCoefficient(Fraction(2, 3))
    assert entries == expected
    assert _entries(adjoint) == {
        (j, i): value.conjugate() for (i, j), value in entries.items()
    }
    amplitudes = (ExactColorCoefficient(1), ExactColorCoefficient(0, 1))
    value = ExactColorCoefficient()
    for entry in matrix.entries:
        value = (
            value
            + amplitudes[entry.left_sector_id].conjugate()
            * entry.weight
            * amplitudes[entry.right_sector_id]
        )
    assert value == ExactColorCoefficient(
        Fraction(-14 if accuracy == "full" else -16, 3), -12
    )
    assert not build_color_correlator_matrix(plan, request, color_accuracy="lc").entries


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_insertions_can_revive_zero_ordinary_metric_entries(accuracy):
    plan = _plan("d d~ > u u~ g")
    born, inserted = build_color_correlator_matrices(
        plan,
        (
            ColorCorrelator("born"),
            ColorCorrelator.dipole("B13", 1, 3),
        ),
        color_accuracy=accuracy,
    )
    assert (0, 2) not in _entries(born)
    assert _entries(inserted)[0, 2] == ExactColorCoefficient(-4)


def test_selected_owner_ids_preserve_requested_coordinates():
    plan = _plan("g g > g g")
    matrix = build_color_correlator_matrix(
        plan, ColorCorrelator("born"), sector_ids=(5, 0)
    )
    assert matrix.sector_ids == (5, 0)
    assert {
        (entry.left_sector_id, entry.right_sector_id) for entry in matrix.entries
    } <= {(5, 5), (5, 0), (0, 5), (0, 0)}


def test_graph_images_are_reused_across_requested_matrices(monkeypatch):
    plan = _plan("d d~ > g g")
    original = correlator_matrices.apply_color_connection
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1].emissions)
        return original(*args, **kwargs)

    monkeypatch.setattr(correlator_matrices, "apply_color_connection", counted)
    build_color_correlator_matrices(
        plan,
        (
            ColorCorrelator("born"),
            ColorCorrelator.dipole("B13", 1, 3),
            ColorCorrelator.dipole("B31", 3, 1),
        ),
    )
    # The identity reuses the inherited metric without constructing images.
    assert len(calls) == 2 * len(plan.sectors)


@pytest.mark.parametrize(
    ("operations", "lc", "full"),
    (
        ((EmitGluon(2, -1),), Fraction(9, 2), Fraction(4)),
        ((EmitGluon(2, -1), EmitGluon(2, -2)), Fraction(27, 4), Fraction(16, 3)),
        ((EmitGluon(2, -1), SplitGluon(-1, -2, -3)), Fraction(9, 4), Fraction(2)),
        (
            (EmitGluon(2, -1), EmitGluon(2, -2), EmitGluon(2, -3)),
            Fraction(81, 8),
            Fraction(64, 9),
        ),
        (
            (EmitGluon(2, -1), SplitGluon(-1, -2, -3), EmitGluon(-2, -4)),
            Fraction(27, 8),
            Fraction(8, 3),
        ),
    ),
)
def test_connected_family_power_counting_includes_splitting(operations, lc, full):
    plan = _plan("d d~ > z")
    request = ColorCorrelator("self", operations, operations)
    for accuracy, expected in (("lc", lc), ("nlc", full), ("full", full)):
        matrix = build_color_correlator_matrix(plan, request, color_accuracy=accuracy)
        assert _entries(matrix) == {(0, 0): ExactColorCoefficient(expected)}


def test_pure_adjoint_nlc_truncates_but_split_output_keeps_admitted_exact_entry():
    plan = _plan("g g > g")
    emission = (EmitGluon(1, -1),)
    pair = (*emission, SplitGluon(-1, -2, -3))
    for operations, expected in ((emission, 54), (pair, 28)):
        matrix = build_color_correlator_matrix(
            plan, ColorCorrelator("self", operations, operations), color_accuracy="nlc"
        )
        assert _entries(matrix)[0, 0] == ExactColorCoefficient(expected)


def test_nlc_selection_includes_odd_powers_and_does_not_promote_suppressed_entries():
    polynomial = {
        3: ExactColorCoefficient(-1),
        1: ExactColorCoefficient(2),
        -1: ExactColorCoefficient(-1),
    }
    select = correlator_matrices._select_color_nc_terms
    assert (
        select(
            polynomial, color_accuracy="lc", leading_power=4, fundamental_output=True
        )
        == {}
    )
    assert (
        select(
            polynomial, color_accuracy="nlc", leading_power=4, fundamental_output=True
        )
        == polynomial
    )
    assert select(
        polynomial, color_accuracy="nlc", leading_power=4, fundamental_output=False
    ) == {3: ExactColorCoefficient(-1)}
    assert (
        select(
            {1: ExactColorCoefficient(1)},
            color_accuracy="nlc",
            leading_power=4,
            fundamental_output=True,
        )
        == {}
    )


@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
@pytest.mark.parametrize(
    ("prefix", "emitted"),
    (
        ((EmitGluon(2, -1),), -2),
        ((EmitGluon(2, -1), EmitGluon(-1, -2)), -3),
        ((EmitGluon(2, -1), SplitGluon(-1, -2, -3)), -4),
    ),
)
def test_connected_nnlo_n3lo_coherence_holds_through_retained_order(
    accuracy, prefix, emitted
):
    plan = _plan("d d~ > z")
    tensor = sector_color_tensor(plan.sectors[0], plan.process)
    legs = process_color_legs(plan.process)
    intermediate = ColorConnection(legs, prefix)
    images = tuple(
        apply_color_connection(
            tensor.tensor,
            ColorConnection(legs, (*prefix, EmitGluon(leg.label, emitted))),
            coefficient=tensor.coefficient,
        )
        for leg in intermediate.output_legs
        if leg.representation != 1
    )
    leading = 1 + len(prefix) + 1 - sum(isinstance(step, SplitGluon) for step in prefix)
    threshold = leading - (2 if accuracy == "nlc" else 0)
    for bra in images:
        summed = {}
        for ket in images:
            selected = correlator_matrices._select_color_nc_terms(
                contract_connected_tensor_nc_terms(bra, ket),
                color_accuracy=accuracy,
                leading_power=leading,
                fundamental_output=True,
            )
            for power, coefficient in selected.items():
                summed[power] = summed.get(power, ExactColorCoefficient()) + coefficient
        assert not {
            power: value
            for power, value in summed.items()
            if value and power >= threshold
        }
        if accuracy == "lc":
            assert evaluate_color_nc_terms(summed) == ExactColorCoefficient()


def test_invalid_output_accuracy_is_rejected():
    with pytest.raises(ValueError, match="accuracy"):
        build_color_correlator_matrix(
            _plan("d d~ > z"), ColorCorrelator("born"), color_accuracy="unknown"
        )


def test_json_is_exact_primitive_data_and_preserves_imaginary_weights():
    matrix = build_color_correlator_matrix(_plan("d d~ > z"), ColorCorrelator("born"))
    complex_entry = ColorCorrelatorMatrixEntry(
        0, 0, ExactColorCoefficient(Fraction(2**70, 3), Fraction(-2, 7))
    )
    matrix = replace(matrix, entries=(complex_entry,))
    payload = matrix.to_json_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["storage"] == "sparse-directed"
    assert payload["includes_color_factor"] is True
    assert payload["entries"] == [
        {
            "left_sector_id": 0,
            "right_sector_id": 0,
            "weight": {"real": [str(2**70), "3"], "imag": ["-2", "7"]},
        }
    ]


@pytest.mark.parametrize("id", ("", "  ", None, 3))
def test_correlator_ids_must_be_nonempty_strings(id):
    with pytest.raises(ValueError, match="ID"):
        ColorCorrelator(id)


def test_duplicate_request_ids_and_incompatible_outputs_are_rejected():
    plan = _plan("d d~ > z")
    with pytest.raises(ValueError, match="unique"):
        build_color_correlator_matrices(
            plan, (ColorCorrelator("x"), ColorCorrelator("x"))
        )
    with pytest.raises(ValueError, match="incompatible"):
        build_color_correlator_matrix(
            plan, ColorCorrelator("x", (EmitGluon(1, -1),), (EmitGluon(1, -2),))
        )
    with pytest.raises(ValueError, match="same number"):
        ColorCorrelator("x", (EmitGluon(1, -1),))
    with pytest.raises(ValueError, match="active coloured"):
        build_color_correlator_matrix(plan, ColorCorrelator.dipole("singlet", 1, 3))


@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
def test_non_full_plans_are_rejected(accuracy):
    with pytest.raises(ValueError, match="full-colour"):
        build_color_correlator_matrix(
            _plan("g g > g g", accuracy=accuracy), ColorCorrelator("born")
        )


@pytest.mark.parametrize(
    "correlator", (ColorCorrelator("born"), ColorCorrelator.dipole("T12", 1, 2))
)
@pytest.mark.parametrize("plural", (False, True))
def test_adjoint_plans_are_rejected_before_literal_tensor_conversion(
    correlator, plural, monkeypatch
):
    plan = build_color_plan(
        build_process_ir("g g > g g", color_accuracy="full"),
        color_accuracy="full",
        basis="adjoint",
    )

    def unexpected_tensor_conversion(*_args):
        pytest.fail("adjoint ordering records are not literal trace tensors")

    monkeypatch.setattr(
        correlator_matrices, "_sector_color_tensor", unexpected_tensor_conversion
    )
    with pytest.raises(ValueError, match="do not support the adjoint DDM basis"):
        if plural:
            build_color_correlator_matrices(plan, (correlator,))
        else:
            build_color_correlator_matrix(plan, correlator)


@pytest.mark.parametrize(
    "change", ({"truncated": True}, {"trace_reflections_folded": True}, {"sectors": ()})
)
def test_incomplete_or_folded_plans_are_rejected(change):
    with pytest.raises(ValueError, match="complete unfolded"):
        build_color_correlator_matrix(
            replace(_plan("g g > g g"), **change), ColorCorrelator("born")
        )


@pytest.mark.parametrize("ids", ((), (0, 0), (999,), (True,)))
def test_invalid_selected_sector_ids_are_rejected(ids):
    with pytest.raises(ValueError, match="sector IDs"):
        build_color_correlator_matrix(
            _plan("g g > g g"), ColorCorrelator("born"), sector_ids=ids
        )


def test_empty_correlator_request_does_not_touch_the_ordinary_lc_path():
    assert build_color_correlator_matrices(_plan("g g > g g", accuracy="lc"), ()) == ()
