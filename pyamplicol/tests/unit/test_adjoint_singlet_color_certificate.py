# SPDX-License-Identifier: 0BSD
"""Small exact checks for DDM colour with a colourless operator insertion."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from typing import cast

import pytest

from pyamplicol.color import (
    build_color_plan,
    color_contraction_factor,
    exact_color_contraction_factor,
)
from pyamplicol.color.adjoint_kernel import adjoint_color_kernel
from pyamplicol.color.contraction_trace import _simplify_trace_terms_nc_power
from pyamplicol.models.adjoint_color_certificate import (
    _is_connected_adjoint_tree_tensor,
    compiled_adjoint_tree_color_basis_is_proven,
)
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.contracts import CompiledModelIR
from pyamplicol.models.tensors import normalize_color_expression


def _term(index, particles, source, orders=()):
    representations = tuple(1 if name == "spectator" else 8 for name in particles)
    return SimpleNamespace(
        id=index,
        particles=particles,
        color_source=source,
        color_expression=normalize_color_expression(source, representations).expression,
        coupling_orders=orders,
    )


def _theory(*, source="UFO::Identity(1,2)", mixed_order="EFT"):
    # Names, spin, Lorentz structures and physical meaning are immaterial:
    # this proof is exclusively about colour and insertion budgets.
    return cast(
        CompiledModelIR,
        SimpleNamespace(
            particles=(
                SimpleNamespace(name="adj", antiname="adj", pdg_code=21, color=8),
                SimpleNamespace(
                    name="spectator", antiname="spectator", pdg_code=23, color=1
                ),
            ),
            vertex_terms=(
                _term(0, ("adj",) * 3, "UFO::f(1,2,3)", (("STRONG", 1),)),
                _term(1, ("adj",) * 2 + ("spectator",), source, ((mixed_order, 1),)),
                _term(2, ("spectator",) * 3, "1", (("WEAK", 1),)),
            ),
        ),
    )


@pytest.mark.parametrize("process", ("g g > z", "g g > z g", "g g > z g g"))
def test_single_mixed_insertion_is_certified_without_yang_mills_helicity_claim(process):
    assert compiled_adjoint_tree_color_basis_is_proven(
        _theory(), build_process_ir(process, color_accuracy="full"),
        max_coupling_orders={"EFT": 1},
    )


@pytest.mark.parametrize("limits", ({}, {"EFT": 2}, {"STRONG": 0}))
@pytest.mark.parametrize("process", ("g g > g g", "g g > z g"))
def test_extra_singlet_exchange_is_not_mistaken_for_one_ddm_tree(limits, process):
    assert not compiled_adjoint_tree_color_basis_is_proven(
        _theory(), build_process_ir(process, color_accuracy="full"),
        max_coupling_orders=limits,
    )


def test_zero_insertion_budget_excludes_singlet_exchange_even_in_extended_model():
    assert compiled_adjoint_tree_color_basis_is_proven(
        _theory(), build_process_ir("g g > g g", color_accuracy="full"),
        max_coupling_orders={"EFT": 0},
    )


def test_quark_domain_is_not_certified_by_the_gluonic_tree_theorem():
    assert not compiled_adjoint_tree_color_basis_is_proven(
        _theory(), build_process_ir("d d~ > z g", color_accuracy="full"),
        max_coupling_orders={"EFT": 1},
    )


@pytest.mark.parametrize(
    ("source", "representations", "expected"),
    (
        ("UFO::Identity(1,2)", (8, 8, 1), True),
        ("UFO::f(1,2,3)", (8, 8, 8, 1), True),
        ("UFO::f(-1,1,2)*UFO::f(3,4,-1)", (8, 8, 8, 8, 1), True),
        ("UFO::f(-1,1,2)*UFO::f(3,4,-1)", (8, 8, 8, 8), True),
        ("UFO::Identity(1,2)*UFO::Identity(3,4)", (8, 8, 8, 8), False),
        ("UFO::d(1,2,3)", (8, 8, 8), False),
    ),
)
def test_tree_certificate_distinguishes_connected_and_symmetric_tensors(
    source, representations, expected
):
    expression = normalize_color_expression(source, representations).expression
    actual = _is_connected_adjoint_tree_tensor(source, expression, representations)
    assert actual is expected


def test_source_spelling_cannot_certify_a_different_actual_tensor():
    source = "UFO::f(1,2,3)"
    expression = normalize_color_expression("UFO::d(1,2,3)", (8, 8, 8)).expression
    assert not _is_connected_adjoint_tree_tensor(source, expression, (8, 8, 8))


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_singlet_labels_and_two_adjoint_identity_survive_ddm_planning(accuracy):
    process = build_process_ir("g g > z", color_accuracy=accuracy)
    adjoint = build_color_plan(process, color_accuracy=accuracy, basis="adjoint")
    trace = build_color_plan(process, color_accuracy=accuracy, basis="trace")
    assert adjoint.sector_count == trace.sector_count == 1
    assert adjoint.sectors[0].singlet_labels == (3,)
    assert adjoint.sectors[0].trace_labels == (1, 2)
    assert exact_color_contraction_factor(
        adjoint, adjoint.sectors[0], adjoint.sectors[0], accuracy=accuracy
    ) == exact_color_contraction_factor(
        trace, trace.sectors[0], trace.sectors[0], accuracy=accuracy
    ) == 8


@pytest.mark.parametrize("nc", (2, 3, 4))
def test_degenerate_adjoint_ladder_keeps_su_nc_identity(nc):
    assert adjoint_color_kernel(2, nc=nc)[0] == nc * nc - 1
    assert adjoint_color_kernel(2, nc=nc, accuracy="nlc")[0] == nc * nc - 1
    assert adjoint_color_kernel(2, nc=nc, accuracy="lc")[0] == nc * nc
    assert adjoint_color_kernel(2, nc=nc, full_col_acc=0)[0] == nc * nc


def test_two_adjoint_trace_nlc_keeps_the_complete_fierz_polynomial():
    # Independent Fierz reduction proves that there is no NNLC term to
    # discard. Nc^2 - n*Nc^(n-2) is not valid for the degenerate n=2 trace.
    polynomial = _simplify_trace_terms_nc_power(
        ((Fraction(1), 0, ((1, 2), (2, 1))),)
    )
    assert polynomial == {2: Fraction(1), 0: Fraction(-1)}
    process = build_process_ir("g g > z", color_accuracy="nlc")
    plan = build_color_plan(process, color_accuracy="nlc", basis="trace")
    sector = plan.sectors[0]
    assert exact_color_contraction_factor(plan, sector, sector, accuracy="nlc") == 8
    assert color_contraction_factor(plan, sector, sector, accuracy="nlc") == 8.0


def test_singlet_labels_are_retained_for_all_primitive_orderings():
    process = build_process_ir("g g > z g g", color_accuracy="full")
    adjoint = build_color_plan(process, color_accuracy="full", basis="adjoint")
    assert adjoint.sector_count == 2
    assert {sector.singlet_labels for sector in adjoint.sectors} == {(3,)}
    assert {sector.trace_labels for sector in adjoint.sectors} == {
        (1, 2, 4, 5), (1, 4, 2, 5)
    }
    reversed_plan = build_color_plan(
        replace(process, color_accuracy="nlc"), color_accuracy="nlc", basis="adjoint"
    )
    assert reversed_plan.sectors == adjoint.sectors
