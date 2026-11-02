# SPDX-License-Identifier: 0BSD
"""Exact basis selection must not disguise fundamental chains as DDM."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from pyamplicol.color.fft_basis_strategy import best_general_fft_basis
from pyamplicol.color.plan import build_color_plan
from pyamplicol.models.builtin.process_ir import build_process_ir


@pytest.mark.parametrize(
    ("process", "name", "degree", "count"),
    (
        ("d d~ > z", "fundamental-chain", 0, 1),
        ("d d~ > z g g g", "fundamental-chain", 3, 6),
        ("d d~ > d d~", "fundamental-chain-products", 0, 2),
        ("d d~ > d d~ g", "fundamental-chain-products", 1, 4),
        ("d d~ > d d~ g g", "fundamental-chain-products", 2, 12),
        ("d d~ > d d~ g g g", "fundamental-chain-products", 3, 48),
        ("d d~ > u u~ s s~ g", "fundamental-chain-products", 1, 18),
        ("e- e+ > mu- mu+", "singlet", 0, 1),
    ),
)
def test_fundamental_and_singlet_strategy_is_explicit(process, name, degree, count):
    process_ir = build_process_ir(process, color_accuracy="full")
    for certificate in (False, True):
        strategy = best_general_fft_basis(process_ir, ddm_proven=certificate)
        assert strategy.actual_basis == "trace"
        assert strategy.name == name
        assert strategy.permutation_degree == degree
        assert strategy.tensor_count == count
    plan = build_color_plan(process_ir, color_accuracy="full")
    tensor_keys = {
        (sector.open_color_lines, sector.trace_labels, sector.singlet_labels)
        for sector in plan.sectors
    }
    assert len(tensor_keys) == count


@pytest.mark.parametrize("gluons", (3, 4, 5, 6))
def test_only_a_certificate_selects_ddm(gluons):
    process = build_process_ir(
        "g g > " + " ".join(["g"] * (gluons - 2)), color_accuracy="full"
    )
    ordinary = best_general_fft_basis(process, ddm_proven=False)
    assert ordinary.actual_basis == "trace"
    assert ordinary.name == "trace"
    assert ordinary.permutation_degree == gluons - 1
    assert ordinary.tensor_count == math.factorial(gluons - 1)
    certified = best_general_fft_basis(process, ddm_proven=True)
    assert certified.actual_basis == "adjoint"
    assert certified.name == "ddm"
    assert certified.permutation_degree == gluons - 2
    assert certified.tensor_count == math.factorial(gluons - 2)


def test_certified_singlet_spectators_do_not_change_the_ddm_group():
    process = build_process_ir("g g > g g h", color_accuracy="full")
    strategy = best_general_fft_basis(process, ddm_proven=True)
    assert (strategy.name, strategy.permutation_degree, strategy.tensor_count) == (
        "ddm",
        2,
        2,
    )


def test_two_certified_adjoint_legs_select_the_degree_zero_delta_metric():
    process = build_process_ir("g g > h", color_accuracy="full")
    strategy = best_general_fft_basis(process, ddm_proven=True)
    assert (
        strategy.actual_basis,
        strategy.name,
        strategy.permutation_degree,
        strategy.tensor_count,
    ) == (
        "adjoint",
        "ddm",
        0,
        1,
    )


def test_unbalanced_endpoints_are_not_certified_by_the_dispatcher():
    process = build_process_ir("d d~ > g g", color_accuracy="full")
    process = replace(
        process,
        legs=tuple(
            replace(leg, color_role="singlet")
            if leg.color_role == "antifundamental"
            else leg
            for leg in process.legs
        ),
    )
    strategy = best_general_fft_basis(process, ddm_proven=True)
    assert strategy.actual_basis == "trace"
    assert strategy.tensor_count is None
