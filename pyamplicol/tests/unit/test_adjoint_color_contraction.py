# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from pyamplicol.color import (
    ColorGroupDescriptor,
    LCColorSector,
    build_color_plan,
    build_symmetric_group_color_contraction_plan,
    certify_symmetric_group_orbits,
    color_contraction_factor,
    exact_color_contraction_factor,
    reconstruct_symmetric_group_dense_exact,
)
from pyamplicol.color.adjoint_kernel import adjoint_color_kernel
from pyamplicol.models.builtin.process_ir import build_process_ir


def _plan(n: int, *, basis: str = "adjoint", accuracy: str = "full", **kwargs):
    process = build_process_ir(
        "g g > " + " ".join(["g"] * (n - 2)), color_accuracy=accuracy
    )
    return build_color_plan(process, color_accuracy=accuracy, basis=basis, **kwargs)


def _expanded_chain(word):
    """Independent literal expansion of Tr([...[T1,T2],...] Tn)."""

    terms = [(1, word[:1])]
    for label in word[1:-1]:
        terms = [
            term
            for sign, current in terms
            for term in ((sign, (*current, label)), (-sign, (label, *current)))
        ]
    result = []
    for sign, current in terms:
        trace = (*current, word[-1])
        pivot = trace.index(word[0])
        trace = trace[pivot:] + trace[:pivot]
        result.append(
            (sign, LCColorSector(id=0, kind="single-trace", trace_labels=trace))
        )
    return result


def _trace_expanded_overlap(trace_plan, left, right, accuracy):
    return sum(
        (
            left_sign
            * right_sign
            * exact_color_contraction_factor(
                trace_plan, left_trace, right_trace, accuracy=accuracy
            )
            for (left_sign, left_trace), (right_sign, right_trace) in product(
                _expanded_chain(left.trace_labels), _expanded_chain(right.trace_labels)
            )
        ),
        Fraction(0),
    )


@pytest.mark.parametrize("n", (4, 5))
@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_adjoint_kernel_matches_literal_su3_commutator_expansion(n, accuracy):
    plan = _plan(n, accuracy=accuracy)
    trace_plan = _plan(n, basis="trace", accuracy=accuracy)
    for left, right in product(plan.sectors, repeat=2):
        expected = _trace_expanded_overlap(trace_plan, left, right, accuracy)
        assert (
            exact_color_contraction_factor(plan, left, right, accuracy=accuracy)
            == expected
        )
        assert color_contraction_factor(plan, left, right, accuracy=accuracy) == float(
            expected
        )


def test_nlc_truncates_the_transformed_metric_at_fixed_primitive_amplitudes():
    plan = _plan(6, accuracy="nlc")
    trace_plan = _plan(6, basis="trace", accuracy="nlc")
    full = adjoint_color_kernel(6)
    nlc = adjoint_color_kernel(6, accuracy="nlc")
    differing = np.flatnonzero(full != nlc)
    assert len(differing) > 0
    for index in (0, int(differing[0]), len(plan.sectors) - 1):
        left, right = plan.sectors[0], plan.sectors[index]
        assert exact_color_contraction_factor(plan, left, right, accuracy="nlc") == (
            _trace_expanded_overlap(trace_plan, left, right, "nlc")
        )


@pytest.mark.parametrize("n", (4, 5, 6))
@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_adjoint_orbit_is_two_anchor_and_reconstructs_exact_metric(n, accuracy):
    plan = _plan(n, accuracy=accuracy)
    descriptors = tuple(
        ColorGroupDescriptor(i, (), sector.id, sector.trace_labels, 1.0)
        for i, sector in enumerate(plan.sectors)
    )
    partition = certify_symmetric_group_orbits(plan, tuple(range(plan.sector_count)))
    assert partition.fixed_adjoint_labels == (1, n)
    assert partition.permuted_adjoint_labels == tuple(range(2, n))
    assert partition.residual_sector_ids == ()
    block = build_symmetric_group_color_contraction_plan(
        plan, descriptors
    ).symmetric_group_block
    assert block is not None
    assert block.degree == n - 2
    assert block.channel_count == 1
    assert len(block.kernel_entries) == math.factorial(n - 2)
    dense = reconstruct_symmetric_group_dense_exact(block)
    for left in range(plan.sector_count):
        for right in range(left, plan.sector_count):
            assert dense[(left, right)] == exact_color_contraction_factor(
                plan, plan.sectors[left], plan.sectors[right], accuracy=accuracy
            )


@pytest.mark.parametrize("n", (3, 4, 5, 6))
def test_adjoint_kernel_normalization_and_read_only_storage(n):
    kernel = adjoint_color_kernel(n)
    assert len(kernel) == math.factorial(n - 2)
    assert kernel[0] == (3**2 - 1) * (2 * 3) ** (n - 2)
    assert not kernel.flags.writeable


@pytest.mark.parametrize("nc", (2, 4))
def test_adjoint_finite_differences_preserve_general_nc_normalization(nc):
    kernel = adjoint_color_kernel(6, nc=nc)
    assert kernel[0] == (nc**2 - 1) * (2 * nc) ** 4


def test_adjoint_full_colour_cutoff_preserves_scalar_factor_contract():
    np.testing.assert_array_equal(
        adjoint_color_kernel(6, full_col_acc=0),
        adjoint_color_kernel(6, accuracy="lc"),
    )
    np.testing.assert_array_equal(
        adjoint_color_kernel(6, full_col_acc=1),
        adjoint_color_kernel(6, accuracy="nlc"),
    )
    plan = _plan(6)
    left, right = plan.sectors[0], plan.sectors[7]
    assert exact_color_contraction_factor(
        plan, left, right, accuracy="full", full_col_acc=1
    ) == exact_color_contraction_factor(plan, left, right, accuracy="nlc")


def test_adjoint_plan_keeps_basis_separate_from_primitive_words():
    trace = _plan(5, basis="trace")
    adjoint = _plan(5)
    assert "basis" not in trace.to_json_dict()
    assert adjoint.to_json_dict()["basis"] == "adjoint"
    assert len(trace.sectors) == 24
    assert len(adjoint.sectors) == 6
    assert all(sector.kind == "single-trace" for sector in adjoint.sectors)
    assert all(sector.trace_labels[::4] == (1, 5) for sector in adjoint.sectors)
    reordered = _plan(5, reference_color_order=(1, 3, 2, 4, 5))
    assert reordered.sectors[0].trace_labels == (1, 3, 2, 4, 5)
    assert len(reordered.sectors) == 6
    cyclic = _plan(5, reference_color_order=(3, 2, 4, 5, 1))
    assert cyclic.sectors == reordered.sectors
    with pytest.raises(ValueError, match="both anchors"):
        _plan(5, reference_color_order=(1, 3, 2, 5, 4))
    truncated = _plan(5, max_sectors=2)
    assert truncated.truncated and len(truncated.sectors) == 2
    empty = _plan(5, max_sectors=0, reference_color_order=(1, 3, 2, 4, 5))
    assert empty.truncated and empty.sectors == ()


def test_ddm_plan_rejects_fundamental_domains_and_lc():
    with pytest.raises(ValueError, match="adjoint"):
        process = build_process_ir("d d~ > g g", color_accuracy="full")
        build_color_plan(process, color_accuracy="full", basis="adjoint")
    with pytest.raises(ValueError, match="NLC or full"):
        _plan(4, accuracy="lc")
    with pytest.raises(ValueError, match="at most 12"):
        _plan(13)
    with pytest.raises(ValueError, match="int64"):
        adjoint_color_kernel(12, nc=100)
