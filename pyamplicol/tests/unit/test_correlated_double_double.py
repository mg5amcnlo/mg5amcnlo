# SPDX-License-Identifier: 0BSD
"""Arithmetic identity tests distinguish DoubleFloat from f64 and Decimal."""

from decimal import Decimal, getcontext, localcontext

import pytest

from pyamplicol.api.errors import EvaluationError
from pyamplicol.runtime._double_double import (
    DoubleDoubleArithmetic,
    correlated_precision,
)
from pyamplicol.runtime.correlated_exact import CorrelatedDoubleDoubleExecutor
from pyamplicol.runtime.correlations import _contract, _Matrix, _MatrixEntry
from pyamplicol.runtime.symbolica_exact import (
    _ExactEvaluator,
    _ExactExpressionEvaluator,
)


def test_double_double_retains_half_ulp_without_decimal_context_leak():
    pytest.importorskip("symbolica")
    first, second = DoubleDoubleArithmetic(), DoubleDoubleArithmetic()
    scalar = first.scalar
    original = getcontext().copy()
    with localcontext() as context:
        context.prec = 80
        half_ulp = Decimal(2) ** -53
        observed = scalar(1) + scalar(half_ulp)
        assert observed > 1  # Binary64 would round the midpoint to one.
        assert abs(Decimal(observed - scalar(1)) / half_ulp - 1) < Decimal("1e-14")
        perturbation = Decimal("1e-25")
        recovered = Decimal((scalar(1) + scalar(perturbation)) - scalar(1))
        # Symbolica transports DoubleFloat results at 106 binary bits. After
        # cancellation, bound the absolute rounding error near the original
        # value one; a relative 1e-14 bound here would demand 39 digits.
        assert abs(recovered - perturbation) <= Decimal(2) ** -105
        # Double-double cannot retain this difference; arbitrary Decimal can.
        assert scalar(1) + scalar("1e-40") == 1
        assert Decimal(1) + Decimal("1e-40") > 1
        assert (scalar(3).sqrt() ** 2 - scalar(3)).copy_abs() < Decimal("1e-29")
        assert abs(Decimal(scalar(2) ** -3) - Decimal("0.125")) < Decimal("1e-30")
    assert getcontext().prec == original.prec
    assert getcontext().rounding == original.rounding
    assert first.scalar is not second.scalar
    assert first._evaluators
    assert second._evaluators == {}


def test_derived_parameters_and_complex_stages_really_use_double_double():
    symbolica = pytest.importorskip("symbolica")
    with localcontext() as context:
        context.prec = 80
        half_ulp = Decimal(2) ** -53
        parameters = _ExactExpressionEvaluator(("x+1", "sqrt(pi)*x"), ("x",))
        values = parameters.evaluate_double_double(((half_ulp, Decimal(0)),))
        assert values[0][0] > 1
        expected = parameters.evaluate(((half_ulp, Decimal(0)),), 70)
        assert abs(values[1][0] / expected[1][0] - 1) < Decimal("1e-28")
        executor = object.__new__(CorrelatedDoubleDoubleExecutor)
        executor._double_double = DoubleDoubleArithmetic()
        executor._scalar = executor._double_double.scalar
        stage = _ExactEvaluator(1, symbolica.E("x+1").evaluator([symbolica.S("x")]))
        output = executor._evaluate_stage(stage, ((half_ulp, half_ulp),), 31)[0]
        assert output[0] > 1 and output[1] > 0
        assert type(output[0]) is executor._scalar
        # Chunked stages use the same backend, not a hidden Decimal fallback.
        chunked = _ExactEvaluator(1, chunks=(stage,), chunk_input_indices=((0,),))
        assert (
            executor._evaluate_stage(chunked, ((half_ulp, half_ulp),), 31)[0] == output
        )


@pytest.mark.parametrize(
    "precision, arithmetic",
    [
        (32, "double-double"),
        (100, "double-double"),
        (31, "unknown"),
        (True, "double-double"),
    ],
)
def test_invalid_arithmetic_is_rejected(precision, arithmetic):
    with pytest.raises(EvaluationError):
        correlated_precision(precision, arithmetic)


def test_colour_reduction_uses_double_double_weights_and_operations():
    from fractions import Fraction

    from pyamplicol.runtime.correlated_exact import (
        CoherentAmplitudeBatch,
        CoherentAmplitudeGroup,
    )

    pytest.importorskip("symbolica")
    scalar = DoubleDoubleArithmetic().scalar
    group = CoherentAmplitudeGroup(0, "h", (1,), (1,), 0)
    batch = CoherentAmplitudeBatch(
        (((scalar(1), scalar("1e-20")),),), (group,), scalar(1), 31, ()
    )
    matrix = _Matrix("test", (0,), (_MatrixEntry(0, 0, Fraction(1), Fraction(0)),))
    result = _contract(batch, matrix, precision=31)[0]
    assert result.real == 1  # DD drops the 1e-40 term; an Arb(40+) sum retains it.
    assert result.imag == 0
    assert type(result.real) is Decimal
