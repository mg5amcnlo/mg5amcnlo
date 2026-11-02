# SPDX-License-Identifier: 0BSD
"""Call-owned DoubleFloat arithmetic with lossless Decimal input transport.

Decimal is the transport container, not the arithmetic backend. Every operation
is evaluated by Symbolica's double-double backend; no Decimal context or module
global is changed. The owner retains its small scalar evaluators locally.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Any, ClassVar

from pyamplicol.api.errors import EvaluationError
from pyamplicol.runtime.symbolica_exact import _upcast_decimal


class _DoubleDoubleScalar(Decimal):
    _arithmetic: ClassVar[DoubleDoubleArithmetic]

    def __new__(cls, value: Any = 0) -> _DoubleDoubleScalar:
        if isinstance(value, cls):
            return value
        return cls._arithmetic.operation("identity", Decimal(value))

    def __add__(self, other: Any) -> _DoubleDoubleScalar:
        return self._arithmetic.operation("add", self, other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> _DoubleDoubleScalar:
        return self._arithmetic.operation("sub", self, other)

    def __rsub__(self, other: Any) -> _DoubleDoubleScalar:
        return self._arithmetic.operation("sub", other, self)

    def __mul__(self, other: Any) -> _DoubleDoubleScalar:
        return self._arithmetic.operation("mul", self, other)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> _DoubleDoubleScalar:
        return self._arithmetic.operation("div", self, other)

    def __rtruediv__(self, other: Any) -> _DoubleDoubleScalar:
        return self._arithmetic.operation("div", other, self)

    def __neg__(self) -> _DoubleDoubleScalar:
        return Decimal.__new__(type(self), self.copy_negate())

    def __abs__(self) -> _DoubleDoubleScalar:
        return Decimal.__new__(type(self), self.copy_abs())

    def __pos__(self) -> _DoubleDoubleScalar:
        return self

    def __pow__(self, exponent: Any, modulo: Any = None) -> _DoubleDoubleScalar:
        if modulo is not None or int(exponent) != exponent:
            raise TypeError("double-double scalar powers require an integer")
        exponent = int(exponent)
        if exponent < 0:
            return type(self)(1) / self.__pow__(-exponent)
        result, factor = type(self)(1), self
        while exponent:
            if exponent & 1:
                result *= factor
            exponent >>= 1
            if exponent:
                factor *= factor
        return result

    def sqrt(self, context: Any = None) -> _DoubleDoubleScalar:
        return self._arithmetic.operation("sqrt", self)


class DoubleDoubleArithmetic:
    """An independently owned arithmetic backend, never a process-wide mode."""

    def __init__(self) -> None:
        self.scalar = type(
            "_OwnedDoubleDouble", (_DoubleDoubleScalar,), {"_arithmetic": self}
        )
        self._evaluators: dict[str, Any] = {}

    def operation(self, operation: str, *values: Any) -> _DoubleDoubleScalar:
        from symbolica import E

        from pyamplicol._internal.physics.symbols import symbols

        evaluator = self._evaluators.get(operation)
        if evaluator is None:
            parameters = [
                symbols.symbol(f"runtime::double_double::{name}")
                for name in (
                    ("x",) if operation in {"identity", "sqrt"} else ("x", "y")
                )
            ]
            expressions: dict[str, Callable[..., Any]] = {
                "identity": lambda x: x,
                "add": lambda x, y: x + y,
                "sub": lambda x, y: x - y,
                "mul": lambda x, y: x * y,
                "div": lambda x, y: x / y,
                "sqrt": lambda x: x ** E("1/2"),
            }
            expression = expressions[operation](*parameters)
            evaluator = expression.evaluator(parameters, iterations=0, n_cores=1)
            self._evaluators[operation] = evaluator
        # 32 selects Symbolica DoubleFloat, transported through a 106-bit float
        # (about 31 significant digits). Padding inputs adds no information.
        result = evaluator.evaluate_with_prec(
            [_upcast_decimal(Decimal(value), 80) for value in values], 32
        )[0].to_decimal()
        if not result.is_finite():
            raise EvaluationError(f"non-finite double-double {operation}")
        return Decimal.__new__(self.scalar, result)


def correlated_precision(precision: int, arithmetic: str) -> int:
    """Validate public arithmetic and return the amplitude transport precision."""
    if type(precision) is not int or precision < 1:
        raise EvaluationError("precision must be a positive number of decimal digits")
    if arithmetic == "double-double":
        if precision > 31:
            raise EvaluationError(
                "double-double results support at most 31 decimal digits"
            )
        return 31
    if arithmetic != "arbitrary":
        raise EvaluationError("arithmetic must be 'arbitrary' or 'double-double'")
    from pyamplicol.runtime.symbolica_exact import _working_precision

    return _working_precision(precision)
