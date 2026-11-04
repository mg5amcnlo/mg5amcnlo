# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Number
from typing import TYPE_CHECKING, Any

from .contracts import (
    CompiledParameterRecord,
)
from .expressions import (
    _exact_complex_expression,
    _imaginary_unit_for,
    _minkowski_square_expression,
)

if TYPE_CHECKING:
    pass

from . import compiler_symbolica as _sym


def _record_default(record: CompiledParameterRecord) -> complex:
    if record.value is not None:
        return complex(record.value[0], record.value[1])
    return complex(_sym.E(record.resolved_expression).evaluate({}))


def _spin_dimension(spin: int) -> int:
    try:
        return {-1: 1, 1: 1, 2: 4, 3: 4, 5: 16}[int(spin)]
    except KeyError as exc:
        raise ValueError(f"unsupported UFO spin code {spin}") from exc


def _chirality_tag(chirality: int) -> str:
    value = int(chirality)
    if value < 0:
        return f"m{abs(value)}"
    if value > 0:
        return f"p{value}"
    return "z"


def _replace_symbols(
    expression: _sym.Expression, substitutions: Mapping[_sym.Expression, Any]
) -> Any:
    symbols = set(expression.get_all_symbols(False))
    replacements = [
        _sym.Replacement(symbol, value)
        for symbol, value in substitutions.items()
        if symbol in symbols
    ]
    if not replacements:
        return expression
    return expression.replace_multiple(replacements)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, Number)


def _is_zero(value: Any) -> bool:
    if isinstance(value, int | float | complex):
        return value == 0
    return isinstance(value, _sym.Expression) and value.to_canonical_string() == "0"


def _expr_spin2_propagator(
    value: Sequence[Any],
    momentum: Sequence[Any],
    mass: Any,
    width: Any,
    *,
    dimension: Any,
    massive: bool,
) -> tuple[Any, ...]:
    if len(value) != 16:
        raise ValueError("spin-2 propagator expects sixteen current components")
    if len(momentum) != 4:
        raise ValueError("spin-2 propagator expects four momentum components")
    tensor = tuple(tuple(value[4 * mu + nu] for nu in range(4)) for mu in range(4))
    metric = (1, -1, -1, -1)
    imaginary = _imaginary_unit_for((*value, *momentum, mass, width, dimension))
    one = 1 if isinstance(imaginary, complex) else _exact_complex_expression(1 + 0j)
    if not isinstance(imaginary, complex) and isinstance(dimension, float | complex):
        dimension = _exact_complex_expression(complex(dimension))
    half = one / 2
    denominator = (
        _minkowski_square_expression(momentum) - mass * mass + imaginary * mass * width
    )
    if not massive:
        trace = sum(
            (metric[index] * tensor[index][index] for index in range(4)),
            0,
        )
        trace_weight = one / (dimension - 2)
        projected = tuple(
            half * (tensor[mu][nu] + tensor[nu][mu])
            - (metric[mu] * trace * trace_weight if mu == nu else 0)
            for mu in range(4)
            for nu in range(4)
        )
        return tuple(imaginary * component / denominator for component in projected)

    mass_squared = mass * mass
    first_projected = tuple(
        tuple(
            tensor[mu][nu]
            - momentum[mu]
            * sum(
                (
                    metric[alpha] * momentum[alpha] * tensor[alpha][nu]
                    for alpha in range(4)
                ),
                0,
            )
            / mass_squared
            for nu in range(4)
        )
        for mu in range(4)
    )
    transverse = tuple(
        tuple(
            first_projected[mu][nu]
            - momentum[nu]
            * sum(
                (
                    metric[beta] * momentum[beta] * first_projected[mu][beta]
                    for beta in range(4)
                ),
                0,
            )
            / mass_squared
            for nu in range(4)
        )
        for mu in range(4)
    )
    transverse_trace = sum(
        (metric[index] * transverse[index][index] for index in range(4)),
        0,
    )
    projected = tuple(
        half * (transverse[mu][nu] + transverse[nu][mu])
        - ((metric[mu] if mu == nu else 0) - momentum[mu] * momentum[nu] / mass_squared)
        * transverse_trace
        / 3
        for mu in range(4)
        for nu in range(4)
    )
    return tuple(imaginary * component / denominator for component in projected)
