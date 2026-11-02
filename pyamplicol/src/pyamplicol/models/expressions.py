# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from numbers import Number
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import CouplingOrders, Model, Vertex


def _number(value: complex | float) -> Any:
    from symbolica import Expression

    return Expression.num(value)


@lru_cache(maxsize=128)
def _exact_complex_expression(value: complex) -> Any:
    """Represent a stored binary64 weight without rounding symbolic factors.

    In particular, multiplying a rational coefficient by a Python complex
    unit must not turn that coefficient into a floating-point approximation.
    """
    from symbolica import E

    real_numerator, real_denominator = value.real.as_integer_ratio()
    imag_numerator, imag_denominator = value.imag.as_integer_ratio()
    return E(
        f"({real_numerator}/{real_denominator})"
        f"+1i*({imag_numerator}/{imag_denominator})"
    )


def _imaginary_unit_for(values: Sequence[Any]) -> Any:
    """Keep symbolic propagator phases exact without changing numeric inputs."""
    if all(isinstance(value, Number) for value in values):
        return 1j
    from symbolica import Expression

    if any(isinstance(value, Expression) for value in values):
        return _exact_complex_expression(1j)
    return 1j


def _as_expression(value: Any) -> Any:
    if isinstance(value, int | float | complex):
        return _number(value)
    return value


def _minkowski_square_expression(momentum: Sequence[Any]) -> Any:
    if len(momentum) != 4:
        raise ValueError("Minkowski momentum needs four components")
    if all(isinstance(value, int | float | complex) for value in momentum):
        p0, p1, p2, p3 = momentum
        return p0 * p0 - p1 * p1 - p2 * p2 - p3 * p3
    p0, p1, p2, p3 = (_as_expression(value) for value in momentum)
    return p0 * p0 - p1 * p1 - p2 * p2 - p3 * p3


def _model_vertex_result_chiralities(
    model: Model,
    vertex: Vertex,
    left_index: Any,
    right_index: Any,
) -> tuple[int, ...]:
    resolver = getattr(model, "_vertex_result_chiralities", None)
    if resolver is None:
        return (0,)
    return tuple(resolver(vertex, left_index, right_index))


def _index_particle_id(index: Any) -> int:
    if hasattr(index, "particle_id"):
        return int(index.particle_id)
    return int(index.pdg)


def _index_flavour_flow(index: Any) -> tuple[int, ...]:
    flow = getattr(index, "flavour_flow", None)
    if flow is None:
        return (_index_particle_id(index),)
    return tuple(int(value) for value in flow)


def _index_coupling_orders(index: Any) -> CouplingOrders:
    orders = getattr(index, "coupling_orders", None)
    if orders is None:
        return ()
    return tuple(
        sorted(
            (str(name).upper(), int(value)) for name, value in orders if int(value) != 0
        )
    )


def _append_flavour_transition(
    flow: tuple[int, ...],
    result_particle: int,
) -> tuple[int, ...]:
    if flow and flow[-1] == result_particle:
        return flow
    return (*flow, result_particle)


def _expr_fermion_propagator_weyl(
    fermion: tuple[Any, ...],
    momentum: tuple[Any, ...],
    chirality: int,
) -> tuple[Any, ...]:
    energy, px, py, pz = momentum
    denominator = _minkowski_square_expression(momentum)
    imaginary = _imaginary_unit_for((*fermion, *momentum))
    prefactor = imaginary / denominator
    tmp1 = energy + pz
    tmp2 = energy - pz
    tmp3 = px + imaginary * py
    tmp4 = px - imaginary * py
    f1, f2 = fermion
    if chirality == 1:
        return (
            (tmp1 * f1 + tmp3 * f2) * prefactor,
            (tmp2 * f2 + tmp4 * f1) * prefactor,
        )
    if chirality == -1:
        return (
            (tmp2 * f1 - tmp3 * f2) * prefactor,
            (tmp1 * f2 - tmp4 * f1) * prefactor,
        )
    raise ValueError("Weyl fermion propagator expression needs nonzero chirality")


def _expr_antifermion_propagator_weyl(
    antifermion: tuple[Any, ...],
    momentum: tuple[Any, ...],
    chirality: int,
) -> tuple[Any, ...]:
    energy, px, py, pz = momentum
    denominator = _minkowski_square_expression(momentum)
    imaginary = _imaginary_unit_for((*antifermion, *momentum))
    prefactor = imaginary / denominator
    tmp1 = -(energy + pz)
    tmp2 = -(energy - pz)
    tmp3 = -(px + imaginary * py)
    tmp4 = -(px - imaginary * py)
    a1, a2 = antifermion
    if chirality == 1:
        return (
            (tmp2 * a1 - tmp4 * a2) * prefactor,
            (tmp1 * a2 - tmp3 * a1) * prefactor,
        )
    if chirality == -1:
        return (
            (tmp1 * a1 + tmp4 * a2) * prefactor,
            (tmp2 * a2 + tmp3 * a1) * prefactor,
        )
    raise ValueError("Weyl antifermion propagator expression needs nonzero chirality")


def _expr_fermion_propagator_dirac(
    fermion: tuple[Any, ...],
    momentum: tuple[Any, ...],
    mass: float,
    width: float,
) -> tuple[Any, ...]:
    if len(fermion) != 4 or len(momentum) != 4:
        raise ValueError("Dirac fermion propagator expects four components")
    energy, px, py, pz = momentum
    imaginary = _imaginary_unit_for((*fermion, *momentum, mass, width))
    denominator = (
        _minkowski_square_expression(momentum) - mass * mass + imaginary * mass * width
    )
    prefactor = imaginary / denominator
    tmp1 = energy + pz
    tmp2 = energy - pz
    tmp3 = px + imaginary * py
    tmp4 = px - imaginary * py
    f1, f2, f3, f4 = fermion
    return (
        (tmp1 * f3 + tmp3 * f4 + mass * f1) * prefactor,
        (tmp2 * f4 + tmp4 * f3 + mass * f2) * prefactor,
        (tmp2 * f1 - tmp3 * f2 + mass * f3) * prefactor,
        (tmp1 * f2 - tmp4 * f1 + mass * f4) * prefactor,
    )


def _expr_antifermion_propagator_dirac(
    antifermion: tuple[Any, ...],
    momentum: tuple[Any, ...],
    mass: float,
    width: float,
) -> tuple[Any, ...]:
    if len(antifermion) != 4 or len(momentum) != 4:
        raise ValueError("Dirac antifermion propagator expects four components")
    energy, px, py, pz = momentum
    imaginary = _imaginary_unit_for((*antifermion, *momentum, mass, width))
    denominator = (
        _minkowski_square_expression(momentum) - mass * mass + imaginary * mass * width
    )
    prefactor = imaginary / denominator
    tmp1 = -(energy + pz)
    tmp2 = -(energy - pz)
    tmp3 = -(px + imaginary * py)
    tmp4 = -(px - imaginary * py)
    a1, a2, a3, a4 = antifermion
    return (
        (tmp2 * a3 - tmp4 * a4 + mass * a1) * prefactor,
        (tmp1 * a4 - tmp3 * a3 + mass * a2) * prefactor,
        (tmp1 * a1 + tmp4 * a2 + mass * a3) * prefactor,
        (tmp2 * a2 + tmp3 * a1 + mass * a4) * prefactor,
    )


def _expr_minkowski_dot(
    left: tuple[Any, ...],
    right: tuple[Any, ...],
) -> Any:
    return (
        left[0] * right[0]
        - left[1] * right[1]
        - left[2] * right[2]
        - left[3] * right[3]
    )
