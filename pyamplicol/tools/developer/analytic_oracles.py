#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Small independent closed-form oracles for developer physics fixtures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    localcontext,
)
from typing import cast

Momentum = tuple[Decimal, Decimal, Decimal, Decimal]

_DEFAULT_PRECISION = 100
_DEFAULT_CERTIFIED_ACCURACY = 12
_MINIMUM_PRECISION = 32
_ARITHMETIC_GUARD_DIGITS = 8
_VALIDATION_GUARD_DIGITS = 3

_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_SIXTEEN = Decimal("16")


class AnalyticOracleError(ValueError):
    """The requested point is outside an analytic oracle's domain."""


@dataclass(frozen=True, slots=True)
class AnalyticOracleMetadata:
    """Precision and accuracy claims attached to one oracle observation.

    Arithmetic precision is the working ``Decimal`` context.  Certified accuracy
    is the relative decimal accuracy requested by the caller and protected by
    working guard digits.  It assumes the supplied decimal components themselves
    carry that accuracy; the oracle checks only their on-shell and conservation
    residuals.  Kinematic oracles report the stricter residual tolerance used for
    those checks.
    """

    arithmetic_precision_decimal_digits: int
    certified_accuracy_decimal_digits: int
    kinematic_validation_relative_tolerance: Decimal | None


@dataclass(frozen=True, slots=True)
class AnalyticObservation:
    total: Decimal
    helicity_ids: tuple[str, ...]
    resolved: tuple[Decimal, ...]
    metadata: AnalyticOracleMetadata

    @property
    def resolved_by_helicity(self) -> tuple[tuple[str, Decimal], ...]:
        return tuple(zip(self.helicity_ids, self.resolved, strict=True))


def scalar_contact_2to2(
    coupling: Decimal = _ONE,
    *,
    precision: int = _DEFAULT_PRECISION,
    certified_accuracy_digits: int = _DEFAULT_CERTIFIED_ACCURACY,
) -> AnalyticObservation:
    """Return the minimal-order massless ``scalar_0^4`` contact result."""

    _validate_numeric_policy(precision, certified_accuracy_digits)
    coupling = _finite_decimal(coupling, "scalar contact coupling")
    try:
        with localcontext(_decimal_context(precision)):
            total = +(coupling * coupling / _TWO)
    except DecimalException as exc:
        raise AnalyticOracleError("scalar contact Decimal arithmetic failed") from exc
    return AnalyticObservation(
        total=total,
        helicity_ids=("h:+0,+0,+0,+0",),
        resolved=(total,),
        metadata=_metadata(precision, certified_accuracy_digits, None),
    )


def scalar_gravity_2to2(
    momenta: Sequence[Sequence[Decimal]],
    coupling: Decimal = _ONE,
    *,
    precision: int = _DEFAULT_PRECISION,
    certified_accuracy_digits: int = _DEFAULT_CERTIFIED_ACCURACY,
) -> AnalyticObservation:
    """Return ``scalar_0 scalar_0 -> graviton graviton`` at tree level.

    The physical momentum order is ``p1, p2, p3, p4`` with the first two
    particles incoming.  For massless external states the shipped model gives
    ``kappa^4 t^2 u^2 / (16 s^2)`` after its identical-particle factor.  In the
    public helicity order ``(--), (-+), (+-), (++)``, only the two opposite
    graviton helicities contribute and each carries half of the total.

    ``u`` is constructed as ``-s-t`` after independently validating all four
    momenta.  This enforces the exact massless invariant relation instead of
    feeding a tolerated input residual into the closed-form result.
    """

    _validate_numeric_policy(precision, certified_accuracy_digits)
    coupling = _finite_decimal(coupling, "scalar-gravity coupling")
    try:
        with localcontext(_decimal_context(precision)):
            point = _point(momenta, expected_count=4, process="2->2")
            relative_tolerance = _validate_kinematics(
                point,
                masses=(_ZERO, _ZERO, _ZERO, _ZERO),
                initial_state_count=2,
                certified_accuracy_digits=certified_accuracy_digits,
            )
            s = +_square(_add(point[0], point[1]))
            if s <= 0:
                raise AnalyticOracleError(
                    "scalar-gravity oracle requires positive nonzero s"
                )
            t = +_square(_subtract(point[0], point[2]))
            direct_u = +_square(_subtract(point[0], point[3]))
            _validate_invariant_identity(s, t, direct_u, relative_tolerance)
            u = +(-s - t)
            total = +(coupling**4 * t**2 * u**2 / (_SIXTEEN * s**2))
            half = +(total / _TWO)
    except DecimalException as exc:
        raise AnalyticOracleError("scalar-gravity Decimal arithmetic failed") from exc
    return AnalyticObservation(
        total=total,
        helicity_ids=(
            "h:+0,+0,-2,-2",
            "h:+0,+0,-2,+2",
            "h:+0,+0,+2,-2",
            "h:+0,+0,+2,+2",
        ),
        resolved=(_ZERO, half, half, _ZERO),
        metadata=_metadata(precision, certified_accuracy_digits, relative_tolerance),
    )


def chiral_current_2to1(
    momenta: Sequence[Sequence[Decimal]],
    *,
    vector_mass: Decimal,
    left_coupling: Decimal,
    right_coupling: Decimal,
    precision: int = _DEFAULT_PRECISION,
    certified_accuracy_digits: int = _DEFAULT_CERTIFIED_ACCURACY,
) -> AnalyticObservation:
    """Return a massless chiral-current ``fermion antifermion -> vector`` probe.

    For a vertex ``gamma^mu (g_L P_L + g_R P_R)``, the two nonzero resolved
    members are ``2 s g_L^2`` and ``2 s g_R^2`` in the public helicity order
    ``(-,+,-), (+,-,+)``.  Choosing unequal nonzero couplings makes leg or
    helicity permutations observable, unlike the parity-even scalar-gravity
    rows above.
    """

    _validate_numeric_policy(precision, certified_accuracy_digits)
    vector_mass = _finite_decimal(vector_mass, "vector mass")
    left_coupling = _finite_decimal(left_coupling, "left chiral coupling")
    right_coupling = _finite_decimal(right_coupling, "right chiral coupling")
    if vector_mass <= 0:
        raise AnalyticOracleError("vector mass must be positive")
    try:
        with localcontext(_decimal_context(precision)):
            point = _point(momenta, expected_count=3, process="2->1")
            relative_tolerance = _validate_kinematics(
                point,
                masses=(_ZERO, _ZERO, vector_mass),
                initial_state_count=2,
                certified_accuracy_digits=certified_accuracy_digits,
            )
            s = +_square(_add(point[0], point[1]))
            if s <= 0:
                raise AnalyticOracleError(
                    "chiral-current oracle requires positive nonzero s"
                )
            left = +(_TWO * s * left_coupling**2)
            right = +(_TWO * s * right_coupling**2)
            total = +(left + right)
    except DecimalException as exc:
        raise AnalyticOracleError("chiral-current Decimal arithmetic failed") from exc
    return AnalyticObservation(
        total=total,
        helicity_ids=("h:-1,+1,-1", "h:+1,-1,+1"),
        resolved=(left, right),
        metadata=_metadata(precision, certified_accuracy_digits, relative_tolerance),
    )


def _decimal_context(precision: int) -> Context:
    return Context(prec=precision, rounding=ROUND_HALF_EVEN)


def _validate_numeric_policy(precision: int, certified_accuracy_digits: int) -> None:
    if isinstance(precision, bool) or not isinstance(precision, int):
        raise AnalyticOracleError("analytic precision must be an integer")
    if precision < _MINIMUM_PRECISION:
        raise AnalyticOracleError(
            f"analytic precision must be at least {_MINIMUM_PRECISION} digits"
        )
    if isinstance(certified_accuracy_digits, bool) or not isinstance(
        certified_accuracy_digits, int
    ):
        raise AnalyticOracleError("certified accuracy must be an integer")
    if certified_accuracy_digits <= 0:
        raise AnalyticOracleError("certified accuracy must be positive")
    maximum_accuracy = precision - _ARITHMETIC_GUARD_DIGITS
    if certified_accuracy_digits > maximum_accuracy:
        raise AnalyticOracleError(
            "certified accuracy must leave at least "
            f"{_ARITHMETIC_GUARD_DIGITS} arithmetic guard digits"
        )


def _finite_decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise AnalyticOracleError(f"{label} must be a Decimal")
    if not value.is_finite():
        raise AnalyticOracleError(f"{label} must be finite")
    return value


def _metadata(
    precision: int,
    certified_accuracy_digits: int,
    relative_tolerance: Decimal | None,
) -> AnalyticOracleMetadata:
    return AnalyticOracleMetadata(
        arithmetic_precision_decimal_digits=precision,
        certified_accuracy_decimal_digits=certified_accuracy_digits,
        kinematic_validation_relative_tolerance=relative_tolerance,
    )


def _point(
    momenta: Sequence[Sequence[Decimal]],
    *,
    expected_count: int,
    process: str,
) -> tuple[Momentum, ...]:
    count_name = {3: "three", 4: "four"}.get(expected_count, str(expected_count))
    if (
        isinstance(momenta, str | bytes)
        or not isinstance(momenta, Sequence)
        or len(momenta) != expected_count
    ):
        raise AnalyticOracleError(
            f"{process} oracle requires {count_name} external momenta"
        )
    point = []
    for index, vector in enumerate(momenta):
        if (
            isinstance(vector, str | bytes)
            or not isinstance(vector, Sequence)
            or len(vector) != 4
        ):
            raise AnalyticOracleError(
                f"momentum {index} must contain four Decimal components"
            )
        components = tuple(
            _finite_decimal(value, f"momentum {index} component {component_index}")
            for component_index, value in enumerate(vector)
        )
        point.append(cast(Momentum, components))
    return tuple(point)


def _add(left: Momentum, right: Momentum) -> Momentum:
    return cast(Momentum, tuple(a + b for a, b in zip(left, right, strict=True)))


def _subtract(left: Momentum, right: Momentum) -> Momentum:
    return cast(Momentum, tuple(a - b for a, b in zip(left, right, strict=True)))


def _square(momentum: Momentum) -> Decimal:
    return momentum[0] ** 2 - sum((component**2 for component in momentum[1:]), _ZERO)


def _validation_relative_tolerance(
    certified_accuracy_digits: int,
) -> Decimal:
    return _ONE.scaleb(-(certified_accuracy_digits + _VALIDATION_GUARD_DIGITS))


def _validate_kinematics(
    point: tuple[Momentum, ...],
    *,
    masses: tuple[Decimal, ...],
    initial_state_count: int,
    certified_accuracy_digits: int,
) -> Decimal:
    relative_tolerance = _validation_relative_tolerance(certified_accuracy_digits)
    scale = max(
        max(
            (abs(component) for momentum in point for component in momentum),
            default=_ONE,
        ),
        _ONE,
    )
    momentum_tolerance = +(scale * relative_tolerance)
    shell_tolerance = +(max(scale * scale, _ONE) * relative_tolerance)
    for index, (momentum, mass) in enumerate(zip(point, masses, strict=True)):
        if momentum[0] <= 0:
            raise AnalyticOracleError(f"momentum {index} has non-positive energy")
        residual = +(_square(momentum) - mass * mass)
        if abs(residual) > shell_tolerance:
            if mass == 0:
                raise AnalyticOracleError(
                    f"momentum {index} is not massless: residual {residual} "
                    f"exceeds {shell_tolerance}"
                )
            raise AnalyticOracleError(
                f"momentum {index} is off shell: residual {residual} "
                f"exceeds {shell_tolerance}"
            )
    imbalance = tuple(
        sum(
            (momentum[component] for momentum in point[:initial_state_count]),
            _ZERO,
        )
        - sum(
            (momentum[component] for momentum in point[initial_state_count:]),
            _ZERO,
        )
        for component in range(4)
    )
    if any(abs(component) > momentum_tolerance for component in imbalance):
        raise AnalyticOracleError(
            "momenta do not conserve four-momentum: "
            f"residual {imbalance} exceeds {momentum_tolerance}"
        )
    return relative_tolerance


def _validate_invariant_identity(
    s: Decimal,
    t: Decimal,
    direct_u: Decimal,
    relative_tolerance: Decimal,
) -> None:
    scale = max(abs(s), abs(t), abs(direct_u), _ONE)
    tolerance = +(scale * relative_tolerance)
    residual = +(s + t + direct_u)
    if abs(residual) > tolerance:
        raise AnalyticOracleError(
            "massless invariants violate s+t+u=0: "
            f"residual {residual} exceeds {tolerance}"
        )


__all__ = [
    "AnalyticObservation",
    "AnalyticOracleError",
    "AnalyticOracleMetadata",
    "chiral_current_2to1",
    "scalar_contact_2to2",
    "scalar_gravity_2to2",
]
