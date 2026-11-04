# SPDX-License-Identifier: 0BSD
"""Exact arithmetic and kinematic validation for reference fixtures."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import Decimal
from fractions import Fraction

from .model import Process, ReferenceFixtureError, ReferencePoint, Tolerances

_LOG10_2_NUMERATOR = 301_029_995_663_981_195
_LOG10_2_DENOMINATOR = 1_000_000_000_000_000_000
_MAX_STRESS_METRIC = Decimal("0.000001")
_BASELINE_CERTIFIED_DIGITS = 12


def _as_fraction(value: Decimal | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _exact_sum(values: Iterable[Decimal]) -> Fraction:
    return sum((_as_fraction(value) for value in values), Fraction())


def _validate_precision_metadata(
    *,
    arithmetic_precision_bits: int,
    round_trip_decimal_digits: int,
    certified_decimal_digits: int,
    where: str,
) -> None:
    if certified_decimal_digits > round_trip_decimal_digits:
        raise ReferenceFixtureError(
            f"{where} certified digits exceed its round-trip digits"
        )
    supported_certified_digits = (
        arithmetic_precision_bits * _LOG10_2_NUMERATOR
    ) // _LOG10_2_DENOMINATOR
    if certified_decimal_digits > supported_certified_digits:
        raise ReferenceFixtureError(
            f"{where} certified digits exceed its arithmetic-bit capacity"
        )
    maximum_round_trip_digits = (
        arithmetic_precision_bits * _LOG10_2_NUMERATOR + _LOG10_2_DENOMINATOR - 1
    ) // _LOG10_2_DENOMINATOR + 1
    if round_trip_decimal_digits > maximum_round_trip_digits:
        raise ReferenceFixtureError(
            f"{where} round-trip digits exceed its arithmetic-bit capacity"
        )


def _minkowski_square(momentum: Sequence[Decimal | Fraction]) -> Fraction:
    values = tuple(_as_fraction(component) for component in momentum)
    return values[0] * values[0] - sum(
        (component * component for component in values[1:]), Fraction()
    )


def _stress_metric_value(
    point: ReferencePoint,
    process: Process,
) -> Fraction:
    final_momenta = point.momenta[process.initial_state_count :]
    if point.stress_metric is None:
        raise ReferenceFixtureError(f"stress point {point.id} lacks a stress metric")
    sqrt_s = _as_fraction(point.sqrt_s)
    if point.stress_metric.kind == "minimum-final-energy-fraction":
        return min(_as_fraction(momentum[0]) / sqrt_s for momentum in final_momenta)
    if point.stress_metric.kind == "relative-excess-energy":
        threshold = sum(
            (
                _as_fraction(mass)
                for mass in point.masses[process.initial_state_count :]
            ),
            Fraction(),
        )
        if threshold <= 0:
            raise ReferenceFixtureError(
                f"threshold stress point {point.id} has no massive final state"
            )
        return (sqrt_s - threshold) / threshold
    sqrt_s_squared = sqrt_s * sqrt_s
    return min(
        (_as_fraction(momentum[1]) ** 2 + _as_fraction(momentum[2]) ** 2)
        / sqrt_s_squared
        for momentum in final_momenta
    )


def _required_stress_certified_digits(point: ReferencePoint) -> int:
    metric = point.stress_metric
    if metric is None:
        return 0
    severity_digits = max(0, -metric.value.adjusted())
    return max(6, severity_digits + 2)


def _validate_point_kinematics(point: ReferencePoint, process: Process) -> None:
    width = len(process.external_pdgs)
    if len(point.momenta) != width or len(point.masses) != width:
        raise ReferenceFixtureError(
            f"point {point.id} momentum and mass metadata must match process "
            f"{process.id}"
        )
    if point.masses != process.external_masses:
        raise ReferenceFixtureError(
            f"point {point.id} masses differ from process {process.id} "
            "model-derived masses"
        )
    _validate_precision_metadata(
        arithmetic_precision_bits=point.arithmetic_precision_bits,
        round_trip_decimal_digits=point.round_trip_decimal_digits,
        certified_decimal_digits=point.certified_decimal_digits,
        where=f"point {point.id}",
    )
    if point.certified_decimal_digits < _BASELINE_CERTIFIED_DIGITS:
        raise ReferenceFixtureError(
            f"point {point.id} must certify at least "
            f"{_BASELINE_CERTIFIED_DIGITS} decimal digits"
        )
    if point.sqrt_s <= 0:
        raise ReferenceFixtureError(f"point {point.id} sqrt_s must be positive")
    if any(mass < 0 for mass in point.masses):
        raise ReferenceFixtureError(f"point {point.id} masses must be nonnegative")
    if any(momentum[0] <= 0 for momentum in point.momenta):
        raise ReferenceFixtureError(
            f"point {point.id} external energies must be positive"
        )

    scale = max(
        Fraction(1),
        _as_fraction(point.sqrt_s),
        *(abs(_as_fraction(component)) for row in point.momenta for component in row),
    )
    relative_tolerance = Fraction(1, 10**point.certified_decimal_digits)
    momentum_tolerance = scale * relative_tolerance
    shell_tolerance = scale * scale * relative_tolerance
    incoming = tuple(
        sum(
            (
                _as_fraction(momentum[component])
                for momentum in point.momenta[: process.initial_state_count]
            ),
            Fraction(),
        )
        for component in range(4)
    )
    outgoing = tuple(
        sum(
            (
                _as_fraction(momentum[component])
                for momentum in point.momenta[process.initial_state_count :]
            ),
            Fraction(),
        )
        for component in range(4)
    )
    if any(
        abs(incoming_component - outgoing_component) > momentum_tolerance
        for incoming_component, outgoing_component in zip(
            incoming, outgoing, strict=True
        )
    ):
        raise ReferenceFixtureError(
            f"point {point.id} violates four-momentum conservation"
        )
    if abs(_minkowski_square(incoming) - _as_fraction(point.sqrt_s) ** 2) > (
        shell_tolerance
    ):
        raise ReferenceFixtureError(
            f"point {point.id} incoming invariant does not match sqrt_s"
        )
    for index, (momentum, mass) in enumerate(
        zip(point.momenta, point.masses, strict=True)
    ):
        residual = _minkowski_square(momentum) - _as_fraction(mass) ** 2
        if abs(residual) > shell_tolerance:
            raise ReferenceFixtureError(
                f"point {point.id} leg {index} is off shell for its declared mass"
            )

    if point.point_class == "stress":
        metric = point.stress_metric
        if metric is None:
            raise ReferenceFixtureError(
                f"stress point {point.id} lacks a quantified stress metric"
            )
        expected_metric = _stress_metric_value(point, process)
        metric_value = _as_fraction(metric.value)
        metric_tolerance = max(
            Fraction(1, 10**point.certified_decimal_digits),
            abs(expected_metric) / 10**point.certified_decimal_digits,
        )
        if abs(expected_metric - metric_value) > metric_tolerance:
            raise ReferenceFixtureError(
                f"stress point {point.id} metric does not match its momenta"
            )
        if metric.value > _MAX_STRESS_METRIC:
            raise ReferenceFixtureError(
                f"stress point {point.id} metric is not sufficiently singular"
            )
        required_digits = _required_stress_certified_digits(point)
        if point.certified_decimal_digits < required_digits:
            raise ReferenceFixtureError(
                f"stress point {point.id} certification is not commensurate with "
                "its stress metric"
            )
    elif point.stress_metric is not None:
        raise ReferenceFixtureError(
            f"non-stress point {point.id} must not declare a stress metric"
        )


def _within_tolerance(
    expected: Decimal | Fraction,
    observed: Decimal | Fraction,
    tolerances: Tolerances,
) -> bool:
    exact_expected = _as_fraction(expected)
    exact_observed = _as_fraction(observed)
    scale = max(abs(exact_expected), abs(exact_observed))
    allowed = max(
        _as_fraction(tolerances.absolute),
        _as_fraction(tolerances.relative) * scale,
    )
    return abs(exact_expected - exact_observed) <= allowed
