# SPDX-License-Identifier: 0BSD
"""Deterministic canonical phase-space points and external leg identity."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, localcontext
from typing import cast

from .common import (
    CaptureError,
    CapturePoint,
    Momentum,
    ProcessCaptureSpec,
    StressMetric,
    canonical_decimal,
    decimal_digits_to_bits,
)

_PUBLIC_ID_PART_RE = re.compile(r"[^A-Za-z0-9._~-]+")
_GENERIC_SQRT_S = Decimal("1000")
_BINARY64_BITS = 53
_BINARY64_ROUND_TRIP_DIGITS = 17
_BINARY64_CERTIFIED_DIGITS = 12
_DECIMAL_WORKING_DIGITS = 110
_DECIMAL_ROUND_TRIP_DIGITS = _DECIMAL_WORKING_DIGITS
_DECIMAL_CERTIFIED_DIGITS = 80
# Keep this point genuinely soft and near-collinear while retaining the ten
# decimal digits certified by the independent binary64 Fortran oracle.
_THREE_BODY_STRESS_ENERGY_FRACTION = Decimal("1e-3")
_THREE_BODY_STRESS_SINE = Decimal("0.2")
_THRESHOLD_STRESS_RELATIVE_EXCESS = Decimal("1e-6")
def _f64_decimal(value: float) -> Decimal:
    return Decimal(canonical_decimal(value))


def _lambda(x: Decimal, y: Decimal, z: Decimal) -> Decimal:
    return x * x + y * y + z * z - 2 * (x * y + x * z + y * z)


def _incoming(sqrt_s: Decimal) -> tuple[Momentum, Momentum]:
    half = sqrt_s / 2
    zero = Decimal(0)
    return (
        (half, zero, zero, half),
        (half, zero, zero, -half),
    )


def _transverse_stress_metric(
    momenta: Sequence[Momentum],
    sqrt_s: Decimal,
    *,
    initial_state_count: int = 2,
) -> StressMetric:
    with localcontext() as context:
        context.prec = _DECIMAL_WORKING_DIGITS
        value = +min(
            (momentum[1] * momentum[1] + momentum[2] * momentum[2]) / (sqrt_s * sqrt_s)
            for momentum in momenta[initial_state_count:]
        )
    return StressMetric(
        "minimum-final-transverse-momentum-squared-fraction",
        value,
    )


def _two_body_generic(
    process_id: str,
    expression: str,
    seed: int,
    index: int,
    masses: tuple[Decimal, Decimal],
) -> CapturePoint:
    sqrt_s = float(_GENERIC_SQRT_S)
    mass_3, mass_4 = (float(mass) for mass in masses)
    s = sqrt_s * sqrt_s
    radicand = (s - (mass_3 + mass_4) ** 2) * (s - (mass_3 - mass_4) ** 2)
    if radicand <= 0:
        raise CaptureError(f"two-body point is below threshold for {expression}")
    momentum = math.sqrt(radicand) / (2.0 * sqrt_s)
    energy_3 = (s + mass_3 * mass_3 - mass_4 * mass_4) / (2.0 * sqrt_s)
    energy_4 = sqrt_s - energy_3
    generator = random.Random(seed)
    cosine = generator.uniform(-0.82, 0.82)
    sine = math.sqrt(1.0 - cosine * cosine)
    phi = generator.uniform(0.0, 2.0 * math.pi)
    px = momentum * sine * math.cos(phi)
    py = momentum * sine * math.sin(phi)
    pz = momentum * cosine
    raw = (
        (sqrt_s / 2.0, 0.0, 0.0, sqrt_s / 2.0),
        (sqrt_s / 2.0, 0.0, 0.0, -sqrt_s / 2.0),
        (energy_3, px, py, pz),
        (energy_4, -px, -py, -pz),
    )
    point = CapturePoint(
        id=f"point:{process_id}:generic-{index}",
        process_id=process_id,
        point_class="generic",
        algorithm_name="seeded-two-body-binary64",
        algorithm_version="1",
        rng="python-random-mt19937",
        seed=seed,
        sqrt_s=_f64_decimal(sqrt_s),
        momenta=tuple(
            cast(Momentum, tuple(_f64_decimal(component) for component in row))
            for row in raw
        ),
        masses=(Decimal(0), Decimal(0), *masses),
        arithmetic_precision_bits=_BINARY64_BITS,
        round_trip_decimal_digits=_BINARY64_ROUND_TRIP_DIGITS,
        certified_decimal_digits=_BINARY64_CERTIFIED_DIGITS,
        stress_metric=None,
    )
    validate_point_kinematics(point, point.masses)
    return point


def _two_body_stress(
    process_id: str,
    expression: str,
    masses: tuple[Decimal, Decimal],
) -> CapturePoint:
    with localcontext() as context:
        context.prec = _DECIMAL_WORKING_DIGITS
        sqrt_s = +_GENERIC_SQRT_S
        mass_3, mass_4 = masses
        s = sqrt_s * sqrt_s
        radicand = _lambda(s, mass_3 * mass_3, mass_4 * mass_4)
        if radicand <= 0:
            raise CaptureError(
                f"two-body stress point is below threshold: {expression}"
            )
        momentum = +(radicand.sqrt() / (2 * sqrt_s))
        energy_3 = +(s + mass_3 * mass_3 - mass_4 * mass_4) / (2 * sqrt_s)
        energy_4 = +(sqrt_s - energy_3)
        sine = Decimal("0.002")
        cosine = +(Decimal(1) - sine * sine).sqrt()
        px = +(momentum * sine)
        pz = +(momentum * cosine)
        zero = Decimal(0)
        momenta = (
            *_incoming(sqrt_s),
            (energy_3, px, zero, pz),
            (energy_4, -px, zero, -pz),
        )
    point = CapturePoint(
        id=f"point:{process_id}:stress-near-collinear",
        process_id=process_id,
        point_class="stress",
        algorithm_name="decimal-near-collinear-two-body",
        algorithm_version="1",
        rng=None,
        seed=None,
        sqrt_s=sqrt_s,
        momenta=momenta,
        masses=(Decimal(0), Decimal(0), *masses),
        arithmetic_precision_bits=decimal_digits_to_bits(_DECIMAL_WORKING_DIGITS),
        round_trip_decimal_digits=_DECIMAL_ROUND_TRIP_DIGITS,
        certified_decimal_digits=_DECIMAL_CERTIFIED_DIGITS,
        stress_metric=_transverse_stress_metric(momenta, sqrt_s),
    )
    validate_point_kinematics(point, point.masses)
    return point


def _two_body_threshold_stress(
    process_id: str,
    expression: str,
    masses: tuple[Decimal, Decimal],
) -> CapturePoint:
    with localcontext() as context:
        context.prec = _DECIMAL_WORKING_DIGITS
        mass_3, mass_4 = masses
        threshold = +(mass_3 + mass_4)
        if mass_3 <= 0 or mass_4 <= 0:
            raise CaptureError(
                f"threshold stress point requires two massive final legs: "
                f"{expression}"
            )
        sqrt_s = +(threshold * (Decimal(1) + _THRESHOLD_STRESS_RELATIVE_EXCESS))
        s = +(sqrt_s * sqrt_s)
        radicand = _lambda(s, mass_3 * mass_3, mass_4 * mass_4)
        if radicand <= 0:
            raise CaptureError(
                f"threshold stress point is below threshold: {expression}"
            )
        momentum = +(radicand.sqrt() / (2 * sqrt_s))
        energy_3 = +(s + mass_3 * mass_3 - mass_4 * mass_4) / (2 * sqrt_s)
        energy_4 = +(sqrt_s - energy_3)
        sine = Decimal("0.6")
        cosine = Decimal("0.8")
        zero = Decimal(0)
        momenta = (
            *_incoming(sqrt_s),
            (energy_3, +(momentum * sine), zero, +(momentum * cosine)),
            (energy_4, -(momentum * sine), zero, -(momentum * cosine)),
        )
    point = CapturePoint(
        id=f"point:{process_id}:stress-near-threshold",
        process_id=process_id,
        point_class="stress",
        algorithm_name="decimal-near-threshold-two-body",
        algorithm_version="1",
        rng=None,
        seed=None,
        sqrt_s=sqrt_s,
        momenta=momenta,
        masses=(Decimal(0), Decimal(0), *masses),
        arithmetic_precision_bits=decimal_digits_to_bits(_DECIMAL_WORKING_DIGITS),
        round_trip_decimal_digits=_DECIMAL_ROUND_TRIP_DIGITS,
        certified_decimal_digits=_DECIMAL_CERTIFIED_DIGITS,
        stress_metric=StressMetric(
            "relative-excess-energy",
            _THRESHOLD_STRESS_RELATIVE_EXCESS,
        ),
    )
    validate_point_kinematics(point, point.masses)
    return point


def _boost_z_f64(
    momentum: tuple[float, float, float, float], beta: float
) -> tuple[float, float, float, float]:
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    energy, px, py, pz = momentum
    return (
        gamma * (energy + beta * pz),
        px,
        py,
        gamma * (pz + beta * energy),
    )


def _rotate_f64(
    momentum: tuple[float, float, float, float],
    *,
    cosine: float,
    sine: float,
    phi: float,
) -> tuple[float, float, float, float]:
    energy, px, py, pz = momentum
    x_tilted = cosine * px + sine * pz
    z_tilted = -sine * px + cosine * pz
    cosine_phi = math.cos(phi)
    sine_phi = math.sin(phi)
    return (
        energy,
        cosine_phi * x_tilted - sine_phi * py,
        sine_phi * x_tilted + cosine_phi * py,
        z_tilted,
    )


def _three_body_generic(
    process_id: str,
    seed: int,
    index: int,
    mass: Decimal,
) -> CapturePoint:
    sqrt_s = float(_GENERIC_SQRT_S)
    final_mass = float(mass)
    generator = random.Random(seed)
    soft_energy = sqrt_s * (0.08 + 0.12 * generator.random())
    q_energy = sqrt_s - soft_energy
    q_mass = math.sqrt(sqrt_s * sqrt_s - 2.0 * sqrt_s * soft_energy)
    hard_energy = (q_mass * q_mass - final_mass * final_mass) / (2.0 * q_mass)
    massive_energy = (q_mass * q_mass + final_mass * final_mass) / (2.0 * q_mass)
    cosine = generator.uniform(-0.72, 0.72)
    sine = math.sqrt(1.0 - cosine * cosine)
    phi = generator.uniform(0.0, 2.0 * math.pi)
    px = hard_energy * sine * math.cos(phi)
    py = hard_energy * sine * math.sin(phi)
    pz = hard_energy * cosine
    beta = -soft_energy / q_energy
    hard = _boost_z_f64((hard_energy, px, py, pz), beta)
    massive = _boost_z_f64((massive_energy, -px, -py, -pz), beta)
    rotation_cosine = generator.uniform(-0.75, 0.75)
    rotation_sine = math.sqrt(1.0 - rotation_cosine * rotation_cosine)
    rotation_phi = generator.uniform(0.0, 2.0 * math.pi)
    final_state = tuple(
        _rotate_f64(
            momentum,
            cosine=rotation_cosine,
            sine=rotation_sine,
            phi=rotation_phi,
        )
        for momentum in (
            massive,
            hard,
            (soft_energy, 0.0, 0.0, soft_energy),
        )
    )
    raw = (
        (sqrt_s / 2.0, 0.0, 0.0, sqrt_s / 2.0),
        (sqrt_s / 2.0, 0.0, 0.0, -sqrt_s / 2.0),
        *final_state,
    )
    point = CapturePoint(
        id=f"point:{process_id}:generic-{index}",
        process_id=process_id,
        point_class="generic",
        algorithm_name="seeded-sequential-three-body-binary64",
        algorithm_version="2",
        rng="python-random-mt19937",
        seed=seed,
        sqrt_s=_f64_decimal(sqrt_s),
        momenta=tuple(
            cast(Momentum, tuple(_f64_decimal(component) for component in row))
            for row in raw
        ),
        masses=(Decimal(0), Decimal(0), mass, Decimal(0), Decimal(0)),
        arithmetic_precision_bits=_BINARY64_BITS,
        round_trip_decimal_digits=_BINARY64_ROUND_TRIP_DIGITS,
        certified_decimal_digits=_BINARY64_CERTIFIED_DIGITS,
        stress_metric=None,
    )
    validate_point_kinematics(point, point.masses)
    return point


def _boost_decimal(
    momentum: Momentum,
    beta: tuple[Decimal, Decimal, Decimal],
) -> Momentum:
    beta_square = sum((component * component for component in beta), Decimal(0))
    if beta_square == 0:
        return momentum
    if beta_square >= 1:
        raise CaptureError("decimal boost velocity is not subluminal")
    gamma = Decimal(1) / (Decimal(1) - beta_square).sqrt()
    energy, px, py, pz = momentum
    spatial = (px, py, pz)
    beta_dot_momentum = sum(
        (left * right for left, right in zip(beta, spatial, strict=True)),
        Decimal(0),
    )
    spatial_factor = (
        gamma - Decimal(1)
    ) * beta_dot_momentum / beta_square + gamma * energy
    boosted = tuple(
        +(component + spatial_factor * velocity)
        for component, velocity in zip(spatial, beta, strict=True)
    )
    return cast(Momentum, (+(gamma * (energy + beta_dot_momentum)), *boosted))


def _three_body_stress(process_id: str, mass: Decimal) -> CapturePoint:
    with localcontext() as context:
        context.prec = _DECIMAL_WORKING_DIGITS
        sqrt_s = +_GENERIC_SQRT_S
        mass = +mass
        soft_energy = +(sqrt_s * _THREE_BODY_STRESS_ENERGY_FRACTION)
        q_energy = +(sqrt_s - soft_energy)
        q_mass = +(sqrt_s * sqrt_s - 2 * sqrt_s * soft_energy).sqrt()
        hard_energy = +(q_mass * q_mass - mass * mass) / (2 * q_mass)
        massive_energy = +(q_mass * q_mass + mass * mass) / (2 * q_mass)
        sine = _THREE_BODY_STRESS_SINE
        cosine = +(Decimal(1) - sine * sine).sqrt()
        zero = Decimal(0)
        soft_px = +(soft_energy * sine)
        soft_pz = +(soft_energy * cosine)
        beta = (-soft_px / q_energy, zero, -soft_pz / q_energy)
        hard = _boost_decimal((hard_energy, zero, hard_energy, zero), beta)
        massive = _boost_decimal((massive_energy, zero, -hard_energy, zero), beta)
        momenta = (
            *_incoming(sqrt_s),
            massive,
            hard,
            (soft_energy, soft_px, zero, soft_pz),
        )
    point = CapturePoint(
        id=f"point:{process_id}:stress-soft-collinear",
        process_id=process_id,
        point_class="stress",
        algorithm_name="decimal-soft-near-collinear-sequential-three-body",
        algorithm_version="2",
        rng=None,
        seed=None,
        sqrt_s=sqrt_s,
        momenta=momenta,
        masses=(Decimal(0), Decimal(0), mass, Decimal(0), Decimal(0)),
        arithmetic_precision_bits=decimal_digits_to_bits(_DECIMAL_WORKING_DIGITS),
        round_trip_decimal_digits=_DECIMAL_ROUND_TRIP_DIGITS,
        certified_decimal_digits=_DECIMAL_CERTIFIED_DIGITS,
        stress_metric=_transverse_stress_metric(momenta, sqrt_s),
    )
    validate_point_kinematics(point, point.masses)
    return point


def _exact_two_to_one(
    process: ProcessCaptureSpec,
    masses: tuple[Decimal, ...],
) -> CapturePoint:
    if len(masses) != 3 or any(mass != 0 for mass in masses[:2]):
        raise CaptureError(
            f"exact two-to-one point requires two massless incoming legs: "
            f"{process.expression}"
        )
    final_mass = masses[2]
    if final_mass <= 0:
        raise CaptureError(
            f"exact two-to-one point requires a massive final leg: "
            f"{process.expression}"
        )
    half = final_mass / 2
    zero = Decimal(0)
    point = CapturePoint(
        id=f"point:{process.id}:canonical",
        process_id=process.id,
        point_class="canonical",
        algorithm_name="exact-two-to-one",
        algorithm_version="1",
        rng=None,
        seed=None,
        sqrt_s=final_mass,
        momenta=(
            (half, zero, zero, half),
            (half, zero, zero, -half),
            (final_mass, zero, zero, zero),
        ),
        masses=masses,
        arithmetic_precision_bits=_BINARY64_BITS,
        round_trip_decimal_digits=_BINARY64_ROUND_TRIP_DIGITS,
        certified_decimal_digits=15,
        stress_metric=None,
    )
    validate_point_kinematics(point, masses)
    return point


def build_reference_points(
    process: ProcessCaptureSpec,
    masses: tuple[Decimal, ...],
) -> tuple[CapturePoint, ...]:
    """Construct the deterministic compact point ladder for one process."""

    if process.point_policy == "exact-2to1":
        if process.point_seeds:
            raise CaptureError(
                f"exact point policy for {process.expression!r} must not define seeds"
            )
        return (_exact_two_to_one(process, masses),)
    if len(process.point_seeds) != 3:
        raise CaptureError(
            f"point policy for {process.expression!r} requires exactly three seeds"
        )
    if any(mass != 0 for mass in masses[:2]):
        raise CaptureError(
            f"reference point policy currently requires massless incoming legs: "
            f"{process.expression}"
        )
    if process.point_policy == "one-massive-two-massless":
        if len(masses) != 5 or masses[2] <= 0 or any(
            mass != 0 for mass in masses[3:]
        ):
            raise CaptureError(
                "one-massive-two-massless point policy received incompatible "
                f"masses for {process.expression!r}"
            )
        return (
            *(
                _three_body_generic(process.id, seed, index, masses[2])
                for index, seed in enumerate(process.point_seeds, start=1)
            ),
            _three_body_stress(process.id, masses[2]),
        )
    if process.point_policy != "seeded-two-body" or len(masses) != 4:
        raise CaptureError(
            f"no strict point policy is defined for {process.expression!r}"
        )
    final_masses = (masses[2], masses[3])
    return (
        *(
            _two_body_generic(
                process.id,
                process.expression,
                seed,
                index,
                final_masses,
            )
            for index, seed in enumerate(process.point_seeds, start=1)
        ),
        (
            _two_body_threshold_stress(
                process.id,
                process.expression,
                final_masses,
            )
            if all(mass > 0 for mass in final_masses)
            else _two_body_stress(process.id, process.expression, final_masses)
        ),
    )


def _mass_square(momentum: Sequence[Decimal]) -> Decimal:
    return momentum[0] ** 2 - sum(
        (component**2 for component in momentum[1:]), Decimal(0)
    )


def validate_point_kinematics(
    point: CapturePoint,
    masses: Sequence[Decimal],
    *,
    initial_state_count: int = 2,
) -> None:
    """Reject non-finite, off-shell, or non-conserving capture points."""

    if len(masses) != len(point.momenta):
        raise CaptureError(f"{point.id} mass and momentum dimensions differ")
    if initial_state_count <= 0 or initial_state_count >= len(point.momenta):
        raise CaptureError(f"{point.id} has an invalid initial-state count")
    if any(
        not component.is_finite()
        for momentum in point.momenta
        for component in momentum
    ):
        raise CaptureError(f"{point.id} contains non-finite momentum components")
    with localcontext() as context:
        context.prec = max(
            _DECIMAL_WORKING_DIGITS,
            point.certified_decimal_digits + 30,
        )
        scale = max(
            (abs(component) for momentum in point.momenta for component in momentum),
            default=Decimal(1),
        )
        relative = Decimal(1).scaleb(-point.certified_decimal_digits)
        momentum_tolerance = max(scale, Decimal(1)) * relative
        shell_tolerance = max(scale * scale, Decimal(1)) * relative
        imbalance = tuple(
            sum(
                (
                    momentum[component]
                    for momentum in point.momenta[:initial_state_count]
                ),
                Decimal(0),
            )
            - sum(
                (
                    momentum[component]
                    for momentum in point.momenta[initial_state_count:]
                ),
                Decimal(0),
            )
            for component in range(4)
        )
        if any(abs(component) > momentum_tolerance for component in imbalance):
            raise CaptureError(
                f"{point.id} violates four-momentum conservation: {imbalance}"
            )
        for index, (momentum, mass) in enumerate(
            zip(point.momenta, masses, strict=True)
        ):
            residual = _mass_square(momentum) - mass * mass
            if abs(residual) > shell_tolerance:
                raise CaptureError(
                    f"{point.id} momentum {index} is off shell by {residual}"
                )
            if momentum[0] <= 0:
                raise CaptureError(
                    f"{point.id} momentum {index} has non-positive energy"
                )


def _particle_field(particle: object, *names: str) -> object:
    if isinstance(particle, Mapping):
        for name in names:
            if name in particle:
                return particle[name]
    for name in names:
        if hasattr(particle, name):
            return getattr(particle, name)
    raise CaptureError(f"external particle is missing {names[0]}")


def _leg_particle_token(name: str) -> str:
    token = name.replace("~", "bar").replace("+", "plus").replace("-", "minus")
    token = _PUBLIC_ID_PART_RE.sub("-", token).strip("-")
    if not token:
        raise CaptureError(f"cannot derive a stable leg ID from particle {name!r}")
    return token


def stable_external_leg_ids(particles: Sequence[object]) -> tuple[str, ...]:
    """Derive role-aware stable IDs, numbering only identical role/particle legs."""

    identities: list[tuple[str, str]] = []
    for particle in particles:
        raw_state = str(_particle_field(particle, "state", "role"))
        state = {"initial": "incoming", "final": "outgoing"}.get(raw_state, raw_state)
        if state not in {"incoming", "outgoing"}:
            raise CaptureError(f"unsupported external-particle state {raw_state!r}")
        name = str(_particle_field(particle, "name", "particle"))
        identities.append((state, _leg_particle_token(name)))
    totals = Counter(identities)
    seen: Counter[tuple[str, str]] = Counter()
    identifiers: list[str] = []
    for identity in identities:
        seen[identity] += 1
        suffix = f"-{seen[identity]}" if totals[identity] > 1 else ""
        identifiers.append(f"leg:{identity[0]}-{identity[1]}{suffix}")
    if len(set(identifiers)) != len(identifiers):
        raise CaptureError("stable external leg IDs are not unique")
    return tuple(identifiers)
