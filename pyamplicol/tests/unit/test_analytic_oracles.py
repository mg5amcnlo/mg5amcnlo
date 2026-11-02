# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from decimal import Decimal, localcontext

import pytest

from tools.developer.analytic_oracles import (
    AnalyticOracleError,
    chiral_current_2to1,
    scalar_contact_2to2,
    scalar_gravity_2to2,
)


def _point(px: str, pz: str) -> tuple[tuple[Decimal, ...], ...]:
    """Construct an exact rational massless 2->2 point."""

    energy = Decimal("500")
    x = Decimal(px)
    z = Decimal(pz)
    zero = Decimal("0")
    return (
        (energy, zero, zero, energy),
        (energy, zero, zero, -energy),
        (energy, x, zero, z),
        (energy, -x, zero, -z),
    )


def _chiral_point() -> tuple[tuple[Decimal, ...], ...]:
    zero = Decimal("0")
    five = Decimal("5")
    ten = Decimal("10")
    return (
        (five, zero, zero, five),
        (five, zero, zero, -five),
        (ten, zero, zero, zero),
    )


def test_scalar_contact_oracle_includes_identical_particle_factor() -> None:
    assert scalar_contact_2to2().total == Decimal("0.5")
    assert scalar_contact_2to2(Decimal("3")).total == Decimal("4.5")
    assert scalar_contact_2to2().helicity_ids == ("h:+0,+0,+0,+0",)


def test_oracle_metadata_separates_precision_from_certified_accuracy() -> None:
    contact = scalar_contact_2to2(
        Decimal("0.1234567890123456789"),
        precision=60,
        certified_accuracy_digits=24,
    )
    gravity = scalar_gravity_2to2(
        _point("300", "400"),
        precision=80,
        certified_accuracy_digits=30,
    )

    assert contact.metadata.arithmetic_precision_decimal_digits == 60
    assert contact.metadata.certified_accuracy_decimal_digits == 24
    assert contact.metadata.kinematic_validation_relative_tolerance is None
    assert gravity.metadata.arithmetic_precision_decimal_digits == 80
    assert gravity.metadata.certified_accuracy_decimal_digits == 30
    assert gravity.metadata.kinematic_validation_relative_tolerance == Decimal("1e-33")


@pytest.mark.parametrize(
    ("point", "expected"),
    (
        (_point("500", "0"), Decimal("3906250000")),
        (_point("300", "400"), Decimal("506250000")),
        (_point("400", "300"), Decimal("1600000000")),
    ),
)
def test_scalar_gravity_oracle_matches_exact_kinematic_values(
    point: tuple[tuple[Decimal, ...], ...], expected: Decimal
) -> None:
    observation = scalar_gravity_2to2(point)

    assert observation.total == expected
    assert observation.helicity_ids == (
        "h:+0,+0,-2,-2",
        "h:+0,+0,-2,+2",
        "h:+0,+0,+2,-2",
        "h:+0,+0,+2,+2",
    )
    assert observation.resolved == (
        Decimal("0"),
        expected / 2,
        expected / 2,
        Decimal("0"),
    )
    assert sum(observation.resolved, Decimal("0")) == observation.total


def test_scalar_gravity_oracle_has_quartic_coupling_scaling() -> None:
    baseline = scalar_gravity_2to2(_point("300", "400"))
    scaled = scalar_gravity_2to2(_point("300", "400"), Decimal("2"))

    assert scaled.total == 16 * baseline.total


def test_chiral_current_oracle_exposes_leg_and_helicity_swaps() -> None:
    observation = chiral_current_2to1(
        _chiral_point(),
        vector_mass=Decimal("10"),
        left_coupling=Decimal("2"),
        right_coupling=Decimal("3"),
        certified_accuracy_digits=30,
    )

    assert observation.total == Decimal("2600")
    assert observation.resolved_by_helicity == (
        ("h:-1,+1,-1", Decimal("800")),
        ("h:+1,-1,+1", Decimal("1800")),
    )
    assert all(value != 0 for value in observation.resolved)
    assert observation.resolved[0] != observation.resolved[1]

    swapped_values = tuple(reversed(observation.resolved))
    assert swapped_values != observation.resolved
    assert dict(zip(observation.helicity_ids, swapped_values, strict=True)) != dict(
        observation.resolved_by_helicity
    )


def test_scalar_gravity_oracle_constructs_exact_redundant_invariant() -> None:
    exact_point = _point("300", "400")
    below_default_tolerance = list(exact_point)
    below_default_tolerance[3] = (
        *below_default_tolerance[3][:-1],
        below_default_tolerance[3][-1] + Decimal("1e-20"),
    )

    exact = scalar_gravity_2to2(exact_point)
    projected = scalar_gravity_2to2(tuple(below_default_tolerance))

    assert projected.total == exact.total
    with pytest.raises(AnalyticOracleError, match="not massless"):
        scalar_gravity_2to2(
            tuple(below_default_tolerance),
            precision=60,
            certified_accuracy_digits=30,
        )


def test_mass_shell_tolerance_tracks_certified_accuracy() -> None:
    point = list(_point("300", "400"))
    defect = Decimal("1e-20")
    point[2] = (*point[2][:-1], point[2][-1] + defect)
    point[3] = (*point[3][:-1], point[3][-1] - defect)

    scalar_gravity_2to2(tuple(point), certified_accuracy_digits=12)
    with pytest.raises(AnalyticOracleError, match="not massless"):
        scalar_gravity_2to2(tuple(point), precision=60, certified_accuracy_digits=30)


def test_momentum_conservation_tolerance_tracks_certified_accuracy() -> None:
    point = _point("300", "400")
    with localcontext() as context:
        context.prec = 80
        factor = Decimal("1.00000000000000000001")
        scaled_outgoing = tuple(
            tuple(+(component * factor) for component in momentum)
            for momentum in point[2:]
        )
    defect_point = (*point[:2], *scaled_outgoing)

    scalar_gravity_2to2(defect_point, certified_accuracy_digits=12)
    with pytest.raises(AnalyticOracleError, match="do not conserve"):
        scalar_gravity_2to2(defect_point, precision=60, certified_accuracy_digits=30)


def test_scalar_gravity_oracle_rejects_invalid_kinematics() -> None:
    with pytest.raises(AnalyticOracleError, match="four external momenta"):
        scalar_gravity_2to2(_point("300", "400")[:3])

    off_shell = list(_point("300", "400"))
    off_shell[2] = (*off_shell[2][:-1], Decimal("401"))
    with pytest.raises(AnalyticOracleError, match="not massless"):
        scalar_gravity_2to2(tuple(off_shell))

    nonconserving = list(_point("300", "400"))
    nonconserving[3] = (
        Decimal("500"),
        Decimal("-400"),
        Decimal("0"),
        Decimal("-300"),
    )
    with pytest.raises(AnalyticOracleError, match="do not conserve"):
        scalar_gravity_2to2(tuple(nonconserving))


@pytest.mark.parametrize("nonfinite", ("NaN", "Infinity", "-Infinity"))
def test_scalar_gravity_oracle_rejects_nonfinite_inputs(nonfinite: str) -> None:
    point = list(_point("300", "400"))
    point[2] = (Decimal(nonfinite), *point[2][1:])

    with pytest.raises(AnalyticOracleError, match="must be finite"):
        scalar_gravity_2to2(tuple(point))
    with pytest.raises(AnalyticOracleError, match="coupling must be finite"):
        scalar_gravity_2to2(_point("300", "400"), Decimal(nonfinite))


@pytest.mark.parametrize(
    ("precision", "accuracy", "message"),
    (
        (True, 12, "precision must be an integer"),
        (31, 12, "precision must be at least 32"),
        (40, True, "certified accuracy must be an integer"),
        (40, 0, "certified accuracy must be positive"),
        (40, 33, "must leave at least 8 arithmetic guard digits"),
    ),
)
def test_oracles_reject_invalid_precision_policy(
    precision: int, accuracy: int, message: str
) -> None:
    with pytest.raises(AnalyticOracleError, match=message):
        scalar_gravity_2to2(
            _point("300", "400"),
            precision=precision,
            certified_accuracy_digits=accuracy,
        )


def test_oracles_do_not_inherit_ambient_decimal_precision() -> None:
    coupling = Decimal("0.12345678901234567890123456789")
    with localcontext() as context:
        context.prec = 6
        contact_under_low_context = scalar_contact_2to2(
            coupling, precision=50, certified_accuracy_digits=20
        )
        gravity_under_low_context = scalar_gravity_2to2(
            _point("300", "400"),
            coupling,
            precision=50,
            certified_accuracy_digits=20,
        )
    with localcontext() as context:
        context.prec = 28
        contact_under_default_context = scalar_contact_2to2(
            coupling, precision=50, certified_accuracy_digits=20
        )
        gravity_under_default_context = scalar_gravity_2to2(
            _point("300", "400"),
            coupling,
            precision=50,
            certified_accuracy_digits=20,
        )

    assert contact_under_low_context == contact_under_default_context
    assert gravity_under_low_context == gravity_under_default_context


def test_scalar_gravity_oracle_uses_requested_arithmetic_precision() -> None:
    point = _point("300", "400")
    coupling = Decimal("0.1234567890123456789")
    low = scalar_gravity_2to2(point, coupling, precision=40)
    high = scalar_gravity_2to2(point, coupling, precision=100)

    assert len(str(high.total)) > len(str(low.total))
    assert low.metadata.arithmetic_precision_decimal_digits == 40
    assert high.metadata.arithmetic_precision_decimal_digits == 100


def test_chiral_current_rejects_wrong_vector_mass() -> None:
    with pytest.raises(AnalyticOracleError, match="off shell"):
        chiral_current_2to1(
            _chiral_point(),
            vector_mass=Decimal("11"),
            left_coupling=Decimal("2"),
            right_coupling=Decimal("3"),
        )
