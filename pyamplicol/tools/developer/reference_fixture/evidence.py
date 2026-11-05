# SPDX-License-Identifier: 0BSD
"""Independent-oracle evidence validation for reference fixtures."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

from .model import (
    Observation,
    OracleEvidence,
    OracleProvenance,
    Process,
    ReferenceCase,
    ReferenceFixtureError,
    ReferencePoint,
    Tolerances,
    _EvidenceCoverage,
)
from .numerics import (
    _exact_sum,
    _required_stress_certified_digits,
    _validate_precision_metadata,
    _within_tolerance,
)


def _oracle_profile_limit(profile: str) -> Tolerances:
    if profile == "exact":
        return Tolerances(relative=Decimal(0), absolute=Decimal(0))
    if profile == "binary64":
        return Tolerances(
            relative=Decimal("0.0000000001"),
            absolute=Decimal("0.000000000001"),
        )
    return Tolerances(
        relative=Decimal("0.000000000001"),
        absolute=Decimal("0.000000000000001"),
    )


def _validate_oracle_provenance(oracle: OracleProvenance) -> None:
    ceiling = oracle.tolerance_ceiling
    if ceiling.relative < 0 or ceiling.absolute < 0:
        raise ReferenceFixtureError(
            f"oracle {oracle.id} tolerance ceiling must be nonnegative"
        )
    hard_limit = _oracle_profile_limit(oracle.validation_profile)
    if ceiling.relative > hard_limit.relative or ceiling.absolute > hard_limit.absolute:
        raise ReferenceFixtureError(
            f"oracle {oracle.id} tolerance ceiling exceeds its profile limit"
        )


def _validate_evidence(
    evidence: OracleEvidence,
    *,
    oracle: OracleProvenance,
    case: ReferenceCase,
    process: Process,
    point: ReferencePoint,
    observation: Observation,
) -> _EvidenceCoverage:
    if evidence.case_id != case.id or evidence.point_id != point.id:
        raise ReferenceFixtureError(
            f"evidence {evidence.id} does not identify observation {case.id}/{point.id}"
        )
    if evidence.input_sha256 != point.input_sha256():
        raise ReferenceFixtureError(
            f"evidence {evidence.id} input hash does not match point {point.id}"
        )
    if evidence.physics_case_sha256 != case.physics_case_sha256:
        raise ReferenceFixtureError(
            f"evidence {evidence.id} physics case hash does not match case {case.id}"
        )
    _validate_precision_metadata(
        arithmetic_precision_bits=evidence.arithmetic_precision_bits,
        round_trip_decimal_digits=evidence.round_trip_decimal_digits,
        certified_decimal_digits=evidence.certified_decimal_digits,
        where=f"evidence {evidence.id}",
    )
    if evidence.certified_decimal_digits > point.certified_decimal_digits:
        raise ReferenceFixtureError(
            f"evidence {evidence.id} overclaims the certified point precision"
        )
    if evidence.certified_decimal_digits < observation.certified_decimal_digits:
        raise ReferenceFixtureError(
            f"evidence {evidence.id} certification is below the observation claim"
        )
    if point.point_class == "stress" and evidence.certified_decimal_digits < (
        _required_stress_certified_digits(point)
    ):
        raise ReferenceFixtureError(
            f"stress evidence {evidence.id} certification is not commensurate "
            "with the stress metric"
        )
    allowed_arithmetic = {
        "exact": {"analytic", "rational"},
        "binary64": {"binary64"},
        "high-precision": {"analytic", "decimal", "rational"},
    }[oracle.validation_profile]
    if evidence.arithmetic not in allowed_arithmetic:
        raise ReferenceFixtureError(
            f"evidence {evidence.id} arithmetic is incompatible with oracle "
            f"profile {oracle.validation_profile}"
        )
    if (
        oracle.validation_profile == "high-precision"
        and evidence.arithmetic_precision_bits < 128
    ):
        raise ReferenceFixtureError(
            f"high-precision evidence {evidence.id} requires at least 128 "
            "arithmetic bits"
        )
    identity = evidence.process_identity
    if identity.expression != process.expression:
        raise ReferenceFixtureError(
            f"evidence {evidence.id} has the wrong process expression"
        )
    permutation = identity.source_to_row_permutation
    if sorted(permutation) != list(range(len(process.external_pdgs))):
        raise ReferenceFixtureError(
            f"evidence {evidence.id} has an invalid source-to-row permutation"
        )
    expected_pdgs = tuple(process.external_pdgs[position] for position in permutation)
    if identity.ordered_external_pdgs != expected_pdgs:
        raise ReferenceFixtureError(
            f"evidence {evidence.id} ordered PDGs do not match its permutation"
        )
    expected_legs = tuple(
        process.external_leg_ids[position] for position in permutation
    )
    if identity.ordered_external_leg_ids != expected_legs:
        raise ReferenceFixtureError(
            f"evidence {evidence.id} ordered leg identities do not match its "
            "permutation"
        )
    if not set(identity.ordered_color_legs) <= set(process.external_leg_ids):
        raise ReferenceFixtureError(
            f"evidence {evidence.id} color order references unknown external legs"
        )
    helicity_indices = {axis.id: index for index, axis in enumerate(case.helicities)}
    color_indices = {axis.id: index for index, axis in enumerate(case.colors)}
    if not set(evidence.helicity_ids) <= helicity_indices.keys():
        raise ReferenceFixtureError(
            f"evidence {evidence.id} references invalid helicity axes"
        )
    if not set(evidence.color_ids) <= color_indices.keys():
        raise ReferenceFixtureError(
            f"evidence {evidence.id} references invalid color axes"
        )
    if (
        case.color_accuracy == "lc"
        and case.coverage.color == "complete"
        and (identity.color_order_count != len(case.colors))
    ):
        raise ReferenceFixtureError(
            f"evidence {evidence.id} color-order count does not match the "
            "complete LC axis"
        )
    if evidence.tolerances.relative < 0 or evidence.tolerances.absolute < 0:
        raise ReferenceFixtureError(
            f"evidence {evidence.id} tolerances must be nonnegative"
        )
    if (
        evidence.tolerances.relative > oracle.tolerance_ceiling.relative
        or evidence.tolerances.absolute > oracle.tolerance_ceiling.absolute
    ):
        raise ReferenceFixtureError(
            f"evidence {evidence.id} exceeds its oracle-specific tolerance ceiling"
        )
    digit_ceiling = Decimal(1).scaleb(-evidence.certified_decimal_digits)
    if (
        evidence.tolerances.relative > digit_ceiling
        or evidence.tolerances.absolute > digit_ceiling
    ):
        raise ReferenceFixtureError(
            f"evidence {evidence.id} tolerances are not commensurate with its "
            "certified digits"
        )
    if evidence.arithmetic == "binary64" and (
        evidence.arithmetic_precision_bits != 53
        or evidence.round_trip_decimal_digits != 17
        or evidence.certified_decimal_digits > 15
    ):
        raise ReferenceFixtureError(
            f"binary64 evidence {evidence.id} must declare 53 arithmetic bits, "
            "17 round-trip digits, and at most 15 certified digits"
        )

    if evidence.coverage == "total":
        if evidence.helicity_ids or evidence.color_ids:
            raise ReferenceFixtureError(
                f"total evidence {evidence.id} must not declare resolved axes"
            )
        expected_total: Decimal | Fraction = observation.total
        if not _within_tolerance(
            expected_total,
            evidence.observed_total,
            evidence.tolerances,
        ):
            raise ReferenceFixtureError(
                f"evidence {evidence.id} total is outside its declared tolerance"
            )
        return _EvidenceCoverage(total=True)

    if evidence.coverage == "helicity-aggregate":
        totals = evidence.observed_helicity_totals
        if (
            totals is None
            or len(totals) != len(evidence.helicity_ids)
            or not evidence.helicity_ids
            or not evidence.color_ids
        ):
            raise ReferenceFixtureError(
                f"evidence {evidence.id} has the wrong helicity-aggregate shape"
            )
        if set(evidence.color_ids) != set(color_indices):
            raise ReferenceFixtureError(
                f"helicity-aggregate evidence {evidence.id} must cover the "
                "complete color axis"
            )
        expected_totals = tuple(
            _exact_sum(
                observation.values[helicity_indices[helicity_id]][
                    color_indices[color_id]
                ]
                for color_id in evidence.color_ids
            )
            for helicity_id in evidence.helicity_ids
        )
        for helicity_id, observed in zip(evidence.helicity_ids, totals, strict=True):
            if (
                case.helicities[helicity_indices[helicity_id]].structural_zero
                and observed != 0
            ):
                raise ReferenceFixtureError(
                    f"evidence {evidence.id} does not preserve an exact structural zero"
                )
        if any(
            not _within_tolerance(expected, observed, evidence.tolerances)
            for expected, observed in zip(expected_totals, totals, strict=True)
        ):
            raise ReferenceFixtureError(
                f"evidence {evidence.id} helicity aggregates are outside tolerance"
            )
        expected_total = sum(expected_totals, Fraction())
        if not _within_tolerance(
            expected_total,
            evidence.observed_total,
            evidence.tolerances,
        ):
            raise ReferenceFixtureError(
                f"evidence {evidence.id} aggregate total is outside tolerance"
            )
        return _EvidenceCoverage(helicity_aggregates=frozenset(evidence.helicity_ids))

    values = evidence.observed_values
    if (
        values is None
        or not evidence.helicity_ids
        or not evidence.color_ids
        or len(values) != len(evidence.helicity_ids)
        or any(len(row) != len(evidence.color_ids) for row in values)
    ):
        raise ReferenceFixtureError(
            f"evidence {evidence.id} has the wrong resolved shape"
        )
    covered: set[tuple[str, str]] = set()
    for helicity_id, observed_row in zip(evidence.helicity_ids, values, strict=True):
        row_index = helicity_indices[helicity_id]
        for color_id, observed in zip(evidence.color_ids, observed_row, strict=True):
            expected = observation.values[row_index][color_indices[color_id]]
            if case.helicities[row_index].structural_zero and observed != 0:
                raise ReferenceFixtureError(
                    f"evidence {evidence.id} does not preserve an exact structural zero"
                )
            if not _within_tolerance(expected, observed, evidence.tolerances):
                raise ReferenceFixtureError(
                    f"evidence {evidence.id} resolved values are outside tolerance"
                )
            covered.add((helicity_id, color_id))
    expected_total = _exact_sum(
        observation.values[helicity_indices[helicity_id]][color_indices[color_id]]
        for helicity_id in evidence.helicity_ids
        for color_id in evidence.color_ids
    )
    if not _within_tolerance(
        expected_total,
        evidence.observed_total,
        evidence.tolerances,
    ):
        raise ReferenceFixtureError(
            f"evidence {evidence.id} resolved total is outside tolerance"
        )
    return _EvidenceCoverage(resolved_cells=frozenset(covered))
