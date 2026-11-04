# SPDX-License-Identifier: 0BSD
"""Adversarial acceptance tests for proof-less numerical current relations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from pyamplicol.generation import numerical_current_warmup
from pyamplicol.generation.dag_equivalence import (
    NumericalCurrentRelationCertificate,
    _canonical_decimal_string,
    _numerical_current_observation_batch_sha256,
    _numerical_observations_sha256,
    certify_numerical_current_observations,
    verify_numerical_current_relation_certificate,
)

_SOURCE_SEMANTICS = hashlib.sha256(b"acceptance-source-semantics").hexdigest()
_OTHER_SOURCE_SEMANTICS = hashlib.sha256(b"drifted-source-semantics").hexdigest()
_RUNTIME_SCHEMA = hashlib.sha256(b"acceptance-runtime-schema").hexdigest()
_SOURCE_DAG = hashlib.sha256(b"acceptance-source-dag").hexdigest()
_CANDIDATE_CAPTURE = hashlib.sha256(b"candidate-capture").hexdigest()
_VERIFICATION_CAPTURE = hashlib.sha256(
    b"verification-capture"
).hexdigest()
_CANDIDATE_BATCH = hashlib.sha256(b"candidate-batch").hexdigest()
_VERIFICATION_BATCH = hashlib.sha256(b"verification-batch").hexdigest()
_ZERO = Decimal(0)

_CANDIDATE_REPRESENTATIVE = (
    (Decimal("1.25"), Decimal("0.5")),
    (Decimal("-2.0"), Decimal("3.5")),
    (Decimal("4.125"), Decimal("-0.75")),
    (Decimal("-8.0"), Decimal("-1.5")),
)
_VERIFICATION_REPRESENTATIVE = (
    (Decimal("0.375"), Decimal("-2.25")),
    (Decimal("6.5"), Decimal("1.0")),
    (Decimal("-3.125"), Decimal("4.75")),
    (Decimal("9.0"), Decimal("-0.625")),
)


def _scaled(
    values: Sequence[tuple[Decimal, Decimal]],
    factor: Decimal,
) -> tuple[tuple[Decimal, Decimal], ...]:
    return tuple((factor * real, factor * imag) for real, imag in values)


def _zero_values(count: int = 4) -> tuple[tuple[Decimal, Decimal], ...]:
    return tuple((_ZERO, _ZERO) for _ in range(count))


def _certify(
    relation_kind: str,
    *,
    candidate_current_values: Sequence[tuple[Decimal, Decimal]],
    verification_current_values: Sequence[tuple[Decimal, Decimal]],
    candidate_representative_values: (
        Sequence[tuple[Decimal, Decimal]] | None
    ) = _CANDIDATE_REPRESENTATIVE,
    verification_representative_values: (
        Sequence[tuple[Decimal, Decimal]] | None
    ) = _VERIFICATION_REPRESENTATIVE,
    current_id: int = 11,
    representative_id: int | None = 3,
    seed: int = 0x5059414D,
    relative_tolerance: float = 1.0e-70,
    absolute_tolerance: float = 1.0e-80,
) -> NumericalCurrentRelationCertificate | None:
    return certify_numerical_current_observations(
        current_id=current_id,
        representative_id=representative_id,
        relation_kind=relation_kind,  # type: ignore[arg-type]
        source_semantics_sha256=_SOURCE_SEMANTICS,
        runtime_schema_sha256=_RUNTIME_SCHEMA,
        source_dag_sha256=_SOURCE_DAG,
        candidate_capture_sha256=_CANDIDATE_CAPTURE,
        verification_capture_sha256=_VERIFICATION_CAPTURE,
        candidate_observation_batch_sha256=_CANDIDATE_BATCH,
        verification_observation_batch_sha256=_VERIFICATION_BATCH,
        candidate_current_values=candidate_current_values,
        candidate_representative_values=candidate_representative_values,
        verification_current_values=verification_current_values,
        verification_representative_values=verification_representative_values,
        precision_digits=96,
        seed=seed,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )


@pytest.mark.parametrize(
    (
        "relation_kind",
        "candidate_current",
        "verification_current",
        "candidate_representative",
        "verification_representative",
        "representative_id",
        "expected_factor",
    ),
    (
        (
            "equal",
            _CANDIDATE_REPRESENTATIVE,
            _VERIFICATION_REPRESENTATIVE,
            _CANDIDATE_REPRESENTATIVE,
            _VERIFICATION_REPRESENTATIVE,
            3,
            ["0x1.0000000000000p+0", "0x0.0p+0"],
        ),
        (
            "opposite",
            _scaled(_CANDIDATE_REPRESENTATIVE, Decimal(-1)),
            _scaled(_VERIFICATION_REPRESENTATIVE, Decimal(-1)),
            _CANDIDATE_REPRESENTATIVE,
            _VERIFICATION_REPRESENTATIVE,
            3,
            ["-0x1.0000000000000p+0", "0x0.0p+0"],
        ),
        (
            "zero",
            _zero_values(),
            _zero_values(),
            None,
            None,
            None,
            None,
        ),
    ),
)
@pytest.mark.parametrize("ambient_precision", (28, 50, 96))
def test_equal_opposite_and_zero_relations_certify_and_replay(
    relation_kind: str,
    candidate_current: Sequence[tuple[Decimal, Decimal]],
    verification_current: Sequence[tuple[Decimal, Decimal]],
    candidate_representative: Sequence[tuple[Decimal, Decimal]] | None,
    verification_representative: Sequence[tuple[Decimal, Decimal]] | None,
    representative_id: int | None,
    expected_factor: list[str] | None,
    ambient_precision: int,
) -> None:
    with localcontext() as context:
        context.prec = ambient_precision
        certificate = _certify(
            relation_kind,
            candidate_current_values=candidate_current,
            verification_current_values=verification_current,
            candidate_representative_values=candidate_representative,
            verification_representative_values=verification_representative,
            representative_id=representative_id,
        )

    assert certificate is not None
    payload = certificate.to_json_dict()
    assert payload["algorithm"] == (
        "authenticated-independent-recursive-decimal-probes-v1"
    )
    assert payload["proof_kind"] == "authenticated-numerical"
    assert payload["relation_kind"] == relation_kind
    assert payload["current_id"] == 11
    assert payload["representative_id"] == representative_id
    assert payload["factor_binary64"] == expected_factor
    assert payload["source_semantics_sha256"] == _SOURCE_SEMANTICS
    assert payload["candidate_probe_count"] == 4
    assert payload["verification_probe_count"] == 4
    assert payload["current_dimension"] == 1
    for digest_field in (
        "candidate_observations_sha256",
        "verification_observations_sha256",
        "runtime_schema_sha256",
        "source_dag_sha256",
        "candidate_capture_sha256",
        "verification_capture_sha256",
        "candidate_observation_batch_sha256",
        "verification_observation_batch_sha256",
        "probe_contract_sha256",
        "proof_sha256",
    ):
        assert len(payload[digest_field]) == 64

    restored = NumericalCurrentRelationCertificate.from_json_dict(payload)
    assert restored == certificate
    assert verify_numerical_current_relation_certificate(
        restored,
        source_semantics_sha256=_SOURCE_SEMANTICS,
    )
    assert not verify_numerical_current_relation_certificate(
        restored,
        source_semantics_sha256=_OTHER_SOURCE_SEMANTICS,
    )

    for field in (
        "relative_tolerance_binary64",
        "absolute_tolerance_binary64",
    ):
        noncanonical = dict(payload)
        noncanonical[field] = str(payload[field]).upper()
        with pytest.raises(ValueError, match="tolerances are invalid"):
            NumericalCurrentRelationCertificate.from_json_dict(noncanonical)
    negative_zero = dict(payload)
    negative_zero["relative_tolerance_binary64"] = "-0x0.0p+0"
    with pytest.raises(ValueError, match="tolerances are invalid"):
        NumericalCurrentRelationCertificate.from_json_dict(negative_zero)
    for spelling in ("-0", "0.0", "0e99"):
        noncanonical_residual = dict(payload)
        noncanonical_residual[
            "candidate_maximum_absolute_residual"
        ] = spelling
        with pytest.raises(ValueError, match="canonical decimal"):
            NumericalCurrentRelationCertificate.from_json_dict(
                noncanonical_residual
            )


def test_generic_numerical_tolerance_boundaries_reject_negative_zero() -> None:
    exact = _certify(
        "equal",
        candidate_current_values=_CANDIDATE_REPRESENTATIVE,
        verification_current_values=_VERIFICATION_REPRESENTATIVE,
        relative_tolerance=float.fromhex("0x0.0000000000001p-1022"),
        absolute_tolerance=0.0,
    )
    assert exact is not None
    assert exact.relative_tolerance.hex() == "0x0.0000000000001p-1022"
    for invalid in (-0.0, float("nan"), float("inf")):
        assert _certify(
            "equal",
            candidate_current_values=_CANDIDATE_REPRESENTATIVE,
            verification_current_values=_VERIFICATION_REPRESENTATIVE,
            relative_tolerance=invalid,
            absolute_tolerance=1.0e-80,
        ) is None


@pytest.mark.parametrize("relation_kind", ("equal", "opposite", "zero"))
def test_nearest_false_relation_with_documented_residual_is_rejected(
    relation_kind: str,
) -> None:
    residual = Decimal("2.7396e-2")
    if relation_kind == "zero":
        candidate_current = ((residual, _ZERO), *_zero_values(3))
        verification_current = ((_ZERO, residual), *_zero_values(3))
        candidate_representative = None
        verification_representative = None
        representative_id = None
    else:
        factor = Decimal(1) if relation_kind == "equal" else Decimal(-1)
        candidate_current = list(_scaled(_CANDIDATE_REPRESENTATIVE, factor))
        verification_current = list(_scaled(_VERIFICATION_REPRESENTATIVE, factor))
        candidate_current[2] = (
            candidate_current[2][0] + residual,
            candidate_current[2][1],
        )
        verification_current[1] = (
            verification_current[1][0],
            verification_current[1][1] - residual,
        )
        candidate_representative = _CANDIDATE_REPRESENTATIVE
        verification_representative = _VERIFICATION_REPRESENTATIVE
        representative_id = 3

    assert (
        _certify(
            relation_kind,
            candidate_current_values=candidate_current,
            verification_current_values=verification_current,
            candidate_representative_values=candidate_representative,
            verification_representative_values=verification_representative,
            representative_id=representative_id,
        )
        is None
    )


@pytest.mark.parametrize("ambient_precision", (28, 50, 96))
def test_tolerance_boundary_is_inclusive_but_nearest_outside_value_fails(
    ambient_precision: int,
) -> None:
    representative = tuple((Decimal(index), _ZERO) for index in range(1, 5))
    at_boundary = tuple(
        (Decimal(f"{index}.125"), _ZERO) for index in range(1, 5)
    )
    outside = tuple(
        (
            Decimal(f"{index}.125000000000000000000000000001"),
            _ZERO,
        )
        for index in range(1, 5)
    )

    with localcontext() as context:
        context.prec = ambient_precision
        assert (
            _certify(
                "equal",
                candidate_current_values=at_boundary,
                verification_current_values=at_boundary,
                candidate_representative_values=representative,
                verification_representative_values=representative,
                relative_tolerance=0.0,
                absolute_tolerance=0.125,
            )
            is not None
        )
        assert (
            _certify(
                "equal",
                candidate_current_values=outside,
                verification_current_values=outside,
                candidate_representative_values=representative,
                verification_representative_values=representative,
                relative_tolerance=0.0,
                absolute_tolerance=0.125,
            )
            is None
        )


@pytest.mark.parametrize("ambient_precision", (28, 50, 96))
def test_distinct_40_digit_observations_and_batch_hashes_do_not_collide(
    ambient_precision: int,
) -> None:
    first = Decimal("1.123456789012345678901234567890123456789")
    second = Decimal("1.123456789012345678901234567890123456788")
    signed_zero = Decimal("-0.000")
    point_sha256s = (hashlib.sha256(b"decimal-hash-point").hexdigest(),)
    first_observations = {0: ((first, signed_zero),)}
    second_observations = {0: ((second, signed_zero),)}
    assert len(first.as_tuple().digits) == 40
    assert len(second.as_tuple().digits) == 40

    with localcontext() as context:
        context.prec = ambient_precision
        first_relation_digest = _numerical_observations_sha256(
            relation_kind="zero",
            current_values=first_observations[0],
            representative_values=None,
        )
        second_relation_digest = _numerical_observations_sha256(
            relation_kind="zero",
            current_values=second_observations[0],
            representative_values=None,
        )
        first_discovery_batch = (
            _numerical_current_observation_batch_sha256(
                first_observations,
                point_sha256s=point_sha256s,
            )
        )
        second_discovery_batch = (
            _numerical_current_observation_batch_sha256(
                second_observations,
                point_sha256s=point_sha256s,
            )
        )
        first_capture_batch = (
            numerical_current_warmup._current_observation_batch_sha256(
                first_observations,
                point_sha256s=point_sha256s,
            )
        )
        second_capture_batch = (
            numerical_current_warmup._current_observation_batch_sha256(
                second_observations,
                point_sha256s=point_sha256s,
            )
        )

    assert _canonical_decimal_string(first) != _canonical_decimal_string(second)
    assert _canonical_decimal_string(signed_zero) == "0"
    assert numerical_current_warmup._decimal_string(signed_zero) == "0"
    assert _canonical_decimal_string(Decimal("1.0")) != (
        _canonical_decimal_string(Decimal("1.00"))
    )
    assert numerical_current_warmup._decimal_string(Decimal("1.0")) != (
        numerical_current_warmup._decimal_string(Decimal("1.00"))
    )
    assert first_relation_digest != second_relation_digest
    assert first_discovery_batch != second_discovery_batch
    assert first_capture_batch != second_capture_batch
    assert first_discovery_batch == first_capture_batch
    assert second_discovery_batch == second_capture_batch


def test_candidate_match_with_independent_verification_mismatch_fails_closed() -> None:
    assert (
        _certify(
            "equal",
            candidate_current_values=_CANDIDATE_REPRESENTATIVE,
            verification_current_values=_scaled(
                _VERIFICATION_REPRESENTATIVE,
                Decimal(-1),
            ),
        )
        is None
    )


@pytest.mark.parametrize(
    "nonfinite",
    (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")),
)
def test_nonfinite_observations_fail_closed(nonfinite: Decimal) -> None:
    current = list(_CANDIDATE_REPRESENTATIVE)
    current[1] = (nonfinite, current[1][1])
    assert (
        _certify(
            "equal",
            candidate_current_values=current,
            verification_current_values=_VERIFICATION_REPRESENTATIVE,
        )
        is None
    )


def test_malformed_relation_shapes_and_forward_representatives_fail_closed() -> None:
    assert (
        _certify(
            "equal",
            candidate_current_values=_CANDIDATE_REPRESENTATIVE[:-1],
            verification_current_values=_VERIFICATION_REPRESENTATIVE,
        )
        is None
    )
    assert (
        _certify(
            "equal",
            candidate_current_values=_CANDIDATE_REPRESENTATIVE,
            verification_current_values=_VERIFICATION_REPRESENTATIVE,
            candidate_representative_values=None,
            verification_representative_values=None,
        )
        is None
    )
    assert (
        _certify(
            "zero",
            candidate_current_values=_zero_values(),
            verification_current_values=_zero_values(),
            candidate_representative_values=_CANDIDATE_REPRESENTATIVE,
            verification_representative_values=_VERIFICATION_REPRESENTATIVE,
            representative_id=None,
        )
        is None
    )
    assert (
        _certify(
            "equal",
            candidate_current_values=_CANDIDATE_REPRESENTATIVE,
            verification_current_values=_VERIFICATION_REPRESENTATIVE,
            current_id=11,
            representative_id=11,
        )
        is None
    )


def test_certificate_is_deterministic_and_seed_bound() -> None:
    first = _certify(
        "equal",
        candidate_current_values=_CANDIDATE_REPRESENTATIVE,
        verification_current_values=_VERIFICATION_REPRESENTATIVE,
        seed=101,
    )
    repeated = _certify(
        "equal",
        candidate_current_values=_CANDIDATE_REPRESENTATIVE,
        verification_current_values=_VERIFICATION_REPRESENTATIVE,
        seed=101,
    )
    changed_seed = _certify(
        "equal",
        candidate_current_values=_CANDIDATE_REPRESENTATIVE,
        verification_current_values=_VERIFICATION_REPRESENTATIVE,
        seed=102,
    )
    assert first is not None
    assert repeated is not None
    assert changed_seed is not None
    assert first.to_json_dict() == repeated.to_json_dict()
    assert (
        first.to_json_dict()["proof_sha256"]
        != changed_seed.to_json_dict()["proof_sha256"]
    )


def test_certificate_fails_closed_on_capture_or_runtime_splicing() -> None:
    certificate = _certify(
        "equal",
        candidate_current_values=_CANDIDATE_REPRESENTATIVE,
        verification_current_values=_VERIFICATION_REPRESENTATIVE,
    )
    assert certificate is not None
    replacement_digest = hashlib.sha256(b"spliced-evidence").hexdigest()
    for field_name in (
        "runtime_schema_sha256",
        "source_dag_sha256",
        "candidate_capture_sha256",
        "verification_capture_sha256",
        "candidate_observation_batch_sha256",
        "verification_observation_batch_sha256",
    ):
        tampered = replace(
            certificate,
            **{field_name: replacement_digest},
        )
        assert not verify_numerical_current_relation_certificate(
            tampered,
            source_semantics_sha256=_SOURCE_SEMANTICS,
        )


def test_selector_permutation_is_not_silently_inferred_by_relation_certifier() -> None:
    permutation = (2, 0, 3, 1)
    permuted = tuple(_CANDIDATE_REPRESENTATIVE[index] for index in permutation)

    assert (
        _certify(
            "equal",
            candidate_current_values=permuted,
            verification_current_values=_VERIFICATION_REPRESENTATIVE,
        )
        is None
    )

    canonicalized = _certify(
        "equal",
        candidate_current_values=permuted,
        verification_current_values=tuple(
            _VERIFICATION_REPRESENTATIVE[index] for index in permutation
        ),
        candidate_representative_values=permuted,
        verification_representative_values=tuple(
            _VERIFICATION_REPRESENTATIVE[index] for index in permutation
        ),
    )
    assert canonicalized is not None
