# SPDX-License-Identifier: 0BSD
"""Proof-gated recursive current-value reuse for generated DAGs.

Current identity includes colour-sector and ordering metadata needed to build
and reduce amplitudes.  Those fields do not necessarily change the numerical
current value.  This module proves such value equivalences from the complete
recursive computation instead of guessing them from particle names or PDGs.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from math import copysign, fsum, isfinite
from typing import Literal, TypeAlias

from ..models.base import Model, VertexEvaluationEquivalence
from .contracts import runtime_coupling_parameter_names
from .dag_types import AmplitudeRoot, CurrentNode, GenericDAG, InteractionNode
from .numerical_candidate_index import (
    NumericalObservationCandidateIndex,
    build_numerical_observation_candidate_index,
    numerical_observation_tolerance_window_ids,
)

_ComplexWeight: TypeAlias = tuple[float, float]
_CurrentContract: TypeAlias = tuple[object, ...]
_EvaluationKey: TypeAlias = tuple[object, ...]
_CurrentTermVector: TypeAlias = tuple[tuple[_EvaluationKey, _ComplexWeight], ...]
_DiscoveryMode: TypeAlias = Literal["diagnostic", "certified-reuse"]

RELATION_DISCOVERY_SCHEMA_VERSION = 1
RELATION_DISCOVERY_CERTIFICATE_ALGORITHM = "exact-binary64-term-vector-replay-v1"
NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM = (
    "authenticated-independent-recursive-decimal-probes-v1"
)
GENERIC_DAG_NUMERICAL_SOURCE_SEMANTICS_ABI = (
    "pyamplicol-generic-dag-numerical-current-source-v1"
)
NUMERICAL_CURRENT_RELATION_SET_ABI = (
    "pyamplicol-authenticated-numerical-current-relation-set-v1"
)
NUMERICAL_CURRENT_CAPTURE_ABI = (
    "pyamplicol-generic-dag-current-observation-capture-v2"
)
NUMERICAL_CURRENT_CAPTURE_OUTPUT_PARTITION_ABI = (
    "pyamplicol-generic-dag-current-id-output-partitions-v1"
)
NUMERICAL_CURRENT_RELATION_WARNING_CODE = (
    "proofless-numerical-current-relations-applied-v1"
)
NUMERICAL_CURRENT_RELATION_WARNING = (
    "applied authenticated numerical equal/opposite/zero current reuse "
    "without an exact structural proof; disable with "
    "--no-numerical-current-reuse"
)
_MAX_REJECTED_DIAGNOSTICS = 16
_NUMERICAL_DECIMAL_GUARD_DIGITS = 16


def _is_negative_zero(value: float) -> bool:
    return value == 0.0 and copysign(1.0, value) < 0.0


def _rejected_candidate_diagnostics(
    rows: Sequence[Mapping[str, object]],
    *,
    total_rejected: int,
    full_census: Mapping[str, object],
    full_rejection_sha256: str,
) -> dict[str, object]:
    retained = [dict(row) for row in rows]
    return {
        "total_rejected_hypothesis_count": total_rejected,
        "retained_count": len(retained),
        "truncated": total_rejected > len(retained),
        "truncation_policy": (
            f"first-{_MAX_REJECTED_DIAGNOSTICS}-in-canonical-discovery-order-v1"
        ),
        "retained_sha256": _canonical_payload_sha256(
            {
                "abi": "pyamplicol-rejected-relation-diagnostics-v1",
                "rows": retained,
            }
        ),
        "full_census_sha256": _canonical_payload_sha256(
            {
                "abi": "pyamplicol-relation-discovery-full-census-v1",
                **dict(full_census),
            }
        ),
        "full_rejection_sha256": full_rejection_sha256,
    }


def _advance_rejection_digest(previous: str, row: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(row),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(bytes.fromhex(previous) + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class _CurrentValueEquivalence:
    """Exact relation ``current = factor * representative``."""

    representative_id: int
    factor: _ComplexWeight


@dataclass(frozen=True, slots=True)
class _CanonicalProjectiveTermVector:
    """A deterministic lookup key and its concrete normalization factor."""

    term_vector: _CurrentTermVector
    factor: _ComplexWeight


@dataclass(frozen=True, slots=True)
class _ProjectiveExpressionRepresentative:
    """One concrete current retained as a projective-class representative."""

    representative_id: int
    term_vector: _CurrentTermVector
    normalization_factor: _ComplexWeight


@dataclass(frozen=True, slots=True)
class DynamicColorProjectionCertificate:
    """Fail-closed proof summary for one multilinear color projection.

    The projection removes only ``CurrentIndex.color_state.sector_id``.
    Accuracy, line groups, basis keys, every non-colour current field, and the
    complete downstream physical-selector domain remain part of the class
    key.  A projected kernel or closure row is emitted only when the old
    ordered parent tuples are exactly the full Cartesian product of the
    corresponding member classes, with no duplicate tuple.
    """

    abi: str
    source_revision: str | None
    source_semantics_sha256: str
    selector_domains_sha256: str
    current_class_members_sha256: str
    old_to_new_current_ids: tuple[int, ...]
    current_remap_sha256: str
    interaction_groups_sha256: str
    closure_groups_sha256: str
    rectangle_cardinalities_sha256: str
    row_identity_sha256: str
    equality_check_status: str
    retained_color_metadata: str
    root_sector_policy: str
    before_current_count: int
    after_current_count: int
    before_interaction_count: int
    after_interaction_count: int
    before_evaluation_count: int
    after_evaluation_count: int
    before_amplitude_root_count: int
    after_amplitude_root_count: int
    projected_current_class_count: int
    rectangular_interaction_group_count: int
    rectangular_closure_group_count: int
    split_current_class_count: int

    @property
    def applied(self) -> bool:
        return (
            self.after_current_count < self.before_current_count
            or self.after_interaction_count < self.before_interaction_count
            or self.after_amplitude_root_count < self.before_amplitude_root_count
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "abi": self.abi,
            "source_revision": self.source_revision,
            "source_semantics_sha256": self.source_semantics_sha256,
            "selector_domains_sha256": self.selector_domains_sha256,
            "current_class_members_sha256": self.current_class_members_sha256,
            "old_to_new_current_ids": list(self.old_to_new_current_ids),
            "current_remap_sha256": self.current_remap_sha256,
            "interaction_groups_sha256": self.interaction_groups_sha256,
            "closure_groups_sha256": self.closure_groups_sha256,
            "rectangle_cardinalities_sha256": (
                self.rectangle_cardinalities_sha256
            ),
            "row_identity_sha256": self.row_identity_sha256,
            "equality_check_status": self.equality_check_status,
            "retained_color_metadata": self.retained_color_metadata,
            "root_sector_policy": self.root_sector_policy,
            "applied": self.applied,
            "before_current_count": self.before_current_count,
            "after_current_count": self.after_current_count,
            "before_interaction_count": self.before_interaction_count,
            "after_interaction_count": self.after_interaction_count,
            "before_evaluation_count": self.before_evaluation_count,
            "after_evaluation_count": self.after_evaluation_count,
            "before_amplitude_root_count": self.before_amplitude_root_count,
            "after_amplitude_root_count": self.after_amplitude_root_count,
            "projected_current_class_count": self.projected_current_class_count,
            "rectangular_interaction_group_count": (
                self.rectangular_interaction_group_count
            ),
            "rectangular_closure_group_count": self.rectangular_closure_group_count,
            "split_current_class_count": self.split_current_class_count,
        }


@dataclass(frozen=True, slots=True)
class ExactCurrentRelationCertificate:
    """Replayable proof that one concrete current is an exact multiple."""

    current_id: int
    representative_id: int
    factor: _ComplexWeight
    current_term_vector_sha256: str
    representative_term_vector_sha256: str
    proof_sha256: str
    algorithm: str = RELATION_DISCOVERY_CERTIFICATE_ALGORITHM

    def to_json_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "current_id": self.current_id,
            "representative_id": self.representative_id,
            "factor_binary64": [self.factor[0].hex(), self.factor[1].hex()],
            "current_term_vector_sha256": self.current_term_vector_sha256,
            "representative_term_vector_sha256": (
                self.representative_term_vector_sha256
            ),
            "proof_sha256": self.proof_sha256,
        }

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, object],
    ) -> ExactCurrentRelationCertificate:
        """Decode the exact representation emitted into generation metadata."""

        expected_fields = {
            "algorithm",
            "current_id",
            "representative_id",
            "factor_binary64",
            "current_term_vector_sha256",
            "representative_term_vector_sha256",
            "proof_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError("relation certificate fields are not canonical")
        try:
            algorithm = payload["algorithm"]
            current_id = payload["current_id"]
            representative_id = payload["representative_id"]
            raw_factor = payload["factor_binary64"]
            current_digest = payload["current_term_vector_sha256"]
            representative_digest = payload["representative_term_vector_sha256"]
            proof_digest = payload["proof_sha256"]
        except KeyError as error:
            raise ValueError(
                f"relation certificate is missing {error.args[0]!r}"
            ) from error
        if algorithm != RELATION_DISCOVERY_CERTIFICATE_ALGORITHM:
            raise ValueError("relation certificate uses an unsupported algorithm")
        if (
            type(current_id) is not int
            or type(representative_id) is not int
            or current_id < 0
            or representative_id < 0
        ):
            raise ValueError("relation certificate current IDs must be nonnegative")
        if (
            not isinstance(raw_factor, list)
            or len(raw_factor) != 2
            or any(not isinstance(component, str) for component in raw_factor)
        ):
            raise ValueError(
                "relation certificate factor must contain two binary64 hex strings"
            )
        try:
            factor = (float.fromhex(raw_factor[0]), float.fromhex(raw_factor[1]))
        except ValueError as error:
            raise ValueError(
                "relation certificate factor has invalid binary64 encoding"
            ) from error
        if not _complex_weight_is_finite(factor) or factor == (0.0, 0.0):
            raise ValueError("relation certificate factor must be finite and nonzero")
        digests = (current_digest, representative_digest, proof_digest)
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        ):
            raise ValueError("relation certificate digests must be lowercase SHA-256")
        assert isinstance(current_digest, str)
        assert isinstance(representative_digest, str)
        assert isinstance(proof_digest, str)
        certificate = cls(
            current_id=current_id,
            representative_id=representative_id,
            factor=factor,
            current_term_vector_sha256=current_digest,
            representative_term_vector_sha256=representative_digest,
            proof_sha256=proof_digest,
        )
        if (
            _certificate_proof_sha256(
                current_id=certificate.current_id,
                representative_id=certificate.representative_id,
                factor=certificate.factor,
                current_term_vector_sha256=(certificate.current_term_vector_sha256),
                representative_term_vector_sha256=(
                    certificate.representative_term_vector_sha256
                ),
            )
            != certificate.proof_sha256
        ):
            raise ValueError("relation certificate proof digest does not replay")
        return certificate


@dataclass(frozen=True, slots=True)
class NumericalCurrentRelationCertificate:
    """Replayable evidence for one numerically certified current relation.

    The certificate commits to two disjoint observation sets.  The raw current
    values remain generation evidence; artifact loading validates this
    canonical certificate and never reruns relation discovery.
    """

    current_id: int
    representative_id: int | None
    relation_kind: Literal["equal", "opposite", "zero"]
    factor: _ComplexWeight | None
    source_semantics_sha256: str
    runtime_schema_sha256: str
    source_dag_sha256: str
    candidate_capture_sha256: str
    verification_capture_sha256: str
    candidate_observation_batch_sha256: str
    verification_observation_batch_sha256: str
    precision_digits: int
    seed: int
    relative_tolerance: float
    absolute_tolerance: float
    candidate_probe_count: int
    verification_probe_count: int
    current_dimension: int
    candidate_maximum_absolute_residual: Decimal
    candidate_maximum_relative_residual: Decimal
    candidate_maximum_tolerance_ratio: Decimal
    verification_maximum_absolute_residual: Decimal
    verification_maximum_relative_residual: Decimal
    verification_maximum_tolerance_ratio: Decimal
    candidate_observations_sha256: str
    verification_observations_sha256: str
    probe_contract_sha256: str
    proof_sha256: str
    proof_kind: str = "authenticated-numerical"
    algorithm: str = NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM

    def to_json_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "proof_kind": self.proof_kind,
            "relation_kind": self.relation_kind,
            "current_id": self.current_id,
            "representative_id": self.representative_id,
            "factor_binary64": (
                None
                if self.factor is None
                else [self.factor[0].hex(), self.factor[1].hex()]
            ),
            "source_semantics_sha256": self.source_semantics_sha256,
            "runtime_schema_sha256": self.runtime_schema_sha256,
            "source_dag_sha256": self.source_dag_sha256,
            "candidate_capture_sha256": self.candidate_capture_sha256,
            "verification_capture_sha256": (
                self.verification_capture_sha256
            ),
            "candidate_observation_batch_sha256": (
                self.candidate_observation_batch_sha256
            ),
            "verification_observation_batch_sha256": (
                self.verification_observation_batch_sha256
            ),
            "precision_digits": self.precision_digits,
            "seed": self.seed,
            "relative_tolerance_binary64": self.relative_tolerance.hex(),
            "absolute_tolerance_binary64": self.absolute_tolerance.hex(),
            "candidate_probe_count": self.candidate_probe_count,
            "verification_probe_count": self.verification_probe_count,
            "current_dimension": self.current_dimension,
            "candidate_maximum_absolute_residual": (
                _canonical_decimal_string(
                    self.candidate_maximum_absolute_residual
                )
            ),
            "candidate_maximum_relative_residual": (
                _canonical_decimal_string(
                    self.candidate_maximum_relative_residual
                )
            ),
            "candidate_maximum_tolerance_ratio": (
                _canonical_decimal_string(
                    self.candidate_maximum_tolerance_ratio
                )
            ),
            "verification_maximum_absolute_residual": (
                _canonical_decimal_string(
                    self.verification_maximum_absolute_residual
                )
            ),
            "verification_maximum_relative_residual": (
                _canonical_decimal_string(
                    self.verification_maximum_relative_residual
                )
            ),
            "verification_maximum_tolerance_ratio": (
                _canonical_decimal_string(
                    self.verification_maximum_tolerance_ratio
                )
            ),
            "candidate_observations_sha256": (
                self.candidate_observations_sha256
            ),
            "verification_observations_sha256": (
                self.verification_observations_sha256
            ),
            "probe_contract_sha256": self.probe_contract_sha256,
            "proof_sha256": self.proof_sha256,
        }

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, object],
    ) -> NumericalCurrentRelationCertificate:
        expected_fields = {
            "algorithm",
            "proof_kind",
            "relation_kind",
            "current_id",
            "representative_id",
            "factor_binary64",
            "source_semantics_sha256",
            "runtime_schema_sha256",
            "source_dag_sha256",
            "candidate_capture_sha256",
            "verification_capture_sha256",
            "candidate_observation_batch_sha256",
            "verification_observation_batch_sha256",
            "precision_digits",
            "seed",
            "relative_tolerance_binary64",
            "absolute_tolerance_binary64",
            "candidate_probe_count",
            "verification_probe_count",
            "current_dimension",
            "candidate_maximum_absolute_residual",
            "candidate_maximum_relative_residual",
            "candidate_maximum_tolerance_ratio",
            "verification_maximum_absolute_residual",
            "verification_maximum_relative_residual",
            "verification_maximum_tolerance_ratio",
            "candidate_observations_sha256",
            "verification_observations_sha256",
            "probe_contract_sha256",
            "proof_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError(
                "numerical current relation certificate fields are not canonical"
            )
        algorithm = payload["algorithm"]
        proof_kind = payload["proof_kind"]
        relation_kind = payload["relation_kind"]
        current_id = payload["current_id"]
        representative_id = payload["representative_id"]
        factor_payload = payload["factor_binary64"]
        source_digest = payload["source_semantics_sha256"]
        runtime_schema_digest = payload["runtime_schema_sha256"]
        source_dag_digest = payload["source_dag_sha256"]
        candidate_capture_digest = payload["candidate_capture_sha256"]
        verification_capture_digest = payload[
            "verification_capture_sha256"
        ]
        candidate_batch_digest = payload[
            "candidate_observation_batch_sha256"
        ]
        verification_batch_digest = payload[
            "verification_observation_batch_sha256"
        ]
        precision_digits = payload["precision_digits"]
        seed = payload["seed"]
        relative_payload = payload["relative_tolerance_binary64"]
        absolute_payload = payload["absolute_tolerance_binary64"]
        candidate_probe_count = payload["candidate_probe_count"]
        verification_probe_count = payload["verification_probe_count"]
        current_dimension = payload["current_dimension"]
        if (
            algorithm != NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM
            or proof_kind != "authenticated-numerical"
            or relation_kind not in {"equal", "opposite", "zero"}
        ):
            raise ValueError(
                "numerical current relation certificate has unsupported policy"
            )
        if type(current_id) is not int or current_id < 0:
            raise ValueError(
                "numerical current relation current ID must be nonnegative"
            )
        if representative_id is not None and (
            type(representative_id) is not int
            or representative_id < 0
            or representative_id >= current_id
        ):
            raise ValueError(
                "numerical current relation representative must precede its current"
            )
        if (
            type(precision_digits) is not int
            or precision_digits < 80
            or type(seed) is not int
            or seed < 0
            or type(candidate_probe_count) is not int
            or candidate_probe_count < 2
            or type(verification_probe_count) is not int
            or verification_probe_count < 2
            or type(current_dimension) is not int
            or current_dimension < 1
        ):
            raise ValueError(
                "numerical current relation probe contract is invalid"
            )
        if not isinstance(relative_payload, str) or not isinstance(
            absolute_payload,
            str,
        ):
            raise ValueError(
                "numerical current relation tolerances must use binary64 hex"
            )
        try:
            relative_tolerance = float.fromhex(relative_payload)
            absolute_tolerance = float.fromhex(absolute_payload)
        except ValueError as error:
            raise ValueError(
                "numerical current relation tolerance encoding is invalid"
            ) from error
        if (
            not isfinite(relative_tolerance)
            or not isfinite(absolute_tolerance)
            or relative_tolerance < 0.0
            or absolute_tolerance < 0.0
            or _is_negative_zero(relative_tolerance)
            or _is_negative_zero(absolute_tolerance)
            or relative_payload != relative_tolerance.hex()
            or absolute_payload != absolute_tolerance.hex()
            or (relative_tolerance == 0.0 and absolute_tolerance == 0.0)
        ):
            raise ValueError(
                "numerical current relation tolerances are invalid"
            )
        factor = _decode_numerical_relation_factor(
            relation_kind,
            representative_id,
            factor_payload,
        )
        digest_fields = (
            source_digest,
            runtime_schema_digest,
            source_dag_digest,
            candidate_capture_digest,
            verification_capture_digest,
            candidate_batch_digest,
            verification_batch_digest,
            payload["candidate_observations_sha256"],
            payload["verification_observations_sha256"],
            payload["probe_contract_sha256"],
            payload["proof_sha256"],
        )
        if any(not _is_sha256(value) for value in digest_fields):
            raise ValueError(
                "numerical current relation digests must be lowercase SHA-256"
            )
        decimal_field_names = (
            "candidate_maximum_absolute_residual",
            "candidate_maximum_relative_residual",
            "candidate_maximum_tolerance_ratio",
            "verification_maximum_absolute_residual",
            "verification_maximum_relative_residual",
            "verification_maximum_tolerance_ratio",
        )
        residuals: list[Decimal] = []
        for field_name in decimal_field_names:
            raw = payload[field_name]
            if not isinstance(raw, str):
                raise ValueError(
                    "numerical current relation residuals must be decimal strings"
                )
            try:
                residual = Decimal(raw)
            except ArithmeticError as error:
                raise ValueError(
                    "numerical current relation residual encoding is invalid"
                ) from error
            if (
                not residual.is_finite()
                or residual < 0
                or raw != _canonical_decimal_string(residual)
            ):
                raise ValueError(
                    "numerical current relation residuals must be finite "
                    "and nonnegative canonical decimal strings"
                )
            residuals.append(residual)
        certificate = cls(
            current_id=current_id,
            representative_id=representative_id,
            relation_kind=relation_kind,
            factor=factor,
            source_semantics_sha256=source_digest,
            runtime_schema_sha256=str(runtime_schema_digest),
            source_dag_sha256=str(source_dag_digest),
            candidate_capture_sha256=str(candidate_capture_digest),
            verification_capture_sha256=str(
                verification_capture_digest
            ),
            candidate_observation_batch_sha256=str(
                candidate_batch_digest
            ),
            verification_observation_batch_sha256=str(
                verification_batch_digest
            ),
            precision_digits=precision_digits,
            seed=seed,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            candidate_probe_count=candidate_probe_count,
            verification_probe_count=verification_probe_count,
            current_dimension=current_dimension,
            candidate_maximum_absolute_residual=residuals[0],
            candidate_maximum_relative_residual=residuals[1],
            candidate_maximum_tolerance_ratio=residuals[2],
            verification_maximum_absolute_residual=residuals[3],
            verification_maximum_relative_residual=residuals[4],
            verification_maximum_tolerance_ratio=residuals[5],
            candidate_observations_sha256=str(
                payload["candidate_observations_sha256"]
            ),
            verification_observations_sha256=str(
                payload["verification_observations_sha256"]
            ),
            probe_contract_sha256=str(payload["probe_contract_sha256"]),
            proof_sha256=str(payload["proof_sha256"]),
        )
        if not verify_numerical_current_relation_certificate(
            certificate,
            source_semantics_sha256=source_digest,
        ):
            raise ValueError(
                "numerical current relation certificate proof does not replay"
            )
        return certificate


@dataclass(frozen=True, slots=True)
class NumericalCurrentAppliedMapping:
    """Execution mapping derived from one authenticated certificate."""

    current_id: int
    representative_id: int | None
    execution_representative_id: int
    relation_kind: Literal["equal", "opposite", "zero"]
    factor: _ComplexWeight
    certificate_proof_sha256: str
    resolved_current_ids: tuple[int, ...]
    projected_interaction_count: int
    projected_interaction_ids_sha256: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "current_id": self.current_id,
            "representative_id": self.representative_id,
            "execution_representative_id": self.execution_representative_id,
            "relation_kind": self.relation_kind,
            "factor_binary64": [self.factor[0].hex(), self.factor[1].hex()],
            "certificate_proof_sha256": self.certificate_proof_sha256,
            "resolved_current_ids": list(self.resolved_current_ids),
            "projected_interaction_count": self.projected_interaction_count,
            "projected_interaction_ids_sha256": (
                self.projected_interaction_ids_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class NumericalCurrentRelationApplicationReport:
    """Manifest-ready replay result for one generic-DAG relation set."""

    requested_mode: Literal["diagnostic", "certified-reuse"]
    execution_mode: Literal["compiled", "eager"]
    color_accuracy: str
    source_semantics_sha256: str
    runtime_schema_sha256: str | None
    source_dag_sha256: str | None
    candidate_capture_sha256: str | None
    verification_capture_sha256: str | None
    state: str
    certificate_replay_status: str
    certificate_set_sha256: str
    certificates: tuple[NumericalCurrentRelationCertificate, ...]
    mappings: tuple[NumericalCurrentAppliedMapping, ...]
    interaction_evaluation_count_before: int
    interaction_evaluation_count_projected: int
    applied_relation_count: int

    @property
    def warning_required(self) -> bool:
        return self.applied_relation_count > 0

    def to_json_dict(self) -> dict[str, object]:
        relation_kind_counts = {
            relation_kind: sum(
                certificate.relation_kind == relation_kind
                for certificate in self.certificates
            )
            for relation_kind in ("equal", "opposite", "zero")
        }
        return {
            "schema_version": 1,
            "abi": NUMERICAL_CURRENT_RELATION_SET_ABI,
            "requested_mode": self.requested_mode,
            "state": self.state,
            "scope": {
                "execution_mode": self.execution_mode,
                "color_accuracy": self.color_accuracy,
                "representation": "generic-dag",
            },
            "source_semantics": {
                "abi": GENERIC_DAG_NUMERICAL_SOURCE_SEMANTICS_ABI,
                "sha256": self.source_semantics_sha256,
            },
            "runtime_schema_sha256": self.runtime_schema_sha256,
            "source_dag_sha256": self.source_dag_sha256,
            "candidate_capture_sha256": self.candidate_capture_sha256,
            "verification_capture_sha256": (
                self.verification_capture_sha256
            ),
            "certificate_replay": {
                "algorithm": (
                    NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM
                ),
                "status": self.certificate_replay_status,
                "certificate_set_sha256": self.certificate_set_sha256,
            },
            "certified_relation_count": len(self.certificates),
            "applied_relation_count": self.applied_relation_count,
            "relation_kind_counts": relation_kind_counts,
            "interaction_evaluation_count_before": (
                self.interaction_evaluation_count_before
            ),
            "interaction_evaluation_count_projected": (
                self.interaction_evaluation_count_projected
            ),
            "interaction_evaluation_savings_projected": max(
                0,
                self.interaction_evaluation_count_before
                - self.interaction_evaluation_count_projected,
            ),
            "certificates": [
                certificate.to_json_dict()
                for certificate in self.certificates
            ],
            "mappings": [mapping.to_json_dict() for mapping in self.mappings],
            "warning": (
                {
                    "required": True,
                    "emit": "once-per-generated-artifact",
                    "code": NUMERICAL_CURRENT_RELATION_WARNING_CODE,
                    "message": NUMERICAL_CURRENT_RELATION_WARNING,
                }
                if self.warning_required
                else {
                    "required": False,
                    "emit": "never",
                    "code": None,
                    "message": None,
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class NumericalCurrentRelationApplicationResult:
    dag: GenericDAG
    report: NumericalCurrentRelationApplicationReport


@dataclass(frozen=True, slots=True)
class NumericalCurrentObservationDiscoveryReport:
    """Bounded warm-up evidence before any numerical mapping is applied."""

    execution_mode: Literal["compiled", "eager"]
    color_accuracy: str
    source_semantics_sha256: str
    runtime_schema_sha256: str
    source_dag_sha256: str
    precision_digits: int
    seed: int
    relative_tolerance: float
    absolute_tolerance: float
    candidate_point_sha256s: tuple[str, ...]
    verification_point_sha256s: tuple[str, ...]
    candidate_kinematic_sha256s: tuple[str, ...]
    verification_kinematic_sha256s: tuple[str, ...]
    candidate_parameter_context_sha256s: tuple[str, ...]
    verification_parameter_context_sha256s: tuple[str, ...]
    candidate_capture_sha256: str
    verification_capture_sha256: str
    candidate_observation_batch_sha256: str
    verification_observation_batch_sha256: str
    state: str
    inspected_current_count: int
    structurally_proven_current_count: int
    tested_hypothesis_count: int
    theoretical_pair_hypothesis_count: int
    screened_pair_hypothesis_count: int
    candidate_index_contract_count: int
    exhaustive_fallback_contract_count: int
    numerical_candidate_count: int
    verification_rejected_count: int
    rejected_hypothesis_count: int
    rejected_decision_sha256: str
    certificates: tuple[NumericalCurrentRelationCertificate, ...]
    rejected_candidates: tuple[dict[str, object], ...]
    nearest_rejected_hypothesis: dict[str, object] | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "state": self.state,
            "scope": {
                "execution_mode": self.execution_mode,
                "color_accuracy": self.color_accuracy,
                "representation": "generic-dag",
            },
            "source_semantics": {
                "abi": GENERIC_DAG_NUMERICAL_SOURCE_SEMANTICS_ABI,
                "sha256": self.source_semantics_sha256,
            },
            "runtime_schema_sha256": self.runtime_schema_sha256,
            "source_dag_sha256": self.source_dag_sha256,
            "probe_contract": {
                "algorithm": (
                    NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM
                ),
                "precision_digits": self.precision_digits,
                "seed": self.seed,
                "relative_tolerance_binary64": (
                    self.relative_tolerance.hex()
                ),
                "absolute_tolerance_binary64": (
                    self.absolute_tolerance.hex()
                ),
                "candidate_point_sha256s": list(
                    self.candidate_point_sha256s
                ),
                "verification_point_sha256s": list(
                    self.verification_point_sha256s
                ),
                "candidate_kinematic_sha256s": list(
                    self.candidate_kinematic_sha256s
                ),
                "verification_kinematic_sha256s": list(
                    self.verification_kinematic_sha256s
                ),
                "candidate_parameter_context_sha256s": list(
                    self.candidate_parameter_context_sha256s
                ),
                "verification_parameter_context_sha256s": list(
                    self.verification_parameter_context_sha256s
                ),
                "candidate_capture_sha256": (
                    self.candidate_capture_sha256
                ),
                "verification_capture_sha256": (
                    self.verification_capture_sha256
                ),
                "candidate_observation_batch_sha256": (
                    self.candidate_observation_batch_sha256
                ),
                "verification_observation_batch_sha256": (
                    self.verification_observation_batch_sha256
                ),
                "deterministic": True,
                "independent_verification": True,
            },
            "inspected_current_count": self.inspected_current_count,
            "structurally_proven_current_count": (
                self.structurally_proven_current_count
            ),
            "tested_hypothesis_count": self.tested_hypothesis_count,
            "candidate_index": {
                "algorithm": (
                    "complete-contract-anchor-tolerance-window-v1"
                ),
                "completeness": "complete-within-configured-tolerance",
                "contract_count": self.candidate_index_contract_count,
                "exhaustive_fallback_contract_count": (
                    self.exhaustive_fallback_contract_count
                ),
                "theoretical_pair_hypothesis_count": (
                    self.theoretical_pair_hypothesis_count
                ),
                "screened_pair_hypothesis_count": (
                    self.screened_pair_hypothesis_count
                ),
                "zero_hypothesis_count": (
                    self.tested_hypothesis_count
                    - self.screened_pair_hypothesis_count
                ),
                "nearest_rejected_scope": (
                    "zero-and-tolerance-window-screened-hypotheses"
                ),
            },
            "numerical_candidate_count": self.numerical_candidate_count,
            "verification_rejected_count": self.verification_rejected_count,
            "rejected_hypothesis_count": self.rejected_hypothesis_count,
            "certified_numerical_relation_count": len(self.certificates),
            "certificates": [
                certificate.to_json_dict()
                for certificate in self.certificates
            ],
            "rejected_candidates": list(self.rejected_candidates),
            "rejected_candidate_diagnostics": (
                _rejected_candidate_diagnostics(
                    self.rejected_candidates,
                    total_rejected=self.rejected_hypothesis_count,
                    full_census={
                        "candidate_capture_sha256": (
                            self.candidate_capture_sha256
                        ),
                        "verification_capture_sha256": (
                            self.verification_capture_sha256
                        ),
                        "tested_hypothesis_count": (
                            self.tested_hypothesis_count
                        ),
                        "numerical_candidate_count": (
                            self.numerical_candidate_count
                        ),
                        "verification_rejected_count": (
                            self.verification_rejected_count
                        ),
                        "rejected_hypothesis_count": (
                            self.rejected_hypothesis_count
                        ),
                        "certified_relation_count": len(self.certificates),
                        "rejection_decision_sha256": (
                            self.rejected_decision_sha256
                        ),
                    },
                    full_rejection_sha256=self.rejected_decision_sha256,
                )
            ),
            "nearest_rejected_hypothesis": (
                self.nearest_rejected_hypothesis
            ),
            "warning": {
                "required": False,
                "reason": (
                    "discovery alone never warns; the artifact aggregator "
                    "warns once only when certified proof-less mappings are "
                    "actually applied"
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class NumericalCurrentObservationDiscoveryResult:
    certificates: tuple[NumericalCurrentRelationCertificate, ...]
    report: NumericalCurrentObservationDiscoveryReport
    structural_equivalences: tuple[_CurrentValueEquivalence, ...]


_NumericalObservationCandidateIndex = NumericalObservationCandidateIndex[Decimal]


def _build_numerical_observation_candidate_index(
    member_ids: Sequence[int],
    observations: Mapping[int, tuple[tuple[Decimal, Decimal], ...]],
) -> _NumericalObservationCandidateIndex:
    return build_numerical_observation_candidate_index(
        member_ids,
        observations,
        normalize=lambda value: value,
    )


def _numerical_observation_tolerance_window_ids(
    index: _NumericalObservationCandidateIndex,
    current_values: tuple[tuple[Decimal, Decimal], ...],
    *,
    relation_kind: Literal["equal", "opposite"],
    current_id: int,
    relative_tolerance: Decimal,
    absolute_tolerance: Decimal,
    precision_digits: int,
) -> tuple[int, ...]:
    """Use the shared sound window under the configured Decimal context."""

    if type(precision_digits) is not int or precision_digits < 80:
        raise ValueError("numerical candidate index precision is invalid")
    with localcontext() as context:
        context.prec = precision_digits + _NUMERICAL_DECIMAL_GUARD_DIGITS
        context.rounding = ROUND_HALF_EVEN
        return numerical_observation_tolerance_window_ids(
            index,
            current_values,
            relation_kind=relation_kind,
            current_id=current_id,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            normalize=lambda value: value,
        )


def certify_numerical_current_observations(
    *,
    current_id: int,
    representative_id: int | None,
    relation_kind: Literal["equal", "opposite", "zero"],
    source_semantics_sha256: str,
    runtime_schema_sha256: str,
    source_dag_sha256: str,
    candidate_capture_sha256: str,
    verification_capture_sha256: str,
    candidate_observation_batch_sha256: str,
    verification_observation_batch_sha256: str,
    candidate_current_values: Sequence[tuple[Decimal, Decimal]],
    candidate_representative_values: (
        Sequence[tuple[Decimal, Decimal]] | None
    ),
    verification_current_values: Sequence[tuple[Decimal, Decimal]],
    verification_representative_values: (
        Sequence[tuple[Decimal, Decimal]] | None
    ),
    precision_digits: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    candidate_probe_count: int | None = None,
    verification_probe_count: int | None = None,
) -> NumericalCurrentRelationCertificate | None:
    """Certify one ±1/zero relation over independent Decimal observations.

    Invalid, non-finite, forward, shape-inconsistent, or unstable hypotheses
    fail closed by returning ``None``.
    """

    if (
        relation_kind not in {"equal", "opposite", "zero"}
        or type(current_id) is not int
        or current_id < 0
        or not _is_sha256(source_semantics_sha256)
        or any(
            not _is_sha256(value)
            for value in (
                runtime_schema_sha256,
                source_dag_sha256,
                candidate_capture_sha256,
                verification_capture_sha256,
                candidate_observation_batch_sha256,
                verification_observation_batch_sha256,
            )
        )
        or candidate_capture_sha256 == verification_capture_sha256
        or candidate_observation_batch_sha256
        == verification_observation_batch_sha256
        or type(precision_digits) is not int
        or precision_digits < 80
        or type(seed) is not int
        or seed < 0
        or not isinstance(relative_tolerance, int | float)
        or isinstance(relative_tolerance, bool)
        or not isinstance(absolute_tolerance, int | float)
        or isinstance(absolute_tolerance, bool)
    ):
        return None
    relative = float(relative_tolerance)
    absolute = float(absolute_tolerance)
    if (
        not isfinite(relative)
        or not isfinite(absolute)
        or relative < 0.0
        or absolute < 0.0
        or _is_negative_zero(relative)
        or _is_negative_zero(absolute)
        or (relative == 0.0 and absolute == 0.0)
    ):
        return None
    candidate_current = _validated_decimal_observations(
        candidate_current_values
    )
    verification_current = _validated_decimal_observations(
        verification_current_values
    )
    if (
        candidate_current is None
        or verification_current is None
        or len(candidate_current) < 2
        or len(verification_current) < 2
    ):
        return None
    resolved_candidate_probe_count = (
        len(candidate_current)
        if candidate_probe_count is None
        else candidate_probe_count
    )
    resolved_verification_probe_count = (
        len(verification_current)
        if verification_probe_count is None
        else verification_probe_count
    )
    if (
        type(resolved_candidate_probe_count) is not int
        or resolved_candidate_probe_count < 2
        or type(resolved_verification_probe_count) is not int
        or resolved_verification_probe_count < 2
        or len(candidate_current) % resolved_candidate_probe_count != 0
        or len(verification_current) % resolved_verification_probe_count != 0
    ):
        return None
    current_dimension = (
        len(candidate_current) // resolved_candidate_probe_count
    )
    if (
        current_dimension < 1
        or len(verification_current)
        // resolved_verification_probe_count
        != current_dimension
    ):
        return None
    if relation_kind == "zero":
        if (
            representative_id is not None
            or candidate_representative_values is not None
            or verification_representative_values is not None
        ):
            return None
        candidate_representative = None
        verification_representative = None
        factor = None
    else:
        if (
            type(representative_id) is not int
            or representative_id < 0
            or representative_id >= current_id
        ):
            return None
        candidate_representative = _validated_decimal_observations(
            candidate_representative_values
        )
        verification_representative = _validated_decimal_observations(
            verification_representative_values
        )
        if (
            candidate_representative is None
            or verification_representative is None
            or len(candidate_representative) != len(candidate_current)
            or len(verification_representative) != len(verification_current)
        ):
            return None
        factor = (
            (1.0, 0.0)
            if relation_kind == "equal"
            else (-1.0, 0.0)
        )
    absolute_decimal = Decimal.from_float(absolute)
    relative_decimal = Decimal.from_float(relative)
    candidate_residuals = _numerical_relation_residuals(
        relation_kind,
        candidate_current,
        candidate_representative,
        relative_tolerance=relative_decimal,
        absolute_tolerance=absolute_decimal,
        precision_digits=precision_digits,
    )
    verification_residuals = _numerical_relation_residuals(
        relation_kind,
        verification_current,
        verification_representative,
        relative_tolerance=relative_decimal,
        absolute_tolerance=absolute_decimal,
        precision_digits=precision_digits,
    )
    if candidate_residuals is None or verification_residuals is None:
        return None
    if not _numerical_relation_residuals_pass(
        candidate_residuals,
        precision_digits=precision_digits,
    ) or not _numerical_relation_residuals_pass(
        verification_residuals,
        precision_digits=precision_digits,
    ):
        return None
    candidate_digest = _numerical_observations_sha256(
        relation_kind=relation_kind,
        current_values=candidate_current,
        representative_values=candidate_representative,
    )
    verification_digest = _numerical_observations_sha256(
        relation_kind=relation_kind,
        current_values=verification_current,
        representative_values=verification_representative,
    )
    probe_contract = {
        "algorithm": NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM,
        "source_semantics_sha256": source_semantics_sha256,
        "runtime_schema_sha256": runtime_schema_sha256,
        "source_dag_sha256": source_dag_sha256,
        "candidate_capture_sha256": candidate_capture_sha256,
        "verification_capture_sha256": verification_capture_sha256,
        "candidate_observation_batch_sha256": (
            candidate_observation_batch_sha256
        ),
        "verification_observation_batch_sha256": (
            verification_observation_batch_sha256
        ),
        "current_id": current_id,
        "representative_id": representative_id,
        "relation_kind": relation_kind,
        "precision_digits": precision_digits,
        "seed": seed,
        "candidate_domain": "candidate-current-probes-v1",
        "verification_domain": "independent-verification-current-probes-v1",
        "relative_tolerance_binary64": relative.hex(),
        "absolute_tolerance_binary64": absolute.hex(),
        "candidate_probe_count": resolved_candidate_probe_count,
        "verification_probe_count": resolved_verification_probe_count,
        "current_dimension": current_dimension,
        "candidate_observations_sha256": candidate_digest,
        "verification_observations_sha256": verification_digest,
    }
    probe_contract_digest = _canonical_payload_sha256(probe_contract)
    proof_payload = {
        **probe_contract,
        "proof_kind": "authenticated-numerical",
        "factor_binary64": (
            None
            if factor is None
            else [factor[0].hex(), factor[1].hex()]
        ),
        "candidate_maximum_absolute_residual": _canonical_decimal_string(
            candidate_residuals[0]
        ),
        "candidate_maximum_relative_residual": _canonical_decimal_string(
            candidate_residuals[1]
        ),
        "candidate_maximum_tolerance_ratio": _canonical_decimal_string(
            candidate_residuals[2]
        ),
        "verification_maximum_absolute_residual": _canonical_decimal_string(
            verification_residuals[0]
        ),
        "verification_maximum_relative_residual": _canonical_decimal_string(
            verification_residuals[1]
        ),
        "verification_maximum_tolerance_ratio": _canonical_decimal_string(
            verification_residuals[2]
        ),
        "probe_contract_sha256": probe_contract_digest,
    }
    return NumericalCurrentRelationCertificate(
        current_id=current_id,
        representative_id=representative_id,
        relation_kind=relation_kind,
        factor=factor,
        source_semantics_sha256=source_semantics_sha256,
        runtime_schema_sha256=runtime_schema_sha256,
        source_dag_sha256=source_dag_sha256,
        candidate_capture_sha256=candidate_capture_sha256,
        verification_capture_sha256=verification_capture_sha256,
        candidate_observation_batch_sha256=(
            candidate_observation_batch_sha256
        ),
        verification_observation_batch_sha256=(
            verification_observation_batch_sha256
        ),
        precision_digits=precision_digits,
        seed=seed,
        relative_tolerance=relative,
        absolute_tolerance=absolute,
        candidate_probe_count=resolved_candidate_probe_count,
        verification_probe_count=resolved_verification_probe_count,
        current_dimension=current_dimension,
        candidate_maximum_absolute_residual=candidate_residuals[0],
        candidate_maximum_relative_residual=candidate_residuals[1],
        candidate_maximum_tolerance_ratio=candidate_residuals[2],
        verification_maximum_absolute_residual=verification_residuals[0],
        verification_maximum_relative_residual=verification_residuals[1],
        verification_maximum_tolerance_ratio=verification_residuals[2],
        candidate_observations_sha256=candidate_digest,
        verification_observations_sha256=verification_digest,
        probe_contract_sha256=probe_contract_digest,
        proof_sha256=_canonical_payload_sha256(proof_payload),
    )


def verify_numerical_current_relation_certificate(
    certificate: NumericalCurrentRelationCertificate,
    *,
    source_semantics_sha256: str,
) -> bool:
    """Validate canonical certificate integrity without rediscovery."""

    if (
        not isinstance(certificate, NumericalCurrentRelationCertificate)
        or certificate.algorithm
        != NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM
        or certificate.proof_kind != "authenticated-numerical"
        or certificate.source_semantics_sha256 != source_semantics_sha256
        or not _is_sha256(source_semantics_sha256)
    ):
        return False
    try:
        factor = _decode_numerical_relation_factor(
            certificate.relation_kind,
            certificate.representative_id,
            (
                None
                if certificate.factor is None
                else [
                    certificate.factor[0].hex(),
                    certificate.factor[1].hex(),
                ]
            ),
        )
    except (TypeError, ValueError):
        return False
    if factor != certificate.factor:
        return False
    if (
        type(certificate.current_id) is not int
        or certificate.current_id < 0
        or type(certificate.precision_digits) is not int
        or certificate.precision_digits < 80
        or type(certificate.seed) is not int
        or certificate.seed < 0
        or type(certificate.candidate_probe_count) is not int
        or certificate.candidate_probe_count < 2
        or type(certificate.verification_probe_count) is not int
        or certificate.verification_probe_count < 2
        or type(certificate.current_dimension) is not int
        or certificate.current_dimension < 1
        or not isfinite(certificate.relative_tolerance)
        or not isfinite(certificate.absolute_tolerance)
        or certificate.relative_tolerance < 0.0
        or certificate.absolute_tolerance < 0.0
        or _is_negative_zero(certificate.relative_tolerance)
        or _is_negative_zero(certificate.absolute_tolerance)
        or (
            certificate.relative_tolerance == 0.0
            and certificate.absolute_tolerance == 0.0
        )
        or any(
            not _is_sha256(value)
            for value in (
                certificate.runtime_schema_sha256,
                certificate.source_dag_sha256,
                certificate.candidate_capture_sha256,
                certificate.verification_capture_sha256,
                certificate.candidate_observation_batch_sha256,
                certificate.verification_observation_batch_sha256,
                certificate.candidate_observations_sha256,
                certificate.verification_observations_sha256,
                certificate.probe_contract_sha256,
                certificate.proof_sha256,
            )
        )
        or any(
            not value.is_finite() or value < 0
            for value in (
                certificate.candidate_maximum_absolute_residual,
                certificate.candidate_maximum_relative_residual,
                certificate.candidate_maximum_tolerance_ratio,
                certificate.verification_maximum_absolute_residual,
                certificate.verification_maximum_relative_residual,
                certificate.verification_maximum_tolerance_ratio,
            )
        )
        or certificate.candidate_maximum_tolerance_ratio > 1
        or certificate.verification_maximum_tolerance_ratio > 1
        or certificate.candidate_capture_sha256
        == certificate.verification_capture_sha256
        or certificate.candidate_observation_batch_sha256
        == certificate.verification_observation_batch_sha256
    ):
        return False
    probe_contract = {
        "algorithm": certificate.algorithm,
        "source_semantics_sha256": certificate.source_semantics_sha256,
        "runtime_schema_sha256": certificate.runtime_schema_sha256,
        "source_dag_sha256": certificate.source_dag_sha256,
        "candidate_capture_sha256": certificate.candidate_capture_sha256,
        "verification_capture_sha256": (
            certificate.verification_capture_sha256
        ),
        "candidate_observation_batch_sha256": (
            certificate.candidate_observation_batch_sha256
        ),
        "verification_observation_batch_sha256": (
            certificate.verification_observation_batch_sha256
        ),
        "current_id": certificate.current_id,
        "representative_id": certificate.representative_id,
        "relation_kind": certificate.relation_kind,
        "precision_digits": certificate.precision_digits,
        "seed": certificate.seed,
        "candidate_domain": "candidate-current-probes-v1",
        "verification_domain": "independent-verification-current-probes-v1",
        "relative_tolerance_binary64": (
            certificate.relative_tolerance.hex()
        ),
        "absolute_tolerance_binary64": (
            certificate.absolute_tolerance.hex()
        ),
        "candidate_probe_count": certificate.candidate_probe_count,
        "verification_probe_count": certificate.verification_probe_count,
        "current_dimension": certificate.current_dimension,
        "candidate_observations_sha256": (
            certificate.candidate_observations_sha256
        ),
        "verification_observations_sha256": (
            certificate.verification_observations_sha256
        ),
    }
    if (
        _canonical_payload_sha256(probe_contract)
        != certificate.probe_contract_sha256
    ):
        return False
    proof_payload = {
        **probe_contract,
        "proof_kind": certificate.proof_kind,
        "factor_binary64": (
            None
            if certificate.factor is None
            else [
                certificate.factor[0].hex(),
                certificate.factor[1].hex(),
            ]
        ),
        "candidate_maximum_absolute_residual": _canonical_decimal_string(
            certificate.candidate_maximum_absolute_residual
        ),
        "candidate_maximum_relative_residual": _canonical_decimal_string(
            certificate.candidate_maximum_relative_residual
        ),
        "candidate_maximum_tolerance_ratio": _canonical_decimal_string(
            certificate.candidate_maximum_tolerance_ratio
        ),
        "verification_maximum_absolute_residual": _canonical_decimal_string(
            certificate.verification_maximum_absolute_residual
        ),
        "verification_maximum_relative_residual": _canonical_decimal_string(
            certificate.verification_maximum_relative_residual
        ),
        "verification_maximum_tolerance_ratio": _canonical_decimal_string(
            certificate.verification_maximum_tolerance_ratio
        ),
        "probe_contract_sha256": certificate.probe_contract_sha256,
    }
    return _canonical_payload_sha256(proof_payload) == certificate.proof_sha256


def generic_dag_numerical_source_semantics_sha256(
    dag: GenericDAG,
    *,
    execution_mode: Literal["compiled", "eager"],
) -> str:
    """Bind numerical observations to an execution-scoped source DAG.

    Evaluation-group IDs and factors are derived optimizations rather than
    source semantics.  Removing them makes the digest stable across baseline
    exact sharing and authenticated numerical sharing while retaining every
    current, interaction, selector, colour, helicity, and model-facing field.
    """

    if execution_mode not in {"compiled", "eager"}:
        raise ValueError(
            "generic-DAG numerical relations require compiled or eager mode"
        )
    source_dag = _replace_dag_interactions_preserving_materialization(
        dag,
        tuple(
            replace(
                interaction,
                evaluation_group_id=None,
                evaluation_factor=(1.0, 0.0),
            )
            for interaction in dag.interactions
        ),
    )
    return _canonical_payload_sha256(
        {
            "abi": GENERIC_DAG_NUMERICAL_SOURCE_SEMANTICS_ABI,
            "execution_mode": execution_mode,
            "color_accuracy": str(dag.process.color_accuracy),
            "dag": source_dag.to_json_dict(),
        }
    )


def generic_dag_numerical_source_dag_sha256(dag: GenericDAG) -> str:
    """Return the exact pre-application DAG digest bound into evidence."""

    return _canonical_payload_sha256(dag.to_json_dict())


def generic_dag_numerical_runtime_schema_sha256(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str | None = None,
) -> str:
    """Return the model- and process-bound runtime schema digest."""

    from .runtime_schema import build_runtime_expression_schema

    return build_runtime_expression_schema(
        dag,
        model,
        process_id=process_id or dag.process.key,
    ).sha256


def generic_dag_numerical_model_parameter_schema_sha256(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str | None = None,
) -> str:
    """Bind numerical probes to the ordered runtime-parameter schema."""

    from .runtime_schema import build_runtime_expression_schema

    runtime_schema = build_runtime_expression_schema(
        dag,
        model,
        process_id=process_id or dag.process.key,
    ).to_mapping()
    return _canonical_payload_sha256(
        runtime_schema.get("model_parameters", ())
    )


def generic_dag_numerical_capture_output_partition_sha256(
    runtime_schema: Mapping[str, object],
) -> str:
    """Authenticate current-ID partitions from source runtime-slot geometry."""

    raw_stages = runtime_schema.get("stages")
    raw_value_storage = runtime_schema.get("value_storage")
    if not isinstance(raw_stages, list) or not isinstance(
        raw_value_storage,
        Mapping,
    ):
        raise ValueError(
            "numerical current partition runtime schema is invalid"
        )
    raw_value_slots = raw_value_storage.get("value_slots")
    if not isinstance(raw_value_slots, list):
        raise ValueError(
            "numerical current partition value slots are invalid"
        )
    value_slots: dict[int, Mapping[str, object]] = {}
    for raw_slot in raw_value_slots:
        if not isinstance(raw_slot, Mapping):
            raise ValueError(
                "numerical current partition value slot is invalid"
            )
        value_slot_id = raw_slot.get("value_slot_id")
        if (
            type(value_slot_id) is not int
            or value_slot_id < 0
            or value_slot_id in value_slots
        ):
            raise ValueError(
                "numerical current partition value-slot identity drifted"
            )
        value_slots[value_slot_id] = raw_slot

    stage_contracts: list[dict[str, object]] = []
    previous_stage_index: int | None = None
    seen_output_slot_ids: set[int] = set()
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, Mapping):
            raise ValueError(
                "numerical current partition stage is invalid"
            )
        stage_index = raw_stage.get("stage_index")
        stage_kind = raw_stage.get("stage_kind")
        subset_size = raw_stage.get("subset_size")
        raw_output_slot_ids = raw_stage.get("output_value_slot_ids")
        raw_output_current_ids = raw_stage.get("output_current_ids")
        if (
            type(stage_index) is not int
            or (
                previous_stage_index is not None
                and stage_index <= previous_stage_index
            )
            or not isinstance(stage_kind, str)
            or type(subset_size) is not int
            or not isinstance(raw_output_slot_ids, list)
            or not raw_output_slot_ids
            or not isinstance(raw_output_current_ids, list)
        ):
            raise ValueError(
                "numerical current partition stage identity drifted"
            )

        records: list[dict[str, object]] = []
        partition_slots: list[dict[str, object]] = []
        partition_start = 0
        output_start = 0
        current_id: int | None = None
        seen_current_ids: set[int] = set()

        def partition_record(
            *,
            bound_current_id: int | None,
            bound_slots: Sequence[Mapping[str, object]],
            bound_start: int,
            stop: int,
        ) -> dict[str, object]:
            if (
                bound_current_id is None
                or not bound_slots
                or bound_start >= stop
            ):
                raise ValueError(
                    "numerical current partition contains an empty range"
                )
            return {
                "current_id": bound_current_id,
                "output_start": bound_start,
                "output_stop": stop,
                "slots": list(bound_slots),
            }

        for raw_value_slot_id in raw_output_slot_ids:
            if (
                type(raw_value_slot_id) is not int
                or raw_value_slot_id in seen_output_slot_ids
                or raw_value_slot_id not in value_slots
            ):
                raise ValueError(
                    "numerical current partition output-slot identity drifted"
                )
            slot = value_slots[raw_value_slot_id]
            slot_current_id = slot.get("current_id")
            component_start = slot.get("component_start")
            component_stop = slot.get("component_stop")
            variant = slot.get("variant")
            if (
                type(slot_current_id) is not int
                or slot_current_id < 0
                or type(component_start) is not int
                or type(component_stop) is not int
                or component_start < 0
                or component_stop <= component_start
                or not isinstance(variant, str)
                or slot.get("is_source") is not False
            ):
                raise ValueError(
                    "numerical current partition output slot is invalid"
                )
            if current_id is None:
                current_id = slot_current_id
                seen_current_ids.add(current_id)
            elif slot_current_id != current_id:
                records.append(
                    partition_record(
                        bound_current_id=current_id,
                        bound_slots=partition_slots,
                        bound_start=partition_start,
                        stop=output_start,
                    )
                )
                if (
                    slot_current_id in seen_current_ids
                    or slot_current_id <= current_id
                ):
                    raise ValueError(
                        "numerical current partition output current order "
                        "drifted"
                    )
                current_id = slot_current_id
                seen_current_ids.add(current_id)
                partition_start = output_start
                partition_slots = []
            output_stop = output_start + component_stop - component_start
            partition_slots.append(
                {
                    "value_slot_id": raw_value_slot_id,
                    "variant": variant,
                    "component_start": component_start,
                    "component_stop": component_stop,
                    "output_start": output_start,
                    "output_stop": output_stop,
                }
            )
            output_start = output_stop
            seen_output_slot_ids.add(raw_value_slot_id)
        records.append(
            partition_record(
                bound_current_id=current_id,
                bound_slots=partition_slots,
                bound_start=partition_start,
                stop=output_start,
            )
        )
        if raw_output_current_ids != [
            record["current_id"] for record in records
        ]:
            raise ValueError(
                "numerical current partition stage current identity drifted"
            )
        stage_contracts.append(
            {
                "stage_index": stage_index,
                "stage_kind": stage_kind,
                "subset_size": subset_size,
                "output_length": output_start,
                "partitions": records,
            }
        )
        previous_stage_index = stage_index
    if not stage_contracts:
        raise ValueError(
            "numerical current partition runtime schema has no stages"
        )
    return _canonical_payload_sha256(
        {
            "abi": NUMERICAL_CURRENT_CAPTURE_OUTPUT_PARTITION_ABI,
            "strategy": "one-contiguous-output-partition-per-current-id",
            "stages": stage_contracts,
        }
    )


def discover_generic_dag_numerical_current_relations(
    dag: GenericDAG,
    model: Model,
    *,
    candidate_observations: Mapping[
        int,
        Sequence[tuple[Decimal, Decimal]],
    ],
    verification_observations: Mapping[
        int,
        Sequence[tuple[Decimal, Decimal]],
    ],
    candidate_point_sha256s: Sequence[str],
    verification_point_sha256s: Sequence[str],
    candidate_kinematic_sha256s: Sequence[str],
    verification_kinematic_sha256s: Sequence[str],
    candidate_parameter_context_sha256s: Sequence[str],
    verification_parameter_context_sha256s: Sequence[str],
    runtime_schema_sha256: str,
    source_dag_sha256: str,
    evaluator_output_partition_abi: str,
    evaluator_output_partition_sha256: str,
    candidate_capture_sha256: str,
    verification_capture_sha256: str,
    process_id: str | None = None,
    execution_mode: Literal["compiled", "eager"],
    precision_digits: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> NumericalCurrentObservationDiscoveryResult:
    """Discover equal, opposite, and zero currents from complete warm-up data.

    Every generated current is compared exhaustively within its exact runtime
    contract.  Candidate and verification point domains must be disjoint.
    Exact structural relations are retained in preference to numerical
    evidence and are not re-certified as proof-less mappings.
    """

    source_digest = generic_dag_numerical_source_semantics_sha256(
        dag,
        execution_mode=execution_mode,
    )
    from .runtime_schema import build_runtime_expression_schema

    actual_runtime_schema = build_runtime_expression_schema(
        dag,
        model,
        process_id=process_id or dag.process.key,
    )
    actual_runtime_schema_mapping = actual_runtime_schema.to_mapping()
    actual_runtime_schema_digest = actual_runtime_schema.sha256
    actual_model_parameter_schema_digest = _canonical_payload_sha256(
        actual_runtime_schema_mapping.get("model_parameters", ())
    )
    actual_output_partition_digest = (
        generic_dag_numerical_capture_output_partition_sha256(
            actual_runtime_schema_mapping
        )
    )
    actual_source_dag_digest = generic_dag_numerical_source_dag_sha256(dag)
    candidate_points = tuple(candidate_point_sha256s)
    verification_points = tuple(verification_point_sha256s)
    candidate_kinematics = tuple(candidate_kinematic_sha256s)
    verification_kinematics = tuple(verification_kinematic_sha256s)
    candidate_parameters = tuple(
        candidate_parameter_context_sha256s
    )
    verification_parameters = tuple(
        verification_parameter_context_sha256s
    )
    if (
        type(precision_digits) is not int
        or precision_digits < 80
        or type(seed) is not int
        or seed < 0
        or len(candidate_points) < 2
        or len(verification_points) < 2
        or any(not _is_sha256(value) for value in candidate_points)
        or any(not _is_sha256(value) for value in verification_points)
        or len(set(candidate_points)) != len(candidate_points)
        or len(set(verification_points)) != len(verification_points)
        or not set(candidate_points).isdisjoint(verification_points)
        or len(candidate_kinematics) != len(candidate_points)
        or len(verification_kinematics) != len(verification_points)
        or len(candidate_parameters) != len(candidate_points)
        or len(verification_parameters) != len(verification_points)
        or any(not _is_sha256(value) for value in candidate_kinematics)
        or any(not _is_sha256(value) for value in verification_kinematics)
        or any(not _is_sha256(value) for value in candidate_parameters)
        or any(not _is_sha256(value) for value in verification_parameters)
        or len(set(candidate_kinematics)) != len(candidate_kinematics)
        or len(set(verification_kinematics))
        != len(verification_kinematics)
        or not set(candidate_kinematics).isdisjoint(
            verification_kinematics
        )
        or not set(candidate_parameters).isdisjoint(
            verification_parameters
        )
        or any(
            not _is_sha256(value)
            for value in (
                runtime_schema_sha256,
                source_dag_sha256,
                evaluator_output_partition_sha256,
                candidate_capture_sha256,
                verification_capture_sha256,
            )
        )
        or evaluator_output_partition_abi
        != NUMERICAL_CURRENT_CAPTURE_OUTPUT_PARTITION_ABI
        or evaluator_output_partition_sha256
        != actual_output_partition_digest
        or candidate_capture_sha256 == verification_capture_sha256
        or runtime_schema_sha256 != actual_runtime_schema_digest
        or source_dag_sha256 != actual_source_dag_digest
    ):
        raise ValueError(
            "numerical current discovery point contract is invalid"
        )
    if (
        not isinstance(relative_tolerance, int | float)
        or isinstance(relative_tolerance, bool)
        or not isinstance(absolute_tolerance, int | float)
        or isinstance(absolute_tolerance, bool)
    ):
        raise ValueError(
            "numerical current discovery tolerances must be real numbers"
        )
    relative = float(relative_tolerance)
    absolute = float(absolute_tolerance)
    if (
        not isfinite(relative)
        or not isfinite(absolute)
        or relative < 0.0
        or absolute < 0.0
        or _is_negative_zero(relative)
        or _is_negative_zero(absolute)
        or (relative == 0.0 and absolute == 0.0)
    ):
        raise ValueError("numerical current discovery tolerances are invalid")

    current_ids = tuple(range(len(dag.currents)))
    if tuple(current.id for current in dag.currents) != current_ids:
        raise ValueError(
            "numerical current discovery requires canonical current IDs"
        )
    if set(candidate_observations) != set(current_ids) or set(
        verification_observations
    ) != set(current_ids):
        raise ValueError(
            "numerical current discovery requires observations for every current"
        )
    candidate_values: dict[
        int,
        tuple[tuple[Decimal, Decimal], ...],
    ] = {}
    verification_values: dict[
        int,
        tuple[tuple[Decimal, Decimal], ...],
    ] = {}
    for current in dag.currents:
        candidate = _validated_decimal_observations(
            candidate_observations[current.id]
        )
        verification = _validated_decimal_observations(
            verification_observations[current.id]
        )
        expected_candidate_count = (
            len(candidate_points) * current.dimension
        )
        expected_verification_count = (
            len(verification_points) * current.dimension
        )
        if (
            candidate is None
            or verification is None
            or len(candidate) != expected_candidate_count
            or len(verification) != expected_verification_count
        ):
            raise ValueError(
                f"numerical observations for current {current.id} do not "
                "cover every point and component"
            )
        candidate_values[current.id] = candidate
        verification_values[current.id] = verification

    candidate_batch_digest = _numerical_current_observation_batch_sha256(
        candidate_values,
        point_sha256s=candidate_points,
    )
    verification_batch_digest = _numerical_current_observation_batch_sha256(
        verification_values,
        point_sha256s=verification_points,
    )
    expected_candidate_capture_digest = _canonical_payload_sha256(
        {
            "abi": NUMERICAL_CURRENT_CAPTURE_ABI,
            "precision_digits": precision_digits,
            "point_sha256s": list(candidate_points),
            "kinematic_sha256s": list(candidate_kinematics),
            "parameter_context_sha256s": list(candidate_parameters),
            "runtime_schema_sha256": runtime_schema_sha256,
            "model_parameter_schema_sha256": (
                actual_model_parameter_schema_digest
            ),
            "source_dag_sha256": source_dag_sha256,
            "evaluator_output_partition_abi": (
                evaluator_output_partition_abi
            ),
            "evaluator_output_partition_sha256": (
                evaluator_output_partition_sha256
            ),
            "observation_batch_sha256": candidate_batch_digest,
        }
    )
    expected_verification_capture_digest = _canonical_payload_sha256(
        {
            "abi": NUMERICAL_CURRENT_CAPTURE_ABI,
            "precision_digits": precision_digits,
            "point_sha256s": list(verification_points),
            "kinematic_sha256s": list(verification_kinematics),
            "parameter_context_sha256s": list(verification_parameters),
            "runtime_schema_sha256": runtime_schema_sha256,
            "model_parameter_schema_sha256": (
                actual_model_parameter_schema_digest
            ),
            "source_dag_sha256": source_dag_sha256,
            "evaluator_output_partition_abi": (
                evaluator_output_partition_abi
            ),
            "evaluator_output_partition_sha256": (
                evaluator_output_partition_sha256
            ),
            "observation_batch_sha256": verification_batch_digest,
        }
    )
    if (
        candidate_capture_sha256 != expected_candidate_capture_digest
        or verification_capture_sha256
        != expected_verification_capture_digest
    ):
        raise ValueError(
            "numerical current capture provenance does not authenticate its "
            "complete observation batches"
        )
    structural = _derive_current_value_equivalences(dag, model)
    contracts = tuple(
        _current_evaluation_contract(current) for current in dag.currents
    )
    members_by_contract: dict[_CurrentContract, list[int]] = defaultdict(list)
    for current_id, contract in enumerate(contracts):
        members_by_contract[contract].append(current_id)
    candidate_indexes = {
        contract: _build_numerical_observation_candidate_index(
            member_ids,
            candidate_values,
        )
        for contract, member_ids in members_by_contract.items()
    }
    prior_by_contract: dict[_CurrentContract, list[int]] = defaultdict(list)
    source_ids = set(dag.sources)
    relative_decimal = Decimal.from_float(relative)
    absolute_decimal = Decimal.from_float(absolute)
    certificates: list[NumericalCurrentRelationCertificate] = []
    rejected_candidates: list[dict[str, object]] = []
    nearest_rejected: tuple[
        Decimal,
        Decimal,
        Decimal,
        int,
        int,
        str,
    ] | None = None
    structurally_proven_count = 0
    tested_hypothesis_count = 0
    theoretical_pair_hypothesis_count = 0
    screened_pair_hypothesis_count = 0
    numerical_candidate_count = 0
    verification_rejected_count = 0
    rejected_hypothesis_count = 0
    rejected_decision_sha256 = _canonical_payload_sha256(
        {
            "abi": "pyamplicol-generic-dag-rejected-decision-chain-v1",
            "candidate_capture_sha256": candidate_capture_sha256,
            "verification_capture_sha256": verification_capture_sha256,
        }
    )

    def record_rejected(
        *,
        current_id: int,
        representative_id: int | None,
        relation_kind: str,
        residuals: tuple[
            Decimal,
            Decimal,
            Decimal,
            tuple[tuple[Decimal, Decimal], ...],
        ],
        reason: str,
    ) -> None:
        nonlocal nearest_rejected
        nonlocal rejected_decision_sha256
        nonlocal rejected_hypothesis_count
        rejected_hypothesis_count += 1
        candidate = (
            residuals[2],
            residuals[0],
            residuals[1],
            current_id,
            -1 if representative_id is None else representative_id,
            relation_kind,
        )
        if nearest_rejected is None or candidate < nearest_rejected:
            nearest_rejected = candidate
        row = {
            "current_id": current_id,
            "representative_id": representative_id,
            "relation_kind": relation_kind,
            "reason": reason,
            "maximum_absolute_residual": (
                _canonical_decimal_string(residuals[0])
            ),
            "maximum_relative_residual": (
                _canonical_decimal_string(residuals[1])
            ),
            "maximum_tolerance_ratio": (
                _canonical_decimal_string(residuals[2])
            ),
        }
        rejected_decision_sha256 = _advance_rejection_digest(
            rejected_decision_sha256,
            row,
        )
        if len(rejected_candidates) < _MAX_REJECTED_DIAGNOSTICS:
            rejected_candidates.append(row)

    for current in dag.currents:
        contract = contracts[current.id]
        prior_ids = prior_by_contract.setdefault(contract, [])
        if current.id in source_ids or current.is_source:
            prior_ids.append(current.id)
            continue
        baseline = structural[current.id]
        if (
            baseline.representative_id != current.id
            or baseline.factor != (1.0, 0.0)
        ):
            structurally_proven_count += 1
            prior_ids.append(current.id)
            continue

        equal_representatives = set(
            _numerical_observation_tolerance_window_ids(
                candidate_indexes[contract],
                candidate_values[current.id],
                relation_kind="equal",
                current_id=current.id,
                relative_tolerance=relative_decimal,
                absolute_tolerance=absolute_decimal,
                precision_digits=precision_digits,
            )
        )
        opposite_representatives = set(
            _numerical_observation_tolerance_window_ids(
                candidate_indexes[contract],
                candidate_values[current.id],
                relation_kind="opposite",
                current_id=current.id,
                relative_tolerance=relative_decimal,
                absolute_tolerance=absolute_decimal,
                precision_digits=precision_digits,
            )
        )
        theoretical_pair_hypothesis_count += 2 * len(prior_ids)
        screened_pair_hypothesis_count += (
            len(equal_representatives) + len(opposite_representatives)
        )
        hypotheses: list[
            tuple[
                Literal["equal", "opposite", "zero"],
                int | None,
                tuple[tuple[Decimal, Decimal], ...] | None,
                tuple[tuple[Decimal, Decimal], ...] | None,
            ]
        ] = [
            ("zero", None, None, None),
            *[
                (
                    relation_kind,
                    representative_id,
                    candidate_values[representative_id],
                    verification_values[representative_id],
                )
                for representative_id in sorted(
                    equal_representatives | opposite_representatives
                )
                for relation_kind in (
                    *(
                        ("equal",)
                        if representative_id in equal_representatives
                        else ()
                    ),
                    *(
                        ("opposite",)
                        if representative_id in opposite_representatives
                        else ()
                    ),
                )
            ],
        ]
        accepted: NumericalCurrentRelationCertificate | None = None
        for (
            relation_kind,
            representative_id,
            candidate_representative,
            verification_representative,
        ) in hypotheses:
            tested_hypothesis_count += 1
            candidate_residuals = _numerical_relation_residuals(
                relation_kind,
                candidate_values[current.id],
                candidate_representative,
                relative_tolerance=relative_decimal,
                absolute_tolerance=absolute_decimal,
                precision_digits=precision_digits,
            )
            if candidate_residuals is None:
                raise ValueError(
                    "numerical current discovery produced a malformed hypothesis"
                )
            if not _numerical_relation_residuals_pass(
                candidate_residuals,
                precision_digits=precision_digits,
            ):
                record_rejected(
                    current_id=current.id,
                    representative_id=representative_id,
                    relation_kind=relation_kind,
                    residuals=candidate_residuals,
                    reason="candidate-observations-not-equal",
                )
                continue
            numerical_candidate_count += 1
            accepted = certify_numerical_current_observations(
                current_id=current.id,
                representative_id=representative_id,
                relation_kind=relation_kind,
                source_semantics_sha256=source_digest,
                runtime_schema_sha256=runtime_schema_sha256,
                source_dag_sha256=source_dag_sha256,
                candidate_capture_sha256=candidate_capture_sha256,
                verification_capture_sha256=(
                    verification_capture_sha256
                ),
                candidate_observation_batch_sha256=(
                    candidate_batch_digest
                ),
                verification_observation_batch_sha256=(
                    verification_batch_digest
                ),
                candidate_current_values=candidate_values[current.id],
                candidate_representative_values=candidate_representative,
                verification_current_values=verification_values[current.id],
                verification_representative_values=(
                    verification_representative
                ),
                precision_digits=precision_digits,
                seed=seed,
                relative_tolerance=relative,
                absolute_tolerance=absolute,
                candidate_probe_count=len(candidate_points),
                verification_probe_count=len(verification_points),
            )
            if accepted is not None:
                certificates.append(accepted)
                break
            verification_rejected_count += 1
            verification_residuals = _numerical_relation_residuals(
                relation_kind,
                verification_values[current.id],
                verification_representative,
                relative_tolerance=relative_decimal,
                absolute_tolerance=absolute_decimal,
                precision_digits=precision_digits,
            )
            if verification_residuals is None:
                raise ValueError(
                    "numerical current verification produced a malformed "
                    "hypothesis"
                )
            record_rejected(
                current_id=current.id,
                representative_id=representative_id,
                relation_kind=relation_kind,
                residuals=verification_residuals,
                reason="independent-verification-rejected-candidate",
            )
        prior_ids.append(current.id)

    nearest_payload: dict[str, object] | None = None
    if nearest_rejected is not None:
        (
            tolerance_ratio,
            absolute_residual,
            relative_residual,
            current_id,
            representative_key,
            nearest_relation_kind,
        ) = nearest_rejected
        nearest_payload = {
            "current_id": current_id,
            "representative_id": (
                None if representative_key < 0 else representative_key
            ),
            "relation_kind": nearest_relation_kind,
            "maximum_absolute_residual": (
                _canonical_decimal_string(absolute_residual)
            ),
            "maximum_relative_residual": (
                _canonical_decimal_string(relative_residual)
            ),
            "maximum_tolerance_ratio": (
                _canonical_decimal_string(tolerance_ratio)
            ),
        }
    certificate_tuple = tuple(certificates)
    if rejected_hypothesis_count != (
        tested_hypothesis_count - len(certificate_tuple)
    ):
        raise AssertionError(
            "generic numerical rejection census is internally inconsistent"
        )
    state = (
        "certified_numerical_relations"
        if certificate_tuple
        else "no_certified_numerical_relation"
    )
    report = NumericalCurrentObservationDiscoveryReport(
        execution_mode=execution_mode,
        color_accuracy=str(dag.process.color_accuracy),
        source_semantics_sha256=source_digest,
        runtime_schema_sha256=runtime_schema_sha256,
        source_dag_sha256=source_dag_sha256,
        precision_digits=precision_digits,
        seed=seed,
        relative_tolerance=relative,
        absolute_tolerance=absolute,
        candidate_point_sha256s=candidate_points,
        verification_point_sha256s=verification_points,
        candidate_kinematic_sha256s=candidate_kinematics,
        verification_kinematic_sha256s=verification_kinematics,
        candidate_parameter_context_sha256s=candidate_parameters,
        verification_parameter_context_sha256s=verification_parameters,
        candidate_capture_sha256=candidate_capture_sha256,
        verification_capture_sha256=verification_capture_sha256,
        candidate_observation_batch_sha256=candidate_batch_digest,
        verification_observation_batch_sha256=verification_batch_digest,
        state=state,
        inspected_current_count=len(dag.currents),
        structurally_proven_current_count=structurally_proven_count,
        tested_hypothesis_count=tested_hypothesis_count,
        theoretical_pair_hypothesis_count=(
            theoretical_pair_hypothesis_count
        ),
        screened_pair_hypothesis_count=screened_pair_hypothesis_count,
        candidate_index_contract_count=len(candidate_indexes),
        exhaustive_fallback_contract_count=(
            len(candidate_indexes) if relative_decimal >= 1 else 0
        ),
        numerical_candidate_count=numerical_candidate_count,
        verification_rejected_count=verification_rejected_count,
        rejected_hypothesis_count=rejected_hypothesis_count,
        rejected_decision_sha256=rejected_decision_sha256,
        certificates=certificate_tuple,
        rejected_candidates=tuple(rejected_candidates),
        nearest_rejected_hypothesis=nearest_payload,
    )
    return NumericalCurrentObservationDiscoveryResult(
        certificates=certificate_tuple,
        report=report,
        structural_equivalences=structural,
    )


def _numerical_current_observation_batch_sha256(
    observations: Mapping[
        int,
        tuple[tuple[Decimal, Decimal], ...],
    ],
    *,
    point_sha256s: tuple[str, ...],
) -> str:
    return _canonical_payload_sha256(
        {
            "point_sha256s": list(point_sha256s),
            "currents": [
                {
                    "current_id": current_id,
                    "values": [
                        [
                            _canonical_decimal_string(real),
                            _canonical_decimal_string(imaginary),
                        ]
                        for real, imaginary in observations[current_id]
                    ],
                }
                for current_id in sorted(observations)
            ],
        }
    )


def apply_numerical_current_relation_certificates(
    dag: GenericDAG,
    model: Model,
    certificates: Iterable[NumericalCurrentRelationCertificate],
    *,
    mode: _DiscoveryMode,
    execution_mode: Literal["compiled", "eager"],
    process_id: str | None = None,
    runtime_schema_sha256: str | None = None,
    source_dag_sha256: str | None = None,
    candidate_capture_sha256: str | None = None,
    verification_capture_sha256: str | None = None,
    _structural_equivalences: (
        tuple[_CurrentValueEquivalence, ...] | None
    ) = None,
) -> NumericalCurrentRelationApplicationResult:
    """Replay and apply one canonical authenticated numerical relation set.

    Application is monotonic with respect to the DAG's existing evaluator
    partition.  Certified zeros suppress only interactions that consume the
    zero current or an exactly proven member of its structural class.
    Equal/opposite mappings merge such an interaction into an already
    compatible concrete evaluator group.  Unaffected interactions are never
    regrouped, and the evaluator count is forbidden to increase.

    Any malformed, stale, inconsistent, non-topological, or inexact binary64
    factor composition fails closed.  A valid certificate for which no safe
    downstream reuse opportunity exists remains recorded but unapplied.
    """

    if mode not in {"diagnostic", "certified-reuse"}:
        raise ValueError(f"unsupported relation discovery mode {mode!r}")
    source_digest = generic_dag_numerical_source_semantics_sha256(
        dag,
        execution_mode=execution_mode,
    )
    ordered = tuple(certificates)
    if ordered:
        resolved_process_id = process_id or dag.process.key
        actual_runtime_schema_digest = (
            generic_dag_numerical_runtime_schema_sha256(
                dag,
                model,
                process_id=resolved_process_id,
            )
        )
        actual_source_dag_digest = generic_dag_numerical_source_dag_sha256(
            dag
        )
        expected_runtime_schema_digest = (
            actual_runtime_schema_digest
            if runtime_schema_sha256 is None
            else runtime_schema_sha256
        )
        expected_source_dag_digest = (
            actual_source_dag_digest
            if source_dag_sha256 is None
            else source_dag_sha256
        )
        if (
            expected_runtime_schema_digest != actual_runtime_schema_digest
            or expected_source_dag_digest != actual_source_dag_digest
            or not _is_sha256(expected_runtime_schema_digest)
            or not _is_sha256(expected_source_dag_digest)
        ):
            raise ValueError(
                "numerical current application model/schema provenance "
                "drifted from discovery"
            )
        expected_candidate_capture_digest = (
            ordered[0].candidate_capture_sha256
            if candidate_capture_sha256 is None
            else candidate_capture_sha256
        )
        expected_verification_capture_digest = (
            ordered[0].verification_capture_sha256
            if verification_capture_sha256 is None
            else verification_capture_sha256
        )
        if (
            not _is_sha256(expected_candidate_capture_digest)
            or not _is_sha256(expected_verification_capture_digest)
            or expected_candidate_capture_digest
            == expected_verification_capture_digest
        ):
            raise ValueError(
                "numerical current application capture provenance is invalid"
            )
    else:
        expected_runtime_schema_digest = runtime_schema_sha256
        expected_source_dag_digest = source_dag_sha256
        expected_candidate_capture_digest = candidate_capture_sha256
        expected_verification_capture_digest = (
            verification_capture_sha256
        )
    if tuple(sorted(ordered, key=lambda item: item.current_id)) != ordered:
        raise ValueError(
            "numerical current certificates must use increasing current IDs"
        )
    if len({certificate.current_id for certificate in ordered}) != len(ordered):
        raise ValueError(
            "numerical current certificates contain duplicate current IDs"
        )

    source_ids = set(dag.sources)
    current_contracts = tuple(
        _current_evaluation_contract(current) for current in dag.currents
    )
    common_probe_contract: tuple[object, ...] | None = None
    authenticated_relations: dict[int, _CurrentValueEquivalence] = {}
    for certificate in ordered:
        if not verify_numerical_current_relation_certificate(
            certificate,
            source_semantics_sha256=source_digest,
        ):
            raise ValueError(
                f"numerical current certificate {certificate.current_id} "
                "does not replay against the source DAG"
            )
        if (
            certificate.runtime_schema_sha256
            != expected_runtime_schema_digest
            or certificate.source_dag_sha256
            != expected_source_dag_digest
            or certificate.candidate_capture_sha256
            != expected_candidate_capture_digest
            or certificate.verification_capture_sha256
            != expected_verification_capture_digest
        ):
            raise ValueError(
                "numerical current certificate evidence provenance drifted "
                "from its application contract"
            )
        current_id = certificate.current_id
        if (
            current_id >= len(dag.currents)
            or current_id in source_ids
            or dag.currents[current_id].is_source
            or certificate.current_dimension
            != dag.currents[current_id].dimension
        ):
            raise ValueError(
                "numerical current relations may target only matching "
                "generated currents"
            )
        probe_contract = (
            certificate.precision_digits,
            certificate.seed,
            certificate.relative_tolerance.hex(),
            certificate.absolute_tolerance.hex(),
            certificate.candidate_probe_count,
            certificate.verification_probe_count,
            certificate.runtime_schema_sha256,
            certificate.source_dag_sha256,
            certificate.candidate_capture_sha256,
            certificate.verification_capture_sha256,
            certificate.candidate_observation_batch_sha256,
            certificate.verification_observation_batch_sha256,
        )
        if common_probe_contract is None:
            common_probe_contract = probe_contract
        elif common_probe_contract != probe_contract:
            raise ValueError(
                "numerical current relation set mixes probe contracts"
            )

        if certificate.relation_kind == "zero":
            compatible_representatives = tuple(
                candidate_id
                for candidate_id in range(current_id)
                if current_contracts[candidate_id]
                == current_contracts[current_id]
            )
            execution_representative_id = (
                compatible_representatives[0]
                if compatible_representatives
                else current_id
            )
            factor = (0.0, 0.0)
        else:
            representative_id = certificate.representative_id
            if (
                representative_id is None
                or representative_id >= current_id
                or current_contracts[representative_id]
                != current_contracts[current_id]
            ):
                raise ValueError(
                    "numerical current representative violates its "
                    "execution contract"
                )
            execution_representative_id = representative_id
            assert certificate.factor is not None
            factor = certificate.factor
        authenticated_relations[current_id] = _CurrentValueEquivalence(
            representative_id=execution_representative_id,
            factor=factor,
        )

    if ordered:
        structural_equivalences = (
            _derive_current_value_equivalences(dag, model)
            if _structural_equivalences is None
            else _structural_equivalences
        )
        if len(structural_equivalences) != len(dag.currents):
            raise ValueError(
                "numerical current structural equivalence cache drifted"
            )
        baseline = _numerical_current_application_baseline(dag, model)
        (
            projected,
            resolved_relations,
            certificate_chains,
            projected_interaction_ids,
        ) = _project_authenticated_numerical_interaction_reuse(
            baseline,
            model,
            authenticated_relations,
            structural_equivalences=structural_equivalences,
        )
    else:
        baseline = projected = dag
        resolved_relations = tuple(
            _CurrentValueEquivalence(current.id, (1.0, 0.0))
            for current in dag.currents
        )
        certificate_chains = tuple(() for _current in dag.currents)
        projected_interaction_ids = {}

    mappings = [
        NumericalCurrentAppliedMapping(
            current_id=certificate.current_id,
            representative_id=certificate.representative_id,
            execution_representative_id=(
                resolved_relations[certificate.current_id].representative_id
            ),
            relation_kind=certificate.relation_kind,
            factor=resolved_relations[certificate.current_id].factor,
            certificate_proof_sha256=certificate.proof_sha256,
            resolved_current_ids=tuple(
                current_id
                for current_id, chain in enumerate(certificate_chains)
                if certificate.current_id in chain
            ),
            projected_interaction_count=len(
                projected_interaction_ids.get(certificate.current_id, ())
            ),
            projected_interaction_ids_sha256=_canonical_payload_sha256(
                {
                    "interaction_ids": list(
                        projected_interaction_ids.get(
                            certificate.current_id,
                            (),
                        )
                    )
                }
            ),
        )
        for certificate in ordered
    ]

    mapping_payloads = [mapping.to_json_dict() for mapping in mappings]
    certificate_set_digest = _canonical_payload_sha256(
        {
            "abi": NUMERICAL_CURRENT_RELATION_SET_ABI,
            "source_semantics_sha256": source_digest,
            "execution_mode": execution_mode,
            "certificates": [
                certificate.to_json_dict() for certificate in ordered
            ],
            "mappings": mapping_payloads,
        }
    )
    projected_relation_count = sum(
        bool(interaction_ids)
        for interaction_ids in projected_interaction_ids.values()
    )
    apply_relations = (
        mode == "certified-reuse" and projected_relation_count > 0
    )
    if not ordered:
        state = "no_certified_numerical_relation"
        replay_status = "no_certified_numerical_relation"
    elif apply_relations:
        state = "authenticated-numerical-applied"
        replay_status = "verified"
    elif mode == "certified-reuse":
        state = "authenticated-numerical-no-reuse-opportunity"
        replay_status = "verified"
    else:
        state = "authenticated-numerical-diagnostic-only"
        replay_status = "verified"
    report = NumericalCurrentRelationApplicationReport(
        requested_mode=mode,
        execution_mode=execution_mode,
        color_accuracy=str(dag.process.color_accuracy),
        source_semantics_sha256=source_digest,
        runtime_schema_sha256=expected_runtime_schema_digest,
        source_dag_sha256=expected_source_dag_digest,
        candidate_capture_sha256=expected_candidate_capture_digest,
        verification_capture_sha256=(
            expected_verification_capture_digest
        ),
        state=state,
        certificate_replay_status=replay_status,
        certificate_set_sha256=certificate_set_digest,
        certificates=ordered,
        mappings=tuple(mappings),
        interaction_evaluation_count_before=(
            baseline.interaction_evaluation_count
        ),
        interaction_evaluation_count_projected=(
            projected.interaction_evaluation_count
        ),
        applied_relation_count=(
            projected_relation_count if apply_relations else 0
        ),
    )
    return NumericalCurrentRelationApplicationResult(
        dag=projected if apply_relations else dag,
        report=report,
    )


def _numerical_current_application_baseline(
    dag: GenericDAG,
    model: Model,
) -> GenericDAG:
    """Retain an existing evaluator partition or establish one once.

    Final generation DAGs already carry their online structural grouping and
    must not be globally regrouped by a numerical pass.  Small API-built DAGs
    with no evaluation metadata at all receive the ordinary exact structural
    baseline before numerical projection.
    """

    has_evaluation_metadata = any(
        interaction.evaluation_group_id is not None
        or interaction.evaluation_factor != (1.0, 0.0)
        for interaction in dag.interactions
    )
    baseline = (
        dag
        if has_evaluation_metadata
        else assign_recursive_current_evaluation_reuse(dag, model)
    )
    _validate_numerical_current_application_baseline(baseline, model)
    return baseline


def _validate_numerical_current_application_baseline(
    dag: GenericDAG,
    model: Model,
) -> None:
    """Authenticate every live member of the existing evaluator partition."""

    structural = _derive_current_value_equivalences(dag, model)
    equivalence_by_kind: dict[int, VertexEvaluationEquivalence] = {}
    key_by_group: dict[int, _EvaluationKey] = {}
    for interaction in dag.interactions:
        if not _complex_weight_is_finite(interaction.evaluation_factor):
            raise ValueError(
                "numerical current application baseline has a non-finite "
                "evaluation factor"
            )
        if interaction.evaluation_factor == (0.0, 0.0):
            continue
        left = structural[interaction.left_id]
        right = structural[interaction.right_id]
        parent_factor = _exact_representable_complex_product(
            left.factor,
            right.factor,
        )
        key, kernel_factor = _numerical_interaction_evaluation_key(
            dag,
            model,
            interaction,
            left_id=left.representative_id,
            right_id=right.representative_id,
            equivalence_by_kind=equivalence_by_kind,
        )
        expected_factor = (
            None
            if parent_factor is None
            else _exact_representable_complex_product(
                kernel_factor,
                parent_factor,
            )
        )
        if expected_factor is None or not _complex_weight_bits_equal(
            _canonical_complex_weight(expected_factor),
            _canonical_complex_weight(interaction.evaluation_factor),
        ):
            raise ValueError(
                "numerical current application baseline evaluation metadata "
                f"is unauthenticated for interaction {interaction.id}"
            )
        group_id = interaction.evaluation_group_id
        if group_id is None:
            continue
        prior = key_by_group.setdefault(group_id, key)
        if prior != key:
            raise ValueError(
                "numerical current application baseline aliases distinct "
                f"evaluation contracts in group {group_id}"
            )


def _resolve_authenticated_numerical_current_relations(
    dag: GenericDAG,
    authenticated_relations: Mapping[int, _CurrentValueEquivalence],
    *,
    structural_equivalences: tuple[_CurrentValueEquivalence, ...],
) -> tuple[
    tuple[_CurrentValueEquivalence, ...],
    tuple[tuple[int, ...], ...],
]:
    """Resolve authenticated chains through exact structural current classes."""

    resolved: list[_CurrentValueEquivalence] = []
    certificate_chains: list[tuple[int, ...]] = []
    for expected_id, current in enumerate(dag.currents):
        if current.id != expected_id:
            raise ValueError(
                "numerical current application requires contiguous current IDs"
            )
        structural = structural_equivalences[current.id]
        if not 0 <= structural.representative_id <= current.id:
            raise ValueError(
                "numerical current structural equivalence is not topological"
            )
        authenticated = authenticated_relations.get(current.id)
        if structural.representative_id != current.id:
            if authenticated is not None:
                raise ValueError(
                    "authenticated numerical relations must target exact "
                    "structural current representatives"
                )
            representative = resolved[structural.representative_id]
            factor = _exact_representable_complex_product(
                structural.factor,
                representative.factor,
            )
            if factor is None:
                raise ValueError(
                    "structural and authenticated numerical current factors "
                    "are not exactly composable"
                )
            resolved.append(
                _CurrentValueEquivalence(
                    representative_id=representative.representative_id,
                    factor=factor,
                )
            )
            certificate_chains.append(
                certificate_chains[structural.representative_id]
            )
            continue
        if structural.factor != (1.0, 0.0):
            raise ValueError(
                "self-represented structural current equivalence is not "
                "canonical"
            )
        if authenticated is None:
            resolved.append(
                _CurrentValueEquivalence(current.id, (1.0, 0.0))
            )
            certificate_chains.append(())
            continue
        if authenticated.factor == (0.0, 0.0):
            resolved.append(authenticated)
            certificate_chains.append((current.id,))
            continue
        if not 0 <= authenticated.representative_id < current.id:
            raise ValueError(
                "authenticated numerical current relation is not topological"
            )
        representative = resolved[authenticated.representative_id]
        factor = _exact_representable_complex_product(
            authenticated.factor,
            representative.factor,
        )
        if factor is None:
            raise ValueError(
                "authenticated numerical current relation factor chain is "
                "not exactly representable"
            )
        resolved.append(
            _CurrentValueEquivalence(
                representative_id=representative.representative_id,
                factor=factor,
            )
        )
        certificate_chains.append(
            (
                current.id,
                *certificate_chains[authenticated.representative_id],
            )
        )
    return tuple(resolved), tuple(certificate_chains)


def _numerical_interaction_evaluation_key(
    dag: GenericDAG,
    model: Model,
    interaction: InteractionNode,
    *,
    left_id: int,
    right_id: int,
    equivalence_by_kind: dict[int, VertexEvaluationEquivalence],
) -> tuple[_EvaluationKey, _ComplexWeight]:
    equivalence = _kernel_equivalence(
        model,
        interaction.vertex_kind,
        equivalence_by_kind,
    )
    canonical_inputs, kernel_factor = _canonical_kernel_evaluation(
        equivalence,
        left_id,
        right_id,
    )
    result = dag.currents[interaction.result_id]
    return (
        (
            equivalence.class_id,
            canonical_inputs,
            int(result.index.particle_id),
            int(result.index.chirality),
            _runtime_coupling_identity(
                model,
                vertex_kind=interaction.vertex_kind,
                vertex_particles=interaction.vertex_particles,
                coupling=interaction.coupling,
            ),
        ),
        kernel_factor,
    )


def _project_authenticated_numerical_interaction_reuse(
    dag: GenericDAG,
    model: Model,
    authenticated_relations: Mapping[int, _CurrentValueEquivalence],
    *,
    structural_equivalences: tuple[_CurrentValueEquivalence, ...],
) -> tuple[
    GenericDAG,
    tuple[_CurrentValueEquivalence, ...],
    tuple[tuple[int, ...], ...],
    dict[int, tuple[int, ...]],
]:
    """Monotonically project authenticated current mappings onto interactions."""

    if tuple(interaction.id for interaction in dag.interactions) != tuple(
        range(len(dag.interactions))
    ):
        raise ValueError(
            "numerical current application requires contiguous interaction IDs"
        )
    resolved, certificate_chains = (
        _resolve_authenticated_numerical_current_relations(
            dag,
            authenticated_relations,
            structural_equivalences=structural_equivalences,
        )
    )
    equivalence_by_kind: dict[int, VertexEvaluationEquivalence] = {}
    targets_by_key: dict[
        _EvaluationKey,
        tuple[int, _ComplexWeight],
    ] = {}
    for interaction in dag.interactions:
        if (
            interaction.evaluation_factor == (0.0, 0.0)
            or certificate_chains[interaction.left_id]
            or certificate_chains[interaction.right_id]
        ):
            continue
        key, kernel_factor = _numerical_interaction_evaluation_key(
            dag,
            model,
            interaction,
            left_id=interaction.left_id,
            right_id=interaction.right_id,
            equivalence_by_kind=equivalence_by_kind,
        )
        prior = targets_by_key.get(key)
        candidate_rank = (
            interaction.evaluation_group_id is None,
            interaction.id,
        )
        if prior is None:
            targets_by_key[key] = (interaction.id, kernel_factor)
            continue
        prior_interaction = dag.interactions[prior[0]]
        prior_rank = (
            prior_interaction.evaluation_group_id is None,
            prior_interaction.id,
        )
        if candidate_rank < prior_rank:
            targets_by_key[key] = (interaction.id, kernel_factor)

    interactions = list(dag.interactions)
    next_group_id = 1 + max(
        (
            interaction.evaluation_group_id
            for interaction in dag.interactions
            if interaction.evaluation_group_id is not None
        ),
        default=-1,
    )
    projected_interaction_ids: dict[int, list[int]] = defaultdict(list)
    for interaction in dag.interactions:
        left = resolved[interaction.left_id]
        right = resolved[interaction.right_id]
        left_chain = certificate_chains[interaction.left_id]
        right_chain = certificate_chains[interaction.right_id]
        if not left_chain and not right_chain:
            continue

        if left.factor == (0.0, 0.0) or right.factor == (0.0, 0.0):
            if interaction.evaluation_factor == (0.0, 0.0):
                continue
            interactions[interaction.id] = replace(
                interaction,
                evaluation_factor=(0.0, 0.0),
            )
            used_chains = (
                *(left_chain if left.factor == (0.0, 0.0) else ()),
                *(right_chain if right.factor == (0.0, 0.0) else ()),
            )
            for certificate_id in set(used_chains):
                projected_interaction_ids[certificate_id].append(
                    interaction.id
                )
            continue

        parent_factor = _exact_representable_complex_product(
            left.factor,
            right.factor,
        )
        if parent_factor is None:
            continue
        key, mapped_kernel_factor = _numerical_interaction_evaluation_key(
            dag,
            model,
            interaction,
            left_id=left.representative_id,
            right_id=right.representative_id,
            equivalence_by_kind=equivalence_by_kind,
        )
        target = targets_by_key.get(key)
        if target is None:
            continue
        target_id, target_kernel_factor = target
        target_interaction = interactions[target_id]
        effective_kernel_factor = _exact_representable_complex_product(
            mapped_kernel_factor,
            parent_factor,
        )
        if effective_kernel_factor is None:
            continue
        relative_factor = _exact_representable_complex_ratio(
            effective_kernel_factor,
            target_kernel_factor,
        )
        if relative_factor is None:
            continue
        evaluation_factor = _exact_representable_complex_product(
            relative_factor,
            target_interaction.evaluation_factor,
        )
        if evaluation_factor is None or evaluation_factor == (0.0, 0.0):
            continue

        target_group_id = target_interaction.evaluation_group_id
        if target_group_id is None:
            target_group_id = next_group_id
            next_group_id += 1
            target_interaction = replace(
                target_interaction,
                evaluation_group_id=target_group_id,
            )
            interactions[target_id] = target_interaction
        if interaction.evaluation_group_id == target_group_id:
            continue
        interactions[interaction.id] = replace(
            interaction,
            evaluation_group_id=target_group_id,
            evaluation_factor=evaluation_factor,
        )
        for certificate_id in set((*left_chain, *right_chain)):
            projected_interaction_ids[certificate_id].append(interaction.id)

    # A numerical projection may coarsen an existing evaluator partition, but
    # it must never split one old live group.  If only part of a cached group
    # found a new target, retain that whole group unchanged and record no
    # application for those members.
    members_by_old_group: dict[tuple[str, int], list[int]] = defaultdict(list)
    for interaction in dag.interactions:
        if interaction.evaluation_factor == (0.0, 0.0):
            continue
        token = (
            ("group", interaction.evaluation_group_id)
            if interaction.evaluation_group_id is not None
            else ("interaction", interaction.id)
        )
        members_by_old_group[token].append(interaction.id)
    reverted_ids: set[int] = set()
    for member_ids in members_by_old_group.values():
        projected_live_groups = {
            (
                ("group", projected_member.evaluation_group_id)
                if projected_member.evaluation_group_id is not None
                else ("interaction", projected_member.id)
            )
            for interaction_id in member_ids
            if (
                projected_member := interactions[interaction_id]
            ).evaluation_factor
            != (0.0, 0.0)
        }
        if len(projected_live_groups) <= 1:
            continue
        for interaction_id in member_ids:
            if interactions[interaction_id] != dag.interactions[interaction_id]:
                interactions[interaction_id] = dag.interactions[
                    interaction_id
                ]
                reverted_ids.add(interaction_id)
    if reverted_ids:
        for certificate_id in tuple(projected_interaction_ids):
            retained = [
                interaction_id
                for interaction_id in projected_interaction_ids[
                    certificate_id
                ]
                if interaction_id not in reverted_ids
            ]
            if retained:
                projected_interaction_ids[certificate_id] = retained
            else:
                del projected_interaction_ids[certificate_id]

    rewritten = tuple(interactions)
    projected = (
        dag
        if rewritten == dag.interactions
        else _replace_dag_interactions_preserving_materialization(
            dag,
            rewritten,
        )
    )
    if projected.interaction_evaluation_count > dag.interaction_evaluation_count:
        raise ValueError(
            "authenticated numerical current application would increase "
            "the interaction evaluator count"
        )
    for member_ids in members_by_old_group.values():
        projected_live_groups = {
            (
                ("group", interaction.evaluation_group_id)
                if interaction.evaluation_group_id is not None
                else ("interaction", interaction.id)
            )
            for interaction_id in member_ids
            if (
                interaction := projected.interactions[interaction_id]
            ).evaluation_factor
            != (0.0, 0.0)
        }
        if len(projected_live_groups) > 1:
            raise ValueError(
                "authenticated numerical current application would split "
                "an existing evaluator group"
            )
    canonical_projected_ids = {
        certificate_id: tuple(interaction_ids)
        for certificate_id, interaction_ids in projected_interaction_ids.items()
    }
    return projected, resolved, certificate_chains, canonical_projected_ids


def _validated_decimal_observations(
    values: Sequence[tuple[Decimal, Decimal]] | None,
) -> tuple[tuple[Decimal, Decimal], ...] | None:
    if values is None or isinstance(values, str | bytes):
        return None
    result: list[tuple[Decimal, Decimal]] = []
    try:
        for value in values:
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or not all(isinstance(component, Decimal) for component in value)
                or not all(component.is_finite() for component in value)
            ):
                return None
            result.append((value[0], value[1]))
    except TypeError:
        return None
    return tuple(result)


def _numerical_relation_residuals(
    relation_kind: Literal["equal", "opposite", "zero"],
    current_values: tuple[tuple[Decimal, Decimal], ...],
    representative_values: tuple[tuple[Decimal, Decimal], ...] | None,
    *,
    relative_tolerance: Decimal,
    absolute_tolerance: Decimal,
    precision_digits: int,
) -> tuple[
    Decimal,
    Decimal,
    Decimal,
    tuple[tuple[Decimal, Decimal], ...],
] | None:
    if type(precision_digits) is not int or precision_digits < 80:
        raise ValueError("numerical residual precision is invalid")
    with localcontext() as context:
        context.prec = precision_digits + _NUMERICAL_DECIMAL_GUARD_DIGITS
        context.rounding = ROUND_HALF_EVEN
        if relation_kind == "zero":
            expected = tuple(
                (Decimal(0), Decimal(0)) for _ in current_values
            )
        else:
            if representative_values is None or len(
                representative_values
            ) != len(current_values):
                return None
            sign = Decimal(1) if relation_kind == "equal" else Decimal(-1)
            expected = tuple(
                (sign * real, sign * imaginary)
                for real, imaginary in representative_values
            )
        maximum_absolute = Decimal(0)
        maximum_relative = Decimal(0)
        maximum_tolerance_ratio = Decimal(0)
        component_checks: list[tuple[Decimal, Decimal]] = []
        for current, reference in zip(
            current_values,
            expected,
            strict=True,
        ):
            difference = max(
                abs(current[0] - reference[0]),
                abs(current[1] - reference[1]),
            )
            scale = max(
                abs(current[0]),
                abs(current[1]),
                abs(reference[0]),
                abs(reference[1]),
            )
            relative = (
                Decimal(0)
                if difference == 0
                else difference / max(scale, absolute_tolerance)
            )
            maximum_absolute = max(maximum_absolute, difference)
            maximum_relative = max(maximum_relative, relative)
            allowed = absolute_tolerance + relative_tolerance * scale
            tolerance_ratio = (
                Decimal(0) if difference == 0 else difference / allowed
            )
            maximum_tolerance_ratio = max(
                maximum_tolerance_ratio,
                tolerance_ratio,
            )
            component_checks.append((difference, allowed))
        return (
            maximum_absolute,
            maximum_relative,
            maximum_tolerance_ratio,
            tuple(component_checks),
        )


def _numerical_relation_residuals_pass(
    residuals: tuple[
        Decimal,
        Decimal,
        Decimal,
        tuple[tuple[Decimal, Decimal], ...],
    ],
    *,
    precision_digits: int,
) -> bool:
    if type(precision_digits) is not int or precision_digits < 80:
        raise ValueError("numerical residual precision is invalid")
    with localcontext() as context:
        context.prec = precision_digits + _NUMERICAL_DECIMAL_GUARD_DIGITS
        context.rounding = ROUND_HALF_EVEN
        return all(
            difference <= allowed
            for difference, allowed in residuals[3]
        )


def _numerical_observations_sha256(
    *,
    relation_kind: str,
    current_values: tuple[tuple[Decimal, Decimal], ...],
    representative_values: tuple[tuple[Decimal, Decimal], ...] | None,
) -> str:
    return _canonical_payload_sha256(
        {
            "relation_kind": relation_kind,
            "current": [
                [
                    _canonical_decimal_string(real),
                    _canonical_decimal_string(imaginary),
                ]
                for real, imaginary in current_values
            ],
            "representative": (
                None
                if representative_values is None
                else [
                    [
                        _canonical_decimal_string(real),
                        _canonical_decimal_string(imaginary),
                    ]
                    for real, imaginary in representative_values
                ]
            ),
        }
    )


def _canonical_decimal_string(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("canonical Decimal serialization requires finiteness")
    sign, digits, exponent = value.as_tuple()
    if not any(digits):
        return "0"
    if not isinstance(exponent, int):
        raise ValueError("canonical Decimal serialization requires an exponent")
    coefficient = "".join(str(digit) for digit in digits)
    return f"{'-' if sign else ''}{coefficient}e{exponent}"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _decode_numerical_relation_factor(
    relation_kind: object,
    representative_id: object,
    payload: object,
) -> _ComplexWeight | None:
    expected: _ComplexWeight | None
    if relation_kind == "zero":
        if representative_id is not None or payload is not None:
            raise ValueError("zero-current relation cannot have a representative")
        return None
    if relation_kind == "equal":
        expected = (1.0, 0.0)
    elif relation_kind == "opposite":
        expected = (-1.0, 0.0)
    else:
        raise ValueError("unsupported numerical current relation kind")
    if (
        type(representative_id) is not int
        or not isinstance(payload, list)
        or len(payload) != 2
        or any(not isinstance(component, str) for component in payload)
    ):
        raise ValueError("numerical current relation factor is malformed")
    try:
        decoded = (float.fromhex(payload[0]), float.fromhex(payload[1]))
    except ValueError as error:
        raise ValueError("numerical current relation factor is malformed") from error
    if not _complex_weight_bits_equal(decoded, expected):
        raise ValueError("numerical current relation factor disagrees with its kind")
    return expected


@dataclass(frozen=True, slots=True)
class RelationDiscoveryReport:
    """Mode- and colour-scoped outcome of one opt-in discovery pass."""

    requested_mode: str
    state: str
    execution_mode: str
    color_accuracy: str
    representation: str
    precision_digits: int
    probe_count: int
    seed: int
    probe_status: str
    certificate_replay_status: str
    numerical_candidate_count: int
    uncertified_candidate_count: int
    exact_certified_relation_count: int
    applied_relation_count: int
    interaction_evaluation_count_before: int | None
    interaction_evaluation_count_after: int | None
    certificates: tuple[ExactCurrentRelationCertificate, ...] = ()
    rejected_candidates: tuple[dict[str, object], ...] = ()
    rejected_decision_sha256: str = ""
    follow_up_boundary: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": RELATION_DISCOVERY_SCHEMA_VERSION,
            "requested_mode": self.requested_mode,
            "state": self.state,
            "scope": {
                "execution_mode": self.execution_mode,
                "color_accuracy": self.color_accuracy,
                "representation": self.representation,
            },
            "probe": {
                "status": self.probe_status,
                "precision_digits": self.precision_digits,
                "probe_count": self.probe_count,
                "seed": self.seed,
                "deterministic": True,
                "candidate_only": True,
            },
            "certificate_replay": {
                "algorithm": RELATION_DISCOVERY_CERTIFICATE_ALGORITHM,
                "status": self.certificate_replay_status,
            },
            "numerical_candidate_count": self.numerical_candidate_count,
            "uncertified_candidate_count": self.uncertified_candidate_count,
            "rejected_hypothesis_count": self.uncertified_candidate_count,
            "exact_certified_relation_count": (self.exact_certified_relation_count),
            "applied_relation_count": self.applied_relation_count,
            "interaction_evaluation_count_before": (
                self.interaction_evaluation_count_before
            ),
            "interaction_evaluation_count_after": (
                self.interaction_evaluation_count_after
            ),
            "interaction_evaluation_savings": (
                None
                if self.interaction_evaluation_count_before is None
                or self.interaction_evaluation_count_after is None
                else max(
                    0,
                    self.interaction_evaluation_count_before
                    - self.interaction_evaluation_count_after,
                )
            ),
            "certificates": [
                certificate.to_json_dict() for certificate in self.certificates
            ],
            "rejected_candidates": list(self.rejected_candidates),
            "rejected_candidate_diagnostics": (
                _rejected_candidate_diagnostics(
                    self.rejected_candidates,
                    total_rejected=self.uncertified_candidate_count,
                    full_census={
                        "requested_mode": self.requested_mode,
                        "numerical_candidate_count": (
                            self.numerical_candidate_count
                        ),
                        "uncertified_candidate_count": (
                            self.uncertified_candidate_count
                        ),
                        "rejected_hypothesis_count": (
                            self.uncertified_candidate_count
                        ),
                        "exact_certified_relation_count": (
                            self.exact_certified_relation_count
                        ),
                        "applied_relation_count": self.applied_relation_count,
                        "rejection_decision_sha256": (
                            self.rejected_decision_sha256
                        ),
                    },
                    full_rejection_sha256=self.rejected_decision_sha256,
                )
            ),
            "follow_up_boundary": self.follow_up_boundary,
        }


@dataclass(frozen=True, slots=True)
class RelationDiscoveryResult:
    dag: GenericDAG
    report: RelationDiscoveryReport


class _NumericalRelationDiscovery:
    """Nominate projective pairs numerically, then require an exact replay."""

    def __init__(
        self,
        *,
        precision_digits: int,
        probe_count: int,
        seed: int,
    ) -> None:
        if precision_digits < 80:
            raise ValueError("relation discovery precision must be at least 80 digits")
        if probe_count < 2:
            raise ValueError("relation discovery requires at least two probe points")
        if seed < 0:
            raise ValueError("relation discovery seed must be nonnegative")
        self.precision_digits = int(precision_digits)
        self.probe_count = int(probe_count)
        self.seed = int(seed)
        self.numerical_candidate_count = 0
        self.uncertified_candidate_count = 0
        self.certificates: list[ExactCurrentRelationCertificate] = []
        self.rejected_candidates: list[dict[str, object]] = []
        self.rejected_decision_sha256 = _canonical_payload_sha256(
            {
                "abi": "pyamplicol-exact-rejected-decision-chain-v1",
                "precision_digits": self.precision_digits,
                "probe_count": self.probe_count,
                "seed": self.seed,
            }
        )
        self._representatives_by_fingerprint: dict[
            tuple[_CurrentContract, tuple[object, ...]],
            list[tuple[int, _CurrentTermVector]],
        ] = {}

    def record_rejected(self, row: Mapping[str, object]) -> None:
        payload = dict(row)
        self.rejected_decision_sha256 = _advance_rejection_digest(
            self.rejected_decision_sha256,
            payload,
        )
        if len(self.rejected_candidates) < _MAX_REJECTED_DIAGNOSTICS:
            self.rejected_candidates.append(payload)

    def consider(
        self,
        *,
        current_id: int,
        contract: _CurrentContract,
        term_vector: _CurrentTermVector,
        baseline: _CurrentValueEquivalence,
    ) -> _CurrentValueEquivalence:
        fingerprint = _numerical_projective_fingerprint(
            term_vector,
            precision_digits=self.precision_digits,
            probe_count=self.probe_count,
            seed=self.seed,
        )
        if fingerprint is None:
            return baseline
        key = (contract, fingerprint)
        representatives = self._representatives_by_fingerprint.setdefault(key, [])

        promoted = baseline
        if baseline.representative_id == current_id and baseline.factor == (1.0, 0.0):
            exact_matches: list[tuple[int, _ComplexWeight, _CurrentTermVector]] = []
            for representative_id, representative_vector in representatives:
                self.numerical_candidate_count += 1
                factor = _certify_exact_term_vector_relation(
                    representative_vector,
                    term_vector,
                )
                if factor is None:
                    self.uncertified_candidate_count += 1
                    self.record_rejected(
                        {
                            "current_id": current_id,
                            "representative_id": representative_id,
                            "reason": "numerical-candidate-lacks-exact-certificate",
                        }
                    )
                    continue
                exact_matches.append((representative_id, factor, representative_vector))
            if exact_matches:
                representative_id, factor, representative_vector = min(
                    exact_matches,
                    key=lambda item: item[0],
                )
                certificate = _build_exact_current_relation_certificate(
                    current_id=current_id,
                    representative_id=representative_id,
                    factor=factor,
                    current_term_vector=term_vector,
                    representative_term_vector=representative_vector,
                )
                self.certificates.append(certificate)
                promoted = _CurrentValueEquivalence(
                    representative_id=representative_id,
                    factor=factor,
                )

        representatives.append((current_id, term_vector))
        return promoted


class RecursiveEvaluationReuseTracker:
    """Certify recursive-current reuse as external subsets are completed."""

    def __init__(self, model: Model) -> None:
        self._model = model
        self._kernel_equivalences: dict[int, VertexEvaluationEquivalence] = {}
        self._runtime_coupling_identities: dict[
            tuple[int, tuple[int, int, int], tuple[float, float]],
            tuple[tuple[float, float], tuple[tuple[int, str], ...]],
        ] = {}
        self._current_equivalences: list[_CurrentValueEquivalence] = []
        self._source_representative_by_key: dict[tuple[object, ...], int] = {}
        self._equivalence_by_expression: dict[
            tuple[_CurrentContract, _CurrentTermVector], _CurrentValueEquivalence
        ] = {}
        self._projective_representatives_by_expression: dict[
            tuple[_CurrentContract, _CurrentTermVector],
            list[_ProjectiveExpressionRepresentative],
        ] = {}
        self._evaluation_group_by_key: dict[_EvaluationKey, int] = {}
        self._coefficients_by_result: list[
            dict[int, _ComplexWeight | list[_ComplexWeight]] | None
        ] = []

    def register_source(self, current: CurrentNode) -> None:
        if not current.is_source:
            raise ValueError("recursive-reuse source registration requires a source")
        if current.source_leg_label is None or current.source_helicity is None:
            raise ValueError(
                f"source current {current.id} lacks physical source metadata"
            )
        contract = _current_evaluation_contract(current)
        source_key = (
            contract,
            int(current.source_leg_label),
            int(current.source_helicity),
        )
        representative_id = self._source_representative_by_key.setdefault(
            source_key,
            current.id,
        )
        self._append_current_equivalence(
            current,
            _CurrentValueEquivalence(representative_id, (1.0, 0.0)),
        )

    def interaction_evaluation(
        self,
        *,
        vertex_kind: int,
        vertex_particles: tuple[int, int, int],
        left_id: int,
        right_id: int,
        result: CurrentNode,
        coupling: tuple[float, float],
        color_weight: _ComplexWeight,
    ) -> tuple[int, _ComplexWeight]:
        kernel_equivalence = _kernel_equivalence(
            self._model,
            vertex_kind,
            self._kernel_equivalences,
        )
        try:
            left = self._current_equivalences[left_id]
            right = self._current_equivalences[right_id]
        except IndexError as error:
            raise ValueError(
                "online recursive reuse requires completed parent subsets"
            ) from error
        canonical_inputs, evaluation_factor = _canonical_interaction_evaluation(
            kernel_equivalence,
            left_id=left_id,
            right_id=right_id,
            left=left,
            right=right,
        )
        coupling_key = (vertex_kind, vertex_particles, coupling)
        coupling_identity = self._runtime_coupling_identities.get(coupling_key)
        if coupling_identity is None:
            coupling_identity = _runtime_coupling_identity(
                self._model,
                vertex_kind=vertex_kind,
                vertex_particles=vertex_particles,
                coupling=coupling,
            )
            self._runtime_coupling_identities[coupling_key] = coupling_identity
        evaluation_key = (
            kernel_equivalence.class_id,
            canonical_inputs,
            int(result.index.particle_id),
            int(result.index.chirality),
            coupling_identity,
        )
        evaluation_group_id = self._evaluation_group_by_key.setdefault(
            evaluation_key,
            len(self._evaluation_group_by_key),
        )
        if result.id >= len(self._coefficients_by_result):
            self._coefficients_by_result.extend(
                [None] * (result.id + 1 - len(self._coefficients_by_result))
            )
        coefficients_by_group = self._coefficients_by_result[result.id]
        if coefficients_by_group is None:
            coefficients_by_group = {}
            self._coefficients_by_result[result.id] = coefficients_by_group
        coefficient = _complex_weight_mul(color_weight, evaluation_factor)
        coefficients = coefficients_by_group.get(evaluation_group_id)
        if coefficients is None:
            coefficients_by_group[evaluation_group_id] = coefficient
        elif isinstance(coefficients, list):
            coefficients.append(coefficient)
        else:
            coefficients_by_group[evaluation_group_id] = [coefficients, coefficient]
        return evaluation_group_id, evaluation_factor

    def finalize_currents(
        self,
        currents: Iterable[CurrentNode],
    ) -> None:
        for current in currents:
            if current.is_source:
                raise ValueError("generated-current finalization received a source")
            coefficients_by_group = self._coefficients_by_result[current.id]
            self._coefficients_by_result[current.id] = None
            terms: list[tuple[_EvaluationKey, _ComplexWeight]] = []
            for group_id in sorted(coefficients_by_group or ()):
                assert coefficients_by_group is not None
                coefficients = coefficients_by_group[group_id]
                if isinstance(coefficients, list):
                    coefficient = (
                        _canonical_zero(fsum(value[0] for value in coefficients)),
                        _canonical_zero(fsum(value[1] for value in coefficients)),
                    )
                else:
                    coefficient = _canonical_complex_weight(coefficients)
                if coefficient != (0.0, 0.0):
                    terms.append(((group_id,), coefficient))
            term_vector = tuple(terms)
            contract = _current_evaluation_contract(current)
            equivalence = _classify_current_term_vector(
                current_id=current.id,
                contract=contract,
                term_vector=term_vector,
                equivalence_by_expression=self._equivalence_by_expression,
                projective_representatives_by_expression=(
                    self._projective_representatives_by_expression
                ),
            )
            self._append_current_equivalence(
                current,
                equivalence,
            )

    def _append_current_equivalence(
        self,
        current: CurrentNode,
        equivalence: _CurrentValueEquivalence,
    ) -> None:
        if current.id != len(self._current_equivalences):
            raise ValueError(
                "online recursive reuse requires currents in contiguous ID order"
            )
        self._current_equivalences.append(equivalence)
        if current.id >= len(self._coefficients_by_result):
            self._coefficients_by_result.append(None)


def _canonical_kernel_evaluation(
    equivalence: VertexEvaluationEquivalence,
    left_id: int,
    right_id: int,
) -> tuple[tuple[int, int], _ComplexWeight]:
    """Return canonical representative inputs and the concrete-kernel factor."""

    canonical_inputs = (left_id, right_id)
    if equivalence.input_order == (1, 0):
        canonical_inputs = (right_id, left_id)
    factor = equivalence.factor
    if (
        equivalence.input_exchange_factor is not None
        and canonical_inputs[1] < canonical_inputs[0]
    ):
        canonical_inputs = (canonical_inputs[1], canonical_inputs[0])
        factor = _complex_weight_mul(
            factor,
            equivalence.input_exchange_factor,
        )
    return canonical_inputs, factor


def _canonical_interaction_evaluation(
    equivalence: VertexEvaluationEquivalence,
    *,
    left_id: int,
    right_id: int,
    left: _CurrentValueEquivalence,
    right: _CurrentValueEquivalence,
) -> tuple[tuple[int, int], _ComplexWeight]:
    """Compose recursive factors exactly or retain the concrete inputs.

    Projective equivalence is an exact statement about the ideal algebraic
    current.  Its recursive factor may be moved through another kernel only
    when both complex products remain finite and exactly representable as
    binary64.  Otherwise this interaction fails closed to its concrete parent
    IDs while retaining the pre-existing model-certified kernel symmetry.
    """

    canonical_inputs, kernel_factor = _canonical_kernel_evaluation(
        equivalence,
        left.representative_id,
        right.representative_id,
    )
    parent_factor = _exact_representable_complex_product(left.factor, right.factor)
    if parent_factor is not None:
        evaluation_factor = _exact_representable_complex_product(
            kernel_factor,
            parent_factor,
        )
        if evaluation_factor is not None:
            return canonical_inputs, evaluation_factor
    return _canonical_kernel_evaluation(equivalence, left_id, right_id)


def assign_recursive_current_evaluation_reuse(
    dag: GenericDAG,
    model: Model,
) -> GenericDAG:
    """Share kernel evaluations through exactly proven current equivalences.

    The proof is recursive.  Duplicate source wavefunctions form the base
    classes.  A generated current joins an existing class only when its full
    vector of model-certified kernel terms and coefficients is byte-exactly
    equal, opposite, or algebraically projective through a finite nonzero
    binary64 factor whose coefficient reconstruction is bit-exact.  Moving a
    general projective factor changes floating-point association, so runtime
    parity remains a tolerance-checked numerical contract rather than a claim
    of bit-identical materialized currents.  Recursive factor products fail
    closed unless they are finite and exactly representable.  The current
    contract keeps every field consumed by source, kernel, and propagator
    evaluation.  In particular, ordered external labels and helicity
    ancestry remain part of the contract: both encode physical runtime
    routing and cannot be inferred from the sorted external-label set.
    Dynamic colour-sector bookkeeping is deliberately excluded.  Input
    ordering may differ only through the exact model-certified permutation
    and reflection factors included in the term signature below.

    This recovers AmpliCol-style reflection fan-out, but also recognizes exact
    reuse across colour sectors and helicity subgraphs.  No approximate
    numerical comparison or process/model-family classification is involved.
    """

    if not dag.interactions:
        return dag

    equivalence_by_kind: dict[int, VertexEvaluationEquivalence] = {}
    current_equivalences = _derive_current_value_equivalences(
        dag,
        model,
        equivalence_by_kind=equivalence_by_kind,
    )
    return _rewrite_interaction_evaluation_reuse(
        dag,
        model,
        current_equivalences=current_equivalences,
        equivalence_by_kind=equivalence_by_kind,
    )


def discover_recursive_evaluation_relations(
    dag: GenericDAG,
    model: Model,
    *,
    mode: _DiscoveryMode,
    execution_mode: Literal["compiled", "eager"],
    precision_digits: int = 96,
    probe_count: int = 4,
    seed: int = 0x5059414D,
) -> RelationDiscoveryResult:
    """Discover candidates numerically and promote only exact DAG relations.

    Probe values are deterministic high-precision synthetic evaluations of the
    full current term vectors. They are candidate indexes only. Every promoted
    relation is independently replayed over the complete binary64 coefficient
    vectors, and unsafe recursive factor composition continues to fail closed.
    """

    if mode not in {"diagnostic", "certified-reuse"}:
        raise ValueError(f"unsupported relation discovery mode {mode!r}")
    if execution_mode not in {"compiled", "eager"}:
        raise ValueError(
            "GenericDAG relation discovery supports compiled and eager execution"
        )

    baseline = assign_recursive_current_evaluation_reuse(dag, model)
    discovery = _NumericalRelationDiscovery(
        precision_digits=precision_digits,
        probe_count=probe_count,
        seed=seed,
    )
    equivalence_by_kind: dict[int, VertexEvaluationEquivalence] = {}
    current_equivalences = _derive_current_value_equivalences(
        dag,
        model,
        equivalence_by_kind=equivalence_by_kind,
        discovery=discovery,
    )
    promoted = _rewrite_interaction_evaluation_reuse(
        dag,
        model,
        current_equivalences=current_equivalences,
        equivalence_by_kind=equivalence_by_kind,
    )
    if discovery.certificates and not verify_dag_relation_certificates(
        dag,
        model,
        discovery.certificates,
    ):
        discovery.uncertified_candidate_count += len(discovery.certificates)
        for certificate in discovery.certificates:
            discovery.record_rejected(
                {
                    "reason": "exact-certificate-chain-replay-failed",
                    "current_id": certificate.current_id,
                    "representative_id": certificate.representative_id,
                }
            )
        discovery.certificates.clear()
        promoted = baseline
    interaction_changed = _interaction_reuse_signature(
        promoted
    ) != _interaction_reuse_signature(baseline)
    apply_promotions = (
        mode == "certified-reuse"
        and bool(discovery.certificates)
        and interaction_changed
    )
    result_dag = promoted if apply_promotions else dag
    report = RelationDiscoveryReport(
        requested_mode=mode,
        state=("exact-certified-applied" if apply_promotions else "diagnostic-only"),
        execution_mode=execution_mode,
        color_accuracy=str(dag.process.color_accuracy),
        representation="generic-dag",
        precision_digits=precision_digits,
        probe_count=probe_count,
        seed=seed,
        probe_status="completed",
        certificate_replay_status=(
            "verified" if discovery.certificates else "no-certified-relations"
        ),
        numerical_candidate_count=discovery.numerical_candidate_count,
        uncertified_candidate_count=discovery.uncertified_candidate_count,
        exact_certified_relation_count=len(discovery.certificates),
        applied_relation_count=(len(discovery.certificates) if apply_promotions else 0),
        interaction_evaluation_count_before=baseline.interaction_evaluation_count,
        interaction_evaluation_count_after=(
            promoted.interaction_evaluation_count
            if apply_promotions or mode == "diagnostic"
            else baseline.interaction_evaluation_count
        ),
        certificates=tuple(discovery.certificates),
        rejected_candidates=tuple(discovery.rejected_candidates),
        rejected_decision_sha256=discovery.rejected_decision_sha256,
        follow_up_boundary=(
            "direct vertex-kernel equivalence remains model-certificate-owned; "
            "this pass promotes exact current proportionality and its induced "
            "interaction fan-out only"
        ),
    )
    return RelationDiscoveryResult(dag=result_dag, report=report)


def _rewrite_interaction_evaluation_reuse(
    dag: GenericDAG,
    model: Model,
    *,
    current_equivalences: tuple[_CurrentValueEquivalence, ...],
    equivalence_by_kind: dict[int, VertexEvaluationEquivalence],
) -> GenericDAG:
    evaluation_group_by_key: dict[_EvaluationKey, int] = {}
    interactions: list[InteractionNode] = []

    for interaction in dag.interactions:
        kernel_equivalence = _kernel_equivalence(
            model,
            interaction.vertex_kind,
            equivalence_by_kind,
        )
        left = current_equivalences[interaction.left_id]
        right = current_equivalences[interaction.right_id]
        canonical_inputs, evaluation_factor = _canonical_interaction_evaluation(
            kernel_equivalence,
            left_id=interaction.left_id,
            right_id=interaction.right_id,
            left=left,
            right=right,
        )
        result = dag.currents[interaction.result_id]
        evaluation_key = (
            kernel_equivalence.class_id,
            canonical_inputs,
            int(result.index.particle_id),
            int(result.index.chirality),
            _runtime_coupling_identity(
                model,
                vertex_kind=interaction.vertex_kind,
                vertex_particles=interaction.vertex_particles,
                coupling=interaction.coupling,
            ),
        )
        evaluation_group_id = evaluation_group_by_key.setdefault(
            evaluation_key,
            len(evaluation_group_by_key),
        )
        interactions.append(
            replace(
                interaction,
                evaluation_group_id=evaluation_group_id,
                evaluation_factor=evaluation_factor,
            )
        )

    rewritten = tuple(interactions)
    if rewritten == dag.interactions:
        return dag
    return _replace_dag_interactions_preserving_materialization(dag, rewritten)


def _replace_dag_interactions_preserving_materialization(
    dag: GenericDAG,
    interactions: tuple[InteractionNode, ...],
) -> GenericDAG:
    """Replace evaluation metadata without invalidating a quotient witness."""

    materialization = dag.helicity_materialization
    if materialization is None:
        return replace(dag, interactions=interactions)
    rebound_dag = replace(
        dag,
        interactions=interactions,
        helicity_materialization=None,
    )
    rebound_materialization = replace(materialization, dag=rebound_dag)
    return replace(
        rebound_dag,
        helicity_materialization=rebound_materialization,
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolved_root_color_sector_id(dag: GenericDAG, root: AmplitudeRoot) -> int:
    """Resolve the physical sector before any representative is selected."""

    if root.color_sector_id is not None:
        return int(root.color_sector_id)
    return int(dag.currents[root.left_id].index.color_state.sector_id)


def _downstream_color_selector_domains(
    dag: GenericDAG,
) -> tuple[tuple[int, ...], ...]:
    """Return the complete physical LC selector domain of every current."""

    domains: list[set[int]] = [set() for _current in dag.currents]
    for root in dag.amplitude_roots:
        sector_id = _resolved_root_color_sector_id(dag, root)
        domains[root.left_id].add(sector_id)
        domains[root.right_id].add(sector_id)
    for interaction in reversed(dag.interactions):
        result_domains = domains[interaction.result_id]
        if not result_domains:
            continue
        domains[interaction.left_id].update(result_domains)
        domains[interaction.right_id].update(result_domains)
    return tuple(tuple(sorted(values)) for values in domains)


def _dynamic_color_projection_key(
    current: CurrentNode,
    selector_domains: tuple[int, ...],
) -> tuple[object, ...]:
    """Return exact current identity with only dynamic LC sector omitted.

    ``basis_key`` participates in coherent amplitude grouping and eager
    metadata, while ``line_groups`` is the exact open-colour identity.  Both
    therefore remain invariant within a projected class.  The lowerings use
    particle/chirality/momentum plus the explicit row identity for numerical
    kernels; the retained representative sector is metadata only after every
    amplitude root has received its pre-projection resolved sector.
    """

    index = current.index
    return (
        int(index.particle_id),
        int(index.external_mask),
        index.external_labels,
        index.ordered_external_labels,
        int(index.helicity_ancestry),
        int(index.chirality),
        index.spin_state,
        index.flavour_flow,
        index.quantum_number_flow,
        index.color_state.accuracy,
        index.color_state.line_groups,
        index.color_state.basis_key,
        int(index.momentum_mask),
        index.coupling_orders,
        index.auxiliary_kind,
        int(current.dimension),
        selector_domains,
    )


def _projection_partition(
    dag: GenericDAG,
    selector_domains: tuple[tuple[int, ...], ...],
    split_current_ids: set[int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    grouped: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for current in dag.currents:
        # External source rows remain singleton.  Their physical source route
        # is an initialization contract, not a multilinear recurrence row.
        if current.is_source or current.id in split_current_ids:
            key = ("singleton", current.id)
        else:
            key = (
                "dynamic-color",
                _dynamic_color_projection_key(
                    current,
                    selector_domains[current.id],
                ),
            )
        grouped[key].append(current.id)
    classes = tuple(
        sorted(
            (tuple(sorted(members)) for members in grouped.values()),
            key=lambda members: (
                len(dag.currents[members[0]].index.external_labels),
                members[0],
            ),
        )
    )
    class_by_current = [-1] * len(dag.currents)
    for class_id, members in enumerate(classes):
        for current_id in members:
            class_by_current[current_id] = class_id
    if any(class_id < 0 for class_id in class_by_current):
        raise ValueError("dynamic-color projection left a current unpartitioned")
    return classes, tuple(class_by_current)


def _interaction_projection_key(
    interaction: InteractionNode,
    class_by_current: tuple[int, ...],
    *,
    keep_row_distinct: bool,
) -> tuple[object, ...]:
    return (
        class_by_current[interaction.result_id],
        class_by_current[interaction.left_id],
        class_by_current[interaction.right_id],
        int(interaction.vertex_kind),
        interaction.vertex_particles,
        interaction.coupling,
        interaction.color_weight,
        interaction.lowering_backend,
        bool(interaction.full_tensor_network_ready),
        interaction.id if keep_row_distinct else None,
    )


def _root_projection_key(
    dag: GenericDAG,
    root: AmplitudeRoot,
    class_by_current: tuple[int, ...],
    *,
    keep_row_distinct: bool,
) -> tuple[object, ...]:
    return (
        class_by_current[root.left_id],
        class_by_current[root.right_id],
        root.kind,
        root.color_weight,
        root.contraction_ir,
        _resolved_root_color_sector_id(dag, root),
        root.vertex_kind,
        root.vertex_particles,
        root.coupling,
        float(root.helicity_weight),
        root.id if keep_row_distinct else None,
    )


def _projection_groups(
    dag: GenericDAG,
    classes: tuple[tuple[int, ...], ...],
    class_by_current: tuple[int, ...],
) -> tuple[
    dict[tuple[object, ...], list[InteractionNode]],
    dict[tuple[object, ...], list[AmplitudeRoot]],
]:
    interaction_groups: dict[tuple[object, ...], list[InteractionNode]] = defaultdict(
        list
    )
    for interaction in dag.interactions:
        involved = (
            class_by_current[interaction.result_id],
            class_by_current[interaction.left_id],
            class_by_current[interaction.right_id],
        )
        keep_distinct = all(len(classes[class_id]) == 1 for class_id in involved)
        interaction_groups[
            _interaction_projection_key(
                interaction,
                class_by_current,
                keep_row_distinct=keep_distinct,
            )
        ].append(interaction)

    root_groups: dict[tuple[object, ...], list[AmplitudeRoot]] = defaultdict(list)
    for root in dag.amplitude_roots:
        involved = (
            class_by_current[root.left_id],
            class_by_current[root.right_id],
        )
        keep_distinct = all(len(classes[class_id]) == 1 for class_id in involved)
        root_groups[
            _root_projection_key(
                dag,
                root,
                class_by_current,
                keep_row_distinct=keep_distinct,
            )
        ].append(root)
    return interaction_groups, root_groups


def _invalid_projection_classes(
    dag: GenericDAG,
    classes: tuple[tuple[int, ...], ...],
    class_by_current: tuple[int, ...],
    interaction_groups: dict[tuple[object, ...], list[InteractionNode]],
    root_groups: dict[tuple[object, ...], list[AmplitudeRoot]],
) -> set[int]:
    """Return every non-singleton class implicated in a non-rectangle."""

    invalid: set[int] = set()
    produced_by_class: dict[int, set[int]] = defaultdict(set)
    for rows in interaction_groups.values():
        first = rows[0]
        result_class = class_by_current[first.result_id]
        left_class = class_by_current[first.left_id]
        right_class = class_by_current[first.right_id]
        produced_by_class[result_class].update(row.result_id for row in rows)
        if all(
            len(classes[class_id]) == 1
            for class_id in (result_class, left_class, right_class)
        ):
            continue
        actual = [(row.left_id, row.right_id) for row in rows]
        expected = {
            (left_id, right_id)
            for left_id in classes[left_class]
            for right_id in classes[right_class]
        }
        if len(actual) != len(set(actual)) or set(actual) != expected:
            invalid.update(
                class_id
                for class_id in (result_class, left_class, right_class)
                if len(classes[class_id]) > 1
            )

    for class_id, members in enumerate(classes):
        if len(members) == 1 or dag.currents[members[0]].is_source:
            continue
        if produced_by_class.get(class_id, set()) != set(members):
            invalid.add(class_id)

    for roots in root_groups.values():
        first = roots[0]
        left_class = class_by_current[first.left_id]
        right_class = class_by_current[first.right_id]
        if len(classes[left_class]) == len(classes[right_class]) == 1:
            continue
        actual = [(root.left_id, root.right_id) for root in roots]
        expected = {
            (left_id, right_id)
            for left_id in classes[left_class]
            for right_id in classes[right_class]
        }
        if len(actual) != len(set(actual)) or set(actual) != expected:
            invalid.update(
                class_id
                for class_id in (left_class, right_class)
                if len(classes[class_id]) > 1
            )
    return invalid


_DYNAMIC_COLOR_PROJECTION_ABI = (
    "pyamplicol-generic-dag-dynamic-color-projection-v2"
)
_RETAINED_COLOR_METADATA_POLICY = (
    "accuracy-line-groups-basis-key-identical; representative-sector-metadata-only"
)
_ROOT_SECTOR_POLICY = (
    "pre-projection-resolved-sector-is-explicit-and-never-cross-projected"
)


def dynamic_color_projection_not_applicable_report(
    dag: GenericDAG,
) -> dict[str, object]:
    """Return a count-only report when the LC-only projection cannot apply."""

    if dag.process.color_accuracy == "lc":
        raise ValueError("LC DAGs require the dynamic-color projection pass")
    return {
        "abi": _DYNAMIC_COLOR_PROJECTION_ABI,
        "applied": False,
        "color_accuracy": str(dag.process.color_accuracy),
        "equality_check_status": "not-applicable-color-accuracy",
        "before_current_count": len(dag.currents),
        "after_current_count": len(dag.currents),
        "before_interaction_count": len(dag.interactions),
        "after_interaction_count": len(dag.interactions),
        "before_evaluation_count": dag.interaction_evaluation_count,
        "after_evaluation_count": dag.interaction_evaluation_count,
        "before_amplitude_root_count": len(dag.amplitude_roots),
        "after_amplitude_root_count": len(dag.amplitude_roots),
    }


def _initial_dynamic_color_projection_certificate(
    dag: GenericDAG,
    *,
    source_revision: str | None,
    equality_check_status: str,
) -> DynamicColorProjectionCertificate:
    identity = tuple(range(len(dag.currents)))
    return DynamicColorProjectionCertificate(
        abi=_DYNAMIC_COLOR_PROJECTION_ABI,
        source_revision=source_revision,
        source_semantics_sha256=_canonical_sha256(dag.to_json_dict()),
        selector_domains_sha256=_canonical_sha256([]),
        current_class_members_sha256=_canonical_sha256(
            [[current.id] for current in dag.currents]
        ),
        old_to_new_current_ids=identity,
        current_remap_sha256=_canonical_sha256(list(identity)),
        interaction_groups_sha256=_canonical_sha256(
            [[interaction.id] for interaction in dag.interactions]
        ),
        closure_groups_sha256=_canonical_sha256(
            [[root.id] for root in dag.amplitude_roots]
        ),
        rectangle_cardinalities_sha256=_canonical_sha256([]),
        row_identity_sha256=_canonical_sha256(
            {
                "interactions": [
                    interaction.to_json_dict() for interaction in dag.interactions
                ],
                "roots": [root.to_json_dict() for root in dag.amplitude_roots],
            }
        ),
        equality_check_status=equality_check_status,
        retained_color_metadata=_RETAINED_COLOR_METADATA_POLICY,
        root_sector_policy=_ROOT_SECTOR_POLICY,
        before_current_count=len(dag.currents),
        after_current_count=len(dag.currents),
        before_interaction_count=len(dag.interactions),
        after_interaction_count=len(dag.interactions),
        before_evaluation_count=dag.interaction_evaluation_count,
        after_evaluation_count=dag.interaction_evaluation_count,
        before_amplitude_root_count=len(dag.amplitude_roots),
        after_amplitude_root_count=len(dag.amplitude_roots),
        projected_current_class_count=0,
        rectangular_interaction_group_count=0,
        rectangular_closure_group_count=0,
        split_current_class_count=0,
    )


def _projection_proof_payloads(
    dag: GenericDAG,
    selector_domains: tuple[tuple[int, ...], ...],
    classes: tuple[tuple[int, ...], ...],
    class_by_current: tuple[int, ...],
    interaction_groups: dict[tuple[object, ...], list[InteractionNode]],
    root_groups: dict[tuple[object, ...], list[AmplitudeRoot]],
) -> dict[str, object]:
    """Return the complete deterministic witness committed by the certificate."""

    ordered_interaction_groups = sorted(
        interaction_groups.values(),
        key=lambda rows: min(row.id for row in rows),
    )
    ordered_root_groups = sorted(
        root_groups.values(),
        key=lambda rows: min(row.id for row in rows),
    )
    rectangle_cardinalities: list[dict[str, object]] = []
    row_identities: list[dict[str, object]] = []
    for rows in ordered_interaction_groups:
        first = rows[0]
        result_class = class_by_current[first.result_id]
        left_class = class_by_current[first.left_id]
        right_class = class_by_current[first.right_id]
        actual_pairs = sorted((row.left_id, row.right_id) for row in rows)
        rectangle_cardinalities.append(
            {
                "kind": "interaction",
                "row_ids": sorted(row.id for row in rows),
                "result_class_size": len(classes[result_class]),
                "left_class_size": len(classes[left_class]),
                "right_class_size": len(classes[right_class]),
                "row_count": len(rows),
                "unique_parent_pair_count": len(set(actual_pairs)),
                "expected_parent_pair_count": (
                    len(classes[left_class]) * len(classes[right_class])
                ),
            }
        )
        row_identities.append(
            {
                "kind": "interaction",
                "result_class_id": result_class,
                "left_class_id": left_class,
                "right_class_id": right_class,
                "vertex_kind": int(first.vertex_kind),
                "vertex_particles": list(first.vertex_particles),
                "coupling": list(first.coupling),
                "color_weight": list(first.color_weight),
                "lowering_backend": first.lowering_backend,
                "full_tensor_network_ready": bool(
                    first.full_tensor_network_ready
                ),
                "row_ids": sorted(row.id for row in rows),
            }
        )
    for roots in ordered_root_groups:
        first = roots[0]
        left_class = class_by_current[first.left_id]
        right_class = class_by_current[first.right_id]
        actual_pairs = sorted((root.left_id, root.right_id) for root in roots)
        resolved_sector = _resolved_root_color_sector_id(dag, first)
        rectangle_cardinalities.append(
            {
                "kind": "closure",
                "row_ids": sorted(root.id for root in roots),
                "left_class_size": len(classes[left_class]),
                "right_class_size": len(classes[right_class]),
                "row_count": len(roots),
                "unique_parent_pair_count": len(set(actual_pairs)),
                "expected_parent_pair_count": (
                    len(classes[left_class]) * len(classes[right_class])
                ),
                "resolved_color_sector_id": resolved_sector,
            }
        )
        row_identities.append(
            {
                "kind": "closure",
                "left_class_id": left_class,
                "right_class_id": right_class,
                "root_kind": first.kind,
                "color_weight": list(first.color_weight),
                "contraction_ir": first.contraction_ir.to_json_dict(),
                "resolved_color_sector_id": resolved_sector,
                "vertex_kind": first.vertex_kind,
                "vertex_particles": (
                    None
                    if first.vertex_particles is None
                    else list(first.vertex_particles)
                ),
                "coupling": list(first.coupling),
                "helicity_weight": float(first.helicity_weight),
                "row_ids": sorted(root.id for root in roots),
            }
        )
    return {
        "selector_domains_sha256": _canonical_sha256(
            [list(domain) for domain in selector_domains]
        ),
        "current_class_members_sha256": _canonical_sha256(
            [list(members) for members in classes]
        ),
        "current_remap_sha256": _canonical_sha256(list(class_by_current)),
        "interaction_groups_sha256": _canonical_sha256(
            [sorted(row.id for row in rows) for rows in ordered_interaction_groups]
        ),
        "closure_groups_sha256": _canonical_sha256(
            [sorted(root.id for root in roots) for roots in ordered_root_groups]
        ),
        "rectangle_cardinalities_sha256": _canonical_sha256(
            rectangle_cardinalities
        ),
        "row_identity_sha256": _canonical_sha256(row_identities),
    }


def project_rectangular_dynamic_color_classes(
    dag: GenericDAG,
    model: Model,
    *,
    source_revision: str | None = None,
) -> tuple[GenericDAG, DynamicColorProjectionCertificate]:
    """Project exact dynamic-color rectangles into shared sum currents.

    This is the GenericDAG counterpart of topology-replay value projection.
    It is intentionally conservative:

    * source currents remain singleton;
    * every current field except dynamic LC sector remains in the class key;
    * the downstream physical-selector domain is part of that key;
    * concrete kernel identity, coupling, coefficient, and parent order remain
      exact;
    * both transition and amplitude-closure rows must form complete Cartesian
      products without duplicates.

    A failed rectangle splits every implicated class and retries to a fixed
    point.  Therefore a malformed or incomplete color orbit can only reduce
    optimization, never alter the generated amplitude.
    """

    before = _initial_dynamic_color_projection_certificate(
        dag,
        source_revision=source_revision,
        equality_check_status="not-applicable",
    )
    if (
        dag.process.color_accuracy != "lc"
        or not dag.currents
        or not dag.interactions
        or not dag.amplitude_roots
    ):
        return dag, before
    if dag.helicity_recurrence is not None or dag.helicity_materialization is not None:
        raise ValueError(
            "dynamic-color projection must precede helicity recurrence materialization"
        )

    selector_domains = _downstream_color_selector_domains(dag)
    split_current_ids: set[int] = set()
    split_class_count = 0
    while True:
        classes, class_by_current = _projection_partition(
            dag,
            selector_domains,
            split_current_ids,
        )
        interaction_groups, root_groups = _projection_groups(
            dag,
            classes,
            class_by_current,
        )
        invalid_classes = _invalid_projection_classes(
            dag,
            classes,
            class_by_current,
            interaction_groups,
            root_groups,
        )
        newly_split = {
            current_id
            for class_id in invalid_classes
            for current_id in classes[class_id]
            if current_id not in split_current_ids
        }
        if not newly_split:
            break
        split_class_count += len(invalid_classes)
        split_current_ids.update(newly_split)

    proof_payloads = _projection_proof_payloads(
        dag,
        selector_domains,
        classes,
        class_by_current,
        interaction_groups,
        root_groups,
    )
    projected_class_count = sum(len(members) > 1 for members in classes)
    if projected_class_count == 0:
        return dag, replace(
            before,
            selector_domains_sha256=str(
                proof_payloads["selector_domains_sha256"]
            ),
            current_class_members_sha256=str(
                proof_payloads["current_class_members_sha256"]
            ),
            old_to_new_current_ids=class_by_current,
            current_remap_sha256=str(proof_payloads["current_remap_sha256"]),
            interaction_groups_sha256=str(
                proof_payloads["interaction_groups_sha256"]
            ),
            closure_groups_sha256=str(
                proof_payloads["closure_groups_sha256"]
            ),
            rectangle_cardinalities_sha256=str(
                proof_payloads["rectangle_cardinalities_sha256"]
            ),
            row_identity_sha256=str(proof_payloads["row_identity_sha256"]),
            equality_check_status="passed-no-projectable-classes",
            split_current_class_count=split_class_count,
        )

    old_to_new = class_by_current
    currents = tuple(
        replace(dag.currents[members[0]], id=class_id)
        for class_id, members in enumerate(classes)
    )
    sources = tuple(
        old_to_new[source_id]
        for source_id in dag.sources
    )
    if len(sources) != len(set(sources)):
        raise ValueError("dynamic-color projection unexpectedly merged source currents")

    interactions: list[InteractionNode] = []
    rectangular_interaction_count = 0
    for rows in sorted(
        interaction_groups.values(),
        key=lambda rows: min(row.id for row in rows),
    ):
        first = rows[0]
        involved = (
            old_to_new[first.result_id],
            old_to_new[first.left_id],
            old_to_new[first.right_id],
        )
        if any(len(classes[class_id]) > 1 for class_id in involved):
            rectangular_interaction_count += 1
        interactions.append(
            replace(
                first,
                id=len(interactions),
                left_id=old_to_new[first.left_id],
                right_id=old_to_new[first.right_id],
                result_id=old_to_new[first.result_id],
                evaluation_group_id=None,
                evaluation_factor=(1.0, 0.0),
            )
        )

    roots: list[AmplitudeRoot] = []
    rectangular_closure_count = 0
    for grouped_roots in sorted(
        root_groups.values(),
        key=lambda rows: min(row.id for row in rows),
    ):
        first = grouped_roots[0]
        if (
            len(classes[old_to_new[first.left_id]]) > 1
            or len(classes[old_to_new[first.right_id]]) > 1
        ):
            rectangular_closure_count += 1
        roots.append(
            replace(
                first,
                id=len(roots),
                left_id=old_to_new[first.left_id],
                right_id=old_to_new[first.right_id],
                color_sector_id=_resolved_root_color_sector_id(dag, first),
            )
        )

    projected = replace(
        dag,
        currents=currents,
        sources=sources,
        interactions=tuple(interactions),
        amplitude_roots=tuple(roots),
    )
    projected = assign_recursive_current_evaluation_reuse(projected, model)
    certificate = DynamicColorProjectionCertificate(
        abi=_DYNAMIC_COLOR_PROJECTION_ABI,
        source_revision=source_revision,
        source_semantics_sha256=before.source_semantics_sha256,
        selector_domains_sha256=str(
            proof_payloads["selector_domains_sha256"]
        ),
        current_class_members_sha256=str(
            proof_payloads["current_class_members_sha256"]
        ),
        old_to_new_current_ids=old_to_new,
        current_remap_sha256=str(proof_payloads["current_remap_sha256"]),
        interaction_groups_sha256=str(
            proof_payloads["interaction_groups_sha256"]
        ),
        closure_groups_sha256=str(proof_payloads["closure_groups_sha256"]),
        rectangle_cardinalities_sha256=str(
            proof_payloads["rectangle_cardinalities_sha256"]
        ),
        row_identity_sha256=str(proof_payloads["row_identity_sha256"]),
        equality_check_status="passed-exact-cartesian-products-no-duplicates",
        retained_color_metadata=_RETAINED_COLOR_METADATA_POLICY,
        root_sector_policy=_ROOT_SECTOR_POLICY,
        before_current_count=len(dag.currents),
        after_current_count=len(projected.currents),
        before_interaction_count=len(dag.interactions),
        after_interaction_count=len(projected.interactions),
        before_evaluation_count=dag.interaction_evaluation_count,
        after_evaluation_count=projected.interaction_evaluation_count,
        before_amplitude_root_count=len(dag.amplitude_roots),
        after_amplitude_root_count=len(projected.amplitude_roots),
        projected_current_class_count=projected_class_count,
        rectangular_interaction_group_count=rectangular_interaction_count,
        rectangular_closure_group_count=rectangular_closure_count,
        split_current_class_count=split_class_count,
    )
    return projected, certificate


def _derive_current_value_equivalences(
    dag: GenericDAG,
    model: Model,
    *,
    equivalence_by_kind: dict[int, VertexEvaluationEquivalence] | None = None,
    discovery: _NumericalRelationDiscovery | None = None,
    authenticated_relations: (
        Mapping[int, _CurrentValueEquivalence] | None
    ) = None,
) -> tuple[_CurrentValueEquivalence, ...]:
    """Derive current classes in increasing external-subset order."""

    kernel_equivalences = {} if equivalence_by_kind is None else equivalence_by_kind
    interactions_by_result: dict[int, list[InteractionNode]] = defaultdict(list)
    for interaction in dag.interactions:
        interactions_by_result[interaction.result_id].append(interaction)

    current_equivalences: list[_CurrentValueEquivalence | None] = [None] * len(
        dag.currents
    )
    source_representative_by_key: dict[tuple[object, ...], int] = {}
    equivalence_by_expression: dict[
        tuple[_CurrentContract, _CurrentTermVector], _CurrentValueEquivalence
    ] = {}
    projective_representatives_by_expression: dict[
        tuple[_CurrentContract, _CurrentTermVector],
        list[_ProjectiveExpressionRepresentative],
    ] = {}

    ordered_currents = sorted(
        dag.currents,
        key=lambda current: (len(current.index.external_labels), current.id),
    )
    for current in ordered_currents:
        contract = _current_evaluation_contract(current)
        if current.is_source:
            if current.source_leg_label is None or current.source_helicity is None:
                raise ValueError(
                    f"source current {current.id} lacks physical source metadata"
                )
            source_key = (
                contract,
                int(current.source_leg_label),
                int(current.source_helicity),
            )
            representative_id = source_representative_by_key.setdefault(
                source_key,
                current.id,
            )
            current_equivalences[current.id] = _CurrentValueEquivalence(
                representative_id=representative_id,
                factor=(1.0, 0.0),
            )
            continue

        term_vector = _current_term_vector(
            dag,
            current,
            interactions_by_result[current.id],
            model,
            current_equivalences=current_equivalences,
            equivalence_by_kind=kernel_equivalences,
        )
        equivalence = _classify_current_term_vector(
            current_id=current.id,
            contract=contract,
            term_vector=term_vector,
            equivalence_by_expression=equivalence_by_expression,
            projective_representatives_by_expression=(
                projective_representatives_by_expression
            ),
        )
        if discovery is not None:
            equivalence = discovery.consider(
                current_id=current.id,
                contract=contract,
                term_vector=term_vector,
                baseline=equivalence,
            )
        if authenticated_relations is not None:
            authenticated = authenticated_relations.get(current.id)
            if authenticated is not None:
                if authenticated.representative_id == current.id:
                    if authenticated.factor != (0.0, 0.0):
                        raise ValueError(
                            "self-represented authenticated relation must be zero"
                        )
                    equivalence = authenticated
                else:
                    representative = current_equivalences[
                        authenticated.representative_id
                    ]
                    if representative is None:
                        raise ValueError(
                            "authenticated relation representative is not "
                            "topologically complete"
                        )
                    factor = _exact_representable_complex_product(
                        authenticated.factor,
                        representative.factor,
                    )
                    if factor is None:
                        raise ValueError(
                            "authenticated relation factor composition is not exact"
                        )
                    equivalence = _CurrentValueEquivalence(
                        representative_id=representative.representative_id,
                        factor=factor,
                    )
        current_equivalences[current.id] = equivalence

    if any(item is None for item in current_equivalences):
        raise ValueError(
            "current-value equivalence derivation left an unclassified current"
        )
    return tuple(item for item in current_equivalences if item is not None)


def verify_dag_relation_certificates(
    dag: GenericDAG,
    model: Model,
    certificates: Iterable[ExactCurrentRelationCertificate],
) -> bool:
    """Replay a certificate chain without consulting numerical probe results."""

    ordered_certificates = tuple(certificates)
    if (
        tuple(sorted(ordered_certificates, key=lambda item: item.current_id))
        != ordered_certificates
    ):
        return False
    certificate_by_current = {
        certificate.current_id: certificate for certificate in ordered_certificates
    }
    if len(certificate_by_current) != len(ordered_certificates):
        return False

    try:
        interactions_by_result: dict[int, list[InteractionNode]] = defaultdict(list)
        for interaction in dag.interactions:
            interactions_by_result[interaction.result_id].append(interaction)
        current_equivalences: list[_CurrentValueEquivalence | None] = [None] * len(
            dag.currents
        )
        source_representative_by_key: dict[tuple[object, ...], int] = {}
        equivalence_by_expression: dict[
            tuple[_CurrentContract, _CurrentTermVector], _CurrentValueEquivalence
        ] = {}
        projective_representatives_by_expression: dict[
            tuple[_CurrentContract, _CurrentTermVector],
            list[_ProjectiveExpressionRepresentative],
        ] = {}
        kernel_equivalences: dict[int, VertexEvaluationEquivalence] = {}
        term_vector_by_current: dict[int, _CurrentTermVector] = {}
        contract_by_current: dict[int, _CurrentContract] = {}
        consumed: set[int] = set()

        ordered_currents = sorted(
            dag.currents,
            key=lambda current: (len(current.index.external_labels), current.id),
        )
        for current in ordered_currents:
            contract = _current_evaluation_contract(current)
            contract_by_current[current.id] = contract
            if current.is_source:
                if current.id in certificate_by_current:
                    return False
                if current.source_leg_label is None or current.source_helicity is None:
                    return False
                source_key = (
                    contract,
                    int(current.source_leg_label),
                    int(current.source_helicity),
                )
                representative_id = source_representative_by_key.setdefault(
                    source_key,
                    current.id,
                )
                current_equivalences[current.id] = _CurrentValueEquivalence(
                    representative_id,
                    (1.0, 0.0),
                )
                continue

            term_vector = _current_term_vector(
                dag,
                current,
                interactions_by_result[current.id],
                model,
                current_equivalences=current_equivalences,
                equivalence_by_kind=kernel_equivalences,
            )
            term_vector_by_current[current.id] = term_vector
            equivalence = _classify_current_term_vector(
                current_id=current.id,
                contract=contract,
                term_vector=term_vector,
                equivalence_by_expression=equivalence_by_expression,
                projective_representatives_by_expression=(
                    projective_representatives_by_expression
                ),
            )
            certificate = certificate_by_current.get(current.id)
            if certificate is not None:
                if (
                    certificate.algorithm != RELATION_DISCOVERY_CERTIFICATE_ALGORITHM
                    or certificate.representative_id >= current.id
                    or certificate.representative_id not in term_vector_by_current
                    or contract_by_current[certificate.representative_id] != contract
                ):
                    return False
                representative_vector = term_vector_by_current[
                    certificate.representative_id
                ]
                factor = _certify_exact_term_vector_relation(
                    representative_vector,
                    term_vector,
                )
                if factor is None or not _complex_weight_bits_equal(
                    factor,
                    certificate.factor,
                ):
                    return False
                if (
                    _term_vector_sha256(term_vector)
                    != certificate.current_term_vector_sha256
                    or _term_vector_sha256(representative_vector)
                    != certificate.representative_term_vector_sha256
                    or _certificate_proof_sha256(
                        current_id=certificate.current_id,
                        representative_id=certificate.representative_id,
                        factor=certificate.factor,
                        current_term_vector_sha256=(
                            certificate.current_term_vector_sha256
                        ),
                        representative_term_vector_sha256=(
                            certificate.representative_term_vector_sha256
                        ),
                    )
                    != certificate.proof_sha256
                ):
                    return False
                equivalence = _CurrentValueEquivalence(
                    certificate.representative_id,
                    certificate.factor,
                )
                consumed.add(current.id)
            current_equivalences[current.id] = equivalence
        return consumed == set(certificate_by_current)
    except (IndexError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _numerical_projective_fingerprint(
    term_vector: _CurrentTermVector,
    *,
    precision_digits: int,
    probe_count: int,
    seed: int,
) -> tuple[object, ...] | None:
    """Return a scale-free high-precision candidate key.

    This fingerprint is deliberately insufficient as a proof. Its only use is
    to reduce the number of pairs sent to exact coefficient-vector replay.
    """

    if not _term_vector_is_finite_nonzero(term_vector):
        return None
    with localcontext() as context:
        context.prec = precision_digits
        values: list[tuple[Decimal, Decimal]] = []
        for probe_index in range(probe_count):
            total_real = Decimal(0)
            total_imaginary = Decimal(0)
            for evaluation_key, coefficient in term_vector:
                probe_real, probe_imaginary = _deterministic_decimal_probe_value(
                    evaluation_key,
                    seed=seed,
                    probe_index=probe_index,
                )
                coefficient_real = Decimal.from_float(coefficient[0])
                coefficient_imaginary = Decimal.from_float(coefficient[1])
                total_real += (
                    coefficient_real * probe_real
                    - coefficient_imaginary * probe_imaginary
                )
                total_imaginary += (
                    coefficient_real * probe_imaginary
                    + coefficient_imaginary * probe_real
                )
            values.append((total_real, total_imaginary))

        pivot_index = next(
            (
                index
                for index, value in enumerate(values)
                if value != (Decimal(0), Decimal(0))
            ),
            None,
        )
        if pivot_index is None:
            return None
        pivot = values[pivot_index]
        normalized = tuple(_decimal_complex_ratio(value, pivot) for value in values)
        if any(value is None for value in normalized):
            return None
        fingerprint_digits = max(24, precision_digits // 2)
        return (
            "deterministic-projective-probe-v1",
            pivot_index,
            tuple(
                (
                    _decimal_fingerprint_component(value[0], fingerprint_digits),
                    _decimal_fingerprint_component(value[1], fingerprint_digits),
                )
                for value in normalized
                if value is not None
            ),
        )


def _deterministic_decimal_probe_value(
    evaluation_key: _EvaluationKey,
    *,
    seed: int,
    probe_index: int,
) -> tuple[Decimal, Decimal]:
    payload = json.dumps(
        _canonical_proof_value(evaluation_key),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(f"{seed}:{probe_index}:".encode("ascii") + payload).digest()

    def component(chunk: bytes) -> Decimal:
        raw = int.from_bytes(chunk, "big", signed=False)
        value = raw % 2_000_003 - 1_000_001
        return Decimal(1 if value == 0 else value)

    return component(digest[:8]), component(digest[8:16])


def _decimal_complex_ratio(
    numerator: tuple[Decimal, Decimal],
    denominator: tuple[Decimal, Decimal],
) -> tuple[Decimal, Decimal] | None:
    denominator_norm = denominator[0] * denominator[0] + denominator[1] * denominator[1]
    if denominator_norm == 0:
        return None
    return (
        (numerator[0] * denominator[0] + numerator[1] * denominator[1])
        / denominator_norm,
        (numerator[1] * denominator[0] - numerator[0] * denominator[1])
        / denominator_norm,
    )


def _decimal_fingerprint_component(value: Decimal, digits: int) -> str:
    if value == 0:
        return "0"
    return format(value, f".{digits}E")


def _certify_exact_term_vector_relation(
    representative: _CurrentTermVector,
    candidate: _CurrentTermVector,
) -> _ComplexWeight | None:
    if (
        not _term_vector_is_finite_nonzero(representative)
        or not _term_vector_is_finite_nonzero(candidate)
        or len(representative) != len(candidate)
    ):
        return None
    if any(
        representative_key != candidate_key
        for (representative_key, _), (candidate_key, _) in zip(
            representative,
            candidate,
            strict=True,
        )
    ):
        return None
    factor = _exact_representable_complex_ratio(
        candidate[0][1],
        representative[0][1],
    )
    if factor is None or not _term_vector_scaled_exactly(
        representative,
        factor,
        candidate,
    ):
        return None
    return factor


def _build_exact_current_relation_certificate(
    *,
    current_id: int,
    representative_id: int,
    factor: _ComplexWeight,
    current_term_vector: _CurrentTermVector,
    representative_term_vector: _CurrentTermVector,
) -> ExactCurrentRelationCertificate:
    current_digest = _term_vector_sha256(current_term_vector)
    representative_digest = _term_vector_sha256(representative_term_vector)
    return ExactCurrentRelationCertificate(
        current_id=current_id,
        representative_id=representative_id,
        factor=factor,
        current_term_vector_sha256=current_digest,
        representative_term_vector_sha256=representative_digest,
        proof_sha256=_certificate_proof_sha256(
            current_id=current_id,
            representative_id=representative_id,
            factor=factor,
            current_term_vector_sha256=current_digest,
            representative_term_vector_sha256=representative_digest,
        ),
    )


def _certificate_proof_sha256(
    *,
    current_id: int,
    representative_id: int,
    factor: _ComplexWeight,
    current_term_vector_sha256: str,
    representative_term_vector_sha256: str,
) -> str:
    return _canonical_payload_sha256(
        {
            "algorithm": RELATION_DISCOVERY_CERTIFICATE_ALGORITHM,
            "current_id": current_id,
            "representative_id": representative_id,
            "factor_binary64": [factor[0].hex(), factor[1].hex()],
            "current_term_vector_sha256": current_term_vector_sha256,
            "representative_term_vector_sha256": (representative_term_vector_sha256),
        }
    )


def _term_vector_sha256(term_vector: _CurrentTermVector) -> str:
    return _canonical_payload_sha256(
        [
            {
                "evaluation_key": _canonical_proof_value(evaluation_key),
                "coefficient_binary64": [
                    coefficient[0].hex(),
                    coefficient[1].hex(),
                ],
            }
            for evaluation_key, coefficient in term_vector
        ]
    )


def _canonical_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_proof_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_canonical_proof_value(item) for item in value]
    if isinstance(value, float):
        return {"binary64": value.hex()}
    if value is None or isinstance(value, bool | int | str):
        return value
    raise TypeError(
        "relation certificate contains a noncanonical evaluation-key value "
        f"{type(value).__name__}"
    )


def _interaction_reuse_signature(dag: GenericDAG) -> tuple[tuple[object, ...], ...]:
    canonical_group_by_key: dict[tuple[str, int], int] = {}
    signature: list[tuple[object, ...]] = []
    for interaction in dag.interactions:
        raw_group = (
            ("group", interaction.evaluation_group_id)
            if interaction.evaluation_group_id is not None
            else ("interaction", interaction.id)
        )
        canonical_group = canonical_group_by_key.setdefault(
            raw_group,
            len(canonical_group_by_key),
        )
        signature.append(
            (
                canonical_group,
                interaction.evaluation_factor[0].hex(),
                interaction.evaluation_factor[1].hex(),
            )
        )
    return tuple(signature)


def _classify_current_term_vector(
    *,
    current_id: int,
    contract: _CurrentContract,
    term_vector: _CurrentTermVector,
    equivalence_by_expression: dict[
        tuple[_CurrentContract, _CurrentTermVector], _CurrentValueEquivalence
    ],
    projective_representatives_by_expression: dict[
        tuple[_CurrentContract, _CurrentTermVector],
        list[_ProjectiveExpressionRepresentative],
    ],
) -> _CurrentValueEquivalence:
    """Return the earliest exactly proven representative for one current.

    Exact equality and sign reuse remain the fast paths.  More general
    projective reuse is accepted only when a finite, nonzero binary64 factor
    exists exactly and applying it reproduces every coefficient bit-for-bit.
    """

    identity = _CurrentValueEquivalence(current_id, (1.0, 0.0))
    expression_key = (contract, term_vector)
    if not term_vector:
        exact_zero = equivalence_by_expression.get(expression_key)
        if exact_zero is not None:
            return exact_zero
        equivalence_by_expression[expression_key] = identity
        return identity
    if not _term_vector_is_finite_nonzero(term_vector):
        return identity

    exact = equivalence_by_expression.get(expression_key)
    if exact is not None:
        return exact

    opposite = equivalence_by_expression.get(
        (contract, _negate_term_vector(term_vector))
    )
    if opposite is not None:
        opposite_factor = (
            _canonical_zero(-opposite.factor[0]),
            _canonical_zero(-opposite.factor[1]),
        )
        representative_vector = _term_vector_divided_by_exact_factor(
            term_vector,
            opposite_factor,
        )
        if representative_vector is not None and _term_vector_scaled_exactly(
            representative_vector,
            opposite_factor,
            term_vector,
        ):
            equivalence = _CurrentValueEquivalence(
                opposite.representative_id,
                opposite_factor,
            )
            equivalence_by_expression[expression_key] = equivalence
            return equivalence

    canonical = _canonicalize_projective_term_vector(term_vector)
    if canonical is None:
        equivalence_by_expression[expression_key] = identity
        return identity

    projective_key = (contract, canonical.term_vector)
    representatives = projective_representatives_by_expression.get(projective_key)
    matches: list[_CurrentValueEquivalence] = []
    if representatives is not None:
        for representative in representatives:
            factor = _exact_representable_complex_ratio(
                canonical.factor,
                representative.normalization_factor,
            )
            if factor is None or not _term_vector_scaled_exactly(
                representative.term_vector,
                factor,
                term_vector,
            ):
                continue
            matches.append(
                _CurrentValueEquivalence(
                    representative.representative_id,
                    factor,
                )
            )

    if len(matches) == 1:
        equivalence = matches[0]
        equivalence_by_expression[expression_key] = equivalence
        return equivalence

    # Zero matches mean that the normalized lookup key was only a rounded
    # collision.  Multiple matches make the concrete representative
    # ambiguous.  Both cases fail closed by retaining this current.
    projective_representatives_by_expression.setdefault(
        projective_key,
        [],
    ).append(
        _ProjectiveExpressionRepresentative(
            representative_id=current_id,
            term_vector=term_vector,
            normalization_factor=canonical.factor,
        )
    )
    equivalence_by_expression[expression_key] = identity
    return identity


def _canonicalize_projective_term_vector(
    term_vector: _CurrentTermVector,
) -> _CanonicalProjectiveTermVector | None:
    """Normalize a finite nonzero vector for projective-class lookup.

    The first canonical term is the deterministic pivot.  The normalized
    binary64 coefficients are only an index: every accepted equivalence is
    separately proven with an exactly representable factor and bit-exact
    coefficient reconstruction.
    """

    if not term_vector or not _term_vector_is_finite_nonzero(term_vector):
        return None
    keys = tuple(key for key, _coefficient in term_vector)
    if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
        return None

    factor = _canonical_complex_weight(term_vector[0][1])
    normalized_terms: list[tuple[_EvaluationKey, _ComplexWeight]] = []
    for key, raw_coefficient in term_vector:
        coefficient = _canonical_complex_weight(raw_coefficient)
        normalized = _roundtrip_complex_ratio(coefficient, factor)
        if normalized is None:
            return None
        normalized_terms.append((key, normalized))
    return _CanonicalProjectiveTermVector(
        term_vector=tuple(normalized_terms),
        factor=factor,
    )


def _roundtrip_complex_ratio(
    numerator: _ComplexWeight,
    denominator: _ComplexWeight,
) -> _ComplexWeight | None:
    """Return a finite lookup ratio that reconstructs ``numerator`` exactly."""

    if (
        not _complex_weight_is_finite(numerator)
        or not _complex_weight_is_finite(denominator)
        or denominator == (0.0, 0.0)
    ):
        return None
    try:
        quotient = complex(*numerator) / complex(*denominator)
    except (OverflowError, ZeroDivisionError):
        return None
    result = _canonical_complex_weight((quotient.real, quotient.imag))
    if not _complex_weight_is_finite(result):
        return None
    reconstructed = _canonical_complex_weight(_complex_weight_mul(denominator, result))
    if not _complex_weight_bits_equal(reconstructed, numerator):
        return None
    return result


def _exact_representable_complex_ratio(
    numerator: _ComplexWeight,
    denominator: _ComplexWeight,
) -> _ComplexWeight | None:
    """Return the exact binary64 complex quotient, or fail closed.

    ``Fraction`` is used only after two vectors collide on their inexpensive
    normalized key.  It proves that both quotient components are exactly
    representable rather than merely rounded values that happen to be close.
    """

    if (
        not _complex_weight_is_finite(numerator)
        or not _complex_weight_is_finite(denominator)
        or denominator == (0.0, 0.0)
    ):
        return None
    numerator_real = Fraction.from_float(numerator[0])
    numerator_imag = Fraction.from_float(numerator[1])
    denominator_real = Fraction.from_float(denominator[0])
    denominator_imag = Fraction.from_float(denominator[1])
    denominator_norm = (
        denominator_real * denominator_real + denominator_imag * denominator_imag
    )
    if denominator_norm == 0:
        return None
    real = (
        numerator_real * denominator_real + numerator_imag * denominator_imag
    ) / denominator_norm
    imaginary = (
        numerator_imag * denominator_real - numerator_real * denominator_imag
    ) / denominator_norm
    real_f64 = _exact_fraction_as_f64(real)
    imaginary_f64 = _exact_fraction_as_f64(imaginary)
    if real_f64 is None or imaginary_f64 is None:
        return None
    factor = (real_f64, imaginary_f64)
    reconstructed = _canonical_complex_weight(_complex_weight_mul(denominator, factor))
    if not _complex_weight_bits_equal(
        reconstructed,
        _canonical_complex_weight(numerator),
    ):
        return None
    return factor


def _exact_representable_complex_product(
    left: _ComplexWeight,
    right: _ComplexWeight,
) -> _ComplexWeight | None:
    """Return the exact binary64 complex product, or fail closed."""

    if not _complex_weight_is_finite(left) or not _complex_weight_is_finite(right):
        return None
    left_real = Fraction.from_float(left[0])
    left_imaginary = Fraction.from_float(left[1])
    right_real = Fraction.from_float(right[0])
    right_imaginary = Fraction.from_float(right[1])
    real = left_real * right_real - left_imaginary * right_imaginary
    imaginary = left_real * right_imaginary + left_imaginary * right_real
    real_f64 = _exact_fraction_as_f64(real)
    imaginary_f64 = _exact_fraction_as_f64(imaginary)
    if real_f64 is None or imaginary_f64 is None:
        return None
    product = (real_f64, imaginary_f64)
    reconstructed = _canonical_complex_weight(_complex_weight_mul(left, right))
    if not _complex_weight_bits_equal(reconstructed, product):
        return None
    return product


def _exact_fraction_as_f64(value: Fraction) -> float | None:
    try:
        result = float(value)
    except OverflowError:
        return None
    if not isfinite(result) or Fraction.from_float(result) != value:
        return None
    return _canonical_zero(result)


def _term_vector_scaled_exactly(
    representative: _CurrentTermVector,
    factor: _ComplexWeight,
    candidate: _CurrentTermVector,
) -> bool:
    if (
        not _complex_weight_is_finite(factor)
        or factor == (0.0, 0.0)
        or len(representative) != len(candidate)
    ):
        return False
    for (representative_key, representative_value), (
        candidate_key,
        candidate_value,
    ) in zip(representative, candidate, strict=True):
        if representative_key != candidate_key:
            return False
        scaled = _canonical_complex_weight(
            _complex_weight_mul(factor, representative_value)
        )
        if not _complex_weight_bits_equal(
            scaled,
            _canonical_complex_weight(candidate_value),
        ):
            return False
    return True


def _term_vector_divided_by_exact_factor(
    candidate: _CurrentTermVector,
    factor: _ComplexWeight,
) -> _CurrentTermVector | None:
    """Recover a representative vector only for the sign fast path."""

    if factor not in ((1.0, 0.0), (-1.0, 0.0)):
        return None
    return tuple(
        (
            key,
            value
            if factor == (1.0, 0.0)
            else (_canonical_zero(-value[0]), _canonical_zero(-value[1])),
        )
        for key, value in candidate
    )


def _term_vector_is_finite_nonzero(term_vector: _CurrentTermVector) -> bool:
    return bool(term_vector) and all(
        coefficient != (0.0, 0.0) and _complex_weight_is_finite(coefficient)
        for _key, coefficient in term_vector
    )


def _complex_weight_is_finite(value: _ComplexWeight) -> bool:
    return isfinite(value[0]) and isfinite(value[1])


def _canonical_complex_weight(value: _ComplexWeight) -> _ComplexWeight:
    return (_canonical_zero(value[0]), _canonical_zero(value[1]))


def _complex_weight_bits_equal(
    left: _ComplexWeight,
    right: _ComplexWeight,
) -> bool:
    return left[0].hex() == right[0].hex() and left[1].hex() == right[1].hex()


def _current_evaluation_contract(current: CurrentNode) -> _CurrentContract:
    """Return fields that can affect source, kernel, or propagator values."""

    index = current.index
    return (
        int(index.particle_id),
        int(index.external_mask),
        index.external_labels,
        index.ordered_external_labels,
        int(index.helicity_ancestry),
        int(index.chirality),
        index.spin_state,
        index.flavour_flow,
        index.quantum_number_flow,
        int(index.momentum_mask),
        index.coupling_orders,
        index.auxiliary_kind,
        int(current.dimension),
        bool(current.is_source),
    )


def _current_term_vector(
    dag: GenericDAG,
    current: CurrentNode,
    interactions: list[InteractionNode],
    model: Model,
    *,
    current_equivalences: list[_CurrentValueEquivalence | None],
    equivalence_by_kind: dict[int, VertexEvaluationEquivalence],
) -> _CurrentTermVector:
    coefficients_by_key: dict[_EvaluationKey, list[_ComplexWeight]] = defaultdict(list)
    for interaction in interactions:
        kernel_equivalence = _kernel_equivalence(
            model,
            interaction.vertex_kind,
            equivalence_by_kind,
        )
        left = current_equivalences[interaction.left_id]
        right = current_equivalences[interaction.right_id]
        if left is None or right is None:
            raise ValueError(
                "current-value equivalence requires parents from an earlier subset"
            )
        canonical_inputs, evaluation_factor = _canonical_interaction_evaluation(
            kernel_equivalence,
            left_id=interaction.left_id,
            right_id=interaction.right_id,
            left=left,
            right=right,
        )
        term_key = (
            kernel_equivalence.class_id,
            canonical_inputs,
            int(current.index.particle_id),
            int(current.index.chirality),
            _runtime_coupling_identity(
                model,
                vertex_kind=interaction.vertex_kind,
                vertex_particles=interaction.vertex_particles,
                coupling=interaction.coupling,
            ),
        )
        coefficient = _complex_weight_mul(
            interaction.color_weight,
            evaluation_factor,
        )
        coefficients_by_key[term_key].append(coefficient)

    terms: list[tuple[_EvaluationKey, _ComplexWeight]] = []
    for grouped_term_key in sorted(coefficients_by_key):
        coefficients = coefficients_by_key[grouped_term_key]
        coefficient = (
            _canonical_zero(fsum(value[0] for value in coefficients)),
            _canonical_zero(fsum(value[1] for value in coefficients)),
        )
        if coefficient != (0.0, 0.0):
            terms.append((grouped_term_key, coefficient))
    return tuple(terms)


def _kernel_equivalence(
    model: Model,
    kind: int,
    cache: dict[int, VertexEvaluationEquivalence],
) -> VertexEvaluationEquivalence:
    cached = cache.get(kind)
    if cached is not None:
        return cached
    equivalence = model.vertex_evaluation_equivalence(kind)
    if not equivalence.verified:
        model_type = f"{type(model).__module__}.{type(model).__qualname__}"
        equivalence = VertexEvaluationEquivalence(class_id=f"{model_type}:{int(kind)}")
    cache[kind] = equivalence
    return equivalence


def _runtime_coupling_identity(
    model: Model,
    *,
    vertex_kind: int,
    vertex_particles: tuple[int, int, int],
    coupling: tuple[float, float],
) -> tuple[tuple[float, float], tuple[tuple[int, str], ...]]:
    """Return defaults plus stable mutable-parameter provenance for reuse."""

    names = runtime_coupling_parameter_names(
        vertex_kind,
        vertex_particles,
        coupling,
        model=model,
    )
    provenance = tuple((0, "") if name is None else (1, str(name)) for name in names)
    return coupling, provenance


def _negate_term_vector(vector: _CurrentTermVector) -> _CurrentTermVector:
    return tuple((key, (-value[0], -value[1])) for key, value in vector)


def _complex_weight_mul(
    left: _ComplexWeight,
    right: _ComplexWeight,
) -> _ComplexWeight:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _canonical_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


__all__ = [
    "GENERIC_DAG_NUMERICAL_SOURCE_SEMANTICS_ABI",
    "NUMERICAL_CURRENT_CAPTURE_ABI",
    "NUMERICAL_CURRENT_CAPTURE_OUTPUT_PARTITION_ABI",
    "NUMERICAL_CURRENT_RELATION_CERTIFICATE_ALGORITHM",
    "NUMERICAL_CURRENT_RELATION_SET_ABI",
    "NUMERICAL_CURRENT_RELATION_WARNING",
    "NUMERICAL_CURRENT_RELATION_WARNING_CODE",
    "RELATION_DISCOVERY_CERTIFICATE_ALGORITHM",
    "RELATION_DISCOVERY_SCHEMA_VERSION",
    "DynamicColorProjectionCertificate",
    "ExactCurrentRelationCertificate",
    "NumericalCurrentAppliedMapping",
    "NumericalCurrentObservationDiscoveryReport",
    "NumericalCurrentObservationDiscoveryResult",
    "NumericalCurrentRelationApplicationReport",
    "NumericalCurrentRelationApplicationResult",
    "NumericalCurrentRelationCertificate",
    "RecursiveEvaluationReuseTracker",
    "RelationDiscoveryReport",
    "RelationDiscoveryResult",
    "apply_numerical_current_relation_certificates",
    "assign_recursive_current_evaluation_reuse",
    "certify_numerical_current_observations",
    "discover_generic_dag_numerical_current_relations",
    "discover_recursive_evaluation_relations",
    "dynamic_color_projection_not_applicable_report",
    "generic_dag_numerical_capture_output_partition_sha256",
    "generic_dag_numerical_model_parameter_schema_sha256",
    "generic_dag_numerical_runtime_schema_sha256",
    "generic_dag_numerical_source_dag_sha256",
    "generic_dag_numerical_source_semantics_sha256",
    "project_rectangular_dynamic_color_classes",
    "verify_dag_relation_certificates",
    "verify_numerical_current_relation_certificate",
]
