# SPDX-License-Identifier: 0BSD
"""Authenticated high-precision current reuse for recurrence Direct plans.

The first native lowering is an unpublished baseline.  This module evaluates
that exact plan at disjoint Decimal/Symbolica probe domains, certifies only
equal, opposite, and zero relations, and emits canonical evidence for a
second native lowering.  No binary64 runtime result participates in discovery.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import struct
import tempfile
import time
import zlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from math import copysign, isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, BinaryIO, Literal, cast

from pyamplicol.api.errors import ArtifactError
from pyamplicol.models.base import Model
from pyamplicol.runtime.recurrence_exact._execution import (
    _evaluate_contracted_point,
    _evaluate_replay_point,
    _evaluate_union_point,
)
from pyamplicol.runtime.recurrence_exact._plan import _RecurrenceExactPlan
from pyamplicol.runtime.recurrence_exact._plan_v2 import (
    DIRECT_NONE_U32,
    _Current,
    _RecurrenceExactSectionsV1,
)
from pyamplicol.runtime.symbolica_exact import _ComplexDecimal

from .dag_equivalence import (
    NUMERICAL_CURRENT_RELATION_WARNING,
    NUMERICAL_CURRENT_RELATION_WARNING_CODE,
)
from .numerical_candidate_index import (
    NumericalObservationCandidateIndex,
    build_numerical_observation_candidate_index,
    numerical_observation_tolerance_window_ids,
)
from .validation import (
    ValidationPointRecord,
    build_process_validation_point,
    rotate_validation_point,
)

if TYPE_CHECKING:
    from pyamplicol.processes.ir import CanonicalProcessIR

_CAPTURE_ABI = "pyamplicol-recurrence-current-observation-capture-v2"
_EVIDENCE_ABI = "pyamplicol-recurrence-numerical-current-evidence-v3"
_SOURCE_ABI = "pyamplicol-recurrence-numerical-current-source-v3"
_WARMUP_ABI = "pyamplicol-recurrence-numerical-current-warmup-v1"
_RELATION_SET_ABI = "pyamplicol-authenticated-numerical-current-relation-set-v2"
_OBSERVATION_BATCH_ABI = "pyamplicol-recurrence-current-observation-batch-v2"
_PARAMETER_CONTEXT_ABI = "pyamplicol-recurrence-parameter-context-v1"
_PARAMETER_SCHEMA_ABI = "pyamplicol-recurrence-runtime-parameter-schema-v2"
_RELATION_OBSERVATION_ABI = "pyamplicol-recurrence-relation-observation-v2"
_DECISION_CHAIN_ABI = "pyamplicol-recurrence-numerical-decision-chain-v1"
_REJECTION_CHAIN_ABI = "pyamplicol-recurrence-rejected-numerical-decision-chain-v1"
_PERSISTED_EVIDENCE_ABI = "pyamplicol-recurrence-numerical-persisted-evidence-v1"
# Select a sequential spool for large captures; this is not a memory ceiling.
_RAW_EVIDENCE_MEMORY_TARGET_BYTES = 1 << 30
_COMPRESSED_EVIDENCE_MAGIC = b"PACNCEZ1"
_COMPRESSED_EVIDENCE_HEADER = struct.Struct(">8sQ32s")
_COMPRESSED_EVIDENCE_ENCODING = "zlib-canonical-json-v1"
_RAW_EVIDENCE_ENCODING = "canonical-json-v3"
_SPOOLED_CAPTURE_COMPRESSION_RESERVE_BYTES = 8 << 20
_SPOOLED_CANDIDATE_INDEX_BYTES_PER_CURRENT = 1_024
# The producer-side peak ends when the canonical bytes have been encoded and
# the complete Decimal graphs are detached.  The per-scalar allowance covers
# the Python Decimal/tuple capture graph and encoder temporaries; the row
# allowance covers capture dictionaries and rows. These estimates select
# resident or spooled storage; native validates the observations independently
# without imposing a fixed memory ceiling.
_RAW_EVIDENCE_SCALAR_RESIDENT_BYTES = 640
_RAW_EVIDENCE_ROW_RESIDENT_BYTES = 512
_RAW_EVIDENCE_FIXED_RESERVE_BYTES = 32 << 20
_RAW_EVIDENCE_WIRE_PEAK_COPIES = 2
_MIN_RAW_EVIDENCE_WIRE_BYTES = 1 << 20
_RECURRENCE_CERTIFICATE_ALGORITHM = (
    "authenticated-independent-recursive-decimal-raw-probes-v2"
)
_MAX_REJECTED_DIAGNOSTICS = 32
_CANDIDATE_INDEX_ALGORITHM = "complete-contract-anchor-tolerance-window-v1"
_APPLICATION_SCOPE_ABI = "pyamplicol-recurrence-numerical-relation-application-scope-v1"
NUMERICAL_RELATION_CORRECTNESS_ABI = (
    "pyamplicol-numerical-current-relation-correctness-v1"
)
NUMERICAL_RELATION_FALLBACK_ABI = "pyamplicol-numerical-current-reuse-fallback-v1"
EVIDENCE_ENVELOPE_FALLBACK_REASON = "evidence-envelope-fallback"
EVIDENCE_ENVELOPE_FALLBACK_WARNING = (
    "Numerical-current reuse was disabled because its optional evidence "
    "geometry exceeds the bounded memory envelope; recurrence generation "
    "continues without numerical relations."
)

_RelationKind = Literal["equal", "opposite", "zero"]
_Mode = Literal["diagnostic", "certified-reuse"]
_Point = tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]
_CurrentContract = tuple[object, ...]
_EvidenceEncoding = Literal["canonical-json-v3", "zlib-canonical-json-v1"]
_Residuals = tuple[Fraction, Fraction, Fraction]
_ResidualStrings = tuple[str, str, str]
_Tolerance = Decimal | Fraction


@dataclass(frozen=True, slots=True)
class _RawEvidenceStorageGeometry:
    scalar_count: int
    row_count: int
    canonical_byte_limit: int | None
    encoding: _EvidenceEncoding
    producer_resident_upper_bound: int


@dataclass(frozen=True, slots=True)
class RecurrenceNumericalEvidenceGeometry:
    """Allocation-free shape of one optional numerical-evidence capture."""

    current_count: int
    component_count: int
    candidate_probe_count: int
    verification_probe_count: int
    runtime_parameter_count: int
    scalar_count: int
    row_count: int

    @classmethod
    def from_counts(
        cls,
        *,
        current_count: int,
        component_count: int,
        candidate_probe_count: int,
        verification_probe_count: int,
        runtime_parameter_count: int,
    ) -> RecurrenceNumericalEvidenceGeometry:
        values = (
            current_count,
            component_count,
            candidate_probe_count,
            verification_probe_count,
            runtime_parameter_count,
        )
        if (
            any(type(value) is not int or value < 0 for value in values)
            or candidate_probe_count == 0
            or verification_probe_count == 0
        ):
            raise ValueError("recurrence raw-evidence probe geometry is invalid")
        point_count = candidate_probe_count + verification_probe_count
        return cls(
            current_count=current_count,
            component_count=component_count,
            candidate_probe_count=candidate_probe_count,
            verification_probe_count=verification_probe_count,
            runtime_parameter_count=runtime_parameter_count,
            scalar_count=(
                2 * component_count * point_count
                + runtime_parameter_count * point_count
            ),
            row_count=current_count * 2 + runtime_parameter_count,
        )

    def to_json_dict(self) -> dict[str, int]:
        return {
            "current_count": self.current_count,
            "component_count": self.component_count,
            "candidate_probe_count": self.candidate_probe_count,
            "verification_probe_count": self.verification_probe_count,
            "runtime_parameter_count": self.runtime_parameter_count,
            "scalar_count": self.scalar_count,
            "row_count": self.row_count,
        }

    def spooled_producer_resident_upper_bound(self) -> int:
        """Recompute the sequential-spool producer bound for this geometry."""

        return _spooled_capture_memory_upper_bound(
            current_count=self.current_count,
            component_count=self.component_count,
            maximum_probe_count=max(
                self.candidate_probe_count,
                self.verification_probe_count,
            ),
            runtime_parameter_count=self.runtime_parameter_count,
        )


class RecurrenceNumericalEvidenceEnvelopeExceeded(ValueError):
    """Optional recurrence evidence cannot fit its authenticated envelope."""

    def __init__(
        self,
        geometry: RecurrenceNumericalEvidenceGeometry,
        *,
        memory_envelope_bytes: int,
        spooled_producer_resident_bytes: int,
        raw_reason: str,
    ) -> None:
        self.geometry = geometry
        self.memory_envelope_bytes = memory_envelope_bytes
        self.spooled_producer_resident_bytes = spooled_producer_resident_bytes
        self.raw_reason = raw_reason
        values = geometry.to_json_dict()
        super().__init__(
            "recurrence optional numerical evidence cannot fit its bounded "
            "resident or transport envelope "
            f"(currents={values['current_count']}, "
            f"components={values['component_count']}, "
            f"candidate_points={values['candidate_probe_count']}, "
            f"verification_points={values['verification_probe_count']}, "
            f"runtime_parameters={values['runtime_parameter_count']}, "
            f"scalars={values['scalar_count']}, rows={values['row_count']}, "
            "spooled_producer_resident="
            f"{spooled_producer_resident_bytes}, raw_reason={raw_reason})"
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "reason": EVIDENCE_ENVELOPE_FALLBACK_REASON,
            "memory_envelope_bytes": self.memory_envelope_bytes,
            "spooled_producer_resident_bytes": (self.spooled_producer_resident_bytes),
            "geometry": self.geometry.to_json_dict(),
        }


class _RawEvidenceWireEnvelopeExceeded(ValueError):
    """The validated raw-wire shape cannot fit its bounded envelope."""


class _CanonicalEvidenceCapacityExceeded(ValueError):
    """A valid optional-evidence encoding cannot fit its bounded transport."""


class _IndependentVerificationRequired(Exception):
    """Candidate screening found a hypothesis that needs independent probes."""


_NATIVE_OPTIONAL_EVIDENCE_CAPACITY_MARKERS = (
    "raw numerical evidence exceeds the explicit pre-metadata 1 GiB resident "
    "memory envelope",
    "compressed numerical evidence exceeds the explicit pre-shape 1 GiB resident "
    "memory envelope",
    "compressed numerical relation evidence exceeds the explicit native 1 GiB "
    "resident memory envelope",
    "raw numerical evidence exceeds the streaming 1 GiB resident memory envelope",
    "raw numerical capture geometry leaves no minimum canonical wire reserve "
    "inside the 1 GiB resident memory envelope",
    "raw numerical sequential-spool capture geometry exceeds the explicit 1 GiB "
    "resident memory envelope",
    "could not reserve bounded decompressed numerical evidence",
)


@dataclass(frozen=True, slots=True)
class _WarmupStaticContext:
    current_dimensions: Mapping[int, int]
    current_contracts: tuple[_CurrentContract, ...]
    runtime_defaults: tuple[Decimal, ...]
    runtime_parameter_schema: Mapping[str, object]
    runtime_parameter_schema_sha256: str
    source_semantics: Mapping[str, object]
    source_semantics_sha256: str


class _SpooledObservationMapping(Mapping[int, tuple[_ComplexDecimal, ...]]):
    """Trusted temporary row store with bounded one-row materialization."""

    def __init__(
        self,
        observations: Mapping[int, Sequence[_ComplexDecimal]],
        *,
        candidate_indexes: Mapping[
            _CurrentContract,
            NumericalObservationCandidateIndex[Fraction],
        ]
        | None = None,
    ) -> None:
        self._stream: BinaryIO = tempfile.TemporaryFile(  # noqa: SIM115
            prefix="pyamplicol-recurrence-current-observations-",
        )
        self._rows: dict[int, tuple[int, int]] = {}
        self._candidate_indexes_row: tuple[int, int] | None = None
        self._closed = False
        try:
            for current_id in sorted(observations):
                offset = self._stream.tell()
                pickle.dump(
                    tuple(observations[current_id]),
                    self._stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                self._rows[current_id] = (
                    offset,
                    self._stream.tell() - offset,
                )
            if candidate_indexes is not None:
                offset = self._stream.tell()
                pickle.dump(
                    dict(candidate_indexes),
                    self._stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                self._candidate_indexes_row = (
                    offset,
                    self._stream.tell() - offset,
                )
            self._stream.flush()
        except BaseException:
            self.close()
            raise

    @property
    def candidate_indexes(
        self,
    ) -> (
        Mapping[
            _CurrentContract,
            NumericalObservationCandidateIndex[Fraction],
        ]
        | None
    ):
        if self._candidate_indexes_row is None:
            return None
        if self._closed:
            raise RuntimeError("recurrence observation spool is closed")
        offset, length = self._candidate_indexes_row
        self._stream.seek(offset)
        payload = self._stream.read(length)
        if len(payload) != length:
            raise ValueError("recurrence candidate-index spool row is truncated")
        return cast(
            dict[
                _CurrentContract,
                NumericalObservationCandidateIndex[Fraction],
            ],
            pickle.loads(payload),
        )

    def __getitem__(self, current_id: int) -> tuple[_ComplexDecimal, ...]:
        if self._closed:
            raise RuntimeError("recurrence observation spool is closed")
        try:
            offset, length = self._rows[current_id]
        except KeyError:
            raise KeyError(current_id) from None
        self._stream.seek(offset)
        payload = self._stream.read(length)
        if len(payload) != length:
            raise ValueError("recurrence observation spool row is truncated")
        values = pickle.loads(payload)
        if not isinstance(values, tuple):
            raise ValueError("recurrence observation spool row is malformed")
        return cast(tuple[_ComplexDecimal, ...], values)

    def __iter__(self) -> Iterator[int]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream.close()


class _CanonicalJsonStreamValue:
    """A value that writes canonical JSON without first building a JSON graph."""

    def iter_canonical_json_chunks(self) -> Iterator[bytes]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _ObservationValuesStream(_CanonicalJsonStreamValue):
    values: Sequence[_ComplexDecimal]

    def iter_canonical_json_chunks(self) -> Iterator[bytes]:
        yield b"["
        for index, (real, imaginary) in enumerate(self.values):
            if index:
                yield b","
            yield b"["
            yield from _iter_canonical_json_chunks(_decimal_string(real))
            yield b","
            yield from _iter_canonical_json_chunks(_decimal_string(imaginary))
            yield b"]"
        yield b"]"


@dataclass(frozen=True, slots=True)
class _ObservationRowsStream(_CanonicalJsonStreamValue):
    observations: Mapping[int, Sequence[_ComplexDecimal]]
    dimensions: Mapping[int, int]

    def iter_canonical_json_chunks(self) -> Iterator[bytes]:
        yield b"["
        for index, current_id in enumerate(sorted(self.observations)):
            if index:
                yield b","
            yield from _iter_canonical_json_chunks(
                {
                    "current_id": current_id,
                    "dimension": self.dimensions[current_id],
                    "values": _ObservationValuesStream(self.observations[current_id]),
                }
            )
        yield b"]"


@dataclass(frozen=True, slots=True)
class _DimensionRowsStream(_CanonicalJsonStreamValue):
    dimensions: Mapping[int, int]

    def iter_canonical_json_chunks(self) -> Iterator[bytes]:
        yield b"["
        for index, (current_id, dimension) in enumerate(
            sorted(self.dimensions.items())
        ):
            if index:
                yield b","
            yield from _iter_canonical_json_chunks(
                {"current_id": current_id, "dimension": dimension}
            )
        yield b"]"


@dataclass(frozen=True, slots=True)
class _CertificateRowsStream(_CanonicalJsonStreamValue):
    certificates: Sequence[RecurrenceNumericalCurrentCertificate]

    def iter_canonical_json_chunks(self) -> Iterator[bytes]:
        yield b"["
        for index, certificate in enumerate(self.certificates):
            if index:
                yield b","
            yield from _iter_canonical_json_chunks(certificate.to_json_dict())
        yield b"]"


@dataclass(frozen=True, slots=True)
class _MappingRowsStream(_CanonicalJsonStreamValue):
    certificates: Sequence[RecurrenceNumericalCurrentCertificate]

    def iter_canonical_json_chunks(self) -> Iterator[bytes]:
        yield b"["
        for index, certificate in enumerate(self.certificates):
            if index:
                yield b","
            yield from _iter_canonical_json_chunks(_mapping_payload(certificate))
        yield b"]"


@dataclass(frozen=True, slots=True)
class RecurrenceCurrentObservationCapture:
    """Complete point-major exact current observations."""

    precision_digits: int
    points: tuple[ValidationPointRecord, ...]
    point_sha256s: tuple[str, ...]
    kinematic_sha256s: tuple[str, ...]
    kinematic_binary64: tuple[tuple[str, ...], ...]
    selector_contexts: tuple[object, ...]
    context_sha256s: tuple[str, ...]
    parameter_contexts: tuple[tuple[Decimal, ...], ...]
    parameter_context_sha256s: tuple[str, ...]
    observations: Mapping[int, tuple[_ComplexDecimal, ...]]
    current_dimensions: Mapping[int, int]
    runtime_parameter_schema_sha256: str
    source_semantics_sha256: str
    observation_batch_sha256: str
    capture_contract_sha256: str
    context_policy: str
    full_current_count: int | None = None
    complete_observations_retained: bool = True

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def current_count(self) -> int:
        return (
            len(self.observations)
            if self.full_current_count is None
            else self.full_current_count
        )

    def detached_for_replay(
        self,
        current_ids: Sequence[int],
    ) -> RecurrenceCurrentObservationCapture:
        """Retain only rows needed by the persisted certificate replay."""

        selected = tuple(sorted(set(current_ids)))
        if any(current_id not in self.observations for current_id in selected):
            raise ValueError(
                "certificate replay references an absent recurrence current"
            )
        return replace(
            self,
            observations={
                current_id: self.observations[current_id] for current_id in selected
            },
            full_current_count=self.current_count,
            complete_observations_retained=False,
        )

    def to_provenance_dict(self) -> dict[str, object]:
        return {
            "abi": _CAPTURE_ABI,
            "precision_digits": self.precision_digits,
            "point_count": self.point_count,
            "point_sha256s": list(self.point_sha256s),
            "kinematic_sha256s": list(self.kinematic_sha256s),
            "parameter_contexts": [
                [_decimal_string(value) for value in context]
                for context in self.parameter_contexts
            ],
            "parameter_context_sha256s": list(self.parameter_context_sha256s),
            "context_sha256s": list(self.context_sha256s),
            "points": [point.to_mapping() for point in self.points],
            "current_count": self.current_count,
            "runtime_parameter_schema_sha256": (self.runtime_parameter_schema_sha256),
            "source_semantics_sha256": self.source_semantics_sha256,
            "observation_batch_sha256": self.observation_batch_sha256,
            "capture_contract_sha256": self.capture_contract_sha256,
            "context_policy": self.context_policy,
            "complete_current_components": True,
            "point_major": True,
            "current_dimensions_sha256": _canonical_sha256(
                {
                    str(current_id): dimension
                    for current_id, dimension in sorted(self.current_dimensions.items())
                }
            ),
            "evaluator": "recurrence-direct-plan-decimal-symbolica-exact",
        }

    def to_evidence_dict(self) -> dict[str, object]:
        if not self.complete_observations_retained:
            raise ValueError(
                "detached recurrence observations cannot recreate raw evidence"
            )
        return {
            **self.to_provenance_dict(),
            "kinematic_binary64": [list(row) for row in self.kinematic_binary64],
            "selector_contexts": list(self.selector_contexts),
            "current_dimensions": [
                {"current_id": current_id, "dimension": dimension}
                for current_id, dimension in sorted(self.current_dimensions.items())
            ],
            "observations": [
                {
                    "current_id": current_id,
                    "dimension": self.current_dimensions[current_id],
                    "values": [
                        [_decimal_string(real), _decimal_string(imaginary)]
                        for real, imaginary in self.observations[current_id]
                    ],
                }
                for current_id in sorted(self.observations)
            ],
        }

    def to_certificate_replay_dict(
        self,
        current_ids: Sequence[int],
    ) -> dict[str, object]:
        """Return deduplicated raw rows needed to replay applied certificates."""

        selected = tuple(sorted(set(current_ids)))
        if any(current_id not in self.observations for current_id in selected):
            raise ValueError(
                "certificate replay references an absent recurrence current"
            )
        observation_rows: list[object] = []
        for current_id in selected:
            row = {
                "current_id": current_id,
                "dimension": self.current_dimensions[current_id],
                "values": [
                    [_decimal_string(real), _decimal_string(imaginary)]
                    for real, imaginary in self.observations[current_id]
                ],
            }
            observation_rows.append(row)
        return {
            "abi": "pyamplicol-recurrence-certificate-capture-replay-v1",
            "precision_digits": self.precision_digits,
            "points": [point.to_mapping() for point in self.points],
            "kinematic_binary64": [list(row) for row in self.kinematic_binary64],
            "selector_contexts": list(self.selector_contexts),
            "parameter_contexts": [
                [_decimal_string(value) for value in context]
                for context in self.parameter_contexts
            ],
            "point_sha256s": list(self.point_sha256s),
            "kinematic_sha256s": list(self.kinematic_sha256s),
            "selector_context_sha256s": list(self.context_sha256s),
            "parameter_context_sha256s": list(self.parameter_context_sha256s),
            "runtime_parameter_schema_sha256": (self.runtime_parameter_schema_sha256),
            "source_semantics_sha256": self.source_semantics_sha256,
            "certificate_current_ids": list(selected),
            "observations": observation_rows,
            "full_batch_commitment": {
                "current_count": self.current_count,
                "current_dimensions_sha256": _canonical_sha256(
                    {
                        str(current_id): dimension
                        for current_id, dimension in sorted(
                            self.current_dimensions.items()
                        )
                    }
                ),
                "observation_batch_sha256": self.observation_batch_sha256,
                "capture_contract_sha256": self.capture_contract_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class RecurrenceNumericalCurrentCertificate:
    """Canonical candidate plus independent-verification evidence."""

    current_id: int
    representative_id: int | None
    execution_representative_id: int
    relation_kind: _RelationKind
    factor: tuple[int, int]
    source_semantics_sha256: str
    precision_digits: int
    seed: int
    relative_tolerance: float
    absolute_tolerance: float
    candidate_probe_count: int
    verification_probe_count: int
    current_dimension: int
    runtime_parameter_schema_sha256: str
    candidate_capture_sha256: str
    verification_capture_sha256: str
    candidate_maximum_absolute_residual: Fraction
    candidate_maximum_relative_residual: Fraction
    candidate_maximum_tolerance_ratio: Fraction
    verification_maximum_absolute_residual: Fraction
    verification_maximum_relative_residual: Fraction
    verification_maximum_tolerance_ratio: Fraction
    candidate_observations_sha256: str
    verification_observations_sha256: str
    probe_contract_sha256: str
    proof_sha256: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "algorithm": _RECURRENCE_CERTIFICATE_ALGORITHM,
            "proof_kind": "authenticated-numerical",
            "relation_kind": self.relation_kind,
            "current_id": self.current_id,
            "representative_id": self.representative_id,
            "execution_representative_id": self.execution_representative_id,
            "factor_integer": list(self.factor),
            "source_semantics_sha256": self.source_semantics_sha256,
            "precision_digits": self.precision_digits,
            "seed": self.seed,
            "relative_tolerance_binary64": self.relative_tolerance.hex(),
            "absolute_tolerance_binary64": self.absolute_tolerance.hex(),
            "candidate_probe_count": self.candidate_probe_count,
            "verification_probe_count": self.verification_probe_count,
            "current_dimension": self.current_dimension,
            "runtime_parameter_schema_sha256": (self.runtime_parameter_schema_sha256),
            "candidate_capture_sha256": self.candidate_capture_sha256,
            "verification_capture_sha256": self.verification_capture_sha256,
            "candidate_maximum_absolute_residual": _fraction_string(
                self.candidate_maximum_absolute_residual
            ),
            "candidate_maximum_relative_residual": _fraction_string(
                self.candidate_maximum_relative_residual
            ),
            "candidate_maximum_tolerance_ratio": _fraction_string(
                self.candidate_maximum_tolerance_ratio
            ),
            "verification_maximum_absolute_residual": _fraction_string(
                self.verification_maximum_absolute_residual
            ),
            "verification_maximum_relative_residual": _fraction_string(
                self.verification_maximum_relative_residual
            ),
            "verification_maximum_tolerance_ratio": _fraction_string(
                self.verification_maximum_tolerance_ratio
            ),
            "candidate_observations_sha256": (self.candidate_observations_sha256),
            "verification_observations_sha256": (self.verification_observations_sha256),
            "probe_contract_sha256": self.probe_contract_sha256,
            "proof_sha256": self.proof_sha256,
        }


@dataclass(frozen=True, slots=True)
class RecurrenceNumericalCurrentWarmupResult:
    """One unpublished-baseline discovery transaction."""

    requested_mode: _Mode
    effective_mode: _Mode
    color_accuracy: str
    source_semantics_sha256: str
    runtime_parameter_schema: Mapping[str, object]
    candidate_capture: RecurrenceCurrentObservationCapture
    verification_capture: RecurrenceCurrentObservationCapture | None
    certificates: tuple[RecurrenceNumericalCurrentCertificate, ...]
    evidence_json: bytes
    evidence_transport_bytes: int
    evidence_encoding: _EvidenceEncoding | None
    evidence_canonical_bytes: int
    discovery_report: Mapping[str, object]
    application_validation: Mapping[str, object]
    generation_profile_timings: Mapping[str, float] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    @property
    def applied_relation_count(self) -> int:
        return len(self.applied_certificates)

    @property
    def applied_certificates(self) -> tuple[RecurrenceNumericalCurrentCertificate, ...]:
        """Keep discovery complete while applying only supported relations."""

        if self.effective_mode != "certified-reuse":
            return ()
        return _applicable_numerical_certificates(
            self.certificates,
            application_scope=cast(
                Mapping[str, object], self.discovery_report["application_scope"]
            ),
        )

    @property
    def warning_required(self) -> bool:
        return self.applied_relation_count > 0

    @property
    def effective_mode_reason(self) -> str | None:
        """Explain a conservative requested-to-effective mode downgrade."""

        if self.effective_mode == self.requested_mode:
            return None
        application_scope = cast(
            Mapping[str, object],
            self.discovery_report["application_scope"],
        )
        if cast(Sequence[int], application_scope["suppressed_selector_domain_ids"]):
            return "multi-helicity-all-flow-member-scope-unproven"
        if any(
            certificate.relation_kind == "opposite" for certificate in self.certificates
        ):
            return "contracted-opposite-binary64-disabled"
        raise AssertionError("recurrence numerical mode changed without a reason")

    def with_application_validation(
        self,
        payload: Mapping[str, object],
    ) -> RecurrenceNumericalCurrentWarmupResult:
        return replace(self, application_validation=dict(payload))

    def without_evidence_transport(
        self,
    ) -> RecurrenceNumericalCurrentWarmupResult:
        """Release the native-only transport after second-pass lowering."""

        return replace(self, evidence_json=b"")

    def close(self) -> None:
        """Close any trusted temporary observation stores owned by the result."""

        for capture in (self.candidate_capture, self.verification_capture):
            if capture is not None and isinstance(
                capture.observations,
                _SpooledObservationMapping,
            ):
                capture.observations.close()

    def to_json_dict(self) -> dict[str, object]:
        if not self.certificates:
            state = "no_certified_numerical_relation"
        elif self.effective_mode == "certified-reuse":
            state = "authenticated-numerical-applied"
        else:
            state = "authenticated-numerical-diagnostic-only"
        warning = _warning_payload(self.warning_required)
        relation_correctness = numerical_relation_correctness_payload(
            self.applied_relation_count
        )
        applied_current_ids = (
            tuple(
                current_id
                for certificate in self.applied_certificates
                for current_id in (
                    certificate.current_id,
                    certificate.representative_id,
                )
                if current_id is not None
            )
            if self.effective_mode == "certified-reuse"
            else ()
        )
        persisted_evidence = {
            "abi": _PERSISTED_EVIDENCE_ABI,
            "generation_raw_evidence_bytes": self.evidence_canonical_bytes,
            "generation_evidence_transport_bytes": self.evidence_transport_bytes,
            "generation_evidence_encoding": self.evidence_encoding,
            "raw_evidence_retained": False,
            "runtime_parameter_schema": dict(self.runtime_parameter_schema),
            "runtime_parameter_schema_sha256": (
                self.candidate_capture.runtime_parameter_schema_sha256
            ),
            "candidate_capture": self.candidate_capture.to_certificate_replay_dict(
                applied_current_ids
            ),
            "verification_capture": (
                None
                if self.verification_capture is None
                else self.verification_capture.to_certificate_replay_dict(
                    applied_current_ids,
                )
            ),
            "full_census": {
                "inspected_current_count": self.discovery_report[
                    "inspected_current_count"
                ],
                "tested_hypothesis_count": self.discovery_report[
                    "tested_hypothesis_count"
                ],
                "candidate_index": dict(
                    cast(
                        Mapping[str, object],
                        self.discovery_report["candidate_index"],
                    )
                ),
                "numerical_candidate_count": self.discovery_report[
                    "numerical_candidate_count"
                ],
                "verification_rejected_count": self.discovery_report[
                    "verification_rejected_count"
                ],
                "uncertified_candidate_count": self.discovery_report[
                    "verification_rejected_count"
                ],
                "certified_relation_count": len(self.certificates),
                "rejected_hypothesis_count": self.discovery_report[
                    "rejected_hypothesis_count"
                ],
                "decision_sha256": self.discovery_report["decision_sha256"],
                "rejection_decision_sha256": self.discovery_report[
                    "rejection_decision_sha256"
                ],
                "certificate_set_sha256": _certificate_set_sha256(self.certificates),
            },
        }
        persisted_encoded = _bounded_canonical_json_bytes(
            persisted_evidence,
            byte_limit=None,
            label="recurrence persisted numerical evidence",
        )
        persisted_evidence["measured_payload_bytes"] = len(persisted_encoded)
        persisted_evidence["sha256"] = _canonical_sha256(persisted_evidence)
        application_scope = dict(
            cast(
                Mapping[str, object],
                self.discovery_report["application_scope"],
            )
        )
        application_scope["effective_mode_reason"] = self.effective_mode_reason
        return {
            "schema_version": 1,
            "abi": _WARMUP_ABI,
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "effective_mode_reason": self.effective_mode_reason,
            "effective_reuse_state": (
                "enabled"
                if self.effective_mode == "certified-reuse"
                or self.requested_mode == "diagnostic"
                else "disabled"
            ),
            "state": state,
            "scope": {
                "execution_mode": "recurrence",
                "color_accuracy": self.color_accuracy,
                "representation": "recurrence-direct-plan-v2",
            },
            "source_semantics": {
                "abi": _SOURCE_ABI,
                "sha256": self.source_semantics_sha256,
            },
            "candidate_capture": self.candidate_capture.to_provenance_dict(),
            "verification_capture": (
                None
                if self.verification_capture is None
                else self.verification_capture.to_provenance_dict()
            ),
            "application_capture": (
                self.application_validation.get("application_capture")
            ),
            "application_validation": dict(self.application_validation),
            "application_scope": application_scope,
            "persisted_numerical_evidence": persisted_evidence,
            "discovery": dict(self.discovery_report),
            "application": {
                "schema_version": 1,
                "abi": _RELATION_SET_ABI,
                "requested_mode": self.requested_mode,
                "effective_mode": self.effective_mode,
                "state": state,
                "certificate_replay": {
                    "algorithm": _RECURRENCE_CERTIFICATE_ALGORITHM,
                    "status": (
                        "verified"
                        if self.certificates
                        else "no_certified_numerical_relation"
                    ),
                    "certificate_set_sha256": _certificate_set_sha256(
                        self.certificates
                    ),
                },
                "certified_relation_count": len(self.certificates),
                "applied_relation_count": self.applied_relation_count,
                "relation_kind_counts": {
                    kind: sum(
                        certificate.relation_kind == kind
                        for certificate in self.certificates
                    )
                    for kind in ("equal", "opposite", "zero")
                },
                "certificates": [
                    certificate.to_json_dict() for certificate in self.certificates
                ],
                "mappings": [
                    _mapping_payload(certificate) for certificate in self.certificates
                ],
                "warning": warning,
            },
            "certified_relation_count": len(self.certificates),
            "applied_relation_count": self.applied_relation_count,
            "relation_correctness": relation_correctness,
            "warning": warning,
        }


def run_recurrence_numerical_current_warmup(
    plan: _RecurrenceExactPlan,
    *,
    candidate_points: Sequence[ValidationPointRecord],
    verification_points: Sequence[ValidationPointRecord],
    mode: _Mode,
    color_accuracy: str,
    precision_digits: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> RecurrenceNumericalCurrentWarmupResult:
    """Capture, discover, certify, and encode a second-pass relation set."""

    if mode not in {"diagnostic", "certified-reuse"}:
        raise ValueError(
            "recurrence numerical current warm-up requires diagnostic or "
            "certified-reuse mode"
        )
    _validated_tolerances(relative_tolerance, absolute_tolerance)
    relative_tolerance = float(relative_tolerance)
    absolute_tolerance = float(absolute_tolerance)
    generation_profile_timings: dict[str, float] = {}
    geometry_started = time.perf_counter()
    geometry = _select_raw_evidence_storage_geometry(
        plan.sections,
        candidate_probe_count=len(candidate_points),
        verification_probe_count=len(verification_points),
        runtime_parameter_count=len(plan.runtime_parameter_schema),
    )
    generation_profile_timings["warmup_geometry"] = (
        time.perf_counter() - geometry_started
    )
    static_context_started = time.perf_counter()
    static_context = _build_warmup_static_context(plan)
    generation_profile_timings["warmup_static_context"] = (
        time.perf_counter() - static_context_started
    )
    source_digest = static_context.source_semantics_sha256
    runtime_parameter_schema = static_context.runtime_parameter_schema
    runtime_parameter_schema_sha256 = static_context.runtime_parameter_schema_sha256
    runtime_defaults = static_context.runtime_defaults
    probe_policies = tuple(row.probe_policy for row in plan.runtime_parameter_schema)
    candidate_parameter_contexts = _build_parameter_probe_contexts(
        runtime_defaults,
        probe_policies=probe_policies,
        precision_digits=precision_digits,
        seed=seed,
        domain="candidate-current-parameter-probes-v1",
        count=len(candidate_points),
        include_defaults=True,
    )
    candidate_spool: _SpooledObservationMapping | None = None
    verification_spool: _SpooledObservationMapping | None = None
    evidence_spool: BinaryIO | None = None
    try:
        candidate_probe_timings: list[float] = []
        candidate_capture_started = time.perf_counter()
        candidate = _capture_recurrence_current_observations(
            plan,
            candidate_points,
            precision_digits=precision_digits,
            source_semantics_sha256=source_digest,
            seed=seed,
            domain="candidate-current-probes-v1",
            parameter_contexts=candidate_parameter_contexts,
            runtime_parameter_schema_sha256=runtime_parameter_schema_sha256,
            static_context=static_context,
            probe_timings=candidate_probe_timings,
        )
        generation_profile_timings["warmup_candidate_capture"] = (
            time.perf_counter() - candidate_capture_started
        )
        generation_profile_timings.update(
            {
                f"warmup_candidate_probe_{index}": seconds
                for index, seconds in enumerate(candidate_probe_timings)
            }
        )
        candidate_index_started = time.perf_counter()
        candidate_indexes: (
            Mapping[
                _CurrentContract,
                NumericalObservationCandidateIndex[Fraction],
            ]
            | None
        ) = _build_candidate_indexes(
            plan.sections,
            candidate.observations,
            contracts=static_context.current_contracts,
        )
        generation_profile_timings["warmup_candidate_index"] = (
            time.perf_counter() - candidate_index_started
        )
        candidate_preflight_started = time.perf_counter()
        try:
            certificates, discovery = _discover_relations(
                plan.sections,
                candidate,
                None,
                source_semantics_sha256=source_digest,
                precision_digits=precision_digits,
                seed=seed,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                color_accuracy=color_accuracy,
                contracts=static_context.current_contracts,
                candidate_indexes=candidate_indexes,
            )
        except _IndependentVerificationRequired:
            generation_profile_timings["warmup_candidate_preflight"] = (
                time.perf_counter() - candidate_preflight_started
            )
        else:
            generation_profile_timings["warmup_relation_discovery"] = (
                time.perf_counter() - candidate_preflight_started
            )
            if certificates:
                raise AssertionError(
                    "candidate-only recurrence discovery produced certificates"
                )
            effective_mode = _effective_numerical_relation_mode(
                plan.sections,
                requested_mode=mode,
                certificates=certificates,
            )
            candidate = candidate.detached_for_replay(())
            return RecurrenceNumericalCurrentWarmupResult(
                requested_mode=mode,
                effective_mode=effective_mode,
                color_accuracy=color_accuracy,
                source_semantics_sha256=source_digest,
                runtime_parameter_schema=runtime_parameter_schema,
                candidate_capture=candidate,
                verification_capture=None,
                certificates=certificates,
                evidence_json=b"",
                evidence_transport_bytes=0,
                evidence_encoding=None,
                evidence_canonical_bytes=0,
                discovery_report=discovery,
                application_validation={
                    "status": "not-required-no-applied-relations",
                    "checked_current_count": 0,
                    "checked_component_count": 0,
                    "maximum_absolute_residual": None,
                    "maximum_relative_residual": None,
                    "maximum_tolerance_ratio": None,
                    "application_capture": None,
                },
                generation_profile_timings=MappingProxyType(
                    dict(generation_profile_timings)
                ),
            )
        if geometry.encoding == _COMPRESSED_EVIDENCE_ENCODING:
            candidate_spool_started = time.perf_counter()
            candidate_spool = _SpooledObservationMapping(
                candidate.observations,
                candidate_indexes=candidate_indexes,
            )
            candidate = replace(candidate, observations=candidate_spool)
            # The verification capture can itself be large.  Keep the exact
            # reusable index on disk until relation discovery instead of
            # retaining both resident captures and the index simultaneously.
            candidate_indexes = None
            generation_profile_timings["warmup_candidate_spool"] = (
                time.perf_counter() - candidate_spool_started
            )
        verification_parameter_contexts = _build_parameter_probe_contexts(
            runtime_defaults,
            probe_policies=probe_policies,
            precision_digits=precision_digits,
            seed=seed,
            domain="independent-verification-current-parameter-probes-v1",
            count=len(verification_points),
            include_defaults=False,
        )
        verification_probe_timings: list[float] = []
        verification_capture_started = time.perf_counter()
        verification = _capture_recurrence_current_observations(
            plan,
            verification_points,
            precision_digits=precision_digits,
            source_semantics_sha256=source_digest,
            seed=seed,
            domain="independent-verification-current-probes-v1",
            parameter_contexts=verification_parameter_contexts,
            runtime_parameter_schema_sha256=runtime_parameter_schema_sha256,
            static_context=static_context,
            probe_timings=verification_probe_timings,
        )
        generation_profile_timings["warmup_verification_capture"] = (
            time.perf_counter() - verification_capture_started
        )
        generation_profile_timings.update(
            {
                f"warmup_verification_probe_{index}": seconds
                for index, seconds in enumerate(verification_probe_timings)
            }
        )
        if geometry.encoding == _COMPRESSED_EVIDENCE_ENCODING:
            verification_spool_started = time.perf_counter()
            verification_spool = _SpooledObservationMapping(verification.observations)
            verification = replace(
                verification,
                observations=verification_spool,
            )
            generation_profile_timings["warmup_verification_spool"] = (
                time.perf_counter() - verification_spool_started
            )
        _validate_independent_captures(candidate, verification)
        discovery_started = time.perf_counter()
        certificates, discovery = _discover_relations(
            plan.sections,
            candidate,
            verification,
            source_semantics_sha256=source_digest,
            precision_digits=precision_digits,
            seed=seed,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            color_accuracy=color_accuracy,
            contracts=static_context.current_contracts,
            candidate_indexes=candidate_indexes,
        )
        generation_profile_timings["warmup_relation_discovery"] = (
            time.perf_counter() - discovery_started
        )
        evidence_payload_started = time.perf_counter()
        effective_mode = _effective_numerical_relation_mode(
            plan.sections,
            requested_mode=mode,
            certificates=certificates,
        )
        evidence = _evidence_payload(
            plan.sections,
            mode=effective_mode,
            source_semantics=static_context.source_semantics,
            source_semantics_sha256=source_digest,
            certificates=certificates,
            discovery=discovery,
            precision_digits=precision_digits,
            seed=seed,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            probe_count=candidate.point_count,
            verification_probe_count=verification.point_count,
            runtime_parameter_schema=runtime_parameter_schema,
            candidate=candidate,
            verification=verification,
        )
        generation_profile_timings["warmup_evidence_payload"] = (
            time.perf_counter() - evidence_payload_started
        )
        evidence_serialization_started = time.perf_counter()
        if geometry.encoding == _RAW_EVIDENCE_ENCODING:
            encoded = _bounded_canonical_json_bytes(
                evidence,
                byte_limit=geometry.canonical_byte_limit,
                label="recurrence numerical raw evidence",
            )
            canonical_byte_count = len(encoded)
        else:
            evidence_spool = tempfile.TemporaryFile(  # noqa: SIM115
                prefix="pyamplicol-recurrence-evidence-",
            )
            canonical_byte_count = _write_compressed_canonical_evidence(
                evidence,
                evidence_spool,
                canonical_byte_limit=geometry.canonical_byte_limit,
            )
            encoded = b""
        generation_profile_timings["warmup_evidence_serialization"] = (
            time.perf_counter() - evidence_serialization_started
        )
        replay_current_ids = (
            tuple(
                current_id
                for certificate in _applicable_numerical_certificates(
                    certificates,
                    application_scope=cast(
                        Mapping[str, object], discovery["application_scope"]
                    ),
                )
                for current_id in (
                    certificate.current_id,
                    certificate.representative_id,
                )
                if current_id is not None
            )
            if effective_mode == "certified-reuse"
            else ()
        )
        if not isinstance(candidate.observations, _SpooledObservationMapping):
            candidate = candidate.detached_for_replay(replay_current_ids)
            verification = verification.detached_for_replay(replay_current_ids)
        else:
            # Transfer both trusted row stores to the result. Certificate
            # replay and application validation consume rows lazily, so the
            # resident graph never grows with the number of relations.
            candidate_spool = None
            verification_spool = None
        # The streamed evidence mapping owns wrappers pointing at the complete
        # captures. Drop it before reading the compressed transport bytes.
        del evidence
        if evidence_spool is not None:
            evidence_transport_started = time.perf_counter()
            encoded = _read_compressed_evidence_spool(evidence_spool)
            evidence_spool.close()
            evidence_spool = None
            generation_profile_timings["warmup_evidence_transport_read"] = (
                time.perf_counter() - evidence_transport_started
            )
        return RecurrenceNumericalCurrentWarmupResult(
            requested_mode=mode,
            effective_mode=effective_mode,
            color_accuracy=color_accuracy,
            source_semantics_sha256=source_digest,
            runtime_parameter_schema=runtime_parameter_schema,
            candidate_capture=candidate,
            verification_capture=verification,
            certificates=certificates,
            evidence_json=encoded,
            evidence_transport_bytes=len(encoded),
            evidence_encoding=geometry.encoding,
            evidence_canonical_bytes=canonical_byte_count,
            discovery_report=discovery,
            application_validation={
                "status": (
                    "pending-second-pass-validation"
                    if effective_mode == "certified-reuse" and certificates
                    else "not-required-no-applied-relations"
                ),
                "checked_current_count": 0,
                "checked_component_count": 0,
                "maximum_absolute_residual": None,
                "maximum_relative_residual": None,
                "maximum_tolerance_ratio": None,
                "application_capture": None,
            },
            generation_profile_timings=MappingProxyType(
                dict(generation_profile_timings)
            ),
        )
    finally:
        if candidate_spool is not None:
            candidate_spool.close()
        if verification_spool is not None:
            verification_spool.close()
        if evidence_spool is not None:
            evidence_spool.close()


def build_recurrence_numerical_current_probe_points(
    process: CanonicalProcessIR,
    model: Model,
    *,
    process_id: str,
    seed: int,
    candidate_count: int,
    verification_count: int,
) -> tuple[
    tuple[ValidationPointRecord, ...],
    tuple[ValidationPointRecord, ...],
]:
    """Build disjoint deterministic physical points for recurrence probes."""

    if (
        type(seed) is not int
        or seed < 0
        or type(candidate_count) is not int
        or candidate_count < 2
        or type(verification_count) is not int
        or verification_count < 2
    ):
        raise ValueError("recurrence numerical probe-point contract is invalid")

    def points(domain: str, count: int) -> tuple[ValidationPointRecord, ...]:
        return tuple(
            rotate_validation_point(
                build_process_validation_point(
                    process,
                    model,
                    process_id=process_id,
                    seed=_domain_seed(seed, domain=domain, index=index),
                ),
                rotation_seed=_domain_seed(
                    seed,
                    domain=f"{domain}:spatial-rotation-v1",
                    index=index,
                ),
            )
            for index in range(count)
        )

    candidate = points("candidate-current-probes-v1", candidate_count)
    verification = points(
        "independent-verification-current-probes-v1",
        verification_count,
    )
    if any(not point.available for point in (*candidate, *verification)):
        errors = tuple(
            point.error for point in (*candidate, *verification) if not point.available
        )
        raise ValueError(
            "recurrence numerical probe-point generation failed: "
            + "; ".join(str(error) for error in errors)
        )
    candidate_hashes = {_kinematic_sha256(point) for point in candidate}
    verification_hashes = {_kinematic_sha256(point) for point in verification}
    if (
        len(candidate_hashes) != len(candidate)
        or len(verification_hashes) != len(verification)
        or not candidate_hashes.isdisjoint(verification_hashes)
    ):
        raise ValueError("recurrence numerical probe domains are not independent")
    return candidate, verification


def validate_recurrence_numerical_current_application(
    baseline: RecurrenceNumericalCurrentWarmupResult,
    applied_plan: _RecurrenceExactPlan,
    *,
    baseline_plan: _RecurrenceExactPlan,
    precision_digits: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> RecurrenceNumericalCurrentWarmupResult:
    """Fail closed unless every second-pass current replays on verification."""

    if not baseline.applied_relation_count:
        return baseline.with_application_validation(
            {
                "status": "not-required-no-applied-relations",
                "checked_current_count": 0,
                "checked_component_count": 0,
                "maximum_absolute_residual": None,
                "maximum_relative_residual": None,
                "maximum_tolerance_ratio": None,
                "application_capture": None,
            }
        )
    verification_capture = baseline.verification_capture
    if verification_capture is None:
        raise ValueError(
            "applied recurrence numerical relations lack verification observations"
        )
    reference_spool: _SpooledObservationMapping | None = None
    try:
        reference = capture_recurrence_current_observations(
            baseline_plan,
            verification_capture.points,
            precision_digits=precision_digits,
            source_semantics_sha256=baseline.source_semantics_sha256,
            seed=seed,
            domain="independent-verification-current-probes-v1",
            parameter_contexts=verification_capture.parameter_contexts,
            runtime_parameter_schema_sha256=(
                verification_capture.runtime_parameter_schema_sha256
            ),
        )
        _validate_recaptured_reference(
            verification_capture,
            reference,
        )
        reference_spool = _SpooledObservationMapping(reference.observations)
        reference = replace(reference, observations=reference_spool)
        applied = capture_recurrence_current_observations(
            applied_plan,
            verification_capture.points,
            precision_digits=precision_digits,
            source_semantics_sha256=baseline.source_semantics_sha256,
            seed=seed,
            domain="independent-verification-current-probes-v1",
            parameter_contexts=verification_capture.parameter_contexts,
            runtime_parameter_schema_sha256=(
                verification_capture.runtime_parameter_schema_sha256
            ),
        )
        validation = _validate_applied_observations(
            reference,
            applied,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        validation["application_capture"] = applied.to_provenance_dict()
        return baseline.with_application_validation(validation)
    finally:
        if reference_spool is not None:
            reference_spool.close()


def recurrence_numerical_current_opt_out_report(
    sections: _RecurrenceExactSectionsV1,
    *,
    color_accuracy: str,
) -> dict[str, object]:
    """Return stable provenance without evaluating a single exact current."""

    return {
        "schema_version": 1,
        "abi": _WARMUP_ABI,
        "requested_mode": "off",
        "effective_mode": "off",
        "effective_reuse_state": "disabled",
        "state": "disabled-by-user",
        "scope": {
            "execution_mode": "recurrence",
            "color_accuracy": color_accuracy,
            "representation": "recurrence-direct-plan-v2",
        },
        "source_semantics": {
            "abi": _SOURCE_ABI,
            "sha256": recurrence_numerical_source_semantics_sha256(sections),
        },
        "candidate_capture": None,
        "verification_capture": None,
        "application_capture": None,
        "application_validation": {
            "status": "disabled-by-user",
            "checked_current_count": 0,
            "checked_component_count": 0,
            "maximum_absolute_residual": None,
            "maximum_relative_residual": None,
            "maximum_tolerance_ratio": None,
            "application_capture": None,
        },
        "application_scope": {
            "abi": _APPLICATION_SCOPE_ABI,
            "policy": "disabled-by-user",
            "layout": sections.strategy,
            "suppressed_selector_domain_ids": [],
            "contracted_opposite_application": "not-applicable",
            "suppressed_current_count": 0,
        },
        "discovery": None,
        "application": None,
        "certified_relation_count": 0,
        "applied_relation_count": 0,
        "relation_correctness": numerical_relation_correctness_payload(0),
        "warning": _warning_payload(False),
    }


def recurrence_numerical_current_envelope_fallback_report(
    sections: _RecurrenceExactSectionsV1,
    *,
    color_accuracy: str,
    requested_mode: _Mode,
    outcome: RecurrenceNumericalEvidenceEnvelopeExceeded,
) -> dict[str, object]:
    """Publish the explicit reuse-off result of one safe geometry fallback."""

    if requested_mode != "certified-reuse":
        raise ValueError("the evidence-envelope fallback is limited to certified reuse")
    fallback = outcome.to_json_dict()
    warning = {
        "required": True,
        "emit": "once-per-generated-artifact",
        "code": EVIDENCE_ENVELOPE_FALLBACK_REASON,
        "message": EVIDENCE_ENVELOPE_FALLBACK_WARNING,
    }
    return {
        "schema_version": 1,
        "abi": _WARMUP_ABI,
        "requested_mode": requested_mode,
        "effective_mode": "off",
        "effective_reuse_state": "disabled",
        "state": "disabled-evidence-envelope-fallback",
        "fallback": fallback,
        "scope": {
            "execution_mode": "recurrence",
            "color_accuracy": color_accuracy,
            "representation": "recurrence-direct-plan-v2",
        },
        "source_semantics": {
            "abi": _SOURCE_ABI,
            "sha256": recurrence_numerical_source_semantics_sha256(sections),
        },
        "candidate_capture": None,
        "verification_capture": None,
        "application_capture": None,
        "application_validation": {
            "status": "not-required-reuse-disabled",
            "checked_current_count": 0,
            "checked_component_count": 0,
            "maximum_absolute_residual": None,
            "maximum_relative_residual": None,
            "maximum_tolerance_ratio": None,
            "application_capture": None,
        },
        "application_scope": {
            "abi": _APPLICATION_SCOPE_ABI,
            "policy": "evidence-envelope-fallback",
            "layout": sections.strategy,
            "suppressed_selector_domain_ids": [],
            "contracted_opposite_application": "not-applicable",
            "suppressed_current_count": 0,
        },
        "discovery": None,
        "application": None,
        "certified_relation_count": 0,
        "applied_relation_count": 0,
        "relation_correctness": numerical_relation_correctness_payload(0),
        "warning": warning,
    }


def capture_recurrence_current_observations(
    plan: _RecurrenceExactPlan,
    points: Sequence[ValidationPointRecord],
    *,
    precision_digits: int,
    source_semantics_sha256: str,
    seed: int,
    domain: str,
    parameter_contexts: Sequence[Sequence[Decimal]] | None = None,
    runtime_parameter_schema_sha256: str | None = None,
) -> RecurrenceCurrentObservationCapture:
    """Evaluate every current before Direct-Arena stage recycling."""

    return _capture_recurrence_current_observations(
        plan,
        points,
        precision_digits=precision_digits,
        source_semantics_sha256=source_semantics_sha256,
        seed=seed,
        domain=domain,
        parameter_contexts=parameter_contexts,
        runtime_parameter_schema_sha256=runtime_parameter_schema_sha256,
        static_context=None,
        probe_timings=None,
    )


def _capture_recurrence_current_observations(
    plan: _RecurrenceExactPlan,
    points: Sequence[ValidationPointRecord],
    *,
    precision_digits: int,
    source_semantics_sha256: str,
    seed: int,
    domain: str,
    parameter_contexts: Sequence[Sequence[Decimal]] | None,
    runtime_parameter_schema_sha256: str | None,
    static_context: _WarmupStaticContext | None,
    probe_timings: list[float] | None,
) -> RecurrenceCurrentObservationCapture:
    records = tuple(points)
    if (
        type(precision_digits) is not int
        or precision_digits < 80
        or len(records) < 2
        or any(not isinstance(point, ValidationPointRecord) for point in records)
        or any(not point.available for point in records)
        or len({point.process_id for point in records}) != 1
    ):
        raise ValueError("recurrence current capture point contract is invalid")
    sections = plan.sections
    dimensions = (
        {current.semantic_id: current.component_count for current in sections.currents}
        if static_context is None
        else static_context.current_dimensions
    )
    expected_current_ids = frozenset(dimensions)
    observations: dict[int, list[_ComplexDecimal]] = {
        current_id: [] for current_id in dimensions
    }
    point_sha256s = tuple(_canonical_sha256(point.to_mapping()) for point in records)
    kinematic_binary64 = tuple(_kinematic_binary64(point) for point in records)
    kinematic_sha256s = tuple(_canonical_sha256(row) for row in kinematic_binary64)
    if len(set(point_sha256s)) != len(records) or len(set(kinematic_sha256s)) != len(
        records
    ):
        raise ValueError("recurrence current capture requires distinct physical points")
    if static_context is None:
        runtime_defaults = _runtime_parameter_defaults(plan)
        schema_payload = _runtime_parameter_schema_payload(
            plan,
            defaults=runtime_defaults,
        )
        actual_schema_sha256 = _canonical_sha256(schema_payload)
    else:
        runtime_defaults = static_context.runtime_defaults
        actual_schema_sha256 = static_context.runtime_parameter_schema_sha256
    if (
        runtime_parameter_schema_sha256 is not None
        and runtime_parameter_schema_sha256 != actual_schema_sha256
    ):
        raise ValueError("recurrence runtime parameter schema digest drifted")
    resolved_parameter_contexts = (
        tuple(runtime_defaults for _record in records)
        if parameter_contexts is None
        else tuple(tuple(values) for values in parameter_contexts)
    )
    if (
        len(resolved_parameter_contexts) != len(records)
        or any(
            len(values) != len(runtime_defaults)
            or any(
                not isinstance(value, Decimal) or not value.is_finite()
                for value in values
            )
            for values in resolved_parameter_contexts
        )
        or any(
            row.probe_policy == "derived-overwritten-fixed-zero-v1"
            and any(
                values[row.runtime_slot] != runtime_defaults[row.runtime_slot]
                for values in resolved_parameter_contexts
            )
            for row in plan.runtime_parameter_schema
        )
    ):
        raise ValueError(
            "recurrence parameter contexts do not cover every flattened runtime slot"
        )
    parameter_context_sha256s = tuple(
        _canonical_sha256(
            {
                "abi": _PARAMETER_CONTEXT_ABI,
                "domain": domain,
                "point_index": point_index,
                "point_sha256": point_sha256,
                "runtime_parameter_schema_sha256": actual_schema_sha256,
                "values": [_decimal_string(value) for value in values],
            }
        )
        for point_index, (point_sha256, values) in enumerate(
            zip(point_sha256s, resolved_parameter_contexts, strict=True)
        )
    )
    context_payloads: list[object] = []
    working_precision = precision_digits + 16
    with localcontext() as context:
        context.prec = working_precision
        context.rounding = ROUND_HALF_EVEN
        for point_index, (record, parameter_context) in enumerate(
            zip(records, resolved_parameter_contexts, strict=True)
        ):
            probe_started = time.perf_counter()
            parameters = plan.resolve_model_parameters(
                parameter_context,
                working_precision,
            )
            point = cast(
                _Point,
                tuple(
                    tuple(Decimal.from_float(float(value)) for value in momentum)
                    for momentum in record.four_vectors
                ),
            )
            captured, context_payload = _capture_one_point(
                plan,
                point,
                parameters.prepared,
                working_precision,
                selector_seed=_domain_seed(
                    seed,
                    domain=domain,
                    index=point_index,
                ),
                expected_current_ids=expected_current_ids,
            )
            if probe_timings is not None:
                probe_timings.append(time.perf_counter() - probe_started)
            context_payloads.append(context_payload)
            if captured.keys() != expected_current_ids:
                raise ValueError("recurrence exact current capture is incomplete")
            for current_id, dimension in dimensions.items():
                values = captured[current_id]
                if len(values) != dimension:
                    raise ValueError(
                        f"recurrence current {current_id} capture width "
                        f"{len(values)} does not match {dimension}"
                    )
                observations[current_id].extend(values)
    frozen = {current_id: tuple(values) for current_id, values in observations.items()}
    if any(
        not real.is_finite() or not imaginary.is_finite()
        for values in frozen.values()
        for real, imaginary in values
    ):
        raise ValueError(
            "recurrence exact current capture produced a non-finite component"
        )
    context_sha256s = tuple(_canonical_sha256(payload) for payload in context_payloads)
    batch_digest = _canonical_sha256(
        {
            "abi": _OBSERVATION_BATCH_ABI,
            "source_semantics_sha256": source_semantics_sha256,
            "runtime_parameter_schema_sha256": actual_schema_sha256,
            "point_sha256s": list(point_sha256s),
            "selector_context_sha256s": list(context_sha256s),
            "parameter_context_sha256s": list(parameter_context_sha256s),
            "currents": _ObservationRowsStream(frozen, dimensions),
        }
    )
    context_policy = _context_policy(sections.strategy)
    capture_contract_sha256 = _canonical_sha256(
        {
            "abi": _CAPTURE_ABI,
            "precision_digits": precision_digits,
            "source_semantics_sha256": source_semantics_sha256,
            "runtime_parameter_schema_sha256": actual_schema_sha256,
            "point_sha256s": list(point_sha256s),
            "kinematic_sha256s": list(kinematic_sha256s),
            "selector_context_sha256s": list(context_sha256s),
            "parameter_context_sha256s": list(parameter_context_sha256s),
            "current_dimensions": _DimensionRowsStream(dimensions),
            "observation_batch_sha256": batch_digest,
            "context_policy": context_policy,
        }
    )
    capture_dimensions = dimensions if static_context is None else dict(dimensions)
    return RecurrenceCurrentObservationCapture(
        precision_digits=precision_digits,
        points=records,
        point_sha256s=point_sha256s,
        kinematic_sha256s=kinematic_sha256s,
        kinematic_binary64=kinematic_binary64,
        selector_contexts=tuple(context_payloads),
        context_sha256s=context_sha256s,
        parameter_contexts=resolved_parameter_contexts,
        parameter_context_sha256s=parameter_context_sha256s,
        observations=frozen,
        current_dimensions=capture_dimensions,
        runtime_parameter_schema_sha256=actual_schema_sha256,
        source_semantics_sha256=source_semantics_sha256,
        observation_batch_sha256=batch_digest,
        capture_contract_sha256=capture_contract_sha256,
        context_policy=context_policy,
    )


def recurrence_numerical_source_semantics_sha256(
    sections: _RecurrenceExactSectionsV1,
) -> str:
    """Bind probes to the baseline semantic schedule and current contracts."""

    return _canonical_sha256(_source_semantics_payload(sections))


def _source_semantics_payload(
    sections: _RecurrenceExactSectionsV1,
    *,
    contracts: tuple[_CurrentContract, ...] | None = None,
) -> dict[str, object]:
    if contracts is None:
        contracts = _current_contracts(sections)
    if sections.strategy == "topology-replay":
        selector_schedule: object = {
            "policy": _context_policy(sections.strategy),
            "replay_targets": [
                {
                    "target_index": target_index,
                    "public_flow_id": target.public_flow_id,
                    "representative_id": target.representative_id,
                    "selector_domain_id": target.selector_domain_id,
                }
                for target_index, target in enumerate(sections.replay_targets)
            ],
        }
    elif sections.strategy == "all-flow-union":
        selector_schedule = {
            "policy": _context_policy(sections.strategy),
            "resolved_helicities": [
                {
                    "helicity_index": helicity_index,
                    "helicity_id": helicity.helicity_id,
                    "selector_domain_id": helicity.selector_domain_id,
                }
                for helicity_index, helicity in enumerate(sections.resolved_helicities)
            ],
        }
    else:
        selector_schedule = {
            "policy": _context_policy(sections.strategy),
            "fixed_source_schedule": True,
        }
    return {
        "abi": _SOURCE_ABI,
        "process_id": sections.process_id,
        "strategy": sections.strategy,
        "schedule_semantic_digest": sections.semantic_digest,
        "baseline_runtime_layout_digest": sections.runtime_layout_digest,
        "selector_schedule": selector_schedule,
        "currents": [
            {
                "current_id": current.semantic_id,
                "is_source": current.source_row != DIRECT_NONE_U32,
                "contract": _plain_contract(contracts[current.semantic_id]),
            }
            for current in sections.currents
        ],
    }


def _capture_one_point(
    plan: _RecurrenceExactPlan,
    point: _Point,
    prepared_parameters: Sequence[_ComplexDecimal],
    precision: int,
    *,
    selector_seed: int,
    expected_current_ids: frozenset[int],
) -> tuple[dict[int, tuple[_ComplexDecimal, ...]], object]:
    sections = plan.sections
    if sections.strategy == "topology-replay":
        if not sections.replay_targets:
            raise ArtifactError("topology replay has no exact probe target")
        target_index = selector_seed % len(sections.replay_targets)
        target = sections.replay_targets[target_index]
        captured = _capture_execution(
            sections,
            lambda observer: _evaluate_replay_point(
                plan,
                point,
                target,
                prepared_parameters,
                precision,
                current_observer=observer,
            ),
            expected_current_ids=expected_current_ids,
        )
        return captured, {
            "strategy": sections.strategy,
            "target_index": target_index,
            "public_flow_id": target.public_flow_id,
            "representative_id": target.representative_id,
            "selector_domain_id": target.selector_domain_id,
        }
    if sections.strategy == "all-flow-union":
        if not sections.resolved_helicities:
            raise ArtifactError("all-flow union has no exact probe helicity")
        by_domain: dict[int, list[int]] = defaultdict(list)
        for index, helicity in enumerate(sections.resolved_helicities):
            by_domain[helicity.selector_domain_id].append(index)
        captures: dict[int, dict[int, tuple[_ComplexDecimal, ...]]] = {}
        selected: list[dict[str, int]] = []
        for domain_id, indices in sorted(by_domain.items()):
            position = selector_seed % len(indices)
            helicity_index = indices[position]
            helicity = sections.resolved_helicities[helicity_index]
            captures[domain_id] = _capture_execution(
                sections,
                lambda observer, helicity=helicity: _evaluate_union_point(
                    plan,
                    point,
                    helicity,
                    prepared_parameters,
                    precision,
                    current_observer=observer,
                ),
                expected_current_ids=expected_current_ids,
            )
            selected.append(
                {
                    "selector_domain_id": domain_id,
                    "helicity_index": helicity_index,
                    "helicity_id": helicity.helicity_id,
                }
            )
        default_capture = captures[min(captures)]
        merged = {
            current.semantic_id: captures.get(
                current.selector_domain_id,
                default_capture,
            )[current.semantic_id]
            for current in sections.currents
        }
        return merged, {
            "strategy": sections.strategy,
            "selector_domain_helicities": selected,
        }
    captured = _capture_execution(
        sections,
        lambda observer: _evaluate_contracted_point(
            plan,
            point,
            prepared_parameters,
            precision,
            current_observer=observer,
        ),
        expected_current_ids=expected_current_ids,
    )
    return captured, {
        "strategy": sections.strategy,
        "fixed_source_schedule": True,
    }


def _capture_execution(
    sections: _RecurrenceExactSectionsV1,
    operation: object,
    *,
    expected_current_ids: frozenset[int] | None = None,
) -> dict[int, tuple[_ComplexDecimal, ...]]:
    captured: dict[int, tuple[_ComplexDecimal, ...]] = {}

    def observe(
        current_id: int,
        values: tuple[_ComplexDecimal, ...],
    ) -> None:
        if current_id in captured:
            raise ArtifactError(
                f"recurrence current {current_id} was observed more than once"
            )
        captured[current_id] = values

    if not callable(operation):  # pragma: no cover - internal type contract
        raise TypeError("recurrence capture operation is not callable")
    operation(observe)
    expected = (
        frozenset(current.semantic_id for current in sections.currents)
        if expected_current_ids is None
        else expected_current_ids
    )
    if captured.keys() != expected:
        missing = sorted(expected - captured.keys())
        raise ArtifactError(
            f"recurrence current observation missed semantic currents {missing[:8]}"
        )
    return captured


def _build_raw_numerical_candidate_index(
    member_ids: Sequence[int],
    observations: Mapping[int, Sequence[_ComplexDecimal]],
) -> NumericalObservationCandidateIndex[Fraction]:
    return build_numerical_observation_candidate_index(
        member_ids,
        observations,
        normalize=Fraction,
    )


def _raw_numerical_tolerance_window_ids(
    index: NumericalObservationCandidateIndex[Fraction],
    current_values: Sequence[_ComplexDecimal],
    *,
    relation_kind: Literal["equal", "opposite"],
    current_id: int,
    relative_tolerance: Fraction,
    absolute_tolerance: Fraction,
) -> tuple[int, ...]:
    return numerical_observation_tolerance_window_ids(
        index,
        current_values,
        relation_kind=relation_kind,
        current_id=current_id,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        normalize=Fraction,
    )


def _build_candidate_indexes(
    sections: _RecurrenceExactSectionsV1,
    observations: Mapping[int, Sequence[_ComplexDecimal]],
    *,
    contracts: tuple[_CurrentContract, ...] | None = None,
) -> dict[
    _CurrentContract,
    NumericalObservationCandidateIndex[Fraction],
]:
    if contracts is None:
        contracts = _current_contracts(sections)
    members_by_contract: dict[_CurrentContract, list[int]] = defaultdict(list)
    for current in sections.currents:
        members_by_contract[contracts[current.semantic_id]].append(current.semantic_id)
    return {
        contract: _build_raw_numerical_candidate_index(
            member_ids,
            observations,
        )
        for contract, member_ids in members_by_contract.items()
    }


def _numerical_relation_application_scope(
    sections: _RecurrenceExactSectionsV1,
) -> dict[str, object]:
    """Return the narrow execution members eligible for numerical reuse."""

    helicities_by_domain: dict[int, int] = defaultdict(int)
    for helicity in sections.resolved_helicities:
        helicities_by_domain[helicity.selector_domain_id] += 1
    unsafe_all_flow_domains = (
        tuple(
            domain_id
            for domain_id, member_count in sorted(helicities_by_domain.items())
            if member_count > 1
        )
        if sections.strategy == "all-flow-union"
        else ()
    )
    return {
        "abi": _APPLICATION_SCOPE_ABI,
        "policy": "exact-physical-selector-member-set-v1",
        "layout": sections.strategy,
        "multi_helicity_all_flow_domain_ids": list(unsafe_all_flow_domains),
        "multi_helicity_all_flow_application": (
            "disabled" if unsafe_all_flow_domains else "not-applicable"
        ),
        "suppressed_selector_domain_ids": list(unsafe_all_flow_domains),
        "contracted_opposite_application": (
            "disabled"
            if sections.strategy == "contracted-color-union"
            else "not-applicable"
        ),
    }


def _effective_numerical_relation_mode(
    sections: _RecurrenceExactSectionsV1,
    *,
    requested_mode: _Mode,
    certificates: Sequence[RecurrenceNumericalCurrentCertificate],
) -> _Mode:
    """Contain relations that the native mapper cannot apply member-wise."""

    if requested_mode != "certified-reuse":
        return requested_mode
    scope = _numerical_relation_application_scope(sections)
    if cast(Sequence[int], scope["suppressed_selector_domain_ids"]):
        return "diagnostic"
    if certificates and not _applicable_numerical_certificates(
        certificates, application_scope=scope
    ):
        return "diagnostic"
    return requested_mode


def _applicable_numerical_certificates(
    certificates: Sequence[RecurrenceNumericalCurrentCertificate],
    *,
    application_scope: Mapping[str, object],
) -> tuple[RecurrenceNumericalCurrentCertificate, ...]:
    """Mirror native application containment without changing its evidence.

    A skipped opposite relation keeps its original current evaluation. Equal
    relations may therefore still refer to that current; no representative is
    rewritten and no relation depends on applying a skipped certificate.
    """

    if application_scope["suppressed_selector_domain_ids"]:
        return ()
    suppress_opposite = (
        application_scope["contracted_opposite_application"] == "disabled"
    )
    return tuple(
        certificate
        for certificate in certificates
        if not (suppress_opposite and certificate.relation_kind == "opposite")
    )


def _discover_relations(
    sections: _RecurrenceExactSectionsV1,
    candidate: RecurrenceCurrentObservationCapture,
    verification: RecurrenceCurrentObservationCapture | None,
    *,
    source_semantics_sha256: str,
    precision_digits: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    color_accuracy: str,
    contracts: tuple[_CurrentContract, ...] | None = None,
    candidate_indexes: Mapping[
        _CurrentContract,
        NumericalObservationCandidateIndex[Fraction],
    ]
    | None = None,
) -> tuple[
    tuple[RecurrenceNumericalCurrentCertificate, ...],
    dict[str, object],
]:
    relative, absolute = _validated_tolerances(
        relative_tolerance,
        absolute_tolerance,
    )
    relative_fraction = Fraction(relative)
    absolute_fraction = Fraction(absolute)
    application_scope = _numerical_relation_application_scope(sections)
    unsafe_selector_domains = frozenset(
        cast(
            Sequence[int],
            application_scope["suppressed_selector_domain_ids"],
        )
    )
    if contracts is None:
        contracts = _current_contracts(sections)
    if candidate_indexes is None:
        spooled_indexes = (
            candidate.observations.candidate_indexes
            if isinstance(candidate.observations, _SpooledObservationMapping)
            else None
        )
        candidate_indexes = spooled_indexes
    if candidate_indexes is None:
        candidate_indexes = _build_candidate_indexes(
            sections,
            candidate.observations,
            contracts=contracts,
        )
    prior_by_contract: dict[_CurrentContract, list[int]] = defaultdict(list)
    certificates: list[RecurrenceNumericalCurrentCertificate] = []
    rejected: list[dict[str, object]] = []
    nearest: tuple[Fraction, Fraction, Fraction, int, int, str] | None = None
    tested = 0
    theoretical_pair_hypotheses = 0
    screened_pair_hypotheses = 0
    zero_hypotheses = 0
    candidates = 0
    verification_rejected = 0
    rejected_hypotheses = 0
    scope_suppressed_current_count = 0
    decision_chain = _canonical_sha256(
        {
            "abi": _DECISION_CHAIN_ABI,
            "source_semantics_sha256": source_semantics_sha256,
            "candidate_capture_sha256": candidate.capture_contract_sha256,
            "verification_capture_sha256": (
                None if verification is None else verification.capture_contract_sha256
            ),
        }
    )
    rejection_chain = _canonical_sha256(
        {
            "abi": _REJECTION_CHAIN_ABI,
            "source_semantics_sha256": source_semantics_sha256,
            "candidate_capture_sha256": candidate.capture_contract_sha256,
            "verification_capture_sha256": (
                None if verification is None else verification.capture_contract_sha256
            ),
        }
    )

    def record_decision(
        *,
        current_id: int,
        representative_id: int | None,
        relation_kind: _RelationKind,
        candidate_residuals: _Residuals,
        candidate_residual_strings: _ResidualStrings,
        verification_residuals: _Residuals | None,
        verification_residual_strings: _ResidualStrings | None,
        selected: bool,
    ) -> None:
        nonlocal decision_chain
        decision_chain = hashlib.sha256(
            bytes.fromhex(decision_chain)
            + _canonical_json_bytes(
                {
                    "current_id": current_id,
                    "representative_id": representative_id,
                    "relation_kind": relation_kind,
                    "candidate_maximum_absolute_residual": (
                        candidate_residual_strings[0]
                    ),
                    "candidate_maximum_relative_residual": (
                        candidate_residual_strings[1]
                    ),
                    "candidate_maximum_tolerance_ratio": (
                        candidate_residual_strings[2]
                    ),
                    "candidate_accepted": candidate_residuals[2] <= 1,
                    "verification_maximum_absolute_residual": (
                        None
                        if verification_residual_strings is None
                        else verification_residual_strings[0]
                    ),
                    "verification_maximum_relative_residual": (
                        None
                        if verification_residual_strings is None
                        else verification_residual_strings[1]
                    ),
                    "verification_maximum_tolerance_ratio": (
                        None
                        if verification_residual_strings is None
                        else verification_residual_strings[2]
                    ),
                    "verification_accepted": (
                        None
                        if verification_residuals is None
                        else verification_residuals[2] <= 1
                    ),
                    "selected": selected,
                }
            )
        ).hexdigest()

    def record_rejected(
        *,
        current_id: int,
        representative_id: int | None,
        relation_kind: _RelationKind,
        residuals: _Residuals,
        residual_strings: _ResidualStrings,
        reason: str,
    ) -> None:
        nonlocal nearest, rejected_hypotheses, rejection_chain
        rejected_hypotheses += 1
        rejected_row = {
            "current_id": current_id,
            "representative_id": representative_id,
            "relation_kind": relation_kind,
            "reason": reason,
            "maximum_absolute_residual": residual_strings[0],
            "maximum_relative_residual": residual_strings[1],
            "maximum_tolerance_ratio": residual_strings[2],
        }
        rejection_chain = hashlib.sha256(
            bytes.fromhex(rejection_chain) + _canonical_json_bytes(rejected_row)
        ).hexdigest()
        item = (
            residuals[2],
            residuals[0],
            residuals[1],
            current_id,
            -1 if representative_id is None else representative_id,
            relation_kind,
        )
        if nearest is None or item < nearest:
            nearest = item
        if len(rejected) < _MAX_REJECTED_DIAGNOSTICS:
            rejected.append(rejected_row)

    for current in sections.currents:
        contract = contracts[current.semantic_id]
        prior = prior_by_contract[contract]
        if current.source_row != DIRECT_NONE_U32:
            prior.append(current.semantic_id)
            continue
        if current.selector_domain_id in unsafe_selector_domains:
            scope_suppressed_current_count += 1
            continue
        candidate_current_observations = candidate.observations[current.semantic_id]
        equal_representatives = set(
            _raw_numerical_tolerance_window_ids(
                candidate_indexes[contract],
                candidate_current_observations,
                relation_kind="equal",
                current_id=current.semantic_id,
                relative_tolerance=relative_fraction,
                absolute_tolerance=absolute_fraction,
            )
        )
        opposite_representatives = set(
            _raw_numerical_tolerance_window_ids(
                candidate_indexes[contract],
                candidate_current_observations,
                relation_kind="opposite",
                current_id=current.semantic_id,
                relative_tolerance=relative_fraction,
                absolute_tolerance=absolute_fraction,
            )
        )
        theoretical_pair_hypotheses += 2 * len(prior)
        screened_pair_hypotheses += len(equal_representatives) + len(
            opposite_representatives
        )
        zero_hypotheses += 1
        hypotheses: list[tuple[_RelationKind, int | None]] = [
            ("zero", None),
            *[
                (relation_kind, representative_id)
                for representative_id in sorted(
                    equal_representatives | opposite_representatives
                )
                for relation_kind in (
                    *(("equal",) if representative_id in equal_representatives else ()),
                    *(
                        ("opposite",)
                        if representative_id in opposite_representatives
                        else ()
                    ),
                )
            ],
        ]
        accepted = False
        verification_current_observations: Sequence[_ComplexDecimal] | None = None
        for relation_kind, representative_id in hypotheses:
            tested += 1
            candidate_residuals = _relation_residuals(
                relation_kind,
                candidate_current_observations,
                (
                    None
                    if representative_id is None
                    else candidate.observations[representative_id]
                ),
                relative_tolerance=relative_fraction,
                absolute_tolerance=absolute_fraction,
            )
            candidate_residual_strings = _residual_strings(candidate_residuals)
            if candidate_residuals[2] > 1:
                record_decision(
                    current_id=current.semantic_id,
                    representative_id=representative_id,
                    relation_kind=relation_kind,
                    candidate_residuals=candidate_residuals,
                    candidate_residual_strings=candidate_residual_strings,
                    verification_residuals=None,
                    verification_residual_strings=None,
                    selected=False,
                )
                record_rejected(
                    current_id=current.semantic_id,
                    representative_id=representative_id,
                    relation_kind=relation_kind,
                    residuals=candidate_residuals,
                    residual_strings=candidate_residual_strings,
                    reason="candidate-observations-not-equal",
                )
                continue
            if verification is None:
                raise _IndependentVerificationRequired
            candidates += 1
            if verification_current_observations is None:
                verification_current_observations = verification.observations[
                    current.semantic_id
                ]
            verification_residuals = _relation_residuals(
                relation_kind,
                verification_current_observations,
                (
                    None
                    if representative_id is None
                    else verification.observations[representative_id]
                ),
                relative_tolerance=relative_fraction,
                absolute_tolerance=absolute_fraction,
            )
            verification_residual_strings = _residual_strings(verification_residuals)
            if verification_residuals[2] > 1:
                verification_rejected += 1
                record_decision(
                    current_id=current.semantic_id,
                    representative_id=representative_id,
                    relation_kind=relation_kind,
                    candidate_residuals=candidate_residuals,
                    candidate_residual_strings=candidate_residual_strings,
                    verification_residuals=verification_residuals,
                    verification_residual_strings=(verification_residual_strings),
                    selected=False,
                )
                record_rejected(
                    current_id=current.semantic_id,
                    representative_id=representative_id,
                    relation_kind=relation_kind,
                    residuals=verification_residuals,
                    residual_strings=verification_residual_strings,
                    reason="independent-verification-rejected-candidate",
                )
                continue
            if relation_kind == "zero":
                execution_representative_id = prior[0] if prior else current.semantic_id
            else:
                if representative_id is None:
                    raise AssertionError(
                        "non-zero recurrence relation lacks a representative"
                    )
                execution_representative_id = representative_id
            certificates.append(
                _build_certificate(
                    current=current,
                    representative_id=representative_id,
                    execution_representative_id=(execution_representative_id),
                    relation_kind=relation_kind,
                    source_semantics_sha256=source_semantics_sha256,
                    candidate=candidate,
                    verification=verification,
                    candidate_residuals=candidate_residuals,
                    candidate_residual_strings=candidate_residual_strings,
                    verification_residuals=verification_residuals,
                    verification_residual_strings=(verification_residual_strings),
                    precision_digits=precision_digits,
                    seed=seed,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                )
            )
            record_decision(
                current_id=current.semantic_id,
                representative_id=representative_id,
                relation_kind=relation_kind,
                candidate_residuals=candidate_residuals,
                candidate_residual_strings=candidate_residual_strings,
                verification_residuals=verification_residuals,
                verification_residual_strings=verification_residual_strings,
                selected=True,
            )
            accepted = True
            break
        prior.append(current.semantic_id)
        if accepted:
            continue

    nearest_payload = None
    if nearest is not None:
        ratio, absolute_residual, relative_residual, current_id, rep, kind = nearest
        nearest_payload = {
            "current_id": current_id,
            "representative_id": None if rep < 0 else rep,
            "relation_kind": kind,
            "maximum_absolute_residual": _fraction_string(absolute_residual),
            "maximum_relative_residual": _fraction_string(relative_residual),
            "maximum_tolerance_ratio": _fraction_string(ratio),
        }
    certificate_tuple = tuple(certificates)
    decision_sha256 = _canonical_sha256(
        {
            "abi": _DECISION_CHAIN_ABI,
            "chain_tail_sha256": decision_chain,
            "tested_hypothesis_count": tested,
            "theoretical_pair_hypothesis_count": (theoretical_pair_hypotheses),
            "screened_pair_hypothesis_count": screened_pair_hypotheses,
            "zero_hypothesis_count": zero_hypotheses,
            "numerical_candidate_count": candidates,
            "verification_rejected_count": verification_rejected,
            "rejected_hypothesis_count": rejected_hypotheses,
            "certified_relation_count": len(certificate_tuple),
        }
    )
    rejection_sha256 = rejection_chain
    rejected_diagnostics = {
        "total_rejected_hypothesis_count": rejected_hypotheses,
        "retained_count": len(rejected),
        "truncated": rejected_hypotheses > len(rejected),
        "truncation_policy": (
            f"first-{_MAX_REJECTED_DIAGNOSTICS}-in-canonical-decision-order-v1"
        ),
        "retained_sha256": _canonical_sha256(
            {
                "abi": ("pyamplicol-recurrence-rejected-numerical-diagnostics-v1"),
                "rows": rejected,
            }
        ),
        "full_decision_sha256": decision_sha256,
        "full_rejection_sha256": rejection_sha256,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "state": (
            "certified_numerical_relations"
            if certificate_tuple
            else "no_certified_numerical_relation"
        ),
        "scope": {
            "execution_mode": "recurrence",
            "color_accuracy": color_accuracy,
            "representation": "recurrence-direct-plan-v2",
            "lc_flow_layout": sections.strategy,
        },
        "application_scope": {
            **application_scope,
            "suppressed_current_count": scope_suppressed_current_count,
        },
        "source_semantics": {
            "abi": _SOURCE_ABI,
            "sha256": source_semantics_sha256,
        },
        "probe_contract": {
            "algorithm": _RECURRENCE_CERTIFICATE_ALGORITHM,
            "precision_digits": precision_digits,
            "seed": seed,
            "relative_tolerance_binary64": relative_tolerance.hex(),
            "absolute_tolerance_binary64": absolute_tolerance.hex(),
            "candidate_point_sha256s": list(candidate.point_sha256s),
            "verification_point_sha256s": (
                [] if verification is None else list(verification.point_sha256s)
            ),
            "candidate_context_sha256s": list(candidate.context_sha256s),
            "verification_context_sha256s": (
                [] if verification is None else list(verification.context_sha256s)
            ),
            "candidate_parameter_context_sha256s": list(
                candidate.parameter_context_sha256s
            ),
            "verification_parameter_context_sha256s": list(
                () if verification is None else verification.parameter_context_sha256s
            ),
            "runtime_parameter_schema_sha256": (
                candidate.runtime_parameter_schema_sha256
            ),
            "candidate_capture_sha256": candidate.capture_contract_sha256,
            "verification_capture_sha256": (
                None if verification is None else verification.capture_contract_sha256
            ),
            "candidate_observation_batch_sha256": (candidate.observation_batch_sha256),
            "verification_observation_batch_sha256": (
                None if verification is None else verification.observation_batch_sha256
            ),
            "deterministic": True,
            "independent_verification": verification is not None,
            **(
                {"verification_status": ("not-required-no-candidate-hypothesis")}
                if verification is None
                else {}
            ),
            "current_dimension_bound": True,
        },
        "inspected_current_count": len(sections.currents),
        "structurally_proven_current_count": 0,
        "tested_hypothesis_count": tested,
        "candidate_index": {
            "algorithm": _CANDIDATE_INDEX_ALGORITHM,
            "completeness": "complete-within-configured-tolerance",
            "contract_count": len(candidate_indexes),
            "exhaustive_fallback_contract_count": (
                len(candidate_indexes) if relative_fraction >= 1 else 0
            ),
            "theoretical_pair_hypothesis_count": (theoretical_pair_hypotheses),
            "screened_pair_hypothesis_count": screened_pair_hypotheses,
            "zero_hypothesis_count": zero_hypotheses,
            "nearest_rejected_scope": ("zero-and-tolerance-window-screened-hypotheses"),
        },
        "numerical_candidate_count": candidates,
        "verification_rejected_count": verification_rejected,
        "rejected_hypothesis_count": rejected_hypotheses,
        "decision_sha256": decision_sha256,
        "rejection_decision_sha256": rejection_sha256,
        "certified_numerical_relation_count": len(certificate_tuple),
        "certificates": [
            certificate.to_json_dict() for certificate in certificate_tuple
        ],
        "rejected_candidates": rejected,
        "rejected_candidate_diagnostics": rejected_diagnostics,
        "nearest_rejected_hypothesis": nearest_payload,
        "warning": _warning_payload(False),
    }
    return certificate_tuple, report


def _build_certificate(
    *,
    current: _Current,
    representative_id: int | None,
    execution_representative_id: int,
    relation_kind: _RelationKind,
    source_semantics_sha256: str,
    candidate: RecurrenceCurrentObservationCapture,
    verification: RecurrenceCurrentObservationCapture,
    candidate_residuals: _Residuals,
    candidate_residual_strings: _ResidualStrings,
    verification_residuals: _Residuals,
    verification_residual_strings: _ResidualStrings,
    precision_digits: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> RecurrenceNumericalCurrentCertificate:
    factor = {
        "equal": (1, 0),
        "opposite": (-1, 0),
        "zero": (0, 0),
    }[relation_kind]
    candidate_digest = _relation_observation_sha256(
        current.semantic_id,
        representative_id,
        relation_kind,
        candidate,
    )
    verification_digest = _relation_observation_sha256(
        current.semantic_id,
        representative_id,
        relation_kind,
        verification,
    )
    probe_contract = {
        "algorithm": _RECURRENCE_CERTIFICATE_ALGORITHM,
        "source_semantics_sha256": source_semantics_sha256,
        "current_id": current.semantic_id,
        "representative_id": representative_id,
        "execution_representative_id": execution_representative_id,
        "relation_kind": relation_kind,
        "precision_digits": precision_digits,
        "seed": seed,
        "candidate_domain": "candidate-current-probes-v1",
        "verification_domain": ("independent-verification-current-probes-v1"),
        "relative_tolerance_binary64": relative_tolerance.hex(),
        "absolute_tolerance_binary64": absolute_tolerance.hex(),
        "candidate_probe_count": candidate.point_count,
        "verification_probe_count": verification.point_count,
        "current_dimension": current.component_count,
        "runtime_parameter_schema_sha256": (candidate.runtime_parameter_schema_sha256),
        "candidate_capture_sha256": candidate.capture_contract_sha256,
        "verification_capture_sha256": verification.capture_contract_sha256,
        "candidate_observations_sha256": candidate_digest,
        "verification_observations_sha256": verification_digest,
    }
    probe_digest = _canonical_sha256(probe_contract)
    proof_payload = {
        **probe_contract,
        "proof_kind": "authenticated-numerical",
        "factor_integer": list(factor),
        "candidate_maximum_absolute_residual": candidate_residual_strings[0],
        "candidate_maximum_relative_residual": candidate_residual_strings[1],
        "candidate_maximum_tolerance_ratio": candidate_residual_strings[2],
        "verification_maximum_absolute_residual": verification_residual_strings[0],
        "verification_maximum_relative_residual": verification_residual_strings[1],
        "verification_maximum_tolerance_ratio": verification_residual_strings[2],
        "probe_contract_sha256": probe_digest,
    }
    return RecurrenceNumericalCurrentCertificate(
        current_id=current.semantic_id,
        representative_id=representative_id,
        execution_representative_id=execution_representative_id,
        relation_kind=relation_kind,
        factor=factor,
        source_semantics_sha256=source_semantics_sha256,
        precision_digits=precision_digits,
        seed=seed,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        candidate_probe_count=candidate.point_count,
        verification_probe_count=verification.point_count,
        current_dimension=current.component_count,
        runtime_parameter_schema_sha256=(candidate.runtime_parameter_schema_sha256),
        candidate_capture_sha256=candidate.capture_contract_sha256,
        verification_capture_sha256=verification.capture_contract_sha256,
        candidate_maximum_absolute_residual=candidate_residuals[0],
        candidate_maximum_relative_residual=candidate_residuals[1],
        candidate_maximum_tolerance_ratio=candidate_residuals[2],
        verification_maximum_absolute_residual=verification_residuals[0],
        verification_maximum_relative_residual=verification_residuals[1],
        verification_maximum_tolerance_ratio=verification_residuals[2],
        candidate_observations_sha256=candidate_digest,
        verification_observations_sha256=verification_digest,
        probe_contract_sha256=probe_digest,
        proof_sha256=_canonical_sha256(proof_payload),
    )


def _evidence_payload(
    sections: _RecurrenceExactSectionsV1,
    *,
    mode: _Mode,
    source_semantics: Mapping[str, object],
    source_semantics_sha256: str,
    certificates: tuple[RecurrenceNumericalCurrentCertificate, ...],
    discovery: Mapping[str, object],
    precision_digits: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    probe_count: int,
    verification_probe_count: int,
    runtime_parameter_schema: Mapping[str, object],
    candidate: RecurrenceCurrentObservationCapture,
    verification: RecurrenceCurrentObservationCapture,
) -> dict[str, object]:
    numerical_candidate_count = _required_report_integer(
        discovery,
        "numerical_candidate_count",
    )
    verification_rejected_count = _required_report_integer(
        discovery,
        "verification_rejected_count",
    )
    tested_hypothesis_count = _required_report_integer(
        discovery,
        "tested_hypothesis_count",
    )
    rejected_hypothesis_count = _required_report_integer(
        discovery,
        "rejected_hypothesis_count",
    )
    return {
        "abi": _EVIDENCE_ABI,
        "requested_mode": mode,
        "schedule_semantic_digest": sections.semantic_digest,
        "baseline_runtime_layout_digest": sections.runtime_layout_digest,
        "source_semantics": source_semantics,
        "source_semantics_sha256": source_semantics_sha256,
        "runtime_parameter_schema": dict(runtime_parameter_schema),
        "runtime_parameter_schema_sha256": (candidate.runtime_parameter_schema_sha256),
        "candidate_capture": _capture_evidence_payload(candidate),
        "verification_capture": _capture_evidence_payload(verification),
        "certificate_algorithm": _RECURRENCE_CERTIFICATE_ALGORITHM,
        "certificate_set_sha256": _certificate_set_sha256(certificates),
        "precision_digits": precision_digits,
        "probe_count": probe_count,
        "verification_probe_count": verification_probe_count,
        "relative_tolerance_binary64": relative_tolerance.hex(),
        "absolute_tolerance_binary64": absolute_tolerance.hex(),
        "seed": seed,
        "candidate_index": dict(
            cast(Mapping[str, object], discovery["candidate_index"])
        ),
        "numerical_candidate_count": numerical_candidate_count,
        "verification_rejected_count": verification_rejected_count,
        "rejected_hypothesis_count": rejected_hypothesis_count,
        "tested_hypothesis_count": tested_hypothesis_count,
        "decision_sha256": str(discovery["decision_sha256"]),
        "rejection_decision_sha256": str(discovery["rejection_decision_sha256"]),
        "certificates": _CertificateRowsStream(certificates),
        "mappings": _MappingRowsStream(certificates),
    }


def _capture_evidence_payload(
    capture: RecurrenceCurrentObservationCapture,
) -> dict[str, object]:
    """Return a small mapping whose large rows are streamed during encoding."""

    return {
        "abi": _CAPTURE_ABI,
        "precision_digits": capture.precision_digits,
        "point_count": capture.point_count,
        "point_sha256s": capture.point_sha256s,
        "kinematic_sha256s": capture.kinematic_sha256s,
        "parameter_contexts": tuple(
            tuple(_decimal_string(value) for value in context)
            for context in capture.parameter_contexts
        ),
        "parameter_context_sha256s": capture.parameter_context_sha256s,
        "context_sha256s": capture.context_sha256s,
        "points": tuple(point.to_mapping() for point in capture.points),
        "current_count": capture.current_count,
        "runtime_parameter_schema_sha256": (capture.runtime_parameter_schema_sha256),
        "source_semantics_sha256": capture.source_semantics_sha256,
        "observation_batch_sha256": capture.observation_batch_sha256,
        "capture_contract_sha256": capture.capture_contract_sha256,
        "context_policy": capture.context_policy,
        "complete_current_components": True,
        "point_major": True,
        "current_dimensions_sha256": _canonical_sha256(
            {
                str(current_id): dimension
                for current_id, dimension in sorted(capture.current_dimensions.items())
            }
        ),
        "evaluator": "recurrence-direct-plan-decimal-symbolica-exact",
        "kinematic_binary64": capture.kinematic_binary64,
        "selector_contexts": capture.selector_contexts,
        "current_dimensions": _DimensionRowsStream(capture.current_dimensions),
        "observations": _ObservationRowsStream(
            capture.observations,
            capture.current_dimensions,
        ),
    }


def _required_report_integer(
    report: Mapping[str, object],
    key: str,
) -> int:
    value = report.get(key)
    if type(value) is not int:
        raise ValueError(f"recurrence numerical report {key!r} is not an integer")
    return value


def _mapping_payload(
    certificate: RecurrenceNumericalCurrentCertificate,
) -> dict[str, object]:
    return {
        "current_id": certificate.current_id,
        "representative_id": certificate.representative_id,
        "execution_representative_id": (certificate.execution_representative_id),
        "relation_kind": certificate.relation_kind,
        "factor_integer": list(certificate.factor),
        "current_dimension": certificate.current_dimension,
        "certificate_proof_sha256": certificate.proof_sha256,
        "candidate_observations_sha256": (certificate.candidate_observations_sha256),
        "verification_observations_sha256": (
            certificate.verification_observations_sha256
        ),
    }


def _current_contracts(
    sections: _RecurrenceExactSectionsV1,
) -> tuple[_CurrentContract, ...]:
    finalization_executors: dict[int, int] = {}
    for group in sections.row_groups:
        if group.role == 2:
            for row_id in range(group.row_start, group.row_start + group.row_count):
                if row_id in finalization_executors:
                    raise ArtifactError(
                        "recurrence finalization row belongs to multiple groups"
                    )
                finalization_executors[row_id] = group.executor_id
    factors = tuple(
        (
            str(factor.real_numerator),
            str(factor.real_denominator),
            str(factor.imaginary_numerator),
            str(factor.imaginary_denominator),
        )
        for factor in sections.exact_factors
    )
    result = []
    for current in sections.currents:
        if current.finalization_row == DIRECT_NONE_U32:
            finalization = (
                DIRECT_NONE_U32,
                DIRECT_NONE_U32,
                ("1", "1", "0", "1"),
            )
        else:
            try:
                row = sections.finalizations[current.finalization_row]
                executor_id = finalization_executors[current.finalization_row]
                factor = factors[row.exact_factor_id]
            except (IndexError, KeyError) as exc:
                raise ArtifactError(
                    f"recurrence current {current.semantic_id} has an invalid "
                    "finalization contract"
                ) from exc
            finalization = (executor_id, row.momentum_form_id, factor)
        result.append(
            (
                current.node_kind,
                current.stage,
                current.state_template_id,
                current.component_count,
                current.momentum_form_id,
                current.selector_domain_id,
                *finalization,
            )
        )
    return tuple(result)


def _validate_independent_captures(
    candidate: RecurrenceCurrentObservationCapture,
    verification: RecurrenceCurrentObservationCapture,
) -> None:
    if (
        candidate.precision_digits != verification.precision_digits
        or candidate.source_semantics_sha256 != verification.source_semantics_sha256
        or candidate.runtime_parameter_schema_sha256
        != verification.runtime_parameter_schema_sha256
        or candidate.current_dimensions != verification.current_dimensions
        or set(candidate.observations) != set(verification.observations)
        or candidate.point_count < 2
        or verification.point_count < 2
        or not set(candidate.point_sha256s).isdisjoint(verification.point_sha256s)
        or not set(candidate.kinematic_sha256s).isdisjoint(
            verification.kinematic_sha256s
        )
        or len(candidate.parameter_context_sha256s) != candidate.point_count
        or len(verification.parameter_context_sha256s) != verification.point_count
        or not set(candidate.parameter_context_sha256s).isdisjoint(
            verification.parameter_context_sha256s
        )
    ):
        raise ValueError(
            "recurrence candidate and verification captures are not independent"
        )
    for current_id, dimension in candidate.current_dimensions.items():
        if (
            len(candidate.observations[current_id]) != candidate.point_count * dimension
            or len(verification.observations[current_id])
            != verification.point_count * dimension
        ):
            raise ValueError(f"recurrence current {current_id} probe geometry drifted")


def _validate_applied_observations(
    reference: RecurrenceCurrentObservationCapture,
    applied: RecurrenceCurrentObservationCapture,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, object]:
    if (
        reference.precision_digits != applied.precision_digits
        or reference.point_sha256s != applied.point_sha256s
        or reference.kinematic_sha256s != applied.kinematic_sha256s
        or reference.kinematic_binary64 != applied.kinematic_binary64
        or reference.selector_contexts != applied.selector_contexts
        or reference.context_sha256s != applied.context_sha256s
        or reference.parameter_contexts != applied.parameter_contexts
        or reference.parameter_context_sha256s != applied.parameter_context_sha256s
        or reference.runtime_parameter_schema_sha256
        != applied.runtime_parameter_schema_sha256
        or reference.current_dimensions != applied.current_dimensions
        or set(reference.observations) != set(applied.observations)
    ):
        raise ValueError("recurrence application validation provenance drifted")
    relative, absolute = _validated_tolerances(
        relative_tolerance,
        absolute_tolerance,
    )
    relative_fraction = Fraction(relative)
    absolute_fraction = Fraction(absolute)
    maximum_absolute = Fraction(0)
    maximum_relative = Fraction(0)
    maximum_ratio = Fraction(0)
    checked = 0
    for current_id in sorted(reference.observations):
        before = reference.observations[current_id]
        after = applied.observations[current_id]
        if len(before) != len(after):
            raise ValueError(
                f"recurrence current {current_id} application width drifted"
            )
        residuals = _pair_residuals(
            before,
            after,
            sign=1,
            relative_tolerance=relative_fraction,
            absolute_tolerance=absolute_fraction,
        )
        maximum_absolute = max(maximum_absolute, residuals[0])
        maximum_relative = max(maximum_relative, residuals[1])
        maximum_ratio = max(maximum_ratio, residuals[2])
        checked += len(before)
        if residuals[2] > 1:
            raise ValueError(
                "recurrence numerical reuse changed current "
                f"{current_id} beyond its authenticated tolerance"
            )
    return {
        "status": "verified",
        "checked_current_count": len(reference.observations),
        "checked_component_count": checked,
        "maximum_absolute_residual": _fraction_string(maximum_absolute),
        "maximum_relative_residual": _fraction_string(maximum_relative),
        "maximum_tolerance_ratio": _fraction_string(maximum_ratio),
        "reference_observation_batch_sha256": (reference.observation_batch_sha256),
        "applied_observation_batch_sha256": (applied.observation_batch_sha256),
    }


def _validate_recaptured_reference(
    authenticated: RecurrenceCurrentObservationCapture,
    recaptured: RecurrenceCurrentObservationCapture,
) -> None:
    """Bind a post-native baseline recapture to the original full commitment."""

    if (
        authenticated.precision_digits != recaptured.precision_digits
        or authenticated.points != recaptured.points
        or authenticated.point_sha256s != recaptured.point_sha256s
        or authenticated.kinematic_sha256s != recaptured.kinematic_sha256s
        or authenticated.kinematic_binary64 != recaptured.kinematic_binary64
        or authenticated.selector_contexts != recaptured.selector_contexts
        or authenticated.context_sha256s != recaptured.context_sha256s
        or authenticated.parameter_contexts != recaptured.parameter_contexts
        or authenticated.parameter_context_sha256s
        != recaptured.parameter_context_sha256s
        or authenticated.runtime_parameter_schema_sha256
        != recaptured.runtime_parameter_schema_sha256
        or authenticated.source_semantics_sha256 != recaptured.source_semantics_sha256
        or authenticated.current_dimensions != recaptured.current_dimensions
        or authenticated.current_count != recaptured.current_count
        or authenticated.observation_batch_sha256 != recaptured.observation_batch_sha256
        or authenticated.capture_contract_sha256 != recaptured.capture_contract_sha256
        or authenticated.context_policy != recaptured.context_policy
    ):
        raise ValueError(
            "recurrence baseline application reference did not reproduce its "
            "authenticated capture commitment"
        )


def _relation_residuals(
    relation_kind: _RelationKind,
    current: Sequence[_ComplexDecimal],
    representative: Sequence[_ComplexDecimal] | None,
    *,
    relative_tolerance: _Tolerance,
    absolute_tolerance: _Tolerance,
) -> _Residuals:
    if relation_kind == "zero":
        return _zero_residuals(
            current,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    if representative is None or len(current) != len(representative):
        raise ValueError("recurrence numerical relation width is invalid")
    return _pair_residuals(
        current,
        representative,
        sign=1 if relation_kind == "equal" else -1,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )


def _zero_residuals(
    current: Sequence[_ComplexDecimal],
    *,
    relative_tolerance: _Tolerance,
    absolute_tolerance: _Tolerance,
) -> _Residuals:
    maximum_absolute = Fraction(0)
    maximum_relative = Fraction(0)
    maximum_ratio = Fraction(0)
    relative_fraction = _tolerance_fraction(relative_tolerance)
    absolute_fraction = _tolerance_fraction(absolute_tolerance)
    for current_value in current:
        current_real = Fraction(current_value[0])
        current_imaginary = Fraction(current_value[1])
        difference = max(abs(current_real), abs(current_imaginary))
        allowed = absolute_fraction + relative_fraction * difference
        relative = (
            Fraction(0)
            if difference == 0
            else difference / max(difference, absolute_fraction)
        )
        ratio = Fraction(0) if difference == 0 else difference / allowed
        maximum_absolute = max(maximum_absolute, difference)
        maximum_relative = max(maximum_relative, relative)
        maximum_ratio = max(maximum_ratio, ratio)
    return maximum_absolute, maximum_relative, maximum_ratio


def _pair_residuals(
    left: Sequence[_ComplexDecimal],
    right: Sequence[_ComplexDecimal],
    *,
    sign: int,
    relative_tolerance: _Tolerance,
    absolute_tolerance: _Tolerance,
) -> _Residuals:
    if len(left) != len(right):
        raise ValueError("recurrence numerical comparison width is invalid")
    maximum_absolute = Fraction(0)
    maximum_relative = Fraction(0)
    maximum_ratio = Fraction(0)
    relative_fraction = _tolerance_fraction(relative_tolerance)
    absolute_fraction = _tolerance_fraction(absolute_tolerance)
    for left_value, right_value in zip(left, right, strict=True):
        left_real = Fraction(left_value[0])
        left_imaginary = Fraction(left_value[1])
        right_real = Fraction(right_value[0])
        right_imaginary = Fraction(right_value[1])
        signed = (
            (right_real, right_imaginary)
            if sign == 1
            else (-right_real, -right_imaginary)
        )
        difference = max(
            abs(left_real - signed[0]),
            abs(left_imaginary - signed[1]),
        )
        scale = max(
            abs(left_real),
            abs(left_imaginary),
            abs(right_real),
            abs(right_imaginary),
        )
        allowed = absolute_fraction + relative_fraction * scale
        relative = (
            Fraction(0)
            if difference == 0
            else difference / max(scale, absolute_fraction)
        )
        ratio = Fraction(0) if difference == 0 else difference / allowed
        maximum_absolute = max(maximum_absolute, difference)
        maximum_relative = max(maximum_relative, relative)
        maximum_ratio = max(maximum_ratio, ratio)
    return maximum_absolute, maximum_relative, maximum_ratio


def _tolerance_fraction(value: _Tolerance) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _relation_observation_sha256(
    current_id: int,
    representative_id: int | None,
    relation_kind: _RelationKind,
    capture: RecurrenceCurrentObservationCapture,
) -> str:
    return _canonical_sha256(
        {
            "abi": _RELATION_OBSERVATION_ABI,
            "capture_contract_sha256": capture.capture_contract_sha256,
            "current_id": current_id,
            "representative_id": representative_id,
            "relation_kind": relation_kind,
            "current_dimension": capture.current_dimensions[current_id],
            "current_values": [
                [_decimal_string(real), _decimal_string(imaginary)]
                for real, imaginary in capture.observations[current_id]
            ],
            "representative_values": (
                None
                if representative_id is None
                else [
                    [_decimal_string(real), _decimal_string(imaginary)]
                    for real, imaginary in capture.observations[representative_id]
                ]
            ),
        }
    )


def _runtime_parameter_defaults(
    plan: _RecurrenceExactPlan,
) -> tuple[Decimal, ...]:
    if (
        len(plan.runtime_parameter_schema) != len(plan.runtime_defaults)
        or any(
            row.runtime_slot != index
            for index, row in enumerate(plan.runtime_parameter_schema)
        )
        or any(
            row.runtime_slot >= len(plan.runtime_defaults)
            for row in plan.parameter_projection
        )
    ):
        raise ArtifactError(
            "recurrence runtime parameter schema/defaults are incomplete"
        )
    defaults: list[Decimal] = []
    for row, default in zip(
        plan.runtime_parameter_schema, plan.runtime_defaults, strict=True
    ):
        if row.probe_policy == "derived-overwritten-fixed-zero-v1":
            defaults.append(Decimal(0))
            continue
        if row.probe_policy != "native-template-default-perturbed-v1":
            raise ArtifactError(
                "recurrence runtime parameter schema has an unsupported "
                "numerical probe policy"
            )
        binary64_default = float(default)
        if not isfinite(binary64_default):
            raise ArtifactError(
                "recurrence runtime parameter default is not finite binary64"
            )
        defaults.append(Decimal.from_float(binary64_default))
    return tuple(defaults)


def _runtime_parameter_schema_payload(
    plan: _RecurrenceExactPlan,
    *,
    defaults: tuple[Decimal, ...] | None = None,
) -> dict[str, object]:
    if defaults is None:
        defaults = _runtime_parameter_defaults(plan)
    return {
        "abi": _PARAMETER_SCHEMA_ABI,
        "parameters": [
            {
                "runtime_slot": row.runtime_slot,
                "runtime_name": row.runtime_name,
                "parameter_template_id": row.parameter_template_id,
                "prepared_parameter_id": row.prepared_parameter_id,
                "component": row.component,
                "default_binary64": float(default).hex(),
                "probe_policy": row.probe_policy,
            }
            for row, default in zip(
                plan.runtime_parameter_schema, defaults, strict=True
            )
        ],
    }


def _build_warmup_static_context(
    plan: _RecurrenceExactPlan,
) -> _WarmupStaticContext:
    """Build immutable generation-scoped data shared by every warm-up pass."""

    current_dimensions = {
        current.semantic_id: current.component_count
        for current in plan.sections.currents
    }
    current_contracts = _current_contracts(plan.sections)
    source_semantics = _source_semantics_payload(
        plan.sections,
        contracts=current_contracts,
    )
    source_semantics_sha256 = _canonical_sha256(source_semantics)
    runtime_defaults = _runtime_parameter_defaults(plan)
    runtime_parameter_schema = _runtime_parameter_schema_payload(
        plan,
        defaults=runtime_defaults,
    )
    return _WarmupStaticContext(
        current_dimensions=current_dimensions,
        current_contracts=current_contracts,
        runtime_defaults=runtime_defaults,
        runtime_parameter_schema=runtime_parameter_schema,
        runtime_parameter_schema_sha256=_canonical_sha256(runtime_parameter_schema),
        source_semantics=source_semantics,
        source_semantics_sha256=source_semantics_sha256,
    )


def _build_parameter_probe_contexts(
    defaults: tuple[Decimal, ...],
    *,
    probe_policies: tuple[str, ...],
    precision_digits: int,
    seed: int,
    domain: str,
    count: int,
    include_defaults: bool,
) -> tuple[tuple[Decimal, ...], ...]:
    if (
        type(precision_digits) is not int
        or precision_digits < 80
        or type(seed) is not int
        or seed < 0
        or not isinstance(domain, str)
        or not domain
        or type(count) is not int
        or count < 2
        or len(probe_policies) != len(defaults)
        or any(
            policy
            not in {
                "native-template-default-perturbed-v1",
                "derived-overwritten-fixed-zero-v1",
            }
            for policy in probe_policies
        )
        or any(not value.is_finite() for value in defaults)
    ):
        raise ValueError("recurrence numerical parameter-probe contract is invalid")
    contexts: list[tuple[Decimal, ...]] = []
    for probe_index in range(count):
        if include_defaults and probe_index == 0:
            contexts.append(defaults)
            continue
        values: list[Decimal] = []
        for parameter_index, (default, probe_policy) in enumerate(
            zip(defaults, probe_policies, strict=True)
        ):
            if probe_policy == "derived-overwritten-fixed-zero-v1":
                values.append(default)
                continue
            digest = hashlib.sha256(
                (f"{seed}:{domain}:{probe_index}:{parameter_index}").encode("ascii")
            ).digest()
            signed = int.from_bytes(digest[:8], "big") - (1 << 63)
            if signed == 0:
                signed = 1
            scale = max(abs(Fraction(default)), Fraction(1))
            value = Fraction(default) + scale * Fraction(signed, 1 << 67)
            values.append(_terminating_fraction_decimal(value))
        contexts.append(tuple(values))
    return tuple(contexts)


def _validated_tolerances(
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[Decimal, Decimal]:
    if (
        isinstance(relative_tolerance, bool)
        or not isinstance(relative_tolerance, int | float)
        or isinstance(absolute_tolerance, bool)
        or not isinstance(absolute_tolerance, int | float)
        or not isfinite(relative_tolerance)
        or not isfinite(absolute_tolerance)
        or relative_tolerance < 0
        or absolute_tolerance < 0
        or (
            float(relative_tolerance) == 0.0
            and copysign(1.0, float(relative_tolerance)) < 0
        )
        or (
            float(absolute_tolerance) == 0.0
            and copysign(1.0, float(absolute_tolerance)) < 0
        )
        or (relative_tolerance == 0 and absolute_tolerance == 0)
    ):
        raise ValueError("recurrence numerical current tolerances are invalid")
    return (
        Decimal.from_float(float(relative_tolerance)),
        Decimal.from_float(float(absolute_tolerance)),
    )


def _certificate_set_sha256(
    certificates: Sequence[RecurrenceNumericalCurrentCertificate],
) -> str:
    return _canonical_sha256(
        {
            "abi": _RELATION_SET_ABI,
            "certificates": _CertificateRowsStream(certificates),
            "mappings": _MappingRowsStream(certificates),
        }
    )


def _kinematic_binary64(
    point: ValidationPointRecord,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(float(value).hex() for value in momentum)
        for momentum in point.four_vectors
    )


def _kinematic_sha256(point: ValidationPointRecord) -> str:
    return _canonical_sha256(_kinematic_binary64(point))


def _domain_seed(seed: int, *, domain: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{domain}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def _context_policy(strategy: str) -> str:
    return {
        "topology-replay": "seeded-replay-target-per-physical-point-v1",
        "all-flow-union": ("seeded-helicity-per-selector-domain-and-physical-point-v1"),
        "contracted-color-union": "fixed-contracted-source-schedule-v1",
    }[strategy]


def _plain_contract(contract: _CurrentContract) -> list[object]:
    result = []
    for value in contract:
        result.append(list(value) if isinstance(value, tuple) else value)
    return result


def _warning_payload(required: bool) -> dict[str, object]:
    return (
        {
            "required": True,
            "emit": "once-per-generated-artifact",
            "code": NUMERICAL_CURRENT_RELATION_WARNING_CODE,
            "message": NUMERICAL_CURRENT_RELATION_WARNING,
        }
        if required
        else {
            "required": False,
            "emit": "never",
            "code": None,
            "message": None,
        }
    )


def numerical_relation_correctness_payload(
    applied_relation_count: int,
) -> dict[str, object]:
    if type(applied_relation_count) is not int or applied_relation_count < 0:
        raise ValueError("numerical relation application count is invalid")
    return {
        "abi": NUMERICAL_RELATION_CORRECTNESS_ABI,
        "state": (
            "member-scoped-v1" if applied_relation_count else "no-applied-relations"
        ),
        "applied_relation_count": applied_relation_count,
    }


def recurrence_numerical_relation_correctness_summary(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Extract the compact metadata needed for lightweight result reuse."""

    value = report.get("relation_correctness")
    if not isinstance(value, Mapping):
        raise ValueError(
            "recurrence numerical relation report has no correctness state"
        )
    applied = report.get("applied_relation_count")
    if type(applied) is not int or applied < 0:
        raise ValueError(
            "recurrence numerical relation report has an invalid applied count"
        )
    expected = numerical_relation_correctness_payload(applied)
    if any(value.get(key) != expected[key] for key in expected):
        raise ValueError(
            "recurrence numerical relation correctness state is inconsistent"
        )
    return dict(expected)


def validate_recurrence_numerical_evidence_fallback(
    value: object,
) -> dict[str, object]:
    """Validate one full evidence-envelope fallback payload."""

    required_fields = {
        "reason",
        "memory_envelope_bytes",
        "spooled_producer_resident_bytes",
        "geometry",
    }
    if not isinstance(value, Mapping) or set(value) != required_fields:
        raise ValueError("recurrence numerical current fallback provenance is invalid")
    if value.get("reason") != EVIDENCE_ENVELOPE_FALLBACK_REASON:
        raise ValueError("recurrence numerical current fallback reason is invalid")

    def required_integer(
        raw: object,
        *,
        name: str,
        minimum: int,
    ) -> int:
        if type(raw) is not int or raw < minimum:
            raise ValueError(f"recurrence numerical fallback {name} is invalid")
        return raw

    memory_envelope_bytes = required_integer(
        value.get("memory_envelope_bytes"),
        name="memory envelope",
        minimum=1,
    )
    producer_resident_bytes = required_integer(
        value.get("spooled_producer_resident_bytes"),
        name="producer resident bound",
        minimum=1,
    )
    raw_geometry = value.get("geometry")
    geometry_fields = {
        "current_count",
        "component_count",
        "candidate_probe_count",
        "verification_probe_count",
        "runtime_parameter_count",
        "scalar_count",
        "row_count",
    }
    if not isinstance(raw_geometry, Mapping) or set(raw_geometry) != geometry_fields:
        raise ValueError("recurrence numerical current fallback geometry is invalid")
    counts = {
        key: required_integer(
            raw_geometry.get(key),
            name=f"geometry {key}",
            minimum=(
                1 if key in {"candidate_probe_count", "verification_probe_count"} else 0
            ),
        )
        for key in geometry_fields
    }
    geometry = RecurrenceNumericalEvidenceGeometry.from_counts(
        current_count=counts["current_count"],
        component_count=counts["component_count"],
        candidate_probe_count=counts["candidate_probe_count"],
        verification_probe_count=counts["verification_probe_count"],
        runtime_parameter_count=counts["runtime_parameter_count"],
    )
    expected_geometry = geometry.to_json_dict()
    expected_producer_resident_bytes = geometry.spooled_producer_resident_upper_bound()
    # A fallback may occur either before capture because this producer bound
    # exceeds the envelope, or later at a recognized canonical/native
    # transport boundary.  In both cases the recomputed bound must be exact;
    # it need not itself be the boundary that fired.
    if (
        counts != expected_geometry
        or producer_resident_bytes != expected_producer_resident_bytes
    ):
        raise ValueError(
            "recurrence numerical current fallback geometry is inconsistent"
        )
    return {
        "reason": EVIDENCE_ENVELOPE_FALLBACK_REASON,
        "memory_envelope_bytes": memory_envelope_bytes,
        "spooled_producer_resident_bytes": producer_resident_bytes,
        "geometry": expected_geometry,
    }


def recurrence_numerical_relation_fallback_summary(
    report: Mapping[str, object],
) -> dict[str, object] | None:
    """Extract lightweight fallback provenance without expanding correctness ABI."""

    fallback = report.get("fallback")
    if fallback is None:
        if (
            report.get("requested_mode") == "certified-reuse"
            and report.get("effective_mode") == "off"
        ):
            raise ValueError(
                "recurrence numerical relation fallback provenance is absent"
            )
        return None
    certified = _required_report_integer(report, "certified_relation_count")
    applied = _required_report_integer(report, "applied_relation_count")
    if (
        report.get("requested_mode") != "certified-reuse"
        or report.get("effective_mode") != "off"
        or report.get("effective_reuse_state") != "disabled"
        or certified != 0
        or applied != 0
    ):
        raise ValueError("recurrence numerical relation fallback state is inconsistent")
    validated = validate_recurrence_numerical_evidence_fallback(fallback)
    return {
        "abi": NUMERICAL_RELATION_FALLBACK_ABI,
        "requested_mode": "certified-reuse",
        "effective_mode": "off",
        "effective_reuse_state": "disabled",
        "reason": EVIDENCE_ENVELOPE_FALLBACK_REASON,
        "geometry": dict(cast(Mapping[str, int], validated["geometry"])),
        "certified_relation_count": 0,
        "applied_relation_count": 0,
    }


def _decimal_string(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("non-finite Decimal cannot be canonicalized")
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", "+0", ""} else text


def _fraction_string(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _residual_strings(residuals: _Residuals) -> _ResidualStrings:
    return (
        _fraction_string(residuals[0]),
        _fraction_string(residuals[1]),
        _fraction_string(residuals[2]),
    )


def _terminating_fraction_decimal(value: Fraction) -> Decimal:
    denominator = value.denominator
    twos = 0
    fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError("recurrence parameter context is not a finite decimal")
    places = max(twos, fives)
    scaled = value.numerator
    scaled *= 2 ** (places - twos)
    scaled *= 5 ** (places - fives)
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled))
    if places == 0:
        return Decimal(f"{sign}{digits}")
    digits = digits.rjust(places + 1, "0")
    return Decimal(f"{sign}{digits[:-places]}.{digits[-places:]}")


def _canonical_sha256(value: object) -> str:
    digest = hashlib.sha256()
    for chunk in _iter_canonical_json_chunks(value):
        digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    if isinstance(value, _CanonicalJsonStreamValue):
        return b"".join(_iter_canonical_json_chunks(value))
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _iter_canonical_json_chunks(value: object) -> Iterator[bytes]:
    """Yield the unique ASCII encoding accepted at the native boundary."""

    if isinstance(value, _CanonicalJsonStreamValue):
        yield from value.iter_canonical_json_chunks()
        return
    if value is None:
        yield b"null"
        return
    if value is True:
        yield b"true"
        return
    if value is False:
        yield b"false"
        return
    if type(value) is int:
        yield str(value).encode("ascii")
        return
    if isinstance(value, str):
        yield json.dumps(value, ensure_ascii=True).encode("ascii")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        yield b"{"
        for index, key in enumerate(sorted(value)):
            if index:
                yield b","
            yield json.dumps(key, ensure_ascii=True).encode("ascii")
            yield b":"
            yield from _iter_canonical_json_chunks(value[key])
        yield b"}"
        return
    if isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    ):
        yield b"["
        for index, item in enumerate(value):
            if index:
                yield b","
            yield from _iter_canonical_json_chunks(item)
        yield b"]"
        return
    if isinstance(value, float):
        raise TypeError(
            "canonical recurrence evidence uses binary64 hex strings, not "
            "JSON float spellings"
        )
    raise TypeError(f"unsupported canonical JSON value {type(value).__name__}")


def _bounded_canonical_json_bytes(
    value: object,
    *,
    byte_limit: int | None,
    label: str,
) -> bytes:
    """Encode canonically, honoring an explicitly requested byte limit."""

    if byte_limit is not None and byte_limit <= 0:
        raise ValueError(f"{label} byte limit is invalid")
    encoded = bytearray()
    for chunk in _iter_canonical_json_chunks(value):
        next_size = len(encoded) + len(chunk)
        if byte_limit is not None and next_size > byte_limit:
            raise _CanonicalEvidenceCapacityExceeded(
                f"{label} exceeds the explicit {byte_limit}-byte canonical boundary"
            )
        encoded.extend(chunk)
    return bytes(encoded)


def _write_compressed_canonical_evidence(
    value: object,
    stream: BinaryIO,
    *,
    canonical_byte_limit: int | None,
) -> int:
    """Stream canonical JSON into one zlib transport envelope."""

    if canonical_byte_limit is not None and canonical_byte_limit <= 0:
        raise ValueError("recurrence compressed evidence canonical boundary is invalid")
    stream.seek(0)
    stream.truncate(0)
    stream.write(b"\0" * _COMPRESSED_EVIDENCE_HEADER.size)
    compressor = zlib.compressobj(level=1, wbits=zlib.MAX_WBITS)
    digest = hashlib.sha256()
    canonical_byte_count = 0
    transport_byte_count = _COMPRESSED_EVIDENCE_HEADER.size

    def write_compressed(chunk: bytes) -> None:
        nonlocal transport_byte_count
        if not chunk:
            return
        transport_byte_count += len(chunk)
        stream.write(chunk)

    for chunk in _iter_canonical_json_chunks(value):
        canonical_byte_count += len(chunk)
        if (
            canonical_byte_limit is not None
            and canonical_byte_count > canonical_byte_limit
        ):
            raise _CanonicalEvidenceCapacityExceeded(
                "recurrence numerical raw evidence exceeds the explicit "
                f"{canonical_byte_limit}-byte decompression boundary"
            )
        digest.update(chunk)
        write_compressed(compressor.compress(chunk))
    write_compressed(compressor.flush())
    stream.seek(0)
    stream.write(
        _COMPRESSED_EVIDENCE_HEADER.pack(
            _COMPRESSED_EVIDENCE_MAGIC,
            canonical_byte_count,
            digest.digest(),
        )
    )
    stream.flush()
    stream.seek(0, 2)
    if stream.tell() != transport_byte_count:
        raise ValueError("recurrence compressed evidence spool length drifted")
    return canonical_byte_count


def _read_compressed_evidence_spool(stream: BinaryIO) -> bytes:
    stream.seek(0)
    encoded = stream.read()
    if len(encoded) < _COMPRESSED_EVIDENCE_HEADER.size:
        raise ValueError("recurrence compressed evidence spool is truncated")
    magic, _canonical_byte_count, _digest = _COMPRESSED_EVIDENCE_HEADER.unpack_from(
        encoded
    )
    if magic != _COMPRESSED_EVIDENCE_MAGIC:
        raise ValueError("recurrence compressed evidence spool has invalid magic")
    return encoded


def _validate_raw_evidence_canonical_size(
    size: int,
    *,
    byte_limit: int | None = None,
) -> None:
    if size < 0 or (byte_limit is not None and (byte_limit <= 0 or size > byte_limit)):
        raise ValueError(
            "recurrence numerical raw evidence exceeds its explicit memory envelope "
            f"(size={size}, byte_limit={byte_limit})"
        )


def _raw_evidence_wire_byte_limit(
    *,
    scalar_count: int,
    row_count: int,
    memory_envelope_bytes: int = _RAW_EVIDENCE_MEMORY_TARGET_BYTES,
) -> int:
    """Return the per-shape wire allowance for a supplied batching target."""

    if (
        scalar_count < 0
        or row_count < 0
        or type(memory_envelope_bytes) is not int
        or memory_envelope_bytes <= 0
    ):
        raise ValueError("recurrence raw-evidence memory geometry is invalid")
    resident_without_wire = _raw_evidence_memory_upper_bound(
        0,
        scalar_count=scalar_count,
        row_count=row_count,
    )
    minimum_resident = resident_without_wire + (
        _MIN_RAW_EVIDENCE_WIRE_BYTES * _RAW_EVIDENCE_WIRE_PEAK_COPIES
    )
    if minimum_resident > memory_envelope_bytes:
        raise _RawEvidenceWireEnvelopeExceeded(
            "recurrence raw-evidence capture geometry leaves less than the "
            f"required {_MIN_RAW_EVIDENCE_WIRE_BYTES}-byte canonical wire "
            f"reserve inside its {memory_envelope_bytes}-byte memory "
            f"envelope (scalars={scalar_count}, rows={row_count}, "
            f"resident_without_wire={resident_without_wire})"
        )
    return (
        memory_envelope_bytes - resident_without_wire
    ) // _RAW_EVIDENCE_WIRE_PEAK_COPIES


def _raw_evidence_geometry_counts(
    sections: _RecurrenceExactSectionsV1,
    *,
    candidate_probe_count: int,
    verification_probe_count: int,
    runtime_parameter_count: int,
) -> tuple[int, int, int, int]:
    if (
        candidate_probe_count <= 0
        or verification_probe_count <= 0
        or runtime_parameter_count < 0
    ):
        raise ValueError("recurrence raw-evidence probe geometry is invalid")
    geometry = RecurrenceNumericalEvidenceGeometry.from_counts(
        current_count=len(sections.currents),
        component_count=sum(current.component_count for current in sections.currents),
        candidate_probe_count=candidate_probe_count,
        verification_probe_count=verification_probe_count,
        runtime_parameter_count=runtime_parameter_count,
    )
    return (
        geometry.current_count,
        geometry.component_count,
        geometry.scalar_count,
        geometry.row_count,
    )


def _spooled_capture_memory_upper_bound(
    *,
    current_count: int,
    component_count: int,
    maximum_probe_count: int,
    runtime_parameter_count: int,
    include_candidate_index: bool = True,
) -> int:
    """Bound one capture graph plus the global index before both are spooled."""

    values = (
        current_count,
        component_count,
        maximum_probe_count,
        runtime_parameter_count,
    )
    if any(value < 0 for value in values) or maximum_probe_count == 0:
        raise ValueError("recurrence spooled capture geometry is invalid")
    scalar_count = (
        2 * component_count * maximum_probe_count
        + runtime_parameter_count * maximum_probe_count
    )
    row_count = current_count + runtime_parameter_count
    return (
        _RAW_EVIDENCE_FIXED_RESERVE_BYTES
        + scalar_count * _RAW_EVIDENCE_SCALAR_RESIDENT_BYTES
        + row_count * _RAW_EVIDENCE_ROW_RESIDENT_BYTES
        + _SPOOLED_CAPTURE_COMPRESSION_RESERVE_BYTES
        + (
            current_count * _SPOOLED_CANDIDATE_INDEX_BYTES_PER_CURRENT
            if include_candidate_index
            else 0
        )
    )


def _select_raw_evidence_storage_geometry(
    sections: _RecurrenceExactSectionsV1,
    *,
    candidate_probe_count: int,
    verification_probe_count: int,
    runtime_parameter_count: int,
) -> _RawEvidenceStorageGeometry:
    geometry = RecurrenceNumericalEvidenceGeometry.from_counts(
        current_count=len(sections.currents),
        component_count=sum(current.component_count for current in sections.currents),
        candidate_probe_count=candidate_probe_count,
        verification_probe_count=verification_probe_count,
        runtime_parameter_count=runtime_parameter_count,
    )
    return _select_raw_evidence_storage_geometry_for_counts(geometry)


def _select_raw_evidence_storage_geometry_for_counts(
    geometry: RecurrenceNumericalEvidenceGeometry,
    *,
    memory_envelope_bytes: int | None = None,
) -> _RawEvidenceStorageGeometry:
    """Select storage from scalar geometry before allocating observations."""

    if memory_envelope_bytes is not None and (
        type(memory_envelope_bytes) is not int or memory_envelope_bytes <= 0
    ):
        raise ValueError("recurrence raw-evidence memory envelope is invalid")
    try:
        byte_limit = _raw_evidence_wire_byte_limit(
            scalar_count=geometry.scalar_count,
            row_count=geometry.row_count,
            memory_envelope_bytes=(
                memory_envelope_bytes or _RAW_EVIDENCE_MEMORY_TARGET_BYTES
            ),
        )
    except _RawEvidenceWireEnvelopeExceeded as raw_error:
        return _compressed_raw_evidence_storage_geometry_for_counts(
            geometry,
            memory_envelope_bytes=memory_envelope_bytes,
            raw_reason=str(raw_error),
        )
    return _RawEvidenceStorageGeometry(
        scalar_count=geometry.scalar_count,
        row_count=geometry.row_count,
        canonical_byte_limit=byte_limit if memory_envelope_bytes is not None else None,
        encoding=_RAW_EVIDENCE_ENCODING,
        producer_resident_upper_bound=_raw_evidence_memory_upper_bound(
            _MIN_RAW_EVIDENCE_WIRE_BYTES,
            scalar_count=geometry.scalar_count,
            row_count=geometry.row_count,
        ),
    )


def _compressed_raw_evidence_storage_geometry_for_counts(
    geometry: RecurrenceNumericalEvidenceGeometry,
    *,
    memory_envelope_bytes: int | None = None,
    raw_reason: str,
) -> _RawEvidenceStorageGeometry:
    """Select the sequential compressed spool or return the typed fallback."""

    producer_bound = geometry.spooled_producer_resident_upper_bound()
    if memory_envelope_bytes is not None and producer_bound > memory_envelope_bytes:
        raise RecurrenceNumericalEvidenceEnvelopeExceeded(
            geometry,
            memory_envelope_bytes=memory_envelope_bytes,
            spooled_producer_resident_bytes=producer_bound,
            raw_reason=raw_reason,
        )
    return _RawEvidenceStorageGeometry(
        scalar_count=geometry.scalar_count,
        row_count=geometry.row_count,
        canonical_byte_limit=None,
        encoding=_COMPRESSED_EVIDENCE_ENCODING,
        producer_resident_upper_bound=producer_bound,
    )


def recurrence_numerical_evidence_capacity_outcome(
    sections: _RecurrenceExactSectionsV1,
    *,
    candidate_probe_count: int,
    verification_probe_count: int,
    runtime_parameter_count: int,
    error: BaseException,
) -> RecurrenceNumericalEvidenceEnvelopeExceeded | None:
    """Map only recognized native optional-evidence capacity failures.

    Integrity, lexical, structural, and relation-correctness failures remain
    fail-closed.  The native messages listed here are the bounded-memory exits
    emitted after (or while establishing) authenticated evidence geometry.
    """

    reason = str(error)
    if not any(
        marker in reason for marker in _NATIVE_OPTIONAL_EVIDENCE_CAPACITY_MARKERS
    ):
        return None
    geometry = RecurrenceNumericalEvidenceGeometry.from_counts(
        current_count=len(sections.currents),
        component_count=sum(current.component_count for current in sections.currents),
        candidate_probe_count=candidate_probe_count,
        verification_probe_count=verification_probe_count,
        runtime_parameter_count=runtime_parameter_count,
    )
    return RecurrenceNumericalEvidenceEnvelopeExceeded(
        geometry,
        memory_envelope_bytes=_RAW_EVIDENCE_MEMORY_TARGET_BYTES,
        spooled_producer_resident_bytes=(
            geometry.spooled_producer_resident_upper_bound()
        ),
        raw_reason=reason,
    )


def _validate_raw_evidence_geometry(
    sections: _RecurrenceExactSectionsV1,
    *,
    candidate_probe_count: int,
    verification_probe_count: int,
    runtime_parameter_count: int,
) -> tuple[int, int, None]:
    """Validate capture dimensions without imposing a memory ceiling."""

    _current_count, _component_count, scalar_count, row_count = (
        _raw_evidence_geometry_counts(
            sections,
            candidate_probe_count=candidate_probe_count,
            verification_probe_count=verification_probe_count,
            runtime_parameter_count=runtime_parameter_count,
        )
    )
    return scalar_count, row_count, None


def _raw_evidence_memory_upper_bound(
    size: int,
    *,
    scalar_count: int,
    row_count: int,
) -> int:
    if size < 0 or scalar_count < 0 or row_count < 0:
        raise ValueError("recurrence raw-evidence memory geometry is invalid")
    return (
        _RAW_EVIDENCE_FIXED_RESERVE_BYTES
        + scalar_count * _RAW_EVIDENCE_SCALAR_RESIDENT_BYTES
        + row_count * _RAW_EVIDENCE_ROW_RESIDENT_BYTES
        + size * _RAW_EVIDENCE_WIRE_PEAK_COPIES
    )


def _synthetic_raw_evidence_bytes(
    current_count: int,
    *,
    component_count: int,
    candidate_probe_count: int,
    verification_probe_count: int,
    decimal_characters: int,
) -> int:
    """Measure the canonical raw-observation footprint without allocating it."""

    if (
        current_count < 0
        or component_count <= 0
        or candidate_probe_count <= 0
        or verification_probe_count <= 0
        or decimal_characters <= 0
    ):
        raise ValueError("synthetic raw-evidence geometry is invalid")
    return _synthetic_raw_evidence_bytes_by_dimension_counts(
        {component_count: current_count},
        candidate_probe_count=candidate_probe_count,
        verification_probe_count=verification_probe_count,
        decimal_characters=decimal_characters,
    )


def _synthetic_raw_evidence_bytes_by_dimension_counts(
    dimension_counts: Mapping[int, int],
    *,
    candidate_probe_count: int,
    verification_probe_count: int,
    decimal_characters: int,
) -> int:
    """Measure a mixed-dimension canonical raw-observation footprint."""

    if (
        not dimension_counts
        or candidate_probe_count <= 0
        or verification_probe_count <= 0
        or decimal_characters <= 0
        or any(
            dimension <= 0 or count < 0 for dimension, count in dimension_counts.items()
        )
    ):
        raise ValueError("synthetic raw-evidence geometry is invalid")
    dimensions = tuple(
        dimension
        for dimension, count in sorted(dimension_counts.items())
        for _ in range(count)
    )
    current_count = len(dimensions)
    scalar = "1" + "2" * (decimal_characters - 1)

    def array_size(point_count: int) -> int:
        total = 2 + max(0, current_count - 1)
        for current_id, dimension in enumerate(dimensions):
            values = [[scalar, f"-{scalar}"]] * (dimension * point_count)
            total += len(
                _canonical_json_bytes(
                    {
                        "current_id": current_id,
                        "dimension": dimension,
                        "values": values,
                    }
                )
            )
        return total

    skeleton_size = len(
        _canonical_json_bytes(
            {
                "candidate_capture": {"observations": []},
                "verification_capture": {"observations": []},
            }
        )
    )
    return (
        skeleton_size
        + array_size(candidate_probe_count)
        + array_size(verification_probe_count)
        - 4
    )


__all__ = [
    "EVIDENCE_ENVELOPE_FALLBACK_REASON",
    "EVIDENCE_ENVELOPE_FALLBACK_WARNING",
    "NUMERICAL_RELATION_CORRECTNESS_ABI",
    "NUMERICAL_RELATION_FALLBACK_ABI",
    "RecurrenceCurrentObservationCapture",
    "RecurrenceNumericalCurrentCertificate",
    "RecurrenceNumericalCurrentWarmupResult",
    "RecurrenceNumericalEvidenceEnvelopeExceeded",
    "RecurrenceNumericalEvidenceGeometry",
    "build_recurrence_numerical_current_probe_points",
    "capture_recurrence_current_observations",
    "numerical_relation_correctness_payload",
    "recurrence_numerical_current_envelope_fallback_report",
    "recurrence_numerical_current_opt_out_report",
    "recurrence_numerical_evidence_capacity_outcome",
    "recurrence_numerical_relation_correctness_summary",
    "recurrence_numerical_relation_fallback_summary",
    "recurrence_numerical_source_semantics_sha256",
    "run_recurrence_numerical_current_warmup",
    "validate_recurrence_numerical_current_application",
    "validate_recurrence_numerical_evidence_fallback",
]
