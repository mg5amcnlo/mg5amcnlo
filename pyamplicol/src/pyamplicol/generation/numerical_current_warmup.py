# SPDX-License-Identifier: 0BSD
"""High-precision current snapshots for bounded relation-discovery warm-up."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from itertools import pairwise
from math import isfinite
from typing import Any, Literal

from ..evaluators.symbolica_compile import _compile_symbolica_outputs
from ..evaluators.symbolica_settings import SymbolicaEvaluatorSettings
from ..models.base import Model
from ..runtime.symbolica_exact import (
    _decimal,
    _fill_momenta,
    _fill_sources,
)
from .contracts import RuntimeExpressionSchema, StageCompilationInput
from .dag_equivalence import (
    NUMERICAL_CURRENT_CAPTURE_ABI,
    NUMERICAL_CURRENT_CAPTURE_OUTPUT_PARTITION_ABI,
    NumericalCurrentObservationDiscoveryResult,
    NumericalCurrentRelationApplicationResult,
    NumericalCurrentRelationCertificate,
    apply_numerical_current_relation_certificates,
    discover_generic_dag_numerical_current_relations,
    generic_dag_numerical_capture_output_partition_sha256,
)
from .dag_types import GenericDAG
from .recurrence_numerical_current_warmup import (
    numerical_relation_correctness_payload,
)
from .runtime_schema import build_runtime_expression_schema
from .stage_planning import build_generic_stage_compiler_blueprint
from .stage_types import GenericCompiledStageBlueprint
from .validation import (
    ValidationPointRecord,
    build_validation_point,
    rotate_validation_point,
)

_ComplexDecimal = tuple[Decimal, Decimal]
_CAPTURE_ABI = NUMERICAL_CURRENT_CAPTURE_ABI
_WARMUP_ABI = "pyamplicol-generic-dag-numerical-current-warmup-v1"
_CAPTURE_OUTPUT_PARTITION_ABI = NUMERICAL_CURRENT_CAPTURE_OUTPUT_PARTITION_ABI
_NUMERICAL_DECIMAL_GUARD_DIGITS = 16


@dataclass(frozen=True, slots=True)
class GenericDAGCurrentObservationCapture:
    """Complete point-major current observations and replay provenance."""

    precision_digits: int
    points: tuple[ValidationPointRecord, ...]
    point_sha256s: tuple[str, ...]
    kinematic_sha256s: tuple[str, ...]
    parameter_contexts: tuple[tuple[Decimal, ...], ...]
    parameter_context_sha256s: tuple[str, ...]
    observations: Mapping[int, tuple[_ComplexDecimal, ...]]
    runtime_schema_sha256: str
    model_parameter_schema_sha256: str
    source_dag_sha256: str
    evaluator_output_partition_abi: str
    evaluator_output_partition_sha256: str
    observation_batch_sha256: str
    capture_contract_sha256: str

    @property
    def point_count(self) -> int:
        return len(self.point_sha256s)

    @property
    def current_count(self) -> int:
        return len(self.observations)

    def to_provenance_dict(self) -> dict[str, object]:
        return {
            "abi": _CAPTURE_ABI,
            "precision_digits": self.precision_digits,
            "point_count": self.point_count,
            "point_sha256s": list(self.point_sha256s),
            "kinematic_sha256s": list(self.kinematic_sha256s),
            "points": [point.to_mapping() for point in self.points],
            "parameter_contexts": [
                [_decimal_string(value) for value in context]
                for context in self.parameter_contexts
            ],
            "parameter_context_sha256s": list(self.parameter_context_sha256s),
            "current_count": self.current_count,
            "runtime_schema_sha256": self.runtime_schema_sha256,
            "model_parameter_schema_sha256": (self.model_parameter_schema_sha256),
            "source_dag_sha256": self.source_dag_sha256,
            "evaluator_output_partition": {
                "abi": self.evaluator_output_partition_abi,
                "sha256": self.evaluator_output_partition_sha256,
            },
            "observation_batch_sha256": self.observation_batch_sha256,
            "capture_contract_sha256": self.capture_contract_sha256,
            "complete_current_component_digest": True,
            "components_embedded": False,
            "point_major": True,
            "evaluator": "symbolica-interpreted-high-precision-stage-replay",
        }


@dataclass(frozen=True, slots=True)
class _PartitionedCurrentCaptureStageEvaluator:
    """Warm-up-only exact replay over current-ID-isolated evaluator leaves."""

    evaluators: tuple[Any, ...]
    chunk_input_indices: tuple[tuple[int, ...], ...]
    output_partitions: tuple[tuple[int, int], ...]
    input_len: int
    output_len: int

    def __post_init__(self) -> None:
        if (
            type(self.input_len) is not int
            or self.input_len < 0
            or type(self.output_len) is not int
            or self.output_len < 1
            or len(self.evaluators) != len(self.chunk_input_indices)
            or len(self.evaluators) != len(self.output_partitions)
            or not self.evaluators
        ):
            raise ValueError("numerical current partitioned evaluator shape is invalid")
        expected_output_start = 0
        for evaluator, indices, partition in zip(
            self.evaluators,
            self.chunk_input_indices,
            self.output_partitions,
            strict=True,
        ):
            start, stop = partition
            if (
                start != expected_output_start
                or stop <= start
                or stop > self.output_len
                or any(
                    type(index) is not int or index < 0 or index >= self.input_len
                    for index in indices
                )
                or any(left >= right for left, right in pairwise(indices))
                or getattr(evaluator, "input_len", None) != len(indices)
            ):
                raise ValueError(
                    "numerical current partitioned evaluator mapping drifted"
                )
            expected_output_start = stop
        if expected_output_start != self.output_len:
            raise ValueError(
                "numerical current partitioned evaluator outputs are not exhaustive"
            )

    def evaluate_complex_with_prec(
        self,
        values: Sequence[_ComplexDecimal],
        precision: int,
    ) -> tuple[object, ...]:
        prepared = tuple(values)
        if len(prepared) != self.input_len:
            raise ValueError(
                "numerical current partitioned evaluator input width drifted"
            )
        outputs: list[object] = []
        for evaluator, indices, (start, stop) in zip(
            self.evaluators,
            self.chunk_input_indices,
            self.output_partitions,
            strict=True,
        ):
            source = getattr(evaluator, "_source_evaluator", None)
            evaluate = getattr(source, "evaluate_complex_with_prec", None)
            if not callable(evaluate):
                raise ValueError(
                    "numerical current warm-up evaluator leaf lacks "
                    "high-precision replay"
                )
            chunk_outputs = tuple(
                evaluate(
                    tuple(prepared[index] for index in indices),
                    precision,
                )
            )
            if len(chunk_outputs) != stop - start:
                raise ValueError(
                    "numerical current warm-up evaluator leaf returned the "
                    "wrong output width"
                )
            outputs.extend(chunk_outputs)
        if len(outputs) != self.output_len:
            raise ValueError(
                "numerical current partitioned evaluator output width drifted"
            )
        return tuple(outputs)


@dataclass(frozen=True, slots=True)
class _GenericDAGCurrentCaptureSession:
    """Reusable interpreted stage evaluators for one immutable source DAG."""

    dag: GenericDAG
    process_id: str
    runtime_schema: RuntimeExpressionSchema
    schema: Mapping[str, object]
    stages: tuple[GenericCompiledStageBlueprint, ...]
    stage_evaluators: tuple[Any, ...]
    model_parameters: tuple[Decimal, ...]
    source_dag_sha256: str
    evaluator_output_partition_abi: str
    evaluator_output_partition_sha256: str


@dataclass(frozen=True, slots=True)
class GenericDAGNumericalCurrentWarmupResult:
    """One complete discovery, verification, and application transaction."""

    dag: GenericDAG
    requested_mode: Literal["diagnostic", "certified-reuse"]
    execution_mode: Literal["compiled", "eager"]
    candidate_capture: GenericDAGCurrentObservationCapture
    verification_capture: GenericDAGCurrentObservationCapture
    application_capture: GenericDAGCurrentObservationCapture | None
    application_validation: Mapping[str, object]
    discovery: NumericalCurrentObservationDiscoveryResult
    application: NumericalCurrentRelationApplicationResult
    application_scope: Mapping[str, object]

    @property
    def warning_required(self) -> bool:
        return self.application.report.warning_required

    def to_json_dict(self) -> dict[str, object]:
        applied_relation_count = self.application.report.applied_relation_count
        return {
            "schema_version": 1,
            "abi": _WARMUP_ABI,
            "requested_mode": self.requested_mode,
            "effective_mode": self.requested_mode,
            "effective_reuse_state": "enabled",
            "state": self.application.report.state,
            "scope": {
                "execution_mode": self.execution_mode,
                "color_accuracy": str(self.dag.process.color_accuracy),
                "representation": "generic-dag",
            },
            "source_semantics": {
                "sha256": (self.application.report.source_semantics_sha256),
            },
            "candidate_capture": (self.candidate_capture.to_provenance_dict()),
            "verification_capture": (self.verification_capture.to_provenance_dict()),
            "application_capture": (
                None
                if self.application_capture is None
                else self.application_capture.to_provenance_dict()
            ),
            "application_validation": dict(self.application_validation),
            "application_scope": dict(self.application_scope),
            "discovery": self.discovery.report.to_json_dict(),
            "application": self.application.report.to_json_dict(),
            "certified_relation_count": len(self.discovery.certificates),
            "applied_relation_count": applied_relation_count,
            "relation_correctness": numerical_relation_correctness_payload(
                applied_relation_count
            ),
            "warning": (self.application.report.to_json_dict()["warning"]),
        }


def run_generic_dag_numerical_current_warmup(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str,
    mode: Literal["diagnostic", "certified-reuse"],
    execution_mode: Literal["compiled", "eager"],
    precision_digits: int,
    probe_count: int,
    verification_probe_count: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    seed: int,
    application_relation_kinds: Sequence[Literal["equal", "opposite", "zero"]] = (
        "equal",
        "opposite",
        "zero",
    ),
    application_scope_reason: str = "complete-generic-dag",
) -> GenericDAGNumericalCurrentWarmupResult:
    """Run the bounded default-on numerical relation transaction."""

    if mode not in {"diagnostic", "certified-reuse"}:
        raise ValueError(
            "generic-DAG numerical current warm-up requires diagnostic or "
            "certified-reuse mode"
        )
    allowed_relation_kinds = _validated_application_relation_kinds(
        application_relation_kinds
    )
    if not isinstance(application_scope_reason, str) or not application_scope_reason:
        raise ValueError("generic-DAG numerical relation application scope is invalid")
    candidate_points, verification_points = build_numerical_current_probe_points(
        dag,
        model,
        process_id=process_id,
        seed=seed,
        candidate_count=probe_count,
        verification_count=verification_probe_count,
    )
    baseline_session = _build_current_capture_session(
        dag,
        model,
        process_id=process_id,
    )
    candidate_parameter_contexts = _build_parameter_probe_contexts(
        baseline_session.model_parameters,
        precision_digits=precision_digits,
        seed=seed,
        domain="candidate-current-parameter-probes-v1",
        count=len(candidate_points),
        include_defaults=True,
    )
    verification_parameter_contexts = _build_parameter_probe_contexts(
        baseline_session.model_parameters,
        precision_digits=precision_digits,
        seed=seed,
        domain="independent-verification-current-parameter-probes-v1",
        count=len(verification_points),
        include_defaults=False,
    )
    candidate = capture_generic_dag_current_observations(
        dag,
        model,
        candidate_points,
        precision_digits=precision_digits,
        parameter_contexts=candidate_parameter_contexts,
        _session=baseline_session,
    )
    verification = capture_generic_dag_current_observations(
        dag,
        model,
        verification_points,
        precision_digits=precision_digits,
        parameter_contexts=verification_parameter_contexts,
        _session=baseline_session,
    )
    validate_independent_current_observation_captures(
        candidate,
        verification,
    )
    discovery = discover_generic_dag_numerical_current_relations(
        dag,
        model,
        candidate_observations=candidate.observations,
        verification_observations=verification.observations,
        candidate_point_sha256s=candidate.point_sha256s,
        verification_point_sha256s=verification.point_sha256s,
        candidate_kinematic_sha256s=candidate.kinematic_sha256s,
        verification_kinematic_sha256s=verification.kinematic_sha256s,
        candidate_parameter_context_sha256s=(candidate.parameter_context_sha256s),
        verification_parameter_context_sha256s=(verification.parameter_context_sha256s),
        runtime_schema_sha256=candidate.runtime_schema_sha256,
        source_dag_sha256=candidate.source_dag_sha256,
        evaluator_output_partition_abi=(candidate.evaluator_output_partition_abi),
        evaluator_output_partition_sha256=(candidate.evaluator_output_partition_sha256),
        candidate_capture_sha256=candidate.capture_contract_sha256,
        verification_capture_sha256=(verification.capture_contract_sha256),
        process_id=process_id,
        execution_mode=execution_mode,
        precision_digits=precision_digits,
        seed=seed,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    if (
        discovery.report.candidate_observation_batch_sha256
        != candidate.observation_batch_sha256
        or discovery.report.verification_observation_batch_sha256
        != verification.observation_batch_sha256
        or discovery.report.runtime_schema_sha256 != candidate.runtime_schema_sha256
        or discovery.report.source_dag_sha256 != candidate.source_dag_sha256
        or discovery.report.candidate_capture_sha256
        != candidate.capture_contract_sha256
        or discovery.report.verification_capture_sha256
        != verification.capture_contract_sha256
    ):
        raise ValueError(
            "numerical current detector did not authenticate the captured "
            "observation batches"
        )
    applicable_certificates = _application_scoped_certificates(
        discovery.certificates,
        allowed_relation_kinds=allowed_relation_kinds,
    )
    application = apply_numerical_current_relation_certificates(
        dag,
        model,
        applicable_certificates,
        mode=mode,
        execution_mode=execution_mode,
        process_id=process_id,
        runtime_schema_sha256=candidate.runtime_schema_sha256,
        source_dag_sha256=candidate.source_dag_sha256,
        candidate_capture_sha256=candidate.capture_contract_sha256,
        verification_capture_sha256=(verification.capture_contract_sha256),
        _structural_equivalences=discovery.structural_equivalences,
    )
    if (
        application.report.certificates != applicable_certificates
        or application.report.source_semantics_sha256
        != discovery.report.source_semantics_sha256
    ):
        raise ValueError("numerical current application replay drifted from discovery")
    application_capture: GenericDAGCurrentObservationCapture | None = None
    if application.report.applied_relation_count:
        application_capture = capture_generic_dag_current_observations(
            application.dag,
            model,
            verification_points,
            precision_digits=precision_digits,
            parameter_contexts=verification_parameter_contexts,
        )
        application_validation = _validate_applied_current_observations(
            verification,
            application_capture,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    else:
        application_validation = {
            "status": "not-required-no-applied-relations",
            "checked_current_count": 0,
            "checked_component_count": 0,
            "maximum_absolute_residual": None,
            "maximum_relative_residual": None,
            "maximum_tolerance_ratio": None,
        }
    return GenericDAGNumericalCurrentWarmupResult(
        dag=application.dag,
        requested_mode=mode,
        execution_mode=execution_mode,
        candidate_capture=candidate,
        verification_capture=verification,
        application_capture=application_capture,
        application_validation=application_validation,
        discovery=discovery,
        application=application,
        application_scope={
            "abi": ("pyamplicol-generic-dag-numerical-relation-application-scope-v1"),
            "policy": "exact-physical-selector-member-set-v1",
            "reason": application_scope_reason,
            "allowed_relation_kinds": list(allowed_relation_kinds),
            "discovered_relation_count": len(discovery.certificates),
            "application_eligible_relation_count": len(applicable_certificates),
            "scope_suppressed_relation_count": (
                len(discovery.certificates) - len(applicable_certificates)
            ),
        },
    )


def _validated_application_relation_kinds(
    values: Sequence[Literal["equal", "opposite", "zero"]],
) -> tuple[Literal["equal", "opposite", "zero"], ...]:
    resolved = tuple(values)
    if len(set(resolved)) != len(resolved) or any(
        relation_kind not in {"equal", "opposite", "zero"} for relation_kind in resolved
    ):
        raise ValueError("generic-DAG numerical relation application scope is invalid")
    return resolved


def _application_scoped_certificates(
    certificates: Sequence[NumericalCurrentRelationCertificate],
    *,
    allowed_relation_kinds: Sequence[Literal["equal", "opposite", "zero"]],
) -> tuple[NumericalCurrentRelationCertificate, ...]:
    """Contain application without changing authenticated discovery evidence."""

    allowed = frozenset(_validated_application_relation_kinds(allowed_relation_kinds))
    result: list[NumericalCurrentRelationCertificate] = []
    for certificate in certificates:
        relation_kind = getattr(certificate, "relation_kind", None)
        if relation_kind not in {"equal", "opposite", "zero"}:
            raise ValueError(
                "generic-DAG numerical certificate relation kind is invalid"
            )
        if relation_kind in allowed:
            result.append(certificate)
    return tuple(result)


def generic_dag_numerical_current_opt_out_report(
    dag: GenericDAG,
    *,
    execution_mode: Literal["compiled", "eager"],
) -> dict[str, object]:
    """Report the explicit opt-out without hashing unused source semantics."""

    return {
        "schema_version": 1,
        "abi": _WARMUP_ABI,
        "requested_mode": "off",
        "effective_mode": "off",
        "effective_reuse_state": "disabled",
        "state": "disabled-by-user",
        "scope": {
            "execution_mode": execution_mode,
            "color_accuracy": str(dag.process.color_accuracy),
            "representation": "generic-dag",
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
        },
        "application_scope": {
            "abi": ("pyamplicol-generic-dag-numerical-relation-application-scope-v1"),
            "policy": "disabled-by-user",
            "reason": "explicit-opt-out",
            "allowed_relation_kinds": [],
            "discovered_relation_count": 0,
            "application_eligible_relation_count": 0,
            "scope_suppressed_relation_count": 0,
        },
        "discovery": None,
        "application": None,
        "certified_relation_count": 0,
        "applied_relation_count": 0,
        "relation_correctness": numerical_relation_correctness_payload(0),
        "warning": {
            "required": False,
            "emit": "never",
            "code": None,
            "message": None,
        },
    }


def build_numerical_current_probe_points(
    dag: GenericDAG,
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
    """Build deterministic domain-separated physical warm-up points."""

    if (
        type(seed) is not int
        or seed < 0
        or type(candidate_count) is not int
        or candidate_count < 2
        or type(verification_count) is not int
        or verification_count < 2
    ):
        raise ValueError("numerical current probe-point contract is invalid")

    def domain_points(
        domain: str,
        count: int,
    ) -> tuple[ValidationPointRecord, ...]:
        return tuple(
            rotate_validation_point(
                build_validation_point(
                    dag,
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

    candidate = domain_points("candidate-current-probes-v1", candidate_count)
    verification = domain_points(
        "independent-verification-current-probes-v1",
        verification_count,
    )
    if any(not point.available for point in (*candidate, *verification)):
        errors = tuple(
            point.error for point in (*candidate, *verification) if not point.available
        )
        raise ValueError(
            "numerical current probe-point generation failed: "
            + "; ".join(str(error) for error in errors)
        )
    candidate_hashes = {_kinematic_sha256(point) for point in candidate}
    verification_hashes = {_kinematic_sha256(point) for point in verification}
    if (
        len(candidate_hashes) != len(candidate)
        or len(verification_hashes) != len(verification)
        or not candidate_hashes.isdisjoint(verification_hashes)
    ):
        raise ValueError(
            "numerical current probe domains do not contain distinct, "
            "independent physical momentum records"
        )
    return candidate, verification


def capture_generic_dag_current_observations(
    dag: GenericDAG,
    model: Model,
    points: Sequence[ValidationPointRecord],
    *,
    precision_digits: int,
    parameter_contexts: Sequence[Sequence[Decimal]] | None = None,
    _session: _GenericDAGCurrentCaptureSession | None = None,
) -> GenericDAGCurrentObservationCapture:
    """Evaluate and snapshot every current at every supplied physical point.

    The evaluators remain interpreted Symbolica states: this warm-up does not
    launch a JIT, C++, ASM, or native compiler.  Current components are copied
    immediately after their producing stage, before value-slot recycling can
    overwrite them.
    """

    if type(precision_digits) is not int or precision_digits < 80:
        raise ValueError(
            "numerical current capture precision must be at least 80 digits"
        )
    point_records = tuple(points)
    if len(point_records) < 2:
        raise ValueError(
            "numerical current capture requires at least two physical points"
        )
    if any(
        not isinstance(point, ValidationPointRecord) or not point.available
        for point in point_records
    ):
        raise ValueError("numerical current capture received an unavailable point")
    if any(point.process != dag.process.process for point in point_records):
        raise ValueError(
            "numerical current capture point process does not match the DAG"
        )
    process_ids = {point.process_id for point in point_records}
    if len(process_ids) != 1:
        raise ValueError("numerical current capture points mix process identities")
    process_id = next(iter(process_ids))
    session = (
        _build_current_capture_session(
            dag,
            model,
            process_id=process_id,
        )
        if _session is None
        else _session
    )
    if session.dag is not dag or session.process_id != process_id:
        raise ValueError(
            "numerical current capture session drifted from its source DAG"
        )

    schema = session.schema
    blueprint_stages = session.stages
    stage_evaluators = session.stage_evaluators
    model_parameters = session.model_parameters
    resolved_parameter_contexts = (
        tuple(model_parameters for _point in point_records)
        if parameter_contexts is None
        else tuple(tuple(context) for context in parameter_contexts)
    )
    if len(resolved_parameter_contexts) != len(point_records) or any(
        len(context) != len(model_parameters)
        or any(
            not isinstance(value, Decimal) or not value.is_finite() for value in context
        )
        for context in resolved_parameter_contexts
    ):
        raise ValueError(
            "numerical current parameter-probe contexts do not match the runtime schema"
        )
    point_hashes = tuple(_validation_point_sha256(point) for point in point_records)
    parameter_context_hashes = tuple(
        _canonical_sha256(
            {
                "abi": "pyamplicol-numerical-current-parameter-context-v1",
                "point_index": point_index,
                "point_sha256": point_hash,
                "values": [_decimal_string(value) for value in context],
            }
        )
        for point_index, (point_hash, context) in enumerate(
            zip(
                point_hashes,
                resolved_parameter_contexts,
                strict=True,
            )
        )
    )
    kinematic_hashes = tuple(_kinematic_sha256(point) for point in point_records)
    if len(set(kinematic_hashes)) != len(kinematic_hashes):
        raise ValueError("numerical current capture requires distinct physical momenta")
    captured_by_current: dict[int, list[_ComplexDecimal]] = {
        current.id: [] for current in dag.currents
    }
    working_precision = precision_digits + _NUMERICAL_DECIMAL_GUARD_DIGITS
    with localcontext() as context:
        context.prec = working_precision
        context.rounding = ROUND_HALF_EVEN
        for point, parameter_context in zip(
            point_records,
            resolved_parameter_contexts,
            strict=True,
        ):
            point_values = tuple(
                tuple(Decimal.from_float(float(value)) for value in momentum)
                for momentum in point.four_vectors
            )
            point_capture = _capture_one_point(
                dag,
                schema,
                blueprint_stages,
                stage_evaluators,
                point_values,
                parameter_context,
                precision=working_precision,
            )
            for current in dag.currents:
                values = point_capture[current.id]
                if len(values) != current.dimension:
                    raise ValueError(
                        f"numerical current {current.id} capture has "
                        f"{len(values)} components, expected {current.dimension}"
                    )
                captured_by_current[current.id].extend(values)

    observations = {
        current_id: tuple(values) for current_id, values in captured_by_current.items()
    }
    if any(
        not real.is_finite() or not imaginary.is_finite()
        for values in observations.values()
        for real, imaginary in values
    ):
        raise ValueError("numerical current capture produced a non-finite component")
    observation_digest = _current_observation_batch_sha256(
        observations,
        point_sha256s=point_hashes,
    )
    model_parameter_schema_digest = _canonical_sha256(
        schema.get("model_parameters", ())
    )
    capture_contract_digest = _canonical_sha256(
        {
            "abi": _CAPTURE_ABI,
            "precision_digits": precision_digits,
            "point_sha256s": list(point_hashes),
            "kinematic_sha256s": list(kinematic_hashes),
            "parameter_context_sha256s": list(parameter_context_hashes),
            "runtime_schema_sha256": session.runtime_schema.sha256,
            "model_parameter_schema_sha256": (model_parameter_schema_digest),
            "source_dag_sha256": session.source_dag_sha256,
            "evaluator_output_partition_abi": (session.evaluator_output_partition_abi),
            "evaluator_output_partition_sha256": (
                session.evaluator_output_partition_sha256
            ),
            "observation_batch_sha256": observation_digest,
        }
    )
    return GenericDAGCurrentObservationCapture(
        precision_digits=precision_digits,
        points=point_records,
        point_sha256s=point_hashes,
        kinematic_sha256s=kinematic_hashes,
        parameter_contexts=resolved_parameter_contexts,
        parameter_context_sha256s=parameter_context_hashes,
        observations=observations,
        runtime_schema_sha256=session.runtime_schema.sha256,
        model_parameter_schema_sha256=model_parameter_schema_digest,
        source_dag_sha256=session.source_dag_sha256,
        evaluator_output_partition_abi=(session.evaluator_output_partition_abi),
        evaluator_output_partition_sha256=(session.evaluator_output_partition_sha256),
        observation_batch_sha256=observation_digest,
        capture_contract_sha256=capture_contract_digest,
    )


def _current_observation_batch_sha256(
    observations: Mapping[int, tuple[_ComplexDecimal, ...]],
    *,
    point_sha256s: tuple[str, ...],
) -> str:
    return _canonical_sha256(
        {
            "point_sha256s": list(point_sha256s),
            "currents": [
                {
                    "current_id": current_id,
                    "values": [
                        [_decimal_string(real), _decimal_string(imaginary)]
                        for real, imaginary in observations[current_id]
                    ],
                }
                for current_id in sorted(observations)
            ],
        }
    )


def _current_capture_output_partitions(
    stage: GenericCompiledStageBlueprint,
) -> tuple[tuple[tuple[int, int], ...], dict[str, object]]:
    """Return exhaustive current-ID output partitions and their identity."""

    if (
        type(stage.output_length) is not int
        or stage.output_length < 1
        or not stage.output_slots
    ):
        raise ValueError("numerical current capture stage has no partitionable outputs")
    slot_ids = tuple(slot.value_slot_id for slot in stage.output_slots)
    if slot_ids != stage.output_value_slot_ids:
        raise ValueError("numerical current capture output-slot identity drifted")

    ranges: list[tuple[int, int]] = []
    records: list[dict[str, object]] = []
    current_id: int | None = None
    partition_start = 0
    partition_slots: list[dict[str, object]] = []
    seen_current_ids: set[int] = set()
    seen_value_slot_ids: set[int] = set()
    expected_output_start = 0

    def finish_partition(stop: int) -> None:
        if current_id is None or not partition_slots or partition_start >= stop:
            raise ValueError("numerical current capture output partition is empty")
        ranges.append((partition_start, stop))
        records.append(
            {
                "current_id": current_id,
                "output_start": partition_start,
                "output_stop": stop,
                "slots": list(partition_slots),
            }
        )

    for slot in stage.output_slots:
        if (
            type(slot.current_id) is not int
            or slot.current_id < 0
            or type(slot.value_slot_id) is not int
            or slot.value_slot_id < 0
            or slot.value_slot_id in seen_value_slot_ids
            or type(slot.output_start) is not int
            or type(slot.output_stop) is not int
            or slot.output_start != expected_output_start
            or slot.output_stop <= slot.output_start
            or slot.output_stop > stage.output_length
            or slot.output_stop - slot.output_start
            != slot.component_stop - slot.component_start
        ):
            raise ValueError(
                "numerical current capture output slots overlap, gap, or "
                "have an invalid identity"
            )
        if current_id is None:
            current_id = slot.current_id
            seen_current_ids.add(current_id)
        elif slot.current_id != current_id:
            finish_partition(slot.output_start)
            if slot.current_id in seen_current_ids or slot.current_id <= current_id:
                raise ValueError(
                    "numerical current capture output-slot current order drifted"
                )
            current_id = slot.current_id
            seen_current_ids.add(current_id)
            partition_start = slot.output_start
            partition_slots = []
        seen_value_slot_ids.add(slot.value_slot_id)
        partition_slots.append(
            {
                "value_slot_id": slot.value_slot_id,
                "variant": slot.variant,
                "component_start": slot.component_start,
                "component_stop": slot.component_stop,
                "output_start": slot.output_start,
                "output_stop": slot.output_stop,
            }
        )
        expected_output_start = slot.output_stop

    if expected_output_start != stage.output_length:
        raise ValueError(
            "numerical current capture output partitions are not exhaustive"
        )
    finish_partition(stage.output_length)
    return tuple(ranges), {
        "stage_index": stage.stage_index,
        "stage_kind": stage.stage_kind,
        "subset_size": stage.subset_size,
        "output_length": stage.output_length,
        "partitions": records,
    }


def _current_capture_partition_plan(
    stages: Sequence[GenericCompiledStageBlueprint],
) -> tuple[tuple[tuple[tuple[int, int], ...], ...], str]:
    """Authenticate one stable current-ID partition plan for all stages."""

    stage_partitions: list[tuple[tuple[int, int], ...]] = []
    stage_contracts: list[dict[str, object]] = []
    previous_stage_index: int | None = None
    for stage in stages:
        if type(stage.stage_index) is not int or (
            previous_stage_index is not None
            and stage.stage_index <= previous_stage_index
        ):
            raise ValueError("numerical current capture stage order drifted")
        partitions, contract = _current_capture_output_partitions(stage)
        stage_partitions.append(partitions)
        stage_contracts.append(contract)
        previous_stage_index = stage.stage_index
    if not stage_partitions:
        raise ValueError("numerical current capture has no evaluator stages")
    payload = {
        "abi": _CAPTURE_OUTPUT_PARTITION_ABI,
        "strategy": "one-contiguous-output-partition-per-current-id",
        "stages": stage_contracts,
    }
    return tuple(stage_partitions), _canonical_sha256(payload)


def _build_current_capture_session(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str,
) -> _GenericDAGCurrentCaptureSession:
    runtime_schema = build_runtime_expression_schema(
        dag,
        model,
        process_id=process_id,
    )
    schema = runtime_schema.to_mapping()
    blueprint = build_generic_stage_compiler_blueprint(
        StageCompilationInput(dag, model, runtime_schema)
    )
    if not blueprint.expression_ready:
        raise ValueError(
            "numerical current capture cannot lower stage expressions: "
            + "; ".join(blueprint.blockers)
        )
    settings = SymbolicaEvaluatorSettings(
        iterations=1,
        cpe_iterations=0,
        n_cores=1,
        direct_translation=False,
        jit_direct_translation=False,
        jit_optimization_level=0,
        compiled_output_chunk_size=None,
        output_chunk_strategy="uniform",
        compiled_chunk_compile_workers=1,
    )
    stage_output_partitions, output_partition_digest = _current_capture_partition_plan(
        blueprint.stages
    )
    runtime_output_partition_digest = (
        generic_dag_numerical_capture_output_partition_sha256(schema)
    )
    if output_partition_digest != runtime_output_partition_digest:
        raise ValueError(
            "numerical current capture evaluator partitions drifted from "
            "runtime output-slot geometry"
        )
    stage_evaluators = tuple(
        _build_interpreted_stage_evaluator(
            stage,
            settings=settings,
            output_partitions=output_partitions,
        )
        for stage, output_partitions in zip(
            blueprint.stages,
            stage_output_partitions,
            strict=True,
        )
    )
    model_parameters = _runtime_model_parameter_values(schema)
    return _GenericDAGCurrentCaptureSession(
        dag=dag,
        process_id=process_id,
        runtime_schema=runtime_schema,
        schema=schema,
        stages=blueprint.stages,
        stage_evaluators=stage_evaluators,
        model_parameters=model_parameters,
        source_dag_sha256=_canonical_sha256(dag.to_json_dict()),
        evaluator_output_partition_abi=_CAPTURE_OUTPUT_PARTITION_ABI,
        evaluator_output_partition_sha256=output_partition_digest,
    )


def validate_independent_current_observation_captures(
    candidate: GenericDAGCurrentObservationCapture,
    verification: GenericDAGCurrentObservationCapture,
) -> None:
    """Fail closed unless two capture batches form one independent contract."""

    if (
        candidate.precision_digits != verification.precision_digits
        or candidate.runtime_schema_sha256 != verification.runtime_schema_sha256
        or candidate.model_parameter_schema_sha256
        != verification.model_parameter_schema_sha256
        or candidate.source_dag_sha256 != verification.source_dag_sha256
        or candidate.evaluator_output_partition_abi
        != verification.evaluator_output_partition_abi
        or candidate.evaluator_output_partition_sha256
        != verification.evaluator_output_partition_sha256
        or set(candidate.observations) != set(verification.observations)
        or candidate.point_count < 2
        or verification.point_count < 2
        or len(set(candidate.point_sha256s)) != candidate.point_count
        or len(set(verification.point_sha256s)) != verification.point_count
        or not set(candidate.point_sha256s).isdisjoint(verification.point_sha256s)
        or len(set(candidate.kinematic_sha256s)) != candidate.point_count
        or len(set(verification.kinematic_sha256s)) != verification.point_count
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
            "numerical current candidate and verification captures do not "
            "form independent replay domains"
        )
    for current_id in candidate.observations:
        candidate_values = candidate.observations[current_id]
        verification_values = verification.observations[current_id]
        if (
            len(candidate_values) % candidate.point_count != 0
            or len(verification_values) % verification.point_count != 0
            or len(candidate_values) // candidate.point_count
            != len(verification_values) // verification.point_count
        ):
            raise ValueError(
                f"numerical current {current_id} capture widths drifted "
                "between candidate and verification domains"
            )


def _validate_applied_current_observations(
    reference: GenericDAGCurrentObservationCapture,
    applied: GenericDAGCurrentObservationCapture,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, object]:
    """Authenticate that reuse application preserves every observed current."""

    if (
        reference.precision_digits != applied.precision_digits
        or reference.point_sha256s != applied.point_sha256s
        or reference.kinematic_sha256s != applied.kinematic_sha256s
        or reference.parameter_contexts != applied.parameter_contexts
        or reference.parameter_context_sha256s != applied.parameter_context_sha256s
        or reference.model_parameter_schema_sha256
        != applied.model_parameter_schema_sha256
        or reference.evaluator_output_partition_abi
        != applied.evaluator_output_partition_abi
        or reference.evaluator_output_partition_sha256
        != applied.evaluator_output_partition_sha256
        or set(reference.observations) != set(applied.observations)
    ):
        raise ValueError("numerical current application validation provenance drifted")
    with localcontext() as context:
        context.prec = reference.precision_digits + _NUMERICAL_DECIMAL_GUARD_DIGITS
        context.rounding = ROUND_HALF_EVEN
        relative = Decimal.from_float(float(relative_tolerance))
        absolute = Decimal.from_float(float(absolute_tolerance))
        if (
            not relative.is_finite()
            or not absolute.is_finite()
            or relative < 0
            or absolute < 0
            or (relative == 0 and absolute == 0)
        ):
            raise ValueError(
                "numerical current application validation tolerances are invalid"
            )
        maximum_absolute = Decimal(0)
        maximum_relative = Decimal(0)
        maximum_tolerance_ratio = Decimal(0)
        checked_components = 0
        for current_id in sorted(reference.observations):
            before = reference.observations[current_id]
            after = applied.observations[current_id]
            if len(before) != len(after):
                raise ValueError(
                    f"numerical current {current_id} application width drifted"
                )
            for observation_index, (baseline, optimized) in enumerate(
                zip(before, after, strict=True)
            ):
                difference = max(
                    abs(baseline[0] - optimized[0]),
                    abs(baseline[1] - optimized[1]),
                )
                scale = max(
                    abs(baseline[0]),
                    abs(baseline[1]),
                    abs(optimized[0]),
                    abs(optimized[1]),
                )
                allowed = absolute + relative * scale
                relative_residual = (
                    Decimal(0) if difference == 0 else difference / max(scale, absolute)
                )
                tolerance_ratio = (
                    Decimal(0) if difference == 0 else difference / allowed
                )
                maximum_absolute = max(maximum_absolute, difference)
                maximum_relative = max(
                    maximum_relative,
                    relative_residual,
                )
                maximum_tolerance_ratio = max(
                    maximum_tolerance_ratio,
                    tolerance_ratio,
                )
                checked_components += 1
                if not difference <= allowed:
                    raise ValueError(
                        "numerical current application changed current "
                        f"{current_id} observation {observation_index} beyond "
                        "its authenticated tolerance: "
                        f"difference={_decimal_string(difference)}, "
                        f"allowed={_decimal_string(allowed)}, "
                        "tolerance_ratio="
                        f"{_decimal_string(tolerance_ratio)}"
                    )
        return {
            "status": "verified",
            "checked_current_count": len(reference.observations),
            "checked_component_count": checked_components,
            "maximum_absolute_residual": _decimal_string(maximum_absolute),
            "maximum_relative_residual": _decimal_string(maximum_relative),
            "maximum_tolerance_ratio": _decimal_string(maximum_tolerance_ratio),
            "reference_observation_batch_sha256": (reference.observation_batch_sha256),
            "applied_observation_batch_sha256": (applied.observation_batch_sha256),
            "evaluator_output_partition_abi": (
                reference.evaluator_output_partition_abi
            ),
            "evaluator_output_partition_sha256": (
                reference.evaluator_output_partition_sha256
            ),
        }


def _build_interpreted_stage_evaluator(
    stage: GenericCompiledStageBlueprint,
    *,
    settings: SymbolicaEvaluatorSettings,
    output_partitions: Sequence[tuple[int, int]],
) -> Any:
    evaluator = _compile_symbolica_outputs(
        stage.output_expressions,
        list(stage.parameter_symbols),
        merge_evaluators_strategy=False,
        verbose_evaluator_build=False,
        functions={
            (function, arguments): body
            for function, arguments, body in stage.symbolica_functions
        },
        real_params=stage.real_valued_inputs,
        symbolica_settings=settings,
        jit_compile=False,
        label=f"numerical_current_warmup_stage_{stage.stage_index}",
        output_partitions=output_partitions,
    )
    partitions = tuple(output_partitions)
    if len(partitions) == 1:
        evaluators = (evaluator,)
        chunk_input_indices = (tuple(range(stage.parameter_count)),)
    else:
        evaluators = getattr(evaluator, "_evaluators", None)
        chunk_input_indices = getattr(
            evaluator,
            "_chunk_input_indices",
            None,
        )
        if not isinstance(evaluators, tuple) or not isinstance(
            chunk_input_indices,
            tuple,
        ):
            raise ValueError(
                "numerical current warm-up partition compilation did not "
                "retain its evaluator leaves"
            )
    return _PartitionedCurrentCaptureStageEvaluator(
        evaluators=evaluators,
        chunk_input_indices=chunk_input_indices,
        output_partitions=partitions,
        input_len=stage.parameter_count,
        output_len=stage.output_length,
    )


def _capture_one_point(
    dag: GenericDAG,
    schema: Mapping[str, object],
    stages: tuple[GenericCompiledStageBlueprint, ...],
    evaluators: tuple[Any, ...],
    point: tuple[tuple[Decimal, ...], ...],
    model_parameters: tuple[Decimal, ...],
    *,
    precision: int,
) -> dict[int, tuple[_ComplexDecimal, ...]]:
    layout = _mapping(schema.get("parameter_layout"), "parameter layout")
    value_count = _integer(layout.get("value_component_count"))
    momentum_count = _integer(layout.get("momentum_parameter_count"))
    flattened_count = _integer(layout.get("parameter_count_if_flattened"))
    model_start = value_count + momentum_count
    state = [
        (Decimal(0), Decimal(0))
        for _index in range(max(flattened_count, model_start + len(model_parameters)))
    ]
    _fill_sources(state, point, schema, model_parameters)
    _fill_momenta(state, point, schema)
    for index, value in enumerate(model_parameters):
        state[model_start + index] = (value, Decimal(0))

    captured: dict[int, tuple[_ComplexDecimal, ...]] = {}
    source_fill = _mapping(schema.get("source_fill"), "source fill")
    sources = _sequence(source_fill.get("sources"), "source rows")
    for raw_source in sources:
        source = _mapping(raw_source, "source row")
        slot = _mapping(source.get("value_slot"), "source value slot")
        current_id = _integer(source.get("current_id"))
        start = _integer(slot.get("component_start"))
        stop = _integer(slot.get("component_stop"))
        if current_id in captured or not 0 <= start <= stop <= len(state):
            raise ValueError("numerical current source capture has an invalid slot")
        captured[current_id] = tuple(state[start:stop])

    for stage, evaluator in zip(stages, evaluators, strict=True):
        inputs: list[_ComplexDecimal | None] = [None] * stage.parameter_count
        for component in stage.input_components:
            if (
                not 0 <= component.parameter_index < len(inputs)
                or not 0 <= component.global_component < len(state)
                or inputs[component.parameter_index] is not None
            ):
                raise ValueError("numerical current stage input mapping is invalid")
            inputs[component.parameter_index] = state[component.global_component]
        if any(value is None for value in inputs):
            raise ValueError("numerical current stage input mapping is incomplete")
        outputs = _evaluate_interpreted_stage(
            evaluator,
            tuple(value for value in inputs if value is not None),
            precision=precision,
        )
        if len(outputs) != stage.output_length:
            raise ValueError(
                "numerical current stage evaluator returned the wrong width"
            )
        for slot in stage.output_slots:
            values = outputs[slot.output_start : slot.output_stop]
            if (
                slot.current_id in captured
                or len(values) != slot.component_stop - slot.component_start
                or not 0 <= slot.component_start <= slot.component_stop <= len(state)
            ):
                raise ValueError("numerical current stage output mapping is invalid")
            state[slot.component_start : slot.component_stop] = values
            captured[slot.current_id] = values
    if set(captured) != {current.id for current in dag.currents}:
        raise ValueError(
            "numerical current capture did not visit every current exactly once"
        )
    return captured


def _evaluate_interpreted_stage(
    evaluator: Any,
    values: tuple[_ComplexDecimal, ...],
    *,
    precision: int,
) -> tuple[_ComplexDecimal, ...]:
    evaluate = getattr(evaluator, "evaluate_complex_with_prec", None)
    if not callable(evaluate):
        raise ValueError(
            "numerical current warm-up evaluator lacks high-precision replay"
        )
    try:
        raw = evaluate(values, precision)
    except Exception as error:
        raise ValueError(
            f"numerical current high-precision stage replay failed: {error}"
        ) from error
    return tuple(
        (
            _decimal(real, "numerical current real component"),
            _decimal(imaginary, "numerical current imaginary component"),
        )
        for value in raw
        for real, imaginary in (value.to_decimal_tuple(),)
    )


def _runtime_model_parameter_values(
    schema: Mapping[str, object],
) -> tuple[Decimal, ...]:
    records = _sequence(schema.get("model_parameters", ()), "model parameters")
    ordered: list[Decimal | None] = [None] * len(records)
    for raw_record in records:
        record = _mapping(raw_record, "model parameter")
        index = _integer(record.get("parameter_index"))
        if (
            not 0 <= index < len(ordered)
            or ordered[index] is not None
            or "default" not in record
        ):
            raise ValueError("numerical current model-parameter layout is invalid")
        value = float(record["default"])
        if not isfinite(value):
            raise ValueError("numerical current model parameter is non-finite")
        ordered[index] = Decimal.from_float(value)
    if any(value is None for value in ordered):
        raise ValueError("numerical current model-parameter layout is incomplete")
    return tuple(value for value in ordered if value is not None)


def _build_parameter_probe_contexts(
    defaults: tuple[Decimal, ...],
    *,
    precision_digits: int,
    seed: int,
    domain: str,
    count: int,
    include_defaults: bool,
) -> tuple[tuple[Decimal, ...], ...]:
    """Build independent deterministic probes over every runtime slot.

    Runtime APIs can override external and derived flattened parameter slots.
    Sampling every slot independently is therefore a conservative check: a
    relation that exists only at the model defaults is not eligible for
    permanent evaluator reuse.
    """

    if (
        type(precision_digits) is not int
        or precision_digits < 80
        or type(seed) is not int
        or seed < 0
        or not isinstance(domain, str)
        or not domain
        or type(count) is not int
        or count < 2
    ):
        raise ValueError("numerical current parameter-probe contract is invalid")
    working_precision = precision_digits + 16
    contexts: list[tuple[Decimal, ...]] = []
    with localcontext() as context:
        context.prec = working_precision
        context.rounding = ROUND_HALF_EVEN
        for probe_index in range(count):
            if include_defaults and probe_index == 0:
                contexts.append(defaults)
                continue
            values: list[Decimal] = []
            for parameter_index, default in enumerate(defaults):
                digest = hashlib.sha256(
                    (f"{seed}:{domain}:{probe_index}:{parameter_index}").encode("ascii")
                ).digest()
                signed = int.from_bytes(digest[:8], "big") - (1 << 63)
                if signed == 0:
                    signed = 1
                # At most a 1/16 perturbation of max(|default|, 1).
                relative_offset = Decimal(signed) / Decimal(1 << 67)
                scale = max(abs(default), Decimal(1))
                values.append(default + scale * relative_offset)
            contexts.append(tuple(values))
    return tuple(contexts)


def _validation_point_sha256(point: ValidationPointRecord) -> str:
    return _canonical_sha256(point.to_mapping())


def _kinematic_sha256(point: ValidationPointRecord) -> str:
    return _canonical_sha256(
        [
            {
                "pdg": pdg,
                "momentum": [format(float(value), ".17g") for value in momentum],
            }
            for pdg, momentum in point.particles
        ]
    )


def _domain_seed(seed: int, *, domain: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{domain}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _decimal_string(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("canonical Decimal serialization requires finiteness")
    sign, digits, exponent = value.as_tuple()
    if not any(digits):
        return "0"
    if not isinstance(exponent, int):
        raise ValueError("canonical Decimal serialization requires an exponent")
    coefficient = "".join(str(digit) for digit in digits)
    return f"{'-' if sign else ''}{coefficient}e{exponent}"


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"numerical current {label} is not an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"numerical current {label} is not a sequence")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("numerical current layout integer is invalid")
    return value


__all__ = [
    "GenericDAGCurrentObservationCapture",
    "GenericDAGNumericalCurrentWarmupResult",
    "build_numerical_current_probe_points",
    "capture_generic_dag_current_observations",
    "generic_dag_numerical_current_opt_out_report",
    "run_generic_dag_numerical_current_warmup",
    "validate_independent_current_observation_captures",
]
