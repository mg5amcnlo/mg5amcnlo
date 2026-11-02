# SPDX-License-Identifier: 0BSD
"""ID-selected direct Born correlations, isolated from ordinary runtime calls."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.api.protocols import Momenta
from pyamplicol.artifacts import load_manifest
from pyamplicol.artifacts.security import confined_path
from pyamplicol.color.connections import COLOR_CONNECTION_CONVENTION
from pyamplicol.correlators import CorrelatedRequest, CorrelatedValue, CorrelatorConfig
from pyamplicol.runtime._double_double import correlated_precision
from pyamplicol.runtime._native_selection import native_process_selection
from pyamplicol.runtime.correlated_exact import (
    CoherentAmplitudeBatch,
    CoherentAmplitudeGroup,
    CorrelatedDoubleDoubleExecutor,
    CorrelatedExactExecutor,
    _is_spin_vector,
    _prepare_spin_vectors,
)
from pyamplicol.runtime.symbolica_exact import _complex_mul, _working_precision


@dataclass(frozen=True)
class _MatrixEntry:
    left: int
    right: int
    real: Fraction
    imag: Fraction


@dataclass(frozen=True)
class _Matrix:
    id: str
    sectors: tuple[int, ...]
    entries: tuple[_MatrixEntry, ...]


@dataclass(frozen=True)
class CorrelatorCatalogue:
    """Validated selected-process metadata, without an amplitude executor."""

    artifact: Path
    process_id: str
    native_runtime: Any
    declarations: CorrelatorConfig
    matrices: tuple[_Matrix, ...]
    coherent_groups: Sequence[Mapping[str, object]]
    # Exact factor of a process-group member evaluated by its representative.
    factor: Fraction = Fraction(1)


def _exact_integer(value: object) -> int:
    """Read the writer's integer strings without accepting lossy coercions."""
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value):
        return int(value)
    raise ValueError("invalid exact matrix integer")


def _matrix(record: object) -> _Matrix:
    try:
        if not isinstance(record, Mapping):
            raise ValueError("matrix is not an object")
        if (
            record["convention"] != COLOR_CONNECTION_CONVENTION
            or record["color_accuracy"] not in {"lc", "nlc", "full"}
            or record["storage"] != "sparse-directed"
            or record["includes_color_factor"] is not True
        ):
            raise ValueError("unsupported matrix convention")
        identifier = record["id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("missing correlation ID")
        sectors = tuple(record["sector_ids"])
        if (
            not sectors
            or any(type(item) is not int or item < 0 for item in sectors)
            or len(set(sectors)) != len(sectors)
        ):
            raise ValueError("invalid colour sector IDs")
        entries = []
        pairs: set[tuple[int, int]] = set()
        for item in record["entries"]:
            left, right = item["left_sector_id"], item["right_sector_id"]
            if (
                type(left) is not int
                or type(right) is not int
                or left not in sectors
                or right not in sectors
                or (left, right) in pairs
            ):
                raise ValueError("invalid or repeated matrix entry")
            pairs.add((left, right))
            real, imag = item["weight"]["real"], item["weight"]["imag"]
            if any(
                isinstance(parts, (str, bytes))
                or not isinstance(parts, Sequence)
                or len(parts) != 2
                for parts in (real, imag)
            ):
                raise ValueError("invalid exact matrix coefficient")
            entries.append(
                _MatrixEntry(
                    left,
                    right,
                    Fraction(_exact_integer(real[0]), _exact_integer(real[1])),
                    Fraction(_exact_integer(imag[0]), _exact_integer(imag[1])),
                )
            )
        return _Matrix(identifier, sectors, tuple(entries))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ArtifactError(f"invalid correlated colour matrix: {exc}") from exc


@dataclass(frozen=True)
class _ContractionPlan:
    group_count: int
    terms: tuple[tuple[int, int, tuple[Decimal, Decimal]], ...]

    def evaluate(
        self, batch: CoherentAmplitudeBatch, *, precision: int
    ) -> tuple[CorrelatedValue, ...]:
        results = []
        with localcontext() as context:
            context.prec = _working_precision(precision)
            context.rounding = ROUND_HALF_EVEN
            for point in batch.values:
                if len(point) != self.group_count:
                    raise ArtifactError(
                        "correlated amplitude point has an invalid group count"
                    )
                scalar = type(batch.normalization_factor)
                real, imag = scalar(0), scalar(0)
                for left, right, weight in self.terms:
                    a, b = point[left], point[right]
                    # Keep the single-ID multiplication/accumulation order.
                    term = _complex_mul(_complex_mul((a[0], -a[1]), weight), b)
                    real += term[0]
                    imag += term[1]
                results.append(
                    (
                        real * batch.normalization_factor,
                        imag * batch.normalization_factor,
                    )
                )
        with localcontext() as context:
            context.prec = precision
            context.rounding = ROUND_HALF_EVEN
            return tuple(
                CorrelatedValue(+Decimal(real), +Decimal(imag))
                for real, imag in results
            )


def _prepare_contraction(
    amplitude_groups: tuple[CoherentAmplitudeGroup, ...],
    matrix: _Matrix,
    *,
    precision: int,
    scalar: type[Decimal] = Decimal,
) -> _ContractionPlan:
    """Prepare exact weights and indices once, without numerical amplitude state."""
    groups: dict[str, dict[int, int]] = {}
    for index, group in enumerate(amplitude_groups):
        sectors = groups.setdefault(group.helicity_id, {})
        if group.color_sector_id in sectors:
            raise ArtifactError("repeated coherent helicity/colour group")
        sectors[group.color_sector_id] = index
    expected = set(matrix.sectors)
    if any(set(indices) != expected for indices in groups.values()):
        raise ArtifactError("correlated amplitudes do not cover the colour matrix")
    with localcontext() as context:
        context.prec = _working_precision(precision)
        context.rounding = ROUND_HALF_EVEN
        entries = tuple(
            (
                entry.left,
                entry.right,
                (
                    scalar(entry.real.numerator) / scalar(entry.real.denominator),
                    scalar(entry.imag.numerator) / scalar(entry.imag.denominator),
                ),
            )
            for entry in matrix.entries
        )
    return _ContractionPlan(
        len(amplitude_groups),
        tuple(
            (indices[left], indices[right], weight)
            for indices in groups.values()
            for left, right, weight in entries
        ),
    )


def _contract(
    batch: CoherentAmplitudeBatch, matrix: _Matrix, *, precision: int
) -> tuple[CorrelatedValue, ...]:
    """Sum spectator helicities incoherently, but retain all colour phases."""
    return _prepare_contraction(
        batch.groups,
        matrix,
        precision=precision,
        scalar=type(batch.normalization_factor),
    ).evaluate(batch, precision=precision)


def load_correlator_catalogue(backend: Any) -> CorrelatorCatalogue:
    """Read and validate the correlation sidecar once for a runtime selection."""
    artifact = getattr(backend, "_artifact_path", None)
    native = getattr(backend, "_runtime", None)
    if not isinstance(artifact, Path) or native is None:
        raise CompatibilityError("this runtime does not support Born correlations")
    manifest = load_manifest(artifact)
    extension = manifest.extensions.get("correlators")
    if not isinstance(extension, Mapping):
        raise CompatibilityError(
            "correlations must be declared at generation with CorrelatorConfig"
        )
    try:
        if (
            type(extension["schema_version"]) is not int
            or extension["schema_version"] != 1
        ):
            raise ValueError("unsupported correlator schema version")
        path = extension["path"]
        if not isinstance(path, str):
            raise ValueError("missing correlator catalogue path")
        payload = json.loads(confined_path(artifact, path).read_text())
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["complete_source_basis"] is not True
        ):
            raise ValueError("correlations require a complete source basis")
        selection = native_process_selection(native, manifest.processes)
        identity = selection.external_permutation == tuple(
            range(len(selection.external_permutation))
        )
        relation = selection.process_group
        if relation is not None and (relation.conjugated or not identity):
            raise CompatibilityError(
                "correlated evaluation of process-group member "
                f"{relation.member_process_id!r} through its representative "
                f"{relation.representative_process_key!r} is available only for "
                "an identity leg order without charge conjugation; generate the "
                "process explicitly"
            )
        if not identity:
            raise CompatibilityError(
                "correlated catalogues and evaluation do not support "
                "non-identity process aliases"
            )
        process_id = selection.representative_process_id
        process = payload["processes"][process_id]
        declarations = CorrelatorConfig.from_json_dict(process["declarations"])
        requests = declarations.color_requests
        if (
            any(type(leg) is not int for leg in process["spin_legs"])
            or tuple(process["spin_legs"]) != declarations.spin_legs
        ):
            raise ValueError("inconsistent declared spin legs")
        matrices = tuple(_matrix(item) for item in process["matrices"])
        identifiers = {matrix.id for matrix in matrices}
        if len(identifiers) != len(matrices) or identifiers != {
            request.id for request in requests
        }:
            raise ValueError("inconsistent colour correlation IDs")
        coherent_groups = cast(
            Sequence[Mapping[str, object]], process["coherent_groups"]
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ArtifactError(f"invalid correlator catalogue: {exc}") from exc
    return CorrelatorCatalogue(
        artifact,
        process_id,
        native,
        declarations,
        matrices,
        coherent_groups,
        Fraction(1) if relation is None else relation.factor,
    )


class CorrelatorEvaluator:
    """Small stateful facade; source vectors belong only to this runtime."""

    def __init__(
        self, backend: Any, *, catalogue: CorrelatorCatalogue | None = None
    ) -> None:
        if catalogue is None:
            catalogue = load_correlator_catalogue(backend)
        self._matrices = {matrix.id: matrix for matrix in catalogue.matrices}
        self._catalogue = catalogue
        self._double_double_executor: CorrelatedDoubleDoubleExecutor | None = None
        self._declarations = catalogue.declarations
        self._vectors: dict[int, object] = {}
        self._executor = CorrelatedExactExecutor(
            catalogue.artifact,
            catalogue.process_id,
            catalogue.native_runtime,
            spin_correlated_legs=catalogue.declarations.spin_legs,
            coherent_groups=catalogue.coherent_groups,
        )

    def _select_executor(self, arithmetic: str) -> CorrelatedExactExecutor:
        if arithmetic != "double-double":
            return self._executor
        if self._double_double_executor is None:
            catalogue = self._catalogue
            self._double_double_executor = CorrelatedDoubleDoubleExecutor(
                catalogue.artifact,
                catalogue.process_id,
                catalogue.native_runtime,
                spin_correlated_legs=catalogue.declarations.spin_legs,
                coherent_groups=catalogue.coherent_groups,
            )
        return self._double_double_executor

    def set_spin_correlation_vectors(
        self, vectors: Mapping[int, object] | None
    ) -> None:
        self._vectors = self._copy_spin_vectors(vectors)

    def _copy_spin_vectors(
        self, vectors: Mapping[int, object] | None
    ) -> dict[int, object]:
        if vectors is None:
            vectors = {}
        if not isinstance(vectors, Mapping):
            raise EvaluationError("spin vectors must map public leg labels to vectors")
        legs = tuple(vectors)
        if any(type(leg) is not int for leg in legs):
            raise EvaluationError("spin-vector legs must be positive integer labels")
        if legs and tuple(sorted(legs)) not in self._declarations.spin_correlations:
            raise EvaluationError("this joint spin-correlation class was not declared")
        frozen: dict[int, object] = {}

        def freeze(value: object) -> object:
            if isinstance(value, Sequence) and not isinstance(value, str | bytes):
                return tuple(freeze(item) for item in value)
            return value

        for leg, value in vectors.items():
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise EvaluationError("spin vectors must be four-vectors or a batch")
            vector = tuple(freeze(item) for item in value)
            if not vector:
                raise EvaluationError(
                    "a spin-vector batch must contain at least one point"
                )
            broadcast = _is_spin_vector(vector)
            # Validate before replacing the old state; a failed setter is atomic.
            _prepare_spin_vectors(
                {leg: vector},
                allowed_legs=self._declarations.spin_legs,
                point_count=1 if broadcast else len(vector),
            )
            frozen[leg] = vector
        return frozen

    def clear(self) -> None:
        """Release warmed evaluators, preserving current numerical spin inputs."""
        self._executor.clear()
        if self._double_double_executor is not None:
            self._double_double_executor.clear()

    def evaluate(
        self,
        momenta: Momenta,
        *,
        color_correlation: str = "born",
        helicities: Sequence[str] | None = None,
        precision: int = 16,
        arithmetic: str = "arbitrary",
    ) -> tuple[CorrelatedValue, ...]:
        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision < 1
        ):
            raise EvaluationError(
                "precision must be a positive number of decimal digits"
            )
        if (
            not isinstance(color_correlation, str)
            or color_correlation not in self._matrices
        ):
            raise EvaluationError(
                f"unknown colour correlation ID {color_correlation!r}"
            )
        amplitude_precision = correlated_precision(precision, arithmetic)
        batch = self._select_executor(arithmetic).coherent_amplitudes(
            momenta,
            spin_vectors=self._vectors,
            helicities=helicities,
            precision=amplitude_precision,
        )
        return self._scaled(
            _contract(batch, self._matrices[color_correlation], precision=precision),
            precision,
        )

    def _scaled(
        self, values: tuple[CorrelatedValue, ...], precision: int
    ) -> tuple[CorrelatedValue, ...]:
        """Apply a process-group member's exact factor with one rounding."""

        factor = self._catalogue.factor
        if factor == 1:
            return values
        with localcontext() as context:
            context.rounding = ROUND_HALF_EVEN
            scaled = []
            for value in values:
                parts = []
                for part in (value.real, value.imag):
                    context.prec = max(precision, len(part.as_tuple().digits)) + len(
                        str(factor.numerator)
                    )
                    product = part * factor.numerator
                    context.prec = precision
                    parts.append(product / factor.denominator)
                scaled.append(CorrelatedValue(parts[0], parts[1]))
        return tuple(scaled)

    def evaluate_many(
        self,
        momenta: Momenta,
        requests: Mapping[str, CorrelatedRequest],
        *,
        helicities: Sequence[str] | None = None,
        precision: int = 16,
        arithmetic: str = "arbitrary",
    ) -> dict[str, tuple[CorrelatedValue, ...]]:
        """Share amplitudes and exact unchanged stages within one labelled batch."""
        if type(precision) is not int or precision < 1:
            raise EvaluationError(
                "precision must be a positive number of decimal digits"
            )
        amplitude_precision = correlated_precision(precision, arithmetic)
        if not isinstance(requests, Mapping):
            raise EvaluationError(
                "correlated requests must map labels to CorrelatedRequest"
            )
        snapshot = dict(self._vectors)
        assignments: list[dict[int, object]] = []
        assignment_ids: dict[tuple[object, ...], int] = {}
        routes: list[tuple[str, str, int]] = []
        for label, request in requests.items():
            if not isinstance(label, str) or not label:
                raise EvaluationError(
                    "correlated request labels must be nonempty strings"
                )
            if not isinstance(request, CorrelatedRequest):
                raise EvaluationError(
                    "correlated requests must contain CorrelatedRequest records"
                )
            identifier = request.color_correlation
            if identifier not in self._matrices:
                raise EvaluationError(f"unknown colour correlation ID {identifier!r}")
            vectors = self._copy_spin_vectors(
                snapshot if request.spin_vectors is None else request.spin_vectors
            )
            prepared = _prepare_spin_vectors(
                vectors,
                allowed_legs=self._declarations.spin_legs,
                point_count=len(momenta),
            )
            # Normalize broadcast/per-point shapes, without a float conversion.
            # Decimal tuples conservatively preserve signed zero and precision.
            key: tuple[object, ...] = tuple(
                tuple(
                    (
                        leg,
                        tuple(
                            (real.as_tuple(), imag.as_tuple()) for real, imag in wave
                        ),
                    )
                    for leg, wave in sorted(point.items())
                )
                for point in prepared
            )
            if key not in assignment_ids:
                assignment_ids[key] = len(assignments)
                assignments.append(vectors)
            routes.append((label, identifier, assignment_ids[key]))
        if not routes:
            return {}
        by_assignment: dict[int, dict[str, list[str]]] = {}
        for label, identifier, assignment in routes:
            by_assignment.setdefault(assignment, {}).setdefault(identifier, []).append(
                label
            )
        results: dict[str, list[CorrelatedValue]] = {
            label: [] for label, _, _ in routes
        }
        plans: dict[tuple[int, str], _ContractionPlan] = {}
        for (
            point_index,
            assignment,
            batch,
        ) in self._select_executor(arithmetic).iter_coherent_amplitudes_many(
            momenta,
            spin_vector_sets=assignments,
            helicities=helicities,
            precision=amplitude_precision,
        ):
            for identifier, labels in by_assignment[assignment].items():
                plan_key = (assignment, identifier)
                if plan_key not in plans:
                    plans[plan_key] = _prepare_contraction(
                        batch.groups,
                        self._matrices[identifier],
                        precision=precision,
                        scalar=type(batch.normalization_factor),
                    )
                value = plans[plan_key].evaluate(batch, precision=precision)[0]
                for label in labels:
                    if len(results[label]) != point_index:
                        raise ArtifactError(
                            "correlated batch returned inconsistent point ordering"
                        )
                    results[label].append(value)
        if any(len(values) != len(momenta) for values in results.values()):
            raise ArtifactError("correlated batch did not cover all requested points")
        return {
            label: self._scaled(tuple(values), precision)
            for label, values in results.items()
        }
