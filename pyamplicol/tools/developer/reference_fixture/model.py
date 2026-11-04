# SPDX-License-Identifier: 0BSD
"""Immutable typed records for reference-physics fixtures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import TypeAlias


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


class ReferenceFixtureError(ValueError):
    """A reference fixture violates its wire or semantic contract."""


@dataclass(frozen=True, slots=True)
class ComplexDecimal:
    real: Decimal
    imag: Decimal


@dataclass(frozen=True, slots=True)
class FixtureProvenance:
    source_repository: str
    source_revision: str
    source_tree_sha256: str
    captured_at: str
    capture_command: tuple[str, ...]
    working_tree_clean: bool
    memory_watchdog_gb: int


@dataclass(frozen=True, slots=True)
class DependencyProvenance:
    id: str
    name: str
    version: str
    revision: str | None
    content_sha256: str
    serialization_abi: str | None
    license: str


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    id: str
    name: str
    source_kind: str
    content_sha256: str
    compiled_model_sha256: str
    compiled_schema_version: int
    restriction: str | None
    dependency_ids: tuple[str, ...]
    parameter_defaults: tuple[tuple[str, ComplexDecimal], ...]


@dataclass(frozen=True, slots=True)
class Process:
    id: str
    expression: str
    external_pdgs: tuple[int, ...]
    external_labels: tuple[int, ...]
    external_leg_ids: tuple[str, ...]
    external_spins: tuple[int, ...]
    external_colors: tuple[int, ...]
    external_masses: tuple[Decimal, ...]
    external_helicity_domains: tuple[tuple[int, ...], ...]
    initial_state_count: int
    alias_of: str | None
    final_state_permutation: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class PointAlgorithm:
    name: str
    version: str
    rng: str | None
    seed: int | None


@dataclass(frozen=True, slots=True)
class StressMetric:
    kind: str
    value: Decimal


@dataclass(frozen=True, slots=True)
class ReferencePoint:
    id: str
    process_id: str
    point_class: str
    algorithm: PointAlgorithm
    sqrt_s: Decimal
    momenta: tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]
    masses: tuple[Decimal, ...]
    arithmetic_precision_bits: int
    round_trip_decimal_digits: int
    certified_decimal_digits: int
    stress_metric: StressMetric | None

    def runtime_momenta(
        self,
    ) -> tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]:
        """Return momenta without reducing their recorded decimal precision."""

        return self.momenta

    def f64_momenta(self) -> tuple[tuple[float, float, float, float], ...]:
        """Return the explicit binary64 projection used by f64-only tests."""

        return tuple(
            (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            for row in self.momenta
        )

    def input_sha256(self) -> str:
        """Hash the ordered process and exact kinematic input contract."""

        payload = {
            "arithmetic_precision_bits": self.arithmetic_precision_bits,
            "certified_decimal_digits": self.certified_decimal_digits,
            "masses": [_decimal_text(mass) for mass in self.masses],
            "momenta": [
                [_decimal_text(component) for component in row] for row in self.momenta
            ],
            "process_id": self.process_id,
            "round_trip_decimal_digits": self.round_trip_decimal_digits,
            "sqrt_s": _decimal_text(self.sqrt_s),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Coverage:
    helicities: str
    color: str
    color_kind: str
    helicity_count: int
    color_component_count: int
    structural_zero_helicity_count: int


@dataclass(frozen=True, slots=True)
class Selectors:
    helicity: bool
    color_flow: bool
    omitted_helicity: str
    omitted_color: str


@dataclass(frozen=True, slots=True)
class Normalization:
    average_factor: Decimal
    color_factor: Decimal
    identical_factor: Decimal
    global_coupling_factor: Decimal
    quark_line_partner_factor: Decimal
    couplings_in_stage_evaluators: bool


@dataclass(frozen=True, slots=True)
class Topology:
    currents: int
    interactions: int
    roots: int
    reduction_groups: int


@dataclass(frozen=True, slots=True)
class HelicityAxis:
    id: str
    index: int
    values: tuple[int, ...]
    computed: bool
    structural_zero: bool
    representative_id: str
    coefficient: Decimal


@dataclass(frozen=True, slots=True)
class LCColorFlow:
    id: str
    index: int
    word: tuple[int, ...]
    computed: bool
    representative_id: str
    coefficient: Decimal


@dataclass(frozen=True, slots=True)
class ContractedColorComponent:
    id: str
    index: int
    description: str


ColorAxis: TypeAlias = LCColorFlow | ContractedColorComponent


@dataclass(frozen=True, slots=True)
class ReductionGroup:
    id: str
    representative_helicity_id: str
    representative_color_id: str
    physical_helicity_ids: tuple[str, ...]
    physical_color_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Reduction:
    kind: str
    cell_semantics: str
    groups: tuple[ReductionGroup, ...]
    plan_sha256: str


@dataclass(frozen=True, slots=True)
class Observation:
    point_id: str
    arithmetic_precision_bits: int
    round_trip_decimal_digits: int
    certified_decimal_digits: int
    values: tuple[tuple[Decimal, ...], ...]
    total: Decimal
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReferenceCase:
    id: str
    case_kind: str
    model_id: str
    process_id: str
    color_accuracy: str
    point_policy: str
    point_ids: tuple[str, ...]
    coverage: Coverage
    selectors: Selectors
    normalization: Normalization
    topology: Topology
    artifact_physics_sha256: str
    artifact_execution_sha256: str
    physics_case_sha256: str
    helicities: tuple[HelicityAxis, ...]
    colors: tuple[ColorAxis, ...]
    reduction: Reduction
    observations: tuple[Observation, ...]


@dataclass(frozen=True, slots=True)
class OracleProvenance:
    id: str
    name: str
    implementation: str
    revision: str
    content_sha256: str
    independence_statement: str
    validation_profile: str
    tolerance_ceiling: Tolerances


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    expression: str
    ordered_external_pdgs: tuple[int, ...]
    ordered_external_leg_ids: tuple[str, ...]
    source_to_row_permutation: tuple[int, ...]
    row_id: str | None
    color_order_count: int
    ordered_color_legs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Tolerances:
    relative: Decimal
    absolute: Decimal


@dataclass(frozen=True, slots=True)
class _EvidenceCoverage:
    resolved_cells: frozenset[tuple[str, str]] = frozenset()
    helicity_aggregates: frozenset[str] = frozenset()
    total: bool = False


@dataclass(frozen=True, slots=True)
class OracleEvidence:
    id: str
    evidence_set_id: str
    oracle_id: str
    case_id: str
    point_id: str
    arithmetic_precision_bits: int
    round_trip_decimal_digits: int
    certified_decimal_digits: int
    arithmetic: str
    coverage: str
    helicity_ids: tuple[str, ...]
    color_ids: tuple[str, ...]
    observed_total: Decimal
    observed_helicity_totals: tuple[Decimal, ...] | None
    observed_values: tuple[tuple[Decimal, ...], ...] | None
    process_identity: ProcessIdentity
    input_sha256: str
    physics_case_sha256: str
    oracle_output_sha256: str
    evidence_record_sha256: str
    command: tuple[str, ...]
    tolerances: Tolerances


@dataclass(frozen=True, slots=True)
class OracleEvidenceSet:
    id: str
    captured_at: str
    oracle: OracleProvenance
    dependency_ids: tuple[str, ...]
    records: tuple[OracleEvidence, ...]


@dataclass(frozen=True, slots=True)
class ReferenceFixture:
    id: str
    provenance: FixtureProvenance
    dependencies: tuple[DependencyProvenance, ...]
    models: tuple[ModelProvenance, ...]
    processes: tuple[Process, ...]
    points: tuple[ReferencePoint, ...]
    cases: tuple[ReferenceCase, ...]
    evidence_sets: tuple[OracleEvidenceSet, ...]

    def point(self, point_id: str) -> ReferencePoint:
        for point in self.points:
            if point.id == point_id:
                return point
        raise KeyError(point_id)
