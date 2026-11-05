# SPDX-License-Identifier: 0BSD
"""Public parsing API and cross-document fixture validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path

from .codec import (
    _array,
    _canonical_sha256,
    _object,
    _parse_case,
    _parse_dependency,
    _parse_evidence_set,
    _parse_model,
    _parse_point,
    _parse_process,
    _parse_provenance,
    _string,
    _strings,
    _validate_evidence_record_hashes,
    _validate_physics_case_hashes,
    _validate_reduction_plan_hashes,
    _validate_wire,
)
from .evidence import _validate_evidence, _validate_oracle_provenance
from .model import (
    Observation,
    OracleEvidence,
    Process,
    ReferenceFixture,
    ReferenceFixtureError,
    ReferencePoint,
)
from .numerics import (
    _as_fraction,
    _exact_sum,
    _required_stress_certified_digits,
    _validate_point_kinematics,
    _validate_precision_metadata,
)
from .structure import (
    _unique,
    _validate_case_axes,
    _validate_digest,
    _validate_observation_reduction_relations,
    _validate_processes,
)


def _validate_semantics(fixture: ReferenceFixture) -> None:
    dependencies = _unique(fixture.dependencies, "dependency")
    models = _unique(fixture.models, "model")
    processes = _unique(fixture.processes, "process")
    points = _unique(fixture.points, "point")
    _unique(fixture.cases, "case")
    evidence_sets = _unique(fixture.evidence_sets, "evidence set")
    evidence = _unique(
        (
            record
            for evidence_set in fixture.evidence_sets
            for record in evidence_set.records
        ),
        "evidence",
    )
    oracle_by_evidence_id = {
        record.id: evidence_set.oracle
        for evidence_set in fixture.evidence_sets
        for record in evidence_set.records
    }

    if not fixture.provenance.working_tree_clean:
        raise ReferenceFixtureError("reference fixtures require a clean source tree")
    if fixture.provenance.source_revision == "0" * 40:
        raise ReferenceFixtureError("reference fixtures require a real source revision")
    _validate_digest(
        fixture.provenance.source_tree_sha256,
        "provenance.source_tree_sha256",
    )
    for dependency in fixture.dependencies:
        _validate_digest(
            dependency.content_sha256,
            f"dependency {dependency.id} content_sha256",
        )
    for model in fixture.models:
        _validate_digest(model.content_sha256, f"model {model.id} content_sha256")
        _validate_digest(
            model.compiled_model_sha256,
            f"model {model.id} compiled_model_sha256",
        )
    for case in fixture.cases:
        _validate_digest(
            case.artifact_physics_sha256,
            f"case {case.id} artifact_physics_sha256",
        )
        _validate_digest(
            case.artifact_execution_sha256,
            f"case {case.id} artifact_execution_sha256",
        )
        _validate_digest(
            case.physics_case_sha256,
            f"case {case.id} physics_case_sha256",
        )
        _validate_digest(
            case.reduction.plan_sha256,
            f"case {case.id} reduction.plan_sha256",
        )
    for evidence_set in fixture.evidence_sets:
        _validate_oracle_provenance(evidence_set.oracle)
        _validate_digest(
            evidence_set.oracle.content_sha256,
            f"oracle {evidence_set.oracle.id} content_sha256",
        )
        for record in evidence_set.records:
            _validate_digest(
                record.oracle_output_sha256,
                f"evidence {record.id} oracle_output_sha256",
            )
            _validate_digest(
                record.evidence_record_sha256,
                f"evidence {record.id} evidence_record_sha256",
            )
            _validate_digest(
                record.physics_case_sha256,
                f"evidence {record.id} physics_case_sha256",
            )

    _validate_processes(processes)
    for model in fixture.models:
        unknown = set(model.dependency_ids) - dependencies.keys()
        if unknown:
            raise ReferenceFixtureError(
                f"model {model.id} references unknown dependencies: {sorted(unknown)}"
            )
    for evidence_set in fixture.evidence_sets:
        unknown = set(evidence_set.dependency_ids) - dependencies.keys()
        if unknown:
            raise ReferenceFixtureError(
                f"evidence set {evidence_set.id} references unknown dependencies: "
                f"{sorted(unknown)}"
            )

    masses_by_process: dict[str, tuple[Decimal, ...]] = {}
    for point in fixture.points:
        process = processes.get(point.process_id)
        if not isinstance(process, Process):
            raise ReferenceFixtureError(
                f"point {point.id} references unknown process {point.process_id}"
            )
        _validate_point_kinematics(point, process)
        previous_masses = masses_by_process.setdefault(point.process_id, point.masses)
        if previous_masses != point.masses:
            raise ReferenceFixtureError(
                f"point {point.id} changes explicit per-leg masses for process "
                f"{process.id}"
            )

    used_evidence: set[str] = set()
    for case in fixture.cases:
        if case.model_id not in models:
            raise ReferenceFixtureError(
                f"case {case.id} references unknown model {case.model_id}"
            )
        process = processes.get(case.process_id)
        if not isinstance(process, Process):
            raise ReferenceFixtureError(
                f"case {case.id} references unknown process {case.process_id}"
            )
        case_points: list[ReferencePoint] = []
        for point_id in case.point_ids:
            case_point = points.get(point_id)
            if not isinstance(case_point, ReferencePoint):
                raise ReferenceFixtureError(
                    f"case {case.id} references unknown point {point_id}"
                )
            if case_point.process_id != case.process_id:
                raise ReferenceFixtureError(
                    f"case {case.id} point {case_point.id} belongs to another process"
                )
            case_points.append(case_point)
        if case.point_policy == "standard":
            generic_count = sum(point.point_class == "generic" for point in case_points)
            stress_count = sum(point.point_class == "stress" for point in case_points)
            if generic_count < 3 or stress_count < 1:
                raise ReferenceFixtureError(
                    f"standard case {case.id} requires at least three generic "
                    "points and one stress point"
                )
            physics_momenta = [
                point.momenta
                for point in case_points
                if point.point_class in {"generic", "stress"}
            ]
            if len(set(physics_momenta)) != len(physics_momenta):
                raise ReferenceFixtureError(
                    f"standard case {case.id} requires distinct generic/stress momenta"
                )
        else:
            if process.initial_state_count != 2 or len(process.external_pdgs) != 3:
                raise ReferenceFixtureError(
                    f"degenerate-2to1 case {case.id} requires exactly two incoming "
                    "and one outgoing particle"
                )
            if len(case_points) != 1 or case_points[0].point_class != "canonical":
                raise ReferenceFixtureError(
                    f"degenerate-2to1 case {case.id} requires exactly one "
                    "canonical point"
                )
        _validate_case_axes(case, process)
        for value in (
            case.normalization.average_factor,
            case.normalization.color_factor,
            case.normalization.identical_factor,
            case.normalization.global_coupling_factor,
            case.normalization.quark_line_partner_factor,
        ):
            if value <= 0:
                raise ReferenceFixtureError(
                    f"case {case.id} normalization factors must be positive"
                )

        observations = _unique(
            case.observations,
            f"case {case.id} observation",
            key="point_id",
        )
        if set(observations) != set(case.point_ids):
            raise ReferenceFixtureError(
                f"case {case.id} must contain exactly one observation per point"
            )
        for point in case_points:
            observation = observations[point.id]
            assert isinstance(observation, Observation)
            _validate_precision_metadata(
                arithmetic_precision_bits=observation.arithmetic_precision_bits,
                round_trip_decimal_digits=observation.round_trip_decimal_digits,
                certified_decimal_digits=observation.certified_decimal_digits,
                where=f"observation {case.id}/{point.id}",
            )
            if observation.certified_decimal_digits > point.certified_decimal_digits:
                raise ReferenceFixtureError(
                    f"observation {case.id}/{point.id} overclaims the certified "
                    "point precision"
                )
            if point.point_class == "stress" and (
                observation.certified_decimal_digits
                < _required_stress_certified_digits(point)
            ):
                raise ReferenceFixtureError(
                    f"stress observation {case.id}/{point.id} certification is not "
                    "commensurate with the stress metric"
                )
            if len(observation.values) != len(case.helicities) or any(
                len(row) != len(case.colors) for row in observation.values
            ):
                raise ReferenceFixtureError(
                    f"observation {case.id}/{point.id} must be a dense "
                    "[helicity][color] matrix"
                )
            for helicity, row in zip(case.helicities, observation.values, strict=True):
                if helicity.structural_zero and any(value != 0 for value in row):
                    raise ReferenceFixtureError(
                        f"observation {case.id}/{point.id} does not explicitly "
                        f"zero structural helicity {helicity.id}"
                    )
            _validate_observation_reduction_relations(case, observation)
            total = _exact_sum(value for row in observation.values for value in row)
            if total != _as_fraction(observation.total):
                raise ReferenceFixtureError(
                    f"observation {case.id}/{point.id} component sum does not "
                    "equal its declared total"
                )
            covered_cells: set[tuple[str, str]] = set()
            covered_aggregates: set[str] = set()
            for evidence_id in observation.evidence_refs:
                evidence_record = evidence.get(evidence_id)
                if not isinstance(evidence_record, OracleEvidence):
                    raise ReferenceFixtureError(
                        f"observation {case.id}/{point.id} references unknown "
                        f"evidence {evidence_id}"
                    )
                used_evidence.add(evidence_id)
                oracle = oracle_by_evidence_id[evidence_id]
                coverage = _validate_evidence(
                    evidence_record,
                    oracle=oracle,
                    case=case,
                    process=process,
                    point=point,
                    observation=observation,
                )
                covered_cells.update(coverage.resolved_cells)
                covered_aggregates.update(coverage.helicity_aggregates)
            color_ids = {color.id for color in case.colors}
            fully_resolved_helicities = {
                helicity.id
                for helicity in case.helicities
                if all(
                    (helicity.id, color_id) in covered_cells for color_id in color_ids
                )
            }
            required_helicities = {helicity.id for helicity in case.helicities}
            if fully_resolved_helicities | covered_aggregates != required_helicities:
                raise ReferenceFixtureError(
                    f"observation {case.id}/{point.id} lacks independent resolved "
                    "or complete-color aggregate evidence for every helicity"
                )

    unused = set(evidence) - used_evidence
    if unused:
        raise ReferenceFixtureError(f"unreferenced evidence records: {sorted(unused)}")
    if not evidence_sets:
        raise ReferenceFixtureError("at least one evidence set is required")


def parse_reference_fixture(
    payload: Mapping[str, object],
    evidence_payloads: Sequence[Mapping[str, object]],
) -> ReferenceFixture:
    """Validate wire payloads and return an immutable typed fixture."""

    _validate_wire(payload, "reference-physics-v2.schema.json", "fixture")
    for index, evidence_payload in enumerate(evidence_payloads):
        _validate_wire(
            evidence_payload,
            "reference-oracle-evidence-v2.schema.json",
            f"evidence[{index}]",
        )
        _validate_evidence_record_hashes(evidence_payload)
    _validate_reduction_plan_hashes(payload)
    _validate_physics_case_hashes(payload)

    fixture_evidence_ids = _strings(payload["evidence_sets"], "evidence_sets")
    parsed_evidence = tuple(
        _parse_evidence_set(evidence_payload, index)
        for index, evidence_payload in enumerate(evidence_payloads)
    )
    if set(fixture_evidence_ids) != {item.id for item in parsed_evidence} or len(
        fixture_evidence_ids
    ) != len(parsed_evidence):
        raise ReferenceFixtureError(
            "fixture evidence_sets must match the supplied evidence documents exactly"
        )

    fixture = ReferenceFixture(
        id=_string(payload["fixture_id"], "fixture_id"),
        provenance=_parse_provenance(payload["provenance"]),
        dependencies=tuple(
            _parse_dependency(item, index)
            for index, item in enumerate(
                _array(payload["dependencies"], "dependencies")
            )
        ),
        models=tuple(
            _parse_model(item, index)
            for index, item in enumerate(_array(payload["models"], "models"))
        ),
        processes=tuple(
            _parse_process(item, index)
            for index, item in enumerate(_array(payload["processes"], "processes"))
        ),
        points=tuple(
            _parse_point(item, index)
            for index, item in enumerate(_array(payload["points"], "points"))
        ),
        cases=tuple(
            _parse_case(item, index)
            for index, item in enumerate(_array(payload["cases"], "cases"))
        ),
        evidence_sets=parsed_evidence,
    )
    _validate_semantics(fixture)
    return fixture


def load_reference_fixture(
    fixture_path: Path,
    evidence_paths: Sequence[Path],
) -> ReferenceFixture:
    """Load a reference fixture and its independent evidence documents."""

    all_paths = (fixture_path, *evidence_paths)
    parents = {path.resolve(strict=False).parent for path in all_paths}
    if len(parents) != 1:
        raise ReferenceFixtureError(
            "fixture and evidence documents must share one bundle directory"
        )
    directory = next(iter(parents))
    manifest_path = directory / "reference-fixture-v2.manifest.json"
    try:
        manifest = _object(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            str(manifest_path),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ReferenceFixtureError(
            f"reference fixture bundle commit marker is unavailable: {error}"
        ) from error
    _validate_wire(
        manifest,
        "reference-fixture-bundle-v1.schema.json",
        "fixture bundle manifest",
    )
    expected_names = tuple(sorted(path.name for path in all_paths))
    records = tuple(
        _object(value, f"fixture bundle files[{index}]")
        for index, value in enumerate(_array(manifest["files"], "files"))
    )
    actual_names = tuple(str(record["path"]) for record in records)
    if actual_names != expected_names:
        raise ReferenceFixtureError(
            "fixture bundle manifest does not identify exactly the requested documents"
        )
    for record in records:
        path = directory / str(record["path"])
        try:
            actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ReferenceFixtureError(
                f"fixture bundle member cannot be read: {path.name}: {error}"
            ) from error
        if actual_digest != record["sha256"]:
            raise ReferenceFixtureError(
                f"fixture bundle member digest mismatch: {path.name}"
            )
    expected_bundle_digest = _canonical_sha256(
        {
            "contract": "pyamplicol-reference-fixture-bundle-v1",
            "files": [dict(record) for record in records],
        }
    )
    if manifest["bundle_sha256"] != expected_bundle_digest:
        raise ReferenceFixtureError("fixture bundle manifest digest is inconsistent")

    payload = _object(
        json.loads(fixture_path.read_text(encoding="utf-8")),
        str(fixture_path),
    )
    evidence_payloads = tuple(
        _object(json.loads(path.read_text(encoding="utf-8")), str(path))
        for path in evidence_paths
    )
    return parse_reference_fixture(payload, evidence_payloads)
