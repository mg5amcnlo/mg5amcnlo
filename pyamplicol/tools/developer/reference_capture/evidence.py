# SPDX-License-Identifier: 0BSD
"""Independent oracle evidence and atomic fixture publication."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

from .common import (
    ANALYTIC_CERTIFIED_DIGITS,
    ANALYTIC_EVIDENCE_FILENAME,
    ANALYTIC_EVIDENCE_SET_ID,
    BUNDLE_MANIFEST_FILENAME,
    FORTRAN_EVIDENCE_FILENAME,
    PHYSICS_FILENAME,
    ROOT,
    CaptureError,
    Momentum,
    as_mapping,
    as_sequence,
    canonical_decimal,
    canonical_sha256,
    decimal_digits_to_bits,
    developer_module,
    json_bytes,
    sha256_file,
)


def point_input_sha256(point: Mapping[str, object]) -> str:
    """Digest the canonical physics input independently of fixture metadata."""

    payload = {
        "arithmetic_precision_bits": point["arithmetic_precision_bits"],
        "certified_decimal_digits": point["certified_decimal_digits"],
        "masses": point["masses"],
        "momenta": point["momenta"],
        "process_id": point["process_id"],
        "round_trip_decimal_digits": point["round_trip_decimal_digits"],
        "sqrt_s": point["sqrt_s"],
    }
    return canonical_sha256(payload)


def _point_momenta(point: Mapping[str, object]) -> tuple[Momentum, ...]:
    result: list[Momentum] = []
    for row_index, raw_row in enumerate(
        as_sequence(point.get("momenta"), "point.momenta")
    ):
        row = as_sequence(raw_row, f"point.momenta[{row_index}]")
        if len(row) != 4:
            raise CaptureError("analytic point momentum must contain four components")
        try:
            result.append(cast(Momentum, tuple(Decimal(str(value)) for value in row)))
        except InvalidOperation as error:
            raise CaptureError("analytic point contains an invalid decimal") from error
    return tuple(result)


def _analytic_output_digest(
    *,
    formula: str,
    input_sha256: str,
    coupling: Decimal,
    helicity_ids: Sequence[str],
    color_ids: Sequence[str],
    values: Sequence[Sequence[str]],
    total: str,
) -> str:
    return canonical_sha256(
        {
            "color_ids": list(color_ids),
            "coupling": canonical_decimal(coupling),
            "formula": formula,
            "helicity_ids": list(helicity_ids),
            "input_sha256": input_sha256,
            "total": total,
            "values": [list(row) for row in values],
        }
    )


def build_analytic_evidence(fixture: Mapping[str, object]) -> dict[str, object]:
    """Evaluate scalar formulas with explicit arithmetic and certified precision."""

    analytic = developer_module("analytic_oracles")
    process_records = tuple(
        as_mapping(value, "fixture process")
        for value in as_sequence(fixture.get("processes"), "fixture.processes")
    )
    point_records = tuple(
        as_mapping(value, "fixture point")
        for value in as_sequence(fixture.get("points"), "fixture.points")
    )
    model_records = tuple(
        as_mapping(value, "fixture model")
        for value in as_sequence(fixture.get("models"), "fixture.models")
    )
    processes = {str(process["id"]): process for process in process_records}
    points = {str(point["id"]): point for point in point_records}
    models = {str(model["id"]): model for model in model_records}
    records: list[dict[str, object]] = []
    for raw_case in as_sequence(fixture.get("cases"), "fixture.cases"):
        case = as_mapping(raw_case, "fixture case")
        if case["model_id"] == "model:builtin-sm":
            continue
        process = processes[str(case["process_id"])]
        model = models[str(case["model_id"])]
        axes = as_mapping(case.get("axes"), "case.axes")
        helicities = tuple(
            as_mapping(value, "case helicity")
            for value in as_sequence(axes.get("helicities"), "case.axes.helicities")
        )
        colors = tuple(
            as_mapping(value, "case color")
            for value in as_sequence(axes.get("colors"), "case.axes.colors")
        )
        if len(colors) != 1:
            raise CaptureError(
                f"analytic colorless case {case['id']} must have one color axis"
            )
        helicity_ids = tuple(str(value["id"]) for value in helicities)
        color_ids = (str(colors[0]["id"]),)
        defaults = as_mapping(
            model.get("parameter_defaults"), "model.parameter_defaults"
        )
        if case["model_id"] == "model:scalars-json":
            formula = "scalar_contact_2to2"
            coupling = Decimal(1)
        elif case["model_id"] == "model:scalar-gravity-json":
            formula = "scalar_gravity_2to2"
            kappa = as_mapping(defaults.get("kappa"), "model parameter kappa")
            coupling = Decimal(str(kappa["real"]))
        else:
            raise CaptureError(f"no analytic oracle is defined for {case['model_id']}")
        for point_id_value in as_sequence(case.get("point_ids"), "case.point_ids"):
            point_id = str(point_id_value)
            point = points[point_id]
            observation = (
                analytic.scalar_contact_2to2(
                    coupling,
                    certified_accuracy_digits=ANALYTIC_CERTIFIED_DIGITS,
                )
                if formula == "scalar_contact_2to2"
                else analytic.scalar_gravity_2to2(
                    _point_momenta(point),
                    coupling,
                    precision=100,
                    certified_accuracy_digits=ANALYTIC_CERTIFIED_DIGITS,
                )
            )
            values_by_id = dict(observation.resolved_by_helicity)
            if set(values_by_id) != set(helicity_ids):
                raise CaptureError(
                    f"analytic oracle axis differs from runtime axis for {case['id']}"
                )
            values = [
                [canonical_decimal(values_by_id[helicity_id])]
                for helicity_id in helicity_ids
            ]
            for helicity, row in zip(helicities, values, strict=True):
                if bool(helicity["structural_zero"]) and row != ["0"]:
                    raise CaptureError(
                        f"analytic oracle violated structural zero {helicity['id']}"
                    )
            total = canonical_decimal(observation.total)
            input_digest = point_input_sha256(point)
            record_id = f"evidence:analytic:{case['id']}:{point_id}"
            record: dict[str, object] = {
                "id": record_id,
                "case_id": case["id"],
                "point_id": point_id,
                "independent_of_pyamplicol": True,
                "arithmetic_precision_bits": decimal_digits_to_bits(
                    observation.metadata.arithmetic_precision_decimal_digits
                ),
                "round_trip_decimal_digits": (
                    observation.metadata.arithmetic_precision_decimal_digits
                ),
                "certified_decimal_digits": (
                    observation.metadata.certified_accuracy_decimal_digits
                ),
                "arithmetic": "analytic",
                "coverage": "resolved",
                "helicity_ids": list(helicity_ids),
                "color_ids": list(color_ids),
                "observed_total": total,
                "observed_helicity_totals": None,
                "observed_values": values,
                "process_identity": {
                    "expression": process["expression"],
                    "ordered_external_pdgs": process["external_pdgs"],
                    "ordered_external_leg_ids": process["external_leg_ids"],
                    "source_to_row_permutation": list(
                        range(len(cast(Sequence[object], process["external_pdgs"])))
                    ),
                    "row_id": None,
                    "color_order_count": len(colors),
                    "ordered_color_legs": [],
                },
                "input_sha256": input_digest,
                "physics_case_sha256": case["physics_case_sha256"],
                "oracle_output_sha256": _analytic_output_digest(
                    formula=formula,
                    input_sha256=input_digest,
                    coupling=coupling,
                    helicity_ids=helicity_ids,
                    color_ids=color_ids,
                    values=values,
                    total=total,
                ),
                "command": [
                    "tools/developer/analytic_oracles.py",
                    formula,
                    f"case={case['id']}",
                    f"point={point_id}",
                ],
                "tolerances": {
                    "relative": "0.000000000001",
                    "absolute": "0.000000000000001",
                },
            }
            reference = developer_module("reference_fixture_v2")
            record["evidence_record_sha256"] = reference.evidence_record_sha256(record)
            records.append(record)
    if not records:
        raise CaptureError("fixture contains no external-model analytic cases")
    provenance = as_mapping(fixture.get("provenance"), "fixture.provenance")
    return {
        "evidence_schema_version": 2,
        "kind": "pyamplicol-reference-oracle-evidence",
        "evidence_set_id": ANALYTIC_EVIDENCE_SET_ID,
        "captured_at": provenance["captured_at"],
        "oracle": {
            "id": "oracle:analytic-scalars",
            "name": "Independent closed-form scalar oracles",
            "implementation": "tools/developer/analytic_oracles.py Decimal formulas",
            "revision": provenance["source_revision"],
            "content_sha256": sha256_file(
                ROOT / "tools" / "developer" / "analytic_oracles.py"
            ),
            "independence_statement": (
                "Closed-form scalar formulas consume only captured momenta and "
                "model defaults; they do not import generation or runtime evaluators."
            ),
            "validation_profile": "high-precision",
            "tolerance_ceiling": {
                "relative": "0.000000000001",
                "absolute": "0.000000000000001",
            },
        },
        "dependency_ids": ["dependency:analytic-oracles"],
        "records": records,
    }


def build_fortran_evidence(
    fixture: Mapping[str, object],
    repository: Path,
    jobs: int,
) -> dict[str, object]:
    """Run the pinned legacy Fortran oracle through its public adapter."""

    if jobs < 1:
        raise CaptureError("legacy oracle jobs must be a positive integer")
    legacy = developer_module("legacy_amplicol")
    try:
        legacy.prepare_checkout(repository)
        with tempfile.TemporaryDirectory(
            prefix="pyamplicol-reference-v2-draft-"
        ) as raw:
            draft_path = Path(raw) / PHYSICS_FILENAME
            draft_path.write_bytes(json_bytes(fixture))
            return cast(
                dict[str, object],
                legacy.verify_fixture(
                    repository,
                    draft_path,
                    jobs=jobs,
                    build=True,
                ),
            )
    except Exception as error:
        raise CaptureError(f"pinned legacy Fortran oracle failed: {error}") from error


def validate_capture_documents(
    fixture: Mapping[str, object],
    evidence_documents: Sequence[Mapping[str, object]],
) -> None:
    """Invoke the strict fixture parser with the complete evidence set."""

    reference = developer_module("reference_fixture_v2")
    try:
        reference.parse_reference_fixture(fixture, evidence_documents)
    except Exception as error:
        raise CaptureError(
            "strict reference fixture validation failed; no final files written: "
            f"{error}"
        ) from error


def atomic_write_documents(
    output_directory: Path,
    documents: Mapping[str, Mapping[str, object]],
    *,
    bundle_manifest_name: str | None = None,
) -> tuple[Path, ...]:
    """Publish JSON documents, optionally committing them with a final manifest."""

    if not documents:
        raise CaptureError("no capture documents were supplied")
    if any(Path(name).name != name for name in documents):
        raise CaptureError("capture document names must be plain filenames")
    if bundle_manifest_name is not None and (
        Path(bundle_manifest_name).name != bundle_manifest_name
        or bundle_manifest_name in documents
    ):
        raise CaptureError("capture bundle manifest name is invalid")
    output_directory.mkdir(parents=True, exist_ok=True)
    encoded_documents = {
        name: json_bytes(document) for name, document in documents.items()
    }
    document_targets = tuple(output_directory / name for name in documents)
    publication = list(encoded_documents.items())
    manifest_target: Path | None = None
    if bundle_manifest_name is not None:
        file_records = [
            {
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in sorted(encoded_documents.items())
        ]
        manifest: dict[str, object] = {
            "schema_version": 1,
            "kind": "pyamplicol-reference-fixture-bundle",
            "files": file_records,
            "bundle_sha256": canonical_sha256(
                {
                    "contract": "pyamplicol-reference-fixture-bundle-v1",
                    "files": file_records,
                }
            ),
        }
        publication.append((bundle_manifest_name, json_bytes(manifest)))
        manifest_target = output_directory / bundle_manifest_name
    all_targets = tuple(output_directory / name for name, _content in publication)
    existing = tuple(path for path in all_targets if path.exists())
    if existing:
        raise CaptureError(
            "capture output already exists and will not be overwritten: "
            + ", ".join(str(path) for path in existing)
        )
    staged: list[Path] = []
    published: list[Path] = []
    try:
        for name, content in publication:
            descriptor, raw_path = tempfile.mkstemp(
                prefix=f".{name}.",
                suffix=".tmp",
                dir=output_directory,
            )
            path = Path(raw_path)
            staged.append(path)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for path, target in zip(staged, all_targets, strict=True):
            try:
                os.link(path, target)
            except FileExistsError as error:
                raise CaptureError(
                    "capture output appeared concurrently and was not overwritten: "
                    f"{target}"
                ) from error
            published.append(target)
        directory_descriptor = os.open(output_directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
    if manifest_target is not None and published[-1] != manifest_target:
        raise AssertionError("bundle manifest was not published last")
    return document_targets


def validate_and_publish(
    output_directory: Path,
    fixture: Mapping[str, object],
    fortran_evidence: Mapping[str, object],
    analytic_evidence: Mapping[str, object],
) -> tuple[Path, ...]:
    """Strictly validate the whole bundle before creating any final file."""

    validate_capture_documents(fixture, (fortran_evidence, analytic_evidence))
    return atomic_write_documents(
        output_directory,
        {
            PHYSICS_FILENAME: fixture,
            FORTRAN_EVIDENCE_FILENAME: fortran_evidence,
            ANALYTIC_EVIDENCE_FILENAME: analytic_evidence,
        },
        bundle_manifest_name=BUNDLE_MANIFEST_FILENAME,
    )
