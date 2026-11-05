# SPDX-License-Identifier: 0BSD
"""Wire-format parsing and canonical hashing for reference fixtures."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

import jsonschema  # type: ignore[import-untyped]

from .model import (
    ColorAxis,
    ComplexDecimal,
    ContractedColorComponent,
    Coverage,
    DependencyProvenance,
    FixtureProvenance,
    HelicityAxis,
    LCColorFlow,
    ModelProvenance,
    Normalization,
    Observation,
    OracleEvidence,
    OracleEvidenceSet,
    OracleProvenance,
    PointAlgorithm,
    Process,
    ProcessIdentity,
    Reduction,
    ReductionGroup,
    ReferenceCase,
    ReferenceFixtureError,
    ReferencePoint,
    Selectors,
    StressMetric,
    Tolerances,
    Topology,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_ROOT = ROOT / "schemas"

_DECIMAL_RE = re.compile(
    r"^(?:0|-?(?:(?:[1-9][0-9]*)(?:\.[0-9]*[1-9])?|0\.[0-9]*[1-9]))$"
)


def _schema(name: str) -> Mapping[str, object]:
    payload = json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))
    return _object(payload, f"schema {name}")


def _validate_wire(payload: object, schema_name: str, label: str) -> None:
    validator = jsonschema.Draft202012Validator(
        _schema(schema_name),
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    raise ReferenceFixtureError(f"{label} schema violation at {path}: {error.message}")


def _object(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReferenceFixtureError(f"{where} must be an object")
    return value


def _array(value: object, where: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReferenceFixtureError(f"{where} must be an array")
    return value


def _string(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise ReferenceFixtureError(f"{where} must be a string")
    return value


def _optional_string(value: object, where: str) -> str | None:
    if value is None:
        return None
    return _string(value, where)


def _integer(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReferenceFixtureError(f"{where} must be an integer")
    return value


def _optional_integer(value: object, where: str) -> int | None:
    if value is None:
        return None
    return _integer(value, where)


def _boolean(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise ReferenceFixtureError(f"{where} must be a boolean")
    return value


def _decimal(value: object, where: str) -> Decimal:
    text = _string(value, where)
    if not _DECIMAL_RE.fullmatch(text):
        raise ReferenceFixtureError(
            f"{where} must be a canonical fixed-point decimal string"
        )
    try:
        parsed = Decimal(text)
    except InvalidOperation as error:
        raise ReferenceFixtureError(f"{where} is not a finite decimal") from error
    if not parsed.is_finite() or _decimal_text(parsed) != text:
        raise ReferenceFixtureError(
            f"{where} must be a canonical fixed-point decimal string"
        )
    return parsed


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def evidence_record_sha256(record: Mapping[str, object]) -> str:
    """Hash every canonical evidence field except the digest itself."""

    payload = dict(record)
    payload.pop("evidence_record_sha256", None)
    return _canonical_sha256(
        {
            "contract": "pyamplicol-reference-oracle-evidence-record-v2",
            "record": payload,
        }
    )


def reduction_plan_sha256(reduction: Mapping[str, object]) -> str:
    """Hash the complete normalized reduction mapping except its digest."""

    payload = dict(reduction)
    payload.pop("plan_sha256", None)
    return _canonical_sha256(
        {
            "contract": "pyamplicol-reference-reduction-plan-v1",
            "reduction": payload,
        }
    )


def _validate_reduction_plan_hashes(payload: Mapping[str, object]) -> None:
    for index, raw_case in enumerate(_array(payload["cases"], "cases")):
        case = _object(raw_case, f"cases[{index}]")
        reduction = _object(case["reduction"], f"cases[{index}].reduction")
        actual = _string(
            reduction["plan_sha256"],
            f"cases[{index}].reduction.plan_sha256",
        )
        expected = reduction_plan_sha256(reduction)
        if actual != expected:
            raise ReferenceFixtureError(
                f"case {case.get('id', index)!r} reduction plan hash does not "
                "cover its complete normalized mapping"
            )


def _validate_evidence_record_hashes(payload: Mapping[str, object]) -> None:
    for index, raw_record in enumerate(_array(payload["records"], "records")):
        record = _object(raw_record, f"records[{index}]")
        actual = _string(
            record["evidence_record_sha256"],
            f"records[{index}].evidence_record_sha256",
        )
        expected = evidence_record_sha256(record)
        if actual != expected:
            raise ReferenceFixtureError(
                f"evidence record {record.get('id', index)!r} canonical hash "
                "does not cover its complete normalized record"
            )


def physics_case_sha256(
    fixture_payload: Mapping[str, object],
    case_id: str,
) -> str:
    """Hash the complete canonical contract for one fixture physics case."""

    cases = [
        _object(value, "case")
        for value in _array(fixture_payload["cases"], "cases")
        if _object(value, "case").get("id") == case_id
    ]
    if len(cases) != 1:
        raise ReferenceFixtureError(
            f"physics case contract requires exactly one case {case_id}"
        )
    case = cases[0]
    model_id = _string(case["model_id"], f"case {case_id}.model_id")
    process_id = _string(case["process_id"], f"case {case_id}.process_id")

    models = [
        _object(value, "model")
        for value in _array(fixture_payload["models"], "models")
        if _object(value, "model").get("id") == model_id
    ]
    process_records = [
        _object(value, "process")
        for value in _array(fixture_payload["processes"], "processes")
    ]
    processes_by_id = {
        _string(value["id"], "process.id"): value for value in process_records
    }
    process = processes_by_id.get(process_id)
    if len(models) != 1 or process is None:
        raise ReferenceFixtureError(
            f"physics case contract {case_id} has ambiguous model or process identity"
        )
    model = models[0]
    process_chain = [process]
    seen_process_ids = {process_id}
    while process_chain[-1].get("alias_of") is not None:
        source_id = _string(
            process_chain[-1]["alias_of"],
            f"process {process_chain[-1]['id']}.alias_of",
        )
        if source_id in seen_process_ids:
            raise ReferenceFixtureError(
                f"physics case contract {case_id} contains a process alias cycle"
            )
        source = processes_by_id.get(source_id)
        if source is None:
            raise ReferenceFixtureError(
                f"physics case contract {case_id} references unknown alias source "
                f"{source_id}"
            )
        seen_process_ids.add(source_id)
        process_chain.append(source)

    dependencies_by_id = {
        _string(item["id"], "dependency.id"): item
        for item in (
            _object(value, "dependency")
            for value in _array(fixture_payload["dependencies"], "dependencies")
        )
    }
    dependency_ids = _strings(
        model["dependency_ids"], f"model {model_id}.dependency_ids"
    )
    try:
        model_dependencies = [dependencies_by_id[value] for value in dependency_ids]
    except KeyError as error:
        raise ReferenceFixtureError(
            f"physics case contract {case_id} references an unknown dependency"
        ) from error

    points_by_id = {
        _string(item["id"], "point.id"): item
        for item in (
            _object(value, "point")
            for value in _array(fixture_payload["points"], "points")
        )
    }
    point_ids = _strings(case["point_ids"], f"case {case_id}.point_ids")
    try:
        case_points = [points_by_id[value] for value in point_ids]
    except KeyError as error:
        raise ReferenceFixtureError(
            f"physics case contract {case_id} references an unknown point"
        ) from error

    canonical_case = {
        key: value for key, value in case.items() if key != "physics_case_sha256"
    }
    return _canonical_sha256(
        {
            "case": canonical_case,
            "contract": "pyamplicol-reference-physics-case-v2",
            "fixture_kind": fixture_payload["kind"],
            "fixture_schema_version": fixture_payload["fixture_schema_version"],
            "model": model,
            "model_dependencies": model_dependencies,
            "points": case_points,
            "process": process,
            "process_alias_sources": process_chain[1:],
        }
    )


def _validate_physics_case_hashes(payload: Mapping[str, object]) -> None:
    for index, value in enumerate(_array(payload["cases"], "cases")):
        case = _object(value, f"cases[{index}]")
        case_id = _string(case["id"], f"cases[{index}].id")
        actual = _string(
            case["physics_case_sha256"], f"cases[{index}].physics_case_sha256"
        )
        expected = physics_case_sha256(payload, case_id)
        if actual != expected:
            raise ReferenceFixtureError(
                f"case {case_id} physics case hash does not cover its complete "
                "canonical contract"
            )


def _strings(value: object, where: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{where}[{index}]")
        for index, item in enumerate(_array(value, where))
    )


def _integers(value: object, where: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, f"{where}[{index}]")
        for index, item in enumerate(_array(value, where))
    )


def _decimals(value: object, where: str) -> tuple[Decimal, ...]:
    return tuple(
        _decimal(item, f"{where}[{index}]")
        for index, item in enumerate(_array(value, where))
    )


def _decimal_matrix(
    value: object,
    where: str,
) -> tuple[tuple[Decimal, ...], ...]:
    return tuple(
        _decimals(row, f"{where}[{index}]")
        for index, row in enumerate(_array(value, where))
    )


def _parse_provenance(value: object) -> FixtureProvenance:
    item = _object(value, "provenance")
    return FixtureProvenance(
        source_repository=_string(item["source_repository"], "source_repository"),
        source_revision=_string(item["source_revision"], "source_revision"),
        source_tree_sha256=_string(item["source_tree_sha256"], "source_tree_sha256"),
        captured_at=_string(item["captured_at"], "captured_at"),
        capture_command=_strings(item["capture_command"], "capture_command"),
        working_tree_clean=_boolean(item["working_tree_clean"], "working_tree_clean"),
        memory_watchdog_gb=_integer(item["memory_watchdog_gb"], "memory_watchdog_gb"),
    )


def _parse_dependency(value: object, index: int) -> DependencyProvenance:
    where = f"dependencies[{index}]"
    item = _object(value, where)
    return DependencyProvenance(
        id=_string(item["id"], f"{where}.id"),
        name=_string(item["name"], f"{where}.name"),
        version=_string(item["version"], f"{where}.version"),
        revision=_optional_string(item["revision"], f"{where}.revision"),
        content_sha256=_string(item["content_sha256"], f"{where}.content_sha256"),
        serialization_abi=_optional_string(
            item["serialization_abi"], f"{where}.serialization_abi"
        ),
        license=_string(item["license"], f"{where}.license"),
    )


def _parse_complex_decimal(value: object, where: str) -> ComplexDecimal:
    item = _object(value, where)
    return ComplexDecimal(
        real=_decimal(item["real"], f"{where}.real"),
        imag=_decimal(item["imag"], f"{where}.imag"),
    )


def _parse_model(value: object, index: int) -> ModelProvenance:
    where = f"models[{index}]"
    item = _object(value, where)
    parameters = _object(item["parameter_defaults"], f"{where}.parameter_defaults")
    return ModelProvenance(
        id=_string(item["id"], f"{where}.id"),
        name=_string(item["name"], f"{where}.name"),
        source_kind=_string(item["source_kind"], f"{where}.source_kind"),
        content_sha256=_string(item["content_sha256"], f"{where}.content_sha256"),
        compiled_model_sha256=_string(
            item["compiled_model_sha256"], f"{where}.compiled_model_sha256"
        ),
        compiled_schema_version=_integer(
            item["compiled_schema_version"], f"{where}.compiled_schema_version"
        ),
        restriction=_optional_string(item["restriction"], f"{where}.restriction"),
        dependency_ids=_strings(item["dependency_ids"], f"{where}.dependency_ids"),
        parameter_defaults=tuple(
            (
                name,
                _parse_complex_decimal(parameter, f"{where}.parameter_defaults.{name}"),
            )
            for name, parameter in sorted(parameters.items())
        ),
    )


def _parse_process(value: object, index: int) -> Process:
    where = f"processes[{index}]"
    item = _object(value, where)
    permutation = item["final_state_permutation"]
    return Process(
        id=_string(item["id"], f"{where}.id"),
        expression=_string(item["expression"], f"{where}.expression"),
        external_pdgs=_integers(item["external_pdgs"], f"{where}.external_pdgs"),
        external_labels=_integers(item["external_labels"], f"{where}.external_labels"),
        external_leg_ids=_strings(
            item["external_leg_ids"], f"{where}.external_leg_ids"
        ),
        external_spins=_integers(item["external_spins"], f"{where}.external_spins"),
        external_colors=_integers(item["external_colors"], f"{where}.external_colors"),
        external_masses=_decimals(item["external_masses"], f"{where}.external_masses"),
        external_helicity_domains=tuple(
            _integers(domain, f"{where}.external_helicity_domains[{domain_index}]")
            for domain_index, domain in enumerate(
                _array(
                    item["external_helicity_domains"],
                    f"{where}.external_helicity_domains",
                )
            )
        ),
        initial_state_count=_integer(
            item["initial_state_count"], f"{where}.initial_state_count"
        ),
        alias_of=_optional_string(item["alias_of"], f"{where}.alias_of"),
        final_state_permutation=(
            None
            if permutation is None
            else _integers(permutation, f"{where}.final_state_permutation")
        ),
    )


def _parse_stress_metric(value: object, where: str) -> StressMetric | None:
    if value is None:
        return None
    item = _object(value, where)
    return StressMetric(
        kind=_string(item["kind"], f"{where}.kind"),
        value=_decimal(item["value"], f"{where}.value"),
    )


def _parse_point(value: object, index: int) -> ReferencePoint:
    where = f"points[{index}]"
    item = _object(value, where)
    algorithm = _object(item["algorithm"], f"{where}.algorithm")
    momenta: list[tuple[Decimal, Decimal, Decimal, Decimal]] = []
    for row_index, row_value in enumerate(_array(item["momenta"], f"{where}.momenta")):
        row = _decimals(row_value, f"{where}.momenta[{row_index}]")
        if len(row) != 4:
            raise ReferenceFixtureError(
                f"{where}.momenta[{row_index}] must contain four components"
            )
        momenta.append((row[0], row[1], row[2], row[3]))
    return ReferencePoint(
        id=_string(item["id"], f"{where}.id"),
        process_id=_string(item["process_id"], f"{where}.process_id"),
        point_class=_string(item["class"], f"{where}.class"),
        algorithm=PointAlgorithm(
            name=_string(algorithm["name"], f"{where}.algorithm.name"),
            version=_string(algorithm["version"], f"{where}.algorithm.version"),
            rng=_optional_string(algorithm["rng"], f"{where}.algorithm.rng"),
            seed=_optional_integer(algorithm["seed"], f"{where}.algorithm.seed"),
        ),
        sqrt_s=_decimal(item["sqrt_s"], f"{where}.sqrt_s"),
        momenta=tuple(momenta),
        masses=_decimals(item["masses"], f"{where}.masses"),
        arithmetic_precision_bits=_integer(
            item["arithmetic_precision_bits"],
            f"{where}.arithmetic_precision_bits",
        ),
        round_trip_decimal_digits=_integer(
            item["round_trip_decimal_digits"],
            f"{where}.round_trip_decimal_digits",
        ),
        certified_decimal_digits=_integer(
            item["certified_decimal_digits"],
            f"{where}.certified_decimal_digits",
        ),
        stress_metric=_parse_stress_metric(
            item["stress_metric"], f"{where}.stress_metric"
        ),
    )


def _parse_coverage(value: object, where: str) -> Coverage:
    item = _object(value, where)
    return Coverage(
        helicities=_string(item["helicities"], f"{where}.helicities"),
        color=_string(item["color"], f"{where}.color"),
        color_kind=_string(item["color_kind"], f"{where}.color_kind"),
        helicity_count=_integer(item["helicity_count"], f"{where}.helicity_count"),
        color_component_count=_integer(
            item["color_component_count"], f"{where}.color_component_count"
        ),
        structural_zero_helicity_count=_integer(
            item["structural_zero_helicity_count"],
            f"{where}.structural_zero_helicity_count",
        ),
    )


def _parse_selectors(value: object, where: str) -> Selectors:
    item = _object(value, where)
    return Selectors(
        helicity=_boolean(item["helicity"], f"{where}.helicity"),
        color_flow=_boolean(item["color_flow"], f"{where}.color_flow"),
        omitted_helicity=_string(item["omitted_helicity"], f"{where}.omitted_helicity"),
        omitted_color=_string(item["omitted_color"], f"{where}.omitted_color"),
    )


def _parse_normalization(value: object, where: str) -> Normalization:
    item = _object(value, where)
    return Normalization(
        average_factor=_decimal(item["average_factor"], f"{where}.average_factor"),
        color_factor=_decimal(item["color_factor"], f"{where}.color_factor"),
        identical_factor=_decimal(
            item["identical_factor"], f"{where}.identical_factor"
        ),
        global_coupling_factor=_decimal(
            item["global_coupling_factor"], f"{where}.global_coupling_factor"
        ),
        quark_line_partner_factor=_decimal(
            item["quark_line_partner_factor"],
            f"{where}.quark_line_partner_factor",
        ),
        couplings_in_stage_evaluators=_boolean(
            item["couplings_in_stage_evaluators"],
            f"{where}.couplings_in_stage_evaluators",
        ),
    )


def _parse_topology(value: object, where: str) -> Topology:
    item = _object(value, where)
    return Topology(
        currents=_integer(item["currents"], f"{where}.currents"),
        interactions=_integer(item["interactions"], f"{where}.interactions"),
        roots=_integer(item["roots"], f"{where}.roots"),
        reduction_groups=_integer(
            item["reduction_groups"], f"{where}.reduction_groups"
        ),
    )


def _parse_helicity(value: object, where: str) -> HelicityAxis:
    item = _object(value, where)
    return HelicityAxis(
        id=_string(item["id"], f"{where}.id"),
        index=_integer(item["index"], f"{where}.index"),
        values=_integers(item["values"], f"{where}.values"),
        computed=_boolean(item["computed"], f"{where}.computed"),
        structural_zero=_boolean(item["structural_zero"], f"{where}.structural_zero"),
        representative_id=_string(
            item["representative_id"], f"{where}.representative_id"
        ),
        coefficient=_decimal(item["coefficient"], f"{where}.coefficient"),
    )


def _parse_color(value: object, where: str) -> ColorAxis:
    item = _object(value, where)
    kind = _string(item["kind"], f"{where}.kind")
    if kind == "lc-flow":
        return LCColorFlow(
            id=_string(item["id"], f"{where}.id"),
            index=_integer(item["index"], f"{where}.index"),
            word=_integers(item["word"], f"{where}.word"),
            computed=_boolean(item["computed"], f"{where}.computed"),
            representative_id=_string(
                item["representative_id"], f"{where}.representative_id"
            ),
            coefficient=_decimal(item["coefficient"], f"{where}.coefficient"),
        )
    return ContractedColorComponent(
        id=_string(item["id"], f"{where}.id"),
        index=_integer(item["index"], f"{where}.index"),
        description=_string(item["description"], f"{where}.description"),
    )


def _parse_reduction_group(value: object, where: str) -> ReductionGroup:
    item = _object(value, where)
    return ReductionGroup(
        id=_string(item["id"], f"{where}.id"),
        representative_helicity_id=_string(
            item["representative_helicity_id"],
            f"{where}.representative_helicity_id",
        ),
        representative_color_id=_string(
            item["representative_color_id"],
            f"{where}.representative_color_id",
        ),
        physical_helicity_ids=_strings(
            item["physical_helicity_ids"],
            f"{where}.physical_helicity_ids",
        ),
        physical_color_ids=_strings(
            item["physical_color_ids"],
            f"{where}.physical_color_ids",
        ),
    )


def _parse_observation(value: object, where: str) -> Observation:
    item = _object(value, where)
    return Observation(
        point_id=_string(item["point_id"], f"{where}.point_id"),
        arithmetic_precision_bits=_integer(
            item["arithmetic_precision_bits"],
            f"{where}.arithmetic_precision_bits",
        ),
        round_trip_decimal_digits=_integer(
            item["round_trip_decimal_digits"],
            f"{where}.round_trip_decimal_digits",
        ),
        certified_decimal_digits=_integer(
            item["certified_decimal_digits"],
            f"{where}.certified_decimal_digits",
        ),
        values=_decimal_matrix(item["values"], f"{where}.values"),
        total=_decimal(item["total"], f"{where}.total"),
        evidence_refs=_strings(item["evidence_refs"], f"{where}.evidence_refs"),
    )


def _parse_case(value: object, index: int) -> ReferenceCase:
    where = f"cases[{index}]"
    item = _object(value, where)
    axes = _object(item["axes"], f"{where}.axes")
    reduction = _object(item["reduction"], f"{where}.reduction")
    return ReferenceCase(
        id=_string(item["id"], f"{where}.id"),
        case_kind=_string(item["case_kind"], f"{where}.case_kind"),
        model_id=_string(item["model_id"], f"{where}.model_id"),
        process_id=_string(item["process_id"], f"{where}.process_id"),
        color_accuracy=_string(item["color_accuracy"], f"{where}.color_accuracy"),
        point_policy=_string(item["point_policy"], f"{where}.point_policy"),
        point_ids=_strings(item["point_ids"], f"{where}.point_ids"),
        coverage=_parse_coverage(item["coverage"], f"{where}.coverage"),
        selectors=_parse_selectors(item["selectors"], f"{where}.selectors"),
        normalization=_parse_normalization(
            item["normalization"], f"{where}.normalization"
        ),
        topology=_parse_topology(item["topology"], f"{where}.topology"),
        artifact_physics_sha256=_string(
            item["artifact_physics_sha256"], f"{where}.artifact_physics_sha256"
        ),
        artifact_execution_sha256=_string(
            item["artifact_execution_sha256"],
            f"{where}.artifact_execution_sha256",
        ),
        physics_case_sha256=_string(
            item["physics_case_sha256"], f"{where}.physics_case_sha256"
        ),
        helicities=tuple(
            _parse_helicity(entry, f"{where}.axes.helicities[{axis_index}]")
            for axis_index, entry in enumerate(
                _array(axes["helicities"], f"{where}.axes.helicities")
            )
        ),
        colors=tuple(
            _parse_color(entry, f"{where}.axes.colors[{axis_index}]")
            for axis_index, entry in enumerate(
                _array(axes["colors"], f"{where}.axes.colors")
            )
        ),
        reduction=Reduction(
            kind=_string(reduction["kind"], f"{where}.reduction.kind"),
            cell_semantics=_string(
                reduction["cell_semantics"], f"{where}.reduction.cell_semantics"
            ),
            groups=tuple(
                _parse_reduction_group(
                    entry,
                    f"{where}.reduction.groups[{group_index}]",
                )
                for group_index, entry in enumerate(
                    _array(reduction["groups"], f"{where}.reduction.groups")
                )
            ),
            plan_sha256=_string(
                reduction["plan_sha256"],
                f"{where}.reduction.plan_sha256",
            ),
        ),
        observations=tuple(
            _parse_observation(entry, f"{where}.observations[{observation_index}]")
            for observation_index, entry in enumerate(
                _array(item["observations"], f"{where}.observations")
            )
        ),
    )


def _parse_oracle(value: object, where: str) -> OracleProvenance:
    item = _object(value, where)
    tolerance_ceiling = _object(item["tolerance_ceiling"], f"{where}.tolerance_ceiling")
    return OracleProvenance(
        id=_string(item["id"], f"{where}.id"),
        name=_string(item["name"], f"{where}.name"),
        implementation=_string(item["implementation"], f"{where}.implementation"),
        revision=_string(item["revision"], f"{where}.revision"),
        content_sha256=_string(item["content_sha256"], f"{where}.content_sha256"),
        independence_statement=_string(
            item["independence_statement"], f"{where}.independence_statement"
        ),
        validation_profile=_string(
            item["validation_profile"], f"{where}.validation_profile"
        ),
        tolerance_ceiling=Tolerances(
            relative=_decimal(
                tolerance_ceiling["relative"],
                f"{where}.tolerance_ceiling.relative",
            ),
            absolute=_decimal(
                tolerance_ceiling["absolute"],
                f"{where}.tolerance_ceiling.absolute",
            ),
        ),
    )


def _parse_evidence_record(
    value: object,
    *,
    index: int,
    evidence_set_id: str,
    oracle_id: str,
) -> OracleEvidence:
    where = f"evidence {evidence_set_id}.records[{index}]"
    item = _object(value, where)
    identity = _object(item["process_identity"], f"{where}.process_identity")
    tolerances = _object(item["tolerances"], f"{where}.tolerances")
    helicity_totals = item["observed_helicity_totals"]
    observed_values = item["observed_values"]
    return OracleEvidence(
        id=_string(item["id"], f"{where}.id"),
        evidence_set_id=evidence_set_id,
        oracle_id=oracle_id,
        case_id=_string(item["case_id"], f"{where}.case_id"),
        point_id=_string(item["point_id"], f"{where}.point_id"),
        arithmetic_precision_bits=_integer(
            item["arithmetic_precision_bits"],
            f"{where}.arithmetic_precision_bits",
        ),
        round_trip_decimal_digits=_integer(
            item["round_trip_decimal_digits"],
            f"{where}.round_trip_decimal_digits",
        ),
        certified_decimal_digits=_integer(
            item["certified_decimal_digits"],
            f"{where}.certified_decimal_digits",
        ),
        arithmetic=_string(item["arithmetic"], f"{where}.arithmetic"),
        coverage=_string(item["coverage"], f"{where}.coverage"),
        helicity_ids=_strings(item["helicity_ids"], f"{where}.helicity_ids"),
        color_ids=_strings(item["color_ids"], f"{where}.color_ids"),
        observed_total=_decimal(item["observed_total"], f"{where}.observed_total"),
        observed_helicity_totals=(
            None
            if helicity_totals is None
            else _decimals(helicity_totals, f"{where}.observed_helicity_totals")
        ),
        observed_values=(
            None
            if observed_values is None
            else _decimal_matrix(observed_values, f"{where}.observed_values")
        ),
        process_identity=ProcessIdentity(
            expression=_string(
                identity["expression"], f"{where}.process_identity.expression"
            ),
            ordered_external_pdgs=_integers(
                identity["ordered_external_pdgs"],
                f"{where}.process_identity.ordered_external_pdgs",
            ),
            ordered_external_leg_ids=_strings(
                identity["ordered_external_leg_ids"],
                f"{where}.process_identity.ordered_external_leg_ids",
            ),
            source_to_row_permutation=_integers(
                identity["source_to_row_permutation"],
                f"{where}.process_identity.source_to_row_permutation",
            ),
            row_id=_optional_string(
                identity["row_id"], f"{where}.process_identity.row_id"
            ),
            color_order_count=_integer(
                identity["color_order_count"],
                f"{where}.process_identity.color_order_count",
            ),
            ordered_color_legs=_strings(
                identity["ordered_color_legs"],
                f"{where}.process_identity.ordered_color_legs",
            ),
        ),
        input_sha256=_string(item["input_sha256"], f"{where}.input_sha256"),
        physics_case_sha256=_string(
            item["physics_case_sha256"], f"{where}.physics_case_sha256"
        ),
        oracle_output_sha256=_string(
            item["oracle_output_sha256"], f"{where}.oracle_output_sha256"
        ),
        evidence_record_sha256=_string(
            item["evidence_record_sha256"], f"{where}.evidence_record_sha256"
        ),
        command=_strings(item["command"], f"{where}.command"),
        tolerances=Tolerances(
            relative=_decimal(tolerances["relative"], f"{where}.tolerances.relative"),
            absolute=_decimal(tolerances["absolute"], f"{where}.tolerances.absolute"),
        ),
    )


def _parse_evidence_set(value: object, index: int) -> OracleEvidenceSet:
    where = f"evidence_sets[{index}]"
    item = _object(value, where)
    evidence_set_id = _string(item["evidence_set_id"], f"{where}.evidence_set_id")
    oracle = _parse_oracle(item["oracle"], f"{where}.oracle")
    return OracleEvidenceSet(
        id=evidence_set_id,
        captured_at=_string(item["captured_at"], f"{where}.captured_at"),
        oracle=oracle,
        dependency_ids=_strings(item["dependency_ids"], f"{where}.dependency_ids"),
        records=tuple(
            _parse_evidence_record(
                entry,
                index=record_index,
                evidence_set_id=evidence_set_id,
                oracle_id=oracle.id,
            )
            for record_index, entry in enumerate(
                _array(item["records"], f"{where}.records")
            )
        ),
    )
