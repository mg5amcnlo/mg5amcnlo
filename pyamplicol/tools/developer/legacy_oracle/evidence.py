# SPDX-License-Identifier: 0BSD
"""Fixture loading, evidence identities, and semantic replay helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from .checkout import (
    _contributor_lock,
    checkout_branch,
    expected_revision,
)
from .model import ROOT, CompilerProvenance, LegacyOracleError, ProcessEntry


def load_fixture(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _reference_fixture_v2_module() -> Any:
    name = "_pyamplicol_reference_fixture_v2"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = ROOT / "tools" / "developer" / "reference_fixture_v2.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LegacyOracleError(f"cannot import strict fixture-v2 parser from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fortran_evidence_set_id(fixture: Mapping[str, Any]) -> str:
    identifiers = tuple(str(value) for value in fixture["evidence_sets"])
    candidates = tuple(
        identifier
        for identifier in identifiers
        if "fortran" in identifier.lower() or "amplicol" in identifier.lower()
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(identifiers) == 1:
        return identifiers[0]
    raise LegacyOracleError(
        "fixture-v2 must identify one unambiguous Fortran AmpliCol evidence set"
    )


def _fortran_dependency_ids(
    fixture: Mapping[str, Any],
) -> tuple[str, ...]:
    identifiers = tuple(
        str(dependency["id"])
        for dependency in fixture["dependencies"]
        if "legacy" in str(dependency["name"]).lower()
        and "amplicol" in str(dependency["name"]).lower()
    )
    if not identifiers:
        raise LegacyOracleError(
            "fixture-v2 does not contain a legacy AmpliCol dependency"
        )
    if len(identifiers) != 1:
        raise LegacyOracleError(
            "fixture-v2 contains ambiguous legacy AmpliCol dependencies"
        )
    return identifiers


def _oracle_content_sha256(
    *,
    lock: Mapping[str, Any] | None = None,
    revision: str | None = None,
) -> str:
    contributor = _contributor_lock() if lock is None else lock
    pinned_revision = expected_revision() if revision is None else revision
    payload = {
        "branch": checkout_branch(lock=contributor),
        "revision": pinned_revision,
        "source_url": str(contributor["legacy_amplicol"]["source_url"]),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _stable_command_description(
    *,
    process: str,
    case_id: str,
    point_id: str,
    color_accuracy: str,
    entry: ProcessEntry,
    pdg_match_count: int,
    compiler: CompilerProvenance,
    binary64_input_sha256: str,
) -> list[str]:
    return [
        "amplicol_color_probe",
        "build_target=amplicol_color_probe",
        f"compiler_identity={compiler.identity}",
        f"compiler_version={compiler.version}",
        "compiler_flags="
        + json.dumps(list(compiler.flags), ensure_ascii=True, separators=(",", ":")),
        f"compiler_target={compiler.target}",
        f"oracle_executable_sha256={compiler.executable_sha256}",
        f"binary64_input_sha256={binary64_input_sha256}",
        "mode=1",
        f"group={entry.group}",
        f"integral={entry.integral}",
        f"pdg_match_count={pdg_match_count}",
        f"color_accuracy={color_accuracy}",
        f"process={process}",
        f"case={case_id}",
        f"point={point_id}",
        "helicities=summed-then-each-physical",
    ]


def _evidence_id(case_id: str, point_id: str) -> str:
    return f"evidence:fortran:{case_id}:{point_id}"


_COMPILER_VARIANT_COMMAND_KEYS = frozenset(
    {
        "compiler_identity",
        "compiler_version",
        "compiler_flags",
        "compiler_target",
        "oracle_executable_sha256",
    }
)
_REQUIRED_PROVENANCE_COMMAND_KEYS = _COMPILER_VARIANT_COMMAND_KEYS | {
    "binary64_input_sha256"
}


def _semantic_replay_command(command: Sequence[object], *, context: str) -> list[str]:
    seen: dict[str, str] = {}
    semantic: list[str] = []
    for raw_argument in command:
        argument = str(raw_argument)
        key, separator, value = argument.partition("=")
        if key in _REQUIRED_PROVENANCE_COMMAND_KEYS:
            if not separator or not value or key in seen:
                raise LegacyOracleError(
                    f"{context} has invalid or duplicate command provenance {key!r}"
                )
            seen[key] = value
            if key not in _COMPILER_VARIANT_COMMAND_KEYS:
                semantic.append(argument)
        else:
            semantic.append(argument)
    missing = sorted(_REQUIRED_PROVENANCE_COMMAND_KEYS - seen.keys())
    if missing:
        raise LegacyOracleError(f"{context} lacks command provenance fields {missing}")
    for key in ("oracle_executable_sha256", "binary64_input_sha256"):
        if re.fullmatch(r"[a-f0-9]{64}", seen[key]) is None:
            raise LegacyOracleError(f"{context} has invalid {key}")
    try:
        compiler_flags = json.loads(seen["compiler_flags"])
    except json.JSONDecodeError as error:
        raise LegacyOracleError(f"{context} has invalid compiler_flags") from error
    if not isinstance(compiler_flags, list) or any(
        not isinstance(flag, str) or not flag for flag in compiler_flags
    ):
        raise LegacyOracleError(f"{context} has invalid compiler_flags")
    return semantic


def _assert_v2_reports_semantically_equal(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    reference_loader: Any = _reference_fixture_v2_module,
    semantic_replay: Any = _semantic_replay_command,
) -> None:
    reference = reference_loader()
    for label, payload in (("tracked", expected), ("fresh", actual)):
        reference._validate_wire(
            payload,
            "reference-oracle-evidence-v2.schema.json",
            f"{label} Fortran evidence",
        )
        reference._validate_evidence_record_hashes(payload)

    expected_header = dict(expected)
    actual_header = dict(actual)
    expected_records = expected_header.pop("records")
    actual_records = actual_header.pop("records")
    if expected_header != actual_header:
        raise LegacyOracleError(
            "fresh Fortran evidence provenance differs from the tracked record"
        )

    def by_id(records: Sequence[object], label: str) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for raw in records:
            record = cast(Mapping[str, Any], raw)
            identifier = str(record["id"])
            if identifier in result:
                raise LegacyOracleError(
                    f"{label} Fortran evidence repeats record {identifier}"
                )
            result[identifier] = record
        return result

    expected_by_id = by_id(expected_records, "tracked")
    actual_by_id = by_id(actual_records, "fresh")
    if expected_by_id.keys() != actual_by_id.keys():
        raise LegacyOracleError(
            "fresh Fortran evidence record identities differ from the tracked record"
        )

    def compare_numeric(
        expected_value: object,
        actual_value: object,
        *,
        tolerances: Mapping[str, Any],
        context: str,
    ) -> None:
        if expected_value is None or actual_value is None:
            if expected_value is not actual_value:
                raise LegacyOracleError(f"{context} has a different null shape")
            return
        if isinstance(expected_value, list) or isinstance(actual_value, list):
            if (
                not isinstance(expected_value, list)
                or not isinstance(actual_value, list)
                or len(expected_value) != len(actual_value)
            ):
                raise LegacyOracleError(f"{context} has a different numeric shape")
            for index, (expected_item, actual_item) in enumerate(
                zip(expected_value, actual_value, strict=True)
            ):
                compare_numeric(
                    expected_item,
                    actual_item,
                    tolerances=tolerances,
                    context=f"{context}[{index}]",
                )
            return
        expected_decimal = Decimal(str(expected_value))
        actual_decimal = Decimal(str(actual_value))
        if expected_decimal == 0:
            if actual_decimal != 0:
                raise LegacyOracleError(f"{context} no longer preserves an exact zero")
            return
        declared = reference.Tolerances(
            relative=Decimal(str(tolerances["relative"])),
            absolute=Decimal(str(tolerances["absolute"])),
        )
        if not reference._within_tolerance(
            expected_decimal,
            actual_decimal,
            declared,
        ):
            raise LegacyOracleError(
                f"{context} changed from {expected_decimal} to {actual_decimal} "
                "outside the tracked tolerance"
            )

    numeric_fields = (
        "observed_total",
        "observed_helicity_totals",
        "observed_values",
    )
    for identifier in expected_by_id:
        expected_record = dict(expected_by_id[identifier])
        actual_record = dict(actual_by_id[identifier])
        expected_digest = str(expected_record.pop("oracle_output_sha256"))
        actual_digest = str(actual_record.pop("oracle_output_sha256"))
        expected_record.pop("evidence_record_sha256")
        actual_record.pop("evidence_record_sha256")
        if expected_digest == "0" * 64 or actual_digest == "0" * 64:
            raise LegacyOracleError(
                f"Fortran evidence {identifier} contains an invalid output digest"
            )
        expected_values = {
            field: expected_record.pop(field) for field in numeric_fields
        }
        actual_values = {field: actual_record.pop(field) for field in numeric_fields}
        expected_record["command"] = semantic_replay(
            cast(Sequence[object], expected_record["command"]),
            context=f"tracked Fortran evidence {identifier}",
        )
        actual_record["command"] = semantic_replay(
            cast(Sequence[object], actual_record["command"]),
            context=f"fresh Fortran evidence {identifier}",
        )
        if expected_record != actual_record:
            raise LegacyOracleError(
                f"fresh Fortran evidence identity differs for {identifier}"
            )
        tolerances = cast(Mapping[str, Any], expected_record["tolerances"])
        for field_name in numeric_fields:
            compare_numeric(
                expected_values[field_name],
                actual_values[field_name],
                tolerances=tolerances,
                context=f"Fortran evidence {identifier}.{field_name}",
            )
