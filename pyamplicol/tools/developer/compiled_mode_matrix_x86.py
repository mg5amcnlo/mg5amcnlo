#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run and aggregate the frozen x86-64 Direct-Arena matrix in eight shards.

The ordinary :mod:`compiled_mode_matrix` command remains the authoritative
single-host implementation.  This companion exists only because a complete
168-cell run exceeds the six-hour GitHub-hosted job limit.  It partitions
whole artifact groups, validates every artifact tree on the shard host, and
allows a later pure-JSON job to apply the unchanged global matrix gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.developer import compiled_mode_matrix as matrix  # noqa: E402
from tools.developer import compiled_mode_regression as regression  # noqa: E402

SHARD_KIND = "pyamplicol-compiled-mode-matrix-x86-shard"
AGGREGATE_KIND = "pyamplicol-compiled-mode-matrix-x86-aggregate"
SCHEMA_VERSION = 1
SHARD_COUNT = 8
PARTITION_CONTRACT = "arena-matrix-168-artifact-groups-eight-way-v1"
CONTENT_IDENTITY_ALGORITHM = "sha256-canonical-json-body-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_JSON_LIMIT_BYTES = 512 * 1024 * 1024


class ShardError(RuntimeError):
    """Raised when x86 shard evidence cannot be trusted."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _attach_content_identity(body: Mapping[str, object]) -> dict[str, object]:
    payload = dict(body)
    payload["content_identity"] = {
        "algorithm": CONTENT_IDENTITY_ALGORITHM,
        "sha256": _canonical_sha256(body),
    }
    return payload


def _require_content_identity(payload: Mapping[str, object], *, label: str) -> None:
    identity = payload.get("content_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("algorithm") != CONTENT_IDENTITY_ALGORITHM
        or not isinstance(identity.get("sha256"), str)
    ):
        raise ShardError(f"{label} has no valid content identity")
    body = dict(payload)
    body.pop("content_identity")
    if identity["sha256"] != _canonical_sha256(body):
        raise ShardError(f"{label} content identity does not match")


def _reject_constant(value: str) -> None:
    raise ShardError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ShardError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _checked_json(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    try:
        before = path.lstat()
    except OSError as error:
        raise ShardError(f"cannot inspect {label}: {path}") from error
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ShardError(f"{label} is not a regular non-symlink file: {path}")
    if before.st_size <= 0 or before.st_size > _JSON_LIMIT_BYTES:
        raise ShardError(f"{label} has an invalid size: {before.st_size}")
    try:
        encoded = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise ShardError(f"cannot read {label}: {path}") from error
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ShardError(f"{label} changed while it was read: {path}")
    try:
        payload = json.loads(
            encoded,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShardError(f"{label} is not strict JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ShardError(f"{label} is not a JSON object: {path}")
    return payload, {
        "path": str(path.resolve(strict=True)),
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "canonical_sha256": _canonical_sha256(payload),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _artifact_groups() -> tuple[tuple[matrix.MatrixCell, ...], ...]:
    groups: dict[str, list[matrix.MatrixCell]] = {}
    order: list[str] = []
    for cell in matrix.CANONICAL_CELLS:
        if cell.artifact_group_id not in groups:
            groups[cell.artifact_group_id] = []
            order.append(cell.artifact_group_id)
        groups[cell.artifact_group_id].append(cell)
    result = tuple(tuple(groups[group_id]) for group_id in order)
    if len(result) != 56 or any(
        tuple(cell.batch_size for cell in group) != matrix.BATCH_SIZES
        for group in result
    ):
        raise AssertionError("the frozen matrix no longer has 56 batch-sharing groups")
    return result


ARTIFACT_GROUPS = _artifact_groups()


def _medium_process_order() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            group[0].process_key
            for group in ARTIFACT_GROUPS
            if group[0].category == "medium"
        )
    )


def _color_process_order() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            group[0].process_key
            for group in ARTIFACT_GROUPS
            if group[0].category == "color-heavy"
        )
    )


_MEDIUM_PROCESSES = _medium_process_order()
_COLOR_PROCESSES = _color_process_order()
_PRIMARY_GROUP_IDS = tuple(
    group[0].artifact_group_id
    for group in ARTIFACT_GROUPS
    if group[0].category == "primary"
)
_MEDIUM_ROLES = (
    ("eager", "lc-topology"),
    ("eager", "lc-union"),
    ("eager", "nlc-summed"),
    ("eager", "full-summed"),
    ("compiled", "lc-topology"),
    ("compiled", "lc-union"),
    ("compiled", "nlc-summed"),
    ("compiled", "full-summed"),
)
_COLOR_ROLES = (
    ("eager", "nlc-summed"),
    ("eager", "full-summed"),
    ("compiled", "nlc-summed"),
    ("compiled", "full-summed"),
)


def artifact_group_shard(group: Sequence[matrix.MatrixCell]) -> int:
    """Return the frozen load-balanced shard for one whole artifact group."""

    if not group:
        raise ShardError("artifact group cannot be empty")
    first = group[0]
    if any(cell.artifact_group_id != first.artifact_group_id for cell in group):
        raise ShardError("artifact group contains more than one identity")
    if first.category == "primary":
        try:
            return _PRIMARY_GROUP_IDS.index(first.artifact_group_id)
        except ValueError as error:
            raise ShardError("unknown primary artifact group") from error
    if first.category == "medium":
        try:
            process_index = _MEDIUM_PROCESSES.index(first.process_key)
            role_index = _MEDIUM_ROLES.index(
                (first.execution_mode, first.workload_key)
            )
        except ValueError as error:
            raise ShardError("unknown medium artifact-group role") from error
        return (role_index + 3 * process_index) % SHARD_COUNT
    if first.category == "color-heavy":
        try:
            process_index = _COLOR_PROCESSES.index(first.process_key)
            role_index = _COLOR_ROLES.index(
                (first.execution_mode, first.workload_key)
            )
        except ValueError as error:
            raise ShardError("unknown color-heavy artifact-group role") from error
        return (role_index + 4 * process_index + 5) % SHARD_COUNT
    raise ShardError(f"unknown matrix category: {first.category}")


def shard_groups(shard_index: int) -> tuple[tuple[matrix.MatrixCell, ...], ...]:
    if shard_index < 0 or shard_index >= SHARD_COUNT:
        raise ShardError(f"shard index must be in [0,{SHARD_COUNT})")
    groups = tuple(
        group
        for group in ARTIFACT_GROUPS
        if artifact_group_shard(group) == shard_index
    )
    if len(groups) != 7:
        raise AssertionError("each frozen x86 shard must contain seven groups")
    return groups


def shard_cells(shard_index: int) -> tuple[matrix.MatrixCell, ...]:
    cells = tuple(cell for group in shard_groups(shard_index) for cell in group)
    if len(cells) != 21:
        raise AssertionError("each frozen x86 shard must contain 21 cells")
    return cells


def partition_definition() -> dict[str, object]:
    assignments = {
        str(index): {
            "artifact_group_ids": [
                group[0].artifact_group_id for group in shard_groups(index)
            ],
            "cell_ids": [cell.cell_id for cell in shard_cells(index)],
        }
        for index in range(SHARD_COUNT)
    }
    body = {
        "contract": PARTITION_CONTRACT,
        "shard_count": SHARD_COUNT,
        "matrix_contract": matrix.MATRIX_CONTRACT,
        "matrix_definition_sha256": _canonical_sha256(
            [
                {
                    **matrix.asdict(cell),
                    "cell_id": cell.cell_id,
                }
                for cell in matrix.CANONICAL_CELLS
            ]
        ),
        "assignments": assignments,
    }
    return {**body, "sha256": _canonical_sha256(body)}


PARTITION_DEFINITION = partition_definition()


def _result_inventory(
    output_root: Path,
    cells: Sequence[matrix.MatrixCell],
) -> tuple[list[dict[str, object]], list[dict[str, Any]]]:
    files: list[dict[str, object]] = []
    results: list[dict[str, Any]] = []
    for cell in cells:
        path = output_root / "cells" / cell.cell_id / "result.json"
        result, identity = _checked_json(path, label=f"cell result {cell.cell_id}")
        files.append({"cell_id": cell.cell_id, **identity})
        results.append(result)
    return files, results


def _expected_builds(arguments: argparse.Namespace) -> dict[str, dict[str, str]]:
    return {
        "baseline": {
            "source_revision": arguments.expected_baseline_source_revision,
            "native_build_inputs_sha256": (
                arguments.expected_baseline_native_inputs_sha256
            ),
            "distribution_content_sha256": (
                arguments.expected_baseline_distribution_sha256
            ),
            "native_module_sha256": arguments.expected_baseline_native_module_sha256,
        },
        "current": {
            "source_revision": arguments.expected_current_source_revision,
            "native_build_inputs_sha256": (
                arguments.expected_current_native_inputs_sha256
            ),
            "distribution_content_sha256": (
                arguments.expected_current_distribution_sha256
            ),
            "native_module_sha256": arguments.expected_current_native_module_sha256,
        },
    }


def _cell_evidence(
    cells: Sequence[matrix.MatrixCell],
    results: Sequence[Mapping[str, Any]],
    *,
    arguments: argparse.Namespace,
    expected_platform: str,
    expected_builds: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    return [
        matrix._cell_evidence(
            cell,
            result,
            baseline_python=arguments.baseline_python,
            current_python=arguments.current_python,
            ufo_sm_model=arguments.ufo_sm_model,
            output_root=arguments.output_root,
            expected_platform=expected_platform,
            expected_builds=expected_builds,
        )
        for cell, result in zip(cells, results, strict=True)
    ]


def _run_shard(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.shard_count != SHARD_COUNT:
        raise ShardError(
            f"the acceptance contract requires exactly {SHARD_COUNT} shards"
        )
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise ShardError("x86 matrix shards require a Linux x86-64 host")
    output_root = matrix._absolute(arguments.output_root)
    ufo_model = matrix._absolute(arguments.ufo_sm_model)
    output_root.mkdir(parents=True, exist_ok=True)
    expected_builds, preflight = matrix._preflight(
        arguments,
        ufo_sm_model=ufo_model,
    )
    expected_platform = str(preflight["platform"])
    cells = shard_cells(arguments.shard_index)
    expected_ids = {cell.cell_id for cell in cells}
    discovered = matrix._discover_results(output_root)
    unexpected = sorted(set(discovered) - expected_ids)
    if unexpected:
        raise ShardError(f"shard output contains unexpected cell results: {unexpected}")

    regenerated: set[str] = set()
    for ordinal, cell in enumerate(cells, start=1):
        path = output_root / "cells" / cell.cell_id / "result.json"
        if path.is_file() and not arguments.rerun_results:
            existing = matrix._cell_evidence(
                cell,
                matrix._json_object(path),
                baseline_python=arguments.baseline_python,
                current_python=arguments.current_python,
                ufo_sm_model=ufo_model,
                output_root=output_root,
                expected_platform=expected_platform,
                expected_builds=expected_builds,
            )
            if existing["passes"]:
                continue
        command = matrix.cell_command(
            cell,
            baseline_python=arguments.baseline_python,
            current_python=arguments.current_python,
            output_root=output_root,
            ufo_sm_model=ufo_model,
            baseline_dependency_site=arguments.baseline_dependency_site,
            current_dependency_site=arguments.current_dependency_site,
            samples=arguments.samples,
            target_runtime=arguments.target_runtime,
            minimum_samples=arguments.minimum_samples,
            warmup_runs=arguments.warmup_runs,
            generation_timeout=arguments.generation_timeout,
            profile_timeout=arguments.profile_timeout,
            regenerate_artifacts=(
                arguments.regenerate_artifacts
                and cell.artifact_group_id not in regenerated
            ),
        )
        print(
            f"[shard {arguments.shard_index} {ordinal:02d}/{len(cells)}] "
            f"{cell.cell_id}",
            file=sys.stderr,
            flush=True,
        )
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode not in {0, 1} or not path.is_file():
            raise ShardError(
                f"cell driver failed for {cell.cell_id} "
                f"(exit {completed.returncode}): {completed.stderr.strip()}"
            )
        if arguments.regenerate_artifacts:
            regenerated.add(cell.artifact_group_id)

    files, results = _result_inventory(output_root, cells)
    evidence = _cell_evidence(
        cells,
        results,
        arguments=arguments,
        expected_platform=expected_platform,
        expected_builds=expected_builds,
    )
    cell_failures = {
        record["cell_id"]: record["errors"] for record in evidence if record["errors"]
    }
    indexed_results = {
        cell.cell_id: result for cell, result in zip(cells, results, strict=True)
    }
    artifact_errors = matrix._artifact_postflight_errors(
        indexed_results,
        output_root=output_root,
    )
    postflight = matrix._acceptance_state(
        baseline_python=arguments.baseline_python,
        current_python=arguments.current_python,
        baseline_dependency_site=arguments.baseline_dependency_site,
        current_dependency_site=arguments.current_dependency_site,
        ufo_sm_model=ufo_model,
    )
    if postflight != preflight:
        artifact_errors.append(
            "acceptance host/source/interpreter/dependency/model state changed "
            "between shard preflight and postflight"
        )
    complete = len(results) == len(cells)
    passes = complete and not cell_failures and not artifact_errors
    payload = _attach_content_identity(
        {
            "kind": SHARD_KIND,
            "schema_version": SCHEMA_VERSION,
            "matrix_contract": matrix.MATRIX_CONTRACT,
            "partition": PARTITION_DEFINITION,
            "shard_count": SHARD_COUNT,
            "shard_index": arguments.shard_index,
            "workflow_run_id": arguments.workflow_run_id,
            "runtime_bundle_sha256": arguments.runtime_bundle_sha256,
            "output_root": str(output_root),
            "expected_builds": expected_builds,
            "platform": {
                "platform": expected_platform,
                "system": preflight["system"],
                "machine": preflight["machine"],
            },
            "selected_artifact_group_ids": [
                group[0].artifact_group_id
                for group in shard_groups(arguments.shard_index)
            ],
            "selected_cell_ids": [cell.cell_id for cell in cells],
            "result_files": files,
            "cell_evidence": evidence,
            "cell_gate": {
                "failures": cell_failures,
                "passes": not cell_failures and complete,
            },
            "artifact_postflight_gate": {
                "errors": artifact_errors,
                "passes": not artifact_errors and complete,
            },
            "provenance": {
                "preflight": preflight,
                "postflight": postflight,
                "all_match": preflight == postflight,
                "shard_driver": regression._path_identity(Path(__file__)),
            },
            "complete": complete,
            "passes": passes,
        }
    )
    destination = (
        Path(arguments.shard_result)
        if arguments.shard_result is not None
        else output_root / "shards" / f"shard-{arguments.shard_index}.json"
    )
    _write_json_atomic(destination, payload)
    return payload


def _require_sha(value: object, *, label: str, git: bool = False) -> str:
    pattern = _GIT_SHA if git else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ShardError(f"{label} is not a valid {'Git ' if git else ''}SHA")
    return value


def _aggregate(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.shard_count != SHARD_COUNT:
        raise ShardError(
            f"the acceptance contract requires exactly {SHARD_COUNT} shards"
        )
    repository = matrix._repository_identity()
    if (
        repository.get("clean") is not True
        or repository.get("head_revision") != arguments.expected_current_source_revision
    ):
        raise ShardError("aggregate requires the exact clean current source revision")
    output_root = matrix._absolute(arguments.output_root)
    shard_root = matrix._absolute(arguments.shard_root)
    expected_builds = _expected_builds(arguments)
    result_map: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    manifest_files: list[dict[str, object]] = []
    errors: list[str] = []
    observed_manifest_paths = sorted(shard_root.glob("shard-*.json"))
    expected_names = {f"shard-{index}.json" for index in range(SHARD_COUNT)}
    observed_names = {path.name for path in observed_manifest_paths}
    if observed_names != expected_names:
        raise ShardError(
            "aggregate shard manifest set is incomplete or unexpected: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unexpected={sorted(observed_names - expected_names)}"
        )
    for index in range(SHARD_COUNT):
        path = shard_root / f"shard-{index}.json"
        shard, file_identity = _checked_json(path, label=f"shard {index} manifest")
        _require_content_identity(shard, label=f"shard {index} manifest")
        manifest_files.append({"shard_index": index, **file_identity})
        if (
            shard.get("kind") != SHARD_KIND
            or shard.get("schema_version") != SCHEMA_VERSION
            or shard.get("matrix_contract") != matrix.MATRIX_CONTRACT
            or _canonical_sha256(shard.get("partition"))
            != _canonical_sha256(PARTITION_DEFINITION)
            or shard.get("shard_count") != SHARD_COUNT
            or shard.get("shard_index") != index
            or shard.get("workflow_run_id") != arguments.workflow_run_id
            or shard.get("runtime_bundle_sha256")
            != arguments.runtime_bundle_sha256
            or shard.get("output_root") != str(output_root)
            or shard.get("expected_builds") != expected_builds
            or shard.get("complete") is not True
            or shard.get("passes") is not True
        ):
            errors.append(f"shard {index} top-level contract is invalid")
        platform_record = shard.get("platform")
        if (
            not isinstance(platform_record, Mapping)
            or platform_record.get("system") != "Linux"
            or platform_record.get("machine") not in {"x86_64", "AMD64"}
            or not isinstance(platform_record.get("platform"), str)
        ):
            errors.append(f"shard {index} is not authenticated Linux x86-64")
            expected_platform = ""
        else:
            expected_platform = str(platform_record["platform"])
        expected_cells = shard_cells(index)
        if shard.get("selected_cell_ids") != [
            cell.cell_id for cell in expected_cells
        ] or shard.get("selected_artifact_group_ids") != [
            group[0].artifact_group_id for group in shard_groups(index)
        ]:
            errors.append(f"shard {index} selected-cell partition is invalid")
        artifact_gate = shard.get("artifact_postflight_gate")
        provenance = shard.get("provenance")
        if (
            not isinstance(artifact_gate, Mapping)
            or artifact_gate.get("passes") is not True
            or artifact_gate.get("errors") != []
            or not isinstance(provenance, Mapping)
            or provenance.get("all_match") is not True
            or provenance.get("preflight") != provenance.get("postflight")
        ):
            errors.append(f"shard {index} host-local postflight is invalid")
        raw_files = shard.get("result_files")
        raw_evidence = shard.get("cell_evidence")
        if (
            not isinstance(raw_files, list)
            or not isinstance(raw_evidence, list)
            or len(raw_files) != len(expected_cells)
            or len(raw_evidence) != len(expected_cells)
        ):
            errors.append(f"shard {index} result inventory is incomplete")
            continue
        for cell, recorded_file, recorded_evidence in zip(
            expected_cells,
            raw_files,
            raw_evidence,
            strict=True,
        ):
            if not isinstance(recorded_file, Mapping):
                errors.append(f"shard {index} has an invalid result-file record")
                continue
            result_path = output_root / "cells" / cell.cell_id / "result.json"
            result, observed_file = _checked_json(
                result_path,
                label=f"aggregate cell result {cell.cell_id}",
            )
            expected_file = {"cell_id": cell.cell_id, **observed_file}
            if dict(recorded_file) != expected_file:
                errors.append(f"cell {cell.cell_id} result file identity changed")
            if cell.cell_id in result_map:
                errors.append(f"cell {cell.cell_id} appears in multiple shards")
            result_map[cell.cell_id] = result
            observed_evidence = matrix._cell_evidence(
                cell,
                result,
                baseline_python=arguments.baseline_python,
                current_python=arguments.current_python,
                ufo_sm_model=arguments.ufo_sm_model,
                output_root=output_root,
                expected_platform=expected_platform,
                expected_builds=expected_builds,
            )
            if _canonical_sha256(recorded_evidence) != _canonical_sha256(
                observed_evidence
            ):
                errors.append(f"cell {cell.cell_id} semantic evidence changed")
            evidence.append(observed_evidence)

    expected_ids = {cell.cell_id for cell in matrix.CANONICAL_CELLS}
    observed_ids = set(result_map)
    global_audit = matrix.audit_cell_evidence(
        evidence,
        missing=sorted(expected_ids - observed_ids),
        unexpected=sorted(observed_ids - expected_ids),
        expected_builds=expected_builds,
    )
    shard_gate_passes = not errors and len(evidence) == len(matrix.CANONICAL_CELLS)
    passes = shard_gate_passes and global_audit.get("passes") is True
    payload = _attach_content_identity(
        {
            "kind": AGGREGATE_KIND,
            "schema_version": SCHEMA_VERSION,
            "matrix_contract": matrix.MATRIX_CONTRACT,
            "partition": PARTITION_DEFINITION,
            "workflow_run_id": arguments.workflow_run_id,
            "runtime_bundle_sha256": arguments.runtime_bundle_sha256,
            "expected_current_source_revision": (
                arguments.expected_current_source_revision
            ),
            "expected_builds": expected_builds,
            "repository": repository,
            "output_root": str(output_root),
            "shard_manifests": manifest_files,
            "shard_gate": {
                "errors": errors,
                "passes": shard_gate_passes,
            },
            "matrix_audit": global_audit,
            "complete": global_audit.get("complete") is True,
            "passes": passes,
        }
    )
    _write_json_atomic(arguments.aggregate_result, payload)
    return payload


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _sha256_argument(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256")
    return value


def _git_sha_argument(value: str) -> str:
    if _GIT_SHA.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase Git SHA")
    return value


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--current-python", type=Path, required=True)
    parser.add_argument("--baseline-dependency-site", type=Path, required=True)
    parser.add_argument("--current-dependency-site", type=Path, required=True)
    parser.add_argument("--ufo-sm-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--expected-baseline-source-revision",
        type=_git_sha_argument,
        required=True,
    )
    parser.add_argument(
        "--expected-current-source-revision",
        type=_git_sha_argument,
        required=True,
    )
    parser.add_argument(
        "--expected-baseline-native-inputs-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument(
        "--expected-current-native-inputs-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument(
        "--expected-baseline-distribution-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument(
        "--expected-current-distribution-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument(
        "--expected-baseline-native-module-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument(
        "--expected-current-native-module-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument(
        "--runtime-bundle-sha256",
        type=_sha256_argument,
        required=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("shard")
    _add_identity_arguments(shard)
    shard.add_argument("--shard-count", type=_positive_int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-result", type=Path)
    shard.add_argument("--samples", type=_positive_int, default=7)
    shard.add_argument("--target-runtime", type=float, default=5.0)
    shard.add_argument("--minimum-samples", type=_positive_int, default=7)
    shard.add_argument("--warmup-runs", type=_positive_int, default=2)
    shard.add_argument("--generation-timeout", type=float, default=2400.0)
    shard.add_argument("--profile-timeout", type=float, default=1200.0)
    shard.add_argument("--rerun-results", action="store_true")
    shard.add_argument("--regenerate-artifacts", action="store_true")

    aggregate = commands.add_parser("aggregate")
    _add_identity_arguments(aggregate)
    aggregate.add_argument("--shard-count", type=_positive_int, required=True)
    aggregate.add_argument("--shard-root", type=Path, required=True)
    aggregate.add_argument("--aggregate-result", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        payload = (
            _run_shard(arguments)
            if arguments.command == "shard"
            else _aggregate(arguments)
        )
    except (ShardError, matrix.MatrixError, regression.RegressionError) as error:
        print(f"compiled-mode-matrix-x86: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0 if payload.get("passes") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
