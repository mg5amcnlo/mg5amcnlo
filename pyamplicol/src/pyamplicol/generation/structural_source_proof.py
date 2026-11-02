# SPDX-License-Identifier: 0BSD
"""Generation-time structural source and physical-lane evidence.

This module deliberately operates before ``artifact.json`` exists. Its output
is provenance rather than runtime state, so callers bind its declared payload
SHA directly when a later numerical witness needs the proof identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pyamplicol.artifacts.security import sha256_file

from .evaluator_container import PacbinReader

SCHEMA = "pyamplicol-generation-structural-source-proof-v1"
ROLE = "structural-source-proof"
SEMANTIC_MAP_DOMAINS = {
    "current_member_map": "current_member_map-v1",
    "interaction_row_map": "interaction_row_map-v1",
    "closure_map": "closure_map-v1",
    "source_contract": "source_contract-v1",
}
# Compiler executables record provenance; they are not members of the artifact payload.
_DIAGNOSTIC_PATH_FIELDS = frozenset({"compiler_path"})


def canonical_sha256(domain: str, value: object) -> str:
    """Hash canonical JSON under a distinct semantic domain."""

    encoded = json.dumps(
        {"domain": domain, "value": value},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def semantic_map_witnesses(execution: Mapping[str, object]) -> dict[str, object]:
    """Project one execution DTO into four non-interchangeable exact maps."""

    predicates = {
        "current_member_map": lambda key: (
            "current" in key
            or key
            in {
                "source_count",
                "source_fill",
                "source_row_count",
                "sources",
                "value_storage",
            }
        ),
        "interaction_row_map": lambda key: (
            "interaction" in key
            or "contribution" in key
            or key in {"stages", "schedule"}
        ),
        "closure_map": lambda key: (
            "closure" in key
            or "amplitude" in key
            or "destination" in key
            or "reduction" in key
        ),
        "source_contract": lambda key: key
        in {
            "color_accuracy",
            "external_particles",
            "external_pdg_order",
            "kind",
            "model",
            "normalization",
            "parameter_layout",
            "process",
            "process_key",
            "runtime_metadata",
            "schema_version",
        },
    }

    def project(predicate: Any) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []

        def visit(value: object, path: str) -> None:
            if isinstance(value, Mapping):
                for raw_key in sorted(value, key=str):
                    key = str(raw_key)
                    child = value[raw_key]
                    child_path = f"{path}.{key}" if path else key
                    if predicate(key):
                        rows.append({"path": child_path, "value": child})
                    else:
                        visit(child, child_path)
            elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")

        visit(execution, "")
        if not rows:
            raise ValueError("execution lacks a required exact semantic-map witness")
        return rows

    result: dict[str, object] = {}
    identities: set[str] = set()
    for name, predicate in predicates.items():
        rows = project(predicate)
        digest = canonical_sha256(SEMANTIC_MAP_DOMAINS[name], rows)
        if digest in identities:
            raise ValueError("semantic maps repeat one content identity")
        identities.add(digest)
        result[name] = {"sha256": digest, "rows": rows}
    return result


def _payload_paths(value: object) -> tuple[str, ...]:
    paths: set[str] = set()

    def visit(item: object, field: str | None = None) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, str(key))
        elif isinstance(item, Sequence) and not isinstance(item, str | bytes):
            for child in item:
                visit(child, field)
        elif (
            isinstance(item, str)
            and field is not None
            and field not in _DIAGNOSTIC_PATH_FIELDS
            and (field == "path" or field.endswith("_path"))
            and item
        ):
            candidate = Path(item)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"execution payload path is not confined: {item!r}")
            paths.add(candidate.as_posix())

    visit(value)
    return tuple(sorted(paths))


def _structural_metrics(value: object) -> list[dict[str, object]]:
    """Retain every explicit structural counter without inferred aliases."""

    rows: list[dict[str, object]] = []

    def visit(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            for raw_key in sorted(item, key=str):
                key = str(raw_key)
                child = item[raw_key]
                child_path = f"{path}.{key}" if path else key
                if (
                    isinstance(child, int)
                    and not isinstance(child, bool)
                    and (
                        key.endswith("_count")
                        or key.endswith("_size")
                        or key.endswith("_size_bytes")
                    )
                ):
                    if child < 0:
                        raise ValueError(
                            f"structural counter {child_path} must be non-negative"
                        )
                    rows.append({"path": child_path, "value": child})
                else:
                    visit(child, child_path)
        elif isinstance(item, Sequence) and not isinstance(item, str | bytes):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    if not rows:
        raise ValueError("execution lane has no explicit structural counters")
    return rows


def execution_lanes(
    execution: Mapping[str, object],
) -> tuple[tuple[str, Mapping[str, object]], ...]:
    """Enumerate each logical execution lane; preserve physical aliases."""

    kind = execution.get("kind")
    if kind != "pyamplicol-runtime-execution":
        return (("primary", execution),)

    lanes: list[tuple[str, Mapping[str, object]]] = []

    def add(record: object, path: str) -> None:
        if not isinstance(record, Mapping):
            raise ValueError(f"compiled execution lane {path} is malformed")
        if not isinstance(record.get("dag_summary"), Mapping):
            raise ValueError(f"compiled execution lane {path} lacks a DAG summary")
        lanes.append((path, record))
        helicity_sum = record.get("helicity_sum_execution")
        if helicity_sum is not None:
            add(helicity_sum, f"{path}.helicity_sum_execution")
        for key in ("color_selector_executions", "helicity_selector_executions"):
            wrappers = record.get(key, ())
            if wrappers is None:
                continue
            if not isinstance(wrappers, Sequence) or isinstance(wrappers, str | bytes):
                raise ValueError(f"compiled execution {path}.{key} is malformed")
            for index, wrapper in enumerate(wrappers):
                if not isinstance(wrapper, Mapping):
                    raise ValueError(
                        f"compiled execution {path}.{key}[{index}] is malformed"
                    )
                add(wrapper.get("execution"), f"{path}.{key}[{index}].execution")

    add(execution, "primary")
    return tuple(lanes)


def _resolve_loose_payload(
    root: Path,
    process_id: str,
    relative: str,
) -> str | None:
    candidates = (relative, f"processes/{process_id}/{relative}")
    for candidate in candidates:
        path = root / candidate
        if path.is_file() and not path.is_symlink():
            return candidate
    return None


def build_generation_structural_proof(
    *,
    artifact_root: Path,
    process_id: str,
    source_revision: str,
    native_build_inputs_sha256: str,
    execution_path: str,
    execution_sha256: str,
    execution: Mapping[str, object],
    evaluator_container_path: str | None,
    evaluator_container_index_sha256: str | None,
) -> dict[str, object]:
    """Build exact source/lane evidence from the private artifact staging tree."""

    lanes = execution_lanes(execution)
    member_records: dict[str, dict[str, object]] = {}
    container_record: dict[str, object] | None = None
    if evaluator_container_path is not None:
        container = artifact_root / evaluator_container_path
        if not container.is_file() or container.is_symlink():
            raise ValueError("evaluator container is absent during structural proof")
        with PacbinReader.open(container, verify_payloads=True) as reader:
            if (
                evaluator_container_index_sha256 is None
                or reader.index.index_sha256 != evaluator_container_index_sha256
            ):
                raise ValueError("evaluator container index identity changed")
            member_records = {
                member.logical_path: {
                    "logical_path": member.logical_path,
                    "kind": member.kind.name.lower(),
                    "length": member.length,
                    "sha256": member.sha256,
                }
                for member in reader.members
            }
            container_record = {
                "path": evaluator_container_path,
                "sha256": sha256_file(container),
                "size_bytes": container.stat().st_size,
                "index_sha256": reader.index.index_sha256,
                "member_count": len(reader.members),
            }

    loose_objects: dict[str, dict[str, object]] = {
        execution_path: {
            "path": execution_path,
            "sha256": execution_sha256,
            "size_bytes": (artifact_root / execution_path).stat().st_size,
        }
    }
    used_members: set[str] = set()
    lane_records: list[dict[str, object]] = []
    roles: list[dict[str, str]] = []
    for lane_id, lane in lanes:
        referenced_loose: set[str] = {execution_path}
        referenced_members: set[str] = set()

        def include(
            relative: str,
            lane_loose: set[str],
            lane_members: set[str],
        ) -> None:
            if relative in member_records:
                lane_members.add(relative)
                used_members.add(relative)
                return
            loose = _resolve_loose_payload(artifact_root, process_id, relative)
            if loose is None:
                # Some manifest paths are logical roots interpreted by the
                # runtime rather than physical files.  Preserve them in the
                # lane source map, but never claim them as persisted objects.
                return
            if loose in lane_loose:
                return
            lane_loose.add(loose)
            path = artifact_root / loose
            loose_objects.setdefault(
                loose,
                {
                    "path": loose,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                },
            )
            if path.suffix == ".json":
                try:
                    nested = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"referenced lane manifest is not valid JSON: {loose}"
                    ) from error
                for nested_relative in _payload_paths(nested):
                    include(nested_relative, lane_loose, lane_members)

        for relative in _payload_paths(lane):
            include(relative, referenced_loose, referenced_members)
        metrics = _structural_metrics(lane)
        subtree_sha = canonical_sha256("execution-lane-subtree-v1", lane)
        metric_sha = canonical_sha256("execution-lane-structural-metrics-v1", metrics)
        lane_records.append(
            {
                "lane_id": lane_id,
                "subtree_sha256": subtree_sha,
                "structural_metrics_sha256": metric_sha,
                "structural_metrics": metrics,
                "loose_payload_paths": sorted(referenced_loose),
                "pacbin_member_paths": sorted(referenced_members),
            }
        )
        roles.extend(
            {
                "lane_id": lane_id,
                "object_kind": "loose-payload",
                "object_path": path,
            }
            for path in sorted(referenced_loose)
        )
        roles.extend(
            {
                "lane_id": lane_id,
                "object_kind": "pacbin-member",
                "object_path": path,
            }
            for path in sorted(referenced_members)
        )

    inventory: dict[str, object] = {
        "status": "complete",
        "loose_payloads": [loose_objects[path] for path in sorted(loose_objects)],
        "pacbin_container": container_record,
        "pacbin_members": [member_records[path] for path in sorted(used_members)],
        "lanes": lane_records,
        "roles": roles,
    }
    inventory["inventory_sha256"] = canonical_sha256(
        "physical-lane-inventory-v1",
        inventory,
    )
    proof: dict[str, object] = {
        "schema": SCHEMA,
        "status": "complete",
        "process_id": process_id,
        "source_identity": {
            "git_revision": source_revision,
            "native_build_inputs_sha256": native_build_inputs_sha256,
        },
        "execution": {
            "path": execution_path,
            "sha256": execution_sha256,
            "kind": execution.get("kind"),
            "color_accuracy": execution.get("color_accuracy"),
        },
        "semantic_maps": semantic_map_witnesses(execution),
        "physical_lane_inventory": inventory,
    }
    proof["proof_content_sha256"] = canonical_sha256(
        "generation-structural-source-proof-v1",
        proof,
    )
    return proof


def validate_generation_structural_proof(
    raw: Mapping[str, object],
    *,
    artifact_root: Path,
    expected_process_id: str,
    expected_source_revision: str,
    expected_native_build_inputs_sha256: str | None = None,
    expected_execution_path: str | None = None,
    expected_execution_sha256: str | None = None,
    execution: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Recompute a generation proof and authenticate its persisted inventory."""

    proof = dict(raw)
    if proof.get("schema") != SCHEMA or proof.get("status") != "complete":
        raise ValueError("generation structural proof schema/status is invalid")
    if proof.get("process_id") != expected_process_id:
        raise ValueError("generation structural proof process identity changed")
    source_identity = proof.get("source_identity")
    if not isinstance(source_identity, Mapping) or (
        source_identity.get("git_revision") != expected_source_revision
    ):
        raise ValueError("generation structural proof source revision changed")
    native_inputs = source_identity.get("native_build_inputs_sha256")
    if (
        not isinstance(native_inputs, str)
        or len(native_inputs) != 64
        or (
            expected_native_build_inputs_sha256 is not None
            and native_inputs != expected_native_build_inputs_sha256
        )
    ):
        raise ValueError("generation structural proof native-input identity changed")
    execution_identity = proof.get("execution")
    if not isinstance(execution_identity, Mapping):
        raise ValueError("generation structural proof execution identity is absent")
    execution_path = execution_identity.get("path")
    execution_sha = execution_identity.get("sha256")
    if (
        not isinstance(execution_path, str)
        or not isinstance(execution_sha, str)
        or (
            expected_execution_path is not None
            and execution_path != expected_execution_path
        )
        or (
            expected_execution_sha256 is not None
            and execution_sha != expected_execution_sha256
        )
    ):
        raise ValueError("generation structural proof execution identity changed")
    execution_file = artifact_root / execution_path
    if (
        not execution_file.is_file()
        or execution_file.is_symlink()
        or sha256_file(execution_file) != execution_sha
    ):
        raise ValueError("generation structural proof execution payload changed")
    if execution is None:
        loaded = json.loads(execution_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("generation structural execution is not an object")
        execution = loaded
    if execution_identity.get("kind") != execution.get("kind") or (
        execution_identity.get("color_accuracy") != execution.get("color_accuracy")
    ):
        raise ValueError("generation structural execution semantics changed")

    maps = proof.get("semantic_maps")
    if not isinstance(maps, Mapping):
        raise ValueError("generation structural semantic maps are absent")
    expected_maps = semantic_map_witnesses(execution)
    if dict(maps) != expected_maps:
        raise ValueError("generation structural semantic maps do not recompute")

    inventory = proof.get("physical_lane_inventory")
    if not isinstance(inventory, Mapping) or inventory.get("status") != "complete":
        raise ValueError("generation structural physical inventory is incomplete")
    inventory_value = dict(inventory)
    declared_inventory_sha = inventory_value.pop("inventory_sha256", None)
    if declared_inventory_sha != canonical_sha256(
        "physical-lane-inventory-v1",
        inventory_value,
    ):
        raise ValueError("generation structural inventory digest changed")

    loose = inventory.get("loose_payloads")
    if not isinstance(loose, list) or not loose:
        raise ValueError("generation structural loose-payload inventory is empty")
    loose_paths: set[str] = set()
    content_ids: set[str] = set()
    for index, item in enumerate(loose):
        if not isinstance(item, Mapping):
            raise ValueError(f"generation structural loose payload {index} is invalid")
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if (
            not isinstance(path, str)
            or path in loose_paths
            or not isinstance(digest, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ValueError(f"generation structural loose payload {index} is invalid")
        loose_paths.add(path)
        concrete = artifact_root / path
        if (
            not concrete.is_file()
            or concrete.is_symlink()
            or concrete.stat().st_size != size
            or sha256_file(concrete) != digest
        ):
            raise ValueError(f"generation structural loose payload {path} changed")
        content_ids.add(digest)

    members = inventory.get("pacbin_members")
    if not isinstance(members, list):
        raise ValueError("generation structural pacbin member inventory is invalid")
    container = inventory.get("pacbin_container")
    actual_members: dict[str, dict[str, object]] = {}
    if container is not None:
        if not isinstance(container, Mapping):
            raise ValueError("generation structural pacbin container is invalid")
        path = container.get("path")
        if not isinstance(path, str):
            raise ValueError("generation structural pacbin container path is invalid")
        concrete = artifact_root / path
        if (
            not concrete.is_file()
            or concrete.is_symlink()
            or concrete.stat().st_size != container.get("size_bytes")
            or sha256_file(concrete) != container.get("sha256")
        ):
            raise ValueError("generation structural pacbin container changed")
        with PacbinReader.open(concrete, verify_payloads=True) as reader:
            if reader.index.index_sha256 != container.get("index_sha256") or len(
                reader.members
            ) != container.get("member_count"):
                raise ValueError("generation structural pacbin index changed")
            actual_members = {
                member.logical_path: {
                    "logical_path": member.logical_path,
                    "kind": member.kind.name.lower(),
                    "length": member.length,
                    "sha256": member.sha256,
                }
                for member in reader.members
            }
    declared_member_paths: set[str] = set()
    for index, item in enumerate(members):
        if not isinstance(item, Mapping):
            raise ValueError(f"generation structural pacbin member {index} is invalid")
        path = item.get("logical_path")
        if not isinstance(path, str) or path in declared_member_paths:
            raise ValueError(f"generation structural pacbin member {index} is invalid")
        declared_member_paths.add(path)
        if actual_members.get(path) != dict(item):
            raise ValueError(f"generation structural pacbin member {path} changed")
        digest = item.get("sha256")
        if isinstance(digest, str):
            content_ids.add(digest)
    if container is None and members:
        raise ValueError("generation structural pacbin members lack a container")

    lane_values = inventory.get("lanes")
    roles = inventory.get("roles")
    if not isinstance(lane_values, list) or not isinstance(roles, list):
        raise ValueError("generation structural lane/role inventory is invalid")
    expected_lanes = execution_lanes(execution)
    if len(lane_values) != len(expected_lanes):
        raise ValueError("generation structural lane count changed")
    lane_ids: set[str] = set()
    for raw_lane, (lane_id, lane) in zip(
        lane_values,
        expected_lanes,
        strict=True,
    ):
        if not isinstance(raw_lane, Mapping) or raw_lane.get("lane_id") != lane_id:
            raise ValueError("generation structural lane identity changed")
        if lane_id in lane_ids:
            raise ValueError("generation structural lane identity is duplicated")
        lane_ids.add(lane_id)
        metrics = _structural_metrics(lane)
        if (
            raw_lane.get("subtree_sha256")
            != canonical_sha256("execution-lane-subtree-v1", lane)
            or raw_lane.get("structural_metrics") != metrics
            or raw_lane.get("structural_metrics_sha256")
            != canonical_sha256("execution-lane-structural-metrics-v1", metrics)
        ):
            raise ValueError("generation structural lane evidence changed")
        loose_refs = raw_lane.get("loose_payload_paths")
        member_refs = raw_lane.get("pacbin_member_paths")
        if (
            not isinstance(loose_refs, list)
            or not set(loose_refs) <= loose_paths
            or not isinstance(member_refs, list)
            or not set(member_refs) <= declared_member_paths
        ):
            raise ValueError("generation structural lane references are incomplete")
    expected_roles = {
        (
            str(lane["lane_id"]),
            "loose-payload",
            str(path),
        )
        for lane in lane_values
        if isinstance(lane, Mapping)
        for path in lane.get("loose_payload_paths", ())
    } | {
        (
            str(lane["lane_id"]),
            "pacbin-member",
            str(path),
        )
        for lane in lane_values
        if isinstance(lane, Mapping)
        for path in lane.get("pacbin_member_paths", ())
    }
    actual_roles = {
        (
            str(role.get("lane_id")),
            str(role.get("object_kind")),
            str(role.get("object_path")),
        )
        for role in roles
        if isinstance(role, Mapping)
    }
    if actual_roles != expected_roles or len(actual_roles) != len(roles):
        raise ValueError("generation structural role coverage is not exact")

    declared_proof_sha = proof.pop("proof_content_sha256", None)
    if declared_proof_sha != canonical_sha256(
        "generation-structural-source-proof-v1",
        proof,
    ):
        raise ValueError("generation structural proof content digest changed")
    return dict(raw)


__all__ = [
    "ROLE",
    "SCHEMA",
    "SEMANTIC_MAP_DOMAINS",
    "build_generation_structural_proof",
    "canonical_sha256",
    "execution_lanes",
    "semantic_map_witnesses",
    "validate_generation_structural_proof",
]
