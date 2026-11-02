# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from tools.developer import recurrence_artifact_compare as compare

_SCHEDULE_DIGEST = "f" * 64
_SCHEDULE_PATH = f"recurrence/schedules/{_SCHEDULE_DIGEST}/recurrence-runtime.pacbin"
_PLAN_MEMBER = "schedule/recurrence-direct-schedule-v2.bin"
_CERTIFICATE_MEMBER = "proof/recurrence-color-projection-v1.bin"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_payload(root: Path, relative: str, data: bytes) -> None:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _rewrite_authenticated_manifest(root: Path, manifest: dict[str, object]) -> None:
    manifest.pop("artifact_id", None)
    manifest["artifact_id"] = compare._runtime_artifact_id(manifest)
    _write_json(root / "artifact.json", manifest)


def _domain_sha256(domain: str, value: object) -> str:
    encoded = json.dumps(
        {"domain": domain, "value": value},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _add_structural_source_proof(root: Path) -> None:
    execution_path = "processes/process/execution.json"
    proof_path = "processes/process/structural-source-proof.json"
    execution = json.loads((root / execution_path).read_text())
    manifest = json.loads((root / "artifact.json").read_text())
    execution_sha = _sha256(root / execution_path)
    execution_size = (root / execution_path).stat().st_size
    metrics = compare._structural_metrics(execution)
    lane = {
        "lane_id": "primary",
        "loose_payload_paths": [execution_path],
        "pacbin_member_paths": [],
        "structural_metrics": metrics,
        "structural_metrics_sha256": _domain_sha256(
            "execution-lane-structural-metrics-v1",
            metrics,
        ),
        "subtree_sha256": _domain_sha256(
            "execution-lane-subtree-v1",
            execution,
        ),
    }
    inventory = {
        "status": "complete",
        "loose_payloads": [
            {
                "path": execution_path,
                "sha256": execution_sha,
                "size_bytes": execution_size,
            }
        ],
        "pacbin_container": None,
        "pacbin_members": [],
        "lanes": [lane],
        "roles": [
            {
                "lane_id": "primary",
                "object_kind": "loose-payload",
                "object_path": execution_path,
            }
        ],
    }
    inventory["inventory_sha256"] = _domain_sha256(
        "physical-lane-inventory-v1",
        inventory,
    )
    producer = manifest["producer"]
    proof = {
        "schema": "pyamplicol-generation-structural-source-proof-v1",
        "status": "complete",
        "process_id": "process",
        "source_identity": {
            "git_revision": producer["git_revision"],
            "native_build_inputs_sha256": producer["native_build_inputs_sha256"],
        },
        "execution": {
            "path": execution_path,
            "sha256": execution_sha,
            "kind": execution["kind"],
            "color_accuracy": execution.get("color_accuracy"),
        },
        "semantic_maps": compare._structural_semantic_map_witnesses(execution),
        "physical_lane_inventory": inventory,
    }
    proof["proof_content_sha256"] = _domain_sha256(
        "generation-structural-source-proof-v1",
        proof,
    )
    _write_json(root / proof_path, proof)
    manifest["payloads"].append(
        {
            "executable": False,
            "media_type": "application/json",
            "path": proof_path,
            "role": "structural-source-proof",
            "sha256": _sha256(root / proof_path),
            "size_bytes": (root / proof_path).stat().st_size,
        }
    )
    manifest["payloads"].sort(key=lambda record: record["path"])
    _rewrite_authenticated_manifest(root, manifest)


def _rewrite_structural_source_proof(root: Path, proof: dict[str, object]) -> None:
    proof_path = root / "processes/process/structural-source-proof.json"
    inventory = proof["physical_lane_inventory"]
    inventory.pop("inventory_sha256", None)
    inventory["inventory_sha256"] = _domain_sha256(
        "physical-lane-inventory-v1",
        inventory,
    )
    proof.pop("proof_content_sha256", None)
    proof["proof_content_sha256"] = _domain_sha256(
        "generation-structural-source-proof-v1",
        proof,
    )
    _write_json(proof_path, proof)
    manifest = json.loads((root / "artifact.json").read_text())
    proof_record = next(
        record
        for record in manifest["payloads"]
        if record["path"].endswith("structural-source-proof.json")
    )
    proof_record["sha256"] = _sha256(proof_path)
    proof_record["size_bytes"] = proof_path.stat().st_size
    _rewrite_authenticated_manifest(root, manifest)


def _projection_body(payload: bytes = b"structural-projection-proof") -> bytes:
    framed = b"PYAMP-COLOR-PROJECTION-BODY-V1\0\0" + struct.pack("<I", 1) + payload
    return framed + hashlib.sha256(framed).digest()


def _pacbin(members: dict[str, tuple[int, bytes]]) -> tuple[bytes, dict[str, object]]:
    header_struct = struct.Struct("<8sHHIIIQQ24s")
    index_header_struct = struct.Struct("<8sHHIQQ")
    index_entry_struct = struct.Struct("<IHHQQ32s")
    footer_struct = struct.Struct("<8sHHIQQ32s")
    content = bytearray(b"\0" * header_struct.size)
    indexed: list[tuple[str, int, int, int, bytes]] = []
    for path in sorted(members, key=lambda item: item.encode("utf-8")):
        kind, payload = members[path]
        content.extend(b"\0" * ((-len(content)) % 64))
        offset = len(content)
        content.extend(payload)
        indexed.append(
            (path, kind, offset, len(payload), hashlib.sha256(payload).digest())
        )
    content.extend(b"\0" * ((-len(content)) % 64))
    index_offset = len(content)
    index = bytearray(
        index_header_struct.pack(b"PACIDX\0\0", 1, 32, 0, len(indexed), 0)
    )
    for path, kind, offset, length, digest in indexed:
        encoded_path = path.encode("utf-8")
        entry = (
            index_entry_struct.pack(
                len(encoded_path),
                kind,
                0,
                offset,
                length,
                digest,
            )
            + encoded_path
        )
        entry += b"\0" * ((-len(entry)) % 8)
        index.extend(entry)
    index_sha256 = hashlib.sha256(index).digest()
    content.extend(index)
    content.extend(
        footer_struct.pack(
            b"PACEND\0\0",
            1,
            64,
            0,
            index_offset,
            len(indexed),
            index_sha256,
        )
    )
    content[: header_struct.size] = header_struct.pack(
        b"PACBIN\0\0",
        1,
        64,
        0,
        64,
        0,
        index_offset,
        len(indexed),
        b"\0" * 24,
    )
    metadata = {
        "index_sha256": index_sha256.hex(),
        "member_count": len(indexed),
        "unpacked_size_bytes": sum(length for _, _, _, length, _ in indexed),
        "members": {
            path: {
                "kind": kind,
                "size_bytes": length,
                "sha256": digest.hex(),
            }
            for path, kind, _, length, digest in indexed
        },
    }
    return bytes(content), metadata


def _artifact(
    root: Path,
    *,
    timing: float = 1.0,
    revision_digit: str = "1",
    native_digit: str = "a",
    version: str = "0.1.0+baseline",
    runtime_payload: bytes = b"runtime-schedule",
    current_count: int = 7,
    model_name: str = "same-model",
    projection_body: bytes | None = None,
    generation_profile_counter: int | None = None,
) -> Path:
    revision = revision_digit * 40
    native = native_digit * 64
    members = {_PLAN_MEMBER: (7, runtime_payload)}
    if projection_body is not None:
        members[_CERTIFICATE_MEMBER] = (8, projection_body)
    runtime_container, runtime_metadata = _pacbin(members)
    _write_payload(root, _SCHEDULE_PATH, runtime_container)

    schedule_index = {
        "binding_count": 1,
        "bindings": [
            {
                "abi": "pyamplicol-recurrence-process-binding-v4",
                "native_schedule_semantic_digest": "d" * 64,
                "path": "recurrence-binding.bin",
                "process_id": "process",
                "process_digest": "f" * 64,
                "process_semantic_digest": "e" * 64,
                "process_support_words": [1],
                "remap": {"bijection_digest": "1" * 64},
                "schedule_digest": _SCHEDULE_DIGEST,
                "sha256": hashlib.sha256(b"binding").hexdigest(),
                "size_bytes": len(b"binding"),
            }
        ],
        "interning_phase": "before-direct-lowering",
        "kind": "pyamplicol-recurrence-schedule-sharing",
        "runtime_ownership": "root-schedule-plus-process-binding",
        "schedule_alias_count": 0,
        "schedule_count": 1,
        "schedules": [
            {
                "digest": _SCHEDULE_DIGEST,
                "index_sha256": runtime_metadata["index_sha256"],
                "member_count": runtime_metadata["member_count"],
                "path": _SCHEDULE_PATH,
                "process_ids": ["process"],
                "sha256": hashlib.sha256(runtime_container).hexdigest(),
                "size_bytes": len(runtime_container),
                "unpacked_size_bytes": runtime_metadata["unpacked_size_bytes"],
            }
        ],
        "schema_version": 3,
    }
    _write_json(root / "recurrence/schedule-index.json", schedule_index)

    inspection_summary: dict[str, object] = {
        "generation_timings_seconds": {
            "direct_lowering": timing,
            "native_total": timing + 0.25,
        },
        "schedule": {"closure_count": 3, "current_count": current_count},
        "semantic_digest": "b" * 64,
    }
    if projection_body is not None:
        certificate_metadata = runtime_metadata["members"]
        assert isinstance(certificate_metadata, dict)
        certificate_record = certificate_metadata[_CERTIFICATE_MEMBER]
        assert isinstance(certificate_record, dict)
        inspection_summary["color_projection_certificate"] = {
            "path": _CERTIFICATE_MEMBER,
            "proof_kind": "exact-rectangular-sum-projection",
            "publishable": True,
            "schema_version": 1,
            "sha256": certificate_record["sha256"],
            "size_bytes": certificate_record["size_bytes"],
        }
    execution_path = "processes/process/execution.json"
    execution = {
        "kind": "pyamplicol-runtime-recurrence-execution",
        "plan": {
            "inspection_summary": inspection_summary,
            "runtime_schedule": {
                "index_sha256": runtime_metadata["index_sha256"],
                "kind": "pyamplicol-recurrence-runtime-container",
                "member_count": runtime_metadata["member_count"],
                "path": _SCHEDULE_PATH,
                "plan_member_path": _PLAN_MEMBER,
                "schema_version": 1,
                "sha256": hashlib.sha256(runtime_container).hexdigest(),
                "size_bytes": len(runtime_container),
                "storage_abi": "pacbin-v1",
                "unpacked_size_bytes": runtime_metadata["unpacked_size_bytes"],
            },
        },
        "process": "d d~ > z g",
        "schema_version": 3,
    }
    _write_json(root / execution_path, execution)

    payload_data = {
        "evaluators.pacbin": b"evaluator-state",
        "processes/process/recurrence-binding.bin": b"binding",
    }
    for relative, data in payload_data.items():
        _write_payload(root, relative, data)

    roles = {
        "evaluators.pacbin": "evaluator-state",
        "recurrence/schedule-index.json": "evaluator-manifest",
        _SCHEDULE_PATH: "evaluator-state",
        "processes/process/recurrence-binding.bin": "evaluator-state",
        execution_path: "evaluator-manifest",
    }
    payloads = []
    for relative in sorted(roles):
        path = root.joinpath(*relative.split("/"))
        payloads.append(
            {
                "executable": False,
                "media_type": "application/octet-stream",
                "path": relative,
                "role": roles[relative],
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )

    manifest = {
        "created_utc": f"2026-07-30T21:00:0{int(timing) % 10}Z",
        "extensions": {
            "generation": {
                "concrete_processes": [
                    {
                        "execution_manifest_sha256": _sha256(root / execution_path),
                        "expression": "d d~ > z g",
                        "id": "process",
                    }
                ],
                "phase_timings_seconds": {
                    "model-loading": timing,
                    "recurrence-construction": timing + 0.5,
                },
            },
            "recurrence_schedule_sharing": {
                "binding_count": 1,
                "index_path": "recurrence/schedule-index.json",
                "index_sha256": _sha256(root / "recurrence/schedule-index.json"),
                "interning_phase": "before-direct-lowering",
                "kind": "pyamplicol-recurrence-schedule-sharing",
                "runtime_ownership": "root-schedule-plus-process-binding",
                "schedule_alias_count": 0,
                "schedule_count": 1,
                "schema_version": 3,
            },
        },
        "kind": "pyamplicol-process",
        "model": {"name": model_name},
        "payloads": payloads,
        "producer": {
            "distribution": "pyamplicol",
            "git_revision": revision,
            "native_build_inputs_sha256": native,
            "version": version,
        },
        "schema_version": 3,
    }
    if generation_profile_counter is not None:
        manifest["extensions"]["generation"]["recurrence_schedule_profiles"] = {
            _SCHEDULE_DIGEST: {
                "native_passes": {
                    "final": {
                        "operation_counters": {
                            name: generation_profile_counter
                            for name in compare._GENERATION_PROFILE_COUNTERS
                        },
                        "schema_version": 1,
                        "scope": "generation-only",
                        "serialized_bytes": {
                            "container": len(runtime_container),
                            "plan_payload": len(runtime_payload),
                            "unpacked_container": runtime_metadata[
                                "unpacked_size_bytes"
                            ],
                        },
                        "timings_seconds": {
                            name: timing for name in compare._GENERATION_PROFILE_TIMINGS
                        },
                    }
                },
                "schema_version": 1,
            }
        }
    manifest["artifact_id"] = compare._runtime_artifact_id(manifest)
    _write_json(root / "artifact.json", manifest)
    return root


def test_exact_artifacts_pass_without_allowed_differences(tmp_path: Path) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(tmp_path / "candidate")

    report = compare.compare_artifacts(baseline, candidate)

    assert report["passes"] is True
    assert report["allowed_metadata_difference_count"] == 0
    assert report["unknown_difference_count"] == 0
    assert report["summary"] == {
        "payload_inventories_match": True,
        "exact_payload_bytes_match": True,
        "payloads_match_policy": True,
        "execution_semantics_match": True,
        "manifest_semantics_match": True,
        "runtime_schedule_plan_bytes_match": True,
        "projection_certificate_semantic_bodies_match": True,
        "runtime_bearing_payloads_match": True,
    }


def test_only_enumerated_timing_and_provenance_differences_pass(
    tmp_path: Path,
) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(
        tmp_path / "candidate",
        timing=2.0,
        revision_digit="2",
        native_digit="c",
        version="0.1.0+candidate",
    )

    report = compare.compare_artifacts(baseline, candidate)

    assert report["passes"] is True
    assert report["unknown_difference_count"] == 0
    allowed_paths = {
        record["json_path"] for record in report["allowed_metadata_differences"]
    }
    assert {
        "/artifact_id",
        "/created_utc",
        "/producer/git_revision",
        "/producer/native_build_inputs_sha256",
        "/producer/version",
        "/extensions/generation/phase_timings_seconds",
        "/plan/inspection_summary/generation_timings_seconds",
    } <= allowed_paths
    assert any("execution_manifest_sha256" in path for path in allowed_paths)
    assert any(
        path.endswith("/sha256") and "execution.json" in path for path in allowed_paths
    )


def test_structural_proof_allows_only_authenticated_execution_derivations(
    tmp_path: Path,
) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(
        tmp_path / "candidate",
        timing=2.0,
        revision_digit="2",
        native_digit="c",
        version="0.1.0+candidate",
    )
    _add_structural_source_proof(baseline)
    _add_structural_source_proof(candidate)

    report = compare.compare_artifacts(baseline, candidate)

    assert report["passes"] is True
    assert report["unknown_difference_count"] == 0
    proof_comparison = next(
        record
        for record in report["payload_comparisons"]
        if record["path"].endswith("structural-source-proof.json")
    )
    assert proof_comparison["byte_for_byte_match"] is False
    assert proof_comparison["normalized_match"] is True


def test_coherently_rehashed_structural_proof_semantics_are_rejected(
    tmp_path: Path,
) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(tmp_path / "candidate")
    _add_structural_source_proof(baseline)
    _add_structural_source_proof(candidate)
    proof_path = candidate / "processes/process/structural-source-proof.json"
    proof = json.loads(proof_path.read_text())
    source_contract = proof["semantic_maps"]["source_contract"]
    source_contract["rows"].append({"path": "forged", "value": True})
    source_contract["sha256"] = _domain_sha256(
        "source_contract-v1",
        source_contract["rows"],
    )
    _rewrite_structural_source_proof(candidate, proof)

    with pytest.raises(compare.ComparisonError, match="semantic maps are stale"):
        compare.compare_artifacts(baseline, candidate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("lane-metrics", "lane evidence is stale"),
        ("duplicate-role", "role coverage is invalid"),
        ("member-without-container", "members lack a container"),
    ],
)
def test_coherently_rehashed_structural_inventory_forgery_is_rejected(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(tmp_path / "candidate")
    _add_structural_source_proof(baseline)
    _add_structural_source_proof(candidate)
    proof_path = candidate / "processes/process/structural-source-proof.json"
    proof = json.loads(proof_path.read_text())
    inventory = proof["physical_lane_inventory"]
    lane = inventory["lanes"][0]
    if mutation == "lane-metrics":
        lane["structural_metrics"].append({"path": "forged_count", "value": 1})
        lane["structural_metrics_sha256"] = _domain_sha256(
            "execution-lane-structural-metrics-v1",
            lane["structural_metrics"],
        )
    elif mutation == "duplicate-role":
        inventory["roles"].append(dict(inventory["roles"][0]))
    else:
        inventory["pacbin_members"].append(
            {
                "logical_path": "forged",
                "kind": "symjit_application",
                "length": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
        )
    _rewrite_structural_source_proof(candidate, proof)

    with pytest.raises(compare.ComparisonError, match=message):
        compare.compare_artifacts(baseline, candidate)


def test_runtime_payload_difference_fails_even_with_authenticated_hash(
    tmp_path: Path,
) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(
        tmp_path / "candidate", runtime_payload=b"changed-runtime-schedule"
    )

    report = compare.compare_artifacts(baseline, candidate)

    assert report["passes"] is False
    assert report["summary"]["exact_payload_bytes_match"] is False
    payload_differences = [
        item
        for item in report["unknown_differences"]
        if item["kind"] == "pacbin-runtime-plan-bytes"
    ]
    assert [item["path"] for item in payload_differences] == [_SCHEDULE_PATH]


def test_unknown_execution_semantic_difference_is_reported(
    tmp_path: Path,
) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(tmp_path / "candidate", current_count=8)

    report = compare.compare_artifacts(baseline, candidate)

    assert report["passes"] is False
    assert report["summary"]["execution_semantics_match"] is False
    assert any(
        item.get("file") == "processes/process/execution.json"
        and item.get("json_pointer")
        == "/plan/inspection_summary/schedule/current_count"
        for item in report["unknown_differences"]
    )


def test_unknown_artifact_manifest_difference_is_reported(
    tmp_path: Path,
) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(tmp_path / "candidate", model_name="different-model")

    report = compare.compare_artifacts(baseline, candidate)

    assert report["passes"] is False
    assert report["summary"]["manifest_semantics_match"] is False
    assert any(
        item.get("file") == "artifact.json"
        and item.get("json_pointer") == "/model/name"
        for item in report["unknown_differences"]
    )


def test_stale_declared_payload_hash_is_rejected(tmp_path: Path) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(tmp_path / "candidate")
    runtime = (
        candidate
        / "recurrence"
        / "schedules"
        / _SCHEDULE_DIGEST
        / "recurrence-runtime.pacbin"
    )
    runtime.write_bytes(b"tampered-after-manifest")

    with pytest.raises(
        compare.ComparisonError, match="payload metadata does not authenticate"
    ):
        compare.compare_artifacts(baseline, candidate)


def test_unlisted_file_is_rejected(tmp_path: Path) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(tmp_path / "candidate")
    (candidate / "unknown.bin").write_bytes(b"unknown")

    with pytest.raises(compare.ComparisonError, match="inventory is not exact"):
        compare.compare_artifacts(baseline, candidate)


def test_projection_certificate_is_independent_of_producer_identity(
    tmp_path: Path,
) -> None:
    body = _projection_body()
    baseline = _artifact(
        tmp_path / "baseline",
        projection_body=body,
        revision_digit="1",
        native_digit="a",
    )
    candidate = _artifact(
        tmp_path / "candidate",
        projection_body=body,
        revision_digit="2",
        native_digit="c",
        version="0.1.0+candidate",
    )

    report = compare.compare_artifacts(baseline, candidate)

    assert report["passes"] is True
    assert report["summary"]["exact_payload_bytes_match"] is True
    assert report["summary"]["payloads_match_policy"] is True
    assert report["summary"]["runtime_schedule_plan_bytes_match"] is True
    assert report["summary"]["projection_certificate_semantic_bodies_match"] is True
    schedule_comparison = next(
        record
        for record in report["payload_comparisons"]
        if record["path"] == _SCHEDULE_PATH
    )
    assert schedule_comparison["byte_for_byte_match"] is True


def test_projection_certificate_structural_body_difference_fails(
    tmp_path: Path,
) -> None:
    baseline = _artifact(
        tmp_path / "baseline",
        projection_body=_projection_body(b"baseline semantics"),
    )
    candidate = _artifact(
        tmp_path / "candidate",
        projection_body=_projection_body(b"candidate semantics"),
    )

    report = compare.compare_artifacts(baseline, candidate)

    assert report["passes"] is False
    assert report["summary"]["projection_certificate_semantic_bodies_match"] is False
    assert any(
        item["kind"] == "projection-certificate-structural-body"
        for item in report["unknown_differences"]
    )


def test_evaluator_kernel_difference_remains_byte_exact_failure(
    tmp_path: Path,
) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(tmp_path / "candidate")
    evaluator = candidate / "evaluators.pacbin"
    evaluator.write_bytes(b"different-evaluator")
    manifest = json.loads((candidate / "artifact.json").read_text())
    for record in manifest["payloads"]:
        if record["path"] == "evaluators.pacbin":
            record["sha256"] = _sha256(evaluator)
            record["size_bytes"] = evaluator.stat().st_size
    _rewrite_authenticated_manifest(candidate, manifest)

    report = compare.compare_artifacts(baseline, candidate)

    assert report["passes"] is False
    assert any(
        item["kind"] == "payload-bytes" and item["path"] == "evaluators.pacbin"
        for item in report["unknown_differences"]
    )


def test_strict_generation_only_profile_is_allowed_and_may_be_candidate_only(
    tmp_path: Path,
) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(
        tmp_path / "candidate",
        generation_profile_counter=17,
    )

    report = compare.compare_artifacts(baseline, candidate)

    assert report["passes"] is True
    assert any(
        record["file"] == "artifact.json"
        and record["json_path"] == "/extensions/generation/recurrence_schedule_profiles"
        and record["category"] == "generation-only-telemetry"
        for record in report["allowed_metadata_differences"]
    )


def test_generation_profile_unknown_field_is_rejected(tmp_path: Path) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(
        tmp_path / "candidate",
        generation_profile_counter=17,
    )
    manifest = json.loads((candidate / "artifact.json").read_text())
    profiles = manifest["extensions"]["generation"]["recurrence_schedule_profiles"]
    profile = profiles[_SCHEDULE_DIGEST]["native_passes"]["final"]
    profile["unknown_runtime_claim"] = 1
    _rewrite_authenticated_manifest(candidate, manifest)

    with pytest.raises(compare.ComparisonError, match="fields are invalid"):
        compare.compare_artifacts(baseline, candidate)


def test_generation_profile_serialized_bytes_must_link_to_schedule(
    tmp_path: Path,
) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(
        tmp_path / "candidate",
        generation_profile_counter=17,
    )
    manifest = json.loads((candidate / "artifact.json").read_text())
    profiles = manifest["extensions"]["generation"]["recurrence_schedule_profiles"]
    serialized = profiles[_SCHEDULE_DIGEST]["native_passes"]["final"][
        "serialized_bytes"
    ]
    serialized["plan_payload"] += 1
    _rewrite_authenticated_manifest(candidate, manifest)

    with pytest.raises(
        compare.ComparisonError,
        match="not linked to its authenticated runtime schedule",
    ):
        compare.compare_artifacts(baseline, candidate)


def test_unknown_schedule_index_difference_is_not_hidden_by_derived_hashes(
    tmp_path: Path,
) -> None:
    baseline = _artifact(tmp_path / "baseline")
    candidate = _artifact(tmp_path / "candidate")
    schedule_index_path = candidate / "recurrence/schedule-index.json"
    schedule_index = json.loads(schedule_index_path.read_text())
    schedule_index["schedules"][0]["process_ids"] = ["different-process"]
    _write_json(schedule_index_path, schedule_index)

    manifest = json.loads((candidate / "artifact.json").read_text())
    schedule_index_sha = _sha256(schedule_index_path)
    manifest["extensions"]["recurrence_schedule_sharing"]["index_sha256"] = (
        schedule_index_sha
    )
    for record in manifest["payloads"]:
        if record["path"] == "recurrence/schedule-index.json":
            record["sha256"] = schedule_index_sha
            record["size_bytes"] = schedule_index_path.stat().st_size
    _rewrite_authenticated_manifest(candidate, manifest)

    report = compare.compare_artifacts(baseline, candidate)

    assert report["passes"] is False
    assert any(
        item.get("file") == "recurrence/schedule-index.json"
        and item.get("json_pointer") == "/schedules/0/process_ids/0"
        for item in report["unknown_differences"]
    )


def test_cli_exit_codes_distinguish_match_diff_and_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _artifact(tmp_path / "baseline")
    matching = _artifact(tmp_path / "matching", timing=2.0)
    different = _artifact(tmp_path / "different", runtime_payload=b"different")
    invalid = _artifact(tmp_path / "invalid")
    (invalid / "unexpected").write_bytes(b"unexpected")

    assert (
        compare.main(["--baseline", str(baseline), "--candidate", str(matching)]) == 0
    )
    assert json.loads(capsys.readouterr().out)["passes"] is True

    assert (
        compare.main(["--baseline", str(baseline), "--candidate", str(different)]) == 1
    )
    assert json.loads(capsys.readouterr().out)["passes"] is False

    assert compare.main(["--baseline", str(baseline), "--candidate", str(invalid)]) == 2
    assert "inventory is not exact" in capsys.readouterr().err
