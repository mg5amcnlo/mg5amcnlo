#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Fail-closed comparison of baseline and candidate recurrence artifacts.

This is a developer tool, not a public pyAmpliCol CLI command.  Runtime plan,
projection certificate, binding, evaluator, and model bytes must be exact.
Only explicitly permitted metadata differences are normalized.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import re
import struct
import sys
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

COMPARISON_KIND = "pyamplicol-recurrence-artifact-exact-comparison"
COMPARISON_SCHEMA_VERSION = 1
MAX_REPORTED_UNKNOWN_DIFFERENCES = 256

ARTIFACT_ALLOWED_METADATA_PATHS = (
    "/artifact_id",
    "/created_utc",
    "/producer/git_revision",
    "/producer/native_build_inputs_sha256",
    "/producer/version",
    "/extensions/generation/phase_timings_seconds",
    "/extensions/generation/recurrence_schedule_profiles",
    "/extensions/generation/concrete_processes/*/execution_manifest_sha256",
    "/payloads/*[execution.json]/sha256",
    "/payloads/*[execution.json]/size_bytes",
    "/payloads/*[structural-source-proof.json]/sha256",
    "/payloads/*[structural-source-proof.json]/size_bytes",
)
EXECUTION_ALLOWED_METADATA_PATHS = (
    "/plan/inspection_summary/generation_timings_seconds",
)
STRUCTURAL_PROOF_ALLOWED_METADATA_PATHS = (
    "/source_identity/git_revision",
    "/source_identity/native_build_inputs_sha256",
    "/execution/sha256",
    "/physical_lane_inventory/loose_payloads[path=processes/*/execution.json]/sha256",
    "/physical_lane_inventory/loose_payloads[path=processes/*/execution.json]/size_bytes",
    "/physical_lane_inventory/lanes[id=primary]/subtree_sha256",
    "/physical_lane_inventory/inventory_sha256",
    "/proof_content_sha256",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_RUNTIME_ARTIFACT_ID_PAYLOAD_ROLES = frozenset(
    {
        "compiled-model",
        "evaluator-manifest",
        "evaluator-state",
        "model-parameters",
        "runtime-physics",
    }
)
_EXECUTION_PATH_RE = re.compile(r"processes/([^/]+)/execution[.]json")
_STRUCTURAL_PROOF_PATH_RE = re.compile(
    r"processes/([^/]+)/structural-source-proof[.]json"
)
_BINDING_PATH_RE = re.compile(r"processes/([^/]+)/recurrence-binding[.]bin")
_RUNTIME_SCHEDULE_RE = re.compile(
    r"recurrence/schedules/([0-9a-f]{64})/recurrence-runtime[.]pacbin"
)
_SCHEDULE_INDEX_PATH = "recurrence/schedule-index.json"
_PLAN_MEMBER_PATH = "schedule/recurrence-direct-schedule-v2.bin"
_CERTIFICATE_MEMBER_PATH = "proof/recurrence-color-projection-v1.bin"
_PLAN_MEMBER_KIND = 7
_CERTIFICATE_MEMBER_KIND = 8
_PACBIN_MEMBER_KIND_NAMES = {
    1: "symjit_application",
    2: "symbolica_exact_state",
    3: "native_library",
    4: "eager_runtime_metadata",
    5: "eager_runtime_table",
    7: "recurrence_direct_plan",
    8: "recurrence_color_projection_certificate",
}
_STRUCTURAL_SEMANTIC_MAP_DOMAINS = {
    "current_member_map": "current_member_map-v1",
    "interaction_row_map": "interaction_row_map-v1",
    "closure_map": "closure_map-v1",
    "source_contract": "source_contract-v1",
}

_PACBIN_VERSION = 1
_PACBIN_ALIGNMENT = 64
_PACBIN_INDEX_ALIGNMENT = 8
_PACBIN_HEADER = struct.Struct("<8sHHIIIQQ24s")
_PACBIN_INDEX_HEADER = struct.Struct("<8sHHIQQ")
_PACBIN_INDEX_ENTRY = struct.Struct("<IHHQQ32s")
_PACBIN_FOOTER = struct.Struct("<8sHHIQQ32s")
_PACBIN_HEADER_MAGIC = b"PACBIN\x00\x00"
_PACBIN_INDEX_MAGIC = b"PACIDX\x00\x00"
_PACBIN_FOOTER_MAGIC = b"PACEND\x00\x00"

_CERTIFICATE_BODY_MAGIC = b"PYAMP-COLOR-PROJECTION-BODY-V1\0\0"
_GENERATION_PROFILE_TIMINGS = frozenset(
    {
        "transition-catalog",
        "structural-feasibility",
        "color-target-index",
        "structural-demand",
        "support-indexing",
        "candidate-processing",
        "closure-processing",
        "canonical-emission",
        "python-extraction",
        "catalog-authentication",
        "semantic-construction-total",
        "direct-lowering",
        "serialization",
        "native-total",
    }
)
_GENERATION_PROFILE_COUNTERS = frozenset(
    {
        "support_bucket_count",
        "support_bucket_probe_count",
        "support_bucket_cache_hit_count",
        "support_bucket_cache_miss_count",
        "candidate_parent_pair_theoretical_count",
        "candidate_parent_pair_visited_count",
        "structural_feasible_support_count",
        "structural_decomposition_count",
        "structural_forward_transition_probe_count",
        "structural_demand_support_count",
        "structural_demand_state_count",
        "structural_reject_count",
        "transition_index_hit_count",
        "transition_index_miss_count",
        "transition_candidate_count",
        "quantum_match_count",
        "coupling_match_count",
        "transition_accept_count",
        "color_shape_match_count",
        "color_result_count",
        "color_target_accept_count",
        "color_target_reject_count",
        "color_acceptance_cache_hit_count",
        "color_acceptance_cache_miss_count",
        "color_fragment_bucket_count",
        "color_fragment_hash_lookup_count",
        "color_posting_incidence_count",
        "color_sparse_posting_bucket_count",
        "color_dense_posting_bucket_count",
        "color_sparse_posting_bytes",
        "color_dense_posting_bytes",
        "accepted_parent_key_clone_count",
        "current_key_lookup_count",
        "current_key_hit_count",
        "current_insert_count",
        "current_key_clone_count",
        "indexed_hash_lookup_count",
        "contribution_attempt_count",
        "contribution_insert_count",
        "contribution_merge_count",
        "closure_candidate_theoretical_count",
        "closure_candidate_count",
        "closure_support_lookup_count",
        "closure_state_match_count",
        "closure_color_attempt_count",
        "closure_group_count",
        "closure_proof_contribution_count",
        "constructed_current_count",
        "constructed_contribution_count",
        "constructed_interaction_count",
        "constructed_dynamic_color_state_count",
        "emitted_current_count",
        "emitted_contribution_count",
        "emitted_interaction_count",
        "emitted_finalization_count",
        "emitted_closure_count",
    }
)
_GENERATION_PROFILE_SERIALIZED_BYTES = frozenset(
    {"plan_payload", "container", "unpacked_container"}
)
_NORMALIZED = "<allowed-metadata-difference>"
_MISSING = object()


class ComparisonError(RuntimeError):
    """Raised when either comparison input violates the artifact contract."""


@dataclass(frozen=True)
class PayloadSnapshot:
    """Authenticated payload metadata without retaining payload bytes."""

    relative_path: str
    path: Path
    role: str
    size_bytes: int
    sha256: str
    record: dict[str, Any]


@dataclass(frozen=True)
class PacbinMemberSnapshot:
    """One authenticated member in a canonical recurrence PACBIN."""

    logical_path: str
    kind: int
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class ProjectionCertificateSnapshot:
    """Authenticated byte range for one structural projection certificate."""

    body_offset: int
    body_length: int
    sha256: str


@dataclass(frozen=True)
class RecurrencePacbinSnapshot:
    """Authenticated direct-plan container and optional projection proof."""

    path: Path
    file_size: int
    index_sha256: str
    unpacked_size_bytes: int
    members: dict[str, PacbinMemberSnapshot]
    projection_certificate: ProjectionCertificateSnapshot | None


@dataclass(frozen=True)
class RecurrencePacbinComparison:
    """Exact and policy-normalized comparison result for one runtime container."""

    exact_bytes_match: bool
    policy_match: bool
    plan_bytes_match: bool
    projection_certificate_bodies_match: bool


@dataclass(frozen=True)
class ArtifactSnapshot:
    """One independently authenticated recurrence artifact."""

    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    payloads: dict[str, PayloadSnapshot]
    executions: dict[str, dict[str, Any]]
    schedule_index: dict[str, Any]
    recurrence_schedules: dict[str, RecurrencePacbinSnapshot]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _runtime_artifact_id(manifest: Mapping[str, Any]) -> str:
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise ComparisonError("artifact payload inventory is invalid")
    records = [
        dict(payload)
        for payload in payloads
        if isinstance(payload, Mapping)
        and payload.get("role") in _RUNTIME_ARTIFACT_ID_PAYLOAD_ROLES
    ]
    records.sort(key=lambda payload: str(payload.get("path", "")))
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "kind": "pyamplicol-runtime-payload-identity",
                "schema_version": 1,
                "payloads": records,
            }
        )
    ).hexdigest()


def _json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ComparisonError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise ComparisonError(f"{description} must be a JSON object: {path}")
    if raw != _canonical_json_bytes(value):
        raise ComparisonError(f"{description} is not canonical compact JSON: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ComparisonError(f"cannot hash artifact file: {path}") from error
    return digest.hexdigest()


def _files_equal(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as left_stream, right.open("rb") as right_stream:
            while True:
                left_block = left_stream.read(1024 * 1024)
                right_block = right_stream.read(1024 * 1024)
                if left_block != right_block:
                    return False
                if not left_block:
                    return True
    except OSError as error:
        raise ComparisonError(
            f"cannot compare payload bytes: {left} and {right}"
        ) from error


def _read_exact(
    stream: BinaryIO,
    length: int,
    *,
    description: str,
) -> bytes:
    if length < 0:
        raise ComparisonError(f"{description} has a negative read length")
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ComparisonError(f"{description} is truncated")
        if not isinstance(chunk, bytes | bytearray | memoryview):
            raise ComparisonError(f"{description} did not return bytes")
        if len(chunk) > remaining:
            raise ComparisonError(f"{description} returned an oversized read")
        chunks.append(bytes(chunk))
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_exact_at(
    stream: BinaryIO,
    offset: int,
    length: int,
    *,
    description: str,
) -> bytes:
    try:
        stream.seek(offset)
    except (OSError, ValueError) as error:
        raise ComparisonError(f"cannot seek while reading {description}") from error
    return _read_exact(stream, length, description=description)


def _hash_stream_range(
    stream: BinaryIO,
    *,
    offset: int,
    length: int,
    description: str,
) -> str:
    digest = hashlib.sha256()
    try:
        stream.seek(offset)
        remaining = length
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise ComparisonError(f"{description} is truncated")
            if not isinstance(block, bytes | bytearray | memoryview):
                raise ComparisonError(f"{description} did not return bytes")
            if len(block) > remaining:
                raise ComparisonError(f"{description} returned an oversized read")
            digest.update(block)
            remaining -= len(block)
    except OSError as error:
        raise ComparisonError(f"cannot hash {description}") from error
    return digest.hexdigest()


def _file_ranges_equal(
    left: Path,
    right: Path,
    *,
    left_offset: int,
    right_offset: int,
    length: int,
) -> bool:
    try:
        with left.open("rb") as left_stream, right.open("rb") as right_stream:
            left_stream.seek(left_offset)
            right_stream.seek(right_offset)
            remaining = length
            while remaining:
                requested = min(1024 * 1024, remaining)
                left_block = _read_exact(
                    left_stream,
                    requested,
                    description=f"PACBIN member range in {left}",
                )
                right_block = _read_exact(
                    right_stream,
                    requested,
                    description=f"PACBIN member range in {right}",
                )
                if left_block != right_block:
                    return False
                remaining -= requested
    except OSError as error:
        raise ComparisonError(
            f"cannot compare PACBIN member ranges: {left} and {right}"
        ) from error
    return True


def _sha256_file_range(path: Path, *, offset: int, length: int) -> str:
    try:
        with path.open("rb") as stream:
            return _hash_stream_range(
                stream,
                offset=offset,
                length=length,
                description=f"file range in {path}",
            )
    except OSError as error:
        raise ComparisonError(f"cannot hash file range in {path}") from error


def _alignment_padding(position: int, alignment: int) -> int:
    return (-position) % alignment


def _canonical_pacbin_member_path(value: bytes) -> str:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ComparisonError("recurrence PACBIN member path is not UTF-8") from error
    if not text or "\\" in text or text.startswith("/") or "\x00" in text:
        raise ComparisonError("recurrence PACBIN member path is not canonical")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ComparisonError("recurrence PACBIN member path is not canonical")
    if "/".join(unicodedata.normalize("NFC", part) for part in parts) != text:
        raise ComparisonError("recurrence PACBIN member path is not NFC")
    return text


def _validate_pacbin_version_size_flags(
    *,
    description: str,
    version: int,
    encoded_size: int,
    expected_size: int,
    flags: int,
) -> None:
    if version != _PACBIN_VERSION or encoded_size != expected_size or flags != 0:
        raise ComparisonError(
            f"recurrence PACBIN has incompatible {description} framing"
        )


def _parse_projection_certificate(
    stream: BinaryIO,
    *,
    member: PacbinMemberSnapshot,
    description: str,
) -> ProjectionCertificateSnapshot:
    minimum_size = len(_CERTIFICATE_BODY_MAGIC) + 4 + 32
    if member.length < minimum_size:
        raise ComparisonError(f"{description} is too short")
    member_end = member.offset + member.length
    prefix = _read_exact_at(
        stream,
        member.offset,
        len(_CERTIFICATE_BODY_MAGIC) + 4,
        description=f"{description} prefix",
    )
    if not prefix.startswith(_CERTIFICATE_BODY_MAGIC) or prefix[
        len(_CERTIFICATE_BODY_MAGIC) :
    ] != (1).to_bytes(4, "little"):
        raise ComparisonError(f"{description} has invalid framing")
    expected_body_digest = _read_exact_at(
        stream,
        member_end - 32,
        32,
        description=f"{description} structural-body digest",
    ).hex()
    actual_body_digest = _hash_stream_range(
        stream,
        offset=member.offset,
        length=member.length - 32,
        description=f"{description} structural body",
    )
    if actual_body_digest != expected_body_digest:
        raise ComparisonError(f"{description} structural-body digest mismatch")
    return ProjectionCertificateSnapshot(
        body_offset=member.offset,
        body_length=member.length,
        sha256=member.sha256,
    )


def _load_recurrence_pacbin(
    path: Path,
    *,
    description: str,
    recurrence_contract: bool = True,
) -> RecurrencePacbinSnapshot:
    try:
        file_size = path.stat().st_size
        stream = path.open("rb")
    except OSError as error:
        raise ComparisonError(f"cannot open {description}: {path}") from error
    with stream:
        minimum_size = (
            _PACBIN_HEADER.size + _PACBIN_INDEX_HEADER.size + _PACBIN_FOOTER.size
        )
        if file_size < minimum_size:
            raise ComparisonError(f"{description} is truncated")
        header = _PACBIN_HEADER.unpack(
            _read_exact_at(
                stream,
                0,
                _PACBIN_HEADER.size,
                description=f"{description} header",
            )
        )
        (
            magic,
            version,
            header_size,
            flags,
            alignment,
            reserved,
            index_offset,
            member_count,
            reserved_bytes,
        ) = header
        if magic != _PACBIN_HEADER_MAGIC:
            raise ComparisonError(f"{description} has invalid header magic")
        _validate_pacbin_version_size_flags(
            description="header",
            version=version,
            encoded_size=header_size,
            expected_size=_PACBIN_HEADER.size,
            flags=flags,
        )
        if (
            alignment != _PACBIN_ALIGNMENT
            or reserved != 0
            or reserved_bytes != b"\0" * 24
            or index_offset % _PACBIN_ALIGNMENT
        ):
            raise ComparisonError(f"{description} has non-canonical header fields")
        footer_offset = file_size - _PACBIN_FOOTER.size
        if index_offset < _PACBIN_HEADER.size or index_offset >= footer_offset:
            raise ComparisonError(f"{description} has invalid index bounds")
        footer = _PACBIN_FOOTER.unpack(
            _read_exact_at(
                stream,
                footer_offset,
                _PACBIN_FOOTER.size,
                description=f"{description} footer",
            )
        )
        (
            footer_magic,
            footer_version,
            footer_size,
            footer_flags,
            footer_index_offset,
            footer_member_count,
            expected_index_digest,
        ) = footer
        if footer_magic != _PACBIN_FOOTER_MAGIC:
            raise ComparisonError(f"{description} has invalid footer magic")
        _validate_pacbin_version_size_flags(
            description="footer",
            version=footer_version,
            encoded_size=footer_size,
            expected_size=_PACBIN_FOOTER.size,
            flags=footer_flags,
        )
        if footer_index_offset != index_offset or footer_member_count != member_count:
            raise ComparisonError(f"{description} header and footer disagree")
        available_index_bytes = footer_offset - index_offset

        stream.seek(index_offset)
        index_digest = hashlib.sha256()

        def read_index(length: int, label: str) -> bytes:
            if length < 0 or stream.tell() > footer_offset - length:
                raise ComparisonError(f"{description} {label} is truncated")
            value = _read_exact(
                stream,
                length,
                description=f"{description} {label}",
            )
            index_digest.update(value)
            return value

        index_header = _PACBIN_INDEX_HEADER.unpack(
            read_index(_PACBIN_INDEX_HEADER.size, "index header")
        )
        (
            index_magic,
            index_version,
            index_header_size,
            index_flags,
            index_member_count,
            index_reserved,
        ) = index_header
        if index_magic != _PACBIN_INDEX_MAGIC:
            raise ComparisonError(f"{description} has invalid index magic")
        _validate_pacbin_version_size_flags(
            description="index",
            version=index_version,
            encoded_size=index_header_size,
            expected_size=_PACBIN_INDEX_HEADER.size,
            flags=index_flags,
        )
        if index_member_count != member_count or index_reserved != 0:
            raise ComparisonError(f"{description} has inconsistent index metadata")
        if member_count > available_index_bytes // (_PACBIN_INDEX_ENTRY.size + 8):
            raise ComparisonError(f"{description} member count cannot fit in index")

        members: dict[str, PacbinMemberSnapshot] = {}
        folded_paths: set[str] = set()
        previous_path_bytes: bytes | None = None
        for _ in range(member_count):
            (
                path_length,
                kind,
                entry_flags,
                offset,
                length,
                digest,
            ) = _PACBIN_INDEX_ENTRY.unpack(
                read_index(_PACBIN_INDEX_ENTRY.size, "index entry")
            )
            if entry_flags != 0:
                raise ComparisonError(f"{description} has unknown member flags")
            path_bytes = read_index(path_length, "member path")
            logical_path = _canonical_pacbin_member_path(path_bytes)
            folded = logical_path.casefold()
            if (
                logical_path in members
                or folded in folded_paths
                or (
                    previous_path_bytes is not None
                    and path_bytes <= previous_path_bytes
                )
            ):
                raise ComparisonError(
                    f"{description} member paths are not unique and sorted"
                )
            previous_path_bytes = path_bytes
            folded_paths.add(folded)
            padding = _alignment_padding(
                _PACBIN_INDEX_ENTRY.size + path_length,
                _PACBIN_INDEX_ALIGNMENT,
            )
            if read_index(padding, "index padding") != b"\0" * padding:
                raise ComparisonError(f"{description} index padding is nonzero")
            members[logical_path] = PacbinMemberSnapshot(
                logical_path=logical_path,
                kind=kind,
                offset=offset,
                length=length,
                sha256=digest.hex(),
            )
        if stream.tell() != footer_offset:
            raise ComparisonError(f"{description} index has trailing bytes")
        if index_digest.digest() != expected_index_digest:
            raise ComparisonError(f"{description} index digest mismatch")

        expected_offset = _PACBIN_HEADER.size
        for member in members.values():
            aligned_offset = expected_offset + _alignment_padding(
                expected_offset, _PACBIN_ALIGNMENT
            )
            if (
                member.offset != aligned_offset
                or member.offset % _PACBIN_ALIGNMENT
                or member.length > (1 << 64) - 1 - member.offset
                or member.offset + member.length > index_offset
            ):
                raise ComparisonError(f"{description} member layout is non-canonical")
            padding = member.offset - expected_offset
            if (
                _read_exact_at(
                    stream,
                    expected_offset,
                    padding,
                    description=f"{description} payload padding",
                )
                != b"\0" * padding
            ):
                raise ComparisonError(f"{description} payload padding is nonzero")
            actual_sha = _hash_stream_range(
                stream,
                offset=member.offset,
                length=member.length,
                description=f"{description} member {member.logical_path}",
            )
            if actual_sha != member.sha256:
                raise ComparisonError(
                    f"{description} member digest mismatch: {member.logical_path}"
                )
            expected_offset = member.offset + member.length
        canonical_index_offset = expected_offset + _alignment_padding(
            expected_offset, _PACBIN_ALIGNMENT
        )
        if canonical_index_offset != index_offset:
            raise ComparisonError(f"{description} payload layout has a gap")
        trailing_padding = index_offset - expected_offset
        if (
            _read_exact_at(
                stream,
                expected_offset,
                trailing_padding,
                description=f"{description} trailing payload padding",
            )
            != b"\0" * trailing_padding
        ):
            raise ComparisonError(f"{description} trailing padding is nonzero")

        snapshot = RecurrencePacbinSnapshot(
            path=path,
            file_size=file_size,
            index_sha256=index_digest.hexdigest(),
            unpacked_size_bytes=sum(member.length for member in members.values()),
            members=members,
            projection_certificate=None,
        )
        if not recurrence_contract:
            return snapshot

        expected_members = {_PLAN_MEMBER_PATH}
        if _CERTIFICATE_MEMBER_PATH in members:
            expected_members.add(_CERTIFICATE_MEMBER_PATH)
        if set(members) != expected_members:
            raise ComparisonError(
                f"{description} must contain one plan and at most one projection "
                "certificate"
            )
        plan = members[_PLAN_MEMBER_PATH]
        if plan.kind != _PLAN_MEMBER_KIND:
            raise ComparisonError(f"{description} plan member kind is incompatible")
        certificate_member = members.get(_CERTIFICATE_MEMBER_PATH)
        certificate = None
        if certificate_member is not None:
            if certificate_member.kind != _CERTIFICATE_MEMBER_KIND:
                raise ComparisonError(
                    f"{description} projection-certificate kind is incompatible"
                )
            certificate = _parse_projection_certificate(
                stream,
                member=certificate_member,
                description=f"{description} projection certificate",
            )
        return RecurrencePacbinSnapshot(
            path=snapshot.path,
            file_size=snapshot.file_size,
            index_sha256=snapshot.index_sha256,
            unpacked_size_bytes=snapshot.unpacked_size_bytes,
            members=snapshot.members,
            projection_certificate=certificate,
        )


def _payload_relative_path(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ComparisonError(f"{description} has an invalid payload path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ComparisonError(
            f"{description} payload path is not normalized relative POSIX: {value!r}"
        )
    return value


def _mapping(value: object, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonError(f"{description} must be an object")
    return value


def _sequence(value: object, *, description: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ComparisonError(f"{description} must be an array")
    return value


def _sha256(value: object, *, description: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ComparisonError(f"{description} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: object, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ComparisonError(f"{description} must be a non-negative integer")
    return value


def _nonempty_string(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComparisonError(f"{description} must be a non-empty string")
    return value


def _utc_timestamp(value: object, *, description: str) -> str:
    text = _nonempty_string(value, description=description)
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ComparisonError(f"{description} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ComparisonError(f"{description} must have an explicit UTC offset")
    return text


def _timings(value: object, *, description: str) -> Mapping[str, Any]:
    timings = _mapping(value, description=description)
    for key, seconds in timings.items():
        if not isinstance(key, str) or not key:
            raise ComparisonError(f"{description} has an invalid phase name")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or float(seconds) < 0.0
        ):
            raise ComparisonError(
                f"{description}.{key} must be a non-negative finite number"
            )
    return timings


def _strict_mapping(
    value: object,
    *,
    description: str,
    expected_fields: frozenset[str],
) -> Mapping[str, Any]:
    result = _mapping(value, description=description)
    if set(result) != expected_fields:
        missing = sorted(expected_fields - set(result))
        extra = sorted(set(result) - expected_fields)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ComparisonError(f"{description} fields are invalid: {'; '.join(details)}")
    return result


def _validate_native_generation_profile(
    value: object,
    *,
    description: str,
    expected_serialized_bytes: Mapping[str, int] | None,
) -> None:
    profile = _strict_mapping(
        value,
        description=description,
        expected_fields=frozenset(
            {
                "schema_version",
                "scope",
                "timings_seconds",
                "operation_counters",
                "serialized_bytes",
            }
        ),
    )
    if profile.get("schema_version") != 1 or profile.get("scope") != "generation-only":
        raise ComparisonError(f"{description} has an incompatible contract")
    timings = _strict_mapping(
        profile.get("timings_seconds"),
        description=f"{description}.timings_seconds",
        expected_fields=_GENERATION_PROFILE_TIMINGS,
    )
    _timings(timings, description=f"{description}.timings_seconds")
    counters = _strict_mapping(
        profile.get("operation_counters"),
        description=f"{description}.operation_counters",
        expected_fields=_GENERATION_PROFILE_COUNTERS,
    )
    for name, value in counters.items():
        _nonnegative_int(value, description=f"{description}.{name}")
    serialized = _strict_mapping(
        profile.get("serialized_bytes"),
        description=f"{description}.serialized_bytes",
        expected_fields=_GENERATION_PROFILE_SERIALIZED_BYTES,
    )
    normalized_serialized = {
        name: _nonnegative_int(value, description=f"{description}.{name}")
        for name, value in serialized.items()
    }
    if expected_serialized_bytes is not None and normalized_serialized != dict(
        expected_serialized_bytes
    ):
        raise ComparisonError(
            f"{description} serialized-byte telemetry is not linked to its "
            "authenticated runtime schedule"
        )


def _validate_recurrence_schedule_profiles(
    manifest: Mapping[str, Any],
    *,
    label: str,
    schedule_records: Mapping[str, Mapping[str, Any]],
    schedules: Mapping[str, RecurrencePacbinSnapshot],
) -> None:
    generation = _manifest_generation(manifest)
    raw_profiles = generation.get("recurrence_schedule_profiles", _MISSING)
    if raw_profiles is _MISSING:
        return
    profiles = _mapping(
        raw_profiles,
        description=f"{label} recurrence schedule generation profiles",
    )
    schedule_path_by_digest = {
        _sha256(
            record.get("digest"),
            description=f"{label} recurrence schedule digest",
        ): path
        for path, record in schedule_records.items()
    }
    if set(profiles) != set(schedule_path_by_digest):
        raise ComparisonError(
            f"{label} recurrence schedule generation-profile inventory is stale"
        )
    for schedule_digest, raw_profile in profiles.items():
        profile = _strict_mapping(
            raw_profile,
            description=(f"{label} recurrence schedule profile {schedule_digest}"),
            expected_fields=frozenset({"schema_version", "native_passes"}),
        )
        if profile.get("schema_version") != 1:
            raise ComparisonError(
                f"{label} recurrence schedule profile {schedule_digest} "
                "has an incompatible schema"
            )
        native_passes = _mapping(
            profile.get("native_passes"),
            description=(
                f"{label} recurrence schedule profile {schedule_digest} native_passes"
            ),
        )
        if "final" not in native_passes or not set(native_passes) <= {
            "baseline",
            "final",
        }:
            raise ComparisonError(
                f"{label} recurrence schedule profile {schedule_digest} "
                "has invalid native-pass names"
            )
        schedule_path = schedule_path_by_digest[schedule_digest]
        schedule = schedules[schedule_path]
        final_serialized = {
            "plan_payload": schedule.members[_PLAN_MEMBER_PATH].length,
            "container": schedule.file_size,
            "unpacked_container": schedule.unpacked_size_bytes,
        }
        for pass_name, native_profile in native_passes.items():
            _validate_native_generation_profile(
                native_profile,
                description=(
                    f"{label} recurrence schedule profile {schedule_digest} "
                    f"native_passes.{pass_name}"
                ),
                expected_serialized_bytes=(
                    final_serialized if pass_name == "final" else None
                ),
            )


def _manifest_generation(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    extensions = _mapping(manifest.get("extensions"), description="artifact extensions")
    return _mapping(
        extensions.get("generation"), description="artifact generation extension"
    )


def _validate_manifest_metadata(manifest: Mapping[str, Any]) -> None:
    if manifest.get("kind") != "pyamplicol-process":
        raise ComparisonError("artifact manifest is not a pyAmplicol process artifact")
    if manifest.get("schema_version") != 3:
        raise ComparisonError("artifact manifest does not use schema version 3")
    artifact_id = _sha256(manifest.get("artifact_id"), description="artifact_id")
    expected_artifact_id = _runtime_artifact_id(manifest)
    if artifact_id != expected_artifact_id:
        raise ComparisonError("artifact_id does not authenticate the manifest")
    _utc_timestamp(manifest.get("created_utc"), description="created_utc")
    producer = _mapping(manifest.get("producer"), description="artifact producer")
    revision = producer.get("git_revision")
    if not isinstance(revision, str) or _GIT_REVISION_RE.fullmatch(revision) is None:
        raise ComparisonError("artifact producer.git_revision is invalid")
    _sha256(
        producer.get("native_build_inputs_sha256"),
        description="artifact producer.native_build_inputs_sha256",
    )
    _nonempty_string(producer.get("version"), description="artifact producer.version")
    generation = _manifest_generation(manifest)
    _timings(
        generation.get("phase_timings_seconds"),
        description="artifact generation phase timings",
    )


def _validate_recurrence_inventory(
    payloads: Mapping[str, PayloadSnapshot],
    executions: Mapping[str, dict[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    required = {"evaluators.pacbin", "recurrence/schedule-index.json"}
    missing = sorted(required - payloads.keys())
    if missing:
        raise ComparisonError(
            "recurrence artifact is missing required payloads: " + ", ".join(missing)
        )
    schedules = [path for path in payloads if _RUNTIME_SCHEDULE_RE.fullmatch(path)]
    if not schedules:
        raise ComparisonError("recurrence artifact has no runtime schedule payload")

    execution_ids = {
        match.group(1)
        for path in executions
        if (match := _EXECUTION_PATH_RE.fullmatch(path)) is not None
    }
    binding_ids = {
        match.group(1)
        for path in payloads
        if (match := _BINDING_PATH_RE.fullmatch(path)) is not None
    }
    if not execution_ids or binding_ids != execution_ids:
        raise ComparisonError(
            "recurrence execution manifests and process bindings do not match"
        )

    generation = _manifest_generation(manifest)
    concrete = _sequence(
        generation.get("concrete_processes"),
        description="artifact concrete processes",
    )
    concrete_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_record in enumerate(concrete):
        record = _mapping(raw_record, description=f"artifact concrete process {index}")
        process_id = _nonempty_string(
            record.get("id"), description=f"artifact concrete process {index}.id"
        )
        if process_id in concrete_by_id:
            raise ComparisonError(f"duplicate concrete process id: {process_id}")
        concrete_by_id[process_id] = record
    if set(concrete_by_id) != execution_ids:
        raise ComparisonError(
            "artifact concrete-process inventory does not match executions"
        )
    for process_id, record in concrete_by_id.items():
        execution_path = f"processes/{process_id}/execution.json"
        expected = payloads[execution_path].sha256
        observed = _sha256(
            record.get("execution_manifest_sha256"),
            description=(
                f"artifact concrete process {process_id} execution manifest digest"
            ),
        )
        if observed != expected:
            raise ComparisonError(
                f"concrete process {process_id} execution digest is stale"
            )


def _validate_linked_recurrence_metadata(
    *,
    label: str,
    payloads: Mapping[str, PayloadSnapshot],
    executions: Mapping[str, dict[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, RecurrencePacbinSnapshot]]:
    schedule_index_payload = payloads[_SCHEDULE_INDEX_PATH]
    schedule_index = _json_object(
        schedule_index_payload.path,
        description=f"{label} recurrence schedule index",
    )
    if (
        schedule_index.get("kind") != "pyamplicol-recurrence-schedule-sharing"
        or schedule_index.get("schema_version") != 3
        or schedule_index.get("runtime_ownership")
        != "root-schedule-plus-process-binding"
        or schedule_index.get("interning_phase") != "before-direct-lowering"
    ):
        raise ComparisonError(
            f"{label} recurrence schedule index has an incompatible contract"
        )
    raw_schedule_records = _sequence(
        schedule_index.get("schedules"),
        description=f"{label} recurrence schedule-index schedules",
    )
    schedule_records: dict[str, Mapping[str, Any]] = {}
    for index, raw_record in enumerate(raw_schedule_records):
        record = _mapping(
            raw_record,
            description=f"{label} recurrence schedule record {index}",
        )
        path = _payload_relative_path(
            record.get("path"),
            description=f"{label} recurrence schedule record {index}",
        )
        if path in schedule_records:
            raise ComparisonError(f"{label} recurrence schedule path is duplicated")
        schedule_records[path] = record
    runtime_paths = {path for path in payloads if _RUNTIME_SCHEDULE_RE.fullmatch(path)}
    if set(schedule_records) != runtime_paths:
        raise ComparisonError(
            f"{label} schedule-index inventory does not match runtime PACBINs"
        )

    schedules: dict[str, RecurrencePacbinSnapshot] = {}
    for path, record in schedule_records.items():
        payload = payloads[path]
        match = _RUNTIME_SCHEDULE_RE.fullmatch(path)
        assert match is not None
        schedule_digest = _sha256(
            record.get("digest"),
            description=f"{label} schedule {path}.digest",
        )
        if schedule_digest != match.group(1):
            raise ComparisonError(
                f"{label} schedule digest does not match its artifact path: {path}"
            )
        pacbin = _load_recurrence_pacbin(
            payload.path,
            description=f"{label} recurrence runtime PACBIN {path}",
        )
        linked_values = (
            (
                "sha256",
                _sha256(
                    record.get("sha256"),
                    description=f"{label} schedule {path}.sha256",
                ),
                payload.sha256,
            ),
            (
                "size_bytes",
                _nonnegative_int(
                    record.get("size_bytes"),
                    description=f"{label} schedule {path}.size_bytes",
                ),
                payload.size_bytes,
            ),
            (
                "member_count",
                _nonnegative_int(
                    record.get("member_count"),
                    description=f"{label} schedule {path}.member_count",
                ),
                len(pacbin.members),
            ),
            (
                "unpacked_size_bytes",
                _nonnegative_int(
                    record.get("unpacked_size_bytes"),
                    description=f"{label} schedule {path}.unpacked_size_bytes",
                ),
                pacbin.unpacked_size_bytes,
            ),
            (
                "index_sha256",
                _sha256(
                    record.get("index_sha256"),
                    description=f"{label} schedule {path}.index_sha256",
                ),
                pacbin.index_sha256,
            ),
        )
        stale_fields = [
            name for name, observed, expected in linked_values if observed != expected
        ]
        if stale_fields:
            raise ComparisonError(
                f"{label} schedule record has stale linked fields for {path}: "
                + ", ".join(stale_fields)
            )
        schedules[path] = pacbin

    extension = _mapping(
        _mapping(
            manifest.get("extensions"),
            description=f"{label} artifact extensions",
        ).get("recurrence_schedule_sharing"),
        description=f"{label} recurrence schedule-sharing extension",
    )
    if (
        extension.get("kind") != schedule_index.get("kind")
        or extension.get("schema_version") != schedule_index.get("schema_version")
        or extension.get("index_path") != _SCHEDULE_INDEX_PATH
        or extension.get("index_sha256") != schedule_index_payload.sha256
        or extension.get("runtime_ownership") != schedule_index.get("runtime_ownership")
        or extension.get("interning_phase") != schedule_index.get("interning_phase")
    ):
        raise ComparisonError(f"{label} recurrence schedule-sharing extension is stale")
    for field, expected in (
        ("schedule_count", len(schedule_records)),
        (
            "binding_count",
            len(
                _sequence(
                    schedule_index.get("bindings"),
                    description=f"{label} recurrence schedule-index bindings",
                )
            ),
        ),
    ):
        observed = _nonnegative_int(
            extension.get(field),
            description=f"{label} recurrence sharing {field}",
        )
        if observed != expected or schedule_index.get(field) != expected:
            raise ComparisonError(f"{label} recurrence sharing {field} is stale")
    expected_alias_count = int(extension["binding_count"]) - int(
        extension["schedule_count"]
    )
    if (
        extension.get("schedule_alias_count") != expected_alias_count
        or schedule_index.get("schedule_alias_count") != expected_alias_count
    ):
        raise ComparisonError(f"{label} recurrence sharing alias count is stale")

    for execution_path, execution in executions.items():
        plan = _mapping(
            execution.get("plan"),
            description=f"{label} execution {execution_path}.plan",
        )
        runtime = _mapping(
            plan.get("runtime_schedule"),
            description=f"{label} execution {execution_path}.runtime_schedule",
        )
        schedule_path = _payload_relative_path(
            runtime.get("path"),
            description=f"{label} execution {execution_path}.runtime_schedule",
        )
        if schedule_path not in schedules:
            raise ComparisonError(
                f"{label} execution {execution_path} names an unknown schedule"
            )
        pacbin = schedules[schedule_path]
        schedule_payload = payloads[schedule_path]
        expected_runtime_values = {
            "kind": "pyamplicol-recurrence-runtime-container",
            "schema_version": 1,
            "storage_abi": "pacbin-v1",
            "plan_member_path": _PLAN_MEMBER_PATH,
            "sha256": schedule_payload.sha256,
            "size_bytes": schedule_payload.size_bytes,
            "member_count": len(pacbin.members),
            "unpacked_size_bytes": pacbin.unpacked_size_bytes,
            "index_sha256": pacbin.index_sha256,
        }
        stale_runtime = [
            field
            for field, expected in expected_runtime_values.items()
            if runtime.get(field) != expected
        ]
        if stale_runtime:
            raise ComparisonError(
                f"{label} execution {execution_path} has stale runtime-schedule "
                "metadata: " + ", ".join(stale_runtime)
            )
        inspection = _mapping(
            plan.get("inspection_summary"),
            description=f"{label} execution {execution_path}.inspection_summary",
        )
        certificate_record = inspection.get("color_projection_certificate")
        certificate = pacbin.projection_certificate
        if certificate is None:
            if certificate_record is not None:
                raise ComparisonError(
                    f"{label} execution {execution_path} declares an absent "
                    "projection certificate"
                )
        else:
            record = _mapping(
                certificate_record,
                description=(
                    f"{label} execution {execution_path} projection certificate"
                ),
            )
            expected_certificate_values = {
                "path": _CERTIFICATE_MEMBER_PATH,
                "schema_version": 1,
                "proof_kind": "exact-rectangular-sum-projection",
                "publishable": True,
                "size_bytes": pacbin.members[_CERTIFICATE_MEMBER_PATH].length,
                "sha256": certificate.sha256,
            }
            stale_certificate = [
                field
                for field, expected in expected_certificate_values.items()
                if record.get(field) != expected
            ]
            if stale_certificate:
                raise ComparisonError(
                    f"{label} execution {execution_path} has stale projection-"
                    "certificate metadata: " + ", ".join(stale_certificate)
                )
    _validate_recurrence_schedule_profiles(
        manifest,
        label=label,
        schedule_records=schedule_records,
        schedules=schedules,
    )
    return schedule_index, schedules


def _load_artifact(path: Path, *, label: str) -> ArtifactSnapshot:
    raw_root = path.expanduser()
    if raw_root.is_symlink():
        raise ComparisonError(f"{label} artifact root may not be a symlink")
    try:
        root = raw_root.resolve(strict=True)
    except OSError as error:
        raise ComparisonError(f"{label} artifact does not exist: {path}") from error
    if not root.is_dir():
        raise ComparisonError(f"{label} artifact is not a directory: {root}")

    actual_files: set[str] = set()
    try:
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise ComparisonError(
                    f"{label} artifact contains a symlink: {candidate}"
                )
            if candidate.is_file():
                actual_files.add(candidate.relative_to(root).as_posix())
    except OSError as error:
        raise ComparisonError(f"cannot inspect {label} artifact: {root}") from error

    manifest_path = root / "artifact.json"
    manifest = _json_object(manifest_path, description=f"{label} artifact manifest")
    _validate_manifest_metadata(manifest)
    raw_payloads = _sequence(
        manifest.get("payloads"), description=f"{label} artifact payloads"
    )
    payloads: dict[str, PayloadSnapshot] = {}
    executions: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(raw_payloads):
        record = dict(
            _mapping(raw_record, description=f"{label} payload record {index}")
        )
        relative = _payload_relative_path(
            record.get("path"), description=f"{label} payload record {index}"
        )
        if relative in payloads:
            raise ComparisonError(f"{label} artifact has duplicate payload {relative}")
        role = _nonempty_string(
            record.get("role"), description=f"{label} payload {relative}.role"
        )
        declared_size = _nonnegative_int(
            record.get("size_bytes"),
            description=f"{label} payload {relative}.size_bytes",
        )
        declared_sha = _sha256(
            record.get("sha256"),
            description=f"{label} payload {relative}.sha256",
        )
        payload_path = root.joinpath(*PurePosixPath(relative).parts)
        if (
            not payload_path.is_file()
            or payload_path.is_symlink()
            or payload_path.resolve().parent == root.parent
        ):
            raise ComparisonError(
                f"{label} payload is missing or not a regular file: {relative}"
            )
        actual_size = payload_path.stat().st_size
        actual_sha = _sha256_file(payload_path)
        if declared_size != actual_size or declared_sha != actual_sha:
            raise ComparisonError(
                f"{label} payload metadata does not authenticate {relative}"
            )
        snapshot = PayloadSnapshot(
            relative_path=relative,
            path=payload_path,
            role=role,
            size_bytes=actual_size,
            sha256=actual_sha,
            record=record,
        )
        payloads[relative] = snapshot
        if _EXECUTION_PATH_RE.fullmatch(relative):
            execution = _json_object(
                payload_path, description=f"{label} recurrence execution manifest"
            )
            if execution.get("kind") != "pyamplicol-runtime-recurrence-execution":
                raise ComparisonError(
                    f"{label} execution payload is not recurrence mode: {relative}"
                )
            plan = _mapping(
                execution.get("plan"),
                description=f"{label} execution {relative}.plan",
            )
            inspection = _mapping(
                plan.get("inspection_summary"),
                description=f"{label} execution {relative}.inspection_summary",
            )
            _timings(
                inspection.get("generation_timings_seconds"),
                description=f"{label} execution {relative} generation timings",
            )
            executions[relative] = execution

    expected_files = {"artifact.json", *payloads}
    if actual_files != expected_files:
        extra = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        details = []
        if extra:
            details.append("unlisted=" + ",".join(extra))
        if missing:
            details.append("missing=" + ",".join(missing))
        raise ComparisonError(
            f"{label} artifact file inventory is not exact: {'; '.join(details)}"
        )
    _validate_recurrence_inventory(payloads, executions, manifest)
    schedule_index, recurrence_schedules = _validate_linked_recurrence_metadata(
        label=label,
        payloads=payloads,
        executions=executions,
        manifest=manifest,
    )
    return ArtifactSnapshot(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=_sha256_file(manifest_path),
        payloads=payloads,
        executions=executions,
        schedule_index=schedule_index,
        recurrence_schedules=recurrence_schedules,
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_domain_sha256(domain: str, value: object) -> str:
    return _canonical_sha256({"domain": domain, "value": value})


def _structural_semantic_map_witnesses(
    execution: Mapping[str, Any],
) -> dict[str, object]:
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
        "source_contract": lambda key: (
            key
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
            }
        ),
    }

    def project(predicate: Callable[[str], bool]) -> list[dict[str, object]]:
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
            raise ComparisonError(
                "structural proof execution lacks a semantic-map witness"
            )
        return rows

    result: dict[str, object] = {}
    identities: set[str] = set()
    for name, predicate in predicates.items():
        rows = project(predicate)
        digest = _canonical_domain_sha256(
            _STRUCTURAL_SEMANTIC_MAP_DOMAINS[name],
            rows,
        )
        if digest in identities:
            raise ComparisonError("structural proof semantic-map identities repeat")
        identities.add(digest)
        result[name] = {"sha256": digest, "rows": rows}
    return result


def _structural_metrics(value: object) -> list[dict[str, object]]:
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
                        raise ComparisonError(
                            f"structural metric {child_path} must be non-negative"
                        )
                    rows.append({"path": child_path, "value": child})
                else:
                    visit(child, child_path)
        elif isinstance(item, Sequence) and not isinstance(item, str | bytes):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    if not rows:
        raise ComparisonError("structural proof execution has no structural metrics")
    return rows


def _describe(value: object) -> object:
    if value is _MISSING:
        return {"kind": "missing"}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= 160:
            return value
        return {
            "kind": "string",
            "length": len(value),
            "preview": value[:157] + "...",
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            "kind": "object",
            "key_count": len(value),
            "sha256": _canonical_sha256(value),
        }
    if isinstance(value, list):
        return {
            "kind": "array",
            "length": len(value),
            "sha256": _canonical_sha256(value),
        }
    return {"kind": type(value).__name__, "repr": repr(value)}


def _pointer(path: Sequence[str | int]) -> str:
    if not path:
        return ""
    return "/" + "/".join(
        str(part).replace("~", "~0").replace("/", "~1") for part in path
    )


def _json_differences(
    left: object,
    right: object,
    *,
    file: str,
    path: tuple[str | int, ...] = (),
) -> list[dict[str, object]]:
    if type(left) is not type(right):
        return [
            {
                "kind": "json-value",
                "file": file,
                "json_pointer": _pointer(path),
                "baseline": _describe(left),
                "candidate": _describe(right),
            }
        ]
    if isinstance(left, dict):
        assert isinstance(right, dict)
        differences: list[dict[str, object]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append(
                    {
                        "kind": "json-value",
                        "file": file,
                        "json_pointer": _pointer((*path, key)),
                        "baseline": _describe(left.get(key, _MISSING)),
                        "candidate": _describe(right.get(key, _MISSING)),
                    }
                )
                continue
            differences.extend(
                _json_differences(
                    left[key],
                    right[key],
                    file=file,
                    path=(*path, key),
                )
            )
        return differences
    if isinstance(left, list):
        assert isinstance(right, list)
        if len(left) != len(right):
            return [
                {
                    "kind": "json-value",
                    "file": file,
                    "json_pointer": _pointer(path),
                    "baseline": _describe(left),
                    "candidate": _describe(right),
                }
            ]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            differences.extend(
                _json_differences(
                    left_item,
                    right_item,
                    file=file,
                    path=(*path, index),
                )
            )
        return differences
    if left == right:
        return []
    return [
        {
            "kind": "json-value",
            "file": file,
            "json_pointer": _pointer(path),
            "baseline": _describe(left),
            "candidate": _describe(right),
        }
    ]


def _record_allowed(
    records: list[dict[str, object]],
    *,
    file: str,
    json_path: str,
    category: str,
    baseline: object,
    candidate: object,
) -> None:
    if baseline == candidate:
        return
    records.append(
        {
            "file": file,
            "json_path": json_path,
            "category": category,
            "baseline": _describe(baseline),
            "candidate": _describe(candidate),
        }
    )


def _compare_recurrence_pacbins(
    baseline: RecurrencePacbinSnapshot,
    candidate: RecurrencePacbinSnapshot,
    *,
    relative_path: str,
    unknown: list[dict[str, object]],
) -> RecurrencePacbinComparison:
    exact_bytes_match = _files_equal(baseline.path, candidate.path)
    baseline_paths = set(baseline.members)
    candidate_paths = set(candidate.members)
    if baseline_paths != candidate_paths:
        unknown.append(
            {
                "kind": "pacbin-member-inventory",
                "path": relative_path,
                "baseline": sorted(baseline_paths),
                "candidate": sorted(candidate_paths),
            }
        )
        return RecurrencePacbinComparison(
            exact_bytes_match=exact_bytes_match,
            policy_match=False,
            plan_bytes_match=False,
            projection_certificate_bodies_match=False,
        )

    structural_member_mismatch = False
    for member_path in sorted(baseline_paths):
        left = baseline.members[member_path]
        right = candidate.members[member_path]
        left_shape = (left.kind, left.offset, left.length)
        right_shape = (right.kind, right.offset, right.length)
        if left_shape != right_shape:
            structural_member_mismatch = True
            unknown.append(
                {
                    "kind": "pacbin-member-layout",
                    "path": relative_path,
                    "member": member_path,
                    "baseline": {
                        "kind": left.kind,
                        "offset": left.offset,
                        "length": left.length,
                    },
                    "candidate": {
                        "kind": right.kind,
                        "offset": right.offset,
                        "length": right.length,
                    },
                }
            )
    if baseline.file_size != candidate.file_size:
        structural_member_mismatch = True
        unknown.append(
            {
                "kind": "pacbin-container-size",
                "path": relative_path,
                "baseline": baseline.file_size,
                "candidate": candidate.file_size,
            }
        )

    left_plan = baseline.members[_PLAN_MEMBER_PATH]
    right_plan = candidate.members[_PLAN_MEMBER_PATH]
    plan_bytes_match = (
        left_plan.kind == right_plan.kind
        and left_plan.length == right_plan.length
        and left_plan.sha256 == right_plan.sha256
        and _file_ranges_equal(
            baseline.path,
            candidate.path,
            left_offset=left_plan.offset,
            right_offset=right_plan.offset,
            length=left_plan.length,
        )
    )
    if not plan_bytes_match:
        unknown.append(
            {
                "kind": "pacbin-runtime-plan-bytes",
                "path": relative_path,
                "member": _PLAN_MEMBER_PATH,
                "baseline": {
                    "sha256": left_plan.sha256,
                    "size_bytes": left_plan.length,
                },
                "candidate": {
                    "sha256": right_plan.sha256,
                    "size_bytes": right_plan.length,
                },
            }
        )

    left_certificate = baseline.projection_certificate
    right_certificate = candidate.projection_certificate
    if (left_certificate is None) != (right_certificate is None):
        unknown.append(
            {
                "kind": "projection-certificate-presence",
                "path": relative_path,
                "baseline": left_certificate is not None,
                "candidate": right_certificate is not None,
            }
        )
        return RecurrencePacbinComparison(
            exact_bytes_match=exact_bytes_match,
            policy_match=False,
            plan_bytes_match=plan_bytes_match,
            projection_certificate_bodies_match=False,
        )
    if left_certificate is None:
        if not exact_bytes_match:
            unknown.append(
                {
                    "kind": "pacbin-container-bytes",
                    "path": relative_path,
                    "baseline": {
                        "sha256": _sha256_file(baseline.path),
                        "size_bytes": baseline.file_size,
                    },
                    "candidate": {
                        "sha256": _sha256_file(candidate.path),
                        "size_bytes": candidate.file_size,
                    },
                }
            )
        return RecurrencePacbinComparison(
            exact_bytes_match=exact_bytes_match,
            policy_match=exact_bytes_match and plan_bytes_match,
            plan_bytes_match=plan_bytes_match,
            projection_certificate_bodies_match=True,
        )

    assert right_certificate is not None
    certificate_bodies_match = (
        left_certificate.body_length == right_certificate.body_length
        and _file_ranges_equal(
            baseline.path,
            candidate.path,
            left_offset=left_certificate.body_offset,
            right_offset=right_certificate.body_offset,
            length=left_certificate.body_length,
        )
    )
    if not certificate_bodies_match:
        unknown.append(
            {
                "kind": "projection-certificate-structural-body",
                "path": relative_path,
                "member": _CERTIFICATE_MEMBER_PATH,
                "baseline": {
                    "length": left_certificate.body_length,
                    "sha256": _sha256_file_range(
                        baseline.path,
                        offset=left_certificate.body_offset,
                        length=left_certificate.body_length,
                    ),
                },
                "candidate": {
                    "length": right_certificate.body_length,
                    "sha256": _sha256_file_range(
                        candidate.path,
                        offset=right_certificate.body_offset,
                        length=right_certificate.body_length,
                    ),
                },
            }
        )
    policy_match = (
        not structural_member_mismatch
        and plan_bytes_match
        and certificate_bodies_match
        and exact_bytes_match
    )
    if not exact_bytes_match:
        unknown.append(
            {
                "kind": "pacbin-unexplained-container-difference",
                "path": relative_path,
                "baseline": {
                    "sha256": _sha256_file(baseline.path),
                    "index_sha256": baseline.index_sha256,
                },
                "candidate": {
                    "sha256": _sha256_file(candidate.path),
                    "index_sha256": candidate.index_sha256,
                },
            }
        )
    return RecurrencePacbinComparison(
        exact_bytes_match=exact_bytes_match,
        policy_match=policy_match,
        plan_bytes_match=plan_bytes_match,
        projection_certificate_bodies_match=certificate_bodies_match,
    )


def _replace_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    path: tuple[str, ...],
    *,
    file: str,
    json_path: str,
    category: str,
    validate: Callable[[object], object],
    allowed: list[dict[str, object]],
) -> None:
    left_parent = left
    right_parent = right
    for key in path[:-1]:
        left_child = left_parent.get(key)
        right_child = right_parent.get(key)
        if not isinstance(left_child, dict) or not isinstance(right_child, dict):
            raise ComparisonError(
                f"required allowed metadata parent is missing: {file}{json_path}"
            )
        left_parent = left_child
        right_parent = right_child
    key = path[-1]
    if key not in left_parent or key not in right_parent:
        raise ComparisonError(f"required allowed metadata path is missing: {json_path}")
    left_value = validate(left_parent[key])
    right_value = validate(right_parent[key])
    _record_allowed(
        allowed,
        file=file,
        json_path=json_path,
        category=category,
        baseline=left_value,
        candidate=right_value,
    )
    left_parent[key] = _NORMALIZED
    right_parent[key] = _NORMALIZED


def _validate_structural_proof(
    artifact: ArtifactSnapshot,
    proof: Mapping[str, Any],
    *,
    relative_path: str,
    label: str,
) -> tuple[str, str]:
    match = _STRUCTURAL_PROOF_PATH_RE.fullmatch(relative_path)
    if match is None:
        raise ComparisonError(f"{label} structural-proof path is invalid")
    process_id = match.group(1)
    expected_execution_path = f"processes/{process_id}/execution.json"
    if (
        proof.get("schema") != "pyamplicol-generation-structural-source-proof-v1"
        or proof.get("status") != "complete"
        or proof.get("process_id") != process_id
    ):
        raise ComparisonError(
            f"{label} structural proof {relative_path} has an incompatible contract"
        )
    payload = artifact.payloads[relative_path]
    if payload.role != "structural-source-proof":
        raise ComparisonError(
            f"{label} structural proof {relative_path} has an invalid payload role"
        )

    producer = _mapping(
        artifact.manifest.get("producer"),
        description=f"{label} artifact producer",
    )
    source_identity = _mapping(
        proof.get("source_identity"),
        description=f"{label} structural proof source identity",
    )
    source_revision = source_identity.get("git_revision")
    native_build_inputs = source_identity.get("native_build_inputs_sha256")
    if source_revision != producer.get(
        "git_revision"
    ) or native_build_inputs != producer.get("native_build_inputs_sha256"):
        raise ComparisonError(
            f"{label} structural proof {relative_path} is not bound to its producer"
        )

    execution_payload = artifact.payloads.get(expected_execution_path)
    execution = artifact.executions.get(expected_execution_path)
    if execution_payload is None or execution is None:
        raise ComparisonError(
            f"{label} structural proof {relative_path} lacks its execution payload"
        )
    execution_identity = _mapping(
        proof.get("execution"),
        description=f"{label} structural proof execution identity",
    )
    if (
        execution_identity.get("path") != expected_execution_path
        or execution_identity.get("sha256") != execution_payload.sha256
        or execution_identity.get("kind") != execution.get("kind")
        or execution_identity.get("color_accuracy") != execution.get("color_accuracy")
    ):
        raise ComparisonError(
            f"{label} structural proof {relative_path} has a stale execution identity"
        )

    semantic_maps = _mapping(
        proof.get("semantic_maps"),
        description=f"{label} structural proof semantic maps",
    )
    if dict(semantic_maps) != _structural_semantic_map_witnesses(execution):
        raise ComparisonError(
            f"{label} structural proof {relative_path} semantic maps are stale"
        )

    inventory = _mapping(
        proof.get("physical_lane_inventory"),
        description=f"{label} structural proof physical inventory",
    )
    if inventory.get("status") != "complete":
        raise ComparisonError(
            f"{label} structural proof {relative_path} inventory is incomplete"
        )
    inventory_body = dict(inventory)
    declared_inventory_digest = _sha256(
        inventory_body.pop("inventory_sha256", None),
        description=f"{label} structural proof inventory digest",
    )
    if declared_inventory_digest != _canonical_domain_sha256(
        "physical-lane-inventory-v1",
        inventory_body,
    ):
        raise ComparisonError(
            f"{label} structural proof {relative_path} inventory digest is stale"
        )

    loose_records = _sequence(
        inventory.get("loose_payloads"),
        description=f"{label} structural proof loose payloads",
    )
    loose_paths: set[str] = set()
    for index, raw_record in enumerate(loose_records):
        record = _mapping(
            raw_record,
            description=f"{label} structural proof loose payload {index}",
        )
        record_path = _payload_relative_path(
            record.get("path"),
            description=f"{label} structural proof loose payload {index}",
        )
        if record_path in loose_paths or record_path not in artifact.payloads:
            raise ComparisonError(
                f"{label} structural proof {relative_path} has an invalid loose payload"
            )
        loose_paths.add(record_path)
        concrete = artifact.payloads[record_path]
        if (
            record.get("sha256") != concrete.sha256
            or record.get("size_bytes") != concrete.size_bytes
        ):
            raise ComparisonError(
                f"{label} structural proof {relative_path} has stale loose payload "
                f"metadata for {record_path}"
            )
    if expected_execution_path not in loose_paths:
        raise ComparisonError(
            f"{label} structural proof {relative_path} omits its execution payload"
        )

    raw_members = _sequence(
        inventory.get("pacbin_members"),
        description=f"{label} structural proof PACBIN members",
    )
    raw_container = inventory.get("pacbin_container")
    actual_members: dict[str, dict[str, object]] = {}
    if raw_container is not None:
        container = _mapping(
            raw_container,
            description=f"{label} structural proof PACBIN container",
        )
        container_path = _payload_relative_path(
            container.get("path"),
            description=f"{label} structural proof PACBIN container",
        )
        container_payload = artifact.payloads.get(container_path)
        if (
            container_payload is None
            or container.get("sha256") != container_payload.sha256
            or container.get("size_bytes") != container_payload.size_bytes
        ):
            raise ComparisonError(
                f"{label} structural proof {relative_path} PACBIN container is stale"
            )
        parsed_container = _load_recurrence_pacbin(
            container_payload.path,
            description=f"{label} structural proof evaluator container",
            recurrence_contract=False,
        )
        if container.get(
            "index_sha256"
        ) != parsed_container.index_sha256 or container.get("member_count") != len(
            parsed_container.members
        ):
            raise ComparisonError(
                f"{label} structural proof {relative_path} PACBIN index is stale"
            )
        for member_path, member in parsed_container.members.items():
            kind_name = _PACBIN_MEMBER_KIND_NAMES.get(member.kind)
            if kind_name is None:
                raise ComparisonError(
                    f"{label} structural proof {relative_path} PACBIN member kind "
                    "is unknown"
                )
            actual_members[member_path] = {
                "logical_path": member_path,
                "kind": kind_name,
                "length": member.length,
                "sha256": member.sha256,
            }
    elif raw_members:
        raise ComparisonError(
            f"{label} structural proof {relative_path} members lack a container"
        )

    declared_member_paths: set[str] = set()
    for index, raw_member in enumerate(raw_members):
        member = _mapping(
            raw_member,
            description=f"{label} structural proof PACBIN member {index}",
        )
        member_path = _nonempty_string(
            member.get("logical_path"),
            description=f"{label} structural proof PACBIN member {index}.logical_path",
        )
        if member_path in declared_member_paths or actual_members.get(
            member_path
        ) != dict(member):
            raise ComparisonError(
                f"{label} structural proof {relative_path} PACBIN member is stale"
            )
        declared_member_paths.add(member_path)

    lanes = _sequence(
        inventory.get("lanes"),
        description=f"{label} structural proof execution lanes",
    )
    if len(lanes) != 1:
        raise ComparisonError(
            f"{label} recurrence structural proof {relative_path} must have one lane"
        )
    lane = _mapping(
        lanes[0],
        description=f"{label} structural proof primary lane",
    )
    metrics = _structural_metrics(execution)
    if (
        lane.get("lane_id") != "primary"
        or lane.get("subtree_sha256")
        != _canonical_domain_sha256("execution-lane-subtree-v1", execution)
        or lane.get("structural_metrics") != metrics
        or lane.get("structural_metrics_sha256")
        != _canonical_domain_sha256("execution-lane-structural-metrics-v1", metrics)
    ):
        raise ComparisonError(
            f"{label} structural proof {relative_path} lane evidence is stale"
        )
    lane_loose = _sequence(
        lane.get("loose_payload_paths"),
        description=f"{label} structural proof lane loose paths",
    )
    lane_members = _sequence(
        lane.get("pacbin_member_paths"),
        description=f"{label} structural proof lane member paths",
    )
    if (
        any(not isinstance(path, str) for path in lane_loose)
        or len(set(lane_loose)) != len(lane_loose)
        or not set(lane_loose) <= loose_paths
        or any(not isinstance(path, str) for path in lane_members)
        or len(set(lane_members)) != len(lane_members)
        or not set(lane_members) <= declared_member_paths
    ):
        raise ComparisonError(
            f"{label} structural proof {relative_path} lane references are invalid"
        )

    expected_roles = {
        ("primary", "loose-payload", str(path)) for path in lane_loose
    } | {("primary", "pacbin-member", str(path)) for path in lane_members}
    raw_roles = _sequence(
        inventory.get("roles"),
        description=f"{label} structural proof roles",
    )
    actual_roles: set[tuple[str, str, str]] = set()
    for index, raw_role in enumerate(raw_roles):
        role = _mapping(
            raw_role,
            description=f"{label} structural proof role {index}",
        )
        actual_roles.add(
            (
                str(role.get("lane_id")),
                str(role.get("object_kind")),
                str(role.get("object_path")),
            )
        )
    if actual_roles != expected_roles or len(actual_roles) != len(raw_roles):
        raise ComparisonError(
            f"{label} structural proof {relative_path} role coverage is invalid"
        )

    proof_body = dict(proof)
    declared_proof_digest = _sha256(
        proof_body.pop("proof_content_sha256", None),
        description=f"{label} structural proof content digest",
    )
    if declared_proof_digest != _canonical_domain_sha256(
        "generation-structural-source-proof-v1",
        proof_body,
    ):
        raise ComparisonError(
            f"{label} structural proof {relative_path} content digest is stale"
        )
    return expected_execution_path, process_id


def _normalize_structural_proof_pair(
    baseline_artifact: ArtifactSnapshot,
    candidate_artifact: ArtifactSnapshot,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    relative_path: str,
    baseline_execution: Mapping[str, Any],
    candidate_execution: Mapping[str, Any],
    allowed: list[dict[str, object]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    left_execution_path, _ = _validate_structural_proof(
        baseline_artifact,
        baseline,
        relative_path=relative_path,
        label="baseline",
    )
    right_execution_path, _ = _validate_structural_proof(
        candidate_artifact,
        candidate,
        relative_path=relative_path,
        label="candidate",
    )
    if (
        left_execution_path != right_execution_path
        or baseline_execution != candidate_execution
    ):
        raise ComparisonError(
            f"structural proof {relative_path} does not have matched "
            "execution semantics"
        )

    left = copy.deepcopy(dict(baseline))
    right = copy.deepcopy(dict(candidate))
    for path, json_path, category, validate in (
        (
            ("source_identity", "git_revision"),
            "/source_identity/git_revision",
            "provenance",
            lambda value: (
                value
                if isinstance(value, str)
                and _GIT_REVISION_RE.fullmatch(value) is not None
                else _raise_invalid("structural-proof source revision")
            ),
        ),
        (
            ("source_identity", "native_build_inputs_sha256"),
            "/source_identity/native_build_inputs_sha256",
            "provenance",
            lambda value: _sha256(
                value,
                description="structural-proof native-build identity",
            ),
        ),
        (
            ("execution", "sha256"),
            "/execution/sha256",
            "derived-execution-metadata",
            lambda value: _sha256(
                value,
                description="structural-proof execution digest",
            ),
        ),
    ):
        _replace_pair(
            left,
            right,
            path,
            file=relative_path,
            json_path=json_path,
            category=category,
            validate=validate,
            allowed=allowed,
        )

    def mutable_execution_loose_record(
        proof: dict[str, Any],
        *,
        label: str,
    ) -> dict[str, Any]:
        inventory = proof["physical_lane_inventory"]
        if not isinstance(inventory, dict):
            raise ComparisonError(
                f"{label} structural proof {relative_path} inventory is invalid"
            )
        matches = [
            record
            for record in inventory.get("loose_payloads", ())
            if isinstance(record, dict) and record.get("path") == left_execution_path
        ]
        if len(matches) != 1:
            raise ComparisonError(
                f"{label} structural proof {relative_path} execution inventory "
                "is ambiguous"
            )
        return matches[0]

    left_loose = mutable_execution_loose_record(left, label="baseline")
    right_loose = mutable_execution_loose_record(right, label="candidate")
    for key, validator in (
        (
            "sha256",
            lambda value: _sha256(
                value,
                description="structural-proof loose execution digest",
            ),
        ),
        (
            "size_bytes",
            lambda value: _nonnegative_int(
                value,
                description="structural-proof loose execution size",
            ),
        ),
    ):
        left_value = validator(left_loose.get(key))
        right_value = validator(right_loose.get(key))
        _record_allowed(
            allowed,
            file=relative_path,
            json_path=(
                "/physical_lane_inventory/loose_payloads"
                f"[path={left_execution_path}]/{key}"
            ),
            category="derived-execution-metadata",
            baseline=left_value,
            candidate=right_value,
        )
        left_loose[key] = _NORMALIZED
        right_loose[key] = _NORMALIZED

    def mutable_primary_lane(
        proof: dict[str, Any],
        *,
        label: str,
    ) -> dict[str, Any]:
        inventory = proof["physical_lane_inventory"]
        if not isinstance(inventory, dict):
            raise ComparisonError(
                f"{label} structural proof {relative_path} inventory is invalid"
            )
        matches = [
            lane
            for lane in inventory.get("lanes", ())
            if isinstance(lane, dict) and lane.get("lane_id") == "primary"
        ]
        if len(matches) != 1:
            raise ComparisonError(
                f"{label} structural proof {relative_path} primary lane is ambiguous"
            )
        return matches[0]

    left_lane = mutable_primary_lane(left, label="baseline")
    right_lane = mutable_primary_lane(right, label="candidate")
    left_lane_digest = _sha256(
        left_lane.get("subtree_sha256"),
        description="baseline structural-proof lane digest",
    )
    right_lane_digest = _sha256(
        right_lane.get("subtree_sha256"),
        description="candidate structural-proof lane digest",
    )
    _record_allowed(
        allowed,
        file=relative_path,
        json_path="/physical_lane_inventory/lanes[id=primary]/subtree_sha256",
        category="derived-execution-metadata",
        baseline=left_lane_digest,
        candidate=right_lane_digest,
    )
    left_lane["subtree_sha256"] = _NORMALIZED
    right_lane["subtree_sha256"] = _NORMALIZED

    for path, json_path, domain in (
        (
            ("physical_lane_inventory", "inventory_sha256"),
            "/physical_lane_inventory/inventory_sha256",
            "derived-execution-metadata",
        ),
        (
            ("proof_content_sha256",),
            "/proof_content_sha256",
            "derived-execution-and-provenance-metadata",
        ),
    ):
        _replace_pair(
            left,
            right,
            path,
            file=relative_path,
            json_path=json_path,
            category=domain,
            validate=lambda value, field=json_path: _sha256(
                value,
                description=f"structural-proof derived digest {field}",
            ),
            allowed=allowed,
        )
    return left, right


def _normalize_manifest_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    matched_execution_paths: set[str],
    matched_structural_proof_paths: set[str],
    allowed: list[dict[str, object]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    left = copy.deepcopy(dict(baseline))
    right = copy.deepcopy(dict(candidate))
    scalar_fields: tuple[
        tuple[tuple[str, ...], str, str, Callable[[object], object]], ...
    ] = (
        (
            ("artifact_id",),
            "/artifact_id",
            "derived-identity",
            lambda value: _sha256(value, description="artifact_id"),
        ),
        (
            ("created_utc",),
            "/created_utc",
            "timing",
            lambda value: _utc_timestamp(value, description="created_utc"),
        ),
        (
            ("producer", "git_revision"),
            "/producer/git_revision",
            "provenance",
            lambda value: (
                value
                if isinstance(value, str)
                and _GIT_REVISION_RE.fullmatch(value) is not None
                else (_raise_invalid("producer.git_revision"))
            ),
        ),
        (
            ("producer", "native_build_inputs_sha256"),
            "/producer/native_build_inputs_sha256",
            "provenance",
            lambda value: _sha256(
                value, description="producer.native_build_inputs_sha256"
            ),
        ),
        (
            ("producer", "version"),
            "/producer/version",
            "provenance",
            lambda value: _nonempty_string(value, description="producer.version"),
        ),
        (
            ("extensions", "generation", "phase_timings_seconds"),
            "/extensions/generation/phase_timings_seconds",
            "timing",
            lambda value: dict(_timings(value, description="generation phase timings")),
        ),
    )
    for path, json_path, category, validate in scalar_fields:
        _replace_pair(
            left,
            right,
            path,
            file="artifact.json",
            json_path=json_path,
            category=category,
            validate=validate,
            allowed=allowed,
        )

    left_generation = left["extensions"]["generation"]
    right_generation = right["extensions"]["generation"]
    left_profiles = left_generation.get(
        "recurrence_schedule_profiles",
        _MISSING,
    )
    right_profiles = right_generation.get(
        "recurrence_schedule_profiles",
        _MISSING,
    )
    _record_allowed(
        allowed,
        file="artifact.json",
        json_path="/extensions/generation/recurrence_schedule_profiles",
        category="generation-only-telemetry",
        baseline=left_profiles,
        candidate=right_profiles,
    )
    left_generation["recurrence_schedule_profiles"] = _NORMALIZED
    right_generation["recurrence_schedule_profiles"] = _NORMALIZED
    left_concrete = _records_by_id(
        left_generation.get("concrete_processes"),
        description="baseline concrete processes",
    )
    right_concrete = _records_by_id(
        right_generation.get("concrete_processes"),
        description="candidate concrete processes",
    )
    for process_id in sorted(set(left_concrete) & set(right_concrete)):
        execution_path = f"processes/{process_id}/execution.json"
        if execution_path not in matched_execution_paths:
            continue
        left_record = left_concrete[process_id]
        right_record = right_concrete[process_id]
        left_digest = _sha256(
            left_record.get("execution_manifest_sha256"),
            description=f"concrete process {process_id} execution digest",
        )
        right_digest = _sha256(
            right_record.get("execution_manifest_sha256"),
            description=f"concrete process {process_id} execution digest",
        )
        _record_allowed(
            allowed,
            file="artifact.json",
            json_path=(
                "/extensions/generation/concrete_processes"
                f"[id={process_id}]/execution_manifest_sha256"
            ),
            category="derived-execution-metadata",
            baseline=left_digest,
            candidate=right_digest,
        )
        left_record["execution_manifest_sha256"] = _NORMALIZED
        right_record["execution_manifest_sha256"] = _NORMALIZED

    left_payloads = _payload_records_by_path(left.get("payloads"), "baseline")
    right_payloads = _payload_records_by_path(right.get("payloads"), "candidate")
    for path in sorted(
        matched_execution_paths & set(left_payloads) & set(right_payloads)
    ):
        left_record = left_payloads[path]
        right_record = right_payloads[path]
        for key, validator in (
            (
                "sha256",
                lambda value, payload_path=path: _sha256(
                    value, description=f"payload {payload_path}.sha256"
                ),
            ),
            (
                "size_bytes",
                lambda value, payload_path=path: _nonnegative_int(
                    value, description=f"payload {payload_path}.size_bytes"
                ),
            ),
        ):
            left_value = validator(left_record.get(key))
            right_value = validator(right_record.get(key))
            _record_allowed(
                allowed,
                file="artifact.json",
                json_path=f"/payloads[path={path}]/{key}",
                category="derived-execution-metadata",
                baseline=left_value,
                candidate=right_value,
            )
            left_record[key] = _NORMALIZED
            right_record[key] = _NORMALIZED

    for path in sorted(
        matched_structural_proof_paths & set(left_payloads) & set(right_payloads)
    ):
        left_record = left_payloads[path]
        right_record = right_payloads[path]
        for key, validator in (
            (
                "sha256",
                lambda value, payload_path=path: _sha256(
                    value,
                    description=f"payload {payload_path}.sha256",
                ),
            ),
            (
                "size_bytes",
                lambda value, payload_path=path: _nonnegative_int(
                    value,
                    description=f"payload {payload_path}.size_bytes",
                ),
            ),
        ):
            left_value = validator(left_record.get(key))
            right_value = validator(right_record.get(key))
            _record_allowed(
                allowed,
                file="artifact.json",
                json_path=f"/payloads[path={path}]/{key}",
                category="derived-structural-proof-metadata",
                baseline=left_value,
                candidate=right_value,
            )
            left_record[key] = _NORMALIZED
            right_record[key] = _NORMALIZED

    return left, right


def _raise_invalid(description: str) -> None:
    raise ComparisonError(f"{description} is invalid")


def _records_by_id(value: object, *, description: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(_sequence(value, description=description)):
        if not isinstance(raw_record, dict):
            raise ComparisonError(f"{description}[{index}] must be an object")
        process_id = _nonempty_string(
            raw_record.get("id"), description=f"{description}[{index}].id"
        )
        if process_id in records:
            raise ComparisonError(f"{description} has duplicate id {process_id}")
        records[process_id] = raw_record
    return records


def _payload_records_by_path(value: object, label: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    description = f"{label} manifest payloads"
    for index, raw_record in enumerate(_sequence(value, description=description)):
        if not isinstance(raw_record, dict):
            raise ComparisonError(f"{description}[{index}] must be an object")
        path = _payload_relative_path(
            raw_record.get("path"), description=f"{description}[{index}]"
        )
        if path in records:
            raise ComparisonError(f"{description} has duplicate path {path}")
        records[path] = raw_record
    return records


def _normalize_execution_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    path: str,
    allowed: list[dict[str, object]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    left = copy.deepcopy(dict(baseline))
    right = copy.deepcopy(dict(candidate))
    _replace_pair(
        left,
        right,
        ("plan", "inspection_summary", "generation_timings_seconds"),
        file=path,
        json_path="/plan/inspection_summary/generation_timings_seconds",
        category="timing",
        validate=lambda value: dict(
            _timings(value, description=f"{path} generation timings")
        ),
        allowed=allowed,
    )
    return left, right


def compare_artifacts(
    baseline_artifact: Path, candidate_artifact: Path
) -> dict[str, object]:
    """Authenticate and compare two recurrence artifacts."""

    baseline = _load_artifact(baseline_artifact, label="baseline")
    candidate = _load_artifact(candidate_artifact, label="candidate")
    unknown: list[dict[str, object]] = []
    allowed: list[dict[str, object]] = []
    payload_comparisons: list[dict[str, object]] = []

    baseline_paths = set(baseline.payloads)
    candidate_paths = set(candidate.payloads)
    inventories_match = baseline_paths == candidate_paths
    for path in sorted(baseline_paths - candidate_paths):
        unknown.append(
            {
                "kind": "payload-inventory",
                "path": path,
                "baseline": "present",
                "candidate": "missing",
            }
        )
    for path in sorted(candidate_paths - baseline_paths):
        unknown.append(
            {
                "kind": "payload-inventory",
                "path": path,
                "baseline": "missing",
                "candidate": "present",
            }
        )

    common_paths = baseline_paths & candidate_paths
    exact_payload_bytes_match = inventories_match
    payloads_match_policy = inventories_match
    pacbin_comparisons: dict[str, RecurrencePacbinComparison] = {}

    for path in sorted(
        set(baseline.recurrence_schedules) & set(candidate.recurrence_schedules)
    ):
        left = baseline.payloads[path]
        right = candidate.payloads[path]
        comparison = _compare_recurrence_pacbins(
            baseline.recurrence_schedules[path],
            candidate.recurrence_schedules[path],
            relative_path=path,
            unknown=unknown,
        )
        pacbin_comparisons[path] = comparison
        exact_payload_bytes_match = (
            exact_payload_bytes_match and comparison.exact_bytes_match
        )
        payloads_match_policy = payloads_match_policy and comparison.policy_match
        payload_comparisons.append(
            {
                "path": path,
                "comparison": "recurrence-pacbin-exact-plan-and-projection-body",
                "baseline_sha256": left.sha256,
                "candidate_sha256": right.sha256,
                "byte_for_byte_match": comparison.exact_bytes_match,
                "runtime_plan_bytes_match": comparison.plan_bytes_match,
                "projection_certificate_body_match": (
                    comparison.projection_certificate_bodies_match
                ),
                "matches_policy": comparison.policy_match,
            }
        )

    schedule_index_differences = _json_differences(
        baseline.schedule_index,
        candidate.schedule_index,
        file=_SCHEDULE_INDEX_PATH,
    )
    unknown.extend(schedule_index_differences)
    left_schedule_index_payload = baseline.payloads[_SCHEDULE_INDEX_PATH]
    right_schedule_index_payload = candidate.payloads[_SCHEDULE_INDEX_PATH]
    schedule_index_bytes_match = (
        left_schedule_index_payload.sha256 == right_schedule_index_payload.sha256
        and left_schedule_index_payload.size_bytes
        == right_schedule_index_payload.size_bytes
        and _files_equal(
            left_schedule_index_payload.path,
            right_schedule_index_payload.path,
        )
    )
    if not schedule_index_bytes_match:
        unknown.append(
            {
                "kind": "payload-bytes",
                "path": _SCHEDULE_INDEX_PATH,
                "baseline": {
                    "sha256": left_schedule_index_payload.sha256,
                    "size_bytes": left_schedule_index_payload.size_bytes,
                },
                "candidate": {
                    "sha256": right_schedule_index_payload.sha256,
                    "size_bytes": right_schedule_index_payload.size_bytes,
                },
            }
        )
    schedule_index_matches = (
        schedule_index_bytes_match and not schedule_index_differences
    )
    exact_payload_bytes_match = exact_payload_bytes_match and schedule_index_bytes_match
    payloads_match_policy = payloads_match_policy and schedule_index_matches
    payload_comparisons.append(
        {
            "path": _SCHEDULE_INDEX_PATH,
            "comparison": "byte-for-byte",
            "baseline_sha256": left_schedule_index_payload.sha256,
            "candidate_sha256": right_schedule_index_payload.sha256,
            "byte_for_byte_match": schedule_index_bytes_match,
            "matches": schedule_index_matches,
        }
    )

    matched_execution_paths: set[str] = set()
    normalized_execution_pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for path in sorted(set(baseline.executions) & set(candidate.executions)):
        left_payload = baseline.payloads[path]
        right_payload = candidate.payloads[path]
        execution_bytes_match = (
            left_payload.sha256 == right_payload.sha256
            and left_payload.size_bytes == right_payload.size_bytes
            and _files_equal(left_payload.path, right_payload.path)
        )
        exact_payload_bytes_match = exact_payload_bytes_match and execution_bytes_match
        left_json, right_json = _normalize_execution_pair(
            baseline.executions[path],
            candidate.executions[path],
            path=path,
            allowed=allowed,
        )
        differences = _json_differences(left_json, right_json, file=path)
        unknown.extend(differences)
        execution_matches = not differences
        payloads_match_policy = payloads_match_policy and execution_matches
        if execution_matches:
            matched_execution_paths.add(path)
            normalized_execution_pairs[path] = (left_json, right_json)
        payload_comparisons.append(
            {
                "path": path,
                "comparison": "json-exact-except-timing",
                "baseline_sha256": left_payload.sha256,
                "candidate_sha256": right_payload.sha256,
                "byte_for_byte_match": execution_bytes_match,
                "normalized_match": execution_matches,
            }
        )

    matched_structural_proof_paths: set[str] = set()
    baseline_structural_proofs = {
        path for path in baseline.payloads if _STRUCTURAL_PROOF_PATH_RE.fullmatch(path)
    }
    candidate_structural_proofs = {
        path for path in candidate.payloads if _STRUCTURAL_PROOF_PATH_RE.fullmatch(path)
    }
    for path in sorted(baseline_structural_proofs & candidate_structural_proofs):
        left_payload = baseline.payloads[path]
        right_payload = candidate.payloads[path]
        proof_bytes_match = (
            left_payload.sha256 == right_payload.sha256
            and left_payload.size_bytes == right_payload.size_bytes
            and _files_equal(left_payload.path, right_payload.path)
        )
        exact_payload_bytes_match = exact_payload_bytes_match and proof_bytes_match
        left_proof = _json_object(
            left_payload.path,
            description="baseline recurrence structural source proof",
        )
        right_proof = _json_object(
            right_payload.path,
            description="candidate recurrence structural source proof",
        )
        left_execution_path, _ = _validate_structural_proof(
            baseline,
            left_proof,
            relative_path=path,
            label="baseline",
        )
        right_execution_path, _ = _validate_structural_proof(
            candidate,
            right_proof,
            relative_path=path,
            label="candidate",
        )
        execution_pair = normalized_execution_pairs.get(left_execution_path)
        if left_execution_path == right_execution_path and execution_pair is not None:
            left_normalized, right_normalized = _normalize_structural_proof_pair(
                baseline,
                candidate,
                left_proof,
                right_proof,
                relative_path=path,
                baseline_execution=execution_pair[0],
                candidate_execution=execution_pair[1],
                allowed=allowed,
            )
        else:
            left_normalized = left_proof
            right_normalized = right_proof
        differences = _json_differences(
            left_normalized,
            right_normalized,
            file=path,
        )
        unknown.extend(differences)
        proof_matches = not differences
        payloads_match_policy = payloads_match_policy and proof_matches
        if proof_matches:
            matched_structural_proof_paths.add(path)
        payload_comparisons.append(
            {
                "path": path,
                "comparison": (
                    "json-exact-except-authenticated-provenance-and-"
                    "derived-execution-metadata"
                ),
                "baseline_sha256": left_payload.sha256,
                "candidate_sha256": right_payload.sha256,
                "byte_for_byte_match": proof_bytes_match,
                "normalized_match": proof_matches,
            }
        )

    special_paths = {
        *baseline.recurrence_schedules,
        *candidate.recurrence_schedules,
        _SCHEDULE_INDEX_PATH,
        *baseline.executions,
        *candidate.executions,
        *baseline_structural_proofs,
        *candidate_structural_proofs,
    }
    for path in sorted(common_paths - special_paths):
        left = baseline.payloads[path]
        right = candidate.payloads[path]
        bytes_match = (
            left.sha256 == right.sha256
            and left.size_bytes == right.size_bytes
            and _files_equal(left.path, right.path)
        )
        exact_payload_bytes_match = exact_payload_bytes_match and bytes_match
        payloads_match_policy = payloads_match_policy and bytes_match
        if not bytes_match:
            unknown.append(
                {
                    "kind": "payload-bytes",
                    "path": path,
                    "baseline": {
                        "sha256": left.sha256,
                        "size_bytes": left.size_bytes,
                    },
                    "candidate": {
                        "sha256": right.sha256,
                        "size_bytes": right.size_bytes,
                    },
                }
            )
        payload_comparisons.append(
            {
                "path": path,
                "comparison": "byte-for-byte",
                "baseline_sha256": left.sha256,
                "candidate_sha256": right.sha256,
                "size_bytes": left.size_bytes if bytes_match else None,
                "matches": bytes_match,
            }
        )

    left_manifest, right_manifest = _normalize_manifest_pair(
        baseline.manifest,
        candidate.manifest,
        matched_execution_paths=matched_execution_paths,
        matched_structural_proof_paths=matched_structural_proof_paths,
        allowed=allowed,
    )
    manifest_differences = _json_differences(
        left_manifest, right_manifest, file="artifact.json"
    )
    unknown.extend(manifest_differences)

    unknown_count = len(unknown)
    report_unknown = unknown[:MAX_REPORTED_UNKNOWN_DIFFERENCES]
    execution_metadata_match = not any(
        difference.get("file", "").startswith("processes/")
        for difference in unknown
        if isinstance(difference.get("file"), str)
    )
    manifest_metadata_match = not manifest_differences
    runtime_plans_match = set(baseline.recurrence_schedules) == set(
        candidate.recurrence_schedules
    ) and all(item.plan_bytes_match for item in pacbin_comparisons.values())
    projection_certificate_bodies_match = set(baseline.recurrence_schedules) == set(
        candidate.recurrence_schedules
    ) and all(
        item.projection_certificate_bodies_match for item in pacbin_comparisons.values()
    )
    passes = (
        inventories_match
        and payloads_match_policy
        and execution_metadata_match
        and manifest_metadata_match
        and runtime_plans_match
        and projection_certificate_bodies_match
        and unknown_count == 0
    )
    return {
        "kind": COMPARISON_KIND,
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "passes": passes,
        "baseline": {
            "path": str(baseline.root),
            "artifact_id": baseline.manifest["artifact_id"],
            "manifest_sha256": baseline.manifest_sha256,
            "payload_count": len(baseline.payloads),
        },
        "candidate": {
            "path": str(candidate.root),
            "artifact_id": candidate.manifest["artifact_id"],
            "manifest_sha256": candidate.manifest_sha256,
            "payload_count": len(candidate.payloads),
        },
        "policy": {
            "ordinary_payloads": "byte-for-byte-and-sha256-v1",
            "recurrence_runtime_pacbins": "exact-plan-and-projection-body-v1",
            "schedule_index": "exact-v1",
            "artifact_manifest": "exact-except-enumerated-metadata-v2",
            "execution_manifests": "exact-except-generation-timings-v2",
            "structural_source_proofs": (
                "exact-except-authenticated-provenance-and-derived-execution-"
                "metadata-v1"
            ),
            "artifact_allowed_metadata_paths": list(ARTIFACT_ALLOWED_METADATA_PATHS),
            "execution_allowed_metadata_paths": list(EXECUTION_ALLOWED_METADATA_PATHS),
            "structural_proof_allowed_metadata_paths": list(
                STRUCTURAL_PROOF_ALLOWED_METADATA_PATHS
            ),
        },
        "summary": {
            "payload_inventories_match": inventories_match,
            "exact_payload_bytes_match": exact_payload_bytes_match,
            "payloads_match_policy": payloads_match_policy,
            "execution_semantics_match": execution_metadata_match,
            "manifest_semantics_match": manifest_metadata_match,
            "runtime_schedule_plan_bytes_match": runtime_plans_match,
            "projection_certificate_semantic_bodies_match": (
                projection_certificate_bodies_match
            ),
            "runtime_bearing_payloads_match": passes,
        },
        "payload_comparisons": payload_comparisons,
        "allowed_metadata_difference_count": len(allowed),
        "allowed_metadata_differences": allowed,
        "unknown_difference_count": unknown_count,
        "unknown_differences_truncated": unknown_count > len(report_unknown),
        "unknown_differences": report_unknown,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline", type=Path, required=True)
    result.add_argument("--candidate", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        report = compare_artifacts(arguments.baseline, arguments.candidate)
    except ComparisonError as error:
        print(f"recurrence-artifact-compare: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
