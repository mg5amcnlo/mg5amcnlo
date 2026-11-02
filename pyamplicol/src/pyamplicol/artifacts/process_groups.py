# SPDX-License-Identifier: 0BSD
"""Read and update the process-group map stored in an artifact.

``process-groups.json`` is a declared payload whose role is not part of the
runtime-payload identity: rewriting it updates its payload record in
``artifact.json`` but never ``artifact_id``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

from pyamplicol.api.errors import ArtifactError
from pyamplicol.generation.process_grouping import (
    PROCESS_GROUPS_PATH,
    PROCESS_GROUPS_ROLE,
    GroupingProcess,
    GroupMember,
    ProcessGroupMap,
)

from .manifest import (
    MANIFEST_NAME,
    PayloadRecord,
    canonical_manifest_bytes,
    load_manifest,
)
from .security import confined_path, fsync_directory

PROCESS_GROUPS_MEDIA_TYPE = "application/json"


def grouping_process(
    process_id: str,
    expression: str,
    external_pdgs: Sequence[int],
) -> GroupingProcess:
    """Return the grouping view of one generated artifact process."""

    initial, _separator, _final = expression.partition(">")
    return GroupingProcess(
        process_id=process_id,
        expression=expression,
        pdgs=tuple(int(pdg) for pdg in external_pdgs),
        initial_count=len(initial.split()),
    )


def initial_process_groups(
    processes: Iterable[tuple[str, str, Sequence[int]]],
    members: Mapping[str, Sequence[GroupMember]] | None = None,
) -> ProcessGroupMap:
    """Group each generated ``(id, expression, pdgs)`` with its known members.

    ``members`` holds, per representative ID, the relations recorded before
    generation (flavour-blind members); every other process maps to itself.
    """

    result = ProcessGroupMap()
    for process_id, expression, external_pdgs in processes:
        representative = GroupMember.identity(
            grouping_process(process_id, expression, external_pdgs)
        )
        result.add_group([representative, *(members or {}).get(process_id, ())])
    return result


def process_groups_payload(group_map: ProcessGroupMap) -> bytes:
    return group_map.to_json().encode("utf-8")


def validate_process_groups(
    group_map: ProcessGroupMap,
    process_ids: Collection[str],
) -> None:
    """Require exactly the artifact's processes to be generated members.

    Members that were not generated are evaluated through their representative,
    so, as the runtime requires, their factor must be positive.
    """

    generated = [
        member.process_id for member in group_map.members() if member.generated
    ]
    if len(generated) != len(set(generated)) or set(generated) != set(process_ids):
        missing = sorted(set(process_ids) - set(generated))
        unknown = sorted(set(generated) - set(process_ids))
        raise ArtifactError(
            "process-group map does not list each artifact process exactly once "
            f"as a generated member (missing: {missing}, not in artifact: {unknown})"
        )
    for member in group_map.members():
        if not member.generated and member.factor <= 0:
            raise ArtifactError(
                f"process-group member {member.process_id!r} is not generated, so "
                f"its factor {member.factor} must be positive"
            )


@contextmanager
def _artifact_lock(root: Path) -> Iterator[None]:
    # The lock file of ArtifactTransaction, which serializes generation.
    with root.with_name(f".{root.name}.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _replace_file(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def read_process_groups(artifact: str | Path) -> ProcessGroupMap | None:
    """Return the stored process-group map, or ``None`` when none is declared."""

    manifest = load_manifest(artifact)
    record = next(
        (item for item in manifest.payloads if item.role == PROCESS_GROUPS_ROLE),
        None,
    )
    if record is None:
        return None
    path = confined_path(manifest.root, record.path)
    try:
        return ProcessGroupMap.from_json(path.read_text(encoding="utf-8"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid process-group map {record.path}: {exc}") from exc


def write_process_groups(artifact: str | Path, group_map: ProcessGroupMap) -> None:
    """Replace ``process-groups.json`` and its payload record in place.

    The update runs under the artifact's generation lock. The payload file is
    replaced first and the manifest second, each atomically, so an interrupted
    update leaves a digest mismatch that the next update repairs.
    ``artifact_id`` does not change because the role is not an identity role.
    """

    root = Path(artifact).expanduser().resolve(strict=True)
    content = process_groups_payload(group_map)
    with _artifact_lock(root):
        manifest = load_manifest(root)
        validate_process_groups(
            group_map, [str(process["id"]) for process in manifest.processes]
        )
        manifest_path = confined_path(root, MANIFEST_NAME)
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload_path = confined_path(root, PROCESS_GROUPS_PATH, must_exist=False)
        _replace_file(payload_path, content)
        record = PayloadRecord(
            path=PROCESS_GROUPS_PATH,
            role=PROCESS_GROUPS_ROLE,
            media_type=PROCESS_GROUPS_MEDIA_TYPE,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            executable=False,
        )
        payloads = [
            item for item in raw["payloads"] if item.get("path") != PROCESS_GROUPS_PATH
        ]
        payloads.append(record.as_dict())
        raw["payloads"] = sorted(payloads, key=lambda item: str(item["path"]))
        _replace_file(manifest_path, canonical_manifest_bytes(raw))
        fsync_directory(root)


__all__ = [
    "PROCESS_GROUPS_MEDIA_TYPE",
    "grouping_process",
    "initial_process_groups",
    "process_groups_payload",
    "read_process_groups",
    "validate_process_groups",
    "write_process_groups",
]
