# SPDX-License-Identifier: 0BSD
"""Artifact-scoped access to retained exact evaluator states."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from pyamplicol.api.errors import ArtifactError, CompatibilityError
from pyamplicol.artifacts.manifest import ArtifactManifest, PayloadRecord
from pyamplicol.artifacts.security import confined_path, normalize_relative_path
from pyamplicol.generation.evaluator_container import (
    PacbinError,
    PacbinMember,
    PacbinMemberKind,
    PacbinReader,
)

_EXTENSION_NAME = "evaluator_payload_container"
_CONTAINER_KIND = "pyamplicol-evaluator-payload-container"
_CONTAINER_SCHEMA_VERSION = 1
_CONTAINER_STORAGE_ABI = "pacbin-v1"


class ExactEvaluatorPayloadResolver:
    """Resolve loose or packed Symbolica evaluator states by artifact path.

    The pacbin index is parsed once. The parsing reader is closed before the
    resolver is returned; each member read uses its own bounded file stream.
    Consequently the resolver owns no persistent descriptor and can safely be
    retained by a lazy exact executor.
    """

    def __init__(self, manifest: ArtifactManifest) -> None:
        self._root = manifest.root
        self._loose = {record.path: record for record in manifest.payloads}
        self._container_path: Path | None = None
        self._members: Mapping[str, PacbinMember] = {}

        extension = manifest.extensions.get(_EXTENSION_NAME)
        if extension is None:
            return
        if not isinstance(extension, Mapping):
            raise ArtifactError(
                f"extensions.{_EXTENSION_NAME} must be an object"
            )
        required = {
            "kind",
            "schema_version",
            "storage_abi",
            "path",
            "member_count",
            "unpacked_size_bytes",
            "index_sha256",
        }
        if set(extension) != required:
            raise ArtifactError(
                f"extensions.{_EXTENSION_NAME} has invalid fields"
            )
        if (
            extension.get("kind") != _CONTAINER_KIND
            or extension.get("schema_version") != _CONTAINER_SCHEMA_VERSION
            or extension.get("storage_abi") != _CONTAINER_STORAGE_ABI
        ):
            raise CompatibilityError(
                "unsupported evaluator payload container contract"
            )
        relative = extension.get("path")
        if not isinstance(relative, str):
            raise ArtifactError("evaluator payload container path is not a string")
        relative = normalize_relative_path(relative)
        record = self._loose.get(relative)
        if record is None:
            raise ArtifactError("evaluator payload container is undeclared")
        _validate_container_record(record)
        container_path = confined_path(self._root, relative)
        try:
            with PacbinReader.open(container_path, verify_payloads=False) as reader:
                index = reader.index
        except (OSError, PacbinError) as exc:
            raise ArtifactError(
                f"could not open evaluator payload container {relative}: {exc}"
            ) from exc

        member_count = extension.get("member_count")
        unpacked_size = extension.get("unpacked_size_bytes")
        index_sha256 = extension.get("index_sha256")
        if (
            isinstance(member_count, bool)
            or not isinstance(member_count, int)
            or member_count != len(index.members)
            or isinstance(unpacked_size, bool)
            or not isinstance(unpacked_size, int)
            or unpacked_size != sum(member.length for member in index.members)
            or not isinstance(index_sha256, str)
            or index_sha256 != index.index_sha256
        ):
            raise ArtifactError(
                "evaluator payload container metadata does not match its index"
            )
        self._container_path = container_path
        self._members = {member.logical_path: member for member in index.members}

    def require_exact_state(
        self,
        logical_path: str,
        *,
        process_id: str | None,
    ) -> None:
        """Require one declared loose or packed exact evaluator state."""

        normalized = normalize_relative_path(logical_path)
        record = self._loose.get(normalized)
        if record is not None:
            if record.role != "evaluator-state" or record.process_id != process_id:
                raise ArtifactError(
                    f"exact evaluator payload {normalized} has role/process "
                    f"{record.role!r}/{record.process_id!r}, expected "
                    f"'evaluator-state'/{process_id!r}"
                )
            return
        member = self._members.get(normalized)
        if member is None:
            raise ArtifactError(
                f"exact evaluator payload is absent: {normalized}"
            )
        if process_id is not None:
            process_prefix = normalize_relative_path(
                f"processes/{process_id}"
            ) + "/"
            if not normalized.startswith(process_prefix):
                raise ArtifactError(
                    f"packed exact evaluator payload {normalized} does not belong "
                    f"to process {process_id!r}"
                )
        if member.kind is not PacbinMemberKind.SYMBOLICA_EXACT_STATE:
            raise ArtifactError(
                f"packed evaluator payload {normalized} is not an exact state"
            )

    def read_exact_state(
        self,
        logical_path: str,
        *,
        process_id: str | None,
    ) -> bytes:
        """Read and authenticate one retained exact evaluator state."""

        normalized = normalize_relative_path(logical_path)
        self.require_exact_state(normalized, process_id=process_id)
        record = self._loose.get(normalized)
        if record is not None:
            try:
                return confined_path(self._root, normalized).read_bytes()
            except OSError as exc:
                raise CompatibilityError(
                    f"could not read retained Symbolica evaluator state "
                    f"{normalized}: {exc}"
                ) from exc

        member = self._members[normalized]
        if self._container_path is None:
            raise ArtifactError("packed evaluator payload has no container")
        try:
            with self._container_path.open("rb") as stream:
                stream.seek(member.offset)
                state = stream.read(member.length)
        except OSError as exc:
            raise CompatibilityError(
                f"could not read packed Symbolica evaluator state "
                f"{normalized}: {exc}"
            ) from exc
        if len(state) != member.length:
            raise ArtifactError(
                f"packed evaluator payload is truncated: {normalized}"
            )
        if hashlib.sha256(state).hexdigest() != member.sha256:
            raise ArtifactError(
                f"packed evaluator payload digest mismatch: {normalized}"
            )
        return state


def _validate_container_record(record: PayloadRecord) -> None:
    if (
        record.role != "evaluator-state"
        or record.media_type != "application/octet-stream"
        or record.process_id is not None
    ):
        raise ArtifactError("evaluator payload container record is invalid")


__all__ = ["ExactEvaluatorPayloadResolver"]
