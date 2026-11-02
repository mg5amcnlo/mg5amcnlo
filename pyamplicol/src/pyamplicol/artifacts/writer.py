# SPDX-License-Identifier: 0BSD
"""High-level schema-v3 artifact builder."""

from __future__ import annotations

import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal

from .manifest import (
    ARTIFACT_IDENTITY_EXTENSION,
    MANIFEST_NAME,
    PayloadRecord,
    artifact_identity_extension,
    canonical_manifest_bytes,
    compute_artifact_id,
    load_manifest,
)
from .security import (
    confined_path,
    executable_bit,
    fsync_file,
    normalize_relative_path,
    sha256_file,
)
from .transaction import ArtifactTransaction, ArtifactWriteMode


class ArtifactBuilder:
    """Build one artifact in a private staging directory."""

    def __init__(
        self,
        destination: str | Path,
        *,
        mode: ArtifactWriteMode = "error",
        expected_artifact_id: str | None = None,
    ) -> None:
        if expected_artifact_id is not None and mode != "append":
            raise ValueError("expected_artifact_id is valid only for append mode")
        self._transaction = ArtifactTransaction(destination, mode=mode)
        self._expected_artifact_id = expected_artifact_id
        self.root: Path | None = None
        self._payloads: dict[str, PayloadRecord] = {}

    def __enter__(self) -> ArtifactBuilder:
        self.root = self._transaction.__enter__()
        try:
            if self._transaction.mode == "append":
                existing = self.root / MANIFEST_NAME
                if existing.is_file():
                    manifest = load_manifest(self.root)
                    if (
                        self._expected_artifact_id is not None
                        and manifest.artifact_id != self._expected_artifact_id
                    ):
                        raise ValueError(
                            "append artifact changed before the transaction lock was "
                            "acquired; retry the append"
                        )
                    self._payloads.update(
                        (record.path, record) for record in manifest.payloads
                    )
                    existing.unlink()
            return self
        except BaseException as error:
            self._transaction.__exit__(type(error), error, error.__traceback__)
            self.root = None
            raise

    def _root(self) -> Path:
        if self.root is None:
            raise RuntimeError("artifact builder is not active")
        return self.root

    def staged_path(self, relative: str, *, create_parent: bool = False) -> Path:
        """Return a confined path in the private staging directory.

        This is intended for producers that already provide their own atomic,
        streaming writer.  Call :meth:`register_staged_file` after the file is
        complete and independently validated.
        """

        path_value = normalize_relative_path(relative)
        if path_value == MANIFEST_NAME:
            raise ValueError(f"{MANIFEST_NAME} is reserved for the artifact manifest")
        path = confined_path(self._root(), path_value, must_exist=False)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def register_staged_file(
        self,
        relative: str,
        *,
        role: str,
        media_type: str,
        executable: bool | None = None,
        target: Mapping[str, object] | None = None,
        process_id: str | None = None,
    ) -> PayloadRecord:
        """Register a complete regular file already present in staging."""

        path_value = normalize_relative_path(relative)
        path = self.staged_path(path_value)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"staged artifact payload must be a regular file: {path}")
        if executable is not None:
            path.chmod(0o755 if executable else 0o644)
        fsync_file(path)
        record = PayloadRecord(
            path=path_value,
            role=role,
            media_type=media_type,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
            executable=executable_bit(path),
            target=dict(target) if target is not None else None,
            process_id=process_id,
        )
        self._payloads[path_value] = record
        return record

    def add_bytes(
        self,
        relative: str,
        content: bytes,
        *,
        role: str,
        media_type: str,
        executable: bool = False,
        target: Mapping[str, object] | None = None,
        process_id: str | None = None,
    ) -> PayloadRecord:
        return self.add_stream(
            relative,
            io.BytesIO(content),
            role=role,
            media_type=media_type,
            executable=executable,
            target=target,
            process_id=process_id,
        )

    def add_stream(
        self,
        relative: str,
        source: BinaryIO,
        *,
        role: str,
        media_type: str,
        executable: bool = False,
        target: Mapping[str, object] | None = None,
        process_id: str | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> PayloadRecord:
        """Copy a binary stream into staging with bounded memory."""

        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError("artifact stream chunk size must be a positive integer")
        read = getattr(source, "read", None)
        if not callable(read):
            raise TypeError("artifact stream source must provide read()")
        path_value = normalize_relative_path(relative)
        path = self.staged_path(path_value, create_parent=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        descriptor_open = True
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor_open = False
                while True:
                    chunk = source.read(chunk_size)
                    if not isinstance(chunk, bytes | bytearray | memoryview):
                        raise TypeError("artifact stream source must return bytes")
                    if not chunk:
                        break
                    if len(chunk) > chunk_size:
                        raise ValueError(
                            "artifact stream source returned more bytes than requested"
                        )
                    view = memoryview(chunk)
                    written = 0
                    while written < len(view):
                        count = stream.write(view[written:])
                        if not isinstance(count, int) or count <= 0:
                            raise OSError("short write while staging artifact payload")
                        written += count
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o755 if executable else 0o644)
            os.replace(temporary, path)
        finally:
            if descriptor_open:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                temporary.unlink()
        return self.register_staged_file(
            path_value,
            role=role,
            media_type=media_type,
            executable=executable,
            target=target,
            process_id=process_id,
        )

    def add_json(
        self,
        relative: str,
        value: object,
        *,
        role: str,
        process_id: str | None = None,
        compact: bool = False,
    ) -> PayloadRecord:
        content = (
            json.dumps(
                value,
                indent=None if compact else 2,
                separators=(",", ":") if compact else None,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        return self.add_bytes(
            relative,
            content,
            role=role,
            media_type="application/json",
            process_id=process_id,
        )

    def add_file(
        self,
        relative: str,
        source: str | Path,
        *,
        role: str,
        media_type: str,
        executable: bool | None = None,
        target: Mapping[str, object] | None = None,
        process_id: str | None = None,
    ) -> PayloadRecord:
        requested_source = Path(source).expanduser()
        if requested_source.is_symlink():
            raise ValueError(
                f"artifact source must be a regular file: {requested_source}"
            )
        source_path = requested_source.resolve(strict=True)
        if not source_path.is_file():
            raise ValueError(f"artifact source must be a regular file: {source_path}")
        with source_path.open("rb") as stream:
            return self.add_stream(
                relative,
                stream,
                role=role,
                media_type=media_type,
                executable=(
                    executable_bit(source_path) if executable is None else executable
                ),
                target=target,
                process_id=process_id,
            )

    def discard_payloads(self, relative: str, *, recursive: bool = False) -> None:
        """Remove an owned payload or payload subtree from the staging snapshot."""

        path_value = normalize_relative_path(relative)
        if path_value == MANIFEST_NAME:
            raise ValueError(f"{MANIFEST_NAME} is reserved for the artifact manifest")
        prefix = path_value.rstrip("/") + "/"
        owned = tuple(
            path
            for path in self._payloads
            if path == path_value or (recursive and path.startswith(prefix))
        )
        for owned_path in owned:
            path = confined_path(self._root(), owned_path, must_exist=False)
            if path.is_symlink() or path.is_dir():
                raise ValueError(
                    f"artifact payload must be a regular file: {owned_path}"
                )
            if path.exists():
                path.unlink()
            self._payloads.pop(owned_path, None)
        if recursive:
            root = confined_path(self._root(), path_value, must_exist=False)
            if root.is_symlink():
                raise ValueError(
                    f"artifact payload path must not be a symlink: {path_value}"
                )
            if root.is_dir():
                directories = sorted(
                    (path for path in root.rglob("*") if path.is_dir()),
                    key=lambda path: len(path.parts),
                    reverse=True,
                )
                for directory in (*directories, root):
                    with suppress(OSError):
                        directory.rmdir()

    def payload_records(self) -> tuple[PayloadRecord, ...]:
        """Return the immutable, path-ordered staged payload inventory.

        Derived runtime caches use this snapshot to bind their compact native
        representation to the ordinary artifact payload declarations.  The
        cache is added only after the snapshot is encoded, so it cannot create
        a self-hash cycle; normal finalization still includes the cache record
        in the authoritative runtime artifact identity.
        """

        self._root()
        return tuple(sorted(self._payloads.values(), key=lambda item: item.path))

    def finalize(
        self,
        *,
        kind: Literal["pyamplicol-process", "pyamplicol-process-set"],
        producer: Mapping[str, object],
        model: Mapping[str, object],
        configuration: Mapping[str, object],
        processes: Sequence[Mapping[str, object]],
        runtime: Mapping[str, object],
        dependencies: Sequence[Mapping[str, object]] = (),
        default_process_id: str | None = None,
        extensions: Mapping[str, object] | None = None,
    ) -> Path:
        if not self._payloads:
            raise ValueError("an artifact must contain at least one payload")
        manifest_extensions = dict(extensions or {})
        supplied_identity = manifest_extensions.get(ARTIFACT_IDENTITY_EXTENSION)
        expected_identity = artifact_identity_extension()
        if supplied_identity is not None and supplied_identity != expected_identity:
            raise ValueError(
                "extensions.artifact_identity is compiler-owned and must use the "
                "current runtime-payload identity contract"
            )
        manifest_extensions[ARTIFACT_IDENTITY_EXTENSION] = expected_identity
        manifest: dict[str, object] = {
            "schema_version": 3,
            "kind": kind,
            "artifact_id": "0" * 64,
            "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "producer": dict(producer),
            "model": dict(model),
            "configuration": dict(configuration),
            "processes": [dict(process) for process in processes],
            "runtime": dict(runtime),
            "payloads": [
                record.as_dict()
                for record in sorted(
                    self._payloads.values(),
                    key=lambda item: item.path,
                )
            ],
            "dependencies": [dict(dependency) for dependency in dependencies],
            "extensions": manifest_extensions,
        }
        if default_process_id is not None:
            manifest["default_process_id"] = default_process_id
        manifest["artifact_id"] = compute_artifact_id(manifest)
        path = self._root() / MANIFEST_NAME
        path.write_bytes(canonical_manifest_bytes(manifest))
        path.chmod(0o644)
        fsync_file(path)
        return path

    def __exit__(self, *exception: object) -> bool:
        if exception[0] is None and not (self._root() / MANIFEST_NAME).is_file():
            raise RuntimeError("artifact builder exited without finalize()")
        return self._transaction.__exit__(*exception)  # type: ignore[arg-type]


__all__ = ["ArtifactBuilder"]
