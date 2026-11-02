# SPDX-License-Identifier: 0BSD
"""Filesystem and digest checks for schema-v3 process artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath

from pyamplicol.api.errors import ArtifactError


def normalize_relative_path(value: str) -> str:
    """Return a normalized artifact path or reject unsafe input."""

    if not isinstance(value, str) or not value:
        raise ArtifactError("artifact payload paths must be non-empty strings")
    if "\\" in value or "\x00" in value:
        raise ArtifactError(f"artifact payload path is not portable: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactError(f"artifact payload path is not confined: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise ArtifactError(
            f"artifact payload path is not normalized: {value!r} != {normalized!r}"
        )
    return normalized


def confined_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    """Resolve a payload while rejecting symlinks and root escapes."""

    normalized = normalize_relative_path(relative)
    root = root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    current = root
    for part in PurePosixPath(normalized).parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if must_exist:
                raise ArtifactError(
                    f"artifact payload is missing: {normalized}"
                ) from None
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactError(
                f"artifact payload may not traverse a symlink: {normalized}"
            )
    if must_exist and not candidate.is_file():
        raise ArtifactError(f"artifact payload is not a regular file: {normalized}")
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"artifact payload escapes its root: {normalized}") from exc
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def executable_bit(path: Path) -> bool:
    return bool(path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "confined_path",
    "executable_bit",
    "fsync_directory",
    "fsync_file",
    "normalize_relative_path",
    "sha256_file",
]
