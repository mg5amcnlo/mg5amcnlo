# SPDX-License-Identifier: 0BSD
"""Portable binary containers for generated evaluator payloads.

``pacbin-v1`` is deliberately independent of Python serialization formats.  A
container consists of a fixed 64-byte header, 64-byte-aligned uncompressed
member payloads, a sorted variable-width index, and a fixed 64-byte footer.
Every integer is little-endian.  The footer authenticates the exact index
bytes, while every index entry authenticates its member payload with SHA-256.

This module implements the producer-side codec. Artifact publication uses it
to write one evaluator pack, while Rusticol implements the independent native
reader for runtime loading.
"""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
import unicodedata
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO

PACBIN_VERSION = 1
PACBIN_ALIGNMENT = 64

_HEADER_MAGIC = b"PACBIN\x00\x00"
_INDEX_MAGIC = b"PACIDX\x00\x00"
_FOOTER_MAGIC = b"PACEND\x00\x00"
_SUPPORTED_FLAGS = 0
_INDEX_ALIGNMENT = 8
_DEFAULT_CHUNK_SIZE = 1024 * 1024
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1

# magic, version, size, flags, alignment, reserved, index offset, member count,
# reserved.  Keeping the header exactly one payload-alignment unit makes the
# first member naturally aligned.
_HEADER_STRUCT = struct.Struct("<8sHHIIIQQ24s")
# magic, version, size, flags, member count, reserved
_INDEX_HEADER_STRUCT = struct.Struct("<8sHHIQQ")
# path byte count, kind, flags, payload offset, payload length, payload digest
_INDEX_ENTRY_STRUCT = struct.Struct("<IHHQQ32s")
# magic, version, size, flags, index offset, member count, index digest
_FOOTER_STRUCT = struct.Struct("<8sHHIQQ32s")


class PacbinError(ValueError):
    """A malformed container or invalid codec request."""


class PacbinMemberKind(IntEnum):
    """Portable evaluator payload kinds encoded by pacbin-v1."""

    SYMJIT_APPLICATION = 1
    SYMBOLICA_EXACT_STATE = 2
    NATIVE_LIBRARY = 3
    EAGER_RUNTIME_METADATA = 4
    EAGER_RUNTIME_TABLE = 5
    RECURRENCE_DIRECT_PLAN = 7
    RECURRENCE_COLOR_PROJECTION_CERTIFICATE = 8
    ON_THE_FLY_PROCESS_SEED = 9
    COLOR_CONTRACTION = 10


@dataclass(frozen=True, slots=True)
class PacbinMemberSource:
    """One logical member and the stream from which its bytes are copied.

    A path source is opened from its beginning.  A binary file object is read
    from its current position to EOF and is never closed by the writer.
    """

    logical_path: str
    kind: PacbinMemberKind
    source: str | os.PathLike[str] | BinaryIO

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "logical_path",
            normalize_logical_path(self.logical_path),
        )
        try:
            kind = PacbinMemberKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise PacbinError(f"unknown pacbin member kind: {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class PacbinMember:
    """Validated metadata for one indexed member."""

    logical_path: str
    kind: PacbinMemberKind
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PacbinIndex:
    """Validated top-level metadata returned by the writer or reader."""

    version: int
    index_offset: int
    index_sha256: str
    file_size: int
    members: tuple[PacbinMember, ...]

    def member(self, logical_path: str) -> PacbinMember:
        """Return one member by normalized logical path."""

        normalized = normalize_logical_path(logical_path)
        for member in self.members:
            if member.logical_path == normalized:
                return member
        raise KeyError(normalized)


def normalize_logical_path(value: str) -> str:
    """Return the canonical NFC POSIX path used in a pacbin index."""

    if not isinstance(value, str):
        raise TypeError("pacbin logical path must be a string")
    if not value or "\x00" in value:
        raise PacbinError("pacbin logical path must be non-empty and contain no NUL")
    if "\\" in value:
        raise PacbinError("pacbin logical path must use POSIX '/' separators")
    if value.startswith("/"):
        raise PacbinError("pacbin logical path must be relative")

    parts: list[str] = []
    for part in value.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            raise PacbinError("pacbin logical path must not contain '..'")
        parts.append(unicodedata.normalize("NFC", part))
    if not parts:
        raise PacbinError("pacbin logical path must name a member")
    return "/".join(parts)


def write_pacbin(
    destination: str | os.PathLike[str] | BinaryIO,
    members: Iterable[PacbinMemberSource],
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> PacbinIndex:
    """Stream ``members`` into one deterministic pacbin-v1 container.

    Member descriptors and index metadata are retained in memory, but payload
    contents are copied and hashed in bounded chunks.  File-object destinations
    must be seekable because the completed index location is written back into
    the header.
    """

    chunk_size = _checked_chunk_size(chunk_size)
    ordered = _ordered_sources(members)
    expected_index_size = _validate_index_bounds(ordered)
    with _open_destination(destination) as stream:
        _require_seekable(stream, "pacbin destination")
        stream.seek(0)
        stream.truncate(0)
        _write_all(stream, b"\x00" * _HEADER_STRUCT.size, "pacbin header")

        indexed_members: list[PacbinMember] = []
        for source in ordered:
            _write_alignment_padding(stream, PACBIN_ALIGNMENT)
            offset = _checked_u64_position(stream.tell(), "pacbin member offset")
            digest = hashlib.sha256()
            length = 0
            with _open_member_source(source.source) as member_stream:
                while True:
                    chunk = member_stream.read(chunk_size)
                    if not isinstance(chunk, bytes | bytearray | memoryview):
                        raise TypeError("pacbin member source must return bytes")
                    if not chunk:
                        break
                    if len(chunk) > chunk_size:
                        raise PacbinError(
                            "pacbin member source returned more bytes than requested"
                        )
                    if len(chunk) > _MAX_U64 - length:
                        raise PacbinError("pacbin member length exceeds u64")
                    length += len(chunk)
                    digest.update(chunk)
                    _write_all(stream, chunk, "pacbin member payload")
            indexed_members.append(
                PacbinMember(
                    logical_path=source.logical_path,
                    kind=source.kind,
                    offset=offset,
                    length=length,
                    sha256=digest.hexdigest(),
                )
            )

        _write_alignment_padding(stream, PACBIN_ALIGNMENT)
        index_offset = _checked_u64_position(stream.tell(), "pacbin index offset")
        index_digest = hashlib.sha256()
        _write_hashed(
            stream,
            index_digest,
            _INDEX_HEADER_STRUCT.pack(
                _INDEX_MAGIC,
                PACBIN_VERSION,
                _INDEX_HEADER_STRUCT.size,
                _SUPPORTED_FLAGS,
                len(indexed_members),
                0,
            ),
        )
        for member in indexed_members:
            path_bytes = member.logical_path.encode("utf-8")
            if len(path_bytes) > _MAX_U32:
                raise PacbinError("pacbin logical path exceeds u32 byte length")
            prefix = _INDEX_ENTRY_STRUCT.pack(
                len(path_bytes),
                int(member.kind),
                _SUPPORTED_FLAGS,
                member.offset,
                member.length,
                bytes.fromhex(member.sha256),
            )
            _write_hashed(stream, index_digest, prefix)
            _write_hashed(stream, index_digest, path_bytes)
            padding = _padding_length(len(prefix) + len(path_bytes), _INDEX_ALIGNMENT)
            if padding:
                _write_hashed(stream, index_digest, b"\x00" * padding)

        index_sha256 = index_digest.digest()
        if stream.tell() - index_offset != expected_index_size:
            raise PacbinError("pacbin index size disagrees with preflight")
        _write_all(
            stream,
            _FOOTER_STRUCT.pack(
                _FOOTER_MAGIC,
                PACBIN_VERSION,
                _FOOTER_STRUCT.size,
                _SUPPORTED_FLAGS,
                index_offset,
                len(indexed_members),
                index_sha256,
            ),
            "pacbin footer",
        )
        file_size = _checked_u64_position(stream.tell(), "pacbin file size")
        stream.seek(0)
        _write_all(
            stream,
            _HEADER_STRUCT.pack(
                _HEADER_MAGIC,
                PACBIN_VERSION,
                _HEADER_STRUCT.size,
                _SUPPORTED_FLAGS,
                PACBIN_ALIGNMENT,
                0,
                index_offset,
                len(indexed_members),
                b"\x00" * 24,
            ),
            "pacbin header",
        )
        stream.seek(file_size)
        stream.truncate()
        stream.flush()

    return PacbinIndex(
        version=PACBIN_VERSION,
        index_offset=index_offset,
        index_sha256=index_sha256.hex(),
        file_size=file_size,
        members=tuple(indexed_members),
    )


def write_pacbin_atomic(
    destination: str | os.PathLike[str],
    members: Iterable[PacbinMemberSource],
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    mode: int = 0o644,
) -> PacbinIndex:
    """Atomically publish a complete container at ``destination``.

    The container is written, flushed, and synced through a uniquely named
    temporary file in the destination directory before ``os.replace`` makes
    it visible.  Any failure before replacement leaves an existing
    destination untouched and removes the temporary file.  Directory syncing
    is best effort because not every supported filesystem exposes it.
    """

    if not isinstance(mode, int) or isinstance(mode, bool) or not 0 <= mode <= 0o777:
        raise PacbinError("pacbin publication mode must be between 0o000 and 0o777")
    path = Path(destination)
    parent = path.parent
    if not parent.is_dir():
        raise PacbinError(f"pacbin destination directory does not exist: {parent}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary_path = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            descriptor_open = False
            index = write_pacbin(stream, members, chunk_size=chunk_size)
            os.chmod(temporary_path, mode)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _fsync_directory_best_effort(parent)
        return index
    finally:
        if descriptor_open:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary_path.unlink()


class PacbinReader:
    """Strict, bounded-access reader for one pacbin-v1 container."""

    def __init__(
        self,
        stream: BinaryIO,
        *,
        owns_stream: bool,
        verify_payloads: bool,
        chunk_size: int,
    ) -> None:
        self._stream = stream
        self._owns_stream = owns_stream
        self._closed = False
        self._chunk_size = _checked_chunk_size(chunk_size)
        self._require_readable_seekable()
        self.index = self._load_index()
        self._members_by_path = {
            member.logical_path: member for member in self.index.members
        }
        if verify_payloads:
            self.verify_payloads()

    @classmethod
    def open(
        cls,
        source: str | os.PathLike[str] | BinaryIO,
        *,
        verify_payloads: bool = True,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> PacbinReader:
        """Open and structurally validate a container.

        Payload digests are verified by default.  Passing an existing binary
        stream leaves ownership with the caller.
        """

        if isinstance(source, str | os.PathLike):
            # The returned reader owns this stream until PacbinReader.close().
            stream = Path(source).open("rb")  # noqa: SIM115
            owns_stream = True
        else:
            stream = source
            owns_stream = False
        try:
            return cls(
                stream,
                owns_stream=owns_stream,
                verify_payloads=verify_payloads,
                chunk_size=chunk_size,
            )
        except BaseException:
            if owns_stream:
                stream.close()
            raise

    @property
    def members(self) -> tuple[PacbinMember, ...]:
        """Return members in canonical logical-path order."""

        return self.index.members

    def member(self, logical_path: str) -> PacbinMember:
        """Return metadata for one logical member."""

        self._ensure_open()
        normalized = normalize_logical_path(logical_path)
        try:
            return self._members_by_path[normalized]
        except KeyError as exc:
            raise KeyError(normalized) from exc

    def read_member(
        self,
        logical_path: str,
        *,
        offset: int = 0,
        length: int,
    ) -> bytes:
        """Read exactly one bounded range relative to a member's start."""

        self._ensure_open()
        member = self.member(logical_path)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise PacbinError("member read offset must be a non-negative integer")
        if not isinstance(length, int) or isinstance(length, bool) or length < 0:
            raise PacbinError("member read length must be a non-negative integer")
        if offset > member.length or length > member.length - offset:
            raise PacbinError("member read exceeds indexed payload bounds")
        self._stream.seek(member.offset + offset)
        return _read_exact(self._stream, length, "member payload")

    def open_member_stream(self, logical_path: str) -> PacbinMemberStream:
        """Return a bounded streaming view over one member.

        The returned view never materializes the complete member and remains
        valid only while this reader is open.  Each read seeks to its indexed
        position, so several member views may safely be consumed in any
        sequential order while repacking a container.
        """

        self._ensure_open()
        return PacbinMemberStream(self, self.member(logical_path))

    def verify_payloads(self) -> None:
        """Stream and verify every indexed member digest."""

        self._ensure_open()
        for member in self.index.members:
            digest = hashlib.sha256()
            remaining = member.length
            self._stream.seek(member.offset)
            while remaining:
                requested = min(self._chunk_size, remaining)
                chunk = self._stream.read(requested)
                if not chunk:
                    raise PacbinError(
                        f"truncated pacbin member payload: {member.logical_path}"
                    )
                if not isinstance(chunk, bytes | bytearray | memoryview):
                    raise TypeError("pacbin member source must return bytes")
                if len(chunk) > requested:
                    raise PacbinError(
                        f"oversized read for pacbin member: {member.logical_path}"
                    )
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != member.sha256:
                raise PacbinError(
                    f"pacbin member digest mismatch: {member.logical_path}"
                )

    def close(self) -> None:
        """Close an owned stream and invalidate further reads."""

        if self._closed:
            return
        self._closed = True
        if self._owns_stream:
            self._stream.close()

    def __enter__(self) -> PacbinReader:
        self._ensure_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _require_readable_seekable(self) -> None:
        _require_seekable(self._stream, "pacbin source")
        readable = getattr(self._stream, "readable", None)
        if callable(readable) and not readable():
            raise PacbinError("pacbin source must be readable")

    def _load_index(self) -> PacbinIndex:
        stream = self._stream
        stream.seek(0, os.SEEK_END)
        file_size = stream.tell()
        if file_size < 0 or file_size > _MAX_U64:
            raise PacbinError("pacbin file size exceeds u64")
        minimum_size = (
            _HEADER_STRUCT.size + _INDEX_HEADER_STRUCT.size + _FOOTER_STRUCT.size
        )
        if file_size < minimum_size:
            raise PacbinError("truncated pacbin container")

        stream.seek(0)
        header = _HEADER_STRUCT.unpack(
            _read_exact(stream, _HEADER_STRUCT.size, "pacbin header")
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
        if magic != _HEADER_MAGIC:
            raise PacbinError("invalid pacbin header magic")
        _validate_version_size_flags(
            "header", version, header_size, _HEADER_STRUCT.size, flags
        )
        if alignment != PACBIN_ALIGNMENT:
            raise PacbinError(f"unsupported pacbin payload alignment: {alignment}")
        if reserved != 0 or reserved_bytes != b"\x00" * 24:
            raise PacbinError("pacbin header reserved fields must be zero")
        if index_offset % PACBIN_ALIGNMENT:
            raise PacbinError("pacbin index offset is not payload-aligned")

        footer_offset = file_size - _FOOTER_STRUCT.size
        if index_offset < _HEADER_STRUCT.size or index_offset >= footer_offset:
            raise PacbinError("pacbin index offset is out of bounds")
        stream.seek(footer_offset)
        footer = _FOOTER_STRUCT.unpack(
            _read_exact(stream, _FOOTER_STRUCT.size, "pacbin footer")
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
        if footer_magic != _FOOTER_MAGIC:
            raise PacbinError("invalid pacbin footer magic")
        _validate_version_size_flags(
            "footer",
            footer_version,
            footer_size,
            _FOOTER_STRUCT.size,
            footer_flags,
        )
        if footer_index_offset != index_offset:
            raise PacbinError("pacbin footer index offset disagrees with header")
        if footer_member_count != member_count:
            raise PacbinError("pacbin footer member count disagrees with header")

        stream.seek(index_offset)
        index_digest = hashlib.sha256()
        index_header_bytes = _read_exact(
            stream, _INDEX_HEADER_STRUCT.size, "pacbin index header"
        )
        index_digest.update(index_header_bytes)
        (
            index_magic,
            index_version,
            index_header_size,
            index_flags,
            index_member_count,
            index_reserved,
        ) = _INDEX_HEADER_STRUCT.unpack(index_header_bytes)
        if index_magic != _INDEX_MAGIC:
            raise PacbinError("invalid pacbin index magic")
        _validate_version_size_flags(
            "index",
            index_version,
            index_header_size,
            _INDEX_HEADER_STRUCT.size,
            index_flags,
        )
        if index_reserved != 0:
            raise PacbinError("pacbin index reserved field must be zero")
        if index_member_count != member_count:
            raise PacbinError("pacbin index member count disagrees with header")
        available_index_bytes = footer_offset - index_offset
        if member_count > available_index_bytes // (_INDEX_ENTRY_STRUCT.size + 8):
            raise PacbinError("pacbin member count cannot fit in index")

        members: list[PacbinMember] = []
        seen_paths: set[str] = set()
        seen_folded_paths: set[str] = set()
        previous_path_bytes: bytes | None = None
        for _entry_index in range(member_count):
            prefix = _read_index_bytes(
                stream,
                index_digest,
                _INDEX_ENTRY_STRUCT.size,
                footer_offset,
                "pacbin index entry",
            )
            path_length, kind_value, entry_flags, offset, length, digest = (
                _INDEX_ENTRY_STRUCT.unpack(prefix)
            )
            if entry_flags != _SUPPORTED_FLAGS:
                raise PacbinError(f"unknown pacbin member flags: {entry_flags}")
            try:
                kind = PacbinMemberKind(kind_value)
            except ValueError as exc:
                raise PacbinError(f"unknown pacbin member kind: {kind_value}") from exc
            path_bytes = _read_index_bytes(
                stream,
                index_digest,
                path_length,
                footer_offset,
                "pacbin member path",
            )
            try:
                logical_path = path_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PacbinError("pacbin member path is not valid UTF-8") from exc
            if normalize_logical_path(logical_path) != logical_path:
                raise PacbinError(
                    f"pacbin member path is not canonical: {logical_path!r}"
                )
            folded = logical_path.casefold()
            if logical_path in seen_paths:
                raise PacbinError(f"duplicate pacbin member path: {logical_path}")
            if folded in seen_folded_paths:
                raise PacbinError(f"case-colliding pacbin member path: {logical_path}")
            if previous_path_bytes is not None and path_bytes <= previous_path_bytes:
                raise PacbinError("pacbin index paths are not strictly sorted")
            previous_path_bytes = path_bytes
            seen_paths.add(logical_path)
            seen_folded_paths.add(folded)
            padding = _padding_length(
                _INDEX_ENTRY_STRUCT.size + path_length, _INDEX_ALIGNMENT
            )
            if padding:
                padding_bytes = _read_index_bytes(
                    stream,
                    index_digest,
                    padding,
                    footer_offset,
                    "pacbin index padding",
                )
                if padding_bytes != b"\x00" * padding:
                    raise PacbinError("pacbin index padding must be zero")
            members.append(
                PacbinMember(
                    logical_path=logical_path,
                    kind=kind,
                    offset=offset,
                    length=length,
                    sha256=digest.hex(),
                )
            )

        if stream.tell() != footer_offset:
            raise PacbinError("pacbin index has trailing or missing bytes")
        if index_digest.digest() != expected_index_digest:
            raise PacbinError("pacbin index digest mismatch")
        _validate_canonical_payload_layout(stream, members, index_offset)
        return PacbinIndex(
            version=version,
            index_offset=index_offset,
            index_sha256=index_digest.hexdigest(),
            file_size=file_size,
            members=tuple(members),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("I/O operation on closed pacbin reader")


class PacbinMemberStream:
    """Read-only, bounded view over one member owned by a :class:`PacbinReader`."""

    def __init__(self, reader: PacbinReader, member: PacbinMember) -> None:
        self._reader = reader
        self.member = member
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._position

    def read(self, size: int = -1) -> bytes:
        self._reader._ensure_open()
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError("pacbin member stream read size must be an integer")
        if size < 0:
            raise PacbinError("pacbin member stream requires a bounded read size")
        remaining = self.member.length - self._position
        if size == 0 or remaining == 0:
            return b""
        requested = min(size, remaining)
        self._reader._stream.seek(self.member.offset + self._position)
        payload = _read_exact(
            self._reader._stream,
            requested,
            f"pacbin member {self.member.logical_path!r}",
        )
        self._position += len(payload)
        return payload


def _ordered_sources(
    members: Iterable[PacbinMemberSource],
) -> tuple[PacbinMemberSource, ...]:
    prepared = tuple(members)
    if not all(isinstance(member, PacbinMemberSource) for member in prepared):
        raise TypeError("pacbin members must be PacbinMemberSource instances")
    ordered = tuple(
        sorted(prepared, key=lambda member: member.logical_path.encode("utf-8"))
    )
    seen_paths: set[str] = set()
    seen_folded_paths: set[str] = set()
    for member in ordered:
        folded = member.logical_path.casefold()
        if member.logical_path in seen_paths:
            raise PacbinError(f"duplicate pacbin member path: {member.logical_path}")
        if folded in seen_folded_paths:
            raise PacbinError(
                f"case-colliding pacbin member path: {member.logical_path}"
            )
        seen_paths.add(member.logical_path)
        seen_folded_paths.add(folded)
    return ordered


def _validate_index_bounds(members: tuple[PacbinMemberSource, ...]) -> int:
    if len(members) > _MAX_U64:
        raise PacbinError("pacbin member count exceeds u64")
    index_size = _INDEX_HEADER_STRUCT.size
    for member in members:
        path_length = len(member.logical_path.encode("utf-8"))
        if path_length > _MAX_U32:
            raise PacbinError("pacbin logical path exceeds u32 byte length")
        record_size = _INDEX_ENTRY_STRUCT.size + path_length
        record_size += _padding_length(record_size, _INDEX_ALIGNMENT)
        if record_size > _MAX_U64 - index_size:
            raise PacbinError("pacbin index size exceeds u64")
        index_size += record_size
    return index_size


@contextmanager
def _open_destination(
    destination: str | os.PathLike[str] | BinaryIO,
) -> Iterator[BinaryIO]:
    if isinstance(destination, str | os.PathLike):
        with Path(destination).open("w+b") as stream:
            yield stream
    else:
        yield destination


@contextmanager
def _open_member_source(
    source: str | os.PathLike[str] | BinaryIO,
) -> Iterator[BinaryIO]:
    if isinstance(source, str | os.PathLike):
        path = Path(source)
        if not path.is_file():
            raise PacbinError(f"pacbin member source is not a file: {path}")
        with path.open("rb") as stream:
            yield stream
    else:
        read = getattr(source, "read", None)
        if not callable(read):
            raise TypeError("pacbin member source must be a path or binary file object")
        yield source


def _checked_chunk_size(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PacbinError("pacbin chunk size must be a positive integer")
    return value


def _require_seekable(stream: BinaryIO, label: str) -> None:
    seekable = getattr(stream, "seekable", None)
    if not callable(seekable) or not seekable():
        raise PacbinError(f"{label} must be seekable")


def _padding_length(position: int, alignment: int) -> int:
    return (-position) % alignment


def _write_alignment_padding(stream: BinaryIO, alignment: int) -> None:
    padding = _padding_length(stream.tell(), alignment)
    if padding:
        _write_all(stream, b"\x00" * padding, "pacbin payload padding")


def _write_hashed(stream: BinaryIO, digest: object, payload: bytes) -> None:
    _write_all(stream, payload, "pacbin index")
    digest.update(payload)  # type: ignore[attr-defined]


def _write_all(
    stream: BinaryIO,
    payload: bytes | bytearray | memoryview,
    label: str,
) -> None:
    view = memoryview(payload)
    written_total = 0
    while written_total < len(view):
        written = stream.write(view[written_total:])
        if (
            not isinstance(written, int)
            or isinstance(written, bool)
            or written <= 0
            or written > len(view) - written_total
        ):
            raise PacbinError(f"short write while writing {label}")
        written_total += written


def _checked_u64_position(value: int, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= _MAX_U64
    ):
        raise PacbinError(f"{label} exceeds u64")
    return value


def _read_exact(stream: BinaryIO, length: int, label: str) -> bytes:
    remaining = length
    chunks: list[bytes] = []
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise PacbinError(f"truncated {label}")
        if not isinstance(chunk, bytes | bytearray | memoryview):
            raise TypeError(f"{label} source must return bytes")
        if len(chunk) > remaining:
            raise PacbinError(f"oversized read while reading {label}")
        chunks.append(bytes(chunk))
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_index_bytes(
    stream: BinaryIO,
    digest: object,
    length: int,
    footer_offset: int,
    label: str,
) -> bytes:
    if length < 0 or stream.tell() > footer_offset - length:
        raise PacbinError(f"truncated {label}")
    payload = _read_exact(stream, length, label)
    digest.update(payload)  # type: ignore[attr-defined]
    return payload


def _validate_version_size_flags(
    label: str,
    version: int,
    encoded_size: int,
    expected_size: int,
    flags: int,
) -> None:
    if version != PACBIN_VERSION:
        raise PacbinError(f"unsupported pacbin {label} version: {version}")
    if encoded_size != expected_size:
        raise PacbinError(f"unsupported pacbin {label} size: {encoded_size}")
    if flags != _SUPPORTED_FLAGS:
        raise PacbinError(f"unknown pacbin {label} flags: {flags}")


def _validate_canonical_payload_layout(
    stream: BinaryIO,
    members: list[PacbinMember],
    index_offset: int,
) -> None:
    expected_offset = _HEADER_STRUCT.size
    for member in members:
        if member.length > _MAX_U64 - member.offset:
            raise PacbinError(
                f"pacbin member range overflows u64: {member.logical_path}"
            )
        unaligned_offset = expected_offset
        padding_length = _padding_length(unaligned_offset, PACBIN_ALIGNMENT)
        expected_offset += padding_length
        if member.offset < expected_offset:
            raise PacbinError(
                f"overlapping pacbin member payload: {member.logical_path}"
            )
        if member.offset > expected_offset:
            raise PacbinError(f"non-canonical pacbin member gap: {member.logical_path}")
        if member.offset % PACBIN_ALIGNMENT:
            raise PacbinError(f"unaligned pacbin member payload: {member.logical_path}")
        end = member.offset + member.length
        if end > index_offset:
            raise PacbinError(
                f"pacbin member payload is out of bounds: {member.logical_path}"
            )
        if padding_length:
            stream.seek(unaligned_offset)
            if _read_exact(stream, padding_length, "pacbin payload padding") != (
                b"\x00" * padding_length
            ):
                raise PacbinError("pacbin payload padding must be zero")
        expected_offset = end

    payload_end = expected_offset
    canonical_index_offset = payload_end + _padding_length(
        payload_end, PACBIN_ALIGNMENT
    )
    if canonical_index_offset != index_offset:
        if canonical_index_offset > index_offset:
            raise PacbinError("pacbin payloads overlap the index")
        raise PacbinError("pacbin payload region has a non-canonical trailing gap")
    padding_length = index_offset - payload_end
    if padding_length:
        stream.seek(payload_end)
        if _read_exact(stream, padding_length, "pacbin payload padding") != (
            b"\x00" * padding_length
        ):
            raise PacbinError("pacbin payload padding must be zero")


def _fsync_directory_best_effort(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        with suppress(OSError):
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "PACBIN_ALIGNMENT",
    "PACBIN_VERSION",
    "PacbinError",
    "PacbinIndex",
    "PacbinMember",
    "PacbinMemberKind",
    "PacbinMemberSource",
    "PacbinMemberStream",
    "PacbinReader",
    "normalize_logical_path",
    "write_pacbin",
    "write_pacbin_atomic",
]
