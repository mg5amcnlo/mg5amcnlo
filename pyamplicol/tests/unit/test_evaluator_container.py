# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import io
import stat
import struct
from collections.abc import Callable
from pathlib import Path

import pytest

from pyamplicol.generation import evaluator_container as pacbin
from pyamplicol.generation.evaluator_container import (
    PACBIN_ALIGNMENT,
    PACBIN_VERSION,
    PacbinError,
    PacbinMemberKind,
    PacbinMemberSource,
    PacbinReader,
    normalize_logical_path,
    write_pacbin,
    write_pacbin_atomic,
)


class _BoundedSource(io.BytesIO):
    def __init__(self, payload: bytes, maximum_read: int) -> None:
        super().__init__(payload)
        self.maximum_read = maximum_read
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        if size < 0 or size > self.maximum_read:
            raise AssertionError(f"unbounded source read: {size}")
        self.read_sizes.append(size)
        return super().read(size)


class _NonSeekable(io.BytesIO):
    def seekable(self) -> bool:
        return False


class _ShortWrite(io.BytesIO):
    def write(self, payload: bytes | bytearray | memoryview) -> int:
        return super().write(payload[:3])


class _StalledWrite(io.BytesIO):
    def write(self, payload: bytes | bytearray | memoryview) -> int:
        del payload
        return 0


class _OversizedRead(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        return super().read(size + 1 if size >= 0 else size)


class _FailingSource(io.BytesIO):
    def __init__(self) -> None:
        super().__init__(b"partial")
        self._failed = False

    def read(self, size: int = -1) -> bytes:
        if self._failed:
            raise RuntimeError("injected source failure")
        self._failed = True
        return super().read(size)


def _sources(payload: bytes) -> tuple[PacbinMemberSource, ...]:
    return (
        PacbinMemberSource(
            "z/native/library.so",
            PacbinMemberKind.NATIVE_LIBRARY,
            io.BytesIO(b"native"),
        ),
        PacbinMemberSource(
            "a//jit/./application.symjit",
            PacbinMemberKind.SYMJIT_APPLICATION,
            io.BytesIO(payload),
        ),
        PacbinMemberSource(
            "m/exact/state.bin",
            PacbinMemberKind.SYMBOLICA_EXACT_STATE,
            io.BytesIO(b"exact-state"),
        ),
    )


def _container_bytes(payload: bytes = b"symjit-payload") -> bytes:
    destination = io.BytesIO()
    write_pacbin(destination, _sources(payload), chunk_size=4)
    return destination.getvalue()


def _index_entry_offsets(data: bytes | bytearray) -> list[int]:
    header = pacbin._HEADER_STRUCT.unpack_from(data, 0)
    index_offset = header[6]
    index_header = pacbin._INDEX_HEADER_STRUCT.unpack_from(data, index_offset)
    count = index_header[4]
    cursor = index_offset + pacbin._INDEX_HEADER_STRUCT.size
    offsets: list[int] = []
    for _ in range(count):
        offsets.append(cursor)
        path_length = pacbin._INDEX_ENTRY_STRUCT.unpack_from(data, cursor)[0]
        record_length = pacbin._INDEX_ENTRY_STRUCT.size + path_length
        cursor += record_length + (-record_length) % pacbin._INDEX_ALIGNMENT
    return offsets


def _rewrite_index_digest(data: bytearray) -> None:
    index_offset = pacbin._HEADER_STRUCT.unpack_from(data, 0)[6]
    footer_offset = len(data) - pacbin._FOOTER_STRUCT.size
    digest = hashlib.sha256(data[index_offset:footer_offset]).digest()
    footer = list(pacbin._FOOTER_STRUCT.unpack_from(data, footer_offset))
    footer[-1] = digest
    pacbin._FOOTER_STRUCT.pack_into(data, footer_offset, *footer)


def _mutated_container(
    mutation: Callable[[bytearray], None],
    *,
    rewrite_index_digest: bool = False,
) -> io.BytesIO:
    data = bytearray(_container_bytes())
    mutation(data)
    if rewrite_index_digest:
        _rewrite_index_digest(data)
    return io.BytesIO(data)


def test_pacbin_round_trip_is_deterministic_and_aligned(tmp_path: Path) -> None:
    payload = bytes(range(251)) * 17
    source_path = tmp_path / "native.so"
    source_path.write_bytes(b"native")
    first_members = (
        PacbinMemberSource(
            "z/native/library.so", PacbinMemberKind.NATIVE_LIBRARY, source_path
        ),
        PacbinMemberSource(
            "a//jit/./application.symjit",
            PacbinMemberKind.SYMJIT_APPLICATION,
            _BoundedSource(payload, 113),
        ),
        PacbinMemberSource(
            "m/exact/state.bin",
            PacbinMemberKind.SYMBOLICA_EXACT_STATE,
            io.BytesIO(b"exact-state"),
        ),
    )
    first = io.BytesIO()
    written = write_pacbin(first, first_members, chunk_size=113)
    assert first_members[1].source.read_sizes
    assert max(first_members[1].source.read_sizes) <= 113

    second = io.BytesIO()
    write_pacbin(second, reversed(_sources(payload)), chunk_size=7)
    assert first.getvalue() == second.getvalue()
    assert written.version == PACBIN_VERSION
    assert written.file_size == len(first.getvalue())
    assert [member.logical_path for member in written.members] == [
        "a/jit/application.symjit",
        "m/exact/state.bin",
        "z/native/library.so",
    ]
    assert all(member.offset % PACBIN_ALIGNMENT == 0 for member in written.members)

    first.seek(0)
    with PacbinReader.open(first) as reader:
        assert reader.index == written
        assert (
            reader.read_member("a/./jit/application.symjit", offset=5, length=19)
            == payload[5:24]
        )
        assert (
            reader.member("m/exact/state.bin").sha256
            == hashlib.sha256(b"exact-state").hexdigest()
        )
    assert not first.closed


def test_runtime_plan_member_kinds_round_trip() -> None:
    destination = io.BytesIO()
    sources = tuple(
        PacbinMemberSource(path, kind, io.BytesIO(kind.name.encode()))
        for path, kind in (
            ("eager/metadata.bin", PacbinMemberKind.EAGER_RUNTIME_METADATA),
            ("eager/table.bin", PacbinMemberKind.EAGER_RUNTIME_TABLE),
            ("recurrence/plan.bin", PacbinMemberKind.RECURRENCE_DIRECT_PLAN),
            (
                "recurrence/projection.bin",
                PacbinMemberKind.RECURRENCE_COLOR_PROJECTION_CERTIFICATE,
            ),
            (
                "compiled/color.pacrclr3",
                PacbinMemberKind.COLOR_CONTRACTION,
            ),
        )
    )
    write_pacbin(destination, sources)
    destination.seek(0)
    with PacbinReader.open(destination) as reader:
        assert {member.kind for member in reader.members} == {
            PacbinMemberKind.EAGER_RUNTIME_METADATA,
            PacbinMemberKind.EAGER_RUNTIME_TABLE,
            PacbinMemberKind.RECURRENCE_DIRECT_PLAN,
            PacbinMemberKind.RECURRENCE_COLOR_PROJECTION_CERTIFICATE,
            PacbinMemberKind.COLOR_CONTRACTION,
        }


def test_empty_and_zero_length_members_round_trip() -> None:
    empty = io.BytesIO()
    index = write_pacbin(empty, ())
    assert index.members == ()
    empty.seek(0)
    with PacbinReader.open(empty) as reader:
        assert reader.members == ()

    zero = io.BytesIO()
    write_pacbin(
        zero,
        (
            PacbinMemberSource(
                "a.empty", PacbinMemberKind.SYMBOLICA_EXACT_STATE, io.BytesIO()
            ),
            PacbinMemberSource(
                "b.empty", PacbinMemberKind.SYMJIT_APPLICATION, io.BytesIO()
            ),
        ),
    )
    zero.seek(0)
    with PacbinReader.open(zero) as reader:
        assert [member.length for member in reader.members] == [0, 0]
        assert reader.read_member("b.empty", length=0) == b""


def test_member_stream_repackages_with_bounded_reads() -> None:
    payload = bytes(range(251)) * 101
    source = _BoundedSource(_container_bytes(payload), 1 << 20)
    with PacbinReader.open(source, verify_payloads=False) as reader:
        source.read_sizes.clear()
        member_stream = reader.open_member_stream("a/jit/application.symjit")
        destination = io.BytesIO()
        write_pacbin(
            destination,
            (
                PacbinMemberSource(
                    "repacked/application.symjit",
                    PacbinMemberKind.SYMJIT_APPLICATION,
                    member_stream,
                ),
            ),
            chunk_size=17,
        )
        assert source.read_sizes
        assert max(source.read_sizes) <= 17
        with pytest.raises(PacbinError, match="bounded read size"):
            reader.open_member_stream("a/jit/application.symjit").read()

    destination.seek(0)
    with PacbinReader.open(destination) as repacked:
        assert (
            repacked.read_member(
                "repacked/application.symjit",
                length=len(payload),
            )
            == payload
        )


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("", "non-empty"),
        (".", "name a member"),
        ("/absolute", "relative"),
        ("safe/../escape", "must not contain"),
        ("windows\\path", "POSIX"),
        ("bad\x00path", "NUL"),
    ),
)
def test_logical_path_rejects_unsafe_values(value: str, message: str) -> None:
    with pytest.raises(PacbinError, match=message):
        normalize_logical_path(value)


def test_logical_path_is_nfc_and_posix_normalized() -> None:
    assert normalize_logical_path("a//./cafe\u0301/state") == "a/caf\u00e9/state"


def test_writer_rejects_duplicate_case_collision_and_unknown_kind() -> None:
    duplicate = PacbinMemberSource(
        "same/path", PacbinMemberKind.SYMJIT_APPLICATION, io.BytesIO()
    )
    with pytest.raises(PacbinError, match="duplicate"):
        write_pacbin(io.BytesIO(), (duplicate, duplicate))
    with pytest.raises(PacbinError, match="case-colliding"):
        write_pacbin(
            io.BytesIO(),
            (
                PacbinMemberSource(
                    "Case/path", PacbinMemberKind.SYMJIT_APPLICATION, io.BytesIO()
                ),
                PacbinMemberSource(
                    "case/PATH", PacbinMemberKind.NATIVE_LIBRARY, io.BytesIO()
                ),
            ),
        )
    with pytest.raises(PacbinError, match="unknown pacbin member kind"):
        PacbinMemberSource("member", 99, io.BytesIO())  # type: ignore[arg-type]


def test_writer_requires_typed_members_and_seekable_destination() -> None:
    with pytest.raises(TypeError, match="PacbinMemberSource"):
        write_pacbin(io.BytesIO(), (("member", 1, io.BytesIO()),))  # type: ignore[arg-type]
    with pytest.raises(PacbinError, match="seekable"):
        write_pacbin(_NonSeekable(), ())


def test_writer_handles_short_writes_and_rejects_stalled_writes() -> None:
    destination = _ShortWrite()
    written = write_pacbin(destination, _sources(b"payload"), chunk_size=3)
    destination.seek(0)
    with PacbinReader.open(destination) as reader:
        assert reader.index == written

    with pytest.raises(PacbinError, match="short write"):
        write_pacbin(_StalledWrite(), ())


def test_writer_rejects_source_that_exceeds_requested_chunk() -> None:
    with pytest.raises(PacbinError, match="more bytes than requested"):
        write_pacbin(
            io.BytesIO(),
            (
                PacbinMemberSource(
                    "payload.bin",
                    PacbinMemberKind.SYMJIT_APPLICATION,
                    _OversizedRead(b"abcdef"),
                ),
            ),
            chunk_size=2,
        )


def test_writer_preflights_wire_bounds_before_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = io.BytesIO(b"existing")
    monkeypatch.setattr(pacbin, "_MAX_U32", 3)
    with pytest.raises(PacbinError, match="path exceeds u32 byte length"):
        write_pacbin(destination, _sources(b"payload"))
    assert destination.getvalue() == b"existing"

    monkeypatch.setattr(pacbin, "_MAX_U32", 2**32 - 1)
    monkeypatch.setattr(pacbin, "_MAX_U64", 2)
    with pytest.raises(PacbinError, match="member count exceeds u64"):
        write_pacbin(destination, _sources(b"payload"))
    assert destination.getvalue() == b"existing"

    monkeypatch.setattr(
        pacbin,
        "_MAX_U64",
        pacbin._INDEX_HEADER_STRUCT.size,
    )
    with pytest.raises(PacbinError, match="index size exceeds u64"):
        write_pacbin(destination, _sources(b"payload"))
    assert destination.getvalue() == b"existing"


def test_atomic_publication_replaces_only_complete_container(tmp_path: Path) -> None:
    destination = tmp_path / "evaluators.pacbin"
    destination.write_bytes(b"previous-container")

    with pytest.raises(RuntimeError, match="injected source failure"):
        write_pacbin_atomic(
            destination,
            (
                PacbinMemberSource(
                    "payload.bin",
                    PacbinMemberKind.SYMJIT_APPLICATION,
                    _FailingSource(),
                ),
            ),
            chunk_size=3,
        )
    assert destination.read_bytes() == b"previous-container"
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []

    index = write_pacbin_atomic(destination, _sources(b"published"), mode=0o640)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    with PacbinReader.open(destination) as reader:
        assert reader.index == index
        assert (
            reader.read_member("a/jit/application.symjit", offset=0, length=9)
            == b"published"
        )
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


@pytest.mark.parametrize("mode", (-1, 0o1000, True))
def test_atomic_publication_rejects_invalid_mode(tmp_path: Path, mode: object) -> None:
    with pytest.raises(PacbinError, match="publication mode"):
        write_pacbin_atomic(
            tmp_path / "evaluators.pacbin",
            (),
            mode=mode,  # type: ignore[arg-type]
        )


def test_reader_bounded_reads_and_close_contract() -> None:
    stream = io.BytesIO(_container_bytes(b"abcdef"))
    reader = PacbinReader.open(stream)
    with pytest.raises(PacbinError, match="offset"):
        reader.read_member("a/jit/application.symjit", offset=-1, length=1)
    with pytest.raises(PacbinError, match="length"):
        reader.read_member("a/jit/application.symjit", length=-1)
    with pytest.raises(PacbinError, match="exceeds"):
        reader.read_member("a/jit/application.symjit", offset=5, length=2)
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.member("a/jit/application.symjit")


@pytest.mark.parametrize(
    ("field_index", "value", "message"),
    (
        (1, PACBIN_VERSION + 1, "header version"),
        (2, 1, "header size"),
        (3, 1, "header flags"),
        (4, 32, "payload alignment"),
    ),
)
def test_reader_rejects_unknown_header_contract(
    field_index: int, value: int, message: str
) -> None:
    def mutate(data: bytearray) -> None:
        header = list(pacbin._HEADER_STRUCT.unpack_from(data, 0))
        header[field_index] = value
        pacbin._HEADER_STRUCT.pack_into(data, 0, *header)

    with pytest.raises(PacbinError, match=message):
        PacbinReader.open(_mutated_container(mutate))


@pytest.mark.parametrize(
    ("location", "message"),
    (
        ("header", "header magic"),
        ("index", "index magic"),
        ("footer", "footer magic"),
    ),
)
def test_reader_rejects_invalid_magic(location: str, message: str) -> None:
    def mutate(data: bytearray) -> None:
        if location == "header":
            offset = 0
        elif location == "index":
            offset = pacbin._HEADER_STRUCT.unpack_from(data, 0)[6]
        else:
            offset = len(data) - pacbin._FOOTER_STRUCT.size
        data[offset : offset + 8] = b"INVALID!"

    with pytest.raises(PacbinError, match=message):
        PacbinReader.open(
            _mutated_container(
                mutate,
                rewrite_index_digest=location == "index",
            )
        )


def test_reader_rejects_nonzero_header_reserved_fields() -> None:
    def reserved_integer(data: bytearray) -> None:
        header = list(pacbin._HEADER_STRUCT.unpack_from(data, 0))
        header[5] = 1
        pacbin._HEADER_STRUCT.pack_into(data, 0, *header)

    with pytest.raises(PacbinError, match="reserved fields"):
        PacbinReader.open(_mutated_container(reserved_integer))

    def reserved_bytes(data: bytearray) -> None:
        header = list(pacbin._HEADER_STRUCT.unpack_from(data, 0))
        header[8] = b"\x00" * 23 + b"\x01"
        pacbin._HEADER_STRUCT.pack_into(data, 0, *header)

    with pytest.raises(PacbinError, match="reserved fields"):
        PacbinReader.open(_mutated_container(reserved_bytes))


@pytest.mark.parametrize(
    ("field_index", "value", "message"),
    (
        (1, PACBIN_VERSION + 1, "footer version"),
        (2, 1, "footer size"),
        (3, 1, "footer flags"),
        (4, 0, "offset disagrees"),
        (5, 99, "count disagrees"),
    ),
)
def test_reader_rejects_unknown_footer_contract(
    field_index: int, value: int, message: str
) -> None:
    def mutate(data: bytearray) -> None:
        offset = len(data) - pacbin._FOOTER_STRUCT.size
        footer = list(pacbin._FOOTER_STRUCT.unpack_from(data, offset))
        footer[field_index] = value
        pacbin._FOOTER_STRUCT.pack_into(data, offset, *footer)

    with pytest.raises(PacbinError, match=message):
        PacbinReader.open(_mutated_container(mutate))


@pytest.mark.parametrize(
    ("field_index", "value", "message"),
    (
        (1, PACBIN_VERSION + 1, "index version"),
        (2, 1, "index size"),
        (3, 1, "index flags"),
        (5, 1, "reserved"),
    ),
)
def test_reader_rejects_unknown_index_contract(
    field_index: int, value: int, message: str
) -> None:
    def mutate(data: bytearray) -> None:
        offset = pacbin._HEADER_STRUCT.unpack_from(data, 0)[6]
        header = list(pacbin._INDEX_HEADER_STRUCT.unpack_from(data, offset))
        header[field_index] = value
        pacbin._INDEX_HEADER_STRUCT.pack_into(data, offset, *header)

    with pytest.raises(PacbinError, match=message):
        PacbinReader.open(_mutated_container(mutate, rewrite_index_digest=True))


@pytest.mark.parametrize(
    ("field_offset", "format_string", "value", "message"),
    (
        (4, "<H", 99, "member kind"),
        (6, "<H", 1, "member flags"),
    ),
)
def test_reader_rejects_unknown_member_contract(
    field_offset: int, format_string: str, value: int, message: str
) -> None:
    def mutate(data: bytearray) -> None:
        entry = _index_entry_offsets(data)[0]
        struct.pack_into(format_string, data, entry + field_offset, value)

    with pytest.raises(PacbinError, match=message):
        PacbinReader.open(_mutated_container(mutate, rewrite_index_digest=True))


def test_reader_rejects_noncanonical_and_case_colliding_paths() -> None:
    def traversal(data: bytearray) -> None:
        entry = _index_entry_offsets(data)[0]
        path_offset = entry + pacbin._INDEX_ENTRY_STRUCT.size
        original_length = pacbin._INDEX_ENTRY_STRUCT.unpack_from(data, entry)[0]
        replacement = b"../escape//member.symjit"
        assert len(replacement) == original_length
        data[path_offset : path_offset + original_length] = replacement

    with pytest.raises(PacbinError, match="must not contain"):
        PacbinReader.open(_mutated_container(traversal, rewrite_index_digest=True))

    case_container = io.BytesIO()
    write_pacbin(
        case_container,
        (
            PacbinMemberSource(
                "case/A.bin", PacbinMemberKind.SYMJIT_APPLICATION, io.BytesIO()
            ),
            PacbinMemberSource(
                "case/B.bin", PacbinMemberKind.NATIVE_LIBRARY, io.BytesIO()
            ),
        ),
    )
    case_data = bytearray(case_container.getvalue())

    def collision(data: bytearray) -> None:
        first, second = _index_entry_offsets(data)
        first_length = pacbin._INDEX_ENTRY_STRUCT.unpack_from(data, first)[0]
        second_length = pacbin._INDEX_ENTRY_STRUCT.unpack_from(data, second)[0]
        assert first_length == second_length
        first_path = bytes(
            data[
                first + pacbin._INDEX_ENTRY_STRUCT.size : first
                + pacbin._INDEX_ENTRY_STRUCT.size
                + first_length
            ]
        )
        replacement = first_path.replace(b"A", b"a", 1)
        data[
            second + pacbin._INDEX_ENTRY_STRUCT.size : second
            + pacbin._INDEX_ENTRY_STRUCT.size
            + second_length
        ] = replacement

    collision(case_data)
    _rewrite_index_digest(case_data)
    with pytest.raises(PacbinError, match="case-colliding"):
        PacbinReader.open(io.BytesIO(case_data))


def test_reader_rejects_duplicate_and_unsorted_keys() -> None:
    container = io.BytesIO()
    write_pacbin(
        container,
        (
            PacbinMemberSource(
                "case/A.bin", PacbinMemberKind.SYMJIT_APPLICATION, io.BytesIO()
            ),
            PacbinMemberSource(
                "case/B.bin", PacbinMemberKind.NATIVE_LIBRARY, io.BytesIO()
            ),
        ),
    )

    duplicate = bytearray(container.getvalue())
    first, second = _index_entry_offsets(duplicate)
    path_length = pacbin._INDEX_ENTRY_STRUCT.unpack_from(duplicate, first)[0]
    first_path_start = first + pacbin._INDEX_ENTRY_STRUCT.size
    second_path_start = second + pacbin._INDEX_ENTRY_STRUCT.size
    duplicate[second_path_start : second_path_start + path_length] = duplicate[
        first_path_start : first_path_start + path_length
    ]
    _rewrite_index_digest(duplicate)
    with pytest.raises(PacbinError, match="duplicate"):
        PacbinReader.open(io.BytesIO(duplicate))

    unsorted = bytearray(container.getvalue())
    first, second = _index_entry_offsets(unsorted)
    first_path_start = first + pacbin._INDEX_ENTRY_STRUCT.size
    second_path_start = second + pacbin._INDEX_ENTRY_STRUCT.size
    first_path = bytes(unsorted[first_path_start : first_path_start + path_length])
    second_path = bytes(unsorted[second_path_start : second_path_start + path_length])
    unsorted[first_path_start : first_path_start + path_length] = second_path
    unsorted[second_path_start : second_path_start + path_length] = first_path
    _rewrite_index_digest(unsorted)
    with pytest.raises(PacbinError, match="strictly sorted"):
        PacbinReader.open(io.BytesIO(unsorted))


def test_reader_rejects_overlap_out_of_bounds_and_nonzero_padding() -> None:
    def overlap(data: bytearray) -> None:
        first, second, _third = _index_entry_offsets(data)
        first_offset = pacbin._INDEX_ENTRY_STRUCT.unpack_from(data, first)[3]
        struct.pack_into("<Q", data, second + 8, first_offset)

    with pytest.raises(PacbinError, match="overlapping"):
        PacbinReader.open(_mutated_container(overlap, rewrite_index_digest=True))

    def out_of_bounds(data: bytearray) -> None:
        first = _index_entry_offsets(data)[0]
        index_offset = pacbin._HEADER_STRUCT.unpack_from(data, 0)[6]
        struct.pack_into("<Q", data, first + 16, index_offset)

    with pytest.raises(PacbinError, match="out of bounds"):
        PacbinReader.open(_mutated_container(out_of_bounds, rewrite_index_digest=True))

    def nonzero_payload_padding(data: bytearray) -> None:
        first, second, _third = _index_entry_offsets(data)
        first_record = pacbin._INDEX_ENTRY_STRUCT.unpack_from(data, first)
        second_record = pacbin._INDEX_ENTRY_STRUCT.unpack_from(data, second)
        padding_start = first_record[3] + first_record[4]
        assert padding_start < second_record[3]
        data[padding_start] = 1

    with pytest.raises(PacbinError, match="payload padding"):
        PacbinReader.open(_mutated_container(nonzero_payload_padding))


def test_reader_rejects_u64_member_range_overflow() -> None:
    def overflow(data: bytearray) -> None:
        first = _index_entry_offsets(data)[0]
        struct.pack_into("<Q", data, first + 16, pacbin._MAX_U64)

    with pytest.raises(PacbinError, match="overflows u64"):
        PacbinReader.open(_mutated_container(overflow, rewrite_index_digest=True))


def test_reader_rejects_digest_corruption_and_truncation() -> None:
    def corrupt_payload(data: bytearray) -> None:
        first = _index_entry_offsets(data)[0]
        payload_offset = pacbin._INDEX_ENTRY_STRUCT.unpack_from(data, first)[3]
        data[payload_offset] ^= 1

    with pytest.raises(PacbinError, match="member digest mismatch"):
        PacbinReader.open(_mutated_container(corrupt_payload))

    def corrupt_index_digest(data: bytearray) -> None:
        data[-1] ^= 1

    with pytest.raises(PacbinError, match="index digest mismatch"):
        PacbinReader.open(_mutated_container(corrupt_index_digest))

    truncated = _container_bytes()[:-17]
    with pytest.raises(PacbinError, match=r"footer|truncated"):
        PacbinReader.open(io.BytesIO(truncated))
