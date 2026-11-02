// SPDX-License-Identifier: 0BSD

use super::*;
use crate::RusticolErrorKind;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Read};
use std::sync::atomic::{AtomicU64, Ordering};

const PYTHON_GOLDEN_HEX: &str = concat!(
    "50414342494e000001004000000000004000000000000000c000000000000000",
    "0200000000000000000000000000000000000000000000000000000000000000",
    "6a69742d76310000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "65786163742d7374617465000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "5041434944580000010020000000000002000000000000000000000000000000",
    "0c0000000100000040000000000000000600000000000000e9a628134c31e4a5",
    "7771ecd7a1dd51c5c5bc7daca10a2553c83f8af1652d2587612f6a69742e7379",
    "6d6a6974000000000b0000000200000080000000000000000b00000000000000",
    "a9f46d72eb6f1c7e5d62e97d3d8e67b80605aeba2df022a8a25b51ca3fc89805",
    "7a2f65786163742e62696e0000000000504143454e4400000100400000000000",
    "c0000000000000000200000000000000263b30eb85e5b2a82600201822438767",
    "c5f66d9eecf2e08a0202383aafe0e3c1",
);

#[derive(Clone)]
struct TestMember<'a> {
    path: &'a str,
    kind: PacbinMemberKind,
    payload: &'a [u8],
}

fn decode_hex(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0);
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).expect("ASCII hex pair");
            u8::from_str_radix(text, 16).expect("valid hex pair")
        })
        .collect()
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn align(bytes: &mut Vec<u8>, alignment: usize) {
    let padding = (alignment - bytes.len() % alignment) % alignment;
    bytes.resize(bytes.len() + padding, 0);
}

fn build_container(members: &[TestMember<'_>]) -> Vec<u8> {
    let mut bytes = vec![0; HEADER_SIZE];
    let mut records = Vec::new();
    for member in members {
        align(&mut bytes, PACBIN_ALIGNMENT as usize);
        let offset = bytes.len() as u64;
        bytes.extend_from_slice(member.payload);
        records.push((member, offset, Sha256::digest(member.payload)));
    }
    align(&mut bytes, PACBIN_ALIGNMENT as usize);
    let index_offset = bytes.len() as u64;
    bytes.extend_from_slice(INDEX_MAGIC);
    bytes.extend_from_slice(&PACBIN_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(INDEX_HEADER_SIZE as u16).to_le_bytes());
    bytes.extend_from_slice(&SUPPORTED_FLAGS.to_le_bytes());
    bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    for (member, offset, digest) in records {
        let path = member.path.as_bytes();
        bytes.extend_from_slice(&(path.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(member.kind as u16).to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes.extend_from_slice(&(member.payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&digest);
        bytes.extend_from_slice(path);
        align(&mut bytes, INDEX_ALIGNMENT as usize);
    }
    let index_digest = Sha256::digest(&bytes[index_offset as usize..]);
    bytes.extend_from_slice(FOOTER_MAGIC);
    bytes.extend_from_slice(&PACBIN_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(FOOTER_SIZE as u16).to_le_bytes());
    bytes.extend_from_slice(&SUPPORTED_FLAGS.to_le_bytes());
    bytes.extend_from_slice(&index_offset.to_le_bytes());
    bytes.extend_from_slice(&(members.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&index_digest);

    bytes[0..8].copy_from_slice(HEADER_MAGIC);
    put_u16(&mut bytes, 8, PACBIN_VERSION);
    put_u16(&mut bytes, 10, HEADER_SIZE as u16);
    put_u32(&mut bytes, 12, SUPPORTED_FLAGS);
    put_u32(&mut bytes, 16, PACBIN_ALIGNMENT);
    put_u32(&mut bytes, 20, 0);
    put_u64(&mut bytes, 24, index_offset);
    put_u64(&mut bytes, 32, members.len() as u64);
    bytes
}

fn index_offset(bytes: &[u8]) -> usize {
    u64_at(bytes, 24, "test index offset").expect("index offset") as usize
}

fn footer_offset(bytes: &[u8]) -> usize {
    bytes.len() - FOOTER_SIZE
}

fn entry_offsets(bytes: &[u8]) -> Vec<usize> {
    let index = index_offset(bytes);
    let footer = footer_offset(bytes);
    let count = u64_at(bytes, 32, "test member count").expect("member count");
    let mut cursor = index + INDEX_HEADER_SIZE;
    let mut offsets = Vec::new();
    for _ in 0..count {
        offsets.push(cursor);
        let path_length = u32_at(bytes, cursor, "test path length").expect("path length");
        let record = INDEX_ENTRY_SIZE as u64 + u64::from(path_length);
        cursor += (record + padding_length(record, INDEX_ALIGNMENT)) as usize;
    }
    assert!(cursor <= footer);
    offsets
}

fn rewrite_index_digest(bytes: &mut [u8]) {
    let index = index_offset(bytes);
    let footer = footer_offset(bytes);
    let digest = Sha256::digest(&bytes[index..footer]);
    bytes[footer + 32..footer + 64].copy_from_slice(&digest);
}

fn error(bytes: Vec<u8>) -> (RusticolErrorKind, String) {
    let error = PacbinReader::from_bytes(bytes).expect_err("container must be rejected");
    (error.kind(), error.to_string())
}

fn assert_error(bytes: Vec<u8>, kind: RusticolErrorKind, message: &str) {
    let (actual_kind, actual_message) = error(bytes);
    assert_eq!(actual_kind, kind);
    assert!(
        actual_message.contains(message),
        "expected {actual_message:?} to contain {message:?}"
    );
}

#[test]
fn python_golden_round_trips_and_supports_indexed_borrowed_ranges() {
    let bytes = decode_hex(PYTHON_GOLDEN_HEX);
    let reader = PacbinReader::from_bytes(bytes).expect("Python pacbin-v1 golden container");
    assert_eq!(reader.index().version(), 1);
    assert_eq!(reader.index().index_offset(), 192);
    assert_eq!(reader.index().file_size(), 432);
    assert_eq!(reader.container_size(), 432);
    assert_eq!(reader.members().len(), 2);
    assert_eq!(
        reader.member("a//./jit.symjit").unwrap().kind(),
        PacbinMemberKind::SymjitApplication
    );
    assert_eq!(reader.member_bytes("a/jit.symjit").unwrap(), b"jit-v1");
    assert_eq!(reader.member_range("z/exact.bin", 2, 5).unwrap(), b"act-s");
    reader.verify_payloads().unwrap();

    let missing = reader.member("missing.bin").unwrap_err();
    assert_eq!(missing.kind(), RusticolErrorKind::InvalidArgument);
    assert_eq!(missing.to_string(), "unknown pacbin member: missing.bin");
    let out_of_bounds = reader.member_range("a/jit.symjit", 5, 2).unwrap_err();
    assert_eq!(out_of_bounds.kind(), RusticolErrorKind::InvalidArgument);
    assert_eq!(
        out_of_bounds.to_string(),
        "member read exceeds indexed payload bounds"
    );
}

#[test]
fn open_keeps_one_immutable_snapshot_after_atomic_source_replacement() {
    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "rusticol-pacbin-{}-{}.pacbin",
        std::process::id(),
        NEXT_PATH.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&path, decode_hex(PYTHON_GOLDEN_HEX)).unwrap();
    let reader = PacbinReader::open(&path).unwrap();
    let replacement = path.with_extension("replacement");
    fs::write(&replacement, b"replaced after open").unwrap();
    fs::rename(&replacement, &path).unwrap();
    assert_eq!(reader.member_bytes("a/jit.symjit").unwrap(), b"jit-v1");
    fs::remove_file(path).unwrap();
}

#[test]
fn expected_container_digest_authenticates_an_immutable_snapshot() {
    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "rusticol-pacbin-authenticated-{}-{}.pacbin",
        std::process::id(),
        NEXT_PATH.fetch_add(1, Ordering::Relaxed)
    ));
    let bytes = decode_hex(PYTHON_GOLDEN_HEX);
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    fs::write(&path, &bytes).unwrap();
    let reader = PacbinReader::open_with_sha256(&path, &digest).unwrap();

    fs::write(&path, vec![0_u8; bytes.len()]).unwrap();
    assert_eq!(reader.member_bytes("a/jit.symjit").unwrap(), b"jit-v1");
    reader.verify_payloads().unwrap();
    OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(0)
        .unwrap();
    assert_eq!(reader.member_bytes("a/jit.symjit").unwrap(), b"jit-v1");
    reader.verify_payloads().unwrap();

    let replacement = path.with_extension("replacement");
    fs::write(&replacement, b"replaced after authenticated open").unwrap();
    fs::rename(&replacement, &path).unwrap();
    assert_eq!(reader.member_bytes("a/jit.symjit").unwrap(), b"jit-v1");

    let error = PacbinReader::open_with_sha256(&path, &digest).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("payload digest mismatch"));
    fs::remove_file(path).unwrap();
}

#[test]
fn empty_and_zero_length_members_are_valid() {
    let empty = PacbinReader::from_bytes(build_container(&[])).unwrap();
    assert!(empty.members().is_empty());

    let zero = build_container(&[
        TestMember {
            path: "a.empty",
            kind: PacbinMemberKind::SymjitApplication,
            payload: b"",
        },
        TestMember {
            path: "b.empty",
            kind: PacbinMemberKind::SymbolicaExactState,
            payload: b"",
        },
    ]);
    let zero = PacbinReader::from_bytes(zero).unwrap();
    assert_eq!(zero.members()[0].offset(), zero.members()[1].offset());
    assert_eq!(zero.member_bytes("b.empty").unwrap(), b"");
}

#[test]
fn header_footer_and_index_contracts_are_strict() {
    let golden = decode_hex(PYTHON_GOLDEN_HEX);
    for (offset, replacement, kind, message) in [
        (0, vec![b'X'], RusticolErrorKind::Integrity, "header magic"),
        (
            8,
            2_u16.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "header version",
        ),
        (
            10,
            1_u16.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "header size",
        ),
        (
            12,
            1_u32.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "header flags",
        ),
        (
            16,
            32_u32.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "payload alignment",
        ),
        (
            20,
            1_u32.to_le_bytes().to_vec(),
            RusticolErrorKind::Integrity,
            "reserved fields",
        ),
    ] {
        let mut bytes = golden.clone();
        bytes[offset..offset + replacement.len()].copy_from_slice(&replacement);
        assert_error(bytes, kind, message);
    }
    let mut reserved = golden.clone();
    reserved[63] = 1;
    assert_error(reserved, RusticolErrorKind::Integrity, "reserved fields");
    let mut unaligned_index = golden.clone();
    put_u64(&mut unaligned_index, 24, 193);
    assert_error(
        unaligned_index,
        RusticolErrorKind::Integrity,
        "index offset is not payload-aligned",
    );
    let mut out_of_bounds_index = golden.clone();
    put_u64(&mut out_of_bounds_index, 24, 0);
    let footer = footer_offset(&out_of_bounds_index);
    put_u64(&mut out_of_bounds_index, footer + 16, 0);
    assert_error(
        out_of_bounds_index,
        RusticolErrorKind::Integrity,
        "index offset is out of bounds",
    );

    let footer = footer_offset(&golden);
    for (offset, replacement, kind, message) in [
        (
            footer,
            vec![b'X'],
            RusticolErrorKind::Integrity,
            "footer magic",
        ),
        (
            footer + 8,
            2_u16.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "footer version",
        ),
        (
            footer + 10,
            1_u16.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "footer size",
        ),
        (
            footer + 12,
            1_u32.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "footer flags",
        ),
        (
            footer + 16,
            0_u64.to_le_bytes().to_vec(),
            RusticolErrorKind::Integrity,
            "offset disagrees",
        ),
        (
            footer + 24,
            99_u64.to_le_bytes().to_vec(),
            RusticolErrorKind::Integrity,
            "count disagrees",
        ),
    ] {
        let mut bytes = golden.clone();
        bytes[offset..offset + replacement.len()].copy_from_slice(&replacement);
        assert_error(bytes, kind, message);
    }
    let index = index_offset(&golden);
    let mut index_count = golden.clone();
    put_u64(&mut index_count, index + 16, 1);
    rewrite_index_digest(&mut index_count);
    assert_error(
        index_count,
        RusticolErrorKind::Integrity,
        "index member count disagrees with header",
    );

    for (offset, replacement, kind, message) in [
        (
            index,
            vec![b'X'],
            RusticolErrorKind::Integrity,
            "index magic",
        ),
        (
            index + 8,
            2_u16.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "index version",
        ),
        (
            index + 10,
            1_u16.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "index size",
        ),
        (
            index + 12,
            1_u32.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "index flags",
        ),
        (
            index + 24,
            1_u64.to_le_bytes().to_vec(),
            RusticolErrorKind::Integrity,
            "reserved field",
        ),
    ] {
        let mut bytes = golden.clone();
        bytes[offset..offset + replacement.len()].copy_from_slice(&replacement);
        rewrite_index_digest(&mut bytes);
        assert_error(bytes, kind, message);
    }
}

#[test]
fn count_and_index_size_bounds_fail_before_allocation() {
    assert_error(
        {
            let mut bytes = decode_hex(PYTHON_GOLDEN_HEX);
            put_u64(&mut bytes, 32, u64::MAX);
            let footer = footer_offset(&bytes);
            put_u64(&mut bytes, footer + 24, u64::MAX);
            bytes
        },
        RusticolErrorKind::Integrity,
        "member count cannot fit in index",
    );
    validate_index_bounds(0, 256 * 1024 * 1024 + 1).unwrap();
    let impossible = validate_index_bounds(2, 64).unwrap_err();
    assert_eq!(
        impossible.to_string(),
        "pacbin member count cannot fit in index"
    );
}

#[test]
fn member_contract_path_bounds_and_utf8_are_strict() {
    let golden = decode_hex(PYTHON_GOLDEN_HEX);
    let entry = entry_offsets(&golden)[0];
    let mut kind = golden.clone();
    put_u16(&mut kind, entry + 4, 99);
    rewrite_index_digest(&mut kind);
    assert_error(kind, RusticolErrorKind::Compatibility, "member kind");

    let mut flags = golden.clone();
    put_u16(&mut flags, entry + 6, 1);
    rewrite_index_digest(&mut flags);
    assert_error(flags, RusticolErrorKind::Compatibility, "member flags");

    let mut path_bound = golden.clone();
    put_u32(&mut path_bound, entry, u32::MAX);
    rewrite_index_digest(&mut path_bound);
    assert_error(
        path_bound,
        RusticolErrorKind::Integrity,
        "pacbin member path",
    );

    let mut utf8 = golden.clone();
    utf8[entry + INDEX_ENTRY_SIZE] = 0xff;
    rewrite_index_digest(&mut utf8);
    assert_error(utf8, RusticolErrorKind::Integrity, "not valid UTF-8");

    let unicode = build_container(&[TestMember {
        path: "café/state.bin",
        kind: PacbinMemberKind::SymbolicaExactState,
        payload: b"state",
    }]);
    assert_error(unicode, RusticolErrorKind::Integrity, "portable ASCII");
}

#[test]
fn paths_must_be_canonical_unique_case_distinct_and_sorted() {
    for (members, message) in [
        (
            vec![
                TestMember {
                    path: "same.bin",
                    kind: PacbinMemberKind::SymjitApplication,
                    payload: b"a",
                },
                TestMember {
                    path: "same.bin",
                    kind: PacbinMemberKind::NativeLibrary,
                    payload: b"b",
                },
            ],
            "duplicate",
        ),
        (
            vec![
                TestMember {
                    path: "Case.bin",
                    kind: PacbinMemberKind::SymjitApplication,
                    payload: b"a",
                },
                TestMember {
                    path: "case.bin",
                    kind: PacbinMemberKind::NativeLibrary,
                    payload: b"b",
                },
            ],
            "case-colliding",
        ),
        (
            vec![
                TestMember {
                    path: "z.bin",
                    kind: PacbinMemberKind::SymjitApplication,
                    payload: b"a",
                },
                TestMember {
                    path: "a.bin",
                    kind: PacbinMemberKind::NativeLibrary,
                    payload: b"b",
                },
            ],
            "strictly sorted",
        ),
    ] {
        assert_error(
            build_container(&members),
            RusticolErrorKind::Integrity,
            message,
        );
    }

    let noncanonical = build_container(&[TestMember {
        path: "../escape.bin",
        kind: PacbinMemberKind::SymjitApplication,
        payload: b"payload",
    }]);
    assert_error(
        noncanonical,
        RusticolErrorKind::Integrity,
        "must not contain '..'",
    );
}

#[test]
fn index_padding_and_exact_coverage_are_enforced() {
    let mut padding = decode_hex(PYTHON_GOLDEN_HEX);
    let first = entry_offsets(&padding)[0];
    let path_length = u32_at(&padding, first, "path length").unwrap() as usize;
    padding[first + INDEX_ENTRY_SIZE + path_length] = 1;
    rewrite_index_digest(&mut padding);
    assert_error(padding, RusticolErrorKind::Integrity, "index padding");

    let mut trailing = decode_hex(PYTHON_GOLDEN_HEX);
    let footer = footer_offset(&trailing);
    trailing.splice(footer..footer, [0_u8; 8]);
    rewrite_index_digest(&mut trailing);
    assert_error(
        trailing,
        RusticolErrorKind::Integrity,
        "trailing or missing bytes",
    );
}

#[test]
fn payload_layout_rejects_overlap_gap_bounds_overflow_and_nonzero_padding() {
    let golden = decode_hex(PYTHON_GOLDEN_HEX);
    let entries = entry_offsets(&golden);
    let first_offset = u64_at(&golden, entries[0] + 8, "first offset").unwrap();

    let mut overlap = golden.clone();
    put_u64(&mut overlap, entries[1] + 8, first_offset);
    rewrite_index_digest(&mut overlap);
    assert_error(overlap, RusticolErrorKind::Integrity, "overlapping");

    let mut gap = golden.clone();
    put_u64(&mut gap, entries[1] + 8, 192);
    rewrite_index_digest(&mut gap);
    assert_error(
        gap,
        RusticolErrorKind::Integrity,
        "non-canonical pacbin member gap",
    );

    let mut out_of_bounds = golden.clone();
    put_u64(&mut out_of_bounds, entries[0] + 16, 192);
    rewrite_index_digest(&mut out_of_bounds);
    assert_error(out_of_bounds, RusticolErrorKind::Integrity, "out of bounds");

    let mut overflow = golden.clone();
    put_u64(&mut overflow, entries[0] + 16, u64::MAX);
    rewrite_index_digest(&mut overflow);
    assert_error(overflow, RusticolErrorKind::Integrity, "overflows u64");

    let mut padding = golden.clone();
    padding[70] = 1;
    assert_error(padding, RusticolErrorKind::Integrity, "payload padding");

    let mut trailing_gap = golden.clone();
    let old_index = index_offset(&trailing_gap);
    trailing_gap.splice(old_index..old_index, [0_u8; PACBIN_ALIGNMENT as usize]);
    let new_index = old_index as u64 + u64::from(PACBIN_ALIGNMENT);
    put_u64(&mut trailing_gap, 24, new_index);
    let footer = footer_offset(&trailing_gap);
    put_u64(&mut trailing_gap, footer + 16, new_index);
    rewrite_index_digest(&mut trailing_gap);
    assert_error(
        trailing_gap,
        RusticolErrorKind::Integrity,
        "non-canonical trailing gap",
    );
}

#[test]
fn payload_and_index_authentication_reject_corruption_and_truncation() {
    let mut payload = decode_hex(PYTHON_GOLDEN_HEX);
    payload[64] ^= 1;
    assert_error(
        payload,
        RusticolErrorKind::Integrity,
        "member digest mismatch",
    );

    let mut index = decode_hex(PYTHON_GOLDEN_HEX);
    let footer = footer_offset(&index);
    index[footer + 63] ^= 1;
    assert_error(index, RusticolErrorKind::Integrity, "index digest mismatch");

    let mut truncated = decode_hex(PYTHON_GOLDEN_HEX);
    truncated.truncate(truncated.len() - 17);
    let (_, message) = error(truncated);
    assert!(message.contains("footer") || message.contains("truncated"));
}

#[test]
fn normalization_follows_portable_python_posix_rules() {
    assert_eq!(normalize_logical_path("a//./b.bin").unwrap(), "a/b.bin");
    for (path, message) in [
        ("", "non-empty"),
        (".", "name a member"),
        ("/absolute", "relative"),
        ("safe/../escape", "must not contain"),
        ("windows\\path", "POSIX"),
        ("bad\0path", "NUL"),
    ] {
        let error = normalize_logical_path(path).unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
        assert!(error.to_string().contains(message));
    }
}

fn temporary_directory(label: &str) -> std::path::PathBuf {
    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "rusticol-pacbin-{label}-{}-{}",
        std::process::id(),
        NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    path
}

#[test]
fn rust_writer_matches_python_golden_byte_for_byte() {
    let directory = temporary_directory("python-golden");
    let destination = directory.join("golden.pacbin");
    let members = vec![
        PacbinWriteMember::from_bytes(
            "z/exact.bin",
            PacbinMemberKind::SymbolicaExactState,
            b"exact-state",
        )
        .unwrap(),
        PacbinWriteMember::from_bytes(
            "a//./jit.symjit",
            PacbinMemberKind::SymjitApplication,
            b"jit-v1",
        )
        .unwrap(),
    ];
    let index = write_pacbin_atomic(&destination, members, PacbinWriteOptions::default()).unwrap();
    let bytes = fs::read(&destination).unwrap();
    assert_eq!(bytes, decode_hex(PYTHON_GOLDEN_HEX));
    assert_eq!(index.index_offset(), 192);
    assert_eq!(index.file_size(), 432);
    assert_eq!(index.members().len(), 2);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn writer_is_deterministic_and_supports_runtime_member_kinds() {
    let directory = temporary_directory("determinism");
    let first = directory.join("first.pacbin");
    let second = directory.join("second.pacbin");
    for destination in [&first, &second] {
        write_pacbin_atomic(
            destination,
            vec![
                PacbinWriteMember::from_bytes(
                    "tables/invocations.bin",
                    PacbinMemberKind::EagerRuntimeTable,
                    b"table-payload",
                )
                .unwrap(),
                PacbinWriteMember::from_bytes(
                    "metadata/runtime.bin",
                    PacbinMemberKind::EagerRuntimeMetadata,
                    b"metadata-payload",
                )
                .unwrap(),
                PacbinWriteMember::from_bytes(
                    "schedule/recurrence-direct-schedule-v2.bin",
                    PacbinMemberKind::RecurrenceDirectPlan,
                    b"recurrence-direct-plan-payload",
                )
                .unwrap(),
                PacbinWriteMember::from_bytes(
                    "compiled/color.pacrclr3",
                    PacbinMemberKind::ColorContraction,
                    b"compiled-color-contraction-payload",
                )
                .unwrap(),
            ],
            PacbinWriteOptions::new(3, 0o644).unwrap(),
        )
        .unwrap();
    }
    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
    let reader = PacbinReader::open(&first).unwrap();
    assert_eq!(
        reader.member("metadata/runtime.bin").unwrap().kind(),
        PacbinMemberKind::EagerRuntimeMetadata
    );
    assert_eq!(
        reader.member("tables/invocations.bin").unwrap().kind(),
        PacbinMemberKind::EagerRuntimeTable
    );
    assert_eq!(
        reader
            .member("schedule/recurrence-direct-schedule-v2.bin")
            .unwrap()
            .kind(),
        PacbinMemberKind::RecurrenceDirectPlan
    );
    assert_eq!(
        reader.member("compiled/color.pacrclr3").unwrap().kind(),
        PacbinMemberKind::ColorContraction
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn deleted_recurrence_v1_member_kind_is_not_a_compatibility_alias() {
    assert!(PacbinMemberKind::parse(6).is_err());
    assert_eq!(
        PacbinMemberKind::parse(7).unwrap(),
        PacbinMemberKind::RecurrenceDirectPlan
    );
    assert_eq!(
        PacbinMemberKind::parse(8).unwrap(),
        PacbinMemberKind::RecurrenceColorProjectionCertificate
    );
    assert_eq!(
        PacbinMemberKind::parse(10).unwrap(),
        PacbinMemberKind::ColorContraction
    );
}

struct TrackingReader {
    payload: Vec<u8>,
    position: usize,
    maximum_request: usize,
    fail_at: Option<usize>,
}

impl TrackingReader {
    fn new(payload: impl Into<Vec<u8>>) -> Self {
        Self {
            payload: payload.into(),
            position: 0,
            maximum_request: 0,
            fail_at: None,
        }
    }

    fn failing(payload: impl Into<Vec<u8>>, fail_at: usize) -> Self {
        Self {
            fail_at: Some(fail_at),
            ..Self::new(payload)
        }
    }
}

impl Read for TrackingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.maximum_request = self.maximum_request.max(buffer.len());
        if self.fail_at.is_some_and(|limit| self.position >= limit) {
            return Err(io::Error::other("injected member read failure"));
        }
        let remaining = self.payload.len().saturating_sub(self.position);
        let mut length = remaining.min(buffer.len());
        if let Some(limit) = self.fail_at {
            length = length.min(limit.saturating_sub(self.position));
        }
        if length == 0 {
            return Ok(0);
        }
        buffer[..length].copy_from_slice(&self.payload[self.position..self.position + length]);
        self.position += length;
        Ok(length)
    }
}

#[test]
fn writer_streams_reader_and_path_sources_in_bounded_chunks() {
    let directory = temporary_directory("bounded-streaming");
    let source_path = directory.join("source.bin");
    fs::write(&source_path, b"path-source").unwrap();
    let destination = directory.join("container.pacbin");
    let mut source = TrackingReader::new(vec![0x5a; 100]);
    write_pacbin_atomic(
        &destination,
        vec![
            PacbinWriteMember::from_reader(
                "a/reader.bin",
                PacbinMemberKind::EagerRuntimeTable,
                &mut source,
            )
            .unwrap(),
            PacbinWriteMember::from_path(
                "b/path.bin",
                PacbinMemberKind::EagerRuntimeMetadata,
                &source_path,
            )
            .unwrap(),
        ],
        PacbinWriteOptions::new(7, 0o600).unwrap(),
    )
    .unwrap();
    assert_eq!(source.maximum_request, 7);
    let reader = PacbinReader::open(&destination).unwrap();
    assert_eq!(reader.member_bytes("a/reader.bin").unwrap(), &[0x5a; 100]);
    assert_eq!(reader.member_bytes("b/path.bin").unwrap(), b"path-source");
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn atomic_writer_preserves_destination_and_cleans_temporary_file_on_error() {
    let directory = temporary_directory("atomic-failure");
    let destination = directory.join("container.pacbin");
    fs::write(&destination, b"previous-container").unwrap();
    let mut source = TrackingReader::failing(vec![0x41; 32], 8);
    let error = write_pacbin_atomic(
        &destination,
        vec![
            PacbinWriteMember::from_reader(
                "tables/failing.bin",
                PacbinMemberKind::EagerRuntimeTable,
                &mut source,
            )
            .unwrap(),
        ],
        PacbinWriteOptions::new(4, 0o644).unwrap(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("injected member read failure"));
    assert_eq!(fs::read(&destination).unwrap(), b"previous-container");
    let names: Vec<_> = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(names, vec![std::ffi::OsString::from("container.pacbin")]);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn writer_preflight_rejects_invalid_options_and_colliding_paths() {
    assert!(PacbinWriteOptions::new(0, 0o644).is_err());
    assert!(PacbinWriteOptions::new(1, 0o1000).is_err());
    let directory = temporary_directory("preflight");
    let destination = directory.join("container.pacbin");
    let duplicate = write_pacbin_atomic(
        &destination,
        vec![
            PacbinWriteMember::from_bytes("same.bin", PacbinMemberKind::EagerRuntimeMetadata, b"a")
                .unwrap(),
            PacbinWriteMember::from_bytes("same.bin", PacbinMemberKind::EagerRuntimeTable, b"b")
                .unwrap(),
        ],
        PacbinWriteOptions::default(),
    )
    .unwrap_err();
    assert_eq!(duplicate.kind(), RusticolErrorKind::InvalidArgument);
    assert!(duplicate.to_string().contains("duplicate"));
    let case_collision = write_pacbin_atomic(
        &destination,
        vec![
            PacbinWriteMember::from_bytes("Case.bin", PacbinMemberKind::EagerRuntimeMetadata, b"a")
                .unwrap(),
            PacbinWriteMember::from_bytes("case.bin", PacbinMemberKind::EagerRuntimeTable, b"b")
                .unwrap(),
        ],
        PacbinWriteOptions::default(),
    )
    .unwrap_err();
    assert_eq!(case_collision.kind(), RusticolErrorKind::InvalidArgument);
    assert!(case_collision.to_string().contains("case-colliding"));
    assert!(!destination.exists());
    fs::remove_dir_all(directory).unwrap();
}
