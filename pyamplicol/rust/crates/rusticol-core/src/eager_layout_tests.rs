// SPDX-License-Identifier: 0BSD

use super::*;
use crate::RusticolErrorKind;

fn section(kind: EagerSectionKind, record_size: u32, record_count: u64) -> Vec<u8> {
    let header = EagerSectionHeader::new(kind, record_size, record_count).unwrap();
    let mut bytes = header.encode().to_vec();
    bytes.resize(
        EAGER_SECTION_HEADER_SIZE + header.payload_length() as usize,
        0x5a,
    );
    bytes
}

fn assert_error(bytes: &[u8], kind: RusticolErrorKind, message: &str) {
    let error = EagerSectionHeader::decode(bytes).unwrap_err();
    assert_eq!(error.kind(), kind);
    assert!(
        error.to_string().contains(message),
        "expected {:?} to contain {message:?}",
        error.to_string()
    );
}

#[test]
fn public_contract_constants_are_stable() {
    assert_eq!(
        EAGER_LOWERING_INPUT_ABI,
        "pyamplicol-eager-lowering-input-v1"
    );
    assert_eq!(EAGER_PLAN_ABI, "pyamplicol-eager-plan-v3");
    assert_eq!(
        EAGER_RUNTIME_LAYOUT_ABI,
        "pyamplicol-eager-runtime-layout-v1"
    );
    assert_eq!(
        EAGER_RUNTIME_CAPABILITY,
        "rusticol.eager-runtime-layout.complex-f64.v1"
    );
    assert_eq!(
        EAGER_RUNTIME_CONTAINER_KIND,
        "pyamplicol-eager-runtime-container"
    );
    assert_eq!(EAGER_RUNTIME_CONTAINER_SCHEMA, 1);
}

#[test]
fn section_header_is_deterministic_little_endian_and_round_trips() {
    let bytes = section(EagerSectionKind::Invocations, 40, 3);
    assert_eq!(&bytes[0..8], b"PACERT\0\0");
    assert_eq!(&bytes[8..10], &1_u16.to_le_bytes());
    assert_eq!(&bytes[10..12], &64_u16.to_le_bytes());
    assert_eq!(&bytes[12..14], &9_u16.to_le_bytes());
    assert_eq!(&bytes[16..20], &40_u32.to_le_bytes());
    assert_eq!(&bytes[24..32], &3_u64.to_le_bytes());
    assert_eq!(&bytes[32..40], &64_u64.to_le_bytes());
    assert_eq!(&bytes[40..48], &120_u64.to_le_bytes());
    let (header, payload) = EagerSectionHeader::decode(&bytes).unwrap();
    assert_eq!(header.kind(), EagerSectionKind::Invocations);
    assert_eq!(header.record_size(), 40);
    assert_eq!(header.record_count(), 3);
    assert_eq!(header.payload_length(), 120);
    assert_eq!(payload, vec![0x5a; 120]);
    assert_eq!(header.encode(), bytes[..EAGER_SECTION_HEADER_SIZE]);
}

#[test]
fn zero_record_sections_are_valid_but_zero_width_and_overflow_are_rejected() {
    let bytes = section(EagerSectionKind::Metadata, 16, 0);
    let (header, payload) = EagerSectionHeader::decode(&bytes).unwrap();
    assert_eq!(header.record_count(), 0);
    assert!(payload.is_empty());

    let zero = EagerSectionHeader::new(EagerSectionKind::Metadata, 0, 1).unwrap_err();
    assert_eq!(zero.kind(), RusticolErrorKind::InvalidArgument);
    assert!(zero.to_string().contains("record size"));
    let overflow =
        EagerSectionHeader::new(EagerSectionKind::Metadata, u32::MAX, u64::MAX).unwrap_err();
    assert_eq!(overflow.kind(), RusticolErrorKind::InvalidArgument);
    assert!(overflow.to_string().contains("exceeds u64"));
}

#[test]
fn section_header_rejects_contract_corruption_and_noncanonical_lengths() {
    let golden = section(EagerSectionKind::Attachments, 24, 2);
    for (offset, replacement, kind, message) in [
        (0, vec![b'X'], RusticolErrorKind::Integrity, "magic"),
        (
            8,
            2_u16.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "schema",
        ),
        (
            10,
            32_u16.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "header size",
        ),
        (
            12,
            99_u16.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "section kind",
        ),
        (
            14,
            1_u16.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "flags",
        ),
        (
            16,
            0_u32.to_le_bytes().to_vec(),
            RusticolErrorKind::Integrity,
            "record size",
        ),
        (
            20,
            1_u32.to_le_bytes().to_vec(),
            RusticolErrorKind::Integrity,
            "reserved",
        ),
        (
            32,
            63_u64.to_le_bytes().to_vec(),
            RusticolErrorKind::Integrity,
            "payload offset",
        ),
        (
            40,
            47_u64.to_le_bytes().to_vec(),
            RusticolErrorKind::Integrity,
            "payload length",
        ),
    ] {
        let mut bytes = golden.clone();
        bytes[offset..offset + replacement.len()].copy_from_slice(&replacement);
        assert_error(&bytes, kind, message);
    }
    let mut reserved = golden.clone();
    reserved[63] = 1;
    assert_error(&reserved, RusticolErrorKind::Integrity, "reserved fields");
    assert_error(
        &golden[..golden.len() - 1],
        RusticolErrorKind::Integrity,
        "truncated",
    );
    let mut trailing = golden;
    trailing.push(0);
    assert_error(&trailing, RusticolErrorKind::Integrity, "trailing bytes");
}

#[test]
fn section_references_round_trip_and_validate_bounds_and_kind() {
    let target = EagerSectionHeader::new(EagerSectionKind::Invocations, 40, 100).unwrap();
    let reference = EagerSectionReference::new(EagerSectionKind::Invocations, 10, 20).unwrap();
    let bytes = reference.encode();
    assert_eq!(&bytes[0..2], &9_u16.to_le_bytes());
    assert_eq!(&bytes[8..16], &10_u64.to_le_bytes());
    assert_eq!(&bytes[16..24], &20_u64.to_le_bytes());
    let decoded = EagerSectionReference::decode(&bytes).unwrap();
    assert_eq!(decoded, reference);
    decoded.validate_against(&target).unwrap();

    EagerSectionReference::new(EagerSectionKind::Invocations, 100, 0)
        .unwrap()
        .validate_against(&target)
        .unwrap();
    let out_of_bounds = EagerSectionReference::new(EagerSectionKind::Invocations, 99, 2)
        .unwrap()
        .validate_against(&target)
        .unwrap_err();
    assert_eq!(out_of_bounds.kind(), RusticolErrorKind::Integrity);
    assert!(out_of_bounds.to_string().contains("bounds"));
    let wrong_kind = EagerSectionReference::new(EagerSectionKind::Closures, 0, 1)
        .unwrap()
        .validate_against(&target)
        .unwrap_err();
    assert_eq!(wrong_kind.kind(), RusticolErrorKind::Integrity);
    assert!(wrong_kind.to_string().contains("does not match"));
    let overflow =
        EagerSectionReference::new(EagerSectionKind::Invocations, u64::MAX, 1).unwrap_err();
    assert_eq!(overflow.kind(), RusticolErrorKind::InvalidArgument);
}

#[test]
fn section_reference_rejects_corruption() {
    let golden = EagerSectionReference::new(EagerSectionKind::Closures, 2, 3)
        .unwrap()
        .encode();
    for (offset, replacement, kind, message) in [
        (
            0,
            99_u16.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "kind",
        ),
        (
            2,
            1_u16.to_le_bytes().to_vec(),
            RusticolErrorKind::Compatibility,
            "flags",
        ),
        (
            4,
            1_u32.to_le_bytes().to_vec(),
            RusticolErrorKind::Integrity,
            "reserved",
        ),
    ] {
        let mut bytes = golden;
        bytes[offset..offset + replacement.len()].copy_from_slice(&replacement);
        let error = EagerSectionReference::decode(&bytes).unwrap_err();
        assert_eq!(error.kind(), kind);
        assert!(error.to_string().contains(message));
    }
    assert!(
        EagerSectionReference::decode(&golden[..golden.len() - 1])
            .unwrap_err()
            .to_string()
            .contains("truncated")
    );
    let mut trailing = golden.to_vec();
    trailing.push(0);
    assert!(
        EagerSectionReference::decode(&trailing)
            .unwrap_err()
            .to_string()
            .contains("trailing")
    );
}
